"""
constraint_optimizer/code/train/module.py
-----------------------------------------
约束优化模块。

forward(selected_traj, ctx) -> refined_traj

输入：
    selected_traj: [N, T, 6]    GNN1 选出的 top-K 候选（fusion 已展平到 N=B*K，
                                坐标已是物理绝对坐标 km，前 3 维 pos / 后 3 维 vel）
    ctx.road_points: [N, NB_max, NP_max, 3]  km-xyz，多分支折线点（按 fusion._expand_ctx 展平到 N）
    ctx.road_mask:   [N, NB_max, NP_max]     bool，True = 有效点

输出：
    refined_traj:   [N, T, 6]   位置投影到最近的"有效路网线段"上，速度通道保留 LSTM1 原值

支持的 module_type：
  - "pass_through"           原样返回
  - "road_projection"        逐点硬投影到最近线段（旧实现，会"折一下穿过去"，
                             仅作 ablation 对照保留）
  - "road_arc_projection"    沿弧长投影（推荐）：每条候选挑一条最匹配的分支，
                             把候选轨迹的累积弧长映射到该分支上，输出严格
                             沿着路网折线走 → 不会"穿过路网空白处"。

road_arc_projection 算法（每条候选独立处理）：
  1) 对每个有效 branch（含双向）：
       a. 算分支总弧长 total_s 和段累积长度 cum_s
       b. 把 traj[0] 投到该分支，得"入路弧长" entry_s
       c. 算 traj 自身的累积弧长 traj_cum_s（从起点开始）
       d. 每帧目标弧长 = entry_s + traj_cum_s[t]，超过 total_s 就夹到路尾
       e. 在分支折线上按目标弧长插值得到投影点
       f. 总代价 = sum( |traj[t] - proj[t]| )
  2) 选总代价最小的 (branch, direction) 组合，返回对应投影
  3) 没任何有效分支的样本 → 保持原 pos

  - 速度通道：保持 LSTM1 原值（vel 不重算）
  - 多分支：road_points [NB, NP, 3]；同分支相邻两点连段，跨分支不连
  - 鲁棒性：road_mask 全 False 时自动 fallback 等价 pass_through
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.context_schema import ContextBatch  # noqa: E402


# ==============================================================================
# 工具：逐点投影到一组线段
# ==============================================================================

def _project_points_to_segments(
    pos: torch.Tensor,        # [N, T, 3]
    a: torch.Tensor,          # [N, NS, 3]   线段起点
    b: torch.Tensor,          # [N, NS, 3]   线段终点
    seg_mask: torch.Tensor,   # [N, NS]      bool，True=线段有效
) -> torch.Tensor:            # [N, T, 3]
    """
    经典"点到折线"投影：
      对每条线段 (a, b)，把 p 投影到该线段上得到 q：
        t = clamp(dot(p-a, b-a) / dot(b-a, b-a), 0, 1)
        q = a + t * (b - a)
      在所有"有效"线段里取距离最近的 q 作为输出。

    若某个样本一条有效线段都没有，则该样本所有 T 个点保持原值（fallback）。
    """
    N, T, _ = pos.shape
    NS = a.shape[1]

    p = pos.unsqueeze(2)                                 # [N, T,  1, 3]
    a_ = a.unsqueeze(1)                                  # [N, 1, NS, 3]
    b_ = b.unsqueeze(1)                                  # [N, 1, NS, 3]
    d_ = b_ - a_                                         # [N, 1, NS, 3]

    dd = (d_ * d_).sum(dim=-1).clamp_min(1e-12)          # [N, 1, NS]
    t_param = ((p - a_) * d_).sum(dim=-1) / dd           # [N, T, NS]
    t_param = t_param.clamp(0.0, 1.0)
    q = a_ + t_param.unsqueeze(-1) * d_                  # [N, T, NS, 3]
    diff = p - q                                          # [N, T, NS, 3]
    dist2 = (diff * diff).sum(dim=-1)                    # [N, T, NS]

    # 屏蔽无效线段：用一个超大值替代，让 argmin 不会选它
    seg_mask_e = seg_mask.unsqueeze(1).expand(N, T, NS)  # [N, T, NS]
    big = torch.tensor(float("inf"), dtype=dist2.dtype, device=dist2.device)
    dist2 = torch.where(seg_mask_e, dist2, big.expand_as(dist2))

    best_idx = dist2.argmin(dim=-1, keepdim=True)        # [N, T, 1]
    best_idx_e = best_idx.unsqueeze(-1).expand(N, T, 1, 3)
    q_best = torch.gather(q, 2, best_idx_e).squeeze(2)   # [N, T, 3]

    # 没有任何有效线段的样本 → fallback 为原 pos
    has_valid = seg_mask.any(dim=1)                      # [N]
    has_valid_e = has_valid.view(N, 1, 1).expand_as(pos)  # [N, T, 3]
    return torch.where(has_valid_e, q_best, pos)


# ==============================================================================
# ConstraintOptimizer
# ==============================================================================

class ConstraintOptimizer(nn.Module):
    """
    forward(selected_traj, ctx) -> refined_traj
    """

    def __init__(
        self,
        enable: bool = True,
        module_type: str = "road_arc_projection",
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        self.module_type = str(module_type)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, selected_traj: torch.Tensor, ctx: ContextBatch) -> torch.Tensor:
        if not self.enable or self.module_type == "pass_through":
            return selected_traj
        if self.module_type == "road_projection":
            return self._road_projection(selected_traj, ctx)
        if self.module_type == "road_arc_projection":
            return self._road_arc_projection(selected_traj, ctx)
        raise NotImplementedError(f"Unknown constraint_optimizer type: {self.module_type}")

    # ------------------------------------------------------------------
    # road_projection 实现
    # ------------------------------------------------------------------
    def _road_projection(
        self,
        selected_traj: torch.Tensor,    # [N, T, 6]
        ctx: ContextBatch,
    ) -> torch.Tensor:
        if selected_traj.ndim != 3 or selected_traj.shape[-1] < 3:
            raise ValueError(
                f"selected_traj 形状应为 [N, T, >=3]，实际 {tuple(selected_traj.shape)}"
            )

        pos = selected_traj[..., 0:3]
        rest = selected_traj[..., 3:]   # 速度通道（如果有），原样保留

        rp = ctx.road_points
        rm = ctx.road_mask

        # 兼容老的单分支 schema [N, NP, 3] / [N, NP]
        if rp.ndim == 3:
            rp = rp.unsqueeze(1)        # [N, 1, NP, 3]
            rm = rm.unsqueeze(1)        # [N, 1, NP]
        if rp.ndim != 4:
            raise ValueError(
                f"ctx.road_points 期望 [N, NB, NP, 3]，实际 {tuple(rp.shape)}"
            )

        N, NB, NP, D = rp.shape
        if N != selected_traj.shape[0]:
            raise ValueError(
                f"road_points batch={N} 与 selected_traj batch={selected_traj.shape[0]} 不一致"
            )

        # 路网点不足 2 个就没法构造线段，直接返回
        if NP < 2:
            return selected_traj

        # 构造线段：同分支相邻两点成段
        a = rp[:, :, :-1, :]                 # [N, NB, NP-1, 3]
        b = rp[:, :, 1:, :]                  # [N, NB, NP-1, 3]
        seg_mask = rm[:, :, :-1] & rm[:, :, 1:]   # [N, NB, NP-1]

        NS_total = NB * (NP - 1)
        a = a.reshape(N, NS_total, D).to(pos.dtype)
        b = b.reshape(N, NS_total, D).to(pos.dtype)
        seg_mask = seg_mask.reshape(N, NS_total).bool()

        # 整个 batch 都无 valid segment：等价 pass_through，省掉投影计算
        if not bool(seg_mask.any()):
            return selected_traj

        pos_proj = _project_points_to_segments(pos, a, b, seg_mask)
        if rest.shape[-1] == 0:
            return pos_proj
        return torch.cat([pos_proj, rest], dim=-1)

    # ------------------------------------------------------------------
    # road_arc_projection 实现
    # ------------------------------------------------------------------
    def _road_arc_projection(
        self,
        selected_traj: torch.Tensor,    # [N, T, 6]
        ctx: ContextBatch,
    ) -> torch.Tensor:
        if selected_traj.ndim != 3 or selected_traj.shape[-1] < 3:
            raise ValueError(
                f"selected_traj 形状应为 [N, T, >=3]，实际 {tuple(selected_traj.shape)}"
            )

        pos = selected_traj[..., 0:3]   # [N, T, 3]
        rest = selected_traj[..., 3:]

        rp = ctx.road_points
        rm = ctx.road_mask
        if rp.ndim == 3:
            rp = rp.unsqueeze(1)
            rm = rm.unsqueeze(1)
        if rp.ndim != 4:
            raise ValueError(
                f"ctx.road_points 期望 [N, NB, NP, 3]，实际 {tuple(rp.shape)}"
            )

        N, NB, NP, _ = rp.shape
        T = pos.shape[1]
        if NP < 2:
            return selected_traj

        # 整 batch 完全无路网 → fallback
        if not bool(rm.any()):
            return selected_traj

        out_pos = pos.clone()

        # 候选轨迹自身的累积弧长（每个样本独立）
        # traj_seg_len[n, t] = |pos[n, t+1] - pos[n, t]|
        traj_seg_len = torch.linalg.norm(pos[:, 1:] - pos[:, :-1], dim=-1)  # [N, T-1]
        traj_cum_s = torch.cat(
            [torch.zeros(N, 1, dtype=pos.dtype, device=pos.device), traj_seg_len.cumsum(dim=1)],
            dim=1,
        )  # [N, T]

        for n in range(N):
            best_cost = None
            best_proj = None
            traj_n = pos[n]              # [T, 3]
            traj_cs = traj_cum_s[n]      # [T]

            for bi in range(NB):
                mask_nb = rm[n, bi]      # [NP] bool
                n_valid = int(mask_nb.sum().item())
                if n_valid < 2:
                    continue
                pts_full = rp[n, bi][mask_nb]    # [Nv, 3]，已紧致

                # 双向都试：默认正向，再试反向
                for direction in (1, -1):
                    pts = pts_full if direction == 1 else pts_full.flip(0)
                    pts = pts.to(pos.dtype)

                    # 段累积弧长
                    seg_vec = pts[1:] - pts[:-1]                        # [Nv-1, 3]
                    seg_len = torch.linalg.norm(seg_vec, dim=-1).clamp_min(1e-9)  # [Nv-1]
                    cum_s = torch.cat(
                        [torch.zeros(1, dtype=pos.dtype, device=pos.device), seg_len.cumsum(dim=0)],
                        dim=0,
                    )                                                    # [Nv]
                    total_s = cum_s[-1]

                    # 入路弧长 entry_s：把 traj[0] 投到该分支取最近段及段内 t
                    p0 = traj_n[0]                                       # [3]
                    a0 = pts[:-1]                                        # [Nv-1, 3]
                    d0 = seg_vec                                         # [Nv-1, 3]
                    dd0 = (d0 * d0).sum(dim=-1).clamp_min(1e-12)         # [Nv-1]
                    t0 = ((p0 - a0) * d0).sum(dim=-1) / dd0              # [Nv-1]
                    t0 = t0.clamp(0.0, 1.0)
                    q0 = a0 + t0.unsqueeze(-1) * d0                      # [Nv-1, 3]
                    d2 = ((p0 - q0) ** 2).sum(dim=-1)                    # [Nv-1]
                    seg_idx0 = int(torch.argmin(d2).item())
                    entry_s = cum_s[seg_idx0] + t0[seg_idx0] * seg_len[seg_idx0]

                    # 每帧目标弧长
                    target_s = (entry_s + traj_cs).clamp(0.0, total_s)   # [T]

                    # 在 polyline 上按 target_s 插值
                    # 找每个 target_s_t 落在第 j 段：cum_s[j] <= target_s_t <= cum_s[j+1]
                    j_idx = torch.searchsorted(cum_s, target_s, right=True) - 1
                    j_idx = j_idx.clamp(0, pts.shape[0] - 2)             # [T]
                    seg_l = seg_len[j_idx]                               # [T]
                    t_local = ((target_s - cum_s[j_idx]) / seg_l.clamp_min(1e-9)).clamp(0.0, 1.0)
                    proj = pts[j_idx] + t_local.unsqueeze(-1) * (pts[j_idx + 1] - pts[j_idx])  # [T, 3]

                    cost = float(torch.linalg.norm(traj_n - proj, dim=-1).sum().item())
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_proj = proj

            if best_proj is not None:
                out_pos[n] = best_proj
            # else: 该样本无任何 valid branch，保持 out_pos[n] = traj_n（已 clone）

        if rest.shape[-1] == 0:
            return out_pos
        return torch.cat([out_pos, rest], dim=-1)


# ==============================================================================
# 工厂
# ==============================================================================

def build_module_from_config(cfg: Dict[str, Any]) -> ConstraintOptimizer:
    mod_cfg = cfg.get("module", {}) or {}
    return ConstraintOptimizer(
        enable=bool(mod_cfg.get("enable", True)),
        module_type=str(mod_cfg.get("type", "road_arc_projection")),
    )
