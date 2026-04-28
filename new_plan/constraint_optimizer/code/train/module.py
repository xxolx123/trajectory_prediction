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
    # road_arc_projection 实现（向量化 / trace 友好 / opset 11 兼容）
    # ------------------------------------------------------------------
    def _road_arc_projection(
        self,
        selected_traj: torch.Tensor,    # [N, T, 6]
        ctx: ContextBatch,
    ) -> torch.Tensor:
        """
        向量化版本（无 Python for 循环 / 无 .item() / 无布尔索引 / 无 searchsorted）。

        与循环版 `_road_arc_projection_loop` 数学等价，差别仅在浮点累加顺序，
        在"前 Nv 槽位连续 True、后续全 False"的标准 mask 形态下输出与循环版
        差异在 1e-5 量级以内（单测 `test_arc_projection_vectorized.py` 验证）。

        关键替换：
          - `torch.searchsorted` → broadcast 比较
                ``j_idx = (target_s.unsqueeze(-1) >= cum_s.unsqueeze(-2)).sum(-1) - 1``
            opset ≥ 11 即可（`GreaterOrEqual / Cast / ReduceSum / Sub / Clip`）。
          - `pts.flip(0)` 紧致反向 → 全长 `rp.flip(NP)` + mask 跟着 flip；
            前缀连续 True 形态下两者几何等价。
          - 两个方向直接 concat 到 NB 维变成 2*NB，统一处理。
          - 完全无 valid branch 的样本通过 `torch.where` fallback 到原 traj。
        """
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

        rp = rp.to(pos.dtype)

        # ---- 段级 mask（相邻两点都 valid）+ 反向（沿 NP 维 flip）----
        seg_mask_fw = rm[..., :-1] & rm[..., 1:]            # [N, NB, NP-1]
        rp_bw = rp.flip(-2)                                 # [N, NB, NP, 3]
        rm_bw = rm.flip(-1)                                 # [N, NB, NP]
        seg_mask_bw = rm_bw[..., :-1] & rm_bw[..., 1:]      # [N, NB, NP-1]

        # 段起点 / 终点 / 向量 / 长度（无效段长度置 0，cum_s 中不贡献距离）
        a_fw = rp[..., :-1, :]                              # [N, NB, NP-1, 3]
        b_fw = rp[..., 1:, :]
        sv_fw = b_fw - a_fw
        sl_raw_fw = torch.linalg.norm(sv_fw, dim=-1)        # [N, NB, NP-1]
        sl_fw = torch.where(seg_mask_fw, sl_raw_fw, torch.zeros_like(sl_raw_fw))

        a_bw = rp_bw[..., :-1, :]
        b_bw = rp_bw[..., 1:, :]
        sv_bw = b_bw - a_bw
        sl_raw_bw = torch.linalg.norm(sv_bw, dim=-1)
        sl_bw = torch.where(seg_mask_bw, sl_raw_bw, torch.zeros_like(sl_raw_bw))

        # 把正反两方向 concat 到 NB 维 → NB2 = 2*NB
        a_all = torch.cat([a_fw, a_bw], dim=1)              # [N, NB2, NP-1, 3]
        sv_all = torch.cat([sv_fw, sv_bw], dim=1)
        sm_all = torch.cat([seg_mask_fw, seg_mask_bw], dim=1)  # [N, NB2, NP-1]
        sl_all = torch.cat([sl_fw, sl_bw], dim=1)              # [N, NB2, NP-1]
        NB2 = sm_all.shape[1]

        # ---- 累积弧长 cum_s [N, NB2, NP] ----
        zero_col = torch.zeros(N, NB2, 1, dtype=pos.dtype, device=pos.device)
        cum_s = torch.cat([zero_col, sl_all.cumsum(dim=-1)], dim=-1)  # [N, NB2, NP]
        total_s = cum_s[..., -1]                                       # [N, NB2]

        # ---- 候选轨迹自身累积弧长 [N, T] ----
        traj_seg_len = torch.linalg.norm(
            pos[:, 1:] - pos[:, :-1], dim=-1
        )                                                              # [N, T-1]
        traj_cum_s = torch.cat(
            [
                torch.zeros(N, 1, dtype=pos.dtype, device=pos.device),
                traj_seg_len.cumsum(dim=-1),
            ],
            dim=-1,
        )                                                              # [N, T]

        # ---- entry_s：把 traj[0] 投到每条 (n, NB2) branch 取最近段及段内 t ----
        # p0 [N, 3] vs a_all [N, NB2, NP-1, 3]
        p0 = pos[:, 0, :]                                              # [N, 3]
        diff_p0 = p0[:, None, None, :] - a_all                         # [N, NB2, NP-1, 3]
        sv_dot = (sv_all * sv_all).sum(dim=-1).clamp_min(1e-12)        # [N, NB2, NP-1]
        t0 = (diff_p0 * sv_all).sum(dim=-1) / sv_dot                   # [N, NB2, NP-1]
        t0 = t0.clamp(0.0, 1.0)
        q0 = a_all + t0.unsqueeze(-1) * sv_all                         # [N, NB2, NP-1, 3]
        d2_p0 = ((p0[:, None, None, :] - q0) ** 2).sum(dim=-1)         # [N, NB2, NP-1]

        # 屏蔽无效段：用一个超大有限值代替 inf（部分 ONNX runtime 对 inf 不友好）
        BIG = 1.0e30
        big_t = torch.full_like(d2_p0, BIG)
        d2_p0_masked = torch.where(sm_all, d2_p0, big_t)

        seg_idx0 = d2_p0_masked.argmin(dim=-1, keepdim=True)           # [N, NB2, 1]
        cum_s_at_i = torch.gather(cum_s[..., :-1], -1, seg_idx0).squeeze(-1)  # [N, NB2]
        t0_at_i = torch.gather(t0, -1, seg_idx0).squeeze(-1)
        sl_at_i = torch.gather(sl_all, -1, seg_idx0).squeeze(-1)
        entry_s = cum_s_at_i + t0_at_i * sl_at_i                       # [N, NB2]

        # ---- 每帧目标弧长 [N, NB2, T]，clamp 到 [0, total_s] ----
        target_s = entry_s.unsqueeze(-1) + traj_cum_s.unsqueeze(1)     # [N, NB2, T]
        total_s_e = total_s.unsqueeze(-1)                              # [N, NB2, 1]
        target_s = torch.minimum(target_s, total_s_e).clamp_min(0.0)   # [N, NB2, T]

        # ---- 段索引：broadcast 比较代替 searchsorted（opset 11 兼容）----
        # cum_s [N, NB2, NP], target_s [N, NB2, T]
        # j = sum( target_s >= cum_s ) - 1
        cum_s_e = cum_s.unsqueeze(2)                                    # [N, NB2, 1, NP]
        tgt_e = target_s.unsqueeze(-1)                                  # [N, NB2, T, 1]
        ge = (tgt_e >= cum_s_e).to(torch.int64)                         # [N, NB2, T, NP]
        j_idx = ge.sum(dim=-1) - 1                                      # [N, NB2, T]

        # j_idx clamp 上界 = 该 branch 的"最大有效段索引"（最右 True 段的下标）。
        # 正向 valid 段集中在 [0, Nv-2]；反向 valid 段集中在 [NP-Nv, NP-2]：
        # 直接用固定 NP-2 会让正向 target_s = total_s 跑到无效段去（= 0 错位）。
        # 用 (arange * sm_int).max() 拿到最右 True 的 index（前缀连续 True 形态下精确）。
        seg_arange = torch.arange(
            NP - 1, dtype=torch.int64, device=sm_all.device,
        ).view(1, 1, -1)                                                # [1, 1, NP-1]
        sm_int = sm_all.to(torch.int64)                                 # [N, NB2, NP-1]
        max_valid_j = (seg_arange * sm_int).max(dim=-1).values          # [N, NB2]
        max_valid_j_e = max_valid_j.unsqueeze(-1)                       # [N, NB2, 1]
        # 注意：ONNX opset 11 下 Clip / Max / Min 算子都不支持 int64（int 支持
        # 要 opset ≥ 12）。这里用 torch.where 实现整数 clamp，opset 9+ 都支持。
        j_idx_zero = torch.zeros_like(j_idx)
        j_idx = torch.where(j_idx < 0, j_idx_zero, j_idx)
        max_valid_j_b = max_valid_j_e.expand_as(j_idx)
        j_idx = torch.where(j_idx > max_valid_j_b, max_valid_j_b, j_idx)

        # ---- 段内插值取投影点 ----
        j_idx_3d = j_idx.unsqueeze(-1).expand(N, NB2, T, 3)             # [N, NB2, T, 3]
        a_at_j = torch.gather(a_all, 2, j_idx_3d)                       # [N, NB2, T, 3]
        sv_at_j = torch.gather(sv_all, 2, j_idx_3d)                     # [N, NB2, T, 3]
        sl_at_j = torch.gather(sl_all, -1, j_idx)                       # [N, NB2, T]
        cum_s_at_j = torch.gather(cum_s[..., :-1], -1, j_idx)           # [N, NB2, T]

        sl_safe = sl_at_j.clamp_min(1e-9)
        t_local = ((target_s - cum_s_at_j) / sl_safe).clamp(0.0, 1.0)   # [N, NB2, T]
        proj = a_at_j + t_local.unsqueeze(-1) * sv_at_j                 # [N, NB2, T, 3]

        # ---- cost = sum_t |traj - proj|，无 valid 段的 branch 强制 BIG ----
        diff_traj = pos.unsqueeze(1) - proj                             # [N, NB2, T, 3]
        dist_t = torch.linalg.norm(diff_traj, dim=-1)                   # [N, NB2, T]
        cost = dist_t.sum(dim=-1)                                       # [N, NB2]
        branch_has_seg = sm_all.any(dim=-1)                             # [N, NB2]
        big_cost = torch.full_like(cost, BIG)
        cost = torch.where(branch_has_seg, cost, big_cost)

        # ---- 选最优 branch ----
        best_b = cost.argmin(dim=-1)                                    # [N]
        best_b_e = best_b.view(N, 1, 1, 1).expand(N, 1, T, 3)
        out_pos_proj = torch.gather(proj, 1, best_b_e).squeeze(1)       # [N, T, 3]

        # ---- 整个样本完全无 valid branch → fallback 到原 traj（含 mask 全 False）----
        sample_has_branch = branch_has_seg.any(dim=-1)                  # [N]
        out_pos = torch.where(
            sample_has_branch.view(N, 1, 1).expand_as(pos),
            out_pos_proj,
            pos,
        )

        if rest.shape[-1] == 0:
            return out_pos
        return torch.cat([out_pos, rest], dim=-1)

    # ------------------------------------------------------------------
    # road_arc_projection 循环版（保留作 ablation / 单测 oracle）
    # ------------------------------------------------------------------
    def _road_arc_projection_loop(
        self,
        selected_traj: torch.Tensor,    # [N, T, 6]
        ctx: ContextBatch,
    ) -> torch.Tensor:
        """
        旧的循环实现，保留给单测做数值对照。**不要**用在 forward 里：
        含 Python for / .item() / 布尔索引 / searchsorted，无法 trace 进 ONNX，
        且 searchsorted 需要 opset ≥ 16，与 mindspore-lite 1.8.1（opset 11）不兼容。
        """
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

        if not bool(rm.any()):
            return selected_traj

        out_pos = pos.clone()

        traj_seg_len = torch.linalg.norm(pos[:, 1:] - pos[:, :-1], dim=-1)
        traj_cum_s = torch.cat(
            [torch.zeros(N, 1, dtype=pos.dtype, device=pos.device), traj_seg_len.cumsum(dim=1)],
            dim=1,
        )

        for n in range(N):
            best_cost = None
            best_proj = None
            traj_n = pos[n]
            traj_cs = traj_cum_s[n]

            for bi in range(NB):
                mask_nb = rm[n, bi]
                n_valid = int(mask_nb.sum().item())
                if n_valid < 2:
                    continue
                pts_full = rp[n, bi][mask_nb]

                for direction in (1, -1):
                    pts = pts_full if direction == 1 else pts_full.flip(0)
                    pts = pts.to(pos.dtype)

                    seg_vec = pts[1:] - pts[:-1]
                    seg_len = torch.linalg.norm(seg_vec, dim=-1).clamp_min(1e-9)
                    cum_s = torch.cat(
                        [torch.zeros(1, dtype=pos.dtype, device=pos.device), seg_len.cumsum(dim=0)],
                        dim=0,
                    )
                    total_s = cum_s[-1]

                    p0 = traj_n[0]
                    a0 = pts[:-1]
                    d0 = seg_vec
                    dd0 = (d0 * d0).sum(dim=-1).clamp_min(1e-12)
                    t0 = ((p0 - a0) * d0).sum(dim=-1) / dd0
                    t0 = t0.clamp(0.0, 1.0)
                    q0 = a0 + t0.unsqueeze(-1) * d0
                    d2 = ((p0 - q0) ** 2).sum(dim=-1)
                    seg_idx0 = int(torch.argmin(d2).item())
                    entry_s = cum_s[seg_idx0] + t0[seg_idx0] * seg_len[seg_idx0]

                    target_s = (entry_s + traj_cs).clamp(0.0, total_s)

                    j_idx = torch.searchsorted(cum_s, target_s, right=True) - 1
                    j_idx = j_idx.clamp(0, pts.shape[0] - 2)
                    seg_l = seg_len[j_idx]
                    t_local = ((target_s - cum_s[j_idx]) / seg_l.clamp_min(1e-9)).clamp(0.0, 1.0)
                    proj = pts[j_idx] + t_local.unsqueeze(-1) * (pts[j_idx + 1] - pts[j_idx])

                    cost = float(torch.linalg.norm(traj_n - proj, dim=-1).sum().item())
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_proj = proj

            if best_proj is not None:
                out_pos[n] = best_proj

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
