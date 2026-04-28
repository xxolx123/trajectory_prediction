"""
fusion/code/full_net_v2.py
--------------------------
把各个子网络串起来的**纯推理组合模型**（不参与训练；每个子网络各自单独训练）。

FullNetV2 只是在部署 / 推理时把所有子网络和模块按固定流水线组合起来，对外
提供"一次 forward 拿到最终输出"的封装。它自己不学任何权重，也不定义任何
训练 loss。子网络的 ckpt 分别来自各自的训练流程（lstm1、gnn1、lstm2、gnn2）。

模块开关（来自 fusion/config.yaml 的 enable 字段）：
    lstm1 / gnn1                 -> 必开（流水线主干）
    constraint_optimizer / lstm2 / gnn2 -> 可关；fusion 内部跳过对应步骤，
                                            输出 [B, K, 68] 中相应位置写哨兵值。

哨兵值约定：
    constraint_optimizer 关：0..59 列 = LSTM1 物理轨迹（不再投影路网）
    lstm2 关：              60 列 = -1（intent_class 占位），61 列 = NaN
    gnn2 关：               62..66 列 = NaN
    mode_prob (67 列)        始终由 GNN1 输出（永远有效）

流程（inference only，**所有模块都启用时**）：

    hist -> OutlierFilter -> LSTM1 -> M 条候选 [B,M,T,6]（归一化+delta）
                                        |
        (task_type / type / position) → GNN1
                                        |
                                   GNN1 内部就做 topk + 重归一化
                                   直接输出 top_idx [B,K] 和 top_probs [B,K]
                                        |
              gather top-K 归一化候选 → 反归一化 + cumsum → 物理绝对坐标 [B, K, T, 6]
                                        |
                        展平成 [B*K, T, 6] 走下游三次
                                        |
                ConstraintOptimizer → refined [B*K, T, 6]
                                        |
                       LSTM2 (意图/威胁) + GNN2 (打击)
                                        |
                        reshape 回 [B, K, *] 拼成 [B, K, 68]

最终输出 [B, K=3, 68]，布局与 old_plan/model_fusion_v0/full_net.py 兼容：
    0..59 : 未来 10 步 × 6 维（**路网约束后**的 refined 轨迹，每条候选各自）
    60    : intent_class（每条候选各自）
    61    : threat_prob（每条候选各自）
    62..64: strike_pos（每条候选各自）
    65    : strike_radius（每条候选各自）
    66    : strike_conf（每条候选各自）
    67    : mode_prob = renormalized top-K 概率（K 条和 = 1）

用法：
    cd new_plan
    $env:PYTHONPATH = "$PWD"
    # 推理冒烟：串一遍 forward，检查形状 + top-K 重归一化
    python -m fusion.code.full_net_v2 --smoke
    # 端到端测试（按 fusion/config.yaml 当前 enable 状态跑）
    python -m fusion.code.test_pipeline
    # 导出 ONNX 部署用（同样按 enable 状态导，禁用模块对应位置写哨兵值）
    python -m fusion.code.export_onnx --onnx-out fusion/full_net_v2.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.context_schema import (  # noqa: E402
    ContextBatch,
    DEFAULT_CTX_DIMS,
    build_ctx_dims_from_config,
    build_dummy_context,
)
from common.outlier_filter import OutlierFilter  # noqa: E402

from fusion.code.build import build_subnetworks, load_fusion_config  # noqa: E402


# ==============================================================================
# 哨兵值常量
# ==============================================================================

INTENT_SENTINEL = -1.0       # lstm2 禁用时 intent_class 填这个值
SCALAR_NAN = float("nan")    # threat_prob / strike_* 禁用时填 NaN


# ==============================================================================
# FullNetV2
# ==============================================================================

class FullNetV2(nn.Module):
    # LSTM2 输入工程化后维度（6 原始 + 3 Δ + 1 dist + 1 speed）
    LSTM2_INPUT_DIM = 11

    def __init__(
        self,
        lstm1: nn.Module,
        gnn1: nn.Module,
        constraint: Optional[nn.Module],
        lstm2: Optional[nn.Module],
        gnn2: Optional[nn.Module],
        mean_A,
        std_A,
        hist_len: int = 20,
        fut_len: int = 10,
        feature_dim: int = 6,
        lstm1_modes: int = 5,
        top_k: int = 3,
        use_delta_A: bool = True,
        n_intent_classes: int = 4,
        enable_flags: Optional[Dict[str, bool]] = None,
        lstm2_mean=None,
        lstm2_std=None,
    ) -> None:
        super().__init__()

        if lstm1 is None or gnn1 is None:
            raise ValueError("FullNetV2: lstm1 和 gnn1 是流水线主干，不能为 None")

        # 必有
        self.lstm1 = lstm1
        self.gnn1 = gnn1
        self.outlier_filter = OutlierFilter()

        # 可选；nn.Module.__setattr__ 支持 None：会在 _modules 里登记成 None，
        # 不会污染 state_dict / parameters；forward 里直接用 if 判断即可。
        self.constraint = constraint  # type: Optional[nn.Module]
        self.lstm2 = lstm2            # type: Optional[nn.Module]
        self.gnn2 = gnn2              # type: Optional[nn.Module]

        self.hist_len = int(hist_len)
        self.fut_len = int(fut_len)
        self.feature_dim = int(feature_dim)
        self.lstm1_modes = int(lstm1_modes)
        self.top_k = int(top_k)
        if self.top_k > self.lstm1_modes:
            raise ValueError(
                f"top_k ({self.top_k}) 不能大于 LSTM1 的候选数 ({self.lstm1_modes})"
            )
        self.n_modes = self.top_k
        self.use_delta_A = bool(use_delta_A)
        self.n_intent_classes = int(n_intent_classes)

        # enable_flags 仅作元信息，便于打印 / 调试
        if enable_flags is None:
            enable_flags = {
                "lstm1": True,
                "gnn1": True,
                "constraint_optimizer": constraint is not None,
                "lstm2": lstm2 is not None,
                "gnn2": gnn2 is not None,
            }
        self.enable_flags: Dict[str, bool] = dict(enable_flags)

        # ---- LSTM1 A 域 scaler（6 维）----
        mean_A_t = torch.as_tensor(np.asarray(mean_A, dtype=np.float32)).view(1, 1, self.feature_dim)
        std_A_t = torch.as_tensor(np.asarray(std_A, dtype=np.float32)).view(1, 1, self.feature_dim)
        std_A_t = torch.where(std_A_t.abs() < 1e-6, torch.ones_like(std_A_t), std_A_t)
        self.register_buffer("mean_A", mean_A_t)
        self.register_buffer("std_A", std_A_t)

        # ---- LSTM2 11 维 scaler（即使 lstm2 关闭也注册 0/1 占位，避免 forward 分支）----
        if lstm2_mean is not None and lstm2_std is not None:
            mean_l = torch.as_tensor(
                np.asarray(lstm2_mean, dtype=np.float32)
            ).view(1, 1, self.LSTM2_INPUT_DIM)
            std_l = torch.as_tensor(
                np.asarray(lstm2_std, dtype=np.float32)
            ).view(1, 1, self.LSTM2_INPUT_DIM)
            std_l = torch.where(std_l.abs() < 1e-6, torch.ones_like(std_l), std_l)
        else:
            mean_l = torch.zeros(1, 1, self.LSTM2_INPUT_DIM, dtype=torch.float32)
            std_l = torch.ones(1, 1, self.LSTM2_INPUT_DIM, dtype=torch.float32)
        self.register_buffer("mean_lstm2", mean_l)
        self.register_buffer("std_lstm2", std_l)

    # --------- A 域归一化工具 ---------
    def _norm_A(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_A) / self.std_A

    def _denorm_A(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_A + self.mean_A

    # --------- 原始 -> A 域输入 ---------
    def _build_A_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, T, D = x_raw.shape
        pos = x_raw[..., 0:3]
        vel = x_raw[..., 3:6]
        if self.use_delta_A:
            delta_pos = torch.zeros_like(pos)
            if T > 1:
                delta_pos[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
            pos_A = delta_pos
        else:
            pos_A = pos
        return self._norm_A(torch.cat([pos_A, vel], dim=-1))

    # --------- LSTM1 输出 -> 绝对坐标 ---------
    def _decode_future_to_abs(
        self,
        x_raw: torch.Tensor,
        fut_A_norm: torch.Tensor,
    ) -> torch.Tensor:
        """把 [B, K, Tout, 6] 的"归一化+delta"候选解码到物理绝对坐标。"""
        B, Tin, _ = x_raw.shape
        _, K, Tout, _ = fut_A_norm.shape

        fut_A_raw = self._denorm_A(fut_A_norm)
        fut_pos_delta = fut_A_raw[..., 0:3]
        fut_vel = fut_A_raw[..., 3:6]

        last_hist_pos = x_raw[:, -1, 0:3].view(B, 1, 1, 3)
        if self.use_delta_A:
            fut_pos_cum = torch.cumsum(fut_pos_delta, dim=2)
            fut_pos_phys = last_hist_pos + fut_pos_cum
        else:
            fut_pos_phys = fut_pos_delta
        return torch.cat([fut_pos_phys, fut_vel], dim=-1)

    # --------- ContextBatch → gnn1 batch dict ---------
    @staticmethod
    def _ctx_to_gnn1_batch(
        cand_trajs: torch.Tensor,
        ctx: ContextBatch,
    ) -> Dict[str, torch.Tensor]:
        """
        把 fut_A_norm 和 ContextBatch 的 3 个字段打包成 gnn1 期望的 dict。
        GNN1 只消费 task_type / type / position，不读路网。
        """
        return {
            "cand_trajs": cand_trajs,
            "task_type":  ctx.task_type,
            "type":       ctx.type,
            "position":   ctx.position,
        }

    # --------- ContextBatch 按 K 扩展到 B*K ---------
    @staticmethod
    def _expand_ctx(ctx: ContextBatch, K: int) -> ContextBatch:
        def _rep(t: torch.Tensor) -> torch.Tensor:
            B = t.shape[0]
            rest = t.shape[1:]
            return (
                t.unsqueeze(1)
                 .expand(B, K, *rest)
                 .contiguous()
                 .reshape(B * K, *rest)
            )

        return ContextBatch(
            task_type=_rep(ctx.task_type),
            type=_rep(ctx.type),
            position=_rep(ctx.position),
            road_points=_rep(ctx.road_points),
            road_mask=_rep(ctx.road_mask),
            eta=_rep(ctx.eta),
        )

    # --------- ContextBatch 缺失字段补零（允许 ctx 字段为 None）---------
    def _normalize_ctx(
        self,
        ctx: ContextBatch,
        batch_size: int,
        device: torch.device,
    ) -> ContextBatch:
        """
        把 ctx 中可能为 None 的字段补成全零张量。
        用于：测试 / 推理时上层只关心已启用模块需要的字段，
              其他字段（如 lstm2/gnn2 关闭时的 eta）传 None 即可。

        对各字段缺失时的形状假设：
          - task_type / type / eta:        [B] long
          - position:                       [B, 3] float
          - road_points:                    [B, NB, NP, 3] float（NB=NP=1 占位）
          - road_mask:                      [B, NB, NP] bool
        若某字段非 None，直接保留；若 None，用对应 dtype/形状全零张量补。
        """
        B = int(batch_size)

        def _zeros(shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        task_type = ctx.task_type if ctx.task_type is not None else _zeros((B,), torch.long)
        type_id = ctx.type if ctx.type is not None else _zeros((B,), torch.long)
        position = ctx.position if ctx.position is not None else _zeros((B, 3), torch.float32)

        # 路网默认占位：NB=1, NP=2，全 0 + 全 False；够 ConstraintOptimizer 走
        # "全 batch 无 valid segment" 的 fallback 分支（等价 pass_through）
        if ctx.road_points is not None:
            road_points = ctx.road_points
        else:
            road_points = _zeros((B, 1, 2, 3), torch.float32)
        if ctx.road_mask is not None:
            road_mask = ctx.road_mask
        else:
            road_mask = torch.zeros(B, 1, 2, dtype=torch.bool, device=device)

        eta = ctx.eta if ctx.eta is not None else _zeros((B,), torch.long)

        return ContextBatch(
            task_type=task_type,
            type=type_id,
            position=position,
            road_points=road_points,
            road_mask=road_mask,
            eta=eta,
        )

    # --------- 哨兵填充工具 ---------
    @staticmethod
    def _full(shape, value, dtype, device) -> torch.Tensor:
        return torch.full(shape, value, dtype=dtype, device=device)

    # --------- LSTM2 11 维特征工程（与 lstm2/data/dataset.py 的 numpy 版对齐）---------
    @staticmethod
    def _engineer_lstm2_features(
        fut: torch.Tensor,           # [N, T, 6]   物理 km / km·s⁻¹
        position: torch.Tensor,      # [N, 3]      物理 km
    ) -> torch.Tensor:
        """
        把 (fut [N,T,6], position [N,3]) 扩展到 [N,T,11]：
            [x, y, z, vx, vy, vz, dx, dy, dz, ||Δ||, ||v||]
        其中 (dx, dy, dz) = pos_t - position。
        与 lstm2/code/data/dataset.py:engineer_features_np 完全等价（torch 版）。
        """
        pos = fut[..., 0:3]                                            # [N, T, 3]
        vel = fut[..., 3:6]                                            # [N, T, 3]
        delta = pos - position.unsqueeze(1)                            # [N, T, 3]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)          # [N, T, 1]
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)           # [N, T, 1]
        return torch.cat([fut, delta, dist, speed], dim=-1)            # [N, T, 11]

    # --------- 前向 ---------
    def forward(self, x_raw: torch.Tensor, ctx: ContextBatch) -> torch.Tensor:
        if x_raw.ndim != 3:
            raise ValueError(f"x_raw 需 [B,T,D]，实际 {x_raw.shape}")
        x_raw = x_raw.to(self.mean_A.device).float()
        device = x_raw.device
        B = int(x_raw.shape[0])

        # 0) 把 ctx 中可能为 None 的字段补零
        ctx = self._normalize_ctx(ctx, batch_size=B, device=device)

        # 1) 异常值剔除
        x_clean, _ = self.outlier_filter(x_raw)

        # 2) A 域编码 → LSTM1
        x_A = self._build_A_input(x_clean)
        fut_A_norm = self.lstm1(x_A)  # [B, M, Tout, 6] 归一化+delta

        B, M, Tout, Df = fut_A_norm.shape

        # 3) GNN1：按新 API 组 dict，传入归一化空间的候选
        gnn1_batch = self._ctx_to_gnn1_batch(fut_A_norm, ctx)
        gnn1_out = self.gnn1(gnn1_batch)
        top_idx = gnn1_out["top_idx"]       # [B, K]
        top_probs = gnn1_out["top_probs"]   # [B, K]，K 条和 = 1
        K = int(top_idx.shape[-1])
        if K != self.top_k:
            raise RuntimeError(
                f"GNN1 输出的 top_k={K} 与 fusion 配置的 top_k={self.top_k} 不一致"
            )

        # 4) gather top-K 归一化候选 → 反归一化到物理绝对坐标
        idx_exp = (
            top_idx[..., None, None]
            .expand(B, K, Tout, Df)
        )
        fut_A_top = torch.gather(fut_A_norm, 1, idx_exp)                # [B, K, Tout, 6]
        fut_phys_top = self._decode_future_to_abs(x_clean, fut_A_top)   # [B, K, Tout, 6]

        # 5) 展平到 B*K
        fut_flat = fut_phys_top.reshape(B * K, Tout, Df)
        ctx_flat = self._expand_ctx(ctx, K)

        # 6) ConstraintOptimizer：禁用时直接用未约束物理轨迹
        if self.constraint is not None:
            refined_flat = self.constraint(fut_flat, ctx_flat)          # [B*K, Tout, 6]
        else:
            refined_flat = fut_flat

        # 7) LSTM2（意图/威胁）：禁用时填哨兵值
        #    新接口：forward(fut_norm [N,T,11], position [N,3])
        #    fusion 在外面做 11 维工程化 + scaler 归一化，再喂给模型
        if self.lstm2 is not None:
            feat11 = self._engineer_lstm2_features(
                refined_flat, ctx_flat.position,
            )                                                           # [B*K, T, 11]
            feat_norm = (feat11 - self.mean_lstm2) / self.std_lstm2     # 广播 [1,1,11]
            intent_out = self.lstm2(feat_norm, ctx_flat.position)
            intent_logits = intent_out["logits_intent"]                 # [B*K, n_intent]
            threat_raw = intent_out["threat_raw"]                       # [B*K, 1]
            intent_class = (
                torch.argmax(intent_logits, dim=-1)
                     .to(refined_flat.dtype)
                     .reshape(B, K, 1)
            )
            threat_prob = torch.sigmoid(threat_raw).reshape(B, K, 1)
        else:
            intent_class = self._full(
                (B, K, 1), INTENT_SENTINEL, refined_flat.dtype, device,
            )
            threat_prob = self._full(
                (B, K, 1), SCALAR_NAN, refined_flat.dtype, device,
            )

        # 8) GNN2（打击区域/置信度）：禁用时填哨兵值
        if self.gnn2 is not None:
            strike_out = self.gnn2(refined_flat, ctx_flat.eta)
            strike_pos_r = strike_out["strike_pos"].reshape(B, K, 3)
            strike_radius_r = strike_out["strike_radius"].reshape(B, K, 1)
            strike_conf_r = strike_out["strike_conf"].reshape(B, K, 1)
        else:
            strike_pos_r = self._full(
                (B, K, 3), SCALAR_NAN, refined_flat.dtype, device,
            )
            strike_radius_r = self._full(
                (B, K, 1), SCALAR_NAN, refined_flat.dtype, device,
            )
            strike_conf_r = self._full(
                (B, K, 1), SCALAR_NAN, refined_flat.dtype, device,
            )

        # 9) 拼 [B, K, 68]：0..59 用路网约束后（或 constraint 禁用时的物理）轨迹
        traj_flat = refined_flat.reshape(B, K, Tout * Df)               # [B, K, 60]
        mode_prob = top_probs.reshape(B, K, 1)                          # 始终有效

        tail = torch.cat(
            [
                intent_class,      # 60
                threat_prob,       # 61
                strike_pos_r,      # 62..64
                strike_radius_r,   # 65
                strike_conf_r,     # 66
                mode_prob,         # 67
            ],
            dim=-1,
        )
        return torch.cat([traj_flat, tail], dim=-1)


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_full_net_from_fusion_config(fusion_cfg_path: Path) -> FullNetV2:
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)

    (
        lstm1,
        gnn1,
        constraint,
        lstm2,
        gnn2,
        mean_A,
        std_A,
        lstm1_modes,
        top_k,
        enable_flags,
        lstm2_mean,
        lstm2_std,
    ) = build_subnetworks(fusion_cfg, fusion_cfg_dir)

    full_cfg = fusion_cfg.get("full_net", {}) or {}
    return FullNetV2(
        lstm1=lstm1,
        gnn1=gnn1,
        constraint=constraint,
        lstm2=lstm2,
        gnn2=gnn2,
        mean_A=mean_A,
        std_A=std_A,
        hist_len=int(full_cfg.get("hist_len", 20)),
        fut_len=int(full_cfg.get("fut_len", 10)),
        feature_dim=int(full_cfg.get("feature_dim", 6)),
        lstm1_modes=int(lstm1_modes),
        top_k=int(top_k),
        use_delta_A=bool(full_cfg.get("use_delta_A", True)),
        n_intent_classes=int(full_cfg.get("n_intent_classes", 4)),
        enable_flags=enable_flags,
        lstm2_mean=lstm2_mean,
        lstm2_std=lstm2_std,
    )


# ==============================================================================
# 冒烟测试
# ==============================================================================

def run_smoke(fusion_cfg_path: Path) -> None:
    """
    推理冒烟：FullNetV2 是纯推理封装（每个子网络各自单独训练），
    这里只检查 forward 的形状 / 数值约束，不做反向梯度检查。

    注意：因为可能有模块禁用（lstm2/gnn2 关闭时 60..66 列含 NaN/-1），
    所以这里对禁用模块对应的列只校验形状，不校验数值范围。
    """
    print("=" * 60)
    print("[Smoke/Fusion] start  (inference-only wrapper)")
    print("=" * 60)

    model = build_full_net_from_fusion_config(fusion_cfg_path)
    print(f"[Smoke/Fusion] params = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Smoke/Fusion] lstm1_modes = {model.lstm1_modes}, top_k = {model.top_k}")
    print(f"[Smoke/Fusion] enable_flags = {model.enable_flags}")

    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    full_cfg = fusion_cfg.get("full_net", {}) or {}
    hist_len = int(full_cfg.get("hist_len", 20))
    feat_dim = int(full_cfg.get("feature_dim", 6))

    # 用 gnn1 的 ctx_dims 作为 ctx 的形状
    gnn1_sec = fusion_cfg.get("gnn1", {})
    gnn1_cfg_path = fusion_cfg_dir / gnn1_sec.get("config", "")
    import yaml
    with open(gnn1_cfg_path, "r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    ctx_dims = build_ctx_dims_from_config(gnn1_cfg)

    device = torch.device("cpu")
    model = model.to(device).eval()

    B = 2
    torch.manual_seed(0)
    x_raw = torch.randn(B, hist_len, feat_dim, device=device)
    ctx = build_dummy_context(B, device=device, ctx_dims=ctx_dims)

    with torch.no_grad():
        out = model(x_raw, ctx)

    # 1) 形状
    expected = (B, model.top_k, 68)
    assert out.shape == expected, f"Output shape {tuple(out.shape)} != {expected}"
    print(f"[Smoke/Fusion] forward OK: output shape = {tuple(out.shape)}")

    # 2) top-K 重归一化（mode_prob，67 列；始终有效）
    mode_prob = out[..., 67]
    mode_prob_sum = mode_prob.sum(dim=-1)
    err = (mode_prob_sum - 1.0).abs().max().item()
    min_prob = mode_prob.min().item()
    assert err < 1e-5, f"top-K 重归一化失败：每行和应 ≈ 1，误差 {err}"
    assert min_prob >= 0.0, f"mode_prob 出现负值：{min_prob}"
    print(f"[Smoke/Fusion] top-K renorm OK: max |sum-1| = {err:.2e}, min_prob = {min_prob:.4f}")

    # 3) 60..66 各字段：根据 enable 状态做不同校验
    intent_col = out[..., 60]
    threat_col = out[..., 61]
    strike_pos_col = out[..., 62:65]
    strike_radius_col = out[..., 65]
    strike_conf_col = out[..., 66]

    if model.enable_flags.get("lstm2", False):
        assert torch.isfinite(threat_col).all(), "lstm2 启用，threat_prob 不应含 NaN"
        assert threat_col.min() >= 0 and threat_col.max() <= 1, \
            f"threat_prob 越界: [{threat_col.min()}, {threat_col.max()}]"
        print(f"[Smoke/Fusion] lstm2 ON  threat_prob ∈ "
              f"[{threat_col.min():.3f}, {threat_col.max():.3f}]")
    else:
        assert torch.all(intent_col == INTENT_SENTINEL), \
            "lstm2 禁用，intent_class 应全 = -1"
        assert torch.isnan(threat_col).all(), "lstm2 禁用，threat_prob 应全 NaN"
        print("[Smoke/Fusion] lstm2 OFF intent=-1 / threat=NaN  OK")

    if model.enable_flags.get("gnn2", False):
        assert torch.isfinite(strike_pos_col).all(), "gnn2 启用，strike_pos 不应含 NaN"
        assert torch.isfinite(strike_radius_col).all(), "gnn2 启用，strike_radius 不应含 NaN"
        assert strike_conf_col.min() >= 0 and strike_conf_col.max() <= 1, \
            f"strike_conf 越界: [{strike_conf_col.min()}, {strike_conf_col.max()}]"
        print(f"[Smoke/Fusion] gnn2  ON  strike_conf ∈ "
              f"[{strike_conf_col.min():.3f}, {strike_conf_col.max():.3f}]")
    else:
        assert torch.isnan(strike_pos_col).all(), "gnn2 禁用，strike_pos 应全 NaN"
        assert torch.isnan(strike_radius_col).all(), "gnn2 禁用，strike_radius 应全 NaN"
        assert torch.isnan(strike_conf_col).all(), "gnn2 禁用，strike_conf 应全 NaN"
        print("[Smoke/Fusion] gnn2  OFF strike_*=NaN  OK")

    # 4) 0..59 轨迹一定有限
    traj_col = out[..., 0:60]
    assert torch.isfinite(traj_col).all(), "0..59 轨迹列含 NaN/Inf"

    # 5) 相同输入 → 相同输出（确定性，eval 模式下；NaN 不能直接比 ==，所以
    #    用 nan_to_num 再比）
    with torch.no_grad():
        out2 = model(x_raw, ctx)
    a = torch.nan_to_num(out, nan=0.0)
    b = torch.nan_to_num(out2, nan=0.0)
    diff = (a - b).abs().max().item()
    assert diff < 1e-6, f"eval 模式下 forward 不确定？max diff = {diff}"
    # 同样校验 NaN 出现位置一致
    assert torch.equal(torch.isnan(out), torch.isnan(out2)), "两次 forward 的 NaN 模式不一致"
    print(f"[Smoke/Fusion] determinism OK: repeated forward diff = {diff:.2e}")

    print("=" * 60)
    print("[Smoke/Fusion] OK")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="FullNetV2 组合模型入口。")
    parser.add_argument(
        "--fusion-config",
        type=str,
        default=str(REPO_ROOT / "fusion" / "config.yaml"),
        help="fusion/config.yaml 的路径",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试：构造随机输入跑一遍 forward 并校验形状/哨兵值")
    args = parser.parse_args()

    if args.smoke:
        run_smoke(Path(args.fusion_config))
        return

    build_full_net_from_fusion_config(Path(args.fusion_config))
    print("[Fusion] build_full_net_from_fusion_config 成功（未跑推理）")


if __name__ == "__main__":
    main()
