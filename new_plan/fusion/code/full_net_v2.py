"""
fusion/code/full_net_v2.py
--------------------------
把各个子网络串起来的**纯推理组合模型**（不参与训练；每个子网络各自单独训练）。

FullNetV2 只是在部署 / 推理时把所有子网络和模块按固定流水线组合起来，对外
提供"一次 forward 拿到最终输出"的封装。它自己不学任何权重，也不定义任何
训练 loss。子网络的 ckpt 分别来自各自的训练流程（lstm1、gnn1、lstm2、gnn2）。

流程（inference only）：

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
    0..59 : 未来 10 步 × 6 维（每条候选各自）
    60    : intent_class（每条候选各自）
    61    : threat_prob（每条候选各自）
    62..64: strike_pos（每条候选各自）
    65    : strike_radius（每条候选各自）
    66    : strike_conf（每条候选各自）
    67    : mode_prob = renormalized top-K 概率（3 条和 = 1）

用法：
    cd new_plan
    $env:PYTHONPATH = "$PWD"
    # 推理冒烟：串一遍 forward，检查形状 + top-K 重归一化
    python -m fusion.code.full_net_v2 --smoke
    # 导出 ONNX 部署用：
    python -m fusion.code.export_onnx --onnx-out fusion/full_net_v2.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

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
# FullNetV2
# ==============================================================================

class FullNetV2(nn.Module):
    def __init__(
        self,
        lstm1: nn.Module,
        gnn1: nn.Module,
        constraint: nn.Module,
        lstm2: nn.Module,
        gnn2: nn.Module,
        mean_A,
        std_A,
        hist_len: int = 20,
        fut_len: int = 10,
        feature_dim: int = 6,
        lstm1_modes: int = 5,
        top_k: int = 3,
        use_delta_A: bool = True,
    ) -> None:
        super().__init__()

        self.lstm1 = lstm1
        self.gnn1 = gnn1
        self.constraint = constraint
        self.lstm2 = lstm2
        self.gnn2 = gnn2
        self.outlier_filter = OutlierFilter()

        self.hist_len = int(hist_len)
        self.fut_len = int(fut_len)
        self.feature_dim = int(feature_dim)
        self.lstm1_modes = int(lstm1_modes)
        self.top_k = int(top_k)
        if self.top_k > self.lstm1_modes:
            raise ValueError(
                f"top_k ({self.top_k}) 不能大于 LSTM1 的候选数 ({self.lstm1_modes})"
            )
        # 对外暴露：最终输出的 mode 数 == top_k
        self.n_modes = self.top_k
        self.use_delta_A = bool(use_delta_A)

        mean_A_t = torch.as_tensor(np.asarray(mean_A, dtype=np.float32)).view(1, 1, self.feature_dim)
        std_A_t = torch.as_tensor(np.asarray(std_A, dtype=np.float32)).view(1, 1, self.feature_dim)
        std_A_t = torch.where(std_A_t.abs() < 1e-6, torch.ones_like(std_A_t), std_A_t)
        self.register_buffer("mean_A", mean_A_t)
        self.register_buffer("std_A", std_A_t)

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
            # unsqueeze(1).expand(B, K, ...).reshape(B*K, ...)
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
            own_info=_rep(ctx.own_info),
        )

    # --------- 前向 ---------
    def forward(self, x_raw: torch.Tensor, ctx: ContextBatch) -> torch.Tensor:
        if x_raw.ndim != 3:
            raise ValueError(f"x_raw 需 [B,T,D]，实际 {x_raw.shape}")
        x_raw = x_raw.to(self.mean_A.device).float()

        # 1) 异常值剔除
        x_clean, _ = self.outlier_filter(x_raw)

        # 2) A 域编码 → LSTM1
        x_A = self._build_A_input(x_clean)
        fut_A_norm = self.lstm1(x_A)  # [B, M, Tout, 6] 归一化+delta

        B, M, Tout, Df = fut_A_norm.shape

        # 3) GNN1：按新 API 组 dict，传入归一化空间的候选
        #    GNN1 内部已做 topk + 重归一化，直接拿 top_idx / top_probs 即可。
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

        # 5) 展平到 B*K，走下游（ConstraintOptimizer → LSTM2 → GNN2）
        fut_flat = fut_phys_top.reshape(B * K, Tout, Df)
        hist_flat = (
            x_clean.unsqueeze(1)
                   .expand(B, K, self.hist_len, Df)
                   .contiguous()
                   .reshape(B * K, self.hist_len, Df)
        )
        ctx_flat = self._expand_ctx(ctx, K)

        refined_flat = self.constraint(fut_flat, ctx_flat)                # [B*K, Tout, 6]
        intent_out = self.lstm2(hist_flat, refined_flat)
        intent_logits = intent_out["logits_intent"]                       # [B*K, 4]
        threat_raw = intent_out["threat_raw"]                             # [B*K, 1]

        strike_out = self.gnn2(refined_flat, ctx_flat, intent_logits)
        strike_pos = strike_out["strike_pos"]        # [B*K, 3]
        strike_radius = strike_out["strike_radius"]  # [B*K, 1]
        strike_conf = strike_out["strike_conf"]      # [B*K, 1]

        # 6) 拼 [B, K, 68]
        traj_flat = fut_phys_top.reshape(B, K, Tout * Df)   # [B, K, 60]

        intent_class_f = (
            torch.argmax(intent_logits, dim=-1)
                 .to(fut_phys_top.dtype)
                 .reshape(B, K, 1)
        )
        threat_prob = torch.sigmoid(threat_raw).reshape(B, K, 1)
        strike_pos_r = strike_pos.reshape(B, K, 3)
        strike_radius_r = strike_radius.reshape(B, K, 1)
        strike_conf_r = strike_conf.reshape(B, K, 1)
        mode_prob = top_probs.reshape(B, K, 1)

        tail = torch.cat(
            [
                intent_class_f,    # 60
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
    ) = build_subnetworks(fusion_cfg, fusion_cfg_dir)

    full_cfg = fusion_cfg.get("full_net", {})
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
    )


# ==============================================================================
# 冒烟测试
# ==============================================================================

def run_smoke(fusion_cfg_path: Path) -> None:
    """
    推理冒烟：FullNetV2 是纯推理封装（每个子网络各自单独训练），
    这里只检查 forward 的形状 / 数值约束，不做反向梯度检查。
    """
    print("=" * 60)
    print("[Smoke/Fusion] start  (inference-only wrapper)")
    print("=" * 60)

    model = build_full_net_from_fusion_config(fusion_cfg_path)
    print(f"[Smoke/Fusion] params = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Smoke/Fusion] lstm1_modes = {model.lstm1_modes}, top_k = {model.top_k}")

    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    full_cfg = fusion_cfg.get("full_net", {})
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

    # 2) top-K 重归一化（每行概率和 ≈ 1，且全正）
    mode_prob = out[..., 67]
    mode_prob_sum = mode_prob.sum(dim=-1)
    err = (mode_prob_sum - 1.0).abs().max().item()
    min_prob = mode_prob.min().item()
    assert err < 1e-5, f"top-K 重归一化失败：每行和应 ≈ 1，误差 {err}"
    assert min_prob >= 0.0, f"mode_prob 出现负值：{min_prob}"
    print(f"[Smoke/Fusion] top-K renorm OK: max |sum-1| = {err:.2e}, min_prob = {min_prob:.4f}")

    # 3) 68 维字段值域粗检查（threat_prob ∈ [0,1], strike_conf ∈ [0,1]）
    threat = out[..., 61]
    conf = out[..., 66]
    assert threat.min() >= 0 and threat.max() <= 1, f"threat_prob 越界: [{threat.min()}, {threat.max()}]"
    assert conf.min() >= 0 and conf.max() <= 1, f"strike_conf 越界: [{conf.min()}, {conf.max()}]"
    print(f"[Smoke/Fusion] field ranges OK: "
          f"threat_prob in [{threat.min():.3f}, {threat.max():.3f}], "
          f"strike_conf in [{conf.min():.3f}, {conf.max():.3f}]")

    # 4) 相同输入 → 相同输出（确定性，eval 模式下）
    with torch.no_grad():
        out2 = model(x_raw, ctx)
    diff = (out - out2).abs().max().item()
    assert diff < 1e-6, f"eval 模式下 forward 不确定？max diff = {diff}"
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
                        help="冒烟测试：串联所有子网络跑一遍 forward+backward")
    args = parser.parse_args()

    if args.smoke:
        run_smoke(Path(args.fusion_config))
        return

    build_full_net_from_fusion_config(Path(args.fusion_config))
    print("[Fusion] build_full_net_from_fusion_config 成功（未跑推理）")


if __name__ == "__main__":
    main()
