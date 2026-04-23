"""
fusion/code/full_net_v2.py
--------------------------
把各个子网络串起来的组合模型：

    hist -> OutlierFilter -> LSTM1 -> 候选 [B,M,T,6]
                                        |
        (target_task / fixed_targets / target_type / road_network) -> GNN1
                                        |                              |
                                        argmax                       traj_probs
                                        |
                                   主轨迹 [B,T,6]
                                        |
                                ConstraintOptimizer
                                        |
                                  精修轨迹 [B,T,6]
                                  |             |
                                  v             v
                               LSTM2 ------> intent_feat
                              intent/threat      v
                                              GNN2
                                              |
                                       strike_pos/radius/conf

最终输出 [B, M, 68]，布局与 old_plan/model_fusion_v0/full_net.py 完全一致：
    0..59 : 未来 10 步 × 6 维
    60    : intent_class
    61    : threat_prob
    62..64: strike_pos
    65    : strike_radius
    66    : strike_conf
    67    : mode_prob     (来自 GNN1)

用法：
    cd new_plan
    export PYTHONPATH="$PWD"
    python -m fusion.code.full_net_v2 --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.context_schema import (  # noqa: E402
    ContextBatch,
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
        n_modes: int = 3,
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
        self.n_modes = int(n_modes)
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
        B, Tin, _ = x_raw.shape
        _, M, Tout, _ = fut_A_norm.shape

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

    @staticmethod
    def _gather_best_mode(fut_full_phys: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
        B = fut_full_phys.shape[0]
        return fut_full_phys[torch.arange(B, device=fut_full_phys.device), best_idx]

    # --------- 前向 ---------
    def forward(self, x_raw: torch.Tensor, ctx: ContextBatch) -> torch.Tensor:
        if x_raw.ndim != 3:
            raise ValueError(f"x_raw 需 [B,T,D]，实际 {x_raw.shape}")
        x_raw = x_raw.to(self.mean_A.device).float()

        # 1) 异常值剔除
        x_clean, _ = self.outlier_filter(x_raw)

        # 2) A 域编码 → LSTM1
        x_A = self._build_A_input(x_clean)
        fut_A_norm = self.lstm1(x_A)  # [B, M, Tout, 6]

        # 3) 反归一化 + Δ→绝对
        fut_full_phys = self._decode_future_to_abs(x_clean, fut_A_norm)  # [B, M, Tout, 6]

        # 4) GNN1 选轨迹概率
        gnn1_out = self.gnn1(fut_full_phys, ctx)
        traj_probs = gnn1_out["traj_probs"]   # [B, M]

        # 5) argmax 选主轨迹 → 约束优化
        best_idx = torch.argmax(traj_probs, dim=-1)
        selected = self._gather_best_mode(fut_full_phys, best_idx)  # [B, Tout, 6]
        refined = self.constraint(selected, ctx)                     # [B, Tout, 6]

        # 6) LSTM2 意图 + 威胁度
        intent_out = self.lstm2(x_clean, refined)
        intent_logits = intent_out["logits_intent"]  # [B, 4]
        threat_raw = intent_out["threat_raw"]        # [B, 1]

        # 7) GNN2 打击区域
        strike_out = self.gnn2(refined, ctx, intent_logits)
        strike_pos = strike_out["strike_pos"]        # [B, 3]
        strike_radius = strike_out["strike_radius"]  # [B, 1]
        strike_conf = strike_out["strike_conf"]      # [B, 1]

        # 8) 打包 [B, M, 68]
        B, M, Tout, Df = fut_full_phys.shape
        traj_flat = fut_full_phys.reshape(B, M, Tout * Df)  # [B, M, 60]

        intent_class_f = torch.argmax(intent_logits, dim=-1).to(x_raw.dtype)
        threat_prob = torch.sigmoid(threat_raw).squeeze(-1)

        def expand_to_M(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(-1, M, *([-1] * (t.dim() - 1)))

        tail = torch.cat(
            [
                expand_to_M(intent_class_f.unsqueeze(-1)),   # 60
                expand_to_M(threat_prob.unsqueeze(-1)),      # 61
                expand_to_M(strike_pos),                     # 62..64
                expand_to_M(strike_radius),                  # 65
                expand_to_M(strike_conf),                    # 66
                traj_probs.unsqueeze(-1),                    # 67
            ],
            dim=-1,
        )
        return torch.cat([traj_flat, tail], dim=-1)


# ==============================================================================
# 工厂函数
# ==============================================================================

def build_full_net_from_fusion_config(fusion_cfg_path: Path) -> FullNetV2:
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)

    lstm1, gnn1, constraint, lstm2, gnn2, mean_A, std_A = build_subnetworks(fusion_cfg, fusion_cfg_dir)

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
        n_modes=int(full_cfg.get("n_modes", 3)),
        use_delta_A=bool(full_cfg.get("use_delta_A", True)),
    )


# ==============================================================================
# 冒烟测试
# ==============================================================================

def run_smoke(fusion_cfg_path: Path) -> None:
    print("=" * 60)
    print("[Smoke/Fusion] start")
    print("=" * 60)

    model = build_full_net_from_fusion_config(fusion_cfg_path)
    print(f"[Smoke/Fusion] params = {sum(p.numel() for p in model.parameters()):,}")

    # 从 lstm1 的 config 里拿一下 hist_len / fut_len / n_modes 等
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    full_cfg = fusion_cfg.get("full_net", {})
    hist_len = int(full_cfg.get("hist_len", 20))
    feat_dim = int(full_cfg.get("feature_dim", 6))
    n_modes = int(full_cfg.get("n_modes", 3))

    # 用 gnn1 的 ctx_dims 作为 ctx 的形状（各子网络的 context 段应对齐）
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
    expected = (B, n_modes, 68)
    assert out.shape == expected, f"Output shape {tuple(out.shape)} != {expected}"
    print(f"[Smoke/Fusion] forward OK: output shape = {tuple(out.shape)}")

    # 反向（只验证能 backward 即可）
    model.train()
    out = model(x_raw, ctx)
    loss = out.mean()
    loss.backward()
    has_grad_lstm1 = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.lstm1.parameters())
    has_grad_gnn1 = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.gnn1.parameters())
    has_grad_lstm2 = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.lstm2.parameters())
    has_grad_gnn2 = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.gnn2.parameters())
    print(f"[Smoke/Fusion] gradients present:  LSTM1={has_grad_lstm1}  GNN1={has_grad_gnn1}  LSTM2={has_grad_lstm2}  GNN2={has_grad_gnn2}")

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

    # 非 --smoke 情况下，目前只用来验证模型能否构造成功
    build_full_net_from_fusion_config(Path(args.fusion_config))
    print("[Fusion] build_full_net_from_fusion_config 成功（未跑推理）")


if __name__ == "__main__":
    main()
