"""
gnn2/code/train/model.py
------------------------
StrikeZoneGNN：打击区域 + 置信度。

业务接口（与最终需求对齐）::

    输入:
      pred_traj  [B, T, 6]   ConstraintOptimizer 输出的"路网约束后预测轨迹"
                            （fusion 里 B = batch * top_k = batch * 3，三条候选各跑一次）
      eta        [B]         我方预计到达时间（int64 秒；模型内部归一化为小时）

    输出:
      strike_pos    [B, 3]   打击区域中心 (x, y, z) km
      strike_radius [B, 1]   打击半径 km（>= 0，softplus）
      strike_conf   [B, 1]   置信度 (0..1，sigmoid)

当前实现：MLP 占位（concat → MLP → 5 维）。等打击区域 GT 接入后再换成真正的 GNN。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class StrikeZoneGNN(nn.Module):
    def __init__(
        self,
        fut_len: int,
        feat_dim: int,
        eta_scale_seconds: float = 3600.0,
        hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fut_len = int(fut_len)
        self.feat_dim = int(feat_dim)
        self.eta_scale_seconds = float(eta_scale_seconds)
        self.hidden = int(hidden)

        # ---- 路网约束后轨迹编码 ----
        traj_in = self.fut_len * self.feat_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        # ---- ETA 编码（标量 → hidden） ----
        self.eta_encoder = nn.Sequential(
            nn.Linear(1, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
        )

        # ---- 融合头 ----
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5),    # 3 (pos) + 1 (radius) + 1 (conf)
        )

    def forward(
        self,
        pred_traj: torch.Tensor,    # [B, T, 6]   km / km·s⁻¹
        eta: torch.Tensor,          # [B]         long 秒
    ) -> Dict[str, torch.Tensor]:
        if pred_traj.ndim != 3:
            raise ValueError(f"pred_traj 形状应为 [B, T, D]，实际 {tuple(pred_traj.shape)}")
        B, T, D = pred_traj.shape
        if T != self.fut_len or D != self.feat_dim:
            raise ValueError(
                f"pred_traj 形状 [B,T,D]=[{B},{T},{D}] 与配置 "
                f"[T={self.fut_len},D={self.feat_dim}] 不一致"
            )
        if eta.ndim != 1 or int(eta.shape[0]) != B:
            raise ValueError(
                f"eta 形状应为 [B={B}]，实际 {tuple(eta.shape)}"
            )

        traj_emb = self.traj_encoder(pred_traj.reshape(B, T * D))

        eta_hours = (eta.to(dtype=pred_traj.dtype) / self.eta_scale_seconds).reshape(B, 1)
        eta_emb = self.eta_encoder(eta_hours)

        merged = torch.cat([traj_emb, eta_emb], dim=-1)
        raw = self.head(merged)                                    # [B, 5]

        pos = raw[:, 0:3]                                          # km xyz
        radius = F.softplus(raw[:, 3:4])                           # km, >=0
        conf = torch.sigmoid(raw[:, 4:5])                          # [0,1]
        return {
            "strike_pos": pos,
            "strike_radius": radius,
            "strike_conf": conf,
        }


def build_model_from_config(cfg: Dict[str, Any]) -> StrikeZoneGNN:
    m = cfg.get("model", {})
    ctx = cfg.get("context", {}) or {}
    return StrikeZoneGNN(
        fut_len=int(m.get("fut_len", 10)),
        feat_dim=int(m.get("feat_dim", 6)),
        eta_scale_seconds=float(ctx.get("eta_scale_seconds", 3600.0)),
        hidden=int(m.get("hidden_size", 128)),
        dropout=float(m.get("dropout", 0.0)),
    )
