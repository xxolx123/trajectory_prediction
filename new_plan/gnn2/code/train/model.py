"""
gnn2/code/train/model.py
------------------------
StrikeZoneGNN：打击区域 + 置信度。

当前实现（占位）：MLP
    traj flatten + ctx flatten + intent_feat concat → MLP → [B, 5]
    拆成 pos(3)、softplus(radius)、sigmoid(conf)

TODO（真正的 GNN）：见 README.txt。
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

from common.context_schema import (  # noqa: E402
    ContextBatch,
    flatten_context_for_mlp,
    flattened_ctx_dim,
)


class StrikeZoneGNN(nn.Module):
    def __init__(
        self,
        fut_len: int,
        feat_dim: int,
        intent_feat_dim: int,
        ctx_dims: Dict[str, Any],
        hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fut_len = int(fut_len)
        self.feat_dim = int(feat_dim)
        self.intent_feat_dim = int(intent_feat_dim)
        self.hidden = int(hidden)

        traj_in = self.fut_len * self.feat_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        ctx_in = flattened_ctx_dim(ctx_dims)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.intent_encoder = nn.Sequential(
            nn.Linear(self.intent_feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5),
        )

    def forward(
        self,
        pred_traj: torch.Tensor,
        ctx: ContextBatch,
        intent_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = pred_traj.shape
        if T != self.fut_len or D != self.feat_dim:
            raise ValueError(
                f"pred_traj 形状 [B,T,D]=[{B},{T},{D}] 与配置 [T={self.fut_len},D={self.feat_dim}] 不一致"
            )

        traj_emb = self.traj_encoder(pred_traj.reshape(B, T * D))
        ctx_emb = self.ctx_encoder(flatten_context_for_mlp(ctx))
        intent_emb = self.intent_encoder(intent_feat)
        merged = torch.cat([traj_emb, ctx_emb, intent_emb], dim=-1)
        raw = self.head(merged)  # [B, 5]

        pos = raw[:, 0:3]
        radius = F.softplus(raw[:, 3:4])
        conf = torch.sigmoid(raw[:, 4:5])
        return {"strike_pos": pos, "strike_radius": radius, "strike_conf": conf}


def build_model_from_config(cfg: Dict[str, Any]) -> StrikeZoneGNN:
    m = cfg.get("model", {})
    ctx_cfg = cfg.get("context", {})
    ctx_dims = {k: ctx_cfg.get(k) for k in (
        "target_task_dim", "n_fixed_targets", "fixed_target_dim",
        "target_type_dim", "road_network_dim", "own_info_dim",
    )}
    return StrikeZoneGNN(
        fut_len=int(m.get("fut_len", 10)),
        feat_dim=int(m.get("feat_dim", 6)),
        intent_feat_dim=int(m.get("intent_feat_dim", 4)),
        ctx_dims=ctx_dims,
        hidden=int(m.get("hidden_size", 128)),
        dropout=float(m.get("dropout", 0.0)),
    )
