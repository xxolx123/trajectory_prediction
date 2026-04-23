"""
gnn1/code/train/model.py
------------------------
TrajSelectionGNN：根据"LSTM1 的 M 条候选 + 目标信息"给每条轨迹打分。

当前实现（**占位**）：纯 MLP
    轨迹 flatten + ctx flatten → concat → MLP → [B, M] logit → softmax

TODO（后续替换为真正的 GNN）：
    节点 = M 条轨迹（用 LSTM/Conv 聚合成向量）+ N_tgt 个固定目标节点 + 1 个全局节点
    边   = 轨迹↔目标（按轨迹终点到目标的距离建权）
           轨迹↔全局 / 目标↔全局（全连接）
           轨迹↔轨迹（按终点相似度，可选）
    消息传递 2~3 层 GraphSAGE / GAT
    读出 = 只在"轨迹节点"上接 Linear → 1 维 logit → softmax
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

# 把 new_plan/ 加到 sys.path，便于 `from common.context_schema import ...`
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.context_schema import (  # noqa: E402
    ContextBatch,
    flatten_context_for_mlp,
    flattened_ctx_dim,
)


class TrajSelectionGNN(nn.Module):
    def __init__(
        self,
        n_modes: int,
        fut_len: int,
        feat_dim: int,
        ctx_dims: Dict[str, Any],
        hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_modes = int(n_modes)
        self.fut_len = int(fut_len)
        self.feat_dim = int(feat_dim)
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

        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand_trajs: torch.Tensor, ctx: ContextBatch) -> Dict[str, torch.Tensor]:
        """
        Args:
            cand_trajs: [B, M, T, D]
            ctx:        ContextBatch
        Returns:
            {"traj_logits": [B, M], "traj_probs": [B, M]}
        """
        B, M, T, D = cand_trajs.shape
        if M != self.n_modes or T != self.fut_len or D != self.feat_dim:
            raise ValueError(
                f"cand_trajs 形状 [B,M,T,D]=[{B},{M},{T},{D}] 与配置 "
                f"[M={self.n_modes},T={self.fut_len},D={self.feat_dim}] 不一致"
            )

        flat = cand_trajs.reshape(B, M, T * D)
        traj_emb = self.traj_encoder(flat)  # [B, M, H]

        ctx_flat = flatten_context_for_mlp(ctx)
        ctx_emb = self.ctx_encoder(ctx_flat)  # [B, H]
        ctx_emb_expand = ctx_emb.unsqueeze(1).expand(-1, M, -1)  # [B, M, H]

        scorer_in = torch.cat([traj_emb, ctx_emb_expand], dim=-1)  # [B, M, 2H]
        logits = self.scorer(scorer_in).squeeze(-1)                # [B, M]
        probs = torch.softmax(logits, dim=-1)
        return {"traj_logits": logits, "traj_probs": probs}


def build_model_from_config(cfg: Dict[str, Any]) -> TrajSelectionGNN:
    model_cfg = cfg.get("model", {})
    ctx_cfg = cfg.get("context", {})
    ctx_dims = {k: ctx_cfg.get(k) for k in (
        "target_task_dim", "n_fixed_targets", "fixed_target_dim",
        "target_type_dim", "road_network_dim", "own_info_dim",
    )}
    return TrajSelectionGNN(
        n_modes=int(model_cfg.get("n_modes", 3)),
        fut_len=int(model_cfg.get("fut_len", 10)),
        feat_dim=int(model_cfg.get("feat_dim", 6)),
        ctx_dims=ctx_dims,
        hidden=int(model_cfg.get("hidden_size", 128)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
