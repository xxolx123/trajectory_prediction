"""
gnn1/code/train/model.py
------------------------
Gnn1Selector：Attention-based 候选轨迹打分器。

输入 batch (来自 gnn1/code/data/dataset.py):
    cand_trajs:  [B, M, T, D]   LSTM1 候选（归一化 + delta 空间）
    task_type:   [B]   long     敌方作战任务（目前只有 0 = 打击）
    type:        [B]   long     我方固定目标类型（0/1/2）
    position:    [B, 3]         我方固定目标 xyz（km，局部 ENU）

输出:
    {
      "logits":    [B, M],      # 训练 CE 用
      "probs":     [B, M],      # softmax over M（可选用来诊断）
      "top_idx":   [B, K],      # top-K 候选在 0..M-1 里的索引（按概率降序）
      "top_probs": [B, K],      # 对 top-K 概率重归一化，K 条和 = 1
    }

结构:
    1) 类别 embedding + position MLP → concat → MLP(ctx) → [B, d_emb]
    2) 候选编码：LSTM 读每条候选 → 末态 → [B, M, d_emb]
    3) Cross-Attention: Q = ctx.unsqueeze(1), K = V = cand_emb
    4) per-candidate score: concat(cand_emb, attended_ctx) → MLP → 1 logit
    5) topk(probs, k=top_k) + 重归一化 → top_idx / top_probs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


# 默认容量：防御性使用，让 config 缺字段时也能跑
DEFAULT_TASK_TYPE_VOCAB = 1     # 目前只有 0 = 打击
DEFAULT_TYPE_VOCAB = 3          # 我方固定目标类型 0..2


@dataclass
class Gnn1Config:
    n_modes: int = 5
    fut_len: int = 10
    feat_dim: int = 6

    d_cat: int = 16
    d_emb: int = 64
    n_heads: int = 4
    dropout: float = 0.1

    # top-K 输出（由 forward 内部 topk + 重归一化产出）
    top_k: int = 3

    # 词汇表大小
    task_type_vocab: int = DEFAULT_TASK_TYPE_VOCAB
    type_vocab: int = DEFAULT_TYPE_VOCAB

    def __post_init__(self) -> None:
        if not (1 <= self.top_k <= self.n_modes):
            raise ValueError(
                f"top_k ({self.top_k}) 必须在 [1, n_modes={self.n_modes}]"
            )


class _MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0, last_act: bool = False) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or last_act:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gnn1Selector(nn.Module):
    """Attention-based selector，给 M 条候选打分。"""

    def __init__(self, cfg: Gnn1Config) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- 上下文侧 ----
        self.task_emb = nn.Embedding(cfg.task_type_vocab, cfg.d_cat)
        self.type_emb = nn.Embedding(cfg.type_vocab, cfg.d_cat)
        self.pos_mlp = _MLP([3, cfg.d_cat], dropout=cfg.dropout, last_act=True)

        # 3 段 concat：task_emb + type_emb + pos_vec
        self.ctx_mlp = _MLP([3 * cfg.d_cat, cfg.d_emb, cfg.d_emb],
                            dropout=cfg.dropout, last_act=False)

        # ---- 候选轨迹编码 ----
        self.cand_encoder = nn.LSTM(
            input_size=cfg.feat_dim,
            hidden_size=cfg.d_emb,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # ---- Cross-attention: Q=ctx, K=V=candidates ----
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_emb,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(cfg.d_emb)

        # ---- Scoring head: 每个候选独立得一个分数 ----
        self.score_head = _MLP([2 * cfg.d_emb, cfg.d_emb, 1],
                               dropout=cfg.dropout, last_act=False)

    # ---------- forward ----------

    def _encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """batch 中各类别 long、position [B,3] → ctx [B, d_emb]"""
        t_task = self.task_emb(batch["task_type"])      # [B, d_cat]
        t_type = self.type_emb(batch["type"])           # [B, d_cat]
        p_vec = self.pos_mlp(batch["position"])         # [B, d_cat]
        ctx = torch.cat([t_task, t_type, p_vec], dim=-1)  # [B, 3*d_cat]
        return self.ctx_mlp(ctx)                        # [B, d_emb]

    def _encode_candidates(self, cand_trajs: torch.Tensor) -> torch.Tensor:
        """cand_trajs [B, M, T, D] → [B, M, d_emb]"""
        B, M, T, D = cand_trajs.shape
        flat = cand_trajs.reshape(B * M, T, D)
        _, (h_n, _) = self.cand_encoder(flat)
        # h_n: [num_layers, B*M, d_emb] → 取最后一层
        cand_emb = h_n[-1].reshape(B, M, -1)
        return cand_emb

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cand_trajs = batch["cand_trajs"].float()                # [B, M, T, D]
        B, M, T, D = cand_trajs.shape
        cfg = self.cfg
        assert M == cfg.n_modes and T == cfg.fut_len and D == cfg.feat_dim, (
            f"cand_trajs 形状 [B,M,T,D]=[{B},{M},{T},{D}] 与 cfg "
            f"[M={cfg.n_modes},T={cfg.fut_len},D={cfg.feat_dim}] 不一致"
        )

        ctx = self._encode_context(batch)                       # [B, d_emb]
        cand_emb = self._encode_candidates(cand_trajs)          # [B, M, d_emb]

        # Cross attention
        ctx_q = ctx.unsqueeze(1)                                # [B, 1, d_emb]
        attn_out, _ = self.attn(ctx_q, cand_emb, cand_emb)      # [B, 1, d_emb]
        attn_out = self.attn_ln(attn_out + ctx_q)               # residual + LN

        # 打分
        ctx_expand = attn_out.expand(-1, M, -1)                 # [B, M, d_emb]
        feat = torch.cat([cand_emb, ctx_expand], dim=-1)        # [B, M, 2*d_emb]
        logits = self.score_head(feat).squeeze(-1)              # [B, M]
        probs = torch.softmax(logits, dim=-1)                    # [B, M]

        # top-K + 重归一化；部署侧（fusion）直接用这两个字段
        K = self.cfg.top_k
        top_p_raw, top_idx = torch.topk(probs, k=K, dim=-1)      # [B, K]
        top_probs = top_p_raw / top_p_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {
            "logits":    logits,
            "probs":     probs,
            "top_idx":   top_idx,
            "top_probs": top_probs,
        }


# ------------------------------------------------------------
# 构造工具
# ------------------------------------------------------------

def build_model_from_config(cfg: Dict[str, Any]) -> Gnn1Selector:
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    # type_vocab：用 config 里 type_range 的上界 + 1（我方固定目标类型 0..2）
    type_hi = int((data_cfg.get("type_range") or [0, 2])[1])

    gcfg = Gnn1Config(
        n_modes=int(model_cfg.get("n_modes", 5)),
        fut_len=int(model_cfg.get("fut_len", 10)),
        feat_dim=int(model_cfg.get("feat_dim", 6)),
        d_cat=int(model_cfg.get("d_cat", 16)),
        d_emb=int(model_cfg.get("d_emb", 64)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        top_k=int(model_cfg.get("top_k", 3)),
        task_type_vocab=DEFAULT_TASK_TYPE_VOCAB,
        type_vocab=type_hi + 1,
    )
    return Gnn1Selector(gcfg)
