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
      # ===== 对外 / 部署的"主输出"，fusion / deploy / 可视化只看这两个 =====
      "top_idx":   [B, K],      # top-K 候选在 0..M-1 里的索引（按概率降序）
      "top_probs": [B, K],      # 对 top-K 概率重归一化，K 条和 = 1

      # ===== 训练用的中间量；非训练场景请忽略 =====
      "logits":    [B, M],      # 训练 CE / soft_CE 用
      "probs":     [B, M],      # softmax over M（可选诊断）
    }

GNN1 的"对外语义" = 只输出 K=3 条候选 + 重归一化概率；
被 topk 截掉的那 M-K 条候选不会出现在 fusion 输出里，也不应该被可视化。

结构:
    1) 类别 embedding + position MLP → concat → MLP(ctx) → [B, d_emb]
    2) 候选编码：LSTM 读每条候选 → 末态 → [B, M, d_emb]
    3) Cross-Attention: Q = ctx.unsqueeze(1), K = V = cand_emb
    4) per-candidate score: concat(cand_emb, attended_ctx) → MLP → 1 logit
    5) topk(probs, k=top_k) + 重归一化 → top_idx / top_probs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


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

    # 部署 opset ≤ 13 时启用：用 _ManualCrossAttention 替代 nn.MultiheadAttention，
    # 避开 SDPA / aten::unflatten。state_dict key 严格兼容，可直接 load 现有 ckpt。
    manual_attention: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.top_k <= self.n_modes):
            raise ValueError(
                f"top_k ({self.top_k}) 必须在 [1, n_modes={self.n_modes}]"
            )


# ============================================================
# Manual cross-attention（ONNX opset ≤ 13 / mindspore-lite 1.8.1 兼容版）
# ============================================================

class _ManualCrossAttention(nn.Module):
    """
    与 ``nn.MultiheadAttention(batch_first=True)`` 数值等价、参数布局完全一致的
    手写版本。专为 ONNX opset ≤ 13 部署目标设计，避开两类不兼容算子：

      - ``scaled_dot_product_attention``  → opset ≥ 14
      - ``aten::unflatten`` (出现在 ``view([B, T, h, hd])`` 这种"把一维 split 成
         多维"的 reshape 上)  → opset ≥ 13

    state_dict keys（严格匹配 ``nn.MultiheadAttention``）::

        in_proj_weight     [3*embed_dim, embed_dim]
        in_proj_bias       [3*embed_dim]
        out_proj.weight    [embed_dim, embed_dim]
        out_proj.bias      [embed_dim]

    forward 接口与 ``nn.MultiheadAttention`` 一致：
      ``forward(query, key, value) → (out, None)``。
    第二个返回值 ``attn_weights`` 始终为 None（不计算，避免 ONNX 多输出干扰）。
    本项目里 GNN1 的 cross-attention 调用方仅取 [0]，与此兼容。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) 必须能被 num_heads({num_heads}) 整除"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_p = float(dropout)

        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.embed_dim, self.embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim))
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,    # [B, Tq, D]
        key: torch.Tensor,      # [B, Tk, D]
        value: torch.Tensor,    # [B, Tk, D]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        Bq, Tq, D = query.shape
        Bk, Tk, _ = key.shape

        # 三组投影矩阵：与 nn.MultiheadAttention 的 in_proj_weight chunk(3) 一致
        wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, wq, bq)                              # [B, Tq, D]
        k = F.linear(key,   wk, bk)                              # [B, Tk, D]
        v = F.linear(value, wv, bv)                              # [B, Tk, D]

        h = self.num_heads
        hd = self.head_dim
        # 两步 reshape 避免 ONNX 把单步 view([B, T, h, hd]) 识别成 aten::unflatten
        q = q.reshape(Bq * Tq, h, hd).reshape(Bq, Tq, h, hd).transpose(1, 2)
        k = k.reshape(Bk * Tk, h, hd).reshape(Bk, Tk, h, hd).transpose(1, 2)
        v = v.reshape(Bk * Tk, h, hd).reshape(Bk, Tk, h, hd).transpose(1, 2)
        # q/k/v: [B, h, T*, hd]

        scale = 1.0 / math.sqrt(hd)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale       # [B, h, Tq, Tk]
        attn = F.softmax(attn, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)                               # [B, h, Tq, hd]

        # 合并 head 回 D 维：同样拆两步 reshape 避免 unflatten 形态
        out = out.transpose(1, 2).contiguous()                    # [B, Tq, h, hd]
        out = out.reshape(Bq * Tq, D).reshape(Bq, Tq, D)
        return self.out_proj(out), None


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
        # manual_attention=True 时用手写版（state_dict 完全兼容 nn.MultiheadAttention，
        # 现有 ckpt 直接 load），避开 ONNX opset ≥ 14 的 SDPA。
        if cfg.manual_attention:
            self.attn: nn.Module = _ManualCrossAttention(
                embed_dim=cfg.d_emb,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
            )
        else:
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
        manual_attention=bool(model_cfg.get("manual_attention", False)),
    )
    return Gnn1Selector(gcfg)
