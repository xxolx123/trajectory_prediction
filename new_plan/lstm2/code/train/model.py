"""
lstm2/code/train/model.py
-------------------------
意图（4 类）+ 威胁度（0..1 raw, sigmoid 后 0..1）预测模型。

输入接口（BREAKING；旧 LSTM 版是 forward(hist, fut)）：
    forward(
        fut_refined: [B, Tout, 6]  归一化后；scaler 在 dataset 里已经吃过了
        position:    [B, 3]        我方目标点（物理 km）；本模型用 dataset 里
                                   也已经做过工程化（11 维），这里 forward 接收
                                   的是已经被 dataset 打成 11 维 + 归一化后的
                                   张量。但为了让 --smoke / 推理代码不必依赖
                                   dataset，我们也支持"传 6 维 fut + position"
                                   的旧形态：内部自动做 _engineer_features
                                   并跳过归一化。
    )

输出：
    {
        "logits_intent": [B, num_intent_classes],
        "threat_raw":    [B, 1],
    }

特征工程（per-step 11 维）：
    [x, y, z, vx, vy, vz,
     x - tx, y - ty, z - tz,        # Δ to target
     ||Δ||, ||v||]

config.model.type ∈ {"transformer", "bilstm", "lstm"}：
    transformer (默认): 小型 Transformer Encoder
    bilstm           : 2 层双向 LSTM
    lstm             : 2 层单向 LSTM（兼容旧版）
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn


# ============================================================
# 工程化特征：6 → 11
# ============================================================

def engineer_features(
    fut: torch.Tensor,            # [B, T, 6]
    position: torch.Tensor,       # [B, 3]
) -> torch.Tensor:
    """
    把 (fut [B,T,6], position [B,3]) 扩展成 [B,T,11]：
        [x, y, z, vx, vy, vz, dx, dy, dz, dist, speed]

    其中 (dx, dy, dz) = pos_t - position，dist = ||(dx,dy,dz)||，
    speed = ||(vx,vy,vz)||。

    注意：本函数假设 fut 是"物理空间"（未归一化）。
    在 dataset 里，特征工程是先做、再统一归一化的。
    """
    if fut.ndim != 3 or fut.shape[-1] != 6:
        raise ValueError(f"fut 形状应为 [B,T,6]，实际 {tuple(fut.shape)}")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position 形状应为 [B,3]，实际 {tuple(position.shape)}")

    pos = fut[..., 0:3]                                # [B, T, 3]
    vel = fut[..., 3:6]                                # [B, T, 3]
    delta = pos - position.unsqueeze(1)                # [B, T, 3]
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)   # [B, T, 1]
    speed = torch.linalg.norm(vel, dim=-1, keepdim=True)    # [B, T, 1]

    return torch.cat([fut, delta, dist, speed], dim=-1)     # [B, T, 11]


def _maybe_make_features(
    fut_or_feat: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """
    forward 入口的小适配器：
      - 若传入 [B,T,6]（原始 fut）：内部做 engineer_features → [B,T,11]
      - 若传入 [B,T,11]（dataset 已经工程化 + 归一化好的）：原样返回

    这样模型既能在训练流程里被 dataset 直接喂 11 维归一化输入，也能在
    --smoke / 部署侧用 6 维原始 fut 直接 forward，互不冲突。
    """
    if fut_or_feat.ndim != 3:
        raise ValueError(
            f"fut_refined 期望 3D，实际 {tuple(fut_or_feat.shape)}"
        )
    F = int(fut_or_feat.shape[-1])
    if F == 6:
        return engineer_features(fut_or_feat, position)
    if F == 11:
        return fut_or_feat
    raise ValueError(
        f"fut_refined 最后一维应是 6（原始）或 11（已工程化），实际 {F}"
    )


# ============================================================
# Transformer Encoder（默认）
# ============================================================

class IntentTransformer(nn.Module):
    """
    小型 Transformer Encoder：
      Linear(11 → d_model) + learnable PE → TransformerEncoder × num_layers
      → mean⊕max pool → intent_head, threat_head
    """

    def __init__(
        self,
        input_size: int = 11,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_intent_classes: int = 4,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.d_model = int(d_model)
        self.num_intent_classes = int(num_intent_classes)
        self.max_seq_len = int(max_seq_len)

        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(self.max_seq_len, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_layers)
        )

        self.intent_head = nn.Linear(self.d_model * 2, self.num_intent_classes)
        self.threat_head = nn.Linear(self.d_model * 2, 1)

    def forward(
        self,
        fut_refined: torch.Tensor,
        position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feat = _maybe_make_features(fut_refined, position)   # [B, T, 11]
        B, T, _ = feat.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"序列长度 T={T} 超出 max_seq_len={self.max_seq_len}"
            )

        x = self.input_proj(feat)                            # [B, T, d]
        x = x + self.pos_embed[:T].unsqueeze(0)              # 广播加位置嵌入
        h = self.encoder(x)                                  # [B, T, d]

        h_mean = h.mean(dim=1)                               # [B, d]
        h_max = h.max(dim=1).values                          # [B, d]
        pooled = torch.cat([h_mean, h_max], dim=-1)          # [B, 2d]

        return {
            "logits_intent": self.intent_head(pooled),       # [B, n_cls]
            "threat_raw":    self.threat_head(pooled),       # [B, 1]
        }


# ============================================================
# Bi-LSTM（对照）
# ============================================================

class IntentBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_intent_classes: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_intent_classes = int(num_intent_classes)

        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        # 双向：每个方向 hidden，共 2 * hidden；mean+max pool 再翻一倍
        feat_dim = self.hidden_size * 2 * 2
        self.intent_head = nn.Linear(feat_dim, self.num_intent_classes)
        self.threat_head = nn.Linear(feat_dim, 1)

    def forward(
        self,
        fut_refined: torch.Tensor,
        position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feat = _maybe_make_features(fut_refined, position)   # [B, T, 11]
        out, _ = self.lstm(feat)                             # [B, T, 2H]

        h_mean = out.mean(dim=1)
        h_max = out.max(dim=1).values
        pooled = torch.cat([h_mean, h_max], dim=-1)          # [B, 4H]

        return {
            "logits_intent": self.intent_head(pooled),
            "threat_raw":    self.threat_head(pooled),
        }


# ============================================================
# 单向 LSTM（兼容旧版；接口仍按 forward(fut, position) 走 11 维）
# ============================================================

class IntentLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_intent_classes: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_intent_classes = int(num_intent_classes)

        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout,
        )
        self.intent_head = nn.Linear(self.hidden_size, self.num_intent_classes)
        self.threat_head = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        fut_refined: torch.Tensor,
        position: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feat = _maybe_make_features(fut_refined, position)   # [B, T, 11]
        _, (h_n, _) = self.lstm(feat)
        last_h = h_n[-1]                                     # [B, hidden]
        return {
            "logits_intent": self.intent_head(last_h),
            "threat_raw":    self.threat_head(last_h),
        }


# ============================================================
# 工厂
# ============================================================

def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    按 cfg.model.type 构造模型。所有模型对外接口统一：
        forward(fut_refined: [B,T,6 or 11], position: [B,3])
        → {"logits_intent": [B,n_cls], "threat_raw": [B,1]}
    """
    m = cfg.get("model", {}) or {}
    model_type = str(m.get("type", "transformer")).lower()

    input_size = int(m.get("input_size", 11))
    num_intent_classes = int(m.get("num_intent_classes", 4))

    if model_type == "transformer":
        return IntentTransformer(
            input_size=input_size,
            d_model=int(m.get("d_model", 128)),
            nhead=int(m.get("nhead", 4)),
            num_layers=int(m.get("num_layers", 2)),
            ffn_dim=int(m.get("ffn_dim", 256)),
            dropout=float(m.get("dropout", 0.1)),
            num_intent_classes=num_intent_classes,
            max_seq_len=int(m.get("max_seq_len", 32)),
        )

    if model_type == "bilstm":
        return IntentBiLSTM(
            input_size=input_size,
            hidden_size=int(m.get("hidden_size", 128)),
            num_layers=int(m.get("num_layers", 2)),
            dropout=float(m.get("dropout", 0.1)),
            num_intent_classes=num_intent_classes,
        )

    if model_type == "lstm":
        return IntentLSTM(
            input_size=input_size,
            hidden_size=int(m.get("hidden_size", 128)),
            num_layers=int(m.get("num_layers", 2)),
            dropout=float(m.get("dropout", 0.0)),
            num_intent_classes=num_intent_classes,
        )

    raise ValueError(
        f"未知的 model.type='{model_type}'；可选 transformer/bilstm/lstm"
    )


__all__ = [
    "engineer_features",
    "IntentTransformer",
    "IntentBiLSTM",
    "IntentLSTM",
    "build_model_from_config",
]


# =============================================================================
# 备选：手写 attention 版本的 IntentTransformer（默认注释掉，保留代码备查）
# =============================================================================
#
# 用途
# ----
# 当部署目标只支持 ONNX opset ≤ 13 时（老版 ORT / 特定推理引擎），
# nn.TransformerEncoder 内部的 `scaled_dot_product_attention` 算子
# 要 opset ≥ 14，无法导出。
#
# 这个手写版本用 `Q · K^T → softmax → · V` 显式表达 attention，
# 仅用 matmul / softmax / layer_norm / linear / gelu 等基础算子，
# 任何 opset（甚至 9）都能导出。
#
# 关键设计：state_dict 的 key 与 nn.TransformerEncoder 完全一致，
# 所以用现有 IntentTransformer 训练好的 ckpt 可以直接 load 进
# IntentTransformerManual，不需要重训。
#
# 数学等价性：pre-LN + 单头/多头 SDPA 数学定义完全一致，
# 输出应当 bit-for-bit（或受浮点累积顺序影响在 1e-6 以内）一致。
#
# 使用方法（如果某天需要解开）
# --------------------------
# 1. 把下面整段（从 `# import math` 起到 `# IntentTransformerManual end`）
#    每行最前面的 `# ` 删掉
# 2. 在 build_model_from_config 里把 `model_type == "transformer"` 分支换成
#    `IntentTransformerManual(...)`（参数同 IntentTransformer）
# 3. 训练好的 ckpt 直接 load：
#        model = IntentTransformerManual(...)
#        model.load_state_dict(torch.load("best_intent_*.pt"))
# 4. fusion/code/export_onnx.py 的 --opset 可以改回 11/13
#
# -----------------------------------------------------------------------------
# import math
# from torch.nn import functional as F
#
#
# class _ManualMultiheadAttention(nn.Module):
#     """
#     与 nn.MultiheadAttention 数值等价、参数布局完全一致的手写版本。
#
#     state_dict keys（严格匹配 nn.MultiheadAttention）：
#         in_proj_weight     [3*d_model, d_model]
#         in_proj_bias       [3*d_model]
#         out_proj.weight    [d_model, d_model]
#         out_proj.bias      [d_model]
#     """
#
#     def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
#         super().__init__()
#         if d_model % nhead != 0:
#             raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
#         self.d_model = int(d_model)
#         self.nhead = int(nhead)
#         self.head_dim = self.d_model // self.nhead
#         self.dropout_p = float(dropout)
#
#         # 与 nn.MultiheadAttention 完全相同的参数布局
#         self.in_proj_weight = nn.Parameter(torch.empty(3 * self.d_model, self.d_model))
#         self.in_proj_bias = nn.Parameter(torch.empty(3 * self.d_model))
#         self.out_proj = nn.Linear(self.d_model, self.d_model)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self) -> None:
#         nn.init.xavier_uniform_(self.in_proj_weight)
#         nn.init.zeros_(self.in_proj_bias)
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         nn.init.zeros_(self.out_proj.bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, D]
#         B, T, D = x.shape
#         qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)        # [B, T, 3D]
#         q, k, v = qkv.chunk(3, dim=-1)                                   # 3 x [B, T, D]
#
#         def split_heads(t: torch.Tensor) -> torch.Tensor:
#             return t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, h, T, hd]
#
#         q, k, v = split_heads(q), split_heads(k), split_heads(v)
#         scale = 1.0 / math.sqrt(self.head_dim)
#         attn = torch.matmul(q, k.transpose(-2, -1)) * scale              # [B, h, T, T]
#         attn = F.softmax(attn, dim=-1)
#         if self.dropout_p > 0.0 and self.training:
#             attn = F.dropout(attn, p=self.dropout_p)
#         out = torch.matmul(attn, v)                                      # [B, h, T, hd]
#         out = out.transpose(1, 2).contiguous().view(B, T, D)             # [B, T, D]
#         return self.out_proj(out)
#
#
# class _ManualEncoderLayer(nn.Module):
#     """
#     pre-LN + GELU FFN，结构与 nn.TransformerEncoderLayer(norm_first=True) 一致。
#
#     state_dict keys：
#         self_attn.{in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias}
#         linear1.{weight, bias}        FFN 第一层（d_model → ffn_dim）
#         linear2.{weight, bias}        FFN 第二层（ffn_dim → d_model）
#         norm1.{weight, bias}          attention 前的 LN
#         norm2.{weight, bias}          FFN 前的 LN
#     """
#
#     def __init__(
#         self,
#         d_model: int,
#         nhead: int,
#         dim_feedforward: int,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.self_attn = _ManualMultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.dropout1(self.self_attn(self.norm1(x)))
#         x = x + self.dropout2(
#             self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
#         )
#         return x
#
#
# class _ManualTransformerEncoder(nn.Module):
#     """state_dict keys: layers.{i}.<EncoderLayer keys>"""
#     def __init__(self, layer_factory, num_layers: int) -> None:
#         super().__init__()
#         self.layers = nn.ModuleList([layer_factory() for _ in range(num_layers)])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#
# class IntentTransformerManual(nn.Module):
#     """
#     与 IntentTransformer 数值等价、ckpt 兼容的手写 attention 版本。
#     可导出到任意 ONNX opset（含 11/13）。
#
#     state_dict keys 与 IntentTransformer 完全一致：
#         input_proj.{weight, bias}
#         pos_embed
#         encoder.layers.{i}.<...>
#         intent_head.{weight, bias}
#         threat_head.{weight, bias}
#     """
#
#     def __init__(
#         self,
#         input_size: int = 11,
#         d_model: int = 128,
#         nhead: int = 4,
#         num_layers: int = 2,
#         ffn_dim: int = 256,
#         dropout: float = 0.1,
#         num_intent_classes: int = 4,
#         max_seq_len: int = 32,
#     ) -> None:
#         super().__init__()
#         self.input_size = int(input_size)
#         self.d_model = int(d_model)
#         self.num_intent_classes = int(num_intent_classes)
#         self.max_seq_len = int(max_seq_len)
#
#         self.input_proj = nn.Linear(self.input_size, self.d_model)
#         self.pos_embed = nn.Parameter(torch.zeros(self.max_seq_len, self.d_model))
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#
#         def _layer_factory():
#             return _ManualEncoderLayer(
#                 d_model=self.d_model,
#                 nhead=int(nhead),
#                 dim_feedforward=int(ffn_dim),
#                 dropout=float(dropout),
#             )
#
#         self.encoder = _ManualTransformerEncoder(_layer_factory, int(num_layers))
#
#         self.intent_head = nn.Linear(self.d_model * 2, self.num_intent_classes)
#         self.threat_head = nn.Linear(self.d_model * 2, 1)
#
#     def forward(
#         self,
#         fut_refined: torch.Tensor,
#         position: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         feat = _maybe_make_features(fut_refined, position)               # [B, T, 11]
#         B, T, _ = feat.shape
#         if T > self.max_seq_len:
#             raise ValueError(f"序列长度 T={T} 超出 max_seq_len={self.max_seq_len}")
#
#         x = self.input_proj(feat)                                        # [B, T, d]
#         x = x + self.pos_embed[:T].unsqueeze(0)
#         h = self.encoder(x)                                              # [B, T, d]
#
#         h_mean = h.mean(dim=1)
#         h_max = h.max(dim=1).values
#         pooled = torch.cat([h_mean, h_max], dim=-1)                      # [B, 2d]
#
#         return {
#             "logits_intent": self.intent_head(pooled),
#             "threat_raw":    self.threat_head(pooled),
#         }
# # IntentTransformerManual end
# =============================================================================
