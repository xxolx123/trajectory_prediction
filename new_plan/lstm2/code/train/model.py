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
