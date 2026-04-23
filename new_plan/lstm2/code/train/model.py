"""
lstm2/code/train/model.py
-------------------------
IntentLSTM：意图 + 威胁度预测。

接口：
    forward(hist_raw, fut_refined)
        hist_raw:    [B, Tin, D]
        fut_refined: [B, Tout, D]
    return:
        {"logits_intent": [B, num_intent_classes], "threat_raw": [B, 1]}

注意：
  - 输入假设已经"对齐 LSTM2 自己的归一化空间"。是否归一化由 fusion 层决定。
  - 为了保持与旧 IntentThreatNet 接口兼容（fusion 里拼 68 维时要用），
    输出的 key 与旧版保持一致。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class IntentLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 6,
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

        lstm_dropout = dropout if num_layers > 1 else 0.0
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
        hist_raw: torch.Tensor,
        fut_refined: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if hist_raw.ndim != 3 or fut_refined.ndim != 3:
            raise ValueError(
                f"hist_raw/fut_refined 需 [B,T,D]，实际 {hist_raw.shape}/{fut_refined.shape}"
            )
        if hist_raw.shape[0] != fut_refined.shape[0]:
            raise ValueError("hist_raw/fut_refined 的 batch size 不一致")
        if hist_raw.shape[-1] != fut_refined.shape[-1]:
            raise ValueError("hist_raw/fut_refined 的特征维度不一致")

        full_traj = torch.cat([hist_raw, fut_refined], dim=1)  # [B, Tin+Tout, D]

        output, (h_n, c_n) = self.lstm(full_traj)
        last_h = h_n[-1]  # [B, hidden]

        return {
            "logits_intent": self.intent_head(last_h),   # [B, num_intent_classes]
            "threat_raw": self.threat_head(last_h),       # [B, 1]
        }


def build_model_from_config(cfg: Dict[str, Any]) -> IntentLSTM:
    m = cfg.get("model", {})
    return IntentLSTM(
        input_size=int(m.get("input_size", 6)),
        hidden_size=int(m.get("hidden_size", 128)),
        num_layers=int(m.get("num_layers", 2)),
        dropout=float(m.get("dropout", 0.0)),
        num_intent_classes=int(m.get("num_intent_classes", 4)),
    )
