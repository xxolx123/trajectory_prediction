#!/usr/bin/env python3
"""
intent_model.py
----------------
意图 + 威胁度预测模型（PyTorch 版本）

输入：
  - x: [B, L, 6]，L = window_len（比如 10），6 维特征：
        [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]
  - 通常来自 IntentThreatWindowDataset（见 traj_dataset.py）

输出（dict）：
  - "logits_intent": [B, 4]，意图分类 logits
        0 = ATTACK, 1 = EVASION, 2 = DEFENSE, 3 = RETREAT
  - "threat_raw":   [B, 1]，威胁度的原始回归值（未做 sigmoid/缩放）

网络结构：
  - 先把 [B, L, 6] flatten 成 [B, L*6]
  - 共享 MLP 干路（2~3 层全连接 + ReLU + Dropout）
  - 分成两个 head：
      * intent_head:  Linear -> 4
      * threat_head:  Linear -> 1  （后续在 loss 里做 sigmoid & 0~1/0~100 映射）
"""

from typing import Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentThreatNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        window_len: int = 10,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.1,
        num_intent_classes: int = 4,
    ) -> None:
        """
        Args:
            in_dim:        单步特征维度（默认 6）
            window_len:    窗口长度 L（默认 10）
            hidden_dims:   MLP 每层隐藏单元数
            dropout:       Dropout 概率
            num_intent_classes: 意图类别数（默认 4）
        """
        super().__init__()
        self.in_dim = in_dim
        self.window_len = window_len
        self.num_intent_classes = num_intent_classes

        input_dim = in_dim * window_len

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        # 意图分类 head
        self.intent_head = nn.Linear(prev_dim, num_intent_classes)

        # 威胁度回归 head（输出 raw 标量，后续在 loss 中做 sigmoid + 0~1 映射）
        self.threat_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, 6] 或 [B, L, in_dim]

        Returns:
            dict 包含：
              - "logits_intent": [B, 4]
              - "threat_raw":    [B, 1]
        """
        # 保证形状，[B, L, D]
        if x.ndim != 3:
            raise ValueError(f"Expect x with shape [B, L, D], got {x.shape}")
        B, L, D = x.shape
        if D != self.in_dim:
            raise ValueError(f"in_dim mismatch: model in_dim={self.in_dim}, but x.shape[-1]={D}")
        if L != self.window_len:
            # 不强制报错，有时候你可能想用不同 window_len，只要维度能对应就行
            # 但这里至少 warn 一下
            # print(f"[IntentThreatNet] Warning: x window_len={L} != model.window_len={self.window_len}")
            pass

        # flatten 到 [B, L*D]
        x_flat = x.reshape(B, -1)

        feat = self.backbone(x_flat)          # [B, H]
        logits_intent = self.intent_head(feat)  # [B, 4]
        threat_raw = self.threat_head(feat)     # [B, 1]

        return {
            "logits_intent": logits_intent,
            "threat_raw": threat_raw,
        }


def build_model_from_config(cfg: Dict[str, Any]) -> IntentThreatNet:
    """
    从 config 字典中构建 IntentThreatNet。

    推荐在训练脚本中这样使用：
        from intent_model import build_model_from_config
        cfg = yaml.safe_load(open("config.yaml"))
        model = build_model_from_config(cfg)
    """
    data_cfg = cfg.get("data", {})
    label_cfg = data_cfg.get("intent_threat", {})

    window_len = int(label_cfg.get("window_len", 10))
    in_dim = 6  # 目前我们固定为 [x,y,z,vx,vy,vz]

    model_cfg = cfg.get("intent_model", {})
    hidden_dims = model_cfg.get("hidden_dims", [128, 128])
    dropout = float(model_cfg.get("dropout", 0.1))

    # 你也可以在 config 里调 num_intent_classes，但一般固定为 4
    num_intent_classes = int(model_cfg.get("num_intent_classes", 4))

    model = IntentThreatNet(
        in_dim=in_dim,
        window_len=window_len,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_intent_classes=num_intent_classes,
    )
    return model
