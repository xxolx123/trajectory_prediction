#!/usr/bin/env python3
"""
intent_loss.py
---------------
意图 + 威胁度联合损失函数。

总损失：
  L_total = lambda_cls * CrossEntropy(intent) + lambda_reg * MSE(threat)

其中：
  - intent:   4 类分类（0=ATTACK, 1=EVASION, 2=DEFENSE, 3=RETREAT）
  - threat:   0~100 的威胁度整数，先缩放到 0~1 再做 MSE 回归
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentThreatLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        ignore_label: int = -1,
    ) -> None:
        """
        Args:
            lambda_cls: 意图分类损失的权重
            lambda_reg: 威胁度回归损失的权重
            ignore_label: 若 intent_labels 中存在该值，则在分类/回归中忽略对应样本
        """
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.ignore_label = int(ignore_label)

        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        intent_labels: torch.Tensor,
        threat_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            outputs:
                - "logits_intent": [B, 4]
                - "threat_raw":    [B, 1]
            intent_labels: [B]，int64，0..3 或 ignore_label
            threat_scores: [B]，float / int，0..100

        Returns:
            total_loss, cls_loss, reg_loss （都为标量张量）
        """
        logits_intent = outputs["logits_intent"]     # [B, 4]
        threat_raw = outputs["threat_raw"]           # [B, 1]

        if threat_raw.ndim == 2 and threat_raw.shape[1] == 1:
            threat_raw = threat_raw.squeeze(1)       # [B]
        elif threat_raw.ndim != 1:
            raise ValueError(f"threat_raw 形状不合法: {threat_raw.shape}")

        # 有些样本可能被标了 ignore_label，这里统一做 mask
        if intent_labels.dtype != torch.long:
            intent_labels = intent_labels.long()

        mask = intent_labels != self.ignore_label
        if not mask.any():
            # 理论上不会发生，但防止全是 -1
            raise RuntimeError("All intent_labels are ignore_label, no valid sample for loss.")

        logits_intent = logits_intent[mask]
        intent_valid = intent_labels[mask]
        threat_raw_valid = threat_raw[mask]
        threat_scores_valid = threat_scores[mask]

        # 1) 分类损失：CrossEntropy
        cls_loss = self.ce(logits_intent, intent_valid)

        # 2) 威胁度回归：
        #    - label: 缩放到 [0,1]
        #    - pred:  raw -> sigmoid -> [0,1]
        threat_target_norm = torch.clamp(threat_scores_valid / 100.0, 0.0, 1.0)
        threat_pred_norm = torch.sigmoid(threat_raw_valid)

        reg_loss = self.mse(threat_pred_norm, threat_target_norm)

        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss

        return total_loss, cls_loss, reg_loss


def build_loss_from_config(cfg: Dict[str, Any]) -> IntentThreatLoss:
    """
    从 config 字典中构建 IntentThreatLoss。

    config 示例：

    intent_loss:
      lambda_cls: 1.0
      lambda_reg: 1.0
      ignore_label: -1
    """
    loss_cfg = cfg.get("intent_loss", {})
    lambda_cls = float(loss_cfg.get("lambda_cls", 1.0))
    lambda_reg = float(loss_cfg.get("lambda_reg", 1.0))
    ignore_label = int(loss_cfg.get("ignore_label", -1))

    loss_fn = IntentThreatLoss(
        lambda_cls=lambda_cls,
        lambda_reg=lambda_reg,
        ignore_label=ignore_label,
    )
    return loss_fn
