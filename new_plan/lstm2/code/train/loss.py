"""
lstm2/code/train/loss.py
------------------------
意图 + 威胁度联合损失。

与 old_plan/203_prediction_intention_pytorch_v0/code/train/intent_loss.py
接口一致，只是重命名到 lstm2/loss.py。

total = lambda_cls * CrossEntropy(intent) + lambda_reg * MSE(threat)
    - intent 支持 ignore_label=-1
    - threat: sigmoid(raw) vs (threat_score / 100)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn


class IntentThreatLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        ignore_label: int = -1,
    ) -> None:
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
        logits_intent = outputs["logits_intent"]
        threat_raw = outputs["threat_raw"]

        if threat_raw.ndim == 2 and threat_raw.shape[1] == 1:
            threat_raw = threat_raw.squeeze(1)

        if intent_labels.dtype != torch.long:
            intent_labels = intent_labels.long()

        mask = intent_labels != self.ignore_label
        if not mask.any():
            zero = torch.zeros((), device=logits_intent.device, dtype=logits_intent.dtype)
            return zero, zero.clone(), zero.clone()

        logits_intent = logits_intent[mask]
        intent_valid = intent_labels[mask]
        threat_raw_valid = threat_raw[mask]
        threat_scores_valid = threat_scores[mask]

        cls_loss = self.ce(logits_intent, intent_valid)

        threat_target_norm = torch.clamp(threat_scores_valid / 100.0, 0.0, 1.0)
        threat_pred_norm = torch.sigmoid(threat_raw_valid)
        reg_loss = self.mse(threat_pred_norm, threat_target_norm)

        total = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss
        return total, cls_loss, reg_loss


def build_loss_from_config(cfg: Dict[str, Any]) -> IntentThreatLoss:
    c = cfg.get("loss", {})
    return IntentThreatLoss(
        lambda_cls=float(c.get("lambda_cls", 1.0)),
        lambda_reg=float(c.get("lambda_reg", 1.0)),
        ignore_label=int(c.get("ignore_label", -1)),
    )
