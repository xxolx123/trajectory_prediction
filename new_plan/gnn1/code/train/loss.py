"""
gnn1/code/train/loss.py
-----------------------
GNN1 的分类损失：CrossEntropy on [B, M] logits vs [B] label。

可选 label smoothing。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Gnn1LossConfig:
    label_smoothing: float = 0.0
    return_components: bool = False


class Gnn1Loss(nn.Module):
    """
    输入:
        logits: [B, M]
        label : [B] long, 取值 0..M-1
    返回:
        标量 loss（return_components=False）
        或 (loss, components dict)（return_components=True）
    """

    def __init__(self, cfg: Optional[Gnn1LossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or Gnn1LossConfig()
        if not (0.0 <= self.cfg.label_smoothing < 1.0):
            raise ValueError("label_smoothing 必须在 [0, 1)")

    def forward(self, logits: torch.Tensor, label: torch.Tensor):
        loss = F.cross_entropy(
            logits,
            label,
            label_smoothing=self.cfg.label_smoothing,
        )
        if not self.cfg.return_components:
            return loss

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            top1 = (pred == label).float().mean()

        components = {
            "L_ce": loss.detach(),
            "top1": top1.detach(),
        }
        return loss, components


def build_loss_from_config(cfg: Dict[str, Any]) -> Gnn1Loss:
    train_cfg = (cfg or {}).get("train", {}) or {}
    return Gnn1Loss(
        Gnn1LossConfig(
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            return_components=True,  # trainer 里需要 components
        )
    )
