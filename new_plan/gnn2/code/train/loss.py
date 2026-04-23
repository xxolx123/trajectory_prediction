"""
gnn2/code/train/loss.py
-----------------------
打击区域损失（骨架占位）。

TODO:
    - pos:    MSE(pred_pos, gt_pos)
    - radius: MSE(pred_radius, gt_radius)
    - conf:   BCE(pred_conf, gt_conf)
    或合成"可微分 IoU"这类。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class StrikeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        outputs,
        gt_strike: Optional[dict] = None,
    ) -> torch.Tensor:
        # TODO: 真正实现
        if outputs is None or "strike_pos" not in outputs:
            return torch.zeros(())
        dev = outputs["strike_pos"].device
        dtype = outputs["strike_pos"].dtype
        _ = gt_strike
        return torch.zeros((), device=dev, dtype=dtype)
