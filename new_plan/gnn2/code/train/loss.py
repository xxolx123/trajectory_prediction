"""
gnn2/code/train/loss.py
-----------------------
打击区域损失：MSE(pos) + MSE(radius) + BCE(conf)。

forward 签名（兼容 trainer / smoke 两种用法）::

    loss_fn(outputs, gt)              # gt 是一个 dict，标准用法
    loss_fn(outputs, gt_strike=None)  # smoke 时 gt 缺失返回 0（保留旧占位行为）

其中:
    outputs = {
        "strike_pos":    [B, 3],      km
        "strike_radius": [B, 1],      km，>= radius_min
        "strike_conf":   [B, 1],      [0, 1]
    }
    gt = {
        "pos":    [B, 3]      或 batch["gt_strike_pos"]
        "radius": [B] / [B, 1] 或 batch["gt_strike_radius"]
        "conf":   [B] / [B, 1] 或 batch["gt_strike_conf"]
    }

权重来自 cfg.loss.{w_pos, w_radius, w_conf}。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StrikeLossConfig:
    w_pos: float = 1.0
    w_radius: float = 1.0
    w_conf: float = 0.5
    return_components: bool = False


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """把 [B] / [B, 1] 都规整成 [B, 1]。"""
    if t.ndim == 1:
        return t.unsqueeze(-1)
    if t.ndim == 2 and t.shape[-1] == 1:
        return t
    raise ValueError(f"期望 [B] 或 [B, 1]，实际 {tuple(t.shape)}")


class StrikeLoss(nn.Module):
    def __init__(self, cfg: Optional[StrikeLossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or StrikeLossConfig()

    def forward(
        self,
        outputs: Optional[Dict[str, torch.Tensor]],
        gt_strike: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # 占位行为：smoke 没传 gt 时返回 0（保留旧外部接口的 backward 兼容）
        if outputs is None or gt_strike is None or "strike_pos" not in outputs:
            dev = (
                outputs["strike_pos"].device
                if outputs is not None and "strike_pos" in outputs
                else torch.device("cpu")
            )
            dtype = (
                outputs["strike_pos"].dtype
                if outputs is not None and "strike_pos" in outputs
                else torch.float32
            )
            zero = torch.zeros((), device=dev, dtype=dtype)
            if not self.cfg.return_components:
                return zero
            return zero, {"L": zero, "pos": zero, "radius": zero, "conf": zero}

        pred_pos = outputs["strike_pos"]                        # [B, 3]
        pred_radius = outputs["strike_radius"]                  # [B, 1]
        pred_conf = outputs["strike_conf"]                      # [B, 1]

        # gt：兼容 (a) {"pos":..., "radius":..., "conf":...} 或
        #          (b) {"gt_strike_pos":..., "gt_strike_radius":..., "gt_strike_conf":...}
        gt_pos = gt_strike.get("pos", gt_strike.get("gt_strike_pos"))
        gt_radius = gt_strike.get("radius", gt_strike.get("gt_strike_radius"))
        gt_conf = gt_strike.get("conf", gt_strike.get("gt_strike_conf"))
        if gt_pos is None or gt_radius is None or gt_conf is None:
            raise KeyError(
                "gt_strike 必须含 pos/radius/conf 或 gt_strike_pos/gt_strike_radius/"
                f"gt_strike_conf，实际 keys={list(gt_strike.keys())}"
            )
        gt_radius = _ensure_2d(gt_radius)
        gt_conf = _ensure_2d(gt_conf)

        l_pos = F.mse_loss(pred_pos, gt_pos.to(pred_pos.dtype))
        l_radius = F.mse_loss(pred_radius, gt_radius.to(pred_radius.dtype))
        # BCE 需要 pred ∈ (0, 1)；模型已经过 sigmoid，这里直接用 binary_cross_entropy
        l_conf = F.binary_cross_entropy(pred_conf, gt_conf.to(pred_conf.dtype))

        total = (
            self.cfg.w_pos * l_pos
            + self.cfg.w_radius * l_radius
            + self.cfg.w_conf * l_conf
        )

        if not self.cfg.return_components:
            return total

        components = {
            "L":      total.detach(),
            "pos":    l_pos.detach(),
            "radius": l_radius.detach(),
            "conf":   l_conf.detach(),
        }
        return total, components


def build_loss_from_config(cfg: Dict[str, Any]) -> StrikeLoss:
    loss_cfg = (cfg or {}).get("loss", {}) or {}
    return StrikeLoss(
        StrikeLossConfig(
            w_pos=float(loss_cfg.get("w_pos", 1.0)),
            w_radius=float(loss_cfg.get("w_radius", 1.0)),
            w_conf=float(loss_cfg.get("w_conf", 0.5)),
            return_components=True,
        )
    )
