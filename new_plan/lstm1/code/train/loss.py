"""
lstm1/code/train/loss.py
------------------------
多模态轨迹预测损失：Winner-Takes-All 回归版。

与 old_plan 相比的差异：
    **去掉分类损失**（不再有 mode_logits，自然也就没有 CE loss）。

损失：
    对每个样本、每个 mode 计算 MSE：
        L_reg[b, m] = mean_{t,d} (pred[b,m,t,d] - gt[b,t,d])^2
    为每个样本选最小的 mode：m* = argmin_m L_reg[b, m]
    L_total = mean_b (L_reg[b, m*])

注意：
    winner 仅用来做回归监督。因为我们不再产出 mode_logits，所以不在这里
    做分类监督；"选一条轨迹"的概率由下游 GNN1 单独出。
    （GNN1 自己的训练脚本里可以**复用**这里的 winner 计算作为标签。）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class TrajLossConfig:
    return_components: bool = False


class TrajLoss(nn.Module):
    """
    输入：
        pred_trajs: [B, M, T, D]
        gt_future:  [B, T, D]
    返回:
        total_loss                          （若 return_components=False）
        或 (total_loss, components: dict)   （若 return_components=True）
    """

    def __init__(self, cfg: Optional[TrajLossConfig] = None):
        super().__init__()
        self.cfg = cfg or TrajLossConfig()

    def forward(
        self,
        pred_trajs: torch.Tensor,   # [B, M, T, D]
        gt_future: torch.Tensor,    # [B, T, D]
    ):
        pred_trajs = pred_trajs.float()
        gt_future = gt_future.float()

        B, M, T, D = pred_trajs.shape

        # per-mode MSE
        diff = pred_trajs - gt_future.unsqueeze(1)
        mse_per_mode = (diff ** 2).mean(dim=(-1, -2))  # [B, M]

        # WTA
        best_mode_idx = torch.argmin(mse_per_mode, dim=1)  # [B]

        reg_loss = mse_per_mode[torch.arange(B, device=pred_trajs.device), best_mode_idx].mean()

        if not self.cfg.return_components:
            return reg_loss

        components: Dict[str, torch.Tensor] = {
            "L_total": reg_loss.detach(),
            "L_reg": reg_loss.detach(),
            "best_mode_idx": best_mode_idx.detach(),
        }
        return reg_loss, components


def compute_wta_best_mode(
    pred_trajs: torch.Tensor,
    gt_future: torch.Tensor,
) -> torch.Tensor:
    """
    给外部（比如 GNN1 的 trainer）复用的工具：
    输入同 TrajLoss.forward，返回 [B] 的 best_mode_idx。
    """
    with torch.no_grad():
        diff = pred_trajs.float() - gt_future.float().unsqueeze(1)
        mse_per_mode = (diff ** 2).mean(dim=(-1, -2))
        return torch.argmin(mse_per_mode, dim=1)
