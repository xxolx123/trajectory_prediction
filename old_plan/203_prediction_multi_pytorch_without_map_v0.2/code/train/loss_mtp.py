"""
loss_traj.py (PyTorch version, MTP hard assignment)
---------------------------------------------------
多模态轨迹预测损失（硬分配 Winner-Takes-All 版）。

输入：
  pred_trajs:  Tensor, shape [B, M, T, D]
               模型预测的多模态未来轨迹，M 为 mode 数量
  mode_logits: Tensor, shape [B, M]
               每个 mode 的概率 logit（未做 softmax）
  gt_future:   Tensor, shape [B, T, D]
               真实未来轨迹

损失构成：
  1) 回归损失（硬分配）：
       - 对每个样本 b、每个 mode m 计算 MSE：
           L_reg[b, m] = mean_{t,d} (pred_trajs[b,m,t,d] - gt[b,t,d])^2
       - 对每个样本选误差最小的 mode：
           m* = argmin_m L_reg[b, m]
       - 只用 m* 的误差作为该样本的回归 loss，并在 batch 上取平均。

  2) 分类损失：
       - 把 m* 视为“正确类别”，对 mode_logits 做交叉熵：
           L_cls = CrossEntropy(mode_logits[b], target = m*)

  总损失：
       L_total = L_reg_mean + alpha_class * L_cls
"""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn


@dataclass
class TrajLossConfig:
    # 分类 loss 的权重：总损失 = L_reg + alpha_class * L_cls
    alpha_class: float = 0.1
    # 是否返回分解结果，方便以后打印 / 画图
    return_components: bool = False


class TrajLoss(nn.Module):
    def __init__(self, cfg: Optional[TrajLossConfig] = None):
        super().__init__()
        self.cfg = cfg or TrajLossConfig()
        # 分类部分直接用 PyTorch 自带 CrossEntropyLoss
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        pred_trajs: torch.Tensor,   # [B, M, T, D]
        mode_logits: torch.Tensor,  # [B, M]
        gt_future: torch.Tensor,    # [B, T, D]
    ):
        """
        返回：
          - 如果 cfg.return_components=False:
                只返回 total_loss (标量)
          - 如果 cfg.return_components=True :
                返回 (total_loss, {"L_reg":..., "L_cls":..., "L_total":...})
        """
        pred_trajs = pred_trajs.float()
        mode_logits = mode_logits.float()
        gt_future = gt_future.float()

        B, M, T, D = pred_trajs.shape

        # 1) 计算每个样本 / 每个 mode 的 MSE：L_reg_per_mode: [B, M]
        #    diff: [B, M, T, D]
        diff = pred_trajs - gt_future.unsqueeze(1)  # 广播 gt 到每个 mode
        mse_per_mode = (diff ** 2).mean(dim=(-1, -2))  # 在 T、D 上取均值 → [B, M]

        # 2) Winner-Takes-All：为每个样本选误差最小的 mode
        #    best_mode_idx: [B]，每个元素是 [0, M-1] 之间的整型
        best_mode_idx = torch.argmin(mse_per_mode, dim=1)  # [B]

        # 3) 回归损失：只取 winner 的那一列，再在 batch 上取平均
        reg_loss = mse_per_mode[torch.arange(B), best_mode_idx].mean()

        # 4) 分类损失：把 best_mode_idx 当作标签，监督 mode_logits
        cls_loss = self.ce(mode_logits, best_mode_idx)

        # 5) 总损失
        total_loss = reg_loss + self.cfg.alpha_class * cls_loss

        if not self.cfg.return_components:
            return total_loss

        components: Dict[str, torch.Tensor] = {
            "L_total": total_loss.detach(),
            "L_reg": reg_loss.detach(),
            "L_cls": cls_loss.detach(),
        }
        return total_loss, components
