"""
gnn1/code/train/loss.py
-----------------------
GNN1 的损失（骨架占位）。

当前只提供最简单的交叉熵；正式使用时可改为 soft label 的 KL divergence。
标签来源：
    - WTA 伪标签：用 LSTM1 产出的 M 条候选 vs GT 未来，选误差最小那条作为
      "正类"。可以离线批量生成，也可以在线跟 LSTM1 联合训练。
      （lstm1/code/train/loss.py 里提供了 compute_wta_best_mode 工具）
    - 规则标签：根据目标作战任务 / 固定目标距离打分再取 argmax
    - 混合：上面两种的加权
TODO: 等数据生成方案定了再补齐具体实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class Gnn1LossConfig:
    use_soft_label: bool = False


class Gnn1Loss(nn.Module):
    def __init__(self, cfg: Optional[Gnn1LossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or Gnn1LossConfig()
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        traj_logits: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            traj_logits: [B, M]
            label:
                - hard:  long [B]
                - soft:  float [B, M]（概率分布，和为 1）
        """
        if self.cfg.use_soft_label:
            # TODO: KL divergence（当前简化为对 soft label 的加权 CE）
            logp = torch.log_softmax(traj_logits.float(), dim=-1)
            return -(label * logp).sum(dim=-1).mean()
        else:
            return self.ce(traj_logits.float(), label.long())
