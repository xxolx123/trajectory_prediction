"""
gnn1/code/train/loss.py
-----------------------
GNN1 的分类损失，两种模式：

  mode="ce"        —— 经典 CrossEntropy，target 是 hard label [B] long
                       可选 label_smoothing 做轻正则
  mode="soft_ce"   —— 软 CE：target 是 soft_label [B, M] 概率分布（和=1）
                       loss = - Σ_k q_k · log p_k
                       等价于 KL(q || p) 差一个常数 H(q)（不影响梯度）
                       在 generate_data.py 里按 softmax(-dist/tau) 产出

无论哪种模式，返回的 components 里都会带一个用 hard label 算的 top1，
方便训练 / 诊断时观察准确率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Gnn1LossConfig:
    mode: str = "ce"                     # "ce" | "soft_ce"
    label_smoothing: float = 0.0         # 仅 ce 模式生效
    return_components: bool = False


class Gnn1Loss(nn.Module):
    """
    forward 签名：
        loss_fn(logits, label, soft_label=None)

    其中:
        logits:     [B, M]  模型输出
        label:      [B]     hard label（ce 模式是监督信号，soft_ce 模式只用来算 top1）
        soft_label: [B, M]  soft label（soft_ce 模式必须提供）

    返回:
        标量 loss（return_components=False）
        或 (loss, components dict)（return_components=True）
    """

    def __init__(self, cfg: Optional[Gnn1LossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or Gnn1LossConfig()
        if self.cfg.mode not in ("ce", "soft_ce"):
            raise ValueError(f"loss_mode 只支持 'ce' / 'soft_ce'，收到 {self.cfg.mode}")
        if not (0.0 <= self.cfg.label_smoothing < 1.0):
            raise ValueError("label_smoothing 必须在 [0, 1)")

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        soft_label: Optional[torch.Tensor] = None,
    ):
        if self.cfg.mode == "soft_ce":
            if soft_label is None:
                raise RuntimeError(
                    "loss_mode=soft_ce 但 batch 里没有 soft_label 字段。"
                    "请确认 generate_data.py 里 data.soft_label_tau > 0 并重新产出数据。"
                )
            # soft CE：-Σ_k q_k · log p_k
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(soft_label.float() * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(
                logits,
                label,
                label_smoothing=self.cfg.label_smoothing,
            )

        if not self.cfg.return_components:
            return loss

        # 诊断：永远用 hard label 算 top1 准确率，方便和旧日志对齐
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            top1 = (pred == label).float().mean()

        components = {
            "L": loss.detach(),
            "top1": top1.detach(),
        }
        return loss, components


def build_loss_from_config(cfg: Dict[str, Any]) -> Gnn1Loss:
    train_cfg = (cfg or {}).get("train", {}) or {}
    return Gnn1Loss(
        Gnn1LossConfig(
            mode=str(train_cfg.get("loss_mode", "ce")),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            return_components=True,  # trainer 里需要 components
        )
    )
