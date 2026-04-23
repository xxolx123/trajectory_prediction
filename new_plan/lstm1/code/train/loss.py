"""
lstm1/code/train/loss.py
------------------------
多模态轨迹预测损失（无分类分支）。

支持三种 mode 选择策略（通过 TrajLossConfig.mode_selection 切换）：

  "hard"  — 经典 Winner-Takes-All：
      L_reg = mean_b min_m MSE[b, m]
      只有最佳 mode 的参数收到梯度。容易 mode collapse。

  "soft"  — Soft-WTA（推荐起点）：
      weight[b, m] = softmax(-MSE[b, m] / T, dim=modes)
      L_reg = mean_b sum_m weight[b, m] * MSE[b, m]
      每个 mode 都拿到加权梯度，最佳 mode 得分最多。
      温度 T 越小越接近 hard WTA，越大越接近均匀平均。

  "topk"  — 只让前 k 小的 mode 拿梯度：
      L_reg = mean_b mean_{top-k smallest m} MSE[b, m]

另外可以打开"多样性正则"来强行让 M 条 mode 散开（缓解即使用了 soft-WTA
也会在单峰目标下发生的 "uniform collapse"）：

  L_div = mean_b mean_{i<j} max(0, margin - ||pred_i - pred_j||²_mean)
  L_total = L_reg + diversity_weight * L_div

  margin 在"逐步逐维 MSE"单位下（和 L_reg 同量纲），设 1.0 ≈ 让两 mode
  平均有 1 std 的差距。weight 默认 0（关闭）。

统计：winner 始终按"硬 argmin"定义，components 里带 winner_counts，
方便追踪 mode 利用率。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class TrajLossConfig:
    # ---- WTA 形式 ----
    mode_selection: str = "soft"   # "hard" / "soft" / "topk"
    soft_temperature: float = 1.0  # soft-WTA 用
    top_k: int = 2                 # topk 用

    # ---- Diversity 正则 ----
    diversity_weight: float = 0.0  # 0 = 关闭；常用 0.05 ~ 0.3
    diversity_margin: float = 1.0  # mode 之间的目标最小 MSE 距离

    # ---- 返回值 ----
    return_components: bool = False


class TrajLoss(nn.Module):
    """
    输入:
        pred_trajs: [B, M, T, D]
        gt_future : [B, T, D]
    返回:
        total_loss                          （若 return_components=False）
        或 (total_loss, components: dict)   （若 return_components=True）

    components 里包含：
        L_total       : 标量 tensor
        L_reg         : WTA 回归项（不含 diversity）
        L_div         : diversity 项（已乘权重）
        div_raw       : 未乘权重的 diversity 原值（便于调 margin）
        best_mode_idx : [B] long，每个样本硬 argmin
        winner_counts : [M] long，本 batch 内每个 mode 成为 winner 的次数
    """

    def __init__(self, cfg: Optional[TrajLossConfig] = None):
        super().__init__()
        self.cfg = cfg or TrajLossConfig()
        m = self.cfg.mode_selection.lower()
        if m not in ("hard", "soft", "topk"):
            raise ValueError(f"mode_selection 只能是 hard/soft/topk，收到 {m!r}")
        self.cfg.mode_selection = m
        if self.cfg.soft_temperature <= 0:
            raise ValueError("soft_temperature 必须 > 0")
        if self.cfg.top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        if self.cfg.diversity_weight < 0:
            raise ValueError("diversity_weight 必须 >= 0")
        if self.cfg.diversity_margin < 0:
            raise ValueError("diversity_margin 必须 >= 0")

    # -------- 工具方法：两两 mode 之间的"平均逐元 MSE"距离 --------
    @staticmethod
    def _pairwise_mse_dist(preds_flat: torch.Tensor) -> torch.Tensor:
        """
        preds_flat: [B, M, L]   L = T*D
        返回: [B, M, M]，每个元素是 mean((preds_i - preds_j)^2)
        """
        # [B, M, 1, L] - [B, 1, M, L]
        diff = preds_flat.unsqueeze(2) - preds_flat.unsqueeze(1)
        dist = (diff ** 2).mean(dim=-1)  # [B, M, M]
        return dist

    def forward(
        self,
        pred_trajs: torch.Tensor,   # [B, M, T, D]
        gt_future: torch.Tensor,    # [B, T, D]
    ):
        pred_trajs = pred_trajs.float()
        gt_future = gt_future.float()

        B, M, T, D = pred_trajs.shape
        device = pred_trajs.device

        # ============= 1) 回归项 L_reg =============
        diff = pred_trajs - gt_future.unsqueeze(1)
        mse_per_mode = (diff ** 2).mean(dim=(-1, -2))  # [B, M]

        if self.cfg.mode_selection == "hard":
            best_mode_idx = torch.argmin(mse_per_mode, dim=1)  # [B]
            L_reg = mse_per_mode[
                torch.arange(B, device=device), best_mode_idx
            ].mean()

        elif self.cfg.mode_selection == "soft":
            T_soft = float(self.cfg.soft_temperature)
            weights = torch.softmax(-mse_per_mode / T_soft, dim=1)  # [B, M]
            weights_d = weights.detach()
            L_reg = (weights_d * mse_per_mode).sum(dim=1).mean()

        else:  # topk
            k = min(self.cfg.top_k, M)
            topk_vals, _ = torch.topk(mse_per_mode, k=k, dim=1, largest=False)
            L_reg = topk_vals.mean()

        # ============= 2) Diversity 正则 L_div =============
        if self.cfg.diversity_weight > 0 and M >= 2:
            preds_flat = pred_trajs.reshape(B, M, -1)           # [B, M, T*D]
            pair_dist = self._pairwise_mse_dist(preds_flat)     # [B, M, M]

            # 只取严格上三角（i < j），避开 i==j 的 0 和下三角重复
            iu = torch.triu_indices(M, M, offset=1, device=device)
            upper = pair_dist[:, iu[0], iu[1]]                  # [B, P] P=M*(M-1)/2

            margin = float(self.cfg.diversity_margin)
            hinge = torch.clamp(margin - upper, min=0.0)        # [B, P]
            L_div_raw = hinge.mean()                            # 标量
            L_div = self.cfg.diversity_weight * L_div_raw
        else:
            L_div_raw = torch.zeros((), device=device)
            L_div = torch.zeros((), device=device)

        # ============= 3) 合并 =============
        L_total = L_reg + L_div

        if not self.cfg.return_components:
            return L_total

        with torch.no_grad():
            best_mode_idx = torch.argmin(mse_per_mode, dim=1)
            winner_counts = torch.bincount(best_mode_idx, minlength=M)

        components: Dict[str, torch.Tensor] = {
            "L_total": L_total.detach(),
            "L_reg": L_reg.detach(),
            "L_div": L_div.detach(),
            "div_raw": L_div_raw.detach(),
            "best_mode_idx": best_mode_idx,
            "winner_counts": winner_counts,
        }
        return L_total, components


def compute_wta_best_mode(
    pred_trajs: torch.Tensor,
    gt_future: torch.Tensor,
) -> torch.Tensor:
    """
    给外部（比如 GNN1 的 trainer）复用的工具：
    输入同 TrajLoss.forward，返回 [B] 的 best_mode_idx（硬 argmin）。
    """
    with torch.no_grad():
        diff = pred_trajs.float() - gt_future.float().unsqueeze(1)
        mse_per_mode = (diff ** 2).mean(dim=(-1, -2))
        return torch.argmin(mse_per_mode, dim=1)


def build_loss_from_config(cfg: Dict) -> TrajLoss:
    """从顶层 config 读取 loss: 段，构造 TrajLoss。"""
    loss_cfg = (cfg or {}).get("loss", {}) or {}
    return TrajLoss(
        TrajLossConfig(
            mode_selection=str(loss_cfg.get("mode_selection", "soft")).lower(),
            soft_temperature=float(loss_cfg.get("soft_temperature", 1.0)),
            top_k=int(loss_cfg.get("top_k", 2)),
            diversity_weight=float(loss_cfg.get("diversity_weight", 0.0)),
            diversity_margin=float(loss_cfg.get("diversity_margin", 1.0)),
            return_components=True,  # trainer 里总是需要组件
        )
    )
