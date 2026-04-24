"""
constraint_optimizer/code/train/module.py
-----------------------------------------
约束优化模块（骨架）。

当前实现：pass_through，直接返回输入。

路网输入归本模块管：
    ctx.road_points  [B, N_max, 3]   km-xyz，主干道折线点
    ctx.road_mask    [B, N_max]      bool，True = 有效点

TODO（二选一，或混合）：
    (a) 算法型：
        - 路网 link 投影：把 trajectory 点拉到最近的 road_points 连线上
          (记得用 ctx.road_mask 过滤 padding 点)
        - 速度/加速度夹取
        - 任务区域边界裁剪
    (b) 可学习型：
        - 用 MLP 出残差 Δtraj，然后 refined = selected + Δtraj
        - 训练 loss：(reconstruction MSE) + (约束违反惩罚)
        - road_points / road_mask 作为 set-encoder 的输入之一
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.context_schema import ContextBatch  # noqa: E402


class ConstraintOptimizer(nn.Module):
    """
    forward(selected_traj, ctx) -> refined_traj
    """

    def __init__(self, enable: bool = True, module_type: str = "pass_through") -> None:
        super().__init__()
        self.enable = bool(enable)
        self.module_type = str(module_type)
        # TODO: 若走可学习路线，在这里注册 nn.Linear / nn.Sequential 等

    def forward(self, selected_traj: torch.Tensor, ctx: ContextBatch) -> torch.Tensor:
        if not self.enable:
            return selected_traj
        if self.module_type == "pass_through":
            # TODO: 将来真正实现时从 ctx.road_points / ctx.road_mask 读主干路网
            _ = ctx
            return selected_traj
        # TODO: 其它类型
        raise NotImplementedError(f"Unknown constraint_optimizer type: {self.module_type}")


def build_module_from_config(cfg: Dict[str, Any]) -> ConstraintOptimizer:
    mod_cfg = cfg.get("module", {})
    return ConstraintOptimizer(
        enable=bool(mod_cfg.get("enable", True)),
        module_type=str(mod_cfg.get("type", "pass_through")),
    )
