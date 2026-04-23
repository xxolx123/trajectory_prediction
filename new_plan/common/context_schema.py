"""
common/context_schema.py
------------------------
新方案新增的"外部上下文"输入的统一 Schema。

所有涉及 context 的子网络（gnn1 / gnn2 / constraint_optimizer / fusion）
都 import 这里的 ContextBatch + build_dummy_context。

字段速查（全部 TODO: 等甲方/实测接口定稿后替换）：
  target_task    [B, D_task]        作战任务编码
  fixed_targets  [B, N_tgt, D_tgt]  固定目标 (x, y, z, type_id)
  target_type    [B, D_type]        目标类型 one-hot / embedding
  road_network   [B, D_road]        路网嵌入
  own_info       [B, D_own]         我方信息
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import torch


DEFAULT_CTX_DIMS: Dict[str, Any] = {
    "target_task_dim": 8,
    "n_fixed_targets": 4,
    "fixed_target_dim": 4,
    "target_type_dim": 4,
    "road_network_dim": 16,
    "own_info_dim": 4,
}


@dataclass
class ContextBatch:
    target_task: torch.Tensor     # [B, D_task]
    fixed_targets: torch.Tensor   # [B, N_tgt, D_tgt]
    target_type: torch.Tensor     # [B, D_type]
    road_network: torch.Tensor    # [B, D_road]
    own_info: torch.Tensor        # [B, D_own]

    def to(self, device: torch.device) -> "ContextBatch":
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            kwargs[f.name] = v
        return ContextBatch(**kwargs)

    def batch_size(self) -> int:
        return int(self.target_task.shape[0])


def build_dummy_context(
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    ctx_dims: Optional[Dict[str, Any]] = None,
) -> ContextBatch:
    """
    返回全零 ContextBatch，用于接口尚未接入时的训练 / 冒烟。
    TODO: 接入真实数据后，写 build_context_from_*()。
    """
    if ctx_dims is None:
        ctx_dims = DEFAULT_CTX_DIMS
    if device is None:
        device = torch.device("cpu")

    B = int(batch_size)
    D_task = int(ctx_dims["target_task_dim"])
    N_tgt = int(ctx_dims["n_fixed_targets"])
    D_tgt = int(ctx_dims["fixed_target_dim"])
    D_type = int(ctx_dims["target_type_dim"])
    D_road = int(ctx_dims["road_network_dim"])
    D_own = int(ctx_dims["own_info_dim"])

    return ContextBatch(
        target_task=torch.zeros(B, D_task, device=device, dtype=dtype),
        fixed_targets=torch.zeros(B, N_tgt, D_tgt, device=device, dtype=dtype),
        target_type=torch.zeros(B, D_type, device=device, dtype=dtype),
        road_network=torch.zeros(B, D_road, device=device, dtype=dtype),
        own_info=torch.zeros(B, D_own, device=device, dtype=dtype),
    )


def build_ctx_dims_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx_cfg = cfg.get("context", {}) if cfg else {}
    out = dict(DEFAULT_CTX_DIMS)
    for k in list(out.keys()):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def flatten_context_for_mlp(ctx: ContextBatch) -> torch.Tensor:
    """
    占位用工具：把 ContextBatch 里所有字段 flatten + concat 成 [B, D_all] 向量。
    子网络里"MLP 占位阶段"会用；真上 GNN 后可以不用。
    """
    B = ctx.batch_size()
    parts = [
        ctx.target_task.reshape(B, -1),
        ctx.fixed_targets.reshape(B, -1),
        ctx.target_type.reshape(B, -1),
        ctx.road_network.reshape(B, -1),
        ctx.own_info.reshape(B, -1),
    ]
    return torch.cat(parts, dim=-1)


def flattened_ctx_dim(ctx_dims: Dict[str, Any]) -> int:
    return (
        int(ctx_dims["target_task_dim"])
        + int(ctx_dims["n_fixed_targets"]) * int(ctx_dims["fixed_target_dim"])
        + int(ctx_dims["target_type_dim"])
        + int(ctx_dims["road_network_dim"])
        + int(ctx_dims["own_info_dim"])
    )
