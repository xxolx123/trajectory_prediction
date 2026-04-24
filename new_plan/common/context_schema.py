"""
common/context_schema.py
------------------------
新方案统一的"外部上下文"输入 Schema。

所有涉及 context 的子网络（gnn1 / gnn2 / constraint_optimizer / fusion）
都 import 这里的 ContextBatch + build_dummy_context。

字段归属：
  GNN1 会用:
    task_type       [B]        long   敌方作战任务（目前只有 0 = 打击）
    type            [B]        long   我方固定目标类型 (0/1/2)
    position        [B, 3]     float  我方固定目标 xyz km（局部 ENU，以 hist 最后一帧为原点）

  ConstraintOptimizer 会用（单主干道 v1）:
    road_points     [B, N_max, 3]  float  路网主干道点的 xyz km
    road_mask       [B, N_max]     bool   有效点掩码

  LSTM2 / GNN2 侧占位:
    own_info        [B, D_own]  float  我方自身信息（占位，待接入）

TODO（等接口定稿）:
  - 多分支路网: road_points 升到 [B, N_branch_max, N_point_max, 3] + branch_mask
  - 部署端 C++ 侧需要把 RoadPointLLH (lon/lat/alt) → 局部 ENU (km) 后再喂进来
  - own_info 的维度 / 字段含义待甲方给出

注意:
  flatten_context_for_mlp 不会把 road_points 拼进去（变长 + mask 不适合 flatten）。
  GNN2 目前用它构造粗粒度 ctx 向量；需要路网信息的网络应直接索引 ctx.road_points / ctx.road_mask。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import torch


DEFAULT_CTX_DIMS: Dict[str, Any] = {
    # GNN1 相关
    "task_type_vocab": 1,   # 目前只有 0 = 打击
    "type_vocab": 3,        # 我方固定目标类型 0..2
    "position_dim": 3,      # xyz km

    # ConstraintOptimizer 相关（单主干道 v1）
    "road_max_points": 128,
    "road_point_dim": 3,

    # LSTM2 / GNN2 占位
    "own_info_dim": 4,
}


@dataclass
class ContextBatch:
    # ---- GNN1 会用 ----
    task_type: torch.Tensor       # [B]        long
    type: torch.Tensor            # [B]        long
    position: torch.Tensor        # [B, 3]     float

    # ---- ConstraintOptimizer 会用 ----
    road_points: torch.Tensor     # [B, N_max, 3]  float
    road_mask: torch.Tensor       # [B, N_max]     bool

    # ---- LSTM2 / GNN2 占位 ----
    own_info: torch.Tensor        # [B, D_own]  float

    def to(self, device: torch.device) -> "ContextBatch":
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            kwargs[f.name] = v
        return ContextBatch(**kwargs)

    def batch_size(self) -> int:
        return int(self.task_type.shape[0])


def build_dummy_context(
    batch_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    ctx_dims: Optional[Dict[str, Any]] = None,
) -> ContextBatch:
    """返回全零 ContextBatch，用于接口尚未接入时的训练 / 冒烟 / ONNX 导出。"""
    if ctx_dims is None:
        ctx_dims = DEFAULT_CTX_DIMS
    if device is None:
        device = torch.device("cpu")

    B = int(batch_size)
    N_max = int(ctx_dims["road_max_points"])
    D_road = int(ctx_dims["road_point_dim"])
    D_pos = int(ctx_dims["position_dim"])
    D_own = int(ctx_dims["own_info_dim"])

    return ContextBatch(
        task_type=torch.zeros(B, device=device, dtype=torch.long),
        type=torch.zeros(B, device=device, dtype=torch.long),
        position=torch.zeros(B, D_pos, device=device, dtype=dtype),
        road_points=torch.zeros(B, N_max, D_road, device=device, dtype=dtype),
        road_mask=torch.zeros(B, N_max, device=device, dtype=torch.bool),
        own_info=torch.zeros(B, D_own, device=device, dtype=dtype),
    )


def build_ctx_dims_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx_cfg = cfg.get("context", {}) if cfg else {}
    out = dict(DEFAULT_CTX_DIMS)
    for k in list(out.keys()):
        if k in ctx_cfg:
            out[k] = ctx_cfg[k]
    return out


def flatten_context_for_mlp(ctx: ContextBatch, ctx_dims: Dict[str, Any]) -> torch.Tensor:
    """
    把粗粒度字段 flatten 成 [B, D_ctx_flat] 向量，给 GNN2 这种 MLP 占位用。

    拼接顺序：
      [task_type_onehot(task_type_vocab),
       type_onehot(type_vocab),
       position(position_dim),
       own_info(own_info_dim)]

    road_points / road_mask 不参与 flatten；需要路网信息的网络应直接索引 ctx.road_*。
    """
    B = ctx.batch_size()
    device = ctx.position.device
    dtype = ctx.position.dtype

    task_vocab = int(ctx_dims["task_type_vocab"])
    type_vocab = int(ctx_dims["type_vocab"])

    task_oh = torch.nn.functional.one_hot(
        ctx.task_type.clamp(min=0, max=task_vocab - 1),
        num_classes=task_vocab,
    ).to(dtype=dtype, device=device)
    type_oh = torch.nn.functional.one_hot(
        ctx.type.clamp(min=0, max=type_vocab - 1),
        num_classes=type_vocab,
    ).to(dtype=dtype, device=device)

    parts = [
        task_oh,                              # [B, task_type_vocab]
        type_oh,                              # [B, type_vocab]
        ctx.position.reshape(B, -1),          # [B, position_dim]
        ctx.own_info.reshape(B, -1),          # [B, own_info_dim]
    ]
    return torch.cat(parts, dim=-1)


def flattened_ctx_dim(ctx_dims: Dict[str, Any]) -> int:
    return (
        int(ctx_dims["task_type_vocab"])
        + int(ctx_dims["type_vocab"])
        + int(ctx_dims["position_dim"])
        + int(ctx_dims["own_info_dim"])
    )
