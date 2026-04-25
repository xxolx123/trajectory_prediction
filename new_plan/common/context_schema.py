"""
common/context_schema.py
------------------------
新方案统一的"外部上下文"输入 Schema。

所有涉及 context 的子网络（gnn1 / gnn2 / constraint_optimizer / fusion）
都 import 这里的 ContextBatch + build_dummy_context。

字段归属（与甲方流程图 5 路 + ETA 占位一致）：

  GNN1 会用:
    task_type   [B]                 long   敌方作战任务（目前只有 0 = 打击）
    type        [B]                 long   我方固定目标类型 (0/1/2)
    position    [B, 3]              float  我方固定目标 xyz km（局部 ENU，
                                          以 hist 最后一帧为原点）

  ConstraintOptimizer 会用:
    road_points [B, NB_max, NP_max, 3]  float  每条分支折线点的 xyz km
    road_mask   [B, NB_max, NP_max]     bool   有效点掩码（按点；某条分支整条
                                              无效就把该分支所有 NP 个点全置 False）

  下游公共占位:
    eta         [B]                 long   我方预计到达时间（秒，int64 占位；
                                          flatten_context_for_mlp 里会按 /3600
                                          归一化到"小时"喂 MLP，避免数值过大）

历史备注:
  - v1 schema 里曾有 own_info[B, D_own] 占位字段，含义是"我方自身信息"；现在
    已被语义更明确的 eta 取代（已与业务侧确认）。"我方固定目标"信息由
    type / position 覆盖，不再由 own_info 表达。

部署侧约定:
  - C++ 端拿到 RoadNetwork（lon/lat/alt 度+米）后，需先用 hist 末帧 LLH 作为
    ENU 原点，把每个 RoadPointLLH 转成 km xyz，再填到 road_points / road_mask。
  - eta 由我方任务系统给，单位秒（int64）。

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

    # ConstraintOptimizer 相关（多分支）
    "road_max_branches": 4,   # 每个 batch 最多多少条分支
    "road_max_points": 128,   # 每条分支最多多少个折线点
    "road_point_dim": 3,

    # ETA 占位（我方预计到达时间）
    # ONNX 输入是 int64 秒；flatten_context_for_mlp 里会用 /eta_scale_seconds
    # 把它归一化成"小时"再喂 MLP（数值更稳定）。
    "eta_scale_seconds": 3600,
}


@dataclass
class ContextBatch:
    # ---- GNN1 会用 ----
    task_type: torch.Tensor       # [B]                       long
    type: torch.Tensor            # [B]                       long
    position: torch.Tensor        # [B, 3]                    float

    # ---- ConstraintOptimizer 会用 ----
    road_points: torch.Tensor     # [B, NB_max, NP_max, 3]    float
    road_mask: torch.Tensor       # [B, NB_max, NP_max]       bool

    # ---- 占位：我方预计到达时间（秒） ----
    eta: torch.Tensor             # [B]                       long (int64)

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
    NB_max = int(ctx_dims.get("road_max_branches", 1))
    NP_max = int(ctx_dims["road_max_points"])
    D_road = int(ctx_dims["road_point_dim"])
    D_pos = int(ctx_dims["position_dim"])

    return ContextBatch(
        task_type=torch.zeros(B, device=device, dtype=torch.long),
        type=torch.zeros(B, device=device, dtype=torch.long),
        position=torch.zeros(B, D_pos, device=device, dtype=dtype),
        road_points=torch.zeros(B, NB_max, NP_max, D_road, device=device, dtype=dtype),
        road_mask=torch.zeros(B, NB_max, NP_max, device=device, dtype=torch.bool),
        eta=torch.zeros(B, device=device, dtype=torch.long),
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
       eta_hours(1)]                   # = eta(秒) / eta_scale_seconds

    road_points / road_mask 不参与 flatten；需要路网信息的网络应直接索引 ctx.road_*。
    """
    B = ctx.batch_size()
    device = ctx.position.device
    dtype = ctx.position.dtype

    task_vocab = int(ctx_dims["task_type_vocab"])
    type_vocab = int(ctx_dims["type_vocab"])
    eta_scale = float(ctx_dims.get("eta_scale_seconds", 3600))

    task_oh = torch.nn.functional.one_hot(
        ctx.task_type.clamp(min=0, max=task_vocab - 1),
        num_classes=task_vocab,
    ).to(dtype=dtype, device=device)
    type_oh = torch.nn.functional.one_hot(
        ctx.type.clamp(min=0, max=type_vocab - 1),
        num_classes=type_vocab,
    ).to(dtype=dtype, device=device)

    eta_hours = (ctx.eta.to(dtype=dtype, device=device) / eta_scale).reshape(B, 1)

    parts = [
        task_oh,                              # [B, task_type_vocab]
        type_oh,                              # [B, type_vocab]
        ctx.position.reshape(B, -1),          # [B, position_dim]
        eta_hours,                            # [B, 1]
    ]
    return torch.cat(parts, dim=-1)


def flattened_ctx_dim(ctx_dims: Dict[str, Any]) -> int:
    return (
        int(ctx_dims["task_type_vocab"])
        + int(ctx_dims["type_vocab"])
        + int(ctx_dims["position_dim"])
        + 1   # eta_hours scalar
    )
