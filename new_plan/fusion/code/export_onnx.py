"""
fusion/code/export_onnx.py
--------------------------
导出 FullNetV2 为 ONNX。

接口：**7 路独立输入**（与甲方流程图 5 路 + ETA 占位一致）

  hist_traj    [B, hist_len, 6]      float32   我方观测的历史轨迹（km / km·s⁻¹）
  task_type    [B]                   int64     敌方作战任务（0=打击）
  type         [B]                   int64     我方固定目标类型（0/1/2）
  position     [B, 3]                float32   我方固定目标 xyz（km，局部 ENU）
  road_points  [B, NB_max, NP_max, 3] float32  路网折线点（km，局部 ENU）
  road_mask    [B, NB_max, NP_max]   bool      路网点掩码
  eta          [B]                   int64     我方预计到达时间（秒，占位）

输出：
  output       [B, K, 68]            float32   K=3，布局见 fusion/README.txt

ENU 原点约定：以 hist_traj 末帧的 LLH 作为局部原点；C++ 部署侧把
RoadNetwork(LLH) → road_points(km) 之前必须用同一原点。

C++ 部署侧 TODO：
  - TrajSystem::Feed(...) 需要新增 RoadNetwork road、int64_t eta_sec
  - InferenceEngine 改成喂 7 输入 ORT session
  - LLH→ENU 转换可参考 constraint_optimizer/test_road_net/road_schema.py 的 llh_to_enu_km
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.context_schema import (  # noqa: E402
    ContextBatch,
    build_ctx_dims_from_config,
)
from fusion.code.full_net_v2 import FullNetV2, build_full_net_from_fusion_config  # noqa: E402
from fusion.code.build import load_fusion_config  # noqa: E402


# 输入名固定，部署 C++ 侧据此查找 ORT session 的输入张量
INPUT_NAMES = [
    "hist_traj",
    "task_type",
    "type",
    "position",
    "road_points",
    "road_mask",
    "eta",
]
OUTPUT_NAMES = ["output"]


class FullNetV2ForOnnx(nn.Module):
    """把 7 个独立张量打包成 ContextBatch 后喂 FullNetV2。"""

    def __init__(self, full_net: FullNetV2) -> None:
        super().__init__()
        self.full_net = full_net

    def forward(
        self,
        hist_traj: torch.Tensor,    # [B, hist_len, 6]      float32
        task_type: torch.Tensor,    # [B]                   int64
        type_id: torch.Tensor,      # [B]                   int64
        position: torch.Tensor,     # [B, 3]                float32
        road_points: torch.Tensor,  # [B, NB, NP, 3]        float32
        road_mask: torch.Tensor,    # [B, NB, NP]           bool
        eta: torch.Tensor,          # [B]                   int64
    ) -> torch.Tensor:              # [B, K=3, 68]
        ctx = ContextBatch(
            task_type=task_type,
            type=type_id,
            position=position,
            road_points=road_points,
            road_mask=road_mask,
            eta=eta,
        )
        return self.full_net(hist_traj, ctx)


def _get_ctx_dims_from_gnn1(fusion_cfg_path: Path) -> dict:
    import yaml
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    gnn1_sec = fusion_cfg.get("gnn1", {})
    gnn1_cfg_path = fusion_cfg_dir / gnn1_sec["config"]
    with open(gnn1_cfg_path, "r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    return build_ctx_dims_from_config(gnn1_cfg)


def _build_dummy_inputs(
    full_net: FullNetV2,
    ctx_dims: dict,
    device: torch.device,
    batch_size: int = 1,
) -> tuple:
    """构造一组 dummy 张量，用于 torch.onnx.export 的 args。"""
    B = int(batch_size)
    hist_len = int(full_net.hist_len)
    feat_dim = int(full_net.feature_dim)

    NB_max = int(ctx_dims.get("road_max_branches", 1))
    NP_max = int(ctx_dims["road_max_points"])
    D_road = int(ctx_dims["road_point_dim"])
    D_pos = int(ctx_dims["position_dim"])

    hist_traj = torch.zeros(B, hist_len, feat_dim, dtype=torch.float32, device=device)
    task_type = torch.zeros(B, dtype=torch.long, device=device)
    type_id = torch.zeros(B, dtype=torch.long, device=device)
    position = torch.zeros(B, D_pos, dtype=torch.float32, device=device)
    road_points = torch.zeros(B, NB_max, NP_max, D_road, dtype=torch.float32, device=device)
    road_mask = torch.zeros(B, NB_max, NP_max, dtype=torch.bool, device=device)
    eta = torch.zeros(B, dtype=torch.long, device=device)

    return (hist_traj, task_type, type_id, position, road_points, road_mask, eta)


def export_onnx(
    fusion_cfg_path: Path,
    onnx_out: Path,
    opset: int = 13,
) -> None:
    full_net = build_full_net_from_fusion_config(fusion_cfg_path)
    full_net.eval()
    device = next(full_net.parameters()).device

    ctx_dims = _get_ctx_dims_from_gnn1(fusion_cfg_path)
    model_for_onnx = FullNetV2ForOnnx(full_net).to(device).eval()

    dummy_inputs = _build_dummy_inputs(full_net, ctx_dims, device, batch_size=1)

    # 每个输入的 batch 维都开放成 dynamic
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    dynamic_axes["output"] = {0: "batch"}

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_for_onnx,
        dummy_inputs,
        onnx_out.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False,
    )
    print(
        f"[Fusion] ONNX 已导出: {onnx_out}\n"
        f"  inputs  = {INPUT_NAMES}\n"
        f"  outputs = {OUTPUT_NAMES}  shape = [batch, {full_net.top_k}, 68]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FullNetV2 to ONNX (7-input).")
    parser.add_argument(
        "--fusion-config",
        type=str,
        default=str(REPO_ROOT / "fusion" / "config.yaml"),
        help="fusion/config.yaml 的路径",
    )
    parser.add_argument("--onnx-out", type=str, required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    fusion_cfg_path = Path(args.fusion_config).resolve()
    onnx_out = Path(args.onnx_out)
    if not onnx_out.is_absolute():
        onnx_out = (REPO_ROOT / onnx_out).resolve()

    export_onnx(fusion_cfg_path, onnx_out, opset=args.opset)


if __name__ == "__main__":
    main()
