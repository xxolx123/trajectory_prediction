"""
fusion/code/export_onnx.py
--------------------------
导出 FullNetV2 为 ONNX。

当前策略：**只暴露 1 个输入 x_raw**；ContextBatch 在模型内用 buffer 零张量，
导出后的 ONNX 只有一个 3D 输入 [B, hist_len, 6]，和旧部署端
deploy/.../deploy_3trajs.cpp 期望一致（InferenceEngine 按 3D 输入识别）。

输出 shape = [B, K, 68]，K 由 gnn1/config.yaml 的 train.keep_top_k 决定（默认 3）。

TODO（C++ 侧接入真 ContextBatch 时）：
  改写 FullNetV2ForOnnx 让 forward 接受多个独立 Tensor（task_type/type/position/
  road_points/road_mask/own_info），并同步更新 cpp。
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


class FullNetV2ForOnnx(nn.Module):
    """
    只暴露 1 个输入 x_raw 给 ONNX；ContextBatch 在模型内部用全零 buffer 构造。
    """

    def __init__(self, full_net: FullNetV2, ctx_dims: dict) -> None:
        super().__init__()
        self.full_net = full_net
        self.ctx_dims = dict(ctx_dims)

        N_max = int(ctx_dims["road_max_points"])
        D_road = int(ctx_dims["road_point_dim"])
        D_pos = int(ctx_dims["position_dim"])
        D_own = int(ctx_dims["own_info_dim"])

        # GNN1 用
        self.register_buffer("_ctx_task_type", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_ctx_type", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_ctx_position", torch.zeros(1, D_pos))
        # ConstraintOptimizer 用
        self.register_buffer("_ctx_road_points", torch.zeros(1, N_max, D_road))
        self.register_buffer("_ctx_road_mask", torch.zeros(1, N_max, dtype=torch.bool))
        # LSTM2 / GNN2 占位
        self.register_buffer("_ctx_own_info", torch.zeros(1, D_own))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B = x_raw.shape[0]
        ctx = ContextBatch(
            task_type=self._ctx_task_type.expand(B).contiguous(),
            type=self._ctx_type.expand(B).contiguous(),
            position=self._ctx_position.expand(B, -1).contiguous(),
            road_points=self._ctx_road_points.expand(B, -1, -1).contiguous(),
            road_mask=self._ctx_road_mask.expand(B, -1).contiguous(),
            own_info=self._ctx_own_info.expand(B, -1).contiguous(),
        )
        return self.full_net(x_raw, ctx)


def _get_ctx_dims_from_gnn1(fusion_cfg_path: Path) -> dict:
    import yaml
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    gnn1_sec = fusion_cfg.get("gnn1", {})
    gnn1_cfg_path = fusion_cfg_dir / gnn1_sec["config"]
    with open(gnn1_cfg_path, "r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    return build_ctx_dims_from_config(gnn1_cfg)


def export_onnx(
    fusion_cfg_path: Path,
    onnx_out: Path,
    opset: int = 13,
) -> None:
    full_net = build_full_net_from_fusion_config(fusion_cfg_path)
    full_net.eval()
    device = next(full_net.parameters()).device

    ctx_dims = _get_ctx_dims_from_gnn1(fusion_cfg_path)
    model_for_onnx = FullNetV2ForOnnx(full_net, ctx_dims).to(device).eval()

    hist_len = full_net.hist_len
    feature_dim = full_net.feature_dim

    dummy_input = torch.zeros(1, hist_len, feature_dim, dtype=torch.float32, device=device)

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_for_onnx,
        dummy_input,
        onnx_out.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
        dynamo=False,
    )
    print(f"[Fusion] ONNX 已导出: {onnx_out}  (output shape = [batch, {full_net.top_k}, 68])")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FullNetV2 to ONNX.")
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
