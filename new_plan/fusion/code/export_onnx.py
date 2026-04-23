"""
fusion/code/export_onnx.py
--------------------------
导出 FullNetV2 为 ONNX。

当前策略：**只暴露 1 个输入 x_raw**；context 在模型内用 buffer 零张量，
导出后的 ONNX 只有一个 3D 输入 [B, hist_len, 6]，和旧部署端
deploy/.../deploy_3trajs.cpp 期望一致（InferenceEngine 按 3D 输入识别）。

TODO（等真正的 context 接入后）：
  改写 FullNetV2ForOnnx 让 forward 接受多个独立 Tensor，并同步改 cpp。
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
    只暴露 1 个输入 x_raw 给 ONNX；context 在模型内部用全零 buffer 构造。
    """

    def __init__(self, full_net: FullNetV2, ctx_dims: dict) -> None:
        super().__init__()
        self.full_net = full_net
        self.ctx_dims = dict(ctx_dims)

        D_task = int(ctx_dims["target_task_dim"])
        N_tgt = int(ctx_dims["n_fixed_targets"])
        D_tgt = int(ctx_dims["fixed_target_dim"])
        D_type = int(ctx_dims["target_type_dim"])
        D_road = int(ctx_dims["road_network_dim"])
        D_own = int(ctx_dims["own_info_dim"])

        self.register_buffer("_ctx_target_task", torch.zeros(1, D_task))
        self.register_buffer("_ctx_fixed_targets", torch.zeros(1, N_tgt, D_tgt))
        self.register_buffer("_ctx_target_type", torch.zeros(1, D_type))
        self.register_buffer("_ctx_road_network", torch.zeros(1, D_road))
        self.register_buffer("_ctx_own_info", torch.zeros(1, D_own))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B = x_raw.shape[0]
        ctx = ContextBatch(
            target_task=self._ctx_target_task.expand(B, -1).contiguous(),
            fixed_targets=self._ctx_fixed_targets.expand(B, -1, -1).contiguous(),
            target_type=self._ctx_target_type.expand(B, -1).contiguous(),
            road_network=self._ctx_road_network.expand(B, -1).contiguous(),
            own_info=self._ctx_own_info.expand(B, -1).contiguous(),
        )
        return self.full_net(x_raw, ctx)


def _get_gnn1_ctx_dims(fusion_cfg_path: Path) -> dict:
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

    ctx_dims = _get_gnn1_ctx_dims(fusion_cfg_path)
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
    print(f"[Fusion] ONNX 已导出: {onnx_out}")


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
