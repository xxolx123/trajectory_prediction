"""
fusion/code/export_onnx.py
--------------------------
导出 FullNetV2 为 ONNX。**双模式 + opset 11**（mindspore-lite 1.8.1 适配）。

部署侧用一个**外部 bool flag**（不进图）决定加载哪份 ONNX；ONNX 自身仍是
静态图，输入数量固定，无运行时 if 子图。两份 ONNX 共用同一份 fusion config，
区别仅是导出时的 ``--mode``。

----------------------------------------------------------------
两份 ONNX 的输入契约
----------------------------------------------------------------

no_road 版（5 输入）::

    hist_traj   [B, hist_len, 6]      float32   我方观测的历史轨迹
    task_type   [B]                   int64     敌方作战任务（0=打击）
    type        [B]                   int64     我方固定目标类型 (0/1/2)
    position    [B, 3]                float32   我方固定目标 xyz (km，局部 ENU)
    eta         [B]                   int64     我方预计到达时间（秒，占位）

with_road 版（7 输入）::

    hist_traj   [B, hist_len, 6]      float32
    task_type   [B]                   int64
    type        [B]                   int64
    position    [B, 3]                float32
    road_points [B, NB_max, NP_max, 3] float32  路网折线点 (km，局部 ENU)
    road_mask   [B, NB_max, NP_max]   bool      路网点掩码 (True=有效)
    eta         [B]                   int64

输出统一::

    output      [B, K, 68]            float32   K=3，布局见 fusion/README.txt

----------------------------------------------------------------
no_road 版的实现细节
----------------------------------------------------------------
- 导出前**强制把 full_net.constraint = None**（即使 fusion config 中
  constraint_optimizer.enable=true 也覆盖为 None），FullNetV2.forward 走
  ``else: refined_flat = fut_flat`` 分支，不调任何路网算子。
- 5 输入 wrapper 把 road_points / road_mask 设为 None，依赖 ``_normalize_ctx``
  在 forward 里用零张量补占位（trace 后是 ONNX 内部 Constant，不进 graph.input）。

with_road 版的实现细节
----------------------------------------------------------------
- 复用现有 7 输入 wrapper。
- dummy 路网必须是**有效**的（mask 至少局部 True、points 非零 polyline），
  否则 ConstraintOptimizer 内部 `torch.where(branch_has_seg, cost, BIG)` 仍会
  让全部 batch 走 fallback 分支，但向量化重写后 fallback 也会保留路网算子在
  graph 里——稳妥起见还是给一组真实路网。

opset 11 兼容前提（**已在仓库其它处落地**）
----------------------------------------------------------------
- LSTM2: ``fusion/config.yaml`` 设 ``lstm2.manual_attention: true``，
  build.py 会强制走 ``transformer_manual``，避开 SDPA（要 opset ≥ 14）。
- ConstraintOptimizer: ``_road_arc_projection`` 已向量化重写，去掉
  ``torch.searchsorted``（要 opset ≥ 16），改用 broadcast 比较。

C++ 部署侧 TODO
----------------------------------------------------------------
- 接外部 bool flag → 选 ``full_net_v2_no_road.onnx`` 或 ``full_net_v2_with_road.onnx``
- 拿不到路网时：直接加载 no_road 版，喂 5 个输入即可
- 拿到路网时：加载 with_road 版，按 7 输入约定准备张量

用法（在 new_plan/ 下激活 LSTM_traj_predict 环境后）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"

    # 默认：同时导出 no_road + with_road 两份 ONNX 到 fusion/ 目录
    python -m fusion.code.export_onnx

    # 只导其中一份：
    python -m fusion.code.export_onnx --mode no_road
    python -m fusion.code.export_onnx --mode with_road

    # 显式指定 fusion config / 输出目录 / opset：
    python -m fusion.code.export_onnx \
        --fusion-config fusion/config.yaml \
        --out-dir fusion \
        --mode both \
        --opset 11

输出文件名固定为 ``full_net_v2_no_road.onnx`` / ``full_net_v2_with_road.onnx``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

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
from fusion.code.build import load_fusion_config, _is_enabled  # noqa: E402


# ---- 输入名集合（部署 C++ 侧据此查找 ORT session 的输入张量）----
# 实际输入数取决于 fusion 配置中 gnn2 是否启用：eta 仅被 GNN2 消费，
# gnn2.enable=false 时 trace 会裁掉 eta（因为图里没人用它）。
# 由 _expected_input_names 根据 mode + gnn2 状态动态算出最终列表。
OUTPUT_NAMES = ["output"]

DEFAULT_OPSET = 11
VALID_MODES = ("no_road", "with_road", "both")


def _expected_input_names(mode: str, gnn2_enabled: bool) -> List[str]:
    """
    根据 mode 和 GNN2 启用状态算最终 ONNX 输入名集合。

      mode=no_road,   gnn2=True  → 5 输入：hist / task_type / type / position / eta
      mode=no_road,   gnn2=False → 4 输入：hist / task_type / type / position
      mode=with_road, gnn2=True  → 7 输入：上面 + road_points / road_mask + eta
      mode=with_road, gnn2=False → 6 输入：上面 + road_points / road_mask
    """
    names = ["hist_traj", "task_type", "type", "position"]
    if mode == "with_road":
        names.extend(["road_points", "road_mask"])
    if gnn2_enabled:
        names.append("eta")
    return names


# ============================================================
# 两套 wrapper
# ============================================================

class _FullNetV2ForOnnxNoRoadWithEta(nn.Module):
    """no_road + GNN2 启用：5 输入。"""

    def __init__(self, full_net: FullNetV2) -> None:
        super().__init__()
        self.full_net = full_net

    def forward(
        self,
        hist_traj: torch.Tensor,
        task_type: torch.Tensor,
        type_id: torch.Tensor,
        position: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        ctx = ContextBatch(
            task_type=task_type, type=type_id, position=position,
            road_points=None,       # type: ignore[arg-type]
            road_mask=None,         # type: ignore[arg-type]
            eta=eta,
        )
        return self.full_net(hist_traj, ctx)


class _FullNetV2ForOnnxNoRoadNoEta(nn.Module):
    """no_road + GNN2 关闭：4 输入。"""

    def __init__(self, full_net: FullNetV2) -> None:
        super().__init__()
        self.full_net = full_net

    def forward(
        self,
        hist_traj: torch.Tensor,
        task_type: torch.Tensor,
        type_id: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        ctx = ContextBatch(
            task_type=task_type, type=type_id, position=position,
            road_points=None,       # type: ignore[arg-type]
            road_mask=None,         # type: ignore[arg-type]
            eta=None,               # type: ignore[arg-type]
        )
        return self.full_net(hist_traj, ctx)


class _FullNetV2ForOnnxWithRoadWithEta(nn.Module):
    """with_road + GNN2 启用：7 输入。"""

    def __init__(self, full_net: FullNetV2) -> None:
        super().__init__()
        self.full_net = full_net

    def forward(
        self,
        hist_traj: torch.Tensor,
        task_type: torch.Tensor,
        type_id: torch.Tensor,
        position: torch.Tensor,
        road_points: torch.Tensor,
        road_mask: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        ctx = ContextBatch(
            task_type=task_type, type=type_id, position=position,
            road_points=road_points, road_mask=road_mask, eta=eta,
        )
        return self.full_net(hist_traj, ctx)


class _FullNetV2ForOnnxWithRoadNoEta(nn.Module):
    """with_road + GNN2 关闭：6 输入。"""

    def __init__(self, full_net: FullNetV2) -> None:
        super().__init__()
        self.full_net = full_net

    def forward(
        self,
        hist_traj: torch.Tensor,
        task_type: torch.Tensor,
        type_id: torch.Tensor,
        position: torch.Tensor,
        road_points: torch.Tensor,
        road_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx = ContextBatch(
            task_type=task_type, type=type_id, position=position,
            road_points=road_points, road_mask=road_mask,
            eta=None,               # type: ignore[arg-type]
        )
        return self.full_net(hist_traj, ctx)


def _pick_wrapper(
    full_net: FullNetV2, mode: str, gnn2_enabled: bool,
) -> nn.Module:
    """根据 mode + gnn2 状态选合适的 wrapper 类。"""
    if mode == "no_road":
        if gnn2_enabled:
            return _FullNetV2ForOnnxNoRoadWithEta(full_net)
        return _FullNetV2ForOnnxNoRoadNoEta(full_net)
    if mode == "with_road":
        if gnn2_enabled:
            return _FullNetV2ForOnnxWithRoadWithEta(full_net)
        return _FullNetV2ForOnnxWithRoadNoEta(full_net)
    raise ValueError(f"unknown mode: {mode}")


# ============================================================
# 工具
# ============================================================

def _get_ctx_dims_from_gnn1(fusion_cfg_path: Path) -> dict:
    import yaml
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    gnn1_sec = fusion_cfg.get("gnn1", {})
    gnn1_cfg_path = fusion_cfg_dir / gnn1_sec["config"]
    with open(gnn1_cfg_path, "r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    return build_ctx_dims_from_config(gnn1_cfg)


def _build_dummy(
    full_net: FullNetV2,
    ctx_dims: dict,
    device: torch.device,
    mode: str,
    gnn2_enabled: bool,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, ...]:
    """
    根据 mode + gnn2 状态构造 dummy 输入元组（顺序与 _expected_input_names 严格对齐）。

    with_road 时路网填充一条 NV 个点的直线 + 占位分支：保证 ConstraintOptimizer
    向量化版进入正常投影路径，让 trace 把所有路网算子录进图（mask 全 False 时
    虽然新版也用 torch.where fallback 保留算子，但稳妥起见还是给真实路网）。
    """
    B = int(batch_size)
    hist_len = int(full_net.hist_len)
    feat_dim = int(full_net.feature_dim)
    D_pos = int(ctx_dims["position_dim"])

    inputs: List[torch.Tensor] = [
        torch.zeros(B, hist_len, feat_dim, dtype=torch.float32, device=device),  # hist_traj
        torch.zeros(B, dtype=torch.long, device=device),                          # task_type
        torch.zeros(B, dtype=torch.long, device=device),                          # type
        torch.zeros(B, D_pos, dtype=torch.float32, device=device),                # position
    ]

    if mode == "with_road":
        NB_max = int(ctx_dims.get("road_max_branches", 1))
        NP_max = int(ctx_dims["road_max_points"])
        D_road = int(ctx_dims["road_point_dim"])
        road_points = torch.zeros(B, NB_max, NP_max, D_road,
                                  dtype=torch.float32, device=device)
        road_mask = torch.zeros(B, NB_max, NP_max, dtype=torch.bool, device=device)
        NV = max(2, min(NP_max, 8))
        line = torch.arange(NV, dtype=torch.float32, device=device).view(NV, 1)
        line = torch.cat(
            [
                line,
                torch.zeros(NV, 1, dtype=torch.float32, device=device),
                torch.zeros(NV, 1, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )
        road_points[:, 0, :NV, :] = line.unsqueeze(0)
        road_mask[:, 0, :NV] = True
        inputs.extend([road_points, road_mask])

    if gnn2_enabled:
        inputs.append(torch.zeros(B, dtype=torch.long, device=device))            # eta

    return tuple(inputs)


def _build_no_road_full_net(fusion_cfg_path: Path) -> FullNetV2:
    """
    构建 no_road 版 full_net：先正常 build，再强制 constraint=None。

    更稳的做法是临时改 fusion_cfg 把 constraint_optimizer.enable=false 重新
    build；但实测 `setattr(full_net, 'constraint', None)` 后，FullNetV2.forward
    里 ``if self.constraint is not None`` 会走 else 分支不调路网，
    trace 完全不会引入路网算子，行为等价。少一次 build，省事。
    """
    full_net = build_full_net_from_fusion_config(fusion_cfg_path)
    full_net.constraint = None
    full_net.enable_flags = dict(full_net.enable_flags)
    full_net.enable_flags["constraint_optimizer"] = False
    return full_net


def _check_onnx(onnx_path: Path, expected_inputs: Sequence[str]) -> None:
    """导出后用 onnx.checker + 输入名集合断言。"""
    try:
        import onnx
    except ImportError:
        print(f"[Fusion] 警告：onnx 未安装，跳过 onnx.checker / 输入名校验 ({onnx_path})")
        return
    m = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(m)
    got = [i.name for i in m.graph.input]
    expect = list(expected_inputs)
    assert got == expect, (
        f"ONNX 输入名集合不匹配:\n"
        f"  期望: {expect}\n"
        f"  实际: {got}"
    )
    print(f"[Fusion] onnx.checker OK; 输入名集合匹配: {got}")


# ============================================================
# 单模式导出
# ============================================================

def _export_single_mode(
    fusion_cfg_path: Path,
    onnx_out: Path,
    mode: str,
    opset: int,
) -> None:
    """导出单一 mode 的 ONNX。mode ∈ {'no_road', 'with_road'}。"""
    fusion_cfg, _ = load_fusion_config(fusion_cfg_path)
    gnn2_enabled = bool(_is_enabled(fusion_cfg.get("gnn2", {}) or {}, default=True))

    if mode == "no_road":
        full_net = _build_no_road_full_net(fusion_cfg_path)
    elif mode == "with_road":
        full_net = build_full_net_from_fusion_config(fusion_cfg_path)
    else:
        raise ValueError(f"unknown mode: {mode}")

    full_net.eval()
    device = next(full_net.parameters()).device
    ctx_dims = _get_ctx_dims_from_gnn1(fusion_cfg_path)

    wrapper = _pick_wrapper(full_net, mode, gnn2_enabled).to(device).eval()
    dummy = _build_dummy(full_net, ctx_dims, device, mode, gnn2_enabled, batch_size=1)
    input_names = _expected_input_names(mode, gnn2_enabled)

    if len(dummy) != len(input_names):
        raise RuntimeError(
            f"内部一致性错误：dummy ({len(dummy)} 个) 与 input_names "
            f"({len(input_names)} 个) 数量不匹配；input_names = {input_names}"
        )

    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["output"] = {0: "batch"}

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_out.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False,
    )
    print(
        f"[Fusion] ONNX[{mode}] 已导出: {onnx_out}\n"
        f"  inputs  = {input_names}  (gnn2_enabled={gnn2_enabled})\n"
        f"  outputs = {OUTPUT_NAMES}  shape = [batch, {full_net.top_k}, 68]\n"
        f"  opset   = {opset}"
    )

    _check_onnx(onnx_out, input_names)


# ============================================================
# 多模式入口
# ============================================================

def export_onnx(
    fusion_cfg_path: Path,
    out_dir: Path,
    mode: str = "both",
    opset: int = DEFAULT_OPSET,
) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"--mode 只能是 {VALID_MODES} 之一，得到 '{mode}'")

    out_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("no_road", "both"):
        _export_single_mode(
            fusion_cfg_path=fusion_cfg_path,
            onnx_out=out_dir / "full_net_v2_no_road.onnx",
            mode="no_road",
            opset=opset,
        )

    if mode in ("with_road", "both"):
        _export_single_mode(
            fusion_cfg_path=fusion_cfg_path,
            onnx_out=out_dir / "full_net_v2_with_road.onnx",
            mode="with_road",
            opset=opset,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 FullNetV2 为 ONNX（双模式 no_road / with_road，"
                    "默认 opset 11，mindspore-lite 1.8.1 适配）。"
    )
    parser.add_argument(
        "--fusion-config",
        type=str,
        default=str(REPO_ROOT / "fusion" / "config.yaml"),
        help="fusion/config.yaml 的路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "fusion"),
        help="ONNX 输出目录；文件名固定为 full_net_v2_{mode}.onnx",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=list(VALID_MODES),
        help="导出哪一/两份 ONNX。默认 both 同时导 no_road + with_road",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset；默认 {DEFAULT_OPSET}（mindspore-lite 1.8.1 验证过）。"
             f" 需要 lstm2.manual_attention=true 且 ConstraintOptimizer 已向量化。",
    )
    args = parser.parse_args()

    fusion_cfg_path = Path(args.fusion_config).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    export_onnx(
        fusion_cfg_path=fusion_cfg_path,
        out_dir=out_dir,
        mode=args.mode,
        opset=int(args.opset),
    )


if __name__ == "__main__":
    main()
