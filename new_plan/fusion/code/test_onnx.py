"""
fusion/code/test_onnx.py
------------------------
直接调用导出的 ONNX 跑数值推理，并与 PyTorch FullNetV2 的同输入输出做**逐字段**
对比。用于补齐 fusion 当前缺失的 PyTorch ↔ ONNX 数值闭环（mindspore-lite
与 ONNX 的对比由部署侧 run_minimal 完成；本脚本只覆盖到 ONNX 这一段）。

输入数据来源
-------------
默认复用 fusion/eval_vis/ 下**最新**一次 test_pipeline 的产出：

    fusion/eval_vis/run_test_<stamp>/
        no_road/
            inputs.npz   ← FullNetV2 真正吃下去的张量（含 hist / ctx / road / eta）
            outputs.npz  ← PyTorch FullNetV2 输出（baseline）
        with_road/
            ...

这样保证两边喂的是**完全相同**的张量；不会再因为数据准备链路不同导致误判。
也可以用 --from-eval-vis 显式指定一次特定的 run。

输出
-------
fusion/eval_vis_onnx/run_test_<stamp>/{no_road,with_road}/
    inputs.npz            （直接拷贝；便于部署侧 run_minimal 复用同一份输入做 .ms 验证）
    outputs_torch.npz     （直接拷贝 eval_vis/outputs.npz，重命名）
    outputs_onnx.npz      （新；onnxruntime 推理结果，字段命名与 outputs_torch 完全一致）
    diff.txt              （逐字段 max|Δ| / mean|Δ| 文本摘要）
    vis_compare.png       （每个样本两子图：左 PyTorch top-K，右 ONNX top-K；颜色对齐）

字段对比清单
-------
    output [B, K, 68]      整体张量 max|Δ|
    refined_traj 0..59     轨迹（路网约束后或未约束）
    intent_class 60        LSTM2 argmax 后的 float
    threat_prob 61         LSTM2 sigmoid
    strike_pos 62..64      GNN2 km xyz
    strike_radius 65       GNN2 km
    strike_conf 66         GNN2 [0,1]
    mode_prob 67           GNN1 top-K 重归一化（K 条和 = 1）

判定阈值（按字段量纲分组）
-------
| 字段                            | 默认 atol | 说明                                       |
|---------------------------------|-----------|-------------------------------------------|
| intent_class                    | 0         | argmax 整数，必须精确相等                  |
| threat_prob / strike_conf / mode_prob | 1e-4 | 值域 [0,1]，差应非常小                    |
| strike_radius                   | 1e-3 km   | 单步标量，无累积                          |
| strike_pos / refined_traj / output | 2e-3 km | km 量纲；refined_traj 含 10 步 cumsum +    |
|                                 |           | denormalize，fp32 噪声放到 ~1-2 米量级    |

CLI `--atol` 覆盖所有字段的阈值（适合做更严格的 1e-5 等价性测试）。

用法
-------
    cd new_plan
    $env:PYTHONPATH = "$PWD"
    # Windows 上 PyTorch 的 MKL OpenMP 与 onnxruntime 的 LLVM OpenMP 会冲突，
    # 需要设置以下环境变量绕过（Linux 一般不需要）：
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"

    # 默认：自动找 eval_vis 下最新 run，跑 no_road + with_road 两份
    python -m fusion.code.test_onnx

    # 仅跑一种模式：
    python -m fusion.code.test_onnx --mode no_road

    # 显式指定 eval_vis run 与输出目录：
    python -m fusion.code.test_onnx \
        --from-eval-vis fusion/eval_vis/run_test_20260429_003430 \
        --out-root fusion/eval_vis_onnx \
        --mode both
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 注意：onnxruntime 是新增可选依赖，requirements.txt 里也已正式放进去
try:
    import onnxruntime as ort  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "[Test/ONNX] 未安装 onnxruntime；请在当前 Python 环境中执行：\n"
        "    pip install onnxruntime>=1.15\n"
        "（已在 new_plan/requirements.txt 中要求）"
    ) from e

from fusion.code.build import _is_enabled, load_fusion_config  # noqa: E402


# ============================================================
# 常量：与 fusion/code/export_onnx.py / build.py 严格对齐
# ============================================================

VALID_MODES = ("no_road", "with_road")

ONNX_FILENAME = {
    "no_road":   "full_net_v2_no_road.onnx",
    "with_road": "full_net_v2_with_road.onnx",
}

# 出现在 ONNX graph.input 里的顺序与名称（与 export_onnx._expected_input_names 对齐）
def _expected_input_names(mode: str, gnn2_enabled: bool) -> List[str]:
    names = ["hist_traj", "task_type", "type", "position"]
    if mode == "with_road":
        names.extend(["road_points", "road_mask"])
    if gnn2_enabled:
        names.append("eta")
    return names


# ============================================================
# eval_vis 自动定位
# ============================================================

def _find_latest_eval_vis(eval_vis_root: Path) -> Path:
    if not eval_vis_root.exists():
        raise FileNotFoundError(
            f"找不到 eval_vis 根目录：{eval_vis_root}；"
            f"请先跑 test_pipeline 产出输入。"
        )
    runs = sorted(
        [p for p in eval_vis_root.iterdir()
         if p.is_dir() and p.name.startswith("run_test_")],
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(
            f"{eval_vis_root} 下没有 run_test_* 子目录；"
            f"请先跑 fusion/code/test_pipeline.py。"
        )
    return runs[0]


def _check_eval_vis_run(run_dir: Path, mode: str) -> Tuple[Path, Path]:
    sub = run_dir / mode
    inputs_npz = sub / "inputs.npz"
    outputs_npz = sub / "outputs.npz"
    if not inputs_npz.exists() or not outputs_npz.exists():
        raise FileNotFoundError(
            f"{sub} 下缺少 inputs.npz 或 outputs.npz；\n"
            f"请确认 test_pipeline 已经跑过 mode={mode}（默认 --mode both）。"
        )
    return inputs_npz, outputs_npz


# ============================================================
# ONNX 推理
# ============================================================

def _build_ort_inputs(
    inputs_npz: Path,
    expected_names: Sequence[str],
    mode: str,
    gnn2_enabled: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    """从 eval_vis/inputs.npz 取张量，组装成 onnxruntime 的 input dict。"""
    z = np.load(inputs_npz)
    feed: Dict[str, np.ndarray] = {}

    feed["hist_traj"] = z["hist_traj"].astype(np.float32, copy=False)        # [B, 20, 6]
    feed["task_type"] = z["task_type"].astype(np.int64, copy=False)          # [B]
    feed["type"]      = z["type"].astype(np.int64, copy=False)               # [B]
    feed["position"]  = z["position"].astype(np.float32, copy=False)         # [B, 3]

    if mode == "with_road":
        feed["road_points"] = z["road_points"].astype(np.float32, copy=False)  # [B, NB, NP, 3]
        feed["road_mask"]   = z["road_mask"].astype(np.bool_, copy=False)      # [B, NB, NP]

    if gnn2_enabled:
        feed["eta"] = z["eta"].astype(np.int64, copy=False)                  # [B]

    # 按 ONNX 输入名集合校验：feed 的 key 集合必须和 expected_names 完全一致
    expected_set = set(expected_names)
    actual_set = set(feed.keys())
    if expected_set != actual_set:
        raise RuntimeError(
            f"[Test/ONNX][{mode}] inputs.npz 与 ONNX 输入名集合不匹配：\n"
            f"  期望: {sorted(expected_set)}\n"
            f"  实际: {sorted(actual_set)}"
        )

    B = int(feed["hist_traj"].shape[0])
    return feed, B


def _run_onnx(onnx_path: Path, feed: Dict[str, np.ndarray]) -> np.ndarray:
    sess = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )

    # 用 ONNX 模型自身的 graph.input 顺序再核一次，提前发现 input_names 漂移
    sess_input_names = [i.name for i in sess.get_inputs()]
    if set(sess_input_names) != set(feed.keys()):
        raise RuntimeError(
            f"[Test/ONNX] {onnx_path.name} session 输入名 {sess_input_names} "
            f"与 feed {sorted(feed.keys())} 不匹配（可能 ONNX 是旧版本？）"
        )

    out_list = sess.run(None, feed)
    if not out_list:
        raise RuntimeError(f"[Test/ONNX] {onnx_path.name} session.run 返回空")
    out = out_list[0]
    if out.ndim != 3 or out.shape[-1] != 68:
        raise RuntimeError(
            f"[Test/ONNX] {onnx_path.name} 输出形状异常：{out.shape}（期望 [B,K,68]）"
        )
    return out.astype(np.float32, copy=False)


# ============================================================
# 字段拆解 + 数值对比
# ============================================================

def _split_fields(out: np.ndarray, fut_len: int) -> Dict[str, np.ndarray]:
    """[B, K, 68] → 字段 dict。命名与 outputs.npz 完全对齐，便于后续对比。"""
    B, K, D = out.shape
    assert D == 68
    return {
        "output":        out,
        "refined_traj":  out[..., 0:60].reshape(B, K, fut_len, 6),
        "intent_class":  out[..., 60],
        "threat_prob":   out[..., 61],
        "strike_pos":    out[..., 62:65],
        "strike_radius": out[..., 65],
        "strike_conf":   out[..., 66],
        "mode_prob":     out[..., 67],
    }


def _safe_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int, int]:
    """
    NaN-aware 差异统计：
      返回 (max|Δ|, mean|Δ|, 总元素数, 仅一侧 NaN 的元素数)
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    both_nan = nan_a & nan_b
    only_one_nan = (nan_a ^ nan_b).sum()
    valid = (~nan_a) & (~nan_b)
    if valid.any():
        d = np.abs(a[valid] - b[valid])
        return float(d.max()), float(d.mean()), int(valid.size), int(only_one_nan)
    if both_nan.all():
        return 0.0, 0.0, int(a.size), int(only_one_nan)
    return float("inf"), float("inf"), int(a.size), int(only_one_nan)


# 各字段默认 atol（按量纲分组；CLI --atol 非 None 时强制覆盖全部字段）
# 轨迹/坐标类用 2e-3 km（= 2 米）：km 量级轨迹上 fp32 cumsum + denormalize 的累积
# 噪声会到这个量级；2 米相对几 km 的预测距离是 0.1% 量级，对应用毫无影响。
_DEFAULT_PER_FIELD_ATOL: Dict[str, float] = {
    "intent_class":  0.0,    # argmax 整数，精确相等
    "threat_prob":   1e-4,   # [0, 1]
    "strike_conf":   1e-4,   # [0, 1]
    "mode_prob":     1e-4,   # K 条和 = 1
    "strike_radius": 1e-3,   # km；单步标量，无 cumsum
    "strike_pos":    2e-3,   # km；含路网投影 / FiLM 等多 op 累积
    "refined_traj":  2e-3,   # km；10 步 cumsum + denormalize 的 fp32 累积噪声
    "output":        2e-3,   # 上面全部字段的并集
}


def _diff_per_field(
    torch_fields: Dict[str, np.ndarray],
    onnx_fields: Dict[str, np.ndarray],
    override_atol: Optional[float],
    rtol: float,
) -> Tuple[List[str], bool]:
    """
    返回 (文本行列表, overall_pass)。
    若 override_atol 非 None，所有字段都用它；否则用 _DEFAULT_PER_FIELD_ATOL。
    """
    lines: List[str] = []
    overall_pass = True
    for key in [
        "output", "refined_traj", "intent_class", "threat_prob",
        "strike_pos", "strike_radius", "strike_conf", "mode_prob",
    ]:
        a = torch_fields[key]
        b = onnx_fields[key]
        if a.shape != b.shape:
            lines.append(
                f"[FAIL] {key:<14}  shape mismatch: torch={a.shape} onnx={b.shape}"
            )
            overall_pass = False
            continue
        max_d, mean_d, n, n_only_one_nan = _safe_diff(a, b)
        local_tol = (
            override_atol if override_atol is not None
            else _DEFAULT_PER_FIELD_ATOL.get(key, 1e-4)
        )
        ok = (max_d <= local_tol) and (n_only_one_nan == 0)
        tag = "PASS" if ok else "FAIL"
        nan_note = (
            f"  one-side-NaN={n_only_one_nan}" if n_only_one_nan > 0 else ""
        )
        lines.append(
            f"[{tag}] {key:<14}  shape={a.shape}  "
            f"max|Δ|={max_d:.3e}  mean|Δ|={mean_d:.3e}  "
            f"atol={local_tol:.0e}  n={n}{nan_note}"
        )
        if not ok:
            overall_pass = False

    lines.append("")
    if override_atol is not None:
        lines.append(f"overall = {'PASS' if overall_pass else 'FAIL'}  "
                     f"(global atol={override_atol})")
    else:
        lines.append(f"overall = {'PASS' if overall_pass else 'FAIL'}  "
                     f"(per-field atol; 见上方每行)")
    return lines, overall_pass


# ============================================================
# 可视化：PyTorch top-K vs ONNX top-K 并排
# ============================================================

_TOPK_COLORS = ["#D62728", "#FF7F0E", "#C71585", "#8B4513", "#2F4F4F"]


def _grid_layout(B: int) -> Tuple[int, int]:
    if B <= 1:
        return 1, 1
    if B <= 3:
        return B, 1
    if B <= 8:
        return B, 1
    return int(np.ceil(B / 2)), 2


def _vis_compare(
    hist_traj: np.ndarray,        # [B, 20, 6]
    position: np.ndarray,         # [B, 3]
    refined_torch: np.ndarray,    # [B, K, T, 6]
    refined_onnx: np.ndarray,     # [B, K, T, 6]
    mode_prob_torch: np.ndarray,  # [B, K]
    mode_prob_onnx: np.ndarray,   # [B, K]
    out_path: Path, mode: str,
) -> None:
    B = int(hist_traj.shape[0])
    K = int(refined_torch.shape[1])

    rows = B
    cols = 2
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.0 * cols, 4.5 * rows + 0.6),
        squeeze=False,
    )
    for i in range(B):
        for c, (label, traj, probs) in enumerate([
            ("PyTorch", refined_torch[i], mode_prob_torch[i]),
            ("ONNX",    refined_onnx[i],  mode_prob_onnx[i]),
        ]):
            ax = axes[i, c]
            # history（黑色实线）
            hx = hist_traj[i, :, 0]
            hy = hist_traj[i, :, 1]
            ax.plot(hx, hy, color="black", linewidth=2.0, alpha=0.85,
                    zorder=4, label="history")
            ax.scatter(0, 0, s=70, marker="*", color="black", zorder=6,
                       label="hist last（原点）")
            # position（红五角星）
            px, py = float(position[i, 0]), float(position[i, 1])
            ax.scatter(px, py, s=200, color="red", marker="*",
                       edgecolors="black", linewidths=1.0, zorder=8,
                       label=f"position ({px:.1f},{py:.1f})")
            # top-K 候选
            for r in range(K):
                col = _TOPK_COLORS[r % len(_TOPK_COLORS)]
                xy = traj[r, :, 0:2]
                ax.plot(xy[:, 0], xy[:, 1], color=col, linewidth=1.6,
                        alpha=0.95, zorder=5,
                        label=f"top{r+1}  p={probs[r]*100:.1f}%")
            ax.set_title(f"sample {i}  [{label}]", fontsize=10)
            ax.set_xlabel("x (km, ENU)")
            ax.set_ylabel("y (km, ENU)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
            ax.legend(fontsize=7.5, loc="best", framealpha=0.85)

    fig.suptitle(
        f"PyTorch FullNetV2 vs ONNX  [{mode}]  "
        f"B={B}  K={K}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================================================
# 主流程：单模式
# ============================================================

def _run_single_mode(
    mode: str,
    fusion_cfg: dict,
    onnx_dir: Path,
    eval_vis_run_dir: Path,
    out_run_dir: Path,
    fut_len: int,
    atol: Optional[float], rtol: float,
) -> bool:
    print("=" * 64)
    print(f"[Test/ONNX] mode = {mode}")
    print("=" * 64)

    onnx_path = onnx_dir / ONNX_FILENAME[mode]
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"找不到 ONNX 文件：{onnx_path}；"
            f"请先跑 fusion/code/export_onnx.py --mode both。"
        )

    inputs_npz, outputs_npz_torch = _check_eval_vis_run(eval_vis_run_dir, mode)
    print(f"[Test/ONNX] eval_vis     = {eval_vis_run_dir / mode}")
    print(f"[Test/ONNX] inputs.npz   = {inputs_npz}")
    print(f"[Test/ONNX] torch out    = {outputs_npz_torch}")
    print(f"[Test/ONNX] onnx model   = {onnx_path}")

    gnn2_enabled = bool(_is_enabled(fusion_cfg.get("gnn2", {}) or {}, default=True))
    expected_names = _expected_input_names(mode, gnn2_enabled)
    print(f"[Test/ONNX] expected ins = {expected_names}  (gnn2_enabled={gnn2_enabled})")

    feed, B = _build_ort_inputs(inputs_npz, expected_names, mode, gnn2_enabled)
    t0 = time.perf_counter()
    out_onnx = _run_onnx(onnx_path, feed)
    t1 = time.perf_counter()
    print(f"[Test/ONNX] ONNX forward done: out.shape={out_onnx.shape} "
          f"  elapsed={(t1 - t0) * 1e3:.1f} ms")

    z_torch = np.load(outputs_npz_torch)
    out_torch = z_torch["output"].astype(np.float32, copy=False)
    if out_torch.shape != out_onnx.shape:
        raise RuntimeError(
            f"形状不一致 torch={out_torch.shape} onnx={out_onnx.shape}"
        )

    onnx_fields = _split_fields(out_onnx, fut_len)
    torch_fields = _split_fields(out_torch, fut_len)

    diff_lines, overall_ok = _diff_per_field(
        torch_fields, onnx_fields, override_atol=atol, rtol=rtol,
    )

    out_run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 复制 inputs.npz / outputs_torch.npz；写 outputs_onnx.npz ----
    shutil.copy2(inputs_npz, out_run_dir / "inputs.npz")
    shutil.copy2(outputs_npz_torch, out_run_dir / "outputs_torch.npz")
    np.savez(
        out_run_dir / "outputs_onnx.npz",
        output=onnx_fields["output"].astype(np.float32),
        refined_traj=onnx_fields["refined_traj"].astype(np.float32),
        intent_class=onnx_fields["intent_class"].astype(np.float32),
        threat_prob=onnx_fields["threat_prob"].astype(np.float32),
        strike_pos=onnx_fields["strike_pos"].astype(np.float32),
        strike_radius=onnx_fields["strike_radius"].astype(np.float32),
        strike_conf=onnx_fields["strike_conf"].astype(np.float32),
        mode_prob=onnx_fields["mode_prob"].astype(np.float32),
    )

    # ---- diff.txt ----
    atol_desc = (
        f"global override = {atol}" if atol is not None
        else "per-field default（见 _DEFAULT_PER_FIELD_ATOL）"
    )
    diff_header = [
        f"PyTorch FullNetV2 vs ONNX runtime ({onnx_path.name})",
        f"mode            = {mode}",
        f"B               = {B}",
        f"gnn2_enabled    = {gnn2_enabled}",
        f"expected inputs = {expected_names}",
        f"atol            = {atol_desc}",
        f"rtol            = {rtol}",
        "",
        "字段对比（PASS = max|Δ| ≤ 该字段 atol 且无单边 NaN）：",
    ]
    (out_run_dir / "diff.txt").write_text(
        "\n".join(diff_header + diff_lines), encoding="utf-8",
    )

    # ---- vis_compare.png ----
    z_in = np.load(inputs_npz)
    _vis_compare(
        hist_traj=z_in["hist_traj"],
        position=z_in["position"],
        refined_torch=torch_fields["refined_traj"],
        refined_onnx=onnx_fields["refined_traj"],
        mode_prob_torch=torch_fields["mode_prob"],
        mode_prob_onnx=onnx_fields["mode_prob"],
        out_path=out_run_dir / "vis_compare.png",
        mode=mode,
    )

    print(f"[Test/ONNX] saved {out_run_dir}/{{inputs.npz, outputs_torch.npz, outputs_onnx.npz, diff.txt, vis_compare.png}}")
    print(f"[Test/ONNX] {mode}: " + ("PASS" if overall_ok else "FAIL"))
    for line in diff_lines:
        print("    " + line)
    return overall_ok


# ============================================================
# 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX 数值校验：PyTorch FullNetV2 vs onnxruntime；"
                    "复用 fusion/eval_vis 的输入做端到端逐字段对比。"
    )
    parser.add_argument(
        "--fusion-config", type=str,
        default=str(REPO_ROOT / "fusion" / "config.yaml"),
    )
    parser.add_argument(
        "--from-eval-vis", type=str, default="",
        help="指定 eval_vis 下某次 run（如 fusion/eval_vis/run_test_20260429_003430）；"
             "默认空 = 自动找 fusion/eval_vis/ 下最新的 run_test_*。",
    )
    parser.add_argument(
        "--out-root", type=str,
        default=str(REPO_ROOT / "fusion" / "eval_vis_onnx"),
        help="输出根目录；脚本会在此下创建 run_test_<timestamp>/{mode}/ 子目录。",
    )
    parser.add_argument(
        "--onnx-dir", type=str,
        default=str(REPO_ROOT / "fusion"),
        help="ONNX 文件所在目录（默认 fusion/，与 export_onnx.py 默认输出对齐）。",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["no_road", "with_road", "both"],
    )
    parser.add_argument(
        "--atol", type=float, default=None,
        help="若设置，所有字段统一用此 atol（强制覆盖；适合做 1e-5 等价性测试）；"
             "默认 None = 按字段量纲分组（见脚本内 _DEFAULT_PER_FIELD_ATOL）。",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    fusion_cfg_path = Path(args.fusion_config).resolve()
    fusion_cfg, _ = load_fusion_config(fusion_cfg_path)
    fut_len = int((fusion_cfg.get("full_net", {}) or {}).get("fut_len", 10))

    # eval_vis 定位
    if args.from_eval_vis:
        eval_vis_run_dir = Path(args.from_eval_vis)
        if not eval_vis_run_dir.is_absolute():
            eval_vis_run_dir = (REPO_ROOT / eval_vis_run_dir).resolve()
        if not eval_vis_run_dir.exists():
            raise FileNotFoundError(
                f"--from-eval-vis 指向的目录不存在: {eval_vis_run_dir}"
            )
    else:
        eval_vis_root = REPO_ROOT / "fusion" / "eval_vis"
        eval_vis_run_dir = _find_latest_eval_vis(eval_vis_root)
        print(f"[Test/ONNX] auto-pick latest eval_vis run: {eval_vis_run_dir}")

    # 输出目录
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parent_run_dir = out_root / f"run_test_{stamp}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Test/ONNX] out parent run_dir = {parent_run_dir}")

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.is_absolute():
        onnx_dir = (REPO_ROOT / onnx_dir).resolve()

    modes = (args.mode,) if args.mode in ("no_road", "with_road") else VALID_MODES
    all_ok = True
    for mode in modes:
        ok = _run_single_mode(
            mode=mode,
            fusion_cfg=fusion_cfg,
            onnx_dir=onnx_dir,
            eval_vis_run_dir=eval_vis_run_dir,
            out_run_dir=parent_run_dir / mode,
            fut_len=fut_len,
            atol=args.atol, rtol=args.rtol,
        )
        all_ok = all_ok and ok

    print("=" * 64)
    print(f"[Test/ONNX] OVERALL = {'PASS' if all_ok else 'FAIL'}")
    print(f"[Test/ONNX] artifacts under {parent_run_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
