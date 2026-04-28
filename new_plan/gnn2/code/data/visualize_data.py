#!/usr/bin/env python3
"""
gnn2/code/data/visualize_data.py
--------------------------------
可视化 GNN2 训练样本（由 generate_data.py 产出的 data/raw/{split}.npz）。

每个子图画一条 gnn2 训练样本，包含：
  - 原点 (0, 0)：敌方"现在"位置（hist 末帧；gnn2 数据约定原点）
  - pred_traj：敌方未来 10 步预测轨迹（路网约束后），蓝色折线 + 时间渐变点
                * 每个点旁注 t=60s, 120s, ..., 600s
                * 离 eta 最近的那个时间步用大圆点突出
  - gt_strike_pos：模型应该输出的领前打击点 = pred_traj 在 t=eta 的位置（红色五角星）
  - radius 圆：绕 strike_pos 的虚线圆，半径 = gt_strike_radius
                * 圆的填充透明度 ∝ gt_strike_conf（conf 越高越实）
  - 文本注释：eta(秒) / cand_k / radius(km) / conf / top_prob / src_split / gnn1_top_idx

为了可视化方便，所有图都用 xy 投影（top-down view），轴单位 km。
高度 z 维度通常很小，单图里不画。

用法（在 new_plan/gnn2/ 下）::

    # Windows PowerShell
    $env:PYTHONPATH = "$PWD/code"
    # Linux/macOS
    # export PYTHONPATH="$PWD/code"

    # 默认从 val 随机挑 6 个样本画：
    python -m data.visualize_data --split val

    # 指定数量 / 种子 / 输出目录：
    python -m data.visualize_data --split test --n 9 --seed 123
    python -m data.visualize_data --split train --indices 0,17,42,99
    python -m data.visualize_data --split val --n 6 --outdir vis

    # 按 eta 分桶展示（每个桶各画一个样本）：
    python -m data.visualize_data --split val --by-eta

    # 按 cand_k (top-1/2/3) 分组展示（同一 scene 的 3 条候选）：
    python -m data.visualize_data --split val --by-cand-k --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_indices(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def pick_indices(n_samples: int, n_want: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n_want = min(n_want, n_samples)
    return sorted(rng.choice(n_samples, size=n_want, replace=False).tolist())


def pick_indices_by_eta_bins(
    eta_arr: np.ndarray,
    n_bins: int,
    eta_min: int,
    eta_max: int,
    seed: int,
) -> List[int]:
    """每个 eta 桶随机挑 1 条样本。返回的索引顺序按桶递增。"""
    rng = np.random.default_rng(seed)
    edges = np.linspace(eta_min, eta_max, n_bins + 1, dtype=np.int64)
    chosen: List[int] = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1]) + (1 if i == n_bins - 1 else 0)
        m = (eta_arr >= lo) & (eta_arr < hi)
        idxs = np.nonzero(m)[0]
        if idxs.size == 0:
            continue
        chosen.append(int(rng.choice(idxs)))
    return chosen


def pick_indices_by_cand_k(
    cand_k_arr: np.ndarray,
    scene_idx_arr: np.ndarray,
    seed: int,
) -> List[int]:
    """随机挑一个 scene，把它的 3 条 top-K 各取一条 eta 副本，按 k=0/1/2 排序。"""
    rng = np.random.default_rng(seed)
    unique_scenes = np.unique(scene_idx_arr)
    if unique_scenes.size == 0:
        return []
    s = int(rng.choice(unique_scenes))
    chosen: List[int] = []
    for k in range(3):
        m = (scene_idx_arr == s) & (cand_k_arr == k)
        idxs = np.nonzero(m)[0]
        if idxs.size == 0:
            continue
        chosen.append(int(rng.choice(idxs)))
    return chosen


def grid_layout(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, int(np.ceil(n / 2))
    if n <= 9:
        return 3, int(np.ceil(n / 3))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


# ---------------------------------------------------------------------
# 单个样本画图
# ---------------------------------------------------------------------

SRC_SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def _plot_one_sample(
    ax: plt.Axes,
    sample_idx: int,
    pred_traj: np.ndarray,           # [T, 6]   物理 km / km·s⁻¹
    eta_sec: int,
    gt_strike_pos: np.ndarray,       # [3]      km xyz
    gt_strike_radius: float,         # km
    gt_strike_conf: float,           # ∈ [0, 1]
    cand_k: int,
    eta_idx: int,
    gnn1_top_idx: int,
    top_prob: float,
    src_split: int,
    scene_idx: int,
    time_step_s: float,
) -> None:
    T, D = pred_traj.shape
    xy = pred_traj[:, :2]                                       # [T, 2]
    times_sec = np.array([(t + 1) * time_step_s for t in range(T)], dtype=np.float64)

    # 0) 先 plot 浅色折线
    ax.plot(
        xy[:, 0], xy[:, 1],
        color="tab:blue", linewidth=1.6, alpha=0.7, zorder=2,
    )

    # 1) 各时间步打点（颜色随时间渐变）
    cmap = plt.get_cmap("Blues")
    for t in range(T):
        c = cmap(0.35 + 0.55 * (t / max(1, T - 1)))             # 由浅到深
        ax.scatter(
            xy[t, 0], xy[t, 1], s=42, color=c,
            edgecolors="black", linewidths=0.6, zorder=4,
        )
        ax.annotate(
            f"{int(times_sec[t])}s",
            xy=(xy[t, 0], xy[t, 1]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6, alpha=0.7, zorder=5,
        )

    # 2) 离 eta 最近的那个 step 突出（仅当 eta 落在 [60s, T*time_step_s] 内）
    step_float = eta_sec / time_step_s - 1.0                    # eta=60 → step 0
    if -1.0 <= step_float <= T - 1:
        # 圆点高亮：取最近整数步
        nearest = int(np.clip(np.round(step_float), 0, T - 1))
        ax.scatter(
            xy[nearest, 0], xy[nearest, 1],
            s=180, facecolors="none", edgecolors="orange",
            linewidths=2.2, zorder=6,
            label=f"step≈eta (t={int(times_sec[nearest])}s)",
        )

    # 3) 原点（hist 末帧 = 敌方"现在"位置）
    ax.scatter(0, 0, s=80, color="black", marker="*", zorder=7,
               label="enemy NOW (hist end)")

    # 4) gt_strike_pos：领前打击点 = 敌方在 t=eta 的位置
    ax.scatter(
        gt_strike_pos[0], gt_strike_pos[1], s=260, color="red",
        marker="*", edgecolors="black", linewidths=1.4, zorder=8,
        label=f"strike_pos @ t={eta_sec}s",
    )

    # 5) radius 圆（透明度 ∝ conf）
    edge_alpha = 0.3 + 0.7 * float(gt_strike_conf)
    fill_alpha = 0.05 + 0.20 * float(gt_strike_conf)
    circ = Circle(
        (gt_strike_pos[0], gt_strike_pos[1]),
        radius=gt_strike_radius,
        fill=True, facecolor="red", alpha=fill_alpha,
        edgecolor="red", linestyle="--", linewidth=1.4, zorder=3,
    )
    # alpha 设到 edgecolor 不行，得 set 单独
    circ.set_alpha(None)
    circ.set_facecolor((1.0, 0.2, 0.2, fill_alpha))
    circ.set_edgecolor((0.8, 0.0, 0.0, edge_alpha))
    ax.add_patch(circ)

    # 6) 标题 + 文本
    src_name = SRC_SPLIT_NAMES.get(int(src_split), str(src_split))
    title = (
        f"sample {sample_idx}  scene={scene_idx}  "
        f"cand_k={cand_k} (top-{cand_k + 1}, gnn1_idx={gnn1_top_idx})\n"
        f"eta = {eta_sec} s  (eta_idx={eta_idx})   |   "
        f"radius = {gt_strike_radius:.3f} km   conf = {gt_strike_conf:.3f}\n"
        f"top_prob (gnn1) = {top_prob:.3f}   src_split = {src_name}"
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)


# ---------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------

def visualize(
    config_path: Path,
    split: str,
    n_want: int,
    indices: Optional[List[int]],
    seed: int,
    outdir: Path,
    show: bool,
    by_eta: bool,
    by_cand_k: bool,
) -> None:
    gnn2_root = Path(__file__).resolve().parents[2]   # .../gnn2

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {}) or {}

    raw_dir = (gnn2_root / data_cfg.get("raw_dir", "data/raw")).resolve()
    npz_path = raw_dir / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"找不到 {npz_path}；请先在 gnn2/ 下跑 "
            "`python -m data.generate_data --config config.yaml`"
        )

    data = np.load(npz_path)
    required = ["pred_traj", "eta", "gt_strike_pos", "gt_strike_radius", "gt_strike_conf"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"{npz_path} 缺字段 {k}")

    pred_traj = data["pred_traj"]                                   # [N, 10, 6]
    eta_arr = data["eta"]                                           # [N]
    gt_pos = data["gt_strike_pos"]                                  # [N, 3]
    gt_radius = data["gt_strike_radius"]                            # [N]
    gt_conf = data["gt_strike_conf"]                                # [N]

    # 诊断字段（缺失时填 -1 / 0）
    cand_k_arr = data["cand_k"] if "cand_k" in data.files else np.zeros(eta_arr.shape, np.int8)
    eta_idx_arr = data["eta_idx"] if "eta_idx" in data.files else np.zeros(eta_arr.shape, np.int8)
    gnn1_top_idx_arr = (
        data["gnn1_top_idx"] if "gnn1_top_idx" in data.files
        else np.full(eta_arr.shape, -1, np.int8)
    )
    top_prob_arr = (
        data["top_prob"] if "top_prob" in data.files
        else np.zeros(eta_arr.shape, np.float32)
    )
    src_split_arr = (
        data["src_split"] if "src_split" in data.files
        else np.zeros(eta_arr.shape, np.int8)
    )
    scene_idx_arr = (
        data["scene_idx"] if "scene_idx" in data.files
        else np.arange(eta_arr.shape[0], dtype=np.int32)
    )

    N = int(pred_traj.shape[0])
    print(f"[vis] {npz_path}: {N} samples")

    time_step_s = float(data_cfg.get("time_step_s", 60.0))
    eta_min_sec = int(data_cfg.get("eta_min_sec", 0))
    eta_max_sec = int(data_cfg.get("eta_max_sec", 600))

    # ---- 选样本 ----
    if indices is not None:
        for i in indices:
            if not (0 <= i < N):
                raise ValueError(f"索引 {i} 越界 [0, {N})")
        chosen = list(indices)
        mode = "indices"
    elif by_eta:
        chosen = pick_indices_by_eta_bins(
            eta_arr=eta_arr,
            n_bins=max(1, n_want),
            eta_min=eta_min_sec,
            eta_max=eta_max_sec,
            seed=seed,
        )
        mode = f"by-eta (n_bins={n_want})"
    elif by_cand_k:
        chosen = pick_indices_by_cand_k(
            cand_k_arr=cand_k_arr,
            scene_idx_arr=scene_idx_arr,
            seed=seed,
        )
        mode = "by-cand-k (same scene, k=0/1/2)"
    else:
        chosen = pick_indices(N, n_want, seed)
        mode = "random"

    if not chosen:
        print(f"[vis] 没选到样本（mode={mode}）")
        return
    print(f"[vis] mode={mode}  picked {len(chosen)} samples: {chosen}")

    # ---- 画布 ----
    n_plots = len(chosen)
    rows, cols = grid_layout(n_plots)
    fig_w = max(5.0 * cols, 6.0)
    fig_h = max(4.8 * rows, 5.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for k, idx in enumerate(chosen):
        ax = axes[k]
        _plot_one_sample(
            ax=ax,
            sample_idx=int(idx),
            pred_traj=pred_traj[idx],
            eta_sec=int(eta_arr[idx]),
            gt_strike_pos=gt_pos[idx],
            gt_strike_radius=float(gt_radius[idx]),
            gt_strike_conf=float(gt_conf[idx]),
            cand_k=int(cand_k_arr[idx]),
            eta_idx=int(eta_idx_arr[idx]),
            gnn1_top_idx=int(gnn1_top_idx_arr[idx]),
            top_prob=float(top_prob_arr[idx]),
            src_split=int(src_split_arr[idx]),
            scene_idx=int(scene_idx_arr[idx]),
            time_step_s=time_step_s,
        )

    for k in range(n_plots, rows * cols):
        axes[k].axis("off")

    suptitle = (
        f"GNN2 training samples | split={split} | n={n_plots} | mode={mode} | "
        f"eta∈[{eta_min_sec}, {eta_max_sec}]s | dt={time_step_s}s/step"
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if not outdir.is_absolute():
        outdir = (gnn2_root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode_tag = mode.split()[0].replace("(", "").replace(")", "")
    out_path = outdir / f"gnn2_samples_{split}_{mode_tag}_{stamp}.png"
    fig.savefig(out_path, dpi=140)
    print(f"[vis] saved: {out_path}")
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
            plt.show()
        except Exception as e:
            print(f"[vis] show 失败（可能没 GUI 后端）：{e}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GNN2 training samples.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="gnn2 config.yaml 路径")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=6,
                        help="挑几个样本（被 --indices 覆盖）；by-eta 模式下作为桶数")
    parser.add_argument("--indices", type=str, default="",
                        help="逗号分隔的样本索引，例如 0,17,42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="vis",
                        help="输出 png 的目录（相对 gnn2 根目录）")
    parser.add_argument("--show", action="store_true",
                        help="画完后 plt.show()（需要 GUI 后端，默认不开）")
    parser.add_argument("--by-eta", action="store_true",
                        help="按 eta 桶各挑 1 条样本（n 控制桶数；用来看 eta 不同 → strike_pos 不同）")
    parser.add_argument("--by-cand-k", action="store_true",
                        help="同一 scene 的 top-1/2/3 候选各挑 1 条（用来看 GNN1 选轨差异）")
    args = parser.parse_args()

    gnn2_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = gnn2_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config 不存在: {cfg_path}")

    indices = parse_indices(args.indices) if args.indices else None
    if args.by_eta and args.by_cand_k:
        raise ValueError("--by-eta 和 --by-cand-k 不能同时开")

    visualize(
        config_path=cfg_path,
        split=args.split,
        n_want=int(args.n),
        indices=indices,
        seed=int(args.seed),
        outdir=Path(args.outdir),
        show=bool(args.show),
        by_eta=bool(args.by_eta),
        by_cand_k=bool(args.by_cand_k),
    )


if __name__ == "__main__":
    main()
