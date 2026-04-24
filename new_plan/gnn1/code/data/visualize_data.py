#!/usr/bin/env python3
"""
gnn1/code/data/visualize_data.py
--------------------------------
可视化 GNN1 训练样本（由 generate_data.py 产出的 data/raw/{split}.npz）。

每个子图里画：
  - 原点 (0,0)：对应 window 的 hist 最后一帧（五角小黑点）
  - 5 条候选轨迹（不同颜色的折线）：
      * label 对应那条：加粗 + 实心圆端点
      * k_seed 对应那条：方块端点
      * 其余：普通线 + 小三角端点
  - position（我方固定目标）：红色大五角星
  - position 周围画一个半径 = position_noise_km (默认 0.3) 的虚线圆，
    直观表示合成噪声量级
  - 若 npz 里保存了 targets（GT 未来），画黑色虚线 + 小圆点

用法（在 new_plan/gnn1/ 下）：
    # Windows PowerShell
    $env:PYTHONPATH = "$PWD/code"
    # Linux/macOS
    # export PYTHONPATH="$PWD/code"

    # 从 test split 随机挑 5 个样本画：
    python -m data.visualize_data --split test

    # 指定数量 / 指定索引 / 指定输出目录：
    python -m data.visualize_data --split val --n 8 --seed 123
    python -m data.visualize_data --split train --indices 0,17,42,99,501
    python -m data.visualize_data --split test --n 5 --outdir vis
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
# Scaler：和 generate_data.py / eval.py 同口径
# ---------------------------------------------------------------------

class _Scaler:
    """最小版 StandardScaler.inverse_transform；和 lstm1 的 npz 兼容。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        m = self.mean.reshape((1,) * (x.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (x.ndim - 1) + (-1,))
        return x * s + m

    @classmethod
    def load(cls, path: Path) -> "_Scaler":
        a = np.load(path)
        return cls(a["mean"], a["std"])


def decode_to_xy(feat_norm: np.ndarray, scaler: _Scaler) -> np.ndarray:
    """[*, T, D] 归一化+delta → [*, T, 2] xy（以 (0,0) 为原点做 cumsum）"""
    orig = scaler.inverse_transform(feat_norm.astype(np.float64))
    dxy = orig[..., :2]
    return np.cumsum(dxy, axis=-2)


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


def grid_layout(n: int) -> Tuple[int, int]:
    """选一个比较方的 (rows, cols) 布局。"""
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

def _plot_one_sample(
    ax: plt.Axes,
    sample_idx: int,
    cand_xy: np.ndarray,          # [M, T, 2]
    position: np.ndarray,         # [3]
    label: int,
    k_seed: int,
    task_type: int,
    type_id: int,
    gt_xy: Optional[np.ndarray],  # [T, 2] or None
    noise_sigma_km: float,
    palette: List[str],
) -> None:
    M, T, _ = cand_xy.shape

    # 1) 候选轨迹（5 条）
    for m in range(M):
        is_label = (m == label)
        is_seed = (m == k_seed)

        color = palette[m % len(palette)]
        lw = 2.6 if is_label else 1.2
        alpha = 1.0 if is_label else 0.75

        ax.plot(
            cand_xy[m, :, 0], cand_xy[m, :, 1],
            color=color, linewidth=lw, alpha=alpha,
            label=f"cand {m}" + ("  <label>" if is_label else ""),
            zorder=3 if is_label else 2,
        )

        # 端点
        ex, ey = cand_xy[m, -1, 0], cand_xy[m, -1, 1]
        if is_label:
            ax.scatter(ex, ey, s=85, color=color, edgecolors="black",
                       linewidths=1.2, marker="o", zorder=5)
        elif is_seed:
            ax.scatter(ex, ey, s=55, color=color, edgecolors="black",
                       linewidths=1.0, marker="s", zorder=4)
        else:
            ax.scatter(ex, ey, s=35, color=color, marker="^",
                       alpha=0.85, zorder=4)

    # 2) 原点（hist 最后一帧）
    ax.scatter(0, 0, s=60, color="black", marker="*", zorder=6,
               label="hist last")

    # 3) position（我方固定目标）
    ax.scatter(
        position[0], position[1], s=240, color="red",
        marker="*", edgecolors="black", linewidths=1.2, zorder=7,
        label=f"position (type={type_id})",
    )
    # 噪声圈
    if noise_sigma_km > 0:
        circ = Circle(
            (position[0], position[1]),
            radius=noise_sigma_km,
            fill=False, linestyle="--", linewidth=1.0,
            edgecolor="red", alpha=0.6, zorder=1,
        )
        ax.add_patch(circ)

    # 4) GT 未来轨迹（可选）
    if gt_xy is not None:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1],
                color="black", linestyle="--", linewidth=1.5,
                alpha=0.9, zorder=3, label="GT future")
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1],
                   s=15, color="black", marker="o", zorder=4)

    # 5) 标题 + 距离信息
    ep_label = cand_xy[label, -1, :2]
    ep_seed = cand_xy[k_seed, -1, :2]
    pos_xy = position[:2]
    d_label = float(np.linalg.norm(ep_label - pos_xy))
    d_seed = float(np.linalg.norm(ep_seed - pos_xy))
    flipped = (label != k_seed)

    title = (
        f"idx={sample_idx}  task={task_type}  type={type_id}\n"
        f"k_seed={k_seed}  label={label}"
        f"{'  (flipped)' if flipped else ''}\n"
        f"|ep_label→pos|={d_label:.3f} km   "
        f"|ep_seed→pos|={d_seed:.3f} km"
    )
    ax.set_title(title, fontsize=9)
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
) -> None:
    gnn1_root = Path(__file__).resolve().parents[2]  # .../gnn1

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {}) or {}

    raw_dir = (gnn1_root / data_cfg.get("raw_dir", "data/raw")).resolve()
    cache_dir = (gnn1_root / data_cfg.get("cache_dir", "data/cache")).resolve()
    npz_path = raw_dir / f"{split}.npz"
    scaler_path = cache_dir / "scaler_posvel.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"找不到 {npz_path}；请先跑 cache_lstm1_preds.py + generate_data.py"
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"找不到 {scaler_path}；请先跑 cache_lstm1_preds.py（它会落 scaler）"
        )

    data = np.load(npz_path)
    for k in ("candidates", "task_type", "type", "position", "label"):
        if k not in data.files:
            raise KeyError(f"{npz_path} 缺字段 {k}")

    candidates = data["candidates"]    # [N, M, T, D]
    task_arr = data["task_type"]
    type_arr = data["type"]
    pos_arr = data["position"]         # [N, 3]
    label_arr = data["label"]
    k_seed_arr = data["k_seed"] if "k_seed" in data.files else np.full_like(label_arr, -1)
    have_targets = "targets" in data.files
    targets_arr = data["targets"] if have_targets else None

    scaler = _Scaler.load(scaler_path)
    noise_sigma_km = float(data_cfg.get("position_noise_km", 0.3))

    # 选样本索引
    N = int(candidates.shape[0])
    if indices is not None:
        for i in indices:
            if not (0 <= i < N):
                raise ValueError(f"索引 {i} 越界 [0, {N})")
        chosen = list(indices)
    else:
        chosen = pick_indices(N, n_want, seed)

    # 批量解码到 xy
    cand_sel = candidates[chosen]                        # [n, M, T, D]
    cand_xy_all = decode_to_xy(cand_sel, scaler)         # [n, M, T, 2]
    if have_targets:
        tgt_sel = targets_arr[chosen]                    # [n, T, D]
        gt_xy_all = decode_to_xy(tgt_sel, scaler)        # [n, T, 2]
    else:
        gt_xy_all = None

    # 画布
    n_plots = len(chosen)
    rows, cols = grid_layout(n_plots)
    fig_w = max(4.8 * cols, 5.5)
    fig_h = max(4.6 * rows, 4.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    for k, idx in enumerate(chosen):
        ax = axes[k]
        _plot_one_sample(
            ax=ax,
            sample_idx=int(idx),
            cand_xy=cand_xy_all[k],
            position=pos_arr[idx],
            label=int(label_arr[idx]),
            k_seed=int(k_seed_arr[idx]),
            task_type=int(task_arr[idx]),
            type_id=int(type_arr[idx]),
            gt_xy=(gt_xy_all[k] if have_targets else None),
            noise_sigma_km=noise_sigma_km,
            palette=palette,
        )

    # 关闭多余 subplot
    for k in range(n_plots, rows * cols):
        axes[k].axis("off")

    suptitle = (
        f"GNN1 training samples | split={split} | n={n_plots}"
        f" | σ={noise_sigma_km} km | GT={'yes' if have_targets else 'no'}"
    )
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if not outdir.is_absolute():
        outdir = (gnn1_root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"gnn1_samples_{split}_{stamp}.png"
    fig.savefig(out_path, dpi=140)
    print(f"[vis] saved: {out_path}")
    print(f"[vis] indices = {chosen}")
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
            plt.show()
        except Exception as e:
            print(f"[vis] show 失败（可能没 GUI 后端）：{e}")

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GNN1 training samples.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="gnn1 config.yaml 路径")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=5,
                        help="随机挑几个样本（被 --indices 覆盖）")
    parser.add_argument("--indices", type=str, default="",
                        help="逗号分隔的样本索引，例如 0,17,42")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机挑样本用的种子")
    parser.add_argument("--outdir", type=str, default="vis",
                        help="输出 png 的目录（相对 gnn1 根目录）")
    parser.add_argument("--show", action="store_true",
                        help="画完后 plt.show()（需要 GUI 后端，默认不开）")
    args = parser.parse_args()

    gnn1_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config 不存在: {cfg_path}")

    indices = parse_indices(args.indices) if args.indices else None

    visualize(
        config_path=cfg_path,
        split=args.split,
        n_want=int(args.n),
        indices=indices,
        seed=int(args.seed),
        outdir=Path(args.outdir),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
