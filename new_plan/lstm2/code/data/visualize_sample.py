#!/usr/bin/env python3
"""
lstm2/code/data/visualize_sample.py
-----------------------------------
从 lstm2/data/raw/{split}.npz 随机抽 N 个 gnn1-sample，画 (hist + 3 条 refined +
position + 3 份路网) 的可视化。

预测轨迹只画线，不画点。路网用类似"高亮粗线段"的形式叠加。

用法（在 new_plan/lstm2 下）：
    $env:PYTHONPATH = "$PWD/code;$PWD/.."
    python -m data.visualize_sample --split val --n-samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt   # noqa: E402

INTENT_NAME = {0: "ATTACK", 1: "EVASION", 2: "DEFENSE", 3: "RETREAT"}

# 3 条候选用 3 种颜色
CAND_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]
# 路网用同色但更浅 + 加粗，和候选轨迹形成"高亮"叠加效果
ROAD_ALPHA = 0.35
ROAD_LW = 6.0


def _draw_road_network(
    ax,
    road_points: np.ndarray,    # [NB, NP, 3]
    road_mask: np.ndarray,      # [NB, NP] bool
    color: str,
    label: Optional[str] = None,
) -> None:
    """
    把一份路网（多分支折线）画成"高亮粗线段"。同分支相邻两个有效点之间连线。
    """
    NB, NP = road_mask.shape
    drawn_label = False
    for bi in range(NB):
        m = road_mask[bi]
        n_valid = int(m.sum())
        if n_valid < 2:
            continue
        pts = road_points[bi][m]   # [n_valid, 3]
        ax.plot(
            pts[:, 0], pts[:, 1],
            "-", color=color, lw=ROAD_LW, alpha=ROAD_ALPHA, solid_capstyle="round",
            label=(label if not drawn_label else None),
            zorder=1,
        )
        drawn_label = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="可视化的 gnn1 sample 数（每个 sample 画 K=3 条候选）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="vis")
    parser.add_argument("--out-name", type=str, default="vis_{split}.png")
    args = parser.parse_args()

    lstm2_root = Path(__file__).resolve().parents[2]   # .../new_plan/lstm2
    npz_path = lstm2_root / "data" / "raw" / f"{args.split}.npz"
    if not npz_path.exists():
        print(f"找不到 {npz_path}，请先跑 generate_trajs.py")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    hist = data["hist_raw"]            # [M, 20, 6]
    fut = data["fut_refined"]          # [M, 10, 6]
    pos = data["position"]             # [M, 3]
    intent = data["intent_label"]      # [M]
    threat = data["threat_score"]      # [M]
    topology = data["topology"]        # [M] str（per-candidate）
    sample_idx = data["sample_idx"]    # [M] int  原 gnn1 sample id
    cand_k = data["cand_k"]            # [M] int  在 K 中的位置
    fut_gt = data["fut_gt"] if "fut_gt" in data.files else None
    have_roads = ("road_points" in data.files) and ("road_mask" in data.files)
    road_points = data["road_points"] if have_roads else None  # [M, NB, NP, 3]
    road_mask = data["road_mask"] if have_roads else None      # [M, NB, NP]

    # 每个 gnn1 sample 对应 K=3 行；按 sample_idx 聚合
    uniq = np.unique(sample_idx)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(uniq, size=min(args.n_samples, len(uniq)), replace=False)

    out_dir = lstm2_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(pick)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, sid in enumerate(pick):
        rows = np.where(sample_idx == sid)[0]
        rows_sorted = rows[np.argsort(cand_k[rows])]   # 按 k=0,1,2 排序
        ax = axes[0, i]

        # 1) 路网（最先画，最底层，用浅色粗线）
        if have_roads:
            for j, ridx in enumerate(rows_sorted):
                _draw_road_network(
                    ax,
                    road_points=road_points[ridx],
                    road_mask=road_mask[ridx],
                    color=CAND_COLORS[j],
                    label=None,
                )

        # 2) 历史轨迹
        h = hist[rows_sorted[0]]                       # [20, 6]
        ax.plot(h[:, 0], h[:, 1], "-", color="gray", lw=2, label="hist", zorder=3)
        ax.scatter(h[-1, 0], h[-1, 1], color="black", s=40, zorder=6, label="now (0,0)")

        # 3) 我方目标位置
        p = pos[rows_sorted[0]]
        ax.scatter(p[0], p[1], color="red", marker="*", s=180, zorder=7, label="our target")

        # 4) 3 条候选预测轨迹（线，无点）
        legend_lines = []
        for j, ridx in enumerate(rows_sorted):
            f = fut[ridx]                              # [10, 6]
            it = INTENT_NAME[int(intent[ridx])]
            tr = int(threat[ridx])
            tp_name = str(topology[ridx])
            line, = ax.plot(
                f[:, 0], f[:, 1], "-",
                color=CAND_COLORS[j], lw=2.0, zorder=5,
                label=f"k={int(cand_k[ridx])}  {it}  thr={tr}  road={tp_name}",
            )
            legend_lines.append(line)

        # 5) GT 未来（虚线参考）
        if fut_gt is not None:
            g = fut_gt[rows_sorted[0]]
            ax.plot(g[:, 0], g[:, 1], "--", color="purple", lw=1.5, alpha=0.8,
                    label="GT future", zorder=4)

        ax.set_title(f"sample {int(sid)}")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"lstm2 vis  split={args.split}  n_samples={n}  "
                 f"(per-candidate road: {'yes' if have_roads else 'unknown'})",
                 fontsize=12)
    fig.tight_layout()
    out_png = out_dir / args.out_name.format(split=args.split)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
