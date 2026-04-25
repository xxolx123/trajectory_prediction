"""
constraint_optimizer/test_road_net/test_projection.py
-----------------------------------------------------
端到端测试：GNN1 选 top-K 候选 → 用合成路网（甲方 LLH 接口格式）→
ConstraintOptimizer.road_projection 投影 → 可视化对比。

每个样本一张子图，画：
  - 5 条 LSTM1 候选（faded 灰色）
  - GNN1 选出的 top-K（K=3）原始轨迹（虚线，分色 + 端点星号）
  - 合成路网（粗灰色折线 + 节点圆点；每条分支不同灰阶）
  - 投影后的 top-K 轨迹（同色实线 + 端点正方形）
  - 我方固定目标 position（红色五角星）
  - 起点 origin（黑色五角星）

用法（在 new_plan/ 下，激活 conda 环境后）::

    $env:PYTHONPATH = "$PWD"
    python -m constraint_optimizer.test_road_net.test_projection \
        --gnn1-ckpt new_plan/gnn1/checkpoints/20260426001534/best_gnn1_epoch001_valloss0.9508.pt \
        --split test --n 6 --out constraint_optimizer/test_road_net/vis

如果 --gnn1-ckpt 不指定，会从 gnn1/checkpoints 里挑最新的 .pt。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
# 中文字体（Windows 上常见的 CJK 字体；按可用性自动 fallback）
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle  # noqa: F401  (留给后续画 noise 圈备用)


# 把 new_plan/ 加到 sys.path
_THIS_FILE = Path(__file__).resolve()
# .../new_plan/constraint_optimizer/test_road_net/test_projection.py
#   parents[0]=test_road_net  parents[1]=constraint_optimizer  parents[2]=new_plan
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.context_schema import build_dummy_context  # noqa: E402

from constraint_optimizer.code.train.module import ConstraintOptimizer  # noqa: E402
from constraint_optimizer.test_road_net.road_schema import (  # noqa: E402
    RoadNetwork,
    road_network_to_tensors,
    road_network_summary,
)
from constraint_optimizer.test_road_net.synth_road import (  # noqa: E402
    build_road_network_for_sample,
)

# gnn1 相关 import 用其 code 目录
_GNN1_CODE = _REPO_ROOT / "gnn1" / "code"
if str(_GNN1_CODE) not in sys.path:
    sys.path.insert(0, str(_GNN1_CODE))

from data.dataset import build_datasets_from_config  # noqa: E402
from train.model import build_model_from_config       # noqa: E402


# ============================================================
# 工具
# ============================================================

class _Scaler:
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


def _decode_to_phys_xyz_vel(feat_norm: np.ndarray, scaler: _Scaler) -> np.ndarray:
    """
    [..., T, 6] (归一化 + delta 空间) → [..., T, 6] 物理空间，
    其中前 3 维是绝对 xyz（cumsum 自原点 (0,0,0)），后 3 维是 vel。
    """
    orig = scaler.inverse_transform(feat_norm.astype(np.float64))
    pos_delta = orig[..., :3]
    vel = orig[..., 3:6]
    pos_abs = np.cumsum(pos_delta, axis=-2)
    return np.concatenate([pos_abs, vel], axis=-1)


def _find_latest_ckpt(ckpt_root: Path) -> Optional[Path]:
    if ckpt_root.is_file():
        return ckpt_root
    if not ckpt_root.exists():
        return None
    cands = list(ckpt_root.rglob("*.pt"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _setup_device(name: str) -> torch.device:
    n = name.lower()
    if n == "cpu":
        return torch.device("cpu")
    if n in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 可视化（重设计：每个样本一行，左=原始 / 右=投影）
# ============================================================

# top-K 三条候选用的颜色（红 / 橙 / 品红）—— 故意避开蓝/绿/紫，与路网色不冲突
_TOPK_COLORS = ["#D62728", "#FF7F0E", "#C71585", "#8B4513", "#2F4F4F"]

# 路网分支用的颜色（蓝 / 绿 / 紫）—— 鲜艳，能撑得住主背景
_BRANCH_COLORS = ["#1F77B4", "#2CA02C", "#9467BD", "#17BECF"]
_BRANCH_NAMES = [
    "支路 0（→top1 终点）",
    "支路 1（→top2 终点）",
    "支路 2（→top3 终点）",
    "支路 3",
]


def _draw_road_network(
    ax: plt.Axes,
    road_branches_enu: List[np.ndarray],
    label_legend: bool = True,
) -> None:
    """画路网分支：每条分支不同鲜艳色 + 节点圆点。"""
    for bi, br in enumerate(road_branches_enu):
        if br.shape[0] < 2:
            continue
        c = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        name = _BRANCH_NAMES[bi % len(_BRANCH_NAMES)]
        ax.plot(
            br[:, 0], br[:, 1],
            color=c, linewidth=3.2, alpha=0.9, zorder=2,
            solid_capstyle="round",
            label=(name if label_legend else None),
        )
        ax.scatter(
            br[:, 0], br[:, 1],
            s=28, color=c, edgecolors="white", linewidths=0.8, zorder=2,
        )


def _viewport_from(
    *arrays: np.ndarray,
    margin_frac: float = 0.10,
) -> Tuple[float, float, float, float]:
    """从若干 [...,2] 数组算 xy 视口 (xmin, xmax, ymin, ymax)，并加 margin。"""
    xs: List[float] = []
    ys: List[float] = []
    for a in arrays:
        if a is None or a.size == 0:
            continue
        a2 = a.reshape(-1, a.shape[-1])
        xs.extend(a2[:, 0].tolist())
        ys.extend(a2[:, 1].tolist())
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    return (
        xmin - dx * margin_frac, xmax + dx * margin_frac,
        ymin - dy * margin_frac, ymax + dy * margin_frac,
    )


def _draw_position(
    ax: plt.Axes,
    pos_xy: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """position（红色五角星）：在视口内就直接画；在外就在视口边沿画箭头指示。"""
    px, py = float(pos_xy[0]), float(pos_xy[1])
    inside = (xlim[0] <= px <= xlim[1]) and (ylim[0] <= py <= ylim[1])
    if inside:
        ax.scatter(
            px, py, s=220, color="red", marker="*",
            edgecolors="black", linewidths=1.2, zorder=9,
        )
        return

    # 不在视口内：在视口边沿画箭头朝着 position，并标注距离
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    dx, dy = px - cx, py - cy
    n = float(np.hypot(dx, dy)) + 1e-9
    ux, uy = dx / n, dy / n
    # 视口边沿点
    half_w = 0.5 * (xlim[1] - xlim[0])
    half_h = 0.5 * (ylim[1] - ylim[0])
    t = min(half_w / abs(ux + 1e-9), half_h / abs(uy + 1e-9)) * 0.92
    edge_x, edge_y = cx + t * ux, cy + t * uy
    arrow_x, arrow_y = edge_x - 0.08 * t * ux, edge_y - 0.08 * t * uy
    ax.annotate(
        f"→ position\n  ({px:.1f},{py:.1f}) km",
        xy=(edge_x, edge_y),
        xytext=(arrow_x, arrow_y),
        fontsize=8, color="red", weight="bold",
        ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.6),
    )


def _plot_one_sample_pair(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    sample_idx: int,
    top_phys: np.ndarray,             # [K, T, 6]   原始 top-K 物理坐标
    top_idx: np.ndarray,              # [K]
    top_probs: np.ndarray,            # [K]
    refined_xyz: np.ndarray,          # [K, T, 3]
    road_branches_enu: List[np.ndarray],
    position: np.ndarray,
    label: int,
    task_type: int,
    type_id: int,
) -> None:
    K = int(top_phys.shape[0])

    # 共同视口：只跟随轨迹，以"原始 + 投影 + origin"为参考；
    # 路网在视口外的部分由 matplotlib 自动裁剪掉，避免空白拉远整张图。
    origin_pt = np.array([[0.0, 0.0]], dtype=np.float64)
    xmin, xmax, ymin, ymax = _viewport_from(
        top_phys[..., :2],
        refined_xyz[..., :2],
        origin_pt,
        margin_frac=0.25,
    )

    def _draw_top_k_traj(
        ax: plt.Axes,
        traj_xy_arr: np.ndarray,        # [K, T, 2]
        end_marker: str,
        labels: List[str],
    ) -> None:
        """统一画 top-K 三条轨迹：粗实线 + 每个时间步小圆点 + 端点 marker。"""
        for rank in range(K):
            col = _TOPK_COLORS[rank % len(_TOPK_COLORS)]
            xy = traj_xy_arr[rank]
            # 粗实线
            ax.plot(
                xy[:, 0], xy[:, 1],
                color=col, linewidth=3.0, alpha=1.0, zorder=4,
                label=labels[rank],
                solid_capstyle="round",
            )
            # 每个时间步小圆点（让线即便很短或被遮住，也能看到一串点）
            ax.scatter(
                xy[:, 0], xy[:, 1],
                s=22, color=col, edgecolors="white", linewidths=0.6,
                zorder=5,
            )
            # 起点：黑边小圆（强调"从哪儿开始")
            ax.scatter(
                xy[0, 0], xy[0, 1],
                s=42, color=col, edgecolors="black", linewidths=1.0,
                marker="o", zorder=6,
            )
            # 末端 marker
            if end_marker == "*":
                ax.scatter(
                    xy[-1, 0], xy[-1, 1],
                    s=160, color=col, edgecolors="black", linewidths=1.2,
                    marker="*", zorder=7,
                )
            else:
                ax.scatter(
                    xy[-1, 0], xy[-1, 1],
                    s=140, color=col, edgecolors="black", linewidths=1.4,
                    marker="s", zorder=7,
                )

    # ========== 左图：原始 top-K ==========
    _draw_road_network(ax_left, road_branches_enu, label_legend=True)

    left_labels = [
        f"top{r+1}  cand{int(top_idx[r])}  p={float(top_probs[r])*100:.1f}%"
        for r in range(K)
    ]
    _draw_top_k_traj(ax_left, top_phys[..., :2], end_marker="*", labels=left_labels)

    ax_left.scatter(0, 0, s=70, color="black", marker="*", zorder=8)
    ax_left.set_xlim(xmin, xmax)
    ax_left.set_ylim(ymin, ymax)
    _draw_position(ax_left, position[:2], (xmin, xmax), (ymin, ymax))
    ax_left.set_title(
        f"idx={sample_idx}   原始 top-{K}（GNN1 选出）\n"
        f"task={task_type}  type={type_id}  label={label}",
        fontsize=10,
    )
    ax_left.set_xlabel("x (km, ENU)")
    ax_left.set_ylabel("y (km, ENU)")
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax_left.legend(fontsize=8, loc="best", framealpha=0.85)

    # ========== 右图：投影后 top-K ==========
    # 不再画原始的浅虚线 / 末端位移线，避免视觉干扰
    _draw_road_network(ax_right, road_branches_enu, label_legend=True)

    right_labels = [
        f"top{r+1}  cand{int(top_idx[r])}  p={float(top_probs[r])*100:.1f}%"
        for r in range(K)
    ]
    _draw_top_k_traj(ax_right, refined_xyz[..., :2], end_marker="s", labels=right_labels)

    # 计算平均位移用于标题
    shifts = [
        float(np.linalg.norm(refined_xyz[r, :, :2] - top_phys[r, :, :2], axis=-1).mean())
        for r in range(K)
    ]
    avg_shift = float(np.mean(shifts)) if shifts else 0.0

    ax_right.scatter(0, 0, s=70, color="black", marker="*", zorder=8)
    ax_right.set_xlim(xmin, xmax)
    ax_right.set_ylim(ymin, ymax)
    _draw_position(ax_right, position[:2], (xmin, xmax), (ymin, ymax))
    ax_right.set_title(
        f"idx={sample_idx}   投影后 top-{K}（贴路网）\n"
        f"平均位移 |Δxy| = {avg_shift:.3f} km",
        fontsize=10,
    )
    ax_right.set_xlabel("x (km, ENU)")
    ax_right.set_ylabel("y (km, ENU)")
    ax_right.set_aspect("equal", adjustable="box")
    ax_right.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax_right.legend(fontsize=8, loc="best", framealpha=0.85)


def _add_global_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=_BRANCH_COLORS[0], lw=3.2, label=_BRANCH_NAMES[0]),
        Line2D([0], [0], color=_BRANCH_COLORS[1], lw=3.2, label=_BRANCH_NAMES[1]),
        Line2D([0], [0], color=_BRANCH_COLORS[2], lw=3.2, label=_BRANCH_NAMES[2]),
        Line2D([0], [0], color=_TOPK_COLORS[0], lw=3.0,
               label="top-K 轨迹（粗实线，每步带圆点）"),
        Line2D([0], [0], color="black", marker="o", linestyle="None",
               markersize=6, label="起点（黑边小圆）"),
        Line2D([0], [0], color="black", marker="*", linestyle="None",
               markersize=10, label="左:原始末端 ★  /  origin"),
        Line2D([0], [0], color="black", marker="s", linestyle="None",
               markersize=8, label="右:投影末端 ■"),
        Line2D([0], [0], color="red", marker="*", linestyle="None",
               markersize=12, markeredgecolor="black",
               label="position（视口外用红色箭头标注）"),
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=3, fontsize=8.5, frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )


# ============================================================
# 主流程
# ============================================================

def run(
    gnn1_config_path: Path,
    gnn1_ckpt: Optional[Path],
    split: str,
    n: int,
    seed: int,
    out_dir: Path,
    device_name: str,
    origin_lon: float,
    origin_lat: float,
    origin_alt: float,
    nb_max: int,
    np_max: int,
    mode: str = "road_arc_projection",
) -> None:
    device = _setup_device(device_name)
    print(f"[Test/Road] device = {device}")

    # ---- 读 gnn1 config ----
    with gnn1_config_path.open("r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)

    # ---- 数据集 ----
    train_ds, val_ds, test_ds = build_datasets_from_config(str(gnn1_config_path))
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    print(f"[Test/Road] split={split}  size={len(ds)}")

    # ---- scaler（候选反归一化必备） ----
    gnn1_root = gnn1_config_path.resolve().parent
    cache_dir = (gnn1_root / gnn1_cfg.get("data", {}).get("cache_dir", "data/cache")).resolve()
    scaler_path = cache_dir / "scaler_posvel.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"找不到 scaler {scaler_path}；请先在 gnn1/ 下跑 cache_lstm1_preds.py"
        )
    scaler = _Scaler.load(scaler_path)
    print(f"[Test/Road] scaler 已加载 {scaler_path.name}")

    # ---- gnn1 模型 + ckpt ----
    gnn1 = build_model_from_config(gnn1_cfg).to(device)
    if gnn1_ckpt is None:
        ckpt_root = (gnn1_root / gnn1_cfg.get("train", {}).get("ckpt_dir", "checkpoints")).resolve()
        gnn1_ckpt = _find_latest_ckpt(ckpt_root)
        if gnn1_ckpt is None:
            raise FileNotFoundError(f"没找到 gnn1 ckpt（{ckpt_root}），请用 --gnn1-ckpt 指定")
    print(f"[Test/Road] 加载 GNN1 ckpt: {gnn1_ckpt}")
    state = torch.load(gnn1_ckpt, map_location=device)
    gnn1.load_state_dict(state)
    gnn1.eval()

    # ---- constraint optimizer ----
    constraint = ConstraintOptimizer(
        enable=True, module_type=mode,
    ).to(device).eval()
    print(f"[Test/Road] constraint_optimizer mode = {mode}")

    # ---- 挑 N 个样本 ----
    rng = np.random.default_rng(seed)
    n_want = min(n, len(ds))
    chosen = sorted(rng.choice(len(ds), size=n_want, replace=False).tolist())
    print(f"[Test/Road] 选样本: {chosen}")

    # 一次性 forward
    items = [ds[i] for i in chosen]
    batch = {
        k: torch.stack([b[k] for b in items], dim=0)
        for k in items[0].keys()
    }
    batch_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    with torch.no_grad():
        out = gnn1(batch_dev)
    top_idx_all = out["top_idx"].cpu().numpy()        # [n_want, K]
    top_probs_all = out["top_probs"].cpu().numpy()    # [n_want, K]
    K = int(top_idx_all.shape[-1])

    cand_norm = batch["cand_trajs"].numpy()           # [n_want, M, T, D]
    cand_phys = _decode_to_phys_xyz_vel(cand_norm, scaler)   # [n_want, M, T, 6]
    pos_arr = batch["position"].numpy()
    label_arr = batch["label"].numpy()
    task_arr = batch["task_type"].numpy()
    type_arr = batch["type"].numpy()

    # ---- 逐样本造路网、投影、画图 ----
    # 新版布局：每个样本一行，左 = 原始，右 = 投影
    rows = n_want
    cols = 2
    fig_w = 13.0
    fig_h = max(5.2 * rows, 5.6) + 0.6   # 留腿位给 legend
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.array(axes).reshape(1, 2)
    else:
        axes = np.asarray(axes).reshape(rows, 2)

    print()
    for k_i, sample_idx in enumerate(chosen):
        cand_phys_i = cand_phys[k_i]                  # [M, T, 6]
        top_idx_i = top_idx_all[k_i]                  # [K]
        top_probs_i = top_probs_all[k_i]

        # 取 top-K 候选的物理坐标（[K, T, 6]）
        top_phys_i = cand_phys_i[top_idx_i]           # [K, T, 6]

        # 用 top-K 候选造一个 K 叉路网：每条候选一条专属支路
        road_net: RoadNetwork = build_road_network_for_sample(
            cand_xyz_km=top_phys_i[..., :3],
            origin_llh=(origin_lon, origin_lat, origin_alt),
            n_branches=K,
        )
        if k_i == 0:
            print(road_network_summary(road_net))
            print()

        # LLH → 张量 [1, NB_max, NP_max, 3]
        rp_t, rm_t = road_network_to_tensors(
            road_net,
            origin_llh=(origin_lon, origin_lat, origin_alt),
            nb_max=nb_max, np_max=np_max,
            device=device,
        )
        # broadcast 到 batch=K（每条候选共享同一路网）
        rp_K = rp_t.expand(K, -1, -1, -1).contiguous()
        rm_K = rm_t.expand(K, -1, -1).contiguous()

        # 给 constraint optimizer 的 ctx：road 是真的，其他字段全 0 占位
        ctx = build_dummy_context(K, device=device)
        ctx.road_points = rp_K
        ctx.road_mask = rm_K

        # forward
        traj_K = torch.from_numpy(top_phys_i.astype(np.float32)).to(device)   # [K, T, 6]
        with torch.no_grad():
            refined = constraint(traj_K, ctx)                                 # [K, T, 6]
        refined_xyz = refined[..., :3].cpu().numpy()                          # [K, T, 3]

        # 路网在局部 ENU 下的坐标（用 mask 取出）
        rp_np = rp_t[0].cpu().numpy()      # [NB_max, NP_max, 3]
        rm_np = rm_t[0].cpu().numpy()      # [NB_max, NP_max]
        road_branches_enu: List[np.ndarray] = []
        for bi in range(rp_np.shape[0]):
            valid = rm_np[bi]
            if not valid.any():
                continue
            road_branches_enu.append(rp_np[bi][valid])

        _plot_one_sample_pair(
            ax_left=axes[k_i, 0],
            ax_right=axes[k_i, 1],
            sample_idx=int(sample_idx),
            top_phys=top_phys_i,
            top_idx=top_idx_i,
            top_probs=top_probs_i,
            refined_xyz=refined_xyz,
            road_branches_enu=road_branches_enu,
            position=pos_arr[k_i],
            label=int(label_arr[k_i]),
            task_type=int(task_arr[k_i]),
            type_id=int(type_arr[k_i]),
        )

        # 控制台打一行投影统计
        orig_xy = top_phys_i[..., :2]
        mean_shift = float(np.linalg.norm(refined_xyz[..., :2] - orig_xy, axis=-1).mean())
        print(
            f"  sample {sample_idx}: top_idx={top_idx_i.tolist()} "
            f"probs={[round(float(p), 3) for p in top_probs_i]} "
            f"mean(|Δxy|)={mean_shift:.3f} km"
        )

    fig.suptitle(
        f"ConstraintOptimizer.{mode}  |  split={split}  n={n_want}  K={K}\n"
        f"路网通过甲方 LLH 接口合成，origin=({origin_lon:.4f},{origin_lat:.4f},{origin_alt:.1f})",
        fontsize=12,
    )
    _add_global_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{mode}_{split}_{stamp}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[Test/Road] saved: {out_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="测试 GNN1 + ConstraintOptimizer.road_projection")
    parser.add_argument("--gnn1-config", type=str,
                        default=str(_REPO_ROOT / "gnn1" / "config.yaml"))
    parser.add_argument("--gnn1-ckpt", type=str, default="",
                        help="GNN1 ckpt 路径；不填则取 gnn1/checkpoints 下最新 .pt")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="可视化样本个数（每个一行）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default=str(_THIS_FILE.parent / "vis"),
                        help="可视化 png 输出目录")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--origin-lon", type=float, default=116.30)
    parser.add_argument("--origin-lat", type=float, default=39.90)
    parser.add_argument("--origin-alt", type=float, default=0.0)

    parser.add_argument("--nb-max", type=int, default=4,
                        help="ContextBatch 里 road_max_branches")
    parser.add_argument("--np-max", type=int, default=128,
                        help="ContextBatch 里 road_max_points")

    parser.add_argument("--mode", type=str, default="road_arc_projection",
                        choices=["road_projection", "road_arc_projection", "pass_through"],
                        help="constraint_optimizer 算法：默认 road_arc_projection（沿弧长贴路）")

    args = parser.parse_args()

    gnn1_cfg_path = Path(args.gnn1_config).resolve()
    gnn1_ckpt = Path(args.gnn1_ckpt).resolve() if args.gnn1_ckpt else None
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (_THIS_FILE.parent / out_dir).resolve()

    run(
        gnn1_config_path=gnn1_cfg_path,
        gnn1_ckpt=gnn1_ckpt,
        split=args.split,
        n=int(args.n),
        seed=int(args.seed),
        out_dir=out_dir,
        device_name=args.device,
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        origin_alt=args.origin_alt,
        nb_max=int(args.nb_max),
        np_max=int(args.np_max),
        mode=str(args.mode),
    )


if __name__ == "__main__":
    main()
