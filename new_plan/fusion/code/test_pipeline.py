"""
fusion/code/test_pipeline.py
----------------------------
fusion 端到端测试脚本：用 lstm1 / gnn1 真实数据 + 合成路网跑一遍 FullNetV2，
然后把"输入 / 输出 / 4 张可视化图"全落到一个时间戳运行目录里。

整体流程：
  1) 加载 fusion/config.yaml 构出 FullNetV2（按 enable 状态）。
     默认 lstm2 / gnn2 关闭，只跑 lstm1 / gnn1 / constraint_optimizer。
     `fusion/config.yaml` 中 lstm1.ckpt / gnn1.ckpt 默认指向 checkpoints 目录，
     build.py 会自动取目录下最新的 .pt 加载。
  2) 从 gnn1 的 npz 数据里抽 N 条样本：
       - hist_traj：从 gnn1/data/cache/{split}.npz 反解出物理空间历史
                   （hist 末帧落在原点 (0,0,0)，与 gnn1 frame 对齐）→ 喂给 LSTM1
       - task_type / type / position：直接来自 gnn1/data/raw/{split}.npz
       - eta：None（gnn2 关，fusion 内部补零）
  3) 第 1 趟 forward：临时关 constraint_optimizer，拿到 LSTM1+GNN1 未投影 top-K。
  4) 用未投影 top-K 的物理坐标，调 synth_road 造一张 K 叉真路网（LLH 接口格式）。
  5) 第 2 趟 forward：恢复 constraint_optimizer，喂入路网 → 拿到投影后 top-K。

输出（写到 fusion/eval_vis/run_<split>_<timestamp>/）::
  inputs.npz             FullNetV2 实际入参（hist_traj + 7 个 ctx 张量）
  outputs.npz            完整 [B, K, 68] 输出 + 各子字段拆解 + 未投影对照
  vis_1_history.png       仅画 history + GT future + position（"模型看到的世界"）
  vis_2_road.png          仅画路网折线（每条样本的合成路网）
  vis_3_predictions.png   投影后 top-K 预测 + GT future（"模型给出的预测"）
  vis_4_combined.png      history + GT + road + position + 投影后 top-K（全景）
  meta.txt               文字摘要（enable 状态、ckpt 路径、各字段哨兵值校验）

用法（在 new_plan/ 下激活 LSTM_traj_predict 环境后）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m fusion.code.test_pipeline --n 4   # 默认读取最新的ckpt

    # 显式指定 ckpt：
    python -m fusion.code.test_pipeline --n 4 \
        --lstm1-ckpt ../lstm1/checkpoints/<run_id>/best_lstm_epoch048_valloss0.0136.pt \
        --gnn1-ckpt  ../gnn1/checkpoints/<run_id>/best_gnn1_epoch001_valloss0.9508.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.context_schema import ContextBatch  # noqa: E402

from fusion.code.build import load_fusion_config, _resolve_rel  # noqa: E402
from fusion.code.full_net_v2 import (  # noqa: E402
    INTENT_SENTINEL,
    build_full_net_from_fusion_config,
)

from constraint_optimizer.test_road_net.road_schema import (  # noqa: E402
    RoadNetwork,
    road_network_to_tensors,
)
from constraint_optimizer.test_road_net.synth_road import (  # noqa: E402
    build_road_network_for_sample,
)


# ============================================================
# Scaler / 反解工具
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


def _decode_window_to_phys(
    hist_norm: np.ndarray,           # [Tin, 6]
    fut_norm: Optional[np.ndarray],  # [Tout, 6] 或 None
    scaler: _Scaler,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    cache 里的 (history, targets) 反解到物理 km，并把 hist 末帧平移到 (0,0,0)。

      1) inverse_transform → [delta_x, delta_y, delta_z, vx, vy, vz]
      2) hist + future 的位置 deltas 合起来 cumsum → 相对 hist[0] 的位置
      3) 减去 hist 末帧位置 → hist 末帧 = (0,0,0)（与 gnn1 position 同坐标系）
      4) 速度通道保持原值
    """
    Tin = int(hist_norm.shape[0])
    hist_orig = scaler.inverse_transform(hist_norm.astype(np.float64))
    hist_pos_d = hist_orig[..., :3]
    hist_vel = hist_orig[..., 3:6]

    if fut_norm is not None:
        fut_orig = scaler.inverse_transform(fut_norm.astype(np.float64))
        fut_pos_d = fut_orig[..., :3]
        fut_vel = fut_orig[..., 3:6]
        all_d = np.concatenate([hist_pos_d, fut_pos_d], axis=0)
    else:
        fut_pos_d = None
        fut_vel = None
        all_d = hist_pos_d

    pos_abs = np.cumsum(all_d, axis=0)
    hist_last_pos = pos_abs[Tin - 1].copy()
    pos_rel = pos_abs - hist_last_pos

    hist_pos = pos_rel[:Tin]
    hist_phys = np.concatenate([hist_pos, hist_vel], axis=-1).astype(np.float32)

    fut_phys = None
    if fut_norm is not None:
        fut_pos = pos_rel[Tin:]
        fut_phys = np.concatenate([fut_pos, fut_vel], axis=-1).astype(np.float32)

    return hist_phys, fut_phys


# ============================================================
# 模型构造：默认走 fusion/config.yaml；CLI 可覆盖 ckpt
# ============================================================

def _build_full_net_with_overrides(
    fusion_cfg_path: Path,
    lstm1_ckpt: Optional[Path],
    gnn1_ckpt: Optional[Path],
):
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in fusion_cfg.items()}
    if lstm1_ckpt is not None:
        cfg.setdefault("lstm1", {})["ckpt"] = str(lstm1_ckpt)
    if gnn1_ckpt is not None:
        cfg.setdefault("gnn1", {})["ckpt"] = str(gnn1_ckpt)

    if lstm1_ckpt is None and gnn1_ckpt is None:
        return build_full_net_from_fusion_config(fusion_cfg_path), cfg, fusion_cfg_dir

    tmp_yaml = fusion_cfg_dir / ".tmp_fusion_config_for_test.yaml"
    with tmp_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    try:
        return build_full_net_from_fusion_config(tmp_yaml), cfg, fusion_cfg_dir
    finally:
        try:
            tmp_yaml.unlink()
        except OSError:
            pass


# ============================================================
# 数据加载
# ============================================================

def load_samples(
    gnn1_root: Path,
    split: str,
    n: int,
    seed: int,
) -> Tuple[
    np.ndarray,             # hist_phys [N, 20, 6]
    Optional[np.ndarray],   # fut_phys  [N, 10, 6] 或 None
    np.ndarray,             # task_type [N]
    np.ndarray,             # type      [N]
    np.ndarray,             # position  [N, 3]
    np.ndarray,             # label     [N]
    List[int],              # 选中的 raw 样本索引
    int,                    # samples_per_window
]:
    raw_path = gnn1_root / "data" / "raw" / f"{split}.npz"
    cache_path = gnn1_root / "data" / "cache" / f"{split}.npz"
    scaler_path = gnn1_root / "data" / "cache" / "scaler_posvel.npz"
    cfg_path = gnn1_root / "config.yaml"

    for p in (raw_path, cache_path, scaler_path, cfg_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到必要文件: {p}")

    with cfg_path.open("r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    samples_per_window = int(gnn1_cfg.get("data", {}).get("samples_per_window", 5))

    raw = np.load(raw_path)
    cache = np.load(cache_path)
    scaler = _Scaler.load(scaler_path)

    if "history" not in cache.files:
        raise KeyError(f"{cache_path} 缺少 history 字段")
    history = cache["history"]
    targets_cache = cache["targets"] if "targets" in cache.files else None
    Nw = int(history.shape[0])
    Nraw = int(raw["task_type"].shape[0])

    rng = np.random.default_rng(seed)
    n_want = min(int(n), Nraw)
    chosen = sorted(rng.choice(Nraw, size=n_want, replace=False).tolist())

    hist_phys_list, fut_phys_list = [], []
    for i in chosen:
        cw = i // samples_per_window
        if cw >= Nw:
            cw = cw % Nw
        hp, fp = _decode_window_to_phys(
            history[cw],
            targets_cache[cw] if targets_cache is not None else None,
            scaler,
        )
        hist_phys_list.append(hp)
        if fp is not None:
            fut_phys_list.append(fp)

    hist_phys = np.stack(hist_phys_list, axis=0)
    fut_phys = (
        np.stack(fut_phys_list, axis=0)
        if (targets_cache is not None and fut_phys_list)
        else None
    )

    task_type = raw["task_type"][chosen].astype(np.int64)
    type_id = raw["type"][chosen].astype(np.int64)
    position = raw["position"][chosen].astype(np.float32)
    label = raw["label"][chosen].astype(np.int64)

    return hist_phys, fut_phys, task_type, type_id, position, label, chosen, samples_per_window


# ============================================================
# 路网构造
# ============================================================

def build_roads_for_batch(
    cand_xyz_phys: np.ndarray,       # [B, K, T, 3]
    origin_lon: float,
    origin_lat: float,
    origin_alt: float,
    nb_max: int,
    np_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[np.ndarray]]]:
    B = int(cand_xyz_phys.shape[0])
    rp_list, rm_list = [], []
    branches_enu_per_sample: List[List[np.ndarray]] = []
    for b in range(B):
        road_net: RoadNetwork = build_road_network_for_sample(
            cand_xyz_km=cand_xyz_phys[b].astype(np.float64),
            origin_llh=(origin_lon, origin_lat, origin_alt),
            n_branches=int(cand_xyz_phys.shape[1]),
        )
        rp_t, rm_t = road_network_to_tensors(
            road_net,
            origin_llh=(origin_lon, origin_lat, origin_alt),
            nb_max=nb_max, np_max=np_max,
            device=device,
        )
        rp_list.append(rp_t.squeeze(0))
        rm_list.append(rm_t.squeeze(0))

        rp_np = rp_t[0].cpu().numpy()
        rm_np = rm_t[0].cpu().numpy()
        per_branches: List[np.ndarray] = []
        for bi in range(rp_np.shape[0]):
            valid = rm_np[bi]
            if not valid.any():
                continue
            per_branches.append(rp_np[bi][valid])
        branches_enu_per_sample.append(per_branches)

    rp_full = torch.stack(rp_list, dim=0)
    rm_full = torch.stack(rm_list, dim=0)
    return rp_full, rm_full, branches_enu_per_sample


# ============================================================
# 通用绘图原子
# ============================================================

_TOPK_COLORS = ["#D62728", "#FF7F0E", "#C71585", "#8B4513", "#2F4F4F"]
_BRANCH_COLORS = ["#1F77B4", "#2CA02C", "#9467BD", "#17BECF"]


def _grid_layout(n: int) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, int(np.ceil(n / 2))
    if n <= 9:
        return 3, 3
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _viewport_from(*arrs, margin_frac: float = 0.18) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for a in arrs:
        if a is None or a.size == 0:
            continue
        a2 = a.reshape(-1, a.shape[-1])
        xs += a2[:, 0].tolist()
        ys += a2[:, 1].tolist()
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    return (xmin - dx * margin_frac, xmax + dx * margin_frac,
            ymin - dy * margin_frac, ymax + dy * margin_frac)


def _draw_history(ax: plt.Axes, hist_xy: np.ndarray, with_label: bool = True) -> None:
    ax.plot(hist_xy[:, 0], hist_xy[:, 1], color="black", linewidth=2.4,
            alpha=0.9, zorder=4, label=("history" if with_label else None))
    ax.scatter(hist_xy[:, 0], hist_xy[:, 1], s=14, color="black", zorder=5)
    ax.scatter(hist_xy[0, 0], hist_xy[0, 1], s=46, marker="o",
               color="white", edgecolors="black", linewidths=1.4,
               zorder=6, label=("hist start" if with_label else None))
    ax.scatter(0, 0, s=70, marker="*", color="black", zorder=7,
               label=("hist last（原点）" if with_label else None))


def _draw_gt_future(ax: plt.Axes, gt_xy: Optional[np.ndarray], with_label: bool = True) -> None:
    if gt_xy is None:
        return
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linestyle="--",
            linewidth=1.6, alpha=0.85, zorder=3,
            label=("GT future" if with_label else None))
    ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=12, color="black", zorder=4)


def _draw_top_k(
    ax: plt.Axes,
    traj_xy: np.ndarray,        # [K, T, 2]
    probs: np.ndarray,          # [K]
    end_marker: str = "s",      # 兼容老接口；实际不再画大端点
    with_label: bool = True,
) -> None:
    """画 top-K 预测轨迹：只画线，不画每步点 / 不画大端点 marker。"""
    del end_marker  # 不再使用
    K = int(traj_xy.shape[0])
    for r in range(K):
        col = _TOPK_COLORS[r % len(_TOPK_COLORS)]
        xy = traj_xy[r]
        ax.plot(xy[:, 0], xy[:, 1], color=col, linewidth=1.6, alpha=0.95,
                zorder=5,
                label=(f"top{r+1}  p={probs[r]*100:.1f}%" if with_label else None),
                solid_capstyle="round")


def _draw_road_network(ax: plt.Axes, branches_enu: List[np.ndarray],
                       with_label: bool = True) -> None:
    for bi, br in enumerate(branches_enu):
        if br.shape[0] < 2:
            continue
        c = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        name = f"分支 {bi}（{br.shape[0]} 点）"
        ax.plot(br[:, 0], br[:, 1], color=c, linewidth=2.8, alpha=0.9,
                zorder=2, solid_capstyle="round",
                label=(name if with_label else None))
        ax.scatter(br[:, 0], br[:, 1], s=20, color=c,
                   edgecolors="white", linewidths=0.6, zorder=2)


def _draw_position(ax: plt.Axes, pos: np.ndarray,
                   xlim: Tuple[float, float], ylim: Tuple[float, float],
                   label_text: str = "position（我方目标）") -> None:
    """直接把 position 画成红色五角星；调用前请保证 viewport 已包含它。"""
    del xlim, ylim  # 不再做边界判断
    px, py = float(pos[0]), float(pos[1])
    ax.scatter(px, py, s=220, color="red", marker="*",
               edgecolors="black", linewidths=1.2, zorder=9,
               label=f"{label_text}  ({px:.1f},{py:.1f})")


def _finalize_ax(ax: plt.Axes, title: str,
                 xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km, ENU)")
    ax.set_ylabel("y (km, ENU)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.85)


# ============================================================
# 4 张可视化（每张一个 N 子图的 figure）
# ============================================================

def _make_grid_fig(B: int, base_w: float = 5.5, base_h: float = 5.0):
    rows, cols = _grid_layout(B)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(base_w * cols, base_h * rows + 0.8))
    if B == 1:
        axes = np.array([[axes]])
    else:
        axes = np.atleast_2d(axes).reshape(rows, cols)
    return fig, axes, rows, cols


def vis_history_only(
    hists_xy: List[np.ndarray],
    gt_xys: List[Optional[np.ndarray]],
    positions: np.ndarray,
    chosen: List[int],
    task_type: np.ndarray, type_id: np.ndarray,
    out_path: Path, title_suffix: str,
) -> None:
    """图 1：仅 history + GT future + position（"模型看到的世界"）。"""
    B = len(hists_xy)
    fig, axes, rows, cols = _make_grid_fig(B)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i >= B:
            ax.axis("off"); continue
        hist_xy = hists_xy[i]; gt_xy = gt_xys[i]; pos = positions[i]
        arrs = [hist_xy, np.zeros((1, 2)), pos[None, :2]]
        if gt_xy is not None:
            arrs.append(gt_xy)
        xmin, xmax, ymin, ymax = _viewport_from(*arrs, margin_frac=0.22)
        _draw_history(ax, hist_xy)
        _draw_gt_future(ax, gt_xy)
        _draw_position(ax, pos, (xmin, xmax), (ymin, ymax))
        _finalize_ax(
            ax,
            f"sample {chosen[i]}  task={int(task_type[i])}  type={int(type_id[i])}",
            (xmin, xmax), (ymin, ymax),
        )
    fig.suptitle(f"图 1 / 原始输入：history + GT future + position  |  {title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=140); plt.close(fig)


def vis_road_only(
    branches_per_sample: List[List[np.ndarray]],
    positions: np.ndarray,
    chosen: List[int],
    out_path: Path, title_suffix: str,
) -> None:
    """图 2：路网折线 + position（每条样本独立路网）。"""
    B = len(branches_per_sample)
    fig, axes, rows, cols = _make_grid_fig(B)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i >= B:
            ax.axis("off"); continue
        branches = branches_per_sample[i]
        all_pts = np.concatenate(branches, axis=0) if branches else np.zeros((1, 3))
        # 视口包含路网 + origin + position
        xmin, xmax, ymin, ymax = _viewport_from(
            all_pts[..., :2], np.zeros((1, 2)), positions[i, None, :2],
            margin_frac=0.15,
        )
        _draw_road_network(ax, branches)
        ax.scatter(0, 0, s=70, marker="*", color="black", zorder=7,
                   label="hist last（原点）")
        _draw_position(ax, positions[i], (xmin, xmax), (ymin, ymax))
        n_branches = sum(1 for br in branches if br.shape[0] >= 2)
        n_pts = int(sum(br.shape[0] for br in branches))
        _finalize_ax(
            ax,
            f"sample {chosen[i]}  branches={n_branches}  points={n_pts}",
            (xmin, xmax), (ymin, ymax),
        )
    fig.suptitle(f"图 2 / 路网（合成 K 叉，LLH→ENU 后回填）  |  {title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=140); plt.close(fig)


def vis_predictions_only(
    hists_xy: List[np.ndarray],
    gt_xys: List[Optional[np.ndarray]],
    unrefined_topk_xy: np.ndarray,     # [B, K, T, 2]   LSTM1+GNN1 原始（未投影）
    mode_prob: np.ndarray,              # [B, K]
    positions: np.ndarray,
    chosen: List[int],
    out_path: Path, title_suffix: str,
) -> None:
    """
    图 3：top-K 预测（GNN1 输出，未经路网约束）+ GT future + position。
    与 gnn1/eval_vis 语义一致。
    """
    B = len(hists_xy)
    K = int(unrefined_topk_xy.shape[1])
    fig, axes, rows, cols = _make_grid_fig(B)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i >= B:
            ax.axis("off"); continue
        hist_xy = hists_xy[i]; gt_xy = gt_xys[i]
        topk_xy = unrefined_topk_xy[i]
        arrs = [hist_xy, topk_xy.reshape(-1, 2),
                np.zeros((1, 2)), positions[i, None, :2]]
        if gt_xy is not None:
            arrs.append(gt_xy)
        xmin, xmax, ymin, ymax = _viewport_from(*arrs, margin_frac=0.22)

        _draw_history(ax, hist_xy)
        _draw_gt_future(ax, gt_xy)
        _draw_top_k(ax, topk_xy, mode_prob[i])
        _draw_position(ax, positions[i], (xmin, xmax), (ymin, ymax))
        _finalize_ax(
            ax,
            f"sample {chosen[i]}   top-{K}（GNN1 输出，未经路网约束）",
            (xmin, xmax), (ymin, ymax),
        )
    fig.suptitle(f"图 3 / 输出：top-K 预测（GNN1 输出，未经路网约束）  |  {title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=140); plt.close(fig)


def vis_combined(
    hists_xy: List[np.ndarray],
    gt_xys: List[Optional[np.ndarray]],
    refined_topk_xy: np.ndarray,
    mode_prob: np.ndarray,
    branches_per_sample: List[List[np.ndarray]],
    positions: np.ndarray,
    chosen: List[int],
    task_type: np.ndarray, type_id: np.ndarray, label_arr: np.ndarray,
    avg_shifts: List[float],
    out_path: Path, title_suffix: str,
) -> None:
    """图 4：路网 + history + GT + position + 投影后 top-K（全景）。"""
    B = len(hists_xy)
    fig, axes, rows, cols = _make_grid_fig(B)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i >= B:
            ax.axis("off"); continue
        hist_xy = hists_xy[i]; gt_xy = gt_xys[i]
        topk_xy = refined_topk_xy[i]
        branches = branches_per_sample[i]

        arrs = [hist_xy, topk_xy.reshape(-1, 2), np.zeros((1, 2)),
                positions[i, None, :2]]
        if gt_xy is not None:
            arrs.append(gt_xy)
        if branches:
            arrs.append(np.concatenate(branches, axis=0)[..., :2])
        xmin, xmax, ymin, ymax = _viewport_from(*arrs, margin_frac=0.20)

        _draw_road_network(ax, branches)
        _draw_history(ax, hist_xy)
        _draw_gt_future(ax, gt_xy)
        _draw_top_k(ax, topk_xy, mode_prob[i], end_marker="s")
        _draw_position(ax, positions[i], (xmin, xmax), (ymin, ymax))
        _finalize_ax(
            ax,
            f"sample {chosen[i]}  task={int(task_type[i])}  type={int(type_id[i])}  "
            f"label={int(label_arr[i])}\n"
            f"|Δxy| = {avg_shifts[i]:.3f} km（路网投影位移）",
            (xmin, xmax), (ymin, ymax),
        )
    fig.suptitle(f"图 4 / 全景：路网 + history + GT + position + top-K  |  {title_suffix}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=140); plt.close(fig)


def vis_compare_with_gnn1eval(
    hists_xy: List[np.ndarray],
    gt_xys: List[Optional[np.ndarray]],
    cand_full_xy: np.ndarray,        # [B, M=5, T, 2]
    top_idx: np.ndarray,             # [B, K]
    top_probs: np.ndarray,           # [B, K]
    positions: np.ndarray,
    label_arr: np.ndarray,
    chosen: List[int],
    task_type: np.ndarray, type_id: np.ndarray,
    out_path: Path, title_suffix: str,
) -> None:
    """
    图 5：和 gnn1/eval_vis 同风格 ——
      - 全部 5 条 LSTM1 候选（淡灰色细线 + cN 文字）
      - label 那条（绿色加粗）
      - GNN1 选出的 top-K（红/橙/品红 粗实线 + 概率）
      - history（黑实线）/ GT future（黑虚线）/ position（红五角星）
    便于直接和 gnn1/eval_vis 出的图做 1:1 对照。
    """
    B, M, T, _ = cand_full_xy.shape
    K = int(top_idx.shape[-1])
    fig, axes, rows, cols = _make_grid_fig(B)
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i >= B:
            ax.axis("off"); continue

        hist_xy = hists_xy[i]; gt_xy = gt_xys[i]
        cand_xy = cand_full_xy[i]
        ti = top_idx[i]; tp = top_probs[i]
        lab = int(label_arr[i])

        arrs = [hist_xy, cand_xy.reshape(-1, 2), np.zeros((1, 2)),
                positions[i, None, :2]]
        if gt_xy is not None:
            arrs.append(gt_xy)
        xmin, xmax, ymin, ymax = _viewport_from(*arrs, margin_frac=0.18)

        for m in range(M):
            ax.plot(cand_xy[m, :, 0], cand_xy[m, :, 1],
                    color="#BBBBBB", linewidth=0.9, alpha=0.85, zorder=2)
            ax.text(cand_xy[m, -1, 0], cand_xy[m, -1, 1], f"  c{m}",
                    fontsize=7, color="#888888", zorder=3)

        if 0 <= lab < M:
            ax.plot(cand_xy[lab, :, 0], cand_xy[lab, :, 1],
                    color="#2CA02C", linewidth=2.0, alpha=0.85, zorder=3,
                    label=f"label = c{lab}")

        for r_idx in range(K):
            m = int(ti[r_idx])
            col = _TOPK_COLORS[r_idx % len(_TOPK_COLORS)]
            ax.plot(cand_xy[m, :, 0], cand_xy[m, :, 1],
                    color=col, linewidth=2.2, alpha=1.0, zorder=5,
                    label=f"top{r_idx+1}: c{m}  p={float(tp[r_idx])*100:.1f}%")

        _draw_history(ax, hist_xy)
        _draw_gt_future(ax, gt_xy)
        _draw_position(ax, positions[i], (xmin, xmax), (ymin, ymax))
        ti_str = ", ".join(f"c{int(t)}:{float(p)*100:.1f}%"
                           for t, p in zip(ti, tp))
        _finalize_ax(
            ax,
            f"sample {chosen[i]}  task={int(task_type[i])}  type={int(type_id[i])}\n"
            f"label=c{lab}   top-K (renorm): [{ti_str}]",
            (xmin, xmax), (ymin, ymax),
        )
    fig.suptitle(
        f"图 5 / 与 gnn1/eval_vis 对照：5 候选 + GNN1 top-K 高亮  |  {title_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(out_path, dpi=140); plt.close(fig)


# ============================================================
# 输入 / 输出 落到 txt（人类可读）
# ============================================================

def _array_to_txt_block(arr: np.ndarray, name: str, precision: int = 4) -> str:
    lines = [f"{name}  shape={tuple(arr.shape)}  dtype={arr.dtype}"]
    if arr.ndim == 0:
        lines.append(f"  {arr.item()}")
    elif arr.ndim == 1:
        lines.append("  " + ", ".join(
            (f"{v}" if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool
             else f"{v:.{precision}f}")
            for v in arr.tolist()))
    else:
        inner = arr.reshape(-1, arr.shape[-1])
        idx_shape = arr.shape[:-1]
        for k, idx in enumerate(np.ndindex(*idx_shape)):
            vec = inner[k]
            sidx = ",".join(str(j) for j in idx)
            row = ", ".join(
                (f"{v}" if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool
                 else f"{v:>9.{precision}f}")
                for v in vec.tolist())
            lines.append(f"  [{sidx}]  {row}")
    return "\n".join(lines)


def save_inputs_txt(
    out_path: Path,
    hist_phys: np.ndarray,
    fut_phys: Optional[np.ndarray],
    task_type: np.ndarray,
    type_id: np.ndarray,
    position: np.ndarray,
    rp: np.ndarray,
    rm: np.ndarray,
    eta_np: np.ndarray,
    raw_idx: List[int],
) -> None:
    blocks = [
        "=" * 64,
        "FullNetV2 真实输入（pass-2，含路网；ctx 字段在 fusion 内被各子模块消费）",
        "=" * 64,
        f"raw_idx = {raw_idx}",
        "",
        _array_to_txt_block(hist_phys, "hist_traj  [B, 20, 6]  km / km·s⁻¹"),
        "",
        _array_to_txt_block(task_type, "task_type  [B]  long  (0=打击)"),
        _array_to_txt_block(type_id,   "type       [B]  long  (我方目标类型 0..2)"),
        _array_to_txt_block(position,  "position   [B, 3]  km xyz（hist 末帧 = 原点）"),
        "",
        _array_to_txt_block(rp.astype(np.float32),
                            "road_points [B, NB, NP, 3]  km xyz"),
        _array_to_txt_block(rm.astype(np.int8),
                            "road_mask   [B, NB, NP]  (1=有效)"),
        "",
        _array_to_txt_block(eta_np, "eta  [B]  long  (秒；fusion 内 None→补 0)"),
        "",
    ]
    if fut_phys is not None:
        blocks += [
            "(参考：cache 里同窗口的 GT future，仅画图用，不在 fusion 入口里)",
            _array_to_txt_block(fut_phys, "gt_future  [B, 10, 6]  km / km·s⁻¹"),
        ]
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def save_outputs_txt(
    out_path: Path,
    out_full: np.ndarray,
    refined_traj: np.ndarray,
    unrefined_traj: np.ndarray,
    intent: np.ndarray,
    threat: np.ndarray,
    strike_pos: np.ndarray,
    strike_radius: np.ndarray,
    strike_conf: np.ndarray,
    mode_prob: np.ndarray,
    raw_idx: List[int],
) -> None:
    blocks = [
        "=" * 64,
        "FullNetV2 完整输出 [B, K, 68] 布局：",
        "    0..59  refined traj（10 步 × 6 维：x,y,z,vx,vy,vz）",
        "    60     intent_class    (lstm2 关 → -1)",
        "    61     threat_prob     (lstm2 关 → NaN)",
        "    62..64 strike_pos      (gnn2 关 → NaN)",
        "    65     strike_radius   (gnn2 关 → NaN)",
        "    66     strike_conf     (gnn2 关 → NaN)",
        "    67     mode_prob       (gnn1 重归一化，K 条概率和 = 1)",
        "=" * 64,
        f"raw_idx = {raw_idx}",
        "",
        _array_to_txt_block(out_full, "output  [B, K, 68]"),
        "",
        "—— 各字段拆解 ——",
        _array_to_txt_block(refined_traj,
                            "refined_traj   [B, K, 10, 6]  路网投影后（业务输出）"),
        _array_to_txt_block(unrefined_traj,
                            "unrefined_traj [B, K, 10, 6]  路网投影前（LSTM1+GNN1）"),
        _array_to_txt_block(intent,        "intent_class   [B, K]"),
        _array_to_txt_block(threat,        "threat_prob    [B, K]"),
        _array_to_txt_block(strike_pos,    "strike_pos     [B, K, 3]"),
        _array_to_txt_block(strike_radius, "strike_radius  [B, K]"),
        _array_to_txt_block(strike_conf,   "strike_conf    [B, K]"),
        _array_to_txt_block(mode_prob,     "mode_prob      [B, K]  每行 K 条概率和=1"),
        "",
    ]
    out_path.write_text("\n".join(blocks), encoding="utf-8")


# ============================================================
# 主流程
# ============================================================

def run(
    fusion_cfg_path: Path,
    lstm1_ckpt: Optional[Path],
    gnn1_ckpt: Optional[Path],
    split: str,
    n: int,
    seed: int,
    out_root: Path,
    device_name: str,
    origin_lon: float,
    origin_lat: float,
    origin_alt: float,
    nb_max: int,
    np_max: int,
) -> None:
    # ---- device ----
    if device_name.lower() == "cpu":
        device = torch.device("cpu")
    elif device_name.lower() in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test/Pipeline] device = {device}")

    # ---- 模型 ----
    model, patched_cfg, fusion_cfg_dir = _build_full_net_with_overrides(
        fusion_cfg_path, lstm1_ckpt, gnn1_ckpt,
    )
    model = model.to(device).eval()
    print(f"[Test/Pipeline] enable_flags = {model.enable_flags}")
    print(f"[Test/Pipeline] params       = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Test/Pipeline] lstm1_modes  = {model.lstm1_modes}, top_k = {model.top_k}")

    # ---- 数据 ----
    gnn1_root = _resolve_rel(patched_cfg.get("gnn1", {}).get("config", ""), fusion_cfg_dir).parent
    print(f"[Test/Pipeline] gnn1 数据根 = {gnn1_root}")

    (hist_phys, fut_phys, task_type, type_id, position,
     label, chosen, samples_per_window) = load_samples(gnn1_root, split, n, seed)
    B = int(hist_phys.shape[0])
    K = int(model.top_k)
    print(f"[Test/Pipeline] split={split}  B={B}  selected raw idx = {chosen}  "
          f"samples_per_window={samples_per_window}")

    hist_t = torch.from_numpy(hist_phys).to(device).float()

    # ---- Pass 1：未投影 ----
    ctx_no_road = ContextBatch(
        task_type=torch.from_numpy(task_type).to(device),
        type=torch.from_numpy(type_id).to(device),
        position=torch.from_numpy(position).to(device),
        road_points=None, road_mask=None, eta=None,
    )

    # 同时把 fusion 内部 LSTM1 的 5 条候选（物理 km）和 GNN1 的 top_idx/top_probs
    # 拿出来，留给图 5 做"和 gnn1/eval_vis 同风格"对照
    ctx_n_pre = model._normalize_ctx(ctx_no_road, batch_size=B, device=device)
    with torch.no_grad():
        x_clean_pre, _ = model.outlier_filter(hist_t)
        x_A_pre = model._build_A_input(x_clean_pre)
        fut_A_full = model.lstm1(x_A_pre)                              # [B, M=5, T, 6]
        gnn1_out = model.gnn1({
            "cand_trajs": fut_A_full,
            "task_type":  ctx_n_pre.task_type,
            "type":       ctx_n_pre.type,
            "position":   ctx_n_pre.position,
        })
        full_top_idx = gnn1_out["top_idx"].cpu().numpy()                # [B, K]
        full_top_probs = gnn1_out["top_probs"].cpu().numpy()            # [B, K]
        # M 条候选反归一化 + cumsum 到物理（hist 末帧为原点）
        cand_full_phys = model._decode_future_to_abs(
            x_clean_pre, fut_A_full
        ).cpu().numpy()                                                 # [B, M, T, 6]

    saved_constraint = model.constraint
    model.constraint = None
    model.enable_flags["constraint_optimizer"] = False
    try:
        with torch.no_grad():
            out_no_road = model(hist_t, ctx_no_road)
    finally:
        model.constraint = saved_constraint
        model.enable_flags["constraint_optimizer"] = True

    traj_no_road = out_no_road[..., 0:60].reshape(B, K, model.fut_len, 6).cpu().numpy()
    mode_prob = out_no_road[..., 67].cpu().numpy()
    print(f"[Test/Pipeline] mode_prob (top-K, K 条和=1):")
    for i in range(B):
        ps = ", ".join(f"{p*100:.1f}%" for p in mode_prob[i])
        print(f"   sample {chosen[i]}:  {ps}")

    # ---- 路网 ----
    rp, rm, branches_enu_per_sample = build_roads_for_batch(
        cand_xyz_phys=traj_no_road[..., 0:3],
        origin_lon=origin_lon, origin_lat=origin_lat, origin_alt=origin_alt,
        nb_max=nb_max, np_max=np_max, device=device,
    )
    n_valid_pts = int(rm.sum().item())
    print(f"[Test/Pipeline] road_points = {tuple(rp.shape)}, mask True = {n_valid_pts}")

    # ---- Pass 2：投影后 ----
    ctx_with_road = ContextBatch(
        task_type=torch.from_numpy(task_type).to(device),
        type=torch.from_numpy(type_id).to(device),
        position=torch.from_numpy(position).to(device),
        road_points=rp, road_mask=rm, eta=None,
    )
    with torch.no_grad():
        out_with_road = model(hist_t, ctx_with_road)
    out_full = out_with_road.cpu().numpy()                           # [B, K, 68]
    traj_refined = out_full[..., 0:60].reshape(B, K, model.fut_len, 6)

    # ---- 哨兵值校验 ----
    intent = out_full[..., 60]
    threat = out_full[..., 61]
    strike_pos = out_full[..., 62:65]
    strike_radius = out_full[..., 65]
    strike_conf = out_full[..., 66]
    mode_prob_v2 = out_full[..., 67]
    sentinel_lines = []
    if not model.enable_flags.get("lstm2", False):
        ok_intent = bool(np.all(intent == INTENT_SENTINEL))
        ok_threat = bool(np.all(np.isnan(threat)))
        sentinel_lines.append(
            f"lstm2 OFF  intent==-1: {ok_intent}   threat==NaN: {ok_threat}"
        )
    if not model.enable_flags.get("gnn2", False):
        ok_pos = bool(np.all(np.isnan(strike_pos)))
        ok_rad = bool(np.all(np.isnan(strike_radius)))
        ok_cnf = bool(np.all(np.isnan(strike_conf)))
        sentinel_lines.append(
            f"gnn2  OFF  strike_pos NaN: {ok_pos}  "
            f"strike_radius NaN: {ok_rad}  strike_conf NaN: {ok_cnf}"
        )
    diff_prob = float(np.max(np.abs(mode_prob - mode_prob_v2)))
    sentinel_lines.append(
        f"mode_prob 在两次 forward 间最大差 = {diff_prob:.2e}（应≈0）"
    )
    for line in sentinel_lines:
        print(f"[Test/Pipeline] {line}")

    # ============================================================
    # 落盘：run 目录
    # ============================================================
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{split}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Test/Pipeline] run_dir = {run_dir}")

    # ---- inputs.npz：FullNetV2 真正吃下去的张量（pass-2 的） ----
    eta_np = np.zeros(B, dtype=np.int64)   # ctx.eta 实际传入的是 None → 内部补零
    np.savez(
        run_dir / "inputs.npz",
        hist_traj=hist_phys.astype(np.float32),                      # [B, 20, 6]
        task_type=task_type.astype(np.int64),                         # [B]
        type=type_id.astype(np.int64),                                # [B]
        position=position.astype(np.float32),                         # [B, 3]
        road_points=rp.cpu().numpy().astype(np.float32),              # [B, NB, NP, 3]
        road_mask=rm.cpu().numpy().astype(bool),                      # [B, NB, NP]
        eta=eta_np,                                                   # [B]
        raw_idx=np.asarray(chosen, dtype=np.int64),                   # [B]
        gt_future=fut_phys.astype(np.float32) if fut_phys is not None
                   else np.zeros((B, model.fut_len, 6), dtype=np.float32),
        has_gt=np.array([fut_phys is not None], dtype=bool),
    )
    print(f"[Test/Pipeline] saved inputs.npz")

    # ---- outputs.npz：完整 [B, K, 68] + 各字段拆解 + 未投影对照 ----
    np.savez(
        run_dir / "outputs.npz",
        output=out_full.astype(np.float32),                           # [B, K, 68]
        refined_traj=traj_refined.astype(np.float32),                 # [B, K, T, 6]
        intent_class=intent.astype(np.float32),                       # [B, K]  禁用 → -1
        threat_prob=threat.astype(np.float32),                        # [B, K]  禁用 → NaN
        strike_pos=strike_pos.astype(np.float32),                     # [B, K, 3]
        strike_radius=strike_radius.astype(np.float32),               # [B, K]
        strike_conf=strike_conf.astype(np.float32),                   # [B, K]
        mode_prob=mode_prob_v2.astype(np.float32),                    # [B, K]
        unrefined_traj=traj_no_road.astype(np.float32),               # [B, K, T, 6]
    )
    print(f"[Test/Pipeline] saved outputs.npz")

    # ---- meta.txt：人类可读摘要 ----
    meta_lines = [
        f"fusion test pipeline run @ {stamp}",
        f"split           = {split}",
        f"B (n samples)   = {B}",
        f"K (top_k)       = {K}",
        f"selected raw idx= {chosen}",
        f"samples/window  = {samples_per_window}",
        "",
        f"enable_flags    = {model.enable_flags}",
        f"params          = {sum(p.numel() for p in model.parameters()):,}",
        f"lstm1_modes     = {model.lstm1_modes}",
        "",
        "ckpt（实际加载，run 时通过 build.py 解析）：",
        f"  lstm1.ckpt   = {patched_cfg.get('lstm1', {}).get('ckpt', '')}",
        f"  gnn1.ckpt    = {patched_cfg.get('gnn1', {}).get('ckpt', '')}",
        "",
        "哨兵值校验：",
    ] + ["  " + s for s in sentinel_lines] + [
        "",
        "mode_prob (pass-2，每行：top1, top2, top3)：",
    ]
    for i in range(B):
        meta_lines.append(
            f"  sample {chosen[i]}: " +
            ", ".join(f"{p*100:6.2f}%" for p in mode_prob_v2[i])
        )
    (run_dir / "meta.txt").write_text("\n".join(meta_lines), encoding="utf-8")
    print(f"[Test/Pipeline] saved meta.txt")

    # ---- inputs.txt / outputs.txt：人类可读完整 dump ----
    save_inputs_txt(
        out_path=run_dir / "inputs.txt",
        hist_phys=hist_phys, fut_phys=fut_phys,
        task_type=task_type, type_id=type_id, position=position,
        rp=rp.cpu().numpy(), rm=rm.cpu().numpy(),
        eta_np=eta_np, raw_idx=chosen,
    )
    print(f"[Test/Pipeline] saved inputs.txt")
    save_outputs_txt(
        out_path=run_dir / "outputs.txt",
        out_full=out_full, refined_traj=traj_refined,
        unrefined_traj=traj_no_road,
        intent=intent, threat=threat,
        strike_pos=strike_pos, strike_radius=strike_radius,
        strike_conf=strike_conf, mode_prob=mode_prob_v2,
        raw_idx=chosen,
    )
    print(f"[Test/Pipeline] saved outputs.txt")

    # ============================================================
    # 5 张可视化
    # ============================================================
    title_suffix = f"split={split}  B={B}  K={K}"
    hists_xy = [hist_phys[i, :, 0:2] for i in range(B)]
    gt_xys: List[Optional[np.ndarray]] = (
        [fut_phys[i, :, 0:2] for i in range(B)] if fut_phys is not None
        else [None] * B
    )
    refined_topk_xy = traj_refined[..., 0:2]                          # [B, K, T, 2]
    unrefined_topk_xy = traj_no_road[..., 0:2]                        # [B, K, T, 2]

    avg_shifts = [
        float(np.linalg.norm(
            traj_refined[i, ..., :2] - traj_no_road[i, ..., :2], axis=-1
        ).mean())
        for i in range(B)
    ]

    vis_history_only(
        hists_xy=hists_xy, gt_xys=gt_xys, positions=position,
        chosen=chosen, task_type=task_type, type_id=type_id,
        out_path=run_dir / "vis_1_history.png", title_suffix=title_suffix,
    )
    print(f"[Test/Pipeline] saved vis_1_history.png")

    vis_road_only(
        branches_per_sample=branches_enu_per_sample,
        positions=position, chosen=chosen,
        out_path=run_dir / "vis_2_road.png", title_suffix=title_suffix,
    )
    print(f"[Test/Pipeline] saved vis_2_road.png")

    vis_predictions_only(
        hists_xy=hists_xy, gt_xys=gt_xys, positions=position,
        unrefined_topk_xy=unrefined_topk_xy,
        mode_prob=mode_prob_v2, chosen=chosen,
        out_path=run_dir / "vis_3_predictions.png", title_suffix=title_suffix,
    )
    print(f"[Test/Pipeline] saved vis_3_predictions.png")

    vis_combined(
        hists_xy=hists_xy, gt_xys=gt_xys,
        refined_topk_xy=refined_topk_xy, mode_prob=mode_prob_v2,
        branches_per_sample=branches_enu_per_sample,
        positions=position, chosen=chosen,
        task_type=task_type, type_id=type_id, label_arr=label,
        avg_shifts=avg_shifts,
        out_path=run_dir / "vis_4_combined.png", title_suffix=title_suffix,
    )
    print(f"[Test/Pipeline] saved vis_4_combined.png")

    # 图 5：和 gnn1/eval_vis 同风格的 5 候选 + top-K 高亮（直接 1:1 对照）
    vis_compare_with_gnn1eval(
        hists_xy=hists_xy, gt_xys=gt_xys,
        cand_full_xy=cand_full_phys[..., :2],
        top_idx=full_top_idx, top_probs=full_top_probs,
        positions=position, label_arr=label, chosen=chosen,
        task_type=task_type, type_id=type_id,
        out_path=run_dir / "vis_5_compare_with_gnn1_eval.png",
        title_suffix=title_suffix,
    )
    print(f"[Test/Pipeline] saved vis_5_compare_with_gnn1_eval.png")

    print(f"\n[Test/Pipeline] DONE → {run_dir}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="fusion 端到端测试 + 可视化（用 lstm1/gnn1 真实数据 + 合成路网）"
    )
    parser.add_argument("--fusion-config", type=str,
                        default=str(REPO_ROOT / "fusion" / "config.yaml"))
    parser.add_argument("--lstm1-ckpt", type=str, default="",
                        help="LSTM1 ckpt；不填则用 fusion/config.yaml 默认（指向"
                             " lstm1/checkpoints 目录，build.py 自动取最新 .pt）")
    parser.add_argument("--gnn1-ckpt", type=str, default="",
                        help="GNN1 ckpt；同上")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="可视化样本个数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "fusion" / "eval_vis"),
                        help="run 目录的父目录；脚本会在其下建 run_<split>_<stamp>")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--origin-lon", type=float, default=116.30)
    parser.add_argument("--origin-lat", type=float, default=39.90)
    parser.add_argument("--origin-alt", type=float, default=0.0)

    parser.add_argument("--nb-max", type=int, default=4)
    parser.add_argument("--np-max", type=int, default=128)

    args = parser.parse_args()

    fusion_cfg_path = Path(args.fusion_config).resolve()
    lstm1_ckpt = Path(args.lstm1_ckpt).resolve() if args.lstm1_ckpt else None
    gnn1_ckpt = Path(args.gnn1_ckpt).resolve() if args.gnn1_ckpt else None
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()

    run(
        fusion_cfg_path=fusion_cfg_path,
        lstm1_ckpt=lstm1_ckpt,
        gnn1_ckpt=gnn1_ckpt,
        split=args.split,
        n=int(args.n),
        seed=int(args.seed),
        out_root=out_root,
        device_name=args.device,
        origin_lon=args.origin_lon,
        origin_lat=args.origin_lat,
        origin_alt=args.origin_alt,
        nb_max=int(args.nb_max),
        np_max=int(args.np_max),
    )


if __name__ == "__main__":
    main()
