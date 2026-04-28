#!/usr/bin/env python3
"""
gnn2/code/train/eval.py
-----------------------
GNN2 评估脚本（打击区域 / 半径 / 置信度）。

两种互斥模式（参考 lstm2/code/train/eval.py 的风格）：

  ---------- 模式 A：批量指标评估（默认）----------
  跑全量 split，不画图，输出：
    * 全局：pos_err / radius_err / conf_err 的 mean / p50 / p90
    * 全局：radius / conf 的 Pearson 相关系数（pred vs gt）
    * 按 eta 分桶（[0,60), [60,120), [120,240), ...）的误差
    * 按 cand_k 分组（top-1 / top-2 / top-3）的误差
    * 按 src_split 分组（gnn1 train/val/test 来源）的误差
    * forward 耗时（per batch / per sample）

  ---------- 模式 B：可视化评估（--vis）----------
  从指定 split 抽几个样本，每个样本画一张子图，对比 GT vs Pred：
    * 蓝色折线 + 时间渐变点：pred_traj（模型输入轨迹）
    * 红色五角星 + 红色虚线圆：GT strike_pos / gt_radius
    * 橙色三角 + 橙色实线圆：Pred strike_pos / pred_radius
    * 标题里附 |Δpos|、|Δradius|、|Δconf| 三项误差
  挑选模式：random / by-eta（按 eta 桶各 1 张）/ by-cand-k（同 scene 三条候选）/ indices

用法（在 new_plan/gnn2/ 下）::

    $env:PYTHONPATH = "$PWD/code"

    # ---------- 模式 A：批量指标（默认）----------
    python -m train.eval --config config.yaml --split test
    # 显式指定 ckpt：
    python -m train.eval --config config.yaml --split test \
        --ckpt checkpoints/<run>/best_gnn2_epoch*.pt

    # ---------- 模式 B：可视化 ----------
    python -m train.eval --config config.yaml --split val --vis --vis-num 6
    python -m train.eval --config config.yaml --split val --vis --vis-mode by-eta --vis-num 6
    python -m train.eval --config config.yaml --split val --vis --vis-mode by-cand-k --vis-seed 7
    python -m train.eval --config config.yaml --split val --vis --vis-indices 0,17,42
"""

from __future__ import annotations

import os
# Windows 上 PyTorch (libomp) 和 NumPy/MKL (libiomp5md) 共存时会触发 OMP Error #15。
# 必须在 import numpy / torch 之前 set，所以放在文件最顶端。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                     # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from data.dataset import build_datasets_from_config, Gnn2Dataset    # noqa: E402
from train.model import build_model_from_config                     # noqa: E402


SRC_SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


# ============================================================
# 通用工具
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    dev = str(train_cfg.get("device", "auto")).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_ckpt(p: Path) -> Optional[Path]:
    if p is None:
        return None
    if p.is_file():
        return p
    if p.is_dir():
        cands = list(p.rglob("*.pt"))
        if not cands:
            return None
        cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]
    return None


def pick_split(split: str, train_ds, val_ds, test_ds):
    s = split.lower()
    if s == "train":
        return train_ds
    if s in ("val", "valid", "validation"):
        return val_ds
    if s == "test":
        return test_ds
    raise ValueError(f"unknown split: {split}")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    手写 Pearson 相关系数，只用基础元素运算（mean/sub/mul/sqrt/sum），
    避免 np.corrcoef 内部走 MKL BLAS 触发 Windows 上的 OpenMP libomp / libiomp5md 冲突。
    """
    if x.size < 2:
        return float("nan")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    mx = float(x.mean())
    my = float(y.mean())
    xm = x - mx
    ym = y - my
    sx2 = float((xm * xm).sum())
    sy2 = float((ym * ym).sum())
    if sx2 < 1e-24 or sy2 < 1e-24:
        return float("nan")
    cov = float((xm * ym).sum())
    return cov / (sx2 ** 0.5) / (sy2 ** 0.5)


def _resolve_ckpt(ckpt_arg: str, gnn2_root: Path, train_cfg: Dict[str, Any]) -> Path:
    if ckpt_arg:
        p = Path(ckpt_arg)
        if not p.is_absolute():
            p = (gnn2_root / p).resolve()
        resolved = find_latest_ckpt(p)
    else:
        ckpt_root = (gnn2_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
        resolved = find_latest_ckpt(ckpt_root)
    if resolved is None:
        raise FileNotFoundError(
            f"找不到 ckpt（输入='{ckpt_arg}'）。请训练完后再 eval，或 --ckpt 显式指定。"
        )
    return resolved


# ============================================================
# 模式 A：批量指标评估
# ============================================================

def run_batch_eval(
    model: torch.nn.Module,
    ds: Gnn2Dataset,
    device: torch.device,
    cfg: Dict[str, Any],
    ckpt_path: Path,
    split: str,
    batch_size: int,
) -> None:
    print("\n" + "=" * 64)
    print(f"[Eval/GNN2 batch]  split={split}  ckpt={ckpt_path.name}")
    print("=" * 64)
    print(f"samples = {len(ds)}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    pred_pos_all: List[np.ndarray] = []
    gt_pos_all: List[np.ndarray] = []
    pred_radius_all: List[np.ndarray] = []
    gt_radius_all: List[np.ndarray] = []
    pred_conf_all: List[np.ndarray] = []
    gt_conf_all: List[np.ndarray] = []
    eta_all: List[np.ndarray] = []
    cand_k_all: List[np.ndarray] = []
    src_split_all: List[np.ndarray] = []

    n_batches = 0
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = _to_device(batch, device)
            out = model(batch_dev["pred_traj"], batch_dev["eta"])

            pred_pos_all.append(out["strike_pos"].detach().cpu().numpy().astype(np.float32))
            pred_radius_all.append(
                out["strike_radius"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            )
            pred_conf_all.append(
                out["strike_conf"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            )
            gt_pos_all.append(batch["gt_strike_pos"].numpy().astype(np.float32))
            gt_radius_all.append(batch["gt_strike_radius"].numpy().astype(np.float32))
            gt_conf_all.append(batch["gt_strike_conf"].numpy().astype(np.float32))
            eta_all.append(batch["eta"].numpy().astype(np.int64))
            if "cand_k" in batch:
                cand_k_all.append(batch["cand_k"].numpy().astype(np.int64))
            if "src_split" in batch:
                src_split_all.append(batch["src_split"].numpy().astype(np.int64))
            n_batches += 1

    elapsed = time.time() - t0

    pred_pos = np.concatenate(pred_pos_all, axis=0)            # [N, 3]
    gt_pos = np.concatenate(gt_pos_all, axis=0)                # [N, 3]
    pred_radius = np.concatenate(pred_radius_all, axis=0)      # [N]
    gt_radius = np.concatenate(gt_radius_all, axis=0)          # [N]
    pred_conf = np.concatenate(pred_conf_all, axis=0)          # [N]
    gt_conf = np.concatenate(gt_conf_all, axis=0)              # [N]
    eta_arr = np.concatenate(eta_all, axis=0)                  # [N]
    cand_k_arr = (
        np.concatenate(cand_k_all, axis=0) if cand_k_all else None
    )
    src_split_arr = (
        np.concatenate(src_split_all, axis=0) if src_split_all else None
    )

    N = pred_pos.shape[0]
    print(f"forward  : {n_batches} batches, {elapsed:.2f}s "
          f"({1000 * elapsed / max(1, N):.3f} ms/sample, "
          f"{1000 * elapsed / max(1, n_batches):.1f} ms/batch)")

    # ---- 误差 ----
    pos_err = np.linalg.norm(pred_pos - gt_pos, axis=-1)        # [N] km
    radius_signed = pred_radius - gt_radius                     # [N] km (>0 = overestimate)
    radius_abs = np.abs(radius_signed)                          # [N] km
    conf_signed = pred_conf - gt_conf                           # [N]
    conf_abs = np.abs(conf_signed)                              # [N]

    # ---- 全局摘要 ----
    print("\n--- Global metrics ---")
    print(_format_metric_row("pos_err (km)", pos_err))
    print(_format_metric_row("|radius_err| (km)", radius_abs))
    print(
        f"  radius_err signed: mean={radius_signed.mean():+.4f} km  "
        f"(>0 = pred overestimates)"
    )
    print(_format_metric_row("|conf_err|", conf_abs))
    print(
        f"  conf_err signed: mean={conf_signed.mean():+.4f}  "
        f"(>0 = pred is more confident than GT)"
    )

    pearson_radius = _safe_pearson(pred_radius, gt_radius)
    pearson_conf = _safe_pearson(pred_conf, gt_conf)
    print(
        f"\n  Pearson(pred_radius, gt_radius) = {pearson_radius:+.4f}  "
        f"(模型能否区分高/低不确定度场景；>0.5 优秀)"
    )
    print(
        f"  Pearson(pred_conf,   gt_conf)   = {pearson_conf:+.4f}"
    )

    # ---- 按 eta 分桶 ----
    eta_min_sec = int(cfg.get("data", {}).get("eta_min_sec", 0))
    eta_max_sec = int(cfg.get("data", {}).get("eta_max_sec", 600))
    eta_bins = [eta_min_sec, 60, 120, 240, 360, 480, eta_max_sec + 1]
    print("\n--- By eta bin ---")
    print(
        f"  {'eta_bin':<14}  {'n':>8}  {'pos_err':>10}  "
        f"{'|rad_err|':>10}  {'rad_signed':>11}  {'|conf_err|':>11}"
    )
    for i in range(len(eta_bins) - 1):
        lo, hi = eta_bins[i], eta_bins[i + 1]
        mask = (eta_arr >= lo) & (eta_arr < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        print(
            f"  [{lo:>3d},{hi:>4d})  {n_b:>8d}  "
            f"{float(pos_err[mask].mean()):>10.4f}  "
            f"{float(radius_abs[mask].mean()):>10.4f}  "
            f"{float(radius_signed[mask].mean()):>+11.4f}  "
            f"{float(conf_abs[mask].mean()):>11.4f}"
        )

    # ---- 按 cand_k 分组 ----
    if cand_k_arr is not None:
        print("\n--- By cand_k (top-1 / top-2 / top-3) ---")
        print(
            f"  {'cand_k':<8}  {'n':>8}  {'pos_err':>10}  "
            f"{'|rad_err|':>10}  {'rad_signed':>11}  {'|conf_err|':>11}"
        )
        for k in sorted(set(cand_k_arr.tolist())):
            mask = (cand_k_arr == k)
            n_b = int(mask.sum())
            print(
                f"  k={int(k)}      {n_b:>8d}  "
                f"{float(pos_err[mask].mean()):>10.4f}  "
                f"{float(radius_abs[mask].mean()):>10.4f}  "
                f"{float(radius_signed[mask].mean()):>+11.4f}  "
                f"{float(conf_abs[mask].mean()):>11.4f}"
            )

    # ---- 按 src_split 分组 ----
    if src_split_arr is not None:
        print("\n--- By src_split (来源 gnn1 split) ---")
        print(
            f"  {'src':<8}  {'n':>8}  {'pos_err':>10}  "
            f"{'|rad_err|':>10}  {'|conf_err|':>11}"
        )
        for ssid in sorted(set(src_split_arr.tolist())):
            mask = (src_split_arr == ssid)
            name = SRC_SPLIT_NAMES.get(int(ssid), str(ssid))
            n_b = int(mask.sum())
            print(
                f"  {name:<8}  {n_b:>8d}  "
                f"{float(pos_err[mask].mean()):>10.4f}  "
                f"{float(radius_abs[mask].mean()):>10.4f}  "
                f"{float(conf_abs[mask].mean()):>11.4f}"
            )

    # ---- 数值范围健康检查 ----
    radius_min_km = float(cfg.get("data", {}).get("radius_min_km", 0.5))
    radius_max_km = float(cfg.get("data", {}).get("radius_max_km", 10.0))
    print("\n--- Output range sanity ---")
    print(
        f"  pred_radius  ∈ [{float(pred_radius.min()):.3f}, "
        f"{float(pred_radius.max()):.3f}] km   "
        f"(config: [{radius_min_km}, {radius_max_km}])"
    )
    print(
        f"  pred_conf    ∈ [{float(pred_conf.min()):.3f}, "
        f"{float(pred_conf.max()):.3f}]"
    )

    out_of_range_r = int(((pred_radius < radius_min_km - 1e-3)
                          | (pred_radius > radius_max_km + 1e-3)).sum())
    out_of_range_c = int(((pred_conf < -1e-3) | (pred_conf > 1.0 + 1e-3)).sum())
    if out_of_range_r > 0:
        print(f"  [WARN] pred_radius 越界样本: {out_of_range_r}")
    if out_of_range_c > 0:
        print(f"  [WARN] pred_conf 越界样本: {out_of_range_c}")

    print("=" * 64)


def _format_metric_row(name: str, vals: np.ndarray) -> str:
    if vals.size == 0:
        return f"  {name:<22}  (empty)"
    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))
    return (
        f"  {name:<22}  mean={float(vals.mean()):.4f}   "
        f"p50={p50:.4f}   p90={p90:.4f}   max={float(vals.max()):.4f}"
    )


# ============================================================
# 模式 B：可视化评估
# ============================================================

def _grid_layout(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, int(np.ceil(n / 2))
    if n <= 9:
        return 3, int(np.ceil(n / 3))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _pick_indices_random(N: int, n_want: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n_want = min(n_want, N)
    return sorted(rng.choice(N, size=n_want, replace=False).tolist())


def _pick_indices_by_eta(
    eta_arr: np.ndarray, n_bins: int, eta_min: int, eta_max: int, seed: int,
) -> List[int]:
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


def _pick_indices_by_cand_k(
    cand_k_arr: np.ndarray, scene_idx_arr: np.ndarray, seed: int,
) -> List[int]:
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


def _plot_eval_sample(
    ax: plt.Axes,
    sample_idx: int,
    pred_traj: np.ndarray,           # [T, 6]
    eta_sec: int,
    gt_pos: np.ndarray,              # [3]
    gt_radius: float,
    gt_conf: float,
    pred_pos: np.ndarray,            # [3]
    pred_radius: float,
    pred_conf: float,
    cand_k: int,
    eta_idx: int,
    top_prob: float,
    src_split: int,
    scene_idx: int,
    time_step_s: float,
) -> None:
    T, _ = pred_traj.shape
    xy = pred_traj[:, :2]
    times_sec = np.array([(t + 1) * time_step_s for t in range(T)], dtype=np.float64)

    # 1) pred_traj 折线 + 渐变点
    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue",
            linewidth=1.4, alpha=0.6, zorder=2)
    cmap = plt.get_cmap("Blues")
    for t in range(T):
        c = cmap(0.35 + 0.55 * (t / max(1, T - 1)))
        ax.scatter(
            xy[t, 0], xy[t, 1], s=28, color=c,
            edgecolors="black", linewidths=0.5, zorder=4,
        )

    # 2) 离 eta 最近的 step 高亮
    step_float = eta_sec / time_step_s - 1.0
    if -1.0 <= step_float <= T - 1:
        nearest = int(np.clip(np.round(step_float), 0, T - 1))
        ax.scatter(
            xy[nearest, 0], xy[nearest, 1],
            s=140, facecolors="none", edgecolors="orange",
            linewidths=1.8, zorder=5,
            label=f"step≈eta (t={int(times_sec[nearest])}s)",
        )

    # 3) origin
    ax.scatter(0, 0, s=60, color="black", marker="*", zorder=6,
               label="enemy NOW (hist end)")

    # 4) GT strike
    ax.scatter(
        gt_pos[0], gt_pos[1], s=240, color="red",
        marker="*", edgecolors="black", linewidths=1.3, zorder=8,
        label=f"GT strike_pos (r={gt_radius:.2f}km)",
    )
    gt_circ = Circle(
        (gt_pos[0], gt_pos[1]),
        radius=gt_radius,
        fill=False, linestyle="--", linewidth=1.5,
        edgecolor="red", alpha=0.85, zorder=3,
    )
    ax.add_patch(gt_circ)

    # 5) Pred strike
    ax.scatter(
        pred_pos[0], pred_pos[1], s=160, color="orange",
        marker="^", edgecolors="black", linewidths=1.2, zorder=7,
        label=f"Pred strike_pos (r={pred_radius:.2f}km)",
    )
    pred_circ = Circle(
        (pred_pos[0], pred_pos[1]),
        radius=pred_radius,
        fill=False, linestyle="-", linewidth=1.5,
        edgecolor="orange", alpha=0.9, zorder=3,
    )
    ax.add_patch(pred_circ)

    # 6) 误差注释 + 标题
    pos_err = float(np.linalg.norm(pred_pos - gt_pos))
    rad_err = float(pred_radius - gt_radius)
    conf_err = float(pred_conf - gt_conf)

    src_name = SRC_SPLIT_NAMES.get(int(src_split), str(src_split))
    title = (
        f"sample {sample_idx}  scene={scene_idx}  cand_k={cand_k}\n"
        f"eta = {eta_sec} s   "
        f"|Δpos| = {pos_err:.3f} km   "
        f"Δradius = {rad_err:+.3f} km   "
        f"Δconf = {conf_err:+.3f}\n"
        f"GT:   r={gt_radius:.3f} km   conf={gt_conf:.3f}\n"
        f"Pred: r={pred_radius:.3f} km   conf={pred_conf:.3f}\n"
        f"top_prob={top_prob:.3f}   src={src_name}"
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)


def run_vis_eval(
    model: torch.nn.Module,
    ds: Gnn2Dataset,
    device: torch.device,
    cfg: Dict[str, Any],
    ckpt_path: Path,
    split: str,
    n_want: int,
    vis_mode: str,
    vis_indices: Optional[List[int]],
    seed: int,
    outdir: Path,
) -> None:
    print("\n" + "=" * 64)
    print(f"[Eval/GNN2 vis]  split={split}  ckpt={ckpt_path.name}  mode={vis_mode}")
    print("=" * 64)

    N = len(ds)
    eta_arr = ds.eta
    cand_k_arr = ds.cand_k if ds.cand_k is not None else None
    scene_idx_arr = ds.scene_idx if ds.scene_idx is not None else None

    eta_min_sec = int(cfg.get("data", {}).get("eta_min_sec", 0))
    eta_max_sec = int(cfg.get("data", {}).get("eta_max_sec", 600))

    if vis_indices:
        for i in vis_indices:
            if not (0 <= i < N):
                raise ValueError(f"--vis-indices {i} 越界 [0, {N})")
        chosen = list(vis_indices)
    elif vis_mode == "by-eta":
        chosen = _pick_indices_by_eta(
            eta_arr=eta_arr,
            n_bins=max(1, n_want),
            eta_min=eta_min_sec,
            eta_max=eta_max_sec,
            seed=seed,
        )
    elif vis_mode == "by-cand-k":
        if cand_k_arr is None or scene_idx_arr is None:
            raise RuntimeError("--vis-mode by-cand-k 需要 cand_k / scene_idx 字段")
        chosen = _pick_indices_by_cand_k(cand_k_arr, scene_idx_arr, seed)
    else:  # random
        chosen = _pick_indices_random(N, n_want, seed)

    if not chosen:
        print(f"[Eval/Vis] 没选到样本（mode={vis_mode}）")
        return
    print(f"[Eval/Vis] picked {len(chosen)} samples: {chosen}")

    # 一次性 forward
    batch_list = [ds[i] for i in chosen]
    batch = {
        k: torch.stack([b[k] for b in batch_list], dim=0)
        for k in batch_list[0].keys()
        if isinstance(batch_list[0][k], torch.Tensor)
    }
    batch_dev = _to_device(batch, device)

    model.eval()
    with torch.no_grad():
        out = model(batch_dev["pred_traj"], batch_dev["eta"])
    pred_pos = out["strike_pos"].detach().cpu().numpy()                       # [n, 3]
    pred_radius = out["strike_radius"].detach().cpu().numpy().reshape(-1)     # [n]
    pred_conf = out["strike_conf"].detach().cpu().numpy().reshape(-1)         # [n]

    pred_traj_all = batch["pred_traj"].numpy()                                # [n, T, 6]
    gt_pos_all = batch["gt_strike_pos"].numpy()                               # [n, 3]
    gt_radius_all = batch["gt_strike_radius"].numpy()                         # [n]
    gt_conf_all = batch["gt_strike_conf"].numpy()                             # [n]
    eta_arr_b = batch["eta"].numpy()                                          # [n]
    cand_k_b = batch["cand_k"].numpy() if "cand_k" in batch else np.zeros(len(chosen), np.int64)
    eta_idx_b = batch["eta_idx"].numpy() if "eta_idx" in batch else np.zeros(len(chosen), np.int64)
    top_prob_b = batch["top_prob"].numpy() if "top_prob" in batch else np.zeros(len(chosen), np.float32)
    src_split_b = batch["src_split"].numpy() if "src_split" in batch else np.zeros(len(chosen), np.int64)
    scene_idx_b = batch["scene_idx"].numpy() if "scene_idx" in batch else np.array(chosen, np.int64)

    time_step_s = float(cfg.get("data", {}).get("time_step_s", 60.0))

    # 画布
    n_plots = len(chosen)
    rows, cols = _grid_layout(n_plots)
    fig_w = max(5.4 * cols, 6.0)
    fig_h = max(5.4 * rows, 5.5)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for k, idx in enumerate(chosen):
        _plot_eval_sample(
            ax=axes[k],
            sample_idx=int(idx),
            pred_traj=pred_traj_all[k],
            eta_sec=int(eta_arr_b[k]),
            gt_pos=gt_pos_all[k],
            gt_radius=float(gt_radius_all[k]),
            gt_conf=float(gt_conf_all[k]),
            pred_pos=pred_pos[k],
            pred_radius=float(pred_radius[k]),
            pred_conf=float(pred_conf[k]),
            cand_k=int(cand_k_b[k]),
            eta_idx=int(eta_idx_b[k]),
            top_prob=float(top_prob_b[k]),
            src_split=int(src_split_b[k]),
            scene_idx=int(scene_idx_b[k]),
            time_step_s=time_step_s,
        )

    for k in range(n_plots, rows * cols):
        axes[k].axis("off")

    suptitle = (
        f"GNN2 eval | split={split} | n={n_plots} | mode={vis_mode} | "
        f"ckpt={ckpt_path.name}"
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"gnn2_eval_{split}_{vis_mode}_{stamp}.png"
    fig.savefig(out_path, dpi=140)
    print(f"[Eval/Vis] saved: {out_path}")
    plt.close(fig)


# ============================================================
# main
# ============================================================

def _parse_indices(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN2 evaluator (batch metrics OR visualization).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="",
                        help="ckpt 路径（.pt 或目录）；为空时自动取 config.train.ckpt_dir 下最新 .pt")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=0,
                        help="批量评估的 batch_size；0 = 用 config.train.batch_size")

    # ---- 可视化模式 ----
    parser.add_argument("--vis", action="store_true",
                        help="开启可视化模式（不算批量指标）")
    parser.add_argument("--vis-num", type=int, default=6,
                        help="可视化样本数；by-eta 模式下为桶数")
    parser.add_argument("--vis-mode", type=str, default="random",
                        choices=["random", "by-eta", "by-cand-k"],
                        help="挑样本方式（被 --vis-indices 覆盖）")
    parser.add_argument("--vis-indices", type=str, default="",
                        help="逗号分隔的样本索引")
    parser.add_argument("--vis-seed", type=int, default=42)
    parser.add_argument("--vis-outdir", type=str, default="eval_vis",
                        help="可视化 png 输出目录（相对 gnn2 根目录）")
    args = parser.parse_args()

    gnn2_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = gnn2_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {}) or {}
    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    # ---- ckpt ----
    ckpt_path = _resolve_ckpt(args.ckpt, gnn2_root, train_cfg)
    print(f"[Info] ckpt   = {ckpt_path}")

    # ---- 模型 ----
    model = build_model_from_config(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    miss, unex = model.load_state_dict(sd, strict=False)
    if miss:
        print(f"[Info] ckpt missing keys ({len(miss)}): {miss[:3]}{'...' if len(miss) > 3 else ''}")
    if unex:
        print(f"[Info] ckpt unexpected keys ({len(unex)}): {unex[:3]}{'...' if len(unex) > 3 else ''}")
    model.eval()

    # ---- 数据 ----
    rel_cfg = (
        cfg_path.relative_to(gnn2_root) if cfg_path.is_absolute() else Path(args.config)
    )
    train_ds, val_ds, test_ds = build_datasets_from_config(str(rel_cfg))
    ds = pick_split(args.split, train_ds, val_ds, test_ds)
    print(f"[Info] split  = '{args.split}', size = {len(ds)}")

    # ---- 路由模式 ----
    if args.vis:
        vis_indices = _parse_indices(args.vis_indices) if args.vis_indices else None
        vis_outdir = Path(args.vis_outdir)
        if not vis_outdir.is_absolute():
            vis_outdir = (gnn2_root / vis_outdir).resolve()
        run_vis_eval(
            model=model, ds=ds, device=device, cfg=cfg, ckpt_path=ckpt_path,
            split=args.split,
            n_want=int(args.vis_num),
            vis_mode=args.vis_mode,
            vis_indices=vis_indices,
            seed=int(args.vis_seed),
            outdir=vis_outdir,
        )
    else:
        bs = int(args.batch_size) or int(train_cfg.get("batch_size", 256))
        run_batch_eval(
            model=model, ds=ds, device=device, cfg=cfg, ckpt_path=ckpt_path,
            split=args.split,
            batch_size=bs,
        )


if __name__ == "__main__":
    main()
