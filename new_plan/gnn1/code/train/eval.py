#!/usr/bin/env python3
"""
gnn1/code/train/eval.py
-----------------------
评估 GNN1：
  1) 分类指标：top-1 accuracy（argmax(logits) == label）
  2) 目标匹配指标：所选 candidate 的 endpoint 到 position 的距离
     对比三种选择策略：
         - GNN1:  argmax(logits)
         - Random: 均匀随机（理论期望：等概率）
         - Oracle: label（= argmin(endpoint→position)）
  3) 若 cache 中保存了 targets（GT 未来），再算 ADE / FDE（km）
     对比同样三种策略。
  4) 可选：从该 split 随机挑 --vis-n 个样本画可视化（含 5 条候选的概率 +
     top-3 重归一化概率），保存到 --vis-outdir。

用法（在 new_plan/gnn1/ 下）:
    $env:PYTHONPATH = "$PWD/code"
    python -m train.eval --config config.yaml --split test
    python -m train.eval --config config.yaml --split test --vis-n 0    # 只评估不画图
    python -m train.eval --config config.yaml --split test --vis-n 8 --vis-outdir eval_vis
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                    # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from data.dataset import build_datasets_from_config, Gnn1Dataset  # noqa: E402
from train.model import build_model_from_config                   # noqa: E402


# =================== 通用工具 ===================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    dev = train_cfg.get("device", "auto").lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_ckpt(ckpt_root: Path) -> Optional[Path]:
    if ckpt_root.is_file():
        return ckpt_root
    if not ckpt_root.exists():
        return None
    candidates = list(ckpt_root.rglob("*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def pick_split(split: str, train_ds, val_ds, test_ds):
    s = split.lower()
    if s == "train":
        return train_ds
    if s in ("val", "valid", "validation"):
        return val_ds
    if s == "test":
        return test_ds
    raise ValueError(f"unknown split: {split}")


# =================== Scaler 反归一化（和 generate_data 同口径） ===================

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


def decode_to_xy(feat_norm: np.ndarray, scaler: _Scaler) -> np.ndarray:
    """[*, T, D] → [*, T, 2] 先反归一化再对 xy delta 做 cumsum；原点取 (0,0)。"""
    orig = scaler.inverse_transform(feat_norm.astype(np.float64))
    dxy = orig[..., :2]
    return np.cumsum(dxy, axis=-2)


# =================== 可视化 ===================

_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]


def _grid_layout(n: int):
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, int(np.ceil(n / 2))
    if n <= 9:
        return 3, int(np.ceil(n / 3))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _plot_eval_sample(
    ax: plt.Axes,
    sample_idx: int,
    cand_xy: np.ndarray,           # [M, T, 2]
    position: np.ndarray,          # [3]
    label: int,
    k_seed: int,
    task_type: int,
    type_id: int,
    gt_xy: Optional[np.ndarray],   # [T, 2] or None
    probs: np.ndarray,             # [M]   仅诊断用，可视化不读
    top_idx: np.ndarray,           # [K]   GNN1 选的 top-K 索引（降序）
    top_probs: np.ndarray,         # [K]   top-K 重归一化概率（和 = 1）
    noise_sigma_km: float,
) -> None:
    """只画 top-K 候选（K = top_idx 长度），被 GNN1 截掉的 M-K 条不画。"""
    _ = probs  # 故意不读：GNN1 对外只输出 top_idx / top_probs
    top1 = int(top_idx[0])
    correct = (top1 == label)
    label_in_top = (label in [int(i) for i in top_idx])

    # 候选轨迹：只画 top-K
    for rank, m_t in enumerate(top_idx):
        m = int(m_t)
        is_label = (m == label)
        is_top1 = (rank == 0)
        color = _PALETTE[m % len(_PALETTE)]

        # 线宽：top1 最粗，其它 top-k 次粗
        lw = 2.8 if is_top1 else 2.0
        ax.plot(cand_xy[m, :, 0], cand_xy[m, :, 1],
                color=color, linewidth=lw, alpha=1.0, zorder=3)

        # 端点：top1 = 五角星；label = 实心圆边框（如果它在 top-K 里）；其余 = 方块
        ex, ey = cand_xy[m, -1, 0], cand_xy[m, -1, 1]
        if is_top1:
            ax.scatter(ex, ey, s=140, color=color,
                       edgecolors="black", linewidths=1.4,
                       marker="*", zorder=6)
        elif is_label:
            ax.scatter(ex, ey, s=95, color=color,
                       edgecolors="black", linewidths=1.4,
                       marker="o", zorder=5)
        else:
            ax.scatter(ex, ey, s=70, color=color,
                       edgecolors="black", linewidths=1.0,
                       marker="s", zorder=4)

        # 端点旁边标重归一化概率
        prob_txt = f"cand{m}: {float(top_probs[rank]) * 100:.1f}%"
        if is_top1:
            prob_txt = "★" + prob_txt
        if is_label:
            prob_txt += "  (label)"
        ax.annotate(prob_txt, xy=(ex, ey),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=color, weight="bold")

    # 2) 原点
    ax.scatter(0, 0, s=55, color="black", marker="*", zorder=6)

    # 3) position
    ax.scatter(position[0], position[1], s=230, color="red",
               marker="*", edgecolors="black", linewidths=1.2, zorder=7)
    if noise_sigma_km > 0:
        circ = Circle(
            (position[0], position[1]),
            radius=noise_sigma_km,
            fill=False, linestyle="--", linewidth=1.0,
            edgecolor="red", alpha=0.55, zorder=1,
        )
        ax.add_patch(circ)

    # 4) GT
    if gt_xy is not None:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1],
                color="black", linestyle="--", linewidth=1.4,
                alpha=0.85, zorder=3)
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1],
                   s=12, color="black", marker="o", zorder=4)

    # 5) 标题 + top-3 信息
    top3_str = ", ".join(
        f"c{int(i)}:{float(p) * 100:.1f}%"
        for i, p in zip(top_idx, top_probs)
    )
    if correct:
        flag = "OK"
    elif label_in_top:
        flag = "miss-top1 (label in top-K)"
    else:
        flag = "label NOT in top-K"
    title = (
        f"idx={sample_idx}  task={task_type}  type={type_id}\n"
        f"label={label}  top1={top1} ({flag})  k_seed={k_seed}\n"
        f"top-K (renorm): [{top3_str}]"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)


def _render_eval_samples(
    ds,
    model: torch.nn.Module,
    device: torch.device,
    scaler: Optional[_Scaler],
    chosen: List[int],
    noise_sigma_km: float,
    split: str,
    ckpt_name: str,
    outdir: Path,
    show: bool = False,
) -> None:
    n = len(chosen)
    if n == 0:
        return
    if scaler is None:
        print("[Eval/Vis] 没找到 scaler，可视化需要反归一化；跳过。")
        return

    # 一次性 forward 这 n 个样本
    batch_list = [ds[i] for i in chosen]
    batch = {
        k: torch.stack([b[k] for b in batch_list], dim=0)
        for k in batch_list[0].keys()
    }
    batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        out = model(batch_dev)
    probs = out["probs"].cpu().numpy()            # [n, M]
    top_idx = out["top_idx"].cpu().numpy()        # [n, K]
    top_probs = out["top_probs"].cpu().numpy()    # [n, K]

    cand_np = batch["cand_trajs"].numpy()          # [n, M, T, D]
    cand_xy = decode_to_xy(cand_np, scaler)        # [n, M, T, 2]

    have_gt = ("targets" in batch)
    if have_gt:
        gt_np = batch["targets"].numpy()           # [n, T, D]
        gt_xy = decode_to_xy(gt_np, scaler)        # [n, T, 2]
    else:
        gt_xy = None

    pos_np = batch["position"].numpy()             # [n, 3]
    labels = batch["label"].numpy()
    task_arr = batch["task_type"].numpy()
    type_arr = batch["type"].numpy()
    if "k_seed" in batch:
        k_seed_arr = batch["k_seed"].numpy()
    else:
        k_seed_arr = np.full(n, -1, dtype=np.int64)

    rows, cols = _grid_layout(n)
    fig_w = max(5.0 * cols, 5.5)
    fig_h = max(4.8 * rows, 5.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for k, idx in enumerate(chosen):
        _plot_eval_sample(
            ax=axes[k],
            sample_idx=int(idx),
            cand_xy=cand_xy[k],
            position=pos_np[k],
            label=int(labels[k]),
            k_seed=int(k_seed_arr[k]),
            task_type=int(task_arr[k]),
            type_id=int(type_arr[k]),
            gt_xy=(gt_xy[k] if have_gt else None),
            probs=probs[k],
            top_idx=top_idx[k],
            top_probs=top_probs[k],
            noise_sigma_km=noise_sigma_km,
        )
    for k in range(n, rows * cols):
        axes[k].axis("off")

    fig.suptitle(
        f"GNN1 eval | split={split} | n={n} | ckpt={ckpt_name}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"gnn1_eval_{split}_{stamp}.png"
    fig.savefig(out_path, dpi=140)
    print(f"[Eval/Vis] saved: {out_path}")
    print(f"[Eval/Vis] indices = {chosen}")
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
            plt.show()
        except Exception as e:
            print(f"[Eval/Vis] show 失败：{e}")
    plt.close(fig)


# =================== 评估主流程 ===================

def evaluate(
    config_path: str,
    ckpt_path: Optional[str],
    split: str,
    vis_n: int = 5,
    vis_outdir: str = "eval_vis",
    vis_seed: int = 42,
    vis_indices: Optional[List[int]] = None,
) -> None:
    gnn1_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    # ---- 数据 ----
    rel_cfg = cfg_path.relative_to(gnn1_root) if cfg_path.is_absolute() else Path(config_path)
    train_ds, val_ds, test_ds = build_datasets_from_config(str(rel_cfg))
    ds: Gnn1Dataset = pick_split(split, train_ds, val_ds, test_ds)
    print(f"[Info] using split='{split}', size={len(ds)}")

    bs = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 128)))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))

    # ---- 模型 ----
    model = build_model_from_config(cfg).to(device)
    if ckpt_path:
        p = Path(ckpt_path)
        if not p.is_absolute():
            p = (gnn1_root / p).resolve()
    else:
        ckpt_root = (gnn1_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
        p = find_latest_ckpt(ckpt_root)
        if p is None:
            raise FileNotFoundError(f"没找到 ckpt（{ckpt_root}）。请 --ckpt 显式指定。")
    print(f"[Info] loading ckpt: {p}")
    state = torch.load(p, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- Scaler（eval 里只为 ADE/FDE 需要） ----
    scaler_path = (gnn1_root / data_cfg.get("cache_dir", "data/cache") / "scaler_posvel.npz").resolve()
    scaler = _Scaler.load(scaler_path) if scaler_path.exists() else None
    if scaler is None:
        print(f"[Warn] 没找到 {scaler_path}，无法算 ADE/FDE，只汇报 top1 + 距离")

    M = int(model_cfg.get("n_modes", 5))
    rng = np.random.default_rng(int(train_cfg.get("seed", 42)))

    # ---- 聚合器 ----
    total_samples = 0
    total_top1 = 0

    # 选择策略产生的 endpoint→position 距离（km）
    dist_gnn: list = []
    dist_rand: list = []
    dist_oracle: list = []

    # ADE/FDE（km），仅当 targets + scaler 都可用时
    have_targets = ds.targets is not None and scaler is not None
    ade_gnn: list = []
    fde_gnn: list = []
    ade_rand: list = []
    fde_rand: list = []
    ade_oracle: list = []
    fde_oracle: list = []

    t_start = time.time()
    with torch.no_grad():
        for batch in loader:
            # to device
            batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch_dev)
            logits = out["logits"]                       # [B, M]
            gnn_pick = torch.argmax(logits, dim=-1).cpu().numpy()  # [B]
            labels = batch_dev["label"].cpu().numpy()             # [B]
            B = int(labels.shape[0])

            total_samples += B
            total_top1 += int((gnn_pick == labels).sum())

            # --- endpoint→position 距离 ---
            # cand_trajs 是归一化 + delta，需要解码
            cand_np = batch["cand_trajs"].cpu().numpy()     # [B, M, T, D]
            pos_np = batch["position"].cpu().numpy()        # [B, 3]
            # scaler 可能为 None（但距离计算没 scaler 也能用"归一化空间"代替，
            # 这里给一个温和的回退：没 scaler 时，dist 用归一化空间的 xy cumsum，和 position 的前两维比较）
            if scaler is not None:
                cand_xy = decode_to_xy(cand_np, scaler)     # [B, M, T, 2]
            else:
                # 归一化空间的"代理 xy"：直接 cumsum 前两维
                cand_xy = np.cumsum(cand_np[..., :2].astype(np.float64), axis=-2)
            endpoints = cand_xy[:, :, -1, :]                # [B, M, 2]

            pos_xy = pos_np[:, :2]                          # [B, 2]
            diff_to_pos = endpoints - pos_xy[:, None, :]    # [B, M, 2]
            dist_pm = np.linalg.norm(diff_to_pos, axis=-1)  # [B, M]

            idx = np.arange(B)
            rand_pick = rng.integers(0, M, size=B)
            dist_gnn.extend(dist_pm[idx, gnn_pick].tolist())
            dist_rand.extend(dist_pm[idx, rand_pick].tolist())
            dist_oracle.extend(dist_pm[idx, labels].tolist())  # label 就是 argmin → oracle

            # --- ADE / FDE（需要 GT + scaler） ---
            if have_targets:
                target_np = batch["targets"].cpu().numpy()  # [B, T, D]
                gt_xy = decode_to_xy(target_np, scaler)     # [B, T, 2]
                cand_full_xy = cand_xy                      # [B, M, T, 2]  已算好
                diff_gt = cand_full_xy - gt_xy[:, None, :, :]   # [B, M, T, 2]
                dist_t = np.linalg.norm(diff_gt, axis=-1)   # [B, M, T]
                ade_pm = dist_t.mean(axis=-1)               # [B, M]
                fde_pm = dist_t[..., -1]                    # [B, M]

                ade_gnn.extend(ade_pm[idx, gnn_pick].tolist())
                fde_gnn.extend(fde_pm[idx, gnn_pick].tolist())
                ade_rand.extend(ade_pm[idx, rand_pick].tolist())
                fde_rand.extend(fde_pm[idx, rand_pick].tolist())
                # 这里 oracle 也按 label（即 endpoint-to-position 最近），不是 argmin-ADE；
                # 如果想要"真正 ADE 最小的 mode"做对照，独立 argmin：
                ade_oracle.extend(ade_pm.min(axis=-1).tolist())
                fde_oracle.extend(fde_pm.min(axis=-1).tolist())

    elapsed = time.time() - t_start
    top1 = total_top1 / max(1, total_samples)

    print("\n" + "=" * 60)
    print(f"[Eval/GNN1]  split = {split}   ckpt = {p.name}")
    print("=" * 60)
    print(f"samples = {total_samples}   eval_time = {elapsed:.1f}s")
    print(f"top-1 accuracy = {top1:.4f}   (random baseline ≈ {1.0 / M:.4f})")

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    print("\n--- endpoint→position distance (km, 越小越好) ---")
    print(f"GNN1 选  (argmax) : {_mean(dist_gnn):.4f}")
    print(f"Random  选        : {_mean(dist_rand):.4f}")
    print(f"Oracle  选 (label): {_mean(dist_oracle):.4f}   # 理论下界")

    if have_targets and ade_gnn:
        print("\n--- ADE / FDE vs GT trajectory (km) ---")
        print(f"              {'ADE':>10s}  {'FDE':>10s}")
        print(f"GNN1 (argmax) {_mean(ade_gnn):>10.4f}  {_mean(fde_gnn):>10.4f}")
        print(f"Random        {_mean(ade_rand):>10.4f}  {_mean(fde_rand):>10.4f}")
        print(f"Oracle (minADE/minFDE, 独立 min) "
              f"{_mean(ade_oracle):>8.4f}  {_mean(fde_oracle):>10.4f}")

    print("=" * 60)

    # ---- 可视化 N 个样本（含 5 条候选的概率 + top-3 重归一化概率） ----
    if vis_n > 0 or vis_indices:
        if vis_indices:
            for i in vis_indices:
                if not (0 <= i < len(ds)):
                    raise ValueError(f"vis_indices 里的 {i} 越界 [0, {len(ds)})")
            chosen = list(vis_indices)
        else:
            rng_vis = np.random.default_rng(vis_seed)
            n_want = min(int(vis_n), len(ds))
            chosen = sorted(rng_vis.choice(len(ds), size=n_want, replace=False).tolist())

        vis_dir = Path(vis_outdir)
        if not vis_dir.is_absolute():
            vis_dir = (gnn1_root / vis_dir).resolve()

        _render_eval_samples(
            ds=ds,
            model=model,
            device=device,
            scaler=scaler,
            chosen=chosen,
            noise_sigma_km=float(data_cfg.get("position_noise_km", 0.3)),
            split=split,
            ckpt_name=p.name,
            outdir=vis_dir,
            show=False,
        )


# =================== CLI ===================

def _parse_indices(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--vis-n", type=int, default=5,
                        help="评估完后额外画 N 个样本（默认 5；设为 0 关闭）")
    parser.add_argument("--vis-indices", type=str, default="",
                        help="逗号分隔的样本索引（会覆盖 --vis-n 的随机挑法）")
    parser.add_argument("--vis-seed", type=int, default=42,
                        help="随机挑样本用的 seed")
    parser.add_argument("--vis-outdir", type=str, default="eval_vis",
                        help="可视化 png 保存目录（相对 gnn1 根目录）")
    args = parser.parse_args()

    vis_indices = _parse_indices(args.vis_indices) if args.vis_indices else None
    evaluate(
        args.config, args.ckpt or None, args.split,
        vis_n=int(args.vis_n),
        vis_outdir=args.vis_outdir,
        vis_seed=int(args.vis_seed),
        vis_indices=vis_indices,
    )


if __name__ == "__main__":
    main()
