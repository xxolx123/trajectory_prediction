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

用法（在 new_plan/gnn1/ 下）:
    $env:PYTHONPATH = "$PWD/code"
    python -m train.eval --config config.yaml --split test [--ckpt <path>]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

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


# =================== 评估主流程 ===================

def evaluate(config_path: str, ckpt_path: Optional[str], split: str) -> None:
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


# =================== CLI ===================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.config, args.ckpt or None, args.split)


if __name__ == "__main__":
    main()
