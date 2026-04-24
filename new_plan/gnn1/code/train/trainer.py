#!/usr/bin/env python3
"""
gnn1/code/train/trainer.py
--------------------------
GNN1 训练入口。

用法（在 new_plan/gnn1/ 下）：
    $env:PYTHONPATH = "$PWD/code"
    # 冒烟测试（不依赖 .npz，造随机数据走一遍）
    python -m train.trainer --smoke
    # 正式训练
    python -m train.trainer --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                    # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from train.model import Gnn1Config, Gnn1Selector, build_model_from_config  # noqa: E402
from train.loss import Gnn1Loss, Gnn1LossConfig, build_loss_from_config    # noqa: E402


# =================== 工具 ===================

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    dev = train_cfg.get("device", "auto").lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =================== ckpt 管理 ===================

def maybe_save_topk(
    model: nn.Module,
    ckpt_dir: Path,
    epoch_idx: int,
    val_loss: float,
    saved: List[Tuple[float, int, Path]],
    keep_top_k: int,
) -> bool:
    """动态维护 top-K（按 val_loss 最低）。参考 lstm1/trainer.py。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qualifies = len(saved) < keep_top_k or val_loss < saved[-1][0]
    if not qualifies:
        return False
    path = ckpt_dir / f"best_gnn1_epoch{epoch_idx:03d}_valloss{val_loss:.4f}.pt"
    torch.save(model.state_dict(), path)
    saved.append((val_loss, epoch_idx, path))
    saved.sort(key=lambda x: x[0])
    while len(saved) > keep_top_k:
        worst_loss, _, worst_path = saved.pop()
        try:
            if worst_path.exists():
                worst_path.unlink()
            print(f"[ckpt] dropped {worst_path.name} (val_loss={worst_loss:.4f})")
        except OSError as e:
            print(f"[ckpt][WARN] 删除失败 {worst_path}: {e}")
    top_desc = ", ".join(f"ep{e}:{l:.4f}" for l, e, _ in saved)
    print(f"[ckpt] saved {path.name} | Top-{keep_top_k}: [{top_desc}]")
    return True


# =================== 训练 / 验证循环 ===================

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    log_interval: int = 100,
    grad_clip: float = 0.0,
    num_modes: int = 5,
) -> Dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_top1 = 0.0
    total_samples = 0
    total_batches = 0
    grad_norm_sum = 0.0

    label_counts = torch.zeros(num_modes, dtype=torch.long, device=device)
    pred_counts = torch.zeros(num_modes, dtype=torch.long, device=device)

    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        B = int(batch["label"].size(0))

        optimizer.zero_grad()
        out = model(batch)                           # logits [B, M]
        loss, comps = loss_fn(out["logits"], batch["label"])
        loss.backward()

        # 梯度裁剪
        if grad_clip and grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
        else:
            with torch.no_grad():
                total_norm = torch.sqrt(
                    sum((p.grad.detach() ** 2).sum()
                        for p in model.parameters() if p.grad is not None)
                )
        grad_norm_sum += float(total_norm)

        optimizer.step()

        # 统计
        loss_val = float(loss.detach().cpu().item())
        top1_val = float(comps["top1"].detach().cpu().item())
        total_loss += loss_val
        total_top1 += top1_val * B
        total_samples += B
        total_batches += 1

        with torch.no_grad():
            pred = torch.argmax(out["logits"], dim=-1)
            label_counts += torch.bincount(batch["label"], minlength=num_modes)
            pred_counts += torch.bincount(pred, minlength=num_modes)

        if log_interval > 0 and (step % log_interval == 0):
            elapsed = time.time() - t0
            print(f"[Epoch {epoch_idx:03d}] step {step:05d}  "
                  f"loss = {loss_val:.4f}  top1 = {top1_val:.3f}  "
                  f"|grad| = {float(total_norm):.3f}  elapsed = {elapsed:.1f}s")

    avg_loss = total_loss / max(1, total_batches)
    avg_top1 = total_top1 / max(1, total_samples)
    grad_norm_mean = grad_norm_sum / max(1, total_batches)
    label_dist = (label_counts.float() / max(1, total_samples)).cpu().tolist()
    pred_dist = (pred_counts.float() / max(1, total_samples)).cpu().tolist()
    return {
        "avg_loss": avg_loss,
        "avg_top1": avg_top1,
        "grad_norm_mean": grad_norm_mean,
        "label_dist": label_dist,
        "pred_dist": pred_dist,
    }


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    num_modes: int = 5,
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_samples = 0
    total_batches = 0
    label_counts = torch.zeros(num_modes, dtype=torch.long, device=device)
    pred_counts = torch.zeros(num_modes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            B = int(batch["label"].size(0))
            out = model(batch)
            loss, comps = loss_fn(out["logits"], batch["label"])
            total_loss += float(loss.detach().cpu().item())
            total_top1 += float(comps["top1"].detach().cpu().item()) * B
            total_samples += B
            total_batches += 1
            pred = torch.argmax(out["logits"], dim=-1)
            label_counts += torch.bincount(batch["label"], minlength=num_modes)
            pred_counts += torch.bincount(pred, minlength=num_modes)

    avg_loss = total_loss / max(1, total_batches)
    avg_top1 = total_top1 / max(1, total_samples)
    label_dist = (label_counts.float() / max(1, total_samples)).cpu().tolist()
    pred_dist = (pred_counts.float() / max(1, total_samples)).cpu().tolist()
    print(f"[Epoch {epoch_idx:03d}] Val loss = {avg_loss:.4f}  "
          f"top1 = {avg_top1:.3f}  "
          f"label_dist = [{', '.join(f'{x:.2f}' for x in label_dist)}]  "
          f"pred_dist  = [{', '.join(f'{x:.2f}' for x in pred_dist)}]")
    return {"avg_loss": avg_loss, "avg_top1": avg_top1,
            "label_dist": label_dist, "pred_dist": pred_dist}


# =================== 总控 ===================

def train(config_path: str) -> None:
    # 延迟 import（避免 --smoke 也要求 numpy .npz 数据）
    from data.dataset import build_datasets_from_config  # noqa: F401

    gnn1_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    set_seed(int(train_cfg.get("seed", 42)))
    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    # ---- Datasets ----
    rel_cfg = cfg_path.relative_to(gnn1_root) if cfg_path.is_absolute() else Path(config_path)
    train_ds, val_ds, test_ds = build_datasets_from_config(str(rel_cfg))
    print(f"[Info] datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    bs = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model / Loss / Optimizer ----
    model = build_model_from_config(cfg).to(device)
    loss_fn = build_loss_from_config(cfg)
    print(f"[Info] params = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Info] loss   = {loss_fn.cfg}")

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    num_epochs = int(train_cfg.get("num_epochs", 20))
    log_interval = int(train_cfg.get("log_interval", 100))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    keep_top_k = int(train_cfg.get("keep_top_k", 3))
    num_modes = int(model_cfg.get("n_modes", 5))

    run_id = time.strftime("%Y%m%d%H%M%S")
    ckpt_root = (gnn1_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
    ckpt_dir = ckpt_root / run_id
    print(f"[Info] ckpt dir = {ckpt_dir} (keep top-{keep_top_k})")

    saved: List[Tuple[float, int, Path]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")
        t_ep = time.time()
        tr = train_one_epoch(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            loader=train_loader, device=device, epoch_idx=epoch,
            log_interval=log_interval, grad_clip=grad_clip, num_modes=num_modes,
        )
        print(f"[Epoch {epoch:03d}] Train loss = {tr['avg_loss']:.4f}  "
              f"top1 = {tr['avg_top1']:.3f}  "
              f"|grad| = {tr['grad_norm_mean']:.3f}  "
              f"pred_dist = [{', '.join(f'{x:.2f}' for x in tr['pred_dist'])}]  "
              f"time = {time.time() - t_ep:.1f}s")

        ev = evaluate(model, loss_fn, val_loader, device, epoch, num_modes=num_modes)
        val_loss = ev["avg_loss"]

        maybe_save_topk(model, ckpt_dir, epoch, val_loss, saved, keep_top_k)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    print(f"\n[Finished] Best epoch = {best_epoch}  val_loss = {best_val_loss:.4f}")
    if saved:
        print(f"[Finished] Top-{keep_top_k} checkpoints:")
        for rank, (l, e, p) in enumerate(saved, start=1):
            print(f"  #{rank}  epoch={e:03d}  val_loss={l:.4f}  {p.name}")


# =================== Smoke ===================

def run_smoke(cfg: Dict[str, Any]) -> None:
    print("=" * 60)
    print("[Smoke/GNN1] start")
    print("=" * 60)

    device = torch.device("cpu")
    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke] params = {sum(p.numel() for p in model.parameters()):,}")

    model_cfg = cfg.get("model", {})
    n_modes = int(model_cfg.get("n_modes", 5))
    fut_len = int(model_cfg.get("fut_len", 10))
    feat_dim = int(model_cfg.get("feat_dim", 6))

    data_cfg = cfg.get("data", {})
    type_hi = int((data_cfg.get("type_range") or [0, 2])[1])

    B = 4
    torch.manual_seed(0)
    batch = {
        "cand_trajs": torch.randn(B, n_modes, fut_len, feat_dim),
        "task_type": torch.zeros(B, dtype=torch.long),
        "type": torch.randint(0, type_hi + 1, (B,), dtype=torch.long),
        "position": torch.randn(B, 3),
        "label": torch.randint(0, n_modes, (B,), dtype=torch.long),
    }

    out = model(batch)
    assert out["logits"].shape == (B, n_modes)
    print(f"[Smoke] logits shape = {tuple(out['logits'].shape)}  ok")

    loss_fn = build_loss_from_config(cfg)
    loss, comps = loss_fn(out["logits"], batch["label"])
    print(f"[Smoke] loss = {float(loss):.4f}  top1 = {float(comps['top1']):.3f}")

    loss.backward()
    # 查几个关键模块的梯度
    for name in ["cand_encoder.weight_ih_l0", "attn.in_proj_weight", "score_head.net.0.weight"]:
        p = dict(model.named_parameters()).get(name)
        if p is not None and p.grad is not None:
            print(f"[Smoke] grad_norm({name}) = {p.grad.norm().item():.4f}")

    print("=" * 60)
    print("[Smoke/GNN1] OK")
    print("=" * 60)


# =================== CLI ===================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        gnn1_root = Path(__file__).resolve().parents[2]
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = gnn1_root / cfg_path
        cfg = load_config(str(cfg_path))
        run_smoke(cfg)
        return

    train(args.config)


if __name__ == "__main__":
    main()
