#!/usr/bin/env python3
"""
gnn2/code/train/trainer.py
--------------------------
GNN2 训练入口。镜像 gnn1/lstm2 的 trainer 风格，保留 --smoke。

用法（在 new_plan/gnn2/ 下）::

    # Windows PowerShell
    $env:PYTHONPATH = "$PWD/code"
    # Linux/macOS
    # export PYTHONPATH="$PWD/code"

    # 冒烟测试（不依赖 .npz，构造随机输入跑一遍 forward + 真 loss + backward）
    python -m train.trainer --smoke

    # 正式训练（前置：先在 gnn1/ 下跑 cache_lstm1_preds.py + generate_data.py，
    # 再在 gnn2/ 下跑 `python -m data.generate_data`）
    python -m train.trainer --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                    # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from train.model import build_model_from_config                                  # noqa: E402
from train.loss import StrikeLoss, build_loss_from_config                        # noqa: E402


# =================== 工具 ===================

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
    """动态维护 top-K（按 val_loss 最低）。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qualifies = len(saved) < keep_top_k or val_loss < saved[-1][0]
    if not qualifies:
        return False
    path = ckpt_dir / f"best_gnn2_epoch{epoch_idx:03d}_valloss{val_loss:.4f}.pt"
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

def _model_forward(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(batch["pred_traj"], batch["eta"])


def train_one_epoch(
    model: nn.Module,
    loss_fn: StrikeLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    log_interval: int = 50,
    grad_clip: float = 0.0,
) -> Dict[str, Any]:
    model.train()

    sum_total = 0.0
    sum_pos = 0.0
    sum_radius = 0.0
    sum_conf = 0.0
    n_batches = 0
    n_samples = 0
    grad_norm_sum = 0.0

    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        B = int(batch["pred_traj"].size(0))

        optimizer.zero_grad()
        out = _model_forward(model, batch)
        loss, comps = loss_fn(out, batch)
        loss.backward()

        if grad_clip and grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
        else:
            with torch.no_grad():
                total_norm = torch.sqrt(
                    sum(
                        (p.grad.detach() ** 2).sum()
                        for p in model.parameters() if p.grad is not None
                    )
                )
        grad_norm_sum += float(total_norm)

        optimizer.step()

        sum_total += float(comps["L"].cpu().item())
        sum_pos += float(comps["pos"].cpu().item())
        sum_radius += float(comps["radius"].cpu().item())
        sum_conf += float(comps["conf"].cpu().item())
        n_batches += 1
        n_samples += B

        if log_interval > 0 and (step % log_interval == 0):
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch_idx:03d}] step {step:05d}  "
                f"L={float(comps['L']):.4f}  "
                f"pos={float(comps['pos']):.4f}  "
                f"rad={float(comps['radius']):.4f}  "
                f"conf={float(comps['conf']):.4f}  "
                f"|grad|={float(total_norm):.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

    return {
        "avg_total":      sum_total / max(1, n_batches),
        "avg_pos":        sum_pos / max(1, n_batches),
        "avg_radius":     sum_radius / max(1, n_batches),
        "avg_conf":       sum_conf / max(1, n_batches),
        "grad_norm_mean": grad_norm_sum / max(1, n_batches),
        "n_samples":      n_samples,
    }


def evaluate(
    model: nn.Module,
    loss_fn: StrikeLoss,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
) -> Dict[str, Any]:
    model.eval()

    sum_total = 0.0
    sum_pos = 0.0
    sum_radius = 0.0
    sum_conf = 0.0
    n_batches = 0
    n_samples = 0

    # 物理范围健康检查
    radius_min_seen = float("inf")
    radius_max_seen = -float("inf")
    conf_min_seen = float("inf")
    conf_max_seen = -float("inf")

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            B = int(batch["pred_traj"].size(0))
            out = _model_forward(model, batch)
            loss, comps = loss_fn(out, batch)
            sum_total += float(comps["L"].cpu().item())
            sum_pos += float(comps["pos"].cpu().item())
            sum_radius += float(comps["radius"].cpu().item())
            sum_conf += float(comps["conf"].cpu().item())
            n_batches += 1
            n_samples += B

            r = out["strike_radius"]
            c = out["strike_conf"]
            radius_min_seen = min(radius_min_seen, float(r.min()))
            radius_max_seen = max(radius_max_seen, float(r.max()))
            conf_min_seen = min(conf_min_seen, float(c.min()))
            conf_max_seen = max(conf_max_seen, float(c.max()))

    avg_total = sum_total / max(1, n_batches)
    avg_pos = sum_pos / max(1, n_batches)
    avg_radius = sum_radius / max(1, n_batches)
    avg_conf = sum_conf / max(1, n_batches)
    print(
        f"[Epoch {epoch_idx:03d}] Val  L={avg_total:.4f}  "
        f"pos={avg_pos:.4f}  rad={avg_radius:.4f}  conf={avg_conf:.4f}  "
        f"radius∈[{radius_min_seen:.3f}, {radius_max_seen:.3f}]  "
        f"conf∈[{conf_min_seen:.3f}, {conf_max_seen:.3f}]  "
        f"n={n_samples}"
    )
    return {
        "avg_total":  avg_total,
        "avg_pos":    avg_pos,
        "avg_radius": avg_radius,
        "avg_conf":   avg_conf,
    }


# =================== 总控 ===================

def train(config_path: str) -> None:
    # 延迟 import（避免 --smoke 也要求 .npz 数据）
    from data.dataset import build_datasets_from_config  # noqa: F401

    gnn2_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = gnn2_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {}) or {}

    set_seed(int(train_cfg.get("seed", 42)))
    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    # ---- Datasets ----
    rel_cfg = (
        cfg_path.relative_to(gnn2_root) if cfg_path.is_absolute() else Path(config_path)
    )
    train_ds, val_ds, test_ds = build_datasets_from_config(str(rel_cfg))
    print(f"[Info] datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    bs = int(train_cfg.get("batch_size", 256))
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
    print(
        f"[Info] loss weights: w_pos={loss_fn.cfg.w_pos}, "
        f"w_radius={loss_fn.cfg.w_radius}, w_conf={loss_fn.cfg.w_conf}"
    )

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    num_epochs = int(train_cfg.get("num_epochs", 20))
    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    keep_top_k = int(train_cfg.get("keep_top_k", 3))

    run_id = time.strftime("%Y%m%d%H%M%S")
    ckpt_root = (gnn2_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
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
            log_interval=log_interval, grad_clip=grad_clip,
        )
        print(
            f"[Epoch {epoch:03d}] Train  L={tr['avg_total']:.4f}  "
            f"pos={tr['avg_pos']:.4f}  rad={tr['avg_radius']:.4f}  "
            f"conf={tr['avg_conf']:.4f}  |grad|={tr['grad_norm_mean']:.3f}  "
            f"time={time.time() - t_ep:.1f}s"
        )

        ev = evaluate(model, loss_fn, val_loader, device, epoch)
        val_loss = float(ev["avg_total"])

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
    print("[Smoke/GNN2] start")
    print("=" * 60)
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/GNN2] params = {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Smoke/GNN2] model class = {type(model).__name__}")

    m = cfg["model"]
    data_cfg = cfg.get("data", {}) or {}
    B = 4
    T = int(m["fut_len"])
    D = int(m["feat_dim"])

    torch.manual_seed(0)
    pred = torch.randn(B, T, D, device=device)
    eta = torch.tensor([60, 600, 1800, 3600], dtype=torch.long, device=device)

    out = model(pred, eta)
    assert out["strike_pos"].shape == (B, 3), out["strike_pos"].shape
    assert out["strike_radius"].shape == (B, 1), out["strike_radius"].shape
    assert out["strike_conf"].shape == (B, 1), out["strike_conf"].shape

    # 数值范围硬校验
    r_min = float(data_cfg.get("radius_min_km", 0.5))
    r_max = float(data_cfg.get("radius_max_km", 10.0))
    assert torch.all(out["strike_radius"] >= r_min - 1e-4), \
        f"radius < r_min({r_min}); got {out['strike_radius'].min().item()}"
    assert torch.all(out["strike_radius"] <= r_max + 1e-4), \
        f"radius > r_max({r_max}); got {out['strike_radius'].max().item()}"
    assert torch.all(out["strike_conf"] >= 0.0) and torch.all(out["strike_conf"] <= 1.0), \
        "conf 必须在 [0, 1]"
    print(
        f"[Smoke/GNN2] forward OK: pos {tuple(out['strike_pos'].shape)}, "
        f"radius∈[{out['strike_radius'].min().item():.3f}, "
        f"{out['strike_radius'].max().item():.3f}] km, "
        f"conf∈[{out['strike_conf'].min().item():.3f}, "
        f"{out['strike_conf'].max().item():.3f}]"
    )

    # 真 loss + backward（造一份合理的假 GT）
    fake_gt = {
        "gt_strike_pos":    torch.randn(B, 3, device=device),
        "gt_strike_radius": torch.full((B,), (r_min + r_max) / 2, device=device),
        "gt_strike_conf":   torch.rand(B, device=device),
    }
    loss_fn = build_loss_from_config(cfg)
    loss, comps = loss_fn(out, fake_gt)
    print(
        f"[Smoke/GNN2] loss = {float(loss):.4f}  "
        f"(pos={float(comps['pos']):.4f}, rad={float(comps['radius']):.4f}, "
        f"conf={float(comps['conf']):.4f})"
    )

    loss.backward()
    # 关键模块的梯度（用 'encoder.layers.0' 兼容 SDPA / Manual 两种实现，
    # 因为 nn.TransformerEncoder 和 _ManualTransformerEncoder 都把子层放在 encoder.layers）
    seen = set()
    for name in [
        "input_proj.weight",
        "encoder.layers.0.self_attn.in_proj_weight",
        "eta_film.0.weight",
        "head.0.weight",
        "head.3.weight",
    ]:
        p = dict(model.named_parameters()).get(name)
        if p is not None and p.grad is not None:
            print(f"[Smoke/GNN2] grad_norm({name}) = {p.grad.norm().item():.4f}")
            seen.add(name)
    if not seen:
        print("[Smoke/GNN2][WARN] 没找到任何关键参数的 grad；模型结构可能改了。")

    print("=" * 60)
    print("[Smoke/GNN2] OK")
    print("=" * 60)


# =================== CLI ===================

def main() -> None:
    parser = argparse.ArgumentParser(description="GNN2 trainer.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        gnn2_root = Path(__file__).resolve().parents[2]
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = gnn2_root / cfg_path
        cfg = load_config(str(cfg_path))
        run_smoke(cfg)
        return

    train(args.config)


if __name__ == "__main__":
    main()
