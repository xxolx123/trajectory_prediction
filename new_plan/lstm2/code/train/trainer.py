#!/usr/bin/env python3
"""
lstm2/code/train/trainer.py
---------------------------
LSTM2 训练脚本（意图 + 威胁度联合预测）。

整体流程：
  - 读 lstm2/config.yaml
  - data.dataset.build_datasets_from_config 构建 train/val/test
      * 每个样本：fut_norm[10,11]、position[3]、intent[int]、threat[float]
  - train.model.build_model_from_config 构建模型（默认 Transformer Encoder）
  - train.loss.build_loss_from_config 构建 IntentThreatLoss (CE + MSE)
  - 训练循环 + 验证 + top-K ckpt（按 val_loss）

用法（在 new_plan/lstm2/ 下）：
    # Linux/macOS:
    export PYTHONPATH="$PWD/code"
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD/code"

    # 冒烟（不依赖 npz）
    python -m train.trainer --smoke

    # 正式训练
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

from train.model import build_model_from_config         # noqa: E402
from train.loss import build_loss_from_config           # noqa: E402


# ============================================================
# 工具
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据集统计
# ============================================================

INTENT_NAMES = ["ATTACK(0)", "EVASION(1)", "DEFENSE(2)", "RETREAT(3)"]


def analyze_intent_distribution(dataset, name: str) -> None:
    """打印数据集中 4 类意图分布；intent < 0 / > 3 视为 ignored。"""
    if len(dataset) == 0:
        print(f"\n[Stats] {name} dataset is empty; skip.")
        return

    # 优先走 dataset.intent_labels 这个 numpy 数组（O(1)），否则逐样本读
    if hasattr(dataset, "intent_labels"):
        arr = np.asarray(dataset.intent_labels, dtype=np.int64)
    else:
        arr = np.asarray(
            [int(dataset[i]["intent"]) for i in range(len(dataset))],
            dtype=np.int64,
        )

    counts = [int((arr == k).sum()) for k in range(4)]
    ignored = int(((arr < 0) | (arr > 3)).sum())
    total = sum(counts)

    print(f"\n[Stats] Intent distribution for {name} dataset:")
    print(f"        total windows (valid) = {total}, ignored = {ignored}")
    for k in range(4):
        c = counts[k]
        ratio = (c / total * 100.0) if total > 0 else 0.0
        print(f"        {INTENT_NAMES[k]:>11}: {c:7d} ({ratio:6.2f}%)")
    print("")


# ============================================================
# Collate
# ============================================================

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    取 dataset[i] 返回的 dict 列表，把训练用得到的字段堆成 tensor。
    其它调试字段（hist_phys / fut_phys / fut_gt / sample_idx ...）不动；
    eval/可视化时单独走另一条路。
    """
    out: Dict[str, torch.Tensor] = {}
    fut_norm = np.stack([b["fut_norm"] for b in batch], axis=0)
    position = np.stack([b["position"] for b in batch], axis=0)
    intent = np.asarray([b["intent"] for b in batch], dtype=np.int64)
    threat = np.asarray([b["threat"] for b in batch], dtype=np.float32)

    out["fut_norm"] = torch.from_numpy(fut_norm).float()
    out["position"] = torch.from_numpy(position).float()
    out["intent"] = torch.from_numpy(intent).long()
    out["threat"] = torch.from_numpy(threat).float()
    return out


# ============================================================
# 单 batch 指标
# ============================================================

def _batch_metrics(
    outputs: Dict[str, torch.Tensor],
    intent: torch.Tensor,
    threat: torch.Tensor,
    ignore_label: int = -1,
) -> Tuple[int, int, float]:
    """
    Returns: intent_correct, intent_total, threat_mae(0..100 标度，按 batch 平均)
    """
    logits = outputs["logits_intent"]
    raw = outputs["threat_raw"]
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw.squeeze(1)

    mask = intent != ignore_label
    if not mask.any():
        return 0, 0, 0.0

    pred = torch.argmax(logits[mask], dim=1)
    correct = int((pred == intent[mask]).sum().item())
    total = int(mask.sum().item())

    pred_threat = torch.sigmoid(raw[mask]) * 100.0
    mae = float(torch.abs(pred_threat - threat[mask]).mean().item())
    return correct, total, mae


# ============================================================
# 训练 / 验证
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    log_interval: int = 100,
    grad_clip: float = 1.0,
    ignore_label: int = -1,
) -> Dict[str, float]:
    model.train()
    t0 = time.time()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0
    total_batches = 0
    grad_norm_sum = 0.0

    for step, batch in enumerate(loader, start=1):
        fut = batch["fut_norm"].to(device)
        pos = batch["position"].to(device)
        intent = batch["intent"].to(device)
        threat = batch["threat"].to(device)

        optimizer.zero_grad()
        out = model(fut, pos)
        total, cls, reg = loss_fn(out, intent, threat)
        total.backward()

        if grad_clip and grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        else:
            with torch.no_grad():
                gn = torch.sqrt(sum(
                    (p.grad.detach() ** 2).sum()
                    for p in model.parameters()
                    if p.grad is not None
                ))
        grad_norm_sum += float(gn)

        optimizer.step()

        total_loss += float(total.detach().cpu().item())
        total_cls += float(cls.detach().cpu().item())
        total_reg += float(reg.detach().cpu().item())
        total_batches += 1

        with torch.no_grad():
            c, n, mae = _batch_metrics(out, intent, threat, ignore_label)
        total_correct += c
        total_samples += n
        total_mae_sum += mae

        if log_interval > 0 and (step % log_interval == 0):
            elapsed = time.time() - t0
            cur_acc = 100.0 * total_correct / max(1, total_samples)
            cur_mae = total_mae_sum / max(1, total_batches)
            print(
                f"[Epoch {epoch_idx:03d}] step {step:05d}  "
                f"loss = {float(total):.4f}  cls = {float(cls):.4f}  reg = {float(reg):.4f}  "
                f"acc = {cur_acc:.2f}%  mae = {cur_mae:.3f}  "
                f"|grad| = {float(gn):.3f}  elapsed = {elapsed:.1f}s"
            )

    nb = max(1, total_batches)
    return {
        "avg_loss":    total_loss / nb,
        "avg_cls":     total_cls / nb,
        "avg_reg":     total_reg / nb,
        "intent_acc":  100.0 * total_correct / max(1, total_samples),
        "threat_mae":  total_mae_sum / nb,
        "grad_norm":   grad_norm_sum / nb,
    }


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    ignore_label: int = -1,
    num_classes: int = 4,
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0
    total_batches = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            fut = batch["fut_norm"].to(device)
            pos = batch["position"].to(device)
            intent = batch["intent"].to(device)
            threat = batch["threat"].to(device)

            out = model(fut, pos)
            total, _cls, _reg = loss_fn(out, intent, threat)
            total_loss += float(total.detach().cpu().item())
            total_batches += 1

            c, n, mae = _batch_metrics(out, intent, threat, ignore_label)
            total_correct += c
            total_samples += n
            total_mae_sum += mae

            mask = intent != ignore_label
            if mask.any():
                pred = torch.argmax(out["logits_intent"][mask], dim=1).cpu().numpy()
                gt = intent[mask].cpu().numpy()
                for g, p in zip(gt, pred):
                    if 0 <= g < num_classes and 0 <= p < num_classes:
                        confusion[int(g), int(p)] += 1

    nb = max(1, total_batches)
    avg_loss = total_loss / nb
    acc = 100.0 * total_correct / max(1, total_samples)
    mae = total_mae_sum / nb

    print(
        f"[Epoch {epoch_idx:03d}] Val loss = {avg_loss:.4f}  "
        f"acc = {acc:.2f}%  mae = {mae:.3f}"
    )
    return {
        "avg_loss":   avg_loss,
        "intent_acc": acc,
        "threat_mae": mae,
        "confusion":  confusion,
    }


# ============================================================
# Top-K ckpt 管理
# ============================================================

def maybe_save_topk_checkpoint(
    model: nn.Module,
    ckpt_dir: Path,
    epoch_idx: int,
    val_loss: float,
    val_acc: float,
    val_mae: float,
    saved: List[Tuple[float, int, Path]],
    keep_top_k: int,
) -> bool:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qualifies = len(saved) < keep_top_k or val_loss < saved[-1][0]
    if not qualifies:
        return False

    ckpt_path = ckpt_dir / (
        f"best_intent_epoch{epoch_idx:03d}"
        f"_valloss{val_loss:.4f}"
        f"_acc{val_acc:.2f}"
        f"_mae{val_mae:.3f}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)
    saved.append((val_loss, epoch_idx, ckpt_path))
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
    print(f"[ckpt] saved {ckpt_path.name} | Top-{keep_top_k}: [{top_desc}]")
    return True


# ============================================================
# train()
# ============================================================

def train(config_path: str) -> None:
    # 延迟 import：避免 --smoke 也要求 npz
    from data.dataset import build_datasets_from_config       # noqa: F401

    project_root = Path(__file__).resolve().parents[2]   # .../lstm2
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}

    set_seed(int(train_cfg.get("seed", 42)))
    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    # ---- Datasets ----
    print("[Info] Building lstm2 datasets ...")
    rel_cfg = cfg_path.relative_to(project_root) if cfg_path.is_absolute() else Path(config_path)
    train_ds, val_ds, test_ds, scaler = build_datasets_from_config(str(rel_cfg))
    print(
        f"[Info] dataset sizes: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )

    analyze_intent_distribution(train_ds, "train")
    analyze_intent_distribution(val_ds, "val")
    analyze_intent_distribution(test_ds, "test")

    # ---- DataLoader ----
    bs = int(train_cfg.get("batch_size", 256))
    nw = int(train_cfg.get("num_workers", 0))
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=pin, collate_fn=_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=pin, collate_fn=_collate, drop_last=False,
    )

    # ---- Model / Loss / Optim ----
    model = build_model_from_config(cfg).to(device)
    loss_fn = build_loss_from_config(cfg)
    print(f"[Info] params = {sum(p.numel() for p in model.parameters()):,}  "
          f"(type = {cfg.get('model', {}).get('type', 'transformer')})")

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ---- Loop ----
    num_epochs = int(train_cfg.get("num_epochs", 30))
    log_interval = int(train_cfg.get("log_interval", 100))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    keep_top_k = int(train_cfg.get("keep_top_k", 3))
    ignore_label = int(loss_cfg.get("ignore_label", -1))

    run_id = time.strftime("%Y%m%d%H%M%S")
    ckpt_root = (project_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
    ckpt_dir = ckpt_root / run_id
    print(f"[Info] ckpt dir = {ckpt_dir} (keep top-{keep_top_k})")

    saved: List[Tuple[float, int, Path]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    best_val_mae = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")
        t_ep = time.time()
        tr = train_one_epoch(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            loader=train_loader, device=device, epoch_idx=epoch,
            log_interval=log_interval, grad_clip=grad_clip,
            ignore_label=ignore_label,
        )
        print(
            f"[Epoch {epoch:03d}] Train loss = {tr['avg_loss']:.4f}  "
            f"(cls={tr['avg_cls']:.4f}, reg={tr['avg_reg']:.4f})  "
            f"acc = {tr['intent_acc']:.2f}%  mae = {tr['threat_mae']:.3f}  "
            f"|grad| = {tr['grad_norm']:.3f}  time = {time.time() - t_ep:.1f}s"
        )

        if len(val_ds) > 0:
            ev = evaluate(
                model=model, loss_fn=loss_fn,
                loader=val_loader, device=device,
                epoch_idx=epoch, ignore_label=ignore_label,
            )
            val_loss = ev["avg_loss"]
            val_acc = ev["intent_acc"]
            val_mae = ev["threat_mae"]
            confusion = ev["confusion"]
            print("[Val Confusion]\n" + _confusion_to_str(confusion))

            maybe_save_topk_checkpoint(
                model, ckpt_dir, epoch, val_loss, val_acc, val_mae,
                saved, keep_top_k,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_acc = val_acc
                best_val_mae = val_mae
        else:
            # 没 val：用 train_loss 做存档
            print("[WARN] val_ds is empty; saving on train_loss instead.")
            maybe_save_topk_checkpoint(
                model, ckpt_dir, epoch, tr["avg_loss"], tr["intent_acc"],
                tr["threat_mae"], saved, keep_top_k,
            )

    print(
        f"\n[Training Finished] Best epoch = {best_epoch}, "
        f"val_loss = {best_val_loss:.6f}, "
        f"val_acc = {best_val_acc:.2f}%, "
        f"val_mae = {best_val_mae:.3f}"
    )
    if saved:
        print(f"[Training Finished] Top-{keep_top_k} checkpoints:")
        for rank, (l, e, p) in enumerate(saved, start=1):
            print(f"  #{rank}  epoch={e:03d}  val_loss={l:.6f}  {p.name}")


def _confusion_to_str(confusion: np.ndarray) -> str:
    """4×4 混淆矩阵的简单文本渲染。"""
    headers = ["ATK", "EVA", "DEF", "RET"]
    rows = []
    rows.append("       " + "  ".join(f"{h:>6}" for h in headers))
    for i, h in enumerate(headers):
        rows.append(f"  {h:<3}: " + "  ".join(f"{int(confusion[i, j]):6d}"
                                              for j in range(confusion.shape[1])))
    return "\n".join(rows)


# ============================================================
# Smoke
# ============================================================

def run_smoke(cfg: Dict[str, Any]) -> None:
    print("=" * 60)
    print("[Smoke/LSTM2] start")
    print("=" * 60)
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/LSTM2] params = {sum(p.numel() for p in model.parameters()):,}  "
          f"(type = {cfg.get('model', {}).get('type', 'transformer')})")

    data_cfg = cfg.get("data", {}) or {}
    Tout = int(data_cfg.get("fut_len", 10))
    n_cls = int(cfg.get("model", {}).get("num_intent_classes", 4))

    B = 2
    torch.manual_seed(0)
    fut = torch.randn(B, Tout, 6, device=device)        # 直接喂 6 维：模型内部会工程化
    pos = torch.randn(B, 3, device=device)

    out = model(fut, pos)
    assert out["logits_intent"].shape == (B, n_cls), out["logits_intent"].shape
    assert out["threat_raw"].shape == (B, 1), out["threat_raw"].shape
    print(
        f"[Smoke/LSTM2] forward OK: "
        f"logits_intent {tuple(out['logits_intent'].shape)}, "
        f"threat_raw {tuple(out['threat_raw'].shape)}"
    )

    loss_fn = build_loss_from_config(cfg)
    intent_lbl = torch.zeros(B, dtype=torch.long, device=device)
    threat = torch.full((B,), 50.0, dtype=torch.float32, device=device)
    total, cls, reg = loss_fn(out, intent_lbl, threat)
    print(f"[Smoke/LSTM2] loss total={float(total):.4f}, "
          f"cls={float(cls):.4f}, reg={float(reg):.4f}")
    total.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    print(f"[Smoke/LSTM2] gradients present? {has_grad}")
    print("=" * 60)
    print("[Smoke/LSTM2] OK")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM2 trainer (intent + threat).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试：不依赖 npz")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]   # .../lstm2
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_config(str(cfg_path))

    if args.smoke:
        run_smoke(cfg)
        return
    train(args.config)


if __name__ == "__main__":
    main()
