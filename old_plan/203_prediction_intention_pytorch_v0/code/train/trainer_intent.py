#!/usr/bin/env python3
"""
trainer_intent.py (PyTorch version)
-----------------------------------
意图 + 威胁度预测 训练脚本（基于简单 MLP）。

整体流程：
  - 读取 config.yaml
  - 利用 data.traj_dataset.build_datasets_from_config 构建 train/val/test
      * 每个样本：inputs=[L,6], intent_label(0~3), threat_score(0~100)
  - 利用 train.intent_model.build_model_from_config 构建 IntentThreatNet
  - 利用 train.intent_loss.build_loss_from_config 构建 IntentThreatLoss
  - 使用 PyTorch DataLoader + Adam 完成训练与验证
  - 保存最优 checkpoint (state_dict) 到 checkpoints_intent/<run_id>/...

额外：
  - 训练前统计 train / val / test 三个数据集中各类意图窗口数量（0~3），
    并忽略 intent_label < 0 的样本（例如 -1）。

用法（在项目根目录）：
    export PYTHONPATH="$PWD/code"
    python -m train.trainer_intent --config config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

# ====== 确保可以 import data.*, train.* ======
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)  # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from data.traj_dataset import build_datasets_from_config  # noqa: E402
from train.intent_model import build_model_from_config  # noqa: E402
from train.intent_loss import build_loss_from_config  # noqa: E402


# ====================== 工具函数 ======================

def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    """
    根据 train 段配置选择设备：
      train.device: "cuda" / "cpu" / "mps" / "auto"
      如果没有给，就优先 cuda → mps → cpu
    """
    dev_str = train_cfg.get("device", "auto").lower()

    if dev_str == "cpu":
        device = torch.device("cpu")
    elif dev_str in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev_str == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:  # "auto"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[Info] Using device: {device}")
    return device


def set_seed(seed: int) -> None:
    """统一设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================== 统计意图分布 ======================

def analyze_intent_distribution(dataset, name: str) -> None:
    """
    统计给定 dataset 中，各类意图窗口数量。
    假设 dataset[i] 返回 (inputs, intent_label, threat_score)。

    约定：
      - intent_label ∈ {0,1,2,3}：有效类别
      - intent_label < 0          ：忽略（例如 -1）
      - 其它异常值会被计入 ignored。
    """
    counts = [0, 0, 0, 0]  # [ATTACK, EVASION, DEFENSE, RETREAT]
    ignored = 0

    for i in range(len(dataset)):
        _, intent_label, _ = dataset[i]
        # intent_label 可能是 numpy scalar / torch tensor / Python int，都转成 int
        val = int(intent_label)

        if val < 0 or val > 3:
            ignored += 1
            continue

        counts[val] += 1

    total_valid = sum(counts)
    print(f"\n[Stats] Intent distribution for {name} dataset:")
    print(f"        total windows (valid) = {total_valid}, ignored = {ignored}")
    label_names = ["ATTACK(0)", "EVASION(1)", "DEFENSE(2)", "RETREAT(3)"]
    for k in range(4):
        c = counts[k]
        ratio = (c / total_valid * 100.0) if total_valid > 0 else 0.0
        print(f"        {label_names[k]:>11}: {c:7d} ({ratio:6.2f}%)")
    print("")  # 空行分隔一下


# ====================== 训练与验证 ======================

def _compute_batch_metrics(
        outputs: Dict[str, torch.Tensor],
        intent_labels: torch.Tensor,
        threat_scores: torch.Tensor,
) -> Tuple[int, int, float]:
    """
    根据模型输出和标签，计算一个 batch 的：
      - intent_correct: 预测意图正确的样本数
      - intent_total:   样本总数
      - threat_mae:     威胁度 MAE（0~100 标度）

    注意：这里假设 intent_labels 全部合法（>=0），
    若有 ignore_label（-1），请在外层提前过滤。
    """
    logits_intent = outputs["logits_intent"]  # [B, 4]
    threat_raw = outputs["threat_raw"]  # [B, 1] or [B]

    if threat_raw.ndim == 2 and threat_raw.shape[1] == 1:
        threat_raw = threat_raw.squeeze(1)  # [B]

    # 意图预测：argmax
    pred_intent = torch.argmax(logits_intent, dim=1)  # [B]
    intent_correct = (pred_intent == intent_labels).sum().item()
    intent_total = intent_labels.numel()

    # 威胁度预测：raw -> sigmoid -> [0,1] -> *100
    threat_pred_norm = torch.sigmoid(threat_raw)  # [B]
    threat_pred = threat_pred_norm * 100.0  # [B]
    # MAE
    threat_mae = torch.abs(threat_pred - threat_scores).mean().item()

    return intent_correct, intent_total, threat_mae


def train_one_epoch(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        epoch_idx: int,
        log_interval: int = 100,
) -> Tuple[float, float, float]:
    """
    单个 epoch 的训练循环。

    返回：
      avg_total_loss, avg_intent_acc, avg_threat_mae
    """
    model.train()

    total_loss_sum = 0.0
    total_batches = 0

    total_correct_intent = 0
    total_samples = 0
    threat_mae_sum = 0.0

    t0 = time.time()

    for step_idx, batch in enumerate(train_loader, start=1):
        # DataLoader 默认会把 numpy 自动转成 Tensor，
        # 这里 to(device).float() 保守一些。
        inputs, intent_labels, threat_scores = batch
        inputs = inputs.to(device).float()  # [B,L,6]
        intent_labels = intent_labels.to(device).long()  # [B]
        threat_scores = threat_scores.to(device).float()  # [B]

        optimizer.zero_grad()

        outputs = model(inputs)  # dict: logits_intent, threat_raw
        total_loss, cls_loss, reg_loss = loss_fn(
            outputs, intent_labels, threat_scores
        )

        total_loss.backward()
        optimizer.step()

        # 标量化
        loss_val = float(total_loss.detach().cpu().item())
        total_loss_sum += loss_val
        total_batches += 1

        # 计算训练集上的 metrics
        with torch.no_grad():
            correct_intent, num_samples, threat_mae = _compute_batch_metrics(
                outputs, intent_labels, threat_scores
            )
        total_correct_intent += correct_intent
        total_samples += num_samples
        threat_mae_sum += threat_mae

        if log_interval > 0 and (step_idx % log_interval == 0):
            elapsed = time.time() - t0
            cur_acc = 100.0 * total_correct_intent / max(1, total_samples)
            cur_mae = threat_mae_sum / max(1, total_batches)
            print(
                f"[Epoch {epoch_idx:03d}] step {step_idx:05d}, "
                f"loss = {loss_val:.6f}, "
                f"intent_acc = {cur_acc:.2f}%, "
                f"threat_mae = {cur_mae:.3f}, "
                f"elapsed = {elapsed:.1f}s"
            )

    avg_loss = total_loss_sum / max(1, total_batches)
    avg_acc = 100.0 * total_correct_intent / max(1, total_samples)
    avg_mae = threat_mae_sum / max(1, total_batches)
    return avg_loss, avg_acc, avg_mae


def evaluate(
        model: nn.Module,
        loss_fn: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        epoch_idx: int,
) -> Tuple[float, float, float]:
    """
    在验证集上评估：
      - 平均 total_loss
      - intent 准确率 (%)
      - threat MAE (0~100 标度)
    """
    model.eval()

    total_loss_sum = 0.0
    total_batches = 0

    total_correct_intent = 0
    total_samples = 0
    threat_mae_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, intent_labels, threat_scores = batch
            inputs = inputs.to(device).float()
            intent_labels = intent_labels.to(device).long()
            threat_scores = threat_scores.to(device).float()

            outputs = model(inputs)
            total_loss, cls_loss, reg_loss = loss_fn(
                outputs, intent_labels, threat_scores
            )

            loss_val = float(total_loss.detach().cpu().item())
            total_loss_sum += loss_val
            total_batches += 1

            correct_intent, num_samples, threat_mae = _compute_batch_metrics(
                outputs, intent_labels, threat_scores
            )
            total_correct_intent += correct_intent
            total_samples += num_samples
            threat_mae_sum += threat_mae

    avg_loss = total_loss_sum / max(1, total_batches)
    avg_acc = 100.0 * total_correct_intent / max(1, total_samples)
    avg_mae = threat_mae_sum / max(1, total_batches)

    print(
        f"[Epoch {epoch_idx:03d}] Validation: "
        f"loss = {avg_loss:.6f}, "
        f"intent_acc = {avg_acc:.2f}%, "
        f"threat_mae = {avg_mae:.3f}"
    )
    return avg_loss, avg_acc, avg_mae


def save_best_checkpoint(
        model: nn.Module,
        ckpt_dir: Path,
        epoch_idx: int,
        val_loss: float,
        val_acc: float,
        val_mae: float,
) -> None:
    """
    保存当前最优模型 checkpoint（只保存 model.state_dict）。
    文件名中包含 epoch、val_loss、intent_acc、threat_mae。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / (
        f"best_intent_epoch{epoch_idx:03d}"
        f"_valloss{val_loss:.4f}"
        f"_acc{val_acc:.2f}"
        f"_mae{val_mae:.3f}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved best model to: {ckpt_path}")


# ====================== 总控：train() ======================

def train(config_path: str) -> None:
    """
    总训练入口：
      - 读配置
      - 构建数据集、模型、loss、optimizer
      - 循环 epoch 训练 + 验证 + 存 checkpoint
    """
    # 项目根目录：.../code/train/trainer_intent.py → .../code → 项目根
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train_intent", cfg.get("train", {}))
    # 为了不和轨迹预测冲突，你可以在 config 里单独加 train_intent 段；
    # 如果没有，就复用 train 段。

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = setup_device(train_cfg)

    # ==== 构建 Python 层数据集 (train/val/test) ====
    print("[Info] Building intent/threat datasets from config ...")
    # 这里传入的是相对项目根目录的路径（保持和你原来写法一致）
    rel_cfg_path = cfg_path.relative_to(project_root)
    train_ds_py, val_ds_py, test_ds_py, scaler = build_datasets_from_config(
        config_path=str(rel_cfg_path)
    )
    print(
        f"[Info] Dataset sizes: "
        f"train={len(train_ds_py)}, val={len(val_ds_py)}, test={len(test_ds_py)}"
    )

    # ==== 训练前，统计各数据集中 4 类意图窗口的数量 ====
    analyze_intent_distribution(train_ds_py, "train")
    analyze_intent_distribution(val_ds_py, "val")
    analyze_intent_distribution(test_ds_py, "test")

    # ==== 封装成 PyTorch DataLoader ====
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))  # 可以按需调大

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds_py,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds_py,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ==== 构建模型 & 损失 ====
    print("[Info] Building IntentThreatNet (PyTorch MLP) ...")
    model = build_model_from_config(cfg).to(device)

    print("[Info] Building IntentThreatLoss ...")
    loss_fn = build_loss_from_config(cfg)

    # ==== 优化器 ====
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ==== 训练循环 ====
    num_epochs = int(train_cfg.get("num_epochs", 30))
    log_interval = int(train_cfg.get("log_interval", 100))

    # ---- 以训练开始时间设置一个 run 目录 ----
    # 格式：YYYYmmddHHMMSS，例如 20251128143000
    run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
    ckpt_root_str = train_cfg.get("ckpt_dir", "checkpoints_intent")  # 单独一个根目录
    ckpt_root = (project_root / ckpt_root_str).resolve()
    ckpt_dir = ckpt_root / run_id

    print(f"[Info] Checkpoints will be saved under: {ckpt_dir}")

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    best_val_mae = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        t_epoch_start = time.time()
        train_loss, train_acc, train_mae = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch_idx=epoch,
            log_interval=log_interval,
        )
        t_epoch_end = time.time()

        print(
            f"[Epoch {epoch:03d}] Train: "
            f"loss = {train_loss:.6f}, "
            f"intent_acc = {train_acc:.2f}%, "
            f"threat_mae = {train_mae:.3f}, "
            f"time = {t_epoch_end - t_epoch_start:.1f}s"
        )

        # 验证
        val_loss, val_acc, val_mae = evaluate(
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            device=device,
            epoch_idx=epoch,
        )

        # 以 val_loss 为主，顺带参考 acc / mae
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_mae = val_mae
            save_best_checkpoint(model, ckpt_dir, epoch, val_loss, val_acc, val_mae)
            print(
                f"[Best] Updated best model: "
                f"epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f}, "
                f"val_intent_acc={best_val_acc:.2f}%, "
                f"val_threat_mae={best_val_mae:.3f}"
            )

    print(
        f"\n[Training Finished] Best epoch = {best_epoch}, "
        f"best val_loss = {best_val_loss:.6f}, "
        f"best val_intent_acc = {best_val_acc:.2f}%, "
        f"best val_threat_mae = {best_val_mae:.3f}"
    )


# ====================== CLI 入口 ======================

def main():
    parser = argparse.ArgumentParser(
        description="Train intent & threat predictor (PyTorch MLP)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative to project root or absolute).",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
