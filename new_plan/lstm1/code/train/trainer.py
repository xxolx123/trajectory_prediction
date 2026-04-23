#!/usr/bin/env python3
"""
trainer.py (PyTorch version)
----------------------------
多模态 MTP LSTM 轨迹预测训练脚本（硬分配版）。

整体流程：
  - 读取 config.yaml
  - 利用 data.traj_dataset.build_datasets_from_config 构建 train/val/test
  - 利用 train.model.build_model_from_config 构建多模态 LSTMForecaster
  - 利用 train.loss_mtp.TrajLoss 计算多模态 MTP 损失（winner-takes-all）
  - 使用 PyTorch DataLoader + Adam 完成训练与验证
  - 保存最优 checkpoint (state_dict)

用法（在 new_plan/lstm1/ 目录下）：
    # Linux/macOS:
    export PYTHONPATH="$PWD/code"
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD/code"

    # 冒烟测试（不依赖 CSV）
    python -m train.trainer --smoke

    # 正式训练
    python -m train.trainer --config config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

# ====== 确保可以 import data.*, train.* ======
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                   # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

# data.traj_dataset 依赖 pandas；--smoke 不需要，所以延迟到 train() 内 import
from train.model import build_model_from_config                 # noqa: E402
from train.loss import TrajLoss, TrajLossConfig                 # noqa: E402


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


# ====================== 训练与验证 ======================

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    log_interval: int = 100,
) -> float:
    """
    单个 epoch 的训练循环。

    返回：
      avg_loss: 当前 epoch 的平均训练 loss（Python float）
    """
    model.train()

    total_loss = 0.0
    total_batches = 0
    t0 = time.time()

    for step_idx, batch in enumerate(train_loader, start=1):
        # DataLoader 默认会把 numpy 自动转成 Tensor，
        # 但这里我们依然 to(device).float() 保守一些。
        inputs, targets = batch  # inputs:[B,in_len,D]  targets:[B,out_len,D]
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()

        # LSTM1：只输出 pred_trajs，概率由下游 GNN1 算
        pred_trajs = model(inputs)                     # [B,M,T,D]
        loss = loss_fn(pred_trajs, targets)            # 标量 Tensor

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        total_loss += loss_val
        total_batches += 1

        if log_interval > 0 and (step_idx % log_interval == 0):
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch_idx:03d}] step {step_idx:05d}, "
                f"loss = {loss_val:.6f}, elapsed = {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
) -> float:
    """
    在验证集上评估平均 loss。
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            pred_trajs = model(inputs)
            loss = loss_fn(pred_trajs, targets)

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    print(f"[Epoch {epoch_idx:03d}] Validation loss = {avg_loss:.6f}")
    return avg_loss


def save_best_checkpoint(
    model: nn.Module,
    ckpt_dir: Path,
    epoch_idx: int,
    val_loss: float,
) -> None:
    """
    保存当前最优模型 checkpoint（只保存 model.state_dict）。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_lstm_epoch{epoch_idx:03d}_valloss{val_loss:.4f}.pt"
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
    # 延迟 import：避免 --smoke 依赖 pandas
    from data.traj_dataset import build_datasets_from_config  # noqa: F401

    # 项目根目录：.../code/train/trainer.py → .../code → 项目根
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = setup_device(train_cfg)

    # ==== 构建 Python 层数据集 (train/val/test) ====
    #   注意：这里只用到 train / val，test 可单独写脚本评估
    print("[Info] Building trajectory datasets from config ...")
    # 这里传入的是相对项目根目录的路径（保持和原 MindSpore 版一致）
    rel_cfg_path = cfg_path.relative_to(project_root)
    train_ds_py, val_ds_py, test_ds_py, scaler = build_datasets_from_config(
        config_path=str(rel_cfg_path)
    )
    print(
        f"[Info] Dataset sizes: "
        f"train={len(train_ds_py)}, val={len(val_ds_py)}, test={len(test_ds_py)}"
    )

    # ==== 封装成 PyTorch DataLoader ====
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))  # 你可以按需调整

    train_loader = DataLoader(
        train_ds_py,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds_py,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ==== 构建模型 & 损失 ====
    print("[Info] Building model (LSTM1 multi-modal forecaster, no mode_logits) ...")
    model = build_model_from_config(cfg).to(device)

    # 多模态 WTA TrajLoss（只剩回归项）
    loss_cfg = TrajLossConfig(return_components=False)
    loss_fn = TrajLoss(loss_cfg)

    # ==== 优化器 ====
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ==== 训练循环 ====
    num_epochs = int(train_cfg.get("num_epochs", 50))
    log_interval = int(train_cfg.get("log_interval", 100))

    # ---- 新增：以训练开始时间设置一个 run 目录 ----
    # 格式：YYYYmmddHHMMSS，例如 20251119171508
    run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
    ckpt_root_str = train_cfg.get("ckpt_dir", "checkpoints")  # 作为根目录
    ckpt_root = (project_root / ckpt_root_str).resolve()
    ckpt_dir = ckpt_root / run_id

    print(f"[Info] Checkpoints will be saved under: {ckpt_dir}")

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        t_epoch_start = time.time()
        train_loss = train_one_epoch(
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
            f"[Epoch {epoch:03d}] Train loss = {train_loss:.6f}, "
            f"epoch time = {t_epoch_end - t_epoch_start:.1f}s"
        )

        # 验证
        val_loss = evaluate(
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            device=device,
            epoch_idx=epoch,
        )

        # 保存最优模型（都会丢进 ckpt_dir = checkpoints/<run_id> 里）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_best_checkpoint(model, ckpt_dir, epoch, val_loss)
            print(
                f"[Best] Updated best model: epoch={best_epoch}, "
                f"val_loss={best_val_loss:.6f}"
            )

    print(
        f"\n[Training Finished] Best epoch = {best_epoch}, "
        f"best val_loss = {best_val_loss:.6f}"
    )


# ====================== 冒烟测试 --smoke ======================

def run_smoke(cfg: Dict[str, Any]) -> None:
    """
    不依赖 CSV，直接用随机数据走一遍 LSTM1：
      - 构造 x = randn(B, in_len, 6)
      - forward -> 断言 [B, M, out_len, 6]
      - 反向 -> 检查 LSTM1 参数有梯度
    """
    print("=" * 60)
    print("[Smoke/LSTM1] start")
    print("=" * 60)

    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/LSTM1] params = {sum(p.numel() for p in model.parameters()):,}")

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    in_len = int(data_cfg.get("in_len", 20))
    out_len = int(model_cfg.get("out_len", data_cfg.get("out_len", 10)))
    in_size = int(model_cfg.get("input_size", 6))
    out_size = int(model_cfg.get("output_size", 6))
    modes = int(model_cfg.get("modes", 3))

    B = 2
    torch.manual_seed(0)
    x = torch.randn(B, in_len, in_size, device=device)
    targets = torch.zeros(B, out_len, out_size, device=device)

    pred = model(x)
    expected = (B, modes, out_len, out_size)
    assert pred.shape == expected, f"shape 不对: {tuple(pred.shape)} vs {expected}"
    print(f"[Smoke/LSTM1] forward OK, output shape = {tuple(pred.shape)}")

    loss_fn = TrajLoss(TrajLossConfig(return_components=False))
    loss = loss_fn(pred, targets)
    print(f"[Smoke/LSTM1] loss = {float(loss):.4f}")

    loss.backward()
    lstm_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    print(f"[Smoke/LSTM1] gradients present? {lstm_has_grad}")

    print("=" * 60)
    print("[Smoke/LSTM1] OK")
    print("=" * 60)


# ====================== CLI 入口 ======================

def main():
    parser = argparse.ArgumentParser(
        description="Train multi-modal LSTM1 trajectory predictor (no mode_logits)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (相对 new_plan/lstm1/ 或绝对路径).",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试：不依赖 CSV")
    args = parser.parse_args()

    if args.smoke:
        project_root = Path(__file__).resolve().parents[2]  # .../lstm1
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        cfg = load_config(str(cfg_path))
        run_smoke(cfg)
        return

    train(args.config)


if __name__ == "__main__":
    main()
