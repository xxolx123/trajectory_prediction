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
from train.loss import TrajLoss, TrajLossConfig, build_loss_from_config  # noqa: E402


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
    grad_clip: float = 0.0,
    num_modes: int = 3,
) -> Dict[str, float]:
    """
    单个 epoch 的训练循环。

    返回字典：
      avg_loss       : 本 epoch 平均训练 loss
      winner_frac    : 长度 M 的列表，每个 mode 在本 epoch 成为 argmin 的比例
                       （若严重偏离 1/M，提示 mode collapse）
      grad_norm_mean : 裁剪前的平均梯度 L2 范数
    """
    model.train()

    total_loss = 0.0
    total_reg = 0.0
    total_div = 0.0
    total_div_raw = 0.0
    total_batches = 0
    total_samples = 0
    winner_counts = torch.zeros(num_modes, dtype=torch.long, device=device)
    grad_norm_sum = 0.0
    t0 = time.time()

    for step_idx, batch in enumerate(train_loader, start=1):
        inputs, targets = batch  # inputs:[B,in_len,D]  targets:[B,out_len,D]
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()

        pred_trajs = model(inputs)                     # [B,M,T,D]
        # loss_fn 统一走 return_components=True，方便监控
        loss, comps = loss_fn(pred_trajs, targets)
        loss.backward()
        total_reg += float(comps["L_reg"].detach().cpu().item())
        total_div += float(comps["L_div"].detach().cpu().item())
        total_div_raw += float(comps["div_raw"].detach().cpu().item())

        # ---- 梯度裁剪（LSTM 防爆炸） ----
        if grad_clip and grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
        else:
            # 即使不裁剪也记一下梯度范数，便于观察
            with torch.no_grad():
                total_norm = torch.sqrt(
                    sum(
                        (p.grad.detach() ** 2).sum()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
        grad_norm_sum += float(total_norm)

        optimizer.step()

        # ---- 统计 ----
        loss_val = float(loss.detach().cpu().item())
        total_loss += loss_val
        total_batches += 1
        B = int(inputs.size(0))
        total_samples += B
        winner_counts += comps["winner_counts"].to(winner_counts.device)

        if log_interval > 0 and (step_idx % log_interval == 0):
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch_idx:03d}] step {step_idx:05d}, "
                f"loss = {loss_val:.6f}, |grad| = {float(total_norm):.3f}, "
                f"elapsed = {elapsed:.1f}s"
            )

    winner_frac = (
        (winner_counts.float() / max(1, total_samples)).detach().cpu().tolist()
    )
    nb = max(1, total_batches)
    avg_loss = total_loss / nb
    grad_norm_mean = grad_norm_sum / nb
    return {
        "avg_loss": avg_loss,
        "avg_reg": total_reg / nb,
        "avg_div": total_div / nb,
        "avg_div_raw": total_div_raw / nb,
        "winner_frac": winner_frac,
        "grad_norm_mean": grad_norm_mean,
    }


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch_idx: int,
    num_modes: int = 3,
) -> Dict[str, float]:
    """在验证集上评估平均 loss + winner 分布。"""
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    winner_counts = torch.zeros(num_modes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            pred_trajs = model(inputs)
            loss, comps = loss_fn(pred_trajs, targets)

            total_loss += float(loss.detach().cpu().item())
            total_batches += 1
            total_samples += int(inputs.size(0))
            winner_counts += comps["winner_counts"].to(winner_counts.device)

    avg_loss = total_loss / max(1, total_batches)
    winner_frac = (
        (winner_counts.float() / max(1, total_samples)).detach().cpu().tolist()
    )
    print(
        f"[Epoch {epoch_idx:03d}] Val loss = {avg_loss:.6f}, "
        f"winner_frac = [{', '.join(f'{w:.3f}' for w in winner_frac)}]"
    )
    return {"avg_loss": avg_loss, "winner_frac": winner_frac}


def maybe_save_topk_checkpoint(
    model: nn.Module,
    ckpt_dir: Path,
    epoch_idx: int,
    val_loss: float,
    saved_ckpts: "list[tuple[float, int, Path]]",
    keep_top_k: int,
) -> bool:
    """
    动态维护 top-K 个最优 checkpoint：
      - 当前 val_loss 进入 top-K（列表未满 或 优于列表末尾最差者）才保存；
      - 保存后按 val_loss 升序排序，pop 掉末尾超出 K 的，并把对应文件删掉。

    `saved_ckpts` 会被**原地修改**，保存了 (val_loss, epoch, path)。

    返回：True 表示本 epoch 新保存了一个 ckpt（不管有没有淘汰），False 表示未保存。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 先判断值不值得保存
    qualifies = len(saved_ckpts) < keep_top_k or val_loss < saved_ckpts[-1][0]
    if not qualifies:
        return False

    # 保存
    ckpt_path = ckpt_dir / f"best_lstm_epoch{epoch_idx:03d}_valloss{val_loss:.4f}.pt"
    torch.save(model.state_dict(), ckpt_path)
    saved_ckpts.append((val_loss, epoch_idx, ckpt_path))
    saved_ckpts.sort(key=lambda x: x[0])  # val_loss 升序

    # 淘汰超出 K 的
    while len(saved_ckpts) > keep_top_k:
        worst_loss, worst_epoch, worst_path = saved_ckpts.pop()
        try:
            if worst_path.exists():
                worst_path.unlink()
            print(
                f"[Checkpoint] Dropped worse ckpt: {worst_path.name} "
                f"(val_loss={worst_loss:.6f})"
            )
        except OSError as e:
            print(f"[Checkpoint][WARN] Failed to remove {worst_path}: {e}")

    # 打印当前 top-K
    top_desc = ", ".join(
        f"ep{e}:{l:.4f}" for l, e, _ in saved_ckpts
    )
    print(
        f"[Checkpoint] Saved {ckpt_path.name} | "
        f"Top-{keep_top_k}: [{top_desc}]"
    )
    return True


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

    # 多模态 TrajLoss（默认 soft-WTA，可在 config.loss 里切换）
    loss_fn = build_loss_from_config(cfg)
    print(
        f"[Info] Loss: mode_selection={loss_fn.cfg.mode_selection}"
        f"{'（T=' + str(loss_fn.cfg.soft_temperature) + '）' if loss_fn.cfg.mode_selection == 'soft' else ''}"
    )

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
    grad_clip = float(train_cfg.get("grad_clip", 1.0))  # LSTM 默认裁 1.0

    num_modes = int(cfg.get("model", {}).get("modes", 3))

    # ---- 新增：以训练开始时间设置一个 run 目录 ----
    # 格式：YYYYmmddHHMMSS，例如 20251119171508
    run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
    ckpt_root_str = train_cfg.get("ckpt_dir", "checkpoints")  # 作为根目录
    ckpt_root = (project_root / ckpt_root_str).resolve()
    ckpt_dir = ckpt_root / run_id

    keep_top_k = int(train_cfg.get("keep_top_k", 3))
    print(
        f"[Info] Checkpoints will be saved under: {ckpt_dir} "
        f"(keep top-{keep_top_k})"
    )

    # 动态维护的 top-K 列表：每项 = (val_loss, epoch, path)
    saved_ckpts: list = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        t_epoch_start = time.time()
        train_stats = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch_idx=epoch,
            log_interval=log_interval,
            grad_clip=grad_clip,
            num_modes=num_modes,
        )
        t_epoch_end = time.time()

        train_loss = train_stats["avg_loss"]
        winner_frac = train_stats["winner_frac"]
        grad_norm_mean = train_stats["grad_norm_mean"]
        avg_reg = train_stats["avg_reg"]
        avg_div = train_stats["avg_div"]
        avg_div_raw = train_stats["avg_div_raw"]

        print(
            f"[Epoch {epoch:03d}] Train loss = {train_loss:.6f} "
            f"(reg={avg_reg:.6f}, div={avg_div:.6f}, div_raw={avg_div_raw:.4f}), "
            f"winner_frac = [{', '.join(f'{w:.3f}' for w in winner_frac)}], "
            f"mean |grad| = {grad_norm_mean:.3f}, "
            f"epoch time = {t_epoch_end - t_epoch_start:.1f}s"
        )

        # 验证
        val_stats = evaluate(
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            device=device,
            epoch_idx=epoch,
            num_modes=num_modes,
        )
        val_loss = val_stats["avg_loss"]

        # 动态维护 top-K 个 ckpt：新 ckpt 进入 top-K 才保存，超出 K 的自动淘汰
        maybe_save_topk_checkpoint(
            model=model,
            ckpt_dir=ckpt_dir,
            epoch_idx=epoch,
            val_loss=val_loss,
            saved_ckpts=saved_ckpts,
            keep_top_k=keep_top_k,
        )
        # 更新"历史最好"（仅用于训练结束时的总结打印）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    print(
        f"\n[Training Finished] Best epoch = {best_epoch}, "
        f"best val_loss = {best_val_loss:.6f}"
    )
    if saved_ckpts:
        print(f"[Training Finished] Top-{keep_top_k} checkpoints:")
        for rank, (loss, ep, path) in enumerate(saved_ckpts, start=1):
            print(f"  #{rank}  epoch={ep:03d}  val_loss={loss:.6f}  {path.name}")


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

    loss_fn = build_loss_from_config(cfg)
    print(f"[Smoke/LSTM1] loss config: {loss_fn.cfg}")

    loss, comps = loss_fn(pred, targets)
    print(f"[Smoke/LSTM1] loss = {float(loss):.4f}")
    print(f"[Smoke/LSTM1] winner_counts = {comps['winner_counts'].tolist()}")

    loss.backward()
    lstm_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    print(f"[Smoke/LSTM1] gradients present? {lstm_has_grad}")

    # 观察每个 mode 的 fc_traj 切片梯度：
    #   hard WTA：只有 winner 的切片有较大梯度，其余接近 0
    #   soft WTA：每个 mode 都有相近大小的梯度
    fc_grad = model.fc_traj.weight.grad  # [M*T*D, fc_in]
    if fc_grad is not None:
        M, T_out, D = model.modes, model.out_len, model.output_size
        per_mode_norm = fc_grad.view(M, T_out * D, -1).norm(dim=(1, 2)).tolist()
        print(f"[Smoke/LSTM1] per-mode grad norm in fc_traj = "
              f"{[f'{g:.4f}' for g in per_mode_norm]}")

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
