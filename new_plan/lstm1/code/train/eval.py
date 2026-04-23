#!/usr/bin/env python3
"""
eval.py (PyTorch version, MTP + denorm + delta decoding + visualization)
----------------------------------------------------------------------
多模态 LSTM + MTP 轨迹预测评估脚本（硬分配版）。

功能：
  - 读取 config.yaml
  - 使用 data.traj_dataset.build_datasets_from_config 构建 train/val/test 数据集
  - 构建多模态模型并加载 checkpoint
  - 在指定 split 上计算：
      * Net MSE（best-of-M）：归一化 6 维特征上的 MSE，
        对每个样本选误差最小的 mode（与训练中的 winner 一致）
      * Pos MSE：反归一化 + 增量还原后的 (x,y) MSE（单位：km^2），同样使用 best mode
      * ADE / FDE：位置误差（km 和 m，best-of-M）
      * 相对误差（%）：delta_x / x * 100%，其中
          - x 为 GT 两相邻时刻之间的位移长度（step 长度）
          - delta_x 为该时刻预测位置与真实位置之间的误差
        统计：所有 future 步的平均相对误差，以及最后一步的相对误差
  - 随机从该 split 中抽一条样本（完整 20+10 步），画：
      * 20 步历史
      * 10 步 GT 未来
      * 10 步 Pred 未来（best mode）

用法（在项目根目录）：
    export PYTHONPATH="$PWD/code"

    # 显式指定 ckpt

    # 多模态更加明显
    python -m train.eval --config config.yaml --ckpt checkpoints/20251201163510/best_lstm_epoch001_valloss0.0326.pt --split test

    # 训练 epoch 增加后，多模态不太明显
    python -m train.eval --config config.yaml --ckpt checkpoints/20251201163510/best_lstm_epoch003_valloss0.0314.pt --split test

    # 或不指定 ckpt，自动从 train.ckpt_dir 里找最新 .pt
    python -m train.eval --config config.yaml --split test
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ====== 保证可以 import data.*, train.* ======
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                   # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from data.traj_dataset import build_datasets_from_config  # noqa: E402
from train.model import build_model_from_config           # noqa: E402
from train.loss import TrajLoss, TrajLossConfig           # noqa: E402


# ====================== 通用工具 ======================

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    dev_str = train_cfg.get("device", "auto").lower()

    if dev_str == "cpu":
        device = torch.device("cpu")
    elif dev_str in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev_str == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[Info] Using device: {device}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ====================== 反归一化 + Δ→绝对坐标 ======================

def inverse_transform_features(x: np.ndarray, scaler) -> np.ndarray:
    """
    使用数据集构建时的 scaler 做反归一化。
    x: [..., D]
    """
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return x
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    flat_orig = scaler.inverse_transform(flat)
    return flat_orig.reshape(orig_shape)


def decode_batch_to_positions(
    inputs_norm: torch.Tensor,   # [B,Tin,D]
    targets_norm: torch.Tensor,  # [B,Tout,D]
    preds_norm: torch.Tensor,    # [B,Tout,D]  ← 已经选好某个 mode 的输出
    scaler,
    use_delta: bool,
    pos_dims=(0, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把一个 batch 从“网络空间（归一化 + 可能是增量）”还原到：
      - hist_pos: [B,Tin,2]  历史绝对位置
      - gt_pos  : [B,Tout,2] 真实未来绝对位置
      - pred_pos: [B,Tout,2] 预测未来绝对位置

    若 use_delta=False：
      - pos_dims 的特征本身就是绝对坐标

    若 use_delta=True：
      - pos_dims 的特征是增量 dx,dy
      - 对 [history, future] 做整体 cumsum，并从 history 末位置接 future
    """
    B, Tin, D = inputs_norm.shape
    _, Tout, _ = targets_norm.shape

    # 拼成一个大串，一次性 inverse_transform（效率更好）
    all_norm = torch.cat(
        [inputs_norm.cpu(), targets_norm.cpu(), preds_norm.cpu()], dim=1
    )  # [B,Tin+Tout+Tout,D]
    all_np = all_norm.numpy()

    # 反归一化到原始特征
    all_orig = inverse_transform_features(all_np, scaler)

    all_T = Tin + Tout + Tout
    assert all_orig.shape[1] == all_T
    hist_orig = all_orig[:, :Tin, :]
    gt_orig   = all_orig[:, Tin:Tin + Tout, :]
    pred_orig = all_orig[:, Tin + Tout:, :]

    # 只取位置维度
    hist_pos_feat = hist_orig[..., list(pos_dims)]  # [B,Tin,2]
    gt_pos_feat   = gt_orig[...,  list(pos_dims)]   # [B,Tout,2]
    pred_pos_feat = pred_orig[..., list(pos_dims)]  # [B,Tout,2]

    if not use_delta:
        # 直接就是绝对位置
        hist_pos = hist_pos_feat
        gt_pos   = gt_pos_feat
        pred_pos = pred_pos_feat
    else:
        # Δ → 绝对位置
        hist_pos = np.cumsum(hist_pos_feat, axis=1)    # [B,Tin,2]

        last_hist_pos = hist_pos[:, -1:, :]            # [B,1,2]
        gt_delta_cum   = np.cumsum(gt_pos_feat, axis=1)
        pred_delta_cum = np.cumsum(pred_pos_feat, axis=1)

        gt_pos   = last_hist_pos + gt_delta_cum
        pred_pos = last_hist_pos + pred_delta_cum

    return hist_pos, gt_pos, pred_pos


# ====================== 位置级指标 ======================

def compute_pos_metrics(
    pred_pos: np.ndarray,   # [B,T,2]
    gt_pos: np.ndarray,     # [B,T,2]
) -> Tuple[float, float, float]:
    """
    在“位置空间”（km）上计算：
      - pos_mse：x,y MSE（km^2）
      - ADE：平均位移误差（km）
      - FDE：最终时刻位移误差（km）
    """
    diff = pred_pos - gt_pos          # [B,T,2]
    mse = float(np.mean(diff ** 2))   # km^2

    dist = np.linalg.norm(diff, axis=-1)  # [B,T]，km
    ade = float(dist.mean())
    fde = float(dist[:, -1].mean())
    return mse, ade, fde


def compute_relative_errors(
    hist_pos: np.ndarray,  # [B,Tin,2]
    gt_pos: np.ndarray,    # [B,Tout,2]
    pred_pos: np.ndarray,  # [B,Tout,2]
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    计算“相对误差”：
      对每个样本、每个 future 步 t（0..Tout-1）：
        - x_t   = GT 在该步的位移长度：
                  t = 0:  从 history 最后一个点 到 GT 第一个 future 点 的距离
                  t > 0: GT[t] 和 GT[t-1] 之间的距离
        - delta = 预测位置与 GT 位置之间的误差：
                  || pred_pos[t] - gt_pos[t] ||

        相对误差 = delta / x_t * 100 (%)

    返回：
      - mean_rel_err_percent: 所有样本、所有 future 步的平均相对误差（单位：%）
      - final_rel_err_percent: 所有样本在最后一步的相对误差（单位：%）
    """
    B, Tin, _ = hist_pos.shape
    _, Tout, _ = gt_pos.shape

    # 构造每个 future step 的“前一个 GT 点”
    prev_gt = np.zeros_like(gt_pos)          # [B,Tout,2]
    prev_gt[:, 0, :] = hist_pos[:, -1, :]    # 第 0 步：前一个点是 history 的最后一个点
    if Tout > 1:
        prev_gt[:, 1:, :] = gt_pos[:, :-1, :]  # 之后的步：前一个点是上一时刻 GT

    # GT 每一步的位移长度 x_t
    step_len = np.linalg.norm(gt_pos - prev_gt, axis=-1)   # [B,Tout]

    # 预测误差 delta_x
    delta = np.linalg.norm(pred_pos - gt_pos, axis=-1)     # [B,Tout]

    # 所有步整体的平均相对误差
    mask_all = step_len > eps
    if np.any(mask_all):
        rel_all = (delta[mask_all] / step_len[mask_all]) * 100.0
        mean_rel_err_percent = float(rel_all.mean())
    else:
        mean_rel_err_percent = 0.0

    # 最后一步的相对误差
    step_last = step_len[:, -1]   # [B]
    delta_last = delta[:, -1]     # [B]
    mask_last = step_last > eps
    if np.any(mask_last):
        rel_last = (delta_last[mask_last] / step_last[mask_last]) * 100.0
        final_rel_err_percent = float(rel_last.mean())
    else:
        final_rel_err_percent = 0.0

    return mean_rel_err_percent, final_rel_err_percent


# ====================== 绘图 ======================

def plot_example_trajectory_multi(
    hist_pos: np.ndarray,      # [Tin,2]
    gt_pos: np.ndarray,        # [Tout,2]
    pred_pos_all: np.ndarray,  # [M,Tout,2]
    best_mode: int,
    probs: np.ndarray,         # [M]
    save_dir: Path,
    title: str = "",
) -> None:
    """
    画一条样本的多模态轨迹：
      - 历史 Tin 步
      - 真实未来 Tout 步
      - M 条预测未来 Tout 步（全部画出来），其中 best_mode 高亮。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    hx, hy = hist_pos[:, 0], hist_pos[:, 1]
    gx, gy = gt_pos[:,   0], gt_pos[:,   1]

    plt.figure(figsize=(5, 5))

    # 历史
    plt.plot(hx, hy, marker="o", linestyle="-", label="History (20)")

    # GT 未来
    plt.plot(gx, gy, marker="o", linestyle="-", label="GT Future (10)")

    # 预测的多条未来
    M = pred_pos_all.shape[0]
    for m in range(M):
        px = pred_pos_all[m, :, 0]
        py = pred_pos_all[m, :, 1]

        if m == best_mode:
            # 高亮 best mode
            plt.plot(
                px, py,
                marker="x",
                linestyle="-",
                linewidth=2.0,
                label=f"Pred mode {m} (BEST, p={probs[m]:.2f})",
            )
        else:
            plt.plot(
                px, py,
                marker="x",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=f"Pred mode {m} (p={probs[m]:.2f})",
            )

    Tin = len(hx)
    Tout = len(gx)
    for i in range(Tin):
        plt.text(hx[i], hy[i], f"{i-Tin}", fontsize=7, alpha=0.6)
    for i in range(Tout):
        plt.text(gx[i], gy[i], f"{i}", fontsize=7, alpha=0.8)

    plt.legend()
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title(title or "History + GT Future + Multi-modal Pred Futures (XY)")
    plt.axis("equal")

    fname = save_dir / f"eval_traj_example_multi_{int(time.time())}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Info] Saved multi-modal trajectory visualization to: {fname}")



# ====================== 在 DataLoader 上评估 ======================

def evaluate_loader(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    use_delta: bool,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    在给定 DataLoader 上评估：
      - 平均 Net MSE（best-of-M，归一化空间）
      - 平均 Pos MSE / ADE / FDE（位置空间，best-of-M）
      - 平均相对误差（%）（位置空间，best-of-M）
      - 最后一步的相对误差（%）（位置空间，best-of-M）
      - 模型 forward 耗时统计（只算 model(inputs)）：
          * avg_forward_batch: 每个 batch 平均耗时（秒）
          * avg_forward_sample: 单个样本平均耗时（秒）
          * min_forward_batch, max_forward_batch: 每个 batch 耗时范围（秒）
    """
    model.eval()

    total_net_mse = 0.0          # best mode 的归一化 MSE
    total_pos_mse = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_rel_err = 0.0          # 所有步的平均相对误差（按 batch 平均）
    total_rel_fde = 0.0          # 最后一步相对误差（按 batch 平均）
    total_batches = 0

    # forward 时间统计
    total_forward_time = 0.0     # 所有 batch 的 forward 总时间
    total_forward_calls = 0      # batch 次数
    total_samples = 0            # 总样本数（用于换算单样本）
    min_forward_batch = float("inf")
    max_forward_batch = 0.0

    with torch.no_grad():
        for inputs_norm, targets_norm in loader:
            inputs_norm = inputs_norm.to(device).float()   # [B,Tin,D]
            targets_norm = targets_norm.to(device).float() # [B,Tout,D]
            B = inputs_norm.size(0)
            total_samples += B

            # ================== 只测 forward(model(inputs)) 的时间 ==================
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred_trajs_norm = model(inputs_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward = time.time() - t0

            total_forward_time += t_forward
            total_forward_calls += 1
            min_forward_batch = min(min_forward_batch, t_forward)
            max_forward_batch = max(max_forward_batch, t_forward)
            # ============================================================

            # 调用 WTA 损失，看数值是否正常
            _ = loss_fn(pred_trajs_norm, targets_norm)

            # ---- 计算 best-of-M 的归一化 MSE（和训练 winner 一致）----
            # diff: [B,M,T,D]
            diff = pred_trajs_norm - targets_norm.unsqueeze(1)
            mse_per_mode = (diff ** 2).mean(dim=(-1, -2))   # [B,M]
            best_mode_idx = torch.argmin(mse_per_mode, dim=1)  # [B]

            # 当前 batch 的 MSE（先对样本取平均，再跨 batch 平均）
            batch_best_mse = mse_per_mode[
                torch.arange(mse_per_mode.size(0), device=device),
                best_mode_idx
            ].mean()   # 标量 Tensor
            net_mse_val = float(batch_best_mse.detach().cpu().item())
            total_net_mse += net_mse_val

            # ---- 位置空间指标：用 best mode 的预测 ----
            batch_indices = torch.arange(B, device=device)
            preds_best_norm = pred_trajs_norm[batch_indices, best_mode_idx]  # [B,Tout,D]

            hist_pos, gt_pos, pred_pos = decode_batch_to_positions(
                inputs_norm, targets_norm, preds_best_norm,
                scaler=scaler,
                use_delta=use_delta,
                pos_dims=(0, 1),
            )

            pos_mse, ade, fde = compute_pos_metrics(pred_pos, gt_pos)
            total_pos_mse += pos_mse
            total_ade += ade
            total_fde += fde

            # ---- 相对误差：delta_x / x * 100% ----
            # x：GT 每步的位移长度（上一时刻 GT -> 当前 GT）
            # delta_x：当前步预测位置与 GT 位置的距离
            batch_rel_err, batch_rel_fde = compute_relative_errors(
                hist_pos, gt_pos, pred_pos
            )
            total_rel_err += batch_rel_err
            total_rel_fde += batch_rel_fde

            total_batches += 1

    avg_net_mse = total_net_mse / max(1, total_batches)
    avg_pos_mse = total_pos_mse / max(1, total_batches)
    avg_ade = total_ade / max(1, total_batches)
    avg_fde = total_fde / max(1, total_batches)
    avg_rel_err = total_rel_err / max(1, total_batches)
    avg_rel_fde = total_rel_fde / max(1, total_batches)

    # forward 时间的平均值
    avg_forward_batch = total_forward_time / max(1, total_forward_calls)  # s / batch
    avg_forward_sample = total_forward_time / max(1, total_samples)       # s / sample

    # 如果数据集为空，防止 inf 乱打印
    if total_forward_calls == 0:
        min_forward_batch = 0.0
        max_forward_batch = 0.0

    return (
        avg_net_mse,
        avg_pos_mse,
        avg_ade,
        avg_fde,
        avg_forward_batch,
        avg_forward_sample,
        min_forward_batch,
        max_forward_batch,
        avg_rel_err,
        avg_rel_fde,
    )



# ====================== 随机抽样绘图（仿 MindSpore 版本） ======================

def plot_random_trajectory(
    model: torch.nn.Module,
    eval_ds_py,
    scaler,
    cfg: Dict[str, Any],
    split: str,
    device: torch.device,
    save_dir: Path,
) -> None:
    """
    随机从 eval_ds_py 中抽一个样本，预测并可视化：
      - 使用 best-of-M 选一个“最佳模式”；
      - 同时把所有 M 条预测轨迹都画出来（best mode 高亮）。
    """
    data_cfg = cfg.get("data", {})
    in_len = int(data_cfg.get("in_len", 20))
    out_len = int(data_cfg.get("out_len", 10))
    use_delta = bool(data_cfg.get("use_delta", False))

    n_samples = len(eval_ds_py)
    if n_samples == 0:
        print("[Eval] No samples to plot.")
        return

    rng = np.random.default_rng()  # 每次 eval 都变化
    idx = int(rng.integers(low=0, high=n_samples))

    # eval_ds_py[idx] 是归一化空间的数据（可能是 Δ）
    inputs_norm_np, targets_norm_np = eval_ds_py[idx]   # [Tin,D], [Tout,D]

    inputs_norm = torch.from_numpy(inputs_norm_np).unsqueeze(0).to(device).float()
    targets_norm = torch.from_numpy(targets_norm_np).unsqueeze(0).to(device).float()

    model.eval()
    with torch.no_grad():
        # LSTM1 只输出 pred_trajs：[1, M, Tout, D]
        pred_trajs_norm = model(inputs_norm)

    B, M, Tout, D = pred_trajs_norm.shape

    # ---- 对该样本选 best mode（按 MSE 最小）----
    diff = pred_trajs_norm - targets_norm.unsqueeze(1)     # [1,M,T,D]
    mse_per_mode = (diff ** 2).mean(dim=(-1, -2))          # [1,M]
    best_mode_idx = torch.argmin(mse_per_mode, dim=1)      # [1]
    m_best = int(best_mode_idx.item())

    # LSTM1 不再产出 mode_logits；这里为了可视化用"均匀分布 + best mode 标亮"。
    # 实际多模态概率留给下游 GNN1 计算。
    probs = np.full((M,), 1.0 / max(M, 1), dtype=np.float32)

    # ---- 反归一化 + Δ→绝对位置 ----
    # 先算历史 + GT（只有一份，和 mode 无关）
    hist_pos, gt_pos, _dummy = decode_batch_to_positions(
        inputs_norm,
        targets_norm,
        preds_norm=targets_norm,  # 这里不关心 pred，只是占位
        scaler=scaler,
        use_delta=use_delta,
        pos_dims=(0, 1),
    )
    hist_pos_1 = hist_pos[0]   # [Tin,2]
    gt_pos_1   = gt_pos[0]     # [Tout,2]

    # 再对每个 mode 分别 decode 预测，收集成 [M,Tout,2]
    pred_pos_all = []
    for m in range(M):
        preds_m = pred_trajs_norm[0, m].unsqueeze(0)  # [1,Tout,D]
        _, _gt_m, pred_pos_m = decode_batch_to_positions(
            inputs_norm,
            targets_norm,
            preds_m,
            scaler=scaler,
            use_delta=use_delta,
            pos_dims=(0, 1),
        )
        pred_pos_all.append(pred_pos_m[0])  # [Tout,2]

    pred_pos_all = np.stack(pred_pos_all, axis=0)  # [M,Tout,2]

    title = (
        f"Random sample idx={idx}, split={split}, "
        f"best_mode={m_best}, prob={probs[m_best]:.3f}"
    )

    # 画多模态版本（所有 mode）
    plot_example_trajectory_multi(
        hist_pos_1,
        gt_pos_1,
        pred_pos_all,
        best_mode=m_best,
        probs=probs,
        save_dir=save_dir,
        title=title,
    )



def pick_split_dataset(split: str, train_ds_py, val_ds_py, test_ds_py):
    split = split.lower()
    if split == "train":
        return train_ds_py
    elif split in ("val", "valid", "validation"):
        return val_ds_py
    elif split == "test":
        return test_ds_py
    else:
        raise ValueError(f"Unknown split: {split}")


# ====================== 总评估入口 ======================

def evaluate(config_path: str, ckpt_path: Optional[str] = None, split: str = "test") -> None:
    project_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = setup_device(train_cfg)
    use_delta = bool(data_cfg.get("use_delta", False))
    print(f"[Info] use_delta = {use_delta}")

    # 先构建 Python 层数据集
    rel_cfg_path = cfg_path.relative_to(project_root)
    train_ds_py, val_ds_py, test_ds_py, scaler = build_datasets_from_config(
        config_path=str(rel_cfg_path)
    )
    print(
        f"[Info] Dataset sizes: "
        f"train={len(train_ds_py)}, val={len(val_ds_py)}, test={len(test_ds_py)}"
    )

    eval_ds_py = pick_split_dataset(split, train_ds_py, val_ds_py, test_ds_py)
    print(f"[Info] Using split='{split}', size={len(eval_ds_py)}")

    # DataLoader 用于统计指标（顺序固定即可）
    batch_size = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 64)))
    num_workers = int(train_cfg.get("num_workers", 0))

    eval_loader = DataLoader(
        eval_ds_py,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 构建模型 & 加载权重
    print("[Info] Building model ...")
    model = build_model_from_config(cfg).to(device)

    ckpt_dir_str = train_cfg.get("ckpt_dir", "checkpoints")
    ckpt_dir = (project_root / ckpt_dir_str).resolve()

    if ckpt_path is None or ckpt_path == "":
        latest = find_latest_ckpt(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(
                f"在 {ckpt_dir} 下没有找到任何 .pt checkpoint，"
                f"请检查 ckpt_dir 或使用 --ckpt 显式指定。"
            )
        ckpt_path = str(latest)
        print(f"[Info] No ckpt specified. Using latest: {ckpt_path}")
    else:
        ckpt_path = str(Path(ckpt_path).resolve())

    print(f"[Info] Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    loss_cfg = TrajLossConfig(return_components=False)
    loss_fn = TrajLoss(loss_cfg)

    print("[Info] Evaluating ...")
    t_start = time.time()
    (
        avg_net_mse,
        avg_pos_mse,
        avg_ade,
        avg_fde,
        avg_forward_batch,
        avg_forward_sample,
        min_forward_batch,
        max_forward_batch,
        avg_rel_err,
        avg_rel_fde,
    ) = evaluate_loader(
        model, loss_fn, eval_loader, device, scaler, use_delta
    )
    elapsed = time.time() - t_start

    print("\n========== Evaluation ==========")
    print(f"Split                          = {split}")
    # print(f"Net MSE (best-of-M, norm 6D)   = {avg_net_mse:.6f}")
    # print(f"Pos MSE (x,y, km^2)            = {avg_pos_mse:.6f}")
    print(f"ADE (km, best-of-M)            = {avg_ade:.6f}")
    # print(f"ADE (m, best-of-M)             = {avg_ade * 1000:.3f}")
    print(f"FDE (km, best-of-M)            = {avg_fde:.6f}")
    # print(f"FDE (m, best-of-M)             = {avg_fde * 1000:.3f}")
    print(f"Rel ADE (%, best-of-M)         = {avg_rel_err:.2f}")
    print(f"Rel FDE (%, best-of-M)         = {avg_rel_fde:.2f}")
    print(f"Elapsed time (whole eval)      = {elapsed:.1f}s")

    # ====== 关注的：单次预测耗时（只算 model forward） ======
    print("\n------ Forward timing (model only) ------")
    print(f"Avg forward time / batch       = {avg_forward_batch * 1000:.3f} ms")
    print(f"Avg forward time / sample      = {avg_forward_sample * 1000:.3f} ms")
    print(f"Min / Max forward / batch      = "
          f"{min_forward_batch * 1000:.3f} ms / {max_forward_batch * 1000:.3f} ms")
    print("================================\n")

    # ===== 单独随机抽样画一条轨迹 =====
    vis_dir = project_root / "eval_vis"
    plot_random_trajectory(model, eval_ds_py, scaler, cfg, split, device, vis_dir)


# ====================== CLI 入口 ======================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-modal MTP LSTM trajectory predictor (PyTorch)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative to project root or absolute).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to .pt checkpoint. If empty, will use the latest in train.ckpt_dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on.",
    )
    args = parser.parse_args()
    evaluate(args.config, args.ckpt, args.split)


if __name__ == "__main__":
    main()
