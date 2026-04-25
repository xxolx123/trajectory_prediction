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
    export PYTHONPATH="$PWD/code"             # Linux/macOS
    $env:PYTHONPATH = "$PWD/code"             # Windows PowerShell

    # ---------- 模式 A：指标评估（默认）----------
    # 只跑全量指标，不画图。
    python -m train.eval --config config.yaml --split test

    # 显式指定 ckpt：
    python -m train.eval --config config.yaml --split test \
        --ckpt checkpoints/20260425222518/best_lstm_epoch048_valloss0.0136.pt

    # ---------- 模式 B：可视化（--vis）----------
    # 只画图（不算指标），默认 10 张，保存到
    #   eval_vis/<ckpt_run>__<ckpt_name>/
    # 例如：eval_vis/20260425222518__best_lstm_epoch048_valloss0.0136/
    python -m train.eval --config config.yaml --split test --vis

    # 一次画 30 张图：
    python -m train.eval --config config.yaml --split test --vis --vis-num 30

    # 想要可复现就给 --vis-seed：
    python -m train.eval --config config.yaml --split test --vis --vis-num 30 --vis-seed 42
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
    save_dir: Path,
    title: str = "",
    save_filename: Optional[str] = None,
) -> None:
    """
    画一条样本的多模态轨迹：
      - 历史 Tin 步
      - 真实未来 Tout 步
      - M 条预测未来 Tout 步（全部画出来），其中 best_mode 高亮。

    LSTM1 不输出 mode 概率，所以图上不显示 p=xx，概率交给下游 GNN1。
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
                label=f"Pred mode {m} (BEST)",
            )
        else:
            plt.plot(
                px, py,
                marker="x",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=f"Pred mode {m}",
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

    if save_filename:
        fname = save_dir / save_filename
    else:
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
) -> Tuple:
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

    # 每个样本各指标的 per-sample 数组，最后拼起来统计分位数
    all_min_ade: List[np.ndarray] = []
    all_min_fde: List[np.ndarray] = []

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

            # ---- 归一化 6D MSE（保留，和训练 winner 口径一致，但只作参考）----
            diff = pred_trajs_norm - targets_norm.unsqueeze(1)
            mse_per_mode = (diff ** 2).mean(dim=(-1, -2))          # [B, M]

            # ============================================================
            # best_mode 的挑选标准：xy-ADE 最小的那个 mode（而非 6D-MSE）。
            # 后续 ADE / FDE / RelADE / RelFDE 全部用这个 mode 的预测计算，
            # 和 old_plan 流程一致，仅 mode 选择准则不同。
            # ============================================================
            M = pred_trajs_norm.size(1)
            _, Tin, _D = inputs_norm.shape
            _, Tout, _ = targets_norm.shape

            # 一次性 decode 所有 mode 到 xy 位置
            preds_flat_norm = pred_trajs_norm.reshape(B * M, Tout, _D)
            inputs_rep = (
                inputs_norm.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Tin, _D)
            )
            targets_rep = (
                targets_norm.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, Tout, _D)
            )
            hist_pos_flat, gt_pos_flat, pred_pos_flat = decode_batch_to_positions(
                inputs_rep, targets_rep, preds_flat_norm,
                scaler=scaler,
                use_delta=use_delta,
                pos_dims=(0, 1),
            )
            pred_pos_per_mode = pred_pos_flat.reshape(B, M, Tout, 2)
            hist_pos = hist_pos_flat.reshape(B, M, Tin, 2)[:, 0]   # [B, Tin, 2]
            gt_pos = gt_pos_flat.reshape(B, M, Tout, 2)[:, 0]      # [B, Tout, 2]

            # per-mode ADE 用来挑 best_mode
            dist_pm = np.linalg.norm(
                pred_pos_per_mode - gt_pos[:, None, :, :], axis=-1
            )                                              # [B, M, Tout]
            ade_pm = dist_pm.mean(axis=-1)                 # [B, M]
            best_mode = np.argmin(ade_pm, axis=1)          # [B]

            # 被挑中那一 mode 的预测 + 6D MSE（按 xy-ADE 的 best 对应的 mode）
            batch_idx = np.arange(B)
            pred_pos_best = pred_pos_per_mode[batch_idx, best_mode]   # [B, Tout, 2]

            # Net MSE 用训练 winner（6D argmin），作训练口径参考
            best_6d = torch.argmin(mse_per_mode, dim=1)
            total_net_mse += float(
                mse_per_mode[torch.arange(B, device=device), best_6d].mean().cpu().item()
            )

            # === 按 old_plan 的单 mode 流程算 ADE / FDE / RelADE / RelFDE ===
            pos_mse, ade, fde = compute_pos_metrics(pred_pos_best, gt_pos)
            total_pos_mse += pos_mse
            total_ade += ade
            total_fde += fde

            rel_err, rel_fde = compute_relative_errors(
                hist_pos, gt_pos, pred_pos_best
            )
            total_rel_err += rel_err
            total_rel_fde += rel_fde

            # 收集样本级 minADE/minFDE（这里是"被选中 mode 的 ADE/FDE"）供分位数统计
            all_min_ade.append(ade_pm[batch_idx, best_mode].copy())
            fde_pm = dist_pm[..., -1]                      # [B, M]
            all_min_fde.append(fde_pm[batch_idx, best_mode].copy())

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

    # 分位数统计（基于样本级 minADE / minFDE）
    if all_min_ade:
        min_ade_all = np.concatenate(all_min_ade)
        min_fde_all = np.concatenate(all_min_fde)
        quantiles = np.percentile(min_ade_all, [10, 25, 50, 75, 90, 95])
        quantiles_fde = np.percentile(min_fde_all, [10, 25, 50, 75, 90, 95])
    else:
        quantiles = np.zeros(6)
        quantiles_fde = np.zeros(6)

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
        quantiles,        # [10, 25, 50, 75, 90, 95] 百分位的 minADE (km)
        quantiles_fde,    # 同上 minFDE
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
    idx: Optional[int] = None,
    save_filename: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    从 eval_ds_py 中抽一个样本，预测并可视化：
      - 使用 best-of-M 选一个“最佳模式”；
      - 同时把所有 M 条预测轨迹都画出来（best mode 高亮）。

    参数：
      idx           : 指定样本下标；None 时随机抽
      save_filename : 指定输出图文件名（不带目录）；None 时按时间戳生成
      rng           : 随机源；None 时新建（每次 eval 变化）
    """
    data_cfg = cfg.get("data", {})
    in_len = int(data_cfg.get("in_len", 20))
    out_len = int(data_cfg.get("out_len", 10))
    use_delta = bool(data_cfg.get("use_delta", False))

    n_samples = len(eval_ds_py)
    if n_samples == 0:
        print("[Eval] No samples to plot.")
        return

    if rng is None:
        rng = np.random.default_rng()  # 每次 eval 都变化
    if idx is None:
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

    # LSTM1 不再产出 mode_logits；概率交给下游 GNN1，可视化不显示概率。
    # best_mode 会在反归一化到 xy 后基于"xy 距离"选取（和 minADE 口径一致）。

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

    # 按 xy 距离选 best mode：
    #   每个 mode 对 GT 的 ADE（平均 L2 距离）
    per_mode_ade = np.linalg.norm(
        pred_pos_all - gt_pos_1[None, :, :], axis=-1
    ).mean(axis=-1)                       # [M]
    m_best = int(np.argmin(per_mode_ade))

    title = (
        f"Random sample idx={idx}, split={split}, "
        f"best_mode={m_best} (ADE={per_mode_ade[m_best]:.4f} km)"
    )

    # 画多模态版本（所有 mode）
    plot_example_trajectory_multi(
        hist_pos_1,
        gt_pos_1,
        pred_pos_all,
        best_mode=m_best,
        save_dir=save_dir,
        title=title,
        save_filename=save_filename,
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

def evaluate(
    config_path: str,
    ckpt_path: Optional[str] = None,
    split: str = "test",
    vis: bool = False,
    vis_num: int = 10,
    vis_seed: Optional[int] = None,
) -> None:
    """
    两种互斥模式：
      - vis=False（默认）：只跑全量指标评估（ADE/FDE/RelADE/RelFDE/...），不画图。
      - vis=True：只画图，跳过指标评估，画 vis_num 张多模态轨迹图，
                  保存到 eval_vis/<ckpt_run>__<ckpt_name>/ 目录下。
    """
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

    # ====== 模式分流：可视化 / 指标评估 ======
    if vis:
        # ----- 可视化模式：画 vis_num 张图，跳过指标 -----
        ckpt_path_obj = Path(ckpt_path)
        ckpt_stem = ckpt_path_obj.stem               # e.g. best_lstm_epoch048_valloss0.0136
        ckpt_run = ckpt_path_obj.parent.name         # e.g. 20260425222518
        sub_dirname = f"{ckpt_run}__{ckpt_stem}"
        vis_subdir = (project_root / "eval_vis" / sub_dirname).resolve()
        vis_subdir.mkdir(parents=True, exist_ok=True)

        n_samples = len(eval_ds_py)
        if n_samples == 0:
            print("[Vis] split 为空，无法可视化。")
            return

        if vis_seed is not None:
            rng = np.random.default_rng(int(vis_seed))
        else:
            rng = np.random.default_rng()

        # 不重复抽样：vis_num <= n_samples 时用 choice(replace=False)
        if vis_num <= n_samples:
            sample_indices = rng.choice(n_samples, size=vis_num, replace=False).tolist()
        else:
            print(f"[Vis] vis_num={vis_num} 超过 split 大小 {n_samples}，"
                  f"将采用允许重复抽样。")
            sample_indices = rng.integers(0, n_samples, size=vis_num).tolist()

        print(f"\n========== Visualization ==========")
        print(f"Split           = {split}")
        print(f"Save dir        = {vis_subdir}")
        print(f"Num samples     = {vis_num}")
        print(f"Sample indices  = {sample_indices[:20]}{'...' if len(sample_indices) > 20 else ''}")
        print("===================================\n")

        t_start = time.time()
        for k, idx in enumerate(sample_indices):
            fname = f"sample_{k:03d}_idx{int(idx):07d}.png"
            plot_random_trajectory(
                model, eval_ds_py, scaler, cfg, split, device,
                save_dir=vis_subdir,
                idx=int(idx),
                save_filename=fname,
                rng=rng,
            )
        elapsed = time.time() - t_start
        print(f"\n[Info] Visualization done. {vis_num} figures in: {vis_subdir}")
        print(f"[Info] Elapsed: {elapsed:.1f}s")
        return

    # ----- 指标评估模式：跑全量指标，不画图 -----
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
        quantiles_ade,
        quantiles_fde,
    ) = evaluate_loader(
        model, loss_fn, eval_loader, device, scaler, use_delta,
    )
    elapsed = time.time() - t_start

    print("\n========== Evaluation ==========")
    print(f"Split                          = {split}")
    # 每个样本按 xy-ADE 最小挑出一个 best_mode，再用它的预测算四个指标。
    # 因此 ADE = minADE；FDE / RelADE / RelFDE 是"该 best_mode 对应的值"。
    print(f"ADE (km, min-ADE mode)         = {avg_ade:.6f}")
    print(f"FDE (km, min-ADE mode)         = {avg_fde:.6f}")
    print(f"ADE (m, min-ADE mode)          = {avg_ade * 1000:.3f}")
    print(f"FDE (m, min-ADE mode)          = {avg_fde * 1000:.3f}")
    print(f"Rel ADE (%, min-ADE mode)      = {avg_rel_err:.2f}")
    print(f"Rel FDE (%, min-ADE mode)      = {avg_rel_fde:.2f}")
    print(f"Net MSE (6D-best, norm)        = {avg_net_mse:.6f}  # 训练口径参考")

    # 分位数：看 per-sample ADE / FDE 的分布，对判断"是不是被少数难样本拉高"很有用
    print("\n--- Distribution of per-sample ADE (km, min-ADE mode) ---")
    print(f"  p10   p25   p50   p75   p90   p95")
    print("  " + "  ".join(f"{v:.3f}" for v in quantiles_ade))
    print("--- Distribution of per-sample FDE (km, min-ADE mode) ---")
    print(f"  p10   p25   p50   p75   p90   p95")
    print("  " + "  ".join(f"{v:.3f}" for v in quantiles_fde))

    print(f"\nElapsed time (whole eval)      = {elapsed:.1f}s")

    # ====== 关注的：单次预测耗时（只算 model forward） ======
    print("\n------ Forward timing (model only) ------")
    print(f"Avg forward time / batch       = {avg_forward_batch * 1000:.3f} ms")
    print(f"Avg forward time / sample      = {avg_forward_sample * 1000:.3f} ms")
    print(f"Min / Max forward / batch      = "
          f"{min_forward_batch * 1000:.3f} ms / {max_forward_batch * 1000:.3f} ms")
    print("================================\n")
    print("[Info] 指标模式：未画图。要可视化请加 --vis （可配 --vis-num N）。")


# ====================== CLI 入口 ======================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-modal MTP LSTM trajectory predictor (PyTorch).\n\n"
            "两种互斥模式：\n"
            "  默认 (不加 --vis)：只跑全量指标评估（ADE/FDE/RelADE/RelFDE/...）\n"
            "  --vis             ：只画 vis_num 张可视化图，跳过指标\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--vis",
        action="store_true",
        help="开启可视化模式（只画图、不算指标）。图保存到 "
             "eval_vis/<ckpt_run>__<ckpt_name>/ 子目录。",
    )
    parser.add_argument(
        "--vis-num",
        type=int,
        default=10,
        help="可视化模式下画几张图（每张随机抽一个不同样本）。默认 10。",
    )
    parser.add_argument(
        "--vis-seed",
        type=int,
        default=None,
        help="可视化抽样的随机种子（不指定则每次结果不同）。",
    )
    args = parser.parse_args()
    evaluate(
        args.config,
        args.ckpt,
        args.split,
        vis=args.vis,
        vis_num=args.vis_num,
        vis_seed=args.vis_seed,
    )


if __name__ == "__main__":
    main()
