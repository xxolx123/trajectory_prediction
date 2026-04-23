#!/usr/bin/env python3
"""
eval_intent.py (PyTorch version, intent + threat)
-------------------------------------------------
意图 + 威胁度预测评估脚本（基于简单 MLP）。

功能：
  - 读取 config.yaml
  - 使用 data.traj_dataset.build_datasets_from_config 构建 train/val/test 数据集
      * 每个样本：
          inputs: [window_len, 6]       最近 window_len 步轨迹片段
          intent_label: 0..3            (0=ATTACK,1=EVASION,2=DEFENSE,3=RETREAT)
          threat_score: 0..100          威胁度
  - 构建 IntentThreatNet 并加载 checkpoint
  - 在指定 split 上计算：
      * 总 loss（分类 + 回归）
      * intent 准确率 (%)
      * threat MAE (0~100 标度)
  - 随机从该 split 中抽取四条样本（四种意图各一条），画 2x2 子图：
      * 展示 10 步轨迹 (XY) —— 通过“速度积分”恢复相对轨迹（起点在 (0,0)）
      * 在每条轨迹的末端标注：
          - GT 意图
          - Pred 意图
          - Pred 威胁度

用法（在项目根目录）：
    export PYTHONPATH="$PWD/code"

    python -m train.eval_intent --config config.yaml --split test
    # 或显式指定 ckpt:
    # python -m train.eval_intent --config config.yaml --ckpt path/to/xx.pt --split test

     python -m train.eval_intent --config config.yaml \
        --ckpt checkpoints_intent/20251203215429/best_intent_epoch003_valloss0.0777_acc96.39_mae2.972.pt \
        --split test
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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

from data.traj_dataset import build_datasets_from_config      # noqa: E402
from train.intent_model import build_model_from_config        # noqa: E402
from train.intent_loss import build_loss_from_config          # noqa: E402


INTENT_ID2NAME = {
    0: "ATTACK",
    1: "EVASION",
    2: "DEFENSE",
    3: "RETREAT",
}


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


# ====================== 通过“速度积分”恢复轨迹 ======================

def decode_window_to_xy_from_vel(
    window_norm: np.ndarray,   # [L,6] 归一化后的输入（可能做过 Δ / normalize）
    scaler,
    time_step: float,
) -> np.ndarray:
    """
    只用 vx, vy（km/s）通过积分恢复 10 步窗口的相对轨迹 (X,Y)，
    起点统一放在 (0,0)，不依赖原始 x,y，避免 Δ / 原点 等问题。

    步骤：
      1) 若使用了标准化，则 inverse_transform 恢复物理量；
      2) 取出 vx, vy（索引 3,4）；
      3) x[0] = 0, y[0] = 0；
         对 t=1..L-1:
             x[t] = x[t-1] + vx[t-1] * time_step
             y[t] = y[t-1] + vy[t-1] * time_step
    """
    assert window_norm.ndim == 2, f"expect [L,D], got {window_norm.shape}"
    L, D = window_norm.shape
    assert D >= 5, "expect at least 5 dims: [x,y,z,vx,vy,...]"

    # ---- 1) 反归一化：把 vx,vy 恢复到 km/s ----
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        feats = scaler.inverse_transform(window_norm)   # [L,6]
    else:
        feats = window_norm

    vx = feats[:, 3]   # km/s
    vy = feats[:, 4]   # km/s

    # ---- 2) 积分得到相对轨迹 ----
    x = np.zeros(L, dtype=np.float32)
    y = np.zeros(L, dtype=np.float32)

    # 用上一时刻速度积分，可以视为简单的“前向欧拉”
    for t in range(1, L):
        x[t] = x[t - 1] + vx[t - 1] * time_step
        y[t] = y[t - 1] + vy[t - 1] * time_step

    xy = np.stack([x, y], axis=1)  # [L,2]
    return xy


# ====================== metrics 计算 ======================

def _compute_batch_metrics(
    outputs: Dict[str, torch.Tensor],
    intent_labels: torch.Tensor,
    threat_scores: torch.Tensor,
) -> Tuple[int, int, float]:
    """
    计算一个 batch 的：
      - intent_correct: 预测意图正确的样本数
      - intent_total:   样本总数
      - threat_mae:     威胁度 MAE（0~100 标度）

    假定 intent_labels 中已不存在 ignore_label（-1），
    若存在请在调用前过滤。
    """
    logits_intent = outputs["logits_intent"]      # [B,4]
    threat_raw = outputs["threat_raw"]           # [B,1] or [B]

    if threat_raw.ndim == 2 and threat_raw.shape[1] == 1:
        threat_raw = threat_raw.squeeze(1)       # [B]

    if intent_labels.dtype != torch.long:
        intent_labels = intent_labels.long()

    # 意图预测
    pred_intent = torch.argmax(logits_intent, dim=1)    # [B]
    intent_correct = (pred_intent == intent_labels).sum().item()
    intent_total = intent_labels.numel()

    # 威胁度预测：raw -> sigmoid -> [0,1] -> *100
    threat_pred_norm = torch.sigmoid(threat_raw)
    threat_pred = threat_pred_norm * 100.0
    threat_mae = torch.abs(threat_pred - threat_scores).mean().item()

    return intent_correct, intent_total, threat_mae


def evaluate_loader(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    在给定 DataLoader 上评估：
      - 平均总 loss
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
        for batch in loader:
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

    return avg_loss, avg_acc, avg_mae


# ====================== 可视化：四种意图各一条轨迹 ======================

def visualize_four_intents(
    model: torch.nn.Module,
    ds_py,
    scaler,
    cfg: Dict[str, Any],
    device: torch.device,
    save_dir: Path,
) -> None:
    """
    从给定数据集 ds_py 中，对四种意图类别各随机选取一条样本，
    在一个 2x2 的图中画出 10 步轨迹，并在每条轨迹末端标注：
      - GT 意图
      - Pred 意图
      - Pred 威胁度

    轨迹恢复方式：只使用 vx, vy 通过积分得到“相对轨迹”，
    起点统一在 (0,0)，避免原点位置影响视觉尺度。
    """
    data_cfg = cfg.get("data", {})
    label_cfg = data_cfg.get("intent_threat", {})
    window_len = int(label_cfg.get("window_len", 10))
    time_step = float(data_cfg.get("time_step", 60.0))   # 秒

    save_dir.mkdir(parents=True, exist_ok=True)

    # 假设 ds_py 是 IntentThreatWindowDataset，并且有 intent_labels 属性
    intent_labels_np = ds_py.intent_labels  # [N]
    N = len(ds_py)
    if N == 0:
        print("[Eval] Dataset is empty, skip visualization.")
        return

    rng = np.random.default_rng()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for class_id in range(4):
        ax = axes[class_id]

        # 找到该类别的所有样本索引
        idx_list: List[int] = np.where(intent_labels_np == class_id)[0].tolist()
        if not idx_list:
            ax.set_title(f"{INTENT_ID2NAME.get(class_id, str(class_id))} (no sample)")
            ax.axis("off")
            continue

        idx = int(rng.choice(idx_list))

        # 取出一个样本
        x_np, intent_gt, threat_gt = ds_py[idx]   # x_np:[L,6]
        intent_gt = int(intent_gt)
        threat_gt = float(threat_gt)

        # 模型预测
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device).float()  # [1,L,6]
        model.eval()
        with torch.no_grad():
            outputs = model(x_tensor)
            logits_intent = outputs["logits_intent"]        # [1,4]
            threat_raw = outputs["threat_raw"]              # [1,1] or [1]

        pred_intent_id = int(torch.argmax(logits_intent, dim=1).item())
        if threat_raw.ndim == 2 and threat_raw.shape[1] == 1:
            threat_raw = threat_raw.squeeze(1)
        pred_threat_norm = torch.sigmoid(threat_raw)[0].item()
        pred_threat_score = float(pred_threat_norm * 100.0)

        # 通过速度积分恢复 XY 轨迹（相对坐标，起点在 0,0）
        pos_xy = decode_window_to_xy_from_vel(
            x_np, scaler, time_step=time_step
        )  # [L,2]
        x_coords = pos_xy[:, 0]
        y_coords = pos_xy[:, 1]

        # Debug：计算窗口总位移和平均速度
        dx = float(x_coords[-1] - x_coords[0])
        dy = float(y_coords[-1] - y_coords[0])
        dist_km = (dx ** 2 + dy ** 2) ** 0.5
        total_time_h = (window_len - 1) * time_step / 3600.0
        avg_speed_kmh = dist_km / total_time_h if total_time_h > 0 else 0.0
        print(
            f"[Debug] class={INTENT_ID2NAME.get(class_id)}, "
            f"sample_idx={idx}, window_dist={dist_km:.2f} km, "
            f"avg_speed={avg_speed_kmh:.2f} km/h"
        )

        # 画轨迹
        ax.plot(x_coords, y_coords, marker="o", linestyle="-", label="Window (10 steps)")

        # 标出起点/终点
        ax.scatter(x_coords[0], y_coords[0], marker="s", s=40, label="Start")
        ax.scatter(x_coords[-1], y_coords[-1], marker="X", s=60, label="End")

        # 在终点附近标注 Pred 信息
        text_str = (
            f"Pred: {INTENT_ID2NAME.get(pred_intent_id, pred_intent_id)} "
            f"({pred_intent_id})\n"
            f"Threat: {pred_threat_score:.1f}"
        )
        ax.text(
            x_coords[-1],
            y_coords[-1],
            text_str,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        # 标题显示 GT vs Pred
        title = (
            f"GT: {INTENT_ID2NAME.get(intent_gt, intent_gt)} ({intent_gt}), "
            f"GT Threat: {threat_gt:.1f}\n"
            f"Pred: {INTENT_ID2NAME.get(pred_intent_id, pred_intent_id)} ({pred_intent_id}), "
            f"Pred Threat: {pred_threat_score:.1f}"
        )
        ax.set_title(title, fontsize=10)

        ax.set_xlabel("ΔX (km)")
        ax.set_ylabel("ΔY (km)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        # 让范围稍微紧一点，避免看起来“跨度很大”
        margin = 0.5
        xmin, xmax = x_coords.min(), x_coords.max()
        ymin, ymax = y_coords.min(), y_coords.max()
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        ax.legend(fontsize=8)

    # 对于可能缺样本的类，把多余的 subplot 关闭
    for k in range(4, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    fname = save_dir / f"intent_four_examples_{int(time.time())}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved 4-intent visualization to: {fname}")


def pick_split_dataset(split: str, train_ds_py, val_ds_py, test_ds_py):
    split_l = split.lower()
    if split_l == "train":
        return train_ds_py
    elif split_l in ("val", "valid", "validation"):
        return val_ds_py
    elif split_l == "test":
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

    # 优先用 train_intent 段；没有就 fallback 到 train
    train_cfg = cfg.get("train_intent", cfg.get("train", {}))
    data_cfg = cfg.get("data", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = setup_device(train_cfg)

    # 构建 Python 层数据集
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

    # DataLoader 用于统计指标
    batch_size = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 256)))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = (device.type == "cuda")

    eval_loader = DataLoader(
        eval_ds_py,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 构建模型 & 加载权重
    print("[Info] Building IntentThreatNet ...")
    model = build_model_from_config(cfg).to(device)

    # 确定 ckpt 目录
    ckpt_dir_str = train_cfg.get("ckpt_dir", "checkpoints_intent")
    ckpt_dir = (project_root / ckpt_dir_str).resolve()

    if ckpt_path is None or ckpt_path == "":
        latest = find_latest_ckpt(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(
                f"在 {ckpt_dir} 下没有找到任何 .pt checkpoint，"
                f"请检查 train_intent.ckpt_dir 或使用 --ckpt 显式指定。"
            )
        ckpt_path = str(latest)
        print(f"[Info] No ckpt specified. Using latest: {ckpt_path}")
    else:
        ckpt_path = str(Path(ckpt_path).resolve())

    print(f"[Info] Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 损失函数
    print("[Info] Building IntentThreatLoss ...")
    loss_fn = build_loss_from_config(cfg)

    # ===== 在 eval_loader 上统计指标 =====
    print("[Info] Evaluating on split ...")
    t_start = time.time()
    avg_loss, avg_acc, avg_mae = evaluate_loader(
        model, loss_fn, eval_loader, device
    )
    elapsed = time.time() - t_start

    print("\n========== Evaluation (Intent + Threat) ==========")
    print(f"Split                     = {split}")
    print(f"Total loss                = {avg_loss:.6f}")
    print(f"Intent accuracy           = {avg_acc:.2f}%")
    print(f"Threat MAE (0~100 scale)  = {avg_mae:.3f}")
    print(f"Elapsed time              = {elapsed:.1f}s")
    print("==================================================\n")

    # ===== 可视化四种意图各一条轨迹 =====
    vis_dir = project_root / "eval_vis_intent"
    visualize_four_intents(model, eval_ds_py, scaler, cfg, device, vis_dir)


# ====================== CLI 入口 ======================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate intent & threat predictor (PyTorch MLP)."
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
        help=(
            "Path to .pt checkpoint of intent model. "
            "If empty, will use latest in train_intent.ckpt_dir "
            "(or checkpoints_intent by default)."
        ),
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
