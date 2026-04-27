#!/usr/bin/env python3
"""
lstm2/code/train/eval.py
------------------------
LSTM2 评估脚本（意图 + 威胁度）。

两种互斥模式（参考 lstm1/code/train/eval.py 的风格）：

  ---------- 模式 A：指标评估（默认） ----------
  只跑全量指标，不画图：
      * 总 loss / Intent acc / Threat MAE
      * Per-class precision / recall / F1
      * 4×4 混淆矩阵
      * Threat 分桶（[0,20), [20,40), [40,60), [60,80), [80,100]）准确率
      * forward 耗时（per batch / per sample）

  ---------- 模式 B：可视化（--vis） ----------
  只画图（不算指标）：从指定 split 中抽 vis_num 个 gnn1 sample，
  每个 sample 单独保存一张 .png 到
      lstm2/eval_vis/<ckpt_run>__<ckpt_name>/eval_<split>_<YYYYMMDD_HHMMSS>/
                     └── sample_XXX_idxYYYYYYY.png
  ckpt 子目录下再嵌一层 eval 时间戳，同一 ckpt 多次跑 vis 不互相覆盖。
  每张图含 (hist + GT future + 3 条 refined)，标注 GT/Pred intent + threat。

用法（在 new_plan/lstm2/ 下）：
    $env:PYTHONPATH = "$PWD/code"

    # ---------- 模式 A：指标评估（默认） ----------
    python -m train.eval --config config.yaml --split test
    # 显式指定 ckpt：
    python -m train.eval --config config.yaml --split test \
        --ckpt checkpoints/<run>/best_intent_*.pt

    # ---------- 模式 B：可视化（--vis） ----------
    # 默认 10 张：
    python -m train.eval --config config.yaml --split test --vis
    # 想要 30 张：
    python -m train.eval --config config.yaml --split test --vis --vis-num 30
    # 想可复现就给 --vis-seed：
    python -m train.eval --config config.yaml --split test --vis \
        --vis-num 30 --vis-seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt   # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../code/train
CODE_DIR = os.path.dirname(THIS_DIR)                     # .../code
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from data.dataset import build_datasets_from_config       # noqa: E402
from train.model import build_model_from_config            # noqa: E402
from train.loss import build_loss_from_config              # noqa: E402

# 从 trainer 复用 collate
from train.trainer import _collate                          # noqa: E402


# ============================================================
# 通用
# ============================================================

INTENT_NAMES = ["ATTACK", "EVASION", "DEFENSE", "RETREAT"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(train_cfg: Dict[str, Any]) -> torch.device:
    dev = str(train_cfg.get("device", "auto")).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def find_latest_ckpt(p: Path) -> Optional[Path]:
    if p is None:
        return None
    if p.is_file():
        return p
    if not p.exists():
        return None
    cands = list(p.rglob("*.pt"))
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def pick_split_dataset(split: str, train_ds, val_ds, test_ds):
    s = split.lower()
    if s == "train":
        return train_ds
    if s in ("val", "valid", "validation"):
        return val_ds
    if s == "test":
        return test_ds
    raise ValueError(f"Unknown split: {split}")


# ============================================================
# 指标计算
# ============================================================

def _compute_class_metrics(confusion: np.ndarray) -> Dict[str, np.ndarray]:
    """
    confusion[g, p]：行=GT，列=Pred。
    每类返回 precision / recall / F1。
    """
    K = confusion.shape[0]
    precision = np.zeros(K, dtype=np.float64)
    recall = np.zeros(K, dtype=np.float64)
    f1 = np.zeros(K, dtype=np.float64)
    for k in range(K):
        tp = float(confusion[k, k])
        fp = float(confusion[:, k].sum() - tp)
        fn = float(confusion[k, :].sum() - tp)
        precision[k] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[k] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision[k] + recall[k] > 0:
            f1[k] = 2.0 * precision[k] * recall[k] / (precision[k] + recall[k])
    return {"precision": precision, "recall": recall, "f1": f1}


def _threat_bucket_acc(
    pred_threat: np.ndarray,    # [N] 0..100
    gt_threat: np.ndarray,      # [N] 0..100
) -> Dict[str, float]:
    """
    分桶 [0,20), [20,40), [40,60), [60,80), [80,100]，
    pred 落在 GT 同一桶就算对。
    """
    bins = np.array([0, 20, 40, 60, 80, 101])
    g_bin = np.digitize(gt_threat, bins[1:-1], right=False)   # 0..4
    p_bin = np.digitize(pred_threat, bins[1:-1], right=False)
    overall = float((g_bin == p_bin).mean()) if len(g_bin) > 0 else 0.0

    out = {"overall": overall}
    for b in range(5):
        m = g_bin == b
        if m.any():
            out[f"bucket_{int(bins[b])}_{int(bins[b+1]-1)}"] = float(
                (p_bin[m] == g_bin[m]).mean()
            )
        else:
            out[f"bucket_{int(bins[b])}_{int(bins[b+1]-1)}"] = float("nan")
    return out


def evaluate_loader(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
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

    pred_threat_all: List[np.ndarray] = []
    gt_threat_all: List[np.ndarray] = []

    total_forward_time = 0.0
    total_forward_calls = 0
    min_fwd = float("inf")
    max_fwd = 0.0

    with torch.no_grad():
        for batch in loader:
            fut = batch["fut_norm"].to(device)
            pos = batch["position"].to(device)
            intent = batch["intent"].to(device)
            threat = batch["threat"].to(device)
            B = int(intent.size(0))

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(fut, pos)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            total_forward_time += dt
            total_forward_calls += 1
            min_fwd = min(min_fwd, dt)
            max_fwd = max(max_fwd, dt)

            total, _, _ = loss_fn(out, intent, threat)
            total_loss += float(total.detach().cpu().item())
            total_batches += 1

            mask = intent != ignore_label
            if mask.any():
                logits = out["logits_intent"][mask]
                pred = torch.argmax(logits, dim=1)
                gt = intent[mask]
                total_correct += int((pred == gt).sum().item())
                total_samples += int(mask.sum().item())
                for g, p in zip(gt.cpu().numpy(), pred.cpu().numpy()):
                    if 0 <= g < num_classes and 0 <= p < num_classes:
                        confusion[int(g), int(p)] += 1

                raw = out["threat_raw"]
                if raw.ndim == 2 and raw.shape[1] == 1:
                    raw = raw.squeeze(1)
                pt = (torch.sigmoid(raw[mask]) * 100.0).cpu().numpy()
                gt_t = threat[mask].cpu().numpy()
                total_mae_sum += float(np.abs(pt - gt_t).mean())
                pred_threat_all.append(pt)
                gt_threat_all.append(gt_t)

    nb = max(1, total_batches)
    avg_loss = total_loss / nb
    acc = 100.0 * total_correct / max(1, total_samples)
    mae = total_mae_sum / nb

    cls_metrics = _compute_class_metrics(confusion)

    if pred_threat_all:
        pt = np.concatenate(pred_threat_all)
        gt = np.concatenate(gt_threat_all)
        bucket = _threat_bucket_acc(pt, gt)
    else:
        bucket = {}

    if total_forward_calls == 0:
        min_fwd = 0.0
        max_fwd = 0.0

    return {
        "avg_loss":      avg_loss,
        "intent_acc":    acc,
        "threat_mae":    mae,
        "confusion":     confusion,
        "cls_metrics":   cls_metrics,
        "threat_bucket": bucket,
        "fwd_avg_batch_s":  total_forward_time / max(1, total_forward_calls),
        "fwd_avg_sample_s": total_forward_time / max(1, total_samples),
        "fwd_min_batch_s":  min_fwd,
        "fwd_max_batch_s":  max_fwd,
        "n_samples":     total_samples,
    }


# ============================================================
# 可视化（每个 gnn1 sample 一张子图：hist + GT future + 3 条 refined）
# ============================================================

INTENT_COLOR_BY_PRED = {
    0: "#d62728",   # ATTACK 红
    1: "#9467bd",   # EVASION 紫
    2: "#1f77b4",   # DEFENSE 蓝
    3: "#2ca02c",   # RETREAT 绿
}


def _gather_grouped_samples(
    dataset,
    sample_ids: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    把 dataset 里属于这些 sample_ids 的所有候选行聚合成 per-sample 字典。
    每个字典：
        sample_id, hist, position, fut_gt(可选), candidates: [{cand_k, fut_phys, intent_gt, threat_gt, topology, fut_norm}]
    """
    sample_idx_arr = dataset._sample_idx     # [M]
    cand_k_arr = dataset._cand_k             # [M]

    out: List[Dict[str, Any]] = []
    for sid in sample_ids:
        rows = np.where(sample_idx_arr == sid)[0]
        if len(rows) == 0:
            continue
        rows = rows[np.argsort(cand_k_arr[rows])]
        first = int(rows[0])

        cands = []
        for r in rows:
            r = int(r)
            cands.append({
                "cand_k":     int(dataset._cand_k[r]),
                "fut_phys":   dataset._fut_phys[r],
                "fut_norm":   dataset._fut_norm[r],
                "intent_gt":  int(dataset._intent[r]),
                "threat_gt":  float(dataset._threat[r]),
                "topology":   (str(dataset._topology[r])
                               if dataset._topology is not None else ""),
            })

        item = {
            "sample_id":  int(sid),
            "hist":       dataset._hist[first],         # [20, 6]
            "position":   dataset._pos[first],          # [3]
            "fut_gt":     (dataset._fut_gt[first]
                           if dataset._fut_gt is not None else None),
            "candidates": cands,
        }
        out.append(item)
    return out


def _predict_grouped(
    model: torch.nn.Module,
    grouped: List[Dict[str, Any]],
    device: torch.device,
) -> None:
    """对每个 grouped 中的候选做预测，原地写入 pred_intent / pred_threat。"""
    if not grouped:
        return
    fut_norm_list = []
    pos_list = []
    flat_index: List[Tuple[int, int]] = []   # (sample_idx_in_grouped, cand_idx_in_sample)
    for si, item in enumerate(grouped):
        for ci, cand in enumerate(item["candidates"]):
            fut_norm_list.append(cand["fut_norm"])
            pos_list.append(item["position"])
            flat_index.append((si, ci))

    if not fut_norm_list:
        return

    fut_t = torch.from_numpy(np.stack(fut_norm_list, axis=0)).float().to(device)
    pos_t = torch.from_numpy(np.stack(pos_list, axis=0)).float().to(device)

    model.eval()
    with torch.no_grad():
        out = model(fut_t, pos_t)
    pred_intent = torch.argmax(out["logits_intent"], dim=1).cpu().numpy()
    raw = out["threat_raw"]
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw.squeeze(1)
    pred_threat = (torch.sigmoid(raw) * 100.0).cpu().numpy()

    for k, (si, ci) in enumerate(flat_index):
        grouped[si]["candidates"][ci]["pred_intent"] = int(pred_intent[k])
        grouped[si]["candidates"][ci]["pred_threat"] = float(pred_threat[k])


def _plot_one_grouped_sample(
    item: Dict[str, Any],
    save_dir: Path,
    save_filename: str,
) -> None:
    """
    一个 gnn1 sample 一张图：hist（灰）+ GT future（紫虚线）+ K 条 refined
    （按 pred_intent 上色），位置点用红星，每条候选 legend 注 GT/Pred intent + threat。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    hist = item["hist"]
    pos = item["position"]
    fut_gt = item.get("fut_gt", None)

    fig, ax = plt.subplots(figsize=(6, 6))

    # 历史
    ax.plot(hist[:, 0], hist[:, 1], "-", color="gray", lw=1.8,
            label="hist", zorder=2)
    ax.scatter(hist[-1, 0], hist[-1, 1], color="black", s=36, zorder=4,
               label="now (0,0)")

    # 我方目标点
    ax.scatter(pos[0], pos[1], color="red", marker="*", s=180,
               zorder=6, label="our target")

    # GT 未来（虚线）
    if fut_gt is not None:
        ax.plot(fut_gt[:, 0], fut_gt[:, 1], "--", color="purple",
                lw=1.3, alpha=0.85, zorder=3, label="GT future")

    # K 条候选
    for cand in item["candidates"]:
        f = cand["fut_phys"]
        it_gt = INTENT_NAMES[int(cand["intent_gt"])]
        tr_gt = int(cand["threat_gt"])
        it_pd = INTENT_NAMES[int(cand["pred_intent"])]
        tr_pd = float(cand["pred_threat"])
        color = INTENT_COLOR_BY_PRED[int(cand["pred_intent"])]
        mark = "OK" if it_pd == it_gt else "X"
        ax.plot(
            f[:, 0], f[:, 1], "-", color=color, lw=2.0, zorder=5,
            label=(
                f"k={int(cand['cand_k'])}  "
                f"GT={it_gt}/{tr_gt}  "
                f"Pred={it_pd}/{tr_pd:.0f}  [{mark}]"
            ),
        )

    ax.set_title(f"lstm2 eval  sample={item['sample_id']}", fontsize=11)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_path = save_dir / save_filename
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 总评估入口
# ============================================================

def evaluate(
    config_path: str,
    ckpt_path: Optional[str],
    split: str,
    vis: bool,
    vis_num: int,
    vis_seed: Optional[int],
) -> None:
    project_root = Path(__file__).resolve().parents[2]   # .../lstm2

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    cfg = load_config(str(cfg_path))

    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = setup_device(train_cfg)
    print(f"[Info] device = {device}")

    rel_cfg = cfg_path.relative_to(project_root) if cfg_path.is_absolute() else Path(config_path)
    train_ds, val_ds, test_ds, _scaler = build_datasets_from_config(str(rel_cfg))
    print(
        f"[Info] dataset sizes: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )

    eval_ds = pick_split_dataset(split, train_ds, val_ds, test_ds)
    print(f"[Info] using split='{split}', size={len(eval_ds)}")
    if len(eval_ds) == 0:
        print("[Eval] split 为空，退出。")
        return

    # ---- 模型 ----
    print("[Info] building model ...")
    model = build_model_from_config(cfg).to(device)

    ckpt_root = (project_root / train_cfg.get("ckpt_dir", "checkpoints")).resolve()
    if ckpt_path is None or ckpt_path == "":
        latest = find_latest_ckpt(ckpt_root)
        if latest is None:
            raise FileNotFoundError(
                f"在 {ckpt_root} 下没找到任何 .pt；用 --ckpt 显式指定。"
            )
        ckpt_used = str(latest)
        print(f"[Info] no --ckpt; using latest: {ckpt_used}")
    else:
        ckpt_used = str(Path(ckpt_path).resolve())
    print(f"[Info] loading ckpt: {ckpt_used}")
    sd = torch.load(ckpt_used, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ====== 模式 B：可视化（逐样本独立 .png，仿 lstm1/eval.py） ======
    if vis:
        ckpt_obj = Path(ckpt_used)
        ckpt_sub = f"{ckpt_obj.parent.name}__{ckpt_obj.stem}"
        # 同一 ckpt 多次跑 vis 不互相覆盖：再嵌一层 eval 时间戳
        eval_ts = time.strftime("%Y%m%d_%H%M%S")
        eval_sub = f"eval_{split}_{eval_ts}"
        vis_subdir = (project_root / "eval_vis" / ckpt_sub / eval_sub).resolve()
        vis_subdir.mkdir(parents=True, exist_ok=True)

        if vis_seed is not None:
            rng = np.random.default_rng(int(vis_seed))
        else:
            rng = np.random.default_rng()

        # 抽 vis_num 个不重复的 gnn1 sample（不够就退到允许重复）
        uniq = np.unique(eval_ds._sample_idx)
        n_uniq = int(len(uniq))
        if n_uniq == 0:
            print("[Vis] split 中没有 gnn1 sample，无法可视化。")
            return
        if int(vis_num) <= 0:
            print(f"[Vis] vis_num={vis_num} <= 0，跳过画图。")
            return

        if int(vis_num) <= n_uniq:
            chosen = rng.choice(uniq, size=int(vis_num), replace=False)
        else:
            print(f"[Vis] vis_num={vis_num} 超过 split 中 sample 数 {n_uniq}，"
                  f"将允许重复抽样。")
            chosen = rng.choice(uniq, size=int(vis_num), replace=True)

        print(f"\n========== Visualization ==========")
        print(f"split           = {split}")
        print(f"save dir        = {vis_subdir}")
        print(f"num samples     = {int(vis_num)}")
        preview = ", ".join(str(int(s)) for s in chosen[:20].tolist())
        print(f"sample ids      = {preview}{' ...' if len(chosen) > 20 else ''}")
        print(f"===================================\n")

        # 一次性 gather + predict（比逐样本走快得多）
        grouped = _gather_grouped_samples(eval_ds, chosen)
        _predict_grouped(model, grouped, device)

        # 逐样本独立保存：sample_XXX_idxYYYYYYY.png（仿 lstm1）
        t_start = time.time()
        for k, item in enumerate(grouped):
            sid = int(item["sample_id"])
            fname = f"sample_{k:03d}_idx{sid:07d}.png"
            _plot_one_grouped_sample(item, save_dir=vis_subdir, save_filename=fname)
        elapsed = time.time() - t_start
        print(f"[Info] visualization done. {len(grouped)} figures in: {vis_subdir}")
        print(f"[Info] elapsed: {elapsed:.1f}s")
        return

    # ====== 模式 A：指标 ======
    bs = int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 256)))
    nw = int(train_cfg.get("num_workers", 0))
    pin = (device.type == "cuda")

    eval_loader = DataLoader(
        eval_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=pin, collate_fn=_collate, drop_last=False,
    )
    loss_fn = build_loss_from_config(cfg).to(device)
    ignore_label = int(loss_cfg.get("ignore_label", -1))

    print("[Info] evaluating ...")
    t_start = time.time()
    res = evaluate_loader(
        model=model, loss_fn=loss_fn, loader=eval_loader,
        device=device, ignore_label=ignore_label, num_classes=4,
    )
    elapsed = time.time() - t_start

    print("\n========== Evaluation (LSTM2) ==========")
    print(f"split           = {split}")
    print(f"valid samples   = {res['n_samples']}")
    print(f"total loss      = {res['avg_loss']:.6f}")
    print(f"intent_acc      = {res['intent_acc']:.2f}%")
    print(f"threat_mae      = {res['threat_mae']:.3f}  (0..100 scale)")

    print("\n--- Per-class metrics ---")
    print(f"  {'class':<10}  {'P':>7}  {'R':>7}  {'F1':>7}")
    cm = res["cls_metrics"]
    for k in range(4):
        print(f"  {INTENT_NAMES[k]:<10}  "
              f"{cm['precision'][k]:7.4f}  "
              f"{cm['recall'][k]:7.4f}  "
              f"{cm['f1'][k]:7.4f}")

    print("\n--- Confusion matrix (rows=GT, cols=Pred) ---")
    headers = ["ATK", "EVA", "DEF", "RET"]
    print("       " + "  ".join(f"{h:>6}" for h in headers))
    for i, h in enumerate(headers):
        row = "  ".join(f"{int(res['confusion'][i, j]):6d}" for j in range(4))
        print(f"  {h:<3}: " + row)

    print("\n--- Threat bucket accuracy ---")
    if res["threat_bucket"]:
        b = res["threat_bucket"]
        print(f"  overall = {b['overall']:.4f}")
        for key in ["bucket_0_19", "bucket_20_39", "bucket_40_59",
                    "bucket_60_79", "bucket_80_100"]:
            v = b.get(key, float("nan"))
            print(f"  {key:<16} = {v:.4f}" if not np.isnan(v)
                  else f"  {key:<16} = (no GT in bucket)")

    print("\n--- Forward timing ---")
    print(f"  avg per batch  = {res['fwd_avg_batch_s'] * 1000:.3f} ms")
    print(f"  avg per sample = {res['fwd_avg_sample_s'] * 1000:.3f} ms")
    print(f"  min/max batch  = "
          f"{res['fwd_min_batch_s'] * 1000:.3f} / "
          f"{res['fwd_max_batch_s'] * 1000:.3f} ms")

    print(f"\nelapsed = {elapsed:.1f}s")
    print("=========================================\n")
    print("[Info] 指标模式：未画图。要可视化请加 --vis（可配 --vis-num N）。")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate lstm2 (intent + threat).\n\n"
            "两种互斥模式：\n"
            "  默认 (不加 --vis)：只跑全量指标评估\n"
            "                      （loss / acc / MAE / per-class P/R/F1 /\n"
            "                       confusion / threat 分桶 / forward 耗时）\n"
            "  --vis             ：只画 vis_num 张可视化图，跳过指标\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="lstm2 config 路径")
    parser.add_argument("--ckpt", type=str, default="",
                        help="ckpt 路径；空则用 train.ckpt_dir 下最新 .pt")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--vis", action="store_true",
                        help="开启可视化模式（只画图、不算指标）。每个 gnn1 sample "
                             "单独存一张图，保存到 "
                             "eval_vis/<ckpt_run>__<ckpt_name>/ 子目录。")
    parser.add_argument("--vis-num", type=int, default=10,
                        help="可视化模式下画几张图（每张随机抽一个不同 gnn1 sample）。"
                             "默认 10。")
    parser.add_argument("--vis-seed", type=int, default=None,
                        help="可视化抽样的随机种子（不指定则每次结果不同）。")
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        vis=args.vis,
        vis_num=args.vis_num,
        vis_seed=args.vis_seed,
    )


if __name__ == "__main__":
    main()
