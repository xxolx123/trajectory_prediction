#!/usr/bin/env python3
"""
lstm2/code/data/diagnose_gnn1_ood.py
------------------------------------
诊断 GNN1 在 OOD position 下的行为。

把 val 集里的 position 替换成 4 种方向（相对候选群"主方向"）：
    forward   :  +u_main * dist        （in-distribution，sanity check）
    backward  :  -u_main * dist        （OOD：position 在反方向）
    side_left :  +n_main * dist        （OOD：position 在左侧）
    side_right:  -n_main * dist        （OOD：position 在右侧）

其中 u_main = 5 候选平均终点的单位方向，dist = 1.5 × |avg_endpoint|，
保证 4 种 position 离原点距离相同，仅方向不同。

对每种方向跑 GNN1 forward，统计：
    - mean(max(softmax))           top-1 概率均值，越接近 1 = 越自信
    - mean(entropy / log(M))       归一化熵，越接近 1 = 越接近均匀（"承认不确定"）
    - mean(top3_probs sum)         top-3 重归一前的总概率
    - top-1 selection 一致性       和 forward 比，OOD 时 top-1 是否还是同一条
                                   （高 = GNN1 不响应 position）

判断：
    - OOD 熵 ≥ 0.7 ：GNN1 老实承认不确定 → LSTM2 fix 路线（B）足够
    - OOD 熵 ≤ 0.5 且 OOD top1prob ≥ 0.6 ：GNN1 自信瞎选 → 必须改 GNN1 数据（路线 C）

用法（在 new_plan/lstm2 下）：
    $env:PYTHONPATH = "$PWD/code;$PWD/.."
    python -m data.diagnose_gnn1_ood --n 5000 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml


# ====== sys.path / imports ======================================
_THIS_FILE = Path(__file__).resolve()
_LSTM2_ROOT = _THIS_FILE.parents[2]
_REPO_ROOT = _LSTM2_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_LSTM2_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(_LSTM2_ROOT / "code"))

from gnn1.code.train.model import build_model_from_config as build_gnn1_model     # noqa: E402


# ====== 工具 =====================================================

def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_ckpt(p: Path) -> Optional[Path]:
    if p.is_file():
        return p
    if p.is_dir():
        cands = list(p.rglob("*.pt"))
        if not cands:
            return None
        cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]
    return None


class _Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.std[np.abs(self.std) < 1e-9] = 1.0

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        m = self.mean.reshape((1,) * (x.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (x.ndim - 1) + (-1,))
        return x * s + m

    @classmethod
    def load(cls, path: Path) -> "_Scaler":
        a = np.load(path)
        return cls(a["mean"], a["std"])


def decode_future_to_phys(fut_norm: np.ndarray, scaler: _Scaler) -> np.ndarray:
    feat = scaler.inverse_transform(fut_norm.astype(np.float64))
    pos_delta = feat[..., :3]
    vel = feat[..., 3:6]
    pos = np.cumsum(pos_delta, axis=-2)
    return np.concatenate([pos, vel], axis=-1).astype(np.float32)


# ====== 主流程 ===================================================

def run_gnn1_full_probs(
    model: torch.nn.Module,
    candidates_norm: np.ndarray,
    task_type: np.ndarray,
    type_id: np.ndarray,
    position: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    跑 GNN1，返回:
        probs       [N, M]
        top_idx     [N, K]
    """
    model.eval()
    N = candidates_norm.shape[0]
    probs_list, top_idx_list = [], []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            batch = {
                "cand_trajs": torch.from_numpy(candidates_norm[i:j]).to(device).float(),
                "task_type":  torch.from_numpy(task_type[i:j]).to(device).long(),
                "type":       torch.from_numpy(type_id[i:j]).to(device).long(),
                "position":   torch.from_numpy(position[i:j]).to(device).float(),
            }
            out = model(batch)
            probs_list.append(out["probs"].detach().cpu().numpy())
            top_idx_list.append(out["top_idx"].detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0), np.concatenate(top_idx_list, axis=0)


def build_position_variants(
    cand_phys: np.ndarray,                  # [N, 5, 10, 6]
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    给每个样本造 4 种 position 变体。
        u_main = avg_endpoint / |avg_endpoint|   (xy 方向，单位向量)
        dist   = 1.5 * |avg_endpoint|            (固定倍数，保证 4 个 position 半径相同)
    """
    N = cand_phys.shape[0]
    avg_end = cand_phys[:, :, -1, :2].mean(axis=1)            # [N, 2]
    L = np.linalg.norm(avg_end, axis=-1)                       # [N]
    L_safe = np.where(L < 1e-3, 1.0, L)                        # 兜底
    u = avg_end / L_safe[:, None]                              # [N, 2]
    n = np.stack([-u[:, 1], u[:, 0]], axis=-1)                 # 90° CCW

    dist = (1.5 * L_safe).reshape(N, 1)                        # [N, 1]
    z = np.zeros((N, 1), dtype=np.float64)

    fwd = np.concatenate([dist * u, z], axis=-1)
    bwd = np.concatenate([-dist * u, z], axis=-1)
    sl = np.concatenate([dist * n, z], axis=-1)
    sr = np.concatenate([-dist * n, z], axis=-1)
    return {
        "forward":    fwd.astype(np.float32),
        "backward":   bwd.astype(np.float32),
        "side_left":  sl.astype(np.float32),
        "side_right": sr.astype(np.float32),
    }


def stats_for_probs(probs: np.ndarray, top_idx: np.ndarray) -> Dict[str, float]:
    """
    probs   [N, M]
    top_idx [N, K]   (按概率降序)
    """
    M = probs.shape[1]
    top1_prob = probs.max(axis=-1)                                  # [N]
    eps = 1e-12
    H = -np.sum(probs * np.log(probs + eps), axis=-1)               # [N]
    H_norm = H / np.log(M)                                          # [N], in [0,1]

    K = top_idx.shape[1]
    sum_topk = np.take_along_axis(probs, top_idx, axis=1).sum(axis=-1)   # [N]

    # 顶端尖锐度：top1 > 0.5 / > 0.7 的比例
    sharp_50 = float((top1_prob > 0.5).mean())
    sharp_70 = float((top1_prob > 0.7).mean())

    return {
        "top1_mean":   float(top1_prob.mean()),
        "top1_median": float(np.median(top1_prob)),
        "Hnorm_mean":  float(H_norm.mean()),
        "Hnorm_median": float(np.median(H_norm)),
        "sumtopK":     float(sum_topk.mean()),
        "sharp_50":    sharp_50,
        "sharp_70":    sharp_70,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000,
                        help="抽取多少个 window 做诊断")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_LSTM2_ROOT / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    data_cfg = cfg.get("data", {})
    cache_dir = (_LSTM2_ROOT / data_cfg.get("gnn1_cache_dir", "../gnn1/data/cache")).resolve()
    raw_dir = (_LSTM2_ROOT / data_cfg.get("gnn1_raw_dir", "../gnn1/data/raw")).resolve()
    gnn1_cfg_path = (_LSTM2_ROOT / data_cfg.get("gnn1_config", "../gnn1/config.yaml")).resolve()
    gnn1_ckpt_path = (_LSTM2_ROOT / data_cfg.get("gnn1_ckpt", "../gnn1/checkpoints")).resolve()
    scaler_path = (_LSTM2_ROOT / data_cfg.get(
        "lstm1_scaler", "../lstm1/data/processed/scaler_posvel.npz")).resolve()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device = {device}")

    # ---- 加载 GNN1 ----
    gnn1_cfg = load_yaml(gnn1_cfg_path)
    model = build_gnn1_model(gnn1_cfg).to(device)
    ckpt = find_latest_ckpt(gnn1_ckpt_path)
    if ckpt is None:
        print(f"找不到 GNN1 ckpt: {gnn1_ckpt_path}")
        sys.exit(1)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[diag] loaded {ckpt.name}")

    # ---- 数据 ----
    raw = np.load(raw_dir / "val.npz")
    cand_norm = raw["candidates"]                # [N_s, 5, 10, 6]
    task_type = raw["task_type"].astype(np.int64)
    type_id = raw["type"].astype(np.int64)
    pos_orig = raw["position"].astype(np.float32)
    N_full = cand_norm.shape[0]

    rng = np.random.default_rng(args.seed)
    n = min(args.n, N_full)
    sel = rng.choice(N_full, size=n, replace=False)
    cand_norm = cand_norm[sel]
    task_type = task_type[sel]
    type_id = type_id[sel]
    pos_orig = pos_orig[sel]
    print(f"[diag] sampled {n} / {N_full} windows")

    # ---- decode candidates 到物理 ----
    scaler = _Scaler.load(scaler_path)
    cand_phys = decode_future_to_phys(cand_norm, scaler)   # [n, 5, 10, 6]

    # ---- 构造 4 种方向 + 'orig' 共 5 种 position ----
    variants = build_position_variants(cand_phys, rng)
    variants["orig"] = pos_orig

    # ---- 跑 GNN1 ----
    probs_by: Dict[str, np.ndarray] = {}
    top_by: Dict[str, np.ndarray] = {}
    for name, pos in variants.items():
        probs, top_idx = run_gnn1_full_probs(
            model, cand_norm, task_type, type_id, pos, device,
        )
        probs_by[name] = probs
        top_by[name] = top_idx

    # ---- 统计 ----
    print()
    print("=" * 78)
    print(f"{'direction':<12} {'top1_mean':>10} {'top1_med':>10} "
          f"{'Hnorm_mean':>11} {'Hnorm_med':>11} "
          f"{'sumtop3':>9} {'>0.5':>7} {'>0.7':>7}")
    print("-" * 78)
    for name in ["orig", "forward", "backward", "side_left", "side_right"]:
        s = stats_for_probs(probs_by[name], top_by[name])
        print(f"{name:<12} {s['top1_mean']:>10.3f} {s['top1_median']:>10.3f} "
              f"{s['Hnorm_mean']:>11.3f} {s['Hnorm_median']:>11.3f} "
              f"{s['sumtopK']:>9.3f} {s['sharp_50']:>7.1%} {s['sharp_70']:>7.1%}")
    print("=" * 78)

    # ---- top-1 一致性（forward vs others）----
    print()
    print("top-1 一致性（forward 选的 top-1 在其他方向是否还是同一条）：")
    print("    高 = GNN1 不响应 position 变化 / 低 = 真的在响应")
    fwd_top1 = top_by["forward"][:, 0]
    for name in ["orig", "backward", "side_left", "side_right"]:
        other_top1 = top_by[name][:, 0]
        agree = float((fwd_top1 == other_top1).mean())
        print(f"    forward vs {name:<12}: {agree:.1%}")

    # ---- 距离匹配度：所选 top-1 endpoint 到该方向 position 的距离 ----
    print()
    print("top-1 候选 endpoint 到 position 的距离 (km)：")
    print("    in-distribution (forward) 应该最小，OOD 方向应该都偏大")
    cand_end = cand_phys[:, :, -1, :3]      # [n, 5, 3]
    for name in ["orig", "forward", "backward", "side_left", "side_right"]:
        pos = variants[name]                # [n, 3]
        top1 = top_by[name][:, 0]           # [n]
        sel_end = cand_end[np.arange(n), top1]   # [n, 3]
        d = np.linalg.norm(sel_end - pos, axis=-1)
        print(f"    {name:<12}: mean={d.mean():.3f}  median={np.median(d):.3f}  "
              f"p90={np.quantile(d, 0.9):.3f}")

    # ---- 5 候选到 position 的距离统计（衡量 in/out of distribution 程度）----
    print()
    print("5 候选到 position 距离的 spread (max - min)：")
    print("    in-distribution 一般 > 1 km；OOD 时可能很小说明 5 条都同样远")
    for name in ["orig", "forward", "backward", "side_left", "side_right"]:
        pos = variants[name][:, None, :]                          # [n, 1, 3]
        d = np.linalg.norm(cand_end - pos, axis=-1)               # [n, 5]
        spread = d.max(axis=-1) - d.min(axis=-1)                  # [n]
        print(f"    {name:<12}: mean={spread.mean():.3f}  median={np.median(spread):.3f}")

    # ---- 判断 ----
    print()
    print("=" * 78)
    s_bwd = stats_for_probs(probs_by["backward"], top_by["backward"])
    s_fwd = stats_for_probs(probs_by["forward"], top_by["forward"])
    print("判断:")
    print(f"  forward (in-dist)  H_norm = {s_fwd['Hnorm_mean']:.3f}, top1 = {s_fwd['top1_mean']:.3f}")
    print(f"  backward (OOD)     H_norm = {s_bwd['Hnorm_mean']:.3f}, top1 = {s_bwd['top1_mean']:.3f}")
    if s_bwd["Hnorm_mean"] >= 0.7:
        print("  -> GNN1 OOD 时熵高（接近均匀），承认不确定。[OK] 走 LSTM2 fix（路线 B）足够")
    elif s_bwd["Hnorm_mean"] <= 0.5 and s_bwd["top1_mean"] >= 0.6:
        print("  [FAIL] GNN1 OOD 时仍然自信地选某一条，可能是错误选择 -> 路线 C（改 GNN1 数据 + 重训）")
    else:
        print("  [WARN] GNN1 OOD 介于两者之间。建议：路线 B 但额外检查训练后 LSTM2 在 OOD 上的可视化")
    print("=" * 78)


if __name__ == "__main__":
    main()
