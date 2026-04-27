#!/usr/bin/env python3
"""
gnn1/code/data/generate_data.py
-------------------------------
基于 cache_lstm1_preds.py 产出的 {split}.npz（含 history + 5 candidates），
为每个 window 生成 samples_per_window 份训练样本。

每条样本字段：
  candidates:   [5, 10, 6]  float32   LSTM1 输出（归一化 + delta 空间）
  task_type:    ()          int8      敌方作战任务（目前只有 0 = 打击）
  type:         ()          int8      我方固定目标类型  0..2
  position:     [3]         float32   我方固定目标 xyz（km），每样本独立
  label:        ()          int8      5 个候选终点中离 position 最近的索引 k*
                                      （通常 == k_seed，但噪声大时会翻）
  k_seed:       ()          int8      生成 position 时用的"参考候选"索引
  position_dir: ()          int8      position 采样方向：
                                       0=forward / 1=backward
                                       2=side_left / 3=side_right
  soft_label:   [5]         float32   (仅当 data.soft_label_tau > 0 时产出)
                                      soft_k = softmax(-dist_k / tau)
                                      和 = 1，给训练用软 CE，避免预测 100% 极端

坐标系约定：
  - history 所在 window 的"最后一帧位置" 设为原点 (0, 0, 0)；
  - candidates 的端点、position 均是相对于该原点的 km 坐标。

position 4 方向采样（让 GNN1 在 OOD 时也学会输出均匀概率）：
  - forward    : pos = endpoint_k + ext * step_len * direction         (in-distribution)
  - backward   : pos = -direction × (|endpoint_k| + ext*step_len)      (OOD)
  - side_left  : pos = direction(90° CCW) × (|endpoint_k| + ext*step_len)
  - side_right : pos = direction(90° CW) × (|endpoint_k| + ext*step_len)
  各方向比例由 data.position_direction_mix 控制。

产出：data/raw/{train,val,test}.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

EPS = 1e-6


# =============== Scaler 加载（和 lstm1 用同一个） ===============

class _Scaler:
    """最小版的 StandardScaler，只为了反归一化 6 维特征。
    和 lstm1 的 StandardScaler 完全兼容的 npz 格式：键 mean/std 各一个 [D] 向量。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        m = self.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (data.ndim - 1) + (-1,))
        return data * s + m

    @classmethod
    def load(cls, path: Path) -> "_Scaler":
        arr = np.load(path)
        return cls(arr["mean"], arr["std"])


# =============== 配置读取 ===============

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============== 核心：从 delta+normalize 还原到 xy ===============

def decode_to_xy(
    feat_norm: np.ndarray,        # [*, T, D] 归一化 + delta
    scaler: _Scaler,
    hist_last_xy: Optional[np.ndarray] = None,  # [2] 或 None
) -> np.ndarray:
    """
    反归一化并 cumsum 到 xy。
      - 若 hist_last_xy=None：把每条序列首帧位置视为 (0, 0) → 直接累加 delta；
      - 若提供 hist_last_xy：把该坐标作为"起点的前一步"，cumsum 后加上它。

    返回 [*, T, 2]。
    """
    feat_orig = scaler.inverse_transform(feat_norm.astype(np.float64))
    delta_xy = feat_orig[..., :2]               # 前 2 维是 (dx, dy) in km
    xy = np.cumsum(delta_xy, axis=-2)           # [*, T, 2]
    if hist_last_xy is not None:
        xy = xy + hist_last_xy.reshape((1,) * (xy.ndim - 1) + (2,))
    return xy


# =============== 主：每个 window 产 M 条样本 ===============

def generate_for_split(
    history: np.ndarray,           # [N, Tin, 6] 归一化 + delta
    candidates: np.ndarray,        # [N, M, Tout, 6] 归一化 + delta
    scaler: _Scaler,
    data_cfg: Dict[str, Any],
    rng: np.random.Generator,
    targets: Optional[np.ndarray] = None,   # [N, Tout, 6] GT，可选（eval 要用）
) -> Dict[str, np.ndarray]:
    """返回一个 dict，字段见模块 docstring。"""
    N, Tin, D = history.shape
    _, M, Tout, _ = candidates.shape

    samples_per_window = int(data_cfg.get("samples_per_window", M))
    ext_lo, ext_hi = data_cfg.get("position_ext_steps", [1.0, 5.0])
    ext_lo, ext_hi = float(ext_lo), float(ext_hi)
    sigma_km = float(data_cfg.get("position_noise_km", 0.3))
    pos_z_default = float(data_cfg.get("position_z", 0.0))

    task_type_fixed = int(data_cfg.get("task_type_fixed", 0))
    type_lo, type_hi = data_cfg.get("type_range", [0, 2])
    type_lo, type_hi = int(type_lo), int(type_hi)

    soft_tau = float(data_cfg.get("soft_label_tau", 0.0))
    save_soft = soft_tau > 0.0

    # ---- position 采样方向混合 ----
    # 缺省：全 forward（兼容旧行为）
    dir_mix_cfg = data_cfg.get("position_direction_mix") or {}
    dir_p = {
        "forward":    float(dir_mix_cfg.get("forward",    1.0)),
        "backward":   float(dir_mix_cfg.get("backward",   0.0)),
        "side_left":  float(dir_mix_cfg.get("side_left",  0.0)),
        "side_right": float(dir_mix_cfg.get("side_right", 0.0)),
    }
    s = sum(dir_p.values())
    if s <= 0:
        raise ValueError(f"position_direction_mix 概率和必须 > 0，当前 {dir_mix_cfg}")
    for k in dir_p:
        dir_p[k] /= s
    # 累积概率（4 个方向，最后一个不用）
    p_fwd = dir_p["forward"]
    p_fwd_bwd = p_fwd + dir_p["backward"]
    p_fwd_bwd_sl = p_fwd_bwd + dir_p["side_left"]

    # ---- Step A: 对每个 window，一次性解码 5 条候选到 xy ----
    # 把 history 的最后一帧位置作为原点 (0,0)。
    # 注意：由于我们不需要绝对坐标，只需一致的相对坐标即可，直接令 hist_last_xy=(0,0)。
    # candidates[n, m] 展开成 xy（相对于 hist_last）。
    hist_last_xy = np.zeros(2, dtype=np.float64)  # 每个样本的原点统一取 (0,0)
    # candidates: [N, M, Tout, D] → decode → [N, M, Tout, 2]
    cand_xy = decode_to_xy(candidates, scaler, hist_last_xy=hist_last_xy)
    endpoints = cand_xy[:, :, -1, :]           # [N, M, 2]
    pre_endpoints = cand_xy[:, :, -2, :]       # [N, M, 2]
    # 退化兜底用的整段方向
    net_direction = cand_xy[:, :, -1, :] - cand_xy[:, :, 0, :]   # [N, M, 2]

    # ---- Step B: 扫描每个 window × 每个 k_seed 产生样本 ----
    n_samples = int(N) * samples_per_window

    out_candidates = np.empty((n_samples, M, Tout, D), dtype=np.float32)
    out_task = np.empty(n_samples, dtype=np.int8)
    out_type = np.empty(n_samples, dtype=np.int8)
    out_position = np.empty((n_samples, 3), dtype=np.float32)
    out_label = np.empty(n_samples, dtype=np.int8)
    out_k_seed = np.empty(n_samples, dtype=np.int8)  # 调试用，记录当时用的 k_seed
    # 0=forward, 1=backward, 2=side_left, 3=side_right
    out_position_dir = np.empty(n_samples, dtype=np.int8)
    out_soft_label = np.empty((n_samples, M), dtype=np.float32) if save_soft else None

    save_targets = targets is not None
    if save_targets:
        out_targets = np.empty((n_samples, Tout, D), dtype=np.float32)

    write_idx = 0
    for n in range(N):
        # 预先取出本 window 的 endpoints 和方向信息
        ep = endpoints[n]          # [M, 2]
        last_minus_prev = ep - pre_endpoints[n]  # [M, 2]
        net_dir = net_direction[n]               # [M, 2]

        for j in range(samples_per_window):
            k_seed = j % M

            # ---- 方向 & step_len ----
            direction = last_minus_prev[k_seed].astype(np.float64)
            step_len = float(np.linalg.norm(direction))
            if step_len < EPS:
                direction = net_dir[k_seed].astype(np.float64)
                step_len = float(np.linalg.norm(direction))
            if step_len < EPS:
                direction = rng.normal(size=2)
                step_len = float(np.linalg.norm(direction))
                if step_len < EPS:
                    direction = np.array([1.0, 0.0])
                    step_len = 1.0
            direction = direction / step_len

            # ---- 方向抽样 + 外推 + 噪声 ----
            ext = float(rng.uniform(ext_lo, ext_hi))
            r_dir = float(rng.random())
            if r_dir < p_fwd:
                # forward：保持原行为，pos = endpoint_k 沿候选末端方向外推
                pos_xy = ep[k_seed].astype(np.float64) + ext * step_len * direction
                pos_dir_id = 0
            else:
                # OOD 三个方向：以 origin 为中心，沿目标方向，距离 = |endpoint_k| + ext*step_len
                # 这样 4 个方向的 |position| 量级一致；forward 等价于沿原方向到该距离
                base_dist = float(np.linalg.norm(ep[k_seed]))
                total_dist = base_dist + ext * step_len
                if r_dir < p_fwd_bwd:
                    u_target = -direction
                    pos_dir_id = 1
                elif r_dir < p_fwd_bwd_sl:
                    u_target = np.array([-direction[1], direction[0]])   # 90° CCW
                    pos_dir_id = 2
                else:
                    u_target = np.array([direction[1], -direction[0]])   # 90° CW
                    pos_dir_id = 3
                pos_xy = total_dist * u_target
            pos_xy = pos_xy + rng.normal(0.0, sigma_km, size=2)

            # ---- label = argmin endpoint 距离 ----
            diffs = ep - pos_xy.reshape(1, 2)    # [M, 2]
            dists = np.linalg.norm(diffs, axis=-1)
            label = int(np.argmin(dists))

            # ---- soft label = softmax(-dist / tau) ----
            # 数值稳定：先减去最小值（等价变换）
            if save_soft:
                neg = -dists / soft_tau
                neg = neg - neg.max()
                exp_neg = np.exp(neg)
                soft = exp_neg / exp_neg.sum()       # [M]，和 = 1

            # ---- 上下文采样 ----
            type_i = int(rng.integers(type_lo, type_hi + 1))

            # ---- 写入 ----
            out_candidates[write_idx] = candidates[n]         # M, Tout, D
            out_task[write_idx] = task_type_fixed
            out_type[write_idx] = type_i
            out_position[write_idx, 0] = float(pos_xy[0])
            out_position[write_idx, 1] = float(pos_xy[1])
            out_position[write_idx, 2] = pos_z_default
            out_label[write_idx] = label
            out_k_seed[write_idx] = k_seed
            out_position_dir[write_idx] = pos_dir_id
            if save_soft:
                out_soft_label[write_idx] = soft
            if save_targets:
                out_targets[write_idx] = targets[n]
            write_idx += 1

    assert write_idx == n_samples
    res = {
        "candidates": out_candidates,
        "task_type": out_task,
        "type": out_type,
        "position": out_position,
        "label": out_label,
        "k_seed": out_k_seed,             # 调试/健康检查用
        "position_dir": out_position_dir, # 0=fwd, 1=bwd, 2=side_L, 3=side_R
    }
    if save_soft:
        res["soft_label"] = out_soft_label   # [N_samples, M]，训练用
    if save_targets:
        res["targets"] = out_targets     # [N_samples, Tout, D] GT，eval 用
    return res


# =============== 主函数 ===============

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GNN1 training samples from cached LSTM1 predictions.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"])
    args = parser.parse_args()

    gnn1_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    cfg = load_config(cfg_path)

    data_cfg = cfg.get("data", {})
    cache_dir = (gnn1_root / data_cfg.get("cache_dir", "data/cache")).resolve()
    raw_dir = (gnn1_root / data_cfg.get("raw_dir", "data/raw")).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = cache_dir / "scaler_posvel.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"找不到 scaler：{scaler_path}。"
            "请先跑 cache_lstm1_preds.py。"
        )
    scaler = _Scaler.load(scaler_path)

    seed = int(data_cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    for split in args.splits:
        cache_file = cache_dir / f"{split}.npz"
        if not cache_file.exists():
            print(f"[gen] cache 缺失跳过: {cache_file}")
            continue

        d = np.load(cache_file)
        history = d["history"]
        candidates = d["candidates"]
        targets = d["targets"] if "targets" in d.files else None
        print(f"[gen] {split}: history {history.shape}, candidates {candidates.shape}, "
              f"targets {None if targets is None else targets.shape}")

        out = generate_for_split(history, candidates, scaler, data_cfg, rng, targets=targets)

        out_path = raw_dir / f"{split}.npz"
        np.savez(out_path, **out)

        # ---- 健康检查 ----
        label = out["label"]
        k_seed = out["k_seed"]
        flip_ratio = float(np.mean(label != k_seed))
        unique, counts = np.unique(label, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"[gen] {split} → {out_path}")
        print(f"       samples   = {len(label)}")
        print(f"       label dist= {dist}")
        print(f"       P(label != k_seed) = {flip_ratio:.3f}   "
              f"(健康范围 0.10 ~ 0.30；>0.5 说明噪声太大，<0.05 说明太小)")

        # position 方向分布
        if "position_dir" in out:
            pdir = out["position_dir"]
            dir_names = {0: "forward", 1: "backward", 2: "side_left", 3: "side_right"}
            n_total = int(len(pdir))
            print(f"       position_dir dist:", end="  ")
            for did, name in dir_names.items():
                cnt = int((pdir == did).sum())
                print(f"{name}={cnt / max(1, n_total):.1%}", end="  ")
            print()
            # P(label != k_seed) 按方向分组（OOD 应该明显比 forward 高）
            print("       P(label!=k_seed) by direction:", end="  ")
            for did, name in dir_names.items():
                m = (pdir == did)
                if m.sum() == 0:
                    continue
                f = float(np.mean(label[m] != k_seed[m]))
                print(f"{name}={f:.2f}", end="  ")
            print()

        if "soft_label" in out:
            soft = out["soft_label"].astype(np.float64)
            M = soft.shape[-1]
            avg_max = float(soft.max(axis=-1).mean())
            srt = -np.sort(-soft, axis=-1)
            avg_top = srt.mean(axis=0)
            H = -np.sum(soft * np.log(soft + 1e-12), axis=-1)
            H_norm = float((H / np.log(M)).mean())
            top_str = "  ".join(f"top{k + 1}={avg_top[k]:.3f}" for k in range(min(3, M)))
            print(f"       soft_label: avg_max={avg_max:.3f}  H_norm={H_norm:.3f}  "
                  f"({top_str})")
            # 按方向分组打印 H_norm，OOD 方向应该明显更接近 1
            if "position_dir" in out:
                pdir = out["position_dir"]
                print("       soft_label H_norm by direction:", end="  ")
                for did, name in {0: "forward", 1: "backward",
                                  2: "side_left", 3: "side_right"}.items():
                    m = (pdir == did)
                    if m.sum() == 0:
                        continue
                    H_d = -np.sum(soft[m] * np.log(soft[m] + 1e-12), axis=-1)
                    print(f"{name}={float(H_d.mean() / np.log(M)):.3f}", end="  ")
                print()

    print("[gen] 完成。")


if __name__ == "__main__":
    main()
