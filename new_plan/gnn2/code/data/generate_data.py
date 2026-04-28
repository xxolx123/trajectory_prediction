#!/usr/bin/env python3
"""
gnn2/code/data/generate_data.py
-------------------------------
为 GNN2 离线生成训练数据：
  1. 加载 GNN1 ckpt，对 gnn1 三个 split (train/val/test) **分别**跑 forward 取 top-3
  2. 反归一化 candidates / targets 到物理 6D（hist_end=(0,0,0)）
  3. **池化**三个 split 的 (top_phys, top_probs, top_idx, target_phys) → 一个大场景池
  4. 按 max_scenes 上限随机抽样
  5. **重新切分**成 train/val/test（按 split_ratios，独立于 gnn1 的切分边界）
  6. 对每个 gnn2 split 做 **n_eta 倍扩张**：
       同一 (scene, top-K 内的 k) 配 n_eta 个不同 eta 值 → n_eta 条 gnn2 训练样本
  7. 按 R2 规则算 GT，写出 gnn2/data/raw/{split}.npz

业务语义（甲方对齐版）：
    我方反击装备从 position 发射，飞行 eta 秒后抵达打击点；
    敌方沿 pred_traj 移动；t=eta 时落在 pred_traj 上某一帧；
    GNN2 输出 "敌方届时所在的位置 (strike_pos) + 不确定圆 (radius) + 概率 (conf)"。

合成规则（R2）：
    cand_at_eta   = interp(top_phys[k], eta)
    target_at_eta = interp(target_phys, eta)
    gt_strike_pos    = cand_at_eta
    gt_strike_radius = clip(a + b * ||cand_at_eta - target_at_eta||, r_min, r_max)
    gt_strike_conf   = clip(1 - radius / r_max, 0, 1)

输出（gnn2/data/raw/{split}.npz，M = N_split * 3 * n_eta_per_sample）：
    pred_traj         [M, 10, 6]   float32   top-K 候选物理 6D 轨迹（hist_end=(0,0,0)）
    eta               [M]          int64     秒，∈ [eta_min_sec, eta_max_sec]
    gt_strike_pos     [M, 3]       float32   km xyz
    gt_strike_radius  [M]          float32   km
    gt_strike_conf    [M]          float32   ∈ [0, 1]
    # 诊断字段
    scene_idx         [M]          int32     场景在池中的全局索引（0 .. max_scenes-1）
    cand_k            [M]          int8      top-K 内的位置 0..K-1（按 GNN1 概率降序）
    eta_idx           [M]          int8      0..n_eta-1（同一 scene+k 的第几个 eta 副本）
    gnn1_top_idx      [M]          int8      0..M_lstm1-1，原 LSTM1 候选索引
    top_prob          [M]          float32   GNN1 给该候选的归一化 top-K 概率
    src_split         [M]          int8      场景源自哪个 gnn1 split（0=train, 1=val, 2=test）

用法（在 new_plan/gnn2/ 下）::

    cd new_plan/gnn2
    $env:PYTHONPATH = "$PWD/code;$PWD/.."

    # 默认：池化所有 gnn1 split，按 max_scenes 抽样，8:1:1 重切分
    python -m data.generate_data --config config.yaml

    # 改 max_scenes / n_eta_per_sample 直接改 config.yaml 里的 data 段
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm


# ============================================================
# 路径 setup（让 import gnn1.* 可用）
# ============================================================

_THIS_FILE = Path(__file__).resolve()
_GNN2_ROOT = _THIS_FILE.parents[2]                  # .../new_plan/gnn2
_REPO_ROOT = _GNN2_ROOT.parent                       # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_GNN2_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(_GNN2_ROOT / "code"))

from gnn1.code.train.model import build_model_from_config as build_gnn1_model    # noqa: E402


# 切分名 → src_split 编号
SRC_SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
GNN2_SPLIT_NAMES = ("train", "val", "test")


# ============================================================
# 工具
# ============================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_rel(rel: str, base: Path) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def find_latest_ckpt(p: Path) -> Optional[Path]:
    if p is None:
        return None
    if p.is_file():
        return p
    if p.is_dir():
        cands = list(p.rglob("*.pt"))
        if not cands:
            return None
        cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]
    return None


def setup_device(s: str) -> torch.device:
    s = (s or "auto").lower()
    if s == "cpu":
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Scaler
# ============================================================

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


# ============================================================
# 解码：归一化+delta → 物理 km / km·s⁻¹（hist_end 平移到 (0,0,0)）
# ============================================================

def decode_future_to_phys(
    fut_norm: np.ndarray,            # [..., T, 6] norm+delta
    scaler: _Scaler,
) -> np.ndarray:
    """
    把候选 / GT 未来轨迹（norm+delta）解码到物理空间。
    hist 末帧 = (0, 0, 0) 约定，所以 cumsum 自然就是物理 xyz。

    返回 [..., T, 6] float32。
    """
    feat = scaler.inverse_transform(fut_norm.astype(np.float64))
    pos_delta = feat[..., :3]
    vel = feat[..., 3:6]
    pos_phys = np.cumsum(pos_delta, axis=-2)
    return np.concatenate([pos_phys, vel], axis=-1).astype(np.float32)


# ============================================================
# 时间索引插值：traj 在 t=eta 处的 xyz
# ============================================================

def interp_xyz_at_eta(
    traj_xyz: np.ndarray,            # [..., T, 3]   物理 xyz，hist_end 已对齐到原点
    eta_sec: np.ndarray,             # [...]         秒，∈ [0, T*time_step_s]
    time_step_s: float,
) -> np.ndarray:
    """
    线性插值返回 [..., 3]。

    时间约定：traj_xyz[t] = hist_end 后 (t+1)*time_step_s 秒处的位置；
    因此把 origin 拼到首位组成 [..., T+1, 3]，eta=0 对应 ext[0]、eta=T*time_step_s 对应 ext[T]。
    """
    if traj_xyz.shape[-1] != 3:
        raise ValueError(f"traj_xyz 最后一维必须是 3，实际 {traj_xyz.shape}")
    T = traj_xyz.shape[-2]
    leading = traj_xyz.shape[:-2]
    if eta_sec.shape != leading:
        raise ValueError(
            f"eta_sec.shape {eta_sec.shape} 必须等于 traj_xyz.shape[:-2] {leading}"
        )

    origin = np.zeros((*leading, 1, 3), dtype=traj_xyz.dtype)
    ext = np.concatenate([origin, traj_xyz], axis=-2)            # [..., T+1, 3]

    sf = np.clip(
        eta_sec.astype(np.float64) / float(time_step_s), 0.0, float(T),
    )                                                            # [...] ∈ [0, T]
    lo = np.clip(np.floor(sf).astype(np.int64), 0, T)
    hi = np.clip(lo + 1, 0, T)
    alpha = sf - lo                                              # [...]

    lo_idx = np.broadcast_to(lo[..., None, None], (*leading, 1, 3))
    hi_idx = np.broadcast_to(hi[..., None, None], (*leading, 1, 3))
    pos_lo = np.take_along_axis(ext, lo_idx, axis=-2).squeeze(-2)
    pos_hi = np.take_along_axis(ext, hi_idx, axis=-2).squeeze(-2)

    return ((1.0 - alpha[..., None]) * pos_lo + alpha[..., None] * pos_hi).astype(
        np.float32
    )


# ============================================================
# GNN1 推理（批量）
# ============================================================

def run_gnn1_topk(
    model: torch.nn.Module,
    candidates_norm: np.ndarray,     # [N_s, M, T, 6] norm+delta
    task_type: np.ndarray,           # [N_s]
    type_id: np.ndarray,             # [N_s]
    position: np.ndarray,            # [N_s, 3]
    device: torch.device,
    batch_size: int = 512,
    desc: str = "GNN1 forward",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    N = int(candidates_norm.shape[0])
    out_idx: List[np.ndarray] = []
    out_probs: List[np.ndarray] = []

    pbar = tqdm(total=N, desc=desc, unit="samp", ncols=100, smoothing=0.1)
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
            out_idx.append(out["top_idx"].detach().cpu().numpy().astype(np.int64))
            out_probs.append(out["top_probs"].detach().cpu().numpy().astype(np.float32))
            pbar.update(j - i)
    pbar.close()

    return np.concatenate(out_idx, axis=0), np.concatenate(out_probs, axis=0)


# ============================================================
# 阶段 1：处理一个 gnn1 split，得到该 split 的 top-K 物理 + GT 物理
# ============================================================

class _SceneShard:
    """单个 gnn1 split 处理后的数据片，准备池化。"""

    def __init__(
        self,
        top_phys: np.ndarray,        # [N_s, K, 10, 6]
        top_probs: np.ndarray,       # [N_s, K]
        top_idx: np.ndarray,         # [N_s, K]
        target_phys: np.ndarray,     # [N_s, 10, 6]
        src_split_id: int,
    ) -> None:
        self.top_phys = top_phys.astype(np.float32)
        self.top_probs = top_probs.astype(np.float32)
        self.top_idx = top_idx.astype(np.int64)
        self.target_phys = target_phys.astype(np.float32)
        self.src_split_id = int(src_split_id)
        self.N_s = int(top_phys.shape[0])


def process_gnn1_split_to_shard(
    split: str,
    gnn1_model: torch.nn.Module,
    scaler: _Scaler,
    device: torch.device,
    cache_dir: Path,
    raw_dir_in: Path,
    gnn1_batch: int,
    keep_indices: Optional[np.ndarray] = None,
) -> Optional[_SceneShard]:
    """
    加载某个 gnn1 split 的 raw + cache，跑 GNN1 forward，返回 _SceneShard。

    keep_indices: 若不为 None，只处理 raw[keep_indices] 这部分样本（**先抽样再 forward**），
                  这样 GNN1 推理量 / decode 量都按 keep_indices 缩到对应规模。
    """
    cache_path = cache_dir / f"{split}.npz"
    raw_path = raw_dir_in / f"{split}.npz"
    if not cache_path.exists():
        print(f"[gnn2/gen] {split}: 缺 cache {cache_path}，跳过")
        return None
    if not raw_path.exists():
        print(f"[gnn2/gen] {split}: 缺 gnn1 raw {raw_path}，跳过")
        return None

    print(f"[gnn2/gen] === [phase 1] gnn1 split: {split} ===")
    cache_d = np.load(cache_path)
    raw_d = np.load(raw_path)

    if "targets" not in cache_d.files:
        raise RuntimeError(
            f"{cache_path} 中无 'targets' 字段；请用最新 cache_lstm1_preds.py 重生成 cache。"
        )
    targets_w = cache_d["targets"]                              # [N_w, 10, 6] norm+delta

    raw_cand_full = raw_d["candidates"]                         # [N_s_full, M, 10, 6] norm+delta
    raw_task_full = raw_d["task_type"].astype(np.int64)
    raw_type_full = raw_d["type"].astype(np.int64)
    raw_pos_full = raw_d["position"].astype(np.float32)

    N_w = int(targets_w.shape[0])
    N_s_full = int(raw_cand_full.shape[0])
    if N_s_full % N_w != 0:
        raise RuntimeError(
            f"{split}: raw 样本数 {N_s_full} 不能整除 cache window 数 {N_w}"
        )
    spw = N_s_full // N_w

    # ---- 先抽样再 forward：把 GNN1 推理 / decode 都按 keep_indices 缩到对应规模 ----
    if keep_indices is not None and keep_indices.size < N_s_full:
        sub_idx = np.asarray(keep_indices, dtype=np.int64)
        if sub_idx.size == 0:
            print(f"  N_w={N_w}, N_s=0 (sampled from {N_s_full})，跳过")
            return None
        # 加 sort 让索引访问 cache-friendly（也利于后面的 win_idx 复刻）
        sub_idx_sorted = np.sort(sub_idx)
        raw_cand = raw_cand_full[sub_idx_sorted]
        raw_task = raw_task_full[sub_idx_sorted]
        raw_type = raw_type_full[sub_idx_sorted]
        raw_pos = raw_pos_full[sub_idx_sorted]
        win_idx = sub_idx_sorted // spw
        print(
            f"  N_w={N_w}, N_s={raw_cand.shape[0]} "
            f"(sampled from {N_s_full}, ratio={raw_cand.shape[0] / N_s_full:.3f}), "
            f"samples_per_window={spw}"
        )
    else:
        raw_cand = raw_cand_full
        raw_task = raw_task_full
        raw_type = raw_type_full
        raw_pos = raw_pos_full
        win_idx = np.arange(N_s_full) // spw
        print(f"  N_w={N_w}, N_s={N_s_full} (全量), samples_per_window={spw}")

    # 释放原始大数组（已经把 subset 拷出来了）
    del raw_cand_full, raw_task_full, raw_type_full, raw_pos_full
    N_s = int(raw_cand.shape[0])

    # GNN1 forward（只跑 subset）
    t0 = time.time()
    top_idx, top_probs = run_gnn1_topk(
        gnn1_model,
        candidates_norm=raw_cand,
        task_type=raw_task,
        type_id=raw_type,
        position=raw_pos,
        device=device,
        batch_size=gnn1_batch,
        desc=f"  GNN1 {split}",
    )                                                            # [N_s, K], [N_s, K]
    K = int(top_idx.shape[-1])
    print(f"  GNN1 forward: K={K}, {time.time() - t0:.1f}s on {N_s} samples")

    # 反归一化候选（subset） + gather top-K
    cand_phys = decode_future_to_phys(raw_cand, scaler)         # [N_s, M, 10, 6]
    Tout = int(cand_phys.shape[2])
    Df = int(cand_phys.shape[3])
    top_phys = np.take_along_axis(
        cand_phys,
        top_idx[:, :, None, None].repeat(Tout, axis=2).repeat(Df, axis=3),
        axis=1,
    )                                                            # [N_s, K, 10, 6]
    del cand_phys, raw_cand

    # 反归一化 GT 未来 → per-sample（按 win_idx 复刻；只保留 subset 对应的 window）
    target_phys_w = decode_future_to_phys(targets_w, scaler)    # [N_w, 10, 6]
    target_phys = target_phys_w[win_idx]                         # [N_s, 10, 6]
    del target_phys_w

    return _SceneShard(
        top_phys=top_phys,
        top_probs=top_probs,
        top_idx=top_idx,
        target_phys=target_phys,
        src_split_id=SRC_SPLIT_TO_ID[split],
    )


# ============================================================
# 阶段 2：池化 + 上限抽样 + 8:1:1 切分
# ============================================================

def pool_and_split(
    shards: List[_SceneShard],
    split_ratios: List[float],
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """
    1) 拼接 shards（每个 shard 已经在 phase 1 按 keep_indices 抽过样了）
    2) 全局 shuffle + 按 split_ratios 切成 train/val/test

    Returns:
        pool: dict {top_phys, top_probs, top_idx, target_phys, src_split}
        scene_idx_all: [N_pool] int32  scene 在池里的索引（诊断用）
        split_indices: dict {train, val, test} → 各自在 pool 里的 indices
    """
    if not shards:
        raise RuntimeError("没有可用的 gnn1 shard")
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios 之和必须为 1，实际 {sum(split_ratios)}")
    if len(split_ratios) != 3:
        raise ValueError(f"split_ratios 必须是 3 元（train/val/test），实际 {split_ratios}")

    print(f"\n[gnn2/gen] === [phase 2] 池化 + 切分 ===")

    # 1) 拼接
    top_phys = np.concatenate([s.top_phys for s in shards], axis=0)
    top_probs = np.concatenate([s.top_probs for s in shards], axis=0)
    top_idx = np.concatenate([s.top_idx for s in shards], axis=0)
    target_phys = np.concatenate([s.target_phys for s in shards], axis=0)
    src_split = np.concatenate(
        [np.full(s.N_s, s.src_split_id, dtype=np.int8) for s in shards], axis=0
    )
    N_pool = int(top_phys.shape[0])
    src_dist = ", ".join(
        f"{['train', 'val', 'test'][s.src_split_id]}={s.N_s}" for s in shards
    )
    print(f"  pool size = {N_pool} scenes (来源: {src_dist})")

    # 2) 全局 shuffle + 切 train/val/test
    perm = rng.permutation(N_pool)
    n_train = int(round(N_pool * split_ratios[0]))
    n_val = int(round(N_pool * split_ratios[1]))
    n_test = N_pool - n_train - n_val                            # 剩下都给 test，避免舍入丢样本
    if n_test < 0:
        raise RuntimeError(
            f"split_ratios 计算后 n_test={n_test} < 0；检查比例"
        )
    print(f"  split sizes: train={n_train}, val={n_val}, test={n_test}")

    split_indices = {
        "train": perm[:n_train],
        "val":   perm[n_train: n_train + n_val],
        "test":  perm[n_train + n_val:],
    }

    pool = {
        "top_phys":   top_phys,
        "top_probs":  top_probs,
        "top_idx":    top_idx,
        "target_phys": target_phys,
        "src_split":  src_split,
    }
    scene_idx_all = np.arange(N_pool, dtype=np.int32)

    # src_split 在每个 gnn2 split 里的分布（健康检查用）
    print(f"  src_split distribution per gnn2 split:")
    for split_name in GNN2_SPLIT_NAMES:
        idx = split_indices[split_name]
        ss = src_split[idx]
        cnt = np.bincount(ss.astype(np.int64), minlength=3)
        print(
            f"    {split_name:>5}  n={len(idx):7d}  "
            f"from gnn1: train={int(cnt[0])}, val={int(cnt[1])}, test={int(cnt[2])}"
        )

    return pool, scene_idx_all, split_indices


def compute_per_split_keep_indices(
    raw_dir_in: Path,
    splits: List[str],
    max_scenes: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, int]]:
    """
    在不加载 candidates 的前提下，先按 max_scenes 等比例从每个 gnn1 split 抽样。

    用 task_type [N_s] (int8) 这个最小字段读 N_s_full，避免触发 candidates 大数组的解压。

    Returns:
        keep_per_split: {split → indices array (sorted) 或 None（None = 该 split 取全量）}
        n_full_per_split: {split → N_s_full}
    """
    # 1) 读 N_s_per_split（只读 task_type 字段，开销极小）
    n_full_per_split: Dict[str, int] = {}
    for split in splits:
        raw_path = raw_dir_in / f"{split}.npz"
        if not raw_path.exists():
            continue
        with np.load(raw_path) as f:
            if "task_type" not in f.files:
                # 退化方案：读 candidates 第一维（昂贵）
                n_full_per_split[split] = int(f["candidates"].shape[0])
            else:
                n_full_per_split[split] = int(f["task_type"].shape[0])
    if not n_full_per_split:
        raise FileNotFoundError(f"在 {raw_dir_in} 里找不到任何 split 的 raw npz")

    N_total_full = sum(n_full_per_split.values())
    print(
        f"[gnn2/gen] gnn1 raw pool: total {N_total_full} = "
        + ", ".join(f"{k}={v}" for k, v in n_full_per_split.items())
    )

    # 2) 计算抽样
    if max_scenes <= 0 or max_scenes >= N_total_full:
        print(
            f"[gnn2/gen] max_scenes={max_scenes}，不限上限（每个 split 用全量）"
        )
        return {s: None for s in n_full_per_split}, n_full_per_split

    # 把 [0, N_total_full) 看作连接的全局索引空间，sample 出 max_scenes 个
    # 然后按 split 边界拆回 per-split 局部索引；这样自然得到等比例抽样。
    global_idx = rng.choice(N_total_full, size=max_scenes, replace=False)
    global_idx.sort()

    keep_per_split: Dict[str, Optional[np.ndarray]] = {}
    cum = 0
    for split in splits:
        if split not in n_full_per_split:
            continue
        n = n_full_per_split[split]
        mask = (global_idx >= cum) & (global_idx < cum + n)
        local = (global_idx[mask] - cum).astype(np.int64)
        keep_per_split[split] = local
        cum += n

    print(
        f"[gnn2/gen] proportional subsample (max_scenes={max_scenes}): "
        + ", ".join(
            f"{k}={(0 if v is None else v.size)}" for k, v in keep_per_split.items()
        )
    )
    return keep_per_split, n_full_per_split


# ============================================================
# 阶段 3：n_eta 扩张 + 计算 GT + save
# ============================================================

def expand_and_save(
    split_name: str,
    pool: Dict[str, np.ndarray],
    scene_idx_all: np.ndarray,
    indices: np.ndarray,
    n_eta: int,
    eta_min_sec: int,
    eta_max_sec: int,
    time_step_s: float,
    radius_a: float,
    radius_b: float,
    radius_min: float,
    radius_max: float,
    out_dir: Path,
    rng: np.random.Generator,
) -> None:
    print(f"\n[gnn2/gen] === [phase 3] expand+save: {split_name} ===")
    if indices.size == 0:
        print(f"  {split_name}: 0 scenes，跳过保存")
        return

    # 抽这个 gnn2 split 对应的 scene 子集
    top_phys = pool["top_phys"][indices]                         # [N, K, 10, 6]
    top_probs = pool["top_probs"][indices]                       # [N, K]
    top_idx = pool["top_idx"][indices]                           # [N, K]
    target_phys = pool["target_phys"][indices]                   # [N, 10, 6]
    src_split = pool["src_split"][indices]                       # [N]
    scene_idx = scene_idx_all[indices]                           # [N]

    N = int(top_phys.shape[0])
    K = int(top_phys.shape[1])
    Tout = int(top_phys.shape[2])
    Df = int(top_phys.shape[3])

    # 每条 scene × 每条 top-K 候选 × n_eta 个 eta
    # 把扩张后的张量索引设计成：[N, K, n_eta, ...]，最后 flatten 到 [M = N*K*n_eta, ...]
    eta_arr = rng.integers(
        eta_min_sec, eta_max_sec + 1, size=(N, K, n_eta), dtype=np.int64,
    )                                                            # [N, K, n_eta]

    # cand_at_eta：top_phys[..., :3] 在 eta 处插值
    # top_phys 的 K 维与 eta 的 K 维对齐；扩到 n_eta 维需要广播插值函数支持
    # 拆开做：把 top_phys 在 K 后面新增 n_eta 维（重复），与 eta 形状 [N, K, n_eta] 对齐
    top_xyz = top_phys[..., :3]                                  # [N, K, T, 3]
    top_xyz_e = np.broadcast_to(
        top_xyz[:, :, None, :, :], (N, K, n_eta, Tout, 3),
    ).reshape(N * K * n_eta, Tout, 3)
    eta_flat = eta_arr.reshape(N * K * n_eta)                    # [N*K*n_eta]
    cand_at_eta_flat = interp_xyz_at_eta(
        top_xyz_e, eta_flat, time_step_s,
    )                                                            # [N*K*n_eta, 3]

    # target_at_eta：target_phys 没有 K 维，先扩到 K，再扩到 n_eta
    tgt_xyz = target_phys[..., :3]                               # [N, T, 3]
    tgt_xyz_e = np.broadcast_to(
        tgt_xyz[:, None, None, :, :], (N, K, n_eta, Tout, 3),
    ).reshape(N * K * n_eta, Tout, 3)
    target_at_eta_flat = interp_xyz_at_eta(
        tgt_xyz_e, eta_flat, time_step_s,
    )                                                            # [N*K*n_eta, 3]

    # radius / conf
    diff = cand_at_eta_flat.astype(np.float64) - target_at_eta_flat.astype(np.float64)
    dist = np.linalg.norm(diff, axis=-1)                         # [M]
    radius = np.clip(radius_a + radius_b * dist, radius_min, radius_max).astype(
        np.float32,
    )
    conf = np.clip(1.0 - radius / radius_max, 0.0, 1.0).astype(np.float32)

    # pred_traj: [N, K, 10, 6] 复制到 [N, K, n_eta, 10, 6] → flatten
    pred_traj_out = (
        np.broadcast_to(
            top_phys[:, :, None, :, :], (N, K, n_eta, Tout, Df),
        )
        .reshape(N * K * n_eta, Tout, Df)
        .astype(np.float32)
        .copy()
    )
    eta_out = eta_flat.astype(np.int64)
    gt_pos_out = cand_at_eta_flat.astype(np.float32)
    gt_radius_out = radius.astype(np.float32)
    gt_conf_out = conf.astype(np.float32)

    # 诊断字段
    M_out = N * K * n_eta
    scene_idx_out = np.broadcast_to(
        scene_idx[:, None, None], (N, K, n_eta),
    ).reshape(M_out).astype(np.int32)
    cand_k_out = np.broadcast_to(
        np.arange(K, dtype=np.int8)[None, :, None], (N, K, n_eta),
    ).reshape(M_out).astype(np.int8)
    eta_idx_out = np.broadcast_to(
        np.arange(n_eta, dtype=np.int8)[None, None, :], (N, K, n_eta),
    ).reshape(M_out).astype(np.int8)
    gnn1_top_idx_out = np.broadcast_to(
        top_idx[:, :, None], (N, K, n_eta),
    ).reshape(M_out).astype(np.int8)
    top_prob_out = np.broadcast_to(
        top_probs[:, :, None], (N, K, n_eta),
    ).reshape(M_out).astype(np.float32)
    src_split_out = np.broadcast_to(
        src_split[:, None, None], (N, K, n_eta),
    ).reshape(M_out).astype(np.int8)

    out: Dict[str, np.ndarray] = {
        "pred_traj":         pred_traj_out,
        "eta":               eta_out,
        "gt_strike_pos":     gt_pos_out,
        "gt_strike_radius":  gt_radius_out,
        "gt_strike_conf":    gt_conf_out,
        "scene_idx":         scene_idx_out,
        "cand_k":            cand_k_out,
        "eta_idx":           eta_idx_out,
        "gnn1_top_idx":      gnn1_top_idx_out,
        "top_prob":          top_prob_out,
        "src_split":         src_split_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}.npz"
    np.savez(out_path, **out)
    print(
        f"  saved → {out_path}  "
        f"({M_out} samples = {N} scenes × K={K} × n_eta={n_eta})"
    )

    print_split_summary(split_name, out, eta_max_sec=eta_max_sec, radius_max_km=radius_max)


def print_split_summary(
    split: str,
    out: Dict[str, np.ndarray],
    eta_max_sec: int,
    radius_max_km: float,
) -> None:
    print(f"  summary {split}:")
    M = int(out["pred_traj"].shape[0])
    print(f"    samples = {M}")

    eta = out["eta"]
    radius = out["gt_strike_radius"]
    conf = out["gt_strike_conf"]
    cand_k = out["cand_k"]
    top_prob = out["top_prob"]

    print(
        f"    eta      ∈ [{int(eta.min())}, {int(eta.max())}] s   "
        f"mean={float(eta.mean()):.1f}"
    )
    print(
        f"    radius   ∈ [{float(radius.min()):.3f}, {float(radius.max()):.3f}] km  "
        f"mean={float(radius.mean()):.3f}"
    )
    print(
        f"    conf     ∈ [{float(conf.min()):.3f}, {float(conf.max()):.3f}]  "
        f"mean={float(conf.mean()):.3f}"
    )
    print(
        f"    top_prob ∈ [{float(top_prob.min()):.3f}, {float(top_prob.max()):.3f}]  "
        f"mean={float(top_prob.mean()):.3f}  (top-K 重归一化和应 ≈ 1)"
    )

    # 按 cand_k 分组
    print("    by cand_k:")
    for k in sorted(set(cand_k.tolist())):
        m = (cand_k == k)
        print(
            f"      k={int(k)}  n={int(m.sum())}  "
            f"radius_mean={float(radius[m].mean()):.3f}  "
            f"conf_mean={float(conf[m].mean()):.3f}  "
            f"top_prob_mean={float(top_prob[m].mean()):.3f}"
        )

    # 按 eta 分桶
    eta_bins = [0, 60, 120, 240, 360, 480, eta_max_sec + 1]
    print(f"    by eta bin:")
    for i in range(len(eta_bins) - 1):
        lo, hi = eta_bins[i], eta_bins[i + 1]
        m = (eta >= lo) & (eta < hi)
        if m.sum() == 0:
            continue
        print(
            f"      [{lo:>3d}, {hi:>3d})  n={int(m.sum()):7d}  "
            f"radius_mean={float(radius[m].mean()):6.3f}  "
            f"conf_mean={float(conf[m].mean()):.3f}"
        )

    n_clip_max = int((radius >= radius_max_km - 1e-3).sum())
    print(
        f"    radius@r_max: {n_clip_max} samples ({n_clip_max / max(1, M) * 100:.1f}%)"
    )


# ============================================================
# main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GNN2 training data: pool gnn1 splits, cap by max_scenes, "
                    "8:1:1 re-split, n_eta-fold expansion, R2 strike GT."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gnn1-batch", type=int, default=0)
    parser.add_argument("--gnn1-ckpt", type=str, default="")
    parser.add_argument("--max-scenes", type=int, default=-1,
                        help="覆盖 config.data.max_scenes；-1 = 用 config 值")
    parser.add_argument("--n-eta", type=int, default=-1,
                        help="覆盖 config.data.n_eta_per_sample；-1 = 用 config 值")
    args = parser.parse_args()

    # ---- 路径 ----
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_GNN2_ROOT / cfg_path).resolve()
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {}) or {}

    cache_dir = resolve_rel(
        data_cfg.get("gnn1_cache_dir", "../gnn1/data/cache"), _GNN2_ROOT,
    )
    raw_dir_in = resolve_rel(
        data_cfg.get("gnn1_raw_dir", "../gnn1/data/raw"), _GNN2_ROOT,
    )
    gnn1_cfg_path = resolve_rel(
        data_cfg.get("gnn1_config", "../gnn1/config.yaml"), _GNN2_ROOT,
    )
    gnn1_ckpt_arg = args.gnn1_ckpt or data_cfg.get("gnn1_ckpt", "../gnn1/checkpoints")
    gnn1_ckpt_path = resolve_rel(gnn1_ckpt_arg, _GNN2_ROOT)
    out_dir = resolve_rel(data_cfg.get("raw_dir", "data/raw"), _GNN2_ROOT)

    scaler_candidates = [
        cache_dir / "scaler_posvel.npz",
        resolve_rel(
            data_cfg.get("lstm1_scaler", "../lstm1/data/processed/scaler_posvel.npz"),
            _GNN2_ROOT,
        ),
    ]
    scaler_path: Optional[Path] = None
    for sp in scaler_candidates:
        if sp.exists():
            scaler_path = sp
            break
    if scaler_path is None:
        raise FileNotFoundError(
            f"找不到 scaler_posvel.npz；尝试过：{[str(p) for p in scaler_candidates]}"
        )

    print(f"[gnn2/gen] cfg                = {cfg_path}")
    print(f"[gnn2/gen] gnn1 cache_dir     = {cache_dir}")
    print(f"[gnn2/gen] gnn1 raw_dir       = {raw_dir_in}")
    print(f"[gnn2/gen] gnn1 config        = {gnn1_cfg_path}")
    print(f"[gnn2/gen] gnn1 ckpt          = {gnn1_ckpt_path}")
    print(f"[gnn2/gen] scaler             = {scaler_path}")
    print(f"[gnn2/gen] out_dir            = {out_dir}")

    if not cache_dir.exists():
        raise FileNotFoundError(f"找不到 gnn1 cache 目录：{cache_dir}")
    if not raw_dir_in.exists():
        raise FileNotFoundError(f"找不到 gnn1 raw 目录：{raw_dir_in}")
    if not gnn1_cfg_path.exists():
        raise FileNotFoundError(f"找不到 gnn1 config：{gnn1_cfg_path}")

    # ---- 设备 + GNN1 ----
    device_str = args.device
    if device_str == "auto":
        device_str = str(data_cfg.get("gnn1_device", "auto"))
    device = setup_device(device_str)
    print(f"[gnn2/gen] device             = {device}")

    gnn1_cfg = load_yaml(gnn1_cfg_path)
    gnn1_model = build_gnn1_model(gnn1_cfg).to(device)
    gnn1_ckpt_file = find_latest_ckpt(gnn1_ckpt_path)
    if gnn1_ckpt_file is None:
        raise FileNotFoundError(
            f"找不到 GNN1 ckpt（{gnn1_ckpt_path}）；请先训练 GNN1 或在 config 里指定。"
        )
    print(f"[gnn2/gen] GNN1 ckpt picked   = {gnn1_ckpt_file}")
    sd = torch.load(gnn1_ckpt_file, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    gnn1_model.load_state_dict(sd, strict=True)
    gnn1_model.eval()

    scaler = _Scaler.load(scaler_path)
    print(f"[gnn2/gen] scaler loaded      = mean.shape={scaler.mean.shape}")

    # ---- 配置参数 ----
    seed = int(data_cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    gnn1_batch = int(args.gnn1_batch) or int(data_cfg.get("gnn1_batch_size", 512))

    max_scenes = int(args.max_scenes if args.max_scenes >= 0 else data_cfg.get("max_scenes", 0))
    n_eta = int(args.n_eta if args.n_eta > 0 else data_cfg.get("n_eta_per_sample", 5))
    if n_eta <= 0:
        raise ValueError(f"n_eta_per_sample 必须 >= 1，实际 {n_eta}")
    split_ratios = list(data_cfg.get("split_ratios", [0.8, 0.1, 0.1]))

    eta_min_sec = int(data_cfg.get("eta_min_sec", 0))
    eta_max_sec = int(data_cfg.get("eta_max_sec", 600))
    time_step_s = float(data_cfg.get("time_step_s", 60.0))
    fut_len = int(data_cfg.get("fut_len_steps", 10))
    if eta_min_sec < 0 or eta_max_sec > int(fut_len * time_step_s):
        raise ValueError(
            f"eta_range [{eta_min_sec}, {eta_max_sec}] 必须 ⊂ "
            f"[0, fut_len*time_step_s={int(fut_len * time_step_s)}]"
        )

    radius_a = float(data_cfg.get("radius_a_km", 0.0))
    radius_b = float(data_cfg.get("radius_b", 0.5))
    radius_min = float(data_cfg.get("radius_min_km", 0.5))
    radius_max = float(data_cfg.get("radius_max_km", 10.0))
    if radius_max <= radius_min:
        raise ValueError(f"radius_max_km({radius_max}) 必须 > radius_min_km({radius_min})")

    print(
        f"[gnn2/gen] params: max_scenes={max_scenes}, n_eta={n_eta}, "
        f"split_ratios={split_ratios}, eta=[{eta_min_sec}, {eta_max_sec}]s, "
        f"time_step={time_step_s}s, radius=clip({radius_a}+{radius_b}*dist, "
        f"{radius_min}, {radius_max}) km"
    )

    # ---- 阶段 0：在加载 candidates 之前先确定每个 split 抽哪些 idx ----
    # 这是关键优化：让 GNN1 forward / decode 都只跑这个 subset，
    # 避免在 max_scenes 远小于全量时白白处理几百万条没用的样本。
    splits_order = ["train", "val", "test"]
    keep_per_split, n_full_per_split = compute_per_split_keep_indices(
        raw_dir_in=raw_dir_in,
        splits=splits_order,
        max_scenes=max_scenes,
        rng=rng,
    )

    # ---- 阶段 1：处理每个 gnn1 split（只对 keep_indices 跑 forward + decode）----
    shards: List[_SceneShard] = []
    for split in splits_order:
        if split not in n_full_per_split:
            continue
        shard = process_gnn1_split_to_shard(
            split=split,
            gnn1_model=gnn1_model,
            scaler=scaler,
            device=device,
            cache_dir=cache_dir,
            raw_dir_in=raw_dir_in,
            gnn1_batch=gnn1_batch,
            keep_indices=keep_per_split.get(split),
        )
        if shard is not None:
            shards.append(shard)

    # ---- 阶段 2：池化 + 切分（不再二次抽样，phase 0 已抽过）----
    pool, scene_idx_all, split_indices = pool_and_split(
        shards=shards,
        split_ratios=split_ratios,
        rng=rng,
    )

    # ---- 阶段 3：每个 gnn2 split 扩张并保存 ----
    for split_name in GNN2_SPLIT_NAMES:
        expand_and_save(
            split_name=split_name,
            pool=pool,
            scene_idx_all=scene_idx_all,
            indices=split_indices[split_name],
            n_eta=n_eta,
            eta_min_sec=eta_min_sec,
            eta_max_sec=eta_max_sec,
            time_step_s=time_step_s,
            radius_a=radius_a,
            radius_b=radius_b,
            radius_min=radius_min,
            radius_max=radius_max,
            out_dir=out_dir,
            rng=rng,
        )

    print("\n[gnn2/gen] all splits done.")


if __name__ == "__main__":
    main()
