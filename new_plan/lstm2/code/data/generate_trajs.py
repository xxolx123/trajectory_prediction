#!/usr/bin/env python3
"""
lstm2/code/data/generate_trajs.py
---------------------------------
为 LSTM2 离线生成"历史 + 道路约束后预测轨迹 → 意图/威胁度"训练数据。

数据流：
    gnn1/data/cache/{split}.npz   →  history [N_w, 20, 6] (norm+delta)
                                     candidates [N_w, 5, 10, 6] (norm+delta)
                                     targets [N_w, 10, 6]      (norm+delta，可选)
    gnn1/data/raw/{split}.npz     →  candidates [N_s, 5, 10, 6]   (复刻自 cache)
                                     task_type / type / position [N_s, ...]
                                     soft_label / label / k_seed   (调试用)
    gnn1/checkpoints/.../*.pt     →  载入 GNN1 跑 top-K=3 选轨
    lstm1/data/processed/scaler   →  反归一化到物理 km / km/s

    每个 raw_idx：
        win_idx = raw_idx // (N_s / N_w)
        hist_phys = decode(history[win_idx])  # hist 末帧 = (0,0,0)
        cand_phys = decode(candidates_raw[raw_idx])
        top_idx, top_probs = GNN1({cand_trajs, task_type, type, position})
        top_phys = gather(cand_phys, top_idx)         # [3, 10, 6]
        name, road = synth_roads.random_road_topology(top_phys, rng)
        refined = ConstraintOptimizer(top_phys, ctx_with_road)  # [3, 10, 6]
        for k in 0..2:
            (intent_k, threat_k) = labels.compute_intent_threat(refined[k], position)

输出：lstm2/data/raw/{split}.npz，字段（M = N_s * K=3）：
    hist_raw     [M, 20, 6]   float32   物理 km / km/s（hist 末帧 = (0,0,0)）
    fut_refined  [M, 10, 6]   float32   物理（路网约束后）
    fut_gt       [M, 10, 6]   float32   物理（GT 未来；若 cache 有 targets 则有）
    position     [M, 3]       float32   我方固定目标 km
    task_type    [M]          int8
    type         [M]          int8
    intent_label [M]          int8      0..3
    threat_score [M]          int16     0..100
    topology     [M]          str       路网拓扑名（调试用）
    top_prob     [M]          float32   GNN1 给该候选的归一化 top-K 概率
    sample_idx   [M]          int32     原 gnn1 raw 的 sample 索引
    cand_k       [M]          int8      在 top-K 里的位置 0..K-1
    gnn1_top_idx [M]          int8      在 0..M_lstm1-1 里的索引

用法（在 new_plan/lstm2 下）：
    cd new_plan/lstm2
    $env:PYTHONPATH = "$PWD/code;$PWD/.."        # 同时挂 lstm2/code 和 new_plan
    python -m data.generate_trajs --config config.yaml --splits val
    python -m data.generate_trajs --config config.yaml --splits train val test
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm


# ============================================================
# 把 new_plan/ 加到 sys.path，方便 import gnn1.* / constraint_optimizer.*
# ============================================================

_THIS_FILE = Path(__file__).resolve()
_LSTM2_ROOT = _THIS_FILE.parents[2]                  # .../new_plan/lstm2
_REPO_ROOT = _LSTM2_ROOT.parent                       # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_LSTM2_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(_LSTM2_ROOT / "code"))

# 项目内导入
from common.context_schema import ContextBatch                                          # noqa: E402
from constraint_optimizer.code.train.module import (                                     # noqa: E402
    ConstraintOptimizer,
    build_module_from_config,
)
from constraint_optimizer.test_road_net.road_schema import (                              # noqa: E402
    RoadNetwork,
    enu_km_to_llh,
    llh_to_enu_km,
    road_network_to_tensors,
)
from gnn1.code.train.model import build_model_from_config as build_gnn1_model            # noqa: E402

# lstm2/code/data 下的同包模块
from data.synth_roads import (                                                           # noqa: E402
    random_road_topology,
    random_road_topology_per_candidate,
)
from data.labels import (                                                                # noqa: E402
    LabelConfig,
    compute_intent_threat_batch,
    INTENT_NAME,
)


# ============================================================
# 工具
# ============================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_ckpt(p: Path) -> Optional[Path]:
    """ckpt 路径解析：.pt 文件直接返回；目录则找最新 .pt；其余 None。"""
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


def resolve_rel(rel: str, base: Path) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


# ============================================================
# Scaler（兼容 lstm1 / gnn1 的 scaler_posvel.npz：键 mean / std）
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
# 解码：norm+delta → 物理 km/(km/s)
# ============================================================

def decode_history_to_phys(
    history_norm: np.ndarray,        # [N_w, 20, 6]  norm+delta
    scaler: _Scaler,
) -> np.ndarray:
    """
    把 history（norm+delta）解码成物理空间，并把 hist 末帧位置平移到 (0,0,0)。
    速度通道直接 inverse_transform。

    Returns:
        hist_phys: [N_w, 20, 6]   float32
    """
    feat = scaler.inverse_transform(history_norm.astype(np.float64))   # [N_w, 20, 6]
    pos_delta = feat[..., :3]
    vel = feat[..., 3:6]
    pos_cum = np.cumsum(pos_delta, axis=-2)                            # [N_w, 20, 3]
    last = pos_cum[:, -1:, :]                                          # [N_w, 1, 3]
    pos_phys = pos_cum - last                                          # 末帧 = 0
    return np.concatenate([pos_phys, vel], axis=-1).astype(np.float32)


def decode_future_to_phys(
    fut_norm: np.ndarray,            # [..., 10, 6]  norm+delta，相对 hist 末帧
    scaler: _Scaler,
) -> np.ndarray:
    """
    把 future（norm+delta）解码成物理空间。hist 末帧已经在 (0,0,0)，
    所以这里直接 cumsum 即可（不再加 last_hist_pos）。
    """
    feat = scaler.inverse_transform(fut_norm.astype(np.float64))
    pos_delta = feat[..., :3]
    vel = feat[..., 3:6]
    pos_phys = np.cumsum(pos_delta, axis=-2)
    return np.concatenate([pos_phys, vel], axis=-1).astype(np.float32)


# ============================================================
# GNN1 推理（批量）
# ============================================================

def run_gnn1_topk(
    model: torch.nn.Module,
    candidates_norm: np.ndarray,     # [N_s, M, T, D]
    task_type: np.ndarray,           # [N_s]
    type_id: np.ndarray,             # [N_s]
    position: np.ndarray,            # [N_s, 3]
    device: torch.device,
    batch_size: int = 512,
    desc: str = "GNN1 forward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        top_idx   [N_s, K]  int64
        top_probs [N_s, K]  float32   重归一化后 K 条和 = 1
    """
    model.eval()
    N = candidates_norm.shape[0]
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
# 路网批构造 + 约束优化（批量）
# ============================================================

def _network_to_km_tensor(
    network: "RoadNetwork",
    origin_llh: Tuple[float, float, float],
    nb_max: int,
    np_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接把 RoadNetwork(LLH) 转成 [NB_max, NP_max, 3] km 张量 + mask，
    用 numpy 批量把 LLH 转回 km，避免逐点 Python 函数调用。
    """
    rp = np.zeros((nb_max, np_max, 3), dtype=np.float32)
    rm = np.zeros((nb_max, np_max), dtype=bool)
    lon0, lat0, alt0 = origin_llh
    lat0_rad = np.deg2rad(lat0)
    cos_lat0 = float(np.cos(lat0_rad))
    EARTH_R_KM = 6371.0
    for bi, branch in enumerate(network[:nb_max]):
        pts = branch.points[:np_max]
        if not pts:
            continue
        n = len(pts)
        lons = np.fromiter((p.lon_deg for p in pts), dtype=np.float64, count=n)
        lats = np.fromiter((p.lat_deg for p in pts), dtype=np.float64, count=n)
        alts = np.fromiter((p.alt_m for p in pts), dtype=np.float64, count=n)
        dx = np.deg2rad(lons - lon0) * EARTH_R_KM * cos_lat0
        dy = np.deg2rad(lats - lat0) * EARTH_R_KM
        dz = (alts - alt0) / 1000.0
        rp[bi, :n, 0] = dx.astype(np.float32)
        rp[bi, :n, 1] = dy.astype(np.float32)
        rp[bi, :n, 2] = dz.astype(np.float32)
        rm[bi, :n] = True
    return rp, rm


def build_road_batch(
    top_phys: np.ndarray,           # [B, K, T, 6]   物理坐标，前 3 维是 pos
    rng: np.random.Generator,
    origin_llh: Tuple[float, float, float],
    nb_max: int,
    np_max: int,
    topology_mix: Dict[str, float],
    per_candidate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    对每个样本随机抽拓扑造路网。

    若 per_candidate=False（共享路网模式）:
        每个样本一份路网，K 条候选共用 → topo_names[i] = [name] * K
    若 per_candidate=True（独立路网模式，方案 B）:
        每条候选一份路网 → topo_names[i] = [name_k0, name_k1, name_k2]

    Returns:
        road_points [B, K, NB_max, NP_max, 3]  float32
        road_mask   [B, K, NB_max, NP_max]     bool
        topo_names  List[List[str]]            len = B, 每个内层 len = K
    """
    B, K, T, _ = top_phys.shape
    rp = np.zeros((B, K, nb_max, np_max, 3), dtype=np.float32)
    rm = np.zeros((B, K, nb_max, np_max), dtype=bool)
    topo_names: List[List[str]] = []

    for i in range(B):
        cand_xyz = top_phys[i, :, :, :3].astype(np.float64)         # [K, T, 3]
        if per_candidate:
            names_k, networks_k = random_road_topology_per_candidate(
                cand_xyz_km=cand_xyz,
                rng=rng,
                origin_llh=origin_llh,
                mix=topology_mix,
            )
            for k in range(K):
                rp_k, rm_k = _network_to_km_tensor(networks_k[k], origin_llh, nb_max, np_max)
                rp[i, k] = rp_k
                rm[i, k] = rm_k
            topo_names.append(names_k)
        else:
            name, network = random_road_topology(
                cand_xyz_km=cand_xyz,
                rng=rng,
                origin_llh=origin_llh,
                mix=topology_mix,
            )
            rp_i, rm_i = _network_to_km_tensor(network, origin_llh, nb_max, np_max)
            for k in range(K):
                rp[i, k] = rp_i
                rm[i, k] = rm_i
            topo_names.append([name] * K)

    return rp, rm, topo_names


def run_constraint_batch(
    constraint: torch.nn.Module,
    top_phys: np.ndarray,           # [B, K, T, 6]
    road_points: np.ndarray,        # [B, K, NB, NP, 3]
    road_mask: np.ndarray,          # [B, K, NB, NP]
    device: torch.device,
) -> np.ndarray:
    """
    把 [B, K, T, 6] 展平到 [B*K, T, 6] 喂给 ConstraintOptimizer，返回 refined 同形。
    路网 ctx 也是 [B, K, NB, NP, 3] → reshape 到 [B*K, NB, NP, 3]，
    支持每条候选独立路网（方案 B）也支持共享路网（前 K 维都相同）。
    """
    B, K, T, D = top_phys.shape
    NB, NP = road_points.shape[2], road_points.shape[3]

    sel_t = torch.from_numpy(top_phys.reshape(B * K, T, D)).to(device).float()
    rp_t = torch.from_numpy(road_points.reshape(B * K, NB, NP, 3)).to(device).float()
    rm_t = torch.from_numpy(road_mask.reshape(B * K, NB, NP)).to(device).bool()

    BK = B * K
    ctx = ContextBatch(
        task_type=torch.zeros(BK, dtype=torch.long, device=device),
        type=torch.zeros(BK, dtype=torch.long, device=device),
        position=torch.zeros(BK, 3, dtype=torch.float32, device=device),
        road_points=rp_t,
        road_mask=rm_t,
        eta=torch.zeros(BK, dtype=torch.long, device=device),
    )

    with torch.no_grad():
        refined_flat = constraint(sel_t, ctx)               # [B*K, T, 6]

    return refined_flat.detach().cpu().numpy().reshape(B, K, T, D).astype(np.float32)


# ============================================================
# 单 split 主流程
# ============================================================

# ============================================================
# 多进程 worker（必须放在模块顶层，spawn 才能 pickle）
# ============================================================

# 每个 worker 进程独立持有的全局对象（由 _worker_init 初始化一次）
_WORKER_GLOBALS: Dict[str, Any] = {}


def _worker_init(constraint_module_type: str, constraint_enable: bool) -> None:
    """
    Pool worker initializer：每个进程启动时只重建 ConstraintOptimizer 一次，
    后续所有任务复用。ConstraintOptimizer 没有可学权重，重建很便宜。
    """
    _WORKER_GLOBALS["constraint"] = ConstraintOptimizer(
        enable=constraint_enable,
        module_type=constraint_module_type,
    ).eval()


def _constrain_worker(
    args: Tuple[
        int,                           # chunk_idx
        np.ndarray,                    # top_phys_chunk [n, K, T, 6]
        np.ndarray,                    # raw_pos_chunk  [n, 3]
        Tuple[float, float, float],    # origin_llh
        int, int,                      # nb_max, np_max
        Dict[str, float],              # topology_mix
        Dict[str, Any],                # label_cfg_dict
        float,                         # time_step_s
        bool,                          # per_candidate_road
        bool,                          # save_roads
        int,                           # worker_seed
    ],
) -> Tuple[
    int,                               # chunk_idx
    np.ndarray,                        # refined [n, K, T, 6]
    List[List[str]],                   # topology
    np.ndarray,                        # intent [n*K]
    np.ndarray,                        # threat [n*K]
    Optional[np.ndarray],              # roads [n, K, NB, NP, 3] or None
    Optional[np.ndarray],              # roads_mask [n, K, NB, NP] or None
]:
    """
    单个 chunk 的 constrain 处理：路网生成 + 约束优化 + intent/threat 计算。
    """
    (
        chunk_idx,
        top_phys_chunk,
        raw_pos_chunk,
        origin_llh,
        nb_max, np_max,
        topology_mix,
        label_cfg_dict, time_step_s,
        per_candidate_road, save_roads,
        worker_seed,
    ) = args

    constraint = _WORKER_GLOBALS["constraint"]
    rng = np.random.default_rng(worker_seed)
    label_cfg = LabelConfig.from_dict(label_cfg_dict, time_step_s=time_step_s)

    rp_b, rm_b, topo_b = build_road_batch(
        top_phys=top_phys_chunk,
        rng=rng,
        origin_llh=origin_llh,
        nb_max=nb_max,
        np_max=np_max,
        topology_mix=topology_mix,
        per_candidate=per_candidate_road,
    )
    refined_b = run_constraint_batch(
        constraint,
        top_phys=top_phys_chunk,
        road_points=rp_b,
        road_mask=rm_b,
        device=torch.device("cpu"),
    )

    n, K, T, D = refined_b.shape
    refined_flat = refined_b.reshape(n * K, T, D)
    pos_flat = np.repeat(raw_pos_chunk, K, axis=0)
    intent, threat = compute_intent_threat_batch(refined_flat, pos_flat, label_cfg)

    if save_roads:
        return chunk_idx, refined_b, topo_b, intent, threat, rp_b, rm_b
    return chunk_idx, refined_b, topo_b, intent, threat, None, None


# ============================================================
# 单 split 主流程
# ============================================================

def process_split(
    split: str,
    cfg: Dict[str, Any],
    gnn1_model: torch.nn.Module,
    constraint: torch.nn.Module,
    scaler: _Scaler,
    rng: np.random.Generator,
    device: torch.device,
    constraint_device: torch.device,
    cache_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    label_cfg: LabelConfig,
    sample_batch: int,
    gnn1_batch: int,
    max_samples: int = 0,
    per_candidate_road: bool = True,
    save_roads: bool = False,
    num_workers: int = 0,
    constraint_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    cache_path = cache_dir / f"{split}.npz"
    raw_path = raw_dir / f"{split}.npz"
    if not cache_path.exists():
        print(f"[lstm2/gen] {split}: 缺 cache {cache_path}，跳过")
        return
    if not raw_path.exists():
        print(f"[lstm2/gen] {split}: 缺 gnn1 raw {raw_path}，跳过")
        return

    print(f"[lstm2/gen] === {split} ===")
    cache_d = np.load(cache_path)
    raw_d = np.load(raw_path)

    history_w = cache_d["history"]                          # [N_w, 20, 6] norm+delta
    targets_w = cache_d["targets"] if "targets" in cache_d.files else None

    raw_cand = raw_d["candidates"]                          # [N_s, M, 10, 6] norm+delta
    raw_task = raw_d["task_type"].astype(np.int64)
    raw_type = raw_d["type"].astype(np.int64)
    raw_pos = raw_d["position"].astype(np.float32)

    N_w = int(history_w.shape[0])
    N_s_full = int(raw_cand.shape[0])
    if N_s_full % N_w != 0:
        raise RuntimeError(
            f"raw 样本数 {N_s_full} 不能整除 cache window 数 {N_w}；"
            "请检查 gnn1 generate_data 的 samples_per_window 是否一致"
        )
    spw = N_s_full // N_w

    # 可选截断
    if max_samples > 0 and max_samples < N_s_full:
        N_s = int(max_samples)
        raw_cand = raw_cand[:N_s]
        raw_task = raw_task[:N_s]
        raw_type = raw_type[:N_s]
        raw_pos = raw_pos[:N_s]
        print(f"[lstm2/gen] {split}: N_w={N_w}, samples_per_window={spw}, "
              f"N_s={N_s} / {N_s_full} (truncated by --max-samples)")
    else:
        N_s = N_s_full
        print(f"[lstm2/gen] {split}: N_w={N_w}, N_s={N_s}, samples_per_window={spw}")

    # ---- 1) GNN1 跑 top-K ----
    t0 = time.time()
    top_idx, top_probs = run_gnn1_topk(
        gnn1_model,
        candidates_norm=raw_cand,
        task_type=raw_task,
        type_id=raw_type,
        position=raw_pos,
        device=device,
        batch_size=gnn1_batch,
        desc=f"[lstm2/gen] {split} GNN1",
    )                                                       # [N_s, K], [N_s, K]
    K = top_idx.shape[-1]
    print(f"[lstm2/gen] {split}: GNN1 forward done, K={K}, "
          f"{time.time() - t0:.1f}s")

    # ---- 2) 反归一化候选到物理 ----
    cand_phys = decode_future_to_phys(raw_cand, scaler)     # [N_s, M, 10, 6]
    M = cand_phys.shape[1]
    Tout = cand_phys.shape[2]
    Df = cand_phys.shape[3]
    # 释放 raw_cand：GNN1 forward 完 + decode 完，后面用不到（train 时 ~3 GB）
    del raw_cand

    # ---- 3) gather top-K 物理候选 ----
    # top_idx[n, k] ∈ [0, M) → cand_phys[n, top_idx[n, k]]
    top_phys = np.take_along_axis(
        cand_phys,
        top_idx[:, :, None, None].repeat(Tout, axis=2).repeat(Df, axis=3),
        axis=1,
    )                                                       # [N_s, K, 10, 6]
    # 释放 cand_phys：已 gather 出 top_phys，5 条全量物理候选不再需要
    del cand_phys

    # ---- 4) 解码 history 到物理（per window, 然后按 raw_idx → window_idx 复刻）----
    hist_phys_w = decode_history_to_phys(history_w, scaler)         # [N_w, 20, 6]
    win_idx = np.arange(N_s) // spw                                 # [N_s]
    hist_phys = hist_phys_w[win_idx]                                # [N_s, 20, 6]
    del hist_phys_w

    # ---- 5) GT 未来（可选）----
    fut_gt_phys: Optional[np.ndarray] = None
    if targets_w is not None:
        gt_w = decode_future_to_phys(targets_w, scaler)             # [N_w, 10, 6]
        fut_gt_phys = gt_w[win_idx]                                 # [N_s, 10, 6]
        del gt_w

    # ---- 6) 路网造假 + 约束优化（batched）----
    nb_max = int(cfg.get("data", {}).get("nb_max", 4))
    np_max = int(cfg.get("data", {}).get("np_max", 64))
    origin_llh = tuple(cfg.get("data", {}).get("origin_llh", [116.30, 39.90, 0.0]))
    if len(origin_llh) != 3:
        raise ValueError(f"origin_llh 必须是 [lon, lat, alt]，实际 {origin_llh}")
    topology_mix = dict(cfg.get("data", {}).get("topology_mix", {}))

    refined = np.empty((N_s, K, Tout, Df), dtype=np.float32)
    # topology: [N_s, K]，per-candidate road 时每条候选有自己的拓扑名
    topology = np.empty((N_s, K), dtype=object)
    intent_out = np.empty(N_s * K, dtype=np.int8)
    threat_out = np.empty(N_s * K, dtype=np.int16)
    # 可选保存 road_points / road_mask 用于可视化
    if save_roads:
        roads_save = np.zeros((N_s, K, nb_max, np_max, 3), dtype=np.float32)
        roads_mask_save = np.zeros((N_s, K, nb_max, np_max), dtype=bool)

    t1 = time.time()

    # 准备 LabelConfig 的 dict + time_step_s（多进程要可 pickle）
    label_cfg_dict_for_worker = cfg.get("data", {}).get("intent_threat", {}) or {}
    time_step_s_for_worker = float(cfg.get("data", {}).get("time_step_s", 60.0))

    if num_workers and num_workers > 1:
        # ---- 多进程并行（spawn）----
        # chunk_size：每个 worker 任务的样本数；为了摊薄 IPC 开销，比 sample_batch 大一点
        chunk_size = max(sample_batch, 512)
        chunks = []
        for i in range(0, N_s, chunk_size):
            j = min(i + chunk_size, N_s)
            chunks.append((
                i // chunk_size,                            # chunk_idx
                top_phys[i:j].copy(),                       # 必须 copy 才能 pickle
                raw_pos[i:j].copy(),
                origin_llh,                                 # tuple, picklable
                nb_max, np_max,
                topology_mix,
                label_cfg_dict_for_worker,
                time_step_s_for_worker,
                per_candidate_road, save_roads,
                int(rng.integers(0, 2 ** 31 - 1)),
            ))

        # 取 constraint cfg 用于 worker initializer
        c_cfg = (constraint_cfg or {}).get("module", {}) or {}
        c_type = str(c_cfg.get("type", "road_arc_projection"))
        c_enable = bool(c_cfg.get("enable", True))

        ctx = mp.get_context("spawn")
        pbar = tqdm(
            total=N_s,
            desc=f"[lstm2/gen] {split} constrain (mp x{num_workers})",
            unit="samp", ncols=100, smoothing=0.1,
        )
        with ctx.Pool(
            processes=int(num_workers),
            initializer=_worker_init,
            initargs=(c_type, c_enable),
        ) as pool:
            # imap_unordered 完成更快，按 chunk_idx 写回到正确位置
            for result in pool.imap_unordered(_constrain_worker, chunks, chunksize=1):
                (chunk_idx, refined_chunk, topo_chunk,
                 intent_chunk, threat_chunk, rp_chunk, rm_chunk) = result
                i = chunk_idx * chunk_size
                n_chunk = refined_chunk.shape[0]
                j = i + n_chunk
                refined[i:j] = refined_chunk
                for bi, names_k in enumerate(topo_chunk):
                    topology[i + bi] = np.asarray(names_k, dtype=object)
                intent_out[i * K: j * K] = intent_chunk
                threat_out[i * K: j * K] = threat_chunk
                if save_roads:
                    roads_save[i:j] = rp_chunk
                    roads_mask_save[i:j] = rm_chunk
                pbar.update(n_chunk)
        pbar.close()
    else:
        # ---- 单进程串行（保持向后兼容）----
        pbar = tqdm(
            total=N_s,
            desc=f"[lstm2/gen] {split} constrain",
            unit="samp", ncols=100, smoothing=0.1,
        )
        for bi_idx, i in enumerate(range(0, N_s, sample_batch)):
            j = min(i + sample_batch, N_s)
            rp_b, rm_b, topo_b = build_road_batch(
                top_phys=top_phys[i:j],
                rng=rng,
                origin_llh=origin_llh,  # type: ignore
                nb_max=nb_max,
                np_max=np_max,
                topology_mix=topology_mix,
                per_candidate=per_candidate_road,
            )
            refined_b = run_constraint_batch(
                constraint,
                top_phys=top_phys[i:j],
                road_points=rp_b,
                road_mask=rm_b,
                device=constraint_device,
            )
            refined[i:j] = refined_b
            for bi, names_k in enumerate(topo_b):
                topology[i + bi] = np.asarray(names_k, dtype=object)
            if save_roads:
                roads_save[i:j] = rp_b
                roads_mask_save[i:j] = rm_b
            pbar.update(j - i)
        pbar.close()

        # 单进程模式下 intent/threat 在循环外算
        refined_flat_tmp = refined.reshape(N_s * K, Tout, Df)
        pos_flat_tmp = np.repeat(raw_pos, K, axis=0)
        intent_out, threat_out = compute_intent_threat_batch(
            refined_flat_tmp, pos_flat_tmp, label_cfg,
        )

    print(f"[lstm2/gen] {split}: constrain all done {time.time() - t1:.1f}s "
          f"(per_candidate_road={per_candidate_road}, "
          f"workers={num_workers if num_workers else 1})")

    # 释放 top_phys：已经经过约束优化，refined 是输出，top_phys 不再需要
    del top_phys

    # ---- 7) 用上面的 intent/threat ----
    refined_flat = refined.reshape(N_s * K, Tout, Df)
    pos_flat = np.repeat(raw_pos, K, axis=0)
    intent = intent_out
    threat = threat_out

    # ---- 8) 组装输出 ----
    M_out = N_s * K
    hist_raw_out = np.repeat(hist_phys, K, axis=0).astype(np.float32)
    fut_refined_out = refined_flat.astype(np.float32)
    position_out = pos_flat.astype(np.float32)
    task_out = np.repeat(raw_task.astype(np.int8), K, axis=0)
    type_out = np.repeat(raw_type.astype(np.int8), K, axis=0)
    # topology 现在是 [N_s, K] → flatten 到 [M_out]
    topology_out = topology.reshape(M_out).astype("U16")
    top_prob_out = top_probs.reshape(M_out).astype(np.float32)
    gnn1_top_idx_out = top_idx.reshape(M_out).astype(np.int8)
    sample_idx_out = np.repeat(np.arange(N_s, dtype=np.int32), K, axis=0)
    cand_k_out = np.tile(np.arange(K, dtype=np.int8), N_s)

    out: Dict[str, np.ndarray] = {
        "hist_raw":     hist_raw_out,
        "fut_refined":  fut_refined_out,
        "position":     position_out,
        "task_type":    task_out,
        "type":         type_out,
        "intent_label": intent.astype(np.int8),
        "threat_score": threat.astype(np.int16),
        "topology":     topology_out,
        "top_prob":     top_prob_out,
        "sample_idx":   sample_idx_out,
        "cand_k":       cand_k_out,
        "gnn1_top_idx": gnn1_top_idx_out,
    }
    if fut_gt_phys is not None:
        out["fut_gt"] = np.repeat(fut_gt_phys, K, axis=0).astype(np.float32)
    if save_roads:
        # 路网形状 [N_s, K, NB, NP, 3] → flatten 到 [M_out, NB, NP, 3]
        out["road_points"] = roads_save.reshape(M_out, nb_max, np_max, 3)
        out["road_mask"] = roads_mask_save.reshape(M_out, nb_max, np_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npz"
    np.savez(out_path, **out)
    print(f"[lstm2/gen] {split}: saved → {out_path}  ({M_out} samples)")

    # ---- 9) 健康检查 ----
    print_split_summary(out_path, out)


def print_split_summary(out_path: Path, out: Dict[str, np.ndarray]) -> None:
    print(f"\n[lstm2/gen] summary  {out_path.name}")
    intent = out["intent_label"]
    threat = out["threat_score"]
    topology = out["topology"]

    n = int(intent.shape[0])
    print(f"  total samples = {n}")

    intent_dist: Dict[str, float] = {}
    for k, name in INTENT_NAME.items():
        c = int((intent == k).sum())
        intent_dist[name] = c / max(1, n)
    intent_str = "  ".join(f"{name}={intent_dist[name] * 100:.1f}%" for name in
                           ["ATTACK", "EVASION", "DEFENSE", "RETREAT"])
    print(f"  intent dist = {intent_str}")

    threat = threat.astype(np.float32)
    print(f"  threat       min={threat.min():.1f}  max={threat.max():.1f}  "
          f"mean={threat.mean():.1f}  std={threat.std():.1f}")
    bins = [0, 20, 40, 60, 80, 101]
    hist, _ = np.histogram(threat, bins=bins)
    bin_str = "  ".join(
        f"[{bins[i]},{bins[i + 1] - 1 if i + 1 < len(bins) - 1 else bins[i + 1]}]={hist[i] / n * 100:.1f}%"
        for i in range(len(hist))
    )
    print(f"  threat dist  {bin_str}")

    uniq, cnt = np.unique(topology, return_counts=True)
    topo_str = "  ".join(f"{u}={c / n * 100:.1f}%" for u, c in zip(uniq, cnt))
    print(f"  topology     {topo_str}")
    print()


# ============================================================
# main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LSTM2 training data.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="lstm2 config 路径，相对 lstm2/ 或绝对路径")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto",
                        help="GNN1 forward 的设备：auto/cuda/cpu")
    parser.add_argument("--constraint-device", type=str, default="cpu",
                        help="ConstraintOptimizer 设备；默认 cpu，因为 road_arc_projection 内部"
                             "有 Python loop + .item() 同步，CPU 反而比 CUDA 快")
    parser.add_argument("--gnn1-batch", type=int, default=512)
    parser.add_argument("--sample-batch", type=int, default=128,
                        help="路网造假 + 约束优化的样本批大小")
    parser.add_argument("--gnn1-ckpt", type=str, default="",
                        help="覆盖 config 里的 data.gnn1_ckpt；可以是 .pt 或目录")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="每个 split 最多取的样本数（0 = 全部）；用于快速冒烟")
    parser.add_argument("--per-candidate-road", action="store_true", default=True,
                        help="方案B：每条候选独立路网（默认 True，更分散）")
    parser.add_argument("--shared-road", dest="per_candidate_road",
                        action="store_false",
                        help="方案A：K 条候选共享一份路网（部署形态）")
    parser.add_argument("--save-roads", type=str, default="auto",
                        choices=["auto", "yes", "no"],
                        help="是否把 road_points/road_mask 写入 npz（auto: max_samples<50000 才存）")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="constrain 阶段并行进程数；0 或 1 = 单进程（默认）；"
                             "推荐 8-16（实测 28 核机器 8 进程已经能从 15 分钟降到 ~3 分钟）")
    args = parser.parse_args()

    # ---- 解析路径 ----
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_LSTM2_ROOT / cfg_path).resolve()
    cfg = load_yaml(cfg_path)

    data_cfg = cfg.get("data", {})

    cache_dir = resolve_rel(data_cfg.get("gnn1_cache_dir", "../gnn1/data/cache"), _LSTM2_ROOT)
    raw_dir = resolve_rel(data_cfg.get("gnn1_raw_dir", "../gnn1/data/raw"), _LSTM2_ROOT)
    gnn1_cfg_path = resolve_rel(data_cfg.get("gnn1_config", "../gnn1/config.yaml"), _LSTM2_ROOT)
    gnn1_ckpt_arg = args.gnn1_ckpt or data_cfg.get("gnn1_ckpt", "../gnn1/checkpoints")
    gnn1_ckpt_path = resolve_rel(gnn1_ckpt_arg, _LSTM2_ROOT)
    scaler_path = resolve_rel(
        data_cfg.get("lstm1_scaler", "../lstm1/data/processed/scaler_posvel.npz"),
        _LSTM2_ROOT,
    )
    constraint_cfg_path = resolve_rel(
        data_cfg.get("constraint_config", "../constraint_optimizer/config.yaml"),
        _LSTM2_ROOT,
    )
    out_dir = resolve_rel(data_cfg.get("raw_dir", "data/raw"), _LSTM2_ROOT)

    print(f"[lstm2/gen] cfg                = {cfg_path}")
    print(f"[lstm2/gen] gnn1 cache_dir     = {cache_dir}")
    print(f"[lstm2/gen] gnn1 raw_dir       = {raw_dir}")
    print(f"[lstm2/gen] gnn1 config        = {gnn1_cfg_path}")
    print(f"[lstm2/gen] gnn1 ckpt          = {gnn1_ckpt_path}")
    print(f"[lstm2/gen] lstm1 scaler       = {scaler_path}")
    print(f"[lstm2/gen] constraint config  = {constraint_cfg_path}")
    print(f"[lstm2/gen] out_dir            = {out_dir}")

    if not cache_dir.exists():
        raise FileNotFoundError(f"找不到 gnn1 cache 目录：{cache_dir}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"找不到 gnn1 raw 目录：{raw_dir}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 lstm1 scaler：{scaler_path}")

    # ---- 设备 ----
    device = setup_device(args.device)
    constraint_device = setup_device(args.constraint_device)
    print(f"[lstm2/gen] gnn1 device        = {device}")
    print(f"[lstm2/gen] constraint device  = {constraint_device}")

    # ---- 加载 GNN1 ----
    gnn1_cfg = load_yaml(gnn1_cfg_path)
    gnn1_model = build_gnn1_model(gnn1_cfg).to(device)
    gnn1_ckpt_file = find_latest_ckpt(gnn1_ckpt_path)
    if gnn1_ckpt_file is None:
        raise FileNotFoundError(
            f"找不到 GNN1 ckpt（{gnn1_ckpt_path}）；请先训练 GNN1 或在 config 里指定"
        )
    print(f"[lstm2/gen] GNN1 ckpt picked   = {gnn1_ckpt_file}")
    sd = torch.load(gnn1_ckpt_file, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    gnn1_model.load_state_dict(sd, strict=True)
    gnn1_model.eval()

    # ---- 加载 ConstraintOptimizer ----
    constraint_cfg = load_yaml(constraint_cfg_path)
    constraint = build_module_from_config(constraint_cfg).to(constraint_device).eval()
    print(f"[lstm2/gen] constraint type    = "
          f"{constraint_cfg.get('module', {}).get('type', '?')}, "
          f"enable={constraint_cfg.get('module', {}).get('enable', True)}")

    # ---- Scaler ----
    scaler = _Scaler.load(scaler_path)
    print(f"[lstm2/gen] scaler loaded      = {scaler_path}")

    # ---- LabelConfig ----
    label_cfg_dict = data_cfg.get("intent_threat", {}) or {}
    time_step_s = float(data_cfg.get("time_step_s", 60.0))
    label_cfg = LabelConfig.from_dict(label_cfg_dict, time_step_s=time_step_s)

    # ---- 随机数 ----
    seed = int(data_cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- save_roads 解析 ----
    save_roads_mode = args.save_roads
    if save_roads_mode == "auto":
        # auto: 只有当 --max-samples 小于 50000 时才存路网（避免大数据集体积爆炸）
        save_roads = (0 < int(args.max_samples) < 50000)
    elif save_roads_mode == "yes":
        save_roads = True
    else:
        save_roads = False
    print(f"[lstm2/gen] per_candidate_road = {bool(args.per_candidate_road)}")
    print(f"[lstm2/gen] save_roads         = {save_roads}  (mode={save_roads_mode})")
    print(f"[lstm2/gen] num_workers        = {int(args.num_workers)}")

    # ---- 跑每个 split ----
    for split in args.splits:
        process_split(
            split=split,
            cfg=cfg,
            gnn1_model=gnn1_model,
            constraint=constraint,
            scaler=scaler,
            rng=rng,
            device=device,
            constraint_device=constraint_device,
            cache_dir=cache_dir,
            raw_dir=raw_dir,
            out_dir=out_dir,
            label_cfg=label_cfg,
            sample_batch=int(args.sample_batch),
            gnn1_batch=int(args.gnn1_batch),
            max_samples=int(args.max_samples),
            per_candidate_road=bool(args.per_candidate_road),
            save_roads=save_roads,
            num_workers=int(args.num_workers),
            constraint_cfg=constraint_cfg,
        )

    print("[lstm2/gen] all splits done.")


if __name__ == "__main__":
    main()
