"""
lstm2/code/data/synth_roads.py
------------------------------
为 LSTM2 训练数据生成多种"造假路网"，让约束优化器在不同拓扑下产出有差异
的精修轨迹（让 4 类意图的标签都更容易出现）。

每种生成器的接口对齐：
    def build_xxx(cand_xyz_km: np.ndarray,            # [K, T, 3]
                  *,
                  origin_llh=(116.30, 39.90, 0.0),
                  rng: np.random.Generator,
                  ) -> RoadNetwork

派发函数:
    random_road_topology(cand_xyz_km, rng, mix=None) -> Tuple[name, RoadNetwork]

5 种拓扑（v2 简化版：路网几何更平滑，单条 branch 拐点 ≤ 3 个，弯曲幅度 ≤ 8% L_main）：
  - "y_fork"        : K 叉路网，每条候选一条几乎直线的专属支路
  - "t_junction"    : 1 条直主干 + 1 条直横向支路（共 2 条 branch）
  - "straight"      : 单条直走廊
  - "curved"        : 单条温和弧形（鼓包 15% L_main，1 个中间点）
  - "dead_end"      : 主干 + 折返段，3 个点
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 复用 constraint_optimizer 已经定义的 RoadNetwork / 转换工具
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from constraint_optimizer.test_road_net.road_schema import (  # noqa: E402
    RoadBranchLLH,
    RoadNetwork,
    enu_km_to_llh,
)


# ============================================================
# 工具
# ============================================================

def _polyline_to_branch(
    xy: np.ndarray,                              # [N, 2]
    z_km: float,
    origin_llh: Tuple[float, float, float],
) -> RoadBranchLLH:
    branch = RoadBranchLLH(points=[])
    for (x, y) in xy:
        branch.points.append(
            enu_km_to_llh(float(x), float(y), float(z_km), *origin_llh)
        )
    return branch


def _densify(polyline: np.ndarray, n_pts_per_seg: int = 5) -> np.ndarray:
    """在折线每两个角点之间均匀插点。返回 [M, 2]。"""
    if polyline.shape[0] < 2:
        return polyline.astype(np.float64)
    pts = []
    for i in range(polyline.shape[0] - 1):
        a = polyline[i]
        b = polyline[i + 1]
        for k in range(n_pts_per_seg):
            t = k / n_pts_per_seg
            pts.append((1.0 - t) * a + t * b)
    pts.append(polyline[-1])
    return np.asarray(pts, dtype=np.float64)


def _perp(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]])


def _safe_unit(v: np.ndarray) -> Tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0]), 0.0
    return v / n, n


def _z_mean(cand_xyz_km: np.ndarray) -> float:
    return float(cand_xyz_km[..., 2].mean())


def _avg_end_direction(cand_xyz_km: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    返回:
      u_main   : 单位主方向 [2]
      L_main   : 平均终点距 |avg_end| (km)
      avg_end  : 平均终点 [2]
    """
    endpoints = cand_xyz_km[:, -1, :2].astype(np.float64)   # [K, 2]
    avg_end = endpoints.mean(axis=0)                        # [2]
    u_main, L_main = _safe_unit(avg_end - np.zeros(2))
    return u_main, max(L_main, 2.0), avg_end


# ============================================================
# 1) Y 叉：每条候选一条几乎直线的专属支路
# ============================================================

def build_y_fork(
    cand_xyz_km: np.ndarray,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    rng: np.random.Generator,
) -> RoadNetwork:
    """
    每条候选一条专属支路：origin → endpoint_k → tail_k(延长 25%)。
    可选 5% 侧向轻微弯，保留"路看起来不是绝对直"的视觉。

    Per-candidate 模式下 K=1，会得到 1 条 branch；共享模式下 K=3，得到 3 条。
    """
    p0 = np.array([0.0, 0.0])
    z = _z_mean(cand_xyz_km)
    K = cand_xyz_km.shape[0]

    branches: RoadNetwork = []
    for k in range(K):
        end_k = cand_xyz_km[k, -1, :2].astype(np.float64)
        u_k, L_k = _safe_unit(end_k - p0)
        if L_k < 1e-3:
            # 候选退化成原地（极少见），用一段单位长度的 dummy
            corners = np.stack([p0, p0 + np.array([1.0, 0.0])], axis=0)
        else:
            n_k = _perp(u_k)
            side = 1.0 if rng.random() < 0.5 else -1.0
            # 中段轻微弯（5% L_k 的侧偏）
            mid = 0.5 * (p0 + end_k) + side * 0.05 * L_k * n_k
            # 末端延长 25%，避免候选走超尾被夹
            tail = end_k + 0.25 * L_k * u_k
            corners = np.stack([p0, mid, end_k, tail], axis=0)
        branches.append(_polyline_to_branch(_densify(corners, 6), z, origin_llh))
    return branches


# ============================================================
# 2) T 字路口：直主干 + 直横向支路
# ============================================================

def build_t_junction(
    cand_xyz_km: np.ndarray,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    rng: np.random.Generator,
) -> RoadNetwork:
    """
    T 字路口：1 条直主干（沿主方向 1.3 L_main） + 1 条直横向支路（垂直 1.0 L_main）。
    两条都是两点直线，无中间拐点。
    """
    p0 = np.array([0.0, 0.0])
    u_main, L_main, _ = _avg_end_direction(cand_xyz_km)
    n_main = _perp(u_main)
    z = _z_mean(cand_xyz_km)

    # 主干：直接 origin → end_main，无 trunk_bend
    end_main = p0 + 1.3 * L_main * u_main
    main_corners = np.stack([p0, end_main], axis=0)

    # 横向支路：直接 origin → cross_end（垂直方向）
    cross_side = 1.0 if rng.random() < 0.5 else -1.0
    cross_end = p0 + cross_side * 1.0 * L_main * n_main
    cross_corners = np.stack([p0, cross_end], axis=0)

    return [
        _polyline_to_branch(_densify(main_corners, 8), z, origin_llh),
        _polyline_to_branch(_densify(cross_corners, 8), z, origin_llh),
    ]


# ============================================================
# 3) 直走廊
# ============================================================

def build_straight(
    cand_xyz_km: np.ndarray,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    rng: np.random.Generator,
) -> RoadNetwork:
    """
    单条直走廊：沿 K 候选终点平均方向，从 -0.2 L_main 到 +1.5 L_main。
    K 条候选都会被压到这条直线上 → 同向直奔，更容易出 ATTACK / DEFENSE。
    """
    p0 = np.array([0.0, 0.0])
    u_main, L_main, _ = _avg_end_direction(cand_xyz_km)
    z = _z_mean(cand_xyz_km)

    back = -0.20 * L_main * u_main
    end = p0 + 1.5 * L_main * u_main

    corners = np.stack([back, end], axis=0)
    branch = _polyline_to_branch(_densify(corners, 10), z, origin_llh)
    return [branch]


# ============================================================
# 4) 温和弧形
# ============================================================

def build_curved(
    cand_xyz_km: np.ndarray,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    rng: np.random.Generator,
) -> RoadNetwork:
    """
    单条温和弧形：3 个角点 (origin, mid, end_extended)。
    mid 在 origin → avg_end 中点 + 侧向 15% L_main 偏移；end_extended 在
    avg_end 沿主方向延长 20%。

    比起以前的 5 拐点弧，这版温和很多，仍能制造一定曲率让 EVASION 触发。
    """
    p0 = np.array([0.0, 0.0])
    u_main, L_main, avg_end = _avg_end_direction(cand_xyz_km)
    n_main = _perp(u_main)
    z = _z_mean(cand_xyz_km)

    side = 1.0 if rng.random() < 0.5 else -1.0
    mid = 0.5 * (p0 + avg_end) + side * 0.15 * L_main * n_main
    end_extended = avg_end + 0.20 * L_main * u_main

    corners = np.stack([p0, mid, end_extended], axis=0)
    branch = _polyline_to_branch(_densify(corners, 8), z, origin_llh)
    return [branch]


# ============================================================
# 5) 死路（折返段）
# ============================================================

def build_dead_end(
    cand_xyz_km: np.ndarray,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    rng: np.random.Generator,
) -> RoadNetwork:
    """
    死路：origin → tip(0.5 L_main) → return_point(微微侧偏回到原点附近)。
    3 个角点，比之前 4 点 U 弯更平滑。
    约束优化沿弧长投影 → 候选到达尽头后被压回 → 容易出 RETREAT。
    """
    p0 = np.array([0.0, 0.0])
    u_main, L_main, _ = _avg_end_direction(cand_xyz_km)
    n_main = _perp(u_main)
    z = _z_mean(cand_xyz_km)

    side = 1.0 if rng.random() < 0.5 else -1.0
    tip = p0 + 0.5 * L_main * u_main
    return_point = p0 + side * 0.05 * L_main * n_main   # 几乎回原点，微偏避免 0 长度段

    corners = np.stack([p0, tip, return_point], axis=0)
    branch = _polyline_to_branch(_densify(corners, 8), z, origin_llh)
    return [branch]


# ============================================================
# 派发
# ============================================================

_BUILDERS = {
    "y_fork":     build_y_fork,
    "t_junction": build_t_junction,
    "straight":   build_straight,
    "curved":     build_curved,
    "dead_end":   build_dead_end,
}

_DEFAULT_MIX: Dict[str, float] = {
    "y_fork":     0.25,
    "t_junction": 0.20,
    "straight":   0.20,
    "curved":     0.20,
    "dead_end":   0.15,
}


def _normalize_mix(mix: Optional[Dict[str, float]]) -> Tuple[List[str], np.ndarray]:
    if mix is None:
        mix = _DEFAULT_MIX
    names: List[str] = []
    probs: List[float] = []
    for name in _BUILDERS.keys():
        p = float(mix.get(name, 0.0))
        if p < 0:
            raise ValueError(f"topology_mix[{name}] 不能为负: {p}")
        names.append(name)
        probs.append(p)
    s = sum(probs)
    if s <= 0:
        raise ValueError(f"topology_mix 全部为 0: {mix}")
    probs_arr = np.asarray(probs, dtype=np.float64) / s
    return names, probs_arr


def random_road_topology(
    cand_xyz_km: np.ndarray,
    rng: np.random.Generator,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[str, RoadNetwork]:
    """
    随机抽一种拓扑生成路网（共享路网模式：K 条候选共用一份）。

    Args:
        cand_xyz_km: [K, T, 3]，GNN1 选出的 top-K 候选物理坐标
        rng        : numpy.Generator
        origin_llh : ENU 原点
        mix        : 名→概率，缺省走 _DEFAULT_MIX

    Returns:
        (name, road_network)
    """
    if cand_xyz_km.ndim != 3 or cand_xyz_km.shape[-1] != 3:
        raise ValueError(f"cand_xyz_km 形状应为 [K, T, 3]，实际 {cand_xyz_km.shape}")
    names, probs = _normalize_mix(mix)
    name = str(rng.choice(names, p=probs))
    builder = _BUILDERS[name]
    network = builder(cand_xyz_km, origin_llh=origin_llh, rng=rng)
    return name, network


def random_road_topology_per_candidate(
    cand_xyz_km: np.ndarray,
    rng: np.random.Generator,
    *,
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[RoadNetwork]]:
    """
    Per-candidate 路网模式：每条候选独立抽一种拓扑造一份路网，
    路网形状会基于"该条候选的轨迹"量身定做（让 3 条候选呈现 3 种不同的
    "如果走这条路会怎样"，从而提升 intent 标签多样性）。

    Args:
        cand_xyz_km: [K, T, 3]，GNN1 选出的 top-K 候选物理坐标
        rng        : numpy.Generator
        origin_llh : ENU 原点
        mix        : 名→概率，缺省走 _DEFAULT_MIX

    Returns:
        names    : List[str]   长度 K
        networks : List[RoadNetwork]  长度 K
    """
    if cand_xyz_km.ndim != 3 or cand_xyz_km.shape[-1] != 3:
        raise ValueError(f"cand_xyz_km 形状应为 [K, T, 3]，实际 {cand_xyz_km.shape}")
    K = cand_xyz_km.shape[0]
    names_pool, probs = _normalize_mix(mix)

    out_names: List[str] = []
    out_nets: List[RoadNetwork] = []
    for k in range(K):
        name = str(rng.choice(names_pool, p=probs))
        builder = _BUILDERS[name]
        # 把单条候选 reshape 成 [1, T, 3] 喂给 builder
        single = cand_xyz_km[k:k + 1]
        network = builder(single, origin_llh=origin_llh, rng=rng)
        out_names.append(name)
        out_nets.append(network)
    return out_names, out_nets


__all__ = [
    "build_y_fork",
    "build_t_junction",
    "build_straight",
    "build_curved",
    "build_dead_end",
    "random_road_topology",
    "random_road_topology_per_candidate",
]
