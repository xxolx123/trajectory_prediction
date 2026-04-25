"""
constraint_optimizer/test_road_net/synth_road.py
------------------------------------------------
"造假路网"：根据 GNN1 选出的 top-K 候选轨迹，造一个"K 叉"路网，让每条候选
都有一条最匹配的支路可贴。输出 **甲方 LLH 接口格式**（List[RoadBranchLLH]）。

设计 v3（按用户反馈：让 3 条候选分散到不同支路上）：

  * 共 K 条 branch，每条 = "共用主干 + 该候选专属支路"
  * 共用主干：origin → bend → fork_point
        - 沿"K 条候选终点的平均方向"走 ~30% 平均距离
        - 中间加 1 个偏移弯，模拟有过弯的引道
  * 专属支路 (针对 candidate k)：fork_point → mid_k → end_k  （+ 适度延长）
        - end_k = candidate k 的最后一帧 xy
        - mid_k = 在 fork_point 到 end_k 的中点附近横向偏移，制造一个转弯
  * 拼成完整 polyline 后做 _densify() 插点，让节点稠密、投影平滑

为什么这样能让 3 条候选分散：
  road_arc_projection 选 branch 时算"轨迹到该 branch 的总匹配代价"。每条
  candidate 的终点都正好落在自己专属那条 branch 的尾端，自然 cost 最低 →
  各自被分到不同 branch 上。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from constraint_optimizer.test_road_net.road_schema import (
    RoadBranchLLH,
    RoadNetwork,
    enu_km_to_llh,
)


# ============================================================
# 工具
# ============================================================

def _polyline_from_xy(
    xy: np.ndarray,                          # [N, 2]
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
    """在折线每两个角点之间均匀插点。"""
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
    """xy 平面的逆时针 90°。"""
    return np.array([-u[1], u[0]])


def _safe_unit(v: np.ndarray) -> Tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0]), 0.0
    return v / n, n


# ============================================================
# 主入口
# ============================================================

def build_road_network_for_sample(
    cand_xyz_km: np.ndarray,                 # [K, T, 3]  GNN1 选出的 top-K 候选
    origin_llh: Tuple[float, float, float] = (116.30, 39.90, 0.0),
    n_branches: int | None = None,           # 默认就用 K
    rng_seed: int = 0,
) -> RoadNetwork:
    """
    Y 形 / 多叉路网：每条候选一条专属支路。

    返回 K 条 RoadBranchLLH，第 k 条 = origin → trunk_bend → fork_point
                                       → mid_k → end_k → tail_k
    其中 end_k 取自 candidate k 的最后一帧 xy，tail_k 是再延长一点点防止候选
    走超尾。
    """
    rng = np.random.default_rng(rng_seed)

    if cand_xyz_km.ndim != 3 or cand_xyz_km.shape[-1] != 3:
        raise ValueError(f"cand_xyz_km 形状应为 [K, T, 3]，实际 {cand_xyz_km.shape}")

    K = int(cand_xyz_km.shape[0])
    if n_branches is None:
        n_branches = K
    n_branches = int(min(n_branches, K))

    p0 = np.array([0.0, 0.0])

    # 每条候选的终点 + 中点（用于"过弯"参考）
    endpoints = cand_xyz_km[:, -1, :2].astype(np.float64)        # [K, 2]
    mid_t = cand_xyz_km.shape[1] // 2
    midpoints = cand_xyz_km[:, mid_t, :2].astype(np.float64)     # [K, 2]

    # 共用主干方向 = K 条候选终点的平均方向
    avg_end = endpoints.mean(axis=0)
    u_main, L_main = _safe_unit(avg_end - p0)
    n_main = _perp(u_main)
    extent = max(L_main, 2.0)

    # ---- 共用主干：origin → trunk_bend → fork_point ----
    fork_dist = 0.30 * extent                                    # 30% 处分叉
    fork_point = p0 + fork_dist * u_main
    # 主干中段加一个小偏移弯，让主干不是死直
    trunk_bend = (
        p0 + 0.5 * fork_dist * u_main
        + 0.18 * fork_dist * n_main
    )

    # 路网平均高度 = 候选 z 平均
    z_mean = float(cand_xyz_km[..., 2].mean())

    # ---- 每条候选造一条专属支路 ----
    branches: RoadNetwork = []
    for k in range(n_branches):
        end_k = endpoints[k]
        u_k, L_k = _safe_unit(end_k - fork_point)
        n_k = _perp(u_k)

        # mid_k：fork_point 到 end_k 的中点 + 横向偏移（造个转弯）
        # 偏移方向交错，让相邻候选的弯弯方向不一样
        side_sign = 1.0 if (k % 2 == 0) else -1.0
        bend_amp = 0.18 * max(L_k, 0.5)
        # 用候选自身的 mid 作为参考做个混合，让弯更"真实"
        mid_k_geom = fork_point + 0.5 * (end_k - fork_point) + side_sign * bend_amp * n_k
        if midpoints.shape[0] > k:
            # 50% 几何中点 + 50% 候选自己中点（让支路更贴候选 mid）
            mid_k = 0.5 * mid_k_geom + 0.5 * midpoints[k]
        else:
            mid_k = mid_k_geom

        # tail_k：在 end_k 沿 u_k 再延 ~25%，避免候选走超尾导致夹取
        tail_k = end_k + 0.25 * max(L_k, 0.5) * u_k

        corners = np.stack(
            [p0, trunk_bend, fork_point, mid_k, end_k, tail_k],
            axis=0,
        )
        dense = _densify(corners, n_pts_per_seg=4)
        branches.append(_polyline_from_xy(dense, z_mean, origin_llh))

    _ = rng  # 当前不用，未来可加 jitter 时启用
    return branches


__all__ = ["build_road_network_for_sample"]
