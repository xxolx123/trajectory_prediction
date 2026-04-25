"""
constraint_optimizer/test_road_net/road_schema.py
-------------------------------------------------
镜像甲方 C++ 接口里的路网类型，并提供 LLH ↔ 局部 ENU(km) 的转换工具。

C++ 侧定义（来自甲方）::

    struct RoadPointLLH {
        double lon_deg = 0.0;
        double lat_deg = 0.0;
        double alt_m   = 0.0;
    };
    struct RoadBranchLLH {
        std::vector<RoadPointLLH> points;
    };
    using RoadPolyline  = std::vector<RoadPointLLH>;
    using RoadNetwork   = std::vector<RoadBranchLLH>;

本文件提供：
  - 与上面三个 C++ 类型一一对应的 Python dataclass / 类型别名
  - 将 RoadNetwork(LLH) 转成 ContextBatch 需要的张量
      road_points [1, NB_max, NP_max, 3]   km xyz, 局部 ENU
      road_mask   [1, NB_max, NP_max]      bool
  - 反向：局部 ENU(km) → RoadPointLLH（用于"造假路网"环节）

ENU 转换约定（与部署端一致）：
  - 给定参考原点 (lon0_deg, lat0_deg, alt0_m)
  - 用 flat-earth 球面近似（精度对几公里量级的路网完全够；deploy 实际可能用
    更精确的 WGS84，本测试 demo 阶段不需要）：
        dy_km = (lat - lat0) * deg2rad * R_earth
        dx_km = (lon - lon0) * deg2rad * R_earth * cos(lat0_rad)
        dz_km = (alt_m - alt0_m) / 1000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


# ===== C++ 镜像 =====================================================

@dataclass
class RoadPointLLH:
    lon_deg: float = 0.0
    lat_deg: float = 0.0
    alt_m: float = 0.0


@dataclass
class RoadBranchLLH:
    points: List[RoadPointLLH] = field(default_factory=list)


RoadPolyline = List[RoadPointLLH]       # 与 C++ using 对齐
RoadNetwork = List[RoadBranchLLH]       # 与 C++ using 对齐


# ===== LLH ↔ 局部 ENU(km) ===========================================

EARTH_R_KM = 6371.0  # 球面近似，足够用于路网级 demo


def llh_to_enu_km(
    lon_deg: float, lat_deg: float, alt_m: float,
    lon0_deg: float, lat0_deg: float, alt0_m: float,
) -> np.ndarray:
    """LLH → 局部 ENU (km)，flat-earth 近似。返回 [3] 数组 (x, y, z)。"""
    lat0_rad = np.deg2rad(lat0_deg)
    dy_km = float(np.deg2rad(lat_deg - lat0_deg)) * EARTH_R_KM
    dx_km = float(np.deg2rad(lon_deg - lon0_deg)) * EARTH_R_KM * float(np.cos(lat0_rad))
    dz_km = (alt_m - alt0_m) / 1000.0
    return np.array([dx_km, dy_km, dz_km], dtype=np.float64)


def enu_km_to_llh(
    x_km: float, y_km: float, z_km: float,
    lon0_deg: float, lat0_deg: float, alt0_m: float,
) -> RoadPointLLH:
    """局部 ENU(km) → LLH，flat-earth 近似。"""
    lat0_rad = np.deg2rad(lat0_deg)
    lat_deg = lat0_deg + float(np.rad2deg(y_km / EARTH_R_KM))
    lon_deg = lon0_deg + float(np.rad2deg(x_km / (EARTH_R_KM * float(np.cos(lat0_rad)))))
    alt_m = alt0_m + z_km * 1000.0
    return RoadPointLLH(lon_deg=lon_deg, lat_deg=lat_deg, alt_m=alt_m)


# ===== RoadNetwork(LLH) → 张量 =====================================

def road_network_to_tensors(
    road_network: RoadNetwork,
    origin_llh: Tuple[float, float, float],
    nb_max: int,
    np_max: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把 RoadNetwork(LLH) → ContextBatch 期望的两个张量：
      road_points [1, NB_max, NP_max, 3]  float32   km xyz
      road_mask   [1, NB_max, NP_max]     bool

    超出 NB_max / NP_max 的分支 / 点会被截断。
    """
    rp = np.zeros((1, nb_max, np_max, 3), dtype=np.float32)
    rm = np.zeros((1, nb_max, np_max), dtype=bool)

    for bi, branch in enumerate(road_network[:nb_max]):
        for pi, pt in enumerate(branch.points[:np_max]):
            v = llh_to_enu_km(pt.lon_deg, pt.lat_deg, pt.alt_m, *origin_llh)
            rp[0, bi, pi] = v.astype(np.float32)
            rm[0, bi, pi] = True

    return (
        torch.from_numpy(rp).to(device),
        torch.from_numpy(rm).to(device),
    )


# ===== 调试小工具 ===================================================

def road_network_summary(road_network: RoadNetwork) -> str:
    """打印路网摘要：分支数 / 每条分支点数 / LLH 范围。"""
    if not road_network:
        return "<empty road_network>"
    lines = [f"RoadNetwork: {len(road_network)} branches"]
    for bi, br in enumerate(road_network):
        if not br.points:
            lines.append(f"  branch {bi}: 0 points")
            continue
        lons = [p.lon_deg for p in br.points]
        lats = [p.lat_deg for p in br.points]
        alts = [p.alt_m for p in br.points]
        lines.append(
            f"  branch {bi}: {len(br.points)} points  "
            f"lon=[{min(lons):.5f}, {max(lons):.5f}]  "
            f"lat=[{min(lats):.5f}, {max(lats):.5f}]  "
            f"alt=[{min(alts):.1f}, {max(alts):.1f}] m"
        )
    return "\n".join(lines)
