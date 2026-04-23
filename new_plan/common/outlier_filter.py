"""
common/outlier_filter.py
------------------------
历史轨迹的异常值剔除（骨架占位）。

当前实现：pass（直接返回输入）
TODO:
    - 基于物理约束（速度/加速度上界）标记异常点
    - 基于 3-sigma / IQR 的统计型剔除
    - 基于 Kalman / 平滑滤波的插值修复

接口：
    输入:  traj: numpy.ndarray 或 torch.Tensor，形状 [B, T, D]
    输出:
        clean_traj: 同形状、同类型
        keep_mask: [B, T] bool；当前永远全 True
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

try:
    import torch
    _Tensor = torch.Tensor
except ImportError:
    torch = None  # type: ignore
    _Tensor = None  # type: ignore


ArrayLike = Union[np.ndarray, "_Tensor"]


def remove_outliers(traj: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """异常值剔除：当前 pass，后续补逻辑。"""
    if torch is not None and isinstance(traj, torch.Tensor):
        keep_mask = torch.ones(traj.shape[:2], dtype=torch.bool, device=traj.device)
        return traj, keep_mask

    traj_np = np.asarray(traj)
    keep_mask = np.ones(traj_np.shape[:2], dtype=bool)
    return traj_np, keep_mask


class OutlierFilter:
    """可注入的 Outlier 剔除器；当前是无参 pass 版。"""

    def __init__(self) -> None:
        # TODO: 从 config 接受可调阈值
        pass

    def __call__(self, traj: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        return remove_outliers(traj)
