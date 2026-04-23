"""
common/scaler.py
----------------
StandardScaler：所有子网络都可能要用的归一化工具。

各子网络（lstm1 / lstm2 / gnn1 / ...）训练时各自 fit 一个 scaler，
保存到自己目录下的 data/processed/*.npz；fusion 阶段会分别读取再拼到
FullNet 里使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class StandardScaler:
    """
    通用标准化器：对每一维做 (x - mean) / std。
    支持 [N, D] fit + 任意 [..., D] transform。
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"StandardScaler.fit 期望 [N, D]，实际 {data.shape}")
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未 fit")
        mean = self.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        std = self.std.reshape((1,) * (data.ndim - 1) + (-1,))
        return (data - mean) / std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未 fit")
        mean = self.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        std = self.std.reshape((1,) * (data.ndim - 1) + (-1,))
        return data * std + mean

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        arr = np.load(path)
        obj = cls()
        obj.mean = arr["mean"]
        obj.std = arr["std"]
        return obj


def load_mean_std_from_npz(path: Path):
    """
    兼容 old_plan 和 sklearn 的各种 npz 布局，返回 (mean, std) 两个 float32 数组。
    """
    arr = np.load(path)
    keys = set(arr.files)
    if "mean" in keys and "std" in keys:
        mean = arr["mean"]; std = arr["std"]
    elif "mean_" in keys and "scale_" in keys:
        mean = arr["mean_"]; std = arr["scale_"]
    elif "mu" in keys and "sigma" in keys:
        mean = arr["mu"]; std = arr["sigma"]
    else:
        raise ValueError(f"{path} 中找不到 mean/std 字段，实际 keys={keys}")
    return mean.astype(np.float32), std.astype(np.float32)
