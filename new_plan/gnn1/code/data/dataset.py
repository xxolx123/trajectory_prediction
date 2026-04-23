"""
gnn1/code/data/dataset.py
-------------------------
GNN1 的 Dataset 封装（骨架占位）。

预期 __getitem__ 返回 dict：
    "cand_trajs":    float32 [M, T, 6]           LSTM1 输出的 M 条候选
    "target_task":   float32 [D_task]
    "fixed_targets": float32 [N_tgt, D_tgt]
    "target_type":   float32 [D_type]
    "road_network":  float32 [D_road]
    "label":         int64 scalar                 正类 mode 的索引（0..M-1）
                     （或 soft label: float32 [M]）

TODO:
    - 从 data/raw/ 读入 generate_data.py 产出的样本
    - 复用 common.scaler.StandardScaler 做归一化（若有必要）
"""

from __future__ import annotations

from typing import Any, Dict


class Gnn1Dataset:
    """
    最小占位版：__len__ 返回 0，__getitem__ 抛 NotImplementedError。
    真实实现请替换这里。
    """

    def __init__(self, *args, **kwargs) -> None:
        # TODO: 从 raw/ 读文件；或接受内存中的样本列表
        self._placeholder = True

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("TODO: 等 generate_data.py 实现后，补 Dataset 读取")


def build_datasets_from_config(config_path: str):
    """
    TODO: 返回 (train_ds, val_ds, test_ds)。
    """
    _ = config_path
    raise NotImplementedError("TODO")
