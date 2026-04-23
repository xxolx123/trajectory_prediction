"""
gnn2/code/data/dataset.py
-------------------------
GNN2 的 Dataset（骨架占位）。

预期 __getitem__ 返回：
    pred_traj:    [T, 6]
    context 各字段
    intent_feat:  [D_intent]            LSTM2 的意图特征
    gt_strike:    dict                  {"pos":[3], "radius":[1], "conf":[1]}

TODO：待数据标注/生成方案定了再实现。
"""

from __future__ import annotations

from typing import Any, Dict


class Gnn2Dataset:
    def __init__(self, *args, **kwargs) -> None:
        self._placeholder = True

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("TODO")


def build_datasets_from_config(config_path: str):
    raise NotImplementedError("TODO")
