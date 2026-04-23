"""
lstm2/code/data/dataset.py
--------------------------
LSTM2 的 Dataset（骨架占位）。

预期 __getitem__ 返回 dict:
    "hist_raw":    float32 [Tin, 6]
    "fut":         float32 [Tout, 6]    训练时可用 GT 未来，部署时用精修未来
    "intent_lbl":  int64 scalar         意图类别；-1 表示 ignore
    "threat":      float32 scalar       0..100；-1 表示 ignore

TODO: generate_trajs.py 产出后，在这里完成读取 + 归一化 + 切窗。
"""

from __future__ import annotations

from typing import Any, Dict


class Lstm2Dataset:
    def __init__(self, *args, **kwargs) -> None:
        self._placeholder = True

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("TODO: 等 generate_trajs.py 实现后补齐")


def build_datasets_from_config(config_path: str):
    raise NotImplementedError("TODO")
