"""
gnn1/code/data/dataset.py
-------------------------
GNN1 的 PyTorch Dataset：从 generate_data.py 产出的 data/raw/{split}.npz 加载样本。

__getitem__ 返回 dict（全部是 torch.Tensor）：
    cand_trajs:  float32 [M, Tout, D]     LSTM1 候选（归一化 + delta 空间）
    task_type:   long    ()               敌方作战任务（目前只有 0 = 打击）
    type:        long    ()               我方固定目标类型（0/1/2）
    position:    float32 [3]              我方固定目标 xyz km
    label:       long    ()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Gnn1Dataset(torch.utils.data.Dataset):
    """从一个 .npz 文件读样本。"""

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"找不到 {npz_path}；请先跑 cache_lstm1_preds.py + generate_data.py"
            )
        data = np.load(npz_path)

        required = ["candidates", "task_type", "type", "position", "label"]
        for k in required:
            if k not in data.files:
                raise KeyError(f"{npz_path} 缺少字段 {k}")

        self.candidates = data["candidates"].astype(np.float32)  # [N, M, T, D]
        self.task_type = data["task_type"].astype(np.int64)
        self.type = data["type"].astype(np.int64)
        self.position = data["position"].astype(np.float32)
        self.label = data["label"].astype(np.int64)

        # 可选字段
        self.k_seed: Optional[np.ndarray] = None
        if "k_seed" in data.files:
            self.k_seed = data["k_seed"].astype(np.int64)

        self.targets: Optional[np.ndarray] = None
        if "targets" in data.files:
            self.targets = data["targets"].astype(np.float32)

        self._n = int(self.candidates.shape[0])
        self._M = int(self.candidates.shape[1])
        self._T = int(self.candidates.shape[2])
        self._D = int(self.candidates.shape[3])

        assert self.task_type.shape == (self._n,), f"task_type shape 不对: {self.task_type.shape}"
        assert self.position.shape == (self._n, 3), f"position shape 不对: {self.position.shape}"

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "cand_trajs": torch.from_numpy(self.candidates[idx]),     # [M, T, D]
            "task_type":  torch.tensor(int(self.task_type[idx]), dtype=torch.long),
            "type":       torch.tensor(int(self.type[idx]),      dtype=torch.long),
            "position":   torch.from_numpy(self.position[idx]),       # [3]
            "label":      torch.tensor(int(self.label[idx]),     dtype=torch.long),
        }
        if self.k_seed is not None:
            item["k_seed"] = torch.tensor(int(self.k_seed[idx]), dtype=torch.long)
        if self.targets is not None:
            item["targets"] = torch.from_numpy(self.targets[idx])  # [Tout, D]
        return item

    # 一些方便的元信息
    @property
    def n_modes(self) -> int:
        return self._M

    @property
    def fut_len(self) -> int:
        return self._T

    @property
    def feat_dim(self) -> int:
        return self._D


# =============== 对外接口 ===============

def build_datasets_from_config(
    config_path: str = "config.yaml",
) -> Tuple[Gnn1Dataset, Gnn1Dataset, Gnn1Dataset]:
    """按 config.data.raw_dir 读取 train/val/test 三个 .npz。"""
    gnn1_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    cfg = _load_config(cfg_path)

    data_cfg = cfg.get("data", {})
    raw_dir = (gnn1_root / data_cfg.get("raw_dir", "data/raw")).resolve()

    train_ds = Gnn1Dataset(raw_dir / "train.npz")
    val_ds = Gnn1Dataset(raw_dir / "val.npz")
    test_ds = Gnn1Dataset(raw_dir / "test.npz")
    return train_ds, val_ds, test_ds
