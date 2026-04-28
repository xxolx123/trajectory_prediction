"""
gnn2/code/data/dataset.py
-------------------------
GNN2 的 PyTorch Dataset：从 generate_data.py 产出的 data/raw/{split}.npz 加载样本。

__getitem__ 返回 dict（全部是 torch.Tensor）::

    pred_traj:        float32 [T, 6]   top-K 候选的物理 6D 轨迹（hist_end=原点）
    eta:              long    ()       秒，∈ [eta_min_sec, eta_max_sec]
    gt_strike_pos:    float32 [3]      km xyz
    gt_strike_radius: float32 ()       km
    gt_strike_conf:   float32 ()       [0, 1]
    # 可选诊断字段（trainer 不读，只用于评测/可视化）
    scene_idx:        long    ()       场景在池中的全局索引（0..max_scenes-1）
    cand_k:           long    ()       top-K 内的位置 0..K-1（按 GNN1 概率降序）
    eta_idx:          long    ()       同一 (scene, k) 的第几个 eta 副本（0..n_eta-1）
    gnn1_top_idx:     long    ()       0..M_lstm1-1 内的索引
    top_prob:         float32 ()       GNN1 给该候选的归一化 top-K 概率
    src_split:        long    ()       源自哪个 gnn1 split（0=train, 1=val, 2=test）
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


class Gnn2Dataset(torch.utils.data.Dataset):
    """从一个 .npz 文件读样本。"""

    def __init__(self, npz_path: Path) -> None:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"找不到 {npz_path}；请先在 gnn2/ 下跑 "
                "`python -m data.generate_data --config config.yaml`"
            )
        data = np.load(npz_path)

        required = [
            "pred_traj",
            "eta",
            "gt_strike_pos",
            "gt_strike_radius",
            "gt_strike_conf",
        ]
        for k in required:
            if k not in data.files:
                raise KeyError(f"{npz_path} 缺少字段 {k}")

        self.pred_traj = data["pred_traj"].astype(np.float32)              # [N, T, 6]
        self.eta = data["eta"].astype(np.int64)                            # [N]
        self.gt_strike_pos = data["gt_strike_pos"].astype(np.float32)      # [N, 3]
        self.gt_strike_radius = data["gt_strike_radius"].astype(np.float32)  # [N]
        self.gt_strike_conf = data["gt_strike_conf"].astype(np.float32)    # [N]

        # 可选诊断字段（当前 generate_data.py 产出）
        self.scene_idx: Optional[np.ndarray] = None
        if "scene_idx" in data.files:
            self.scene_idx = data["scene_idx"].astype(np.int64)
        self.cand_k: Optional[np.ndarray] = None
        if "cand_k" in data.files:
            self.cand_k = data["cand_k"].astype(np.int64)
        self.eta_idx: Optional[np.ndarray] = None
        if "eta_idx" in data.files:
            self.eta_idx = data["eta_idx"].astype(np.int64)
        self.gnn1_top_idx: Optional[np.ndarray] = None
        if "gnn1_top_idx" in data.files:
            self.gnn1_top_idx = data["gnn1_top_idx"].astype(np.int64)
        self.top_prob: Optional[np.ndarray] = None
        if "top_prob" in data.files:
            self.top_prob = data["top_prob"].astype(np.float32)
        self.src_split: Optional[np.ndarray] = None
        if "src_split" in data.files:
            self.src_split = data["src_split"].astype(np.int64)

        self._n = int(self.pred_traj.shape[0])
        self._T = int(self.pred_traj.shape[1])
        self._D = int(self.pred_traj.shape[2])

        if self.eta.shape != (self._n,):
            raise ValueError(f"eta shape {self.eta.shape} != ({self._n},)")
        if self.gt_strike_pos.shape != (self._n, 3):
            raise ValueError(
                f"gt_strike_pos shape {self.gt_strike_pos.shape} != ({self._n}, 3)"
            )
        if self.gt_strike_radius.shape != (self._n,):
            raise ValueError(
                f"gt_strike_radius shape {self.gt_strike_radius.shape} != ({self._n},)"
            )
        if self.gt_strike_conf.shape != (self._n,):
            raise ValueError(
                f"gt_strike_conf shape {self.gt_strike_conf.shape} != ({self._n},)"
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "pred_traj":        torch.from_numpy(self.pred_traj[idx]),         # [T, 6]
            "eta":              torch.tensor(int(self.eta[idx]), dtype=torch.long),
            "gt_strike_pos":    torch.from_numpy(self.gt_strike_pos[idx]),     # [3]
            "gt_strike_radius": torch.tensor(
                float(self.gt_strike_radius[idx]), dtype=torch.float32
            ),
            "gt_strike_conf":   torch.tensor(
                float(self.gt_strike_conf[idx]), dtype=torch.float32
            ),
        }
        if self.scene_idx is not None:
            item["scene_idx"] = torch.tensor(
                int(self.scene_idx[idx]), dtype=torch.long,
            )
        if self.cand_k is not None:
            item["cand_k"] = torch.tensor(
                int(self.cand_k[idx]), dtype=torch.long,
            )
        if self.eta_idx is not None:
            item["eta_idx"] = torch.tensor(
                int(self.eta_idx[idx]), dtype=torch.long,
            )
        if self.gnn1_top_idx is not None:
            item["gnn1_top_idx"] = torch.tensor(
                int(self.gnn1_top_idx[idx]), dtype=torch.long,
            )
        if self.top_prob is not None:
            item["top_prob"] = torch.tensor(
                float(self.top_prob[idx]), dtype=torch.float32,
            )
        if self.src_split is not None:
            item["src_split"] = torch.tensor(
                int(self.src_split[idx]), dtype=torch.long,
            )
        return item

    @property
    def fut_len(self) -> int:
        return self._T

    @property
    def feat_dim(self) -> int:
        return self._D


# =============== 对外接口 ===============

def build_datasets_from_config(
    config_path: str = "config.yaml",
) -> Tuple[Gnn2Dataset, Gnn2Dataset, Gnn2Dataset]:
    """按 config.data.raw_dir 读取 train/val/test 三个 .npz。"""
    gnn2_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = gnn2_root / cfg_path
    cfg = _load_config(cfg_path)

    data_cfg = cfg.get("data", {}) or {}
    raw_dir = (gnn2_root / data_cfg.get("raw_dir", "data/raw")).resolve()

    train_ds = Gnn2Dataset(raw_dir / "train.npz")
    val_ds = Gnn2Dataset(raw_dir / "val.npz")
    test_ds = Gnn2Dataset(raw_dir / "test.npz")
    return train_ds, val_ds, test_ds
