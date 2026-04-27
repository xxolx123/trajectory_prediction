"""
lstm2/code/data/dataset.py
--------------------------
LSTM2 的 Dataset。

数据来源：
    lstm2/data/raw/{split}.npz   ← 由 generate_trajs.py 离线产出
    每行（M = N_s × K=3）：
        hist_raw     [20, 6]    物理 km / km/s    （仅可视化用）
        fut_refined  [10, 6]    物理（约束优化后）  ← 模型主输入
        position     [3]        物理 km            ← 模型条件输入
        intent_label int8       0..3
        threat_score int16      0..100
        fut_gt       [10, 6]    物理（GT 未来；可选）
        sample_idx   int32      原 gnn1 sample 索引（同一 sample → 同一 hist）
        cand_k       int8       0..K-1
        topology     str        路网拓扑名（调试）
        ...

每个样本独立处理（per-候选输出 intent + threat），即：
    fut_norm: float32 [10, 11]   工程化 + StandardScaler
    position: float32 [3]        物理（不归一化，给可视化）
    intent:   int64               0..3
    threat:   float32             0..100
    （以及调试用的 hist_phys / fut_phys / fut_gt_phys / sample_idx / cand_k）

特征工程在 dataset 端做，模型端不再重复（model 里 _maybe_make_features 检测
最后一维：6 → 自动 engineer，11 → 跳过）。

scaler：在 train 集上 fit StandardScaler（11 维），保存到
    data.processed_dir / data.scaler_filename
val/test 复用，避免数据泄漏。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from torch.utils.data import Dataset


# ============================================================
# 标准化
# ============================================================

class StandardScaler:
    """
    简易 StandardScaler，复刻 lstm1 的实现，保证可 npz 持久化。
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"fit 期望 [N, D]，实际 {data.shape}")
        self.mean = data.mean(axis=0).astype(np.float64)
        self.std = data.std(axis=0).astype(np.float64)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 未 fit")
        m = self.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (data.ndim - 1) + (-1,))
        return (data - m) / s

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 未 fit")
        m = self.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        s = self.std.reshape((1,) * (data.ndim - 1) + (-1,))
        return data * s + m

    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 未 fit，无法保存")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean.astype(np.float32),
                 std=self.std.astype(np.float32))

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        a = np.load(path)
        s = cls()
        s.mean = a["mean"].astype(np.float64)
        s.std = a["std"].astype(np.float64)
        s.std[s.std < 1e-9] = 1.0
        return s


# ============================================================
# 工程化特征（numpy 版本）
# ============================================================

def engineer_features_np(
    fut: np.ndarray,           # [N, T, 6]
    position: np.ndarray,      # [N, 3]
) -> np.ndarray:
    """
    把 (fut [N,T,6], position [N,3]) 扩展到 [N,T,11]：
        [x, y, z, vx, vy, vz, dx, dy, dz, ||Δ||, ||v||]
    与 model.engineer_features 对齐，但走 numpy。
    """
    if fut.ndim != 3 or fut.shape[-1] != 6:
        raise ValueError(f"fut 形状应为 [N,T,6]，实际 {fut.shape}")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position 形状应为 [N,3]，实际 {position.shape}")

    pos = fut[..., 0:3]                              # [N, T, 3]
    vel = fut[..., 3:6]                              # [N, T, 3]
    delta = pos - position[:, None, :]               # [N, T, 3]
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)   # [N, T, 1]
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)    # [N, T, 1]
    return np.concatenate([fut, delta, dist, speed], axis=-1).astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class Lstm2Dataset(Dataset):
    """
    Per-候选独立样本：M = N_s × K=3。

    Args:
        npz_path: lstm2/data/raw/{split}.npz
        scaler:   已 fit 好的 StandardScaler（在 11 维上拟合的）
        cache_in_memory: 默认 True，把 npz 全量预处理到内存
    """

    def __init__(
        self,
        npz_path: Path,
        scaler: StandardScaler,
        cache_in_memory: bool = True,
    ) -> None:
        super().__init__()
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"未找到 lstm2 npz：{npz_path}")
        self._path = Path(npz_path)
        self.scaler = scaler

        d = np.load(self._path, allow_pickle=True)

        fut = d["fut_refined"].astype(np.float32)            # [M, T, 6]
        pos = d["position"].astype(np.float32)               # [M, 3]
        intent = d["intent_label"].astype(np.int64)          # [M]
        threat = d["threat_score"].astype(np.float32)        # [M]
        hist = d["hist_raw"].astype(np.float32)              # [M, 20, 6]

        sample_idx = d["sample_idx"].astype(np.int64) if "sample_idx" in d.files \
            else np.arange(fut.shape[0], dtype=np.int64)
        cand_k = d["cand_k"].astype(np.int64) if "cand_k" in d.files \
            else np.zeros(fut.shape[0], dtype=np.int64)
        fut_gt = d["fut_gt"].astype(np.float32) if "fut_gt" in d.files else None
        topology = d["topology"].astype(str) if "topology" in d.files else None

        # 工程化 + 归一化（一次性全量做完，吃内存换速度）
        feat11 = engineer_features_np(fut, pos)              # [M, T, 11]
        feat11_norm = scaler.transform(feat11.astype(np.float64)).astype(np.float32)

        if cache_in_memory:
            self._fut_norm = feat11_norm                     # [M, T, 11]
            self._fut_phys = fut                             # [M, T, 6]   保留物理量给可视化
            self._pos = pos                                  # [M, 3]
            self._intent = intent
            self._threat = threat
            self._hist = hist
            self._sample_idx = sample_idx
            self._cand_k = cand_k
            self._fut_gt = fut_gt
            self._topology = topology
        else:
            raise NotImplementedError("当前实现仅支持 cache_in_memory=True")

    # 调试用属性
    @property
    def intent_labels(self) -> np.ndarray:
        return self._intent

    @property
    def threat_scores(self) -> np.ndarray:
        return self._threat

    def __len__(self) -> int:
        return int(self._fut_norm.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "fut_norm":    self._fut_norm[idx],         # [T, 11]   归一化 + 工程化
            "position":    self._pos[idx],              # [3]       物理（仅供可视化）
            "intent":      self._intent[idx],           # int64
            "threat":      self._threat[idx],           # float32
            "hist_phys":   self._hist[idx],             # [20, 6]   物理（可视化）
            "fut_phys":    self._fut_phys[idx],         # [10, 6]   物理（可视化）
            "sample_idx":  self._sample_idx[idx],
            "cand_k":      self._cand_k[idx],
        }
        if self._fut_gt is not None:
            item["fut_gt"] = self._fut_gt[idx]
        if self._topology is not None:
            item["topology"] = str(self._topology[idx])
        return item


# ============================================================
# 工厂
# ============================================================

def _resolve_rel(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets_from_config(
    config_path: str,
) -> Tuple[Lstm2Dataset, Lstm2Dataset, Lstm2Dataset, StandardScaler]:
    """
    按 lstm2/config.yaml 构造 train/val/test 三个 dataset，并返回 scaler。

    流程：
      1) 读取 config，解析 raw_dir / processed_dir / scaler_filename
      2) 必备 train.npz 存在；val.npz / test.npz 缺失则抛错
      3) 用 train.npz fit StandardScaler（11 维），保存到 processed_dir
         若 processed_dir 已有 scaler 文件，则**直接复用**（避免重复 fit）
      4) 三个 split 共享 scaler 构造 dataset
    """
    lstm2_root = Path(__file__).resolve().parents[2]   # .../new_plan/lstm2
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (lstm2_root / cfg_path).resolve()
    cfg = _load_yaml(cfg_path)

    data_cfg = cfg.get("data", {}) or {}
    raw_dir = _resolve_rel(data_cfg.get("raw_dir", "data/raw"), lstm2_root)
    proc_dir = _resolve_rel(data_cfg.get("processed_dir", "data/processed"),
                            lstm2_root)
    scaler_name = str(data_cfg.get("scaler_filename", "scaler_intent_posvel.npz"))
    scaler_path = proc_dir / scaler_name

    train_npz = raw_dir / "train.npz"
    val_npz = raw_dir / "val.npz"
    test_npz = raw_dir / "test.npz"

    if not train_npz.exists():
        raise FileNotFoundError(
            f"缺 {train_npz}；请先在 lstm2/ 下跑 "
            f"`python -m data.generate_trajs --splits train val test`"
        )

    # ---- 加载 / 拟合 scaler ----
    if scaler_path.exists():
        scaler = StandardScaler.load(scaler_path)
        print(f"[Lstm2Dataset] scaler 复用：{scaler_path}")
        if scaler.mean is None or scaler.std is None:
            raise RuntimeError("scaler 加载异常")
    else:
        print(f"[Lstm2Dataset] 在 train.npz 上 fit StandardScaler ...")
        d = np.load(train_npz, allow_pickle=True)
        fut_train = d["fut_refined"].astype(np.float32)      # [M, T, 6]
        pos_train = d["position"].astype(np.float32)         # [M, 3]
        feat = engineer_features_np(fut_train, pos_train)    # [M, T, 11]
        flat = feat.reshape(-1, feat.shape[-1])              # [M*T, 11]
        scaler = StandardScaler()
        scaler.fit(flat.astype(np.float64))
        scaler.save(scaler_path)
        print(f"[Lstm2Dataset] scaler 已保存：{scaler_path}")
        del d, fut_train, pos_train, feat, flat

    # ---- 构造 dataset ----
    train_ds = Lstm2Dataset(train_npz, scaler)
    val_ds = (
        Lstm2Dataset(val_npz, scaler)
        if val_npz.exists()
        else _empty_dataset(scaler)
    )
    test_ds = (
        Lstm2Dataset(test_npz, scaler)
        if test_npz.exists()
        else _empty_dataset(scaler)
    )
    if not val_npz.exists():
        print(f"[Lstm2Dataset][WARN] 缺 {val_npz}，val_ds 长度=0")
    if not test_npz.exists():
        print(f"[Lstm2Dataset][WARN] 缺 {test_npz}，test_ds 长度=0")

    return train_ds, val_ds, test_ds, scaler


def _empty_dataset(scaler: StandardScaler) -> "Lstm2Dataset":
    """返回一个 size=0 的占位 Dataset（用于 val/test 缺失场景）。"""
    ds = Lstm2Dataset.__new__(Lstm2Dataset)
    ds.scaler = scaler
    ds._path = Path("/dev/null")
    ds._fut_norm = np.zeros((0, 1, 11), dtype=np.float32)
    ds._fut_phys = np.zeros((0, 1, 6), dtype=np.float32)
    ds._pos = np.zeros((0, 3), dtype=np.float32)
    ds._intent = np.zeros((0,), dtype=np.int64)
    ds._threat = np.zeros((0,), dtype=np.float32)
    ds._hist = np.zeros((0, 1, 6), dtype=np.float32)
    ds._sample_idx = np.zeros((0,), dtype=np.int64)
    ds._cand_k = np.zeros((0,), dtype=np.int64)
    ds._fut_gt = None
    ds._topology = None
    return ds


__all__ = [
    "StandardScaler",
    "engineer_features_np",
    "Lstm2Dataset",
    "build_datasets_from_config",
]
