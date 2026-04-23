"""
traj_dataset.py
---------------
从 synthetic_trajectories.csv 中构造用于序列预测的滑动窗口数据集，并支持：
  - 绝对坐标 → 增量坐标（仅对位置 x,y,z 做差分）
  - 特征归一化（StandardScaler）
  - 按轨迹划分 train/val/test

约定：
  1) 生成轨迹的脚本 generate_trajs.py 已经根据 config.yaml 在
     data.raw_dir / data.output_csv 生成了 CSV 文件；
  2) CSV 至少包含以下列：
       - traj_id
       - step_idx
       - x_km, y_km, z_km
       - vx_kmps, vy_kmps, vz_kmps
  3) 输入/输出特征统一为 6 维：
       [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]

config.yaml 中推荐的数据相关字段示例：

data:
  raw_dir: "data/raw"
  output_csv: "synthetic_trajectories.csv"

  processed_dir: "data/processed"
  scaler_filename: "scaler_posvel.npz"

  in_len: 20
  out_len: 10
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

  use_delta: true       # 是否把位置转成增量
  normalize: true       # 是否做标准化

"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml


FEATURE_COLS = ["x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps"]
POS_DIM_INDEX = [0, 1, 2]  # 在 6 维特征中，前 3 维是位置


# ===================== 工具类：StandardScaler =====================


class StandardScaler:
    """
    简单版标准化器：对每一维做 (x - mean) / std

    - fit(data): data 形状 [N, D]
    - transform(data) / inverse_transform(data): 支持 [N, D] 或 [*, D]
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"StandardScaler.fit 期望 [N, D]，收到形状 {data.shape}")
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        # 避免除以 0
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

    # 保存/加载到 npz，方便推理阶段复用
    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler 尚未 fit，无法保存")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        arr = np.load(path)
        scaler = cls()
        scaler.mean = arr["mean"]
        scaler.std = arr["std"]
        return scaler


# ===================== 数据集类 =====================


class TrajectoryWindowDataset:
    """
    滑动窗口数据集：
      - inputs:  [N, in_len, D]
      - targets: [N, out_len, D]
    D 默认是 6（x,y,z,vx,vy,vz）

    __getitem__(idx) 返回 (inputs[idx], targets[idx])，类型为 np.ndarray。
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        scaler: Optional[StandardScaler] = None,
    ):
        assert inputs.ndim == 3 and targets.ndim == 3, "inputs/targets 形状应为 [N, L, D]"
        assert inputs.shape[0] == targets.shape[0], "inputs 和 targets 的样本数不一致"

        self.inputs = inputs.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.scaler = scaler  # 只是引用，方便外部使用

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


# ===================== 辅助函数：读取 config / CSV =====================


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """假定本文件在 project_root/code/data/traj_dataset.py"""
    return Path(__file__).resolve().parents[2]


def load_raw_trajectories(cfg: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    output_csv = data_cfg.get("output_csv", "synthetic_trajectories.csv")

    raw_dir_path = (project_root / raw_dir).resolve()
    csv_path = raw_dir_path / output_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到轨迹 CSV：{csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["traj_id", "step_idx"] + FEATURE_COLS
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"轨迹 CSV 缺少列：{col}")
    return df


# ===================== 核心：构造滑动窗口 =====================


def _make_windows_for_ids(
    df: pd.DataFrame,
    traj_ids: List[int],
    in_len: int,
    out_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对指定的 traj_ids 生成滑动窗口：
      - 先对每个 traj_id 按 step_idx 排序
      - 再做滑窗：start 从 0 到 T - in_len - out_len
    返回：
      inputs:  [N, in_len, D]
      targets: [N, out_len, D]
    """
    input_seqs: List[np.ndarray] = []
    target_seqs: List[np.ndarray] = []

    grouped = df.groupby("traj_id")

    for tid in traj_ids:
        if tid not in grouped.indices:
            continue
        sub = grouped.get_group(tid).sort_values("step_idx")
        feats = sub[FEATURE_COLS].to_numpy(dtype=np.float32)  # [T, D]
        T, D = feats.shape

        max_start = T - in_len - out_len
        if max_start < 0:
            # 这条轨迹太短，跳过
            continue

        for start in range(max_start + 1):
            inp = feats[start : start + in_len]
            out = feats[start + in_len : start + in_len + out_len]
            input_seqs.append(inp)
            target_seqs.append(out)

    if not input_seqs:
        raise RuntimeError("没有生成任何滑动窗口，请检查 in_len/out_len 与轨迹长度")

    inputs = np.stack(input_seqs, axis=0)
    targets = np.stack(target_seqs, axis=0)
    return inputs, targets


def _apply_delta_inplace(
    inputs: np.ndarray,
    targets: np.ndarray,
    pos_indices: List[int] = POS_DIM_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    就地把位置 x,y,z 转为“相邻时间步的增量”(delta)，速度保持不变。

    规则（对每个样本单独处理）：
      - 把 [in_len + out_len] 个位置拼在一起做一次差分：
          all_pos = [in_pos; out_pos]  →  shape = [L_all, 3]
          delta = all_pos[1:] - all_pos[:-1]  → [L_all-1, 3]
      - 在最前面插入一行 0，使得长度恢复 L_all：
          delta_full = [0; delta]  → [L_all, 3]
      - 前 in_len 行作为 input 的位置增量，
        后 out_len 行作为 target 的位置增量。
      - 速度分量不变。
    """
    B, in_len, D = inputs.shape
    _, out_len, _ = targets.shape

    for b in range(B):
        # 先取出该样本的序列视图
        in_seq = inputs[b]      # [in_len, D]
        out_seq = targets[b]    # [out_len, D]

        in_pos = in_seq[:, pos_indices]      # [in_len, P]
        out_pos = out_seq[:, pos_indices]    # [out_len, P]

        all_pos = np.concatenate([in_pos, out_pos], axis=0)  # [L_all, P]
        delta = np.diff(all_pos, axis=0)                     # [L_all-1, P]
        delta_full = np.vstack(
            [np.zeros((1, delta.shape[1]), dtype=delta.dtype), delta]
        )  # [L_all, P]

        # 写回到原数组（in_seq/out_seq 是视图，修改会反映到 inputs/targets）
        in_seq[:, pos_indices] = delta_full[:in_len]
        out_seq[:, pos_indices] = delta_full[in_len : in_len + out_len]

    return inputs, targets


# ===================== 对外接口：构建数据集和 scaler =====================


def build_datasets_from_config(
    config_path: str = "config.yaml",
) -> Tuple[TrajectoryWindowDataset, TrajectoryWindowDataset, TrajectoryWindowDataset, Optional[StandardScaler]]:
    """
    从 config.yaml 构建 train/val/test 三个数据集，并返回一个 scaler（若 normalize=True）。

    返回：
      train_ds, val_ds, test_ds, scaler
    """
    project_root = get_project_root()

    cfg = load_config(project_root / config_path)
    data_cfg = cfg.get("data", {})

    in_len = int(data_cfg.get("in_len", 20))
    out_len = int(data_cfg.get("out_len", 10))

    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))

    use_delta = bool(data_cfg.get("use_delta", True))
    normalize = bool(data_cfg.get("normalize", True))

    processed_dir = data_cfg.get("processed_dir", "data/processed")
    scaler_filename = data_cfg.get("scaler_filename", "scaler_posvel.npz")
    processed_dir_path = (project_root / processed_dir).resolve()
    scaler_path = processed_dir_path / scaler_filename

    # 1) 读原始 CSV
    df = load_raw_trajectories(cfg, project_root)

    # 2) 根据 traj_id 划分 train/val/test
    traj_ids = sorted(df["traj_id"].unique().tolist())
    n_total = len(traj_ids)
    if n_total == 0:
        raise RuntimeError("CSV 中没有任何 traj_id")

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    # 其余全部给 test，避免加和误差
    n_test = max(1, n_total - n_train - n_val)
    if n_train + n_val + n_test > n_total:
        # 如果因为 max(1, ...) 导致超过了，就微调一下
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            n_test = max(1, n_total - n_train)

    train_ids = traj_ids[:n_train]
    val_ids = traj_ids[n_train : n_train + n_val]
    test_ids = traj_ids[n_train + n_val : n_train + n_val + n_test]

    # 3) 先构造“未增量、未归一化”的窗口
    train_in, train_out = _make_windows_for_ids(df, train_ids, in_len, out_len)
    val_in, val_out = _make_windows_for_ids(df, val_ids, in_len, out_len)
    test_in, test_out = _make_windows_for_ids(df, test_ids, in_len, out_len)

    # 4) 根据配置，把位置转换成增量
    if use_delta:
        train_in, train_out = _apply_delta_inplace(train_in, train_out)
        val_in, val_out = _apply_delta_inplace(val_in, val_out)
        test_in, test_out = _apply_delta_inplace(test_in, test_out)

    # 5) 如果需要归一化：只用 train 的（input+target）来 fit scaler
    scaler: Optional[StandardScaler] = None
    if normalize:
        scaler = StandardScaler()
        # 拼成 [N * (in_len+out_len), D]
        train_all = np.concatenate(
            [
                train_in.reshape(-1, train_in.shape[-1]),
                train_out.reshape(-1, train_out.shape[-1]),
            ],
            axis=0,
        )
        scaler.fit(train_all)

        # 对三个集合分别做 transform
        train_in = scaler.transform(train_in)
        train_out = scaler.transform(train_out)
        val_in = scaler.transform(val_in)
        val_out = scaler.transform(val_out)
        test_in = scaler.transform(test_in)
        test_out = scaler.transform(test_out)

        # 保存 scaler 以便推理/可视化时反归一化
        scaler.save(scaler_path)

    # 6) 构造数据集对象
    train_ds = TrajectoryWindowDataset(train_in, train_out, scaler=scaler)
    val_ds = TrajectoryWindowDataset(val_in, val_out, scaler=scaler)
    test_ds = TrajectoryWindowDataset(test_in, test_out, scaler=scaler)

    return train_ds, val_ds, test_ds, scaler
