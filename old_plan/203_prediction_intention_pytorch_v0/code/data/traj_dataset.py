#!/usr/bin/env python3
"""
traj_dataset.py (for intent & threat prediction)
-----------------------------------------------
从 synthetic_trajectories.csv 中构造“意图 + 威胁度”预测的数据集。

特点：
  - 使用 generate_trajs.py 生成的 CSV（已包含 intent_label, threat_score）
  - 以最近 window_len 步 (默认 10) 的轨迹窗口作为输入：
        X: [window_len, 6] = [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]
        y_intent:   标量，枚举 {0=ATTACK,1=EVASION,2=DEFENSE,3=RETREAT}
        y_threat:   标量，0~100 的整数（威胁度）
  - 只使用 CSV 中 intent_label >= 0 的行作为样本（即窗口已满且有有效标签）
  - 可以选择：
        * use_delta:   是否将 (x,y,z) 换成相邻时间步的增量
        * normalize:   是否对 6 维特征做 StandardScaler 标准化
  - 按 traj_id 划分 train/val/test，避免轨迹泄漏；划分前会随机打乱 traj_id。
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
        # 支持任意维度 [..., D]
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


class IntentThreatWindowDataset:
    """
    意图 + 威胁度 滑动窗口数据集：

      - inputs: [N, window_len, D]  (D=6)
      - intent_labels: [N]         (int64)
      - threat_scores: [N]         (float32 or int, 0~100)

    __getitem__(idx) 返回:
      (inputs[idx], intent_labels[idx], threat_scores[idx])
      其中 inputs[idx] 是 np.float32，[window_len, 6]
    """

    def __init__(
        self,
        inputs: np.ndarray,
        intent_labels: np.ndarray,
        threat_scores: np.ndarray,
        scaler: Optional[StandardScaler] = None,
    ):
        assert inputs.ndim == 3, "inputs 形状应为 [N, L, D]"
        assert inputs.shape[0] == intent_labels.shape[0] == threat_scores.shape[0], \
            "inputs / intent_labels / threat_scores 样本数不一致"

        self.inputs = inputs.astype(np.float32)
        self.intent_labels = intent_labels.astype(np.int64)
        # 威胁度可以用 float 来做回归，也可以在网络里 cast 回 int
        self.threat_scores = threat_scores.astype(np.float32)
        self.scaler = scaler  # 只是引用，方便外部使用

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.intent_labels[idx], self.threat_scores[idx]


# ===================== 辅助函数：读取 config / CSV =====================


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """假定本文件在 project_root/code/data/traj_dataset.py"""
    return Path(__file__).resolve().parents[2]


def load_raw_trajectories(cfg: Dict[str, Any], project_root: Path) -> pd.DataFrame:
    """
    读取 synthetic_trajectories.csv，并检查必要列是否存在。
    这里除了 FEATURE_COLS，还要求有 intent_label, threat_score。
    """
    data_cfg = cfg.get("data", {})

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    output_csv = data_cfg.get("output_csv", "synthetic_trajectories.csv")

    raw_dir_path = (project_root / raw_dir).resolve()
    csv_path = raw_dir_path / output_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到轨迹 CSV：{csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["traj_id", "step_idx"] + FEATURE_COLS + ["intent_label", "threat_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"轨迹 CSV 缺少列：{col}")
    return df


# ===================== 核心：构造“窗口 → (意图, 威胁)” =====================


def _make_intent_windows_for_ids(
    df: pd.DataFrame,
    traj_ids: List[int],
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对指定的 traj_ids 生成意图/威胁度的滑动窗口：

    规则：
      - 对每个 traj_id 按 step_idx 排序；
      - 只使用 intent_label >= 0 的行（表示这一行对应的窗口已经满了）；
      - 假设当前行 step_idx = t，intent_label >= 0，
        则窗口为 [t - window_len + 1, ..., t] 这 window_len 步的 6 维特征；
      - 标签：
          intent_labels = 当前行的 intent_label
          threat_scores = 当前行的 threat_score

    返回：
      inputs:        [N, window_len, D]
      intent_labels: [N]
      threat_scores: [N]
    """
    input_seqs: List[np.ndarray] = []
    intent_labels: List[int] = []
    threat_scores: List[float] = []

    grouped = df.groupby("traj_id")

    for tid in traj_ids:
        if tid not in grouped.indices:
            continue
        sub = grouped.get_group(tid).sort_values("step_idx")
        feats = sub[FEATURE_COLS].to_numpy(dtype=np.float32)  # [T, D]
        intents = sub["intent_label"].to_numpy()
        threats = sub["threat_score"].to_numpy()
        T, D = feats.shape

        # 找到所有 intent_label >= 0 的索引
        valid_indices = np.where(intents >= 0)[0]
        if valid_indices.size == 0:
            continue

        for idx in valid_indices:
            # 窗口末尾 = idx，窗口长度=window_len
            start = idx - window_len + 1
            if start < 0:
                # 理论上不会发生（因为生成时只有步数>=window_len-1 才会有 label）
                continue
            end = idx + 1  # Python 切片右开区间
            window = feats[start:end]  # [window_len, D]
            if window.shape[0] != window_len:
                # 安全检查
                continue

            input_seqs.append(window)
            intent_labels.append(int(intents[idx]))
            threat_scores.append(float(threats[idx]))

    if not input_seqs:
        raise RuntimeError("没有生成任何意图/威胁度窗口，请检查 window_len 与 CSV 中的标签。")

    inputs = np.stack(input_seqs, axis=0)
    intents_arr = np.asarray(intent_labels, dtype=np.int64)
    threats_arr = np.asarray(threat_scores, dtype=np.float32)
    return inputs, intents_arr, threats_arr


def _apply_delta_inplace_inputs(
    inputs: np.ndarray,
    pos_indices: List[int] = POS_DIM_INDEX,
) -> np.ndarray:
    """
    就地把位置 x,y,z 转为“相邻时间步的增量”(delta)，速度保持不变。

    对每个样本单独处理：
      - pos = inputs[b, :, pos_indices]  -> [L, P]
      - delta = pos[1:] - pos[:-1]      -> [L-1, P]
      - 在最前面插入一行 0，使得长度恢复 L：
          delta_full = [0; delta]       -> [L, P]
      - 再写回 inputs[b, :, pos_indices]
    """
    B, L, D = inputs.shape

    for b in range(B):
        pos = inputs[b, :, pos_indices]                   # [L, P]
        if L <= 1:
            # 单步没什么可 diff 的，直接置零
            inputs[b, :, pos_indices] = 0.0
            continue
        delta = np.diff(pos, axis=0)                      # [L-1, P]
        delta_full = np.vstack(
            [np.zeros((1, delta.shape[1]), dtype=delta.dtype), delta]
        )                                                 # [L, P]
        inputs[b, :, pos_indices] = delta_full

    return inputs


# ===================== 对外接口：构建数据集和 scaler =====================


def build_datasets_from_config(
    config_path: str = "config.yaml",
) -> Tuple[
    IntentThreatWindowDataset,
    IntentThreatWindowDataset,
    IntentThreatWindowDataset,
    Optional[StandardScaler],
]:
    """
    从 config.yaml 构建“意图 + 威胁度”的 train/val/test 三个数据集，并返回一个 scaler（若 normalize=True）。

    返回：
      train_ds, val_ds, test_ds, scaler
    """
    project_root = get_project_root()

    cfg = load_config(project_root / config_path)
    data_cfg = cfg.get("data", {})

    # 窗口长度：优先从 data.intent_threat.window_len 读
    label_cfg = data_cfg.get("intent_threat", {})
    window_len = int(label_cfg.get("window_len", 10))

    # 训练/验证/测试比例：沿用 data 里的 train_ratio / val_ratio / test_ratio
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))

    # 是否将位置换成增量 + 是否对输入做标准化
    use_delta = bool(data_cfg.get("intent_use_delta", True))
    normalize = bool(data_cfg.get("intent_normalize", True))

    processed_dir = data_cfg.get("processed_dir", "data/processed")
    # 单独给意图任务一个 scaler 文件名，避免覆盖轨迹预测的 scaler
    intent_scaler_filename = data_cfg.get(
        "intent_scaler_filename", "scaler_intent_posvel.npz"
    )
    processed_dir_path = (project_root / processed_dir).resolve()
    scaler_path = processed_dir_path / intent_scaler_filename

    # ==== NEW: 用一个确定性的随机种子来打乱 traj_id 顺序 ====
    # 优先从 train_intent.seed / train.seed 读；如果都没有，再看 data.intent_split_seed；最后默认 42。
    train_intent_cfg = cfg.get("train_intent", {})
    train_cfg_global = cfg.get("train", {})
    split_seed = train_intent_cfg.get(
        "seed",
        train_cfg_global.get(
            "seed",
            data_cfg.get("intent_split_seed", 42),
        ),
    )
    split_seed = int(split_seed)

    # 1) 读原始 CSV（含 intent_label / threat_score）
    df = load_raw_trajectories(cfg, project_root)

    # 2) 根据 traj_id 划分 train/val/test（划分前先打乱 traj_id）
    traj_ids_unique = df["traj_id"].unique()
    # 确保是 Python int 列表，避免后续 set 里类型不一致
    traj_ids = [int(t) for t in traj_ids_unique]
    n_total = len(traj_ids)
    if n_total == 0:
        raise RuntimeError("CSV 中没有任何 traj_id")

    rng = np.random.default_rng(split_seed)
    rng.shuffle(traj_ids)  # 关键：随机打乱轨迹顺序，避免生成顺序带来的偏差

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
    val_ids = traj_ids[n_train: n_train + n_val]
    test_ids = traj_ids[n_train + n_val: n_train + n_val + n_test]

    print(
        f"[IntentDataset] Total traj_ids={n_total}, "
        f"train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}, "
        f"split_seed={split_seed}"
    )

    # 3) 构造“窗口 → (intent, threat)”数据
    train_in, train_intent, train_threat = _make_intent_windows_for_ids(
        df, train_ids, window_len
    )
    val_in, val_intent, val_threat = _make_intent_windows_for_ids(
        df, val_ids, window_len
    )
    test_in, test_intent, test_threat = _make_intent_windows_for_ids(
        df, test_ids, window_len
    )

    # 4) 位置增量（可选）
    if use_delta:
        train_in = _apply_delta_inplace_inputs(train_in)
        val_in = _apply_delta_inplace_inputs(val_in)
        test_in = _apply_delta_inplace_inputs(test_in)

    # 5) 归一化（可选）：只用 train 的输入来 fit scaler
    scaler: Optional[StandardScaler] = None
    if normalize:
        scaler = StandardScaler()
        # 拼成 [N * window_len, D]
        train_all = train_in.reshape(-1, train_in.shape[-1])
        scaler.fit(train_all)

        # 对三个集合分别做 transform
        train_in = scaler.transform(train_in)
        val_in = scaler.transform(val_in)
        test_in = scaler.transform(test_in)

        # 保存 scaler 以便推理/可视化时反归一化
        scaler.save(scaler_path)

    # 6) 构造数据集对象
    train_ds = IntentThreatWindowDataset(train_in, train_intent, train_threat, scaler=scaler)
    val_ds = IntentThreatWindowDataset(val_in, val_intent, val_threat, scaler=scaler)
    test_ds = IntentThreatWindowDataset(test_in, test_intent, test_threat, scaler=scaler)

    return train_ds, val_ds, test_ds, scaler
