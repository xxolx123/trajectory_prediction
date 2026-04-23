#!/usr/bin/env python3
"""
generate_trajs.py

- 运动模型：CV / CA
- 速度档：可配置（如 20 / 30 / 50 km/h）
- 加速度档：根据“总时长内最大变化 DV_MAX_KMH”自动算出三档加速度
- 轨迹类型：直线 / 左转 / 右转 / S 型弯（只转一次，不再绕圈）

新增：
- 转向总角度列表：turn_total_deg_list: [30, 45, 60, 90]
- 左右转开始点：turn_start_coeffs: [0.3, 0.5, 0.7]，按 (num_steps-1)*coeff 求索引
- S 弯左转段用系数组对：s_left_coeff_pairs: [[0.2,0.7],[0.5,0.9],...]

分布设计：
- straight / left_turn / right_turn：
    遍历 turn_total_deg_list × turn_start_coeffs（S 段用一个代表区间）
- s_curve：
    遍历 turn_total_deg_list × turn_start_coeffs × s_left_coeff_pairs
  → S 弯样本相对更多；如果觉得过多可以减少 s_left_coeff_pairs 或减小 num_traj_per_combo

单位：
- 位置 x,y,z：km
- 速度 vx,vy,vz：km/s

Domain randomization：
- 对同一 combo 内的每条轨迹，额外对以下量加噪声：
    * 初始位置：init_x, init_y, init_z（均匀采样于给定范围）
    * 初始方向：init_heading（均匀采样于 [0, 2π)）
    * 初始速度：speed_kmh ± speed_noise_kmh
    * 加速度（CA）：accel_mps2 × (1 ± accel_noise_rel)

用法（在项目根目录）：
    python -m code.data.generate_trajs --config config.yaml
"""

import math
import csv
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml

# ======================= 全局参数（将由 config 填充） =========================

TIME_STEP: float
NUM_STEPS: int

MOTION_MODELS: List[str]
SPEEDS_KMH: List[float]

DV_MAX_KMH: float
TOTAL_TIME: float
ACCEL_LEVELS: List[float]
DV_MAX_MPS: float
BASE_A_MPS2: float
ACCELS_MPS2: List[float]

MIN_SPEED_KMH: float

TRAJ_TYPES: List[str]

# 转向相关（多角度、多起始点）
TURN_TOTAL_DEG: float                      # 代表性角度（列表第一个）
TURN_TOTAL_DEG_LIST: List[float]          # 多个备选总转向角（度）
TURN_START_IDX: int                       # 代表性起始步（第一个系数推出来）
TURN_START_COEFFS: List[float]            # 多个起始系数（0~1）

# S 弯左右段：既保留绝对索引，又支持系数区间对
S_LEFT_START: int
S_LEFT_END: int
S_LEFT_COEFF_PAIRS: List[Tuple[float, float]]

NUM_TRAJ_PER_COMBO: int

INIT_X_RANGE: Tuple[float, float]
INIT_Y_RANGE: Tuple[float, float]
INIT_Z_RANGE: Tuple[float, float]

USE_VERTICAL_MOTION: bool
VERTICAL_VZ_RANGE_KMPS: Tuple[float, float]

# domain randomization 噪声
SPEED_NOISE_KMH: float          # 速度噪声幅度（km/h）
ACCEL_NOISE_REL: float          # 加速度相对噪声比例（例如 0.3 表示 ±30%）

RANDOM_SEED: int

RAW_DIR: Path
OUTPUT_CSV_NAME: str  # 只保存文件名，最终路径 = ROOT/RAW_DIR/OUTPUT_CSV_NAME


# ======================= 从 config.yaml 读取并填充全局参数 ====================

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_data_config(cfg: Dict[str, Any], project_root: Path) -> None:
    """
    读取 config['data']，填充本文件顶部所有全局变量。
    如果某些字段没写，就使用与原脚本一致的默认值。
    """
    global TIME_STEP, NUM_STEPS
    global MOTION_MODELS, SPEEDS_KMH
    global DV_MAX_KMH, TOTAL_TIME, ACCEL_LEVELS, DV_MAX_MPS, BASE_A_MPS2, ACCELS_MPS2
    global MIN_SPEED_KMH
    global TRAJ_TYPES
    global TURN_TOTAL_DEG, TURN_TOTAL_DEG_LIST, TURN_START_IDX, TURN_START_COEFFS
    global S_LEFT_START, S_LEFT_END, S_LEFT_COEFF_PAIRS
    global NUM_TRAJ_PER_COMBO
    global INIT_X_RANGE, INIT_Y_RANGE, INIT_Z_RANGE
    global USE_VERTICAL_MOTION, VERTICAL_VZ_RANGE_KMPS
    global SPEED_NOISE_KMH, ACCEL_NOISE_REL
    global RANDOM_SEED
    global RAW_DIR, OUTPUT_CSV_NAME

    data_cfg = cfg.get("data", {})

    # 基本时间/步数
    TIME_STEP = float(data_cfg.get("time_step", 60.0))      # 每步时间间隔（秒）
    NUM_STEPS = int(data_cfg.get("num_steps", 100))         # 每条轨迹总步数

    # 模型 & 速度
    MOTION_MODELS = list(data_cfg.get("motion_models", ["CV", "CA"]))
    SPEEDS_KMH = [float(v) for v in data_cfg.get("speeds_kmh", [20.0, 30.0, 50.0])]

    # “总时长内最多变化 DV_MAX_KMH” 的加速度逻辑
    DV_MAX_KMH = float(data_cfg.get("dv_max_kmh", 20.0))
    TOTAL_TIME = (NUM_STEPS - 1) * TIME_STEP

    ACCEL_LEVELS = [float(a) for a in data_cfg.get("accel_levels", [-1.0, 0.0, 1.0])]
    DV_MAX_MPS = DV_MAX_KMH * 1000.0 / 3600.0
    BASE_A_MPS2 = DV_MAX_MPS / TOTAL_TIME
    ACCELS_MPS2 = [alpha * BASE_A_MPS2 for alpha in ACCEL_LEVELS]

    # 最小速度
    MIN_SPEED_KMH = float(data_cfg.get("min_speed_kmh", 15.0))

    # 轨迹类型
    TRAJ_TYPES = list(
        data_cfg.get(
            "traj_types",
            ["straight", "left_turn", "right_turn", "s_curve"],
        )
    )

    # 转弯总角度（列表）
    turn_total_deg_list_cfg = data_cfg.get("turn_total_deg_list")
    if turn_total_deg_list_cfg is None:
        default_deg = float(data_cfg.get("turn_total_deg", 60.0))
        TURN_TOTAL_DEG_LIST = [default_deg]
    else:
        TURN_TOTAL_DEG_LIST = [float(v) for v in turn_total_deg_list_cfg]

    TURN_TOTAL_DEG = TURN_TOTAL_DEG_LIST[0]

    # 左/右转起始点系数
    turn_start_coeffs_cfg = data_cfg.get("turn_start_coeffs")
    if turn_start_coeffs_cfg is None:
        default_idx = int(data_cfg.get("turn_start_idx", 50))
        denom = max(1, NUM_STEPS - 1)
        TURN_START_COEFFS = [default_idx / float(denom)]
    else:
        TURN_START_COEFFS = [float(c) for c in turn_start_coeffs_cfg]

    denom = max(1, NUM_STEPS - 1)
    TURN_START_IDX = int(round(TURN_START_COEFFS[0] * denom))

    # 旧版 S 弯的绝对索引（用于默认或兜底）
    S_LEFT_START = int(data_cfg.get("s_left_start", 33))
    S_LEFT_END = int(data_cfg.get("s_left_end", 66))

    # S 弯用的系数组对列表
    s_pairs_cfg = data_cfg.get("s_left_coeff_pairs")
    if s_pairs_cfg is None:
        # 如果没给系数组对，就由绝对索引反推一个默认 pair
        denom_s = max(1, NUM_STEPS - 1)
        s_start_coeff = S_LEFT_START / float(denom_s)
        s_end_coeff = S_LEFT_END / float(denom_s)
        S_LEFT_COEFF_PAIRS = [(s_start_coeff, s_end_coeff)]
    else:
        pairs: List[Tuple[float, float]] = []
        for pair in s_pairs_cfg:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = float(pair[0]), float(pair[1])
            pairs.append((a, b))
        if not pairs:
            # 避免写错导致空列表，退回默认
            denom_s = max(1, NUM_STEPS - 1)
            s_start_coeff = S_LEFT_START / float(denom_s)
            s_end_coeff = S_LEFT_END / float(denom_s)
            S_LEFT_COEFF_PAIRS = [(s_start_coeff, s_end_coeff)]
        else:
            S_LEFT_COEFF_PAIRS = pairs
            # 用第一个 pair 更新代表性索引（方便调试/打印）
            denom_s = max(1, NUM_STEPS - 1)
            S_LEFT_START = int(round(S_LEFT_COEFF_PAIRS[0][0] * denom_s))
            S_LEFT_END = int(round(S_LEFT_COEFF_PAIRS[0][1] * denom_s))

    # 每个组合生成多少条轨迹
    NUM_TRAJ_PER_COMBO = int(data_cfg.get("num_traj_per_combo", 50))

    # 初始位置范围
    INIT_X_RANGE = tuple(data_cfg.get("init_x_range", [-30.0, 30.0]))  # type: ignore
    INIT_Y_RANGE = tuple(data_cfg.get("init_y_range", [-30.0, 30.0]))  # type: ignore
    INIT_Z_RANGE = tuple(data_cfg.get("init_z_range", [0.0, 0.0]))     # type: ignore

    # 垂直运动
    USE_VERTICAL_MOTION = bool(data_cfg.get("use_vertical_motion", False))
    VERTICAL_VZ_RANGE_KMPS = tuple(
        data_cfg.get("vertical_vz_range_kmps", [-0.005, 0.005])
    )  # type: ignore

    # domain randomization 噪声（不想用就设为 0）
    SPEED_NOISE_KMH = float(data_cfg.get("speed_noise_kmh", 0.0))
    ACCEL_NOISE_REL = float(data_cfg.get("accel_noise_rel", 0.0))

    # 随机种子
    RANDOM_SEED = int(data_cfg.get("random_seed", 42))

    # 输出路径
    raw_dir_str = data_cfg.get("raw_dir", "data/raw")
    RAW_DIR = (project_root / raw_dir_str).resolve()

    OUTPUT_CSV_NAME = data_cfg.get("output_csv", "synthetic_trajectories.csv")


# ======================= 轨迹生成逻辑 ====================

def heading_schedule(
    traj_type: str,
    step_idx: int,
    num_steps: int,
    base_heading_rad: float,
    turn_total_deg: float,
    turn_start_idx: int,
    s_left_start: int,
    s_left_end: int,
) -> float:
    """
    给定轨迹类型 + 步索引，返回当前步的航向角（弧度）。

    设计：
      - straight: 一直是 base_heading
      - left_turn/right_turn:
          前 turn_start_idx 步保持 base_heading，
          后面步数线性从 base_heading 过渡到 base_heading ± turn_total_deg
      - s_curve:
          [0, s_left_start]      ：base_heading
          [s_left_start, s_left_end] 左转到 base_heading + θ
          [s_left_end, N-1]      ：再右转回 base_heading
    """
    theta = math.radians(turn_total_deg)
    N = num_steps
    i = step_idx

    if traj_type == "straight":
        return base_heading_rad

    if traj_type in ("left_turn", "right_turn"):
        if i <= turn_start_idx:
            frac = 0.0
        else:
            # 从 turn_start_idx -> N-1 线性过渡到 1.0
            denom = max(1, (N - 1 - turn_start_idx))
            frac = (i - turn_start_idx) / denom
            frac = max(0.0, min(1.0, frac))

        signed_theta = theta if traj_type == "left_turn" else -theta
        return base_heading_rad + signed_theta * frac

    if traj_type == "s_curve":
        # S: 左转到 +θ，再右转回 base_heading
        if i <= s_left_start:
            return base_heading_rad

        if s_left_start < i <= s_left_end:
            # 左转段：从 0 -> +θ
            denom = max(1, (s_left_end - s_left_start))
            frac = (i - s_left_start) / denom
            frac = max(0.0, min(1.0, frac))
            return base_heading_rad + theta * frac

        # 右转段：从 +θ -> 0
        denom = max(1, (N - 1 - s_left_end))
        frac = (i - s_left_end) / denom
        frac = max(0.0, min(1.0, frac))
        return base_heading_rad + theta * (1.0 - frac)

    # 未知类型就当直线处理
    return base_heading_rad


def simulate_trajectory(
    motion_model: str,
    base_speed_kmh: float,
    accel_mps2: float,
    traj_type: str,
    init_x_km: float,
    init_y_km: float,
    init_z_km: float,
    init_heading_rad: float,
    turn_total_deg: float,
    turn_start_idx: int,
    s_left_start: int,
    s_left_end: int,
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    生成一条轨迹：
      [(x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps), ...] 长度 NUM_STEPS
    """

    # 速度 & 加速度单位转换
    v_kmps = base_speed_kmh / 3600.0       # km/h → km/s
    a_kmps2 = accel_mps2 / 1000.0          # m/s^2 → km/s^2
    min_speed_kmps = MIN_SPEED_KMH / 3600.0

    # 初始状态
    x = init_x_km
    y = init_y_km
    z = init_z_km

    if USE_VERTICAL_MOTION:
        vz = random.uniform(*VERTICAL_VZ_RANGE_KMPS)
    else:
        vz = 0.0

    states: List[Tuple[float, float, float, float, float, float]] = []

    for step in range(NUM_STEPS):
        heading = heading_schedule(
            traj_type=traj_type,
            step_idx=step,
            num_steps=NUM_STEPS,
            base_heading_rad=init_heading_rad,
            turn_total_deg=turn_total_deg,
            turn_start_idx=turn_start_idx,
            s_left_start=s_left_start,
            s_left_end=s_left_end,
        )

        vx = v_kmps * math.cos(heading)
        vy = v_kmps * math.sin(heading)

        states.append((x, y, z, vx, vy, vz))

        x += vx * TIME_STEP
        y += vy * TIME_STEP
        z += vz * TIME_STEP

        if motion_model == "CA":
            v_kmps += a_kmps2 * TIME_STEP
            if v_kmps < min_speed_kmps:
                v_kmps = min_speed_kmps

    return states


# ======================= 主函数 ====================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative to project root or absolute).",
    )
    args = parser.parse_args()

    # 计算项目根目录（…/project_root/code/data/generate_trajs.py → project_root）
    project_root = Path(__file__).resolve().parents[2]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_config(config_path)
    apply_data_config(cfg, project_root)

    random.seed(RANDOM_SEED)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / OUTPUT_CSV_NAME

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "traj_id",
                "motion_model",
                "base_speed_kmh",
                "accel_mps2",
                "traj_type",
                "step_idx",
                "time_s",
                "x_km",
                "y_km",
                "z_km",
                "vx_kmps",
                "vy_kmps",
                "vz_kmps",
            ]
        )

        traj_id = 0
        num_rows = 0

        denom_turn = max(1, NUM_STEPS - 1)
        denom_s = max(1, NUM_STEPS - 1)

        for motion_model in MOTION_MODELS:
            for speed_kmh in SPEEDS_KMH:

                # CV 下加速度强制为 0
                if motion_model == "CV":
                    accel_list = [0.0]
                else:
                    accel_list = ACCELS_MPS2

                for accel_mps2 in accel_list:
                    for traj_type in TRAJ_TYPES:

                        # 直线 / 左转 / 右转：用所有角度和起始系数组合
                        # S 弯：额外乘以 s_left_coeff_pairs
                        angle_list = TURN_TOTAL_DEG_LIST
                        start_coeff_list = TURN_START_COEFFS

                        if traj_type == "s_curve":
                            s_pair_list: List[Optional[Tuple[float, float]]] = S_LEFT_COEFF_PAIRS
                        else:
                            # 非 S 弯只用一个代表性 S 段区间
                            s_pair_list = [None]

                        for turn_total_deg in angle_list:
                            for start_coeff in start_coeff_list:
                                # 左/右转起始步索引
                                raw_idx = int(round(start_coeff * denom_turn))
                                turn_start_idx = max(0, min(NUM_STEPS - 2, raw_idx))

                                for s_pair in s_pair_list:
                                    # 计算 S 段左右索引
                                    if traj_type == "s_curve":
                                        if s_pair is None:
                                            s_left_start = S_LEFT_START
                                            s_left_end = S_LEFT_END
                                        else:
                                            s_start_c, s_end_c = s_pair
                                            s_left_start = int(round(s_start_c * denom_s))
                                            s_left_end = int(round(s_end_c * denom_s))
                                            # 夹在合法范围内
                                            s_left_start = max(0, min(NUM_STEPS - 2, s_left_start))
                                            s_left_end = max(1, min(NUM_STEPS - 1, s_left_end))
                                            if s_left_start >= s_left_end:
                                                # 强制至少有一段：如果写反或重合就调一下
                                                s_left_start = max(0, min(NUM_STEPS - 2, s_left_start))
                                                s_left_end = min(NUM_STEPS - 1, s_left_start + 1)
                                    else:
                                        # 非 S 弯时 S 段只作为占位参数
                                        s_left_start = S_LEFT_START
                                        s_left_end = S_LEFT_END

                                    for _ in range(NUM_TRAJ_PER_COMBO):
                                        traj_id += 1

                                        # ---------- 1) 对速度 / 加速度做 domain randomization ----------
                                        # speed_sample 以 combo 的 speed_kmh 为中心，加一点噪声
                                        if SPEED_NOISE_KMH > 0.0:
                                            speed_sample = speed_kmh + random.uniform(
                                                -SPEED_NOISE_KMH, SPEED_NOISE_KMH
                                            )
                                        else:
                                            speed_sample = speed_kmh

                                        # 限制最小速度
                                        if speed_sample < MIN_SPEED_KMH:
                                            speed_sample = MIN_SPEED_KMH

                                        # CA 模型下的加速度也加一点相对噪声；CV 保持 0
                                        if motion_model == "CA":
                                            if ACCEL_NOISE_REL > 0.0 and accel_mps2 != 0.0:
                                                factor = 1.0 + random.uniform(
                                                    -ACCEL_NOISE_REL, ACCEL_NOISE_REL
                                                )
                                                accel_sample = accel_mps2 * factor
                                            else:
                                                accel_sample = accel_mps2
                                        else:
                                            accel_sample = 0.0

                                        # ---------- 2) 初始位置 / 方向随机 ----------
                                        init_x = random.uniform(*INIT_X_RANGE)
                                        init_y = random.uniform(*INIT_Y_RANGE)
                                        init_z = random.uniform(*INIT_Z_RANGE)
                                        init_heading = random.uniform(0.0, 2.0 * math.pi)

                                        # ---------- 3) 用“带噪声”的参数生成这条轨迹 ----------
                                        states = simulate_trajectory(
                                            motion_model=motion_model,
                                            base_speed_kmh=speed_sample,
                                            accel_mps2=accel_sample,
                                            traj_type=traj_type,
                                            init_x_km=init_x,
                                            init_y_km=init_y,
                                            init_z_km=init_z,
                                            init_heading_rad=init_heading,
                                            turn_total_deg=turn_total_deg,
                                            turn_start_idx=turn_start_idx,
                                            s_left_start=s_left_start,
                                            s_left_end=s_left_end,
                                        )

                                        # ---------- 4) 写出：记录实际使用的 speed/accel ----------
                                        for step_idx, (x, y, z, vx, vy, vz) in enumerate(states):
                                            time_s = step_idx * TIME_STEP
                                            writer.writerow(
                                                [
                                                    traj_id,
                                                    motion_model,
                                                    speed_sample,
                                                    accel_sample,
                                                    traj_type,
                                                    step_idx,
                                                    time_s,
                                                    x,
                                                    y,
                                                    z,
                                                    vx,
                                                    vy,
                                                    vz,
                                                ]
                                            )
                                            num_rows += 1

    print(
        f"生成完成：{traj_id} 条轨迹，总 {num_rows} 行，输出到 {out_path}"
    )


if __name__ == "__main__":
    main()
