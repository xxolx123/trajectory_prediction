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

意图 & 威胁度：
- 使用最近 INTENT_WINDOW_LEN 步的轨迹窗口，根据曲率/速度变化判断意图并计算威胁度

随机性（domain randomization）：
- 在 combo 的基础上，对每条轨迹随机扰动：
    * 初始位置、初始航向
    * 速度档（± speed_noise_kmh）
    * 加速度（× (1 ± accel_noise_ratio)）
    * 总转弯角度（× (1 ± turn_deg_noise_ratio)）
    * 转弯起点（index 附近小范围平移）
    * S 段区间（start/end index 附近小范围平移）

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

RANDOM_SEED: int

RAW_DIR: Path
OUTPUT_CSV_NAME: str  # 只保存文件名，最终路径 = ROOT/RAW_DIR/OUTPUT_CSV_NAME

# ======================= 随机扰动相关全局参数（domain randomization） =========================
# 速度扰动（绝对值，单位 km/h），每条轨迹在 combo 的基础上 ± speed_noise_kmh
SPEED_NOISE_KMH: float

# 加速度扰动（相对比例），每条轨迹在 combo 的基础上乘以 (1 ± accel_noise_ratio)
ACCEL_NOISE_RATIO: float

# 转向总角度扰动（相对比例），每条轨迹的 turn_total_deg 乘以 (1 ± turn_deg_noise_ratio)
TURN_DEG_NOISE_RATIO: float

# 转弯起点扰动（系数级），最大平移约为 (num_steps-1) * turn_start_noise_coeff
TURN_START_NOISE_COEFF: float

# S 段左右边界扰动（系数级），最大平移约为 (num_steps-1) * s_left_noise_coeff
S_LEFT_NOISE_COEFF: float

# ======================= 意图 & 威胁度 标注相关全局参数 =========================
# 意图枚举（顺序严格按需求）
INTENT_ATTACK  = 0  # 进攻
INTENT_EVASION = 1  # 规避
INTENT_DEFENSE = 2  # 防御
INTENT_RETREAT = 3  # 撤退

# 窗口长度（默认 10 步）
INTENT_WINDOW_LEN: int

# 曲率/转向角阈值（弧度）
CURV_STRAIGHT_RAD: float
CURV_EVASION_RAD: float
TURN_UTURN_RAD: float

# 速度变化 & 速度阈值（m/s）
DV_ATTACK_MIN_MS: float
DV_DEFENSE_MIN_MS: float
SPEED_NOCHANGE_EPS_MS: float
SPEED_SPLIT_THRESHOLD_MS: float

# 威胁度相关（m/s）
THREAT_V_MIN_MS: float
THREAT_V_MAX_MS: float
THREAT_DV_MAX_MS: float
THREAT_W_INTENT: float
THREAT_W_SPEED: float
THREAT_W_ACCEL: float
THREAT_BASE_ATTACK: float
THREAT_BASE_EVASION: float
THREAT_BASE_DEFENSE: float
THREAT_BASE_RETREAT: float


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
    global RANDOM_SEED
    global RAW_DIR, OUTPUT_CSV_NAME
    global INTENT_WINDOW_LEN, CURV_STRAIGHT_RAD, CURV_EVASION_RAD, TURN_UTURN_RAD
    global DV_ATTACK_MIN_MS, DV_DEFENSE_MIN_MS, SPEED_NOCHANGE_EPS_MS, SPEED_SPLIT_THRESHOLD_MS
    global THREAT_V_MIN_MS, THREAT_V_MAX_MS, THREAT_DV_MAX_MS
    global THREAT_W_INTENT, THREAT_W_SPEED, THREAT_W_ACCEL
    global THREAT_BASE_ATTACK, THREAT_BASE_EVASION, THREAT_BASE_DEFENSE, THREAT_BASE_RETREAT
    global SPEED_NOISE_KMH, ACCEL_NOISE_RATIO
    global TURN_DEG_NOISE_RATIO, TURN_START_NOISE_COEFF, S_LEFT_NOISE_COEFF

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

    # === 新增：domain randomization 参数（可以在 config.yaml 里覆盖） ===
    SPEED_NOISE_KMH = float(data_cfg.get("speed_noise_kmh", 3.0))
    ACCEL_NOISE_RATIO = float(data_cfg.get("accel_noise_ratio", 0.3))
    TURN_DEG_NOISE_RATIO = float(data_cfg.get("turn_deg_noise_ratio", 0.1))
    TURN_START_NOISE_COEFF = float(data_cfg.get("turn_start_noise_coeff", 0.05))
    S_LEFT_NOISE_COEFF = float(data_cfg.get("s_left_noise_coeff", 0.05))

    # 初始位置范围
    INIT_X_RANGE = tuple(data_cfg.get("init_x_range", [-30.0, 30.0]))  # type: ignore
    INIT_Y_RANGE = tuple(data_cfg.get("init_y_range", [-30.0, 30.0]))  # type: ignore
    INIT_Z_RANGE = tuple(data_cfg.get("init_z_range", [0.0, 0.0]))     # type: ignore

    # 垂直运动
    USE_VERTICAL_MOTION = bool(data_cfg.get("use_vertical_motion", False))
    VERTICAL_VZ_RANGE_KMPS = tuple(
        data_cfg.get("vertical_vz_range_kmps", [-0.005, 0.005])
    )  # type: ignore

    # 随机种子
    RANDOM_SEED = int(data_cfg.get("random_seed", 42))

    # 输出路径
    raw_dir_str = data_cfg.get("raw_dir", "data/raw")
    RAW_DIR = (project_root / raw_dir_str).resolve()

    OUTPUT_CSV_NAME = data_cfg.get("output_csv", "synthetic_trajectories.csv")

    # ====== 意图 & 威胁度配置（data.intent_threat 下） ======
    label_cfg = data_cfg.get("intent_threat", {})

    INTENT_WINDOW_LEN = int(label_cfg.get("window_len", 10))

    curv_straight_deg = float(label_cfg.get("curv_straight_deg", 5.0))
    curv_evasion_deg = float(label_cfg.get("curv_evasion_deg", 15.0))
    turn_uturn_deg = float(label_cfg.get("turn_uturn_deg", 120.0))

    CURV_STRAIGHT_RAD = math.radians(curv_straight_deg)
    CURV_EVASION_RAD = math.radians(curv_evasion_deg)
    TURN_UTURN_RAD = math.radians(turn_uturn_deg)

    # 速度变化（m/s）
    DV_ATTACK_MIN_MS = float(label_cfg.get("dv_attack_min_ms", 5.0))
    DV_DEFENSE_MIN_MS = float(label_cfg.get("dv_defense_min_ms", 5.0))
    SPEED_NOCHANGE_EPS_MS = float(label_cfg.get("speed_nochange_eps_ms", 2.0))

    # 速度高低分界（m/s），默认用速度档的平均加一点
    default_speed_split_ms = sum(SPEEDS_KMH) / len(SPEEDS_KMH) * 1000.0 / 3600.0
    SPEED_SPLIT_THRESHOLD_MS = float(
        label_cfg.get("speed_split_threshold_ms", default_speed_split_ms)
    )

    # 威胁度速度范围（默认按速度档 min/max 来）
    default_v_min_ms = min(SPEEDS_KMH) * 1000.0 / 3600.0
    default_v_max_ms = max(SPEEDS_KMH) * 1000.0 / 3600.0

    THREAT_V_MIN_MS = float(label_cfg.get("threat_v_min_ms", default_v_min_ms))
    THREAT_V_MAX_MS = float(label_cfg.get("threat_v_max_ms", default_v_max_ms))
    THREAT_DV_MAX_MS = float(label_cfg.get("threat_dv_max_ms", DV_MAX_MPS))

    # 权重（和不一定是 1，后面会自动归一）
    THREAT_W_INTENT = float(label_cfg.get("w_intent", 0.5))
    THREAT_W_SPEED = float(label_cfg.get("w_speed", 0.3))
    THREAT_W_ACCEL = float(label_cfg.get("w_accel", 0.2))

    # 各意图基础威胁
    THREAT_BASE_ATTACK = float(label_cfg.get("base_attack", 1.0))
    THREAT_BASE_EVASION = float(label_cfg.get("base_evasion", 0.8))
    THREAT_BASE_DEFENSE = float(label_cfg.get("base_defense", 0.6))
    THREAT_BASE_RETREAT = float(label_cfg.get("base_retreat", 0.2))


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


# ======================= 意图 & 威胁度 计算函数 ====================

def wrap_angle_rad(angle: float) -> float:
    """把任意角度规约到 [-pi, pi] 区间"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_intent_and_threat(
    window_states: List[Tuple[float, float, float, float, float, float]]
) -> Tuple[int, int]:
    """
    根据最近 INTENT_WINDOW_LEN 步的状态（位置 + 速度），计算意图标签和威胁度。

    输入:
      window_states: 长度 INTENT_WINDOW_LEN 的列表，每个元素为
                     (x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps)

    输出:
      (intent_label, threat_score)
        intent_label ∈ {0,1,2,3}（对应 ATTACK, EVASION, DEFENSE, RETREAT）
        threat_score ∈ [0,100] 整数
    """
    assert len(window_states) == INTENT_WINDOW_LEN

    # 1) 速度标量 & 航向角（速度转成 m/s 计算）
    speeds_ms: List[float] = []
    headings: List[float] = []

    for (_, _, _, vx_kmps, vy_kmps, _) in window_states:
        v_ms = math.sqrt(vx_kmps * vx_kmps + vy_kmps * vy_kmps) * 1000.0  # km/s → m/s
        speeds_ms.append(v_ms)
        headings.append(math.atan2(vy_kmps, vx_kmps))

    v0 = speeds_ms[0]
    v_last = speeds_ms[-1]
    dv = v_last - v0
    v_mean = sum(speeds_ms) / len(speeds_ms)

    # 2) 航向变化，用来衡量转弯强度
    delta_headings: List[float] = []
    for i in range(1, len(headings)):
        dh = wrap_angle_rad(headings[i] - headings[i - 1])
        delta_headings.append(dh)

    if delta_headings:
        curv = sum(abs(dh) for dh in delta_headings) / len(delta_headings)
    else:
        curv = 0.0

    turn_total = abs(wrap_angle_rad(headings[-1] - headings[0]))

    # 3) 意图判定
    intent = INTENT_DEFENSE  # 默认防御

    # 3.1 撤退：总转角接近掉头
    if turn_total >= TURN_UTURN_RAD:
        intent = INTENT_RETREAT

    # 3.2 规避：平均转弯强度大
    elif curv >= CURV_EVASION_RAD:
        intent = INTENT_EVASION

    # 3.3 直线主导：再看速度变化 & 平均速度
    elif curv <= CURV_STRAIGHT_RAD:
        # 明显加速 → 进攻
        if dv >= DV_ATTACK_MIN_MS:
            intent = INTENT_ATTACK
        # 明显减速 → 防御
        elif dv <= -DV_DEFENSE_MIN_MS:
            intent = INTENT_DEFENSE
        else:
            # 速度变化不明显，用平均速度分高低
            if abs(dv) <= SPEED_NOCHANGE_EPS_MS:
                if v_mean >= SPEED_SPLIT_THRESHOLD_MS:
                    intent = INTENT_ATTACK   # 高速直线
                else:
                    intent = INTENT_DEFENSE  # 低速直线
            else:
                # 边界情况，按平均速度拉一把
                if v_mean >= SPEED_SPLIT_THRESHOLD_MS:
                    intent = INTENT_ATTACK
                else:
                    intent = INTENT_DEFENSE
    else:
        # 转弯不小，但还没到规避阈值，可以按规避处理
        intent = INTENT_EVASION

    # 4) 威胁度计算
    # 4.1 速度归一化
    if THREAT_V_MAX_MS > THREAT_V_MIN_MS:
        v_norm = (v_mean - THREAT_V_MIN_MS) / (THREAT_V_MAX_MS - THREAT_V_MIN_MS)
    else:
        v_norm = 0.0
    v_norm = max(0.0, min(1.0, v_norm))

    # 4.2 加速度归一化（只考虑加速）
    dv_pos = max(dv, 0.0)
    if THREAT_DV_MAX_MS > 0.0:
        dv_norm = dv_pos / THREAT_DV_MAX_MS
    else:
        dv_norm = 0.0
    dv_norm = max(0.0, min(1.0, dv_norm))

    # 4.3 意图基础威胁
    if intent == INTENT_ATTACK:
        base = THREAT_BASE_ATTACK
    elif intent == INTENT_EVASION:
        base = THREAT_BASE_EVASION
    elif intent == INTENT_DEFENSE:
        base = THREAT_BASE_DEFENSE
    elif intent == INTENT_RETREAT:
        base = THREAT_BASE_RETREAT
    else:
        base = 0.5

    # 4.4 综合
    w_sum = THREAT_W_INTENT + THREAT_W_SPEED + THREAT_W_ACCEL
    if w_sum <= 0.0:
        danger_0_1 = max(0.0, min(1.0, base))
    else:
        danger_0_1 = (
            THREAT_W_INTENT * base
            + THREAT_W_SPEED * v_norm
            + THREAT_W_ACCEL * dv_norm
        ) / w_sum
        danger_0_1 = max(0.0, min(1.0, danger_0_1))

    threat_score = int(round(100.0 * danger_0_1))

    return intent, threat_score


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

        # 新增 intent_label、threat_score 两列
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
                "intent_label",
                "threat_score",
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

                                        # ========= 1) 在 combo 的基础上做参数随机扰动 =========

                                        # 初始位置 & 初始航向
                                        init_x = random.uniform(*INIT_X_RANGE)
                                        init_y = random.uniform(*INIT_Y_RANGE)
                                        init_z = random.uniform(*INIT_Z_RANGE)
                                        init_heading = random.uniform(0.0, 2.0 * math.pi)

                                        # 速度：在速度档基础上加一个对称的绝对噪声，并不低于 MIN_SPEED_KMH
                                        if SPEED_NOISE_KMH > 0.0:
                                            speed_kmh_rand = speed_kmh + random.uniform(
                                                -SPEED_NOISE_KMH, SPEED_NOISE_KMH
                                            )
                                        else:
                                            speed_kmh_rand = speed_kmh
                                        speed_kmh_rand = max(MIN_SPEED_KMH, speed_kmh_rand)

                                        # 加速度：对 CA 模型加相对扰动，CV 仍为 0
                                        accel_mps2_rand = accel_mps2
                                        if (
                                            motion_model == "CA"
                                            and accel_mps2 != 0.0
                                            and ACCEL_NOISE_RATIO > 0.0
                                        ):
                                            factor_a = 1.0 + random.uniform(
                                                -ACCEL_NOISE_RATIO, ACCEL_NOISE_RATIO
                                            )
                                            accel_mps2_rand = accel_mps2 * factor_a

                                        # 转弯总角度：在 combo 的角度基础上 ± TURN_DEG_NOISE_RATIO
                                        turn_total_deg_rand = turn_total_deg
                                        if TURN_DEG_NOISE_RATIO > 0.0:
                                            factor_deg = 1.0 + random.uniform(
                                                -TURN_DEG_NOISE_RATIO, TURN_DEG_NOISE_RATIO
                                            )
                                            turn_total_deg_rand = max(
                                                0.0, turn_total_deg * factor_deg
                                            )

                                        # 转弯起始步：在 combo 的起点 index 附近小范围平移
                                        jitter_turn_start_idx = turn_start_idx
                                        if TURN_START_NOISE_COEFF > 0.0:
                                            max_shift = int(round(TURN_START_NOISE_COEFF * denom_turn))
                                            if max_shift > 0:
                                                shift = random.randint(-max_shift, max_shift)
                                                jitter_turn_start_idx = turn_start_idx + shift
                                                jitter_turn_start_idx = max(
                                                    0, min(NUM_STEPS - 2, jitter_turn_start_idx)
                                                )

                                        # S 弯左段区间：在 combo 的 [s_left_start, s_left_end] 附近小范围平移
                                        jitter_s_left_start = s_left_start
                                        jitter_s_left_end = s_left_end
                                        if traj_type == "s_curve" and S_LEFT_NOISE_COEFF > 0.0:
                                            max_shift_s = int(round(S_LEFT_NOISE_COEFF * denom_s))
                                            if max_shift_s > 0:
                                                jitter_s_left_start = s_left_start + random.randint(
                                                    -max_shift_s, max_shift_s
                                                )
                                                jitter_s_left_end = s_left_end + random.randint(
                                                    -max_shift_s, max_shift_s
                                                )

                                                # 合法化：保证 0 <= start < end <= N-1
                                                jitter_s_left_start = max(
                                                    0, min(NUM_STEPS - 2, jitter_s_left_start)
                                                )
                                                jitter_s_left_end = max(
                                                    jitter_s_left_start + 1,
                                                    min(NUM_STEPS - 1, jitter_s_left_end),
                                                )

                                        # ========= 2) 用“带噪声”的参数去模拟这条轨迹 =========

                                        states = simulate_trajectory(
                                            motion_model=motion_model,
                                            base_speed_kmh=speed_kmh_rand,
                                            accel_mps2=accel_mps2_rand,
                                            traj_type=traj_type,
                                            init_x_km=init_x,
                                            init_y_km=init_y,
                                            init_z_km=init_z,
                                            init_heading_rad=init_heading,
                                            turn_total_deg=turn_total_deg_rand,
                                            turn_start_idx=jitter_turn_start_idx,
                                            s_left_start=jitter_s_left_start,
                                            s_left_end=jitter_s_left_end,
                                        )

                                        # ========= 3) 意图 & 威胁度标注（保持原逻辑） =========

                                        # 维护最近 INTENT_WINDOW_LEN 步的缓存，用于标注
                                        window_states: List[
                                            Tuple[float, float, float, float, float, float]
                                        ] = []

                                        for step_idx, (x, y, z, vx, vy, vz) in enumerate(states):
                                            time_s = step_idx * TIME_STEP

                                            # 更新滑动窗口
                                            window_states.append((x, y, z, vx, vy, vz))
                                            if len(window_states) > INTENT_WINDOW_LEN:
                                                window_states.pop(0)

                                            # 窗口长度不足时，不标注（写 -1）
                                            if len(window_states) < INTENT_WINDOW_LEN:
                                                intent_label = -1
                                                threat_score = -1
                                            else:
                                                intent_label, threat_score = compute_intent_and_threat(
                                                    window_states
                                                )

                                            # 注意：这里的 base_speed_kmh / accel_mps2 仍然写 combo 的“名义值”
                                            # 如果你希望写入实际随机值，可以改成 speed_kmh_rand / accel_mps2_rand
                                            writer.writerow(
                                                [
                                                    traj_id,
                                                    motion_model,
                                                    speed_kmh,
                                                    accel_mps2,
                                                    traj_type,
                                                    step_idx,
                                                    time_s,
                                                    x,
                                                    y,
                                                    z,
                                                    vx,
                                                    vy,
                                                    vz,
                                                    intent_label,
                                                    threat_score,
                                                ]
                                            )
                                            num_rows += 1

    print(
        f"生成完成：{traj_id} 条轨迹，总 {num_rows} 行，输出到 {out_path}"
    )


if __name__ == "__main__":
    main()
