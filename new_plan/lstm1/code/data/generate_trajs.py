#!/usr/bin/env python3
"""
generate_trajs.py
-----------------
车辆轨迹合成数据生成器（采样式，Per-sample 运动学积分）。

相较旧版的关键变化：
- 不再做 (motion_model × speed × accel × traj_type × angle × start_coeff × s_pair)
  的穷举枚举，每条轨迹都独立采样一条"速度曲线 + 偏航率曲线"，再前向积分；
- 保留 4 个 `traj_type` 标签，但每类内部的"转几次 / 转多少 / 何时转 / 速度怎么
  变"全部连续采样，类内多样性显著提升；
- 输出 CSV 的列保持不变，下游 `traj_dataset.py` / 训练代码不需要改动。

单位：
- 位置 x,y,z: km
- 速度 vx,vy,vz: km/s
- 时间 step: 秒（`time_step`，默认 60s/步）

用法（在 `new_plan/lstm1/` 下）：
    python -m data.generate_trajs --config config.yaml
    # 小量预生成（覆盖 num_traj_per_type）：
    python -m data.generate_trajs --config config.yaml --num-per-type 20
"""

import math
import csv
import random
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import yaml

# ======================= 全局参数（由 config 填充） =========================

TIME_STEP: float
NUM_STEPS: int

TRAJ_TYPES: List[str]
NUM_TRAJ_PER_TYPE: Dict[str, int]

# 车辆动力学上限
SPEED_RANGE_KMH: Tuple[float, float]
MIN_SPEED_KMH: float
MAX_ACCEL_MPS2: float
MAX_DECEL_MPS2: float
MAX_YAW_RATE_DEG_S: float

# 速度曲线采样
SPEED_SEGMENTS_RANGE: Tuple[int, int]
SPEED_JITTER_KMH: float
ALLOW_STOP: bool
STOP_PROB: float

# 各类 yaw-rate 采样参数（子 dict）
STRAIGHT_CFG: Dict[str, Any]
TURN_CFG: Dict[str, Any]
S_CURVE_CFG: Dict[str, Any]
U_TURN_CFG: Dict[str, Any]

# 初始位姿
INIT_X_RANGE: Tuple[float, float]
INIT_Y_RANGE: Tuple[float, float]
INIT_Z_RANGE: Tuple[float, float]

USE_VERTICAL_MOTION: bool
VERTICAL_VZ_RANGE_KMPS: Tuple[float, float]

RANDOM_SEED: int

RAW_DIR: Path
OUTPUT_CSV_NAME: str


# 旧版字段名（被弃用，若在 config 中出现会打 deprecation 警告）
_DEPRECATED_KEYS = [
    "speeds_kmh",
    "motion_models",
    "accel_levels",
    "dv_max_kmh",
    "turn_total_deg",
    "turn_total_deg_list",
    "turn_start_idx",
    "turn_start_coeffs",
    "s_left_start",
    "s_left_end",
    "s_left_coeff_pairs",
    "num_traj_per_combo",
    "speed_noise_kmh",
    "accel_noise_rel",
]


# ======================= 从 config.yaml 读取并填充全局参数 ====================

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_int_range(value: Any, name: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"config.data.{name} 必须是长度为 2 的 [lo, hi] 列表")
    lo, hi = int(value[0]), int(value[1])
    if lo > hi:
        raise ValueError(f"config.data.{name}: lo({lo}) > hi({hi})")
    return lo, hi


def _as_float_range(value: Any, name: str, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"config.data.{name} 必须是长度为 2 的 [lo, hi] 列表")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f"config.data.{name}: lo({lo}) > hi({hi})")
    return lo, hi


def apply_data_config(cfg: Dict[str, Any], project_root: Path) -> None:
    """读取 config['data'] 并填充本文件顶部所有全局变量。"""
    global TIME_STEP, NUM_STEPS
    global TRAJ_TYPES, NUM_TRAJ_PER_TYPE
    global SPEED_RANGE_KMH, MIN_SPEED_KMH
    global MAX_ACCEL_MPS2, MAX_DECEL_MPS2, MAX_YAW_RATE_DEG_S
    global SPEED_SEGMENTS_RANGE, SPEED_JITTER_KMH, ALLOW_STOP, STOP_PROB
    global STRAIGHT_CFG, TURN_CFG, S_CURVE_CFG, U_TURN_CFG
    global INIT_X_RANGE, INIT_Y_RANGE, INIT_Z_RANGE
    global USE_VERTICAL_MOTION, VERTICAL_VZ_RANGE_KMPS
    global RANDOM_SEED
    global RAW_DIR, OUTPUT_CSV_NAME

    data_cfg = cfg.get("data", {}) or {}

    # --- deprecation 警告（不抛错，只是忽略） ---
    legacy_used = [k for k in _DEPRECATED_KEYS if k in data_cfg]
    if legacy_used:
        warnings.warn(
            "[generate_trajs] 以下 config.data 字段已弃用，新生成器不再读取："
            f"{legacy_used}。你可以删掉它们；轨迹将由新的采样式流程生成。",
            stacklevel=2,
        )

    # --- 基本时间 / 步数 ---
    TIME_STEP = float(data_cfg.get("time_step", 60.0))
    NUM_STEPS = int(data_cfg.get("num_steps", 100))
    if NUM_STEPS < 4:
        raise ValueError("num_steps 必须 >= 4")

    # --- 5 类标签与每类样本数 ---
    TRAJ_TYPES = list(
        data_cfg.get(
            "traj_types",
            ["straight", "left_turn", "right_turn", "s_curve", "u_turn"],
        )
    )

    num_per_type_cfg = data_cfg.get("num_traj_per_type", {}) or {}
    default_num = 200
    NUM_TRAJ_PER_TYPE = {
        t: int(num_per_type_cfg.get(t, default_num)) for t in TRAJ_TYPES
    }

    # --- 动力学上限 ---
    SPEED_RANGE_KMH = _as_float_range(
        data_cfg.get("speed_range_kmh"), "speed_range_kmh", (15.0, 60.0)
    )
    MIN_SPEED_KMH = float(data_cfg.get("min_speed_kmh", 10.0))
    if MIN_SPEED_KMH > SPEED_RANGE_KMH[0]:
        # 允许 min_speed 低于 speed_range 的下界，但不允许高于下界
        raise ValueError(
            f"min_speed_kmh({MIN_SPEED_KMH}) 必须 <= speed_range_kmh[0]({SPEED_RANGE_KMH[0]})"
        )

    MAX_ACCEL_MPS2 = float(data_cfg.get("max_accel_mps2", 0.8))
    MAX_DECEL_MPS2 = float(data_cfg.get("max_decel_mps2", 1.5))
    if MAX_ACCEL_MPS2 <= 0 or MAX_DECEL_MPS2 <= 0:
        raise ValueError("max_accel_mps2 / max_decel_mps2 必须 > 0")

    MAX_YAW_RATE_DEG_S = float(data_cfg.get("max_yaw_rate_deg_s", 20.0))
    if MAX_YAW_RATE_DEG_S <= 0:
        raise ValueError("max_yaw_rate_deg_s 必须 > 0")

    # --- 速度曲线采样 ---
    SPEED_SEGMENTS_RANGE = _as_int_range(
        data_cfg.get("speed_segments_range"), "speed_segments_range", (1, 4)
    )
    if SPEED_SEGMENTS_RANGE[0] < 1:
        raise ValueError("speed_segments_range 下界必须 >= 1")

    SPEED_JITTER_KMH = float(data_cfg.get("speed_jitter_kmh", 2.0))
    if SPEED_JITTER_KMH < 0:
        raise ValueError("speed_jitter_kmh 必须 >= 0")

    ALLOW_STOP = bool(data_cfg.get("allow_stop", False))
    STOP_PROB = float(data_cfg.get("stop_prob", 0.0))
    if not (0.0 <= STOP_PROB <= 1.0):
        raise ValueError("stop_prob 必须在 [0, 1] 内")

    # --- 每类 yaw-rate 参数 ---
    STRAIGHT_CFG = dict(data_cfg.get("straight", {}) or {})
    STRAIGHT_CFG.setdefault("heading_drift_deg_s", 0.2)

    TURN_CFG = dict(data_cfg.get("turn", {}) or {})
    TURN_CFG["n_turns_range"] = _as_int_range(
        TURN_CFG.get("n_turns_range"), "turn.n_turns_range", (1, 3)
    )
    TURN_CFG["angle_range_deg"] = _as_float_range(
        TURN_CFG.get("angle_range_deg"), "turn.angle_range_deg", (20.0, 120.0)
    )
    TURN_CFG["duration_range_steps"] = _as_int_range(
        TURN_CFG.get("duration_range_steps"), "turn.duration_range_steps", (5, 40)
    )
    TURN_CFG["min_gap_steps"] = int(TURN_CFG.get("min_gap_steps", 3))

    S_CURVE_CFG = dict(data_cfg.get("s_curve", {}) or {})
    S_CURVE_CFG["n_segments_range"] = _as_int_range(
        S_CURVE_CFG.get("n_segments_range"), "s_curve.n_segments_range", (2, 4)
    )
    if S_CURVE_CFG["n_segments_range"][0] < 2:
        raise ValueError("s_curve.n_segments_range 下界必须 >= 2（S 型至少两段交替）")

    S_CURVE_CFG["angle_range_deg"] = _as_float_range(
        S_CURVE_CFG.get("angle_range_deg"), "s_curve.angle_range_deg", (20.0, 90.0)
    )
    S_CURVE_CFG["segment_len_range_steps"] = _as_int_range(
        S_CURVE_CFG.get("segment_len_range_steps"),
        "s_curve.segment_len_range_steps",
        (8, 30),
    )
    S_CURVE_CFG["gap_steps_range"] = _as_int_range(
        S_CURVE_CFG.get("gap_steps_range"), "s_curve.gap_steps_range", (0, 5)
    )
    S_CURVE_CFG["first_direction"] = str(
        S_CURVE_CFG.get("first_direction", "random")
    ).lower()
    if S_CURVE_CFG["first_direction"] not in ("left", "right", "random"):
        raise ValueError("s_curve.first_direction 只能是 left/right/random")

    # --- u_turn（一次 ≈180° 的大转向，约束到能装进 in_len 或训练窗口） ---
    U_TURN_CFG = dict(data_cfg.get("u_turn", {}) or {})
    U_TURN_CFG["n_turns_range"] = _as_int_range(
        U_TURN_CFG.get("n_turns_range"), "u_turn.n_turns_range", (1, 1)
    )
    U_TURN_CFG["angle_range_deg"] = _as_float_range(
        U_TURN_CFG.get("angle_range_deg"), "u_turn.angle_range_deg", (150.0, 210.0)
    )
    U_TURN_CFG["duration_range_steps"] = _as_int_range(
        U_TURN_CFG.get("duration_range_steps"),
        "u_turn.duration_range_steps",
        (6, 18),
    )
    U_TURN_CFG["min_gap_steps"] = int(U_TURN_CFG.get("min_gap_steps", 10))
    U_TURN_CFG["direction"] = str(U_TURN_CFG.get("direction", "random")).lower()
    if U_TURN_CFG["direction"] not in ("left", "right", "random"):
        raise ValueError("u_turn.direction 只能是 left/right/random")

    # --- 约束冲突早期校验：角度下界 vs yaw-rate 上限 ---
    # 最短的 duration + 最大的 mag → 应不超过 max_yaw_rate
    max_yaw_rate_rad_s = math.radians(MAX_YAW_RATE_DEG_S)
    for cname, ccfg in (
        ("turn", TURN_CFG),
        ("s_curve", S_CURVE_CFG),
        ("u_turn", U_TURN_CFG),
    ):
        if cname == "s_curve":
            dur_lo, _ = ccfg["segment_len_range_steps"]
        else:
            dur_lo, _ = ccfg["duration_range_steps"]
        ang_lo, ang_hi = ccfg["angle_range_deg"]
        # 最极端：角度上界 + 持续步下界
        required = math.radians(ang_hi) / (max(1, dur_lo) * TIME_STEP)
        if required > max_yaw_rate_rad_s * 1.001:
            raise ValueError(
                f"[{cname}] angle_range_deg 上界 {ang_hi}° 在 {dur_lo} 步内完成"
                f" 需要 {math.degrees(required):.2f} deg/s，超过 max_yaw_rate_deg_s="
                f"{MAX_YAW_RATE_DEG_S}。请增大 duration 下界、降低 angle 上界或放宽 max_yaw_rate。"
            )
        # 同时用下界粗略估一下是否极度保守
        _ = ang_lo  # 占位，避免 linter 提示未用

    # --- 初始位姿 ---
    INIT_X_RANGE = tuple(data_cfg.get("init_x_range", [-30.0, 30.0]))  # type: ignore
    INIT_Y_RANGE = tuple(data_cfg.get("init_y_range", [-30.0, 30.0]))  # type: ignore
    INIT_Z_RANGE = tuple(data_cfg.get("init_z_range", [0.0, 0.0]))     # type: ignore

    # --- 垂直运动 ---
    USE_VERTICAL_MOTION = bool(data_cfg.get("use_vertical_motion", False))
    VERTICAL_VZ_RANGE_KMPS = tuple(
        data_cfg.get("vertical_vz_range_kmps", [-0.005, 0.005])
    )  # type: ignore

    # --- 随机种子 ---
    RANDOM_SEED = int(data_cfg.get("random_seed", 42))

    # --- 输出路径 ---
    raw_dir_str = data_cfg.get("raw_dir", "data/raw")
    RAW_DIR = (project_root / raw_dir_str).resolve()
    OUTPUT_CSV_NAME = data_cfg.get("output_csv", "synthetic_trajectories.csv")


# ======================= 采样器 =========================

def sample_speed_profile_kmh(
    num_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, bool]:
    """
    返回 (profile, n_segments, has_stop)：
      - profile [T]: 速度曲线（km/h）
      - n_segments: 本条轨迹分了几段
      - has_stop: 是否命中了 allow_stop 的停车段

    生成规则：
      - 分 K 段（K ∈ SPEED_SEGMENTS_RANGE），每段一个目标速度（均匀采自 SPEED_RANGE_KMH）；
      - 段间做受加速度约束的线性过渡；
      - 叠加小幅抖动；
      - 裁剪到 [max(MIN_SPEED_KMH, 0), SPEED_RANGE_KMH[1]]；
      - 如 ALLOW_STOP，以 STOP_PROB 概率在某一段强制为 0。
    """
    T = num_steps
    lo_seg, hi_seg = SPEED_SEGMENTS_RANGE
    K = int(rng.integers(lo_seg, hi_seg + 1))
    K = max(1, K)

    v_lo, v_hi = SPEED_RANGE_KMH

    # 段边界：第 0 段从 step 0 开始，最后一段到 step T-1 结束
    # 选 K-1 个内部切点（单调递增、严格在 (0, T-1) 区间）
    if K == 1:
        seg_starts = [0]
    else:
        cut_candidates = np.arange(1, T - 1)
        if len(cut_candidates) < K - 1:
            K = max(1, len(cut_candidates) + 1)
        cuts = (
            np.sort(rng.choice(cut_candidates, size=K - 1, replace=False))
            if K > 1
            else np.array([], dtype=int)
        )
        seg_starts = [0] + cuts.tolist()
    seg_ends = seg_starts[1:] + [T]  # 段 i 覆盖 [seg_starts[i], seg_ends[i])

    # 每段目标速度
    targets = rng.uniform(v_lo, v_hi, size=K)

    # 可选：强制某一段为 0（停车）
    has_stop = False
    if ALLOW_STOP and rng.random() < STOP_PROB and K >= 1:
        stop_seg_idx = int(rng.integers(0, K))
        targets[stop_seg_idx] = 0.0
        has_stop = True

    # 拼段：每段内先按"目标速度"填，但用受加速度约束的线性过渡
    profile = np.empty(T, dtype=np.float64)
    # 初始速度：第一段的目标速度作为起点
    v = float(targets[0])
    max_dv_per_step_kmh = max(MAX_ACCEL_MPS2, MAX_DECEL_MPS2) * 3.6 * TIME_STEP  # m/s^2 -> km/h per step
    for k in range(K):
        s, e = seg_starts[k], seg_ends[k]
        tgt = float(targets[k])
        for i in range(s, e):
            # 向 tgt 线性过渡，但单步变化不超过 max_dv_per_step_kmh
            dv = tgt - v
            if dv > max_dv_per_step_kmh:
                v = v + max_dv_per_step_kmh
            elif dv < -max_dv_per_step_kmh:
                v = v - max_dv_per_step_kmh
            else:
                v = tgt
            profile[i] = v

    # 步间抖动
    if SPEED_JITTER_KMH > 0:
        profile = profile + rng.normal(0.0, SPEED_JITTER_KMH, size=T)

    # 裁剪
    v_floor = max(MIN_SPEED_KMH, 0.0)
    profile = np.clip(profile, v_floor, v_hi)

    # 若命中了 stop 段，让该段真的为 0（绕开 MIN_SPEED 的 floor）
    if ALLOW_STOP:
        for k in range(K):
            if targets[k] == 0.0:
                s, e = seg_starts[k], seg_ends[k]
                profile[s:e] = 0.0

    return profile, int(K), bool(has_stop)


def _sample_nonoverlap_segments(
    num_steps: int,
    n_segments: int,
    duration_range: Tuple[int, int],
    min_gap_steps: int,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> List[Tuple[int, int]]:
    """
    在 [0, num_steps) 里采 n_segments 个不重叠区间 [start, start+dur)，
    段间至少留 min_gap_steps 空隙，段首/末与 0 / num_steps 也保留 min_gap_steps。

    返回按起点升序的 [(start, duration), ...]。

    采样策略：
      - 先尝试"等分 + 抖动"：把 num_steps 均分 n 份，每份内放一段；
      - 若 max_attempts 次都放不下，则缩小 duration 上界重试；
      - 再失败则降低 n_segments。
    """
    dur_lo, dur_hi = duration_range
    n = max(1, n_segments)

    def _try(n_try: int, dur_hi_try: int) -> Optional[List[Tuple[int, int]]]:
        # 估计总占用步数是否放得下
        #   需要：n * dur_lo + (n + 1) * min_gap_steps <= num_steps
        if n_try * dur_lo + (n_try + 1) * min_gap_steps > num_steps:
            return None

        for _ in range(max_attempts):
            # 把 num_steps 均分 n_try 份，每份内采一个段
            slot_len = num_steps // n_try
            if slot_len <= dur_lo + 2 * min_gap_steps:
                # 每槽太小，降低 dur_hi 再试
                return None
            segs: List[Tuple[int, int]] = []
            ok = True
            for k in range(n_try):
                slot_start = k * slot_len
                slot_end = (k + 1) * slot_len if k < n_try - 1 else num_steps
                dur = int(rng.integers(dur_lo, min(dur_hi_try, slot_end - slot_start - 2 * min_gap_steps) + 1))
                if dur < dur_lo:
                    ok = False
                    break
                latest_start = slot_end - dur - min_gap_steps
                earliest_start = slot_start + min_gap_steps
                if latest_start < earliest_start:
                    ok = False
                    break
                start = int(rng.integers(earliest_start, latest_start + 1))
                segs.append((start, dur))
            if ok:
                return segs
        return None

    # 尝试：原参数 → 缩小 dur_hi → 减少段数
    cur_dur_hi = dur_hi
    cur_n = n
    while cur_n >= 1:
        while cur_dur_hi >= dur_lo:
            res = _try(cur_n, cur_dur_hi)
            if res is not None:
                return res
            cur_dur_hi = max(dur_lo, cur_dur_hi - 1)
            if cur_dur_hi == dur_lo:
                res = _try(cur_n, cur_dur_hi)
                if res is not None:
                    return res
                break
        cur_n -= 1
        cur_dur_hi = dur_hi  # 下一轮重置

    # 兜底：至少返回一段，起点 min_gap_steps，持续 dur_lo
    dur = min(dur_lo, max(1, num_steps - 2 * min_gap_steps))
    return [(max(0, min_gap_steps), max(1, dur))]


def _build_yaw_rate_from_segments(
    num_steps: int,
    segments: List[Tuple[int, int, float]],
    rng: np.random.Generator,
    drift_deg_s: float = 0.0,
) -> np.ndarray:
    """
    segments: [(start, duration, angle_rad), ...]，每段内用常数 yaw_rate
              = angle_rad / (duration * TIME_STEP)。
    空隙处 yaw_rate = 0，叠加极小的 gaussian drift（drift_deg_s）。
    """
    T = num_steps
    yaw = np.zeros(T, dtype=np.float64)

    # 段内常数 yaw rate
    max_yaw_rate_rad_s = math.radians(MAX_YAW_RATE_DEG_S)
    for start, dur, ang in segments:
        if dur <= 0:
            continue
        rate = ang / (dur * TIME_STEP)
        # 裁剪到 yaw rate 上限（理论上 apply_data_config 保证不会超）
        if rate > max_yaw_rate_rad_s:
            rate = max_yaw_rate_rad_s
        elif rate < -max_yaw_rate_rad_s:
            rate = -max_yaw_rate_rad_s
        end = min(T, start + dur)
        yaw[start:end] = rate

    # 非常小的漂移（车道内的方向抖动感）
    if drift_deg_s > 0:
        yaw = yaw + rng.normal(0.0, math.radians(drift_deg_s), size=T)
        yaw = np.clip(yaw, -max_yaw_rate_rad_s, max_yaw_rate_rad_s)

    return yaw


def sample_yaw_rate_profile_rad_s(
    traj_type: str,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """按 traj_type 采样一条长度 num_steps 的 yaw-rate 曲线（rad/s）。"""
    if traj_type == "straight":
        # 仅极小漂移
        drift = float(STRAIGHT_CFG.get("heading_drift_deg_s", 0.2))
        return _build_yaw_rate_from_segments(
            num_steps=num_steps,
            segments=[],
            rng=rng,
            drift_deg_s=drift,
        )

    if traj_type in ("left_turn", "right_turn"):
        sign = +1.0 if traj_type == "left_turn" else -1.0
        n_lo, n_hi = TURN_CFG["n_turns_range"]
        a_lo, a_hi = TURN_CFG["angle_range_deg"]
        d_range = TURN_CFG["duration_range_steps"]
        gap = int(TURN_CFG["min_gap_steps"])

        n_turns = int(rng.integers(n_lo, n_hi + 1))
        segs_raw = _sample_nonoverlap_segments(
            num_steps=num_steps,
            n_segments=n_turns,
            duration_range=d_range,
            min_gap_steps=gap,
            rng=rng,
        )
        segments: List[Tuple[int, int, float]] = []
        for start, dur in segs_raw:
            mag_deg = float(rng.uniform(a_lo, a_hi))
            segments.append((start, dur, sign * math.radians(mag_deg)))
        # 直线段也加一点极小 drift，让非转向段不是完美的 0
        drift = float(TURN_CFG.get("straight_drift_deg_s", 0.0))
        return _build_yaw_rate_from_segments(
            num_steps=num_steps,
            segments=segments,
            rng=rng,
            drift_deg_s=drift,
        )

    if traj_type == "s_curve":
        n_lo, n_hi = S_CURVE_CFG["n_segments_range"]
        a_lo, a_hi = S_CURVE_CFG["angle_range_deg"]
        seg_range = S_CURVE_CFG["segment_len_range_steps"]
        gap_lo, gap_hi = S_CURVE_CFG["gap_steps_range"]
        first_dir = S_CURVE_CFG["first_direction"]

        n_segs = int(rng.integers(n_lo, n_hi + 1))
        # 用 _sample_nonoverlap_segments 放置位置，但它要的是统一 min_gap；
        # 这里用 gap 随机下界作为 min_gap，后续再对段顺序做"交替"处理。
        min_gap = int(gap_lo)
        segs_raw = _sample_nonoverlap_segments(
            num_steps=num_steps,
            n_segments=n_segs,
            duration_range=seg_range,
            min_gap_steps=max(0, min_gap),
            rng=rng,
        )
        # 按 start 升序已有；构造交替符号
        if first_dir == "random":
            sign = +1.0 if rng.random() < 0.5 else -1.0
        elif first_dir == "left":
            sign = +1.0
        else:
            sign = -1.0

        # gap 也可以再随机些，但段间已经满足 min_gap，这里只影响语义
        _ = gap_hi

        segments = []
        for i, (start, dur) in enumerate(segs_raw):
            s = sign if (i % 2 == 0) else -sign
            mag_deg = float(rng.uniform(a_lo, a_hi))
            segments.append((start, dur, s * math.radians(mag_deg)))
        drift = float(S_CURVE_CFG.get("drift_deg_s", 0.0))
        return _build_yaw_rate_from_segments(
            num_steps=num_steps,
            segments=segments,
            rng=rng,
            drift_deg_s=drift,
        )

    if traj_type == "u_turn":
        # 一次 ~180° 的大转向；duration 控制在能装进训练窗口 (in_len) 内。
        n_lo, n_hi = U_TURN_CFG["n_turns_range"]
        a_lo, a_hi = U_TURN_CFG["angle_range_deg"]
        d_range = U_TURN_CFG["duration_range_steps"]
        gap = int(U_TURN_CFG["min_gap_steps"])
        direction = U_TURN_CFG["direction"]

        n_turns = int(rng.integers(n_lo, n_hi + 1))
        segs_raw = _sample_nonoverlap_segments(
            num_steps=num_steps,
            n_segments=n_turns,
            duration_range=d_range,
            min_gap_steps=gap,
            rng=rng,
        )

        # 方向决定：整条轨迹里每一次掉头都用同一个方向，避免和 s_curve 混淆。
        # （n_turns_range 默认 [1,1]，所以这里基本上就是单次的方向。）
        if direction == "random":
            sign = +1.0 if rng.random() < 0.5 else -1.0
        elif direction == "left":
            sign = +1.0
        else:
            sign = -1.0

        segments = []
        for start, dur in segs_raw:
            mag_deg = float(rng.uniform(a_lo, a_hi))
            segments.append((start, dur, sign * math.radians(mag_deg)))
        drift = float(U_TURN_CFG.get("straight_drift_deg_s", 0.0))
        return _build_yaw_rate_from_segments(
            num_steps=num_steps,
            segments=segments,
            rng=rng,
            drift_deg_s=drift,
        )

    # 未知类别：当作直线
    return _build_yaw_rate_from_segments(
        num_steps=num_steps,
        segments=[],
        rng=rng,
        drift_deg_s=0.0,
    )


# ======================= 积分器 =========================

def integrate_trajectory(
    x0_km: float,
    y0_km: float,
    z0_km: float,
    heading0_rad: float,
    speed_profile_kmh: np.ndarray,
    yaw_rate_rad_s: np.ndarray,
    dt_s: float,
    use_vertical: bool,
    vz_range_kmps: Tuple[float, float],
    rng: np.random.Generator,
) -> List[Tuple[float, float, float, float, float, float]]:
    """前向欧拉积分，输出与旧版 simulate_trajectory 同构：
       [(x, y, z, vx, vy, vz), ...]，长度 = len(speed_profile_kmh)
       单位：位置 km，速度 km/s
    """
    T = len(speed_profile_kmh)
    assert len(yaw_rate_rad_s) == T

    if use_vertical:
        vz = float(rng.uniform(vz_range_kmps[0], vz_range_kmps[1]))
    else:
        vz = 0.0

    states: List[Tuple[float, float, float, float, float, float]] = []
    x, y, z = float(x0_km), float(y0_km), float(z0_km)
    heading = float(heading0_rad)

    for t in range(T):
        v_kmps = float(speed_profile_kmh[t]) / 3600.0
        vx = v_kmps * math.cos(heading)
        vy = v_kmps * math.sin(heading)
        states.append((x, y, z, vx, vy, vz))

        # 积分位置
        x += vx * dt_s
        y += vy * dt_s
        z += vz * dt_s

        # 更新 heading（用本步的 yaw_rate）
        heading = heading + float(yaw_rate_rad_s[t]) * dt_s

    return states


# ======================= 轨迹统计（回填 CSV 兼容列） =========================

def _infer_compat_cols(
    speed_profile_kmh: np.ndarray,
    n_segments: int,
    has_stop: bool,
) -> Tuple[str, float, float]:
    """为 CSV 中的 motion_model / base_speed_kmh / accel_mps2 列回填
    一个"事后统计"的值，仅为兼容，不影响下游 dataset。

    语义约定（与速度曲线采样器一致）：
      - 若本条轨迹只有 1 段速度目标（n_segments == 1）且没有触发 stop，
        虽然有 speed_jitter 抖动，但整体是"围绕同一个目标速度"，记作 CV；
      - 否则（多段 / 停车段）记作 CA。
    """
    T = len(speed_profile_kmh)
    if T == 0:
        return "CV", 0.0, 0.0

    mean_v = float(np.mean(speed_profile_kmh))
    if T >= 2:
        dv_kmh = np.diff(speed_profile_kmh)
        dv_mps2 = dv_kmh * (1000.0 / 3600.0) / max(TIME_STEP, 1e-6)
        mean_abs_a = float(np.mean(np.abs(dv_mps2)))
    else:
        mean_abs_a = 0.0

    motion_model = "CV" if (n_segments <= 1 and not has_stop) else "CA"
    return motion_model, mean_v, mean_abs_a


# ======================= 主函数 =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle trajectories (sampled).")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="config.yaml 路径（相对 new_plan/lstm1/ 或绝对路径）。",
    )
    parser.add_argument(
        "--num-per-type",
        type=int,
        default=None,
        help="可选：覆盖 config.data.num_traj_per_type 中所有类的值，便于小量预生成。",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    cfg = load_config(config_path)
    apply_data_config(cfg, project_root)

    if args.num_per_type is not None:
        n_override = int(args.num_per_type)
        for k in NUM_TRAJ_PER_TYPE:
            NUM_TRAJ_PER_TYPE[k] = n_override

    # 随机源（Python random + numpy Generator 同种子）
    rng_py = random.Random(RANDOM_SEED)
    rng_np = np.random.default_rng(RANDOM_SEED)

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

        per_type_counts: Dict[str, int] = {t: 0 for t in TRAJ_TYPES}

        for traj_type in TRAJ_TYPES:
            n = int(NUM_TRAJ_PER_TYPE.get(traj_type, 0))
            for _ in range(n):
                traj_id += 1

                # --- 1) 初始位姿 ---
                x0 = rng_py.uniform(*INIT_X_RANGE)
                y0 = rng_py.uniform(*INIT_Y_RANGE)
                z0 = rng_py.uniform(*INIT_Z_RANGE)
                heading0 = rng_py.uniform(0.0, 2.0 * math.pi)

                # --- 2) 速度曲线 ---
                speed_profile, n_speed_segments, has_stop = sample_speed_profile_kmh(
                    NUM_STEPS, rng_np
                )

                # --- 3) yaw-rate 曲线 ---
                yaw_rate = sample_yaw_rate_profile_rad_s(
                    traj_type, NUM_STEPS, rng_np
                )

                # --- 4) 积分 ---
                states = integrate_trajectory(
                    x0_km=x0,
                    y0_km=y0,
                    z0_km=z0,
                    heading0_rad=heading0,
                    speed_profile_kmh=speed_profile,
                    yaw_rate_rad_s=yaw_rate,
                    dt_s=TIME_STEP,
                    use_vertical=USE_VERTICAL_MOTION,
                    vz_range_kmps=VERTICAL_VZ_RANGE_KMPS,
                    rng=rng_np,
                )

                # --- 5) 回填兼容列 ---
                motion_model, base_speed_kmh, accel_mps2 = _infer_compat_cols(
                    speed_profile, n_speed_segments, has_stop
                )

                for step_idx, (x, y, z, vx, vy, vz) in enumerate(states):
                    time_s = step_idx * TIME_STEP
                    writer.writerow(
                        [
                            traj_id,
                            motion_model,
                            base_speed_kmh,
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
                        ]
                    )
                    num_rows += 1

                per_type_counts[traj_type] += 1

    print(
        f"[generate_trajs] 生成完成：{traj_id} 条轨迹，总 {num_rows} 行。\n"
        f"  每类数量：{per_type_counts}\n"
        f"  输出：{out_path}"
    )


if __name__ == "__main__":
    main()
