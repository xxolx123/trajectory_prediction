#!/usr/bin/env python3
"""
generate_traj_param_sweep.py

根据参数组合生成轨迹数据：
- 运动模型：CV / CA
- 速度档：30 / 50 / 80 km/h（可配置）
- 加速度档：(-0.5, 0.0, 0.5) m/s^2（只在 CA 下有用）
- 轨迹类型：直线 / 左转 / 右转 / S 型弯（只转一次，不再绕圈）

每种 (motion_model, speed, accel, traj_type) 组合生成多条轨迹。
每条轨迹是单独的，不分多模态 group。

输出 CSV：traj_param_sweep.csv
每行对应一条轨迹中的一个时间步。
单位：
- 位置 x,y,z：km
- 速度 vx,vy,vz：km/s
"""

import math
import csv
import random
from pathlib import Path

# ======================= 配置区域 =========================

TIME_STEP = 60.0          # 每步时间间隔（秒）
NUM_STEPS = 100           # 每条轨迹总步数

MOTION_MODELS = ["CV", "CA"]      # 匀速 / 匀加速
SPEEDS_KMH = [20.0, 30.0, 50.0]   # 初始速度档（km/h）

# ===== 关键部分：根据 ‘100 分钟内最多变化 20 km/h’ 自动算加速度 =====
DV_MAX_KMH = 20.0                          # 最大希望变化的车速（km/h）
TOTAL_TIME = (NUM_STEPS - 1) * TIME_STEP   # 总时间（秒），N 个点有 N-1 个间隔

# 三档“加速等级”：-1, 0, +1，分别代表 “减速 20 km/h / 匀速 / 加速 20 km/h”
ACCEL_LEVELS = [-1.0, 0.0, 1.0]

DV_MAX_MPS = DV_MAX_KMH * 1000.0 / 3600.0  # km/h → m/s
BASE_A_MPS2 = DV_MAX_MPS / TOTAL_TIME      # 使得 |Δv| = 20km/h 的物理加速度

ACCELS_MPS2 = [alpha * BASE_A_MPS2 for alpha in ACCEL_LEVELS]
# 现在 ACCELS_MPS2 大概是 [-9.26e-4, 0, 9.26e-4] m/s^2

MIN_SPEED_KMH = 15.0    # 你可以保留这一行避免减速到 0


TRAJ_TYPES = [
    "straight",   # 直线
    "left_turn",  # 一次左转
    "right_turn", # 一次右转
    "s_curve",    # S 型：左转再右转，最后航向回到初始
]

# 一次转弯的总角度（度）
TURN_TOTAL_DEG = 60.0   # 左/右转的总转角
# 左/右转在时间轴上的区段（索引是 [0, NUM_STEPS-1]）
TURN_START_IDX = 50     # 从第 10 步之后开始逐渐转弯（含）
# S 形弯中“左转段”和“右转段”的分界（大致前 1/3 左转，中间直一点，后 1/3 右转回去）
S_LEFT_START = 33
S_LEFT_END   = 66       # [S_LEFT_START, S_LEFT_END] 左转到 +θ
# [S_LEFT_END, NUM_STEPS-1] 再右转回到 0

NUM_TRAJ_PER_COMBO = 50

INIT_X_RANGE = (-30.0, 30.0)
INIT_Y_RANGE = (-30.0, 30.0)
INIT_Z_RANGE = (0,0)

USE_VERTICAL_MOTION = False
VERTICAL_VZ_RANGE_KMPS = (-0.005, 0.005)  # km/s

RANDOM_SEED = 42
OUTPUT_CSV = "synthetic_trajectories.csv"

# ========================================================


def heading_schedule(traj_type: str,
                     step_idx: int,
                     num_steps: int,
                     base_heading_rad: float) -> float:
    """
    给定轨迹类型 + 步索引，返回当前步的航向角（弧度）。

    设计：
      - straight: 一直是 base_heading
      - left_turn/right_turn:
          前 TURN_START_IDX 步保持 base_heading，
          后面步数线性从 base_heading 过渡到 base_heading ± TURN_TOTAL_DEG
      - s_curve:
          [0, S_LEFT_START]      ：base_heading
          [S_LEFT_START, S_LEFT_END] 左转到 base_heading + θ
          [S_LEFT_END, N-1]      ：再右转回 base_heading
    """
    theta = math.radians(TURN_TOTAL_DEG)
    N = num_steps
    i = step_idx

    if traj_type == "straight":
        return base_heading_rad

    if traj_type in ("left_turn", "right_turn"):
        if i <= TURN_START_IDX:
            frac = 0.0
        else:
            # 从 TURN_START_IDX -> N-1 线性过渡到 1.0
            denom = max(1, (N - 1 - TURN_START_IDX))
            frac = (i - TURN_START_IDX) / denom
            frac = max(0.0, min(1.0, frac))

        signed_theta = theta if traj_type == "left_turn" else -theta
        return base_heading_rad + signed_theta * frac

    if traj_type == "s_curve":
        # S: 左转到 +θ，再右转回 base_heading
        if i <= S_LEFT_START:
            return base_heading_rad

        if S_LEFT_START < i <= S_LEFT_END:
            # 左转段：从 0 -> +θ
            denom = max(1, (S_LEFT_END - S_LEFT_START))
            frac = (i - S_LEFT_START) / denom
            frac = max(0.0, min(1.0, frac))
            return base_heading_rad + theta * frac

        # 右转段：从 +θ -> 0
        denom = max(1, (N - 1 - S_LEFT_END))
        frac = (i - S_LEFT_END) / denom
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
) -> list[tuple[float, float, float, float, float, float]]:
    """
    生成一条轨迹，返回列表：
      [(x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps), ...]
      长度为 NUM_STEPS。
    """

    # 速度 & 加速度单位转换
    v_kmps = base_speed_kmh / 3600.0       # km/h → km/s
    a_kmps2 = accel_mps2 / 1000.0         # m/s^2 → km/s^2
    min_speed_kmps = MIN_SPEED_KMH / 3600.0

    # 初始状态
    x = init_x_km
    y = init_y_km
    z = init_z_km

    if USE_VERTICAL_MOTION:
        vz = random.uniform(*VERTICAL_VZ_RANGE_KMPS)
    else:
        vz = 0.0

    states: list[tuple[float, float, float, float, float, float]] = []

    for step in range(NUM_STEPS):
        # 当前步的航向角（根据轨迹类型一次/两次平滑转弯）
        heading = heading_schedule(traj_type, step, NUM_STEPS, init_heading_rad)

        # 根据当前速度标量 + 航向角得到水平速度分量
        vx = v_kmps * math.cos(heading)
        vy = v_kmps * math.sin(heading)

        # 记录状态（位置是本步开始时的位置）
        states.append((x, y, z, vx, vy, vz))

        # 用当前速度积分出下一步的位置
        x += vx * TIME_STEP
        y += vy * TIME_STEP
        z += vz * TIME_STEP

        # 匀加速模型：更新速度标量
        if motion_model == "CA":
            v_kmps += a_kmps2 * TIME_STEP
            if v_kmps < min_speed_kmps:
                v_kmps = min_speed_kmps

    return states


def main():
    random.seed(RANDOM_SEED)

    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
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
        ])

        traj_id = 0
        num_rows = 0

        for motion_model in MOTION_MODELS:
            for speed_kmh in SPEEDS_KMH:

                # CV 下加速度强制为 0
                if motion_model == "CV":
                    accel_list = [0.0]
                else:
                    accel_list = ACCELS_MPS2

                for accel_mps2 in accel_list:
                    for traj_type in TRAJ_TYPES:
                        for _ in range(NUM_TRAJ_PER_COMBO):
                            traj_id += 1

                            # 每条轨迹的初始状态随机
                            init_x = random.uniform(*INIT_X_RANGE)
                            init_y = random.uniform(*INIT_Y_RANGE)
                            init_z = random.uniform(*INIT_Z_RANGE)
                            init_heading = random.uniform(0.0, 2.0 * math.pi)

                            states = simulate_trajectory(
                                motion_model=motion_model,
                                base_speed_kmh=speed_kmh,
                                accel_mps2=accel_mps2,
                                traj_type=traj_type,
                                init_x_km=init_x,
                                init_y_km=init_y,
                                init_z_km=init_z,
                                init_heading_rad=init_heading,
                            )

                            for step_idx, (x, y, z, vx, vy, vz) in enumerate(states):
                                time_s = step_idx * TIME_STEP
                                writer.writerow([
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
                                ])
                                num_rows += 1

    print(f"生成完成：{traj_id} 条轨迹，总 {num_rows} 行，输出到 {out_path}")


if __name__ == "__main__":
    main()
