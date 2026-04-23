#!/usr/bin/env python3
"""
visualize_trajectories.py

可视化 generate_traj_param_sweep.py 生成的轨迹数据（synthetic_trajectories.csv）。

两种模式：
  1) 默认：每种 traj_type 随机采样一条轨迹绘制
  2) --overlay-all：每种 traj_type 在一个子图上叠加绘制该类型的所有轨迹

用法示例：
  # 默认：每种类型随机画 1 条
  python visualize_trajectories.py --csv synthetic_trajectories.csv

  # 叠加模式：每种类型把所有轨迹都画在一个子图
  python visualize_trajectories.py --csv synthetic_trajectories.csv --overlay-all

  # 改随机种子
  python visualize_trajectories.py --csv synthetic_trajectories.csv --seed 123
"""

import csv
import argparse
import random
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ====================== 数据读取 ======================

def load_trajectories(csv_path):
    """
    从 CSV 中读取轨迹数据。

    输入 CSV 需要包含以下列：
      traj_id, motion_model, base_speed_kmh, accel_mps2,
      traj_type, step_idx, time_s, x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps

    返回：
      traj_dict: {
        traj_id: {
          "motion_model": str,
          "base_speed_kmh": float,
          "accel_mps2": float,
          "traj_type": str,
          "steps": [(x_km, y_km, z_km, time_s), ...]  # 已按时间排序
        },
        ...
      }
      traj_types: 排好序的 traj_type 列表
    """
    traj_dict = {}
    traj_types_set = set()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            traj_id = int(row["traj_id"])
            traj_type = row["traj_type"]
            motion_model = row["motion_model"]
            base_speed_kmh = float(row["base_speed_kmh"])
            accel_mps2 = float(row["accel_mps2"])

            time_s = float(row["time_s"])
            x_km = float(row["x_km"])
            y_km = float(row["y_km"])
            z_km = float(row["z_km"])

            if traj_id not in traj_dict:
                traj_dict[traj_id] = {
                    "motion_model": motion_model,
                    "base_speed_kmh": base_speed_kmh,
                    "accel_mps2": accel_mps2,
                    "traj_type": traj_type,
                    "steps": []
                }

            traj_dict[traj_id]["steps"].append((x_km, y_km, z_km, time_s))
            traj_types_set.add(traj_type)

    # 对每条轨迹按 time_s 排序，保证顺序正确
    for traj_id, info in traj_dict.items():
        info["steps"].sort(key=lambda v: v[3])

    traj_types = sorted(traj_types_set)
    return traj_dict, traj_types


# ====================== 单条轨迹绘制 ======================

def plot_one_traj_on_axis(ax, traj_info, title=None,
                          line_kw=None, start_kw=None, end_kw=None):
    """
    在给定的坐标轴上画一条轨迹（x-y 平面）。
    line_kw / start_kw / end_kw 是可选的样式 dict。
    """
    xs = [s[0] for s in traj_info["steps"]]
    ys = [s[1] for s in traj_info["steps"]]

    line_kw = line_kw or {}
    start_kw = start_kw or {}
    end_kw = end_kw or {}

    ax.plot(xs, ys, **{"marker": "o", "markersize": 2, "linestyle": "-", **line_kw})

    # 起点方块，终点叉
    ax.scatter(xs[0], ys[0],
               **{"marker": "s", "s": 30, "color": "tab:green", "edgecolor": "k",
                  "zorder": 5, **start_kw})
    ax.scatter(xs[-1], ys[-1],
               **{"marker": "X", "s": 40, "color": "tab:red", "edgecolor": "k",
                  "zorder": 5, **end_kw})

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.axis("equal")
    ax.grid(True)

    if title is not None:
        ax.set_title(title)


# ====================== 主可视化逻辑 ======================

def visualize_per_type(csv_path, seed=42, overlay_all=False):
    """
    每种 traj_type 绘制一张图（子图）：
      - overlay_all=False：每种类型随机采样一条轨迹绘制
      - overlay_all=True ：每种类型叠加绘制该类型所有轨迹
    """
    traj_dict, traj_types = load_trajectories(csv_path)
    print(f"读取完成：{len(traj_dict)} 条轨迹，"
          f"{len(traj_types)} 种 traj_type = {traj_types}")

    if len(traj_types) == 0:
        print("[WARN] CSV 中没有 traj_type 信息，无法绘图。")
        return

    # 按类型收集 traj_id
    type_to_ids = {t: [] for t in traj_types}
    for traj_id, info in traj_dict.items():
        t = info["traj_type"]
        if t in type_to_ids:
            type_to_ids[t].append(traj_id)

    random.seed(seed)

    # 如果是“随机抽一条”模式，先确定每种类型选哪条
    chosen_one = {}
    if not overlay_all:
        for t in traj_types:
            ids = type_to_ids[t]
            if not ids:
                print(f"[WARN] 类型 {t} 没有任何轨迹，跳过。")
                continue
            chosen_id = random.choice(ids)
            chosen_one[t] = chosen_id

        if not chosen_one:
            print("[WARN] 没有任何类型成功选择轨迹。")
            return

        print("随机模式：为各类型选择的 traj_id：")
        for t, tid in chosen_one.items():
            print(f"  traj_type = {t:>10s}  -> traj_id = {tid}")

    # 准备子图：每种类型一个子图
    n_types = len(traj_types)
    cols = 2
    rows = math.ceil(n_types / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

    # 统一展平成一维列表
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        axes = [axes]

    for idx, t in enumerate(traj_types):
        ax = axes[idx]
        ids = type_to_ids[t]

        if not ids:
            ax.set_title(f"type={t} (no traj)")
            ax.axis("off")
            continue

        if overlay_all:
            # 模式 1：叠加该类型所有轨迹
            for traj_id in ids:
                info = traj_dict[traj_id]
                # 所有轨迹用比较浅的线
                plot_one_traj_on_axis(
                    ax, info, title=None,
                    line_kw={"linewidth": 0.8, "alpha": 0.3, "marker": ""},
                    start_kw={"alpha": 0.0},  # 不单独标所有轨迹的起终点
                    end_kw={"alpha": 0.0},
                )

            # 再随机挑一条高亮，方便看清一条的细节
            highlight_id = random.choice(ids)
            highlight_info = traj_dict[highlight_id]
            title = (f"type={t}  (overlay {len(ids)} trajs)\n"
                     f"例：id={highlight_id}, "
                     f"model={highlight_info['motion_model']}, "
                     f"v0={highlight_info['base_speed_kmh']} km/h, "
                     f"a={highlight_info['accel_mps2']} m/s²")
            plot_one_traj_on_axis(
                ax, highlight_info, title=title,
                line_kw={"linewidth": 2.0, "alpha": 1.0, "color": "tab:blue"},
            )

        else:
            # 模式 2：只画这一类型中随机选中的一条
            traj_id = chosen_one.get(t)
            if traj_id is None:
                ax.set_title(f"type={t} (no chosen traj)")
                ax.axis("off")
                continue
            info = traj_dict[traj_id]
            title = (f"type={t}\n"
                     f"model={info['motion_model']}, "
                     f"v0={info['base_speed_kmh']} km/h, "
                     f"a={info['accel_mps2']} m/s²")
            plot_one_traj_on_axis(ax, info, title=title)

    # 删除多余子图（如果类型数不是 rows*cols 的整数倍）
    for j in range(n_types, len(axes)):
        fig.delaxes(axes[j])

    mode_desc = "叠加所有轨迹" if overlay_all else "每类随机一条"
    fig.suptitle(f"每种 traj_type：{mode_desc} (x-y)", fontsize=16)
    plt.tight_layout()
    plt.show()


# ====================== main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="synthetic_trajectories.csv",
                        help="轨迹 CSV 文件路径（默认：traj_param_sweep.csv）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（控制随机选哪条轨迹）")
    parser.add_argument(
        "--overlay-all",
        action="store_true",
        help="开启后：每种 traj_type 在一个子图上叠加绘制所有该类型轨迹；"
             "默认关闭：每种类型只随机画一条"
    )
    args = parser.parse_args()

    visualize_per_type(args.csv, seed=args.seed, overlay_all=args.overlay_all)


if __name__ == "__main__":
    main()
