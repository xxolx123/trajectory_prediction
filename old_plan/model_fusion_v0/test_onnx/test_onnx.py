#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_onnx.py （纯 ONNX 版本）
---------------------------
假设 full_net.onnx 已经把“输入归一化 + 输出反归一化 + 意图威胁预测”都写在网络里。

流程：
  - 从 synthetic_trajectories.csv 随机取一条轨迹（30 步）
  - 前 20 步作为原始输入 [1, 20, 6] 喂给 ONNX
  - 输出 [1, 3*68] 或 [1, 3, 68]
  - 解析 3 条预测轨迹 + 意图 + 威胁度 + 置信度
  - 在 x-y 平面画：20 步历史 + 10 步 GT + 3 条预测轨迹
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
ONNX_PATH = THIS_DIR / "full_net.onnx"
CSV_PATH = THIS_DIR / "synthetic_trajectories.csv"


def random_traj_from_csv(csv_path: Path, hist_len: int = 20, fut_len: int = 10):
    """
    从 CSV 中随机取一条轨迹，返回：
      hist_raw: [hist_len, 6]
      fut_raw : [fut_len, 6]
    要求 CSV 至少包含：
      x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps
    若有 'traj_id' 列：按轨迹 ID 分组，否则假定每条轨迹 30 步顺序拼接。
    """
    df = pd.read_csv(csv_path)

    cols = ["x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"CSV 中缺少列: '{c}'")

    need_steps = hist_len + fut_len

    if "traj_id" in df.columns:
        traj_ids = df["traj_id"].unique()
        tid = random.choice(list(traj_ids))

        # 先按 traj_id 过滤
        sub = df[df["traj_id"] == tid]

        # 如果有 step/t_idx 之类就按它排序，没有就保持原始顺序
        if "step" in sub.columns:
            sub = sub.sort_values(by="step")
        elif "t_idx" in sub.columns:
            sub = sub.sort_values(by="t_idx")
        else:
            sub = sub.sort_index()  # 按行号排序，等价于保持原来的顺序

        if len(sub) < need_steps:
            raise ValueError(f"轨迹 {tid} 步数不足 {need_steps}，只有 {len(sub)} 步")

        sub = sub.iloc[:need_steps]

    else:
        total = len(df)
        if total < need_steps:
            raise ValueError(f"CSV 总步数不足 {need_steps}：{total}")
        stride = 30  # 每条 30 步
        max_traj = total // stride
        tidx = random.randrange(max_traj)
        start = tidx * stride
        sub = df.iloc[start : start + need_steps]

    arr = sub[cols].to_numpy(dtype=np.float32)
    hist_raw = arr[:hist_len, :]
    fut_raw = arr[hist_len : hist_len + fut_len, :]
    return hist_raw, fut_raw


def main():
    hist_len = 20
    fut_len = 10
    feat_dim = 6
    n_modes = 3

    print(f"[Info] 使用 ONNX: {ONNX_PATH}")
    print(f"[Info] 使用 CSV:  {CSV_PATH}")

    hist_raw, fut_raw = random_traj_from_csv(CSV_PATH, hist_len=hist_len, fut_len=fut_len)

    # 模型输入：[1, 20, 6] 原始物理量
    x_input = hist_raw.astype(np.float32)[None, :, :]  # [1, 20, 6]

    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x_input})[0]

    print(f"[Info] onnx 输出 shape: {out.shape}")

    if out.ndim == 2 and out.shape[0] == 1:
        out = out.reshape(1, n_modes, -1)      # [1, 3, 68]
    elif out.ndim == 3 and out.shape[0] == 1:
        pass
    else:
        raise ValueError(f"未知输出形状: {out.shape}")

    out_3_68 = out[0]                          # [3, 68]
    print(out_3_68)

    if out_3_68.shape[1] != 68:
        raise ValueError(f"最后一维不是 68，而是 {out_3_68.shape[1]}")

    fut_preds = out_3_68[:, :60].reshape(n_modes, fut_len, feat_dim)  # [3, 10, 6]
    intent_class = out_3_68[:, 60]
    threat_prob = out_3_68[:, 61]
    radius = out_3_68[:, 65]
    conf = out_3_68[:, 66]
    mode_prob = out_3_68[:, 67]

    for m in range(n_modes):
        print(
            f"[Mode {m}] intent={int(intent_class[m])}, "
            f"threat={threat_prob[m]:.3f}, "
            f"radius={radius[m]:.3f}, "
            f"conf={conf[m]:.3f}, "
            f"mode_p={mode_prob[m]:.3f}"
        )

    # 可视化 x-y
    plt.figure(figsize=(6, 6))
    plt.plot(hist_raw[:, 0], hist_raw[:, 1], "k-o", label="history (20)")
    plt.plot(fut_raw[:, 0], fut_raw[:, 1], "g-o", label="future GT (10)")

    colors = ["r", "b", "m"]
    for m in range(n_modes):
        traj = fut_preds[m]
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            color=colors[m],
            marker="o",
            linestyle="--",
            label=f"mode{m}: intent={int(intent_class[m])}, thr={threat_prob[m]:.2f}, p={mode_prob[m]:.2f}",
        )

    plt.xlabel("x_km")
    plt.ylabel("y_km")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("FullNet ONNX test (raw input)")

    out_png = THIS_DIR / "test_onnx_result.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 已保存可视化到: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
