#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fullnet_pytorch.py
-----------------------
直接在 PyTorch 下调用 FullNet，检查 3 条轨迹 + 意图/威胁度 输出是否合理，
并（可选）与 full_net.onnx 的输出做对比。

用法示例（在 model_fusion_v0 目录下）：

    python test_fullnet_pytorch.py \
        --traj-ckpt ../203_prediction_multi_pytorch_without_map_v0.2/checkpoints/20251201163510/best_lstm_epoch006_valloss0.0296.pt \
        --intent-ckpt ../203_prediction_intention_pytorch_v0/checkpoints_intent/20251203215429/best_intent_epoch003_valloss0.0777_acc96.39_mae2.972.pt \
        --config config_mean_std.yaml \
        --csv test_onnx/synthetic_trajectories.csv \
        --onnx test_onnx/full_net.onnx        # 可选，用于对比 PyTorch vs ONNX

"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# 可选：对比 ONNX 时才需要 onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ----------------------------------------------------------------------
# 路径 & import
# ----------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR  # model_fusion_v0 目录


# 直接复用 export_full_net.py 里的工具函数
from export_full_net import (
    load_traj_net,
    load_intent_net,
    build_fullnet_from_config,
)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def random_traj_from_csv(
    csv_path: Path, hist_len: int = 20, fut_len: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CSV 中随机取一条轨迹，返回：
      hist_raw: [hist_len, 6]
      fut_raw : [fut_len, 6]
    要求 CSV 至少包含：
      x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps
    若有 'traj_id' 列：按轨迹 ID 分组，否则假定每条轨迹 30 步顺序拼接。
    """
    csv_path = csv_path.resolve()
    df = pd.read_csv(csv_path)

    cols = ["x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"CSV 中缺少列: '{c}'")

    need_steps = hist_len + fut_len

    if "traj_id" in df.columns:
        traj_ids = df["traj_id"].unique()
        tid = random.choice(list(traj_ids))

        sub = df[df["traj_id"] == tid]
        # 如果有 step/t_idx 就按它排序，没有就用原始顺序
        if "step" in sub.columns:
            sub = sub.sort_values(by="step")
        elif "t_idx" in sub.columns:
            sub = sub.sort_values(by="t_idx")
        else:
            sub = sub.sort_index()

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


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-ckpt", type=str, required=True,
                        help="轨迹 LSTM ckpt 路径（.pt）")
    parser.add_argument("--intent-ckpt", type=str, required=True,
                        help="意图/威胁 ckpt 路径（.pt）")
    parser.add_argument("--config", type=str, required=True,
                        help="config_mean_std.yaml 路径")
    parser.add_argument("--csv", type=str, default=str(THIS_DIR / "test_onnx" / "synthetic_trajectories.csv"),
                        help="用于测试的 synthetic_trajectories.csv 路径")
    parser.add_argument("--onnx", type=str, default=None,
                        help="可选：full_net.onnx 路径，用于对比 PyTorch vs ONNX")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    traj_ckpt = Path(args.traj_ckpt)
    intent_ckpt = Path(args.intent_ckpt)
    cfg_path = Path(args.config)
    csv_path = Path(args.csv)

    # 1) 读取 config_mean_std.yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Info] 读取配置: {cfg_path}")

    # 2) 加载子网络
    traj_net = load_traj_net(traj_ckpt, device)
    intent_net = load_intent_net(intent_ckpt, device)

    # 3) 构建 FullNet（里面会加载 npz 的 mean/std）
    full_net = build_fullnet_from_config(cfg, traj_net, intent_net)
    full_net.to(device)
    full_net.eval()

    print("[Info] FullNet 构建完成。")

    # 4) 从 CSV 取一条轨迹
    hist_len, fut_len, feat_dim, n_modes = 20, 10, 6, 3
    hist_raw, fut_raw = random_traj_from_csv(csv_path, hist_len=hist_len, fut_len=fut_len)

    x_raw = torch.from_numpy(hist_raw).unsqueeze(0).to(device)  # [1, 20, 6]

    # 5) PyTorch 前向
    with torch.no_grad():
        out_full = full_net(x_raw)  # [1, 3*68] 或 [1,3,68]
    print(f"[Info] FullNet(Pytorch) 输出 shape: {tuple(out_full.shape)}")

    out_pt = out_full.detach().cpu().numpy()
    if out_pt.ndim == 2 and out_pt.shape[0] == 1:
        out_pt = out_pt.reshape(1, n_modes, -1)        # [1,3,68]
    elif out_pt.ndim == 3 and out_pt.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Unexpected PyTorch output shape: {out_pt.shape}")

    out_pt_3_68 = out_pt[0]
    if out_pt_3_68.shape[1] != 68:
        raise ValueError(f"PyTorch 输出最后一维不是 68，而是 {out_pt_3_68.shape[1]}")

    fut_preds_pt = out_pt_3_68[:, :60].reshape(n_modes, fut_len, feat_dim)
    intent_class_pt = out_pt_3_68[:, 60]
    threat_prob_pt = out_pt_3_68[:, 61]
    radius_pt = out_pt_3_68[:, 65]
    conf_pt = out_pt_3_68[:, 66]
    mode_prob_pt = out_pt_3_68[:, 67]

    for m in range(n_modes):
        print(
            f"[PT Mode {m}] intent={int(intent_class_pt[m])}, "
            f"thr={threat_prob_pt[m]:.3f}, "
            f"radius={radius_pt[m]:.3f}, "
            f"conf={conf_pt[m]:.3f}, "
            f"mode_p={mode_prob_pt[m]:.3f}"
        )

    # ------------------------------------------------------------------
    # 6) （可选）与 ONNX 对比
    # ------------------------------------------------------------------
    if args.onnx is not None:
        if ort is None:
            print("[Warn] 未安装 onnxruntime，无法对比 ONNX。")
        else:
            onnx_path = Path(args.onnx).resolve()
            print(f"[Info] 对比 ONNX: {onnx_path}")

            sess = ort.InferenceSession(
                onnx_path.as_posix(), providers=["CPUExecutionProvider"]
            )
            in_name = sess.get_inputs()[0].name
            x_np = hist_raw.astype(np.float32)[None, :, :]
            out_onnx = sess.run(None, {in_name: x_np})[0]

            if out_onnx.ndim == 2 and out_onnx.shape[0] == 1:
                out_onnx = out_onnx.reshape(1, n_modes, -1)
            elif out_onnx.ndim == 3 and out_onnx.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected ONNX output shape: {out_onnx.shape}")

            out_onnx_3_68 = out_onnx[0]

            if out_onnx_3_68.shape != out_pt_3_68.shape:
                print(
                    f"[Warn] PyTorch 输出形状 {out_pt_3_68.shape} "
                    f"与 ONNX 输出形状 {out_onnx_3_68.shape} 不一致，无法直接比较。"
                )
            else:
                diff = np.abs(out_pt_3_68 - out_onnx_3_68).max()
                print(f"[Compare] max |PyTorch - ONNX| = {diff:.6e}")

    # ------------------------------------------------------------------
    # 7) 画图（用 PyTorch 的 fut_preds_pt）
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(hist_raw[:, 0], hist_raw[:, 1], "k-o", label="history (20)")
    plt.plot(fut_raw[:, 0], fut_raw[:, 1], "g-o", label="future GT (10)")

    colors = ["r", "b", "m"]
    for m in range(n_modes):
        traj = fut_preds_pt[m]
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            color=colors[m],
            marker="o",
            linestyle="--",
            label=f"PT mode{m}: intent={int(intent_class_pt[m])}, "
                  f"thr={threat_prob_pt[m]:.2f}, p={mode_prob_pt[m]:.2f}",
        )

    plt.xlabel("x_km")
    plt.ylabel("y_km")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("FullNet PyTorch test (raw input)")

    out_png = THIS_DIR / "test_fullnet_pytorch.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] 已保存可视化到: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
