#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_full_net.py
------------------
用途：
  - 加载已经训练好的：
      * 多模态轨迹网络 traj_net
      * 意图/威胁度网络 intent_net
  - 读取 config_mean_std.yaml 中的：
      * 方案 A / 方案 B 的 mean/std
      * FullNet 的一些配置（可选）
  - 构造 FullNet，并导出为 ONNX 文件，供部署端使用。

使用示例（在项目根目录下，或在 model_fusion/ 下）：
    python export_full_net.py \
        --traj-ckpt ../203_prediction_multi_pytorch_without_map_v0.2/checkpoints/20251201163510/best_lstm_epoch006_valloss0.0296.pt \
        --intent-ckpt ../203_prediction_intention_pytorch_v0/checkpoints_intent/20251203215429/best_intent_epoch003_valloss0.0777_acc96.39_mae2.972.pt \
        --config config_mean_std.yaml \
        --onnx-out full_net.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import numpy as np

import yaml

# 保证可以 import 到 full_net.py 和你原来的两个子网络定义
ROOT_DIR = Path(__file__).resolve().parents[1]   # 假定结构：project_root/model_fusion/export_full_net.py
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 1) 导入 FullNet
from model_fusion_v0.full_net import FullNet


# ============================================================
# 一些工具函数
# ============================================================

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件 {path} 内容不是 dict")
    return cfg

def load_mean_std_from_npz(path: Path):
    """从 .npz 文件中读取 mean/std 或 mean_/scale_。"""
    data = np.load(path)
    keys = set(data.files)
    print(f"[Info] 加载 npz: {path}, keys={keys}")

    if "mean" in keys and "std" in keys:
        mean = data["mean"]
        std = data["std"]
    elif "mean_" in keys and "scale_" in keys:  # sklearn StandardScaler
        mean = data["mean_"]
        std = data["scale_"]
    elif "mu" in keys and "sigma" in keys:
        mean = data["mu"]
        std = data["sigma"]
    else:
        raise ValueError(
            f"无法在 {path} 中找到 mean/std（或 mean_/scale_ / mu/sigma），"
            f"实际 keys={keys}"
        )

    mean = mean.astype("float32")
    std = std.astype("float32")
    return mean, std



def load_traj_net(traj_ckpt: Path, device: torch.device) -> nn.Module:
    """
    加载多模态轨迹网络（MTP LSTM）。

    自动根据 ckpt 路径推断项目根目录：
      ckpt:  .../203_prediction_multi_pytorch_without_map_v0.2/checkpoints/时间戳/best_xxx.pt
      根目录: .../203_prediction_multi_pytorch_without_map_v0.2

    然后：
      1) 把 根目录/code 加到 sys.path
      2) 读取 根目录/config.yaml
      3) 用 train.model_mtp.build_model_from_config(cfg) 构建网络
      4) 加载 ckpt 里的 state_dict
    """
    traj_ckpt = traj_ckpt.resolve()
    # 根目录 = ckpt 的上上级：.../project/checkpoints/ts/best.pt
    proj_root = traj_ckpt.parents[2]
    code_dir = proj_root / "code"
    config_path = proj_root / "config.yaml"

    if not code_dir.is_dir():
        raise FileNotFoundError(f"[traj] 未找到 code 目录: {code_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"[traj] 未找到 config.yaml: {config_path}")

    # 确保可以 import 到项目里的模块
    import sys
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ===== 这里是唯一可能需要你改模块名的地方 =====
    # 我之前和你写多模态 PyTorch 代码时，用的是 train/model_mtp.py，
    # 里面有 build_model_from_config(cfg)。
    try:
        from train.model_mtp import build_model_from_config as build_traj_model
    except ImportError:
        # 如果你项目里叫 train.model，就走这里，把上面那行删掉也可以
        from train.model import build_model_from_config as build_traj_model  # type: ignore

    model = build_traj_model(cfg)  # 之前 trainer 里也是直接把整个 cfg 传进去的

    # ===== 加载 ckpt 权重 =====
    ckpt = torch.load(traj_ckpt, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        # 万一你直接 torch.save(model.state_dict())
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[traj] load_state_dict: missing={missing}, unexpected={unexpected}")

    model.to(device)
    model.eval()
    print(f"[traj] 载入完成: {traj_ckpt}")
    return model



def load_intent_net(intent_ckpt: Path, device: torch.device) -> nn.Module:
    """
    加载意图/威胁度网络 IntentThreatNet。

    目录假设（和你截图一致）：
      .../203_prediction_intention_pytorch_v0/
        ├─ config.yaml
        ├─ code/
        │   └─ train/
        │        └─ intent_model.py  (定义 IntentThreatNet)
        └─ checkpoints_intent/时间戳/best_xxx.pt
    """
    intent_ckpt = intent_ckpt.resolve()
    # ckpt: .../203_prediction_intention_pytorch_v0/checkpoints_intent/时间戳/best_xxx.pt
    proj_root = intent_ckpt.parents[2]          # -> .../203_prediction_intention_pytorch_v0
    code_dir = proj_root / "code"
    config_path = proj_root / "config.yaml"

    if not code_dir.is_dir():
        raise FileNotFoundError(f"[intent] 未找到 code 目录: {code_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"[intent] 未找到 config.yaml: {config_path}")

    # 把 code/ 加到 sys.path，这样才能 import train.intent_model
    import sys, yaml
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ===== 1) 解析 config 里的模型参数 =====
    # 你自己的 config.yaml 里模型那一节可能叫 model_intent / intent_model / model
    model_cfg = cfg.get("model_intent", cfg.get("intent_model", cfg.get("model", {})))

    # 限定一下能传给 IntentThreatNet 的关键字
    allowed_keys = {"in_dim", "window_len", "hidden_dims", "dropout", "num_intent_classes"}
    kwargs = {k: v for k, v in model_cfg.items() if k in allowed_keys}

    # ===== 2) 导入并构造 IntentThreatNet =====
    from train.intent_model import IntentThreatNet

    try:
        model = IntentThreatNet(**kwargs)
        print(f"[intent] 使用 config 中的参数构造 IntentThreatNet: {kwargs}")
    except TypeError as e:
        print(f"[intent] 用 config 参数构造失败({e})，改用默认参数构造 IntentThreatNet")
        model = IntentThreatNet()

    # ===== 3) 加载 ckpt 权重 =====
    ckpt = torch.load(intent_ckpt, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[intent] load_state_dict: missing={missing}, unexpected={unexpected}")
    else:
        print("[intent] state_dict 完全匹配")

    model.to(device)
    model.eval()
    print(f"[intent] 载入完成: {intent_ckpt}")
    return model



# ============================================================
# 主导出逻辑
# ============================================================

def build_fullnet_from_config(cfg, traj_net, intent_net):
    """
    高层工厂函数：从 config + 已加载好的子网络，构造 FullNet。

    Args:
        cfg: 已经用 yaml.safe_load 读出来的 dict（config_mean_std.yaml）
        traj_net: 轨迹预测 LSTM 网络（已经 load_state_dict 并 .eval()）
        intent_net: 意图 + 威胁度 MLP 网络（已经 load_state_dict 并 .eval()）
    """
    # ---- 1) 读 full_net 配置 ----
    fusion_cfg = cfg.get("full_net", cfg.get("fusion", {}))

    hist_len     = int(fusion_cfg.get("hist_len", 20))
    fut_len      = int(fusion_cfg.get("fut_len", 10))
    feature_dim  = int(fusion_cfg.get("feature_dim", 6))
    window_len_B = int(fusion_cfg.get("window_len_B", 10))   # 意图窗口长度
    n_modes      = int(fusion_cfg.get("n_modes", 3))
    use_delta_A  = bool(fusion_cfg.get("use_delta_A", True))
    use_traj_logits = bool(fusion_cfg.get("use_traj_logits", True))

    # ---- 2) 读 A/B 两套归一化的 npz 路径 ----
    # A：轨迹训练用 scaler_posvel.npz
    # B：意图训练用 scaler_intent_posvel.npz
    normA_cfg = cfg.get("traj_norm", cfg.get("norm_A", {}))
    normB_cfg = cfg.get("intent_norm", cfg.get("norm_B", {}))

    cfg_dir = Path(__file__).resolve().parent  # 默认：config_mean_std.yaml 与本脚本同目录

    scaler_A_path = normA_cfg.get("scaler_npz", "")
    scaler_B_path = normB_cfg.get("scaler_npz", "")

    if scaler_A_path:
        scaler_A_path = (cfg_dir / scaler_A_path).resolve()
    else:
        # 兜底：按你现在的工程目录推一个默认路径
        scaler_A_path = (
            cfg_dir.parent
            / "203_prediction_multi_pytorch_without_map_v0.2"
            / "data"
            / "processed"
            / "scaler_posvel.npz"
        ).resolve()

    if scaler_B_path:
        scaler_B_path = (cfg_dir / scaler_B_path).resolve()
    else:
        scaler_B_path = (
            cfg_dir.parent
            / "203_prediction_intention_pytorch_v0"
            / "data"
            / "processed"
            / "scaler_intent_posvel.npz"
        ).resolve()

    # ---- 3) 加载 npz，取 mean/std ----
    npz_A = np.load(scaler_A_path)
    print(f"[Info] 加载 npz: {scaler_A_path}, keys={set(npz_A.files)}")
    mean_A_np = npz_A["mean"].astype(np.float32)
    std_A_np  = npz_A["std"].astype(np.float32)

    npz_B = np.load(scaler_B_path)
    print(f"[Info] 加载 npz: {scaler_B_path}, keys={set(npz_B.files)}")
    mean_B_np = npz_B["mean"].astype(np.float32)
    std_B_np  = npz_B["std"].astype(np.float32)

    # ---- 4) 转成 torch.Tensor，并放到与轨迹网络相同的 device 上 ----
    device = next(traj_net.parameters()).device

    mean_A = torch.from_numpy(mean_A_np).to(device)
    std_A  = torch.from_numpy(std_A_np).to(device)
    mean_B = torch.from_numpy(mean_B_np).to(device)
    std_B  = torch.from_numpy(std_B_np).to(device)

    # ---- 5) 构造 FullNet ----
    model = FullNet(
        traj_net=traj_net,
        intent_net=intent_net,
        mean_A=mean_A,
        std_A=std_A,
        mean_B=mean_B,
        std_B=std_B,
        hist_len=hist_len,
        fut_len=fut_len,
        feature_dim=feature_dim,
        window_len_B=window_len_B,
        n_modes=n_modes,
        use_delta_A=use_delta_A,
        use_traj_logits=use_traj_logits,
    )

    return model





def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    hist_len: int = 20,
    feature_dim: int = 6,
    opset: int = 18,  # 建议直接用 18
) -> None:
    model.eval()
    device = next(model.parameters()).device

    dummy_input = torch.zeros(1, hist_len, feature_dim, dtype=torch.float32, device=device)

    dynamic_axes = {
        "input": {0: "batch"},
        "output": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False,          # ★★ 关键：关闭新导出器，使用旧 trace 导出器
    )
    print(f"[OK] 导出 ONNX 完成：{onnx_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 FullNet 为 ONNX")
    parser.add_argument(
        "--traj-ckpt",
        type=str,
        required=True,
        help="多模态轨迹网络 ckpt 路径",
    )
    parser.add_argument(
        "--intent-ckpt",
        type=str,
        required=True,
        help="意图/威胁网络 ckpt 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config_mean_std.yaml 路径（包含 norm_A/norm_B/full_net）",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        required=True,
        help="导出的 ONNX 文件路径",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本（默认 13，可按部署端要求调整）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    traj_ckpt = Path(args.traj_ckpt).resolve()
    intent_ckpt = Path(args.intent_ckpt).resolve()
    cfg_path = Path(args.config).resolve()
    onnx_out = Path(args.onnx_out).resolve()

    if not traj_ckpt.is_file():
        raise FileNotFoundError(f"轨迹 ckpt 不存在：{traj_ckpt}")
    if not intent_ckpt.is_file():
        raise FileNotFoundError(f"意图 ckpt 不存在：{intent_ckpt}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")

    device = torch.device("cpu")  # 导出 ONNX 通常用 CPU 即可
    print(f"[Info] 使用设备: {device}")

    print(f"[Info] 读取配置: {cfg_path}")
    cfg = load_yaml_config(cfg_path)

    print(f"[Info] 加载轨迹网络 ckpt: {traj_ckpt}")
    traj_net = load_traj_net(traj_ckpt, device=device)

    print(f"[Info] 加载意图/威胁网络 ckpt: {intent_ckpt}")
    intent_net = load_intent_net(intent_ckpt, device=device)

    print("[Info] 构建 FullNet ...")
    full_net = build_fullnet_from_config(cfg, traj_net, intent_net)
    full_net.to(device)
    full_net.eval()

    # 从 full_net 里取 hist_len / feature_dim（如果没写就用默认）
    full_cfg = cfg.get("full_net", {})
    hist_len = int(full_cfg.get("hist_len", 20))
    feature_dim = int(full_cfg.get("feature_dim", 6))

    print(f"[Info] 导出 ONNX 到: {onnx_out}")
    export_onnx(
        model=full_net,
        onnx_path=onnx_out,
        hist_len=hist_len,
        feature_dim=feature_dim,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
