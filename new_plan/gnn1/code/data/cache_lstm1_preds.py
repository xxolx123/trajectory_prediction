#!/usr/bin/env python3
"""
cache_lstm1_preds.py
--------------------
对冻结的 LSTM1 跑一遍 train/val/test 的所有 window，把 (history, candidates)
缓存成 data/cache/{split}.npz。

candidates 保存在"归一化 + delta"空间（和 LSTM1 训练对齐），后续用 LSTM1
的 scaler 解码到 xy 进行 generate_data.py 的 position 生成。

用法（在 new_plan/gnn1/ 下）：
    # 环境变量 PYTHONPATH 指向 gnn1/code + lstm1/code
    $env:PYTHONPATH = "$PWD/code;$PWD/../lstm1/code"
    python -m data.cache_lstm1_preds --config config.yaml
    # 也可以显式指定 lstm1 ckpt：
    python -m data.cache_lstm1_preds --config config.yaml \
        --lstm1-ckpt ..\lstm1\checkpoints\<run_id>\best_lstm_epoch010_valloss0.0171.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# =============== 让我们能 import lstm1 的代码 ===============

def _ensure_lstm1_importable(lstm1_root: Path) -> None:
    """把 lstm1/code 加到 sys.path，以便可以 `from data.traj_dataset import ...`
    和 `from train.model import ...`。"""
    lstm1_code = (lstm1_root / "code").resolve()
    if not lstm1_code.exists():
        raise FileNotFoundError(
            f"找不到 lstm1 代码目录：{lstm1_code}。"
            "请检查 config.yaml 里 lstm1.config_path 是否指向正确的 lstm1/config.yaml。"
        )
    s = str(lstm1_code)
    if s not in sys.path:
        sys.path.insert(0, s)


# =============== 工具：查找最新 ckpt ===============

def _find_latest_ckpt(ckpt_root: Path) -> Optional[Path]:
    """在 ckpt_root 下递归查找 .pt，按修改时间取最新。
    支持 ckpt_root 是目录或直接是一个 .pt 文件。"""
    if ckpt_root.is_file():
        return ckpt_root
    if not ckpt_root.exists():
        return None
    candidates = list(ckpt_root.rglob("*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =============== 主流程 ===============

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache LSTM1 candidates for GNN1.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="GNN1 config.yaml 路径")
    parser.add_argument("--lstm1-ckpt", type=str, default="",
                        help="覆盖 config 里的 lstm1.ckpt_path；可以是 .pt 文件或目录")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto / cuda / cpu / mps")
    args = parser.parse_args()

    # --- GNN1 项目根：本文件位于 gnn1/code/data/cache_lstm1_preds.py ---
    gnn1_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = gnn1_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"config 不存在：{cfg_path}")

    cfg = load_config(cfg_path)

    # --- 定位 lstm1 项目根（通过 lstm1.config_path 推出） ---
    lstm1_cfg_rel = Path(cfg.get("lstm1", {}).get("config_path", "../lstm1/config.yaml"))
    lstm1_cfg_abs = (gnn1_root / lstm1_cfg_rel).resolve() if not lstm1_cfg_rel.is_absolute() else lstm1_cfg_rel
    if not lstm1_cfg_abs.exists():
        raise FileNotFoundError(f"找不到 lstm1 config：{lstm1_cfg_abs}")
    lstm1_root = lstm1_cfg_abs.parent
    _ensure_lstm1_importable(lstm1_root)

    # 延迟 import：必须在 sys.path 补好之后
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    from data.traj_dataset import build_datasets_from_config  # type: ignore  # noqa: E402
    from train.model import build_model_from_config  # type: ignore  # noqa: E402

    # --- 设备 ---
    dev_str = args.device.lower()
    if dev_str == "cpu":
        device = torch.device("cpu")
    elif dev_str in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev_str == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"[cache] device = {device}")

    # --- 构建 lstm1 数据集（得到 scaler 空间下的 window） ---
    # lstm1 的 build_datasets_from_config 接受相对于 lstm1_root 的路径
    try:
        rel_lstm1_cfg = lstm1_cfg_abs.relative_to(lstm1_root)
    except ValueError:
        rel_lstm1_cfg = lstm1_cfg_abs
    print(f"[cache] building lstm1 datasets via {lstm1_cfg_abs} ...")

    # 注意：lstm1 的 build_datasets_from_config 内部会用 Path(__file__).parents[2] 作为
    # 项目根，即 lstm1_root。我们只需传入相对路径即可。
    prev_cwd = os.getcwd()
    try:
        # 切到 lstm1 根目录，避免相对路径错配
        os.chdir(lstm1_root)
        train_ds, val_ds, test_ds, scaler = build_datasets_from_config(
            config_path=str(rel_lstm1_cfg)
        )
    finally:
        os.chdir(prev_cwd)

    if scaler is None:
        raise RuntimeError("lstm1 dataset 没有返回 scaler；请确保 lstm1 的 data.normalize=true")

    print(f"[cache] dataset sizes: "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # --- 构建并加载 lstm1 模型 ---
    # 读 lstm1 config 用来 build_model
    lstm1_cfg = load_config(lstm1_cfg_abs)
    model = build_model_from_config(lstm1_cfg).to(device)

    # 选 ckpt
    ckpt_override = args.lstm1_ckpt or cfg.get("lstm1", {}).get("ckpt_path", "")
    if ckpt_override:
        ckpt_p = Path(ckpt_override)
        if not ckpt_p.is_absolute():
            ckpt_p = (gnn1_root / ckpt_p).resolve()
    else:
        ckpt_p = (lstm1_root / "checkpoints").resolve()
    ckpt_path = _find_latest_ckpt(ckpt_p)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"没找到 LSTM1 ckpt。查找路径：{ckpt_p}。"
            "请在 config.yaml 里填 lstm1.ckpt_path，或传 --lstm1-ckpt。"
        )
    print(f"[cache] loading LSTM1 ckpt: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 输出目录 ---
    cache_dir_rel = cfg.get("data", {}).get("cache_dir", "data/cache")
    cache_dir = (gnn1_root / cache_dir_rel).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 同时把 lstm1 的 scaler 落到 gnn1 这边一份，方便 generate_data 读 ---
    scaler_out = cache_dir / "scaler_posvel.npz"
    scaler.save(scaler_out)
    print(f"[cache] scaler saved: {scaler_out}")

    # --- 对每个 split 做 forward 并保存 ---
    splits: List[Tuple[str, Any]] = [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]
    for name, ds in splits:
        if len(ds) == 0:
            print(f"[cache] split {name} 为空，跳过")
            continue
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        history_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        candidates_list: List[np.ndarray] = []
        n_done = 0
        import time as _time
        t0 = _time.time()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs_t = inputs.to(device).float()
                cand = model(inputs_t)                          # [B, M, T, D]
                history_list.append(inputs.cpu().numpy().astype(np.float32))
                targets_list.append(targets.cpu().numpy().astype(np.float32))
                candidates_list.append(cand.cpu().numpy().astype(np.float32))
                n_done += inputs.size(0)
        history = np.concatenate(history_list, axis=0)         # [N, 20, 6]
        targets = np.concatenate(targets_list, axis=0)         # [N, 10, 6]  GT future
        candidates = np.concatenate(candidates_list, axis=0)   # [N, 5, 10, 6]

        out_path = cache_dir / f"{name}.npz"
        np.savez(out_path,
                 history=history, targets=targets, candidates=candidates)
        elapsed = _time.time() - t0
        print(f"[cache] {name}: history {history.shape} + targets {targets.shape} + "
              f"candidates {candidates.shape} → {out_path} ({elapsed:.1f}s)")

    print("[cache] 完成。")


if __name__ == "__main__":
    main()
