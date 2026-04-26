"""
fusion/code/verify_round_trip.py
--------------------------------
诊断小工具：验证 fusion 的"cache.history → 物理空间 → fusion.LSTM1 → 归一化+delta 候选"
往返之后，能否复现 gnn1 cache 里离线存的 cache.candidates。

如果完全一致，说明：
  - 反归一化 / cumsum / hist 末帧归零 / 重新算 delta / 重新归一化 这条往返链是无损的
  - fusion 内部 LSTM1 的前向调用方式和 cache_lstm1_preds.py 完全一致
  - 此时若预测结果"看起来不合理"，那就是 LSTM1 模型本身在该样本上的表现，
    不是 fusion 流水线的错位

用法::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m fusion.code.verify_round_trip --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.code.build import load_fusion_config, _resolve_rel  # noqa: E402
from fusion.code.full_net_v2 import build_full_net_from_fusion_config  # noqa: E402
from fusion.code.test_pipeline import _Scaler, _decode_window_to_phys  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="fusion round-trip 验证")
    parser.add_argument("--fusion-config", type=str,
                        default=str(REPO_ROOT / "fusion" / "config.yaml"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n", type=int, default=8, help="抽几个 cache 窗口比对")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Verify] device = {device}")

    fusion_cfg, fusion_cfg_dir = load_fusion_config(Path(args.fusion_config))
    model = build_full_net_from_fusion_config(Path(args.fusion_config)).to(device).eval()
    print(f"[Verify] enable_flags = {model.enable_flags}")

    gnn1_root = _resolve_rel(fusion_cfg.get("gnn1", {}).get("config", ""), fusion_cfg_dir).parent
    cache_path = gnn1_root / "data" / "cache" / f"{args.split}.npz"
    scaler_path = gnn1_root / "data" / "cache" / "scaler_posvel.npz"
    print(f"[Verify] cache  = {cache_path}")
    print(f"[Verify] scaler = {scaler_path}")

    cache = np.load(cache_path)
    scaler = _Scaler.load(scaler_path)
    history = cache["history"]            # [Nw, 20, 6] 归一化 + delta
    candidates = cache["candidates"]      # [Nw, M, T, 6] 归一化 + delta（LSTM1 离线产物）
    Nw = int(history.shape[0])
    print(f"[Verify] cache windows = {Nw}, candidates shape = {candidates.shape}")

    n_want = min(int(args.n), Nw)
    rng = np.random.default_rng(42)
    chosen: List[int] = sorted(rng.choice(Nw, size=n_want, replace=False).tolist())
    print(f"[Verify] sampled cache windows = {chosen}")

    # 1) 走 fusion 的"反解 → cumsum → hist 末帧归零"得到物理空间 hist
    hist_phys_list = []
    for cw in chosen:
        hp, _ = _decode_window_to_phys(history[cw], None, scaler)
        hist_phys_list.append(hp)
    hist_phys = np.stack(hist_phys_list, axis=0)
    hist_t = torch.from_numpy(hist_phys).to(device).float()

    # 2) 走 fusion 内部的 _build_A_input + lstm1 拿"归一化 + delta"候选
    with torch.no_grad():
        x_A = model._build_A_input(hist_t)
        fut_A_fusion = model.lstm1(x_A).cpu().numpy()              # [B, M, T, 6]

    # 3) 拿 cache.candidates 同窗口的离线产物
    fut_A_cache = np.stack([candidates[cw] for cw in chosen], axis=0)

    # 4) 数值对比
    if fut_A_fusion.shape != fut_A_cache.shape:
        raise RuntimeError(
            f"形状不一致 fusion={fut_A_fusion.shape} cache={fut_A_cache.shape}"
        )

    diff = np.abs(fut_A_fusion - fut_A_cache)
    print()
    print("=" * 60)
    print(f"[Verify] LSTM1 输出对比（归一化+delta 空间）")
    print(f"  shape       = {fut_A_fusion.shape}")
    print(f"  max  |diff| = {diff.max():.3e}")
    print(f"  mean |diff| = {diff.mean():.3e}")
    print(f"  fusion vs cache 相对误差中位数 = "
          f"{np.median(diff / (np.abs(fut_A_cache) + 1e-9)):.3e}")
    if diff.max() < 1e-4:
        print("  → 完全一致（差异 < 1e-4），fusion 流水线复现 cache 无误。")
        print("    ※ '预测看起来不合理' 不是 fusion 的 bug，而是 LSTM1 本身在")
        print("      这些样本上的输出。可以用更深训练 / 更多模态 / 更长历史等方式优化。")
    elif diff.max() < 1e-2:
        print("  → 接近一致，差异在 fp32 噪声量级以内（cumsum 误差累积）。")
    else:
        print("  → 明显不一致！请排查 _decode_window_to_phys / _build_A_input 链路。")
    print("=" * 60)

    # 5) 进一步：把 fusion 候选和 cache 候选都解到物理 km，画两组终点距离
    #     检查"反归一化+cumsum"再解出来的物理空间是不是一致
    def _to_phys_xy(feat_norm: np.ndarray) -> np.ndarray:
        orig = scaler.inverse_transform(feat_norm.astype(np.float64))
        return np.cumsum(orig[..., :2], axis=-2)

    end_fusion = _to_phys_xy(fut_A_fusion)[:, :, -1, :]              # [B, M, 2]
    end_cache = _to_phys_xy(fut_A_cache)[:, :, -1, :]
    end_diff = np.linalg.norm(end_fusion - end_cache, axis=-1)        # [B, M]
    print(f"[Verify] 候选终点 km 距离差: max = {end_diff.max():.3e}, "
          f"mean = {end_diff.mean():.3e}")


if __name__ == "__main__":
    main()
