"""
gnn2/code/train/_verify_manual_attention.py
-------------------------------------------
一次性验证脚本（非生产代码）：

  1) 用同一份 ckpt 同时加载 SDPA 版 StrikeZoneNet 和手写版 StrikeZoneNetManual
  2) 在相同输入下做 forward
  3) 检查 strike_pos / strike_radius / strike_conf 的最大差 ≤ 1e-4

通过即说明 manual 版与 SDPA 版数学等价、ckpt 兼容；可以放心用于
fusion 端 opset ≤ 13 的 ONNX 导出（mindspore-lite 1.8.1 默认 opset 11）。

用法（在 LSTM_traj_predict 环境内）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m gnn2.code.train._verify_manual_attention
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnn2.code.train.model import (  # noqa: E402
    StrikeZoneNet,
    StrikeZoneNetManual,
)


def _latest_ckpt(d: Path) -> Path:
    cands = sorted(d.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"无 .pt 于 {d}")
    return cands[0]


def main() -> None:
    cfg_path = REPO_ROOT / "gnn2" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {}) or {}
    data = cfg.get("data", {}) or {}
    ctx = cfg.get("context", {}) or {}

    kwargs = dict(
        fut_len=int(m.get("fut_len", 10)),
        feat_dim=int(m.get("feat_dim", 6)),
        d_emb=int(m.get("d_emb", 64)),
        nhead=int(m.get("nhead", 4)),
        num_layers=int(m.get("num_layers", 2)),
        ffn_dim=int(m.get("ffn_dim", 128)),
        dropout=0.0,                                              # 关 dropout 保证可比
        max_seq_len=int(m.get("max_seq_len", 16)),
        eta_scale_seconds=float(ctx.get("eta_scale_seconds", 3600.0)),
        radius_min_km=float(data.get("radius_min_km", 0.5)),
        radius_max_km=float(data.get("radius_max_km", 10.0)),
    )

    ckpt_dir = REPO_ROOT / "gnn2" / "checkpoints"
    use_random = False
    if ckpt_dir.exists():
        try:
            ckpt_path = _latest_ckpt(ckpt_dir)
            print(f"[Verify] ckpt = {ckpt_path}")
            sd_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print(f"[Verify] ckpt 目录 {ckpt_dir} 下无 .pt，使用随机权重对比")
            sd_obj = None
            use_random = True
    else:
        print(f"[Verify] ckpt 目录 {ckpt_dir} 不存在，使用随机权重对比")
        sd_obj = None
        use_random = True

    if isinstance(sd_obj, dict):
        if "state_dict" in sd_obj:
            sd_obj = sd_obj["state_dict"]
        elif "model" in sd_obj:
            sd_obj = sd_obj["model"]

    sdpa = StrikeZoneNet(**kwargs).eval()
    manual = StrikeZoneNetManual(**kwargs).eval()

    if use_random:
        # 随机权重场景：把 sdpa 的 state_dict 喂给 manual，确保两边权重一致
        sd = sdpa.state_dict()
    else:
        sd = sd_obj  # type: ignore[assignment]

    miss_s, unex_s = sdpa.load_state_dict(sd, strict=False)
    miss_m, unex_m = manual.load_state_dict(sd, strict=False)
    print(f"[Verify] SDPA   load: missing={len(miss_s)} unexpected={len(unex_s)}")
    print(f"[Verify] Manual load: missing={len(miss_m)} unexpected={len(unex_m)}")
    if miss_m:
        print(f"  manual missing keys (head 5): {miss_m[:5]}")
    if unex_m:
        print(f"  manual unexpected keys (head 5): {unex_m[:5]}")

    torch.manual_seed(0)
    B, T = 4, int(m.get("fut_len", 10))
    pred = torch.randn(B, T, int(m.get("feat_dim", 6)))
    eta = torch.tensor([60, 600, 1800, 3600], dtype=torch.long)

    with torch.no_grad():
        out_s = sdpa(pred, eta)
        out_m = manual(pred, eta)

    diff_pos = (out_s["strike_pos"] - out_m["strike_pos"]).abs().max().item()
    diff_radius = (out_s["strike_radius"] - out_m["strike_radius"]).abs().max().item()
    diff_conf = (out_s["strike_conf"] - out_m["strike_conf"]).abs().max().item()

    print(f"[Verify] max|Δ strike_pos|    = {diff_pos:.3e}")
    print(f"[Verify] max|Δ strike_radius| = {diff_radius:.3e}")
    print(f"[Verify] max|Δ strike_conf|   = {diff_conf:.3e}")

    tol = 1e-4
    ok = diff_pos < tol and diff_radius < tol and diff_conf < tol
    print(f"[Verify] {'OK' if ok else 'FAIL'}  (tol={tol})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
