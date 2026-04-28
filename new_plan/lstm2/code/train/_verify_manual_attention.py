"""
lstm2/code/train/_verify_manual_attention.py
--------------------------------------------
一次性验证脚本（非生产代码）：

  1) 用同一份 ckpt 同时加载 SDPA 版 IntentTransformer 和手写版
     IntentTransformerManual
  2) 在相同输入下做 forward
  3) 检查 logits / threat 的最大差 ≤ 1e-4

通过即说明 manual 版与 SDPA 版数学等价、ckpt 兼容；可以放心用于
fusion 端 opset ≤ 13 的 ONNX 导出。

用法（在 LSTM_traj_predict 环境内）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m lstm2.code.train._verify_manual_attention
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lstm2.code.train.model import (  # noqa: E402
    IntentTransformer,
    IntentTransformerManual,
)


def _latest_ckpt(d: Path) -> Path:
    cands = sorted(d.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"无 .pt 于 {d}")
    return cands[0]


def main() -> None:
    cfg_path = REPO_ROOT / "lstm2" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    kwargs = dict(
        input_size=int(m.get("input_size", 11)),
        d_model=int(m.get("d_model", 128)),
        nhead=int(m.get("nhead", 4)),
        num_layers=int(m.get("num_layers", 2)),
        ffn_dim=int(m.get("ffn_dim", 256)),
        dropout=0.0,                                            # 关 dropout 保证可比
        num_intent_classes=int(m.get("num_intent_classes", 4)),
        max_seq_len=int(m.get("max_seq_len", 32)),
    )

    ckpt_dir = REPO_ROOT / "lstm2" / "checkpoints"
    ckpt_path = _latest_ckpt(ckpt_dir)
    print(f"[Verify] ckpt = {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    sdpa = IntentTransformer(**kwargs).eval()
    manual = IntentTransformerManual(**kwargs).eval()

    miss_s, unex_s = sdpa.load_state_dict(sd, strict=False)
    miss_m, unex_m = manual.load_state_dict(sd, strict=False)
    print(f"[Verify] SDPA   load: missing={len(miss_s)} unexpected={len(unex_s)}")
    print(f"[Verify] Manual load: missing={len(miss_m)} unexpected={len(unex_m)}")
    if miss_m:
        print(f"  manual missing keys (head): {miss_m[:5]}")
    if unex_m:
        print(f"  manual unexpected keys (head): {unex_m[:5]}")

    torch.manual_seed(0)
    B, T = 4, 10
    fut = torch.randn(B, T, 6)
    pos = torch.randn(B, 3)

    with torch.no_grad():
        out_s = sdpa(fut, pos)
        out_m = manual(fut, pos)

    diff_logits = (out_s["logits_intent"] - out_m["logits_intent"]).abs().max().item()
    diff_threat = (out_s["threat_raw"]    - out_m["threat_raw"]   ).abs().max().item()
    argmax_eq = bool(
        torch.equal(
            out_s["logits_intent"].argmax(-1),
            out_m["logits_intent"].argmax(-1),
        )
    )

    print(f"[Verify] max|Δlogits|  = {diff_logits:.3e}")
    print(f"[Verify] max|Δthreat|  = {diff_threat:.3e}")
    print(f"[Verify] argmax intent equal = {argmax_eq}")

    tol = 1e-4
    ok = (diff_logits < tol) and (diff_threat < tol) and argmax_eq
    print(f"[Verify] {'OK' if ok else 'FAIL'}  (tol={tol})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
