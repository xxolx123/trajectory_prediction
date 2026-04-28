"""
gnn1/code/train/_verify_manual_attention.py
-------------------------------------------
一次性验证：用同一份 ckpt 加载 GNN1 的 SDPA 版和 manual 版，比对前向输出。

通过条件：max|Δlogits| < 1e-4 且 top_idx 完全一致。

用法（LSTM_traj_predict 环境）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m gnn1.code.train._verify_manual_attention
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnn1.code.train.model import (  # noqa: E402
    build_model_from_config,
)


def _latest_ckpt(d: Path) -> Path:
    cands = sorted(d.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"无 .pt 于 {d}")
    return cands[0]


def main() -> None:
    cfg_path = REPO_ROOT / "gnn1" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 注意：不要在这里强制 dropout=0；_MLP 的 nn.Sequential 层数会随
    # dropout > 0 多一层 Dropout，强改 dropout 会让 state_dict key 错位。
    # eval() 已经让 Dropout 失效，对 forward 数值没影响。
    sdpa_cfg = {**cfg}
    sdpa_model_sec = dict(sdpa_cfg.get("model", {}) or {})
    sdpa_model_sec["manual_attention"] = False
    sdpa_cfg["model"] = sdpa_model_sec

    manual_cfg = {**cfg}
    manual_model_sec = dict(manual_cfg.get("model", {}) or {})
    manual_model_sec["manual_attention"] = True
    manual_cfg["model"] = manual_model_sec

    sdpa = build_model_from_config(sdpa_cfg).eval()
    manual = build_model_from_config(manual_cfg).eval()

    ckpt_dir = REPO_ROOT / "gnn1" / "checkpoints"
    ckpt_path = _latest_ckpt(ckpt_dir)
    print(f"[Verify] ckpt = {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    miss_s, unex_s = sdpa.load_state_dict(sd, strict=False)
    miss_m, unex_m = manual.load_state_dict(sd, strict=False)
    print(f"[Verify] SDPA   load: missing={len(miss_s)} unexpected={len(unex_s)}")
    print(f"[Verify] Manual load: missing={len(miss_m)} unexpected={len(unex_m)}")
    if miss_m:
        print(f"  manual missing keys (head): {miss_m[:5]}")
    if unex_m:
        print(f"  manual unexpected keys (head): {unex_m[:5]}")

    torch.manual_seed(0)
    B = 4
    M = sdpa.cfg.n_modes
    T = sdpa.cfg.fut_len
    D = sdpa.cfg.feat_dim
    batch = {
        "cand_trajs": torch.randn(B, M, T, D),
        "task_type": torch.zeros(B, dtype=torch.long),
        "type": torch.zeros(B, dtype=torch.long),
        "position": torch.randn(B, 3),
    }

    with torch.no_grad():
        out_s = sdpa(batch)
        out_m = manual(batch)

    diff_logits = (out_s["logits"] - out_m["logits"]).abs().max().item()
    diff_probs = (out_s["probs"] - out_m["probs"]).abs().max().item()
    diff_topp = (out_s["top_probs"] - out_m["top_probs"]).abs().max().item()
    same_topidx = bool(torch.equal(out_s["top_idx"], out_m["top_idx"]))

    print(f"[Verify] max|Δlogits|    = {diff_logits:.3e}")
    print(f"[Verify] max|Δprobs|     = {diff_probs:.3e}")
    print(f"[Verify] max|Δtop_probs| = {diff_topp:.3e}")
    print(f"[Verify] top_idx 一致     = {same_topidx}")

    tol = 1e-4
    ok = (diff_logits < tol) and (diff_probs < tol) and same_topidx
    print(f"[Verify] {'OK' if ok else 'FAIL'}  (tol={tol})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
