"""
gnn1/code/train/trainer.py
--------------------------
GNN1 训练主入口（骨架 + --smoke）。

正式训练（train()）目前抛 NotImplementedError，因为数据生成还未实现。
--smoke 模式用全零 ctx + 随机候选轨迹，验证 TrajSelectionGNN 的 forward/backward。

用法：
    cd new_plan/gnn1
    export PYTHONPATH="$PWD/code:$(cd ..; pwd)"   # 让 common. 能被 import
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD/code;$(Resolve-Path ..).Path"

    python -m train.trainer --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

THIS_DIR = Path(__file__).resolve().parent            # .../gnn1/code/train
CODE_DIR = THIS_DIR.parent                            # .../gnn1/code
GNN1_ROOT = CODE_DIR.parent                           # .../gnn1
REPO_ROOT = GNN1_ROOT.parent                          # .../new_plan
for p in (CODE_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.context_schema import build_ctx_dims_from_config, build_dummy_context  # noqa: E402
from train.model import build_model_from_config         # noqa: E402
from train.loss import Gnn1Loss, Gnn1LossConfig         # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_smoke(cfg: Dict[str, Any]) -> None:
    print("=" * 60)
    print("[Smoke/GNN1] start")
    print("=" * 60)
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/GNN1] params = {sum(p.numel() for p in model.parameters()):,}")

    model_cfg = cfg["model"]
    B = 2
    M = int(model_cfg["n_modes"])
    T = int(model_cfg["fut_len"])
    D = int(model_cfg["feat_dim"])

    torch.manual_seed(0)
    cand = torch.randn(B, M, T, D, device=device)
    ctx_dims = build_ctx_dims_from_config(cfg)
    ctx = build_dummy_context(B, device=device, ctx_dims=ctx_dims)

    out = model(cand, ctx)
    assert out["traj_logits"].shape == (B, M), out["traj_logits"].shape
    assert out["traj_probs"].shape == (B, M), out["traj_probs"].shape
    print(f"[Smoke/GNN1] forward OK: logits/probs shape = {tuple(out['traj_logits'].shape)}")

    loss_fn = Gnn1Loss(Gnn1LossConfig(use_soft_label=False))
    label = torch.zeros(B, dtype=torch.long, device=device)  # 假装正类全是 0
    loss = loss_fn(out["traj_logits"], label)
    print(f"[Smoke/GNN1] loss = {float(loss):.4f}")
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    print(f"[Smoke/GNN1] gradients present? {has_grad}")
    print("=" * 60)
    print("[Smoke/GNN1] OK")
    print("=" * 60)


def train(config_path: str) -> None:
    # TODO: 等 data/generate_data.py + data/dataset.py 实现之后，
    #       这里照 LSTM1 的 trainer.py 模式补齐即可。
    raise NotImplementedError(
        "GNN1 正式训练尚未实现。请先完成 data/generate_data.py 与 data/dataset.py。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN1 trainer skeleton.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = GNN1_ROOT / cfg_path
    cfg = load_config(cfg_path)

    if args.smoke:
        run_smoke(cfg)
        return
    train(str(cfg_path))


if __name__ == "__main__":
    main()
