"""
lstm2/code/train/trainer.py
---------------------------
LSTM2 训练主入口（骨架 + --smoke）。

正式 train() 当前抛 NotImplementedError，等 data/generate_trajs.py +
data/dataset.py 实现后再补齐。

用法：
    cd new_plan/lstm2
    # Linux/macOS:
    export PYTHONPATH="$PWD/code"
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD/code"

    python -m train.trainer --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
LSTM2_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train.model import build_model_from_config      # noqa: E402
from train.loss import build_loss_from_config        # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_smoke(cfg: Dict[str, Any]) -> None:
    print("=" * 60)
    print("[Smoke/LSTM2] start")
    print("=" * 60)
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/LSTM2] params = {sum(p.numel() for p in model.parameters()):,}")

    data_cfg = cfg.get("data", {})
    Tin = int(data_cfg.get("hist_len", 20))
    Tout = int(data_cfg.get("fut_len", 10))
    D = int(cfg.get("model", {}).get("input_size", 6))
    n_cls = int(cfg.get("model", {}).get("num_intent_classes", 4))

    B = 2
    torch.manual_seed(0)
    hist = torch.randn(B, Tin, D, device=device)
    fut = torch.randn(B, Tout, D, device=device)

    out = model(hist, fut)
    assert out["logits_intent"].shape == (B, n_cls), out["logits_intent"].shape
    assert out["threat_raw"].shape == (B, 1), out["threat_raw"].shape
    print(
        f"[Smoke/LSTM2] forward OK: "
        f"logits_intent {tuple(out['logits_intent'].shape)}, "
        f"threat_raw {tuple(out['threat_raw'].shape)}"
    )

    loss_fn = build_loss_from_config(cfg)
    intent_lbl = torch.zeros(B, dtype=torch.long, device=device)
    threat = torch.full((B,), 50.0, dtype=torch.float32, device=device)
    total, cls, reg = loss_fn(out, intent_lbl, threat)
    print(f"[Smoke/LSTM2] loss total={float(total):.4f}, cls={float(cls):.4f}, reg={float(reg):.4f}")
    total.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    print(f"[Smoke/LSTM2] gradients present? {has_grad}")
    print("=" * 60)
    print("[Smoke/LSTM2] OK")
    print("=" * 60)


def train(config_path: str) -> None:
    # TODO: 补齐，参考 lstm1/code/train/trainer.py 的写法
    raise NotImplementedError(
        "LSTM2 正式训练尚未实现：请先完成 data/generate_trajs.py + data/dataset.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM2 trainer skeleton.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = LSTM2_ROOT / cfg_path
    cfg = load_config(cfg_path)

    if args.smoke:
        run_smoke(cfg)
        return
    train(str(cfg_path))


if __name__ == "__main__":
    main()
