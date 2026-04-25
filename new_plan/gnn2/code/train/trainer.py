"""
gnn2/code/train/trainer.py
--------------------------
GNN2 训练主入口（骨架 + --smoke）。
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
GNN2_ROOT = CODE_DIR.parent
REPO_ROOT = GNN2_ROOT.parent
for p in (CODE_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train.model import build_model_from_config                                    # noqa: E402
from train.loss import StrikeLoss                                                  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_smoke(cfg: Dict[str, Any]) -> None:
    print("=" * 60)
    print("[Smoke/GNN2] start")
    print("=" * 60)
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    print(f"[Smoke/GNN2] params = {sum(p.numel() for p in model.parameters()):,}")

    m = cfg["model"]
    B = 2
    T = int(m["fut_len"])
    D = int(m["feat_dim"])

    torch.manual_seed(0)
    pred = torch.randn(B, T, D, device=device)
    eta = torch.tensor([600, 1800], dtype=torch.long, device=device)  # 10 / 30 分钟示例

    out = model(pred, eta)
    assert out["strike_pos"].shape == (B, 3), out["strike_pos"].shape
    assert out["strike_radius"].shape == (B, 1), out["strike_radius"].shape
    assert out["strike_conf"].shape == (B, 1), out["strike_conf"].shape
    # 单位 / 值域校验
    assert torch.all(out["strike_radius"] >= 0), "strike_radius 必须 >= 0"
    assert torch.all((out["strike_conf"] >= 0) & (out["strike_conf"] <= 1)), \
        "strike_conf 必须在 [0, 1]"
    print(
        f"[Smoke/GNN2] forward OK: "
        f"pos {tuple(out['strike_pos'].shape)}, "
        f"radius {tuple(out['strike_radius'].shape)}, "
        f"conf {tuple(out['strike_conf'].shape)}"
    )

    # 当前 loss 永远为 0，无梯度回传也是正常的
    loss_fn = StrikeLoss()
    loss = loss_fn(out, gt_strike=None)
    print(f"[Smoke/GNN2] loss = {float(loss):.4f} (loss 目前是占位 0，属预期)")
    print("=" * 60)
    print("[Smoke/GNN2] OK")
    print("=" * 60)


def train(config_path: str) -> None:
    raise NotImplementedError(
        "GNN2 正式训练尚未实现；需要先有打击区域 GT。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN2 trainer skeleton.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = GNN2_ROOT / cfg_path
    cfg = load_config(cfg_path)

    if args.smoke:
        run_smoke(cfg)
        return
    train(str(cfg_path))


if __name__ == "__main__":
    main()
