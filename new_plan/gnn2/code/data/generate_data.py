"""
gnn2/code/data/generate_data.py
-------------------------------
TODO: 等接口 + 打击区域 GT 标签定型后实现。

用法（在 new_plan/gnn2/ 下，目前仅占位，跑起来会抛 NotImplementedError）：
    # Windows PowerShell
    $env:PYTHONPATH = "$PWD/code"
    # Linux/macOS
    # export PYTHONPATH="$PWD/code"

    python -m data.generate_data --config config.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for GNN2.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    _ = args

    raise NotImplementedError("GNN2 数据生成尚未实现。")


if __name__ == "__main__":
    main()
