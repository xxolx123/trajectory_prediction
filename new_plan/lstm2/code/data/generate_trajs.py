"""
lstm2/code/data/generate_trajs.py
---------------------------------
生成 LSTM2 训练数据（骨架占位）。

TODO:
    - old_plan/203_prediction_intention_pytorch_v0/code/data/generate_trajs.py
      基本可以直接复用（输出带 intent_label / threat_score 的 CSV）。
    - 后续可能还要叠加"用 LSTM1+GNN1+约束优化 离线产出的精修未来"，
      方案未定。详见 README.txt。
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for LSTM2.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    _ = args

    raise NotImplementedError(
        "LSTM2 数据生成脚本还未实现。"
        "可以从 old_plan/203_prediction_intention_pytorch_v0/code/data/generate_trajs.py 改写。"
    )


if __name__ == "__main__":
    main()
