"""
gnn1/code/data/generate_data.py
-------------------------------
生成 GNN1 的训练样本：
  每条样本 = (cand_trajs [M,T,6], context 各字段, label)

TODO: 整个函数体都是占位，需要等以下两件事确定：
    1) 用哪个 LSTM1 ckpt 产生候选轨迹（静态产出？还是在线产出？）
    2) target_task / fixed_targets / target_type / road_network 的接口格式

可能的实现思路见 README.txt。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for GNN1.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    _ = args

    # TODO: 实现
    raise NotImplementedError(
        "GNN1 数据生成脚本还未实现。"
        "请参考 gnn1/README.txt 的 TODO 清单后补齐。"
    )


if __name__ == "__main__":
    main()
