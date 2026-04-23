"""
lstm1/code/train/model.py
-------------------------
LSTM1：多模态轨迹预测。

与 old_plan 的 LSTMForecaster 相比的唯一差异：
    **不再输出 mode_logits**（轨迹概率由下游 GNN1 计算）。

输入：
    x: [B, in_len, input_size]  已做 "delta+归一化" 的 6 维特征
输出：
    pred_trajs: [B, M, out_len, output_size]  M 条候选未来轨迹（归一化空间）

网络结构：
    LSTM 编码器 -> 取最后一层 hidden
    -> Linear: hidden -> (M * out_len * output_size)
    (没有 fc_logit / mode_logits 头)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class LSTMConfig:
    input_size: int = 6
    output_size: int = 6
    hidden_size: int = 256
    num_layers: int = 2
    out_len: int = 10
    dropout: float = 0.0
    bidirectional: bool = False
    batch_first: bool = True
    modes: int = 3


class LSTMForecaster(nn.Module):
    """
    [B, in_len, input_size] -> [B, M, out_len, output_size]
    （不再输出 mode_logits，概率由下游 GNN1 计算）
    """

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.output_size = cfg.output_size
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.out_len = cfg.out_len
        self.bidirectional = cfg.bidirectional
        self.modes = cfg.modes

        self.num_directions = 2 if cfg.bidirectional else 1
        fc_in_dim = self.hidden_size * self.num_directions
        lstm_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=cfg.batch_first,
            bidirectional=cfg.bidirectional,
            dropout=lstm_dropout,
        )
        # 只保留轨迹头
        self.fc_traj = nn.Linear(
            in_features=fc_in_dim,
            out_features=self.modes * self.out_len * self.output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_len, input_size]
        Returns:
            pred_trajs: [B, M, out_len, output_size]
        """
        output, (h_n, c_n) = self.lstm(x)
        B = x.size(0)

        h_n = h_n.view(self.num_layers, self.num_directions, B, self.hidden_size)
        last_layer_h = h_n[-1]  # [num_directions, B, hidden_size]
        last_h = last_layer_h.transpose(0, 1).contiguous().view(
            B, self.num_directions * self.hidden_size
        )

        traj_flat = self.fc_traj(last_h)
        pred_trajs = traj_flat.view(B, self.modes, self.out_len, self.output_size)
        return pred_trajs


def build_model_from_config(cfg: Dict[str, Any]) -> LSTMForecaster:
    """
    从顶层 config（已 yaml.safe_load）中读取 model 段构建 LSTMForecaster。
    期望 config:
        model:
          type: "lstm"
          input_size: 6
          output_size: 6
          hidden_size: 256
          num_layers: 2
          out_len: 10        # 若未指定，默认用 data.out_len
          dropout: 0.0
          bidirectional: false
          modes: 3
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_type = model_cfg.get("type", "lstm").lower()
    if model_type != "lstm":
        raise ValueError(f"lstm1/model.py 只实现 LSTM，收到 type={model_type}")

    lstm_cfg = LSTMConfig(
        input_size=int(model_cfg.get("input_size", 6)),
        output_size=int(model_cfg.get("output_size", 6)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        out_len=int(model_cfg.get("out_len", data_cfg.get("out_len", 10))),
        dropout=float(model_cfg.get("dropout", 0.0)),
        bidirectional=bool(model_cfg.get("bidirectional", False)),
        batch_first=True,
        modes=int(model_cfg.get("modes", 3)),
    )
    return LSTMForecaster(lstm_cfg)
