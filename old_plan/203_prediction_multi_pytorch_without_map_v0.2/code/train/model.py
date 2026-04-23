"""
model.py (PyTorch version)
--------------------------
轨迹预测模型主体（LSTM + 多模态 MTP 版本，硬分配用）。

输入：
  x: Tensor, shape = [batch_size, in_len, input_size]
     一般 input_size = 6，对应 [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]

输出：
  pred_trajs:  Tensor, shape = [batch_size, modes, out_len, output_size]
               多模态候选未来轨迹
  mode_logits: Tensor, shape = [batch_size, modes]
               每个 mode 的概率 logit（后续在 loss 中做 softmax / CE）

网络结构：
  - LSTM 编码器：读取历史 in_len 步
  - 使用最后一层、所有方向（单向/双向）的 hidden state 拼成一个向量
  - 两个全连接头：
      * 轨迹头：hidden → (modes * out_len * output_size)
      * logit 头：hidden → modes
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class LSTMConfig:
    """从 config.yaml 里的 model 段解析出来的配置"""
    input_size: int = 6       # 输入特征维度
    output_size: int = 6      # 输出特征维度（通常与输入一致）
    hidden_size: int = 256    # LSTM 隐藏单元数
    num_layers: int = 2       # LSTM 堆叠层数
    out_len: int = 10         # 需要预测的未来步数
    dropout: float = 0.0      # LSTM 内部的 dropout
    bidirectional: bool = False   # 是否双向 LSTM
    batch_first: bool = True      # 固定为 True：输入 [B, T, C]
    modes: int = 3            # 多模态数量 M（候选轨迹条数）


class LSTMForecaster(nn.Module):
    """
    LSTM 多模态序列预测网络（用于 MTP 硬分配）：
      - 接收 [B, in_len, input_size]
      - 输出：
          * pred_trajs:  [B, M, out_len, output_size]
          * mode_logits: [B, M]
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

        # PyTorch 的 LSTM，同样支持 batch_first=True，输入 [B, T, C]
        # 注意：PyTorch 里 dropout 只有在 num_layers > 1 时才生效
        lstm_dropout = cfg.dropout if cfg.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=cfg.batch_first,
            bidirectional=cfg.bidirectional,
            dropout=lstm_dropout,
        )

        # 轨迹头：hidden → modes * out_len * output_size
        self.fc_traj = nn.Linear(
            in_features=fc_in_dim,
            out_features=self.modes * self.out_len * self.output_size,
        )

        # 概率头：hidden → modes
        self.fc_logit = nn.Linear(
            in_features=fc_in_dim,
            out_features=self.modes,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, in_len, input_size]，dtype=float32

        return:
          pred_trajs:  [B, M, out_len, output_size]
          mode_logits: [B, M]
        """
        # LSTM 输出：
        #   output: [B, in_len, num_directions * hidden_size]
        #   h_n: [num_layers * num_directions, B, hidden_size]
        #   c_n: 同形状，这里用不到
        output, (h_n, c_n) = self.lstm(x)

        B = x.size(0)

        # 取“最后一层”的 hidden，并拼接所有方向：
        # h_n: [num_layers * num_directions, B, hidden_size]
        # 重排为 [num_layers, num_directions, B, hidden_size]
        h_n = h_n.view(self.num_layers, self.num_directions, B, self.hidden_size)
        last_layer_h = h_n[-1]  # [num_directions, B, hidden_size]

        # 交换成 [B, num_directions, hidden_size] → 再展平到 [B, num_directions * hidden_size]
        last_h = last_layer_h.transpose(0, 1).contiguous().view(
            B, self.num_directions * self.hidden_size
        )  # [B, fc_in_dim]

        # 轨迹预测： [B, M * out_len * output_size] → reshape
        traj_flat = self.fc_traj(last_h)
        pred_trajs = traj_flat.view(
            B, self.modes, self.out_len, self.output_size
        )  # [B, M, out_len, output_size]

        # mode 概率 logit： [B, M]
        mode_logits = self.fc_logit(last_h)

        return pred_trajs, mode_logits


# ==================== 从 config.yaml 构建模型的辅助函数 ====================

def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    从顶层 config（已经 load 了 config.yaml）中读取 model 段，构建 LSTMForecaster。

    期望 config 里有类似结构：
      model:
        type: "lstm"
        input_size: 6
        output_size: 6
        hidden_size: 256
        num_layers: 2
        out_len: 10
        dropout: 0.0
        bidirectional: false
        modes: 3
    """
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "lstm").lower()

    if model_type != "lstm":
        raise ValueError(f"当前 model.py 只实现了 LSTM，收到 type={model_type}")

    lstm_cfg = LSTMConfig(
        input_size=int(model_cfg.get("input_size", 6)),
        output_size=int(model_cfg.get("output_size", 6)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        out_len=int(model_cfg.get("out_len", 10)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        bidirectional=bool(model_cfg.get("bidirectional", False)),
        batch_first=True,  # 我们整个工程都用 [B, T, C]
        modes=int(model_cfg.get("modes", 3)),
    )

    net = LSTMForecaster(lstm_cfg)
    return net
