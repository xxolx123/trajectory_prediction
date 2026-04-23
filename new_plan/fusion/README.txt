fusion/  —— 把所有子网络拼成 FullNetV2 + 导出 ONNX
=====================================================

作用
----
1) 加载各子网络 ckpt（lstm1 必加；其他可选、可无）；
2) 把 OutlierFilter → LSTM1 → GNN1（选轨迹概率）→ ConstraintOptimizer
   → LSTM2 → GNN2 串起来；
3) 对外输出 [B, M, 68] 的单一张量，**与 old_plan 部署端完全兼容**：
      0..59 : 未来 10 步 × 6 维
      60    : intent_class (argmax(LSTM2.logits_intent))
      61    : threat_prob  (sigmoid(LSTM2.threat_raw))
      62..64: strike_pos   (GNN2)
      65    : strike_radius(GNN2)
      66    : strike_conf  (GNN2)
      67    : mode_prob    (GNN1)
4) 提供 ONNX 导出（只暴露 1 个 x_raw 输入，context 在模型内置 dummy；
   这样 deploy 端 .cpp 不需要改）。

目录结构
----------
  fusion/
    ├── config.yaml          # 指向各子网络 config / ckpt / scaler 的路径
    ├── README.txt           # 本文件
    ├── code/
    │   ├── full_net_v2.py   # 组合模型
    │   ├── build.py         # 构建 FullNetV2（加载各子网络 ckpt）
    │   └── export_onnx.py   # 导出 ONNX
    └── checkpoints/         # 可选：组合后二次微调的 ckpt


使用
----
# 1) 冒烟（不加载任何 ckpt，用随机权重跑完整流水线）
cd new_plan
export PYTHONPATH="$PWD"
python -m fusion.code.full_net_v2 --smoke

# 2) 加载各子网络 ckpt 后导出 ONNX
python -m fusion.code.export_onnx \
    --fusion-config fusion/config.yaml \
    --onnx-out fusion/full_net_v2.onnx


在 fusion/config.yaml 中配置各子网络的指向：
  lstm1:
    config: "../lstm1/config.yaml"
    ckpt:   "../lstm1/checkpoints/<run_id>/best_lstm_epoch*.pt"
    scaler: "../lstm1/data/processed/scaler_posvel.npz"   # 必填
  gnn1:
    config: "../gnn1/config.yaml"
    ckpt:   ""   # 空字符串 = 用随机权重（骨架阶段 OK）
  constraint_optimizer:
    config: "../constraint_optimizer/config.yaml"
    ckpt:   ""   # 当前是 pass，不需要 ckpt
  lstm2:
    config: "../lstm2/config.yaml"
    ckpt:   ""
  gnn2:
    config: "../gnn2/config.yaml"
    ckpt:   ""
