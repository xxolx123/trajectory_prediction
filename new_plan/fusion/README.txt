fusion/  —— 把所有子网络拼成 FullNetV2 + 导出 ONNX
=====================================================

定位
----
**fusion 是纯推理封装，不训练**。
每个子网络（lstm1 / gnn1 / lstm2 / gnn2）都是各自独立训练的；fusion 只是在
推理/部署时把它们按固定流水线串起来，给一次 `forward` 拿到最终输出。

- 不定义任何 loss、optimizer、trainer
- 没有 checkpoints/ 目录需要维护（可选的 checkpoints/ 只是给将来"组合后微调"占位）
- `--smoke` 只做推理向检查（形状、值域、top-K 重归一化、确定性）

作用
----
1) 加载各子网络 ckpt（lstm1 必加；其他可选、可无）；
2) 把 OutlierFilter → LSTM1 → GNN1（选轨迹概率） → top-K + 重归一化
   → ConstraintOptimizer → LSTM2 → GNN2 串起来；
3) 对外输出 [B, K, 68] 的单一张量，**与 old_plan 部署端完全兼容**：
   其中 K 来自 gnn1/config.yaml 的 train.keep_top_k（默认 3）。
      0..59 : 未来 10 步 × 6 维（每条候选各自）
      60    : intent_class (argmax(LSTM2.logits_intent))      每条候选各自
      61    : threat_prob  (sigmoid(LSTM2.threat_raw))        每条候选各自
      62..64: strike_pos   (GNN2)                             每条候选各自
      65    : strike_radius(GNN2)                             每条候选各自
      66    : strike_conf  (GNN2)                             每条候选各自
      67    : mode_prob    = top-K 重归一化后的概率（K 条和 = 1）
4) 提供 ONNX 导出（只暴露 1 个 x_raw 输入，ContextBatch 在模型内置零 buffer；
   这样 deploy 端 .cpp 不需要改）。


前向流程（简图）
----------
  hist_traj
      │
      ▼
  OutlierFilter
      │
      ▼
  LSTM1 ──►  M=5 候选 [B,5,10,6]（归一化+delta 空间）
                │
    (task_type, type, position) ──► GNN1
                                      │
                                   probs [B,5]
                                      │
                           topk(K) + renorm sum=1
                                      │
                         top-K 归一化候选
                                      │
                           反归一化 + cumsum
                                      │
                      top-K 物理坐标 [B,K,10,6]
                                      │
                      reshape 到 B*K 走下游
                                      │
  (road_points, road_mask) ─► ConstraintOptimizer
                                      │
                                 refined [B*K, 10, 6]
                                      │
                   LSTM2          GNN2
                 意图/威胁      打击区域
                                      │
                             reshape 回 [B,K,*]
                                      │
                                拼  [B, K, 68]


目录结构
----------
  fusion/
    ├── config.yaml          # 指向各子网络 config / ckpt / scaler 的路径
    ├── README.txt           # 本文件
    ├── code/
    │   ├── full_net_v2.py   # 组合模型（top-K + 下游 K 倍 batch-expand）
    │   ├── build.py         # 构建 FullNetV2（加载各子网络 ckpt + 读 keep_top_k）
    │   └── export_onnx.py   # 导出 ONNX（内置零 ContextBatch buffer）
    └── checkpoints/         # 可选：组合后二次微调的 ckpt


使用
----
# 1) 推理冒烟（不加载任何 ckpt，用随机权重跑完整流水线；只检查 forward）
cd new_plan
$env:PYTHONPATH = "$PWD"    # Windows PowerShell
# export PYTHONPATH="$PWD"  # Linux/macOS
python -m fusion.code.full_net_v2 --smoke
# → out.shape == [B, 3, 68]
#   out[..., 67].sum(-1) ≈ 1              （top-3 重归一化校验）
#   threat_prob / strike_conf ∈ [0, 1]
#   eval 下重复 forward 结果一致

# 2) 加载各子网络 ckpt 后导出 ONNX
python -m fusion.code.export_onnx \
    --fusion-config fusion/config.yaml \
    --onnx-out fusion/full_net_v2.onnx


fusion/config.yaml 配置各子网络的指向：
  lstm1:
    config: "../lstm1/config.yaml"
    ckpt:   "../lstm1/checkpoints/<run_id>/best_lstm_epoch*.pt"
    scaler: "../lstm1/data/processed/scaler_posvel.npz"   # 必填
  gnn1:
    config: "../gnn1/config.yaml"           # 从里面读 train.keep_top_k 作为 K
    ckpt:   ""                              # 空字符串 = 用随机权重
  constraint_optimizer:
    config: "../constraint_optimizer/config.yaml"
    ckpt:   ""                              # 当前是 pass_through
  lstm2:
    config: "../lstm2/config.yaml"
    ckpt:   ""
  gnn2:
    config: "../gnn2/config.yaml"
    ckpt:   ""


ContextBatch 字段归属
----------
ContextBatch（见 common/context_schema.py）里的字段会被不同子网络消费：

  task_type   [B]            long    ──► GNN1（作战任务类型，目前只有 0 = 打击）
  type        [B]            long    ──► GNN1（我方固定目标类型 0..2）
  position    [B, 3]         float   ──► GNN1（我方固定目标 xyz km）
  road_points [B, N_max, 3]  float   ──► ConstraintOptimizer（主干路网点 km-xyz）
  road_mask   [B, N_max]     bool    ──► ConstraintOptimizer（有效点掩码）
  own_info    [B, D_own]     float   ──► LSTM2 / GNN2（我方自身信息占位）

部署端传入时，路网（RoadNetwork = vector<RoadBranchLLH>）需在 C++ 侧把
lon_deg/lat_deg/alt_m 转成局部 ENU（km），以 hist 最后一帧位置为原点。

TODO
----
- 多分支路网：road_points 升到 [B, N_branch_max, N_point_max, 3] + branch_mask
- export_onnx.py：真 ContextBatch 接入后，把 task_type/type/position/road_*/own_info
  都作为独立 ONNX 输入暴露，并同步 deploy/.cpp
