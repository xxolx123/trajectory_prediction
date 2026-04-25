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
2) 把 OutlierFilter → LSTM1 → GNN1（选轨迹概率 + 内部 top-K + 重归一化）
   → ConstraintOptimizer → LSTM2 → GNN2 串起来；
3) 对外输出 [B, K, 68] 的单一张量，**与 old_plan 部署端完全兼容**：
   其中 K 来自 gnn1/config.yaml 的 model.top_k（默认 3），**GNN1 自己负责 topk**，
   fusion 只是拿它的 top_idx / top_probs 做 gather 和展平。
      0..59 : 未来 10 步 × 6 维（**路网约束后** refined 轨迹，每条候选各自）
      60    : intent_class (argmax(LSTM2.logits_intent))      每条候选各自
      61    : threat_prob  (sigmoid(LSTM2.threat_raw))        每条候选各自
      62..64: strike_pos   (GNN2)                             每条候选各自
      65    : strike_radius(GNN2)                             每条候选各自
      66    : strike_conf  (GNN2)                             每条候选各自
      67    : mode_prob    = top-K 重归一化后的概率（K 条和 = 1）
4) 提供 ONNX 导出，**7 路独立输入**：
     hist_traj / task_type / type / position / road_points / road_mask / eta
   （含义见下方 ContextBatch 字段归属表）。每个输入的 batch 维都开放成 dynamic_axes。


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
    (task_type, type, position) ──► GNN1（内含 topk + renorm）
                                      │
                                   top_idx  [B,K]
                                   top_probs[B,K]  (K 条和 = 1)
                                      │
                         gather → top-K 归一化候选
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
    config: "../gnn1/config.yaml"           # 从里面读 model.top_k 作为 K
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


ONNX 输入 / 输出（部署契约）
----------
导出后的 ONNX 模型 7 进 1 出，与甲方流程图一致：

  输入                                  类型      含义
  -----------------------------------------------------------------------------
  hist_traj   [B, 20, 6]              float32   我方观测的历史轨迹 (km / km·s⁻¹)
  task_type   [B]                     int64     敌方作战任务（0=打击）
  type        [B]                     int64     我方固定目标类型 (0/1/2)
  position    [B, 3]                  float32   我方固定目标 xyz (km，局部 ENU)
  road_points [B, NB_max, NP_max, 3]  float32   路网折线点 (km，局部 ENU)
  road_mask   [B, NB_max, NP_max]     bool      路网点掩码（True=有效）
  eta         [B]                     int64     我方预计到达时间（秒，占位）

  输出
  -----------------------------------------------------------------------------
  output      [B, K, 68]              float32   K=3，布局见上"作用"小节

ENU 原点约定：以 hist_traj 末帧的 LLH 作为局部坐标系原点；C++ 部署侧把
RoadNetwork(LLH) → road_points(km) 之前必须用同一原点。


ContextBatch 字段归属（Python 侧）
----------
ONNX 输入打包成 ContextBatch 后被各子网络消费：

  task_type   [B]                       long    ──► GNN1
  type        [B]                       long    ──► GNN1   (我方固定目标类型 0..2)
  position    [B, 3]                    float   ──► GNN1   (我方固定目标 xyz km)
  road_points [B, NB_max, NP_max, 3]    float   ──► ConstraintOptimizer
  road_mask   [B, NB_max, NP_max]       bool    ──► ConstraintOptimizer
  eta         [B]                       long    ──► GNN2 / 占位（我方预计到达时间秒）


C++ 部署侧 TODO（接 7 输入 ONNX 时配合改）
----------
- TrajSystem::Feed(...) 增加 RoadNetwork road、int64_t eta_sec
- InferenceEngine 改成喂 7 输入 ORT session（按上面 INPUT_NAMES 顺序）
- 路网 LLH→ENU 转换路径参考 constraint_optimizer/test_road_net/road_schema.py
  里的 llh_to_enu_km（flat-earth 球面近似；部署侧若用 WGS84 椭球，请按
  GeographicLib 等实现，但接口仍然是 RoadPointLLH(lon_deg, lat_deg, alt_m)
  → (x_km, y_km, z_km)，原点取 hist 末帧的 LLH）
- 若暂时拿不到路网/ETA：把对应输入填全零张量、road_mask 全 False，FullNetV2
  内部会自动 fallback 等价跳过路网投影。
