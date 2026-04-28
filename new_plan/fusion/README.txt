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
4) 提供 ONNX 导出，**双形态**（部署侧用一个外部 bool flag 选择加载哪份）：
   - full_net_v2_no_road.onnx    无路网版，4~5 路输入：
       hist_traj / task_type / type / position / [eta]
       (eta 仅在 fusion/config.yaml 的 gnn2.enable=true 时存在)
   - full_net_v2_with_road.onnx  含路网约束版，6~7 路输入：
       hist_traj / task_type / type / position / road_points / road_mask / [eta]
   两份 ONNX 共用一份 fusion/config.yaml；区别仅是导出时的 ``--mode``。
   每个输入的 batch 维都开放成 dynamic_axes。
   默认 opset = 11（mindspore-lite 1.8.1 适配）；详见下方"opset 兼容前提"节。


模块开关（fusion/config.yaml 的 enable 字段）
----------
每个子网络段都有 `enable: true/false`：
  - lstm1 / gnn1                  必开（流水线主干，关了直接报错）
  - constraint_optimizer / lstm2 / gnn2   可关；fusion 内部跳过对应步骤，
                                          [B, K, 68] 中相应位置写哨兵值

哨兵值约定（输出列 → 禁用对应模块时的取值）：
  0..59   refined traj   constraint_optimizer 关 → LSTM1 物理轨迹（不投影）
  60      intent_class   lstm2 关 → -1
  61      threat_prob    lstm2 关 → NaN
  62..64  strike_pos     gnn2  关 → NaN
  65      strike_radius  gnn2  关 → NaN
  66      strike_conf    gnn2  关 → NaN
  67      mode_prob      始终由 GNN1 输出（永远有效）

ContextBatch 字段缺省：
  上层只关心已启用模块需要的字段，其它（如 lstm2/gnn2 关闭时只 gnn2 用的 eta）
  允许直接传 None，fusion 内部 `_normalize_ctx` 会按形状用全零张量补齐。
  ONNX 导出仍按 7 路独立张量喂入，未受 None 影响。


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
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
   LSTM2 (新接口):                 GNN2                       (refined 直出)
   1) engineer 11-dim              打击区域                    0..59 列
      = (refined,position)
        附 Δ/dist/speed
   2) (x-μ)/σ  by mean_lstm2/std_lstm2
   3) lstm2(fut_norm, position)
        │
        ▼
   intent_class / threat_prob
                                      │
                             reshape 回 [B,K,*]
                                      │
                                拼  [B, K, 68]


目录结构
----------
  fusion/
    ├── config.yaml          # 指向各子网络 config / ckpt / scaler 的路径，含 enable 开关
    ├── README.txt           # 本文件
    ├── code/
    │   ├── full_net_v2.py   # 组合模型（top-K + 下游 K 倍 batch-expand；按 enable 跳过禁用模块）
    │   ├── build.py         # 构建 FullNetV2（加载各子网络 ckpt + 解析 enable）
    │   ├── export_onnx.py   # 导出 ONNX（内置零 ContextBatch buffer）
    │   └── test_pipeline.py # 端到端测试脚本（按当前 enable 状态串一遍 forward）
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
#   按当前 enable 状态校验 60..66 列：启用列 ∈ [0,1]，禁用列 = -1 / NaN
#   eval 下重复 forward 结果一致

# 2) 端到端测试 + 可视化（用 lstm1/gnn1 真实数据 + 合成路网）
python -m fusion.code.test_pipeline --n 4 --split test --mode both
# --mode both（默认）：no_road / with_road 两种模式各跑一次
#   no_road   = 不喂路网，0..59 列直接是 GNN1 物理轨迹（与 no_road ONNX 同语义）
#   with_road = 现有 2-pass 路网投影流程（与 with_road ONNX 同语义）
# 输出落到 fusion/eval_vis/run_<split>_<stamp>/{no_road,with_road}/：
#   两种 mode 共享同一个时间戳父目录，下面再按 mode 分两个子目录，便于对照阅读：
#   inputs.npz             FullNetV2 真正吃下去的张量（按 mode 决定是否含路网）
#   outputs.npz            完整 [B, K, 68] 输出 + 各字段拆解 + 未投影对照
#   vis_1_history.png       仅 history + GT future + position
#   vis_2_road.png          路网折线（no_road 模式下为空）
#   vis_3_predictions.png   top-K 预测 + GT future
#   vis_4_combined.png      路网 + history + GT + position + top-K（全景）
#   vis_5_compare_with_gnn1_eval.png   5 候选 + top-K 高亮（与 gnn1/eval_vis 对照）
#   vis_6_intent_threat.png            每样本 K 条候选的意图分类 + 威胁度柱状图
#   vis_7_strike_zone.png              每样本 K 条候选的 GNN2 打击区域：圆心=strike_pos.xy，
#                                       半径=strike_radius，填充透明度∝strike_conf
#                                       （仅 GNN2 启用时生成；关闭时 strike_* 列填 NaN，跳过）
#   meta.txt               文字摘要（enable / ckpt / 哨兵值校验 / mode_prob /
#                                      gnn2 启用时附 eta range + per-sample eta + radius/conf 均值）

# 3) 加载各子网络 ckpt 后导出 ONNX（默认双模式 + opset 11）
python -m fusion.code.export_onnx --mode both
#   → fusion/full_net_v2_no_road.onnx
#   → fusion/full_net_v2_with_road.onnx
# 单独出一份用 --mode no_road / --mode with_road
# 改输出目录用 --out-dir <path>（文件名固定为 full_net_v2_{mode}.onnx）


fusion/config.yaml 配置各子网络的指向（每段都有 enable 字段）：
  lstm1:                           # 必开
    enable: true
    config: "../lstm1/config.yaml"
    ckpt:   "../lstm1/checkpoints/<run_id>/best_lstm_epoch*.pt"
    scaler: "../lstm1/data/processed/scaler_posvel.npz"   # 必填
  gnn1:                            # 必开
    enable: true
    config: "../gnn1/config.yaml"           # 从里面读 model.top_k 作为 K
    ckpt:   "../gnn1/checkpoints"           # 目录，自动取最新 .pt
    manual_attention: true                  # opset ≤ 13 部署必须 true，避开 SDPA
  constraint_optimizer:
    enable: true
    config: "../constraint_optimizer/config.yaml"
    ckpt:   ""                              # 当前是 pass_through
  lstm2:                           # 默认 true：已训练 Transformer Encoder
    enable: true
    config: "../lstm2/config.yaml"
    ckpt:   "../lstm2/checkpoints"                              # 目录，自动取最新 .pt
    scaler: "../lstm2/data/processed/scaler_intent_posvel.npz"  # 必填（11 维 StandardScaler）
    manual_attention: true                                      # opset ≤ 13 部署必须 true
  gnn2:                            # 默认 true：StrikeZoneNet（小型 Transformer + FiLM(ETA)，已训练）
    enable: true                   # 启用后 ONNX 输入会多 1 路 eta
    config: "../gnn2/config.yaml"
    ckpt:   "../gnn2/checkpoints"                               # 目录，自动取最新 .pt
    manual_attention: true                                      # opset ≤ 13 部署必须 true

  full_net:
    hist_len: 20
    fut_len: 10
    feature_dim: 6
    use_delta_A: true
    n_intent_classes: 4    # lstm2 禁用时占位用


ONNX 输入 / 输出（部署契约）
----------
fusion 同时导出**两份 ONNX**，部署侧由一个**外部 bool flag 选择加载哪份**
（flag 不进图，ONNX 本身仍是静态图）：

  full_net_v2_no_road.onnx     —— 无路网版（4~5 输入）
  full_net_v2_with_road.onnx   —— 有路网版（6~7 输入）

输入数量取决于 fusion/config.yaml 中 gnn2 是否启用：

  no_road 版（gnn2 关 → 4 输入；gnn2 开 → 5 输入）
  -----------------------------------------------------------------------------
  hist_traj   [B, 20, 6]              float32   我方观测的历史轨迹 (km / km·s⁻¹)
  task_type   [B]                     int64     敌方作战任务（0=打击）
  type        [B]                     int64     我方固定目标类型 (0/1/2)
  position    [B, 3]                  float32   我方固定目标 xyz (km，局部 ENU)
  eta         [B]                     int64     我方预计到达时间（秒，仅 gnn2 开启时存在）

  with_road 版（gnn2 关 → 6 输入；gnn2 开 → 7 输入）
  -----------------------------------------------------------------------------
  hist_traj   [B, 20, 6]              float32
  task_type   [B]                     int64
  type        [B]                     int64
  position    [B, 3]                  float32
  road_points [B, NB_max, NP_max, 3]  float32   路网折线点 (km，局部 ENU)
  road_mask   [B, NB_max, NP_max]     bool      路网点掩码（True=有效）
  eta         [B]                     int64     仅 gnn2 开启时存在

  输出（两份共享）
  -----------------------------------------------------------------------------
  output      [B, K, 68]              float32   K=3，布局见上"作用"小节

ENU 原点约定：以 hist_traj 末帧的 LLH 作为局部坐标系原点；C++ 部署侧把
RoadNetwork(LLH) → road_points(km) 之前必须用同一原点。

no_road 版的语义：内部强制 ConstraintOptimizer = None，0..59 列输出 = GNN1
top-K 物理轨迹（不经路网投影），下游 LSTM2 / GNN2 也基于 GNN1 物理轨迹。
适合任务 / 部署环境拿不到路网的场景，省去 road_points / road_mask 这两路输入。


ContextBatch 字段归属（Python 侧）
----------
ONNX 输入打包成 ContextBatch 后被各子网络消费：

  task_type   [B]                       long    ──► GNN1
  type        [B]                       long    ──► GNN1   (我方固定目标类型 0..2)
  position    [B, 3]                    float   ──► GNN1 + LSTM2
                                                  GNN1: 选轨迹的目标条件
                                                  LSTM2: 工程化 (pos - position) 等 11 维特征
  road_points [B, NB_max, NP_max, 3]    float   ──► ConstraintOptimizer  (仅 with_road)
  road_mask   [B, NB_max, NP_max]       bool    ──► ConstraintOptimizer  (仅 with_road)
  eta         [B]                       long    ──► GNN2                 (仅 gnn2 启用)


导出 ONNX 用法
----------
# 默认两份都出（推荐）：
python -m fusion.code.export_onnx --mode both
#   → fusion/full_net_v2_no_road.onnx
#   → fusion/full_net_v2_with_road.onnx

# 单独出一份：
python -m fusion.code.export_onnx --mode no_road
python -m fusion.code.export_onnx --mode with_road

# 默认 opset = 11（mindspore-lite 1.8.1 验证过）：
python -m fusion.code.export_onnx --mode both --opset 11


opset 兼容前提（默认 opset = 11，mindspore-lite 1.8.1 适配）
----------
为了导出到 opset 11，下列三处需对应启用：

1) LSTM2: fusion/config.yaml 设 lstm2.manual_attention=true
   → build.py 把 sub_cfg.model.type 强制改成 "transformer_manual"，
     避开 nn.TransformerEncoder 的 scaled_dot_product_attention（要 opset ≥ 14）。
   现有 SDPA 训出的 ckpt 与 IntentTransformerManual state_dict 严格兼容，可
   直接 load，无需重训。两套模型数值差在 1e-6 量级（浮点累加顺序）。

2) GNN1: fusion/config.yaml 设 gnn1.manual_attention=true
   → build.py 给 sub_cfg.model 注入 manual_attention=true，工厂构造
     _ManualCrossAttention 替代 nn.MultiheadAttention。
   现有 ckpt 同样兼容，数值差 1e-7 量级。

3) ConstraintOptimizer: _road_arc_projection 已向量化重写
   （constraint_optimizer/code/train/module.py），去掉 torch.searchsorted（要
   opset ≥ 16）改用 broadcast 比较；无 .item() / for 循环 / 布尔索引；
   整数 clamp 用 torch.where 而非 Clip / Max / Min（这些 op 在 opset 11 下不
   支持 int64）。旧循环版保留为 _road_arc_projection_loop，仅供单测对照。
   单测：constraint_optimizer/test_road_net/test_arc_projection_vectorized.py。

4) GNN2: fusion/config.yaml 设 gnn2.manual_attention=true
   → build.py 给 sub_cfg.model 注入 manual_attention=true，工厂构造
     StrikeZoneNetManual 替代 StrikeZoneNet（nn.TransformerEncoder 内部走 SDPA）。
   现有 ckpt 同样兼容，state_dict key 严格一致；数值等价性已由
   gnn2/code/train/_verify_manual_attention.py 验证（max|Δ| ≈ 0），
   fusion 端的组装链路另由 fusion/code/verify_gnn2_match.py 校验。

如果以后部署环境支持更高 opset，可以把上面 manual_attention 都关掉重新导出，
得到稍小、稍快（fused SDPA）的 ONNX；输入契约 / 输出契约不变。


C++ 部署侧 TODO
----------
- 在调用方持有一个 bool flag「是否有路网」：
    flag=true  → 加载 full_net_v2_with_road.onnx，按 6~7 输入约定准备张量
    flag=false → 加载 full_net_v2_no_road.onnx，按 4~5 输入约定准备张量
  flag 不需要进任何模型输入，ONNX 自身的 graph.input 已经反映该模式契约。
- TrajSystem::Feed(...) 在 with_road 路径上传入 RoadNetwork road；no_road
  路径上完全不需要路网。
- 路网 LLH→ENU 转换路径参考 constraint_optimizer/test_road_net/road_schema.py
  里的 llh_to_enu_km（flat-earth 球面近似；部署侧若用 WGS84 椭球，请按
  GeographicLib 等实现，但接口仍然是 RoadPointLLH(lon_deg, lat_deg, alt_m)
  → (x_km, y_km, z_km)，原点取 hist 末帧的 LLH）。
- with_road 版即便临时拿不到路网，也可以喂全零 road_points + 全 False road_mask：
  ConstraintOptimizer 向量化版会用 torch.where 自动 fallback 到 GNN1 物理轨迹，
  与 no_road 版语义等价（但 no_road 版 ONNX 体积更小、算子更少，建议优先选）。
- LSTM2 的 11 维 StandardScaler（mean_lstm2 / std_lstm2）和 LSTM1 的
  mean_A / std_A 都以 register_buffer 形式打包进 ONNX，C++ 侧无需感知/读取
  额外 scaler 文件。
- 默认 ONNX opset = 11，由 mindspore-lite-1.8.1-linux-x64 的 converter_lite
  --fmk=ONNX 路径验证过；导出端用 onnx.checker.check_model 自检，目标转换工具
  上实际兼容性建议在部署侧再过一遍冒烟。
