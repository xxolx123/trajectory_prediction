gnn2/  —— 打击区域 + 置信度预测（领前打击决策）
=================================================

业务语义（甲方对齐版）
----------
场景：**我方反击装备**从 position（我方固定资产）发射，飞行 eta 秒后抵达打击点；
同时敌方沿预测轨迹移动；t=eta 那一刻敌方落在 pred_traj 的某一帧。
GNN2 要算出"敌方届时所在的位置 + 不确定圆 + 概率"，作为我方武器的领前瞄准点。

输入：
  pred_traj  [B, T=10, 6]   ConstraintOptimizer 输出的"路网约束后预测轨迹"
                             （fusion 里 B = batch * top_k = batch * 3，三条候选各跑一次；
                              前 3 维 xyz km，后 3 维 vel km·s⁻¹；每步 60 秒）
  eta        [B]             我方装备抵达时间（int64 秒；范围 [0, 600] = [0, T*time_step_s]）

输出：
  strike_pos    [B, 3]   敌方在 t=eta 时刻的位置（pred_traj 在 eta 处的插值）km
  strike_radius [B, 1]   该位置的不确定半径 km（softplus + clamp，∈ [r_min, r_max]）
  strike_conf   [B, 1]   敌方届时确实落在圆内的概率 ∈ [0, 1]

时间索引约定：
  pred_traj[t, 0:3] = hist_end 后 (t+1)*time_step_s 秒处的位置；
  eta=0   → 敌方"现在"位置（=hist_end，gnn1 数据里 hist 末帧已对齐到原点）
  eta=60  → pred_traj[0]
  eta=120 → pred_traj[1]   ← 甲方举例 "我方装备 120s 抵达" 对应这里
  eta=600 → pred_traj[9]   ← 末帧
  中间值：线性插值（含 [0, 60] 区间在 hist_end ↔ pred_traj[0] 之间）


网络结构（不真用 GNN，per-trajectory 序列回归）
----------------------------------------------
fusion 已经把 3 条候选拍平成独立样本走 [B*K] 维度，gnn2 一次只看一条轨迹，
"GNN" 形态没有意义。最终选用：

    pred_traj [B,T,6]
        │  xyz - 首帧 xyz（平移不变化；vel 不动）
        ▼
    Linear(6, d_emb) + learnable PE
        │
        ├──────────── ETA[B] long
        │             │  /eta_scale_seconds → MLP → (γ, β) ∈ R^{2*d_emb}
        │             ▼
        ▼  FiLM:   x ← x * (1 + γ) + β
    Transformer Encoder × num_layers (pre-LN, GELU FFN)
        │
        ▼  mean pool over T
    head MLP → 5 维 raw
        │
        ├ raw[:, 0:3] + 首帧 xyz   = strike_pos     (km)
        ├ softplus(raw[:, 3:4])
        │   ⇣ clamp(max = r_max - r_min)
        │   + r_min                = strike_radius  (km, ∈ [r_min, r_max])
        └ sigmoid(raw[:, 4:5])     = strike_conf    (∈ [0, 1])

平移不变化：训练时 hist_end 在原点，推理时 hist_end 在任意 ENU；
模型把 xyz 减首帧再加回首帧，对绝对位置不敏感。

ETA 通过 FiLM 进入网络，可以理解为"按 eta 选择性地放大/平移轨迹特征"，
让网络据此输出对应时刻的位置 + 不确定度。


双实现 + ckpt 互通（ONNX opset 11 兼容）
---------------------------------------
两份 nn.Module 类，**state_dict key 严格一致**，可互相 load_state_dict：

  StrikeZoneNet         库版，用 nn.TransformerEncoder（SDPA，GPU 训练快；需 opset >= 14）
  StrikeZoneNetManual   手写版，用 _ManualMultiheadAttention + _ManualEncoderLayer
                        （仅 matmul / softmax / linear / layernorm / gelu / reshape，
                         opset 11 全支持；mindspore-lite 1.8.1 默认就是 opset 11）

工厂逻辑（gnn2/code/train/model.py:build_model_from_config）：
  - cfg.model.type ∈ {"strike_zone_transformer", "strike_zone_transformer_manual"}
  - 若 cfg.model.manual_attention=true，则**强制覆盖** type 为 _manual

训练永远走 SDPA 版（GPU 快）；导 ONNX 时 fusion/build.py 注入
manual_attention=true，工厂自动切到 Manual 版加载同一份 ckpt，无需重训。

数值等价性已通过 _verify_manual_attention.py 验证，两版输出 max|Δ| = 0。


目录结构
----------
  gnn2/
    ├── config.yaml
    ├── README.txt
    ├── code/
    │   ├── data/
    │   │   ├── generate_data.py    # 加载 gnn1 ckpt 跑 top-3，按 R2 规则合成 GT
    │   │   └── dataset.py
    │   └── train/
    │       ├── model.py            # StrikeZoneNet / StrikeZoneNetManual + 工厂
    │       ├── loss.py             # MSE(pos)+MSE(radius)+BCE(conf)
    │       ├── trainer.py          # 训练主入口（含 --smoke）
    │       └── _verify_manual_attention.py   # 验证 SDPA vs Manual 数值等价
    ├── data/
    │   └── raw/                    # generate_data.py 写这里
    └── checkpoints/


合成数据规则（R2：候选 vs target 在 t=eta 的偏差）
-------------------------------------------------
打击区域 GT 暂时用启发式规则生成（业务侧的真 GT 后续替换 generate_data.py 即可）。

复用 gnn1 已经产出的：
  ../gnn1/data/cache/{split}.npz       history / candidates / **targets** (norm+delta)
  ../gnn1/data/cache/scaler_posvel.npz LSTM1 共用 6 维 scaler
  ../gnn1/data/raw/{split}.npz         candidates / task_type / type / position
  ../gnn1/checkpoints/<run>/*.pt       已训好的 GNN1 ckpt（top-3 选轨用）

为每个 (gnn1 raw_idx, top-K=3 内的 k) 产出一条 gnn2 sample：

  1. 加载 GNN1 ckpt + 跑 forward → top_idx [N_s, K=3], top_probs [N_s, K]
     （和 lstm2/code/data/generate_trajs.py 的 top-3 选轨模式一致；保证训练
      分布对齐推理：fusion 里 gnn2 永远只看到 GNN1 选中的 top-3）

  2. 反归一化 candidates 和 targets 到物理 6D（hist_end=(0,0,0) 约定）
     pos = cumsum(scaler.inverse(delta_pos))
     vel = scaler.inverse(vel)

  3. 对每条 sample / 每条 top-K 候选 k：
       eta_k = rng.integers(eta_min_sec, eta_max_sec + 1)
       step_float = eta_k / time_step_s   ∈ [0, T]
       cand_at_eta   = interp(top_phys[k], step_float)    # 候选在 t=eta 的位置
       target_at_eta = interp(target_phys, step_float)    # GT  在 t=eta 的位置
       # 关键：插值时把 hist_end=(0,0,0) 拼到轨迹前面，让 step_float=0 自然对应
       # hist_end，无需额外外推

  4. gt_strike_pos    = cand_at_eta            # 让模型学会"按 eta 在自己输入轨迹上插值"
     gt_strike_radius = clip(a + b * ||cand_at_eta - target_at_eta||, r_min, r_max)
                        # 该候选在 t=eta 偏离 GT 多远 → radius 多大
     gt_strike_conf   = clip(1 - radius / r_max, 0, 1)

物理直觉验证（运行 generate_data.py 的健康检查会打印）：
  - eta 越大 → 通常 cand vs target 偏差越大 → radius 越大 → conf 越低（"未来越远越不确定"）
  - 同一 window 的 3 条 top-K 候选 conf 不同（甲方说的"概率"语义就是"该候选预测得有多准"）
  - 例（val 上 1.1M 样本）：
        eta∈[  0, 60)  radius_mean = 0.500 km   conf_mean = 0.950
        eta∈[ 60,120)  radius_mean = 0.500 km   conf_mean = 0.950
        eta∈[240,360)  radius_mean = 0.515 km   conf_mean = 0.949
        eta∈[480,600)  radius_mean = 0.916 km   conf_mean = 0.908

config.yaml 的 data: 段控制全部参数：
  eta_min_sec / eta_max_sec / radius_a_km / radius_b /
  radius_min_km / radius_max_km / time_step_s / fut_len_steps

注意：暂未走 ConstraintOptimizer（部署时 fusion 会喂约束后轨迹给 gnn2）；
跳过的好处是 radius 语义干净（纯 LSTM1 候选 vs GT 偏差，不混入"路网投影位移"），
轻微的训练/部署分布漂移对"按 eta 插值"任务影响很小。若后续要加，参考
lstm2/code/data/generate_trajs.py 的 build_road_batch + run_constraint_batch 即可。


使用方式
--------
**冒烟测试**（不依赖 .npz；构造随机输入跑一遍 forward + 真 loss + backward）::

    cd new_plan/gnn2
    # Windows PowerShell
    $env:PYTHONPATH = "$PWD/code"
    # Linux/macOS
    # export PYTHONPATH="$PWD/code"

    python -m train.trainer --smoke


**正式数据生成**（前置：先在 gnn1/ 下跑过 cache_lstm1_preds.py + generate_data.py，
并训练好 gnn1 ckpt）::

    cd new_plan/gnn2
    # 注意 PYTHONPATH 同时挂 gnn2/code 和 new_plan（要 import gnn1.code.train.model）
    $env:PYTHONPATH = "$PWD/code;$PWD/.."

    python -m data.generate_data --config config.yaml --splits val
    python -m data.generate_data --config config.yaml --splits train val test

健康检查会按 eta 分桶打印 radius_mean / conf_mean；按 cand_k 打印 top_prob_mean
（top-1 概率应最大）。若 eta 越大 radius 没有显著增大、或 cand_k 间 conf 没有差异，
说明 GNN1 ckpt 异常或 scaler 不匹配。


**正式训练**::

    python -m train.trainer --config config.yaml

每 epoch 打 train/val 的 total / pos / radius / conf 各自 loss，按 val.total
最低保留 top-3 ckpt 到 checkpoints/<run_id>/。


**验证 SDPA vs Manual 数值等价**（导 ONNX 之前的最后一道保险）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m gnn2.code.train._verify_manual_attention

期望输出：
    max|Δ strike_pos|    ≈ 0
    max|Δ strike_radius| ≈ 0
    max|Δ strike_conf|   ≈ 0
    OK


对 fusion 的接口约定
--------------------
fusion 里这样调用（与 lstm2 / gnn1 同款的 manual_attention 注入）::

    # fusion/config.yaml:
    gnn2:
      enable: true
      config: "../gnn2/config.yaml"
      ckpt:   "../gnn2/checkpoints"
      manual_attention: true   # opset <= 13 部署必开

    # fusion/code/build.py 自动把 manual_attention=true 注入子网络 cfg.model，
    # gnn2 工厂据此构造 StrikeZoneNetManual，加载同一份 ckpt。

    # fusion/code/full_net_v2.py 中的调用没有变化：
    out = self.gnn2(refined_flat, ctx_flat.eta)
    # input shape:  refined_flat [B*K, 10, 6], eta [B*K] long
    # output shape: strike_pos [B*K, 3], strike_radius [B*K, 1], strike_conf [B*K, 1]


等真 GT 来时怎么改
------------------
合成数据规则集中在 gnn2/code/data/generate_data.py 的 process_split 函数里，
把"5. 合成 ETA + 计算 strike GT"那一段换成业务侧真 GT 的提取逻辑即可。
dataset / model / loss / trainer 都不用动。

如果业务侧真 eta 的语义跟"我方装备抵达时间"差很大（比如另有定义），那 model.py
里的 FiLM 路径不变，只是训练数据要相应重生成。
