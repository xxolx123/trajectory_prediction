lstm2/  —— 目标意图（+ 威胁度）预测
======================================

作用
----
对一条 GNN1 选出 + 约束优化精修后的预测轨迹，逐候选地输出：
  - intent_logits: [B, num_intent_classes]   目标意图分类（4 类）
  - threat_raw:    [B, 1]                    威胁度 raw（sigmoid 后 ∈ [0, 1]）

注意：
  - 与原 IntentLSTM（forward(hist, fut)）相比，本版本接口已变更。
  - 标签（labels.py）严格只用 fut_refined + position 计算，hist 不再进入模型。
  - 每个 gnn1 sample 有 K=3 条候选，dataset 内已展开成 M = N_s × 3 条独立样本，
    模型逐候选输出。fusion 端按 sample_idx / cand_k 把 K=3 条结果分组回去。


接口（BREAKING vs 老 IntentThreatNet）
--------------------------------------
forward(
    fut_refined: Tensor,    # [B, Tout=10, 6]   物理 km / km/s（可未归一化）
                            #   或 [B, Tout, 11]（dataset 已工程化 + 归一化好的）
    position:    Tensor,    # [B, 3]            我方目标点（物理 km）
) -> {
    "logits_intent": Tensor [B, num_intent_classes],
    "threat_raw":    Tensor [B, 1],
}

模型内部：若 fut_refined 是 6 维原始量，会自动通过 engineer_features 扩展到
11 维（未归一化）；若已经是 11 维则跳过。dataset 端总是输出 11 维 + 归一化后的张量。


输入特征工程（11 维 per-step）
-----------------------------
[x, y, z, vx, vy, vz,                    # 原 6 维
 dx=x-tx, dy=y-ty, dz=z-tz,              # Δ to target
 ||(dx,dy,dz)||,                         # dist
 ||(vx,vy,vz)||]                         # speed

为什么要做特征工程：标签由 labels.py 的「距离/接近/转角/平均速度」硬规则
推导，把 (Δ, dist, speed) 显式喂给模型可显著加速收敛。


可选模型类型（cfg.model.type）
-----------------------------
  transformer  默认。小型 Transformer Encoder：d_model=128, 2 层, 4 头, FFN=256,
                       learnable PE, mean⊕max pool 后接两个线性头。
  bilstm        2 层双向 LSTM + mean⊕max pool。
  lstm          单向 LSTM，与早期版本一致（仅做 ablation 对照）。

参数量参考（默认 transformer）：约 0.27 M。


目录结构
--------
lstm2/
  ├── config.yaml                # 配置（含 model.type / data / train / loss）
  ├── README.txt                 # 本文件
  ├── code/
  │   ├── data/
  │   │   ├── generate_trajs.py  # 离线生成 data/raw/{split}.npz（已实现）
  │   │   ├── labels.py          # 意图/威胁度硬规则（已实现）
  │   │   ├── synth_roads.py     # 路网合成（已实现）
  │   │   ├── visualize_sample.py# 数据可视化（已实现）
  │   │   ├── dataset.py         # Lstm2Dataset + StandardScaler + build_datasets_from_config
  │   │   └── diagnose_gnn1_ood.py
  │   └── train/
  │       ├── model.py           # IntentTransformer / IntentBiLSTM / IntentLSTM
  │       ├── loss.py            # IntentThreatLoss (CE + MSE)
  │       └── trainer.py         # 训练主入口（含 --smoke）
  │       └── eval.py            # 评估入口（指标 + --vis 可视化）
  ├── data/
  │   ├── raw/                   # generate_trajs.py 输出
  │   │   ├── train.npz
  │   │   ├── val.npz
  │   │   └── test.npz
  │   └── processed/
  │       └── scaler_intent_posvel.npz   # 11 维 StandardScaler（自动生成）
  ├── checkpoints/<run_id>/best_intent_*.pt
  └── eval_vis/<run_id>__<ckpt>/...png


典型流程
--------
0. 准备 GNN1 ckpt（lstm2 数据生成依赖 GNN1）：
       cd new_plan/gnn1
       python -m train.trainer --config config.yaml

1. 生成 lstm2 训练数据：
       cd new_plan/lstm2
       $env:PYTHONPATH = "$PWD/code;$PWD/.."
       python -m data.generate_trajs --config config.yaml --splits train val test

2. 冒烟（不依赖 npz）：
       $env:PYTHONPATH = "$PWD/code"
       python -m train.trainer --smoke

3. 正式训练：
       python -m train.trainer --config config.yaml
   产物：checkpoints/<run_id>/best_intent_epoch*_valloss*_acc*_mae*.pt

4. 评估指标：
       python -m train.eval --config config.yaml --split test
       # 显式指定 ckpt：
       python -m train.eval --config config.yaml --split test \
           --ckpt checkpoints/<run>/best_intent_*.pt

5. 可视化（默认 10 张，每个 gnn1 sample 一张独立 .png）：
       python -m train.eval --config config.yaml --split test --vis
       # 自定义张数 + 可复现种子：
       python -m train.eval --config config.yaml --split test --vis \
           --vis-num 30 --vis-seed 42
       图保存到：
         eval_vis/<ckpt_run>__<ckpt_name>/eval_<split>_<YYYYMMDD_HHMMSS>/
                   └── sample_XXX_idxYYYYYYY.png
       同一 ckpt 多次跑 vis 不互相覆盖（每次 eval 单独一个时间戳子目录）。


对 fusion 的接口约定（BREAKING）
--------------------------------
fusion 调用：
    from train.model import build_model_from_config
    from data.dataset import StandardScaler, engineer_features_np

    # 1) 加载模型 + scaler
    lstm2 = build_model_from_config(cfg_lstm2)
    lstm2.load_state_dict(torch.load(".../best_intent_*.pt"))
    scaler = StandardScaler.load(".../data/processed/scaler_intent_posvel.npz")

    # 2) 准备输入
    #   fut_phys: [B*K, 10, 6]  物理（约束优化后），按候选展开
    #   position: [B*K, 3]      物理目标
    feat11 = engineer_features_np(fut_phys, position)          # [B*K, 10, 11]
    feat_norm = scaler.transform(feat11.astype(np.float64)).astype(np.float32)
    fut_t = torch.from_numpy(feat_norm).float()                # [B*K, 10, 11]
    pos_t = torch.from_numpy(position).float()                 # [B*K, 3]

    # 3) forward
    out = lstm2(fut_t, pos_t)
    # out["logits_intent"]: [B*K, 4]
    # out["threat_raw"]:    [B*K, 1]

    # 4) 按 sample_idx / cand_k 还原回 [B, K, ...]


与早期 old_plan/IntentThreatNet 的差异
--------------------------------------
  - 输入：从「窗口最后 L 步 flatten」改为「逐候选的 fut_refined（10 步）」+ 目标点。
  - 模型：从纯 MLP 改为小型 Transformer Encoder（保留 BiLSTM/LSTM 选项）。
  - 标签：完全由 labels.py 的硬规则导出，且只依赖 fut_refined + position；
          训练样本来自 generate_trajs.py（GNN1 + 约束优化离线产出），
          更接近部署时的分布。
  - 输出 key 兼容：仍是 logits_intent / threat_raw，方便 fusion 拼 [B, M, 68]。
