lstm2/  —— 目标意图（+ 威胁度）预测
======================================

作用
----
输入：
  - hist_raw:     [B, Tin, 6]   原始历史轨迹
  - fut_refined:  [B, Tout, 6]  GNN1 选出、约束优化精修后的预测轨迹
输出：
  - intent_logits: [B, num_intent_classes]   目标意图分类
  - threat_raw:    [B, 1]                    威胁度 raw（sigmoid 后得 0..1）

与 old_plan 的 IntentThreatNet（纯 MLP）区别：
  - 输入从"窗口最后 L 步 flatten"改为"历史 + 精修未来"的整条序列
  - 用 LSTM（而非 MLP）做时序建模
  - 输出字段与老版保持一致，方便 fusion 拼接 [B,M,68] 时复用老逻辑

目录结构
----------
  lstm2/
    ├── config.yaml          # 占位配置
    ├── README.txt           # 本文件
    ├── code/
    │   ├── data/
    │   │   ├── generate_trajs.py    # TODO: 带 intent_label/threat_score 的 CSV
    │   │   └── dataset.py           # TODO: (hist, fut_refined) 对 + 标签
    │   └── train/
    │       ├── model.py             # IntentLSTM（结构已提供，可直接训）
    │       ├── loss.py              # IntentThreatLoss（CE + MSE）
    │       └── trainer.py           # 训练主入口（--smoke 可用）
    ├── data/
    │   └── raw/                     # (占位) 数据
    └── checkpoints/                 # 训练产物


待定 TODO
---------
1. data/generate_trajs.py
   - old_plan/203_prediction_intention_pytorch_v0/code/data/generate_trajs.py
     可以直接复用，输出带 intent_label / threat_score 的 CSV。
   - 但是 LSTM2 的输入是"历史 + 精修未来"，不是"一段历史窗口"。
     训练期：
        - 方案 A：未来用 GT 未来（近似精修后的未来）→ 直接可训
        - 方案 B：用 LSTM1 + GNN1 + 约束优化离线跑一遍，把"精修后未来"当训练
                  样本 → 更接近部署时的分布
     当前脚本未定；TODO。

2. data/dataset.py
   - 按上面方案，产出：
        hist_raw   [Tin, 6]
        fut        [Tout, 6]       （GT 或离线精修轨迹）
        intent_lbl int              0..3 / 或 -1
        threat     float            0..100 / 或 -1

3. train/trainer.py
   - --smoke 已提供；正式 train() 待 Dataset 实现后补。


对 fusion 的接口约定
--------------------
fusion 里这样调用：
    lstm2 = build_model_from_config(cfg_lstm2)
    lstm2.load_state_dict(torch.load("lstm2/checkpoints/<run>/best_*.pt"))
    out = lstm2(hist_raw, fut_refined)
    # out["logits_intent"]: [B, 4]
    # out["threat_raw"]:    [B, 1]
