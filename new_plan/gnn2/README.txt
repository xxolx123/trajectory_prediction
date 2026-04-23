gnn2/  —— 打击区域 + 置信度预测
==================================

作用
----
输入：
  - pred_traj:   [B, T, 6]   精修后的预测轨迹（绝对坐标）
  - context:     ContextBatch  (own_info / road_network / ...)
  - intent_feat: [B, D_intent] LSTM2 输出的意图特征（当前直接用 intent_logits）
输出：
  - strike_pos:    [B, 3]  打击点坐标 (x, y, z)
  - strike_radius: [B, 1]  打击半径（>=0）
  - strike_conf:   [B, 1]  置信度 (0..1)

目录结构
----------
  gnn2/
    ├── config.yaml
    ├── README.txt
    ├── code/
    │   ├── data/
    │   │   ├── generate_data.py     # TODO
    │   │   └── dataset.py           # TODO
    │   └── train/
    │       ├── model.py             # StrikeZoneGNN（MLP 占位）
    │       ├── loss.py              # StrikeLoss（占位，返回 0）
    │       └── trainer.py           # 训练主入口（含 --smoke）
    ├── data/
    │   └── raw/
    └── checkpoints/


待定 TODO
---------
1. data/generate_data.py / dataset.py
   - 数据组成：
       pred_traj [T, 6]
       our_info（我方坐标、到达时间、武器型号 等）
       intent_label（可以从 LSTM2 离线预测 / 或 GT）
       target_strike_pos / target_strike_radius / target_strike_conf （GT 标签）
   - 标签来源：尚未定，TODO。

2. train/model.py
   - 当前：MLP 占位（flatten 所有输入 + concat + MLP → 5 维）
   - 待换为真正的 GNN：
        节点 = 预测轨迹关键步 (比如 t=5, t=10) + 我方节点 + 固定目标节点
        边   = 我方↔轨迹：按"我方到达时间 vs 目标到达时间"建权
              轨迹↔目标：按距离建权
        读出 = 在 "决策节点 / 全局节点" 上接 Linear → 5 维

3. train/loss.py
   - 当前：pass，返回 0
   - 待补：MSE(pos) + MSE(radius) + BCE(conf)，或合成"可微 IoU"


对 fusion 的接口约定
--------------------
fusion 里这样调用：
    gnn2 = build_model_from_config(cfg_gnn2, intent_feat_dim=...)
    gnn2.load_state_dict(torch.load("gnn2/checkpoints/<run>/best_*.pt"))
    out = gnn2(pred_traj, ctx, intent_feat)
