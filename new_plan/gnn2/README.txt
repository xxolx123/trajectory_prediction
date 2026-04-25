gnn2/  —— 打击区域 + 置信度预测
==================================

作用
----
输入（精简版接口，已与业务侧对齐）：
  - pred_traj:  [B, T=10, 6]  ConstraintOptimizer 输出的"路网约束后预测轨迹"
                              （fusion 里 B = batch * top_k = batch * 3，三条候选各跑一次；
                               前 3 维 xyz km，后 3 维 vel km·s⁻¹）
  - eta:        [B]           我方预计到达时间（int64 秒；模型内部归一化为小时）

输出：
  - strike_pos:    [B, 3]  打击区域中心 (x, y, z) km
  - strike_radius: [B, 1]  打击半径 km（>= 0，softplus）
  - strike_conf:   [B, 1]  置信度 (0..1，sigmoid)

注意：GNN2 不再消费 task_type / type / position / road_* / 意图特征；这些字段
属于上游模块的输入，不影响打击区域决策。

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
       pred_traj [T, 6]                   路网约束后轨迹
       eta       int64 秒                 我方预计到达时间
       gt_strike_pos / gt_strike_radius / gt_strike_conf  GT 标签
   - 标签来源：尚未定，TODO。

2. train/model.py
   - 当前：MLP 占位（traj flatten + eta 标量 → MLP → 5 维）
   - 待换为真正的 GNN：
        节点 = 预测轨迹关键步 (比如 t=5, t=10) + 我方节点
        边   = 我方↔轨迹：按"我方到达时间 vs 目标到达时间"建权
              轨迹↔轨迹：按距离建权
        读出 = 在 "决策节点 / 全局节点" 上接 Linear → 5 维

3. train/loss.py
   - 当前：pass，返回 0
   - 待补：MSE(pos) + MSE(radius) + BCE(conf)，或合成"可微 IoU"


对 fusion 的接口约定
--------------------
fusion 里这样调用：
    gnn2 = build_model_from_config(cfg_gnn2)
    gnn2.load_state_dict(torch.load("gnn2/checkpoints/<run>/best_*.pt"))
    out = gnn2(refined_traj, ctx.eta)

input shape: refined_traj [B*K, 10, 6], eta [B*K] long
output shape: strike_pos [B*K, 3], strike_radius [B*K, 1], strike_conf [B*K, 1]
