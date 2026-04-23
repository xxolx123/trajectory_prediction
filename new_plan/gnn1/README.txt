gnn1/  —— 轨迹选择（根据目标信息计算每条候选轨迹的概率）
=============================================================

作用
----
输入：
  - cand_trajs: [B, M, T, 6]     LSTM1 的 M 条候选轨迹（绝对坐标或归一化空间）
  - target_task / fixed_targets / target_type / road_network
上下文输入格式见 new_plan/common/context_schema.py。

输出：
  - traj_logits: [B, M]
  - traj_probs:  [B, M]  （softmax）

思想：在 M 条候选里，挑一条最符合"目标任务 + 目标类型 + 路网约束"的。
**不用于修正终点**（用户明确要求）。

目录结构
----------
  gnn1/
    ├── config.yaml          # (占位) 网络 / 训练配置
    ├── README.txt           # 本文件
    ├── code/
    │   ├── data/
    │   │   ├── generate_data.py     # TODO: 生成"候选轨迹 + 目标信息 + 标签"样本
    │   │   └── dataset.py           # TODO: Dataset 封装
    │   └── train/
    │       ├── model.py             # TrajSelectionGNN（当前 MLP 占位）
    │       ├── loss.py              # WTA CE 占位（由 lstm1 winner 做 pseudo-label）
    │       └── trainer.py           # 训练主入口（含 --smoke）
    ├── data/
    │   └── raw/                     # (占位) 生成数据
    └── checkpoints/                 # 训练产物


待定 TODO 清单
--------------
1. data/generate_data.py
   - 需要"LSTM1 的候选轨迹"+"目标信息"+"最可能标签"的三元组。
   - 两种常见做法：
       (a) 先用训练好的 LSTM1 离线批量推理得到 M 条候选；
           再按规则 / WTA 生成标签（pseudo label）。
       (b) 和 LSTM1 联合训练，在线用 LSTM1 输出做伪标签。
   - 当前骨架不假设任何一种，函数体内留 `raise NotImplementedError` 或 pass。

2. data/dataset.py
   - Dataset.__getitem__ 应返回：
       cand_trajs: [M, T, 6]
       ctx_fields: dict(target_task, fixed_targets, target_type, road_network)
       label:      int in [0, M)   或 soft label [M]
   - 等上下文接口格式定了再具体实现。

3. train/model.py (TrajSelectionGNN)
   - 当前：MLP 占位（轨迹 flatten + ctx flatten → MLP → [B,M] logit）
   - 待换：真正的 GNN：
        节点 = M 条轨迹 + N 个固定目标 + 全局节点
        边   = 几何邻近 + 任务关联
        消息传递 2~3 层 GraphSAGE/GAT
        读出：只在"轨迹节点"上接一个 Linear → logit

4. train/loss.py
   - 当前：CE(traj_logits, label)
   - 如果用 soft pseudo label，改为 KL divergence

5. train/trainer.py
   - 当前：只提供 --smoke；正式训练分支需要数据生成脚本跑完才能工作


对 fusion 的接口约定
--------------------
fusion 会这样调用：
    gnn1 = build_model_from_config(cfg_gnn1)
    gnn1.load_state_dict(torch.load("gnn1/checkpoints/<run>/best_*.pt"))
    out = gnn1(cand_trajs, ctx)
    # out["traj_probs"]: [B, M]
