constraint_optimizer/  —— 约束优化（非学习类模块，或可学习的简易版）
=================================================================

作用
----
输入：
  - selected_traj: [B, T, 6]      GNN1 选出的"最可能那条轨迹"（绝对坐标）
  - context:       ContextBatch   路网 / 作战任务等
输出：
  - refined_traj:  [B, T, 6]      经过路网投影/任务边界裁剪/运动学约束后的轨迹

注意：这**不一定是可学习的网络**。可以是：
  (a) 算法型（QP / 投影 / 平滑滤波），不需要训练、无可学习参数；
  (b) 简易可学习型（比如一个小 MLP 作残差修正），需要训练。
本骨架提供的 module.py 既兼容 (a) 也兼容 (b)：forward 签名固定，内部实现
可以自由替换，是否有可学习参数都 OK。

目录结构
----------
  constraint_optimizer/
    ├── config.yaml          # 占位配置
    ├── README.txt           # 本文件
    ├── code/
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── generate_data.py     # TODO: 若走可学习路线，才需要
    │   └── train/
    │       ├── module.py            # ConstraintOptimizer（当前 pass）
    │       ├── loss.py              # TODO: 若可学习才需要
    │       └── trainer.py           # TODO
    └── checkpoints/                 # 若无学习参数，这里可永远为空


待定 TODO
---------
1. train/module.py
   - 决定路线：算法型 vs 可学习型
   - 算法型思路：
       * 用轨迹点到路网 link 的最短距离投影
       * 对速度/加速度做最大值夹取
       * 对位置做任务区域边界裁剪
   - 可学习型思路：
       * 残差 MLP：refined = selected + small_mlp(selected, ctx)
       * 加约束 loss 保证输出不违反硬约束

2. data/generate_data.py / trainer.py / loss.py
   - 仅在走"可学习型"路线时才需要实现


对 fusion 的接口约定
--------------------
fusion 里这样调用：
    refined = constraint_optimizer(selected_traj, ctx)
