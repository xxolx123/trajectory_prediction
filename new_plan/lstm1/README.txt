lstm1/  —— 多模态轨迹预测（LSTM1）
==========================================

本子工程基于 old_plan/203_prediction_multi_pytorch_without_map_v0.2
几乎原样复用，**唯一差异**：
    LSTM1 不再输出 mode_logits（即不自己计算每条轨迹的概率）。
    轨迹概率由下游 GNN1 根据"候选轨迹 + 目标信息"单独计算。

具体对代码的影响：
  - model.py        : 去掉了 fc_logit 头，forward 只返回 pred_trajs
  - loss.py         : 由 old_plan/loss_mtp.py 改名，去掉 CE 分类损失，
                      只保留 WTA 回归损失（为下游 GNN1 留一个
                      compute_wta_best_mode() 工具函数）
  - trainer.py      : 适配 model/loss 的新接口，并加 --smoke
  - eval.py         : 同上；可视化时不再依赖 mode_logits，而用"均匀+best 标亮"
  - generate_trajs.py / traj_dataset.py / visualize_trajs.py : 原样复用

目录结构
----------
  lstm1/
    ├── config.yaml          # 数据 + 模型 + 训练配置
    ├── README.txt           # 本文件
    ├── code/
    │   ├── data/
    │   │   ├── generate_trajs.py      # 合成 synthetic_trajectories.csv
    │   │   ├── traj_dataset.py        # 切窗 Dataset（delta + normalize）
    │   │   └── visualize_trajs.py     # 样本可视化
    │   └── train/
    │       ├── model.py               # LSTMForecaster（无 mode_logits）
    │       ├── loss.py                # WTA 回归
    │       ├── trainer.py             # 训练主入口（含 --smoke）
    │       └── eval.py                # ADE/FDE 等评估 + 随机画图
    ├── data/
    │   ├── raw/                       # 合成 CSV
    │   └── processed/                 # 标准化器 scaler_posvel.npz
    └── checkpoints/                   # 训练产物 best_lstm_epoch*.pt


使用
----------
# 切到 lstm1/ 根目录
cd new_plan/lstm1

# 设置 PYTHONPATH（Windows 用 $env:...）
export PYTHONPATH="$PWD/code"

# 1) 冒烟测试（不依赖 CSV）
python -m train.trainer --smoke

# 2) 生成轨迹数据（产出 data/raw/synthetic_trajectories.csv）
python -m data.generate_trajs --config config.yaml

# 3) 训练
python -m train.trainer --config config.yaml

# 4) 评估（默认从 checkpoints/ 里找最新 ckpt）
python -m train.eval --config config.yaml --split test


对 fusion 的接口约定
--------------------
fusion/ 组装时会这样调用：
    lstm1 = build_model_from_config(cfg_lstm1)
    lstm1.load_state_dict(torch.load("lstm1/checkpoints/<run>/best_*.pt"))
    fut_A_norm = lstm1(x_A_norm)   # [B, M, Tout, 6]

输出永远是 [B, M, Tout, 6] 的单一张量。
归一化空间：与 data.processed/scaler_posvel.npz 对齐。
