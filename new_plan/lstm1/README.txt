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
  - generate_trajs.py: **已重写为采样式**（见下方"数据生成"）
  - traj_dataset.py : **已修复 train/val/test 划分**（按
                      motion_model × traj_type 分层 shuffle）
  - visualize_trajs.py: 原样复用


目录结构
----------
  lstm1/
    ├── config.yaml          # 数据 + 模型 + 训练配置
    ├── README.txt           # 本文件
    ├── code/
    │   ├── data/
    │   │   ├── generate_trajs.py      # 合成 synthetic_trajectories.csv（采样式）
    │   │   ├── traj_dataset.py        # 切窗 Dataset（delta + normalize + 分层划分）
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


数据生成（新版，采样式）
----------
生成思路：不再做
  (motion_model × speed × accel × traj_type × angle × start_coeff × s_pair)
的穷举枚举，而是对每条轨迹独立采样"速度曲线 + 偏航率曲线"，再做前向积分。
类标签通过偏航率采样器的形状保证：
  - straight   : yaw_rate 恒为 0（可选极小方向漂移）
  - left_turn  : 1~N 次转向，**全部左转**，每次角度 20–120°
  - right_turn : 同上，全部右转
  - u_turn     : 一次 ≈180° 的大转向（默认 duration <= 18 步，保证能装进
                 训练窗口 in_len=20 里，模型能学到"完整 U 形"）
  - s_curve    : >=2 段交替左右转，每段角度 20–90°

类别边界：
  - left/right_turn 角度 [20°, 120°]
  - u_turn 角度 [150°, 210°]
  - 120°~150° 作为缓冲区，保证形态无歧义

关键 config.data 字段（见 config.yaml 有更详细注释）：

  time_step, num_steps         # 时间尺度
  traj_types                   # 4 类标签
  num_traj_per_type            # 每类目标样本数

  speed_range_kmh              # 速度采样区间
  min_speed_kmh
  max_accel_mps2, max_decel_mps2
  max_yaw_rate_deg_s           # 偏航率上限（物理约束）

  speed_segments_range         # 速度曲线分几段
  speed_jitter_kmh             # 每步高斯抖动
  allow_stop, stop_prob        # 是否允许停车段

  straight:
    heading_drift_deg_s

  turn:                        # left_turn / right_turn 共用
    n_turns_range
    angle_range_deg
    duration_range_steps
    min_gap_steps

  u_turn:
    n_turns_range              # 默认 [1,1]；设 [1,2] 允许"去-回"
    angle_range_deg            # 默认 [150, 210]（≈180°）
    duration_range_steps       # 默认 [6, 18]
    min_gap_steps
    direction                  # left/right/random

  s_curve:
    n_segments_range
    angle_range_deg
    segment_len_range_steps
    gap_steps_range
    first_direction            # left/right/random

已废弃（DEPRECATED，新生成器不再读取，出现只会打 warning）：
  speeds_kmh, motion_models, accel_levels, dv_max_kmh,
  turn_total_deg, turn_total_deg_list, turn_start_idx, turn_start_coeffs,
  s_left_start, s_left_end, s_left_coeff_pairs,
  num_traj_per_combo, speed_noise_kmh, accel_noise_rel


使用
----------
# 切到 lstm1/ 根目录
cd new_plan/lstm1

# 设置 PYTHONPATH
# Linux/macOS:
export PYTHONPATH="$PWD/code"
# Windows PowerShell:
$env:PYTHONPATH = "$PWD/code"

# 1) 冒烟测试（不依赖 CSV）
python -m train.trainer --smoke

# 2) 生成轨迹数据（产出 data/raw/synthetic_trajectories.csv）
python -m data.generate_trajs --config config.yaml
#    小量预生成（每类 20 条），便于可视化检查：
python -m data.generate_trajs --config config.yaml --num-per-type 20

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
