new_plan/  —— 轨迹预测新方案（多子网络 + 融合）
================================================================================

总体架构
--------
hist_traj ─► OutlierFilter ─► LSTM1 ─► 候选轨迹 [B,M,10,6]
                                           │
  (target_task / fixed_targets /           │
   target_type / road_network)  ──────────►│
                                           ▼
                                  GNN1（选轨迹概率）
                                           │
                                           ▼ argmax
                                   主轨迹 [B,10,6]
                                           │
  (road_network / target_task) ───────────►│
                                           ▼
                               ConstraintOptimizer
                                           │
                                           ▼
                                 精修预测轨迹 [B,10,6]
                                           │
         ┌─────────────────────────────────┼──────────────────────────┐
         ▼                                 ▼                          │
      LSTM2                           GNN2（打击区域）                 │
  意图 + 威胁度                       │                               │
         │                             │ (意图反馈)                   │
         └────────────────────────────►│                               │
                                       │                               │
                                       ▼                               ▼
                            [strike_pos, radius, conf]      最终 [B,M,68]
                                                            （兼容旧 deploy）

说明：
- LSTM1 与 old_plan 几乎一样，**仅去掉 `mode_logits` 输出**（概率由下游 GNN1 算）。
- GNN1 的功能是"轨迹选择"，不再修正终点。
- 其他子网络（约束优化 / LSTM2 / GNN2）当前都是**骨架 + TODO**，等接口定稿后再补。
- fusion 把各子网络串起来，输出保持 `[B, M, 68]`，以兼容现有部署端 cpp。


目录结构（每个子网络一个独立工程 + 一个公共文件夹 + 一个 fusion）
--------
new_plan/
├── README.txt                 (本文件)
├── requirements.txt
│
├── common/                    公共组件，所有子网络都能 import
│   ├── scaler.py              StandardScaler
│   ├── context_schema.py      ContextBatch（外部输入占位）
│   └── outlier_filter.py      异常值剔除（pass）
│
├── lstm1/                     ◆ 已确定；和 old_plan 几乎一样，只去掉 mode_logits
│   ├── config.yaml
│   ├── code/data/             generate_trajs.py / traj_dataset.py / visualize_trajs.py
│   ├── code/train/            model.py / loss.py / trainer.py / eval.py
│   ├── data/                  raw/ processed/
│   └── checkpoints/
│
├── gnn1/                      ◇ 骨架 + TODO（MLP 占位，可以 --smoke）
│   ├── config.yaml
│   ├── code/data/             generate_data.py / dataset.py
│   ├── code/train/            model.py / loss.py / trainer.py
│   ├── data/raw/
│   └── checkpoints/
│
├── constraint_optimizer/      ◇ 骨架 + TODO（算法型 pass，无可学习参数）
│   ├── config.yaml
│   └── code/train/module.py
│
├── lstm2/                     ◇ 骨架 + TODO（model 已可用，data 待补）
│   ├── config.yaml
│   ├── code/data/             generate_trajs.py / dataset.py
│   ├── code/train/            model.py / loss.py / trainer.py
│   └── checkpoints/
│
├── gnn2/                      ◇ 骨架 + TODO（MLP 占位）
│   ├── config.yaml
│   ├── code/data/             generate_data.py / dataset.py
│   ├── code/train/            model.py / loss.py / trainer.py
│   └── checkpoints/
│
└── fusion/                    ◆ 串联各子网络 + 导出 ONNX
    ├── config.yaml            指向各子网络 config / ckpt / scaler
    └── code/
        ├── build.py           加载各子网络 ckpt
        ├── full_net_v2.py     组合模型（输出 [B,M,68] 兼容旧 deploy）
        └── export_onnx.py

图例：
  ◆ = 已完整实现
  ◇ = 骨架，内部 pass / MLP 占位，并明确标注 TODO


使用方式
--------
**每个子网络 (lstm1 / gnn1 / lstm2 / gnn2) 都可以独立运行 --smoke 冒烟测试：**

    cd new_plan/lstm1
    # Linux/macOS:
    export PYTHONPATH="$PWD/code"
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD/code"

    python -m train.trainer --smoke

类似的 smoke 命令也存在于 gnn1 / lstm2 / gnn2 下。

**fusion 的 --smoke 会把所有子网络串起来走一遍（用随机权重）：**

    cd new_plan
    # Linux/macOS:
    export PYTHONPATH="$PWD"
    # Windows PowerShell:
    $env:PYTHONPATH = "$PWD"

    python -m fusion.code.full_net_v2 --smoke

**正式训练 LSTM1：**

    cd new_plan/lstm1
    export PYTHONPATH="$PWD/code"
    python -m data.generate_trajs --config config.yaml   # 产出 data/raw/synthetic_trajectories.csv
    python -m train.trainer --config config.yaml
    python -m train.eval --config config.yaml --split test

**导出 ONNX（fusion，兼容部署端 cpp）：**

    cd new_plan
    export PYTHONPATH="$PWD"
    python -m fusion.code.export_onnx \
        --fusion-config fusion/config.yaml \
        --onnx-out fusion/full_net_v2.onnx


TODO 全清单（等外部接口 / 数据确定后来补）
--------
| 位置                                    | 状态 / 待补                                    |
|-----------------------------------------|------------------------------------------------|
| common/outlier_filter.py                | pass；待补 3σ/物理/Kalman                      |
| common/context_schema.py                | 维度占位；待接真实 context loader              |
| gnn1/code/data/generate_data.py         | NotImplementedError；待定数据产出方式          |
| gnn1/code/train/model.py                | MLP 占位；待换成真正的 GNN                     |
| gnn1/code/train/loss.py                 | CE 占位；待定 soft label / 规则标签策略        |
| constraint_optimizer/code/train/module.py| pass；待定算法型 vs 可学习型                  |
| lstm2/code/data/generate_trajs.py       | NotImplementedError；可从 old_plan 意图脚本改写|
| lstm2/code/data/dataset.py              | NotImplementedError                            |
| gnn2/code/data/generate_data.py         | NotImplementedError；待打击区域 GT            |
| gnn2/code/train/model.py                | MLP 占位；待换成真正的 GNN                     |
| gnn2/code/train/loss.py                 | 永远返回 0；待打击区域 GT                      |


与 old_plan 的关系
--------
- lstm1/ 的 data 脚本（generate_trajs / traj_dataset / visualize_trajs）基本是
  从 old_plan/203_prediction_multi_pytorch_without_map_v0.2 拷贝过来的；
  model.py / loss.py 去掉了 mode_logits / 分类损失，其余保持一致。
- 旧的 model_fusion_v0 被 fusion/ 取代；输出布局 [B,M,68] 完全兼容。


与部署端的关系
--------
- 现有 deploy/20260120_.../make_so_v6/deploy_3trajs.cpp 不需要改。
- 导出 ONNX 时 FullNetV2ForOnnx 只暴露 1 个 3D 输入 x_raw，context 在模型内
  用 buffer 零张量；.cpp 里的 InferenceEngine 按"有且仅有一个 3D 输入"识别
  成功，has_image_=false，传进去的图像被忽略即可。
- 以后真 context 接入后，再同步更新 .cpp 与 ONNX 导出包装器。
