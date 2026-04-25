test_road_net/  —— GNN1 → ConstraintOptimizer 路网约束的端到端测试 + 可视化
================================================================================

目的
----
在没有真实路网数据的当下，把整条链路串起来跑通：
  GNN1（已训练 ckpt） → top-3 候选轨迹（物理 km 坐标）
  → 合成路网（甲方 LLH 接口格式 RoadNetwork = list<RoadBranchLLH>）
  → 走 ConstraintOptimizer.road_projection 投影
  → 可视化：原始 top-3 vs 投影 top-3 vs 路网

文件
----
  road_schema.py      镜像甲方 C++ 接口的 RoadPointLLH / RoadBranchLLH
                      + LLH ↔ 局部 ENU(km) 转换工具（flat-earth 近似）
                      + RoadNetwork → ContextBatch 张量
  synth_road.py       根据候选轨迹覆盖范围合成 3 条分支（直线主干 + 平行支
                      路 + 弯曲支路），输出 LLH 接口格式
  test_projection.py  主入口：拉 GNN1 ckpt + dataset，挑 N 个样本，造路网，
                      投影，画图
  vis/                可视化 png 输出（脚本默认目录）

用法
----
1. 在 new_plan/ 根目录、激活 conda 环境后：

    $env:PYTHONPATH = "$PWD"
    python -m constraint_optimizer.test_road_net.test_projection `
        --split test --n 6

2. 自定 ckpt / 输出目录 / 路网原点：

    python -m constraint_optimizer.test_road_net.test_projection `
        --gnn1-ckpt new_plan/gnn1/checkpoints/20260426001534/best_gnn1_epoch001_valloss0.9508.pt `
        --split test --n 9 `
        --out constraint_optimizer/test_road_net/vis `
        --origin-lon 116.30 --origin-lat 39.90 --origin-alt 0.0

3. 输出：
    - 控制台：每个样本的 top_idx / top_probs / 平均位移量 (km)
    - vis/road_proj_{split}_{timestamp}.png：网格可视化

可视化图例
----------
  浅灰细线   ：LSTM1 的 5 条候选（faded）
  粗灰折线   ：合成路网（不同分支用不同灰阶，顶点用圆点）
  彩色虚线★ ：GNN1 选出的 top-3 原始轨迹（端点五角星）
  彩色实线■ ：投影到路网后的 top-3 轨迹（端点正方形 + rank/概率标注）
  红色五角星 ：我方固定目标 position
  黑色五角星 ：origin（hist 末帧 = 局部坐标系原点）

期望看到的现象
--------------
- 投影后的实线会"贴"到最近的路网折线上；同一条候选可能被拉到不同分支。
- 起点附近因为 3 条分支汇聚，3 条投影轨迹起点会比较接近；越远离原点
  3 条候选越发散，倾向贴到不同支路。
- 速度通道不被改动（投影只动位置），可视化里只画 xy，影响不可见。
- 若把 --gnn1-ckpt 设为某个未训练好（loss 高）的 ckpt，会看到原始候选
  比较"抖"，但投影后依然能拉到路网上。

注意
----
- 真实部署时，路网由甲方 RoadNetwork 接口给（lon/lat/alt 度+米），C++
  侧需要把它先转成 ContextBatch.road_points / road_mask 再喂模型。
  本测试的合成路网 → ENU 转换路径与部署侧的转换路径在数值上一致。
- 为简化 demo，LLH↔ENU 用 flat-earth 球面近似（误差对几公里量级路网
  完全可接受）；部署若用 WGS84 椭球转换，数值会有亚米级差异，不影响
  投影结果定性结论。
