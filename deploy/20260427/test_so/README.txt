# fusion 部署版 (20260427) —— 测试可执行 run_minimal

# -----------------------------------------------------------------------------
# 前置条件
# -----------------------------------------------------------------------------
# 1) 已经按 make_so/how_to_make_so.txt 编译出 lib_lstm_prediction.so
#    并把它放到 test_so/prebuild/lib/lib_lstm_prediction.so（默认路径）
#    或用 -DPREBUILD_SO=/绝对/路径/lib_lstm_prediction.so 覆盖
# 2) 两份 .ms 都已放到 deploy_cfg.ini 中 model_path_with_road / model_path_no_road
#    指向的位置（默认 ./full_net_v2_with_road.ms 与 ./full_net_v2_no_road.ms，
#    即与 run_minimal 同目录）：
#      - 若仅有一份 .ms，可只放对应那份；另一份缺失也允许（只要 use_road 没指向它）
#      - 旧版部署目录中只有单一 full_net_v2.ms 时，可用 model_path= 兜底键
# 3) deploy_cfg.ini 中的 nb_max / np_max 必须与训练 / 调用方约定一致
# 4) deploy_cfg.ini 中的 use_road 决定加载哪份 .ms：
#      use_road=true  → 加载 with_road 版（7 输入：hist_traj / task_type / type /
#                       position + road_points / road_mask + eta；上层传 routes）
#      use_road=false → 加载 no_road  版（5 输入：hist_traj / task_type / type /
#                       position + eta；上层传的 routes 被忽略）
#    GNN2 已经常驻流水线，eta 是必需输入；.ms 不含 eta 会在引擎构造时报错。

# -----------------------------------------------------------------------------
# 编译（确保已经生成 lib_lstm_prediction.so）
# -----------------------------------------------------------------------------
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DMSLITE_ROOT=/root/mindspore-lite-2.7.0-linux-aarch64/runtime
cmake --build . -j

# -----------------------------------------------------------------------------
# 设置 MindSpore Lite 库路径
# -----------------------------------------------------------------------------
export LD_LIBRARY_PATH=/root/mindspore-lite-2.7.0-linux-aarch64/runtime/lib:$LD_LIBRARY_PATH

# -----------------------------------------------------------------------------
# 运行最小测试
# -----------------------------------------------------------------------------
cd ..
./build/run_minimal

# 成功运行会：
#   1) 在 stdout 打印 [Engine] 一行（model 路径 / has_road 能力位）
#   2) 在 stdout 打印 eta_sec used / trace_prob / pred_trace / strike_areas / area_prob
#   3) 在当前目录写出 trajectories_vis.csv（含 history / prediction / road / target）

# -----------------------------------------------------------------------------
# 切换路网模式（不需要重新编译）
# -----------------------------------------------------------------------------
# 编辑 deploy_cfg.ini：
#   use_road=true   # 路网模式
#   use_road=false  # 无路网模式
# 然后重跑 ./build/run_minimal 即可。
# CSV 中 type=road 的行始终由 run_minimal 写入（仅作可视化参考）；当 use_road=false
# 时这些行不会进模型，仅在最终图上看作辅助参考线。

# -----------------------------------------------------------------------------
# 可视化（可选）
# -----------------------------------------------------------------------------
python visualize_trajs.py
# 会在当前目录生成 trajectories_vis.png
