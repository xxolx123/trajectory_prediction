# fusion 部署版 (20260427) —— 测试可执行 run_minimal

# -----------------------------------------------------------------------------
# 前置条件
# -----------------------------------------------------------------------------
# 1) 已经按 make_so/how_to_make_so.txt 编译出 lib_lstm_prediction.so
#    并把它放到 test_so/prebuild/lib/lib_lstm_prediction.so（默认路径）
#    或用 -DPREBUILD_SO=/绝对/路径/lib_lstm_prediction.so 覆盖
# 2) full_net_v2.ms 已经放在 deploy_cfg.ini 的 model_path 指向的位置
#    （默认 ./full_net_v2.ms，即与 run_minimal 同目录）
# 3) deploy_cfg.ini 中的 nb_max / np_max 必须与训练 / 调用方约定一致

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
#   1) 在 stdout 打印 trace_prob / pred_trace / strike_areas / area_prob
#   2) 在当前目录写出 trajectories_vis.csv（含 history / prediction / road / target）

# -----------------------------------------------------------------------------
# 可视化（可选）
# -----------------------------------------------------------------------------
python visualize_trajs.py
# 会在当前目录生成 trajectories_vis.png
