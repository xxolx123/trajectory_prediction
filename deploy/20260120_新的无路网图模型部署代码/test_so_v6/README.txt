# 构建（确保已经生成 lib_lstm_prediction.so）
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMSLITE_ROOT=/root/mindspore-lite-2.7.0-linux-aarch64/runtime
cmake --build . -j


# 设置 MindSpore Lite 库路径
export LD_LIBRARY_PATH=/root/mindspore-lite-2.7.0-linux-aarch64/runtime/lib:$LD_LIBRARY_PATH

# 运行最小测试
cd ..
./build/run_minimal
