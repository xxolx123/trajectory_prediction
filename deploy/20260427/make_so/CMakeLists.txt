cmake_minimum_required(VERSION 3.12)

project(lstm_prediction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== 手动传入 MindSpore Lite 根目录 =====
# 用法示例：
#   cmake -S . -B build -DMSLITE_ROOT=/opt/mindspore-lite-2.7.0-linux-aarch64/runtime
set(MSLITE_ROOT "" CACHE PATH "Path to MindSpore Lite runtime root (contains 'include' and 'lib')")

# ===== 校验 MindSpore Lite =====
if(NOT EXISTS "${MSLITE_ROOT}/include/api/model.h")
    message(FATAL_ERROR
        "Cannot find ${MSLITE_ROOT}/include/api/model.h . "
        "Please set -DMSLITE_ROOT to the MindSpore Lite runtime root.")
endif()
if(NOT EXISTS "${MSLITE_ROOT}/lib/libmindspore-lite.so")
    message(FATAL_ERROR
        "Cannot find ${MSLITE_ROOT}/lib/libmindspore-lite.so . "
        "Please set -DMSLITE_ROOT correctly.")
endif()

# ===== 生成共享库（保持接口文件名与导出名不变）=====
# fusion 部署版（20260427）：不再依赖 stb_image / OpenCV，仅链接 MindSpore Lite。
add_library(lstm_prediction_shared SHARED
    deploy_3trajs.cpp
    lstm_predict_v1.cpp
    deploy_3trajs.h
    lstm_predict_v1.h
)

# 头文件搜索路径
target_include_directories(lstm_prediction_shared PRIVATE
    ${MSLITE_ROOT}
    ${MSLITE_ROOT}/include
    ${CMAKE_SOURCE_DIR}
)

# 运行库目录
target_link_directories(lstm_prediction_shared PRIVATE
    ${MSLITE_ROOT}/lib
)

# 链接 MindSpore Lite（dlfcn 用 ${CMAKE_DL_LIBS}）
target_link_libraries(lstm_prediction_shared PRIVATE
    mindspore-lite
    ${CMAKE_DL_LIBS}
)

# 运行期 rpath
set_target_properties(lstm_prediction_shared PROPERTIES
    OUTPUT_NAME "lib_lstm_prediction"   # 生成 lib_lstm_prediction.so
    PREFIX ""                           # 取消默认前缀；上面已显式以 lib_ 开头
    VERSION 1.0.0
    SOVERSION 1
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../lib;${MSLITE_ROOT}/lib"
)

install(TARGETS lstm_prediction_shared LIBRARY DESTINATION lib)
install(FILES deploy_3trajs.h lstm_predict_v1.h DESTINATION include)
