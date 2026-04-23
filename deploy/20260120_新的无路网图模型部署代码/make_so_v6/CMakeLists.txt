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

# ===== 仅校验 MindSpore Lite（本版不再依赖 OpenCV）=====
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

# ===== 可选：检查本地是否放置了 stb_image.h（放在源码根目录即可）=====
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/stb_image.h")
    message(WARNING "stb_image.h not found in project root. Please place stb_image.h beside the sources.")
endif()

# ===== 生成共享库（保持接口文件名与导出名不变）=====
add_library(lstm_prediction_shared SHARED
    deploy_3trajs.cpp
    lstm_predict_v1.cpp
    deploy_3trajs.h
    lstm_predict_v1.h
)

# 头文件搜索路径：MindSpore Lite + 项目根目录(含 stb_image.h)
target_include_directories(lstm_prediction_shared PRIVATE
    ${MSLITE_ROOT}
    ${MSLITE_ROOT}/include
    ${CMAKE_SOURCE_DIR}
)

# 运行库目录
target_link_directories(lstm_prediction_shared PRIVATE
    ${MSLITE_ROOT}/lib
)

# 仅链接 MindSpore Lite（本版无 OpenCV 依赖）
target_link_libraries(lstm_prediction_shared PRIVATE
    mindspore-lite
)

# 运行期 rpath，便于和 .so 一起分发
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



