cmake_minimum_required(VERSION 3.12)

project(lstm_prediction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 传入 MindSpore Lite 根目录
set(MSLITE_ROOT "" CACHE PATH "Path to MindSpore Lite runtime root (contains 'include' and 'lib')")

# 校验 MindSpore Lite
if(NOT EXISTS "${MSLITE_ROOT}/include/api/model.h")
    message(FATAL_ERROR "Cannot find ${MSLITE_ROOT}/include/api/model.h. Set -DMSLITE_ROOT correctly.")
endif()
if(NOT EXISTS "${MSLITE_ROOT}/lib/libmindspore-lite.so")
    message(FATAL_ERROR "Cannot find ${MSLITE_ROOT}/lib/libmindspore-lite.so. Set -DMSLITE_ROOT correctly.")
endif()

# 可选: 检查 stb_image.h 是否在工程根目录
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/stb_image.h")
    message(WARNING "stb_image.h not found in project root. Please place stb_image.h beside the sources.")
endif()



add_library(lstm_prediction_shared 	SHARED IMPORTED GLOBAL)
set_target_properties(lstm_prediction_shared PROPERTIES
		IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/prebuild/lib/lib_lstm_prediction.so")

# ===== 仅链接动态库 =====
add_executable(run_minimal run_minimal.cpp)

# 头文件搜索路径：包括本项目目录和 MindSpore Lite 目录
target_include_directories(run_minimal PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MSLITE_ROOT}
    ${MSLITE_ROOT}/include
)

# 编译期：告诉 ld 去哪里找依赖库
target_link_directories(run_minimal PRIVATE
    ${MSLITE_ROOT}/lib
    ${MSLITE_ROOT}/third_party/glog
)

# 直接链接已编译的动态库（假设它在 build/ 目录下）
target_link_libraries(run_minimal PRIVATE
    lstm_prediction_shared
    mindspore-lite
    -l:libmindspore_glog.so.0
)

# 运行期 rpath
get_filename_component(_SO_DIR "${CMAKE_SOURCE_DIR}/prebuild/lib/lib_lstm_prediction.so" DIRECTORY)

set_target_properties(run_minimal PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    # INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../lib;${MSLITE_ROOT}/lib"
    INSTALL_RPATH "\$ORIGIN;${_SO_DIR};${MSLITE_ROOT}/lib;${MSLITE_ROOT}/third_party/glog"

)

install(TARGETS run_minimal RUNTIME DESTINATION bin)
