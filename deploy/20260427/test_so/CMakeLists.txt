cmake_minimum_required(VERSION 3.12)

project(lstm_prediction_test LANGUAGES CXX)

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

# 已编译好的 .so（make_so 产物）的预置路径；如果你把 .so 放在别处，
# 用 -DPREBUILD_SO=/绝对/路径/lib_lstm_prediction.so 覆盖即可
set(PREBUILD_SO "${CMAKE_SOURCE_DIR}/prebuild/lib/lib_lstm_prediction.so"
    CACHE FILEPATH "Path to pre-built lib_lstm_prediction.so")
if(NOT EXISTS "${PREBUILD_SO}")
    message(WARNING
        "Pre-built ${PREBUILD_SO} not found. "
        "Set -DPREBUILD_SO=/path/to/lib_lstm_prediction.so to point to your build output.")
endif()

add_library(lstm_prediction_shared SHARED IMPORTED GLOBAL)
set_target_properties(lstm_prediction_shared PROPERTIES
    IMPORTED_LOCATION "${PREBUILD_SO}")

add_executable(run_minimal run_minimal.cpp)

target_include_directories(run_minimal PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MSLITE_ROOT}
    ${MSLITE_ROOT}/include
)

target_link_directories(run_minimal PRIVATE
    ${MSLITE_ROOT}/lib
    ${MSLITE_ROOT}/third_party/glog
)

target_link_libraries(run_minimal PRIVATE
    lstm_prediction_shared
    mindspore-lite
    -l:libmindspore_glog.so.0
)

# 运行期 rpath：先找 .so 同目录，再找 MindSpore Lite
get_filename_component(_SO_DIR "${PREBUILD_SO}" DIRECTORY)
set_target_properties(run_minimal PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "\$ORIGIN;${_SO_DIR};${MSLITE_ROOT}/lib;${MSLITE_ROOT}/third_party/glog"
)

install(TARGETS run_minimal RUNTIME DESTINATION bin)
