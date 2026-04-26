// lstm_predict_v1.h
// fusion 部署版（20260427）：
// 主签名与 20260120 旧版保持一致，方便上层无感切换；
// 唯一区别是 LocData_route 已重设计为 LLH 折线，上层调用方需按
// deploy_3trajs.h 的新结构构造路网。

#pragma once
#include <map>
#include <tuple>
#include <vector>

#include "deploy_3trajs.h"

// 输入：18+ 列轨迹（含目标/任务/ETA 元信息）+ 路网折线 + 步长（毫秒）
// 输出（与旧版 4 元组完全一致）：
//   pred_trace, trace_prob, strike_areas, area_prob
std::tuple<
    std::map<int, std::vector<LocData_pred>>,
    std::vector<double>,
    std::map<int, std::vector<double>>,
    std::vector<double>
>
LSTM_predict(const std::vector<LocData_loc>& locs,
             const std::vector<LocData_route>& routes,
             int instant_ms);
