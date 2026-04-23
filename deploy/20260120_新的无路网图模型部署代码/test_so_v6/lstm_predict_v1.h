// lstm_prediction_v1.h
#pragma once
#include <map>
#include <tuple>
#include <vector>
#include "deploy_3trajs.h"

// 甲方约定接口：输入 18 列轨迹、可选路网、步长（毫秒）；返回四类输出
// 返回： pred_trace, trace_prob, strike_areas, area_prob
std::tuple<
    std::map<int, std::vector<LocData_pred>>,
    std::vector<double>,
    std::map<int, std::vector<double>>,
    std::vector<double>
>
LSTM_predict(const std::vector<LocData_loc>& locs,
             const std::vector<LocData_route>& routes,
             int instant_ms);
