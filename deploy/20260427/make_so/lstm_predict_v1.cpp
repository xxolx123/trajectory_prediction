// lstm_predict_v1.cpp
// fusion 部署版（20260427）：薄壳，转调 TrajSystem。
// 与旧版完全一致的对外签名，方便上层无感切换。
#include "lstm_predict_v1.h"

#include <iostream>

std::tuple<
    std::map<int, std::vector<LocData_pred>>,
    std::vector<double>,
    std::map<int, std::vector<double>>,
    std::vector<double>
>
LSTM_predict(const std::vector<LocData_loc>& locs,
             const std::vector<LocData_route>& routes,
             int instant_ms)
{
    static TrajSystem sys;

    sys.Feed(locs, routes, instant_ms);

    std::map<int, std::vector<LocData_pred>> pred_trace;
    std::vector<double>                      trace_prob;
    std::map<int, std::vector<double>>       strike_areas;
    std::vector<double>                      area_prob;

    bool ok = sys.Infer(pred_trace, trace_prob, strike_areas, area_prob);
    if (!ok) {
        pred_trace.clear();
        trace_prob.clear();
        strike_areas.clear();
        area_prob.clear();
    }
    return {pred_trace, trace_prob, strike_areas, area_prob};
}
