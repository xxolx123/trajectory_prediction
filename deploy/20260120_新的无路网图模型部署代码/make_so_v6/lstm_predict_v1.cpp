// lstm_prediction_v1.cpp
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
    // 内部构建 TrajSystem：读取 deploy_cfg.ini，按 use_map 决定模型与图像尺寸
    static TrajSystem sys;

    // 喂入（可多次调用；此处一次性喂入本批数据）
    sys.Feed(locs, routes, instant_ms);

    // 推理
    std::map<int, std::vector<LocData_pred>> pred_trace;
    std::vector<double> trace_prob;
    std::map<int, std::vector<double>> strike_areas;
    std::vector<double> area_prob;

    bool ok = sys.Infer(pred_trace, trace_prob, strike_areas, area_prob);
    if (!ok) {
        // 缓冲未满：返回空结果，方便上层做“未就绪”判断
        pred_trace.clear();
        trace_prob.clear();
        strike_areas.clear();
        area_prob.clear();
    }
    return {pred_trace, trace_prob, strike_areas, area_prob};
}