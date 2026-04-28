// deploy_3trajs.h
// fusion 部署版（20260427）：与 new_plan/fusion 的双模式 ONNX/.ms 模型对齐。
//   no_road  版：5 输入  hist_traj / task_type / type / position / eta
//   with_road版：7 输入  上面 + road_points / road_mask
//   两版按 deploy_cfg.ini 的 use_road 切换；eta 是 GNN2 必需输入，本部署版假设
//   GNN2 始终启用，.ms 不含 eta 会在引擎构造期报错。
// 主要差异（相对 20260120 旧版）：
//   1) LocData_loc 新增 task_type / our_type / target_lon/lat/alt_m / eta_sec，
//      库内只取 buffer_ 末帧的元信息喂模型；
//   2) LocData_route 重设计为 LLH 折线 (RoadBranch 列表)，部署侧用 hist 末帧
//      LLH 作为局部 ENU 原点做 LLH→km 转换，再按 nb_max/np_max 打包；
//   3) 速度严格按 ENU 映射：vx = Speed_east, vy = Speed_north, vz = Speed_tianxiang。
// 上层 LSTM_predict(locs, routes, instant_ms) 主签名与旧版保持一致；
// lstm2/gnn2 输出位（intent/threat/strike_*）若为 NaN/-1 则统一回退到 0。
//
// 哨兵约定（fusion 元信息字段）：
//   未填的 task_type / our_type / eta_sec 取 -1，未填的 target_lon/lat/alt_m
//   取 NaN；库内据此决定是否走 deploy_cfg.ini 的 default_*。**真值就是 0** 的
//   场景（例如 eta_sec=0 表示"打击敌方当前位置"）必须由调用方主动写 0；
//   旧版"0 视作未填"的脆弱判断已废弃。

#pragma once
#include <cmath>
#include <cstdint>
#include <list>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ----------------------------------------------------------------------------
// 输入：观测点（LLH + 速度 ENU + 目标/任务/ETA 元信息）
// ----------------------------------------------------------------------------
struct LocData_loc {
    int64_t time;                     // 秒级时间戳（整数秒）
    int    source_recog;
    int    Id_zidan;
    int    target_number;
    double target_accuracy_theo;

    double Lon;                        // 我方观测 LLH
    double Lat;
    double Alt;                        // 单位：米
    double Speed_north;                // km/s
    double Speed_tianxiang;            // km/s（天向 = Up）
    double Speed_east;                 // km/s
    int    Type;
    int    Threat;

    int    Id_picture;
    double Confidence;
    int    Detectype;
    double laser_range;
    int    static_or_dynam;

    // ===== fusion 新增字段：每条观测都带，库内取末帧值喂模型 =====
    // 哨兵语义：未填字段保留默认初值（int=-1, double=NaN），cpp 侧据此走
    // deploy_cfg.ini 的 default_*；调用方真值就是 0/0.0 时必须显式赋值。
    int     task_type    = -1;          // 敌方作战任务（0=打击）；-1=未填
    int     our_type     = -1;          // 我方固定目标类型 0/1/2；-1=未填
    double  target_lon   = std::nan(""); // 我方固定目标 LLH；NaN=未填
    double  target_lat   = std::nan("");
    double  target_alt_m = std::nan(""); // 单位：米
    int64_t eta_sec      = -1;          // 我方预计到达时间（秒，合法 [0,600]）；-1=未填
};

// ----------------------------------------------------------------------------
// 输入：路网折线（LLH）
//   - branches.size() 即 NB；branches[i].points.size() 即 NP_i
//   - 库内会按 deploy_cfg.ini 的 nb_max/np_max 截断/补零，并产出 mask
// ----------------------------------------------------------------------------
struct RoadPointLLH {
    double lon_deg;
    double lat_deg;
    double alt_m;                      // 单位：米
};

struct RoadBranch {
    std::vector<RoadPointLLH> points;
};

struct LocData_route {
    std::vector<RoadBranch> branches;  // 一张完整路网
};

// ----------------------------------------------------------------------------
// 输出
// ----------------------------------------------------------------------------
using LocData_pred = LocData_loc;

// 便于可读的 AOI（打击区域）结构，库内仍用 map<int, vector<double>> 输出
struct AOI { double x, y, z, r; };

// ----------------------------------------------------------------------------
// 与训练 / ONNX 导出一致的常量
// ----------------------------------------------------------------------------
namespace deploy_defaults {
    int constexpr IN_LEN     = 20;     // hist_len
    int constexpr OUT_LEN    = 10;     // fut_len
    int constexpr IN_COLS    = 6;      // [x,y,z,vx,vy,vz]
    int constexpr TOP_K      = 3;      // gnn1.top_k；与 new_plan/gnn1/config.yaml 对齐
    int constexpr OUTPUT_DIM = 68;     // [B, K, 68] 中的 68
}

// ----------------------------------------------------------------------------
// 封装系统：缓冲式喂入，满 (IN_LEN-1) * target_spacing_s 跨度后输出 4 元组
//
// 内部选点策略（重要）：
//   - 单帧间隔 target_spacing_s = round(instant_ms / 1000)，**等比例缩放**：
//       instant_ms=60000  → target_spacing_s=60s  ，buffer 需 19min
//       instant_ms=30000  → target_spacing_s=30s  ，buffer 需 ≈10min
//       instant_ms=120000 → target_spacing_s=120s ，buffer 需 38min
//   - Infer() 用 LocData_loc.time 字段驱动；目标网格 = {t_end - (19-i)*target_spacing_s,
//     i=0..19}，对每个 target_time 在 buffer 里找 time 最接近的那条作为 hist[i]。
//   - 未来 10 步预测时间戳 = base_ts + (t+1) * target_spacing_s，也等比缩放。
//   - 模型按 60s 训练；当 target_spacing_s ≠ 60 时，hist 的 delta_pos / vel 会被
//     模型误读为 60s 时间尺度，预测精度会下降，会打 [WARN/Infer] 日志。
//   - 选点的 jitter 阈值 = target_spacing_s / 2，超过会再打 WARN（不阻断推理）。
// ----------------------------------------------------------------------------
class TrajSystem {
public:
    TrajSystem();   // 配置从 deploy_cfg.ini 读取（路径、nb_max/np_max、默认值）

    // instant_ms: 观测步长（毫秒），决定 target_spacing_s = round(instant_ms / 1000)。
    //             buffer 容量上限 / 成熟跨度 / picked 网格 / 未来步长全部按
    //             target_spacing_s 等比例缩放。**模型按 60s 训练，强烈建议 instant_ms=60000**；
    //             其它取值仍可工作，但精度会下降并打 WARN 日志。
    void Feed(const std::vector<LocData_loc>& data_loc,
              const std::vector<LocData_route>& data_route,
              int instant_ms);

    // buffer 时间跨度 ≥ 19 * target_spacing_s 时返回 true，否则 false
    //   pred_trace[m]   ：第 m 条候选未来 10 步（含末步的 intent / threat 写在 Type/Threat 字段）
    //                     未来时间戳 = base_ts + (t+1) * target_spacing_s（t=0..9）
    //   trace_prob      ：mode_prob（gnn1 重归一化，K 条和=1）
    //   strike_areas[m] ：{lon, lat, z_km, radius_km}（若 fusion 输出 NaN 兜底全 0）
    //   area_prob[m]    ：strike_conf ∈ [0, 1]（若 fusion 输出 NaN 兜底 0）
    bool Infer(std::map<int, std::vector<LocData_pred>>& pred_trace,
               std::vector<double>& trace_prob,
               std::map<int, std::vector<double>>& strike_areas,
               std::vector<double>& area_prob);

private:
    class InferenceEngine;

private:
    std::list<LocData_loc> buffer_;
    int                    instant_ms_ = 1000;   // 用来推 target_spacing_s
    LocData_route          last_road_;           // 仅缓存原始 LLH 折线，原点要用推理时的 hist 末帧
    bool                   has_road_ = false;
    InferenceEngine*       engine_   = nullptr;  // 延迟构造
};
