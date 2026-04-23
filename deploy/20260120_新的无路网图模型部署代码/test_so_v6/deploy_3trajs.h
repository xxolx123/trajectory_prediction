// deploy_3trajs.h
#pragma once
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <list>

struct LocData_loc {
    int64_t time;                // 秒级时间戳（整数秒）
    int    source_recog;
    int    Id_zidan;
    int    target_number;
    double target_accuracy_theo;

    double Lon;
    double Lat;
    double Alt;
    double Speed_north;
    double Speed_tianxiang;
    double Speed_east;
    int    Type;
    int    Threat;

    int    Id_picture;
    double Confidence;
    int    Detectype;
    double laser_range;
    int    static_or_dynam;
};

struct LocData_route {
    int w = 0;
    int h = 0;
    int c = 3;
    std::vector<uint8_t> rgb;   // H*W*3
};

using LocData_pred = LocData_loc;

// 便于可读的 AOI（打击区域）结构，库内用 map<int, vector<double>> 输出
struct AOI { double x, y, z, r; };

// 与训练/导出一致的常量（必要时可在 .cpp 中覆盖）
namespace deploy_defaults {
    int constexpr IN_LEN   = 20;
    int constexpr OUT_LEN  = 10;
    int constexpr IN_COLS  = 6;
    int constexpr TOP_M    = 3;
    int constexpr OUTPUT_SIZE = 68;
}

// 封装系统：支持 Feed 分批喂入，Infer 满 20 分钟窗后输出；支持“无路网=全零图”
class TrajSystem {
public:
    TrajSystem();  // 从 deploy_cfg.ini 读取配置；若 routes 为空也会尝试 ./map.png

    // instant_ms: 观测步长（毫秒），用于把缓冲长度折算成20分钟容量
    void Feed(const std::vector<LocData_loc>& data_loc,
              const std::vector<LocData_route>& data_route,
              int instant_ms);

    // 四类输出；满20分钟返回 true；否则 false
    bool Infer(std::map<int, std::vector<LocData_pred>>& pred_trace,
               std::vector<double>& trace_prob,
               std::map<int, std::vector<double>>& strike_areas,
               std::vector<double>& area_prob);

    // 从文件加载路网为 LocData_route（PNG/JPG）
    static bool LoadRouteFromFile(const std::string& path, LocData_route& out);

private:
    class InferenceEngine;
    std::vector<float> load_map_to_nchw_from_route(const LocData_route& R, int hw) const;

private:
    std::list<LocData_loc> buffer_;
    int instant_ms_ = 1000;
    std::vector<float> last_map_nchw_;  // [1,3,H,W]
    InferenceEngine* engine_ = nullptr; // 延迟构造
};
