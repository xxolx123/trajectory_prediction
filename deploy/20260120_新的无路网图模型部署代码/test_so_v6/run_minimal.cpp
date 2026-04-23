// run_minimal.cpp
#include "lstm_predict_v1.h"
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>

int main() {
    try {
        // 构造 20 条观测：instant_ms=60000（1 分钟），20 条正好满足“20 分钟缓冲”
        std::vector<LocData_loc> locs;
        locs.reserve(20);
        const int64_t t0 = 1700000000; // 任意基准秒级时间戳
        for (int i = 0; i < 20; ++i) {
            LocData_loc r{};
            r.time = t0 + i * 60;     // 每条相隔 60s
            r.Lon = 94.90 + 0.001 * i;
            r.Lat = 36.40 + 0.001 * i;
            r.Alt = 3000 + i;
            r.Speed_north = 10.0;
            r.Speed_tianxiang = 0.5;
            r.Speed_east = 5.0;
            r.Type = 1; r.Threat = 0;

            // 其它占位字段
            r.source_recog = 0; r.Id_zidan = 1; r.target_number = 111; r.target_accuracy_theo = 0.5;
            r.Id_picture = 123; r.Confidence = 0.8; r.Detectype = 0; r.laser_range = 100; r.static_or_dynam = 2;

            locs.push_back(r);
        }

        // routes 留空即可：
        // - 若 deploy_cfg.ini 中 use_map=true，则 .so 会尝试读取 ./map.png；
        // - 若读取失败或 use_map=false，则内部使用全零图, and use the no map model。
        std::vector<LocData_route> routes;

        // 调用 .so：不传任何配置参数，.so 会自动从当前目录的 deploy_cfg.ini 读取
        int instant_ms = 60000; // 观测步长 1 分钟
        auto ret = LSTM_predict(locs, routes, instant_ms);

        const auto& pred_trace   = std::get<0>(ret);
        const auto& trace_prob   = std::get<1>(ret);
        const auto& strike_areas = std::get<2>(ret); // map<int, vector<double>>
        const auto& area_prob    = std::get<3>(ret);

        if (pred_trace.empty()) {
            std::cout << "buffer not full or no result\n";
            return 0;
        }

        std::cout << "trace_prob:";
        for (double p : trace_prob) std::cout << " " << p;
        std::cout << "\n";

        for (const auto& kv : pred_trace) {
            std::cout << "mode " << kv.first << " points=" << kv.second.size();
            if (!kv.second.empty()) {
                const auto& last = kv.second.back();
                std::cout << "  t10=" << last.time << " lon=" << last.Lon << " lat=" << last.Lat;
            }
            std::cout << "\n";
        }

        std::cout << "strike_areas:\n";
        for (const auto& kv : strike_areas) {
            std::cout << "  mode " << kv.first << " :";
            for (double v : kv.second) std::cout << " " << v;
            std::cout << "\n";
        }

        std::cout << "area_prob:";
        for (double p : area_prob) std::cout << " " << p;
        std::cout << "\n";

        // 导出轨迹数据到 CSV 文件
        std::ofstream traj_file("trajectories_vis.csv");
        if (traj_file.is_open()) {
            traj_file << std::fixed << std::setprecision(6);
            traj_file << "type,mode,time,lon,lat,alt\n";
            
            // 输出历史轨迹
            for (size_t i = 0; i < locs.size(); ++i) {
                traj_file << "history,0," << locs[i].time << ","
                         << locs[i].Lon << "," << locs[i].Lat << "," << locs[i].Alt << "\n";
            }
            
            // 输出预测轨迹（每个 mode）
            for (const auto& kv : pred_trace) {
                int mode = kv.first;
                for (const auto& point : kv.second) {
                    traj_file << "prediction," << mode << "," << point.time << ","
                             << point.Lon << "," << point.Lat << "," << point.Alt << "\n";
                }
            }
            traj_file.close();
            std::cout << "\n[导出] 轨迹数据已保存到: trajectories_vis.csv\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 2;
    }
}