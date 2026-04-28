// run_minimal.cpp
// fusion 部署版（20260427）：
//   1) 构造 20 条 LLH 观测（每条 60s 间隔）+ 路网折线 + 目标 / ETA
//   2) 调 LSTM_predict 拿四元组（pred_trace / trace_prob / strike_areas / area_prob）
//   3) 打印结果 + 写 trajectories_vis.csv（含 history / prediction / road / target 四类）
//
// 路网模式由 deploy_cfg.ini 的 use_road 字段控制：
//   use_road=true  → 加载 with_road .ms，使用本测试构造的 routes
//   use_road=false → 加载 no_road  .ms，本测试构造的 routes 会被库内忽略
// 切换该开关无需重新编译，本测试源文件保持不变。

#include "lstm_predict_v1.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        // ---- 公共原点（构造数据时图方便用，模型内部以 hist 末帧 LLH 为 ENU 原点）----
        const double base_lon = 116.300;
        const double base_lat = 39.900;
        const double base_alt = 3000.0;   // 米

        // ---- 1) 构造 20 条 LLH 观测：沿东向 / 北向直线推进 ----
        std::vector<LocData_loc> locs;
        locs.reserve(20);
        const int64_t t0 = 1700000000;    // 任意基准秒级时间戳
        for (int i = 0; i < 20; ++i) {
            LocData_loc r{};
            r.time = t0 + i * 60;          // 每条 60s
            r.Lon  = base_lon + 0.0010 * i;
            r.Lat  = base_lat + 0.0005 * i;
            r.Alt  = base_alt + 1.0 * i;   // 米
            // 速度（km/s）；ENU = (East, North, Up)
            r.Speed_east      = 0.10;
            r.Speed_north     = 0.05;
            r.Speed_tianxiang = 0.001;
            r.Type   = 1;
            r.Threat = 0;

            r.source_recog = 0;
            r.Id_zidan     = 1;
            r.target_number = 111;
            r.target_accuracy_theo = 0.5;
            r.Id_picture   = 123;
            r.Confidence   = 0.8;
            r.Detectype    = 0;
            r.laser_range  = 100;
            r.static_or_dynam = 2;

            // ---- fusion 新增字段：每条观测都带；库内只取末帧值 ----
            r.task_type    = 0;            // 0 = 打击
            r.our_type     = 1;            // 我方目标类型
            r.target_lon   = base_lon + 0.030;
            r.target_lat   = base_lat + 0.015;
            r.target_alt_m = base_alt;
            r.eta_sec      = 600;          // 10 分钟

            locs.push_back(r);
        }

        // ---- 2) 构造 1 张路网：1 条 RoadBranch，10 个 LLH 点 ----
        LocData_route route;
        {
            RoadBranch br;
            br.points.reserve(10);
            // 沿历史轨迹方向继续往前推 10 步（约同样步长）
            for (int i = 0; i < 10; ++i) {
                RoadPointLLH p{};
                p.lon_deg = base_lon + 0.0010 * (20 + i);
                p.lat_deg = base_lat + 0.0005 * (20 + i);
                p.alt_m   = base_alt;
                br.points.push_back(p);
            }
            route.branches.push_back(std::move(br));
        }
        std::vector<LocData_route> routes{ route };

        // ---- 3) 调 .so ----
        const int instant_ms = 60000;     // 1 分钟
        auto ret = LSTM_predict(locs, routes, instant_ms);

        const auto& pred_trace   = std::get<0>(ret);
        const auto& trace_prob   = std::get<1>(ret);
        const auto& strike_areas = std::get<2>(ret);
        const auto& area_prob    = std::get<3>(ret);

        if (pred_trace.empty()) {
            std::cout << "buffer not full or no result\n";
            return 0;
        }

        // ---- 4) 打印 ----
        // eta 是 GNN2 必需输入；这里打印末帧的 eta_sec，方便部署侧第一次连接新 .ms
        // 时一眼确认 eta 实际进了模型
        std::cout << "eta_sec used = " << locs.back().eta_sec << " s\n";
        std::cout << "trace_prob:";
        for (double p : trace_prob) std::cout << " " << p;
        std::cout << "\n";

        for (const auto& kv : pred_trace) {
            std::cout << "mode " << kv.first << " points=" << kv.second.size();
            if (!kv.second.empty()) {
                const auto& last = kv.second.back();
                std::cout << "  t10=" << last.time
                          << " lon="  << last.Lon
                          << " lat="  << last.Lat
                          << " alt="  << last.Alt
                          << " intent(Type)=" << last.Type
                          << " threat%(Threat)=" << last.Threat;
            }
            std::cout << "\n";
        }

        std::cout << "strike_areas (lon, lat, z_km, radius_km):\n";
        for (const auto& kv : strike_areas) {
            std::cout << "  mode " << kv.first << " :";
            for (double v : kv.second) std::cout << " " << v;
            std::cout << "\n";
        }

        std::cout << "area_prob:";
        for (double p : area_prob) std::cout << " " << p;
        std::cout << "\n";

        // ---- 5) 导出 trajectories_vis.csv ----
        std::ofstream traj_file("trajectories_vis.csv");
        if (traj_file.is_open()) {
            traj_file << std::fixed << std::setprecision(6);
            traj_file << "type,mode,time,lon,lat,alt\n";

            // 历史轨迹
            for (size_t i = 0; i < locs.size(); ++i) {
                traj_file << "history,0," << locs[i].time << ","
                          << locs[i].Lon  << "," << locs[i].Lat << ","
                          << locs[i].Alt  << "\n";
            }

            // 预测轨迹（每个 mode）
            for (const auto& kv : pred_trace) {
                int mode = kv.first;
                for (const auto& point : kv.second) {
                    traj_file << "prediction," << mode << "," << point.time << ","
                              << point.Lon << "," << point.Lat << ","
                              << point.Alt << "\n";
                }
            }

            // 路网折线（每个 branch 一组；旧版 visualize 会忽略这些行）
            for (size_t bi = 0; bi < route.branches.size(); ++bi) {
                const auto& br = route.branches[bi];
                for (const auto& p : br.points) {
                    traj_file << "road," << bi << ",0,"
                              << p.lon_deg << "," << p.lat_deg << ","
                              << p.alt_m   << "\n";
                }
            }

            // 我方固定目标
            traj_file << "target,0,0,"
                      << locs.back().target_lon << "," << locs.back().target_lat << ","
                      << locs.back().target_alt_m << "\n";

            traj_file.close();
            std::cout << "\n[导出] 轨迹数据已保存到: trajectories_vis.csv\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 2;
    }
}
