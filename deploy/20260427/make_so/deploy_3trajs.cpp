// deploy_3trajs.cpp
// fusion 部署版（20260427）：与 new_plan/fusion 的双模式 ONNX/.ms 模型对齐。
// 上层调用 LSTM_predict(locs, routes, instant_ms) 主签名不变。
//
// 主要差异（相对 20260120 旧版）：
//   1) LocData_route 从 PNG 图像 改为 LLH 折线，部署侧用 hist 末帧 LLH
//      作为局部 ENU 原点做 LLH→km 转换，再按 nb_max/np_max 打包 + mask；
//   2) fusion 模型改为按 ONNX **输入名字**动态绑定 MSTensor，仅支持两种 .ms：
//        no_road  版：5 输入  hist_traj / task_type / type / position / eta
//        with_road版：7 输入  上面 + road_points / road_mask
//      eta 是 GNN2 必需输入，本部署版假设 GNN2 始终启用；.ms 不含 eta 则在
//      引擎构造期直接抛异常，不再静默回退到"无 eta"分支。
//      具体加载哪份由 deploy_cfg.ini 的 use_road 开关决定；上层传来的 routes
//      在 use_road=false 时会被忽略；
//   3) 速度严格按 ENU 映射：vx=Speed_east, vy=Speed_north, vz=Speed_tianxiang，
//      与 fusion 训练数据轴序一致；
//   4) lstm2/gnn2 输出位（intent_class / threat_prob / strike_*）若为 NaN/-1，
//      回退为 0，避免下游拿到非数。

#include "deploy_3trajs.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>   // readlink
#include <limits.h>   // PATH_MAX
#include <dlfcn.h>    // dladdr

// MindSpore Lite
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"

// ============================================================================
// 与 new_plan/fusion 对齐的常量
// ============================================================================
static const int IN_LEN     = deploy_defaults::IN_LEN;       // 20
static const int OUT_LEN    = deploy_defaults::OUT_LEN;      // 10
static const int IN_COLS    = deploy_defaults::IN_COLS;      // 6
static const int TOP_K      = deploy_defaults::TOP_K;        // 3
static const int OUTPUT_DIM = deploy_defaults::OUTPUT_DIM;   // 68

// 与 ONNX 导出 input_names 严格一致（见 fusion/code/export_onnx.py）
static const char* IN_NAME_HIST   = "hist_traj";
static const char* IN_NAME_TASK   = "task_type";
static const char* IN_NAME_TYPE   = "type";
static const char* IN_NAME_POS    = "position";
static const char* IN_NAME_RPTS   = "road_points";
static const char* IN_NAME_RMASK  = "road_mask";
static const char* IN_NAME_ETA    = "eta";

// flat-earth LLH↔ENU 用到的地球半径（与 road_schema.py 完全一致）
static const double EARTH_R_KM = 6371.0;

// ============================================================================
// 配置文件 (deploy_cfg.ini)
// ============================================================================
struct DeployCfg {
    // ---- 路网模式开关 ----
    // true  → 加载 with_road .ms（上层传的 routes 会被使用）
    // false → 加载 no_road  .ms（routes 被忽略）
    bool        use_road              = true;

    // 双模式 .ms 路径；任一未在 cfg 中显式给出（保持空）则回退到 model_path
    // 这样旧版只配 model_path= 的 deploy_cfg.ini 仍能向后兼容
    std::string model_path_with_road  = "";
    std::string model_path_no_road    = "";

    // 旧版兼容兜底；空表示用上面两个新键
    std::string model_path            = "";

    int         nb_max                = 4;
    int         np_max                = 128;

    int     default_task_type    = 0;
    int     default_our_type     = 0;
    double  default_target_lon   = 116.30;
    double  default_target_lat   = 39.90;
    double  default_target_alt_m = 0.0;
    int64_t default_eta_sec      = 0;
};

static inline std::string trim(const std::string& s) {
    size_t i = 0, j = s.size();
    while (i < j && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
    return s.substr(i, j - i);
}

static inline bool ieq(const std::string& a, const char* b) {
    if (a.size() != std::strlen(b)) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i]))
            != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    }
    return true;
}

static inline bool parse_bool(const std::string& v) {
    return ieq(v, "true") || ieq(v, "1") || ieq(v, "yes") || ieq(v, "on");
}

// 选最终 .ms 路径：优先用对应 use_road 的新键；新键为空则回退 model_path
static const std::string& select_model_path(const DeployCfg& c) {
    if (c.use_road) {
        return !c.model_path_with_road.empty() ? c.model_path_with_road : c.model_path;
    } else {
        return !c.model_path_no_road.empty()   ? c.model_path_no_road   : c.model_path;
    }
}

static DeployCfg load_cfg_file(const std::string& path) {
    DeployCfg c;
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[WARN] cannot open " << path << ", use defaults.\n";
        return c;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        auto pos_hash = line.find('#');
        if (pos_hash != std::string::npos) line = line.substr(0, pos_hash);
        line = trim(line);
        if (line.empty()) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string k = trim(line.substr(0, pos));
        std::string v = trim(line.substr(pos + 1));

        if      (ieq(k, "use_road"))              c.use_road              = parse_bool(v);
        else if (ieq(k, "model_path_with_road"))  c.model_path_with_road  = v;
        else if (ieq(k, "model_path_no_road"))    c.model_path_no_road    = v;
        else if (ieq(k, "model_path"))            c.model_path            = v;
        else if (ieq(k, "nb_max"))                c.nb_max = std::max(1, std::atoi(v.c_str()));
        else if (ieq(k, "np_max"))                c.np_max = std::max(1, std::atoi(v.c_str()));
        else if (ieq(k, "default_task_type"))     c.default_task_type    = std::atoi(v.c_str());
        else if (ieq(k, "default_our_type"))      c.default_our_type     = std::atoi(v.c_str());
        else if (ieq(k, "default_target_lon"))    c.default_target_lon   = std::atof(v.c_str());
        else if (ieq(k, "default_target_lat"))    c.default_target_lat   = std::atof(v.c_str());
        else if (ieq(k, "default_target_alt_m"))  c.default_target_alt_m = std::atof(v.c_str());
        else if (ieq(k, "default_eta_sec"))       c.default_eta_sec      = std::atoll(v.c_str());
    }
    return c;
}

// 配置查找顺序：DEPLOY_CFG 环境变量 → ./deploy_cfg.ini → .so 同目录 → ./deploy_cfg.ini（兜底）
static std::string get_cfg_path() {
    const char* envp = std::getenv("DEPLOY_CFG");
    if (envp && *envp) {
        std::ifstream t(envp);
        if (t.good()) return std::string(envp);
        std::cerr << "[WARN] DEPLOY_CFG given but cannot open: " << envp << "\n";
    }
    {
        std::ifstream t("./deploy_cfg.ini");
        if (t.good()) return std::string("./deploy_cfg.ini");
    }
    std::string so_dir = ".";
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&get_cfg_path), &info) && info.dli_fname) {
        std::string so_path = info.dli_fname;
        auto pos = so_path.find_last_of('/');
        if (pos != std::string::npos) so_dir = so_path.substr(0, pos);
    } else {
        char buf[PATH_MAX]{0};
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            std::string exep(buf);
            auto pos = exep.find_last_of('/');
            if (pos != std::string::npos) so_dir = exep.substr(0, pos);
        }
    }
    std::string cand = so_dir + "/deploy_cfg.ini";
    std::ifstream t(cand);
    if (t.good()) return cand;
    return "./deploy_cfg.ini";
}

static bool      g_cfg_loaded = false;
static DeployCfg g_cfg;

static void ensure_cfg_loaded_once() {
    if (!g_cfg_loaded) {
        std::string path = get_cfg_path();
        std::cerr << "[CFG] using: " << path << "\n";
        g_cfg = load_cfg_file(path);
        g_cfg_loaded = true;
    }
}

// ============================================================================
// LLH ↔ ENU(km)（flat-earth，公式与 new_plan/.../road_schema.py 完全一致）
// ============================================================================
static inline void llh_to_enu_km(double lon_deg,  double lat_deg,  double alt_m,
                                 double lon0_deg, double lat0_deg, double alt0_m,
                                 double& x_km, double& y_km, double& z_km) {
    constexpr double DEG2RAD = M_PI / 180.0;
    const double lat0_rad = lat0_deg * DEG2RAD;
    const double dy = (lat_deg - lat0_deg) * DEG2RAD * EARTH_R_KM;
    const double dx = (lon_deg - lon0_deg) * DEG2RAD * EARTH_R_KM * std::cos(lat0_rad);
    const double dz = (alt_m   - alt0_m) / 1000.0;
    x_km = dx;
    y_km = dy;
    z_km = dz;
}

static inline void enu_km_to_llh(double x_km, double y_km, double z_km,
                                 double lon0_deg, double lat0_deg, double alt0_m,
                                 double& lon_deg, double& lat_deg, double& alt_m) {
    constexpr double RAD2DEG = 180.0 / M_PI;
    constexpr double DEG2RAD = M_PI / 180.0;
    const double lat0_rad = lat0_deg * DEG2RAD;
    lat_deg = lat0_deg + (y_km / EARTH_R_KM) * RAD2DEG;
    double cos_lat0 = std::cos(lat0_rad);
    if (std::abs(cos_lat0) < 1e-12) cos_lat0 = (cos_lat0 < 0 ? -1e-12 : 1e-12);
    lon_deg = lon0_deg + (x_km / (EARTH_R_KM * cos_lat0)) * RAD2DEG;
    alt_m   = alt0_m + z_km * 1000.0;
}

// ----------------------------------------------------------------------------
// LocData_route(LLH) → road_points / road_mask 张量
//   road_points: [1, NB, NP, 3]  float32   ENU km
//   road_mask  : [1, NB, NP]     bool      True=有效
// 超界（branch 数 / 每条点数）截断；不足处全 0 / mask=false
// ----------------------------------------------------------------------------
static void pack_road_to_tensors(const LocData_route& road,
                                 double origin_lon, double origin_lat, double origin_alt_m,
                                 int nb_max, int np_max,
                                 std::vector<float>& rp_buf,    // [NB*NP*3]
                                 std::vector<uint8_t>& rm_buf)  // [NB*NP]
{
    const size_t cap = static_cast<size_t>(nb_max) * np_max;
    rp_buf.assign(cap * 3, 0.0f);
    rm_buf.assign(cap, 0u);

    const int NB = std::min<int>(nb_max, static_cast<int>(road.branches.size()));
    for (int bi = 0; bi < NB; ++bi) {
        const auto& br = road.branches[bi];
        const int NP = std::min<int>(np_max, static_cast<int>(br.points.size()));
        for (int pi = 0; pi < NP; ++pi) {
            const auto& p = br.points[pi];
            double x, y, z;
            llh_to_enu_km(p.lon_deg, p.lat_deg, p.alt_m,
                          origin_lon, origin_lat, origin_alt_m,
                          x, y, z);
            const size_t flat = (static_cast<size_t>(bi) * np_max + pi);
            rp_buf[flat * 3 + 0] = static_cast<float>(x);
            rp_buf[flat * 3 + 1] = static_cast<float>(y);
            rp_buf[flat * 3 + 2] = static_cast<float>(z);
            rm_buf[flat] = 1u;
        }
    }
}

// ============================================================================
// MindSpore Lite 推理引擎：按输入名字动态绑定 MSTensor
//   必需输入（5 路）：hist_traj / task_type / type / position / eta
//   可选输入：road_points + road_mask（成对，with_road 版才有；no_road 版没有）
//
// GNN2 已经常驻流水线，eta 是必需输入；.ms 缺 eta 直接抛异常，不再静默回退。
// has_road_inputs_ 仍按 ONNX 实际清单运行时探测，对应 use_road 双 .ms 切换。
// ============================================================================
class TrajSystem::InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, int nb_max, int np_max)
        : nb_max_(nb_max), np_max_(np_max)
    {
        auto ctx = std::make_shared<mindspore::Context>();
        auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
        ctx->MutableDeviceInfo().push_back(cpu);
        auto ret = model_.Build(model_path, mindspore::kMindIR, ctx);
        if (ret != mindspore::kSuccess) {
            throw std::runtime_error("MindSpore Build failed: " + model_path);
        }

        // 按 ONNX 输入名建立 name -> input_index 映射
        auto inputs = model_.GetInputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            name2idx_[inputs[i].Name()] = static_cast<int>(i);
        }

        // 必需输入（含 eta）：缺一个就报错，同时把 .ms 实际清单打出来辅助排查
        for (const char* n : {IN_NAME_HIST, IN_NAME_TASK, IN_NAME_TYPE, IN_NAME_POS,
                               IN_NAME_ETA}) {
            if (name2idx_.find(n) == name2idx_.end()) {
                std::ostringstream oss;
                oss << "Cannot find required input tensor by name: " << n
                    << ". Available inputs:";
                for (const auto& kv : name2idx_) oss << " '" << kv.first << "'";
                oss << ". 本部署版要求 GNN2 已开启 (.ms 必含 eta)，"
                       "请用 fusion/code/export_onnx.py（gnn2.enable=true）重新导出。";
                throw std::runtime_error(oss.str());
            }
        }

        // 可选输入：road_points / road_mask 必须成对，缺其一报错
        const bool has_rpts  = name2idx_.count(IN_NAME_RPTS)  > 0;
        const bool has_rmask = name2idx_.count(IN_NAME_RMASK) > 0;
        if (has_rpts != has_rmask) {
            throw std::runtime_error(
                "ONNX 输入异常：road_points 与 road_mask 必须成对出现，"
                "但当前 .ms 只含其中一个");
        }
        has_road_inputs_ = has_rpts && has_rmask;

        std::cerr << "[Engine] model=" << model_path
                  << "  inputs=" << inputs.size()
                  << "  has_road=" << (has_road_inputs_ ? "true" : "false")
                  << "\n";

        // Resize 一次：batch=1
        std::vector<std::vector<int64_t>> new_shapes(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            new_shapes[i] = inputs[i].Shape();   // 默认沿用模型自带的形状
        }
        new_shapes[name2idx_[IN_NAME_HIST]] = {1, IN_LEN, IN_COLS};
        new_shapes[name2idx_[IN_NAME_TASK]] = {1};
        new_shapes[name2idx_[IN_NAME_TYPE]] = {1};
        new_shapes[name2idx_[IN_NAME_POS]]  = {1, 3};
        new_shapes[name2idx_[IN_NAME_ETA]]  = {1};
        if (has_road_inputs_) {
            new_shapes[name2idx_[IN_NAME_RPTS]]  = {1, nb_max_, np_max_, 3};
            new_shapes[name2idx_[IN_NAME_RMASK]] = {1, nb_max_, np_max_};
        }

        if (model_.Resize(inputs, new_shapes) != mindspore::kSuccess) {
            throw std::runtime_error("MindSpore Resize failed");
        }
    }

    // hist / task_type / type / position / eta 必传；road_points / road_mask 仅
    // has_road_inputs_=true（with_road .ms）时使用，no_road 下传空 vector 即可。
    // output 至少有 1 个，且第一个为 [1,K,68] float32。
    std::vector<float> forward(const std::vector<float>&    hist,         // [1*20*6]
                               const std::vector<int64_t>&  task_type,    // [1]
                               const std::vector<int64_t>&  type_id,      // [1]
                               const std::vector<float>&    position,     // [1*3]
                               const std::vector<float>&    road_points,  // [1*NB*NP*3]，no_road 时可空
                               const std::vector<uint8_t>&  road_mask,    // [1*NB*NP]，no_road 时可空
                               const std::vector<int64_t>&  eta)          // [1]，必传
    {
        auto inputs = model_.GetInputs();

        auto cpy_f32 = [&](const char* name, const std::vector<float>& src) {
            auto& ts = inputs[name2idx_[name]];
            const size_t need = src.size() * sizeof(float);
            if (ts.DataSize() < need) {
                throw std::runtime_error(std::string("Input ") + name + " too small");
            }
            std::memcpy(ts.MutableData(), src.data(), need);
        };
        auto cpy_i64 = [&](const char* name, const std::vector<int64_t>& src) {
            auto& ts = inputs[name2idx_[name]];
            const size_t need = src.size() * sizeof(int64_t);
            if (ts.DataSize() < need) {
                throw std::runtime_error(std::string("Input ") + name + " too small");
            }
            std::memcpy(ts.MutableData(), src.data(), need);
        };
        auto cpy_bool = [&](const char* name, const std::vector<uint8_t>& src) {
            // ONNX bool → MindSpore Lite kBool；底层每元素 1 字节
            auto& ts = inputs[name2idx_[name]];
            const size_t need = src.size() * sizeof(uint8_t);
            if (ts.DataSize() < need) {
                throw std::runtime_error(std::string("Input ") + name + " too small");
            }
            std::memcpy(ts.MutableData(), src.data(), need);
        };

        cpy_f32 (IN_NAME_HIST,  hist);
        cpy_i64 (IN_NAME_TASK,  task_type);
        cpy_i64 (IN_NAME_TYPE,  type_id);
        cpy_f32 (IN_NAME_POS,   position);
        cpy_i64 (IN_NAME_ETA,   eta);
        if (has_road_inputs_) {
            cpy_f32 (IN_NAME_RPTS,  road_points);
            cpy_bool(IN_NAME_RMASK, road_mask);
        }

        std::vector<mindspore::MSTensor> outputs;
        if (model_.Predict(inputs, &outputs) != mindspore::kSuccess || outputs.empty()) {
            throw std::runtime_error("Predict failed");
        }
        const auto& out = outputs[0];
        const size_t n = out.DataSize() / sizeof(float);
        const float* p = static_cast<const float*>(out.Data().get());
        if (!p || n == 0) throw std::runtime_error("Invalid output");
        return std::vector<float>(p, p + n);
    }

    int  nb_max()           const { return nb_max_; }
    int  np_max()           const { return np_max_; }
    bool has_road_inputs()  const { return has_road_inputs_; }

private:
    mindspore::Model                       model_;
    std::unordered_map<std::string, int>   name2idx_;
    int                                    nb_max_           = 4;
    int                                    np_max_           = 128;
    bool                                   has_road_inputs_  = false;
};

// ============================================================================
// 小工具
// ============================================================================
static inline bool finite_f(float v) {
    return std::isfinite(static_cast<double>(v));
}

// ============================================================================
// TrajSystem 实现
// ============================================================================
TrajSystem::TrajSystem() {
    // engine_ 延迟构造：首次 Feed 时按 select_model_path(cfg) / nb_max / np_max
    // 选择 with_road / no_road .ms 加载，并校验与 use_road 是否匹配
}

void TrajSystem::Feed(const std::vector<LocData_loc>& data_loc,
                      const std::vector<LocData_route>& data_route,
                      int instant_ms) {
    ensure_cfg_loaded_once();
    // instant_ms_ → target_spacing_s = round(instant_ms_ / 1000)，等比例缩放：
    //   buffer 容量上限 / Infer 的成熟跨度 / picked 网格 / 未来步长全部按它放缩。
    //   模型按 60s 步长训练，instant_ms=60000 时各项语义与训练一致；其它值 Infer
    //   会打 [WARN/Infer]，预测精度可能下降。
    instant_ms_ = std::max(1, instant_ms);

    if (!engine_) {
        const std::string& mp = select_model_path(g_cfg);
        if (mp.empty()) {
            throw std::runtime_error(
                "No model path configured: please set model_path_with_road / "
                "model_path_no_road (or legacy model_path) in deploy_cfg.ini");
        }
        engine_ = new InferenceEngine(mp, g_cfg.nb_max, g_cfg.np_max);

        // use_road 与 .ms 实际输入一致性校验
        if (g_cfg.use_road && !engine_->has_road_inputs()) {
            throw std::runtime_error(
                "use_road=true 但 .ms 没有 road_points/road_mask 输入；"
                "请检查 model_path_with_road 是否真的指向 with_road 版 .ms");
        }
        if (!g_cfg.use_road && engine_->has_road_inputs()) {
            std::cerr << "[WARN] use_road=false 但 .ms 含 road 输入，"
                         "将传入 mask 全 false 走 fallback；"
                         "建议改用 no_road 版以减小模型 / 缩短推理\n";
        }
    }

    // 追加观测并按 time 排序
    for (const auto& r : data_loc) buffer_.push_back(r);
    buffer_.sort([](const LocData_loc& a, const LocData_loc& b) {
        return a.time < b.time;
    });

    // 时间跨度上限 = (IN_LEN + 5) * 单帧间隔；多缓存 6 帧用来吃 jitter / 延迟。
    //   instant_ms=60s  → 上限 25min（与旧版 20min 容量+5min 余量等效）
    //   instant_ms=30s  → 上限 12.5min（推理窗口 9.5min + 3min 余量）
    //   instant_ms=120s → 上限 50min（推理窗口 38min + 12min 余量）
    // 硬上限 100k 条防内存爆炸（极快采样 + 超长跨度时兜底）。
    const int64_t target_spacing_s_for_cap = std::max<int64_t>(1, std::llround(
        static_cast<double>(instant_ms_) / 1000.0));
    const int64_t MAX_BUFFER_SPAN_S =
        static_cast<int64_t>(IN_LEN + 5) * target_spacing_s_for_cap;
    constexpr size_t  MAX_BUFFER_COUNT  = 100000;
    while (!buffer_.empty() &&
           (buffer_.back().time - buffer_.front().time) > MAX_BUFFER_SPAN_S) {
        buffer_.pop_front();
    }
    while (buffer_.size() > MAX_BUFFER_COUNT) {
        buffer_.pop_front();
    }

    // 缓存最新一张路网（保留 LLH 原始结构；坐标变换要等到 Infer 时才能定原点）
    // use_road=false 时直接丢弃，省一次 ENU 转换 + 内存占用
    if (g_cfg.use_road && !data_route.empty()) {
        last_road_ = data_route.back();
        has_road_ = !last_road_.branches.empty();
    }
}

bool TrajSystem::Infer(std::map<int, std::vector<LocData_pred>>& pred_trace,
                       std::vector<double>& trace_prob,
                       std::map<int, std::vector<double>>& strike_areas,
                       std::vector<double>& area_prob) {
    if (!engine_) return false;
    if (buffer_.empty()) return false;

    // ===== 单帧间隔 = instant_ms / 1000 =====
    // target 网格随上层 instant_ms 等比例缩放：
    //   instant_ms=60s  → target_spacing_s=60，buffer 需 19*60=1140s≈19min
    //   instant_ms=30s  → target_spacing_s=30，buffer 需 19*30=570s≈10min
    //   instant_ms=120s → target_spacing_s=120，buffer 需 19*120=2280s=38min
    // 注意：模型按 60s 步长训练；target_spacing_s ≠ 60 时，模型把"30s/120s 的
    // delta_pos / vel"误读成 60s，预测精度会下降，会打 [WARN/Infer] 日志。
    const int64_t target_spacing_s = std::max<int64_t>(1, std::llround(
        static_cast<double>(instant_ms_) / 1000.0));
    if (target_spacing_s != 60) {
        std::cerr << "[WARN/Infer] target_spacing_s = " << target_spacing_s
                  << "s ≠ 60s（模型按 60s 训练），buffer 按 "
                  << (IN_LEN - 1) * target_spacing_s
                  << "s 跨度成熟，但 hist 的 delta/vel 尺度会被模型误读，"
                     "预测精度可能下降\n";
    }

    // ===== 成熟判定：buffer 真实时间跨度 >= 19 * target_spacing_s =====
    // 直接看时间戳跨度，对 jitter / 丢帧 / instant_ms 估错都鲁棒。
    // 模型要 20 帧，最早一帧对应 t_end - 19*target_spacing_s。
    const int64_t SPAN_S = static_cast<int64_t>(IN_LEN - 1) * target_spacing_s;
    const int64_t t_end   = buffer_.back().time;
    const int64_t t_start = buffer_.front().time;
    if (t_end - t_start < SPAN_S) return false;

    // ===== 选点：按 time 找最近邻，目标网格 = t_end - (19-i)*target_spacing_s =====
    // i=0..19 对应 hist 的 0~19 个等距锚点（间距 = target_spacing_s）；
    // 上层观测周期 = target_spacing_s 时，picked[i] 几乎与第 i 帧观测重合；
    // 周期与 target_spacing_s 不一致时，picked[i] = buffer 里 time 最接近的观测。
    //
    // 因为 buffer_ 已升序、target_time[i] 也升序，用单调推进的 cursor 即可，
    // 不需要二分；总扫描成本 O(buffer_size + IN_LEN)。
    std::vector<LocData_loc> picked;
    picked.reserve(IN_LEN);
    auto cursor = buffer_.begin();
    int  jitter_warn = 0;
    int64_t worst_jitter_s = 0;
    const int64_t jitter_threshold_s = std::max<int64_t>(1, target_spacing_s / 2);

    for (int i = 0; i < IN_LEN; ++i) {
        const int64_t t_target = t_end -
            static_cast<int64_t>(IN_LEN - 1 - i) * target_spacing_s;

        // 把 cursor 推到 "time <= t_target" 的最右一条
        while (std::next(cursor) != buffer_.end() &&
               std::next(cursor)->time <= t_target) {
            ++cursor;
        }
        // 比较 cursor 与 cursor+1，取离 t_target 更近的一条作为 picked[i]
        auto best = cursor;
        int64_t best_diff = std::llabs(cursor->time - t_target);
        if (std::next(cursor) != buffer_.end()) {
            const int64_t d2 = std::llabs(std::next(cursor)->time - t_target);
            if (d2 < best_diff) { best = std::next(cursor); best_diff = d2; }
        }
        picked.push_back(*best);

        // 记录 jitter；超过单帧间隔的一半就累计计数
        if (best_diff > jitter_threshold_s) ++jitter_warn;
        if (best_diff > worst_jitter_s) worst_jitter_s = best_diff;
    }

    if (jitter_warn > 0) {
        std::cerr << "[WARN/Infer] " << jitter_warn << "/" << IN_LEN
                  << " 帧最近邻偏差 > " << jitter_threshold_s
                  << "s（最大 " << worst_jitter_s << "s），模型按 "
                  << target_spacing_s << "s 步长解读 hist，精度可能下降；"
                     "请检查上层观测周期是否稳定 / 是否有丢帧\n";
    }

    // ENU 原点 = hist 末帧 LLH（与 fusion README 一致）
    const LocData_loc& last = picked.back();
    const double origin_lon   = last.Lon;
    const double origin_lat   = last.Lat;
    const double origin_alt_m = last.Alt;

    // ----------------- 1) hist_traj [1, 20, 6] -----------------
    std::vector<float> hist_buf(1 * IN_LEN * IN_COLS, 0.0f);
    for (int t = 0; t < IN_LEN; ++t) {
        double x, y, z;
        llh_to_enu_km(picked[t].Lon, picked[t].Lat, picked[t].Alt,
                      origin_lon, origin_lat, origin_alt_m,
                      x, y, z);
        // ENU 严格速度轴序（修复旧 deploy 的不一致）
        const float vx = static_cast<float>(picked[t].Speed_east);       // East
        const float vy = static_cast<float>(picked[t].Speed_north);      // North
        const float vz = static_cast<float>(picked[t].Speed_tianxiang);  // Up
        hist_buf[t * IN_COLS + 0] = static_cast<float>(x);
        hist_buf[t * IN_COLS + 1] = static_cast<float>(y);
        hist_buf[t * IN_COLS + 2] = static_cast<float>(z);
        hist_buf[t * IN_COLS + 3] = vx;
        hist_buf[t * IN_COLS + 4] = vy;
        hist_buf[t * IN_COLS + 5] = vz;
    }

    // ----------------- 2) ctx 元信息（取末帧；为 0 → cfg 兜底） -----------------
    auto pick_int = [](int v, int def) { return v != 0 ? v : def; };

    const int     task_type = pick_int(last.task_type, g_cfg.default_task_type);
    const int     our_type  = pick_int(last.our_type,  g_cfg.default_our_type);
    const int64_t eta_sec   = (last.eta_sec != 0)
                              ? last.eta_sec : g_cfg.default_eta_sec;

    const bool has_target_llh =
        (last.target_lon != 0.0 || last.target_lat != 0.0 || last.target_alt_m != 0.0);
    const double tgt_lon   = has_target_llh ? last.target_lon   : g_cfg.default_target_lon;
    const double tgt_lat   = has_target_llh ? last.target_lat   : g_cfg.default_target_lat;
    const double tgt_alt_m = has_target_llh ? last.target_alt_m : g_cfg.default_target_alt_m;

    std::vector<int64_t> task_buf{ static_cast<int64_t>(task_type) };
    std::vector<int64_t> type_buf{ static_cast<int64_t>(our_type) };

    // eta 是 GNN2 必需输入（.ms 不含 eta 已在 InferenceEngine 构造期报错）
    std::vector<int64_t> eta_buf{ static_cast<int64_t>(eta_sec) };

    std::vector<float> pos_buf(3, 0.0f);
    {
        double x, y, z;
        llh_to_enu_km(tgt_lon, tgt_lat, tgt_alt_m,
                      origin_lon, origin_lat, origin_alt_m,
                      x, y, z);
        pos_buf[0] = static_cast<float>(x);
        pos_buf[1] = static_cast<float>(y);
        pos_buf[2] = static_cast<float>(z);
    }

    // ----------------- 3) road_points / road_mask -----------------
    // 仅在引擎确实含 road 输入时才打包；no_road .ms 下全跳过
    std::vector<float>   rp_buf;
    std::vector<uint8_t> rm_buf;
    if (engine_->has_road_inputs()) {
        const size_t cap = static_cast<size_t>(engine_->nb_max()) * engine_->np_max();
        if (g_cfg.use_road && has_road_) {
            pack_road_to_tensors(last_road_,
                                 origin_lon, origin_lat, origin_alt_m,
                                 engine_->nb_max(), engine_->np_max(),
                                 rp_buf, rm_buf);
        } else {
            // 无路网或 use_road=false：全 0 + mask=false
            // ConstraintOptimizer 向量化版用 torch.where(branch_has_seg, ...) 走 fallback，
            // 等价于不做路网投影
            rp_buf.assign(cap * 3, 0.0f);
            rm_buf.assign(cap, 0u);
        }
    }

    // ----------------- 4) forward -----------------
    std::vector<float> y;
    try {
        y = engine_->forward(hist_buf, task_buf, type_buf, pos_buf,
                             rp_buf,   rm_buf,   eta_buf);
    } catch (const std::exception& e) {
        std::cerr << "[ERR] forward failed: " << e.what() << "\n";
        return false;
    }

    // 期望 size = 1 * K * 68 = 204
    if (y.size() < static_cast<size_t>(TOP_K) * OUTPUT_DIM) {
        throw std::runtime_error("Unexpected output size: "
                                 + std::to_string(y.size()));
    }

    // ----------------- 5) 解码 [K, 68] -----------------
    std::vector<std::array<float, deploy_defaults::OUTPUT_DIM>> modes(TOP_K);
    for (int m = 0; m < TOP_K; ++m) {
        std::memcpy(modes[m].data(),
                    y.data() + m * OUTPUT_DIM,
                    sizeof(float) * OUTPUT_DIM);
    }

    pred_trace.clear();
    strike_areas.clear();
    trace_prob.assign(TOP_K, 0.0);
    area_prob.assign(TOP_K, 0.0);

    const int64_t base_ts = picked.back().time;
    double prob_sum = 0.0;

    for (int m = 0; m < TOP_K; ++m) {
        const auto& v = modes[m];

        std::vector<LocData_pred> rows;
        rows.reserve(OUT_LEN);

        // 每步：ENU km → LLH（用同一 origin）
        // 注意：fusion 输出的 0..59 是路网约束后的 refined 轨迹；按 fut_len=10 步铺开
        for (int t = 0; t < OUT_LEN; ++t) {
            const int off = t * 6;
            double lon, lat, alt_m;
            enu_km_to_llh(v[off + 0], v[off + 1], v[off + 2],
                          origin_lon, origin_lat, origin_alt_m,
                          lon, lat, alt_m);
            LocData_pred r = picked.back();
            // 未来 10 步以 target_spacing_s 等距向前推。
            //   target_spacing_s=60: r.time = base + 60, 120, ..., 540, 600
            //                        与旧版"前 9 步 +60s + 第 10 步 +600s"完全一致
            //                        （旧逻辑里 (9+1)*60=600,特例分支等价于无操作）
            //   target_spacing_s=30: r.time = base + 30, 60, ..., 270, 300（10 帧 = 5min）
            //   target_spacing_s=120:r.time = base + 120, 240, ..., 1080, 1200（10 帧 = 20min）
            r.time = base_ts + static_cast<int64_t>(t + 1) * target_spacing_s;
            r.Lon  = lon;
            r.Lat  = lat;
            r.Alt  = alt_m;

            // 速度（ENU）反向写回 LocData：vx→east, vy→north, vz→tianxiang
            r.Speed_east      = v[off + 3];
            r.Speed_north     = v[off + 4];
            r.Speed_tianxiang = v[off + 5];

            // 末步附带 intent / threat（哨兵值回退到 0）
            if (t == OUT_LEN - 1) {
                const float intent_raw = v[60];
                const float threat_raw = v[61];

                int intent_class = 0;
                if (finite_f(intent_raw) && intent_raw >= 0.0f) {
                    intent_class = static_cast<int>(std::llround(intent_raw));
                }
                int threat_pct = 0;
                if (finite_f(threat_raw)) {
                    float clamped = std::max(0.0f, std::min(1.0f, threat_raw));
                    threat_pct = static_cast<int>(std::llround(clamped * 100.0f));
                }
                r.Type   = intent_class;
                r.Threat = threat_pct;
            }
            rows.push_back(r);
        }

        // 打击区域：strike_pos (62..64) ENU km → LLH；radius (65) km；conf (66) 概率
        std::vector<double> aoi(4, 0.0);
        const float sp_x = v[62], sp_y = v[63], sp_z = v[64];
        const float radius = v[65];
        const float conf   = v[66];
        if (finite_f(sp_x) && finite_f(sp_y) && finite_f(sp_z)) {
            double lon, lat, alt_m;
            enu_km_to_llh(sp_x, sp_y, sp_z,
                          origin_lon, origin_lat, origin_alt_m,
                          lon, lat, alt_m);
            aoi[0] = lon;
            aoi[1] = lat;
            aoi[2] = static_cast<double>(sp_z);          // z 仍用 km，便于上层
            aoi[3] = finite_f(radius) ? static_cast<double>(radius) : 0.0;
        } else {
            // GNN2 现已常驻流水线，正常推理不会到这分支；仅用于兜底数值异常
            // （若 fusion 仍出现 NaN 哨兵，等价于上层"无打击区域"语义）
            aoi.assign(4, 0.0);
        }
        strike_areas[m] = std::move(aoi);

        area_prob[m] = finite_f(conf) ? static_cast<double>(conf) : 0.0;

        // mode_prob (67) 始终有效
        const float mp = v[67];
        trace_prob[m] = finite_f(mp) ? static_cast<double>(mp) : 0.0;
        prob_sum += trace_prob[m];

        pred_trace[m] = std::move(rows);
    }

    // mode_prob 重归一化保险（GNN1 已经做过；这里只防浮点漂移）
    if (prob_sum > 1e-18) {
        for (int m = 0; m < TOP_K; ++m) trace_prob[m] /= prob_sum;
    } else {
        for (int m = 0; m < TOP_K; ++m) trace_prob[m] = 1.0 / TOP_K;
    }

    return true;
}
