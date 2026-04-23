// deploy_3trajs.cpp 
#include "deploy_3trajs.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>   // readlink
#include <limits.h>   // PATH_MAX
#include <dlfcn.h>    // dladdr

// MindSpore Lite
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"

// stb_image（单翻译单元实现）
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_GIF
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"

// ========== 常量（与训练保持一致，可视情况调整） ==========
static const int   IN_LEN      = deploy_defaults::IN_LEN;
static const int   OUT_LEN     = deploy_defaults::OUT_LEN;
static const int   IN_COLS     = deploy_defaults::IN_COLS;
static const int   TOP_M       = deploy_defaults::TOP_M;
static const int   OUTPUT_SIZE = deploy_defaults::OUTPUT_SIZE;

static const double MAP_CENTER_LON = 94.90563418;
static const double MAP_CENTER_LAT = 36.40415165;
static const double MAP_SIDE_KM    = 6.5;
static const bool   XY_NORMALIZED  = false;

static const float IMG_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float IMG_STD[3]  = {0.229f, 0.224f, 0.225f};

// ========== 最小配置解析 ==========
struct DeployCfg {
    bool        use_map          = true;
    std::string model_path_map   = "./network_with_map.ms";
    std::string model_path_nomap = "./network_no_map.ms";
    std::string image_path_map   = "./map.png";
    int         image_size_map   = 64;
    int         image_size_nomap = 16;
};

static inline std::string trim(const std::string& s) {
    size_t i=0, j=s.size();
    while (i<j && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    while (j>i && std::isspace(static_cast<unsigned char>(s[j-1]))) --j;
    return s.substr(i, j-i);
}
static inline bool ieq(const std::string& a, const char* b) {
    if (a.size()!=std::strlen(b)) return false;
    for (size_t i=0;i<a.size();++i) if (std::tolower(a[i])!=std::tolower(b[i])) return false;
    return true;
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
        std::string v = trim(line.substr(pos+1));
        if (ieq(k, "use_map")) {
            std::string vl = v; for (auto& ch: vl) ch = std::tolower(ch);
            c.use_map = (vl=="1" || vl=="true" || vl=="yes");
        } else if (ieq(k, "model_path_map")) {
            c.model_path_map = v;
        } else if (ieq(k, "model_path_nomap")) {
            c.model_path_nomap = v;
        } else if (ieq(k, "image_path_map")) {
            c.image_path_map = v;
        } else if (ieq(k, "image_size_map")) {
            c.image_size_map = std::max(8, std::atoi(v.c_str()));
        } else if (ieq(k, "image_size_nomap")) {
            c.image_size_nomap = std::max(8, std::atoi(v.c_str()));
        }
    }
    return c;
}

// ========== 配置查找顺序 ==========
// 1) 环境变量 DEPLOY_CFG 指定的绝对/相对路径
// 2) 当前工作目录 ./deploy_cfg.ini
// 3) 与 .so 同目录的 deploy_cfg.ini
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
    // 同目录 .so
    std::string so_dir = ".";
    Dl_info info{};
    if (dladdr((void*)&get_cfg_path, &info) && info.dli_fname) {
        std::string so_path = info.dli_fname;
        auto pos = so_path.find_last_of('/');
        if (pos != std::string::npos) so_dir = so_path.substr(0, pos);
    } else {
        // 退而求其次：/proc/self/exe
        char buf[PATH_MAX]{0};
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf)-1);
        if (n > 0) {
            std::string exep(buf);
            auto pos = exep.find_last_of('/');
            if (pos != std::string::npos) so_dir = exep.substr(0, pos);
        }
    }
    std::string cand = so_dir + "/deploy_cfg.ini";
    std::ifstream t(cand);
    if (t.good()) return cand;
    return "./deploy_cfg.ini"; // 兜底（即使不存在，load 时会提示用默认）
}

// ========== 小工具 ==========
static inline std::pair<double,double> deg2km_scale(double lat0_deg) {
    double lat_rad = lat0_deg * M_PI / 180.0;
    double k_lat = 111.32;
    double k_lon = 111.32 * std::cos(lat_rad);
    return {k_lon, k_lat};
}
static inline std::pair<double,double> xy_km_to_lonlat(double x, double y,
                                                       double lon0, double lat0) {
    auto sc = deg2km_scale(lat0);
    double lon = lon0 + x / std::max(sc.first, 1e-6);
    double lat = lat0 + y / std::max(sc.second, 1e-6);
    return {lon, lat};
}
static inline std::pair<double,double> lonlat_to_xy_km(double lon, double lat,
                                                        double lon0, double lat0) {
    auto sc = deg2km_scale(lat0);
    double x = (lon - lon0) * sc.first;
    double y = (lat - lat0) * sc.second;
    return {x, y};
}
static inline void softmax1d_strict(const std::vector<double>& logits, std::vector<double>& probs) {
    probs.assign(logits.size(), 0.0);
    if (logits.empty()) return;
    double m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> e(logits.size(), 0.0);
    for (size_t i = 0; i < logits.size(); ++i) { e[i] = std::exp(logits[i] - m); sum += e[i]; }
    if (sum <= 1e-18) sum = 1e-18;
    for (size_t i = 0; i < logits.size(); ++i) probs[i] = e[i] / sum;
}

// ========== 图像处理（stb+手写resize+归一化 → NCHW） ==========
struct RawImage { int w=0,h=0,c=0; std::vector<uint8_t> rgb; };

static RawImage load_image_rgb(const std::string& path) {
    int w,h,c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) throw std::runtime_error(std::string("stbi_load fail: ")+path);
    RawImage img; img.w=w; img.h=h; img.c=3;
    img.rgb.assign(data, data + w*h*3);
    stbi_image_free(data);
    return img;
}
static inline void resize_bilinear_rgb888(const RawImage& src, int outW, int outH, std::vector<uint8_t>& out) {
    out.resize(outW*outH*3);
    const float sx = (float)src.w / outW;
    const float sy = (float)src.h / outH;
    for (int y=0; y<outH; ++y) {
        float fy = (y+0.5f)*sy - 0.5f;
        int y0 = (int)std::floor(fy), y1 = y0+1;
        float wy = fy - y0;
        y0 = std::clamp(y0,0,src.h-1); y1 = std::clamp(y1,0,src.h-1);
        for (int x=0; x<outW; ++x) {
            float fx = (x+0.5f)*sx - 0.5f;
            int x0 = (int)std::floor(fx), x1 = x0+1;
            float wx = fx - x0;
            x0 = std::clamp(x0,0,src.w-1); x1 = std::clamp(x1,0,src.w-1);
            for (int c=0; c<3; ++c) {
                auto idx=[&](int X,int Y){return (Y*src.w+X)*3+c;};
                float p00=src.rgb[idx(x0,y0)], p01=src.rgb[idx(x1,y0)];
                float p10=src.rgb[idx(x0,y1)], p11=src.rgb[idx(x1,y1)];
                float p0=p00*(1-wx)+p01*wx;
                float p1=p10*(1-wx)+p11*wx;
                float v = p0*(1-wy)+p1*wy;
                out[(y*outW+x)*3+c]=(uint8_t)std::round(std::clamp(v,0.f,255.f));
            }
        }
    }
}
static std::vector<float> hwc_to_nchw_and_norm(const std::vector<uint8_t>& hwc, int W, int H) {
    std::vector<float> nchw(1*3*H*W);
    for (int c=0; c<3; ++c)
        for (int y=0; y<H; ++y)
            for (int x=0; x<W; ++x) {
                uint8_t u = hwc[(y*W+x)*3+c];
                float v = (float)u/255.0f;
                v = (v - IMG_MEAN[c]) / IMG_STD[c];
                nchw[c*H*W + y*W + x] = v;
            }
    return nchw;
}

// ========== 推理引擎 ==========
class TrajSystem::InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, int img_hw_hint) : img_hw_(img_hw_hint) {
        auto ctx = std::make_shared<mindspore::Context>();
        auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
        ctx->MutableDeviceInfo().push_back(cpu);
        auto ret = model_.Build(model_path, mindspore::kMindIR, ctx);
        if (ret != mindspore::kSuccess) throw std::runtime_error("MindSpore Build failed: "+model_path);

        auto inputs = model_.GetInputs();
        traj_index_ = img_index_ = -1;
        has_image_ = false;
        for (size_t i=0;i<inputs.size();++i) {
            const auto &sh = inputs[i].Shape();
            if (sh.size()==3) {
                // 轨迹输入 [1, T, C]
                traj_index_ = (int)i;
            }
            if (sh.size()==4) {
                // 图像输入 [1, 3, H, W]
                img_index_ = (int)i;
                has_image_ = true;
                if (sh[2]>0 && sh[3]>0)
                    img_hw_ = (int)std::max<int64_t>(sh[2], sh[3]);
            }
        }
        if (traj_index_ < 0) {
            throw std::runtime_error("Cannot identify traj input by rank");
        }
        // 注意：img_index_ 可能为 -1（无图像输入的模型），这种情况 has_image_ = false
    }

    // 对“无图模型”（fullnet）来说，第二个参数会被忽略，只用 traj_1x20x6
    std::vector<float> forward(const std::vector<float>& traj_1x20x6,
                               const std::vector<float>& img_1x3xHxW) {
        auto inputs = model_.GetInputs();
        std::vector<int64_t> exp_traj{1, IN_LEN, IN_COLS};
        std::vector<int64_t> exp_img;
        if (has_image_) {
            exp_img = {1, 3, img_hw_, img_hw_};
        }

        bool need_resize = false;
        if (inputs[traj_index_].Shape() != exp_traj) {
            need_resize = true;
        }
        if (has_image_ && inputs[img_index_].Shape() != exp_img) {
            need_resize = true;
        }

        if (need_resize) {
            std::vector<std::vector<int64_t>> new_shapes(inputs.size());
            for (size_t i=0;i<inputs.size();++i) {
                if ((int)i==traj_index_) {
                    new_shapes[i]=exp_traj;
                } else if (has_image_ && (int)i==img_index_) {
                    new_shapes[i]=exp_img;
                } else {
                    new_shapes[i]=inputs[i].Shape();
                }
            }
            if (model_.Resize(inputs, new_shapes)!=mindspore::kSuccess)
                throw std::runtime_error("Resize failed");
            inputs = model_.GetInputs();
        }

        if (inputs[traj_index_].DataSize() < traj_1x20x6.size()*sizeof(float)) {
            throw std::runtime_error("Input traj too small");
        }
        std::memcpy(inputs[traj_index_].MutableData(),
                    traj_1x20x6.data(),
                    traj_1x20x6.size()*sizeof(float));

        if (has_image_) {
            if (inputs[img_index_].DataSize() < img_1x3xHxW.size()*sizeof(float)) {
                throw std::runtime_error("Input image too small");
            }
            std::memcpy(inputs[img_index_].MutableData(),
                        img_1x3xHxW.data(),
                        img_1x3xHxW.size()*sizeof(float));
        }

        std::vector<mindspore::MSTensor> outputs;
        if (model_.Predict(inputs, &outputs)!=mindspore::kSuccess || outputs.empty())
            throw std::runtime_error("Predict failed");
        const auto& out=outputs[0];
        const size_t n=out.DataSize()/sizeof(float);
        const float* p=static_cast<const float*>(out.Data().get());
        if (!p || n==0) throw std::runtime_error("Invalid output");
        return std::vector<float>(p, p+n);
    }

    int image_hw() const { return img_hw_; }

private:
    mindspore::Model model_;
    int img_hw_=64;
    int traj_index_=-1, img_index_=-1;
    bool has_image_ = false;
};

// ========== TrajSystem 实现 ==========
static bool         g_cfg_loaded = false;
static DeployCfg    g_cfg;  // 只在首次使用时从文件加载

static void ensure_cfg_loaded_once() {
    if (!g_cfg_loaded) {
        std::string path = get_cfg_path();
        std::cerr << "[CFG] using: " << path << "\n";
        g_cfg = load_cfg_file(path);
        g_cfg_loaded = true;
    }
}

TrajSystem::TrajSystem() {
    // 延迟创建 engine_（首次 Feed 时，结合配置决定模型与图像尺寸）
}

bool TrajSystem::LoadRouteFromFile(const std::string& path, LocData_route& out) {
    try {
        int w,h,c;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
        if (!data) return false;
        out.w = w; out.h = h; out.c = 3;
        out.rgb.assign(data, data + w*h*3);
        stbi_image_free(data);
        return true;
    } catch (...) { return false; }
}

std::vector<float> TrajSystem::load_map_to_nchw_from_route(const LocData_route& R, int hw) const {
    if (R.w<=0 || R.h<=0 || (int)R.rgb.size()!=R.w*R.h*3) return std::vector<float>(1*3*hw*hw, 0.f);
    RawImage src{R.w,R.h,3,R.rgb};
    std::vector<uint8_t> resized;
    resize_bilinear_rgb888(src, hw, hw, resized);
    return hwc_to_nchw_and_norm(resized, hw, hw);
}

void TrajSystem::Feed(const std::vector<LocData_loc>& data_loc,
                      const std::vector<LocData_route>& data_route,
                      int instant_ms) {
    ensure_cfg_loaded_once();
    instant_ms_ = std::max(1, instant_ms);

    // 首次构建引擎：根据配置选择模型，并准备一张图（为零或文件/入参）
    if (!engine_) {
        const std::string& model_path = g_cfg.use_map ? g_cfg.model_path_map : g_cfg.model_path_nomap;
        const int img_hw_hint = g_cfg.use_map ? g_cfg.image_size_map : g_cfg.image_size_nomap;
        engine_ = new InferenceEngine(model_path, img_hw_hint);

        const int real_hw = engine_->image_hw();
        if (!g_cfg.use_map) {
            // 无路网图模型：历史逻辑仍然构造一张全零图；但如果模型没有图像输入，会在 forward 里自动忽略
            last_map_nchw_.assign(1*3*real_hw*real_hw, 0.f);
        } else {
            LocData_route R;
            if (!data_route.empty()) {
                const auto& tail = data_route.back();
                last_map_nchw_ = load_map_to_nchw_from_route(tail, real_hw);
            } else {
                if (LoadRouteFromFile(g_cfg.image_path_map, R))
                    last_map_nchw_ = load_map_to_nchw_from_route(R, real_hw);
                else
                    last_map_nchw_.assign(1*3*real_hw*real_hw, 0.f);
            }
        }
    }

    // 追加入缓冲并按 time 排序
    for (auto& r : data_loc) buffer_.push_back(r);
    buffer_.sort([](const LocData_loc& a, const LocData_loc& b){ return a.time < b.time; });

    // 控制20分钟容量
    const int need_cap = (int)std::llround(20.0 * 60.0 * 1000.0 / instant_ms_);
    while ((int)buffer_.size() > need_cap) buffer_.pop_front();

    // 若传入了新的路线图，更新（仅在 use_map = true 时生效）
    if (g_cfg.use_map && !data_route.empty()) {
        const int real_hw = engine_->image_hw();
        const auto& R = data_route.back();
        last_map_nchw_ = load_map_to_nchw_from_route(R, real_hw);
    }
    if (!g_cfg.use_map) {
        const int real_hw = engine_->image_hw();
        last_map_nchw_.assign(1*3*real_hw*real_hw, 0.f);
    }
}

bool TrajSystem::Infer(std::map<int, std::vector<LocData_pred>>& pred_trace,
                       std::vector<double>& trace_prob,
                       std::map<int, std::vector<double>>& strike_areas,
                       std::vector<double>& area_prob) {
    if (!engine_) return false;
    const int need_cap = (int)std::llround(20.0 * 60.0 * 1000.0 / std::max(1,instant_ms_));
    if ((int)buffer_.size() < need_cap) return false;

    // 每 60s 取一个点
    const int hop = (int)std::llround(60.0 * 1000.0 / std::max(1, instant_ms_));
    std::vector<LocData_loc> picked; picked.reserve(IN_LEN);
    int idx = 0;
    for (auto it=buffer_.begin(); it!=buffer_.end() && (int)picked.size()<IN_LEN; ++it, ++idx)
        if (idx % hop == 0) picked.push_back(*it);
    if ((int)picked.size() < IN_LEN) {
        auto it=buffer_.end(); --it;
        while ((int)picked.size()<IN_LEN) picked.push_back(*it);
    }

    // 构造轨迹输入 [1,20,6]
    // 注意：FullNet 模型期望输入是原始物理量 [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]
    // 模型内部会处理：
    //   1) 增量编码：位置转增量（delta_pos[0]=0, delta_pos[t]=pos[t]-pos[t-1]）
    //   2) 归一化：使用 scaler_A (mean/std) 归一化
    //   3) LSTM 前向
    //   4) 反归一化
    //   5) 增量转绝对坐标（cumsum + 历史最后位置）
    // 所以部署代码只需要：经纬度 → km坐标，然后直接输入模型
    std::vector<float> traj(1*IN_LEN*IN_COLS, 0.f);
    for (int t=0; t<IN_LEN; ++t) {
        // 经纬度 → km坐标（相对于 MAP_CENTER）
        auto xy = lonlat_to_xy_km(picked[t].Lon, picked[t].Lat, MAP_CENTER_LON, MAP_CENTER_LAT);
        traj[t*IN_COLS + 0] = (float)xy.first;   // x_km
        traj[t*IN_COLS + 1] = (float)xy.second;   // y_km
        // 注意：需要确认 Alt 的单位。如果甲方输入是米，需要 /1000.0 转为 km
        // 如果甲方输入已经是 km，则直接使用
        traj[t*IN_COLS + 2] = (float)picked[t].Alt;  // z_km (需要确认单位)
        // 速度单位：根据用户要求"速度单位不用改"，假设输入已经是 km/s
        traj[t*IN_COLS + 3] = (float)picked[t].Speed_north;   // vx_kmps
        traj[t*IN_COLS + 4] = (float)picked[t].Speed_tianxiang;  // vy_kmps
        traj[t*IN_COLS + 5] = (float)picked[t].Speed_east;   // vz_kmps
    }

    // 前向
    auto y = engine_->forward(traj, last_map_nchw_);

    // 组织输出（TOP_M × 68）
    std::vector<std::array<float, OUTPUT_SIZE>> modes;
    if (y.size() == (size_t)TOP_M * OUTPUT_SIZE) {
        modes.resize(TOP_M);
        for (int m=0;m<TOP_M;++m)
            std::memcpy(modes[m].data(), y.data()+m*OUTPUT_SIZE, sizeof(float)*OUTPUT_SIZE);
    } else {
        if (y.size() < (size_t)OUTPUT_SIZE) throw std::runtime_error("Unexpected output size");
        modes.resize(TOP_M);
        const float* last = y.data() + (y.size()/OUTPUT_SIZE - 1)*OUTPUT_SIZE;
        for (int m=0;m<TOP_M;++m) std::memcpy(modes[m].data(), last, sizeof(float)*OUTPUT_SIZE);
    }

    // 概率
    // 注意：网络输出索引 67 已经是 softmax 后的概率（mode_prob），不需要再做 softmax
    // 直接使用，但需要归一化确保和为 1（防止浮点误差）
    trace_prob.assign(TOP_M, 0.0);
    double sum = 0.0;
    for (int m=0;m<TOP_M;++m) {
        trace_prob[m] = (double)modes[m][67];
        sum += trace_prob[m];
    }
    // 归一化确保和为 1
    if (sum > 1e-18) {
        for (int m=0;m<TOP_M;++m) trace_prob[m] /= sum;
    } else {
        // 如果所有概率都很小，均匀分配
        for (int m=0;m<TOP_M;++m) trace_prob[m] = 1.0 / TOP_M;
    }

    // 回写四类输出（含“秒级时间戳”）
    pred_trace.clear(); strike_areas.clear(); area_prob.assign(TOP_M, 0.0);
    const int64_t base_ts = picked.back().time;

    for (int m=0;m<TOP_M;++m) {
        const auto& v = modes[m];
        auto xy2ll = [&](double xkm, double ykm){
            double x = xkm, y = ykm;
            if (XY_NORMALIZED) { x *= MAP_SIDE_KM; y *= MAP_SIDE_KM; }
            auto ll = xy_km_to_lonlat(x, y, MAP_CENTER_LON, MAP_CENTER_LAT);
            return ll;
        };

        std::vector<LocData_pred> rows; rows.reserve(OUT_LEN);

        // 前9步：每步 +60s
        // 注意：模型输出是绝对坐标 [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]
        // 需要转换为经纬度输出
        for (int t=0; t<9; ++t) {
            int off = t*6;
            auto ll = xy2ll(v[off+0], v[off+1]);
            LocData_pred r = picked.back();
            r.time = base_ts + (t+1)*60;  // 秒级时间戳
            r.Lon=ll.first; r.Lat=ll.second;
            // 注意：如果甲方期望输出单位是米，需要 *1000.0；如果已经是 km，则直接使用
            r.Alt = v[off+2];  // z_km (需要确认输出单位要求)
            // 速度单位：根据用户要求"速度单位不用改"，假设输出保持 km/s
            r.Speed_north = v[off+3];   // vx_kmps
            r.Speed_tianxiang = v[off+4];  // vy_kmps
            r.Speed_east = v[off+5];    // vz_kmps
            rows.push_back(r);
        }
        // 第10步：+600s，含类型与威胁度 + 打击区与区域概率
        // 注意：模型输出格式 [0..59: 轨迹, 60: intent_class, 61: threat_prob, 62..64: strike_pos, 65: radius, 66: conf, 67: mode_prob]
        {
            int off = 9*6;  // 第10步的起始索引：9*6 = 54
            auto ll0 = xy2ll(v[off+0], v[off+1]);  // v[54], v[55] 是第10步的 x, y (km)
            LocData_pred r = picked.back();
            r.time = base_ts + 600;
            r.Lon=ll0.first; r.Lat=ll0.second;
            // 注意：如果甲方期望输出单位是米，需要 *1000.0；如果已经是 km，则直接使用
            r.Alt = v[off+2];  // z_km (需要确认输出单位要求)
            // 速度单位：根据用户要求"速度单位不用改"，假设输出保持 km/s
            r.Speed_north = v[off+3];   // vx_kmps
            r.Speed_tianxiang = v[off+4];  // vy_kmps
            r.Speed_east = v[off+5];    // vz_kmps
            r.Type=(int)std::llround(v[60]);      // intent_class
            r.Threat=(int)std::llround(v[61] * 100.0f);  // threat_prob (0~1) 转为百分比
            rows.push_back(r);

            // 打击点位置：索引 62, 63, 64 (x_km, y_km, z_km)
            auto cll = xy2ll(v[62], v[63]);
            // 注意：strike_areas 的 z 和 radius 单位需要确认
            strike_areas[m] = { (double)cll.first, (double)cll.second, (double)v[64], (double)v[65] };  // lon, lat, z(km), radius(km)
            area_prob[m] = (double)v[67];  // mode_prob
        }

        pred_trace[m] = std::move(rows);
    }
    return true;
}
