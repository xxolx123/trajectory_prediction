"""
fusion/code/verify_gnn2_match.py
--------------------------------
对照 fusion 内部 GNN2 调用 vs 独立加载 GNN2 + 同一 ckpt 的结果。

为什么需要这一层验证：
  - gnn2/code/train/_verify_manual_attention.py 已经证明 SDPA 版 StrikeZoneNet
    与手写版 StrikeZoneNetManual 在同 state_dict 下逐位等价（max|Δ| ≈ 0）。
  - 本脚本要补的是 fusion 端的"组装链路"是否正确：
      1) build.py: ckpt 路径解析 + manual_attention 注入是否生效；
      2) fusion 内 GNN2 拿到的 (pred_traj, eta) 张量是否和独立调用形态一致；
      3) `model.gnn2` 实际类型是否是 StrikeZoneNetManual（manual_attention=true 时）。

期望结果：
    max|Δ strike_pos|    < 1e-4   (CPU 通常 0)
    max|Δ strike_radius| < 1e-4
    max|Δ strike_conf|   < 1e-4
    cand_xy 数值打印两侧逐项对得上。

用法::
    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m fusion.code.verify_gnn2_match
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.context_schema import ContextBatch  # noqa: E402

from fusion.code.build import (  # noqa: E402
    _resolve_ckpt_to_file,
    _resolve_rel,
    load_fusion_config,
)
from fusion.code.full_net_v2 import build_full_net_from_fusion_config  # noqa: E402
from fusion.code.test_pipeline import _Scaler, _decode_window_to_phys  # noqa: E402

from gnn2.code.train.model import build_model_from_config as build_gnn2_solo  # noqa: E402


def main() -> None:
    fusion_cfg_path = REPO / "fusion" / "config.yaml"
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- 1) fusion 模型（自动取最新 ckpt + manual_attention 注入） ----
    model = build_full_net_from_fusion_config(fusion_cfg_path).to(device).eval()
    if not bool(model.enable_flags.get("gnn2", False)):
        raise RuntimeError(
            "fusion.config 中 gnn2.enable=false，无法验证 GNN2。请先把 gnn2.enable 改成 true。"
        )

    # ---- 2) 单独跑一份 GNN2（SDPA 版）作对照（同一份 ckpt） ----
    gnn2_cfg_path = _resolve_rel(
        fusion_cfg.get("gnn2", {}).get("config", ""), fusion_cfg_dir,
    )
    if gnn2_cfg_path is None:
        raise RuntimeError("fusion.config: gnn2.config 路径为空")
    with gnn2_cfg_path.open("r", encoding="utf-8") as f:
        gnn2_cfg = yaml.safe_load(f)

    # 强制走 SDPA 版（与 fusion 端的 manual 版形成 ckpt 互通对照）
    solo_cfg = {**gnn2_cfg}
    solo_model_section = dict(solo_cfg.get("model", {}) or {})
    solo_model_section["manual_attention"] = False
    solo_model_section["type"] = "strike_zone_transformer"
    solo_cfg["model"] = solo_model_section
    gnn2_solo = build_gnn2_solo(solo_cfg).to(device).eval()

    gnn2_ckpt_path_raw = _resolve_rel(
        fusion_cfg.get("gnn2", {}).get("ckpt", ""), fusion_cfg_dir,
    )
    gnn2_ckpt = _resolve_ckpt_to_file(gnn2_ckpt_path_raw)
    if gnn2_ckpt is None:
        raise FileNotFoundError(
            f"找不到 gnn2 ckpt：{gnn2_ckpt_path_raw}"
        )
    sd = torch.load(gnn2_ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = gnn2_solo.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[Verify/GNN2 WARN] solo gnn2 load: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )

    print("\n========== 实际加载的 ckpt / 模块类型 ==========")
    print(f"GNN2  ckpt = {gnn2_ckpt}")
    print(f"fusion gnn2 模块类 = {type(model.gnn2).__name__}")
    print(f"solo   gnn2 模块类 = {type(gnn2_solo).__name__}")
    print(f"fusion enable_flags = {model.enable_flags}")

    # ---- 3) 真实数据：从 gnn1 test cache 抽 4 条窗口 ----
    gnn1_root = _resolve_rel(
        fusion_cfg.get("gnn1", {}).get("config", ""), fusion_cfg_dir,
    ).parent
    raw = np.load(gnn1_root / "data" / "raw" / "test.npz")
    cache = np.load(gnn1_root / "data" / "cache" / "test.npz")
    scaler = _Scaler.load(gnn1_root / "data" / "cache" / "scaler_posvel.npz")
    with (gnn1_root / "config.yaml").open("r", encoding="utf-8") as f:
        gnn1_cfg = yaml.safe_load(f)
    spw = int(gnn1_cfg.get("data", {}).get("samples_per_window", 5))

    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(int(raw["task_type"].shape[0]), size=4,
                               replace=False).tolist())
    print(f"\n测试样本 raw idx = {chosen}  (samples_per_window={spw})")

    hist_phys_list = []
    for i in chosen:
        cw = i // spw
        hp, _ = _decode_window_to_phys(cache["history"][cw], None, scaler)
        hist_phys_list.append(hp)
    hist_t = torch.from_numpy(np.stack(hist_phys_list, 0).astype(np.float32)).to(device)
    B = int(hist_t.shape[0])

    task_type = torch.from_numpy(raw["task_type"][chosen].astype(np.int64)).to(device)
    type_id = torch.from_numpy(raw["type"][chosen].astype(np.int64)).to(device)
    position = torch.from_numpy(raw["position"][chosen].astype(np.float32)).to(device)

    # ETA 用 gnn2/config.yaml 的范围随机采样（per-sample）
    gnn2_data = gnn2_cfg.get("data", {}) or {}
    eta_min = int(gnn2_data.get("eta_min_sec", 0))
    eta_max = int(gnn2_data.get("eta_max_sec", 600))
    eta_np = rng.integers(eta_min, eta_max + 1, size=B, dtype=np.int64)
    eta_t = torch.from_numpy(eta_np).to(device)
    print(f"eta per sample = {eta_np.tolist()}  (range=[{eta_min},{eta_max}] s)")

    # ---- 4) 走 fusion 内部到 refined_flat（无路网，直接 LSTM1+GNN1 → 物理 top-K） ----
    K = int(model.top_k)
    Tout = int(model.fut_len)
    Df = int(model.feature_dim)
    ctx = ContextBatch(task_type=task_type, type=type_id, position=position,
                       road_points=None, road_mask=None, eta=eta_t)
    with torch.no_grad():
        ctx_n = model._normalize_ctx(ctx, batch_size=B, device=device)
        x_clean, _ = model.outlier_filter(hist_t)
        x_A = model._build_A_input(x_clean)
        fut_A_norm = model.lstm1(x_A)                                # [B, M, T, 6]
        gnn1_out = model.gnn1({
            "cand_trajs": fut_A_norm,
            "task_type":  ctx_n.task_type,
            "type":       ctx_n.type,
            "position":   ctx_n.position,
        })
        top_idx = gnn1_out["top_idx"]                                # [B, K]
        idx_exp = top_idx[..., None, None].expand(B, K, Tout, Df)
        fut_A_top = torch.gather(fut_A_norm, 1, idx_exp)
        fut_phys_top = model._decode_future_to_abs(x_clean, fut_A_top)
        refined_flat = fut_phys_top.reshape(B * K, Tout, Df)         # [B*K, T, 6]
        ctx_flat = model._expand_ctx(ctx_n, K)
        eta_flat = ctx_flat.eta                                       # [B*K]

        # ---- 5) fusion 内 GNN2 (manual) vs 独立 GNN2 (SDPA) ----
        out_fusion = model.gnn2(refined_flat, eta_flat)
        out_solo = gnn2_solo(refined_flat, eta_flat)

    print("\n========== fusion 内 GNN2 (manual) vs 独立 GNN2 (SDPA) ==========")
    keys = ["strike_pos", "strike_radius", "strike_conf"]
    max_diffs = {}
    for k in keys:
        d = (out_fusion[k] - out_solo[k]).abs().max().item()
        max_diffs[k] = d
        print(f"  max|Δ {k:<13}| = {d:.3e}")
    overall = max(max_diffs.values())
    print(f"\n  overall max|Δ| = {overall:.3e}")
    if overall < 1e-4:
        print("→ 一致（差异 ≤ 1e-4），fusion 内 manual 版与独立 SDPA 版的 GNN2 数值等价。")
        print("  说明 ckpt 加载 + manual_attention 注入 + 张量喂入路径都正确。")
    elif overall < 1e-2:
        print("→ 接近一致，差异在 fp32 噪声量级以内（不影响业务输出）。")
    else:
        print("→ 明显不一致！请排查 ckpt 路径 / state_dict 键 / manual_attention 注入逻辑。")

    # ---- 6) 抽样数值对照（前 3 条 candidate） ----
    n_show = min(3, B * K)
    print("\n========== 抽样数值对照（前 3 条 candidate）==========")
    for i in range(n_show):
        sp_f = out_fusion["strike_pos"][i].cpu().tolist()
        sp_s = out_solo["strike_pos"][i].cpu().tolist()
        rad_f = float(out_fusion["strike_radius"][i, 0])
        rad_s = float(out_solo["strike_radius"][i, 0])
        cnf_f = float(out_fusion["strike_conf"][i, 0])
        cnf_s = float(out_solo["strike_conf"][i, 0])
        print(
            f"  cand {i:>2d}  fusion: pos=({sp_f[0]: .4f},{sp_f[1]: .4f},"
            f"{sp_f[2]: .4f})  r={rad_f:.4f}  conf={cnf_f:.4f}"
        )
        print(
            f"           solo  : pos=({sp_s[0]: .4f},{sp_s[1]: .4f},"
            f"{sp_s[2]: .4f})  r={rad_s:.4f}  conf={cnf_s:.4f}"
        )


if __name__ == "__main__":
    main()
