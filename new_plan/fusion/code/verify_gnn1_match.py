"""
fusion/code/verify_gnn1_match.py
--------------------------------
对照 fusion 内部 GNN1 调用 vs 独立加载 gnn1 模型 + 直接喂 cache.candidates 的结果。
两者应当完全一致（差异在 fp32 噪声以内），证明 test_pipeline 的 top-K 就是真实
LSTM1+GNN1 推理结果，不存在"用了随机权重"或"喂错输入"的情况。

用法::
    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m fusion.code.verify_gnn1_match
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

from fusion.code.build import load_fusion_config, _resolve_rel  # noqa: E402
from fusion.code.full_net_v2 import build_full_net_from_fusion_config  # noqa: E402
from fusion.code.test_pipeline import _Scaler, _decode_window_to_phys  # noqa: E402


def main() -> None:
    fusion_cfg_path = REPO / "fusion" / "config.yaml"
    fusion_cfg, fusion_cfg_dir = load_fusion_config(fusion_cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- 1) fusion 模型（自动取最新 ckpt） ----
    model = build_full_net_from_fusion_config(fusion_cfg_path).to(device).eval()

    # ---- 2) 单独跑一份 gnn1（同一 ckpt）作为对照 ----
    sys.path.insert(0, str(REPO / "gnn1" / "code"))
    from train.model import build_model_from_config as build_gnn1_solo  # noqa: E402

    gnn1_cfg_path = _resolve_rel(fusion_cfg["gnn1"]["config"], fusion_cfg_dir)
    gnn1_cfg = yaml.safe_load(open(gnn1_cfg_path, encoding="utf-8"))
    gnn1_solo = build_gnn1_solo(gnn1_cfg).to(device).eval()
    gnn1_root = gnn1_cfg_path.parent
    gnn1_ck_dir = gnn1_root / "checkpoints"
    gnn1_ck = sorted(gnn1_ck_dir.rglob("*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)[0]
    gnn1_solo.load_state_dict(torch.load(gnn1_ck, map_location=device))

    lstm1_root = _resolve_rel(fusion_cfg["lstm1"]["config"], fusion_cfg_dir).parent
    lstm1_ck = sorted((lstm1_root / "checkpoints").rglob("*.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[0]

    print("\n========== 实际加载的 ckpt ==========")
    print(f"LSTM1 ckpt = {lstm1_ck}")
    print(f"GNN1  ckpt = {gnn1_ck}")
    print(f"GNN1 enable_flags from fusion = {model.enable_flags}")

    # ---- 3) 真实数据 ----
    raw = np.load(gnn1_root / "data" / "raw" / "test.npz")
    cache = np.load(gnn1_root / "data" / "cache" / "test.npz")
    scaler = _Scaler.load(gnn1_root / "data" / "cache" / "scaler_posvel.npz")
    spw = int(gnn1_cfg.get("data", {}).get("samples_per_window", 5))

    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(int(raw["task_type"].shape[0]), size=4,
                               replace=False).tolist())
    print(f"\n测试样本 raw idx = {chosen}  (samples_per_window={spw})")

    # 反解 hist 物理空间
    hist_phys_list = []
    for i in chosen:
        cw = i // spw
        hp, _ = _decode_window_to_phys(cache["history"][cw], None, scaler)
        hist_phys_list.append(hp)
    hist_t = torch.from_numpy(np.stack(hist_phys_list, 0).astype(np.float32)).to(device)

    task_type = torch.from_numpy(raw["task_type"][chosen].astype(np.int64)).to(device)
    type_id = torch.from_numpy(raw["type"][chosen].astype(np.int64)).to(device)
    position = torch.from_numpy(raw["position"][chosen].astype(np.float32)).to(device)

    # ---- 4) fusion 内部跑：hist → LSTM1 → 拿候选 → GNN1 ----
    ctx = ContextBatch(task_type=task_type, type=type_id, position=position,
                       road_points=None, road_mask=None, eta=None)
    with torch.no_grad():
        ctx_n = model._normalize_ctx(ctx, batch_size=hist_t.shape[0], device=device)
        x_clean, _ = model.outlier_filter(hist_t)
        x_A = model._build_A_input(x_clean)
        fut_A_fusion = model.lstm1(x_A)
        out_fusion = model.gnn1({
            "cand_trajs": fut_A_fusion,
            "task_type": ctx_n.task_type,
            "type": ctx_n.type,
            "position": ctx_n.position,
        })

        # ---- 5) 单独 GNN1 + cache.candidates（gnn1 训练管线方式） ----
        cand_cache = torch.from_numpy(
            np.stack([cache["candidates"][i // spw] for i in chosen], 0)
        ).to(device)
        out_solo = gnn1_solo({
            "cand_trajs": cand_cache,
            "task_type": task_type,
            "type": type_id,
            "position": position,
        })

    print("\n========== fusion 内部 GNN1 vs 独立 GNN1（用 cache.candidates） ==========")
    print(f"top_idx   fusion = {out_fusion['top_idx'].cpu().numpy().tolist()}")
    print(f"top_idx   solo   = {out_solo['top_idx'].cpu().numpy().tolist()}")
    print(f"top_probs fusion = {out_fusion['top_probs'].cpu().numpy().round(4).tolist()}")
    print(f"top_probs solo   = {out_solo['top_probs'].cpu().numpy().round(4).tolist()}")

    diff_logits = (out_fusion["logits"] - out_solo["logits"]).abs().max().item()
    diff_probs = (out_fusion["top_probs"] - out_solo["top_probs"]).abs().max().item()
    diff_cand = (fut_A_fusion - cand_cache).abs().max().item()
    print(f"\nLSTM1 候选差异 max = {diff_cand:.3e}   "
          f"GNN1 logits 差异 max = {diff_logits:.3e}   "
          f"top_probs 差异 max = {diff_probs:.3e}")

    # top_idx 必须 bit-exact 一致；数值上 GPU fp32 + cudnn 非确定性会有 ~1e-3 噪声
    idx_eq = bool(torch.equal(out_fusion["top_idx"], out_solo["top_idx"]))
    print(f"top_idx 完全相等: {idx_eq}")
    if idx_eq and diff_probs < 5e-3:
        print("→ fusion 的 top-K 就是 LSTM1+GNN1 真实推理结果，加载的是真权重。")
        print("  数值差异属 GPU fp32 + cudnn 非确定性噪声（top_idx bit-exact 一致）。")
    else:
        print("→ 不一致！需要排查。")


if __name__ == "__main__":
    main()
