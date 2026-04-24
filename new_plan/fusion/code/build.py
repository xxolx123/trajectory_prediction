"""
fusion/code/build.py
--------------------
把各子网络（LSTM1 / GNN1 / ConstraintOptimizer / LSTM2 / GNN2）加载组装
成一个 FullNetV2。

路径策略：
  - fusion/config.yaml 里每个子网络段都有 config / ckpt 字段
  - 它们都相对 fusion/config.yaml 解析
  - fusion/config.yaml 本身的位置通过 main 里的 project_root 推出
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# 把 new_plan/ 加到 sys.path，让各子包能被 import
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 子网络的 build_model_from_config
from lstm1.code.train.model import build_model_from_config as build_lstm1       # noqa: E402
from gnn1.code.train.model import build_model_from_config as build_gnn1         # noqa: E402
from constraint_optimizer.code.train.module import build_module_from_config as build_constraint  # noqa: E402
from lstm2.code.train.model import build_model_from_config as build_lstm2       # noqa: E402
from gnn2.code.train.model import build_model_from_config as build_gnn2         # noqa: E402

from common.scaler import load_mean_std_from_npz  # noqa: E402


# ==============================================================================
# 工具
# ==============================================================================

def _resolve_rel(rel_path: str, base_dir: Path) -> Optional[Path]:
    """
    相对 base_dir 解析 rel_path。rel_path 为空字符串时返回 None。
    """
    if not rel_path:
        return None
    p = Path(rel_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _try_load_state_dict(model: torch.nn.Module, ckpt_path: Optional[Path], tag: str) -> None:
    if ckpt_path is None:
        print(f"[Fusion] {tag}: 未指定 ckpt，用随机权重")
        return
    if not ckpt_path.is_file():
        print(f"[Fusion] {tag}: ckpt 不存在 {ckpt_path}，用随机权重")
        return
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[Fusion] {tag}: 已载入 {ckpt_path}"
        + (f", missing={len(missing)}" if missing else "")
        + (f", unexpected={len(unexpected)}" if unexpected else "")
    )


# ==============================================================================
# 构造各子网络
# ==============================================================================

def build_subnetworks(
    fusion_cfg: Dict[str, Any],
    fusion_cfg_dir: Path,
) -> Tuple[
    torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module,
    np.ndarray, np.ndarray, int, int,
]:
    """
    返回 (lstm1, gnn1, constraint, lstm2, gnn2, mean_A, std_A, lstm1_modes, top_k)

    - lstm1_modes: LSTM1 的候选数 M（来自 lstm1_cfg.model.modes）
    - top_k:       Fusion 保留的候选数（来自 gnn1_cfg.train.keep_top_k，缺省 3）
    """

    # ---- LSTM1 ----
    sec = fusion_cfg.get("lstm1", {})
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: lstm1.config 是必填项")
    lstm1_cfg = _load_yaml(cfg_path)
    lstm1 = build_lstm1(lstm1_cfg)
    _try_load_state_dict(lstm1, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "LSTM1")

    lstm1_modes = int(lstm1_cfg.get("model", {}).get("modes", 5))

    scaler_path = _resolve_rel(sec.get("scaler", ""), fusion_cfg_dir)
    if scaler_path is not None and scaler_path.exists():
        mean_A, std_A = load_mean_std_from_npz(scaler_path)
        print(f"[Fusion] LSTM1: 载入 scaler {scaler_path}")
    else:
        feat_dim = int(lstm1_cfg.get("model", {}).get("input_size", 6))
        mean_A = np.zeros((feat_dim,), dtype=np.float32)
        std_A = np.ones((feat_dim,), dtype=np.float32)
        print("[Fusion] LSTM1: 未找到 scaler，使用 mean=0/std=1 兜底")

    # ---- GNN1 ----
    sec = fusion_cfg.get("gnn1", {})
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: gnn1.config 是必填项")
    gnn1_cfg = _load_yaml(cfg_path)
    gnn1 = build_gnn1(gnn1_cfg)
    _try_load_state_dict(gnn1, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "GNN1")

    top_k = int(gnn1_cfg.get("train", {}).get("keep_top_k", 3))
    print(f"[Fusion] GNN1: lstm1_modes={lstm1_modes}, keep_top_k={top_k}")

    # ---- ConstraintOptimizer ----
    sec = fusion_cfg.get("constraint_optimizer", {})
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: constraint_optimizer.config 是必填项")
    constr_cfg = _load_yaml(cfg_path)
    constraint = build_constraint(constr_cfg)
    _try_load_state_dict(constraint, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "ConstraintOptimizer")

    # ---- LSTM2 ----
    sec = fusion_cfg.get("lstm2", {})
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: lstm2.config 是必填项")
    lstm2_cfg = _load_yaml(cfg_path)
    lstm2 = build_lstm2(lstm2_cfg)
    _try_load_state_dict(lstm2, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "LSTM2")

    # ---- GNN2 ----
    sec = fusion_cfg.get("gnn2", {})
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: gnn2.config 是必填项")
    gnn2_cfg = _load_yaml(cfg_path)
    gnn2 = build_gnn2(gnn2_cfg)
    _try_load_state_dict(gnn2, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "GNN2")

    return lstm1, gnn1, constraint, lstm2, gnn2, mean_A, std_A, lstm1_modes, top_k


def load_fusion_config(fusion_cfg_path: Path) -> Tuple[Dict[str, Any], Path]:
    """
    读取 fusion 自己的 config.yaml。
    返回 (fusion_cfg_dict, fusion_cfg_dir)。
    """
    fusion_cfg = _load_yaml(fusion_cfg_path)
    return fusion_cfg, fusion_cfg_path.resolve().parent
