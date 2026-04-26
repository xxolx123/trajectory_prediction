"""
fusion/code/build.py
--------------------
把各子网络（LSTM1 / GNN1 / ConstraintOptimizer / LSTM2 / GNN2）加载组装
成一个 FullNetV2。

每个子网络段在 fusion/config.yaml 里都有 `enable: true/false`：
  - lstm1 / gnn1 是流水线主干，必须 enable=true（关闭直接报错）。
  - constraint_optimizer / lstm2 / gnn2 关闭时该字段返回 None，FullNetV2 会
    在 forward 里跳过对应步骤，输出 [B, K, 68] 中相应位置写哨兵值。

路径策略：
  - fusion/config.yaml 里每个子网络段都有 config / ckpt 字段
  - 它们都相对 fusion/config.yaml 解析
  - fusion/config.yaml 本身的位置通过 main 里的 project_root 推出
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    """相对 base_dir 解析 rel_path。rel_path 为空字符串时返回 None。"""
    if not rel_path:
        return None
    p = Path(rel_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_ckpt_to_file(ckpt_path: Optional[Path]) -> Optional[Path]:
    """
    把 ckpt 路径解析成具体 .pt 文件：
      - None / 空 → None
      - .pt 文件 → 直接返回
      - 目录       → 递归搜 .pt，按修改时间取最新
      - 其他不存在 → None
    """
    if ckpt_path is None:
        return None
    if ckpt_path.is_file():
        return ckpt_path
    if ckpt_path.is_dir():
        cands = list(ckpt_path.rglob("*.pt"))
        if not cands:
            return None
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    return None


def _try_load_state_dict(model: torch.nn.Module, ckpt_path: Optional[Path], tag: str) -> None:
    resolved = _resolve_ckpt_to_file(ckpt_path)
    if resolved is None:
        if ckpt_path is None:
            print(f"[Fusion] {tag}: 未指定 ckpt，用随机权重")
        else:
            print(f"[Fusion] {tag}: ckpt 路径无可用 .pt（{ckpt_path}），用随机权重")
        return
    sd = torch.load(resolved, map_location="cpu")
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[Fusion] {tag}: 已载入 {resolved}"
        + (f", missing={len(missing)}" if missing else "")
        + (f", unexpected={len(unexpected)}" if unexpected else "")
    )


def _is_enabled(sec: Dict[str, Any], default: bool = True) -> bool:
    """读取 enable 字段；缺省视为 default（向后兼容旧 config）。"""
    if not isinstance(sec, dict):
        return default
    val = sec.get("enable", default)
    return bool(val)


def _build_optional(
    fusion_cfg: Dict[str, Any],
    name: str,
    build_fn: Callable[[Dict[str, Any]], torch.nn.Module],
    fusion_cfg_dir: Path,
    tag: str,
) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]], bool]:
    """
    可选子网络的统一构造入口：
      - enable=false：返回 (None, None, False)
      - enable=true ：读 sub-config -> build_fn -> 加载 ckpt（如指定）

    返回:
      (module_or_None, sub_cfg_or_None, enabled_bool)
    """
    sec = fusion_cfg.get(name, {}) or {}
    enabled = _is_enabled(sec, default=True)
    if not enabled:
        print(f"[Fusion] {tag}: disabled (enable=false)，forward 时将跳过并填哨兵值")
        return None, None, False

    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError(f"fusion.config: {name}.config 是必填项（当前 enable=true）")
    sub_cfg = _load_yaml(cfg_path)
    module = build_fn(sub_cfg)
    _try_load_state_dict(module, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), tag)
    return module, sub_cfg, True


# ==============================================================================
# 构造各子网络
# ==============================================================================

def build_subnetworks(
    fusion_cfg: Dict[str, Any],
    fusion_cfg_dir: Path,
) -> Tuple[
    torch.nn.Module,                      # lstm1（必有）
    torch.nn.Module,                      # gnn1（必有）
    Optional[torch.nn.Module],            # constraint
    Optional[torch.nn.Module],            # lstm2
    Optional[torch.nn.Module],            # gnn2
    np.ndarray,                           # mean_A
    np.ndarray,                           # std_A
    int,                                  # lstm1_modes
    int,                                  # top_k
    Dict[str, bool],                      # enable_flags
]:
    """
    返回 (lstm1, gnn1, constraint, lstm2, gnn2, mean_A, std_A,
          lstm1_modes, top_k, enable_flags)

    - constraint / lstm2 / gnn2 在 enable=false 时为 None
    - lstm1_modes: LSTM1 的候选数 M（来自 lstm1_cfg.model.modes）
    - top_k:       GNN1 保留的 top-K（来自 gnn1_cfg.model.top_k，缺省 3）
    - enable_flags: {'lstm1', 'gnn1', 'constraint_optimizer', 'lstm2', 'gnn2': bool}
    """

    enable_flags: Dict[str, bool] = {}

    # ---- LSTM1（必开）----
    sec = fusion_cfg.get("lstm1", {}) or {}
    if not _is_enabled(sec, default=True):
        raise RuntimeError("fusion.config: lstm1 是流水线主干，必须 enable=true")
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: lstm1.config 是必填项")
    lstm1_cfg = _load_yaml(cfg_path)
    lstm1 = build_lstm1(lstm1_cfg)
    _try_load_state_dict(lstm1, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "LSTM1")
    enable_flags["lstm1"] = True

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

    # ---- GNN1（必开）----
    sec = fusion_cfg.get("gnn1", {}) or {}
    if not _is_enabled(sec, default=True):
        raise RuntimeError("fusion.config: gnn1 是流水线主干，必须 enable=true")
    cfg_path = _resolve_rel(sec.get("config", ""), fusion_cfg_dir)
    if cfg_path is None:
        raise RuntimeError("fusion.config: gnn1.config 是必填项")
    gnn1_cfg = _load_yaml(cfg_path)
    gnn1 = build_gnn1(gnn1_cfg)
    _try_load_state_dict(gnn1, _resolve_rel(sec.get("ckpt", ""), fusion_cfg_dir), "GNN1")
    enable_flags["gnn1"] = True

    top_k = int(gnn1_cfg.get("model", {}).get("top_k", 3))
    print(f"[Fusion] GNN1: lstm1_modes={lstm1_modes}, top_k={top_k}")

    # ---- ConstraintOptimizer（可选）----
    constraint, _, en = _build_optional(
        fusion_cfg, "constraint_optimizer", build_constraint, fusion_cfg_dir,
        tag="ConstraintOptimizer",
    )
    enable_flags["constraint_optimizer"] = en

    # ---- LSTM2（可选）----
    lstm2, _, en = _build_optional(
        fusion_cfg, "lstm2", build_lstm2, fusion_cfg_dir, tag="LSTM2",
    )
    enable_flags["lstm2"] = en

    # ---- GNN2（可选）----
    gnn2, _, en = _build_optional(
        fusion_cfg, "gnn2", build_gnn2, fusion_cfg_dir, tag="GNN2",
    )
    enable_flags["gnn2"] = en

    return (
        lstm1, gnn1, constraint, lstm2, gnn2,
        mean_A, std_A,
        lstm1_modes, top_k,
        enable_flags,
    )


def load_fusion_config(fusion_cfg_path: Path) -> Tuple[Dict[str, Any], Path]:
    """读取 fusion 自己的 config.yaml。返回 (fusion_cfg_dict, fusion_cfg_dir)。"""
    fusion_cfg = _load_yaml(fusion_cfg_path)
    return fusion_cfg, fusion_cfg_path.resolve().parent
