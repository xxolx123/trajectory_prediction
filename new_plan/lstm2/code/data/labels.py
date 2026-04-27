"""
lstm2/code/data/labels.py
-------------------------
意图（4 类）+ 威胁度（0..100 整数）的硬规则标注。

特征基于 refined（路网约束后）的预测轨迹 + 我方固定目标 position：
    pos_t = traj[t, :3]    (km)
    vel_t = traj[t, 3:6]   (km/s)
    d_t   = ||pos_t - position||
    Δd    = d_{T-1} - d_0
    closing = -Δd / (T * dt)   (km/s)   > 0 表示靠近我方
    Δheading_t = wrap(atan2(vy_t, vx_t) - atan2(vy_{t-1}, vx_{t-1}))
    mean_curv  = mean(|Δheading_t|)
    total_turn = wrap(heading_end - heading_start)
    v_mean     = mean(||vel_t||)

意图按"优先级硬规则"判定（严格按下面顺序）：
    RETREAT (3) :
        Δd > τ_retreat_dist  OR  (|total_turn| > τ_uturn AND closing <= 0)
    ATTACK  (0) :
        Δd < -τ_attack_dist  AND  mean_curv < τ_evasion_curv
    EVASION (1) :
        mean_curv >= τ_evasion_curv
    DEFENSE (2) :
        其余兜底

威胁度（4 项加权）：
    intent_base ∈ {1.0, 0.8, 0.6, 0.2} 对应 {ATTACK, EVASION, DEFENSE, RETREAT}
    speed_norm   = clip((v_mean - v_min)/(v_max - v_min), 0, 1)
    closing_norm = clip(closing / closing_max, 0, 1)        # 只取正向
    prox_norm    = 1 - clip(d_end / d_max, 0, 1)            # 末帧越近越高

    threat_0_1 = (w_I*intent_base + w_S*speed_norm + w_C*closing_norm + w_P*prox_norm)
                 / (w_I + w_S + w_C + w_P)
    threat_score = round(100 * threat_0_1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


# ===== 意图编号 ========================================================

INTENT_ATTACK = 0
INTENT_EVASION = 1
INTENT_DEFENSE = 2
INTENT_RETREAT = 3

INTENT_NAME = {
    INTENT_ATTACK: "ATTACK",
    INTENT_EVASION: "EVASION",
    INTENT_DEFENSE: "DEFENSE",
    INTENT_RETREAT: "RETREAT",
}


# ===== 配置 ============================================================

@dataclass
class LabelConfig:
    """从 lstm2/config.yaml -> data.intent_threat 段构造。"""
    # 阈值
    tau_retreat_dist_km: float = 1.5
    tau_attack_dist_km: float = 1.5
    tau_uturn_deg: float = 120.0
    tau_evasion_curv_deg: float = 8.0      # 平均逐步转角阈值（每步）

    # 时间步长（秒），用来把 closing(km/帧累积) 化成 km/s
    time_step_s: float = 60.0

    # 速度归一区间
    v_min_kmps: float = 0.0055      # ≈20 km/h
    v_max_kmps: float = 0.0210      # ≈75 km/h

    # 距离/接近归一
    closing_max_kmps: float = 0.020
    d_max_km: float = 5.0

    # 意图基础威胁
    base_attack: float = 1.0
    base_evasion: float = 0.8
    base_defense: float = 0.6
    base_retreat: float = 0.2

    # 威胁度权重
    w_intent: float = 0.40
    w_speed: float = 0.20
    w_closing: float = 0.25
    w_prox: float = 0.15

    @classmethod
    def from_dict(cls, d: Dict[str, Any], time_step_s: float = 60.0) -> "LabelConfig":
        return cls(
            tau_retreat_dist_km=float(d.get("tau_retreat_dist_km", 1.5)),
            tau_attack_dist_km=float(d.get("tau_attack_dist_km", 1.5)),
            tau_uturn_deg=float(d.get("tau_uturn_deg", 120.0)),
            tau_evasion_curv_deg=float(d.get("tau_evasion_curv_deg", 8.0)),
            time_step_s=float(time_step_s),
            v_min_kmps=float(d.get("v_min_kmps", 0.0055)),
            v_max_kmps=float(d.get("v_max_kmps", 0.0210)),
            closing_max_kmps=float(d.get("closing_max_kmps", 0.020)),
            d_max_km=float(d.get("d_max_km", 5.0)),
            base_attack=float(d.get("base_attack", 1.0)),
            base_evasion=float(d.get("base_evasion", 0.8)),
            base_defense=float(d.get("base_defense", 0.6)),
            base_retreat=float(d.get("base_retreat", 0.2)),
            w_intent=float(d.get("w_intent", 0.40)),
            w_speed=float(d.get("w_speed", 0.20)),
            w_closing=float(d.get("w_closing", 0.25)),
            w_prox=float(d.get("w_prox", 0.15)),
        )


# ===== 工具 ============================================================

def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    """把任意角度规约到 [-pi, pi] 区间，支持向量化。"""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _intent_base(intent: int, cfg: LabelConfig) -> float:
    if intent == INTENT_ATTACK:
        return cfg.base_attack
    if intent == INTENT_EVASION:
        return cfg.base_evasion
    if intent == INTENT_DEFENSE:
        return cfg.base_defense
    return cfg.base_retreat   # INTENT_RETREAT


# ===== 单样本 ==========================================================

def compute_intent_threat(
    refined: np.ndarray,           # [T, 6] 物理 km / km/s
    position: np.ndarray,          # [3]   km
    cfg: LabelConfig,
) -> Tuple[int, int]:
    """
    对一条 refined 轨迹 + 我方固定目标 position 计算 (intent_label, threat_score)。

    Returns:
        intent_label: int  0..3
        threat_score: int  0..100
    """
    if refined.ndim != 2 or refined.shape[-1] < 6:
        raise ValueError(f"refined 形状应为 [T, 6]，实际 {refined.shape}")
    if position.shape != (3,):
        raise ValueError(f"position 形状应为 [3]，实际 {position.shape}")

    T = refined.shape[0]
    pos = refined[:, 0:3].astype(np.float64)
    vel = refined[:, 3:6].astype(np.float64)

    # 距离序列
    d = np.linalg.norm(pos - position.reshape(1, 3), axis=-1)   # [T]
    d_start = float(d[0])
    d_end = float(d[-1])
    delta_d = d_end - d_start                                   # km
    if T > 1:
        closing = -delta_d / (T * cfg.time_step_s) * 1.0        # 单位 km/s
    else:
        closing = 0.0

    # 航向 / 曲率
    heading = np.arctan2(vel[:, 1], vel[:, 0])                  # [T]
    if T >= 2:
        dh = _wrap_angle(heading[1:] - heading[:-1])            # [T-1]
        mean_curv = float(np.mean(np.abs(dh)))
        total_turn = float(_wrap_angle(np.array([heading[-1] - heading[0]]))[0])
        total_turn = abs(total_turn)
    else:
        mean_curv = 0.0
        total_turn = 0.0

    v_speed = np.linalg.norm(vel, axis=-1)                      # [T]
    v_mean = float(np.mean(v_speed))

    tau_uturn_rad = math.radians(cfg.tau_uturn_deg)
    tau_evasion_curv_rad = math.radians(cfg.tau_evasion_curv_deg)

    # ---- 意图（按优先级硬规则）----
    if (delta_d > cfg.tau_retreat_dist_km) or (
        total_turn > tau_uturn_rad and closing <= 0.0
    ):
        intent = INTENT_RETREAT
    elif (delta_d < -cfg.tau_attack_dist_km) and (mean_curv < tau_evasion_curv_rad):
        intent = INTENT_ATTACK
    elif mean_curv >= tau_evasion_curv_rad:
        intent = INTENT_EVASION
    else:
        intent = INTENT_DEFENSE

    # ---- 威胁度 ----
    if cfg.v_max_kmps > cfg.v_min_kmps:
        speed_norm = (v_mean - cfg.v_min_kmps) / (cfg.v_max_kmps - cfg.v_min_kmps)
    else:
        speed_norm = 0.0
    speed_norm = float(min(max(speed_norm, 0.0), 1.0))

    if cfg.closing_max_kmps > 0.0:
        closing_norm = closing / cfg.closing_max_kmps
    else:
        closing_norm = 0.0
    closing_norm = float(min(max(closing_norm, 0.0), 1.0))

    if cfg.d_max_km > 0.0:
        prox_norm = 1.0 - d_end / cfg.d_max_km
    else:
        prox_norm = 0.0
    prox_norm = float(min(max(prox_norm, 0.0), 1.0))

    base = _intent_base(intent, cfg)
    w_sum = cfg.w_intent + cfg.w_speed + cfg.w_closing + cfg.w_prox
    if w_sum <= 0.0:
        threat_0_1 = float(min(max(base, 0.0), 1.0))
    else:
        threat_0_1 = (
            cfg.w_intent * base
            + cfg.w_speed * speed_norm
            + cfg.w_closing * closing_norm
            + cfg.w_prox * prox_norm
        ) / w_sum
        threat_0_1 = float(min(max(threat_0_1, 0.0), 1.0))

    threat_score = int(round(100.0 * threat_0_1))
    return int(intent), threat_score


# ===== 批量版 ==========================================================

def compute_intent_threat_batch(
    refined: np.ndarray,           # [N, T, 6]
    position: np.ndarray,          # [N, 3]
    cfg: LabelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化版（仅 4 项特征都向量化算，最终判定还是 per-sample 的硬规则）。

    Returns:
        intent_labels: [N] int8
        threat_scores: [N] int16
    """
    if refined.ndim != 3 or refined.shape[-1] < 6:
        raise ValueError(f"refined 形状应为 [N, T, 6]，实际 {refined.shape}")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError(f"position 形状应为 [N, 3]，实际 {position.shape}")
    if refined.shape[0] != position.shape[0]:
        raise ValueError(
            f"refined batch={refined.shape[0]} 与 position batch={position.shape[0]} 不一致"
        )

    N, T, _ = refined.shape
    pos = refined[..., 0:3].astype(np.float64)
    vel = refined[..., 3:6].astype(np.float64)
    posn = position.astype(np.float64).reshape(N, 1, 3)

    d = np.linalg.norm(pos - posn, axis=-1)                    # [N, T]
    d_end = d[:, -1]                                           # [N]
    delta_d = d_end - d[:, 0]                                  # [N]
    if T > 1:
        closing = -delta_d / (T * cfg.time_step_s)             # [N] km/s
    else:
        closing = np.zeros(N, dtype=np.float64)

    heading = np.arctan2(vel[..., 1], vel[..., 0])             # [N, T]
    if T >= 2:
        dh = _wrap_angle(heading[:, 1:] - heading[:, :-1])     # [N, T-1]
        mean_curv = np.mean(np.abs(dh), axis=-1)               # [N]
        total_turn = np.abs(_wrap_angle(heading[:, -1] - heading[:, 0]))  # [N]
    else:
        mean_curv = np.zeros(N, dtype=np.float64)
        total_turn = np.zeros(N, dtype=np.float64)

    v_speed = np.linalg.norm(vel, axis=-1)                     # [N, T]
    v_mean = np.mean(v_speed, axis=-1)                         # [N]

    tau_uturn_rad = math.radians(cfg.tau_uturn_deg)
    tau_evasion_curv_rad = math.radians(cfg.tau_evasion_curv_deg)

    intent = np.full(N, INTENT_DEFENSE, dtype=np.int8)
    # RETREAT
    is_retreat = (delta_d > cfg.tau_retreat_dist_km) | (
        (total_turn > tau_uturn_rad) & (closing <= 0.0)
    )
    intent[is_retreat] = INTENT_RETREAT

    # ATTACK：仅在未判 RETREAT 的位置生效
    is_attack = (
        (~is_retreat)
        & (delta_d < -cfg.tau_attack_dist_km)
        & (mean_curv < tau_evasion_curv_rad)
    )
    intent[is_attack] = INTENT_ATTACK

    # EVASION：仅在未判 RETREAT/ATTACK 的位置生效
    is_evasion = (~is_retreat) & (~is_attack) & (mean_curv >= tau_evasion_curv_rad)
    intent[is_evasion] = INTENT_EVASION

    # 其余保持 DEFENSE 兜底

    # ---- threat ----
    if cfg.v_max_kmps > cfg.v_min_kmps:
        speed_norm = (v_mean - cfg.v_min_kmps) / (cfg.v_max_kmps - cfg.v_min_kmps)
    else:
        speed_norm = np.zeros(N, dtype=np.float64)
    speed_norm = np.clip(speed_norm, 0.0, 1.0)

    if cfg.closing_max_kmps > 0.0:
        closing_norm = closing / cfg.closing_max_kmps
    else:
        closing_norm = np.zeros(N, dtype=np.float64)
    closing_norm = np.clip(closing_norm, 0.0, 1.0)

    if cfg.d_max_km > 0.0:
        prox_norm = 1.0 - d_end / cfg.d_max_km
    else:
        prox_norm = np.zeros(N, dtype=np.float64)
    prox_norm = np.clip(prox_norm, 0.0, 1.0)

    base = np.empty(N, dtype=np.float64)
    base[intent == INTENT_ATTACK] = cfg.base_attack
    base[intent == INTENT_EVASION] = cfg.base_evasion
    base[intent == INTENT_DEFENSE] = cfg.base_defense
    base[intent == INTENT_RETREAT] = cfg.base_retreat

    w_sum = cfg.w_intent + cfg.w_speed + cfg.w_closing + cfg.w_prox
    if w_sum <= 0.0:
        threat_0_1 = np.clip(base, 0.0, 1.0)
    else:
        threat_0_1 = (
            cfg.w_intent * base
            + cfg.w_speed * speed_norm
            + cfg.w_closing * closing_norm
            + cfg.w_prox * prox_norm
        ) / w_sum
        threat_0_1 = np.clip(threat_0_1, 0.0, 1.0)

    threat_score = np.rint(100.0 * threat_0_1).astype(np.int16)
    return intent, threat_score


__all__ = [
    "INTENT_ATTACK",
    "INTENT_EVASION",
    "INTENT_DEFENSE",
    "INTENT_RETREAT",
    "INTENT_NAME",
    "LabelConfig",
    "compute_intent_threat",
    "compute_intent_threat_batch",
]
