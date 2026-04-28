"""
gnn2/code/train/model.py
------------------------
StrikeZoneNet：打击区域 + 置信度预测（per-trajectory，不真用 GNN）。

业务接口（与 fusion 现状对齐，不变）::

    输入:
      pred_traj  [B, T, 6]   ConstraintOptimizer 输出的"路网约束后预测轨迹"
                            （fusion 里 B = batch * top_k = batch * 3，三条候选各跑一次）
      eta        [B]         我方预计到达时间（int64 秒；模型内部归一化为小时）

    输出:
      strike_pos    [B, 3]   打击区域中心 (x, y, z) km
      strike_radius [B, 1]   打击半径 km，radius_min <= r <= radius_max（softplus + clamp）
      strike_conf   [B, 1]   置信度 (0..1，sigmoid)

实现：
    平移不变化输入 (xyz - 首帧)
        → Linear(6, d_emb) + learnable PE
        → FiLM(ETA) 调制 (γ, β = MLP(eta_hours))
        → TransformerEncoder × n_layers
        → mean pool over T
        → MLP head → 5 维 (3 pos_rel + 1 radius_raw + 1 conf_logit)
        → pos_abs = pos_rel + 首帧 / radius = clamp(softplus(raw)+r_min, r_max)
        / conf = sigmoid(logit)

双实现 + ckpt 互通：
    StrikeZoneNet         —— 库版 nn.TransformerEncoder（GPU 快，需 opset >= 14）
    StrikeZoneNetManual   —— 手写 attention + encoder layer（opset 11 兼容）
    state_dict key 严格一致；训练用前者，导 ONNX 用后者，加载同一份 ckpt。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 公共数学逻辑：输入预处理 + 输出头（SDPA / Manual 共用）
# ============================================================

def _build_pred_traj_relative(pred_traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pred_traj [B, T, 6] -> (traj_rel [B, T, 6], ref [B, 3])
    平移不变化：xyz 减去首帧 xyz；vel 不动。
    用 torch.cat 而不是 in-place 切片赋值，确保 ONNX 可导。
    """
    if pred_traj.ndim != 3 or pred_traj.shape[-1] != 6:
        raise ValueError(
            f"pred_traj 形状应为 [B, T, 6]，实际 {tuple(pred_traj.shape)}"
        )
    xyz = pred_traj[..., 0:3]
    vel = pred_traj[..., 3:6]
    ref = xyz[:, 0, :]                                  # [B, 3]
    xyz_rel = xyz - ref.unsqueeze(1)                    # [B, T, 3]
    traj_rel = torch.cat([xyz_rel, vel], dim=-1)        # [B, T, 6]
    return traj_rel, ref


# ============================================================
# Manual attention（opset 11 兼容；与 nn.MultiheadAttention 数值等价、ckpt 互通）
# ============================================================
#
# 与 lstm2/code/train/model.py 的 _ManualMultiheadAttention 同款实现：
#   - 用 matmul / softmax / linear 显式表达 SDPA
#   - split_heads 分两步 reshape 避开 aten::unflatten（opset >= 13）
#   - state_dict keys 严格匹配 nn.MultiheadAttention：
#       in_proj_weight / in_proj_bias / out_proj.weight / out_proj.bias
# ============================================================


class _ManualMultiheadAttention(nn.Module):
    """与 nn.MultiheadAttention(batch_first=True) 数值等价、参数布局一致的手写版本。"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.dropout_p = float(dropout)

        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.d_model, self.d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.d_model))
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # 直接 view([B, T, h, hd]) 在 PyTorch ONNX export 下会被识别为
        # aten::unflatten（opset >= 13），与 mindspore-lite 1.8.1 (opset 11) 不兼容。
        # 拆两步：[B*T, h, hd] -> [B, T, h, hd]，trace 后是两个普通 Reshape，opset 11 全支持。
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            flat = t.reshape(B * T, self.nhead, self.head_dim)
            return flat.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)                                # [B, h, T, hd]

        # 同样拆两步合并 head；reshape([B, T, D]) 直接到 3D 也会被识别成 unflatten。
        out = out.transpose(1, 2).contiguous()                     # [B, T, h, hd]
        out = out.reshape(B * T, D).reshape(B, T, D)
        return self.out_proj(out)


class _ManualEncoderLayer(nn.Module):
    """
    pre-LN + GELU FFN，结构与 nn.TransformerEncoderLayer(norm_first=True) 一致。

    state_dict keys（与库版严格匹配）::
        self_attn.{in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias}
        linear1.{weight, bias}        FFN 第一层（d_model -> ffn_dim）
        linear2.{weight, bias}        FFN 第二层（ffn_dim -> d_model）
        norm1.{weight, bias}          attention 前的 LN
        norm2.{weight, bias}          FFN 前的 LN
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = _ManualMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        x = x + self.dropout2(
            self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        )
        return x


class _ManualTransformerEncoder(nn.Module):
    """state_dict keys: layers.{i}.<EncoderLayer keys>"""

    def __init__(self, layer_factory, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================
# 共用模块封装：Linear + PE + ETA-FiLM + Head + radius/conf 后处理
# ============================================================

class _StrikeZoneCommon(nn.Module):
    """
    SDPA / Manual 共用的输入处理 + 输出头。

    state_dict 命名与最终模型保持稳定（不会因为换 SDPA / Manual 影响到这些 key）::

        input_proj.{weight, bias}
        pos_embed                           [max_seq_len, d_emb]
        eta_film.0.{weight, bias}           Linear(1, d_emb)
        eta_film.2.{weight, bias}           Linear(d_emb, 2*d_emb)
        head.0.{weight, bias}               Linear(d_emb, d_emb)
        head.3.{weight, bias}               Linear(d_emb, 5)

        # buffers
        radius_min, radius_max
    """

    def __init__(
        self,
        feat_dim: int,
        d_emb: int,
        dropout: float,
        eta_scale_seconds: float,
        radius_min_km: float,
        radius_max_km: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if radius_max_km <= radius_min_km:
            raise ValueError(
                f"radius_max_km({radius_max_km}) 必须 > radius_min_km({radius_min_km})"
            )

        self.feat_dim = int(feat_dim)
        self.d_emb = int(d_emb)
        self.max_seq_len = int(max_seq_len)
        self.eta_scale_seconds = float(eta_scale_seconds)

        self.input_proj = nn.Linear(self.feat_dim, self.d_emb)

        self.pos_embed = nn.Parameter(
            torch.zeros(self.max_seq_len, self.d_emb)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ETA → (gamma, beta) 双倍宽度，对 d_emb 做 FiLM
        self.eta_film = nn.Sequential(
            nn.Linear(1, self.d_emb),
            nn.GELU(),
            nn.Linear(self.d_emb, 2 * self.d_emb),
        )
        # gamma 初始化为 0 → 训练初期等价 "FiLM 失活"，让模型先学 ETA 无关的 baseline
        nn.init.zeros_(self.eta_film[2].weight)
        nn.init.zeros_(self.eta_film[2].bias)

        # 头：5 维 = 3 (pos_rel) + 1 (radius raw) + 1 (conf logit)
        self.head = nn.Sequential(
            nn.Linear(self.d_emb, self.d_emb),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_emb, 5),
        )

        # radius 上下限作 buffer，随 ckpt 一起保存，避免推理时和训练 GT 单位不一致
        self.register_buffer(
            "radius_min", torch.tensor(float(radius_min_km), dtype=torch.float32),
        )
        self.register_buffer(
            "radius_max", torch.tensor(float(radius_max_km), dtype=torch.float32),
        )

    def pre_encode(
        self,
        pred_traj: torch.Tensor,           # [B, T, 6]
        eta: torch.Tensor,                 # [B] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x [B, T, d_emb], ref [B, 3])。"""
        B, T, _ = pred_traj.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"序列长度 T={T} 超出 max_seq_len={self.max_seq_len}"
            )
        if eta.ndim != 1 or int(eta.shape[0]) != B:
            raise ValueError(
                f"eta 形状应为 [B={B}] long，实际 {tuple(eta.shape)}"
            )

        traj_rel, ref = _build_pred_traj_relative(pred_traj)        # [B, T, 6], [B, 3]

        x = self.input_proj(traj_rel)                                # [B, T, d_emb]
        x = x + self.pos_embed[:T].unsqueeze(0)                      # 广播

        eta_h = (eta.to(dtype=x.dtype) / self.eta_scale_seconds).unsqueeze(-1)  # [B, 1]
        gamma_beta = self.eta_film(eta_h)                            # [B, 2*d_emb]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                    # [B, d_emb] x 2
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return x, ref

    def post_decode(
        self,
        h_pooled: torch.Tensor,            # [B, d_emb]
        ref: torch.Tensor,                 # [B, 3]
    ) -> Dict[str, torch.Tensor]:
        raw = self.head(h_pooled)                                    # [B, 5]
        pos_abs = raw[:, 0:3] + ref                                  # [B, 3]
        # radius = clamp(softplus + r_min, r_max)；cap 用 r_max - r_min（softplus 之后再加 r_min）
        cap = (self.radius_max - self.radius_min).clamp_min(1e-6)
        radius = self.radius_min + F.softplus(raw[:, 3:4]).clamp(max=cap)
        conf = torch.sigmoid(raw[:, 4:5])
        return {
            "strike_pos":    pos_abs,
            "strike_radius": radius,
            "strike_conf":   conf,
        }


# ============================================================
# 库版（默认，训练快）
# ============================================================

class StrikeZoneNet(_StrikeZoneCommon):
    """
    使用 nn.TransformerEncoder（内部走 SDPA，opset >= 14）。
    state_dict 与 StrikeZoneNetManual 完全一致，可互相 load。
    """

    def __init__(
        self,
        fut_len: int,
        feat_dim: int,
        d_emb: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int,
        eta_scale_seconds: float,
        radius_min_km: float,
        radius_max_km: float,
    ) -> None:
        super().__init__(
            feat_dim=feat_dim,
            d_emb=d_emb,
            dropout=dropout,
            eta_scale_seconds=eta_scale_seconds,
            radius_min_km=radius_min_km,
            radius_max_km=radius_max_km,
            max_seq_len=max_seq_len,
        )
        self.fut_len = int(fut_len)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_emb,
            nhead=int(nhead),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(
        self,
        pred_traj: torch.Tensor,        # [B, T, 6]
        eta: torch.Tensor,              # [B] long
    ) -> Dict[str, torch.Tensor]:
        x, ref = self.pre_encode(pred_traj, eta)
        h = self.encoder(x).mean(dim=1)                                # [B, d_emb]
        return self.post_decode(h, ref)


# ============================================================
# 手写版（opset 11 兼容；与库版数值等价、ckpt 严格兼容）
# ============================================================

class StrikeZoneNetManual(_StrikeZoneCommon):
    """
    使用 _ManualTransformerEncoder（仅基础算子 matmul/softmax/linear/layernorm/gelu/reshape）。
    state_dict 与 StrikeZoneNet 完全一致，训练好的 ckpt 直接 load 即可，无需重训。

    用途：
      - 部署目标 ONNX opset <= 13（含 mindspore-lite 1.8.1 默认 opset 11）。
      - fusion/code/build.py 在 fusion.config.gnn2.manual_attention=true 时强制注入。
    """

    def __init__(
        self,
        fut_len: int,
        feat_dim: int,
        d_emb: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int,
        eta_scale_seconds: float,
        radius_min_km: float,
        radius_max_km: float,
    ) -> None:
        super().__init__(
            feat_dim=feat_dim,
            d_emb=d_emb,
            dropout=dropout,
            eta_scale_seconds=eta_scale_seconds,
            radius_min_km=radius_min_km,
            radius_max_km=radius_max_km,
            max_seq_len=max_seq_len,
        )
        self.fut_len = int(fut_len)

        def _layer_factory():
            return _ManualEncoderLayer(
                d_model=self.d_emb,
                nhead=int(nhead),
                dim_feedforward=int(ffn_dim),
                dropout=float(dropout),
            )

        self.encoder = _ManualTransformerEncoder(_layer_factory, int(num_layers))

    def forward(
        self,
        pred_traj: torch.Tensor,
        eta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x, ref = self.pre_encode(pred_traj, eta)
        h = self.encoder(x).mean(dim=1)
        return self.post_decode(h, ref)


# ============================================================
# 工厂
# ============================================================

def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    按 cfg.model.type 构造模型；fusion 在 fusion.config.gnn2.manual_attention=true 时
    会把 cfg.model.manual_attention 注入为 True，工厂据此强制走 _manual 分支。

    返回 StrikeZoneNet（SDPA 版）或 StrikeZoneNetManual（手写版）；两版 state_dict 互通。
    """
    m = cfg.get("model", {}) or {}
    data = cfg.get("data", {}) or {}
    ctx = cfg.get("context", {}) or {}

    common_kwargs = dict(
        fut_len=int(m.get("fut_len", 10)),
        feat_dim=int(m.get("feat_dim", 6)),
        d_emb=int(m.get("d_emb", 64)),
        nhead=int(m.get("nhead", 4)),
        num_layers=int(m.get("num_layers", 2)),
        ffn_dim=int(m.get("ffn_dim", 128)),
        dropout=float(m.get("dropout", 0.1)),
        max_seq_len=int(m.get("max_seq_len", 16)),
        eta_scale_seconds=float(ctx.get("eta_scale_seconds", 3600.0)),
        radius_min_km=float(data.get("radius_min_km", 0.5)),
        radius_max_km=float(data.get("radius_max_km", 10.0)),
    )

    type_str = str(m.get("type", "strike_zone_transformer")).lower()
    # fusion 端注入开关：true → 强制 _manual（覆盖 type）
    if bool(m.get("manual_attention", False)):
        type_str = "strike_zone_transformer_manual"

    if type_str == "strike_zone_transformer":
        return StrikeZoneNet(**common_kwargs)
    if type_str == "strike_zone_transformer_manual":
        return StrikeZoneNetManual(**common_kwargs)

    raise ValueError(
        f"未知 model.type='{type_str}'；可选 "
        f"strike_zone_transformer / strike_zone_transformer_manual"
    )


__all__ = [
    "StrikeZoneNet",
    "StrikeZoneNetManual",
    "build_model_from_config",
]
