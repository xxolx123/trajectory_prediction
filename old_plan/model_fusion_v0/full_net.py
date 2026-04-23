#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_net.py
-----------
PyTorch 端的“轨迹 + 意图/威胁一体化”模型。

输入：
    x_raw: [B, Tin, 6]，原始物理量：
           [x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps]
           Tin 一般为 20。

内部流程（对齐轨迹训练 / eval 逻辑）：
  1) 轨迹子网（A 域）：
      - 对位置 (x,y,z) 做相邻差分 → Δx,Δy,Δz
          * 第 0 步位置增量为 0
          * 第 t 步为 pos[t] - pos[t-1]
      - 与速度 (vx,vy,vz) 拼成 6 维特征，再用
        scaler_posvel.npz 的 mean/std 做标准化
      - 喂入多模态 LSTMForecaster
      - 输出 [B,M,Tout,6] 的“未来 Δpos + 绝对速度”（仍在 A 域的特征空间）
      - 使用同一 mean/std 做反归一化，得到 Δpos_phys + v_phys
      - 对 Δpos 做沿时间 cumsum，并在历史最后一点绝对坐标上累加，
        得到未来 10 步的绝对物理坐标（世界坐标）

  2) 意图/威胁子网（B 域）：
      - 用 scaler_intent_posvel.npz 对 [x,y,z,vx,vy,vz] 做标准化
      - 对每个 mode，取“完整 20+10 条轨迹”的最后 window_len_B 步
        作为一个窗口，喂入 IntentThreatNet
      - 得到 logits_intent / threat_raw，再做:
          * intent_class = argmax(logits)
          * threat_prob = sigmoid(threat_raw)
          * mode_prob   = softmax(mode_logits)

  3) 输出：
      对每个 batch 的每个 mode，打包为 68 维向量：
          0..59 : 未来 10 步的 [x,y,z,vx,vy,vz] 展平成 60 维
          60    : intent_class (float，导出 ONNX 时方便)
          61    : threat_prob   (0~1)
          62..64: strike_pos_xyz（打击点的 [x,y,z]，即预测第 10 步的位置）
          65    : strike_radius（示例：由打击时刻速度线性映射得到）
          66    : conf（示例：速度越慢 & mode_prob 越大 → 置信度越高）
          67    : mode_prob（来自轨迹子网的 softmax）

      输出张量形状：
          out: [B, M, 68]
      ONNX 侧若拿到 [B, M*68] 也可以 reshape 回来。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class FullNet(nn.Module):
    def __init__(
        self,
        traj_net: nn.Module,
        intent_net: nn.Module,
        mean_A,
        std_A,
        mean_B,
        std_B,
        hist_len: int = 20,
        fut_len: int = 10,
        feature_dim: int = 6,
        window_len_B: int = 10,
        n_modes: int = 3,
        use_delta_A: bool = True,
        use_traj_logits: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            traj_net:      轨迹 LSTMForecaster（已加载好权重）
            intent_net:    IntentThreatNet（已加载好权重）
            mean_A, std_A: 轨迹训练用 scaler_posvel.npz 中的 mean/std（1D）
            mean_B, std_B: 意图训练用 scaler_intent_posvel.npz 中的 mean/std（1D）
            hist_len:      输入历史长度 Tin（一般 20）
            fut_len:       未来长度 Tout（一般 10）
            feature_dim:   特征维度（默认 6）
            window_len_B:  意图子网窗口长度（一般 10）
            n_modes:       多模态条数 M
            use_delta_A:   是否对 A 域位置采用 Δ 编码（本项目为 True）
            use_traj_logits: 是否输出 mode_logits 参与最终 68 维特征（保留接口）
        """
        super().__init__()
        self.traj_net = traj_net
        self.intent_net = intent_net

        self.hist_len = int(hist_len)
        self.fut_len = int(fut_len)
        self.feature_dim = int(feature_dim)
        self.window_len_B = int(window_len_B)
        self.n_modes = int(n_modes)
        self.use_delta_A = bool(use_delta_A)
        self.use_traj_logits = bool(use_traj_logits)

        # A 域 / B 域的 scaler（以 buffer 形式注册，导出 ONNX 时会一并固化）
        mean_A_t = torch.as_tensor(mean_A, dtype=torch.float32).view(1, 1, self.feature_dim)
        std_A_t = torch.as_tensor(std_A, dtype=torch.float32).view(1, 1, self.feature_dim)
        std_A_t = torch.where(std_A_t.abs() < 1e-6, torch.ones_like(std_A_t), std_A_t)

        mean_B_t = torch.as_tensor(mean_B, dtype=torch.float32).view(1, 1, self.feature_dim)
        std_B_t = torch.as_tensor(std_B, dtype=torch.float32).view(1, 1, self.feature_dim)
        std_B_t = torch.where(std_B_t.abs() < 1e-6, torch.ones_like(std_B_t), std_B_t)

        self.register_buffer("mean_A", mean_A_t)
        self.register_buffer("std_A", std_A_t)
        self.register_buffer("mean_B", mean_B_t)
        self.register_buffer("std_B", std_B_t)

        # 只在 x,y 上做位置评估 / 可视化
        self.pos_dims_xy: Tuple[int, int] = (0, 1)

    # ------------------------------------------------------------------
    # 一些小工具：A / B 域的归一化 / 反归一化
    # ------------------------------------------------------------------
    def _norm_A(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 6]
        return (x - self.mean_A) / self.std_A

    def _denorm_A(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 6]
        return x * self.std_A + self.mean_A

    def _norm_B(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_B) / self.std_B

    # ------------------------------------------------------------------
    # 1) 把原始物理量编码成“轨迹子网训练时的输入特征”（Δpos + v）
    # ------------------------------------------------------------------
    def _build_A_input_from_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [B, Tin, 6]，原始物理量。

        返回：
            x_A_norm: [B, Tin, 6]，已经过 Δ 编码 + 标准化，
                      可以直接喂给 LSTMForecaster。
        """
        B, T, D = x_raw.shape
        if D != self.feature_dim:
            raise ValueError(
                f"输入特征维度不一致: feature_dim={self.feature_dim}, 但 x_raw.shape[-1]={D}"
            )
        if T != self.hist_len:
            raise ValueError(
                f"历史长度不一致: hist_len={self.hist_len}, 但 x_raw.shape[1]={T}"
            )

        # 拆成位置与速度
        pos = x_raw[..., 0:3]   # [B,T,3]
        vel = x_raw[..., 3:6]   # [B,T,3]

        if self.use_delta_A:
            # 参照 traj_dataset._apply_delta_inplace 对 input 部分的规则：
            #   delta_pos[0]   = 0
            #   delta_pos[t]   = pos[t] - pos[t-1], t >= 1
            delta_pos = torch.zeros_like(pos)
            if T > 1:
                delta_pos[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
            pos_A = delta_pos
        else:
            # 不做 Δ 编码，直接用绝对坐标
            pos_A = pos

        feat_A_raw = torch.cat([pos_A, vel], dim=-1)  # [B,T,6]
        x_A_norm = self._norm_A(feat_A_raw)
        return x_A_norm

    # ------------------------------------------------------------------
    # 2) 把 LSTM 输出的“未来特征”（Δpos + v，已归一化）解码成绝对物理坐标
    # ------------------------------------------------------------------
    def _decode_future_traj(
        self,
        x_raw: torch.Tensor,          # [B,Tin,6] 原始历史
        fut_A_norm: torch.Tensor,     # [B,M,Tout,6] 归一化后的未来特征
    ) -> torch.Tensor:
        """
        返回：
            fut_full_phys: [B,M,Tout,6]，
                其中前 3 维是未来每步的绝对坐标 [x,y,z]（世界坐标），
                后 3 维是绝对速度 [vx,vy,vz]。
        """
        B, Tin, _ = x_raw.shape
        B2, M, Tout, D = fut_A_norm.shape

        if B2 != B or M != self.n_modes or Tout != self.fut_len or D != self.feature_dim:
            raise ValueError(
                "FullNet 配置与 traj_net 输出不一致: "
                f"B={B}, M={self.n_modes}, fut_len={self.fut_len}, feature_dim={self.feature_dim}, "
                f"但 fut_A_norm.shape={tuple(fut_A_norm.shape)}"
            )

        # 反归一化：得到 Δpos + v
        fut_A_raw = self._denorm_A(fut_A_norm)       # [B,M,Tout,6]
        fut_pos_delta = fut_A_raw[..., 0:3]          # [B,M,Tout,3]
        fut_vel_phys = fut_A_raw[..., 3:6]           # [B,M,Tout,3]

        # 历史最后一帧的“绝对位置”（世界坐标）
        last_hist_pos = x_raw[:, -1, 0:3]           # [B,3]
        last_hist_pos = last_hist_pos.view(B, 1, 1, 3)  # [B,1,1,3]

        if self.use_delta_A:
            # 和 eval.decode_batch_to_positions 一致：
            # future_pos = last_hist_pos + cumsum(Δpos, dim=2)
            fut_pos_cum = torch.cumsum(fut_pos_delta, dim=2)  # [B,M,Tout,3]
            fut_pos_phys = last_hist_pos + fut_pos_cum        # [B,M,Tout,3]
        else:
            # 若没有 Δ 编码，则 fut_pos_delta 本身就是绝对坐标
            fut_pos_phys = fut_pos_delta

        fut_full_phys = torch.cat([fut_pos_phys, fut_vel_phys], dim=-1)  # [B,M,Tout,6]
        return fut_full_phys

    # ------------------------------------------------------------------
    # 3) 构造意图/威胁子网的输入窗口并前向
    # ------------------------------------------------------------------
    def _run_intent_head(
        self,
        x_raw: torch.Tensor,          # [B,Tin,6] 历史绝对物理量
        fut_full_phys: torch.Tensor,  # [B,M,Tout,6] 未来绝对物理量
    ):
        """
        对每个 mode，取“20+10 条轨迹”的最后 window_len_B 步，
        按照意图任务的 scaler_intent_posvel 做标准化，喂入 IntentThreatNet。

        返回：
            intent_class: [B,M]  (long)
            threat_prob : [B,M]  (float, 0~1)
        """
        B, Tin, _ = x_raw.shape
        _, M, Tout, _ = fut_full_phys.shape

        # 拼成完整轨迹：history(20) + future(10)
        hist_full = x_raw.unsqueeze(1).expand(-1, M, -1, -1)          # [B,M,Tin,6]
        full_traj = torch.cat([hist_full, fut_full_phys], dim=2)     # [B,M,Tin+Tout,6]

        # 取最后 window_len_B 步：一般为 10
        if self.window_len_B <= Tin + Tout:
            win = full_traj[:, :, -self.window_len_B :, :]           # [B,M,win_L,6]
        else:
            # 若窗口比整体还长，就在前面补零（几乎不会发生）
            pad_len = self.window_len_B - (Tin + Tout)
            pad = torch.zeros(
                (B, M, pad_len, self.feature_dim),
                dtype=full_traj.dtype,
                device=full_traj.device,
            )
            win = torch.cat([pad, full_traj], dim=2)

        B_, M_, L, D = win.shape
        assert B_ == B and M_ == M and L == self.window_len_B and D == self.feature_dim

        # 归一化到 B 域，并 reshape 成 [B*M, L, 6]
        win_B = self._norm_B(win)
        win_B = win_B.view(B * M, L, D)  # [B*M,win_L,6]

        # IntentThreatNet 前向
        intent_out = self.intent_net(win_B)  # dict
        logits_intent = intent_out["logits_intent"]      # [B*M,4]
        threat_raw = intent_out["threat_raw"]           # [B*M,1]

        # 分类结果
        intent_class = torch.argmax(logits_intent, dim=-1)      # [B*M]
        intent_class = intent_class.view(B, M)

        # 威胁度概率（0~1）
        threat_prob = torch.sigmoid(threat_raw).view(B, M)

        return intent_class, threat_prob

    # ------------------------------------------------------------------
    # 4) 前向：一体化输出 [B,M,68]
    # ------------------------------------------------------------------
    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: [B, Tin, 6]，原始物理量
        Returns:
            out: [B, M, 68]
        """
        if x_raw.dim() != 3:
            raise ValueError(f"期望 x_raw 形状为 [B,T,6]，但得到 {x_raw.shape}")

        B, T, D = x_raw.shape
        if D != self.feature_dim:
            raise ValueError(
                f"输入特征维度不一致: feature_dim={self.feature_dim}, 但 x_raw.shape[-1]={D}"
            )
        if T != self.hist_len:
            raise ValueError(
                f"历史长度不一致: hist_len={self.hist_len}, 但 x_raw.shape[1]={T}"
            )

        x_raw = x_raw.to(self.mean_A.device).float()

        # 1) 构造 A 域输入（Δpos + v，已归一化）
        x_A_norm = self._build_A_input_from_raw(x_raw)          # [B,Tin,6]

        # 2) 轨迹子网：得到未来特征（仍在 A 域，归一化空间）
        fut_A_norm, mode_logits = self.traj_net(x_A_norm)       # [B,M,Tout,6], [B,M]

        # 3) 反归一化 + Δ→绝对坐标（世界坐标）
        fut_full_phys = self._decode_future_traj(x_raw, fut_A_norm)  # [B,M,Tout,6]

        # ---- 打击点位置 = 未来第 10 步的绝对坐标 [x,y,z] ----
        # 这里直接取 fut_full_phys 的最后一个时间步（索引 -1）
        strike_pos = fut_full_phys[:, :, -1, 0:3]  # [B,M,3]

        # 4) 轨迹概率（根据 mode_logits 做 softmax）
        mode_prob = F.softmax(mode_logits, dim=-1)  # [B,M]

        # 5) 意图 / 威胁子网
        intent_class, threat_prob = self._run_intent_head(x_raw, fut_full_phys)  # [B,M],[B,M]

        # 6) 其它几个信息（radius / conf 等）
        #    - 打击半径：和打击时刻的速度正相关，速度越快，半径越大
        #    - 置信度：和速度（或半径）负相关，并考虑 mode_prob，
        #              速度越慢、该 mode 概率越大 → 置信度越高

        # 打击时刻（第 10 步）的速度向量 [vx,vy,vz]
        strike_vel = fut_full_phys[:, :, -1, 3:6]        # [B,M,3]
        strike_speed = torch.norm(strike_vel, dim=-1)    # [B,M]  单位：km/s

        # ---- 打击半径：基于速度的简单线性模型（按需要可调整）----
        # 例如：基础半径 0.2 km + k * 速度
        base_R = 0.2   # km，基础半径，可根据业务调节
        k_R    = 1.0   # 速度系数，可根据业务调节
        radius = base_R + k_R * strike_speed             # [B,M]
        radius = torch.clamp(radius, min=0.0)

        # ---- 置信度：速度越慢 / 半径越小 → 置信度越高 ----
        # 参考速度，用来做归一化，比如 0.3 km/s (约 300 m/s)
        v_ref = 0.3
        speed_norm = strike_speed / (v_ref + 1e-6)       # [B,M]
        # 一个简单的单调递减函数：速度越小，conf_speed 越接近 1
        conf_speed = 1.0 / (1.0 + speed_norm)            # (0,1] 左右

        # 再叠加 mode_prob：该 mode 概率越大，我们越信任该预测
        conf = conf_speed * mode_prob                    # [B,M]

        # 7) 打包成 [B,M,68]
        #    前 60 维：flatten(10,6)，后 8 维为各种 scalar 信息。
        B, M, Tout, Df = fut_full_phys.shape
        assert Tout == self.fut_len and Df == self.feature_dim

        traj_flat = fut_full_phys.reshape(B, M, Tout * Df)  # [B,M,60]

        # intent_class / threat_prob / radius / conf / mode_prob 都转成 float，再拼接
        intent_f = intent_class.to(x_raw.dtype)           # [B,M]
        threat_f = threat_prob                            # [B,M]
        radius_f = radius                                 # [B,M]
        conf_f   = conf                                   # [B,M]
        mode_p_f = mode_prob                              # [B,M]

        # 拼接成 [B,M,68]
        # layout:
        #   0..59 : traj_flat
        #   60    : intent_class
        #   61    : threat_prob
        #   62    : strike_pos_x  （打击点 x / 经度）
        #   63    : strike_pos_y  （打击点 y / 纬度）
        #   64    : strike_pos_z  （打击点 z / 高度）
        #   65    : radius        （打击半径）
        #   66    : conf          （置信度）
        #   67    : mode_prob     （该 mode 的概率）
        tail_list = [
            intent_f.unsqueeze(-1),   # 60
            threat_f.unsqueeze(-1),   # 61
            strike_pos,               # 62..64: [B,M,3]
            radius_f.unsqueeze(-1),   # 65
            conf_f.unsqueeze(-1),     # 66
            mode_p_f.unsqueeze(-1),   # 67
        ]
        tail = torch.cat(tail_list, dim=-1)  # [B,M,8]

        out = torch.cat([traj_flat, tail], dim=-1)  # [B,M,60+8] = [B,M,68]
        return out


# ============================================================
# 兼容用工厂函数：供 export_full_net.py 调用
# ============================================================

def build_fullnet_from_config(
    traj_net: nn.Module,
    intent_net: nn.Module,
    mean_A,
    std_A,
    mean_B,
    std_B,
    hist_len: int = 20,
    fut_len: int = 10,
    feature_dim: int = 6,
    window_len_B: int = 10,
    n_modes: int = 3,
    use_delta_A: bool = True,
    use_traj_logits: bool = True,
    **kwargs,
) -> FullNet:
    """
    这是给 export_full_net.py 用的“底层构建函数”。

    注意：
      - export_full_net 会先从 config_mean_std.yaml 里解析出
        hist_len / fut_len / window_len_B / n_modes / use_delta_A 等，
        再从两个 npz 中读出 mean/std，然后通过这个函数来真正
        构造 FullNet。
      - 这里的参数设计得比较“宽松”，支持位置 / 关键字 + **kwargs，
        这样就算上层多传了一两个无关参数也不会出错。
    """
    return FullNet(
        traj_net=traj_net,
        intent_net=intent_net,
        mean_A=mean_A,
        std_A=std_A,
        mean_B=mean_B,
        std_B=std_B,
        hist_len=hist_len,
        fut_len=fut_len,
        feature_dim=feature_dim,
        window_len_B=window_len_B,
        n_modes=n_modes,
        use_delta_A=use_delta_A,
        use_traj_logits=use_traj_logits,
    )
