"""
constraint_optimizer/test_road_net/test_arc_projection_vectorized.py
--------------------------------------------------------------------
对照单测：向量化版 `_road_arc_projection` 与循环版 `_road_arc_projection_loop`
的输出在前缀连续 True 形态的 mask 下应当数学等价（差异仅来自浮点累加顺序）。

覆盖三组场景：
  1) mask 全 False                     → 应直接 fallback 等于输入
  2) 单分支 + 单方向                   → 与循环版 1e-5 量级一致
  3) 多分支 + 双向（含反向最优）       → 与循环版 1e-5 量级一致
  4) 部分样本无 valid branch           → 该样本 fallback，其它样本仍走投影

用法（LSTM_traj_predict 环境内）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m constraint_optimizer.test_road_net.test_arc_projection_vectorized

正常退出码 = 0；任一断言失败退出码非 0。
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../new_plan
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.context_schema import ContextBatch  # noqa: E402
from constraint_optimizer.code.train.module import ConstraintOptimizer  # noqa: E402


# ============================================================
# 工具
# ============================================================

def _zeros_ctx_extras(N: int, device, dtype=torch.float32) -> dict:
    """ConstraintOptimizer 仅消费 road_points / road_mask，其它字段补占位。"""
    return dict(
        task_type=torch.zeros(N, dtype=torch.long, device=device),
        type=torch.zeros(N, dtype=torch.long, device=device),
        position=torch.zeros(N, 3, dtype=dtype, device=device),
        eta=torch.zeros(N, dtype=torch.long, device=device),
    )


def _make_branch(
    pts: torch.Tensor,            # [Nv, 3]
    NP_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把紧致折线 pts 填到 [NP_max, 3] 槽位（前缀连续 True，其余 False）。"""
    Nv = int(pts.shape[0])
    if Nv > NP_max:
        raise ValueError(f"Nv={Nv} > NP_max={NP_max}")
    full = torch.zeros(NP_max, 3, dtype=pts.dtype, device=pts.device)
    mask = torch.zeros(NP_max, dtype=torch.bool, device=pts.device)
    full[:Nv] = pts
    mask[:Nv] = True
    return full, mask


def _build_traj_along(
    branch_pts: torch.Tensor,     # [Nv, 3]
    T: int,
    noise_std: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """
    沿一条 polyline 等弧长采 T 个点，再加一点法向噪声，让投影测试更有意义。
    返回 [T, 6]（pos + 简单匀速向量）。
    """
    g = torch.Generator(device=branch_pts.device).manual_seed(seed)
    seg_vec = branch_pts[1:] - branch_pts[:-1]
    seg_len = torch.linalg.norm(seg_vec, dim=-1).clamp_min(1e-9)
    cum_s = torch.cat(
        [torch.zeros(1, dtype=branch_pts.dtype, device=branch_pts.device), seg_len.cumsum(0)],
        dim=0,
    )
    total_s = cum_s[-1]
    target_s = torch.linspace(0.0, float(total_s), T, dtype=branch_pts.dtype, device=branch_pts.device)
    j_idx = torch.searchsorted(cum_s, target_s, right=True) - 1
    j_idx = j_idx.clamp(0, branch_pts.shape[0] - 2)
    seg_l = seg_len[j_idx]
    t_local = ((target_s - cum_s[j_idx]) / seg_l.clamp_min(1e-9)).clamp(0.0, 1.0)
    pos = branch_pts[j_idx] + t_local.unsqueeze(-1) * (branch_pts[j_idx + 1] - branch_pts[j_idx])
    pos = pos + torch.randn(pos.shape, generator=g, dtype=pos.dtype, device=pos.device) * noise_std
    vel = torch.zeros_like(pos)
    if T > 1:
        vel[1:] = pos[1:] - pos[:-1]
    return torch.cat([pos, vel], dim=-1)


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# ============================================================
# 测试
# ============================================================

def case_all_false_mask() -> None:
    """mask 全 False → 向量化 / 循环版都应直接 fallback 等于输入。"""
    print("[Case 1] mask 全 False → fallback")
    N, T, NB, NP = 3, 10, 2, 8
    pos = torch.randn(N, T, 6)
    rp = torch.zeros(N, NB, NP, 3)
    rm = torch.zeros(N, NB, NP, dtype=torch.bool)
    ctx = ContextBatch(road_points=rp, road_mask=rm, **_zeros_ctx_extras(N, pos.device))

    co = ConstraintOptimizer(enable=True, module_type="road_arc_projection").eval()

    with torch.no_grad():
        out_v = co._road_arc_projection(pos, ctx)
        out_l = co._road_arc_projection_loop(pos, ctx)

    assert torch.equal(out_v, pos), "向量化版未 fallback"
    assert torch.equal(out_l, pos), "循环版未 fallback"
    print("  OK\n")


def case_single_branch_single_direction() -> None:
    """单分支 + 正方向：向量化版 vs 循环版 max diff < 1e-5。"""
    print("[Case 2] 单分支 + 单方向")
    torch.manual_seed(42)

    NP = 12
    Nv = 8
    branch = torch.tensor([[float(i) * 1.0, 0.5 * float(i) ** 1.05, 0.0] for i in range(Nv)],
                          dtype=torch.float32)
    rp_full, rm_full = _make_branch(branch, NP)

    N, T = 4, 10
    rp = rp_full.unsqueeze(0).unsqueeze(0).expand(N, 1, NP, 3).clone()
    rm = rm_full.unsqueeze(0).unsqueeze(0).expand(N, 1, NP).clone()

    pos_list = [_build_traj_along(branch, T, noise_std=0.07, seed=s) for s in range(N)]
    pos = torch.stack(pos_list, dim=0)

    ctx = ContextBatch(road_points=rp, road_mask=rm, **_zeros_ctx_extras(N, pos.device))
    co = ConstraintOptimizer(enable=True, module_type="road_arc_projection").eval()

    with torch.no_grad():
        out_v = co._road_arc_projection(pos, ctx)
        out_l = co._road_arc_projection_loop(pos, ctx)

    diff = _diff(out_v[..., :3], out_l[..., :3])
    print(f"  max|Δpos| = {diff:.3e}")
    assert diff < 1e-4, f"单分支单方向差异过大: {diff}"
    # 速度通道未被改动 → 应严格相等
    assert torch.equal(out_v[..., 3:], pos[..., 3:]), "向量化版误改了速度通道"
    print("  OK\n")


def case_multi_branch_with_reverse() -> None:
    """多分支 + 双向（含反向最优）：向量化版 vs 循环版 max diff < 1e-4。"""
    print("[Case 3] 多分支 + 双向（含反向最优）")
    torch.manual_seed(123)

    NP = 12
    branch_a = torch.tensor([[float(i), 0.0, 0.0] for i in range(8)], dtype=torch.float32)
    branch_b = torch.tensor([[7.0 - float(i), float(i), 0.0] for i in range(7)], dtype=torch.float32)
    branch_c = torch.tensor([[float(i), float(i) * 0.3, 0.0] for i in range(6)], dtype=torch.float32)

    NB = 3
    rp_full = torch.zeros(NB, NP, 3)
    rm_full = torch.zeros(NB, NP, dtype=torch.bool)
    for bi, br in enumerate([branch_a, branch_b, branch_c]):
        f, m = _make_branch(br, NP)
        rp_full[bi] = f
        rm_full[bi] = m

    N, T = 5, 10
    rp = rp_full.unsqueeze(0).expand(N, NB, NP, 3).clone()
    rm = rm_full.unsqueeze(0).expand(N, NB, NP).clone()

    # 构造的 traj：
    #   样本 0~1: 沿 branch_a 正向
    #   样本 2:   沿 branch_a 反向
    #   样本 3:   沿 branch_b 正向
    #   样本 4:   沿 branch_c 正向
    pos_list = [
        _build_traj_along(branch_a, T, noise_std=0.06, seed=10),
        _build_traj_along(branch_a, T, noise_std=0.06, seed=11),
        _build_traj_along(branch_a.flip(0), T, noise_std=0.06, seed=12),
        _build_traj_along(branch_b, T, noise_std=0.06, seed=13),
        _build_traj_along(branch_c, T, noise_std=0.06, seed=14),
    ]
    pos = torch.stack(pos_list, dim=0)

    ctx = ContextBatch(road_points=rp, road_mask=rm, **_zeros_ctx_extras(N, pos.device))
    co = ConstraintOptimizer(enable=True, module_type="road_arc_projection").eval()

    with torch.no_grad():
        out_v = co._road_arc_projection(pos, ctx)
        out_l = co._road_arc_projection_loop(pos, ctx)

    diff = _diff(out_v[..., :3], out_l[..., :3])
    print(f"  max|Δpos| = {diff:.3e}")
    assert diff < 1e-4, f"多分支双向差异过大: {diff}"
    print("  OK\n")


def case_partial_invalid_samples() -> None:
    """部分样本无任何 valid branch（mask 全 False） → 该样本 fallback，其它正常投影。"""
    print("[Case 4] 部分样本无 valid branch")
    torch.manual_seed(7)

    NP = 10
    branch = torch.tensor([[float(i), float(i) ** 0.95, 0.0] for i in range(7)], dtype=torch.float32)
    rp_full, rm_full = _make_branch(branch, NP)

    N, T, NB = 4, 10, 1
    rp = rp_full.unsqueeze(0).unsqueeze(0).expand(N, NB, NP, 3).clone()
    rm = rm_full.unsqueeze(0).unsqueeze(0).expand(N, NB, NP).clone()
    # 把样本 1 / 3 的 mask 整片清成 False（模拟"该样本无路网"）
    rm[1] = False
    rm[3] = False

    pos = torch.stack(
        [_build_traj_along(branch, T, noise_std=0.05, seed=20 + s) for s in range(N)],
        dim=0,
    )

    ctx = ContextBatch(road_points=rp, road_mask=rm, **_zeros_ctx_extras(N, pos.device))
    co = ConstraintOptimizer(enable=True, module_type="road_arc_projection").eval()

    with torch.no_grad():
        out_v = co._road_arc_projection(pos, ctx)
        out_l = co._road_arc_projection_loop(pos, ctx)

    # 样本 1/3 应该 fallback 等于原 pos
    assert torch.equal(out_v[1, :, :3], pos[1, :, :3]), "向量化版样本 1 未 fallback"
    assert torch.equal(out_v[3, :, :3], pos[3, :, :3]), "向量化版样本 3 未 fallback"
    # 样本 0/2 应该和循环版一致
    diff_02 = max(
        _diff(out_v[0, :, :3], out_l[0, :, :3]),
        _diff(out_v[2, :, :3], out_l[2, :, :3]),
    )
    print(f"  max|Δpos| (样本 0/2) = {diff_02:.3e}")
    assert diff_02 < 1e-4, f"有效样本差异过大: {diff_02}"
    print("  OK\n")


def main() -> None:
    print("=" * 60)
    print("[Test] ConstraintOptimizer._road_arc_projection vs _road_arc_projection_loop")
    print("=" * 60)
    case_all_false_mask()
    case_single_branch_single_direction()
    case_multi_branch_with_reverse()
    case_partial_invalid_samples()
    print("=" * 60)
    print("[Test] all OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
