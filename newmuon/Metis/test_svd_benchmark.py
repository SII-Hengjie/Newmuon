#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比几种 SVD low-rank 算子与 full SVD 的精度 & 开销。

矩阵来源：
    ROOT_DIR   = /inspire/hdd/project/yunweiyuhuifu/p-shangli/quant/gpt/visual_ckpt/visual
    使用 FN_GRAD = warmup_linear_weight_grad.pt

对比的算子：
  - full SVD (torch.linalg.svd, 截断到 rank=k)
  - torch.svd_lowrank
  - lr.svd_lowrank_eig
  - lr.svd_lowrank_eig_graph
  - lr.svd_lowrank_eig_graph_pipelined

精度指标：
  - 前 k 个奇异值的逐元素相对误差 & 平均相对误差
  - 前 k 个左奇异向量 U 的“最大匹配”相似度（|cos|）
  - 前 k 个右奇异向量 V 的“最大匹配”相似度（|cos|）

时间指标：
  - GPU event time（核函数区段）
  - wall-clock time
"""

import time
import math
import importlib
from pathlib import Path

import numpy as np
import torch

# ======== 根据你的实现文件名修改这里 ========
# 假定你的实现文件叫 lowrank_eig.py，且在同级目录
import lowrank_eig as lr


# ======== 数据路径配置 ========
ROOT_DIR = Path("/inspire/hdd/project/yunweiyuhuifu/p-shangli/quant/gpt/visual_ckpt/visual")

FN_GRAD = ROOT_DIR / "warmup_linear_weight_grad.pt"  # 本脚本使用的矩阵


# ======== 性能相关开关 ========
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# rank 和 oversampling
RANK_K = 32
OVERSAMPLE = 8   # q = k + OVERSAMPLE


# ======== 工具函数 ========

def _cuda_sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(xs):
    xs = np.array(xs, dtype=float)
    return dict(
        mean=float(xs.mean()),
        std=float(xs.std(ddof=0)),
        p50=float(np.percentile(xs, 50)),
        p95=float(np.percentile(xs, 95)),
        n=int(xs.size),
    )


@torch.no_grad()
def _dummy_use(*tensors):
    # 防止编译器过度优化；不把张量拷回 CPU
    s = 0.0
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            s += float(t.flatten()[0].item())
    return s


def load_matrix_from_pt(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    尝试从 .pt 文件中提取一个 2D Tensor：
      - 如果本身是 Tensor，直接用
      - 如果是 dict，优先从常见 key 中找 tensor
    """
    print(f"[load] Loading tensor from: {path}")
    obj = torch.load(path, map_location=device)

    if isinstance(obj, torch.Tensor):
        mat = obj
    elif isinstance(obj, dict):
        for key in ("tensor", "data", "grad", "weight", "value"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                print(f"[load] Using tensor from key '{key}' in dict.")
                mat = obj[key]
                break
        else:
            # fallback：找第一个 Tensor
            for key, val in obj.items():
                if isinstance(val, torch.Tensor):
                    print(f"[load] Using first Tensor value from key '{key}' in dict.")
                    mat = val
                    break
            else:
                raise TypeError(f"No Tensor found in dict loaded from {path}. Keys={list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported object type loaded from {path}: {type(obj)}")

    if mat.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(mat.shape)} from {path}")
    mat = mat.to(device=device, dtype=dtype).contiguous()
    print(f"[load] Matrix shape={tuple(mat.shape)}, dtype={mat.dtype}, device={mat.device}")
    return mat


def full_svd_wrapper(A: torch.Tensor, q: int, niter=None, M=None):
    """
    统一接口：返回截断到 q 的 (U, S, V)，与低秩算子对齐。
    """
    if M is not None:
        A_eff = A - M
    else:
        A_eff = A

    U, S, Vh = torch.linalg.svd(A_eff, full_matrices=False)
    U = U[:, :q]
    S = S[:q]
    V = Vh.mH[:, :q]
    return U, S, V


def get_torch_svd_lowrank():
    """
    兼容 torch.svd_lowrank / torch.linalg.svd_lowrank 的 helper。
    """
    if hasattr(torch, "svd_lowrank"):
        return torch.svd_lowrank
    elif hasattr(torch.linalg, "svd_lowrank"):
        return torch.linalg.svd_lowrank
    else:
        raise RuntimeError("This PyTorch build has no svd_lowrank / linalg.svd_lowrank.")


def bench_op(fn, A: torch.Tensor, name: str, q: int, niter: int, *, build_warmup: int = 0):
    """
    统一基准：同时收集 GPU event 时间（算子纯 GPU 核心耗时）与 wall-clock。
    - build_warmup：仅用于 graph 版的额外“建图”预热（不计入统计）
    """
    WARMUP_COMMON = 5
    RUNS = 50

    # 公共 warmup（触发 cuBLAS/cuSOLVER 初始化等）
    for _ in range(WARMUP_COMMON):
        U, S, V = fn(A, q, niter, None)
        _dummy_use(U, S, V)
    _cuda_sync()

    # graph 版：额外建图 warmup
    for _ in range(build_warmup):
        U, S, V = fn(A, q, niter, None)
        _dummy_use(U, S, V)
    _cuda_sync()

    gpu_ms_list = []
    wall_ms_list = []

    start_evt = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    stop_evt = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    for _ in range(RUNS):
        if start_evt is not None:
            start_evt.record()
        t0 = time.perf_counter()

        U, S, V = fn(A, q, niter, None)

        if stop_evt is not None:
            stop_evt.record()
        _cuda_sync()
        t1 = time.perf_counter()

        if start_evt is not None and stop_evt is not None:
            gpu_ms = start_evt.elapsed_time(stop_evt)
        else:
            gpu_ms = (t1 - t0) * 1000.0

        wall_ms = (t1 - t0) * 1000.0
        gpu_ms_list.append(gpu_ms)
        wall_ms_list.append(wall_ms)

        _dummy_use(U, S, V)

    gpu_stats = _stats(gpu_ms_list)
    wall_stats = _stats(wall_ms_list)

    print(f"\n[{name}]  GPU(ms): mean={gpu_stats['mean']:.3f}, std={gpu_stats['std']:.3f}, "
          f"p50={gpu_stats['p50']:.3f}, p95={gpu_stats['p95']:.3f}, n={gpu_stats['n']}")
    print(f"[{name}] WALL(ms): mean={wall_stats['mean']:.3f}, std={wall_stats['std']:.3f}, "
          f"p50={wall_stats['p50']:.3f}, p95={wall_stats['p95']:.3f}, n={wall_stats['n']}")

    return gpu_stats, wall_stats


# ======== 相似度 & 误差计算 ========

def _relative_sv_errors(S_approx: torch.Tensor, S_ref: torch.Tensor, k: int):
    """
    奇异值相对误差：|S_approx - S_ref| / (|S_ref| + eps)
    返回：max_rel, mean_rel
    """
    S_ref_k = S_ref[:k]
    S_approx_k = S_approx[:k]
    eps = 1e-12
    rel = (S_approx_k - S_ref_k).abs() / (S_ref_k.abs() + eps)
    rel_np = rel.detach().cpu().numpy()
    return float(rel_np.max()), float(rel_np.mean())


def _max_matching_sim_greedy(sim_mat: np.ndarray):
    """
    退化版最大匹配：贪心取当前最大值并删行删列。
    sim_mat: (k, k) 的相似度矩阵，值为 [0,1]。
    返回匹配的相似度列表 sims。
    """
    k = sim_mat.shape[0]
    sims = []
    used_rows = set()
    used_cols = set()
    for _ in range(k):
        best = -1.0
        best_i = best_j = None
        for i in range(k):
            if i in used_rows:
                continue
            for j in range(k):
                if j in used_cols:
                    continue
                if sim_mat[i, j] > best:
                    best = sim_mat[i, j]
                    best_i, best_j = i, j
        if best_i is None:
            break
        sims.append(best)
        used_rows.add(best_i)
        used_cols.add(best_j)
    return sims


def _max_matching_sim(sim_mat: np.ndarray):
    """
    优先使用 SciPy 的 Hungarian 算最大匹配；
    若没安装 SciPy，则退化为贪心匹配。

    sim_mat: (k, k)，相似度矩阵（越大越好）。
    返回：sims(list)
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        # Hungarian 是最小化 cost，这里 cost = -sim
        row_ind, col_ind = linear_sum_assignment(-sim_mat)
        sims = [sim_mat[i, j] for i, j in zip(row_ind, col_ind)]
        return sims
    except Exception:
        print("[warn] SciPy not available; fall back to greedy matching.")
        return _max_matching_sim_greedy(sim_mat)


def singular_vector_similarity(U_ref: torch.Tensor,
                               U_approx: torch.Tensor,
                               k: int):
    """
    计算前 k 个奇异向量的“最大匹配”相似度（|cos|），
    返回：mean_sim, min_sim。
    """
    U_r = U_ref[:, :k]
    U_a = U_approx[:, :k]

    # 理论上已经是正交基，这里再做一次归一化以防数值误差
    U_r = torch.nn.functional.normalize(U_r, dim=0)
    U_a = torch.nn.functional.normalize(U_a, dim=0)

    # 相似度矩阵：C[i,j] = |<u_i, \hat{u}_j>|
    C = (U_r.mT @ U_a).abs().detach().cpu().numpy()  # (k, k)

    sims = _max_matching_sim(C)  # list length k
    sims = np.array(sims, dtype=float)
    return float(sims.mean()), float(sims.min()), sims


# ======== 主流程 ========

def main():
    print(f"Device={device}, dtype={dtype}")
    # A = load_matrix_from_pt(FN_GRAD, device=device, dtype=dtype)
    A = torch.randn(8192, 4096, device=device, dtype=torch.float32)
    m, n = A.shape
    k = min(RANK_K, m, n)
    q = min(k + OVERSAMPLE, m, n)
    niter = 1

    print(f"\n[config] shape=({m},{n}), k={k}, q={q}, niter={niter}")

    # ---------- 先算一遍 full SVD，作为“真值” ----------
    print("\n[full SVD] computing reference SVD ...")
    with torch.no_grad():
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
    U_ref = U_full[:, :k]
    S_ref = S_full[:k]
    V_ref = Vh_full.mH[:, :k]
    print("[full SVD] done.")

    # ---------- 准备各个算子 ----------
    torch_svd_lowrank = get_torch_svd_lowrank()

    # 统一包装成 (A, q, niter, M) 签名
    def op_full(A_, q_, niter_, M_):
        return full_svd_wrapper(A_, q_, niter_, M_)

    def op_torch_svd_lowrank(A_, q_, niter_, M_):
        return torch_svd_lowrank(A_, q=q_, niter=niter_, M=M_)

    def op_eig(A_, q_, niter_, M_):
        return lr.svd_lowrank_eig(A_, q=q_, niter=niter_, M=M_)

    def op_graph(A_, q_, niter_, M_):
        return lr.svd_lowrank_eig_graph(A_, q=q_, niter=niter_, M=M_)

    def op_graph_pipe(A_, q_, niter_, M_):
        return lr.svd_lowrank_eig_graph_pipelined(A_, q=q_, niter=niter_, M=M_)

    algos = [
        ("full_svd_truncated", op_full, 0),
        ("torch_svd_lowrank", op_torch_svd_lowrank, 0),
        ("svd_lowrank_eig", op_eig, 0),
    ]

    if device.type == "cuda":
        # graph 版需要 CUDA
        algos.append(("svd_lowrank_eig_graph", op_graph, 1))          # build_warmup=1
        algos.append(("svd_lowrank_eig_graph_pipelined", op_graph_pipe, 1))

    # ---------- 精度 + 时间对比 ----------
    results = {}

    print("\n===== 精度 + 开销对比（相对于 full SVD 前 k 个分量）=====")
    for name, fn, build_warmup in algos:
        print(f"\n=== Algo: {name} ===")

        # 一次前向，拿结果用于精度评估
        with torch.no_grad():
            U_hat, S_hat, V_hat = fn(A, q, niter, None)

        # 精度评估（只看前 k 个）
        # 奇异值相对误差
        sv_max_rel, sv_mean_rel = _relative_sv_errors(S_hat, S_ref, k)

        # 左奇异向量 U 相似度
        u_mean_sim, u_min_sim, _ = singular_vector_similarity(U_ref, U_hat, k)

        # 右奇异向量 V 相似度
        v_mean_sim, v_min_sim, _ = singular_vector_similarity(V_ref, V_hat, k)

        print(f"[{name}] SV relative error: max={sv_max_rel:.3e}, mean={sv_mean_rel:.3e}")
        print(f"[{name}] U similarity (max-matching): mean={u_mean_sim:.4f}, min={u_min_sim:.4f}")
        print(f"[{name}] V similarity (max-matching): mean={v_mean_sim:.4f}, min={v_min_sim:.4f}")

        # 时间基准
        gpu_stats, wall_stats = bench_op(fn, A, name, q=q, niter=niter, build_warmup=build_warmup)

        results[name] = dict(
            sv_max_rel=sv_max_rel,
            sv_mean_rel=sv_mean_rel,
            u_mean_sim=u_mean_sim,
            u_min_sim=u_min_sim,
            v_mean_sim=v_mean_sim,
            v_min_sim=v_min_sim,
            gpu=gpu_stats,
            wall=wall_stats,
        )

    # ---------- 简要汇总 ----------
    print("\n===== Summary (vs full SVD, k = {}) =====".format(k))
    for name in results:
        r = results[name]
        print(f"\n[{name}]")
        print(f"  SV rel error: max={r['sv_max_rel']:.3e}, mean={r['sv_mean_rel']:.3e}")
        print(f"  U sim (mean/min): {r['u_mean_sim']:.4f} / {r['u_min_sim']:.4f}")
        print(f"  V sim (mean/min): {r['v_mean_sim']:.4f} / {r['v_min_sim']:.4f}")
        print(f"  GPU mean(ms): {r['gpu']['mean']:.3f}, WALL mean(ms): {r['wall']['mean']:.3f}")

    # 若有 graph 版，给个 speedup 对比
    if "svd_lowrank_eig_graph" in results:
        g = results["svd_lowrank_eig_graph"]
        eig = results["svd_lowrank_eig"]
        svd = results["torch_svd_lowrank"]
        print("\n== Speedup (mean, graph vs others) ==")
        print(f"graph vs eig  | GPU: {eig['gpu']['mean']/g['gpu']['mean']:.2f}×  | "
              f"WALL: {eig['wall']['mean']/g['wall']['mean']:.2f}×")
        print(f"graph vs svd  | GPU: {svd['gpu']['mean']/g['gpu']['mean']:.2f}×  | "
              f"WALL: {svd['wall']['mean']/g['wall']['mean']:.2f}×")
    if "svd_lowrank_eig_graph_pipelined" in results:
        gp = results["svd_lowrank_eig_graph_pipelined"]
        eig = results["svd_lowrank_eig"]
        print(f"graph_pipelined vs eig | GPU: {eig['gpu']['mean']/gp['gpu']['mean']:.2f}×  | "
              f"WALL: {eig['wall']['mean']/gp['wall']['mean']:.2f}×")


if __name__ == "__main__":
    main()
