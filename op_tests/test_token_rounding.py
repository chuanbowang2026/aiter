#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Token Rounding Routing analysis and benchmark for MoE
# Based on SonicMoE paper Section 5, Algorithm 4

import argparse
import math
import sys
import os
import torch
import torch.nn.functional as F
import time


def analyze_tile_waste(topk_ids, num_experts, block_size_M):
    """Analyze how much compute is wasted due to partial tiles in Grouped GEMM."""
    E = num_experts
    M, topk = topk_ids.shape

    expert_counts = torch.zeros(E, dtype=torch.int32, device=topk_ids.device)
    flat = topk_ids.view(-1)
    for e in range(E):
        expert_counts[e] = (flat == e).sum()

    total_padded = 0
    total_wasted = 0

    for e in range(E):
        f_e = expert_counts[e].item()
        if f_e == 0:
            continue
        padded = math.ceil(f_e / block_size_M) * block_size_M
        total_padded += padded
        total_wasted += padded - f_e

    waste_pct = 100.0 * total_wasted / total_padded if total_padded > 0 else 0
    return {"total_padded": total_padded, "total_wasted": total_wasted, "waste_pct": waste_pct}


def analyze_tile_waste_with_rounding(topk_ids, num_experts, block_size_M):
    """Analyze waste after applying token rounding (nearest multiple)."""
    E = num_experts
    flat = topk_ids.view(-1)

    expert_counts = torch.zeros(E, dtype=torch.int32, device=topk_ids.device)
    for e in range(E):
        expert_counts[e] = (flat == e).sum()

    total_padded_rounded = 0
    tokens_dropped = 0
    tokens_padded = 0

    for e in range(E):
        f_e = expert_counts[e].item()
        if f_e == 0:
            continue
        ceil_f = math.ceil(f_e / block_size_M) * block_size_M
        floor_f = (f_e // block_size_M) * block_size_M
        if floor_f == 0:
            rounded_f = ceil_f
        elif (ceil_f - f_e) <= (f_e - floor_f):
            rounded_f = ceil_f
        else:
            rounded_f = floor_f
        total_padded_rounded += rounded_f
        if rounded_f < f_e:
            tokens_dropped += f_e - rounded_f
        elif rounded_f > f_e:
            tokens_padded += rounded_f - f_e

    waste_pct = 100.0 * tokens_padded / total_padded_rounded if total_padded_rounded > 0 else 0
    return {
        "total_padded_rounded": total_padded_rounded,
        "tokens_dropped": tokens_dropped,
        "tokens_padded": tokens_padded,
        "waste_pct": waste_pct,
    }


def simulate_routing(T, E, K, device="cpu"):
    """Simulate MoE routing: generate topk_ids with realistic distribution."""
    logits = torch.randn(T, E, device=device)
    _, topk_ids = logits.topk(K, dim=-1)
    return topk_ids


def run_analysis(device="cpu"):
    """Run tile waste analysis across common MoE configurations."""
    configs = [
        ("Llama3-MoE", 1, 8, 2, 32),
        ("Llama3-MoE", 1024, 8, 2, 32),
        ("DeepSeek-V3 dec", 1, 256, 8, 32),
        ("DeepSeek-V3 dec", 64, 256, 8, 32),
        ("DeepSeek-V3 pf", 1024, 256, 8, 32),
        ("DeepSeek-V3 pf", 4096, 256, 8, 32),
        ("Qwen3-MoE dec", 1, 128, 8, 32),
        ("Qwen3-MoE dec", 64, 128, 8, 32),
        ("Qwen3-MoE pf", 1024, 128, 8, 32),
        ("Qwen3-MoE pf", 4096, 128, 8, 32),
        ("Kimi-K2 tp4", 1, 384, 8, 32),
        ("Kimi-K2 tp4", 1024, 384, 8, 32),
    ]

    print("=" * 110)
    print(f"{'Model':<20} {'T':>6} {'E':>4} {'K':>3} {'BS':>3} | "
          f"{'Cur Waste%':>10} {'Cur Wasted':>10} | "
          f"{'TR Waste%':>9} {'Dropped':>8} {'Padded':>8} | "
          f"{'Savings':>8}")
    print("=" * 110)

    for name, T, E, K, bs in configs:
        topk_ids = simulate_routing(T, E, K, device=device)
        current = analyze_tile_waste(topk_ids, E, bs)
        rounded = analyze_tile_waste_with_rounding(topk_ids, E, bs)
        savings = current["waste_pct"] - rounded["waste_pct"]
        print(f"{name:<20} {T:>6} {E:>4} {K:>3} {bs:>3} | "
              f"{current['waste_pct']:>9.1f}% {current['total_wasted']:>10} | "
              f"{rounded['waste_pct']:>8.1f}% {rounded['tokens_dropped']:>8} {rounded['tokens_padded']:>8} | "
              f"{savings:>7.1f}%")


def run_benchmark(device="cuda"):
    """Benchmark fused_moe with and without token rounding on GPU."""
    try:
        import aiter
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
    except ImportError:
        print("aiter not importable, skipping GPU benchmark")
        return

    configs = [
        # (name, T, d, n, E, K)
        ("Qwen3-MoE", 1024, 4096, 1536, 128, 8),
        ("Qwen3-MoE", 4096, 4096, 1536, 128, 8),
        ("DeepSeek-V3", 1024, 7168, 2048, 256, 8),
        ("DeepSeek-V3", 4096, 7168, 2048, 256, 8),
    ]

    print("\n" + "=" * 90)
    print("GPU Benchmark: fused_moe with/without token rounding")
    print("=" * 90)
    print(f"{'Model':<16} {'T':>6} {'d':>5} {'n':>5} {'E':>4} {'K':>3} | "
          f"{'No TR (us)':>10} {'TR (us)':>10} {'Speedup':>8}")
    print("-" * 90)

    warmup = 5
    repeat = 20

    for name, T, d, n, E, K in configs:
        torch.manual_seed(42)
        hidden = torch.randn(T, d, dtype=torch.bfloat16, device=device)
        w1 = torch.randn(E, 2 * n, d, dtype=torch.bfloat16, device=device)
        w2 = torch.randn(E, d, n, dtype=torch.bfloat16, device=device)

        logits = torch.randn(T, E, device=device)
        topk_weights, topk_ids = logits.topk(K, dim=-1)
        topk_weights = topk_weights.softmax(dim=-1).float()
        topk_ids = topk_ids.int()

        for _ in range(warmup):
            fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                      activation=ActivationType.Silu, enable_token_rounding=False)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeat):
            fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                      activation=ActivationType.Silu, enable_token_rounding=False)
        torch.cuda.synchronize()
        no_tr_us = (time.perf_counter() - t0) / repeat * 1e6

        for _ in range(warmup):
            fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                      activation=ActivationType.Silu, enable_token_rounding=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeat):
            fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                      activation=ActivationType.Silu, enable_token_rounding=True)
        torch.cuda.synchronize()
        tr_us = (time.perf_counter() - t0) / repeat * 1e6

        speedup = no_tr_us / tr_us if tr_us > 0 else 0
        print(f"{name:<16} {T:>6} {d:>5} {n:>5} {E:>4} {K:>3} | "
              f"{no_tr_us:>10.1f} {tr_us:>10.1f} {speedup:>7.3f}x")


def run_correctness_test(device="cuda"):
    """Test that token rounding produces close output to non-rounded."""
    try:
        import aiter
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
    except ImportError:
        print("aiter not importable, skipping correctness test")
        return

    print("\n" + "=" * 70)
    print("Correctness test: fused_moe with/without token rounding")
    print("=" * 70)

    configs = [
        ("Qwen3-MoE", 1024, 4096, 1536, 128, 8),
        ("DeepSeek-V3", 1024, 7168, 2048, 256, 8),
    ]

    for name, T, d, n, E, K in configs:
        torch.manual_seed(42)
        hidden = torch.randn(T, d, dtype=torch.bfloat16, device=device)
        w1 = torch.randn(E, 2 * n, d, dtype=torch.bfloat16, device=device)
        w2 = torch.randn(E, d, n, dtype=torch.bfloat16, device=device)

        logits = torch.randn(T, E, device=device)
        topk_weights, topk_ids = logits.topk(K, dim=-1)
        topk_weights = topk_weights.softmax(dim=-1).float()
        topk_ids = topk_ids.int()

        out_no_tr = fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                              activation=ActivationType.Silu, enable_token_rounding=False)
        out_tr = fused_moe(hidden, w1, w2, topk_weights, topk_ids,
                           activation=ActivationType.Silu, enable_token_rounding=True)

        max_diff = (out_no_tr - out_tr).abs().max().item()
        rel_diff = ((out_no_tr - out_tr).abs() / (out_no_tr.abs() + 1e-6)).mean().item()
        print(f"{name:<16} T={T} E={E} K={K}: max_diff={max_diff:.6f} rel_diff={rel_diff:.6f}")
        if max_diff < 0.1:
            print(f"  PASS (small difference expected due to dropped tokens)")
        else:
            print(f"  NOTE: large diff expected when tokens are dropped by rounding")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Rounding Routing Analysis & Benchmark")
    parser.add_argument("--analysis", action="store_true", help="Run tile waste analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run GPU benchmark")
    parser.add_argument("--correctness", action="store_true", help="Run correctness test")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not any([args.analysis, args.benchmark, args.correctness]):
        args.analysis = True

    if args.analysis:
        run_analysis(device=args.device)

    if args.benchmark:
        run_benchmark(device="cuda")

    if args.correctness:
        run_correctness_test(device="cuda")
