#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Unit test for token_rounding_routing function

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch flydsl before any aiter import
try:
    import flydsl.compiler.protocol
    flydsl.compiler.protocol.extract_to_ir_values = lambda *a, **k: None
except:
    pass

os.environ["AITER_USE_SYSTEM_TRITON"] = "1"

import torch
import math

from aiter.fused_moe import token_rounding_routing


def create_sorted_layout(topk_ids, topk_weights, num_experts, block_size_M):
    """Simulate what moe_sorting produces (CPU reference).

    The real moe_sorting kernel stores indices into the flattened topk_ids
    (range 0..T*topk-1). Padding slots get sentinel value = T (num_tokens).
    The GEMM kernel divides by topk to get the actual token row.
    """
    M, topk = topk_ids.shape
    flat_ids = topk_ids.view(-1)
    flat_weights = topk_weights.view(-1)

    expert_counts = []
    expert_token_lists = []
    for e in range(num_experts):
        mask = flat_ids == e
        indices = torch.where(mask)[0]
        expert_counts.append(len(indices))
        expert_token_lists.append(indices)

    sorted_ids_list = []
    sorted_weights_list = []
    sorted_expert_ids_list = []

    for e in range(num_experts):
        f_e = expert_counts[e]
        if f_e == 0:
            continue
        padded = math.ceil(f_e / block_size_M) * block_size_M
        num_blocks = padded // block_size_M

        token_indices = expert_token_lists[e]
        weights = flat_weights[token_indices]

        pad_ids = torch.full((padded,), M, dtype=torch.int32)
        pad_weights = torch.zeros(padded, dtype=torch.float32)
        pad_ids[:f_e] = token_indices.int()
        pad_weights[:f_e] = weights.float()

        sorted_ids_list.append(pad_ids)
        sorted_weights_list.append(pad_weights)
        sorted_expert_ids_list.extend([e] * num_blocks)

    sorted_ids = torch.cat(sorted_ids_list) if sorted_ids_list else torch.empty(0, dtype=torch.int32)
    sorted_weights = torch.cat(sorted_weights_list) if sorted_weights_list else torch.empty(0, dtype=torch.float32)
    sorted_expert_ids = torch.tensor(sorted_expert_ids_list, dtype=torch.int32)
    num_valid_ids = torch.tensor([len(sorted_ids), 0], dtype=torch.int32)

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, expert_counts


def test_no_change_when_already_aligned():
    """If all experts have token counts that are multiples of block_size, no change should occur."""
    print("Test: no change when already aligned...", end=" ")
    T, E, K, bs = 256, 8, 2, 32
    # Force exact multiples: 256*2/8 = 64 tokens per expert on average
    topk_ids = torch.arange(T * K).reshape(T, K) % E
    topk_weights = torch.ones(T, K, dtype=torch.float32)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        create_sorted_layout(topk_ids, topk_weights, E, bs)

    new_ids, new_weights, new_expert_ids, new_valid = token_rounding_routing(
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk_ids, topk_weights, E, bs, K)

    assert new_valid[0].item() == num_valid_ids[0].item(), \
        f"Should not change: {new_valid[0].item()} vs {num_valid_ids[0].item()}"
    print("PASS")


def test_rounding_reduces_padding():
    """Token rounding should reduce total padded slots."""
    print("Test: rounding reduces padding...", end=" ")
    T, E, K, bs = 1024, 128, 8, 32
    torch.manual_seed(42)
    logits = torch.randn(T, E)
    _, topk_ids = logits.topk(K, dim=-1)
    topk_weights = torch.randn(T, K).softmax(dim=-1)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, expert_counts = \
        create_sorted_layout(topk_ids, topk_weights, E, bs)

    original_padded = num_valid_ids[0].item()

    new_ids, new_weights, new_expert_ids, new_valid = token_rounding_routing(
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk_ids, topk_weights, E, bs, K)

    new_padded = new_valid[0].item()

    print(f"original={original_padded} -> rounded={new_padded} "
          f"(saved {original_padded - new_padded} slots, "
          f"{100*(original_padded - new_padded)/original_padded:.1f}%) ", end="")

    assert new_padded <= original_padded, "Rounding should not increase padding"
    assert new_padded < original_padded, "With E=128, K=8, some rounding should happen"
    print("PASS")


def test_sorted_expert_ids_consistent():
    """Verify sorted_expert_ids matches the actual token layout."""
    print("Test: sorted_expert_ids consistent...", end=" ")
    T, E, K, bs = 512, 64, 4, 32
    torch.manual_seed(123)
    logits = torch.randn(T, E)
    _, topk_ids = logits.topk(K, dim=-1)
    topk_weights = torch.randn(T, K).softmax(dim=-1)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        create_sorted_layout(topk_ids, topk_weights, E, bs)

    new_ids, new_weights, new_expert_ids, new_valid = token_rounding_routing(
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk_ids, topk_weights, E, bs, K)

    total_slots = new_valid[0].item()
    num_blocks = total_slots // bs
    assert len(new_expert_ids) == num_blocks, \
        f"Expert IDs length {len(new_expert_ids)} != num_blocks {num_blocks}"

    for b in range(num_blocks):
        e = new_expert_ids[b].item()
        assert 0 <= e < E, f"Invalid expert id {e} at block {b}"

    # Check expert ids are non-decreasing
    for b in range(1, num_blocks):
        assert new_expert_ids[b] >= new_expert_ids[b-1], \
            f"Expert ids not sorted: block {b-1}={new_expert_ids[b-1]}, block {b}={new_expert_ids[b]}"

    print("PASS")


def test_decode_no_change():
    """For decode (T=1), rounding should not drop any tokens."""
    print("Test: decode (T=1) no token drop...", end=" ")
    T, E, K, bs = 1, 256, 8, 32
    torch.manual_seed(42)
    logits = torch.randn(T, E)
    _, topk_ids = logits.topk(K, dim=-1)
    topk_weights = torch.randn(T, K).softmax(dim=-1)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        create_sorted_layout(topk_ids, topk_weights, E, bs)

    new_ids, new_weights, new_expert_ids, new_valid = token_rounding_routing(
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk_ids, topk_weights, E, bs, K)

    # For T=1, each expert gets at most 1 token, floor=0, so all round up
    assert new_valid[0].item() == num_valid_ids[0].item(), \
        "Decode should not change anything (all experts have <=1 token)"
    print("PASS")


def test_weights_preserved():
    """Ensure kept tokens retain their original weights."""
    print("Test: weights preserved for kept tokens...", end=" ")
    T, E, K, bs = 256, 32, 4, 32
    torch.manual_seed(42)
    logits = torch.randn(T, E)
    _, topk_ids = logits.topk(K, dim=-1)
    topk_weights = torch.randn(T, K).softmax(dim=-1)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = \
        create_sorted_layout(topk_ids, topk_weights, E, bs)

    new_ids, new_weights, new_expert_ids, new_valid = token_rounding_routing(
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk_ids, topk_weights, E, bs, K)

    # For each non-padding token in new output, its weight should match the original
    flat_orig_weights = topk_weights.view(-1).float()
    total_slots = new_valid[0].item()
    sentinel = T  # padding sentinel used by moe_sorting
    for i in range(total_slots):
        tid = new_ids[i].item()
        if tid >= sentinel:
            continue
        w_new = new_weights[i].item()
        w_orig = flat_orig_weights[tid].item()
        assert abs(w_new - w_orig) < 1e-5, \
            f"Weight mismatch at slot {i} (tid={tid}): new={w_new:.6f} orig={w_orig:.6f}"

    print("PASS")


if __name__ == "__main__":
    test_no_change_when_already_aligned()
    test_rounding_reduces_padding()
    test_sorted_expert_ids_consistent()
    test_decode_no_change()
    test_weights_preserved()
    print("\nAll tests passed!")
