"""Direct test of cktile_moe_stage2 with gather-reduce.

Tests stage2 only, using pre-prepared a16w4 inputs.
Compares atomic path vs gather-reduce path.
"""
import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch
import aiter
from aiter import dtypes
from aiter.ops.moe_op import ActivationType, QuantType, moe_cktile2stages_gemm2
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    cktile_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight, shuffle_scale
from aiter.utility.fp4_utils import dynamic_mxfp4_quant

torch.set_default_device('cuda')
torch.manual_seed(42)

T = 32
E = 8
K = 2
H = 128  # model_dim (output)
I = 256  # inter_dim (input)
block_m = 32
dtype = dtypes.bf16

print(f'T={T}, E={E}, K={K}, H={H}, I={I}')

# Create routing
hidden = torch.randn(T, H, dtype=dtype)
gate_output = torch.randn(T, E, dtype=torch.float32)
topk_weights, topk_ids = fused_topk(hidden, gate_output, K, renormalize=True)
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, E, H, dtype, block_m
)
print(f'sorted_ids.shape={sorted_ids.shape}')

# Create stage2 input (simulate stage1 output)
# Shape: [T, K, I] or equivalent
a2 = torch.randn(T, K, I, dtype=dtype)

# Create and quantize w2 [E, H, I]
w2_bf16 = torch.randn(E, H, I, dtype=dtype)
w2_fp4_list, w2_scale_list = [], []
for e in range(E):
    q, s = dynamic_mxfp4_quant(w2_bf16[e])
    w2_fp4_list.append(q)
    w2_scale_list.append(s)
w2_fp4 = torch.stack(w2_fp4_list)     # [E, H, I/2]
w2_scale = torch.stack(w2_scale_list)  # [E, H, I/32]
print(f'w2_fp4.shape={w2_fp4.shape}, w2_scale.shape={w2_scale.shape}')

# Shuffle weights for CK tile preshuffled path
w2_shuf = shuffle_weight(w2_fp4, is_guinterleave=True, gate_up=False)
w2_shuf.is_shuffled = True

# Reshape scale for shuffle: need [E*H, I/32] -> shuffle -> back
w2_s_flat = w2_scale.view(dtypes.fp8_e8m0).reshape(E * H, -1)
w2_s_shuf = shuffle_scale(w2_s_flat, experts_cnt=E, is_guinterleave=True, gate_up=False)
print(f'w2_shuf.shape={w2_shuf.shape}, w2_s_shuf.shape={w2_s_shuf.shape}')

# Dummy w1 (not used in stage2)
w1_dummy = w2_shuf

# Test 1: Standard atomic path
print('\n=== Atomic path ===')
out_atomic = torch.zeros(T, H, dtype=dtype)
try:
    cktile_moe_stage2(
        a2, w1_dummy, w2_shuf,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out_atomic, K,
        w2_scale=w2_s_shuf, a2_scale=None,
        block_m=block_m, activation=ActivationType.Swiglu,
        sorted_weights=sorted_weights,
        n_pad_zeros=0, k_pad_zeros=0,
        use_gather_reduce=False,
    )
    print(f'Atomic norm: {out_atomic.norm().item():.4f}')
    print(f'Atomic[0,:4]: {out_atomic[0, :4]}')
    atomic_ok = True
except Exception as e:
    import traceback
    traceback.print_exc()
    atomic_ok = False

# Test 2: Gather-reduce path  
print('\n=== Gather-reduce path ===')
out_gr = torch.zeros(T, H, dtype=dtype)
try:
    cktile_moe_stage2(
        a2, w1_dummy, w2_shuf,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        out_gr, K,
        w2_scale=w2_s_shuf, a2_scale=None,
        block_m=block_m, activation=ActivationType.Swiglu,
        sorted_weights=sorted_weights,
        n_pad_zeros=0, k_pad_zeros=0,
        use_gather_reduce=True,
        topk_weights=topk_weights,
    )
    print(f'GR norm: {out_gr.norm().item():.4f}')
    print(f'GR[0,:4]: {out_gr[0, :4]}')
    gr_ok = True
except Exception as e:
    import traceback
    traceback.print_exc()
    gr_ok = False

# Compare
if atomic_ok and gr_ok:
    diff = (out_atomic - out_gr).abs().max().item()
    rdiff = diff / (out_atomic.abs().max().item() + 1e-6)
    print(f'\n=== Comparison ===')
    print(f'Abs diff: {diff:.6f}')
    print(f'Rel diff: {rdiff:.6f}')
    if rdiff < 0.05:
        print('PASSED: Gather-reduce matches atomic path!')
    else:
        print(f'FAILED: relative diff {rdiff} > threshold')
elif atomic_ok:
    print('\nGather-reduce path failed, atomic path succeeded')
elif gr_ok:
    print('\nAtomic path failed, gather-reduce path succeeded')
else:
    print('\nBoth paths failed')

print('\nDone!')
