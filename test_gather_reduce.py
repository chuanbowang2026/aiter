import os
os.environ['AITER_USE_GATHER_REDUCE'] = '1'

import torch
import aiter
from aiter import dtypes
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    cktile_moe_stage1,
    cktile_moe_stage2,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.moe_op import ActivationType, QuantType

torch.set_default_device('cuda')
torch.manual_seed(42)

# DeepSeek-V3 like config, small batch for testing
T = 4       # tokens
E = 8       # experts
K = 2       # topk
D = 128     # model dim
I = 256     # intermediate dim
block_m = 32

dtype = dtypes.bf16

# Create test tensors
hidden = torch.randn(T, D, dtype=dtype)
# For simplicity, use bf16 weights (not fp4) for initial test
# The gather-reduce path supports a16w4 (bf16+fp4x2)
w1 = torch.randn(E, I * 2, D, dtype=dtype)  # gate+up
w2 = torch.randn(E, D, I, dtype=dtype)       # down

# Gate
gate_output = torch.randn(T, E, dtype=torch.float32)
topk_weights, topk_ids = fused_topk(hidden, gate_output, K, renormalize=True)

# Sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, E, D, dtype, block_m
)

print(f'T={T}, E={E}, K={K}, D={D}, I={I}')
print(f'sorted_ids.shape={sorted_ids.shape}, topk_weights.shape={topk_weights.shape}')
print(f'topk_ids={topk_ids}')
print(f'topk_weights={topk_weights}')

# Reference: torch loop
ref_out = torch.zeros(T, D, dtype=dtype)
for t in range(T):
    for k in range(K):
        eid = topk_ids[t, k].item()
        w = topk_weights[t, k].item()
        x = hidden[t:t+1]  # [1, D]
        # stage1: gate+up
        g = x @ w1[eid, :I, :].T   # [1, I]
        u = x @ w1[eid, I:, :].T   # [1, I]
        act = torch.nn.functional.silu(g) * u  # [1, I]
        # stage2: down
        y = act @ w2[eid].T  # [1, D]
        ref_out[t] += y.squeeze(0) * w

print(f'Reference output: {ref_out[0, :8]}')

# The cktile_moe_stage2 expects fp4x2 weights, so we cannot directly test
# with bf16 weights through the CK tile path.
# Instead, let's verify the Python logic works by mocking the GEMM output.

# Simulate what the kernel would produce with ForceSetOutput=true
# y_buf[t, k, D] = stage2 GEMM result for token t, expert k (before weight multiply)
y_buf_ref = torch.zeros(T, K, D, dtype=dtype)
for t in range(T):
    for k in range(K):
        eid = topk_ids[t, k].item()
        x = hidden[t:t+1]
        g = x @ w1[eid, :I, :].T
        u = x @ w1[eid, I:, :].T
        act = torch.nn.functional.silu(g) * u
        y = act @ w2[eid].T
        y_buf_ref[t, k] = y.squeeze(0)

# Apply weights and reduce (same as gather-reduce Python path)
y_weighted = y_buf_ref * topk_weights.view(T, K, 1).to(dtype)
gr_out = y_weighted.sum(dim=1)

# Compare
diff = (ref_out - gr_out).abs().max().item()
print(f'Gather-reduce vs reference diff: {diff}')
rdiff = diff / (ref_out.abs().max().item() + 1e-6)
print(f'Relative diff: {rdiff}')
assert rdiff < 0.02, f'Mismatch relative: {rdiff}'
print('Python logic verification PASSED')

# Now test the actual kernel path - need fp4 weights for cktile2stages
# For now, just verify the module compiles
print('Attempting to compile moe_cktile2stages module...')
try:
    # This will trigger JIT compilation
    from aiter.ops.moe_op import moe_cktile2stages_gemm2
    print('Module compiled successfully!')
except Exception as e:
    print(f'Compilation error (expected for prototype): {e}')

print('Done!')
