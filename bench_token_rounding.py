"""Token Rounding E2E benchmark with padding analysis."""
import os, sys, time, torch, aiter
from aiter import dtypes
from aiter.ops.moe_op import ActivationType, QuantType
from aiter.fused_moe import fused_moe, fused_topk, moe_sorting

torch.set_default_device('cuda')

WARMUP = 5
ITERS = 20

CONFIGS = [
    ('DSv3-scaled',  256, 8, 1024,  512, [32, 128, 1024, 4096]),
    ('Mixtral-8x7B',   8, 2, 4096, 4096, [32, 128, 1024, 4096]),
    ('Qwen3-scaled', 128, 8, 1024,  512, [32, 128, 1024, 4096]),
]

dtype = dtypes.bf16
activation = ActivationType.Silu
block_m = 32


def bench_one(hidden, w1, w2, topk_ids, topk_weight, enable_rounding):
    for _ in range(WARMUP):
        fused_moe(hidden, w1, w2, topk_weight, topk_ids,
                  activation=activation, enable_token_rounding=enable_rounding)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fused_moe(hidden, w1, w2, topk_weight, topk_ids,
                  activation=activation, enable_token_rounding=enable_rounding)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / ITERS * 1000


def get_padded_tokens(topk_ids, topk_weight, E, D, enable_rounding):
    _, _, _, num_valid, _ = moe_sorting(
        topk_ids, topk_weight, E, D, dtype, block_m,
        enable_token_rounding=enable_rounding)
    return num_valid[0].item()


hdr = "{:<14} {:>5} {:>6} {:>6} {:>6} {:>9} {:>9} {:>7}".format(
    'Model', 'T', 'actual', 'ceil', 'near', 'ceil(ms)', 'near(ms)', 'speedup')
print(hdr)
print('-' * len(hdr))

for model_name, E, K, D, I, T_list in CONFIGS:
    torch.manual_seed(42)
    w1 = torch.randn(E, I * 2, D, dtype=dtype)
    w2 = torch.randn(E, D, I, dtype=dtype)

    for T in T_list:
        hidden = torch.randn(T, D, dtype=dtype)
        gate = torch.randn(T, E, dtype=torch.float32)
        tw, ti = fused_topk(hidden, gate, K, renormalize=True)
        actual = T * K

        try:
            pad_ceil = get_padded_tokens(ti, tw, E, D, False)
            pad_near = get_padded_tokens(ti, tw, E, D, True)
            t_ceil = bench_one(hidden, w1, w2, ti, tw, False)
            t_near = bench_one(hidden, w1, w2, ti, tw, True)
            speedup = t_ceil / t_near if t_near > 0 else 0
            print('{:<14} {:>5} {:>6} {:>6} {:>6} {:>9.3f} {:>9.3f} {:>6.3f}x'.format(
                model_name, T, actual, pad_ceil, pad_near, t_ceil, t_near, speedup))
        except Exception as e:
            print('{:<14} {:>5} ERROR: {}'.format(model_name, T, e))

    del w1, w2
    torch.cuda.empty_cache()

print('\nDone!')
