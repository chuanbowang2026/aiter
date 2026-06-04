# SonicMoe-inspired MoE Optimizations for aiter

Reference: SonicMoe (2512.14080v2)

## What's in this branch

Two optimizations applied to aiter's CK Tile 2-stage MoE pipeline:

### 1. Token Rounding (verified on gfx942)

Reduces GEMM tile padding waste by using nearest-round instead of ceil-pad when assigning tokens to tile blocks.

**How it works**: When an expert gets e.g. 35 tokens with block_size=32, ceil-pad rounds up to 64 (29 wasted). Nearest-round picks 32 instead (only 3 tokens dropped, but 32 less compute). Controlled by `enable_token_rounding=True` on `fused_moe()`.

**E2E benchmark results** (MI308X gfx942, bf16, block_m=32):

| Model config | T=1024 speedup | T=4096 speedup |
|---|---|---|
| DeepSeek-V3 (E=256, K=8) | 1.00x | **1.42x** |
| Mixtral-8x7B (E=8, K=2) | **1.14x** | **1.04x** |
| Qwen3-235B (E=128, K=8) | 1.00x | **1.26x** |

Best for prefill (T≥1024) + many experts (E≥128). No effect on decode (T≤32).

**Trade-off**: Round-down drops tail tokens (approximate). Acceptable for MoE routing where individual expert weights are small.

### 2. Gather-Reduce (code complete, needs gfx950 verification)

Eliminates K-way `atomic_add` contention in stage2 output by writing to a per-expert buffer, then reducing with explicit weight multiplication.

**How it works**: Instead of `atomic_add(out[token], result * weight)` from K experts racing on the same row, the kernel writes `result` (no weight) to `y_buf[token*K+k]` using `memory_operation_enum::set`, then Python does `out = (y_buf * topk_weights).sum(dim=1)`.

**Status**: Code complete across 7 files. Compiles on gfx942 but E2E test blocked by pre-existing MXFP4 pipeline issue (`mixed_prec_flatmm_pipeline` `scale.data` error) that affects gfx942 for bf16+fp4x2 weights. **Needs gfx950 to test.**

Enable with: `AITER_USE_GATHER_REDUCE=1`

## Files changed

### Token Rounding
| File | Change |
|---|---|
| `csrc/include/moe_sorting_opus.h` | Nearest-round logic in 3 ceil-pad sites |
| `csrc/py_itfs_cu/moe_sorting_opus_kernels.cu` | Pass `enable_token_rounding` to host args |
| `csrc/include/rocm_ops.hpp` | Pybind `enable_token_rounding` param |
| `aiter/ops/moe_sorting_opus.py` | Python wrapper param |
| `aiter/fused_moe.py` | Propagate param through `moe_sorting` → `fused_moe` |

### Gather-Reduce
| File | Change |
|---|---|
| `3rdparty/composable_kernel/.../moe_flatmm_kernel.hpp` | `ForceSetOutput` template param: skip weight mul, use `set` output, scatter to `[T*K, H]` |
| `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh` | `ForceSetOutput_` param on `moe_gemm<>` template |
| `csrc/ck_tile_gemm_moe_2stages/moe_cktile2stages.cu` | `moe_gemm2_gather_reduce()` + `use_gather_reduce` dispatch |
| `csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages.h` | Declaration update |
| `csrc/include/rocm_ops.hpp` | Pybind `use_gather_reduce` param |
| `aiter/ops/moe_op.py` | Python wrapper param |
| `aiter/fused_moe.py` | Gather-reduce path + `AITER_USE_GATHER_REDUCE` env var |

### Other (SonicMoe bitonic topk - prior work)
| File | Change |
|---|---|
| `csrc/kernels/topk_softmax_kernels.cu` | Bitonic sort network for topk selection |

## How to test on gfx950

```bash
# 1. Clone and checkout
git clone https://github.com/chuanbowang2026/aiter.git
cd aiter
git checkout sonicmoe-optimizations
git submodule update --init 3rdparty/composable_kernel

# 2. Build (standard aiter build)
pip install -e .

# 3. Test Token Rounding
python bench_token_rounding.py

# 4. Test Gather-Reduce correctness
AITER_USE_GATHER_REDUCE=1 python test_gather_reduce_e2e.py

# 5. Run standard MoE test suite to verify no regression
cd op_tests
python test_moe_2stage.py --dtype bf16 --token 32 --expert 8 --topk 2

# 6. Compare gather-reduce vs atomic
AITER_USE_GATHER_REDUCE=0 python test_moe_2stage.py --dtype bf16 --token 128 --expert 8 --topk 2
AITER_USE_GATHER_REDUCE=1 python test_moe_2stage.py --dtype bf16 --token 128 --expert 8 --topk 2
```

## Notes

- Token Rounding is off by default (`enable_token_rounding=False`). It drops tokens, so outputs differ from ceil-pad.
- Gather-Reduce is off by default (`AITER_USE_GATHER_REDUCE=0`). It allocates a `[T, K, H]` temp buffer.
- The CK submodule has 1 file changed (`moe_flatmm_kernel.hpp`). Make sure to run `git submodule update --init`.
