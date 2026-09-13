# AITER: AI Tensor Engine for ROCm — Investigation Report

**Resource:** AMD AITER  
**URL:** https://github.com/ROCm/aiter  
**Commit:** e742eeb  
**Date:** 2026-04-05  
**Chip:** amd-gpu  
**Device Class:** GPU

---

## Overview

AITER (AI Tensor Engine for ROCm) is AMD's centralized, production-grade repository for high-performance AI operator kernels targeting AMD Instinct GPUs (MI300X/gfx942, MI350/gfx950, and older Vega/RDNA). It is AMD's direct structural answer to NVIDIA's CUTLASS/cuDNN ecosystem: a unified operator library that exposes C++ and Python-level APIs while routing each call to the best available backend — hand-written assembly (`hsa/*.co`), CK (Composable Kernel), CK-Tile, Triton, HIP, or the experimental FlyDSL — depending on the operator, precision, and detected GPU architecture. The library covers the full inference and training kernel surface needed by large-scale LLM deployments: FMHA/MLA attention (prefill and paged-decode), MoE routing and fused MLP, GEMM with FP8/INT8/INT4/BF16 precision including block-scale quantization, KV-cache management, all-reduce/reduce-scatter communication kernels, RoPE, RMSNorm/LayerNorm, and sampling. vLLM, SGLang, and other inference frameworks are the primary integration targets. The repository ships per-architecture tuned CSV configuration files and a CI autotuning pipeline so every operator is performance-regressed on real MI300X and MI350 hardware before release.

---

## Architecture

### Module Structure

```
aiter/                  # Python package root
  __init__.py           # Top-level re-exports of all ops
  ops/                  # Per-operator Python dispatch layer
    attention.py        # Paged-attention, MHA, MLA dispatch
    mha.py              # mha_fwd / fmha_v3_fwd / batch-prefill dispatch
    moe_op.py           # FusedMoE, topk_softmax, moe_align_block_size
    gemm_op_a8w8.py     # CK GEMM for INT8 activations + weights
    gemm_op_a4w4.py     # FP4/INT4 GEMM
    gemm_op_a16w16.py   # BF16/FP16 GEMM (hipBLASLt wrapper)
    quant.py            # Smoothquant, per-token, per-block-scale quant
    moe_sorting.py      # Token→expert sorting primitives
    rope.py             # RoPE position encoding
    rmsnorm.py          # RMSNorm
    communication.py    # All-reduce, reduce-scatter (Iris/Triton)
    triton/             # Pure-Triton kernel implementations
    flydsl/             # FlyDSL-based mixed-precision MoE
  jit/
    core.py             # compile_ops decorator, JIT build, AOT config load
    utils/
      chip_info.py      # rocminfo-based GFX arch + CU count detection
      mha_recipes.py    # MHA variant name→CK codegen filter mapping
  aot/
    triton/             # AOT-compiled Triton HSACO blobs
  configs/              # Tuned CSV files (a8w8, a4w4, fmha shapes)
  dist/
    device_communicators/  # Multi-GPU all-reduce strategies

csrc/                   # C++ / HIP kernel sources
  include/              # Unified headers (attention.h, mha_fwd.h, moe_op.h, …)
  cpp_itfs/             # HIP/CUDA entrypoints (mha_fwd.cu, mha_bwd.cu, …)
  py_itfs_ck/           # Python-facing CK bindings
  py_itfs_cu/           # Python-facing HIP/CUDA bindings
  ck_gemm_*/            # CK GEMM specializations per precision
  ck_tile_gemm_moe_*/   # CK-Tile MoE GEMM 2-stage codegen

hsa/                    # Pre-compiled Assembly (HSACO .co files)
  gfx942/               # MI300X/MI308 kernels
    fmha_v3_fwd/        # Attention forward variants per head-dim
    fmha_v3_bwd/        # Attention backward
    fmoe/               # FusedMoE INT8/FP8/INT4 variants
    fmoe_2stages/       # 2-stage MoE pipeline
    bf16gemm/  i8gemm/  f4gemm/  fp8gemm_blockscale/
    mla/  pa/           # Multi-head Latent Attention, Paged Attention
  gfx950/               # MI350 kernels (same structure)

gradlib/                # Training gradient kernels (hipBLASLt solution tuning)

3rdparty/
  composable_kernel/    # CK submodule (full tile library)
  ck_helper/            # CK codegen helpers
```

### Key Abstractions

| Abstraction | Location | Description |
|---|---|---|
| `@compile_ops` decorator | `aiter/jit/core.py` | Universal lazy-compile wrapper. Wraps a stub Python function; on first call, JIT-compiles the corresponding C++/HIP module (via `_jit_compile`) or loads a pre-built `.so`, then dispatches the real call. Supports `pybind` and `ctypes` FFI modes. |
| Backend selection via `cmdGenFunc` | `aiter/ops/mha.py`, `moe_op.py` | Each complex op defines a `cmdGenFunc_*` that inspects runtime properties (dtype, mask type, head-dim, dropout) to produce the correct CK codegen command string and HSACO filter glob, passed to `compile_ops(gen_func=...)`. The JIT core executes CK's Python code generator, compiles the resulting HIP source, and caches the `.so`. |
| HSACO `.co` pre-compiled blobs | `hsa/gfx942/`, `hsa/gfx950/` | Hand-written AMDGPU assembly kernels shipped as compiled code objects. Each file encodes a specific (op, dtype, tile-size, decode-mode) combination. The HIP C++ interface layer (`mha_fwd.cu`) selects the correct `.co` by matching `arch_id`, `dtype`, `hdim_q`, `hdim_v`, and mask flags through a config table. |
| `mha_fwd_traits` / `fmha_fwd_traits` | `csrc/include/mha_fwd.h` | C++ traits struct inheriting from CK-Tile's `fmha_fwd_traits`. Carries all compile-time policy choices: head-size, dtype, masking, bias mode, LSE, dropout, quant-scale type, and a flag `use_ext_asm` that routes the call to the HSACO assembly path instead of CK-generated code. |
| Tuned CSV configs | `aiter/configs/*.csv` | Per-(op, shape) best-kernel config files produced by the CI autotuning pipeline. Loaded at startup by `jit/core.py` (via `AITER_CONFIGS` dict) and consulted at op dispatch time to select optimal tile sizes and splitK settings without runtime search. |
| `chip_info` / `GFX_MAP` | `aiter/jit/utils/chip_info.py` | Runtime GPU detection via `rocminfo`. Maps GFX ISA string → integer index → arch name. Used pervasively to gate kernel registration: e.g., MI308 variants of GEMM kernels vs. MI300 vs. MI350 (gfx950). |

### Dependency Graph

```
Python caller (vLLM/PyTorch)
        |
aiter.ops.*  (Python dispatch, @compile_ops stubs)
        |
aiter.jit.core  (JIT compile / AOT load / CSV config)
        |
   [Backend selection]
   /      |      \         \
CK-Tile  Triton  HIP/CUDA  Assembly (.co)
  |        |        |          |
csrc/   aiter/   csrc/     hsa/gfx942
ck_*   ops/triton cpp_itfs  hsa/gfx950
        aot/triton
        |
  ROCm runtime (HIP, HSA)
        |
  AMD Matrix Cores (MFMA)
```

---

## Data Flow: Tracing an Attention Op through the CK Backend

This traces a `mha_fwd` call with BF16 inputs, causal masking, and LSE return — the most common vLLM prefill path.

### Step 1 — Python call

```python
# vLLM or user code
out, lse, _, _ = aiter.mha_fwd(q, k, v, dropout_p=0.0, softmax_scale=scale,
                                 is_causal=True, ...)
```

`aiter/__init__.py` re-exports `mha_fwd` from `aiter/ops/mha.py`.

### Step 2 — `@compile_ops` dispatch (`aiter/ops/mha.py` + `aiter/jit/core.py`)

The stub function decorated with `@compile_ops("mha_fwd_bf16_nbias_mask_lse_ndropout_nqscale", gen_func=cmdGenFunc_mha_fwd)` is called. The decorator resolves the module name by calling `cmdGenFunc_mha_fwd(q, k, v, ...)` which inspects `q.dtype` (BF16 → `"_bf16"`), bias (None → `"_nbias"`), causal (True → `"_mask"`), LSE (True → `"_lse"`), dropout (0.0 → `"_ndropout"`), and quantization (None → `"_nqscale"`), producing module name `mha_fwd_bf16_nbias_mask_lse_ndropout_nqscale` and a CK codegen blob filter string `*_bf16*_nbias*_mask*_lse*_ndropout*_nqscale*`.

### Step 3 — JIT compilation / cache lookup (`aiter/jit/core.py`)

`_jit_compile` checks the `.so` cache. On a cache miss it executes: `python {CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --receipt 100 --filter {filter} --output_dir {tmpdir}` which uses CK's Python code-generator to emit a set of HIP `.cpp` files instantiating the correct `ck_tile::FmhaFwdKernel<...>` template for the matching BF16, causal, LSE configuration. These are then compiled by `hipcc` with the detected arch (gfx942/gfx950), producing a shared `.so`.

### Step 4 — Fallback / override via `use_ext_asm`

If `arch_id == "gfx942"` and the head-dim and mask type appear in `asm_fmha_v3_fwd_configs.hpp`, `mha_fwd.cu:init_fmha_fwd_v3_args` sets `use_ext_asm = true`. The HIP launcher then loads the matching HSACO `.co` file from `hsa/gfx942/fmha_v3_fwd/MI300/` and submits it directly via the HSA runtime, bypassing CK's tile kernel entirely. This path is used for the highest-performance shapes (e.g., hdim=128 BF16 on MI300X).

### Step 5 — GPU execution

The CK-Tile kernel (or HSACO stub) executes on AMD Matrix Cores (MFMA instructions). For BF16 attention, the tile computes `S = Q * K^T` using `v_mfma_f32_32x32x8bf16` (or `v_mfma_f32_16x16x16bf16`), applies softmax in shared memory (using LDS for the online softmax accumulator), then multiplies by `V` via a second MFMA pass. The LSE tensor is written back to HBM as a side output.

### Step 6 — Return to Python

The `.so` pybind11 binding returns `(out, lse, softmax_lse, rng_state)` tuples back to the Python caller.

---

## Hardware Interface

### Matrix Core Utilization

AITER's kernel selection is organized around AMD's Matrix Core instructions:

| GPU Family | ISA Target | Matrix Core Width | Key MFMA Instructions |
|---|---|---|---|
| MI300X / MI308 | gfx942 | 32x32x8 (BF16), 32x32x16 (FP8) | `v_mfma_f32_32x32x8bf16`, `v_mfma_f32_32x32x16_fp8` |
| MI350 | gfx950 | 32x32x16 (BF16), 32x32x32 (FP8), 32x32x64 (INT4) | Extended MFMA with larger INT4/FP4 tiles |

The `csrc/include/ck_tile/` headers expose CK-Tile's tile-level MFMA wrappers. The `hsa/` HSACO blobs are hand-written GCN/CDNA assembly that directly encode `v_mfma_*` instruction sequences for peak compute density. Per-architecture tuning CSV files record measured optimal tile sizes (e.g., `M=128, N=256, K=64` for A8W8 GEMM on MI300X).

### ROCm Runtime Interface

- Kernel dispatch goes through HIP (`hipLaunchKernelGGL` / `hipModuleLaunchKernel` for HSACO).
- `chip_info.py` calls `rocminfo` to detect the exact GFX ISA string.
- The `aiter/dist/device_communicators/` layer wraps ROCm RCCL (or Iris Triton-communication) for multi-GPU all-reduce operations.
- Block-scale FP8 quantization kernels (`fp8gemm_blockscale`) directly address MI350's native FP8 block-scale hardware units.

---

## Key Findings

1. **Assembly-first for peak throughput.** The `hsa/` directory contains dozens of pre-compiled `.co` (code object) files covering the most performance-critical op/dtype/shape combinations. These bypass CK entirely. CK-Tile is the fallback for shapes not covered by the hand-coded assembly.

2. **CK-Tile code generation is triggered lazily at runtime.** Unlike CUTLASS which requires offline compilation, AITER's `@compile_ops` + CK code-generator approach means kernels can be specialized for the exact dtypes and feature flags encountered in production without shipping an exponentially large binary. The tradeoff is a one-time JIT compilation cost on first use.

3. **Multi-precision operator coverage is extensive.** AITER covers A8W8 (INT8 act + INT8 wt), A4W4 (FP4/INT4), A8W8 block-scale, BF16, and FP16 GEMM variants, often with separate CK, CK-Tile, and ASM backends per precision. This directly mirrors the quantization strategies used in DeepSeek-V2/V3 (MLA + FP8 block-scale).

4. **MLA (Multi-head Latent Attention) is a first-class citizen.** Dedicated kernels exist in `hsa/gfx942/mla/`, `csrc/cpp_itfs/mla/`, and `aiter/aot/triton/decode_mla.py` reflecting the importance of DeepSeek-style KV compression for AMD's inference story.

5. **Communication kernels are co-located with compute kernels.** Unlike NVIDIA's separation of cuDNN (compute) and NCCL (comms), AITER integrates all-reduce, reduce-scatter, and the Iris Triton-based GPU-initiated communication into the same repository and Python package, enabling fused compute+comms patterns needed for tensor-parallel inference.

6. **Autotuning is CI-integrated.** Per-shape best-kernel CSVs are generated by a GitHub Actions workflow that benchmarks on real MI300X/MI350 hardware and commits results back to `aiter/configs/`. This is closer to NVIDIA's cuDNN heuristic tables than to CUTLASS's purely static dispatch.

7. **FlyDSL provides mixed-precision extensibility.** For A4W4 MoE (e.g., Kimi-K2.5 on MI355X), AITER can use FlyDSL-generated kernels, with automatic fallback to CK when FlyDSL is not installed. This hints at a pluggable DSL layer above the kernel backends.

---

## Relation to Hardware Architecture

AITER sits at the intersection of the AMD software stack where operator semantics meet ISA-level hardware primitives:

```
LLM Framework (vLLM, SGLang, PyTorch)
         |
   aiter Python API (ops/*)
         |
   JIT/AOT dispatch (jit/core.py + configs/*.csv)
         |
 +-----------+-----------+-----------+
 |           |           |           |
CK-Tile   Triton      HIP/CUDA    ASM (.co)
 |           |           |           |
 +------+----+-----------+-----------+
        |
   ROCm HIP Runtime / HSA
        |
   AMD CDNA3/CDNA4 GPU
   - Matrix Cores (MFMA v_mfma_*)
   - LDS (shared memory, ~64KB/CU on MI300X)
   - HBM3 (3.2 TB/s on MI300X, 6TB/s on MI350)
   - Infinity Fabric (inter-die NPS on MI300X)
```

AITER's layer specifically targets the "Kernel Library" layer of the AMD GPU software stack. It does not manage scheduling (that is HIP/ROCm runtime), does not compile HLL code (that is hipcc/LLVM/AMDGPU backend), and does not own the ISA definition (that is CDNA architecture documentation). Its niche is the performance-portable operator library layer: providing hand-tuned and code-generated kernels that saturate MFMA throughput and HBM bandwidth for AI workloads, with enough abstraction to be consumed by any Python-level framework without requiring framework authors to understand CK tile programming or GCN assembly.

**Comparison to NVIDIA ecosystem:**

| Dimension | AITER | NVIDIA Equivalent |
|---|---|---|
| Kernel library | AITER + CK | CUTLASS + cuDNN |
| Code generation | CK-Tile Python codegen (runtime JIT) | CUTLASS C++ templates (offline) |
| Hand-written ASM | `hsa/*.co` HSACO assembly | PTX + SASS hand-written kernels in cuDNN |
| Autotuning | CSV files via CI pipeline | cuDNN heuristic engine (closed source) |
| Triton support | First-class, in `aiter/ops/triton/` | First-class, Triton targets CUDA |
| Communication | Co-located (Iris, custom_all_reduce) | Separate (NCCL) |
| MoE support | Fused MoE + 2-stage pipeline + FlyDSL | CUTLASS grouped GEMM (less integrated) |
| Framework integration | vLLM, SGLang, PyTorch custom ops | PyTorch ATen, vLLM, TensorRT |
| Open source | Full source including ASM blobs | CUTLASS open; cuDNN closed |
