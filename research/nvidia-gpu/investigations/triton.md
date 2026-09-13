# Triton: OpenAI Triton Language and Compiler

**Source**: https://github.com/triton-lang/triton  
**Version investigated**: main branch (shallow clone, April 2026)  
**Layer**: Compiler / IR + Kernel Library

---

## Overview

Triton is an open-source Python DSL and MLIR-based compiler for writing highly efficient custom GPU kernels. It occupies two layers of the AI software stack simultaneously:

- **Compiler / IR layer**: Triton provides its own MLIR dialects (Triton IR, TritonGPU IR) and a multi-stage lowering pipeline all the way to PTX and SASS. The compiler automates shared memory allocation, memory coalescing, software pipelining, and Tensor Core instruction selection.
- **Kernel Library layer**: The `python/triton_kernels/` package ships production-ready kernels (e.g., `matmul.py`) built with the Triton DSL and exposed as a callable Python library.

The core programming model is **blocked SPMD**: rather than writing per-thread scalar code as in CUDA, the programmer writes per-tile (block-level) code. Each `@triton.jit`-decorated kernel computes one output tile per program instance. This lifts coalescing and Tensor Core mapping decisions to the compiler rather than requiring the programmer to manage individual warp lanes.

The original paper is "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MAPL 2019).

---

## Architecture

### Python Frontend

Kernels are Python functions decorated with `@triton.jit`. The JIT layer (`python/triton/runtime/jit.py`) uses Python's `ast` module to walk the function body through a `CodeGenerator` visitor (`python/triton/compiler/code_generator.py`). This visitor translates Python expressions into Triton IR MLIR operations via the `libtriton` C extension. Kernel arguments can be marked `tl.constexpr` to fold them as compile-time constants (e.g., tile sizes `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`).

Key language primitives live in `python/triton/language/core.py`:
- `tl.program_id(axis)` — returns the logical block index along a grid dimension
- `tl.arange(start, stop)` — creates a tile-shaped index vector
- `tl.load(ptr, mask)` / `tl.store(ptr, val, mask)` — blocked load/store with optional predication
- `tl.dot(a, b, acc)` — blocked matrix multiply accumulate, lowered to MMA instructions

### Compiler Pipeline (NVIDIA backend)

The full pipeline is defined in `third_party/nvidia/backend/compiler.py:add_stages()`:

```
Python AST
   |
   v  [code_generator.py — ast_to_ttir()]
Triton IR (ttir)         — high-level tensor operations, loops, pointer arithmetic
   |
   v  [make_ttir()]
Triton IR (optimized)    — inliner, combine, CSE, DCE, loop unroll
   |
   v  [make_ttgir()]
TritonGPU IR (ttgir)     — layout encoding attached to every tensor; coalescing, matmul acceleration,
                           software pipelining (num_stages), warp specialization, TMA lowering (Hopper+),
                           tensor memory allocation (Blackwell)
   |
   v  [make_llir()]
LLVM IR (MLIR dialect)   — ttgpuir → LLVMIR conversion; shared memory allocation; nvvm intrinsics
   |
   v  [make_llir() cont.] LLVM IR (LLVM native) — O3 optimization, datalayout for nvptx64
   |
   v  [make_ptx()]
PTX                      — llvm.translate_to_asm() targeting nvptx64-nvidia-cuda + sm_XX
   |
   v  [make_cubin()]
CUBIN                    — ptxas assembles PTX → SASS binary for the target SM
```

Compiled artifacts are cached by content hash under `~/.triton/cache/`. The `CompiledKernel` class holds all intermediate representations in its `.asm` dict, so callers can inspect `.asm["ptx"]`, `.asm["ttgir"]`, etc.

### TritonGPU IR: Layout Encodings

The key innovation in TritonGPU IR is that every tensor value carries an **encoding attribute** describing how its elements are distributed across warps and threads. The `AccelerateMatmul` MLIR pass (`lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`) converts `tt.dot` operations into `tt.dot` with `NvidiaMmaEncodingAttr` layout. It selects the highest supported MMA version automatically:

| Compute Capability | MMA Version | Hardware Instruction |
|--------------------|-------------|----------------------|
| < 7.5              | v1          | Volta HMMA           |
| 7.5 – 8.9          | v2          | Ampere WMMA/HMMA     |
| 9.0 (Hopper)       | v3          | WGMMA (warpgroup MMA)|
| 10.0 – 11.9 (Blackwell) | v5     | tcgen05 MMA          |

The `Coalesce` pass reorders tensor memory accesses for L1/L2 locality. The `RemoveLayoutConversions` pass eliminates redundant layout casts introduced by encoding propagation.

### Auto-tuning

`@triton.autotune` wraps a kernel with a list of `triton.Config` objects specifying tile sizes (`BLOCK_SIZE_M/N/K`), `num_warps`, `num_stages`, and `num_ctas`. On first call the `Autotuner` class benchmarks each config using the driver's benchmarker, then caches the winner keyed by the shapes of the input tensors. Subsequent calls with the same shapes skip benchmarking entirely.

### Gluon (Blackwell-specific explicit-layout DSL)

`python/triton/experimental/gluon/` is a lower-level DSL that exposes explicit shared memory allocation, TMA descriptors, and `tcgen05`-family tensor memory (`tmem`) operations for Blackwell (SM100+). The Gluon compiler path enters the pipeline at `ttgir` rather than `ttir`, bypassing automatic layout inference and giving the programmer direct control over hardware resources.

---

## Data Flow: tl.dot Matmul Kernel

Tracing the matmul kernel from `python/tutorials/03-matrix-multiplication.py`:

**Step 1 — Python DSL**

```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                   'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ...
], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ...):
    pid = tl.program_id(axis=0)
    ...
    a = tl.load(a_ptrs, mask=...)      # [BLOCK_SIZE_M, BLOCK_SIZE_K] tile
    b = tl.load(b_ptrs, mask=...)      # [BLOCK_SIZE_K, BLOCK_SIZE_N] tile
    accumulator = tl.dot(a, b, accumulator)  # blocked matmul-accumulate
```

**Step 2 — Python AST to Triton IR (ttir)**

`code_generator.py:ast_to_ttir()` walks the AST and emits MLIR operations:
- `tl.load` → `tt.load %ptr, %mask`
- `tl.dot(a, b, acc)` → `tt.dot %a, %b, %acc {inputPrecision = tf32}`

**Step 3 — make_ttir: Triton IR optimization**

Passes: inliner, combine, reorder-broadcast, CSE, symbol-DCE, loop unroll. The dot op remains as `tt.dot` with `#blocked` encoding on all tensors.

**Step 4 — make_ttgir: TritonGPU IR**

`add_convert_to_ttgpuir` assigns initial layouts. `add_accelerate_matmul` rewrites `tt.dot` to carry `NvidiaMmaEncodingAttr` (MMA v2 for A100, MMA v3 for H100, MMA v5 for B200). `add_coalesce` inserts explicit `ttg.convert_layout` ops to ensure the `tl.load` tiles arrive in a layout compatible with `cp.async` (Ampere) or TMA (Hopper). `add_pipeline` (controlled by `num_stages`) introduces multi-buffering in shared memory, turning the synchronous load loop into an overlapped async-copy + compute pipeline.

For Hopper (sm90): `add_tma_lowering` replaces `tt.load` of large tiles with `ttng.async_tma_copy_global_to_local` using hardware TMA descriptors, bypassing the L1 cache path entirely.

**Step 5 — make_llir: TritonGPU IR → LLVM IR**

`add_allocate_shared_memory_nv` statically computes the required shared memory size and allocates a single flat SRAM buffer; pointers into it are expressed as byte offsets. `add_to_llvmir` converts `ttg` ops to NVVM/LLVM intrinsics: `ttg.mma` → `nvvm.wmma.*` or `nvvm.wgmma.*`; async copies → `nvvm.cp.async.*`. The resulting LLVM module is optimized at O3.

**Step 6 — make_ptx: LLVM IR → PTX**

`llvm.translate_to_asm()` calls LLVM's NVPTX backend targeting `nvptx64-nvidia-cuda` with `sm_XX` target. The output is a PTX text string. For a 128×256 output tile on A100 with 8 warps the inner loop emits `mma.sync.aligned.m16n8k16` instructions.

**Step 7 — make_cubin: PTX → CUBIN (SASS)**

`ptxas` assembles PTX to a CUBIN binary. Register allocation, instruction scheduling, and bank-conflict resolution happen here. The resulting CUBIN is loaded into the driver via `load_binary()` and the kernel handle is stored in `CompiledKernel.function`.

**Step 8 — GPU execution**

`kernel[grid](*args)` calls the `runner` closure, which invokes the driver launcher with `(grid_x, grid_y, grid_z, stream, function, packed_metadata, ...)`. The CUDA driver dispatches one program instance (CTA) per grid point.

---

## Hardware Interface

### Tile Operations to Tensor Cores

Triton's tile-based programming model maps directly onto the Tensor Core / MMA instruction hierarchy:

- Each `tl.dot(a, b, acc)` on an `[M_tile, K_tile] x [K_tile, N_tile]` pair is lowered to a sequence of `mma.sync` (Ampere), `wgmma.mma_async` (Hopper), or `tcgen05.mma` (Blackwell) instructions.
- The tile sizes must satisfy minimum MMA fragment constraints enforced at `make_ttgir` time by `min_dot_size()` in the NVIDIA backend: for fp16/bf16 operands the minimum K-block is 16; for int8 it is 32.
- `NvidiaMmaEncodingAttr` records the MMA version and the warp-level fragment layout so that the LLVM lowering can generate the correct register assignment.

### Shared Memory Management

Triton performs **automatic shared memory allocation**: the programmer never calls `__syncthreads()` or declares `__shared__` arrays. Instead:

1. The `AllocateSharedMemoryNV` pass in `make_llir` performs liveness analysis over the `ttgir` module, assigns non-overlapping byte offsets for each live tensor in SRAM, and emits a single `ttg.shared_memory` allocation of size recorded in the `ttg.shared` module attribute.
2. `cp.async` (Ampere) or TMA (Hopper/Blackwell) copies fill shared memory buffers asynchronously; `add_pipeline` inserts `mbarrier` synchronization barriers.
3. The shared memory size is checked at kernel load time (`CompiledKernel._init_handles`) against `max_shared_mem(device)` and raises `OutOfResources` if it exceeds the device limit (typically 164 KB for A100, 228 KB for H100).

### Hopper-specific: TMA and Warp Specialization

On SM90 (Hopper), `add_tma_lowering` replaces explicit `tl.load` of large tiles with `AsyncTMACopyGlobalToLocalOp`, which generates `cp.async.bulk.tensor` instructions. TMA handles address computation and L2-bypass automatically, dramatically reducing instruction issue pressure for load-heavy kernels like attention.

`add_warp_specialize` (for `num_stages` with `num_ctas > 1` or explicitly) partitions the warps within a CTA into producer warps (running loads) and consumer warps (running MMA), exploiting the independent scheduling of warp groups on H100's SM.

### Blackwell-specific: Tensor Memory and tcgen05

On SM100 (Blackwell), a new on-chip **tensor memory (tmem)** buffer separate from SRAM is available. `add_allocate_tensor_memory` and `add_hoist_tmem_alloc` manage this buffer. `add_lower_mma` emits `tcgen05.mma` instructions that operate directly on tmem-resident accumulator tiles, eliminating accumulator round-trips through registers for large tiles. The Gluon DSL provides explicit access to tmem via `GluonSemantic`.

---

## Key Findings

1. **Automatic memory coalescing**: The `Coalesce` MLIR pass (`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`) analyzes pointer access patterns and inserts layout conversion ops so that warp-level load/store instructions access consecutive DRAM addresses, maximizing memory bus utilization without programmer intervention.

2. **Tile-based programming model vs. scalar CUDA**: Where CUDA programs one scalar thread at a time (requiring explicit warp-level coordination), Triton programs one output tile per CTA. This abstraction hides shared memory staging, double-buffering, and barrier placement behind the compiler, enabling productivity close to Python while achieving near-cuBLAS performance.

3. **MMA version auto-selection**: `getMMAVersionSafe()` in `AccelerateMatmul.cpp` selects MMA v1/v2/v3/v5 at compile time based on compute capability and operand dtype, so the same Triton source code runs on Volta through Blackwell without user modification.

4. **Software pipelining via `num_stages`**: The `AddPipeline` MLIR pass implements multi-stage software pipelining by unrolling the inner loop `num_stages` times and inserting `cp.async` / `mbarrier` pairs to overlap global memory fetches with Tensor Core compute. This is equivalent to manually double-buffering a CUDA kernel but expressed as a single compiler option.

5. **CUDA Tile IR (Gluon) for Blackwell**: The `experimental/gluon/` sub-package introduces an explicit layout DSL for Blackwell that exposes `tcgen05` tensor memory operations, `wgmma` descriptors, and fine-grained warp specialization, targeting the gap between Triton's automatic model and raw PTX for cutting-edge hardware.

6. **TorchInductor integration**: PyTorch's `torch.compile()` uses TorchInductor as the default backend. TorchInductor generates Triton kernels for element-wise, reduction, and custom-fusion operations at the CUDA target. This makes Triton the de facto kernel generation target for PyTorch 2.x eager-graph compilation.

7. **JAX/Pallas integration**: JAX's Pallas frontend (`jax.experimental.pallas`) provides a grid-based API that lowers to Triton IR on NVIDIA GPUs and to Mosaic on TPUs, sharing the same conceptual tile abstraction.

8. **Disk cache keyed by source hash + options + env vars**: The compiler's `get_cache_key()` combines source hash, backend hash, CUDA options, and selected environment variables. Cache invalidation is automatic and environment-driven (`get_cache_invalidating_env_vars()`).

---

## Relation to Hardware

The hardware constraints of NVIDIA SMs directly shaped Triton's design:

- **Shared memory as the primary fast-memory tier**: SM SRAM (up to 228 KB on H100) is the only fast memory available to all threads in a CTA. Triton's tile abstraction and automatic shared memory allocator are a direct response to this: the compiler stages global memory reads through SRAM tiles of precisely the sizes that fit in the available budget.

- **Tensor Core fragment ownership**: MMA instructions (mma.sync, wgmma) operate on fixed-size fragments distributed across warp lanes in a non-obvious register layout. `NvidiaMmaEncodingAttr` encodes this layout so the compiler can generate correct data movement without the programmer managing fragment ownership manually.

- **Warp-level parallelism**: Triton's `num_warps` parameter directly sets the number of warps per CTA (`ttg.num-warps` module attribute). The `AccelerateMatmul` and `WarpSpecialize` passes partition work across warps to saturate the SM's MMA and load/store units in parallel.

- **Asynchronous data paths (cp.async / TMA)**: Ampere introduced `cp.async` to overlap DRAM→SRAM transfers with compute. Hopper's TMA engine added hardware address calculation and multicast. Triton's pipeliner and TMA lowering are the compiler-level response to these hardware features.

- **Blackwell tensor memory**: SM100 adds a separate on-chip accumulator store (tmem) with dedicated `tcgen05` instructions. Triton's Blackwell backend adds `AllocateTensorMemory`, `HoistTMEMAlloc`, and `LowerMMA` passes to manage and exploit this new memory tier, showing how each GPU generation forces new compiler infrastructure.
