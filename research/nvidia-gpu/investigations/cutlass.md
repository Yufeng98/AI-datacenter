# CUTLASS Investigation Report

**as_of:** 2026-04-05
**commit_sha:** 418d38a5de245373d5bac809c06d1ecd1a196c63
**source:** https://github.com/NVIDIA/cutlass

---

## Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is a header-only C++ template library from NVIDIA that provides composable building blocks for high-performance matrix-matrix multiplication (GEMM) and related dense linear algebra operations on CUDA GPUs. Targeting performance engineers, GPU kernel authors, DL framework developers, and researchers, CUTLASS sits between raw PTX/SASS at the bottom and cuBLAS/deep learning frameworks at the top. It exposes the full GPU programming hierarchy — from device-wide tile scheduling down to individual Tensor Core instructions — as reusable, parameterizable C++ template abstractions. As of version 4.x, it also ships CuTe DSL, a Python-native DSL that compiles through MLIR to CUDA, enabling high-performance kernel authoring without deep C++ expertise. CUTLASS is the foundational substrate used internally by cuBLAS, NVIDIA's Transformer Engine, and is directly consumed by projects like xFormers and FlashAttention for custom attention kernels.

---

## Architecture

### Module Structure

| Directory / Module | Language | Role |
|---|---|---|
| `include/cutlass/` | C++ (header-only) | Core CUTLASS 2.x/3.x template library: GEMM, convolution, epilogue, layout, arch wrappers |
| `include/cutlass/arch/` | C++ | Thin PTX wrappers: `mma_sm*.h`, `memory.h`, `barrier.h` per architecture generation |
| `include/cutlass/gemm/` | C++ | GEMM hierarchy: `device/`, `kernel/`, `collective/`, `threadblock/`, `warp/`, `thread/` |
| `include/cutlass/epilogue/` | C++ | Output pipelines: `collective/`, `fusion/` (EVT visitor trees), `thread/` |
| `include/cutlass/pipeline/` | C++ | `PipelineTmaAsync` and related barrier abstractions for producer/consumer warp specialization |
| `include/cute/` | C++ (header-only) | CuTe: layout algebra, Tensor abstraction, MMA/Copy atoms, tiled MMA/Copy |
| `include/cute/arch/` | C++ | PTX wrappers for CuTe: `mma_sm90_gmma.hpp` (wgmma), `mma_sm100_umma.hpp` (tcgen05), `copy_sm90_tma.hpp` |
| `include/cute/atom/` | C++ | `MMA_Atom`, `TiledMMA`, `Copy_Atom`, `TiledCopy` and per-arch traits |
| `python/CuTeDSL/` | Python + MLIR | CuTe DSL: Python kernel authoring, MLIR compilation, PyTorch/JAX integration |
| `python/cutlass_library/` | Python | Kernel enumeration, instantiation, and profiler manifest generation |
| `python/pycute/` | Python | Pure-Python layout/swizzle utilities mirroring CuTe C++ concepts |
| `tools/profiler/` | C++ | `cutlass_profiler` CLI for benchmarking kernel instances |
| `tools/library/` | C++ | Compiled kernel instance library (CUTLASS Instance Library) |
| `test/unit/` | C++ | GoogleTest-based unit tests mirroring top-level namespaces |
| `examples/` | C++ / Python | 90+ numbered examples from basic GEMM to Blackwell attention kernels |

### Key Abstractions

| Abstraction | Location | Description |
|---|---|---|
| **CuTe Layout** | `include/cute/layout.hpp` | Hierarchically multidimensional layout: Shape + Stride as compile-time or runtime integer tuples. Foundation for all data indexing. |
| **CuTe Tensor** | `include/cute/tensor.hpp` | Pairs a Layout with a typed pointer/engine (global, shared, or register memory). Unifies all data movement in the library. |
| **MMA_Atom / TiledMMA** | `include/cute/atom/mma_atom.hpp` | Wraps a single hardware MMA instruction (e.g., `mma.sync`, `wgmma.mma_async`, `tcgen05.mma`) with thread/register layout metadata; `TiledMMA` tiles it across a warp group. |
| **Copy_Atom / TiledCopy** | `include/cute/atom/copy_atom.hpp` | Wraps a hardware copy instruction (e.g., `cp.async`, `cp.async.bulk` TMA) with layout metadata; `TiledCopy` tiles it across threads. |
| **CollectiveMma** | `include/cutlass/gemm/collective/collective_mma.hpp` | Tag-dispatched struct that owns the mainloop: loads tiles from global to shared memory and issues MMA instructions. Parameterized by `DispatchPolicy` (e.g., `MainloopSm90TmaGmmaWarpSpecialized`). |
| **CollectiveEpilogue / EVT** | `include/cutlass/epilogue/collective/` | Post-MMA output stage. Epilogue Visitor Tree (EVT) allows composable fused operations (bias, activation, quantization) as a tree of callback nodes. |
| **PipelineTmaAsync** | `include/cutlass/pipeline/sm90_pipeline.hpp` | Software pipeline abstraction over hardware `mbarrier` barriers. Coordinates producer (DMA/TMA) and consumer (MMA) warp groups with `producer_commit` / `consumer_wait` API. |
| **GemmUniversalAdapter** | `include/cutlass/gemm/device/gemm_universal_adapter.h` | Host-side entry point that constructs `Params`, calls `initialize_workspace`, and invokes `device_kernel<GemmKernel>`. |
| **TileScheduler** | `include/cutlass/gemm/kernel/tile_scheduler.hpp` | Persistent or StreamK tile dispatch. Maps SM-resident warps to output tiles across the problem grid; `PersistentTileSchedulerSm90`, `StreamKScheduler`. |
| **DispatchPolicy** | `include/cutlass/gemm/dispatch_policy.hpp` | Tag types (e.g., `MainloopSm90TmaGmmaWarpSpecialized`, `MainloopSm80CpAsyncMultistage`) that select the correct CollectiveMma specialization at compile time. |

### Dependency Graph

```
[User / DL Framework]
        |
        v
[GemmUniversalAdapter]  (device/)
        |
        v
[GemmKernel sm90/sm100] (kernel/)
   |             |
   v             v
[CollectiveMma]  [CollectiveEpilogue / EVT]
   |                  |
   v                  v
[TiledMMA]       [TiledCopy (TMA output)]
[TiledCopy]      [EVT visitor callbacks]
   |
   v
[CuTe: Layout, Tensor, MMA_Atom, Copy_Atom]
   |
   v
[arch/ PTX wrappers: wgmma / tcgen05 / cp.async.bulk]
   |
   v
[Tensor Cores / TMA engine / DSMEM]
```

---

## Data Flow: Hopper (SM90) GEMM Path

The following traces a 16-bit GEMM from user call through to Tensor Core execution on H100.

1. **User calls `GemmUniversalAdapter<GemmKernel>::run(args)`** (`include/cutlass/gemm/device/gemm_universal_adapter.h`).
   - `args` are lowered to a `GemmKernel::Params` struct (tile shape, strides, data pointers).
   - `initialize_workspace()` allocates barrier arrays and tile scheduler state in device memory.
   - `device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params)` launches the CUDA kernel.

2. **`device_kernel<GemmKernel>` dispatches to `sm90_gemm_tma_warpspecialized_pingpong`** (`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`).
   - Block is split into warp groups: one **DMA producer** warp group and one or two **MMA consumer** warp groups.
   - The `PersistentTileSchedulerSm90` assigns output tiles to thread blocks persistently until all tiles are processed.

3. **DMA warp group issues TMA loads via `PipelineTmaAsync`** (`include/cutlass/pipeline/sm90_pipeline.hpp`).
   - `producer_try_acquire()` checks `mbarrier` (empty signal); then calls `CollectiveMma::load()`.
   - Inside `load()`, `TiledCopy<SM90_TMA_LOAD>` emits `cp.async.bulk.tensor.Nd.shared::cluster.global` PTX (`include/cute/arch/copy_sm90_tma.hpp`).
   - TMA hardware DMA engines asynchronously move the A/B tile from HBM through L2 into shared memory.
   - `producer_commit()` signals the `mbarrier` full barrier.

4. **MMA warp group waits and computes via `CollectiveMma`** (`include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`).
   - `consumer_wait()` on `PipelineTmaAsync` spins on `mbarrier` until the tile is in shared memory.
   - `CollectiveMma::mma()` iterates over K tiles, calling `cute::gemm(TiledMMA, accum, smem_A_tensor, smem_B_tensor)` (`include/cute/algorithm/gemm.hpp`).
   - `TiledMMA` wraps `MMA_Atom<GMMA::MMA_64x64x16_F32F16F16_SS>` which calls `warpgroup_arrive()` / `wgmma.mma_async.sync.aligned` PTX.
   - The `wgmma` instruction reads operand descriptors pointing into shared memory and accumulates into register-file accumulators.
   - `warpgroup_commit_batch()` and `warpgroup_wait<0>()` synchronize the warp group MMA pipeline.

5. **Epilogue writes output via `CollectiveEpilogue`** (`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`).
   - Accumulator registers are optionally scaled/biased through an EVT callback tree.
   - `TiledCopy<SM90_TMA_STORE>` writes the output tile back to global memory via TMA store.

---

## Hardware Interface

CUTLASS exposes hardware interfaces through thin PTX wrapper structs, never directly including CUDA intrinsics in user-visible code.

| Hardware Feature | CUTLASS Abstraction | PTX Instruction | Architecture |
|---|---|---|---|
| Tensor Core (warp-level MMA) | `MMA_Atom<MMAop>`, `mma.sync` | `mma.sync.aligned.m*n*k*.*` | Volta–Ada (SM70–SM89) |
| Tensor Core (warpgroup async MMA) | `MMA_Atom<GMMA::...>`, `wgmma_arrive/wait` | `wgmma.mma_async.sync.aligned.*` | Hopper SM90 |
| Tensor Core (UMMA with TMEM) | `SM100_MMA_TF32_SS`, `SM100_MMA_F16BF16_SS` | `tcgen05.mma.cta_group::1.*` | Blackwell SM100/SM103 |
| Tensor Memory (TMEM) | `tmem_c` pointer in `SM100_MMA_*` structs | TMEM address registers | Blackwell SM100 |
| Asynchronous Global-to-SMEM copy | `Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS>` | `cp.async.ca.shared.global` | Ampere SM80+ |
| TMA (Tensor Memory Accelerator) | `Copy_Atom<SM90_TMA_LOAD>`, TMA descriptor | `cp.async.bulk.tensor.*` | Hopper SM90+ |
| SMEM barriers (mbarrier) | `PipelineTmaAsync`, `Barrier::arrive_and_expect_tx()` | `mbarrier.arrive.expect_tx` | Hopper SM90+ |
| Distributed shared memory | `SM90_TMA_LOAD_MULTICAST` | `cp.async.bulk.tensor.*.shared::cluster` | Hopper SM90 cluster |

The architecture boundary is entirely within `include/cute/arch/` and `include/cutlass/arch/`. Every hardware feature has a guard macro (e.g., `CUTE_ARCH_MMA_SM90A_ENABLED`, `CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED`) and a fallback `CUTE_INVALID_CONTROL_PATH` runtime error for host-side or unsupported targets.

---

## Key Findings

1. **CuTe is the real foundation of CUTLASS 3.x.** The entire GEMM 3.x stack is built on CuTe's Layout/Tensor algebra. A `CollectiveMma` is essentially a composition of `TiledMMA` and `TiledCopy` atoms, both of which are just CuTe objects tiling a PTX wrapper. CUTLASS 3.x removed all hand-coded indexing in favor of CuTe's compile-time layout algebra.

2. **Dispatch policy encodes the full hardware/software strategy.** The tag type `MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>` selects the exact combination of TMA loading, GMMA computation, warp specialization, and pipeline depth — entirely at compile time. This makes CUTLASS effectively a compile-time kernel specialization engine.

3. **CuTe DSL generates PTX via MLIR, not NVCC template expansion.** The Python-native `CuTeDSL` package uses MLIR dialects (`_cutlass_ir`) and an AST preprocessor to lower Python kernel descriptions to CUDA PTX. This enables runtime compilation (JIT), ahead-of-time export, and direct DLPack/`from_dlpack` interop with PyTorch and JAX tensors without any C++ glue code.

4. **The Epilogue Visitor Tree (EVT) is a general fused-op compiler.** Rather than fixed epilogues (scale + bias + relu), the EVT lets users compose arbitrary DAGs of elementwise operations as C++ template nodes (load, store, compute). The Hopper epilogue uses TMA for output writes, meaning fused ops execute in registers while output I/O stays asynchronous — matching what hand-tuned FlashAttention kernels do manually.

5. **Persistent/StreamK scheduling hides tail quantization.** Classic tiled GEMM wastes SMs when `M*N` is not a multiple of the tile grid. CUTLASS 3.x's `PersistentTileSchedulerSm90` and `StreamKScheduler` keep all SMs busy by dynamically assigning partial K-slices, recovering near-peak utilization on non-square problem shapes common in LLM inference (e.g., decode steps with small batch size).

6. **Blackwell introduces a new memory tier: TMEM.** `SM100_MMA_*` structs in `cute/arch/mma_sm100_umma.hpp` pass a `tmem_c` uint32 pointer for the accumulator, meaning the C matrix lives in a dedicated on-chip Tensor Memory rather than the register file. This is a hardware architectural change that required new CUTLASS abstractions separate from Hopper's `wgmma`.

---

## Relation to Hardware Architecture

CUTLASS's design is shaped directly by the NVIDIA GPU memory hierarchy and Tensor Core evolution:

- **Register file → Shared Memory → L2 → HBM hierarchy maps to CUTLASS's hierarchy levels.** `thread/` operates in registers, `warp/` coordinates register tiles, `threadblock/` manages shared memory tiles, and `device/` orchestrates HBM loading. Each level is independently tunable via tile sizes.

- **Tensor Core shape constraints drive tile sizing.** The minimum TiledMMA tile must be a multiple of the underlying MMA instruction shape (e.g., `m64n8k16` for wgmma on H100). CuTe's compile-time Layout algebra enforces these constraints statically via `static_assert`.

- **Hopper TMA eliminated the load-loop bottleneck.** Pre-Hopper kernels used `cp.async` loads that required threads to issue addresses in a loop. Hopper's TMA engine accepts a descriptor computed once on the host and issued by a single thread, freeing all compute warp threads to focus on MMA. PipelineTmaAsync's `mbarrier`-based handoff directly models this hardware producer-consumer decoupling.

- **Blackwell's UMMA + TMEM enables CTA-group-scoped MMA.** The `tcgen05.mma.cta_group::1` instruction operates at the CTA-group (multi-CTA cluster) scope and accumulates into TMEM rather than registers. This allows much larger effective accumulator tiles and reduces register pressure, directly influencing CUTLASS's new `sm100_mma_warpspecialized` collective designs.

- **Warp specialization mirrors hardware datapath width.** On Hopper, the warpgroup (128 threads = 4 warps) is the natural unit for `wgmma`. CUTLASS's warp-specialized kernel templates assign entire warp groups to either DMA or MMA roles, saturating both the TMA unit and the Tensor Core datapaths simultaneously.
