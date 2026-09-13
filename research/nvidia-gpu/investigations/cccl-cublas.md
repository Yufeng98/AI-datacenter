# CCCL (CUDA C++ Core Libraries) and cuBLAS

**Source**: CCCL v3.4.0 — https://github.com/NVIDIA/cccl (shallow clone at /tmp/cccl-investigate)  
**cuBLAS**: v13.2 — https://docs.nvidia.com/cuda/cublas/  
**Date investigated**: 2026-04-05

---

## Overview

CCCL and cuBLAS occupy complementary but distinct layers of the NVIDIA GPU software stack.

**CCCL** (CUDA C++ Core Libraries, v3.4.0) is the foundational parallel-primitives layer. It unifies three formerly separate libraries — CUB, Thrust, and libcu++ — into a single repository. Its mission is to provide GPU developers with safe, high-performance, reusable building blocks analogous to what the C++ Standard Library provides on the host side. CCCL ships as a header-only library bundled with the CUDA Toolkit.

**cuBLAS** (v13.2, March 2026) is NVIDIA's GPU-accelerated implementation of the BLAS (Basic Linear Algebra Subprograms) standard. It sits one abstraction layer above CCCL: where CCCL provides general-purpose parallel primitives, cuBLAS provides the specific, highly-tuned matrix and vector operations that power neural network training and inference. Internally, cuBLAS dispatches GEMM work through CUTLASS-generated kernels, with CUB-style primitives underpinning auxiliary operations such as reductions and index computations.

Together they form the backbone of GPU-accelerated linear algebra in every major deep learning framework.

---

## Architecture

### CCCL: Three-Layer Structure

CCCL is structured as three cooperating components with a clear abstraction hierarchy:

#### 1. libcu++ — CUDA C++ Standard Library (lowest, widest scope)
- Provides a device-side implementation of the C++ Standard Library usable in both `__host__` and `__device__` code.
- Key namespaces: `cuda::std::` (STL containers, algorithms, type traits, atomics, chrono, memory) and `cuda::` (CUDA-specific extensions).
- CUDA-specific extensions include: `cuda::atomic`, `cuda::atomic_ref`, `cuda::barrier`, `cuda::semaphore`, `cuda::pipeline`, cache control (`cuda::access_property`), TMA (Tensor Memory Accelerator) wrappers, PTX intrinsics (`cuda::ptx`), and warp-level primitives (`cuda::warp`).
- Headers reside in `libcudacxx/include/cuda/` and `libcudacxx/include/cuda/std/`.
- No kernel launches; purely inline device-callable code.

#### 2. CUB — CUDA Unbound (device-wide parallel primitives)
- A CUDA-specific library targeting "speed-of-light" performance for standard parallel algorithms at three granularities:
  - **Thread-level**: thin wrappers around PTX/intrinsics for I/O and math.
  - **Block-level**: cooperative algorithms running within a single thread block (`cub::BlockReduce`, `cub::BlockScan`, `cub::BlockRadixSort`). Requires shared-memory `TempStorage` objects managed by the caller.
  - **Device-level**: end-to-end kernel-launching algorithms that operate over arbitrary-length device arrays, including:
    - `cub::DeviceReduce` (sum, min, max, arg-extremum, reduce-by-key)
    - `cub::DeviceScan` (inclusive/exclusive prefix scan, segmented variants)
    - `cub::DeviceRadixSort` / `cub::DeviceSegmentedRadixSort` (ascending/descending, all numeric types including `__half`)
    - `cub::DeviceMergeSort` / `cub::DeviceSegmentedSort`
    - `cub::DeviceHistogram`
    - `cub::DevicePartition`, `cub::DeviceSelect`, `cub::DeviceTopK`
    - `cub::DeviceMemcpy`, `cub::DeviceTransform`
    - `cub::DeviceAdjacentDifference`, `cub::DeviceRunLengthEncode`
- Device-level primitives accept a user-provided scratch buffer (`d_temp_storage`) sized with a two-pass API (first call with `nullptr` returns size, second call performs work).
- Internally uses dispatch layers (`cub/device/dispatch/`) that select tuning parameters based on the runtime `cudaDeviceProp` (SM count, shared-memory size, warp size) and compiled architecture flags.
- CUB is architecture-aware: `dispatch_reduce_nondeterministic.cuh` vs `dispatch_reduce_deterministic.cuh` expose determinism control via `cuda::execution::determinism`.

#### 3. Thrust — High-Level Parallel Algorithms (highest abstraction)
- STL-inspired parallel algorithms library; directly inspired the C++17 `std::execution::par_unseq` model.
- Provides device-vector and host-vector containers (`thrust::device_vector<T>`, `thrust::host_vector<T>`, `thrust::universal_vector<T>`), iterators, and the full range of STL algorithms over these containers.
- Key algorithms: `thrust::reduce`, `thrust::scan`, `thrust::sort`, `thrust::sort_by_key`, `thrust::transform`, `thrust::for_each`, `thrust::merge`, `thrust::partition`, `thrust::scatter`, `thrust::gather`, `thrust::unique`, `thrust::binary_search`, `thrust::inner_product`, `thrust::transform_reduce`, `thrust::transform_scan`.
- Multiple execution backends via policies: `thrust::cuda::par` (default), `thrust::tbb::par` (Intel TBB), `thrust::omp::par` (OpenMP), `thrust::host` (serial CPU). Backend is selected at call site, enabling cross-platform portability.
- Thrust is a thin coordination layer: most CUDA backend algorithms delegate immediately to CUB device-wide primitives, inheriting CUB's hardware tuning.

### cuBLAS: Three API Layers

cuBLAS exposes GPU-accelerated BLAS at three distinct API levels:

#### Layer 1 — Legacy v1 API (deprecated)
- Flat C API using global state (a singleton internal context).
- Retained for backwards compatibility; not recommended for new code.

#### Layer 2 — cuBLAS v2 API (current standard interface)
- Explicit `cublasHandle_t` context object; thread-safe when handles are per-thread.
- Covers the full BLAS specification:
  - **Level 1** (vector-vector): `cublasIsamax`, `cublasSdot`, `cublasSaxpy`, `cublasSnrm2`, `cublasSscal`, `cublasSswap`, `cublasSasum`, and their double/half/complex variants.
  - **Level 2** (matrix-vector): `cublasSgemv`, `cublasSsymv`, `cublasSger`, `cublasStrsv`, `cublasSsymm`, and variants.
  - **Level 3** (matrix-matrix): `cublasSgemm`, `cublasDgemm`, `cublasHgemm`, `cublasSgemmBatched`, `cublasSgemmStridedBatched`, `cublasSsymm`, `cublasSsyrk`, `cublasSgeam`, `cublasSdgmm`.
  - Extended precision: `cublasGemmEx` — explicitly selects compute type (`CUBLAS_COMPUTE_32F`, `CUBLAS_COMPUTE_32F_FAST_16F`, `CUBLAS_COMPUTE_32F_FAST_TF32`, `CUBLAS_COMPUTE_16F`, `CUBLAS_COMPUTE_64F`) and data types (`CUDA_R_16F`, `CUDA_R_8F_E4M3`, `CUDA_R_32F`, etc.).
- Internally dispatches to a runtime heuristics engine that selects among pre-compiled CUTLASS kernel variants based on problem dimensions, data types, and GPU architecture.

#### Layer 3 — cublasLt (Lightweight, flexible GEMM-only API)
- Dedicated to GEMM and matmul operations; not a full BLAS replacement but a high-flexibility extension.
- Core objects: `cublasLtHandle_t`, `cublasLtMatmulDesc_t` (operation descriptor: transpose flags, compute type, epilogue), `cublasLtMatrixLayout_t` (row/column-major, leading dimension, batch stride), `cublasLtMatmulPreference_t` (workspace size budget, algorithm constraints).
- Algorithm selection: `cublasLtMatmulAlgoGetHeuristic()` returns an ordered list of candidate algorithms ranked by the cuBLAS heuristics engine (93% accuracy across a large problem space in benchmarks). Users can iterate candidates and benchmark at runtime to find the optimal kernel for their specific shape.
- Epilogue fusion: cublasLt can fuse activation functions (ReLU, GELU, bias addition) as epilogue operators directly within the GEMM kernel, eliminating a separate element-wise kernel launch and reducing memory traffic.
- Once a `cublasLtMatmulDesc_t` + layout + algorithm tuple is determined, it can be cached and reused for repeated calls with the same shape and precision — analogous to FFT plan reuse in cuFFT.
- Mixed-precision support via `cublasComputeType_t`: `CUBLAS_COMPUTE_32F_FAST_TF32` routes single-precision operands through TF32 Tensor Cores (19-bit mantissa × 10-bit mantissa → 32-bit accumulate) on Ampere+ hardware. Environment variable `NVIDIA_TF32_OVERRIDE=0` globally disables TF32 acceleration.
- FP8 GEMM (via `cublasLtMatmul` with `CUDA_R_8F_E4M3`/`CUDA_R_8F_E5M2` layouts) was introduced for Ada/Hopper and is the primary throughput path on H100.

---

## Data Flow: cuBLAS GEMM Call

A complete trace from Python/C++ call to Tensor Core execution:

```
1. User calls: torch.matmul(A, B)  [PyTorch, GPU tensors]
       │
       ▼
2. PyTorch ATen backend dispatches to:
   at::cuda::blas::gemm() → cublasGemmEx() or cublasLtMatmul()
   (selection via TORCH_BLAS_PREFER_CUBLASLT env var or torch.backends.cuda.preferred_blas_library)
       │
       ▼
3. cuBLAS v2 / cublasLt entry point
   - Validate inputs (alignment, leading dimensions, batch count)
   - Query cublasHandle_t's cached device properties (SM count, compute capability)
       │
       ▼
4. Heuristic dispatch (nvMatmulHeuristics / cublasLtMatmulAlgoGetHeuristic)
   - Problem descriptor: M×N×K, data types (e.g., FP16 × FP16 → FP32 accumulate), transpose flags
   - Selects from a database of pre-compiled CUTLASS kernel variants indexed by:
     (sm_arch, compute_type, data_type_A, data_type_B, data_type_C, tile_M, tile_N, tile_K, stages, split_k_factor)
   - Returns ranked list of (algorithm_id, expected_perf) tuples
       │
       ▼
5. CUTLASS kernel instantiation (internal to cuBLAS binary)
   - Selected kernel is a CUTLASS 3.x template instantiation, e.g.:
     cutlass_80_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_align8
     (sm=80, Tensor Op, FP16 in/out, warpshape 16×8×16 MMA instruction, 256×128 tile, 3 pipeline stages)
   - For SM90 (Hopper): uses TMA (Tensor Memory Accelerator) warp-specialized kernels with persistent scheduling
   - Kernel grid/block dimensions computed from tile sizes and problem shape
       │
       ▼
6. Tensor Core execution
   - Each warp executes `mma.sync` PTX instruction (Volta/Turing/Ampere) or `wgmma.mma_async` (Hopper SM90)
   - Operand tiles loaded from DRAM → L2 → shared memory (via async-copy / TMA on Hopper)
   - Register-level matrix fragments fed to Tensor Core hardware units
   - Accumulator tiles in registers → written back to global memory (with optional epilogue: bias add, ReLU)
```

Key architectural insight: cuBLAS itself contains no hand-written GEMM inner loops. The GEMM kernels are pre-compiled CUTLASS template instantiations embedded in the cuBLAS shared library binary, selected at runtime via heuristics.

---

## Hardware Interface

### Compute Capability Adaptation

cuBLAS ships with pre-compiled GEMM kernels for each supported GPU architecture:

| Architecture | SM | Key Tensor Core Feature | cuBLAS/CUTLASS Kernel Style |
|---|---|---|---|
| Volta | sm_70 | FP16 → FP32, first Tensor Cores | `mma.sync` wmma-based kernels |
| Turing | sm_75 | INT8, INT4 matmul | `mma.sync` with 16×8×8 tiles |
| Ampere | sm_80/sm_86 | BF16, TF32, sparsity (2:4) | CUTLASS 2.x, async-copy, 3-stage pipeline |
| Ada Lovelace | sm_89 | FP8 (E4M3/E5M2) | CUTLASS 3.x FP8 kernels |
| Hopper | sm_90 | FP8, TMA, wgmma, warp specialization | CUTLASS 3.x TMA warp-specialized persistent kernels |
| Blackwell | sm_100 | FP4 (E2M1), UMMA | Early access in CUTLASS 4.x / cuBLAS 13.x |

At runtime, cuBLAS queries `cudaDeviceProp.major` and `.minor` to select the correct kernel binary. For forward compatibility, cuBLAS falls back to the closest older architecture if the current GPU is newer than any embedded kernel.

### Tensor Core Precision Modes (cuBLAS compute types)

| `cublasComputeType_t` | Input Types | Accumulator | Tensor Cores? | Notes |
|---|---|---|---|---|
| `CUBLAS_COMPUTE_16F` | FP16 | FP16 | Yes (sm_70+) | Fastest FP16 path, reduced precision accumulate |
| `CUBLAS_COMPUTE_32F` | FP32 | FP32 | No | Scalar CUDA cores, exact IEEE-754 |
| `CUBLAS_COMPUTE_32F_FAST_16F` | FP32→FP16 | FP32 | Yes (sm_70+) | Inputs downcast to FP16, accumulate in FP32 |
| `CUBLAS_COMPUTE_32F_FAST_TF32` | FP32→TF32 | FP32 | Yes (sm_80+) | Default on Ampere+; ~8× vs scalar; toggled by `NVIDIA_TF32_OVERRIDE` |
| `CUBLAS_COMPUTE_64F` | FP64 | FP64 | Yes (sm_80+) | FP64 Tensor Cores on A100/H100 (via `dmma`) |

FP8 mode (Hopper/Ada): `cublasLtMatmul` with `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2` input types and `CUBLAS_COMPUTE_32F` accumulator. Requires explicit scaling factors per tensor for dynamic range management.

### CUB Hardware Tuning

CUB's device-level dispatch layers compile separate tuned parameter sets for each target SM architecture:
- **Tile size** (items per thread × threads per block) scaled to match L1/shared-memory capacity.
- **Unroll depth** tuned against register pressure for each SM's register file size.
- **Pipeline stages** (async-copy depth) set to 1 for pre-Ampere, 2–4 for Ampere+ to hide global-memory latency.
- `dispatch_reduce_deterministic.cuh` vs non-deterministic variants select between tree-reduction (deterministic, hardware-independent) and warp-shuffle-based reduction (non-deterministic but lower latency), controlled via `cuda::execution::determinism` policy.

---

## Key Findings

### cublasLt: Flexible Algorithm Selection is the Production Path
For high-performance inference and training serving, cublasLt's `cublasLtMatmulAlgoGetHeuristic` + optional benchmark loop is the standard approach in production frameworks. PyTorch's `torch.backends.cuda.preferred_blas_library("cublaslt")` routes all GEMM through cublasLt. The heuristic recommender achieves 93% accuracy in selecting the optimal kernel across a large problem space, but the benchmark-then-cache pattern eliminates this residual gap for fixed shapes (common in transformer inference).

### CUB as the Universal Building Block
CUB's device-wide primitives are not just used directly by application code — they are the building blocks for higher-level libraries. cuDNN uses CUB reductions for batch normalization and layer normalization. Thrust's CUDA backend is a thin wrapper over CUB (e.g., `thrust::reduce` → `cub::DeviceReduce::Sum`). This layering means CUB tuning improvements automatically propagate up the entire software stack.

### Thrust as the High-Level Onramp
For developers prototyping GPU algorithms, Thrust provides an STL-familiar entry point that runs on GPU without requiring any knowledge of CUDA kernel programming. Algorithms written with Thrust can be trivially ported between GPU (CUDA), multicore CPU (TBB), and shared-memory parallel (OpenMP) backends by changing the execution policy. This is the intended "delightful" developer experience CCCL advertises.

### Epilogue Fusion in cublasLt Reduces Memory Pressure
Fusing bias addition and activation (GELU, ReLU) as cublasLt epilogues eliminates the round-trip to DRAM between GEMM and the element-wise kernel. For a 4096×4096 FP16 GEMM on H100, this fusion saves ~128 MB of DRAM traffic (the output matrix + input to activation kernel), which at 3.35 TB/s peak bandwidth corresponds to ~38 µs of savings per layer.

### Determinism is Explicit and Opt-In
CUB v3.4 introduces a `cuda::execution::determinism` policy: `dispatch_reduce_deterministic.cuh` vs `dispatch_reduce_nondeterministic.cuh`. This reflects the GPU computing community's growing need for reproducible results in scientific computing and model debugging.

---

## Relation to Hardware: Expanding Precision Formats Drive cuBLAS Evolution

The single strongest driver of cuBLAS API evolution is the introduction of new Tensor Core precision formats with each GPU generation:

- **Volta (2017)**: FP16 Tensor Cores → `cublasHgemm`, `cublasGemmEx` with `CUBLAS_COMPUTE_16F`.
- **Ampere (2020)**: TF32 and BF16 Tensor Cores → `CUBLAS_COMPUTE_32F_FAST_TF32` becomes the default; BF16 training enables LLM scaling.
- **Hopper (2022)**: FP8 Tensor Cores + TMA hardware → cublasLt gains FP8 layouts; TMA-based async loads enable deeper software pipelines in CUTLASS 3.x kernels, boosting utilization.
- **Blackwell (2024–2025)**: FP4 (E2M1) Tensor Cores + UMMA (Unified Matrix Multiply Accumulate) → CUTLASS 4.x and cuBLAS 13.x introduce FP4 GEMM in early access, with 2× throughput improvement over FP8 at the cost of further reduced dynamic range.

Each new precision format requires: (1) new CUTLASS kernel templates compiled into the cuBLAS binary, (2) new `cublasComputeType_t` or layout enumerants in the API, (3) new heuristic training data to guide algorithm selection. This tight coupling between Tensor Core hardware evolution and cuBLAS API versioning explains why cuBLAS releases are synchronized with CUDA Toolkit releases.

The broader pattern: **hardware precision capabilities (FP4 → FP8 → FP16 → TF32 → FP32) are exposed as a precision stack through cublasLt's flexible type system**, allowing frameworks like PyTorch and JAX to adopt new precisions by changing a single `cublasComputeType_t` argument rather than rewriting GEMM kernels.

---

## References

- CCCL v3.4.0 source: https://github.com/NVIDIA/cccl
- cuBLAS 13.2 documentation: https://docs.nvidia.com/cuda/cublas/
- CUTLASS overview: https://docs.nvidia.com/cutlass/latest/overview.html
- cuBLAS 12.0 Hopper features blog: https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/
- GEMM heuristics and CUTLASS 4.2: https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/
- Grouped GEMM APIs blog: https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
- PyTorch backends (BLAS selection): https://docs.pytorch.org/docs/stable/backends.html
