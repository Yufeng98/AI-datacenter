# NVIDIA GPU Software and Hardware Stack Summary

*as_of: 2026-09-13*

---

## Overview

NVIDIA GPUs are the dominant accelerators for AI training and inference in datacenter-scale deployments. Their architecture is organized around arrays of **Streaming Multiprocessors (SMs)** -- independent compute engines containing CUDA cores, Tensor Cores, register files, and shared memory -- connected to high-bandwidth memory (HBM) and to each other via NVLink. The current datacenter generations are **Hopper** (H100/H200, 132 SMs, 80-141 GB HBM3/3e) and **Blackwell** (B200/GB200, 2x148 SMs dual-chiplet, 192 GB HBM3e at 8 TB/s).

The NVIDIA ecosystem's defining characteristic is the **CUDA platform**: a vertically integrated software stack spanning from Python-level frameworks down to hardware-specific microcode. CUDA has been shipping since 2007 and provides the programming model (thread/warp/block hierarchy), compilation toolchain (nvcc, PTX, SASS), runtime (libcudart, libcuda), and a comprehensive library ecosystem (cuBLAS, cuDNN, CUTLASS, NCCL). This deep vertical integration -- where each software layer is co-designed with the SM microarchitecture it targets -- is the primary reason NVIDIA GPUs maintain performance leadership despite competitors with comparable raw FLOPS.

---

## Software Stack

### Framework Integration

NVIDIA GPUs are the primary backend for all major ML frameworks:

- **PyTorch**: The most widely used training framework. `torch.cuda` provides device management; `c10::cuda::CUDAStream` and `CUDAGuard` manage streams and device context. PyTorch's CUDA Caching Allocator uses the Driver API's Virtual Memory Management (VMM) to pool HBM efficiently. `torch.compile()` with TorchInductor generates Triton kernels for fused operations. `torch.nn.functional.scaled_dot_product_attention` dispatches to cuDNN's fused Flash Attention on H100+.
- **JAX**: Uses XLA as its compilation backend, which lowers HLO operations to cuBLAS/cuDNN calls and generates PTX via LLVM. JAX's `jax.experimental.pallas` provides a Triton-like tiled kernel authoring interface.
- **TensorFlow**: Uses StreamExecutor to wrap the CUDA Runtime; delegates compute to cuBLAS and cuDNN through XLA. Supports CUDA Graphs via `tf.function`.
- **CUDA C/C++ Language Extensions**: The `__global__`, `__device__`, `__shared__` annotations and `<<<grid, block>>>` launch syntax allow direct GPU kernel authoring in C++. Cooperative Groups provide flexible synchronization beyond `__syncthreads()`.

### Compiler / IR

- **nvcc / NVVM**: NVIDIA's CUDA C++ compiler. Separates host and device code, compiles device code through an LLVM-based IR (NVVM) to PTX and then to SASS (native microcode). Produces fat binaries bundling multiple SASS cubins and PTX for forward compatibility.
- **Triton**: OpenAI's Python DSL and MLIR-based compiler. A `@triton.jit` kernel computes one output tile per CTA using blocked operations (`tl.load`, `tl.dot`). The compiler automatically handles shared memory allocation, memory coalescing, software pipelining, and Tensor Core instruction selection (MMA v2 for Ampere, v3/WGMMA for Hopper, v5/tcgen05 for Blackwell). Triton is the primary code generation target for PyTorch 2.x via TorchInductor.
- **XLA**: Google's compiler for TensorFlow and JAX. Lowers HLO (High Level Operations) through LLVM to PTX. Integrates with cuBLAS and cuDNN for key operations.
- **TensorRT**: NVIDIA's inference optimization engine. Performs graph-level optimizations (layer fusion, precision calibration, kernel auto-tuning) and dispatches to cuDNN/CUTLASS engines.
- **CUDA Tile IR / cuTile** (NVIDIA's first-party tile programming stack; *added to this survey 2026-08-08 — the components themselves predate the survey baseline*): `NVIDIA/cuda-tile` is an MLIR-based IR and compiler infrastructure "targeting NVIDIA tensor core units" (repo created 2025-11-05; v13.1.0 2026-01-14, v13.2.0 2026-03-24, v13.3.3 2026-07-22), assembled by a separate **CUDA TILE-IR AS** toolkit component that has shipped in GA since CUDA 13.2 (13.2.51) and is at 13.3.36 in CUDA 13.3 Update 1. `NVIDIA/cutile-python` (created 2025-06-13; v1.5.0 2026-07-07) is the Python DSL for authoring tile-level kernels on top of it. Tile IR sits alongside PTX as a second compiler-facing virtual ISA, making it NVIDIA's own answer to Triton and to CUTLASS's CuTe DSL. **Limitations:** per the cutile-python README, the `tileiras` compiler (version 13.2) supports only Blackwell and Ampere/Ada GPUs — "Hopper GPU will be supported in the coming versions" — and requires CUDA Toolkit 13.1+ with an r580+ driver. No NVIDIA release note announces an experimental-to-stable transition for cuTile; the README distinguishes a stable core `cuda.tile` API from separately marked experimental features.

### Op Library

- **cuDNN** (CUDA Deep Neural Network library): Closed-source library providing hand-tuned implementations of convolutions, matrix multiplications, normalization, and attention. The v8+ Graph API enables declarative dataflow graphs with up to 50 fused operations per graph, eliminating intermediate global memory round-trips. cuDNN's engine heuristics select from pre-compiled and NVRTC runtime-compiled kernels ranked by expected performance. On Hopper, cuDNN's fused Flash Attention (SDPA) provides up to 75% speedup over FlashAttention v2 by exploiting WGMMA and TMA. Blackwell support adds `block_scale_quantize`/`block_scale_dequantize` nodes for MXFP8/NVFP4 block-scaled Tensor Cores.
- **cuBLAS / cublasLt**: GPU-accelerated BLAS covering Levels 1-3 plus the flexible `cublasLt` GEMM API. Internally dispatches to pre-compiled CUTLASS kernel variants selected by a heuristics engine (~93% accuracy). `cublasLt` supports epilogue fusion (bias + activation within the GEMM kernel), FP8 layouts for Hopper, and the full precision stack from FP4 to FP64. PyTorch routes all GEMM through cuBLAS v2 or cublasLt.

### Kernel Library

- **CUTLASS / CuTe**: Header-only C++ template library for composable high-performance GEMM and convolution kernels. CuTe provides the foundational layout algebra (Shape + Stride tuples) and Tensor abstraction. `MMA_Atom` / `TiledMMA` wrap hardware Tensor Core instructions; `Copy_Atom` / `TiledCopy` wrap memory copy instructions (cp.async, TMA). `CollectiveMma` and `CollectiveEpilogue` compose these into full kernel mainloops. The Epilogue Visitor Tree (EVT) enables arbitrary fused post-GEMM operations. CUTLASS is the implementation substrate used internally by cuBLAS and NVIDIA's Transformer Engine. CuTe DSL enables Python-native kernel authoring via MLIR compilation.
- **CUB / Thrust** (CCCL): CUB provides "speed-of-light" parallel primitives at thread, block, and device granularity -- `DeviceReduce`, `DeviceScan`, `DeviceRadixSort`, etc. -- with architecture-aware tuning. Thrust provides an STL-like high-level API (`thrust::reduce`, `thrust::sort`) that delegates to CUB on the CUDA backend. CUB primitives are used internally by cuDNN, cuBLAS, and across the NVIDIA ecosystem.
- **FlashAttention**: IO-aware attention kernels that tile Q/K/V through shared memory to avoid materializing the full attention matrix in HBM. Available as standalone kernels and integrated into cuDNN's SDPA engine.
- **cuTile** (`NVIDIA/cutile-python`): a Python kernel-authoring DSL — "a programming model for writing parallel kernels for NVIDIA GPUs" — whose kernels lower to Tile IR rather than to PTX directly. Positioned as a first-party sibling to CuTe DSL and Triton for tile-level kernel work; see Compiler / IR above for the hardware-support caveats.

### Runtime

- **CUDA Runtime API** (`libcudart.so`): The primary user-facing API. Provides `cudaMalloc`, `cudaMemcpy`, kernel launch via `<<<>>>` syntax (which compiles to `cudaLaunchKernel`), streams, events, and CUDA Graphs. Manages one primary context per (device, process) pair automatically. CUDA Graphs capture DAGs of GPU operations and replay them with a single launch call, eliminating ~5-10 us per-operation CPU overhead critical for inference workloads.
- **CUDA Driver API** (`libcuda.so`): Low-level API with explicit context management, dynamic module loading (`cuModuleLoad`), and VMM (Virtual Memory Management: `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap`). PyTorch's CUDA Caching Allocator uses VMM to grow HBM pools without data movement. The Driver API ships with the NVIDIA driver (not the Toolkit) and is backward-compatible across driver versions.
- **MPS / MIG**: Multi-Process Service (MPS) multiplexes multiple CUDA processes onto one GPU via Hyper-Q. Multi-Instance GPU (MIG) provides hardware-level GPU partitioning into up to 7 isolated instances with dedicated SMs, L2, and HBM on A100/H100.
- **Unified Memory**: `cudaMallocManaged` provides a single pointer accessible from both CPU and GPU. The UVM kernel driver handles page migration via hardware GPU page faults. `cudaMemPrefetchAsync` and `cudaMemAdvise` allow programmer-guided migration policy.

### Driver / Firmware

- **nvidia.ko** (Open GPU Kernel Modules): Open-source Linux kernel driver providing PCI device management, BAR mapping, interrupt handling, ioctl dispatch, and the CPU-side Resource Manager (RM). The RM uses the NVOC object model -- an in-house single-inheritance C OOP framework with HAL polymorphism across GPU generations (Turing through Blackwell).
- **GSP Firmware**: On Turing+ GPUs, the bulk of GPU resource management runs on the on-chip GPU System Processor (GSP), a RISC-V microcontroller. The CPU kernel module communicates with GSP via shared-memory RPC message queues in VRAM. This GSP-split design means the open-source kernel module is primarily an RPC client and OS integration shim.
- **nvidia-uvm.ko**: Unified Virtual Memory kernel module implementing GPU page fault handling for CUDA Managed Memory, with per-architecture HAL files (Ampere, Hopper, Blackwell).
- **nvidia-peermem.ko**: GPUDirect RDMA peer memory registration for NIC-to-GPU direct DMA.
- **User-mode doorbells**: Once a GPFIFO channel is set up, the CUDA runtime writes directly to a BAR1-mapped page to submit work -- no kernel transition needed for steady-state operation.

### Communication

- **NCCL** (NVIDIA Collective Communications Library): The de-facto standard collective communication library for multi-GPU and multi-node GPU clusters. Provides AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv. At communicator initialization, NCCL discovers the full hardware topology (PCI tree, NVLink fabric via NVML) and selects optimal algorithms (Ring, Tree, NVLS, CollNet) and protocols (Simple for large messages, LL for small, LL128 for medium NVLink-optimized).
  - **NVLS (NVLink SHARP)**: Uses CUDA Multicast Objects to perform hardware-accelerated AllReduce on NVSwitch, achieving near-wire-speed single-pass reduction on DGX H100/B200.
  - **MNNVL (Multi-Node NVLink)**: On GB200 NVL72, NCCL treats cross-node GPUs as local NVLink peers via FABRIC-type memory handles.
  - **GIN / GDAKI**: GPU Initiated Networking allows SM threads to post InfiniBand work requests directly, bypassing the CPU proxy thread bottleneck.
  - **Plugin architecture**: Versioned interfaces for network transport, tuning, and profiling enable vendor-specific extensions (AWS EFA, Google TCPx) without forking NCCL.
  - **Zero-SM collectives (NCCL 2.30.7-1, 2026-06-04)**: hierarchical AllGather and All2all that use an RMA CPU proxy between nodes and Copy Engines inside a node, enabled with `NCCL_CTA_POLICY_ZERO`. Collectives no longer occupy SMs, so communication overlaps compute without stealing occupancy — architecturally the most significant NCCL change of 2026.
  - **Elastic buffers and TMA symmetric kernels (NCCL 2.30.3-1, 2026-04-15)**: large tensors are split into multi-segment windows with the active region resident in GPU memory and the remainder in host memory; select built-in symmetric kernels use TMA (`NCCL_SYM_TMA_ENABLE=1`). The same release isolates GIN contexts between device communicators on a host communicator and adds Dynamic Direct Path (DDP).
- **NCCL EP** (release tag `nccl-ep-v0.1.0` in the NVIDIA/nccl repository, 2026-06-08): "a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication," providing dispatch/combine primitives for Expert Parallelism built on the NCCL Device API (LSA + GIN). CUDA-Graph-compatible handle management splits a control path (`ncclEpInitHandle`) from a data path (`ncclEpUpdateHandle`); SM count is user-settable; NCCL Window association enables zero-copy; Flat / Expert-major / Rank-major dispatch layouts and active-rank masking for fault tolerance. MoE-specific collectives becoming a first-class NVIDIA library is a genuine programming-model shift.
- **nccl4py** (`nccl4py-v0.3.1`, 2026-06-11): the Python API for NCCL, adding the `nccl.ep` package, `nccl.core.device.cute` (CuTeDSL kernels that call NCCL device APIs), and free-threaded CPython support. Not new in this window — v0.2.0 (2026-04-24) already exposed RMA/GIN/elastic-communicator bindings.
- **NVIDIA/nccl-extensions** (repository created 2026-07-07): "Communication patterns for AI, built on top of NCCL device and host APIs," actively developed as of 2026-08-08. Likely the consolidation point for NCCL EP-style extension libraries; watch item.
- **GPUDirect RDMA**: Allows NICs to DMA directly to/from GPU HBM, eliminating CPU bounce buffers for cross-node communication.

### Assembler / ISA

- **PTX** (Parallel Thread Execution): NVIDIA's virtual ISA -- the stable, architecture-independent intermediate representation between CUDA C++ and hardware. PTX serves two roles: (1) compiler target for nvcc, producing human-readable assembly; (2) forward-compatibility vehicle via driver JIT compilation to SASS. PTX code targeting `compute_X` runs on any GPU with compute capability >= X.
  - Key instruction families: `mma.sync` (Volta-Ampere warp-level Tensor Core), `wgmma.mma_async` (Hopper warp-group MMA), `tcgen05.mma` (Blackwell single-thread-issue with TMEM), `cp.async` / `cp.async.bulk.tensor` (async memory copy / TMA).
  - Current published specification is **PTX ISA v9.3**. The CUDA 13.4 Developer Preview (2026-07-16) lists "New and updated PTX ISA, including some Rubin capabilities" — the first public acknowledgement of Rubin-targeted PTX. **No Rubin compute capability (`sm_`) is named in any GA CUDA release**; GA release notes reference sm_90, sm_100, sm_103, sm_120 and sm_121.
- **Tile IR**: a second, tile-level virtual ISA that has shipped alongside PTX in the CUDA Toolkit since 13.2, assembled by the `tileiras` / **CUDA TILE-IR AS** component (13.2.51 in CUDA 13.2 GA → 13.3.36 in 13.3 Update 1 → 13.4.46 in the 13.4 preview). Where PTX is scalar-thread-centric, Tile IR represents tiles, tile views/layouts/sub-views, tile-level MMA, reductions, and tile-level synchronization/communication directly.
- **SASS** (Streaming Assembly): The native hardware ISA executed by SMs. Not fully publicly documented. SASS is not forward-compatible -- a binary for sm_90 will not run on sm_80. `ptxas` performs register allocation, instruction scheduling, and optimization when assembling PTX to SASS.
- **Fat binary**: Container format bundling multiple SASS cubins (for known GPU targets) plus PTX (for forward compatibility).
- **Inline PTX**: `asm volatile("...")` allows embedding PTX directly in CUDA C++ device code, used extensively by CUTLASS and performance-critical libraries.

---

## Hardware Architecture

### Compute Engine

Each SM contains 128 FP32 CUDA cores, 4 Tensor Cores, 4 warp schedulers (each supporting up to 16 warps, 64 total per SM), Special Function Units (SFUs), and load/store units. H100 has 132 SMs; B200 has 148 SMs per die (296 total in dual-chiplet package).

Tensor Core evolution is the primary hardware driver of AI performance improvements:
- **4th gen (Hopper)**: FP8 (E4M3/E5M2), TF32, BF16, FP16, FP64. ~989 TFLOPS BF16.
- **5th gen (Blackwell)**: Adds MXFP4 and MXFP6 microscaling formats. Peak ~20 PFLOPS FP4, ~10 PFLOPS FP8.

### Data Path

NVIDIA GPUs implement **SIMT** (Single Instruction, Multiple Threads). Threads are grouped into **warps** of 32, executing the same instruction in lock-step with independent register contexts. Divergent branches are serialized via predicated execution. 4 warp schedulers per SM each select a ready warp every cycle, with dual-issue capability.

Hopper introduced the **Tensor Memory Accelerator (TMA)** for asynchronous 1D-5D tensor moves between global and shared memory, and **Thread Block Clusters** enabling cooperative groups of up to 8 CTAs spanning multiple SMs with distributed shared memory.

### On-chip Memory

- **Register file**: 256 KB per SM (32 banks of 2 KB), sub-cycle latency.
- **Shared Memory / L1**: Unified SRAM pool per SM. H100: up to 228 KB shared memory from 256 KB total pool. Configurable split with L1 at kernel launch.
- **L2 Cache**: Shared across all SMs. H100: 50 MB; B200: ~96-128 MB (2-2.5x H100) with 4 partitions. Hopper introduced L2 residency controls for pinning data.
- **TMEM (Blackwell)**: Dedicated on-SM tensor memory for MMA accumulator results, decoupling accumulators from the general register file.

### Off-chip Memory

All current datacenter GPUs use HBM stacked on a 2.5D CoWoS interposer:
- **H100 SXM5**: 80 GB HBM3, 3.35 TB/s
- **H200 SXM5**: 141 GB HBM3e, 4.8 TB/s
- **B200**: 192 GB HBM3e, 8.0 TB/s (2.4x H100)
- **GB200 NVL72**: 13.4 TB aggregate, ~576 TB/s aggregate

Blackwell's dual-die architecture uses a 10 TB/s NV-HBI die-to-die link, presenting a unified 192 GB address space to software.

### Host Interface / Package

- **PCIe Gen 5.0 x16**: ~128 GB/s bidirectional for standard form-factor GPUs.
- **NVLink-C2C**: 900 GB/s bidirectional coherent CPU-GPU link in Grace Hopper (GH200) and Grace Blackwell (GB200) Superchips. 7x faster than PCIe Gen 5.
- **Blackwell packaging**: TSMC CoWoS-L 2.5D with two ~800 mm^2 GB100 dies (TSMC 4NP), 208B transistors total.

### Scale-up Interconnect

- **NVLink 4th gen (Hopper)**: 900 GB/s per GPU, 18 links.
- **NVLink 5th gen (Blackwell)**: 1.8 TB/s per GPU (2x Hopper).
- **NVSwitch**: Non-blocking all-to-all NVLink crossbar ASIC. 5th-gen: 64 ports, 7.2 TB/s aggregate per switch, scalable to 576 GPUs.
- **GB200 NVL72**: 72 Blackwell GPUs + 36 Grace CPUs in a rack-scale liquid-cooled system; 130 TB/s total GPU-to-GPU bandwidth; all 72 GPUs in a single NVLink domain.

### Scale-out Interconnect

- **ConnectX-7**: NDR InfiniBand at 400 Gb/s per port (dual-port 800 Gb/s). Supports IB and Ethernet (RoCE v2) on the same ASIC. PCIe Gen 5.0 x16 host interface.
- **GPUDirect RDMA**: NIC DMAs directly to/from GPU HBM, eliminating CPU bounce buffers.
- **SHARP**: In-network AllReduce on InfiniBand switches, reducing GPU idle time during collective operations.
- **Standard topology**: NVLink for intra-node (scale-up) + NDR InfiniBand fat-tree for inter-node (scale-out).

---

## Programming Model Rationale

The NVIDIA GPU software stack's design is driven by fundamental hardware constraints:

**1. The warp is the true unit of execution, not the thread.** The SM executes 32 threads in lock-step (SIMT). CUDA exposes a scalar thread programming model as syntactic convenience, but performance-critical code must reason at warp granularity: memory coalescing requires consecutive threads to access consecutive addresses, Tensor Core instructions operate on warp-level or warp-group-level matrix tiles, and divergent branches serialize execution within a warp. This is why Triton's tile-based abstraction (where the compiler handles warp-level management) has proven so productive -- it lifts the warp-awareness burden from the programmer to the compiler.

**2. Memory bandwidth, not compute, is the dominant bottleneck.** Even on B200 (8 TB/s HBM3e, 20 PFLOPS FP4), the arithmetic intensity crossover for FP8 operations is ~2.5 FLOP/Byte. Most attention layers remain memory-bound; most GEMM layers are compute-bound. This fundamental tension drives the entire tiling/shared-memory strategy: CUTLASS, cuDNN, and FlashAttention exist to maximize data reuse through the on-chip memory hierarchy (registers -> shared memory -> L1 -> L2) and minimize HBM traffic. The 228 KB shared memory per SM on Hopper/Blackwell is explicitly sized to hold meaningful matrix tiles for GEMM.

**3. Tensor Cores require software to deliver data in specific tile layouts.** Each Tensor Core generation introduces new matrix tile shapes and data movement patterns (mma.sync warp-level, wgmma warp-group-level, tcgen05 with TMEM). The layered library stack exists to hide this complexity: cuBLAS provides a drop-in API where the user specifies matrices and precision; internally it selects from hundreds of pre-compiled CUTLASS kernel instantiations that match the hardware's tile requirements. CUTLASS exposes these tile abstractions for users who need custom kernels. Triton automates the mapping entirely.

**4. Asynchronous pipelining is essential for utilization.** Hopper's TMA + warp specialization pattern (producer warps doing DMA, consumer warps doing MMA) directly reflects the hardware's separate data movement and compute engines. CUTLASS's `PipelineTmaAsync` and Triton's `num_stages` parameter both express the same hardware concept: overlapping global-to-shared memory transfers with Tensor Core computation using mbarrier synchronization. Without this overlap, Tensor Cores idle while waiting for data.

**5. NVLink bandwidth determines parallelism strategy.** At 900 GB/s (Hopper) or 1.8 TB/s (Blackwell), NVLink enables tensor-parallel splits across GPUs within a node without AllReduce becoming the bottleneck. NCCL's topology-driven algorithm selection exists because the optimal collective algorithm depends entirely on the physical interconnect: Ring for bandwidth-bound large messages, Tree for latency-bound small messages, NVLS for NVSwitch-connected nodes with hardware multicast, and GIN/GDAKI to bypass CPU proxy threads on modern hardware.

**6. Forward compatibility via PTX enables ecosystem stability.** The PTX virtual ISA decouples compilation from hardware generations. Libraries compiled with PTX for compute_90 will JIT-compile to SASS on Blackwell (sm_100) or future architectures. This contract allows the massive CUDA ecosystem (thousands of libraries and applications) to run on new hardware without recompilation, a competitive moat that alternative platforms struggle to match.

**7. The GSP-split driver design minimizes host overhead.** Moving the Resource Manager to the on-GPU GSP processor means that steady-state GPU operation (kernel dispatch via user-mode doorbells, NVLink P2P transfers) requires no CPU kernel transitions. The open-source kernel module is primarily a thin RPC client for setup and teardown. This design choice reflects the asymmetry between host CPU frequency (~3 GHz) and GPU operation rates (millions of kernel launches per second in inference).

---

## Architecture Roadmap

### Current State (2026)

NVIDIA is in active production transition from Blackwell to Rubin. The **Vera Rubin platform** was announced at CES 2026 and has been in full production since May 2026; NVIDIA states that production shipments begin in fall 2026 — as of August 2026 no customer shipments have been announced. (NVIDIA newsroom, 2026-05-31: Vera Rubin "ramps into full production"; "Production shipments of Vera Rubin are set to begin starting this fall.")

### Vera Rubin Platform (2026)

The Vera Rubin platform is NVIDIA's most co-designed platform to date — seven chips (GPU, CPU, NVLink switch, SuperNIC, DPU, Ethernet switch, LPU) architected together from the ground up:

**Rubin GPU key specifications:**
- **Process**: TSMC 3nm (one node ahead of Blackwell's 4NP)
- **Memory**: HBM4, 288 GB per GPU, 22 TB/s bandwidth (2.8× Blackwell)
- **Compute**: 50 PFLOPS FP4 inference (2.5× Blackwell), 35 PFLOPS FP4 training
- **Interconnect**: NVLink 6 at 3.6 TB/s per GPU (2× NVLink 5)
- **Vera CPU**: 88 Olympus cores, connected via NVLink-C2C

**Vera Rubin NVL72 rack:**
- 72 Rubin GPUs + 36 Vera CPUs in a single NVLink domain
- 20.7 TB total HBM4 capacity; 1.6 PB/s total HBM bandwidth
- 260 TB/s scale-up bandwidth (2× GB200 NVL72)
- 3.6 EFLOPS FP4 inference per rack
- Up to 10× lower inference token cost vs Blackwell platform

**Rubin CPX**: A specialized inference variant optimized for massive-context workloads and long context windows.

**NVIDIA Groq 3 LPX rack (quantified 2026-09-13, Hot Chips 38):** 256 LPUs per rack, 128 GB SRAM, 40 PB/s aggregate SRAM bandwidth, 640 TB/s scale-up bandwidth per rack, 315 PFLOPS FP8 compute per rack, 11,000 tok/s decode on a 31B-parameter Gemma-4-class model per rack. Full production confirmed as of 2026-08-24; process node still not disclosed. See the Update section below for sources and caveats.

### Multi-Year Roadmap

| Generation | Year | Memory | FP4 Perf/GPU | Notable Change |
|---|---|---|---|---|
| Blackwell (B200) | 2024-H1 2025 | HBM3e 192 GB / 8 TB/s | 20 PFLOPS | Dual-die chiplet, MXFP4 |
| Blackwell Ultra (B300) | H2 2025 | HBM3e 288 GB | ~30 PFLOPS | 1.5× B200 compute |
| Rubin | H2 2026 | HBM4 288 GB / 22 TB/s | 50 PFLOPS | TSMC 3nm, NVLink 6 |
| Rubin Ultra | H2 2027 | HBM4E 1 TB | 100 PFLOPS | Quad-chiplet; NVL576 = **eight** MGX NVL racks × 72 Rubin Ultra GPUs (corrected 2026-08-08) |
| Feynman | 2028 | TBD | TBD | Named after Richard Feynman |

NVIDIA has maintained an annual release cadence for datacenter GPU generations since Hopper, and the roadmap above confirms continuation through at least 2028.

> **Correction (2026-08-08).** The earlier text "NVL576 (576 GPUs/rack)" was wrong. NVIDIA's own Vera Rubin POD blog (2026-03-16) defines Rubin Ultra NVL576 as combining "eight separate MGX NVL racks, each with 72 Rubin Ultra GPUs" into one NVLink domain via a two-layer all-to-all topology. The 144-GPU-per-rack step is **Kyber** ("will double the NVLink domain per rack to fit 144 GPUs"), which scales further to **NVL1152** (eight Kyber racks). The NVLink domain ladder is therefore NVL72 → NVL576 → Kyber NVL144 → NVL1152.

---

## Platform and Software-Stack Update (2026-08-08)

*Window covered: 2026-04-05 → 2026-08-08. Sources are listed per item; all figures below are NVIDIA-published unless explicitly marked as a vendor claim.*

**No new NVIDIA silicon generation and no change to any published Rubin specification in this window.** The Vera Rubin NVL72 spec page was re-checked line-by-line against the numbers recorded earlier in this document and matches exactly: per GPU, 50 PFLOPS NVFP4 inference / 35 PFLOPS NVFP4 training / 17.5 PFLOPS FP8-FP6 training, 288 GB HBM4 at 22 TB/s, NVLink 6 at 3.6 TB/s; per rack, 3,600 PFLOPS NVFP4 inference, 20.7 TB HBM4, 1,580 TB/s HBM bandwidth, 260 TB/s NVLink switch bandwidth, 3,168 Olympus CPU cores, 54 TB LPDDR5X. NVIDIA stamps that table **"Preliminary information. All values are up to and subject to change."**

### 1. Production status (corrected)

NVIDIA newsroom, dateline **2026-05-31**: Vera Rubin "ramps into full production," and "**Production shipments of Vera Rubin are set to begin starting this fall.**" As of 2026-08-08 no customer shipments have been announced. The same release quotes "10x agent throughput at scale compared with the previous-generation NVIDIA Grace Blackwell platform" — an NVIDIA marketing claim, not an independently confirmed measurement. Named adopters: CoreWeave, Firmus, GMI Cloud, IBM Cloud, IREN, Lambda, Microsoft Azure, Nebius, Nscale, SpaceXAI, Vultr. Fifteen named system builders: ASRock Rack, ASUS, Dell, Foxconn, GIGABYTE, HPE, IBM, Inventec, Lenovo, MSI, Pegatron, QCT, Supermicro, Wistron, Wiwynn. (Oracle Cloud Infrastructure does *not* appear in this release and has been removed from circulating adopter lists.)

### 2. Rubin Transformer Engine — adaptive compression

NVIDIA's Rubin platform page states that Vera Rubin NVL72 "features a **new Transformer Engine with adaptive compression** to boost NVFP4 inference performance." This is a 6th-generation Tensor Core / Transformer Engine capability beyond Blackwell's 5th-gen MXFP4/MXFP6 description. **No quantified figures are published**, and the page is undated — this is survey coverage that was previously missing, not a confirmed in-window change.

### 3. CUDA tile programming stack (coverage gap, pre-baseline)

Recorded in the Compiler / IR, Kernel Library and Assembler / ISA sections above. Timeline for the record: `NVIDIA/cutile-python` created **2025-06-13**; `NVIDIA/cuda-tile` created **2025-11-05** with v13.1.0 on **2026-01-14** and v13.2.0 on **2026-03-24** — all before this survey's 2026-04-05 baseline. In-window activity is incremental only: cutile-python v1.3.0 (2026-04-19), v1.4.0 (2026-05-26), v1.5.0 (2026-07-07); cuda-tile v13.3.0 (2026-05-28), v13.3.1/v13.3.2 (2026-06-27), v13.3.3 (2026-07-22). The **CUDA TILE-IR AS** toolkit component is likewise not new: 13.2.51 in CUDA 13.2 GA, 13.3.36 in 13.3 Update 1, 13.4.46 in the 13.4 preview.

### 4. CUDA Toolkit

- **Current GA is CUDA 13.3 Update 1** (this document previously implied 13.2).
- **CUDA 13.4 Developer Preview**, release notes dated **2026-07-16**. Listed contents: "New and updated PTX ISA, including some Rubin capabilities"; "CUDA Tile and TileIR compilation and inspection"; "CUDA Toolkit preview for RTX Spark and Rubin"; an RTX Spark Windows-on-Arm preview. Component versions: CUDA TILE-IR AS 13.4.46, cuBLAS 13.7.0.10, cuFFT 12.4.0.16. CUDA C++/Tile additions: numeric formats and mixed precision, MMA operations, tile views/layouts/sub-views, reductions and user-defined reductions, tile-level synchronization and communication. Two packaging changes with real downstream impact: **starting with CUDA 13.4 the NVIDIA Linux driver is no longer bundled with the CUDA Toolkit**, and 13.4 features require an **R616+** driver. NVIDIA marks the preview pre-release: performance data from it "must not be published, reported, compared, or used to characterize NVIDIA hardware" — it is cited here only for feature existence, never for numbers.
- **CUDA 13.2 GA libraries**: cuBLASLt experimental Grouped GEMM extended to MXFP8 inputs on compute capability 10.x and 11.0; FP64 fixed-point emulation added to `cublas[D|Z]syrk` / `syr2k` / `Zherk` / `Zher2k`; cuSOLVER fixed-point FP64 emulation APIs.
- Documentation pins refreshed: **PTX ISA v9.3** (was v9.2) and **cuBLAS 13.7.x** (was v13.2).

### 5. NCCL 2.30.x

| Release | Date | Notable content |
|---|---|---|
| v2.30.3-1 | 2026-04-15 | GIN contexts no longer shared between device communicators on the same host communicator; per-context GPU-/CTA-scoped GIN resource sharing; **elastic buffers** (multi-segment windows, active region in GPU memory, remainder in host memory); **TMA in select built-in symmetric kernels** (`NCCL_SYM_TMA_ENABLE=1`); experimental `gin.get` with nonblocking flush; Dynamic Direct Path (DDP) |
| v2.30.4-1 | 2026-04-22 | Elastic buffer support with GIN; nccl_param header fix |
| v2.30.7-1 | 2026-06-04 | **Hierarchical zero-SM collectives (AllGather, All2all)** via RMA CPU proxy inter-node + Copy Engines intra-node, enabled by `NCCL_CTA_POLICY_ZERO`; **experimental GPU Push Interface (GPI) backend for GIN**; explicit Strong/Weak GIN signal semantics; RMA plugin restructure; window registration during CUDA graph capture; **experimental MPS with MLOPart** (CUDA Memory Locality Optimized Partition, up to 2 ranks per physical GPU) |

Dates verified against git committer timestamps on the tagged commits, not just the release page.

### 6. MoE collectives become a first-class NVIDIA library

`nccl-ep-v0.1.0` (2026-06-08) and `nccl4py-v0.3.1` (2026-06-11) are **release tags inside the NVIDIA/nccl repository**, not standalone projects; details are recorded in the Communication section above. The v0.1.0 notes describe NCCL EP as built on the NCCL Device API (LSA + GIN) and make **no** Multi-Node-NVLink claim. Separately, **NVIDIA/nccl-extensions** was created 2026-07-07 and is actively developed.

### 7. NVIDIA Groq 3 LPX — the seventh chip

The seventh co-designed chip of the Vera Rubin platform is the **NVIDIA Groq 3 LPX**, an SRAM-based inference accelerator announced at GTC 2026 on 2026-03-16. NVIDIA's Rubin product page states verbatim: *"NVIDIA Groq 3 LPX is the inference accelerator for NVIDIA Vera Rubin, designed to meet the low-latency and large-context demands of agentic systems."* It frames the pairing as "combining Rubin GPUs for high-bandwidth memory (HBM) and LPUs for static random-access memory (SRAM)," and claims only "more tokens per watt and lower cost per token compared to the NVIDIA Blackwell architecture" — **unquantified**. The widely circulated "up to 35× higher tokens per watt" figure could not be traced to any NVIDIA primary source and is deliberately excluded.

**Naming (verified 2026-08-08 against NVIDIA's Rubin page).** "NVIDIA Groq 3 LPX" is the name NVIDIA itself uses for the accelerator. The strings **"Groq 3 LPU" and "LP30" do NOT appear** on that page; they circulate in secondary coverage as an alleged chip/die designation distinct from the LPX rack-scale system, but that distinction is **unconfirmed by any NVIDIA primary source** and should not be asserted. NVIDIA first described Vera Rubin as an extreme co-design across **six** chips (Rubin GPU, Vera CPU, NVLink 6 Switch, ConnectX-9 SuperNIC, BlueField-4 DPU, Spectrum-6 Ethernet Switch) and later as **seven**, the seventh being the LPX. Process node and ship date for it remain **not disclosed** by NVIDIA — the "Samsung 4nm" and "Q3 2026" attributions appear in no primary source.

**Corporate framing — corrected 2026-08-08.** This section previously said "following NVIDIA's acquisition of Groq." **NVIDIA did not acquire Groq.** On 2025-12-24 the two companies entered a **non-exclusive inference-technology licensing agreement**; Groq continues to operate independently and GroqCloud continues without interruption, while Jonathan Ross, Sunny Madra and others joined NVIDIA. NVIDIA explicitly told TechCrunch "this is not an acquisition of the company." The ~$20B figure is CNBC reporting, unconfirmed by either party. See `chips/groq/summary.md` for the full correction.

### 8. MLPerf Training v6.0 (results published 2026-06-16)

The round added two benchmarks: **DeepSeek-V3** (671B total / 37B activated per token, C4, 3.6 log-perplexity target) and **GPT-OSS 20B** (21B total / 3.6B activated); 24 organizations submitted. NVIDIA's own results post reports these headline times:

| Benchmark | Time to target | Scale |
|---|---|---|
| DeepSeek-V3 671B | 2.02 min | 8,192 GB300 (Blackwell Ultra) |
| Llama 3.1 405B | 7.07 min | 8,192 GB200 |
| GPT-OSS 20B | 7.43 min | 512 GB300 |
| Llama 3.1 8B | 4.46 min | 1,024 GB200 |

Software stack: **NeMo container 26.06**, full-iteration CUDA Graphs for token-dropless MoEs, Spectrum-X Ethernet Advanced Adaptive Routing. Per-submitter attribution (CoreWeave, Dell, Cisco, etc.) requires the MLCommons results table and is not asserted here.

### 9. Rack roadmap detail (NVIDIA Vera Rubin POD blog, 2026-03-16 — pre-baseline)

Beyond the NVL576 correction above: **Spectrum-6 SPX** at 102.4 Tb/s with 512 lanes and 200 Gb/s co-packaged optics; **eight ConnectX-9 SuperNICs per compute tray**; **one BlueField-4 DPU per compute tray**, which "combines the Vera CPU and ConnectX-9 SuperNIC."

### 10. Deliberately excluded

- The SemiAnalysis-via-CNBC report (2026-07-06) that the Kyber rack system slips to 2028: third-party analyst speculation with no NVIDIA statement and no retrievable primary source.
- Any change to Rubin Ultra (H2 2027) or Feynman (2028) timing: none confirmed.
- NVLink Fusion: no news in the window.
- **Hot Chips 38 (Aug 23–25, 2026)**: the program is public but no slides, abstracts, or specs exist — disclosure scheduled, content not yet public. Nothing from it is used here.

---

## Update (2026-09-13)

*Window covered: 2026-08-08 → 2026-09-13. Hot Chips 38 (2026-08-24/25) is the main event in this window. NVIDIA's own slide decks are attendee-gated — direct fetches of the Rubin GPU deck (`FINAL_NV_HC2026_Rubin.pdf`) and the LPU deck (`NV_HC2026_LP30_Final.pdf`) both returned HTTP 401 — so figures below come from NVIDIA's own Hot Chips event page and from ServeTheHome's on-site conference coverage (both dated 2026-08-24/25), not from the slides themselves.*

**1. Vera Rubin NVL72 — reconfirmed + new rack engineering detail.** NVLink 6 at 3.6 TB/s all-to-all bandwidth per GPU is reconfirmed. New: 130 TFLOPS of in-network compute vs. Ethernet scale-up ("10x lower latency," vendor claim); 45°C inlet liquid cooling; 800 VDC power distribution; a claimed 13% peak-power reduction for LLM training with "40% more GPUs per provisioned watt" (vendor claims). [ServeTheHome](https://www.servethehome.com/nvidia-vera-rubin-nvl72-rack-at-hot-chips-2026/).

**2. New scale: "100 MW AI Factory."** Factory-scale figures labeled "at-scale figures using DSX with MaxLPS," distinct from the per-rack numbers already recorded (3,600 PFLOPS/rack, 20.7 TB HBM4/rack): 2 ZFLOPS NVFP4 inference, 1.4 ZFLOPS NVFP4 training, 11 PB HBM4, 800 PB/s aggregate memory bandwidth. No rack count given, so not decomposed against the per-rack figures.

**3. NVIDIA Groq 3 LPX — quantified for the first time, and now in full production.** Hot Chips 38 gave this survey its first quantified LPX numbers (previously only naming/unquantified marketing framing): 256 LPUs/rack, 128 GB SRAM, 40 PB/s aggregate SRAM bandwidth, 640 TB/s scale-up bandwidth/rack, 315 PFLOPS FP8/rack, 11,000 tok/s decode on a 31B-parameter Gemma-4-class model/rack (NVIDIA event page; [ServeTheHome](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/)). Separately, [NVIDIA Newsroom (2026-08-24)](https://nvidianews.nvidia.com/news/nvidia-groq-3-lpx-now-in-full-production-with-world-class-speed-for-agentic-ai) confirms **"NVIDIA Groq 3 LPX...is now in full production"**, reports 3,400 tok/s on Gemma 4 31B with a 100,000-token context (a different, non-comparable benchmark configuration from the 11,000 tok/s rack-aggregate figure), reconfirms **"Groq and LPU are used under license from Groq, Inc."** (licensing, not acquisition — consistent with the existing correction below), and names **Nebius** as "the first AI cloud to adopt NVIDIA Groq 3 LPX." Process node remains not disclosed. The HC38 slide filename `NV_HC2026_LP30_Final.pdf` is a circumstantial (unconfirmed) signal for an "LP30" internal name; the deck itself is attendee-gated.

**4. Vera CPU — new marketing claim.** NVIDIA's Hot Chips event page states Vera CPU delivers "1.8x faster task completion and twice the efficiency of traditional x86 CPUs" — a vendor claim with no published methodology. Core count (88 Olympus) unchanged.

**5. Rubin GPU per-die specifics — investigated, not confirmed, excluded.** A secondary aggregator circulated per-die Rubin numbers (224 SMs / 896 tensor cores, 336B transistors, dual-die TSMC 3nm, 1,800–2,300 W/GPU, 190–230 kW/cabinet). Could not be verified: absent from ServeTheHome's HC38 coverage, the actual NVIDIA slide deck is attendee-gated (HTTP 401), and no NVIDIA primary source states them. Excluded per this survey's evidence discipline — SM count, tensor-core count, transistor count, and TDP remain not disclosed.

**6. Not included.** No confirmed change to Rubin Ultra (H2 2027) or Feynman (2028) timing; no new statement on the Kyber-to-2028 slip report; BlueField-4 and Spectrum-X Multiplane HC38 talk content is out of scope for this GPU-focused document.

---

## Resources

### Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Driver API Reference](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA v9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [cuBLAS Documentation (13.7.x)](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Toolkit Release Notes (GA — 13.3 Update 1)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.4 Developer Preview Release Notes (2026-07-16 — pre-release; feature existence only, no performance data)](https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/CUDA_Toolkit_Release_Notes.pdf)
- [CUDA 13.2 Archived Release Notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
- [cuDNN Developer Guide](https://developer.nvidia.com/cudnn)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

### Open-Source Repositories
- [CUTLASS](https://github.com/NVIDIA/cutlass) -- GEMM/Conv template library + CuTe DSL
- [NCCL](https://github.com/NVIDIA/nccl) -- Collective communications
- [CCCL](https://github.com/NVIDIA/cccl) -- CUB + Thrust + libcu++
- [Open GPU Kernel Modules](https://github.com/NVIDIA/open-gpu-kernel-modules) -- Linux kernel driver
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) -- cuDNN C++/Python wrapper
- [Triton](https://github.com/triton-lang/triton) -- Python DSL + MLIR compiler
- [cuda-tile](https://github.com/NVIDIA/cuda-tile) -- MLIR-based Tile IR and compiler infrastructure targeting Tensor Core units (created 2025-11-05)
- [cutile-python](https://github.com/NVIDIA/cutile-python) -- cuTile Python tile-kernel DSL (created 2025-06-13; v1.5.0 2026-07-07)
- [nccl-extensions](https://github.com/NVIDIA/nccl-extensions) -- Communication patterns for AI on NCCL device/host APIs (created 2026-07-07)

### Hardware
- [NVIDIA H100 Product Page](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 Product Page](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA GB200 NVL72 Product Page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA ConnectX-7 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband/connectx-7-datasheet.pdf)
- [NVIDIA Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/)
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [NVIDIA Rubin Platform Newsroom](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [Inside the NVIDIA Vera Rubin Platform — NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [Vera Rubin Ramps Into Full Production — NVIDIA Newsroom (2026-05-31)](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)
- [NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems — NVIDIA Technical Blog (2026-03-16)](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/)

### Benchmarks
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)
- [NVIDIA Blackwell Tops MLPerf Training v6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/)

### Added 2026-09-13
- [NVIDIA Groq 3 LPX Now in Full Production — NVIDIA Newsroom (2026-08-24)](https://nvidianews.nvidia.com/news/nvidia-groq-3-lpx-now-in-full-production-with-world-class-speed-for-agentic-ai)
- [NVIDIA Vera Rubin NVL72 Rack at Hot Chips 2026 — ServeTheHome (2026-08-24)](https://www.servethehome.com/nvidia-vera-rubin-nvl72-rack-at-hot-chips-2026/)
- [NVIDIA's Groq 3 LPU Accelerators for Heterogeneous AI Compute at Hot Chips 2026 — ServeTheHome (2026-08-25)](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/)
- [NVIDIA Hot Chips Conference Event Page](https://www.nvidia.com/en-us/events/hot-chips-conference/)
- [Hot Chips 2026 Program — hc2026.hotchips.org](https://hc2026.hotchips.org/program/)
