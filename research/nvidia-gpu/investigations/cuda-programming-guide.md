# CUDA Programming Guide — Investigation Report

*resource: https://docs.nvidia.com/cuda/cuda-c-programming-guide/*
*as_of: 2026-04-05*
*chip: nvidia-gpu*
*layer: Runtime, Compiler/IR, Framework Integration*

---

## Overview

The CUDA (Compute Unified Device Architecture) Programming Guide is NVIDIA's canonical reference for the CUDA parallel computing platform and programming model. First introduced in 2007 with the Tesla architecture, CUDA is the foundational programming abstraction through which virtually all NVIDIA GPU compute software is written — directly or indirectly. Frameworks such as PyTorch, JAX, and TensorFlow all ultimately express GPU work through CUDA kernels, streams, and memory operations.

CUDA exposes the GPU as a massively parallel co-processor. A developer writes *kernels* — functions executed by thousands of GPU threads simultaneously — using a C/C++ dialect extended with execution configuration syntax (`<<<grid, block>>>`). The CUDA model deliberately hides low-level GPU microarchitectural details (e.g., exact warp scheduling policy, cache replacement) while exposing the structural hierarchy that matters for performance: thread organization, memory spaces, and asynchronous execution.

The guide covers two related APIs:
- **CUDA Runtime API** (`cudaXxx` functions) — the primary high-level interface used by most CUDA programs.
- **CUDA Driver API** (`cuXxx` functions) — a lower-level C API that gives finer control over context management, module loading, and virtual memory.

---

## Architecture

### Thread Hierarchy

CUDA organizes threads in a three-level hierarchy (four levels from compute capability 9.0):

| Level | Unit | Hardware mapping |
|---|---|---|
| Grid | Collection of blocks launched by one kernel call | Entire GPU |
| Thread Block Cluster (cc 9.0+) | Optional group of blocks guaranteed to be co-scheduled | Group of SMs sharing L2 slice |
| Thread Block | Up to 1024 threads sharing shared memory and synchronization | Single SM |
| Warp | 32 consecutive threads within a block | Hardware scheduling unit; SIMT execution |
| Thread | Individual scalar execution lane | One CUDA core lane |

Thread IDs within a block are expressed as 1D, 2D, or 3D indices (`threadIdx.{x,y,z}`), and blocks within a grid similarly (`blockIdx.{x,y,z}`). This maps naturally onto tensor layouts.

The **warp** (32 threads) is the fundamental hardware scheduling atom. The SM creates, manages, and schedules threads in warp granularity. All 32 threads in a warp execute the same instruction each clock cycle under the SIMT model. Branch divergence is handled by predication: divergent threads are masked off while active threads execute, then roles are reversed — serializing execution and reducing throughput proportionally to divergence depth.

**Thread Block Clusters** (Hopper/cc 9.0+) form a new hierarchy level. Blocks within a cluster are guaranteed to be co-scheduled on proximate SMs, allowing inter-block data sharing via distributed shared memory accessible over the SM-to-SM interconnect (NVLink fabric within a GPC), and synchronization via cluster-level barriers.

### Memory Model

CUDA exposes a multi-level memory hierarchy, each level with distinct scope, lifetime, bandwidth, and latency:

| Memory Space | Scope | Location | Cached? | Latency (approx.) |
|---|---|---|---|---|
| Register | Per-thread | On-chip register file | N/A | ~1 cycle |
| Shared Memory | Per-block | On-chip (unified data cache, partitioned with L1) | N/A | ~5–10 cycles |
| L1 Cache | Per-SM | On-chip (shared with shared memory) | Hardware-managed | ~5–10 cycles |
| L2 Cache | GPU-wide | On-chip | Hardware-managed | ~30–80 cycles |
| Global Memory | GPU-wide | HBM/GDDR off-chip | via L1/L2 | ~300–700 cycles |
| Constant Memory | GPU-wide (read-only) | HBM, cached in dedicated constant cache | Yes | ~1–5 cycles if cached |
| Texture Memory | GPU-wide (read-only) | HBM, cached with spatial locality optimization | Yes | Variable |
| Local Memory | Per-thread (register spills) | HBM (treated as global) | via L1/L2 | ~300–700 cycles |

The unified data cache on each SM partitions its capacity between L1 cache and shared memory; the split ratio is configurable at runtime via `cudaFuncSetAttribute()` or at compile time with `__launch_bounds__`.

**Unified Memory** (`cudaMallocManaged`) provides a single address space visible to both CPU and GPU. The CUDA runtime and hardware Page Migration Engine transparently migrate 4KB pages on demand when a GPU thread faults on a non-resident address. Pascal (cc 6.x) introduced hardware page-fault support enabling oversubscription beyond GPU DRAM capacity. `cudaMemPrefetchAsync()` and `cudaMemAdvise()` allow the developer to hint prefetch behavior and access patterns to reduce fault stalls.

Shared memory is divided into 32 banks (one per 32-bit word width). Simultaneous accesses to the same bank by multiple threads in a warp are serialized (bank conflict), degrading throughput. Stride-1 access patterns across consecutive threads avoid conflicts.

### Execution Model

The **SIMT (Single Instruction, Multiple Threads)** model is the core execution paradigm. Unlike pure SIMD, SIMT allows each thread to maintain independent register state and a separate program counter, so threads can diverge through conditional branches. The hardware reconverges divergent threads at the earliest post-dominator (reconvergence point).

**Occupancy** is the ratio of active warps on an SM to the SM's maximum warp capacity. Active warps hide memory latency through interleaving: while one warp stalls awaiting a memory response, the warp scheduler selects another ready warp at zero cost. Achieving sufficient occupancy (typically 50–75%) is more important than chasing 100%, because register pressure from high occupancy reduces per-thread register availability, causing register spills to local (global) memory.

Key SM resource limits that constrain occupancy:
- Maximum warps per SM (48–64 depending on compute capability)
- Maximum thread blocks per SM (16–32)
- Total registers per SM (e.g., 65,536 for A100, H100)
- Shared memory per SM (e.g., 228 KB for H100 per SM, configurable split with L1)

**CUDA Graphs** capture a sequence of GPU operations (kernels, memcopies, memsets) as a directed acyclic graph. Once captured, the graph can be replayed with minimal CPU overhead, eliminating per-launch driver interaction — critical for inference workloads with microsecond-latency budgets.

---

## Data Flow

A CUDA kernel launch follows this pipeline from host code to SM execution:

```
Developer Code (CUDA C/C++)
        |
        | nvcc compilation
        v
PTX (virtual ISA) ─────── cubin (SASS for target arch)
        |                        |
        └──── fat binary (.cubin/.ptx bundle) ────┐
                                                   |
Host Code (stub functions replace <<<>>>)          |
        |                                          |
        | cudaLaunchKernel() or <<<grid,block>>>   |
        v                                          |
CUDA Runtime API (libcudart.so)                   |
        |                                          |
        v                                          |
CUDA Driver API (libcuda.so)  <────────────────────┘
        |  (cuLaunchKernel, cuModuleLoad)
        v
NVIDIA Kernel Module (nvidia.ko) [privilege boundary]
        |
        | PCIe / NVLink command submission
        v
GigaThread Engine (GPU front-end)
        |  (global work distributor)
        v
Work Distributor → Thread Block Scheduler
        |  (assigns blocks to SMs based on resource availability)
        v
SM: Warp Partitioner splits block into warps of 32 threads
        |
        v
Warp Scheduler (2–4 per SM depending on arch)
        |  (selects ready warp each cycle; hides latency by interleaving)
        v
Instruction Dispatch → CUDA Cores / Tensor Cores / SFUs / LSUs
        |
        v
Memory System: Register File → Shared Mem → L1 → L2 → HBM
```

Key points in this flow:
1. **JIT compilation**: If only PTX is embedded in the fat binary, the driver JIT-compiles PTX to SASS for the runtime GPU architecture. The resulting cubin is cached on disk to avoid repeat compilation.
2. **Asynchronous launch**: Kernel launches return immediately to the host. The GPU executes asynchronously in a *stream*; `cudaStreamSynchronize()` or `cudaDeviceSynchronize()` blocks the CPU until completion.
3. **Multiple streams**: Kernels launched in different streams may execute concurrently on the GPU (subject to resource availability), enabling overlap of compute and data transfers.
4. **Command queuing**: The driver packages the launch descriptor and queues it into the GPU's pushbuffer. The GigaThread engine dequeues and dispatches blocks to available SMs.

---

## Hardware Interface

CUDA abstractions map directly to SM hardware in well-defined ways:

| CUDA Abstraction | Hardware Realization |
|---|---|
| Thread | One lane in a warp; one register file slot |
| Warp (32 threads) | Hardware scheduling unit; one instruction issue per warp scheduler per clock |
| Thread Block | Occupies one SM; allocates from SM's shared memory and register file partition |
| Thread Block Cluster | Group of blocks on adjacent SMs within a GPC; share L2 slice, use SM-SM interconnect for distributed shared mem |
| Grid | Entire GPU; GigaThread engine dispatches blocks to all SMs |
| Stream | GPU hardware queue / channel |
| CUDA Context | GPU virtual address space; MMU context |
| Shared Memory | On-chip SRAM in unified data cache; split with L1 |
| Global Memory | HBM2e/HBM3 DRAM, accessed through L1/L2 cache hierarchy |
| Tensor Core operation | Single warp-level matrix instruction (WMMA / MMA) targeting dedicated Tensor Core units per SM |

**SM Warp Schedulers**: Modern SMs have 4 warp schedulers (H100), each capable of issuing one instruction per clock to its assigned warp pool. With 64 warps resident per SM and 4 schedulers, each warp is eligible for issue every ~16 cycles, which matches typical L1 hit latency.

**Compute Capability** is the versioning mechanism that encodes hardware feature availability. Examples:
- cc 8.0 (A100): 108 SMs, 64 warps/SM, TF32/BF16 Tensor Cores, NVLink 3
- cc 9.0 (H100): 132 SMs, Thread Block Clusters, TMA unit, warpgroup-level MMA, NVLink 4
- cc 10.0 (Blackwell B100): 4th-gen Tensor Cores, FP4 support, NVSwitch 5th-gen

**Tensor Memory Accelerator (TMA)** on Hopper offloads address generation for multi-dimensional tensor loads/stores from software to dedicated hardware. A single thread invokes TMA to asynchronously fetch a tile into shared memory, freeing all other threads in the warp for computation. TMA supports up to 5D tensor descriptors, handling strides and boundary conditions in hardware.

---

## Key Findings

1. **The warp is the true unit of parallelism.** The thread abstraction is a programmer convenience; the GPU schedules, executes, and measures efficiency at warp granularity. Thinking in warps — not threads — is required to reason about memory coalescing, divergence penalties, and occupancy.

2. **Memory bandwidth is the dominant bottleneck.** The gap between compute throughput (e.g., A100: 312 TFLOPS BF16) and memory bandwidth (A100: 2 TB/s HBM2e) means that nearly all production AI kernels are memory-bandwidth-bound. CUDA's memory hierarchy (registers → shared memory → L1/L2 → HBM) exists specifically to maximize data reuse and minimize HBM accesses. This is the core motivation for tiling algorithms in CUTLASS and FlashAttention.

3. **Occupancy is a means, not an end.** The CUDA guide explicitly warns that maximizing occupancy does not always maximize throughput. High occupancy competes with per-thread register allocation; reducing registers to fit more warps can trigger register spills to local (global) memory, dramatically worsening performance. The optimal occupancy is kernel-specific.

4. **Streams enable asynchronous heterogeneous pipelines.** CUDA streams are the fundamental mechanism for overlapping CPU work, H2D/D2H transfers, and multiple kernel executions. High-performance inference stacks (TensorRT, vLLM) exploit multi-stream scheduling aggressively to saturate PCIe and keep all SMs busy.

5. **PTX provides forward binary compatibility.** Embedding PTX alongside cubin in fat binaries allows JIT compilation on future GPUs not known at compile time. This is how libraries like cuDNN and CUTLASS can ship a single binary that runs on GPU generations released after their compile date.

6. **Thread Block Clusters formalize inter-SM cooperation.** Before Hopper, inter-block synchronization required a full kernel relaunch or global atomic operations. Clusters provide a hardware-guaranteed cooperative group that can synchronize mid-kernel, enabling algorithmic patterns (e.g., multi-stage matrix pipelines across SMs) that were previously impossible without software workarounds.

7. **The CUDA programming model is intentionally leaky.** Unlike OpenCL's abstraction goals, CUDA deliberately exposes hardware concepts (warp size = 32, shared memory banks = 32, register file size per SM) because hiding them would preclude the performance tuning required for high-throughput AI workloads.

---

## Relation to Hardware Architecture

CUDA's design is inseparable from the SM microarchitecture it exposes:

**Thread hierarchy ← SM resource partitioning**: The 3-level (thread/block/grid) hierarchy maps directly to how SM hardware partitions registers and shared memory. Register file size and shared memory capacity per SM set hard ceilings on occupancy; the programmer must fit thread block resource consumption within these limits.

**SIMT ← SIMD execution units with independent program counters**: Each SM has wide SIMD execution units (e.g., 128 FP32 CUDA cores on A100 SM). SIMT layers 32-thread warp scheduling over this SIMD array, adding per-thread register state and branch predication. This lets CUDA present a scalar thread model while executing vectorially.

**Shared memory ← On-chip SRAM scratchpad**: Shared memory is programmer-managed L1 cache (unlike the hardware-managed L1 alongside it). The partition between L1 and shared memory is configurable because different kernel patterns (cache-friendly irregular access vs. hand-tiled regular access) benefit differently.

**Streams and concurrency ← Multiple GPU hardware queues + GigaThread dispatch engine**: The GigaThread engine processes work from multiple hardware queues (channels) and dispatches thread blocks to idle SMs. CUDA streams map to these hardware channels; the hardware-level dispatch gives streams their concurrency semantics.

**Tensor Cores ← Specialized matrix execution units per SM**: The `wmma::` and MMA PTX instructions in CUDA map directly to the Tensor Core units in each SM (4 per SM warp scheduler partition on A100). The warpgroup MMA abstraction on Hopper extends this to 4-warp cooperative operations matching the TMA tile size, tightly coupling the memory subsystem to the compute subsystem.

**Compute capability versioning ← Incremental SM feature additions**: Each new GPU architecture adds SM capabilities (new data types, new memory instructions, new synchronization primitives) that require new CUDA API surface. Compute capability numbers version this expansion in a way that preserves backward compatibility while exposing new hardware.

---

## Sources

- [CUDA C++ Programming Guide (Legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — canonical reference
- [1.2 Programming Model — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [2.1 Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [2.2 Writing CUDA SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [2.4 Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [3.2 Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [4.1 Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [4.4 Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [4.11 Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [5.1 Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
