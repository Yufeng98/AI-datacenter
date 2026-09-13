# HIP Runtime and ROCm Ecosystem — Investigation Report

**Resource:** ROCm/HIP  
**URL:** https://github.com/ROCm/HIP  
**Commit SHA:** 7ddf4f9fea867d9241d2465c6baf93eceb8694d8  
**HIP Version:** 7.2.0  
**As of:** 2026-04-05  
**Chip:** amd-gpu  
**Device class:** GPU

---

## Overview

HIP (Heterogeneous-compute Interface for Portability) is AMD's primary C++ runtime API and kernel language for GPU programming. It provides a CUDA-like interface that allows developers to write portable GPU-accelerated code that targets both AMD and NVIDIA hardware with minimal source changes. On AMD hardware, HIP compiles through `hipcc` (a thin compiler driver wrapper) to `amdclang++`, which emits LLVM AMDGPU IR before final lowering to GFX ISA machine code specific to the target GPU. The runtime library (`libamdhip64.so`) sits above the Compute Language Runtime (CLR / `rocclr`), which in turn calls into the ROCr (HSA Runtime, `libhsa-runtime64.so`) for queue management, memory allocation, and kernel dispatch using the Architected Queuing Language (AQL) packet protocol. The kernel-mode interface is the KFD (Kernel Fusion Driver), a Linux driver that manages GPU contexts, memory mappings, and hardware command queues.

---

## Architecture

### Module Structure

```
HIP public headers (include/hip/)
  ├── hip_runtime.h          — convenience umbrella include
  ├── hip_runtime_api.h      — C-style runtime API (hipMalloc, hipMemcpy, hipLaunchKernelGGL, …)
  ├── hip_ext.h              — AMD extensions (hipExtLaunchKernel, cooperative launch)
  ├── hip_cooperative_groups.h — cooperative groups API
  ├── hip_fp16.h / hip_bf16.h / hip_fp8.h — low-precision types
  ├── hiprtc.h               — runtime compilation API (HIPRTC)
  └── driver_types.h / linker_types.h — type definitions

Runtime implementation (in CLR/hipamd — separate repo: ROCm/clr)
  ├── hipamd/    — AMD-platform HIP implementation (streams, events, memory, kernel launch)
  └── rocclr/    — ROCm Compute Language Runtime: virtual device interface

Lower layers
  ├── ROCr / HSA Runtime (libhsa-runtime64.so) — AQL queue dispatch, agent enumeration
  ├── KFD (amdgpu kernel driver) — kernel-mode GPU context management
  └── GFX ISA — architecture-specific microcode executed on CUs
```

### Key Abstractions

| Abstraction | Layer | Description |
|---|---|---|
| `hipStream_t` | HIP Runtime | FIFO command queue; maps to an HSA/AQL queue. All async ops (kernels, memcpy) enqueue commands here. |
| `hipEvent_t` | HIP Runtime | Timestamp/synchronization marker inserted into a stream; resolved via HSA signal objects. |
| `hipGraph_t` / `hipGraphExec_t` | HIP Runtime | DAG of operations (kernels, memcpy, host callbacks) compiled once and launched repeatedly with a single call, amortizing driver overhead. |
| AQL Packet (Architected Queuing Language) | ROCr / HSA | 64-byte hardware packet format specifying kernel dispatch parameters (grid dims, group dims, kernel object pointer, kernarg pointer). Written directly into hardware-visible ring buffers. |
| Wavefront (Warp) | Hardware | Fundamental SIMD execution unit: 64 threads on CDNA, 32 on RDNA. All threads execute the same instruction in lockstep across SIMD lanes. |
| Compute Unit (CU) | Hardware | Contains: sequencer (SQ), 4× SIMD-16 pipelines (CDNA), VGPR/SGPR register files, MFMA matrix units, LDS, vL1D cache, SALU, VMEM, SFU, branch unit. |
| LDS (Local Data Share) | Hardware / HIP `__shared__` | Per-CU on-chip SRAM (32–64 banks, 4 bytes/bank); programmer-visible as `__shared__` arrays; enables intra-workgroup communication at ~100× the bandwidth of HBM. |
| MFMA Unit | Hardware | Matrix fused multiply-add accelerator on CDNA. Accepts VGPR/AGPR tiles, performs `D = A×B + C` over entire matrix fragments per instruction (e.g., `v_mfma_f32_16x16x4f16`). |
| GFX IP Version | Compiler/ISA | Virtual hardware target string (e.g., `gfx942` for MI300X CDNA3). Decouples HIP source from physical GPU, controls ISA features emitted by `amdclang++`. |

### Dependency Graph (Simplified)

```
[Python / C++ Application]
        |
[HIP Runtime API]  ←  libamdhip64.so
        |
[CLR / rocclr]     ←  virtual device interface
        |
[ROCr / HSA Runtime]  ←  libhsa-runtime64.so
        |
[KFD (amdgpu kernel driver)]  ←  Linux kernel module
        |
[GPU Hardware: CPs, ACEs, SEs, CUs, HBM]
```

---

## Data Flow: HIP Kernel Launch → Hardware Execution

### Step-by-step trace

1. **Application** calls `hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args...)` or the triple-chevron `kernel<<<grid, block, sharedMem, stream>>>(args...)`.

2. **HIP Runtime (`libamdhip64.so`)** translates the launch into an internal representation. It resolves the kernel object (a compiled `.co` code object embedded in the host binary) and constructs a 64-byte **AQL Dispatch Packet** containing:
   - `grid_size_x/y/z`, `workgroup_size_x/y/z`
   - `kernel_object` (pointer to GPU-resident code object)
   - `kernarg_address` (pointer to packed kernel arguments in device memory)
   - `group_segment_size` (LDS bytes), `private_segment_size` (scratch bytes)
   - `completion_signal` (HSA signal to decrement on completion)

3. **ROCr / HSA Runtime** writes the AQL packet into the **hardware AQL queue** (a ring buffer in GPU-visible memory). It writes a doorbell register to notify the GPU's Command Processor Fetcher (CPF).

4. **GPU Command Processor (CP)**:
   - **CPF** fetches the AQL packet from the ring buffer
   - **CPC** (microcontroller) decodes it and forwards the dispatch to an **Asynchronous Compute Engine (ACE)**

5. **ACE** breaks the dispatch into workgroups and feeds them to the **Shader Processor Input (SPI)** on each Shader Engine.

6. **SPI** schedules workgroups onto available **Compute Units**, initializing VGPRs with kernel arguments and wavefront slots.

7. **Compute Unit** executes:
   - Wavefronts (64 threads for CDNA) are scheduled by the **Sequencer (SQ)**
   - The issue arbiter dispatches up to 5 instructions/cycle across VALU, SALU, VMEM, LDS, and branch units
   - Memory loads/stores flow: vL1D → L2 (shared across all CUs) → Infinity Fabric → HBM
   - MFMA instructions dispatch to Matrix Core units independently from VALU

8. **Completion**: When all workgroups finish, the hardware decrements the **HSA completion signal**. ROCr notifies the HIP runtime, which unblocks any `hipStreamSynchronize` or `hipEventSynchronize` calls on the host.

---

## Hardware Interface

### Wavefronts and SIMD Mapping

- CDNA architectures: wavefront = 64 threads → executed across 4× SIMD-16 pipelines (4 cycles per instruction, 16 threads/cycle/SIMD)
- RDNA architectures: wavefront = 32 threads → executed in 1 cycle across 32-wide SIMD
- Each CU supports up to 40 resident wavefronts (10 slots × 4 pools); actual occupancy limited by VGPR/SGPR/LDS pressure

### Memory Hierarchy

| Level | Size | Scope | HIP keyword |
|---|---|---|---|
| VGPR register file | 256–512 KiB/CU | Per-thread | automatic variables |
| LDS (Local Data Share) | 64 KiB/CU (CDNA) | Per-workgroup | `__shared__` |
| vL1D cache | 16 KiB/CU | Per-CU | transparent |
| L2 cache | varies (32 channels, 256B interleave) | Per-GPU | transparent |
| HBM | GB scale | Per-device | `hipMalloc` |

### AQL Queue Protocol

The HIP stream maps directly to an HSA **AQL queue**. The ring buffer lives in GPU-accessible (pinned) host memory or GPU VRAM. The CPU writes packets and rings a doorbell MMIO register; the GPU CP polls and fetches. This zero-copy dispatch mechanism minimizes CPU–GPU round-trip latency.

### KFD Driver Interface

`libhsa-runtime64.so` calls `hsaKmtOpenKFD()` to establish a session with the KFD kernel driver. KFD manages:
- GPU process address space (GPUVM)
- Queue creation and doorbell mapping
- Memory pool allocation (`hsa_amd_memory_pool_allocate`)
- Signal objects (used for event completion)
- XNACK (page-fault) handling for Heterogeneous Memory Management (HMM)

---

## Key Findings

1. **Three-layer runtime stack**: HIP API → CLR/rocclr (virtual device) → ROCr/HSA (AQL dispatch) → KFD (kernel driver). Each layer has a distinct responsibility; `rocclr` provides platform abstraction so HIP can also back an OpenCL runtime without code duplication.

2. **AQL is the true hardware interface**: The Architected Queuing Language packet is written directly into GPU-visible ring buffers. There is no kernel-mode driver call per kernel launch at steady state — the hot path is a userspace ring-buffer write plus a doorbell MMIO write, matching CUDA's cuLaunchKernel model.

3. **Compilation pipeline**: `hipcc` (wrapper) → `amdclang++` (offload compilation, LLVM) → AMDGPU IR (virtual ISA, target-agnostic within a family) → GFX ISA (architecture-specific binary embedded in host ELF as `.co` code object). The `--offload-arch=gfxXXX` flag selects the final ISA target; HIPRTC can perform this compilation at runtime.

4. **CDNA wavefront = 64 threads**: Unlike NVIDIA CUDA (warp = 32), CDNA's wavefront is 64 lanes. RDNA uses 32. Code must account for this when reasoning about occupancy, register file sizing (256 VGPRs × 64 lanes per wavefront slot), and LDS bank conflicts.

5. **MFMA units are first-class**: On CDNA (MI100+), `v_mfma_*` instructions run independently from the main VALU pipelines and use AGPRs (Accumulation VGPRs, up to 256 KiB extra per CU) as accumulators. This is the hardware primitive underlying rocBLAS GEMM and hipBLASLt.

6. **HIP graphs reduce launch overhead**: For AI inference workloads executing thousands of short kernels per forward pass, `hipGraph_t` pre-records the DAG and replays it with a single `hipGraphLaunch`, eliminating per-kernel driver overhead that dominates when GPU kernel time < framework dispatch time.

7. **Portability via CUDA compatibility**: The HIP API is intentionally isomorphic to CUDA Runtime API. `hipify` tools mechanically translate CUDA source. On NVIDIA targets, HIP calls map directly to CUDA calls with near-zero overhead.

8. **Software-managed coherence**: GPU L1 caches are write-through; coherence between CUs requires explicit `__threadfence` / `s_barrier` / `s_mem_realtime` instructions. The L2 cache is the coherence point for all atomics and cross-CU accesses.

---

## Relation to Hardware Architecture

| HIP / ROCm Concept | Hardware Realization |
|---|---|
| `hipStream_t` | AQL hardware queue ring buffer + doorbell MMIO register |
| `hipEvent_t` | HSA signal object; resolved by hardware signal decrement on CU completion |
| Grid (kernel launch) | Entire GPU dispatch: ACEs break into workgroups, SPI schedules onto CUs |
| Thread block / work-group | Entire workgroup always resides on a single CU; shares LDS |
| Warp / wavefront (CDNA) | 64-thread SIMD execution unit across 4× SIMD-16 pipelines |
| `__shared__` memory | LDS: 64 KiB/CU, 32 banks, ~100× HBM bandwidth |
| `__global__` memory | HBM (CDNA) / GDDR (RDNA consumer) accessed via vL1D → L2 → Infinity Fabric |
| MFMA intrinsics | MFMA hardware matrix core units; AGPR accumulator register file |
| `hipMalloc` | `hsa_amd_memory_pool_allocate` → KFD GPUVM page tables |
| `hipMemcpy` | DMA engines (two per GPU); bidirectional PCIe/XGMI transfers without CPU intervention |
| HIPRTC JIT | `amdclang++` invoked at runtime; produces `.co` code object for any installed GFX target |

---

## Software-Stack Update: ROCm 7.14.0, Primus, ROCm.AI / Hyperloom

*Added: 2026-08-08. Scan 2026-08-08, adversarially verified. This chip has no separate `software-stack.md`; the ROCm/HIP investigation is the software-stack record, so the 2026 software update is appended here.*

### ROCm 7.14.0 — released 2026-07-16

Release tag `rocm-7.14.0` on GitHub and the ROCm compatibility matrix both carry **2026-07-16**; AMD's announcement blog is dated 2026-07-15.

**Headline: TheRock goes production.** The release "transitions ROCm to TheRock, a build and release system that introduces a modular architecture" — i.e. the ROCm build/packaging pipeline itself is the flagship change, not a runtime feature.

Confirmed additions by layer:

| Layer | Addition |
|---|---|
| Runtime | **HIP Execution Context APIs** for GPU compute-resource partitioning |
| Runtime | Batch memory APIs — `hipMemDiscardBatchAsync`, `hipMemPrefetchBatchAsync` |
| Runtime | Faster HIP graph replay for async allocations |
| Communication (RCCL) | **Hierarchical AllGather** separating inter-node from intra-node communication; direct reduce-scatter |
| Profiling | **ROCprofiler-SDK beta Streaming Performance Monitors**; PyTorch Profiler integration |
| Op Library | Per-matrix bias in hipBLASLt batched GEMM |
| I/O | hipFile direct storage I/O |
| Assembler / ISA | Adds Ryzen AI APU targets **gfx1151 / gfx1153** |
| Virtualization | Multi-VF partition modes for MI355X / MI350X |

Framework support in this release: **PyTorch 2.12.0, JAX 0.10.0, vLLM 0.23.0, TensorFlow 2.21**.

**Versioning discontinuity worth recording**: the GitHub release list shows ROCm jumping from 7.2.x to a 7.9.0 preview and then 7.10–7.14. **"ROCm 8" does not exist as of 2026-08-08.** The 7.2.0 HIP version recorded at the top of this report is the 2026-04-05 baseline.

### CRITICAL GAP — no public gfx target for CDNA 5

The ROCm 7.14.0 compatibility matrix's newest **Instinct** targets are **gfx950** (MI355X / MI350X / MI350P), **gfx942**, **gfx90a**, **gfx908**. There is **no MI455X / CDNA 5 / gfx96x-class entry**, and neither the CDNA 5 architecture blog nor the ROCm 7.14 blog names a gfx target for CDNA 5.

**The CDNA 5 gfx ISA identifier is not disclosed.** Consequences for the stack described in this report:

- No published `--offload-arch=` value for CDNA 5, so the hipcc / amdclang++ / HIPRTC pipeline documented above has no public CDNA 5 path.
- AITER ships `hsa/gfx942/` and `hsa/gfx950/` HSACO blob directories; there is no CDNA 5 equivalent in the open tree.
- CK-Tile and AITER per-architecture tuning CSVs have no CDNA 5 entries.
- This is a genuine and reportable state: **hardware is in production ahead of public ISA-target disclosure.**

### AMD Primus — new framework-integration component

**Primus / Primus-LM** (`github.com/AMD-AIG-AIMA/Primus`) is AMD's open training framework for large-scale foundation-model pretraining, post-training (SFT / LoRA) and RL on AMD GPUs. Structure:

- Backends: **Megatron-LM, TorchTitan, JAX MaxText**
- A unified cluster CLI over those backends
- Latest release **v26.4**; actively developed through 2026-07-29
- Published MLPerf Training v6.0 examples on MI355X
- A companion "Primus Tuning Agent" ROCm blog is dated 2026-07-06

AMD's own MLPerf blog states this was "the first-ever use of AMD's Primus training framework in MLPerf Training submissions". Primus was **absent from this repo's framework-integration layer** before 2026-08-08 and is a real gap now closed.

### ROCm.AI and Hyperloom — announced 2026-07-23

Both were presented at Advancing AI 2026.

- **ROCm.AI**: described as AI-assisted kernel generation — a layer letting coding agents (Claude, Codex, Cursor) work against AMD hardware natively.
- **Hyperloom**: agentic loops that profile a workload, identify bottlenecks, and tune or rewrite kernels. More than a slide — AMD published "Hyperloom — Autonomous Agentic Inference Optimization for AMD GPUs" on the ROCm blog dated 2026-07-23.

**Evidence status: medium confidence on existence, low on the numbers.** The performance figures are **vendor marketing claims and are not independently verified**: 3.3x inference over ROCm 7, and a demo showing Hyperloom improving token rate by 38%. A "2.4x training" figure circulated but is not confirmed. **No public release or version number was found.**

### Benchmarks

**MLPerf Training v6.0** (results public 2026-06-16) — AMD's **first multi-node MLPerf Training submission**:

- 512x MI300X (64 nodes) with **Oracle Cloud Infrastructure**
- 8-node / 64x MI325X Flux.1-schnell FP8 data-parallel submission
- MI350X and MI355X single-node LLM submissions
- First production-ready **MXFP4 training recipe** (~2x the compute density of FP8)
- Gains vs MLPerf 5.1: Llama2-70B LoRA **+19% (MI355X) / +16% (MI350X)**; Llama3.1-8B pretraining **+13% (MI355X) / +11% (MI350X)**
- OCI is the confirmed partner; a broader partner list (Dell, HPE, Asus, Cisco, Supermicro, MiTAC, KRAI, Vultr) was **not confirmed**

**MLPerf Inference v6.0** (2026-04-01, at this repo's prior baseline) — AMD published MI355X single-node and multi-node results including its first >1M tok/s aggregate figure. **None of the specific throughput numbers or the NVIDIA B300 comparison percentages could be independently verified in this pass; treat them as unverified AMD-reported comparisons.**

### Sources (2026-08-08 software update)

- [ROCm 7.14 release blog — AMD ROCm Blogs](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html)
- [ROCm/ROCm release rocm-7.14.0 — GitHub](https://github.com/ROCm/ROCm/releases/tag/rocm-7.14.0)
- [ROCm release list — GitHub](https://github.com/ROCm/ROCm/releases)
- [ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [AMD-AIG-AIMA/Primus — GitHub](https://github.com/AMD-AIG-AIMA/Primus)
- [Primus Tuning Agent — AMD ROCm Blogs (2026-07-06)](https://rocm.blogs.amd.com/software-tools-optimization/primus-tuning-agent/README.html)
- [Hyperloom — Autonomous Agentic Inference Optimization for AMD GPUs — AMD ROCm Blogs (2026-07-23)](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom/README.html)
- [MLPerf Training v6.0 — AMD ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html)
- [MLPerf Inference v6.0 — AMD ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html)
- [MLPerf Training v6.0 Supplemental Discussion — MLCommons](https://mlcommons.org/benchmarks/training/)
