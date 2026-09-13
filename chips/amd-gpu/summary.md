# AMD GPU Software and Hardware Stack Summary

*as_of: 2026-08-08*

---

## Overview

AMD Instinct GPUs are the primary challenger to NVIDIA in datacenter AI training and inference. Their architecture is organized around arrays of **Compute Units (CUs)** -- independent SIMD processors containing 4x SIMD16 vector pipelines, a Matrix Core (MFMA) unit, register files (VGPR/SGPR/AGPR), and Local Data Share (LDS) -- connected through a multi-level memory hierarchy (L1/L2/Infinity Cache/HBM) and the Infinity Fabric interconnect (XGMI for GPU-to-GPU). The datacenter generations are **CDNA3** (MI300X: 304 CUs, 192 GB HBM3, 5.3 TB/s), **CDNA4** (MI350X/MI355X: 256 CUs with 2x throughput/CU, 288 GB HBM3e, ~8 TB/s), and, from July 2026, **CDNA 5** (MI455X: 256 WGPs across 8 XCDs, 432 GB HBM4, 23.3 TB/s, native Wave32, UALink-over-Ethernet scale-up). See the [CDNA 5 / MI455X / Helios update](#cdna-5--instinct-mi455x--helios-update-julyaugust-2026) below — CDNA 5 breaks several long-standing CDNA invariants described in the older sections of this document.

AMD's ecosystem strategy is defined by the **ROCm platform**: an open-source software stack that mirrors CUDA's layer structure -- HIP (runtime API), hipcc/amdclang++ (compiler), MIOpen/hipBLASLt (op libraries), CK/AITER (kernel libraries), RCCL (communication) -- while targeting fundamentally different hardware (64-wide wavefronts on CDNA1-CDNA4, AQL queue dispatch, chiplet MCM packaging with Infinity Fabric). The deliberate isomorphism with CUDA minimizes porting cost for the existing CUDA ecosystem, while AMD's open-source approach (including hand-written assembly kernels in AITER) provides transparency that NVIDIA's closed-source cuDNN/cuBLAS do not.

---

## Software Stack

### Framework Integration

AMD GPUs are supported as backends for the major ML frameworks:

- **PyTorch ROCm**: The primary framework for AMD AI workloads. PyTorch's ROCm backend uses HIP runtime for device management and kernel launches, hipBLASLt/rocBLAS for GEMM (selected per-shape by `TunableOp` since PyTorch 2.2), MIOpen for convolution/batch-norm/attention, and hipFFT for FFT. `torch.compile()` with TorchInductor generates Triton kernels targeting the ROCm Triton backend. CUDA Graphs map to HIP Graphs (`hipGraph_t`).
- **JAX ROCm**: XLA's ROCm backend uses MIOpen via the hipDNN plugin for convolution and hipBLASLt for GEMM/dot operations. MIOpen's Fusion API is exposed to XLA for operator fusion (conv + bias + activation as single kernels).
- **vLLM / SGLang**: The primary inference framework targets for AMD. AITER provides drop-in replacements for vLLM's CUDA custom kernels (paged attention, Flash Attention, FusedMoE, GEMM, KV-cache management) on AMD hardware.
- **HIP C++ Language**: The `__global__`, `__device__`, `__shared__` annotations and `hipLaunchKernelGGL` / triple-chevron syntax provide direct GPU kernel authoring isomorphic to CUDA C++. `hipify` tools mechanically translate CUDA source to HIP.
- **HIPRTC**: Runtime compilation API enabling JIT compilation of HIP kernels to any installed GFX target at runtime.
- **AMD Primus** (added 2026-08-08): AMD's open large-scale training framework (`AMD-AIG-AIMA/Primus`) for foundation-model pretraining, post-training (SFT/LoRA) and RL on Instinct GPUs. Wraps **Megatron-LM, TorchTitan and JAX MaxText** backends behind a unified cluster CLI. AMD's MLPerf Training v6.0 submissions were its first MLPerf use. Latest release v26.4.

### Compiler / IR

- **hipcc / amdclang++**: `hipcc` is a thin compiler driver wrapper that invokes `amdclang++`, an LLVM-based offload compiler. Device code is compiled through the LLVM AMDGPU backend, producing AMDGPU IR (virtual ISA) before lowering to GFX ISA machine code. The `--offload-arch=gfxXXX` flag selects the target (e.g., `gfx942` for MI300X, `gfx950` for MI350). Code objects (`.co`) are embedded in host ELF binaries.
- **Triton ROCm**: OpenAI's Triton compiler has a ROCm backend that generates AMDGPU IR instead of PTX. AITER ships Triton-based kernels for normalization, decode MLA, and communication where Triton provides competitive performance. AOT-compiled Triton HSACO blobs are stored in `aiter/aot/triton/`.
- **torch.compile / TorchInductor**: PyTorch 2.x's compilation pipeline generates Triton kernels for fused operations on ROCm, using the same graph capture and optimization passes as the CUDA path.
- **XLA ROCm**: Google's compiler backend for JAX/TensorFlow on ROCm. Lowers HLO through LLVM to AMDGPU IR.

### Op Library

- **MIOpen** (AMD's cuDNN equivalent): Provides hand-tuned implementations of convolutions (GEMM, Winograd, FFT, Direct algorithms), batch normalization, pooling, and attention (through CK-tile FMHA pipelines). The Fusion API (`miopenFusionPlan_t`) enables multi-operator fusion (conv + bias + batchnorm + activation) compiled to single CK or OpenCL kernels, eliminating intermediate HBM round-trips. Primary consumers are PyTorch ATen ROCm backend and TensorFlow/XLA.
- **hipBLASLt / rocBLAS** (AMD's cuBLASLt/cuBLAS equivalents): hipBLASLt provides GPU-accelerated GEMM with fused epilogues (bias, ReLU, GELU, SiLU/Swish) via `hipblasLtMatmul`. Algorithm selection uses ranked heuristics keyed on problem shape and GPU target. Internally dispatches to Tensile (offline code-generated MFMA assembly `.co` kernels) or rocRoller (JIT kernel compiler using Origami tile-size prediction). rocBLAS handles traditional BLAS levels 1-3. PyTorch's `TunableOp` auto-benchmarks hipBLASLt vs rocBLAS per shape and caches the winner.
- **hipDNN** (emerging): New graph-based engine dispatch layer with provider plugins (`miopen-provider`, `hipblaslt-provider`) behind a common API, analogous to cuDNN 8's backend API. Under active development in the `rocm-libraries` monorepo.

### Kernel Library

- **AITER** (AI Tensor Engine for ROCm): AMD's centralized production-grade kernel library for AI operators -- the structural counterpart to NVIDIA's CUTLASS/cuDNN ecosystem. Provides a unified Python+C++ API covering FMHA/MLA attention (prefill and paged-decode), MoE routing and fused MLP, GEMM with FP8/INT8/INT4/BF16 precision, KV-cache management, all-reduce/reduce-scatter, RoPE, RMSNorm/LayerNorm, and sampling. Each op dispatches to the best available backend: hand-written AMDGPU assembly (`hsa/*.co`), CK/CK-Tile, Triton, HIP, or FlyDSL, depending on operator, precision, and detected GPU architecture. Per-architecture tuned CSV configuration files from CI autotuning ensure optimal kernel selection.
- **Composable Kernel (CK / CK-Tile)**: Header-only HIP C++ template library providing the shared kernel foundation for both MIOpen and hipBLASLt. CK maps GEMM to a four-level hierarchy (grid -> block -> warp -> thread -> MFMA instruction). CK-Tile is the actively developed next-generation variant with runtime Python code generation: `generate.py` emits specialized HIP source for exact dtype/feature combinations, JIT-compiled by `hipcc` and cached. This approach trades a one-time compilation cost for precise kernel specialization without shipping exponentially large binaries.
- **Hand-written Assembly (HSACO)**: Pre-compiled `.co` code objects in AITER's `hsa/gfx942/` and `hsa/gfx950/` directories covering the most performance-critical op/dtype/shape combinations (FMHA, FusedMoE, GEMM). These bypass CK entirely and represent the highest-performance path for production inference.
- **hipCUB / rocPRIM**: AMD equivalents of CUB/Thrust. hipCUB provides device-level parallel primitives (reduce, scan, sort); rocPRIM provides the underlying algorithm implementations.
- **Tensile / rocRoller**: GEMM code-generation backends for hipBLASLt. Tensile is a Python-based offline generator producing pre-compiled `.co` files with MFMA assembly. rocRoller is its emerging JIT replacement using Origami for tile-size prediction and runtime kernel compilation.

### Runtime

- **HIP Runtime API** (`libamdhip64.so`): The primary user-facing API. Provides `hipMalloc`, `hipMemcpy`, kernel launch via triple-chevron or `hipLaunchKernelGGL`, streams (`hipStream_t`), events (`hipEvent_t`), and HIP Graphs (`hipGraph_t`). Internally implemented through CLR/rocclr (ROCm Compute Language Runtime), which provides a virtual device abstraction shared with the OpenCL runtime.
- **ROCr / HSA Runtime** (`libhsa-runtime64.so`): The low-level runtime managing AQL (Architected Queuing Language) queue dispatch, agent enumeration, memory pool allocation, and signal objects. AQL packets (64-byte hardware dispatch format) are written directly into GPU-visible ring buffers; a doorbell MMIO write notifies the GPU Command Processor. This zero-copy dispatch path matches CUDA's cuLaunchKernel model for minimal CPU-GPU round-trip latency.
- **ROCm SMI / AMD SMI**: Hardware telemetry and topology query tool used by RCCL during communicator initialization to enumerate XGMI links and GPU properties.
- **HIP Graphs**: DAG of GPU operations recorded once and replayed with `hipGraphLaunch`, eliminating per-kernel driver overhead. Critical for AI inference workloads with thousands of short kernels per forward pass.

### Driver / Firmware

- **amdgpu kernel module**: Linux kernel driver for AMD GPUs. The KFD (Kernel Fusion Driver) component manages GPU process address spaces (GPUVM), AQL queue creation and doorbell mapping, memory pool allocation, signal objects, and XNACK (page-fault) handling for Heterogeneous Memory Management (HMM).
- **KFD (Kernel Fusion Driver)**: The kernel-mode interface between ROCr and GPU hardware. Manages GPU contexts, memory mappings, and hardware command queues. `libhsa-runtime64.so` calls `hsaKmtOpenKFD()` to establish sessions.

### Communication

- **RCCL** (ROCm Communication Collectives Library, "Rickle"): AMD's collective communication library -- a maintained fork of NVIDIA's NCCL providing AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv. Retains the NCCL internal namespace with AMD-specific extensions for XGMI/Infinity Fabric topology awareness. Key AMD differentiators: (1) per-generation XGMI bandwidth constants (MI300X: 48 GB/s/link); (2) ~30 pre-computed `rcclRomeModel` topology templates for known EPYC server configurations; (3) rail-optimized ring and tree algorithms for multi-node clusters; (4) optional MSCCL++ integration.
  - **Intra-node**: P2P transport uses HIP IPC (`hipIpcGetMemHandle`) with `P2P_DIRECT` mode for XGMI-connected GPUs, writing directly into peer GPU VRAM over Infinity Fabric.
  - **Inter-node**: InfiniBand Verbs (`net_ib.cc`) with GPU-Direct RDMA via the AMD kernel driver's PeerDirect interface.
  - **Algorithm selection**: `getAlgoInfo` considers topology bandwidth, message size, and rank count to select Ring/Tree/CollNet algorithm and LL/LL128/Simple wire protocol. XGMI's 48 GB/s/link bandwidth biases toward Ring (high throughput) for large AllReduces.
- **Iris**: Triton-based GPU-initiated communication library for reduce-scatter and all-gather, co-located in AITER. Enables compute-communication overlap in tensor-parallel inference.
- **GPU-aware MPI**: AMD kernel driver exposes PeerDirect RDMA interfaces; NICs can directly DMA to/from GPU HBM. UCX provides the transport layer for OpenMPI/MPICH integration.

### Assembler / ISA

- **AMDGPU IR**: LLVM's AMDGPU target IR -- the intermediate representation between HIP C++ and hardware-specific machine code. Architecture-agnostic within a GFX family; the `--offload-arch` flag selects the final ISA target.
- **GFX ISA**: Architecture-specific native machine code. Key targets: `gfx942` (MI300X, CDNA3), `gfx950` (MI350X/MI355X/MI350P, CDNA4). Not forward-compatible across families -- a binary for gfx942 will not run on gfx950. Key MFMA instructions: `v_mfma_f32_32x32x8bf16` (BF16), `v_mfma_f32_32x32x16_fp8` (FP8), extended FP4/FP6 tiles on gfx950. **The CDNA 5 (MI455X) gfx target is not disclosed** as of 2026-08-08 — the public ROCm 7.14.0 compatibility matrix stops at gfx950 for Instinct parts (it does add Ryzen AI APU targets gfx1151/gfx1153). CDNA 5 additionally introduces an E5M3 scale format, a 4-bit Tensor LUT instruction and Tensor Data Movers, none of which have public ISA-level documentation yet.
- **HSACO (.co) Code Objects**: Compiled binary format for AMD GPU kernels. Can be loaded at runtime via `hipModuleLoadData` / HSA runtime. AITER ships dozens of hand-written HSACO blobs for peak-performance operators.

---

## Hardware Architecture

### Compute Engine

Each CU contains 4 SIMD16 vector pipelines (64-wide execution per CU), one Matrix Core (MFMA) unit, VGPR/SGPR/AGPR register files, LDS, L1 caches, and a wavefront scheduler. CDNA3 (MI300X) has 304 CUs across 8 XCDs; CDNA4 (MI350X) has 256 CUs with 2x throughput per CU.

Matrix Core data type support:
- **CDNA3**: FP64, FP32, FP16, BF16, FP8, INT8. Peak: ~2.6 PFLOPS FP8, ~1.3 PFLOPS BF16.
- **CDNA4**: Adds FP6, FP4 (OCP MX micro-scale formats). Peak: ~20 PFLOPS FP4 (4x gen-on-gen).

### Data Path

AMD GPUs implement **64-wide wavefronts** on CDNA architectures. Each wavefront executes across 4 SIMD16 pipelines over 4 clock cycles. The scheduler issues in round-robin across SIMDs, with dual-issue FP32 capability (vector + scalar/branch simultaneously). MFMA instructions run on the dedicated Matrix Core independently from the main VALU pipelines, using AGPRs (Accumulation VGPRs) as accumulators.

### On-chip Memory

- **LDS**: 64 KB per CU, programmer-managed scratchpad (`__shared__`), 32 banks.
- **L1 Vector Cache**: 32 KB per CU, transparent.
- **L1 Scalar Cache**: 16 KB per CU, read-only constants/instructions.
- **L2 Cache**: 4 MB per XCD (32 MB total on MI300X); coherence point for atomics and cross-CU accesses.
- **Infinity Cache**: 256 MB on IODs (4 x 64 MB); package-wide last-level on-package cache before HBM.

### Off-chip Memory

- **MI300X**: 192 GB HBM3, 5.3 TB/s (8 stacks x 24 GB, 8192-bit bus).
- **MI325X**: 288 GB HBM3e, ~6.0 TB/s (memory-capacity refresh).
- **MI350X**: 288 GB HBM3e, ~8 TB/s (12-hi stacks, 8 Gbps/pin).

### Host Interface / Package

- **MI300X**: 8 XCDs (TSMC 5nm) 3D-stacked on 4 IODs (TSMC 6nm) via SoIC hybrid bonding; CoWoS silicon interposer with 8 HBM3 stacks. PCIe Gen 5 x16 (128 GB/s). 146B transistors, 750W TBP.
- **MI300A APU**: 3 CCDs (Zen 4, 24 CPU cores) + 6 XCDs (228 GPU CUs) + 4 IODs sharing 128 GB unified HBM3. CPU and GPU share the same memory pool with no PCIe copy overhead.
- **MI350X**: 8 XCDs (TSMC N3P) + 2 IODs (TSMC N6). 185B transistors, 1400W TBP.

### Scale-up Interconnect

- **Infinity Fabric / XGMI**: 4th-generation AMD Infinity Fabric. MI300X: 7 XGMI links per GPU, each 16 lanes x 32 Gbps = 128 GB/s bidirectional (~48 GB/s practical unidirectional). 8-GPU OAM baseboard uses full mesh topology; aggregate 896 GB/s per GPU.
- Within the MI300X package, XCDs and IODs communicate via direct die-to-die Infinity Fabric (not XGMI).

### Scale-out Interconnect

- **RCCL**: Uses XGMI for intra-node, InfiniBand/RoCEv2/TCP for inter-node.
- **Pensando Pollara 400**: AMD's AI NIC, up to 400 Gbps Ethernet, UEC-ready, 3rd-gen P4 engine. Up to 25% RCCL performance improvement vs commodity NICs. Validated for 16-1024 node clusters.
- **GPU-Direct RDMA**: AMD kernel driver exposes PeerDirect interface; NICs directly DMA to/from GPU HBM. UCX provides the transport layer.

---

## Programming Model Rationale

The AMD GPU software stack's design is driven by the interplay between ecosystem pragmatism and fundamentally different hardware:

**1. HIP mirrors CUDA for portability, but underneath ROCr/HSA is a fundamentally different runtime.** HIP's API is intentionally isomorphic to CUDA Runtime API -- `hipMalloc` maps to `cudaMalloc`, `hipStream_t` to `cudaStream_t`, and `hipify` tools mechanically translate CUDA source. This is AMD's primary ecosystem strategy: minimizing the porting barrier for the massive CUDA codebase. However, the implementation underneath is architecturally distinct. Where CUDA uses opaque driver contexts and GPU command FIFOs managed by closed-source firmware, ROCm uses AQL (Architected Queuing Language) -- an open, standardized 64-byte packet format written directly into GPU-visible ring buffers. The hot path for kernel dispatch is a userspace ring-buffer write plus a doorbell MMIO write, with no kernel-mode driver call at steady state. This open-specification approach gives AMD transparency and debuggability that CUDA's closed driver stack does not provide.

**2. 64-wide wavefronts (vs NVIDIA's 32-wide warps) affect kernel occupancy and register pressure -- on CDNA1 through CDNA4. CDNA 5 breaks this invariant.** On CDNA1-CDNA4 the fundamental SIMD execution unit is 64 threads, executing across 4 SIMD16 pipelines over 4 cycles. This 2x wider wavefront has cascading effects: each wavefront consumes 2x the register file (64 lanes x 256 VGPRs), so occupancy is more sensitive to register pressure. LDS bank conflict patterns differ (32 banks serving 64 lanes). And synchronization primitives like `__syncwarp()` have different width semantics. This is why AITER and CK maintain separate tuning configurations per GFX target rather than assuming CUDA-like parameters will transfer directly.

  **CDNA 5 (MI455X, 2026) executes native Wave32.** AMD's CDNA 5 blog lists "Wave32 compute execution", and Chips and Cheese's MI455X deep-dive describes 4 dual-issue Wave32 SIMD32 units per WGP -- an explicit shift away from Wave64-over-SIMD16. (No source states that Wave64 is *removed from the ISA*; treat "Wave64 dropped" as **not disclosed**.) Practical consequence: the occupancy, register-pressure and LDS-bank-conflict reasoning above no longer transfers to CDNA 5, and AITER / CK-Tile / Triton-ROCm tuning tables for the new target must be re-derived rather than scaled from gfx942/gfx950 configs. The 64-wide-wavefront assumption is now generation-scoped, not an AMD-wide invariant.

**3. RCCL is a maintained NCCL fork -- a pragmatic choice for ecosystem compatibility.** Rather than building a clean-room collective communication library, AMD forked NCCL and added XGMI-specific extensions under `#if defined(__HIP_PLATFORM_AMD__)` guards. This means RCCL inherits NCCL's battle-tested ring/tree algorithms and wire protocols while AMD patches in topology awareness for Infinity Fabric. The tradeoff is that RCCL also inherits NCCL assumptions (e.g., NVLink topology concepts mapped 1:1 onto XGMI via `LINK_NVL`), which may not always optimize perfectly for AMD's full-mesh XGMI topology. The `rcclRomeModel` pre-computed topology database compensates by hard-coding optimal ring/tree orderings for known EPYC server configurations.

**4. Infinity Fabric topology (XGMI mesh) shapes RCCL's ring/tree algorithms differently from NVLink/NVSwitch -- and CDNA 5 replaces the mesh with a switched 72-GPU UALoE domain.** MI300X's 8-GPU nodes use a full XGMI mesh where every GPU has a direct link to every other GPU, each at 48 GB/s. This is structurally different from NVIDIA's NVLink + NVSwitch topology, where a central switch ASIC provides non-blocking all-to-all bandwidth. RCCL's "rail-optimized trees" and pre-computed ring orderings exist specifically to exploit the XGMI mesh structure: routing intra-rail GPUs first before crossing to network NICs, and using single-slice pipeline depth on XGMI systems where latency is low (`rcclUseOneSlice`). On MI300X/MI355X AMD has no equivalent to NVLink SHARP (in-switch hardware reduction), so all-reduce operations must be executed entirely in GPU compute.

  **CDNA 5 moves off the 8-GPU XGMI mesh.** MI455X scale-up is **UALink-over-Ethernet (UALoE)** -- 36 x 400 Gb/s interfaces per GPU (1.8 TB/s each way, 3.6 TB/s bidirectional) into a **single 72-GPU scale-up domain**, switched rather than meshed, with 260 TB/s of rack-level scale-up bandwidth. The scale-up domain therefore grows 9x (8 -> 72 GPUs) and stops being a direct point-to-point mesh, which invalidates the pre-computed 8-GPU ring orderings and the `rcclRomeModel`-style templates for this generation; RCCL's 2026 work on hierarchical AllGather (separating inter-node from intra-node phases, shipped in ROCm 7.14.0) is the visible software-side response. Whether UALoE switching performs in-network reduction is **not disclosed**.

**5. AITER's multi-backend approach (CK + Triton + ASM) reflects AMD catching up on kernel optimization.** Where NVIDIA has decades of cuDNN/cuBLAS hand-tuning behind a closed-source wall, AMD takes an open, multi-pronged approach. AITER dispatches each operator to whichever backend provides the best performance: hand-written AMDGPU assembly (`.co` blobs in `hsa/`) for the highest-throughput critical paths, CK-Tile code generation for shapes not covered by assembly, Triton for ops where its compiler produces competitive results, and FlyDSL for experimental mixed-precision MoE. The CI-integrated autotuning pipeline (per-shape CSV configs measured on real MI300X/MI350 hardware) mirrors NVIDIA's cuDNN heuristic engine, but the open-source nature means the entire tuning process is auditable.

**6. The chiplet MCM packaging (XCDs + IODs) creates a NUMA-like memory topology within a single GPU.** MI300X's 8 XCDs, each with its own 4 MB L2, communicate through the 256 MB Infinity Cache on 4 IODs. Cross-XCD memory access has higher latency than intra-XCD L2 access. This is why RCCL's topology scanning includes intra-package Infinity Fabric paths, and why CK's tile-level kernel tuning must account for L2 locality per XCD. The unified HBM pool masks this complexity at the programming model level (`hipMalloc` returns a flat address), but performance-critical kernels must reason about data placement.

**7. The open-source strategy is a deliberate ecosystem differentiation.** Unlike NVIDIA's mixed open/closed approach (open kernel modules but closed cuDNN/cuBLAS/firmware), AMD's entire software stack including AITER's hand-written assembly blobs, CK templates, MIOpen, and RCCL modifications is fully open source. This attracts the growing community of AI infrastructure engineers who need to debug, profile, and customize the full stack -- particularly important for inference serving companies deploying on AMD hardware at scale.

---

## Roadmap: MI400 / UDNA (2026+)

*Added: 2026-04-05. **Partially superseded 2026-08-08** — the MI400-series architecture shipped as **CDNA 5**, not "CDNA Next" and not UDNA; UDNA/CDNA 6 is now attached to MI500 (2027). This section is retained for the UDNA strategy background; for shipping MI455X specifications see the [CDNA 5 / MI455X / Helios update](#cdna-5--instinct-mi455x--helios-update-julyaugust-2026) below.*

### UDNA — What It Is

**UDNA (Unified DNA)** is AMD's convergence of its two previously separate GPU architecture families — **RDNA** (consumer gaming, Radeon RX) and **CDNA** (datacenter AI/HPC, Instinct) — into a single unified microarchitecture, announced at IFA 2024. The split between RDNA and CDNA since 2019 created developer friction analogous to the cost CUDA avoids on the NVIDIA side: two ISAs, two toolchains, no forward compatibility. UDNA addresses this directly, mirroring NVIDIA's single-architecture strategy.

Key changes UDNA brings:
- **Matrix cores in all products**: Today only CDNA GPUs have MFMA units; RDNA gaming GPUs emulate matrix math on shader ALUs. UDNA adds dedicated matrix cores to all SKUs including gaming GPUs.
- **GCN-inspired unified ALU**: UDNA returns to a shared ALU design (analogous to the original GCN ancestor of both RDNA and CDNA) while preserving modern compute features.
- **Forward/backward ISA compatibility**: AMD has committed to full forward and backward compatibility across the first two UDNA generations (UDNA 6 and UDNA 7), eliminating today's per-generation recompilation requirement (`gfx942` vs `gfx950` binaries are not interchangeable today).
- **Single ecosystem**: One ROCm install and one ISA target for gaming, workstation, datacenter, and console (Sony PS6 will use UDNA).

### Instinct Generation Roadmap

| GPU | Architecture | Process | Status | Peak FP4 | Memory | Bandwidth |
|-----|-------------|---------|--------|----------|--------|-----------|
| MI300X | CDNA3 | TSMC 5nm + 6nm | Shipping | ~2.6 PFLOPS FP8 | 192 GB HBM3 | 5.3 TB/s |
| MI325X | CDNA3+ | TSMC 5nm + 6nm | Shipping (Q4 2024) | ~2.6 PFLOPS FP8 | 288 GB HBM3e | ~6.0 TB/s |
| MI350X | CDNA4 | TSMC N3P + N6 | Shipping (2025) | 20 PFLOPS FP4 | 288 GB HBM3e | ~8 TB/s |
| **MI455X (MI400 series)** | **CDNA 5** | TSMC N2 (XCD) + N3 (IOD/FCD) | **In production; shipping from Q3 2026** | **40.26 PFLOPS MXFP4** | **432 GB HBM4** | **23.3 TB/s** |
| MI430X (MI400 series, HPC/sovereign) | CDNA 5 | TSMC N2 (XCD) + N3 | Announced; H1 2027 availability | up to 288 TFLOPS FP64 (hardware) | 432 GB HBM4 | 23.3 TB/s |
| MI500 | CDNA 6 | not disclosed | 2027 | not disclosed | not disclosed | not disclosed |

*Table corrected 2026-08-08. The earlier "MI400 / CDNA 'Next' / 19.6 TB/s" row and the "~1,000x vs MI300X" MI500 figure were superseded: the shipping architecture is CDNA 5 at 23.3 TB/s, and the ~1,000x figure had no primary source (AMD's on-stage line was a "2000x performance improvement in just 4 years" marketing claim, not a spec). AMD's press release says only "Next-generation AMD Instinct MI500 Series GPUs are coming in 2027"; the CDNA 6 attachment comes from keynote coverage.*

### MI400-Series Key Specifications (as launched — see the CDNA 5 update below for detail)

- **Peak MXFP4**: 40.26 PFLOPS per MI455X (confirmed)
- **Peak FP32**: 315 TFLOPS (confirmed)
- **Peak MXFP6 / FP8**: 20.13 PFLOPS; **Peak FP16/BF16**: 5.03 PFLOPS — arithmetically consistent halvings, *not independently confirmed*
- **Memory**: 432 GB HBM4 (12 stacks x 36 GB, 2,048-bit bus, 192 channels)
- **Memory bandwidth**: 23.3 TB/s (AMD: "2.91x" the prior generation). *The 19.6 TB/s figure previously carried in this repo is stale.*
- **Scale-up**: UALink-over-Ethernet, 3.6 TB/s bidirectional per GPU, 72-GPU domain. *The previously listed "300 GB/s per GPU scale-out" figure is superseded.*
- **Scale-out**: 43 TB/s per Helios rack via Pensando "Vulcano" NICs
- **Rack system**: "Helios" — 72x MI455X + 18x EPYC "Venice" CPUs; 2.9 EFLOPS peak MXFP4 per rack (the earlier "3 AI exaflops" was a pre-launch rounding)

### UDNA Implications for ROCm / Software Stack

1. **Single GFX ISA target** for all AMD GPU products (gaming, workstation, datacenter)
2. **MFMA/matrix kernels portable** to consumer Radeon GPUs — AITER, CK, hipBLASLt kernel libraries apply across product lines
3. **HSACO blobs reusable** across UDNA generations without recompilation (resolves the gfx942/gfx950 incompatibility)
4. **Simplified ROCm packaging**: one `--offload-arch=gfxUDNA_X` flag covers the full product range

*Status note (2026-08-08): none of the four UDNA implications above have materialized in shipping software. CDNA 5 launched as a CDNA-branded architecture, and the public ROCm compatibility matrix still enumerates per-generation gfx targets. Treat this subsection as forward-looking strategy, not current state.*

---

## CDNA 5 / Instinct MI455X / Helios Update (July–August 2026)

*Updated: 2026-08-08. Sources: AMD Advancing AI 2026 press release (2026-07-23); AMD ROCm blog "Introducing AMD CDNA™ 5 and the AMD Helios™ Rackscale Solution" (2026-08-04); Chips and Cheese MI455X deep-dive (2026-07-23); ServeTheHome Advancing AI 2026 keynote coverage and Helios architecture deep-dive; ROCm 7.14.0 release notes and compatibility matrix (2026-07-16); AMD ROCm MLPerf Training v6.0 blog (2026-06-16).*

### What changed and what was wrong in this repo

AMD launched the MI400-series flagship at **Advancing AI 2026 on 2026-07-23** (keynote and press release both carry that date; a two-day "July 22–23" framing is not confirmed). The product itself was not new news — MI450/Helios were public from October 2025 and MI455X + Venice + Helios hardware was shown physically at CES 2026 — but the **CDNA 5 architecture disclosure, the production declaration, and the detailed specifications** all post-date this repo's 2026-04-05 baseline.

Four repo statements were superseded or wrong:

| Repo statement (pre-2026-08-08) | Corrected |
|---|---|
| MI400 architecture = "CDNA Next" / UDNA | **CDNA 5** (AMD's own blog title). UDNA/CDNA 6 attaches to MI500 (2027) |
| MI400 memory bandwidth 19.6 TB/s | **23.3 TB/s** |
| "AMD GPUs implement 64-wide wavefronts on CDNA architectures" | True for CDNA1–CDNA4 only; **CDNA 5 executes native Wave32** |
| "AMD + Meta: 6 GW of AMD GPU deployments" | The 6 GW figure is the **OpenAI** agreement (October 2025, restated in keynote coverage). Meta "has begun testing and validating workloads on AMD Helios racks" — no capacity figure |

### MI455X — Compute

- **Shader organization**: 8 XCDs x 32 active WGPs = **256 WGPs** (34 physical WGPs per XCD, 2 fused off); **2 shader engines per XCD**, 16 WGPs each; 4 matrix units per WGP. In RDNA-style nomenclature 256 WGPs corresponds to 512 CUs — do not read "256" as a CU count comparable to MI350X's 256 CUs.
- **Clock**: 2.4 GHz max engine clock.
- **Wave width**: native **Wave32**, 4 dual-issue SIMD32 units per WGP (vs CDNA3/4's Wave64 over SIMD16). Whether Wave64 remains available in the ISA is **not disclosed**.
- **Peak throughput per GPU**: **40.26 PFLOPS MXFP4** and **315 TFLOPS FP32** (confirmed); 20.13 PFLOPS MXFP6/FP8 and 5.03 PFLOPS FP16/BF16 (arithmetically consistent, *not independently confirmed*). AMD's own wording is "up to 4x greater AI compute throughput for key low-precision formats" — a **vendor marketing claim**, not a measured 4x-over-MI355X figure.
- **New numeric / ISA features** (from AMD's CDNA 5 blog, with no CDNA3/CDNA4 counterpart): an **E5M3 floating-point scale format** (an extra exponent bit for wider scale range in micro-scaled formats), a **4-bit Tensor Lookup Table (LUT) instruction** for low-precision compute, and **Tensor Data Movers**. AMD gives no microarchitectural detail for Tensor Data Movers; the common reading of them as an analogue of NVIDIA's TMA is interpretation, not vendor language.
- **Transistor count**: **not disclosed** (the 320B figure circulating in aggregator coverage is unconfirmed). Process split (N2 compute chiplets, N3 elsewhere) is corroborated only at keynote-slide level. **CoWoS-L** packaging is confirmed; "largest chip yet built on CoWoS-L" is a marketing superlative.

### MI455X — Memory hierarchy (structural change)

- **HBM**: 432 GB HBM4, 12 stacks x 36 GB, 2,048-bit per stack, 192 channels, **23.3 TB/s** (AMD: "2.91x memory bandwidth" vs prior generation; 23.3 / 8.0 ≈ 2.91, consistent).
- **L2**: **192 MB global L2, split 2 x 96 MB across two Fabric-and-Cache Dies (FCDs)**, 27 TB/s per FCD = **54 TB/s aggregate**. This replaces CDNA3's structure of 4 MB L2 per XCD plus 256 MB Infinity Cache on the IODs — the last-level on-package cache moves from a separate Infinity Cache tier into a large global L2 on dedicated cache dies. Third-party deep-dive (Chips and Cheese); AMD's own blog says only "larger L2 caches".
- **Per-WGP**: **320 KB LDS per WGP** and **128 KB L1 per WGP** (Chips and Cheese). The 320 KB LDS figure matters more than the L1 number for CK-Tile / AITER tiling. A "64 KB L1 per WGP" figure appears in some coverage and is **not confirmed**.
- **Registers**: 128 KB vector registers per SIMD (raw-scan figure, *not independently confirmed*).

### MI455X — Fabric

- **Scale-up**: **UALink-over-Ethernet (UALoE)** replaces the 8-GPU XGMI mesh. **36 x 400 Gb/s UALoE interfaces per GPU** = 1.8 TB/s each way, **3.6 TB/s bidirectional**; **72-GPU single scale-up domain**; **260 TB/s rack-level scale-up bandwidth**. (A "72 lanes" phrasing appears in some coverage and is not confirmed.)
- **Host link**: **256 GB/s bidirectional** over a dedicated 16-lane Infinity Fabric to the EPYC CPU — not PCIe.
- **Scale-out**: **43 TB/s per rack** via AMD Pensando "Vulcano" NICs. A per-GPU 2,400 Gb/s figure is consistent with that rack total (72 x 2,400 Gb/s ≈ 43.2 TB/s) but is not independently stated. "3x UALink128 ports per GPU" and "12x Pensando Salina 400G DPUs, one per compute tray" are **not confirmed**.
- **Switching**: the AMD–Broadcom hardware partnership for Helios is corroborated; the specific claim of **12x Broadcom Tomahawk 6 ASICs in a 12-plane topology** is **not confirmed**.

### Helios rack

| Attribute | Value | Evidence |
|---|---|---|
| GPUs | 72x MI455X | confirmed |
| CPUs | 18x EPYC "Venice" | confirmed |
| Aggregate HBM4 | 31 TB | confirmed |
| Aggregate DDR5 | up to 36 TB | confirmed |
| Peak MXFP4 | 2.9 EFLOPS | confirmed |
| Aggregate HBM bandwidth | ~1.7 PB/s | consistent with 72 x 23.3 TB/s ≈ 1.68 PB/s; a 1.4 PB/s aggregator figure is arithmetically inconsistent and discarded |
| Rack power | **not disclosed** | no source states the "225–245 kW" figure that circulated; AMD's press-release footnote references a "100 kW power envelope per rack" only as comparison methodology |
| Rack format / bus bar | **not disclosed** | Open Rack Wide (1.2 m x 1.3 m, 44 OU) and a 50 V liquid-cooled bus bar are reported but not confirmed against a primary AMD spec sheet |

**Availability**: Helios is "now in production to be deployed by leading AI companies at gigawatt scale" (AMD press release, 2026-07-23); on stage AMD said shipments start in Q3 2026. OpenAI "expects to bring Helios online beginning in the fourth quarter of 2026". The correct characterization is **in production / shipping from Q3 2026, ramping into Q4 2026 — not deployed at scale.**

### SKU segmentation

| SKU | Architecture | Position | Status |
|---|---|---|---|
| MI455X | CDNA 5 | Flagship rack-scale training/inference | In production; shipping from Q3 2026 |
| MI430X | CDNA 5 | HPC / sovereign; up to **288 TFLOPS hardware FP64**; same 432 GB HBM4 | Announced; **H1 2027** availability |
| MI350P | **CDNA 4 (gfx950)** | Dual-slot PCIe enterprise-inference card, announced 2026-05-07, re-promoted at Advancing AI 2026 | Shipping; listed in the ROCm 7.14.0 compatibility matrix. **Not** an MI400-series part; its memory capacity is not confirmed |

"MI450 / MI450-series" naming coexists with MI455X in AMD's own materials. **MI440X** appears in some aggregator coverage but is not in AMD's press release and is not entered here.

**Customer commitments** (corporate/roadmap announcements, not deployed capacity): Anthropic to deploy up to **2 GW** of AMD Instinct MI455X GPUs; OpenAI Helios online from Q4 2026 (under the pre-existing October 2025 6 GW agreement); Meta testing and validating workloads on Helios racks; Microsoft Azure named among Helios adopters with no capacity figure. A reported "$5B AMD investment in Anthropic" is **not confirmed**.

### Software stack changes

**ROCm 7.14.0 (released 2026-07-16; AMD blog 2026-07-15).** Headline is **TheRock**, AMD's build-and-release system, going production — "transitions ROCm to TheRock, a build and release system that introduces a modular architecture". Confirmed additions relevant to the layer stack:

| Layer | Change |
|---|---|
| Runtime | **HIP Execution Context APIs** for GPU compute-resource partitioning; batch memory APIs including `hipMemDiscardBatchAsync`; faster HIP graph replay for async allocations |
| Communication | **RCCL hierarchical AllGather** separating inter-node from intra-node communication; direct reduce-scatter |
| Profiling | **ROCprofiler-SDK beta Streaming Performance Monitors**, with PyTorch Profiler integration |
| Op Library | per-matrix bias in hipBLASLt batched GEMM |
| I/O | hipFile direct storage I/O |
| Framework Integration | PyTorch 2.12.0, JAX 0.10.0, vLLM 0.23.0, TensorFlow 2.21 |
| Assembler / ISA | adds Ryzen AI APU targets gfx1151 / gfx1153; multi-VF partition modes for MI355X/MI350X |

Versioning note: ROCm jumped 7.2.x → 7.9.0 preview → 7.10–7.14. **"ROCm 8" does not exist as of 2026-08-08.**

**Critical gap — no public gfx target for CDNA 5.** The ROCm 7.14.0 compatibility matrix's newest Instinct targets are **gfx950** (MI355X / MI350X / MI350P), gfx942, gfx90a, gfx908. There is **no MI455X / CDNA 5 / gfx96x-class entry**, and neither the CDNA 5 blog nor the ROCm 7.14 blog names a gfx target for CDNA 5. **The CDNA 5 gfx ISA identifier is not disclosed.** Hardware is in production ahead of public ISA-target disclosure — which also means the AITER `hsa/gfx*` assembly blobs, CK-Tile tuning CSVs, and `--offload-arch` flow described above have no published CDNA 5 path yet.

**AMD Primus — new framework-integration-layer component (previously absent from this repo).** Primus / Primus-LM (github.com/AMD-AIG-AIMA/Primus) is AMD's open training framework for large-scale foundation-model pretraining, post-training (SFT/LoRA) and RL on AMD GPUs, with **Megatron-LM / TorchTitan / JAX MaxText backends** and a unified cluster CLI. Latest release v26.4, actively developed through 2026-07-29. AMD states MLPerf Training v6.0 was "the first-ever use of AMD's Primus training framework in MLPerf Training submissions". A companion "Primus Tuning Agent" blog is dated 2026-07-06.

**ROCm.AI and Hyperloom (Advancing AI 2026, 2026-07-23).** ROCm.AI is described as AI-assisted kernel generation letting coding agents (Claude, Codex, Cursor) work against AMD hardware natively. Hyperloom runs agentic loops that profile a workload, find bottlenecks, and tune/rewrite kernels; AMD published a Hyperloom blog on 2026-07-23. **The performance numbers are vendor marketing claims and are not independently verified: 3.3x inference over ROCm 7, and a demo showing Hyperloom improving token rate by 38%.** A "2.4x training" figure circulated but is not confirmed. No public release or version number was found — medium confidence on existence, low on the numbers.

### Benchmarks

**MLPerf Training v6.0 (results public 2026-06-16)** — AMD's **first multi-node MLPerf Training submission**: 512x MI300X (64 nodes) with Oracle Cloud Infrastructure, plus an 8-node / 64x MI325X Flux.1-schnell FP8 data-parallel submission, plus MI350X and MI355X single-node LLM submissions. First production-ready **MXFP4 training recipe** (~2x the compute density of FP8). Gains vs MLPerf 5.1: Llama2-70B LoRA **+19% (MI355X) / +16% (MI350X)**; Llama3.1-8B pretraining **+13% (MI355X) / +11% (MI350X)**. OCI is the confirmed partner; a broader partner list circulated but is not confirmed.

**MLPerf Inference v6.0 (2026-04-01)** — AMD published MI355X single-node and multi-node results, including its first >1M tok/s aggregate result. **The specific throughput figures and the B300 comparison percentages are AMD-reported and were not independently verified in this pass — treat them as unverified vendor comparisons.**

---

## Resources

### Documentation
- [ROCm Documentation Hub](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD CDNA3 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Instinct MI300 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [MI300X Platform Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [Introducing AMD CDNA 5 and the AMD Helios Rackscale Solution — AMD ROCm Blogs (2026-08-04)](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)
- [ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [ROCm 7.14 Release Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html)

### Open-Source Repositories
- [ROCm/HIP](https://github.com/ROCm/HIP) -- HIP Runtime API
- [ROCm/aiter](https://github.com/ROCm/aiter) -- AITER AI Kernel Library
- [ROCm/rccl](https://github.com/ROCm/rccl) -- RCCL Collective Communications
- [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) -- MIOpen, rocBLAS, hipBLASLt, CK monorepo
- [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) -- CK/CK-Tile kernel templates
- [Triton ROCm Backend](https://github.com/triton-lang/triton) -- Triton with AMD GPU support
- [AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus) -- AMD open training framework (Megatron-LM / TorchTitan / JAX MaxText backends)
- [ROCm/ROCm release rocm-7.14.0](https://github.com/ROCm/ROCm/releases/tag/rocm-7.14.0)

### Hardware
- [AMD Instinct MI300X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI350 Product Page](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
- [AMD Pensando Pollara 400 AI NIC](https://www.amd.com/en/products/network-interface-cards/pensando.html)
- [Advancing AI 2026 press release — AMD Investor Relations (2026-07-23)](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era)
- [AMD's Instinct MI455X: Aiming for the Top — Chips and Cheese (2026-07-23)](https://chipsandcheese.com/p/amds-instinct-mi455x-aiming-for-the)
- [AMD Helios Architecture Deep Dive — ServeTheHome](https://www.servethehome.com/amd-helios-architecture-deep-dive-amd-broadcom-hardware-combined/)
- [AMD Advancing AI 2026 Keynote Live Coverage — ServeTheHome](https://www.servethehome.com/amd-advancing-ai-2026-keynote-live-coverage/)
