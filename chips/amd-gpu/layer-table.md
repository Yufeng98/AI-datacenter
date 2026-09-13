# AMD GPU Layer Mapping Table

*as_of: 2026-08-08*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch ROCm (torch.cuda via HIP, TunableOp, HIP Graphs, CachingAllocator) | confirmed | hip-rocm, miopen-rocblas |
| Framework Integration | JAX ROCm (XLA backend, hipBLASLt + MIOpen via hipDNN) | confirmed | miopen-rocblas |
| Framework Integration | vLLM / SGLang (inference serving, AITER kernel dispatch) | confirmed | aiter |
| Framework Integration | HIP C++ Language Extensions (__global__, __device__, __shared__, triple-chevron) | confirmed | hip-rocm |
| Framework Integration | HIPRTC (runtime compilation API) | confirmed | hip-rocm |
| Framework Integration | AMD Primus / Primus-LM (open training framework: pretraining, SFT/LoRA, RL; Megatron-LM / TorchTitan / JAX MaxText backends; unified cluster CLI; v26.4) | confirmed | summary, hip-rocm |
| Framework Integration | ROCm 7.14.0 framework support: PyTorch 2.12.0, JAX 0.10.0, vLLM 0.23.0, TensorFlow 2.21 | confirmed | hip-rocm |
| Compiler / IR | hipcc / amdclang++ (LLVM AMDGPU offload compiler) | confirmed | hip-rocm |
| Compiler / IR | Triton ROCm (Python DSL, AMDGPU backend) | confirmed | aiter |
| Compiler / IR | XLA ROCm (HLO to LLVM to AMDGPU IR) | confirmed | miopen-rocblas |
| Compiler / IR | torch.compile / TorchInductor (Triton codegen for ROCm) | confirmed | hip-rocm, aiter |
| Op Library | MIOpen (Conv, BN, Pooling, Attention; Fusion API; hipDNN provider) | confirmed | miopen-rocblas |
| Op Library | hipBLASLt (GEMM + epilogue fusion, cuBLASLt-compatible API) | confirmed | miopen-rocblas |
| Op Library | rocBLAS (BLAS L1-3, traditional GEMM) | confirmed | miopen-rocblas |
| Op Library | hipDNN (graph-based engine dispatch, provider plugins; emerging) | confirmed | miopen-rocblas |
| Kernel Library | AITER (AI Tensor Engine for ROCm: attention, MoE, GEMM, multi-backend dispatch) | confirmed | aiter |
| Kernel Library | CK / CK-Tile (Composable Kernel: GEMM/Conv/FMHA templates, runtime codegen) | confirmed | aiter, miopen-rocblas |
| Kernel Library | Hand-written Assembly HSACO (.co blobs for gfx942/gfx950) | confirmed | aiter |
| Kernel Library | Tensile (offline GEMM code-generator, pre-compiled .co kernels) | confirmed | miopen-rocblas |
| Kernel Library | rocRoller (JIT GEMM compiler, Origami tile selection; emerging) | confirmed | miopen-rocblas |
| Kernel Library | hipCUB / rocPRIM (parallel primitives: reduce, scan, sort) | confirmed | hip-rocm |
| Runtime | HIP Runtime API (libamdhip64.so: Streams, Events, Graphs) | confirmed | hip-rocm |
| Runtime | CLR / rocclr (ROCm Compute Language Runtime, virtual device layer) | confirmed | hip-rocm |
| Runtime | ROCr / HSA Runtime (libhsa-runtime64.so: AQL queues, Signals, Memory pools) | confirmed | hip-rocm |
| Runtime | ROCm SMI / AMD SMI (hardware telemetry, topology query) | confirmed | rccl |
| Runtime | HIP Execution Context APIs (GPU compute-resource partitioning; ROCm 7.14.0) | confirmed | hip-rocm |
| Runtime | Batch memory APIs (hipMemDiscardBatchAsync, hipMemPrefetchBatchAsync); faster HIP graph replay for async allocations (ROCm 7.14.0) | confirmed | hip-rocm |
| Runtime | TheRock — modular ROCm build-and-release system, production in ROCm 7.14.0 | confirmed | hip-rocm |
| Runtime | ROCprofiler-SDK beta Streaming Performance Monitors + PyTorch Profiler integration (ROCm 7.14.0) | confirmed | hip-rocm |
| Runtime | ROCm.AI / Hyperloom — agentic kernel generation and autonomous inference tuning (announced 2026-07-23; no public release/version; performance figures are unverified vendor claims) | announced | summary |
| Driver / Firmware | amdgpu kernel module (Linux GPU driver) | confirmed | hip-rocm |
| Driver / Firmware | KFD (Kernel Fusion Driver: GPUVM, queues, doorbells, XNACK/HMM) | confirmed | hip-rocm |
| Communication | RCCL (AllReduce, AllGather, ReduceScatter; Ring/Tree; NCCL fork) | confirmed | rccl |
| Communication | XGMI/Infinity Fabric P2P Transport (P2P_DIRECT via HIP IPC) | confirmed | rccl |
| Communication | InfiniBand / RoCE Network Transport (ibverbs, GPU-Direct RDMA) | confirmed | rccl |
| Communication | Rail-Optimized Multi-Node Trees | confirmed | rccl |
| Communication | Rome/EPYC Topology Model Database (~30 pre-computed templates) | confirmed | rccl |
| Communication | RCCL hierarchical AllGather (inter-node / intra-node phase separation) + direct reduce-scatter (ROCm 7.14.0) | confirmed | hip-rocm, rccl |
| Communication | UALink-over-Ethernet (UALoE) 72-GPU scale-up domain — supersedes the 8-GPU XGMI mesh assumption in RCCL's pre-computed topology templates for CDNA 5 | confirmed | hw-architecture |
| Communication | Iris (Triton-based GPU-initiated comms for reduce-scatter/all-gather) | confirmed | aiter |
| Communication | GPU-aware MPI (PeerDirect RDMA, UCX transport) | confirmed | hw-architecture |
| Assembler / ISA | AMDGPU IR (LLVM virtual ISA, target-agnostic within GFX family) | confirmed | hip-rocm |
| Assembler / ISA | GFX ISA: gfx942 (MI300X, CDNA3) | confirmed | hip-rocm, aiter |
| Assembler / ISA | GFX ISA: gfx950 (MI355X / MI350X / MI350P, CDNA4) | confirmed | aiter |
| Assembler / ISA | GFX ISA: gfx1151 / gfx1153 (Ryzen AI APU targets added in ROCm 7.14.0) | confirmed | hip-rocm |
| Assembler / ISA | GFX ISA for CDNA 5 / MI455X — **not disclosed**; ROCm 7.14.0 compatibility matrix stops at gfx950 for Instinct, and neither the CDNA 5 blog nor the ROCm 7.14 blog names a target | not disclosed | hw-architecture, summary |
| Assembler / ISA | CDNA 5 numeric/ISA additions: E5M3 scale format, 4-bit Tensor LUT instruction, Tensor Data Movers (no public ISA documentation yet) | confirmed (existence); microarchitecture not disclosed | hw-architecture |
| Assembler / ISA | HSACO .co Code Objects (compiled binary format, runtime-loadable) | confirmed | aiter, miopen-rocblas |
| Assembler / ISA | UDNA unified GFX ISA target: single --offload-arch covers gaming + datacenter; HSACO blobs reusable across UDNA generations without recompilation (resolves gfx942/gfx950 incompatibility). *Not materialized as of 2026-08-08 — CDNA 5 shipped as a CDNA-branded architecture and ROCm still enumerates per-generation gfx targets.* | roadmap | summary |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | CU: 4x SIMD16 + Matrix Core (MFMA) per CU (MI300X: 304 CUs, MI350X: 256 CUs) | confirmed | hw-architecture, hip-rocm |
| Compute Engine | CDNA3 Matrix Core: FP64/FP32/FP16/BF16/FP8/INT8 | confirmed | hw-architecture, aiter |
| Compute Engine | CDNA4 Matrix Core: +FP6/FP4 (OCP MX), 2x throughput/CU | confirmed | hw-architecture |
| Compute Engine | CDNA 5 WGP (MI455X): 256 WGPs = 8 XCDs x 32 active (34 physical); 2 shader engines/XCD x 16 WGPs; 4 dual-issue Wave32 SIMD32 + 4 matrix units per WGP; 2.4 GHz | confirmed (third-party deep-dive) | hw-architecture |
| Compute Engine | CDNA 5 peak: 40.26 PFLOPS MXFP4 and 315 TFLOPS FP32 (confirmed); 20.13 PFLOPS MXFP6/FP8 and 5.03 PFLOPS FP16/BF16 (not independently confirmed). MI430X: up to 288 TFLOPS FP64 | mixed — see per-figure note | hw-architecture, summary |
| Compute Engine | CDNA 5 numeric/ISA additions: E5M3 scale format, 4-bit Tensor LUT instruction, Tensor Data Movers | confirmed (AMD blog); microarchitecture not disclosed | hw-architecture |
| Compute Engine | MI455X transistor count and TBP | not disclosed | hw-architecture |
| Data Path | 64-thread wavefronts, 4x SIMD16 over 4 cycles, round-robin scheduler (CDNA1–CDNA4 only) | confirmed | hw-architecture, hip-rocm |
| Data Path | CDNA 5: native Wave32, 4 dual-issue SIMD32 units per WGP; Wave64 removal from the ISA not disclosed | confirmed (AMD blog + deep-dive) | hw-architecture |
| Data Path | Dual-issue FP32 (CDNA3+): vector + scalar/branch simultaneously | confirmed | hw-architecture |
| Data Path | XCD: 8 chiplets per package, self-contained compute + L2 (CDNA3/CDNA4); CDNA 5 moves L2 off the XCD onto Fabric-and-Cache Dies | confirmed | hw-architecture |
| On-chip Memory | LDS: 64 KB/CU, 32 banks, programmer-managed scratchpad (CDNA3/CDNA4) | confirmed | hw-architecture, hip-rocm |
| On-chip Memory | CDNA 5 LDS: 320 KB per WGP — largest change for CK-Tile / AITER tile sizing | confirmed (third-party deep-dive) | hw-architecture |
| On-chip Memory | L1 Vector Cache: 32 KB/CU; L1 Scalar: 16 KB/CU (CDNA3/CDNA4) | confirmed | hw-architecture |
| On-chip Memory | CDNA 5 L1: 128 KB per WGP (a 64 KB figure in some coverage is not confirmed); 128 KB vector registers per SIMD (not independently confirmed) | mixed | hw-architecture |
| On-chip Memory | L2 Cache: 4 MB/XCD, 32 MB total (MI300X) | confirmed | hw-architecture |
| On-chip Memory | CDNA 5 global L2: 192 MB = 2 x 96 MB on two Fabric-and-Cache Dies; 27 TB/s per FCD = 54 TB/s aggregate; no separate Infinity Cache tier | confirmed (third-party deep-dive; AMD says only "larger L2 caches") | hw-architecture |
| On-chip Memory | Infinity Cache: 256 MB on IODs, package-wide LLC (CDNA3/CDNA4 only) | confirmed | hw-architecture |
| Off-chip Memory | HBM3 (MI300X): 192 GB, 5.3 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM3e (MI325X): 288 GB, 6.0 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM3e (MI350X): 288 GB, ~8 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM4 (MI455X / MI430X): 432 GB, 23.3 TB/s — 12 stacks x 36 GB, 2,048-bit per stack, 192 channels; AMD frames it as 2.91x prior gen. *Supersedes the pre-launch 19.6 TB/s figure.* | confirmed | hw-architecture, summary |
| Host Interface / Package | PCIe Gen 5.0 x16: 128 GB/s bidir (CDNA3/CDNA4) | confirmed | hw-architecture |
| Host Interface / Package | MI300X: 8 XCD + 4 IOD, SoIC 3D + CoWoS, 146B transistors, 750W | confirmed | hw-architecture |
| Host Interface / Package | MI350X: 8 XCD + 2 IOD, TSMC N3P + N6, 185B transistors, 1400W | confirmed | hw-architecture |
| Host Interface / Package | MI455X: 8 XCD (TSMC N2) + 2 IOD + 2 Fabric-and-Cache Die (TSMC N3) + 12 HBM4 stacks; CoWoS-L; host link is 256 GB/s bidir 16-lane Infinity Fabric to EPYC "Venice" (not PCIe) | confirmed (process split at keynote-slide level) | hw-architecture |
| Host Interface / Package | Helios rack: 72x MI455X + 18x EPYC "Venice"; 2.9 EFLOPS MXFP4; 31 TB HBM4; up to 36 TB DDR5; ~1.7 PB/s aggregate HBM BW. Rack power, rack format and bus-bar voltage **not disclosed** | confirmed / partly not disclosed | hw-architecture, summary |
| Scale-up Interconnect | Infinity Fabric / XGMI: 7 links x 128 GB/s bidir per MI300X GPU | confirmed | hw-architecture, rccl |
| Scale-up Interconnect | 8-GPU full mesh: 896 GB/s aggregate per GPU (CDNA3/CDNA4) | confirmed | hw-architecture, rccl |
| Scale-up Interconnect | CDNA 5 UALink-over-Ethernet (UALoE): 36 x 400 Gb/s per GPU = 3.6 TB/s bidir; 72-GPU single scale-up domain; 260 TB/s rack scale-up; switched (AMD + Broadcom). Specific "12x Tomahawk 6 in a 12-plane topology" not confirmed; in-network reduction not disclosed | confirmed / partly not confirmed | hw-architecture |
| Scale-out Interconnect | Pensando Pollara 400: 400 Gbps Ethernet, UEC-ready | confirmed | hw-architecture |
| Scale-out Interconnect | GPU-Direct RDMA: PeerDirect NIC-to-GPU DMA | confirmed | hw-architecture, rccl |
| Scale-out Interconnect | InfiniBand NDR/HDR + RoCEv2 via UCX transport | confirmed | hw-architecture, rccl |
| Scale-out Interconnect | CDNA 5 / Helios scale-out: 43 TB/s per 72-GPU rack via Pensando "Vulcano" NICs. Per-GPU 2,400 Gb/s is consistent with the rack total but not independently stated; "3x UALink128 ports" and "12x Salina 400G DPUs" not confirmed. *Supersedes the pre-launch 300 GB/s/GPU figure.* | confirmed / partly not confirmed | hw-architecture, summary |
