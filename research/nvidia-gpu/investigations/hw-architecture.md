# NVIDIA GPU Hardware Architecture Investigation

**Layer**: Hardware Architecture
**Chip**: NVIDIA GPU (Hopper H100/H200, Blackwell B100/B200/GB200, and Rubin — preliminary vendor specifications)
**as_of**: 2026-08-08

---

## Overview

NVIDIA GPUs are the dominant AI accelerators in large-scale datacenter deployments. Their architectural design is organized around arrays of **Streaming Multiprocessors (SMs)** — independent compute engines that share a common memory hierarchy and interconnect fabric. Each generation of NVIDIA datacenter GPU since Volta (2017) has incrementally deepened the Tensor Core pipeline (specialized matrix-multiply units), widened memory bandwidth via HBM stacks, and scaled out connectivity via NVLink.

The two current-generation families are:

| Family | Key SKUs | Process | Dies | Transistors |
|--------|----------|---------|------|-------------|
| Hopper | H100 SXM5, H100 PCIe, H200 | TSMC 4N (custom) | Monolithic | ~80 B |
| Blackwell | B100, B200, GB200 NVL72 | TSMC 4NP (custom) | Dual chiplet (2 × ~800 mm²) | ~208 B |

The shift to a dual-die chiplet on Blackwell is the most significant packaging change since the introduction of HBM, enabling a 2× die-area increase without exceeding the TSMC reticle limit.

---

## 1. Compute Engine

### 1.1 Streaming Multiprocessor (SM) Architecture

The SM is NVIDIA's fundamental compute building block. Each SM is a self-contained multi-threaded processor containing:

- **CUDA Cores** (FP32 and INT32 ALUs): 128 per SM on both Hopper and Blackwell.
- **Tensor Cores**: Specialized matrix-multiply-accumulate (MMA) units; 4 per SM.
- **Special Function Units (SFUs)** for transcendentals.
- **Load/Store units** for memory traffic.
- **Warp schedulers**: 4 per SM, each managing up to 16 active warps (up to 64 concurrent warps total per SM).

GPU-level counts:
- **H100 SXM5**: 132 SMs
- **H200**: 132 SMs (same die as H100)
- **B200**: 148 SMs per die (GB102 die); dual-die configuration gives 296 SMs for the full B200 package (Blackwell B200 GPU product uses the full 208B transistor package)

### 1.2 Tensor Core Generations

| Generation | Architecture | Key Formats Supported |
|---|---|---|
| 1st gen | Volta (V100) | FP16 |
| 2nd gen | Turing | FP16, INT8, INT4, INT1 |
| 3rd gen | Ampere (A100) | FP64, FP32 (TF32), FP16, BF16, INT8 |
| 4th gen | Hopper (H100/H200) | FP64, FP32 (TF32), FP16, BF16, FP8 (E4M3, E5M2), INT8 |
| 5th gen | Blackwell (B100/B200) | FP64, TF32, FP16, BF16, FP8, INT8, MXFP6, MXFP4 |

**Hopper 4th-gen Tensor Cores** deliver 2× the MMA throughput of A100 on equivalent data types and 4× on the new FP8 data type. The Hopper **Transformer Engine** dynamically selects between FP8 and 16-bit precision per transformer layer, enabling up to 9× faster training versus A100.

**Blackwell 5th-gen Tensor Cores** add native support for the OCP-defined **MXFP4** and **MXFP6** microscaling formats. MXFP4 delivers the highest arithmetic density by assigning a shared scale factor per block of 32 consecutive values. Peak performance on B200:
- **FP4 (NVFP4) Tensor**: ~20 petaFLOPS
- **FP8 / FP6 Tensor**: ~10 petaFLOPS
- **FP16 / BF16 Tensor**: ~5 petaFLOPS
- **FP64 Tensor**: ~45 teraFLOPS

### 1.3 Data Types Summary

| Format | Bits | Notes |
|--------|------|-------|
| FP64 | 64 | HPC, scientific; Tensor Core support from Ampere |
| FP32 / TF32 | 32 / 19 | TF32 crops mantissa to 10 bits for 8× A100 throughput |
| FP16 | 16 | Standard DNN training and inference |
| BF16 | 16 | Brain float; 8-bit exponent, preferred for training stability |
| FP8 E4M3 | 8 | Higher precision; preferred for inference forward pass |
| FP8 E5M2 | 8 | Wider range; preferred for gradient storage |
| MXFP6 | 6 | Microscaling, block-shared exponent; Blackwell |
| MXFP4 | 4 | Highest density; block-shared exponent; Blackwell |
| INT8 | 8 | Quantized inference |

---

## 2. Data Path

### 2.1 SIMT Execution Model

NVIDIA GPUs implement **Single Instruction Multiple Threads (SIMT)**. Threads are grouped into **warps** of 32. All 32 threads of a warp execute the same instruction in lock-step, but each thread maintains its own register context and can diverge via predicated execution. Divergent branches are serialized using a SIMT stack (or, from Volta onward, independent thread scheduling).

### 2.2 Warp Schedulers and Issue

Each SM contains **4 warp schedulers**, each responsible for a partition of the SM's resident warps:
- Up to **64 warps** can be resident per SM (2,048 threads).
- Each scheduler selects a ready warp every clock cycle.
- A dual-issue mechanism allows two independent instructions from the same warp to be dispatched simultaneously to different functional units (e.g., one FP ALU + one memory operation), improving instruction-level parallelism (ILP).
- Warp schedulers hide latency by round-robining among many resident warps: while one warp waits on a memory access (~200 clock cycles for HBM), another ready warp runs.

### 2.3 Instruction Pipeline Stages

The SM front-end follows an in-order fetch/decode pipeline with out-of-order memory completion:

1. **Instruction Fetch** — Warp scheduler selects a warp with a ready instruction.
2. **Decode** — Instruction decoded and operand availability checked (scoreboard).
3. **Issue** — Up to 2 instructions dispatched to functional unit queues.
4. **Execute** — FP/INT ALU, Tensor Core MMA, SFU, or LD/ST units execute.
5. **Writeback** — Results written to register file; scoreboard cleared.

The Hopper architecture added a **Tensor Memory Accelerator (TMA)** — a dedicated DMA engine within the SM that performs asynchronous multi-dimensional (up to 5D) tensor moves between global memory and shared memory, decoupling data movement from execution.

### 2.4 Thread Block Clusters (Hopper+)

Hopper introduced **Thread Block Clusters**: a cooperative group of up to 8 Thread Blocks guaranteed to execute concurrently across multiple SMs. Clusters share a distributed shared memory space and can exchange data via direct SM-to-SM transfers without going through L2 cache, reducing latency for producer-consumer pipelines.

---

## 3. On-chip Memory

### 3.1 Register File

Each SM has a **256 KB register file**, organized as 32 banks of 2 KB each. The register file is the fastest storage tier, with sub-cycle read latency. Occupancy (the number of resident warps) is limited by register file pressure: if a kernel uses many registers per thread, fewer warps can be active simultaneously.

### 3.2 L1 Cache / Shared Memory (SMEM)

The L1 data cache and shared memory share a unified SRAM pool per SM:

| Architecture | Shared Memory / SM | Notes |
|---|---|---|
| Ampere (A100) | up to 164 KB | 192 KB total pool |
| Hopper (H100) | up to 228 KB | 256 KB total pool; 39% increase vs A100 |
| Blackwell (B200) | up to 228 KB | Same per-SM; 19% increase reported vs H100 in some configs |

Shared memory is explicitly managed by the programmer (via CUDA `__shared__`), allowing cooperative data reuse within a thread block. The remainder of the pool acts as an L1 data cache for global memory accesses. The split ratio is configurable at kernel launch.

The L1 cache also incorporates a **texture cache** and supports hardware-managed memory access coalescing.

### 3.3 L2 Cache

The L2 cache is shared across all SMs on the GPU:

| Architecture | L2 Cache Capacity | Notes |
|---|---|---|
| A100 | 40 MB | |
| H100 | 50 MB | 1.25× A100 |
| Blackwell (B200) | ~96–128 MB | 2–2.5× H100; 4 partitions vs 2 in Hopper |

Hopper introduced **L2 cache residency controls**: kernels can pin a portion of the L2 for specific data sets (useful for weight matrices accessed repeatedly in inference). On Blackwell the doubled L2 partition count improves bandwidth and reduces port contention at scale.

---

## 4. Off-chip Memory

### 4.1 HBM Generations

NVIDIA datacenter GPUs use High Bandwidth Memory (HBM) stacks integrated on the same package via 2.5D CoWoS interposer:

| SKU | HBM Gen | Capacity | Bandwidth |
|-----|---------|---------|-----------|
| H100 SXM5 | HBM3 | 80 GB | 3.35 TB/s |
| H200 SXM5 | HBM3e | 141 GB | 4.8 TB/s |
| B100 | HBM3e | 192 GB | 8.0 TB/s |
| B200 | HBM3e | 192 GB | 8.0 TB/s |
| GB200 (full rack, 72 GPUs) | HBM3e | 13.4 TB aggregate | ~576 TB/s aggregate |

HBM3e improves on HBM3 via higher per-pin data rates and wider stacks. The B200's 8 TB/s represents a ~2.4× improvement over the H100 and enables feeding the 5th-gen Tensor Cores at peak throughput.

### 4.2 Memory Controllers

NVIDIA GPUs use a crossbar interconnect between the memory controllers and the SM L2 slices. Blackwell's dual-die architecture connects each die to its own HBM stacks, with a 10 TB/s die-to-die NV-High Bandwidth Interface (NV-HBI) link allowing either die to access the full 192 GB unified address space with minimal overhead.

---

## 5. Host Interface / Package

### 5.1 PCIe

Standard host-to-device interface:

- **H100 PCIe**: PCIe Gen 5.0 × 16 (~128 GB/s bidirectional)
- **H100 SXM5 / H200 / B200**: Use SXM mezzanine; PCIe is still present for management but not the primary data path in HGX/DGX baseboard configurations.

PCIe Gen 5 delivers 7× higher bandwidth than PCIe Gen 4 and is 7× slower than NVLink-C2C, illustrating the performance gap for CPU-GPU coupling.

### 5.2 NVLink-C2C (Grace Hopper / Grace Blackwell)

For tightly integrated CPU+GPU modules:
- **GH200 Grace Hopper Superchip**: Grace CPU + H100/H200 GPU connected by NVLink-C2C at **900 GB/s** bidirectional, coherent. Enables a unified, cache-coherent CPU+GPU memory pool spanning LPDDR5X and HBM.
- **GB200 Grace Blackwell Superchip**: Grace CPU + two Blackwell GPU dies; same NVLink-C2C technology.

NVLink-C2C eliminates the PCIe bottleneck for workloads with significant CPU-to-GPU data movement and enables CPU-side coherency for pointer-traversal workloads.

### 5.3 Chiplet Packaging (Blackwell)

Blackwell uses TSMC CoWoS-L 2.5D packaging:
- Two GB100 dies (~800 mm² each, TSMC 4NP custom node).
- Joined by a **10 TB/s NV-HBI** die-to-die link, presenting a single logical GPU to software.
- 208 billion total transistors across both dies.
- The CoWoS-L interposer also carries the HBM3e stacks, keeping total wire lengths short for maximum bandwidth.

Hopper (H100) uses a monolithic ~814 mm² die — effectively at the single-reticle limit — which is why Blackwell pivoted to dual-die.

---

## 6. Scale-up Interconnect (NVLink + NVSwitch)

### 6.1 NVLink Generations

| NVLink Gen | Architecture | Per-GPU BW | Notes |
|---|---|---|---|
| 3rd gen | Ampere (A100) | 600 GB/s | |
| 4th gen | Hopper (H100) | 900 GB/s | 50% increase vs 3rd gen |
| 5th gen | Blackwell (B200) | 1.8 TB/s | 2× Hopper |

NVLink is NVIDIA's proprietary GPU-to-GPU interconnect, delivering much higher bandwidth and lower latency than PCIe. Within a single server or DGX node, NVLink enables direct peer GPU memory access.

### 6.2 NVSwitch

NVSwitch is a purpose-built ASIC that implements a non-blocking NVLink crossbar, enabling all-to-all GPU communication without host CPU involvement:

- **3rd-gen NVSwitch** (DGX H100): 8 GPU HGX baseboard; 900 GB/s per GPU.
- **5th-gen NVSwitch** (Blackwell): 64 NVLink ports per switch; 7.2 TB/s aggregate bandwidth per switch; scales to 576 GPUs in a non-blocking fabric.

### 6.3 GB200 NVL72

The GB200 NVL72 is a rack-scale system integrating 72 Blackwell GPUs and 36 Grace CPUs in a single liquid-cooled rack:

- **NVLink domain**: All 72 GPUs connected as a single NVLink fabric.
- **Total GPU-to-GPU bandwidth**: 130 TB/s across the full rack.
- **Total HBM3e**: 13.4 TB unified memory.
- **NVLink-C2C**: Each Grace CPU connected to its paired Blackwell GPUs at 900 GB/s, coherent.
- Software sees the 72 GPUs as one logical accelerator for LLM inference of trillion-parameter models.

---

## 7. Scale-out Interconnect (InfiniBand, RoCE, GPUDirect)

### 7.1 InfiniBand — ConnectX-7

For inter-node GPU cluster communication:

- **NVIDIA ConnectX-7** HCA (Host Channel Adapter): NDR InfiniBand at **400 Gb/s** (single port); dual-port configurations reach 800 Gb/s.
- Supports both InfiniBand (IB) and Ethernet (400GbE / RoCE) on the same ASIC.
- PCIe Gen 5.0 × 16 host interface.
- Hardware-offloaded RDMA, collective operations via **SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol) executed in-network on InfiniBand switches, reducing GPU idle time during AllReduce.

### 7.2 RoCE (RDMA over Converged Ethernet)

RoCE v2 enables RDMA semantics over standard Ethernet at 400 GbE using ConnectX-7. It is the Ethernet-native alternative to InfiniBand for cloud providers who run Clos-fabric Ethernet networks rather than fat-tree InfiniBand fabrics.

### 7.3 GPUDirect RDMA

GPUDirect RDMA allows the NIC to **DMA data directly to/from GPU HBM**, bypassing CPU and system memory:
- Eliminates double-copy: network adapter ↔ CPU memory ↔ GPU (legacy) becomes network adapter ↔ GPU HBM (direct).
- Critical for distributed training and inference: NCCL uses GPUDirect RDMA to implement cross-node AllReduce with minimal CPU involvement.
- Works with both InfiniBand and RoCE when the NIC and GPU share PCIe fabric.

### 7.4 Topology at Scale

A canonical H100/B200 AI training cluster uses a two-tier fabric:
1. **Scale-up tier** (intra-node): NVLink + NVSwitch for near-zero latency GPU-to-GPU.
2. **Scale-out tier** (inter-node): ConnectX-7 NDR InfiniBand in a fat-tree topology, with SHARP aggregation in switches.

For GB200 NVL72 racks, scale-out is handled by InfiniBand connecting rack switches, while the NVLink fabric handles all intra-rack traffic.

---

## Key Findings

### How Hardware Architecture Drives the Software Stack

1. **SM + Warp Model → CUDA / PTX**: The 4-scheduler, 64-warp-per-SM design is directly exposed through CUDA's thread block / warp / lane programming model. Achieving peak throughput requires saturating all warp schedulers, which the CUDA occupancy calculator formalizes.

2. **Tensor Cores → cuBLAS / cuDNN / Transformer Engine**: 4th- and 5th-gen Tensor Cores require data in specific matrix tile layouts (e.g., 16×16×16 MMA tiles for FP16). Libraries like cuBLAS abstract tile packing; Transformer Engine automates FP8/FP16 casting for Hopper, and MXFP4/MXFP6 for Blackwell.

3. **Shared Memory → Kernel Optimization**: The 228 KB/SM shared memory on Hopper/Blackwell is the primary mechanism for re-use of HBM data within a thread block (tiling). Software libraries explicitly manage SMEM layouts (e.g., CUTLASS tile descriptors, FlashAttention SMEM staging).

4. **TMA (Tensor Memory Accelerator) → Async pipelining**: Hopper's TMA enables software to overlap data loading with Tensor Core execution via `cp.async.bulk` instructions and memory barriers, yielding near-theoretical peak Tensor Core utilization.

5. **HBM Bandwidth Scaling → Memory-bound vs Compute-bound regimes**: At 8 TB/s (B200), the arithmetic-intensity crossover point for FP8 ops is ~2.5 FLOP/Byte. Most attention layers remain memory-bound even at B200; most GEMM layers are compute-bound. This boundary shapes fusion strategies (FlashAttention, fused MoE kernels).

6. **NVLink → Tensor Parallelism**: 1.8 TB/s NVLink (B200) enables tensor-parallel splits across 8 GPUs within a node without AllReduce becoming the bottleneck, supporting models too large for single-GPU HBM.

7. **NVLink Domain (NVL72) → Single-system trillion-parameter inference**: 72 GPUs with 13.4 TB combined HBM, fully connected at 130 TB/s, allows a 70T+ parameter model to reside in GPU memory entirely, enabling real-time inference without model sharding across nodes.

8. **GPUDirect RDMA + InfiniBand → Large-scale training**: Cross-node gradient synchronization via NCCL's AllReduce leverages GPUDirect RDMA over ConnectX-7 NDR, making 1,000+ GPU jobs feasible with bandwidth scaling that tracks GPU count.

---

## Rubin Architecture Roadmap

### Overview

The **NVIDIA Vera Rubin platform** is the direct successor to Blackwell, announced at CES 2026 and in full production since May 2026 (NVIDIA states production shipments begin fall 2026; see the 2026-08-08 update below). It is NVIDIA's first extreme-codesigned six-chip (later expanded to seven-chip with the NVIDIA Groq 3 LPX) AI platform, where every component — GPU, CPU, networking, DPU, power delivery, and cooling — is co-designed as a single system rather than independently optimized.

The platform is named after astrophysicist Vera Rubin. The GPU is named **Rubin** and the CPU is named **Vera**, together forming the **Vera Rubin Superchip**.

---

### Platform Composition

The Vera Rubin platform integrates seven co-designed chips:

| Chip | Role |
|------|------|
| NVIDIA Rubin GPU | Compute accelerator |
| NVIDIA Vera CPU | Host processor (88 Olympus cores) |
| NVIDIA NVLink 6 Switch | Scale-up fabric |
| NVIDIA ConnectX-9 SuperNIC | Scale-out networking |
| NVIDIA BlueField-4 DPU | Data processing unit |
| NVIDIA Spectrum-6 Ethernet Switch | Ethernet switching |
| NVIDIA Groq 3 LPX | SRAM-based low-latency inference accelerator (added post-launch; official name corrected from "Groq 3 LPU" on 2026-08-08) |

---

### Rubin GPU Specifications

| Spec | Rubin (R100) | Blackwell (B200) | Improvement |
|------|-------------|-----------------|-------------|
| Process node | TSMC 3nm | TSMC 4NP | 1 node advance |
| HBM generation | HBM4 | HBM3e | +1 gen |
| HBM capacity per GPU | 288 GB | 192 GB | +50% |
| HBM bandwidth per GPU | 22 TB/s | 8.0 TB/s | 2.8x |
| Peak FP4 inference | 50 PFLOPS | 20 PFLOPS | 2.5x |
| Peak FP4 training | 35 PFLOPS | ~14 PFLOPS (est.) | ~2.5x |
| NVLink generation | NVLink 6 | NVLink 5 | +1 gen |
| NVLink per-GPU BW | 3.6 TB/s | 1.8 TB/s | 2x |

**Key architectural changes from Blackwell:**
- Move from dual-die (2×~800 mm²) chiplet to a new die configuration enabled by the TSMC 3nm process shrink.
- HBM4 delivers 2.8× higher memory bandwidth than the HBM3e used in Blackwell, addressing the memory-bandwidth bottleneck for long-context inference.
- NVLink 6 doubles the scale-up bandwidth vs NVLink 5 (3.6 TB/s vs 1.8 TB/s per GPU bidirectional).
- Extreme co-design philosophy: GPU, CPU, networking, and storage are all designed together from inception rather than assembled post-hoc.

---

### Vera Rubin NVL72 Rack-Scale System

The **Vera Rubin NVL72** is the flagship rack-scale deployment unit, continuing the NVL72 form factor established by GB200 NVL72:

| Spec | Vera Rubin NVL72 | GB200 NVL72 |
|------|-----------------|-------------|
| GPUs | 72 Rubin GPUs | 72 Blackwell GPUs |
| CPUs | 36 Vera CPUs | 36 Grace CPUs |
| HBM capacity total | 20.7 TB | 13.4 TB |
| Scale-up BW total | 260 TB/s | 130 TB/s |
| HBM bandwidth total | 1.6 PB/s | ~576 TB/s |
| FP4 inference per rack | 3.6 EFLOPS | ~1.4 EFLOPS |

The VR NVL72 doubles effective scale-up bandwidth and rack-level HBM capacity vs GB200 NVL72, enabling even larger models to reside entirely within a single NVLink domain.

**System-level efficiency gains vs Blackwell:**
- Up to 10× reduction in inference token cost.
- 4× fewer GPUs required to train MoE models of equivalent size.
- 5× greater inference performance per rack.

---

### Rubin CPX — Specialized Inference Variant

NVIDIA also unveiled **Rubin CPX**, a new GPU class designed specifically for massive-context inference workloads. CPX (Context Processing Extreme) optimizes for extremely long context windows and high-token-rate inference serving, as opposed to the general-purpose Rubin GPU.

---

### Roadmap Context

| Generation | Platform | Release | Key Memory | FP4 Perf/GPU | Notes |
|---|---|---|---|---|---|
| Hopper | H100/H200 | 2022/2023 | HBM3/3e | N/A | FP8 Tensor Engine |
| Blackwell | B200/GB200 | 2024/2025 | HBM3e (192 GB) | 20 PFLOPS | MXFP4 support |
| Blackwell Ultra | B300 | H2 2025 | HBM3e (288 GB) | ~30 PFLOPS | 1.5x B200 compute |
| Rubin (Vera Rubin) | VR NVL72 | H2 2026 | HBM4 (288 GB) | 50 PFLOPS | TSMC 3nm; NVLink 6 |
| Rubin Ultra | — | H2 2027 | HBM4E (1 TB) | 100 PFLOPS | Quad-chiplet; NVL576 |
| Feynman | — | 2028 | TBD | TBD | Named after physicist Richard Feynman |

**Rubin Ultra (2027):** Four reticle-limited GPU chiplets in a single socket, 1 TB HBM4E memory, 100 PFLOPS FP4, deployable in **Rubin Ultra NVL576** — per NVIDIA, "eight separate MGX NVL racks, each with 72 Rubin Ultra GPUs" in one NVLink domain, i.e. 576 GPUs across eight racks. *(Corrected 2026-08-08; see section D of the update below. The earlier "144 quad-chiplet GPUs = 576 compute chiplets per rack" reading was wrong — 144 GPUs per rack is the separate Kyber step.)*

**Feynman (2028):** Post-Rubin architecture currently in planning. Limited public details; continues the one-year cadence.

---

## Update — 2026-08-08 (re-verification window 2026-04-05 → 2026-08-08)

**Headline: no new NVIDIA silicon generation, and no change to any published Rubin specification.** Every per-GPU and per-rack figure recorded in the Rubin sections above was re-checked directly against NVIDIA's Vera Rubin NVL72 specification page and matches. That page carries NVIDIA's own disclaimer — "Preliminary information. All values are up to and subject to change" — which the survey should keep attached to every Rubin number.

### A. Figures re-verified (no edits required)

Per GPU: 50 PFLOPS NVFP4 inference; 35 PFLOPS NVFP4 training; **17.5 PFLOPS FP8-FP6 training** (newly recorded — the earlier tables omitted the FP8/FP6 line); 288 GB HBM4 at 22 TB/s; NVLink 6 at 3.6 TB/s.
Per Vera CPU: 88 Olympus cores; 1.5 TB LPDDR5X; NVLink-C2C at 1.8 TB/s.
Per VR NVL72 rack: 3,600 PFLOPS NVFP4 inference; 20.7 TB HBM4; 1,580 TB/s aggregate HBM bandwidth; 260 TB/s NVLink switch bandwidth; 3,168 Olympus CPU cores; 54 TB LPDDR5X; 1,296 total chips.

### B. Production status — tightened

NVIDIA newsroom, dateline **2026-05-31**: Vera Rubin "ramps into full production"; "**Production shipments of Vera Rubin are set to begin starting this fall.**" So as of 2026-08-08 the platform is in production but customer shipments have not started. The earlier phrasing "entering full production in early 2026 … partner products shipping H2 2026" is superseded. The release also claims **"10x agent throughput at scale compared with the previous-generation NVIDIA Grace Blackwell platform"** — an NVIDIA marketing claim with no published methodology.

Named cloud/AI adopters in that release: CoreWeave, Firmus, GMI Cloud, IBM Cloud, IREN, Lambda, Microsoft Azure, Nebius, Nscale, SpaceXAI, Vultr. Named system builders (15): ASRock Rack, ASUS, Dell, Foxconn, GIGABYTE, HPE, IBM, Inventec, Lenovo, MSI, Pegatron, QCT, Supermicro, Wistron, Wiwynn. **Oracle Cloud Infrastructure is not in this release** and must not be listed as a Rubin adopter on its authority.

### C. New architectural detail — Rubin Transformer Engine

NVIDIA's Rubin platform page: Vera Rubin NVL72 "features a **new Transformer Engine with adaptive compression** to boost NVFP4 inference performance." Treat as a 6th-generation Tensor Core / TE capability. Two caveats recorded deliberately: (1) **no quantified figures are published**, and (2) the page is undated, so there is no evidence this wording postdates the 2026-04-05 baseline — it is a coverage gap being closed, not a window development.

### D. NVLink domain ladder — factual correction to prior repo content

Source: NVIDIA's Vera Rubin POD technical blog, **2026-03-16** (pre-baseline; a valid repo fix, not news).

| Domain | NVIDIA's definition | GPUs |
|---|---|---|
| Vera Rubin NVL72 | One rack: 72 Rubin GPUs + 36 Vera CPUs | 72 |
| Rubin Ultra NVL576 | "Eight separate MGX NVL racks, each with 72 Rubin Ultra GPUs," two-layer all-to-all NVLink topology | 576 across 8 racks |
| Kyber NVL144 | Kyber "will double the NVLink domain per rack to fit 144 GPUs" | 144 per rack |
| NVL1152 | Eight Kyber racks | 1,152 |

The prior repo statements — "NVL576 (576 GPUs/rack)" in `chips/nvidia-gpu/summary.md` and "NVL576 systems with 144 quad-chiplet GPUs (576 compute chiplets per rack)" in `chips/nvidia-gpu/hw-architecture.md` and in the Roadmap Context section above — were both wrong and have been corrected.

### E. Vera Rubin rack networking (same 2026-03-16 blog)

- **Spectrum-6 SPX**: 102.4 Tb/s, 512 lanes, 200 Gb/s co-packaged optics.
- **8× ConnectX-9 SuperNIC per compute tray** (per-port rate not stated in this source).
- **1× BlueField-4 DPU per compute tray**; BlueField-4 "combines the Vera CPU and ConnectX-9 SuperNIC."

### F. Seventh chip naming

NVIDIA's official name is **NVIDIA Groq 3 LPX** (corrected above). NVIDIA's Rubin page frames the pairing as "combining Rubin GPUs for high-bandwidth memory (HBM) and LPUs for static random-access memory (SRAM)" and claims only "more tokens per watt and lower cost per token compared to the NVIDIA Blackwell architecture" — unquantified. The circulating "up to 35× higher tokens per watt vs Blackwell NVL72" figure could **not** be traced to an NVIDIA primary source and is excluded from the survey.

### G. Systems-level benchmark datapoint — MLPerf Training v6.0

Published **2026-06-16**; 24 submitting organizations; two new benchmarks — **DeepSeek-V3** (671B total / 37B activated per token, C4 dataset, 3.6 log-perplexity target) and **GPT-OSS 20B** (21B total / 3.6B activated). NVIDIA's own results post reports: DeepSeek-V3 671B in **2.02 min on 8,192 GB300**; Llama 3.1 405B in **7.07 min on 8,192 GB200**; GPT-OSS 20B in **7.43 min on 512 GB300**; Llama 3.1 8B in **4.46 min on 1,024 GB200**; software stack **NeMo container 26.06**, full-iteration CUDA Graphs for token-dropless MoEs, Spectrum-X Ethernet Advanced Adaptive Routing. These are Blackwell/Blackwell Ultra results — **no Rubin MLPerf submission exists**. Per-submitter attribution requires the MLCommons results table, which was not opened, so no vendor-by-vendor claims are recorded.

### H. Explicitly excluded

- **Kyber-to-2028 slip** (SemiAnalysis via CNBC, 2026-07-06): third-party analyst speculation; the CNBC article returned HTTP 403 and no NVIDIA statement exists either way. Not recorded.
- No confirmed change to **Rubin Ultra (H2 2027)** or **Feynman (2028)** timing; no **NVLink Fusion** news in the window.
- **Hot Chips 38 (Aug 23–25, 2026)**: 15 days in the future as of this update — disclosure scheduled, content not yet public. No talk title is used as a source for any spec.

---

## Sources

- [NVIDIA Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture)
- [NVIDIA Blackwell Architecture Official Product Page](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Hopper Architecture In-Depth — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper v1.01](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
- [NVIDIA H100 GPU Product Page](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 GPU Product Page](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA GB200 NVL72 Product Page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [NVIDIA GB200 NVL72 Technical Blog](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/)
- [NVIDIA ConnectX-7 400G Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband/connectx-7-datasheet.pdf)
- [NVIDIA H100 Transformer Engine Blog](https://blogs.nvidia.com/blog/h100-transformer-engine/)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [Blackwell Microarchitecture — Wikipedia](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))
- [Microbenchmarking NVIDIA Blackwell — arXiv 2512.02189](https://arxiv.org/html/2512.02189v1)
- [Nvidia introduces Blackwell 800mm² N4P dies — SemiWiki](https://semiwiki.com/forum/threads/nvidia-introduces-blackwell-800mm2-reticle-limit-n4p-dies.19856/)
- [GPUDirect RDMA deep-dive — QSysArch](https://qsysarch.com/posts/gpu-direct-from-rdma-from-pcie-to-using-doca/)
- [NVIDIA Vera Rubin Platform Product Page](https://www.nvidia.com/en-us/data-center/technologies/rubin/)
- [NVIDIA Kicks Off the Next Generation of AI With Rubin — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [Inside the NVIDIA Vera Rubin Platform: Six New Chips, One AI Supercomputer — NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [NVIDIA Vera Rubin NVL72 Product Page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [NVIDIA Vera Rubin NVL72 Detailed: 72 GPUs, 36 CPUs, 260 TB/s Scale-Up Bandwidth — VideoCardz](https://videocardz.com/newz/nvidia-vera-rubin-nvl72-detailed-72-gpus-36-cpus-260-tb-s-scale-up-bandwidth)
- [NVIDIA Launches Next-Generation Rubin AI Compute Platform at CES 2026 — ServeTheHome](https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/)
- [Nvidia announces Rubin GPUs in 2026, Rubin Ultra in 2027, Feynman also added to roadmap — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after)
- [Nvidia's Vera Rubin platform in depth — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-vera-rubin-platform-in-depth-inside-nvidias-most-complex-ai-and-hpc-platform-to-date)
- [Rubin (microarchitecture) — Wikipedia](https://en.wikipedia.org/wiki/Rubin_(microarchitecture))
- [NVIDIA Unveils Rubin CPX — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)

### Added 2026-08-08
- [Vera Rubin Ramps Into Full Production — NVIDIA Newsroom (2026-05-31)](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory) — production status, "shipments … starting this fall", adopter and OEM lists, 10x agent-throughput claim
- [NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems, One AI Supercomputer — NVIDIA Technical Blog (2026-03-16)](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/) — NVL576 = 8 racks × 72 GPUs; Kyber NVL144 / NVL1152; Spectrum-6 SPX 102.4 Tb/s; ConnectX-9 / BlueField-4 per tray
- [NVIDIA Vera Rubin NVL72 Product Page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) — full per-GPU and per-rack spec table plus the "Preliminary information" disclaimer
- [NVIDIA Rubin Platform Page](https://www.nvidia.com/en-us/data-center/technologies/rubin/) — "new Transformer Engine with adaptive compression"; "Rubin GPUs for HBM and LPUs for SRAM"
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — round composition, DeepSeek-V3 and GPT-OSS 20B benchmarks, 24 submitters
- [NVIDIA Blackwell Tops MLPerf Training v6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/) — headline times and NeMo container 26.06
