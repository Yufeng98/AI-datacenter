# NVIDIA GPU Hardware Architecture

*as_of: 2026-08-08*
*Architectures: Hopper (H100/H200), Blackwell (B200/GB200), and Rubin (Vera Rubin NVL72 — preliminary vendor specifications)*

---

## Overview

NVIDIA datacenter GPUs are organized around arrays of Streaming Multiprocessors (SMs), each an independent multi-threaded processor, connected through a memory hierarchy (registers, shared memory, L2 cache, HBM) and interconnect fabric (NVLink, NVSwitch, InfiniBand). This document covers all 7 hardware layers with quantitative specifications for the Hopper and Blackwell generations.

---

## 1. Compute Engine

### Streaming Multiprocessor (SM)

The SM is the fundamental compute building block. Each SM contains:

| Component | Per SM |
|-----------|--------|
| FP32 CUDA Cores | 128 |
| Tensor Cores | 4 |
| Warp Schedulers | 4 |
| Max Concurrent Warps | 64 (2,048 threads) |
| Max Thread Blocks | 16-32 |
| Special Function Units (SFUs) | 16 |
| Load/Store Units | 32 |
| Register File | 256 KB (65,536 x 32-bit) |

**GPU-level SM counts:**

| SKU | SMs | Total FP32 Cores | Process |
|-----|-----|-------------------|---------|
| H100 SXM5 | 132 | 16,896 | TSMC 4N |
| H200 SXM5 | 132 | 16,896 | TSMC 4N |
| B200 | 296 (2x148) | 37,888 | TSMC 4NP |

### Tensor Core Generations

| Gen | Architecture | Supported Formats | Peak Throughput (per GPU) |
|-----|-------------|-------------------|--------------------------|
| 4th | Hopper (H100) | FP64, TF32, FP16, BF16, FP8 (E4M3/E5M2), INT8 | 989 TFLOPS BF16, ~2 PFLOPS FP8, 67 TFLOPS FP64 |
| 5th | Blackwell (B200) | FP64, TF32, FP16, BF16, FP8, MXFP6, MXFP4, INT8 | ~5 PFLOPS BF16, ~10 PFLOPS FP8, ~20 PFLOPS FP4, 45 TFLOPS FP64 |
| 6th | Rubin (R100) — preliminary | NVFP4, FP8/FP6 (fuller list not disclosed) | 50 PFLOPS NVFP4 inference, 35 PFLOPS NVFP4 training, 17.5 PFLOPS FP8-FP6 training; BF16 and FP64 peaks **not disclosed** |

Hopper's **Transformer Engine** dynamically selects between FP8 and 16-bit precision per transformer layer for up to 9x faster training vs A100.

Blackwell adds **MXFP4/MXFP6** microscaling formats (OCP-defined): shared E8M0 scale factor per block of 32 values, enabling highest arithmetic density at reduced dynamic range.

Rubin adds a **new Transformer Engine with adaptive compression** to boost NVFP4 inference performance (NVIDIA Rubin platform page wording; *recorded 2026-08-08*). No quantified figures are published for the adaptive-compression mechanism, and NVIDIA does not describe its implementation. All Rubin figures in this document carry NVIDIA's own disclaimer: "Preliminary information. All values are up to and subject to change."

---

## 2. Data Path

### SIMT Execution Model

- **Warp**: 32 threads executing the same instruction in lock-step. The fundamental scheduling unit.
- **Independent Thread Scheduling** (Volta+): Each thread has its own program counter; divergence handled via predication rather than a reconvergence stack.
- **4 Warp Schedulers per SM**: Each manages up to 16 warps. Dual-issue capability allows two independent instructions from the same warp dispatched to different functional units.
- **Latency hiding**: With 64 resident warps per SM, the scheduler round-robins among ready warps, tolerating ~200-800 cycle HBM latencies.

### Pipeline Stages

1. **Instruction Fetch** -- warp scheduler selects ready warp
2. **Decode** -- operand availability checked via scoreboard
3. **Issue** -- up to 2 instructions dispatched per scheduler per cycle
4. **Execute** -- FP/INT ALU, Tensor Core MMA, SFU, or LD/ST
5. **Writeback** -- results to register file; scoreboard cleared

### Hopper Additions

- **Tensor Memory Accelerator (TMA)**: Dedicated DMA engine per SM for asynchronous 1D-5D tensor transfers between global and shared memory. Single-thread issue; descriptor-based. Decouples data movement from compute.
- **Thread Block Clusters**: Cooperative groups of up to 8 Thread Blocks guaranteed concurrent on proximate SMs. Share distributed shared memory (DSMEM) for SM-to-SM data exchange without L2 round-trips. Cluster-level barriers for synchronization.

### Blackwell Additions

- **Tensor Memory (TMEM)**: Dedicated on-SM accumulator buffer, decoupled from the general register file. `tcgen05.mma` writes results to TMEM rather than registers.
- **2-SM Cooperative MMA**: `tcgen05.mma.cta_group::2` enables a CTA pair spanning 2 SMs to cooperatively compute a 256x256x16 tile.

---

## 3. On-chip Memory

### Register File

| Spec | Value |
|------|-------|
| Size per SM | 256 KB (65,536 x 32-bit) |
| Banks | 32 |
| Latency | sub-cycle |
| Occupancy Impact | More registers/thread = fewer active warps |

### Shared Memory / L1 Cache

Unified SRAM pool per SM, configurable split between shared memory and L1:

| Architecture | Total Pool | Max Shared Memory | Max L1 Cache |
|-------------|------------|-------------------|--------------|
| Ampere (A100) | 192 KB | 164 KB | 192 KB |
| Hopper (H100) | 256 KB | 228 KB | 256 KB |
| Blackwell (B200) | ~256 KB | 228 KB | ~256 KB |

Shared memory is organized into 32 banks (32-bit width). Bank conflicts serialize accesses within a warp. Accessed via `__shared__` in CUDA or `tl.` in Triton (compiler-managed).

### L2 Cache

| Architecture | L2 Capacity | L2 Partitions | Notes |
|-------------|-------------|---------------|-------|
| A100 | 40 MB | 2 | |
| H100 | 50 MB | 2 | L2 residency controls added |
| B200 | ~96-128 MB | 4 | 2-2.5x H100; improved bandwidth |

Hopper introduced **L2 cache residency controls**: `cudaStreamAttrValue::accessPolicyWindow` pins data regions in L2 across kernel launches.

### Tensor Memory (Blackwell)

New on-SM hardware buffer dedicated to MMA accumulator results. Managed via `tcgen05.alloc` / `tcgen05.dealloc` PTX instructions. Eliminates register pressure from large accumulator tiles.

---

## 4. Off-chip Memory

### HBM Specifications

| SKU | HBM Gen | Stacks | Capacity | Bandwidth | Interface |
|-----|---------|--------|----------|-----------|-----------|
| H100 SXM5 | HBM3 | 5 | 80 GB | 3.35 TB/s | 5120-bit |
| H200 SXM5 | HBM3e | 6 | 141 GB | 4.8 TB/s | 6144-bit |
| B200 | HBM3e | 8 | 192 GB | 8.0 TB/s | 8192-bit |
| Rubin (R100) — preliminary | HBM4 | Not disclosed | 288 GB | 22 TB/s | Not disclosed |

All stacks are integrated on the same CoWoS 2.5D interposer as the GPU die(s).

### Memory Controllers

NVIDIA uses a crossbar between memory controllers and SM L2 slices. Blackwell's dual-die connects each die to its own HBM stacks, bridged by a **10 TB/s NV-HBI** die-to-die link for unified 192 GB address space.

### Arithmetic Intensity Implications

| Precision | B200 FLOPS | B200 HBM BW | AI Crossover (FLOP/Byte) |
|-----------|-----------|-------------|--------------------------|
| FP4 | 20 PFLOPS | 8 TB/s | 2,500 |
| FP8 | 10 PFLOPS | 8 TB/s | 1,250 |
| FP16/BF16 | 5 PFLOPS | 8 TB/s | 625 |
| FP32 (non-TC) | ~120 TFLOPS | 8 TB/s | 15 |

Most attention layers are memory-bound; most large GEMM layers are compute-bound. This boundary drives fusion strategies.

---

## 5. Host Interface / Package

### PCIe

| Interface | Bandwidth | Notes |
|-----------|-----------|-------|
| PCIe Gen 5.0 x16 | ~128 GB/s bidir | Standard host interface |
| PCIe Gen 4.0 x16 | ~64 GB/s bidir | Older servers |

### NVLink-C2C (CPU-GPU Coherent Link)

| Configuration | Bandwidth | Coherent? |
|--------------|-----------|-----------|
| GH200 (Grace + H100/H200) | 900 GB/s bidir | Yes |
| GB200 (Grace + 2x Blackwell) | 900 GB/s bidir | Yes |

NVLink-C2C enables a unified cache-coherent CPU+GPU memory pool spanning LPDDR5X (Grace) and HBM.

### Chiplet Packaging

| Architecture | Die Config | Process | Transistors | Die Size |
|-------------|-----------|---------|-------------|----------|
| Hopper (H100) | Monolithic | TSMC 4N | ~80B | ~814 mm^2 |
| Blackwell (B200) | Dual chiplet | TSMC 4NP | ~208B | 2x ~800 mm^2 |

Blackwell uses TSMC CoWoS-L 2.5D packaging. The two dies are connected by a 10 TB/s NV-HBI link on the interposer, presenting a single logical GPU to software.

---

## 6. Scale-up Interconnect (NVLink + NVSwitch)

### NVLink Generations

| Gen | Architecture | Links/GPU | Per-GPU BW | Per-Link BW |
|-----|-------------|-----------|------------|-------------|
| 3rd | Ampere (A100) | 12 | 600 GB/s | 50 GB/s |
| 4th | Hopper (H100) | 18 | 900 GB/s | 50 GB/s |
| 5th | Blackwell (B200) | 18 | 1.8 TB/s | 100 GB/s |
| 6th | Rubin (R100) — preliminary | Not disclosed | 3.6 TB/s | Not disclosed |

### NVSwitch

| Gen | Architecture | Ports | Aggregate BW | Max GPUs |
|-----|-------------|-------|-------------|----------|
| 3rd | Hopper | 32 | 3.6 TB/s | 8 (per HGX) |
| 5th | Blackwell | 64 | 7.2 TB/s | 576 |
| 6th | Rubin (NVLink 6 Switch) — preliminary | Not disclosed | Not disclosed (rack total 260 TB/s switch bandwidth in VR NVL72) | 72 per NVL72 rack; 576 across eight racks (Rubin Ultra NVL576) |

NVSwitch implements a non-blocking all-to-all NVLink crossbar. With NVLink SHARP, the switch hardware can perform in-network reductions (AllReduce) directly.

### GB200 NVL72

| Spec | Value |
|------|-------|
| Blackwell GPUs | 72 |
| Grace CPUs | 36 |
| NVLink domain | All 72 GPUs |
| Total GPU-GPU BW | 130 TB/s |
| Total HBM | 13.4 TB |
| NVLink-C2C per pair | 900 GB/s |
| Cooling | Liquid-cooled rack |

All 72 GPUs are in a single NVLink domain, enabling trillion-parameter model inference without cross-node communication.

---

## 7. Scale-out Interconnect (InfiniBand / RoCE / GPUDirect)

### ConnectX-7 HCA

| Spec | Value |
|------|-------|
| InfiniBand speed | NDR 400 Gb/s (single port) |
| Dual-port | 800 Gb/s |
| Ethernet | 400 GbE (RoCE v2) |
| Host interface | PCIe Gen 5.0 x16 |
| RDMA | Hardware-offloaded |
| SHARP | In-network AllReduce |

### GPUDirect RDMA

NIC DMAs directly to/from GPU HBM, eliminating CPU memory bounce buffers. Critical for NCCL cross-node AllReduce performance. Works with both InfiniBand and RoCE when NIC and GPU share PCIe fabric.

### Canonical Cluster Topology

**Intra-node (scale-up):** NVLink + NVSwitch -- all GPUs in a node fully connected at 900 GB/s (H100) or 1.8 TB/s (B200) per GPU.

**Inter-node (scale-out):** ConnectX-7 NDR InfiniBand in a fat-tree topology with SHARP aggregation switches. GPUDirect RDMA enables direct GPU-to-GPU data paths across nodes.

**GB200 NVL72 racks:** NVLink handles all intra-rack traffic (72 GPUs). InfiniBand connects rack-level switches for inter-rack scale-out. This collapses the intra-node/inter-node distinction for the first time.

---

## 8. Rubin Architecture (Next Generation — 2026+)

### 8.1 Vera Rubin Platform Overview

The **NVIDIA Vera Rubin platform** is the successor to Blackwell, announced at CES 2026. It has been in **full production since May 2026**; NVIDIA states production shipments begin in **fall 2026**, and as of 2026-08-08 no customer shipments have been announced (NVIDIA newsroom, 2026-05-31). It is the first NVIDIA platform designed with extreme co-design across all six (later seven) chips simultaneously: Rubin GPU, Vera CPU, NVLink 6 Switch, ConnectX-9 SuperNIC, BlueField-4 DPU, Spectrum-6 Ethernet Switch, and (subsequently) **NVIDIA Groq 3 LPX** — an SRAM-based inference accelerator; NVIDIA's Rubin page frames the pairing as "combining Rubin GPUs for high-bandwidth memory (HBM) and LPUs for static random-access memory (SRAM)." *(Name corrected from "Groq 3 LPU" on 2026-08-08.)*

### 8.2 Rubin GPU Specifications

| Spec | Rubin (R100) | Blackwell (B200) |
|------|-------------|-----------------|
| Process node | TSMC 3nm | TSMC 4NP (custom) |
| HBM generation | HBM4 | HBM3e |
| HBM capacity per GPU | 288 GB | 192 GB |
| HBM bandwidth per GPU | 22 TB/s | 8.0 TB/s |
| Peak NVFP4 inference | 50 PFLOPS | 20 PFLOPS (FP4) |
| Peak NVFP4 training | 35 PFLOPS | ~14 PFLOPS (FP4) |
| Peak FP8-FP6 training | 17.5 PFLOPS | ~10 PFLOPS (FP8) |
| NVLink generation | NVLink 6 | NVLink 5 |
| NVLink per-GPU BW | 3.6 TB/s | 1.8 TB/s |
| Transformer Engine | 6th gen, **adaptive compression** for NVFP4 inference (no figures published) | 5th gen, MXFP4/MXFP6 |

HBM4 provides 2.8× the bandwidth of HBM3e (22 TB/s vs 8 TB/s). NVLink 6 provides a 50% bandwidth improvement over NVLink 5 (3.6 TB/s vs 1.8 TB/s bidirectional per GPU, a 2× improvement). FP4 compute doubles vs Blackwell (50 vs 20 PFLOPS).

### 8.3 NVLink 6

Sixth-generation scale-up fabric delivering **3.6 TB/s bidirectional bandwidth per GPU**, a 2× increase over NVLink 5. The NVLink 6 Switch is one of the seven co-designed chips in the Vera Rubin platform.

### 8.4 Vera Rubin NVL72

| Spec | VR NVL72 | GB200 NVL72 |
|------|---------|-------------|
| Rubin/Blackwell GPUs | 72 | 72 |
| Vera/Grace CPUs | 36 | 36 |
| Total HBM capacity | 20.7 TB | 13.4 TB |
| Scale-up bandwidth | 260 TB/s | 130 TB/s |
| HBM bandwidth total | 1.6 PB/s | ~576 TB/s |
| FP4 inference per rack | 3.6 EFLOPS | ~1.4 EFLOPS |

System-level gains vs Blackwell: up to 10× reduction in inference token cost, 4× fewer GPUs for MoE training, 5× greater inference throughput per rack.

### 8.5 Rubin CPX — Inference Specialization

NVIDIA introduced **Rubin CPX** as a new GPU class optimized for massive-context inference, targeting extremely long context windows and high token-rate serving rather than general-purpose training workloads.

### 8.6 Full Generation Roadmap

| Generation | Platform | Release | HBM | FP4 Perf/GPU |
|---|---|---|---|---|
| Hopper | H100/H200 | 2022-2023 | HBM3/3e | — |
| Blackwell | B200/GB200 | 2024-2025 | HBM3e (192 GB, 8 TB/s) | 20 PFLOPS |
| Blackwell Ultra | B300 | H2 2025 | HBM3e (288 GB) | ~30 PFLOPS |
| Rubin | VR NVL72 | H2 2026 | HBM4 (288 GB, 22 TB/s) | 50 PFLOPS |
| Rubin Ultra | — | H2 2027 | HBM4E (1 TB) | 100 PFLOPS |
| Feynman | — | 2028 | TBD | TBD |

**Rubin Ultra (2027):** Four reticle-limited GPU chiplets per socket, 1 TB HBM4E, 100 PFLOPS FP4. Deployable in **Rubin Ultra NVL576**, which per NVIDIA "will combine eight separate MGX NVL racks, each with 72 Rubin Ultra GPUs" into a single NVLink domain via a two-layer all-to-all topology — i.e. 576 GPUs across **eight** racks, not per rack. *(Corrected 2026-08-08; the earlier "144 quad-chiplet GPUs / 576 compute chiplets per rack" reading was wrong.)* **Feynman (2028):** Next architecture after Rubin, named after physicist Richard Feynman; details limited as of August 2026.

---

## 9. Platform Update (2026-08-08)

*No new NVIDIA silicon generation and no change to any published Rubin number in the 2026-04-05 → 2026-08-08 window. The Vera Rubin NVL72 specification page was re-verified against Sections 8.2 and 8.4 above and matches exactly. Everything below is either a correction to previously recorded content or coverage the survey previously lacked; the sources for the rack-ladder and networking items are an NVIDIA blog dated 2026-03-16, i.e. they predate the survey baseline and are not window developments.*

### 9.1 NVLink domain ladder (correction)

NVIDIA's Vera Rubin POD blog (2026-03-16) defines the scale-up ladder as:

| Domain | Composition | NVLink domain size |
|---|---|---|
| Vera Rubin NVL72 | 1 rack: 72 Rubin GPUs + 36 Vera CPUs | 72 GPUs |
| Rubin Ultra NVL576 | **8 MGX NVL racks × 72 Rubin Ultra GPUs**, two-layer all-to-all NVLink | 576 GPUs |
| Kyber NVL144 | 1 Kyber rack — "will double the NVLink domain per rack to fit 144 GPUs" | 144 GPUs |
| NVL1152 | 8 Kyber racks | 1,152 GPUs |

This supersedes the earlier "NVL576 = 576 GPUs per rack" statement in both this file and `summary.md`.

### 9.2 Vera Rubin rack-level networking

| Component | Specification | Placement |
|---|---|---|
| Spectrum-6 SPX Ethernet switch | 102.4 Tb/s, 512 lanes, 200 Gb/s co-packaged optics | Rack/pod scale-out |
| ConnectX-9 SuperNIC | Per-port rate not disclosed in this source | 8 per compute tray |
| BlueField-4 DPU | "Combines the Vera CPU and ConnectX-9 SuperNIC" | 1 per compute tray |

### 9.3 Vera Rubin NVL72 rack totals as re-verified

3,600 PFLOPS NVFP4 inference; 20.7 TB HBM4; 1,580 TB/s aggregate HBM bandwidth; 260 TB/s NVLink switch bandwidth; 3,168 Olympus CPU cores; 54 TB LPDDR5X; 1,296 total chips. Per Vera CPU: 88 Olympus cores, 1.5 TB LPDDR5X, NVLink-C2C at 1.8 TB/s. NVIDIA labels all of these "Preliminary information. All values are up to and subject to change."

### 9.4 System-level benchmark datapoint — MLPerf Training v6.0 (2026-06-16)

Results are for **Blackwell/Blackwell Ultra** silicon, not Rubin. NVIDIA's own results post reports DeepSeek-V3 671B (new benchmark this round) to target quality in **2.02 min on 8,192 GB300**, Llama 3.1 405B in **7.07 min on 8,192 GB200**, GPT-OSS 20B in **7.43 min on 512 GB300**, and Llama 3.1 8B in **4.46 min on 1,024 GB200**, using NeMo container 26.06, full-iteration CUDA Graphs for token-dropless MoEs, and Spectrum-X Ethernet Advanced Adaptive Routing.

### 9.5 Not included

Third-party reporting (SemiAnalysis via CNBC, 2026-07-06) that the Kyber rack system slips to 2028 is analyst speculation with no NVIDIA statement and no retrievable primary source; it is excluded. Hot Chips 38 (Aug 23–25, 2026) is disclosure scheduled — content not yet public — and contributes nothing here.

---

## Sources

- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
- [NVIDIA Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA GB200 NVL72 Technical Blog](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/)
- [NVIDIA ConnectX-7 400G Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband/connectx-7-datasheet.pdf)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [Microbenchmarking NVIDIA Blackwell](https://arxiv.org/html/2512.02189v1)
- [NVIDIA Vera Rubin Platform Product Page](https://www.nvidia.com/en-us/data-center/technologies/rubin/)
- [NVIDIA Kicks Off the Next Generation of AI With Rubin — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer)
- [Inside the NVIDIA Vera Rubin Platform: Six New Chips, One AI Supercomputer — NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [NVIDIA Vera Rubin NVL72 Product Page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [NVIDIA Vera Rubin NVL72 Detailed: 260 TB/s Scale-Up Bandwidth — VideoCardz](https://videocardz.com/newz/nvidia-vera-rubin-nvl72-detailed-72-gpus-36-cpus-260-tb-s-scale-up-bandwidth)
- [Nvidia announces Rubin GPUs in 2026, Rubin Ultra in 2027, Feynman — Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after)
- [Rubin (microarchitecture) — Wikipedia](https://en.wikipedia.org/wiki/Rubin_(microarchitecture))
- [NVIDIA Unveils Rubin CPX — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference)
- [Vera Rubin Ramps Into Full Production — NVIDIA Newsroom (2026-05-31)](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)
- [NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems, One AI Supercomputer — NVIDIA Technical Blog (2026-03-16)](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/)
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)
- [NVIDIA Blackwell Tops MLPerf Training v6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/)
