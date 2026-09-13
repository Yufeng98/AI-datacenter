# Huawei Ascend Hardware Architecture Investigation

*as_of: 2026-04-05*
*chip: huawei-ascend*
*focus: Da Vinci core, Cube Unit, Vector Unit, memory hierarchy, interconnect*

---

## Overview

The Huawei Ascend series uses the **Da Vinci architecture** — a heterogeneous AI core design developed entirely in-house by HiSilicon. Unlike GPU streaming multiprocessors (which contain general-purpose CUDA cores) or TPU systolic arrays (which are deeply pipelined weight-stationary grids), the Da Vinci core is a **multi-engine AI core** with five independent parallel execution units sharing a hierarchical on-chip memory subsystem. The architecture has evolved through three generations: Da Vinci 1.0 (Ascend 910, 2019), Da Vinci 2.0 (Ascend 910B, 2022–2023 on SMIC N+1), and Da Vinci 3.0 (Ascend 910C, 2024–2025, dual-die).

---

## 1. Da Vinci AI Core Structure

Each Da Vinci AI Core (also called an **AIC** or **AI Core**) contains five parallel functional units:

### 1.1 Cube Unit (Matrix Engine)
- Implements a **16×16×16 systolic matrix multiply-accumulate** per clock cycle in FP16
  - 16 × 16 × 16 = 4,096 FP16 MACs/cycle/core → 8,192 INT8 MACs/cycle/core
- Operand buffers: **L0A** (input matrix A, 16 KB) and **L0B** (input matrix B, 16 KB), fed by DMA from L1
- Output written to **L0C** accumulator buffer (32 KB); result drained to Unified Buffer by MTE2
- Supports FP16, INT8, INT4 (generation-dependent) precision
- In Da Vinci 3.0 (910C), per-core throughput improved ~30–35% through higher clock + microarchitectural changes

### 1.2 Vector Unit
- Performs element-wise operations on 128-element FP16 or 256-element INT8 vectors per cycle
- Supports: activation functions (ReLU, Sigmoid, Tanh), normalization, element-wise add/mul, type conversion, reduction
- Data types: FP32, FP16, BF16 (later gens), INT32, INT8, INT4
- Reads/writes the **Unified Buffer (UB)** directly

### 1.3 Scalar Unit
- Acts as a micro-CPU controller for the AI Core
- Handles loop control, conditional branches, address calculation, parameter setup
- Generates memory addresses and strides for MTE DMA engines
- Runs the **CCE (Core Compute Engine) scalar instruction stream**

### 1.4 MTE1 — Memory Transfer Engine (Input DMA)
- Moves data from **L2 → L1 (input feature buffer)** and **L1 → L0A/L0B**
- Supports tiling/transposition during transfer; hardware transpose for matrix layout conversion
- Enables double-buffering: while Cube processes tile N, MTE1 prefetches tile N+1

### 1.5 MTE2 — Memory Transfer Engine (Output DMA)
- Drains **L0C → Unified Buffer**, **UB → L2**, and **L2 → HBM**
- Also handles neuron activation, quantization, and format conversion during drain

All five units execute **concurrently** under static scheduling by the CCE compiler; there is no dynamic out-of-order issue. The compiler must explicitly pipeline Cube, Vector, and MTE operations to achieve peak utilization.

---

## 2. Memory Hierarchy

### Per-Core Local Memory (Da Vinci 1.0 / Ascend 910 reference)

| Level | Name | Size/Core | Bandwidth | Notes |
|-------|------|-----------|-----------|-------|
| L0A | Cube Input A Buffer | 16 KB | — | Dedicated to Cube unit left operand |
| L0B | Cube Input B Buffer | 16 KB | — | Dedicated to Cube unit right operand (weight tile) |
| L0C | Accumulator Buffer | 32 KB | — | Cube output; FP32 accumulation |
| L1  | Input Feature Buffer | 512 KB | ~512 GB/s | Staging for L0A/L0B; fed by MTE1 from L2 |
| UB  | Unified Buffer | ~256 KB | ~2 TB/s | Shared among Vector, MTE2, Scalar |

### Shared / Global On-chip Memory

| Level | Size | Bandwidth | Notes |
|-------|------|-----------|-------|
| L2 Cache | 32 MB total (shared across all cores) | 4 TB/s (NoC) | NoC mesh at 2 GHz, 1024-bit bus per core |
| Total SRAM | 84 MB (Ascend 910) | — | Includes all per-core buffers + shared L2 |

### Off-chip Memory (HBM)

| Generation | HBM Type | Capacity | Bandwidth | Notes |
|------------|----------|----------|-----------|-------|
| Ascend 910 (2019) | HBM2 | 32 GB | 1,228 GB/s | 4 HBM2 stacks; TSMC 7nm |
| Ascend 910B (2022) | HBM2e | 64 GB | ~800 GB/s | SMIC N+1 process; reduced BW vs 910 |
| Ascend 910C (2024) | HBM2e | 128 GB | ~1.6 TB/s (dual-die) | 2× 910B dies on organic interposer |
| Ascend 910D (planned) | HBM3/3e | TBD | TBD | SMIC advanced node, 2025+ |

Note: 910B has lower absolute bandwidth than 910 despite larger capacity, due to SMIC process constraints vs TSMC.

---

## 3. Core Count and Clustering

### Ascend 910 (Da Vinci 1.0)
- **32 Da Vinci Max AI Cores** arranged in 4 clusters of 8 cores
- Clusters interconnected via **NoC mesh** (1024-bit, 2 GHz, ~128 GB/s/core)
- 4 AI CPU cores (ARM-based, 1 KB L2 cache) for host-side control tasks
- Peak: 256 TFLOPS FP16, 512 TOPS INT8
- Power: 310 W (chip), 350 W (card)

### Ascend 910B (Da Vinci 2.0)
- **25 "New DaVinci" AI Cores** (reduced count vs 910, higher efficiency per core)
- Die: 21.32 × 31.22 mm, SMIC N+1 (~7nm equivalent)
- Peak: ~320 TFLOPS FP16, ~640 TOPS INT8
- HBM2e: 64 GB / ~800 GB/s

### Ascend 910C (Da Vinci 3.0)
- **Dual-die MCM**: 2× Ascend 910B dies on an organic substrate (not silicon interposer)
- ~32 effective Da Vinci 3.0 cores per die (microarchitecturally improved)
- Per-core throughput: +30–35% vs 910B via higher clock + IPC improvements
- Peak: ~800 TFLOPS FP16 (chip-level, ~1.6 PFLOPS INT8)
- Memory: 128 GB HBM2e, ~1.6 TB/s aggregate
- Package: CPU companion die (TSMC 7nm, vintage 2020) + 2× SMIC NPU dies
- TDP: ~400 W

---

## 4. Interconnect Architecture

### Intra-Chip: NoC
- 1024-bit mesh Network-on-Chip operating at 2 GHz
- Provides ~4 TB/s aggregate on-chip bandwidth for L2 access

### Scale-Up: HCCS (Huawei Compute Communication System)
- Proprietary high-speed intra-server NPU-to-NPU interconnect
- Designed as Huawei's analog to NVIDIA NVLink
- Atlas 800T server: 8× Ascend 910 interconnected via HCCS
- HCCS delivers higher bandwidth than PCIe switch-based configurations
- **HCCS 4.0** (planned with 910D): targeting 100,000-chip cluster scale

### CloudMatrix 384 (Ascend 910C Scale-Up/Out)
- 384 Ascend 910C chips across 16 racks (12 compute + 4 switch)
- Each compute server: **56 × 400G OSFP silicon photonic (SiPh) LPO** for scale-up
- Each compute server: **8 × 400G OSFP SiPh LPO** for scale-out
- Total: **6,912 × 400G** optical transceivers; full-mesh all-to-all topology
- Per-chip interconnect bandwidth: **2.8 Tbps** (scale-up)
- Total system bandwidth: 147.2 Tbps (384 chips × 384 Gbps effective)
- Exclusively optical links within and between racks (no copper backplane)

### Host Interface
- PCIe 4.0 × 16 (~64 GB/s bidirectional) for Atlas 300/800 cards

---

## 5. Precision and Data Type Support

| Precision | Cube Unit | Vector Unit | Notes |
|-----------|-----------|-------------|-------|
| FP16 | Yes (native) | Yes | Primary training format |
| INT8 | Yes (2× FP16 throughput) | Yes | Inference optimization |
| INT4 | Yes (gen-dependent) | Yes | Post-training quantization |
| FP32 | No (accumulation only via L0C) | Yes | Vector-only compute |
| BF16 | Yes (Da Vinci 2.0+) | Yes | Large model training |
| FP8 | Roadmap (910D/920) | — | Emerging format |

---

## 6. Key Design Tradeoffs vs Competitors

| Dimension | Ascend Da Vinci | NVIDIA Hopper | Google TPUv5 |
|-----------|----------------|--------------|-------------|
| Compute paradigm | 5-unit heterogeneous core | SIMT SM + Tensor Cores | Systolic MXU + VPU |
| Matrix tile size | 16×16 (FP16) | 16×8×16 (wgmma) | 128×128 (MXU) |
| On-chip SRAM | 84 MB total (910) | ~228 KB/SM × 132 = 30 MB SMEM | ~16–32 MB/TensorCore |
| HBM capacity | 32 GB (910) / 128 GB (910C) | 80–141 GB (H100/H200) | 96–192 GB (v5p/v7) |
| Scale-up interconnect | HCCS (proprietary) | NVLink 4 (900 GB/s) | ICI 3D torus (1.2 TB/s) |
| Compiler model | Static CCE scheduling | Dynamic SIMT + nvcc/Triton | Static XLA + Pallas |
| Software openness | Partially open (CANN, AscendC) | Open (CUDA ecosystem) | Open (JAX/XLA) |

---

## Ascend 950 / Next-Generation Roadmap

*Updated: 2026-04-05. Sources: Huawei HC 2025 keynote (September 2025), TrendForce, Tom's Hardware, Huawei Central, MWC 2026 announcements.*

### 7.1 Roadmap Overview

Huawei announced a formal three-year Ascend chip cadence at its 2025 Huawei Connect conference:

| Year | Chip | Variant(s) | Focus |
|------|------|-----------|-------|
| 2025 | Ascend 910C | (production ramp) | Training + inference, dual-die MCM |
| 2026 Q1 | Ascend 950PR | Prefill & Recommendation | High-throughput inference, cost-optimized HBM |
| 2026 Q4 | Ascend 950DT | Decode & Training | High-bandwidth HBM, full training workloads |
| 2027 Q4 | Ascend 960 | TBD | 2× 950 compute, memory, interconnect; HiF4 precision |
| 2028 | Ascend 970 | TBD | Targeting 4 ZettaFLOPS FP4 aggregate at cluster scale |

Note: "910D" references in early 2025 press (5nm, 4-die packaging, FP8 support) were superseded by the official 950-series branding announced at HC 2025.

### 7.2 Da Vinci 4.0 Architecture Changes (Ascend 950)

The 950 series introduces **Da Vinci 4.0** with a new SIMD+SIMT hybrid execution model — a departure from the pure SIMD vector unit in Da Vinci 1.0–3.0. Key architectural changes:

- **Precision formats expanded**: native FP8, MXFP8, HiF8 (Huawei-proprietary 8-bit float), MXFP4 — enabling 1 PFLOPS (FP8/HiF8) and 2 PFLOPS (MXFP4) per chip
- **Vector unit upgraded**: significantly enhanced SIMT-style vector throughput
- **Interconnect**: scale-up bandwidth increased 2.5× to **2 TB/s** (from ~800 GB/s intra-chip on 910C)
- **HBM replaced by in-house Huawei memory**: two new proprietary formats (see §7.3), circumventing export control restrictions on HBM3E from SK Hynix / Samsung / Micron

### 7.3 Ascend 950PR — Specifications

*Launched Q1 2026. Platform: Atlas 350 card and Atlas 950 SuperPoD.*

| Spec | Value |
|------|-------|
| Memory type | **HiBL 1.0** (Huawei in-house, low-cost HBM alternative) |
| Memory capacity | **128 GB** |
| Memory bandwidth | **1.6 TB/s** |
| FP8 / HiF8 compute | **1 PFLOPS** |
| MXFP4 compute | **2 PFLOPS** |
| Scale-up interconnect BW | **2 TB/s** (2.5× vs 910C) |
| Host card form factor | Atlas 350 (OAM-compatible) |
| Primary workloads | Inference prefill, recommendation systems |
| Competitor comparison | ~2.8× NVIDIA H20 FP4 performance (Atlas 350 card level) |

HiBL 1.0 is positioned as a cost-effective alternative to HBM3E — lower peak bandwidth than HiZQ 2.0, but sufficient for compute-bound prefill and recommendation workloads that are not memory-bandwidth-limited. This avoids dependence on foreign HBM supply chains constrained by US export controls.

### 7.4 Ascend 950DT — Specifications

*Planned Q4 2026. Targets training and autoregressive decode (memory-bandwidth-bound).*

| Spec | Value |
|------|-------|
| Memory type | **HiZQ 2.0** (Huawei in-house, high-performance HBM analog) |
| Memory capacity | **144 GB** |
| Memory bandwidth | **4 TB/s** |
| FP8 / HiF8 compute | **1 PFLOPS** |
| MXFP4 compute | **2 PFLOPS** |
| Scale-up interconnect BW | **2 TB/s** |
| Primary workloads | LLM training, autoregressive decode (memory-bandwidth-sensitive) |

HiZQ 2.0 achieves 4 TB/s memory bandwidth — roughly 2.5× the 910C dual-die bandwidth (~1.6 TB/s), closing the gap with HBM3E-class products. This is critical for decode-phase inference of large LLMs where memory bandwidth, not FLOPs, is the bottleneck.

### 7.5 Atlas 950 SuperPoD / CloudMatrix Evolution

Huawei's system-level architecture evolves with the 950 series:

| System | Chips | Scale-up BW/chip | Launch |
|--------|-------|-----------------|--------|
| CloudMatrix 384 (910C) | 384 Ascend 910C | 2.8 Tbps (optical SiPh) | 2024–2025 |
| Atlas 950 SuperPoD | 8,192 Ascend 950 | 2 TB/s (electrical + optical) | Q4 2026 |
| Atlas 960 SuperPoD | 15,488 Ascend 960 | TBD (2× 950) | Q4 2027 |

The Atlas 950 SuperPoD was demonstrated at MWC 2026 with **16 EFLOPS** peak FP4 performance across 8,192 Ascend 950 chips. By 2027, the Huawei roadmap targets **1 million-chip supernode clusters** (百万卡超节点集群).

### 7.6 Process Node and Manufacturing

Huawei has not officially disclosed the process node for Ascend 950. Industry analysis (as of early 2026):
- SMIC's advanced N+2 / N+3 node (estimated 5–7 nm equivalent) most likely
- The move to in-house HBM (HiBL 1.0, HiZQ 2.0) decouples memory supply from SK Hynix/Samsung/Micron export restrictions
- Production capacity target: Huawei planned ~1.6 million Ascend dies across all models in 2026

### 7.7 CANN SDK Updates for 950 Series

The 950 series requires CANN SDK updates to support new precision formats and the upgraded interconnect:
- **New data types**: FP8, MXFP8, HiF8, MXFP4 registered in AscendC type system and TBE operator library
- **2 TB/s interconnect**: HCCL collective primitives updated for higher-bandwidth intra-cluster topology
- **ATC / MindSpore GE**: quantization pass extended to emit HiF8/MXFP4 Cube instructions
- **Atlas 950 SuperPoD**: MindSpore auto-parallelism and HCCL topology-aware scheduling updated for 8,192-chip clusters
- **torch_npu**: FP8 autocast and MXFP4 quantization paths added in torch_npu 2.x release branch

### 7.8 Ascend 960 (2027) and 970 (2028) — Forward Look

- **Ascend 960 (Q4 2027)**: doubles compute, memory, and interconnect vs 950; introduces **HiF4** (Huawei-proprietary 4-bit float, claimed higher accuracy than standard FP4); Atlas 960 SuperPoD at 15,488 chips
- **Ascend 970 (2028)**: cluster-level target of **4 ZettaFLOPS FP4** (distributed across fleet); system-level milestone rather than single-chip spec

---

## Sources
- [DaVinci: A Scalable Architecture (CMC PDF)](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf)
- [DaVinci Architecture (Semantic Scholar)](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf)
- [Ascend AI Processor Architecture and Programming — O'Reilly Ch.3](https://www.oreilly.com/library/view/ascend-ai-processor/9780128234891/B9780128234884000035.xhtml)
- [Performance Modeling on DaVinci AI Core — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S074373152300014X)
- [Huawei Ascend 910B Examined — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)
- [Huawei Ascend 910C — Awesome Agents](https://awesomeagents.ai/hardware/huawei-ascend-910c/)
- [CloudMatrix 384 Analysis — SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72)
- [CloudMatrix 384 LPO Optics — QSFPTEK](https://www.qsfptek.com/qt-news/400g-osfp-siph-lpos-in-huawei-ai-cloudmatrix384-super-node.html)
- [TechInsights Teardown 910C — SemiWiki](https://semiwiki.com/forum/threads/techinsights-teardown-huawei-ascend-910c-still-contains-cpu-dies-from-tsmc-from-2020.23737/)
- [ServeTheHome Ascend 910 Review](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)

---

# Investigation Update — Ascend 950 Generation (2026-08-08)

*as_of: 2026-08-08*
*chip: huawei-ascend*
*focus: Ascend 950 NPU architecture whitepaper, Atlas 350 shipping card, UnifiedBus 2.0, Atlas 950 SuperPoD WAIC demo, 950DT schedule*
*scan window: 2026-04-05 (prior baseline) → 2026-08-08*

## U.1 What changed since the 2026-04-05 baseline

The Ascend 950 generation moved from a roadmap entry to shipping silicon with matching software. Four genuinely new artifacts landed in the window, and two pre-baseline items were missed at the time and are corrected here.

| Item | Date | Class |
|---|---|---|
| CANN 9.0.0 commercial + MindSpore 2.9.0 | 2026-05-07 / 2026-05-09 | new in window |
| Ascend 950 NPU architecture whitepaper (~38 pp) | late May / early Jun 2026 | new in window |
| Ascend 950DT Huawei Cloud deployment pulled forward to Aug 2026 | reported 2026-06-06/08 | new in window (announced schedule) |
| Atlas 950 SuperPoD physically demonstrated at WAIC 2026 | 2026-07-17 | new in window |
| Atlas 350 shipping-card spec (derated vs chip spec) | 2026-03-23/24 | **pre-baseline miss** |
| UnifiedBus / 灵衢 2.0 supersedes HCCS branding | 2025-09-18 | **pre-baseline miss** |

## U.2 Ascend 950 NPU architecture whitepaper

Huawei published *昇腾950 NPU架构白皮书* (~38 pp) at
`https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950 NPU架构白皮书.pdf`.

**Dating (three independent signals, all pointing to late May / early June 2026):**
- The OBS object returns `Last-Modified: Thu, 04 Jun 2026 06:54:10 GMT`
- Chinese trade press covered its release on 2026-06-11
- An independent analyst digest of its contents is dated 2026-05-22

An earlier internal note that placed the whitepaper in July 2026 is **wrong** and has been corrected.

**Extraction status:** the PDF is **image-only** (6.1 MB, no text layer). Its contents therefore reach this survey only via third-party readings. This is the single largest evidence gap for the 950 generation and the highest-value next research step.

### U.2.1 Independently corroborated microarchitecture

- **AIC / AIV core separation.** The 950 splits the Da Vinci core into an independent Cube core (AIC) and Vector core (AIV), at **1 Cube + 2 Vector per AI subsystem**. This is a genuine break from the 910-era five-units-in-one-core design and is the most consequential architectural change in the generation.
- **128 MB global L2** spanning the chip's two AI dies — a memory level with no equivalent on 910B/910C. Roughly 2× per-access improvement; supports per-way cache lock / residency policy. L2 **bandwidth is not disclosed**.
- **STARS 2.0** hardware scheduler arbitrating across AIC / AIV / CPU / DVPP / SDMA / UB / CCU.
- **NDDMA** as a lightweight AIC↔AIV on-chip data path.
- **PCIe 5.0 × 16** host interface (up from PCIe 4.0 on 910C); **dual 400 Gbps UBoE**.
- ~**2 TB/s** inter-chip bandwidth over HiLink SerDes.
- Memory unchanged from roadmap: 128 GB / 1.6 TB/s (950PR), 144 GB / 4 TB/s (950DT).

### U.2.2 Naming discrepancy

The whitepaper describes the architecture as **3rd-generation Da Vinci**; press coverage and this repo call it **Da Vinci 4.0**. Same silicon, two numbering conventions. Huawei has published no reconciliation. The repo retains "Da Vinci 4.0" for continuity and flags the discrepancy in `chips/huawei-ascend/hw-architecture.md` §8.

### U.2.3 Single-source, NOT independently confirmed

All of the following trace to a single third-party whitepaper analysis. They are recorded for future verification and **must not be cited as spec**:

- 18 Da Vinci cores per AI die → 36 Cube + 72 Vector cores per chip (950DT)
- L0A / L0B at 64 KB each, L1 at 512 KB, Unified Buffer at 512 KB
- Sector cache with 512 B lines split into 4 × 128 B sectors
- BufferID-based synchronization replacing the `EnQue` / `DeQue` inter-pipeline handshake
- The fuller NDDMA description (N-Dimensional Data Movement Accelerator fusing movement + layout conversion + address generation into single instructions, up to 5D transforms)
- A "Linx816" on-chip AI CPU with 4 clusters × 2 ARMv8-A cores, dual-threaded, 4 MB L3 per cluster
- UnifiedBus RTP/CTP decomposition (see U.4)
- The precise per-precision compute ladder (see U.2.4)

Note: the repo's existing 16 KB L0A/L0B figure is the **910-era value** and remains correct for that generation; it is not "wrong" — it simply was never a 950 figure.

### U.2.4 Compute ladder — what is disclosed vs estimated

Huawei discloses only the rounded **1 PFLOPS FP8 / 2 PFLOPS MXFP4** per chip, for both 950PR and 950DT. Huawei's own WAIC arithmetic corroborates it: 1,024 cards → 1 EFLOPS FP8.

A single third-party analysis additionally gives MXFP4 2,007 / HiF8 ~1,034 / FP8 1,034 / BF16-FP16 547 / TF32 ~273 TFLOPS. **BF16/FP16 and TF32 peaks are not disclosed by Huawei** and are recorded as single-source estimates only.

## U.3 Atlas 350 shipping card vs Ascend 950PR chip spec (pre-baseline correction)

Announced 2026-03-23/24 by Huawei Ascend computing president **Zhang Dixuan** — i.e. before the 2026-04-05 baseline, and missed at the time.

| Figure | Ascend 950PR chip spec | Atlas 350 shipping card |
|---|---|---|
| FP4 compute | 2 PFLOPS MXFP4 | **1.56 PFLOPS FP4** |
| Memory | 128 GB HiBL 1.0 | **up to 112 GB** |
| Bandwidth | 1.6 TB/s | **1.4 TB/s** |
| TDP | not previously recorded | **600 W** (~1.5× H20) |

The "**2.8× H20**" headline is a **Huawei marketing comparison**, not an independently confirmed benchmark. Corroborated by TrendForce (2026-03-23) and Tom's Hardware (2026-03-24).

## U.4 UnifiedBus / 灵衢 (Lingqu) 2.0 (pre-baseline correction)

The 950-generation scale-up fabric is **UnifiedBus**, launched by Huawei rotating chairman **Xu Zhijun on 2025-09-18 at Huawei Connect 2025** — roughly seven months before the repo baseline. Huawei published it as an **open specification** (base spec, firmware spec, software reference designs) at `unifiedbus.com`, with an openEuler *UB Service Core* software architecture reference design.

- Confirmed: ~2 TB/s per chip over HiLink SerDes; dual 400 Gbps UBoE; whitepaper claims cluster scale **beyond 128K cards**; 3 μs RTT measured in the WAIC 2026 demonstration.
- **Single-source, not confirmed:** 2,016 GB/s bidirectional per chip split into RTP (reliable transport, 4 ports, 448 GB/s dual) and CTP (light transport, 9 ports, 1,008 GB/s dual) over 18 × X4 HiLink ports at 112 Gbps/lane.
- This supersedes the repo's "HCCS 4.0 targets 100,000-chip cluster scale" line.

## U.5 Atlas 950 SuperPoD — WAIC 2026 demonstration

WAIC 2026, Shanghai, **2026-07-17 to 07-20**. Huawei's release of 2026-07-17 labels the showing 真机首次公开亮相 ("first public physical appearance"); the system won the conference SAIL award.

| Demonstrated configuration | Value |
|---|---|
| Cards | 1,024 |
| Compute | 1 EFLOPS FP8 / 2 EFLOPS FP4 |
| Globally unified memory address space | 256 TB |
| Fabric RTT | 3 μs over the 灵衢 protocol |
| Status | **DEMONSTRATED, not shipping** — Q4 2026 commercial delivery |

Also in that release: **Atlas 850E** (air-cooled variant) at 96-card commercial deployment; **750+ commercial deployments** of the older Atlas 384-series supernodes.

The full-scale target — **8,192 Ascend 950DT cards, 16 EFLOPS FP4** — is unchanged from MWC 2026. MWC 2026 material gives the packaging rule (**64 NPUs per cabinet**, scaling to 8,192) and describes Atlas 850E scaling 8 → 1,024 NPUs for conventional datacenters.

Note the correction: the repo previously recorded the 8,192-chip Atlas 950 SuperPoD as built from **950PR**; the verified source attributes it to **950DT**.

## U.6 Ascend 950DT schedule

Huawei VP **Chen Lin**, at the Huawei Cloud 2026 INSPIRE Creators event (reported 2026-06-06/08), said 950DT deployment on Huawei Cloud would be pulled forward from Q4 2026 to **August 2026**.

- **Status verb matters:** this is an *announced schedule*, not an accomplished deployment. As of 2026-08-08 no confirmation was found that the August deployment went live.
- **Q4 2026 remains the official commercial release date.**
- Chen Lin's own description was qualitative: "greatly improve vector compute, memory bandwidth, natively support FP8" — no spec disclosure.
- Restated specs: 144 GB, 4 TB/s, FP8/MXFP8/MXFP4/HiF8.
- **Process node: not disclosed.** "SMIC N+3" is analyst inference (low confidence).
- DeepSeek as expected early adopter (DeepSeek V4 already on the Ascend 950 platform) is press reporting, not a Huawei statement.

## U.7 Negative and non-findings

- **Hot Chips 38 runs 2026-08-23 to 08-25 — 15 days after this scan.** Any absence of Ascend content there carries **zero** evidentiary weight and is not recorded as a finding.
- MLPerf Inference v6.0 results published 2026-04-01 (24 submitting organizations, itself pre-baseline). The submitter list could not be enumerated, so "no Huawei Ascend submission" is **plausible but unverified**, not a confirmed negative.
- No evidence of an **Ascend 960 / 970** schedule change since baseline.
- "**910D**" continues to appear only in older press; nothing retrieved contradicts the existing note that it was superseded by 950-series branding.

## U.8 Methodology caveat

The scan that produced this update exhausted its WebSearch budget before starting; discovery ran through WebFetch against DuckDuckGo HTML/Lite result pages plus direct primary-source fetches. Coverage of low-visibility items (conference programs, auth-gated SDK changelogs) is thinner than a normal scan. `hiascend.com/en/software/cann` and `hotchips.org/program` (HTTP 403) yielded no content.

## U.9 Sources added 2026-08-08

- Ascend 950 NPU architecture whitepaper (PDF, image-only): https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf
- Third-party whitepaper analysis (single-source items above): https://pillumina.github.io/posts/aiinfra/ascend-950-npu/
- Atlas 950 SuperPoD at WAIC 2026 (Huawei, 2026-07-17): https://www.huawei.com/cn/news/2026/7/atlas-950-superpod
- SuperPoD announcements at MWC 2026 (Huawei): https://www.huawei.com/en/news/2026/3/mwc-superpod-ai
- UnifiedBus open specification: https://www.unifiedbus.com/en
- openEuler UB Service Core SW Arch Reference Design: https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf
- UnifiedBus community mirror: https://github.com/codehubcloud/UnifiedBus
- Ascend 950DT pulled forward (TrendForce, 2026-06-08): https://www.trendforce.com/news/2026/06/08/news-huawei-brings-forward-ascend-950dt-deployment-to-august-deepseek-v4-2-seen-as-potential-early-adopter/
- Ascend 950DT August debut (Huawei Central): https://www.huaweicentral.com/huawei-confirms-ascend-950dt-ai-chip-to-debut-in-august/
- Huawei Cloud agentic infra / 950DT window (WinBuzzer, 2026-06-10): https://winbuzzer.com/2026/06/10/huawei-cloud-ties-agentic-infra-to-ascend-950dt-window-xcxwbn/
- Atlas 350 debut (TrendForce, 2026-03-23): https://www.trendforce.com/news/2026/03/23/news-huawei-debuts-atlas-350-on-ascend-950pr-with-in-house-hbm-touting-2-8x-h20-performance/
- Atlas 350 unveiled (Tom's Hardware, 2026-03-24): https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20
- Ascend 950PR / Atlas 350 FP4 coverage: https://intelligentliving.co/huawei-ascend-950pr-atlas-350-fp4-ai/
