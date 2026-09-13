# Cambricon MLU Hardware Architecture

*as_of: 2026-08-08*
*chip: cambricon*
*device_class: Neural Processor*
*Representative products: MLU290 (training), MLU370 / Siyuan 370 (training+inference — the only vendor-documented datacenter part), MLU590 (LLM; specs TBD), MLU690 (reported only, unverified)*

---

## Overview

Cambricon's Machine Learning Unit (MLU) is a **load-store neural processor** built around a hierarchical structure of MLU Cores and Clusters. Unlike GPU SIMT architectures that use hardware caches and warp-level threading, the MLU exposes a software-managed scratchpad memory hierarchy (NRAM, WRAM, shared SRAM) directly to the programmer/compiler. This design choice delivers high determinism and efficiency for regular DNN workloads at the cost of requiring explicit data movement programming.

The core hierarchy is: **MLU Core** (atomic compute unit) → **Cluster** (4 Cores + Memory Core + Shared SRAM) → **Die** (multiple Clusters). This maps directly to the memory hierarchy: per-core NRAM/WRAM → per-cluster Shared SRAM → off-chip GDRAM.

**Disclosure note (2026-08-08).** Cambricon's 2026 half-year report names only 思元100/220/270/290/370 and contains zero occurrences of 思元590, 思元690 or HBM. Everything below about MLU590 and MLU690 is therefore either marked TBD or explicitly flagged as an unverified third-party report. No Cambricon datasheet exists for either part, and the Cambricon product pages were unreachable (HTTP 500) during the 2026-08-08 scan.

---

## 1. Compute Engine

### MLU Core (IPU Core)

The fundamental compute building block. Each MLU Core contains:

| Component | Description |
|-----------|-------------|
| Functional Unit (FU) | Executes scalar, vector, matrix, and tensor operations |
| GPRs | 64 × 32-bit general purpose registers (scalar/control/addressing) |
| NRAM (Neural-RAM) | Private scratchpad for input/output activation data |
| WRAM (Weight-RAM) | Private scratchpad for convolution kernel weights |

Unlike a GPU SM (which contains hundreds of CUDA cores and multiple warp schedulers), an MLU Core is a single execution pipeline. Parallelism comes from running many Cores simultaneously across clusters, with each Core executing the same BANG C kernel on its private data partition (SPMD model).

### Cluster Organization

Four MLU Cores form a **Cluster**, together with:
- A **Memory Core**: dedicated DMA engine managing data movement between GDRAM and per-core scratchpads; runs independently of the 4 compute Cores to enable DMA/compute overlap
- A **Shared SRAM**: accessible by all 4 Cores + Memory Core; staging buffer and inter-core data exchange without GDRAM round-trips

### Product Specifications

| Product | Architecture | Clusters | MLU Cores | Peak INT8 | Peak FP16 | Process |
|---------|-------------|----------|-----------|-----------|-----------|---------|
| MLU220 | MLUarch01 | 1 | 4 | — | — | TSMC 16nm |
| MLU270 | MLUarch02 | 4 | 16 | — | — | TSMC 16nm |
| MLU290-M5 | — | 8 | 32 | 512 TOPS | 256 TFLOPS | TSMC 7nm |
| MLU370 / Siyuan 370 | MLUarch03 | ~8+ | ~32+ | ~256 TOPS | ~96 TFLOPS | TSMC 7nm |
| MLU590 / Siyuan 590 | not disclosed | ~16+ | ~64+ | TBD | TBD | TSMC 7nm+ |
| **MLU690 / Siyuan 690** *(reported, unverified)* | not disclosed | not disclosed | not disclosed | **>2,800 TOPS INT4** *(reported)* | **>700 TFLOPS** *(reported)* | **not disclosed** — sources contradict (TSMC 4nm vs 7nm) |

*MLU370 cluster/core counts are inferred; Cambricon has not fully disclosed MLU370 microarchitecture details.*

*MLU690 row: no Cambricon primary source exists. The FP16/INT4 figures come from a Sina Finance brokerage report and a 51CTO blog respectively; the package is described as dual-die/chiplet. Sources disagree on the process node (TSMC 4nm in one set, 7nm in another), so **no node is published here**. No `MLUarch04`/`MLUarch05` architecture name is confirmed anywhere. Production status is contested — see §8.*

### Precision Support (MLUarch03 / Siyuan 370)

FP32, FP16, BF16, INT16, INT8, INT4.

Precision support for MLU590 and MLU690 is **not disclosed**. The reported MLU690 INT4 figure implies an INT4 path, but no format list has been published by Cambricon.

---

## 2. Data Path

### SPMD Execution Model

The MLU uses **SPMD (Single Program Multiple Data)**. All MLU Cores within a task execution unit execute the same BANG C kernel in parallel, each processing its own private NRAM partition. Key contrasts with GPU SIMT:

| Aspect | NVIDIA GPU (SIMT) | Cambricon MLU (SPMD) |
|--------|------------------|----------------------|
| Threading unit | 32-thread warps | Individual MLU Cores |
| Divergence handling | Warp predication | Not applicable (core-level) |
| Memory model | Cache-based (L1/L2/HBM) | Scratchpad-based (NRAM/WRAM/SRAM) |
| Synchronization | `__syncthreads()`, barriers | `__sync_cluster()`, `__sync_all()` |
| Data movement | Hardware-managed caches | Explicit programmer DMA calls |

### Data Movement Hierarchy

```
GDRAM (off-chip LPDDR5 or HBM)
  │
  └─ Memory Core DMA (async, per-cluster)
        │
        ├─ Shared SRAM (per-cluster, staging)
        │     │
        │     └─ Direct load/store by 4 Cores
        │
        └─ NRAM / WRAM (per-core, via DMA)
              │
              └─ Functional Unit (compute)
```

The standard programming pattern is **double-buffering**: while 4 cores compute on tile N in NRAM, the Memory Core prefetches tile N+1 into Shared SRAM or the next NRAM pingpong buffer. Coordination uses barrier instructions (`__sync_cluster()`).

### Instruction Set Architecture (MLISA)

MLISA is a 64-bit, load-store ISA:

| Category | Operations |
|----------|-----------|
| Scalar | Control flow, address arithmetic, GPR operations |
| Vector | Element-wise ops (add, mul, activation, etc.) on NRAM data |
| Matrix | GEMM, Conv — NRAM operand data + WRAM weights |
| Control | Branch, barrier, DMA sync, memory fence |

All instructions are 64 bits. No vector register file — operands for vector/matrix instructions reside in NRAM/WRAM (not registers). This "scratchpad-as-register" model allows wide operands without a large register file.

---

## 3. On-chip Memory

### Memory Hierarchy Overview

| Level | Name | Scope | Managed By | Primary Use |
|-------|------|-------|-----------|-------------|
| 1st | NRAM (Neural-RAM) | Per MLU Core | Programmer | Activation data, vector/tensor operands |
| 1st | WRAM (Weight-RAM) | Per MLU Core | Programmer | Convolution kernel weights |
| 2nd | Shared SRAM | Per Cluster (4 Cores) | Programmer | Data staging, inter-core exchange |
| 3rd | LLC | Chip-level | Hardware (read-only) | Shared read-only data (weight broadcast) |

### NRAM

- Per-core private scratchpad; no hardware cache backing
- Stores: input/output activations for vector/tensor ops; temporary scalar intermediates
- Programmer allocates regions with `__nram__` qualifier in BANG C
- Typical access pattern: DMA fill → compute → DMA writeback (or via Shared SRAM)
- Size range (published examples): 512 KB–2 MB per core (exact MLU370 size not disclosed)

### WRAM

- Per-core private scratchpad dedicated to convolution weights
- Feeds directly into the matrix/conv functional unit (weight-stationary pattern)
- Loaded once per kernel invocation, reused across multiple activation tiles
- Smaller than NRAM; sized for typical filter footprints

### Shared SRAM (per-Cluster)

- Shared across all 4 MLU Cores + Memory Core within a cluster
- Two roles:
  1. **Staging buffer**: GDRAM → Shared SRAM → individual Core NRAMs
  2. **Inter-core communication**: Core A writes result to Shared SRAM; Core B reads it without GDRAM round-trip
- Accessed via `__mlu_shared__` qualifier in BANG C

### LLC (Last Level Cache)

- Hardware-managed read-only cache on the GDRAM access path
- Primarily used to broadcast shared weight tensors to multiple cores without GDRAM bandwidth saturation
- Not part of the primary programmer-visible memory model (distinct from NRAM/WRAM/Shared SRAM)

---

## 4. Off-chip Memory

### Memory Configurations by Product

| Product | Memory Type | Capacity | Bandwidth | Notes |
|---------|------------|----------|-----------|-------|
| MLU290-M5 | HBM2 | 32 GB | 1,228 GB/s | Training card, high BW |
| MLU370-X8 (dual-chip) | LPDDR5 | 48 GB | 614.4 GB/s | China's first cloud AI chip with LPDDR5 |
| MLU370-M8 | LPDDR5 | 24 GB | ~307 GB/s | Single-chip server card |
| MLU590 | HBM (TBD) | TBD | TBD | Supply-constrained HBM; no capacity/BW figure disclosed |
| **MLU690** *(reported, unverified)* | **HBM3** *(reported)* | **196 GB** *(reported)* | **~3.35 TB/s** *(reported)* | No Cambricon datasheet; HBM is unmentioned in the 2026 H1 filing |

### LPDDR5 vs HBM Tradeoff

MLU370's use of LPDDR5 (instead of HBM) reflects supply constraints: US export controls restrict HBM access for Chinese chip companies. LPDDR5 offers lower bandwidth than HBM but is available from multiple DRAM vendors without export restrictions. Cambricon describes the MLU370's LPDDR5 bandwidth as 3× the previous generation (MLU270).

MLU590 targets HBM but faces production bottlenecks from the same supply constraints; Cambricon has disclosed neither its HBM generation nor its capacity or bandwidth.

**Correction (2026-08-08) to a previously recorded statement.** An earlier revision of this file attributed to Cambricon a claim that HBM availability was "a key production limiter for the MLU590 ramp to 500,000 units in 2026." That attribution was wrong on two counts. The 500,000-unit figure is a **third-party supply-chain target for Cambricon's total 2026 accelerator output** (originating in Bloomberg reporting dated 2025-12-04, echoed by Tom's Hardware and TrendForce), not a Cambricon statement and not specific to the MLU590. The named risks to that target in the same reporting are **low yield, SMIC 7nm-class capacity contention with Huawei, and HBM supply**. Domestic HBM3 from CXMT is targeted for end-2026, i.e. after the stated ramp. The reported MLU690 HBM3 configuration above, if real, sits squarely inside that HBM supply constraint.

---

## 5. Host Interface / Package

### PCIe Interface

| Product | PCIe | Bandwidth | Form Factor |
|---------|------|-----------|-------------|
| MLU370-M8 | Gen4 x16 | ~64 GB/s bidir | FHFL server card |
| MLU370-X | Gen4 x16 | ~64 GB/s bidir | FHFL intelligent accelerating card |
| MLU290-M5 | Gen4 x16 | ~64 GB/s bidir | FHFL server card |
| MLU-X1001 | Gen4 x16 + Mini SAS HD | ~64 GB/s bidir | External accelerator module |

No PCIe Gen5 or coherent CPU-GPU link (analogous to NVLink-C2C) has been disclosed.

### Chiplet Packaging (Research Direction)

The **Cambricon-LLM** paper (arXiv 2409.15654) describes a future chiplet-based architecture targeting on-device 70B LLM inference. The design combines specialized compute chiplets with memory chiplets in a 2.5D integration. This is a published research direction, not a shipping product as of 2026-08-08.

Separately, the reported MLU690 (§8) is described by secondary Chinese sources as a **dual-die / chiplet package**. That description is unverified; no Cambricon packaging disclosure exists for it. The MLU370-X8's two-dies-on-one-card arrangement (§6) is a board-level MLU-Link bridge, not a 2.5D chiplet package, and should not be conflated with either.

---

## 6. Scale-up Interconnect (MLU-Link)

**MLU-Link** is Cambricon's proprietary high-bandwidth chip-to-chip interconnect, analogous to NVIDIA NVLink.

| Spec | MLU370-X8 | MLU690 *(reported, unverified)* |
|------|-----------|-------------------------------|
| Dies per card | 2 (Siyuan 370 × 2) | Dual-die / chiplet package (reported) |
| Shared memory | 48 GB LPDDR5 (unified) | 196 GB HBM3 (reported) |
| Interconnect | MLU-Link (BW not publicly disclosed) | **>890 Gbps (≈111 GB/s)** reported |
| Max cards (scale-up) | 8 (server-level backplane) | not disclosed |

**Units warning on the MLU690 interconnect figure.** Every Chinese source retrieved states 超890**Gbps** — 890 *gigabits* per second, roughly 111 GB/s. A "890 GB/s" restatement circulates widely and is wrong by a factor of ~8. The figure recorded here is the gigabit one, and it is in any case unverified.

The MLU370-X8 presents two Siyuan 370 dies as a single unified device to the host. At the server level, up to 8 MLU-Link-capable cards can be connected for multi-chip training. Cambricon reported 8-card MLU-Link configurations achieving ~155% of a 350W RTX GPU's throughput on BERT/YOLO/ResNet benchmarks (2022, Cambricon internal).

No external MLU-Link switch ASIC (comparable to NVIDIA NVSwitch) has been publicly announced.

---

## 7. Scale-out Interconnect

| Component | Description |
|-----------|-------------|
| Standard Ethernet / InfiniBand | Inter-node communication via commodity or standard HPC NICs |
| CNCL (intra-node) | Cambricon Collective Comms Library over MLU-Link for AllReduce/AllGather within a node |
| Gloo (heterogeneous) | CPU-mediated collectives for cross-vendor (MLU + GPU) mixed clusters |

No proprietary scale-out NIC or network ASIC (analogous to NVIDIA ConnectX or Google Titanium) has been disclosed. Cambricon relies on standard networking for multi-node clusters.

---

## 8. MLU690 / Siyuan 690 — Reported Status (added 2026-08-08)

This section exists to keep the MLU690 out of the confirmed-architecture sections above while still recording what is publicly circulating, because the part is widely discussed and the numbers are widely mis-stated.

**Evidence status: LOW confidence, contested.** There is no Cambricon datasheet, no Cambricon product page (pages returned HTTP 500 during the scan), and no mention of 思元690 in Cambricon's 2026 half-year report. The spec figures trace to Sina Finance brokerage reports, 51CTO blog posts, Zhihu posts and Tianyancha — i.e. secondary aggregation, not vendor documentation.

**Production status is genuinely contested.** Secondary coverage of the 2026 H1 report asserts the 690 reached mass production in early 2026. Other coverage in the same period describes it as still in final testing. TrendForce in December 2025 placed it in the testing phase with large-scale production potentially slipping to H2 2026. The correct record is: *reported by secondary Chinese media as entering production in early 2026; not confirmed by any Cambricon primary source.*

| Attribute | Reported value | Provenance |
|-----------|----------------|-----------|
| Package | Dual-die / chiplet | Secondary (51CTO) |
| Peak FP16 | >700 TFLOPS | Brokerage (Sina Finance) |
| Peak INT4 | >2,800 TOPS | Secondary (51CTO) |
| Memory | 196 GB HBM3 | Brokerage + secondary |
| Memory bandwidth | ~3.35 TB/s | Secondary (51CTO) |
| Interconnect | >890 Gbps (≈111 GB/s) | Brokerage — note bits, not bytes |
| Process node | **not disclosed** | Contradictory: TSMC 4nm (one set) vs 7nm (another, which also claims >50B transistors and 2,000 TOPS FP8) |
| Transistor count | **not disclosed** | Only the contradicting 7nm source gives a figure |
| Cluster / core count | **not disclosed** | — |
| Precision format list | **not disclosed** | — |

Nothing in this table should be propagated into the confirmed comparison tables without a vendor source.

---

## Sources

- [Machine Learning Unit (MLU) — WikiChip](https://en.wikichip.org/wiki/cambricon/mlu)
- [MLU370-M8 Product Manual (FCC)](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf)
- [BANG C Language Developer Guide v2.4.1](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf)
- [Cambricon ISA — IEEE ISCA 2016](https://ieeexplore.ieee.org/document/7551409/)
- [Cambricon ISA Extended — ACM 2019](https://dl.acm.org/doi/fullHtml/10.1145/3331469)
- [Cambricon-LLM Chiplet Architecture (arXiv 2409.15654)](https://arxiv.org/html/2409.15654v1)
- [KAITIAN Communication Framework (arXiv 2505.10183)](https://arxiv.org/html/2505.10183v1)
- [Optimizing Logcumsumexp on Cambricon MLU (ResearchGate)](https://www.researchgate.net/publication/393382745_Optimizing_Logcumsumexp_on_Cambricon_MLU_Architecture-Aware_Scheduling_and_Memory_Management)
- [Cambricon targets 500,000 AI chips in 2026 (Tom's Hardware)](https://www.tomshardware.com/tech-industry/semiconductors/cambricon-targets-500000-ai-chips-in-2026-as-china-accelerates-domestic-hardware-push) — third-party target; downstream of Bloomberg supply-chain reporting dated 2025-12-04
- [Cambricon 2026 半年度报告 (cninfo, 2026-08-08)](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF) — PRIMARY; names only 思元100/220/270/290/370; zero mentions of 思元590/690 or HBM
- [TrendForce: Cambricon remains China's top AI chip startup (2025-12-15)](https://www.trendforce.com/news/2025/12/15/insights-cambricon-remains-chinas-top-ai-chip-startup-rumored-2026-triple-output-faces-smic-limits/) — 690 in testing phase; SMIC 7nm capacity contention
- [Sina Finance brokerage report on Siyuan 690](https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/831133396021/index.phtml) — secondary/unverified: FP16 >700 TFLOPS, 196 GB HBM3, 互连带宽超890Gbps
- [51CTO blog on Siyuan 690](https://blog.51cto.com/u_10819805/14632220) — secondary/unverified: dual-die chiplet, 196 GB HBM3, 3.35 TB/s, INT4 2,800+ TOPS
- [Tianyancha news item on Siyuan 690](https://news.tianyancha.com/ll_5k94fbk79z.html) — secondary, CONFLICTING node claim: TSMC 7nm, >50B transistors, FP8 2,000 TOPS
