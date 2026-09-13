# Cambricon MLU Hardware Architecture — Investigation Report

*resource: https://en.wikichip.org/wiki/cambricon/mlu + https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf + https://arxiv.org/html/2409.15654v1*
*as_of: 2026-04-05*
*chip: cambricon*
*layer: Compute Engine, Data Path, On-chip Memory, Off-chip Memory, Host Interface / Package, Scale-up Interconnect, Scale-out Interconnect*

---

## Overview

Cambricon's Machine Learning Unit (MLU) is a family of neural processors designed for datacenter AI training and inference. The architecture is organized around a hierarchical compute structure: individual **MLU Cores** (also called IPU Cores) group into **Clusters**, which collectively form the full die. This hierarchy directly mirrors the memory subsystem: private per-core scratchpads (NRAM, WRAM) at the bottom, per-cluster shared SRAM in the middle, and off-chip GDRAM (LPDDR5 or HBM) at the top.

The MLU is a **load-store accelerator** — there is no hardware cache in the traditional sense for the primary compute data path. Instead, the BANG C programming model exposes explicit on-chip scratchpad memories (NRAM, WRAM, shared SRAM) that the programmer or compiler manages. This design trades hardware generality for peak efficiency on regular, predictable DNN workloads.

Key product generations:
- **MLU100** (2018): First-gen PCIe accelerator card
- **MLU220** (2019): Edge/embedded, 1 cluster / 4 cores
- **MLU270** (2019): Server inference, 4 clusters / 16 cores
- **MLU290** (2020): Training, 32 GB HBM2 @ 1,228 GB/s
- **MLU370 / Siyuan 370** (2022): MLUarch03, LPDDR5, training+inference, MLU-Link dual-chip card
- **MLU590 / Siyuan 590** (2024+): HBM-based, higher bandwidth, targeting LLM training

---

## 1. Compute Engine

### MLU Core (IPU Core)

The fundamental compute building block. Each MLU Core contains:

| Component | Description |
|-----------|-------------|
| Functional Unit (FU) | Executes scalar, vector, matrix, and tensor operations |
| General Purpose Registers (GPRs) | 64 × 32-bit registers; scalar/control/addressing use |
| NRAM (Neural-RAM) | Private scratchpad for input/output operand data |
| WRAM (Weight-RAM) | Private scratchpad for convolution kernel weights |

The Cambricon ISA (ISCA 2016) defines 43 instruction types, all 64-bit, in four categories:
- **Scalar**: Control flow, address computation, GPR operations
- **Vector**: Element-wise operations on NRAM vectors
- **Matrix**: GEMM-class operations using NRAM+WRAM
- **Control**: Branches, barriers, DMA synchronization

Unlike GPU SIMT, the MLU Core is not warp-based. Each Core executes its instruction stream independently; parallelism comes from running many Cores simultaneously across clusters.

### Cluster Organization

Four MLU Cores form a **Cluster**, together with:
- A **Memory Core**: dedicated DMA engine managing data movement between GDRAM and per-core memories
- A **Shared SRAM**: accessible by all 4 Cores + the Memory Core within the cluster; used for inter-core data sharing without GDRAM round-trips

**Historical cluster/core counts by product:**

| Product | Clusters | MLU Cores (IPU Cores) | Process |
|---------|----------|----------------------|---------|
| MLU220 | 1 | 4 | TSMC 16nm |
| MLU270 | 4 | 16 | TSMC 16nm |
| MLU290 | 8 | 32 | TSMC 7nm |
| MLU370 (Siyuan 370) | 8+ | 32+ | TSMC 7nm |
| MLU590 (Siyuan 590) | ~16+ | ~64+ | TSMC 7nm or N+1 |

*Exact cluster/core counts for MLU370+ are inferred from published die sizes and published TOPS numbers; Cambricon has not fully disclosed the MLU370 cluster count.*

### Precision Support

MLUarch03 (Siyuan 370) supports: FP32, FP16, BF16, INT16, INT8, INT4.

MLU290 peak: 512 INT8 TOPS, 256 INT16 TOPS, 64 FP32 TFLOPS.
MLU370 peak: ~256 INT8 TOPS, ~96 FP16 TFLOPS (single-chip estimate).

---

## 2. Data Path

### Execution Model

The MLU uses a **SPMD (Single Program Multiple Data)** model at the cluster level. All MLU Cores within a task execute the same BANG C kernel code in parallel, each operating on its private NRAM/WRAM partition. This is similar to GPU SIMT but at the granularity of cores rather than threads; each Core has its own independent instruction fetch and execute path.

Within a single Core, there is no out-of-order execution or warp-level threading. The programmer writes explicit synchronization barriers (`__sync_cluster()`, `__sync_all()`) to coordinate across cores.

### Data Movement Pipeline

Data movement follows a fixed hierarchy:

```
GDRAM (off-chip)
  |-- DMA via Memory Core ---> Shared SRAM (per-cluster)
                                    |-- DMA / direct load ---> NRAM (per-core)
                                                               WRAM (per-core)
                                                                   |
                                                               FU compute
```

The **Memory Core** runs independently of the compute Cores, enabling overlap of DMA transfers with computation (double-buffering pattern). The programmer coordinates via barriers.

### Instruction Set Architecture (BANG / MLISA)

The hardware ISA is called **MLISA** (Machine Learning ISA). User-visible programming is via **BANG C**, which compiles (through CNCC) to MLISA assembly, then CNAS assembles to `.cncode`/`.cnbin`/`.cnfatbin` binary formats.

MLISA characteristics:
- All instructions are 64-bit
- 64 × 32-bit GPRs per Core (scalar/address)
- Vector instructions operate on NRAM regions; widths parameterized by data type
- Matrix instructions (GEMM, conv) use NRAM operands and WRAM weight operands
- Control: branch, barrier (`sync`), memory fence

---

## 3. On-chip Memory

### NRAM (Neural-RAM)

- Per-MLU-Core private scratchpad
- Stores vector/tensor operand data (inputs and outputs of vector/matrix ops)
- Also holds temporary scalar intermediates
- Size: varies by generation; BANG C guide implies 512 KB–2 MB per core range (not publicly confirmed for MLU370+)
- Accessed by the FU directly; no cache miss semantics — all accesses are programmer-controlled

### WRAM (Weight-RAM)

- Per-MLU-Core private scratchpad
- Dedicated to storing convolution kernel weights (filters)
- Feeds directly into the matrix/conv functional unit
- Keeping weights in WRAM avoids repeated GDRAM accesses for weight-stationary operations
- Size: smaller than NRAM; tuned for typical conv filter footprint

### Shared SRAM (per-Cluster)

- Shared across all 4 Cores + Memory Core in a cluster
- Acts as a staging buffer for data movement between GDRAM and per-core NRAMs
- Enables inter-core data exchange without GDRAM round-trips
- Also used as LLC (Last Level Cache) for read-only shared data in some configurations

### LLC / L1 Cache

A small L1 cache exists on the path from GDRAM through LDRAM; primarily buffers shared read-only data (weights broadcast to multiple cores). This is distinct from the programmer-visible NRAM/WRAM scratchpads.

---

## 4. Off-chip Memory

### MLU370-X8 (Siyuan 370)

| Spec | Value |
|------|-------|
| Memory type | LPDDR5 |
| Capacity (dual-chip card) | 48 GB |
| Bandwidth (dual-chip) | 614.4 GB/s |
| Configuration | Dual Siyuan 370 dies with MLU-Link |

LPDDR5 was a deliberate choice for the MLU370 generation due to HBM supply constraints (China export controls restrict HBM availability). Siyuan 370 is described as China's first cloud AI chip supporting LPDDR5, with 3× the memory bandwidth of its predecessor (MLU270).

### MLU290-M5

| Spec | Value |
|------|-------|
| Memory type | HBM2 |
| Capacity | 32 GB |
| Bandwidth | 1,228 GB/s |
| Peak INT8 | 512 TOPS |
| Peak FP32 | 64 TFLOPS |

### MLU590 (Siyuan 590)

Uses more advanced HBM (HBM2e or HBM3, not publicly confirmed). Targets LLM training and inference with higher memory bandwidth than MLU370. Cambricon has cited HBM supply constraints as a production bottleneck for this generation.

---

## 5. Host Interface / Package

### PCIe

| Product | PCIe Version | Form Factor |
|---------|-------------|-------------|
| MLU370-M8 | PCIe Gen4 x16 | Full-height, full-length (FHFL) |
| MLU370-X | PCIe Gen4 x16 | FHFL intelligent accelerating card |
| MLU290-M5 | PCIe Gen4 x16 | Server PCIe slot |
| MLU-X1001 | PCIe x16 + Mini SAS HD | External accelerator module |

PCIe Gen4 x16 provides ~64 GB/s bidirectional bandwidth for host-to-device DMA.

### Packaging

MLU chips use a conventional monolithic die-on-PCB approach for most products. The **Cambricon-LLM** research paper (arXiv 2409.15654) describes a future **chiplet-based** design targeting on-device 70B LLM inference, combining compute chiplets with memory chiplets in a 2.5D packaging configuration. This is a research/roadmap direction, not a shipping product as of 2026-04-05.

---

## 6. Scale-up Interconnect (MLU-Link)

**MLU-Link** is Cambricon's proprietary chip-to-chip direct interconnect, analogous to NVIDIA NVLink. It provides high-bandwidth, low-latency connections between multiple MLU dies for scale-up multi-chip configurations.

Key characteristics:
- Used in the **MLU370-X8**: dual Siyuan 370 dies connected via MLU-Link, sharing 48 GB LPDDR5 as a unified memory pool
- Enables 8-card parallel training via a server backplane connecting MLU-Link-capable cards
- Published benchmark: 8-card MLU-Link setup achieved ~155% of a 350W RTX GPU on BERT/YOLO/ResNet workloads (Cambricon internal report, 2022)

Unlike NVIDIA NVLink/NVSwitch, no external MLU-Link switch ASIC (analogous to NVSwitch) has been publicly disclosed. Scale-up beyond 8 cards currently relies on scale-out Ethernet/InfiniBand.

---

## 7. Scale-out Interconnect

Cambricon MLU clusters use standard Ethernet or InfiniBand for inter-node communication:
- **CNCL** (Cambricon Collective Communication Library) handles intra-node MLU-to-MLU collectives over MLU-Link
- **Gloo** is used for cross-vendor heterogeneous scenarios (per KAITIAN paper)
- For large-scale deployments, standard 100/400GbE or InfiniBand host NICs are used; no proprietary scale-out NIC

The KAITIAN framework (arXiv 2505.10183) documents CNCL achieving efficient AllReduce/AllGather/ReduceScatter within an MLU pod, with Gloo bridging to non-MLU accelerators in heterogeneous training clusters.

---

## Sources

- [Machine Learning Unit (MLU) — WikiChip](https://en.wikichip.org/wiki/cambricon/mlu)
- [MLU370-M8 Product Manual (FCC)](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf)
- [Cambricon-LLM: Chiplet-Based Hybrid Architecture (arXiv 2409.15654)](https://arxiv.org/html/2409.15654v1)
- [BANG C Language Developer Guide v2.4.1](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf)
- [Cambricon ISA — IEEE ISCA 2016](https://ieeexplore.ieee.org/document/7551409/)
- [KAITIAN Communication Framework (arXiv 2505.10183)](https://arxiv.org/html/2505.10183v1)
- [Optimizing Logcumsumexp on Cambricon MLU (ResearchGate)](https://www.researchgate.net/publication/393382745_Optimizing_Logcumsumexp_on_Cambricon_MLU_Architecture-Aware_Scheduling_and_Memory_Management)
- [Cambricon AIPE profile](https://aiproduct.engineer/ai-ecosystem/cambricon-a32b3a8eeb26)

---

# Investigation Update — 2026-08-08

*resource: https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF (Cambricon 2026 半年度报告) + secondary Chinese coverage of Siyuan 690 + Bloomberg-derived shipment reporting*
*as_of: 2026-08-08*
*chip: cambricon*
*layer: Compute Engine, Off-chip Memory, Host Interface / Package, Scale-up Interconnect*
*change class: major (mostly a correction of circulating claims rather than new confirmed silicon)*

## Summary of this update

The dominant finding of this window is **negative**, and it matters more than the positive one. A widely circulating narrative holds that Cambricon's product center of gravity has shifted to a Siyuan 590 / Siyuan 690 pair shipping in volume. Cambricon's own 2026 half-year report — the only primary source on the company's product line published in this window — **does not support that**. It names 思元100, 思元220, 思元270, 思元290 and 思元370 in its product and core-technology sections, and contains **zero** occurrences of 思元590, 思元690, or HBM. (All "590" and "690" string hits in the filing are financial figures.) Cambricon's own DeepSeek-V4 Day-0 article, published 2026-04-24, lists **MLU370-S4 / X4 / X8** as the board options. `vllm-mlu` still states `MLU370以上的设备` as its hardware requirement.

Conclusion for this survey: **MLU370 remains the only vendor-documented datacenter part.** The MLU590 row stays TBD. The MLU690 is recorded separately as reported-but-unverified.

## 1. Compute Engine — MLU690 (reported, unverified)

No Cambricon datasheet exists. Product pages returned HTTP 500 during the scan. The figures below recur across Chinese brokerage and blog coverage and are recorded only so the survey can state what is circulating and where it fails:

| Attribute | Reported | Provenance / caveat |
|-----------|----------|--------------------|
| Package | Dual-die / chiplet | 51CTO blog |
| Peak FP16 | >700 TFLOPS | Sina Finance brokerage report |
| Peak INT4 | >2,800 TOPS | 51CTO blog |
| Peak FP8 | 2,000 TOPS | Tianyancha — from the *conflicting* 7nm source; treat as incompatible with the other set |
| Process node | **not disclosed** | Direct contradiction: TSMC 4nm in one source set, 7nm in another. No node is published |
| Transistor count | **not disclosed** | Only the conflicting 7nm source gives >50B |
| Clusters / MLU Cores | **not disclosed** | — |
| Precision format list | **not disclosed** | — |
| Architecture name | **not disclosed** | No `MLUarch04` or `MLUarch05` name is confirmed anywhere |

**Production status is contested, not confirmed.** Secondary coverage of the H1 report asserts mass production in early 2026; other coverage in the same period describes the part as still in final testing; TrendForce in December 2025 placed it in the testing phase with volume production potentially slipping to H2 2026. The correct record: *reported by secondary Chinese media as entering production in early 2026; not confirmed by any Cambricon primary source.*

## 2. Off-chip Memory — MLU690 (reported, unverified)

Reported as **196 GB HBM3 at ~3.35 TB/s**. HBM appears nowhere in Cambricon's H1 filing, so this is unverified. If real, it is the first HBM3 part in the MLU line and it sits directly inside the HBM supply constraint discussed below. MLU590's HBM generation, capacity and bandwidth all remain **not disclosed**.

## 3. Scale-up Interconnect — units correction

Reported MLU690 interconnect bandwidth is **超890Gbps** — 890 **gigabits** per second, ≈111 GB/s. A "890 GB/s" restatement circulates widely and is wrong by roughly 8×. This survey records the gigabit figure, flagged unverified.

## 4. Correction to a pre-existing repo statement

The previous revision of `chips/cambricon/hw-architecture.md` stated that "Cambricon cited HBM availability as a key production limiter for the MLU590 ramp to 500,000 units in 2026." Both halves of that are wrong:

1. The **500,000-unit figure is not a Cambricon statement.** It is a third-party supply-chain target originating in Bloomberg reporting dated **2025-12-04**, echoed downstream by Tom's Hardware, TrendForce and aggregators. The related "~300,000 units of Siyuan 590 and 690" breakdown and the "~116,000–142,000 units in 2025" baseline come from the same reporting.
2. It is **not MLU590-specific** — it is a target for Cambricon's total 2026 accelerator output, and it describes targets, not shipments. No unit-shipment figure is confirmed by any Cambricon source.

Named risks to that target in the same third-party reporting: low yield, **SMIC 7nm-class capacity contention with Huawei** for the same wafer allocation, and **HBM supply**. Domestic HBM3 from CXMT is targeted for end-2026, i.e. after the stated ramp.

Also note the timing: this reporting **pre-dates the repo's 2026-04-05 baseline**. It is not new information in the April–August 2026 window and must not be presented as such.

## 5. Corporate context (bears on roadmap credibility, not on architecture)

From the primary H1 2026 filing (2026-08-08): revenue RMB 5.996 B (+108.13% YoY), net profit attributable to shareholders RMB 2.311 B (+122.61%), ex-nonrecurring net profit RMB 2.166 B (+137.30%). First half-year above RMB 5 B since listing. The filing discloses inventory of RMB 8.247 B at **45.32% of total assets** and names inventory write-down as a risk factor — this is a primary-source disclosure, not an analyst estimate.

On 2026-07-28 Cambricon announced a restricted-stock incentive plan with tiered revenue targets: **2026 ≥ RMB 13.5 B; 2026–2027 cumulative ≥ RMB 40.5 B; 2026–2028 cumulative ≥ RMB 100 B**. Roughly an 8× step up on the current run rate, and the clearest public statement of the company's own three-year ambition.

## 6. Negative results (confirmed absent)

- No Cambricon submission in MLPerf Inference v6.0 (2026-04) or any prior MLPerf round.
- No `MLUarch04` / `MLUarch05` architecture name confirmed anywhere.
- No discontinuation or cancellation news for any MLU part.
- No Cambricon paper or talk found at ISCA 2026 or Hot Chips 2026.

## Sources (2026-08-08 update)

- [Cambricon 2026 半年度报告 (cninfo)](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF) — PRIMARY
- [DeepSeek-V4 Day-0 adaptation — Cambricon developer article (2026-04-24)](https://developer.cambricon.com/index/article/details.html?id=12) — PRIMARY; lists MLU370-S4/X4/X8 as board options
- [vllm-mlu README (master)](https://raw.githubusercontent.com/Cambricon/vllm-mlu/master/README.md) — PRIMARY; `MLU370以上的设备`
- [TrendForce — Cambricon remains China's top AI chip startup (2025-12-15)](https://www.trendforce.com/news/2025/12/15/insights-cambricon-remains-chinas-top-ai-chip-startup-rumored-2026-triple-output-faces-smic-limits/)
- [Tom's Hardware — Cambricon targets 500,000 AI chips in 2026](https://www.tomshardware.com/tech-industry/semiconductors/cambricon-targets-500000-ai-chips-in-2026-as-china-accelerates-domestic-hardware-push) — third-party target, downstream of Bloomberg 2025-12-04
- [Sina Finance brokerage report on Siyuan 690](https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/831133396021/index.phtml) — secondary/unverified
- [51CTO blog on Siyuan 690](https://blog.51cto.com/u_10819805/14632220) — secondary/unverified
- [Tianyancha news item on Siyuan 690](https://news.tianyancha.com/ll_5k94fbk79z.html) — secondary, conflicting node claim
- [Zhihu post claiming early-2026 mass production](https://zhuanlan.zhihu.com/p/2061515468637214392) — secondary, contradicted by other coverage and unmentioned in the H1 filing
