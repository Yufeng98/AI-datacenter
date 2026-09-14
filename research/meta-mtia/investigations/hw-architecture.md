# Meta MTIA — HW Architecture Investigation

*as_of: 2026-09-13*

## Summary

MTIA is a custom inference ASIC built around a 2D grid of RISC-V + fixed-function processing elements. It deliberately uses LPDDR5 (not HBM) to optimize cost/power for recommendation model inference (DLRM-dominated workload). Two generations deployed: v1 (TSMC 7 nm, 25 W) and v2 (TSMC 5 nm, 90 W).

---

## Compute Architecture

### Processing Element (PE)
- 2 RISC-V cores per PE (one with vector extension)
- Fixed-function units: matrix multiply-accumulate (MMA), data movement DMA, nonlinear function unit
- 128 KB local SRAM per PE
- Execution model: asynchronous dataflow — RISC-V cores generate instructions; FFUs execute as data dependencies resolve

### PE Grid
- MTIA v1 & v2: 8×8 = 64 PEs
- PE-to-PE via Network on Chip (NoC): v2 has 2× NoC bandwidth
- All PEs can see shared on-chip SRAM

---

## Memory Hierarchy

| Level | v1 | v2 / 2i |
|---|---|---|
| PE-local SRAM | 128 KB/PE × 64 = 8 MB | 128 KB/PE × 64 = 8 MB |
| Shared on-chip SRAM | 128 MB | 256 MB |
| SRAM bandwidth | — | 2.7 TB/s (2i) |
| Off-chip DRAM | 64 GB LPDDR5, 176 GB/s | 128 GB LPDDR5, 204.8 GB/s |
| Memory type | LPDDR5 (no HBM) | LPDDR5 (no HBM) |

**Why LPDDR5 not HBM?**
DLRM workloads are bandwidth-bound (embedding table lookups) but at lower bandwidth than training. LPDDR5 provides sufficient bandwidth at significantly lower cost and power — a key design trade-off vs GPU/HBM accelerators.

---

## Compute Performance

| Metric | v1 | v2 |
|---|---|---|
| INT8 TOPS | 102.4 | ~357 (3.5× v1) |
| INT8 TOPS with sparsity | — | ~714 (7× v1) |
| FP16 TFLOPS | 51.2 | — |
| Clock | 800 MHz | 1.35 GHz |
| TDP | 25 W | 90 W |
| Efficiency (TOPS/W) | 4.1 | ~3.97 |

---

## Chip Physical

| Parameter | v1 | v2 |
|---|---|---|
| Process | TSMC 7 nm | TSMC 5 nm |
| Die area | 373 mm² | 421 mm² |
| Dimensions | 19.34×19.1 mm | 25.6×16.4 mm |
| Host interface | PCIe Gen4 | PCIe Gen5 ×8 |

---

## System Deployment

- 2 MTIA chips per accelerator module (220 W module TDP)
- 12 modules per chassis = 24 chips
- 3 chassis per rack = 72 chips
- Per-rack: ~50 petaflops INT8 (sparse) at rack level

---

## Key Design Decisions

1. **RISC-V control cores**: no royalty, full customization at Meta's cadence
2. **Fixed-function MMA units**: max throughput for GEMM, embedding lookup
3. **LPDDR5 vs HBM**: cheaper, sufficient for DLRM; HBM cost would increase CapEx
4. **Large on-chip SRAM**: embedding tables partially cached on-chip
5. **Asynchronous dataflow**: overlaps DMA and compute for bandwidth hiding
6. **Sparsity hardware**: 7× boost for sparse weights — important for pruned recommendation models

---

## MTIA v2 Update

*Added 2026-04-05 based on ISCA 2025 paper (dl.acm.org/doi/10.1145/3695053.3731409), Meta engineering blog (engineering.fb.com/2024/08/22/ml-applications/meta-mtia-hardware-co-design/), and roadmap announcement (about.fb.com/news/2026/03/expanding-metas-custom-silicon-to-power-our-ai-workloads/).*

### Corrections and Clarifications to v1-era Notes

The initial research correctly captured most MTIA v2 specs, but two parameters require correction from the ISCA 2025 paper:

| Parameter | Previously Recorded | Corrected (ISCA 2025) |
|---|---|---|
| PE-local SRAM | 128 KB/PE | **384 KB/PE** (3× increase vs v1) |
| NoC bandwidth increase | 2× over v1 | **3.3× over v1** |
| PCIe interface | Gen5 ×8 | ~~Gen5 ×16 (128 GB/s)~~ → **re-corrected 2026-08-08: 8× Gen5 (32 GB/s)** per ISCA 2025 spec table |

### MTIA v2 / 2i Extended Architecture Detail

**Processing Element enhancements:**
- PE-local SRAM: 384 KB per PE (3× increase from v1's 128 KB)
- Improved MMA array throughput — same 8×8 PE grid, each PE delivers higher utilisation
- Local PE storage bandwidth: ~1 TB/s
- Host-to-accelerator decompression engine: new in v2, improves effective PCIe bandwidth by decompressing model weights on-chip

**Network-on-Chip:**
- v2 NoC: 3.3× bandwidth over MTIA v1 (not 2× as initially noted)
- Enables higher PE-to-PE scatter-gather throughput for embedding lookup coordination

**MTIA 2i (2025 revision of v2):**
- Shared on-chip SRAM: 256 MB at 2.7 TB/s bandwidth (same die, firmware/binning improvement)
- Also referred to as MTIA 200 in the newer numerical naming convention
- Served as the production deployment basis for the ISCA 2025 model-chip co-design study

**Secure Boot Processor:**
- v2 introduced a dedicated secure boot processor alongside the PCIe and DMA host interface — not present in v1

### Production Deployment Results (ISCA 2025)

- MTIA 2i reduces total cost of ownership by an average of **44% vs GPUs** for production recommendation inference workloads
- Meta deployed **23 firmware-bundle releases fleet-wide in 2024** alone, enabling continuous production enhancements without chip respins
- Deployed at **hundreds-of-thousands scale** serving Facebook, Instagram, and WhatsApp ranking/ads workloads

### MTIA 300/400/450/500 Roadmap (Announced March 2026)

Meta announced four successor chips in March 2026, co-designed with Broadcom, shifting to a ~6-month cadence and chiplet/modular architecture:

| Chip | Status (as of April 2026) | Workload | Memory | Key Change |
|---|---|---|---|---|
| MTIA 300 | Production (training) | Ranking/rec training | 216 GB HBM | First MTIA training chip; switches to HBM |
| MTIA 400 | Lab testing | GenAI inference | 288 GB HBM | HBM upgrade over 300 |
| MTIA 450 | Deployment ~early 2027 | GenAI inference | — | 2× HBM BW vs 400; 6× MX4 FP16/BF16 FLOPS |
| MTIA 500 | Deployment ~late 2027 | GenAI inference | — | 50% BW boost over 450 |

**Key architectural shift for 300+ series:**
- Transition from **LPDDR5 → HBM** (marks fundamental workload pivot to LLM/GenAI inference)
- Modular chiplet architecture: MTIA 400, 450, and 500 share the same chassis/rack/network, enabling drop-in upgrades
- From MTIA 300 to 500: HBM bandwidth increases **4.5×**, compute FLOPs increase **25×**
- Training capability: MTIA 300 is the first MTIA generation to support training workloads

---

## MTIA 300 Update — ISCA 2026 Silicon Disclosure

*Added 2026-08-08. Primary source: "MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines", MTIA Team, Meta Platforms, ISCA 2026 Industry Track, Session 5B "Industry Track 2", Tuesday 2026-06-30, 12:00–12:20 EDT, Room 302; corresponding author Chunqiang Tang. PDF: https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf. Verified by downloading the PDF from Meta's own AI-system-codesign publications site and extracting text with `pdftotext -layout`, cross-checked against the ISCA 2026 program listing and an IEEE CSDL proceedings entry. Secondary: Meta newsroom Broadcom PR 2026-04-14; Reuters/TechCrunch 2026-07-09.*

### Why this update is material

Before this scan the repo carried **no silicon-level specifications for MTIA 300 at all** — only the roadmap-level "216 GB HBM / first training-capable" line from Meta's 2026-03-11 blog. The ISCA 2026 paper is the first full disclosure, and it also lets us correct two pre-existing repo errors (MTIA 450's MX4 figure and MTIA 400's HBM capacity) and flag one unresolved conflict (MTIA v2/2i PCIe width).

All figures below are **Meta-stated**, from the paper's Table I unless otherwise noted.

### Physical and electrical

| Parameter | MTIA 2i (Table I) | MTIA 300 (Table I) |
|---|---|---|
| Clock | 1.35 GHz | 1.9 GHz |
| Supply voltage | — | 0.85 V |
| TDP | 85 W (65 W typical) | 912 W (667 W typical) — ~10.7× |
| Cooling | Air | Liquid |
| Instances (paper wording) | "2.35B gates, 103M FLOPS" | "7.2B gates, 511M FLOPS" |
| Compute die | monolithic | compute chiplet 25.6 × 31.4 mm (~804 mm², reticle-sized) |
| Network dies | none | 2 × 25.6 × 9.3 mm network chiplets |
| Package | — | 77.5 × 77.5 mm, 50.3 × 51.9 mm 3.2×-reticle interposer, 2.5D CoWoS |
| Process node | TSMC 5 nm | **NOT DISCLOSED anywhere in the paper** |

**Reading "511M FLOPS".** Table I's *Instances* row literally prints "7.2B gates, 511M FLOPS". In context this is an instance count and "FLOPS" is the paper's abbreviation for **flip-flops** — not floating-point throughput. Quote the paper's wording and gloss it; do not silently substitute a different term, and never treat 511M as a performance number.

### Compute architecture

- **12 × 6 = 72 PEs**, plus **one redundant 1×6 PE row** for yield. Each PE column can tolerate one faulty PE; the configuration is applied at boot, is transparent to software, and has no NoC performance impact.
- **16 Message Engines** at the grid edges.
- Per PE: two 64 B-wide RISC-V vector cores; Command Processor; Memory Layout Unit; **Dot Product Engine** (two 32×64B×32 MAC tiles, **7.82 TFLOPS/PE** with FP16/BF16 in and FP32 out, also FP8 S1E4M3/S1E5M2 and TF32); Reduction Engine; SIMD/SFU; Fabric Interface; Memory Bridge; **512 KB software-managed local memory** split into circular buffers.
- Control core: **RISC-V quad SMP core** (CPU-C).
- NoC: cluster routers, **L-routing** (X then Y), virtual lanes for deadlock avoidance. **No memory crossbar** — removed relative to MTIA 2i.

### Throughput

| Metric | MTIA 2i | MTIA 300 |
|---|---|---|
| GEMM FP16/BF16 | 177 TFLOP/s | **560 TFLOP/s** |
| GEMM FP8 | — | **1,120 TFLOP/s** |
| GEMM INT8 | 354 TOPS | **removed** |
| SIMD engine | — | 42.5 TOPS (FP8/FP16/BF16/FP32) |
| RISC-V vector cores | — | 42.5 (INT8/FP16) / 21.3 (BF16/FP32) TOPS |
| SIMD width | 32 elem/cycle | **128 elem/cycle** |
| GEMM:SIMD ratio | 32:1 | **16:1** |

INT8 was removed from the GEMM path and FP8 added, reflecting the shift from DLRM inference to training.

### Memory hierarchy

| Level | MTIA 2i | MTIA 300 |
|---|---|---|
| PE-local | 384 KB, 1.0 TB/s R+W | **512 KB, 1.9 TB/s R+W** |
| LLC / shared SRAM | 256 MB, 2.7 TB/s | **192 MB (96 MB + 48 banks per side; 96 banks total), 11.4 TB/s R+W** |
| Off-chip | 128 GB LPDDR5, 204.8 GB/s | **216 GB HBM3E, 6 twelve-high stacks (3 per side) @ 8.0 Gbps, 6.1 TB/s read *or* write** |
| Measured | — | **5.57 TB/s (91% of peak)** on a BF16-add kernel |

Three precision notes carried forward from verification:

1. The die figure shows "48 LLC (96MB)" on **each side** — that is 48 banks and 96 MB *per side*, 96 banks and 192 MB total. Any phrasing like "48 banks × 96 MB per side" is garbled.
2. The paper states "each side connects to 3 twelve-high HBM3E stacks" and "HBM3E @8.0Gbps". **36 GB per stack is arithmetic (216/6), not a Meta-stated figure** — label it derived.
3. **Unit mixing:** 6.1 TB/s is read-**or**-write; 11.4 TB/s and 1.9 TB/s are read**+**write. Comparing 6.1 TB/s to competitor HBM peaks is legitimate, but the R-or-W qualifier must survive into any downstream table.

MTIA 300 has *less* on-chip SRAM than MTIA 2i (192 vs 256 MB) but 4.2× the bandwidth — a rebalance away from embedding-hot-row caching toward feeding the GEMM pipeline.

### Built-in NIC chiplets

- **Two network chiplets**, each containing **six custom 800 Gbps (100 GB/s) RDMA/RoCE IP blocks** based on third-party NIC IP → 600 GB/s per chiplet, **12 NICs / 1.2 TB/s per accelerator**.
- Die-to-die attach via **112G SerDes**, 8 × 112G per NIC (Fig. 4).
- **Express doorbells**: the work request itself serves as the doorbell write, avoiding an extra HBM ring-buffer read that cost ~800 ns per transaction.
- **24,576 outstanding work requests across 1,024 QPs per IP block.**
- **QP caching removed** to save area, limiting each **NIC chiplet** (not each IP block) to **1,100 active QPs**.
- Simplified packet-processing pipeline.

Meta claims MTIA 300 is, to their knowledge, the **first accelerator with built-in NIC chiplets plus general-purpose collective offloading engines**.

### Message Engines (collective offload)

- **16 MEs** at PE-grid edges next to HBM, cache and I/O.
- Each: one scalar RISC-V core (**CPU-M**), **256 KB context RAM**, a single large shared Completion Queue, and a **Near Memory Compute** block at **128 B/cycle** for reduction or DMA (**96 B/cycle** if all are active concurrently).
- Aggregate: **up to 2.8 TB/s of reduction throughput** — over 2× the 1.2 TB/s I/O bandwidth — using **one-third of the chip area of the compute engines**.
- Accelerate Reduce, AllReduce, ReduceScatter.

### System and rack

- Host: **16× PCIe Gen5 (64 GB/s)**; **1 CPU with 512 GB RAM per MTIA 300** (1:1 host:accelerator, vs 8 accelerators per host on the H100/H200 testbeds), chosen to avoid PCIe contention.
- **Scale-up: 16 nodes @ 800 GB/s per accelerator** (option up to 1,000 GB/s); multiple racks combinable into larger scale-up domains.
- **Scale-out: 200 GB/s per accelerator**, 4,096-node L1 domain ("L2 unlimited" in Table I; "16K nodes or more if needed" in the text).
- Chassis: **16 compute blade slots + 6 network blade slots** on a pair of cable backplanes; not all network blades need be installed; compute blades mounted vertically to shrink the backplane.
- Two network blade types: **scale-up** (ASIC chosen for low latency and power) and **scale-out** (disaggregated scheduled fabric with packet spray, in-order delivery, fabric-level end-to-end credit). Both liquid-cooled.

### Measured results — and the H100 caveat

- **1.42× higher Perf/TCO than H100** on a production DLRM of approximately 150 billion parameters (99% of parameters in embeddings), local batch 10,240 on 24 MTIA 300; **1.39×** at batch 6,144 on 40 MTIA 300.
- **Communication/collective time 3.9× better than H100**; **embedding operators up to 2.5× H100**.
- ⚠️ **The H100 in Table III is a custom 500 W-power-capped configuration** delivering 780 TF/s BF16 instead of 1,000 TFLOPS at 700 W. Any citation of 1.42× / 1.39× / 3.9× that omits this qualifier materially overstates the advantage.

Table III testbeds:

| | MTIA 300 | H100 (500 W-capped custom) | H200 |
|---|---|---|---|
| BF16 | 560 TF/s | 780 TF/s | 1,000 TF/s |
| Memory | 216 GB | 96 GB | 141 GB |
| Memory BW | 6.1 TB/s | 2.4 TB/s | 4.8 TB/s |
| Accel power | 912 W | 500 W | 700 W |
| Host power | 1,500 W | 6,500 W | 8,850 W |
| Accel per host | 1 | 8 | 8 |
| Scale-up | 16 @ 800 GB/s | 8 @ 450 GB/s | — |
| Scale-out | 200 GB/s | 50 GB/s | — |

### LLM inference evaluation

DeepSeek-R1 on **vLLM** under the **InferenceMax** benchmark; 8-accelerator **TP8-TP8** and **DP8-EP8** configurations; **BF16 attention/KV-cache with FP8 MoE**; short-prompt online scenario with input and output lengths drawn independently and uniformly from [0.8×1024, 1024] tokens; concurrency swept 4 → 256. Meta reports MTIA 300 "outperforms H200 overall on the InferenceMax benchmark", specifically **at concurrency over 64** where execution is decode-dominant and HBM-bandwidth-bound. At low concurrency MTIA 300 loses, attributed to higher small-message communication overhead versus NVLink.

### Limitations stated by Meta

- **MX4 and NVFP4 with row-wise/block-wise scaling are not natively supported** and fall back to RISC-V execution; native MX4 support was added in **MTIA 400**.
- **Eager mode is host-bottlenecked** (Python interpreter, dynamic dispatch, device-host communication) and does not scale with faster silicon.
- Numerical-parity tooling for cross-platform convergence debugging exists but "still require[s] time to mature".

### Roadmap statements inside the paper

MTIA 400 "was designed to rival GB300 GPUs"; MTIA 450 and 500 also target GenAI; MTIA 400 was given higher FLOPS for compute-heavy GenAI and native MX4.

### Ratified by primary source

The paper states MTIA 2i (a.k.a. MTIA 200) is **"now deployed at a scale of hundreds of thousands of chips"** — this confirms the repo's pre-existing scale claim.

---

## Corrections to Previously Recorded Roadmap Facts (2026-08-08)

Checked against Meta's own 2026-03-11 blog, "Four MTIA Chips in Two Years" (https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/):

| Previously recorded | Correct | Note |
|---|---|---|
| "MTIA 450: 6× MX4 FP16/BF16 FLOPS vs MTIA 400" | **MTIA 450 increases MX4 FLOPS by 75% over MTIA 400**, and doubles HBM bandwidth | The "6×" figure is MX4 FLOPS relative to FP16/BF16 **on the same chip**, not relative to MTIA 400. Pre-existing repo transcription bug. |
| "MTIA 400: 288 GB HBM" | **Capacity not disclosed.** Meta publishes only "+400% FP8 FLOPS and +51% HBM bandwidth vs MTIA 300", two compute chiplets, native MX8/MX4, 72-device rack scale-up domain | 288 GB appears only in aggregators. Removed. |
| "MTIA 500: ~late 2027" | **"Scheduled for mass deployment in 2027"** | "late" was an unattributed inference. MTIA 500 also: +50% HBM bandwidth, up to +80% capacity, +43% MX4 FLOPS over MTIA 450; built as a 2×2 configuration of smaller compute chiplets plus HBM stacks, two network chiplets, and an SoC chiplet with PCIe to host CPU and scale-out NICs. |
| "MTIA 300 → 500: 4.5× HBM BW, 25× FLOPs" | **Correct** — matches the blog | No change |

**Derived (not Meta-stated) figures**, labelled as such wherever published: applying MTIA 400's stated "+400% FP8 FLOPS and +51% HBM bandwidth vs MTIA 300" to the now-confirmed MTIA 300 baselines implies roughly **5.6 PFLOP/s FP8 and ~9.2 TB/s**.

### Resolved: MTIA v2 / 2i PCIe width (2026-08-08)

**There was no conflict between the papers — only between the papers and this repo.** Re-verification against the ISCA 2025 paper text settles it:

- The **ISCA 2025** MTIA v2 spec table gives *Host connection: 8× PCIe Gen5 (32 GB/s)* for MTIA 2i and *8× PCIe Gen4 (16 GB/s)* for MTIA v1. The same paper states each accelerator module "houses two MTIA 2i chips, connected via two 8× Gen5 PCIe links", with a PCIe switch linking six modules per server.
- **ISCA 2026** Table I independently lists MTIA-2i as *8× PCIe Gen5 (32 GB/s)* — in agreement.
- The repo's 2026-04-05 entry, "PCIe Gen5 ×16 (128 GB/s)", was **wrong on both lane count and bandwidth** and has been corrected throughout. 128 GB/s corresponds to no Gen5 configuration Meta describes (Gen5 ×16 is ~63 GB/s per direction, ~126 GB/s bidirectional).
- MTIA 300's **16× PCIe Gen5 (64 GB/s)** is therefore a genuine doubling of host lanes over v2, paired with the shift to a 1:1 host:accelerator ratio.

### Not confirmed — do not publish

- Process node for **any** of MTIA 300 / 400 / 450 / 500. "2nm" is aggregator-only; the Broadcom PR states no node.
- MTIA 400 absolute HBM capacity (288 GB is aggregator-only).
- MTIA 500 384 GB / 512 GB capacity — NextPlatform explicitly flags these as author estimates.
- Any claim that MTIA 300 was "deployed H2 2024" — contradicts both Meta's March 2026 framing and the ISCA 2026 paper.
- "Broadcom partnership through 2029" — the 2026-04-14 PR states no end date.

### Non-technical developments since the 2026-04-05 baseline

**Broadcom partnership formalized 2026-04-14** (Meta newsroom): work "across chip design, advanced packaging, and networking", built on Broadcom's XPU platform, with Broadcom Ethernet for cluster networking, under "a commitment that exceeds 1GW, which is the first phase of a sustained, multi-gigawatt rollout". **No process node and no end year appear anywhere in the release.**

**Production-timing signal (secondary, medium-low confidence).** Reuters, 2026-07-09, citing an internal Meta memo, reports Meta plans to begin manufacturing an in-house AI chip code-named **"Iris"** in September 2026, with testing completed in roughly six weeks and no major issues, as part of a plan to reach about 7 GW of compute by end-2026 and 14 GW the following year. TechCrunch and Yahoo Finance carried the same memo. Several aggregators map Iris to MTIA 400 — that mapping is **not confirmed by any primary source**. Retrieval note: reuters.com is blocked in this environment, so the memo was verified only via search snippets and secondary outlets.

---

## Update — 2026-09-13 (Hot Chips 38 realized: MTIA 300 production blog + MTIA 400/450/500 silicon disclosure)

*Window: 2026-08-08 → 2026-09-13. The Hot Chips 38 talk flagged as "scheduled, content not yet public" in the 2026-08-08 pass has now happened (2026-08-25, Session "AI 1": "Meta's Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI", Srinagesh Loke, Cindy Chen, Jatinder Singh). Primary sources checked this pass: Meta Engineering blog https://engineering.fb.com/2026/08/24/networking-traffic/mtia-300-meta-training-chip-built-in-nics/ (2026-08-24, authors Rajiv Krishnamurthy and Wes Bland — full text fetched and read); ai.meta.com/blog listing (checked through its most recent posts as of this scan, no MTIA/Hot-Chips post found there — engineering.fb.com is the correct Meta channel for this disclosure, not ai.meta.com). Secondary/analyst: ServeTheHome, "Meta's MTIA Custom AI Silicon at Hot Chips 2026" (2026-08-25), https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/ — used only for figures not present in the Meta primary blog, and labeled as such below.*

### A. MTIA 300 — new primary-confirmed figures (Meta engineering blog, 2026-08-24)

The Meta blog is a companion piece to the ISCA 2026 paper already recorded in this file (§ "MTIA 300 Update — ISCA 2026 Silicon Disclosure") and to the Hot Chips 38 talk. It restates the ISCA 2026 hardware numbers and adds new **measured production figures**:

| Figure | Value | Status |
|---|---|---|
| HCCL achieved rack bandwidth | **up to 940 GB/s** within a single rack of 16 nodes | Measured, Meta-stated (refines the ISCA 2026 "up to 1,000 GB/s option" theoretical ceiling — 940 GB/s is the achieved figure) |
| Compute/collective isolation | **<0.5% degradation** to compute throughput running large GEMMs concurrently with collectives, vs. **>20% degradation** on "traditional GPUs" (unnamed) | Measured, Meta-stated; GPU baseline is unnamed/general — do not read as a specific competitive benchmark |
| Communication time vs GPU cluster | **3.9× faster** than "the equivalent GPU cluster" on a 150B-parameter production recommendation model across 40 accelerators | Matches (does not add to) the ISCA 2026 paper's 3.9× figure already in this file |
| NIC reuse | The same 12 Ethernet-based NICs are used for both scale-up (16-node rack, up to 1 TB/s) and scale-out (200 GB/s) traffic, repartitioned in software rather than hardware | New framing detail; capacities match existing ISCA 2026 figures |

**Forthcoming paper flagged, not yet available:** "HCCL: Collective Communication for Meta Training and Inference Accelerators", to be published at **SC26** (Supercomputing 2026, Nov 2026) — re-scan after that date for the full HCCL micro-architecture writeup.

### B. MTIA 300 — process node: partially disclosed, analyst-sourced only

ServeTheHome's Hot Chips 38 coverage states MTIA 300 uses a **"3nm" compute die and a "5nm" I/O die**. **This process-node split is not stated anywhere in Meta's own engineering blog or in the ISCA 2026 paper** (both were re-checked; neither mentions a node). Per this repo's evidence rule, the figure is recorded as **analyst-sourced (ServeTheHome, Hot Chips 38 slides), not Meta-primary-confirmed** — an upgrade from the prior "not disclosed anywhere" status, but still short of a vendor-primary citation. Do not present it as Meta-stated.

ServeTheHome also reports a vendor performance claim of **">1.8× vs GPU"** for "forward and backward propagation" — no baseline GPU is named. This is a **different, vaguer claim** than the ISCA 2026 paper's precise 1.42×/1.39× Perf/TCO-vs-500W-capped-H100 figures already recorded in this file; do not conflate the two. Label as vendor claim, unspecified baseline.

### C. MTIA 400 — first hard specifications (analyst-sourced, ServeTheHome Hot Chips 38 coverage; not yet in a Meta primary blog)

Meta's own 2026-08-24 blog explicitly declines to name MTIA 400 or give its specs ("Looking Ahead" section refers only to "Meta's next-generation AI silicon"). All MTIA 400 figures below come from **ServeTheHome's Hot Chips 38 coverage only** and should be cited as such, not as Meta-stated, until a primary source appears:

| Parameter | MTIA 400 (ServeTheHome / Hot Chips 38, analyst-sourced) |
|---|---|
| Packaging | 2 compute chiplets (confirms the pre-existing repo entry) |
| PE grid | **8×6 grid of PEs** (plus a redundancy row, per the MTIA 300 precedent — ServeTheHome's phrasing) |
| HBM | **8 stacks of HBM3e, 9.4 TB/s** memory bandwidth — supersedes the repo's prior *derived* "≈9.2 TB/s" estimate (from "+51% BW vs 300") with a directly reported figure; HBM **capacity still not disclosed** |
| FP4 compute | **12 PFLOP/s FP4** — this is a new, distinct precision figure. Do **not** conflate with the repo's prior *derived* "≈5.6 PFLOP/s FP8" (from "+400% FP8 vs MTIA 300") — that was a different datatype. Both may be independently true; neither replaces the other. |
| vs. MTIA 200 (2i), not MTIA 300 | **15× the FP16 compute** and **46× increase in DRAM bandwidth** vs. MTIA 200 — note the baseline is MTIA 200/2i, not MTIA 300, unlike most other roadmap deltas in this repo which are stated vs. the immediately prior generation |
| Precision | **MXFP4 hardware support** confirmed — matches the pre-existing repo note "native MX4 arrives in MTIA 400" |
| Scale-up networking | **1.2 TB/s** Ethernet-based scale-up fabric bandwidth |
| Scale-up domain | **72 ASICs in a single scale-up domain** — confirms the pre-existing repo figure ("72 devices per rack, switched backplane"), now independently corroborated |
| TDP | **667 W** |
| Scale-out | **100 GB/s** PCIe-based scale-out |

**Not disclosed even by ServeTheHome:** MTIA 400 process node; absolute HBM capacity; deployment/production status beyond "lab-tested" (unchanged from 2026-08-08).

### D. MTIA 450 / 500 — reconfirmed at Hot Chips 38 (analyst-sourced)

- **MTIA 450**: reconfirmed as the **GenAI-inference-focused** variant of the family — consistent with, adds no new figures to, the 2026-08-08 roadmap entries already in this file.
- **MTIA 500**: reconfirmed as targeting a **larger scale-up domain, explicitly "beyond just 72 ASICs in a single domain"** — consistent with the existing "+50% HBM BW, up to +80% capacity, +43% MX4 FLOPS vs 450" roadmap figures; the "beyond 72" framing is a new qualitative confirmation of the scale-up intent, not a new number.

### E. Cadence

Hot Chips 38 reconfirms Meta's roughly **6-month generation cadence "through 2027"** — consistent with, and slightly more specific than, the 2026-04-05 baseline's "~6-month cadence" (which had no stated end point).

### F. Cross-reference only — NOT an MTIA finding

Meta's Aayush Ankit co-presented a **Hot Chips 38 Sunday tutorial**, "3D DRAM based Accelerator for Generative Inference," together with d-Matrix's Sudeep Bhoja. This is **d-Matrix technology, not MTIA** — the deep 3D-DRAM technical content belongs in the d-matrix chip entry (a separate update pass). Recorded here only as a cross-reference so a future scan knows the connection exists; no MTIA architectural claim should be drawn from it.

### G. Not confirmed / could not verify this pass

- MTIA 400 process node — still not disclosed anywhere, primary or analyst.
- Whether MTIA 400 has moved beyond "lab-tested" status — no source in this pass states a production milestone for MTIA 400.
- Any Meta-primary confirmation of the MTIA 300 "3nm compute / 5nm I/O" split (§B) — actively checked (engineering.fb.com, ai.meta.com) and not found.

### Sources for this section (added 2026-09-13)

- https://engineering.fb.com/2026/08/24/networking-traffic/mtia-300-meta-training-chip-built-in-nics/ — Meta Engineering blog, "MTIA 300: Meta's First Training Chip with Built-in NICs and Communication-Offloading Engines", 2026-08-24, Rajiv Krishnamurthy & Wes Bland (primary; fetched and read in full)
- https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/ — ServeTheHome, Hot Chips 38 coverage, 2026-08-25 (analyst; sole source for MTIA 400 figures and MTIA 300 process node)
- https://hotchips.org/program/conference/ — Hot Chips 38 program (talk now realized, was "scheduled" as of 2026-08-08)
- https://ai.meta.com/blog/ — checked, no MTIA/Hot-Chips-38 post found at this URL as of this scan (negative check; engineering.fb.com is the correct channel)

**Scheduled, not yet public.** Hot Chips 38, Session "AI 1", Tuesday **2026-08-25**, 2:15–4:15 PM — Meta, "Meta's Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI" (Srinagesh Loke, Cindy Chen, Jatinder Singh). *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* The title does not contain "MTIA"; describing it as "a Hot Chips MTIA talk" is an inference. Re-scan after 2026-08-25.
