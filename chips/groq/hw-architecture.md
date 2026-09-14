# Groq LPU Hardware Architecture

*as_of: 2026-09-13*
*Architectures: LPU v1 (Samsung 14nm), LPU v2 (Samsung 4nm), NVIDIA Groq 3 LPX LP30 (process node not disclosed; announced GTC 2026-03-16; **full production as of 2026-08-24**)*

---

## Generation Overview

| Generation | Vendor / status | Process | On-chip SRAM | SRAM BW | Dense compute | Chip-to-chip | System unit |
|---|---|---|---|---|---|---|---|
| **LPU v1 (TSP)** | Groq, shipping since 2022 | Samsung 14nm, 25×29 mm (~725 mm²), 900 MHz | ~230 MB | 80 TB/s | 750 INT8 TOPS / 188 FP16 TFLOPS | Plesiosynchronous direct mesh | GroqRack: 576 chips, ~130 GB SRAM |
| **LPU v2** | Groq, production 2025 | Samsung 4nm | Not disclosed | Not disclosed | Not disclosed ("2x+ over v1", vendor positioning) | Plesiosynchronous | GroqRack |
| **NVIDIA Groq 3 LPX (LP30)** | NVIDIA, **FULL PRODUCTION** as of 2026-08-24 (announced GTC 2026-03-16) | **Not disclosed** | **500 MB** | **150 TB/s** | **1.2 PFLOPS FP8/chip; 315 PFLOPS FP8/rack** (other datatypes not disclosed) | **96 links @ 112 Gbps = 2.5 TB/s aggregate bidirectional** | LPX rack: **256 LPUs in 32 liquid-cooled 1U trays of 8**; 128 GB SRAM, 40 PB/s aggregate SRAM BW; 12 TB DDR5; 11,000 tok/s decode (Gemma-4-class 31B, vendor benchmark) |

*The LP30 row supersedes earlier revisions of this file, which stated 512 MB SRAM, "Samsung 4nm" and "Q3 2026". See the 2026-08-08 update section below for those corrections and the 2026-09-13 update section for the full-production status change. **Full rack hardware specs are the responsibility of `public/chips/nvidia-gpu/hw-architecture.md` §8.7/§10.3 — this file's NVIDIA-side numbers are kept in sync with, but not the primary record for, that file.***

---

## Overview

The Groq LPU (Language Processing Unit), originally called the Tensor Streaming Processor (TSP), is built around a single architectural insight: **for transformer inference, eliminate all dynamic hardware and let the compiler do everything**. The hardware contains no caches, no branch predictors, no out-of-order execution, and no DRAM. It is a deterministic executor of a compiler-generated cycle-exact schedule.

The result is an architecture that achieves ~80 TB/s internal SRAM bandwidth on a single die — approximately 24x the bandwidth of NVIDIA's H100 HBM3 — and delivers deterministic, zero-jitter inference latency.

---

## 1. Compute Engine

### Matrix Multiplication Unit (MMU)

The core compute unit is a **320x320 fused dot product array** — a hardware matrix multiplier that processes one 320-wide vector dot product per cycle. This is the GEMM engine, handling the dominant operation in transformer inference (QKV projection, FFN layers, output projection).

| Parameter | LPU v1 |
|-----------|--------|
| Process | Samsung 14nm |
| Die size | 25x29 mm (~725 mm²) |
| Clock | 900 MHz |
| Matrix unit | 320x320 fused dot product |
| INT8 TOPS | 750 |
| FP16 TFLOPS | 188 |
| Compute density | >1 TOPS/mm² |

**NVIDIA Groq 3 LPX (LP30) compute — added 2026-08-08.** NVIDIA publishes only FP8 throughput for LP30:

| Parameter | LP30 |
|-----------|------|
| Process | **Not disclosed** (the "Samsung 4nm" attribution in earlier revisions is *not confirmed*) |
| Die size | Not disclosed |
| Clock | Not disclosed |
| TDP | Not disclosed |
| FP8 dense compute | **1.2 PFLOPS per chip** |
| FP8 dense compute per tray | **9.6 PFLOPS** (8 LPUs) |
| FP16 / BF16 / INT8 throughput | **Not disclosed** |
| Matrix/vector unit organization | Not re-disclosed by NVIDIA; assumed to inherit the TSP functional-slice organization |

NVIDIA's blog additionally quotes **315 PFLOPS of AI inference compute per LPX rack**, which does not reconcile with 9.6 PFLOPS/tray × 32 trays = 307.2 PFLOPS. The discrepancy is unexplained in NVIDIA's material; 315 PFLOPS should not be presented as a dense FP8 figure without this caveat.

The ISCA 2020 paper reports: ResNet50 inference in <43 μs at batch=1, 20.4K images/sec — 2.5x faster than TPU v3 at comparable batch.

### Vector ALUs

**5,120 vector ALUs** handle scalar and element-wise operations:
- Activation functions: GeLU, SiLU, ReLU, Sigmoid, Tanh
- Layer normalization
- Element-wise Add, Multiply
- Softmax (decomposed into element-wise ops)
- Attention score scaling

### No Traditional Control Hardware

Unlike GPUs, the LPU contains **none** of:
- Warp schedulers
- Branch predictors
- Out-of-order execution buffers
- Cache hierarchy
- Memory arbiters

These are replaced entirely by the compiler's pre-computed cycle-exact schedule.

---

## 2. Data Path: Tensor Streaming

The TSP's defining architectural pattern is **tensor streaming**: functional units are arranged in a 2D spatial grid, and activations flow through this grid on a compiler-determined conveyor-belt path.

The compiler maps each tensor operation to:
1. A **spatial position** — which physical unit (MMU or specific Vector ALU bank) executes it
2. A **temporal position** — which exact clock cycle it executes
3. A **memory address** — which SRAM bank addresses hold its inputs/outputs

At runtime, data streams from one unit to the next exactly as the compiler planned. There is no hardware scheduler deciding what runs next — the "scheduler" is the pre-loaded instruction stream.

This eliminates:
- Cache miss stalls (no cache — data is already in the right SRAM bank at the right cycle)
- Warp scheduling overhead (no warps — one deterministic execution path)
- Speculative execution waste (no speculation — execution is known-good at compile time)

---

## 3. On-chip Memory (All-SRAM Architecture)

The most radical architectural choice is **all-SRAM primary memory with no off-chip DRAM**.

| Parameter | LPU v1 | **Groq 3 LPX (LP30)** | H100 (comparison) |
|-----------|--------|----------------------|-------------------|
| Memory type | On-chip SRAM | **On-chip SRAM** | Off-chip HBM3 |
| Capacity | ~230 MB | **500 MB** | 80 GB |
| Bandwidth | 80 TB/s | **150 TB/s** | 3.35 TB/s |
| Bandwidth ratio | **24x faster** | **~45x faster** | — |
| Access latency | Single-cycle SRAM | Not disclosed | HBM: hundreds of ns |

*LP30 SRAM capacity corrected 2026-08-08: NVIDIA publishes **500 MB**, not the 512 MB carried in earlier revisions.*

**The SRAM is not a cache** — it is the primary weight storage and activation buffer. The compiler statically assigns every tensor to specific SRAM banks and schedules every read/write. There are no cache misses because there is no cache.

For LLM inference, the dominant bottleneck is the memory bandwidth required to load model weights into compute units for each token generation step. At batch=1, a transformer layer requires loading all weight matrices once per forward pass. At 80 TB/s, the LPU loads these weights 24x faster than an H100 reading from HBM3.

### Precision Support

| Format | LPU v1 | LPU v2 | Groq 3 LPX (LP30) |
|--------|--------|--------|-------------------|
| INT8 | Yes | Yes | Not disclosed |
| FP16 | Yes | Yes | Not disclosed |
| BF16 | No | Yes | Not disclosed |
| FP8 | — | Not disclosed | **Yes — the only datatype NVIDIA publishes throughput for** |

---

## 4. Off-chip Memory

A single LPU v1 die has **no off-chip DRAM**. For models that exceed 230 MB:

- Model weights are **striped across multiple chips** in a GroqRack
- Each chip holds its assigned weight shard in SRAM
- All chips execute in parallel, each processing the portion of the model mapped to them
- The compiler pre-computes which chip holds which weight shard and generates per-chip instruction streams

GroqRack aggregate SRAM:
- 576 chips x 230 MB = ~130 GB total (LPU v1)
- Llama 3 70B (FP16) ~140 GB — fits in a GroqRack with headroom for activations

### Rack-level DDR5 in the LPX generation (added 2026-08-08)

The "no DRAM anywhere in the system" architectural absolute **no longer holds at rack level for Groq 3 LPX**. NVIDIA's LPX product page lists **12 TB of DDR5 per 256-LPU rack**. The LP30 die itself still has no attached DRAM — the all-SRAM per-chip model is intact — but the LPX rack introduces a second, capacity-oriented memory tier that the original TSP design deliberately did not have. NVIDIA does not disclose the DDR5 bandwidth, its attachment point (host CPU vs. tray-local), or how the compiler/runtime schedules movement between the DDR5 tier and LPU SRAM.

---

## 5. Host Interface / Package

### LPU v1

| Parameter | Value |
|-----------|-------|
| Process | Samsung 14nm |
| Die area | ~725 mm² (25x29 mm) |
| Clock | 900 MHz |
| Host interface | PCIe |
| Cooling | Air-cooled (no liquid cooling required) |
| Packaging | Standard monolithic |

### LPU v2 (2025)

- Samsung 4nm process (denser, more compute and SRAM per die)
- Specifications not fully disclosed; positioned as 2x+ improvement over v1

### NVIDIA Groq 3 LPX — LP30 (announced GTC 2026-03-16; corrected 2026-08-08)

| Parameter | Value |
|-----------|-------|
| Status | **FULL PRODUCTION as of 2026-08-24** (NVIDIA Newsroom: "NVIDIA Groq 3 LPX...is now in full production"). Announced GTC 2026-03-16; production confirmed at Hot Chips 38 (2026-08-24/25). **Nebius** is the first cloud adopter; Groq itself is among the earliest adopters. *(Status corrected 2026-09-13 — supersedes the 2026-08-08 "ANNOUNCED / H2 2026 guidance" entry below.)* |
| Process | **Not disclosed** — the "Samsung 4nm" figure in earlier revisions is *not confirmed* (low-quality secondary blogs only) |
| TDP | **Not disclosed** |
| SRAM per die | **500 MB** (was incorrectly 512 MB) |
| SRAM bandwidth | 150 TB/s |
| FP8 compute | 1.2 PFLOPS/chip; 9.6 PFLOPS/tray; **315 PFLOPS/rack** (Hot Chips 38, 2026-08-24/25) |
| Tray | 1U liquid-cooled, 8 LPUs |
| Chips per rack | 256 (32 trays × 8) |
| Rack SRAM | 128 GB, 40 PB/s aggregate |
| Rack DDR5 | 12 TB |
| Rack scale-up BW | 640 TB/s |
| Decode throughput | **11,000 tok/s** on a Gemma-4-class 31B model, per rack (Hot Chips 38, ServeTheHome) — a separate benchmark configuration from NVIDIA Newsroom's 3,400 tok/s / 100K-context figure; the two are not directly comparable |
| Cooling | **Liquid-cooled** (a departure from the air-cooled GroqRack) |
| Role | Latency-sensitive decode co-processor in NVIDIA Vera Rubin platform (see corrected split below) |
| Throughput claim | "up to 35x higher inference throughput per megawatt" — **NVIDIA marketing claim, unaudited** |
| Production status | **Full production as of 2026-08-24** (NVIDIA Newsroom); Nebius first cloud adopter |

*Full derivation and sourcing for the rows above (256-LPU rack composition, 315 PFLOPS FP8, 11,000 tok/s decode, full-production status) is maintained in `public/chips/nvidia-gpu/hw-architecture.md` §8.7/§10.3 — treat that file as authoritative if the two ever diverge.*

**Corrected division of labour (2026-08-08).** Earlier revisions of this survey said "Rubin GPUs handle prefill, Groq LPUs handle token generation." Per NVIDIA's own developer blog, the split is *within decode*: Rubin GPUs take the **throughput-bound** decode work — notably full-context attention over the accumulated KV cache — while LPX accelerates the **latency-sensitive** execution within decode, notably sparse MoE expert feed-forward networks.

---

## 6. Scale-up Interconnect: Plesiosynchronous Fabric

### Protocol

Groq uses a **plesiosynchronous** chip-to-chip interconnect — a near-synchronous protocol where chips run on independent crystal oscillators with known, bounded drift, and software periodically re-synchronizes them.

This is a deliberate choice vs. the alternatives:
- **Synchronous (shared clock)**: clock distribution at scale (hundreds of chips) is an engineering nightmare
- **Asynchronous**: requires handshake signaling, adding latency and hardware complexity
- **Plesiosynchronous**: chips run nearly synchronously, software corrects drift → compiler can predict exact packet arrival cycles

### What this enables

Because the compiler knows the inter-chip packet delivery times to exact clock cycles, it can:
- Schedule inter-chip data movements as if they were on-chip memory reads
- Generate instruction streams for all 576 chips that coordinate precisely without any runtime communication protocol
- Make the entire rack appear as a single flat SRAM address space

### GroqRack Specifications

| Parameter | Value |
|-----------|-------|
| Chips | 576 LPU dies |
| Interconnect | Direct mesh / dragonfly variant |
| External switches | None (direct chip-to-chip) |
| Total SRAM | ~130 GB (576 x 230 MB) |
| Aggregate chip-to-chip BW | **Unverified** — earlier revisions said "~640 TB/s". No Groq primary source for this number was found in the 2026-08-08 scan, and 640 TB/s is *exactly* NVIDIA's published rack-scale scale-up figure for the 256-LPU LPX rack, so this is likely a conflation. Treat as unverified until a Groq source is located. |
| Coherence model | Single coherent memory (compiler-enforced) |
| Cooling | Air-cooled |

### NVIDIA Groq 3 LPX Rack Specifications (added 2026-08-08)

| Parameter | Value |
|-----------|-------|
| Chips | **256 LPUs** |
| Physical organization | **32 liquid-cooled 1U trays × 8 LPUs** |
| Per-chip chip-to-chip links | **96 links @ 112 Gbps** |
| Per-chip aggregate C2C BW | **2.5 TB/s bidirectional** |
| Rack-scale scale-up BW | **640 TB/s** |
| Total SRAM | **128 GB** (256 × 500 MB) |
| Aggregate SRAM BW | **40 PB/s** |
| Rack DDR5 | **12 TB** |
| Fabric protocol | Not re-disclosed by NVIDIA (whether the plesiosynchronous scheme is retained by *name* is **not disclosed**) |
| Cooling | Liquid |
| Decode throughput | 11,000 tok/s (Gemma-4-class 31B) per rack — Hot Chips 38 |
| Production status | **Full production as of 2026-08-24** |

The LPX rack is a substantially smaller node count than the 576-chip GroqRack (256 vs 576) but with ~2.2× the SRAM per chip, giving comparable aggregate SRAM (128 GB vs ~130 GB) at far higher bandwidth. **Hot Chips 38 (2026-08-24/25) has now presented** (see the 2026-09-13 update section below); ServeTheHome's coverage describes "fully deterministic design, software-based instruction scheduling (no hardware scheduler)" and "256 chips act as one unified compute system via on-chip routing" — behaviorally consistent with a plesiosynchronous-style scheme, but NVIDIA has still not used the term "plesiosynchronous" itself, so retention of the *named* Groq protocol remains **not disclosed**.

---

## 7. Scale-out Interconnect

GroqRack units connect to datacenter networks via standard Ethernet or InfiniBand host NICs. There is no proprietary scale-out fabric — the plesiosynchronous protocol is only for intra-rack chip-to-chip communication.

For GroqCloud, GroqRack servers are hosted in datacenters and accessed via HTTPS REST API.

---

## Hardware Design Trade-offs

| Design Decision | What is Gained | What is Given Up |
|-----------------|----------------|------------------|
| All-SRAM (no DRAM) | 80 TB/s BW, zero latency variance | Capacity (230 MB/chip) — large models need 100s of chips |
| Compiler scheduling (no dynamic HW) | Deterministic latency, zero jitter, zero scheduling overhead | Cannot handle dynamic shapes without recompilation |
| No cache hierarchy | Eliminates cache miss stalls; 100% data reuse encoded in schedule | Complex compiler; requires fixed-shape programs |
| Plesiosynchronous fabric | Rack-scale coherence without hardware cost of shared clock | Periodic software sync; not suitable for truly asynchronous workloads |
| No training support | Simpler hardware optimized for inference | No backprop (SRAM too small for activation memory at training scales) |
| VLIW-like instruction format | Hardware simplicity; no dispatch logic | Compiler must solve NP-hard scheduling; compile times can be long |

---

## Comparison: Groq LPU vs. Contemporaries

| Dimension | Groq LPU v1 | NVIDIA H100 | Cerebras CS-3 |
|-----------|-------------|-------------|---------------|
| Memory type | SRAM only | HBM3 + SMEM | SRAM wafer-scale |
| Memory BW | 80 TB/s (SRAM) | 3.35 TB/s (HBM3) | ~20 PB/s (WSE-3) |
| Execution | Compiler-scheduled static | SIMT dynamic | Dataflow static |
| Cache | None (SRAM is primary) | 50 MB L2 + SMEM | None |
| Inference latency | Deterministic / zero jitter | Variable (cache/scheduler) | Low (SRAM-resident) |
| Token gen (70B) | ~900 tok/s | ~200 tok/s (H100 SXM5 BS=1) | High throughput |
| Training | Not designed for | Primary use case | Supported |

---

## Update — 2026-08-08 (LP30 corrections; corporate status; Hot Chips 38 preview)

*Sources: NVIDIA developer blog 2026-03-16; nvidia.com/en-us/data-center/lpx; nvidianews Vera Rubin platform release; groq.com/newsroom 2025-12-24 and 2026-06-22; groq.com/about-us; hotchips.org/advance-program; StorageReview; CRN.*

**Specification corrections applied to this file:**

| Was (≤ 2026-04-05 revision) | Now (2026-08-08) |
|---|---|
| LP30 SRAM 512 MB | **500 MB** — the figure NVIDIA actually publishes |
| LP30 "Samsung 4nm" | **Process node not disclosed**; the 4nm attribution appears only in low-quality secondary blogs |
| LP30 "ships Q3 2026" | **ANNOUNCED** at GTC 2026-03-16; NVIDIA guides Vera Rubin partner availability to **H2 2026**; no LPX-specific ship date |
| "Rubin handles prefill, LPX handles token generation" | Split is *within decode*: GPUs take throughput-bound full-context attention over the KV cache; **LPX takes latency-sensitive work such as sparse MoE expert FFNs** |
| GroqRack "~640 TB/s aggregate" | **Unverified** — probable conflation with NVIDIA's 640 TB/s LPX rack scale-up figure |
| "no DRAM" as a system-wide absolute | Holds per-die; **the LPX rack carries 12 TB of DDR5** |

**New facts added to this file:** 96 chip-to-chip links @ 112 Gbps (2.5 TB/s aggregate bidirectional per chip); 1.2 PFLOPS FP8/chip and 9.6 PFLOPS FP8/tray; rack = 256 LPUs in 32 liquid-cooled 1U trays of 8; 128 GB rack SRAM at 40 PB/s; 640 TB/s rack scale-up; 12 TB rack DDR5. NVIDIA's "up to 35x higher inference throughput per megawatt" is a **vendor marketing claim, unaudited**. NVIDIA's stated 315 PFLOPS/rack does not reconcile with 9.6 × 32 = 307.2 PFLOPS.

**Still not disclosed for LP30:** process node, TDP, die size, clock, transistor count, non-FP8 datatype throughput, whether the plesiosynchronous fabric is retained, and DDR5 tier bandwidth/attachment.

**Corporate status (affects how this chip should be attributed).** The December 2025 NVIDIA–Groq transaction was a **non-exclusive inference-technology licensing agreement plus a large acqui-hire**, *not* an acquisition — Groq remains an independent company and GroqCloud continues to operate. The widely cited ~$20B figure originated with CNBC (2025-12-24) and is **not confirmed by either company**. Groq raised **$650M on 2026-06-22** (Disruptive and Infinitum leading; valuation not disclosed) and has pivoted to inference-cloud operation — 13 data centers, >5M developers, targeting ~200 MW by end of 2027 — and is now itself a **customer for NVIDIA LPX systems** built on its own licensed IP. Groq announced **no new silicon** between April and August 2026. See `chips/groq/summary.md` for the full corporate section.

**Forward-looking (superseded — see 2026-09-13 update below).** Hot Chips 38 (Aug 23–25, 2026, Memorial Auditorium, Stanford), Session AI 1, Tuesday 2026-08-25 2:15–4:15 PM PDT: **"Think Fast: LPU Accelerator for Heterogeneous Compute"** — Igor Arsovski & Santosh Raghavan, NVIDIA. *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public [as of 2026-08-08].* This is the first Hot Chips LPU disclosure under NVIDIA branding and the most likely near-term source for process node, TDP and per-datatype throughput.

---

## Update — 2026-09-13 (Hot Chips 38 presented; full production; corporate)

*Scan window 2026-08-08 → 2026-09-13. Sources: NVIDIA Newsroom (2026-08-24); groq.com/newsroom and groq.com/blog (2026-08-12, 2026-08-17, 2026-08-24); groq.com/about-us. Detailed Hot Chips 38 hardware disclosures for LP30/LPX were investigated by a separate pass and are recorded in `public/chips/nvidia-gpu/hw-architecture.md` §8.7/§10.3 — this section covers only what changes in the Groq-specific file: status fields, the small set of numbers restated above for reader convenience, and Groq Inc./GroqCloud corporate developments.*

**"Think Fast: LPU Accelerator for Heterogeneous Compute" has now been presented** (Hot Chips 38, 2026-08-25). Per `nvidia-gpu/hw-architecture.md` §8.7, NVIDIA's own Hot Chips event page and ServeTheHome's on-site coverage supply the first quantified LPX rack numbers: 256 LPUs/rack, 128 GB SRAM, 40 PB/s aggregate SRAM bandwidth, 640 TB/s rack scale-up (reconfirms the 2026-08-08 GTC figures), 315 PFLOPS FP8/rack, and 11,000 tok/s decode throughput on a Gemma-4-class 31B model. **Process node and TDP remain not disclosed** even after Hot Chips 38 — the actual slide deck is attendee-gated (HTTP 401 on direct fetch), so the talk itself did not close this survey's remaining gap.

**NVIDIA confirms full production (2026-08-24).** Status upgrades from "ANNOUNCED / H2 2026 partner availability guidance" to **full production**, per NVIDIA Newsroom, "NVIDIA Groq 3 LPX Now in Full Production With World-Class Speed for Agentic AI." **Nebius** is named the first AI cloud to adopt LPX; Groq itself is named as planning to be "among the platform's earliest adopters." A separately quoted benchmark — 3,400 output tok/s on Gemma 4 31B with 100K context — is a different configuration from the 11,000 tok/s rack-aggregate figure above and the two should not be conflated. The licensing language is reconfirmed unchanged: **"Groq and LPU are used under license from Groq, Inc."**

**Groq Inc./GroqCloud corporate events (August 2026, three items):** (1) 2026-08-12 — Groq becomes an NVIDIA Cloud Partner, explicitly framed by Groq as confirming rather than superseding its independence. (2) 2026-08-17 — Groq closes a $350M "Series A" at a $3.5B valuation (Disruptive leading, NVIDIA planned participation); combined with June 2026's $650M this brings recent funding to $1B; the "Series A" label and the apparent markdown from the previously-reported ~$6.9B valuation are both flagged as unusual/uncertain rather than confirmed facts (see `chips/groq/summary.md` for the full caveat). (3) 2026-08-24 — Groq deploys LPX + Vera Rubin NVL72 with Dell Technologies, again as operator/early-adopter rather than designer. **No new Groq-designed silicon and no self-hosted LPU deployment outside the NVIDIA relationship were found in this window.**

**Leadership:** unchanged from 2026-08-08 (Adam Winter CEO, Matt Eng CFO, Alan Rice COO, Sinclair Schuller CTO), re-checked against groq.com/about-us on 2026-09-13.

**Still not disclosed for LP30/LPX**, even after Hot Chips 38: process node, TDP, die size, clock, transistor count, non-FP8 datatype throughput, and whether NVIDIA retains a plesiosynchronous-*named* fabric (behavior described by ServeTheHome is consistent with one, but the term itself is not used by NVIDIA).

---

## Resources

- [ISCA 2020: Think Fast — A Tensor Streaming Processor](http://pkamath.com/publications/papers/tsp-isca20.pdf)
- [ISCA 2022: Software-Defined Tensor Streaming Multiprocessor](https://dl.acm.org/doi/10.1145/3470496.3527405)
- [Hot Chips 34 (2022)](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)
- [Groq LPU Architecture Page](https://groq.com/lpu-architecture)
- [Inside the LPU: Deconstructing Groq's Speed](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- [NVIDIA Groq 3 LPX Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) (2026-03-16)
- [NVIDIA LPX product page](https://www.nvidia.com/en-us/data-center/lpx/) — 500 MB SRAM/LPU, 128 GB and 40 PB/s per rack, 640 TB/s scale-up, 256 LPUs, 12 TB DDR5 per rack
- [NVIDIA Vera Rubin platform release](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform) — H2 2026 partner availability
- [StorageReview: NVIDIA Groq 3 LPX — Everything We Know](https://www.storagereview.com/news/nvidia-groq-3-lpx-everything-we-know) — secondary; explicitly notes process node and TDP are unknown
- [Hot Chips 38 advance program](https://hotchips.org/advance-program/) — Session AI 1, 2026-08-25, NVIDIA LPU talk; presented 2026-08-25 (see `nvidia-gpu/hw-architecture.md` §8.7 for resulting specs)
- [Zellic Deep Dive on Groq TSP](https://www.zellic.io/blog/groq-tsp-whitepapers/)

### Added 2026-09-13
- [NVIDIA Groq 3 LPX Now in Full Production — NVIDIA Newsroom (2026-08-24)](https://nvidianews.nvidia.com/news/nvidia-groq-3-lpx-now-in-full-production-with-world-class-speed-for-agentic-ai)
- [NVIDIA's Groq 3 LPU Accelerators for Heterogeneous AI Compute at Hot Chips 2026 — ServeTheHome (2026-08-25)](https://www.servethehome.com/nvidias-groq-3-lpu-accelerators-for-heterogeneous-ai-compute-at-hot-chips-2026/) (cited via `nvidia-gpu/hw-architecture.md`)
- [Groq Becomes an NVIDIA Cloud Partner (2026-08-12)](https://groq.com/newsroom/groq-becomes-an-nvidia-cloud-partner)
- [Groq Closes $350 million Series A (2026-08-17)](https://groq.com/newsroom/groq-closes-usd350-million-series-a-building-the-world-s-leading-ai-inference-cloud)
- [Groq Among the First to Bring NVIDIA Groq 3 LPX and Vera Rubin NVL72 to Market (2026-08-24)](https://groq.com/blog/groq-among-the-first-to-bring-nvidia-groq-3-lpx-and-vera-rubin-nvl72-to-market)
- `public/chips/nvidia-gpu/hw-architecture.md` §8.7, §10.3 (internal cross-reference for full LPX rack spec table)
