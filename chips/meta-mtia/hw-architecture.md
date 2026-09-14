# Meta MTIA — Hardware Architecture

*as_of: 2026-09-13*

## Generation Overview

| | MTIA v1 | MTIA v2 | MTIA 2i (200) | **MTIA 300** | MTIA 400 | MTIA 450 | MTIA 500 |
|---|---|---|---|---|---|---|---|
| Role | Inference | Inference | Inference | **Training + inference** | GenAI inference | GenAI inference | GenAI inference |
| Status (2026-09-13) | Superseded | Superseded | Deployed at scale (100Ks of chips) | **In production (ranking/rec training)** | Lab-tested only (no status change found this pass) | Scheduled early 2027 | Scheduled 2027 |
| Process | TSMC 7 nm | TSMC 5 nm | TSMC 5 nm | **3nm compute die + 5nm I/O die** (analyst-sourced, ServeTheHome Hot Chips 38 — not Meta-primary-confirmed) | Not disclosed | Not disclosed | Not disclosed |
| Packaging | Monolithic | Monolithic | Monolithic | **2.5D CoWoS: 1 compute chiplet + 2 network chiplets** | 2 compute chiplets | Not disclosed | 2×2 smaller compute chiplets + HBM stacks + 2 network chiplets + SoC chiplet |
| Clock | 800 MHz | 1.35 GHz | 1.35 GHz | **1.9 GHz** | — | — | — |
| TDP | 25 W | 90 W | 85 W (65 W typical, ISCA 2026 Table I) | **912 W (667 W typical)** | **667 W** (ServeTheHome, Hot Chips 38) | — | — |
| Cooling | Air | Air | Air | **Liquid** | — | — | AALC / facility liquid |
| Compute grid | 8×8 = 64 PEs | 8×8 = 64 PEs | 8×8 = 64 PEs | **12×6 = 72 PEs + 1 redundant row (6) + 16 Message Engines** | **8×6 grid of PEs + redundancy row** (ServeTheHome, Hot Chips 38, explicitly stated) | — | — |
| Peak GEMM FP16/BF16 | 51.2 TFLOPS | — | 177 TFLOPS | **560 TFLOP/s** | 15× MTIA 200's FP16 compute (ServeTheHome, Hot Chips 38 — baseline is MTIA 200/2i, not MTIA 300) | — | — |
| Peak GEMM FP8/INT8 | 102.4 TOPS INT8 | ~357 TOPS INT8 (~714 sparse) | 354 TOPS INT8 | **1,120 TFLOP/s FP8** (INT8 GEMM removed) | +400% FP8 vs 300 (≈5.6 PFLOP/s, derived) | +75% MX4 vs 400 | +43% MX4 vs 450 |
| Peak GEMM FP4 | — | — | — | Not supported (native MX4/NVFP4 absent; RISC-V fallback) | **12 PFLOP/s FP4** (ServeTheHome, Hot Chips 38); MXFP4 native hardware support | — | — |
| On-chip SRAM | 128 MB | 256 MB | 256 MB @ 2.7 TB/s | **192 MB LLC @ 11.4 TB/s (R+W)** | — | — | — |
| Off-chip memory | 64 GB LPDDR5, 176 GB/s | 128 GB LPDDR5, 204.8 GB/s | 128 GB LPDDR5, 204.8 GB/s | **216 GB HBM3E, 6.1 TB/s (read *or* write)** | **8× HBM3E stacks, 9.4 TB/s** (ServeTheHome, Hot Chips 38 — supersedes the prior derived "≈9.2 TB/s" estimate; also "46× MTIA 200's DRAM bandwidth"); capacity not disclosed | 2× BW vs 400 | +50% BW, up to +80% capacity vs 450 |
| Built-in networking | None | None | None | **12 × 800 Gbps RoCE NICs (1.2 TB/s) on 2 network chiplets** | **1.2 TB/s** Ethernet-based scale-up fabric (ServeTheHome, Hot Chips 38) | — | 2 network chiplets |
| Scale-up domain | — | — | — | **16 accelerators (1 rack) @ 800 GB/s each** | **72 ASICs in a single scale-up domain** (switched backplane) — independently confirmed at Hot Chips 38 | — | **>72 ASICs** ("beyond just 72 in a single domain" — Hot Chips 38) |
| Host interface | 8× PCIe Gen4 (16 GB/s) | 8× PCIe Gen5 (32 GB/s) | 8× PCIe Gen5 (32 GB/s) | **16× PCIe Gen5 (64 GB/s), 1:1 host:accelerator** | **100 GB/s PCIe-based scale-out** (ServeTheHome, Hot Chips 38 — this is scale-out fabric bandwidth, not necessarily the host-CPU PCIe link; the two are not distinguished in the analyst source) | — | SoC chiplet with PCIe to host CPU |

*PCIe corrected 2026-08-08.* This repo previously recorded "MTIA v2: PCIe Gen5 ×16 (128 GB/s)". That was wrong on both lane count and bandwidth. The **ISCA 2025** MTIA v2 paper's spec table gives *Host connection: 8× PCIe Gen5 (32 GB/s)* for MTIA 2i and *8× PCIe Gen4 (16 GB/s)* for MTIA v1, and states that each accelerator module houses two MTIA 2i chips "connected via two 8× Gen5 PCIe links", with a PCIe switch linking six modules per server. **ISCA 2026** Table I independently lists MTIA-2i as 8× PCIe Gen5 (32 GB/s) — so the two papers agree with each other and only the repo was out of step. Separately, 128 GB/s corresponds to no Gen5 configuration Meta describes (Gen5 ×16 is ~63 GB/s per direction). MTIA 300's 16× Gen5 (64 GB/s) is a genuine doubling of host lanes over v2.

**MTIA 400 figures added 2026-09-13 are analyst-sourced (ServeTheHome, Hot Chips 38, 2026-08-25), not Meta-primary.** Meta's own 2026-08-24 engineering blog on MTIA 300 explicitly declines to name or specify MTIA 400 ("Meta's next-generation AI silicon"). Treat the MTIA 400 cells above accordingly until a Meta-primary source appears.

**Still not disclosed anywhere, primary or analyst:** the process node for MTIA 400 / 450 / 500; MTIA 400 and MTIA 500 absolute HBM capacity. Values circulating in aggregators ("2nm", "288 GB", "384/512 GB") are not published here. MTIA 300's process node moved from "not disclosed anywhere" to "analyst-sourced only" on 2026-09-13 (see table row above) — it is still not Meta-primary-confirmed.

## Compute

### Processing Element (PE)
- 2 RISC-V cores per PE (one with vector extension)
- Fixed-function units: MMA (matrix multiply-accumulate), DMA, nonlinear function
- 128 KB local SRAM per PE
- Asynchronous dataflow: RISC-V cores orchestrate; FFUs execute as dependencies resolve

### PE Grid
- 8×8 = 64 PEs
- Interconnected via NoC (v2: 3.3× bandwidth over v1)

### Processing Element — MTIA 300 (ISCA 2026)

MTIA 300 keeps the RISC-V + fixed-function PE concept but rebuilds the PE for training throughput:

- **Two 64 B-wide RISC-V vector cores** per PE (replacing the v1/v2 "one scalar + one vector" pairing)
- **Dot Product Engine**: two 32×64B×32 MAC tiles, **7.82 TFLOPS/PE** with FP16/BF16 inputs and FP32 output; also supports FP8 S1E4M3 / S1E5M2 and TF32
- **Reduction Engine**, **SIMD/SFU**, **Command Processor**, **Memory Layout Unit**, **Fabric Interface**, **Memory Bridge**
- **512 KB software-managed local memory**, split into circular buffers (v2/2i: 384 KB)
- Chip control core is a **RISC-V quad SMP core** (CPU-C)
- **INT8 GEMM removed, FP8 added.** SIMD width raised 32 → 128 elements/cycle, cutting the GEMM:SIMD ratio to **16:1** from MTIA 2i's 32:1

### PE Grid — MTIA 300

- **12 × 6 = 72 PEs**, plus **one redundant 1×6 PE row** for yield: each column tolerates one faulty PE, configured at boot, transparent to software and with no NoC performance impact
- **16 Message Engines** at the grid edges, adjacent to HBM, LLC and I/O
- Mesh NoC with **6-PE cluster routers**, **L-routing** (X then Y) and **virtual lanes** for deadlock avoidance
- **No memory crossbar** — a deliberate removal versus MTIA 2i

## Memory

| Level | v1 | v2 / 2i | **MTIA 300** |
|---|---|---|---|
| PE-local SRAM | 128 KB × 64 = 8 MB | **384 KB × 64 = 24 MB** (~1.0 TB/s R+W) | **512 KB × 72 = 36 MB** (1.9 TB/s R+W) |
| Shared on-chip SRAM | 128 MB | 256 MB | **192 MB LLC** — 96 MB and 48 banks *per side*, 96 banks total |
| SRAM BW | — | 2.7 TB/s (2i) | **11.4 TB/s R+W** (4.2× MTIA 2i) |
| Off-chip DRAM | 64 GB LPDDR5, 176 GB/s | 128 GB LPDDR5, 204.8 GB/s | **216 GB HBM3E @ 8.0 Gbps, 6 twelve-high stacks (3 per side), 6.1 TB/s read *or* write** |
| Measured off-chip BW | — | — | **5.57 TB/s (91% of peak)**, BF16-add kernel (H100 peak 2.4 TB/s, H200 peak 4.8 TB/s) |

**Unit qualifiers matter.** MTIA 300's 6.1 TB/s HBM figure is read-**or**-write; the 11.4 TB/s LLC and 1.9 TB/s local-memory figures are read**+**write. 36 GB per HBM stack is arithmetic (216 GB / 6), not a Meta-stated figure.

**Capacity went down, bandwidth went up.** MTIA 300 has *less* on-chip SRAM than MTIA 2i (192 MB vs 256 MB) but 4.2× the bandwidth — a deliberate rebalance away from the embedding-hot-row caching role the 2i SRAM played toward feeding a 560 TFLOP/s GEMM pipeline.

## Chip Physical

| | v1 | v2 | **MTIA 300** |
|---|---|---|---|
| Process | TSMC 7 nm | TSMC 5 nm | **Not disclosed** |
| Packaging | Monolithic | Monolithic | **2.5D CoWoS chiplet** |
| Die area | 373 mm² | 421 mm² | **Compute chiplet 25.6 × 31.4 mm (~804 mm², reticle-sized); 2 network chiplets 25.6 × 9.3 mm each** |
| Package | — | — | **77.5 × 77.5 mm, with a 50.3 × 51.9 mm 3.2×-reticle interposer** |
| Clock | 800 MHz | 1.35 GHz | **1.9 GHz** |
| Supply voltage | — | — | **0.85 V** |
| Instances (paper wording) | — | "2.35B gates, 103M FLOPS" (MTIA 2i) | **"7.2B gates, 511M FLOPS"** |
| TDP | 25 W | 90 W | **912 W (667 W typical)** |
| Cooling | Air | Air | **Liquid** |
| PCIe | 8× Gen4 (16 GB/s) | 8× Gen5 (32 GB/s) | **16× Gen5 (64 GB/s)** |

**On "511M FLOPS":** ISCA 2026 Table I prints "7.2B gates, 511M FLOPS" in its *Instances* row. In context "FLOPS" is the paper's abbreviation for **flip-flops** — an instance count, not a throughput figure. The paper's wording is quoted verbatim so the number stays traceable.

## Execution Model

- Each PE: asynchronous dataflow
- RISC-V cores generate a stream of instructions for FFUs
- DMA transfers and compute interleave as data becomes available
- Sparsity hardware (v2): structured sparsity for 2× throughput on sparse layers

## System Configuration

### MTIA v2 / 2i

- 2 chips per accelerator module → 220 W TDP
- 12 modules per chassis = 24 chips
- 3 chassis per rack = 72 chips total
- ~50 PetaFLOPS INT8 sparse per rack

### MTIA 300

- **Compute blade = 1 host CPU with 512 GB RAM + 1 MTIA 300** (1:1 host:accelerator, chosen to avoid PCIe contention; the H100/H200 testbeds in the same paper run 8 accelerators per host)
- **Chassis: 16 compute blade slots + 6 network blade slots**, connected by a pair of cable backplanes; not all network blades need be installed; compute blades are mounted vertically to shrink the backplane
- **Scale-up domain = 16 accelerators = 1 rack**, 800 GB/s per accelerator (option up to 1,000 GB/s); multiple racks can be combined into larger scale-up domains
- **Scale-out = 200 GB/s per accelerator**; 4,096-node first-level domain, "L2 unlimited" per Table I ("16K nodes or more if needed" in the text)
- Two network blade types: **scale-up** (ASIC selected for low latency and power) and **scale-out** (disaggregated scheduled fabric with packet spray, in-order delivery, fabric-level end-to-end credit)
- Both blade types **liquid-cooled**

### MTIA 400 (announced, lab-tested only)

- **72 MTIA 400 devices in one rack form a single scale-up domain** over a switched backplane (vs 16 for MTIA 300), with AALC / facility liquid cooling

## Design Trade-offs

- LPDDR5 over HBM (v1 → 2i): DLRM inference is not as BW-hungry as training; major cost saving. **Reversed at MTIA 300**, which moves to HBM3E for training and GenAI.
- Large SRAM: embedding table hot rows cached on-chip to reduce DRAM pressure. **MTIA 300 trades capacity for bandwidth** (192 MB @ 11.4 TB/s vs 256 MB @ 2.7 TB/s).
- RISC-V: customizable ISA, no vendor lock-in — retained and extended on MTIA 300 (two vector cores per PE, a scalar core per Message Engine, a quad SMP control core).
- **MTIA 300: network in the package.** Rather than pairing the accelerator with external NICs, Meta places 12 RoCE NICs on two in-package network chiplets and offloads collectives to on-die Message Engines — trading die area and 912 W of package power for a 1.2 TB/s network attach and 2.8 TB/s of reduction throughput.
- **MTIA 300: no memory crossbar.** Removed relative to MTIA 2i in favour of a cluster-router mesh NoC with L-routing.
- **MTIA 300: redundant PE row.** A 13th row of 6 PEs exists purely for yield on a reticle-sized compute die.

---

## MTIA v2 Update

*Updated 2026-04-05 from ISCA 2025 paper and Meta engineering blog.*

### Specification Corrections

Two v2 parameters recorded in the original tables above require correction per the ISCA 2025 paper:

| Parameter | Original (estimated) | Corrected |
|---|---|---|
| PE-local SRAM | 128 KB/PE | **384 KB/PE** (3× v1) |
| NoC bandwidth increase | 2× v1 | **3.3× v1** |
| PCIe interface | Gen5 ×8 | ~~Gen5 ×16 (128 GB/s)~~ → **re-corrected 2026-08-08 to 8× Gen5 (32 GB/s)**; the original ×8 reading was right |

### New v2 Hardware Details

**Host Interface additions (v2 only):**
- Dedicated **secure boot processor** (new vs v1)
- **Host-to-accelerator decompression engine** — decompresses model weights on-chip as they arrive over PCIe, increasing effective host-to-chip bandwidth for large embedding tables

**MTIA 2i (2025) — also called MTIA 200:**
- Shared on-chip SRAM: 256 MB at **2.7 TB/s** bandwidth (confirmed in production)
- Same physical die as MTIA v2; improvement reflects binning, configuration, and compiler changes
- Basis for the ISCA 2025 model-chip co-design study

**Production at scale:**
- 44% TCO reduction vs GPU inference (ISCA 2025, production recommendation workloads)
- 23 firmware-bundle fleet updates in 2024

### MTIA 300+ Architecture Shift (Announced March 2026)

Starting with MTIA 300, the architecture pivots significantly:

- **Memory: LPDDR5 → HBM**. MTIA 300: 216 GB HBM3E. MTIA 400 capacity **not disclosed**.
- **Modular chiplet design**: MTIA 400/450/500 share chassis/rack/network — drop-in upgrades
- **Training capability**: MTIA 300 is the first MTIA chip to support training (ranking/recommendation training workload)
- **Co-designer**: Broadcom (announced March 2026; partnership formalized by Meta newsroom PR 2026-04-14)
- **Cadence**: ~6-month generation cycle

| Chip | Memory | Key Compute Figure |
|---|---|---|
| MTIA 300 | 216 GB HBM3E, 6.1 TB/s (read or write) | Production (training); 1,120 TFLOP/s FP8, 560 TFLOP/s FP16/BF16 |
| MTIA 400 | +51% HBM BW vs 300 (≈9.2 TB/s, derived); **capacity not disclosed** | Lab testing (inference); +400% FP8 FLOPS vs 300 (≈5.6 PFLOP/s, derived); native MX8/MX4 |
| MTIA 450 | 2× HBM BW vs 400 | **+75% MX4 FLOPS vs MTIA 400** (Meta blog) |
| MTIA 500 | +50% BW and up to +80% capacity vs 450 | +43% MX4 FLOPS vs 450; 2×2 smaller compute chiplets |

*Corrected 2026-08-08:* the previous row "MTIA 450 — 6× MX4 FP16/BF16 vs MTIA 400" was a transcription error. Meta's 2026-03-11 blog says MTIA 450 raises MX4 FLOPS **by 75% over MTIA 400**; the "6×" figure is MX4 FLOPS relative to FP16/BF16 **on the same chip**. The previous "MTIA 400: 288 GB HBM" had no primary source and has been removed. Derived figures above are labelled as such — Meta publishes only relative deltas for 400/450/500.

---

## MTIA 300 Update — Networking, Collective Offload and System (2026-08-08)

*Added 2026-08-08 from "MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines", ISCA 2026 Industry Track Session 5B, 2026-06-30 (https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf). All figures Meta-stated.*

### Built-in NIC chiplets

MTIA 300's defining architectural feature is that the network interface is **inside the accelerator package**, not attached to it.

| Parameter | Value |
|---|---|
| Network chiplets | 2, each 25.6 × 9.3 mm |
| RDMA/RoCE IP blocks per chiplet | 6 custom **800 Gbps (100 GB/s)** blocks, based on third-party NIC IP |
| Bandwidth per network chiplet | 600 GB/s |
| Total NICs / bandwidth per accelerator | **12 NICs / 1.2 TB/s** |
| Die-to-die attach | 112G SerDes; **8 × 112G per NIC** (Fig. 4) |
| Outstanding work requests | **24,576 across 1,024 QPs per IP block** |
| Active QPs | **1,100 per NIC chiplet** (QP caching removed to save area) |

**"Express doorbells."** In a conventional RDMA NIC the doorbell write is a pointer update and the NIC must then read the work request from a ring buffer in HBM. MTIA 300 makes the **work request itself** the doorbell write, eliminating that read — Meta quotes ~**800 ns saved per transaction**. The packet-processing pipeline is also simplified relative to the stock IP.

Meta claims MTIA 300 is, to their knowledge, the **first accelerator with built-in NIC chiplets plus general-purpose collective offloading engines**.

### Message Engines (collective offload)

| Parameter | Value |
|---|---|
| Count | **16**, at the PE-grid edges next to HBM, LLC and I/O |
| Per-ME control | 1 scalar RISC-V core (**CPU-M**) |
| Per-ME context RAM | **256 KB** |
| Completion queues | One large shared CQ per ME |
| Near Memory Compute | **128 B/cycle** for reduction or DMA; **96 B/cycle** when all are concurrently active |
| Aggregate reduction throughput | **up to 2.8 TB/s** — over 2× the 1.2 TB/s I/O bandwidth |
| Area cost | **one-third of the chip area of the compute engines** |
| Collectives accelerated | Reduce, AllReduce, ReduceScatter |

The Message Engines are driven by **HCCL** (Meta's collective communications library): the control core CPU-C dispatches subgraphs of WQEs (SEND / RECV / WRITE / WAIT / SET / REDUCE) to the 16 MEs, which in turn drive the NIC chiplets through RDMA verbs. See the software-stack investigation for the full path.

### Measured system results — with the H100 caveat

| Result | Figure |
|---|---|
| Perf/TCO vs H100, ~150B-param production DLRM (99% embeddings), batch 10,240 on 24 MTIA 300 | **1.42×** |
| Same, batch 6,144 on 40 MTIA 300 | **1.39×** |
| Communication/collective time vs H100 | **3.9× better** |
| Embedding operators vs H100 | **up to 2.5×** |
| HBM bandwidth achieved, BF16-add kernel | **5.57 TB/s (91% of 6.1 TB/s peak)** |

⚠️ **The H100 baseline is a custom 500 W-power-capped configuration** delivering 780 TF/s BF16 rather than 1,000 TFLOPS at 700 W. Table III testbeds: MTIA 300 = 560 TF/s BF16 / 216 GB / 6.1 TB/s / 912 W accel / 1,500 W host / 1 accel per host / 16-accel scale-up @800 GB/s / 200 GB/s scale-out. H100 = 780 TF/s / 96 GB / 2.4 TB/s / 500 W / 6,500 W host / 8 accel per host / 8-accel scale-up @450 GB/s / 50 GB/s scale-out. H200 = 1,000 TF/s / 141 GB / 4.8 TB/s / 700 W / 8,850 W host. The power cap must accompany any citation of these ratios.

**LLM inference.** DeepSeek-R1 on vLLM under the InferenceMax benchmark, 8-accelerator TP8-TP8 and DP8-EP8, BF16 attention/KV-cache with FP8 MoE, short-prompt online scenario (input and output lengths drawn independently and uniformly from [0.8×1024, 1024] tokens), concurrency swept 4 → 256. Meta reports MTIA 300 "outperforms H200 overall", specifically **above concurrency 64** where execution is decode-dominant and HBM-bandwidth-bound; below that, small-message communication overhead versus NVLink costs it.

### Hardware limitations stated by Meta

- **MX4 and NVFP4 with row-wise/block-wise scaling are not natively supported** on MTIA 300 and fall back to RISC-V execution. Native MX4 arrives in MTIA 400.
- **Eager mode is host-bottlenecked** (Python interpreter, dynamic dispatch, device-host communication) and does not improve with faster silicon.
- Numerical-parity tooling for cross-platform convergence debugging "still require[s] time to mature".

### Scheduled disclosure — now realized (see Hot Chips 38 update below)

**Hot Chips 38, Session "AI 1", Tuesday 2026-08-25, 2:15–4:15 PM** — Meta, "Meta's Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI" (Srinagesh Loke, Cindy Chen, Jatinder Singh). This talk **has now happened** (as of 2026-09-13); see "Hot Chips 38 Update" section below for what it disclosed.

### Sources for this section

- https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf (primary — ISCA 2026 Industry Track)
- https://iscaconf.org/isca2026/program/
- https://www.computer.org/csdl/proceedings-article/isca/2026/506500b084/2iG0QI12UeY
- https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/ (Meta blog, 2026-03-11)
- https://about.fb.com/news/2026/04/meta-partners-with-broadcom-to-co-develop-custom-ai-silicon/ (2026-04-14)
- https://hotchips.org/program/conference/ (talk realized 2026-08-25; see below)

---

## Hot Chips 38 Update — MTIA 300 Production Blog + MTIA 400/450/500 Silicon Disclosure (2026-09-13)

*Primary source: Meta Engineering blog, "MTIA 300: Meta's First Training Chip with Built-in NICs and Communication-Offloading Engines" (Rajiv Krishnamurthy, Wes Bland), https://engineering.fb.com/2026/08/24/networking-traffic/mtia-300-meta-training-chip-built-in-nics/, 2026-08-24 — fetched and read in full. Secondary/analyst: ServeTheHome, "Meta's MTIA Custom AI Silicon at Hot Chips 2026", https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/, 2026-08-25 — used only where the Meta blog is silent (chiefly MTIA 400), and labeled as analyst-sourced throughout.*

### A. MTIA 300 — new measured production figures (Meta-primary)

| Figure | Value | Nature |
|---|---|---|
| HCCL achieved rack bandwidth | **up to 940 GB/s** within a 16-node rack | Measured; refines the ISCA 2026 "up to 1,000 GB/s option" ceiling already in this file with an achieved number |
| GEMM/collective isolation | **<0.5%** compute-throughput degradation running large GEMMs concurrently with collectives, vs. **>20%** on unnamed "traditional GPUs" | Measured, Meta-stated; GPU baseline unnamed |
| Communication time vs GPU cluster | **3.9× faster**, 150B-parameter production recommendation model, 40 accelerators | Matches the ISCA 2026 figure already recorded — not a new number |

### B. MTIA 300 — process node, now analyst-sourced (not yet Meta-primary)

ServeTheHome reports **"3nm" compute die + "5nm" I/O die** for MTIA 300, attributed to Hot Chips 38 slides. Meta's own engineering blog (2026-08-24) and the ISCA 2026 paper were both re-checked and **neither states a process node**. Recorded as analyst-sourced only — see the Generation Overview table above.

ServeTheHome also reports a vendor claim of **">1.8× vs GPU"** for forward+backward propagation, baseline unnamed — a distinct, vaguer claim from the ISCA 2026 paper's precise 1.42×/1.39× Perf/TCO-vs-500W-capped-H100 figures already in this file. Do not merge the two numbers.

### C. MTIA 400 — first hardware specifications (analyst-sourced, ServeTheHome / Hot Chips 38 only)

Meta's 2026-08-24 blog explicitly declines to name or specify MTIA 400. All figures below are ServeTheHome-only:

- **2 compute chiplets** (confirms the pre-existing repo entry)
- **8×6 grid of PEs**
- **8× HBM3e stacks, 9.4 TB/s** (supersedes the repo's prior *derived* "≈9.2 TB/s" with a directly reported number); capacity still not disclosed
- **12 PFLOP/s FP4** — a distinct figure from, not a replacement for, the repo's prior *derived* "≈5.6 PFLOP/s FP8"
- **15× MTIA 200's FP16 compute, 46× MTIA 200's DRAM bandwidth** — note the baseline here is MTIA 200 (2i), not MTIA 300, unlike other roadmap deltas in this repo
- **MXFP4 native hardware support** (confirms the repo's prior "native MX4 arrives in MTIA 400")
- **1.2 TB/s** Ethernet-based scale-up bandwidth
- **72 ASICs in a single scale-up domain** — independently corroborates the repo's pre-existing figure
- **667 W** TDP
- **100 GB/s** PCIe-based scale-out

### D. MTIA 450 / 500 — reconfirmed, no new figures

Hot Chips 38 reconfirms MTIA 450 as the GenAI-inference-focused variant and MTIA 500 as targeting scale-up "beyond just 72 ASICs in a single domain" — consistent with, adds no new numbers to, the existing roadmap table.

### E. Cadence

Hot Chips 38 reconfirms a roughly **6-month generation cadence "through 2027"**, slightly more specific than the 2026-04-05 baseline's open-ended "~6-month cadence".

### F. Cross-reference only — not an MTIA architectural finding

Meta's Aayush Ankit co-presented a **Hot Chips 38 Sunday tutorial** with d-Matrix's Sudeep Bhoja, "3D DRAM based Accelerator for Generative Inference." This is **d-Matrix technology**; recorded here only as a cross-reference for a future d-Matrix update pass, not as an MTIA hardware claim.

### G. Not confirmed this pass

- MTIA 400 process node, and any Meta-primary source for MTIA 400 specs generally.
- Any production-status change for MTIA 400 (still "lab-tested only" per the last confirmed status).

### Sources for this section (added 2026-09-13)

- https://engineering.fb.com/2026/08/24/networking-traffic/mtia-300-meta-training-chip-built-in-nics/ (primary — Meta Engineering, 2026-08-24)
- https://www.servethehome.com/metas-mtia-custom-ai-silicon-at-hot-chips-2026/ (analyst — ServeTheHome, Hot Chips 38, 2026-08-25; sole source for MTIA 400 figures)
- https://hotchips.org/program/conference/ (talk realized 2026-08-25)
