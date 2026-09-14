# Huawei Ascend Hardware Architecture

*as_of: 2026-09-13*
*Architectures: Da Vinci 1.0 (Ascend 910), Da Vinci 2.0 (Ascend 910B), Da Vinci 3.0 (Ascend 910C), Da Vinci 4.0 / "3rd-gen Da Vinci" (Ascend 950PR / 950DT)*

---

## Overview

Huawei Ascend NPUs are built around the **Da Vinci architecture** — a multi-engine heterogeneous AI core developed entirely in-house by HiSilicon. Unlike NVIDIA's SIMT streaming multiprocessors (which expose a general-purpose parallel programming model) or Google's TPU (deeply pipelined weight-stationary systolic array), each Da Vinci AI Core is a **five-unit co-processor** with a dedicated matrix engine (Cube), vector engine, scalar controller, and two independent DMA engines (MTE1/MTE2) executing concurrently under static compiler scheduling. The architecture is designed to maximize utilization of HBM bandwidth through an explicit software-managed memory hierarchy (L0A/L0B/L0C/L1/UB), with a large shared L2 SRAM pool providing 4 TB/s on-chip bandwidth.

---

## 1. Compute Engine

### Da Vinci AI Core — Five Parallel Units

| Unit | Function | Key Parameters |
|------|----------|---------------|
| **Cube Unit** | Systolic matrix multiply-accumulate | 16×16×16 FP16 per cycle = 4,096 MACs; 16×32×32 INT8 = 8,192 MACs |
| **Vector Unit** | Element-wise ops, activations, reductions | 128-lane FP16 or 256-lane INT8 per cycle |
| **Scalar Unit** | Loop control, branches, address calc | Micro-CPU; generates MTE strides |
| **MTE1** | Input DMA: HBM/L2 → L1 → L0A/L0B | Hardware transpose; double-buffer prefetch |
| **MTE2** | Output DMA: L0C → UB → L2 → HBM | Quantization/format conversion on drain |

All five units execute **concurrently** under static CCE (Core Compute Engine) compiler scheduling. There is no dynamic out-of-order execution; the compiler must explicitly pipeline Cube, Vector, and MTE operations to approach peak utilization. Double-buffering (load tile N+1 while computing tile N) is the primary throughput technique.

### Chip-Level Specifications by Generation

| Generation | Product | Process | AI Cores | Peak FP16 | Peak INT8 | HBM | HBM BW | TDP |
|------------|---------|---------|----------|-----------|-----------|-----|--------|-----|
| Da Vinci 1.0 | Ascend 910 | TSMC 7nm | 32 (4 clusters × 8) | 256 TFLOPS | 512 TOPS | 32 GB HBM2 | 1,228 GB/s | 310 W |
| Da Vinci 2.0 | Ascend 910B | SMIC N+1 (~7nm) | 25 | 320 TFLOPS | 640 TOPS | 64 GB HBM2e | ~800 GB/s | ~400 W |
| Da Vinci 3.0 | Ascend 910C | MCM: 2× SMIC N+1 | ~64 (32/die) | ~800 TFLOPS | ~1,600 TOPS | 128 GB HBM2e | ~1,600 GB/s | ~400 W |
| **Da Vinci 4.0** | **Ascend 950PR** (chip) | **not disclosed** (analyst inference: SMIC N+2/N+3, low confidence) | **not disclosed**; separated AIC/AIV, 1 Cube + 2 Vector per AI subsystem | **not disclosed** (single-source est. ~547 TFLOPS BF16/FP16) | **1,000 TFLOPS FP8/HiF8; 2,000 TFLOPS MXFP4** | **128 GB HiBL 1.0 (in-house)** | **1,600 GB/s** | **not disclosed** |
| **Da Vinci 4.0** | **Ascend 950PR** (Atlas 350 shipping card) | **not disclosed** | — | **not disclosed** | **1,560 TFLOPS FP4** (derated vs chip spec) | **up to 112 GB** | **1,400 GB/s** | **600 W** |
| **Da Vinci 4.0** | **Ascend 950DT** (chip) | **not disclosed** (analyst inference: SMIC N+3, low confidence) | **not disclosed**; separated AIC/AIV | **not disclosed** (single-source est. ~547 TFLOPS BF16/FP16) | **1,000 TFLOPS FP8/HiF8; 2,000 TFLOPS MXFP4** | **144 GB HiZQ 2.0 (in-house)** | **4,000 GB/s** | **not disclosed** |

Note: Ascend 910B has lower HBM bandwidth than the 910 despite larger capacity — a consequence of SMIC N+1 process I/O constraints vs. TSMC 7nm. Ascend 910C achieves +30–35% per-core throughput vs 910B via higher clock and microarchitectural improvements. Ascend 950 replaces external HBM with Huawei in-house memory (HiBL 1.0 / HiZQ 2.0) to circumvent export controls; the 950DT achieves 4 TB/s — ~2.5× the 910C dual-die bandwidth.

**2026-08-08 revisions to this table.** (i) Process node for the 950 series is **not disclosed** by Huawei; the previous "est. N+2/N+3" entries were analyst inference and are relabelled as such. (ii) The previous "~500 W / ~600 W (est.)" TDP entries were estimates with no vendor source and have been removed; the only disclosed 950-generation power figure is the **600 W Atlas 350 card** (Zhang Dixuan, 2026-03-23). (iii) The Atlas 350 shipping card is **derated versus the 950PR chip spec** — 1.56 PFLOPS FP4 / up to 112 GB / 1.4 TB/s — and now has its own row. (iv) Per-precision FP16/BF16 and TF32 rungs for the 950 are not published; a single third-party analysis gives ~547 TFLOPS BF16/FP16 and ~273 TFLOPS TF32, recorded as single-source estimates only. Independent sources give only the rounded 1 PFLOPS FP8 / 2 PFLOPS MXFP4, corroborated by Huawei's own 1,024-card = 1 EFLOPS FP8 arithmetic.

---

## 2. Data Path

The Da Vinci core executes a **static 5-stream pipeline**: the CCE compiler emits separate instruction sequences for Cube, Vector, Scalar, MTE1, and MTE2, with explicit synchronization barriers between streams (e.g., "MTE1 done before Cube starts"). This is fundamentally different from NVIDIA's dynamic warp scheduler or TPU's lock-step systolic flow.

**Key data flow pattern for GEMM:**
1. MTE1 fetches tile of input A from L2/HBM → L1 → L0A
2. MTE1 fetches tile of weight B from L2/HBM → L1 → L0B
3. Cube Unit reads L0A × L0B, accumulates into L0C (FP32)
4. MTE2 drains L0C → Unified Buffer (UB)
5. Vector Unit applies activation/bias from UB
6. MTE2 writes UB result → L2 → HBM

Steps 1–2 for tile N+1 overlap with steps 3–6 for tile N (double-buffering).

---

## 3. On-chip Memory

### Per-Core Memory (Da Vinci 1.0 / Ascend 910 reference values)

| Level | Size | Purpose | Accessed By |
|-------|------|---------|------------|
| L0A | 16 KB | Cube left-operand buffer | MTE1 (write), Cube (read) |
| L0B | 16 KB | Cube right-operand buffer (weights) | MTE1 (write), Cube (read) |
| L0C | 32 KB | Cube accumulator (FP32) | Cube (write), MTE2 (read) |
| L1  | 512 KB | Input feature staging | MTE1 (DMA target from L2) |
| UB  | ~256 KB | Unified Buffer (activations, outputs) | Vector, MTE2, Scalar |

### Shared On-chip Memory (all cores)

| Level | Size | Bandwidth | Notes |
|-------|------|-----------|-------|
| L2 | 32 MB total | 4 TB/s | NoC mesh; 1024-bit bus per core @ 2 GHz |
| Total SRAM (Ascend 910) | 84 MB | — | All per-core buffers + L2 combined |
| **L2 (Ascend 950, new level)** | **128 MB global** | **not disclosed** | **New in the 950 generation — no equivalent on 910B/910C. Spans the whole chip (both AI dies). ~2× per-access improvement; supports per-way cache lock / residency policy.** |

The 4 TB/s L2 bandwidth is ~3.3× the 1.2 TB/s HBM bandwidth of the 910, making L2 reuse critical for performance.

**Ascend 950 (2026-08-08).** The 128 MB global L2 is the one on-chip memory change that is independently corroborated for the 950 generation. Per-core buffer sizes for the 950 (L0A/L0B, L1, UB) are **not disclosed**; a single third-party whitepaper analysis reports L0A/L0B at 64 KB each, L1 at 512 KB, and UB at 512 KB, plus a sector cache with 512 B lines split into 4 × 128 B sectors — none of that is independently confirmed and it must not be cited as spec. The per-core values in the table above remain **Da Vinci 1.0 / Ascend 910 reference values**, not 950 values.

---

## 4. Off-chip Memory (HBM)

| Product | HBM Type | Capacity | Bandwidth | Stacks | Notes |
|---------|----------|----------|-----------|--------|-------|
| Ascend 910 | HBM2 | 32 GB | 1,228 GB/s | 4 | TSMC 7nm; peak generation BW |
| Ascend 910B | HBM2e | 64 GB | ~800 GB/s | 4 | SMIC N+1; reduced PHY BW |
| Ascend 910C | HBM2e | 128 GB | ~1,600 GB/s | 8 (dual-die) | 2× 910B dies; additive BW |
| Ascend 950PR (chip) | HiBL 1.0 (Huawei in-house) | 128 GB | 1,600 GB/s | not disclosed | Q1 2026; export-control-independent; cost-optimized vs HBM3E |
| **Atlas 350 card (950PR)** | **HiBL 1.0** | **up to 112 GB** | **1,400 GB/s** | not disclosed | **Shipping-card spec, derated vs the chip spec above (Zhang Dixuan, 2026-03-23). 600 W TDP.** |
| Ascend 950DT (chip) | HiZQ 2.0 (Huawei in-house) | 144 GB | 4,000 GB/s | not disclosed | Q4 2026 official release; Huawei Cloud deployment announced for Aug 2026 (not confirmed live as of 2026-08-08); 2.5× 910C BW; training/decode optimized |

---

## 5. Host Interface / Package

### Ascend 910 / 910B
- **PCIe 4.0 × 16**: ~64 GB/s bidirectional host interface
- **Die**: monolithic TSMC 7nm (910) / SMIC N+1 (910B)
- Companion CPU die: ARM-based AICPUs (4 cores) for host-side control

### Ascend 910C (Multi-Die Package)
- **Package**: organic substrate MCM (not silicon interposer)
  - 2× SMIC N+1 NPU compute dies
  - 1× TSMC 7nm CPU companion die (vintage 2020, Nimbus v3)
  - 8× HBM2e stacks (4 per NPU die)
- **Host interface**: PCIe 4.0 × 16
- Die-to-die interconnect: proprietary on-substrate links between NPU dies
- TDP: ~400 W; OAM-compatible form factor for Atlas 800T server

### Ascend 950 (added 2026-08-08)
- **Host interface: PCIe 5.0 × 16** — an upgrade from PCIe 4.0 × 16 on 910/910B/910C
- **Dual 400 Gbps UBoE** (UnifiedBus over Ethernet) ports on-package
- **Dual AI die per chip**; die-to-die and inter-chip links run over HiLink SerDes
- **Atlas 350 card**: 600 W TDP (the only disclosed 950-generation power figure)
- Package construction, die area, and process node: **not disclosed**

---

## 6. Scale-up Interconnect

### HCCS (Huawei Compute Communication System)
- Huawei's proprietary intra-server high-speed NPU-to-NPU bus
- Analogous role to NVIDIA NVLink (intra-node)
- Atlas 800T server: 8× Ascend 910/910B/910C interconnected via HCCS
- Delivers higher bandwidth than PCIe switch topology; enables NPU affinity scheduling in Kubernetes (CCE)
- **HCCS 4.0** (originally reported as planned for Ascend 910D): targeting 100,000-chip cluster-scale connectivity. **Superseded — see UnifiedBus below.**

### UnifiedBus / 灵衢 (Lingqu) 2.0 — the 950-generation scale-up fabric

*Added 2026-08-08. Launched by Huawei rotating chairman Xu Zhijun on 2025-09-18 at Huawei Connect 2025 — i.e. this predates the repo's 2026-04-05 baseline and was missed at the time, not new in this window.*

- UnifiedBus (Chinese: 灵衢, Lingqu) replaces the HCCS branding for the Ascend 950 generation's scale-up fabric.
- Published by Huawei as an **open specification** — base spec, firmware spec, and software reference designs at `unifiedbus.com`; openEuler hosts a *UB Service Core* software architecture reference design.
- **~2 TB/s per-chip** inter-chip bandwidth over HiLink SerDes (independently corroborated round figure).
- **Dual 400 Gbps UBoE** (UnifiedBus over Ethernet) for the Ethernet-carried variant.
- The Ascend 950 whitepaper claims support for clusters **exceeding 128K cards**, superseding the older "HCCS 4.0 → 100,000-chip" figure.
- **3 μs RTT** measured over the Lingqu protocol in the WAIC 2026 Atlas 950 SuperPoD demonstration.
- **Not independently confirmed** (single third-party analysis only): a 2,016 GB/s bidirectional per-chip figure decomposed into RTP (reliable transport, 4 ports, 448 GB/s dual) and CTP (light transport, 9 ports, 1,008 GB/s dual) over 18 × X4 HiLink ports at 112 Gbps/lane. Independent reporting gives only the round ~2 TB/s.

### CloudMatrix 384 (Ascend 910C system-scale)

| Spec | Value |
|------|-------|
| Total chips | 384 Ascend 910C |
| Physical racks | 16 (12 compute + 4 switch racks) |
| Chips per compute rack | 32 (4 servers × 8 chips) |
| Scale-up links per server | 56 × 400G OSFP SiPh LPO |
| Per-chip scale-up BW | 2.8 Tbps |
| Total optical transceivers | 6,912 × 400G |
| Internal topology | Full-mesh all-to-all optical |
| Link technology | Silicon photonic Linear Pluggable Optics (SiPh LPO) |

CloudMatrix 384 is Huawei's answer to NVIDIA's GB200 NVL72 — it achieves comparable or higher aggregate compute with ~4× the power consumption, relying entirely on optical links for all intra- and inter-rack communication. There is no copper backplane.

---

## 7. Scale-out Interconnect

### CloudMatrix 384 Scale-out

| Spec | Value |
|------|-------|
| Scale-out links per server | 8 × 400G OSFP SiPh LPO |
| Total fiber count | 3,168 fibers |
| Topology | Full-mesh optical between pods |

### Atlas 900 Cluster (Ascend 910)
- HCCS (intra-server) + PCIe 4.0 + 100 GE RoCE v2 (inter-server)
- 100 TB/s dedicated full-mesh synchronization fabric across the cluster
- Scale: 1,024 Ascend 910 chips per Atlas 900 cluster; claimed world's fastest AI training cluster at launch (2019)

---

## 8. Ascend 950 — Da Vinci 4.0 Architecture

*Announced HC 2025 (September 2025); 950PR launched Q1 2026 in Atlas 350 and Atlas 950 SuperPoD. Microarchitecture detail added 2026-08-08 from the Ascend 950 NPU architecture whitepaper.*

**Naming:** Huawei's own whitepaper calls this the **3rd-generation Da Vinci** architecture; press coverage and this document call it **Da Vinci 4.0**. Same silicon, two numbering conventions; Huawei has published no reconciliation.

### 8.1 Architectural Evolution: Da Vinci 4.0

Da Vinci 4.0 introduces a **SIMD+SIMT hybrid execution model**, extending the existing five-unit core with:

- **New Cube Unit precision**: FP8, MXFP8, HiF8 (Huawei-proprietary 8-bit), MXFP4 — enabling 1 PFLOPS FP8 and 2 PFLOPS MXFP4 per chip
- **Enhanced Vector Unit**: upgraded to SIMT-style execution alongside legacy SIMD, improving irregular workload throughput
- **Scale-up interconnect**: 2.5× bandwidth increase to **2 TB/s** (from 910C HCCS intra-chip); the fabric is branded UnifiedBus / 灵衢 2.0 (§6)
- **In-house memory**: replaces foreign HBM with proprietary HiBL 1.0 (950PR) and HiZQ 2.0 (950DT)

### 8.1a Whitepaper Microarchitecture (added 2026-08-08)

Huawei published a ~38-page *昇腾950 NPU架构白皮书* on its OBS public-download endpoint. Dating evidence: the PDF carries `Last-Modified: 2026-06-04`, Chinese trade press covered it on 2026-06-11, and an independent analyst digest is dated 2026-05-22 — **late May / early June 2026**. The PDF is **image-only**, so its contents reach this survey via analyst readings rather than extracted vendor text. The confidence split below reflects that.

**Independently corroborated:**

| Feature | Detail |
|---|---|
| Core split | **Cube and Vector become independent cores** — AIC (Cube) and AIV (Vector) — breaking the 910-era five-units-in-one-core design |
| Core ratio | **1 Cube + 2 Vector per AI subsystem** |
| Global L2 | **128 MB**, chip-wide, spanning both AI dies; new level absent on 910B/910C; ~2× per-access improvement; per-way cache lock / residency policy |
| Hardware scheduler | **STARS 2.0**, arbitrating across AIC / AIV / CPU / DVPP / SDMA / UB / CCU |
| On-chip data movement | **NDDMA** as a lightweight AIC↔AIV data path |
| Memory | 128 GB / 1.6 TB/s (950PR); 144 GB / 4 TB/s (950DT) |
| Inter-chip | ~2 TB/s over HiLink SerDes |
| Host / network | PCIe 5.0 × 16; dual 400 Gbps UBoE |

**NOT independently confirmed — single third-party analysis only. Do not cite as spec:**

- 18 Da Vinci cores per AI die → 36 Cube + 72 Vector cores per chip (950DT)
- L0A / L0B at 64 KB each, L1 at 512 KB, Unified Buffer at 512 KB *(the repo's 16 KB L0A/L0B figure is the 910-era value and remains correct for that generation)*
- Sector cache with 512 B lines split into 4 × 128 B sectors
- **BufferID-based synchronization** replacing the `EnQue` / `DeQue` inter-pipeline handshake — if confirmed this would change the core AscendC idiom, so it is called out here for future verification
- NDDMA as an *N-Dimensional Data Movement Accelerator* fusing movement + layout conversion + address generation into single instructions with up to 5D transforms (the *existence* of NDDMA is corroborated; this fuller description is not)
- A **"Linx816"** on-chip AI CPU with 4 clusters × 2 ARMv8-A cores, dual-threaded, 4 MB L3 per cluster
- The UnifiedBus RTP/CTP port decomposition (see §6)

**Per-precision compute ladder.** Huawei discloses only the rounded **1 PFLOPS FP8 / 2 PFLOPS MXFP4** per chip, which its own 1,024-card = 1 EFLOPS FP8 figure corroborates. A single third-party analysis additionally gives MXFP4 2,007 / HiF8 ~1,034 / FP8 1,034 / BF16-FP16 547 / TF32 ~273 TFLOPS. The BF16/FP16 and TF32 rungs are **not disclosed** by Huawei and are recorded here only as single-source estimates.

### 8.2 Ascend 950PR Specifications

Optimized for compute-dense, memory-light workloads (prefill inference, recommendation engines):

| Spec | Value |
|------|-------|
| Memory | HiBL 1.0 (Huawei in-house) — 128 GB |
| Memory bandwidth | 1.6 TB/s |
| FP8 / HiF8 peak | 1 PFLOPS |
| MXFP4 peak | 2 PFLOPS |
| Scale-up interconnect | 2 TB/s (UnifiedBus / 灵衢 2.0) |
| Card form factor | Atlas 350 (OAM-compatible) |
| **Atlas 350 shipping-card spec** | **1.56 PFLOPS FP4 · up to 112 GB · 1.4 TB/s · 600 W** (Zhang Dixuan, 2026-03-23) — **derated vs the chip spec in this table** |
| vs. NVIDIA H20 | "~2.8× FP4 performance" — **Huawei marketing comparison**, not an independently confirmed benchmark. Power comparison stated as ~1.5× H20 TDP. |
| SuperPoD config | 8,192 **Ascend 950DT** chips → 16 EFLOPS FP4 (Atlas 950 SuperPoD, MWC 2026) |

**HiBL 1.0**: Huawei's cost-optimized in-house HBM alternative. Lower peak bandwidth than HiZQ 2.0 / HBM3E, but sufficient for compute-bound prefill. Decouples supply chain from US export controls on HBM from SK Hynix, Samsung, Micron.

### 8.3 Ascend 950DT Specifications

Optimized for memory-bandwidth-sensitive workloads (LLM training, autoregressive decode):

| Spec | Value |
|------|-------|
| Memory | HiZQ 2.0 (Huawei in-house) — 144 GB |
| Memory bandwidth | 4 TB/s |
| FP8 / HiF8 peak | 1 PFLOPS |
| MXFP4 peak | 2 PFLOPS |
| Scale-up interconnect | 2 TB/s (UnifiedBus / 灵衢 2.0) |
| vs. 910C (dual-die) | ~2.5× memory bandwidth |
| **Availability (as of 2026-08-08)** | **Q4 2026 remains the official commercial release.** Huawei VP Chen Lin announced at Huawei Cloud 2026 INSPIRE (reported 2026-06-06/08) that Huawei Cloud deployment would be pulled forward to **August 2026** — this is an **announced schedule**, and no confirmation of a live deployment was found as of 2026-08-08. |
| **Early adopter** | DeepSeek named in press reporting (DeepSeek V4 already runs on the Ascend 950 platform); **not a Huawei statement** |
| **Demand signal (added 2026-09-13)** | Bloomberg reported 2026-09-04 that DeepSeek plans to deploy **≥160,000 Ascend 950DT** accelerators (inference-only) at a new Inner Mongolia data center — press-reported, not vendor-confirmed; Huawei reportedly cannot fill the order for over a year (HBM supply constraints). Separately, Huawei Central (2026-09-10, citing Bloomberg) reports a **~60% price increase** (press-reported ¥150,653 → ¥250,000) attributed to demand and HBM cost. Neither report changes the spec table above or confirms the "August 2026" Huawei Cloud deployment pull-forward went live; Q4 2026 remains the official commercial-release date. |

**HiZQ 2.0**: Huawei's high-performance in-house HBM. 4 TB/s closes the gap with HBM3E-class products. Critical for decode-phase LLM inference where memory bandwidth is the primary bottleneck.

### 8.4 Atlas 950 SuperPoD and CloudMatrix Evolution

| System | Chips | Scale-up BW/chip | System FP4 | Launch |
|--------|-------|-----------------|-----------|--------|
| CloudMatrix 384 | 384 Ascend 910C | 2.8 Tbps (SiPh LPO) | ~0.6 EFLOPS | 2024–2025 |
| Atlas 950 SuperPoD (full scale) | 8,192 **Ascend 950DT** | 2 TB/s (UnifiedBus) | 16 EFLOPS | Q4 2026 (announced MWC 2026) |
| **Atlas 950 SuperPoD (WAIC 2026 demo)** | **1,024 cards** | **2 TB/s (UnifiedBus)** | **2 EFLOPS FP4 (1 EFLOPS FP8)** | **Demonstrated 2026-07-17 — not shipping** |
| **Atlas 850E (air-cooled variant)** | **96 cards commercially deployed; scales 8 → 1,024 NPUs** | not disclosed | not disclosed | **Announced MWC 2026** |
| Atlas 960 SuperPoD | 15,488 Ascend 960 | TBD | ~32 EFLOPS (est.) | Q4 2027 |

The full-scale Atlas 950 SuperPoD announced at MWC 2026 uses 8,192 Ascend 950DT chips, packaged at **64 NPUs per cabinet**. Huawei's roadmap targets **1 million-chip (百万卡) supernode clusters** by 2027.

#### WAIC 2026 demonstration (added 2026-08-08)

At WAIC 2026 (Shanghai, **2026-07-17 to 07-20**) Huawei showed Atlas 950 SuperPoD hardware publicly for the first time — its release of 2026-07-17 labels the showing 真机首次公开亮相 ("first public physical appearance"), and the system won the conference SAIL award.

| Demonstrated configuration | Value |
|---|---|
| Cards | 1,024 |
| Compute | 1 EFLOPS FP8 / 2 EFLOPS FP4 |
| Globally unified memory address space | 256 TB |
| Fabric RTT | 3 μs over the 灵衢 (Lingqu) protocol |
| Status | **DEMONSTRATED, not shipping** — commercial delivery remains Q4 2026 |

The same release records **750+ commercial deployments** of the older Atlas 384-series supernodes.

### 8.5 Process Node

Huawei has not officially disclosed the Ascend 950 process node — it is **not disclosed**, and remained so as of 2026-08-08. What exists is analyst inference only:
- Analyst inference (**low confidence, never vendor-stated**): an SMIC advanced node, variously given as N+2 or N+3, approximately 5–7 nm equivalent. TrendForce's June 2026 reporting on the 950DT repeats "likely SMIC N+3" on the same inferential basis.
- The move to proprietary HiBL 1.0 / HiZQ 2.0 memory eliminates the primary supply-chain exposure to Western export controls
- Huawei planned to ship ~1.6 million Ascend dies across all models in 2026

### 8.6 Forward Look: Ascend 960 (2027) and 970 (2028)

- **Ascend 960 (Q4 2027)**: doubles compute, memory, and interconnect vs 950; introduces **HiF4** (Huawei-proprietary 4-bit format, claimed higher accuracy than standard FP4); Atlas 960 SuperPoD at 15,488 chips
- **Ascend 970 (2028)**: fleet-level target of **4 ZettaFLOPS FP4**; system milestone rather than single-chip spec

---

## Sources
- [DaVinci: A Scalable Architecture (CMC PDF)](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf)
- [DaVinci Architecture (Semantic Scholar)](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf)
- [Ascend AI Processor Architecture — O'Reilly Ch.3](https://www.oreilly.com/library/view/ascend-ai-processor/9780128234891/B9780128234884000035.xhtml)
- [Performance Modeling on DaVinci AI Core](https://www.sciencedirect.com/science/article/abs/pii/S074373152300014X)
- [Huawei Ascend 910B Examined — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)
- [Huawei Ascend 910C — Awesome Agents](https://awesomeagents.ai/hardware/huawei-ascend-910c/)
- [CloudMatrix 384 — SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72)
- [CloudMatrix 384 LPO Optics — QSFPTEK](https://www.qsfptek.com/qt-news/400g-osfp-siph-lpos-in-huawei-ai-cloudmatrix384-super-node.html)
- [TechInsights 910C Teardown — SemiWiki](https://semiwiki.com/forum/threads/techinsights-teardown-huawei-ascend-910c-still-contains-cpu-dies-from-tsmc-from-2020.23737/)
- [ServeTheHome Ascend 910 Review](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/)
- [Atlas 900 Cluster — CIO](https://www.cio.com/article/217641/interpretation-on-supreme-computing-of-huawei-atlas-900-ai-cluster.html)
- [Huawei Ascend NPU Roadmap Examined — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-npu-roadmap-examined-company-targets-4-zettaflops-fp4-performance-by-2028-amid-manufacturing-constraints)
- [Huawei Reveals 3-Year Ascend Roadmap, 950 Coming 2026 — Huawei Central](https://www.huaweicentral.com/huawei-reveals-3-year-ascend-ai-chip-roadmap-950-coming-in-2026/)
- [Huawei Unveils Ascend 950 with In-House HBM — TrendForce](https://www.trendforce.com/news/2025/09/18/news-huawei-unveils-ascend-950-with-in-house-hbm-in-2026-touts-superpod-to-rival-nvidia/)
- [Huawei Debuts Atlas 350 on Ascend 950PR, 2.8× H20 — TrendForce](https://www.trendforce.com/news/2026/03/23/news-huawei-debuts-atlas-350-on-ascend-950pr-with-in-house-hbm-touting-2-8x-h20-performance/)
- [Atlas 950 AI Supercomputer at MWC 2026 — Technetbook](https://www.technetbooks.com/2026/03/huawei-atlas-950-ai-supercomputer.html)
- [Ascend 950PR Challenges NVIDIA — Nerd Level Tech](https://nerdleveltech.com/huawei-ascend-950pr-atlas-350-ai-chip-challenges-nvidia)
- [Huawei Ascend Roadmap Announcement (HC 2025) — DataCenter Dynamics](https://www.datacenterdynamics.com/en/news/huawei-announces-annual-release-cadence-for-three-new-ascend-ai-chips-unveils-supernode-offering-company-says-will-outperform-nvidias-nvl144/)

### Added 2026-08-08
- [Ascend 950 NPU Architecture Whitepaper (PDF, ~38 pp, image-only; `Last-Modified: 2026-06-04`)](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — primary source; contents not directly extractable
- [Ascend 950 NPU whitepaper analysis (third-party, single-source)](https://pillumina.github.io/posts/aiinfra/ascend-950-npu/) — source of the uncorroborated microarchitecture, buffer-size, and RTP/CTP figures explicitly flagged as unconfirmed above
- [Atlas 950 SuperPoD at WAIC 2026 — Huawei (2026-07-17)](https://www.huawei.com/cn/news/2026/7/atlas-950-superpod)
- [SuperPoD announcements at MWC 2026 — Huawei](https://www.huawei.com/en/news/2026/3/mwc-superpod-ai)
- [UnifiedBus open specification](https://www.unifiedbus.com/en) — base spec, firmware spec, SW reference designs (launched 2025-09-18, Huawei Connect 2025)
- [openEuler UB Service Core SW Architecture Reference Design (PDF)](https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf)
- [Ascend 950DT deployment pulled forward to August — TrendForce (2026-06-08)](https://www.trendforce.com/news/2026/06/08/news-huawei-brings-forward-ascend-950dt-deployment-to-august-deepseek-v4-2-seen-as-potential-early-adopter/)
- [Huawei confirms Ascend 950DT to debut in August — Huawei Central](https://www.huaweicentral.com/huawei-confirms-ascend-950dt-ai-chip-to-debut-in-august/)
- [Atlas 350 unveiled: 1.56 PFLOPS FP4, up to 112 GB — Tom's Hardware (2026-03-24)](https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20)

### Added 2026-09-13
- [Bloomberg — DeepSeek plans ≥160,000 Ascend 950DT order (2026-09-04)](https://www.bloomberg.com/news/articles/2026-09-04/deepseek-plans-big-huawei-ai-chip-order-to-power-new-data-center)
- [The Decoder — DeepSeek's Huawei chip cluster (2026-09-04)](https://the-decoder.com/deepseek-plans-the-largest-known-huawei-chip-cluster-with-160000-processors-in-inner-mongolia/)
- [Huawei Central — Ascend 950DT price jumped 60% (2026-09-10)](https://www.huaweicentral.com/huawei-ascend-950dt-price-jumped-60-over-past-three-months/)
- [Huawei Connect 2026 event page](https://www.huawei.com/en/events/huaweiconnect) — 2026-09-17/19, Shanghai; not yet held as of this scan
- [MindSpore release history — PyPI](https://pypi.org/project/mindspore/#history) — 2.10.0, 2026-07-31 (pre-window correction)
