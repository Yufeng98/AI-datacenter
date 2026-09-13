# Tenstorrent Hardware Architecture Investigation

*as_of: 2026-04-05*
*devices: Wormhole (n150/n300), Blackhole (p150/p300/Galaxy)*
*sources: official specs, HotChips 2024 presentation, ASPLOS microbenchmarking paper, corsix deep-dives*

> **Superseded in part — see the "Investigation Update — 2026-08-08" section appended at the end of this file.** The Blackhole "140 Tensix cores / 745 TOPS FP8" figures below are full-die numbers, not shipping-SKU numbers; the Galaxy figures below (~24 PFLOPS, ~6.7 GB SRAM, 1 TBps aggregate) are superseded by the GA spec sheet. The 2026-04-05 content is retained as architectural history.

---

## Overview

Tenstorrent's hardware architecture centers on the **Tensix Processor** — a 2D mesh of specialized compute tiles called **Tensix cores**, connected by a dual **Network-on-Chip (NoC)** with torus topology, surrounding GDDR memory controllers, Ethernet ports, and PCIe connectivity. The defining architectural choice is **explicit data movement with no hardware caches** above each Tensix core's local 1.5 MB SRAM.

Two current datacenter-class chips:
- **Wormhole** (n150/n300): 80 Tensix cores, 12 GB GDDR6, 328 TOPS FP8, 12nm GlobalFoundries
- **Blackhole** (p150/p300): 140 Tensix cores, 32 GB GDDR6, 745 TOPS FP8, 6nm TSMC

---

## 1. Compute Engine: Tensix Core

Each Tensix core is an autonomous compute node containing:

| Component | Wormhole | Blackhole |
|-----------|----------|-----------|
| Baby RISC-V cores | 5 | 5 |
| L1 SRAM | 1.5 MB | 1.5 MB |
| Matrix unit (FPU) | BF16/FP16/FP8/INT8 | BF16/FP16/FP8/INT8 |
| Vector unit (SFPU) | 32-lane SIMD | 32-lane SIMD |
| NoC routers | 2 (NoC 0 + NoC 1) | 2 (NoC 0 + NoC 1) |
| Supported tile formats | BF16, FP16, FP32, INT8, FP8 | BF16, FP16, FP32, INT8, FP8 |

### The Five Baby RISC-V Cores

Each Tensix core contains exactly 5 "Baby" RISC-V CPUs — minimal, in-order, scalar RISC processors (no superscalar, no branch prediction, no vector extensions):

1. **BRISC** (Broadcast RISC / Data Movement 0): Issues NoC 0 read transfers; runs the reader kernel
2. **NCRISC** (NoC RISC / Data Movement 1): Issues NoC 1 write transfers; runs the writer kernel
3. **TRISC0** (Tensix RISC 0 — Unpack): Drives the unpack unit; copies L1 tile data → FPU source registers
4. **TRISC1** (Tensix RISC 1 — Math): Drives the matrix/vector FPU; issues MVM, GEMM, eltwise instructions
5. **TRISC2** (Tensix RISC 2 — Pack): Drives the pack unit; copies FPU destination registers → L1 circular buffers

The three TRISC cores operate as a producer-consumer pipeline (TRISC0 → TRISC1 → TRISC2) synchronized by hardware semaphores on the FPU destination register file. The two data movement cores (BRISC, NCRISC) run independently, overlapping with compute via circular buffer synchronization.

### Matrix Unit (FPU)

- Operates on **32×32 tiles** natively (native tile unit for all operations)
- Supports: MVMUL (matrix-vector multiply), GMPOOL (generalized matrix pool), ELWMUL (elementwise multiply)
- Precision stack: BF16, FP16, FP8 (E4M3 / E5M2), INT8, INT32, FP32
- Instruction references: WormholeB0 MatrixUnit.md in tt-isa-documentation

### Vector Unit (SFPU)

- **32-lane SIMD** operating in parallel
- Supports: Exp, Log, Sqrt, Recip, Tanh, GeLU, ReLU, Sigmoid, and custom-defined SFPUs
- Used for elementwise nonlinear activations and custom fused operations
- Instruction reference: WormholeB0 VectorUnit.md in tt-isa-documentation

### Chip-Level Core Counts

| SKU | Tensix Cores | Big RISC-V Cores | On-chip SRAM | Peak FP8 |
|-----|-------------|-----------------|--------------|---------|
| Wormhole n150 | 72 active | 0 | 108 MB | 262 TOPS |
| Wormhole n300 (dual) | 80 × 2 | 0 | 240 MB | ~656 TOPS |
| Blackhole p150 | 140 | 16 (host CPU) | 210 MB | 745 TOPS |
| Blackhole p300 (dual) | 140 × 2 | 16 × 2 | 420 MB | ~1.5 POPS |

Blackhole uniquely includes **16 "Big RISC-V" 64-bit, dual-issue, in-order CPU cores** (4 clusters of 4) capable of running Linux directly on-chip, making it a standalone AI computer without a separate host CPU.

---

## 2. NoC Mesh: Data Path

### Dual NoC Torus Topology

The chip is organized as a 2D mesh with **wraparound connections** (torus), providing full connectivity:

- **NoC 0**: Unidirectional east + south flow
- **NoC 1**: Unidirectional west + north flow

The two NoCs together provide quasi-full-duplex operation: simultaneous reads (NoC 0) and writes (NoC 1) to/from different locations without contention.

### Physical Parameters

| Parameter | Wormhole | Blackhole |
|-----------|----------|-----------|
| Grid dimensions (total tiles) | 10×12 = 120 | approx 12×14 |
| Active Tensix cores | 80 (72 accessible in n150) | 140 |
| Link width | 32 bytes/cycle | 32 bytes/cycle |
| Topology | 2D torus, wraparound | 2D torus, wraparound |
| NoC frequency | ~1 GHz (GFP 12nm) | Higher (TSMC 6nm) |

### NoC Tile Types (Wormhole)

On Wormhole, the 10×12 NoC grid contains:
- **T (Tensix)**: 80 compute tiles
- **D (DRAM)**: 6 controller tiles (each with 2 channels × 1 GB GDDR6)
- **E (Ethernet)**: 16 Ethernet tiles (100 Gbps each)
- **A (ARC/management)**: 1 management tile (on-chip management processor)
- **P (PCIe)**: 1 PCIe bridge tile

### Explicit Data Movement

**There is no hardware cache coherence** between Tensix cores. All data transfers are explicit NoC operations initiated by the BRISC/NCRISC cores:
- `noc_async_read`: DMA from DRAM or remote L1 into local L1
- `noc_async_write`: DMA from local L1 into DRAM or remote L1
- No implicit cache lines, no cache invalidation, no TLB misses

This is architecturally analogous to a distributed memory system on a single chip. The programmer (or compiler) must explicitly orchestrate all data movement, trading hardware cache complexity for software-controlled bandwidth predictability.

---

## 3. On-chip Memory (SRAM)

Each Tensix core has **1.5 MB local SRAM** (called "L1" in Metalium terminology), used for:
- **Circular buffers**: Producer-consumer inter-kernel data exchange
- **L1 interleaved buffers**: Tensors distributed across multiple Tensix L1s for low-latency access
- **Scratch space**: Intermediate computation results

**Key property**: This is a software-managed scratchpad — not a hardware cache. Data persists until explicitly overwritten. There is no eviction policy, no cache coherence protocol.

### System-Level SRAM

| Device | Total On-chip SRAM | Per Tensix |
|--------|-------------------|------------|
| Wormhole n150 | ~108 MB (72 cores × 1.5 MB) | 1.5 MB |
| Wormhole n300 | ~240 MB (160 cores × 1.5 MB) | 1.5 MB |
| Blackhole p150 | 210 MB (140 cores × 1.5 MB) | 1.5 MB |

The ASPLOS 2025 microbenchmarking paper confirms 210 MB on-chip SRAM for Blackhole with 140 Tensix cores and 24 GDDR6 controllers.

### Tensor Sharding

TT-NN supports tensor sharding strategies that distribute a tensor across multiple Tensix L1s:
- **Height sharding**: rows distributed across cores
- **Width sharding**: columns distributed across cores  
- **Block sharding**: rectangular blocks per core

Sharded tensors enable operations to run with zero DRAM traffic for operands that fit in the distributed L1 pool.

---

## 4. Off-chip Memory (GDDR6)

Unlike NVIDIA GPUs (HBM) and many competitors, Tenstorrent uses **GDDR6** — a deliberate choice to avoid HBM supply chain constraints and leverage commodity memory pricing.

### Memory Specifications

| SKU | GDDR6 Capacity | Memory Controllers | Bandwidth | Interface |
|-----|---------------|-------------------|-----------|-----------|
| Wormhole n150 | 12 GB | 6 controllers × 2 channels | 192–336 GB/s | 192-bit |
| Wormhole n300 | 24 GB | 12 controllers | ~336 GB/s × 2 | — |
| Blackhole p150 | 32 GB | 24 controllers | 512 GB/s | 384-bit |
| Blackhole Galaxy (32× BH) | 1 TB | 24×32 = 768 | ~16 TB/s aggregate | — |

Official Blackhole spec: 32 GB GDDR6, 512 GB/s bandwidth, 24 memory controllers.

### Memory Access Model

DRAM is accessed via the NoC. DRAM controller tiles appear at fixed (noc_x, noc_y) coordinates in the NoC grid. The `get_noc_addr_from_bank_id()` API maps a DRAM bank ID to the corresponding NoC address. Interleaved buffers stripe tensor pages across all DRAM banks for maximum bandwidth.

---

## 5. Host Interface / Package

### Wormhole Cards

| Spec | Value |
|------|-------|
| Form factor | PCIe 4.0 x16 |
| Host interface bandwidth | ~64 GB/s bidir |
| TDP | ~75–150 W (n150), ~200 W (n300) |
| Process | GlobalFoundries 12nm |
| Die size | 670 mm² |
| Power connector | 8-pin EPS12V |

### Blackhole Cards

| Spec | Value |
|------|-------|
| Form factor | PCIe 5.0 |
| GDDR6 | 32 GB |
| Ethernet ports | 4× QSFP-DD (800 Gbps each, ~3.2 Tbps aggregate or 10× 400 Gbps) |
| Process | TSMC 6nm |
| Host CPU option | 16 Big RISC-V cores (can run Linux) |
| Peak FP8 | 745 TOPS |

---

## 6. Scale-up Interconnect (Ethernet-based)

Tenstorrent uses **standard Ethernet** for both on-board chip-to-chip and rack-scale connectivity — a key architectural differentiator from NVLink/proprietary fabrics.

### Wormhole Ethernet

- 16 Ethernet tiles on-chip, each at 100 Gbps
- **1,600 Gbps (1.6 Tbps)** total per chip
- Multi-chip boards (n300) connect two Wormhole chips via direct Ethernet
- T3000 system: 8× Wormhole chips in a 2×4 mesh via Ethernet; ~5 TB/s intra-system

### Blackhole Ethernet

- 10× 400 Gbps Ethernet links per chip = **4 Tbps aggregate** per chip
- QSFP-DD ports (4 ports, each 800 Gbps = total ~3.2 Tbps external)
- Blackhole Galaxy (32 chips in 4×8 mesh): ~16 TB/s aggregate bandwidth

The Ethernet fabric is used for **direct Tensix-to-Tensix data movement** across chips — Ethernet tiles appear in the NoC grid like any other tile, so the same `noc_async_read/write` API works transparently across chip boundaries.

---

## 7. Scale-out Interconnect

Unlike NVIDIA (InfiniBand), AMD (Pensando), and Intel Gaudi (integrated RoCE), Tenstorrent uses **commodity Ethernet switches** for rack-scale deployment.

| Topology | Description |
|----------|-------------|
| Wormhole cluster | 2D mesh via 16× 100 GbE Ethernet per chip; commodity switches |
| Blackhole Galaxy | 4×8 mesh of 32 chips; 400 GbE per inter-chip link; 1 TBps total |
| Blackhole multi-rack | QSFP-DD ports connect to standard 400G Ethernet switches |

**Advantage**: No proprietary switching silicon; uses standard Ethernet stack; lower cost for large clusters.

**Tradeoff**: Higher latency vs. NVLink; software must tolerate or hide link latency.

The Tenstorrent Ethernet model allows a kernel on one chip's Tensix core to issue `noc_async_read` from a Tensix L1 on a different chip, making the multi-chip programming model nearly identical to single-chip programming.

---

## Key Architectural Insights

1. **No hardware caches between Tensix cores**: The programmer controls all data movement. This eliminates cache coherence traffic but requires explicit memory management.

2. **GDDR6 instead of HBM**: Lower peak bandwidth than competitors (512 GB/s vs. 3–8 TB/s for NVIDIA), but GDDR6 has sufficient bandwidth for inference workloads that keep hot data in on-chip SRAM.

3. **Ethernet as primary interconnect**: Commodity Ethernet for both on-chip tile communication and scale-out — no proprietary fabric required.

4. **Blackhole = standalone AI computer**: 16 Big RISC-V host cores + 140 Tensix AI cores on one die; can run Linux and inference workloads without a host x86 CPU.

5. **ISA is fully documented and open**: TT-ISA-Documentation GitHub repo contains complete instruction references for all execution units (Baby RISC-V, Matrix Unit, Vector Unit, Scalar Unit).

---

## Sources

- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Wormhole Specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html)
- [HotChips 2024: Blackhole & TT-Metalium](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf)
- [ASPLOS 2025: Dissecting Blackhole via Microbenchmarking](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Introduction to Tenstorrent (HPC-Asia 2025)](http://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf)
- [Corsix: Tenstorrent Wormhole Series](https://www.corsix.org/content/tt-wh-part1)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [SemiAnalysis: Wormhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-wormhole-analysis-a-scale)
- [SemiAnalysis: Blackhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and)
- [TT-ISA Documentation](https://github.com/tenstorrent/tt-isa-documentation)

---

# Investigation Update — 2026-08-08: Galaxy Blackhole GA, shipping-SKU correction, Quasar

*as_of: 2026-08-08*
*scope: Galaxy Blackhole general availability (2026-04-28) and hardened system spec; correction of pre-existing Blackhole PCIe SKU and Galaxy figures; Quasar next-generation status*
*sources: tenstorrent.com product pages (galaxy, blackhole), Tenstorrent newsroom 2026-04-28 / 2026-05-04 / 2026-06-30, docs.tenstorrent.com Galaxy Blackhole User Guide v1.6 (2026-08-04), The Register 2026-04-28, GitHub tt-metal / tt-llk / tt-isa-documentation, The Register 2023-10-04 (Samsung SF4X)*

## 1. Galaxy Blackhole — general availability

Tenstorrent announced general availability of Galaxy Blackhole on **2026-04-28**. The vendor wording is *"announces today general availability of Tenstorrent Galaxy Blackhole deployed at scale."* Two status framings that circulate in secondary coverage are **not** supported by any primary source retrieved:

- **"Volume production" / "shipping in volume"** — this phrasing appears in no Tenstorrent release, no product page, and no Register write-up. The correct status verb for this survey is *"GA, sold at list price, deployed with named partners."*
- **"Production commenced January 2026"** — traces only to an aggregator blog (ai2.work), not to Tenstorrent.

The 2026-05-01 HPCwire/AIwire item and the 2026-05-04 TT-Deploy event material are downstream of the single 2026-04-28 announcement, not separate launches.

### Hardened system specification

| Parameter | Value | Provenance |
|---|---|---|
| Form factor | 6U rackmount, air-cooled | vendor product page |
| Accelerators | 32 × Blackhole ASIC | vendor product page |
| Peak compute | 23 PFLOPS Block FP8 | vendor product page (press coverage loosely calls this "dense FP8") |
| Accelerator SRAM | 6.2 GB @ 2.9 PB/s | vendor product page |
| Off-chip memory | 1 TB GDDR6 @ 16 TB/s | vendor product page |
| Accelerator fabric | 10 × 400 GbE per ASIC, "for 32 TB/s" | vendor product page (bidirectional) |
| Scale-out | up to 56 × 800 GbE QSFP-DD, "for 11.2 TB/s" | vendor product page (bidirectional) |
| Host CPU | 1 × AMD EPYC 9004 (Zen 4), ≤32 cores, ≤280 W TDP | vendor product page |
| Host memory | up to 576 GB (6 × 96 GB) DDR5-4800 ECC RDIMM | vendor product page |
| Power | "8 – 10 kW avg, 12 kW max (Max system power configurable up to 14.5 kW)" | vendor product page — note 12 kW is the vendor's *max*, not a "standard" figure |
| Weight | 262 lbs / 119 kg | vendor product page |
| Max system scale | 144 nodes / >4,000 chips | vendor statement |

### Bandwidth arithmetic (important for cross-chip comparison)

The vendor's fabric and scale-out numbers are bidirectional:

- 32 ASICs × 10 links × 400 Gb/s = **128 Tb/s = 16 TB/s unidirectional** → 32 TB/s bidirectional
- 56 × 800 Gb/s = **44.8 Tb/s = 5.6 TB/s unidirectional** → 11.2 TB/s bidirectional

The Register (2026-04-28) separately quotes **~100 Tbps aggregate intra-node Ethernet**, which does not reconcile cleanly with 128 Tb/s. No public source explains the gap; recorded here as an open discrepancy rather than resolved in either direction.

### Public list pricing

Galaxy Blackhole from **$110,000** per server; four-node base supercluster from **$440,000**; Galaxy **Wormhole** from **$70,000**. Confirmed on both the vendor product page and in The Register's write-up. This is the first time Tenstorrent has published datacenter system list prices.

### Deployments named at GA

Prodia, Equinix, OrionVM, BetterBrain, Virtu Financial, Turiyam, Cirrascale, ai&. (Coverage that lists only Equinix / Virtu / Turiyam / Cirrascale is incomplete.)

At TT-Deploy (2026-05-04) Tenstorrent described **36-Galaxy superclusters** networked as a single computer already running.

## 2. Correction — Blackhole shipping PCIe SKUs

The 2026-04-05 baseline recorded Blackhole as "140 Tensix cores / 745 TOPS FP8 / 32 GB". That is the **full-die** figure. The vendor Blackhole product page lists three shipping cards, all at 120 enabled Tensix cores:

| SKU | Tensix | Big RISC-V | GDDR6 | Memory BW | Peak | Host I/F | TDP | Cooling | Ports | Price |
|---|---|---|---|---|---|---|---|---|---|---|
| p100a | 120 | 16 | 28 GB | 448 GB/s | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | active | — | $999 |
| p150a | 120 | 16 | 32 GB | 512 GB/s | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | active | 4 × QSFP-DD 800G | $1,399 |
| p150b | 120 | 16 | 32 GB | 512 GB/s | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | passive | 4 × QSFP-DD 800G | $1,399 |

**p300** (dual-die) no longer appears on the product page and is flagged as delisted in the chip deliverables. Consequent SRAM correction: shipping cards carry **180 MB** on-chip SRAM (120 × 1.5 MB); the ASPLOS-2025-confirmed 210 MB corresponds to the 140-core die.

**Unresolved:** the Galaxy per-ASIC configuration. 23 PFLOPS ÷ 32 ≈ 719 TFLOPS/ASIC and 6.2 GB ÷ 1.5 MB ÷ 32 ≈ 129 cores/ASIC — both between the 120-core card and the 140-core die. Tenstorrent has not published the Galaxy ASIC binning. **Not disclosed.**

## 3. Corrections to pre-existing repo figures

| Field | Repo (2026-04-05) | Correct (2026-08-08) |
|---|---|---|
| Galaxy peak | ~25 PFLOPS FP8 (summary) / ~24 PFLOPS (hw-arch) | **23 PFLOPS Block FP8** |
| Galaxy SRAM | ~6.7 GB | **6.2 GB @ 2.9 PB/s** |
| Galaxy memory BW | "~16 TB/s aggregate" (unlabeled) | **16 TB/s** GDDR6 bandwidth (vendor-stated) |
| Galaxy scale-out | "~1 TBps aggregate" | **11.2 TB/s** vendor / 5.6 TB/s unidirectional over 56 × 800 GbE |
| Blackhole PCIe SKU | 140 cores / 745 TOPS FP8 | **120 cores / 664 TFLOPS BLOCKFP8** (full-die 140/745 retained as a reference figure) |
| Blackhole SKU list | p150, p300 | **p100a, p150a, p150b**; p300 delisted |

## 4. Quasar — next-generation target

Quasar is a real next-generation Tenstorrent target under active bring-up, but the "it just landed" framing is wrong by roughly eleven months:

- `feat: initial commit of Quasar LLK` — `tenstorrent/tt-llk` PR #593, **2025-08-15**; pulled into tt-metal **2025-08-17**.
- GitHub commit search: **662** Quasar-mentioning commits in `tt-metal`, **85** in `tt-llk`.
- tt-metal **v0.75.0** (2026-07-30) and Aug-2026 dev builds add incremental items only: "Enable trace capture and replay on Quasar", "Enabling DFBs in Quasar fast-dispatch flow", "Add Quasar fast dispatch stress tests", "Fix Quasar unicast and iDMA VC assignments", "make Quasar-unsupported APIs fail at compile time", "Quasar unary and binary SFPU performance tests", "Quasar LLK perf matmul coverage", "Enable Quasar selection in LLK ttsim regression script".

**Process node.** Tenstorrent announced in **October 2023** that the Quasar chiplet — then described as a "low-cost, low-power chiplet for machine learning" — would be built on **Samsung Foundry SF4X (4 nm-class)** at Taylor, Texas. Tenstorrent has not restated this for the 2025–26 tt-metal Quasar architecture target, so the node attribution is recorded at **medium confidence**, not confirmed.

**Still not disclosed:** Quasar core count, SRAM/memory configuration, peak throughput, data types, packaging, product name, or ship date. `tt-isa-documentation` contains only `WormholeB0` and `BlackholeA0` top-level directories, and docs.tenstorrent.com/aibs lists only Wormhole and Blackhole cards plus Wormhole/Blackhole QuietBox, LoudBox and Galaxy systems.

What *is* inferable from the open-source stack, and recorded as such: Quasar has a dedicated LLK backend, SFPU/matmul perf-test coverage, an iDMA/unicast path with virtual-channel assignment, a DFB-based fast-dispatch flow, and a set of Metalium APIs it deliberately does not support (compile-time rejected). This places it in the Tensix lineage with a revised data-movement engine rather than as a new architecture — an inference from code structure, not a vendor statement.

## 5. Adjacent hardware in the window

- **TT-Ascalon S** (2026-06-30): RISC-V **CPU IP** for agentic AI workloads; ~50% the area of TT-Ascalon X at ~140% performance per mm². Independently syndicated (Mirage News, Morningstar/ACCESSWIRE, Yahoo Finance). CPU IP, not a datacenter accelerator. Process node **not disclosed**.
- **Japan** (2026-06-30, same release): ai& deploying **120+ Galaxy systems**, described as the largest scale of sovereign AI compute **in Japan** (not "in the region"); Turing demonstrated Blackhole in vehicles; RISC-V CPU chiplet contributed to Japan's national **2 nm** program via **Rapidus**; Osaka AI datacenter active; plan to bring up to 200 Japanese silicon engineers into design teams.

## 6. Explicitly not confirmed

- No Tenstorrent talk in the **Hot Chips 38** (Aug 23–25, 2026) advance program — confirmed absent.
- No Tenstorrent submission in **MLPerf Inference v6.0** (2026-04-01) or MLPerf Training.
- No LG deal in this window.
- Tenstorrent / **Infinia Technologies** sovereign-AI partnership is dated **2026-01-27**, before the repo baseline — out of window, not new.

## Sources (added 2026-08-08)

- [Galaxy Product Page](https://tenstorrent.com/en/hardware/galaxy)
- [Blackhole Product Page](https://tenstorrent.com/en/hardware/blackhole)
- [Galaxy Blackhole System Docs (User Guide v1.6, 2026-08-04)](https://docs.tenstorrent.com/systems/galaxy-blackhole/index.html)
- [Galaxy Blackhole datasheet PDF](https://docs.tenstorrent.com/_downloads/3086863c42126fd0d63b01baccf8432e/galaxy-blackhole.pdf)
- [Tenstorrent Enables AI at Scale (GA, 2026-04-28)](https://tenstorrent.com/en/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance)
- [TT-Deploy (2026-05-04)](https://tenstorrent.com/en/newsroom/tt-deploy)
- [Tenstorrent Sets New Performance Records, Launches TT-Ascalon S (2026-06-30)](https://tenstorrent.com/en/newsroom/tenstorrent-sets-new-performance-records-launches-tt--ascalon-s)
- [The Register: Galaxy Blackhole AI servers are finally out (2026-04-28)](https://www.theregister.com/software/2026/04/28/tenstorrents-galaxy-blackhole-ai-servers-are-finally-out/5229759)
- [HPCwire/AIwire: Galaxy Blackhole GA (2026-05-01)](https://www.hpcwire.com/aiwire/2026/05/01/tenstorrent-announces-general-availability-of-galaxy-blackhole-ai-system/)
- [Mirage News: TT-Ascalon S / Japan (2026-06-30)](https://www.miragenews.com/tenstorrent-breaks-records-unveils-tt-ascalon-s-1701731/)
- [Morningstar/ACCESSWIRE: TT-Ascalon S / Japan expansion](https://www.morningstar.com/news/accesswire/1183834msn/tenstorrent-sets-new-performance-records-launches-tt-ascalon-s-and-expands-across-japan)
- [docs.tenstorrent.com/aibs — product index](https://docs.tenstorrent.com/aibs/)
- [tt-isa-documentation repo contents (WormholeB0, BlackholeA0 only)](https://github.com/tenstorrent/tt-isa-documentation)
- [The Register: Samsung to fab RISC-V chips for Tenstorrent (2023-10-04)](https://www.theregister.com/on-prem/2023/10/04/samsung-to-fab-risc-v-chips-for-tenstorrent/)
- [inkl: Tenstorrent to use Samsung SF4X for Quasar low-cost AI chiplet](https://www.inkl.com/news/tenstorrent-to-use-samsung-s-sf4x-for-quasar-low-cost-ai-chiplet)
- [The Register: Blackhole QuietBox workstation reviewed (2025-11-27)](https://www.theregister.com/on-prem/2025/11/27/blackhole-quietbox-tenstorrents-ai-workstation-reviewed/2113269)
- [MLPerf Inference v6.0 results (2026-04-01) — no Tenstorrent entry](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/)
- [Hot Chips 38 advance program — no Tenstorrent talk](https://hotchips.org/advance-program/)
