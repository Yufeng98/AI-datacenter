# Graphcore IPU — Hardware Architecture Investigation

**Chip:** Graphcore IPU (GC200 Colossus Mk2 / Bow IPU)  
**Device Class:** MIMD / BSP (IPU)  
**Investigator:** AI Chip Research Pipeline  
**Date:** 2026-04-05  

---

## 1. Overview

The Graphcore IPU (Intelligence Processing Unit) is a radically different architecture from GPU/TPU/NPU designs. It is a **massively parallel, distributed-memory MIMD processor** that enforces the **Bulk Synchronous Parallel (BSP)** execution model in hardware. It was designed from first principles to accelerate sparse, irregular graph-like workloads in machine learning — where standard SIMD/systolic-array designs are inefficient.

As of 2026, the most recent silicon is the **Bow IPU** (2022), based on the same GC200 microarchitecture as Colossus Mk2 (2020) but enhanced with TSMC's 3D Wafer-on-Wafer (WoW) technology for power delivery. Graphcore was acquired by SoftBank in July 2024; the IPU product line continues under SoftBank, with next-generation details unannounced.

---

## 2. Chip Generations

| Generation | Name | Year | Process | Tiles | SRAM | Compute |
|-----------|------|------|---------|-------|------|---------|
| Mk1 | Colossus GC2 | 2018 | 16nm | 1,216 | ~304 MB | ~125 TFLOPS FP16 |
| Mk2 | Colossus GC200 | 2020 | TSMC 7nm | 1,472 | 900 MB | 250 TFLOPS FP16 |
| Mk2+ | Bow IPU | 2022 | TSMC 7nm WoW | 1,472 | 900 MB | 350 TFLOPS FP16 |

---

## 3. Tile Architecture (Per-Tile Detail)

### 3.1 Tile Structure

Each IPU tile is an **independent, complete processor** with its own private SRAM and execution pipeline. There is no shared L1/L2 cache hierarchy — all memory is private and explicitly managed.

```
IPU Tile (×1,472 per chip)
├── Supervisor context (1 thread)
│   └── Manages worker dispatch, exchange ops
├── Worker contexts (6 threads, barrel/round-robin scheduled)
│   ├── ALU (integer + FP32 + FP16)
│   ├── Vector unit (FP16 MAC, FP32 MAC)
│   └── Load/Store unit (private SRAM only)
└── Private SRAM: 624 KB per tile
```

- **6 hardware worker threads** per tile (barrel-scheduled round-robin, 1 instruction/context/cycle)
- **1 supervisor thread** per tile (privileged; launches exchange and vertex dispatch)
- **7 total hardware contexts** per tile
- **8,832 total hardware threads** across the GC200 (6 × 1,472)
- **No out-of-order execution** — in-order 6-issue pipeline
- **No hardware prefetcher** — software must manage data placement
- Clock: 1.35 GHz (GC200 Mk2) / **1.85 GHz** (Bow IPU)

### 3.2 Functional Units per Tile

- Integer ALU (32-bit)
- FP32 scalar FPU
- FP16 vector MAC (2× FP16 per cycle → dominant for AI workloads)
- Memory access to private 624 KB SRAM

### 3.3 Memory (Per Tile)

- **624 KB SRAM** exclusive to that tile
- No cache; no virtual memory; no page faults
- Deterministic, cycle-accurate access latency
- Total IPU: 1,472 × 624 KB = **~900 MB on-chip SRAM**

---

## 4. Whole-Chip Specifications (GC200 / Bow IPU)

| Parameter | GC200 (Mk2) | Bow IPU |
|-----------|-------------|---------|
| Process | TSMC 7nm | TSMC 7nm WoW |
| Die size | 823 mm² | ~823 mm² (active) + power die |
| Transistors | 59.4 billion | ~59.4B (active) + power wafer |
| Tiles | 1,472 | 1,472 |
| Worker threads total | 8,832 | 8,832 |
| SRAM per chip | 900 MB | 900 MB |
| Compute (FP16.16) | 250 TFLOPS | 350 TFLOPS |
| On-chip exchange BW | 11 TiB/s | 11 TiB/s |
| Clock | 1.35 GHz | 1.85 GHz |
| TDP (per chip) | ~120W | ~120W (lower voltage) |

---

## 5. BSP Execution Model

The IPU faithfully implements the **Bulk Synchronous Parallel** model in hardware:

```
┌─────────────────────────────────────────────────────────┐
│  BSP Superstep (repeating unit of execution)            │
│                                                         │
│  Phase 1: COMPUTE                                       │
│  ├── All 1,472 tiles run their assigned vertex/codelet  │
│  ├── Access only own private SRAM                       │
│  └── No communication possible                          │
│                                                         │
│  Phase 2: EXCHANGE                                      │
│  ├── All tiles simultaneously send/receive via fabric   │
│  ├── Memory-to-memory DMA (supervisor-managed)          │
│  └── No compute during exchange                         │
│                                                         │
│  Phase 3: BARRIER SYNC                                  │
│  └── Global hardware barrier; all tiles must complete   │
│      before next superstep begins                       │
└─────────────────────────────────────────────────────────┘
```

**Key properties of BSP:**
- **Race-free by construction:** no shared mutable state during compute
- **Deadlock-free:** exchange is always collective, not point-to-point blocking
- **Deterministic:** same program always produces same result
- **Programmer contract:** within a superstep, tiles execute independently

This is in sharp contrast to GPU SIMT (same instruction, different data per warp) and TPU (systolic arrays processing fixed-shape matrix tiles).

---

## 6. On-Chip Interconnect (IPU Exchange)

The IPU Exchange fabric is a **non-blocking, all-to-all** communication network that connects all 1,472 tiles during the exchange phase:

- **Bandwidth:** 11 TiB/s (all-to-all, aggregate)
- **Topology:** partial crossbar / folded Clos (exact topology proprietary)
- **Communication model:** send/receive buffers in tile SRAM; supervisor thread initiates DMA
- **No shared memory bus:** each tile sends to explicitly addressed tiles
- **Programmability:** Poplar compiler statically schedules exchange programs (no runtime routing)

---

## 7. Off-Chip Memory (Streaming Memory)

IPUs cannot access off-chip memory directly during compute — they must copy it in/out through explicit exchange operations:

- **Streaming Memory:** DDR4 DIMMs on the IPU-Machine host server
  - Up to **112 GB per IPU** (M2000 configuration)
  - Up to **448 GiB** total per IPU-M2000 (4 IPUs)
  - **180 TB/s** aggregate bandwidth (4 IPUs, M2000)
- Large model parameters live in Streaming Memory; code explicitly pages regions into SRAM
- This enables models with 100s of billions of parameters on multi-IPU systems

---

## 8. Bow IPU — 3D Wafer-on-Wafer Technology

The Bow IPU is the **world's first production chip using TSMC's WoW (Wafer-on-Wafer) hybrid bonding**:

```
┌─────────────────────────────────────┐
│  Top Wafer (Active Die)              │
│  ├── 1,472 IPU-Core tiles           │
│  ├── 900 MB SRAM                    │
│  └── Exchange fabric                │
├─────────────────────────────────────┤
│  Hybrid Bond (Cu-Cu direct bonding)  │
├─────────────────────────────────────┤
│  Bottom Wafer (Power Delivery)       │
│  ├── Decoupling capacitors          │
│  └── Power delivery network (PDN)   │
└─────────────────────────────────────┘
```

**Benefit:** Capacitor reservoirs directly beneath transistors smooth power delivery, enabling:
- Higher clock (1.85 GHz vs 1.35 GHz)
- Lower operating voltage (VDD)
- **40% more TFLOPS** with **16% better energy efficiency**
- Same microarchitecture as GC200 — purely a power/frequency improvement

---

## 9. Multi-IPU Systems (IPU-Machine and IPU-POD)

### 9.1 IPU-Machine M2000

```
IPU-M2000 (1U server)
├── 4× Bow IPU (GC200)
│   └── IPU-Link fabric (direct chip-to-chip)
├── 448 GiB DDR4 Streaming Memory
├── PCIe host interface (to proxy host CPU)
└── GW-Link ports (to IPU-Gateway for rack interconnect)
```

- 4 IPUs connected via **IPU-Link** (ultra-low-latency direct links)
- IPU-Link: ~64 GB/s bidirectional per IPU pair (tight NUMA topology)

### 9.2 IPU-POD Rack Systems

| System | IPUs | AI Compute | IPU-Machines |
|--------|------|-----------|--------------|
| IPU-POD4 | 4 | 1.4 PFLOPS | 1 |
| IPU-POD16 | 16 | 5.6 PFLOPS | 4 |
| IPU-POD64 | 64 | ~22 PFLOPS | 16 |
| IPU-POD128 | 128 | ~45 PFLOPS | 32 |
| IPU-POD256 | 256 | ~90 PFLOPS | 64 |
| IPU-POD1024+ | 64K | ~16 EFLOPS | 16K |

### 9.3 GW-Link (Gateway Link)

- Connects **IPU-Gateway** chips across racks
- Two topologies:
  - **Looped:** direct cable, ring-like; for up to ~POD256
  - **Switched:** each GW-Link → network switch; for large clusters
- Enables **all-reduce and data-parallel** training across racks
- IPU-POD128 = 2 × IPU-POD64 connected via GW-Link

---

## 10. Comparison: IPU vs GPU vs TPU

| Dimension | Graphcore IPU (Bow) | NVIDIA H100 | Google TPUv4 |
|-----------|-------------------|-------------|--------------|
| Architecture | MIMD / BSP | SIMT (warp-based) | Systolic Array |
| Compute model | BSP superstep | CUDA kernel | XLA/HLO |
| On-chip memory | 900 MB SRAM | 50 MB L2 + 80 GB HBM | 32 MB VMEM |
| Memory type | Distributed private SRAM | HBM3 | HBM2e |
| Tile/core count | 1,472 | 132 SMs | 2 MXU TensorCores |
| Thread model | 8,832 HW threads | 32K+ warps | Dataflow |
| FP16 TFLOPS | 350 | 1,979 | 275 (bfloat16) |
| Interconnect | IPU-Link / GW-Link | NVLink 4 | ICI |
| Off-chip BW | 180 TB/s (M2000) | 3.35 TB/s HBM | ~1.2 TB/s |

---

## 11. Key Architectural Insights

1. **Memory-first design:** The IPU trades off raw FLOPS for on-chip memory bandwidth; 11 TiB/s on-chip is ~3,000× H100's HBM bandwidth.
2. **Determinism:** BSP + no caches = bit-reproducible results (critical for research).
3. **Sparsity-native:** MIMD tiles can handle irregular sparse data efficiently; each tile can branch independently.
4. **Static scheduling:** Poplar compiler generates a fully static execution schedule — no dynamic dispatch overhead at runtime.
5. **Model parallelism:** IPU's design encourages sharding models across tiles/chips; it is not primarily a data-parallel device.
6. **Bow WoW:** First production use of Cu-Cu bonded 3D integration to solve power delivery — a significant process innovation.

---

## 12. Re-scan — 2026-08-08 (roadmap update)

**Scan window:** 2026-04-01 → 2026-08-08. **Prior investigation date:** 2026-04-05.
**Verdict: no hardware change.** Bow IPU (2022) remains the current and only shipping Graphcore silicon. Sections 1–11 above stand unmodified.

### 12.1 Confirmed negatives (hardware)

| Question | Answer at 2026-08-08 | Evidence |
|---|---|---|
| New IPU generation (Mk3 / post-Bow)? | **No — none announced** | Graphcore blog index 2026 is recruiting/culture/office posts; no product announcement |
| Any hardware disclosure scheduled? | **No** — Graphcore is absent from the Hot Chips 38 advance program (Aug 23–25, 2026) | hotchips.org advance program; AI-accelerator slots go to Meta, NVIDIA, Cerebras, Microsoft, SambaNova, Google, OpenAI, d-Matrix |
| New datapath (FP8 / BF16 / INT4)? | **No** | No public Graphcore material describes one |
| HBM on any IPU? | **No** — Streaming Memory is still DDR4 | No change to product pages or docs |
| Process migration off TSMC 7 nm? | **Not disclosed** | No public statement |

The single Graphcore technical publication in the window, *"Stochastic Rounding: How randomness helps us build better models"* (2026-05-26), describes the **existing** GC200/Bow FP16.SR feature already covered in §13 of the software-stack investigation and the Precision rows of the layer table. It is not a hardware disclosure.

### 12.2 Explicitly rejected — "Izanagi"

Analyst commentary (Jon Peddie Research, 2026-06-16) and derivative blogs describe a Graphcore/Ampere co-developed **"Izanagi"** chip targeting SoftBank Stargate deployments in 2026. **No primary Graphcore or SoftBank source confirms this.** Assessment: low confidence, likely conflating SoftBank's separately-reported Izanagi initiative with Graphcore. **Recorded here as rejected so it is not re-imported from aggregators on a later scan.** It must not appear in `chips/graphcore/hw-architecture.md`, the layer table, or the comparison tables.

### 12.3 Corporate context bearing on roadmap risk

Not hardware facts, but they are the reason the "next-gen roadmap uncertain" limitation persists:

- **SoftBank recapitalisation (high confidence, primary registry).** A single share issued to SoftBank valued at roughly **$457M**, recorded in a Companies House **SH01 filed 2026-04-13**; reported by CNBC on 2026-05-12 as an injection of ~$450M. Further RES10 allotment resolutions and SH01 filings on **2026-05-29, 2026-06-08, 2026-07-24 and 2026-08-04**, plus articles re-adopted 2026-04-30, indicate an **ongoing recapitalisation rather than a one-off**. The $457M figure is press-derived from the filing, **not** a vendor statement.
- **Both technical co-founders have departed.** Simon Knowles (IPU chief architect) ceased to be a director 2025-08-20; Nigel Toon stepped down as Executive Chair effective 2026-07-31 (Companies House records termination 2026-07-30, TM01 filed 2026-08-06). Marcus William McElroy — a director since **2025-12-18**, not a July 2026 arrival — now leads the company; **his title is not vendor-confirmed** (Graphcore says only "takes the helm").
- **Acquisition price should be hedged.** Terms were never officially disclosed. Sifted reported "$600m+"; Jon Peddie Research "$500–600 million". The repo's bare "~$500M" is an estimate, not a disclosed figure. July 2024 timing is solid: SoftBank-side directors Ippei Mimura and Jared Roscoe were appointed 2024-07-11, the same day seven prior investor-directors resigned.

### 12.4 Re-scan sources

- https://www.graphcore.ai/posts
- https://hotchips.org/advance-program/
- https://api.github.com/orgs/graphcore/repos
- https://find-and-update.company-information.service.gov.uk/company/10185006/filing-history
- https://find-and-update.company-information.service.gov.uk/company/10185006/officers
- https://www.cnbc.com/2026/05/12/softbank-graphcore-ai-chip-investment.html
- https://www.graphcore.ai/posts/graphcore-co-founder-and-executive-chair-nigel-toon-steps-down
- https://www.bristol247.com/business/news-business/co-founder-of-billion-dollar-bristol-ai-firm-steps-down/
- https://sifted.eu/articles/graphcore-cofounder-exits-company-one-year-on-from-softbank-acquisition
- https://www.jonpeddie.com/news/graphcores-ipu-doing-well-at-softbank/ (analyst commentary — "Izanagi" claim, rejected)

---

## Sources

- [IPU Programmer's Guide — Hardware Overview](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html)
- [Hot Chips 2021 — Colossus Mk2](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf)
- [Bow IPU Processors Product Page](https://www.graphcore.ai/bow-processors)
- [TSMC 3D WoW — Tom's Hardware](https://www.tomshardware.com/news/graphcore-tsmc-bow-ipu-3d-wafer-on-wafer-processor)
- [Graphcore Goes 3D — Next Platform](https://www.nextplatform.com/2022/03/03/graphcore-goes-3d-with-ai-chips-architects-10-exaflops-ultra-intelligent-machine/)
- [IPU-POD128 Reference Design](https://docs.graphcore.ai/projects/ipu-pod128-datasheet/en/latest/product-description.html)
- [Switched GW-Links Documentation](https://docs.graphcore.ai/projects/switched-gwlinks/en/latest/switched_gwlinks.html)
- [Dissecting the Graphcore IPU Architecture (Citadel)](https://www.graphcore.ai/hubfs/assets/pdf/Citadel%20Securities%20Technical%20Report%20-%20Dissecting%20the%20Graphcore%20IPU%20Architecture%20via%20Microbenchmarking%20Dec%202019.pdf)
- [Evaluating Emerging AI/ML Accelerators (arxiv)](https://arxiv.org/html/2311.04417v3)
