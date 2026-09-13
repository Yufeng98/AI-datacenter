# Graphcore IPU — Hardware Architecture

*as_of: 2026-08-08*

> **Generation status, 2026-08-08:** re-scanned for the Apr–Aug 2026 window. **No new IPU generation exists.** No Mk3 and no post-Bow part was announced; Graphcore does not appear in the Hot Chips 38 (Aug 2026) advance program; the only 2026 technical blog post covers the *existing* GC200/Bow stochastic-rounding feature. **Bow IPU (2022) remains the current and only shipping silicon**, and every compute, memory, and interconnect figure in this document is unchanged from the 2026-04-05 revision. See "Generation Status — 2026-08-08 Re-scan" at the end of this file, and the corporate/funding context in `chips/graphcore/summary.md`.

## Overview

The Graphcore IPU (GC200 Colossus Mk2 / Bow IPU) is a **MIMD (Multiple Instruction Multiple Data)** processor that implements the **BSP (Bulk Synchronous Parallel)** execution model in hardware. It contains 1,472 independent tile processors, each with 624 KB of private SRAM, connected by an all-to-all 11 TiB/s on-chip exchange fabric.

---

## Compute

### Tile (×1,472 per chip)

Each tile is a complete, independent processor:

- **6 hardware worker threads** (barrel/round-robin scheduled: 1 instruction per context per cycle)
- **1 supervisor thread** (privileged; manages vertex dispatch and exchange initiation)
- **7 total hardware contexts** per tile
- **8,832 total simultaneous hardware threads** (6 × 1,472)
- FP16 vector MAC (2× FP16/cycle) — dominant AI compute path
- FP32 scalar FPU
- 32-bit integer ALU
- In-order pipeline; no out-of-order execution; no hardware prefetch; no branch predictor for speculative execution
- Clock: 1.35 GHz (GC200 Mk2), **1.85 GHz** (Bow IPU)

### Execution Model: BSP Superstep

```
Superstep N:
  [Compute Phase]  ← all 1,472 tiles run independent vertex code
                     - accesses only own private 624 KB SRAM
                     - no communication
  [Barrier Sync]   ← global hardware barrier
                     - all tiles must complete before exchange
  [Exchange Phase] ← all-to-all SRAM DMA through IPU fabric
                     - memory-to-memory transfer
                     - supervisor-initiated
→ Superstep N+1
```

Properties:
- **Race-free**: no shared mutable state during compute
- **Deadlock-free**: exchange is collective, never blocking point-to-point
- **Deterministic**: same program → same bit-exact result
- **MIMD**: each tile runs completely independent instructions (vs GPU SIMT lockstep warps)

---

## Memory

### On-Chip (In-Processor Memory)

| Level | Capacity | Bandwidth | Access |
|-------|----------|-----------|--------|
| Per-tile private SRAM | 624 KB | Cycle-level direct | Tile-exclusive |
| Total on-chip SRAM | ~900 MB | 11 TiB/s (all-to-all exchange) | Distributed |

- No L1/L2 cache hierarchy
- No cache coherence protocol
- No virtual memory; no page faults during compute
- Deterministic, cycle-accurate latency
- All SRAM is private: tile N cannot directly read tile M's SRAM during compute phase

### Off-Chip (Streaming Memory)

| Type | Capacity | Bandwidth | Access Model |
|------|----------|-----------|--------------|
| DDR4 DIMMs (on IPU-Machine server) | Up to 112 GB/IPU | 180 TB/s aggregate (4-IPU M2000) | Explicit DMA into tile SRAM |

- Streaming Memory is on the IPU-Machine host server, not on the chip die
- Access requires explicit DMA copy into tile SRAM before compute
- Enables models with 100B+ parameters through explicit tiling
- **No HBM on-chip** — unlike GPU architecture

---

## Chip Physical Specifications

### Generation overview

| Generation | Year | Status (2026-08-08) | Process | Tiles | On-chip SRAM | FP16 TFLOPS |
|---|---|---|---|---|---|---|
| Colossus Mk1 (GC2) | 2018 | Superseded | 16 nm | 1,216 | ~304 MB | ~125 |
| Colossus Mk2 (GC200) | 2020 | Superseded | TSMC 7 nm | 1,472 | 900 MB | 250 |
| **Bow IPU (Mk2+)** | 2022 | **Current and only shipping part** | TSMC 7 nm WoW (3D) | 1,472 | 900 MB | 350 |
| Next-generation IPU ("Mk3") | — | **Not announced** (as of 2026-08-08) | not disclosed | not disclosed | not disclosed | not disclosed |

No post-Bow IPU has been announced, sampled, or shipped. Graphcore is absent from the Hot Chips 38 (Aug 23–25, 2026) advance program, so no disclosure is scheduled there either.

### Per-chip detail

| Parameter | GC200 (Colossus Mk2) | Bow IPU |
|-----------|---------------------|---------|
| Process | TSMC 7nm | TSMC 7nm WoW (3D) |
| Active die area | 823 mm² | 823 mm² + power die |
| Transistors | 59.4 billion | ~59.4 B (active die) |
| Clock | 1.35 GHz | 1.85 GHz |
| FP16 TFLOPS | 250 | 350 |
| TDP | ~120 W | ~120 W |
| Year | 2020 | 2022 |

### Bow IPU — 3D Wafer-on-Wafer Technology

The Bow IPU is the **world's first production chip using TSMC Wafer-on-Wafer (WoW) hybrid bonding**:

```
┌─────────────────────────────────────┐
│  Top Wafer (Active Die, TSMC 7nm)   │
│  ├── 1,472 IPU-Core tiles           │
│  ├── 900 MB SRAM                    │
│  └── Exchange fabric                │
├─────────────────────────────────────┤
│  Cu-Cu Hybrid Bond (direct bonding) │
├─────────────────────────────────────┤
│  Bottom Wafer (Power Delivery Die)  │
│  ├── Decoupling capacitors          │
│  └── Power delivery network (PDN)  │
└─────────────────────────────────────┘
```

**WoW benefit**: Capacitor reservoirs directly beneath transistors smooth power delivery:
- Higher clock (1.85 vs 1.35 GHz)
- Lower operating voltage (VDD)
- **40% more TFLOPS** with **16% better energy efficiency**
- Same microarchitecture as GC200 — purely a power/frequency improvement

---

## On-Chip Interconnect

### IPU Exchange Fabric

| Property | Value |
|----------|-------|
| Topology | All-to-all, non-blocking |
| Aggregate bandwidth | 11 TiB/s |
| Routing | Static (scheduled by Poplar compiler) |
| Communication model | Memory-to-memory DMA (supervisor-initiated) |
| Phase | Only active during BSP exchange phase |

- The exchange schedule is fully determined at compile time by the Poplar compiler
- No runtime routing decisions; no packet-switched overhead
- Tiles send to explicitly addressed destinations; no broadcast congestion

---

## Off-Chip Interconnect

### IPU-Link (Scale-up)

| Property | Value |
|----------|-------|
| Scope | Within IPU-Machine M2000 (4 IPUs) |
| Topology | Fully connected 4-IPU cluster |
| Bandwidth | ~64 GB/s bidirectional per IPU pair |
| Use case | Intra-machine BSP exchange; data-parallel 4-way |

### GW-Link (Scale-out)

| Property | Value |
|----------|-------|
| Scope | Between IPU-Machine racks via IPU-Gateway chip |
| Topologies | Looped (direct cables, ring-like) or Switched (via network switch) |
| Use case | Multi-rack IPU-POD systems (POD128, POD256, larger) |

---

## System Architecture

### IPU-Machine M2000 (1U Server)

```
IPU-M2000
├── 4× Bow IPU (GC200)
│   └── IPU-Link fabric (direct chip-to-chip)
├── 448 GiB DDR4 Streaming Memory
├── PCIe host interface (to proxy CPU)
└── GW-Link ports (to IPU-Gateway for rack interconnect)
```

- Average power: 1.25 kW
- 1.4 PFLOPS FP16 aggregate (4× Bow IPU)

### IPU-POD Rack Systems

| System | IPUs | AI Compute | M2000 units |
|--------|------|-----------|-------------|
| IPU-POD4 | 4 | 1.4 PFLOPS | 1 |
| IPU-POD16 | 16 | 5.6 PFLOPS | 4 |
| IPU-POD64 | 64 | ~22 PFLOPS | 16 |
| IPU-POD128 | 128 | ~45 PFLOPS | 32 |
| IPU-POD256 | 256 | ~90 PFLOPS | 64 |
| Max (Bow Pod) | ~64K | ~16 EFLOPS | ~16K |

---

## IPU vs GPU vs TPU Comparison

| Dimension | Graphcore Bow IPU | NVIDIA H100 | Google TPUv4 |
|-----------|-------------------|-------------|--------------|
| Architecture | MIMD / BSP | SIMT (warp-based) | Systolic Array |
| Tiles / SMs | 1,472 | 132 SMs | 2 TensorCore MXUs |
| Threads | 8,832 HW threads | ~8K+ warps | Dataflow |
| On-chip memory | 900 MB SRAM | 50 MB L2 | 32 MB VMEM |
| Off-chip memory | DDR4 (Streaming) | 80 GB HBM3 | 32 GB HBM2e |
| On-chip BW | 11 TiB/s | ~50 TB/s L2 | ~1.2 TB/s |
| Off-chip BW | 180 TB/s (M2000) | 3.35 TB/s HBM | ~1.2 TB/s |
| FP16 TFLOPS | 350 | 1,979 | 275 BF16 |
| Execution model | BSP superstep | CUDA stream | XLA/HLO |
| Determinism | Yes | No (caching) | Partial |
| Sparsity-native | Yes (MIMD) | Limited | No |

---

## Generation Status — 2026-08-08 Re-scan

*Scan window: 2026-04-01 → 2026-08-08. Prior revision: 2026-04-05.*

Every subsystem below was re-checked against the sources listed at the end of this section. **Nothing changed.**

| Subsystem | Status at 2026-08-08 | Change vs 2026-04-05 |
|---|---|---|
| Compute (tile, 6 workers + 1 supervisor, 1,472 tiles) | Bow IPU; 350 TFLOPS FP16 @ 1.85 GHz | None |
| Data types | FP32 / FP16 with hardware stochastic rounding; no FP8, no BF16, no INT4 | None |
| On-chip memory (624 KB/tile, 900 MB total) | Unchanged | None |
| On-chip interconnect (IPU Exchange, 11 TiB/s all-to-all) | Unchanged | None |
| Off-chip memory (DDR4 Streaming Memory, ≤112 GB/IPU) | Unchanged — still no HBM anywhere in the IPU line | None |
| Scale-up / scale-out (IPU-Link, GW-Link, IPU-POD) | Unchanged | None |
| Packaging (TSMC 7 nm WoW, Cu-Cu bonded power die) | Unchanged | None |

Notes on evidence:

- The only Graphcore technical publication in the window — *"Stochastic Rounding: How randomness helps us build better models"* (2026-05-26) — describes the **existing** GC200/Bow FP16.SR hardware feature already documented above. It is **not** a disclosure of new hardware.
- **Not disclosed / not announced:** any successor microarchitecture, any process-node migration, any HBM-bearing IPU, any FP8 or BF16 datapath. These are absent from all public Graphcore material, not merely unconfirmed.
- ⚠️ A Graphcore/Ampere co-developed **"Izanagi"** part for SoftBank Stargate deployments appears in analyst commentary (Jon Peddie Research, 2026-06-16) with **no primary Graphcore or SoftBank confirmation**. Confidence: low; likely conflated with SoftBank's separately-reported Izanagi initiative. **Not a Graphcore product for survey purposes.**
- Corporate context relevant to roadmap risk (a ~$457M SoftBank share issue recorded on the UK register in April 2026 plus further allotments through August 2026; the departure of both technical co-founders) is recorded in `chips/graphcore/summary.md` § "Corporate and Roadmap Update (2026-08-08)".

Re-scan sources: Graphcore blog index (https://www.graphcore.ai/posts); Hot Chips 38 advance program (https://hotchips.org/advance-program/); GitHub org listing (https://api.github.com/orgs/graphcore/repos).

---

## Sources

- [IPU Programmer's Guide — Hardware Overview](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html)
- [Hot Chips 2021 — Colossus Mk2](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf)
- [Bow IPU Processors](https://www.graphcore.ai/bow-processors)
- [IPU-POD128 Reference Design](https://docs.graphcore.ai/projects/ipu-pod128-datasheet/en/latest/product-description.html)
- [Switched GW-Links](https://docs.graphcore.ai/projects/switched-gwlinks/en/latest/switched_gwlinks.html)
- [Dissecting the Graphcore IPU (Citadel)](https://www.graphcore.ai/hubfs/assets/pdf/Citadel%20Securities%20Technical%20Report%20-%20Dissecting%20the%20Graphcore%20IPU%20Architecture%20via%20Microbenchmarking%20Dec%202019.pdf)
- [Tom's Hardware — Bow WoW](https://www.tomshardware.com/news/graphcore-tsmc-bow-ipu-3d-wafer-on-wafer-processor)
