# Qualcomm Cloud AI 100 — Hardware Architecture

*as_of: 2026-08-08*
*generations: Cloud AI 100 (Standard / Ultra) / Dragonfly AI200 / Dragonfly AI250 (HBC Gen 1) / Dragonfly AI300 (HBC Gen 2) / Dragonfly C1000 (CPU)*

---

## Generation Overview

| Generation | Brand | Compute unit | Peak throughput | On-chip SRAM | Off-chip memory | Memory bandwidth | Scale-up fabric | Cooling / form factor | Status (2026-08-08) |
|---|---|---|---|---|---|---|---|---|---|
| Cloud AI 100 Standard | Cloud AI | 16 Hexagon AI cores (HMX+HVX+Q6) | 400 TOPS INT8 @ 75 W | 144 MB | 32 GB LPDDR4X | 136 GB/s | — (NoC only) | Air, PCIe FH3/4L Gen4 ×8 | Shipping |
| Cloud AI 100 Ultra | Cloud AI | 64 AI cores (4× AIC100 + PCIe switch) | 870 TOPS INT8 @ 150 W | 576 MB | 128 GB LPDDR4X | 548 GB/s | PCIe switch (intra-card) | Air, PCIe Gen4 ×16 | Shipping |
| **AI200** | **Dragonfly** | not disclosed | **not disclosed** (Qualcomm declined to disclose peak FLOPS) | not disclosed | **768 GB/card LPDDR5x** (Oct 2025 figure; see conflict note) | not disclosed | UALink / ESUN | Direct liquid cooling, rack-scale, 160 kW rack (Oct 2025 figure) | Sampling in FY2026 (Investor-Day-reported); no production-shipment claim |
| **AI250** | **Dragonfly** | not disclosed + **HBC Gen 1** near-memory compute | **not disclosed** | not disclosed | HBC Gen 1 (LPDDR5x stack); The Register reports 768 GB/card | **claimed 133 TB/s per card; "18× effective memory bandwidth vs AI200"** — unaudited | UALink / ESUN | Direct liquid cooling, rack-scale | Commercial sampling expected **mid-2027** |
| **AI300** | **Dragonfly** | not disclosed + **HBC Gen 2** 3D-stacked near-memory compute (DRAM over XPU logic via TSVs) | **not disclosed**; claimed 4×–8× better perf/W vs existing GPU-based architectures | not disclosed | not disclosed | **claimed "54× increase over AI200"** in effective memory bandwidth — unaudited | UALink / ESUN | Air **and** direct-liquid-cooled rack-level platform | Commercial sampling expected **2028** |
| **C1000** (CPU) | **Dragonfly** | **250+ custom Oryon cores**, chiplet, >5 GHz | not disclosed; ">2× perf/W" vs competitive server CPUs (Qualcomm estimate) | not disclosed | not disclosed | not disclosed | — (host CPU) | not disclosed | Commercial availability expected **2028**; Meta production H2 2028 |

**Dragonfly announcement:** Qualcomm Investor Day, 2026-06-24. "Dragonfly" is the umbrella brand for Qualcomm's
data center line; AI200 and AI250 were retroactively rebranded from their October 2025 launch. Process node, TDP
and die-level organization are **not disclosed** for any of AI200/AI250/AI300/C1000.

> **Unaudited-marketing warning.** Every AI250/AI300 bandwidth figure in this document is a Qualcomm "effective
> memory bandwidth" claim with **undisclosed methodology**. The Register (2026-06-30) reports Qualcomm declined to
> disclose peak FLOPS for AI250 or AI300 and declined to explain the multipliers, and shows the implied ~414 TB/s
> aggregated across 56 AI200 chips would need an implausible ~6,720-bit bus on standard LPDDR5x. No independently
> measured figure exists. Qualcomm did not submit to MLPerf Inference v6.0 (2026-04-01) and has no Hot Chips 2026
> (HC38, August 24–25 2026) talk, so no third-party measurement is available.

---

## Compute

### AI Core (×16 per SoC)
Type: Hexagon Q6 DSP + HVX + HMX

**Tensor Unit (HMX)**
- 8192 INT8 ops/cycle per core
- 4096 FP16 ops/cycle per core
- 125+ linear algebra instructions

**Vector Unit (HVX)**
- 512 INT8 ops/cycle per core
- 256 FP16 ops/cycle per core
- 700+ instructions (AI/image/content)
- Supports INT8/16, FP16/32

**Scalar Processor (Q6)**
- 4-way VLIW
- 6 hardware threads
- Local register file + caches

### Dragonfly AI200 / AI250 / AI300 compute (2026-06-24)

Qualcomm has published **no compute-engine detail** for the Dragonfly accelerators: core type and count, array
dimensions, clock, data types, process node, TDP and peak FLOPS/TOPS are all **not disclosed**. Qualcomm
explicitly declined to disclose peak FLOPS for AI250 and AI300 when asked (The Register, 2026-06-30).

The only compute-adjacent claim is architectural rather than numeric: from AI250 onward, part of the "compute" is
moved *into the memory stack* (see **HBC**, below), and AI300 carries a Qualcomm-claimed **4×–8× better
performance-per-watt versus existing GPU-based architectures**, stated on a memory-bandwidth-per-watt-per-card
basis rather than a FLOPS basis. This is a marketing claim, not a measurement.

### Dragonfly C1000 CPU compute (2026-06-24)

A separate product line, not an accelerator:

- **250+ cores**, chiplet design, custom Qualcomm **Oryon** cores at **>5 GHz**
- ">2× better performance per watt" versus competitive server CPUs (Qualcomm estimate); The Register additionally
  reports a "30 percent more speed" claim
- Three configurations: **agentic**, **general-purpose virtualization**, **AI head node**
- Cache hierarchy, memory channels, process node and TDP: **not disclosed**

## Memory Hierarchy

| Level | Standard SoC | Ultra card |
|---|---|---|
| Per-core L1/L2 | Hexagon std | Hexagon std |
| On-chip SRAM | 144 MB | 576 MB (4×SoC) |
| DRAM | 32 GB LPDDR4X, 136 GB/s | 128 GB LPDDR4X, 548 GB/s |

### HBC — High Bandwidth Compute (AI250 onward, 2026-06-24)

**HBC is the defining architectural change of the Dragonfly accelerator line and an explicit bet against HBM.**
It is a 3D-stacked **near-memory computing** tier: DRAM is stacked directly over the XPU logic die and connected
through TSVs, so the bulk of tensor traffic never crosses a conventional off-package memory bus. (The
DRAM-over-logic-via-TSV mechanism is described independently by The Register, 2026-06-30; Qualcomm's own PR uses
the higher-level phrase "near-memory computing".)

Qualcomm-stated technology claims for HBC — **unaudited, methodology not disclosed**:

| Claim | Baseline |
|---|---|
| **6× bandwidth per watt** | versus HBM |
| **200× capacity per watt** | versus SRAM |

Per-generation memory tiers:

| Generation | Memory tier | Qualcomm bandwidth claim | Capacity |
|---|---|---|---|
| Cloud AI 100 | LPDDR4X, 4×64-bit | 136 GB/s (Std) / 548 GB/s (Ultra) — measured spec | 32 GB / 128 GB |
| AI200 | LPDDR5x, no HBM | not disclosed (this is the 1× baseline for the 18×/54× claims) | 768 GB/card (Oct 2025 launch figure) |
| **AI250** | **HBC Gen 1** | **"industry-leading 133 TB/s per card"**; **"18× increase in effective memory bandwidth compared to AI200"** | not disclosed by Qualcomm; The Register reports 768 GB/card on LPDDR5x |
| **AI300** | **HBC Gen 2** (3D-stacked near-memory) | **"54× increase over AI200"** in effective memory bandwidth (PR); The Register renders the same figure as 54× vs AI250 — prefer the PR | **not disclosed** |

**Capacity-attribution conflict (unresolved).** Qualcomm's October 2025 launch attributed **768 GB/card** to
**AI200**; The Register's June 2026 coverage attributes 768 GB/card on LPDDR5x to **AI250**. Both are recorded
here; the survey does not pick a winner.

**Why this matters architecturally.** Cloud AI 100 and AI200 sit in the same design family as Meta MTIA v1/v2 and
Cambricon MLU370 — LPDDR instead of HBM, trading bandwidth for capacity, cost and power on inference workloads.
AI250/AI300 do *not* reverse that choice by adopting HBM; instead they attack the bandwidth deficit by shortening
the wire, stacking DRAM on logic. Qualcomm is the only vendor in this survey whose stated roadmap moves from
LPDDR to 3D near-memory compute rather than from LPDDR to HBM (contrast Meta MTIA 300+, which moves LPDDR5 → HBM).

## Network on Chip (3 NoCs)
1. **Compute NoC:** AI cores ↔ PCIe, multicast, 186 GB/s
2. **Memory NoC:** AI cores ↔ DRAM controllers
3. **Config NoC:** boot and hardware configuration

## Ultra Card Configuration
- 4 AIC100 SoCs + PCIe switch
- Single-card PCIe Gen4 ×16 host interface
- Multi-SoC tensor sharding for LLMs >32 GB

## Interconnect (Dragonfly, 2026-06-24)

Cloud AI 100 has no scale-up fabric — cards communicate through the host over PCIe. The Dragonfly accelerators
introduce a scale-up network for the first time, and Qualcomm chose **open standards over a proprietary fabric**:

| Tier | Technology | Notes |
|---|---|---|
| Intra-SoC | 3× NoC (Compute / Memory / Config) | Cloud AI 100 lineage; unchanged and undocumented for Dragonfly |
| **Scale-up** | **UALink** and **ESUN** (Ethernet for Scale-Up Networking) | AI200/AI250/AI300; per-chip scale-up bandwidth **not disclosed** |
| **Scale-out** | Ethernet over copper and optical; **800G and 1.6T** | Stated reach from intra-data-center up to a **20 km campus** span |
| Host (C1000 CPU) | **>2 TB/s PCIe Gen 7**, plus **CXL** | C1000 only; host interface for AI200/AI250/AI300 not disclosed |

Maximum system scale (chips per rack, racks per pod) is **not disclosed** for any Dragonfly part. Qualcomm named
**over 35 ecosystem supporters**, including Meta, Arista, Supermicro, Lenovo, Samsung SDS and SK hynix America.

Rack-level platform notes: AI200/AI250 were launched (Oct 2025) as direct-liquid-cooled rack-scale systems at
160 kW per rack; AI300 is described as an **air- and direct-liquid-cooled** rack-level platform. AI300 rack power
is not disclosed.

## Performance Summary

| | INT8 TOPS | TOPS/W |
|---|---|---|
| Standard (75 W) | 400 | 5.3 |
| Ultra (150 W) | 870 | 5.8 |
| AI200 / AI250 / AI300 | **not disclosed** (Qualcomm declined) | **not disclosed** |

No independent benchmark data exists for any Qualcomm data center part in this window: Qualcomm is not among the
24 MLPerf Inference v6.0 submitters (published 2026-04-01) and has no talk on the Hot Chips 2026 (HC38,
August 24–25 2026) program.

## Design Philosophy
- Hexagon DSP lineage: proven mobile NPU at data center scale
- LPDDR over HBM: cost/power trade-off for inference
- Three-NoC separation: prevents traffic contention
- Large SRAM: inference-first (KV-cache, weight caching)
- **(Dragonfly, 2026)** Near-memory compute over HBM: rather than adopting HBM to close the bandwidth gap,
  AI250/AI300 stack DRAM over logic (HBC Gen 1/Gen 2) and claim 6× bandwidth-per-watt versus HBM
- **(Dragonfly, 2026)** Open scale-up standards (UALink, ESUN) instead of a proprietary interconnect
- **(Dragonfly, 2026)** Full-stack posture: an in-house data center CPU (C1000, 250+ Oryon cores) alongside the
  accelerators, plus an open silicon-agnostic compiler layer acquired with Modular

## Sources (2026-08-08 update)

- [Qualcomm Unveils Comprehensive Data Center Roadmap … Dragonfly Portfolio (2026-06-24)](https://www.qualcomm.com/news/releases/2026/06/qualcomm-unveils-comprehensive-data-center-roadmap-for-the-agent)
- [Qualcomm and Meta Announce Strategic Multi-Generation Agreement on Data Center CPUs (2026-06-24)](https://www.qualcomm.com/news/releases/2026/06/qualcomm-and-meta-announce-strategic-multi-generation-agreement-)
- [The Register — Qualcomm's proposed solution to catch up in AI infra: bury the compute under the DRAM (2026-06-30)](https://www.theregister.com/systems/2026/06/30/qualcomms-proposed-solution-to-catch-up-in-ai-infra-bury-the-compute-under-the-dram/5264071)
- [The Register — Qualcomm claims it's not too late for Dragonfly to land in datacenters (2026-06-24)](https://www.theregister.com/systems/2026/06/24/qualcomm-claims-its-not-too-late-for-dragonfly-to-land-in-datacenters/5261758)
- [Futurum Group — Qualcomm's Data Center Re-entry at Investor Day 2026](https://futurumgroup.com/insights/qualcomms-data-center-reentry-at-investor-day-2026-arrives-just-in-time-for-the-inference-decode-prize/)
- [Hot Chips 38 program (August 24–25, 2026) — Qualcomm absent](https://hotchips.org/program/conference/)
- [MLPerf Inference v6.0 results (2026-04-01) — Qualcomm not among the 24 submitters](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/)
