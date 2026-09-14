# Tesla Dojo Hardware Architecture

*as_of: 2026-09-13*
*Primary source: Hot Chips 34 (August 2022)*
*generations: D1 (Dojo v1) / D2 (Dojo 2, cancelled) / Dojo 3 (program acknowledged, nothing disclosed)*

---

## Overview

Tesla Dojo's hardware is organized around a single custom ASIC (the D1 chip) that tiles without switching silicon, assembled into Training Tiles, System Trays, Cabinets, and ExaPODs. The design philosophy prioritizes high compute density and on-chip memory bandwidth over programmability and external DRAM bandwidth, making it optimally suited to dense neural network training workloads.

---

## 0. Generation Overview

| Generation | Status (2026-08-08) | Process | Compute die | Peak (BF16/CFP8) | On-chip SRAM | Off-chip memory | Scale-up | Scale-out |
|---|---|---|---|---|---|---|---|---|
| **D1 (Dojo v1)** | Disclosed; production July 2023 | TSMC 7nm, 645 mm², ~50B transistors | 354 training nodes | 376 TFLOPS | 440 MB | HBM via DIP cards (none on die) | 5×5 Training Tile, direct SerDes butt-join | TTPoE over commodity Ethernet |
| **D2 (Dojo 2)** | **Cancelled August 2025** — declared "an evolutionary dead end"; no silicon disclosure ever made | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed |
| **Dojo 3** | Program publicly confirmed 2026-01-19/20; restated by Musk 2026-04-15. **No hardware disclosure of any kind** | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed |

**Dojo 3 caveat.** The only public statements about Dojo 3 concern *purpose*, not design: Musk (January 2026) said "AI7/Dojo3 will be for space-based AI compute," and SpaceX's Form S-1 (2026-05-20) describes a Terafab chip type "optimized for the space environment to be used in our orbital compute infrastructure." An older Q2 2025 earnings-call remark about a converged Dojo 3 / AI6 chip used "2 of them in a car or an Optimus and maybe a larger number on a board, kind of 5, 12 on a board" predates that repositioning. **Neither statement discloses a node, die, memory tier, or fabric**, and nothing confirms whether Dojo 3 retains or abandons the D1 Training Tile topology. Every "not disclosed" in the tables below is literal. All D1 figures in the following sections are unchanged by the 2026 roadmap news.

---

## 1. Compute Engine

### Training Node (per node, within D1)

| Component | Specification |
|-----------|---------------|
| Units per D1 die | 354 |
| Vector unit width | 64 bytes (512 bits) |
| Supported precisions | BF16, CFP8, CFloat16, FP32 |
| Scalar unit | 64-bit custom ISA |
| Clock speed | 2 GHz |
| Peak throughput per node | ~1.06 TFLOPS BF16 (354 nodes × → 376 TFLOPS/chip) |

### D1 Chip Summary

| Parameter | D1 (Dojo v1) | D2 (Dojo 2, cancelled) | Dojo 3 (2026 program) |
|-----------|--------------|------------------------|-----------------------|
| Process node | TSMC 7nm | not disclosed | not disclosed |
| Die area | 645 mm² | not disclosed | not disclosed |
| Transistors | ~50 billion | not disclosed | not disclosed |
| Training nodes | 354 | not disclosed | not disclosed |
| Clock speed | 2 GHz | not disclosed | not disclosed |
| BF16/CFP8 throughput | 376 TFLOPS | not disclosed | not disclosed |
| FP32 throughput | 22 TFLOPS | not disclosed | not disclosed |
| TDP | 400 W | not disclosed | not disclosed |

The 376 TFLOPS BF16/CFP8 figure is the Hot Chips 34 primary number and is retained; a 362 TFLOPS figure circulating in 2026 secondary coverage is not adopted.

The D1 training node is not a GPU SM or a CPU core. It is closer to a small vector DSP: a scalar control unit paired with a wide SIMD vector unit, with a large private SRAM scratchpad. There is no hardware cache, no TLB, and no coherence between nodes — all memory management is handled by the compiler.

**Dojo 3 compute engine:** not disclosed. No compute-unit type, count, clock, precision set, or throughput figure has been published as of 2026-08-08, and no Tesla talk appears in the Hot Chips 38 advance program (Aug 23–25, 2026).

---

## 2. Data Path

### On-chip Mesh

| Feature | Detail |
|---------|--------|
| Topology | 2D mesh across 354 nodes |
| Aggregate on-chip BW | ~10 TB/s (directional) |
| Routing | Software-scheduled (deterministic) |
| Coherence | None — SRAM is node-local |
| Synchronization | Software barriers in the runtime |

The mesh is fundamentally different from NVIDIA's SIMT model: there is no shared memory visible to multiple threads simultaneously. Each node owns its 1.25 MB SRAM exclusively; data exchange between nodes is explicit DMA-like transfer over the mesh fabric.

### Chip-to-Chip (C2C) SerDes

| Feature | Detail |
|---------|--------|
| SerDes lanes per chip | 576 |
| Lane speed | 112 GT/s |
| Aggregate C2C BW | ~8 TB/s |
| Topology | Direct butt-join on PCB substrate |
| External switch required | No |

D1 chips tile directly on a PCB: the SerDes interfaces on chip edges connect to neighboring chips without any routing or switching silicon. This direct-tile topology means the Training Tile PCB is both the package and the interconnect substrate.

**Dojo 3 data path:** not disclosed. Whether Dojo 3 keeps the software-scheduled on-chip mesh or the switchless butt-join C2C SerDes is unknown; no lane count, lane rate, or topology has been published.

---

## 3. On-chip Memory

| Feature | Detail |
|---------|--------|
| Total SRAM per D1 chip | 440 MB |
| SRAM per training node | 1.25 MB |
| SRAM type | Software-managed scratchpad (no cache) |
| Load BW per node | ~400 GB/s |
| Store BW per node | ~270 GB/s |
| Total SRAM aggregate BW (chip) | ~237 TB/s load (estimate: 354 × 400 GB/s / 2 for overlap) |
| Cache hierarchy | None |
| Virtual memory | Not supported |

The 440 MB of SRAM is the key differentiator: by making SRAM the primary memory tier (rather than an L1/L2 cache above DRAM/HBM), Tesla avoids cache-miss latency and achieves extremely high per-node bandwidth. The tradeoff is that the compiler must explicitly manage all data movement, and the working set of any given tile operation must fit in the aggregate SRAM of the participating nodes.

**Dojo 3 on-chip memory:** not disclosed. No SRAM capacity, per-node scratchpad size, bandwidth, or memory-model statement exists. In particular, there is no public basis for assuming Dojo 3 preserves the no-cache / no-virtual-memory model.

---

## 4. Off-chip Memory

Off-chip memory is not present on the D1 die itself. Instead, **DIP (Dojo Interface Processor) cards** at the Training Tile level provide HBM access.

| Feature | Detail |
|---------|--------|
| HBM location | DIP cards, external to D1 die |
| DIP cards per tile edge | 5 |
| HBM BW per tile | ~13 TB/s aggregate |
| Host-side BW per tile | 160 GB/s (from host servers) |
| HBM per ExaPOD | ~13 TB total |
| SRAM per ExaPOD | ~1.3 TB total |

The two-tier NUMA model (SRAM on-chip, HBM on DIP cards) is purely software-managed. The compiler schedules explicit DMA transfers between HBM and node SRAM. There is no hardware page fault handler or demand-paging mechanism.

**Dojo 3 off-chip memory:** not disclosed. No memory technology (HBM generation, LPDDR, or otherwise), capacity, bandwidth, or attach method — on-package versus DIP-style daughter card — has been published.

---

## 5. Host Interface / Package

| Feature | Detail |
|---------|--------|
| Die packaging | Monolithic (single D1 die per package) |
| Package style | Training Tile PCB (25 chips direct-mounted) |
| Host connectivity | Via DIP cards, ~160 GB/s per tile edge |
| D1 die size | 645 mm² (TSMC 7nm) |
| Training Tile TDP | ~15 kW |

Unlike NVIDIA's CoWoS 2.5D interposer strategy (which integrates HBM stacks on-package), Tesla keeps HBM physically separate on DIP daughter cards. This trades packaging density for modularity: HBM capacity can be upgraded by replacing DIP cards.

**Dojo 3 host interface / package:** not disclosed. Note also the *foundry* question is open: D1 was TSMC 7nm, Tesla's AI5/AI6 are Samsung, and the Terafab venture Tesla joined under a framework agreement targets **Intel 14A** — but no public source ties any specific process to Dojo 3.

---

## 6. Scale-up Interconnect (On-tile Mesh)

The "scale-up" within a Training Tile is the direct SerDes mesh between D1 chips.

| Feature | Detail |
|---------|--------|
| Chips per tile | 25 (5×5) |
| Chip-to-chip BW | ~8 TB/s per D1 |
| Tile aggregate BW | 36 TB/s |
| Switching silicon | None |
| Tile SRAM | 11 GB (25 × 440 MB) |
| Tile compute | 9 PFLOPS BF16 |

**Dojo 3 scale-up:** not disclosed. The July 2025 "kind of 5, 12 on a board" remark describes a hypothetical converged Dojo 3 / AI6 board and predates the January 2026 space-compute repositioning; it is stated intent, not a topology disclosure, and is not recorded here as a specification.

---

## 7. Scale-out Interconnect (TTPoE)

| Feature | Detail |
|---------|--------|
| Protocol | TTPoE (Tesla Transport Protocol over Ethernet) |
| Physical layer | Commodity Ethernet (no PFC required) |
| Latency | Microsecond-range one-way |
| Loss model | Lossy — hardware retry in NIC |
| NIC | "Dumb-NIC" (Tesla custom, fully hardware offloaded) |
| Open-source | github.com/teslamotors/ttpoe |
| Industry consortium | UltraEthernet Consortium (UEC) |

ExaPODs connect via TTPoE on commodity Ethernet switches. The choice of lossy Ethernet (rather than PFC-based lossless InfiniBand or RoCEv2) reduces switch complexity and cost at the price of slightly higher latency variability, which the NIC's hardware retry mechanism absorbs.

**Dojo 3 scale-out:** not disclosed. There has been **no new TTPoE release** since the Hot Chips 36 open-sourcing, and no statement that Dojo 3 will use TTPoE. The `teslamotors/ttpoe` repository remains the only open-source artifact of the program.

---

## System Hierarchy Summary

| Level | Unit | Chips | Compute | On-chip SRAM | Off-chip HBM |
|-------|------|-------|---------|-------------|-------------|
| Die | D1 | 1 | 376 TFLOPS BF16 | 440 MB | — |
| Module | Training Tile | 25 | 9 PFLOPS | 11 GB | ~13 TB/s BW via DIP |
| Rack unit | System Tray | 150 | 54 PFLOPS | 66 GB | — |
| Cabinet | Cabinet | 300 | 108 PFLOPS | 132 GB | — |
| Cluster | ExaPOD | 3,000 | 1 EFLOPS | 1.3 TB | 13 TB |

There is **no published Dojo 3 system hierarchy** — no die, module, rack, or cluster unit has been named or specified.

---

## 8. Dojo 3 Program Status (2026-08-08)

*Roadmap context only. This section contains no hardware specifications because none exist publicly.*

| Item | Status |
|---|---|
| Program acknowledged | Yes — publicly confirmed 2026-01-19/20 (Teslarati 01-19; TechCrunch and Tom's Hardware 01-20; Bloomberg 01-17), restated in Musk's 2026-04-15 X post "AI6, Dojo3 & other exciting chips in work" |
| Silicon disclosed | **No** — no node, die, memory, interconnect, tape-out date, or performance target |
| Stated purpose | Space-based / orbital AI compute (Musk, January 2026: "AI7/Dojo3 will be for space-based AI compute"), corroborated by SpaceX's Form S-1 describing a Terafab chip type "optimized for the space environment" |
| Superseded framing | The Q2 2025 (2025-07-23) earnings-call "converged Dojo 3 / AI6 inference chip … kind of 5, 12 on a board" remark predates the space repositioning |
| Engineering leadership | Anant Nivarti rejoined Tesla to lead silicon engineering, including AI6 and Dojo 3 (Data Center Dynamics, 2026-04-07) |
| Related Tesla silicon | AI5 taped out 2026-04-15, reported in wafer production at Samsung Taylor, TX on 2nm (Electrek, 2026-07-13; volume/qualification unverified). AI6 tape-out targeted for December 2026 (Reuters, 2026-03-19), backed by the $16.5B Samsung agreement (2025-07-27/28) |
| Foundry context | Tesla is a **framework-agreement** participant in Terafab (SpaceX-led, Grimes County TX, Intel 14A). SpaceX's S-1 states neither Tesla nor Intel is obligated to remain in the project. No source ties Terafab or 14A to Dojo 3 specifically |
| Adjacent trademark | Tesla filed "MEGAPOD" (USPTO SN 99893717, ~2026-06-18) covering "computer servers, computer hardware for artificial intelligence processing, computer networking hardware, electrical power distribution units, and cooling systems, sold as a unit" — **trademark filing only**; no product, no chip, no specifications |
| Hot Chips 38 | No Tesla or Dojo talk in the advance program (Aug 23–25, 2026), checked 2026-08-08 |
| Mystery AI hardware acquisition (**Update 2026-09-13**) | Tesla's Q1 2026 10-Q disclosed (April 2026) an agreement to acquire an **unnamed** AI hardware company for up to $2.00B in stock/equity; Electrek reports the deal **closed 2026-07-24** at $1.95B (only $222M allocated to patent/technology intangibles; the $1.73B milestone-dependent tranche is described by Tesla as "improbable" to be achieved). Target **not named**; no confirmed link to Dojo 3, AI5, or AI6 hardware. Pre-baseline item, backfilled as a corpus gap |
| DensityAI personnel flow (**Update 2026-09-13**) | 2026-08-24: Shishuang Sun (Tesla Senior Director, AI Hardware Design — packaging/power/PCB/thermal, worked on Dojo and Autopilot computers) departs for DensityAI (founded by ex-Dojo chief Ganesh Venkataramanan). Evidence DensityAI remains independent a month after Tesla's own AI-hardware acquisition closed — weighs against (does not disprove) DensityAI being that acquisition's target |

---

## Sources

- [Hot Chips 34 PDF — Microarchitecture](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)
- [IEEE Xplore — DOJO Microarchitecture paper](https://ieeexplore.ieee.org/document/10078146/)
- [Chips & Cheese — HC34 analysis](https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture)
- [SemiAnalysis — D1 chip critique](https://newsletter.semianalysis.com/p/the-tesla-dojo-chip-is-impressive)
- [The Next Platform — Inside Dojo](https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/)
- [Tesla TTPoE GitHub](https://github.com/teslamotors/ttpoe)
- [HC2024 TTPoE PDF](https://hc2024.hotchips.org/assets/program/conference/day2/17_HC2024_Tesla_TTPoE_v5.pdf)

### Added 2026-08-08 (Dojo 3 program status)

- [SpaceX Form S-1, filed 2026-05-20 (CIK 0001181412, accession 0001628280-26-036936)](https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm)
- [TechCrunch — Dojo3 for space-based AI compute (2026-01-20)](https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute/)
- [Tom's Hardware — Dojo3 space supercomputer restart (2026-01-20)](https://www.tomshardware.com/tech-industry/supercomputers/elon-musk-restarts-dojo3-space-supercomputer-project-as-ai5-chip-design-gets-in-good-shape-will-be-first-tesla-built-supercomputer-to-feature-all-in-house-hardware-with-no-help-from-nvidia)
- [Tom's Hardware — Terafab: 100M sq ft, $16.8B initial capital (2026-08-07)](https://www.tomshardware.com/tech-industry/semiconductors/terafab-starts-to-take-shape-100-million-square-feet-of-manufacturing-space-and-usd16-8b-initial-capital-investment)
- [Wikipedia — Terafab](https://en.wikipedia.org/wiki/Terafab)
- [Hot Chips 38 advance program](https://hotchips.org/advance-program/) — checked 2026-08-08; no Tesla/Dojo talk

### Added 2026-09-13 (roadmap update)

- [Electrek — Tesla quietly closes its secret ~$2 billion AI hardware deal (2026-07-24)](https://electrek.co/2026/07/24/tesla-secret-2-billion-ai-hardware-acquisition-closes/)
- [Electrek — Tesla quietly discloses $2B AI hardware acquisition in Q1 2026 10-Q (2026-04-23)](https://electrek.co/2026/04/23/tesla-tsla-quietly-discloses-2-billion-ai-hardware-acquisition-10q/)
- [Electrek — Tesla loses chip engineer Shishuang Sun to DensityAI (2026-08-24)](https://electrek.co/2026/08/24/tesla-chip-engineer-shishuang-sun-densityai/)
- [Electrek — Terafab to run on gas, not Tesla solar (2026-08-10)](https://electrek.co/2026/08/10/musk-terafab-gas-power-not-tesla-solar/)
