# Cerebras WSE-3 — Hardware Architecture

*as_of: 2026-09-13*
*chip: cerebras (WSE-3 / CS-3; WSE-3T / CS-4 "Nexus" as of 2026-08-25 — see Update below). No WSE-4 (new die design) announced as of 2026-09-13.*
*primary sources: https://www.cerebras.ai/chip, https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf, https://arxiv.org/html/2503.11698v1, https://newsroom.amd.com/news/aai-2026-cerebras-inference/, https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026*

---

## The Wafer as the Chip

Cerebras makes one fundamental architectural decision that shapes everything else: the entire 300mm silicon wafer IS the chip. Rather than dicing a TSMC 5nm wafer into ~80 individual dies, Cerebras uses cross-scribe-line metal interconnects to connect die regions across what would normally be dicing boundaries, yielding a single 46,225 mm² processor. This is 56× larger than an H100 die (814 mm²) and makes inter-PE communication an on-chip problem rather than a multi-chip or networked problem.

---

## Processing Element (PE) Architecture

Each of the 900,000 PEs on the WSE-3 is a self-contained compute+memory unit:

```
PE (0.05 mm² each, ~900,000 total)
├── Compute Unit
│   ├── Vector ALU: FP16/BF16 8-wide SIMD
│   ├── Vector ALU: INT8 16-wide SIMD
│   ├── Scalar ALU: FP32, INT32, control
│   └── Local instruction cache + program counter (independent per PE)
├── On-PE SRAM: 48 KB (single-cycle access, no cache hierarchy)
│   └── Addressed via Data Structure Descriptors (DSDs)
└── 5-port Fabric Router
    ├── 4 bidirectional ports: North, South, East, West
    ├── 1 port: Local (PE ↔ router)
    ├── 24 independent routable colors (virtual channels)
    ├── Link bandwidth: 17.6–32 GB/s per port
    └── Per-hop latency: ~1 ns
```

**Key**: Every PE has its own program counter and runs independently. There is no SIMT warp model, no SM grouping, no shared L1. Each PE is sovereign.

---

## Memory Architecture

| Level | Capacity | Bandwidth | Access Latency | Notes |
|---|---|---|---|---|
| Per-PE SRAM | 48 KB | Full ALU bandwidth | Single cycle | PE-private; no coherence |
| Total on-chip SRAM | 44 GB | 21 PB/s | Single cycle | Distributed; no HBM |
| MemoryX (external) | 4TB – 2.4PB | Dedicated link | ms (prefetched) | Weights only; not for activations |

**There is no HBM.** The 44 GB of on-chip SRAM across 900K PEs IS the working memory. This provides 21 PB/s of total memory bandwidth — 7,000× the bandwidth of a single H100's HBM. The tradeoff is density: SRAM is ~10× less dense than DRAM/HBM. Cerebras compensates by using the entire wafer.

---

## Swarm 2D Mesh Fabric (Scale-up)

The Swarm fabric connects all 900K PEs in a 2D rectangular mesh:

```
On-chip 2D Mesh (Swarm):
- Topology: 2D rectangular grid (all 900K PEs)
- Router: 5-port per PE (NSEW + local)
- Message unit: 32-bit wavelet (data + color tag)
- Virtual channels: 24 colors per PE
- Aggregate bandwidth: ~100 Pb/s (all links)
- Per-hop latency: ~1 ns
- Energy per bit: <1 pJ/hop
- Software overhead: none (hardware router; no MPI/TCP)
```

Wavelets are the fundamental messaging primitive. A 32-bit wavelet carries data and a color tag. The hardware router forwards it toward its destination based on the color's routing table. When it arrives, the destination PE's associated CSL task activates automatically.

---

## Defect Tolerance

Manufacturing a 46,225 mm² chip without defect tolerance would be impossible — a single defect kills the chip. Cerebras solves this with three mechanisms:

1. **Small PE size**: 0.05 mm² per PE = 100× smaller than an H100 SM (~5 mm²). A random defect kills one PE instead of one SM. On an apple-to-apples comparison, a defect impacts 100× less silicon area.

2. **Redundant PEs**: Each region includes ~1–1.5% extra PEs. When no defects are present, these are disabled. When a defect is detected, the local spare PE replaces the defective one.

3. **Dynamic fabric routing**: The mesh routing tables are programmable post-fabrication. Wavelets that would have passed through a defective PE are automatically rerouted around it. The surrounding PEs simply use slightly longer paths (1–2 extra hops).

The result: TSMC 5nm 300mm wafers can be manufactured at commercially viable yields.

---

## Execution Modes

### Layer Pipelined Mode (models ≤ 44 GB weights)

All model weights AND activations reside in on-chip SRAM throughout the entire forward+backward pass. No off-chip memory access occurs during compute. This delivers the maximum possible compute utilization.

```
Forward pass:
  Input batch → PE boundary (wavelet injection)
  Layer 1: PEs execute with local weights+activations → wavelet outputs
  Layer 2: activations from Layer 1 PEs → Layer 2 PEs → outputs
  ...
  Output layer: results at PE array boundary → host

Backward pass:
  Gradient wavelets flow in reverse through same PE array
  Weight updates applied directly to PE SRAM
  → weights persist on-chip for next batch
```

### Weight Streaming Mode (models > 44 GB weights, up to 120T)

Weights are stored externally in MemoryX. Each training step streams one layer's weights at a time:

```
For layer N during forward pass:
  1. MemoryX reads layer N weights → SwarmX
  2. SwarmX broadcasts to all CS-3 systems in cluster
  3. CS-3 distributes weights to PE SRAM for layer N
  4. PEs execute layer N forward: activations held in PE SRAM
  5. Layer N weights discarded; layer N+1 weights prefetched (pipelined)

For layer N during backward pass (reverse order):
  1. MemoryX re-broadcasts layer N weights
  2. PEs compute layer N gradients
  3. Gradients streamed out → SwarmX all-reduce → MemoryX
  4. MemoryX applies optimizer update to stored layer N weights
```

---

## CS-3 System Specifications

| Parameter | Value |
|---|---|
| Chip | 1× WSE-3 |
| Peak FP16 | 125 PetaFLOPS |
| On-chip SRAM | 44 GB, 21 PB/s |
| System TDP | ~23 kW |
| MemoryX (enterprise) | 24 TB or 36 TB |
| MemoryX (hyperscale) | 120 TB or 1.2 PB |
| Max model size | 24 trillion parameters (per CS-3) |
| Cluster size | Up to 2,048 CS-3 via SwarmX |
| Cluster peak | ~0.25 zettaflops FP16 |
| Form factor | Water-cooled appliance, ~mini-fridge |

---

## SwarmX Scale-out Fabric

SwarmX is the inter-system fabric for multi-CS-3 clusters:

```
SwarmX (scale-out):
- Physical layer: 100GbE
- Protocol: Cerebras proprietary (NOT TCP/IP; much lower overhead)
- RDMA: RoCE for bulk weight tensor DMA
- Topology: tree (MemoryX at root, CS-3s at leaves)
- Function: broadcast layer weights to all CS-3s + reduce gradients
- Scale: up to 2,048 CS-3 systems
- Programming model: pure data parallelism only (no tensor/pipeline parallel)
```

---

## Heterogeneous Disaggregated Inference — AMD Helios prefill + WSE decode (ANNOUNCED 2026-07-23)

*Added 2026-08-08. Primary source: AMD newsroom, https://newsroom.amd.com/news/aai-2026-cerebras-inference/ (announced at AMD Advancing AI 2026); corroborated by the Cerebras press release.*

Every subsystem above describes the CS-3 as a self-contained system: the wafer computes, MemoryX holds weights, SwarmX joins CS-3s in a data-parallel tree. On 2026-07-23 AMD and Cerebras announced an inference product that breaks that framing by splitting the two phases of LLM inference across **two different vendors' hardware**:

```
Prompt / long-context (PREFILL — compute-bound, high arithmetic intensity)
        │
   AMD Helios rack-scale (Instinct GPUs)
        │
        │  KV state / activations handed off
        │  ── interconnect between the two tiers: NOT DISCLOSED ──
        ▼
   Cerebras Wafer-Scale Engine
        │
Token generation (DECODE — memory-bandwidth-bound, latency-critical)
```

The rationale is a clean match to the two subsystems' strengths: prefill is FLOP-hungry and tolerant of HBM latency, which suits a GPU rack; decode is bandwidth- and latency-bound per token, which is exactly what 44 GB of SRAM at 21 PB/s is built for.

| Aspect | Value |
|---|---|
| Prefill tier | AMD Helios rack-scale (Instinct) — prompt and large-context processing |
| Decode tier | Cerebras WSE — token generation |
| Vendor claim | "expected to deliver up to **5× higher tokens per second per watt**" |
| Nature of the claim | **Modelling by AMD Performance Labs and Cerebras, July 2026** — not a measurement. Footnote unit is tokens/s/**kilowatt** at a comparable interactivity point on the Kimi 2.6 1T model, versus a **Cerebras-WSE-only** configuration |
| Baseline | Cerebras WSE only (**not** a GPU baseline) |
| Tier-to-tier interconnect | **not disclosed** |
| Memory/KV handoff mechanism | **not disclosed** |
| Status | **Announced.** Expected availability "initially through Cerebras Cloud in the second half of 2026"; Cerebras plans to deploy AMD Helios systems in its own data centers. Not sampling, not shipping, not deployed at scale |

⚠️ **Unit mismatch in the vendor materials:** the headline says tokens/s/W, the substantiating footnote measures tokens/s/kW. Both figures come from the same vendors' modelling. Treat the 5× as a marketing projection.

A parallel AWS-Trainium-prefill / CS-3-decode arrangement appears only in secondary trade press and is **not carried** in this survey (low confidence, no primary source).

---

## WSE Generation History

| Generation | Announced | Process | Die Area | Cores | SRAM | SRAM BW | Peak FP16 | Status (2026-09-13) |
|---|---|---|---|---|---|---|---|---|
| WSE-1 | 2019 | TSMC 16nm | 46,225 mm² | 400,000 | 18 GB | ~9 PB/s | ~1 PF | Superseded |
| WSE-2 | 2021 | TSMC 7nm | 46,225 mm² | 850,000 | 40 GB | 20 PB/s | ~2 PF | Superseded |
| WSE-3 | 2024 | TSMC 5nm | **46,225 mm²** | 900,000 | 44 GB | 21 PB/s | 125 PF | **Shipping; production capacity being scaled ~7× through 2026 (Flex, 2026-07-09)** |
| **WSE-3T** | **2026-08-25 (Hot Chips 38)** | **TSMC 5nm — same die as WSE-3** | **46,225 mm² (unchanged)** | **900,000 (unchanged)** | **44 GB (unchanged)** | **43,000 TB/s (~43 PB/s) — ~2× WSE-3's 21 PB/s** | **not stated in PF; see FLOPS note below** | **Disclosed at Hot Chips 38; ships inside CS-4 "Nexus" racks, early access now, GA target late Q3 2026** |
| WSE-4 | — | — | — | — | — | — | — | **Not announced.** No new-die-design announcement exists as of 2026-09-13 (WSE-3T is a higher-clocked WSE-3, not a new design — see Update below) |

**FLOPS note on WSE-3T:** neither of the two Hot Chips 38 secondary sources used for this pass (ServeTheHome, Next Platform) states a peak-FLOPS figure for WSE-3T. The SRAM-bandwidth figure (43,000 TB/s) and the clock figures below are the only quantitative compute-side numbers found; a peak-PFLOPS figure is **not disclosed**. If peak FLOPS scales with clock (same core count, same die), a naive doubling of WSE-3's 125 PF FP16 would suggest ~250 PF — but this is an unsourced extrapolation by the survey, not a vendor figure, and is not recorded as a spec.

> **Die-area correction (2026-08-08):** the WSE-3 figure was previously recorded here as 46,255 mm². Cerebras's own chip page states **46,225 mm²** — the same reticle-limited wafer area as WSE-1 and WSE-2. The old value was a transposed digit and has been corrected throughout this repo.

> **No fourth generation as of 2026-08-08.** cerebras.ai/chip still describes only the WSE-3; the blog index from April through July 2026 contains no hardware-generation announcement; and the July 2026 Flex partnership expansion is explicitly a scale-up of **CS-3** manufacturing. A third-party wiki listing a WSE-4 as "anticipated" is speculation and is not carried here.

### Hot Chips 38 disclosure — delivered 2026-08-25 (resolved; see Update below)

**"The Cerebras Rack-Scale Architecture for Wafer Scale Engine"**, Jean-Philippe (J.P.) Fricker, Cerebras — Hot Chips 38, Session AI 1, **Tuesday 2026-08-25, 2:15–4:15 PM PDT**, Stanford Memorial Auditorium. As of the 2026-08-08 scan this was a scheduled, not-yet-public disclosure. It has since been delivered and disclosed the **WSE-3T** chip and **CS-4 "Nexus"** rack-scale system — see "## Update — 2026-09-13 (Hot Chips 38: WSE-3T / CS-4 Nexus)" below for the full content.

---

## Update — 2026-09-13 (Hot Chips 38: WSE-3T / CS-4 "Nexus")

*Window: 2026-08-08 → 2026-09-13. Primary source: Cerebras's own Hot Chips 38 deep-dive blog post, https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026 (2026-08-25) — fetched directly. Secondary sources for numeric detail the primary post omits: ServeTheHome's rack-scale-talk writeup, https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/ (2026-08-25), and The Next Platform's WSE-3T writeup, https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400 (2026-08-19). This resolves the disclosure the 2026-08-08 pass had flagged as scheduled.*

**Headline: WSE-3T is a re-clocked WSE-3, not a new die.** Next Platform is explicit: WSE-3T is fabricated on the **same TSMC 5nm process** as WSE-3, with the **same 900,000 cores** and the **same 44 GB on-wafer SRAM** — i.e. the same physical design (46,225 mm² die, unchanged). What changed is clock speed: **1.4 GHz (WSE-3) → 2.8 GHz (WSE-3T)**, roughly a **2× overclock**, corroborated qualitatively by ServeTheHome ("WSE3-T... doubled the clockspeeds"). This is enabled by a new power-delivery architecture (below), not a new mask set. The repo's 2026-08-08 "no WSE-4" finding is **unaffected and remains correct** — this is not a fourth-generation die.

### A. WSE-3T on-wafer bandwidth figures — new, both primary and secondary confirmed

| Figure | Value | Source |
|---|---|---|
| On-wafer SRAM (memory) bandwidth | **43,000 TB/s** (~43 PB/s) | ServeTheHome: "By relying entirely on on-chip SRAM, WSE3-T offers 43,000 TB/second of memory bandwidth" — roughly 2× WSE-3's already-recorded 21 PB/s, consistent with the clock doubling |
| Aggregate on-wafer fabric bandwidth | **53.5 PB/s** | Cerebras primary blog, verbatim: "A single Cerebras WSE-3T provides 53.5 petabytes per second of aggregate on-wafer fabric bandwidth" — Cerebras compares this to "more than 200 times the NVL72 rack's scale-up bandwidth" (260 TB/s) |

These are two distinct figures (SRAM bandwidth vs. on-wafer fabric/routing bandwidth) and should not be conflated. Note for future reconciliation: the repo's existing WSE-3 Swarm-fabric figure ("~100 Pb/s aggregate, all links" — unsourced/estimated) does not cleanly reconcile in units with the new 53.5 PB/s WSE-3T figure (53.5 PB/s = 428 Pb/s); this is flagged for the next pass rather than resolved here, since the two figures may describe different things (total wavelet-network capacity vs. a narrower fabric metric) and only the new number has a primary citation.

### B. Wafer-to-wafer interconnect — new capability, updates Limitation #6

**This is the most architecturally significant new fact.** ServeTheHome, quoting the talk: "The direct wafer links between the backpacks mean that the wafer-to-wafer latency can be as low as **2 microseconds**. There is **2.4 Tb/second of aggregate bandwidth** from each wafer." Each CS-4 "compute backpack" houses one WSE-3T; a Nexus rack holds **three** backpacks (below) with **direct wafer-to-wafer links** between them.

This qualifies the repo's long-standing Limitation #6 ("CS-3 is a single-system design — no NVLink-style peer-to-peer GPU tiling within one node"). As of CS-4/Nexus, there **is** a direct, low-latency inter-wafer interconnect within a rack — not NVLink-equivalent in the GPU sense (topology and protocol not disclosed beyond "direct wafer links"), but a materially new scale-up capability that did not exist in CS-1 through CS-3, where SwarmX (100GbE, tree topology) was the only inter-system path.

### C. CS-4 "Nexus" rack architecture

| Attribute | Value | Source |
|---|---|---|
| WSE per rack | **3** ("three Wafer-Scale Engines, each housed in a modular compute backpack") | ServeTheHome |
| Compute density vs CS-1–CS-3 | "**three times the compute per rack** as the CS-1 through CS-3 machines" | Next Platform |
| Power delivery | AC/DC converters placed **0.5 mm from the wafer** (vs. ~50 mm on a GPU), "delivering nearly twice as much power"; up to **30 power-supply modules per backpack**, accepting up to **277 V AC** and delivering **54.5 V DC** | ServeTheHome |
| Cooling | Each backpack has its own water-conditioning system | ServeTheHome |
| Deployment | "50 percent fewer components" and "up to 3× faster" to deploy vs. CS-1–CS-3 | Next Platform |
| Availability | **Early access to select customers now** (as of 2026-08-19/25); **GA targeted later in Q3 2026** | Next Platform |

This new power-delivery architecture (converters 0.5 mm from the wafer) is the likely enabler of WSE-3T's clock doubling — closer, higher-current power delivery reduces I×R losses and supports higher sustained clock at the same or better power efficiency. This causal link is the survey's inference, not a vendor claim.

### D. Performance claims vs. CS-3 (vendor claims)

ServeTheHome, quoting the talk: **"CS-4 offers 2x more tokens, at 10x more tokens per watt"** compared to CS-3, and separately claims CS-4 is **"30x faster than a GPU"** (baseline GPU/config not specified in the retrieved coverage — treat as an unqualified vendor claim). All three figures are **vendor claims**, not independently measured.

### E. Roadmap — CS-5 (2027) and CS-6

From the Cerebras primary blog (2026-08-25):

- **CS-5**, targeted for **2027**: "up to 10,000 output tokens per second per user" on mid-size/open models; "up to 5,000 output tokens per second per user and 3 million tokens per second per megawatt" for frontier models; designed to support models with "more than 50 trillion parameters."
- **CS-6** (no date given): described as integrating "wafer-scale SRAM and compute with **3D-stacked DRAM**", aimed at "ultrafast inference in an order-of-magnitude smaller system footprint." No further architectural or timing detail is given.

Both are forward-looking roadmap statements from Cerebras's own blog, not shipping products.

### F. Corporate — IPO re-confirmed, no new items

The 2026-08-08 pass had already fully recorded Cerebras's IPO (priced 2026-05-13 at $185/share, ~$5.55B raised, began trading 2026-05-14 as Nasdaq: CBRS, closed +68% on day one, ~$106.75B fully diluted). Independent re-check this pass (Wikipedia, cross-referencing the existing Reuters/CNBC sourcing already in this repo) confirms the same figures with no discrepancy: IPO date 2026-05-14 (trading start), $5.55B gross raised, 30M shares at $185. No new corporate/capacity events were found in this narrow pass beyond what the 2026-08-08 update already recorded; a broader corporate-news sweep was out of scope given the WebSearch budget constraints of this pass (see Method note in the research investigation file).

### G. Still not disclosed

Peak PFLOPS for WSE-3T (any precision); CS-4 pricing; WSE-3T transistor count (presumed unchanged at 4 trillion, same die, but not restated in the Hot Chips 38 coverage); exact wafer-to-wafer link protocol/topology beyond "direct wafer links."

**Scale-out NIC count — explicitly journalist speculation, not a Cerebras figure.** A "6× 200GbE per system" number circulates in some secondary commentary. Checked directly against The Next Platform's own article: the author writes "I *think* the wafer I/O module has six Ethernet ports running at 200 Gb/sec... Based on the specs, which say 'new higher speed wafer links'" and explicitly states "We have tried to confirm many of these network details with Cerebras but have not heard back as yet." This is the article's own reasoned guess, not a disclosed Cerebras specification, and it is **not recorded as confirmed** here.

---

## Architecture Layer Mapping

| Hardware Layer | Cerebras Component |
|---|---|
| Compute Engine | 900K PE array (FP16/BF16 8-wide SIMD, INT8 16-wide, own PC per PE); WSE-3T is the same array at ~2× clock (2.8 GHz vs 1.4 GHz) |
| Data Path | Swarm 2D mesh (5-port routers, 32-bit wavelets, 24 colors, ~1 ns/hop) |
| On-chip Memory | 44 GB distributed SRAM (48 KB/PE, single-cycle, no HBM, no cache hierarchy); WSE-3T: 43,000 TB/s SRAM bandwidth (~2× WSE-3's 21 PB/s) |
| Off-chip Memory | MemoryX (DRAM+Flash, 4TB–2.4PB, weight-only, weight streaming protocol) |
| Host Interface / Package | CS-3 chassis (PCIe, water cooling, custom 300mm wafer PCB, cross-scribe-line metal); CS-4 "Nexus" backpack: AC/DC converters 0.5mm from wafer, dedicated water-conditioning per backpack |
| Scale-up Interconnect | Swarm 2D mesh on-wafer (aggregate on-wafer fabric bandwidth 53.5 PB/s on WSE-3T, Hot Chips 38); **new (2026-08-25): direct wafer-to-wafer links** between the 3 WSE-3T backpacks in a CS-4 Nexus rack — 2.4 Tb/s aggregate bandwidth per wafer, ~2 µs latency |
| Scale-out Interconnect | SwarmX (100GbE + RoCE RDMA, tree topology, up to 2,048 CS-3s) |
| Heterogeneous Disaggregation *(announced 2026-07-23, expected H2 2026)* | AMD Helios rack-scale (Instinct) prefill tier + WSE decode tier; tier-to-tier interconnect **not disclosed** |

---

## Sources

- https://www.cerebras.ai/chip
- https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf
- https://arxiv.org/html/2503.11698v1
- https://newsroom.amd.com/news/aai-2026-cerebras-inference/

### Added 2026-09-13
- [Ultrafast Frontier Inference: Cerebras Deep Dive at Hot Chips 2026 — Cerebras (primary, 2026-08-25)](https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026) — 53.5 PB/s aggregate on-wafer fabric bandwidth (WSE-3T); CS-5 (2027) and CS-6 roadmap targets
- [Cerebras Talks Going Rack-Scale With Their WSEs at Hot Chips 2026 — ServeTheHome (2026-08-25)](https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/) — 43,000 TB/s SRAM bandwidth, 2.4 Tb/s / 2µs wafer-to-wafer interconnect, CS-4 Nexus rack (3 WSE/rack, power delivery detail), 2x tokens/10x tokens-per-W vs CS-3, 30x vs GPU claim
- [Cerebras Overclocks WSE-3 Waferscale Engine to Boost Inference Oomph in Nexus CS-4 — The Next Platform (2026-08-19)](https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400) — WSE-3T clock speed (1.4→2.8 GHz), unchanged core count/SRAM/process, CS-4 compute density and deployment claims, early access + late-Q3-2026 GA target; explicitly flags the 6×200GbE figure as the author's own unconfirmed speculation
