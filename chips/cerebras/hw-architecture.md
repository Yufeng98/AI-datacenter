# Cerebras WSE-3 — Hardware Architecture

*as_of: 2026-08-08*
*chip: cerebras (WSE-3 / CS-3) — no WSE-4 / CS-4 announced as of 2026-08-08*
*primary sources: https://www.cerebras.ai/chip, https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf, https://arxiv.org/html/2503.11698v1, https://newsroom.amd.com/news/aai-2026-cerebras-inference/*

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

| Generation | Announced | Process | Die Area | Cores | SRAM | SRAM BW | Peak FP16 | Status (2026-08-08) |
|---|---|---|---|---|---|---|---|---|
| WSE-1 | 2019 | TSMC 16nm | 46,225 mm² | 400,000 | 18 GB | ~9 PB/s | ~1 PF | Superseded |
| WSE-2 | 2021 | TSMC 7nm | 46,225 mm² | 850,000 | 40 GB | 20 PB/s | ~2 PF | Superseded |
| WSE-3 | 2024 | TSMC 5nm | **46,225 mm²** | 900,000 | 44 GB | 21 PB/s | 125 PF | **Current — shipping; production capacity being scaled ~7× through 2026 (Flex, 2026-07-09)** |
| WSE-4 / CS-4 | — | — | — | — | — | — | — | **Not announced.** No vendor announcement exists as of 2026-08-08 |

> **Die-area correction (2026-08-08):** the WSE-3 figure was previously recorded here as 46,255 mm². Cerebras's own chip page states **46,225 mm²** — the same reticle-limited wafer area as WSE-1 and WSE-2. The old value was a transposed digit and has been corrected throughout this repo.

> **No fourth generation as of 2026-08-08.** cerebras.ai/chip still describes only the WSE-3; the blog index from April through July 2026 contains no hardware-generation announcement; and the July 2026 Flex partnership expansion is explicitly a scale-up of **CS-3** manufacturing. A third-party wiki listing a WSE-4 as "anticipated" is speculation and is not carried here.

### Scheduled disclosure — Hot Chips 38 (content not yet public)

**"The Cerebras Rack-Scale Architecture for Wafer Scale Engine"**, Jean-Philippe (J.P.) Fricker, Cerebras — Hot Chips 38, Session AI 1, **Tuesday 2026-08-25, 2:15–4:15 PM PDT**, Stanford Memorial Auditorium. *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* The talk is 17 days in the future relative to this update; nothing about a rack-scale organisation beyond the CS-3 + MemoryX + SwarmX model documented above is confirmed, and the talk title is **not** a source for any specification. Re-scan after 2026-08-25.

---

## Architecture Layer Mapping

| Hardware Layer | Cerebras Component |
|---|---|
| Compute Engine | 900K PE array (FP16/BF16 8-wide SIMD, INT8 16-wide, own PC per PE) |
| Data Path | Swarm 2D mesh (5-port routers, 32-bit wavelets, 24 colors, ~1 ns/hop) |
| On-chip Memory | 44 GB distributed SRAM (48 KB/PE, single-cycle, no HBM, no cache hierarchy) |
| Off-chip Memory | MemoryX (DRAM+Flash, 4TB–2.4PB, weight-only, weight streaming protocol) |
| Host Interface / Package | CS-3 chassis (PCIe, water cooling, custom 300mm wafer PCB, cross-scribe-line metal) |
| Scale-up Interconnect | Swarm 2D mesh on-wafer (100 Pb/s aggregate, <1 pJ/bit, ~1 ns/hop) |
| Scale-out Interconnect | SwarmX (100GbE + RoCE RDMA, tree topology, up to 2,048 CS-3s) |
| Heterogeneous Disaggregation *(announced 2026-07-23, expected H2 2026)* | AMD Helios rack-scale (Instinct) prefill tier + WSE decode tier; tier-to-tier interconnect **not disclosed** |
