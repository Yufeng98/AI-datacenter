# Cerebras WSE-3 Hardware Architecture — Investigation Report

*as_of: 2026-04-05*
*chip: cerebras*
*device_class: Wafer-Scale Engine*
*primary sources: https://www.cerebras.ai/chip, https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf, https://arxiv.org/html/2503.11698v1, https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning, https://www.cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem*

---

## Overview

The Cerebras WSE-3 (Wafer-Scale Engine 3) is the world's largest semiconductor device, fabricated on a single TSMC 5nm 300mm silicon wafer measuring 46,225 mm² — approximately 56× larger than an NVIDIA H100 die. Rather than dicing wafers into conventional chips, Cerebras uses nearly the entire wafer as a single processor. The WSE-3 integrates 900,000 AI-optimized Processing Elements (PEs) arranged in a 2D rectangular mesh, 44 GB of distributed on-chip SRAM delivering 21 PB/s of memory bandwidth, and 4 trillion transistors achieving 125 FP16 petaflops peak. The architecture deliberately avoids off-chip HBM — instead co-locating 48 KB of SRAM with every PE to eliminate the memory wall for activation storage. Each PE contains its own 8-wide FP16/BF16 SIMD unit, 16-wide INT8 SIMD, scalar ALU, local instruction memory, and a 5-port mesh router. Yield at wafer scale is solved through redundant PE regions (1–1.5% extra cores) plus dynamic fabric routing that automatically steers wavelets around defective cores, achieving commercially viable yields from TSMC using custom cross-scribe-line metal interconnects.

---

## Architecture

### Module Structure

```
WSE-3 Wafer (46,225 mm², TSMC 5nm, 4 trillion transistors)
├── Processing Element (PE) Array — ~900,000 PEs in 2D rectangular mesh
│   ├── Per-PE Compute Unit
│   │   ├── Vector/Tensor ALU
│   │   │   ├── FP16/BF16 8-wide SIMD (125 PF aggregate)
│   │   │   ├── INT8 16-wide SIMD
│   │   │   └── FP32 scalar arithmetic
│   │   ├── Scalar ALU (control, addressing)
│   │   └── Local instruction memory + program counter (each PE runs own code)
│   ├── Per-PE On-chip SRAM — 48 KB private, single-cycle latency
│   │   └── Addressed via Data Structure Descriptors (DSDs) in CSL
│   └── Per-PE Fabric Router — 5-port (North / South / East / West / Local)
│       ├── 32-bit bi-directional ports in each cardinal direction
│       ├── Link bandwidth: 17.6–32 GB/s per hop
│       ├── Per-hop latency: ~1 ns
│       └── Energy: <1 pJ/bit per hop
│
├── Swarm 2D Mesh Fabric (scale-up, on-wafer)
│   ├── Hardware routing engine per PE (no software overhead)
│   ├── Aggregate bandwidth: ~100 Pb/s (all links combined)
│   ├── Message unit: 32-bit wavelet (data + color tag)
│   ├── 24 routable colors per PE (virtual channels)
│   └── Single-word active messages: wavelet arrival triggers CSL task
│
├── Defect Tolerance Infrastructure
│   ├── Core size: 0.05 mm² (100× smaller than H100 SM)
│   ├── Redundant PEs: ~1–1.5% extra per region
│   ├── Dynamic routing table: wavelets routed around failed PEs at runtime
│   └── Post-fabrication remapping: disabled defective PEs, enable spare
│
├── CS-3 System Package
│   ├── Custom PCB substrate with cross-scribe-line metal layers (TSMC collaboration)
│   ├── Water-cooling cold plate (direct to wafer surface)
│   ├── Host interface: PCIe + Ethernet to host CPU server
│   └── Physical form factor: water-cooled appliance, ~mini-fridge size
│
└── MemoryX Appliance (external, separate unit)
    ├── DRAM tier (fast, smaller) + Flash tier (dense, larger)
    ├── Elastic capacity: 4TB → 24TB → 36TB → 120TB → 1.2PB SKUs
    ├── Feeds weights to WSE via SwarmX on each training step
    └── Contains intelligence to schedule weight updates and prevent dependency bottlenecks

SwarmX Scale-out Fabric (connects multiple CS-3 systems)
    ├── Physical layer: 100GbE (but runs proprietary low-latency protocol, not TCP/IP)
    ├── Overlaid protocol: RoCE RDMA for weight DMA transfers
    ├── Topology: tree (MemoryX at root, CS-3 systems at leaves)
    ├── Function: broadcast weights to all CS-3s, reduce gradients from all CS-3s
    └── Scale: up to 2,048 CS-3 systems → quarter-zettaflops (10^23 FLOP/s)
```

### Key Abstractions

| Abstraction | Description |
|---|---|
| **Processing Element (PE)** | Atomic compute+memory unit: 48KB SRAM + FP16/INT8 SIMD + 5-port mesh router. 900K instances on one wafer. Each runs its own CSL program with its own PC. |
| **Swarm 2D Mesh** | On-wafer interconnect: hardware router per PE, 32-bit wavelet messages, ~1 ns/hop, 100 Pb/s aggregate. Eliminates NVLink/PCIe topology for intra-chip communication. |
| **Wavelet** | 32-bit data+color-tag message; the fundamental unit of inter-PE communication. Arrival triggers an associated CSL task on the receiving PE. |
| **Color** | Virtual channel ID (0–23 per PE). Routes wavelets through the mesh and maps to software task handlers in CSL code. |
| **MemoryX** | External weight-only storage appliance: DRAM+Flash, 4TB–2.4PB. Decouples model parameter storage from on-chip compute. Enables models up to 120T parameters. |
| **SwarmX** | Scale-out interconnect fabric: tree topology, broadcasts weights from MemoryX to CS-3s, reduces gradients from CS-3s to MemoryX. Pure data-parallel scale-out. |
| **Defect Tolerance Layer** | 0.05mm² PE size + 1–1.5% redundant PEs + dynamic routing remap = commercially viable wafer-scale manufacturing; 100× better defect tolerance per mm² than GPU designs. |

### WSE Generation Comparison

| Generation | Process | Die Size | Cores | SRAM | SRAM BW | Peak FP16 |
|---|---|---|---|---|---|---|
| WSE-1 (2019) | TSMC 16nm | 46,225 mm² | 400,000 | 18 GB | ~9 PB/s | ~1 PF |
| WSE-2 (2021) | TSMC 7nm | 46,225 mm² | 850,000 | 40 GB | ~20 PB/s | ~2 PF |
| WSE-3 (2024) | TSMC 5nm | 46,225 mm² | 900,000 | 44 GB | 21 PB/s | 125 PF |

### CS-3 System Specifications

| Parameter | Value |
|---|---|
| Chip | WSE-3 (one per CS-3) |
| Peak FP16 | 125 PetaFLOPS |
| On-chip SRAM | 44 GB, 21 PB/s |
| System TDP | ~23 kW |
| MemoryX (enterprise) | 24TB or 36TB |
| MemoryX (hyperscale) | 120TB or 1.2PB |
| Max model size | 24 trillion parameters (per CS-3) |
| Max cluster | 2,048 CS-3s |
| Max cluster perf | ~0.25 zettaflops |
| Form factor | Water-cooled appliance, ~mini-fridge |

### WSE-3 vs NVIDIA H100 Hardware Comparison

| Metric | WSE-3 | H100 SXM5 |
|---|---|---|
| FP16 peak | 125,000 TFLOPS | 1,979 TFLOPS |
| On-chip SRAM | 44 GB | ~50 MB L2 |
| Off-chip memory | None (activations); MemoryX (weights) | 80 GB HBM3 |
| Memory bandwidth | 21 PB/s (SRAM) | 3.35 TB/s (HBM) |
| Die area | 46,225 mm² | ~814 mm² |
| Core count | 900,000 PEs | 16,896 CUDA cores |
| Process node | TSMC 5nm | TSMC 4N |
| Scale-up fabric | Swarm 2D mesh (on-chip) | NVLink 4 (900 GB/s) |
| Scale-out | SwarmX (up to 2048 systems) | NVSwitch + InfiniBand |

---

## Data Flow

### On-Chip Compute — Layer Pipelined Mode (models ≤ 44GB weights)

```
1. Compiler maps model layers onto PE array regions (compile time)
   - Each PE assigned specific weights and activation tiles
   - No weight movement at runtime needed

2. Host streams input batch → CS-3 host interface → PE array edge
   - Data arrives as wavelets injected at mesh boundary

3. Forward pass — layer by layer:
   - PE receives input activation wavelet from neighbor (via color/task)
   - PE computes: input @ weights (FP16 SIMD) + bias + activation fn
   - PE sends output activation as wavelet to next-layer PE neighbors
   - Cross-column/row reductions via Swarm mesh (wavelet broadcast)
   - Activations stay in PE SRAM throughout (no off-chip access)

4. Backward pass:
   - Gradient wavelets flow in reverse through same PE array
   - Each PE computes: d_loss/d_weight (chain rule) + d_loss/d_activation
   - Gradients accumulated in PE SRAM

5. Optimizer step:
   - Weight update applied directly in PE SRAM
   - No weight transfer back to host until checkpoint

6. Output batch returned to host via SdkRuntime memcpy
```

### Weight Streaming Mode — Large Model Training (models > 44GB weights)

```
1. Setup: model weights stored in MemoryX (DRAM+Flash, up to 1.2PB)
   - MemoryX holds single authoritative weight copy for entire model
   - For multi-CS-3 clusters: one MemoryX serves all CS-3s

2. Forward pass — layer N:
   a. MemoryX reads layer N weights → SwarmX fabric
   b. SwarmX broadcasts layer N weights to all CS-3 systems in cluster
   c. Each CS-3 receives weights via host interface → distributes across PE SRAM
   d. Each CS-3 executes layer N forward pass:
      - Activations computed and stored in PE SRAM
      - Inter-PE communication via Swarm wavelet mesh
   e. Layer N activations remain in PE SRAM; layer N weights discarded
   f. Repeat for layer N+1, N+2, ... (streaming one layer at a time)

3. Backward pass — layer N (reverse order):
   a. MemoryX re-broadcasts layer N weights to all CS-3s
   b. Each CS-3 computes gradients for layer N
   c. Gradients streamed OUT from CS-3 → SwarmX
   d. SwarmX performs all-reduce: aggregates gradients from all CS-3s
   e. Reduced gradients sent to MemoryX
   f. MemoryX applies optimizer update to layer N weights
   g. Repeat for layer N-1, N-2, ... (streaming backward)

4. Pipelining: MemoryX prefetches layer N+1 weights while CS-3 executes layer N
   - Hides weight transfer latency
   - SwarmX bandwidth must exceed weight-transfer-per-layer / compute-time-per-layer
```

---

## Hardware Interface Points

### CS-3 Host Interface
- **PCIe**: CS-3 chassis connects to host server PCIe bus for control plane and data I/O
- **SdkRuntime API**: host-side Python/C++ API managing compile, load, launch, and memcpy lifecycle
- **Appliance API**: multi-job datacenter mode; handles concurrent job queuing, device allocation, program lifecycle management via `csctl` CLI

### WSE ↔ MemoryX / SwarmX Interface
- **SwarmX fabric**: 100GbE physical layer but runs Cerebras proprietary low-latency protocol (NOT TCP/IP; much lower overhead)
- **RoCE RDMA**: used for weight tensor bulk DMA transfers from MemoryX to CS-3
- **Tree topology**: MemoryX at root → CS-3 systems at leaves → O(log N) broadcast and reduce

### On-Chip PE Memory Interface
- Each PE addresses its 48KB SRAM directly — single-cycle, no cache miss, no TLB
- Addressing via Data Structure Descriptors (DSDs): typed, strided memory views defined in CSL
- No shared L2 cache; no coherence protocol; memory is PE-private

---

## Key Findings

1. **No HBM, no cache hierarchy — SRAM IS the memory system.** The 44 GB on-chip SRAM distributed across 900K PEs is the only memory accessible to compute during forward/backward pass. There is no L2, no LLC, no DRAM controller for activations. This eliminates the memory bandwidth wall that limits GPU training throughput. The tradeoff: 44 GB SRAM per wafer (vs. 80 GB HBM per H100) means full on-chip execution only for models ≤ 44 GB parameters.

2. **Two architecturally distinct execution modes.** (a) Layer Pipelined: weights and activations both on-chip; best throughput, used for smaller models. (b) Weight Streaming: weights offloaded to MemoryX, streamed layer-by-layer to WSE; enables models up to 120T parameters on a single CS-3 but introduces a weight-transfer pipeline.

3. **Wafer-scale yield is commercially solved.** 0.05mm² PE size (100× smaller than H100 SM) means a single random defect kills one PE instead of one SM. Combined with 1–1.5% redundant PEs per region and dynamic routing tables that steer wavelets around failed nodes, Cerebras achieves production-viable yields at TSMC 5nm 300mm.

4. **Swarm 2D mesh is the architectural differentiator.** 100 Pb/s aggregate on-chip bandwidth at <1 pJ/bit and ~1 ns/hop — more than 200× the bandwidth of NVIDIA's NVLink. Since all 900K cores are on one wafer, "inter-chip" communication becomes "intra-chip" wavelet routing. This makes all-reduce and broadcast operations within a single WSE trivially fast compared to multi-GPU clusters.

5. **Pure data parallelism across CS-3 cluster.** The MemoryX+SwarmX architecture is explicitly designed for data parallelism only — not model parallelism. Every CS-3 in a cluster runs the same model; MemoryX holds one weight copy; SwarmX broadcasts and reduces. This simplifies programming (no tensor parallelism API) but constrains architecture (cannot distribute one model's layers across CS-3s in weight-streaming mode).

6. **Deterministic, reproducible execution.** CSL programs on WSE are deterministic: fixed PE assignment, fixed wavelet routing, no CUDA-style warp divergence or memory bank conflicts. Same inputs always produce the same execution order — critical for debugging and reproducibility at large scale.

7. **Custom wafer packaging.** TSMC extends metal interconnect layers across the scribe lines (normally left blank for dicing). Cerebras developed custom PCB substrates, alignment processes, and cold-plate thermal designs specifically for the 300mm wafer package.

---

## Relation to Hardware Architecture Layers

| Layer | Cerebras Component | Unique Characteristic |
|---|---|---|
| Compute Engine | 900K PE array (FP16/BF16 8-wide SIMD, INT8 16-wide) | No SIMT/warp model; each PE has own PC and executes independently |
| Data Path | Swarm 2D mesh (5-port routers, wavelets, 24 colors) | On-wafer routing; activations never leave silicon during compute |
| On-chip Memory | 44 GB distributed SRAM (48 KB/PE, single-cycle) | No cache hierarchy; no HBM; PE-private SRAM addressed via DSDs |
| Off-chip Memory | MemoryX (DRAM+Flash, 4TB–2.4PB, weight-only) | Weights only; activations never stored here; elastic capacity scaling |
| Host Interface / Package | CS-3 chassis (PCIe, water cooling, custom 300mm wafer PCB) | TSMC cross-scribe-line metal; custom cold plate thermal management |
| Scale-up Interconnect | Swarm 2D mesh (100 Pb/s, <1 pJ/bit, ~1 ns/hop) | Replaces NVLink within the wafer; all 900K cores on one fabric |
| Scale-out Interconnect | SwarmX (100GbE+RoCE, tree topology, up to 2048 CS-3s) | Broadcasts weights / reduces gradients; pure data parallelism only |

---

# Investigation Update — 2026-08-08 (roadmap scan, window 2026-04-01 → 2026-08-08)

*as_of: 2026-08-08*
*scan type: roadmap / landscape refresh, adversarially verified*
*new primary sources: https://newsroom.amd.com/news/aai-2026-cerebras-inference/, https://www.cerebras.ai/chip, https://hotchips.org/advance-program/, https://investors.flex.com/news/news-details/2026/Flex-and-Cerebras-Expand-Partnership-to-Scale-American-Manufacturing-of-Cerebras-AI-Supercomputers/default.aspx, https://investors.cerebras.ai/news-releases/*

## 1. Correction to the existing baseline — WSE-3 die area

The 2026-04-05 investigation recorded the WSE-3 die area as **46,255 mm²**. Cerebras's own chip page states **46,225 mm²**, the same reticle-limited wafer area recorded for WSE-1 and WSE-2 in the generation table above. The prior figure was a transposed digit. All occurrences in `chips/cerebras/` and `research/cerebras/` have been corrected to 46,225 mm² as of this update.

Confidence: **confirmed** (vendor primary source). No other baseline figure changed: 900,000 PEs, 4 trillion transistors, 125 PF FP16, TSMC 5nm, 44 GB SRAM, 21 PB/s all re-confirmed.

## 2. Negative finding — no WSE-4, no CS-4

Explicit negative result as of 2026-08-08:

- cerebras.ai/chip describes only the WSE-3.
- The Cerebras blog index from April through July 2026 contains no hardware-generation announcement.
- The July 2026 Flex manufacturing expansion is a scale-up of **CS-3** assembly, which is corroborating evidence that no generational transition is imminent.
- A third-party wiki lists a "WSE-4" as *anticipated*. That is speculation, not an announcement, and is not carried into the survey.

Confidence: **confirmed negative** for the absence of an announcement. (Absence of an announcement is of course not evidence about Cerebras's internal roadmap.)

## 3. New architecture item — heterogeneous disaggregated inference (AMD Helios prefill + WSE decode)

Announced 2026-07-23 at AMD's *Advancing AI 2026* event. Verified against AMD's own newsroom, i.e. an independent primary source rather than only the Cerebras press release.

| Attribute | Finding | Confidence |
|---|---|---|
| Phase split | AMD Helios rack-scale (Instinct) = prefill / prompt / large-context; Cerebras WSE = decode / token generation | confirmed |
| Performance claim | "expected to deliver up to 5× higher tokens per second per watt" | confirmed **as a vendor claim** |
| Nature of the claim | AMD footnote: modelling by AMD Performance Labs and Cerebras, July 2026; metric **TPS/kW** at a comparable interactivity point on Kimi 2.6 1T; baseline is a **Cerebras-WSE-only** configuration | confirmed |
| Tier-to-tier interconnect | **not disclosed** | — |
| KV / activation handoff mechanism | **not disclosed** | — |
| Rack, power, and node topology of the combined system | **not disclosed** | — |
| Availability | "expected to become available initially through Cerebras Cloud in the second half of 2026"; Cerebras "plans to deploy AMD Helios systems in its data centers" | confirmed — **announced only**, not sampling, not shipping |

**Why this matters for the layer model.** Prior to this, every Cerebras scale-out path in the repo was homogeneous: SwarmX joining CS-3s in a data-parallel tree, MemoryX at the root. This introduces a *second vendor's* accelerator as an upstream compute tier for one phase of a single inference request. It qualifies (does not delete) the long-standing "CS-3 is a self-contained system" limitation. Because the interconnect is undisclosed, the architecture figure represents the link as a dashed, explicitly-labelled "NOT DISCLOSED" edge rather than inventing a fabric.

**Not carried:** a parallel AWS-Trainium-prefill / CS-3-decode arrangement appears only in secondary trade press (eWeek). Confidence **low**, excluded.

## 4. Scheduled disclosure — Hot Chips 38 rack-scale talk

"The Cerebras Rack-Scale Architecture for Wafer Scale Engine", Jean-Philippe (J.P.) Fricker, Cerebras — HC38 Session AI 1, Tuesday **2026-08-25, 2:15–4:15 PM PDT**, Stanford Memorial Auditorium. Confirmed present on the official advance program.

**Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** The talk is 17 days after this scan. Nothing about its technical content is knowable and the title must not be cited as the source of any specification. Follow-up scan required after 2026-08-25.

**Source-hygiene finding:** the repo's `announcing-the-cerebras-architecture-for-extreme-scale-ai` blog source is dated **2021-08-24** and covers WSE-2 / CS-2 weight streaming with MemoryX and SwarmX. It remains valid support for the weight-streaming architecture but is *not* evidence of any 2026 rack-scale product; the summary now annotates it accordingly.

## 5. Manufacturing and capacity context

| Item | Finding | Date | Status |
|---|---|---|---|
| Flex partnership expansion | New lines in Milpitas, California supporting an anticipated **~7× increase in CS-3 production capacity through 2026** | 2026-07-09 | Announced |
| European capacity | Target of **200 MW of AI compute in Europe by end of 2027**; first European capacity online by end of 2026; sited in France and the Nordics (announced by CEO Andrew Feldman at the RAISE Summit, Paris) | 2026-07-09 | Stated target — **not built capacity** |
| OpenAI agreement | Up to **750 MW of compute over three years, running through 2028**, reported worth **more than $10B** | 2026-01-14 | Signed purchase commitment — **not deployed capacity** |

## 6. Benchmarks

MLPerf status **not re-verified** in this pass. No Cerebras submission appears in any retrieved coverage of MLPerf Training v6.0 (results ~2026-06-16) or MLPerf Inference v6.0, but no official MLCommons results table could be retrieved, and absence from trade coverage is not proof of non-submission. The repo limitation "Not in MLPerf Training leaderboard" is retained and marked *last verified 2026-04-05, pending re-check*.

## 7. Method and limitations of this scan

The WebSearch budget was exhausted at the start of the verification pass, so independent verification was performed by fetching primary sources directly (AMD newsroom, hotchips.org, cerebras.ai/chip, investors.cerebras.ai listings, investors.flex.com) plus DuckDuckGo HTML/lite result pages. Several targets returned 403/429/timeout (CNBC direct, SEC EDGAR, HPCwire, DataCenterDynamics, the Cerebras IR PDF), so a small number of corporate figures rest on search-result snippets rather than full-article reads. Hardware figures in sections 1–3 all rest on directly fetched vendor pages.

---

# Investigation Update — 2026-09-13 (Hot Chips 38 disclosure: WSE-3T / CS-4 "Nexus")

*as_of: 2026-09-13*
*scan type: scheduled-disclosure follow-up (the Hot Chips 38 rack-scale talk flagged in §4 above has now been delivered)*
*primary source fetched directly: https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026 (2026-08-25)*
*secondary sources fetched directly for numeric detail the primary post omits: https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/ (2026-08-25), https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400 (2026-08-19)*
*WebSearch budget was unavailable for this pass (session-wide exhaustion); all verification below rests on directly-fetched articles, not search-result snippets.*

## H. WSE-3T identity: re-clocked WSE-3, not a new die

The Next Platform states WSE-3T uses the **same TSMC 5nm process**, the **same 900,000 cores**, and the **same 44 GB on-wafer SRAM** as WSE-3 — i.e. the unchanged 46,225 mm² die — clocked at **2.8 GHz vs. WSE-3's 1.4 GHz** (~2× overclock). ServeTheHome corroborates qualitatively ("WSE3-T... doubled the clockspeeds") without giving GHz figures itself. Confidence: **confirmed** (two independent secondary sources agree; Cerebras's own primary blog does not restate the GHz figures but does not contradict them either).

This means the repo's 2026-08-08 finding — "no WSE-4, no CS-4" — is **still correct** for a new-die-design reading. WSE-3T/CS-4 is a new SKU and new rack system, not a new wafer design.

## I. New bandwidth figures — one primary, one secondary

| Figure | Value | Confidence | Source |
|---|---|---|---|
| On-wafer SRAM bandwidth, WSE-3T | 43,000 TB/s (~43 PB/s) | confirmed (secondary, ServeTheHome, direct quote) | ServeTheHome |
| Aggregate on-wafer fabric bandwidth, WSE-3T | 53.5 PB/s | **confirmed, primary source** — Cerebras's own blog, verbatim quote | cerebras.ai |

The 43,000 TB/s figure is internally consistent with the clock-doubling finding (H): WSE-3's already-recorded 21 PB/s SRAM bandwidth × ~2 ≈ 42–43 PB/s. Flagging for a future pass: the repo's existing WSE-3 Swarm-fabric bandwidth figure ("~100 Pb/s aggregate, all links") does not unit-reconcile cleanly against the new 53.5 PB/s (=428 Pb/s) WSE-3T fabric figure; this was not resolved in this pass since the ~100 Pb/s figure has no clear original citation and the two numbers may describe different things (routing-network wavelet capacity vs. a narrower on-wafer fabric metric).

## J. Wafer-to-wafer interconnect — new architectural capability

ServeTheHome, direct quote: "The direct wafer links between the backpacks mean that the wafer-to-wafer latency can be as low as 2 microseconds. There is 2.4Tb/second of aggregate bandwidth from each wafer." Confidence: **confirmed** (secondary source, specific numeric quote attributed to the talk).

**Architectural significance:** this is the first inter-wafer, intra-rack interconnect Cerebras has disclosed. It directly qualifies the repo's Limitation #6 ("CS-3 is a single-system design — no NVLink-style peer-to-peer GPU tiling within one node"). It is *not* claimed to be NVLink-equivalent (protocol/topology beyond "direct wafer links" not disclosed), but it is a materially new scale-up primitive relative to CS-1–CS-3, where SwarmX (100GbE, tree, inter-*system*) was the only multi-wafer path.

## K. CS-4 "Nexus" rack — physical and performance detail

| Attribute | Finding | Confidence |
|---|---|---|
| WSE per rack | 3, each in its own "compute backpack" | confirmed (ServeTheHome) |
| Compute density vs CS-1–3 | "three times the compute per rack" | confirmed as vendor claim (Next Platform) |
| Power delivery | AC/DC converters 0.5mm from wafer (vs ~50mm on GPU); up to 30 modules/backpack, up to 277VAC → 54.5VDC | confirmed (ServeTheHome) |
| Cooling | Per-backpack water conditioning | confirmed (ServeTheHome) |
| Deployment | 50% fewer components, up to 3× faster deployment vs CS-1–3 | confirmed as vendor claim (Next Platform) |
| Availability | Early access now; GA "later in Q3 2026" | confirmed (Next Platform, 2026-08-19) |
| Performance vs CS-3 | "2x more tokens, at 10x more tokens per watt" | confirmed as **vendor claim**, not independently measured (ServeTheHome) |
| Performance vs GPU | "30x faster than a GPU" | confirmed as **vendor claim**; baseline/config not specified (ServeTheHome) |

**Causal inference (survey's own, not vendor-stated):** the 0.5mm power-delivery proximity plausibly explains how WSE-3T sustains ~2× clock without a new die — closer, higher-current delivery reduces I×R loss. This is analytical reasoning added by this survey, not a claim made in either source.

## L. Roadmap — CS-5 (2027) and CS-6

From Cerebras's own primary blog:
- **CS-5** (targeted 2027): up to 10,000 output tokens/s/user on mid-size/open models; up to 5,000 tokens/s/user and 3M tokens/s/MW for frontier models; support for models >50T parameters.
- **CS-6** (undated): wafer-scale SRAM and compute integrated with 3D-stacked DRAM, targeting an order-of-magnitude smaller inference footprint than current systems.

Both are forward roadmap statements, not shipping products; recorded as such.

## M. Explicitly not confirmed in this pass

- **Peak FLOPS for WSE-3T** (any precision) — absent from all three sources fetched.
- **"6× 200GbE per system"** — traced directly to The Next Platform author's own hedged speculation: "I *think* the wafer I/O module has six Ethernet ports running at 200 Gb/sec... Based on the specs, which say 'new higher speed wafer links'"; the same article states "We have tried to confirm many of these network details with Cerebras but have not heard back as yet." This is a journalist's inference, explicitly unconfirmed by the vendor. **Not promoted to a repo fact.**
- CS-4 pricing; WSE-3T transistor count (presumed unchanged at ~4T, not restated); exact wafer-to-wafer link protocol/topology.
- Any broader corporate/funding news beyond what the 2026-08-08 pass already recorded — this pass was scoped to the Hot Chips 38 disclosure specifically, not a full corporate-news refresh, given WebSearch budget constraints.

## Sources added 2026-09-13

- https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026 — primary, fetched directly twice (general pass + targeted follow-up for bandwidth/latency/clock figures)
- https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/ — fetched directly; primary numeric source for SRAM bandwidth, wafer-to-wafer interconnect, CS-4 rack detail, performance claims
- https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400 — fetched directly twice (general pass + targeted follow-up confirming the 200GbE figure is unconfirmed speculation)
