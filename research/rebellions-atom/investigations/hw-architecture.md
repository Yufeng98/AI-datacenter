# Rebellions ATOM — Hardware Architecture Investigation

*chip: rebellions-atom*
*as_of: 2026-09-13*
*sources: ATOM white paper, ATOM-Max blog, REBEL-Quad Hot Chips 2025, ServeTheHome analysis, Tom's Hardware ISSCC 2026, Next Platform REBEL analysis, EE Times chiplet roadmap; (2026-08-08 update) Rebellions newsroom 2026-04-10 / 2026-06-17 / 2026-07-23, Seoul Economic Daily EN 2026-07-23, MLCommons MLPerf Training v6.0*

---

## Overview

Rebellions has shipped two generations of AI inference accelerators and is deploying a third:

| Generation | Product | Process | Memory | Form Factor |
|---|---|---|---|---|
| Gen 1 | ATOM / ATOM-Max | Samsung 5nm | 16 GB GDDR6 | PCIe Gen5 FHFL |
| Gen 2 | REBEL Single | Samsung 4nm | HBM (per die) | PCIe card |
| Gen 2 Chiplet | REBEL-Quad (Rebel100) | Samsung 4nm | 144 GB HBM3e | PCIe 600 W card |

---

## 1. Compute Engine

### ATOM — CGRA + Neural Engines

The ATOM uses a **CGRA (Coarse-Grained Reconfigurable Array)** architecture. Processing Element (PE) tiles can be reprogrammed to carry out different functions, enabling flexible mapping of diverse AI operators.

| Specification | ATOM | ATOM-Max |
|---|---|---|
| Architecture | CGRA + Neural Engines | CGRA + Neural Engines (boosted) |
| Neural Engines | 8 | 8 (higher clock) |
| Peak FP16 | 32 TFLOPS | higher (undisclosed) |
| Peak INT8 | 128 TOPS | higher (undisclosed) |
| Process | Samsung 5nm | Samsung 5nm |
| Die area | not disclosed | not disclosed |

The CGRA PE tiles are controlled by a **Command Processor** that schedules operations. The Neural Engines contain tensor units, vector units, and scalar units, with input buffers (IBUFs) that expose a custom instruction set for fine-grained operand management.

### REBEL Single — Neural Core Architecture

The REBEL generation replaces the CGRA macro-tile with a more structured **Neural Core** design:

- **16 Neural Cores** per chip
- **8 Neural Cores per cluster**, linked via mesh interconnect through their SRAM blocks
- Each Neural Core contains: IBUF (input buffer with custom ISA), tensor units, vector units, scalar units, and load-store units
- **RISC ISA for Neural Engines**: custom instruction set visible at the compute library level; the RBLN Compiler's backend generates RISC ISA programs
- Built on **Samsung 4nm** process (one node ahead of ATOM)
- Target: frontier-scale MoE inference and multi-billion parameter LLMs

### REBEL-Quad (Rebel100) — Chiplet Architecture

REBEL-Quad is the world's first AI accelerator to adopt **UCIe-Advanced** for chiplet interconnect:

- **4 compute ASIC dies** (320 mm² each) in a single package
- **4 HBM3e stacks** (12Hi, 36 GB each = 144 GB total per package)
- **4 Integrated Silicon Capacitors (ISC)** for power supply decoupling
- Peak compute: **2,048 TFLOPS FP8** (2.048 PFLOPS)
- Presented at ISSCC 2026 as Rebel100: claimed equal to NVIDIA H200 at lower power

---

## 2. Data Path

### ATOM Data Flow

The ATOM executes AI operations as a **compiler-scheduled dataflow graph**:

1. **Host** sends compiled binary and input tensors via PCIe Gen5
2. **Command Processor** fetches instructions and coordinates Neural Engines
3. **Neural Engines** execute tiled GEMM, normalization, and nonlinear ops using CGRA PE arrays
4. **On-chip SRAM** (64 MB) holds intermediate activations; DRAM/SRAM scheduling is performed by the RBLN compiler's dependency analyzer at compile time
5. **GDDR6** (16 GB, 256 GB/s) holds weights and large activations
6. Outputs returned to host via PCIe Gen5

The RBLN compiler performs **global optimization** over the entire computational graph — scheduling, memory allocation, and parallel execution — at compile time, not at runtime. This is a static compilation model (compile-once, deploy anywhere).

### REBEL Data Flow

Same compile-once model, but with:
- 2-tier SRAM (64 MB L1 per-core distributed + 64 MB L2 shared) vs ATOM's flat 64 MB
- Mesh interconnect between neural core clusters for intra-chip data exchange

### REBEL-Quad Data Flow

- 4 ASIC dies communicate via **UCIe-Advanced** at 16 Gbps / die-to-die link, 4 TB/s aggregate die-to-die bandwidth
- Each die has its own HBM3e stack; UCIe enables cross-die data exchange
- The RBLN compiler schedules cross-die data movement via UCIe in the same static compilation model

---

## 3. On-chip Memory

| Memory | ATOM | REBEL Single | REBEL-Quad (per ASIC die) |
|---|---|---|---|
| L1 SRAM | 64 MB (shared) | 64 MB distributed (per-core) | 64 MB distributed (per-core) |
| L2 SRAM | — | 64 MB shared | 64 MB shared |
| Total | 64 MB | 128 MB | 128 MB |
| Management | SW-managed (compiler) | SW-managed (compiler) | SW-managed (compiler) |

Memory is **entirely software-managed** — no hardware caches. The RBLN compiler's dependency analysis and scheduling determines all SRAM allocation and DRAM access patterns at compile time.

---

## 4. Off-chip Memory

| Memory | ATOM | REBEL-Quad |
|---|---|---|
| Type | GDDR6 | HBM3e (12Hi) |
| Capacity | 16 GB | 144 GB (4 × 36 GB) |
| Bandwidth | 256 GB/s | 4.8 TB/s aggregate |
| Interface | — | 4 × HBM3e stack |

The transition from GDDR6 to HBM3e in REBEL-Quad represents an ~18.75× bandwidth increase and 9× capacity increase, enabling large LLM inference without model sharding.

---

## 5. Host Interface / Package

### ATOM Card (RBLN-CA12)
- **Form factor**: Full Height Full Length (FHFL), single PCIe slot
- **Host interface**: PCIe Gen5 x16 (~128 GB/s bidir)
- **Card-to-card**: PCIe Gen5 x16
- **TDP**: 60–130 W (configurable)
- **Multi-instance**: 16 hardware-isolated instances per card

### REBEL-Quad Package (Rebel100)
- **Form factor**: PCIe card (not OAM/SXM)
- **TDP**: ~600 W per package
- **Package**: 4 compute ASIC dies (320 mm² each) + 4 HBM3e stacks + 4 ISC on single interposer
- **Process**: Samsung 4nm for compute dies
- **Reference design**: 8 cards per air-cooled node

---

## 6. Scale-up Interconnect

### ATOM
- No native chip-to-chip interconnect beyond PCIe card-to-card
- Multi-card topologies via PCIe Gen5 x16 card-to-card interface

### REBEL-Quad Intra-Package
- **UCIe-Advanced**: First AI accelerator to use UCIe-Advanced
- Speed: 16 Gbps per die-to-die link
- Aggregate bandwidth: 4 TB/s across all chiplet interconnects
- OCP REBEL-Quad listed in Open Compute Project chiplets registry

### System-Level Scale-Up
- **RebelRack**: 4 nodes, 32 Rebel100 accelerators, quad-400 Gbps networking per node; 64 PFLOPS FP8; 153.6 TB/s aggregate memory BW
- **RebelPOD**: 8–128 nodes × 8 Rebel100 per node; 800 Gbps Ethernet scale-out

---

## 7. Scale-out Interconnect

- Ethernet-based (not InfiniBand)
- RebelRack: quad-400 Gbps (400GbE × 4) per node
- RebelPOD: 800 Gbps Ethernet
- Partners: SK Telecom, DOCOMO Innovations (Japan) for telecom-integrated deployments

---

## 8. Architecture Evolution Summary

```
ATOM (5nm, CGRA)          →  REBEL (4nm, Neural Cores)  →  REBEL-Quad (4nm, UCIe Chiplet)
8 Neural Engines               16 Neural Cores               4 × REBEL dies
64 MB SRAM                     128 MB SRAM                   4 × 128 MB SRAM
16 GB GDDR6, 256 GB/s          HBM (per die)                 144 GB HBM3e, 4.8 TB/s
32 TFLOPS FP16                 ~200–400 TFLOPS (est.)        2,048 TFLOPS FP8
Samsung 5nm                    Samsung 4nm                   Samsung 4nm
PCIe Gen5 FHFL, 60–130W        PCIe 600W                     PCIe 600W
CGRA PE tiles                  Mesh-linked neural cores      UCIe-Advanced die-to-die
```

---

## Key Differentiators

1. **Static compilation model**: All scheduling and memory allocation done at compile time by RBLN compiler; no runtime overhead
2. **CGRA/Neural Core flexibility**: Same hardware can handle CNN, Transformer, and generative AI workloads via reprogramming
3. **UCIe-Advanced first mover**: REBEL-Quad is the first AI accelerator to adopt UCIe-Advanced, enabling chiplet scaling without proprietary die-to-die protocols
4. **Energy efficiency focus**: 3.2× TPS/W vs top-tier GPU on Llama 3.3 70B; 50% lower power consumption
5. **Samsung ecosystem**: Deep Samsung fab + Arm architecture partnerships as strategic differentiators vs US/Taiwan-centric competitors

---

## Investigation Update — 2026-08-08 (window 2026-04-06 → 2026-08-08)

*Scope: what changed in Rebellions' hardware story between the 2026-04-05 baseline above and 2026-08-08. Prior sections are unchanged and remain accurate.*

### Finding 1 — No new silicon generation (high confidence)

Nothing in the window revises any die-level or package-level figure recorded in §§1–8. Specifically checked and found absent:

- No REBEL Gen-3 announcement.
- No tape-out, sampling, or spec disclosure for **REBEL-IO**, **REBEL-CPU** (or any successor part). These remain roadmap names with no published specifications.
- No cancellation, delay, or discontinuation signal for ATOM, ATOM-Max, or REBEL-Quad/Rebel100.
- No revised process node, die area, clock, SRAM capacity, HBM capacity/bandwidth, UCIe rate, or TFLOPS figure. All REBEL-Quad numbers still trace to Hot Chips 2025 and ISSCC 2026.

Benchmark absence (useful as a negative datapoint in a survey): Rebellions is **not** among the 24 MLPerf Training v6.0 submitters (published 2026-06-16 — AMD, ASUSTeK, Azure, Cisco, CoreWeave, Dell, Fujitsu, GigaComputing, Google, HPE, Inventec, Krai, Lambda, MITAC, Nebius, Netweb, NVIDIA, Oracle, QCT, SCITIX, Supermicro, tinycorp, TTA, Vultr), and was also absent from MLPerf Inference v6.0 (2026-04-01). Rebellions has no talk on the **Hot Chips 38** program (Aug 23–25 2026, Stanford). HC38 is in the future relative to this investigation; nothing on its program constitutes evidence. Arm's "Arm AGI: A Disaggregated, Chiplet-Based Server SoC…" talk (Monday, CPU 2 session) is *disclosure scheduled — content not yet public*.

### Finding 2 — RebelCard: new card-level product, no specs (2026-04-10)

The three-way MOU between Rebellions, SK Telecom and Arm introduces **RebelCard**, a module-type accelerator card carrying **Rebel100**. What the primary release actually says:

| Attribute | Disclosed value |
|---|---|
| Silicon | Rebel100 — "four NPU chiplets" |
| Memory | "5th-generation HBM" (HBM3E) — capacity and bandwidth **not disclosed** |
| Cooling | Air-cooled |
| Host CPU | **Arm AGI CPU**, built on **Arm Neoverse CSS V3**, described as "the first Arm-designed data center CPU" |
| Performance | **not disclosed** — only "comparable to current flagship GPUs" with better power efficiency |
| TDP | **not disclosed** |
| Ship date | **not disclosed** |

The "four NPU chiplets + 5th-gen HBM" description is consistent with the REBEL-Quad package already documented in §1; RebelCard is a **packaging/platform product**, not a new compute engine.

**Explicitly refuted secondary claim:** aggregator coverage of this MOU states a "Q3 2026 RebelCard release." The primary Rebellions release contains **no launch date at all**, and the Q3 2026 date could not be corroborated in any primary source. Do not record a RebelCard ship date.

**Status verbs:** MOU-stage co-development; validation planned in SK Telecom's AI datacenter, then global telecom/public-sector expansion. Not a design win, not sampling to customers, not shipping.

### Finding 3 — RebelServer and the A.X K1 run (2026-07-23)

A **single RebelServer** ran SK Telecom's sovereign LLM **A.X K1** — over 500 billion parameters, **Mixture-of-Experts** architecture — serving concurrent real-time requests behind a map-based agent demo. Confirmed independently by Seoul Economic Daily EN as well as the vendor release.

What was **not** published: latency, throughput, tokens/s, TPS/W, batch size, card count, or the number of Rebel100/ATOM devices in the server. The vendor claim is serving efficiency "on par with global high-performance GPU servers" — a marketing claim with no supporting number. Without a device count this run cannot be turned into a memory-capacity or per-package throughput statement, which is exactly the interesting question for a 500B+ MoE model on a 144 GB-per-package part.

Rebellions separately states it has run SKT's "A. Phone Call Summary" high-traffic production service for approximately one year.

*(Date correction: the announcement is 2026-07-23, not 07-24.)*

### Finding 4 — OEM channel: Giga Computing MOU (2026-06-17)

MOU with **Giga Computing** (GIGABYTE's server arm) to co-develop AI servers and rack-scale systems around RebelCard/Rebel100. Signed by Marshall Choy (Rebellions CBO) and Vincent Wang (Giga Computing CCO), reported as signed at Computex 2026 in Taipei. No timeline, no specifications, no product SKU. Intent to co-develop only.

### Finding 5 — Pre-existing repo weakness corrected: RebelRack specs are secondary-sourced

The RebelRack figures carried in §6 and in `chips/rebellions-atom/` (32 accelerators, 64 PFLOPS FP8, 4.6 TB HBM3e, 153.6 TB/s, quad-400 GbE per node) have **no primary-source corroboration**. Rebellions' own 2026-03-30 release that launched RebelRack and RebelPOD published **no technical specifications**. The figures derive from secondary coverage (The Register, 2026-03-30) and have been re-graded accordingly in the chip deliverables. They are not retracted — they are marked as reported-but-unconfirmed.

### Coverage gap

A UPI item dated 2026-08-07 ("SK Telecom, Rebellions expand Korean AI chip infrastructure") appeared in the search index but returned HTTP 403 on fetch. Developments in the last ~48 hours before this investigation are therefore unverified.

---

## Sources

- [ATOM Architecture: Finding the Sweet Spot for GenAI](https://rebellions.ai/atom-architecture-finding-the-sweet-spot-for-genai/)
- [ATOM-Max: Boosted Performance for Large-Scale Inference](https://rebellions.ai/atom-max-boosted-performance-for-large-scale-inference/)
- [REBEL-Quad Hot Chips 2025 Announcement](https://rebellions.ai/newsroom/rebellions-debuts-rebel-quad-at-hot-chips-2025-breaking-ais-energy-tax-with-high-performance-chiplet-innovation/)
- [ServeTheHome: REBEL-Quad UCIe and 144GB HBM3E](https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144tb-hbm3e-accelerator-at-hot-chips-2025/)
- [Tom's Hardware: ISSCC 2026 Rebel100](https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100)
- [Next Platform: REBEL Architecture Deep Dive](https://www.nextplatform.com/2025/12/23/rebellions-ai-puts-together-an-hbm-and-arm-alliance-to-take-on-nvidia/)
- [EE Times: Chiplet Roadmap](https://www.eetimes.com/rebellions-builds-chiplet-roadmap-merges-with-sapeon/)
- [The Register: RebelRack Rack-Scale Platform](https://www.theregister.com/2026/03/30/rebellions_ai_rackscale/)
- [OCP REBEL-Quad Listing](https://www.opencompute.org/chiplets/76/rebel-quad-ai-accelerator-ai-soc)

### Added 2026-08-08

- [Rebellions + SK Telecom + Arm MOU (RebelCard, Arm AGI CPU) — 2026-04-10](https://rebellions.ai/newsroom/rebellions-collaborates-with-sk-telecom-and-arm-targeting-sovereign-ai-and-telecom-infrastructure/)
- [Rebellions + Giga Computing MOU — 2026-06-17](https://rebellions.ai/newsroom/rebellions-and-giga-computing-sign-mou-to-develop-next-generation-ai-server-and-rack-scale-solutions/)
- [RebelServer runs SKT A.X K1 (500B+ MoE) — 2026-07-23](https://rebellions.ai/newsroom/rebelserver_run_k1/)
- [Seoul Economic Daily EN: A.X K1 on RebelServer — 2026-07-23](https://en.sedaily.com/technology/2026/07/23/rebellions-runs-skts-sovereign-model-ax-k1-on-npu-server)
- [Rebellions $400M pre-IPO + RebelRack/RebelPOD launch — 2026-03-30](https://rebellions.ai/newsroom/rebellions-closes-400-million-pre-ipo-and-launches-rebelrack-and-rebelpod-to-accelerate-global-expansion/) (primary; **no** technical specs — basis for the RebelRack sourcing re-grade)
- [MLPerf Training v6.0 results — 2026-06-16](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) (Rebellions absent from 24 submitters)
- [Hot Chips 38 program](https://hotchips.org/) (Aug 23–25 2026 — future; Rebellions absent)

---

## Investigation Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*No new silicon, no new spec, no new performance figure this window. The material development is corporate: NVIDIA is reported in early-stage talks with Rebellions. All prior-generation content above is retained unchanged.*

### Finding — NVIDIA in early talks with Rebellions (2026-08-21, Bloomberg)

Bloomberg reported (2026-08-21) that **NVIDIA is in early discussions with Rebellions** covering "a technical partnership, an investment or perhaps even an acquisition." Key facts, cross-checked against an eWeek summary (2026-08-24) of the same Bloomberg reporting:

- **NVIDIA CEO Jensen Huang met Rebellions CEO Sung-hyun Park** at NVIDIA's Santa Clara HQ the week before the eWeek article (i.e., ~2026-08-17 → 08-21).
- Described explicitly as **early-stage**; may not conclude in any deal.
- eWeek draws an explicit precedent: **NVIDIA used a similar structure with Groq in late 2025** — a nonexclusive technology license plus absorption of engineering staff (an "acqui-hire"-shaped deal, not a full acquisition). This is offered as the likely template, not a confirmed structure for a Rebellions deal.
- No investment amount, equity stake, or licensing scope has been disclosed for any Rebellions deal.
- Financial context repeated in the coverage (not new): ~$850M total funding, ~$2.3B last valuation (March 2026 pre-IPO round), backers including SK hynix, Samsung, Arm.
- **Explicitly absent from both the Bloomberg and eWeek reporting**: any mention of Rebel100 shipping timelines, Saudi Aramco, or IPO plans — none of the pre-verified claims about an Aramco-specific "H1 2027 deployment target" were corroborated by this coverage or by any other source found in this pass. **Treat "Rebel100 → Saudi Aramco deployment, H1 2027" as unconfirmed** — the only H1 2027 date on record anywhere in this research thread is the Reuters/CNBC (2026-07-08) **Korea IPO listing** target, which is a different event; Aramco appears in this repo only as an investor (Wa'ed Ventures, $15M, reported ~2024) and a pre-IPO round participant, never tied to a deployment date.

### Corroborating coverage gap note

The 2026-08-07 UPI item ("SK Telecom, Rebellions expand Korean AI chip infrastructure") flagged as HTTP-403-blocked at the 2026-08-08 pass was **re-attempted and could not be located or retrieved in this pass either** — neither the original URL nor a re-search surfaced a working copy. Still unverified.

### Not found / searched and absent

- No Rebel100/RebelCard confirmed shipment or ship-date (still "entering validation," per the 2026-08-08 baseline).
- No Sapeon-merger status change beyond the December 2024 completion already on record.
- No new ISSCC/Hot Chips/MLPerf disclosure.
- No confirmation of a Saudi Aramco deployment target of any kind (H1 2027 or otherwise).

### Sources added 2026-09-13

- [Bloomberg — Nvidia in Talks With Chip Startup Rebellions for Potential Deal (2026-08-21)](https://www.bloomberg.com/news/articles/2026-08-21/nvidia-in-talks-with-chip-startup-rebellions-for-potential-deal)
- [eWeek — Nvidia Eyes Rebellions Deal as AI Inference Competition Heats Up (2026-08-24)](https://www.eweek.com/news/nvidia-rebellions-ai-chip-potential-deal-apac-south-korea/)
