# OpenAI-Broadcom Jalapeño — Hardware Architecture

*as_of: 2026-08-08*

Public hardware disclosure for this chip is thin. Everything below is tagged by evidence class:
**confirmed** (stated by OpenAI or Broadcom), **reported** (credible press, not vendor-stated),
**estimate** (third-party inference from photography), **inferred** (architectural reasoning from
the chip class), **not disclosed**. No value here is a guess dressed as a spec.

---

## Generation Overview

| Generation | Public name | Node | Memory | Compute engine | Networking | Status (2026-08-08) |
|---|---|---|---|---|---|---|
| Gen 1 | **Jalapeño** (OpenAI's "first Intelligence Processor"; unveiled 2026-06-24) | Not disclosed — 3nm-class (TSMC N3) per prior reporting | Not disclosed; **six HBM stacks** (estimate); generation (HBM3E vs HBM4) never stated | Tiled / systolic matrix engine (reported); dimensions not disclosed | Broadcom all-Ethernet; **Tomahawk confirmed**; Jericho4 inferred only | **Sampling** — engineering samples running ML workloads in lab at production target frequency and power |
| Gen 2 | Not named publicly ("Titan 2" in earlier reporting) | TSMC A16 (1.6nm) — **unconfirmed reporting** | Not disclosed | Not disclosed | Not disclosed | Reported dev start H2 2026, deploy 2027+; not vendor-stated |
| Gen 3+ | Not named | Not disclosed | Not disclosed | Not disclosed | Not disclosed | "Just the beginning of a multi-generation roadmap" (Hock Tan, Broadcom, 2026-06-24) — no specifics |

> The codename **"Project Titan"** carried in earlier revisions of this survey has **no primary source**.
> It traces only to aggregator/SEO sites and is not used by OpenAI or Broadcom. Treat as unsourced.

---

## Compute

| Attribute | Gen 1 (Jalapeño) | Evidence |
|---|---|---|
| Architecture class | Tiled / systolic matrix engine | Reported; the released wafer photography shows a "very regular, repeated, columnar" floorplan that third-party analysts read as consistent with this class. Not vendor-confirmed |
| Primary target | LLM inference | Confirmed ("LLM-optimized inference chip") |
| Model scope | Built for "current and future LLMs across the industry" — architecture is not locked to OpenAI models | Confirmed. **This is not a statement that the chip will be sold externally** |
| Array dimensions / tile size | Not disclosed | — |
| Peak FP16 / FP8 / INT8 | Not disclosed (any precision) | — |
| Data types | Not disclosed (BF16/FP8 inferred from the class and inference target) | Inferred |
| Sparsity support | Not disclosed | — |
| Attention hardware acceleration | Not disclosed | — |
| Programmability | Compiler-programmed (Triton backend, internal) — not a fixed-function transformer ASIC | Inferred from OpenAI compiler job postings |

**Bring-up evidence.** Engineering samples are running ML workloads in the lab at production target
frequency and power, **including GPT-5.3-Codex-Spark** (confirmed, 2026-06-24). This is the strongest
public signal that the compute path, compiler and runtime work end to end on real silicon. It is
*sampling*, not production and not deployment at scale.

**Development cycle (vendor marketing claim).** Initial design → manufacturing tape-out in **nine months**,
which the companies call "the fastest ASIC development cycle ever achieved in high-performance advanced
semiconductors". OpenAI says it used its own models to accelerate parts of design and optimization.
Unverifiable; recorded as a claim, not a fact.

---

## Package and Die

| Attribute | Value | Evidence |
|---|---|---|
| Compute chiplet area | ~25.46 mm × 33 mm ≈ **840 mm²** (EUV reticle limit ≈ 858 mm²) | **Third-party estimate** — Tom's Hardware measurement of released package/wafer photos; medium confidence |
| I/O chiplet | One separate I/O die on package | Third-party estimate (same analysis) |
| Structural dummy dies | Two | Third-party estimate |
| HBM stacks | **Six**, surrounding the compute chiplet | Third-party estimate (photo count) |
| Process node | Not disclosed on 2026-06-24; 3nm-class per prior 10 GW-program reporting | Reported, not confirmed |
| Transistor count | Not disclosed | — |
| TDP / clock | Not disclosed ("production target frequency and power" stated qualitatively only) | — |
| Packaging technology | Not disclosed (2.5D advanced packaging inferred from a chiplet + 6-stack HBM layout) | Inferred |

The package is therefore **multi-die** (compute chiplet + I/O chiplet + HBM + dummies) rather than a
monolithic accelerator — the clearest genuinely new architectural fact from the June 2026 disclosure,
even though it arrives as journalist inference rather than a vendor floorplan.

---

## Memory

| Attribute | Value | Evidence |
|---|---|---|
| Off-chip memory type | HBM — **generation not disclosed** (HBM3E vs HBM4 never stated by either company) | Not disclosed |
| Stack count | Six | Third-party estimate |
| Capacity per chip | Not disclosed | — |
| Bandwidth per chip | Not disclosed | — |
| Supplier | Not disclosed. A Samsung HBM4 exclusive-supply arrangement was *reported* in early 2026 on aggregator sourcing; the June 2026 disclosure neither confirms nor contradicts it | **Reported, not confirmed** — downgraded from "confirmed" in this revision |
| On-chip SRAM capacity / BW | Not disclosed (weight-stationary accumulator buffers + activation SRAM inferred from the class) | Inferred |
| Memory model | Compiler-managed scratchpad (inferred from the systolic/tiled class and the Triton toolchain) | Inferred |

---

## Interconnect

| Attribute | Value | Evidence |
|---|---|---|
| Scale-out fabric | Broadcom **Tomahawk** Ethernet silicon — "Broadcom's silicon implementation and networking technologies, including Tomahawk networking silicon, help bring the platform to large-scale production" | **Confirmed** (upgraded from "inferred" in this revision) |
| Scale-up fabric | Broadcom all-Ethernet. Jericho4 was previously attributed here; the announcement does **not** mention Jericho4 | **Inferred / unconfirmed** — no support gained |
| Scale-up bandwidth per chip | Not disclosed | — |
| Scale-up topology | Not disclosed | — |
| Port speed | Not disclosed (400/800 GbE inferred from the deployment window) | Inferred |
| Optical connectivity | Included in Broadcom's contribution | Confirmed (Oct 2025 announcement) |
| Host interface | Not disclosed (PCIe Gen5/6 inferred) | Inferred |
| Host CPU | Not disclosed. The Information reported Arm is designing a server-class CPU to anchor OpenAI's next-generation racks | **Rumor** — unnamed sources; never confirmed by Arm, OpenAI or Broadcom |

Broadcom is separately scheduled to present "Thor Ultra: An Ethernet NIC Chip Optimized for AI & HPC"
(Hemal Shah) at Hot Chips 38 on 2026-08-25. *Disclosure scheduled — content not yet public*; no
connection to Jalapeño has been stated, and nothing here is sourced from that talk.

---

## Rack / System

| Attribute | Value | Evidence |
|---|---|---|
| System integration partner | **Celestica** — "chip implementation, board, rack system integration, high-performance networking, and scalable production systems" | Confirmed, new as of 2026-06-24 |
| Chips per rack | Not disclosed | — |
| Power per rack | Not disclosed | — |
| Program scale | 10 GW of OpenAI-designed accelerators | Confirmed (Oct 2025) |
| Initial deployment | Targeted **by the end of 2026**, "expanding in the years ahead" | Confirmed as a *target*; not an achieved milestone |
| Named deployment partner | Microsoft — "we are enabling the deployment of gigawatt scale data centers with Microsoft and other partners beginning in 2026" (Hock Tan) | Confirmed vendor statement |
| Estimated program cost | $350B–$500B | Financial Times estimate |

---

## Performance

**No quantitative performance data of any kind has been disclosed.** The complete set of public
performance claims is:

- "performance per watt substantially better than current state-of-the-art" (vendor marketing, unquantified)
- realized utilization "much closer to theoretical peak" (vendor marketing, unquantified)
- a detailed technical report promised "in the coming months"

There is no MLPerf submission and no third-party benchmark. A "50% cheaper than Nvidia" figure
circulating on aggregator sites appears in no primary source and is excluded from this survey.

---

## Scheduled Disclosure

Richard Ho, Ravi Narayanaswami and Chris Leary (OpenAI) are on the Hot Chips 38 advance program with
"You Can Just Build Things … Chips", AI 2 session, Tuesday **2026-08-25**, 4:45–6:15 PM PDT.
*Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* No slides, abstract or
specification exists as of 2026-08-08. Re-scan after 2026-08-25 for the first real microarchitecture
disclosure.

---

## What Is Still Not Public

- Microarchitecture: PE array dimensions, tile size, dataflow, sparsity, attention handling
- Peak throughput at any precision; supported data types
- On-chip SRAM capacity and bandwidth
- HBM generation, supplier, capacity and bandwidth
- Process node (vendor-stated), transistor count, TDP, clock
- Scale-up fabric silicon and bandwidth; topology; host interface generation
- Chips per rack, rack power, rack topology
- ISA, compiler IR, runtime and driver architecture; no SDK
- Any independent or standardized benchmark result
