# Qualcomm Cloud AI 100/200 — HW Architecture Investigation

## Summary

The AIC100 SoC is built around 16 Hexagon AI cores, each with three specialized execution units (Tensor, Vector, Scalar). The architecture separates tensor computation from vector and scalar tasks, enabling high throughput for GEMM while also handling pre/post-processing efficiently. Large on-chip SRAM (144 MB/SoC) reduces DRAM pressure for LLM KV-cache and model weights.

---

## AI Core Architecture (×16 per SoC)

Each AI core is a Qualcomm Hexagon Q6 DSP enhanced with HVX (vector extensions) and HMX (matrix extensions):

### Tensor Unit (HMX)
- Two 2D MAC arrays: INT8 (8192 MACs/cycle) + FP16 (4096 MACs/cycle)
- 125+ instructions for linear algebra
- Targets: GEMM, Conv, MatMul

### Vector Unit (HVX)
- 512 INT8 ops/cycle, 256 FP16 ops/cycle
- 700+ instructions: AI, image processing, content verification
- Supports INT8/16, FP16/32

### Scalar Processor (Q6 core)
- 4-way VLIW
- 6 hardware threads
- Controls tensor/vector dispatch

**Key: Separation of concerns** — three units execute concurrently with independent pipelines.

---

## On-Chip Memory

- Per AIC100 SoC: 144 MB SRAM
- Ultra card (4× SoC): 576 MB SRAM total
- Three NoCs: Compute (186 GB/s), Memory, Config

## Off-Chip Memory

| SKU | DRAM type | Capacity | Bandwidth |
|---|---|---|---|
| Standard SoC | LPDDR4X (4×64b) | 32 GB | 136 GB/s |
| Ultra card | LPDDR4X | 128 GB | 548 GB/s |
| AI 200 (card) | LPDDR | 768 GB | not disclosed |

---

## Multi-SoC Configuration (Ultra Card)

- 4 AIC100 SoCs connected via PCIe switch
- Combined: 64 AI cores, 576 MB SRAM, 128 GB DRAM
- Single PCIe Gen4 ×16 host interface (logical)
- Enables model sharding across SoCs for large LLMs

---

## Performance

| SKU | INT8 TOPS | FP16 TFLOPS |
|---|---|---|
| Standard (1 SoC) | 400 | 200 |
| Ultra (4 SoC) | 870 | ~435 |

---

## Host Interface

- PCIe Gen4 ×8 (standard) / ×16 (Ultra)
- DMA engine for host ↔ device data transfer
- MQ DMA: multi-queue direct memory access for low-latency inference

---

## Key Design Decisions

1. **Hexagon DSP heritage**: leverages Qualcomm's mobile NPU expertise at data center scale
2. **Large SRAM**: 144 MB/SoC enables KV-cache on-chip for long-context LLM inference
3. **Multi-SoC Ultra card**: 4 dies on one PCIe card for 128 GB total — competitive with H100 80 GB SXM
4. **LPDDR not HBM**: lower bandwidth but sufficient for inference; lower cost/power
5. **Three-NoC architecture**: separate compute/memory/config traffic prevents contention

---

# Investigation Update — Qualcomm Dragonfly (2026-08-08)

*Scope of this section: the Qualcomm Investor Day of 2026-06-24 and subsequent releases through 2026-07-29. It is
additive — nothing above is superseded, because Qualcomm published nothing that contradicts the Cloud AI 100
figures. Confidence per fact group is recorded in the companion `hw-architecture.yaml`.*

## What was announced

Qualcomm rebranded its data center line **Qualcomm Dragonfly**. The Dragonfly press release names exactly four
products: the **C1000** CPU and the **AI200 / AI250 / AI300** accelerators. AI200 and AI250 are the October 2025
parts under new branding; AI300 and C1000 are new.

**Taxonomy warning for downstream writing.** A second four-way sequence circulated from Investor Day —
"connectivity → custom silicon → AI accelerator → CPU". That is **revenue-ramp sequencing from the financial
deck**, not a product taxonomy, and only two of its legs are attested (custom-silicon revenue from Q1 FY2027 and
AI-accelerator revenue in H2 FY2027, both via Futurum). Do not present it as the Dragonfly product line-up.

## HBC — High Bandwidth Compute

The single architecturally load-bearing disclosure. HBC is 3D-stacked **near-memory computing**: DRAM stacked over
the XPU logic die, connected via TSVs. Qualcomm's PR uses the phrase "near-memory computing"; the
DRAM-over-logic-via-TSV mechanism is described by The Register (2026-06-30).

| Fact | Value | Source class |
|---|---|---|
| HBC bandwidth per watt | 6× versus HBM | Qualcomm-stated, unaudited |
| HBC capacity per watt | 200× versus SRAM | Qualcomm-stated, unaudited |
| AI250 memory tier | HBC Gen 1 | Qualcomm PR |
| AI250 per-card bandwidth | "industry-leading 133 TB/s" | Qualcomm-stated, unaudited |
| AI250 vs AI200 | "18× increase in effective memory bandwidth" | Qualcomm-stated, unaudited |
| AI250 sampling | "Commercial sampling of HBC Gen 1 with AI250 is expected in mid-2027" (exact wording) | Qualcomm PR |
| AI250 per-card capacity | 768 GB on LPDDR5x | The Register — **conflicts** with Qualcomm attributing 768 GB/card to AI200 |
| AI300 memory tier | HBC Gen 2, 3D-stacked near-memory | Qualcomm PR |
| AI300 vs AI200 | "54× increase over AI200" in effective memory bandwidth | Qualcomm PR — The Register renders the same figure as 54× vs AI250; **prefer the PR** |
| AI300 perf/W | 4×–8× better vs existing GPU-based architectures (stated on a memory-bandwidth-per-watt-per-card basis) | Qualcomm-stated, unaudited |
| AI300 sampling | 2028 | Qualcomm PR |
| AI300 platform | Air- **and** direct-liquid-cooled rack-level platform; scales over UALink and ESUN | Qualcomm PR |

### The caveat that must travel with every one of these numbers

These are **"effective" memory bandwidth multipliers with undisclosed methodology**. The Register (2026-06-30)
reports that Qualcomm **declined to disclose peak FLOPS** for either AI250 or AI300 and **declined to explain the
methodology** behind the multipliers, and argues the figures are inflated by definition rather than physical:
Qualcomm's implied ~414 TB/s aggregated across 56 AI200 chips would require an implausible **~6,720-bit bus** on
standard LPDDR5x.

Because Qualcomm did not submit to MLPerf Inference v6.0 (24 submitters, published 2026-04-01) and has no talk on
the Hot Chips 2026 program (HC38, listed as **August 24–25, 2026**), **no independent measurement of any Qualcomm
data center part exists in this window**. Every Dragonfly performance figure in the repo is unaudited vendor
marketing and is labelled as such.

## Dragonfly C1000 CPU

New product line, announced only:

- 250+ cores, **chiplet** design, custom Qualcomm **Oryon** cores at **>5 GHz**
- ">2× better performance per watt" versus competitive server CPUs (Qualcomm estimate); The Register additionally
  reports a "30 percent more speed" claim
- **>2 TB/s PCIe Gen 7** connectivity, plus **CXL**
- Three configurations: agentic · general-purpose virtualization · AI head node
- Commercial availability expected **2028**
- Process node, TDP, cache hierarchy and memory channels: **not disclosed**

**Meta is a named customer.** "Qualcomm and Meta Announce Strategic Multi-Generation Agreement on Data Center
CPUs" (2026-06-24) has Qualcomm "in production starting in the second half of 2028". No volumes, pricing or
binding terms were disclosed — a supply agreement, not a deployment.

## Interconnect (first Qualcomm scale-up fabric in this survey)

- Scale-up: **UALink** and **ESUN** (Ethernet for Scale-Up Networking) — open standards, no proprietary fabric
- Scale-out: copper and optical Ethernet, **800G and 1.6T**, reach from intra-data-center up to a **20 km campus**
- **35+ named ecosystem supporters**, including Meta, Arista, Supermicro, Lenovo, Samsung SDS, SK hynix America
- Per-chip scale-up bandwidth, chips per rack and maximum system scale: **not disclosed**

## AI200 status

Still **pre-volume**. Qualcomm stated at Investor Day that AI200 is "sampling in fiscal 2026 on LPDDR5x" (reported
by Futurum only). No Qualcomm press release claims production shipment. The 768 GB/card and 160 kW rack figures
already in the repo date from the October 2025 launch and were **neither restated nor contradicted** in June 2026,
so they are retained with that provenance attached.

## Deliberately excluded

- The narrative that "commercial sampling expected mid-2027" represents a slip from an October 2025 "commercial
  availability in 2027" promise. The October 2025 PR could not be retrieved and The Register still describes AI250
  as "launching 2027". Plausible, unverified, **not asserted**.
- A 2026-06-16 report that Qualcomm was circling **Tenstorrent** in a ~$10B deal. Rumor only.
- Any AI300 memory capacity, process node, TDP or rack-power figure. None was published.

## Sources for this section

- https://www.qualcomm.com/news/releases/2026/06/qualcomm-unveils-comprehensive-data-center-roadmap-for-the-agent
- https://www.qualcomm.com/news/releases/2026/06/qualcomm-and-meta-announce-strategic-multi-generation-agreement-
- https://www.qualcomm.com/news/releases/2026/06/qualcomm-accelerates-diversification-with-comprehensive-strategy
- https://www.theregister.com/systems/2026/06/30/qualcomms-proposed-solution-to-catch-up-in-ai-infra-bury-the-compute-under-the-dram/5264071
- https://www.theregister.com/systems/2026/06/24/qualcomm-claims-its-not-too-late-for-dragonfly-to-land-in-datacenters/5261758
- https://futurumgroup.com/insights/qualcomms-data-center-reentry-at-investor-day-2026-arrives-just-in-time-for-the-inference-decode-prize/
- https://hotchips.org/program/conference/
- https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/
