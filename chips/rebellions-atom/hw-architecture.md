# Rebellions ATOM / REBEL Hardware Architecture

*chip: rebellions-atom*
*device_class: Inference Accelerator (Korea)*
*as_of: 2026-08-08*

---

## Overview

Rebellions' hardware architecture has evolved across two generations: **ATOM** (Samsung 5nm CGRA, 2022) and **REBEL-Quad** (Samsung 4nm UCIe chiplet, 2025). Both share a compile-time-scheduled, software-managed memory model with no hardware caches. The REBEL-Quad (Rebel100) is the industry's first AI accelerator to adopt UCIe-Advanced for chiplet interconnect.

**No third silicon generation exists as of 2026-08-08.** Every die-level and package-level figure in this document traces to Hot Chips 2025 and ISSCC 2026 and is unchanged. The 2026 additions below are at the **card and system level** — RebelCard, RebelServer, and an Arm AGI CPU host pairing — none of which has published specifications. See §9.

### Generation / Product Overview

| Generation | Product | Process | Compute | Memory | Form factor | Status (2026-08-08) |
|---|---|---|---|---|---|---|
| Gen 1 | ATOM / ATOM-Max | Samsung 5nm | 8 Neural Engines (CGRA) | 16 GB GDDR6, 256 GB/s | PCIe Gen5 FHFL, 60–130 W | Production |
| Gen 2 | REBEL Single | Samsung 4nm | 16 Neural Cores | HBM (per die) | PCIe card | Production |
| Gen 2 Chiplet | REBEL-Quad (Rebel100) | Samsung 4nm | 4 × 320 mm² dies, 2,048 TFLOPS FP8 | 144 GB HBM3e, 4.8 TB/s | PCIe ~600 W package | Production / shipping since 2025 |
| Gen 2 Card | **RebelCard** | Samsung 4nm (Rebel100) | Rebel100 — "four NPU chiplets" | "5th-generation HBM" (HBM3E); capacity not disclosed | Module-type accelerator card, air-cooled | **Announced 2026-04-10; in validation at SK Telecom — not shipping, no confirmed ship date** |
| Gen 2 Rack | RebelRack ⚠️ / RebelPOD ⚠️ | — | 64 PFLOPS FP8 (RebelRack) | 4.6 TB HBM3e (RebelRack) | 4 nodes (RebelRack) / 8–128 nodes (RebelPOD) | Available 2026; **specs secondary-sourced only** |
| Gen 2 Server | **RebelServer** | — | not disclosed | not disclosed | Single-server chassis | **Announced 2026-07-23; no specs disclosed** |
| Future | REBEL-IO, REBEL-CPU | Samsung 4nm | Trillion-param MoE target | — | — | Roadmap only — no tape-out, sampling, or spec disclosure |

---

## 1. Compute Engine

### ATOM: CGRA + Neural Engines

ATOM uses a **Coarse-Grained Reconfigurable Array (CGRA)** architecture. Processing Element (PE) tiles are reprogrammed by the RBLN compiler to execute different AI operator types — GEMM, convolution, normalization, and activation — without requiring custom hardware units for each.

| Component | Description |
|---|---|
| Architecture | CGRA (Coarse-Grained Reconfigurable Array) |
| Neural Engines | 8 |
| Command Processor | Schedules Neural Engines; fetches programs |
| PE Tiles | Reconfigurable; mapped by RBLN compiler |
| ISA | Custom RISC ISA (internal to Compute Library) |
| Peak FP16 | 32 TFLOPS |
| Peak INT8 | 128 TOPS |
| Process | Samsung 5nm |

Each Neural Engine contains tensor units, vector units, scalar units, and input buffers (IBUFs) with a custom instruction set for fine-grained operand management.

### REBEL Single: Neural Core Architecture

The REBEL generation replaces the CGRA macro-tile with a structured **Neural Core** design:

| Component | Specification |
|---|---|
| Neural Cores | 16 total |
| Cores per Cluster | 8 |
| Cluster Interconnect | Mesh (through SRAM blocks) |
| Core Components | IBUF (custom ISA), tensor unit, vector unit, scalar unit, LD/ST |
| RISC ISA | Programmable via Compute Library |
| Process | Samsung 4nm |
| Target Workloads | Frontier LLMs, MoE, peta-scale inference |

### REBEL-Quad: 4-Die Chiplet

| Component | Specification |
|---|---|
| Compute dies | 4 × 320 mm² REBEL Single ASIC |
| Die interconnect | UCIe-Advanced, 16 Gbps, 4 TB/s aggregate |
| HBM3e stacks | 4 × 36 GB (12Hi) = 144 GB total |
| ISC capacitors | 4 (per-die power supply) |
| Peak FP8 | 2,048 TFLOPS (2.048 PFLOPS) |
| Package TDP | ~600 W |
| Process | Samsung 4nm |

*Added 2026-08-08:* the 2026-04-10 SK Telecom / Arm release describes the Rebel100 carried on RebelCard as "four NPU chiplets" with "5th-generation HBM" — consistent with the REBEL-Quad package above and **not** a new compute engine. No revised compute figure was published.

---

## 2. Data Path

### Compile-Time Scheduled Dataflow

ATOM and REBEL use a **static dataflow model**: the RBLN compiler performs all memory scheduling, tiling, and dependency analysis at compile time. There is no runtime scheduling of individual operations.

```
Host (PCIe) → Command Processor → Neural Engines (CGRA / Neural Cores)
                                        ↕
                               On-chip SRAM (SW-managed)
                                        ↕
                            Off-chip Memory (GDDR6 / HBM3e)
```

Execution flow:
1. Host sends compiled binary and input tensors via PCIe Gen5 DMA
2. Command Processor fetches instructions and dispatches to Neural Engines
3. Neural Engines operate on tiled data from SRAM
4. SRAM ↔ DRAM transfers are pre-scheduled by compiler (no cache misses, no page faults)
5. Outputs returned to host via PCIe DMA

### REBEL-Quad Cross-Die Data Path

In REBEL-Quad, the compiler also schedules cross-die data movement via UCIe-Advanced links:
- Each ASIC die is independent; cross-die communication is explicit
- 4 TB/s UCIe aggregate bandwidth enables near-HBM-speed die-to-die transfers
- The mesh topology of Neural Core clusters extends across die boundaries via UCIe

---

## 3. On-chip Memory

All on-chip memory is **software-managed scratchpad** — no hardware caches anywhere in the design. The RBLN compiler's dependency analysis and memory allocator determine all SRAM usage at compile time.

| Memory | ATOM | REBEL Single | REBEL-Quad (per die) |
|---|---|---|---|
| L1 SRAM | 64 MB (shared) | 64 MB (per-core distributed) | 64 MB (per-core distributed) |
| L2 SRAM | — | 64 MB (shared) | 64 MB (shared) |
| Total per chip | 64 MB | 128 MB | 128 MB |
| Total for REBEL-Quad | — | — | 4 × 128 MB = 512 MB |
| Management | Compiler-managed | Compiler-managed | Compiler-managed |
| Cache hardware | None | None | None |

---

## 4. Off-chip Memory

| Spec | ATOM | REBEL-Quad |
|---|---|---|
| Type | GDDR6 | HBM3e (12Hi) |
| Capacity | 16 GB | 144 GB (4 × 36 GB) |
| Bandwidth | 256 GB/s | 4.8 TB/s aggregate |
| Per-die capacity | 16 GB | 36 GB |
| Per-die bandwidth | 256 GB/s | 1.2 TB/s |
| Memory management | SW-managed (compiler) | SW-managed (compiler) |

The transition from GDDR6 to HBM3e provides ~18.75× bandwidth and 9× capacity increase, which is necessary for serving frontier LLMs (LLaMA 70B, Mixtral 8×7B) without model sharding.

---

## 5. Host Interface / Package

### ATOM — RBLN-CA12 Card

| Spec | Value |
|---|---|
| Form factor | FHFL (Full Height Full Length), single PCIe slot |
| Host interface | PCIe Gen5 x16 (~128 GB/s bidir) |
| Card-to-card | PCIe Gen5 x16 |
| TDP | 60–130 W (configurable) |
| Multi-instance | 16 hardware-isolated instances |
| Physical size | Standard FHFL PCIe add-in card |

### REBEL-Quad — Rebel100 Package

| Spec | Value |
|---|---|
| Form factor | PCIe card (not OAM/SXM) |
| Package TDP | ~600 W |
| Package contents | 4 ASIC dies + 4 HBM3e stacks + 4 ISC |
| Die area (each) | 320 mm² |
| Reference design | 8 cards per air-cooled 1U/2U node |
| OCP listing | REBEL-Quad AI Accelerator / AI SoC |

### RebelCard — Rebel100 module card (announced 2026-04-10)

| Spec | Value |
|---|---|
| Form factor | Module-type accelerator card (not further specified) |
| Silicon | Rebel100 — four NPU chiplets |
| Memory | "5th-generation HBM" (HBM3E); capacity and bandwidth **not disclosed** |
| Cooling | Air-cooled (vendor statement) |
| Host CPU pairing | **Arm AGI CPU** — described in the primary release as built on **Arm Neoverse CSS V3** and as "the first Arm-designed data center CPU" |
| Power / TDP | not disclosed |
| Performance | not disclosed — vendor claims only "comparable to current flagship GPUs" with better power efficiency |
| Status | MOU-stage co-development; validation planned in SK Telecom's AI datacenter. Not sampling to customers, not shipping. **No ship date is confirmed by any primary source** (aggregator "Q3 2026" is uncorroborated) |

The Arm AGI CPU pairing is the first publicly named host CPU for a Rebellions accelerator platform; prior material describes only a generic x86/PCIe host. Arm's own AGI disclosure is *scheduled for Hot Chips 38, Aug 2026 — content not yet public.*

---

## 6. Scale-up Interconnect

### ATOM

No native chip-to-chip scale-up interconnect. Multi-card configurations use PCIe card-to-card links.

### REBEL-Quad Intra-Package (UCIe-Advanced)

| Spec | Value |
|---|---|
| Standard | UCIe-Advanced |
| Speed per link | 16 Gbps |
| Aggregate bandwidth | 4 TB/s |
| Topology | Full mesh across 4 ASIC dies |
| Industry milestone | World's first AI accelerator with UCIe-Advanced (Hot Chips 2025) |
| OCP registry | Listed as open chiplet standard implementation |

### System-Level Scale-Up

| System | Accelerators | Peak Compute | Memory | Networking | Sourcing |
|---|---|---|---|---|---|
| RebelRack | 32 (4 nodes × 8) | 64 PFLOPS FP8 | 4.6 TB HBM3e | quad-400 GbE/node | ⚠️ **secondary only** (The Register 2026-03-30) |
| RebelPOD | 64–1,024 (8–128 nodes × 8) | — | — | 800 GbE | ⚠️ **secondary only** |
| RebelServer | not disclosed | not disclosed | not disclosed | not disclosed | Vendor release 2026-07-23 (no specs) |

> ⚠️ **Sourcing caveat (added 2026-08-08).** Rebellions' own 2026-03-30 release announcing RebelRack and RebelPOD published **no technical specifications**. The 32-accelerator / 64 PFLOPS / 4.6 TB / 153.6 TB/s / quad-400GbE figures derive from secondary coverage and remain uncorroborated by any Rebellions primary source as of 2026-08-08.

**RebelServer** (2026-07-23) is a single-server product built on Rebel100. Rebellions reports running SK Telecom's **A.X K1** sovereign LLM — over 500B parameters, **Mixture-of-Experts** — on a single RebelServer, serving concurrent real-time requests. No device count, latency, throughput, TPS/W, or batch-size figure was published, so the run cannot be converted into a memory-capacity or throughput claim.

---

## 7. Scale-out Interconnect

- Ethernet-based (400 GbE / 800 GbE); no InfiniBand
- RebelRack: quad-400 Gbps (4 × 400 GbE) per node for intra-rack all-reduce
- RebelPOD: 800 Gbps Ethernet for multi-node scale-out
- Key telecom partnerships: SK Telecom (Korea), DOCOMO Innovations (Japan) — aimed at telco AI inference deployments

---

## 8. Architecture Comparison Summary

| Layer | ATOM | REBEL-Quad |
|---|---|---|
| Compute model | CGRA (8 Neural Engines) | Neural Cores (16 per die × 4 dies) |
| Process | Samsung 5nm | Samsung 4nm |
| On-chip memory | 64 MB flat SRAM | 128 MB 2-tier SRAM × 4 = 512 MB |
| Off-chip memory | 16 GB GDDR6, 256 GB/s | 144 GB HBM3e, 4.8 TB/s |
| Host interface | PCIe Gen5 x16 | PCIe ~600 W |
| Chiplet interconnect | None (monolithic) | UCIe-Advanced 4 TB/s |
| Peak compute | 32 TFLOPS FP16 / 128 TOPS INT8 | 2,048 TFLOPS FP8 |
| Memory model | SW-managed (no caches) | SW-managed (no caches) |
| TDP | 60–130 W | ~600 W |
| Multi-instance | 16 hardware | not disclosed |

---

## 9. Update — 2026-08-08 (window 2026-04-06 → 2026-08-08)

**Silicon: unchanged.** No REBEL Gen-3; no tape-out, sampling, or spec disclosure for REBEL-IO / REBEL-CPU; no cancellation or delay signal for ATOM, ATOM-Max, or REBEL-Quad/Rebel100. No revised die area, clock, SRAM, HBM, UCIe, or TFLOPS figure appeared in the window — §§1–8 above stand as written against Hot Chips 2025 and ISSCC 2026.

**What changed is above the die:**

| Change | Date | Layer affected | Specs disclosed |
|---|---|---|---|
| RebelCard — Rebel100 module card | 2026-04-10 | §5 Host Interface / Package | None beyond "four NPU chiplets, 5th-generation HBM, air-cooled" |
| Arm AGI CPU host pairing (Neoverse CSS V3) | 2026-04-10 | §5 Host Interface | Host CPU named; no platform specs |
| Giga Computing MOU (servers + rack-scale around RebelCard/Rebel100) | 2026-06-17 | §6 Scale-up (system) | None — no timeline, no SKU |
| RebelServer + A.X K1 500B MoE single-server run | 2026-07-23 | §6 Scale-up (system) | None — no device count, no throughput |
| RebelRack/RebelPOD specs re-graded to secondary-sourced | 2026-08-08 | §6 Scale-up (system) | n/a — sourcing correction |

**Status-verb discipline.** Both the Arm/SK Telecom and Giga Computing items are **MOUs** — intent to co-develop. Neither is a design win or a shipping product. RebelCard is announced and entering validation; it is not sampling to customers and not shipping.

**Benchmark absence.** Rebellions did not submit to MLPerf Inference v6.0 (2026-04-01) or MLPerf Training v6.0 (2026-06-16). It has no talk on the Hot Chips 38 program (Aug 23–25 2026 — still in the future; nothing there is evidence of anything).

**Unverified.** A UPI item dated 2026-08-07 ("SK Telecom, Rebellions expand Korean AI chip infrastructure") appeared in search but returned HTTP 403; developments in the final week of this window are unverified.

---

## Sources

- [ATOM Architecture: Finding the Sweet Spot for GenAI](https://rebellions.ai/atom-architecture-finding-the-sweet-spot-for-genai/)
- [ATOM White Paper (PDF)](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf)
- [ATOM-Max White Paper](https://rebellions.ai/white-paper-atom-max/)
- [REBEL-Quad Hot Chips 2025](https://rebellions.ai/newsroom/rebellions-debuts-rebel-quad-at-hot-chips-2025-breaking-ais-energy-tax-with-high-performance-chiplet-innovation/)
- [ServeTheHome: REBEL-Quad UCIe and 144GB HBM3E](https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144tb-hbm3e-accelerator-at-hot-chips-2025/)
- [Tom's Hardware: ISSCC 2026 Rebel100](https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100)
- [Next Platform: REBEL Architecture Deep Dive](https://www.nextplatform.com/2025/12/23/rebellions-ai-puts-together-an-hbm-and-arm-alliance-to-take-on-nvidia/)
- [EE Times: Chiplet Roadmap](https://www.eetimes.com/rebellions-builds-chiplet-roadmap-merges-with-sapeon/)
- [The Register: RebelRack](https://www.theregister.com/2026/03/30/rebellions_ai_rackscale/)
- [OCP REBEL-Quad](https://www.opencompute.org/chiplets/76/rebel-quad-ai-accelerator-ai-soc)
- [Synopsys: Rebellions Design Partnership](https://www.synopsys.com/blogs/chip-design/energy-efficient-ai-accelerator-data-centers.html)

### Added 2026-08-08

- [Rebellions + SK Telecom + Arm MOU — RebelCard, Arm AGI CPU (2026-04-10)](https://rebellions.ai/newsroom/rebellions-collaborates-with-sk-telecom-and-arm-targeting-sovereign-ai-and-telecom-infrastructure/)
- [Rebellions + Giga Computing MOU (2026-06-17)](https://rebellions.ai/newsroom/rebellions-and-giga-computing-sign-mou-to-develop-next-generation-ai-server-and-rack-scale-solutions/)
- [RebelServer runs SKT A.X K1, 500B+ MoE (2026-07-23)](https://rebellions.ai/newsroom/rebelserver_run_k1/)
- [Seoul Economic Daily EN: A.X K1 on RebelServer (2026-07-23)](https://en.sedaily.com/technology/2026/07/23/rebellions-runs-skts-sovereign-model-ax-k1-on-npu-server)
- [Rebellions $400M pre-IPO + RebelRack/RebelPOD launch (2026-03-30)](https://rebellions.ai/newsroom/rebellions-closes-400-million-pre-ipo-and-launches-rebelrack-and-rebelpod-to-accelerate-global-expansion/) — primary launch release; contains **no** technical specifications
- [MLPerf Training v6.0 results (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — Rebellions absent
- [Hot Chips 38 program](https://hotchips.org/) — Aug 23–25 2026; Rebellions absent; Arm AGI talk scheduled, content not yet public
