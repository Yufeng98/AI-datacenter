# Rain AI Hardware Architecture

*as_of: 2026-08-08*
*Architecture: Digital In-Memory Computing (D-IMC) AI Accelerator (post-2023 pivot from analog neuromorphic)*
*Vendor status: **DEFUNCT** — non-operating; patents assigned to OpenAI Opco, LLC, USPTO-recorded 2025-10-23. See `summary.md` for the full status block. This document is a historical architecture record.*

---

## Overview

> **Status notice (2026-08-08).** Rain Neuromorphics Inc. (dba Rain AI) is defunct. It never shipped a commercial
> product; evaluation chips delivered to select partners in October 2024 were the high-water mark. Its $150M
> Series B collapsed in Q1 2025 and its patent portfolio was assigned to OpenAI Opco, LLC with a USPTO
> recordation date of 2025-10-23 (verified on US20240281497A1, US20240143541A1, US20250045224A1). Everything
> below describes a program that no longer exists and is written in the past tense.

Rain AI (incorporated as Rain Neuromorphics) underwent a major architectural pivot. The company began (2017–2023) developing an **analog neuromorphic** chip using spiking neurons, ReRAM memristors, and the Equilibrium Propagation training algorithm. By 2024 it had pivoted to a **Digital In-Memory Computing (D-IMC)** architecture paired with RISC-V scalar/vector host cores and an Arteris FlexNoC mesh interconnect.

The D-IMC architecture was Rain's final commercial direction. The analog NPU work (180nm tapeout, 2021) was a research predecessor and was never commercialized. Neither was the D-IMC design: the company wound down after the Series B failure, and its patents were sold.

**Critical note:** Rain AI disclosed very little technical detail about its D-IMC chip. Specifications such as process node, die size, memory capacity, peak TOPS, and power envelope were never disclosed, and with the company non-operating they are unlikely to be. The architecture description below is reconstructed from partnership press releases and Rain's own (now dormant) marketing text.

---

## 1. Compute Engine: Digital In-Memory Computing (D-IMC) Tiles

### What is D-IMC?

Digital In-Memory Computing performs multiply-accumulate (MAC) operations **inside SRAM arrays** rather than in separate ALU units. This eliminates the data-movement bottleneck between memory and compute — the dominant source of energy waste in conventional AI accelerators.

Rain's D-IMC tiles were described as:
- **Digital** (not analog): unlike resistive/analog CIM, Rain's tiles performed digital arithmetic, making them compatible with standard digital calibration and design flows
- **Scalable to high-volume production**: Rain claimed its D-IMC design was production-manufacturable (distinguishing it from analog CIM, which struggles with process variation at scale) — a marketing claim never tested, since the design never reached volume production
- **Supporting both training and inference**: Rain explicitly claimed training capability, which is unusual for in-memory compute designs; never independently verified

### Tile Architecture (inferred)

The D-IMC tile likely followed a standard digital CIM pattern:
- SRAM arrays with integrated MAC circuitry at the bitcell or subarray level
- Partial-sum accumulation within the array, reducing off-array data movement
- Output registers for inter-tile communication

Exact tile dimensions (rows × columns), bitwidth, and TOPS/tile were **not disclosed**.

---

## 2. Host Control: Andes AX45MPV RISC-V Vector Cores

In June 2024, Rain licensed **Andes Technology's AX45MPV** RISC-V processor as the host/control CPU within the SoC:

- **ISA**: RV64GCV (64-bit, RISC-V Vector extension V)
- **AX45MPV features**: Out-of-order pipeline, RVV 1.0 vector extension, multi-core capable
- **Role in Rain's SoC**: Host scalar workloads, orchestrate D-IMC tile dispatch, run OS/runtime
- **Customization**: Rain used Andes' **ACE/COPILOT** framework to add custom instructions that interfaced RISC-V cores with D-IMC tiles

The proprietary interconnect between RISC-V cores and D-IMC tiles was described as a key architectural differentiator — Rain claimed a "balanced pipeline" design that avoided bottlenecking at the RISC-V/D-IMC boundary. No block diagram or measurement was ever published to support the claim.

---

## 3. On-Chip Interconnect: Arteris FlexNoC 5

In January 2024, Rain selected **Arteris FlexNoC 5** as its on-chip network-on-chip (NoC) IP:

- **Topology**: Advanced mesh network (Rain's description)
- **IP**: Arteris FlexNoC 5 Physically Aware — supports heterogeneous SoC integration with timing-closure-aware placement
- **Purpose**: Connects multiple RISC-V cores, D-IMC tiles, memory controllers, and I/O blocks

---

## 4. On-Chip Memory

- D-IMC tiles used SRAM as both storage and compute medium — capacity not disclosed
- No separate L1/L2 cache hierarchy was ever described
- Whether a dedicated weight buffer or activation scratchpad existed beyond the D-IMC arrays was never documented

---

## 5. Off-Chip Memory

- Not disclosed. Type (LPDDR5, HBM, GDDR6), capacity, and bandwidth were never disclosed.

---

## 6. Process Technology

- **D-IMC chip tapeout**: Process node not disclosed. Rain used Synopsys Cloud EDA for the tapeout; Synopsys' case study reported first-pass silicon success (vendor case study, not independently verified).
- **Analog NPU demo chip (2021, superseded)**: 180nm CMOS, 10,000 neurons, ReRAM memristors — research prototype only.

---

## 7. Package & Host Interface

- Not disclosed. PCIe, CXL, or custom host interface was never documented.

---

## 8. Legacy Analog Architecture (2017–2023, not commercialized)

For historical completeness, Rain's original analog neuromorphic design included:

| Component | Description |
|-----------|-------------|
| Neuron circuit | CMOS spiking neuron; claimed 10M+ spiking neurons/cm² |
| Synapse | ReRAM (Resistive RAM) memristor — weight stored as resistance state |
| 3D integration | Vertical bit-line NAND-flash technique for 3D memristor stacking |
| Connectivity | Sparse synaptic connectivity replicating biological sparsity |
| Training algorithm | Equilibrium Propagation — analog-compatible alternative to backpropagation |
| Demo chip (2021) | 180nm CMOS, 10,000 neurons, taped out via Synopsys Cloud |

Rain claimed 1,000× energy efficiency improvement over GPU-class hardware for this design. The claim was not independently verified. Technical challenges (analog variation, yield, training stability) led to the D-IMC pivot.

---

## 9. Performance Claims

All performance figures below are **vendor marketing claims**. None was independently confirmed, and no
benchmark, MLPerf submission, or third-party measurement of Rain silicon was ever published.

| Claim | Source | Verification |
|-------|--------|-------------|
| 1,000× more energy efficient than GPU (analog era) | Rain marketing | Never independently verified |
| 100× more compute for AI training vs GPU | OpenAI LOI coverage | Never independently verified |
| D-IMC training + inference capable | Rain approach page (now dormant) | Never independently verified |
| First chips delivered Oct 2024 (evaluation only) | Press reports | Limited, to select partners; never commercial |

---

## 10. Company Status (as of 2026-08-08) — DEFUNCT

**Rain Neuromorphics Inc. (dba Rain AI) is non-operating.** It never shipped a commercial product. Its patent
portfolio was assigned to **OpenAI Opco, LLC**, USPTO recordation date **2025-10-23**, verified on three
filings spanning both technology eras (US20240281497A1, US20240143541A1, US20250045224A1); patent count and
terms were not disclosed. The deal became public only ~2026-08-06, when The Information reported it — the press
date is the disclosure date, not the transaction date. The proprietary compiler and runtime were never
released or open-sourced and did not survive; their disposition in the wind-down is not disclosed. rain.ai
still resolves but is dormant (last blog post 2024-06-27, footer copyright 2024) and is not evidence of
operations. Formal corporate dissolution is not confirmable from free public records.

| Event | Date | Detail |
|-------|------|--------|
| Founded | 2017 | As Rain Neuromorphics; YC W19 |
| Analog demo chip tapeout | 2021 | 180nm, 10K neurons, ReRAM |
| Series A | 2022 | $25M |
| OpenAI LOI | 2019 (revealed 2023) | $51M non-binding LOI; CFIUS forced Saudi Aramco (Prosperity7) divestiture |
| CEO transition | late 2023 | Gordon Wilson → Executive Advisor; co-founder Jack Kendall became CEO |
| D-IMC pivot + Arteris NoC | Jan 2024 | FlexNoC 5 selected |
| D-IMC pivot + Andes RISC-V | Jun 2024 | AX45MPV licensed |
| Last website activity | 2024-06-27 | Final blog post; site dormant thereafter |
| First (and only) chip samples | Oct 2024 | Evaluation only, not commercial scale |
| Series B collapse | Q1 2025 | $150M round failed; company began exploring a sale |
| Bridge round | May 2025 | $3M bridge |
| Hardware lead departure | by Jun 2025 | Jean-Didier Allegrucci left for Intel |
| **Patents → OpenAI Opco, LLC** | **2025-10-23** | USPTO recordation; portfolio-level transfer, terms not disclosed |
| Public disclosure of patent sale | ~2026-08-06 | The Information, after whole-company acquisition talks collapsed |
| Total raised | — | ~$146M |
