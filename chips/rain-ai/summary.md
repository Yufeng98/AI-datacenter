# Rain AI — Research Summary

*as_of: 2026-08-08*
*device_class: Neuromorphic AI*
*pipeline_stage: complete (historical entry — vendor defunct)*

---

## STATUS: DEFUNCT — non-operating; patent portfolio sold to OpenAI (USPTO-recorded 2025-10-23)

| Field | Value |
|-------|-------|
| Company status | **Defunct / non-operating shell.** Rain Neuromorphics Inc. (dba Rain AI) never shipped a commercial product. |
| Status-change date | **2025-10-23** — USPTO recordation date of the patent assignments out of Rain Neuromorphics Inc. This is the earliest primary-record evidence of the wind-down; the operational collapse began earlier (Series B failure, Q1 2025). |
| Public disclosure date | **~2026-08-06** — The Information reported the OpenAI patent purchase, ~9.5 months after the USPTO recordation. The press date is the *disclosure* date, not the transaction date. |
| What happened to the IP | Patents were assigned from **Rain Neuromorphics Inc. → OpenAI Opco, LLC**, recorded at USPTO on **2025-10-23**. Verified on three separate filings spanning both technology eras: US20240281497A1 ("SRAM Matrix Multiplication Network"), US20240143541A1 ("Compute in-memory architecture for continuous on-chip learning"), US20250045224A1 ("Tiled in-memory computing architecture") — a portfolio-level transfer, not a one-off. Number of patents transferred and financial terms: **not disclosed**. Per The Information, the patent purchase followed the collapse of talks for OpenAI to acquire the whole company. |
| What happened to the SDK | The Rain compiler, runtime, and D-IMC op/kernel libraries were **never publicly released and never open-sourced**. No SDK, no developer documentation, and no public source repository ever existed, and none survives. The disposition of the software stack in the wind-down is **not disclosed** — only the patent assignments are documented in primary records. |
| Product reality | No commercial shipment ever occurred. Evaluation chips delivered to select partners in **October 2024** were the program's high-water mark. |
| Web presence | rain.ai still resolves (HTTP 200) and still serves the old marketing copy, but it is dormant: last blog post **2024-06-27**, footer copyright 2024, no shutdown notice. **Site liveness is not evidence of operations.** |
| Formal dissolution | **Not confirmable** from free public records. Treat the entity as a non-operating shell whose core asset has been sold. |

**Primary sources:** USPTO assignment records via
[US20240281497A1](https://patents.google.com/patent/US20240281497A1/en),
[US20240143541A1](https://patents.google.com/patent/US20240143541A1/en),
[US20250045224A1](https://patents.google.com/patent/US20250045224A1/en)
— all three showing Rain Neuromorphics Inc. → OpenAI Opco, LLC, recorded 2025-10-23.
Secondary/aggregator disclosure: [valueaddvc.com, 2026-08-06](https://valueaddvc.com/pulse/openai-rain-ai-patents-altman-backed-chip-2026),
attributing to The Information. Dormancy evidence: [rain.ai/blog](https://rain.ai/blog) (last post 2024-06-27).

**Why this entry is retained:** the D-IMC architecture and the analog→digital pivot remain a valid data point
for this survey, and the *reason* Rain died — a compute-in-memory program that reached evaluation silicon but
never converted a marquee customer LOI into revenue before its funding round failed — is itself analytically
interesting. All architecture documentation below is preserved, rewritten in the past tense.

---

## TL;DR

Rain AI (formerly Rain Neuromorphics) was a Silicon Valley startup that pivoted from an **analog neuromorphic
chip** (spiking neurons + ReRAM memristors, 2017–2023) to a **Digital In-Memory Computing (D-IMC)** accelerator
architecture (2024–2025). The company was notable primarily for its associations — Sam Altman personal
investment, a $51M OpenAI letter of intent, and a CFIUS-forced Saudi Aramco divestiture — rather than for
disclosed technical specifications. Almost nothing about the D-IMC chip's process node, TOPS, memory, or
software stack was ever made public. The $150M Series B collapsed in Q1 2025, the company exhausted its funding
and shed effectively all staff, and its patents were sold to OpenAI Opco, LLC (USPTO-recorded 2025-10-23).

---

## Architecture Overview

### Era 1: Analog Neuromorphic NPU (2017–2023, not commercialized)

Rain's original mission was to build a brain-inspired analog AI chip using:
- **Memristors (ReRAM)** as analog synaptic weights — memory and compute co-located
- **Spiking neural network circuits** — 10M+ claimed neurons/cm²
- **Equilibrium Propagation** — analog-compatible training algorithm (mathematically equivalent to backpropagation)
- **3D vertical bit-line integration** (borrowed from NAND flash) for high-density ReRAM stacking

A demo chip was taped out in 2021 on 180nm CMOS with 10,000 neurons. This chip demonstrated on-chip AI training.
Rain claimed 1,000× energy efficiency over GPU-class hardware — a marketing claim, never independently verified.

The analog approach was abandoned due to technical challenges (process variation, yield, analog noise, training
stability at scale).

### Era 2: Digital In-Memory Computing (D-IMC) — final architecture (2024–2025)

Rain's final architecture (announced ~2024) comprised:
- **D-IMC Compute Tiles**: Digital MAC operations inside SRAM arrays; eliminated data movement to separate ALUs; claimed production-scalable (unlike analog CIM)
- **Andes AX45MPV RISC-V Host Core**: Licensed June 2024; RV64GCV (OoO + RVV 1.0 vector); extended with custom D-IMC dispatch instructions via Andes ACE/COPILOT
- **Arteris FlexNoC 5 NoC**: Licensed January 2024; advanced mesh topology connecting RISC-V cores ↔ D-IMC tiles ↔ memory
- **Proprietary Compiler + Runtime**: Full-stack, co-designed; claimed to support training and inference; never released

Evaluation chips were delivered to select partners in October 2024. **Commercial shipment never occurred.**

---

## Key Facts

| Attribute | Value |
|-----------|-------|
| Device class | Neuromorphic AI |
| Final architecture | Digital In-Memory Computing (D-IMC) + RISC-V |
| Process node | Not disclosed |
| Peak TOPS | Not disclosed |
| Off-chip memory | Not disclosed |
| Host ISA | RISC-V RV64GCV (Andes AX45MPV) |
| On-chip NoC | Arteris FlexNoC 5 (mesh) |
| EDA toolchain | Synopsys Cloud |
| SDK / compiler | Proprietary; never released, never open-sourced |
| Open-source components | None (RISC-V base spec only) |
| Total funding | ~$146M |
| Commercial product | None — evaluation chips only (Oct 2024) |
| Company status | **Defunct / non-operating (see Status block above); patents → OpenAI Opco, LLC, USPTO-recorded 2025-10-23** |

---

## Corporate Timeline

| Event | Date | Detail |
|-------|------|--------|
| Founded | 2017 | As Rain Neuromorphics; YC W19 |
| Analog demo chip tapeout | 2021 | 180nm, 10K neurons, ReRAM |
| Series A | 2022 | $25M |
| OpenAI LOI | 2019 (revealed 2023) | $51M non-binding LOI; CFIUS forced Saudi Aramco (Prosperity7) divestiture |
| CEO transition | late 2023 | Founder Gordon Wilson stepped down as CEO to Executive Advisor; succeeded by co-founder Jack Kendall |
| D-IMC pivot + Arteris NoC | Jan 2024 | FlexNoC 5 selected |
| D-IMC pivot + Andes RISC-V | Jun 2024 | AX45MPV licensed |
| Last website activity | 2024-06-27 | Final blog post; site dormant thereafter |
| First (and only) chip samples | Oct 2024 | Evaluation only, select partners; never commercial |
| Series B collapse | Q1 2025 | $150M round failed; company began exploring a sale |
| Bridge round | May 2025 | $3M bridge |
| Hardware lead departure | by Jun 2025 | Jean-Didier Allegrucci left for Intel |
| **Patents assigned to OpenAI Opco, LLC** | **2025-10-23 (USPTO recordation)** | Portfolio-level transfer; count and terms not disclosed |
| Public disclosure of patent sale | ~2026-08-06 | The Information; full-company acquisition talks had already collapsed |
| Total raised | — | ~$146M |

---

## Competitive Position (retrospective)

Rain's D-IMC approach placed it against:
- **d-matrix (Corsair)**: Also digital in-SRAM compute (2,048 DIMC cores, 64×64 MAC in 6nm TSMC); far more disclosed specs
- **Samsung AquaBolt PIM / SK Hynix AiMX**: PIM in DRAM (HBM), different tier
- **Mythic**: Analog in-SRAM compute (adjacent in the analog-to-digital CIM space)

Rain's claimed differentiators (company claims, never independently verified):
1. D-IMC scalable to high-volume production (vs. analog yield challenges)
2. Training capability (rare for in-memory compute)
3. RISC-V programmability for operator flexibility

With no public benchmark, no SDK, no commercial product, and the company now defunct, Rain cannot be evaluated
competitively. It is retained here as an architectural data point and as a case study in CIM-startup failure
modes, not as a live option.

---

## Information Gaps

The following critical details were **never disclosed** and, with the company non-operating, are unlikely to be:
- Process node and foundry for the D-IMC chip
- Peak performance (TOPS/TFLOPS at any precision)
- Off-chip memory type, capacity, and bandwidth
- Power envelope (TDP)
- Die size and transistor count
- Software stack details (compiler IR, runtime API, driver interface)
- Multi-chip scaling architecture
- Host interface (PCIe/CXL/other)
- Number of patents transferred to OpenAI, and the financial terms of the transfer
- Whether the software stack (compiler/runtime source) transferred with the patents

---

## Deliverables

| File | Description |
|------|-------------|
| `research/rain-ai/search-results.md` | Categorized resource catalog (15 layers) |
| `research/rain-ai/summary.md` | Mirror of this file |
| `chips/rain-ai/hw-architecture.md` | Detailed hardware architecture narrative |
| `chips/rain-ai/hw-architecture.yaml` | Machine-readable hardware spec |
| `chips/rain-ai/software-stack.md` | Software stack narrative |
| `chips/rain-ai/software-stack.yaml` | Machine-readable software stack |
| `chips/rain-ai/hw-architecture.dot` / `.png` | Graphviz architecture diagram |
| `chips/rain-ai/programming-model.dot` / `.png` | Graphviz programming-model diagram |
| `chips/rain-ai/layer-table.md` | Layer-by-layer summary table |
