# SpiNNcloud SpiNNaker2 — Summary

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*
*Representative products: SpiNNaker2 chip · SpiNNode single-chip board · 48-node board · NERL Braunfels (Sandia) · SpiNNcloud Dresden system*

---

## One-Line Summary

SpiNNaker2 is a 22 nm neuromorphic manycore from **SpiNNcloud Systems GmbH** (TU Dresden spinout,
commercializing Steve Furber's SpiNNaker lineage): **152 ARM Cortex-M4F cores** each with a small **16×4 MAC
array**, **19.8 MB of entirely software-managed SRAM**, **no cache, no OS, no tensor compiler and no kernel
driver**, delivering **4.563 TOPS INT8 measured at 2.06–2.77 TOPS/W** inside a **0.24–2.2 W** chip power
envelope.

---

## Key Facts

| Property | Value |
|----------|-------|
| Company | SpiNNcloud Systems GmbH, Dresden, Germany (TU Dresden spinout) |
| Lineage | SpiNNaker1 (Steve Furber, Univ. of Manchester) → SpiNNaker2 (TU Dresden Höppner/Mayr group) |
| Maturity | **Shipping, low volume.** Commercial availability announced May 2024 at ISC High Performance |
| Documented deployments | **Sandia NNSA "NERL Braunfels"** (175 M neurons, delivered March 2025) · **TU Dresden/SpiNNcloud** (~35,000 chips / 5 M cores / eight racks) |
| Process | GlobalFoundries **22FDX (22 nm FDSOI)** with Racyics ABX adaptive body biasing |
| Die area | **102 mm²** (266 M gate equivalents) |
| Cores | **152 ARM Cortex-M4F PEs** (38 QuadPEs × 4) + 1 periphery Cortex-M4 management core = the vendor's "153 ARM cores"; **148 usable by applications** |
| Peak INT8 | **5.837 TOPS theoretical** · **4.563 TOPS measured** @ 300 MHz/0.8 V |
| Energy efficiency | **2.77 TOPS/W** @ 150 MHz/0.5 V · **2.06 TOPS/W** @ 300 MHz/0.8 V (on-chip operands) |
| On-chip SRAM | **19.8 MB** (128 kB per PE), **entirely software-managed — no hardware data cache** |
| Off-chip memory | **2 GB LPDDR4** per chip (2 interfaces), **6.4 GB/s aggregate raw**; sustained BW not disclosed |
| Interconnect | 6 chip-to-chip event links (hexagonal grid → 48-chip torus); on-die 400 MHz multicast event router; dual NoC |
| Host interface | **1 GbE, UDP/UDT — no PCIe, no RDMA, no kernel driver** |
| Chip power (measured) | 0.075 W (PEs off) → 1.248 W (all 152 PEs on CoreMark @0.8 V); summary range **0.24–2.2 W** |
| Rated TDP | **Not disclosed** |
| SNN capacity | **>150,000 neurons/chip**, **>1.8 G synaptic events/s** at a 1 ms tick |
| Successor | **SpiNNext — announced only.** No silicon, no specs, no process node, no date |
| MLPerf | **None.** No independently audited benchmark exists |

---

## What It Is — and What It Is Not

SpiNNaker2 keeps SpiNNaker1's defining idea — a manycore of small ARM cores tied together by a **multicast
event router** rather than a memory-coherence fabric — and adds per-core fixed-function accelerators, DVFS,
and LPDDR4.

**It is not a dense-GEMM competitor to TPU/Gaudi-class parts, and its designers say so:**

> *"SpiNNaker2 is not a pure DNN inference accelerator… limited DRAM bandwidth often reduces utilization of
> the DNN accelerators if DNN layers are too big to be stored in on-chip memory… scalability of conventional
> DNN workloads on a single chip is limited."*
> — [arXiv:2607.24396](https://arxiv.org/abs/2607.24396), IEEE OJCAS 2026

Its value in this survey is as a **contrasting corner of the design space**: microcontroller-class cores,
sub-watt power, event-driven sparse execution, and a programming model built out of statically placed
bare-metal C rather than a compiler.

**Primary source note.** A definitive full-chip paper appeared on **27 July 2026** — Scholze, Partzsch,
Höppner et al., *"The SpiNNaker2 chip,"* IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974, CC-BY 4.0. It
**supersedes** the 2021/2022 processing-element paper, which described an **8-PE testchip**, not the product
(e.g. the older paper's "8-bit unsigned MACs with 29-bit accumulator" is the testchip revision; the product
is 8-bit **signed/unsigned** with a **24-bit** accumulator).

---

## Software Stack

```
PyTorch (DNN) ──► ONNX ──► AMD Quark INT8 PTQ ──► OctopuScheduler
                                                    │ application graph IR
                                                    │ layer fusion, tiling, PE mapping
                                                    ▼
snnTorch/Norse/Sinabs/                          S2Layer / S2Model
Rockpool/Nengo/Lava-DL/Spyx                     → complete DRAM image
      │ NIR export                                    │
      ▼                                               │
py-spinnaker2 (Apache-2.0)                            │
  Populations/Projections · s2_nir.from_nir()         │
  partitioner → placer → routing tables               │
      └──────────────┬───────────────────────────────┘
                     ▼
        GNU Arm Embedded GCC + custom linker scripts
        (bare-metal C per PE; CMSIS-NN; ml-lib MMIO calls)
                     ▼
        ExperimentRunner (C++) / SpiNNMan2
        spec.json → UDP+UDT over 1 GbE → results.json
                     ▼
        NO kernel driver · NO OS on chip
        scheduler-PE FSM (DNN) / timer_callback loop (SNN)
```

**The headline: there is no tensor compiler.** No MLIR/LLVM lowering path, no XLA/StableHLO, no Triton, no
TorchDynamo backend, no device tensor type, no collective-communication library, and no kernel-mode driver.
Framework reach is achieved through **graph import** (ONNX for DNNs, NIR for SNNs), not through a native
compiler. The compiler-shaped work — tiling, placement, memory planning — lives entirely in host-side Python.

The chip paper describes the stack in three tiers: bare-metal C chip software built with GCC; a C++ low-level
host library giving memory-mapped access over UDP; and Python high-level software for SNN and DNN
applications.

### Two disjoint toolflows

| | SNN flow | DNN flow |
|---|---|---|
| Front end | snnTorch / Norse / Sinabs / Rockpool / Nengo / Lava-DL / Spyx → **NIR** | PyTorch → **ONNX** |
| Quantization | weight rescaling in `ConversionConfig`; PTQ/QAT in application papers | **AMD Quark** INT8 power-of-two PTQ with cross-layer equalization |
| Mapper / compiler | py-spinnaker2 `graph/` pipeline (partitioner → placer → routing tables); optional **PACMAN2** pipeline | **OctopuScheduler** → application graph → `S2Layer` / `S2Model` → complete DRAM image |
| On-chip execution | timer-tick `timer_callback()` loop, **no barrier sync** | scheduler-PE FSM driving up to 151 worker PEs, **no host interaction after one interrupt** |
| Openness | Apache-2.0, fully public | papers only; **no standalone public repo located** |

### Openness split

- **Public Apache-2.0:** `py-spinnaker2` (flagship SDK, v0.8.0 released 31 July 2026), `spinnaker2-ml`,
  `snntorch-to-spinnaker2`, `s2-sim2lab-app-snn`, `spinnaker2-config-tools`, the docs portal.
- **Public ECL-2.0** (SpiNNaker1-lineage host toolchain carried forward): `PACMAN2`, `SpiNNMan2`,
  `SpiNNMachine2`, `SpiNNUtils2`.
- **Access-gated (HTTP 403):** `s2-sim2lab-app` (full board SDK incl. `ml-lib`), `spinnaker2_os`
  (SARK/SCAMP/Spin2API), board firmware, `mlops-infrastructure`.
- **Explicitly proprietary:** SpiNNcloud's large-scale-system stack, confirmed by two primary sources to
  exist and be "currently under development"; name and licensing **not disclosed**.

**Practical consequence:** the open stack is **single-chip and 48-node-board scoped**. The 35k-chip Dresden
machine and Sandia's NERL Braunfels run on software that is not public.

---

## Hardware Architecture

### Compute

152 Cortex-M4F PEs in 38 Quad units. Each PE carries a **16×4 output-stationary MAC array (64 MACs)** doing
8-bit signed/unsigned matmul and 2-D convolution — two adjacent cells fuse for 16-bit operands, with a 24-bit
accumulator and shift+truncate output requantization to 8/16/32-bit. Alongside it sit an **iterative exp/log
unit** (4.4–5.7× faster than the ARM core for exp/log/sigmoid/tanh/softmax), a **rounding unit** with
FP→BFloat16 and **stochastic rounding**, a **per-PE PRNG** (MARS KISS64) with one global jitter-derived TRNG,
and a hardware **event handler** that lands matching SpiNNaker packets in an SRAM-mapped FIFO **without
interrupting the ARM core**.

All accelerators are **memory-mapped peripherals on AHB/APB** — no custom ISA, no vendor compiler backend.

**Utilization is the story, not peak.** Measured MAC utilization is 16–19 % on fully-connected layers, 22–50 %
on convolutions, 78 % on transformer matmuls. With DRAM in the loop, memory dominates: DRAM transfers were
**71.6–98.0 %** of matmul layer time in the OctopuScheduler benchmark, and in one MLP case *"the accelerator
is only used for 9 % of the layer runtime."*

### Memory — the survey-relevant property

**There is no cache hierarchy and no shared virtual memory.** Every byte movement is an explicit DMA or NoC
transaction issued by software, and the memory map is planned **statically by the host** before execution:

- **128 kB SRAM per PE**, 4 banks, linker-partitioned into ≈32 kB **ITCM** (code) + ≈96 kB **DTCM**
  (host-allocated regions: routing tables, synapses, neuron state, recording buffers, log region), heap+stack
  4 kB at `0x1F000`.
- **512 kB per Quad** addressable by all 4 PEs in synchronous mode.
- **19.8 MB total on-chip SRAM** (the portal's "19 MB" rounded).
- **2 GB LPDDR4** per chip over two 800 MHz interfaces, **25.6 Gbit/s raw each = 6.4 GB/s aggregate raw**.
  Sustained bandwidth **not disclosed**. 6.4 GB/s against 5.8 TOPS implies an arithmetic-intensity
  requirement around 900 ops/byte — which is exactly why the DNN benchmarks are DRAM-bound.

### Interconnect

- **DNoC** — 2-D mesh, 192-bit flit (whole packet in one transaction), 300 MHz, **307.2 Gbit/s bisection**,
  5 cycles/hop, XY routing or optional 64-entry LUT routing.
- **CNoC** — 32-bit wormhole config NoC on the 100 MHz reference clock, alive before any PLL is up.
- **Event router** — at die centre, 400 MHz, three parallel routing engines, **16,384 multicast entries**
  (32-bit wildcarded source filters), out-of-order issue buffer, **1,843.2 Gbit/s to PEs / 460.8 Gbit/s to
  external links**.
- **Chip-to-chip** — 6 bidirectional links per chip, hexagonal grid, **48 chips = one toroidal board**.
  Short-range on-board mode: 0.5 V low-swing, 6 lanes DDR on a 500 MHz differential clock = **6 Gbit/s**,
  CRC-12 with resend. Long-range board-to-board mode: LVDS, 1 GHz DDR = **2 Gbit/s**, 8b10b with CDR.
  Aggregate per-chip link bandwidth **not disclosed**.
- **Host** — 1 Gbit/s UDP Ethernet per chip over SGMII with UDT reliability and an Ethernet-to-NoC bridge.
  On a 48-node board **only chip 43** has the GbE connection. **No RDMA, no PCIe, no collective library.**

### Execution and power

**No operating system.** Each core runs a small pre-compiled program that executes tasks on event reception;
dispatch is the ARM interrupt controller, with WFI/WFE sleep between events. In SNN mode a chip-global
interrupt wakes all PEs each 1 ms tick and **there is no barrier synchronization** — a PE that overruns simply
falls out of phase and is logged. In DNN mode one scheduler PE drives up to 151 workers through an FSM with no
host interaction after a single interrupt (measured 80.9–99.6 % utilization, ≈13 µs per-layer overhead).

Every PE is its own switchable power domain with **4 DVFS performance levels** (0.5 V/150 MHz and
0.8 V/300 MHz nominal), GALS-clocked, with Quad hardware semaphores throttling simultaneous switching against
IR droop. Per-core per-timestep automatic level selection cut DVS-gesture SNN energy **28 %** at identical
accuracy.

---

## Scale — Read the Verbs Carefully

| System | Scale | Status |
|---|---|---|
| TU Dresden / SpiNNcloud | 5 M cores, ~35,000 chips, eight racks | **Operating**; large-scale software stack "currently under development" |
| Sandia NNSA **NERL Braunfels** | **175 M neurons**, 3 chassis × up to 18 boards, 48 chips/board | **Delivered March 2025**, publicized 12 June 2025, NNSA ASC funded |
| Dresden full build | "16 racks (69,120 chips) … 10.5 billion neurons" | **Target, future tense — not delivered** |
| Largest *offered* config | "10 billion neurons", "0.3 exaops" | **Vendor/press framing of an offered configuration**, never a measured system |
| **SpiNNext** | — | **Announced only.** No process node, no specs, no silicon status, no date |

**Measured end-to-end performance of either large system on any workload is not public** — every published
SpiNNaker2 measurement is single-chip.

---

## Competitive Position

SpiNNaker2 competes on **energy per inference for sparse, event-driven, small-batch workloads**, not on
throughput. The chip paper's own measured comparison (Tab. 5, INT8, 6 real DNN layers *including* DRAM traffic
and scheduling) is the honest framing:

- SpiNNaker2 @150 MHz: **0.58–0.80 W** · A100: **107–133 W** · Jetson Orin Nano: **3.1–8.2 W**
- SpiNNaker2 won on **energy** for **4 of 6** layers (FC1: 0.064 mJ vs 1.332 mJ on A100)
- It lost badly on the two largest matmuls (MM2: 17.07 mJ vs 4.18 mJ on A100)
- It was **30–700× slower in wall-clock** throughout

The paper's efficiency figure (2.77 TOPS/W) sits mid-pack against Coral EdgeTPU (2.00), Mobileye EyeQ5 (2.40),
Jetson Orin Nano (2.68), IBM NorthPole (2.70), Groq TSP (2.73), ARM Ethos N77 (5.13). Against neuromorphic
peers (chip paper Tab. 6) it trades Loihi 2's 7 nm process and 31 MB SRAM for 152 general-purpose ARM cores
and full C programmability.

### Vendor claims — label as such

- **"18× more energy efficient than GPUs" (SpiNNaker2)** — unaudited marketing. It does have one traceable
  peer-reviewed anchor: EGRU language-model inference at **65 mJ on SpiNNaker2 vs 1.19 J on an A100**, an 18×
  energy reduction *for single-batch inference*, **at 8× longer execution time**. Cite it that way or not at
  all.
- **"78× more energy efficient than GPUs" (SpiNNext)** — unaudited marketing with **no anchor of any kind**;
  the product page carries no specifications.
- The SpiNNcloud homepage renders its headline counters as unpopulated placeholders ("Cores: 0M", "Chips: 0K",
  "Boards: 0", "Racks: 0") and is **unusable as a spec source**.

### Distinguishing design choices

1. **Microcontroller cores as the compute substrate** — full C programmability, upstream GCC/LLVM, zero vendor
   ISA maintenance, at the cost of dense-GEMM throughput.
2. **No hardware cache anywhere** — 100 % software-managed memory hierarchy with a host-planned static map.
3. **No tensor compiler** — graph import (ONNX / NIR) plus a host-side Python mapper replaces the entire
   MLIR/Triton layer that every compiler-centric stack in this registry depends on.
4. **No kernel driver, no PCIe** — the host is a userspace process talking UDP over 1 GbE to a memory-mapped
   chip.
5. **Event router as the fabric** — multicast spike routing with 16,384 wildcarded TCAM entries instead of
   coherence or collectives.
6. **Per-PE DVFS with four levels and adaptive body biasing** — measured 28 % energy saving from per-timestep
   level selection.
7. **No barrier synchronization in SNN mode** — a deliberate choice trading determinism for scalability.

---

## Sources

- [The SpiNNaker2 chip — Scholze, Partzsch, Höppner et al., IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974](https://arxiv.org/abs/2607.24396)
- [SpiNNaker2: A Large-Scale Neuromorphic System… — Gonzalez et al., NeurIPS 2023 MLNCP](https://arxiv.org/abs/2401.04491)
- [The SpiNNaker 2 Processing Element Architecture — Höppner, Yan, Vogginger et al.](https://arxiv.org/abs/2103.08392)
- [An End-to-End DNN Inference Framework for the SpiNNaker2 MPSoC — Jobst et al., ICONS 2025](https://arxiv.org/abs/2507.13736)
- [Language Modeling on a SpiNNaker2 Neuromorphic Chip — Nazeer et al., AICAS 2024](https://arxiv.org/abs/2312.09084)
- [py-spinnaker2 SDK (Apache-2.0)](https://gitlab.com/spinnaker2/py-spinnaker2) · [documentation](https://spinnaker2.gitlab.io/py-spinnaker2/)
- [SpiNNaker2 Developer Portal](https://spinnaker2.gitlab.io/)
- [Sandia Lab News, 12 Jun 2025 — NERL Braunfels](https://www.sandia.gov/labnews/2025/06/12/brain-based-computing-for-nd-solutions/)
- [Sandia, 8 May 2024 — SpiNNcloud partnership](https://www.sandia.gov/research/2024/05/08/neuromorphic-computing-for-nuclear-deterrence-solutions-sandia-partners-with-german-startup-spinncloud/)
- [IEEE Spectrum, 8 May 2024](https://spectrum.ieee.org/neuromorphic-computing-spinnaker2)
- [EE Times — Dresden opening](https://www.eetimes.com/spinnaker-based-neuromorphic-supercomputer-opens-in-dresden/)
- [SpiNNcloud SpiNNext product page (roadmap only)](https://spinncloud.com/spinnext/)
