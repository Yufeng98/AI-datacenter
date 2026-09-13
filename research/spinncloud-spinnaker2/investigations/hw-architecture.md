# SpiNNcloud SpiNNaker2 Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*

---

## Overview

SpiNNaker2 is the second-generation SpiNNaker ("Spiking Neural Network Architecture") chip, designed at
TU Dresden in the Höppner/Mayr group and commercialized by **SpiNNcloud Systems GmbH**. It descends from
Steve Furber's Manchester SpiNNaker1 and keeps that machine's defining idea — a manycore of small ARM cores
tied together by a **multicast event router** rather than a memory-coherence fabric — while adding per-core
fixed-function accelerators, DVFS, and an LPDDR4 interface.

The chip is **not a dense-GEMM accelerator**, and its own designers say so:

> *"SpiNNaker2 is not a pure DNN inference accelerator… limited DRAM bandwidth often reduces utilization of
> the DNN accelerators if DNN layers are too big to be stored in on-chip memory… scalability of conventional
> DNN workloads on a single chip is limited."*
> — [arXiv:2607.24396](https://arxiv.org/abs/2607.24396), IEEE OJCAS 2026

Its value in this survey is as a **contrasting point in the design space**: 152 microcontroller-class cores,
19.8 MB of software-managed SRAM, no cache hierarchy, no shared virtual memory, no operating system, and
sub-watt to ~2 W chip power — the opposite corner from an HBM-fed systolic array.

### Primary source note

A definitive full-chip paper appeared on **27 July 2026**: Scholze, Partzsch, Höppner et al., *"The
SpiNNaker2 chip: a many-core platform for flexible and scalable brain-inspired computing,"* IEEE Open
Journal of Circuits and Systems 2026, DOI 10.1109/OJCAS.2026.3714974,
[arXiv:2607.24396](https://arxiv.org/abs/2607.24396), CC-BY 4.0. It **supersedes** several numbers in the
2021/2022 processing-element paper ([arXiv:2103.08392](https://arxiv.org/abs/2103.08392)), which described an
**8-PE, 8.76 mm² testchip** rather than the product. Where the two disagree, the 2026 chip paper is used and
the older revision is flagged.

---

## 1. Compute Engine

### Processing elements

| Item | Value | Confidence |
|---|---|---|
| Processing elements per chip | **152 PEs**, as **38 Quad units (QPEs) × 4 PEs** | confirmed |
| PE core | **ARM Cortex-M4F**, single-precision FPU, 45 interrupt sources, 3 timers on a 1 MHz reference clock | confirmed |
| Management core (the "153rd core") | **1 × ARM Cortex-M4** in the periphery module, FPU and accelerators removed, **100 MHz, 128 kB ECC SRAM**; boot, background tasks, broadcasting application code to PEs | confirmed |
| Cores available to applications | **148 per chip** (4 reserved for system software) → **7,104 usable of 7,296** on a 48-node board | confirmed |
| PE clock | 150 MHz @ 0.5 V (low power) · 300 MHz @ 0.8 V (performance) | confirmed |

The vendor developer portal's "**153 ARM cores**" and the papers' "**152 PEs**" are reconciled by the
periphery management M4. State whichever convention is in use explicitly; both are defensible.

### Per-PE accelerators (memory-mapped, not ISA extensions)

| Accelerator | Detail | Confidence |
|---|---|---|
| **ML accelerator (MLA)** | **16 × 4 output-stationary MAC array = 64 MACs/PE**; 8-bit signed/unsigned; two neighbouring 8-bit cells fuse for **16-bit** row- or column-wise operands; **24-bit accumulator**; 4 post-processing modules; output quantized by shift + truncate to 8/16/32-bit. Operations: 2-D matrix multiply (MM) and 2-D convolution (CONV), with a shift register for input-feature-map reuse in CONV; operands prefetched directly over the NoC; ReLU fusable with matmul | confirmed |
| **Numerical accelerator** | Iterative **exp / natural-log** unit; s16.15 and s0.31 fixed-point **and** FP32, operand and result formats chosen independently; configurable 1–16 iterations = 7–22 cycles; 1–2 ulp accuracy, monotonic. Measured **4.4×–5.7× speed-up** over the ARM core for exp/log/sigmoid/tanh/softmax | confirmed |
| **Rounding accelerator** | Signed/unsigned integer rounding + saturation of 64/32/16-bit values; FP→**BFloat16**; round-to-nearest and **stochastic rounding** driven by the PRNG; 4 parallel threads, 3–4 cycles each; configurable rounding bit position up to 32 LSBs | confirmed |
| **Random numbers** | **PRNG in every PE** (MARS KISS64, 32 bit/cycle) + **one global TRNG** derived from ADPLL bang-bang jitter; the TRNG can scramble the PRNGs | confirmed |
| **Event handler ("spDMA")** | Two configurable hardware filters plus a default handler; matching SpiNNaker packets (32-bit multicast key and/or payload) are written into an **SRAM-mapped hardware FIFO without interrupting the ARM core**; configurable full-behaviour (drop+count / overwrite / stall NoC); interrupts on non-empty / fill-level / full | confirmed |

> The 2022 prototype paper described the MAC array as **8-bit unsigned only with a 29-bit accumulator**.
> That is the older testchip revision and should not be quoted for the product.

### Throughput

| Metric | Value | Condition |
|---|---|---|
| Total MACs per chip | 152 × 64 = **9,728** 8-bit MACs | structural |
| **Peak INT8** | **5.837 TOPS theoretical** | 300 MHz |
| **Measured INT8** | **4.563 TOPS** (78 % MAC utilization, transformer matmul, full-chip tiling) | 300 MHz / 0.8 V |
| **Measured INT8** | **2.281 TOPS** | 150 MHz / 0.5 V |
| **Energy efficiency** | **2.77 TOPS/W** (0.825 W) | 150 MHz / 0.5 V, operands resident in on-chip SRAM |
| **Energy efficiency** | **2.06 TOPS/W** (2.219 W) | 300 MHz / 0.8 V, same condition |
| CPU-only reference | 65,968 CoreMark/s (all 152 PEs @ 0.5 V/150 MHz); 138,168 CoreMark/s (@ 0.8 V/300 MHz) | — |
| SNN capacity | **>150,000 neurons per chip**; **>1.8 billion synaptic events/s** at a 1 ms time step; up to **12 M synaptic events/s per core** with 1024 sparse-connectivity neurons | 1 ms tick |

**Throughput at any precision other than INT8 is not disclosed.** The MLA is 8/16-bit integer only; the
Cortex-M4F FPU provides FP32 but no aggregate FLOPS figure is published.

The chip paper's own comparison table places the 2.77 TOPS/W figure alongside Coral EdgeTPU (2.00), Mobileye
EyeQ5 (2.40), Jetson Orin Nano (2.68), IBM NorthPole (2.70), Groq TSP (2.73), ARM Ethos N77 (5.13).

### Utilization — the number that matters more than peak

Measured MAC utilization: **16–19 %** on fully-connected layers (only 1 of 4 MAC-array rows used),
**22–50 %** on convolutions, **78 %** on transformer matmuls. Once DRAM is in the loop, memory dominates:
in the OctopuScheduler benchmark DRAM transfers were **71.6–98.0 %** of matrix-multiply layer time; in the
smaller MLP case weight fetch was 192 µs of a 323 µs layer while MLA compute was 29 µs — *"the accelerator is
only used for 9 % of the layer runtime"* ([arXiv:2507.13736](https://arxiv.org/abs/2507.13736)).

---

## 2. Data Path

### DNN mode — scheduler / worker FSM

One **scheduler PE** drives up to **151 worker PEs** through a finite state machine. The host writes the
entire DRAM structure (global config, per-layer config blocks, data memory) and raises one interrupt; from
then on *the whole multi-layer schedule runs on-chip with no host interaction*:

```
Host (UDP/Ethernet)
   └─ writes DRAM image: global config │ per-layer 128-bit headers │ layer params │ data area
   └─ raises ONE interrupt
Scheduler PE  ── reads layer header from DRAM ──► forwards + IRQs assigned worker PEs
   Worker PE  ── fetches own config ──► DMA weights/inputs DRAM→local SRAM
              ── configures MLA (mlacc_params) ──► execute_mm() / execute_conv()
              ── DMA result SRAM→DRAM ──► sets completion flag ──► WFI sleep
Scheduler PE  ── polls completion flags ──► advances to next layer
```

Workers are asynchronous *within* a layer (no intra-layer sync); layers are synchronous. Measured
worker/scheduler utilization 80.9–99.6 %; per-layer scheduling overhead ≈13 µs; full-chip setup/cleanup
39 µs / 93 µs.

### SNN / event mode

Each PE simulates a set of neurons and their incoming synapses. Neuron models are integrated **in software**
on the ARM core by Euler method at a discrete tick (typically 1 ms, derived from the 1 MHz reference clock).
A chip-global interrupt wakes all PEs simultaneously; each tick a PE reads its spike-FIFO pointers, processes
the spikes received in the previous step, updates neurons, emits multicast packets, and returns to WFI.

**There is no barrier synchronization.** If a PE overruns its tick it simply falls out of phase and may catch
up later; such incidents are logged and reported. Multi-chip time sync is planned via a per-chip **SCAMP**
program reusing the SpiNNaker1 drift-compensation mechanism.

### DVFS

Each PE is its own switchable power domain with **4 predefined performance levels** (two supply voltages ×
clock frequency, frequency divided locally from a central ADPLL). PEs run GALS — individually asynchronous,
or synchronously coupled within a Quad. Switching can be driven by the PE itself or by a global controller;
Quad hardware semaphores throttle simultaneous switching against IR droop. In the DVS-gesture SNN, per-core
per-timestep automatic performance-level selection cut energy **28 %** versus always-high-PL at identical
accuracy (0.741 J vs 1.023 J, 92.04 %).

---

## 3. On-chip Memory — entirely software-managed, no hardware data cache

This is the single most survey-relevant property of the chip. There is **no cache hierarchy and no shared
virtual memory**; every byte movement is an explicit DMA or NoC transaction issued by software, and the
memory map is planned **statically by the host** before execution.

| Level | Capacity | Management |
|---|---|---|
| Register file | Cortex-M4F architectural registers + FPv4-SP FP registers | compiler |
| **PE-local SRAM** | **128 kB per PE** (1 Mbit), split into **4 addressable banks** to reduce contention | **Software / linker managed**: ≈**32 kB ITCM** (compiled neuron-model / application code) + ≈**96 kB DTCM** (host-allocated data-spec regions: routing tables, synapses, neuron state, recording buffers, log region), heap+stack of 4 kB at `0x1F000` |
| Quad-shared SRAM | 4 × 128 kB = **512 kB per Quad**; PEs within one Quad can address the full Quad SRAM in synchronous mode (low-latency sharing via the PE crossbar) | software |
| **Total on-chip SRAM** | **158.625 Mbit = 19.8 MByte** (152 PEs × 1 Mbit + event links 896 kbit + Ethernet 1.875 Mbit + GPIO/mgmt 1 Mbit + event router 2.875 Mbit) | — |

The vendor portal's "19 MB" is the same figure rounded.

**Maximum neurons/synapses per PE is deliberately not disclosed**: the chip paper states the number *"varies
significantly with parameters such as weight precision, logging, connectivity and simulation time-step"* and
that studying the trade-offs is out of scope.

---

## 4. Off-chip Memory

| Item | Value |
|---|---|
| Interfaces | **2 × LPDDR4**, each able to address **up to 4 GB** |
| Shipping node configuration | **2 GB per chip** (1 GB per interface) |
| PHY / controller | Uniquify LPDDR4 PHY + controller at **800 MHz** |
| **Raw bitrate** | **25.6 Gbit/s per interface** → 51.2 Gbit/s ≈ **6.4 GB/s per chip aggregate raw** |
| **Sustained bandwidth** | **Not disclosed** — only the raw controller bitrate is published |
| DMA offload | Each interface has its own DMA controller to offload bulk transfers and keep the NoC uncongested |
| Management | Software; **statically partitioned by the host** before the run — global config, per-layer time-measurement area, per-layer config blocks, then a data-memory area holding all layer inputs / intermediates / outputs |
| DMA API | `dma_rd()` / `dma_wr()` (PE↔PE), `dram_dma_rd()` / `dram_dma_wr()` (SRAM↔DRAM); 16-byte alignment required; reads may not cross the interface boundary; addresses `0x0–0x3FF` and `0x40000000–0x400003FF` reserved |

6.4 GB/s raw against 5.8 TOPS peak is an arithmetic intensity requirement of roughly 900 ops/byte — which is
exactly why the measured DNN benchmarks are DRAM-dominated.

---

## 5. Interconnect — on-chip

### Two NoCs plus a separate event router

| Fabric | Specification |
|---|---|
| **Data NoC (DNoC)** | 2-D mesh, **192-bit flit** (a whole packet incl. 128-bit payload moves in one transaction), **300 MHz**, **bisection bandwidth 307.2 Gbit/s**, 5-cycle latency per hop (GALS async-FIFO crossings). XY (x-first) routing by default; optional **64-entry (8×8) lookup-table routing**. Round-robin output arbitration; multicast at QPE granularity via 4 destination-PE bits |
| **Configuration NoC (CNoC)** | 32-bit flit, wormhole switching, runs on the **100 MHz reference clock** so it is alive before any PLL is up; same packet format as DNoC; used for boot, register access and production test; remains usable when the DNoC is blocked by large DMAs |
| NoC packet format | 15-bit NoC header + 17-bit packet header + 32-bit address + 0–128-bit payload |
| **Event router (SpiNNaker router)** | At the **centre of the die**, occupying the area of two Quads; **400 MHz**; 6 NoC ports; input crossbar feeding **three parallel routing engines**. The multicast engine is a 4-stage pipeline (TCAM lookup → priority encode → link-destination resolution → 152-core destination lookup, last stage power-gated when no local destination). **16,384 multicast routing entries**, each a 32-bit source filter with wildcards. Out-of-order issue buffer prevents head-of-line blocking. **Peak 1,843.2 Gbit/s to PEs and 460.8 Gbit/s to external links.** 16 configurable diagnostic counters |

### Packet types

| Type | Fields | Use |
|---|---|---|
| Multicast | 40-bit packet, 32-bit source ID (+ up to 128-bit payload) | the spike/event path |
| Core-to-Core | 16-bit chip + 8-bit core address | point-to-point messaging |
| Nearest-Neighbour | 32-bit address | boot, debug |
| Global Read/Write | direct access to any address on any chip | host and management access |

All four support 0 / 32 / 64 / 128-bit payloads.

---

## 6. Interconnect — chip-to-chip (scale-up)

Each chip has **six bidirectional event links**, placing it as a node in a **hexagonal grid**; **48 chips
form a toroidal mesh** on one board. Each link supports two mutually exclusive physical modes:

| Mode | Reach | Specification |
|---|---|---|
| **Short-range** (on-board) | a few cm | Chiplet-like low-swing interface; eight IO pad cells at **0.5 V**; six data lanes with **DDR on a 500 MHz differential clock** (0.25 V common mode) → **6 Gbit/s**. 96-bit packets, serialization factor 8, transmitted in two 125 MHz system-clock cycles. **CRC-12** error detection with packet-ID-based resend and ordering. Clock and data driven to ground when idle for immediate-restart power saving |
| **Long-range** (board-to-board) | up to 1.5 m | Two LVDS pads per link, **1 GHz DDR = 2 Gbit/s**, 8b10b coded with clock-data recovery (must stay active to hold sync, so its power-down has a longer lead time) |

> The developer portal phrases the link differently — *"2 Gbps per lane in each direction… combined
> throughput of 12 Gbps bidirectionally"*. Attribute that phrasing to the portal, not to the paper.

**Aggregate off-chip event-link bandwidth per chip is not disclosed** — per-link rates are published, but no
aggregate or board/rack bisection figure exists.

---

## 7. Interconnect — scale-out / host

- **1 Gbit/s UDP Ethernet per chip** over SGMII (625 MHz DDR), extended with **UDT** (reliable UDP with
  congestion control) plus an optional packet-counter field. An **Ethernet-to-NoC bridge** interprets UDP
  payloads carrying magic headers as NoC packets with on- or off-chip destinations, with programmable
  routing LUTs.
- On a 48-node board **only chip 43** carries the Gigabit Ethernet connection; the rest are reached through
  the fabric.
- **No RDMA fabric, no PCIe, no collective-communication library.** The host attaches over Ethernet.
- **27 GPIOs at 1.8 V**: 4 × UART, QSPI master/slave to 200 Mbit/s, I²C master/slave to 1 Mbit/s, 13 PWM
  channels, behind a flexible multiplexer — for direct sensor / robot integration.

---

## 8. Physical Specs

| Item | Value |
|---|---|
| Process | **GlobalFoundries 22FDX (22 nm FDSOI)** with **Racyics ABX adaptive body biasing** (forward-body-bias scheme) |
| Die area | **102 mm²** |
| Complexity | **266 M gate equivalents**; 10 independent implementation macros in a tile-based flow sized on the Quad as reference |
| Transistor count | **not disclosed** (266 M gate equivalents is not a transistor count) |
| Supply rails | Only two digital voltages: **0.5 V and 0.8 V**. Always-on zero-body-biased 0.8 V domain for comms/config; each PE in its own switchable FBB domain |
| Clock domains | PE 150 MHz @0.5 V / 300 MHz @0.8 V · DNoC 300 MHz · event router 400 MHz · CNoC + periphery 100 MHz reference · short-range link 500 MHz differential · LPDDR4 800 MHz · SGMII 625 MHz |
| **Measured chip power** | PEs off **75.2 mW** · PEs sleeping @0.5 V **235.4 mW** · sleeping @0.8 V **564.7 mW** · all 152 PEs running CoreMark @0.5 V **424.6 mW** · @0.8 V **1,247.9 mW**. Summary row: **0.24–2.2 W**. Abstract: **baseline power below 250 mW** |
| Single-chip application power | Mamba SSM (170 M parameters) on one chip: **≈0.6 W average** (vendor developer-portal figure) |
| **Rated TDP** | **Not disclosed** — only measured workload power is published |
| Package type / ball count / dimensions / thermal solution | **Not disclosed** |

### Form factors

| Board | Contents |
|---|---|
| **SpiNNode** (single chip) | STM32H743 manager, ≥2 GB LPDDR4, 1G Ethernet RJ45, USB-C serial CLI, DCMI camera connector, 5 V or 12–24 V barrel input, JTAG, fan header |
| **48-node board** | 48 chips, 2 GB LPDDR4 each, toroidal mesh, on-board STM32 with 2 MB flash for boot/management over an SPI chain + I²C, GbE on chip 43, host either via STLink or via a Backbone/STM Card over CAN |

Board- and rack-level power draw, cooling method and physical dimensions are **not disclosed**.

### Predecessor / peer comparison (chip paper Tab. 6)

| | SpiNNaker1 | **SpiNNaker2** | Loihi | Loihi 2 |
|---|---|---|---|---|
| Process | 130 nm | **22 nm FDSOI** | 14 nm | 7 nm |
| Clock | 180 MHz | **150 / 300 MHz** | — | — |
| Die area | 102 mm² | **102 mm²** | 60 mm² | 60 mm² |
| On-chip SRAM | 1.8 MB | **19.8 MB** | 33 MB | 31 MB |
| Power | 0.36–1 W | **0.24–2.2 W** | — | — |
| Cores | 18 × ARM M0 | **152 × ARM M4F** | 128 | 128 |
| Neurons | 16,000 | **150,000** | — | — |

---

## 9. Systems As Built — distinguish carefully

| System | Scale | Status |
|---|---|---|
| TU Dresden / SpiNNcloud | **5 million cores, ~35,000 chips, eight racks** | Operating; *"the software stack for large-scale deployment on this system is currently under development"* |
| Sandia NNSA **NERL Braunfels** | **175 million neurons**, 3 chassis of **up to 18 boards** each, 48 chips per board, >150 PEs per chip | Delivered **March 2025**, unboxed 3 April 2025, publicized 12 June 2025; NNSA ASC funded |
| Dresden full-build target | *"will be 16 racks (69,120 chips total) for a total of 10.5 billion neurons"* | **Target, future tense. Not delivered.** |
| Largest commercially *offered* config | "10 billion neurons", "0.3 exaops" | **Vendor/press framing of an offered configuration**, not a measured deployed system |
| **SpiNNext** | — | **Announced only.** No process node, no specs, no silicon status, no date |

**Measured end-to-end performance of the Dresden or Sandia systems on any workload is not public** — all
published SpiNNaker2 measurements are single-chip.

### Vendor claims — label as such

The "18× higher energy efficiency than GPUs" (SpiNNaker2) and "78×" (SpiNNext) figures on spinncloud.com are
**unaudited marketing**. There is **no MLPerf submission** and no independent benchmark. The 18× figure does
have a traceable peer-reviewed anchor for one specific case: the chip paper reports EGRU language-model
inference at **65 mJ on SpiNNaker2 vs 1.19 J on an A100** — an 18× energy reduction *for single-batch
inference*, at the cost of **8× longer execution time**. Report it that way, or not at all.

The company homepage renders its headline metrics as unpopulated placeholders ("Cores: 0M", "Chips: 0K",
"Boards: 0", "Racks: 0"), so it is unusable as a spec source.

### Independent GPU comparison from the chip paper (Tab. 5, INT8, all measured)

On 6 real DNN layers *including* DRAM traffic and scheduling, SpiNNaker2 at 150 MHz used **0.58–0.80 W** vs
**107–133 W** (A100) and **3.1–8.2 W** (Jetson Orin Nano). It won on energy for **4 of 6** layers (e.g. FC1:
0.064 mJ vs 1.332 mJ A100) and lost badly on the two largest matmuls (MM2: 17.07 mJ vs 4.18 mJ A100), while
being **30–700× slower in wall-clock**.

---

## Open Items / Not Disclosed

- Rated TDP; package type, ball count, dimensions, thermal solution
- Transistor count (only 266 M gate equivalents published)
- Sustained (as opposed to raw) LPDDR4 bandwidth
- Aggregate off-chip event-link bandwidth per chip; board/rack bisection bandwidth
- Board- and rack-level power, cooling, dimensions
- Price of any product; units shipped; revenue; customer list beyond Sandia and TU Dresden
- Peak throughput at any precision other than INT8
- Maximum neurons/synapses per PE as a hard limit (explicitly declined by the chip paper)
- Reconciliation of the portal FAQ's QuadPE coordinate range (x = 1..7, y = 1..6 with (4,3) and (4,4)
  missing → 40 positions) with the confirmed count of **38 QPEs**. The portal does not explain the gap
- The role of `mercury-xu8-bsp` (Enclustra Mercury XU8 / Zynq UltraScale+ SoM) in SpiNNcloud systems
- Everything about **SpiNNext**

---

## Sources

- [The SpiNNaker2 chip (Scholze, Partzsch, Höppner et al., IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974)](https://arxiv.org/abs/2607.24396)
- [SpiNNaker2: A Large-Scale Neuromorphic System… (Gonzalez et al., NeurIPS 2023 MLNCP)](https://arxiv.org/abs/2401.04491)
- [The SpiNNaker 2 Processing Element Architecture (Höppner, Yan, Vogginger et al.)](https://arxiv.org/abs/2103.08392)
- [An End-to-End DNN Inference Framework for the SpiNNaker2 MPSoC (Jobst et al., ICONS 2025)](https://arxiv.org/abs/2507.13736)
- [Developer portal — ML accelerator](https://spinnaker2.gitlab.io/external/documentation/hardware/1-ml-accelerator/)
- [Developer portal — chip topology](https://spinnaker2.gitlab.io/external/documentation/hardware/2-s2-chip-topology/)
- [Developer portal — NoC](https://spinnaker2.gitlab.io/external/documentation/hardware/11-NoC/)
- [Developer portal — SpiNNaker router](https://spinnaker2.gitlab.io/external/documentation/hardware/8-spinnaker-router/)
- [Developer portal — chip-to-chip link](https://spinnaker2.gitlab.io/external/documentation/hardware/2-chip-to-chip-link/)
- [Developer portal — DRAM](https://spinnaker2.gitlab.io/external/documentation/hardware/4-dram/)
- [Developer portal — DMA](https://spinnaker2.gitlab.io/external/documentation/hardware/7-dma/)
- [Developer portal — periphery](https://spinnaker2.gitlab.io/external/documentation/hardware/5-periphery/)
- [Developer portal — 48-node board](https://spinnaker2.gitlab.io/external/documentation/hardware/7-48-node-board/)
- [Developer portal — SpiNNode board](https://spinnaker2.gitlab.io/external/documentation/hardware/10-spinnode-board/)
- [Developer portal — Mamba SSM](https://spinnaker2.gitlab.io/external/about/mamba/)
- [py-spinnaker2 memory-structure doc](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/memory/memory-structure.html)
- [py-spinnaker2 hardware-architecture tutorial](https://spinnaker2.gitlab.io/py-spinnaker2/tutorials/deep_dive/06_hardware_architecture.html)
- [Sandia Lab News, 12 Jun 2025 — NERL Braunfels](https://www.sandia.gov/labnews/2025/06/12/brain-based-computing-for-nd-solutions/)
- [Sandia, 8 May 2024 — SpiNNcloud partnership](https://www.sandia.gov/research/2024/05/08/neuromorphic-computing-for-nuclear-deterrence-solutions-sandia-partners-with-german-startup-spinncloud/)
- [IEEE Spectrum, 8 May 2024](https://spectrum.ieee.org/neuromorphic-computing-spinnaker2)
- [EE Times — Dresden opening](https://www.eetimes.com/spinnaker-based-neuromorphic-supercomputer-opens-in-dresden/)
- [SpiNNcloud SpiNNext product page](https://spinncloud.com/spinnext/)
