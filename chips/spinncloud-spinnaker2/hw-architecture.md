# SpiNNcloud SpiNNaker2 Hardware Architecture

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*
*Representative products: SpiNNaker2 chip · SpiNNode single-chip board · 48-node board*

---

## Overview

SpiNNaker2 is a **102 mm² 22 nm FDSOI manycore** built around 152 ARM Cortex-M4F processing elements, a
400 MHz multicast **event router** at the die centre, and **19.8 MB of entirely software-managed SRAM**.
It is the second generation of the SpiNNaker architecture (Steve Furber, Manchester → TU Dresden
Höppner/Mayr group), commercialized by **SpiNNcloud Systems GmbH**.

Three properties define it against everything else in this survey:

1. **The compute substrate is microcontroller cores.** Each PE is a stock Cortex-M4F with a small
   fixed-function MAC array bolted on as a memory-mapped peripheral. There is no custom ISA.
2. **There is no cache and no shared virtual memory.** Every byte movement is an explicit DMA or NoC
   transaction issued by software, against a memory map planned **statically by the host**.
3. **There is no operating system and no barrier synchronization.** Execution is event-driven: cores sleep
   in WFI until an interrupt arrives.

The designers are explicit that this is not a dense-GEMM machine:

> *"SpiNNaker2 is not a pure DNN inference accelerator… limited DRAM bandwidth often reduces utilization of
> the DNN accelerators if DNN layers are too big to be stored in on-chip memory… scalability of conventional
> DNN workloads on a single chip is limited."*
> — [arXiv:2607.24396](https://arxiv.org/abs/2607.24396), IEEE OJCAS 2026

**Primary source note.** The definitive full-chip paper (Scholze, Partzsch, Höppner et al., IEEE OJCAS 2026,
DOI 10.1109/OJCAS.2026.3714974, posted 27 July 2026, CC-BY 4.0) **supersedes** the 2021/2022 processing-element
paper, which described an **8-PE, 8.76 mm² testchip** rather than the product. Where they disagree the chip
paper is used here and the older revision is flagged.

---

## 1. Compute Engine

### Processing elements

| Component | Specification |
|-----------|---------------|
| PEs per chip | **152**, organized as **38 Quad units (QPEs) × 4 PEs** |
| PE core | **ARM Cortex-M4F** — ARMv7E-M/Thumb-2, DSP extensions, FPv4-SP single-precision FPU |
| Interrupts / timers per PE | 45 interrupt sources; 3 timers on a 1 MHz reference clock |
| Clock | **150 MHz @ 0.5 V** (low power) · **300 MHz @ 0.8 V** (performance) |
| Management core | **1 × Cortex-M4** in the periphery module (FPU and accelerators removed), **100 MHz, 128 kB ECC SRAM**, full access to chip memory/config; boot, background tasks, broadcasting application code to PEs |
| Cores usable by applications | **148 per chip** (4 reserved for system software) → **7,104 of 7,296** on a 48-node board |

The vendor portal's "**153 ARM cores**" = 152 PEs + the periphery management M4. Both conventions are
defensible; state which one is in use.

### Per-PE accelerators (all memory-mapped on AHB/APB — no ISA extensions)

| Accelerator | Specification |
|-------------|---------------|
| **ML accelerator (MLA)** | **16 × 4 output-stationary MAC array = 64 MACs/PE**. 8-bit signed/unsigned; two neighbouring 8-bit cells fuse for **16-bit** row- or column-wise operands; **24-bit accumulator**; 4 post-processing modules; output requantized by shift + truncate to 8/16/32-bit. Ops: 2-D matrix multiply and 2-D convolution, with a shift register for input-feature-map reuse in CONV; operands prefetched directly over the NoC; ReLU fusable with matmul |
| **Numerical accelerator** | Iterative **exp / natural-log**; s16.15 and s0.31 fixed-point **and** FP32, operand and result formats chosen independently; 1–16 configurable iterations = 7–22 cycles; 1–2 ulp, monotonic. **4.4–5.7× faster** than the ARM core for exp/log/sigmoid/tanh/softmax |
| **Rounding accelerator** | Integer rounding + saturation of 64/32/16-bit values; FP→**BFloat16**; round-to-nearest and **stochastic rounding** (PRNG-driven); 4 parallel threads at 3–4 cycles each; configurable rounding position up to 32 LSBs |
| **Random numbers** | **PRNG per PE** (MARS KISS64, 32 bit/cycle) + **one global TRNG** from ADPLL bang-bang jitter, which can scramble the PRNGs |
| **Event handler ("spDMA")** | Two configurable hardware filters plus a default handler; matching SpiNNaker packets are written into an **SRAM-mapped hardware FIFO without interrupting the ARM core**; configurable full-behaviour (drop+count / overwrite / stall NoC); interrupts on non-empty / fill-level / full |

> ⚠ The 2022 prototype paper describes the MAC array as **8-bit unsigned with a 29-bit accumulator**. That is
> the older testchip revision — do not quote it for the product.

### Throughput

| Metric | Value | Condition |
|--------|-------|-----------|
| Total MACs per chip | **9,728** (152 × 64) | structural |
| **Peak INT8** | **5.837 TOPS** theoretical | 300 MHz |
| **Measured INT8** | **4.563 TOPS** (78 % MAC utilization) | 300 MHz / 0.8 V, transformer matmul, full-chip tiling |
| **Measured INT8** | **2.281 TOPS** | 150 MHz / 0.5 V |
| **Energy efficiency** | **2.77 TOPS/W** (0.825 W) | 150 MHz / 0.5 V, operands on-chip |
| **Energy efficiency** | **2.06 TOPS/W** (2.219 W) | 300 MHz / 0.8 V, operands on-chip |
| CoreMark (all 152 PEs) | 65,968 /s @0.5 V/150 MHz · 138,168 /s @0.8 V/300 MHz | — |
| SNN capacity | **>150,000 neurons/chip**; **>1.8 G synaptic events/s** at a 1 ms tick; up to **12 M events/s per core** with 1024 sparse-connectivity neurons | — |
| Any other precision | **Not disclosed** — the MLA is 8/16-bit integer only; FP32 exists on the FPU but no aggregate FLOPS figure is published | — |

The chip paper's own efficiency comparison places 2.77 TOPS/W alongside Coral EdgeTPU (2.00), Mobileye EyeQ5
(2.40), Jetson Orin Nano (2.68), IBM NorthPole (2.70), Groq TSP (2.73), ARM Ethos N77 (5.13).

### Utilization — more important than peak

| Workload | Measured MAC utilization |
|----------|--------------------------|
| Fully-connected layers | **16–19 %** (only 1 of 4 MAC-array rows used) |
| Convolutions | **22–50 %** |
| Transformer matmuls | **78 %** |

Once DRAM is in the loop, memory dominates: DRAM transfers were **71.6–98.0 %** of matrix-multiply layer time
in the OctopuScheduler benchmark. In one MLP case weight fetch was 192 µs of a 323 µs layer while MLA compute
was 29 µs — *"the accelerator is only used for 9 % of the layer runtime."*

---

## 2. Data Path

### DNN mode — scheduler / worker FSM, no host in the loop

```
Host (userspace, UDP over 1 GbE)
  └─ writes the whole DRAM image:
       global config │ per-layer 128-bit headers │ layer params │ data-memory area
  └─ raises ONE interrupt
        │
Scheduler PE ──reads layer header from DRAM──► forwards + IRQs the assigned worker PEs
        │
   Worker PE ──fetches own config
             ──dram_dma_rd(): weights + input tile  DRAM → 128 kB local SRAM
             ──mlacc_params + execute_mm()/execute_conv() ──► __WFE()
             ──dram_dma_wr(): output tile  SRAM → DRAM
             ──sets completion flag ──► WFI sleep
        │
Scheduler PE ──polls completion flags──► advances to the next layer
```

Workers are **asynchronous within a layer** (no intra-layer sync); **layers are synchronous**. Measured
worker/scheduler utilization 80.9–99.6 %; per-layer scheduling overhead ≈13 µs; full-chip setup/cleanup
39 µs / 93 µs.

### SNN / event mode

Neuron models are integrated **in software** on the ARM core by Euler method at a discrete tick (typically
1 ms, from the 1 MHz reference clock). A chip-global interrupt wakes all PEs simultaneously; each tick a PE
reads its spike-FIFO pointers, processes the spikes received in the previous step, updates neurons, emits
multicast packets, and returns to WFI.

**There is no barrier synchronization.** A PE that overruns its tick simply falls out of phase and may catch
up later; incidents are logged and reported. Multi-chip time sync is planned via a per-chip **SCAMP** program
reusing the SpiNNaker1 drift-compensation mechanism.

### DVFS

Each PE is its own switchable power domain with **4 predefined performance levels** (two supply voltages ×
clock frequency, divided locally from a central ADPLL). PEs are GALS — individually asynchronous, or
synchronously coupled within a Quad. Switching is driven by the PE itself or by a global controller; Quad
hardware semaphores throttle simultaneous switching against IR droop. In the DVS-gesture SNN, per-core
per-timestep automatic level selection cut energy **28 %** at identical accuracy (0.741 J vs 1.023 J, 92.04 %).

---

## 3. On-chip Memory — no hardware data cache anywhere

| Level | Capacity | Managed by | Notes |
|-------|----------|-----------|-------|
| Register file | Cortex-M4F architectural + FPv4-SP FP registers | compiler | ARM ISA |
| **PE-local SRAM** | **128 kB per PE** (1 Mbit), 4 addressable banks | **Linker script + host allocator** | ≈**32 kB ITCM** (application/neuron-model code) + ≈**96 kB DTCM** (routing tables, synapses, neuron state, recording buffers, log region); heap+stack 4 kB at `0x1F000` |
| Quad-shared SRAM | 4 × 128 kB = **512 kB per Quad** | software | All 4 PEs can address the full Quad SRAM in synchronous mode, via the PE crossbar |
| **Total on-chip SRAM** | **19.8 MB** (158.625 Mbit) | — | 152 PEs × 1 Mbit + event links 896 kbit + Ethernet 1.875 Mbit + GPIO/mgmt 1 Mbit + event router 2.875 Mbit. The portal's "19 MB" is the same figure rounded |

There is **no cache hierarchy and no shared virtual memory**. The consequence for the software stack is
direct: all data locality is the host mapper's problem, decided ahead of time and baked into a static memory
map.

**Maximum neurons/synapses per PE is deliberately not disclosed** — the chip paper states the number *"varies
significantly with parameters such as weight precision, logging, connectivity and simulation time-step"* and
that studying the trade-offs is out of scope.

---

## 4. Off-chip Memory

| Spec | Value |
|------|-------|
| Interfaces | **2 × LPDDR4**, each addressing up to 4 GB |
| Shipping node configuration | **2 GB per chip** (1 GB per interface) |
| PHY / controller | Uniquify LPDDR4 PHY + controller @ **800 MHz** |
| **Raw bitrate** | **25.6 Gbit/s per interface** → 51.2 Gbit/s ≈ **6.4 GB/s per chip aggregate raw** |
| **Sustained bandwidth** | **Not disclosed** |
| DMA offload | One DMA controller per interface, to keep bulk transfers off the NoC |
| Layout | **Statically partitioned by the host** before the run: global config → per-layer time-measurement area → per-layer config blocks → data-memory area holding all layer inputs/intermediates/outputs |
| DMA API | `dma_rd()`/`dma_wr()` (PE↔PE), `dram_dma_rd()`/`dram_dma_wr()` (SRAM↔DRAM); 16-byte alignment; reads may not cross the interface boundary; `0x0–0x3FF` and `0x40000000–0x400003FF` reserved |

6.4 GB/s raw against 5.837 TOPS peak implies an arithmetic-intensity requirement near **900 ops/byte** — the
structural reason the measured DNN benchmarks are DRAM-dominated rather than MAC-dominated.

---

## 5. On-chip Interconnect

| Fabric | Specification |
|--------|---------------|
| **Data NoC (DNoC)** | 2-D mesh · **192-bit flit** (a whole packet including its 128-bit payload moves in one transaction) · **300 MHz** · **bisection 307.2 Gbit/s** · 5 cycles/hop (GALS async-FIFO crossings) · XY (x-first) routing by default, optional **64-entry (8×8) LUT routing** · round-robin output arbitration · multicast at QPE granularity via 4 destination-PE bits |
| **Config NoC (CNoC)** | 32-bit flit · wormhole switching · **100 MHz reference clock**, so it is alive before any PLL is up · same packet format as DNoC · boot, register access, production test · usable when the DNoC is blocked by large DMAs |
| NoC packet format | 15-bit NoC header + 17-bit packet header + 32-bit address + 0–128-bit payload |
| **Event router** | At the **centre of the die**, occupying two Quads' worth of area · **400 MHz** · 6 NoC ports · input crossbar feeding **three parallel routing engines** · multicast engine is a 4-stage pipeline (TCAM lookup → priority encode → link-destination resolution → 152-core destination lookup, last stage power-gated when no local destination) · **16,384 multicast entries**, each a 32-bit wildcarded source filter · out-of-order issue buffer prevents head-of-line blocking · **peak 1,843.2 Gbit/s to PEs, 460.8 Gbit/s to external links** · 16 diagnostic counters |

### Packet types

| Type | Fields | Purpose |
|------|--------|---------|
| Multicast | 40-bit packet, 32-bit source ID | the spike/event path |
| Core-to-Core | 16-bit chip + 8-bit core address | point-to-point messaging |
| Nearest-Neighbour | 32-bit address | boot, debug |
| Global Read/Write | any address on any chip | host and management access |

All four support 0 / 32 / 64 / 128-bit payloads.

---

## 6. Scale-up Interconnect

Six bidirectional event links per chip place it as a node in a **hexagonal grid**; **48 chips form a toroidal
mesh** on one board. Each link supports two mutually exclusive physical modes:

| Mode | Reach | Rate | Electrical |
|------|-------|------|-----------|
| **Short-range** (on-board) | a few cm | **6 Gbit/s** | Chiplet-like low-swing; eight IO pad cells at 0.5 V; six data lanes, DDR on a 500 MHz differential clock (0.25 V common mode). 96-bit packets, serialization factor 8, two 125 MHz system-clock cycles. **CRC-12** with packet-ID-based resend and ordering. Clock and data driven to ground when idle |
| **Long-range** (board-to-board) | up to 1.5 m | **2 Gbit/s** | Two LVDS pads per link, 1 GHz DDR, 8b10b with clock-data recovery (must stay active to hold sync — longer power-down lead time) |

**Aggregate off-chip event-link bandwidth per chip is not disclosed**, and no board- or rack-level bisection
figure exists.

> The developer portal phrases the link as *"2 Gbps per lane in each direction… combined throughput of
> 12 Gbps bidirectionally"*. Attribute that phrasing to the portal, not to the chip paper.

**There is no PCIe, no RDMA fabric, and no collective-communication library.**

---

## 7. Scale-out Interconnect / Host Interface

- **1 Gbit/s UDP Ethernet per chip** over SGMII (625 MHz DDR), extended with **UDT** (reliable UDP with
  congestion control) plus an optional packet-counter field.
- An **Ethernet-to-NoC bridge** interprets UDP payloads carrying magic headers as NoC packets with on- or
  off-chip destinations, using programmable routing LUTs.
- On a 48-node board **only chip 43** carries the GbE connection; the other 47 are reached through the fabric.
- **27 GPIOs at 1.8 V**: 4 × UART, QSPI master/slave to 200 Mbit/s, I²C master/slave to 1 Mbit/s, 13 PWM
  channels, behind a flexible multiplexer — for direct sensor / robot integration.
- **No kernel-mode driver exists.** The host is a userspace process speaking UDP to a memory-mapped view of
  the chip's register files, SRAM and DRAM.

---

## 8. Physical Specs and Packaging

| Spec | Value |
|------|-------|
| Process | **GlobalFoundries 22FDX (22 nm FDSOI)** + **Racyics ABX** adaptive (forward) body biasing |
| Die area | **102 mm²** |
| Complexity | **266 M gate equivalents**; 10 independent implementation macros in a tile-based flow sized on the Quad |
| Transistor count | **Not disclosed** (gate equivalents ≠ transistors) |
| Supply rails | Two digital voltages only: **0.5 V and 0.8 V**; always-on zero-body-biased 0.8 V domain for comms/config; each PE in its own switchable FBB domain |
| Clock domains | PE 150/300 MHz · DNoC 300 MHz · event router 400 MHz · CNoC + periphery 100 MHz · short-range link 500 MHz diff · LPDDR4 800 MHz · SGMII 625 MHz |
| **Measured power** | PEs off **75.2 mW** · sleeping @0.5 V **235.4 mW** · sleeping @0.8 V **564.7 mW** · all 152 PEs on CoreMark @0.5 V **424.6 mW** · @0.8 V **1,247.9 mW**. Summary range **0.24–2.2 W**; abstract: **baseline below 250 mW** |
| Application power | Mamba SSM (170 M parameters) on one chip: **≈0.6 W average** (vendor portal figure) |
| **Rated TDP** | **Not disclosed** — only measured workload power is published |
| Package type / ball count / dimensions / thermal solution | **Not disclosed** |

### Form factors

| Board | Contents |
|-------|----------|
| **SpiNNode** (1 chip) | STM32H743 manager, ≥2 GB LPDDR4, 1G Ethernet RJ45, USB-C serial CLI, DCMI camera connector, 5 V or 12–24 V barrel input, JTAG, fan header |
| **48-node board** | 48 chips, 2 GB LPDDR4 each, toroidal mesh; on-board STM32 with 2 MB flash boots/manages the chips over an SPI chain + I²C; GbE on chip 43; host via STLink or a Backbone/STM Card over CAN |

Board- and rack-level power draw, cooling method and physical dimensions are **not disclosed**.

---

## 9. Comparison Table (chip paper Tab. 6)

| | SpiNNaker1 | **SpiNNaker2** | Loihi | Loihi 2 |
|---|---|---|---|---|
| Process | 130 nm | **22 nm FDSOI** | 14 nm | 7 nm |
| Clock | 180 MHz | **150 / 300 MHz** | — | — |
| Die area | 102 mm² | **102 mm²** | 60 mm² | 60 mm² |
| On-chip SRAM | 1.8 MB | **19.8 MB** | 33 MB | 31 MB |
| Power | 0.36–1 W | **0.24–2.2 W** | — | — |
| Cores | 18 × ARM M0 | **152 × ARM M4F** | 128 | 128 |
| Neurons | 16,000 | **150,000** | — | — |

### Measured comparison against GPUs (chip paper Tab. 5, INT8, 6 real DNN layers incl. DRAM + scheduling)

| | SpiNNaker2 @150 MHz | Jetson Orin Nano | A100 |
|---|---|---|---|
| Power | **0.58–0.80 W** | 3.1–8.2 W | 107–133 W |
| Energy wins | **4 of 6 layers** (FC1: 0.064 mJ vs 1.332 mJ A100) | — | wins the two largest matmuls (MM2: 4.18 mJ vs 17.07 mJ) |
| Wall clock | **30–700× slower than A100** | — | baseline |

---

## 10. Systems As Built — read the verbs

| System | Scale | Status |
|--------|-------|--------|
| TU Dresden / SpiNNcloud | 5 M cores, ~35,000 chips, eight racks | **Operating**; large-scale software stack "currently under development" |
| Sandia NNSA **NERL Braunfels** | **175 M neurons**, 3 chassis × up to 18 boards, 48 chips/board | **Delivered March 2025**, unboxed 3 April 2025, publicized 12 June 2025; NNSA ASC funded |
| Dresden full build | "16 racks (69,120 chips) … 10.5 billion neurons" | **Target, future tense. Not delivered** |
| Largest *offered* configuration | "10 billion neurons", "0.3 exaops" | **Vendor/press framing of an offered configuration**, not a measured system |
| **SpiNNext** | — | **Announced only** — no process node, no specs, no silicon status, no date |

**Measured end-to-end performance of the Dresden or Sandia systems on any workload is not public.** Every
published SpiNNaker2 measurement is single-chip.

### Vendor claims

"18× more energy efficient than GPUs" (SpiNNaker2) and "78×" (SpiNNext) on spinncloud.com are **unaudited
marketing**; there is **no MLPerf submission** and no independent benchmark. The 18× figure has one traceable
peer-reviewed anchor — EGRU language-model inference at **65 mJ on SpiNNaker2 vs 1.19 J on an A100**, an 18×
energy reduction *for single-batch inference*, **at 8× longer execution time**. The company homepage renders
its headline counters as unpopulated placeholders ("Cores: 0M", "Chips: 0K") and is unusable as a spec source.

---

## Not Disclosed

Rated TDP · package type, ball count, dimensions, thermal solution · transistor count · sustained LPDDR4
bandwidth · aggregate off-chip link bandwidth and board/rack bisection · board and rack power, cooling,
dimensions · price of any product · units shipped, revenue, customers beyond Sandia and TU Dresden · peak
throughput at any precision other than INT8 · hard per-PE neuron/synapse limits · reconciliation of the portal
FAQ's implied 40 QPE coordinate positions with the confirmed 38 QPEs · any MLPerf or audited benchmark ·
everything about SpiNNext.

---

## Sources

- [The SpiNNaker2 chip — IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974](https://arxiv.org/abs/2607.24396)
- [SpiNNaker2: A Large-Scale Neuromorphic System… — NeurIPS 2023 MLNCP](https://arxiv.org/abs/2401.04491)
- [The SpiNNaker 2 Processing Element Architecture](https://arxiv.org/abs/2103.08392)
- [An End-to-End DNN Inference Framework for the SpiNNaker2 MPSoC — ICONS 2025](https://arxiv.org/abs/2507.13736)
- [Developer portal — ML accelerator](https://spinnaker2.gitlab.io/external/documentation/hardware/1-ml-accelerator/)
- [Developer portal — chip topology](https://spinnaker2.gitlab.io/external/documentation/hardware/2-s2-chip-topology/)
- [Developer portal — NoC](https://spinnaker2.gitlab.io/external/documentation/hardware/11-NoC/)
- [Developer portal — SpiNNaker router](https://spinnaker2.gitlab.io/external/documentation/hardware/8-spinnaker-router/)
- [Developer portal — chip-to-chip link](https://spinnaker2.gitlab.io/external/documentation/hardware/2-chip-to-chip-link/)
- [Developer portal — DRAM](https://spinnaker2.gitlab.io/external/documentation/hardware/4-dram/) · [DMA](https://spinnaker2.gitlab.io/external/documentation/hardware/7-dma/) · [periphery](https://spinnaker2.gitlab.io/external/documentation/hardware/5-periphery/)
- [Developer portal — 48-node board](https://spinnaker2.gitlab.io/external/documentation/hardware/7-48-node-board/) · [SpiNNode board](https://spinnaker2.gitlab.io/external/documentation/hardware/10-spinnode-board/)
- [py-spinnaker2 memory-structure doc](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/memory/memory-structure.html)
- [py-spinnaker2 hardware-architecture tutorial](https://spinnaker2.gitlab.io/py-spinnaker2/tutorials/deep_dive/06_hardware_architecture.html)
- [Sandia Lab News, 12 Jun 2025 — NERL Braunfels](https://www.sandia.gov/labnews/2025/06/12/brain-based-computing-for-nd-solutions/)
- [IEEE Spectrum, 8 May 2024](https://spectrum.ieee.org/neuromorphic-computing-spinnaker2)
- [EE Times — Dresden opening](https://www.eetimes.com/spinnaker-based-neuromorphic-supercomputer-opens-in-dresden/)
- [SpiNNcloud SpiNNext product page](https://spinncloud.com/spinnext/)
