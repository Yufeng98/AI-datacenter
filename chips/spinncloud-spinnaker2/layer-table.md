# SpiNNcloud SpiNNaker2 Layer Mapping Table

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch — training front-end for the DNN path (OctopuScheduler starts from a trained PyTorch model) and for deep SNNs before NIR export | confirmed | arxiv-2507.13736 |
| Framework Integration | **NIR** (Neuromorphic Intermediate Representation) — the SNN interoperability layer; py-spinnaker2 imports NIR graphs from snnTorch, Norse, Sinabs, Rockpool, Nengo, Lava-DL, Spyx | confirmed | py-s2-nir-guide, nature-comms-nir |
| Framework Integration | PyNN-**inspired** API (not PyNN itself) — Populations / Projections / Network in py-spinnaker2 | confirmed | py-spinnaker2-repo |
| Framework Integration | Brian2 backend (`brian2_sim.py`) — software simulation that quantizes weights and delays **bit-identically to hardware**, enabling boardless development | confirmed | py-spinnaker2-repo |
| Framework Integration | snntorch-to-spinnaker2 — standalone snnTorch converter (Apache-2.0, created July 2022, 30 commits) | confirmed | gitlab-snntorch-conv |
| Framework Integration | NESTML support — announced in progress ("We are currently working on NESTML support") | confirmed | chip-paper |
| Framework Integration | **Absent**: no TensorFlow/JAX production path, no vLLM, no HuggingFace serving, no `torch.compile` backend | confirmed | chip-paper, py-spinnaker2-repo |
| Graph Capture | DNN: trained PyTorch → **ONNX** export → PTQ → lowering. **No FX/Dynamo tracing** — ONNX is the capture boundary | confirmed | arxiv-2507.13736 |
| Graph Capture | SNN: `spinnaker2.s2_nir.from_nir()` driven by `ConversionConfig` (dt, weight rescaling, Forward vs Exponential-Euler, reset method, recording). Mapping: IF→lif, LIF→lif, CubaLIF→lif_curr_exp, Input→spike_list, Linear→Projection, Affine→Projection+i_offset, Conv2d→Conv2dProjection, SumPool2d→Projection/Conv2dProjection, Flatten merged | confirmed | py-s2-nir-guide |
| Graph Capture | model-explorer — Google Model Explorer mirrored into the `spinnaker2` namespace (Apache-2.0, June 2026, 614 commits); no S2-specific additions evident — tooling, not a compiler component | unconfirmed | gitlab-model-explorer |
| Graph Compiler | **OctopuScheduler** — DNN graph compiler + on-chip scheduler; quantized ONNX → "application graph" IR (layers+quant scales as nodes, tensors as edges) → layer fusion (Linear+ReLU into one MLA op; Quantize/Dequantize pairs folded) → topological linear chain → `S2Layer` (tiling, PE mapping, memory allocation, layer-config gen, per-type reference model) → `S2Model` (whole DRAM image: global config, 128-bit per-layer headers, scheduler config, params, per-PE tile addresses). Automated DSE picks the tiling | confirmed | nice-2025-octopu, arxiv-2507.13736 |
| Graph Compiler | OctopuScheduler operator coverage — chip paper: *"Only a limited number of layer types and operators is currently supported."* Full list **not disclosed**. No standalone public repo located | confirmed | chip-paper |
| Graph Compiler | **AMD Quark** — INT8 power-of-two PTQ of the ONNX graph, MSE optimization + cross-layer equalization on a small calibration set | confirmed | arxiv-2507.13736, github-quark |
| Graph Compiler | py-spinnaker2 mapping pipeline (`src/spinnaker2/graph/`) — partitioner, placer, placement_strategy, pe_config_generator, pipeline_stages, pipeline_factory, snapshot (v0.8.0); two selectable pipelines: `legacy_mapping_pipeline` and `pacman_mapping_pipeline`. Populations split into SRAM-sized sub-populations; machine graph → routing table | confirmed | py-spinnaker2-repo |
| Graph Compiler | **PACMAN2** (ECL-2.0, 5,016 commits since Sept 2024) — "utilities for partitioning, placing and routing on a SpiNNaker machine", token-based algorithm workflow executor. README identical to Manchester SpiNNaker1 PACMAN; GitLab reports standalone repo, so lineage is inferred | likely | gitlab-pacman2 |
| Graph Compiler | **MicroTVM** (research stage) — single-core: microTVM AoT to M4F + 16×4 MAC array, but *"CMSIS-NN kernels outperform generic TVM schedules, making them preferred"*; multi-core: custom partitioner splits the model into PE-sized regions because *"microTVM lacks mature parallel pipeline support"*. Production use: robot-demo GRU gesture classifier | confirmed | s2-tvm-workflow, codai-2024 |
| Graph Compiler | Serial/parallel paradigm selector — AdaBoost classifier (91.69 % accuracy over 12 candidates) picks serial vs MAC-array-parallel SNN inference paradigm before compilation | confirmed | arxiv-2406.17049 |
| Kernel Compiler | **GNU Arm Embedded GCC** (arm-none-eabi) + **custom linker scripts** (`.myDataSpecSection`, `.myLogErrorSection`; heap/stack 4 kB at `0x1F000`) | confirmed | chip-paper, py-s2-memory-doc |
| Kernel Compiler | Build system: historically `make export` in the SDK chip-side app folder; **py-spinnaker2 v0.8.0 (31 Jul 2026) migrated to CMake** with `snn_add_application()` injecting `N_NEURONS`, `SYNAPSE_INDEX_BITS`, `SYNAPSE_WEIGHT_BITS`, `SYNAPSE_DELAY_BITS`, data/log addresses | confirmed | py-s2-custom-model-hw |
| Kernel Compiler | ARM **CMSIS / CMSIS-NN** as the reference CPU kernel library (also the preferred single-PE codegen target for the TVM path) | confirmed | chip-paper, s2-tvm-workflow |
| Kernel Compiler | **No autotuner, no polyhedral scheduler, no JIT** — everything is ahead-of-time | confirmed | chip-paper |
| Kernel Language | **Plain C, bare metal.** No DSL, no intrinsic-based tile language | confirmed | chip-paper |
| Kernel Language | `ml-lib` (inside gated `s2-sim2lab-app`) — MLA C API: fill `struct mlacc_params` → `execute_mm()` / `execute_conv()` → `__WFE()` until the MLA completion interrupt. Software currently reaches only each core's **local** MLA though hardware supports remote access | confirmed | s2-mla-doc |
| Kernel Language | Custom neuron models in C: `lib/<model>/neuron_model_impl.c` (state equations, `neuron_model_has_spiked()`), `neuron.c` (init + timestep), `include/<model>/regions.h` (region enum in lockstep with the Python class), mandatory `neuron_t` / `neuron_params_t`. Profiling via `monitor.h` + `PROFILE_SCOPE()`. Docs warn the neuron-model code is *"undergoing substantial restructuring"* | confirmed | py-s2-custom-model-hw |
| Kernel Language | Accelerators are **memory-mapped behind an AHB slave mux** — programmed by MMIO, not by ISA extensions | confirmed | chip-paper |
| Tensor API | **No device tensor type** (no analogue of `MTIATensor` or a CUDA tensor). Host-side data is NumPy; on-device data is raw memory regions written into a statically planned map | confirmed | py-spinnaker2-repo |
| Tensor API | py-spinnaker2 modules: `snn/` (Population, Projection, Network); `neuron_models/` (22 modules incl. lif_neuron, lif_curr_exp, lif_conv2d, lif_adaptive_threshold, lif_prob, izhikevich, charge_and_spike, qubo_neuron, gas, poisson_process, spike_list, spike_receiver, spike_replicator, relay); `mla/` (conv2d.py, mla_config.py, mla_helpers.py); `synapses.py`, `s2_nir.py`, `hardware.py`, `brian2_sim.py`, `coordinates.py` | confirmed | py-spinnaker2-repo |
| Tensor API | Hardware handles `SpiNNaker2Chip` (148 usable cores) and `SpiNNcloud48NodeBoard` (7,104 usable cores), resolved against `/etc/opt/spinnaker/spinnaker2_network_config.yml` | confirmed | py-spinnaker2-docs |
| Runtime (on chip) | **No operating system.** Bare-metal C. SNN: timer-interrupt `timer_callback()` loop — memory setup → spike reception via master-population-table lookup → ring-buffer synaptic update → neuron state update → spike emission; WFI/WFE sleep between ticks; chip-global interrupt starts the simulation. **No barrier synchronization** | confirmed | py-s2-run-walkthrough, chip-paper |
| Runtime (on chip) | DNN: OctopuScheduler scheduler-PE FSM drives up to **151 worker PEs** entirely on-chip after a single host interrupt; workers async within a layer, layers synchronous; 80.9–99.6 % utilization, ≈13 µs per-layer overhead, 39 µs setup / 93 µs cleanup | confirmed | chip-paper, arxiv-2507.13736 |
| Runtime (on chip) | **SCAMP** (name inherited from SpiNNaker1) — per-chip program planned to trigger the chip-global interrupt and compensate inter-board clock drift in multi-chip runs | confirmed | chip-paper |
| Runtime (host) | **ExperimentRunner (C++)** — reads `spec.json` (`active_pes`, `mem_files`, `duration_in_s`, `mem_data_to_send`, `mem_regions_to_read`), loads binaries/data over UDP Ethernet, runs, retrieves memory, writes `results.json` (raw 32-bit words). Built during py-spinnaker2 install; CMake ≥3.10 | confirmed | py-s2-experiment-runner |
| Runtime (host) | py-spinnaker2 experiment backends: `cpp_experiment_backend.py` and `spinnman2_experiment_backend.py` + `spinnman2_interface.py`, behind `experiment_backend_factory.py` | confirmed | py-spinnaker2-repo |
| Runtime (host) | **SpiNNMan2** (ECL-2.0, 3,091 commits, active Aug 2026) — host↔machine communication for SpiNNaker1 and 2; depends on spinnaker2-stm → SpiNNUtils2 → SpiNNMachine2 | confirmed | gitlab-spinnman2 |
| Runtime (host) | SpiNNMachine2 / SpiNNUtils2 (ECL-2.0) — machine model and configuration manager, SpiNNaker1 lineage | confirmed | gitlab-spinnmachine2 |
| Runtime (host) | spinnaker2-stm — Python interface to the board's STM32 microcontroller | confirmed | gitlab-s2-stm |
| Runtime (host) | Power runtime (v0.8.0): `hw.run(record_power=True)` → `power.parquet` + `power.png` with load_start / sim_start / sim_end / readout_start markers; `live_power=True` browser dashboard; 0.1 s single-chip / 0.2 s 48-node cadence | confirmed | py-s2-power-doc |
| Runtime (host) | Bidirectional real-time spike streaming to/from a running simulation | confirmed | py-spinnaker2-docs |
| Driver / Firmware | **No kernel-mode device driver. No PCIe, no `/dev` node, no ioctl layer.** The host is a userspace process speaking **UDP (+UDT) over 1 GbE** to a memory-mapped view of the chip's registers, SRAM and DRAM | confirmed | chip-paper, py-s2-experiment-runner |
| Driver / Firmware | `spinnaker2_os` — on-chip low-level software: **SARK**, **SCAMP**, **Spin2API** (names inherited from SpiNNaker1). Repository returns **HTTP 403 — access-gated** | confirmed | gitlab-s2-os |
| Driver / Firmware | `s2-sim2lab-app` — the board SDK (ARM-C libs incl. `ml-lib`, host comms utilities, examples, Dockerfile). **HTTP 403 — access-gated**, granted on request. Public Apache-2.0 SNN-only export: `s2-sim2lab-app-snn` (28 May 2025) | confirmed | gitlab-s2-sim2lab, gitlab-s2-sim2lab-snn |
| Driver / Firmware | `s2_cpp_host_interface` (48-node board C++ interface), `s2-stm-software` (single-chip STM32 firmware), `s2_board_firmware` (48-node STM32 + Backbone Card) — all gated | confirmed | s2-software-index |
| Driver / Firmware | Boot chain: STM32 boots the SpiNNaker2 chip over SPI, then the chip is driven over UDP Ethernet. Two-stage on-chip boot (JTAG/I²C/SPI slave or autonomous SPI-flash → high-speed application load, optionally broadcast to all PEs by the periphery M4) | confirmed | chip-paper |
| Driver / Firmware | Digilent **Adept2 Runtime 2.27.9** installed by `libs_install.sh` (x86_64 + aarch64, Debian/RPM) for JTAG/USB-side access | confirmed | py-spinnaker2-repo |
| Driver / Firmware | `mercury-xu8-bsp` — Enclustra Mercury XU8 (Zynq UltraScale+ MPSoC SoM) BSP; consistent with an FPGA backbone/host card in larger systems, but README not retrievable — role **inferred** | unconfirmed | gitlab-mercury-xu8 |
| Driver / Firmware | `spinnaker2-config-tools` — sysadmin scripts for `/etc/opt/spinnaker/spinnaker2_network_config.yml` (machine names, board types, board counts, lockfiles, user groups) | confirmed | gitlab-s2-config-tools |
| Driver / Firmware | `mlops-infrastructure` — present in the org, README returns 403; purpose **not disclosed** | unconfirmed | gitlab-mlops-infra |
| ISA | **ARMv7E-M / Thumb-2** with DSP extensions + **FPv4-SP** single-precision FPU (Cortex-M4F). Standard, off-the-shelf, fully supported by upstream GCC/LLVM — **no custom ISA, no vendor compiler backend** | confirmed | chip-paper |
| ISA | Periphery management core is a Cortex-M4 **without** FPU or accelerators | confirmed | chip-paper, s2-periphery-doc |
| ISA | All acceleration (MLA, exp/log, rounding, PRNG, DMA, NoC interface, event handler) exposed as **memory-mapped AHB/APB peripherals** + 45 interrupt lines — not ISA instructions. Consequence: kernel compilation reduces to ordinary embedded-ARM C plus MMIO driver calls | confirmed | chip-paper |
| Communication | **No collective-communication library** (no NCCL/RCCL/HCCL equivalent) and **no RDMA**. Multi-chip communication is the SpiNNaker multicast event router plus 6 chip-to-chip links; the host attaches over Ethernet | confirmed | chip-paper |
| Openness | Public Apache-2.0: py-spinnaker2 (flagship SDK, v0.8.0 31 Jul 2026, Zenodo-archived), spinnaker2-ml, snntorch-to-spinnaker2, s2-sim2lab-app-snn, spinnaker2-config-tools, docs portal, model-explorer mirror, tud-hpsn/public/s2-chip-paper | confirmed | gitlab-group-api |
| Openness | Public ECL-2.0 (SpiNNaker1-lineage host toolchain carried to S2): PACMAN2, SpiNNMan2, SpiNNMachine2, SpiNNUtils2 — all created 2024, active through mid-2026 | confirmed | gitlab-group-api |
| Openness | `snn-core` (chip-side SNN kernels) pulled as py-spinnaker2 submodule `src/spinnaker2/libs` over SSH; v0.8.0 changelog says chip-side code was "open sourced" but no public HTTPS page resolved — status **unclear** | unconfirmed | py-spinnaker2-repo |
| Openness | **Explicitly proprietary**: SpiNNcloud's large-scale-system stack — *"the proprietary stack being developed at SpiNNcloud for large-scale systems"* (arXiv:2507.13736 §VI) and *"currently under development"* (chip paper §V). Name, architecture and licensing **not disclosed**. The open stack is single-chip / 48-node-board scoped; Dresden and NERL Braunfels run non-public software | confirmed | arxiv-2507.13736, chip-paper |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | **152 ARM Cortex-M4F PEs** per chip, as 38 QuadPEs × 4; 45 interrupt sources and 3 timers per PE on a 1 MHz reference clock; 150 MHz @0.5 V / 300 MHz @0.8 V | confirmed | chip-paper, arxiv-2401.04491 |
| Compute Engine | **1 periphery Cortex-M4 management core** (no FPU, no accelerators), 100 MHz, 128 kB ECC SRAM — reconciles the vendor portal's "153 ARM cores" with the papers' "152 PEs" | confirmed | chip-paper, s2-periphery-doc |
| Compute Engine | **148 application-visible cores per chip** (4 reserved for system software) → 7,104 of 7,296 on a 48-node board | confirmed | py-s2-hw-tutorial |
| Compute Engine | **MLA per PE**: 16×4 output-stationary MAC array = **64 MACs/PE**; 8-bit signed/unsigned, adjacent cells fuse for 16-bit; **24-bit accumulator**; 4 post-processing modules; shift+truncate requantization to 8/16/32-bit; MM and CONV ops with IFM-reuse shift register; NoC operand prefetch; ReLU fusable. (2022 prototype paper's 8-bit unsigned / 29-bit accumulator is the **older testchip**) | confirmed | chip-paper, s2-mla-doc |
| Compute Engine | Numerical accelerator per PE: iterative exp/ln; s16.15, s0.31 and FP32 with independent operand/result formats; 1–16 iterations = 7–22 cycles; 1–2 ulp, monotonic; **4.4–5.7× faster than the ARM core** for exp/log/sigmoid/tanh/softmax | confirmed | chip-paper |
| Compute Engine | Rounding accelerator per PE: integer rounding + saturation (64/32/16-bit), FP→**BFloat16**, round-to-nearest and **stochastic rounding**, 4 threads at 3–4 cycles, configurable position up to 32 LSBs | confirmed | chip-paper |
| Compute Engine | RNG: **PRNG per PE** (MARS KISS64, 32 bit/cycle) + one global **TRNG** from ADPLL bang-bang jitter that can scramble the PRNGs | confirmed | chip-paper |
| Compute Engine | Event handler ("spDMA") per PE: two configurable hardware filters + default handler; matching packets land in an SRAM-mapped hardware FIFO **without interrupting the ARM core**; configurable full-behaviour and fill-level interrupts | confirmed | chip-paper |
| Compute Engine | **9,728 8-bit MACs per chip**; **5.837 TOPS INT8 theoretical**; **4.563 TOPS measured** @300 MHz/0.8 V (78 % utilization); **2.281 TOPS** @150 MHz/0.5 V | confirmed | chip-paper |
| Compute Engine | **2.77 TOPS/W** @150 MHz/0.5 V (0.825 W); **2.06 TOPS/W** @300 MHz/0.8 V (2.219 W), operands resident in on-chip SRAM | confirmed | chip-paper |
| Compute Engine | CoreMark: 65,968/s (all 152 PEs @0.5 V/150 MHz); 138,168/s (@0.8 V/300 MHz) | confirmed | chip-paper |
| Compute Engine | SNN capacity: **>150,000 neurons/chip**, **>1.8 G synaptic events/s** at a 1 ms tick, up to 12 M events/s per core with 1024 sparse-connectivity neurons | confirmed | chip-paper |
| Compute Engine | Peak throughput at any precision other than INT8: **not disclosed** (MLA is 8/16-bit integer only; FP32 exists on the FPU but no aggregate FLOPS figure published) | confirmed | chip-paper |
| Compute Engine | Measured MAC utilization: **16–19 %** fully-connected, **22–50 %** convolution, **78 %** transformer matmul. DRAM = **71.6–98.0 %** of matmul layer time; MLP case — *"the accelerator is only used for 9 % of the layer runtime"* | confirmed | chip-paper, arxiv-2507.13736 |
| Data Path | DNN: scheduler-PE FSM drives up to 151 workers; host writes the whole DRAM image and raises **one** interrupt; workers async within a layer, layers synchronous | confirmed | chip-paper, arxiv-2507.13736 |
| Data Path | SNN: software Euler integration on the ARM core at a 1 ms tick; chip-global interrupt wakes all PEs; **no barrier synchronization** — an overrunning PE falls out of phase and is logged | confirmed | chip-paper |
| Data Path | **DVFS**: each PE its own switchable power domain, **4 performance levels**, GALS clocking from a central ADPLL, Quad hardware semaphores throttle simultaneous switching against IR droop; measured **28 % energy saving** from per-core per-timestep level selection on DVS-gesture SNN | confirmed | chip-paper |
| On-chip Memory | **128 kB SRAM per PE** (1 Mbit) in 4 banks; **software/linker managed** — ≈32 kB ITCM (code) + ≈96 kB DTCM (routing tables, synapses, neuron state, recording buffers, log region); heap+stack 4 kB at `0x1F000` | confirmed | chip-paper, py-s2-memory-doc |
| On-chip Memory | Quad-shared SRAM 4×128 kB = **512 kB**; all 4 PEs address the full Quad SRAM in synchronous mode via the PE crossbar | confirmed | chip-paper |
| On-chip Memory | **19.8 MB total on-chip SRAM** (158.625 Mbit): 152 PEs × 1 Mbit + event links 896 kbit + Ethernet 1.875 Mbit + GPIO/mgmt 1 Mbit + event router 2.875 Mbit. Portal's "19 MB" is this rounded | confirmed | chip-paper |
| On-chip Memory | **No hardware data cache, no shared virtual memory.** Every byte movement is an explicit software-issued DMA or NoC transaction; the memory map is planned **statically by the host** before execution | confirmed | chip-paper, arxiv-2401.04491 |
| On-chip Memory | Hard per-PE neuron/synapse limits: **not disclosed** — chip paper explicitly declines, citing dependence on weight precision, logging, connectivity and time-step | confirmed | chip-paper |
| Off-chip Memory | **2 × LPDDR4** interfaces (each addressing up to 4 GB); shipping node = **2 GB per chip**; Uniquify PHY + controller @800 MHz; **25.6 Gbit/s raw per interface = 6.4 GB/s aggregate raw**; per-interface DMA controller | confirmed | chip-paper, s2-dram-doc |
| Off-chip Memory | **Sustained (effective) LPDDR4 bandwidth: not disclosed** | confirmed | chip-paper |
| Off-chip Memory | DRAM layout statically partitioned by the host: global config → per-layer time-measurement area → per-layer config blocks → data-memory area for all layer inputs/intermediates/outputs | confirmed | arxiv-2507.13736 |
| Off-chip Memory | DMA API: `dma_rd`/`dma_wr` (PE↔PE), `dram_dma_rd`/`dram_dma_wr` (SRAM↔DRAM); 16-byte alignment; reads may not cross the interface boundary; `0x0–0x3FF` and `0x40000000–0x400003FF` reserved | confirmed | s2-dma-doc |
| On-chip Interconnect | **DNoC** — 2-D mesh, **192-bit flit** (whole packet in one transaction), 300 MHz, **bisection 307.2 Gbit/s**, 5 cycles/hop, XY routing by default with optional 64-entry (8×8) LUT routing, round-robin arbitration, QPE-granularity multicast via 4 destination-PE bits | confirmed | chip-paper, s2-noc-doc |
| On-chip Interconnect | **CNoC** — 32-bit flit, wormhole, 100 MHz reference clock (alive before any PLL), same packet format; boot/register access/production test; usable when the DNoC is blocked by large DMAs | confirmed | chip-paper |
| On-chip Interconnect | NoC packet format: 15-bit NoC header + 17-bit packet header + 32-bit address + 0–128-bit payload | confirmed | arxiv-2103.08392 |
| On-chip Interconnect | **Event router** at die centre (area of two Quads), **400 MHz**, 6 NoC ports, input crossbar → **three parallel routing engines**; multicast engine 4-stage pipeline (TCAM lookup → priority encode → link-destination resolution → 152-core destination lookup, last stage power-gated); **16,384 multicast entries** of 32-bit wildcarded source filters; out-of-order issue buffer; **1,843.2 Gbit/s to PEs, 460.8 Gbit/s to external links**; 16 diagnostic counters | confirmed | chip-paper, s2-router-doc |
| On-chip Interconnect | Packet types: Multicast (40-bit, 32-bit source ID — the spike path), Core-to-Core (16-bit chip + 8-bit core), Nearest-Neighbour (32-bit address, boot/debug), Global Read/Write (any address on any chip). All support 0/32/64/128-bit payloads | confirmed | chip-paper, s2-router-doc |
| Scale-up Interconnect | **6 bidirectional event links per chip**, hexagonal grid; **48 chips = one toroidal mesh** board | confirmed | chip-paper, s2-c2c-doc |
| Scale-up Interconnect | Short-range (on-board, cm): 0.5 V low-swing chiplet-like interface, 8 IO pad cells, 6 data lanes DDR on a 500 MHz differential clock (0.25 V common mode) = **6 Gbit/s**; 96-bit packets, serialization 8, two 125 MHz cycles; **CRC-12** with packet-ID resend and ordering; clock/data grounded when idle | confirmed | chip-paper |
| Scale-up Interconnect | Long-range (board-to-board, ≤1.5 m): 2 LVDS pads per link, 1 GHz DDR = **2 Gbit/s**, 8b10b with CDR; must stay active to hold sync | confirmed | chip-paper |
| Scale-up Interconnect | **Aggregate off-chip event-link bandwidth per chip: not disclosed.** Portal alternate phrasing ("2 Gbps per lane… 12 Gbps bidirectionally") should be attributed to the portal, not the paper | confirmed | s2-c2c-doc |
| Scale-up Interconnect | **No PCIe, no RDMA fabric, no collective-communication library** | confirmed | chip-paper |
| Scale-out Interconnect | **1 Gbit/s UDP Ethernet per chip** over SGMII (625 MHz DDR) with **UDT** reliability + optional packet counter; Ethernet-to-NoC bridge turns magic-headered UDP payloads into on-/off-chip NoC packets via programmable LUTs | confirmed | chip-paper |
| Scale-out Interconnect | On a 48-node board only **chip 43** carries the GbE connection; the other 47 are reached through the fabric | confirmed | s2-48node-doc |
| Scale-out Interconnect | 27 GPIOs @1.8 V: 4× UART, QSPI master/slave to 200 Mbit/s, I²C master/slave to 1 Mbit/s, 13 PWM channels, behind a flexible multiplexer (sensor/robot integration) | confirmed | chip-paper |
| Physical / Package | **GlobalFoundries 22FDX (22 nm FDSOI)** with **Racyics ABX** adaptive forward body biasing; **die 102 mm²**; **266 M gate equivalents**; 10 implementation macros in a tile-based flow | confirmed | chip-paper, arxiv-2103.08392 |
| Physical / Package | Two digital supply rails only (**0.5 V and 0.8 V**); always-on zero-body-biased 0.8 V domain for comms/config; each PE in its own switchable FBB domain | confirmed | chip-paper |
| Physical / Package | Measured power: PEs off **75.2 mW**, sleeping @0.5 V **235.4 mW**, @0.8 V **564.7 mW**, CoreMark all-PEs @0.5 V **424.6 mW**, @0.8 V **1,247.9 mW**; summary range **0.24–2.2 W**; abstract "baseline below 250 mW" | confirmed | chip-paper |
| Physical / Package | Single-chip application power: Mamba SSM (170 M params) ≈**0.6 W** average (vendor portal figure) | likely | s2-mamba-page |
| Physical / Package | **Rated TDP: not disclosed.** Transistor count, package type, ball/pin count, package dimensions and thermal solution: **not disclosed** | confirmed | chip-paper |
| Physical / Package | Form factors: **SpiNNode** (1 chip; STM32H743, ≥2 GB LPDDR4, 1G Ethernet, USB-C CLI, DCMI camera, 5 V or 12–24 V, JTAG, fan header) and **48-node board** (48 chips, 2 GB each, torus, STM32 + 2 MB flash over SPI chain + I²C, GbE on chip 43, STLink or Backbone/STM Card over CAN) | confirmed | s2-spinnode-doc, s2-48node-doc |
| Physical / Package | Board- and rack-level power, cooling method and dimensions: **not disclosed** | confirmed | — |
| Deployment | TU Dresden / SpiNNcloud: **5 M cores, ~35,000 chips, eight racks** — operating; large-scale software stack "currently under development" | confirmed | chip-paper, arxiv-2401.04491 |
| Deployment | Sandia NNSA **NERL Braunfels**: **175 M neurons**, 3 chassis × up to 18 boards, 48 chips/board — **delivered March 2025**, unboxed 3 April 2025, publicized 12 June 2025, NNSA ASC funded | confirmed | sandia-labnews-2025 |
| Deployment | Dresden full build "16 racks (69,120 chips) … 10.5 billion neurons" — **target, future tense, not delivered** | confirmed | eetimes-dresden |
| Deployment | "10 billion neurons" / "0.3 exaops" — largest commercially **offered** configuration per press, **not a measured deployed system** | likely | ieee-spectrum-2024 |
| Deployment | **SpiNNext** — announced only; no process node, no specs, no silicon status, no date | confirmed | spinncloud-spinnext |
| Deployment | Measured end-to-end performance of the Dresden or Sandia systems on any workload: **not disclosed** — all published measurements are single-chip | confirmed | — |
| Benchmarks | **No MLPerf submission and no independently audited benchmark exists** | confirmed | — |
| Benchmarks | Vendor claims "18×" (SpiNNaker2) and "78×" (SpiNNext) vs GPUs are unaudited marketing. The 18× has one peer-reviewed anchor: EGRU LM inference **65 mJ vs 1.19 J on A100** — single-batch, **8× longer execution time** | confirmed | chip-paper, spinncloud-home |
| Benchmarks | Measured GPU comparison (chip paper Tab. 5, INT8, 6 layers incl. DRAM + scheduling): SpiNNaker2 **0.58–0.80 W** vs A100 **107–133 W** and Jetson Orin Nano **3.1–8.2 W**; energy win on **4 of 6** layers; loses the two largest matmuls; **30–700× slower wall-clock** | confirmed | chip-paper |
| Open Question | Developer-portal FAQ implies **40** QuadPE coordinate positions (x=1..7, y=1..6 minus (4,3) and (4,4)) against the confirmed **38 QPEs**. The portal does not explain the discrepancy | unconfirmed | s2-faq |

## Source Keys

| Key | URL |
|-----|-----|
| chip-paper | https://arxiv.org/abs/2607.24396 |
| arxiv-2401.04491 | https://arxiv.org/abs/2401.04491 |
| arxiv-2103.08392 | https://arxiv.org/abs/2103.08392 |
| arxiv-2507.13736 | https://arxiv.org/abs/2507.13736 |
| arxiv-2406.17049 | https://arxiv.org/abs/2406.17049 |
| nice-2025-octopu | Langer et al., OctopuScheduler, NICE 2025, pp. 1–10 (no preprint located) |
| codai-2024 | Liu et al., CODAI Workshop 2024, pp. 37–40 (cited as ref [54] of arXiv:2607.24396) |
| nature-comms-nir | https://doi.org/10.1038/s41467-024-52259-9 |
| github-quark | https://github.com/amd/Quark |
| py-spinnaker2-repo | https://gitlab.com/spinnaker2/py-spinnaker2 |
| py-spinnaker2-docs | https://spinnaker2.gitlab.io/py-spinnaker2/ |
| py-s2-nir-guide | https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/qs-nir.html |
| py-s2-experiment-runner | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/experiment-runner.html |
| py-s2-memory-doc | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/memory/memory-structure.html |
| py-s2-run-walkthrough | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/execution_workflow/run-walkthrough-2.html |
| py-s2-custom-model-hw | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/custom_model/custom-model-guide-hw.html |
| py-s2-hw-tutorial | https://spinnaker2.gitlab.io/py-spinnaker2/tutorials/deep_dive/06_hardware_architecture.html |
| py-s2-power-doc | https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/power_measurement.html |
| s2-software-index | https://spinnaker2.gitlab.io/external/documentation/software/ |
| s2-mla-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/1-ml-accelerator/ |
| s2-noc-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/11-NoC/ |
| s2-router-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/8-spinnaker-router/ |
| s2-c2c-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/2-chip-to-chip-link/ |
| s2-dram-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/4-dram/ |
| s2-dma-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/7-dma/ |
| s2-periphery-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/5-periphery/ |
| s2-48node-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/7-48-node-board/ |
| s2-spinnode-doc | https://spinnaker2.gitlab.io/external/documentation/hardware/10-spinnode-board/ |
| s2-tvm-workflow | https://spinnaker2.gitlab.io/internal/research/13-tvm-workflow.internal/ |
| s2-mamba-page | https://spinnaker2.gitlab.io/external/about/mamba/ |
| s2-faq | https://spinnaker2.gitlab.io/external/faq/ |
| gitlab-group-api | https://gitlab.com/api/v4/groups/spinnaker2/projects |
| gitlab-pacman2 | https://gitlab.com/spinnaker2/PACMAN2 |
| gitlab-spinnman2 | https://gitlab.com/spinnaker2/SpiNNMan2 |
| gitlab-spinnmachine2 | https://gitlab.com/spinnaker2/SpiNNMachine2 |
| gitlab-s2-stm | https://gitlab.com/spinnaker2/spinnaker2-stm |
| gitlab-s2-config-tools | https://gitlab.com/spinnaker2/spinnaker2-config-tools |
| gitlab-snntorch-conv | https://gitlab.com/spinnaker2/snntorch-to-spinnaker2 |
| gitlab-model-explorer | https://gitlab.com/spinnaker2/model-explorer |
| gitlab-mercury-xu8 | https://gitlab.com/spinnaker2/mercury-xu8-bsp |
| gitlab-s2-os | https://gitlab.com/spinnaker2/spinnaker2_os |
| gitlab-s2-sim2lab | https://gitlab.com/spinnaker2/s2-sim2lab-app |
| gitlab-s2-sim2lab-snn | https://gitlab.com/spinnaker2/s2-sim2lab-app-snn |
| gitlab-mlops-infra | https://gitlab.com/spinnaker2/mlops-infrastructure |
| sandia-labnews-2025 | https://www.sandia.gov/labnews/2025/06/12/brain-based-computing-for-nd-solutions/ |
| ieee-spectrum-2024 | https://spectrum.ieee.org/neuromorphic-computing-spinnaker2 |
| eetimes-dresden | https://www.eetimes.com/spinnaker-based-neuromorphic-supercomputer-opens-in-dresden/ |
| spinncloud-home | https://spinncloud.com/ |
| spinncloud-spinnext | https://spinncloud.com/spinnext/ |
