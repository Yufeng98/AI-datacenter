# SpiNNcloud SpiNNaker2 Software Stack Investigation

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*

---

## Overview

**The headline for this survey: there is no tensor compiler.**

No MLIR/LLVM lowering path, no XLA/StableHLO, no Triton, no TorchDynamo backend, no device tensor type, no
collective-communication library. Each PE runs a **hand-written, pre-compiled bare-metal C program** built
with the stock **GNU Arm Embedded (GCC) toolchain** and custom linker scripts, statically placed on PEs by a
host-side mapper, with a scheduler-PE finite state machine coordinating workers. Framework reach is achieved
through **graph import** — ONNX for DNNs, NIR for SNNs — rather than through a native compiler. That
"manycore with no tensor compiler" model is a genuinely distinct data point against every compiler-centric
stack in this registry.

The chip paper states the stack in three tiers:

> *"1) Chip software: Bare metal ARM programs to run on the SpiNNaker2 PEs are written in C and compiled
> using a GCC toolchain with custom linker scripts. C-library functions are available for on-chip
> communication and control of hardware units such as accelerators. 2) A C++-based low-level host software
> provides memory-mapped access via UDP to the chip's register files, SRAM, and DRAM… 3) Python-based
> high-level software for SNN and DNN applications."*
> — [arXiv:2607.24396](https://arxiv.org/abs/2607.24396) §III.E

There are two largely disjoint toolflows on top of that base: an **SNN flow** (py-spinnaker2 + NIR +
PACMAN2-style partition/place/route) and a **DNN flow** (PyTorch → ONNX → AMD Quark INT8 → OctopuScheduler →
on-chip scheduler/worker FSM).

---

## Layer 1: Framework Integration

| Component | Function | License | Source |
|---|---|---|---|
| **PyTorch** | Training front-end for the DNN path (the OctopuScheduler flow starts from a trained PyTorch model) and for deep SNNs before NIR export | upstream OSS | [arXiv:2507.13736](https://arxiv.org/abs/2507.13736) |
| **NIR** (Neuromorphic Intermediate Representation) | The interoperability layer for SNNs. py-spinnaker2 imports NIR graphs exported from **snnTorch, Norse, Sinabs, Rockpool, Nengo, Lava-DL, Spyx** | OSS (neuromorphs/NIR); Nature Comms 15:8122 (2024) | [NIR user guide](https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/qs-nir.html) |
| **PyNN-inspired API** (not PyNN itself) | Populations / Projections / Network model in py-spinnaker2 | Apache-2.0 | [py-spinnaker2](https://gitlab.com/spinnaker2/py-spinnaker2) |
| **Brian2 backend** | Software simulation backend that quantizes weights and delays **bit-identically to hardware**, so models can be developed without a board | Apache-2.0 | py-spinnaker2 `brian2_sim.py` |
| **snntorch-to-spinnaker2** | Standalone converter repo (created July 2022, 30 commits) | Apache-2.0 | [gitlab](https://gitlab.com/spinnaker2/snntorch-to-spinnaker2) |
| **NESTML** | Announced as in progress: *"We are currently working on NESTML support"* | roadmap | chip paper §V |

**Absent:** no TensorFlow/JAX production path, no vLLM, no HuggingFace serving integration, no
`torch.compile` backend.

---

## Layer 2: Graph Capture

### DNN path

Trained PyTorch model → **ONNX** export → post-training quantization → lowering. There is **no FX/Dynamo
tracing**; ONNX is the capture boundary. ([arXiv:2507.13736](https://arxiv.org/abs/2507.13736) §IV.A)

### SNN path

**NIR graph import** via `spinnaker2.s2_nir.from_nir()`, driven by a `ConversionConfig` (timestep `dt`,
weight rescaling to the chip's dynamic range, Forward vs Exponential-Euler integration, reset method,
recording spec). Node mapping:

| NIR node | py-spinnaker2 target |
|---|---|
| `IF` | `lif` (leak disabled) |
| `LIF` | `lif` |
| `CubaLIF` | `lif_curr_exp` |
| `Input` | `spike_list` |
| `Linear` | `Projection` |
| `Affine` | `Projection` + `i_offset` |
| `Conv2d` | `Conv2dProjection` |
| `SumPool2d` | `Projection` / `Conv2dProjection` |
| `Flatten` | merged into the projection |

### Tooling

**model-explorer** — Google's model-graph visualizer/debugger mirrored into the `spinnaker2` namespace
(Apache-2.0, created June 2026, 614 commits). No SpiNNaker2-specific additions are evident in its README;
treat as **tooling, not a compiler component** (*inferred*).

---

## Layer 3: Graph Compiler

| Component | What it does | Status |
|---|---|---|
| **OctopuScheduler** | The DNN graph compiler + on-chip scheduler. Lowers the quantized ONNX graph into an **"application graph"** (a simplified IR: layers with parameters and quantization scales as nodes, tensors as edges), applies **layer fusion** (Linear+ReLU fused into a single MLA operation; redundant Quantize/Dequantize pairs emitted by the quantizer removed and their scales folded into neighbours), topologically sorts into a linear chain, then converts each node into a low-level **`S2Layer`** instance. `S2Layer` implements tiling, PE mapping, memory allocation and layer-config generation, and carries a reference model per layer type. **`S2Model`** assembles the whole DRAM image (global config, per-layer 128-bit headers with layer type / worker count / next-layer address, scheduler config, layer parameters, shared worker config, per-PE input/output tile addresses, constants) and plans every activation location. Automated design-space exploration picks the tiling | Published: Langer et al., **NICE 2025** pp. 1–10; multi-layer extension Jobst et al., **ICONS 2025** / [arXiv:2507.13736](https://arxiv.org/abs/2507.13736). **No standalone public repo located**; parts live in `spinnaker2-ml`. Chip paper: *"Only a limited number of layer types and operators is currently supported."* The full supported-op list is **not disclosed** |
| **AMD Quark** | Post-training quantization: **INT8 power-of-two** quantization of the ONNX graph, with MSE optimization and **cross-layer equalization** on a small calibration set | open source, [github.com/amd/Quark](https://github.com/amd/Quark) |
| **py-spinnaker2 mapping pipeline** (`src/spinnaker2/graph/`) | The SNN-side compiler: `partitioner.py`, `placer.py`, `placement_strategy.py`, `pe_config_generator.py`, `pipeline_stages.py`, `pipeline_factory.py`, plus **two selectable pipelines** — `legacy_mapping_pipeline.py` and `pacman_mapping_pipeline.py`. `snapshot.py` adds mapping save/load (new in v0.8.0). Populations are split into sub-populations that fit a PE's SRAM; sub-populations + projections form a machine graph from which the routing table is generated | Apache-2.0 |
| **PACMAN2** | **P**artitioning **A**nd **C**onfiguration **MAN**ager — *"utilities for partitioning, placing and routing on a SpiNNaker machine"*, with a token-based algorithm workflow executor. README text is identical to Manchester's SpiNNaker1 PACMAN and the repo carries 5,016 commits since Sept 2024 — clearly the SpiNNaker1 host toolchain being carried forward to S2. GitLab reports it as a standalone repo rather than a fork, so call the lineage **inferred** | ECL-2.0, public, active to June 2026 |
| **MicroTVM (Apache TVM)** | Research-stage compiler path. Internal docs describe a two-pronged strategy: **single-core** — microTVM AoT targeting the M4F + 16×4 MAC array, where *"CMSIS-NN kernels outperform generic TVM schedules, making them preferred"*; **multi-core** — a custom graph partitioner splitting the model into PE-sized regions each compiled independently through single-core microTVM, because *"microTVM lacks mature parallel pipeline support"*. Used in production for the robot demo's GRU gesture classifier | [TVM workflow doc](https://spinnaker2.gitlab.io/internal/research/13-tvm-workflow.internal/); Liu et al., CODAI Workshop 2024, pp. 37–40 |
| **Serial/parallel paradigm selector** | An AdaBoost classifier (91.69 % accuracy over 12 candidates) predicts *before compilation* whether an SNN layer should use the serial or MAC-array-parallel inference paradigm, cutting host compile time and memory | [arXiv:2406.17049](https://arxiv.org/abs/2406.17049) |

---

## Layer 4: Kernel Compiler

- **GNU Arm Embedded GCC** cross-compiler with **custom linker scripts** (sections `.myDataSpecSection`,
  `.myLogErrorSection`; heap/stack at `0x1F000`, 4 kB).
- Build system: `make export` in the SDK's chip-side app folder historically; **py-spinnaker2 v0.8.0
  (31 July 2026) migrated to CMake**, exposing an `snn_add_application()` helper that injects compile
  definitions (`N_NEURONS`, `SYNAPSE_INDEX_BITS`, `SYNAPSE_WEIGHT_BITS`, `SYNAPSE_DELAY_BITS`, data/log
  addresses).
- **ARM CMSIS / CMSIS-NN** is the reference CPU kernel library (the chip paper's ARM-core baseline; also the
  preferred single-PE codegen target for the TVM path).
- **No autotuner, no polyhedral scheduler, no JIT.** Everything is ahead-of-time.

---

## Layer 5: Kernel Language

- **Plain C, bare metal.** No DSL, no intrinsic-based tile language.
- **`ml-lib`** (shipped inside `s2-sim2lab-app`) is the MLA C API: populate a `struct mlacc_params`, call
  `execute_mm()` or `execute_conv()`, then `__WFE()` until the MLA completion interrupt. Software currently
  only reaches each core's **local** MLA, though the hardware supports remote access.
- **Custom neuron models** are C: `lib/<model>/neuron_model_impl.c` (state equations,
  `neuron_model_has_spiked()` reset logic), `neuron.c` (init + timestep), `include/<model>/regions.h`
  (region enum that must stay in lockstep with the Python class), and two mandatory structs `neuron_t` /
  `neuron_params_t`. Profiling via `monitor.h` and `PROFILE_SCOPE()` macros. The docs warn *"the neuron
  model code is currently undergoing substantial restructuring."*
- Accelerators are **memory-mapped** into the PE address space behind an AHB slave mux — programmed by MMIO,
  not by ISA extensions.

---

## Layer 6: Tensor API

**There is no device tensor type** — no analogue of `MTIATensor`, `bm_device_mem_t`, or a CUDA tensor.
Host-side data is NumPy; on-device data is raw memory regions written by the host into a statically planned
map.

Python-level abstractions live in **py-spinnaker2** (`src/spinnaker2/`):

| Module | Contents |
|---|---|
| `snn/` | `Population`, `Projection`, `Network` |
| `neuron_models/` | 22 modules: `lif_neuron`, `lif_curr_exp`, `lif_conv2d`, `lif_adaptive_threshold`, `lif_prob`, `izhikevich`, `charge_and_spike`, `qubo_neuron`, `gas`, `poisson_process`, `spike_list`, `spike_receiver`, `spike_replicator`, `relay`, `conv2d_if_neuron_rate_code`, `spikes_from_array_latency_code`, … |
| `mla/` | `conv2d.py`, `mla_config.py`, `mla_helpers.py` — direct MLA configuration from Python |
| others | `synapses.py`, `s2_nir.py`, `hardware.py`, `brian2_sim.py`, `coordinates.py` |

Hardware handles: **`SpiNNaker2Chip`** (single chip, 148 usable cores) and **`SpiNNcloud48NodeBoard`**
(7,104 usable cores), resolved against `/etc/opt/spinnaker/spinnaker2_network_config.yml`.

---

## Layer 7: Runtime

### On-chip — no operating system

Bare-metal C. For SNNs, a timer-interrupt `timer_callback()` main loop: memory setup → spike reception via
master-population-table lookup → ring-buffer synaptic update → neuron state update → spike emission, with
WFI/WFE sleep between ticks and a chip-global interrupt for start-of-simulation. **No barrier
synchronization.** For DNNs, the OctopuScheduler scheduler-PE FSM drives up to 151 workers entirely on-chip.
**SCAMP** (name inherited from SpiNNaker1) is the per-chip program planned to trigger the chip-global
interrupt and compensate inter-board clock drift in multi-chip runs.

### Host side

| Component | Role | License |
|---|---|---|
| **ExperimentRunner** (C++) | The workhorse: reads `spec.json` (`active_pes` by coordinate + ID, `mem_files` per PE, `duration_in_s`, `mem_data_to_send`, `mem_regions_to_read`), loads binaries and data to the chip over UDP Ethernet, runs, retrieves memory blocks, writes `results.json` (raw 32-bit words). Built automatically during py-spinnaker2 install; needs CMake ≥ 3.10 | ships in `s2-sim2lab-app` (gated); invoked by Apache-2.0 py-spinnaker2 |
| **py-spinnaker2 experiment backends** | `cpp_experiment_backend.py` (the C++ runner) and `spinnman2_experiment_backend.py` + `spinnman2_interface.py` (the newer SpiNNMan2 path), behind `experiment_backend_factory.py` | Apache-2.0 |
| **SpiNNMan2** | *"Utilities for interacting with a SpiNNaker machine"* — host communication library covering both SpiNNaker1 and SpiNNaker2; depends on `spinnaker2-stm` → SpiNNUtils2 → SpiNNMachine2 | ECL-2.0, public, 3,091 commits, active Aug 2026 |
| **SpiNNMachine2 / SpiNNUtils2** | Machine model / configuration manager (SpiNNaker1 lineage) | ECL-2.0, public |
| **spinnaker2-stm** | Python interface to the board's STM32 microcontroller | public |
| **Power runtime** | `hw.run(record_power=True)` yields `power.parquet` (per-channel timestamped samples) + `power.png` with `load_start` / `sim_start` / `sim_end` / `readout_start` phase markers; `live_power=True` serves a browser dashboard. Readout cadence 0.1 s single-chip, 0.2 s on 48-node boards | Apache-2.0, new in v0.8.0 |
| **Data streaming** | Bidirectional real-time spike streaming to/from a running simulation | Apache-2.0 |

---

## Layer 8: Driver / Firmware

**There is no kernel-mode device driver.** No PCIe, no `/dev` node, no ioctl layer. The host is a userspace
process speaking **UDP (+UDT) over Ethernet** to a memory-mapped view of the chip's register files, SRAM and
DRAM. This is architecturally unusual for a datacenter accelerator and worth calling out explicitly.

| Component | Role | Status |
|---|---|---|
| **Digilent Adept2 Runtime 2.27.9** | Installed by `libs_install.sh` (x86_64 and aarch64, Debian/RPM) for JTAG/USB-side access | third-party |
| **`spinnaker2_os`** | The on-chip low-level software: **SARK**, **SCAMP** and **Spin2API** (names inherited from SpiNNaker1) | **HTTP 403 — access-gated** |
| **`s2-sim2lab-app`** | The board SDK: ARM-C libraries (incl. `ml-lib`), host-side communication utilities, example applications, Dockerfile | **HTTP 403 — access-gated**; access granted on request. Public Apache-2.0 SNN-only export exists as `s2-sim2lab-app-snn` (created 28 May 2025) |
| **`s2_cpp_host_interface`** | C++ interface for the 48-node board | gated |
| **`s2-stm-software` / `s2_board_firmware`** | STM32 firmware for single-chip boards (SpiNNode) and for the 48-node board STM32 + Backbone Card | gated |
| Boot chain | STM32 boots the SpiNNaker2 chip over SPI, then the chip is driven over UDP Ethernet. Two-stage on-chip boot (JTAG/I²C/SPI slave or autonomous SPI-flash, then high-speed application load, optionally broadcast to all PEs by the periphery M4) | confirmed |
| **`mercury-xu8-bsp`** | Board support package for the Enclustra Mercury XU8 (Zynq UltraScale+ MPSoC SoM); present in the org, consistent with an FPGA-based backbone/host card in larger systems. README not retrievable — role **inferred** | public repo, purpose unconfirmed |
| **`spinnaker2-config-tools`** | Sysadmin helper scripts for `/etc/opt/spinnaker/spinnaker2_network_config.yml` (machine names, board types, board counts, lockfiles, user groups) | public |
| **`mlops-infrastructure`** | Present in the org, README returns 403 | gated, purpose not disclosed |

---

## Layer 9: ISA

- **ARMv7E-M / Thumb-2** with DSP extensions and the **FPv4-SP single-precision FPU** (Cortex-M4F).
  Standard, off-the-shelf, fully supported by upstream GCC/LLVM — **there is no custom ISA and no vendor
  compiler backend to maintain**.
- The periphery management core is a Cortex-M4 **without** FPU or accelerators.
- All acceleration (MLA, exp/log, rounding, PRNG, DMA, NoC interface, event handler) is exposed as
  **memory-mapped peripherals on AHB/APB**, controlled by MMIO writes and 45 interrupt lines — not as ISA
  instructions.

**Consequence:** the "kernel compiler" problem reduces to ordinary embedded-ARM C compilation plus explicit
MMIO driver calls. That is why the stack has no vendor codegen backend and no autotuner, and why the
compiler-shaped work (tiling, placement, memory planning) lives entirely in host-side Python.

---

## Open Source vs Proprietary — Summary

**Public, Apache-2.0:** `py-spinnaker2` (the flagship SDK; 340+ commits since 14 June 2022, v0.8.0 released
31 July 2026, Zenodo-archived), `spinnaker2-ml`, `snntorch-to-spinnaker2`, `s2-sim2lab-app-snn`,
`spinnaker2-config-tools`, the `spinnaker2.gitlab.io` docs, the `model-explorer` mirror,
`spinncloud_docs_theme`, and the paper-reproduction repo `tud-hpsn/public/s2-chip-paper`.

**Public, ECL-2.0** (SpiNNaker1-lineage host toolchain being carried to S2): `PACMAN2`, `SpiNNMan2`,
`SpiNNMachine2`, `SpiNNUtils2` — all created 2024 and actively developed through mid-2026.

**Access-gated (HTTP 403):** `s2-sim2lab-app` (the full board SDK containing `ml-lib` and the C++ host
interface), `spinnaker2_os` (SARK/SCAMP/Spin2API), `mlops-infrastructure`. The `snn-core` submodule that
py-spinnaker2 pulls into `src/spinnaker2/libs` is referenced over SSH; the v0.8.0 changelog states chip-side
code was *"open sourced"* in that release, so the gating on the chip-side SNN kernels **appears to be
lifting** as of July 2026 — but no public HTTPS project page resolved, so treat the status as **unconfirmed**.

**Explicitly proprietary:** SpiNNcloud's large-scale-system stack. Two primary sources say so directly —
*"integrating the edge-based approach presented here into the **proprietary stack** being developed at
SpiNNcloud for large-scale systems"* ([arXiv:2507.13736](https://arxiv.org/abs/2507.13736) §VI) and *"The
software stack for large-scale deployment on this system is currently under development"* (chip paper §V).
Its name, architecture and licensing are **not disclosed**.

**Practical consequence:** the open stack is **single-chip and 48-node-board scoped**. The 35k-chip Dresden
machine and Sandia's NERL Braunfels run on software that is not public.

---

## Open Items / Not Disclosed

- Whether **OctopuScheduler** is or will be released as open source — described in two peer-reviewed papers,
  no standalone public repository located
- Public status of the **`snn-core`** submodule (chip-side SNN kernels)
- Complete list of DNN layer types and operators supported by OctopuScheduler
- Contents and purpose of `s2-sim2lab-app`, `spinnaker2_os`, `s2_cpp_host_interface`, `s2-stm-software`,
  `s2_board_firmware`, `mlops-infrastructure` beyond their one-line portal descriptions
- The exact role of `mercury-xu8-bsp` in SpiNNcloud systems
- The name, architecture and licensing of SpiNNcloud's proprietary large-scale-system stack

---

## Sources

- [The SpiNNaker2 chip (IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974)](https://arxiv.org/abs/2607.24396)
- [An End-to-End DNN Inference Framework for the SpiNNaker2 MPSoC (ICONS 2025)](https://arxiv.org/abs/2507.13736)
- [SpiNNaker2: A Large-Scale Neuromorphic System… (NeurIPS 2023 MLNCP)](https://arxiv.org/abs/2401.04491)
- [Fast Switching Serial and Parallel Paradigms of SNN Inference on SpiNNaker2](https://arxiv.org/abs/2406.17049)
- [Efficient Deployment of SNNs on SpiNNaker2 for DVS Gesture Recognition Using NIR](https://arxiv.org/abs/2504.06748)
- [Event-based backpropagation on the neuromorphic platform SpiNNaker2](https://arxiv.org/abs/2412.15021)
- [NIR: Neuromorphic Intermediate Representation (Nature Comms 15:8122)](https://doi.org/10.1038/s41467-024-52259-9)
- [py-spinnaker2 GitLab](https://gitlab.com/spinnaker2/py-spinnaker2)
- [py-spinnaker2 documentation](https://spinnaker2.gitlab.io/py-spinnaker2/)
- [py-spinnaker2 NIR user guide](https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/qs-nir.html)
- [py-spinnaker2 Experiment Runner doc](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/experiment-runner.html)
- [py-spinnaker2 memory-structure doc](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/memory/memory-structure.html)
- [py-spinnaker2 chip-side execution walkthrough](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/execution_workflow/run-walkthrough-2.html)
- [py-spinnaker2 custom neuron model (hardware side)](https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/custom_model/custom-model-guide-hw.html)
- [py-spinnaker2 power-measurement doc](https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/power_measurement.html)
- [Developer portal — software component index](https://spinnaker2.gitlab.io/external/documentation/software/)
- [Developer portal — ML accelerator / ml-lib](https://spinnaker2.gitlab.io/external/documentation/hardware/1-ml-accelerator/)
- [Developer portal — building the first application](https://spinnaker2.gitlab.io/external/documentation/hardware/1-building-first-app/)
- [Developer portal — TVM workflow (internal research page)](https://spinnaker2.gitlab.io/internal/research/13-tvm-workflow.internal/)
- [PACMAN2](https://gitlab.com/spinnaker2/PACMAN2) · [SpiNNMan2](https://gitlab.com/spinnaker2/SpiNNMan2) · [SpiNNMachine2](https://gitlab.com/spinnaker2/SpiNNMachine2) · [SpiNNUtils2](https://gitlab.com/spinnaker2/SpiNNUtils2)
- [spinnaker2-ml](https://gitlab.com/spinnaker2/spinnaker2-ml) · [s2-sim2lab-app-snn](https://gitlab.com/spinnaker2/s2-sim2lab-app-snn) · [snntorch-to-spinnaker2](https://gitlab.com/spinnaker2/snntorch-to-spinnaker2)
- [AMD Quark](https://github.com/amd/Quark)
