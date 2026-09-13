# SpiNNcloud SpiNNaker2 — Search Results

*as_of: 2026-08-08*
*chip: spinncloud-spinnaker2*
*device_class: Event-Driven Neuromorphic Manycore (152 ARM Cortex-M4F PEs + per-PE MAC/2D-conv accelerators)*

---

## Retrieval Note

The WebSearch budget for this session was exhausted (200/200) before this chip was researched. All
retrieval was therefore done by **direct WebFetch against primary-source URLs** plus the **arXiv REST
API** and the **GitLab REST API** (`https://gitlab.com/api/v4/groups/spinnaker2/projects`). The list
below records the query set that drove that retrieval, not a set of executed search-engine calls.

---

## Search Queries

1. "SpiNNaker2 chip architecture 152 PE QuadPE NoC arXiv"
2. arXiv API: `ti:SpiNNaker2` sorted by submittedDate — surfaced **arXiv:2607.24396**, the July 2026 IEEE OJCAS full-chip paper
3. arXiv API: `abs:SpiNNaker2` — full publication list for the platform
4. "SpiNNaker2 die size process node 22FDX GlobalFoundries area mm2"
5. "SpiNNaker2 peak TOPS INT8 MAC array 16x4 throughput TOPS/W"
6. "SpiNNaker2 LPDDR4 bandwidth DRAM interface Uniquify 800 MHz"
7. "SpiNNaker2 NoC DNoC CNoC flit bisection bandwidth event router"
8. "SpiNNaker2 chip-to-chip link SerDes LVDS 8b10b hexagonal torus bandwidth"
9. "SpiNNaker2 baseline power TDP watts measured CoreMark"
10. "SpiNNaker2 153 ARM cores vs 152 PEs periphery management processor"
11. "SpiNNaker2 ITCM DTCM 128 kB SRAM per PE memory map software managed cache"
12. "py-spinnaker2 GitLab README license neuron models NIR Brian2 backend"
13. "py-spinnaker2 experiment runner spec.json C++ ExperimentRunner Ethernet"
14. "py-spinnaker2 mapping pipeline partitioner placer PACMAN2"
15. GitLab API: `groups/spinnaker2/projects` — enumerate every public repo in the org
16. "spinnaker2_os SARK SCAMP Spin2API repository"
17. "s2-sim2lab-app SDK ml-lib execute_mm execute_conv mlacc_params"
18. "OctopuScheduler SpiNNaker2 on-chip DNN scheduling NICE 2025"
19. "SpiNNaker2 end-to-end DNN inference PyTorch ONNX AMD Quark INT8 quantization"
20. "SpiNNaker2 microTVM TVM compiler CMSIS-NN kernels partitioner"
21. "SpiNNaker2 NIR snnTorch Norse Sinabs Rockpool Lava-DL Spyx supported node types"
22. "SpiNNaker2 custom neuron model C neuron_model_impl.c CMake gcc arm-none-eabi"
23. "SpiNNaker2 SCAMP multi-chip synchronization drift compensation"
24. "SpiNNcloud proprietary software stack large-scale systems"
25. "Sandia NERL Braunfels SpiNNaker2 175 million neurons NNSA ASC deployment date"
26. "SpiNNcloud Dresden 35000 chips 5 million cores eight racks"
27. "SpiNNcloud SpiNNext specs process node availability"
28. "SpiNNaker2 Mamba state space model 370M parameters 48 chips power"
29. "SpiNNaker2 language modeling EGRU AICAS 2024 energy vs A100"
30. "SpiNNaker2 DVS gesture NIR quantization accuracy 94.13"
31. "SpiNNaker2 vs Loihi Loihi2 comparison table neurons per chip"
32. "SpiNNaker2 48-node board STM32 Ethernet backbone CAN chip 43"
33. "SpiNNode single chip board specifications power form factor"
34. "spinncloud.com sitemap pages leaflet updates specs"

---

## Resources Found

### Primary Papers — Hardware

| Resource | URL | Category |
|----------|-----|----------|
| **The SpiNNaker2 chip: a many-core platform for flexible and scalable brain-inspired computing** (Scholze, Partzsch, Höppner et al., IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974, posted 27 Jul 2026, CC-BY 4.0) — **the definitive full-chip source** | https://arxiv.org/abs/2607.24396 | Hardware Spec |
| SpiNNaker2: A Large-Scale Neuromorphic System for Event-Based and Asynchronous Machine Learning (Gonzalez et al., NeurIPS 2023 MLNCP workshop) | https://arxiv.org/abs/2401.04491 | Hardware Spec |
| The SpiNNaker 2 Processing Element Architecture for Hybrid Digital Neuromorphic Computing (Höppner, Yan, Vogginger et al., arXiv v2 Aug 2022) — 22FDX **testchip** (8.76 mm², 8 PEs), QPE/NoC, ABB, GALS | https://arxiv.org/abs/2103.08392 | Hardware Spec |
| Neuromorphic computing at scale (Kudithipudi et al., Nature 637:801–812, 2025) — cited by the chip paper for the 5M-core Dresden figure | https://doi.org/10.1038/s41586-024-08253-8 | Context |

### Primary Papers — Software / Compiler

| Resource | URL | Category |
|----------|-----|----------|
| An End-to-End DNN Inference Framework for the SpiNNaker2 Neuromorphic MPSoC (Jobst, Langer, Liu, Alici, Gonzalez, Mayr; ICONS 2025) | https://arxiv.org/abs/2507.13736 | Compiler |
| OctopuScheduler: On-Chip DNN Scheduling on the SpiNNaker2 Neuromorphic MPSoC (Langer, Jobst, Liu, Kelber, Vogginger, Mayr; NICE 2025, pp. 1–10) | (conference proceedings; no arXiv preprint located) | Compiler |
| Deploying ML Models to Ahead-of-Time Runtime on Edge Using MicroTVM (Liu, Jobst, Guo, Shi, Partzsch, Mayr; CODAI Workshop 2024, pp. 37–40) — the TVM path | (workshop proceedings; cited as ref [54] of arXiv:2607.24396) | Compiler |
| Fast Switching Serial and Parallel Paradigms of SNN Inference on SpiNNaker2 (Huang et al.) — AdaBoost paradigm selector, 91.69 % accuracy | https://arxiv.org/abs/2406.17049 | Compiler heuristic |
| NIR: Neuromorphic Intermediate Representation (Pedersen et al., Nature Comms 15:8122, 2024) | https://doi.org/10.1038/s41467-024-52259-9 | IR |

### Primary Papers — Applications / Toolflow

| Resource | URL | Category |
|----------|-----|----------|
| Efficient Deployment of SNNs on SpiNNaker2 for DVS Gesture Recognition Using NIR (Arfa et al., NICE 2025) — PTQ + QAT, 94.13 % on-chip | https://arxiv.org/abs/2504.06748 | Application |
| Event-based backpropagation on the neuromorphic platform SpiNNaker2 (Béna, Wunderlich, Akl, Vogginger, Mayr, Gonzalez; NICE 2025) | https://arxiv.org/abs/2412.15021 | On-chip training |
| Hardware-Aware Fine-Tuning of Spiking Q-Networks on SpiNNaker2 (Arfa, Vogginger, Mayr) | https://arxiv.org/abs/2507.23562 | Application |
| Language Modeling on a SpiNNaker2 Neuromorphic Chip (Nazeer et al., AICAS 2024, DOI 10.1109/AICAS59952.2024.10595870) — EGRU | https://arxiv.org/abs/2312.09084 | Application |
| Efficient SNN multi-cores MAC array acceleration on SpiNNaker2 (Huang et al., Front. Neurosci. 17, 2023) | https://doi.org/10.3389/fnins.2023.1223262 | Application |
| E-prop on SpiNNaker2 (Rostami, Vogginger, Yan, Mayr; Front. Neurosci. 16, 2022) | https://doi.org/10.3389/fnins.2022.1018006 | On-chip training |

### Vendor Documentation — Developer Portal

| Resource | URL | Category |
|----------|-----|----------|
| SpiNNaker2 Developer Portal (root) — "153 ARM cores, 19MB on-chip SRAM, 2GB DRAM" | https://spinnaker2.gitlab.io/ | Overview |
| Software component index | https://spinnaker2.gitlab.io/external/documentation/software/ | Software |
| Chip topology (38 QPEs, dual NoC, DNoC 192-bit, XY routing, dual LPDDR4) | https://spinnaker2.gitlab.io/external/documentation/hardware/2-s2-chip-topology/ | Hardware Spec |
| ML accelerator (16×4 MAC, 64 MAC/cycle, `ml-lib`, `mlacc_params`, `execute_mm`/`execute_conv`) | https://spinnaker2.gitlab.io/external/documentation/hardware/1-ml-accelerator/ | Hardware Spec |
| NoC (CNoC/DNoC, XY vs 64-entry LUT routing, packet headers) | https://spinnaker2.gitlab.io/external/documentation/hardware/11-NoC/ | Hardware Spec |
| SpiNNaker router (4 packet types, 16,384 MC entries, TCR/TKR/TDR registers) | https://spinnaker2.gitlab.io/external/documentation/hardware/8-spinnaker-router/ | Hardware Spec |
| Chip-to-chip links (6 links, triangular mesh, 48-chip torus) | https://spinnaker2.gitlab.io/external/documentation/hardware/2-chip-to-chip-link/ | Hardware Spec |
| DRAM (2×1 GB LPDDR4, `dram_dma_rd`/`_wr`, reserved ranges) | https://spinnaker2.gitlab.io/external/documentation/hardware/4-dram/ | Hardware Spec |
| DMA (`dma_rd`/`dma_wr` inter-PE, DRAM DMA, MEM A/B) | https://spinnaker2.gitlab.io/external/documentation/hardware/7-dma/ | Hardware Spec |
| Periphery block (the "153rd" ARM M4, 100 MHz, 128 kB ECC SRAM, boot controller, ABB generator) | https://spinnaker2.gitlab.io/external/documentation/hardware/5-periphery/ | Hardware Spec |
| 48-node board (48 chips, STM32 + 2 MB flash, SPI chain, I²C, GbE on chip 43, CAN backbone) | https://spinnaker2.gitlab.io/external/documentation/hardware/7-48-node-board/ | Hardware Spec |
| SpiNNode single-chip board (STM32H743, ≥2 GB LPDDR4, 1G Ethernet, USB-C, DCMI, 5 V / 12–24 V) | https://spinnaker2.gitlab.io/external/documentation/hardware/10-spinnode-board/ | Hardware Spec |
| Building the first application (SDK access, `make export`, `make run RUNETHIP=…`) | https://spinnaker2.gitlab.io/external/documentation/hardware/1-building-first-app/ | Software |
| **TVM workflow (internal research page)** — microTVM single-core + custom multi-core partitioner; CMSIS-NN beats generic TVM schedules | https://spinnaker2.gitlab.io/internal/research/13-tvm-workflow.internal/ | Compiler |
| Mamba SSM application — 370 M params over 7,000+ cores on 48 chips; 170 M on one chip at ≈0.6 W | https://spinnaker2.gitlab.io/external/about/mamba/ | Application |
| QUBO / MaxCut solver application | https://spinnaker2.gitlab.io/external/about/qubo/ | Application |
| Research index (6 publications with DOIs) | https://spinnaker2.gitlab.io/external/research/research/ | Overview |
| Access requirements — "we do not sell SpiNNaker2 hardware to individuals"; cloud access not yet available | https://spinnaker2.gitlab.io/external/get_started/requirements/ | Overview |
| FAQ (QuadPE coordinate ranges, DRAM reserved addresses) | https://spinnaker2.gitlab.io/external/faq/ | Overview |

### py-spinnaker2 SDK Documentation

| Resource | URL | Category |
|----------|-----|----------|
| py-spinnaker2 documentation root — user guide, 10 tutorials, examples | https://spinnaker2.gitlab.io/py-spinnaker2/ | SDK Docs |
| NIR user guide — supported frameworks and node-type mapping table | https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/qs-nir.html | SDK Docs |
| Experiment Runner — `spec.json` / `results.json` / C++ binary over Ethernet | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/experiment-runner.html | Runtime Docs |
| Memory structure — ITCM ≈32 kB / DTCM ≈96 kB, host-side allocation | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/memory/memory-structure.html | Hardware Spec |
| Chip-side execution walkthrough — `timer_callback` loop, no barrier sync | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/execution_workflow/run-walkthrough-2.html | Runtime Docs |
| Custom neuron model (hardware side) — C file layout, CMake `snn_add_application()` | https://spinnaker2.gitlab.io/py-spinnaker2/advanced_topic/custom_model/custom-model-guide-hw.html | Kernel Docs |
| Hardware-architecture tutorial — 148 application cores/chip, 7,104/board | https://spinnaker2.gitlab.io/py-spinnaker2/tutorials/deep_dive/06_hardware_architecture.html | Hardware Spec |
| Power-measurement doc — `record_power`, parquet trace, live dashboard | https://spinnaker2.gitlab.io/py-spinnaker2/user_guide/power_measurement.html | Runtime Docs |

### Open-Source GitLab Repositories

| Resource | URL | Category |
|----------|-----|----------|
| **py-spinnaker2** — flagship Python SDK; Apache-2.0; 340+ commits since 14 Jun 2022; v0.8.0 released 31 Jul 2026 | https://gitlab.com/spinnaker2/py-spinnaker2 | SDK |
| PACMAN2 — partitioning / placing / routing; ECL-2.0; 5,016 commits | https://gitlab.com/spinnaker2/PACMAN2 | Mapper |
| SpiNNMan2 — host↔machine communication library; ECL-2.0; 3,091 commits | https://gitlab.com/spinnaker2/SpiNNMan2 | Runtime |
| SpiNNMachine2 — machine model; ECL-2.0 | https://gitlab.com/spinnaker2/SpiNNMachine2 | Runtime |
| SpiNNUtils2 — configuration manager; ECL-2.0 | https://gitlab.com/spinnaker2/SpiNNUtils2 | Runtime |
| spinnaker2-ml — Python + ARM-C ML library; Apache-2.0 | https://gitlab.com/spinnaker2/spinnaker2-ml | ML library |
| s2-sim2lab-app-snn — public SNN-only export of the gated SDK; Apache-2.0; created 28 May 2025 | https://gitlab.com/spinnaker2/s2-sim2lab-app-snn | SDK |
| snntorch-to-spinnaker2 — converter; Apache-2.0 | https://gitlab.com/spinnaker2/snntorch-to-spinnaker2 | Framework |
| spinnaker2-stm — Python interface to the board STM32 | https://gitlab.com/spinnaker2/spinnaker2-stm | Firmware |
| spinnaker2-config-tools — system config helper scripts | https://gitlab.com/spinnaker2/spinnaker2-config-tools | Tooling |
| model-explorer — Google Model Explorer mirrored into the `spinnaker2` namespace; Apache-2.0 | https://gitlab.com/spinnaker2/model-explorer | Tooling |
| mercury-xu8-bsp — Enclustra Mercury XU8 (Zynq UltraScale+) BSP; role *inferred* | https://gitlab.com/spinnaker2/mercury-xu8-bsp | Firmware |
| s2-chip-paper — scripts/source reproducing the chip paper's measurements | https://gitlab.com/tud-hpsn/public/s2-chip-paper | Reproduction |
| GitLab group `spinnaker2` (API listing of all 18 public projects) | https://gitlab.com/api/v4/groups/spinnaker2/projects | Repo index |

### Access-Gated Repositories (HTTP 403 to anonymous fetch)

| Resource | URL | Category |
|----------|-----|----------|
| **s2-sim2lab-app** — full board SDK (`ml-lib`, C++ host interface, examples) | https://gitlab.com/spinnaker2/s2-sim2lab-app | SDK (gated) |
| **spinnaker2_os** — SARK / SCAMP / Spin2API chip low-level software | https://gitlab.com/spinnaker2/spinnaker2_os | Firmware (gated) |
| s2_cpp_host_interface — C++ interface for the 48-node board | https://gitlab.com/spinnaker2/s2_cpp_host_interface | Runtime (gated) |
| s2-stm-software / s2_board_firmware — STM32 firmware | https://gitlab.com/spinnaker2/s2-stm-software · https://gitlab.com/spinnaker2/s2_board_firmware | Firmware (gated) |
| snn-core — chip-side SNN kernels, pulled as py-spinnaker2 submodule over SSH; declared open-sourced in v0.8.0 but no public HTTPS page resolved | `git@gitlab.com:spinnaker2/snn-core.git` | Kernels (status unclear) |
| mlops-infrastructure — README returns 403 | https://gitlab.com/spinnaker2/mlops-infrastructure | Infra (gated) |

### Third-Party Tools

| Resource | URL | Category |
|----------|-----|----------|
| AMD Quark — the INT8 PTQ library used by the OctopuScheduler front end | https://github.com/amd/Quark | Quantizer |
| NIR (neuromorphs/NIR) — Neuromorphic Intermediate Representation | https://doi.org/10.1038/s41467-024-52259-9 | IR |

### Deployment / Press / Vendor Marketing

| Resource | URL | Category |
|----------|-----|----------|
| Sandia Lab News, 12 Jun 2025 — NERL Braunfels: 175 M neurons, 3 chassis × up to 18 boards, 48 chips/board, arrived March 2025, NNSA ASC funded | https://www.sandia.gov/labnews/2025/06/12/brain-based-computing-for-nd-solutions/ | Deployment (primary) |
| Sandia, 8 May 2024 — original SpiNNcloud partnership announcement | https://www.sandia.gov/research/2024/05/08/neuromorphic-computing-for-nuclear-deterrence-solutions-sandia-partners-with-german-startup-spinncloud/ | Deployment (primary) |
| IEEE Spectrum, 8 May 2024 — commercial availability at ISC; 152 PEs; "10 billion neurons"; "0.3 exaops" | https://spectrum.ieee.org/neuromorphic-computing-spinnaker2 | Press |
| EE Times — Dresden opening; 90 boards/rack; full-build **target** "16 racks (69,120 chips) … 10.5 billion neurons" (future tense) | https://www.eetimes.com/spinnaker-based-neuromorphic-supercomputer-opens-in-dresden/ | Press |
| SpiNNcloud homepage — 18× / 78× vs-GPU claims; headline metric counters render as unpopulated placeholders ("0M", "0K") | https://spinncloud.com/ | Vendor marketing |
| SpiNNcloud SpiNNext product page — dynamic-sparsity positioning; **no specs, no process node, no dates** | https://spinncloud.com/spinnext/ | Vendor marketing (roadmap) |
| Wikipedia: SpiNNaker — 2019 EUR 8 M TU Dresden grant; "went into operation in 2025"; no SpiNNaker2 specs | https://en.wikipedia.org/wiki/SpiNNaker | Secondary |

---

## Key Findings

- **New definitive primary source.** A full-chip paper appeared on **27 July 2026** (Scholze, Partzsch,
  Höppner et al., IEEE OJCAS 2026, DOI 10.1109/OJCAS.2026.3714974, arXiv:2607.24396, CC-BY 4.0). It
  supersedes several numbers in the 2022 prototype paper, which described an **8-PE testchip**, not the
  product. Almost every hardware figure in this research folder traces to it.

- **Compute.** 152 ARM Cortex-M4F PEs in 38 Quad units, each PE with a **16×4 output-stationary MAC array
  (64 MACs/PE, 8-bit signed/unsigned, 16-bit fused, 24-bit accumulator)**. Chip total 9,728 MACs →
  **5.837 TOPS INT8 theoretical**, **4.563 TOPS measured** at 300 MHz/0.8 V, **2.77 TOPS/W** at
  150 MHz/0.5 V. The "153 ARM cores" on the vendor portal = 152 PEs + one periphery management M4.

- **No hardware data cache anywhere.** 128 kB SRAM per PE, linker-partitioned into ≈32 kB ITCM +
  ≈96 kB DTCM; 19.8 MB SRAM on chip; 2 GB LPDDR4 per node with the whole memory map planned **statically by
  the host**. Every byte movement is an explicit DMA or NoC transaction issued by software.

- **No tensor compiler.** No MLIR/LLVM lowering path, no XLA/StableHLO, no Triton, no TorchDynamo backend,
  no device tensor type, no collective-communication library. Each PE runs a hand-written, pre-compiled
  bare-metal C program built with stock **GNU Arm Embedded GCC** + custom linker scripts. Framework reach
  comes from **graph import** (ONNX for DNNs, NIR for SNNs), not from a native compiler.

- **No kernel-mode driver.** No PCIe, no `/dev` node, no ioctl layer. The host is a userspace process
  speaking **UDP (+UDT) over 1 GbE** to a memory-mapped view of the chip's registers, SRAM and DRAM.

- **Framing caveat that must survive into the survey.** The PEs are Cortex-M4F *microcontroller-class*
  cores. The designers say so themselves: *"SpiNNaker2 is not a pure DNN inference accelerator… scalability
  of conventional DNN workloads on a single chip is limited"* (arXiv:2607.24396). This is a **contrasting
  point in the design space**, not a TPU/Gaudi throughput rival.

- **Maturity: shipping, low volume.** Commercial availability announced May 2024 at ISC. Two documented
  systems: **Sandia NERL Braunfels** (175 M neurons, delivered March 2025) and **TU Dresden/SpiNNcloud**
  (~35,000 chips / 5 M cores / eight racks). The "10 billion neurons" headline is an *offered configuration*
  and a Dresden *build target*, never a delivered measurement. **SpiNNext is announced only** — no silicon,
  no specs, no date.

- **Open vs gated split.** py-spinnaker2 (Apache-2.0) plus the ECL-2.0 SpiNNaker1-lineage host toolchain
  (PACMAN2 / SpiNNMan2 / SpiNNMachine2 / SpiNNUtils2) are fully public and cover **single chip and 48-node
  board**. The full board SDK (`s2-sim2lab-app`) and the on-chip OS (`spinnaker2_os`: SARK/SCAMP/Spin2API)
  are **access-gated (HTTP 403)**, and SpiNNcloud's **large-scale-system stack is explicitly proprietary**
  and "currently under development" — the 35k-chip Dresden machine and NERL Braunfels run on software that
  is not public.

- **No independent benchmark.** No MLPerf submission exists. The "18× vs GPU" marketing figure does have a
  traceable peer-reviewed anchor for one case — EGRU language-model inference at **65 mJ on SpiNNaker2 vs
  1.19 J on an A100** — but at **8× longer execution time**. The "78×" SpiNNext claim has no anchor at all.
