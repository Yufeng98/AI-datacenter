# Graphcore IPU Software and Hardware Stack Summary

*as_of: 2026-08-08*
*acquisition_note: Graphcore acquired by SoftBank, July 2024 (terms never officially disclosed; reported at roughly $500–600M). Operates as SoftBank subsidiary. Last SDK release: Poplar SDK 3.4.0 (March 2024) — still current as of 2026-08-08. Current and only shipping silicon: Bow IPU (2022). See "Corporate and Roadmap Update (2026-08-08)" below.*

---

## Overview

Graphcore's IPU (Intelligence Processing Unit) is a radically different accelerator architecture built for machine intelligence workloads involving large, sparse, irregular computation graphs. Rather than following the GPU (SIMT warp-based) or TPU (systolic array) design philosophy, the IPU is a **massively parallel, distributed-memory MIMD processor** that enforces the **Bulk Synchronous Parallel (BSP)** execution model in hardware.

The current silicon is the **Bow IPU** (2022), based on the same **GC200 Colossus Mk2** microarchitecture (1,472 independent tiles, 900 MB on-chip SRAM) but featuring TSMC's 3D Wafer-on-Wafer (WoW) power delivery die — the world's first production use of this technology — enabling a 40% TFLOPS improvement at 16% better energy efficiency.

The defining characteristic of the IPU is its **memory architecture**: 900 MB of ultra-fast SRAM is distributed across 1,472 private tile memories (624 KB/tile), connected by an 11 TiB/s all-to-all on-chip exchange fabric. This gives the IPU ~3,000× more on-chip bandwidth than an H100's HBM, at the cost of far less total capacity. Large models are supported via **Streaming Memory** — DDR4 DIMMs explicitly paged into tile SRAM.

The software stack is the **Poplar SDK** — a graph-based, ahead-of-time compiler that statically schedules every BSP superstep, tile assignment, exchange operation, and SRAM placement. PyTorch users interact via **PopTorch** (minimal code changes); ONNX models run via **PopART**. **PopLibs** provides IPU-optimized op libraries (GEMM, Conv, reductions). Low-level custom kernels are written as **codelets** (C++ or assembly vertices) targeting the tile ISA.

---

## Key Specifications

| Parameter | GC200 (Mk2) | Bow IPU |
|---|---|---|
| Process | TSMC 7 nm | TSMC 7 nm WoW |
| Tiles | 1,472 | 1,472 |
| Threads (total) | 8,832 | 8,832 |
| On-chip SRAM | 900 MB | 900 MB |
| FP16 TFLOPS | 250 | 350 |
| Exchange bandwidth | 11 TiB/s | 11 TiB/s |
| Clock | 1.35 GHz | 1.85 GHz |
| Die area | 823 mm² | 823 mm² + power die |
| Transistors | 59.4 B | ~59.4 B (active) |
| Streaming Memory | up to 112 GB/IPU | up to 112 GB/IPU |

---

## Software Stack

### Framework Integration

- **PopTorch (PyTorch)**: Wraps `nn.Module` with `poptorch.trainingModel()` or `poptorch.inferenceModel()`. On first call, traces via `torch.fx`, converts to Poplar graph IR, compiles. Subsequent calls reuse compiled executable. IPU-specific: `poptorch.BeginBlock()`/`EndBlock()` for pipeline staging; `poptorch.recomputationCheckpoint()` for activation recomputation; `poptorch.Options()` for replication factor and gradient accumulation. **Open source.**
- **TensorFlow for IPU (TF-IPU)**: TF2/TF1 backends via `ipu.ipu_strategy.IPUStrategy`. Supports Keras model API.
- **PopART (ONNX)**: Poplar Advanced Runtime — imports ONNX models and runs training (`TrainingSession`) or inference (`InferenceSession`). Python and C++ APIs.

### Compiler / IR

- **Poplar Graph Framework**: Core compiler. Programs are built as static dataflow graphs using the `poplar::Graph` C++ API. The compiler assigns tensors to tiles, generates per-tile compute programs and exchange schedules statically. `poplar::Engine` compiles the graph into a fully static BSP schedule — no dynamic dispatch, no JIT at runtime.
- **Compilation model**: Ahead-of-time, static. Compilation can take minutes; caching compiled executables is essential in production.

### Op Library

- **PopLibs** (open source, github.com/graphcore/poplibs):
  - `poplin`: GEMM, convolution (fwd/bwd/weight-update), grouped conv, matmul
  - `popnn`: LSTM, GRU, RNN, layer norm, batch norm, softmax, GELU
  - `popops`: element-wise ops, reductions (sum/max/mean), gather, scatter, cast
  - `poprand`: uniform, normal, Bernoulli RNG
  - `popsparse`: sparse GEMM, sparse embedding lookup

### Kernel Language

- **Codelets / Vertices**: C++ classes inheriting `poplar::Vertex`; each runs on one tile worker thread; accesses only tile-local SRAM. Can be written in assembly for peak performance. Tile vertex ISA is publicly documented.

### Runtime

- **poplar::Engine**: Host-side runtime loading compiled binary onto IPU via PCIe. Manages Streaming Memory DMA. Supports multi-IPU distribution.
- **V-IPU**: Multi-tenant resource manager; Slurm integration for HPC clusters.

### Driver / Firmware

- **PCIe HAL**: Closed-source kernel driver for IPU-Machine PCIe communication.

### Communication

- **IPU-Link**: Direct chip-to-chip links within IPU-M2000 (4 IPUs); ~64 GB/s bidirectional per pair.
- **GW-Link**: Gateway links between racks; looped (up to ~POD256) or switched topologies.

### Assembler / ISA

- **Tile Vertex ISA**: Native IPU tile instruction set; publicly documented by Graphcore. LLVM-based compilation in Poplar SDK.

---

## Hardware Architecture

### Compute Engine

Each of the 1,472 **tiles** on the GC200/Bow IPU is a complete, independent processor:
- 6 hardware worker threads (barrel/round-robin scheduled: 1 instruction per context per cycle)
- 1 supervisor thread (privileged; manages vertex dispatch, exchange initiation)
- 7 total hardware contexts → **8,832 total threads** on GC200
- FP16 vector MAC unit, FP32 scalar FPU, 32-bit integer ALU
- In-order pipeline; no out-of-order execution; no hardware prefetch
- Clock: 1.35 GHz (GC200 Mk2), **1.85 GHz** (Bow IPU)

### Data Path: BSP

1. **Compute phase**: All 1,472 tiles execute independently on private SRAM. No communication possible.
2. **Exchange phase**: All tiles simultaneously DMA through the IPU Exchange fabric (all-to-all, 11 TiB/s).
3. **Barrier sync**: Global hardware barrier; all tiles sync before next superstep.

Race-free, deadlock-free, bit-reproducible.

### On-chip Memory

- 624 KB private SRAM per tile; **900 MB total**
- No L1/L2 cache, no virtual memory, no page faults during compute
- **11 TiB/s** on-chip all-to-all exchange bandwidth

### Off-chip Memory

- **Streaming Memory**: DDR4 DIMMs, up to 112 GB/IPU; **180 TB/s** aggregate per M2000 (4 IPUs)
- Accessed via explicit DMA copy into tile SRAM only

### Bow IPU — 3D Wafer-on-Wafer

- Top wafer: GC200 active die (1,472 tiles, 900 MB SRAM, exchange fabric, TSMC 7nm)
- Bottom wafer: Cu-Cu hybrid bonded power delivery die (capacitor reservoirs)
- Enables 1.85 GHz at lower VDD → **40% more TFLOPS, 16% better efficiency**
- World's first production 3D WoW chip (2022)

### System Scale

| System | IPUs | AI Compute |
|--------|------|-----------|
| IPU-M2000 | 4 | 1.4 PFLOPS |
| IPU-POD16 | 16 | 5.6 PFLOPS |
| IPU-POD64 | 64 | ~22 PFLOPS |
| IPU-POD128 | 128 | ~45 PFLOPS |
| IPU-POD256 | 256 | ~90 PFLOPS |
| Max Bow Pod | ~64K | ~16 EFLOPS |

---

## Programming Model Rationale

1. **BSP enforces correctness by construction.** Race conditions and deadlocks are impossible: tiles only communicate during the exchange phase, which completes before any tile begins the next compute phase.
2. **Distributed SRAM enables bandwidth-first computing.** 11 TiB/s on-chip exchange bandwidth (~3,000× H100 HBM) allows restructuring algorithms to maximize data reuse in tile-local SRAM. Optimal for sparse models and GNNs.
3. **MIMD enables sparsity-native execution.** Each tile runs an independent instruction stream — tiles can branch and execute completely different code. Irregular, graph-structured workloads don't serialize divergent paths.
4. **Static compilation eliminates runtime overhead.** Poplar produces a fully static schedule with no kernel launch overhead, no runtime routing decisions, no warp divergence handling.
5. **Stochastic rounding enables high-accuracy FP16 training.** Hardware FP16.SR provides convergence closer to FP32 without FP32 compute cost — unique to Graphcore.
6. **Pipeline parallelism maps naturally to BSP.** Model stages assigned to different IPUs; the exchange phase efficiently transfers activations between pipeline stages.

---

## Distinguishing Strengths vs Limitations

### Strengths
- 900 MB SRAM on-chip (18× H100's 50 MB L2)
- 11 TiB/s on-chip BW (~3,000× H100 HBM BW) — ideal for bandwidth-bound sparse/GNN workloads
- MIMD: tiles execute different code — natural for irregular graphs
- BSP: race-free, deterministic, reproducible
- First production WoW chip (Bow, 2022)
- Hardware stochastic rounding (FP16.SR) — unique training accuracy advantage

### Limitations
- 350 TFLOPS FP16 (vs H100 989 TFLOPS BF16) — ~3× lower raw compute
- BSP load imbalance: slow tiles stall all others at barrier
- Static compilation: dynamic shapes and data-dependent control flow require workarounds
- Compilation latency: minutes for large models
- No FP8/BF16/INT4 support (as of SDK 3.4.0)
- Post-SoftBank acquisition: next-gen roadmap and SDK development pace uncertain. As of 2026-08-08 the open-source Poplar stack is effectively dormant (newest release tags in `graphcore/poplibs` and `graphcore/poptorch` are still `sdk/…/3.4.0`; last code push to either repository was October 2023), and both technical co-founders have left the company (Simon Knowles, director until 2025-08-20; Nigel Toon, Executive Chair until 2026-07-31).

---

## Acquisition Status (as of 2026-04-05)

Graphcore was acquired by SoftBank Group in July 2024 for ~$500M. Continues as SoftBank subsidiary. Aligned with Arm holdings toward ASI platform. Last public SDK: Poplar SDK 3.4.0 (March 2024). Next-generation IPU discussed but unannounced.

> **Superseded in part, 2026-08-08.** The July 2024 timing stands (SoftBank-side directors Ippei Mimura and Jared Roscoe were appointed to GRAPHCORE LIMITED on 2024-07-11, the same day seven prior investor-directors resigned). The **"~$500M" price should be read as an estimate, not a disclosed figure** — deal terms were never officially disclosed; Sifted reported "$600m+" and Jon Peddie Research "$500–600 million". Read as "reported at roughly $500–600M; terms not officially disclosed". The "next-generation IPU discussed but unannounced" and "Poplar SDK 3.4.0" statements both remain accurate as of 2026-08-08. Graphcore's own framing is a partnership with SoftBank Corporation from 2023 followed by acquisition of the business in summer 2024.

---

## Corporate and Roadmap Update (2026-08-08)

*Updated 2026-08-08. Sources: UK Companies House filing history and officer records for GRAPHCORE LIMITED (company no. 10185006); CNBC 2026-05-12; Graphcore blog posts (2026-05-14, 2026-08-03, July 2026 leadership post); Bristol24/7 2026-07-31; Sifted 2025-08-27; GitHub API tag/release listings for `graphcore/poplibs` and `graphcore/poptorch`; Hot Chips 38 advance program.*

**Headline: no new silicon, no new SDK — but a large and ongoing SoftBank recapitalisation.** Between 2026-04-01 and 2026-08-08 Graphcore announced no new IPU and shipped no new Poplar SDK. The material in-window developments are financial, corporate, and geographic. Bow (2022) remains the current and only shipping part.

### No new silicon (confirmed negative)

- **No next-generation IPU** — no Mk3, no post-Bow part — was announced in the window. Graphcore's 2026 blog is dominated by recruiting, culture, and office posts; the sole technical post, *"Stochastic Rounding: How randomness helps us build better models"* (2026-05-26), describes an **existing GC200/Bow hardware feature** (FP16.SR), not new silicon.
- **Graphcore does not appear anywhere in the Hot Chips 38 advance program** (Aug 23–25, 2026). The AI-accelerator slots there go to Meta, NVIDIA, Cerebras, Microsoft, SambaNova, Google, OpenAI and d-Matrix.
- ⚠️ **Rumor, not a fact:** a Graphcore/Ampere co-developed **"Izanagi"** chip targeting SoftBank Stargate deployments in 2026 circulates in analyst commentary (Jon Peddie Research, 2026-06-16) and derivative blogs. **No primary Graphcore or SoftBank confirmation exists.** This is recorded here only so the survey does not pick it up from aggregators; it must not be cited as a Graphcore product.

### No new Poplar SDK (confirmed negative)

Verified via the GitHub API rather than the docs site (the docs SDK-overview page states no current version at all): the newest release tags in `graphcore/poplibs` and `graphcore/poptorch` are **`sdk/poplar/3.4.0`** and **`sdk/poptorch/3.4.0`**, and the last code push to either repository was **October 2023**. The open-source Poplar stack is effectively **dormant**. The repo's "Poplar SDK 3.4.0 (March 2024)" baseline stands unchanged.

GitHub org activity exists but does **not** indicate a stack shift. `graphcore/triton-fork` (pushed 2026-06-08) and `graphcore/vllm-fork` (pushed 2025-09-22) both carry **stock upstream READMEs with no IPU backend and no mention of IPU support** — they read as plain upstream mirrors, not IPU ports. `graphcore/pytorch-fork` was pushed 2026-08-05; `graphcore/llvm-project-fork`, which does carry a Colossus IPU backend, was last pushed 2026-03-11. **Do not report a Triton or vLLM IPU inference path** on this evidence.

### SoftBank capital injection — the real in-window headline

- Graphcore issued a **single share to SoftBank valued at roughly $457M**, recorded in a Companies House **SH01 filed 2026-04-13**. This was picked up by CNBC and others on **2026-05-12** ("SoftBank has injected $450 million into this British AI chip company"). The $457M valuation attaches to that April share and is **press-derived from the registry filing, not a vendor statement**.
- The filing history shows this is a **continuing recapitalisation, not a one-off**: further RES10 allotment resolutions and SH01 filings on **2026-05-29, 2026-06-08, 2026-07-24 and 2026-08-04**, plus articles re-adopted 2026-04-30. Reporting indicates the April tranche is "only part of the funding Graphcore is expected to receive from SoftBank this year."
- Confidence: **high** — primary registry filings.

### Leadership

- **Nigel Toon**, co-founder and Executive Chair, **stepped down with effect from Friday 31 July 2026** (Graphcore post and Toon's own LinkedIn article; independently reported by Bristol24/7 on 2026-07-31). Companies House records his directorship of GRAPHCORE LIMITED as **terminated 30 July 2026** (TM01 filed 2026-08-06) — a one-day discrepancy against the announcement, noted but not material.
- **Marcus William McElroy** leads the company. This is **not a discrete July 2026 handover**: Companies House shows he was **appointed a director on 18 December 2025**, i.e. before this repo's 2026-04-05 baseline. July 2026 was Toon's final exit, not McElroy's arrival. **His title is not vendor-confirmed** — Graphcore's own post says only that he "takes the helm"; Bristol24/7 calls him "chief executive"; theofficialboard still lists him as General Manager.
- Graphcore's post names supporting executives **Nick Bishop, Helen Byrne, Sally Doherty, Tim Ramsdale, Rami Sinno and John Walsh**, and states the company is **"approaching 1,000 employees"**. Both are **self-reported vendor claims** with no independent corroboration found.
- Toon's move to **BlankPage Capital** (deep-tech semiconductor/AI VC) is **not new**: Sifted reported the fund launch on 2025-08-27; Crunchbase lists him as co-founder and managing partner. It predates this repo's baseline.
- Co-founder **Simon Knowles** — the IPU's chief architect — **ceased to be a director on 20 August 2025** (Companies House; Sifted). **Both technical co-founders have now left**, which directly bears on the "next-gen roadmap uncertain" limitation above.

### Geographic footprint

| Site | Status | Event date | Note |
|---|---|---|---|
| Bengaluru, India — AI Engineering Campus | Open | **inaugurated 2026-05-06** | Blog post published 2026-05-14; the post date is *not* the event date |
| Taipei, Taiwan | Open | **officially opened 2026-06-04** | "Welcome to Graphcore Taipei" post published 2026-08-03; again, post date ≠ event date |
| Bristol, UK — new global headquarters | Planned | September 2026 | Announced in Graphcore's July 2026 leadership post |

Graphcore's July 2026 post names development centres in **Austin (Texas), Bengaluru (India), Taipei and Hsinchu City (Taiwan), Gdańsk (Poland), and Cambridge and London (UK)**, plus the planned Bristol global HQ.

**India investment (predates the repo baseline, previously absent):** on **2025-10-09** Graphcore announced a **£1bn** investment in India creating **500 semiconductor jobs**, over roughly a decade. Independently confirmed (Bloomberg, Data Center Dynamics, Entrackr). Note the currency framing differs by outlet — Graphcore says £1bn, Bloomberg and Indian outlets say "$1.3 billion"; it is the same commitment.

### Status verbs

Nothing in this window is announced silicon, let alone sampling or shipping. The in-window facts are a **funding event** (completed, per registry), a **leadership exit** (completed), and **office openings** (completed).

---

## Resources

### Official Documentation
- [Poplar SDK Overview](https://docs.graphcore.ai/projects/sdk-overview/en/latest/overview.html)
- [IPU Programmer's Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html)
- [PopTorch User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/intro.html)
- [Memory and Performance Optimisation](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/understand-ipu-programming-model.html)
- [Tile Vertex ISA](https://docs.graphcore.ai/projects/isa/en/latest/)
- [IPU-POD128 Datasheet](https://docs.graphcore.ai/projects/ipu-pod128-datasheet/en/latest/product-description.html)

### Open-Source Repositories
- [poptorch](https://github.com/graphcore/poptorch)
- [poplibs](https://github.com/graphcore/poplibs)
- [examples](https://github.com/graphcore/examples)

### Hardware
- [Bow IPU Processors](https://www.graphcore.ai/bow-processors)
- [Hot Chips 2021 — Colossus Mk2](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf)

### Corporate Status (added 2026-08-08)
- [Companies House — GRAPHCORE LIMITED (10185006) filing history](https://find-and-update.company-information.service.gov.uk/company/10185006/filing-history)
- [Companies House — GRAPHCORE LIMITED (10185006) officers](https://find-and-update.company-information.service.gov.uk/company/10185006/officers)
- [CNBC — SoftBank injects ~$450M into Graphcore (2026-05-12)](https://www.cnbc.com/2026/05/12/softbank-graphcore-ai-chip-investment.html)
- [Graphcore — Nigel Toon steps down](https://www.graphcore.ai/posts/graphcore-co-founder-and-executive-chair-nigel-toon-steps-down)
- [Bristol24/7 — Co-founder of Bristol AI firm steps down (2026-07-31)](https://www.bristol247.com/business/news-business/co-founder-of-billion-dollar-bristol-ai-firm-steps-down/)
- [Sifted — Graphcore co-founder Simon Knowles exits (2025-08-27)](https://sifted.eu/articles/graphcore-cofounder-exits-company-one-year-on-from-softbank-acquisition)
- [Data Center Dynamics — £1bn India investment, Bengaluru AI Engineering Campus](https://www.datacenterdynamics.com/en/news/softbanks-graphcore-announces-1bn-investment-in-india-opens-ai-engineering-campus-in-the-country/)
- [Graphcore — Bengaluru campus opens its doors](https://www.graphcore.ai/posts/graphcores-bengaluru-campus-opens-its-doors)
- [Graphcore — Welcome to Graphcore Taipei](https://www.graphcore.ai/posts/welcome-to-graphcore-taipei)
