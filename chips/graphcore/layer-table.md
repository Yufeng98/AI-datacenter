# Graphcore IPU Layer Mapping Table

*as_of: 2026-08-08*

> **2026-08-08 re-scan:** no layer changed. No new IPU generation and no new Poplar SDK were released between 2026-04-01 and 2026-08-08, so every row below still describes Bow IPU + Poplar SDK 3.4.0. Two rows carry added evidence/caveats (marked ⚠️); see the note under the tables.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (via PopTorch — wraps nn.Module, compiles via Poplar, minimal code changes) | confirmed | poptorch-user-guide, sdk-overview |
| Framework Integration | TensorFlow 2 / TF1 (via TF-IPU plugin, IPUStrategy, Keras API) | confirmed | sdk-overview |
| Framework Integration | ONNX (via PopART — import, training/inference sessions, Python/C++ API) | confirmed | sdk-overview, popart-docs |
| Compiler / IR | Poplar Graph Framework (static graph compiler: tile assignment, BSP superstep, exchange schedule, codelet compilation) | confirmed | poplar-user-guide, ipu-prog-guide |
| Compiler / IR | PopTorch frontend (torch.fx trace → Poplar graph IR conversion) | confirmed | poptorch-user-guide |
| Compiler / IR | PopART frontend (ONNX → Poplar graph IR conversion) | confirmed | sdk-overview |
| Op Library | PopLibs — poplin (GEMM, Conv2D fwd/bwd/upd, grouped conv, matmul) | confirmed | poplibs-api, poplibs-github |
| Op Library | PopLibs — popnn (LSTM, GRU, RNN, layer norm, batch norm, softmax, GELU) | confirmed | poplibs-api |
| Op Library | PopLibs — popops (elementwise: add/mul/exp/log; reduce: sum/max/mean; gather/scatter/cast) | confirmed | poplibs-api |
| Op Library | PopLibs — poprand (uniform, normal, Bernoulli RNG) | confirmed | poplibs-api |
| Op Library | PopLibs — popsparse (sparse GEMM, sparse embedding lookup) | confirmed | poplibs-api |
| Kernel Language | Codelets / Vertices (C++ or assembly; poplar::Vertex base class; per-tile worker thread; SRAM-only access) | confirmed | poplar-user-guide, custom-ops-guide |
| Assembler / ISA | Tile Vertex ISA (native IPU tile instruction set; LLVM-based compilation; publicly documented) | confirmed | tile-isa-docs |
| Runtime | poplar::Engine (host-side: loads compiled binary via PCIe, runs BSP program sequences, manages Streaming Memory DMA) | confirmed | poplar-user-guide |
| Driver / Firmware | PCIe HAL (closed-source kernel driver for IPU-Machine PCIe; part of Poplar SDK) | inferred | sdk-overview |
| Virtualization | V-IPU (virtualized IPU resource manager; Slurm integration; multi-tenant partitioning) | confirmed | vipu-admin-guide |
| Communication | IPU-Link (direct chip-to-chip; ~64 GB/s bidir per pair; intra-M2000) | confirmed | ipu-prog-guide, pod-datasheet |
| Communication | GW-Link (gateway rack-to-rack; looped topology ≤POD256; switched topology for large clusters) | confirmed | switched-gwlinks-docs |
| Profiling | PopVision Graph Analyser (tile mapping, memory usage, Poplar graph visualization) | confirmed | popvision-blog |
| Profiling | PopVision System Analyser (BSP superstep timeline, compute vs exchange ratio, cycle counts) | confirmed | popvision-blog |
| Inference Serving | Triton Inference Server backend (gRPC/HTTP serving of PopTorch/PopART/TF-IPU models) — ships in Poplar SDK 3.4.0; ⚠️ *not* related to `graphcore/triton-fork` | confirmed | sdk-overview |
| Inference Serving | vLLM on IPU — **no evidence of any IPU backend**; `graphcore/vllm-fork` (pushed 2025-09-22) and `graphcore/triton-fork` (pushed 2026-06-08) both carry stock upstream READMEs with no IPU support. Recorded as a negative so aggregators are not mistaken for a port. | absent (verified 2026-08-08) | github-api-graphcore-org, github-triton-fork, github-vllm-fork |
| SDK Maintenance | Open-source Poplar stack **dormant**: newest tags in `graphcore/poplibs` and `graphcore/poptorch` are `sdk/poplar/3.4.0` and `sdk/poptorch/3.4.0`; last code push to either repo was October 2023 | confirmed (2026-08-08, GitHub API) | github-api-poplibs-tags, github-api-poptorch-tags, github-api-poptorch-releases |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | IPU Tile ×1,472: full MIMD processor; 6 worker threads + 1 supervisor; FP16 MAC + FP32 FPU + 32-bit ALU | confirmed | ipu-prog-guide, hot-chips-2021 |
| Compute Engine | 8,832 total hardware threads (6 × 1,472); barrel/round-robin scheduled | confirmed | ipu-prog-guide, citadel-microbenchmark |
| Compute Engine | GC200: 250 TFLOPS FP16 @ 1.35 GHz; Bow IPU: 350 TFLOPS FP16 @ 1.85 GHz | confirmed | bow-processors-page, hot-chips-2021 |
| Data Path | BSP: Compute phase (SRAM-local, MIMD) → Exchange phase (all-to-all DMA) → Barrier sync | confirmed | ipu-prog-guide, bsp-blog |
| Data Path | In-order pipeline; no out-of-order execution; no hardware prefetch; no cache coherence | confirmed | ipu-prog-guide, citadel-microbenchmark |
| On-chip Memory | 624 KB private SRAM per tile; no shared access; deterministic latency; no L1/L2 hierarchy | confirmed | ipu-prog-guide, hot-chips-2021 |
| On-chip Memory | 900 MB total on-chip SRAM (1,472 × 624 KB) | confirmed | ipu-prog-guide, bow-processors-page |
| On-chip Memory | IPU Exchange Fabric: all-to-all non-blocking; 11 TiB/s aggregate; statically scheduled by Poplar | confirmed | ipu-prog-guide, hot-chips-2021 |
| Off-chip Memory | Streaming Memory: DDR4 DIMMs; up to 112 GB/IPU; 180 TB/s aggregate (4-IPU M2000); explicit DMA only | confirmed | ipu-prog-guide, intel-memory-blog |
| Host Interface / Package | GC200: TSMC 7nm; 823 mm²; 59.4 B transistors; PCIe host interface | confirmed | hot-chips-2021 |
| Host Interface / Package | Bow IPU: TSMC 7nm WoW; active die + Cu-Cu bonded power delivery die; 1.85 GHz; ~120W TDP | confirmed | bow-processors-page, toms-hardware |
| Scale-up Interconnect | IPU-Link: direct IPU-to-IPU within M2000; fully connected 4-IPU topology | confirmed | ipu-prog-guide, pod-datasheet |
| Scale-out Interconnect | GW-Link: looped or switched topology; IPU-Gateway chip per M2000 | confirmed | switched-gwlinks-docs, pod128-datasheet |
| Scale-out Interconnect | IPU-POD systems: POD4 (1.4 PFLOPS) → POD128 (45 PFLOPS) → POD256 → 64K IPU (16 EFLOPS) | confirmed | pod128-datasheet, nextplatform |
| Precision | FP16 with stochastic rounding (FP16.SR) — primary AI training mode | confirmed | ipu-prog-guide, bow-blog |
| Precision | FP32 scalar; FP16 multiply + accumulate; no FP8/BF16/INT4 (re-confirmed 2026-08-08 — no new datapath announced) | confirmed | ipu-prog-guide |
| Generation | Bow IPU (2022) is the current and only shipping part; no post-Bow / Mk3 IPU announced as of 2026-08-08; Graphcore absent from Hot Chips 38 advance program | confirmed (negative) | graphcore-posts, hotchips-38-advance-program |

---

## 2026-08-08 Re-scan Note

Scan window 2026-04-01 → 2026-08-08. Result: **zero layer changes**.

- **Hardware:** no new IPU generation. Bow (2022) remains current. The one 2026 Graphcore technical post (stochastic rounding, 2026-05-26) documents an existing GC200/Bow feature.
- **Software:** no new Poplar SDK. Verified via the GitHub API rather than the docs site, whose SDK-overview page states no current version at all.
- **Do not add:** a Graphcore/Ampere "Izanagi" chip circulates in analyst commentary (Jon Peddie Research, 2026-06-16) with no primary confirmation; it is not a Graphcore product and must not be entered as a layer or generation.
- Corporate/funding developments in this window (SoftBank recapitalisation, leadership change, new sites) do not touch any software or hardware layer; they are recorded in `chips/graphcore/summary.md`.
