# Enflame Software Stack Investigation Report

*investigator: search-chip-toolchain*
*as_of: 2026-04-05*
*source: GitHub (EnflameTechnology org), support.enflame-tech.com, HAMi project docs, PaddleNLP docs*

---

## Investigation Summary

Enflame's software platform is called **TopsRider**. It is a full-stack SDK analogous to NVIDIA CUDA Toolkit, covering driver, compiler, runtime, operator library, communication library, and framework integration. The platform is partially proprietary and partially open-sourced.

---

## Framework Integration

**TopsTorch** (proprietary): PyTorch plug-in enabling `torch.device("gcu")`. ATen operations dispatch to TopsBlasOps. Confirmed by HAMi dependency documentation and multiple integration guides.

**vllm-gcu** (open-source, github.com/EnflameTechnology/vllm-gcu): Enflame-maintained vLLM fork for LLM serving on GCU S60. Operator-level GCU optimisations. OpenAI-compatible API server. Supports Llama, Mistral, Qwen2.

**candle-gcu / candle-vllm-gcu** (open-source): HuggingFace Candle (Rust) ported to GCU. Demonstrates multi-language framework ambition beyond Python.

**PaddleNLP GCU backend**: Documented integration for LLaMA2-13B inference on S60.

**Qwen2 GCU support** (PR #456 in QwenLM/Qwen2): GCU backend merged to Alibaba Qwen2 repo.

---

## Compiler / IR

**TopsCC** (proprietary): The central compiler. Accepts PyTorch eager graphs, ONNX, PaddlePaddle models; lowers to GCU-CARE static dataflow. Key: static schedule computed at compile time — no runtime thread dispatch. TopsCC also inserts sparsity annotations and handles all memory tiling (no user-visible DMA calls).

**GCU-CARE Scheduler**: Integrated into TopsCC; maps computation graph nodes to SIPs at compile time.

Note: No public PTX/MLISA equivalent. GCU-CARE ISA is entirely proprietary and not documented externally. This is the main gap vs Cambricon (partial MLISA docs) and NVIDIA (PTX is public).

---

## Op Library

**TopsBlasOps** (proprietary): GEMM, BLAS, Conv, Attention, Norm primitives. Part of TopsPlatform. Dispatched by TopsTorch for standard PyTorch ops.

**TopsCodec / FFmpeg-GCU** (open-source): Hardware video encode/decode kernels exposed as FFmpeg plugins. Confirms lower-level media compute capability of GCU.

---

## Kernel Library

**TopsPlatform Kernel Library** (proprietary): GCU-CARE native binaries bundled with TopsCC SDK. Not open-sourced. Analogous to CUTLASS (NVIDIA) or mlu-ops (Cambricon).

---

## Runtime

**TopsRuntime** (`libTopsRuntime.so`, proprietary): Device runtime API. Device memory alloc/free, stream/event model, kernel launch. Confirmed via HAMi dependency manifest and vllm-gcu README.

**GCU Monitor**: Runtime observability and profiling tools. Kubernetes monitoring integration confirmed via support.enflame-tech.com docs.

---

## Driver / Firmware

**TopsDrv** (proprietary Linux kernel module): PCIe BAR mapping, IOCTL, DMA engine, interrupt handling. Installed with TopsPlatform SDK. No on-device firmware RISC-V (unlike NVIDIA GSP).

**HAMi GCU plugin**: Kubernetes device plugin for GCU resource scheduling and sharing. Confirmed in Project-HAMi repo.

---

## Communication

**ECCL** (Enflame Collective Communication Library): AllReduce, AllGather, ReduceScatter, Broadcast. Installed from `eccl_3.5*.deb`. NCCL analog. Operates over GCU-LARE for intra-node, Ethernet/RoCE for inter-node.

---

## Assembler / ISA

**GCU-CARE ISA** and **GCU-DARE ISA**: Entirely proprietary. Not publicly documented. TopsCC fully abstracts the ISA. No user path to hand-write GCU-CARE instructions (contrast: NVIDIA ptxas, Cambricon CNAS). This is a significant ecosystem openness gap.

---

## Confidence Assessment

| Component | Confidence | Basis |
|-----------|------------|-------|
| vllm-gcu existence and function | High | Open-source GitHub repo |
| TopsRider as platform name | High | Multiple secondary sources |
| TopsCC compiler (name + function) | High | HAMi docs, HC33 paper |
| TopsRuntime API | High | vllm-gcu README, HAMi docs |
| ECCL (name + function) | High | GitHub org, vllm-gcu |
| TopsTorch PyTorch integration | Medium | HAMi docs, secondary sources |
| TopsBlasOps name | Medium | HAMi dependency manifest |
| GCU-CARE ISA details | Low | Not publicly documented |
| Kernel library internals | Low | No open-source repository |

---

# Update Investigation — 2026-08-08

*investigator: update-chip-landscape (scan + round-3 adversarial verification)*
*window: 2026-04-05 → 2026-08-08*
*sources: EnflameTechnology GitHub org API, torch-gcu README/LICENSE/.version, vllm-gcu releases API, PaddlePaddle FastDeploy docs*

## Headline: the PyTorch backend source is now public

Enflame published **[github.com/EnflameTechnology/torch-gcu](https://github.com/EnflameTechnology/torch-gcu)** — repo created **2026-06-23**, BSD-style licence derived from PyTorch's, "Copyright (c) 2026 Enflame 燧原科技", `.version` = 3.7.

This supersedes the baseline characterization "TopsTorch: The proprietary PyTorch plug-in backend" — but the supersession must be stated carefully:

- **No source states `torch_gcu` is a rename of, or a replacement for, TopsTorch.** The two names are not linked by any document found.
- The repo is a **single-commit code drop**: `created_at == pushed_at == 2026-06-23`, no follow-on commits, no press coverage.
- Its own **Hardware Support table lists exactly one product — CloudBlazer S60** (inference). No T20 and no L600 entry, so it does not on its face cover training silicon.

Correct framing: *a source-code publication of the PyTorch backend*, not a demonstrably live open-source project. It nonetheless closes part of the openness gap this survey recorded against Cambricon's `torch_mlu`.

### Verified from the torch-gcu README

| Layer | Detail |
|---|---|
| Backend mechanism | PyTorch's official **PrivateUse1** dispatch key; C++ tensors surface as `privateuseoneFloatType` |
| Operator coverage | Extensive ATen coverage with automatic CPU fallback for unimplemented ops |
| Distributed | **ECCL** registered as a `torch.distributed` backend (`backend="eccl"`); full collectives plus `send`/`recv`/`isend`/`irecv` |
| Compiler integration | `torch.compile` / **Inductor** backend |
| Mixed precision | `torch.gcu.amp.autocast`, `torch.gcu.amp.GradScaler` |
| Graph capture | `torch.gcu.GCUGraph` — CUDA-Graph analog |
| Profiling | `ProfilerActivity.GCU`, Kineto integration |
| Memory | PyTorch-compatible caching allocator |
| Migration utility | `transfer_to_gcu` — one-line CUDA→GCU migration shim |
| C++ inference | `libtorch_gcu`; **single-device inference only** — no multi-device, no training in the C++ API |
| Source tree | includes `csrc/aotfusion/` and `csrc/efficient_ops/` |
| Version matrix | torch_gcu 2.10.0 / PyTorch 2.10.0 / Python 3.9, 3.10, 3.12; against **TopsRider v3.7.1** |
| Documented limitation | GCU has no native 64-bit types; F64/I64 are implicitly down-cast to 32-bit |

## SDK version line — TopsRider 3.7.x

Independently verifiable from the torch-gcu repo:

- README pins **TopsRider v3.7.1**
- Docker tag `v2.10.0-TR3.7.107-ubuntu2204`
- Bundled driver `enflame-x86_64-gcc-1.7.2.2402-20260429134535.run` (dated 2026-04-29)
- Repo `.version` = `3.7`

This replaces the baseline's `onlinedoc_dev_2.5.115` doc references as the current SDK line.

**Not confirmed:** the documentation-site URL slug. `support.enflame-tech.com` returns a Tencent WAF **HTTP 403** for every `onlinedoc_dev_*` path attempted (2.5.115, 3.5, 3.6, 3.7). A claim circulating that docs "moved to onlinedoc_dev_3.6" could not be validated and is *lower* than the version torch-gcu actually pins; do not record a doc slug.

## vllm-gcu v0.11.0 (2026-05-12)

Confirmed against the GitHub releases API. Tag history: v0.8.0 (2025-09-17) → v0.9.2 (2025-12-23) → **v0.11.0 (2026-05-12T02:41:23Z)**; latest push 2026-07-07.

Release-note themes:

- **DeepSeek 3.2 runtime work**: async scheduling, MTP, DBO, FlashMLA, indexer optimization, FP8 KV cache
- **Layer-wise KV-cache transfer** and first-token reuse
- New **GCU custom / native operators**
- **FP8 linear and FP8 MoE** paths; **W8A8-INT8 MoE**
- GCU **Docker source builds**
- Model enablement: Qwen2.5/3-VL, Qwen3-Next, DeepSeek 3.2, DeepSeek-OCR, Hunyuan-OCR/A13B, GLM-4.5-Air, Step3-VL, MiniCPM-V, Paddle-VL

## Other OSS activity (GitHub org API, verified timestamps)

| Repo | created_at | Read |
|---|---|---|
| `FlagOS` | 2026-05-29 | FlagOS support branch |
| `nixl` (fork) | 2026-06-08 | NVIDIA Inference Xfer Library fork; `created_at == pushed_at`; no visible Enflame commits |
| `ucx` (fork) | 2026-06-08 | Unified Communication X fork; same pattern |
| `torch-gcu` | 2026-06-23 | PyTorch backend source (above) |
| `findtops` | 2026-07-24 | Tooling; contents not investigated |

The NIXL + UCX pair is the standard stack for prefill/decode disaggregation and KV-cache transfer, so a disaggregated-inference direction is *plausible* — but these are **fork-creation events only, with zero Enflame commits visible**. Record as a weak signal; do **not** claim a shipped disaggregation capability.

Still actively pushed: `candle-gcu`, `candle-vllm-gcu`, `Ubridge`, `FFmpeg-GCU` (all 2026-07-17/18); `gcushare`, `gcu-exporter` (2026-06-29).

## Pre-baseline ecosystem datapoints (backfill, not this window)

- **2026-02-02:** Enflame and Biren were among the first domestic vendors to complete adaptation of StepFun's **Step 3.5 Flash**; L600 named.
- **PaddlePaddle FastDeploy** documents Enflame **S60 + ERNIE 4.5** under a unified GCU/GPU inference interface, pinning driver 1.5.0.5 / TopsRider 3.4.623 and a `topsrider3.5.102` container. That page dates to the 2025 ERNIE 4.5 open-source release, not this window.

## Ecosystem gaps — re-assessed 2026-08-08

| Baseline gap | Status |
|---|---|
| No public ISA documentation (vs NVIDIA PTX, Cambricon MLISA) | **Unchanged.** GCU-CARE / GCU-DARE remain undocumented |
| No open-source kernel library (vs CUTLASS, mlu-ops) | **Unchanged.** TopsBlasOps and the TopsCC back end remain closed; torch-gcu ships `csrc/aotfusion/` and `csrc/efficient_ops/` but not the kernel library |
| TopsTorch not open-sourced (vs torch_mlu) | **Partially closed.** A PyTorch backend source is public as `torch_gcu` — but single-commit, S60-only support matrix, and not linked by any source to the TopsTorch name |
| No academic papers on TopsCC internals | **Unchanged.** No Enflame paper or talk at ISCA 2026 or any other in-window venue |

## Confidence Assessment (2026-08-08 additions)

| Component | Confidence | Basis |
|-----------|------------|-------|
| torch-gcu exists, licence, creation date, `.version` | High | Repo + raw LICENSE/.version fetched directly |
| torch-gcu feature list (PrivateUse1, ECCL, torch.compile, GCUGraph, transfer_to_gcu, libtorch_gcu) | High | README fetched directly |
| torch-gcu Hardware Support = S60 only | High | README table |
| torch_gcu == renamed TopsTorch | **Not confirmed** — no source | — |
| torch-gcu as a *live* open-source project | Low | Single commit, no follow-on pushes, no coverage |
| TopsRider 3.7.x as current SDK line | High | README pin + Docker tag + `.version` |
| Doc-site slug (onlinedoc_dev_3.6 / 3.7) | **Not confirmed** | WAF 403 on all paths |
| vllm-gcu v0.11.0 date and contents | High | GitHub releases API with full bodies |
| Disaggregated-inference push from nixl/ucx forks | Low (weak signal) | Fork creation only, no commits |
