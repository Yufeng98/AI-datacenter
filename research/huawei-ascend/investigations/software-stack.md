# Huawei Ascend Software Stack Investigation

*as_of: 2026-04-05*
*chip: huawei-ascend*
*focus: CANN, ATC compiler, MindSpore, PyTorch Ascend (torch_npu)*

---

## Overview

The Huawei Ascend software stack is organized as **CANN** (Compute Architecture for Neural Networks) — a multi-layer SDK that spans from C-language runtime APIs through operator libraries, compiler toolchains, and high-level framework integrations. The stack has two primary entry points: **MindSpore** (Huawei's native framework) and **torch_npu** (the PyTorch Ascend adapter). Both ultimately call into CANN's AscendCL runtime, which dispatches to Da Vinci hardware via the driver layer.

```
User Code (Python/C++)
    ↓
MindSpore  OR  torch_npu (PyTorch PrivateUse1)  OR  ONNX Runtime CANN EP
    ↓
CANN Op Library (Ascend OL) — standard operators
    ↓
TBE / AscendC — kernel library (custom + fused ops)
    ↓
AscendCL (AscendCL Runtime API: device/stream/memory/model management)
    ↓
ATC Compiler (offline model → .om format)  ← used at compile time
    ↓
Ascend Driver (kernel module, firmware)
    ↓
Da Vinci AI Core Hardware
```

---

## 1. CANN — Compute Architecture for Neural Networks

CANN is the comprehensive SDK layer, analogous to NVIDIA's CUDA platform. It provides:

### 1.1 AscendCL (Ascend Computing Language Runtime)
- C-language API set for device management, context/stream creation, memory allocation/copy
- Model loading (`.om` files) and execution
- Operator loading and execution (custom operators)
- Media data processing (video decode, image pre-processing via DVPP hardware unit)
- Analogous roles: CUDA Runtime API (`libcudart`) + CUDA Driver API (`libcuda`)
- Python bindings available via `acl` Python wrapper

Key API categories:
- `aclrtCreateContext` / `aclrtSetDevice` — device context management
- `aclrtMalloc` / `aclrtMemcpy` — HBM memory management
- `aclrtCreateStream` / `aclrtSynchronizeStream` — stream and event management
- `aclmdlLoadFromFile` / `aclmdlExecute` — model loading and inference execution
- `aclopExecuteV2` — custom operator execution

### 1.2 CANN Versions and SOPH
- CANN has gone through significant versioning (100R020, 5.0.1, 6.x, 7.x, 8.x as of 2025)
- Each CANN release is tied to specific Ascend driver+firmware versions
- MindSpore and torch_npu releases carry matched CANN version requirements

---

## 2. ATC — Ascend Tensor Compiler (Offline Model Compiler)

### 2.1 Purpose and Role
- **Offline AOT compiler**: converts framework-exported model graphs to `.om` (Offline Model) format optimized for Ascend hardware
- Analogous to: TensorRT (NVIDIA), Neuron Compiler / neuronx-cc (AWS), GaudiAI Graph Compiler (Intel)

### 2.2 Supported Input Formats
| Input Format | Source |
|-------------|--------|
| `.pb` (TensorFlow SavedModel/frozen graph) | TensorFlow |
| `.caffemodel` + `.prototxt` | Caffe |
| `.onnx` | ONNX (from PyTorch, TF, etc.) |
| `.mindir` | MindSpore exported format |
| JSON (single operator descriptor) | Custom operator testing |

### 2.3 Optimization Passes
- **Operator fusion**: fuses Conv+BN+ReLU, MatMul+BiasAdd+Activation into single Cube+Vector compound ops
- **Quantization calibration**: INT8/INT4 post-training quantization; inserts `dequant`/`requant` nodes
- **Memory layout optimization**: converts NHWC ↔ NCHW ↔ Ascend-native NC1HWC0 5D layout to minimize L2 cache thrash
- **AIPP (AI Pre-Processing)**: hardware-accelerated image preprocessing pipeline baked into the model graph
  - Color space conversion (YUV→RGB, NV12→RGB8)
  - Mean/variance normalization
  - Crop and resize
- **SRAM placement**: schedules which tensors reside in L1/UB vs HBM to minimize MTE traffic

### 2.4 Output: `.om` Offline Model
- Self-contained binary: operator binaries + weights + execution graph + memory allocation plan
- Loaded at inference time via `aclmdlLoadFromFile()` — no online recompilation
- Enables deploy-once-run-many patterns analogous to TensorRT engines

### 2.5 ATC Command-Line Usage
```bash
atc --model=model.onnx \
    --framework=5 \          # 5 = ONNX
    --output=model_ascend \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --enable_small_channel=1
```

---

## 3. TBE — Tensor Boost Engine (Operator & Kernel Library)

### 3.1 Overview
TBE is the operator development and execution framework within CANN. It provides:
- **Standard operator library**: 1000+ pre-built ops (Conv2D, MatMul, BatchNorm, LSTM, Attention, etc.) optimized for Da Vinci hardware
- **Custom operator development**: two authoring modes

### 3.2 TBE-DSL (Domain-Specific Language)
- High-level Python API for defining operator computation logic
- Automatic scheduling: TBE compiler determines tiling, double-buffering, instruction ordering
- Suitable for new ops with regular computation patterns
- Less control but faster development

### 3.3 TBE-TIK (Tensor Iterator Kernel)
- Python DSL with explicit Da Vinci hardware control
- Developer specifies: tensor tiling, L1/UB buffer allocation, MTE transfer schedules, Cube/Vector calls
- Fine-grained performance optimization; analogous to CUDA PTX-level authoring
- Instruction primitives: `tik.Tensor`, `data_move()`, `matmul()`, `vconv()`, `vadd()`

### 3.4 AscendC Programming Language
- Announced 2023; C/C++-compatible language for Da Vinci kernel development
- Provides hardware abstractions: `AIC` (AI Core), `AIV` (AI Vector) execution contexts
- Explicit control of Cube, Vector, MTE1, MTE2 units
- Eliminates manual synchronization barriers between hardware units (compiler-managed)
- Closer to CUDA C++ than TIK Python in terms of language familiarity
- Key programming abstractions:
  - `GlobalTensor` / `LocalTensor` — typed tensor descriptors for HBM vs on-chip memory
  - `DataCopy()` — MTE-backed DMA with explicit source/dest memory spaces
  - `Matmul()` — Cube unit invocation with L0A/L0B/L0C buffer specification
  - `pipe.InitEventID()` / `EnQue()` / `DeQue()` — double-buffer pipeline synchronization
- TileLang-Ascend provides MLIR-based compilation targeting AscendC from Python tile specs

---

## 4. MindSpore Framework

### 4.1 Architecture
- **Source-to-source (S2S) automatic differentiation**: Python functions are transformed into computational graphs via abstract syntax tree tracing; analogous to JAX's `jax.jit` JIT tracing
- Modes: `GRAPH_MODE` (full graph compilation, static shapes, optimized) and `PYNATIVE_MODE` (eager, for debugging)
- Unified deployment target: cloud (Ascend), GPU (CUDA), CPU, mobile (HiSilicon NPU/Kirin)

### 4.2 Compilation Pipeline
```
Python model code (nn.Cell subclass)
    ↓ MindSpore graph tracer
MindIR (MindSpore Intermediate Representation)
    ↓ MindSpore Graph Compiler
GE (Graph Engine) — graph fusion, layout optimization
    ↓
CANN Op Library / TBE operators
    ↓
AscendCL runtime dispatch
    ↓
Da Vinci hardware
```

### 4.3 Key Features
- **Auto-parallelism**: `mindspore.set_auto_parallel_context()` enables automatic data-parallel, model-parallel, or pipeline-parallel strategies without user annotation
- **Hybrid parallelism**: supports tensor parallel, pipeline parallel, expert parallel (MoE) for large model training
- **Dynamic shapes**: limited support in GRAPH_MODE; full support in PYNATIVE_MODE
- **MindFormers**: production library for transformer model training (BERT, GPT, LLaMA, Qwen, etc.) on Ascend clusters
- **MindSpeed**: accelerated training library for large models; analogous to Megatron-LM on CUDA

### 4.4 Distributed Training on Ascend
- Uses `mindspore.communication.init()` to initialize HCCL (Huawei Collective Communications Library)
- HCCL handles AllReduce, AllGather, ReduceScatter across HCCS-connected NPUs
- Supports both intra-server (HCCS) and inter-server (RoCE v2 / HCCL-RDMA) communication

---

## 5. torch_npu — PyTorch Ascend Adapter

### 5.1 Architecture
- Implements PyTorch's **PrivateUse1** backend extension mechanism
- Registers `torch.npu` as a device type alongside `torch.cuda`, `torch.xla`
- Provides: `torch.npu.device()`, `torch.npu.synchronize()`, `torch.npu.memory_allocated()`
- Operator dispatch: ATen operators → custom Ascend kernel implementations or CANN standard operators
- Installation: `pip install torch-npu` (matched to PyTorch version)

### 5.2 Integration Depth
- **torch.compile** support: torch_npu can serve as a torch.compile backend (experimental)
- **torch.nn.functional**: standard PyTorch ops dispatch through to CANN; no code changes needed
- **FSDP / DTensor**: supported for large model distributed training (via PyTorch distributed + HCCL backend)
- **Torchtune integration** (Jan 2025): full LLM fine-tuning support on Ascend via torch_npu

### 5.3 Supported PyTorch versions
- Maintained with LTS support branches matching major PyTorch releases (2.1, 2.3, 2.4, 2.5)
- Active development on Gitee (`gitee.com/ascend/pytorch`) with GitHub mirror

---

## 6. Additional Framework Integrations

| Framework | Integration | Notes |
|-----------|-------------|-------|
| ONNX Runtime | CANN Execution Provider | C/C++ + Python; model runs via AscendCL on Ascend |
| OpenCV | CANN Backend | Hardware-accelerated vision preprocessing via DVPP |
| vLLM | Ascend backend (community) | Inference serving with torch_npu backend |
| DeepSpeed | Ascend plugin | ZeRO + HCCL; via torch_npu distributed backend |
| HuggingFace Transformers | Ascend via torch_npu | `device="npu"` device map |

---

## 7. HCCL — Huawei Collective Communications Library

- Analogous to NCCL (NVIDIA) or RCCL (AMD)
- Provides: AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv
- Topology-aware: selects HCCS (intra-server high-bandwidth) vs RoCE (inter-server) transport
- Integrated with PyTorch distributed via `torch.distributed.init_process_group(backend="hccl")`
- Integrated with MindSpore via `mindspore.communication.init(backend_name="hccl")`
- MindSpore 1.1+ supports distributed training on Ascend with HCCL

---

## 8. Programming Model Rationale

**1. Static CCE scheduling requires compiler awareness.** Unlike GPU SIMT (where hardware dynamically schedules warps), the Da Vinci AI Core has no dynamic out-of-order execution. The CCE compiler must statically schedule Cube, Vector, MTE1, and MTE2 instructions to keep all five units busy simultaneously. This is why TBE and AscendC expose explicit pipeline APIs (`EnQue`/`DeQue` for double-buffering) — the programmer or compiler must express the overlap between data movement and computation.

**2. The memory hierarchy demands explicit tiling.** The Cube unit requires data in L0A/L0B; the UB serves Vector; L1 is a staging buffer. A kernel must tile the input tensors into L1-fitting chunks, then sub-tile those into L0-fitting matrix panels. TBE-TIK and AscendC both expose these memory levels as named address spaces, making the tiling obligation visible.

**3. ATC offline compilation avoids runtime JIT overhead.** Unlike CUDA's fat binary + PTX JIT fallback, Ascend's production deployment model compiles models to `.om` at deployment time (via ATC), eliminating JIT latency at inference. The tradeoff is that dynamic shapes require re-compilation or shape-range specification at ATC time.

**4. MindSpore S2S differentiation targets Ascend-specific graph passes.** The S2S AD approach allows the MindSpore compiler to see the full computational graph including backward pass, enabling cross-forward-backward fusion and memory-saving recompute passes that CANN can lower directly to efficient HCCS-aware collective schedules.

**5. torch_npu PrivateUse1 enables zero-code-change migration.** By implementing the PrivateUse1 backend, Huawei ensures that PyTorch models written for CUDA can run on Ascend by simply changing `device="cuda"` to `device="npu"` — the same strategy used by Intel Gaudi (HPU) and AWS Neuron (PrivateUse1 planned). This lowers the adoption barrier for the massive PyTorch ecosystem.

---

## Sources
- [CANN Documentation Portal](https://support.huawei.com/enterprise/en/ascend-computing/cann-pid-251168373)
- [ATC Tool Documentation](https://support.huawei.com/enterprise/en/doc/EDOC1100192457/2a40d134/atc-tool)
- [TBE-TIK Introduction](https://support.huawei.com/enterprise/en/doc/EDOC1100164820/83779921/tik-introduction)
- [AscendC Launch — Huawei Central](https://www.huaweicentral.com/huawei-ascend-c-programming-language-launched/)
- [Ascend/pytorch — GitHub](https://github.com/Ascend/pytorch)
- [PyTorch Torchtune + Ascend Blog](https://pytorch.org/blog/ascend-backend-w-torchtune/)
- [MindSpore Overview](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html)
- [MindSpore White Paper (PDF)](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/white_paper/MindSpore_white_paper_enV1.1.pdf)
- [MindSpore Distributed Training on Ascend — Gitee](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_en/advanced_use/distributed_training_ascend.md)
- [ONNX Runtime CANN EP](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)
- [Parallel Scan on Ascend — arXiv](https://arxiv.org/html/2505.15112v1)

---

# Investigation Update — CANN 9.0.0 / MindSpore 2.9.0 / vllm-ascend / CANN Open Source (2026-08-08)

*as_of: 2026-08-08*
*chip: huawei-ascend*
*focus: CANN 9.0.0 commercial release, MindSpore 2.9.0, AscendC SIMD+SIMT, HCCL group API, vllm-ascend, CANN open-sourcing*
*scan window: 2026-04-05 (prior baseline) → 2026-08-08*

## U.1 CANN 9.0.0 (commercial) — released 2026-05-09

Betas: 2026-03-09 and 2026-03-30. This is the first CANN release with first-class Ascend 950 support. The release notes state verbatim:

> 新增适配Ascend 950PR（Atlas 350加速卡）
> ("newly adds support for Ascend 950PR (Atlas 350 accelerator card)")

### U.1.1 Data types

FP8, MXFP8, and MXFP4 added. HiF8 (Huawei-proprietary 8-bit) is part of the 950 precision set. This is the software delivery of the precision formats the repo previously recorded only as roadmap items.

### U.1.2 AscendC — SIMD+SIMT hybrid programming

Release notes: **AscendC支持SIMD+SIMT混合编程**.

- Roughly **700 new SIMT API entry points**, grouped as warp, atomic, math, and type-conversion operations.
- A **Reg-based (register-level) programming path** exposed specifically on Ascend 950PR.

Important framing: **Da Vinci 4.0's SIMD+SIMT hybrid execution model was already recorded in this repo at the 2026-04-05 baseline as an architectural fact.** What is new here is its *software delivery* in CANN 9.0.0 — the API surface, not the architecture.

This is also the first concrete change to the repo's "Programming Model Rationale" item #1. The five-stream static-CCE model with `EnQue`/`DeQue` pipeline expression is still the documented AscendC idiom; the SIMT APIs sit alongside it rather than replacing it. (A third-party whitepaper analysis claims **BufferID-based synchronization replaces the EnQue/DeQue handshake** on Da Vinci 4.0 — that claim is **single-source and unconfirmed**; see the hw-architecture investigation U.2.3.)

### U.1.3 HCCL

- **Batched-collective merge** via `HcclGroupStart()` / `HcclGroupEnd()` — lets multiple collectives be submitted as one batch.
- **集合通信支持CCU通信加速** — collective communication accelerated through the on-chip CCU (Collective Communication Unit), which appears in the Ascend 950 STARS 2.0 scheduler's arbitration list.

### U.1.4 CATLASS

The CATLASS template library was upgraded for **950-series tensor/vector co-compute paths** and **new Fixpipe move modes** — consistent with the AIC/AIV core split on the hardware side, where Cube and Vector are now separate cores that must exchange data explicitly.

### U.1.5 Packaging

Install methods restructured: **apt and pip added**, so the set is now conda / yum / apt / pip, with a unified download bundle. Huawei markets this as cutting deployment time from ~2 h to ~45 min — a **vendor claim**, not an independently measured figure.

Historical note carried from the MindSpore v2.7.2 release notes: *"CANN 8.5.0 package has completed the upgrade to an open-source architecture"*, which changed CANN naming conventions and install directories. That is the packaging change that preceded the GitCode open-sourcing described in U.4.

### U.1.6 Version drift already visible

**vllm-ascend v0.23.0rc1 requires CANN 9.0.1** for Ascend 950 — CANN has already moved past 9.0.0 within the scan window. Treat 9.0.0 as the notable *feature* release, not as "current".

## U.2 MindSpore 2.9.0 — released 2026-05-07

- Paired with CANN 9.0.0.
- Published on **mindspore.cn and PyPI** — these are the canonical version sources.
- **Native Ascend 950PR support.**

**Sourcing note for future scans:** the Gitee mirror's release list was stale, showing only up to v2.7.2 (2026-01-16, paired with CANN 8.5.0). An earlier internal note hedged 2.9.0 to "medium confidence" on the strength of that stale mirror and dated the release 2026-05-09 from a personal install blog. Both were wrong: 2.9.0 is **confirmed**, released **2026-05-07**. Prefer mindspore.cn / PyPI over the Gitee release list.

## U.3 vllm-ascend — the fastest-moving Ascend software surface

Not previously covered anywhere in this repo. vllm-ascend is where Ascend 950 enablement lands first.

| Release | Date | Ascend-relevant content |
|---|---|---|
| v0.18.0, v0.19.1rc1 | 2026-04-30 | baseline 950-era branches |
| v0.20.2rc1 | 2026-06-03 | upgrades to **CANN 9.0.0**; **MXFP4 quantization on Ascend 950**; DeepSeek-V4 DSA attention |
| v0.21.0rc1 | 2026-06-16 | **full end-to-end DeepSeek-V4 on Ascend 950**; `FULL_AND_PIECEWISE` hybrid graph compilation mode |
| v0.22.1rc1 | 2026-06-30 | Ascend 950 **dynamic quantization**; Mooncake connector for DeepSeek-V4 hybrid KV cache; **HCCL weight transfer for RL** |
| v0.23.0rc1 | **2026-07-19** | aligns with upstream vLLM v0.23; **W4A16 MXFP4** and **all-gather-EP MXFP4**; **requires CANN 9.0.1** for Ascend 950 |

The graph-compilation mode (`FULL_AND_PIECEWISE`) and the RL weight-transfer path are the two items most relevant to this survey's programming-model framing: they show the Ascend serving stack converging on the same hybrid-capture / disaggregated-KV patterns as the CUDA vLLM path, rather than diverging.

## U.4 CANN open-sourcing is live — supersedes the repo's "CANN internals are closed" note

`gitcode.com/cann` is now a live multi-repo organization with **~76 projects**, actively committed through early August 2026:

| Category | Repos |
|---|---|
| Operator libraries | `ops-transformer`, `ops-math`, `ops-nn`, `ops-cv` |
| Communication | `hccl`, `hcomm`, `shmem` |
| Kernel dev | `asc-devkit` (Ascend C dev kit) |
| Python tooling | `pypto`, `pyasc` |
| Runtime / graph | `runtime`, `ge` (graph engine), `graph-autofusion` |
| Recipes / bench | `cann-recipes-infer`, `cann-recipes-train`, `cann-bench` |
| Quantization | `amct` |

**Scope of the opening:** userspace layers only. The **kernel driver and on-chip firmware remain closed**. The repo's Driver/Firmware layer entries stay accurate; the Op Library / Runtime / Communication layer entries that described CANN internals as closed are superseded.

This materially changes the survey's "openness" characterization of Ascend: the operator libraries, collective library, graph engine, and runtime are now inspectable, which puts Ascend closer to the AMD ROCm end of the spectrum than the fully-closed position it previously occupied.

## U.5 Impact on the repo's Programming Model Rationale items

| Item | Status after this update |
|---|---|
| #1 Static CCE scheduling demands compiler-visible pipeline (`EnQue`/`DeQue`) | **Still stands.** SIMT APIs are added alongside, not instead. The single-source BufferID claim, if ever confirmed, would revise this — flagged, not applied. |
| #2 5-level memory hierarchy constrains tile sizes | **Partly outdated for the 950.** The 950 adds a 128 MB global L2 (a sixth level) and per-core buffer sizes are not disclosed. The 16 KB L0A/L0B reasoning in the text is 910-era and should be read as such. |
| #3 ATC offline compilation decouples deployment from JIT | Unchanged. |
| #4 MindSpore S2S AD enables cluster-aware optimization | Unchanged. |
| #5 torch_npu PrivateUse1 is the ecosystem bridge | **Extend:** vllm-ascend is now the highest-velocity consumer of that bridge. |
| #6 CloudMatrix 384's all-optical fabric reflects HCCS bandwidth ceiling | Unchanged for 910C; the 950 generation's fabric is UnifiedBus 2.0. |

## U.6 Sources added 2026-08-08

- CANN 9.0.0 commercial release notes: https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md
- CANN open-source organization (GitCode): https://gitcode.com/cann
- MindSpore release notes (canonical): https://www.mindspore.cn/docs/en/stable/RELEASE.html
- MindSpore 2.9 version notes: https://www.mindspore.cn/version-updates/en/2_9_en
- MindSpore on PyPI: https://pypi.org/project/mindspore/
- MindSpore Gitee releases (stale mirror — do not use as canonical): https://gitee.com/mindspore/mindspore/releases
- vllm-ascend release notes: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html
- vllm-ascend GitHub releases: https://github.com/vllm-project/vllm-ascend/releases
- CANN 9.0.0 / MindSpore 2.9.0 upgrade writeup (community, OrangePi AIpro 20T): https://www.donaldsebleung.com/blog/20260509-upgrading-to-mindspore-290-plus-cann-900-on-the-orangepiaipro-20t
- CANN component coverage (CSDN): https://blog.csdn.net/gitblog_00311/article/details/160916818
