# Cambricon MLU Software Stack — Investigation Report

*resource: https://github.com/Cambricon/torch_mlu + https://www.cambricon.com/docs/bangc/developer_guide_html/ + https://www.cambricon.com/docs/cnrt/user_guide_html/index.html*
*as_of: 2026-04-05*
*chip: cambricon*
*layer: Framework Integration, Compiler / IR, Op Library, Kernel Library, Runtime, Driver / Firmware, Communication, Assembler / ISA*

---

## Overview

Cambricon's software ecosystem is called **NeuWare** (Neuware SDK / CNToolkit). It is Cambricon's answer to NVIDIA's CUDA platform — a vertically integrated stack from Python-level framework bindings through compilers, op libraries, runtimes, and down to the kernel driver. The stack was substantially rebuilt starting with the MLU370 generation and the CNToolkit v3.x release, replacing the older CNML-centric API with a CNNL-centric approach.

The primary programming language for writing custom MLU kernels is **BANG C** — a C/C++ extension (analogous to CUDA C) compiled by **CNCC** (Cambricon Neuware C/C++ Compiler). The full stack parallels CUDA with clear component-to-component analogies:

| CUDA/NVIDIA | Cambricon NeuWare | Role |
|------------|-------------------|------|
| CUDA C/C++ | BANG C | Kernel programming language |
| nvcc | CNCC | Kernel compiler |
| PTX / SASS | MLISA (.cncode/.cnbin) | Machine/virtual ISA |
| cuDNN | CNNL | Op library |
| cuBLAS | CNNL (matrix ops) | GEMM/BLAS ops |
| CUTLASS | mlu-ops / BANGC OPS | Kernel templates |
| NCCL | CNCL | Collective communication |
| libcudart | CNRT | Runtime API |
| libcuda | CNDrv | Driver API |
| nvidia.ko | cambricon-mlu driver | Kernel driver |

---

## 1. Framework Integration

### torch_mlu (Official PyTorch Backend)

`torch_mlu` is Cambricon's official PyTorch extension that registers the MLU as a PyTorch device backend. It is versioned as `{torch_mlu_ver}+torch{pytorch_ver}` (e.g., `1.17.0+torch2.1.0`).

Dependencies: CNToolkit (CNCC, CNAS, CNDrv), CNNL, CNCL, CNCV, BANGC OPS, DALI (optional), CNNL_Extra.

Key integration points:
- Registers MLU device in PyTorch's device dispatch table
- Maps PyTorch ATen operators to CNNL or BANGC OPS implementations
- Exposes `torch.device("mlu")`, `tensor.to("mlu")`, `model.mlu()` APIs
- Supports PyTorch Distributed with CNCL as the communication backend
- Supports `torch.compile` via graph capture and dispatch to MagicMind or BANGC OPS

The earlier **CATCH** (Cambricon Adaptive Tool Chain for Hardware) repository (`Cambricon/catch`) was the predecessor integration layer for PyTorch; superseded by torch_mlu.

### vllm-mlu (LLM Inference Serving)

Cambricon's fork of vLLM adapted for MLU hardware. Supports:
- Chunk Prefill, Prefix Caching, Speculative Decoding, Graph Mode (CUDA Graph analog)
- Requires SDK 25.08+, MLU370 or newer hardware
- Custom Paged Attention and KV cache ops implemented in BANG C
- Upstream vLLM integration PR #10315 adds MLU as a first-class backend

### PaddlePaddle

Two integration paths:
1. **PaddleCustomDevice** (`PaddlePaddle/PaddleCustomDevice`): Plugin loading `libpaddle-custom-mlu.so` with 264+ custom operators for Cambricon MLU. Supports MLU370-X8.
2. **PaddleX**: Higher-level pipeline framework with Cambricon MLU inference across vision and NLP models.

### Other Frameworks

- **MXNet** (historical): CNML/CNRT integration documented in Apache MXNet wiki
- **Geneformer** (bioinformatics): Case study of porting transformer model to MLU (ACM DL 2024)

---

## 2. Compiler / IR

### BANG C Language and CNCC Compiler

**BANG C** is a C/C++ dialect extended with BANG-specific keywords and built-in functions for MLU kernel programming. Key extensions:

| Extension | Description |
|-----------|-------------|
| `__mlu_global__` | Kernel function qualifier (analogous to CUDA `__global__`) |
| `__mlu_shared__` | Variables in cluster shared SRAM |
| `__nram__` | Variables in core-private NRAM |
| `__wram__` | Variables in core-private WRAM |
| `__mlu_builtin_*` | Intrinsic functions for vector/matrix ops |
| `__sync_cluster()` | Intra-cluster barrier (all 4 cores) |
| `__sync_all()` | Full-chip barrier |

**CNCC** (Cambricon Neuware C/C++ Compiler) compiles `.mlu` source files:
1. Parses BANG C dialect
2. Applies optimizations: vectorization, scratchpad allocation, DMA scheduling
3. Emits MLISA assembly (`.mlisa` text format)
4. CNAS (Cambricon Neuware Assembler) assembles to binary: `.cncode`, `.cnbin`, `.cnfatbin`

The compilation pipeline is analogous to `nvcc → PTX → SASS`:
```
.mlu source
  └─ CNCC (compiler)
      └─ MLISA assembly (.mlisa)
          └─ CNAS (assembler)
              └─ .cnbin / .cnfatbin (binary)
                  └─ CNRT (runtime loader)
                      └─ CNDrv (driver JIT / load)
```

### MagicMind (MLIR-Based Inference Compiler)

MagicMind is Cambricon's inference-specific compiler engine. It accepts trained models from multiple frameworks:
- **Input**: TensorFlow SavedModel, PyTorch TorchScript, ONNX, Caffe
- **IR**: Converts to a unified MagicMind computation graph (MLIR-based internal representation)
- **Optimization**: Graph fusion, constant folding, quantization (INT8/FP16), layout optimization
- **Output**: Compiled MLU binary for deployment (analogous to TensorRT engines)
- **Repository**: `Cambricon/magicmind_cloud` (open cloud examples + usage guides)

### triton-linalg

`Cambricon/triton-linalg` converts the Triton MLIR dialect to the Linalg dialect, enabling Cambricon-specific backend lowering from Triton kernels. Status as of 2026: frontend conversion complete; backend compilation to MLISA still under active development. This is the path toward allowing PyTorch Triton kernels to run on MLU hardware.

---

## 3. Op Library

### CNNL (Cambricon Neuware Neural Network Library)

CNNL is the primary high-level operator library for MLU, analogous to cuDNN + cuBLAS:

- Provides optimized implementations of: GEMM, Conv2D, BatchNorm, LayerNorm, Attention (scaled dot-product), Pooling, RNN, Elementwise ops
- Versioned and shipped as part of CNToolkit
- C API: `cnnlConvolutionForward`, `cnnlBatchNormForward`, `cnnlMatMul`, etc.
- Used internally by torch_mlu for ATen operator dispatch
- **CNNL_Extra**: Extended operator library for additional ML ops not in core CNNL

### CNML (Legacy)

CNML (Cambricon Neuware Machine Learning Library) was the predecessor to CNNL, documented in the CNML Developer Guide v7.10.2 (April 2021). It used a graph-based API rather than operator-by-operator dispatch. Still referenced in older integrations (MXNet) but superseded by CNNL in current NeuWare SDK.

---

## 4. Kernel Library

### mlu-ops / BANGC OPS

`Cambricon/mlu-ops` is the open-source kernel library — the closest analog to CUTLASS/CUB for Cambricon. It provides:
- Efficient BANG C implementations of hundreds of ML operators
- Integration helpers for calling CNNL from custom kernels
- Reference implementations for custom op development
- Used as a dependency by torch_mlu for operators not in CNNL

### BANGPy

BANGPy is a Python-level kernel programming interface (analogous to Triton in spirit). It provides a Python API for writing MLU kernels at graph/kernel/operator optimization level, enabling Python-native custom kernel development without writing raw BANG C. Part of NeuWare SDK.

### CNCV (Cambricon Computer Vision Library)

CNCV provides accelerated computer vision primitives (image decode, resize, color convert, normalize) on MLU hardware. A direct dependency of torch_mlu for vision preprocessing pipelines.

---

## 5. Runtime

### CNRT (Cambricon Neuware Runtime)

CNRT (`libcnrt.so`) is the primary user-facing runtime API — the MLU analog of CUDA Runtime (`libcudart.so`):

| CNRT API Category | Examples |
|-------------------|---------|
| Device management | `cnrtGetDeviceCount`, `cnrtSetDevice`, `cnrtDeviceGetInfo` |
| Memory | `cnrtMalloc`, `cnrtFree`, `cnrtMemcpy`, `cnrtMemset` |
| Queue (async) | `cnrtQueueCreate`, `cnrtQueueSync`, `cnrtQueueDestroy` |
| Notification (event) | `cnrtNotifierCreate`, `cnrtPlaceNotifier`, `cnrtWaitNotifier` |
| Kernel launch | `cnrtInvokeKernel` (with dim3 task config, kernel fn, params, queue) |
| Module loading | `cnrtLoadKernelParamsBuffer`, runtime module management |

Key CNRT concepts:
- **Queue**: Ordered sequence of asynchronous MLU operations (analogous to CUDA Stream). Kernels, memcpies, and notifiers are enqueued and execute in order.
- **Notifier**: Lightweight event for cross-queue synchronization (analogous to CUDA Event).
- **cnrtDim3_t**: 3D task configuration specifying kernel parallelism across X, Y, Z dimensions — maps to the number of MLU Cores to use.

CNRT provides a **mixed-call model** with CNDrv: a programmer can use CNRT for high-level resource management and CNDrv for fine-grained control, similar to the CUDA Runtime/Driver API relationship.

### CNDrv (Cambricon Driver API)

CNDrv (`libcndrv.so`) is the low-level driver API, analogous to CUDA Driver API (`libcuda.so`):
- Explicit context and device management
- Module and function loading from `.cnbin` files
- Fine-grained memory management
- Rust bindings available via `cndrv` crate on crates.io

---

## 6. Driver / Firmware

### cambricon-mlu kernel driver

The Linux kernel module for Cambricon MLU devices:
- PCIe BAR mapping and device enumeration
- IOCTL dispatch for CNRT/CNDrv operations
- DMA engine control for host-device memory transfers
- Interrupt handling
- Available as an open mirror on Gitee (`jimpirror/cambricon-mlu-driver`)

There is no documented equivalent to NVIDIA's GSP firmware (on-GPU RISC-V microcontroller for Resource Manager). The Cambricon driver uses a more traditional host-managed driver model.

### hl-smi equivalent (cn-smi)

Cambricon provides system management tools for MLU monitoring: temperature, utilization, memory usage, and health status. The `mlu-exporter` repository (`Cambricon/mlu-exporter`) provides a Prometheus metrics exporter for Kubernetes monitoring.

### Kubernetes Device Plugin

`Cambricon/cambricon-k8s-device-plugin` is a DaemonSet that automatically reports MLU device count, health, and resource availability to the Kubernetes scheduler, enabling containerized workload deployment on MLU hardware.

---

## 7. Communication

### CNCL (Cambricon Collective Communication Library)

CNCL is Cambricon's NCCL analog for intra-MLU collective operations:

- **Operations**: AllReduce, AllGather, ReduceScatter, Broadcast
- **Transport**: MLU-Link for intra-node chip-to-chip; standard network for inter-node
- **Integration**: torch_mlu exposes CNCL as the `torch.distributed` communication backend
- **Heterogeneous**: KAITIAN framework uses CNCL for MLU-internal collectives and Gloo for cross-vendor (e.g., MLU + GPU) heterogeneous training

Per the KAITIAN paper (arXiv 2505.10183), CNCL achieves efficient MLU-to-MLU AllReduce over MLU-Link, while cross-vendor heterogeneous collectives rely on CPU-mediated Gloo paths.

---

## 8. Assembler / ISA

### BANG C / MLISA

The BANG ISA for neural network accelerators was described in the original Cambricon ISA paper (ISCA 2016):
- 43 instruction types, all 64-bit
- Categories: Scalar, Vector, Matrix, Control
- Load-store architecture: no implicit register-to-register ops across memory types
- 64 × 32-bit GPRs per Core (scalar/control)
- No separate vector register file — operands reside in NRAM/WRAM scratchpad

**Compilation pipeline:**
```
BANG C source (.mlu)
  └─ CNCC → MLISA assembly (.mlisa)
      └─ CNAS → binary (.cncode / .cnbin / .cnfatbin)
```

`.cnfatbin` is Cambricon's fat binary format — bundles binaries for multiple MLU generations (analogous to NVIDIA fatbinary).

### Extended ISA (ACM 2019)

The extended journal version (ACM 2019, doi 10.1145/3331469) covers:
- ISA design rationale: higher code density than MIPS/x86/GPGPU for NN workloads
- Data type support evolution: FP32, FP16, INT8, INT4
- Comparison with GPU SIMD: domain-specific ops encode complex NN patterns in single instructions

---

## Programming Model Rationale

**1. Explicit memory management is the performance contract.** Unlike GPU SIMT where cache hierarchies hide latency implicitly, BANG C requires explicit DMA calls to move data between GDRAM and NRAM/WRAM. This means performance is predictable and deterministic — there are no cache miss penalties — but the programmer or compiler must reason about all data movement explicitly.

**2. CNNL hides the complexity for standard ops.** The vast majority of DNN workloads can be expressed as CNNL op calls from torch_mlu, which internally manage all NRAM/WRAM tiling, DMA scheduling, and core synchronization. BANG C custom kernels are needed only for non-standard ops or research workloads.

**3. The Queue/Notifier model enables async overlap.** Similar to CUDA Streams, CNRT Queues allow the host to submit work asynchronously, overlap CPU compute with MLU DMA, and pipeline kernel execution. The producer-consumer DMA overlap pattern (load next tile while computing current tile) is the standard optimization for memory-bound MLU kernels.

**4. MagicMind trades programmability for peak inference efficiency.** For deployment, MagicMind's graph-level optimizations (fusion, quantization, layout optimization) can achieve substantially better latency than framework-level execution, at the cost of a one-time compilation step. This matches the TensorRT model for NVIDIA.

---

## Resources

- [torch_mlu GitHub](https://github.com/Cambricon/torch_mlu)
- [BANG C Developer Guide (online, v2.15.0)](https://www.cambricon.com/docs/bangc/developer_guide_html/)
- [BANG C Language Guide v2.4.1 (PDF)](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf)
- [CNRT User Guide v6.0.0](https://www.cambricon.com/docs/cnrt/user_guide_html/index.html)
- [CNToolkit Installation Guide v3.7.2](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
- [mlu-ops GitHub](https://github.com/Cambricon/mlu-ops)
- [MagicMind Cloud GitHub](https://github.com/Cambricon/magicmind_cloud)
- [triton-linalg GitHub](https://github.com/Cambricon/triton-linalg)
- [vllm-mlu GitHub](https://github.com/Cambricon/vllm-mlu)
- [CNML Developer Guide v7.10.2 (PDF)](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf)
- [KAITIAN Communication Framework (arXiv 2505.10183)](https://arxiv.org/html/2505.10183v1)
- [Cambricon ISA paper (ISCA 2016)](https://ieeexplore.ieee.org/document/7551409/)
- [Cambricon ISA extended (ACM 2019)](https://dl.acm.org/doi/fullHtml/10.1145/3331469)
- [Cambricon Developer Portal](https://developer.cambricon.com/)

---

# Investigation Update — 2026-08-08

*resource: https://developer.cambricon.com/index/article/details.html?id=12 + https://raw.githubusercontent.com/Cambricon/vllm-mlu/master/README.md + https://raw.githubusercontent.com/Cambricon/mlu-ops/master/README.md + https://api.github.com/orgs/Cambricon/repos?sort=pushed&direction=desc*
*as_of: 2026-08-08*
*chip: cambricon*
*layer: Framework Integration, Compiler / IR, Op Library, Kernel Library, Runtime, Driver / Firmware*

## 1. Framework Integration — day-0 DeepSeek-V4 (2026-04-24)

This is the one unambiguously in-window software event. The `vllm-mlu` README changelog carries exactly one entry:

```
[2026.04.24] vllm_mlu day0支持DeepSeek-V4
```

Cambricon's own developer article of the same date confirms **Day-0 adaptation of DeepSeek-V4-flash (285B) and DeepSeek-V4-pro (1.6T)** and names the concrete enabling work:

| Capability | Detail |
|---|---|
| Hybrid parallelism | Five-dimensional: **TP / PP / SP / DP / EP** |
| Overlap | Communication–computation overlap |
| Quantization | Low-precision quantization (formats not enumerated in the article) |
| Serving topology | **PD-separated deployment** — prefill/decode disaggregation |
| Fused ops | **Torch-MLU-Ops** accelerating the DeepSeek-V4 **Compressor** and **mHC** modules |
| Hand-written kernels | **BANG C** kernels for sparse/compressed attention and **GroupGemm** |

The article lists **MLU370-S4 / X4 / X8** as the board options — i.e. this work runs on the MLU370 generation, not on an unannounced part.

Cambricon's 2026 half-year filing independently corroborates PD separation (预填充与解码分离) and DeepSeek Day-0 adaptation in generic terms, but **never names Torch-MLU-Ops or vllm-mlu**.

### Correction to a claim about vllm-mlu staleness

An earlier scan asserted that the repo's `vllm-mlu` description was stale for listing Chunk Prefill / Prefix Caching / Speculative Decoding / Graph Mode and an MLU370+ hardware requirement. That assertion is **wrong**: the current master README still lists exactly those features (plus **Sleep Mode**) and still states `MLU370以上的设备`. The only genuinely missing item was the DeepSeek-V4 changelog entry, now recorded.

A separate upstream documentation PR, vllm-project/vllm **#25942 "[Doc] Add Cambricon MLU support"**, merged 2025-09-30. It pre-dates the repo baseline and is distinct from the older PR #10315 already cited in this survey.

## 2. Op Library — Torch-MLU-Ops (new to this survey, not new to the world)

**Torch-MLU-Ops** (`torch_mlu_ops`) is Cambricon's self-developed **high-performance fused operator library for PyTorch**, sitting alongside CNNL and mlu-ops. It is the layer through which vLLM, TGI and Stable Diffusion WebUI reach fused MLU kernels, and the layer through which the DeepSeek-V4 Compressor and mHC modules were accelerated.

| Attribute | Value |
|---|---|
| Version observed | v1.3.2, released **2026-02-04** |
| Declared dependencies | Torch-MLU **v1.24.1**, CNNL **v1.28.3** |
| Known consumers | vLLM, TGI, Stable Diffusion WebUI |
| Distribution | Licensed SDK bundle; no Cambricon-owned public source repository found (a third-party mirror exists) |

**Important framing correction.** This component is new *to this survey*, not new *in this window*. v1.3.2 shipped 2026-02-04 — before the repo's 2026-04-05 baseline — and already listed those three consumers. It must not be written up as "the software stack picked up a new named component" in the April–August 2026 window.

## 3. Runtime / Toolchain — version floor moves to CNToolkit 4.1.0

The actively maintained `mlu-ops` master README states the current build and runtime requirement as:

- **CNToolkit ≥ v4.1.0**
- **CNNL ≥ v1.28.0**
- **MLU driver ≥ v6.0.3**

This supersedes the survey's prior "CNToolkit v3.7.2 / SDK 1.15.0", which was already years stale (the 3.7.2 manual is dated 2023-10-18). An intermediate **CNToolkit 3.8.4** documentation tree exists and is search-indexed, but its release-note page returns HTTP 401; recording 3.8.4 as "current" would understate the toolchain materially. The **SDK bundle number** corresponding to CNToolkit 4.1.0 is **not disclosed** — do not invent one. `mlu-ops` v1.4.0 release notes already referenced a toolkit update to 4.0, corroborating the 4.x line.

## 4. Compiler / IR — triton-linalg is quiet in public

`Cambricon/triton-linalg` has had **no push since 2025-02-07**. The survey's previous "backend compilation to MLISA under development" should be read as *stalled in public or moved into the licensed SDK*, not as visibly in flight. Countervailing evidence that Triton work continues internally: Cambricon's H1 filing states the company "持续跟进 Triton 社区版本更新" (continues to track Triton community releases).

## 5. Open-source activity picture (GitHub API, retrieved 2026-08-08)

| Repository | Last push | Read |
|---|---|---|
| `mlu-ops` | 2026-08-07 | Actively maintained |
| `cambricon-k8s-device-plugin` | 2026-08-07 | Actively maintained |
| `mlu-exporter` | 2026-08-07 | Actively maintained |
| `mmcv` | 2026-07-27 | Moving |
| `mlu-ops-proto` | 2026-07-07 | Moving |
| `vllm-mlu` | 2026-05-11 | Moving (DeepSeek-V4 work landed 2026-04-24) |
| `torch_mlu` | 2025-03-15 | Quiet in public |
| `triton-linalg` | 2025-02-07 | Quiet in public |

`mlu-ops`' last GitHub **release** is **v1.8.1, published 2026-01-06** (not 2025-01-06 — a year error worth flagging, since it changes the read from "abandoned releases" to "recent release, then moved to master"), carrying a note that future releases will not be published on that page.

**The "development consolidated onto Gitee" inference is not supported.** Gitee `cambricon/torch_mlu` sits on branch `r1.22_pt2.4.0` (PyTorch 2.4.0) with 2025-era end-of-life dates, while `torch_mlu_ops` v1.3.2 already references Torch-MLU **v1.24.1**. The newest code is in the **licensed SDK bundle**; neither public mirror is demonstrably the active development home.

## Sources (2026-08-08 update)

- [DeepSeek-V4 Day-0 adaptation — Cambricon developer article (2026-04-24)](https://developer.cambricon.com/index/article/details.html?id=12) — PRIMARY
- [vllm-mlu README (master)](https://raw.githubusercontent.com/Cambricon/vllm-mlu/master/README.md) — PRIMARY
- [mlu-ops README (master)](https://raw.githubusercontent.com/Cambricon/mlu-ops/master/README.md) — PRIMARY; CNToolkit v4.1.0+ / CNNL v1.28.0+ / driver v6.0.3+
- [mlu-ops releases (GitHub API)](https://api.github.com/repos/Cambricon/mlu-ops/releases) — PRIMARY; v1.8.1 published 2026-01-06
- [Cambricon org repositories, sorted by push (GitHub API)](https://api.github.com/orgs/Cambricon/repos?sort=pushed&direction=desc) — PRIMARY
- [torch_mlu_ops v1.3.2 (2026-02-04)](https://dev.modelhub.org.cn/Chranos/enginex-mlu370-vllm/src/tag/v0.0.6/torch_mlu_ops-v1.3.2)
- [Torch-MLU-Ops — Baidu Baike entry](https://baike.baidu.com/item/Torch-MLU-Ops/67674918)
- [huismiling/torch-mlu-ops (third-party mirror)](https://github.com/huismiling/torch-mlu-ops)
- [Gitee cambricon/torch_mlu](https://gitee.com/cambricon/torch_mlu)
- [CNToolkit 3.8.4 doc tree (intermediate; release notes HTTP 401)](https://sdk.cambricon.com/static/independent/CNToolkit/3.8.4/releasenote/)
- [vllm-project/vllm PR #25942 overview](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/25942/overview)
- [Cambricon 2026 半年度报告 (cninfo)](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF) — PRIMARY; corroborates PD separation and DeepSeek Day-0 generically; Triton community tracking
