# Kunlunxin XPU Software Stack

*as_of: 2026-04-05*
*chip: kunlunxin*
*Stack: XRE (runtime) + XTDK (kernel SDK) + XDNN (op lib) + XTCL (graph compiler) + PaddlePaddle*

---

## Overview

The Kunlunxin software stack is a **vertically integrated, layered ecosystem** built primarily around PaddlePaddle (Baidu's deep learning framework) with expanding PyTorch support. The stack from bottom to top:

```
PaddlePaddle / PyTorch / vLLM
        ↓
XTCL (graph-level AOT/JIT compiler)
        ↓
XDNN (operator library: BLAS, conv, attention, etc.)
        ↓
XTDK (XPU kernel SDK: C/C++ + data-parallel programming model)
        ↓
XRE (XPU Runtime Environment: device management, memory, streams)
        ↓
XPU Hardware (Kunlun 1 / R200 / R300 / P800)
```

---

## 1. XRE — XPU Runtime Environment

### Role

**XRE (XPU Runtime Environment)** is the lowest user-space software layer. It corresponds to CUDA Runtime (`libcudart.so`) in the NVIDIA stack.

### Capabilities

- Device enumeration, initialization, and context management
- Memory allocation and deallocation on XPU device (`xpu_malloc`, `xpu_free`)
- Host-to-device and device-to-host data transfer
- Stream (asynchronous execution queue) creation and synchronization
- Event-based timing and profiling
- Multi-card management (up to 8 XPU cards per host)
- Linux kernel driver interface (ioctl to XPU kernel module)

### Versioning

Current version in public documentation: **XRE 5.0.21.21** (referenced in PaddlePaddle 2.4+ integration guides)

### Installation

Distributed as pre-built `.deb` / `.rpm` packages for Ubuntu 18/20/22 and CentOS 7/8. Kernel module version must match host kernel. Official images available as Docker containers with XRE pre-installed.

### Comparison with CUDA

| Feature | XRE | CUDA Runtime |
|---------|-----|-------------|
| Context management | Yes | Yes (cuCtx*) |
| Stream API | Yes | Yes (cudaStream_t) |
| Unified Memory | Not documented | Yes (cudaMallocManaged) |
| Multi-device | Yes (up to 8) | Yes (unlimited) |
| Profiling hooks | Limited | CUPTI |
| Open source | No | No |

---

## 2. XTDK — XPU Tool Development Kit

### Role

**XTDK** is the kernel-level programming SDK — it is to XPU what CUDA C++ is to GPU. It enables writing custom compute kernels for XPU in C/C++ using a **data-parallel programming model**.

### Programming Model

XTDK uses a **prefix-keyword data-parallel model**:
- Prefixes declare the memory space or execution context of a function/variable:
  - `__global__`-equivalent: kernel executed on XPU
  - `__shared__`-equivalent: cluster-local SRAM
  - `__device__`: per-core private register/SRAM
- Supports **pointer arithmetic** and **inline assembly** for direct hardware control
- SIMD-aware: the XPU core is a vector unit; XTDK exposes vector intrinsics

### XTDK Compiler

- Based on LLVM/Clang extended for XPU ISA
- Input: XPU C/C++ kernel source (`.xpu` files)
- Output: XPU binary (device object code)
- Supports SIMD vectorization, loop unrolling, software pipelining
- Produces `.xpu.o` object files linked with host code via standard linker

### Custom Op Integration

PaddlePaddle 2.4+ added **XPU Plugin** support: users write custom operators using XTDK and register them via `paddle.utils.cpp_extension.load_op_meta_info_and_register_op()`, enabling seamless integration with PaddlePaddle training/inference.

---

## 3. XDNN — XPU DNN Operator Library

### Role

**XDNN** is the optimized operator library — equivalent to cuDNN + cuBLAS combined for XPU.

### Capabilities

| Category | Operations |
|----------|-----------|
| Linear Algebra (BLAS) | GEMM, GEMV, Batched GEMM, Strided GEMM |
| Convolution | Forward, backward weights, backward inputs; depthwise conv |
| Attention | Multi-head attention; FlashAttention-style fused kernel |
| Normalization | LayerNorm, BatchNorm, RMSNorm |
| Activations | GELU, SiLU, ReLU, Swish, Sigmoid |
| Elementwise | Add, Mul, Div, Exp, Log |
| Reduction | Sum, Max, Mean across axes |
| Pooling | Max/Avg 2D/3D |
| Embedding | Embedding lookup, gather |

### API Style

XDNN exposes a **C API** similar to cuBLAS: function calls with explicit tensor descriptors, pointers, and configuration structs. Users call XDNN directly for compute-intensive ops; XTCL calls XDNN internally for graph-level dispatch.

---

## 4. XTCL — XPU Tensor Compilation Library

### Role

**XTCL** is the **graph-level AOT/JIT compiler** for XPU. It sits above XDNN and XTDK, receiving a model graph (from TVM, ONNX, or PaddlePaddle IR) and lowering it to an optimized XPU binary.

### Architecture

XTCL is **TVM-based** (Apache TVM) with XPU-specific backends:
1. Imports model (ONNX / PaddlePaddle IR / Relay IR)
2. Graph-level optimizations: operator fusion, constant folding, layout transforms
3. Schedules computation using XPU-aware tiling and memory placement
4. Invokes XTDK compiler for kernel codegen
5. Outputs a compiled `.so` module for deployment

### AOT vs JIT

- **AOT mode**: compile offline, produce deployable binary; used in production inference
- **JIT mode**: lazy compilation during first inference run; used in development

### Integration with PaddlePaddle Lite

XTCL is the backend powering **Paddle-Lite on XPU** for edge/embedded deployment of Kunlun chips (Kunlunxin R200/R300 in industrial edge scenarios).

---

## 5. PaddlePaddle Framework Integration

### Native XPU Backend

PaddlePaddle is Kunlunxin's **primary framework** with the deepest integration:
- `paddle.device.set_device('xpu')` or `paddle.device.set_device('xpu:N')` for multi-card
- All core ops dispatch to XDNN/XTCL automatically
- Supports static graph (`paddle.static`) and dynamic graph (`paddle.to_tensor`)

### Training Support (PaddlePaddle 2.4+)

| Feature | Supported |
|---------|-----------|
| Mixed precision (FP16/BF16) | Yes |
| Static graph training | Yes |
| Dynamic graph training | Yes |
| Single-machine multi-card (DataParallel) | Yes |
| Multi-machine (DistributedDataParallel) | Yes |
| Model parallelism | Partial |
| Pipeline parallelism | Partial |
| Gradient accumulation | Yes |
| AMP (Automatic Mixed Precision) | Yes |

### Model Coverage (PaddlePaddle on XPU)

Verified models include:
- **Vision**: PPYOLOv3, PPYOLOE, ResNet, MobileNetV3, EfficientNet
- **NLP**: ERNIE 3.0, BERT, GPT-2, Llama, ChatGLM, Qwen
- **Speech**: DeepSpeech, Conformer
- **Recommendation**: DLRM, Wide & Deep
- **LLM**: DeepSeek V3/R1 671B (P800, full-power deployment confirmed)
- Total: **51+ verified models** as of PaddlePaddle 2.4

### FastDeploy Integration

**FastDeploy** (Baidu's inference deployment framework) provides XPU acceleration via:
- `XpuWorker` class for device initialization and memory management
- XTDK and XVLLM libraries downloaded via `setup.sh`
- LLM deployment: XVLLM (vLLM port for XPU) via `xpu_worker_config`

---

## 6. PyTorch and vLLM Support

### PyTorch via PaddlePaddle Bridge

PyTorch is supported via a **plugin/bridge approach**:
- `torch_xpu` custom device extension registers XPU as a PyTorch device (`'xpu'`)
- Dispatches PyTorch ops through XDNN via the ATen custom backend mechanism
- Limited to inference; training via PyTorch is under development

### vLLM-Kunlun

> ⚠️ **Corrected 2026-08-08.** The paragraph below described `baidu/vLLM-Kunlun` as a **fork**. That is wrong — it is an **out-of-tree hardware plugin** for stock vLLM. See "Investigation Update — 2026-08-08" at the end of this document for the corrected description, verified release history, and supported-model list. The original text is retained here only as the historical record of what this repo previously asserted.

~~Baidu maintains an open-source **vLLM fork** (`baidu/vLLM-Kunlun`) that:~~
- Replaces CUDA kernels with XDNN/XTDK equivalents
- Enables PagedAttention and continuous batching on XPU
- Supports Llama, Qwen, ChatGLM, DeepSeek series on R200/R300/P800
- GitHub: https://github.com/baidu/vLLM-Kunlun

---

## 7. Compiler and Toolchain Pipeline

```
User Model (PyTorch / PaddlePaddle)
    ↓  Graph tracing / export
Model IR (PaddlePaddle IR / ONNX / Relay)
    ↓  XTCL graph compiler
Optimized Fused Op Graph
    ↓  XTCL → XTDK kernel codegen
XPU Kernels (.xpu.o)  +  XDNN calls
    ↓  Link
Deployable XPU binary (shared library or static binary)
    ↓  XRE runtime load
XPU Hardware Execution
```

---

## 8. Programming Model Comparison

| Aspect | Kunlunxin XPU | NVIDIA CUDA | Huawei Ascend |
|--------|--------------|-------------|---------------|
| Low-level kernel SDK | XTDK (C/C++ + prefix keywords) | CUDA C++ (thread blocks/warps) | AscendC (AICORE C++) |
| Programming model | Data-parallel SIMD (explicit vector) | SIMT (implicit thread + warp) | AICORE (explicit buffer management) |
| Op library | XDNN (BLAS + DNN) | cuDNN + cuBLAS | CANN AscendCL |
| Graph compiler | XTCL (TVM-based AOT/JIT) | TensorRT / torch.compile | CANN ATC (offline) |
| Runtime | XRE | CUDA Runtime | AscendCL Runtime |
| Primary framework | PaddlePaddle (native) | PyTorch / TF (native) | MindSpore (native) |
| PyTorch support | Plugin (partial) | Native | Plugin (partial) |
| vLLM support | vLLM-Kunlun **out-of-tree platform plugin** (open; stock vLLM unmodified) | vLLM native | vLLM-Ascend plugin |
| Open-source kernel SDK | Partial (XTDK not open) | No (CUDA closed) | No (AscendC closed) |

---

## Sources

- [Kunlunxin XPU Backend — PaddlePaddle FastDeploy DeepWiki](https://deepwiki.com/PaddlePaddle/FastDeploy/8.3-kunlunxin-xpu-backend)
- [KunlunXin XPU — FastDeploy LLM Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/)
- [昆仑芯 XTCL — PaddlePaddle Lite Documentation](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xtcl.html)
- [昆仑芯 XPU — PaddlePaddle Lite Documentation](https://www.paddlepaddle.org.cn/lite/v2.12/demo_guides/kunlunxin_xpu.html)
- [分享：昆仑芯×飞桨适配方案 — 昆仑芯官网](https://www.kunlunxin.com/news/813.html)
- [核心技术 — 昆仑芯官网](https://www.kunlunxin.com/technology)
- [昆仑芯 XPU 芯片 — PaddlePaddle Documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/index_cn.html)
- [Paddle-Lite Kunlunxin XPU Demo Guide (GitHub)](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/kunlunxin_xpu.md)
- [PaddlePaddle 2.4 Release Note](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/release_note_en.html)
- [vLLM-Kunlun (GitHub)](https://github.com/baidu/vLLM-Kunlun)
- [百度昆仑 XPU 详解 — 知乎](https://zhuanlan.zhihu.com/p/646793342)
- [Kunlun XPU PaddlePaddle Installation Guide — PaddleX](https://paddlepaddle.github.io/PaddleX/main/en/other_devices_support/paddlepaddle_install_XPU.html)

---

## Investigation Update — 2026-08-08

*Window: 2026-04-01 → 2026-08-08 against the 2026-04-05 baseline. The main item here is a **correction of a pre-existing repo error**, not a change in the window.*

### 1. vLLM-Kunlun is a PLUGIN, not a fork (repo error corrected)

Both [github.com/baidu/vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) and [vllm-kunlun.readthedocs.io](https://vllm-kunlun.readthedocs.io/en/latest/) describe the project as **"a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU"**, registered **"as a standard vLLM platform plugin via Python entry points, no need to modify vLLM source code"**, following the vLLM **"RFC: Hardware Pluggable"** interface (vllm issue #11162).

Consequences for how the stack should be described:

- Users install **stock upstream vLLM plus the plugin** — there is no divergent vLLM codebase to track, and no CUDA-kernel-replacement patch series in a vendored vLLM tree.
- The earlier phrasing "replaces CUDA kernels with XDNN/XTDK equivalents" is misleading as a description of the *distribution model*; the XPU platform, attention backend, worker and model-runner are supplied **by the plugin** through vLLM's platform interface.
- **This was already true at the 2026-04-05 baseline.** It is a documentation error in this repo, not a development in the window.

### 2. Verified release history (GitHub Releases API)

| Release | Date |
|---|---|
| v0.10.1.1 | 2025-12-24 |
| v0.11.0rc1 | 2025-12-26 |
| **v0.11.0 (latest stable)** | **2026-03-13** — 154 commits from 26 contributors |

**Do not cite a dev-branch version number.** A "v0.25.1-dev" figure circulates; the docs site instead reports v0.15.1-dev, and v0.25.x is implausible against upstream vLLM's 2026 version line. The discrepancy is unresolved, so no dev version is recorded.

### 3. Hardware and environment prerequisites

- **Kunlun3 P800 only.** No M100 support in-tree — confirmed against the prerequisites list.
- Ubuntu 20.04; Python ≥ 3.10; PyTorch ≥ 2.5.1 via the **KL3-customised `xpytorch`** build.

### 4. Supported models (20+)

Qwen 2 / 2.5 / 3 / 3.5, Llama, DeepSeek (including **DeepSeek-V3.2**), GLM, Gemma4, InternLM2, Kimi-K2; multimodal **Qwen-VL**, **InternVL**, **InternS1**.

### 5. Companion tooling (documented, but not in the repo's top-level tree)

Both are documented in the vLLM-Kunlun **developer guide** on readthedocs and are **confirmed to exist**, but they are **not visible in the repository's top-level tree** — they are ecosystem tooling the docs reference rather than artifacts the plugin ships. Describe them that way.

- **`torch_xray`** — module-level operator precision verification: dumps operator-level tensors, performs automatic GPU-vs-P800 layer-by-layer comparison, and auto-generates operator unit tests from captured cases.
- **`xpu_profiler`** — nsys-like profiler producing operator call timelines.

### 6. Searched and absent

- No new public **XRE / XTDK / XDNN / XTCL** release notes in the window; the XRE 5.0.21.21 reference above remains the newest version string found.
- No MLPerf Inference v6.x / Training v5.x submission from Kunlunxin or Baidu XPU.
- No M100 software enablement of any kind (no plugin support, no framework release notes, no toolkit).

### Sources added 2026-08-08

- [baidu/vLLM-Kunlun (GitHub) — plugin description, prerequisites, model list](https://github.com/baidu/vLLM-Kunlun)
- [vLLM-Kunlun releases (GitHub API)](https://api.github.com/repos/baidu/vLLM-Kunlun/releases)
- [vLLM-Kunlun documentation (readthedocs)](https://vllm-kunlun.readthedocs.io/en/latest/)
- [vLLM-Kunlun developer guide — operator accuracy / torch_xray](https://vllm-kunlun.readthedocs.io/en/latest/developer_guide/evaluation/accuracy/accuracy_kernel.html)
