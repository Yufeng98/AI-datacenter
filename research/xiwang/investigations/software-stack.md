# Xiwang (曦望) Software Stack

*as_of: 2026-04-05*
*chip: xiwang*
*device_class: AI Accelerator (China, 曦望)*

---

## Overview

Xiwang's software strategy is centered on **CUDA compatibility** — the stack is designed to minimize migration friction for users moving from NVIDIA GPU infrastructure to the S2/S3 accelerators. The company explicitly claims "software and hardware full compatibility with CUDA," spanning drivers, runtime APIs, development toolchain, operator libraries, and communication libraries.

This positions Xiwang's software approach as a **CUDA replacement stack** (similar in concept to Mthreads MUSA, Biren BIRENSUPA, and Hygon DTK), rather than a novel programming model.

⚠️ **SUPERSEDED 2026-08-08 — read the update section at the end of this file before using anything below.** The baseline sentence read: "The software stack name/brand is not publicly disclosed. Specific open-source repositories have not been identified." Both halves are refuted: the stack is **SIRE**, the programming model **TANG**, the runtime **TangRT**, the collectives library **PCCL**, and there are three upstream open-source integrations (FlagTree, FlagCX, vllm-plugin-FL). The remainder of this baseline section is retained as the historical record of what was knowable in April 2026.

---

## 1. Framework Integration

### PyTorch

Xiwang provides PyTorch integration for S2/S3, enabling the S2 to appear as a PyTorch device. The integration enables:
- Pre-training of common open-source large models
- Fine-tuning and instruction tuning
- Inference serving

**Models confirmed compatible (company claim):** >90% of mainstream large model architectures on ModelScope, including:
- DeepSeek (R1/V3-class)
- Qwen (Alibaba)
- Other standard transformer architectures

### Hugging Face Ecosystem

Xiwang claims compatibility with Hugging Face open-source model toolboxes, implying support for `transformers` library inference paths on the S2/S3 accelerator.

### DeepSpeed

DeepSpeed (distributed training/inference framework) is specifically mentioned as supported, suggesting compatibility with ZeRO optimizer stages and pipeline parallelism for large model training.

### Inference Serving Frameworks

Framework compatibility for inference serving (vLLM-style) is implied by the company's "inference cost reduction" positioning but **specific framework names and integration depth are not publicly documented**.

---

## 2. Compiler / IR

### Self-Developed Compiler Toolchain

Xiwang states the compiler toolchain is "fully self-developed" (全自研) from the instruction set level up. Specific details:

| Component | Details |
|-----------|---------|
| Compiler name/brand | Not publicly disclosed |
| Input format | CUDA C++ (via compatibility layer) + native Xiwang kernel language |
| Output | Xiwang GPU ISA binary |
| CUDA migration support | Claimed (CUDA C++ source → Xiwang GPU via migration tools) |
| MLIR usage | Not confirmed |
| TensorRT analog | Not publicly disclosed (inference graph compiler assumed present) |

The degree of binary CUDA compatibility (vs. source-level compatibility) is **not public**.

### Operator Development

Custom operator development is cited as a fully self-developed capability, implying a kernel authoring toolchain analogous to Triton or CUDA's device-side programming model. The specific APIs and abstractions are **not public**.

---

## 3. Op Library

### Deep Learning Library (CUDA cuDNN/cuBLAS analog)

A deep learning operator library is confirmed present (required for PyTorch/DeepSpeed compatibility claims). Specific details:

| Component | Details |
|-----------|---------|
| Library name | Not publicly disclosed |
| GEMM operations | Present (inferred; required for A100-class performance) |
| Attention kernels | Present (inferred; required for LLM inference support) |
| Normalization, pooling | Present (inferred) |
| Flash-Attention support | Not explicitly confirmed |
| Sparse computation | Not public |

### ModelScope Adaptation

Xiwang specifically cites **ModelScope** (Alibaba's model hub) compatibility at >90% of model architectures. This suggests a robust operator coverage for standard transformer layers (self-attention, FFN, RMS-Norm, RoPE, etc.).

---

## 4. Runtime

### Runtime API

A CUDA Runtime API-compatible runtime is implied by the "CUDA compatibility" claim. Specific runtime API names, library files, and feature parity details are **not publicly disclosed**.

| Component | Assumed (based on CUDA compatibility claim) |
|-----------|---------------------------------------------|
| Memory allocation | cudaMalloc analog |
| Stream management | cudaStream analog |
| Kernel launch | CUDA-style <<<grid, block>>> syntax (likely) |
| Event timing | cudaEvent analog |
| Async memory copy | cudaMemcpyAsync analog |

### Driver API

A lower-level driver API is assumed present but details are **not public**.

---

## 5. Driver / Firmware

### Kernel Driver

A Linux PCIe kernel driver is required for operation and assumed present. Details (open vs. closed source, module name, ioctl interface) are **not public**.

### On-chip Firmware

On-chip resource management firmware is likely present (analogous to NVIDIA GSP firmware). Details **not public**.

### Kubernetes / Container Support

Not publicly documented. Given Xiwang's cloud-deployment positioning, Kubernetes device plugin support for cloud orchestration is expected but unconfirmed.

---

## 6. Communication Library

### Collective Communication (NCCL analog)

A collective communication library is required for multi-GPU training and distributed inference. Whether Xiwang has developed a proprietary NCCL-compatible library or relies on RCCL/NCCL compatibility shims is **not public**.

| Feature | Status |
|---------|--------|
| AllReduce | Assumed present (required for DeepSpeed ZeRO) |
| AllGather / ReduceScatter | Assumed present |
| Ring/Tree topologies | Not confirmed |
| Scale-up interconnect integration | Not public (interconnect details undisclosed) |

---

## 7. ISA / Assembler

### Proprietary ISA

Xiwang has confirmed that the S2 ISA (instruction set architecture) is fully self-developed. The ISA is:

| Attribute | Value |
|-----------|-------|
| ISA type | Proprietary SIMT (GPU-class) |
| Public documentation | None identified |
| Virtual ISA (PTX analog) | Not confirmed |
| Native ISA | Present (not public) |
| Assembler toolchain | Present (not public) |

No public PTX-equivalent virtual ISA has been announced. All development is assumed to go through the self-developed compiler toolchain (source-level, not ISA-level, portability).

---

## 8. Toolchain / Dev Tools

| Tool | Status |
|------|--------|
| Profiler | Assumed present; details not public |
| Debugger | Assumed present; details not public |
| Performance analyzer | Not confirmed |
| CUDA migration tool | Claimed (namespace substitution level assumed) |
| Open-source repos | None identified as of 2026-04-05 |
| Developer portal / documentation | sunrise-ai.com (limited public content) |

---

## 9. Software Stack Summary Diagram (Layers)

```
┌─────────────────────────────────────────────────────┐
│  Framework Integration                              │
│  PyTorch · Hugging Face · DeepSpeed · (others TBD) │
├─────────────────────────────────────────────────────┤
│  Inference / Training Compiler                      │
│  (TensorRT analog — name not public)               │
├─────────────────────────────────────────────────────┤
│  Op Library (cuDNN/cuBLAS analog — name not public) │
├─────────────────────────────────────────────────────┤
│  Collective Comms (NCCL analog — name not public)  │
├─────────────────────────────────────────────────────┤
│  Runtime API (cudart analog — name not public)     │
├─────────────────────────────────────────────────────┤
│  Driver API (lower-level — not public)              │
├─────────────────────────────────────────────────────┤
│  Kernel Driver + Firmware (Linux PCIe, not public) │
├─────────────────────────────────────────────────────┤
│  Self-Developed Compiler + ISA Tools               │
│  (fully self-developed; not public)                 │
├─────────────────────────────────────────────────────┤
│  Xiwang GPU Hardware (S2 / S3)                     │
└─────────────────────────────────────────────────────┘
```

---

## 10. Open Source Status

⚠️ **REFUTED 2026-08-08 — see the update section below.** As recorded at baseline: "As of 2026-04-05, no open-source repositories for the Xiwang software stack have been identified." This contrasts with:
- Mthreads: `torch_musa`, `vllm-musa` on GitHub
- Biren: BIRENSUPA documentation public
- Kunlunxin: `vLLM-Kunlun` on GitHub

Baseline conclusion (**now known to be wrong**): "Xiwang's software stack remains entirely proprietary with no confirmed public repository." In fact the FlagTree `third_party/sunrise` Triton backend had been public since **2026-01-23** — before this baseline was written.

---

## Sources

- [量子位 — 曦望融资近10亿融资 (June 2025)](https://www.qbitai.com/2025/06/303355.html)
- [量子位 — 曦望发布推理GPU S3 (January 2026)](https://www.qbitai.com/2026/01/373113.html)
- [新浪科技 — 新国产GPU曦望](https://finance.sina.com.cn/tech/roll/2025-07-01/doc-infcxsmt5575943.shtml)
- [CSDN — S2/S3 overview](https://blog.csdn.net/suanlix/article/details/149097721)
- [CSDN — 量子位 S2 report](https://blog.csdn.net/QbitAI/article/details/149034468)
- [国产AI芯片公司曦望融资近10亿元 — 钛媒体](https://www.tmtpost.com/7630475.html)
- [sunrise-ai.com official site](https://sunrise-ai.com/)

---

# Update — 2026-08-08: SIRE, TANG, TangRT, PCCL, and the FlagOS open-source integration

*Appended 2026-08-08. Supersedes the 2026-04-05 baseline above where marked. Change class: **major** — this is the largest single correction in this chip's record.*

*Primary sources: sunrise-ai.com S2/S3 product pages and the SIRE product page (site CMS assets republished 2026-08-06); FlagTree `third_party/sunrise`; FlagOpen/FlagCX; flagos-ai/vllm-plugin-FL.*

## 0. Two baseline statements are refuted

1. **"The software stack name/brand is not publicly disclosed."** It is **SIRE — Sunrise Integrated Running Environment**, a first-class product line in the site navigation. Component names are also public: programming model **TANG**, runtime **TangRT**, collectives **PCCL**.
2. **"Specific open-source repositories have not been identified … the stack remains entirely proprietary with no confirmed public repository."** Xiwang has an upstreamed Triton backend in **FlagTree**, a **PCCL** adaptor in **FlagCX**, and a vendor entry in **vllm-plugin-FL**. The FlagTree backend has been public since **2026-01-23** — *before* the 2026-04-05 baseline. This was a research miss, not a change in the world, and it is the reason this update is classed major rather than moderate.

## 1. SIRE — the stack brand and its layer model

Tagline: 软硬协同 · 高度兼容 · 卓越效能 (hardware-software co-design · high compatibility · superior efficiency).

The vendor diagrams SIRE as four layers over the GPU:

| Layer (vendor wording) | English | Known component names |
|---|---|---|
| 框架层与生态应用 | Frameworks + ecosystem applications | PyTorch, vLLM, SGLang, LightLLM, LightX2V |
| 编译器与算子库层 | Compiler + operator libraries | FlagTree Triton backend (`tang`); op-library name **not disclosed** |
| 驱动与运行时层 | Driver + runtime | **TangRT** (`libtangrt_shared`), `libtang.so` |
| SoC 固件/系统软件层 | SoC firmware / system software | not disclosed |

Declared scope tags: 基础工具链 (base toolchain) / 编程接口 (programming interfaces) / 基础库 (base libraries) / 基础平台 (base platform) / 模型引擎 (model engine) / MaaS 服务能力 (MaaS service capability) / IP 模块化 (IP modularization).

## 2. TANG — the programming model, and TangRT — the runtime

The S2 product page states the part runs 依托自研 **TANG 编程模型**及 **SIRE** 软件栈. This is corroborated independently in open source:

| Artifact | Value |
|---|---|
| Triton `backend_name` | `tang` |
| Target string | `tang:S2` |
| Runtime shared object | `libtangrt_shared` |
| Driver-level library | `libtang.so` |
| Default install root | `/usr/local/tangrt` |
| Toolchain path | `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-<arch>/` |
| Env knobs | `TRITON_LIBTANG_PATH`, `TRITON_SUNRISE_LLD_PATH`, `TRITON_SUNRISE_TRANSLATE_TRIPLE` |

**TangRT is the CUDA-runtime analog** — the layer NVIDIA calls cudart, AMD calls HIP runtime, and Moore Threads calls MUSA runtime. Nothing about its API surface is documented publicly; only its file names and install paths are visible, via the build system of the open backend.

## 3. PCCL — the collectives library

FlagCX's README names **PCCL, "Sunrise Collective Communications Library"**, and links it to sunrise-ai.com. Build integration:

```
USE_SUNRISE=1
DEVICE_HOME=/usr/local/tangrt      -ltangrt_shared
CCL_HOME=/usr/local/pccl           -lpccl
-DUSE_SUNRISE_ADAPTOR
```

FlagCX's support matrix for PCCL:

| Operation | Homogeneous | Heterogeneous |
|---|---|---|
| send / recv | supported | supported |
| broadcast | supported | supported |
| reduce | supported | supported |
| allreduce | supported | supported |
| allgather | supported | supported |
| reducescatter | supported | supported |
| group ops | supported | supported |
| **gather** | **not supported** | **not supported** |
| **scatter** | **not supported** | **not supported** |
| **alltoall / alltoallv** | **not supported** | **not supported** |

The missing primitives matter for this vendor's own positioning: **alltoall/alltoallv are the standard MoE expert-dispatch collectives**, and the SC3-256 SuperPOD is marketed for large-EP trillion-parameter MoE serving. Either PCCL grows those ops, or the EP dispatch path runs on something other than PCCL. This is a gap worth re-checking on the next scan; it is **not** evidence that the hardware cannot do it.

This corrects the baseline's "NCCL analog assumed present … Whether Xiwang has developed a proprietary NCCL-compatible library or relies on shims is not public."

## 4. FlagTree — the open Triton backend

Location: `third_party/sunrise/` in **FlagTree**, BAAI FlagOS's unified Triton fork. The README vendor table lists "Sunrise（曦望芯科）".

| Date | Changelog entry |
|---|---|
| **2026-01-23** | sunrise backend added on **Triton 3.4** with CI/CD — *public before the repo's 2026-04-05 baseline* |
| **2026-06-30** | upgraded to **Triton 3.6** with CI/CD |

Distribution: prebuilt wheels `flagtree==0.4.0+sunrise3.4` and `flagtree==0.6.0+sunrise3.6` from `https://resource.flagos.net/repository/flagos-pypi-hosted/simple`.

**How open is it, really?** The backend links a closed-source **`sunriseTritonPlugin.so`**. The Python-level backend is a shim: it selects triples, sets warp size and stage counts, picks libdevice bitcode, and drives `clang-offload-bundler` and LLD — while the actual lowering lives in the proprietary plugin. So the honest characterization is: *open integration surface, closed compiler*. That is still a large step past the baseline's "fully proprietary, nothing public", because the shim leaks real architecture parameters (see the hw-architecture investigation, section 4).

**Silicon scope:** the published FlagTree user manual says the backend is **"Available for S2"**. S3 exists in the backend only as compiler plumbing — a `stcuv2` triple and a separate `*_S3.bc` libdevice, with a source comment noting S2/S3 must be distinguished.

## 5. Wider FlagOS integration

| Date | Repo | Change |
|---|---|---|
| 2026-06-01 | FlagCX | "[PAL] Add torch plugin support for Sunrise" |
| 2026-06-10 | FlagCX | runtime vendor detection replaces the `#ifdef USE_SUNRISE_ADAPTOR` build-time switch |
| undated | vllm-plugin-FL | Sunrise listed as a supported chip vendor |

The 2026-06-10 change is the more interesting of the two: moving vendor selection from compile time to runtime is what a project does when it expects a single binary to encounter Sunrise devices in the field, rather than requiring a vendor-specific build.

> **Version string not published:** the vendor news index carries an item 曦望全栈适配 FlagOS 2.1. The FlagOS organisation page advertises "Latest Release v1.5". The "FlagOS 2.1" string **could not be confirmed** and is deliberately not recorded as fact.

## 6. Framework support is now vendor-stated, not inferred

The baseline inferred serving support ("vLLM-style assumed", "implied"). The vendor now states it:

- **PyTorch** — claimed to auto-track the latest upstream release
- **SGLang**, **vLLM**, **LightLLM**, **LightX2V**
- **零代码无缝迁移** — zero-code migration from third-party GPGPU platforms (company claim; migration depth unverified)

Claimed model coverage: DeepSeek, Qwen, Llama, SD / Flux / Wan / Seko / Hunyuan3D, CNNs and speech models.

## 7. What is still not disclosed

Operator-library name and any operator performance data; inference graph-compiler (TensorRT analog) existence and name; compiler IR and pass structure; TangRT API surface; PCCL algorithms, topology awareness and bandwidth; kernel-driver details and whether it is open; Kubernetes device plugin; profiler and debugger names; ISA reference documentation. No MLPerf submissions and no conference papers were found.

## Sources added 2026-08-08

- [曦望 S2 product page — 依托自研 TANG 编程模型及 SIRE 软件栈; SRLink; framework list](https://sunrise-ai.com/products/s2-product)
- [曦望 S3 product page — SIRE four-layer diagram and scope tags](https://sunrise-ai.com/products/s3-product)
- [FlagTree — README vendor table; changelog 2026/01/23 (Triton 3.4) and 2026/06/30 (Triton 3.6)](https://github.com/FlagTree/flagtree)
- [FlagTree sunrise `backend/compiler.py` — backend_name 'tang', target 'tang:S2', triples, warp size, FP8 dtypes](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/compiler.py)
- [FlagTree sunrise `backend/driver.py` — libtang.so, libtangrt_shared, /usr/local/tangrt](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/driver.py)
- [FlagTree sunrise `language/tang/libdevice.py` — OCML-based device math](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/language/tang/libdevice.py)
- [FlagTree sunrise user manual — plugin v0.4.0 / v0.6.0, wheels, "Available for S2"](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-sunrise)
- [FlagCX README — PCCL "Sunrise Collective Communications Library" + op-support matrix](https://raw.githubusercontent.com/FlagOpen/FlagCX/main/README.md)
- [FlagCX makefiles/sunrise.mk — USE_SUNRISE, DEVICE_HOME, CCL_HOME=/usr/local/pccl, -lpccl](https://raw.githubusercontent.com/FlagOpen/FlagCX/main/makefiles/sunrise.mk)
- [FlagCX commit history — 2026-06-01 torch plugin, 2026-06-10 runtime vendor detection](https://github.com/FlagOpen/FlagCX/commits/main)
- [vllm-plugin-FL README — Sunrise listed as a supported chip vendor](https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/README.md)
