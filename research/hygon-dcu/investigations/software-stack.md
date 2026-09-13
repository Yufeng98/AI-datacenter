# Hygon DCU Software Stack Investigation

*as_of: 2026-08-08*
*chip: hygon-dcu*
*device_class: GPU/DCU (China, 海光)*
*resource: software-stack*

> **2026-08-08:** the body below is the 2026-04-05 investigation, retained. See the dated update appended at the end for the 深算三号 (BW1000)-era software picture — which is, notably, almost entirely *unchanged*: no BW1000-specific DTK branch, no 2026 DTK release confirmed, and the notable new claims (类CUDA positioning, >99% operator coverage, DeepSeek V4 priority adaptation) are unverified marketing.

---

## Summary

Hygon DCU uses **DTK (DCU ToolKit / 深算开发工具包)** as its full software platform, distributed through the **光合开发者社区 (Sourcefind Developer Community)** at `developer.sourcefind.cn`. DTK is explicitly structured as a ROCm analog: it forks or ports every ROCm component for the DCU hardware, preserving HIP API compatibility. Developers write HIP kernels; the DTK LLVM-based compiler (hipcc) targets DCU ISA. All standard math, DNN, communication, and debugging libraries from the ROCm ecosystem have DTK equivalents. Framework integration covers PyTorch, PaddlePaddle, TensorFlow, and vLLM. The CUDA-to-DCU migration path is: CUDA → HIP (hipify) → compile with hipcc on DTK.

---

## Stack Overview

```
Framework Integration
  └── PyTorch (torch with DCU backend via DTK ROCm compat)
  └── PaddlePaddle (native DCU support, recommended for Chinese users)
  └── TensorFlow (via DTK ROCm backend)
  └── vLLM (DCU-compatible port in community)
  └── GPUStack (LLM inference stack with DCU backend)
  └── llama.cpp (community DCU port)

Compiler / IR
  └── hipcc (clang/LLVM-based; targets DCU ISA from HIP C++ source)
  └── DTK LLVM compiler suite (clang for HIP, OpenMP, OpenACC)
  └── hipify-perl / hipify-clang (CUDA → HIP source translator)

Op Library
  └── hipDNN / MIOpen (Conv, BN, Attention, Pooling, Norm — cuDNN analog)
  └── hipBLAS (GEMM, BLAS L1–L3 — cuBLAS analog)
  └── hipSPARSE (sparse BLAS — cuSPARSE analog)
  └── hipFFT (FFT — cuFFT analog)
  └── hipRAND (random number generation)

Kernel Library
  └── hipThrust / hipCUB (Reduce, Scan, Sort, Histogram — Thrust/CUB analog)
  └── rocPRIM (low-level parallel primitives)

Runtime
  └── HIP Runtime (hiprt — streams, events, memory; cudart analog)
  └── HIP Driver API (context, module, VMM management)
  └── ROCr (ROCm runtime for device memory, HSA agent model)

Driver / Firmware
  └── DCU kernel driver (.ko — Linux PCIe, IOCTL, DMA)
  └── dcu-vgpu-device-plugin (Kubernetes resource advertising)
  └── HAMi (Heterogeneous AI device Management Infrastructure) — vGPU sharing

Communication
  └── RCCL (DCU port of ROCm Collective Communication Library)
  │     AllReduce, AllGather, Broadcast, ReduceScatter over xGMI+RoCE
  └── xGMI peer-to-peer (intra-node direct memory access)

Assembler / ISA
  └── DCU ISA (GCN/Vega-derived; not independently published)
  └── amdgcn LLVM backend target (used internally by hipcc)
```

---

## Layer Details

### Framework Integration

**PyTorch:**
PyTorch runs on DCU via the DTK ROCm compatibility layer. Since DTK presents a ROCm-compatible HIP runtime, PyTorch's ROCm backend (`torch.version.hip`) can detect and utilize DCU devices. Community guides confirm PyTorch 2.4+ works with DTK 25.x. The `torch.cuda` API is available via the HIP aliasing (`torch.cuda → torch.hip` mapping).

**PaddlePaddle:**
Baidu's PaddlePaddle provides **first-class DCU support** with dedicated install documentation. Paddle is the recommended framework for DCU in China due to tighter integration with DTK versions. Supports: training, inference, multi-GPU distributed (via NCCL/RCCL), mixed precision (FP16/BF16).

**vLLM / GPUStack:**
Community ports of vLLM for DCU exist; GPUStack has an official DCU backend tutorial for inference workloads. The K100_AI is used for serving LLaMA, DeepSeek, Qwen model families.

**FlyAIBox dcu-in-action:**
Open-source GitHub repo with practical DCU recipes: LLaMA pre-training, ChatGLM fine-tuning, distributed training, HPC workloads. Acts as a community "cookbook" for DCU AI development.

---

### Compiler / IR

**hipcc / DTK LLVM:**
`hipcc` is the primary compiler driver. It invokes the LLVM clang frontend targeting the amdgcn backend (modified for DCU). The compilation pipeline:
```
HIP C++ source
  → hipcc (clang frontend)
  → LLVM IR (amdgcn target)
  → Device ISA (DCU binary, analogous to HSACO)
  → embedded in host binary
```

DTK includes the full LLVM toolchain: clang, clang++, hipcc, rocm-gdb, rocm-smi, rocprofv2 (profiler), and roctracer. OpenMP target offload and OpenACC are supported via the same LLVM backend.

**hipify:**
CUDA-to-HIP automatic translation tools (`hipify-perl`, `hipify-clang`) are included in DTK. Workflow: annotate CUDA code → hipify → compile with hipcc → run on DCU. Handles most CUDA runtime APIs, kernel syntax, and intrinsics automatically.

---

### Op Library

| Library | Analog | Purpose |
|---------|--------|---------|
| hipDNN / MIOpen | cuDNN | Conv, BatchNorm, Pooling, LSTM, Attention |
| hipBLAS | cuBLAS | GEMM, BLAS L1–L3, batched GEMM |
| hipSPARSE | cuSPARSE | Sparse GEMM, SpMV |
| hipFFT | cuFFT | 1D/2D/3D FFT (single/double precision) |
| hipRAND | cuRAND | Pseudo/quasi-random number generation |

DTK bundles specific versions of these libraries. Version compatibility between DTK releases and framework versions must be managed carefully (documented community issue in ROCm migration).

---

### Kernel Library

| Library | Analog | Purpose |
|---------|--------|---------|
| hipThrust | Thrust | High-level parallel algorithms |
| hipCUB / rocPRIM | CUB | Device-level primitives (Reduce, Scan, Sort) |
| rocBLAS | cuBLAS (lower-level) | Architecture-optimized GEMM kernels |

---

### Runtime

**HIP Runtime (hiprt):**
The HIP runtime is the primary user-facing API (analogous to `libcudart.so`). It provides:
- `hipMalloc / hipFree` — device memory allocation
- `hipMemcpy` — host-device transfers
- `hipStream_t` — async stream management
- `hipEvent_t` — timing and synchronization
- `hipLaunchKernelGGL` — kernel dispatch macro
- Unified memory (`hipMallocManaged`) on supported hardware

**ROCr (ROCm Runtime / HSA):**
The lower-level HSA-based runtime (analogous to CUDA Driver API). Manages agents, queues, and AQL packet dispatch. Used internally by DTK; rarely accessed directly by application developers.

---

### Driver / Firmware

**DCU Kernel Driver:**
A Linux kernel module (`.ko`) that exposes DCU devices to userspace. Handles:
- PCIe BAR mapping and DMA management
- IOCTL interface for memory allocation and context management
- Interrupt handling and power management
- Loaded as `dcu.ko` or similar (not open-sourced by Hygon)

**HAMi (Heterogeneous AI device Management Infrastructure):**
HAMi is an open-source Kubernetes device plugin supporting multiple accelerators including Hygon DCU. It enables:
- Device sharing (vGPU) — multiple pods sharing one DCU
- Memory limits per pod
- Core utilization quotas
- Discovery of K100_AI, Z100, K100 devices

Repository: `https://github.com/Project-HAMi/HAMi`
Docs: `https://project-hami.io/docs/userguide/hygon-device/enable-hygon-dcu-sharing/`

---

### Communication

**RCCL (ROCm Collective Communication Library — DCU port):**
DTK includes a RCCL build targeting DCU hardware. Collective operations:
- AllReduce (ring-based)
- AllGather
- ReduceScatter
- Broadcast

For intra-node: leverages **xGMI** peer-to-peer transfers (direct HBM-to-HBM without PCIe).
For inter-node: runs over RoCE v2 or standard Ethernet using RDMA when available.

---

### Assembler / ISA

The DCU ISA is derived from AMD GCN (Graphics Core Next) / Vega architecture. Hygon does not publish a standalone ISA manual. Key characteristics:
- 64-thread wavefront scheduling
- VGPR (vector) and SGPR (scalar) register files
- `v_` prefix vector instructions, `s_` prefix scalar instructions
- `ds_` data share (LDS) instructions
- `buffer_load / buffer_store` for global memory access
- LLVM backend: `amdgcn` target with DCU-specific tuning patches

Developers do not typically write assembly; hipcc handles all ISA targeting. However, inline assembly (`__asm__` with `amdgcn` instructions) is possible for performance-critical paths.

---

## DTK Version Ecosystem

| DTK Version | ROCm Equivalent | Notes |
|-------------|----------------|-------|
| DTK 23.x | ROCm 5.x | Stable production |
| DTK 24.x | ROCm 6.x | Current mainstream |
| DTK 25.x | ROCm 6.x+ | Latest; Alibaba Cloud Linux images available |

DTK images are distributed via:
1. **developer.sourcefind.cn** — official download
2. **Alibaba Cloud DTK images** — pre-built OS images for Alibaba Cloud ECS/ACK
3. **Docker containers** — community-maintained DCU Docker images

---

## CUDA Migration Path

```
NVIDIA CUDA Code
     │
     ├─── hipify-perl / hipify-clang
     │         (90-95% automated translation)
     ↓
HIP C++ Code
     │
     ├─── hipcc (DTK LLVM compiler)
     │         HIP → LLVM IR → DCU ISA
     ↓
DCU Binary (HSACO-like)
     │
     ↓
Run on Hygon K100_AI / Z100 / K100
```

Key differences requiring manual attention:
- Warp size: CUDA 32 → DCU 64 (wavefront)
- `__syncwarp()` → no direct equivalent; use `__syncthreads()` or wavefront-aware sync
- Shared memory bank conflicts: 32-lane model on both, but 64-thread WF occupancy differs
- `__shfl_*` intrinsics: supported in HIP but with 64-lane semantics

---

## Sources

- [DTK Developer Portal — developer.sourcefind.cn](https://developer.sourcefind.cn/dtk)
- [海光 DTK 平台简介 — 知乎](https://zhuanlan.zhihu.com/p/705584420)
- [DCU Programming Practice — CSDN](https://blog.csdn.net/zzzzzucc/article/details/140313167)
- [DTK Environment Install — CSDN](https://blog.csdn.net/qq_27815483/article/details/141327011)
- [DCU Dev FAQ — CSDN](https://blog.csdn.net/qq_27815483/article/details/141311424)
- [FlyAIBox/dcu-in-action — GitHub](https://github.com/FlyAIBox/dcu-in-action)
- [HAMi DCU Support — GitHub](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md)
- [Enable Hygon DCU Sharing — HAMi Docs](https://project-hami.io/docs/userguide/hygon-device/enable-hygon-dcu-sharing/)
- [Running Inference with Hygon DCUs — GPUStack](https://docs.gpustack.ai/0.5/tutorials/running-inference-with-hygon-dcus/)
- [PaddlePaddle DCU Install — PaddlePaddle Docs](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/dcu/install_cn.html)
- [Alibaba Cloud Linux DTK Image Release Notes](https://www.alibabacloud.com/help/en/alinux/user-guide/dtk-image-release-notes)
- [AI Platform Library Overview — CSDN](https://blog.csdn.net/m0_49711991/article/details/135109487)

---

## Investigation Update — 2026-08-08: 深算三号 (BW1000)-era software stack

*Appended 2026-08-08. Headline: the software stack is materially unchanged. The flagship moved a generation (K100_AI → 深算三号 BW1000) with **no corresponding public toolchain change**.*

### DTK release status — no 2026 release confirmed

| Question | Answer (2026-08-08) |
|---|---|
| Latest DTK version located | **DTK 25.04** and **DTK 25.10** — 2025-vintage naming |
| Any DTK 26.x? | **Not found.** No 2026-dated DTK release confirmed |
| `developer.sourcefind.cn` reachable? | **No** — unreachable from the verification environment; version list could not be enumerated first-hand |
| BW1000-specific DTK branch / build target? | **Not documented** |
| ISA change, new op library, new collective library for BW1000? | **Not documented** |

The 2026-04-05 statement "PyTorch 2.4+ confirmed with DTK 24/25" is **neither contradicted nor refreshed**. It is retained as-is with the caveat that it has not been re-verified against a 2026 release.

Confidence on DTK 25.04 / 25.10: **medium** (2025-vintage secondary sources; portal unreachable for first-hand confirmation).

### Model-adaptation claims

| Claim | Sourcing | Confidence | Disposition |
|---|---|---|---|
| Tencent **Hunyuan Hy3** (open-source preview) adaptation completed **May 2026** | Baidu Baike (encyclopedia-grade) only | low-medium | Recorded, explicitly flagged as encyclopedia-sourced |
| **DeepSeek V4** priority adaptation | Vendor/channel claim | unverified | **Not** recorded as fact |
| DeepSeek (V3-era) running on DCU | Prior baseline, EEWorld | confirmed | Unchanged |

### Vendor marketing claims — flagged, not adopted

- **"类CUDA" (CUDA-like)** positioning for 深算三号 — vendor/marketing-channel framing. Architecturally, the public toolchain remains **HIP/ROCm-compatible via DTK**, i.e. the CUDA-likeness is the pre-existing hipify + HIP path, not a newly disclosed mechanism. No primary source describes a new CUDA-compatibility layer.
- **"Algorithm/operator coverage exceeding 99%"** — vendor/marketing claim, no primary source retrieved, no benchmark or operator-list published. **Not independently verified.**

Neither claim should be repeated in the survey without the attribution attached.

### What did NOT change

- HIP/ROCm compatibility model, hipcc/DTK LLVM compiler path, hipify migration route: unchanged.
- Op/kernel libraries (hipDNN/MIOpen, hipBLAS, hipSPARSE, hipFFT, hipRAND, hipThrust/hipCUB, rocPRIM, rocBLAS): unchanged.
- Runtime (HIP Runtime, ROCr/HSA), driver (proprietary `.ko`), HAMi vGPU device plugin: unchanged.
- Communication (RCCL over xGMI intra-node / RoCE inter-node): unchanged; **no BW1000-specific collective or offload engine documented**.
- ISA: still unpublished; still amdgcn-targeted via DTK LLVM. 64-thread wavefront **not confirmed** for BW1000 (presumed from unchanged toolchain, not stated by Hygon).

### Negative results

- No Hygon/DCU **MLPerf** Training or Inference submission (a software-stack-visible absence: no public end-to-end framework performance disclosure).
- No Hygon software or compiler paper at Hot Chips 2026, ISCA 2026, or ISSCC 2026.

### Sources — 2026-08-08 update

- [深算三号 — 百度百科 (Hunyuan Hy3 adaptation claim; encyclopedia-grade sourcing only)](https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890)
- [海光信息投资者互动回复 (2025-12-11) — 新浪财经](https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml)
- [海光信息 accelerator product page (no SDK/spec disclosure)](https://www.hygon.cn/product/accelerator)
- [DTK Developer Portal — developer.sourcefind.cn](https://developer.sourcefind.cn/dtk) — *unreachable during 2026-08-08 verification*
