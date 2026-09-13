# Biren BIRENSUPA Software Stack

*as_of: 2026-04-05*
*chip: biren*
*device_class: GPU-like AI Accelerator (China)*

---

## Overview

Biren's software platform is called **BIRENSUPA** (壁仞 SUPA). The SUPA name stands for the overall SDK and programming model. It is structured to mirror the NVIDIA CUDA ecosystem as closely as possible, enabling CUDA-familiar developers to migrate with minimal friction.

The BIRENSUPA stack has eight documented layers:

1. Hardware Abstraction Layer (HAL)
2. SUPA Programming Model (kernel language + runtime)
3. BRCC Compiler
4. Deep Learning Acceleration Libraries
5. General Computing Acceleration Libraries
6. Toolchain (profiler, debugger, ISA tools)
7. Framework Integration (PyTorch, TensorFlow, PaddlePaddle)
8. Inference Engine + Application SDKs

Publicly, BIRENSUPA is positioned as a complete CUDA alternative, with emphasis on CUDA migration support: developers using standard CUDA C++ patterns can port to SUPA with compiler-assisted translation tools provided by Biren.

---

## 1. Framework Integration

### PyTorch

Biren provides a PyTorch backend that registers the BR100/BR104 as a PyTorch device (`torch.device("supa")` or via a custom device backend). The integration dispatches ATen operations to SUPA's deep learning libraries (analogous to torch_mlu's dispatch to CNNL).

- Supported: PyTorch training and inference
- Distributed training: via BCCL (Biren's collective comms library, NCCL analog)
- Autocast / mixed precision: TF32+, BF16 supported

### TensorFlow

TensorFlow integration via XLA backend or custom device plugin. Specific API details not publicly disclosed, but Biren has confirmed TensorFlow support.

### PaddlePaddle

Given Biren's Chinese market focus, PaddlePaddle (Baidu's framework) integration is documented. Custom operator plugins for the BR100/BR104 are available.

### vLLM / Inference Serving

Biren provides inference-serving integration, enabling deployment of popular open-source LLMs (LLaMA, Qwen, Baichuan) on BR100 hardware through SUPA-accelerated attention kernels and KV-cache management.

---

## 2. Compiler / IR

### BRCC (Biren Runtime Compiler / SUPA Compiler)

**BRCC** is Biren's kernel compiler, analogous to `nvcc`. It compiles SUPA C++ kernel code (GPU-style `__global__` functions, thread/block/grid model) to Biren's native GPU ISA binary.

Key features:
- Input: SUPA C++ (extended CUDA-like syntax)
- Output: Biren GPU binary (`.br` or proprietary format)
- Optimization: Loop unrolling, vectorization, register allocation, memory access coalescing
- CUDA compatibility: Assisted migration path from CUDA C++ to SUPA C++

### MLIR-based Inference Compiler

For inference deployment, Biren provides an MLIR-based graph compiler (analogous to TensorRT or Cambricon's MagicMind) that:
- Accepts ONNX, PyTorch, TensorFlow models
- Performs operator fusion, quantization (INT8/FP16/BF16), layout optimization
- Produces compiled engines optimized for BR100's SPC/EU microarchitecture and 300 MB L2 cache

---

## 3. Op Library

### Deep Learning Library (BILA — Biren DL Library)

Biren's deep learning operator library (cuDNN + cuBLAS analog):

| Operator Category | Examples |
|-------------------|---------|
| GEMM / MatMul | Batched GEMM, FP32/BF16/TF32+/INT8 |
| Convolution | Conv2D forward/backward, depthwise, grouped |
| Normalization | BatchNorm, LayerNorm, GroupNorm |
| Attention | Flash-Attention-style fused SDPA |
| Activation | GELU, ReLU, SiLU, Swish |
| Pooling | MaxPool, AvgPool, AdaptivePool |

### General Computing Library (BLAS / Reduction)

General-purpose acceleration primitives:
- Reduce, Scan, Sort operations
- BLAS L1/L2/L3 equivalents
- Vector elementwise operations

---

## 4. Kernel Library

### SUPA Kernel Library (CUTLASS analog)

Biren provides a SUPA kernel template library for high-performance custom kernel authoring:
- GEMM templates for TF32+/BF16/INT8 precisions
- Convolution templates with epilogue fusion
- Reduction and elementwise primitives
- Targets SPC/EU hierarchy with explicit tiling for 300 MB L2 cache

---

## 5. Runtime

### SUPA Runtime (libsupa.so / SUPA RT API)

The SUPA Runtime API (analogous to CUDA Runtime `libcudart.so`):

| API Component | SUPA Equivalent | CUDA Analog |
|---------------|-----------------|-------------|
| Device management | `supaGetDeviceCount()`, `supaSetDevice()` | `cudaGetDeviceCount()` |
| Memory allocation | `supaMalloc()`, `supaFree()` | `cudaMalloc()` |
| Stream/Queue | `supaStreamCreate()`, `supaStreamSync()` | `cudaStreamCreate()` |
| Kernel launch | `<<<grid, block, shmem, stream>>>` syntax | Same CUDA syntax |
| Event timing | `supaEventCreate()`, `supaEventElapsedTime()` | `cudaEventCreate()` |
| Memory copy | `supaMemcpyAsync()` | `cudaMemcpyAsync()` |

The SUPA runtime is designed for minimal CUDA migration friction — most `cuda*` → `supa*` API name substitutions work directly.

### SUPA Driver API

Lower-level driver API for explicit context management, module loading, and fine-grained memory control (analogous to CUDA Driver API `libcuda.so`).

---

## 6. Driver / Firmware

### Linux Kernel Driver

Biren's Linux kernel driver (`.ko` module) provides:
- PCIe BAR mapping and MMIO
- IOCTL dispatch for SUPA runtime requests
- DMA engine control and memory management
- Interrupt handling and device recovery

### On-GPU Firmware

The BR100 includes on-GPU firmware managing resource allocation, power management, and GPU-side initialization — analogous to NVIDIA's GSP firmware. Specific firmware architecture details are not publicly disclosed.

### Kubernetes Device Plugin

Biren provides a K8s device plugin for cloud-native deployment:
- BR100/BR104 resource reporting (`birentech.com/gpu`)
- Health monitoring and fault detection
- Multi-tenancy support via SVI (stream virtualization isolation)

---

## 7. Communication

### BCCL (Biren Collective Communication Library)

NCCL analog for distributed training:

| Operation | Topology |
|-----------|---------|
| AllReduce | Ring/Tree over BLink (intra-node) |
| AllGather | BLink for intra-node; IB/ETH for inter-node |
| ReduceScatter | Intra-node BLink |
| Broadcast | Intra-node BLink |

BCCL handles BLink-attached peer GPUs natively for intra-node high-bandwidth communication. For inter-node, it falls back to InfiniBand or Ethernet via host NIC.

---

## 8. Assembler / ISA

### Biren GPU ISA (BGISA / SUPA ISA)

Biren's proprietary native ISA for the BR100/BR104. No public ISA documentation has been released (unlike NVIDIA PTX). Key characteristics inferred from Hot Chips 34:

| Attribute | Details |
|-----------|---------|
| Architecture class | SIMT (C-Warp based) |
| Execution model | Warp-based; threads in lock-step with predication for divergence |
| Precision support | FP32, TF32+, BF16, FP16, INT8 native |
| Memory model | Cache-based (300 MB L2, EU L1/SMEM) — NOT scratchpad |
| Compiler input | SUPA C++ (via BRCC) |
| ISA access | Proprietary / closed; no public PTX equivalent |

Unlike NVIDIA PTX (which is a public virtual ISA developers can target directly), Biren's GPU ISA is not publicly documented. Developers interact with the hardware exclusively through SUPA C++ and the BRCC compiler.

---

## 9. CUDA Migration Support

Biren has explicitly targeted CUDA migration as a competitive strategy:

1. **API compatibility layer**: `cuda2supa` migration tool performs automated `cuda` → `supa` namespace substitution
2. **BRCC CUDA mode**: BRCC can accept standard CUDA C++ source and compile for BR100 (limited compatibility)
3. **Library shims**: Biren provides drop-in replacements for cuBLAS, cuDNN, NCCL with SUPA-equivalent APIs
4. **Framework backends**: PyTorch/TF/Paddle backends mean framework users see minimal code changes
5. **Ecosystem support**: NVIDIA's tightening of CUDA export to China (2024) further motivates SUPA adoption

---

## Sources

- [HPCwire — Chinese Startup Biren Details BR100 GPU (BIRENSUPA overview)](https://www.hpcwire.com/2022/08/22/chinese-startup-biren-details-br100-gpu/)
- [Biren Technology Official Product Page — BR10X](https://www.birentech.com/BR10X.html)
- [AllAboutCircuits — Biren Takes on NVIDIA (SUPA ecosystem)](https://www.allaboutcircuits.com/news/chinese-startup-biren-technology-takes-on-nvidia-in-gpu-market/)
- [KR Asia — Software hurdle for semiconductor companies](https://kr-asia.com/why-software-may-be-a-bigger-hurdle-than-hardware-for-semiconductor-companies)
- [Digitimes — NVIDIA CUDA restrictions stir Chinese AI community](https://www.digitimes.com/news/a20240308PD212/nvidia-cuda-china-ai.html)
- [AInvest — Biren Technology China AI Chip Challenger](https://www.ainvest.com/news/biren-technology-china-ai-chip-challenger-quest-disrupt-nvidia-dominance-2508/)

---

# Investigation Update — 2026-08-08: BIRENSUPA GitHub Activity, Serving Stack, Token Factory

*Investigated 2026-08-08. Additive to the 2026-04-05 report above; nothing in the prior BIRENSUPA layer description is retracted.*

## 10. Two GitHub organizations — one live, one archived

Biren maintains two GitHub organizations, and only one is active.

### BIRENSUPA (active)

| Repo | Created | Last push | Language | Assessment |
|---|---|---|---|---|
| `Mooncake` (fork) | 2026-07-20 | 2026-08-08 | C++ | **Actively developed** — the only repo in the org with sustained pushes |
| `mmcv` (fork) | 2026-08-05 | 2026-08-05 | — | Newly forked; OpenMMLab CV operator layer |
| `biren-driver-management-tools-skill` | 2026-06-24 | — | — | Driver management tooling |
| `sglang` (fork) | 2026-05-06 | 2026-05-06 | — | **Static fork — zero pushes since creation.** Not an active port |
| `PaddleCustomDevice` | 2023 | — | — | Pre-existing PaddlePaddle backend (already covered above) |

### BirenTechnology (archived)

`ModelZoo` (last push 2024-12-17), `k8s-device-plugin` (2024-10-17), `go-brml` (2024-06-18) — **all three are archived**. The Kubernetes device plugin and the BRML management-library binding documented in the baseline report therefore live in a dormant org; the live driver tooling is now `biren-driver-management-tools-skill` in BIRENSUPA.

## 11. Mooncake fork — disaggregated serving is the substantive change

The Mooncake fork is the one genuinely new *architectural* layer in Biren's software stack since baseline. Mooncake is the KV-cache-centric disaggregated serving architecture: **prefill and decode run on separate pools**, with a KV-cache transfer engine moving cache blocks between them and a pooled multi-tier cache store.

Why it matters for this survey:

- The baseline report described inference on Biren hardware only as "vLLM-style serving" with SUPA attention kernels — a **monolithic** serving model.
- A Mooncake port implies Biren is targeting **prefill/decode disaggregation** with cross-node KV-cache movement, which places new demands on the interconnect (KV transfer bandwidth, not just collective bandwidth) and lines up with the BLink 2.0 memory-semantic / 1,024-GPU shared-memory-space direction announced at WAIC 2026.
- Creation 2026-07-20 is **three days after** the WAIC 2026 announcement; pushes continue through 2026-08-08.

**Not disclosed:** whether the fork carries a SUPA/BCCL transfer-engine backend, which Biren hardware it targets, or whether it is upstream-tracked. No release, documentation page, or announcement accompanies the repo.

## 12. SGLang fork — do not overstate

`BIRENSUPA/sglang` was created 2026-05-06 with `pushed_at` also 2026-05-06 — a **one-shot fork with zero subsequent commits**. It should be recorded as a fork, not as an SGLang port or an active integration. Any claim that Biren "supports SGLang" is unsupported by the repository state.

## 13. Token Factory — vendor claim, primary-sourced

Biren's own WAIC 2026 press release describes a **"Token Factory"** application framework:

| Claim | Detail |
|---|---|
| Cache hit rate | **95%+**, attributed to a five-level caching hierarchy |
| Cross-vendor heterogeneous co-inference | Co-developed with **China Telecom (中国电信)** |
| Throughput gain | **~20%** from the heterogeneous co-inference scheme |

This is **unaudited vendor marketing**, but it is published on Biren's own newsroom rather than being a third-party rumour — a stronger provenance than "unverified". No benchmark methodology, model, or workload is given for either figure.

The five-level caching claim and the Mooncake fork are consistent with each other: both point at a KV-cache-tiering strategy as Biren's main inference-efficiency lever.

## 14. Day-0 model enablement claims

Biren's newsroom publishes Day-0 enablement announcements for Chinese frontier models:

| Model | Date |
|---|---|
| MiniMax M3 | 2026-06-16 |
| Zhipu GLM-5.2 | 2026-06-26 |
| MiniMax H3 | 2026-08-03 |

Vendor claims; no throughput, latency, or accuracy data accompanies them.

## 15. Software implications of BLink 2.0 (open questions)

BLink 2.0's **in-network computing** offloads collectives into switch nodes. That would shift work out of BCCL's host/GPU execution path in the same way NVIDIA SHARP does for NCCL. **No BCCL version, API, or documentation for BLink 2.0 has been published.** Likewise, the memory-semantic 1,024-GPU shared memory space would require an addressing/allocation model above `supaMalloc` — none has been described. Both are recorded as not disclosed rather than inferred.

## 16. Not disclosed / not found

- No BIRENSUPA SDK version, release notes, or changelog published for the 壁砺 generation
- No BCCL / SUPA runtime detail for BLink 2.0 or the supernode
- No public ISA documentation (unchanged from baseline — BGISA remains closed)
- No MLPerf submission; no Hot Chips / ISCA / ISSCC 2026 paper found (Hot Chips program page returned HTTP 403 — "could not verify", not "absent")
- Token Factory: no benchmark methodology behind the 95% or ~20% figures

## Sources — added 2026-08-08

- [BIRENSUPA GitHub org REST API — repo creation and push timestamps](https://api.github.com/orgs/BIRENSUPA/repos)
- [BirenTechnology GitHub org REST API — all three repos archived, last push 2024-12-17](https://api.github.com/orgs/BirenTechnology/repos)
- [BIRENSUPA GitHub organization page](https://github.com/BIRENSUPA)
- [Biren WAIC 2026 press release — Token Factory (95%+ cache hit via five-level caching; China Telecom cross-vendor heterogeneous co-inference ~20% throughput)](https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/)
- [Biren newsroom index — Day-0 enablement items (MiniMax M3, GLM-5.2, MiniMax H3)](https://www.birentech.com/news/)
- [BIRENSUPA software product page](https://www.birentech.com/product/software/birensupa/)
- [BIRENSUPA developer documentation index](https://developer.birentech.com/Document_Hardware.html)
