# Moore Threads MUSA Software Stack Investigation

*as_of: 2026-08-08 (baseline 2026-04-05; dated update section appended at end)*
*chip: mthreads*
*device_class: GPU (China, 摩尔线程)*
*resource: software-stack*

---

## Summary

> ⚠️ **Read the 2026-08-08 update section at the end of this file first.** The baseline sections below record MUSA SDK 4.0.1 as current (it is now 5.1.0), miss the torch_musa 2.9.x line, and predate the 2026 open-source and AI4Science expansion.

Moore Threads' software platform is called **MUSA** (Meta-computing Unified System Architecture / 摩尔线程统一系统架构). It is explicitly designed as a CUDA alternative, with structural and naming parity across every layer of the stack. The SDK includes a kernel compiler (MCC), runtime libraries, a CUDA migration tool (Musify / CUDA ON MUSA), math libraries (muBLAS, muDNN, muFFT, muThrust), a collective communication library (MCCL), and framework backends for PyTorch and vLLM. All key components are open-sourced on the MooreThreads GitHub organization.

---

## Stack Overview

```
Framework Integration
  └── torch_musa (PyTorch backend for MUSA)
  └── vllm-musa / vllm_musa (vLLM serving engine port)
  └── llama.cpp (MUSA backend, merged PR #8383)

Compiler / IR
  └── MCC (MUSA C Compiler — nvcc analog)
  └── Musify / CUDA ON MUSA (CUDA→MUSA source translator)
  └── TileLang MUSA (tilelang_musa — custom kernel DSL)
  └── TorchInductor + triton_musa backend

Op Library
  └── muDNN (cuDNN analog — Conv, Attn, Norm, Pooling)
  └── muBLAS (cuBLAS analog — GEMM, BLAS L1-L3)
  └── MTML / MATE (MUSA AI Tensor Engine — high-level ops)

Kernel Library
  └── muThrust (Thrust analog — Reduce, Scan, Sort, Scan)
  └── muFFT (cuFFT analog — FFT operations)
  └── tilelang_musa (CUTLASS analog — tile-level kernel authoring)

Runtime
  └── MUSA Runtime (musart — cudart analog)
  └── MUSA Driver API (lower-level context/module management)
  └── torchada (CUDA→MUSA compatibility layer for vLLM)

Driver / Firmware
  └── MUSA Kernel Driver (.ko — Linux PCIe/IOCTL/DMA)
  └── Kubernetes device plugin (MT GPU resource advertising)

Communication
  └── MCCL (MUSA Collective Communication Library — NCCL analog)
  └── MTLink fabric (hardware accelerated for intra-node)

Assembler / ISA
  └── MUSA ISA (proprietary SIMT ISA; not publicly documented)
```

---

## Layer Details

### Framework Integration

**torch_musa** (`github.com/MooreThreads/torch_musa`)
- Open-source PyTorch backend for MUSA GPUs
- Registers MTT GPUs as `torch.device("musa")`
- Implements 470+ ATen operators dispatched to MUSA kernels
- Supports MUSAGraph (CUDAGraph analog for stream capture)
- TorchInductor integration with `triton_musa` backend
- Custom LLM fusion modules (FlashAttention, RoPE, LayerNorm)
- FP8 dtype support (on S4000)
- MUSAExtension interface mirrors CUDAExtension for custom ops

**vllm-musa / vllm_musa** (`github.com/MooreThreads/vllm-musa`)
- vLLM port for running LLM inference on MTT GPUs
- Uses torchada (CUDA→MUSA compatibility shim)
- MATE (MUSA AI Tensor Engine) Python bindings
- Available on PyPI as `vllm-musa`
- Supports LLaMA, Qwen, Baichuan, other open-source LLMs

**llama.cpp MUSA backend**
- Merged via PR #8383 (`ggml-org/llama.cpp`)
- Enables llama.cpp inference on MTT GPUs

---

### Compiler / IR

**MCC (MUSA C Compiler)**
- nvcc analog for MUSA
- Compiles MUSA C++ kernel code (uses `<<<>>>` launch syntax, `__global__` functions, thread/block/grid model)
- Produces MUSA ISA binary for MTT GPU execution
- Included in MUSA SDK (version 4.0.1+)

**Musify / CUDA ON MUSA**
- Automated CUDA-to-MUSA source code translator
- Substitutes CUDA namespaces and API calls with MUSA equivalents
- Handles cuBLAS→muBLAS, cuDNN→muDNN, NCCL→MCCL shims
- Some PTX-level translation at runtime (similar to zluda)
- Supports x86 Intel (Ubuntu) and Hygon (Kylin OS) hosts

**TileLang MUSA** (`github.com/MooreThreads/tilelang_musa`)
- Domain-specific language for custom high-performance kernel authoring
- Deeply adapted for MUSA platform; modifies compilation pipeline to target MCC
- Generates MUSA C code from tile-level abstractions
- Analogous to Triton / CUTLASS template authoring on NVIDIA

**TorchInductor + triton_musa**
- TorchInductor backend uses triton_musa for kernel generation
- Enables `torch.compile()` on MTT GPUs (2024+)

---

### Op Library

**muDNN**
- cuDNN analog
- Convolution, Attention (SDPA), Normalization, Pooling, Activation
- Integrated into torch_musa and MUSA SDK
- Supports FP32/BF16/FP16/INT8 precision

**muBLAS**
- cuBLAS analog
- GEMM (General Matrix Multiply), BLAS L1/L2/L3
- Optimized for 128 Tensor Cores on Chunxiao die
- Epilogue fusion support

**MATE (MUSA AI Tensor Engine)**
- Higher-level op library for LLM inference primitives
- Python bindings via `mthreads-ml-py`
- Used by vllm-musa as primary inference backend

---

### Kernel Library

**muThrust**
- Thrust analog
- Parallel primitives: Reduce, Scan, Sort
- Device and host interfaces

**muFFT**
- cuFFT analog
- Fast Fourier Transform operations
- Used in scientific computing workloads

**TileLang MUSA** (also serves as kernel authoring framework)
- CUTLASS/Triton analog for manual kernel authoring
- Tile-level abstractions; compiles to MUSA C via MCC

---

### Runtime

**MUSA Runtime (musart / libmusa.so)**
- cudart analog
- `musaMalloc` / `musaFree`, `musaMemcpyAsync`
- `musaStreamCreate` / `musaStreamSync`
- `musaEventCreate` / `musaEventElapsedTime`
- `<<<grid, block, shmem, stream>>>` kernel launch syntax
- MUSA SDK 4.0.1 ships this

**MUSA Driver API**
- Lower-level context management, module loading
- Fine-grained memory control

**torchada**
- Thin CUDA→MUSA compatibility layer used by vllm-musa
- Intercepts CUDA API calls and redirects to MUSA runtime

---

### Driver / Firmware

**MUSA Kernel Driver (.ko)**
- Linux PCIe kernel module
- BAR mapping, IOCTL dispatch, DMA engine, interrupt handling
- Supports Ubuntu (Intel x86) and Kylin OS (Hygon x86)

**Kubernetes Device Plugin**
- Advertises MTT GPU resources in Kubernetes clusters
- Health monitoring and multi-tenancy support

---

### Communication

**MCCL (MUSA Collective Communication Library)**
- NCCL analog
- AllReduce, AllGather, ReduceScatter, Broadcast, Scatter, Gather
- Intra-node: over MTLink 1.0 hardware fabric (ring/tree topologies)
- Inter-node: Ethernet / RoCE
- Tested in 10,000-GPU clusters (KUAE server deployments)

---

### Assembler / ISA

**MUSA ISA**
- Proprietary SIMT ISA for MTT GPU hardware
- Based on C-Warp-like warp execution with hardware divergence
- Supports FP32/TF32/BF16/FP16/INT8 precisions (Chunxiao)
- FP4/FP64/MTFP6/MTFP4 added in Flower Harbor (Gen 3)
- Not publicly documented (unlike NVIDIA PTX)
- All development targets MUSA C++ via MCC compiler

---

## GitHub / Open Source

| Repository | Description |
|-----------|-------------|
| `MooreThreads/torch_musa` | PyTorch MUSA backend (primary) |
| `MooreThreads/vllm-musa` | vLLM MUSA fork |
| `MooreThreads/vllm_musa` | Alternative vLLM MUSA repo |
| `MooreThreads/tilelang_musa` | TileLang kernel authoring for MUSA |
| `ggml-org/llama.cpp` | PR #8383 — MUSA backend merged |
| `sgl-project/sglang` | Issue #16565 — SGLang MUSA integration (WIP) |

---

## CUDA Compatibility Summary

| CUDA Component | MUSA Equivalent | Compatibility Method |
|---------------|-----------------|---------------------|
| nvcc | MCC | Source-level recompile |
| CUDA Runtime | MUSA Runtime | API parity (musaMalloc, etc.) |
| CUDA Driver | MUSA Driver | API parity |
| cuDNN | muDNN | Source-level shim |
| cuBLAS | muBLAS | Source-level shim |
| NCCL | MCCL | Source-level shim |
| Thrust | muThrust | Source-level shim |
| cuFFT | muFFT | Source-level shim |
| CUDA source | Musify tool | Automated translation |
| CUDAExtension | MUSAExtension | Direct API parity |
| CUDAGraph | MUSAGraph | Implemented in torch_musa |
| torch.compile | TorchInductor+triton_musa | Implemented in torch_musa |

---

## Ecosystem Maturity

- **LLM training**: MTT S4000 clusters successfully trained 3B-parameter LLMs (reported by Tom's Hardware, 2024)
- **LLM inference**: vllm-musa supports LLaMA/Qwen/Baichuan
- **Scaling**: 10,000-GPU cluster demonstrated with MTLink fabric
- **Limitations**: Much smaller ecosystem than CUDA; 4 MB L2 limits throughput for memory-intensive workloads; GDDR6 vs HBM gap vs H100/A100

---

## Sources

- [Tom's Hardware — MUSA / Musify CUDA alternative](https://www.tomshardware.com/pc-components/gpus/chinas-moore-threads-polishes-homegrown-cuda-alternative-musa-supports-porting-cuda-code-using-musify-toolkit)
- [WCCFTech — MUSA SDK China's CUDA alternative](https://wccftech.com/china-first-in-house-alternative-to-nvidias-cuda-emerges-online/)
- [GitHub — MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa)
- [GitHub — MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa)
- [GitHub — MooreThreads/tilelang_musa](https://github.com/MooreThreads/tilelang_musa)
- [The Register — Moore Threads 10K GPU cluster](https://www.theregister.com/2024/07/09/moore_threads_10k_cluster/)
- [Tom's Hardware — 10K GPU MTLink](https://www.tomshardware.com/pc-components/gpus/chinese-gpu-maker-moore-threads-can-now-scale-to-10000-processors-for-ai-clusters-mtlink-fabric-tech-competes-with-nvidias-nvlink)
- [SGLang GitHub issue #16565 — MUSA support](https://github.com/sgl-project/sglang/issues/16565)
- [llama.cpp PR #8383 — MUSA backend](https://github.com/ggml-org/llama.cpp/pull/8383)
- [Turtles AI — MUSA SDK overview](https://www.turtlesai.com/en/pages-2666/musa_sdk_china_responds_to_cuda_with_its_own)

---

# Investigation Update — 2026-08-08: MUSA SDK 5.x, torch_musa 2.9.x, and the AI4Science port wave

*investigated: 2026-08-08*
*method: GitHub Releases API for torch_musa; GitHub org repos API (sort=created) for authoritative repo creation dates*

## 0. What this update corrects

- **"MUSA SDK 4.0.1" is two major versions stale.** The current line is **MUSA SDK 5.1.0**.
- The SDK version must be attached to the right release: **torch_musa v2.9.0 (2026-03-17) requires MUSA SDK >= 4.3.2**, *not* 5.x. Only **v2.9.1 (2026-06-29) requires MUSA SDK >= 5.1.0**.
- The TF32 opt-in environment variable is **`TORCH_ALLOW_TF32_MUBLAS_OVERRIDE=1`** — not `TORCH_ALLOW_TF32_MUBLAS`.
- **mutlass is not a new component.** `MooreThreads/mutlass` was created 2024-09-29. What is new is its **promotion to a first-class third-party dependency of torch_musa** in v2.9.1.

## 1. torch_musa releases (GitHub release notes)

| Release | Published | MUSA SDK requirement | Contents |
|---------|-----------|---------------------|----------|
| **v2.9.1** | 2026-06-29 | **>= 5.1.0** | "Build torch_musa v2.9.1 on MUSA platform with MUSA SDK >= 5.1.0". Integrates SDK 5.1.0 components with full CUDA-aligned operator coverage across **dense, quantized, sparse, sparsecsr and nested** tensor types. Makes **mutlass** (MUSA Templates for Linear Algebra Subroutines — the CUTLASS analog) a third-party repository of torch_musa, supplying high-performance matmul kernels |
| **v2.9.0** | 2026-03-17 | **>= 4.3.2** | Tracks PyTorch 2.9. Adds **Context Parallel (Ulysses) in FSDP2**; **sparse tensor operators**; `torch.compile` **"reduce-overhead"** mode; GEMM kernels become **FP32 by default** with TF32 opt-in via **`TORCH_ALLOW_TF32_MUBLAS_OVERRIDE=1`**. Known issue: `torch.compile` kernel performance regressed vs v2.7.0 |

The FP32-by-default GEMM change is the most consequential numerics change for anyone reproducing results across torch_musa versions: pre-2.9.0 runs may have used TF32 GEMM implicitly.

## 2. New Moore Threads open source (creation dates verified via GitHub API)

| Repo | Created | Stack layer | Role |
|------|---------|------------|------|
| `mutlass` | **2024-09-29** (not new) | Kernel Library | CUTLASS analog; newly promoted to a torch_musa dependency in v2.9.1 |
| `paddle_musa` | 2025-09-10 | Framework Integration | PaddlePaddle MUSA backend |
| `mate` | 2025-12-10 | Op Library | MUSA AI Tensor Engine, now its own repo |
| `torchada` | 2026-01-04 | Runtime | CUDA-compat adapter promoted from an inline shim to its own repo |
| `mthreads-ml-py` | 2026-01-04 | Driver / Management | GPU management and monitoring Python bindings (nvidia-ml-py analog) |
| `tvm_musa` | 2026-01-09 | Compiler / IR | Apache TVM MUSA target |
| `tilelang_musa` | 2026-01-12 | Compiler / Kernel | TileLang for MUSA (repo creation date) |
| `tensorflow_musa_extension` | 2026-02-05 | Framework Integration | TensorFlow backend extension — supersedes the "TensorFlow only via CUDA ON MUSA" note above |
| `tvm-ffi` | 2026-03-24 | Compiler / IR | TVM FFI layer |
| `MTClaw` | 2026-05-18 | Tooling | Local tool-routing proxy for openclaw / opencode / hermes |
| `TileOPs` | 2026-05-29 | Kernel Library | TileLang-based high-performance LLM operator library |
| `ompi-musa` | 2026-07-21 | Communication | Open MPI port |
| `onnxruntime-musa` | 2026-07-27 | Framework Integration | ONNX Runtime execution provider |

## 3. HPC / AI4Science port wave — 16 repos, 2026-06-11 → 2026-08-07

Earlier and larger than initially reported (the "nine repos in two weeks, 2026-07-27 to 2026-08-07" framing was wrong on start date, count and several individual dates).

| Created | Repos | Domain |
|---------|-------|--------|
| 2026-06-11 | `SpFFT-MUSA`, `flann-musa`, `eigen-musa` | Sparse FFT, ANN search, dense linear algebra |
| 2026-06-12 | `kokkos-musa` | Performance-portability layer |
| 2026-07-21 | `ompi-musa` | MPI |
| 2026-07-27 | `relion-musa`, `fused-ssim-musa`, `onnxruntime-musa` | Cryo-EM, image metrics, inference runtime |
| 2026-07-30 | `cp2k-musa`, `magma-musa`, `amgx-musa` | Quantum chemistry, dense LA on GPU, algebraic multigrid |
| 2026-08-07 | `lammps-musa`, `su2-musa`, `deepmd-kit-musa`, `mahout-musa`, `CV-CUDA_musa` | Molecular dynamics, CFD, ML interatomic potentials, distributed linear algebra, CV preprocessing |

**Why this matters for the survey.** Kokkos and Open MPI are *enabling layers*, not applications — porting them signals an intent to make MUSA a general scientific-computing target rather than a per-app porting exercise. CP2K, LAMMPS, RELION, SU2, MAGMA and AMGX are all FP64-dominant codes, which is directly consistent with the PingHu (Gen 4 / PH100 / MTT S5000) generation adding vendor-confirmed **full-precision FP8 → FP64** support. Software and silicon corroborate each other here: this is a strategy shift beyond LLM training, and it is the strongest evidence in this update that PingHu's FP64 is a real hardware capability rather than a marketing line.

## 4. Open items

- MUSA SDK 5.x release notes themselves were not retrieved; the 5.1.0 requirement is known only through torch_musa's release notes
- Whether `vllm-musa` / `sglang` have been updated for SDK 5.x was not verified
- No MLPerf submission found; no Hot Chips 2026 or ISCA 2026 Moore Threads paper found

## 5. Sources for this update

- https://api.github.com/repos/MooreThreads/torch_musa/releases — v2.9.1 (2026-06-29, SDK >= 5.1.0, mutlass third-party dep); v2.9.0 (2026-03-17, SDK >= 4.3.2, FSDP2 Context Parallel, sparse ops, reduce-overhead, TORCH_ALLOW_TF32_MUBLAS_OVERRIDE)
- https://api.github.com/orgs/MooreThreads/repos?sort=created&direction=desc&per_page=100 — authoritative creation dates for every repo listed above
- https://github.com/MooreThreads — org landing page
- https://en.mthreads.com/product/S5000 — "fourth-generation MUSA full-stack platform"; FP8→FP64 (the hardware capability the AI4Science push depends on)
