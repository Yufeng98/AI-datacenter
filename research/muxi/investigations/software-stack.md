# MetaX (沐曦) MXMACA Software Stack Investigation

*as_of: 2026-09-13 (baseline investigation 2026-04-05; see the dated update sections at the end)*
*chip: muxi*
*device_class: GPU (China, 沐曦)*
*resource: software-stack*

---

## Summary

MetaX's software platform is called **MXMACA** (MetaX Compute Architecture / 沐曦异构计算平台). It is an explicitly CUDA-compatible heterogeneous computing platform, with structural parity across every layer: a C-like GPU kernel language, compiler, runtime libraries (math, deep learning), collective communications, and framework backends. The MXMACA 2.0 platform launched with the MXC500 training GPU in 2023 and has since evolved to include vLLM, PyTorch, and open-source components via the MetaX-MACA GitHub organization.

---

## Stack Overview

```
Framework Integration
  └── PyTorch (via MXMACA backend — torch.device("maca"))
  └── TensorFlow (via MXMACA compatibility layer)
  └── vLLM-MetaX (vLLM hardware plugin for MXMACA — cuda_alike backend)
  └── llama.cpp (MetaX GPU backend — community integration)

Compiler / IR
  └── MXMACA Compiler (CUDA C++-compatible kernel compiler)
  └── CUDA ON MACA (automated CUDA→MXMACA source translation tool)
  └── MXMACA JIT / runtime compilation

Op Library
  └── mxDNN (cuDNN analog — Conv, Attention, Norm, Pooling)
  └── mxBLAS (cuBLAS analog — GEMM, BLAS L1-L3)
  └── mxFFT (cuFFT analog — Fast Fourier Transform)

Kernel Library
  └── mxThrust (Thrust analog — Reduce, Scan, Sort)
  └── MXMACA math libraries

Runtime
  └── MXMACA Runtime (macaRT — cudart analog; macaMalloc, macaMemcpyAsync, macaStream/Event)
  └── MXMACA Driver API (context management, module loading)

Driver / Firmware
  └── MetaX Kernel Driver (.ko — Linux PCIe, IOCTL, DMA, IRQ)
  └── Kubernetes Device Plugin (mx-exporter; MetaX GPU resource monitoring)

Communication
  └── MXCCL (MetaX Collective Communication Library — NCCL analog)

Assembler / ISA
  └── MXMACA ISA (proprietary; not publicly documented)
```

---

## Layer Details

### Framework Integration

**PyTorch via MXMACA**
- Registers MetaX GPUs as `torch.device("maca")` in PyTorch
- Supports standard autograd, tensor operations, distributed training
- TorchInductor backend support (for `torch.compile()` on MetaX hardware)
- Integration path: MXMACA runtime registers as a CUDA-alike device

**TensorFlow**
- Supported through the MXMACA compatibility layer
- CUDA ON MACA provides API-level translation for TensorFlow GPU operations
- Supports end-to-end deep learning pipeline (data processing → training → inference)

**vLLM-MetaX** (`github.com/MetaX-MACA/vLLM-metax`)
- Hardware plugin for running vLLM on MetaX GPU
- Classified as `cuda_alike` backend (similar to ROCm)
- Requires recompilation of all kernels for MACA (not binary-compatible with CUDA)
- Tracks upstream vLLM closely: vllm-metax v0.13.0 aligns with vLLM v0.13.0 — **superseded 2026-08-08: latest is v0.22.0 (2026-07-27); see the update section at the end for the full release chain**
- Plugin architecture since v0.8.5 (migrated from fork to plugin approach)
- RFC to officially integrate maca as a supported backend in vLLM (#23157)
- Provides "near-native CUDA experiences on MetaX Hardware with MACA"
- Supports LLaMA, Qwen, and other major open-source LLMs

**DeepSeek resources**
- MetaX developer portal explicitly supports DeepSeek integration
- MXMACA platform optimized for DeepSeek inference workloads

**llama.cpp**
- Community-maintained MetaX GPU backend integration

---

### Compiler / IR

**MXMACA Compiler (MACA C++ Kernel Compiler)**
- nvcc analog for MetaX GPU hardware
- Compiles MUSA C++/MACA C++ GPU kernels with `<<<>>>` launch syntax
- Targets the proprietary MXMACA ISA binary
- User-friendly, C-like programming language highly compatible with CUDA C++
- Part of MXMACA SDK (version 2.0+)

**CUDA ON MACA (Source Migration Tool)**
- Automated CUDA-to-MACA source code translator
- Substitutes CUDA namespace and API calls with MXMACA equivalents:
  - `cudaMalloc` → `macaMalloc`
  - cuBLAS → mxBLAS, cuDNN → mxDNN, NCCL → MXCCL
- Enables migration of existing CUDA applications with minimal code changes
- Recompilation of CUDA kernels required (not binary translation)

**mcpy** (`github.com/MetaX-MACA/mcpy`)
- CuPy-compatible NumPy-like GPU array library for MACA
- Enables scientific Python code to run on MetaX GPUs

---

### Op Library

**mxDNN**
- cuDNN analog
- Convolution, Attention / SDPA, Normalization, Pooling, Activation
- FP32/BF16/FP16/INT8 precision (C500); adds FP8 (C600)
- Integrated into MXMACA SDK and PyTorch/TF backends

**mxBLAS**
- cuBLAS analog
- GEMM (General Matrix Multiply), BLAS L1–L3
- Optimized for MetaX tensor units
- Supports matrix epilogue fusion

**mxFFT**
- cuFFT analog
- Fast Fourier Transform operations

---

### Kernel Library

**mxThrust**
- Thrust analog
- Parallel primitives: Reduce, Scan, Sort, Transform

**MXMACA Math Libraries**
- Standard math library (sin, cos, exp, log, etc.) for GPU kernels
- Device-side math functions callable from MACA kernels

---

### Runtime

**MXMACA Runtime (macaRT / libmaca.so)**
- cudart analog
- `macaMalloc` / `macaFree`, `macaMemcpyAsync`
- `macaStreamCreate` / `macaStreamSync`
- `macaEventCreate` / `macaEventElapsedTime`
- `<<<grid, block, shmem, stream>>>` kernel launch support
- Device enumeration, context lifecycle, memory pooling

**MXMACA Driver API**
- Lower-level context management, module loading, JIT compilation
- Fine-grained memory control
- Analogous to CUDA Driver API (`cuInit`, `cuDeviceGet`, etc.)

---

### Driver / Firmware

**MetaX Kernel Driver (.ko)**
- Linux PCIe kernel module (MACA device driver)
- BAR mapping, IOCTL dispatch, DMA engine, interrupt handling
- mx-exporter: Prometheus-compatible monitoring tool for MetaX GPU resources
  - GPU utilization, memory usage, temperature metrics
  - Kubernetes-aware for cluster monitoring

**Kubernetes Device Plugin**
- Advertises MetaX GPU resources in Kubernetes clusters
- Health monitoring and multi-tenant resource partitioning

---

### Communication

**MXCCL (MetaX Collective Communication Library)**
- NCCL analog
- AllReduce, AllGather, ReduceScatter, Broadcast, Scatter, Gather
- Intra-node: over MetaXLink hardware fabric (ring/mesh topologies)
- Inter-node: Standard Ethernet / RoCE
- Tested with 10,000+ GPU clusters in Chinese commercial deployments

---

### Assembler / ISA

**MXMACA ISA**
- Proprietary SIMT ISA for MetaX GPU hardware
- Self-developed, fully independent instruction set architecture
- Warp-based execution with hardware divergence handling
- Supports FP32/BF16/FP16/INT8 (C500 generation); adds FP8 (C600 generation)
- **Not publicly documented** (similar to NVIDIA PTX — no public ISA spec)
- All kernel development targets MACA C++ through the MXMACA compiler

---

## GitHub / Open Source

| Repository | Description |
|-----------|-------------|
| `MetaX-MACA/vLLM-metax` | vLLM hardware plugin for MetaX GPU (cuda_alike backend) |
| `MetaX-MACA/mcpy` | CuPy-compatible NumPy GPU library for MACA |
| `MetaX-Repository` | Additional MetaX tooling/utilities |
| `MetaX-MACA` (org) | Main GitHub organization for MXMACA open-source components |

---

## CUDA Compatibility Summary

| CUDA Component | MXMACA Equivalent | Compatibility Method |
|---------------|------------------|---------------------|
| nvcc | MXMACA Compiler | Source-level recompile |
| CUDA Runtime | MXMACA Runtime (macaRT) | API parity (`macaMalloc`, etc.) |
| CUDA Driver | MXMACA Driver | API parity |
| cuDNN | mxDNN | Source-level shim |
| cuBLAS | mxBLAS | Source-level shim |
| NCCL | MXCCL | Source-level shim |
| Thrust | mxThrust | Source-level shim |
| cuFFT | mxFFT | Source-level shim |
| CuPy | mcpy | Source-level shim |
| CUDA source | CUDA ON MACA | Automated source translation |

Note: MXMACA is `cuda_alike` (not binary-compatible with CUDA). All kernels require recompilation targeting the MXMACA ISA. This is the same model as AMD ROCm (HIP) — source-level compatibility, not PTX-binary translation.

---

## Ecosystem Maturity

- **LLM inference**: vLLM-MetaX v0.13.0 supports LLaMA, Qwen, and other major models; tracks upstream vLLM closely
- **Training**: PyTorch backend operational; TensorFlow supported; 100B+ parameter LLMs trained on MXC500 clusters
- **Scaling**: MXCCL + MetaXLink proven at 10,000-GPU cluster scale (China commercial deployments)
- **Developer portal**: `developer.metax-tech.com` — SDK downloads, documentation, DeepSeek resources
- **Limitations**: Smaller ecosystem than CUDA; all kernels require recompilation (no binary CUDA compatibility); limited public ISA documentation; ecosystem breadth behind NVIDIA at ~2–3-year lag

---

## Sources

- [MetaX Official — MXMACA Heterogeneous Platform](https://www.metax-tech.com/en/goods/platform.html?cid=4)
- [MetaX-MACA GitHub Organization](https://github.com/metax-maca)
- [MetaX-MACA/vLLM-metax GitHub](https://github.com/MetaX-MACA/vLLM-metax)
- [MetaX-MACA/mcpy GitHub](https://github.com/MetaX-MACA/mcpy)
- [vLLM GitHub Issue #23157 — RFC MetaX maca backend](https://github.com/vllm-project/vllm/issues/23157)
- [HAMi/docs/metax-support.md](https://github.com/Project-HAMi/HAMi/blob/master/docs/metax-support.md)
- [MetaX developer portal (driver install)](https://developer.metax-tech.com/api/client/document/preview/694/C500_DriverInstallationGuide_CN.html)
- [MetaX mx-exporter manual](https://developer.metax-tech.com/api/client/document/preview/369/C500_mxexporterManual_CN.html)
- [Rise VAST MetaX compatibility](https://www.theriseunion.com/blog/metax-compatibility.html)
- [FinancialContent — MetaX MXMACA ecosystem](https://markets.financialcontent.com/buffnews.buffalonewscom/article/tokenring-2025-11-1-chinese-ai-challenger-metax-ignites-fierce-battle-for-chip-supremacy-threatening-nvidias-reign)
- [MXC500 benchmark blog](https://wangjunjian.com/mxc500/benchmark/2025/02/13/Performance-Stress-Testing-of-the-MuXin-MXC500-for-Large-Model-Inference.html)
- [siiRL MetaX quickstart](https://siirl.readthedocs.io/en/latest/hardware_tutorial/metax_quickstart.html)

---

# Investigation Update — 2026-08-08

*Scan window: 2026-04-05 → 2026-08-08. Post-verification, authoritative version.*

*Primary sources: GitHub REST API (`api.github.com/repos/MetaX-MACA/vLLM-metax/releases`, `api.github.com/orgs/metax-maca/repos`), MetaX newsroom. Medium-confidence secondary: Baidu Baike / CSDN-tier write-ups for MXMACA release internals.*

## 1. MXMACA platform cadence

The baseline record ("MXMACA 2.0") is stale.

| Fact | Value | Confidence |
|---|---|---|
| MXMACA open-sourced | **2025-02-14** | confirmed |
| Latest recorded release | **3.3.0.X**, December 2025 | medium |
| PyTorch adaptation in 3.3.0.X | **PyTorch 2.8** | medium |
| Operator coverage in 3.3.0.X | **2,650 core operators**, of which **2,410 GPU operators** | medium |
| CUDA migration claim | **~92.94% seamless CUDA-project migration** | medium (vendor-adjacent claim) |
| Framework addition in 3.3.0.X | **native PaddlePaddle (飞桨)** support | medium |

⚠️ **Confidence caveat.** The 3.3.0.X specifics (PyTorch 2.8, 2,650 / 2,410 operators, 92.94% migration, PaddlePaddle) rest on **Baidu Baike and CSDN-tier sources, not a vendor changelog or release note**. They are recorded at **medium confidence**. What *is* clearly established is that "MXMACA 2.0" no longer describes the current stack.

### Community and ecosystem figures — all vendor claims

| Claim | Value | As of |
|---|---|---|
| Registered open-source community developers | **300,000+** | March 2026 |
| Developer users (MetaX WAIC 2026 newsroom) | **500,000+** | July 2026 |
| AI frameworks natively supported | **40+** | — |
| Models runnable | **500+** | — |
| Day-0 model adaptations completed | **30** | since December 2025 |

These are MetaX's own numbers and are not independently verified. Note the 300,000 → 500,000 jump across four months is itself a marketing figure, not an audited metric.

## 2. vLLM-metax release chain — CONFIRMED exactly from the GitHub releases API

The baseline record ("v0.13.0") was badly stale. Full chain past v0.13.0:

| Release | Date | Notes |
|---|---|---|
| v0.13.0 | 2026-02-06 | prior repo baseline |
| **v0.14.0** | 2026-03-23 | |
| **v0.15.0** | 2026-03-26 | |
| **v0.17.0** | 2026-04-15 | adds vLLM 0.16 / 0.17 |
| **v0.18.0** | 2026-05-06 | |
| **v0.19.0** | 2026-05-13 | full **Gemma 4** support |
| **v0.20.0** | 2026-06-03 | **DeepGEMM** scheduler-metadata buffer; **fused MoE grouped topk** |
| **v0.21.0** | 2026-07-10 | **DeepSeek V4** rebase |
| **v0.22.0** | 2026-07-27 | latest |

There is no v0.16.0 release tag in the API; v0.17.0 is the release that brings in upstream vLLM 0.16 and 0.17.

Release velocity: nine releases in under six months, i.e. roughly a 2-3 week cadence tracking upstream vLLM. That is materially faster than the baseline text implied.

## 3. New `github.com/metax-maca` components — creation dates from the GitHub API

| Repo | Created | Stack layer | Role |
|---|---|---|---|
| **hpc_warp** | 2026-01-13 | Kernel Library | HPC warp-level primitives |
| **McFlashInfer** | 2026-03-09 | Kernel Library | FlashInfer port — attention kernels |
| **mcTriton** | 2026-03-13 | Compiler / IR | **Triton port for MACA** — a second DSL front end alongside MACA C++ |
| **go-mxsml** | 2026-03-16 | Driver / Firmware | Go bindings for the MXSML management library (NVML analog) |
| **pymxsml** | 2026-03-19 | Driver / Firmware | Python bindings for MXSML |
| **sglang** (fork) | 2026-04-02 | Framework Integration | SGLang serving engine on MACA |
| **TileOPs-Metax** | 2026-04-24 | Compiler / IR | TileLang-based high-performance LLM operator library |
| **TileKernels-Metax** | 2026-04-24 | Compiler / IR | TileLang-based kernel library |
| **vllm-omni-metax** | 2026-04-24 | Framework Integration | Omni-modal vLLM variant |
| **AIModels** | 2026-05-18 | Framework Integration | Model zoo |
| **op_optimization** | 2026-05-26 | Kernel Library | Operator optimization work |
| **mccl_tests** | 2026-05-28 | Communication | MXCCL collective benchmark suite (nccl-tests analog) |
| **MXDeepEP** | 2026-06-10 | Communication | **DeepEP expert-parallel comms** — MoE dispatch/combine |
| **mcFlashMLA** | 2026-07-01 | Kernel Library | FlashMLA port — DeepSeek MLA attention |
| **FluidDynamics** | 2026-07-13 | AI4S | Computational fluid dynamics |
| **LifeScience** | 2026-07-13 | AI4S | Life sciences |
| **MedicalImage** | 2026-07-27 | AI4S | Medical imaging |
| **AI4S-Framework** | 2026-08-07 | AI4S | Umbrella AI4S framework |

Three of these (hpc_warp, AIModels, op_optimization) were missing from the raw scan and are included here.

### Two structural reads

**(a) The kernel-DSL layer arrived.** Before this window MetaX's only documented kernel authoring path was MACA C++ through the MXMACA compiler. Between March and April 2026 MetaX added **mcTriton** (Triton) plus **TileOPs-Metax / TileKernels-Metax** (TileLang). That is the same convergence Moore Threads, Enflame and other Chinese GPU vendors are making: adopt the portable kernel DSLs the LLM-serving ecosystem actually writes in, rather than expecting hand-ported CUDA C++.

**(b) The AI4S repo cluster mirrors the X300 hardware launch.** FluidDynamics and LifeScience (2026-07-13), MedicalImage (2026-07-27) and AI4S-Framework (2026-08-07) appear in the same weeks as the WAIC 2026 曦索 X300 announcement. The scientific-computing hardware line and its software line were staged together.

**(c) MoE/expert-parallel software lines up with the S600's EP claim.** MXDeepEP (2026-06-10) and the fused-MoE grouped-topk work in vLLM-metax v0.20.0 (2026-06-03) land immediately before the S600 supernode is announced with explicit EP-parallelism support. The comms library and the fabric were developed in parallel.

## 4. Ecosystem — MiniMax H3 Day-0 adaptation (CONFIRMED)

Per **MetaX's own newsroom, 2026-08-03**: 曦云 C-series GPUs completed **Day-0 adaptation of MiniMax's newly open-sourced multimodal model H3** via MXMACA. Moore Threads announced Day-0 adaptation of the same model on the same day with the MTT S5000.

This upgrades the raw scan's classification of this item from "medium confidence, single aggregator source" to **confirmed on a vendor primary source**.

## Sources — added 2026-08-08

- [vLLM-metax releases (GitHub API)](https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases)
- [metax-maca org repos (GitHub API)](https://api.github.com/orgs/metax-maca/repos)
- [MetaX-MACA GitHub org](https://github.com/metax-maca)
- [MetaX Newsroom — WAIC 2026 (developer-community claims)](https://www.metax-tech.com/en/ndetail/12629.html)
- [MetaX Newsroom — MiniMax H3 Day-0 adaptation (2026-08-03)](https://www.metax-tech.com/ndetail/12632.html)
- [Baidu Baike — MXMACA 软件栈 (MXMACA 3.3.0.X internals; medium confidence)](https://baike.baidu.com/item/MXMACA%E8%BD%AF%E4%BB%B6%E6%A0%88)

## Update — 2026-09-13

*as_of: 2026-09-13*
*scan window: 2026-08-08 → 2026-09-13*
*classification: Moderate — vLLM-metax release cadence continues; no new hardware; HK-listing rumor could not be corroborated*

### 1. vLLM-metax — two further releases in-window

Per the GitHub releases API, `MetaX-MACA/vLLM-metax` shipped two more releases inside the scan window, continuing the cadence already documented through v0.22.0 (2026-07-27):

| Release | Date | Notes |
|---|---|---|
| v0.23.0 | 2026-08-10 | (changelog not independently pulled this window) |
| v0.24.0 | 2026-08-27 | Sampling-performance optimization; refactor onto PyTorch's stable API; adds **JD JoyAI_LLM_Flash** model support; additional kernel test coverage. 73 commits since v0.23.0; contributors ILikeIneine, caozuoba, Mandaluoren, metax-yi1zhang |

Source: [vLLM-metax releases (GitHub API)](https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases), [v0.24.0 release notes](https://github.com/MetaX-MACA/vLLM-metax/releases/tag/v0.24.0).

### 2. Not corroborated this window: confidential HK listing filing

A pre-verified input for this scan asserted MetaX had made a "confidential HK filing, IPO targeted by end-2026." This survey's existing record already has MetaX STAR-listed since **December 2025** (second Chinese GPU stock after Moore Threads, per the repo's investigation baseline). Searches this window (Bing, Chinese-language) for a MetaX/沐曦 confidential Hong Kong listing filing returned **no results** — this could not be independently corroborated and is **not** added to the record. A share-lockup unlock (13.97M restricted shares, effective 2026-09-17) is scheduled just after this scan's cutoff and is unrelated to any HK listing.

### 3. No new hardware or MXC700/C700 tape-out news found

No C700 tape-out, no new MXC600/C500 spec, and no new AI4S (X-series) disclosure were found in this window beyond what the 2026-08-08 update already records.

## Sources — added 2026-09-13
- vLLM-metax releases (GitHub API): https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases
- vLLM-metax v0.24.0 release notes: https://github.com/MetaX-MACA/vLLM-metax/releases/tag/v0.24.0
