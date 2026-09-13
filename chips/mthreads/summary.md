# Moore Threads (摩尔线程) MTT GPU Software and Hardware Stack Summary

*as_of: 2026-08-08 (baseline research 2026-04-05)*
*chip: mthreads*
*device_class: GPU (China, 摩尔线程)*

---

## Overview

Moore Threads (摩尔线程, Moore Threads Technology), founded in October 2020 in Beijing by **Zhang Jianzhong** (张建中) — former NVIDIA Global Vice President and General Manager of Greater China for 15 years — is China's closest full-GPU analog to NVIDIA. Unlike other Chinese AI chip startups that build inference ASICs, Moore Threads designs full-function SIMT GPUs targeting **AI training, inference, graphics rendering, video encode/decode, and scientific compute on a single chip**.

The company's **MUSA platform** (Meta-computing Unified System Architecture) is an explicit CUDA alternative, with structural parity across every stack layer: MCC compiler (nvcc), MUSA Runtime (cudart), muDNN (cuDNN), muBLAS (cuBLAS), MCCL (NCCL), MTLink (NVLink), and torch_musa (PyTorch CUDA backend). The Musify tool automates CUDA→MUSA source migration.

The current shipping AI flagship is the **MTT S5000** (PH100 die, **PingHu 平湖** architecture, fourth-generation MUSA, FP8→FP64), which entered accelerated mass production during 2026 and superseded the **MTT S4000** (Chunxiao die, TSMC 12nm, 48 GB GDDR6, 200 TFLOPS BF16, 450W, MTLink 1.0) as the flagship. The *forthcoming* **Huashan (华山)** AI GPU and **Lushan (庐山)** graphics GPU sit one generation further out on the **fifth-generation Huagang 花港 ("Flower Harbor")** architecture, announced 2025-12-20 at MDC 2025 for 2026 mass production. See "MTT S5000 / PingHu and MTT C256 Update (2026-08-08)" below — the earlier labelling of Huashan/Huagang as "Gen 3" throughout this repo was wrong and is corrected there.

Moore Threads completed China's **largest STAR Market IPO of 2025** on November 24, 2025, raising **RMB 8 billion (~USD 1.1 billion)** — approved in a record 88 days. Shares surged **425%** on debut, valuing the company at nearly **USD 40 billion**. Founder Zhang Jianzhong's net worth reached approximately **USD 4.3 billion**.

---

## Software Stack

### Framework Integration

- **torch_musa** (open-source, `github.com/MooreThreads/torch_musa`): PyTorch MUSA backend. Registers MTT GPUs as `torch.device("musa")`. 470+ operators. MUSAGraph, TorchInductor, triton_musa, FP8 support, custom LLM fusion modules.
- **vllm-musa** (open-source, `github.com/MooreThreads/vllm-musa`): Full vLLM port for LLM inference on MTT GPUs. Supports LLaMA, Qwen, Baichuan. Uses torchada CUDA compatibility shim + MATE tensor engine.
- **llama.cpp**: MUSA backend merged (PR #8383). Direct GGML inference on MTT GPUs.

### Compiler / IR

- **MCC (MUSA C Compiler)**: nvcc analog. Compiles MUSA C++ GPU kernels (`<<<>>>` launch syntax) to the proprietary MUSA ISA binary. Shipped in the MUSA SDK (4.0.1 at the time of the 2026-04 baseline; **MUSA SDK 5.1.0** is the current line — see the 2026-08-08 update section).
- **Musify / CUDA ON MUSA**: Automated CUDA-to-MUSA source translator. Substitutes cuBLAS→muBLAS, cuDNN→muDNN, NCCL→MCCL. Enables migration of existing CUDA applications.
- **TileLang MUSA** (`tilelang_musa`): Tile-level kernel DSL, Triton/CUTLASS analog. Compiles tile-level abstractions to MUSA C via MCC. Suited for custom GEMM/Attention kernel authoring.
- **TorchInductor + triton_musa**: Enables `torch.compile()` on MTT GPUs via TorchInductor → triton_musa → MCC pipeline.

### Op Library

- **muDNN**: cuDNN analog. Convolution, SDPA/Attention, Normalization, Pooling, Activation in FP32/BF16/FP16/INT8.
- **muBLAS**: cuBLAS analog. GEMM, BLAS L1–L3. Optimized for 128 Tensor Cores on Chunxiao die.
- **MATE (MUSA AI Tensor Engine)**: Higher-level LLM inference primitives; Python bindings via `mthreads-ml-py`.

### Kernel Library

- **muThrust**: Thrust analog. Parallel primitives (Reduce, Scan, Sort).
- **muFFT**: cuFFT analog. FFT operations.
- **TileLang MUSA**: CUTLASS/Triton analog for manual tiled kernel authoring.

### Runtime

- **MUSA Runtime (`libmusa.so`)**: cudart analog. `musaMalloc`, `musaFree`, `musaMemcpyAsync`, `musaStreamCreate`, `musaEventCreate`. Full `<<<>>>` launch support.
- **MUSA Driver API**: Explicit context management, module loading, fine-grained memory control.
- **torchada**: Thin CUDA→MUSA compatibility shim used by vllm-musa.

### Driver / Firmware

- **MUSA Kernel Driver (.ko)**: Linux PCIe kernel module. BAR mapping, IOCTL, DMA, interrupts. Supports Ubuntu (Intel x86) and Kylin (Hygon x86).
- **Kubernetes Device Plugin**: MTT GPU resource advertising and health monitoring.

### Communication

- **MCCL (MUSA Collective Communication Library)**: NCCL analog. AllReduce, AllGather, ReduceScatter. Intra-node over MTLink 1.0 hardware; inter-node over Ethernet/RoCE. Tested at 10,000-GPU cluster scale.

### Assembler / ISA

- **MUSA ISA**: Proprietary SIMT ISA for MTT hardware. Warp-based execution with hardware divergence. Supports FP32/TF32/BF16/FP16/INT8 on Chunxiao. **Not publicly documented** (no PTX equivalent). **PingHu (Gen 4, PH100/S5000)** adds vendor-confirmed "full-precision compute support from FP8 to FP64". Huagang / Flower Harbor (**Gen 5**, not Gen 3 as previously recorded here) is announced to add FP4, MTFP6 and MTFP4.

---

## Hardware Architecture

### Compute Engine (MTT S4000 / Chunxiao)

The Chunxiao die is organized into **MP (MUSA Processors)** — analogous to NVIDIA SMs:

- **4,096 SP (Stream Processors)** organized into MPs
- Each MP contains: 128× FP32, 32× INT8/bitwise, 32× SFU, 2× FP64, TCE (Tensor Compute Engine), 28 KB local shared memory
- **128 Tensor Cores** (TCE — accelerates FP16/BF16/INT8 matrix ops)
- 256 Texture Units, 256 ROPs
- **22 billion transistors** (TSMC 12nm)

| Metric | MTT S4000 |
|--------|-----------|
| Peak FP32 | 25 TFLOPS |
| Peak TF32 | 50 TFLOPS |
| Peak BF16/FP16 | 200 TFLOPS |
| Peak INT8 | 200 TOPS |
| TDP | 450W |
| Form factor | PCIe Gen5 x16 FHFL |

### Memory Hierarchy

```
Per-MP: 28 KB local shared memory
         ↓
On-chip: 4 MB L2 cache (HW-managed)  [small vs A100 ~40 MB]
         ↓
Off-chip: GDDR6
  S4000: 48 GB, 384-bit, 768 GB/s
  S3000: 32 GB, 256-bit, ~512 GB/s
  S80:   16 GB, 256-bit, ~512 GB/s
```

### Interconnect (MTT S4000 / Chunxiao era)

| Layer | Detail |
|-------|--------|
| Scale-up | MTLink 1.0 (NVLink analog) |
| MTLink BW | 240 GB/s/GPU |
| Max per server | 8 GPUs (KUAE D800 / MCCX D800) |
| Max cluster | 10,000 GPUs |
| Scale-out | Standard Ethernet / RoCE via host NIC |
| Host interface | PCIe Gen5 x16 |

*For the S5000 generation the 8-GPU node names change to **MTT MGX** (8-OAM modular platform) and **MTT SGX5000** (8× S5000 server), and the announced **MTT C256** supernode claims a 128-card-per-cabinet / 256-card two-cabinet scale-up domain. See the 2026-08-08 update section; the S5000 per-link MTLink bandwidth and MTLink generation number are **not disclosed**.*

---

## Roadmap

Architecture generations are named after the Ten Scenes of West Lake. The corrected generation map (verified 2026-08-08) is: **Gen 1 苏堤 Sudi → Gen 2 春晓 Chunxiao → Gen 3 曲院 Quyuan → Gen 4 平湖 PingHu → Gen 5 花港 Huagang**.

| Product | Architecture (gen) | Die | Status | Key Specs |
|---------|-------------------|-----|--------|-----------|
| MTT S4000 | Chunxiao 春晓 (Gen 2) | TSMC 12nm | GA (2024) | 48 GB GDDR6, 768 GB/s, 200 TFLOPS BF16, 450W, MTLink 1.0 240 GB/s |
| **MTT S5000** | **PingHu 平湖 (Gen 4)** | **PH100 (process not disclosed)** | **Shipping — accelerated mass production 2026** | Vendor: FP8→FP64 full-precision, fourth-gen MUSA. Media-reported only: 80 GB, 1.6 TB/s, 784 GB/s card-to-card, up to 1,000 TFLOPS dense FP8. TDP / process / memory type **not disclosed** |
| Huashan (华山) AI GPU | Huagang 花港 "Flower Harbor" (**Gen 5**) | Advanced node (undisclosed) | Announced 2025-12-20 (MDC 2025); 2026 mass production | 8× HBM, dual chiplet, between H100–B200 (company claim), FP4/FP64, MTFP6/MTFP4 |
| Lushan (庐山) Gaming GPU | Huagang 花港 (**Gen 5**) | — | Announced 2025-12-20; 2026 status unverified | 15× gaming, 50× RT vs S80, DX12 Ultimate, 64 GB GDDR7 est. |

> **Correction (2026-08-08):** earlier revisions of this file labelled Huashan / Flower Harbor as "Gen 3". That was wrong on two counts — Huagang is the **fifth** generation, and the intervening PingHu generation (Gen 4) is what actually shipped as the MTT S5000. Huashan and the S5000 are **different chips one generation apart**, not two names for the same part.

---

## Business Context

| Event | Date | Detail |
|-------|------|--------|
| Founded | Oct 2020 | Beijing; Zhang Jianzhong (15 yrs NVIDIA VP Greater China) |
| MTT S2000 (first DC GPU) | Apr 2022 | 12nm, 32 GB, 12 TFLOPS FP32 |
| MTT S80 launch | 2022 | Full Chunxiao, gaming GPU |
| MTT S4000 launch | 2024 | 48 GB, 200 TFLOPS BF16, MTLink, PCIe Gen5 |
| MUSA SDK 4.0.1 | 2024 | MCC, Musify, muDNN, muBLAS, MCCL, torch_musa |
| 10K GPU cluster | Jul 2024 | MTLink fabric, KUAE servers |
| STAR Market IPO | Nov 24, 2025 | Shanghai; RMB 8B (~USD 1.1B); 88-day approval |
| IPO debut | Nov 24, 2025 | +425%; market cap ~USD 40B |
| Huagang (花港, Gen 5) announced | Dec 20, 2025 (MDC 2025) | Huashan (AI) + Lushan (graphics) chips, HBM, FP4 — for 2026 mass production |
| MTT S5000 specifications public | ~Feb 11–12, 2026 | PH100 / PingHu Gen 4; GLM-5 "Day-0" adaptation 2026-02-12 |
| torch_musa v2.9.0 | Mar 17, 2026 | Tracks PyTorch 2.9; MUSA SDK ≥ 4.3.2 |
| torch_musa v2.9.1 | Jun 29, 2026 | MUSA SDK ≥ 5.1.0; mutlass as third-party dependency |
| MUSA HPC / AI4Science port wave | Jun 11 – Aug 7, 2026 | 16 MUSA ports (cp2k, LAMMPS, RELION, SU2, MAGMA, AMGX, Kokkos, …) |
| MTT C256 supernode | Jul 17–20, 2026 (WAIC 2026) | First public demonstration; 128-card single-cabinet / 256-card two-cabinet single-layer Scale-up |

---

## MTT S5000 / PingHu and MTT C256 Update (2026-08-08)

*Updated 2026-08-08. Primary sources: Moore Threads S5000 product pages (en.mthreads.com/product/S5000, www.mthreads.com/product/S5000); MooreThreads GitHub org + torch_musa release notes (GitHub API, authoritative for dates); ITHome 2026-07-20 and Sina Finance 2026-07-19 for WAIC 2026; Tencent Cloud news 2026-02-13 for the media-reported S5000 figures.*

### 1. Generation map — the repo's prior labelling was wrong

Moore Threads names architectures after the Ten Scenes of West Lake:

| Gen | Architecture | Products |
|-----|-------------|----------|
| 1 | 苏堤 Sudi | MTT S10 / S30 / S50 / S1000 (also S60, S2000) |
| 2 | 春晓 Chunxiao | MTT S70 / S80 / S2000 / S3000 / S4000 |
| 3 | 曲院 Quyuan | (no datacenter part tracked here) |
| 4 | 平湖 PingHu | **PH100 → MTT S5000 — currently shipping** |
| 5 | 花港 Huagang ("Flower Harbor") | 华山 Huashan (AI) and 庐山 Lushan (graphics) — announced 2025-12-20, forthcoming |

The official English S5000 page states verbatim: *"With Moore Threads' next-generation PH100 chip at its core, and built on the advanced 'PingHu' architecture, MTT S5000 delivers full-precision compute support from FP8 to FP64,"* and *"Powered by the fourth-generation MUSA full-stack platform."* The official generation number is therefore **fourth**, not third; a minority of Chinese aggregators say "third-generation MUSA" and should not be followed. There is **no PingHu-vs-Huashan naming conflict** — they are two different chips one generation apart. What MDC 2025 announced on 2025-12-20 was the fifth-generation Huagang architecture and its two chips, not the S5000.

### 2. MTT S5000 (PH100, PingHu) — shipping

Detailed specifications became public around **2026-02-11/12** (the card also did a "Day-0" adaptation of GLM-5 on 2026-02-12) — *not* on 2025-12-20. Chinese media consistently report the part in accelerated mass production (已进入加速规模化量产阶段) with 10,000-card clusters in commercial service; a thousand-card S5000 cluster was used by BAAI / Zhiyuan (智源研究院) to train the **RoboBrain 2.5** embodied-intelligence model (ITHome, 2026-07-20). Treat "deployed at scale" as media-reported and vendor-corroborated.

**Two tiers of evidence — do not merge them.**

| | Vendor-published (mthreads.com / en.mthreads.com) | Media-reported only (medium confidence) |
|---|---|---|
| Chip / architecture | PH100; PingHu; fourth-generation MUSA | — |
| Data types | Full-precision **FP8 → FP64** | — |
| Memory capacity | *no spec table published* | 80 GB (type implied HBM; **HBM generation not confirmed**) |
| Memory bandwidth | *not published* | 1.6 TB/s |
| Card-to-card interconnect | *not published* | 784 GB/s |
| Peak dense FP8 | *not published* | up to 1,000 TFLOPS single-card |
| TDP | **not disclosed** | not disclosed |
| Process node / transistors | **not disclosed** | a circulating "7nm / 22B transistors" figure duplicates the S4000 Chunxiao transistor count and is **unconfirmed/suspect** — do not carry it |

> ⚠️ **Do not attribute 3.35 TB/s or 4.0 TB/s to the S5000.** Those figures appear on the official page only as *competitor comparison-baseline footnotes*: "reference benchmark 1: dense FP16 989 TFLOPS, memory bandwidth 3.35 TB/s" (NVIDIA H100 SXM) and a second baseline "dense FP16 148 TFLOPS, memory bandwidth 4.0 TB/s" (H20-class).

**Vendor marketing claims** (verbatim from the official product page — label as vendor claims, not independently confirmed): single-GPU prefill ≥ 4,000 tokens/s and decode ≥ 1,000 tokens/s; prefill throughput 2.5× "leading international flagship products" at 16K sequence length; Llama3-70B MFU > 60%; DeepSeek-236B MFU > 40%; 95% cluster linearity at 10K-GPU scale; 0.6% relative precision deviation for DeepSeek-236B on 10K GPUs.

**Separately** (and easy to conflate with the vendor's 0.6% figure): Chinese media (Tencent Cloud news, 2026-02-13) report a comparison in which a **thousand-card S5000 cluster's training loss differed from an H100 cluster by only 0.62%** on RoboBrain 2.5 — media report, low-to-medium confidence, not independently reproduced.

**Form factors (official page):** S5000 **OAM** compute modules in liquid- and air-cooled variants; **MTT MGX**, an 8-GPU modular platform with eight OAM modules interconnected over MTLink; and the **MTT SGX5000** server (8× MTT S5000). This supersedes the S4000-era "KUAE D800 / MCCX D800" node naming for the new generation.

### 3. MTT C256 supernode — announced / first publicly demonstrated

First shown at **WAIC 2026, 2026-07-17** (coverage 07-17 → 07-20); confirmed by multiple independent outlets (ITHome, Sina Finance / Fast Technology, Global Times).

- Claimed industry-first **single-layer Scale-up network** (首创单层Scale-up网络), breaking the prevailing 64-card single-layer limit
- **128 GPUs fully interconnected within one standard cabinet**, extensible to **256 cards across two standard cabinets** (128卡全互联，并柜扩展后可达256卡极速互联)
- **Sub-microsecond** card-to-card communication latency
- **2U node** design, retaining compatibility with existing datacenter hardware and software
- Positioned as a building block for 10,000- to 100,000-card clusters

**Status is announced/demonstrated (首次公开展示)** — no shipping, pricing or customer deployment is confirmed. Per-link MTLink bandwidth, the MTLink generation number and aggregate supernode bandwidth are **not disclosed**. Liquid cooling, three-way blind-mate connectors and RAS features circulating in secondary write-ups were **not confirmed** in the sources reviewed and are omitted here. If it ships as announced, this is a claimed ~32× increase in scale-up domain over the current 8-GPU server.

**Also at WAIC 2026:** Moore Threads framed its offering as three AI "factories" — a **model factory** (pretraining, post-training, RL), a **token factory** (native MoE support, million-token ultra-long-context inference), and an **agent factory** (high concurrency, claimed > 100 tokens/s per user). A marketing slogan sometimes quoted alongside this ("Token Era, Intelligence for All Things") was **not confirmed**.

### 4. Software — MUSA SDK moved to the 5.x line

The "MUSA SDK 4.0.1" recorded above is stale. From GitHub release notes:

| Release | Date | Requires | What's new |
|---------|------|----------|-----------|
| torch_musa **v2.9.0** | 2026-03-17 | MUSA SDK **≥ 4.3.2** (not 5.x) | Tracks PyTorch 2.9; Context Parallel (Ulysses) in FSDP2; sparse tensor operators; `torch.compile` "reduce-overhead" mode; GEMM kernels **FP32 by default** with TF32 opt-in via `TORCH_ALLOW_TF32_MUBLAS_OVERRIDE=1`. Known issue: `torch.compile` kernel performance regressed vs v2.7.0 |
| torch_musa **v2.9.1** | 2026-06-29 | MUSA SDK **≥ 5.1.0** | Integrates SDK 5.1.0 with full CUDA-aligned operator coverage across dense, quantized, sparse, sparsecsr and nested tensors; promotes **mutlass** (MUSA Templates for Linear Algebra Subroutines — the CUTLASS analog) to a first-class third-party dependency for high-performance matmul kernels |

Note: **mutlass itself is not new** (repo created 2024-09-29); what is new is its promotion to a torch_musa dependency.

**New Moore Threads open source not previously tracked here** (creation dates verified via the GitHub API):

| Repo | Created | Role |
|------|---------|------|
| `mate` (MUSA AI Tensor Engine) | 2025-12-10 | Op library — LLM inference primitives, now its own repo |
| `paddle_musa` | 2025-09-10 | Framework integration — PaddlePaddle backend |
| `torchada` | 2026-01-04 | Runtime — CUDA-compat adapter promoted to its own repo |
| `mthreads-ml-py` | 2026-01-04 | GPU management / monitoring Python bindings (nvml analog) |
| `tvm_musa` | 2026-01-09 | Compiler — Apache TVM MUSA target |
| `tilelang_musa` | 2026-01-12 | Compiler / kernel DSL (repo creation date) |
| `tensorflow_musa_extension` | 2026-02-05 | Framework integration — TensorFlow PluggableDevice-style extension |
| `tvm-ffi` | 2026-03-24 | Compiler FFI layer |
| `MTClaw` | 2026-05-18 | Local tool-routing proxy for openclaw / opencode / hermes |
| `TileOPs` | 2026-05-29 | Kernel library — TileLang-based high-performance LLM operator library |
| `ompi-musa` | 2026-07-21 | Communication — Open MPI port |
| `onnxruntime-musa` | 2026-07-27 | Framework integration — ONNX Runtime execution provider |

### 5. HPC / AI4Science expansion — a strategy shift

The MUSA scientific-computing porting push began **2026-06-11**, not late July, and covers **sixteen ports over roughly two months**:

| Date | Repos |
|------|-------|
| 2026-06-11 | `SpFFT-MUSA`, `flann-musa`, `eigen-musa` |
| 2026-06-12 | `kokkos-musa` |
| 2026-07-21 | `ompi-musa` |
| 2026-07-27 | `relion-musa`, `fused-ssim-musa`, `onnxruntime-musa` |
| 2026-07-30 | `cp2k-musa`, `magma-musa`, `amgx-musa` |
| 2026-08-07 | `lammps-musa`, `su2-musa`, `deepmd-kit-musa`, `mahout-musa`, `CV-CUDA_musa` |

The span — quantum chemistry (CP2K), molecular dynamics (LAMMPS, DeePMD-kit), cryo-EM (RELION), CFD (SU2), dense/sparse linear algebra (MAGMA, AMGX, Eigen, SpFFT) and performance-portability layers (Kokkos) — is a coherent move beyond LLM training, and is directly consistent with the PingHu generation adding native **FP64**. This is arguably the most survey-relevant software signal of the update.

### 6. Not confirmed / open

- No MLPerf Training or Inference submission found
- No Hot Chips 2026 or ISCA 2026 Moore Threads paper found (searched, not exhaustively)
- Lushan gaming GPU 2026 status unverified
- MTLink generation number for S5000 and C256 **not disclosed**
- S5000 memory type / HBM generation, TDP, process node and transistor count all **not disclosed** by the vendor

---

## Resources

### Official
- [MTT S5000 Product Page (EN)](https://en.mthreads.com/product/S5000) — PH100 / PingHu / fourth-generation MUSA / FP8→FP64; no spec table
- [MTT S5000 Product Page (CN)](https://www.mthreads.com/product/S5000) — confirms 3.35 TB/s and 4.0 TB/s are competitor comparison-baseline footnotes
- [MTT S4000 Product Page](https://en.mthreads.com/product/S4000)
- [Moore Threads About](https://en.mthreads.com/about)
- [MUSA Architecture Blog (Chinese)](https://blog.mthreads.com/blog/musa/2024-05-11-MUSA%E7%A1%AC%E4%BB%B6%E6%9E%B6%E6%9E%84%E4%B8%8EGPU%E5%B9%B6%E8%A1%8C%E7%A5%8D%E7%9B%AE%E5%9F%BA%E7%A1%80/)
- [Moore Threads GitHub](https://github.com/MooreThreads)

### Hardware
- [ITHome — MTT C256 supernode at WAIC 2026 (2026-07-20)](https://www.ithome.com/0/978/856.htm)
- [Sina Finance — Moore Threads WAIC 2026, single-layer Scale-up, three AI factories (2026-07-19)](https://finance.sina.com.cn/tech/roll/2026-07-19/doc-iniiiwhz3402415.shtml)
- [Tencent Cloud news — MTT S5000 figures and H100 loss comparison (2026-02-13)](https://cloud.tencent.com/developer/news/3588420)
- [VideoCardz — MTT S4000 48GB MTLink](https://videocardz.com/newz/moore-threads-introduces-mtt-s4000-48gb-ai-gpu-with-mtlink-and-zero-cost-nvidia-cuda-framework-translation)
- [Tom's Hardware — Chunxiao GPU unveil](https://www.tomshardware.com/news/moore-threads-unveils-chunxiao-gpu)
- [WCCFTech — Huashan Flower Harbor](https://wccftech.com/moore-threads-lushan-gaming-huashan-ai-gpus-15x-gaming-uplift-50x-rt-boost-dx12-ultimate-support/)
- [TrendForce — Huashan vs Hopper](https://www.trendforce.com/news/2025/12/22/news-chinas-moore-threads-unveils-huashan-ai-chip-reportedly-takes-aim-at-nvidias-hopper/)

### Software
- [Tom's Hardware — MUSA/Musify CUDA alternative](https://www.tomshardware.com/pc-components/gpus/chinas-moore-threads-polishes-homegrown-cuda-alternative-musa-supports-porting-cuda-code-using-musify-toolkit)
- [torch_musa GitHub](https://github.com/MooreThreads/torch_musa)
- [torch_musa releases (v2.9.0 / v2.9.1)](https://github.com/MooreThreads/torch_musa/releases)
- [vllm-musa GitHub](https://github.com/MooreThreads/vllm-musa)
- [TileLang MUSA GitHub](https://github.com/MooreThreads/tilelang_musa)
- [mutlass — MUSA Templates for Linear Algebra Subroutines (CUTLASS analog)](https://github.com/MooreThreads/mutlass)
- [TileOPs — TileLang-based LLM operator library](https://github.com/MooreThreads/TileOPs)
- [onnxruntime-musa](https://github.com/MooreThreads/onnxruntime-musa)
- [ompi-musa — Open MPI port](https://github.com/MooreThreads/ompi-musa)
- [MooreThreads GitHub org (repo creation dates, incl. the 2026 HPC port wave)](https://github.com/orgs/MooreThreads/repositories?sort=created)

### Business / IPO
- [TrendForce — STAR Market IPO RMB 8B](https://www.trendforce.com/news/2025/11/24/news-chinas-largest-star-market-ipo-of-2025-moore-threads-goes-public-on-nov-24-raising-rmb-8b/)
- [China Biz Insider — USD 1.1B listing](https://chinabizinsider.com/moore-threads-leads-chinas-chip-ipo-wave-with-record-us-1-1-billion-star-market-listing/)
- [SCMP — Founder Zhang Jianzhong profile](https://www.scmp.com/tech/big-tech/article/3327614/meet-moore-threads-how-former-nvidia-vice-president-created-chinese-gpu-star)
- [VnExpress — Zhang Jianzhong billionaire](https://e.vnexpress.net/news/tech/personalities/founder-of-moore-threads-china-s-nvidia-becomes-billionaire-after-blockbuster-ipo-4992428.html)
