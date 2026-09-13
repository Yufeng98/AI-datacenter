# MetaX (沐曦) GPU Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: muxi*
*device_class: GPU (China, 沐曦)*

---

## Overview

MetaX Integrated Circuits (Shanghai) Co., Ltd. (沐曦集成电路, 沐曦), founded in September 2020, is China's AMD-lineage full-stack GPGPU company. Unlike other Chinese AI chip startups targeting narrow inference or specialized workloads, MetaX designs general-purpose SIMT GPUs for AI training, inference, and graphics rendering. The company was founded by **Chen Weiliang** (陈维良) — 14 years at AMD, including as top executive in AMD Greater China — alongside co-CTOs **Peng Li** and **Yang Jian**, also AMD veterans. The founding team drew heavily from AMD Shanghai's engineering talent.

MetaX's **MXMACA platform** (MetaX Compute Architecture) is an explicit CUDA alternative — with structural parity across every stack layer: a MACA C++ kernel compiler (nvcc analog), MXMACA Runtime (cudart analog), mxDNN (cuDNN analog), mxBLAS (cuBLAS analog), MXCCL (NCCL analog), MetaXLink (NVLink analog), and vLLM-MetaX (vLLM plugin). The CUDA ON MACA tool automates CUDA→MACA source migration.

The current production AI training GPU is the **MXC550** (Xiyun C550, 7nm, OAM form factor, MetaXLink fabric, C550 Shanghai Cube: 128 liquid-cooled cards in 47U). The flagship shipping product as of 2026 is the **MXC600** (Xiyun C600: 1,000 TFLOPS FP8, HBM3e 144 GB at 3.6 TB/s, fully domestic supply chain — first MetaX GPU built without TSMC), targeting the NVIDIA Hopper H20 tier; MetaX states it reached large-scale shipment during 2026 (see the 2026-08-08 update section below).

As of mid-2026 the product line has widened well beyond the C-series: a 曦思 (Xisi) **N-series** inference line (N100, N260, N300), a 曦索 (Xisuo) **X-series** "scientific intelligence" (AI4S) line (X206, X301, X302), a 曦彩 (Xicai) **G-series** graphics line (G100), and a 曦景 (Xijing) **S-series** supernode line whose first product is the **S600**.

MetaX completed its **STAR Market IPO on December 17, 2025**, surging **+692% on debut**, raising **RMB 4.2 billion (~USD 594 million)**, and reaching a market cap exceeding **RMB 332 billion (~USD 47 billion)**. Chen Weiliang's personal stake reached ~USD 6.5 billion, making him one of China's newest tech billionaires. MetaX attracted more retail investors than Moore Threads in pre-IPO subscriptions.

---

## Software Stack

### Framework Integration

- **PyTorch**: Full backend via MXMACA platform. `torch.device("maca")`. Supports autograd, distributed training, `torch.compile` via TorchInductor.
- **TensorFlow**: Supported through CUDA ON MACA compatibility layer.
- **vLLM-MetaX** (open-source, `github.com/MetaX-MACA/vLLM-metax`): vLLM hardware plugin (cuda_alike backend). Version-aligned with upstream vLLM; **v0.22.0 (2026-07-27)** is the latest release (see the 2026-08-08 update section for the full release chain). Supports LLaMA, Qwen. RFC submitted to integrate maca natively into vLLM (#23157).
- **SGLang** (`github.com/metax-maca` fork, created 2026-04-02): MetaX fork of SGLang.
- **DeepSeek**: Supported via MXMACA developer portal.
- **llama.cpp**: Community MetaX backend integration.

### Compiler / IR

- **MXMACA Compiler**: nvcc analog. Compiles MACA C++ GPU kernels (`<<<>>>` syntax) to the proprietary MXMACA ISA binary.
- **CUDA ON MACA**: Automated CUDA→MACA source translator. Substitutes cuBLAS→mxBLAS, cuDNN→mxDNN, NCCL→MXCCL. Enables migration of existing CUDA code (requires recompilation).
- **mcpy** (`github.com/MetaX-MACA/mcpy`): CuPy-compatible NumPy GPU library for MACA.

### Op Library

- **mxDNN**: cuDNN analog. Conv, Attention/SDPA, Norm, Pooling, Activation. FP32/BF16/FP16/INT8 (C500); FP8 added in C600.
- **mxBLAS**: cuBLAS analog. GEMM, BLAS L1–L3, epilogue fusion.
- **mxFFT**: cuFFT analog. Fast Fourier Transform.

### Kernel Library

- **mxThrust**: Thrust analog. Reduce, Scan, Sort, Transform parallel primitives.
- **MXMACA Math Libraries**: Device-side standard math functions.

### Runtime

- **MXMACA Runtime (`libmaca.so`)**: cudart analog. `macaMalloc`, `macaFree`, `macaMemcpyAsync`, `macaStreamCreate`, `macaEventCreate`, full `<<<>>>` kernel launch.
- **MXMACA Driver API**: Context management, module loading, JIT compilation, fine-grained memory control.

### Driver / Firmware

- **MetaX Kernel Driver (.ko)**: Linux PCIe kernel module. BAR mapping, IOCTL, DMA, interrupt handling.
- **mx-exporter**: Prometheus-compatible GPU monitoring. Kubernetes-aware for cluster deployments.
- **Kubernetes Device Plugin**: MetaX GPU resource advertising and health monitoring.

### Communication

- **MXCCL (MetaX Collective Communication Library)**: NCCL analog. AllReduce, AllGather, ReduceScatter. Intra-node over MetaXLink; inter-node over Ethernet/RoCE. Proven at 10,000+ GPU cluster scale.

### Assembler / ISA

- **MXMACA ISA**: Proprietary SIMT ISA. Warp-based execution with hardware divergence. FP32/BF16/FP16/INT8 on C500 generation; adds FP8/INT4 on C600. **Not publicly documented** (no PTX equivalent). All development targets MACA C++ via MXMACA compiler.

---

## Hardware Architecture

### Compute (MXC500 / Xiyun C500 — GA 2024)

- **Process**: 7nm (TSMC)
- **Architecture**: Full SIMT GPGPU with warp scheduler and hardware divergence
- **FP32**: ~15 TFLOPS (~77% of NVIDIA A100)
- **INT8**: ~160 TOPS / **FP16**: ~80 TFLOPS
- **Memory**: HBM2E, ~64 GB (from benchmarks)
- **Interconnect**: MetaXLink (NVLink analog), 8 GPUs/server
- **Host**: PCIe (gen not officially disclosed)
- **Compatible**: CUDA-compatible via MXMACA 2.0

### Compute (MXN100 / Xisi N100 — Inference + Video)

- **INT8**: 160 TOPS | **FP16**: 80 TFLOPS
- **Memory**: HBM2E
- **Video**: 128-ch encode / 96-ch decode (HEVC, H.264, AV1, AVS2, 8K)

### Compute (MXC600 / Xiyun C600 — Announced 2025, Hopper-class; shipping at volume in 2026)

| Metric | Value |
|--------|-------|
| FP8 | 1,000 TFLOPS (native on-chip FP8 Tensor instructions) |
| Precision | FP32/BF16/FP16/FP8/INT8/INT4 |
| Memory | HBM3e, 144 GB |
| Memory BW | 3.6 TB/s |
| FP8 efficiency | ~2.5 TFLOPS/W |
| Supply chain | Fully domestic (design + mfg + pkg + test) |
| Target | Surpasses NVIDIA H20 (claimed) |
| Status | Small-batch Q4 2025 → **large-scale shipment during 2026** (company statement, CPO Sun Guoliang, 2026-07-08; some orders booked into 2027) |

### Memory Hierarchy

```
Per-CU: local registers + shared memory (capacity undisclosed)
         ↓
On-chip: L2 cache / scratchpad (capacity undisclosed)
         ↓
Off-chip: HBM2E (C500/MXN100) / HBM3e 144 GB @ 3.6 TB/s (C600)
```

### Interconnect

| Layer | Detail |
|-------|--------|
| Scale-up | MetaXLink (MX) — NVLink analog; X300 series adds "MLoE" multi-card high-speed interconnect |
| Cards/server | 8 (OAM server with C550); up to 16 (N300 server, PCIe-Switch full interconnect) |
| Supernode — C550 3D Mesh | Up to 64 GPUs (8 servers × 8 GPUs, 3D Mesh, electrical); still a current listed product |
| Supernode — C500X Optical | C500X Optical Interconnect Supernode (hybrid optical/electrical; secondary reporting says DragonFly, 16→64 GPUs across up to 8 machines — medium confidence, no vendor spec sheet) |
| Supernode — S600 (new, 2026) | 曦景 S600: **64 GPU cards in a single cabinet**, in-cabinet full interconnect, "zero-cable direct-connect" (0线缆直连) between compute and switch nodes; EP/TP parallelism; cabinet-to-cabinet expansion to 10,000+ card clusters. Per-link BW, per-GPU SKU, total FLOPS **not disclosed**. Announced only |
| Scale-out | Ethernet / RoCE via host NIC + MXCCL |
| Flagship cluster | C550 Shanghai Cube: 128 cards in 47U liquid-cooled cabinet |
| Host interface | PCIe (gen not officially disclosed) |

---

## Roadmap

| Product | Architecture | Process | Status | Key Specs |
|---------|-------------|---------|--------|-----------|
| MXN100 (Xisi N100) | GPGPU inference | 7nm | GA 2023 | 160 TOPS INT8; HBM2E; 128-ch encode |
| MXC500 (Xiyun C500) | GPGPU training | 7nm (TSMC) | GA 2024 | 15 TFLOPS FP32; HBM2E ~64 GB; 10K cluster |
| MXC550 (Xiyun C550) | GPGPU training | 7nm | Production 2025 | OAM; MetaXLink; Shanghai Cube 128 cards/47U |
| MXC588 (Xiyun C588) | GPGPU training | Not disclosed | Launched 2025-09-23 (with C600) | Specs not disclosed; C588 Server listed on MetaX product page |
| MXC600 (Xiyun C600) | Hopper-class GPGPU | Domestic (undisclosed) | Q4 2025 small-batch → **large-scale shipment 2026** (company statement) | FP8 1,000 TFLOPS; HBM3e 144 GB 3.6 TB/s; fully domestic |
| MXC700 (Xiyun C700) | Next-gen GPGPU | Not disclosed | **Company-disclosed (2026-04-08): core design + functional verification largely complete, in deeper performance optimization**; project started Apr 2025; no specs, no announced launch date | H100-parity target and "mass production late 2027" appear only in aggregator write-ups — **unverified** |
| MXN260 (Xisi N260) | GPGPU inference | Not disclosed | Listed product (server + workstation SKUs) | Specs not disclosed |
| MXN300 (Xisi N300) | GPGPU inference (high-density) | "Domestic advanced process" (node not disclosed) | Listed product; announcement date not established | FHFL dual-slot PCIe; max 500 W board power; MetaXLink across 4 cards; up to 16 GPUs/chassis in N300 Server (PCIe-Switch full interconnect). Memory capacity and TOPS **not disclosed** |
| MXX206 (Xisuo X206) | AI4S / scientific GPGPU | Not disclosed | Launched 2026-01-27 (智算申城 forum, Shanghai) | First X-series product; specs not disclosed |
| MXX300 series — X301 / X302 (Xisuo) | AI4S / scientific GPGPU, "full-precision mixed compute" | Not disclosed | **Announced WAIC 2026 (Jul 17–20, 2026); announced only** | Self-developed GPGPU on fully domestic supply chain; large-capacity high-bandwidth memory; MetaXLink + MLoE. FP64 rate, memory type/capacity/BW, TDP, node **not disclosed** |
| MXS600 (Xijing S600) | AI supernode (system, not a chip) | — | **Announced WAIC 2026; announced only** | 64 GPUs in one cabinet; in-cabinet full interconnect; zero-cable direct-connect; EP/TP; scales to 10,000+ cards |
| MXG (Xicai) G100 | Cloud Gaming GPU | 7nm | Announced | Cloud graphics; 2025 target |

---

## Business Context

| Event | Date | Detail |
|-------|------|--------|
| Founded | Sep 14, 2020 | Shanghai; Chen Weiliang (14yr AMD), Peng Li, Yang Jian |
| MXN100 mass production | 2023 | First MetaX GPU; AI inference + 8K video |
| MXC500 GA | 2024 | 7nm training GPU; 10,000+ GPUs deployed commercially |
| MXMACA 2.0 launch | 2023–2024 | Full CUDA-compatible stack; vLLM, PyTorch, TF |
| 9 clusters, 10K GPUs | End-2024 | Largest MetaX deployment across China (EDWC) |
| STAR Market IPO approval | Oct 24, 2025 | Shanghai Stock Exchange STAR Market |
| MXC600 announcement | Jul 2025 | FP8, HBM3e, fully domestic supply chain |
| MXC588 + MXC600 launch event (曦果发布会) | Sep 23, 2025 | C588 launched alongside C600 |
| STAR Market IPO debut | Dec 17, 2025 | +692%; RMB 4.2B raised; market cap >RMB 332B (~USD 47B) |
| Founder net worth | Dec 2025 | Chen Weiliang ~USD 6.5B; surpassed Moore Threads in retail investor interest |
| 2025 shipment (audited) | End-2025 | 33,649 training/inference GPU boards shipped in 2025 (+147.31% YoY); MetaX states cumulative GPU sales >55,000 units across 10+ intelligent-computing clusters |
| 曦索 X206 launch | Jan 27, 2026 | First X-series (AI4S) GPU; 智算申城 forum, Shanghai |
| FY2025 results | Apr 8, 2026 | Revenue RMB 1.644 B (+121.26% YoY); net loss RMB 789 M (narrowed 43.97% from RMB 1.409 B). Chairman Chen Weiliang discloses C700 status on the call |
| Q1 2026 results | Q1 2026 | Revenue RMB 562 M (+75.37% YoY); net loss RMB 98.84 M (vs RMB 233 M a year earlier); company targets break-even in 2026 |
| MXC600 large-scale shipment | Jul 8, 2026 | CPO/SVP Sun Guoliang (孙国梁) states C600 series — MetaX's core 2026 product — reached 大规模出货; some orders booked into 2027 (executive statement, not an audited figure) |
| WAIC 2026 launches | Jul 17–20, 2026 | 曦景 S600 supernode + 曦索 X300 series (X301/X302) announced |
| MiniMax H3 Day-0 adaptation | Aug 3, 2026 | 曦云 C-series completed Day-0 adaptation of MiniMax's open-sourced multimodal model H3 via MXMACA (MetaX newsroom) |

---

## Update — WAIC 2026, C600 Volume, and the X/S Series (2026-08-08)

*Updated 2026-08-08. Primary sources: MetaX newsroom (WAIC 2026 item; MiniMax H3 item 2026-08-03), MetaX product catalog pages, GitHub REST API for `metax-maca` and `MetaX-MACA/vLLM-metax`. Secondary: ITHome, Sina Finance (2026-07-08 and 2026-07-20), Tencent News (2026-04-08 FY2025 results call), TrendForce, PEDaily, EastMoney. Prior-generation content above is preserved; this section records what changed and what was previously missing.*

### 1. 曦景 (Xijing) S600 — new supernode line

Announced at **WAIC 2026, Shanghai, July 17–20, 2026**. The S-series is a system-level product line (a cabinet, not a chip). Vendor-stated attributes:

- **64 GPU cards in a single cabinet** (单机柜64卡高密度部署)
- **In-cabinet full-interconnect** architecture to cut data-exchange latency
- **"Zero-cable direct-connect" (0线缆直连)** between compute nodes and switch nodes — the differentiator MetaX emphasises
- Supports **EP (expert-parallel), TP (tensor-parallel)** and other multi-dimensional parallel strategies, across both training and inference
- Cabinet-to-cabinet expansion to **万卡级 (10,000+ card)** clusters

**Not disclosed**: per-link bandwidth, which GPU SKU populates the cabinet, total supernode FLOPS, memory per cabinet, power/cooling envelope.

**Status: announced only.** No sampling, shipping, pricing, customer, or benchmark evidence was found.

> **Correction to a common misreading.** The S600 does **not** supersede or restate MetaX's existing **C550 3D Mesh Supernode** (up to 64 cards across up to 8 servers, full-cabinet, electrical/memory-semantic interconnect). The C550 3D Mesh Supernode remains a listed product. S600 is an *additional, higher-integration* supernode line alongside it — and alongside the **C500 Shanghai Cube** (liquid-cooled full cabinet, 128 C550 cards / 47U) and the **C500X Optical Interconnect Supernode**. All four coexist in MetaX's supernode catalog.

### 2. 曦索 (Xisuo) X-series — AI4S GPUs (X206 is a pre-baseline gap; X300 is the new news)

The X-series is MetaX's "scientific intelligence" (AI4S) line. **It is not new as of WAIC 2026** — the repo simply had never recorded it:

| Product | Launch | Notes |
|---|---|---|
| 曦索 X206 | **2026-01-27**, 智算申城 forum, Shanghai | First X-series product. Pre-dates this survey's 2026-04-05 baseline — a prior gap in this repo, not July 2026 news. X206 Server also listed. |
| 曦索 X300 series (**X301**, **X302**) | **WAIC 2026 (Jul 17–20, 2026)** | Second generation. X302 Server also listed. |

Vendor-stated X300 attributes: fully self-developed GPGPU architecture on a **fully domestic supply chain**; **"full-precision mixed compute" (全精度混合算力)** — i.e. FP64-class numerics alongside low-precision AI datatypes; **large-capacity high-bandwidth memory**; **MetaXLink plus "MLoE"** high-speed multi-card interconnect; MXMACA software stack.

Target workloads (vendor-stated): numerical weather prediction, oceanography, computational fluid dynamics, molecular dynamics, materials science, life sciences.

**Not disclosed**: FP64 rate, any FLOPS/TOPS figure, memory type/capacity/bandwidth, TDP, process node, card form factor.

**Status: announced only.**

### 3. MXC600 — from small-batch to volume

On **2026-07-08**, Chief Product Officer / SVP **孙国梁 (Sun Guoliang)** stated that the MXC600 series — MetaX's core product for 2026 — has achieved **大规模出货 (large-scale shipment)**, that some orders are already booked into 2027 and beyond, and that general-purpose stable domestic compute is supply-constrained. This supersedes the repo's "Q4 2025 small-batch".

Caveat: this is an **executive statement reported by financial press, not an audited shipment disclosure**. The audited figure that does exist — **33,649 training/inference GPU boards shipped in 2025 (+147.31% YoY)**, with MetaX stating cumulative GPU sales exceeding **55,000 units across 10+ intelligent-computing clusters** by end-2025 — predates C600 volume and does not corroborate C600 specifically.

Derived hardware in the C600 generation: 曦云 **C600** cards and the 曦思 **N300**.

### 4. MXC700 — upgraded from "rumored" to company-disclosed

The prior "MXC700 (rumored)" line is now under-stated. On MetaX's **2026-04-08 FY2025 results call**, chairman **陈维良 (Chen Weiliang)** stated that 曦云 **C700 core chip design and functional verification are largely complete** and that the part is undergoing deeper performance optimization; the project was initiated in **April 2025** and is positioned to extend beyond C600's 信创 (state-sector) base into internet-sector customers.

That is a company disclosure and outranks "rumor". However: **C700 remains unannounced as a product, with no disclosed specifications and no announced launch date.** The "H100-parity" target and a "mass production late 2027" timeline appear only in aggregator write-ups and remain **unverified**.

### 5. Product catalog is far wider than previously recorded

MetaX's own product pages now list:

| Series | Products |
|---|---|
| C (曦云 Xiyun) | C500, C550, **C500X**, **C588**, C600 |
| N (曦思 Xisi) | N100, N260, **N300** |
| X (曦索 Xisuo) | **X206**, **X301**, **X302** |
| G (曦彩 Xicai) | G100 |
| Servers | N260, N300, C500, C550, C588, C600, X206, X302 (+ an N260 workstation) |
| Supernodes | C500 Shanghai Cube (liquid-cooled full cabinet), C500X Optical Interconnect Supernode, C550 3D Mesh Supernode (+ the new S600) |

Two of these are **pre-baseline gaps, not 2026 Q2–Q3 news**: **C588** launched alongside C600 at MetaX's 曦果发布会 on **2025-09-23**, and **C500X** optical-interconnect material traces to **WAIC 2025**. Secondary reporting describes C500X as hybrid optical/electrical with a **DragonFly** topology scaling 16→64 GPUs (up to 8 machines) — **medium confidence pending a vendor spec sheet**.

**曦思 N300** (from MetaX's official product page): FHFL dual-slot PCIe card; **maximum board power 500 W**; proprietary GPGPU architecture on a "domestic advanced process"; **MetaXLink high-speed interconnect across 4 cards**; MXMACA stack. The official **N300 Server** page confirms **up to 16 N300 GPUs per chassis** with a **PCIe-Switch full-interconnect** topology (a "Common" topology and 8-GPU configurations are also offered).
⚠️ The widely-circulated **"48 GB memory"** and **"14nm"** figures for N300 appear only in low-quality aggregator listings, are **not confirmed by MetaX**, and are **not** recorded here as specs. INT8/FP8 TOPS not disclosed. N300 announcement date not established.

### 6. Financials

| Period | Revenue | Net result |
|---|---|---|
| FY2025 | RMB **1.644 B** (+121.26% YoY) | Net loss RMB **789 M**, narrowed 43.97% from RMB 1.409 B |
| Q1 2026 | RMB **562 M** (+75.37% YoY) | Net loss RMB **98.84 M** (vs RMB 233 M a year earlier) |

The company targets break-even in 2026.

### 7. Software — MXMACA cadence and the open-source build-out

**MXMACA** was open-sourced on **2025-02-14**. The repo's "MXMACA 2.0" line is stale.

The **3.3.0.X release (December 2025)** is reported to reach **PyTorch 2.8** adaptation, cover **2,650 core operators** (of which **2,410 GPU operators**), claim **~92.94% seamless CUDA-project migration**, and add **native PaddlePaddle (飞桨)** support.
⚠️ **Medium confidence**: these specifics rest on Baidu Baike / CSDN-tier sources, not a vendor changelog or release note.

Community and ecosystem figures are **vendor claims**: **300,000+** registered open-source community developers as of March 2026; **500,000+** developer users claimed in MetaX's own WAIC 2026 newsroom item; native support for **40+ AI frameworks**, **500+ models**, and **30 Day-0 model adaptations** since December 2025.

**vLLM-metax release chain** (confirmed exactly from the GitHub releases API — the repo's "v0.13.0" was badly stale):

| Release | Date | Notes |
|---|---|---|
| v0.13.0 | 2026-02-06 | (prior repo baseline) |
| v0.14.0 | 2026-03-23 | |
| v0.15.0 | 2026-03-26 | |
| v0.17.0 | 2026-04-15 | adds vLLM 0.16 / 0.17 |
| v0.18.0 | 2026-05-06 | |
| v0.19.0 | 2026-05-13 | full Gemma 4 support |
| v0.20.0 | 2026-06-03 | DeepGEMM scheduler-metadata buffer; fused MoE grouped topk |
| v0.21.0 | 2026-07-10 | DeepSeek V4 rebase |
| v0.22.0 | 2026-07-27 | latest |

**New `github.com/metax-maca` components** (creation dates from the GitHub API):

| Repo | Created | Role |
|---|---|---|
| hpc_warp | 2026-01-13 | HPC primitives |
| McFlashInfer | 2026-03-09 | FlashInfer port (attention kernels) |
| mcTriton | 2026-03-13 | **Triton port for MACA** |
| go-mxsml | 2026-03-16 | Go bindings for MXSML management library |
| pymxsml | 2026-03-19 | Python bindings for MXSML |
| sglang (fork) | 2026-04-02 | SGLang serving engine on MACA |
| TileOPs-Metax | 2026-04-24 | TileLang-based high-performance LLM operator library |
| TileKernels-Metax | 2026-04-24 | TileLang-based kernel library |
| vllm-omni-metax | 2026-04-24 | Omni-modal vLLM variant |
| AIModels | 2026-05-18 | Model zoo |
| op_optimization | 2026-05-26 | Operator optimization work |
| mccl_tests | 2026-05-28 | MXCCL collective benchmarks (nccl-tests analog) |
| MXDeepEP | 2026-06-10 | **DeepEP expert-parallel communication library** |
| mcFlashMLA | 2026-07-01 | FlashMLA port (DeepSeek MLA attention) |
| FluidDynamics | 2026-07-13 | AI4S — CFD |
| LifeScience | 2026-07-13 | AI4S — life sciences |
| MedicalImage | 2026-07-27 | AI4S — medical imaging |
| AI4S-Framework | 2026-08-07 | **AI4S umbrella framework** — mirrors the X300 launch |

The AI4S repository cluster (FluidDynamics, LifeScience, MedicalImage, AI4S-Framework) appearing between July and August 2026 is a direct software-side mirror of the X300 hardware launch.

### 8. Ecosystem

**MiniMax H3 Day-0 adaptation — confirmed.** Per MetaX's own newsroom (**2026-08-03**), 曦云 C-series GPUs completed Day-0 adaptation of MiniMax's newly open-sourced multimodal model **H3** via MXMACA. Moore Threads announced Day-0 adaptation of the same model on the same day with the MTT S5000.

---

## Resources

### Official
- [MetaX Official Website](https://www.metax-tech.com/en)
- [MetaX About Page](https://www.metax-tech.com/en/about/about.html)
- [MXMACA Platform](https://www.metax-tech.com/en/goods/platform.html?cid=4)
- [MetaX G-Series (MXG)](https://www.metax-tech.com/en/goods/prod.html?cid=3)
- [MetaX Developer Portal](https://developer.metax-tech.com/)
- [MetaX-MACA GitHub Org](https://github.com/metax-maca)
- [vLLM-MetaX GitHub](https://github.com/MetaX-MACA/vLLM-metax)
- [mcpy GitHub](https://github.com/MetaX-MACA/mcpy)
- [MetaX Newsroom — WAIC 2026 item (S600 + X300)](https://www.metax-tech.com/en/ndetail/12629.html)
- [MetaX Newsroom — MiniMax H3 Day-0 adaptation (2026-08-03)](https://www.metax-tech.com/ndetail/12632.html)
- [MetaX Newsroom index](https://www.metax-tech.com/en/news.html?cid=15)
- [MetaX N300 product page](https://www.metax-tech.com/en/goods/prod.html?cid=106&id=66)
- [MetaX N300 Server product page](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=69)
- [vLLM-metax releases API](https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases)
- [metax-maca org repos API](https://api.github.com/orgs/metax-maca/repos)

### Hardware
- [Tom's Hardware — MetaX Xisi N100 debut](https://www.tomshardware.com/news/metax-chinese-gpu-developer-unveils-first-product)
- [WCCFTech — MetaX N100 160 TOPS](https://wccftech.com/chinese-chipmaker-metax-unveils-first-gpu-targeted-towards-ai-features-160-tops-of-compute/)
- [tphuang X — MXC500 15 TFLOPS](https://x.com/tphuang/status/1700511558961340656)
- [IT之家 — C600 首款全国产 GPU](https://www.ithome.com/0/890/942.htm)
- [Red Hot Cyber — C600](https://www.redhotcyber.com/en/post/made-in-china-muxi-presents-the-xiyun-c600-general-purpose-gpu/)
- [MetaX C550 Server](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=44)
- [MetaX C550 3D Mesh Supernode](https://www.metax-tech.com/en/goods/prod.html?cid=112&id=55)
- [ITHome — WAIC 2026 S600 / X300 coverage](https://www.ithome.com/0/978/460.htm)
- [MetaX product catalog (all series)](https://www.metax-tech.com/en/goods/prod.html?cid=3)
- [ZOL — MetaX product listing](https://ai.zol.com.cn/1212/12124334.html)
- [MXC500 benchmark blog](https://wangjunjian.com/mxc500/benchmark/2025/02/13/Performance-Stress-Testing-of-the-MuXin-MXC500-for-Large-Model-Inference.html)

### Software
- [vLLM RFC #23157 — maca backend](https://github.com/vllm-project/vllm/issues/23157)
- [HAMi MetaX support docs](https://github.com/Project-HAMi/HAMi/blob/master/docs/metax-support.md)
- [Rise VAST MetaX certification](https://www.theriseunion.com/blog/metax-compatibility.html)
- [Baidu Baike — MXMACA 软件栈 (MXMACA 3.3.0.X details; medium confidence)](https://baike.baidu.com/item/MXMACA%E8%BD%AF%E4%BB%B6%E6%A0%88)

### Business / IPO
- [Global Times — MetaX IPO +692%](https://www.globaltimes.cn/page/202512/1350816.shtml)
- [China Biz Insider — MetaX STAR Market IPO](https://chinabizinsider.com/metax-wins-nod-for-star-market-ipo-escalating-race-for-chinas-first-gpu-listing/)
- [SCMP — MetaX AMD veterans profile](https://www.scmp.com/tech/big-tech/article/3330511/meet-metax-chinese-ai-chip-hopeful-challenging-nvidias-dominance)
- [Bloomberg — Chen Weiliang billionaire](https://www.bloomberg.com/news/articles/2025-12-17/ex-amd-engineer-becomes-billionaire-after-metax-s-surging-china-chip-ipo)
- [Tencent News — 2026-04-08 FY2025 results call (C700 status, Chen Weiliang)](https://news.qq.com/rain/a/20260408A066PB00)
- [Sina Finance — 2026-07-08 Sun Guoliang on C600 large-scale shipment](https://finance.sina.com.cn/wm/2026-07-08/doc-inihceis8322692.shtml)
- [Sina Finance — 2026-07-20 WAIC 2026 coverage](https://finance.sina.com.cn/tech/shenji/2026-07-20/doc-iniimuyk4915570.shtml)
- [EastMoney — FY2025 / Q1 2026 financials](https://finance.eastmoney.com/a/202604293724854763)
- [TrendForce CN — 2026-07-09 MetaX shipment note](https://www.trendforce.cn/industry-news/semiconductors/20260709-6285.html)
- [PEDaily — 2026-07-28 MetaX feature](https://news.pedaily.cn/20260728/135041.shtml)
- [TrendForce — Chinese GPU IPO race](https://www.trendforce.com/news/2025/12/19/news-chinese-gpu-makers-pick-up-pace-in-going-public)
- [IndexBox — MetaX IPO debut](https://www.indexbox.io/blog/chinese-ai-chip-designer-metax-begins-trading-on-shanghai-star-market/)
