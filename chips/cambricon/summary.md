# Cambricon MLU Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: cambricon*
*device_class: Neural Processor*

---

## Overview

Cambricon Technologies (寒武纪) is China's leading independent AI chip company, founded in 2016 by researchers from the Institute of Computing Technology, Chinese Academy of Sciences. Their **Machine Learning Unit (MLU)** family of neural processors spans edge inference (MLU220), server inference (MLU270), and datacenter training/inference (MLU290, MLU370, MLU590). The **only vendor-documented production datacenter chip remains the Siyuan 370 (MLU370)** — an MLUarch03 processor using LPDDR5 memory, marketed on a dual-chip **MLU370-X8** card with 48 GB memory and 614.4 GB/s bandwidth. Cambricon's own 2026 half-year report (filed 2026-08-08) names only 思元100/220/270/290/370 in its product and core-technology sections; Siyuan 590 and Siyuan 690 appear nowhere in Cambricon primary sources and are documented here only as third-party reports (see the 2026 H1 update section below).

The MLU architecture is built around a hierarchy of **MLU Cores** (also called IPU Cores), grouped into **Clusters** of four. Each core has private on-chip scratchpad memories (NRAM for activation data, WRAM for weights) that the programmer manages explicitly — there is no hardware data cache in the primary compute path. A **Memory Core** DMA engine per cluster handles asynchronous data movement between off-chip GDRAM and the on-chip scratchpads.

Cambricon's software ecosystem is called **NeuWare** (packaged as **CNToolkit**). It maps closely to NVIDIA CUDA: the **BANG C** kernel language (compiled by **CNCC**) targets the **MLISA** machine ISA; the **CNNL** library provides cuDNN/cuBLAS-equivalent operators; **CNRT** is the runtime API (Queue-based, analogous to CUDA Streams); and **CNCL** handles collective communication over **MLU-Link** (Cambricon's chip-to-chip interconnect). PyTorch users access the hardware through the official `torch_mlu` extension; LLM inference is served via `vllm-mlu`.

Cambricon's strategic position in 2025–2026 is as China's primary domestic alternative to NVIDIA in the datacenter AI market, benefiting from Chinese AI customers who cannot access H100/H200/B200 due to US export controls.

---

## Software Stack

### Framework Integration

Cambricon supports the major frameworks through native integrations:

- **torch_mlu**: The official PyTorch backend for MLU. Registers the MLU as a PyTorch device (`torch.device("mlu")`), dispatches ATen operations to CNNL and BANGC OPS, and exposes MLU-aware distributed training through CNCL. Versioned as `{torch_mlu_ver}+torch{pytorch_ver}`. The earlier **CATCH** repository was the predecessor integration.
- **vllm-mlu**: Cambricon's fork of vLLM for LLM inference serving on MLU hardware. Supports Chunk Prefill, Prefix Caching, Speculative Decoding, Graph Mode and Sleep Mode. Requires SDK 25.08 and MLU370+ hardware (`MLU370以上的设备`, still the stated requirement as of the 2026-08 README). The README changelog carries a single entry — `[2026.04.24] vllm_mlu day0支持DeepSeek-V4` — which is the day-0 DeepSeek-V4 enablement described in the 2026 H1 update section below. An upstream vLLM PR (#10315) integrates MLU as a first-class backend; a later documentation PR (#25942, "[Doc] Add Cambricon MLU support") merged upstream 2025-09-30.
- **PaddlePaddle**: Two integration paths — `PaddleCustomDevice` loads `libpaddle-custom-mlu.so` with 264+ custom operators; PaddleX provides higher-level pipeline support.

### Compiler / IR

- **BANG C + CNCC**: BANG C is the primary kernel programming language — a C/C++ dialect extended with `__mlu_global__`, `__nram__`, `__wram__`, `__mlu_shared__` memory qualifiers and compute intrinsics. CNCC (Cambricon Neuware C/C++ Compiler) compiles `.mlu` source to MLISA assembly; CNAS (Cambricon Neuware Assembler) produces `.cnbin`/`.cnfatbin` binaries. The full pipeline mirrors `nvcc → PTX → SASS`.
- **MagicMind**: MLIR-based inference compiler accepting TF/PyTorch/ONNX/Caffe models, performing graph fusion, quantization (INT8/FP16), and layout optimization to produce compiled MLU binaries. Analogous to TensorRT.
- **triton-linalg**: Cambricon's fork converting Triton MLIR dialect to Linalg for MLU backend lowering. Frontend conversion complete. The public GitHub repository has had **no pushes since 2025-02-07**, so backend compilation to MLISA is best described as stalled in public or moved into the licensed SDK — not as visibly in flight. Cambricon's own 2026 half-year report states it "持续跟进 Triton 社区版本更新" (continues to track Triton community releases), so internal Triton work is ongoing even though the public mirror is quiet.

### Op Library

- **CNNL** (Cambricon Neuware Neural Network Library): The primary high-level operator library (cuDNN + cuBLAS analog). Covers GEMM, Conv2D, BatchNorm, LayerNorm, Attention, Pooling, RNN, elementwise ops via a C API (`cnnlConvolutionForward`, `cnnlMatMul`, etc.). Used internally by torch_mlu for ATen dispatch.
- **CNNL_Extra**: Extended operator library for ML ops beyond core CNNL.
- **Torch-MLU-Ops** (`torch_mlu_ops`): Cambricon's self-developed high-performance **fused** operator library for PyTorch, sitting alongside CNNL and mlu-ops. It is the layer through which vLLM, TGI and Stable Diffusion WebUI reach fused MLU kernels, and the layer through which the DeepSeek-V4 Compressor and mHC modules were accelerated. v1.3.2 was released 2026-02-04 against Torch-MLU v1.24.1 and CNNL v1.28.3. Distributed with the licensed SDK bundle; no Cambricon-owned public source repository was found.
- **CNML** (legacy): Earlier graph-based API (v7.10.2, April 2021); superseded by CNNL in current CNToolkit.

### Kernel Library

- **mlu-ops / BANGC OPS**: Open-source BANG C kernel implementations for hundreds of ML operators. The closest analog to CUTLASS/CUB. Used by torch_mlu for ops not in core CNNL.
- **BANGPy**: Python-level kernel programming API (Triton-in-spirit) for custom MLU kernel authoring without raw BANG C.
- **CNCV**: Accelerated computer vision primitives (decode, resize, color convert, normalize) for inference preprocessing pipelines.

### Runtime

- **CNRT** (`libcnrt.so`): The primary runtime API — the CUDA Runtime analog. Provides device management, `cnrtMalloc`/`cnrtFree`, **Queue** creation/sync (CUDA Stream analog), **Notifier** events (CUDA Event analog), and `cnrtInvokeKernel` for kernel dispatch with `cnrtDim3_t` task sizing.
- **CNDrv** (`libcndrv.so`): Low-level driver API (CUDA Driver API analog). Explicit context management, module loading from `.cnbin`, fine-grained memory control.
- **CNToolkit**: The SDK bundle packaging CNCC, CNAS, CNRT, CNDrv, CNNL, CNCL as a versioned distribution. **Current requirement (2026-08): CNToolkit ≥ v4.1.0, CNNL ≥ v1.28.0, MLU driver ≥ v6.0.3**, per the actively maintained `mlu-ops` master README. The previously recorded "CNToolkit v3.7.2 / SDK 1.15.0" was years stale (that manual is dated 2023-10-18); an intermediate CNToolkit 3.8.4 doc tree also exists. The SDK **bundle** number corresponding to CNToolkit 4.1.0 is **not disclosed** publicly (the release-note tree is behind authentication).

### Driver / Firmware

- **cambricon-mlu kernel driver**: Linux kernel module for PCIe BAR mapping, IOCTL dispatch, DMA control, and interrupt handling. Open-sourced on Gitee. No equivalent to NVIDIA's GSP (on-GPU firmware RISC-V); the Cambricon driver uses a host-managed model.
- **cambricon-k8s-device-plugin**: Kubernetes DaemonSet for automatic MLU resource reporting and health monitoring.
- **mlu-exporter**: Prometheus metrics exporter for MLU device observability.

### Communication

- **CNCL** (Cambricon Collective Communication Library): NCCL analog. AllReduce, AllGather, ReduceScatter, Broadcast over MLU-Link for intra-node communication. Exposed as the `torch.distributed` backend via torch_mlu.
- **Gloo**: Used in heterogeneous multi-vendor clusters (KAITIAN framework) for cross-vendor communication where CNCL cannot be used.

### Assembler / ISA

- **MLISA**: The native MLU ISA. All-64-bit instructions in four categories: Scalar (control/addressing), Vector (element-wise ops on NRAM), Matrix (GEMM/conv using NRAM+WRAM), Control (branch/barrier/DMA sync). 64 × 32-bit GPRs per core. No vector register file — operands reside in scratchpad memory.
- **CNAS** (Cambricon Neuware Assembler): Assembles `.mlisa` text to `.cncode`/`.cnbin`/`.cnfatbin` binary formats.
- **Cambricon ISA** (ISCA 2016): The original ISA paper defining 43 instruction types. Load-store architecture with higher code density than MIPS/x86/GPGPU for NN workloads.

---

## Hardware Architecture

### Compute Engine

The MLU is organized as a hierarchy of compute units:

- **MLU Core (IPU Core)**: The atomic compute unit. Contains a Functional Unit (FU) for scalar/vector/matrix execution, 64 × 32-bit GPRs, private NRAM scratchpad (activation data), and private WRAM scratchpad (weight data). Executes BANG C kernel code independently.
- **Cluster**: Four MLU Cores + a Memory Core DMA engine + a per-cluster Shared SRAM. The DMA engine runs independently, enabling overlap of data movement with computation. `__sync_cluster()` barriers coordinate all 4 cores within a cluster.

Historical core counts: MLU220 (1 cluster, 4 cores), MLU270 (4 clusters, 16 cores), MLU290 (8 clusters, 32 cores). MLU370+ cluster counts are not fully disclosed.

Supported precisions (MLUarch03): FP32, FP16, BF16, INT16, INT8, INT4.

Peak performance: MLU290 — 512 INT8 TOPS, 64 FP32 TFLOPS; MLU370 — ~256 INT8 TOPS (single chip estimate).

### Data Path

The MLU uses a **SPMD (Single Program Multiple Data)** execution model. All MLU Cores within a task execute the same kernel code in parallel over their private NRAM partitions. Unlike GPU SIMT, there is no warp concept; each Core has an independent instruction stream. The programmer writes explicit DMA calls to move data between GDRAM and on-chip scratchpads, and uses barrier instructions for inter-core synchronization.

### On-chip Memory

- **NRAM**: Per-core private scratchpad for vector/tensor operand data. Programmer-managed; no hardware prefetch.
- **WRAM**: Per-core private scratchpad for convolution kernel weights. Feeds directly into the matrix FU.
- **Shared SRAM**: Per-cluster SRAM shared by all 4 cores + Memory Core. Staging buffer and inter-core data sharing.
- **LLC**: Small read-only cache on the GDRAM path; buffers shared weight data broadcast to multiple cores.

### Off-chip Memory

| Product | Memory | Capacity | Bandwidth |
|---------|--------|----------|-----------|
| MLU290-M5 | HBM2 | 32 GB | 1,228 GB/s |
| MLU370-X8 (dual-chip) | LPDDR5 | 48 GB | 614.4 GB/s |
| MLU590 | HBM (gen TBD) | TBD | TBD |
| MLU690 *(reported, unverified)* | HBM3 *(reported)* | 196 GB *(reported)* | ~3.35 TB/s *(reported)* |

The MLU370 generation uses LPDDR5 (not HBM) due to HBM supply constraints from US export controls on China. MLU590 targets HBM but faces production constraints; no capacity or bandwidth figure has been disclosed for it.

The MLU690 row carries **unverified third-party figures only** — Cambricon has published no datasheet, its 2026 half-year report contains zero mentions of 思元690 or HBM, and its product pages were unreachable during the 2026-08-08 scan. Do not treat these numbers as confirmed.

### Host Interface / Package

- **PCIe Gen4 x16**: Standard host interface (~64 GB/s bidir) for MLU370-M8, MLU370-X, and MLU290.
- **Form factor**: Full-height, full-length (FHFL) PCIe cards for server deployment.
- **Chiplet packaging** (research): Cambricon-LLM paper (arXiv 2409.15654) describes a future chiplet-based design for on-device 70B LLM inference.

### Scale-up Interconnect

- **MLU-Link**: Cambricon's proprietary chip-to-chip interconnect (NVLink analog). Used in the MLU370-X8 dual-die card to present two Siyuan 370 chips as a unified 48 GB device. Supports 8-card parallel training at the server level. No external switch ASIC (no NVSwitch equivalent) has been disclosed.

### Scale-out Interconnect

- Standard Ethernet or InfiniBand via host NICs.
- **CNCL** handles intra-MLU collectives (over MLU-Link within a node); Gloo handles cross-vendor heterogeneous scenarios.

---

## Programming Model Rationale

**1. Explicit scratchpad management is the performance contract.** Unlike GPU caches, MLU NRAM/WRAM are deterministic programmer-managed scratchpads. The tradeoff: no cache miss uncertainty, but the programmer (or CNNL/compiler) must schedule every data movement explicitly. This makes performance highly predictable for regular DNN patterns.

**2. The Memory Core DMA engine enables compute-DMA overlap.** Analogous to NVIDIA TMA (Tensor Memory Accelerator), the Memory Core runs independently of the 4 compute cores. The standard optimization is double-buffering: while 4 cores compute on tile N in NRAM, the Memory Core pre-fetches tile N+1 from GDRAM into WRAM/Shared SRAM.

**3. CNNL is the correct abstraction for >95% of workloads.** torch_mlu dispatches standard PyTorch ops to CNNL automatically. Custom BANG C kernels are necessary only for novel ops or research. This mirrors the cuDNN/custom CUDA kernel split in the NVIDIA ecosystem.

**4. MagicMind is the deployment path for production inference.** Similar to TensorRT, MagicMind performs offline graph-level optimization and produces a compiled engine. The one-time compilation cost is amortized across thousands of inference calls. torch_mlu with eager execution is appropriate for training and development.

**5. NeuWare's maturity is Cambricon's competitive moat.** Per industry analysis (Digitimes 2025), NeuWare is considered mature enough to break local developer dependence on NVIDIA CUDA for standard workloads. The framework integration breadth (PyTorch, PaddlePaddle, vLLM, MXNet) and open-sourced mlu-ops library signal an ecosystem deliberately modeled on CUDA's developer strategy.

---

## 2026 H1 Update (April–August 2026)

*Updated 2026-08-08. Primary sources: Cambricon 2026 半年度报告 (cninfo filing, 2026-08-08); Cambricon developer article on DeepSeek-V4 Day-0 adaptation (2026-04-24); `vllm-mlu` and `mlu-ops` master READMEs; GitHub API repository metadata (retrieved 2026-08-08). Third-party sources are labelled as such inline.*

### What did NOT change

The headline worth stating explicitly, because it contradicts a widely repeated secondary narrative: **Cambricon's product center of gravity has not moved off the MLU370.** Cambricon's own 2026 half-year report names only 思元100 / 思元220 / 思元270 / 思元290 / 思元370 in its product and core-technology sections and contains **zero** occurrences of 思元590, 思元690, or HBM. Cambricon's own DeepSeek-V4 Day-0 article lists **MLU370-S4 / X4 / X8** as the board options. `vllm-mlu` still states its hardware requirement as `MLU370以上的设备`. MLU370 remains the only vendor-documented datacenter part, and the MLU590 memory row stays TBD.

Also unchanged: no Cambricon submission appears in MLPerf Inference v6.0 (2026-04) or any prior MLPerf round; no `MLUarch04` or `MLUarch05` architecture name is confirmed anywhere; no MLU part has been announced as discontinued; and no Cambricon paper or talk was found at ISCA 2026 or Hot Chips 2026.

### Siyuan 690 (MLU690) — reported, LOW confidence, status contested

A part called **Siyuan 690 / MLU690** is widely described in Chinese media and brokerage commentary but has **no Cambricon primary source**. Status is genuinely contested: secondary coverage of the H1 report asserts mass production in early 2026, while other coverage in the same period describes the 690 as still "in final testing," and TrendForce in December 2025 placed it in the testing phase with volume production potentially slipping to H2 2026. Record it as *reported by secondary Chinese media as entering production in early 2026; not confirmed by any Cambricon primary source.*

Recurring but **unverified vendor-adjacent** figures across those sources:

| Attribute | Reported value | Note |
|---|---|---|
| Package | Dual-die / chiplet | Reported; no Cambricon confirmation |
| Peak FP16 | >700 TFLOPS | Brokerage report figure |
| Peak INT4 | >2,800 TOPS | Secondary blog figure |
| Memory | 196 GB HBM3 | Reported; HBM is unmentioned in the H1 filing |
| Memory bandwidth | ~3.35 TB/s | Secondary blog figure |
| Interconnect | **>890 Gbps (≈111 GB/s)** | Sources say 超890**Gbps** — gigabits, not gigabytes. A widely circulated "890 GB/s" restatement is wrong by ~8× |
| Process node | **not disclosed** | Sources contradict each other (TSMC 4nm vs 7nm). No node is published here |

### Shipment figures are third-party targets, and pre-date this window

The frequently cited "500,000 AI accelerators in 2026, including as many as 300,000 Siyuan 590 and 690" and the "~116,000–142,000 units in 2025" baseline originate from **Bloomberg supply-chain reporting dated 2025-12-04**, echoed later by Tom's Hardware, TrendForce and others. These are third-party supply-chain *targets/estimates* — not vendor guidance, not confirmed shipments, and not new in the April–August 2026 window. Named binding risks in that reporting: low yield, SMIC 7nm-class capacity contention with Huawei for the same wafer allocation, and HBM supply. Domestic HBM3 from CXMT is targeted for end-2026, after the stated ramp.

### H1 2026 financial results (primary filing, 2026-08-08)

| Metric | H1 2026 | YoY |
|---|---|---|
| Revenue | RMB 5.996 B (599,557.36 万元) | +108.13% |
| Net profit attributable to shareholders | RMB 2.311 B (231,091.21 万元) | +122.61% |
| Net profit ex-nonrecurring items | RMB 2.166 B | +137.30% |
| Inventory | RMB 8.247 B (824,750.95 万元) | **45.32% of total assets** |

This is Cambricon's first half-year above RMB 5 billion since listing. The inventory concentration is a **primary-source disclosure**, not an analyst estimate: the filing itself names inventory write-down as a risk factor.

On **2026-07-28** Cambricon announced a restricted-stock incentive plan with tiered revenue targets: **2026 ≥ RMB 13.5 B; 2026–2027 cumulative ≥ RMB 40.5 B; 2026–2028 cumulative ≥ RMB 100 B** (~US$14.8 B). This is the clearest public statement of Cambricon's own three-year volume ambition — roughly an 8× step up from the current run rate.

### Software: DeepSeek-V4 Day-0 enablement (2026-04-24)

The one unambiguously in-window software event. Cambricon shipped **Day-0 support for DeepSeek-V4** through `vllm-mlu`, covering both **DeepSeek-V4-flash (285B)** and **DeepSeek-V4-pro (1.6T)**. Named capabilities in Cambricon's own developer article:

- **Five-dimensional hybrid parallelism** — TP / PP / SP / DP / EP
- **Communication–computation overlap**
- **Low-precision quantization**
- **PD-separated deployment** (prefill/decode disaggregation; independently corroborated as 预填充与解码分离 in the H1 filing)
- **Torch-MLU-Ops** acceleration of the DeepSeek-V4 **Compressor** and **mHC** modules
- Hand-written **BANG C** kernels for sparse/compressed attention and **GroupGemm**

Two calibrations worth recording. First, **Torch-MLU-Ops is new to this survey, not new to the world** — v1.3.2 shipped 2026-02-04 and already listed vLLM, TGI and Stable Diffusion WebUI as consumers. Second, the H1 filing corroborates PD separation and DeepSeek Day-0 adaptation generically but **never names Torch-MLU-Ops or vllm-mlu**.

### Toolchain and open-source activity

- **CNToolkit ≥ v4.1.0 / CNNL ≥ v1.28.0 / driver ≥ v6.0.3** is the current stated requirement (`mlu-ops` master README). This supersedes the repo's previous CNToolkit v3.7.2 / SDK 1.15.0. The matching SDK bundle number is **not disclosed**.
- **Actively maintained public repos** (last push 2026-08-07): `mlu-ops`, `cambricon-k8s-device-plugin`, `mlu-exporter`. Also moving: `mmcv` (2026-07-27), `mlu-ops-proto` (2026-07-07), `vllm-mlu` (2026-05-11).
- **Quiet public repos**: `torch_mlu` (last push 2025-03-15), `triton-linalg` (last push 2025-02-07).
- `mlu-ops`' last GitHub *release* is **v1.8.1, published 2026-01-06**, with a note that future releases will not be published on that page — development continues on `master`.
- The newest PyTorch integration code is in the **licensed SDK bundle**, not in a public mirror: Gitee `cambricon/torch_mlu` sits on branch `r1.22_pt2.4.0` (PyTorch 2.4.0) with 2025-era end-of-life dates, while `torch_mlu_ops` v1.3.2 already references Torch-MLU v1.24.1.

---

## Resources

### Documentation
- [BANG C Developer Guide v2.15.0 (online)](https://www.cambricon.com/docs/bangc/developer_guide_html/)
- [BANG C Language Guide v2.4.1 (PDF)](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf)
- [CNRT User Guide v6.0.0](https://www.cambricon.com/docs/cnrt/user_guide_html/index.html)
- [CNToolkit Installation Guide v3.7.2](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
- [Cambricon Developer Portal](https://developer.cambricon.com/)
- [MLU370-M8 Product Manual (FCC)](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf)
- [DeepSeek-V4 Day-0 adaptation on MLU — Cambricon developer article (2026-04-24)](https://developer.cambricon.com/index/article/details.html?id=12)
- [CNToolkit 3.8.4 release-note tree (intermediate version; superseded by the v4.1.0+ requirement)](https://sdk.cambricon.com/static/independent/CNToolkit/3.8.4/releasenote/)

### Filings and Corporate
- [Cambricon 2026 半年度报告 (cninfo, 2026-08-08)](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF) — primary source for H1 2026 revenue/profit/inventory and the 思元100/220/270/290/370 product list

### Open-Source Repositories
- [torch_mlu](https://github.com/Cambricon/torch_mlu) — PyTorch MLU backend
- [mlu-ops](https://github.com/Cambricon/mlu-ops) — BANG C kernel library
- [vllm-mlu](https://github.com/Cambricon/vllm-mlu) — vLLM MLU fork
- [magicmind_cloud](https://github.com/Cambricon/magicmind_cloud) — MagicMind inference compiler examples
- [triton-linalg](https://github.com/Cambricon/triton-linalg) — Triton → Linalg conversion
- [CNStream](https://github.com/Cambricon/CNStream) — Streaming inference pipeline framework
- [cambricon-k8s-device-plugin](https://github.com/Cambricon/cambricon-k8s-device-plugin) — Kubernetes integration
- [mlu-exporter](https://github.com/Cambricon/mlu-exporter) — Prometheus metrics exporter (actively maintained)
- [Gitee cambricon/torch_mlu](https://gitee.com/cambricon/torch_mlu) — Gitee mirror; branch `r1.22_pt2.4.0`, 2025-era EOL dates

### Hardware
- [MLU370-M8 FCC Filing](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf)
- [MLU370-X FCC Filing](https://fccid.io/2ARVF-MLU370-X)
- [MLU on WikiChip](https://en.wikichip.org/wiki/cambricon/mlu)
- [Cambricon-LLM Chiplet Paper (arXiv 2409.15654)](https://arxiv.org/html/2409.15654v1)

### Academic Papers
- [Cambricon ISA — ISCA 2016 (IEEE)](https://ieeexplore.ieee.org/document/7551409/)
- [Cambricon ISA extended — ACM 2019](https://dl.acm.org/doi/fullHtml/10.1145/3331469)
- [KAITIAN Communication Framework (arXiv 2505.10183)](https://arxiv.org/html/2505.10183v1)
