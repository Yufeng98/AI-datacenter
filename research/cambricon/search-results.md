# Cambricon MLU Software Stack & Hardware Resources

*as_of: 2026-04-05*
*device_class: Neural Processor*
*seeds: https://developer.cambricon.com/*

## Software Stack

### Framework Integration
- [GitHub - Cambricon/torch_mlu](https://github.com/Cambricon/torch_mlu) — Official PyTorch extension for Cambricon MLU; depends on CNToolkit, CNNL, CNCL, CNCV, BANGC OPS; versioned as `{torch_mlu_ver}+torch{pytorch_ver}`
- [GitHub - Cambricon/catch](https://github.com/Cambricon/catch) — Earlier Cambricon Adaptive Tool Chain for Hardware (CATCH), predecessor PyTorch integration layer
- [GitHub - jklincn/cambricon-pytorch](https://github.com/jklincn/cambricon-pytorch) — Community guide for building Cambricon PyTorch from source
- [GitHub - Cambricon/vllm-mlu](https://github.com/Cambricon/vllm-mlu) — vLLM fork adapted for Cambricon MLU; supports Chunk Prefill, Prefix Caching, Spec Decode, Graph Mode; requires SDK 25.08, MLU370+
- [vLLM PR #10315 — Add Cambricon MLU inference backend](https://github.com/vllm-project/vllm/pull/10315) — Upstream vLLM integration of MLU backend with custom Paged Attention and KV cache ops
- [PaddlePaddle/PaddleCustomDevice — Cambricon MLU backend](https://github.com/PaddlePaddle/PaddleCustomDevice) — PaddlePaddle custom device plugin for MLU; loads libpaddle-custom-mlu.so with 264+ custom kernels
- [PaddleX Cambricon MLU support](https://github.com/PaddlePaddle/PaddleX/blob/release/3.4/README_en.md) — PaddleX pipeline framework with Cambricon MLU inference support across vision/NLP models
- [Design of CNML/CNRT Integration — Apache MXNet](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=120722127) — Documents earlier MXNet integration via CNML/CNRT APIs

### Compiler / IR
- [Cambricon BANG C Developer Guide (online docs)](https://www.cambricon.com/docs/bangc/developer_guide_html/) — Official BANG C language and CNCC compiler reference (version 2.15.0)
- [BANG C Language Developer Guide v2.4.1 (PDF)](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf) — Detailed language spec covering scalar/vector/matrix ops, NRAM/WRAM memory model
- [GitHub - Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg) — Converts Triton dialect to Linalg dialect; supports Cambricon backend as front-end representation
- [MagicMind — Cambricon inference compiler (magicmind_cloud)](https://github.com/Cambricon/magicmind_cloud) — MLIR-based inference engine; converts TF/PyTorch/ONNX/Caffe to optimized MLU code; end-to-end optimization and code generation
- [A high-performance compiler tool chain for deep learning (PDF)](https://subjectnoi.github.io/about/Paleozoic.pdf) — Academic paper covering Cambricon's early CNCC/CNAS toolchain design

### Op Library
- [GitHub - Cambricon/mlu-ops](https://github.com/Cambricon/mlu-ops) — Efficient operator implementations in BANG C for MLU; includes CNNL integration helpers and hundreds of ML ops
- [CNML Developer Guide v7.10.2 (PDF)](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf) — Cambricon Neuware Machine Learning Library (CNML) API and usage guide
- [MLU-OPS: How to Use CNNL API](https://github.com/Cambricon/mlu-ops/blob/master/docs/MLU-OPS-How-To-Use-CNNL-API.md) — Guide for converting MLU-OPS parameters to CNNL types

### Kernel Library
- [Cambricon BANG C Developer Guide (online docs)](https://www.cambricon.com/docs/bangc/developer_guide_html/) — Official BANG C kernel programming reference including built-ins, memory intrinsics, and synchronization barriers
- [BANG C Language Developer Guide v2.4.1 (PDF)](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf) — Covers BANG C kernel programming model including memory intrinsics, barrier synchronization, and built-in functions
- [Cambricon BANGPy — Python-level kernel programming](https://www.nobleprog.com/cambricon-mlu-training) — BANGPy provides Python API for writing MLU kernels (graph/kernel/operator-level optimization); part of NeuWare SDK
- [GitHub - JamiesZhang/BangC-example](https://github.com/JamiesZhang/BangC-example) — Community BANG C kernel examples for MLU

### Runtime
- [Cambricon Developer Portal](https://developer.cambricon.com/) — Official SDK and NeuWare runtime downloads, documentation, and developer community (login required for full access)
- [寒武纪运行时库用户手册 — CNRT User Guide v6.0.0](https://www.cambricon.com/docs/cnrt/user_guide_html/index.html) — Official CNRT API documentation covering device management, memory, queues, synchronization, and CNRT+CNDrv mixed-call programming model
- [寒武纪 CNRT 用户手册 — Developer Portal](https://developer.cambricon.com/index/document/details/classid/3/cid/13/id/505.html) — CNRT manual hosted on Cambricon developer portal
- [寒武纪CNToolkit安装升级使用手册 v3.7.2](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) — CNToolkit (NeuWare SDK bundle) installation and upgrade guide; covers CNCC, CNAS, CNRT, CNDrv, CNNL, CNCL component packaging
- [cndrv — crates.io Rust Package](https://crates.io/crates/cndrv) — Rust bindings for CNDrv (Cambricon Driver API); documents CNRT and CNDrv interface
- [Cambricon PyTorch 使用说明 (MLU270)](https://zhuanlan.zhihu.com/p/536304961) — Chinese tutorial covering driver/runtime installation and model migration for MLU270

### Driver / Firmware
- [Cambricon MLU Driver — Gitee mirror](https://gitee.com/jimpirror/cambricon-mlu-driver) — Open mirror of Cambricon MLU kernel driver source
- [驱动和系统工具 — Cambricon Developer Forum](https://forum.cambricon.com/list-149-3.html) — Official forum section for driver packages and system tools
- [寒武纪MLU220-M.2 edge card driver guide](https://wiki.t-firefly.com/AIO-3399J/AIO_3399J_cambricon_mlu220.html) — Firefly board driver integration guide for MLU220

### Communication
- [KAITIAN: Communication Framework for Heterogeneous Accelerators](https://arxiv.org/html/2505.10183v1) — Documents CNCL (Cambricon Collective Communication Library) for intra-MLU collectives; describes MLU-Link multi-core interconnect usage for distributed training

### Assembler / ISA
- [Cambricon: An Instruction Set Architecture for Neural Networks (IEEE Xplore)](https://ieeexplore.ieee.org/document/7551409/) — Original ISCA 2016 paper defining the Cambricon ISA with 43 64-bit instructions (scalar/vector/matrix/control)
- [Cambricon ISA — ACM SIGARCH](https://dl.acm.org/doi/10.1145/3007787.3001179) — ACM reprint of the Cambricon ISA paper
- [An Instruction Set Architecture for Machine Learning (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3331469) — Extended journal version covering ISA design rationale and code density comparison vs. MIPS/x86/GPGPU
- [CNCC + CNAS toolchain description](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf) — CNCC compiles .mlu files to MLISA assembly; CNAS assembles to .cncode/.cnbin/.cnfatbin

## Hardware Architecture

### Compute Engine
- [Machine Learning Unit (MLU) — WikiChip](https://en.wikichip.org/wiki/cambricon/mlu) — Overview of MLU core microarchitecture: MLU Core → Cluster (4 cores + memory core + shared SRAM) hierarchy
- [MLU200 — WikiChip](https://en.wikichip.org/wiki/cambricon/mlu/mlu200) — MLU200 series specifications: die, process, core count
- [Cambricon: An Instruction Set Architecture for Neural Networks (IEEE)](https://ieeexplore.ieee.org/document/7551409/) — Describes load-store compute engine with 64-bit instructions, 64 GPRs, vector/matrix functional units
- [Optimizing Logcumsumexp on Cambricon MLU (ResearchGate)](https://www.researchgate.net/publication/393382745_Optimizing_Logcumsumexp_on_Cambricon_MLU_Architecture-Aware_Scheduling_and_Memory_Management) — Architecture-aware scheduling study revealing compute pipeline details
- [Cambricon-LLM: Chiplet-Based Hybrid Architecture (arXiv)](https://arxiv.org/html/2409.15654v1) — Describes chiplet packaging approach for on-device 70B LLM inference

### Data Path
- [BANG C Developer Guide (online)](https://www.cambricon.com/docs/bangc/developer_guide_html/) — Documents data path between NRAM/WRAM/shared SRAM and compute units; DMA engine usage
- [Cambricon MLU programming model overview](https://en.wikichip.org/wiki/cambricon/mlu) — Heterogeneous host+MLU data path: PCIe DMA transfers and on-chip data movement

### On-chip Memory
- [BANG C Language Developer Guide v2.4.1 (PDF)](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf) — NRAM (input/output data for vector/tensor ops), WRAM (convolution kernel weights), shared SRAM per cluster
- [WikiChip MLU](https://en.wikichip.org/wiki/cambricon/mlu) — Cluster-level shared SRAM and per-core private memory (NRAM, WRAM) hierarchy

### Off-chip Memory
- [MLU370-M8 Product Manual (FCC filing)](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf) — MLU370-M8 specs: 48 GB LPDDR5; MLU290-M5: 32 GB HBM2 at 1,228 GB/s bandwidth
- [Cambricon targets 500,000 AI chips in 2026 (Tom's Hardware)](https://www.tomshardware.com/tech-industry/semiconductors/cambricon-targets-500000-ai-chips-in-2026-as-china-accelerates-domestic-hardware-push) — Mentions HBM supply constraints; MLU590 uses more advanced HBM
- [Cambricon AIPE profile](https://aiproduct.engineer/ai-ecosystem/cambricon-a32b3a8eeb26) — Summary of MLU memory configurations across product generations

### Host Interface / Package
- [MLU370-M8 Product Manual (FCC)](https://fcc.report/FCC-ID/2ARVF-MLU370-M8/5528126.pdf) — PCIe Gen4 x16 host interface; full-height full-length (FHFL) card form factor
- [MLU-X1001 Accelerator Product Manual](https://fcc.report/FCC-ID/2ARVF-MLU-X1001/5528197.pdf) — External accelerator module using PCIe x16 + Mini SAS HD interface
- [MLU100 — WikiChip](https://en.wikichip.org/wiki/cambricon/mlu/mlu100) — First-gen PCIe accelerator card specs
- [FCC ID 2ARVF-MLU370-X](https://fccid.io/2ARVF-MLU370-X) — FCC filing for MLU370-X intelligent accelerating card
- [Cambricon-LLM chiplet architecture (arXiv)](https://arxiv.org/html/2409.15654v1) — Chiplet-based packaging design for future LLM-optimized MLU variants

### Scale-up Interconnect
- [KAITIAN framework paper (arXiv)](https://arxiv.org/html/2505.10183v1) — MLU-Link technology for chip-to-chip direct connections; 8-card parallel training achieving 155% of 350W RTX GPU on BERT/YOLO/ResNet workloads
- [Cambricon 8-card parallel training report (copyfuture)](https://copyfuture.com/blogs-details/202203220417470682) — Benchmark report on MLU-Link 8-card training setup

### Scale-out Interconnect
- [KAITIAN: Communication for Heterogeneous Accelerators](https://arxiv.org/html/2505.10183v1) — CNCL for intra-MLU collectives; Gloo used for cross-vendor heterogeneous communication
- [Collective Communication Performance Evaluation for Distributed DL (MDPI)](https://www.mdpi.com/2076-3417/14/12/5100) — Benchmarks CNCL collectives vs. NCCL in distributed training scenarios

## Other Resources

### Kubernetes / Cloud Deployment
- [GitHub - Cambricon/cambricon-k8s-device-plugin](https://github.com/Cambricon/cambricon-k8s-device-plugin) — DaemonSet for auto-reporting MLU quantity and health status in Kubernetes clusters
- [GitHub - Cambricon/mlu-exporter](https://github.com/Cambricon/mlu-exporter) — Prometheus exporter for MLU metrics

### Media / Vision SDK
- [GitHub - Cambricon/CNStream](https://github.com/Cambricon/CNStream) — Official streaming framework for building Cambricon ML pipelines; supports RTSP/file/image sources, MLU-accelerated decode and inference
- [GitHub - Cambricon/Cambricon-Gst](https://github.com/Cambricon/Cambricon-Gst) — GStreamer plugin for Cambricon SDK (video pipeline integration)
- [GitHub - Cambricon/easydk](https://github.com/Cambricon/easydk) — Easy Development Kit for Cambricon; simplified API for edge/cloud inference
- [CNCV 使用介绍 (知乎)](https://zhuanlan.zhihu.com/p/612780111) — Introduction to CNCV (Cambricon Computer Vision library)

### Academic & Architecture Papers
- [Cambricon Technologies — Wikipedia](https://en.wikipedia.org/wiki/Cambricon_Technologies) — Company overview, product generations, founding history
- [Cambricon — WikiChip overview](https://en.wikichip.org/wiki/cambricon/mlu) — Consolidated hardware specs for all MLU generations
- [Cambricon-LLM: Chiplet Hybrid Architecture (arXiv 2409.15654)](https://arxiv.org/html/2409.15654v1) — Research paper on chiplet-based design for 70B LLM on-device inference
- [Cambricon: China's Nvidia (TechBuzzChina)](https://techbuzzchina.substack.com/p/cambricon-chinas-nvidiaor-nvidia) — Business and strategic analysis of Cambricon vs. Nvidia

### Benchmarks & Tutorials
- [Cambricon MLU Development with BANGPy and Neuware (NobleProg)](https://www.nobleprog-kz.com/cc/cambriconmlu) — Training course outline for MLU development
- [Performance Optimization on Ascend, Biren, and Cambricon (NobleProg)](https://www.nobleprog-kw.com/cc/poabc) — Cross-platform optimization training
- [Cambricon NeuWare breaks China's dependence on Nvidia CUDA (Digitimes)](https://www.digitimes.com/news/a20251106PD216/cambricon-cuda-nvidia-competition-chipmakers.html) — Industry analysis on NeuWare ecosystem maturity
- [Adopting MLU for Geneformer Model (ACM DL)](https://dl.acm.org/doi/10.1145/3664934.3664943) — Case study of porting a bioinformatics model to Cambricon MLU

---

## Resources Added 2026-08-08

*Scan window: 2026-04-05 → 2026-08-08. Sources are grouped by evidence tier, because the Siyuan 690 material is entirely secondary and must not be mixed with vendor documentation.*

### Primary — Cambricon or vendor-controlled
- [Cambricon 2026 半年度报告 (cninfo filing, 2026-08-08)](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF) — H1 2026 revenue RMB 599,557.36万 (+108.13%), net profit RMB 231,091.21万 (+122.61%), ex-nonrecurring +137.30%, inventory RMB 824,750.95万 = 45.32% of total assets. Names only 思元100/220/270/290/370; **zero** mentions of 思元590, 思元690, HBM, Torch-MLU-Ops or CNToolkit. Corroborates PD separation and DeepSeek Day-0 generically; states continued tracking of Triton community releases
- [DeepSeek-V4 Day-0 adaptation — Cambricon developer article (2026-04-24)](https://developer.cambricon.com/index/article/details.html?id=12) — DeepSeek-V4-flash 285B and V4-pro 1.6T; TP/PP/SP/DP/EP five-dimensional hybrid parallelism; Torch-MLU-Ops as 自研高性能融合算子库 accelerating Compressor and mHC; BANG C kernels for sparse/compressed attention and GroupGemm; PD 分离部署; board options MLU370-S4/X4/X8
- [vllm-mlu README (master)](https://raw.githubusercontent.com/Cambricon/vllm-mlu/master/README.md) — changelog single entry `[2026.04.24] vllm_mlu day0支持DeepSeek-V4`; features Chunk Prefill, Prefix Caching, Spec Decode, Graph Mode, Sleep Mode; hardware requirement `MLU370以上的设备`
- [mlu-ops README (master)](https://raw.githubusercontent.com/Cambricon/mlu-ops/master/README.md) — current requirement: CNToolkit v4.1.0+, CNNL v1.28.0+, driver v6.0.3+
- [mlu-ops releases (GitHub API)](https://api.github.com/repos/Cambricon/mlu-ops/releases) — v1.8.1 published **2026-01-06**, with note that future releases will not be published on that page; v1.4.0 references toolkit update to 4.0
- [Cambricon org repositories sorted by push (GitHub API)](https://api.github.com/orgs/Cambricon/repos?sort=pushed&direction=desc) — pushed_at: mlu-ops / cambricon-k8s-device-plugin / mlu-exporter 2026-08-07; mmcv 2026-07-27; mlu-ops-proto 2026-07-07; vllm-mlu 2026-05-11; torch_mlu 2025-03-15; triton-linalg 2025-02-07
- [Gitee cambricon/torch_mlu](https://gitee.com/cambricon/torch_mlu) — branch `r1.22_pt2.4.0` (PyTorch 2.4.0), 2025-era EOL dates, no MLU590/MLU690 mention
- [CNToolkit 3.8.4 release-note tree](https://sdk.cambricon.com/static/independent/CNToolkit/3.8.4/releasenote/) — intermediate version, search-indexed; page returns HTTP 401; superseded by the v4.1.0+ requirement

### Secondary — component distribution and mirrors
- [torch_mlu_ops v1.3.2 (released 2026-02-04)](https://dev.modelhub.org.cn/Chranos/enginex-mlu370-vllm/src/tag/v0.0.6/torch_mlu_ops-v1.3.2) — "PyTorch third-party operator library designed and developed by Cambricon"; deps Torch-MLU v1.24.1, CNNL v1.28.3; already backend for vLLM, TGI, Stable Diffusion WebUI
- [Torch-MLU-Ops — Baidu Baike](https://baike.baidu.com/item/Torch-MLU-Ops/67674918) — high-performance fused operator library independently developed by Cambricon
- [huismiling/torch-mlu-ops (third-party mirror)](https://github.com/huismiling/torch-mlu-ops)
- [vllm-project/vllm PR #25942 overview](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/25942/overview) — "[Doc] Add Cambricon MLU support", merged 2025-09-30; pre-dates the repo baseline; distinct from PR #10315

### Third-party market reporting — targets and estimates, NOT vendor guidance
- [Tom's Hardware — Cambricon targets 500,000 AI chips in 2026](https://www.tomshardware.com/tech-industry/semiconductors/cambricon-targets-500000-ai-chips-in-2026-as-china-accelerates-domestic-hardware-push) — 500,000-unit 2026 target, of which as many as ~300,000 Siyuan 590/690; cites low yields and limited HBM supply as threats. Figures trace upstream to Bloomberg supply-chain reporting dated **2025-12-04** — pre-dates the 2026-04-05 repo baseline
- [TrendForce — Cambricon remains China's top AI chip startup (2025-12-15)](https://www.trendforce.com/news/2025/12/15/insights-cambricon-remains-chinas-top-ai-chip-startup-rumored-2026-triple-output-faces-smic-limits/) — Siyuan 690 still in testing phase; large-scale production potentially slipping to H2 2026; SMIC 7nm-class capacity contention with Huawei
- [aiexpert.news ticker — 500K shipment target, 300K Siyuan 590/690](https://www.aiexpert.news/en/ticker/cambricon-targets-500k-ai-accelerator-shipments-in-2026-300k-units-siyuan-590690) — downstream echo of the same Bloomberg reporting

### Unverified vendor-adjacent — Siyuan 690 specs (DO NOT promote to confirmed tables)
- [Sina Finance brokerage report on Siyuan 690](https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/831133396021/index.phtml) — FP16 >700 TFLOPS, 196 GB HBM3, 互连带宽超**890Gbps** (gigabits, ≈111 GB/s — not GB/s)
- [51CTO blog on Siyuan 690](https://blog.51cto.com/u_10819805/14632220) — dual-die chiplet, 196 GB HBM3, 3.35 TB/s memory bandwidth, INT4 2,800+ TOPS
- [Tianyancha news item on Siyuan 690](https://news.tianyancha.com/ll_5k94fbk79z.html) — **conflicting** node claim: TSMC 7nm, >50B transistors, FP8 2,000 TOPS. The 4nm-vs-7nm contradiction across sources is real; no node is published in this survey
- [Zhihu post claiming early-2026 mass production](https://zhuanlan.zhihu.com/p/2061515468637214392) — contradicted by concurrent coverage describing the 690 as still in final testing, and unmentioned in the H1 filing
- [news.aibase.com item on Torch-MLU-Ops / DeepSeek-V4 (2026-04-24)](https://news.aibase.com/news/27450) — secondary write-up of the Cambricon developer article

### Negative results (searched, nothing found)
- No Cambricon submission in MLPerf Inference v6.0 (2026-04) or any prior MLPerf round
- No Cambricon paper or talk at ISCA 2026 or Hot Chips 2026
- No `MLUarch04` / `MLUarch05` architecture name confirmed anywhere
- No discontinuation or cancellation news for any MLU part
- Cambricon's own product pages returned HTTP 500 throughout the scan, so no vendor datasheet could be retrieved for MLU590 or MLU690
