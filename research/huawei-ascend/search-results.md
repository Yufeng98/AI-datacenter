# Huawei Ascend Software Stack & Hardware Resources

*as_of: 2026-08-08*
*(baseline 2026-04-05; Ascend 950 update appended 2026-08-08)*
*device_class: Systolic Array Accelerator*
*seeds: https://www.hiascend.com/, https://gitee.com/ascend*

## Software Stack

### Framework Integration
- [Ascend/pytorch (torch_npu) — GitHub mirror](https://github.com/Ascend/pytorch) — Official PyTorch adapter for Ascend NPU via PyTorch PrivateUse1 backend; pip-installable `torch-npu` wheels
- [torch-npu on PyPI](https://pypi.org/project/torch-npu/) — Installable package exposing `torch.npu` device and Ascend-specific ops
- [Integrating Ascend Backend with Torchtune — PyTorch Blog](https://pytorch.org/blog/ascend-backend-w-torchtune/) — Jan 2025 tutorial; demonstrates fine-tuning on Ascend via torch_npu + PrivateUse1
- [MindSpore Overview (master)](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) — Huawei's primary AI framework; full-scenario (cloud/edge/device); source-to-source automatic differentiation; native Ascend backend
- [MindSpore White Paper (PDF)](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/white_paper/MindSpore_white_paper_enV1.1.pdf) — Architecture, design goals, auto-parallelism, and hardware integration details
- [MindSpore on Gitee](https://gitee.com/mindspore) — Primary source repo (open-source, Apache 2.0)
- [MindSpore Models (ModelZoo)](https://gitee.com/mindspore/models) — Model implementations (ResNet, BERT, GPT, etc.) for Ascend, GPU, CPU
- [Huawei Ascend Cloud Introduction — Medium](https://medium.com/@huaweiclouddevelper/a-brief-introduction-to-huawei-ascend-cloud-cbef8f25bc34) — Overview of Ascend cloud ecosystem and MindSpore integration
- [ONNX Runtime CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) — ONNX Runtime backend for Ascend CANN; C/C++ and Python APIs
- [OpenCV CANN Backend — GitHub Wiki](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) — OpenCV integration with Huawei CANN for vision inference on Ascend

### Compiler / IR
- [ATC Tool (Ascend Tensor Compiler) — Huawei Support](https://support.huawei.com/enterprise/en/doc/EDOC1100192457/2a40d134/atc-tool) — Offline model compiler: converts Caffe, TensorFlow, MindSpore, ONNX models to `.om` format for Ascend inference
- [Model Conversion Using ATC — Atlas Data Center Guide](https://support.huawei.com/enterprise/en/doc/EDOC1100155021/4ff6939/model-conversion-using-atc) — Step-by-step ATC usage: graph optimization, quantization, memory optimization, AIPP hardware preprocessing
- [Adapting YOLOv7 to Ascend Processors — Medium](https://medium.com/huawei-developers/adapting-the-yolov7-model-to-ascend-processors-baae63d7d5a3) — Practical ATC workflow tutorial
- [Model Migration on Ascend NPUs — Medium](https://medium.com/huawei-developers/model-migration-on-ascend-npus-a1658176d395) — Guide for migrating PyTorch/TF models via ATC to OM format
- [tilelang-ascend — GitHub](https://github.com/tile-ai/tilelang-ascend) — TileLang (MLIR-based DSL) adapter for Ascend; allows writing tiled kernels targeting Da Vinci hardware

### Op Library
- [CANN Documentation Portal — Huawei Support](https://support.huawei.com/enterprise/en/ascend-computing/cann-pid-251168373?category=developer-documents) — Official CANN SDK docs hub: AscendCL API, standard operator library, custom operator guides
- [CANN HCCS Installer — GitHub (hpcaitech)](https://github.com/hpcaitech/CANN-Installer) — Convenience installer for CANN releases
- [Huawei Releases Full-Stack Ascend AI Software — Huawei News](https://www.huawei.com/en/news/2020/8/huawei-hai-ascend) — Announcement of CANN 3.0 with TBE standard operator library + custom operator capability
- [Huawei Ascend C Programming Language Launched — Huawei Central](https://www.huaweicentral.com/huawei-ascend-c-programming-language-launched/) — AscendC: C/C++-compatible language for fine-grained Da Vinci hardware control (Cube, Vector, MTE units)

### Kernel Library
- [TBE Custom Operator Development Guide (TIK) — Huawei Support](https://support.huawei.com/enterprise/en/doc/EDOC1100164820/83779921/tik-introduction) — TBE-TIK: Python DSL for authoring custom operators; fine-grained control of Cube and Vector units
- [TBE Custom Operator Development Guide (DSL) — Huawei Support](https://support.huawei.com/enterprise/es/doc/EDOC1100192208/519412bd/implementation) — TBE-DSL: higher-level expression for operator definition; automatic scheduling
- [TE Custom Operator Development Guide (CLI/PDF)](https://support.huaweicloud.com/intl/en-us/odevg-te-atlas500app/odevg-te-atlas500app.pdf) — Atlas 500 App developer guide for custom TE operators
- [Parallel Scan on Ascend AI Accelerators — arXiv](https://arxiv.org/html/2505.15112v1) — Research on custom scan kernels using AscendC primitives
- [Huawei-Ascend/samples — GitHub](https://github.com/Huawei-Ascend/samples) — Official AscendCL sample code: image classification, object detection, custom operators
- [Huawei-Ascend/tools — GitHub (msame)](https://github.com/Huawei-Ascend/tools/blob/master/msame/README_EN.md) — Model inference benchmark tool for .om models on Ascend

### Runtime
- [AscendCL API Overview — Huawei Developer](https://developer.huawei.com/consumer/en/doc/hiai-guides/introduction-0000001051486804) — AscendCL: C-language runtime API for device/context/stream/memory management, model loading, op execution
- [Core Components of Ascend Processors — Medium](https://medium.com/huawei-developers/core-components-of-ascend-processors-28dc746fb94d) — Overview of CANN runtime components and their relationship to hardware
- [MindSpore Ascend Install (pip) — Gitee](https://gitee.com/mindspore/docs/blob/5fb7cd8ca54d6def2b33e84da0bf52db95852a30/install/mindspore_ascend_install_pip.md) — Installation guide for CANN + Driver/Firmware + MindSpore runtime stack
- [MindSpore Ascend Install (source) — Gitee](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_source_en.md) — Source-build guide for Ascend runtime integration

### Driver / Firmware
- [Ascend NPU Roadmap — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-npu-roadmap-examined-company-targets-4-zettaflops-fp4-performance-by-2028-amid-manufacturing-constraints) — Overview of Ascend driver stack, firmware constraints under SMIC process, roadmap to 910D/920
- [Ascend-CC: Confidential Computing on Heterogeneous NPU — arXiv](https://arxiv.org/html/2407.11888v1) — Security analysis of Ascend firmware; TEE integration, memory isolation mechanisms
- [TechInsights Teardown: Ascend 910C — SemiWiki](https://semiwiki.com/forum/threads/techinsights-teardown-huawei-ascend-910c-still-contains-cpu-dies-from-tsmc-from-2020.23737/) — Die analysis confirming CPU companion die (TSMC) + SMIC NPU die architecture

### Communication
- [Ascend HCCS Interconnect NPU Affinity — Huawei Cloud Docs](https://support.huaweicloud.com/intl/en-us/drawer-cce/Cluster_cce-gpu-topology-priority.html) — CCE scheduler documentation for HCCS-interconnected NPU affinity scheduling
- [Ascend HCCS Preselection Scheduling — Huawei Cloud Docs](https://support.huaweicloud.com/intl/en-us/drawer-cce/Cluster_cce-gpu-topology-predicate.html) — HCCS topology-aware scheduling policy documentation
- [Atlas 900 AI Cluster — CIO / Yicai Global](https://www.cio.com/article/217641/interpretation-on-supreme-computing-of-huawei-atlas-900-ai-cluster.html) — Atlas 900 networking: HCCS + PCIe 4.0 + 100GE full-mesh 100 TB/s synchronization fabric
- [Huawei CloudMatrix 384 Analysis — SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72) — Deep-dive: 384 × Ascend 910C, 6912 × 400G silicon photonic LPO, 2.8 Tbps/chip scale-up
- [CloudMatrix 384 LPO Optics — QSFPTEK](https://www.qsfptek.com/qt-news/400g-osfp-siph-lpos-in-huawei-ai-cloudmatrix384-super-node.html) — Technical analysis of CloudMatrix 384 optical interconnect: 6912 OSFP 400G SiPh LPO modules
- [Huawei Ascend Chip Roadmap — Convequity](https://convequity.substack.com/p/huawei-ascend-ai-chip-roadmap-and) — HCCS 4.0 roadmap targeting 100K-chip clusters; system-level performance data
- [Serving LLMs on Huawei CloudMatrix 384 — arXiv](https://arxiv.org/html/2506.12708v3) — Practical inference study on CloudMatrix 384 with HCCS interconnect topology
- [Huawei Ascend & Kunpeng Progress — Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/huaweis-ascend-and-kunpeng-progress-shows-how-china-is-rebuilding-an-ai-compute-stack-under-sanctions) — Context on HCCS vs NVLink; scale-up vs scale-out architecture decisions

### Assembler / ISA
- [DaVinci: A Scalable Architecture for Neural Network Computing (Huawei/CMC PDF)](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf) — Primary public ISA reference: CCE (Core Compute Engine) instruction set; Cube/Vector/Scalar/MTE instruction streams
- [DaVinci Scalable Architecture (Semantic Scholar PDF)](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf) — Alternate version of the Da Vinci architecture paper with memory hierarchy details
- [Performance Modeling on DaVinci AI Core — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S074373152300014X) — Performance model for CCE instruction scheduling on Da Vinci; predicts execution time per kernel
- [Ascend AI Processor Architecture and Programming — O'Reilly (Chapter 3)](https://www.oreilly.com/library/view/ascend-ai-processor/9780128234891/B9780128234884000035.xhtml) — Hardware architecture chapter: Cube unit, Vector unit, DMA engines, instruction pipeline
- [Atlas AI Computing Platform — CUHK Lecture (PDF)](https://www1.se.cuhk.edu.hk/~seem5730/l22/06%20Atlas%20AI%20Computing%20Platform_ALEX.pdf) — Academic overview: Da Vinci ISA primitives, programming model layers
- [HCIA-AI Course — GitHub (gabboraron)](https://github.com/gabboraron/HCIA-AI-Course) — Huawei/University HCIA-AI curriculum including Da Vinci architecture, MindSpore, CANN

---

## Hardware Architecture

### Compute Engine
- [Huawei Ascend 910B/910C Architecture — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910) — 910B: 25 "New DaVinci" cores, SMIC N+1, 320 TFLOPS FP16, 640 TOPS INT8; 910C: dual-die 2×910B, ~800 TFLOPS FP16
- [Huawei Ascend 910C Specs — Awesome Agents](https://awesomeagents.ai/hardware/huawei-ascend-910c/) — 910C: Da Vinci 3.0 architecture, dual-die, ~30–35% per-core throughput improvement over 910B
- [Huawei Ascend vs NVIDIA H100 — NexGen Compute](https://www.nexgen-compute.com/blog/huawei-ascend-910c-vs-nvidia-h100-ai-chip-comparison) — Comparative specs: 910C vs H100 compute, memory, interconnect
- [Ascend 910 Feature Highlights](https://www.actfornet.com/products/intelligent-computing/atlas/huawei-ai/ai-chips/Ascend_910/features) — 910: 32 Da Vinci Max cores, 256 TFLOPS FP16, 512 TOPS INT8
- [Huawei Ascend AI Chip Specs 2025 — Bitrue](https://www.bitrue.com/blog/huawei-ascend-ai-chip-specs-2025) — Generational specs table: 910, 910B, 910C, 910D roadmap
- [Understanding Huawei Ascend Chip — EEWorld](https://en.eeworld.com.cn/mp/XSY/a400532.jspx) — Core cluster organization: 4 clusters × 8 cores, NoC mesh, 1024-bit interconnect at 2 GHz
- [DaVinci Architecture Paper (CMC PDF)](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf) — Cube unit: 16×16×16 FP16 MAC per cycle (4096 MACs), INT8 doubles to 8192; Vector unit: 128×FP16 or 256×INT8 per cycle

### Data Path
- [Performance Modeling on DaVinci AI Core — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S074373152300014X) — Instruction pipeline model; Cube/Vector/MTE parallelism; pipeline interlock characterization
- [Ascend AI Processor Architecture — O'Reilly Chapter 3](https://www.oreilly.com/library/view/ascend-ai-processor/9780128234891/B9780128234884000035.xhtml) — Five independent execution units per AI Core: Cube, Vector, Scalar, MTE1 (DMA in), MTE2 (DMA out); double-buffering via L1/L0 memory
- [NeoNeXt: Novel Neural Network Operators — arXiv](https://arxiv.org/pdf/2403.11251) — Ascend data path constraints; operator design tradeoffs for Da Vinci

### On-chip Memory
- [DaVinci Architecture Paper (Semantic Scholar)](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf) — L1 (input buffer, 32 KB per core), L0A/L0B (Cube operand buffers, 16 KB each), L0C (accumulator buffer, 32 KB), UB (Unified Buffer ~256 KB), L2 (32 MB shared); NoC: 4 TB/s
- [Huawei Ascend 910 Provides NVIDIA Alternative — ServeTheHome](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/) — Total on-chip SRAM: 84 MB (including 32 MB shared L2); 4 TB/s NoC bandwidth
- [Practical Storage Hierarchy — Arthur Chiao](http://arthurchiao.art/blog/practical-storage-hierarchy/) — Comparative context for Ascend memory hierarchy vs GPU/TPU

### Off-chip Memory
- [Huawei Ascend 910 Specs — ServeTheHome](https://www.servethehome.com/huawei-ascend-910-provides-a-nvidia-ai-training-alternative/) — 910: 32 GB HBM2, 1,228 GB/s (4 channels); 910B: 64 GB HBM2e; 910C: 128 GB HBM2e (dual-die)
- [Huawei Ascend 910B Examined — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910) — 910B: 64 GB HBM2e, ~800 GB/s bandwidth; SMIC N+1, 21.32×31.22 mm die
- [Huawei Ascend NPU Roadmap — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-npu-roadmap-examined-company-targets-4-zettaflops-fp4-performance-by-2028-amid-manufacturing-constraints) — Roadmap: 910D, 920 targeting 4 ZettaFLOPS FP4 by 2028 with next-gen HBM

### Host Interface / Package
- [Huawei Ascend 910C Die Analysis — TechInsights/SemiWiki](https://semiwiki.com/forum/threads/techinsights-teardown-huawei-ascend-910c-still-contains-cpu-dies-from-tsmc-from-2020.23737/) — 910C: dual-die organic interposer package; CPU companion die (TSMC 7nm, 2020 vintage) + 2× SMIC N+1 NPU dies; PCIe 4.0 host interface
- [Atlas AI Computing Platform — CUHK Lecture](https://www1.se.cuhk.edu.hk/~seem5730/l22/06%20Atlas%20AI%20Computing%20Platform_ALEX.pdf) — Atlas 300 PCIe card form factor; Atlas 800 server (8× Ascend 910 via HCCS)
- [Huawei Ascend 910C Challenge to NVIDIA — Unite.ai](https://www.unite.ai/huaweis-ascend-910c-a-bold-challenge-to-nvidia-in-the-ai-chip-market/) — 910C TDP ~400W; OAM-like form factor for Atlas 800T server

### Scale-up Interconnect
- [Huawei Atlas 900 HCCS Interconnect — CIO](https://www.cio.com/article/217641/interpretation-on-supreme-computing-of-huawei-atlas-900-ai-cluster.html) — HCCS (Huawei Compute Communication System): proprietary high-speed bus, higher BW than PCIe switch; 8 Ascend 910s per server via HCCS
- [Ascend HCCS Affinity Scheduling — Huawei Cloud](https://support.huaweicloud.com/intl/en-us/drawer-cce/Cluster_cce-gpu-topology-priority.html) — HCCS topology-aware scheduling; intra-server NPU affinity groups
- [CloudMatrix 384 — SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72) — CloudMatrix 384: 56 × 400G SiPh LPO per server for scale-up; 2.8 Tbps/chip; 384-chip all-to-all optical fabric; HCCS 4.0 implied
- [Huawei Ascend 910D HCCS 4.0 — Convequity](https://convequity.substack.com/p/huawei-ascend-ai-chip-roadmap-and) — HCCS 4.0: targeting 100,000-chip cluster connectivity

### Scale-out Interconnect
- [CloudMatrix 384 LPO Optics — QSFPTEK](https://www.qsfptek.com/qt-news/400g-osfp-siph-lpos-in-huawei-ai-cloudmatrix384-super-node.html) — 8 × 400G SiPh LPO per server for scale-out; 3,168 fibers total in 16-rack system
- [CloudMatrix 384 Overview — NotebookCheck](https://www.notebookcheck.net/CloudMatrix-384-Huawei-s-384-chip-AI-cluster-challenges-Nvidia-amid-US-export-curbs.1069155.0.html) — Full-mesh optical interconnect across 16 racks (12 compute + 4 switch racks)
- [Huawei Rack-Scale vs NVIDIA — The Register](https://www.theregister.com/2025/07/29/huawei_rackscale_boogeyman/) — CloudMatrix vs GB200 NVL72 comparison; optical scale-out strategy
- [Serving LLMs on CloudMatrix 384 — arXiv](https://arxiv.org/html/2506.12708v3) — Inference topology study; scale-out bandwidth utilization patterns

---

## Other Resources

### Architecture Papers & Academic
- [Huawei MindSpore AI Development Framework — Springer Nature](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_5) — Book chapter: MindSpore architecture, Ascend integration
- [Huawei Atlas AI Computing Solution — Springer Nature](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_6) — Book chapter: Atlas hardware platform, Da Vinci core, CANN stack
- [Analysis of Performance and Optimization in MindSpore on Ascend NPUs — IEEE Xplore](https://ieeexplore.ieee.org/document/10475950/) — Empirical performance study; operator scheduling, memory optimization
- [MindSpore Wikipedia](https://en.wikipedia.org/wiki/MindSpore) — Framework overview, open-source history, supported hardware

### Tutorials & Community
- [Huawei HCIA-AI V3.0 Course — GitHub (gabboraron)](https://github.com/gabboraron/HCIA-AI-Course) — Full certification course: Ascend architecture, MindSpore, CANN, traditional ML
- [MindSpore Quick Guide — Medium](https://medium.com/@huaweiclouddevelper/mindspore-a-quick-guide-to-understanding-the-ascend-ai-framework-and-getting-startedwhat-is-bdbb5f828cfe) — Getting started with MindSpore on Ascend
- [Mastering YOLO/ResNet on HiSilicon NPUs](https://ic-online.com/news/post/mastering-yolo-and-resnet-optimization-on-hisilicon-npus) — Optimization techniques for common models on Ascend NPU

### Benchmarks & Analysis
- [Huawei Ascend 910C vs NVIDIA H100 — NexGen Compute](https://www.nexgen-compute.com/blog/huawei-ascend-910c-vs-nvidia-h100-ai-chip-comparison) — Performance comparison: training throughput, memory capacity, interconnect bandwidth
- [Huawei New AI CloudMatrix vs GB200 — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power) — System-level benchmark: CloudMatrix 384 beats GB200 NVL72 at 4× power envelope
- [Huawei's Ascend 910C and CloudMatrix Fill China Void — XPU.pub](https://xpu.pub/2025/04/22/huawei-ascend/) — Market analysis; 700K–800K annual 910C unit target for 2025

---

## Resources Added 2026-08-08 — Ascend 950 Generation

*Scan window 2026-04-05 → 2026-08-08. Methodology caveat: this scan's WebSearch budget was exhausted before it began; discovery ran via WebFetch against DuckDuckGo HTML/Lite result pages plus direct primary-source fetches. `hiascend.com/en/software/cann` and `hotchips.org/program` (HTTP 403) yielded no content, so low-visibility coverage is thinner than a normal scan.*

### Primary Vendor Sources
- [CANN 9.0.0 Commercial Release Notes — hiascend.com](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md) — **primary**. Released 2026-05-09 (betas 2026-03-09, 2026-03-30). Verbatim "CANN新增适配Ascend 950PR（Atlas 350加速卡）"; FP8/MXFP8/MXFP4; "AscendC支持SIMD+SIMT混合编程" with ~700 new SIMT APIs; Reg-based programming on 950PR; HcclGroupStart/End batched collectives; "集合通信支持CCU通信加速"; CATLASS 950 co-compute + Fixpipe modes; apt/pip packaging
- [Ascend 950 NPU Architecture Whitepaper (PDF, ~38 pp)](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — **primary, but IMAGE-ONLY (6.1 MB, no text layer)**. `Last-Modified: 2026-06-04`. Extracting its text is the highest-value next research step
- [Atlas 950 SuperPoD at WAIC 2026 — Huawei (2026-07-17)](https://www.huawei.com/cn/news/2026/7/atlas-950-superpod) — **primary**. 1,024-card demo, 1 EFLOPS FP8 / 2 EFLOPS FP4, 256 TB unified address space, 3 μs RTT over 灵衢; 真机首次公开亮相; SAIL award; Atlas 850E at 96 cards; 750+ Atlas 384-series deployments
- [SuperPoD announcements at MWC 2026 — Huawei](https://www.huawei.com/en/news/2026/3/mwc-superpod-ai) — **primary**. 64 NPUs/cabinet → 8,192; Atlas 850E air-cooled scaling 8 → 1,024 NPUs
- [UnifiedBus open specification](https://www.unifiedbus.com/en) — **primary**. Base spec, firmware spec, SW reference designs. Launched 2025-09-18 at Huawei Connect 2025 by Xu Zhijun (predates the repo baseline)
- [openEuler UB Service Core SW Architecture Reference Design 2.0 (PDF)](https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf) — UnifiedBus service-core software architecture
- [MindSpore Release Notes (canonical)](https://www.mindspore.cn/docs/en/stable/RELEASE.html) — **use this, not the Gitee list**
- [MindSpore 2.9 Version Notes](https://www.mindspore.cn/version-updates/en/2_9_en) — 2.9.0, 2026-05-07, paired with CANN 9.0.0, native Ascend 950PR
- [MindSpore on PyPI](https://pypi.org/project/mindspore/) — corroborates the 2.9.0 release
- [MindSpore Gitee Releases](https://gitee.com/mindspore/mindspore/releases) — **STALE MIRROR** (showed only up to v2.7.2 / CANN 8.5.0). Do not treat as canonical

### Open-Source Repositories
- [CANN open-source organization — GitCode](https://gitcode.com/cann) — ~76 repos, active through early Aug 2026: `ops-transformer`, `ops-math`, `ops-nn`, `ops-cv`, `hccl`, `hcomm`, `shmem`, `asc-devkit`, `pypto`, `pyasc`, `runtime`, `ge`, `graph-autofusion`, `cann-recipes-infer`, `cann-recipes-train`, `cann-bench`, `amct`. **Userspace only — kernel driver and firmware remain closed**
- [vllm-ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html) — Ascend 950 enablement trail: MXFP4 (v0.20.2rc1), full DeepSeek-V4 (v0.21.0rc1), dynamic quant + Mooncake (v0.22.1rc1), W4A16 MXFP4 + CANN 9.0.1 requirement (v0.23.0rc1, 2026-07-19)
- [vllm-ascend GitHub Releases](https://github.com/vllm-project/vllm-ascend/releases) — corroborating release dates
- [UnifiedBus community mirror — GitHub](https://github.com/codehubcloud/UnifiedBus)

### Press and Analysis
- [Atlas 350 debut on Ascend 950PR — TrendForce (2026-03-23)](https://www.trendforce.com/news/2026/03/23/news-huawei-debuts-atlas-350-on-ascend-950pr-with-in-house-hbm-touting-2-8x-h20-performance/) — **pre-baseline miss**. Zhang Dixuan: 1.56 PFLOPS FP4, up to 112 GB, 1.4 TB/s, 600 W; "2.8× H20" is a vendor marketing comparison
- [Atlas 350 unveiled — Tom's Hardware (2026-03-24)](https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20) — corroborates the shipping-card derate
- [Ascend 950PR / Atlas 350 FP4 — Intelligent Living](https://intelligentliving.co/huawei-ascend-950pr-atlas-350-fp4-ai/) — additional corroboration of the card spec
- [Ascend 950DT deployment pulled forward to August — TrendForce (2026-06-08)](https://www.trendforce.com/news/2026/06/08/news-huawei-brings-forward-ascend-950dt-deployment-to-august-deepseek-v4-2-seen-as-potential-early-adopter/) — Chen Lin at Huawei Cloud 2026 INSPIRE. **Announced schedule, not a live deployment**; Q4 2026 remains official; "SMIC N+3" is analyst inference only
- [Huawei confirms Ascend 950DT debut in August — Huawei Central](https://www.huaweicentral.com/huawei-confirms-ascend-950dt-ai-chip-to-debut-in-august/) — corroborating report
- [Huawei Cloud ties agentic infra to Ascend 950DT window — WinBuzzer (2026-06-10)](https://winbuzzer.com/2026/06/10/huawei-cloud-ties-agentic-infra-to-ascend-950dt-window-xcxwbn/) — cloud-side framing of the same announcement
- [Ascend 950 NPU architecture analysis (third-party)](https://pillumina.github.io/posts/aiinfra/ascend-950-npu/) — **SINGLE-SOURCE. Handle with care.** Sole origin of the unconfirmed 18-cores-per-die / 36 Cube + 72 Vector, 64 KB L0A/L0B, 512 KB L1, 512 KB UB, 512 B/4×128 B sector cache, BufferID synchronization, "Linx816" AI CPU, UnifiedBus RTP/CTP split, and the 547 TFLOPS BF16 / 273 TFLOPS TF32 rungs. Independently corroborated portions: AIC/AIV core split, 1 Cube + 2 Vector per subsystem, 128 MB L2, STARS 2.0, NDDMA, PCIe 5.0 ×16, dual 400 Gbps UBoE
- [CANN 9.0.0 / MindSpore 2.9.0 upgrade writeup (OrangePi AIpro 20T)](https://www.donaldsebleung.com/blog/20260509-upgrading-to-mindspore-290-plus-cann-900-on-the-orangepiaipro-20t) — community install report, 2026-05-09; useful as field evidence, **not** as the release date
- [CANN component coverage — CSDN](https://blog.csdn.net/gitblog_00311/article/details/160916818) — walkthrough of the open-sourced CANN repos
- [CANN developer article — CSDN DevPress](https://devpress.csdn.net/v1/article/detail/153810476) — CANN 9.0.0 feature coverage

### Benchmarks / Negative Results
- [MLPerf Inference v6.0 results — MLCommons (2026-04-01)](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/) and [datacenter inference benchmark index](https://mlcommons.org/benchmarks/inference-datacenter/) — 24 submitting organizations. The submitter list could not be enumerated, so "no Huawei Ascend submission" is **plausible but UNVERIFIED**, not a confirmed negative. Also pre-baseline
- [Hot Chips](https://hotchips.org) — **Hot Chips 38 runs 2026-08-23 to 08-25, i.e. after this scan.** No Ascend content exists yet; absence carries no evidentiary weight. Any Ascend talk there is "disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public". `hotchips.org/program` returned HTTP 403
