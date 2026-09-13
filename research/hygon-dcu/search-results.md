# Hygon DCU (海光深算) Software Stack & Hardware Resources

*as_of: 2026-08-08*
*device_class: GPU/DCU (China, 海光)*
*seeds: https://developer.sourcefind.cn/dtk, https://github.com/FlyAIBox/dcu-in-action, https://github.com/Project-HAMi/HAMi*

---

## Background

Hygon (海光信息, Hygon Information Technology, SHA: 688041) is a publicly listed Chinese fabless semiconductor company. In 2016 AMD and Hygon formed two joint ventures (Chengdu Haiguang Microelectronics and Chengdu Haiguang Integrated Circuit Design) to bring Zen-1 x86 CPUs to China (Dhyana). Hygon was added to the US Entity List in June 2019. They subsequently developed the **DCU (Deep Computing Unit / 深算处理器)** GPU-class accelerator line independently, targeting AI training and HPC. Hygon IPO'd on Shanghai STAR Market in August 2022, raising RMB 10.6 billion. In May 2025 Hygon announced a merger with Sugon (曙光/中科曙光).

**Product generations:**
- **Z100 / Z100L** — first-generation DCU (AMD Vega-derived, HBM2, PCIe Gen4)
- **K100** — second-generation (expanded memory, multi-card xGMI)
- **K100_AI** — current AI flagship (64 GB, 400 W; from HAMi device plugin registry)
- **深算一号 (DCU-Y100)** — 4,096 CUs, 32 GB HBM2, 1 TB/s, PCIe Gen4
- **深算二号 (DCU-Z100)** — 90 TFLOPS FP32 / 180 TFLOPS FP16, 350 W
- **深算三号 (BW1000)** — **current flagship** *(corrected 2026-08-08; K100_AI above is the prior generation)*. Launched 2025; Hygon 2025-12-11: "已经投入市场，受到客户认可". **No specifications disclosed by Hygon** — no CU count, HBM, node, TDP, PCIe, or xGMI figure exists in any primary source.
- **深算四号** — in R&D; Hygon 2025-12-11: "进展顺利". No timeline or specs disclosed.

> **Corporate correction (2026-08-08):** the Hygon–Sugon absorption merger announced 2025-05-25 (预案 2025-06-09, ratio 1 : 0.5525, ~RMB 116 B) was **terminated by Hygon's board on 2025-12-09**. Sugon did **not** delist and remains listed as 603019.SH. The sentence above describing the May 2025 announcement is retained for history only.

---

## Software Stack

### Framework Integration
- [PaddlePaddle DCU Support](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/dcu/install_cn.html) — Official PaddlePaddle documentation for DCU; covers DTK environment setup and paddle-DCU wheel install
- [PaddleOCR-VL Hygon DCU Tutorial](http://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Hygon-DCU.html) — End-to-end VL pipeline on DCU using PaddleOCR
- [PaddlePaddle ROCm/DCU Guide (2.4)](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4/guides/hardware_support/rocm_docs/index_cn.html) — PaddlePaddle via ROCm layer on DCU
- [FlyAIBox/dcu-in-action](https://github.com/FlyAIBox/dcu-in-action) — Community repo: LLM training (LLaMA, ChatGLM), fine-tuning, inference on DCU; PyTorch 2.4+, PaddlePaddle 2.5+, TensorFlow 2.13+
- [Running Inference with Hygon DCUs — GPUStack](https://docs.gpustack.ai/0.5/tutorials/running-inference-with-hygon-dcus/) — Tutorial for deploying LLM inference via GPUStack on K100_AI
- [PaddleX Multi-Device Usage Guide](https://paddlepaddle.github.io/PaddleX/3.2/en/other_devices_support/multi_devices_use_guide.html) — Multi-device training with DCU support
- [Running llama2-7b on DCU(K100_AI) with PaddleNLP](https://paddlenlp.readthedocs.io/en/latest/llm/docs/dcu_install.html) — PaddleNLP LLM guide for K100_AI DCU
- [Haiguang DCU + DeepSeek Localization — EEWorld](https://en.eeworld.com.cn/mp/AIxintianxia/a393532.jspx) — News on DeepSeek running on Hygon DCU

### Compiler / IR
- [DTK — 光合开发者社区 (Hygon Developer Portal)](https://developer.sourcefind.cn/dtk) — Official DTK download hub; contains hipcc (llvm-based), clang, HIP compiler toolchain, ROCm-compatible IR pipeline
- [DCU 编程实战 — CSDN (万字学习)](https://blog.csdn.net/zzzzzucc/article/details/140313167) — Detailed HIP kernel authoring on DTK, including device function compilation and launch config
- [DTK ROCm Migration Guide](https://tbr8.org/%E6%80%8E%E6%A0%B7%E9%80%9A%E8%BF%87%E6%B5%B7%E5%85%89-dcu-%E7%9A%84-dtk-%E7%8E%AF%E5%A2%83%E8%BF%9B%E8%A1%8C-rocm-%E9%A1%B9%E7%9B%AE%E8%BF%81%E7%A7%BB%EF%BC%9A%E8%A7%A3%E5%86%B3%E7%89%88%E6%9C%AC/) — Porting ROCm projects to DTK environment; version dependency management
- [Alibaba Cloud Linux DTK Image Release Notes](https://www.alibabacloud.com/help/en/alinux/user-guide/dtk-image-release-notes) — DTK container/VM image versions on Alibaba Cloud

### Op Library
- [DTK — hipDNN / MIOpen](https://developer.sourcefind.cn/dtk) — Included in DTK; MIOpen-equivalent DNN ops: Conv, BN, Pooling, Attention
- [DTK — hipBLAS](https://developer.sourcefind.cn/dtk) — BLAS-level matrix ops for DCU; GEMM, BLAS L1–L3; cuBLAS analog
- [一文读懂AI计算平台库 (CSDN)](https://blog.csdn.net/m0_49711991/article/details/135109487) — Overview of DTK compute libraries including hipBLAS, hipDNN, hipFFT, hipSPARSE

### Kernel Library
- [DTK — hipThrust / hipCUB](https://developer.sourcefind.cn/dtk) — Parallel primitives: Reduce, Scan, Sort, Histogram; ROCm rocThrust analog
- [DCU Toolkit env install — CSDN](https://blog.csdn.net/qq_27815483/article/details/141327011) — DTK installation and component list including kernel-level libraries

### Runtime
- [DTK HIP Runtime](https://developer.sourcefind.cn/dtk) — hiprt / HIP runtime for DCU; stream, event, memory management; cudart analog
- [海光 DCU DTK 平台简介 — 知乎](https://zhuanlan.zhihu.com/p/705584420) — Overview of DTK runtime environment and API compatibility with ROCm HIP
- [DCU Common Dev Questions — CSDN](https://blog.csdn.net/qq_27815483/article/details/141311424) — FAQs: runtime errors, warp size (64 vs 32), memory model, synchronization
- [Enable Hygon DCU Sharing — HAMi Docs](https://project-hami.io/docs/userguide/hygon-device/enable-hygon-dcu-sharing/) — Kubernetes-level DCU resource sharing via HAMi vGPU plugin

### Driver / Firmware
- [HAMi DCU Support — GitHub](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md) — dcu-vgpu-device-plugin; Kubernetes device plugin for DCU resource management
- [Project-HAMi/dcu-vgpu-device-plugin](https://github.com/Project-HAMi/dcu-vgpu-device-plugin) — Standalone DCU vGPU device plugin for Kubernetes
- [H3C BIOS Release Notes (Z100L support)](https://www.h3c.com/en/d_202310/1949395_294551_0.htm) — Server BIOS notes for Z100L; indicates PCIe driver integration path
- [H3C Accessory Detail R5300 G5 + Z100L](https://www.h3c.com/en/BizPortal/DownLoadAccessory/en_AccessoryDetail.aspx?ID=84eea03e-6539-403e-bf00-6be70f9fcab0) — OEM server hardware certification with DCU Z100L

### Communication
- [DTK RCCL (DCU Collective Communication)](https://developer.sourcefind.cn/dtk) — RCCL port for DCU; AllReduce, AllGather, Broadcast; NCCL analog for scale-out
- [xGMI / Infinity Fabric on DCU](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html) — Background on xGMI bandwidth used in multi-DCU intra-node communication (AMD/Hygon shared protocol)
- [RiseUnion DCU Compatibility Certification](https://www.theriseunion.com/en/blog/dcu-compatibility.html) — Third-party server/storage compatibility with Hygon DCU

### Assembler / ISA
- [DCU Architecture Internal Block Diagram — ResearchGate](https://www.researchgate.net/figure/Internal-block-diagram-of-the-Hygon-DCU-architecture-the-DCU-relies-on-its-DPP-Data_fig5_393802594) — DPP (Data Parallel Processor) internal block diagram; 60 CUs, 4× SIMD16/CU, 64-thread wavefront
- [Optimizing Depthwise Separable Conv on DCU — Springer](https://link.springer.com/article/10.1007/s42514-024-00200-3) — Architecture-level description: 60 CUs, 1.7 GHz, 64-thread wavefront, VGPR 64KB/SIMD
- [Optimizing Sparse GEMM for DCUs — Journal of Supercomputing](https://link.springer.com/article/10.1007/s11227-024-06234-2) — Microarchitecture tuning; wavefront occupancy and register pressure analysis

---

## Hardware Architecture

### Compute Engine
- [DCU Architecture Block Diagram — ResearchGate](https://www.researchgate.net/figure/Internal-block-diagram-of-the-Hygon-DCU-architecture-the-DCU-relies-on-its-DPP-Data_fig5_393802594) — Internal block diagram: DPP (Data Parallel Processor) cluster structure
- [Optimizing Depthwise Separable Conv on DCU — Springer CCF HPC](https://link.springer.com/article/10.1007/s42514-024-00200-3) — 60 CUs, 1.7 GHz, FP16/FP32/FP64/INT8; 64-thread wavefront; 4× SIMD16/CU
- [Haiguang DCU Architecture Specs — CSDN](https://blog.csdn.net/thesky123456/article/details/147719977) — Deep analysis of DCU GPGPU architecture and HPC deployment
- [Z100 Main Hardware Specs — USTC CJCP Supplement](http://cjcp.ustc.edu.cn/hxwlxb/en/supplement/90419368-fad1-4c1f-9bee-84f971f833b1) — Reference table: Hygon C86 7255 CPU + Z100 DCU hardware specifications
- [Performance Analysis of Architectures including DCU — IEEE Xplore](https://ieeexplore.ieee.org/document/10617797/) — CMP process modeling; comparative benchmark vs CPU

### Data Path
- [Optimizing Sparse GEMM for DCUs — Springer](https://link.springer.com/article/10.1007/s11227-024-06234-2) — Data path: wavefront SIMD execution, LDS (Local Data Share) reuse patterns, memory coalescing
- [DCU 64-thread wavefront analysis — CSDN](https://blog.csdn.net/zzzzzucc/article/details/140313167) — Wavefront vs warp (64 vs 32 threads); implications for occupancy and bank conflicts

### On-chip Memory
- [DCU Memory Hierarchy — CSDN Dev FAQ](https://blog.csdn.net/qq_27815483/article/details/141311424) — LDS (Local Data Share) 64 KB/CU; VGPR 64 KB/SIMD; shared memory model
- [Depthwise Conv Optimization — Springer](https://link.springer.com/article/10.1007/s42514-024-00200-3) — Quantitative: 64 KB shared memory/CU, VGPR 256 × 32-bit/thread max

### Off-chip Memory
- [Z100 Specs Reference — USTC](http://cjcp.ustc.edu.cn/hxwlxb/en/supplement/90419368-fad1-4c1f-9bee-84f971f833b1) — Z100: HBM2 memory interface
- [深算一号规格 (DCU-Y100) — 未来智库](https://www.vzkoo.com/read/2024051540a9a146aec194e8412980de.html) — Y100: 32 GB HBM2, 1 TB/s, 4× HBM2 channels
- [K100_AI 64GB — HAMi Device Plugin](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md) — K100_AI: 64 GB memory, 400 W TDP
- [Z100L 32GB — CSDN Question Thread](https://ask.csdn.net/questions/8504176) — Z100L: 32 GB variant

### Host Interface / Package
- [HAMi DCU Support Doc](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md) — PCIe Gen4 x16 host interface; device plugin model
- [H3C R5300 G5 Server with DCU Z100L](https://www.h3c.com/en/BizPortal/DownLoadAccessory/en_AccessoryDetail.aspx?ID=84eea03e-6539-403e-bf00-6be70f9fcab0) — OEM platform integration; dual-slot FHFL card

### Scale-up Interconnect
- [深算一号 xGMI 多卡互联 — 未来智库](https://www.vzkoo.com/read/2024051540a9a146aec194e8412980de.html) — xGMI protocol; Y100: 184 GB/s multi-card bandwidth
- [AMD xGMI / Infinity Fabric Background — ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html) — xGMI technical foundation shared with AMD Instinct (Hygon uses same protocol)
- [Hygon DCU Dual-Chip Roadmap — Digitimes](https://www.digitimes.com/news/a20251223VL208/ai-chip-china-cpu.html) — System-level AI architecture combining Hygon CPU + DCU; scale-up strategy

### Scale-out Interconnect
- [DTK RCCL — developer.sourcefind.cn](https://developer.sourcefind.cn/dtk) — RCCL collective communication for multi-node scale-out over RoCE/Ethernet
- [GPUStack DCU Inference Tutorial](https://docs.gpustack.ai/0.5/tutorials/running-inference-with-hygon-dcus/) — Multi-node deployment patterns for DCU clusters
- [Hygon DCU Rising Adoption — Digitimes](https://www.digitimes.com/news/a20251118VL204/revenue-it-profit-growth-merger.html) — Hygon DCU adoption in China AI compute infrastructure

---

## Other Resources

### Academic Papers
- [Opt4GPTQ: 4-bit GPTQ Inference on Heterogeneous Platforms (arxiv)](https://arxiv.org/html/2511.19438v2) — Quantized LLM inference optimization including Hygon DCU
- [Optimizing Depthwise Separable Conv on DCU — CCF Springer 2024](https://link.springer.com/article/10.1007/s42514-024-00200-3) — Detailed architecture micro-benchmarking
- [Optimizing Sparse GEMM for DCUs — Journal of Supercomputing 2024](https://link.springer.com/article/10.1007/s11227-024-06234-2) — Sparse GEMM tuning on DCU hardware

### Company / Business
- [Hygon Information Technology — Wikipedia](https://en.wikipedia.org/wiki/Hygon_Information_Technology) — Corporate history, AMD JV, Entity List, IPO
- [AMD–Chinese Joint Venture — Wikipedia](https://en.wikipedia.org/wiki/AMD%E2%80%93Chinese_joint_venture) — Detailed account of Zen license agreement, JV structure, $293M license fee
- [Silicon Vanguard: China Chip Leaders — Machine Yearning](https://www.machineyearning.io/p/chinas-silicon-vanguard) — Industry analysis placing Hygon in China AI chip landscape
- [Hygon Dhyana CPU Deep Dive — AnandTech](https://www.anandtech.com/show/15493/hygon-dhyana-reviewed-chinese-x86-cpus-amd) — Technical review of Zen-derived Dhyana CPU
- [China Zen x86 Processor Production — Tom's Hardware](https://www.tomshardware.com/news/china-zen-x86-processor-dryhana,37417.html) — Production announcement

### Chinese Community Resources
- [中科海光 CPU+DCU 产品介绍 — 知乎](https://zhuanlan.zhihu.com/p/693079965) — Comprehensive overview of Hygon CPU+DCU product lines
- [2024年海光研究报告：CPU+DCU双轮驱动 — 未来智库](https://www.vzkoo.com/read/2024082266eda1c2a51eaf56d9404032.html) — Securities research report: Hygon AI compute strategy
- [海光DCU部署全攻略 — CSDN (大模型工程局)](https://blog.csdn.net/liu1983robin/article/details/144493349) — Full DCU deployment guide: unboxing, configuration, AI training
- [GitHub topics: dcu](https://github.com/topics/dcu) — GitHub repos tagged with dcu keyword

---

## Added 2026-08-08 — 深算三号 (BW1000), 深算四号, Sugon merger termination

### Vendor / first-party

- [海光信息 — 加速器产品页](https://www.hygon.cn/product/accelerator) — **Negative result of record**: Hygon's own accelerator page publishes **no DCU model names and no specifications**. This is the reason no BW1000 spec is publishable.
- [海光信息投资者互动回复 (2025-12-11) — 新浪财经转载](https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml) — Only first-party status statement: "深算三号已经投入市场，受到客户认可"; 深算四号研发"进展顺利". **Primary status source for both parts.**

### Deployment evidence (深算三号 BW1000)

- [中科南京信息高铁研究院 — 深算三号 BW1000 国内首发](https://www.ictnj.ac.cn/newsinfo/10874852.html) — CAS-affiliated institute announces first domestic deployment on the 信息高铁智算算力网 AI development platform, Dec 2025. **Best available deployment evidence.**
- [南京麒麟科创园政务新闻 (2025-12-22)](https://qilinpark.nanjing.gov.cn/xwzx/202512/t20251222_5748374.html) — Government newsroom echo of the same deployment.
- [IT之家 相关报道](https://www.ithome.com/0/977/747.htm) — Chinese tech-media coverage.

### Sugon merger termination (2025-12-09) — independently confirmed

- [财新 — 海光信息终止吸收合并中科曙光](https://www.caixin.com/2025-12-09/102391669.html)
- [第一财经 — 终止换股吸收合并](https://www.yicai.com/news/102948899.html)
- [界面新闻 — 海光/曙光合并终止](https://www.jiemian.com/article/13741594.html)
- [东方财富 — 终止重组公告 (2025-12-09)](https://finance.eastmoney.com/a/202512093586757039.html)
- [东方财富 — 终止后续 (2025-12-10)](https://finance.eastmoney.com/a/202512103587902138.html)
- [观察者网 — 终止合并 (2025-12-10)](https://www.guancha.cn/economy/2025_12_10_799933.shtml)
- [同花顺 — 终止吸收合并 (2025-12-09)](https://news.10jqka.com.cn/20251209/c673085520.shtml)

### Financials

- [第一财经 — 海光信息业绩相关报道](https://www.yicai.com/news/102952855.html) — H1 2026 业绩预告 context (disclosed 2026-07-16): revenue RMB 8.5–9.3 B, +55.56%–70.20% YoY; net profit RMB 1.70–1.83 B, +41.50%–52.32% YoY. Consolidated CPU+DCU; no DCU breakout.

### Ecosystem / software

- [深算三号 — 百度百科](https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890) — Source of the "Tencent Hunyuan Hy3 adaptation completed May 2026" claim. **Encyclopedia-grade only**; low-medium confidence.

### ⚠️ Rejected sources — retail forums, contradictory, no primary basis

Recorded so future passes recognise and skip them rather than re-discovering and adopting them.

- [Eastmoney 股吧 BW1000 spec post](https://guba.eastmoney.com/news,688041,1483072345.html) — origin of the FP32 49 / TF32 96 / BF16-FP16 192 TFLOPS / INT8 392 TOPS figure set. Retail stock message board. **Do not cite as a spec source.**
- [知乎 BW1000 长文](https://zhuanlan.zhihu.com/p/2066563534897652881) — states ~20 TFLOPS FP32, contradicting the above.
- [集思录 讨论](https://www.jisilu.cn/question/510441) — retail investor discussion.
- [雪球 note](https://xueqiu.com/2453283973/400298837) — origin of "mass production Q3 2025", "10k→30k wafers/month", "R&D began mid-2022", "tape-out 2024", "明年回片". Channel-check rumor.

### Negative results (searched 2026-08-08, not found)

- No Hygon/DCU **MLPerf** Training or Inference submission.
- No Hygon paper at **Hot Chips 2026**, **ISCA 2026**, or **ISSCC 2026**. *(Note: Hot Chips 38 runs 2026-08-23/25 — 15 days after this pass; its program is public but no content is.)*
- No **DTK 26.x** release; `developer.sourcefind.cn` unreachable during verification. Most recent versions located: DTK 25.04, DTK 25.10.
- No vendor block diagram, die shot, or microarchitecture disclosure for BW1000.
