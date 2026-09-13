# Enflame Technology (燧原科技) Software Stack & Hardware Resources

*as_of: 2026-08-08 (baseline search 2026-04-05; see "Resources Added 2026-08-08" at the end)*
*device_class: Data Transfer Unit (China)*
*seeds: https://github.com/EnflameTechnology, https://support.enflame-tech.com, https://www.enflame-tech.com*

## Company Background

Enflame Technology (燧原科技; Shanghai Enflame Technology Co., Ltd.) was founded in March 2018 by two former AMD employees. Headquartered in Shanghai, it develops full-stack AI compute solutions covering chips, accelerator cards, smart-computing clusters, and software platforms. Major investors include Tencent (20.26% equity), Meitu, Primavera Capital, CITIC PE, and the China Integrated Circuit Industry Investment Fund. As of January 2026 its STAR Market IPO application (RMB 6 billion raise) was accepted by the Shanghai Stock Exchange; it passed the listing committee 2026-06-15 and CSRC registration became effective 2026-07-09, but **shares had not begun trading as of 2026-08-08**.

### Product Generations

| Generation | Chip Code | Card | Process | Key Specs |
|---|---|---|---|---|
| DTU 1.0 (GCU 1.0) | 邃思1.0 | CloudBlazer T10 | GF 12LP FinFET | 14B transistors, 32 SIPs / 4 SICs, HBM2, 2.5D MCM |
| DTU 2.0 (GCU 2.0) | 邃思2.0 | CloudBlazer T20 / i20 | 12 nm | 40 TFLOPS FP32 / 160 TF32 / 320 INT8 TOPS, 64 GB HBM2e, 1.8 TB/s, 57.5×57.5 mm 9-die MCM |
| DTU 3.0 (GCU 3.0) | 邃思3.0 (SCORPIO) | CloudBlazer S60 | TSMC N6NTO-HPC | ~256 TFLOPS FP16 (est.), HBM2e, TSMC N6, 70 k units shipped by 2024 |
| DTU 4.0 (GCU 4.0) | 邃思4.0 | CloudBlazer L600 | TSMC 5 nm (rumored) | 144 GB HBM3, 3.6 TB/s, 800 GB/s interconnect, native FP8, unveiled WAIC 2025-07-27. Status per June 2026 IPO filing: silicon returned, early commercialization — **not** mass production |
| DTU 5.0 / DTU 6.0 | — | — | — | IPO use-of-proceeds line items only (RMB 1.503 B / 1.197 B); no tape-out, silicon, or specification announced as of 2026-08-08 |

---

## Software Stack

### Framework Integration
- [EnflameTechnology/vllm-gcu](https://github.com/EnflameTechnology/vllm-gcu) — vLLM fork optimised for Enflame GCU (S60); OpenAI-compatible API server; operator-level GCU optimisations
- [EnflameTechnology/candle-gcu](https://github.com/EnflameTechnology/candle-gcu) — Candle ML framework (Rust) ported to Enflame GCU
- [EnflameTechnology/candle-vllm-gcu](https://github.com/EnflameTechnology/candle-vllm-gcu) — Candle-based vLLM serving on GCU
- [Running llama2-13b on Enflame S60 with PaddleNLP](https://paddlenlp.readthedocs.io/en/latest/llm/devices/gcu/llama/README.html) — PaddlePaddle / PaddleNLP integration guide for GCU
- [Qwen2 GCU support PR](https://github.com/QwenLM/Qwen2/pull/456) — GCU backend added to Qwen2 model repo
- TopsTorch (proprietary, part of TopsPlatform) — PyTorch plug-in; registers `torch.device("gcu")`; dispatches ATen operations to TopsBlasOps; enables standard PyTorch code to run on GCU without rewrites
- [TopsRider overview — StockCounterparts](https://www.stockcounterparts.com/companies/enflame-technology) — describes TopsRider as Enflame's full-stack software platform (driver + compiler + op libs + toolchain)

### Compiler / IR
- [AI Compute Chip from Enflame — IEEE Xplore HC33](https://ieeexplore.ieee.org/document/9567224) — primary source; covers GCU-CARE 1.0 static dataflow compilation model; confirms compile-time graph scheduling with no runtime thread dispatch
- [Enflame DTU 1.0 at Hot Chips 33 — ServeTheHome](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/) — secondary coverage of GCU-CARE architectural overview and static data-flow graph compilation
- [TopsCC (proprietary, no direct public doc)](https://support.enflame-tech.com) — Enflame's graph compiler; part of TopsPlatform; lowers PyTorch/ONNX/PaddlePaddle graphs to GCU-CARE fabric; handles memory tiling and sparsity annotation

### Op Library
- TopsBlasOps (proprietary) — GEMM and BLAS primitives for GCU; dispatched by TopsTorch for `torch.matmul`, convolutions, norms; part of TopsPlatform
- [HAMi enflame-gcu-support](https://github.com/Project-HAMi/HAMi/blob/master/docs/enflame-gcu-support.md) — Kubernetes GCU sharing; documents TopsBlasOps + TopsRuntime dependency stack

### Kernel Library
- TopsPlatform Kernel Library (proprietary, not open-sourced) — GCU-CARE native binary kernels; bundled with TopsCC SDK; analogous to CUTLASS (NVIDIA) or mlu-ops (Cambricon); no public repository identified
- [EnflameTechnology/FFmpeg-GCU](https://github.com/EnflameTechnology/FFmpeg-GCU) — hardware video encode/decode (topscodec) as FFmpeg plugin; demonstrates GCU-CARE kernel capability for media compute

### Runtime
- [TopsRuntime (proprietary)](https://support.enflame-tech.com) — analogous to CUDA Runtime API; device memory management, stream/event model, kernel launch; installed from `topsruntime_*.deb`
- [GCU Monitor Examples — 燧原软件栈文档中心](https://support.enflame-tech.com/onlinedoc_dev_2.5.115/6-k8s/k8s/GCU_monitor_examples/content/source/index.html) — runtime observability and profiling tools for GCU

### Driver / Firmware
- TopsDrv (proprietary kernel module) — Linux kernel driver for GCU; analogous to nvidia.ko; installed alongside TopsRuntime
- [T20 Product Manual — 燧原软件栈文档中心](https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/product_manual/content/source/T20_product_manual.html) — official hardware + driver documentation for T20 card
- [T10 Product Manual — 燧原硬件文档中心](https://support.enflame-tech.com/onlinedoc_hw/3-t1x/t10/product_manual/content/source/T10_product_manual.html) — official hardware + driver documentation for T10 card

### Communication
- [ECCL (Enflame Collective Communication Library)](https://github.com/EnflameTechnology) — analogous to NCCL; AllReduce, AllGather for multi-GCU training; installed from `eccl_3.5*.deb`; operates over GCU-LARE for intra-node and Ethernet/RoCE for inter-node

### Assembler / ISA
*Note: No public ISA documentation exists. GCU-CARE and GCU-DARE are fully proprietary and not externally documented. The HC33 sources below are indirect architectural evidence only — not an assembler or ISA reference. TopsCC handles all lowering; there is no user-accessible ptxas-equivalent.*
- [AI Compute Chip from Enflame — IEEE HC33 paper](https://ieeexplore.ieee.org/document/9567224) — indirect architectural evidence of GCU-CARE 1.0 static dataflow design; confirms no runtime thread dispatch
- [HC33 GCU-DARE slides — ServeTheHome](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/hc33-2021-enflame-ai-compute-chip-gcu-dare-1-0/) — indirect evidence of GCU Data-path Architecture Reconfigurable Engine; on-chip data movement design

---

## Hardware Architecture

### Compute Engine
- [AI Compute Chip from Enflame — IEEE HC33 paper](https://ieeexplore.ieee.org/document/9567224) — primary source; describes GCU-CARE 1.0: 32 SIPs / 4 SICs; 256 kernels in 32 groups; reconfigurable compute architecture
- [Enflame DTU 1.0 at Hot Chips 33 — ServeTheHome](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/) — illustrated overview; 32 SIPs grouped into 4 SICs; Tensor ALUs + data-transfer engines; sparsity exploitation
- [智东西: 邃思2.0 architecture (Chinese)](https://zhidx.com/p/281331.html) — GCU-CARA (全域计算架构) for DTU 2.0; upgraded spatial compute fabric; largest Chinese AI chip at announcement (57.5×57.5 mm, 9-die MCM)
- [TechInsights: Enflame S60 Floorplan Analysis](https://www.techinsights.com/blog/enflame-s60-ai-accelerator-processor-floorplan-analysis) — SCORPIO-AO compute chiplet, TSMC N6NTO-HPC; die-level reverse engineering

### Data Path
- [HC33 GCU-DARE slides](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/hc33-2021-enflame-ai-compute-chip-gcu-dare-1-0/) — Data-path Architecture Reconfigurable Engine; DMA engines and local scratchpad feeding compute SIPs
- [GF press release 2019](https://gf.com/gf-press-release/enflame-technology-announces-cloudblazer-dtu-chip-globalfoundries-12lp-finfet/) — on-chip configuration algorithm for fast data routing; HBM2 via 2.5D interposer

### On-chip Memory
- [T20 Product Manual](https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/product_manual/content/source/T20_product_manual.html) — per-SIP local SRAM scratchpad (NRAM/WRAM analogues); compiler-managed
- [HC33 paper — IEEE](https://ieeexplore.ieee.org/document/9567224) — on-chip SRAM hierarchy for DTU 1.0; weight and activation buffering

### Off-chip Memory
- [GF press release](https://gf.com/gf-press-release/enflame-technology-announces-cloudblazer-dtu-chip-globalfoundries-12lp-finfet/) — DTU 1.0: HBM2 via 2.5D packaging
- [云燧T20 百度百科](https://baike.baidu.com/item/%E4%BA%91%E7%87%A7T20/59243461) — T20: 4×HBM2e, 64 GB, 1.8 TB/s bandwidth
- [TrendForce: L600 announcement](https://www.trendforce.com/news/2025/07/29/news-chinese-ai-chip-unicorns-enflame-metax-unveil-next-gen-chips-shortly-after-nvidias-h20-return/) — L600: 144 GB HBM3, 3.6 TB/s; unveiled July 2025

### Host Interface / Package
- [GF press release](https://gf.com/gf-press-release/enflame-technology-announces-cloudblazer-dtu-chip-globalfoundries-12lp-finfet/) — PCIe 4.0 host interface; 2.5D CoWoS-like advanced packaging (GF 12LP + HBM2 interposer)
- [TechInsights S60 Floorplan](https://www.techinsights.com/blog/enflame-s60-ai-accelerator-processor-floorplan-analysis) — SCORPIO-AO TSMC N6 compute chiplet + separate memory chiplets; 2.5D MCM package

### Scale-up Interconnect
- [HC33 GCU-LARE slides](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/hc33-2021-enflame-ai-compute-chip-gcu-lare-1-0-2/) — GCU-LARE (Local Area Reconfigurable Engine) 1.0: chip-to-chip link; non-cache-coherent; up to 4 GPUs direct, scalable to 8
- [GCU-LARE Bridge Card Manual](https://support.enflame-tech.com/onlinedoc_hw/3-t1x/t10/link_card/content/source/index.html) — hardware bridge card for 4-card full-mesh within single server
- [智东西: 邃思2.0](https://zhidx.com/p/281331.html) — GCU-LARE 2.0: 300 GB/s bidirectional bandwidth; scales to thousands of accelerator cards

### Scale-out Interconnect
- [ECCL GitHub](https://github.com/EnflameTechnology) — ECCL collective communications over Ethernet/RoCE for multi-node training
- [vllm-gcu README](https://github.com/EnflameTechnology/vllm-gcu/blob/main/ReadMe-EN.md) — multi-node inference mentions standard Ethernet scale-out
- [Enflame IPO prospectus context](https://pandaily.com/enflame-s-star-market-ipo-accepted-signaling-strong-momentum-in-china-s-ai-chip-sector) — SmartCluster product line uses RoCE-based scale-out networking

---

## Other Resources

- [Enflame Wikipedia](https://en.wikipedia.org/wiki/Enflame) — company history, founding team (ex-AMD), investors
- [Enflame official website](https://www.enflame-tech.com/) — product pages for T10/T20/S60/L600 cards and SmartCluster systems
- [NBCNews: TSMC Enflame S60 export control scrutiny](https://www.nbcnews.com/tech/tech-news/ai-chip-tsmc-enflame-techinsights-rcna259342) — regulatory context; S60 TSMC N6 potentially within U.S. export restriction thresholds
- [TrendForce: Enflame + MetaX next-gen chips July 2025](https://www.trendforce.com/news/2025/07/29/news-chinese-ai-chip-unicorns-enflame-metax-unveil-next-gen-chips-shortly-after-nvidias-h20-return/) — competitive landscape vs H20 post-export-restriction era
- [Enflame four-little-dragons analysis — Medium](https://evergreenllc2020.medium.com/chinas-gpu-insurgents-the-four-little-dragons-challenging-nvidia-s-dominance-a00bfea221d7) — market positioning of Enflame alongside Biren, MetaX, Moore Threads
- [Oceanpine Capital: i20 announcement](https://oceanpine.com/oceanpine-news/20220118451.html) — i20 inference card: 32 TFLOPS FP32 / 128 TF32 / 256 INT8 TOPS, 16 GB HBM2e, 819 GB/s
- [Jon Peddie Research: Enflame chip profile](https://www.jonpeddie.com/techwatch/enflame-suiyuan-ai-processor-chip/) — independent analyst coverage
- [HAMi Kubernetes GCU sharing](https://github.com/Project-HAMi/HAMi/blob/master/docs/enflame-gcu-support.md) — Kubernetes device plugin for GCU virtualisation
- [SCMP: Enflame IPO analysis](https://www.scmp.com/tech/article/3343336/will-chinese-ai-chip-designer-enflames-shanghai-ipo-be-another-blockbuster) — financial context; Tencent captive-supplier dynamics

---

# Resources Added 2026-08-08

*window: 2026-04-05 → 2026-08-08. Verified in round-3 adversarial verification.*

## Software Stack — new

### Framework Integration
- [EnflameTechnology/torch-gcu](https://github.com/EnflameTechnology/torch-gcu) — PyTorch backend **source published 2026-06-23**, BSD-style licence. PrivateUse1 dispatch key; broad ATen coverage with CPU fallback; ECCL as a `torch.distributed` backend; `torch.compile`/Inductor; `torch.gcu.amp`; `torch.gcu.GCUGraph`; `ProfilerActivity.GCU`; `transfer_to_gcu` CUDA-migration shim; `libtorch_gcu` C++ inference (single-device only). Version matrix torch_gcu 2.10.0 / PyTorch 2.10.0 / TopsRider v3.7.1. **Hardware Support table lists only the CloudBlazer S60.** Single-commit code drop — no follow-on pushes, no press coverage
- [torch-gcu README (raw)](https://raw.githubusercontent.com/EnflameTechnology/torch-gcu/main/README.md) — primary source for all of the above
- [torch-gcu LICENSE (raw)](https://raw.githubusercontent.com/EnflameTechnology/torch-gcu/main/LICENSE) — BSD-style, Copyright (c) 2026 Enflame 燧原科技
- [EnflameTechnology/FlagOS](https://github.com/EnflameTechnology/FlagOS) — FlagOS support branch, repo created 2026-05-29
- [PaddlePaddle FastDeploy — Enflame GCU installation](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/Enflame_gcu.md) — S60 + ERNIE 4.5 under a unified GCU/GPU inference interface; pins driver 1.5.0.5 / TopsRider 3.4.623. **Pre-baseline** — dates to the 2025 ERNIE 4.5 release

### Runtime / SDK
- [vllm-gcu releases API](https://api.github.com/repos/EnflameTechnology/vllm-gcu/releases) — **v0.11.0 published 2026-05-12**; prior v0.9.2 (2025-12-23), v0.8.0 (2025-09-17). Release bodies cover DeepSeek 3.2 runtime work, layer-wise KV-cache transfer, FP8 linear/MoE, W8A8-INT8 MoE, GCU Docker builds, and broad model enablement
- [EnflameTechnology org repos API](https://api.github.com/orgs/EnflameTechnology/repos) — authoritative `created_at` / `pushed_at` timestamps for all repos below
- ⚠️ `https://support.enflame-tech.com/onlinedoc_dev_3.6/`, `_3.7/`, `_2.5.115/` — **all return Tencent WAF HTTP 403**. No documentation-site version slug could be validated. The verifiable SDK version comes from torch-gcu instead: TopsRider **3.7.x**

### Communication (weak signal — fork creation only)
- [EnflameTechnology/nixl](https://github.com/EnflameTechnology/nixl) and [EnflameTechnology/ucx](https://github.com/EnflameTechnology/ucx) — forks created 2026-06-08, `created_at == pushed_at`, no visible Enflame commits. The NIXL+UCX pair is the standard stack for prefill/decode disaggregation and KV-cache transfer, so the direction is plausible, **but this is not evidence of a shipped disaggregation capability**
- [EnflameTechnology/findtops](https://github.com/EnflameTechnology/findtops) — repo created 2026-07-24; contents not investigated

## Hardware Architecture — new

### Scale-up Interconnect / Systems
- [C114: Enflame + ZTE 云燧 ESL64-O launch (2026-07-18)](https://www.c114.net.cn/industry/101611.html) — OEX 正交无背板 orthogonal backplane-free, "0 线缆" zero-cable interconnect; also covers the CoPoS sample and the 2026-07-09 IPO registration
- [IT之家: ESL64-O launch detail (2026-07-18)](https://www.ithome.com/0/978/716.htm) — confirms **no card count, no bandwidth, no chip model disclosed**
- [Sina Finance: ESL64-O vs ESL64-C positioning (2026-07-21)](https://finance.sina.com.cn/tech/shenji/2026-07-21/doc-iniipxxe9278635.shtml) — also the NPO optical prototype supporting 512+ accelerator-card supernodes
- [Sohu: ESL64-C Cable-tray design, 万卡级以上 cluster networking](https://www.sohu.com/a/1051829208_313745) — the 10,000+ card claim attaches to **ESL64-C**, not ESL64-O
- [ZOL: NPO optical prototype](https://ai.zol.com.cn/1219/12193958.html) — stable support claimed for 512+ card supernode architectures

### Packaging
- [Sina Finance: CoPoS glass-substrate panel-level sample with 先封科技 (2026-07-18)](https://finance.sina.com.cn/stock/t/2026-07-18/doc-iniifanf8343506.shtml) — 国内首款面向AI算力芯片的玻璃基板CoPoS先进封装样品; explicitly a **sample**, 不等同于台积电最终量产规格

### L600 spec backfill (pre-baseline sources)
- [腾讯新闻: L600 at WAIC (2025-07-27)](https://news.qq.com/rain/a/20250727A06S1L00) — 144 GB / 3.6 TB/s / **800 GB/s interconnect**, native FP8
- [DRAMeXchange: L600 coverage (2025-07-28)](https://www.dramx.com/News/IC/20250728-38864.html) — same figures
- [百度百科 燧原L600](https://baike.baidu.com/item/%E7%87%A7%E5%8E%9FL600/67178547) — 4th-gen train+inference chip, WAIC 2025-07-27

## Market / Corporate Context — new

- [DRAMeXchange: IPO use-of-proceeds breakdown (2026-06-16)](https://www.dramx.com/News/IC/20260616-40620.html) — RMB 15.03亿 5th-gen chips, 11.97亿 6th-gen, 33亿 AI hardware/software co-innovation (i.e. only 45% to the chip-generation projects)
- [Sina Finance: listing committee pass (2026-06-15)](https://finance.sina.com.cn/wm/2026-06-15/doc-inicnuhr8205604.shtml) — RMB 6.0 B raise
- [同花顺 10jqka (2026-07-10)](https://stock.10jqka.com.cn) — CSRC published approval of Enflame's IPO registration; share range 4,303.5173万–6,834.9980万股
- [Sohu: prospectus wording on L600](https://www.sohu.com/a/1037503510_237556) — 随着…L600规模化量产及超节点系统的交付 is **forward-looking**; training products = 1.15% of AI-accelerator-card revenue in 2025
- [WAIC 2026 official site](https://www.worldaic.com.cn/) and [Shanghai gov WAIC 2026 page](https://english.shanghai.gov.cn/en-WAIC2026/index.html) — event held 2026-07-17 to 07-20

## Searched and not found

- No DTU 5.0 tape-out or silicon announcement
- No MLPerf submission for S60 or L600
- No Enflame paper or talk at ISCA 2026 or any other venue in-window. **Hot Chips 38 runs 2026-08-23/25 — after this as_of date; nothing from its program is cited**
- No confirmed IPO trading debut (no ticker, price, or subscription date as of 2026-08-08)
- SIP/SIC counts for DTU 3.0 and DTU 4.0 still undisclosed
- ESL64 card count, per-card bandwidth, and host silicon still undisclosed
- No verifiable documentation-site version slug (WAF 403)
