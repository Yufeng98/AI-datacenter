# Huawei Ascend Layer Mapping Table

*as_of: 2026-08-08*

Rows marked **(2026-08)** were added or revised in the 2026-08-08 Ascend 950 update. Confidence values follow the repo convention: `confirmed` = independently corroborated; `inferred` = derived from vendor roadmap statements; `single-source` = one third-party analysis only, not to be cited as spec; `roadmap` = announced future product.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | MindSpore (nn.Cell, GRAPH_MODE/PYNATIVE_MODE, S2S AD, auto-parallelism) | confirmed | mindspore-overview, mindspore-whitepaper, mindspore-gitee |
| Framework Integration | MindFormers (transformer model library: BERT, GPT, LLaMA, Qwen on Ascend) | confirmed | mindspore-gitee, mindspore-models |
| Framework Integration | torch_npu / Ascend PyTorch Adapter (PrivateUse1 backend, torch.npu device) | confirmed | ascend-pytorch-github, pytorch-torchtune-blog |
| Framework Integration | torch-neuronx equivalence: torch_npu + FSDP/DTensor/torch.compile | confirmed | ascend-pytorch-github |
| Framework Integration | ONNX Runtime CANN Execution Provider (C/C++ + Python) | confirmed | onnxruntime-cann-ep |
| Framework Integration | OpenCV CANN Backend (vision preprocessing via DVPP) | confirmed | opencv-cann-wiki |
| Framework Integration | **(2026-08)** MindSpore 2.9.0 (2026-05-07): paired with CANN 9.0.0, native Ascend 950PR support; canonical on mindspore.cn + PyPI | confirmed | mindspore-release-notes, mindspore-2-9-notes, mindspore-pypi |
| Framework Integration | **(2026-08)** vllm-ascend: Ascend serving backend; MXFP4 quant + DeepSeek-V4 (v0.20.2rc1), full E2E DeepSeek-V4 on Ascend 950 (v0.21.0rc1), W4A16/all-gather-EP MXFP4 (v0.23.0rc1, 2026-07-19, needs CANN 9.0.1) | confirmed | vllm-ascend-relnotes, vllm-ascend-releases |
| Compiler / IR | ATC (Ascend Tensor Compiler): ONNX/TF/Caffe/MindIR → .om offline model | confirmed | atc-doc, atc-model-conversion |
| Compiler / IR | MindSpore Graph Compiler + GE (Graph Engine): MindIR → CANN operator dispatch | confirmed | mindspore-overview, mindspore-whitepaper |
| Compiler / IR | TileLang-Ascend (MLIR tiled kernel DSL for Da Vinci) | confirmed | tilelang-ascend-github |
| Op Library | CANN Standard Operator Library / Ascend OL (1000+ ops: Conv2D, MatMul, BN, Attention) | confirmed | cann-doc, huawei-cann-release |
| Op Library | **(2026-08)** CANN 9.0.0 (2026-05-09) adds FP8 / MXFP8 / MXFP4 data types and first-class Ascend 950PR (Atlas 350) support — release notes verbatim 新增适配Ascend 950PR | confirmed | cann-900-relnotes |
| Op Library | **(2026-08)** CATLASS template library upgraded for 950-series tensor/vector co-compute paths and new Fixpipe move modes | confirmed | cann-900-relnotes |
| Op Library | **(2026-08)** CANN open-sourced as a ~76-repo GitCode organization (ops-transformer/ops-math/ops-nn/ops-cv, hccl/hcomm/shmem, asc-devkit, pypto/pyasc, runtime, ge, graph-autofusion, cann-recipes-*, cann-bench, amct); active commits through early Aug 2026 | confirmed | gitcode-cann |
| Op Library | CANN 950 operator extensions: FP8, MXFP8, HiF8, MXFP4 quantization pass in ATC + TBE | inferred | trendforce-950pr, huaweicentral-roadmap |
| Op Library | AIPP (AI Pre-Processing: YUV/NV12→RGB, normalize, crop, resize baked into .om) | confirmed | atc-doc |
| Kernel Library | TBE-DSL (Tensor Boost Engine DSL: high-level operator authoring, auto-scheduling) | confirmed | tbe-dsl-doc, huawei-cann-release |
| Kernel Library | TBE-TIK (Tensor Iterator Kernel: low-level Python DSL, explicit L1/UB/Cube/Vector control) | confirmed | tbe-tik-doc |
| Kernel Library | AscendC (C/C++-compatible kernel language: GlobalTensor, DataCopy, Matmul, EnQue/DeQue pipeline) | confirmed | ascendc-launch, parallel-scan-arxiv |
| Kernel Library | AscendC FP8/HiF8/MXFP4 type extensions (950 series new precision kernel authoring) | inferred | trendforce-950pr |
| Kernel Library | **(2026-08)** AscendC SIMD+SIMT hybrid programming shipped in CANN 9.0.0: ~700 new SIMT API entry points (warp, atomic, math, type-conversion) | confirmed | cann-900-relnotes |
| Kernel Library | **(2026-08)** AscendC Reg-based (register-level) programming path, exposed specifically on Ascend 950PR | confirmed | cann-900-relnotes |
| Kernel Library | **(2026-08)** BufferID-based synchronization replacing the EnQue/DeQue pipeline idiom on Da Vinci 4.0 — would change the core AscendC idiom if confirmed | single-source | ascend950-analysis |
| Kernel Library | Huawei-Ascend/samples (official AscendCL sample kernels and models) | confirmed | ascend-samples-github |
| Runtime | AscendCL (C runtime: aclrtSetDevice, aclrtMalloc, aclrtCreateStream, aclmdlExecute, aclopExecuteV2) | confirmed | ascendcl-doc |
| Runtime | DVPP (Digital Vision Pre-Processor hardware unit for video/image decode) | confirmed | ascendcl-doc, opencv-cann-wiki |
| Runtime | **(2026-08)** CANN packaging restructured: apt + pip installs added (conda/yum/apt/pip) with unified download bundle; Huawei markets ~2 h → ~45 min deployment (vendor claim) | confirmed | cann-900-relnotes |
| Driver / Firmware | Ascend NPU Kernel Driver (PCIe BAR, command queues, IRQ; **still closed** — the 2026 CANN open-sourcing covered userspace layers only) | confirmed | cann-installer-github, gitcode-cann |
| Driver / Firmware | Ascend Firmware (closed; TEE integration; on-chip resource management) | inferred | ascend-cc-arxiv |
| Communication | HCCL (Huawei Collective Comms Lib: AllReduce/AllGather/ReduceScatter; HCCS+RoCEv2 transport) | confirmed | atlas900-cluster, mindspore-distributed |
| Communication | HCCS (Huawei Compute Communication System: proprietary intra-server NPU-to-NPU bus) | confirmed | hccs-cce-doc, atlas900-cluster |
| Communication | CloudMatrix 384 optical fabric (6912 × 400G SiPh LPO, 2.8 Tbps/chip scale-up) | confirmed | semianalysis-cloudmatrix, qsfptek-cloudmatrix |
| Communication | HCCL updated for 950 series: 2 TB/s scale-up interconnect and Atlas 950 SuperPoD (8,192-chip) topology | inferred | trendforce-950pr, huaweicentral-roadmap |
| Communication | **(2026-08)** HCCL batched-collective merge via HcclGroupStart()/HcclGroupEnd(); collective acceleration through the on-chip CCU (集合通信支持CCU通信加速) | confirmed | cann-900-relnotes |
| Communication | **(2026-08)** UnifiedBus / 灵衢 (Lingqu) 2.0 replaces HCCS branding for the 950 scale-up fabric; published as an open specification (base + firmware + SW reference designs); launched 2025-09-18 at Huawei Connect 2025 | confirmed | unifiedbus-spec, openeuler-ub-core |
| Communication | **(2026-08)** UnifiedBus cluster scale beyond 128K cards (supersedes the earlier "HCCS 4.0 → 100,000-chip" figure) | confirmed | ascend950-whitepaper, unifiedbus-spec |
| Assembler / ISA | CCE ISA (Core Compute Engine: 5 parallel streams — Cube, Vector, Scalar, MTE1, MTE2) | confirmed | davinci-paper-cmc, davinci-paper-semantic |
| Assembler / ISA | AscendC / TBE-TIK ISA intrinsics (software-visible ISA abstraction layer) | confirmed | ascendc-launch, tbe-tik-doc |
| Assembler / ISA | CCE binary (.om format: no public virtual ISA equivalent to PTX) | confirmed | atc-doc |
| Assembler / ISA | HiF8 / MXFP4 Cube instructions (Da Vinci 4.0 new CCE opcodes for 950 series) | inferred | trendforce-950pr, huaweicentral-roadmap |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Da Vinci Cube Unit: 16×16×16 FP16 systolic (4,096 MACs/cycle); 8,192 INT8 MACs/cycle | confirmed | davinci-paper-cmc, oreilly-ch3 |
| Compute Engine | Da Vinci Vector Unit: 128-lane FP16 or 256-lane INT8 per cycle; FP32/BF16/INT4 support | confirmed | davinci-paper-cmc, oreilly-ch3 |
| Compute Engine | Ascend 910: 32 DaVinci Max cores, 256 TFLOPS FP16, 512 TOPS INT8, TSMC 7nm | confirmed | davinci-paper-cmc, ascend910-specs |
| Compute Engine | Ascend 910B: 25 New DaVinci cores, 320 TFLOPS FP16, SMIC N+1 | confirmed | tomshardware-910b |
| Compute Engine | Ascend 910C: ~64 Da Vinci 3.0 cores (dual-die), ~800 TFLOPS FP16, ~1600 TOPS INT8 | confirmed | awesomeagents-910c, nexgen-910c |
| Compute Engine | Ascend 950PR (Da Vinci 4.0): FP8/HiF8 1 PFLOPS, MXFP4 2 PFLOPS; SIMD+SIMT hybrid. **Process node: not disclosed** (SMIC N+2/N+3 is analyst inference only) | confirmed | trendforce-950pr, huaweicentral-roadmap |
| Compute Engine | Ascend 950DT (Da Vinci 4.0): FP8/HiF8 1 PFLOPS, MXFP4 2 PFLOPS; training + decode variant; **Q4 2026 official release**, Huawei Cloud deployment announced for Aug 2026 (not confirmed live as of 2026-08-08) | confirmed | trendforce-950dt, trendforce-950dt-pullin, huaweicentral-950dt |
| Compute Engine | **(2026-08)** Atlas 350 shipping card (950PR): **1.56 PFLOPS FP4, up to 112 GB, 1.4 TB/s, 600 W** — derated vs the 950PR chip spec; "2.8× H20" is a Huawei marketing comparison | confirmed | trendforce-atlas350, tomshardware-atlas350 |
| Compute Engine | **(2026-08)** Da Vinci 4.0 separates Cube and Vector into independent AIC / AIV cores, at 1 Cube + 2 Vector per AI subsystem (whitepaper calls this "3rd-generation Da Vinci") | confirmed | ascend950-whitepaper, ascend950-analysis |
| Compute Engine | **(2026-08)** 18 Da Vinci cores per AI die → 36 Cube + 72 Vector cores per chip (950DT); Linx816 on-chip AI CPU (4 clusters × 2 ARMv8-A, 4 MB L3/cluster) | single-source | ascend950-analysis |
| Compute Engine | **(2026-08)** Ascend 950 BF16/FP16 ~547 TFLOPS and TF32 ~273 TFLOPS — **not disclosed** by Huawei; single third-party estimate only | single-source | ascend950-analysis |
| Data Path | **(2026-08)** STARS 2.0 hardware scheduler arbitrating across AIC / AIV / CPU / DVPP / SDMA / UB / CCU | confirmed | ascend950-whitepaper, ascend950-analysis |
| Data Path | **(2026-08)** NDDMA lightweight AIC↔AIV on-chip data path (existence corroborated; the fuller "N-Dimensional Data Movement Accelerator with up to 5D fused transforms" description is single-source) | confirmed | ascend950-whitepaper, ascend950-analysis |
| Compute Engine | Ascend 960 (roadmap Q4 2027): 2× 950 compute; HiF4 precision (Huawei-proprietary 4-bit) | roadmap | tomshardware-roadmap, huaweicentral-roadmap |
| Compute Engine | Ascend 970 (roadmap 2028): cluster-level target 4 ZettaFLOPS FP4 | roadmap | tomshardware-roadmap |
| Data Path | Static 5-stream CCE pipeline (Cube/Vector/Scalar/MTE1/MTE2 concurrent, no dynamic OoO) | confirmed | davinci-paper-cmc, sciencedirect-perf |
| Data Path | Double-buffering via MTE1 prefetch (tile N+1 load overlaps tile N Cube compute) | confirmed | oreilly-ch3, davinci-paper-cmc |
| Data Path | L1→L0A/L0B→Cube→L0C→UB→MTE2→L2→HBM dataflow | confirmed | oreilly-ch3, davinci-paper-semantic |
| On-chip Memory | L0A/L0B: 16 KB each/core (Cube operand buffers) | confirmed | davinci-paper-cmc |
| On-chip Memory | L0C: 32 KB/core (FP32 accumulator for Cube output) | confirmed | davinci-paper-cmc |
| On-chip Memory | L1: 512 KB/core (input feature staging, fed by MTE1) | confirmed | davinci-paper-cmc |
| On-chip Memory | UB (Unified Buffer): ~256 KB/core (Vector/MTE2/Scalar shared scratchpad) | confirmed | davinci-paper-cmc, oreilly-ch3 |
| On-chip Memory | L2: 32 MB shared @ 4 TB/s NoC (1024-bit mesh @ 2 GHz); 84 MB total SRAM | confirmed | servethehome-910, davinci-paper-semantic |
| On-chip Memory | **(2026-08)** Ascend 950: **128 MB global L2** spanning both AI dies — a new level absent on 910B/910C; ~2× per-access improvement; per-way cache lock / residency policy. L2 bandwidth **not disclosed** | confirmed | ascend950-whitepaper, ascend950-analysis |
| On-chip Memory | **(2026-08)** Ascend 950 per-core buffers (L0A/L0B 64 KB, L1 512 KB, UB 512 KB; 512 B sector-cache lines split 4 × 128 B) — **not disclosed** by Huawei; the 16 KB L0A/L0B row above is the 910-era value and stays correct for that generation | single-source | ascend950-analysis |
| Off-chip Memory | HBM2 (Ascend 910): 32 GB, 1,228 GB/s (4 stacks), TSMC 7nm | confirmed | servethehome-910, davinci-paper-semantic |
| Off-chip Memory | HBM2e (Ascend 910B): 64 GB, ~800 GB/s (SMIC N+1 constrained) | confirmed | tomshardware-910b |
| Off-chip Memory | HBM2e (Ascend 910C): 128 GB, ~1,600 GB/s (dual-die additive) | confirmed | awesomeagents-910c |
| Off-chip Memory | HiBL 1.0 (Ascend 950PR): Huawei in-house 128 GB, 1.6 TB/s; export-control-independent; cost-optimized vs HBM3E | confirmed | trendforce-950pr, technetbook-atlas350 |
| Off-chip Memory | HiZQ 2.0 (Ascend 950DT): Huawei in-house 144 GB, 4 TB/s; 2.5× 910C bandwidth; HBM3E-class | confirmed | trendforce-950dt, huaweicentral-roadmap |
| Off-chip Memory | **(2026-08)** Atlas 350 card (950PR): **up to 112 GB @ 1.4 TB/s** — derated vs the 128 GB / 1.6 TB/s chip spec | confirmed | trendforce-atlas350, tomshardware-atlas350 |
| Host Interface / Package | PCIe 4.0 × 16 (~64 GB/s bidirectional) — Ascend 910 / 910B / 910C | confirmed | atlas-platform-doc |
| Host Interface / Package | **(2026-08)** Ascend 950: **PCIe 5.0 × 16** host interface; **dual 400 Gbps UBoE** (UnifiedBus over Ethernet); dual AI die per chip. Package construction and die area **not disclosed** | confirmed | ascend950-whitepaper, ascend950-analysis |
| Host Interface / Package | **(2026-08)** Atlas 350 card TDP: **600 W** (~1.5× H20) — the only disclosed 950-generation power figure | confirmed | trendforce-atlas350, tomshardware-atlas350 |
| Host Interface / Package | Ascend 910C MCM: 2× SMIC N+1 NPU dies + TSMC 7nm CPU companion + organic substrate | confirmed | semiwiki-teardown |
| Host Interface / Package | Atlas 800T server: 8× Ascend NPU via HCCS, OAM-compatible form factor | confirmed | atlas900-cluster, awesomeagents-910c |
| Scale-up Interconnect | HCCS: proprietary high-speed intra-server NPU bus (8 chips/server) | confirmed | hccs-cce-doc, atlas900-cluster |
| Scale-up Interconnect | CloudMatrix 384: 56 × 400G SiPh LPO/server, 2.8 Tbps/chip, full-mesh all-to-all optical | confirmed | semianalysis-cloudmatrix, qsfptek-cloudmatrix |
| Scale-up Interconnect | Ascend 950 scale-up interconnect: ~2 TB/s per chip over HiLink SerDes (2.5× vs 910C HCCS); fabric branded UnifiedBus / 灵衢 2.0 | confirmed | trendforce-950pr, huaweicentral-roadmap, ascend950-whitepaper |
| Scale-up Interconnect | **(2026-08)** UnifiedBus per-chip decomposition: 2,016 GB/s bidirectional split RTP (4 ports, 448 GB/s) + CTP (9 ports, 1,008 GB/s) over 18 × X4 HiLink at 112 Gbps/lane | single-source | ascend950-analysis |
| Scale-up Interconnect | Atlas 950 SuperPoD (full scale): 8,192 **Ascend 950DT** chips, 16 EFLOPS FP4, 64 NPUs/cabinet (announced MWC 2026) | confirmed | huawei-mwc-superpod, technetbook-atlas950 |
| Scale-up Interconnect | **(2026-08)** Atlas 950 SuperPoD WAIC 2026 demo (2026-07-17): 1,024 cards, 1 EFLOPS FP8 / 2 EFLOPS FP4, 256 TB globally unified memory address space, 3 μs RTT over 灵衢. **DEMONSTRATED, not shipping** — Q4 2026 delivery | confirmed | huawei-waic-atlas950 |
| Scale-up Interconnect | **(2026-08)** Atlas 850E air-cooled variant: 96 cards commercially deployed; scales 8 → 1,024 NPUs for conventional datacenters | confirmed | huawei-waic-atlas950, huawei-mwc-superpod |
| Scale-up Interconnect | Atlas 960 SuperPoD (roadmap Q4 2027): 15,488 Ascend 960 chips, ~32 EFLOPS (est.) | roadmap | tomshardware-roadmap, huaweicentral-roadmap |
| Scale-up Interconnect | Million-chip (百万卡) supernode cluster: Huawei target by 2027 | roadmap | huaweicentral-roadmap, trendforce-950pr |
| Scale-out Interconnect | CloudMatrix 384: 8 × 400G SiPh LPO/server, 3,168 total fibers, full-mesh optical | confirmed | qsfptek-cloudmatrix, notebookcheck-cm384 |
| Scale-out Interconnect | Atlas 900: 100 GE RoCE v2 inter-server (PCIe 4.0 + 100 GE fabric) | confirmed | atlas900-cluster |

## Source Keys Added 2026-08-08

| Key | URL |
|-----|-----|
| cann-900-relnotes | https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md |
| gitcode-cann | https://gitcode.com/cann |
| mindspore-release-notes | https://www.mindspore.cn/docs/en/stable/RELEASE.html |
| mindspore-2-9-notes | https://www.mindspore.cn/version-updates/en/2_9_en |
| mindspore-pypi | https://pypi.org/project/mindspore/ |
| vllm-ascend-relnotes | https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html |
| vllm-ascend-releases | https://github.com/vllm-project/vllm-ascend/releases |
| ascend950-whitepaper | https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf |
| ascend950-analysis | https://pillumina.github.io/posts/aiinfra/ascend-950-npu/ |
| huawei-waic-atlas950 | https://www.huawei.com/cn/news/2026/7/atlas-950-superpod |
| huawei-mwc-superpod | https://www.huawei.com/en/news/2026/3/mwc-superpod-ai |
| unifiedbus-spec | https://www.unifiedbus.com/en |
| openeuler-ub-core | https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf |
| trendforce-atlas350 | https://www.trendforce.com/news/2026/03/23/news-huawei-debuts-atlas-350-on-ascend-950pr-with-in-house-hbm-touting-2-8x-h20-performance/ |
| tomshardware-atlas350 | https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20 |
| trendforce-950dt-pullin | https://www.trendforce.com/news/2026/06/08/news-huawei-brings-forward-ascend-950dt-deployment-to-august-deepseek-v4-2-seen-as-potential-early-adopter/ |
| huaweicentral-950dt | https://www.huaweicentral.com/huawei-confirms-ascend-950dt-ai-chip-to-debut-in-august/ |
