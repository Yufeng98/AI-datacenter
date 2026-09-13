# Enflame (燧原科技) Layer Mapping Table

*as_of: 2026-08-08*
*chip: enflame*
*device_class: Data Transfer Unit (China)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | torch-gcu (PyTorch backend source published 2026-06-23, BSD-style; PrivateUse1 dispatch key; broad ATen coverage with CPU fallback; torch_gcu 2.10.0 / PyTorch 2.10.0; Hardware Support table lists **S60 only**) | confirmed | enflame-torch-gcu-readme |
| Framework Integration | torch.compile / Inductor backend in torch-gcu; AMP via torch.gcu.amp; torch.gcu.GCUGraph (CUDA-Graph analog); PyTorch caching allocator | confirmed | enflame-torch-gcu-readme |
| Framework Integration | transfer_to_gcu — one-line CUDA→GCU migration shim in torch-gcu | confirmed | enflame-torch-gcu-readme |
| Framework Integration | libtorch_gcu — C++ inference API (single-device inference only; no multi-device, no training) | confirmed | enflame-torch-gcu-readme |
| Framework Integration | vllm-gcu (vLLM fork for Enflame GCU S60; operator-level GCU optimisations; OpenAI-compatible API); v0.11.0 released 2026-05-12 — DeepSeek 3.2 (async sched/MTP/DBO/FlashMLA/FP8 KV), layer-wise KV-cache transfer, FP8 linear+MoE, W8A8-INT8 MoE, GCU Docker builds | confirmed | EnflameTechnology/vllm-gcu, vllm-gcu-releases-api |
| Framework Integration | PaddlePaddle FastDeploy Enflame GCU backend (S60 + ERNIE 4.5 under a unified GCU/GPU inference interface; driver 1.5.0.5 / TopsRider 3.4.623) — pre-baseline (2025) | confirmed | fastdeploy-enflame-gcu-doc |
| Framework Integration | FlagOS support branch (repo created 2026-05-29) | confirmed | enflame-github-org-api |
| Framework Integration | candle-gcu (Candle ML Rust framework ported to GCU; tensor ops on Enflame hardware) | confirmed | EnflameTechnology/candle-gcu |
| Framework Integration | candle-vllm-gcu (Candle-based vLLM serving on GCU) | confirmed | EnflameTechnology/candle-vllm-gcu |
| Framework Integration | PaddleNLP GCU backend (LLaMA2-13B inference on S60; PaddlePaddle custom device) | confirmed | paddlenlp-gcu-llama |
| Framework Integration | Qwen2 GCU backend (GCU support PR #456 merged to QwenLM/Qwen2) | confirmed | qwenlm-qwen2-pr456 |
| Framework Integration | TopsTorch (PyTorch plug-in; torch.device("gcu"); ATen dispatch to TopsBlasOps) — "proprietary" superseded 2026-08-08 by the torch-gcu source release, though no source states torch_gcu is a rename of TopsTorch | inferred | hami-enflame-docs, stockcounterparts, enflame-torch-gcu-readme |
| Compiler / IR | TopsCC (proprietary; lowers computation graphs to GCU-CARE static dataflow fabric at compile time) | confirmed | hami-enflame-docs, hc33-ieee |
| Compiler / IR | GCU-CARE compiler pipeline (static graph scheduling; no runtime thread dispatch; sparsity annotation) | confirmed | hc33-servethehome, hc33-ieee |
| Op Library | TopsBlasOps (proprietary GEMM + BLAS primitives; part of TopsPlatform) | inferred | hami-enflame-docs |
| Op Library | FFmpeg-GCU topscodec (hardware video encode/decode as FFmpeg plugin; GCU media ops) | confirmed | EnflameTechnology/FFmpeg-GCU |
| Kernel Library | TopsPlatform kernel library (proprietary; bundled with TopsCC + TopsRuntime; analogous to CUTLASS) | inferred | hami-enflame-docs, vllm-gcu-readme |
| Runtime | TopsRuntime (proprietary; device memory management, stream/event model, kernel launch; libTopsRuntime.so) | confirmed | hami-enflame-docs, vllm-gcu-readme |
| Runtime | GCU Monitor / profiling tools (runtime observability; Prometheus-style metrics for GCU) | confirmed | enflame-support-gcu-monitor |
| Runtime | PyTorch profiler integration — ProfilerActivity.GCU, Kineto backend | confirmed | enflame-torch-gcu-readme |
| Runtime | TopsRider SDK 3.7.x is the current line (torch-gcu pins v3.7.1; Docker tag v2.10.0-TR3.7.107-ubuntu2204; repo .version = 3.7) — supersedes the 2.5.115 doc-slug baseline; doc-site slug itself unverifiable (WAF 403) | confirmed | enflame-torch-gcu-readme |
| Driver / Firmware | TopsDrv (Linux kernel module; PCIe BAR mapping, IOCTL dispatch, DMA; analogous to nvidia.ko) | confirmed | hami-enflame-docs |
| Driver / Firmware | T10 Product Manual (official HW+driver documentation for DTU 1.0 card) | confirmed | enflame-support-t10 |
| Driver / Firmware | T20 Product Manual (official HW+driver documentation for DTU 2.0 card) | confirmed | enflame-support-t20 |
| Driver / Firmware | HAMi Kubernetes GCU device plugin (GCU resource sharing and virtualisation in K8s) | confirmed | Project-HAMi/HAMi |
| Communication | ECCL (Enflame Collective Communication Library; AllReduce/AllGather; NCCL analog over Ethernet/RoCE) | confirmed | EnflameTechnology-github, vllm-gcu |
| Communication | ECCL as a torch.distributed backend (`backend="eccl"`): full collectives plus send/recv/isend/irecv — confirmed from torch-gcu source | confirmed | enflame-torch-gcu-readme |
| Communication | nixl / ucx forks (created 2026-06-08) — directionally consistent with prefill/decode disaggregation and KV-cache transfer, but fork-creation only, no visible Enflame commits | inferred (weak) | enflame-github-org-api |
| Communication | GCU-LARE interconnect protocol (proprietary chip-to-chip communication fabric; hardware layer) | confirmed | hc33-lare-slides, enflame-support-bridge |
| Assembler / ISA | GCU-CARE ISA (proprietary static dataflow ISA; not publicly documented; TopsCC handles lowering) | inferred | hc33-ieee, hc33-care-slides |
| Assembler / ISA | GCU-DARE ISA (on-chip data movement instructions; feed into CARE compute fabric) | inferred | hc33-dare-slides |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | SIP (Scalable Intelligent Processor): Tensor ALU + DTE + Local SRAM + Sparsity Engine | confirmed | gf-press-release, hc33-ieee |
| Compute Engine | SIC (Scalable Intelligent Cluster): 8 SIPs + Shared SRAM + GCU-DARE fabric | confirmed | hc33-servethehome, hc33-ieee |
| Compute Engine | DTU 1.0 (T10): 32 SIPs / 4 SICs; 14B transistors; GF 12LP FinFET; HBM2 32 GB | confirmed | gf-press-release |
| Compute Engine | DTU 2.0 i20: 32 SIPs; 12 nm; 32 TFLOPS FP32 / 128 TF32 / 256 INT8 TOPS; 16 GB HBM2e 819 GB/s | confirmed | oceanpine-i20, zhidx-t20 |
| Compute Engine | DTU 2.0 T20: 32+ SIPs; 40 TFLOPS FP32 / 160 TF32 / 320 INT8; 64 GB HBM2e 1.8 TB/s; 9-die MCM | confirmed | zhidx-t20, baidu-baike-t20 |
| Compute Engine | DTU 3.0 S60 (SCORPIO-AO): TSMC N6NTO-HPC; 70k units shipped 2024; ~H100-class positioning | confirmed | techinsights-s60, nbcnews-tsmc |
| Compute Engine | DTU 4.0 L600: native FP8; 144 GB HBM3; 3.6 TB/s; 800 GB/s interconnect; unveiled WAIC 2025-07-27. Status: silicon returned (已回片), early commercialization — scale production and supernode delivery still forward-looking per the June 2026 IPO filing; NOT mass production | confirmed | qq-l600-waic2025, dramx-l600, sohu-prospectus-l600 |
| Compute Engine | DTU 5.0 / DTU 6.0: exist only as IPO use-of-proceeds line items (RMB 1.503 B / 1.197 B); no tape-out, silicon, or specification announced as of 2026-08-08 | confirmed | dramx-ipo-projects |
| Data Path | GCU-CARE (Compute Architecture Reconfigurable Engine): static compile-time graph dispatch; no SIMT | confirmed | hc33-ieee, hc33-care-slides |
| Data Path | GCU-DARE (Data-path Architecture Reconfigurable Engine): inter-SIP tile routing; double-buffering | confirmed | hc33-dare-slides |
| Data Path | Sparsity Engine: hardware unstructured zero-skip for weights + activations (key differentiator) | confirmed | hc33-servethehome, hc33-ieee |
| On-chip Memory | SIP Local SRAM: per-SIP private scratchpad; compiler-managed activation + weight tiling | confirmed | hc33-ieee, gf-press-release |
| On-chip Memory | SIC Shared SRAM: per-cluster inter-SIP staging buffer; compiler-managed | confirmed | hc33-ieee |
| Off-chip Memory | HBM2 (T10): 32 GB / ~512 GB/s; 2.5D CoWoS interposer; GF 12LP | confirmed | gf-press-release |
| Off-chip Memory | HBM2e (T20): 64 GB / 1.8 TB/s; 4× HBM2e stacks; 9-die MCM | confirmed | zhidx-t20, baidu-baike-t20 |
| Off-chip Memory | HBM2e (i20): 16 GB / 819 GB/s; inference-optimised 150W variant | confirmed | oceanpine-i20 |
| Off-chip Memory | HBM3 (L600): 144 GB / 3.6 TB/s; first HBM3 generation for Enflame | confirmed | trendforce-l600 |
| Host Interface / Package | PCIe Gen4 x16 (T10/T20/S60); PCIe Gen5 likely for L600; FHFL card form factor | confirmed | gf-press-release, hami-docs |
| Host Interface / Package | 2.5D MCM advanced package: compute chiplet(s) + HBM stack(s) on silicon interposer | confirmed | gf-press-release, techinsights-s60 |
| Host Interface / Package | S60: SCORPIO-AO compute chiplet (TSMC N6NTO-HPC) + separate HBM chiplets (TechInsights floorplan) | confirmed | techinsights-s60 |
| Host Interface / Package | CoPoS (Chip-on-Panel-on-Substrate) glass-substrate panel-level packaging sample with 先封科技, 2026-07-18; tunable CTE, low signal loss; **sample only, not a production package** — shipping products remain 2.5D MCM | confirmed (as a sample) | c114-esl64-copos, sina-copos |
| Scale-up Interconnect | GCU-LARE 1.0 (T10): chip-to-chip; non-cache-coherent; max 4-chip direct | confirmed | hc33-lare-slides |
| Scale-up Interconnect | GCU-LARE 2.0 (T20): 300 GB/s bidir; scalable to 1,000s of cards | confirmed | zhidx-t20 |
| Scale-up Interconnect | L600 card interconnect: 800 GB/s (announced 2025-07-27); **not attributed by any source to a named LARE generation** | confirmed | qq-l600-waic2025, dramx-l600 |
| Scale-up Interconnect | GCU-LARE Bridge Card: 3× LARE ports; 4-card full-mesh intra-server; no switch ASIC | confirmed | enflame-support-bridge |
| Scale-up Interconnect | 云燧 ESL64-O supernode (with ZTE, 2026-07-18): OEX 正交无背板 orthogonal backplane-free chassis, "0 线缆" zero-cable card-to-card interconnect. Card count, per-card BW, and host silicon **not disclosed** | confirmed (existence + design); parameters not disclosed | c114-esl64-copos, ithome-esl64o, sina-esl64 |
| Scale-up Interconnect | 云燧 ESL64-C supernode: conventional copper Cable-tray design; credited with 万卡级以上 (10,000+ card) cluster networking — claim attributed to ESL64-C specifically, not ESL64-O | confirmed | sohu-esl64c, sina-esl64 |
| Scale-up Interconnect | NPO (near-package optics) optical-interconnect prototype: claimed stable support for 512+ accelerator-card supernodes; prototype only, no link rate or supplier disclosed | confirmed (as prototype) | sina-esl64, zol-npo |
| Scale-out Interconnect | ECCL (Enflame Collective Communication Library): AllReduce/AllGather over Ethernet/RoCE | confirmed | EnflameTechnology-github |
| Scale-out Interconnect | Standard Ethernet / RoCE for inter-node; SmartCluster product integrates scale-out networking | confirmed | pandaily-ipo, hami-enflame-docs |

## Source Keys Added 2026-08-08

| Key | URL |
|---|---|
| enflame-torch-gcu-readme | https://raw.githubusercontent.com/EnflameTechnology/torch-gcu/main/README.md |
| enflame-github-org-api | https://api.github.com/orgs/EnflameTechnology/repos |
| vllm-gcu-releases-api | https://api.github.com/repos/EnflameTechnology/vllm-gcu/releases |
| c114-esl64-copos | https://www.c114.net.cn/industry/101611.html |
| ithome-esl64o | https://www.ithome.com/0/978/716.htm |
| sina-esl64 | https://finance.sina.com.cn/tech/shenji/2026-07-21/doc-iniipxxe9278635.shtml |
| sohu-esl64c | https://www.sohu.com/a/1051829208_313745 |
| zol-npo | https://ai.zol.com.cn/1219/12193958.html |
| sina-copos | https://finance.sina.com.cn/stock/t/2026-07-18/doc-iniifanf8343506.shtml |
| qq-l600-waic2025 | https://news.qq.com/rain/a/20250727A06S1L00 |
| dramx-l600 | https://www.dramx.com/News/IC/20250728-38864.html |
| dramx-ipo-projects | https://www.dramx.com/News/IC/20260616-40620.html |
| sohu-prospectus-l600 | https://www.sohu.com/a/1037503510_237556 |
| fastdeploy-enflame-gcu-doc | https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/get_started/installation/Enflame_gcu.md |
