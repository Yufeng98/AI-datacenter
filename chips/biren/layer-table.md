# Biren BR10X / BR20X Layer Mapping Table

*as_of: 2026-08-08*
*prior revision: 2026-04-05*
*chip: biren*
*device_class: GPU-like AI Accelerator (China)*
*generations covered: BR100 / BR104 (2022, never volume) · BR106 (2023) · BR110 (2024) · BR166 (2025) · BR20X (in development)*

> Rows below tagged **[2026-08]** were added or revised in the 2026-08-08 update. BR100/BR104 rows describe the 2022 Hot Chips 34 parts, which never reached volume production, and are retained as architectural history.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch SUPA backend (torch.device("supa"); ATen dispatch → BILA; BCCL distributed training; TF32+/BF16 autocast) | confirmed | hpcwire-biren, birentech-official |
| Framework Integration | TensorFlow integration (XLA backend or custom device plugin; training + inference on BR100) | confirmed | hpcwire-biren |
| Framework Integration | PaddlePaddle custom device plugin (Baidu ecosystem; custom ops for BR100/BR104) | confirmed | hpcwire-biren |
| Framework Integration | LLM inference serving (vLLM-style; LLaMA/Qwen/Baichuan; SUPA Flash-Attention kernels) | inferred | ainvest-biren |
| Framework Integration | **[2026-08]** SGLang fork — BIRENSUPA/sglang, created 2026-05-06, **zero pushes since creation** (static fork, NOT an active port) | confirmed | github-birensupa-api |
| Framework Integration | **[2026-08]** mmcv fork — BIRENSUPA/mmcv, forked 2026-08-05; OpenMMLab CV op layer | confirmed | github-birensupa-api |
| Framework Integration | **[2026-08]** Day-0 model enablement claims: MiniMax M3 (2026-06-16), Zhipu GLM-5.2 (2026-06-26), MiniMax H3 (2026-08-03) | vendor claim | birentech-newsroom |
| Framework Integration | **[2026-08]** Token Factory application framework — five-level caching claiming 95%+ cache hit rate; cross-vendor heterogeneous co-inference with China Telecom claiming ~20% throughput gain (unaudited vendor claim, primary-sourced on Biren's newsroom) | vendor claim | birentech-waic2026 |
| Serving / KV-cache | **[2026-08]** Mooncake fork — BIRENSUPA/Mooncake, created 2026-07-20, actively pushed through 2026-08-08; KV-cache disaggregation and prefill/decode separation. The only actively developed serving repo in the org | confirmed | github-birensupa-api |
| Compiler / IR | BRCC (Biren Runtime Compiler; nvcc analog; SUPA C++ → Biren native ISA binary; CUDA migration mode) | confirmed | hpcwire-biren, allaboutcircuits |
| Compiler / IR | SUPA MLIR inference compiler (TensorRT analog; ONNX/PyTorch/TF → fused+quantized BR100 engine; INT8/BF16/TF32+) | inferred | hpcwire-biren |
| Compiler / IR | cuda2supa migration tool (automated CUDA→SUPA namespace substitution; cuBLAS/cuDNN/NCCL library shims) | confirmed | kr-asia-supa, hpcwire-biren |
| Op Library | BILA — Biren DL Library (cuDNN+cuBLAS analog; GEMM/Conv/Attn/Norm/Pooling; TF32+/BF16/INT8 engines) | confirmed | hpcwire-biren, birentech-official |
| Op Library | General compute library (BLAS L1-L3; Reduce/Scan/Sort; vector elementwise) | inferred | hpcwire-biren |
| Kernel Library | SUPA kernel templates (CUTLASS analog; GEMM/Conv templates; L2-cache-aware tiling for 300 MB distributed L2) | inferred | hpcwire-biren |
| Runtime | SUPA Runtime API libsupa.so (cudart analog; supaMalloc/Free/MemcpyAsync; supaStream; supaEvent; <<<>>> launch) | confirmed | hpcwire-biren, allaboutcircuits |
| Runtime | SUPA Driver API (CUDA Driver analog; explicit context mgmt; module loading; fine-grained memory) | inferred | hpcwire-biren |
| Driver / Firmware | Biren Linux kernel driver .ko (PCIe BAR/MMIO; IOCTL dispatch; DMA engine; interrupt handling) | confirmed | hpcwire-biren |
| Driver / Firmware | On-GPU firmware (resource management; power control; NVIDIA GSP analog) | inferred | hpcwire-biren |
| Driver / Firmware | Kubernetes device plugin (birentech.com/gpu resource; health monitoring; SVI multi-tenancy) | inferred | hpcwire-biren |
| Driver / Firmware | **[2026-08]** biren-driver-management-tools-skill (BIRENSUPA org, created 2026-06-24) — driver management tooling | confirmed | github-birensupa-api |
| Driver / Firmware | **[2026-08]** Legacy BirenTechnology GitHub org (ModelZoo, k8s-device-plugin, go-brml) is **fully archived**; last push 2024-12-17. Active development has moved to the BIRENSUPA org | confirmed | github-birentechnology-api |
| Communication | BCCL — Biren Collective Communication Library (NCCL analog; AllReduce/AllGather over BLink intra-node; IB/ETH inter-node) | confirmed | hpcwire-biren |
| Assembler / ISA | BGISA — Biren GPU ISA (proprietary SIMT ISA; C-Warp based; FP32/TF32+/BF16/INT8; NOT publicly documented; no PTX equivalent) | confirmed | hc34-hotchips, chipsandcheese |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | SPC (Streaming Processing Cluster): 32 per die (64 BR100 total); 16 EU each; 4,096 threads; C-Warp SIMT scheduler | confirmed | hc34-hotchips, videocardz |
| Compute Engine | EU (Execution Unit): atom compute block; FP32/TF32+/BF16/INT8 pipelines; tensor units; EU L1+shared memory | confirmed | hc34-hotchips, chipsandcheese |
| Compute Engine | BR100 peak: 256 TFLOPS FP32, 512 TFLOPS TF32+, 1024 TFLOPS BF16, 2048 TOPS INT8 | confirmed | hc34-hotchips, wccftech |
| Compute Engine | BR104 peak: 128 TFLOPS FP32, ~1024 TOPS INT8; monolithic single die; 300W PCIe | confirmed | servethehome, videocardz |
| Compute Engine | **[2026-08]** BR100/BR104 never reached volume production — TSMC suspended advanced-node work Oct 2022. The commercially shipping BR10X parts are BR106 / BR110 / BR166 (壁砺106/110/166 — same series, same first-generation architecture, TSMC 7nm) | confirmed | baike-br106, birentech-official |
| Compute Engine | **[2026-08]** BR106 (壁砺106): single die, BR10X arch, TSMC 7nm; dev from 2020, tape-out 2021, mass production Jan 2023. Peak FLOPS **not disclosed** | confirmed | baike-br106 |
| Compute Engine | **[2026-08]** BR110 (壁砺110): single die, same architecture as BR106; mass production Oct 2024; edge-inference positioning; no product page on birentech.com. Peak FLOPS **not disclosed** | confirmed | thepaper-annualreport |
| Compute Engine | **[2026-08]** BR166 (壁砺166): 2.5D chiplet co-packaging two BR106 dies with die-to-die interconnect; mass production Aug 2025, shipped at scale 2H2025. Peak FLOPS **not disclosed**; vendor claims "compute and memory doubled vs prior generation" | confirmed (vendor claim for the 2× figure) | baike-bili166, birentech-official, thepaper-annualreport |
| Compute Engine | **[2026-08]** ⚠️ "BF16 800 TFLOPS / 128 GB HBM" for the 166 series is aggregator-only (Toutiao); all four Biren product pages publish form factor and peak power only. **Not adopted as spec** | refuted as spec | toutiao-aggregator, birentech-166m |
| Compute Engine | **[2026-08]** BR20X ("BR2xx"): 2nd-generation Biren architecture with **native FP8 and FP4** — first sub-INT8 numerics in the line. Chiplet; compute density/memory increases stated but unquantified. Status: architecture design complete, in **physical design and tape-out verification**; NOT taped out, not sampling. Launch planned 2026 | confirmed (status); not disclosed (specs) | birentech-waic2026, sohu-annualreport, jiemian-waic2026 |
| Compute Engine | **[2026-08]** BR30X (cloud training/inference) and BR31X (edge inference) targeted for 2028 commercialization — roadmap only, no specs | confirmed (roadmap) | thepaper-annualreport |
| Compute Engine | NO FP64 support (deliberate AI-first design; unlike NVIDIA H100/AMD MI300X) | confirmed | chipsandcheese, hc34-hotchips |
| Compute Engine | TF32+ extended format: higher dynamic range than NVIDIA TF32; key training accuracy-throughput tradeoff | confirmed | hc34-hotchips, eet-china |
| Data Path | C-Warp SIMT execution: threads in lock-step per EU; hardware divergence via predication | confirmed | hc34-hotchips, zhihu-biren |
| Data Path | Data-flow architecture (BiLi): TF32+, TDA, C-Warp, NME, NUMA/UMA, SVI — six proprietary features | confirmed | hc34-hotchips, zhidx |
| Data Path | Mesh on-chip network: SPCs connected via mesh to L2 cache slices; similar to Intel Sapphire Rapids topology | inferred | chipsandcheese |
| Data Path | Die-to-die link: 896 GB/s bidirectional via CoWoS-S silicon interposer | confirmed | videocardz, servethehome |
| On-chip Memory | Distributed L2 cache: 300 MB total (150 MB per die); hardware-managed; ~6x NVIDIA A100 L2 | confirmed | hc34-hotchips, videocardz |
| On-chip Memory | EU L1 / shared memory: per-EU; specific capacity not publicly disclosed | inferred | chipsandcheese |
| Off-chip Memory | HBM2e (BR100): 64 GB total, 2,300 GB/s (2.3 TB/s), 4,096-bit interface, 4 stacks | confirmed | videocardz, servethehome, hc34-hotchips |
| Off-chip Memory | HBM2e (BR104): 32 GB, 819 GB/s, 2,048-bit interface, 2 stacks | confirmed | videocardz, servethehome |
| Off-chip Memory | HBM supply constraint: US export controls restrict SK Hynix/Samsung HBM for Biren; critical production blocker 2022-2026 | confirmed | indrastra-tsmc, bloomberg-ipo |
| Off-chip Memory | **[2026-08]** BR106 / BR110 / BR166 / BR20X off-chip memory: type, capacity and bandwidth all **not disclosed**. birentech.com publishes no memory table for any SKU | not disclosed | birentech-166m, birentech-106m |
| On-chip Memory | **[2026-08]** BR106 / BR110 / BR166 / BR20X on-chip cache capacity **not disclosed**; the 300 MB L2 figure belongs to BR100 and must not be carried forward | not disclosed | birentech-official |
| Host Interface / Package | PCIe Gen5 x16 + CXL 1.1/2.0: ~128 GB/s bidir host interface; coherent memory expansion via CXL | confirmed | hc34-hotchips, videocardz |
| Host Interface / Package | TSMC CoWoS-S 2.5D packaging (BR100): two dies + 4 HBM2e stacks on silicon interposer | confirmed | videocardz, chipsandcheese |
| Host Interface / Package | OAM form factor (BR100): 550W TDP; Open Accelerator Module; 8-card per server | confirmed | servethehome, tomshardware |
| Host Interface / Package | PCIe FHFL card (BR104): 300W TDP; standard full-height full-length server card | confirmed | servethehome |
| Host Interface / Package | **[2026-08]** 壁砺™166M: 4U OAM V1.1 air-cooled module, **550 W peak** | confirmed | birentech-166m |
| Host Interface / Package | **[2026-08]** 壁砺™166L: cold-plate **liquid-cooled** OAM module, **600 W peak** — first Biren part above the 550 W OAM envelope | confirmed | birentech-166l |
| Host Interface / Package | **[2026-08]** 壁砺™166C: full-height full-length (290 mm) **dual-width PCIe** inference card, **300 W peak** | confirmed | birentech-166c |
| Host Interface / Package | **[2026-08]** 壁砺™106M: air-cooled OAM module, **400 W peak**; 壁砺™106B: FHFL dual-width PCIe card (power not disclosed) | confirmed | birentech-106m, baike-br106 |
| Host Interface / Package | **[2026-08]** BR166 packaging: 2.5D chiplet co-package of two BR106 dies + die-to-die interconnect. Die-to-die bandwidth **not disclosed** (the 896 GB/s figure is BR100-specific) | confirmed (packaging); not disclosed (bandwidth) | baike-bili166 |
| Scale-up Interconnect | BLink: 8 bidirectional links × 64 GB/s = 512 GB/s aggregate; NVLink analog; up to 8 BR100/server — **BR100-era (BLink 1.x) figures** | confirmed | hc34-hotchips, videocardz |
| Scale-up Interconnect | No BLink switch ASIC disclosed (no NVSwitch equivalent); point-to-point topology only — **BR100-era; superseded by BLink 2.0's in-network computing, which implies switch nodes** | confirmed (for BR100) | chipsandcheese |
| Scale-up Interconnect | **[2026-08]** BLink 2.0 (announced WAIC 2026, 2026-07-17): memory-semantic interconnect allowing up to **1,024 GPUs to share one memory space**; in-network computing (collectives offloaded into switch nodes); intelligent congestion control; multi-layer link self-healing (physical→framework). Per-link and aggregate bandwidth **not disclosed** | confirmed (announcement); not disclosed (bandwidth) | birentech-waic2026, jiemian-waic2026 |
| Scale-up Interconnect | **[2026-08]** NPO (near-packaged optics): optical engine fused into the GPU module, **high-power DSP retimer removed**, optical reach "several hundred metres". Motivation stated as >1 TB/s per-GPU scale-up requirement vs copper that "attenuates severely within 3 metres". ⚠️ A "224 Gbps port rate" is **not** in Biren's release | confirmed | birentech-waic2026 |
| Scale-up Interconnect | **[2026-08]** Three-tier supernode matrix: 16-card standard server (electrical) / 128-card high-density cabinet (electrical) / **1,024-card distributed decoupled NPO optical supernode** with GPU nodes and switch nodes physically separated. Breaks the prior 128-card electrical ceiling. Roadmap announcement — no availability date, per-link bandwidth, or customer disclosed | confirmed (announcement) | birentech-waic2026, 163-waic2026, 10jqka-waic2026 |
| Scale-up Interconnect | **[2026-08]** Prior-generation **dOCS** (distributed optical circuit switching) supernode: 32-card / 4-chassis building block, aggregated to a **2,048-card cluster** at a national computing platform (reported as Shanghai INESA). ⚠️ **Not** a 2,048-card scale-up/shared-memory domain and **not** BLink 2.0; Biren's own release states the deployment without a card count | confirmed (deployment); qualified (card count from annual report + media) | sohu-annualreport, 163-waic2026, birentech-waic2026 |
| Scale-out Interconnect | Standard InfiniBand / Ethernet via host NICs (no proprietary scale-out NIC) | confirmed | hpcwire-biren |
| Scale-out Interconnect | BCCL for intra-node BR100 collectives over BLink; IB/ETH for inter-node | confirmed | hpcwire-biren |
| Scale-out Interconnect | **[2026-08]** BLink 2.0's in-network computing moves collective reduction into the fabric, which would shift work out of BCCL's host/GPU path. No BCCL API or version detail has been published for BLink 2.0 | not disclosed | birentech-waic2026 |

## Source Keys Added 2026-08-08

| Key | Reference |
|---|---|
| birentech-official | https://www.birentech.com/ (hardware product menu: 壁砺™166L / 166M / 166C / 106M / 106B) |
| birentech-166m | https://www.birentech.com/product/hardware/166m/ |
| birentech-166l | https://www.birentech.com/product/hardware/166l/ |
| birentech-166c | https://www.birentech.com/product/hardware/166c/ |
| birentech-106m | https://www.birentech.com/product/hardware/106m/ |
| birentech-newsroom | https://www.birentech.com/news/ |
| birentech-waic2026 | https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/ (Biren WAIC 2026 press release) |
| jiemian-waic2026 | https://www.jiemian.com/article/14787272.html |
| 163-waic2026 | https://www.163.com/tech/article/L22F1NH700098IEO.html |
| 10jqka-waic2026 | https://www.10jqka.com.cn (2026-07-18) |
| sohu-annualreport | https://www.sohu.com/a/1003843604_313745 (FY2025 results, 2026-03-30) |
| thepaper-annualreport | https://m.thepaper.cn — Biren first annual report coverage (BR166 mass production Aug 2025; BR110 Oct 2024; BR30X/BR31X 2028) |
| baike-br106 | https://baike.baidu.com/item/BR106/67163892 |
| baike-bili166 | https://baike.baidu.com/item/%E5%A3%81%E7%A0%BA166%E7%B3%BB%E5%88%97/67163995 |
| github-birensupa-api | https://api.github.com/orgs/BIRENSUPA/repos |
| github-birentechnology-api | https://api.github.com/orgs/BirenTechnology/repos |
| toutiao-aggregator | https://www.toutiao.com/w/1865240228792329/ (⚠️ sole source of the unadopted 800 TFLOPS / 128 GB figures) |
