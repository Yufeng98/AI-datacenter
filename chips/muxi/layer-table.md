# MetaX (沐曦) GPU Layer Mapping Table

*as_of: 2026-08-08*
*chip: muxi*
*device_class: GPU (China, 沐曦)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch MXMACA backend (torch.device("maca"); autograd; distributed training; torch.compile via TorchInductor) | confirmed | metax-maca-vllm-rfc, metax-official-platform |
| Framework Integration | TensorFlow (via CUDA ON MACA compatibility layer; end-to-end DL pipeline supported) | confirmed | metax-official-platform |
| Framework Integration | vLLM-MetaX (github.com/MetaX-MACA/vLLM-metax; cuda_alike backend; version-aligned with upstream vLLM; **latest v0.22.0, 2026-07-27**; chain v0.13.0→v0.14.0→v0.15.0→v0.17.0→v0.18.0→v0.19.0→v0.20.0→v0.21.0→v0.22.0; Gemma 4 in v0.19.0, DeepSeek V4 rebase in v0.21.0; LLaMA/Qwen; RFC #23157) | confirmed | github-vllm-metax-releases-api, vllm-rfc-23157 |
| Framework Integration | SGLang fork (github.com/metax-maca/sglang; created 2026-04-02) | confirmed | github-metax-maca-org-api |
| Framework Integration | vllm-omni-metax (omni-modal vLLM variant; created 2026-04-24) | confirmed | github-metax-maca-org-api |
| Framework Integration | llama.cpp MetaX backend (community integration) | inferred | metax-devportal |
| Framework Integration | DeepSeek integration (MetaX developer portal lists DeepSeek resources) | confirmed | metax-devportal |
| Framework Integration | PaddlePaddle (飞桨) native support — added in MXMACA 3.3.0.X, Dec 2025 | medium | baidu-baike-mxmaca |
| Framework Integration | PyTorch 2.8 adaptation (MXMACA 3.3.0.X, Dec 2025); MetaX claims ~92.94% seamless CUDA-project migration | medium | baidu-baike-mxmaca |
| Framework Integration | AI4S frameworks: AI4S-Framework (2026-08-07), LifeScience + FluidDynamics (2026-07-13), MedicalImage (2026-07-27) — software mirror of the X300 AI4S GPU launch | confirmed | github-metax-maca-org-api |
| Framework Integration | Model Day-0 adaptation: MiniMax H3 multimodal model adapted Day-0 on 曦云 C-series via MXMACA (2026-08-03); MetaX claims 30 Day-0 adaptations since Dec 2025, 40+ frameworks, 500+ models (vendor claims) | confirmed (H3) / vendor-claim (aggregates) | metax-newsroom-h3, metax-newsroom-waic2026 |
| Compiler / IR | MXMACA Compiler (nvcc analog; MACA C++ kernels <<<>>> syntax → MXMACA ISA binary; in MXMACA SDK 2.0+) | confirmed | metax-official-platform, vllm-rfc-23157 |
| Compiler / IR | CUDA ON MACA (automated CUDA→MACA source translator; cuBLAS→mxBLAS, cuDNN→mxDNN, NCCL→MXCCL; requires recompile) | confirmed | metax-official-platform, financialcontent-metax |
| Compiler / IR | mcpy (github.com/MetaX-MACA/mcpy; CuPy-compatible NumPy GPU library for MACA) | confirmed | github-metax-maca-org |
| Compiler / IR | mcTriton (Triton port for MACA; created 2026-03-13) — gives MetaX a Triton DSL front end alongside MACA C++ | confirmed | github-metax-maca-org-api |
| Compiler / IR | TileOPs-Metax + TileKernels-Metax (TileLang-based high-performance LLM operator/kernel libraries; both created 2026-04-24) | confirmed | github-metax-maca-org-api |
| Op Library | mxDNN (cuDNN analog; Conv/Attention/SDPA/Norm/Pooling/Activation; FP32/BF16/FP16/INT8; FP8 in C600) | confirmed | metax-official-platform |
| Op Library | mxBLAS (cuBLAS analog; GEMM/BLAS L1-L3; epilogue fusion) | confirmed | metax-official-platform |
| Op Library | mxFFT (cuFFT analog; FFT operations) | confirmed | metax-official-platform |
| Op Library | MXMACA 3.3.0.X operator coverage: 2,650 core operators, of which 2,410 GPU operators (Dec 2025) | medium | baidu-baike-mxmaca |
| Kernel Library | mxThrust (Thrust analog; Reduce/Scan/Sort/Transform parallel primitives) | confirmed | metax-official-platform |
| Kernel Library | MXMACA Math Libraries (device-side standard math: sin, cos, exp, log, etc.) | confirmed | metax-official-platform |
| Kernel Library | McFlashInfer (FlashInfer port — attention kernels; created 2026-03-09) | confirmed | github-metax-maca-org-api |
| Kernel Library | mcFlashMLA (FlashMLA port — DeepSeek MLA attention; created 2026-07-01) | confirmed | github-metax-maca-org-api |
| Kernel Library | hpc_warp (HPC warp-level primitives; created 2026-01-13); op_optimization (created 2026-05-26); AIModels model zoo (created 2026-05-18) | confirmed | github-metax-maca-org-api |
| Runtime | MXMACA Runtime (libmaca.so; cudart analog; macaMalloc/Free, macaMemcpyAsync, macaStream/Event, <<<>>> launch) | confirmed | metax-official-platform, hami-metax-support |
| Runtime | MXMACA Driver API (context management; module loading; JIT compilation; fine-grained memory) | confirmed | metax-official-platform |
| Driver / Firmware | MetaX Kernel Driver (.ko; Linux PCIe; BAR mapping/IOCTL/DMA/IRQ) | confirmed | metax-driver-install-guide |
| Driver / Firmware | mx-exporter (Prometheus-compatible MetaX GPU monitoring; Kubernetes-aware; utilization/memory/temp metrics) | confirmed | metax-exporter-manual |
| Driver / Firmware | Kubernetes Device Plugin (MetaX GPU resource advertising; health monitoring; multi-tenant partitioning) | confirmed | hami-metax-support |
| Driver / Firmware | MXSML management-library bindings: go-mxsml (created 2026-03-16), pymxsml (created 2026-03-19) — NVML-analog device management from Go and Python | confirmed | github-metax-maca-org-api |
| Communication | MXCCL (MetaX Collective Communication Library; NCCL analog; AllReduce/AllGather/ReduceScatter; MetaXLink intra; RoCE inter; 10K+ GPU proven) | confirmed | metax-official, financialcontent-metax |
| Communication | mccl_tests (MXCCL collective benchmark suite, nccl-tests analog; created 2026-05-28) | confirmed | github-metax-maca-org-api |
| Communication | MXDeepEP (DeepEP expert-parallel communication library port; created 2026-06-10) — MoE expert-parallel dispatch/combine, pairing with the S600 supernode's stated EP support | confirmed | github-metax-maca-org-api |
| Assembler / ISA | MXMACA ISA (proprietary SIMT ISA; warp-based; FP32/BF16/FP16/INT8 on C500; adds FP8/INT4 on C600; not publicly documented) | confirmed | metax-official-platform, vllm-rfc-23157 |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | MXN100 (Xisi N100): 160 TOPS INT8; 80 TFLOPS FP16; HBM2E; 128-ch encode/96-ch decode; 8K video (HEVC/H.264/AV1/AVS2); mass production 2023 | confirmed | tomshardware-metax-n100, wccftech-metax-n100, tphuang-n100 |
| Compute Engine | MXC500 (Xiyun C500): 7nm; ~15 TFLOPS FP32; ~77% of A100; ~160 TOPS INT8; HBM2E ~64 GB; PCIe; MXMACA 2.0; 100B+ LLM training (claimed) | confirmed | tphuang-c500, tomshardware-metax, mxc500-benchmark |
| Compute Engine | MXC550 (Xiyun C550): 7nm; OAM form factor; 8 OAM/server; MetaXLink; C550 3D Mesh Supernode (64 cards); Shanghai Cube (128 cards/47U liquid) | confirmed | metax-c550-server, metax-c550-supernode |
| Compute Engine | MXC600 (Xiyun C600): Domestic process; FP8 1,000 TFLOPS; HBM3e 144 GB 3.6 TB/s; ~2.5 TFLOPS/W FP8; FP32/BF16/FP16/FP8/INT8/INT4; fully domestic supply chain; targets H20+; **large-scale shipment during 2026** (company statement 2026-07-08, not audited; orders booked into 2027) | confirmed (specs) / company-statement (volume) | ithome-c600, redhotcyber-c600, eastmoney-c600, sina-c600-volume-20260708 |
| Compute Engine | MXC588 (Xiyun C588): launched 2025-09-23 alongside C600 at 曦果发布会; C588 Server listed; all specs (compute, memory, process, TDP) not disclosed | confirmed (existence) | metax-product-catalog |
| Compute Engine | MXN300 (Xisi N300): C600-generation inference derivative; FHFL dual-slot PCIe; max 500 W board power; "domestic advanced process" (node not disclosed); MetaXLink across 4 cards; memory capacity, memory BW and INT8/FP8 TOPS not disclosed. Aggregator "48 GB" and "14nm" figures NOT confirmed and not adopted | confirmed (vendor page) | metax-n300-product, metax-n300-server |
| Compute Engine | MXX206 (Xisuo X206): first 曦索 X-series AI4S GPU; launched 2026-01-27 at 智算申城 forum, Shanghai; X206 Server listed; all specs not disclosed | confirmed (existence) | metax-product-catalog |
| Compute Engine | MXX300 series — X301 / X302 (Xisuo): second-gen AI4S GPU; announced WAIC 2026 (Jul 17–20, 2026); self-developed GPGPU on fully domestic supply chain; "full-precision mixed compute" (全精度混合算力, implying FP64 alongside low precision); large-capacity high-bandwidth memory; MetaXLink + MLoE; targets NWP/oceanography/CFD/molecular dynamics/materials/life sciences. FP64 rate, all FLOPS/TOPS, memory type-capacity-BW, TDP and process node not disclosed. **Announced only** | confirmed (announcement + vendor attributes) | metax-newsroom-waic2026, ithome-waic2026, metax-product-catalog |
| Compute Engine | MXC700 (Xiyun C700): company-disclosed 2026-04-08 (chairman Chen Weiliang, FY2025 results call) — core chip design and functional verification largely complete, in deeper performance optimization; project started Apr 2025; targets internet-sector customers beyond C600's 信创 base. No specs, no launch date. "H100-parity" and "mass production late 2027" are aggregator-only and unverified | company-disclosure (status) / unverified (perf + timeline) | tencent-news-fy2025-call |
| Data Path | SIMT execution: threads → warps → warp scheduler; hardware divergence; <<<>>> launch (CUDA-compatible syntax) | confirmed | metax-official-platform, vllm-rfc-23157 |
| Data Path | Tensor acceleration units: FP16/BF16/INT8 matrix ops (C500); adds native FP8 Tensor instructions (C600) | confirmed | ithome-c600, eastmoney-c600 |
| Data Path | "Full-precision mixed compute" (全精度混合算力) datapath on X300 series — FP64-class numerics alongside low-precision AI datatypes; first MetaX family positioned for double-precision numerical simulation. FP64 rate not disclosed | confirmed (vendor claim) | metax-newsroom-waic2026 |
| Data Path | Full-function GPU: compute + video pipeline (MXN series) + graphics pipeline (MXG Xicai series) | confirmed | tomshardware-metax-n100, metax-mxg-page |
| On-chip Memory | L2 cache / scratchpad: capacity not publicly disclosed (proprietary) | inferred | general-gpu-arch-analogy |
| Off-chip Memory | MXN100 / MXC500: HBM2E; MXC500 capacity ~64 GB (from benchmark reports) | confirmed | mxc500-benchmark, hami-metax |
| Off-chip Memory | MXC600: HBM3e 144 GB; 3.6 TB/s bandwidth | confirmed | ithome-c600, eastmoney-c600 |
| Off-chip Memory | MXC588 / MXN300 / MXX206 / MXX300: memory type, capacity and bandwidth all **not disclosed**. X300 described only as "large-capacity high-bandwidth memory". N300's circulating "48 GB" is aggregator-only and NOT confirmed | not disclosed | metax-product-catalog, metax-n300-product, metax-newsroom-waic2026 |
| Host Interface / Package | PCIe (gen not officially disclosed) for C500; OAM for C550 | confirmed | metax-c550-server, tphuang-c500 |
| Host Interface / Package | MXN300: FHFL dual-slot PCIe card, max 500 W board power; N300 Server hosts up to 16 N300 GPUs per chassis via PCIe-Switch full interconnect (Common topology and 8-GPU configs also offered) | confirmed | metax-n300-product, metax-n300-server |
| Scale-up Interconnect | MetaXLink (MX): NVLink analog; 8 GPUs/server full-mesh; GPU-to-CPU NUMA topology; 3D Mesh supernode (64 cards); 4-card MetaXLink groups on N300 PCIe cards | confirmed | hami-metax-support, metax-c550-supernode, metax-n300-product |
| Scale-up Interconnect | MLoE: second high-speed multi-card interconnect named alongside MetaXLink for the X300 series. Mechanism, bandwidth, topology and radix all not disclosed | confirmed (name only) | metax-newsroom-waic2026 |
| Scale-up Interconnect | C550 Shanghai Cube: 128 liquid-cooled MetaX C550 cards in 47U single cabinet (ultra-high-density) | confirmed | metax-c550-supernode |
| Scale-up Interconnect | C550 3D Mesh Supernode: up to 64 cards across up to 8 servers, full-cabinet, electrical (memory-semantic) 3D Mesh — **still a current listed product**, not superseded by S600 | confirmed | metax-c550-supernode, metax-product-catalog |
| Scale-up Interconnect | C500X Optical Interconnect Supernode: hybrid optical/electrical; secondary reporting gives DragonFly topology scaling 16→64 GPUs across up to 8 machines. Material traces to WAIC 2025 | medium (topology/scale; no vendor spec sheet) | metax-product-catalog |
| Scale-up Interconnect | 曦景 S600 (Xijing S600) supernode: 64 GPU cards in a **single cabinet**; in-cabinet full interconnect; **"zero-cable direct-connect" (0线缆直连)** between compute and switch nodes; supports EP/TP and other multi-dimensional parallelism for training and inference; cabinet-to-cabinet expansion to 万卡级 (10,000+ card) clusters. Per-link BW, GPU SKU, total FLOPS, power/cooling and switch silicon **not disclosed**. Announced WAIC 2026 (Jul 17–20, 2026); **announced only** — no sampling/shipping/benchmark evidence | confirmed (announcement + vendor attributes) | metax-newsroom-waic2026, ithome-waic2026, sina-waic2026 |
| Scale-out Interconnect | Standard Ethernet / RoCE via host NIC + MXCCL collective library | confirmed | metax-official, financialcontent |
| Scale-out Interconnect | MXCCL handles inter-node collectives; MetaXLink for intra-node; proven at 10,000+ GPU scale (end-2024) | confirmed | financialcontent-metax |

## Source Keys Added 2026-08-08

| Key | URL |
|-----|-----|
| metax-newsroom-waic2026 | https://www.metax-tech.com/en/ndetail/12629.html |
| metax-newsroom-h3 | https://www.metax-tech.com/ndetail/12632.html |
| metax-product-catalog | https://www.metax-tech.com/en/goods/prod.html?cid=3 |
| metax-n300-product | https://www.metax-tech.com/en/goods/prod.html?cid=106&id=66 |
| metax-n300-server | https://www.metax-tech.com/en/goods/prod.html?cid=110&id=69 |
| ithome-waic2026 | https://www.ithome.com/0/978/460.htm |
| sina-waic2026 | https://finance.sina.com.cn/tech/shenji/2026-07-20/doc-iniimuyk4915570.shtml |
| sina-c600-volume-20260708 | https://finance.sina.com.cn/wm/2026-07-08/doc-inihceis8322692.shtml |
| tencent-news-fy2025-call | https://news.qq.com/rain/a/20260408A066PB00 |
| github-vllm-metax-releases-api | https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases |
| github-metax-maca-org-api | https://api.github.com/orgs/metax-maca/repos |
| baidu-baike-mxmaca | https://baike.baidu.com/item/MXMACA%E8%BD%AF%E4%BB%B6%E6%A0%88 |
| eastmoney-financials-2026 | https://finance.eastmoney.com/a/202604293724854763 |
