# Hygon DCU Layer Mapping Table

*as_of: 2026-08-08*
*chip: hygon-dcu*
*device_class: GPU/DCU (China, 海光)*

> **Generation scope (2026-08-08):** hardware rows below describe **深算一号 (Y100) / 深算二号 (Z100, Z100L) / K100 / K100_AI**. The current flagship **深算三号 (BW1000)** has **no disclosed hardware specification**; it appears as explicit "not disclosed" rows.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (DTK ROCm HIP compat; torch.cuda → torch.hip; PyTorch 2.4+; DTK 24/25) | confirmed | gpustack-tutorial, csdn-dtk-guide |
| Framework Integration | PaddlePaddle (first-class DCU support; official install docs; multi-GPU; mixed precision FP16/BF16; recommended for China market) | confirmed | paddlepaddle-dcu-docs, paddlepaddle-rocm |
| Framework Integration | TensorFlow (DTK ROCm backend; TF 2.13+) | confirmed | flyaibox-dcu-in-action |
| Framework Integration | vLLM (community port for DCU; K100_AI LLM serving) | community | csdn-dcu-deploy |
| Framework Integration | GPUStack (official DCU inference backend; K100_AI tutorial) | confirmed | gpustack-dcu-tutorial |
| Framework Integration | PaddleNLP / PaddleOCR-VL (official DCU support; llama2-7b K100_AI guide) | confirmed | paddlenlp-dcu-install, paddleocr-dcu |
| Framework Integration | FlyAIBox/dcu-in-action (community cookbook; LLaMA pre-train, ChatGLM fine-tune, HPC) | community | github-flyaibox |
| Framework Integration | DeepSeek on DCU (Haiguang DCU + DeepSeek localization; confirmed working) | confirmed | eeworld-deepseek-dcu |
| Framework Integration | Tencent Hunyuan Hy3 (open-source preview) adaptation on 深算三号, reported completed May 2026 | low-medium (encyclopedia sourcing only) | baike-shensuan3 |
| Framework Integration | DeepSeek V4 "priority adaptation" on 深算三号 | unverified vendor/channel claim | — |
| Framework Integration | 深算三号 "类CUDA" positioning; "operator/algorithm coverage >99%" | vendor marketing claim, not independently verified | — |
| Compiler / IR | hipcc (DTK LLVM clang driver; HIP C++ → amdgcn IR → DCU ISA; primary dev path) | confirmed | dtk-portal, csdn-dcu-programming |
| Compiler / IR | DTK LLVM suite (clang, clang++, OpenMP target offload, OpenACC, rocm-gdb, rocprofv2) | confirmed | dtk-portal, csdn-dtk-install |
| Compiler / IR | hipify-perl / hipify-clang (CUDA → HIP automated translation ~90-95%; included in DTK) | confirmed | dtk-portal |
| Compiler / IR | DTK ROCm migration path (documented; version dependency management guide) | confirmed | tbr8-dtk-migration |
| Compiler / IR | DTK release cadence — latest versions located are DTK 25.04 / 25.10 (2025-vintage naming); **no 2026-dated DTK (26.x) release confirmed**; developer.sourcefind.cn unreachable during 2026-08-08 verification | not disclosed / unverified | dtk-portal |
| Compiler / IR | No BW1000-specific DTK branch, ISA change, or new library documented | not disclosed | — |
| Op Library | hipDNN / MIOpen (Conv2D, BatchNorm, Pooling, Attention, LSTM; cuDNN analog) | confirmed | dtk-portal, csdn-platform-libs |
| Op Library | hipBLAS (GEMM, BLAS L1–L3, batched GEMM; cuBLAS analog) | confirmed | dtk-portal, csdn-platform-libs |
| Op Library | hipSPARSE (sparse BLAS, SpMV; cuSPARSE analog) | confirmed | dtk-portal |
| Op Library | hipFFT (1D/2D/3D FFT; cuFFT analog) | confirmed | dtk-portal |
| Op Library | hipRAND (pseudo/quasi-random; cuRAND analog) | confirmed | dtk-portal |
| Kernel Library | hipThrust / hipCUB (Reduce, Scan, Sort, Histogram; Thrust/CUB analog) | confirmed | dtk-portal, csdn-dtk-install |
| Kernel Library | rocPRIM (low-level parallel primitives) | confirmed | dtk-portal |
| Kernel Library | rocBLAS (architecture-optimized GEMM kernels) | confirmed | dtk-portal |
| Runtime | HIP Runtime (hiprt / libhip.so; hipMalloc, hipMemcpy, hipStream, hipEvent, hipMallocManaged; cudart analog) | confirmed | zhihu-dtk-overview, csdn-dcu-faq |
| Runtime | ROCr / HSA Runtime (HSA agent model; lower-level; CUDA Driver API analog; rarely used directly) | confirmed | dtk-portal |
| Runtime | HAMi DCU vGPU sharing (resource partitioning at runtime level; Kubernetes integration) | confirmed | hami-dcu-docs, hami-dcu-sharing |
| Driver / Firmware | DCU kernel driver (.ko; PCIe BAR, IOCTL, DMA, IRQ; proprietary, not open-sourced) | confirmed | csdn-dtk-install |
| Driver / Firmware | HAMi dcu-vgpu-device-plugin (K8s device plugin; memory limits, core quotas, health monitoring) | confirmed | github-hami-dcu, hami-dcu-plugin |
| Communication | RCCL (DTK port; AllReduce, AllGather, ReduceScatter, Broadcast; NCCL analog) | confirmed | dtk-portal, gpustack-tutorial |
| Communication | xGMI peer-to-peer (intra-node HBM-to-HBM; AMD Infinity Fabric compatible; no PCIe traversal) | confirmed | vzkoo-y100-specs, amd-xgmi-blog |
| Assembler / ISA | DCU ISA (GCN/Vega-derived; not published independently; 64-thread wavefront; VGPR/SGPR/LDS; v_, s_, ds_, buffer_ instructions) | inferred | researchgate-dcu-block, springer-depthwise |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | CU (Compute Unit) — ~60 CUs @ 1.7 GHz (Z100), ~64 CUs (K100_AI); 4× SIMD16/CU = 64 FP32 ALUs/CU; FP64 supported at ~1/4 FP32 rate | confirmed | springer-depthwise, researchgate-dcu-block |
| Compute Engine | 深算一号 (DCU-Y100): 4,096 shader processors; 32 GB HBM2; 1 TB/s; FP32/FP16/INT8 | analyst-report | vzkoo-y100-specs |
| Compute Engine | 深算二号 (DCU-Z100 / K100_AI): 90 TFLOPS FP32 / 180 TFLOPS FP16; 350–400 W; 64 GB; FP8 optional | analyst-report | vzkoo-z100-specs, csdn-dcu-faq |
| Compute Engine | Z100: first commercial DCU; ~60 CUs; 16–32 GB HBM2; reference hardware for USTC benchmarks | confirmed | ustc-z100-specs |
| Compute Engine | K100_AI: 64 GB; 400 W; core 1% granularity via HAMi; from HAMi device registry — **prior** AI flagship, superseded by 深算三号 (BW1000) in 2025 (corrected 2026-08-08) | confirmed | hami-dcu-support |
| Compute Engine | 深算三号 (BW1000) — **current flagship**; Hygon 2025-12-11: "已经投入市场，受到客户认可"; CU count, clock, and all peak-throughput figures **not disclosed**; no vendor datasheet exists (hygon.cn publishes no DCU specs) | confirmed (existence + in-market status) / not disclosed (specs) | hygon-sse-investor-2025-12-11, hygon-product-page, ictnj-bw1000-deployment |
| Compute Engine | 深算三号 (BW1000) circulating figures FP32 49 / TF32 96 / BF16-FP16 192 TFLOPS / INT8 392 TOPS — **rejected**: Eastmoney 股吧 + Baidu Wenku origin, contradicted by ~20 TFLOPS FP32 (Zhihu) and FP64 ~30 TFLOPS (51CTO) for the same part | rejected — do not publish | guba-bw1000-post, zhihu-bw1000 |
| Compute Engine | 深算四号 — in R&D; Hygon 2025-12-11: "进展顺利"; no timeline, node, or specs disclosed | not disclosed | hygon-sse-investor-2025-12-11 |
| Data Path | 64-thread wavefront (vs NVIDIA 32-thread warp); 4 cycles to dispatch 1 wavefront across 4× SIMD16 | confirmed | springer-depthwise, csdn-dcu-programming |
| Data Path | SIMT pipeline: dispatch → SIMD16 issue → ALU execute → writeback; scalar unit for branch control; LDS access shared across CU | confirmed | researchgate-dcu-block |
| Data Path | Wavefront occupancy model: up to 40 wavefronts in-flight per CU for latency hiding (HBM latency) | inferred | springer-spgemm |
| On-chip Memory | LDS (Local Data Share) — 64 KB per CU; SW-managed via `__shared__` in HIP; 32-bank; workgroup scope | confirmed | springer-depthwise, csdn-dcu-faq |
| On-chip Memory | VGPR register file — 64 KB per SIMD (256× 32-bit/thread max); 4 SIMD groups/CU = 256 KB VGPR/CU | confirmed | springer-depthwise |
| On-chip Memory | SGPR register file — scalar registers per wavefront (~16 KB/CU est.) | inferred | csdn-dcu-programming |
| On-chip Memory | L2 Cache — ~4–8 MB chip-global (HW-managed; CU groups share slices) | inferred | researchgate-dcu-block |
| Off-chip Memory | Z100 — HBM2 16–32 GB, ~1 TB/s, 4-stack | confirmed | ustc-z100-specs |
| Off-chip Memory | Z100L — HBM2 32 GB, ~1 TB/s | confirmed | csdn-z100l-question |
| Off-chip Memory | K100_AI — 64 GB (est. HBM2e), ~1.2 TB/s est., 400 W | confirmed (capacity) | hami-dcu-support |
| Off-chip Memory | 深算一号 (Y100) — HBM2 32 GB, 1 TB/s, 4× HBM2 channels | analyst-report | vzkoo-y100-specs |
| Off-chip Memory | 深算三号 (BW1000) — HBM generation, capacity, and bandwidth **not disclosed**; the HBM2/HBM2e 16–64 GB @ ~1.0–1.2 TB/s figures above are scoped to Y100/Z100/K100 only | not disclosed | hygon-product-page |
| Host Interface / Package | PCIe Gen4 x16 host interface; dual-slot FHFL form factor; 2.5D interposer with HBM stacks | confirmed | hami-dcu-docs, h3c-server-cert |
| Host Interface / Package | OEM integrations: H3C R4900/R5300, Sugon (曙光) server series, Alibaba Cloud ECS. Sugon remains an **independent** OEM partner (603019.SH) — the Hygon absorption merger was terminated 2025-12-09 (corrected 2026-08-08) | confirmed | h3c-r5300-z100l, alibaba-dtk-images, caixin-merger-terminated |
| Host Interface / Package | 深算三号 (BW1000) — PCIe generation, form factor, TDP, and process node all **not disclosed** | not disclosed | hygon-product-page |
| Host Interface / Package | 深算三号 (BW1000) deployed on 中科南京信息高铁研究院 信息高铁智算算力网 AI platform — 国内首发, Dec 2025 | confirmed | ictnj-bw1000-deployment, qilinpark-news |
| Scale-up Interconnect | xGMI (AMD Infinity Fabric derivative); 深算一号 bandwidth 184 GB/s multi-card; peer-to-peer fully connected within node | confirmed | vzkoo-y100-specs |
| Scale-up Interconnect | 深算三号 (BW1000) — xGMI generation and scale-up bandwidth **not disclosed**; the 184 GB/s figure is a Y100 number and must not be carried forward | not disclosed | hygon-product-page |
| Scale-up Interconnect | xGMI protocol shared with AMD Instinct (MI200/MI300 series); same wire protocol enables direct HBM-to-HBM transfers | confirmed | amd-xgmi-blog, digitimes-hygon-roadmap |
| Scale-out Interconnect | Standard Ethernet / RoCE v2 via host NIC; no proprietary scale-out ASIC | confirmed | dtk-portal, gpustack-tutorial |
| Scale-out Interconnect | RCCL collective library for multi-node training; supports ring-based AllReduce over RoCE/Ethernet | confirmed | dtk-portal |
| Scale-out Interconnect | 深算三号 (BW1000) — no BW1000-specific fabric, NIC, or collective-offload change documented; scale-out **not disclosed** | not disclosed | — |

## Source Keys Added 2026-08-08

| Key | URL |
|-----|-----|
| hygon-sse-investor-2025-12-11 | https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml |
| hygon-product-page | https://www.hygon.cn/product/accelerator |
| ictnj-bw1000-deployment | https://www.ictnj.ac.cn/newsinfo/10874852.html |
| qilinpark-news | https://qilinpark.nanjing.gov.cn/xwzx/202512/t20251222_5748374.html |
| caixin-merger-terminated | https://www.caixin.com/2025-12-09/102391669.html |
| yicai-merger-terminated | https://www.yicai.com/news/102948899.html |
| baike-shensuan3 | https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890 |
| guba-bw1000-post *(rejected source)* | https://guba.eastmoney.com/news,688041,1483072345.html |
| zhihu-bw1000 *(rejected source)* | https://zhuanlan.zhihu.com/p/2066563534897652881 |
