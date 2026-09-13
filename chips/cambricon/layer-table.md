# Cambricon MLU Layer Mapping Table

*as_of: 2026-08-08*
*chip: cambricon*
*device_class: Neural Processor*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | torch_mlu (PyTorch backend: ATen dispatch to CNNL, torch.device("mlu"), CNCL distributed) | confirmed | torch_mlu, cntoolkit |
| Framework Integration | vllm-mlu (LLM inference serving: Chunk Prefill, Prefix Cache, Speculative Decode, Graph Mode, Sleep Mode; SDK 25.08+; hardware requirement still "MLU370以上的设备") | confirmed | vllm-mlu |
| Framework Integration | vllm-mlu day-0 DeepSeek-V4 (2026-04-24): V4-flash 285B + V4-pro 1.6T; TP/PP/SP/DP/EP 5-D hybrid parallelism; comm/compute overlap; low-precision quant; PD-separated (prefill/decode) deployment | confirmed | vllm-mlu, cambricon-deepseek-v4-article |
| Framework Integration | PaddlePaddle custom device plugin (libpaddle-custom-mlu.so, 264+ custom ops, MLU370-X8) | confirmed | paddlecustomdevice |
| Framework Integration | CATCH (legacy PyTorch integration layer; superseded by torch_mlu) | confirmed | catch |
| Compiler / IR | BANG C + CNCC (kernel DSL: __mlu_global__, __nram__, __wram__, __sync_cluster(); CNCC → MLISA assembly) | confirmed | bangc-guide, cntoolkit |
| Compiler / IR | CNAS Assembler (MLISA → .cnbin/.cnfatbin; fat binary for multi-generation MLU) | confirmed | bangc-guide |
| Compiler / IR | MagicMind (MLIR inference compiler: TF/PyTorch/ONNX/Caffe → fused+quantized MLU binary) | confirmed | magicmind-cloud |
| Compiler / IR | triton-linalg (Triton MLIR dialect → Linalg dialect; Cambricon backend; frontend complete; **no public push since 2025-02-07** — backend work stalled in public or moved into the licensed SDK; H1 filing says Cambricon continues tracking Triton community releases) | confirmed | triton-linalg, github-org-api, cninfo-h1-2026 |
| Op Library | CNNL — Cambricon Neuware Neural Network Library (GEMM, Conv2D, BN, Attn, Pooling, RNN; C API) | confirmed | cntoolkit, torch_mlu |
| Op Library | CNNL_Extra (extended ML ops beyond core CNNL; torch_mlu dependency) | confirmed | torch_mlu |
| Op Library | Torch-MLU-Ops / torch_mlu_ops (Cambricon self-developed high-performance **fused** operator library for PyTorch; backend for vLLM, TGI, Stable Diffusion WebUI; accelerates DeepSeek-V4 Compressor + mHC; v1.3.2 2026-02-04 vs Torch-MLU v1.24.1 / CNNL v1.28.3; shipped in licensed SDK, no Cambricon public repo) | confirmed | torch-mlu-ops-v132, cambricon-deepseek-v4-article |
| Op Library | CNML v7.10.2 (legacy graph-based op library; superseded by CNNL) | confirmed | cnml-guide |
| Kernel Library | mlu-ops / BANGC OPS (open-source BANG C kernels; 100s of ML ops; CNNL integration helpers; actively maintained on master — last push 2026-08-07; last GitHub release v1.8.1 2026-01-06, future releases not published there) | confirmed | mlu-ops, github-org-api |
| Kernel Library | Hand-written BANG C kernels for DeepSeek-V4 sparse/compressed attention and GroupGemm | confirmed | cambricon-deepseek-v4-article |
| Kernel Library | BANGPy (Python-level kernel DSL; graph/kernel/operator-level custom MLU kernel authoring) | confirmed | neuware-sdk |
| Kernel Library | CNCV — Cambricon Computer Vision Library (decode, resize, color convert, normalize) | confirmed | torch_mlu |
| Runtime | CNRT — libcnrt.so (Queue/Notifier model; cnrtMalloc/cnrtFree; cnrtInvokeKernel; cnrtDim3_t) | confirmed | cnrt-guide |
| Runtime | CNDrv — libcndrv.so (driver API: context management, .cnbin module loading, fine-grained memory) | confirmed | cndrv-crate, cnrt-guide |
| Runtime | CNToolkit bundle (packages CNCC, CNAS, CNRT, CNDrv, CNNL, CNCL; versioned SDK distribution). Current requirement **CNToolkit ≥ v4.1.0 / CNNL ≥ v1.28.0**; supersedes v3.7.2 / SDK 1.15.0. Matching SDK bundle number **not disclosed** | confirmed | mlu-ops-readme |
| Driver / Firmware | cambricon-mlu.ko (Linux kernel driver: PCIe BAR, IOCTL dispatch, DMA engine, interrupt handling). Current requirement **driver ≥ v6.0.3** | confirmed | gitee-mlu-driver, mlu-ops-readme |
| Driver / Firmware | cambricon-k8s-device-plugin (Kubernetes DaemonSet for MLU resource reporting and health) | confirmed | k8s-device-plugin |
| Driver / Firmware | mlu-exporter (Prometheus metrics exporter for MLU device observability) | confirmed | mlu-exporter |
| Communication | CNCL (AllReduce/AllGather/ReduceScatter/Broadcast over MLU-Link; torch.distributed backend) | confirmed | kaitian-paper, torch_mlu |
| Communication | Gloo (cross-vendor heterogeneous communication in KAITIAN framework) | confirmed | kaitian-paper |
| Assembler / ISA | MLISA — Machine Learning ISA (64-bit load-store; scalar/vector/matrix/control; 64 GPRs/core) | confirmed | bangc-guide, cambricon-isa-2016 |
| Assembler / ISA | CNAS — Cambricon Neuware Assembler (.mlisa → .cnbin/.cnfatbin fat binary) | confirmed | bangc-guide |
| Assembler / ISA | Cambricon ISA specification (ISCA 2016: 43 instruction types; load-store NN ISA) | confirmed | cambricon-isa-2016, cambricon-isa-acm |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | MLU Core (IPU Core): FU scalar/vector/matrix, 64 GPRs, private NRAM+WRAM scratchpads | confirmed | wikichip, bangc-guide |
| Compute Engine | Cluster: 4 MLU Cores + Memory Core DMA + Shared SRAM; __sync_cluster() barrier | confirmed | wikichip, bangc-guide |
| Compute Engine | MLU290: 8 clusters / 32 cores, 512 INT8 TOPS, 64 FP32 TFLOPS, TSMC 7nm | confirmed | fcc-mlu370, wikichip |
| Compute Engine | MLU370 (Siyuan 370, MLUarch03): ~32+ cores, ~256 INT8 TOPS est., FP32/FP16/BF16/INT8/INT4 | inferred | wikichip, laitimes |
| Compute Engine | MLU270: 4 clusters / 16 cores, TSMC 16nm (inference) | confirmed | wikichip |
| Compute Engine | MLU690 (Siyuan 690): reported >700 FP16 TFLOPS / >2,800 INT4 TOPS, dual-die-chiplet package; cluster+core counts, precision list and process node all **not disclosed** (sources contradict: TSMC 4nm vs 7nm) | reported-unverified | sina-brokerage-690, 51cto-690, tianyancha-690 |
| Data Path | SPMD execution: all MLU Cores run same BANG C kernel; independent instruction streams; no SIMT warps | confirmed | bangc-guide, wikichip |
| Data Path | Memory Core DMA engine (per-cluster, independent of compute Cores; enables double-buffering) | confirmed | bangc-guide, wikichip |
| Data Path | BANG C barriers: __sync_cluster() intra-cluster, __sync_all() chip-wide | confirmed | bangc-guide |
| On-chip Memory | NRAM (Neural-RAM): per-core private scratchpad for activation data; __nram__ qualifier; programmer-managed | confirmed | bangc-guide |
| On-chip Memory | WRAM (Weight-RAM): per-core private scratchpad for conv weights; feeds matrix FU directly | confirmed | bangc-guide |
| On-chip Memory | Shared SRAM: per-cluster; staging buffer + inter-core exchange; __mlu_shared__ qualifier | confirmed | bangc-guide, wikichip |
| On-chip Memory | LLC: hardware-managed read-only cache on GDRAM path; shared weight broadcast | inferred | bangc-guide |
| Off-chip Memory | LPDDR5 (MLU370-X8): 48 GB / 614.4 GB/s (dual-chip); China's first cloud AI chip with LPDDR5 | confirmed | laitimes, fcc-mlu370 |
| Off-chip Memory | HBM2 (MLU290-M5): 32 GB / 1,228 GB/s | confirmed | fcc-mlu370, wikichip |
| Off-chip Memory | HBM (MLU590): generation TBD, capacity and bandwidth TBD; supply-constrained by US export controls | inferred | tomshardware-cambricon |
| Off-chip Memory | HBM3 (MLU690): reported 196 GB / ~3.35 TB/s; no Cambricon datasheet, and HBM is unmentioned in the 2026 H1 filing | reported-unverified | sina-brokerage-690, 51cto-690 |
| Host Interface / Package | PCIe Gen4 x16: ~64 GB/s bidir; FHFL card form factor (MLU370-M8, MLU370-X, MLU290) | confirmed | fcc-mlu370 |
| Host Interface / Package | Chiplet design (research): 2.5D chiplet packaging for 70B LLM on-device inference (arXiv 2409.15654) | inferred | cambricon-llm-chiplet |
| Scale-up Interconnect | MLU-Link: proprietary chip-to-chip interconnect; dual Siyuan 370 on MLU370-X8 | confirmed | laitimes, kaitian-paper |
| Scale-up Interconnect | 8-card MLU-Link scale-up: server backplane; ~155% of 350W RTX GPU on BERT/YOLO/ResNet | confirmed | kaitian-paper, copyfuture |
| Scale-up Interconnect | MLU690 interconnect: reported >890 **Gbps** (≈111 GB/s) — gigabits, not gigabytes; the circulating "890 GB/s" restatement is wrong by ~8× | reported-unverified | sina-brokerage-690 |
| Scale-out Interconnect | Standard Ethernet / InfiniBand via host NICs (no proprietary scale-out NIC) | confirmed | kaitian-paper |
| Scale-out Interconnect | CNCL for intra-node MLU collectives over MLU-Link | confirmed | kaitian-paper |
| Scale-out Interconnect | Gloo for cross-vendor heterogeneous communication (MLU + non-MLU accelerators) | confirmed | kaitian-paper |

## Confidence Legend

| Value | Meaning |
|-------|---------|
| confirmed | Backed by a vendor primary source or an independently verifiable artifact (repo README, filing, FCC document, peer-reviewed paper) |
| inferred | Derived from published figures or architectural reasoning; not directly stated by the vendor |
| reported-unverified | Circulating in secondary media/brokerage coverage with **no vendor source**; recorded for completeness only. All MLU690 rows are in this class as of 2026-08-08 |

## Source Keys Added 2026-08-08

| Key | Reference |
|-----|-----------|
| cninfo-h1-2026 | Cambricon 2026 半年度报告 — https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF |
| cambricon-deepseek-v4-article | https://developer.cambricon.com/index/article/details.html?id=12 (2026-04-24) |
| mlu-ops-readme | https://raw.githubusercontent.com/Cambricon/mlu-ops/master/README.md |
| github-org-api | https://api.github.com/orgs/Cambricon/repos?sort=pushed&direction=desc |
| torch-mlu-ops-v132 | torch_mlu_ops v1.3.2 release (2026-02-04) — https://dev.modelhub.org.cn/Chranos/enginex-mlu370-vllm/src/tag/v0.0.6/torch_mlu_ops-v1.3.2 |
| sina-brokerage-690 | https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/lastest/rptid/831133396021/index.phtml |
| 51cto-690 | https://blog.51cto.com/u_10819805/14632220 |
| tianyancha-690 | https://news.tianyancha.com/ll_5k94fbk79z.html |
