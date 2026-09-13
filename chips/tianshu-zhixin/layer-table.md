# Iluvatar CoreX (天数智芯) TianGai GPU Layer Mapping Table

*as_of: 2026-08-08*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (via IXUCA device backend / libcuda shim; minimal code modification migration path) | confirmed | wikipedia-iluvatar, digitimes-iluvatar |
| Framework Integration | TensorFlow (via IXUCA compatibility layer) | confirmed | wikipedia-iluvatar |
| Framework Integration | PaddlePaddle (IXUCA support) | confirmed | grokipedia-iluvatar |
| Framework Integration | MindSpore (IXUCA support) | confirmed | grokipedia-iluvatar |
| Framework Integration | FlagPerf (BAAI benchmark framework; ResNet50 + LLM training on TianGai clusters) | confirmed | baai-flagperf-resnet50, elecfans-aquila2 |
| Compiler / IR | IXUCA Compiler (in-house GPU kernel compiler; targets proprietary Iluvatar ISA; architecture undisclosed) | confirmed | digitimes-iluvatar, wikipedia-iluvatar |
| Compiler / IR | IxRT Compiler (graph optimizer for inference; TensorRT analog; layer fusion + quantization; open-source) | confirmed | github-ixrt |
| Kernel-Authoring DSL | FlyDSL (Python DSL + MLIR stack for high-performance GPU kernels with explicit layouts and tiling; fork of an AMD-ROCm-origin project; Deep-Spark fork adds FlyIXDL backend for Iluvatar `ivcore11`-class parts MR-50/MR-100/BI-V150/BI-V150s; lineage = CUTLASS layout algebra + ROCm Composable Kernel tile patterns; explicitly lower-level than Triton; 873 commits on `iluvatar` branch; last update 2026-08-08) | reported | github-flydsl |
| Developer Portal | developer.iluvatar.com (launched ~2026-07-15/19; dev docs, software images, model cases, tech blogs, DeepSpark resources; company claims ~100 acceleration libraries built since 2018 across communication/compilation/drivers/quantization/perf analysis) | reported | iluvatar-dev-portal, tencent-cloud-waic2026 |
| Op Library | ixnn (cuDNN analog; Conv, SDPA/Attention, Norm, Pooling, Activation; FP32/FP16/BF16/INT8) | confirmed | riseunion-hami-iluvatar |
| Op Library | ixblas (cuBLAS analog; GEMM, BLAS L1-L3; optimized for Iluvatar tensor compute units) | confirmed | riseunion-hami-iluvatar |
| Inference Runtime | IxRT — Iluvatar Inference Runtime (TensorRT analog; AI compiler + inference engine + plugin system + deploy tools; open-source Apache 2.0; `Deep-Spark/iluvatar-corex-ixrt`) | confirmed | github-ixrt, deepspark-ixrt |
| Inference Runtime | IGIE — Iluvatar GPU Inference Engine (complementary framework-specific inference engine; TF/PaddlePaddle) | confirmed | deepspark-ixrt |
| Inference Runtime | DeepSparkInference model zoo — 216 curated inference models as of 2026-08-06 (up from prior baseline); `Deep-Spark/DeepSparkInference` | reported | deepspark-github-org |
| Inference Runtime | Deep-Spark vLLM fork (updated 2026-06-12); xllm — multi-accelerator LLM inference engine (added 2026-05-12); LightX2V — video-generation inference (added 2026-05-12) | reported | deepspark-github-org |
| Runtime | IXUCA Runtime / ix-runtime / libcuda shim (cudart analog; ixMalloc/ixFree/ixMemcpy; stream + event management; exposes CUDA-compatible API; enables migration without source rewrite) | confirmed | ix-container-toolkit, hami-iluvatar |
| Runtime | IXUCA Driver API (lower-level context/module management) | confirmed | wikipedia-iluvatar |
| Driver / Firmware | Iluvatar GPU Kernel Driver (.ko; Linux PCIe; BAR/IOCTL/DMA/IRQ; Ubuntu x86 + Kylin/UOS domestic Linux) | confirmed | ix-container-toolkit |
| Driver / Firmware | ix-container-toolkit (Docker/containerd GPU container runtime; ix-container-runtime hook; open-source Apache 2.0; `Deep-Spark/ix-container-toolkit`) | confirmed | github-ix-container-toolkit |
| Driver / Firmware | ix-device-plugin (Kubernetes DaemonSet; GPU resource scheduling; health monitoring; multi-tenancy) | confirmed | hami-iluvatar, riseunion-hami-iluvatar |
| Driver / Firmware | ix-exporter (Prometheus HTTP metrics server for Iluvatar GPU nodes; open-source; `Deep-Spark/ix-exporter`) | confirmed | github-ix-exporter |
| Communication | CLIF (Cluster Interconnect Fabric; proprietary intra-node GPU-to-GPU direct interconnect; NVLink analog; bandwidth undisclosed) | confirmed | leiphone-iluvatar |
| Communication | NCCL-compatible CCL (inter-node collective communication over InfiniBand / RoCE Ethernet) | inferred | wikipedia-iluvatar |
| Debugging / Profiling | ixGDB (GPU debugger; CUDA-GDB 10.2 derived; open-source GPLv3; source-level GPU kernel debugging; `Deep-Spark/ixGDB`) | confirmed | github-ixgdb |
| Debugging / Profiling | ixSMI (GPU status monitor; nvidia-smi analog; utilization, memory, temperature, power) | confirmed | csdn-tianshu-tools |
| Debugging / Profiling | ixPROF (GPU profiler; nsight analog; kernel-level timing and occupancy) | confirmed | csdn-tianshu-tools |
| Debugging / Profiling | ixKN (kernel trace utility) | confirmed | csdn-tianshu-tools |
| Debugging / Profiling | ixSYS (system diagnostics and health check) | confirmed | csdn-tianshu-tools |
| Assembler / ISA | Iluvatar Proprietary SIMT ISA (fully in-house; not publicly documented; warp-based SIMT; FP32/FP16/BF16/INT8; no PTX equivalent) — Big Island generation | confirmed | digitimes-iluvatar, wikipedia-iluvatar |
| Assembler / ISA | New-generation ISA (TianGai 300, announced 2026-07-19): SIMT retained with scalar/vector/tensor paths; adds FP8 and FP4; MMA / DPX / FMA instruction classes; still undocumented publicly | announced | tencent-news-tg300-a, ithome-tg300 |
| Assembler / ISA | ixSMEX extension (TianGai 300) — raises data reuse to cut redundant memory traffic; encoding not disclosed | announced | tencent-news-tg300-a |
| Assembler / ISA | ixDPX extension (TianGai 300) — compresses a six-instruction dynamic-programming sequence into one instruction; nearest analogue NVIDIA Hopper DPX; encoding not disclosed | announced | tencent-news-tg300-a, tencent-news-tg300-b |
| Assembler / ISA | ixTrans extension (TianGai 300) — lossless matrix transpose; reduces bank conflicts and VRAM overhead; encoding not disclosed | announced | tencent-news-tg300-a |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | TianGai-100 (BI-V100): TSMC 7nm CoWoS; 32 GB HBM2; ~80-100 TFLOPS FP16; TPP density 2,352; PCIe Gen4; released Jan 2021; China's first mass-produced 7nm GPGPU | confirmed | wikipedia-iluvatar, exportsemi-tiangai100 |
| Compute Engine | TianGai-150 (BI-V150): TSMC 7nm CoWoS; 64 GB HBM2; ~120-150 TFLOPS FP16; TPP density 3,040; PCIe Gen4 FHFL; current training flagship | confirmed | csdn-tiangai150, riseunion-hami-iluvatar |
| Compute Engine | Zhikai 100 (智铠100; product code MR-V100): Purpose-built GPGPU inference chip; China's first; 32 GB GDDR6 (est.); enhanced INT8 units; optimized data paths for low-latency inference. Vendor site now lists the part as 智铠100, not "MR-V100" | confirmed | wikipedia-iluvatar, iluvatar-site |
| Compute Engine | **TianGai 300 (天垓300), announced 2026-07-19 at WAIC 2026**: first product on a new-generation self-developed architecture; SIMT retained with scalar/vector/tensor paths; adds FP8 and FP4; MMA/DPX/FMA instruction classes; ixSMEX/ixDPX/ixTrans ISA extensions. **Process, foundry, die size, transistor count, packaging, memory type/capacity/BW, peak FLOPS/TOPS at every dtype, host interface, TDP, form factor and card name are ALL not disclosed.** Status: announced ("已具备规模化应用条件"); no tape-out/sampling/mass-production disclosure | announced | iluvatar-site, tencent-news-tg300-a, tencent-news-tg300-b, ithome-tg300, semi-insights-tg300 |
| Compute Engine | 彤央 (Tongyang) TY series edge/endpoint line — TY1000, TY1100, TY1100-NX, TY1100-NX-PRO, TY1200; listed on the vendor site; no specifications retrieved (known survey gap) | announced | iluvatar-site |
| Compute Engine | Tianshu (天枢) arch 2025: Claims to exceed Hopper performance; 300+ benchmark clients deployed; >90% effective compute utilization (company claim) | announced | tomshardware-roadmap, trendforce-roadmap |
| Compute Engine | Tianxuan (天璇) arch 2026: Targets NVIDIA Blackwell B200 performance — ⚠️ roadmap codename only; no retrieved source ties any announced product (including TianGai 300) to this name | unverified | tomshardware-roadmap, trendforce-roadmap |
| Compute Engine | Tianji (天玑) arch 2026: Claims to surpass Blackwell — ⚠️ roadmap codename only; unverified | unverified | trendforce-roadmap-jan28 |
| Compute Engine | Tianquan (天权) arch 2027: Targets NVIDIA Rubin R100 | announced | tomshardware-roadmap, trendforce-roadmap |
| Data Path | SIMT execution model: warp-based parallelism with hardware divergence; analogous to NVIDIA SIMT | confirmed | digitimes-iluvatar, wikipedia-iluvatar |
| Data Path | Tensor Compute Units: hardware matrix acceleration for FP16/BF16/INT8 operations (Big Island generation) | confirmed | wikipedia-iluvatar |
| Data Path | TianGai 300 tensor path: FP8 and FP4 reduced precision added; MMA/DPX/FMA instruction classes. Tensor unit dimensions, count, and clock not disclosed | announced | tencent-news-tg300-a, tencent-news-tg300-b |
| Data Path | Full-stack in-house design: ISA + chip architecture + foundational software (company claim) | confirmed | digitimes-iluvatar |
| On-chip Memory | L2 cache (hardware-managed; capacity not publicly disclosed) | confirmed | wikipedia-iluvatar |
| On-chip Memory | Per-CU local shared memory / L1 scratchpad (capacity not publicly disclosed) | inferred | standard_gpgpu_arch |
| Off-chip Memory | HBM2 (TianGai-100): 32 GB, ~1.2 TB/s | confirmed | exportsemi-tiangai100 |
| Off-chip Memory | HBM2 (TianGai-150): 64 GB, ~1.2–1.6 TB/s | confirmed | csdn-tiangai150 |
| Off-chip Memory | GDDR6 (Zhikai 100 / MR-V100): 32 GB est., ~512 GB/s est. | inferred | standard_inference_gpu |
| Off-chip Memory | TianGai 300: memory **type, capacity and bandwidth all not disclosed** — no vendor or press source states HBM generation, capacity, stack count or bandwidth. Do not estimate | not_disclosed | iluvatar-site, tencent-news-tg300-b |
| On-chip Memory | TianGai 300: L1 scratchpad and L2 capacity **not disclosed** | not_disclosed | iluvatar-site |
| Host Interface / Package | PCIe Gen4 x16 FHFL (training series); standard host interface | confirmed | csdn-tiangai150 |
| Host Interface / Package | TSMC 7nm + 2.5D CoWoS packaging (TianGai training series) | confirmed | wikipedia-iluvatar |
| Scale-up Interconnect | CLIF (Cluster Interconnect Fabric): proprietary direct GPU-to-GPU links; bandwidth undisclosed; 8-GPU server configurations (Big Island generation) | confirmed | leiphone-iluvatar |
| Scale-up Interconnect | **天数超节点 (Tianshu supernode), announced 2026-07-19**: claimed **144-chip high-speed full interconnect** — the first scale-up domain size Iluvatar has ever stated. **Topology, per-link/per-chip bandwidth, switch ASIC, single-coherent-domain status, rack power, and whether it extends CLIF or is a new fabric are all not disclosed.** Explicitly at trial / partnership-discussion stage | announced | tencent-news-tg300-a, tencent-cloud-waic2026 |
| Host Interface / Package | TianGai 300: host interface generation, packaging and form factor **not disclosed** | not_disclosed | iluvatar-site |
| Scale-out Interconnect | Standard Ethernet / InfiniBand via host NIC (inter-node) | inferred | wikipedia-iluvatar |
| Scale-out Interconnect | NCCL-compatible collective communication for multi-node training | inferred | standard_multi_gpu_training |

---

## Source keys added 2026-08-08

| Key | Reference |
|-----|-----------|
| iluvatar-site | https://www.iluvatar.com/ — vendor product navigation listing 天垓300 (`cpjs-yj-xlxl-tg300`), 智铠100, and the 彤央 TY series. No spec sheet published; vendor news feed unchanged since 2021 |
| tencent-news-tg300-a | https://news.qq.com/rain/a/20260719A08MB800 — Tencent News, 2026-07-19 |
| tencent-news-tg300-b | https://news.qq.com/rain/a/20260719A08J8N00 — Tencent News, second piece, 2026-07-19 |
| ithome-tg300 | https://www.ithome.com/0/978/781.htm — IT之家 |
| tencent-cloud-waic2026 | https://cloud.tencent.com/developer/news/4280275 — Tencent Cloud developer news |
| semi-insights-tg300 | https://www.semi-insights.com/s/bdt/15/50538.shtml — 半导体行业观察 |
| github-flydsl | https://github.com/Deep-Spark/FlyDSL — `iluvatar` branch, last updated 2026-08-08 |
| deepspark-github-org | https://github.com/orgs/Deep-Spark/repositories?sort=updated — repository activity as of 2026-08-08 |
| iluvatar-dev-portal | https://developer.iluvatar.com — developer portal, launched ~2026-07-15/19 |

**Confidence vocabulary note.** `announced` = vendor has stated the item exists but has published no verifiable specification; `not_disclosed` = the survey has actively checked and found no public figure — this is a positive finding, not a placeholder to be filled by estimation. `unverified` = the claim appears in secondary coverage but no source connects it to a real product.
