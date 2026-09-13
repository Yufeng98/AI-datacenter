# Iluvatar CoreX (天数智芯) TianGai GPU Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*

---

## Overview

Shanghai Tianshu Zhixin Semiconductor Co., Ltd. (天数智芯; Iluvatar CoreX), founded in **2015** in Shanghai by **Li Yunpeng** (李仲远, former Oracle R&D Director), is China's first and only publicly listed general-purpose GPU (GPGPU) company. Unlike other Chinese AI chip startups that build inference-only ASICs, Iluvatar CoreX designs fully programmable SIMT GPUs for cloud AI training and inference — China's most direct analogue to NVIDIA's data-center GPU business model.

The company's shipping training flagship is the **TianGai-150 (BI-V150)** (天垓150 — TSMC 7nm CoWoS, 64 GB HBM2), and its inference line is the **Zhikai 100** (智铠100; product code **MR-V100** — the vendor site now lists the part as 智铠100). On **2026-07-19** the company announced **TianGai 300** (天垓300), the first product on a new-generation self-developed architecture, alongside a **144-chip "Tianshu supernode" (天数超节点)** scale-up system — see the dated update section below. TianGai 300 is **announced, not confirmed shipping**; TianGai-100/150 remain the shipping products.

The software stack is called **IXUCA** (Iluvatar CoreX Unified Computing Architecture), an explicitly CUDA-compatible full-stack platform with open-source tooling published under the **DeepSpark** community (`github.com/Deep-Spark`).

On **January 8, 2026**, Iluvatar CoreX listed on the **Hong Kong Stock Exchange (HKEX)** (ticker **09903.HK**), becoming the first Chinese GPGPU company to go public in Hong Kong. It raised **USD 475 million** (~HK$3.7B) at a market cap of **USD 4.6–5.3 billion**. Full-year 2025 revenue reached **RMB 1.03 billion** — a **92% year-on-year increase**. Deployment figures are company-stated and must be read with their as-of dates: **52,000+ GPUs cumulatively delivered to 290+ customers as of June 2025** (IPO-prospectus era figure), superseded at the WAIC 2026 briefing by **340+ customers across 30+ industries as of end-2025** with **>10,000 chips in online cluster operation** and a claimed **1,000+ day** continuous cluster uptime. The two metrics are not directly comparable (cumulative units shipped vs. chips concurrently online) and neither is independently audited.

---

## Software Stack

### Framework Integration

- **PyTorch, TensorFlow, PaddlePaddle, MindSpore** — supported via IXUCA device backend. The IXUCA runtime exposes a `libcuda`-compatible interface, enabling most CUDA-targeting framework code to run on Iluvatar GPUs with minimal modification.
- **FlagPerf** (BAAI / 智源研究院) — benchmark framework validated on TianGai clusters; used for ResNet50 and LLM training benchmarks.

### Compiler / IR

- **IXUCA Compiler** — in-house GPU kernel compiler targeting the proprietary Iluvatar ISA; architectural details not publicly disclosed.
- **IxRT Compiler** (open-source, `github.com/Deep-Spark/iluvatar-corex-ixrt`) — graph-level optimizer for inference; performs layer fusion and quantization calibration. Analogous to NVIDIA TensorRT.

### Kernel-Authoring DSL — *new, added 2026-08-08*

- **FlyDSL** (open-source, `github.com/Deep-Spark/FlyDSL`, `iluvatar` branch, last updated 2026-08-08) — a **Python DSL plus MLIR stack for authoring high-performance GPU kernels with explicit layouts and tiling**. The Deep-Spark repository is a fork of an AMD-ROCm-origin project; the fork adds a **FlyIXDL backend for Iluvatar GPUs** covering `ivcore11`-class parts (MR-50, MR-100, BI-V150, BI-V150s). Design lineage is cited as **CUTLASS layout algebra + ROCm Composable Kernel tile patterns**, and the project positions itself as explicitly **lower-level than Triton** (the programmer states layouts and tiles rather than relying on an autotuner). 873 commits on the `iluvatar` branch. This is the first CUTLASS/Triton-tier kernel-authoring layer Iluvatar has published, and it fills a gap the survey's layer table previously recorded as empty for this vendor.

### Op Library

- **ixnn** — cuDNN analog. Convolution, Attention (SDPA), Normalization, Pooling, Activation in FP32/FP16/BF16/INT8.
- **ixblas** — cuBLAS analog. GEMM and BLAS L1–L3 optimized for Iluvatar tensor compute units.

### Inference Runtime

- **IxRT** (open-source) — high-performance inference engine with C++ and Python APIs, custom plugin system, and model deploy tools.
- **IGIE** (Iluvatar GPU Inference Engine) — complementary inference framework for TensorFlow/PaddlePaddle graph execution.

### Runtime

- **IXUCA Runtime** (`ix-runtime` / `libcuda` shim) — cudart analog. Exposes CUDA-compatible memory (`ixMalloc`/`ixFree`), stream, and event APIs. Enables CUDA code migration without source rewriting.
- **IXUCA Driver API** — explicit context and module management layer.

### Driver / Firmware

- **Iluvatar GPU Kernel Driver (.ko)** — Linux PCIe module. BAR mapping, IOCTL, DMA, interrupts. Supports Ubuntu x86 and Kylin/UOS domestic Linux.
- **ix-container-toolkit** (open-source) — Docker/containerd GPU container runtime; `ix-container-runtime` hook analogous to nvidia-container-runtime.
- **ix-device-plugin** — Kubernetes DaemonSet for GPU resource scheduling and health monitoring.
- **ix-exporter** (open-source) — Prometheus metrics HTTP server for Iluvatar GPU nodes.

### Communication

- **CLIF (Cluster Interconnect Fabric)** — proprietary intra-node GPU-to-GPU direct interconnect (NVLink analog); bandwidth not publicly disclosed.
- **NCCL-compatible CCL** — inter-node collective communication over InfiniBand or RoCE Ethernet.

### Debugging / Profiling

- **ixGDB** (open-source) — GPU debugger based on CUDA-GDB 10.2. Source-level GPU kernel debugging.
- **ixSMI** — GPU status monitor (nvidia-smi analog).
- **ixPROF** — GPU profiler; **ixKN** — kernel trace; **ixSYS** — system diagnostics.

### ISA

- **Iluvatar Proprietary SIMT ISA** — fully in-house; not publicly documented. SIMT warp-based execution model. Supports FP32/FP16/BF16/INT8 on Big Island generation. No PTX-equivalent public virtual ISA.
- **New-generation ISA (TianGai 300, announced 2026-07-19)** — retains SIMT with scalar, vector and tensor execution; adds **FP8 and FP4** reduced precision and **MMA / DPX / FMA** instruction classes, plus three named extensions **ixSMEX**, **ixDPX** and **ixTrans**. Still undocumented publicly. See the dated update section below.

### Developer Resources

- **Iluvatar developer portal** (`developer.iluvatar.com`, launched around 2026-07-15/19) — aggregates developer documentation, software images, model cases, technical blogs and DeepSpark resources in one place. The company states it has built **~100 acceleration libraries since 2018** spanning communication, compilation, drivers, quantization and performance analysis (company claim; the libraries are not individually enumerated publicly).

---

## Hardware Architecture

### Compute Engine (TianGai-150 / BI-V150 flagship)

The TianGai GPU is a SIMT GPGPU organized into shader clusters analogous to NVIDIA SMs:

- **SIMT execution** with warp-based parallelism and hardware divergence
- **Tensor Compute Units** for FP16/BF16/INT8 matrix acceleration
- FP32 / FP16 / BF16 / INT8 ALUs
- Per-CU local shared memory (capacity not disclosed)
- On-chip L2 cache (capacity not disclosed, hardware-managed)
- **24 billion transistors estimated** (TSMC 7nm)

| Metric | TianGai-100 | TianGai-150 |
|--------|------------|------------|
| Peak FP16 | ~80–100 TFLOPS | ~120–150 TFLOPS |
| Peak INT8 | ~160–200 TOPS | ~240–300 TOPS |
| Memory | 32 GB HBM2 | 64 GB HBM2 |
| Memory BW | ~1.2 TB/s | ~1.2–1.6 TB/s |
| TDP | ~350W | ~400W |
| Process | TSMC 7nm CoWoS | TSMC 7nm CoWoS |

### Memory Hierarchy

```
Per-CU: Local shared memory (scratchpad; capacity undisclosed)
         ↓
On-chip: L2 cache (hardware-managed; capacity undisclosed)
         ↓
Off-chip: HBM2 (training) / GDDR6 (inference)
  TianGai-100: 32 GB HBM2, ~1.2 TB/s
  TianGai-150: 64 GB HBM2, ~1.2–1.6 TB/s
  Zhikai MR-V100: 32 GB GDDR6, ~512 GB/s (est.)
```

### Interconnect

| Layer | Detail |
|-------|--------|
| Host PCIe | Gen 4 x16 FHFL |
| Scale-up | CLIF (proprietary; bandwidth undisclosed) |
| Max per server | 8 GPUs |
| Scale-out | Ethernet / InfiniBand via host NIC |

---

## Roadmap

| Architecture / Product | Year | NVIDIA Target | Status |
|-------------|------|--------------|--------|
| TianGai-100 / Big Island | 2021 | A100-class (claimed) | GA, shipped |
| TianGai-150 / Big Island Gen2 | 2022 | A100+ | GA, flagship (current shipping training part) |
| Zhikai 100 (智铠100 / MR-V100) | 2022 | Inference-only | GA |
| Tianshu (天枢) | 2025 | Claims to exceed H100/H200 | Deployed, 300+ benchmark clients |
| **TianGai 300 (天垓300)** | **2026** | **Benchmarked vs Hopper (vendor claims); not positioned vs B200** | **Announced 2026-07-19 (WAIC 2026); "meets conditions for large-scale deployment"; no tape-out/sampling/mass-production disclosure** |
| Tianxuan (天璇) | 2026 | Targets B200 (Jan 2026 roadmap coverage) | Announced — ⚠️ *no retrieved source maps TianGai 300 onto this codename; the Tianxuan → B200 mapping remains unverified* |
| Tianji (天玑) | 2026 | Claims to surpass B200 (Jan 2026 roadmap coverage) | Announced — ⚠️ *unverified; no product has been tied to this codename* |
| Tianquan (天权) | 2027 | Targets Rubin R100 | Announced |

> **Codename caution (2026-08-08).** The Big-Dipper roadmap names (天枢/天璇/天玑/天权) come from January 2026 press coverage of an investor-facing roadmap. The product actually launched in 2026 is **天垓300**, a name from the 天垓 *product* line, and its vendor benchmarks are all against **Hopper**, not Blackwell. No retrieved source states that TianGai 300 is the productization of Tianxuan or of any other Big-Dipper architecture. The survey deliberately keeps the two naming systems separate rather than assuming a mapping.

---

## TianGai 300 Update — WAIC 2026 (2026-07-19)

*Updated 2026-08-08. Primary artifact: the vendor's own product navigation, which now lists 天垓300 in the 天垓 training series alongside 天垓150 and 天垓100 (https://www.iluvatar.com/, product node `cpjs-yj-xlxl-tg300`) — **with no specification sheet published**. Iluvatar's website news feed has not been updated since 2021, so there is no vendor press release for this launch. All narrative detail traces to Chinese press coverage of the WAIC session (Tencent News ×2, IT之家, Tencent Cloud developer news, 半导体行业观察, AAStocks on 09903.HK), which appears to derive from a common company briefing — many outlets, low true independence.*

Iluvatar CoreX announced **TianGai 300 (天垓300)**, its new-generation general-purpose GPU flagship, on **July 19, 2026** — day 3 of **WAIC 2026** (World AI Conference, Shanghai, **July 17–20, 2026**). The frequently repeated "July 18" date is not supported by any retrieved source.

### What is confirmed

- **First product on Iluvatar's new-generation self-developed architecture.** Company and all coverage describe it as an architectural generation change, not a derivative of the 7nm "Big Island" TianGai-100/150 line. The **SIMT general-purpose compute model is retained**, with scalar, vector and tensor execution paths.
- **FP8 and FP4 reduced precision**, alongside **MMA / DPX / FMA** instruction classes. This is the first Iluvatar part for which FP8/FP4 is claimed; the survey's prior FP32/FP16/BF16/INT8-only description applies to the Big Island generation and remains correct for it.
- **Three named ISA extensions** — the only concrete microarchitectural detail disclosed:

  | Extension | Stated function |
  |---|---|
  | **ixSMEX** | Raises data reuse to cut redundant memory traffic |
  | **ixDPX** | Compresses a six-instruction dynamic-programming sequence into one instruction |
  | **ixTrans** | Lossless matrix transpose; reduces bank conflicts and VRAM overhead |

- **天数超节点 ("Tianshu supernode")** — a companion scale-up system claimed to provide **144-chip high-speed full interconnect**. This is the **first scale-up domain size Iluvatar has ever stated**, and is arguably the more survey-relevant disclosure. **Topology, per-link bandwidth, switch ASIC, whether it is a single coherent domain, and whether it extends the existing CLIF fabric or introduces a new one are all not disclosed.** The supernode is explicitly at the **trial / partnership-discussion** stage.

### Vendor performance claims (marketing — unverified, relative-only, all vs NVIDIA Hopper)

All figures below are first-party, expressed only as percentages against an unspecified "Hopper-architecture mainstream international solution." **No absolute FLOPS, no MLPerf entry, no third-party benchmark exists.**

| Claim | Vendor figure |
|---|---|
| Attention efficiency | >90% across precisions; ~10% above a Hopper solution at 64k context |
| MoE compute efficiency | >70%; ~10% higher average MoE throughput on a DeepSeek-V4-class workload |
| Time-to-first-token | ~20% lower across Qwen / GLM / Kimi / DeepSeek |
| Decode efficiency | ~10% higher |
| Inter-card communication latency | ~13% lower on average |
| Energy efficiency | Claimed above mainstream Hopper solutions (no figure given) |

Note the tension with January 2026 roadmap coverage, which positioned Iluvatar's 2026 parts against **B200/Blackwell**. TianGai 300's own marketing benchmarks are against **Hopper**.

### Status — announced, not shipping

Chinese coverage says only **"天垓300已具备规模化应用条件"** ("now meets the conditions for large-scale deployment") and that the part is in deep adaptation / joint-solution discussions with domestic cloud providers, server OEMs and supernode system vendors. There is **no** disclosure of tape-out date, sampling, mass production, or any named deployment. Treat TianGai 300 as **announced (2026-07-19)**.

### Explicitly NOT disclosed — do not populate these fields

Process node and foundry; die size; transistor count; packaging (CoWoS or alternative); memory type (HBM2e / HBM3 / HBM3e), capacity and bandwidth; peak FLOPS/TOPS at **any** dtype (FP32/TF32/BF16/FP16/FP8/FP4/INT8); scale-up link bandwidth per chip; host interface generation; TDP; form factor; card/board name; price. **No retrieved source — vendor or press — publishes any of these numbers.** Any such figure appearing elsewhere is aggregator-invented.

### Adjacent corrections to the survey baseline

1. **Edge/endpoint line missing from the survey.** The vendor site now lists a **彤央 (Tongyang) TY series** — **TY1000, TY1100, TY1100-NX, TY1100-NX-PRO, TY1200** — absent from this survey entirely. No specifications retrieved; recorded here as a known gap.
2. **Inference part naming.** The vendor lists the inference part as **智铠100 (Zhikai 100)**; "MR-V100" is the product code, not the current product name.
3. **Deployment figures.** "290+ customers, 52,000 GPUs as of Jun 2025" is superseded by **340+ customers across 30+ industries as of end-2025** and **>10,000 chips in online cluster operation** — both company-stated, and not directly comparable to each other.

### Confirmed absences (checked, found nothing)

- **No MLPerf submission** by Iluvatar CoreX. A Chinese analysis reports zero Iluvatar entries in MLCommons results through 2025, and none appear in v6.0.
- **No Hot Chips / ISCA / ISSCC 2026 presentation** by Iluvatar CoreX.
- **No H1 2026 financial disclosure** post-dating the FY2025 RMB 1.03B figure.

---

## Business Context

| Event | Date | Detail |
|-------|------|--------|
| Founded | 2015 | Shanghai; Li Yunpeng (ex-Oracle R&D Director) |
| TianGai-100 mass production | Jan 2021 | China's first 7nm GPGPU; 200+ customers |
| TianGai-150 launch | ~2022 | 64 GB HBM2; BI-V150 product code |
| Zhikai 100 (智铠100 / MR-V100) launch | ~2022 | China's first GPGPU inference chip |
| BAAI Aquila2-70B training | 2023 | 120-node BI-V100 + 8-node BI-V150; 85.3% peak util. |
| Revenue 2024 | 2024 | RMB 540M; net loss RMB 892M |
| 52K GPU cumulative shipped | Jun 2025 | 290+ customers |
| HKEX IPO | Jan 8, 2026 | First Chinese GPGPU on HKEX (ticker 09903.HK); USD 475M raised; USD 5.3B market cap |
| Revenue 2025 | Apr 2026 | RMB 1.03B (+92% YoY); net loss ~RMB 1B |
| Four-gen roadmap unveiled | Jan 2026 | Tianshu → Tianxuan → Tianji → Tianquan (2027 Rubin target); investor-facing, no product tied to the codenames |
| Developer portal launched | ~Jul 15–19, 2026 | developer.iluvatar.com; docs, images, model cases, DeepSpark resources; ~100 acceleration libraries claimed since 2018 |
| **TianGai 300 (天垓300) announced** | **Jul 19, 2026** | **WAIC 2026 day 3, Shanghai; first product on new-gen self-developed architecture; FP8/FP4; ixSMEX/ixDPX/ixTrans; announced, not shipping** |
| **天数超节点 (Tianshu supernode) announced** | **Jul 19, 2026** | **144-chip full interconnect scale-up domain; trial / partnership-discussion stage; topology and link BW not disclosed** |
| Deployment figures restated | End-2025 (stated Jul 2026) | 340+ customers, 30+ industries, >10,000 chips in online cluster operation (company-stated) |

---

## Resources

### Official
- [Iluvatar CoreX Official Site](https://www.iluvatar.com/) — product navigation lists 天垓300 / 天垓150 / 天垓100, 智铠100, and the 彤央 TY1000/TY1100/TY1100-NX/TY1100-NX-PRO/TY1200 edge line. No spec sheet for 天垓300.
- [Iluvatar Developer Portal](https://developer.iluvatar.com) — launched ~Jul 2026
- [DeepSpark Open Platform](https://www.deepspark.org.cn/)
- [Deep-Spark GitHub Organization](https://github.com/Deep-Spark)

### Hardware
- [Tom's Hardware — Iluvatar four-gen GPU roadmap](https://www.tomshardware.com/pc-components/gpus/chinas-iluvatar-corex-unveils-four-generation-gpu-roadmap-aimed-at-surpassing-nvidia-rubin)
- [TrendForce — Iluvatar 2026-28 roadmap targeting H200, B200](https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/)
- [Wikipedia — Iluvatar CoreX](https://en.wikipedia.org/wiki/Iluvatar_CoreX)

### TianGai 300 / WAIC 2026 (added 2026-08-08)
- [Tencent News, 2026-07-19 — WAIC day-3 launch; SIMT; ixSMEX/ixDPX/ixTrans; FP4/FP8 + MMA/DPX/FMA; 144-chip supernode; 340+ customers](https://news.qq.com/rain/a/20260719A08MB800)
- [Tencent News (second piece), 2026-07-19 — ixDPX six-instructions-to-one; Hopper-relative TTFT/Attention/MoE/decode claims](https://news.qq.com/rain/a/20260719A08J8N00)
- [IT之家 — 2026-07-19 WAIC launch, SIMT general-purpose architecture, Hopper-relative claims](https://www.ithome.com/0/978/781.htm)
- [Tencent Cloud developer news — 144-chip full-interconnect supernode at trial stage; ~100 acceleration libraries; 340+ customers, 30+ industries, 1,000+ day uptime, >10k chips online](https://cloud.tencent.com/developer/news/4280275)
- [半导体行业观察 — July 19 launch; SIMT scalar/vector/tensor; AF-separation / PD-separation optimizations](https://www.semi-insights.com/s/bdt/15/50538.shtml)
- [经济参考报 — WAIC 2026 coverage](https://www.jjckb.cn/20260720/3cb0e7d7378745b58c576c51ee79b65a/c.html)
- [观察者网 — 2026-07-18 pre-launch coverage](https://www.guancha.cn/GongSi/2026_07_18_824207.shtml)

### Software
- [GitHub — Deep-Spark/FlyDSL (Python + MLIR kernel-authoring DSL; FlyIXDL Iluvatar backend)](https://github.com/Deep-Spark/FlyDSL)
- [Deep-Spark repositories by last update](https://github.com/orgs/Deep-Spark/repositories?sort=updated)
- [GitHub — iluvatar-corex-ixrt (IxRT open source)](https://github.com/Deep-Spark/iluvatar-corex-ixrt)
- [GitHub — ix-container-toolkit](https://github.com/Deep-Spark/ix-container-toolkit)
- [GitHub — ixGDB](https://github.com/Deep-Spark/ixGDB)
- [GitHub — ix-exporter](https://github.com/Deep-Spark/ix-exporter)
- [HAMi — Enable Iluvatar GPU sharing](https://project-hami.io/docs/userguide/iluvatar-device/enable-illuvatar-gpu-sharing)

### Business / IPO
- [Caixin — HKEX IPO USD 475M](https://www.caixinglobal.com/2025-12-30/chinese-gpu-maker-iluvatar-corex-seeks-475-million-in-hong-kong-listing-102398766.html)
- [Caixin — HKEX debut USD 5.3B market cap](https://www.caixinglobal.com/2026-01-08/chinese-gpu-maker-iluvatar-corex-climbs-in-hong-kong-debut-with-53-billion-valuation-102401708.html)
- [Startupnews — 2025 92% revenue jump RMB 1.03B](https://startupnews.fyi/2026/04/02/chinese-gpu-designer-iluvatar-corex-reports-92-jump-in-annual-revenue/)
- [SCMP — Iluvatar GPU roadmap vs Rubin](https://www.scmp.com/tech/big-tech/article/3341368/iluvatar-corex-targets-nvidias-rubin-gpu-road-map-amid-china-chip-push)
