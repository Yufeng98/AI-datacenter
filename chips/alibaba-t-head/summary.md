# T-Head (平头哥) AI Chip Software and Hardware Stack Summary

*as_of: 2026-04-05 · last updated: 2026-08-08*

---

## Overview

T-Head Semiconductor (平头哥半导体) is Alibaba Group's chip subsidiary, operating since 2018 under DAMO Academy. Its core thesis: build a vertically integrated AI stack — Hanguang/Zhenwu inference accelerators + Xuantie RISC-V CPUs + Alibaba Cloud + Qwen LLMs — mirroring the Google TPU+GCP+Gemini model but targeting China's AI compute market.

The **Zhenwu 810E** (PPU, publicly revealed January 2026) is China's highest-shipping domestic AI accelerator, at ~470,000 units delivered by February 2026 (cumulative Zhenwu-series shipments reached 560,000 by May 2026 — see the 2026-08-08 update section). Built on SMIC 7nm with 96 GB HBM2e and 700 GB/s inter-chip bandwidth, it matches NVIDIA H20 performance at 40% lower BOM cost, enabling Alibaba Cloud to cut inference prices by 50%.

The **Zhenwu M890** was formally debuted on 2026-05-20 at the Alibaba Cloud Summit as the next Zhenwu-series part: 144 GB of what Alibaba calls "on-chip memory", 800 GB/s inter-chip bandwidth, and native precision "from FP32 down to FP4", deployed in the 128-accelerator **Panjiu AL128** supernode over a new in-house **ICN Switch 1.0** ASIC. On 2026-07-18 at WAIC Shanghai T-Head announced it had **open-sourced the SAIL software stack** — the first break in the fully closed Zhenwu software story recorded in the sections below. See "Zhenwu M890 / Panjiu AL128 / SAIL Update (2026-08-08)".

In parallel, the **XuanTie C950** (March 2026, TSMC 5nm) is the first RISC-V processor to natively support 100B+ parameter LLMs (Qwen3, DeepSeek V3), targeting agentic AI inference without a separate NPU — a distinct market positioning away from GPU-like accelerators. No update to the C950 has been disclosed since its March 2026 launch.

---

## Software Stack

### Framework Integration

Zhenwu 810E supports **PyTorch** (primary) via a proprietary frontend API layer, with **ONNX** as model interchange and claimed **TensorFlow** compatibility. The older Hanguang 800 HGAI SDK additionally supported MXNet and Caffe. There is no eager-mode execution — all computation is ahead-of-time compiled.

Through the 2026-04-05 baseline, no public SDK existed for either chip; access was exclusively through Alibaba Cloud's Model Studio API, making the stack a pure cloud-consumed service. **Superseded 2026-07-18:** T-Head announced the open-sourcing of the **SAIL** stack at WAIC Shanghai, with vendor-claimed compatibility across 260+ mainstream training and inference frameworks. No public Git repository URL or license could be confirmed, so "open source" here is vendor-asserted availability rather than a verified public repo — see the update section.

```python
# Zhenwu 810E access model (conceptual — no public API published)
# Users access via Alibaba Cloud Model Studio / ECS instances
# PyTorch models exported to ONNX, compiled server-side, served via REST
```

### Compiler / IR

T-Head's compiler stack was fully proprietary through the 2026-04-05 baseline; the 2026-07-18 SAIL announcement claims a compiler layer is now part of the released stack, but no repository or license has been confirmed. For Hanguang 800: the **HGAI compiler** performs Graph IR conversion, INT8 quantization + sparse weight compression, and op-to-hardware mapping (Tensor/Pooling/Memory engines per core with 48 MB SRAM tiling).

For Zhenwu 810E: an undisclosed proprietary compiler (MLIR-based internal IR inferred from architectural positioning) lowers PyTorch/ONNX graphs for both training and inference. Known limitation: early operator coverage gaps in complex reinforcement learning workloads.

### Op Library

Hanguang 800 Tensor Engine natively accelerates: convolution, GEMM, deconvolution, dilated convolution, 3D convolution, interpolation, ROI pooling. The architecture is domain-programmable to support future activation functions without hardware respins.

Zhenwu 810E includes LLM-specific op optimizations for Qwen3 and DeepSeek V3 attention, feedforward, and normalization patterns.

### Runtime

- **HGRT** (Hanguang Runtime): PCIe DMA, kernel launch, result collection; cloud-only
- **Zhenwu Runtime**: proprietary cluster orchestration; 10,000+ card deployment-scale
- **NPUSMI**: monitoring tool for Hanguang 800 (frequency, memory/compute utilization)
- **C950 native runtime**: no separate NPU SDK — LLM inference directly via the integrated AI engine

### Driver / Firmware

Both Hanguang and Zhenwu use proprietary closed-source PCIe/ICN drivers. Not open-sourced. Not publicly available.

### Communication

Zhenwu 810E features **ICN (Inter-Chip Network)**: 7 proprietary links achieving 700 GB/s aggregate inter-chip bandwidth. This enables near-linear multi-card scaling for training. No NCCL equivalent — collective operations are integrated into the proprietary runtime.

Hanguang 800 has no scale-up fabric; PCIe-only, single-chip deployment.

### Open-Source: Xuantie Toolchain

The Xuantie CPU ecosystem is genuinely open-source (Apache 2.0):
- **xuantie-gnu-toolchain**: GCC + Binutils for Xuantie RISC-V with vendor vector extensions
- **OpenC906, OpenC910**: full RTL (Verilog) open-sourced
- **buildroot, newlib, riscv-aosp**: embedded and Android support

---

## Hardware Architecture

### Hanguang 800 NPU

4-core ring bus, TSMC 12nm, 17B transistors, 820 TOPS INT8. Each core: Tensor Engine + Pooling Engine + Memory Engine + 48 MB SRAM. Command Processor coordinates cores. PCIe Gen4 x16 host interface. **SRAM-only** (no HBM). Presented at Hot Chips HC32 (2020). Best-in-class at launch: 78,563 IPS ResNet-50, 500 IPS/W.

### Zhenwu 810E (PPU)

SMIC 7nm, 96 GB HBM2e, 400 W TDP, 7× ICN links (700 GB/s). Supports training + inference + autonomous driving. Optimized for Qwen3 and DeepSeek V3. Deployed in 10,000-card clusters, 400+ customers, 470,000 units shipped by February 2026.

### Zhenwu M890 (announced 2026-05-20)

144 GB "on-chip memory" (Alibaba's own wording — memory *type* is not vendor-confirmed), 800 GB/s inter-chip bandwidth, native precision from FP32 down to FP4, positioned for integrated agentic training + inference. Vendor claim: 3× the performance of the Zhenwu 810E, with no benchmark methodology disclosed. Process node, foundry, TDP, die size and peak FLOPS by data type are **not disclosed**. Deployed in the 128-accelerator Panjiu AL128 supernode over ICN Switch 1.0. Status: announced/debuted — Alibaba makes no mass-production or general-availability claim for the chip itself.

### XuanTie C950

TSMC 5nm, RVA23-compliant, 3.2 GHz max, RISC-V world-record SPEC score >70. Self-developed AI engine with native 100B+ LLM inference. >3× over C920. First RISC-V CPU for agentic AI data center inference.

### Memory Comparison

| Chip | On-chip SRAM | Off-chip | Bandwidth |
|---|---|---|---|
| Hanguang 800 | 192 MB (all-SRAM, no HBM) | None (PCIe to host DRAM) | PCIe-limited |
| Zhenwu 810E | Undisclosed on-chip | 96 GB HBM2e | ~2 TB/s est. |
| **Zhenwu M890** | Not separately disclosed | **144 GB, described by Alibaba only as "on-chip memory"** (type not vendor-confirmed; secondary coverage says HBM, some say HBM3 — unsupported) | Not disclosed |
| XuanTie C950 | Standard CPU L1/L2/L3 | DDR5 (server) | Standard CPU |

### Interconnect

| Chip | Scale-up | Scale-out |
|---|---|---|
| Hanguang 800 | None (single chip) | N/A |
| Zhenwu 810E | 7× ICN, 700 GB/s | Alibaba Cloud DC network |
| **Zhenwu M890** | **ICN, 800 GB/s inter-chip; ICN Switch 1.0 (25.6 Tbps aggregate) across 64-accelerator clusters; Panjiu AL128 = 128 accelerators per unit, "PB/s internal bandwidth" (vendor claim)** | Alibaba Cloud DC network |
| XuanTie C950 | Standard CPU interconnect | Standard server fabric |

---

## Zhenwu M890 / Panjiu AL128 / SAIL Update (2026-08-08)

*Updated 2026-08-08. Primary sources: Alibaba Group newsroom and Alibaba Cloud blog (2026-05-20/21); T-Head news post (2026-07-18). Independent corroboration: Reuters, The Register (2026-05-22), YiCai Global, ITHome, The Next Web, SCMP. Roadmap figures are Chinese-media-reported and absent from Alibaba's English newsroom.*

Everything in this section is **new since the 2026-04-05 baseline** above. Prior-generation content is retained unchanged.

### Zhenwu M890 — formally debuted 2026-05-20

Announced at the Alibaba Cloud Summit on **2026-05-20** (Reuters, YiCai Global, ITHome and Alibaba's own newsroom all use this date; CNBC's 2026-05-19 dateline is US time).

| Parameter | Value | Provenance |
|---|---|---|
| Memory capacity | **144 GB** | Alibaba newsroom |
| Memory type | Alibaba says only **"on-chip memory"** — type **not disclosed** | Secondary coverage calls it HBM; digitalcitizen/Toutiao say HBM3 — **not vendor-supported** |
| Inter-chip bandwidth | **800 GB/s** (vs 700 GB/s on 810E) | Alibaba newsroom |
| Data types | Native support **"from FP32 down to FP4"** | Alibaba newsroom |
| Performance | **"Three times the performance of its predecessor, Zhenwu 810E"** | Vendor marketing claim; no benchmark or methodology disclosed |
| Peak FLOPS by dtype | **Not disclosed** | A 0.6 PFLOPS FP16 figure circulates on Toutiao/Zhihu with no primary source; deliberately not recorded here |
| Process / foundry | **Not disclosed** (SMIC vs TSMC unconfirmed) | The Register and TNW note only that it must use a domestically-accessible node |
| TDP, die size, transistor count | **Not disclosed** | — |
| Workload positioning | Integrated training + inference for agentic workloads | Alibaba newsroom |
| Internal codename | **Not disclosed.** The line is generically called *PPU*; the "PPU 1.5" designation circulating in aggregators is **not confirmed by any retrieved source** and is not recorded here | — |

**Status — do not overstate.** Alibaba's newsroom says the M890 was *"formally debuted"* and makes **no mass-production or general-availability claim for the chip**. What is stated as available is the **Panjiu AL128 supernode, "now available through Alibaba's model service platform, Bailian."** The Next Web notes that per-chip pricing and M890-specific shipment volumes were **not disclosed**; its "scaled mass production" phrasing covers T-Head's accelerator line broadly, not the M890. Correct description: **announced, with the AL128 platform in cloud service; M890-specific volume deployment not confirmed.**

### Panjiu AL128 Supernode

- **128 AI accelerators in a single rack-scale unit**; a **64-card configuration** is also offered.
- **"Petabyte-per-second (PB/s) internal bandwidth"** — vendor claim, independently repeated by The Register (2026-05-22); no per-link breakdown published.
- Available through Alibaba's Bailian model service platform.
- **Novelty caveat:** the "Panjiu AL128" name and its ScaleUp / ScaleOut / DCN interconnect architecture were already published by Alibaba Cloud on **2025-11-14**, before this repo's baseline. What is new on 2026-05-20 is the **M890 + ICN Switch 1.0 instantiation** of that supernode, not the supernode concept.

### ICN Switch 1.0

T-Head's own switch ASIC, the scale-up counterpart to the per-chip ICN links already recorded for the 810E.

- **Up to 25.6 Tbps aggregate bandwidth** (Alibaba newsroom; The Register).
- Enables congestion-free communication across **clusters of 64 accelerators**.
- Latency: **hundred-nanosecond-class chip-to-chip** (ITHome, YiCai Global rendering the Chinese 百纳秒级); one secondary source (BigGo) reports "under 150 nanoseconds". Alibaba's English release does not state a latency figure. ⚠️ A "sub-100 ns" figure in circulation is a **mistranslation** of 百纳秒级 and is wrong.
- Context: The Register notes Broadcom and NVIDIA shipped switch silicon at comparable aggregate bandwidth years earlier — 25.6 Tbps is competitive for a first in-house part, not a frontier number.

### Roadmap disclosed 2026-05-20

| Chip | Window | Disclosed specs | Provenance |
|---|---|---|---|
| **Zhenwu V900** | **3Q2027** | 3× the M890; **216 GB memory**; **1,200 GB/s inter-chip bandwidth**; "deeply iterated in-house parallel computing architecture" | Chinese-media-reported vendor disclosure from the summit (Mydrivers via TrendForce; Sohu, Lanjinger, Toutiao, Baidu Baike). **Absent from Alibaba's English newsroom.** Not an analyst estimate |
| **Zhenwu J900** | **3Q2028** | **No specs disclosed** | Same |

This is the first public T-Head accelerator roadmap, and it implies a roughly annual-to-biennial cadence (810E Jan 2026 → M890 May 2026 → V900 3Q2027 → J900 3Q2028).

### SAIL software stack — open-sourced 2026-07-18 (WAIC Shanghai)

Announced at WAIC Shanghai and confirmed by SCMP, The Next Web, Nation Press and T-Head's own news post. This supersedes the "no public SDK / fully proprietary compiler" statements in the Software Stack section above.

- **Vendor claim:** compatibility with **260+ mainstream training and inference frameworks**, explicitly including **PyTorch, TensorFlow, vLLM and SGLang**.
- **Vendor claim:** migration off CUDA in **"fewer than seven days"** (SCMP reports this as a T-Head statement). One technical deep-dive instead reports migration "without modifying existing code" — the two claims are not identical and neither is independently benchmarked.
- **Layer count is not settled.** Several outlets describe a **five-layer** stack (kernel drivers → compiler → high-performance operator + communication libraries → SDK/toolchain → performance-analysis/debug tools); at least one technical deep-dive instead describes a **three-layer** architecture (interface / SDK / OS). Do not treat either as the canonical decomposition.
- **No public Git repository URL and no license could be confirmed** from any source. Distribution appears to run through the **T-Head developer community** rather than a GitHub organization. Native driver adaptation to the **OpenAnolis Anolis Cloud Kernel (ANCK)** is described.
- Consequence for this survey: **"open source" is vendor-asserted availability, not a verified public repo.** The Xuantie RISC-V repositories remain the only confirmed-public T-Head code.

### Deployment base

- **560,000 cumulative Zhenwu-series units shipped** (up from 470,000 at the baseline) — this is a **series-cumulative** figure, **not** M890 shipments.
- **400+ external customers across 20+ industries** (Alibaba newsroom; The Register).
- Named customers independently confirmed: **China Telecom, FAW Group, Shanghai Pudong Development Bank** (YiCai Global, Baidu Baike). XPeng and Sina Weibo appear in some aggregator coverage of this announcement but are **not confirmed by any retrieved source** for it.

### Corporate status — downgraded

Alibaba has **not confirmed** a T-Head restructuring or IPO. Reuters' own headline (2026-01-22) is *"Alibaba to plan IPO for AI chipmaking unit T-Head, Bloomberg News reports"* — a sourced media report of plans to restructure T-Head into a partly employee-owned entity ahead of a possible listing. JPMorgan publicly characterised the T-Head IPO as a **sentiment catalyst rather than a 2026 deal**. As of **2026-08-08** there is **no listing, no filing acceptance and no named exchange**. This is also pre-baseline news, not part of the April–August 2026 window.

### Not confirmed / no change

- **Process node and foundry for the M890** — undisclosed everywhere.
- **No FLOPS-by-dtype figure** for the M890 in any official release.
- **No T-Head presentation confirmed** at ISCA 2026 or ISSCC 2026. Hot Chips 38 runs **2026-08-23–25**, after this update date; no T-Head talk is confirmed on its program and no content from that conference exists yet.
- **No MLPerf submission** found for any T-Head part.
- **XuanTie C950:** no tape-out, sampling or availability update found since the March 2026 launch already recorded above.

---

## Resources

### Documentation
- [T-Head NPU Product Page](https://www.t-head.cn/product/npu?lang=en)
- [Hanguang 800 Hot Chips HC32 2020](https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Inference_Alibaba_HanguangNPU_final.pdf)
- [Alibaba Cloud Hanguang 800 Blog](https://www.alibabacloud.com/blog/announcing-hanguang-800-alibabas-first-ai-inference-chip_595482)

### News / Analysis
- [Zhenwu 810E Launch — TechNode](https://technode.com/2026/01/30/alibabas-t-head-unveils-self-developed-ai-chip-zhenwu-810e/)
- [TrendForce: Zhenwu Matches H20](https://www.trendforce.com/news/2026/01/29/news-alibaba-t-head-unveils-new-ai-chip-said-to-match-nvidia-h20-as-ipo-speculation-builds/)
- [Alibaba AI Golden Triangle — Geopolitechs](https://www.geopolitechs.org/p/zhenwu-ai-chip-and-alibabas-three)
- [XuanTie C950 — The Register](https://www.theregister.com/2026/03/25/alibaba_damo_xuantie_c950_chip/)
- [C910 Architecture Deep Dive — Chips and Cheese](https://chipsandcheese.com/p/alibabat-heads-xuantie-c910)

### 2026-05 / 2026-07 Update Sources
- [Alibaba Group newsroom — M890 / Panjiu AL128 / ICN Switch 1.0 (2026-05-20)](https://www.alibabagroup.com/en-US/document-1994119844504535040)
- [Alibaba Cloud blog — new AI chip, flagship model and rebuilt cloud stack for the agentic era](https://www.alibabacloud.com/blog/alibaba-unveils-new-ai-chip-flagship-model-and-rebuilt-cloud-stack-ai-for-agentic-era_603151)
- [Alibaba Cloud blog — in-depth analysis of the Panjiu AL128 supernode interconnect architecture (2025-11-14)](https://www.alibabacloud.com/blog/in-depth-analysis-of-alibaba-cloud-panjiu-al128-supernode-ai-servers-and-their-interconnect-architecture_602665)
- [The Register — Alibaba's chip and AI position (2026-05-22)](https://www.theregister.com/systems/2026/05/22/alibaba-just-admitted-its-struggling-to-keep-up-with-rival-chipmakers-and-ai-shops/5244665)
- [YiCai Global — T-Head launches AI chip with triple the computing performance](https://www.yicaiglobal.com/news/alibabas-t-head-launches-ai-chip-with-triple-the-computing-performance)
- [ITHome — Zhenwu M890 coverage (Chinese)](https://www.ithome.com/0/952/644.htm)
- [The Next Web — Alibaba Zhenwu M890](https://thenextweb.com/news/alibaba-zhenwu-m890-t-head-china-ai-chip-nvidia)
- [TrendForce — M890 unveiling and 3Q27/3Q28 roadmap](https://www.trendforce.com/news/2026/05/21/news-alibaba-t-head-unveils-zhenwu-m890-with-3x-performance-vs-prior-gen-new-ai-chips-planned-for-3q273q28/)
- [SCMP — Alibaba targets NVIDIA's software ecosystem with open-source AI stack (SAIL)](https://www.scmp.com/tech/tech-war/article/3361048/alibaba-targets-nvidias-dominant-software-ecosystem-open-source-ai-stack)
- [T-Head news post — SAIL open-source announcement (2026-07-18)](https://www.t-head.cn/news/newsDetail?id=189)
- [Reuters — Alibaba to plan IPO for AI chipmaking unit T-Head, Bloomberg News reports (2026-01-22)](https://www.reuters.com/world/asia-pacific/alibaba-plan-ipo-ai-chipmaking-unit-t-head-bloomberg-news-reports-2026-01-22/)

### Open-Source Repositories
- [T-head-Semi GitHub](https://github.com/T-head-Semi)
- [xuantie-gnu-toolchain](https://github.com/T-head-Semi/xuantie-gnu-toolchain)
- [OpenC910 RTL](https://github.com/T-head-Semi) (XUANTIE-RV org)
- [riscv-aosp](https://github.com/T-head-Semi/riscv-aosp)
