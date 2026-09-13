# Alibaba T-Head (平头哥) AI Chip Research Summary

*as_of: 2026-04-05*

> ⚠️ **Superseded in part as of 2026-08-08.** A major update — **Zhenwu M890** (announced 2026-05-20), the **Panjiu AL128** 128-accelerator supernode, **ICN Switch 1.0**, the **V900 (3Q2027) / J900 (3Q2028)** roadmap, and the **SAIL** software stack announced open-sourced 2026-07-18 — is recorded in
> `chips/alibaba-t-head/summary.md` ("Zhenwu M890 / Panjiu AL128 / SAIL Update (2026-08-08)"),
> `research/alibaba-t-head/investigations/hw-architecture.md` §6–13 and
> `research/alibaba-t-head/investigations/software-stack.md` §7.
> In particular, the statements below about "no public SDK", "470,000 units shipped" and "IPO trajectory" are revised there. This file is retained as the 2026-04-05 baseline record.

---

## Overview

T-Head Semiconductor (平头哥半导体, Pingtouge) is Alibaba Group's chip subsidiary, founded in 2018 under DAMO Academy. It operates two distinct product lines: the **Hanguang NPU** family (cloud AI inference/training ASICs) and the **Xuantie CPU** family (RISC-V general-purpose and AI-capable server processors). As of early 2026, T-Head has become the highest-shipping domestically produced AI accelerator vendor in China, with the Zhenwu 810E (PPU) reaching H20-class performance at 40% lower BOM cost.

T-Head's AI "golden triangle" positions Hanguang/Zhenwu chips alongside Alibaba Cloud infrastructure and the Qwen LLM model family — forming a vertically integrated stack analogous to Google (TPU + GCP + Gemini) or Amazon (Trainium/Inferentia + AWS + Titan).

---

## Product Family

### Hanguang 800 NPU (含光 800) — 2019
The original inference ASIC. TSMC 12nm, 17B transistors, 820 TOPS INT8 peak. Architecture: 4-core ring bus, each core with Tensor Engine + Pooling Engine + Memory Engine, 192 MB total SRAM. Benchmarked at 78,563 IPS on ResNet-50 (4× best-in-class at launch); 500 IPS/W (3.3× best-in-class). Presented at Hot Chips HC32 (2020). Never publicly sold; used exclusively in Alibaba Cloud and Hangzhou City Brain. Highlights: sparse compression, INT8 quantization, no HBM (SRAM-only), PCIe Gen4 x16.

### Zhenwu 810E / PPU (真武 810E) — 2026
Second-generation AI accelerator for training + inference. SMIC 7nm domestic process, 96 GB HBM2e, 700 GB/s inter-chip bandwidth (7× ICN links), 400 W TDP. Performance comparable to NVIDIA H20; BOM cost ~40% lower. Deployed in 10,000-card clusters; 400+ customers; 470,000 units shipped by Feb 2026; annualized revenue >RMB 10 billion. Deeply optimized for Qwen3 and DeepSeek V3 inference. China Unicom: 16,384 Zhenwu 810E units in $390M data center.

### XuanTie C910 RISC-V CPU — 2021 (open-sourced)
3-wide OoO, 12-stage pipeline, RVV 0.7.1, TSMC 12nm, 0.8 mm², 2–2.5 GHz. Open-sourced as OpenC910 (Apache 2.0) along with E902/E906/C906 siblings.

### XuanTie C950 RISC-V CPU — March 2026
TSMC 5nm, RVA23, 3.2 GHz max, world-record SPEC score >70 for RISC-V. Self-developed AI acceleration engine natively supports 100B+ parameter models (Qwen3, DeepSeek V3). >3× performance over C920. Targets agentic AI inference in data centers.

---

## Software Stack

### HGAI SDK (Hanguang 800)
Proprietary, cloud-only access. Frontend: TensorFlow, MXNet, Caffe, ONNX. Pipeline: Graph IR conversion → INT8 quantization + sparse compression → NPU compilation → HGRT runtime. Monitoring via NPUSMI. No public download.

### Zhenwu Stack
Fully in-house proprietary compiler stack (MLIR-based internal IR). Frontend: PyTorch (primary), ONNX, TensorFlow. ICN collective operations for multi-card training. Early operator coverage gaps in complex RL workloads. No public SDK; accessed via Alibaba Cloud.

### Xuantie Toolchain (Open Source)
Apache 2.0. Repos: xuantie-gnu-toolchain (GCC + Binutils), newlib, buildroot, riscv-aosp (Android AOSP port), OpenC906 / OpenC910 RTL. Linux kernel upstreaming in progress. C950 natively executes LLM inference without separate NPU SDK.

---

## Key Findings

1. **Two-track strategy:** Hanguang/Zhenwu NPU for cloud AI workloads (closed); Xuantie RISC-V CPU for both edge/embedded (open-source) and emerging server AI (C950).
2. **Zhenwu 810E is China's most deployed domestic AI chip:** 470K units, 16K-chip clusters, >RMB 10B ARR — outpacing Huawei Ascend, Cambricon, Biren in deployment scale.
3. **No CUDA equivalent:** Entirely proprietary inference APIs; software maturity significantly trails NVIDIA. PyTorch support limited to Zhenwu stack with known gaps.
4. **C950 RISC-V differentiation:** First RISC-V processor to natively run 100B+ parameter LLMs without a separate NPU. Signals T-Head's bet on CPU-level AI inference for agentic workloads.
5. **IPO trajectory:** T-Head reportedly eyes spinoff and IPO (early 2026 reports); annualized revenue exceeds RMB 10B; highest-shipping domestic AI chip in China.
6. **Memory architecture gap:** Zhenwu 810E uses HBM2e vs competitors using HBM3/HBM3e; SMIC 7nm vs TSMC 4/3nm — process and memory bandwidth lag behind frontier GPUs.

---

## Resources

- [T-Head Official Site — NPU Product](https://www.t-head.cn/product/npu?lang=en)
- [Hanguang 800 Hot Chips HC32 2020 Paper](https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Inference_Alibaba_HanguangNPU_final.pdf)
- [Announcing Hanguang 800 — Alibaba Cloud Blog](https://www.alibabacloud.com/blog/announcing-hanguang-800-alibabas-first-ai-inference-chip_595482)
- [Zhenwu 810E TechNode Announcement](https://technode.com/2026/01/30/alibabas-t-head-unveils-self-developed-ai-chip-zhenwu-810e/)
- [TrendForce: T-Head Zhenwu Matches H20](https://www.trendforce.com/news/2026/01/29/news-alibaba-t-head-unveils-new-ai-chip-said-to-match-nvidia-h20-as-ipo-speculation-builds/)
- [XuanTie C950 Launch — The Register](https://www.theregister.com/2026/03/25/alibaba_damo_xuantie_c950_chip/)
- [GitHub: T-head-Semi (XUANTIE-RV)](https://github.com/T-head-Semi)
- [GitHub: xuantie-gnu-toolchain](https://github.com/T-head-Semi/xuantie-gnu-toolchain)
- [Alibaba AI Three-Pillar Strategy — Geopolitechs](https://www.geopolitechs.org/p/zhenwu-ai-chip-and-alibabas-three)
