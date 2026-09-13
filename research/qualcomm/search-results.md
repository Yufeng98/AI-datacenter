# Qualcomm Cloud AI 100/200 — Search Results

**Device class:** Cloud AI Inference Accelerator (data center)
**Scope:** Cloud AI 100 (Standard + Ultra) and Cloud AI 200/250
**Research date:** 2026-04-05
**Last updated:** 2026-08-08 (Dragonfly resources appended at the end of this document)

NOTE: This research focuses exclusively on the Cloud AI 100/200 data center inference products, NOT Hexagon DSP or mobile NPU.

---

## Layer 1 — Device Overview

Qualcomm Cloud AI 100 is a data center inference SoC family. The AI 100 Ultra (4-SoC card) is the flagship. AI 200/250 announced for 2026 with rack-scale architecture.

**SKU summary:**
| SKU | SoCs/card | INT8 TOPS | SRAM | DRAM | TDP |
|---|---|---|---|---|---|
| Cloud AI 100 (Standard) | 1 | 400 TOPS | 144 MB | 32 GB LPDDR4X | 75 W |
| Cloud AI 100 Ultra | 4 | 870 TOPS | 576 MB | 128 GB LPDDR4X | 150 W |
| Cloud AI 200 | TBD | not disclosed | — | 768 GB LPDDR/card | 160 kW rack |

Key references:
- https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Architecture/
- https://www.qualcomm.com/artificial-intelligence/data-center/cloud-ai-100-ultra
- https://arxiv.org/html/2507.00418v1

---

## Layer 2 — Chip Specifications

### AIC100 SoC (single die)

| Parameter | Value |
|---|---|
| AI cores | 16 |
| INT8 TOPS (per SoC) | ~400 TOPS |
| On-chip SRAM | 144 MB |
| MAC array (Tensor unit) | 8192 ops/cycle INT8; 4096 ops/cycle FP16 |
| Vector unit | 512 ops/cycle INT8, 700+ instructions |
| DRAM | LPDDR4X, 4×64b = 136 GB/s, up to 32 GB |
| NoC bandwidth (internal) | 186 GB/s |
| Host interface | PCIe Gen4 ×8 |

### Cloud AI 100 Ultra (4×AIC100)

| Parameter | Value |
|---|---|
| INT8 TOPS | 870 TOPS |
| SRAM | 576 MB |
| DRAM | 128 GB LPDDR4x, 548 GB/s |
| PCIe | Gen4 ×16 |
| TDP | 150 W |
| AI cores total | 64 |

---

## Layer 3 — AI Core Microarchitecture

Each AIC100 has 16 AI cores (Hexagon Q6 DSP with HVX + HMX extensions):

### Tensor Unit
- 2× 2D MAC arrays: 8K for INT8, 4K for FP16
- 8192 INT8 ops/cycle, 4096 FP16 ops/cycle
- 125+ linear algebra instructions

### Vector Unit
- 512 INT8 ops/cycle, 256 FP16 ops/cycle
- 700+ AI/image processing instructions
- Supports INT8/16, FP16/32

### Scalar Processor
- 4-way VLIW machine
- 6 hardware threads
- Local scalar register file + instruction/data caches

**Design principle:** Separation of concerns — tensor/vector/scalar have independent execution pipelines.

---

## Layer 4 — Memory Hierarchy

- Per-AI-core: L1/L2 cache (Hexagon standard)
- Shared on-chip SRAM: 144 MB per SoC (576 MB total on Ultra card)
- Three NoCs: Compute NoC (AI cores + PCIe), Memory NoC (AI cores ↔ DRAM), Config NoC (boot/config)
- Off-chip: LPDDR4X (standard SoC: 136 GB/s, 32 GB)
- Ultra card: 4 SoCs + PCIe switch, 548 GB/s combined DRAM bandwidth

---

## Layer 5 — Network on Chip

Three specialized NoCs:
1. **Compute NoC:** AI cores ↔ PCIe, multicast, 186 GB/s
2. **Memory NoC:** AI cores ↔ DRAM controllers
3. **Config NoC:** boot and hardware configuration

---

## Layer 6 — Software Stack

```
PyTorch / TensorFlow / ONNX (model sources)
         ↓
AI Hub Workbench / qaic-exec CLI (model compiler)
         ↓
QNN (Qualcomm AI Engine Direct) graph
         ↓
Cloud AI SDK (libQAic) — runtime API
         ↓
QAIC kernel driver (Linux kernel accel subsystem)
         ↓
AIC100 firmware
         ↓
AI Core hardware
```

---

## Layer 7 — Compiler

- **qaic-exec / qaic-compile:** CLI compilation tool
- Input formats: ONNX, TorchScript (.pt), PyTorch FX
- Output: QNN context binary (chip-specific)
- Supports: operator fusion, INT8 quantization, custom ops (C++)
- ONNX Runtime QNN Execution Provider: production path for ONNX models

---

## Layer 8 — Runtime / SDK

- **Cloud AI SDK (libQAic):** C++ API, Python bindings
- **QAIC driver:** Linux kernel driver (upstreamed: `drivers/accel/qaic/`)
- ExecuTorch Qualcomm backend: PyTorch ExecuTorch integration

---

## Layer 9 — Precision / Data Types

- INT8 (primary)
- FP16
- INT16, FP32 (vector unit)
- Mixed-precision support

---

## Layer 10 — Deployment

- PCIe card form factor (FH3/4L)
- Server integration: Lenovo ThinkSystem, Dell, etc.
- AI 200/250: rack-scale with direct liquid cooling, 160 kW rack

---

## Layer 11 — Open Source

- QAIC Linux kernel driver: upstream kernel (`drivers/accel/qaic/`) — fully open
- Cloud AI SDK: open source (GitHub: quic/cloud-ai-sdk)
- Qualcomm AI Engine Direct (QNN): SDK released publicly
- Chip design: closed

---

## Layer 12 — LLM Performance

- ArXiv 2507.00418: Cloud AI 100 Ultra vs H100 LLM serving comparison
- Competitive for LLM inference with high memory capacity (128 GB vs 80 GB H100)

---

## Layer 13 — Competing Chips

- NVIDIA H100/H200/B200
- AWS Inferentia2
- Intel Gaudi3
- Google TPU v5e

---

## Layer 14 — Roadmap

*(As recorded 2026-04-05; superseded on 2026-08-08 — see the Dragonfly section at the end of this document.)*

- Cloud AI 200 / AI 250: announced, shipping 2026
- 768 GB LPDDR per card, rack-scale architecture
- Hexagon NPU for both data center and edge unified

**Correction (2026-08-08):** neither AI200 nor AI250 shipped in 2026. Qualcomm's June 2026 guidance is AI200
"sampling in fiscal 2026 on LPDDR5x" (Investor-Day-reported, no production-shipment claim), AI250 commercial
sampling expected **mid-2027**, and a new part **AI300** with commercial sampling expected **2028**. The 768 GB
per-card figure is an October 2025 launch figure attributed by Qualcomm to AI200 and by The Register to AI250 —
an unresolved conflict.

---

## Layer 15 — Key Publications

- Architecture docs: https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Architecture/
- Linux kernel docs: https://docs.kernel.org/accel/qaic/aic100.html
- ArXiv LLM serving: https://arxiv.org/html/2507.00418v1
- Product brief: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Prod-Brief-QCOM-Cloud-AI-100-Ultra.pdf

---

# Resources Added 2026-08-08 — Qualcomm Dragonfly

*Discovered in the 2026-08-08 landscape scan, covering the Qualcomm Investor Day of 2026-06-24 and releases
through 2026-07-29. Source class is recorded for each entry because almost every Dragonfly performance figure is
unaudited vendor marketing.*

## Primary — Qualcomm newsroom

| Resource | URL | What it establishes |
|---|---|---|
| Newsroom index | https://www.qualcomm.com/news/releases | Confirms all five 2026-06-24 releases (Dragonfly roadmap, diversification strategy, Meta CPU agreement, Hugging Face, Modular acquisition) and the 2026-07-29 releases |
| **Dragonfly data center roadmap PR (2026-06-24)** | https://www.qualcomm.com/news/releases/2026/06/qualcomm-unveils-comprehensive-data-center-roadmap-for-the-agent | **The single most load-bearing source.** AI300 HBC Gen 2 / "54× over AI200" / 4×–8× perf-per-watt / sampling 2028; AI250 HBC Gen 1 / 133 TB/s / 18× / sampling mid-2027; HBC 6× per watt vs HBM and 200× capacity per watt vs SRAM; C1000 250+ cores, Oryon >5 GHz, >2 TB/s PCIe Gen 7 + CXL, availability 2028; UALink/ESUN, 800G/1.6T, up to 20 km; 35+ ecosystem supporters. Names exactly four Dragonfly products: C1000, AI200, AI250, AI300 |
| Qualcomm to Acquire Modular (2026-06-24) | https://www.qualcomm.com/news/releases/2026/06/qualcomm-to-acquire-modular | Acquisition announced, close guided to H2 2026, terms undisclosed. Notably does **not** name Mojo or MAX |
| July 2026 newsroom listing | https://www.qualcomm.com/news/releases/2026/07 | Confirms "Qualcomm Completes Acquisition of Modular", 2026-07-29 |
| Diversification strategy PR (2026-06-24) | https://www.qualcomm.com/news/releases/2026/06/qualcomm-accelerates-diversification-with-comprehensive-strategy | ">$15 billion by fiscal 2029" data center target and ~$1.7T by 2030 combined TAM. Contains **no** $5B FY2027 figure and **no** accelerator/CPU/custom-silicon TAM split |
| Meta multi-generation CPU agreement (2026-06-24) | https://www.qualcomm.com/news/releases/2026/06/qualcomm-and-meta-announce-strategic-multi-generation-agreement- | Meta agreement is for the Dragonfly C1000 CPU, "in production starting in the second half of 2028"; no volumes or binding terms |
| Q3 FY2026 results (2026-07-29) | https://www.qualcomm.com/news/releases/2026/07/qualcomm-announces-third-quarter-fiscal-2026-results | Revenue "$9.9 billion"; guides non-handset incl. Data Center from 24% YoY growth in FY2026 to >60% in FY2027; **no YoY decline stated** |

## Primary — acquired company

| Resource | URL | What it establishes |
|---|---|---|
| Modular: Qualcomm Completes Acquisition of Modular | https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular | Completion dated 2026-07-29; Mojo, MAX and Modular Cloud continue as products/brands; Chris Lattner becomes Qualcomm EVP of Advanced AI Software and Platforms |
| Modular blog index | https://www.modular.com/blog | Confirms Modular is the company behind Mojo and MAX |

## Independent / adversarial

| Resource | URL | Why it matters |
|---|---|---|
| **The Register — "bury the compute under the DRAM" (2026-06-30)** | https://www.theregister.com/systems/2026/06/30/qualcomms-proposed-solution-to-catch-up-in-ai-infra-bury-the-compute-under-the-dram/5264071 | **The essential counterweight.** Describes HBC as DRAM stacked over XPU logic via TSVs; reports AI250 at 768 GB / 133 TB/s / LPDDR5x; AI300 2028; documents that Qualcomm **declined to disclose peak FLOPS** and disputes the "effective bandwidth" multipliers (implied ~414 TB/s across 56 AI200 chips would need a ~6,720-bit bus) |
| The Register — Dragonfly unveiled (2026-06-24) | https://www.theregister.com/systems/2026/06/24/qualcomm-claims-its-not-too-late-for-dragonfly-to-land-in-datacenters/5261758 | C1000 production H2 2028, 250+ cores, Oryon >5 GHz, 2× perf/W and "30 percent more speed" claims; Meta multi-generation C1000 agreement; no financial targets given |
| The Register Qualcomm tag index | https://www.theregister.com/Tag/Qualcomm/ | Confirms article dates. Also carries a 2026-06-16 Tenstorrent acquisition rumor — **unconfirmed, deliberately excluded from the survey** |

## Analyst (single-source — cite with attribution)

| Resource | URL | Sole source for |
|---|---|---|
| Futurum Group — Qualcomm's Data Center Re-entry at Investor Day 2026 | https://futurumgroup.com/insights/qualcomms-data-center-reentry-at-investor-day-2026-arrives-just-in-time-for-the-inference-decode-prize/ | AI200 "sampling in fiscal 2026 on LPDDR5x"; the $5B FY2027 target; ≥$1B from two hyperscalers; the $680B/$200B/$115B FY2029 TAM split; and the connectivity/custom-silicon/accelerator/CPU **revenue-ramp** sequencing. None of these appear in a retrieved Qualcomm release — attribute, do not state as fact |

## Toolchain

| Resource | URL | Notes |
|---|---|---|
| quic/efficient-transformers releases | https://github.com/quic/efficient-transformers/releases | v1.22.0 dated **2026-06-18** (WAN 2.2 dual-stage high/low-noise transformers, first-block-caching for Diffusers, blocked-KV attention, layerwise ONNX export for large MoE, HF Transformers 5.5.4, Python 3.12); v1.21.0 dated **2025-12-22** (FLUX.1-schnell, WAN 2.2). **No April 2026 release exists.** No AI200/AI250 support — SDK remains Cloud AI 100-targeted |

## Negative-result checks (both confirmed)

| Resource | URL | Result |
|---|---|---|
| Hot Chips 38 program | https://hotchips.org/program/conference/ | HC38 listed as **August 24–25, 2026**; Qualcomm **absent** from the speaker list |
| MLPerf Inference v6.0 results | https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ | 24 submitters, published 2026-04-01; Qualcomm **not among them** |

**Consequence:** no new independent benchmark data for Cloud AI 100 / AI200 / AI250 / AI300 exists in this window.

## Gaps — resources that could not be retrieved

- The **October 2025 AI200/AI250 launch press release**. Without it, the "commercial availability in 2027 → commercial
  sampling mid-2027" timeline-slip narrative cannot be verified and is not asserted in the survey.
- Any Qualcomm document disclosing AI300 memory capacity, process node, TDP or rack power.
- Any Qualcomm document resolving whether 768 GB/card belongs to AI200 (Qualcomm, Oct 2025) or AI250 (The Register, Jun 2026).
- Any 2026 confirmation that the HUMAIN 200 MW Riyadh deployment (announced Oct/Nov 2025) has begun.
