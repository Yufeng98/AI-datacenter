# T-Head (平头哥) Hardware Architecture Investigation

**Chip:** alibaba-t-head  
**Device Class:** RISC-V AI (平头哥)  
**Investigated:** 2026-04-05  
**Sources:** Hot Chips HC32 2020, T-Head official site, TechNode, TrendForce, Tom's Hardware, EE Times

---

## 1. Product Family Overview

T-Head Semiconductor (平头哥半导体, Pingtouge) is Alibaba's chip subsidiary, founded 2018.  
Two main product lines:

| Product Line | Focus | Latest Generation |
|---|---|---|
| **Hanguang NPU** (含光) | Cloud AI inference/training ASIC | Zhenwu 810E (2026) |
| **Xuantie CPU** (玄铁) | RISC-V general-purpose + AI-capable | C950 (March 2026) |

---

## 2. Hanguang 800 NPU — Cloud Inference ASIC

### Process & Die
- **Process:** TSMC 12nm FinFET  
- **Transistors:** 17 billion  
- **Memory:** 192 MB on-chip SRAM (distributed across 4 cores, ~48 MB/core)  
- **Host Interface:** PCIe Gen4 x16

### Top-Level Architecture
```
┌────────────────────────────────────────────┐
│  Hanguang 800 SoC                          │
│  ┌──────────────────────────────────────┐  │
│  │  Command Processor (CP)              │  │
│  └──────────────────────────────────────┘  │
│         Ring Bus                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ Core 0 │ │ Core 1 │ │ Core 2 │ │ Core 3 │ │
│  │ TE+PE+ME│ │ TE+PE+ME│ │ TE+PE+ME│ │ TE+PE+ME│ │
│  └────────┘ └────────┘ └────────┘ └────────┘ │
└────────────────────────────────────────────┘
```

Each core contains:
- **Tensor Engine (TE):** Accelerates convolution, GEMM, deconvolution, dilated/3D convolution, interpolation, ROI
- **Pooling Engine (PE):** Max/Average/Global pooling, activation functions
- **Memory Engine (ME):** Data movement, DMA, compression/decompression

### Key Design Features
- **Sparse Compression + Quantization:** Reduces I/O and data movement for compressed weight tensors
- **Domain-programmable ISA:** Extensible native ops for future activation functions
- **Pipeline depth:** Full-parallel dataflow for CNN workloads
- **SRAM-only on-chip:** No HBM/DRAM on die; relies on host DRAM via PCIe

### Performance
| Metric | Value |
|---|---|
| Peak Compute | 820 TOPS (INT8) |
| ResNet-50 throughput | 78,563 IPS |
| Energy efficiency | 500 IPS/W |
| vs GPU comparison | 1 Hanguang 800 ≈ 10 GPU (Hangzhou City Brain benchmark) |

---

## 3. Zhenwu 810E (PPU) — Next-Gen AI Accelerator

Publicly unveiled January 2026; corresponds to the PPU shown in CCTV broadcast September 2025.

### Process & Memory
- **Process:** 7nm (domestic foundry, likely SMIC)
- **Memory:** 96 GB HBM2e
- **Host Interface:** PCIe + 7× ICN (Inter-Chip Network) links
- **Inter-chip bandwidth:** 700 GB/s (vs NVIDIA A800: 400 GB/s, H20: ~900 GB/s)
- **TDP:** 400 W per board

### Architecture
- Proprietary compute architecture (not publicly disclosed in detail)
- Supports **AI training + AI inference + autonomous driving** workloads
- Deeply optimized for Alibaba **Qwen LLM** series inference and training
- 7× ICN links for near-linear multi-card scaling

### Performance vs NVIDIA
- Comparable to NVIDIA H20 (some modes surpass A100)
- BOM cost ~40% lower than H20 per card
- Enables 50% reduction in Alibaba Cloud inference pricing

### Deployment Scale (2026)
- Deployed in 10,000-card clusters on Alibaba Cloud
- 400+ customers: State Grid, Chinese Academy of Sciences, XPeng Motors
- Hundreds of thousands of units shipped
- China Unicom data center: 16,384 PPU units (72% of total AI chips)
- Annualized revenue > RMB 10 billion; 470,000 units delivered by Feb 2026

---

## 4. XuanTie RISC-V CPU Line

### Core Family (embedded → server)

| Core | Class | Vector ISA | Process | Notes |
|---|---|---|---|---|
| E902 | Embedded in-order | — | — | Open-sourced Apache 2.0 |
| E906 | Embedded in-order | — | — | Open-sourced Apache 2.0 |
| C906 | Mid-range in-order | RVV 0.7.1 | — | Open-sourced Apache 2.0 |
| C910 | High-perf OoO | RVV 0.7.1 | TSMC 12nm | 3-wide, 12-stage, 0.8 mm², 2–2.5 GHz |
| C920 | High-perf OoO | RVV 1.0 | — | C910 + updated vector spec |
| C930 | Server-grade | RVV 1.0 + AI | — | Launched Feb 2025, shipping Mar 2025 |
| C950 | Server-grade AI | RVA23 + AI engine | TSMC 5nm | Launched Mar 2026 |

### XuanTie C950 (March 2026) — AI-Agent CPU
- **Process:** TSMC 5nm  
- **Max frequency:** 3.2 GHz  
- **ISA:** RVA23-compliant (RISC-V Application Profile 2023)  
- **AI engine:** Self-developed, natively supports 100B+ parameter models (Qwen3, DeepSeek V3)  
- **SPEC score:** >70 single-core (world record for RISC-V)  
- **Performance:** >3× over C920  
- **Target:** Data center inference of agentic AI workloads

### C910 Architecture Details
- 3-wide out-of-order superscalar  
- 12-stage pipeline  
- Early adopter of RISC-V Vector Extension (RVV 0.7.1)  
- Area: 0.8 mm² at TSMC 12nm  
- Open-sourced as OpenC910 (Apache 2.0)

---

## 5. Known Gaps / Limitations

- Hanguang 800 SRAM-only (no HBM): limits large model weights on-chip
- Zhenwu 810E uses HBM2e vs HBM3 in NVIDIA H100/H200 — memory bandwidth gap
- Software stack significantly less mature than CUDA ecosystem
- Operator coverage gaps reported in early Zhenwu complex RL workloads
- Domestic 7nm process for Zhenwu vs TSMC 4/3nm for leading competitors

---

# Investigation Update — 2026-08-08

**Investigated:** 2026-08-08 (window: 2026-04-05 baseline → 2026-08-08)
**Change class:** major
**Primary sources:** Alibaba Group newsroom (2026-05-20), Alibaba Cloud blog (2026-05-21 and 2025-11-14), T-Head news post (2026-07-18)
**Independent corroboration:** Reuters, The Register (2026-05-22), YiCai Global, ITHome, The Next Web, SCMP, BigGo, TrendForce
**Verification note:** this section reflects an adversarial re-verification of the raw scan. Several figures that circulated in aggregator coverage were refuted and are recorded here as refuted, not carried forward.

---

## 6. Zhenwu M890 — announced 2026-05-20

### Event and date
Formally debuted at the **Alibaba Cloud Summit, 2026-05-20**. The date is confirmed by Reuters, YiCai Global, ITHome and Alibaba's own newsroom. CNBC's 2026-05-19 dateline is US time for the same event and is **not** a separate announcement.

### Disclosed specifications

| Parameter | Value | Confidence | Note |
|---|---|---|---|
| Memory capacity | 144 GB | confirmed | Alibaba newsroom |
| Memory type | **not disclosed** | — | Alibaba's only wording is "on-chip memory". TrendForce/BigGo say generically "HBM"; digitalcitizen and Toutiao say "HBM3" — **not vendor-supported** |
| Memory bandwidth | **not disclosed** | — | — |
| Inter-chip bandwidth | 800 GB/s | confirmed | vs 700 GB/s on the 810E |
| Data types | native "from FP32 down to FP4" | confirmed (vendor spec) | First FP4 claim in the Zhenwu line |
| Peak FLOPS by dtype | **not disclosed** | — | See "Refuted / withheld figures" below |
| Relative performance | "three times the performance of its predecessor, Zhenwu 810E" | vendor claim | No benchmark, workload or methodology disclosed |
| Process / foundry | **not disclosed** | — | SMIC vs TSMC unconfirmed; The Register and TNW note only a domestically-accessible node |
| TDP / die size / transistors | **not disclosed** | — | — |
| Microarchitecture | **not disclosed** | — | No block diagram published, as with the 810E |
| Internal codename | **not disclosed** | — | The line is generically called *PPU*; see refuted list |
| Positioning | integrated training + inference for agentic workloads | confirmed | Alibaba newsroom |

### Status determination
Alibaba's newsroom uses **"formally debuted"** and makes **no mass-production or general-availability claim for the chip**. The only availability asserted is for the **Panjiu AL128 supernode**, "now available through Alibaba's model service platform, Bailian." The Next Web explicitly notes that per-chip pricing and M890 shipment volumes were not disclosed, and its "scaled mass production" phrasing covers T-Head's accelerator line broadly rather than the M890.

**Recorded status:** announced, with the AL128 platform in cloud service; M890-specific volume deployment **not confirmed**.

---

## 7. Panjiu AL128 Supernode

- **128 AI accelerators** in a single rack-scale unit; a **64-card configuration** is also offered.
- **"Petabyte-per-second (PB/s) internal bandwidth"** — vendor claim (Alibaba newsroom), independently repeated by The Register 2026-05-22. No per-link bandwidth, topology diagram or bisection figure has been published, so the claim cannot be decomposed.
- Available through Alibaba's **Bailian** model service platform.

**Novelty caveat (correction to the raw scan).** The raw finding described the AL128 as "a new scale-up domain". It is not new as a design: Alibaba Cloud published an in-depth analysis of the **Panjiu AL128 supernode AI server and its ScaleUp / ScaleOut / DCN interconnect architecture on 2025-11-14**, before this repo's baseline. What is new on 2026-05-20 is the **M890 + ICN Switch 1.0 instantiation** of that supernode. The repo's earlier silence on the AL128 was a coverage gap, not evidence of novelty.

---

## 8. ICN Switch 1.0

T-Head-developed switch ASIC — the switched counterpart to the point-to-point ICN links recorded for the 810E.

| Parameter | Value | Confidence |
|---|---|---|
| Aggregate bandwidth | up to **25.6 Tbps** | confirmed (Alibaba newsroom; The Register) |
| Congestion-free domain | **64 accelerators** | confirmed |
| Chip-to-chip latency | **hundred-nanosecond class** (百纳秒级) | secondary only (ITHome, YiCai Global). BigGo reports "under 150 nanoseconds". **No latency figure appears in Alibaba's English release** |
| Port count / SerDes rate / process | **not disclosed** | — |

**Competitive context:** The Register notes that Broadcom and NVIDIA shipped switch silicon at comparable aggregate bandwidth years earlier. 25.6 Tbps is a credible first in-house switch, not a frontier figure.

---

## 9. Roadmap disclosed 2026-05-20

| Chip | Window | Disclosed specs |
|---|---|---|
| **Zhenwu V900** | 3Q2027 | 3× the M890; **216 GB memory**; **1,200 GB/s inter-chip bandwidth**; "deeply iterated in-house parallel computing architecture" |
| **Zhenwu J900** | 3Q2028 | **no specs disclosed** |

**Attribution (correction to the raw scan).** These V900 figures are **not** a TrendForce analyst estimate. TrendForce sources them to Mydrivers, and Sohu, Lanjinger, Toutiao and Baidu Baike all report them as disclosed by T-Head at the summit. The correct label is **Chinese-media-reported vendor disclosure**, absent from Alibaba's English newsroom — stronger footing than "analyst-reported", but still secondary. Chinese primary coverage (Sina Finance, Xueqiu, 2026-05-20) confirms the V900/J900 names and the cadence but not the numbers.

This is the first public T-Head accelerator roadmap.

---

## 10. Deployment base

- **560,000 cumulative Zhenwu-series units shipped** (up from 470,000 at the baseline). This is a **series-cumulative** figure, **not** M890 shipments — the raw scan's phrasing risked implying M890 volume.
- **400+ external customers across 20+ industries** (Alibaba newsroom; The Register).
- Named customers independently confirmed for this announcement: **China Telecom, FAW Group, Shanghai Pudong Development Bank** (YiCai Global, Baidu Baike).
- **XPeng and Sina Weibo are not confirmed** by any retrieved source for this announcement and are excluded. (The pre-baseline §3 customer list, sourced from January 2026 coverage, is retained unchanged above.)

---

## 11. Corporate status (out-of-window, verb downgraded)

Alibaba has **not confirmed** a T-Head restructuring or IPO. Reuters' headline (2026-01-22) is *"Alibaba to plan IPO for AI chipmaking unit T-Head, Bloomberg News reports"* — a media report of plans to restructure T-Head into a partly employee-owned entity ahead of a possible listing. JPMorgan publicly characterised the T-Head IPO as a **sentiment catalyst rather than a 2026 deal**. As of 2026-08-08 there is **no listing, no filing acceptance and no named exchange**. This is pre-baseline news and falls outside the April–August 2026 window.

---

## 12. Refuted / withheld figures (do not reintroduce)

| Circulating claim | Disposition |
|---|---|
| ICN Switch 1.0 "sub-100 ns" link latency | **WRONG.** Mistranslation of 百纳秒级 ("hundred-nanosecond class"). ITHome and YiCai say hundred-nanosecond-level; BigGo says "under 150 ns". No source says sub-100 ns |
| M890 internal codename "PPU 1.5" | **Unconfirmed.** Appears in no retrieved source (Alibaba newsroom, Reuters, The Register, YiCai, ITHome, TrendForce, Baidu Baike). The line is generically *PPU* |
| M890 memory is "HBM3" | **Not vendor-supported.** Alibaba says only "144 GB of on-chip memory" |
| M890 "0.6 PFLOPS FP16" | **Aggregator-only** (Toutiao/Zhihu). Not corroborated by any primary source; omitted from the survey rather than recorded low-confidence |
| M890 "launched" / "mass production" | **Overstated.** "Formally debuted"; no MP or GA claim for the chip |
| 560,000 units as M890 shipments | **Mis-attribution.** Cumulative Zhenwu-series |
| V900 specs as "TrendForce analyst estimate" | **Mis-attribution.** Chinese-media-reported vendor disclosure via Mydrivers/Sohu/Lanjinger/Baidu Baike |
| Panjiu AL128 as a brand-new scale-up domain | **Novelty overstated.** Published 2025-11-14 |
| Customers XPeng, Sina Weibo | **Unconfirmed** for this announcement |
| "Alibaba confirmed" the T-Head IPO | **Verb wrong.** Bloomberg-sourced media report of plans; no company confirmation |

---

## 13. Not found in this window

- No T-Head presentation confirmed at **ISCA 2026** or **ISSCC 2026**.
- **Hot Chips 38 runs 2026-08-23–25**, i.e. after this investigation date. No T-Head talk is confirmed on its program, and no slides, abstracts or specs from that conference exist yet.
- No **MLPerf** submission found for any T-Head part.
- **XuanTie C950:** no tape-out, sampling or availability update after the March 2026 launch already recorded in §4.
- No update to **Hanguang 800**.

# Investigation Update — 2026-09-13

**Investigated:** 2026-09-13 (window: 2026-08-08 → 2026-09-13)
**Change class:** Roadmap — commercial deployment scale-up only; no new chip, spec, or architecture fact
**Primary source:** Kuai Keji / MSN (2026-08-20)

## 14. M890 commercialization update

Kuai Keji (2026-08-20, "真武M890芯片商业化提速") reports T-Head's Zhenwu line now deployed with **650+ external customers across 20+ industries**, up from the 400+ figure recorded in the 2026-08-08 update. The article references a **2026-06-16** announcement — predating the 2026-08-08 baseline and not previously captured — that financial-sector Zhenwu deployment alone exceeded **100,000 cards across 150+ institutions**.

This headline is the first to name **M890 specifically** in a commercialization-scale claim, incrementally stronger than the 2026-08-08 update's "M890-specific volume deployment not confirmed" hedge, but it remains **trade-press paraphrase of a vendor claim** — no primary Alibaba disclosure, no per-model unit count, no revenue figure. Treat as press-reported, vendor-sourced.

No new chip, spec, or roadmap change: V900 (3Q2027) / J900 (3Q2028), the 560,000-unit cumulative Zhenwu shipment figure, and M890/ICN Switch 1.0/Panjiu AL128 specs are all unchanged.

**Apsara Conference (云栖大会) 2026 — could not confirm a date.** Historically late October; if repeated, it falls after this scan's cutoff regardless. Not recorded as a finding.

## 15. Sources — added 2026-09-13
- Kuai Keji / MSN — "真武M890芯片商业化提速：650家外部客户已上车，覆盖20+行业" (2026-08-20): https://www.msn.com/zh-cn/news/other/%E5%B9%B3%E5%A4%B4%E5%93%A5%E7%9C%9F%E6%AD%A6m890%E8%8A%AF%E7%89%87%E5%95%86%E4%B8%9A%E5%8C%96%E6%8F%90%E9%80%9F-650%E5%AE%B6%E5%A4%96%E9%83%A8%E5%AE%A2%E6%88%B7%E5%B7%B2%E4%B8%8A%E8%BD%A6-%E8%A6%86%E7%9B%9620-%E8%A1%8C%E4%B8%9A/ar-AA2aywhK
