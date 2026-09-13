# T-Head (平头哥) Hardware Architecture

*as_of: 2026-08-08*
*generations: Hanguang 800 / Zhenwu 810E (PPU) / Zhenwu M890 — roadmap: Zhenwu V900 (3Q2027), Zhenwu J900 (3Q2028); Xuantie C910 → C950*

**Chip:** alibaba-t-head  
**Device Class:** RISC-V AI (平头哥)  
**Investigated:** 2026-04-05 · **Last updated:** 2026-08-08  
**Sources:** Hot Chips HC32 2020, T-Head official site, TechNode, TrendForce, Tom's Hardware, EE Times; (2026-08-08 update) Alibaba Group newsroom 2026-05-20, Alibaba Cloud blog, The Register 2026-05-22, YiCai Global, ITHome, The Next Web, SCMP, T-Head news post 2026-07-18

---

## 1. Product Family Overview

T-Head Semiconductor (平头哥半导体, Pingtouge) is Alibaba's chip subsidiary, founded 2018.  
Two main product lines:

| Product Line | Focus | Latest Generation |
|---|---|---|
| **Hanguang / Zhenwu accelerators** (含光 / 真武) | Cloud AI inference/training ASIC | **Zhenwu M890 (announced 2026-05-20)** |
| **Xuantie CPU** (玄铁) | RISC-V general-purpose + AI-capable | C950 (March 2026) |

### Accelerator Generation Overview

| Generation | Announced | Process | Memory | Inter-chip BW | Data types | Scale-up domain | Status |
|---|---|---|---|---|---|---|---|
| Hanguang 800 | 2019-09 | TSMC 12nm | 192 MB SRAM only (no HBM) | none (PCIe Gen4 x16) | INT8 (primary) | none — single chip | Internal Alibaba Cloud only |
| Zhenwu 810E (PPU) | 2026-01 | 7nm domestic (likely SMIC) | 96 GB HBM2e | 700 GB/s (7× ICN links) | BF16/FP16/INT8 (inferred) | 10,000-card clusters over ICN | In production; 470K units by Feb 2026 |
| **Zhenwu M890** | **2026-05-20** | **Not disclosed** | **144 GB "on-chip memory" (type not vendor-confirmed)** | **800 GB/s** | **FP32 → FP4 native (vendor wording)** | **Panjiu AL128: 128 accelerators/unit (64-card option) over ICN Switch 1.0** | **Debuted; AL128 platform available via Bailian. No mass-production or GA claim for the chip** |
| *Zhenwu V900* | *roadmap: 3Q2027* | *Not disclosed* | *216 GB (Chinese-media-reported)* | *1,200 GB/s (Chinese-media-reported)* | *Not disclosed* | *Not disclosed* | *Roadmap only; 3× M890 claimed; "deeply iterated in-house parallel computing architecture"* |
| *Zhenwu J900* | *roadmap: 3Q2028* | *Not disclosed* | *Not disclosed* | *Not disclosed* | *Not disclosed* | *Not disclosed* | *Roadmap only; no specs disclosed* |

*V900/J900 figures are Chinese-media-reported vendor disclosures from the 2026-05-20 summit (Mydrivers via TrendForce; Sohu, Lanjinger, Toutiao, Baidu Baike) and are **absent from Alibaba's English newsroom**. They are not analyst estimates, but they are secondary.*

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
- Annualized revenue > RMB 10 billion; 470,000 units delivered by Feb 2026 (cumulative **Zhenwu-series** shipments reached **560,000 by May 2026** — see §3b)

---

## 3b. Zhenwu M890 — announced 2026-05-20 (added 2026-08-08)

Formally debuted at the Alibaba Cloud Summit on **2026-05-20** (Reuters, YiCai Global, ITHome, Alibaba Group newsroom; CNBC's 2026-05-19 dateline is US time). Presented as an integrated training + inference part for agentic workloads.

### Process & Memory
- **Process / foundry:** **not disclosed.** SMIC vs TSMC is unconfirmed; The Register and The Next Web note only that the part must use a domestically-accessible node.
- **Memory capacity:** **144 GB**
- **Memory type:** Alibaba's release says only **"on-chip memory"**. Secondary coverage calls it HBM and some outlets specifically HBM3 — **neither is vendor-confirmed**; recorded here as not disclosed.
- **Memory bandwidth:** **not disclosed**
- **Inter-chip bandwidth:** **800 GB/s** (up from 700 GB/s on the 810E)
- **TDP, die size, transistor count:** **not disclosed**

### Compute
- Peak throughput by data type is **not disclosed** in any official release. A 0.6 PFLOPS FP16 figure circulates on Chinese aggregators (Toutiao/Zhihu) with no primary source and is deliberately **not recorded** as a spec here.
- **Data types:** native support **"from FP32 down to FP4"** (Alibaba newsroom wording). This is the first FP4 claim in the Zhenwu line; the 810E was recorded at BF16/FP16/INT8.
- **Vendor performance claim:** "three times the performance of its predecessor, Zhenwu 810E" — marketing claim, **no benchmark or methodology disclosed**.
- Microarchitecture (array type, unit counts, SRAM capacity, clock) is **not disclosed**. As with the 810E, T-Head has published no block diagram.
- **Internal codename: not disclosed.** The line is generically called *PPU*; a "PPU 1.5" designation circulating in aggregators is not confirmed by any retrieved source.

### Status
Alibaba's newsroom states the M890 was **"formally debuted"** and makes **no mass-production or general-availability claim for the chip**. What is stated as available is the **Panjiu AL128 supernode, "now available through Alibaba's model service platform, Bailian"**. Per-chip pricing and M890-specific shipment volumes were **not disclosed** (The Next Web). The 560,000-unit figure below is **cumulative Zhenwu-series**, not M890.

### Deployment base (as of 2026-05)
- **560,000 cumulative Zhenwu-series units shipped** (up from 470,000 at the 2026-04-05 baseline)
- **400+ external customers across 20+ industries** (Alibaba newsroom; The Register)
- Named customers independently confirmed: **China Telecom, FAW Group, Shanghai Pudong Development Bank** (YiCai Global, Baidu Baike). XPeng and Sina Weibo appear in some aggregator coverage of this announcement but are **not confirmed** for it.

---

## 3c. Scale-up: Panjiu AL128 Supernode and ICN Switch 1.0 (added 2026-08-08)

This is the first rack-scale scale-up domain recorded for T-Head. Prior to it, the repo recorded only loose 10,000-card clusters built from 7× per-chip ICN links on the 810E.

### Panjiu AL128 Supernode
- **128 AI accelerators in a single rack-scale unit**; a **64-card configuration** is also offered.
- **"Petabyte-per-second (PB/s) internal bandwidth"** — vendor claim (Alibaba newsroom), repeated independently by The Register 2026-05-22. No per-link breakdown, topology diagram or bisection figure published.
- Available through Alibaba's **Bailian** model service platform.
- **Novelty caveat:** the "Panjiu AL128" name and its ScaleUp / ScaleOut / DCN interconnect architecture were published by Alibaba Cloud on **2025-11-14**, before this repo's baseline. What is new on 2026-05-20 is the **M890 + ICN Switch 1.0 instantiation**, not the supernode concept.

### ICN Switch 1.0
T-Head-developed switch ASIC — the switched counterpart to the point-to-point ICN links of the 810E.

| Parameter | Value | Notes |
|---|---|---|
| Aggregate bandwidth | **up to 25.6 Tbps** | Alibaba newsroom; The Register |
| Congestion-free domain | **64 accelerators** | Alibaba newsroom |
| Chip-to-chip latency | **hundred-nanosecond class** (百纳秒级) | ITHome, YiCai Global. BigGo reports "under 150 ns". **No latency figure appears in Alibaba's English release.** ⚠️ A "sub-100 ns" figure in circulation is a mistranslation of 百纳秒级 and is wrong |
| Port count / SerDes / process | **Not disclosed** | — |

**Context:** The Register notes that Broadcom and NVIDIA shipped switch silicon at comparable aggregate bandwidth years earlier — 25.6 Tbps is a credible first in-house part, not a frontier figure.

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
- Domestic 7nm process for Zhenwu 810E vs TSMC 4/3nm for leading competitors

### Added 2026-08-08
- **Zhenwu M890 process node and foundry are not disclosed** — the survey cannot place it on a process-normalized axis
- **No peak-FLOPS figure by data type for M890** in any official release; the "3× the 810E" claim has no disclosed benchmark, and the 810E itself has no published FLOPS baseline, so the multiplier cannot be resolved into an absolute number
- **M890 memory type not disclosed** — "on-chip memory" is Alibaba's only wording; HBM/HBM3 attributions are secondary-source only
- **ICN Switch 1.0 latency is not vendor-confirmed** in the English release, and no port count, SerDes rate or process is published
- **Panjiu AL128 "PB/s internal bandwidth"** has no published per-link or bisection breakdown
- **V900 / J900 roadmap figures appear only in Chinese media**, not in Alibaba's English newsroom
- **No T-Head presentation confirmed** at ISCA 2026 or ISSCC 2026; **no MLPerf submission** found. Hot Chips 38 runs 2026-08-23–25, after this update — no T-Head talk is confirmed on its program and no content from that conference exists yet
- **XuanTie C950:** no tape-out, sampling or availability update found since the March 2026 launch
