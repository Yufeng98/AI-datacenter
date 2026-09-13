# Kunlunxin XPU Hardware Architecture

*as_of: 2026-04-05*
*chip: kunlunxin*
*Architectures: XPU-K (Gen 1, 14nm), XPU-R (Gen 2, 7nm R200/R300), XPU-P (Gen 3, 7nm P800)*

---

## Overview

Kunlunxin (昆仑芯) is the AI chip subsidiary spun off from Baidu in 2021, producing the **XPU** (AI Processing Unit) family. The XPU architecture is a **multi-core SIMD/VLIW accelerator** designed for both training and inference across vision, NLP, speech, recommendation, and LLM workloads.

Three generations have shipped to date:
- **Gen 1 (XPU-K)**: Kunlun 1, Samsung 14 nm (2019); designed for diversified inference at Baidu data-centers
- **Gen 2 (XPU-R)**: Kunlun 2, Samsung/TSMC 7 nm (2021–22); R200 (SKU-T, training) and R300 (SKU-I, inference); 2–3× Gen 1 in peak throughput
- **Gen 3 (XPU-P)**: P800, 7 nm; HBM3; 345 TFLOPS FP16; full DeepSeek V3/R1 671B on 8 cards; 30,000-card cluster deployed by Baidu in 2025

A Gen 4 roadmap (M100 inference / M300 training) was unveiled in late 2025, targeting 2026–2027.

---

## 1. Compute Engine

### XPU Architecture Philosophy

The XPU is a **cluster-based SIMD accelerator** with two principal compute engines:
1. **SDNN (Spatial DNN accelerator)** — systolic-array-style MAC array optimized for large tensor ops: GEMM, Conv, Deconv, and elementwise math
2. **XVME (Vector Math Engine)** — general-purpose vector unit for activation functions, normalizations, reductions, and non-regular compute

Each XPU die contains multiple clusters; each cluster contains SDNN MACs + XVME units + local SRAM scratchpad. The architecture is explicitly **multi-core**: all clusters execute in SPMD fashion under a unified dispatch model.

### Generation Specifications

| Spec | Gen 1 (Kunlun 1 / XPU-K) | Gen 2 (Kunlun 2 / XPU-R: R200/R300) | Gen 3 (P800 / XPU-P) |
|------|--------------------------|--------------------------------------|----------------------|
| Process | Samsung 14 nm | Samsung/TSMC 7 nm | 7 nm |
| Peak FP16 TFLOPS | ~64 | 128 (FP16) | 345 |
| Peak INT8 TOPS | ~230–256 | 256 | 690 (estimated) |
| Memory | HBM2 16 GB | GDDR6 | HBM3 96 GB |
| Memory BW | ~512 GB/s | 512 GB/s | ~1.6 TB/s (estimated) |
| TDP | 160 W | ~250 W | ~350 W |
| Production year | 2019 | 2021–2022 | 2024–2025 |
| Package | Samsung I-Cube 2.5D (T variant) | Standard / 2.5D | Advanced |

### Hot Chips 2020 Published Specs (Kunlun 1)

From the public Hot Chips 2020 paper:
- **Peak INT8**: 230 TOPS @ 900 MHz nominal; 281 TOPS @ 1.1 GHz boost
- **Memory BW**: 512 GB/s (HBM2)
- **On-chip SRAM**: ~16 MB SRAM distributed across clusters
- **Die size**: ~348 mm² (Samsung 14 nm)
- **TDP**: 160 W

### P800 (Gen 3) Key Differentiators

- **345 TFLOPS FP16** — comparable to Huawei Ascend 910B and Nvidia A100
- **HBM3 96 GB** — sufficient to fit DeepSeek V3/R1 671B (BF16) across 8 cards (single server, no scale-out required)
- **200 GB/s chip-to-chip interconnect** (XLINK or equivalent)
- 30,000-P800 cluster operational at Baidu by early 2025
- First domestic chip to independently serve full-power DeepSeek V3/R1 671B training + inference
- China Mobile 10-billion-RMB P800 procurement awarded August 2025

---

## 2. Memory Hierarchy

### Gen 2 (XPU-R: R200/R300)

| Level | Type | Capacity | Bandwidth |
|-------|------|----------|-----------|
| L1 (cluster local) | SRAM scratchpad | ~4–8 MB distributed | >10 TB/s aggregate |
| L2 (chip shared) | SRAM shared cache | ~16–32 MB | ~3–5 TB/s |
| Off-chip | GDDR6 | 32 GB (R200) / 32 GB (R300) | 512 GB/s |

### Gen 3 (P800)

| Level | Type | Capacity | Bandwidth |
|-------|------|----------|-----------|
| On-chip SRAM | Distributed scratchpad | ~64 MB (estimated) | High (hundreds TB/s aggregate) |
| Off-chip | HBM3 | 96 GB | ~1.6 TB/s (estimated) |
| Chip-to-chip | XLink fabric | 8 cards / server | 200 GB/s |

---

## 3. Data Precision Support

| Precision | Gen 1 | Gen 2 (R200/R300) | Gen 3 (P800) |
|-----------|-------|-------------------|--------------|
| FP32 | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes |
| BF16 | Partial | Yes | Yes |
| INT8 | Yes (primary) | Yes | Yes |
| INT4 | No | Limited | Yes |
| FP8 | No | No | Planned/partial |

---

## 4. Scale-up Interconnect

### Gen 2 (XPU-R)

- **MLU-style PCIe + NVSwitch-equivalent**: up to 8 cards per server via PCIe Gen4 x16
- No dedicated chip-to-chip fabric on R200/R300; standard PCIe for multi-card

### Gen 3 (P800)

- **XLINK chip-to-chip**: 200 GB/s bidirectional per pair; 8 cards per server = 56 GB/s allreduce bandwidth (estimated ring)
- **Tianchi 256 SuperNode** (天池256超节点): 256-card interconnect fabric, announced for H1 2026
- **Tianchi 512 SuperNode** (天池512超节点): 512-card fabric for trillion-parameter training, announced for H2 2026

Scale-out uses standard Ethernet or InfiniBand at the rack/cluster level. The 30,000-P800 cluster uses an undisclosed high-speed fabric.

---

## 5. Die and Package Configuration

### Kunlun 1 (XPU-K)

- **Package**: Samsung I-Cube 2.5D HBM2 package (Baidu Kunlun1-T variant)
- Die attached to interposer; 2 HBM2 stacks (8 GB each = 16 GB)
- Samsung 14 nm logic die

### Kunlun 2 / R200 (XPU-R)

- **Package**: Standard PCB BGA with discrete GDDR6 chips
- 7 nm logic die; GDDR6 off-package
- Floorplan: systolic MAC array (SDNN) + vector units + SRAM + PCIe controllers

### P800 (XPU-P)

- **Package**: Advanced package with integrated HBM3 stacks
- 7 nm logic die; 96 GB HBM3 on-package
- Dual-die configuration suspected (not confirmed publicly)

---

## 6. Architectural Comparison

| Dimension | Kunlunxin XPU (P800) | Huawei Ascend 910B | Nvidia A100 |
|-----------|----------------------|-------------------|-------------|
| Process | 7 nm | SMIC ~14 nm (N+1) | TSMC 7nm |
| FP16 TFLOPS | 345 | 256 | 312 |
| Memory | 96 GB HBM3 | 64 GB HBM2 | 80 GB HBM2e |
| Memory BW | ~1.6 TB/s | ~1.2 TB/s | 2.0 TB/s |
| Execution model | Multi-core SIMD + systolic SDNN | Da Vinci Cube (systolic) + Vector | SIMT + Tensor Core |
| Programming | XTDK / XRE / XDNN | AscendCL / CANN | CUDA |
| Framework | PaddlePaddle native | MindSpore native | PyTorch/TF native |

---

## 7. Roadmap

| Generation | Chip | Target Use | Process | Status (as of 2026-04) |
|------------|------|-----------|---------|----------------------|
| Gen 1 | Kunlun 1 (XPU-K) | Inference (edge + DC) | Samsung 14nm | Deployed (2019) |
| Gen 2 | Kunlun 2 R200/R300 (XPU-R) | Training + Inference | 7nm | Mass production (2021–22) |
| Gen 3 | P800 (XPU-P) | Training + Inference | 7nm | Mass production + 30K cluster (2024–25) |
| Gen 4 | M100 | Large-scale inference | TBD | Announced 2025, launch 2026 |
| Gen 4 | M300 | Training + inference | TBD | Announced 2025, launch 2027 |

---

## Sources

- [Baidu Kunlun: A 14nm High-Performance AI Processor — Hot Chips 2020 (IEEE)](https://ieeexplore.ieee.org/document/9366056)
- [Hot Chips 2020 Kunlun Slide Deck (PDF)](https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Inference_Baidu_Kunlun_v5.pdf)
- [Baidu Kunlun II AI Chip: Rival for Nvidia A100 — Tom's Hardware](https://www.tomshardware.com/news/baidu-unveils-kunlun-ii-processor-for-ai)
- [Baidu Kunlunxin Spinoff and Listing — TrendForce](https://www.trendforce.com/news/2025/12/09/news-baidu-reportedly-explores-kunlunxin-spinoff-and-listing-amid-a-surge-in-china-chip-ipos/)
- [Baidu 3rd-gen Kunlun 10,000-GPU cluster — Digitimes](https://www.digitimes.com/news/a20250208PD210/baidu-ai-chip-chips-2024-production.html)
- [Kunlunxin Wikipedia](https://en.wikipedia.org/wiki/Kunlunxin)
- [Kunlunxin P800 Virtualization Guide — RiseUnion](https://www.theriseunion.com/blog/HAMi-kunlunxin-p800-support.html)
- [Baidu M100/M300 Roadmap — TrendForce](https://www.trendforce.com/news/2025/11/13/news-baidu-rolls-out-kunlun-roadmap-m100-m300-ai-chips-arrive-2026-2027/)
- [Baidu Kunlun1-T I-Cube Package Analysis — TechInsights](https://www.techinsights.com/blog/baidu-kunlun1-t-ai-processor-samsung-i-cube-25d-package-technology-advanced-packaging-analysis)
- [Kunlunxin Hong Kong IPO Filing — SCMP](https://www.scmp.com/news/china-future-tech/semiconductors/article/3338471/baidu-chip-unit-kunlunxin-files-hong-kong-ipo-amid-chinas-push-tech-self-reliance)
- [昆仑芯2代介绍 — 昆仑芯官网](https://www.kunlunxin.com/product/2873.html)
- [昆仑芯P800前世今生 — CSDN](https://blog.csdn.net/Rong_Toa/article/details/151322568)

---

## Investigation Update — 2026-08-08

*Window: 2026-04-01 → 2026-08-08, against the 2026-04-05 baseline above. Everything in this section survived adversarial verification; several items **correct errors in the baseline sections above** and those corrections are noted explicitly.*

### Method and its limits (read before citing anything below)

- WebSearch quota was exhausted before this verification pass began, so discovery ran through WebFetch against DuckDuckGo HTML search plus direct primary-source fetches.
- **kunlunxin.com returns HTTP 403 to automated retrieval**, so **no vendor-primary spec page or newsroom item was obtainable**. Every supernode performance figure below traces to Chinese-media transcription of Baidu executive remarks at Baidu Create 2026 (2026-05-13, Shanghai) or WAIC 2026 (2026-07-17 → 07-22, Shanghai). None is independently benchmarked, and none is backed by a retrieved vendor spec sheet.
- Where a value is not public it is written **"not disclosed"**. Nothing is estimated in this section.

### 1. Scale-up domain — the materially new area

The baseline recorded Kunlunxin scale-up as "8 cards/server" plus two planned Tianchi fabrics. The correct ladder has four tiers:

| Tier | Cards | Status 2026-08-08 | Confidence |
|------|-------|-------------------|-----------|
| Conventional server | 8 | Shipping (XLINK 200 GB/s per port) | confirmed |
| Kunlun Super Node (cabinet) | 32 / 64 | **Launched April 2025** — pre-baseline, missed at first pass; vendor: "国内率先实现量产交付的超节点产品之一"; re-displayed WAIC 2026 | medium (vendor claim relayed by media) |
| Tianchi 256 (天池256卡超节点) | 256 P800 cards / cabinet | "Lit up" (点亮) April 2026; stated to go on sale (正式上市) June 2026 | medium — *availability itself not independently confirmed* |
| Tianchi 512 | 512 | **Not shipped**; H2 2026 target | medium |

**Tianchi 256 status — precise wording matters.** At Baidu Create 2026 (2026-05-13) Baidu EVP / Baidu AI Cloud president **Shen Dou (沈抖)** said the unit had been **点亮 ("lit up" — powered on / brought online)** in April 2026 and would **正式上市 (go on sale)** in June 2026. Two corrections against the raw scan:

1. 点亮 is **not** "completed testing". Powering on a cabinet is a weaker milestone than completing a test campaign.
2. "Reached market June 2026" is a **forward-looking statement made on 2026-05-13**, not an observed fact. Chinese secondary sources (incl. Baidu Aiqicha) state a June launch, but a Tencent News WAIC wrap-up dated **2026-07-22** still describes both Tianchi 256 and 512 as **"即将全面上市"** (about to become fully available). No customer, deployment or unit volume was found. Best phrasing: *"announced generally available June 2026; volume shipping not independently confirmed."*

**Tianchi 256 performance claims must be split by date** — the baseline-era and in-window figures are routinely conflated by aggregators:

| Figure | First stated | In window? |
|---|---|---|
| +25% throughput vs prior generation | Create 2026, 2026-05-13 | **Yes** (new) |
| +50% inference efficiency | Create 2026, 2026-05-13 | **Yes** (new) — but vendor messaging is inconsistent: WAIC-period coverage (2026-07-22) says "整体性能提升50%" (overall performance +50%) instead |
| 4× total inter-card interconnect bandwidth | Baidu World 2025, 2025-11-13 | **No** — pre-baseline |
| 3.5× single-card token throughput, mainstream LLM inference | Baidu World 2025, 2025-11-13 | **No** — pre-baseline |

Vendor also states Tianchi 256 is adapted for Wenxin (文心), DeepSeek, GLM and MiniMax models (Create 2026).

**32/64-card Kunlun Super Node claims, as published:** 8× inter-card communication bandwidth vs the conventional single-server 8-card product; **13×** single-card inference efficiency; **"MoE training performance 5–10×"** — a *range*. The frequently repeated "10× training" is the top of that range presented as a point value and is **overstated**. The cabinet architecture is claimed extensible to 512 cards.

**Tianchi 512:** announced 2025-11-13, still H2 2026, claimed to train trillion-parameter models within a single supernode. The circulating figure **"~2× total inter-card bandwidth vs Tianchi 256" could not be confirmed in any retrieved source and is deliberately left unpopulated.**

**New fabric disclosure — WAIC 2026 (vendor claims, 2026-07-22):**
- Baidu self-developed **XPU-Link** interconnect protocol with **programmable switching**
- Self-designed **copper cabling** and **liquid-cooling CDU**
- Claimed **77% measured bandwidth efficiency**
- Stated roadmap **intent** for **thousand-card and 4,000-card supernodes** on the Kunlun M-series — **no timeline given**
- Separately, Baidu claims **97% effective training efficiency** training Wenxin 5.1 on P800 ten-thousand-card clusters (Create 2026)

### 2. M100 (Gen 4) — physical debut, zero specs

- Announced 2025-11-13 at Baidu World 2025 (pre-baseline).
- **First physical public appearance: WAIC 2026, Shanghai, 2026-07-17 → 07-20.** New and confirmed by multiple independent outlets (芯智讯 via Tencent News 2026-07-20; 10jqka 2026-07-18; Xueqiu).
- 芯智讯 show-floor reporting states explicitly that **no official introduction materials were released** and that the associated carrier boards were **"正在制作中"** (still being made).
- Booth staff positioned M100 against **NVIDIA H20** on inference cost-effectiveness, with **HBM capacity somewhat lower than H20**. Unattributed booth-staff commentary; **low confidence**; it is the only HBM-related statement that exists for M100.
- Vendor framing: **"fully domestic supply chain"**, continuing the self-developed XPU architecture, inference-optimised.
- **NO process node, HBM capacity, memory bandwidth, TFLOPS/TOPS or TDP published. No spec field is populated in this repo.**
- The claim that M100 "entered a commercial ramp (商业放量期) in January 2026" rests on **Baidu Baike (user-editable, not a primary source)** and secondary Chinese media, and is in direct tension with the July 2026 carrier-board finding. Recorded as **low confidence**: *"reportedly launched early 2026; a January 2026 commercial-ramp claim circulates in Chinese media but is unverified and hard to reconcile with July 2026 show-floor reporting."*
- **M300** (training + multimodal, targeted early 2027): unchanged, no new detail in the window.

### 3. Corporate — Hong Kong IPO

- Already in the repo at baseline ("Hong Kong IPO filed January 2026"). Precise facts: joint sponsors submitted a **Form A1 listing application to HKEX on a confidential basis on 2026-01-01**; Baidu announced the proposed spin-off and separate listing on **2026-01-02** (HKEXnews doc 2026010200013; Baidu IR). Both pre-baseline.
- **In-window but press-reported only, not in any filing:** Reuters (2026-06-28, citing The Information) reported a target valuation of **~USD 50 billion** with listing expected around **Q3 2026**; CNBC picked it up 2026-06-29 (Baidu HK shares +6%). 2025 revenue ~RMB 3.5bn / ~USD 500M at breakeven and Baidu retaining ~58% are likewise press-reported.
- Confidence **low-to-medium**: at least one report notes the target rose from ~USD 14.7bn cited roughly two weeks earlier — a fast-moving leak, not a stable figure. **Label as "press-reported target" wherever cited.** The listing had **not** occurred as of 2026-08-08.

### 4. Searched and absent (absence of evidence, not evidence of absence)

- No MLPerf Training v5.x or Inference v6.x submission by Kunlunxin or Baidu XPU.
- No Hot Chips 2026, ISCA 2026 or ISSCC 2026 paper or talk on Kunlunxin. Hot Chips 38 runs **2026-08-23 → 08-25** (15 days after this pass) and carries no Kunlunxin session on its published program.
- No new public XRE / XTDK / XDNN / XTCL release notes in the window.
- No published M100 specification of any kind.

### Sources added 2026-08-08

- [Baidu Create 2026 — Shen Dou on Tianchi 256 (腾讯新闻, 2026-05-13)](https://news.qq.com/rain/a/20260513A0688800)
- [WAIC 2026 wrap-up — XPU-Link, 77% BW efficiency, "即将全面上市" (腾讯新闻, 2026-07-22)](https://news.qq.com/rain/a/20260722A04PTE00)
- [芯智讯 WAIC show-floor — M100 physical debut, no materials, carrier boards in development (腾讯新闻, 2026-07-20)](https://news.qq.com/rain/a/20260720A03JPT00)
- [WAIC 2026 supernode coverage (腾讯新闻, 2026-07-20)](https://news.qq.com/rain/a/20260720A07PE200)
- [M100 at WAIC 2026 (同花顺, 2026-07-17)](https://stock.10jqka.com.cn/20260717/c678258367.shtml)
- [Kunlunxin WAIC preview (同花顺, 2026-07-16)](https://stock.10jqka.com.cn/20260716/c678228712.shtml)
- [Baidu World 2025 — original Tianchi 256 4× / 3.5× figures (新浪财经, 2025-11-13)](https://finance.sina.com.cn/tech/2025-11-13/doc-infxfnye5573223.shtml)
- [WAIC 2026 supernode coverage (新浪财经, 2026-07-22)](https://finance.sina.cn/tech/2026-07-22/detail-iniistty7951726.d.html)
- [天池256卡超节点 — 百度百科 (secondary, user-editable — used only for launch-month cross-check)](https://baike.baidu.com/item/%E5%A4%A9%E6%B1%A0256%E5%8D%A1%E8%B6%85%E8%8A%82%E7%82%B9/67795862)
- [昆仑芯M100 — 百度百科 (user-editable; source of the unverified "Jan 2026 commercial ramp" claim)](https://baike.baidu.com/item/%E6%98%86%E4%BB%91%E8%8A%AFM100/66989616)
- [Baidu spin-off / Form A1 announcement (HKEXnews, 2026-01-02)](https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/2026010200013.pdf)
- [Baidu IR — proposed spin-off and separate listing of Kunlunxin](https://ir.baidu.com/news-releases/news-release-details/baidu-announces-proposed-spin-and-separate-listing-kunlunxin/)
- [Reuters — Kunlunxin targets USD 50B Hong Kong IPO (2026-06-28)](https://www.reuters.com/world/asia-pacific/baidus-ai-chip-unit-kunlunxin-targets-50-billion-hong-kong-ipo-information-2026-06-28/)
- [CNBC — Baidu Kunlunxin Hong Kong IPO (2026-06-29)](https://www.cnbc.com/2026/06/29/baidu-kunlunxin-hong-kong-ipo.html)
- [TrendForce — WAIC 2026 China supernode push (2026-07-20)](https://www.trendforce.com/news/2026/07/20/news-waic-2026-highlights-chinas-supernode-push-led-by-huawei-minimax-m3-unitree-robots-in-focus/)
- [DataCenterDynamics — Baidu launches Kunlun M100 and M300](https://www.datacenterdynamics.com/en/news/baidu-launches-kunlun-m100-and-m300-ai-chips/)
