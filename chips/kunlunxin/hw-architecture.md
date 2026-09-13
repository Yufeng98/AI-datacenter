# Kunlunxin XPU Hardware Architecture

*as_of: 2026-08-08*
*research_baseline: 2026-04-05*
*chip: kunlunxin*
*device_class: AI Accelerator (百度昆仑芯)*

See `hw-programming-model.dot` / `hw-programming-model.png` for the visual diagram.

---

## Architecture Summary

The Kunlunxin XPU is a **cluster-based multi-core SIMD accelerator** with three shipped generations and a fourth announced:

| Gen | Chip | Process | FP16 TFLOPS | Memory | BW | Status (2026-08-08) |
|-----|------|---------|-------------|--------|----|---------------------|
| 1 | Kunlun 1 (XPU-K) | Samsung 14nm | ~64 | 16 GB HBM2 | 512 GB/s | Deployed 2019 |
| 2 | R200/R300 (XPU-R) | 7nm | 128 | 32 GB GDDR6 | 512 GB/s | Mass production 2021–22 |
| 3 | P800 (XPU-P) | 7nm | 345 | 96 GB HBM3 | ~1.6 TB/s | Mass production; 30,000-card Baidu cluster |
| **4** | **M100** (inference) | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** | Announced 2025-11-13; **first physical public showing WAIC 2026 (2026-07-17 → 07-20) with no official materials released and carrier boards still in development** |
| 4 | M300 (training + multimodal) | not disclosed | not disclosed | not disclosed | not disclosed | Announced 2025-11-13; targeted early 2027 |

> **M100 has no published specification of any kind** — no process node, HBM capacity, memory bandwidth, TFLOPS/TOPS or TDP. The only HBM-related statement in existence is unattributed WAIC booth-staff commentary that capacity is "somewhat lower than NVIDIA H20" (low confidence). Nothing is estimated here. See the 2026-08-08 update section at the end of this document.

---

## Compute Engine

Each XPU die contains **N compute clusters**, each comprising:

1. **SDNN (Spatial DNN Accelerator)** — systolic-array-style MAC array:
   - Ops: GEMM, Conv, Deconv, elementwise (INT8/FP16/BF16)
   - Purpose: all large tensor operations (linear layers, attention projections, convolutions)

2. **XVME (XPU Vector Math Engine)** — SIMD vector unit:
   - Ops: activations (GELU, SiLU, ReLU), LayerNorm, RMSNorm, reductions, elementwise math
   - Purpose: all non-GEMM compute (post-GEMM ops, attention softmax, normalization)

3. **Cluster-local SRAM scratchpad** — software-managed (no hardware caches):
   - Explicitly loaded by XTCK/XDNN kernels
   - High aggregate bandwidth (estimated hundreds TB/s across all clusters)

A **shared SRAM buffer** at the chip level spans all clusters for inter-cluster communication.

A **SPMD dispatch controller** broadcasts ops across all clusters in lockstep.

### Gen 4 (M100) compute — what is and is not known

M100 is described by Baidu only as continuing the **self-developed XPU architecture**, **inference-optimised**, and built on a **"fully domestic supply chain"** (vendor framing, Baidu World 2025 / WAIC 2026). Whether the SDNN + XVME cluster structure carries forward unchanged **has not been stated**. Cluster count, MAC array dimensions, clock, supported data types and any FP8/FP4 path are all **not disclosed**. No die photo, block diagram or specification sheet has been published, and show-floor reporting at WAIC 2026 confirms no official introduction materials accompanied the physical unit.

---

## Memory Hierarchy

| Level | Gen 1 | Gen 2 | Gen 3 (P800) | Gen 4 (M100) |
|-------|-------|-------|--------------|--------------|
| On-chip SRAM | ~16 MB (distributed) | ~32 MB (distributed) | ~64 MB (estimated) | not disclosed |
| Off-chip | HBM2 16 GB 512 GB/s | GDDR6 32 GB 512 GB/s | HBM3 96 GB ~1.6 TB/s | HBM assumed but **capacity, type and bandwidth all not disclosed** |

Memory is **software-managed** — no hardware cache hierarchy. XDNN kernels explicitly DMA data between HBM/GDDR6 and cluster SRAM scratchpads.

---

## Scale-up Interconnect

- **Gen 2 (R200/R300)**: PCIe Gen4 x16 only; 8 cards per server; no dedicated chip-to-chip fabric
- **Gen 3 (P800)**: XLINK chip-to-chip fabric at **200 GB/s bidirectional per port**; 8-card server topology

Above the 8-card server sits a **cabinet supernode ladder** built on Baidu's self-developed **XPU-Link** protocol. All performance figures in this ladder are **vendor claims** relayed through Chinese media reporting of Baidu executive remarks — none is independently benchmarked:

| Tier | Cards | Status (2026-08-08) | Vendor claims |
|------|-------|---------------------|---------------|
| Conventional server | 8 | Shipping | XLINK 200 GB/s per port |
| Kunlun Super Node (cabinet) | 32 / 64 | **Launched April 2025**; vendor says among the first domestic supernodes in volume delivery; re-displayed at WAIC 2026 | **8×** inter-card communication bandwidth vs the 8-card server; **13×** single-card inference efficiency; **5–10×** MoE training performance (a range); cabinet architecture claimed extensible to 512 cards |
| Tianchi 256 (天池256卡超节点) | 256 | "Lit up" (点亮) April 2026; stated on sale June 2026 — **volume shipping not independently confirmed** (Tencent News 2026-07-22 still says "即将全面上市") | +25% throughput and +50% inference efficiency vs prior generation (**Create 2026, 2026-05-13**); 4× total inter-card bandwidth and 3.5× single-card token throughput (**Baidu World 2025, 2025-11-13 — pre-baseline figures, not a 2026 result**) |
| Tianchi 512 | 512 | **Not shipped**; H2 2026 target | Trillion-parameter model training within a single supernode. *No inter-card-bandwidth multiplier vs Tianchi 256 has been published — the "~2×" figure circulating in aggregators is not recorded here.* |

**XPU-Link fabric** (disclosed at WAIC 2026, 2026-07-22; vendor claims): Baidu's self-developed interconnect protocol with **programmable switching**, **self-designed copper cabling**, and a **liquid-cooling CDU**, claiming **77% measured bandwidth efficiency**. Baidu stated a roadmap *intent* to build **thousand-card and 4,000-card supernodes** on the Kunlun M-series, with **no timeline given**.

Scale-out: standard Ethernet or InfiniBand at the cluster level. Baidu's 30,000-P800 cluster uses an undisclosed interconnect fabric; Baidu claims **97% effective training efficiency** training Wenxin 5.1 on P800 ten-thousand-card clusters (vendor claim, Create 2026).

---

## Key Differentiators

- **P800 HBM3 96 GB** enables full DeepSeek V3/R1 671B on a single 8-card server (no scale-out)
- **345 TFLOPS FP16** (P800) outperforms Nvidia H20 (148 TFLOPS) and matches Ascend 910B (256 TFLOPS)
- **Software-managed SRAM** model (like Ascend, Neuron) rather than GPU hardware caches
- XLINK 200 GB/s chip-to-chip plus the **XPU-Link cabinet supernode ladder** (32/64 → 256 → 512 cards) is Kunlunxin's answer to the NVLink scale-up gap, and is the dimension on which the product line moved most in 2025–2026

---

## Update — 2026-08-08 (scale-up ladder corrected, M100 physical debut)

*Window covered: 2026-04-01 → 2026-08-08 against the 2026-04-05 baseline. **Sourcing caveat:** kunlunxin.com's newsroom returns HTTP 403 to automated retrieval, so no vendor-primary spec page was obtainable. Every supernode number rests on Chinese-media transcription of executive remarks at Baidu Create 2026 (2026-05-13) and WAIC 2026 (2026-07-17 → 07-22).*

**What changed in this document**

1. **Scale-up section rewritten.** The baseline recorded only "8 cards/server" plus two planned Tianchi fabrics. The real ladder has four tiers, and the **32/64-card cabinet supernode launched in April 2025** — a full year before the repo baseline. It was missed, not new; only its re-display at WAIC 2026 falls in the window.
2. **Tianchi 256 dating corrected.** Shen Dou (沈抖, Baidu EVP / Baidu AI Cloud president) said at Create 2026 that the unit was **"lit up" (点亮)** in April 2026 — powered on, *not* "completed testing" — and would go on sale in June 2026. That June date was forward-looking. Independent confirmation of availability is thin and a 2026-07-22 Tencent News wrap-up still calls both Tianchi 256 and 512 "即将全面上市". No customer, deployment or unit volume was found.
3. **Tianchi 256 performance figures split by date.** Only "+25% throughput" and "+50% inference efficiency" are new (Create 2026, 2026-05-13). "4× inter-card bandwidth" and "3.5× single-card token throughput" are the **original Baidu World 2025 (2025-11-13) announcement figures** and must not be presented as an April/June 2026 result. Vendor messaging is itself inconsistent — WAIC-period coverage says "整体性能提升50%" (overall performance +50%) instead.
4. **32/64-card claims stated as published.** "5–10× MoE training performance" is a range; the 10× figure alone is an overstatement. The 13× single-card inference figure does check out as a vendor claim.
5. **New fabric disclosure recorded**: XPU-Link protocol with programmable switching, self-designed copper cabling, liquid-cooling CDU, 77% claimed measured bandwidth efficiency; stated intent for thousand-card and 4,000-card M-series supernodes with no timeline.
6. **Gen 4 M100 added to the generation table with every spec field marked "not disclosed."** M100's first physical public appearance was WAIC 2026 (2026-07-17 → 07-20), confirmed by multiple independent outlets. 芯智讯 show-floor reporting states no official introduction materials were released and carrier boards were "正在制作中" (still being made). Booth staff positioned it against NVIDIA H20 on inference cost-effectiveness with HBM capacity somewhat lower than H20 — unattributed, low confidence, and the only HBM-related statement that exists. The circulating claim that M100 "entered a commercial ramp in January 2026" rests on Baidu Baike (user-editable) and is hard to reconcile with the July 2026 carrier-board finding; it is recorded at low confidence only.
7. **Not populated on purpose**: "Tianchi 512 ≈ 2× Tianchi 256 inter-card bandwidth" (no source), any M100 spec value, any M300 detail beyond the early-2027 target.

**Searched and absent:** no MLPerf Training v5.x / Inference v6.x submission; no Hot Chips 2026, ISCA 2026 or ISSCC 2026 Kunlunxin paper or talk (Hot Chips 38, 2026-08-23 → 08-25, has no Kunlunxin session on its program); no new XRE/XTDK/XDNN/XTCL public release notes in the window.

---

## Sources

- [Hot Chips 2020 — Baidu Kunlun (IEEE)](https://ieeexplore.ieee.org/document/9366056)
- [Baidu Kunlun II — Tom's Hardware](https://www.tomshardware.com/news/baidu-unveils-kunlun-ii-processor-for-ai)
- [P800 前世今生 — CSDN](https://blog.csdn.net/Rong_Toa/article/details/151322568)
- [昆仑芯3代P800万卡点亮 — 新浪科技](https://finance.sina.com.cn/tech/roll/2025-02-05/doc-ineimcyh2167561.shtml)
- [Baidu 3rd-gen Kunlun cluster — Digitimes](https://www.digitimes.com/news/a20250208PD210/baidu-ai-chip-chips-2024-production.html)
- [M100/M300 roadmap — TrendForce](https://www.trendforce.com/news/2025/11/13/news-baidu-rolls-out-kunlun-roadmap-m100-m300-ai-chips-arrive-2026-2027/)

### Added 2026-08-08

- [Baidu Create 2026 — Shen Dou on Tianchi 256 "点亮" April 2026 / on sale June 2026 (腾讯新闻, 2026-05-13)](https://news.qq.com/rain/a/20260513A0688800)
- [WAIC 2026 wrap-up — XPU-Link, programmable switching, 77% bandwidth efficiency, "即将全面上市" (腾讯新闻, 2026-07-22)](https://news.qq.com/rain/a/20260722A04PTE00)
- [芯智讯 WAIC show-floor report — M100 physical debut, no official materials, carrier boards in development (腾讯新闻, 2026-07-20)](https://news.qq.com/rain/a/20260720A03JPT00)
- [M100 first physical appearance at WAIC 2026 (同花顺, 2026-07-17)](https://stock.10jqka.com.cn/20260717/c678258367.shtml)
- [Baidu World 2025 original Tianchi 256 figures — 4× inter-card BW, 3.5× single-card token throughput (新浪财经, 2025-11-13)](https://finance.sina.com.cn/tech/2025-11-13/doc-infxfnye5573223.shtml)
- [WAIC 2026 supernode coverage (新浪财经, 2026-07-22)](https://finance.sina.cn/tech/2026-07-22/detail-iniistty7951726.d.html)
- [WAIC 2026 China supernode push — TrendForce (2026-07-20)](https://www.trendforce.com/news/2026/07/20/news-waic-2026-highlights-chinas-supernode-push-led-by-huawei-minimax-m3-unitree-robots-in-focus/)
