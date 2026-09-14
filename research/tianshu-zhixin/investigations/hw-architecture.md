# Iluvatar CoreX (天数智芯) GPU Hardware Architecture Investigation

*as_of: 2026-09-13*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*
*resource: hw-architecture*

---

## Summary

Shanghai Tianshu Zhixin Semiconductor Co., Ltd. (天数智芯; Iluvatar CoreX) is China's first and only publicly listed general-purpose GPU (GPGPU) company as of early 2026. Founded in 2015 by **Li Yunpeng** (李仲远), former Oracle R&D Director, the company produces full-stack SIMT GPGPUs designed for both AI training and inference. Unlike inference-only ASICs, Iluvatar CoreX builds fully programmable GPUs targeting cloud AI compute workloads.

The current flagship training GPU is the **TianGai-150 (Big Island Gen2)** (天垓150, product code BI-V150), built on TSMC 7nm with 64 GB HBM2 memory. The inference-oriented line is the **Zhikai series** (智铠), with the flagship being the MR-V100. As of June 2025, the company has delivered over **52,000 GPUs** to more than 290 customers.

The company listed on the **Hong Kong Stock Exchange (HKEX) on January 8, 2026** — the first Chinese GPGPU firm to achieve a Hong Kong IPO. It raised HK$3.7 billion (~USD 475 million) at a valuation of ~USD 4.6–5.3 billion.

---

## Product Generations

| Product | Series | Purpose | Process | Memory | Notes |
|---------|--------|---------|---------|--------|-------|
| TianGai-100 (BI-V100) | Big Island Gen1 | Training | TSMC 7nm | 32 GB HBM2 | First mass-produced Chinese 7nm GPGPU; released Jan 2021 |
| TianGai-150 (BI-V150) | Big Island Gen2 | Training | TSMC 7nm | 64 GB HBM2 | 2.5D CoWoS packaging; current flagship |
| Zhikai (MR-V100) | Zhikai | Inference | ~16nm (est.) | 32 GB GDDR6 | China's first dedicated GPGPU inference chip |
| Tianshu arch (2025) | Next-gen | Training + Inference | Advanced (est.) | HBM (est.) | Claims >Hopper performance; multiple benchmarks with 300+ clients |
| Tianxuan (2026) | Next-gen | Training + Inference | — | — | Targets NVIDIA Blackwell |
| Tianji (2026) | 3rd-gen | Training + Inference | — | — | Claims to surpass Blackwell |
| Tianquan (2027) | 4th-gen | Training + Inference | — | — | Targets NVIDIA Rubin by 2027 |

---

## TianGai-100 (BI-V100) — First Generation

### Chip Identity

| Attribute | Value |
|-----------|-------|
| Product code | BI-V100 |
| Architecture name | Big Island (大岛) |
| Process | TSMC 7nm |
| Packaging | 2.5D CoWoS |
| Transistors | ~24 billion (est.) |
| Target | AI model training (cloud) |
| Positioning | Competitive with NVIDIA A100 / AMD MI100 (at launch 2021) |

### Key Specifications

| Metric | TianGai-100 |
|--------|-------------|
| FP32 | ~40 TFLOPS (est.) |
| FP16/BF16 | ~80–100 TFLOPS (est.) |
| INT8 | ~160–200 TOPS (est.) |
| Memory | 32 GB HBM2 |
| Memory BW | ~1.2 TB/s (est.) |
| PCIe interface | PCIe Gen4 x16 |
| TDP | ~350W (est.) |
| TPP performance density | 2,352 (company metric) |

---

## TianGai-150 (BI-V150) — Current Training Flagship

### Chip Identity

| Attribute | Value |
|-----------|-------|
| Product code | BI-V150 |
| Architecture | Big Island Gen2 |
| Process | TSMC 7nm |
| Packaging | 2.5D CoWoS |
| Purpose | AI training |
| Host interface | PCIe 4.0 x16 FHFL |

### Key Specifications

| Metric | TianGai-150 (BI-V150) |
|--------|-----------------------|
| Memory | 64 GB HBM2 |
| Memory BW | ~1.2–1.6 TB/s (HBM2, est.) |
| FP16 | ~120–150 TFLOPS (est.) |
| INT8 | ~240–300 TOPS (est.) |
| TPP performance density | 3,040 (company metric) |
| TDP | ~400W (est.) |
| Form factor | PCIe 4.0 x16 FHFL |

The BI-V150 has been deployed at the Beijing Academy of Artificial Intelligence (BAAI / 智源研究院) in a **128-node BI-V150 + 120-node BI-V100 mixed heterogeneous cluster** for training Aquila2-70B, achieving 85.3% of theoretical peak utilization on heterogeneous chips.

---

## Zhikai (MR-V100) — Inference Series

| Attribute | Value |
|-----------|-------|
| Product code | MR-V100 |
| Purpose | AI inference (production deployment) |
| Process | Estimated ~16nm or 7nm (undisclosed) |
| Memory | 32 GB GDDR6 (est.) |
| Target | Cloud inference, financial/healthcare/logistics |
| Distinction | China's first purpose-built GPGPU inference chip |

The Zhikai series adds enhanced integer compute units and optimized high-bandwidth data paths specifically for inference throughput (batch diversity, low latency response).

---

## Architecture Characteristics

### Compute Model

Iluvatar CoreX GPUs are SIMT (Single Instruction Multiple Thread) architectures, following the general-purpose GPU paradigm analogous to NVIDIA's CUDA GPU architecture. Key characteristics:

```
TianGai GPU (conceptual hierarchy)
├── Multiple Compute Units (shader clusters, undisclosed exact count)
│   └── Each CU contains:
│       ├── SIMT execution pipelines (warp/wavefront execution)
│       ├── Tensor computation unit (matrix acceleration)
│       ├── FP32 / FP16 / INT8 ALUs
│       └── Local shared memory / L1 scratchpad
├── On-chip L2 cache (shared, hardware-managed)
├── HBM2 memory controller (multi-stack)
└── Multi-GPU interconnect fabric (CLIF / cluster interconnect)
```

Key company claims about their GPU architecture:
- Achieves **>90% effective compute utilization** via architectural features that reduce redundant memory access and dynamically allocate workloads to ease resource contention
- Full-stack in-house development: instruction set, chip architecture, and foundational software
- Diverges from conventional GPU paradigms through integrated hardware-software optimization
- Proprietary internal IP: computing units, cache architectures, data transmission components

### Data Types Supported

| Format | TianGai-100 | TianGai-150 | Tianshu+ |
|--------|------------|------------|---------|
| FP32 | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes |
| BF16 | Yes | Yes | Yes |
| INT8 | Yes | Yes | Yes |
| INT4 | Partial | Partial | Yes |
| FP8 | No | No | Next-gen |

---

## Memory Hierarchy

```
Per-CU: Local shared memory / L1 scratchpad (capacity undisclosed)
         ↓
On-chip: L2 cache (hardware-managed; capacity undisclosed)
         ↓
Off-chip: HBM2 (training series) / GDDR6 (inference series)
  TianGai-100 (BI-V100): 32 GB HBM2, ~1.2 TB/s
  TianGai-150 (BI-V150): 64 GB HBM2, ~1.2–1.6 TB/s
  Zhikai (MR-V100): 32 GB GDDR6, ~512 GB/s (est.)
```

---

## Packaging and Interconnect

| Layer | Detail |
|-------|--------|
| Package | 2.5D CoWoS (TianGai series) |
| Host | PCIe Gen4 x16 FHFL |
| Scale-up | CLIF (Cluster Interconnect Fabric) — proprietary |
| CLIF bandwidth | Not publicly disclosed |
| Max per server | 8 cards (standard 4U AI server) |
| Scale-out | Standard Ethernet / InfiniBand via host NIC |

The **CLIF (Cluster Interconnect Fabric)** is Iluvatar CoreX's NVLink analog — a proprietary GPU-to-GPU direct interconnect for intra-node scaling. Details on bandwidth and topology have not been publicly disclosed.

---

## Architecture Roadmap

Iluvatar CoreX's four-generation roadmap uses names from the Big Dipper (北斗七星) constellation:

| Generation | Name | Target | Year | vs NVIDIA |
|-----------|------|--------|------|-----------|
| Current | Tianshu (天枢) | Cloud AI training + inference | 2025 | Claims to surpass Hopper (H100/H200) |
| Next | Tianxuan (天璇) | High-throughput training | 2026 | Targets Blackwell (B200) |
| 3rd | Tianji (天玑) | Training + inference | 2026 | Claims to surpass Blackwell |
| 4th | Tianquan (天权) | Ultra-high perf | 2027 | Targets Rubin (R100) |
| 5th+ | TBD | "Breakthrough redesign" | 2028+ | Beyond Rubin |

---

## Export Control Context

Unlike Biren (blocked at TSMC) or Cambricon (restricted to older nodes), Iluvatar CoreX has maintained access to TSMC 7nm for its Big Island product line. The 7nm node for non-GPU products is less restricted than advanced nodes for GPUs. The company's 2.5D CoWoS packaging adds complexity to supply chain risk.

The company's HKEX IPO in January 2026 suggests confidence in continued manufacturing access through TSMC or other foundries.

---

## Comparative Performance

| Metric | TianGai-100 | TianGai-150 | NVIDIA A100 | NVIDIA H100 |
|--------|------------|------------|-------------|-------------|
| Process | TSMC 7nm | TSMC 7nm | TSMC 7nm | TSMC 4N |
| FP16 TFLOPS | ~80–100 | ~120–150 | 312 | 989 |
| Memory | 32 GB HBM2 | 64 GB HBM2 | 80 GB HBM2e | 80 GB HBM3 |
| Memory BW | ~1.2 TB/s | ~1.2–1.6 TB/s | 2.0 TB/s | 3.35 TB/s |
| TDP | ~350W | ~400W | 400W | 700W |
| Packaging | CoWoS | CoWoS | CoWoS | CoWoS |

Note: TianGai performance figures are estimates based on company TPP density claims; no independently verified benchmark data has been publicly released.

---

## Sources

- [Iluvatar CoreX Wikipedia](https://en.wikipedia.org/wiki/Iluvatar_CoreX)
- [Iluvatar CoreX official product page (TianGai-100)](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100)
- [Iluvatar CoreX official product page (BI-V150)](http://www.cloudhin.com/xk/showproduct.php?id=275)
- [Tom's Hardware — Iluvatar GPU roadmap to surpass NVIDIA Rubin](https://www.tomshardware.com/pc-components/gpus/chinas-iluvatar-corex-unveils-four-generation-gpu-roadmap-aimed-at-surpassing-nvidia-rubin)
- [TrendForce — Iluvatar 2026-28 GPU roadmap targeting H200, B200](https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/)
- [TrendForce — Iluvatar bold GPU roadmap eyeing Rubin 2027](https://www.trendforce.com/news/2026/01/28/news-chinas-illuvatar-corex-unveils-bold-gpu-roadmap-reportedly-eyeing-nvidias-rubin-by-2027/)
- [Caixin — Iluvatar HKEX IPO USD 475M](https://www.caixinglobal.com/2025-12-30/chinese-gpu-maker-iluvatar-corex-seeks-475-million-in-hong-kong-listing-102398766.html)
- [BAAI / 智源研究院 — Aquila2-70B mixed cluster training with BI-V150](https://www.elecfans.com/d/2328248.html)
- [CSDN — TianGai-150 specifications](https://blog.csdn.net/2402_84466582/article/details/139412485)
- [Leiphone — Iluvatar architecture overview](https://m.leiphone.com/category/chips/fD16hBivUTdFgdUP.html)
- [Startupnews — 92% revenue jump 2025](https://startupnews.fyi/2026/04/02/chinese-gpu-designer-iluvatar-corex-reports-92-jump-in-annual-revenue/)

---

# Investigation Update — TianGai 300 (天垓300) and the Tianshu Supernode

*investigated: 2026-08-08*
*scope: WAIC 2026 launch (2026-07-19); adjacent corrections to the pre-existing baseline*
*verification: adversarial verification run 2026-08-08 — verdict PARTIALLY_CONFIRMED (not refuted). Corrected claim is the authoritative text; several raw-scan errors were removed (see "Errors caught in verification" below).*

## What happened

Iluvatar CoreX announced **TianGai 300 (天垓300)**, described as its new-generation general-purpose GPU flagship, on **July 19, 2026** — day 3 of **WAIC 2026** (World AI Conference, Shanghai, **July 17–20, 2026**). The frequently repeated "July 18" launch date is **not supported** by any retrieved source; every report dates the launch to July 19.

The product now appears in the vendor's own product navigation under the 天垓 (TianGai) training series alongside 天垓150 and 天垓100, at product node `cpjs-yj-xlxl-tg300`. **The vendor publishes no specification sheet for it**, and the Iluvatar website news feed has not been updated since 2021 — there is no vendor press release for this launch. The product-navigation listing is therefore the strongest primary artifact; all narrative detail comes from Chinese press coverage of the WAIC session.

## Evidence-quality caveat

The many Chinese outlets covering the launch (Tencent News ×2, IT之家, Tencent Cloud developer news, 半导体行业观察, 经济参考报, 观察者网, AAStocks on 09903.HK) appear to derive from a **common company briefing**. Source count is high; true independence is low. Nothing in the coverage is independently measured.

## Confirmed architectural facts

| Fact | Detail |
|---|---|
| Architectural status | First product on a **new-generation self-developed architecture** — a generation change, not a Big Island (7nm TianGai-100/150) derivative |
| Execution model | **SIMT retained**, with scalar, vector and tensor execution paths |
| New precisions | **FP8 and FP4** — the first Iluvatar part for which either is claimed |
| Instruction classes | **MMA / DPX / FMA** |
| ISA extension: ixSMEX | Raises data reuse to cut redundant memory traffic |
| ISA extension: ixDPX | Compresses a six-instruction dynamic-programming sequence into one instruction (nearest analogue: NVIDIA Hopper DPX) |
| ISA extension: ixTrans | Lossless matrix transpose; reduces bank conflicts and VRAM overhead |
| Scale-up system | **天数超节点 ("Tianshu supernode")** — claimed **144-chip high-speed full interconnect** |
| Workload optimizations named | Attention/FFN (AF) separation; prefill/decode (PD) separation |

The three ISA extensions are the **only** concrete microarchitectural detail disclosed. Their encodings, issue rates, and interaction with the tensor path are not published.

## The 144-chip supernode — first stated scale-up domain

This is arguably the more survey-relevant disclosure: it is the **first time Iluvatar CoreX has stated a scale-up domain size at all**. The company's prior public ceiling was 8 GPUs in a single server over the CLIF fabric. Everything beyond the number 144 is undisclosed: topology (full mesh vs. switched), per-link and per-chip bandwidth, switch ASIC (none named), whether it forms a single coherent memory domain, rack power/cooling, and **whether it extends the existing CLIF fabric or introduces a new one**. The supernode is explicitly at the **trial / partnership-discussion** stage — pre-production.

Because neither bandwidth nor topology is on the record, the supernode cannot be compared quantitatively to NVL72, CloudMatrix384, or any other rack-scale domain. Only the domain size, as a company claim, is citable.

## Vendor performance claims — marketing, unverified, relative-only

All first-party, all expressed as percentages against an unspecified "Hopper-architecture mainstream international solution." **Not Blackwell.**

| Claim | Vendor figure |
|---|---|
| Attention efficiency | >90% across precisions; ~10% above a Hopper solution at 64k context |
| MoE compute efficiency | >70%; ~10% higher average MoE throughput on a DeepSeek-V4-class workload |
| Time-to-first-token | ~20% lower across Qwen / GLM / Kimi / DeepSeek |
| Decode efficiency | ~10% higher |
| Inter-card communication latency | ~13% lower average |
| Energy efficiency | "above mainstream Hopper solutions" — no figure |

No absolute figure of any kind was published. **No MLPerf submission by Iluvatar CoreX exists** — a Chinese analysis reports zero Iluvatar entries in MLCommons results through 2025, and none appear in v6.0. No third-party benchmark exists.

## Status verb

Chinese coverage says only **"天垓300已具备规模化应用条件"** — "meets the conditions for large-scale deployment" — plus deep adaptation / joint-solution discussions with domestic cloud providers, server OEMs and supernode system vendors. This is a **company readiness assertion**, not evidence of tape-out, sampling, mass production, or any named deployment. **Correct status verb: announced.** TianGai-100/150 remain the shipping products.

## Explicitly not disclosed — do not populate

Process node and foundry; die size; transistor count; packaging (CoWoS or alternative); memory type (HBM2e/HBM3/HBM3e), capacity, bandwidth; peak FLOPS/TOPS at **any** dtype (FP32/TF32/BF16/FP16/FP8/FP4/INT8); scale-up link bandwidth per chip; host interface generation; TDP; form factor; card/board name; price; whether the supernode uses CLIF. **No retrieved source — vendor or press — publishes any of these.** Note in particular that the raw scan's claim that TianGai 300 "uses HBM" is **not supported**: memory type itself is undisclosed.

## Codename mapping — unresolved

**No retrieved source ties TianGai 300 to any Big-Dipper roadmap codename** (天枢 Tianshu / 天璇 Tianxuan / 天玑 Tianji / 天权 Tianquan). Two signals argue against silently assuming TianGai 300 = Tianxuan:

1. 天垓300 is named from the 天垓 **product** line (which already contains 天垓100 and 天垓150), not from the Big-Dipper **architecture** namespace.
2. The January 2026 roadmap coverage positioned the 2026 generation against **B200/Blackwell**; TianGai 300's own marketing benchmarks are against **Hopper**.

The repo's prior line "Tianxuan (天璇) | 2026 | Targets B200 | Announced" is therefore flagged **unverified** rather than deleted: the roadmap claim was really made, but no product has been tied to the name.

## Adjacent corrections to the pre-existing baseline

1. **彤央 (Tongyang) TY edge line missing entirely.** The vendor site now lists **TY1000, TY1100, TY1100-NX, TY1100-NX-PRO, TY1200** — an edge/endpoint family the survey did not record. No specifications retrieved; recorded as a known gap.
2. **Inference part naming.** The vendor lists the inference part as **智铠100 (Zhikai 100)**. "MR-V100" is the product code, not the current product name. Both are now carried.
3. **Deployment figures superseded.** The repo's "290+ customers, 52,000 GPUs as of Jun 2025" is superseded by **340+ customers across 30+ industries as of end-2025** with **>10,000 chips in online cluster operation** and a claimed **1,000+ day** continuous cluster uptime. The two metric families (cumulative units shipped vs. chips concurrently online) are **not directly comparable**; both are company-stated and both are now labeled with their as-of dates. Neither is independently audited.

## Baseline items re-confirmed as still accurate

TianGai-100 / TianGai-150 (BI-V150) specifications; the IXUCA / DeepSpark software stack; and the **January 8, 2026 HKEX IPO, ticker 09903.HK**.

## Errors caught in verification (recorded so they are not reintroduced)

- Launch date is **2026-07-19**, not "2026-07-18/19".
- FP8/FP4 is the **only** architectural spec disclosed — the raw scan implied a broader spec release.
- The raw scan asserted **HBM** memory; memory type is undisclosed.
- The raw scan omitted the **status verb** (announced) and the **144-chip supernode** entirely.
- The raw scan omitted the three named **ISA extensions**.
- The raw scan did not flag that the vendor benchmarks are vs **Hopper**, conflicting with the Blackwell-facing January 2026 roadmap.

## Not found (checked)

- No Hot Chips / ISCA / ISSCC 2026 presentation by Iluvatar CoreX. *(Note: Hot Chips 38 runs 2026-08-23–25, still in the future as of this investigation; its program is public but no content is.)*
- No H1 2026 financial disclosure post-dating the FY2025 RMB 1.03B figure.
- No MLPerf entry, past or present.

## Sources (this update)

- [Iluvatar CoreX official site](https://www.iluvatar.com/) — product navigation; 天垓300 node `cpjs-yj-xlxl-tg300`; 智铠100; 彤央 TY series. **Primary; no spec sheet.**
- [Tencent News, 2026-07-19](https://news.qq.com/rain/a/20260719A08MB800)
- [Tencent News, second piece, 2026-07-19](https://news.qq.com/rain/a/20260719A08J8N00)
- [IT之家](https://www.ithome.com/0/978/781.htm)
- [Tencent Cloud developer news](https://cloud.tencent.com/developer/news/4280275)
- [半导体行业观察 (semi-insights)](https://www.semi-insights.com/s/bdt/15/50538.shtml)
- [经济参考报, 2026-07-20](https://www.jjckb.cn/20260720/3cb0e7d7378745b58c576c51ee79b65a/c.html)
- [观察者网, 2026-07-18](https://www.guancha.cn/GongSi/2026_07_18_824207.shtml)
- AAStocks — 09903.HK company news (https://www.aastocks.com)
- [TrendForce, 2026-01-12 — prior roadmap context](https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/)
- WAIC 2026 official dates (July 17–20, 2026, Shanghai) — WAIC official site; Shanghai municipal government English release

---

## Investigation Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*No new hardware/silicon disclosure this window — TianGai 300 remains "announced 2026-07-19, no tape-out/sampling/mass-production disclosure." What changed is corporate/financial, most of it a July 2026 event that predates the window but was absent from the 2026-08-08 pass, plus routine August/September market coverage.*

### Corporate — H-share placement (predates window, newly recorded)

- **~2026-07-08**, Iluvatar CoreX completed a placement reported at **~USD 850 million**. Sources disagree on mechanics: TheNextWeb (2026-07-08) describes it as a "share sale," while Chinese financial-press aggregation (via Bing News, accessed 2026-09-13) describes it as a placement of **14.857 million new H shares** — language consistent with a primary/follow-on issuance rather than a pure secondary sale by existing holders. **Not resolved between sources; do not assert primary vs. secondary with confidence.** This follows the original ~USD 475M HKEX IPO (2026-01-08) six months prior.
- Stated use of proceeds (TheNextWeb, press paraphrase, not a vendor quote): R&D, fab/production capacity, and inventory to fill "orders for tens of thousands of chips at once."

### Corporate — H1 2026 interim results (predates window, newly recorded)

- **H1 2026 (six months to 2026-06-30) revenue ≈ RMB 9.45–9.46 billion, +191.6% YoY** (Chinese financial press aggregation via Bing News, accessed 2026-09-13; the primary HKEXnews interim report was not independently opened in this pass — **press-reported figure, not verified against the filing**).
- Inference-chip segment (智铠/Zhikai series) revenue reported at **RMB 6.54 billion, +651.8% YoY** — the primary growth driver.
- **Adjusted net profit reported positive for the first time** ("经调整净利扭亏为盈") per aggregated press; one summary source qualifies this with "core business losses" continuing — the two characterizations are not fully reconciled here and should be checked against the primary interim report before being treated as a clean profitability milestone.
- Reported inclusion in the **Hang Seng TECH Index** as the index's first GPU-component constituent; effective date not confirmed in this pass.
- Analyst coverage: Macquarie reported maintaining an "outperform" rating with a HK$1,060 target (via aggregated press; original research note not opened).

### Corporate — ByteDance chip talks (predates window, newly recorded)

- Multiple English-language outlets (U.S. News/Reuters wire, Yahoo Finance, TheNextWeb — all dated **~2026-06-14**) reported ByteDance **in talks** with Iluvatar CoreX to purchase AI inference chips, alongside a parallel evaluation of Baidu Kunlunxin chips. **Reported as in-progress negotiations, not a confirmed order** — no volume or contract figure is attributed to Iluvatar CoreX specifically in these reports (contrast the Kunlunxin research thread, where a 2026-09-09 Zhihu-sourced item states ByteDance has "no current collaboration plans" with Kunlunxin specifically — the two chip vendors' ByteDance relationships should not be conflated).

### Market context (in-window, routine)

- August 2026 sector-wide semiconductor/AI pullback affected 09903.HK share price alongside peers (aggregated press).

### Searched and absent (2026-08-08 → 2026-09-13)

- No TianGai 300 tape-out, sampling, or mass-production disclosure.
- No 天数超节点 (Tianshu supernode) topology/bandwidth disclosure beyond the 2026-07-19 "144-chip, trial stage" statement.
- No MLPerf submission; no Hot Chips 38 (2026-08-23 → 08-25) Iluvatar CoreX talk.
- No confirmed (vs. in-talks) ByteDance order.

### Sources added 2026-09-13

- [TheNextWeb — Iluvatar CoreX seeks to raise $850m (2026-07-08)](https://thenextweb.com/news/iluvatar-corex-850-million-share-sale)
- [U.S. News/Reuters wire — ByteDance in talks with Iluvatar CoreX to purchase AI chips (2026-06-14)](https://money.usnews.com/investing/news/articles/2026-06-14/exclusive-bytedance-in-talks-with-chinas-iluvatar-corex-to-purchase-ai-chips-sources-say)
- [AASTOCKS — ILUVATAR COREX (09903.HK) launches TianGai 300 flagship (secondary coverage)](https://www.aastocks.com/en/stocks/news/aafn-news/NOW.1534101/2)
- Bing News aggregation — 09903.HK H1 2026 interim results (revenue, Hang Seng TECH inclusion, H-share placement); accessed 2026-09-13; primary HKEXnews interim filing not independently opened (query: `https://www.bing.com/news/search?q=09903.HK+%E5%8D%8A%E5%B9%B4%E6%8A%A5`) — **press-reported aggregate, not a primary filing**
