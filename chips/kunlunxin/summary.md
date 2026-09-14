# Kunlunxin XPU — Chip Summary

*as_of: 2026-09-13*
*research_baseline: 2026-04-05*
*chip: kunlunxin*
*device_class: AI Accelerator (百度昆仑芯)*

## One-Line Description

Baidu-origin cluster-based SIMD AI accelerator (Gen 3 P800: 345 TFLOPS FP16, 96 GB HBM3, 7nm), with a PaddlePaddle-native software stack (XRE/XTDK/XDNN/XTCL); spun off as independent company Kunlunxin (昆仑芯) in 2021; confidential HKEX Form A1 listing application filed 2026-01-01 (announced by Baidu 2026-01-02), not yet listed as of 2026-08-08. The differentiating direction since 2025 is **scale-up**: a cabinet supernode line (32/64-card → Tianchi 256 → Tianchi 512) built on Baidu's self-developed XPU-Link fabric — see the 2026-08-08 update section below.

## Key Specs (P800, Gen 3)

| Metric | Value |
|--------|-------|
| Process | 7 nm |
| FP16 TFLOPS | 345 |
| INT8 TOPS | ~690 (estimated) |
| Memory | HBM3 96 GB |
| Memory BW | ~1.6 TB/s |
| Chip-to-chip BW | 200 GB/s (XLINK) |
| Cards per server | 8 |
| TDP | ~350 W |
| LLM support | DeepSeek V3/R1 671B (8-card, single server) |

## Software Stack (one line each)

- **XRE**: CUDA Runtime equivalent — streams, events, xpu_malloc, multi-device
- **XTDK**: CUDA C++ equivalent — LLVM-based C/C++ kernel SDK with SIMD-aware prefix-keyword programming model
- **XDNN**: cuDNN+cuBLAS equivalent — BLAS, conv, fused attention, norm, activation library
- **XTCL**: TVM-based AOT/JIT graph compiler — ONNX + PaddlePaddle IR input, op fusion, XPU-aware tiling
- **PaddlePaddle**: Primary framework — 51+ models, DataParallel, DDP, AMP, XPU plugin for custom ops
- **vLLM-Kunlun**: Out-of-tree **hardware plugin** for stock vLLM (NOT a fork) — registered as a vLLM platform plugin via Python entry points; PagedAttention + continuous batching on XPU

## Competitive Context

| Chip | FP16 TFLOPS | Memory | Process |
|------|-------------|--------|---------|
| Kunlunxin P800 | 345 | 96 GB HBM3 | 7nm |
| Huawei Ascend 910B | 256 | 64 GB HBM2 | ~14nm |
| Nvidia A100 | 312 | 80 GB HBM2e | 7nm |
| Nvidia H20 (China) | 148 | 96 GB HBM3 | 4nm |

## Deployment Scale (2025)

- Baidu: 30,000-P800 cluster for LLM training/inference
- China Mobile: ~10B RMB P800 procurement (August 2025)
- vLLM-Kunlun: production LLM serving on P800

## Roadmap

*(status verbs as of 2026-09-13 — see the update sections below for sourcing and caveats)*

- **M100** (Gen 4, inference) announced 2025-11-13 at Baidu World 2025; first **physical** public appearance at WAIC 2026 (2026-07-17 → 07-20) with **no published specification of any kind**; no spec disclosure in the 2026-08-08 → 2026-09-13 window either
- **M300** (Gen 4, training + multimodal) announced 2025-11-13, targeted early 2027 — no new detail
- **Scale-up ladder**: 8 cards/server (conventional) → **32/64-card cabinet supernode** (launched April 2025, vendor says volume-delivered) → **Tianchi 256** (announced on sale June 2026) → **Tianchi 512** (H2 2026 target, not shipped) — no new shipment/customer/volume disclosure in-window
- **Hong Kong IPO**: confidential HKEX Form A1 submitted 2026-01-01, spin-off announced 2026-01-02; **press-reported** ~USD 50B valuation target and Q3 2026 listing (Reuters 2026-06-28, citing The Information) — not a filed figure; no HKEX hearing/acceptance update found as of 2026-09-13
- **STAR Market ("A+H" dual listing) — new 2026-09-13, fact predates window**: Kunlunxin completed **STAR Market (科创板) IPO tutoring registration (辅导备案)** with the Beijing Securities Regulatory Bureau on **2026-05-07** (agreement dated 2026-04-29), with **CICC** as tutoring institution — confirming Kunlunxin is pursuing simultaneous A-share + H-share listings rather than HKEX alone. **Press-reported** (aggregated search snippets; primary regulator bulletin not independently opened) — treat date, sponsor, and the cited 57.67%-Baidu-stake figure as unverified against a primary filing.

---

## Kunlunxin Update — Supernodes, M100 Physical Debut, IPO Valuation Leak (2026-08-08)

*Updated 2026-08-08. Window covered: 2026-04-01 → 2026-08-08 against the 2026-04-05 research baseline. Sourcing caveat: kunlunxin.com's newsroom returns HTTP 403 to automated retrieval, so **no vendor-primary spec sheet was obtainable**. Every supernode performance number below traces to Chinese-media transcription of Baidu executive remarks (Baidu Create 2026, 2026-05-13, Shanghai; WAIC 2026 coverage, 2026-07-17 → 07-22) and is a **vendor claim, not an independently benchmarked result**.*

### The materially new part: the scale-up domain

The repo previously recorded Kunlunxin scale-up as "8 cards/server" plus two planned Tianchi fabrics. The correct picture is a four-tier ladder, one tier of which (32/64-card) **predates the repo baseline and was simply missed**:

| Tier | Cards | Status as of 2026-08-08 | Notes |
|---|---|---|---|
| Conventional server | 8 | Shipping | XLINK 200 GB/s chip-to-chip; P800 |
| Kunlun Super Node (cabinet) | 32 / 64 | **Launched April 2025**; vendor calls it "国内率先实现量产交付的超节点产品之一" (among the first domestic supernodes to reach volume delivery) | Claimed **8×** inter-card communication bandwidth vs the conventional 8-card server; claimed **13×** single-card inference efficiency; claimed **5–10×** MoE training performance (a *range* — not a 10× point value); cabinet architecture claimed extensible to 512 cards. Re-displayed at WAIC 2026. |
| Tianchi 256 (天池256卡超节点) | 256 | "Lit up" (点亮) April 2026; stated on sale (正式上市) June 2026 | See status caveat below |
| Tianchi 512 | 512 | **Not shipped**; H2 2026 target | Announced 2025-11-13; claimed able to train trillion-parameter models within a single supernode |

**Tianchi 256 status — read carefully.** At Baidu Create 2026 (2026-05-13) Baidu EVP and Baidu AI Cloud president Shen Dou (沈抖) said the unit had been *lit up / powered on* in April 2026 and would *go on sale* in June 2026. "Powered on" is a weaker milestone than "completed testing", and the June date was a forward-looking statement. Chinese secondary sources state it launched in June, but a Tencent News WAIC wrap-up dated 2026-07-22 still describes both Tianchi 256 and 512 as "即将全面上市" (about to become fully available). **Treat volume shipping as NOT independently confirmed** — no customer, deployment, or unit volume was found.

**Tianchi 256 performance claims, split by date** (the two sets are frequently conflated):

| Claim | First stated | Note |
|---|---|---|
| +25% throughput vs prior generation | Baidu Create 2026, **2026-05-13** | New in-window; vendor claim |
| +50% inference efficiency | Baidu Create 2026, **2026-05-13** | New in-window; vendor messaging is internally inconsistent — WAIC-period coverage (2026-07-22) instead says "整体性能提升50%" (overall performance +50%) |
| 4× total inter-card interconnect bandwidth | Baidu World 2025, **2025-11-13** | **Pre-baseline** — not an April/June 2026 result |
| 3.5× single-card token throughput on mainstream LLM inference | Baidu World 2025, **2025-11-13** | **Pre-baseline** |

Adapted for Wenxin (文心), DeepSeek, GLM and MiniMax models (vendor claim, Create 2026).

**New fabric detail disclosed at WAIC 2026** (vendor claims, 2026-07-22): the Tianchi supernode fabric uses Baidu's **self-developed XPU-Link interconnect protocol with programmable switching**, self-designed copper cabling and a liquid-cooling CDU, with a claimed **77% measured bandwidth efficiency**. Baidu also stated a roadmap *intent* for **thousand-card and 4,000-card supernodes** built on the Kunlun M-series, with **no timeline given**. Separately Baidu claims **97% effective training efficiency** training Wenxin 5.1 on P800 ten-thousand-card clusters (vendor claim, Create 2026).

The specific figure "Tianchi 512 has ~2× the total inter-card bandwidth of Tianchi 256" circulates in aggregators but **could not be confirmed in any retrieved source and is deliberately not recorded**.

### M100 (Gen 4) — first physical showing, still zero specs

- Announced 2025-11-13 at Baidu World 2025 (pre-baseline). Its **first physical public appearance** was at WAIC 2026, Shanghai, **2026-07-17 → 07-20** — confirmed by multiple independent outlets (芯智讯 via Tencent News 2026-07-20; 10jqka 2026-07-18; Xueqiu).
- Show-floor reporting (芯智讯) states explicitly that **no official introduction materials were released** and that the associated carrier boards were "正在制作中" (still being made).
- Booth staff positioned M100 against **NVIDIA H20 on inference cost-effectiveness**, with **HBM capacity somewhat lower than H20**. This is unattributed booth-staff commentary, **low confidence**, and is the only HBM-related statement that exists for M100.
- Vendor framing: built on a "fully domestic supply chain", continuing the self-developed XPU architecture, inference-optimised.
- **No process node, HBM capacity, memory bandwidth, TFLOPS/TOPS or TDP has been published.** No spec field for M100 is populated anywhere in this repo.
- The claim that M100 "entered a commercial ramp (商业放量期) in January 2026" rests on Baidu Baike (user-editable) and secondary Chinese media, and sits in direct tension with the July 2026 show-floor finding that carrier boards were still in development. Recorded at **low confidence**: *reportedly launched early 2026; the January 2026 commercial-ramp claim is unverified and hard to reconcile with WAIC show-floor reporting.*
- **M300** (training + multimodal, targeted early 2027): unchanged, no new detail in the window.

### Hong Kong IPO

- Already documented at baseline, now with precise dates: joint sponsors submitted a **Form A1 listing application to HKEX on a confidential basis on 2026-01-01**; Baidu announced the proposed spin-off and separate listing on **2026-01-02** (HKEXnews doc 2026010200013; Baidu IR).
- **New in-window but press-reported only, not in any filing:** Reuters (2026-06-28, citing The Information) reported a target valuation of **~USD 50 billion** with listing expected around **Q3 2026**; CNBC picked this up 2026-06-29 (Baidu HK shares +6%). Associated figures — 2025 revenue ~RMB 3.5bn / ~USD 500M at breakeven, Baidu retaining ~58% — are likewise press-reported.
- Confidence **low-to-medium**: at least one report notes the target rose from ~USD 14.7bn cited roughly two weeks earlier, making this a fast-moving leak rather than a stable figure. Label it a "press-reported target" wherever cited. **The listing had not occurred as of 2026-08-08.**

### Software stack correction (repo error, not new news)

This repo described `baidu/vLLM-Kunlun` as an "open-source vLLM fork". **That is wrong and has been corrected throughout.** Both the GitHub repo and vllm-kunlun.readthedocs.io describe it as "a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU", registered "as a standard vLLM platform plugin via Python entry points, no need to modify vLLM source code", following the vLLM "RFC: Hardware Pluggable" interface (vllm issue #11162). Users install stock vLLM plus the plugin. This was already true at the 2026-04-05 baseline — it is a documentation error, not a change in the window.

- **Verified release history** (GitHub API): v0.10.1.1 (2025-12-24), v0.11.0rc1 (2025-12-26), **v0.11.0 stable (2026-03-13)** — 154 commits from 26 contributors. A "dev branch v0.25.1-dev" figure circulates and is **unreliable** (the docs site reports v0.15.1-dev); no dev-branch version is cited here.
- **Prerequisites list Kunlun3 P800 only** (Ubuntu 20.04, Python ≥3.10, PyTorch ≥2.5.1 via KL3-customised xpytorch). **No M100 support in-tree** — confirmed.
- **20+ models supported**: Qwen 2/2.5/3/3.5, Llama, DeepSeek (incl. DeepSeek-V3.2), GLM, Gemma4, InternLM2, Kimi-K2, and multimodal Qwen-VL / InternVL / InternS1.
- **Companion tooling** documented in the vLLM-Kunlun developer guide (readthedocs), though not present in the repo's top-level tree — they are ecosystem tools the docs reference, not artifacts the repo ships: **torch_xray** (module-level operator precision dump; automatic GPU-vs-P800 layer-by-layer comparison; auto-generates operator unit tests) and **xpu_profiler** (nsys-like profiler producing operator call timelines).

### Searched and absent (record as absent, not as negative evidence)

- No MLPerf Training v5.x or Inference v6.x submission by Kunlunxin or Baidu XPU.
- No Hot Chips 2026, ISCA 2026 or ISSCC 2026 paper or talk on Kunlunxin. (Hot Chips 38 runs 2026-08-23 → 08-25; no Kunlunxin talk is on the program.)
- No new public XRE / XTDK / XDNN / XTCL release notes in the window.
- No published M100 specification of any kind.

### New sources (2026-08-08)

- https://news.qq.com/rain/a/20260513A0688800 — Baidu Create 2026 (2026-05-13), Shen Dou remarks on Tianchi 256
- https://news.qq.com/rain/a/20260722A04PTE00 — Tencent News WAIC 2026 wrap-up: XPU-Link, 77% bandwidth efficiency, "即将全面上市"
- https://news.qq.com/rain/a/20260720A03JPT00 — 芯智讯 WAIC show-floor report on M100 (no official materials; carrier boards in development)
- https://stock.10jqka.com.cn/20260717/c678258367.shtml — M100 physical debut at WAIC 2026
- https://finance.sina.com.cn/tech/2025-11-13/doc-infxfnye5573223.shtml — Baidu World 2025 original 4× / 3.5× Tianchi 256 figures (dating anchor)
- https://finance.sina.cn/tech/2026-07-22/detail-iniistty7951726.d.html — WAIC 2026 supernode coverage
- https://www.reuters.com/world/asia-pacific/baidus-ai-chip-unit-kunlunxin-targets-50-billion-hong-kong-ipo-information-2026-06-28/ — ~USD 50B IPO valuation target (press-reported)
- https://www.cnbc.com/2026/06/29/baidu-kunlunxin-hong-kong-ipo.html — CNBC follow-up, 2026-06-29
- https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/2026010200013.pdf — Baidu spin-off / Form A1 announcement (HKEXnews)
- https://github.com/baidu/vLLM-Kunlun — plugin (not fork) confirmation + release history
- https://vllm-kunlun.readthedocs.io/en/latest/ — vLLM-Kunlun docs, prerequisites, model list, developer guide (torch_xray, xpu_profiler)
- https://www.trendforce.com/news/2026/07/20/news-waic-2026-highlights-chinas-supernode-push-led-by-huawei-minimax-m3-unitree-robots-in-focus/ — WAIC 2026 supernode context

---

## Update (2026-09-13)

*Scan window 2026-08-08 → 2026-09-13. Classification: Roadmap + Minor (no hardware/silicon change).*

- **Roadmap/corporate**: Kunlunxin filed **STAR Market (科创板) IPO tutoring registration** on 2026-05-07 (CICC sponsor) — confirms "A+H" dual-listing intent alongside the confidential HKEX Form A1. Press-reported; predates this scan window but was not previously recorded here. See Roadmap section above.
- **Minor**: 2026-09-08, Kunlunxin + FlagOS announced same-day adaptation of the MiniCPM5-2B model (Mianbi AI/OpenBMB) — an ecosystem/model-support item, not a new SDK release.
- Searched and absent: no M100/M300 spec disclosure, no Tianchi shipment/customer update, no HKEX hearing update, no new XRE/XTDK/XDNN/XTCL release, no Hot Chips 38 Kunlunxin talk.

New sources: see `research/kunlunxin/search-results.md` → "Resources Added 2026-09-13" and `research/kunlunxin/investigations/hw-architecture.md` → "Investigation Update — 2026-09-13".
