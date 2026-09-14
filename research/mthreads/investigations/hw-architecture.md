# Moore Threads MTT GPU Hardware Architecture Investigation

*as_of: 2026-09-13 (baseline 2026-04-05; dated update sections appended at end)*
*chip: mthreads*
*device_class: GPU (China, 摩尔线程)*
*resource: hw-architecture*

---

## Summary

> ⚠️ **Read the 2026-08-08 update section at the end of this file first.** The baseline sections below omit the Gen 4 PingHu (PH100 / MTT S5000) generation entirely and mislabel Huashan / Huagang as "Gen 3" (it is Gen 5).

Moore Threads (摩尔线程) designs full-function SIMT GPUs targeting AI training, inference, graphics rendering, and video encode/decode on a single chip. The MUSA architecture (Moore Threads Unified System Architecture) is their proprietary compute platform. The current production flagship is the **MTT S4000** (Chunxiao die, TSMC 12nm, 48 GB GDDR6). The next-generation **Huashan** AI GPU (Flower Harbor / Huagang architecture, dual chiplet, HBM, targeting Hopper-class performance) was announced December 2025 with 2026 launch expected.

---

## Die Generations

| Generation | Die Name | Architecture | Process | Products |
|-----------|---------|-------------|---------|---------|
| Gen 1 | Sudijia (苏堤甲) | MUSA v1 | TSMC 12nm | MTT S60, MTT S2000 |
| Gen 2 | Chunxiao (春晓) | MUSA v2/v3 | TSMC 12nm | MTT S80, MTT S3000, MTT S4000 |
| Gen 3 (announced) | Huashan (华山) | Flower Harbor (华港) | Advanced node (undisclosed) | Huashan AI GPU (2026) |

---

## MTT S4000 — Current AI Flagship

### Chip Identity

| Attribute | Value |
|-----------|-------|
| Die | Chunxiao (春晓), full chip |
| Architecture | MUSA Chunxiao (3rd-gen MUSA) |
| Process | TSMC 12nm |
| Transistors | 22 billion |
| TDP | 450W |
| Form factor | Dual-slot FHFL PCIe card |
| Host interface | PCIe Gen 5 x16 |
| L2 Cache | 4 MB (hardware-managed) |

### Compute Engine

```
MTT S4000 (Chunxiao die)
├── 128 Tensor Cores (TCE — Tensor Compute Engine)
├── 4,096 MUSA SP (Stream Processors / shaders)
│   └── Organized into MP (MUSA Processors)
│       Each MP contains:
│         ├── 128× FP32 units
│         ├── 32× INT8 / INT32 bitwise units
│         ├── 32× SFU (Special Function Units)
│         ├── 2× FP64 units
│         ├── TCE (Tensor Compute Engine for matrix ops)
│         └── 28 KB local shared memory
├── 256 Texture Units
├── 256 Render Output Units (ROPs)
└── L2 Cache: 4 MB
```

### Peak Performance

| Metric | MTT S4000 | MTT S3000 | MTT S80 |
|--------|-----------|-----------|---------|
| FP32 | 25 TFLOPS | 15.6 TFLOPS | 14.7 TFLOPS |
| TF32 | 50 TFLOPS | — | — |
| FP16 / BF16 | 200 TFLOPS | — | — |
| INT8 | 200 TOPS | — | — |
| Memory | 48 GB GDDR6 | 32 GB GDDR6 | 16 GB GDDR6 |
| Memory BW | 768 GB/s | ~512 GB/s | ~512 GB/s |
| GPU clock | ~1.8–1.9 GHz | ~1.8 GHz | 1.8 GHz |

---

## Memory Hierarchy

```
Per-MP: 28 KB local shared memory (scratchpad)
         ↓
On-chip: 4 MB L2 cache (hardware-managed)
         ↓
Off-chip: GDDR6
  S4000: 48 GB, 384-bit bus, 768 GB/s
  S3000: 32 GB, 256-bit bus, ~512 GB/s
  S80:   16 GB, 256-bit bus, ~512 GB/s
```

Note: 4 MB L2 is very small compared to competitors (NVIDIA A100: ~40 MB, Biren BR100: 300 MB). This is a recognized limitation of the Chunxiao generation.

---

## Host Interface and Scale-up

| Layer | Detail |
|-------|--------|
| PCIe | Gen 5 x16 |
| CXL | Not confirmed |
| Scale-up | MTLink 1.0 (NVLink analog) |
| MTLink BW | 240 GB/s aggregate (per GPU in 8-card KUAE server) |
| Max nodes/server | 8 GPUs (KUAE D800 / MCCX D800) |
| Scale-out | Up to 10,000 GPU cluster (via MTLink fabric) |
| Scale-out protocol | Standard Ethernet / RoCE via host NIC |

---

## Chunxiao vs Sudijia Architecture

| Feature | Sudijia (Gen1) | Chunxiao (Gen2/3) |
|---------|---------------|-------------------|
| SP count | 2,048 (S2000: 4,096 configured) | 4,096 full |
| Tensor Cores | None / limited | 128 TCE |
| FP16/BF16 | Basic | Native |
| INT8 | Basic | Native |
| PCIe | Gen 4 | Gen 5 |
| MTLink | No | Yes (S4000) |
| AI SDK | Early MUSA | MUSA SDK 4.0+ |

---

## Huashan / Flower Harbor (2026 roadmap) — *baseline section; the "Gen 3" label used here is WRONG, it is Gen 5 Huagang (see 2026-08-08 update)*

| Attribute | Detail |
|-----------|--------|
| Architecture | Flower Harbor (华港 / Huagang) |
| Die configuration | Dual chiplet |
| Memory | 8× HBM sites (first HBM-equipped Moore Threads GPU) |
| Target performance | Between NVIDIA Hopper and Blackwell |
| FP compute | Close to B200 (claimed) |
| Memory bandwidth | Matches B200 (claimed) |
| Memory capacity | Exceeds B200 (claimed) |
| New formats | MTFP6, MTFP4 (proprietary mixed precision) |
| FP4 | Supported natively |
| FP64 | Supported (first time in MT lineup) |
| Ray Tracing | 2nd-gen hardware RT engine; 50× vs Chunxiao |
| DirectX | DX12 Ultimate |
| Availability | 2026 (announced Dec 2025) |

---

## Sources

- [MTT S4000 Official Product Page](https://en.mthreads.com/product/S4000)
- [VideoCardz — MTT S4000 48GB MTLink](https://videocardz.com/newz/moore-threads-introduces-mtt-s4000-48gb-ai-gpu-with-mtlink-and-zero-cost-nvidia-cuda-framework-translation)
- [WCCFTech — MTT S4000 specifications](https://wccftech.com/moore-threads-mtt-s4000-gpu-48-gb-memory-200-tops-ai-gen5-ready/)
- [Tom's Hardware — Chunxiao GPU unveil](https://www.tomshardware.com/news/moore-threads-unveils-chunxiao-gpu)
- [TrendForce — MTLink scale-up](https://www.trendforce.com/news/2024/07/11/news-chinas-moore-threads-develops-mtlink-to-challenge-nvidias-nvlink/)
- [Tom's Hardware — 10K GPU cluster](https://www.tomshardware.com/pc-components/gpus/chinese-gpu-maker-moore-threads-can-now-scale-to-10000-processors-for-ai-clusters-mtlink-fabric-tech-competes-with-nvidias-nvlink)
- [WCCFTech — Huashan AI GPU Flower Harbor](https://wccftech.com/moore-threads-lushan-gaming-huashan-ai-gpus-15x-gaming-uplift-50x-rt-boost-dx12-ultimate-support/)
- [TrendForce — Huashan Hopper rival](https://www.trendforce.com/news/2025/12/22/news-chinas-moore-threads-unveils-huashan-ai-chip-reportedly-takes-aim-at-nvidias-hopper/)
- [MUSA Architecture blog](https://blog.mthreads.com/blog/musa/2024-05-11-MUSA%E7%A1%AC%E4%BB%B6%E6%9E%B6%E6%9E%84%E4%B8%8EGPU%E5%B9%B6%E8%A1%8C%E7%A5%8D%E7%9B%AE%E5%9F%BA%E7%A1%80/)
- [Beyond3D Forum — MUSA architecture deep dive](https://forum.beyond3d.com/threads/moore-threads-musa-architecture-and-mtt-video-cards.62808/)

---

# Investigation Update — 2026-08-08: MTT S5000 (PH100 / PingHu) and MTT C256

*investigated: 2026-08-08*
*supersedes: the "Gen 3 Huashan" framing in the sections above*
*method: vendor product pages (EN + CN), GitHub API for repo/release dates, Chinese-language tech media cross-check*

## 0. What this update corrects

The baseline sections above contain three errors that this update fixes and that must not be reintroduced:

1. **Generation numbering.** Huashan / Huagang ("Flower Harbor") was recorded as "Gen 3". It is **Gen 5**. The verified West Lake sequence is Sudi (1) → Chunxiao (2) → Quyuan (3) → PingHu (4) → Huagang (5).
2. **A whole generation was missing.** Gen 4 **PingHu 平湖** shipped as the **PH100** chip in the **MTT S5000**, which is now the flagship. The baseline jumps straight from Chunxiao to Huashan.
3. **Chip conflation.** 平湖/PH100 and 华山/花港 are **not** competing names for one product and there is no naming conflict to resolve. MDC 2025 (2025-12-20) announced the *fifth*-generation Huagang architecture with the Huashan (AI) and Lushan (graphics) chips; the S5000 is a separate, earlier, already-shipping fourth-generation part. Independent coverage separates them explicitly (采用平湖架构的S5000芯片实现量产 vs Huashan as 首款基于该架构的云端AI加速GPU).

The S5000 was also **not launched on 2025-12-20**. No retrieved source shows it being unveiled at MDC 2025; its detailed specifications became public around **2026-02-11/12** (GLM-5 "Day-0" adaptation dated 2026-02-12; the H100 loss-comparison write-up dated 2026-02-13).

## 1. Corrected die generation table

| Generation | Architecture | Die / chip | Process | Datacenter products | Status |
|-----------|-------------|-----------|---------|--------------------|--------|
| Gen 1 | 苏堤 Sudi (Sudijia 苏堤甲) | — | TSMC 12nm | MTT S1000, S2000 (also S10/S30/S50, S60) | Legacy |
| Gen 2 | 春晓 Chunxiao | Chunxiao | TSMC 12nm | MTT S3000, S4000 (also S70/S80) | GA 2024 |
| Gen 3 | 曲院 Quyuan | not disclosed | not disclosed | none tracked | — |
| **Gen 4** | **平湖 PingHu** | **PH100** | **not disclosed** | **MTT S5000** | **Shipping — accelerated mass production 2026** |
| Gen 5 | 花港 Huagang ("Flower Harbor") | 华山 Huashan (AI), 庐山 Lushan (graphics) | Advanced node (undisclosed) | Huashan AI GPU | Announced 2025-12-20 (MDC 2025), 2026 mass production |

## 2. MTT S5000 — evidence tiers

### (a) Vendor-published (en.mthreads.com/product/S5000, www.mthreads.com/product/S5000)

Verbatim: *"With Moore Threads' next-generation PH100 chip at its core, and built on the advanced 'PingHu' architecture, MTT S5000 delivers full-precision compute support from FP8 to FP64,"* and *"Powered by the fourth-generation MUSA full-stack platform."*

- Chip: **PH100**; architecture: **PingHu**; platform generation: **fourth-generation MUSA**
- Data types: **full-precision FP8 → FP64**
- Form factors: **OAM** compute module (liquid-cooled and air-cooled variants); **MTT MGX** 8-GPU modular platform (eight OAM modules over MTLink); **MTT SGX5000** server (8× MTT S5000)
- **The page carries NO specification table.** No capacity, bandwidth, TFLOPS, TDP or process node is vendor-published.

### (b) Media-reported only — medium confidence, likely single common origin, NOT vendor-published

Baidu Baike, EET-China, Toutiao, Zhihu/Sohu roundups, Tencent Cloud news:

- 80 GB memory (type **implied HBM**; **HBM generation not confirmed**)
- 1.6 TB/s memory bandwidth
- 784 GB/s card-to-card interconnect bandwidth
- up to 1,000 TFLOPS single-card dense FP8

Not confirmed anywhere: TDP, process node, transistor count, MP/SP count, tensor-core count, clock, L2/SRAM capacity, host PCIe generation, packaging. A "7nm / 22 billion transistors" figure circulating on low-quality sites **duplicates the S4000 Chunxiao transistor count verbatim** and is treated as suspect.

### (c) Trap to avoid

The only bandwidth numbers on the official page — **3.35 TB/s** and **4.0 TB/s** — are **competitor comparison-baseline footnotes**: "reference benchmark 1: dense FP16 989 TFLOPS, memory bandwidth 3.35 TB/s" (NVIDIA H100 SXM) and a second baseline "dense FP16 148 TFLOPS, memory bandwidth 4.0 TB/s" (H20-class). A naive scrape of that page would misattribute 3.35 TB/s to the S5000.

## 3. Deployment evidence

Chinese media consistently report the S5000 in accelerated mass production (已进入加速规模化量产阶段) with 10,000-card clusters in commercial service. A **thousand-card S5000 cluster** was used by BAAI / Zhiyuan (智源研究院) to train the **RoboBrain 2.5** embodied-intelligence model (ITHome, 2026-07-20). Treat "deployed at scale" as media-reported and vendor-corroborated, not independently audited.

## 4. Vendor marketing claims (label as such)

From the official product page: single-GPU prefill ≥ 4,000 tokens/s, decode ≥ 1,000 tokens/s; prefill 2.5× "leading international flagship products" at 16K sequence length; Llama3-70B MFU > 60%; DeepSeek-236B MFU > 40%; 95% cluster linearity at 10K-GPU scale; 0.6% relative precision deviation for DeepSeek-236B on 10K GPUs.

**Separately and distinctly**, Chinese media (Tencent Cloud news, 2026-02-13) report a thousand-card S5000 cluster whose training loss differed from an H100 cluster by **0.62%** on RoboBrain 2.5. This is a *different* claim from the vendor's 0.6% DeepSeek-236B figure and the two are easily conflated. Low-to-medium confidence, not independently reproduced. The test date is not separable from the report date in the sources retrieved.

## 5. MTT C256 supernode

First publicly demonstrated (首次公开展示) at **WAIC 2026 on 2026-07-17**, coverage 07-17 → 07-20; confirmed by ITHome, Sina Finance / Fast Technology and Global Times.

| Property | Claim | Confidence |
|----------|-------|-----------|
| Single-layer Scale-up network, industry first (首创单层Scale-up网络) | Breaks the prevailing 64-card single-layer limit | confirmed (multi-outlet) |
| Single cabinet | 128 GPUs fully interconnected (128卡全互联) | confirmed |
| Two cabinets | 256 cards (并柜扩展后可达256卡极速互联) | confirmed |
| Card-to-card latency | Sub-microsecond (亚微秒级) | confirmed |
| Node form factor | 2U, compatible with existing DC hardware/software | confirmed |
| Target scale | Building block for 10K–100K-card clusters | confirmed |
| Per-link MTLink bandwidth / generation / aggregate bandwidth | — | **not disclosed** |
| Liquid cooling, three-way blind-mate, comprehensive RAS | — | **not confirmed** in the sources retrieved (ITHome explicitly does not mention them) |
| Shipping / pricing / customer deployment | — | **not confirmed** — status is announced/demonstrated only |

If it ships as announced this is a claimed ~32× increase in scale-up domain over the current 8-GPU server, moving Moore Threads into the rack-scale supernode category. The framing must stay hedged as an announced capability.

## 6. Also at WAIC 2026

Moore Threads framed its offering as three AI "factories": a **model factory** (pretraining, post-training, RL), a **token factory** (native MoE support, million-token ultra-long-context inference), and an **agent factory** (high concurrency, claimed > 100 tokens/s per user). The slogan "Token Era, Intelligence for All Things" was **not confirmed** in the coverage retrieved and is not recorded as fact.

## 7. Open items

- No MLPerf Training or Inference submission found
- No Hot Chips 2026 or ISCA 2026 Moore Threads paper found (searched, not exhaustively). Note Hot Chips 38 runs 2026-08-23 → 25, after this investigation date
- Lushan gaming GPU 2026 status unverified
- MTLink generation number for S5000 and C256 undisclosed
- S5000 memory type / HBM generation, TDP, process node, transistor count all undisclosed by the vendor

## 8. Sources for this update

- https://en.mthreads.com/product/S5000 — primary vendor page (EN): PH100, PingHu, fourth-generation MUSA, FP8→FP64, OAM/MGX/SGX5000, all marketing claims; no spec table
- https://www.mthreads.com/product/S5000 — primary vendor page (CN): confirms 3.35 TB/s / 4.0 TB/s are competitor comparison baselines
- https://www.ithome.com/0/978/856.htm — ITHome, 2026-07-20: MTT C256 details; S5000 thousand-card BAAI RoboBrain 2.5 cluster
- https://finance.sina.com.cn/tech/roll/2026-07-19/doc-iniiiwhz3402415.shtml — Sina Finance, 2026-07-19: single-layer Scale-up, three AI factories
- https://cloud.tencent.com/developer/news/3588420 — Tencent Cloud news, 2026-02-13: media spec set + 0.62% loss comparison
- https://baike.baidu.com/item/MTT%20S5000/67553280 — Baidu Baike (HTTP 403 on direct fetch; snippet-level only)
- https://www.eet-china.com/mp/a474853.html — EET-China (fetch timed out; snippet-level only: 80 GB, 平湖 architecture)
- https://api.github.com/orgs/MooreThreads/repos?sort=created&direction=desc&per_page=100 — authoritative repo creation dates

# Investigation Update — 2026-09-13: Q1 2026 profit backfill and September stock decline (Roadmap)

*scan window: 2026-08-08 → 2026-09-13; classification: Roadmap — market/financial only, no new hardware facts*

## 1. Q1 2026 first quarterly profit (backfill, predates window)

36Kr (2026-04-30) reports Moore Threads' Q1 2026 revenue at **RMB 738 M (+155% YoY)** with **net profit RMB 29.35 M** — described as the company's first quarterly profit. Caveat in the same source: **excluding government subsidies, the adjusted result was a loss of RMB 54.28 M.** This predates the survey's 2026-08-08 baseline and was not previously recorded; treated here as a backfill, not in-window news. Source: https://www.36kr.com/p/3788937709449989

## 2. September 2026 stock decline (in-window)

On 2026-09-07 Moore Threads shares hit the daily 20% limit-down, closing at **RMB 415.49/share** (lowest since the Nov 2025 IPO), cutting market cap to **RMB 195.3 B**, below RMB 200B for the first time. Proximate trigger: a lock-up expiry releasing **25,774,500 shares (5.48% of total share capital, ≈RMB 10.7B unlocked value)**. Secondary coverage also cited earnings pressure / margin concerns and questioned whether China's "four AI-GPU dragons" are overvalued. This is a market event, not a new product or business disclosure — no chip spec, customer, or revenue figure changed.

Sources:
- Sina Finance (2026-09-07): https://finance.sina.com.cn/stock/s/2026-09-07/doc-iniqznpq6265641.shtml
- Sohu (2026-09-07): https://www.sohu.com/a/1073215563_122014422
- Tencent News (2026-09-07): https://news.qq.com/rain/a/20260907A0D33D00
- Phoenix Finance (2026-09-07/08): https://i.ifeng.com/c/8wEZGQTtoK6
- Eastmoney (2026-09-08): https://finance.eastmoney.com/a/202609083868286948.html

## 3. No hardware findings

No new MTT S5000/C256 spec, no MLPerf submission, and no Hot Chips 38 content found this window.
