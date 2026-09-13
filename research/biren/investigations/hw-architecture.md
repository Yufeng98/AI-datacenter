# Biren BR100 / BR104 Hardware Architecture

*as_of: 2026-04-05*
*chip: biren*
*device_class: GPU-like AI Accelerator (China)*
*Representative products: BR100 (dual-die OAM, 550W), BR104 (monolithic PCIe, 300W)*

---

## Overview

Biren Technology (壁仞科技), founded in Shanghai in 2019, designed China's most powerful general-purpose GPU as of its 2022 announcement. The **BR100** is a dual-die, 7nm GPGPU built on TSMC's CoWoS-S 2.5D advanced packaging. It closely resembles an NVIDIA A100 class product in architecture — GPU-style SIMT execution model, HBM2e memory, PCIe 5 + CXL host interface, and a proprietary multi-GPU interconnect called **BLink**.

Biren's internal architecture brand is **BiLi** (壁立仞 architecture). The GPU core is structured around **Streaming Processing Clusters (SPCs)**, each containing **Execution Units (EUs)**, mirroring NVIDIA's SM/CUDA-core hierarchy. Key differentiating features include **C-Warp** (Biren's warp execution model), **TDA** (data-flow access acceleration), **NME** (near-memory engine to reduce data movement), and **SVI** (stream virtualization isolation).

Due to US export controls imposed in October 2022, TSMC suspended production of advanced chips for Biren. This blocked mass production of the BR100 and forced Biren to explore alternative manufacturing paths and adapt products for lower-capability nodes. The BR104 (single-die, PCIe, 300W) was the more producible variant targeting the domestic Chinese market.

---

## 1. Compute Engine

### SPC — Streaming Processing Cluster

The **SPC (Streaming Processing Cluster)** is Biren's top-level compute grouping, analogous to NVIDIA's GPC (Graphics Processing Cluster). Each BR100 die contains **32 SPCs**.

| Attribute | Per SPC |
|-----------|---------|
| Execution Units (EUs) | 16 |
| Threads per SPC | 4,096 |
| Execution model | C-Warp (GPU SIMT variant) |

### EU — Execution Unit

The **EU (Execution Unit)** is the atom-level compute block, analogous to an NVIDIA SM. Each EU contains:
- FP32, BF16, TF32+, INT8 compute pipelines
- Tensor acceleration engines (matrix / tensor units)
- On-chip L1/shared memory (cache-backed, not scratchpad)
- Warp schedulers managing C-Warp threads

### Chip-Level Compute

| Product | Dies | SPCs (total) | EUs | Threads | Peak FP32 | Peak BF16 | Peak INT8 |
|---------|------|-------------|-----|---------|-----------|-----------|-----------|
| BR100 | 2 | 64 (32/die) | ~512+ | 128K | 256 TFLOPS | 1,024 TFLOPS | 2,048 TOPS |
| BR104 | 1 | 32 | ~256 | 64K | 128 TFLOPS | ~512 TFLOPS | ~1,024 TOPS |

*BR100 total: 77 billion transistors across two dies, each 1,074 mm² before packaging (total die area ~2,148 mm² silicon).*

### Precision Support

The BR100 supports a data-flow-oriented precision model:

| Format | Notes |
|--------|-------|
| FP32 | Full IEEE 754 |
| TF32+ | Biren's extended TF32 variant — higher dynamic range than NVIDIA TF32 |
| BF16 | Standard brain float |
| INT8 | 8-bit integer quantized inference |
| FP16 | Half-precision |

**No FP64 support.** This is a deliberate AI/HPC focus decision — unlike NVIDIA H100 (67 TFLOPS FP64) and AMD MI300X, the BR100 has zero FP64 compute, making it unsuitable for traditional scientific HPC but optimal for AI training and inference.

### TF32+ — Biren's Differentiator

Biren's **TF32+** format extends NVIDIA's TF32 with wider exponent/mantissa range. At Hot Chips 34, Biren cited TF32+ as a key throughput-per-accuracy advantage over standard FP32 for deep learning training.

---

## 2. Data Path

### C-Warp Execution Model

Biren uses **C-Warp** (Data-Flow Warp), a SIMT-style execution model. Like NVIDIA warps, C-Warps group threads that execute the same instruction simultaneously. Key characteristics:

- Each SPC contains 4,096 threads organized into C-Warps
- Warp schedulers hide memory latency by swapping in ready warps
- Hardware handles thread divergence via predication
- C-Warp is compatible with SUPA (BIRENSUPA) GPU programming model

### Data-Flow Architecture Features

Biren describes the BR100 as a **data-flow-centric architecture** with six proprietary features:

| Feature | Acronym | Description |
|---------|---------|-------------|
| TF32+ data-flow precision | TF32+ | Enhanced precision format for training |
| Data-flow access acceleration | TDA | Optimized memory access patterns for tensor streaming |
| Data-flow parallel execution | C-Warp | SIMT warp model adapted for data-flow workloads |
| Near-memory engine | NME | Reduces cross-die data movement; data processed near memory |
| Non-Uniform/Uniform Memory Access | NUMA/UMA | Memory access topology optimization |
| Stream virtualization isolation | SVI | Workload isolation and multi-tenancy on shared hardware |

### On-chip Network / Mesh

Based on Hot Chips 34 analysis, the BR100 uses a **mesh interconnect** between SPCs and L2 cache slices — similar to Intel Sapphire Rapids server CPU topology. Each SPC connects to a nearby L2 slice, and the mesh allows all-to-all communication across the chip. This enables the large 300 MB distributed shared L2 cache to be accessed efficiently by any SPC.

### Die-to-Die Interconnect

The two BR100 dies communicate via a **die-to-die link** at **896 GB/s** bidirectional bandwidth, enabled by TSMC's CoWoS-S silicon interposer. This high-bandwidth die link makes the two dies appear as a logically unified GPU to SUPA software, with NUMA-aware memory allocation for optimal locality.

---

## 3. On-chip Memory

### L2 Cache (Distributed)

| Attribute | Value |
|-----------|-------|
| Total L2 capacity | 300 MB (across entire chip) |
| Organization | Distributed cache slices on mesh |
| Backing | Hardware-managed (unlike Cambricon MLU scratchpads) |
| Scope | Unified across all SPCs on a die |

The 300 MB L2 is a defining feature of the BR100 — nearly 6× larger than NVIDIA A100's ~40 MB L2 (and larger than H100's 50 MB). This massive on-chip cache reduces HBM bandwidth pressure and enables large working sets (model weights, KV-cache) to reside on-chip.

### L1 / Shared Memory

Each EU has a per-EU L1 cache and configurable shared memory (analogous to NVIDIA SM's SMEM/L1). Specific capacity per EU is not publicly disclosed.

---

## 4. Off-chip Memory

### HBM2e

| Attribute | BR100 | BR104 |
|-----------|-------|-------|
| Memory type | HBM2e | HBM2e |
| Capacity | 64 GB | 32 GB |
| Bandwidth | 2,300 GB/s (2.3 TB/s) | 819 GB/s |
| HBM stacks | 4 (2 per die) | 2 |
| Memory bus width | 4,096-bit | 2,048-bit |

The 64 GB / 2.3 TB/s configuration is competitive with NVIDIA A100 SXM4 (80 GB HBM2e, 2.0 TB/s). The BR100's higher bandwidth (2.3 vs 2.0 TB/s) reflects higher HBM2e clock speeds.

**HBM supply constraint (post-2022):** US export controls restricted Biren's access to HBM from SK Hynix and Samsung (US-technology-containing). This was the critical bottleneck preventing mass production. Post-IPO (Jan 2026), Biren's access to advanced memory is still constrained; the company has explored Samsung HBM3 supply via non-US channels and domestic CXMT alternatives.

---

## 5. Host Interface / Package

### PCIe 5.0 + CXL

The BR100 uses **PCIe Gen 5.0 x16** as the host interface:

| Attribute | Value |
|-----------|-------|
| PCIe version | Gen 5.0 x16 |
| Bandwidth | ~128 GB/s bidir (PCIe 5.0 x16 = 64 GB/s per direction) |
| CXL support | CXL 1.1/2.0 (memory expansion / coherent access) |
| Form factor | OAM (BR100), PCIe FHFL (BR104) |

CXL support allows the BR100's 64 GB HBM2e to appear as CXL-attached memory to the CPU, enabling coherent memory sharing in heterogeneous compute nodes — a capability not present in NVIDIA A100 (which only got NVLink-C2C in H100 SXM).

### 2.5D CoWoS-S Packaging

TSMC's **CoWoS-S** (Chip on Wafer on Substrate with Silicon interposer):

| Component | Details |
|-----------|---------|
| Dies | 2 BR100 dies on a single silicon interposer |
| HBM stacks | 4 HBM2e stacks (2 per die) |
| Die-to-die bandwidth | 896 GB/s via interposer connections |
| Interposer | Passive silicon interposer |
| Total die area | ~2,148 mm² combined silicon |

This is the same packaging approach used by NVIDIA for the A100 and AMD for the MI250X.

### OAM Form Factor (BR100)

The BR100 targets cloud/hyperscale deployments using the **Open Accelerator Module (OAM)** form factor:
- Power: 550W (peak)
- 8-card per server configuration via BLink
- Compatible with OCP (Open Compute Project) infrastructure

---

## 6. Scale-up Interconnect (BLink)

**BLink** is Biren's proprietary GPU-to-GPU interconnect, analogous to NVIDIA NVLink.

| Attribute | BLink |
|-----------|-------|
| Connections per GPU | 8 bidirectional links |
| Per-link bandwidth | 64 GB/s bidirectional |
| Total GPU BLink bandwidth | 512 GB/s aggregate |
| Max GPUs per server | 8 (via BLink mesh) |
| Topology | Point-to-point mesh (no dedicated switch ASIC disclosed) |

The 8-way BLink topology enables 8 BR100 GPUs to be connected in a single server node with full peer-to-peer GPU communication — analogous to NVIDIA's 8x NVLink configuration in DGX A100. No external BLink switch (NVSwitch equivalent) has been publicly announced.

---

## 7. Scale-out Interconnect

For multi-server training, Biren relies on standard networking:

| Component | Description |
|-----------|-------------|
| InfiniBand / Ethernet | Standard NIC-based inter-node communication |
| No proprietary scale-out NIC | Unlike NVIDIA ConnectX-7 / Quantum-3 ecosystem |
| SUPA collective library (BCCL) | NCCL-analog for AllReduce/AllGather over SUPA runtime |

---

## 8. Export Control Impact

| Date | Event |
|------|-------|
| August 2022 | BR100 announced at Hot Chips 34 |
| October 2022 | US Department of Commerce adds advanced chip export controls; TSMC suspends Biren advanced node production |
| 2022–2024 | Biren adapts products; investigates non-TSMC fabs (SMIC N+2, CXMT) |
| January 2026 | Biren IPO on Hong Kong Stock Exchange; raises HK$5.58B (~$717M USD) |
| 2026 | Continued development; BR200 / next-gen roadmap; domestic fab collaboration |

The export controls specifically target chips manufactured on sub-14nm processes with certain performance thresholds. The BR100 (TSMC 7nm) was blocked from mass production. Biren's production volumes as of 2026 are a fraction of original projections.

---

## Sources

- [Hot Chips 34 BR100 Slide Deck (official)](https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf)
- [Chips and Cheese — Hot Chips 34 BR100 Deep Dive](https://chipsandcheese.com/p/hot-chips-34-birens-br100-a-machine-learning-gpu-from-china)
- [ServeTheHome — Biren BR100 GPU Review](https://www.servethehome.com/biren-br100-gpu-for-datacenter-compute-and-ai-workloads/)
- [VideoCardz — BR100 7nm 77B transistors 64GB HBM2e](https://videocardz.com/newz/chinas-biren-br100-is-7nm-hpc-gpu-with-77b-transistors-and-64gb-hbm2e-memory)
- [Tom's Hardware — Chinese Biren 77B transistors 2 PFLOPS](https://www.tomshardware.com/news/chinese-biren-rolls-out-new-gpus-with-77-billion-transistors-2-pflops-of-ai-performance)
- [WCCFTech — BR100 2.8x faster than Ampere](https://wccftech.com/birentech-china-most-powerful-gpu-biren-br100-architecture-disclosed-2-8x-faster-than-nvidia-ampere/)
- [Zhihu — 陈巍 BR100 deep dive (Chinese)](https://zhuanlan.zhihu.com/p/551888300)
- [EET-China — BR100 architecture explainer (Chinese)](https://www.eet-china.com/news/202208100913.html)
- [ZhiDX — 8 architectural features (Chinese)](https://zhidx.com/p/341643.html)
- [Bloomberg — Biren $717M Hong Kong IPO](https://www.bloomberg.com/news/articles/2026-01-02/ai-chip-designer-biren-to-debut-after-717-million-hong-kong-ipo)
- [IndraStra — TSMC suspends Biren amid US curbs](https://www.indrastra.com/2022/10/sources-tsmc-suspend-work-for-chinese.html)

---

# Investigation Update — 2026-08-08: 壁砺 (BR10X) Product Line, WAIC 2026 Fabric, and BR20X

*Investigated 2026-08-08. Supersedes parts of the 2026-04-05 report above; prior-generation content is retained deliberately.*

## Scope of this update

The 2026-04-05 report documented only the BR100/BR104 parts announced at Hot Chips 34 in August 2022. Two categories of correction apply:

1. **Baseline miss.** Biren's commercially shipping silicon (BR106 / BR110 / BR166) and its FY2025 annual results (published 2026-03-30) were already public before the 2026-04-05 baseline and were never recorded.
2. **Genuinely new since baseline.** The WAIC 2026 announcements (2026-07-17 to 07-20) — NPO optics, BLink 2.0, the three-tier supernode matrix, and BR2xx — plus BIRENSUPA GitHub activity from May to August 2026.

## 9. Naming: 壁砺 IS the BR series

An important framing correction. 壁砺™ (Bili) is the **product brand for the same BR series**, not a separate naming system:

- 壁砺106 = **BR106**
- 壁砺110 = **BR110**
- 壁砺166 = **BR166**

All are the **BR10X first-generation architecture on TSMC 7nm** — the same architecture family as BR100/BR104. The stale element in the baseline report is the **generation**, not the naming convention: BR100/BR104 were announced in 2022, blocked by the October 2022 TSMC suspension, and never reached volume. Biren's product URLs are literally `/product/hardware/106m/`, `/166m/` etc., and there is **no BR100 or BR104 product page on birentech.com**.

## 10. Shipping BR10X SKUs

birentech.com's hardware menu lists exactly five SKUs. Every page publishes **form factor and peak power only** — no FLOPS table, no memory table, no process node.

| SKU | Form factor | Peak power | Silicon |
|---|---|---|---|
| 壁砺™166M | 4U OAM V1.1 module, air-cooled | 550 W | 2.5D chiplet, 2× BR106 dies |
| 壁砺™166L | OAM module, cold-plate liquid-cooled | 600 W | 2.5D chiplet, 2× BR106 dies |
| 壁砺™166C | FHFL (290 mm) dual-width PCIe inference card | 300 W | 2.5D chiplet, 2× BR106 dies |
| 壁砺™106M | OAM module, air-cooled | 400 W | Single BR106 die |
| 壁砺™106B | FHFL dual-width PCIe card | not disclosed | Single BR106 die |

**Production history:**

| Part | Development | Tape-out | Mass production | Notes |
|---|---|---|---|---|
| BR106 | from 2020 | 2021 | **January 2023** | Single die; first commercially shipping Biren silicon |
| BR110 | — | — | **October 2024** | Same architecture as BR106, single die, edge-inference positioning; no product page |
| BR166 | — | — | **August 2025** | 2.5D chiplet co-packaging two BR106 dies with a die-to-die interconnect; shipped at scale 2H2025 |

Vendor claim for BR166: *"compute and memory performance doubled versus the prior generation."* Unquantified and unverified.

**Sales:** combined BR106 + BR110 sales *"exceeded 12,000 units"* — a **cumulative** figure from the HK IPO prospectus (HKEX listing hearing 2025-12-17, IPO January 2026), reported via secondary Chinese media. Cut-off dated, not a current run rate.

### ⚠️ Rejected figure

The "**BF16 800 TFLOPS / 128 GB HBM**" specification widely attributed to the 166 series appears **only** on Chinese aggregators (traced to a Toutiao post). All four Biren product pages were fetched directly and contain no such table. A search summarizer falsely attributed these numbers to the 166M product page. **Not adopted.** No public datasheet gives FLOPS, HBM type/capacity/bandwidth, per-link BLink bandwidth, or process node for the 166 series or for BR20X.

## 11. BLink 2.0, NPO optics, and the distributed decoupled supernode (WAIC 2026)

Announced at **WAIC 2026 (Shanghai, 2026-07-17 to 07-20)**. This is an **architecture/roadmap announcement**: no availability date, per-link bandwidth, or customer was disclosed.

### Distributed decoupled supernode

The structural idea is that **GPU nodes and switch nodes are physically separated and optically linked**. Because the switch nodes leave the GPU cabinet, the GPU nodes can stay in **standard server form factors** rather than a bespoke high-density rack. Three tiers:

| Tier | Scale | Interconnect |
|---|---|---|
| Standard server | 16 cards | Electrical |
| High-density cabinet | 128 cards | Electrical — the prior scale-up ceiling |
| Distributed decoupled supernode | **1,024 cards** | **NPO optical** |

### BLink 2.0 capabilities (as stated)

1. **Memory-semantic interconnect** — up to 1,024 GPUs sharing one memory space.
2. **In-network computing** — collectives offloaded into the switches. This implies a BLink switch ASIC, a departure from BR100-era BLink 1.x, which was explicitly switchless point-to-point.
3. **Intelligent congestion control.**
4. **Multi-layer link self-healing**, physical layer through framework layer.

**Bandwidth: not disclosed.** Biren cites a per-GPU scale-up bandwidth *requirement* **exceeding 1 TB/s** as motivation; that is a requirement statement, not a product figure.

### NPO — near-packaged optics

Biren's stated rationale: copper *"attenuates severely within 3 metres."* NPO fuses the optical engine into the GPU module, **removes the high-power DSP chip** (retimer), and reaches *"several hundred metres."*

> ⚠️ A "**224 Gbps port rate**" attributed to this announcement in some secondary write-ups is **not present** in Biren's press release and has been dropped.

## 12. BR20X — second-generation architecture

| Attribute | Value |
|---|---|
| Also called | "BR2xx series" in WAIC-day press coverage |
| Generation | 2nd-generation Biren architecture (successor to BR10X) |
| Numerics | **Native FP8 and FP4** |
| Packaging | Chiplet (die count not disclosed) |
| Interconnect | Native supernode interconnect — BLink 2.0, ~1,000-card scale |
| Compute / memory | "Increased compute density, memory capacity and memory bandwidth" — no absolute figures |
| Process node | **not disclosed** |
| R&D start | 2024 |
| Status | **Architecture design complete; in physical design and tape-out verification (流片验证).** Not taped out. Not sampling. Not shipping. |
| Planned launch | 2026 (some reports 2H26 / autumn 2026) — a stated plan |
| Follow-ons | BR30X (cloud training/inference), BR31X (edge inference) — targeted 2028 |

**Sourcing caveat:** the string "BR20X" comes from prospectus / annual-report language, analyst notes (国海证券), Baidu Baike, and a Biren LinkedIn post — all secondary. Primary WAIC coverage uses "BR2xx series."

## 13. Deployment at scale — the 2,048-card cluster, qualified

Biren's FY2025 report states it delivered multiple thousand-card intelligent computing clusters, *"including a 2,048-card optical-interconnect / optical-switching GPU supernode cluster."*

Independent WAIC coverage (163.com, Toutiao) attributes this to the **previous-generation dOCS** (distributed optical circuit switching) supernode: a **32-card / 4-chassis building block**, aggregated into a 2,048-card cluster at a national-level computing platform reported as **Shanghai INESA (上海仪电)**.

**Correct characterization:** this is *not* a 2,048-card single scale-up / shared-memory domain, and it is *not* BLink 2.0. Biren's own WAIC press release mentions the national-platform dOCS deployment **without any card count**; the 2,048 figure originates in the annual report and media coverage.

## 14. FY2025 results (published 2026-03-30 — pre-baseline)

| Metric | FY2025 | YoY |
|---|---|---|
| Revenue | RMB 1.0346 B (10.346 亿元) | +207.2% (from RMB 337 M) |
| Gross profit | RMB 557.0 M | +210.8% |
| Gross margin | 53.8% | +63 bps |
| R&D expense | RMB 1.476 B | +78.5% |
| Adjusted net loss | ~RMB 874 M | — |
| Cash + financial assets (end-2025) | RMB 2.896 B | — |
| IPO net proceeds (early 2026) | RMB 5.631 B | — |
| Total funding reserve | > RMB 8.5 B | — |

Two hazards: (1) sources are 亿元-denominated — RMB 10.35亿 = RMB 1.035 **billion**, not RMB 10.35 billion; (2) the **IFRS reported loss is far larger** than the adjusted loss, dominated by non-cash fair-value movements on convertible preferred shares. Do not quote the headline loss without that caveat.

## 15. Not disclosed / not verified

- FLOPS, HBM type/capacity/bandwidth, process node for BR106 / BR110 / BR166 / BR20X
- Per-link and aggregate BLink 2.0 bandwidth; BLink bandwidth for the shipping 壁砺 generation
- BR166 die-to-die bandwidth (the 896 GB/s figure is BR100/CoWoS-S specific and must not be carried over)
- BR20X die count, chiplet composition, tape-out date
- No Biren paper found at Hot Chips 2026 / ISCA 2026 / ISSCC 2026. For Hot Chips 38 the program page returned **HTTP 403**, so this is *"could not verify"*, not *"absent"*. Hot Chips 38 runs 2026-08-23 to 08-25 — after this investigation — and no slides or abstracts exist yet.
- No MLPerf submission found for any Biren part

**Methodology caveat:** this update ran with an exhausted web-search budget; independent discovery went through WebFetch against DuckDuckGo HTML and direct primary URLs. English-language coverage is thinner than ideal, and Baidu Baike returned HTTP 403 on direct fetch and was read only via search snippets.

## Sources — added 2026-08-08

**Primary (vendor):**
- [birentech.com homepage — hardware menu: 壁砺™166L / 166M / 166C / 106M / 106B; no BR100/BR104 page](https://www.birentech.com/)
- [壁砺™166M — 4U OAM V1.1 air-cooled, 峰值功耗 550W; no FLOPS/memory](https://www.birentech.com/product/hardware/166m/)
- [壁砺™166L — cold-plate liquid-cooled OAM, 600W peak; no FLOPS/memory](https://www.birentech.com/product/hardware/166l/)
- [壁砺™166C — 全高全长(290mm) 双宽 PCIe inference card, 峰值功耗 300W](https://www.birentech.com/product/hardware/166c/)
- [壁砺™106M — air-cooled OAM, 峰值功耗 400W](https://www.birentech.com/product/hardware/106m/)
- [Biren newsroom index](https://www.birentech.com/news/)
- [Biren WAIC 2026 press release — NPO, BLink™2.0 four capabilities, 16/128/1024 tiers, ">1TB/s", copper <3m, "数百米", "去掉了高功耗的DSP芯片", dOCS national platform (no card count), BR2xx FP8/FP4, Token Factory](https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/)

**Independent media:**
- [IT之家 via Sohu (2026-07-19) — WAIC 2026 NPO, BLink2.0, three tiers, BR2xx](https://www.sohu.com/a/1052133469_114760)
- [Jiemian — WAIC 2026; "BLink2.0超节点互连协议，可让最多1024张GPU共享同一个内存空间"; no tape-out date for BR2xx](https://www.jiemian.com/article/14787272.html)
- [NetEase 163.com (2026-07-17) — NPO upgrade from dOCS; prior-gen dOCS supernode "reaching a scale of 2,048 cards"](https://www.163.com/tech/article/L22F1NH700098IEO.html)
- [10jqka (2026-07-18) — 首次推出NPO光互连、分布式解耦架构超节点，支持单个超节点1024卡](https://www.10jqka.com.cn)
- [WAIC 2026 official — Shanghai, July 17–20, 2026](https://english.shanghai.gov.cn/en-DWAIC2026/index.html)
- [Tencent News (2026-03-30) — FY2025 revenue 10.35亿元, +207.2%](https://news.qq.com/rain/a/20260330A08PTQ00)
- [Jiemian — FY2025 revenue 10.346亿, gross profit 5.570亿 (+210.8%)](https://www.jiemian.com/article/14185757.html)
- [Futu — R&D 14.76亿元 (+78.5%)](https://news.futunn.com/post/70839289/)
- [Xueqiu — gross margin 53.8% (+63 bps), adjusted loss ~8.74亿](https://xueqiu.com/9983210953/381860568)
- [Sohu (2026-03-30) — cash 28.96亿, IPO net proceeds 56.31亿, reserve >85亿; 2,048-card optical-switching supernode cluster; BR20X planned 2026 FP8/FP4](https://www.sohu.com/a/1003843604_313745)
- [ChinaBizInsider — HKEX listing hearing 2025-12-17; prospectus 12,000+ combined units; backlog ~RMB 1.2 B](https://chinabizinsider.com/chinese-gpu-chip-designer-biren-technology-clears-hong-kong-ipo-hurdle-eyes-listing/)

**Encyclopedia (cites Biren official):**
- [Baidu Baike — BR106: dev 2020, tape-out 2021, mass production Jan 2023, 7nm, BR10X; form factors 106B and 106M](https://baike.baidu.com/item/BR106/67163892)
- [Baidu Baike — 壁砺166系列: "通过共封装两个壁砺106芯片裸晶，利用芯粒技术及裸晶间互连技术提升性能"; launched 2025](https://baike.baidu.com/item/%E5%A3%81%E7%A0%BA166%E7%B3%BB%E5%88%97/67163995)
- [The Paper — annual report coverage: "BR166系列于2025年8月开始量产"; BR110 mass production Oct 2024; BR30X/BR31X targeted 2028](https://m.thepaper.cn)

**Rejected as spec (recorded for traceability):**
- [Toutiao aggregator — "BF16：800 TFLOPS…显存：128GB HBM" for the 166 series; sole source, contradicted by the absence of any such table on birentech.com](https://www.toutiao.com/w/1865240228792329/)
