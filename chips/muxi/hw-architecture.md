# MetaX (沐曦) GPU Hardware Architecture Reference

*as_of: 2026-08-08*
*chip: muxi*
*device_class: GPU (China, 沐曦)*
*families: 曦云 C-series (training) / 曦思 N-series (inference) / 曦索 X-series (AI4S) / 曦彩 G-series (graphics) / 曦景 S-series (supernode systems)*

---

## Chip Identity

| Attribute | MXC500 (C500) | MXC550 (C550) | MXC600 (C600) | MXN100 (N100) |
|-----------|---------------|---------------|---------------|---------------|
| Brand | 曦云 Xiyun | 曦云 Xiyun | 曦云 Xiyun | 曦思 Xisi |
| Purpose | AI Training | AI Training + Cluster | Hopper-class Training | AI Inference + Video |
| Process | TSMC 7nm | 7nm | Domestic (undisclosed) | 7nm |
| FP32 | ~15 TFLOPS | N/D | ~60 TFLOPS (est.) | — |
| FP16 | ~80 TFLOPS | N/D | ~500 TFLOPS (est.) | 80 TFLOPS |
| FP8 | — | — | 1,000 TFLOPS | — |
| INT8 | ~160 TOPS | N/D | ~2,000 TOPS (est.) | 160 TOPS |
| Memory | HBM2E ~64 GB | N/D | HBM3e 144 GB | HBM2E |
| Mem BW | N/D | N/D | 3.6 TB/s | N/D |
| TDP | N/D | N/D | N/D | N/D |
| Form factor | PCIe FHFL | OAM | OAM (expected) | PCIe |

N/D = Not officially disclosed

### Chip Identity — products added to this survey 2026-08-08

| Attribute | MXC588 (C588) | MXN300 (N300) | MXX206 (X206) | MXX300 (X301 / X302) |
|-----------|---------------|---------------|---------------|----------------------|
| Brand | 曦云 Xiyun | 曦思 Xisi | 曦索 Xisuo | 曦索 Xisuo |
| Purpose | AI Training | AI Inference (high density) | Scientific intelligence (AI4S) | Scientific intelligence (AI4S), "full-precision mixed compute" |
| Launch | 2025-09-23 (曦果发布会, with C600) | Not established | 2026-01-27 (智算申城 forum) | WAIC 2026, Jul 17–20 2026 |
| Status | Listed product (+ C588 Server) | Listed product (+ N300 Server) | Listed product (+ X206 Server) | **Announced only** (+ X302 Server listed) |
| Process | N/D | "Domestic advanced process" (node N/D) | N/D | N/D — fully domestic supply chain (vendor-stated) |
| FP64 | N/D | N/D | N/D | Supported per "全精度混合算力" claim; **rate N/D** |
| FP32 / FP16 / FP8 / INT8 | N/D | N/D | N/D | N/D |
| Memory | N/D | **N/D** (aggregator "48 GB" NOT confirmed) | N/D | "Large-capacity high-bandwidth memory" (type/capacity/BW N/D) |
| Mem BW | N/D | N/D | N/D | N/D |
| TDP | N/D | **500 W max board power** | N/D | N/D |
| Form factor | N/D | FHFL dual-slot PCIe | N/D | N/D |
| Scale-up | N/D | MetaXLink across 4 cards; up to 16 GPUs/chassis via PCIe-Switch full interconnect | N/D | MetaXLink + **MLoE** multi-card high-speed interconnect |

⚠️ The "48 GB" memory capacity and "14nm" process figures circulating for the N300 come from low-quality aggregator listings only. MetaX's own product page states neither. They are recorded here as **not confirmed** and are deliberately not entered as specs.

---

## Compute Engine

```
MetaX GPU — SIMT Architecture
├── WARP-BASED SIMT EXECUTION
│   ├── Threads organized into warps (warp width undisclosed)
│   ├── Hardware warp scheduler with divergence handling
│   └── Kernel launch: <<<grid, block, shmem, stream>>> (CUDA-compatible)
│
├── COMPUTE UNIT (CU) — Fundamental unit
│   ├── FP32 / BF16 / FP16 vector SIMT lanes
│   ├── INT8 / INT4 quantized inference units
│   ├── Tensor Acceleration Unit:
│   │   ├── FP16/BF16/INT8 matrix ops (C500 generation)
│   │   └── + native FP8 Tensor instructions (C600 generation)
│   └── Local Shared Memory (programmer-managed; capacity undisclosed)
│
└── DIE SUMMARY
    ├── MXC500: ~15 TFLOPS FP32 | ~160 TOPS INT8
    └── MXC600: 1,000 TFLOPS FP8 | ~2.5 TFLOPS/W efficiency
```

### Peak Performance

| Metric | MXC500 | MXC600 | NVIDIA A100 (ref) | NVIDIA H20 (ref) |
|--------|--------|--------|-------------------|------------------|
| FP32 | ~15 TFLOPS | ~60 (est.) | 19.5 TFLOPS | 19.5 TFLOPS |
| FP16/BF16 | ~80 TFLOPS | ~500 (est.) | 77.97 TFLOPS | 148 TFLOPS |
| FP8 | — | 1,000 TFLOPS | — | 296 TFLOPS |
| INT8 | ~160 TOPS | ~2,000 (est.) | 624 TOPS | 592 TOPS |
| Memory BW | N/D | 3.6 TB/s | 2.0 TB/s | 4.0 TB/s |

MXC500 represents ~2–3 year lag vs NVIDIA frontier. MXC600 targets H20/Hopper FP8 competitive tier.

### 曦索 X-series — full-precision mixed compute (added 2026-08-08)

The X-series is a second compute profile inside MetaX's GPGPU line, aimed at AI-for-Science (AI4S) rather than LLM throughput. MetaX describes the X300 series (X301 / X302) as delivering **"全精度混合算力" — full-precision mixed compute** — meaning FP64-class numerics coexist with the low-precision AI datatypes on the same self-developed GPGPU architecture, backed by large-capacity high-bandwidth memory.

This matters architecturally because the C-series lineage above documents no FP64 path at all: the C500 generation is FP32/BF16/FP16/INT8 and C600 adds FP8/INT4. The X-series is therefore the first MetaX product family positioned for double-precision numerical simulation.

Vendor-stated target workloads: numerical weather prediction, oceanography, computational fluid dynamics, molecular dynamics, materials science, life sciences.

**Not disclosed for X206 / X301 / X302**: FP64 rate, any FLOPS or TOPS figure, memory type / capacity / bandwidth, TDP, process node, form factor, die configuration. **Status: X206 launched 2026-01-27; X300 series announced only (WAIC 2026).**

---

## Memory Hierarchy

```
Per-CU: local registers + shared memory (capacity undisclosed)
         ↓
On-chip: L2 cache / SRAM (capacity undisclosed)
         ↓
Off-chip:
  MXN100:  HBM2E (capacity undisclosed)
  MXC500:  HBM2E, ~64 GB (benchmark-reported)
  MXC600:  HBM3e, 144 GB, 3.6 TB/s
  MXC588:  not disclosed
  MXN300:  not disclosed  (aggregator "48 GB" NOT confirmed by MetaX)
  MXX206:  not disclosed
  MXX300 (X301/X302):  "large-capacity high-bandwidth memory" (vendor phrasing;
                        type, capacity and bandwidth all not disclosed)
```

MXC600 remains the only MetaX part with a vendor-published memory type, capacity and bandwidth. No memory figures have been released for the C588, N300 or any X-series part.

---

## Host Interface and Packaging

| Product | Form Factor | Interface | Cooling |
|---------|-------------|-----------|---------|
| MXN100 | PCIe card | PCIe | Air |
| MXC500 | PCIe FHFL | PCIe (gen N/D) | Air / Liquid |
| MXC550 | OAM | Custom OAM baseboard | Liquid |
| MXC600 | TBD (OAM expected) | TBD | Liquid (expected) |
| MXC588 | N/D | N/D | N/D |
| MXN300 | **PCIe FHFL dual-slot** | PCIe (gen N/D) | N/D (**max 500 W board power**) |
| MXX206 / MXX301 / MXX302 | N/D | N/D | N/D |

**C550 Shanghai Cube**: 47U single cabinet containing 128 liquid-cooled MetaX C550 OAM cards — the flagship ultra-high-density deployment configuration.

**N300 Server**: up to **16 N300 GPUs per chassis**, connected by a **PCIe-Switch full-interconnect** topology; a "Common" topology and 8-GPU configurations are also offered. Within the card group, MetaX specifies **MetaXLink across 4 cards**, so the 16-GPU chassis is a hybrid — MetaXLink islands of 4 bridged by PCIe switching, not a flat PCIe-only fabric.

---

## MetaXLink Scale-up Interconnect

MetaX's proprietary GPU-to-GPU interconnect (NVLink/BLink analog):

| Attribute | Detail |
|-----------|--------|
| Name | MetaXLink (abbreviated MX) |
| Topology | Full-mesh per server; 3D Mesh at supernode |
| GPUs/server | 8 (OAM format, C550); 4-card MetaXLink groups on N300 PCIe cards |
| NUMA | All 8 GPUs appear in NUMA node 0 |
| Supernode | Up to 64 GPUs (8 servers × 8 GPUs, 3D Mesh topology, low-latency electrical) |
| Scale-out | Standard Ethernet / RoCE via host NIC + MXCCL |
| Cluster proven | 10,000+ GPUs in commercial operation (9 clusters across China, end-2024) |
| Switch chip | Not disclosed |
| MLoE (X300 series) | A second high-speed multi-card interconnect named alongside MetaXLink for the X300 series. Mechanism, bandwidth, topology and radix all **not disclosed** |

### Supernode Family (updated 2026-08-08)

MetaX now ships/lists four distinct rack-scale configurations. They coexist — the newest does not replace the older ones.

| Supernode | Base part | Scale | Fabric | Status |
|-----------|-----------|-------|--------|--------|
| **C500 Shanghai Cube** | C550 OAM | 128 liquid-cooled cards in a 47U full cabinet | MetaXLink, electrical | Listed product |
| **C550 3D Mesh Supernode** | C550 OAM | Up to 64 cards across up to 8 servers, full-cabinet deployment | 3D Mesh, electrical (memory-semantic) | Listed product — **still current** |
| **C500X Optical Interconnect Supernode** | C500X | Secondary reporting: 16→64 GPUs, up to 8 machines | Hybrid optical/electrical; **DragonFly** topology per secondary reporting | Listed product. ⚠️ Scale and topology figures are **medium confidence** — no vendor spec sheet; material traces to WAIC 2025 |
| **曦景 S600 (Xijing S600)** | Not disclosed | **64 GPU cards in a single cabinet**; cabinet-to-cabinet expansion to 万卡级 (10,000+ card) clusters | **In-cabinet full interconnect** with **"zero-cable direct-connect" (0线缆直连)** between compute nodes and switch nodes; supports EP / TP and other multi-dimensional parallel strategies for both training and inference | **Announced only** — WAIC 2026, Shanghai, Jul 17–20 2026. No sampling, shipping, pricing, customer or benchmark evidence |

**Not disclosed for S600**: per-link bandwidth, aggregate bisection bandwidth, which GPU SKU populates the cabinet, total supernode FLOPS, memory per cabinet, power and cooling envelope, switch silicon.

> The S600 does **not** supersede the C550 3D Mesh Supernode row above. Both are 64-card-class configurations, but the C550 supernode distributes those 64 cards across up to 8 servers with an electrical 3D Mesh, while the S600 is a single-cabinet, cable-free, fully-interconnected design. MetaX lists them as separate products.

---

## Architecture Generations

### Gen 1: MXN100 (Xisi 曦思 N100) — 2023
- First MetaX GPU to reach mass production
- 7nm process
- AI inference + video transcoding (not general training)
- 160 TOPS INT8, 80 TFLOPS FP16, HBM2E
- 128-channel encode / 96-channel decode, 8K HEVC/H.264/AV1/AVS2
- Deployed in Chinese cloud video platforms

### Gen 2: MXC500 (Xiyun 曦云 C500) — 2024
- First MetaX AI training GPU at scale
- 7nm (TSMC), ~15 TFLOPS FP32, HBM2E ~64 GB
- MXMACA 2.0 platform (CUDA-compatible)
- 100B+ parameter LLM training (claimed)
- 10,000+ GPUs deployed in 9 clusters across China
- EDWC (East Data West Compute) project in Ningxia DC

### Gen 2.5: MXC550 (Xiyun 曦云 C550) — 2025
- OAM form factor (Open Accelerator Module standard)
- MetaXLink full-mesh interconnect
- C550 3D Mesh Supernode: 64 cards (8 servers × 8 GPUs)
- C550 Shanghai Cube: 128 liquid-cooled cards in 47U cabinet

### Gen 3: MXC600 (Xiyun 曦云 C600) — Announced July 2025
- **First fully domestic MetaX GPU** (design + mfg + pkg + test within China)
- Native FP8 Tensor instructions: 1,000 TFLOPS
- HBM3e: 144 GB, 3.6 TB/s (3× C500 bandwidth)
- Multi-precision: FP32/BF16/FP16/FP8/INT8/INT4
- ~2.5 TFLOPS/W at FP8 (claims first tier among domestic GPUs)
- Surpasses NVIDIA H20 single-card compute (claimed)
- Targets NVIDIA Hopper FP8 experience tier
- Small-batch Q4 2025 → **large-scale shipment during 2026** per company statement (CPO Sun Guoliang, 2026-07-08); some orders booked into 2027. Executive statement, not an audited shipment disclosure

### Gen 3 (cont.): MXC588 (Xiyun 曦云 C588) — Launched 2025-09-23
- Launched alongside the C600 at MetaX's 曦果发布会
- C588 Server is a listed product
- No specifications disclosed (compute, memory, process, TDP, form factor all N/D)
- Recorded here 2026-08-08 as a prior gap in this survey, not as new 2026 news

### Gen 3 derivative: MXN300 (Xisi 曦思 N300) — inference / high density
- C600-generation inference derivative
- FHFL dual-slot PCIe card; **500 W maximum board power**
- Proprietary GPGPU architecture on a "domestic advanced process" (node N/D)
- MetaXLink high-speed interconnect across 4 cards
- N300 Server: up to 16 N300 GPUs per chassis, PCIe-Switch full interconnect
- Memory capacity, memory bandwidth and INT8/FP8 TOPS all **not disclosed**; announcement date not established
- ⚠️ Aggregator-sourced "48 GB" and "14nm" figures are **not confirmed** and are not adopted here

### Gen 4 (parallel line): 曦索 X-series (Xisuo) — AI4S / scientific compute
- **X206** — launched 2026-01-27 at the 智算申城 forum, Shanghai. First X-series product. Specs N/D. X206 Server listed
- **X300 series (X301, X302)** — announced at WAIC 2026 (Jul 17–20, 2026). Second X-series generation. X302 Server listed
- Vendor-stated: fully self-developed GPGPU architecture on a fully domestic supply chain; **"full-precision mixed compute" (全精度混合算力)** implying FP64 alongside low-precision AI datatypes; large-capacity high-bandwidth memory; **MetaXLink + MLoE** multi-card interconnect; MXMACA software stack
- Target workloads: numerical weather prediction, oceanography, computational fluid dynamics, molecular dynamics, materials science, life sciences
- FP64 rate, all FLOPS/TOPS, memory type/capacity/BW, TDP and process node **not disclosed**
- **Status: announced only** (X300 series)

### Gen 4 (parallel line): 曦景 S-series (Xijing) — supernode systems
- **S600** — announced at WAIC 2026 (Jul 17–20, 2026). MetaX's first S-series product; a rack-scale system rather than a chip
- 64 GPU cards in a single cabinet; in-cabinet full interconnect; zero-cable direct-connect between compute and switch nodes; EP/TP parallel support; cabinet-to-cabinet expansion to 10,000+ card clusters
- Per-link bandwidth, GPU SKU, total FLOPS **not disclosed**
- **Status: announced only**

### Gen 4+: MXC700 (Xiyun 曦云 C700) — disclosed, unannounced
- On MetaX's **2026-04-08 FY2025 results call**, chairman 陈维良 (Chen Weiliang) stated that C700 **core chip design and functional verification are largely complete** and the part is undergoing deeper performance optimization
- Project initiated **April 2025**; positioned to extend beyond C600's 信创 (state-sector) base into internet-sector customers
- This is a **company disclosure**, which outranks the earlier "rumored" classification — but C700 remains **unannounced as a product with no disclosed specifications and no announced launch date**
- The "H100-parity" performance target and a "mass production late 2027" timeline appear only in aggregator write-ups and are **unverified**

---

## Export Control Context

MetaX originally designed on TSMC 7nm. With the MXC600, MetaX pivots to a fully domestic supply chain, bypassing TSMC dependency under US export controls. The process node is not officially disclosed but is presumed to use SMIC or equivalent domestic node. The MXC600 announcement on the same day as Enflame's L600 (July 28, 2025) signals coordinated industry response to H20 potential re-entry into the China market.

---

## Sources

- [Tom's Hardware — MetaX Xisi N100 debut](https://www.tomshardware.com/news/metax-chinese-gpu-developer-unveils-first-product)
- [WCCFTech — MetaX N100 160 TOPS](https://wccftech.com/chinese-chipmaker-metax-unveils-first-gpu-targeted-towards-ai-features-160-tops-of-compute/)
- [tphuang X — MXC500 15 TFLOPS](https://x.com/tphuang/status/1700511558961340656)
- [IT之家 — 曦云 C600 首款全国产](https://www.ithome.com/0/890/942.htm)
- [EastMoney — C600 对标 Hopper FP8](https://caifuhao.eastmoney.com/news/20250824025550377224840)
- [Red Hot Cyber — C600](https://www.redhotcyber.com/en/post/made-in-china-muxi-presents-the-xiyun-c600-general-purpose-gpu/)
- [MetaX C550 Server product page](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=44)
- [MetaX C550 3D Mesh Supernode](https://www.metax-tech.com/en/goods/prod.html?cid=112&id=55)
- [MXC500 benchmark blog](https://wangjunjian.com/mxc500/benchmark/2025/02/13/Performance-Stress-Testing-of-the-MuXin-MXC500-for-Large-Model-Inference.html)
- [HAMi MetaX support](https://github.com/Project-HAMi/HAMi/blob/master/docs/metax-support.md)
- [MetaX Wikipedia](https://en.wikipedia.org/wiki/MetaX)

### Added 2026-08-08

- [MetaX Newsroom — WAIC 2026 (S600 + X300)](https://www.metax-tech.com/en/ndetail/12629.html)
- [MetaX product catalog — all series](https://www.metax-tech.com/en/goods/prod.html?cid=3)
- [MetaX N300 product page](https://www.metax-tech.com/en/goods/prod.html?cid=106&id=66)
- [MetaX N300 Server product page](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=69)
- [ITHome — WAIC 2026 MetaX coverage](https://www.ithome.com/0/978/460.htm)
- [Sina Finance — WAIC 2026 (2026-07-20)](https://finance.sina.com.cn/tech/shenji/2026-07-20/doc-iniimuyk4915570.shtml)
- [Sina Finance — Sun Guoliang on C600 volume (2026-07-08)](https://finance.sina.com.cn/wm/2026-07-08/doc-inihceis8322692.shtml)
- [Tencent News — FY2025 results call, C700 status (2026-04-08)](https://news.qq.com/rain/a/20260408A066PB00)
- [TrendForce CN — MetaX shipment note (2026-07-09)](https://www.trendforce.cn/industry-news/semiconductors/20260709-6285.html)
- [EastMoney — FY2025 / Q1 2026 financials](https://finance.eastmoney.com/a/202604293724854763)
- [PEDaily — MetaX feature (2026-07-28)](https://news.pedaily.cn/20260728/135041.shtml)
