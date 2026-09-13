# MetaX (沐曦) GPU Hardware Architecture Investigation

*as_of: 2026-08-08 (baseline investigation 2026-04-05; see the dated update section at the end)*
*chip: muxi*
*device_class: GPU (China, 沐曦)*
*resource: hw-architecture*

---

## Summary

MetaX Integrated Circuits (Shanghai) Co., Ltd. (沐曦集成电路) is a Chinese GPGPU designer founded in 2020 by three veterans of AMD — chairman and CEO Chen Weiliang (陈维良, 14-year AMD career) and co-CTOs Peng Li and Yang Jian. The company produces full-stack GPGPU chips under three series: the **MXC / 曦云 (Xiyun)** training/compute line, the **MXN / 曦思 (Xisi)** inference line, and the **MXG / 曦彩 (Xicai)** graphics/cloud-gaming line. All products are built on proprietary GPU IP with a self-developed ISA and the MXMACA heterogeneous computing software stack.

---

## Chip Identity

### Product Families

| Series | Brand | Target Workload | Status |
|--------|-------|----------------|--------|
| MXC (曦云 Xiyun) | C-Series | AI Training + General Compute (GPGPU) | Production: C500, C550; announced: C600 |
| MXN (曦思 Xisi) | N-Series | AI Inference + Video Transcoding | Production: N100 (MXN100), N260 |
| MXG (曦彩 Xicai) | G-Series | Cloud Gaming / Graphics Render | Announced; 7nm target 2025 |

> **Superseded 2026-08-08** — this three-series view is incomplete. MetaX now also fields the **曦索 (Xisuo) X-series** (AI4S / scientific compute: X206, X301, X302) and the **曦景 (Xijing) S-series** of supernode systems (S600). The C-series also includes C500X and C588, and the N-series includes N260 and N300. See the "Investigation Update — 2026-08-08" section at the end of this file.

### Key Products

| Product | Architecture | Process | Target | Status |
|---------|-------------|---------|--------|--------|
| MXN100 (曦思 N100) | GPGPU (inference-optimized) | 7nm | A100/inference peer | Mass production (2023) |
| MXC500 (曦云 C500) | GPGPU (training) | 7nm | NVIDIA A100 | GA 2024; 10,000+ deployed |
| MXC550 (曦云 C550) | GPGPU (next-gen training) | 7nm | Beyond A100 | Production 2025 |
| MXC600 (曦云 C600) | GPGPU (Hopper-class) | Domestic (not disclosed) | NVIDIA Hopper H20+ | Small batch Q4 2025 |

---

## MXN100 (Xisi N100) — Inference GPU

The first MetaX product to reach mass production:

```
MXN100 Compute
├── INT8 performance: 160 TOPS
├── FP16 performance: 80 TFLOPS
├── Memory: HBM2E
├── Video: 128-ch encode / 96-ch decode (HEVC, H.264, AV1, AVS2, 8K)
└── Interconnect: PCIe (gen not disclosed)
```

### MXN100 Use Cases
- AI inference in data center (NLP, vision)
- Video transcoding at scale (cloud streaming platforms)

---

## MXC500 (Xiyun C500) — Training GPU

The flagship AI training accelerator, benchmarked against NVIDIA A100:

```
MXC500 (Xiyun C500) — AI Training GPU
├── Process: 7nm
├── Architecture: GPGPU (full warp-based SIMT; proprietary ISA)
├── FP32 performance: ~15 TFLOPS (~77% of A100 19.5 TFLOPS)
├── INT8 / FP16 performance: 160 TOPS / 80 TFLOPS (est.)
├── Memory: HBM2E
│   └── Capacity: ~64 GB (reported from benchmarks)
│   └── Bandwidth: Not officially disclosed
├── Host interface: PCIe
├── Compatible: CUDA-compatible via MXMACA 2.0 platform
└── Capability: 100B+ parameter LLM training (claimed)
```

### Deployment Scale
- By end-2024: Nine compute clusters deployed across China
- Over 10,000 MXC500 GPUs in commercial operation
- Used in EDWC (East Data West Compute) project in Ningxia DC

### Variants
| Model | Capacity | Notes |
|-------|----------|-------|
| C500 | 64 GB HBM2E | Standard PCIe card |
| C500X | Optical Interconnect | C500X Optical Supernode — 3D Mesh, 8 servers × 8 GPUs = 64 cards |
| C280 | Lower-end variant | Lighter compute configuration |
| C290 | Mid-range variant | Mid-tier config |

---

## MXC550 (Xiyun C550) — Next-Generation Training GPU

The current-generation training accelerator (OAM form factor, 2025):

```
MXC550 (Xiyun C550)
├── Form factor: OAM (Open Accelerator Module) standard
├── Server: 8 × C550 OAM per dual-socket server
├── Topology support: 8-card full interconnection (MetaXLink)
├── Infrastructure: C550 3D Mesh Supernode
│   └── Up to 8 servers × 8 GPUs = 64 cards per supernode
│   └── Low-cost, low-latency 3D Mesh electrical interconnection
├── Flagship deployment: C550 Shanghai Cube
│   └── 47U single cabinet
│   └── 128 liquid-cooled C550 cards per cabinet
│   └── Ultra-high-density liquid-cooled design
└── Interconnect: MetaXLink (MX) — fastest GPU-to-GPU connection method
    └── All GPUs connected to NUMA node 0 in MX topology
```

---

## MXC600 (Xiyun C600) — Hopper-Class GPU

Announced July 2025, targeting NVIDIA Hopper FP8 capabilities — the most advanced MetaX product:

```
MXC600 (Xiyun C600) — Announced 2025
├── Memory: HBM3e
│   ├── Capacity: 144 GB
│   └── Bandwidth: 3.6 TB/s
├── FP8 performance: 1,000 TFLOPS (native on-chip FP8 Tensor instructions)
├── Precision support: FP32, FP16, BF16, FP8, INT8, INT4 (multi-precision)
├── Efficiency: ~2.5 TFLOPS/W at FP8 (first tier among domestic GPUs)
├── Supply chain: Fully domestic — design, manufacturing, packaging, test
├── Target: Surpasses NVIDIA H20 single-card compute (claimed)
├── FP8 competitor: Targets NVIDIA Hopper FP8 experience
├── Process node: Domestic foundry (not officially disclosed)
└── Status: Functional test phase → small-batch Q4 2025 → mass production TBD
```

### C600 Significance
- First "fully domestic" general-purpose GPU with HBM3e + FP8
- Achieves closed domestic supply chain (vs prior gens that used TSMC 7nm)
- Announced same day as Enflame L600 (July 28, 2025) — coordinated domestic GPU push
- Directly targets H20 replacement market (post NVIDIA export control restrictions)

---

## Compute Engine Architecture (General GPGPU)

MetaX GPUs implement a full-featured SIMT (Single Instruction Multiple Thread) GPGPU architecture, not a narrow inference ASIC:

```
SIMT Execution Model
├── Threads organized into warps (warp width not publicly disclosed)
├── Hardware warp scheduler with divergence handling
├── Kernel launch: <<<grid, block, shmem, stream>>> (CUDA-compatible syntax via MXMACA)
├── Compute units: Shader processors with FP32, INT8, tensor acceleration
├── Tensor/matrix acceleration: Dedicated tensor units for FP16/BF16/INT8 (C500)
│   → FP8 tensor units added in C600
└── Full-function GPU: compute + video encode/decode + graphics pipeline (MXG series)
```

---

## Memory Hierarchy

```
Per-Compute-Unit: local registers + shared memory (capacity not disclosed)
         ↓
On-chip: L2 cache / scratchpad (capacity not disclosed)
         ↓
Off-chip Memory:
  MXN100:  HBM2E (capacity not disclosed)
  MXC500:  HBM2E, ~64 GB (from benchmark reports)
  MXC550:  Not officially disclosed
  MXC600:  HBM3e, 144 GB, 3.6 TB/s
```

---

## Host Interface and Packaging

| Product | Form Factor | Host Interface | Cooling |
|---------|-------------|---------------|---------|
| MXN100 | PCIe card | PCIe | Air |
| MXC500 | PCIe FHFL | PCIe Gen4/5 (not disclosed) | Air / Liquid |
| MXC550 | OAM | Custom OAM baseboard | Liquid (Shanghai Cube: 128 cards/47U) |
| MXC600 | TBD | TBD | Liquid (expected) |

---

## Interconnect: MetaXLink (MX)

MetaX's proprietary GPU-to-GPU interconnect (NVLink/BLink analog):

| Attribute | Detail |
|-----------|--------|
| Name | MetaXLink (abbreviated MX) |
| Topology | Ring / full-mesh per server; 3D Mesh at supernode level |
| Cards per server | 8 (C550 OAM server) |
| Scale-up | Up to 64 GPUs per supernode (C550 3D Mesh Supernode) |
| Scale-out | Standard Ethernet / RoCE via host NIC + MXCCL collective library |
| NUMA topology | All 8 GPUs in one NUMA node 0 via MetaXLink |
| Switch chip | Not disclosed (may be point-to-point mesh) |

---

## Architecture Generations

| Generation | Product | Process | Key Features |
|-----------|---------|---------|-------------|
| Gen 1 | MXN100 (Xisi) | 7nm | First mass-produced; inference + video; HBM2E; 160 TOPS INT8 |
| Gen 2 | MXC500 (Xiyun C500) | 7nm (TSMC) | Full training; 15 TFLOPS FP32; HBM2E; MXMACA 2.0; 10K cluster proven |
| Gen 2.5 | MXC550 (Xiyun C550) | 7nm | OAM form factor; MetaXLink fabric; C550 Shanghai Cube |
| Gen 3 | MXC600 (Xiyun C600) | Domestic | FP8 1,000 TFLOPS; HBM3e 144 GB 3.6 TB/s; fully domestic supply chain |
| Gen 3+ | MXC700 (rumored) | TBD | Targeting Hopper-class full parity (H100-equivalent) |

> **Corrected 2026-08-08** — "MXC700 (rumored)" is now under-stated. Chairman Chen Weiliang disclosed on the **2026-04-08 FY2025 results call** that C700 core chip design and functional verification are largely complete and the part is in deeper performance optimization; project initiated April 2025. Correct classification: **company-disclosed, in design-verification/optimization, no specs released, no announced launch date.** The "H100-parity" target remains **aggregator-only and unverified**. See the 2026-08-08 update section.

---

## Comparison: MetaX vs Peers

| Metric | MXC500 | MXC600 | NVIDIA A100 | NVIDIA H20 | mthreads MTT S4000 |
|--------|--------|--------|-------------|------------|-------------------|
| Process | 7nm | Domestic | TSMC 7nm | TSMC 4nm | TSMC 12nm |
| FP32 | ~15 TFLOPS | ~60 TFLOPS (est.) | 19.5 TFLOPS | 19.5 TFLOPS | 25 TFLOPS |
| FP16/BF16 | ~80 TFLOPS | ~500 TFLOPS (est.) | 77.97 TFLOPS | 148 TFLOPS | 200 TFLOPS |
| FP8 | No | 1,000 TFLOPS | No | 296 TFLOPS | No |
| INT8 | 160 TOPS | ~2,000 TOPS (est.) | 624 TOPS | 592 TOPS | 200 TOPS |
| Memory | HBM2E ~64GB | HBM3e 144 GB | HBM2e 80 GB | HBM3 96 GB | GDDR6 48 GB |
| Mem BW | Not disclosed | 3.6 TB/s | 2.0 TB/s | 4.0 TB/s | 768 GB/s |
| TDP | Not disclosed | Not disclosed | 400W | 500W | 450W |
| Scale-up | MetaXLink | MetaXLink | NVLink 3 | NVLink 4 | MTLink 1.0 |

Note: MXC500 represents a ~2–3 year lag vs NVIDIA frontier; MXC600 attempts to close the gap to H20/Hopper FP8 tier.

---

## Export Control Context

MetaX originally designed on TSMC 7nm (MXN100, MXC500). The C600 announcement of a "fully domestic supply chain" signals a deliberate pivot away from TSMC in response to US export controls that restrict advanced foundry services to Chinese AI chip companies. The process node for C600 is not officially disclosed but is presumed to use SMIC N+2 or an equivalent domestic node — a significant achievement enabling HBM3e integration within China's manufacturing ecosystem.

---

## Sources

- [MetaX Wikipedia](https://en.wikipedia.org/wiki/MetaX)
- [Tom's Hardware — MetaX Xisi N100 debut](https://www.tomshardware.com/news/metax-chinese-gpu-developer-unveils-first-product)
- [WCCFTech — MetaX N100 160 TOPS](https://wccftech.com/chinese-chipmaker-metax-unveils-first-gpu-targeted-towards-ai-features-160-tops-of-compute/)
- [tphuang X — MXC500 15 TFLOPS FP32](https://x.com/tphuang/status/1700511558961340656)
- [IT之家 — 曦云 C600 首款全国产 GPU](https://www.ithome.com/0/890/942.htm)
- [ITHome Finance Sina — 曦云 C600](https://finance.sina.com.cn/tech/digi/2025-10-20/doc-infupzmi5164585.shtml)
- [EastMoney — C600 对标 Hopper FP8](https://caifuhao.eastmoney.com/news/20250824025550377224840)
- [Red Hot Cyber — C600 general GPU](https://www.redhotcyber.com/en/post/made-in-china-muxi-presents-the-xiyun-c600-general-purpose-gpu/)
- [MetaX Official — C550 Server](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=44)
- [MetaX Official — C550 3D Mesh Supernode](https://www.metax-tech.com/en/goods/prod.html?cid=112&id=55)
- [MetaX Official — MXMACA Platform](https://www.metax-tech.com/en/goods/platform.html?cid=4)
- [HAMi docs — MetaX GPU support](https://github.com/Project-HAMi/HAMi/blob/master/docs/metax-support.md)
- [China Biz Insider — MetaX IPO](https://chinabizinsider.com/metax-wins-nod-for-star-market-ipo-escalating-race-for-chinas-first-gpu-listing/)
- [SCMP — MetaX AMD veterans](https://www.scmp.com/tech/big-tech/article/3330511/meet-metax-chinese-ai-chip-hopeful-challenging-nvidias-dominance)
- [Global Times — MetaX IPO +692%](https://www.globaltimes.cn/page/202512/1350816.shtml)

---

# Investigation Update — 2026-08-08

*Scan window: 2026-04-05 → 2026-08-08. Verdict on the raw scan finding: PARTIALLY_CONFIRMED (one raw claim refuted). The text below reflects the post-verification, authoritative version — where the raw scan and the verification disagreed, the verification wins.*

*Primary sources: MetaX newsroom (WAIC 2026 item; MiniMax H3 item 2026-08-03), MetaX product catalog and per-product pages. Secondary: ITHome, Sina Finance (2026-07-08, 2026-07-20), Tencent News (2026-04-08 FY2025 results call), TrendForce CN, PEDaily, EastMoney, ZOL.*

## What changed

Four things happened in this window, plus two pre-baseline gaps were uncovered:

1. **New (WAIC 2026, Jul 17–20, 2026)**: 曦景 (Xijing) **S600** AI supernode, MetaX's first S-series product.
2. **New (WAIC 2026)**: 曦索 (Xisuo) **X300 series** (X301, X302) — second generation of the AI4S GPU line.
3. **Status change**: **MXC600** moved from Q4-2025 small-batch to large-scale shipment during 2026 (company statement, independent of WAIC).
4. **Status upgrade**: **MXC700** is no longer a rumor — it was disclosed on the 2026-04-08 FY2025 results call.
5. **Pre-baseline gap**: the entire **曦索 X-series** — the X206 launched **2026-01-27**, before this survey's 2026-04-05 baseline.
6. **Pre-baseline gap**: **C588** (launched 2025-09-23 alongside C600) and the **C500X Optical Interconnect Supernode** (material traces to WAIC 2025), plus the **N260** and **N300** inference parts.

## 1. 曦景 (Xijing) S600 supernode — CONFIRMED, new

MetaX's own newsroom item covers July 17–20, 2026 (the full WAIC 2026 window); press coverage ran July 17–18. **Use "at WAIC 2026, July 17–20, 2026" — a single-day date is more precision than the sources support.**

Vendor-stated attributes:

| Attribute | Value |
|---|---|
| Density | 64 GPU cards in a single cabinet (单机柜64卡高密度部署) |
| Intra-cabinet fabric | In-cabinet full interconnect, to cut data-exchange latency |
| Cabling | **"Zero-cable direct-connect" (0线缆直连)** between compute nodes and switch nodes |
| Parallelism | EP (expert-parallel), TP (tensor-parallel) and other multi-dimensional strategies; training and inference |
| Scale-out | Cabinet-to-cabinet expansion to 万卡级 (10,000+ card) clusters |

**Not disclosed**: per-link bandwidth, aggregate/bisection bandwidth, which GPU SKU populates the cabinet, total supernode FLOPS, memory per cabinet, power and cooling envelope, switch silicon.

**Status: announced only.** No sampling, shipping, pricing, customer or benchmark evidence found.

### Correction carried from verification

The raw scan claimed the S600 "supersedes the repo's 64 GPUs = 8 servers × 8 GPUs, 3D Mesh; the density is now 64 per cabinet, not per 8-server group." **This is wrong.** MetaX's **C550 3D Mesh Supernode** is still a listed product and is itself specified as up to 64 cards / up to 8 servers / full-cabinet deployment with electrical (memory-semantic) interconnect. The S600 is an *additional, higher-integration* supernode line alongside it. Both rows are kept in the deliverables.

The raw scan also claimed the S600 "combines Scale-up and Scale-out into one fusion architecture." That phrasing was **not found** in MetaX's own WAIC newsroom item or in the ITHome/Sina coverage and has been dropped. The confirmed differentiator the raw scan missed is the zero-cable direct-connect design.

## 2. 曦索 (Xisuo) X-series — CONFIRMED, but NOT a new family

The X-series launched **2026-01-27** at the 智算申城 forum in Shanghai with the **曦索 X206** — before this survey's baseline. The X300 series is the **second generation**, comprising **X301 and X302** (both on MetaX's product page; an X302 Server and an X206 Server are also listed).

Vendor-stated X300 attributes:

- Fully self-developed GPGPU architecture on a **fully domestic supply chain**
- **"Full-precision mixed compute" (全精度混合算力)** — FP64-class numerics alongside low-precision AI datatypes
- Large-capacity high-bandwidth memory
- **MetaXLink plus "MLoE"** high-speed multi-card interconnect
- MXMACA software stack

Target workloads: numerical weather prediction, oceanography, computational fluid dynamics, molecular dynamics, materials science, life sciences.

**Not disclosed**: FP64 rate, any FLOPS/TOPS figure, memory type/capacity/bandwidth, TDP, process node, form factor, die configuration.

**Status: announced only.**

Architectural significance for this survey: the C-series lineage documents no FP64 path (C500 generation is FP32/BF16/FP16/INT8; C600 adds FP8/INT4). The X-series is the first MetaX family positioned for double-precision numerical simulation, and it is mirrored on the software side by an AI4S repository cluster appearing on GitHub in July–August 2026.

## 3. MXC600 — large-scale shipment

On **2026-07-08**, Chief Product Officer / SVP **孙国梁 (Sun Guoliang)** stated that the MXC600 series — MetaX's core product for 2026 — has achieved **大规模出货 (large-scale shipment)**, with some orders already booked into 2027 and beyond; he framed general-purpose stable domestic compute as supply-constrained.

**Label this as a company statement, not an audited disclosure.** The audited numbers that do exist are:

- **33,649** training/inference GPU boards shipped in 2025 (**+147.31% YoY**)
- MetaX states cumulative GPU sales exceeded **55,000 units** across **10+** intelligent-computing clusters by end-2025

Neither corroborates C600 volume specifically — both predate it.

Derived hardware in the C600 generation: 曦云 **C600** cards and 曦思 **N300**.

## 4. MXC700 — MATERIAL CORRECTION to the prior repo state

The baseline investigation recorded "MXC700 (rumored) | TBD | Targeting full H100-class parity." That understates what is public.

On MetaX's **2026-04-08 FY2025 results call**, chairman **陈维良 (Chen Weiliang)** stated that 曦云 **C700 core chip design and functional verification are largely complete**, and that the part is undergoing **deeper performance optimization**. The project was **initiated in April 2025** and is positioned to extend beyond C600's 信创 (state-sector) base into **internet-sector customers**.

That is a company disclosure and outranks "rumor". Correct classification: **"disclosed by the company as in design-verification / performance-optimization; no specs released; no announced launch date."**

The **H100-parity target** and a **"mass production late 2027"** timeline appear only in aggregator write-ups — **LOW confidence, keep flagged as unverified.**

## 5. Product catalog (confirmed against MetaX's own pages)

| Series | Products |
|---|---|
| C (曦云 Xiyun) | C500, C550, C500X, C588, C600 |
| N (曦思 Xisi) | N100, N260, N300 |
| X (曦索 Xisuo) | X206, X301, X302 |
| G (曦彩 Xicai) | G100 |
| Servers | N260, N300, C500, C550, C588, C600, X206, X302; plus an N260 workstation |
| Supernodes | "C500 Shanghai Cube" liquid-cooled full cabinet; "C500X Optical Interconnect Supernode"; "C550 3D Mesh Supernode" |

**Freshness caveat.** The C500X optical supernode and the C588 are **not** new in this window. C588 was launched alongside C600 at MetaX's 曦果发布会 on **2025-09-23**; C500X optical-interconnect material traces to **WAIC 2025**. These are pre-baseline repo misses and must not be presented as April–August 2026 news.

Secondary reporting describes **C500X** as hybrid optical/electrical with a **DragonFly** topology scaling **16→64 GPUs** (up to 8 machines) — **medium confidence pending a vendor spec sheet.**

## 6. 曦思 N300 — PARTIALLY confirmed

Confirmed from MetaX's official product page:

- **FHFL dual-slot PCIe** card
- **Maximum board power 500 W**
- Proprietary GPGPU architecture on a **"domestic advanced process"** (node not stated)
- **MetaXLink high-speed interconnect across 4 cards**
- MXMACA stack

Confirmed from the official **N300 Server** page: **up to 16 N300 GPUs per chassis** with a **PCIe-Switch full-interconnect** topology; a "Common" topology and 8-GPU configurations are also offered.

The 4-card MetaXLink grouping matters: it contradicts a pure "PCIe-Switch only" reading of the 16-GPU server. The chassis is a hybrid — MetaXLink islands of 4 bridged by PCIe switching.

**Rejected figures.** The **48 GB memory** capacity appears only in low-quality aggregator listings and is **not confirmed by MetaX**. The **14nm** process figure comes from the same aggregator tier, is inconsistent with N300 being a C600-generation derivative, and should be treated as **NOT CONFIRMED / likely wrong**. Neither is entered as a spec anywhere in the deliverables. INT8/FP8 TOPS not disclosed. N300 announcement date not established.

## 7. Financials (confirmed)

| Period | Revenue | Net result |
|---|---|---|
| FY2025 | RMB **1.644 B** (+121.26% YoY) | Net loss RMB **789 M**, narrowed 43.97% from RMB 1.409 B |
| Q1 2026 | RMB **562 M** (+75.37% YoY) | Net loss RMB **98.84 M** (vs RMB 233 M a year earlier) |

Company targets break-even in 2026.

## Evidence-quality note

The raw scan's source list was 7-of-10 search-engine query URLs (`lite.duckduckgo.com/lite/?q=...`), which are not sources, plus an aggregator (sohu). Only two entries were primary. The facts above have been re-grounded on MetaX's own newsroom and product pages plus named Chinese tech/financial press; the source list below reflects that re-grounding.

## Sources — added 2026-08-08

**Primary (vendor)**
- [MetaX Newsroom — WAIC 2026 item (S600 + X300)](https://www.metax-tech.com/en/ndetail/12629.html)
- [MetaX Newsroom — MiniMax H3 Day-0 adaptation (2026-08-03)](https://www.metax-tech.com/ndetail/12632.html)
- [MetaX Newsroom index](https://www.metax-tech.com/en/news.html?cid=15)
- [MetaX product catalog — all series](https://www.metax-tech.com/en/goods/prod.html?cid=3)
- [MetaX N300 product page](https://www.metax-tech.com/en/goods/prod.html?cid=106&id=66)
- [MetaX N300 Server product page](https://www.metax-tech.com/en/goods/prod.html?cid=110&id=69)
- [MetaX C550 3D Mesh Supernode](https://www.metax-tech.com/en/goods/prod.html?cid=112&id=55)

**Independent press / filings**
- [ITHome — WAIC 2026 MetaX coverage](https://www.ithome.com/0/978/460.htm)
- [Sina Finance — WAIC 2026 (2026-07-20)](https://finance.sina.com.cn/tech/shenji/2026-07-20/doc-iniimuyk4915570.shtml)
- [Sina Finance — Sun Guoliang on C600 large-scale shipment (2026-07-08)](https://finance.sina.com.cn/wm/2026-07-08/doc-inihceis8322692.shtml)
- [Tencent News — FY2025 results call, C700 status (2026-04-08)](https://news.qq.com/rain/a/20260408A066PB00)
- [TrendForce CN — MetaX shipment note (2026-07-09)](https://www.trendforce.cn/industry-news/semiconductors/20260709-6285.html)
- [EastMoney — FY2025 / Q1 2026 financials](https://finance.eastmoney.com/a/202604293724854763)
- [PEDaily — MetaX feature (2026-07-28)](https://news.pedaily.cn/20260728/135041.shtml)
- [ZOL — MetaX product listing](https://ai.zol.com.cn/1212/12124334.html)
