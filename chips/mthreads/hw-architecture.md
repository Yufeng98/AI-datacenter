# Moore Threads MTT S5000 / PingHu and MTT S4000 / Chunxiao Hardware Architecture Reference

*as_of: 2026-08-08*
*chip: mthreads*
*device_class: GPU (China, 摩尔线程)*

---

## Chip Identity

| Attribute | **MTT S5000** | MTT S4000 | MTT S3000 | MTT S80 |
|-----------|---------------|-----------|-----------|---------|
| Architecture brand | **MUSA PingHu (平湖) — Gen 4; PH100 die** | MUSA Chunxiao (春晓) | MUSA Chunxiao (1/2 die) | MUSA Chunxiao |
| Purpose | **AI Training + Inference + FP64 HPC** | AI Training + Inference | AI Inference | Gaming |
| Transistors | **not disclosed** (a circulating "22B" figure duplicates Chunxiao's and is unconfirmed) | 22 billion | ~11 billion (est.) | 22 billion |
| Process | **not disclosed** (a circulating "7nm" figure is unconfirmed) | TSMC 12nm | TSMC 12nm | TSMC 12nm |
| TDP | **not disclosed** | 450W | 250W | ~250W |
| Form factor | **OAM module (liquid- and air-cooled variants)** | PCIe FHFL | PCIe FHFL | PCIe consumer |
| Host interface | **not disclosed** | PCIe Gen5 x16 | PCIe Gen4/5 | PCIe Gen4 |
| Status | **Shipping — accelerated mass production 2026** | GA 2024 | GA | GA |

---

## Generation Overview

Architectures are named after the Ten Scenes of West Lake. Corrected map, verified 2026-08-08 — earlier revisions of this file wrongly labelled Huagang / Flower Harbor as "Gen 3" and omitted PingHu entirely.

| Gen | Architecture | Die / chip | Datacenter products | Process | Memory | Data types | Scale-up | Status |
|-----|-------------|-----------|--------------------|---------|--------|-----------|----------|--------|
| 1 | 苏堤 Sudi (Sudijia 苏堤甲) | — | MTT S1000, MTT S2000 (also S10/S30/S50, S60) | TSMC 12nm | GDDR6 | FP32/FP16 (no tensor cores) | PCIe only | Legacy |
| 2 | 春晓 Chunxiao | Chunxiao | MTT S3000, MTT S4000 (also S70/S80) | TSMC 12nm | GDDR6 48 GB / 768 GB/s (S4000) | FP32/TF32/BF16/FP16/INT8; 2× FP64 units/MP only | MTLink 1.0, 240 GB/s/GPU, 8 GPUs/server | GA 2024 |
| 3 | 曲院 Quyuan | not disclosed | no datacenter part tracked here | not disclosed | not disclosed | not disclosed | not disclosed | — |
| **4** | **平湖 PingHu** | **PH100** | **MTT S5000** | **not disclosed** | **media-reported 80 GB @ 1.6 TB/s (type implied HBM, generation not confirmed); vendor publishes no spec table** | **vendor-confirmed full-precision FP8 → FP64** | **media-reported 784 GB/s card-to-card; MTLink generation not disclosed; MTT MGX / SGX5000 8-GPU nodes; MTT C256 supernode announced** | **Shipping 2026** |
| 5 | 花港 Huagang ("Flower Harbor") | 华山 Huashan (AI), 庐山 Lushan (graphics) | Huashan AI GPU | Advanced node (undisclosed) | 8× HBM sites, dual chiplet | + FP4, MTFP6, MTFP4 (announced) | not disclosed | Announced 2025-12-20 (MDC 2025), 2026 mass production |

---

## Compute Engine

```
MTT S4000 (Chunxiao full die)
├── 4,096 SP (Stream Processors) arranged in MPs
│   └── Each MP (MUSA Processor) contains:
│       ├── 128× FP32 units
│       ├── 32× INT8/bitwise units
│       ├── 32× SFU (Special Function Units: trig, exp, log)
│       ├── 2× FP64 units
│       ├── TCE (Tensor Compute Engine — matrix/tensor acceleration)
│       └── 28 KB local shared memory (scratchpad)
├── 128 Tensor Cores (TCE units across all MPs)
├── 256 Texture Units
├── 256 Render Output Units (ROPs)
└── L2 Cache: 4 MB (hardware-managed)
```

### Peak Performance

| Metric | **MTT S5000 (PingHu / PH100)** | MTT S4000 | MTT S3000 | Notes |
|--------|-------------------------------|-----------|-----------|-------|
| FP8 (dense) | **up to 1,000 TFLOPS** — *media-reported only* | — | — | Vendor publishes no spec table; figure from Baidu Baike / Chinese tech media |
| FP16 / BF16 | **not disclosed** | 200 TFLOPS | — | via TCE tensor cores on Chunxiao |
| INT8 | **not disclosed** | 200 TOPS | — | via TCE tensor cores on Chunxiao |
| TF32 | **not disclosed** | 50 TFLOPS | — | |
| FP32 | **not disclosed** | 25 TFLOPS | 15.6 TFLOPS | |
| FP64 | **supported (vendor-confirmed as part of "full-precision FP8 to FP64"); throughput not disclosed** | minimal (~0.4 TFLOPS est.) | — | 2 FP64 units per MP on Chunxiao |

No FP4 or FP8 on Chunxiao. **PingHu (Gen 4) adds vendor-confirmed FP8→FP64**; FP4 / MTFP6 / MTFP4 are announced for **Huagang (Gen 5)**, not for PingHu.

### MTT S5000 compute engine — what is and is not known

The vendor discloses only the architecture family and the data-type range. **No MP/SP count, tensor-core count, clock, die configuration, packaging or L2 capacity is published for PH100.** The one architecturally load-bearing fact is the promotion of **FP64 to a first-class data type**, which the software side corroborates: Moore Threads created sixteen MUSA ports of FP64-heavy scientific codes (CP2K, LAMMPS, RELION, SU2, MAGMA, AMGX, Eigen, SpFFT, Kokkos, DeePMD-kit, …) between 2026-06-11 and 2026-08-07.

Vendor performance claims for the S5000 (marketing, not independently confirmed): single-GPU prefill ≥ 4,000 tokens/s and decode ≥ 1,000 tokens/s; prefill 2.5× "leading international flagship products" at 16K sequence length; Llama3-70B MFU > 60%; DeepSeek-236B MFU > 40%; 95% cluster linearity at 10K-GPU scale; 0.6% relative precision deviation for DeepSeek-236B on 10K GPUs. Separately, Chinese media (2026-02-13) report a thousand-card S5000 cluster whose training loss differed from an H100 cluster by 0.62% on BAAI RoboBrain 2.5 — media-reported, not reproduced, and a *different* claim from the vendor's 0.6% figure.

---

## Memory Hierarchy

```
Per-MP: 28 KB local shared memory (programmer-managed)
         ↓
On-chip: 4 MB L2 cache (hardware-managed, all MPs share)
  Comparison: NVIDIA A100 ~40 MB, H100 ~50 MB, Biren BR100 300 MB
         ↓
Off-chip: GDDR6
  S4000: 48 GB total, 384-bit interface, 768 GB/s bandwidth, 16 Gbps/pin
  S3000: 32 GB total, 256-bit interface, ~512 GB/s bandwidth
  S80:   16 GB total, 256-bit interface, ~512 GB/s bandwidth
```

### MTT S5000 (PingHu) memory — media-reported tier

| Level | MTT S5000 | Evidence |
|-------|-----------|----------|
| Per-MP shared memory | **not disclosed** | — |
| L2 / on-chip SRAM | **not disclosed** | — |
| Off-chip capacity | **80 GB** | Media-reported only (Baidu Baike, EET-China, Chinese tech media). Vendor page has no spec table |
| Off-chip type | **implied HBM; HBM generation not confirmed** | Never stated by the vendor |
| Off-chip bandwidth | **1.6 TB/s** | Media-reported only |

> ⚠️ **The "3.35 TB/s" and "4.0 TB/s" figures on the official S5000 page are NOT the S5000's.** They are competitor comparison-baseline footnotes: "reference benchmark 1: dense FP16 989 TFLOPS, memory bandwidth 3.35 TB/s" (NVIDIA H100 SXM) and a second baseline "dense FP16 148 TFLOPS, memory bandwidth 4.0 TB/s" (H20-class). Any scrape of that page must not attribute either to the S5000.

If the 80 GB / 1.6 TB/s figures hold, PingHu is the first Moore Threads datacenter part to leave GDDR6 behind — a 1.7× capacity and 2.1× bandwidth step over the S4000, and the change that makes the 4 MB-L2-era bandwidth criticism of Chunxiao obsolete for the shipping flagship.

---

## Host / Packaging / Interconnect

| Layer | **MTT S5000** | MTT S4000 | MTT S3000 |
|-------|---------------|-----------|-----------|
| Host PCIe | **not disclosed** | Gen 5 x16 | Gen 4/5 x16 |
| CXL | Not confirmed | Not confirmed | Not confirmed |
| Packaging | **not disclosed** | TSMC 12nm monolithic | TSMC 12nm monolithic |
| Scale-up | **MTLink (generation number not disclosed)** | MTLink 1.0 | Not confirmed |
| Card-to-card BW/GPU | **784 GB/s — media-reported only** | 240 GB/s | — |
| Max GPUs/server | **8 (MTT MGX 8-OAM platform / MTT SGX5000 server)** | 8 (KUAE D800) | 4–8 |
| Announced scale-up domain | **128 GPUs/cabinet, 256 across two cabinets (MTT C256, announced only)** | 8 | — |
| Scale-out | Ethernet/RoCE via host | Ethernet/RoCE via host | Ethernet/RoCE via host |

---

## MTLink Interconnect

**MTLink 1.0** is Moore Threads' proprietary GPU-to-GPU interconnect (NVLink analog):
- 240 GB/s aggregate bandwidth per GPU in an 8-card KUAE D800 server
- Enables ring/tree collective operations via MCCL (NCCL analog)
- Fabric scales to **10,000 GPUs in a single cluster** (demonstrated 2024)
- No MTLink switch ASIC disclosed (direct point-to-point like BLink)

### MTLink on the S5000 generation (2026)

- Card-to-card bandwidth **784 GB/s** — media-reported only; the vendor publishes no interconnect figure
- **MTLink generation number for the S5000 is not disclosed.** Do not assume "MTLink 2.0"
- Node building blocks are the **MTT MGX** modular platform (eight S5000 OAM modules interconnected over MTLink) and the **MTT SGX5000** server (8× S5000). These supersede the S4000-era KUAE D800 / MCCX D800 naming for the new generation
- No MTLink switch ASIC has been disclosed for this generation either

---

## MTT C256 Supernode (announced / first publicly demonstrated, WAIC 2026)

First shown **2026-07-17** at WAIC 2026 (coverage 07-17 → 07-20), confirmed by ITHome, Sina Finance / Fast Technology and Global Times. **Status: announced and publicly demonstrated (首次公开展示) — not sampling, shipping or customer-deployed.**

| Property | Claim |
|----------|-------|
| Topology | Industry-first **single-layer Scale-up network** (首创单层Scale-up网络), breaking the prevailing 64-card single-layer limit |
| Single cabinet | **128 GPUs fully interconnected** in one standard cabinet |
| Two cabinets | **256 cards** via cabinet-adjacent extension (并柜扩展) |
| Card-to-card latency | **Sub-microsecond** |
| Node form factor | **2U**, retaining compatibility with existing datacenter hardware/software |
| Positioning | Building block for 10,000- to 100,000-card clusters |
| Per-link MTLink bandwidth | **not disclosed** |
| MTLink generation | **not disclosed** |
| Aggregate supernode bandwidth | **not disclosed** |

Claims of high-efficiency liquid cooling, three-way blind-mate connectors and comprehensive RAS circulate in secondary write-ups but were **not confirmed** in the primary coverage reviewed (ITHome explicitly does not mention them) — they are omitted here rather than recorded.

Architecturally this is the significant interconnect change of the generation: if it ships as announced, the scale-up domain grows from 8 GPUs to 128–256, a claimed ~32× increase, moving Moore Threads from a "server is the scale-up domain" design into the rack-scale supernode category alongside NVL72-class systems (at a much lower per-link bandwidth tier, insofar as 784 GB/s card-to-card holds).

---

## Architecture Generations

### Gen 1: Sudijia (苏堤甲), 12nm
- 2,048–4,096 SP depending on configuration
- No dedicated Tensor Cores
- Products: MTT S60 (desktop), MTT S2000 (datacenter, 12 TFLOPS FP32, 32 GB)
- PCIe Gen 4

### Gen 2: Chunxiao (春晓), 12nm
- 4,096 SP + 128 TCE Tensor Cores
- Native FP16/BF16/INT8 acceleration
- PCIe Gen 5
- MTLink 1.0 (S4000 only)
- Products: MTT S70, MTT S80, MTT S2000, MTT S3000, MTT S4000

### Gen 3: Quyuan (曲院)
- Named generation in Moore Threads' West Lake sequence, positioned between Chunxiao and PingHu
- No datacenter product is tracked in this survey for this generation; die details **not disclosed**

### Gen 4: PingHu (平湖) — PH100 → MTT S5000, shipping
- **The generation that actually shipped as the current flagship**, and the one earlier revisions of this file omitted
- Vendor-confirmed: "full-precision compute support from FP8 to FP64"; "fourth-generation MUSA full-stack platform"
- Media-reported (not vendor-published): 80 GB memory @ 1.6 TB/s, 784 GB/s card-to-card, up to 1,000 TFLOPS dense FP8
- Process node, transistor count, TDP, die/MP configuration, memory type/HBM generation: **not disclosed**
- Form factors: OAM (liquid- and air-cooled), MTT MGX 8-OAM platform, MTT SGX5000 server
- Announced supernode: MTT C256 (128 cards/cabinet, 256 across two cabinets)
- Products: MTT S5000

### Gen 5 (roadmap): Huagang (花港, "Flower Harbor"), 2026
- Announced 2025-12-20 at MDC 2025 for 2026 mass production — **one generation beyond the shipping S5000**
- Dual chiplet architecture
- 8× HBM memory stacks
- New compute formats: FP4, MTFP6 (custom), MTFP4 (custom), alongside FP64
- 50% higher compute density vs Chunxiao
- 10% better power efficiency
- 2nd-gen hardware ray tracing
- DirectX 12 Ultimate
- AI Generative Rendering (AGR) block
- Target: between NVIDIA Hopper and Blackwell (company claim)
- Products: Huashan 华山 (AI), Lushan 庐山 (graphics; 2026 status unverified)

> **Correction (2026-08-08).** Prior revisions labelled Chunxiao "Gen 2/3" and Flower Harbor "Gen 3" in the summary and "Gen 4" in the figure. The verified West Lake sequence is Sudi = 1, Chunxiao = 2, Quyuan = 3, PingHu = 4, Huagang = 5. Huashan (Gen 5 Huagang) and the S5000 (Gen 4 PingHu) are different chips one generation apart — earlier text that treated 平湖/PH100 and 华山/花港 as competing names for one product was wrong.

---

## Export Control Context

Moore Threads operates under US export controls that restrict access to:
- Advanced EDA tools (Synopsys, Cadence products)
- Advanced foundry nodes at TSMC (<7nm for Chinese AI chip companies)
- US-origin chip design IP

This is why Chunxiao uses TSMC 12nm (permitted tier) rather than more advanced nodes. The **PingHu (PH100 / S5000) process node is not disclosed**, and neither is the forthcoming Huagang / Flower Harbor node. A "7nm" figure circulating for the S5000 comes from low-quality sources that also reproduce Chunxiao's 22-billion transistor count verbatim, and is treated here as unconfirmed.

---

## Comparison: MTT S4000 vs Biren BR100 vs NVIDIA A100

| Metric | MTT S4000 | Biren BR100 | NVIDIA A100 |
|--------|-----------|-------------|-------------|
| Process | TSMC 12nm | TSMC 7nm CoWoS-S | TSMC 7nm |
| Transistors | 22B | 77B (dual-die) | 54B |
| FP32 | 25 TFLOPS | 256 TFLOPS | 312 TFLOPS |
| BF16 | 200 TFLOPS | 1,024 TFLOPS | 312 TFLOPS |
| INT8 | 200 TOPS | 2,048 TOPS | 624 TOPS |
| Memory | 48 GB GDDR6 | 64 GB HBM2e | 80 GB HBM2e |
| Mem BW | 768 GB/s | 2,300 GB/s | 2,000 GB/s |
| L2 Cache | 4 MB | 300 MB | ~40 MB |
| TDP | 450W | 550W | 400W |
| Scale-up | MTLink (240 GB/s) | BLink (512 GB/s) | NVLink 3 (600 GB/s) |

Note: MTT S4000 targets the mid-tier AI market; Biren BR100 targeted A100-competitive performance (blocked by export controls). MTT S4000 is primarily used for Chinese domestic cloud AI inference and moderate-scale training.

**Where the S5000 sits (2026).** Using only the media-reported tier — 80 GB @ 1.6 TB/s, 784 GB/s card-to-card, ~1,000 TFLOPS dense FP8 — the S5000 lands roughly at H100-SXM memory capacity with about half its HBM bandwidth and comparable dense-FP8 order of magnitude, while adding FP64 that neither Chunxiao nor Biren BR100 offered. Because none of these numbers is vendor-published, this positioning is indicative only and must not be reported as a confirmed comparison.

---

## Sources

### S5000 / PingHu / C256 (added 2026-08-08)

- [MTT S5000 Official Product Page (EN)](https://en.mthreads.com/product/S5000) — PH100 chip, "PingHu" architecture, "fourth-generation MUSA full-stack platform", FP8→FP64, OAM / MGX / SGX5000 form factors, all tokens-per-second / MFU / cluster-linearity marketing claims. **Contains no specification table.**
- [MTT S5000 Official Product Page (CN)](https://www.mthreads.com/product/S5000) — confirms 3.35 TB/s and 4.0 TB/s are competitor comparison-baseline footnotes, not S5000 specs
- [ITHome — MTT C256 supernode, WAIC 2026 (2026-07-20)](https://www.ithome.com/0/978/856.htm) — 128 GPUs/cabinet, 256 across two cabinets, sub-microsecond latency, 2U nodes, 64-card single-layer limit broken, 10K–100K-card scaling; also the S5000 thousand-card BAAI RoboBrain 2.5 cluster
- [Sina Finance — Moore Threads at WAIC 2026 (2026-07-19)](https://finance.sina.com.cn/tech/roll/2026-07-19/doc-iniiiwhz3402415.shtml) — 首创单层Scale-up网络; three AI factories (model / token / agent)
- [Tencent Cloud news (2026-02-13)](https://cloud.tencent.com/developer/news/3588420) — S5000 1,000 TFLOPS FP8, 80 GB, FP8→FP64, 10K-card clusters, the 0.62% loss-vs-H100 comparison
- [Baidu Baike — MTT S5000](https://baike.baidu.com/item/MTT%20S5000/67553280) — snippet-level only (HTTP 403 on direct fetch); origin of the 1,000 TFLOPS / 80 GB / 1.6 TB/s / 784 GB/s figure set

### S4000 / Chunxiao and earlier

- [MTT S4000 Official Product Page (EN)](https://en.mthreads.com/product/S4000)
- [VideoCardz — MTT S4000 48GB MTLink](https://videocardz.com/newz/moore-threads-introduces-mtt-s4000-48gb-ai-gpu-with-mtlink-and-zero-cost-nvidia-cuda-framework-translation)
- [WCCFTech — MTT S4000 Specifications](https://wccftech.com/moore-threads-mtt-s4000-gpu-48-gb-memory-200-tops-ai-gen5-ready/)
- [Tom's Hardware — MTT S4000 LLM Training](https://www.tomshardware.com/pc-components/gpus/china-made-moore-threads-ai-gpus-used-for-three-billion-parameter-llm-training-mtt-s4000-appears-competitive-against-unspecified-nvidia-solutions)
- [Tom's Hardware — Chunxiao die details](https://www.tomshardware.com/news/moore-threads-unveils-chunxiao-gpu)
- [MUSA Hardware Architecture Blog](https://blog.mthreads.com/blog/musa/2024-05-11-MUSA%E7%A1%AC%E4%BB%B6%E6%9E%B6%E6%9E%84%E4%B8%8EGPU%E5%B9%B6%E8%A1%8C%E7%A8%8B%E5%BA%8F%E5%9F%BA%E7%A1%80/)
- [TrendForce — MTLink to challenge NVLink](https://www.trendforce.com/news/2024/07/11/news-chinas-moore-threads-develops-mtlink-to-challenge-nvidias-nvlink/)
- [Tom's Hardware — 10K GPU cluster](https://www.tomshardware.com/pc-components/gpus/chinese-gpu-maker-moore-threads-can-now-scale-to-10000-processors-for-ai-clusters-mtlink-fabric-tech-competes-with-nvidias-nvlink)
- [WCCFTech — Huashan / Lushan on Flower Harbor (Gen 5 Huagang)](https://wccftech.com/moore-threads-lushan-gaming-huashan-ai-gpus-15x-gaming-uplift-50x-rt-boost-dx12-ultimate-support/)
- [TrendForce — Huashan vs Hopper](https://www.trendforce.com/news/2025/12/22/news-chinas-moore-threads-unveils-huashan-ai-chip-reportedly-takes-aim-at-nvidias-hopper/)
