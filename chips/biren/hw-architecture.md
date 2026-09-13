# Biren BR10X / BR20X Hardware Architecture Reference

*as_of: 2026-08-08*
*prior revision: 2026-04-05*
*chip: biren*
*device_class: GPU-like AI Accelerator (China)*
*generations: BR100 / BR104 (2022, never volume) · BR106 (2023) · BR110 (2024) · BR166 (2025) · BR20X (in development)*

---

## Generation Overview

All of BR100, BR104, BR106, BR110 and BR166 are the **BR10X first-generation architecture** on TSMC 7nm. 壁砺 (Bili) is the product brand for the same BR series — 壁砺106 = BR106, 壁砺110 = BR110, 壁砺166 = BR166. **BR20X** is the second-generation architecture and is not yet taped out.

| Generation | Year | Dies | Process | Peak compute | Memory | Package | Form factor / power | Status |
|---|---|---|---|---|---|---|---|---|
| BR100 | 2022 | 2 | TSMC 7nm CoWoS-S | 256 TF FP32 / 1,024 TF BF16 / 2,048 TOPS INT8 | HBM2e 64 GB, 2.3 TB/s | CoWoS-S 2.5D | OAM, 550 W | Announced Hot Chips 34; **never volume** (TSMC suspension Oct 2022) |
| BR104 | 2022 | 1 | TSMC 7nm | 128 TF FP32 / ~512 TF BF16 / ~1,024 TOPS INT8 | HBM2e 32 GB, 819 GB/s | Monolithic | PCIe FHFL, 300 W | Announced; more producible variant |
| **BR106 (壁砺106)** | **2023** | **1** | **TSMC 7nm (BR10X)** | **not disclosed** | **not disclosed** | **Monolithic** | **106M: OAM, 400 W peak; 106B: FHFL dual-width PCIe** | **Mass production Jan 2023** (dev from 2020, tape-out 2021) |
| **BR110 (壁砺110)** | **2024** | **1** | **TSMC 7nm (BR10X)** | **not disclosed** | **not disclosed** | **Monolithic** | **not disclosed** (no product page on birentech.com) | **Mass production Oct 2024**; edge-inference positioning |
| **BR166 (壁砺166)** | **2025** | **2 (2× BR106)** | **TSMC 7nm (BR10X)** | **not disclosed** — vendor claim "compute and memory doubled vs prior generation" | **not disclosed** | **2.5D chiplet co-package, die-to-die interconnect** | **166M: OAM air, 550 W · 166L: OAM cold-plate liquid, 600 W · 166C: FHFL 290 mm dual-width PCIe, 300 W** | **Mass production Aug 2025**; shipped at scale 2H2025 |
| **BR20X ("BR2xx")** | **planned 2026** | **chiplet (count not disclosed)** | **not disclosed** | **not disclosed** — "increased compute density" | **not disclosed** — "increased capacity and bandwidth" | **Chiplet** | **not disclosed** | **In development: architecture design complete, in physical design and tape-out verification. NOT taped out, not sampling.** |
| BR30X / BR31X | targeted 2028 | — | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | Roadmap only — BR30X cloud training/inference, BR31X edge inference |

> **No public datasheet exists for BR106 / BR110 / BR166 / BR20X.** birentech.com publishes **form factor and peak power only** for the five listed SKUs. A widely circulated "BF16 800 TFLOPS / 128 GB HBM" figure for the 166 series traces to a single Chinese aggregator (Toutiao) and appears nowhere on Biren's site — it is **not** recorded as a spec here.

---

## Chip Identity

| Attribute | BR100 | BR104 |
|-----------|-------|-------|
| Architecture brand | BiLi (壁立仞) | BiLi |
| Dies | 2 (dual-die) | 1 (monolithic) |
| Transistors | 77 billion | ~38 billion (est.) |
| Die area | 1,074 mm² per die | ~1,074 mm² |
| Process | TSMC 7nm CoWoS-S | TSMC 7nm |
| TDP | 550W | 300W |
| Form factor | OAM | PCIe FHFL |

*Transistor count, die area and process are published only for the 2022 BR100/BR104 parts. Equivalent figures for BR106 / BR110 / BR166 / BR20X are **not disclosed**.*

---

## Compute Engine

```
BR100 GPU
├── Die 0
│   ├── 32× SPC (Streaming Processing Cluster)
│   │   ├── 16× EU (Execution Unit)
│   │   │   ├── FP32 / TF32+ / BF16 / INT8 pipelines
│   │   │   ├── Tensor units (matrix acceleration)
│   │   │   ├── C-Warp warp schedulers
│   │   │   └── EU L1 cache / shared memory
│   │   ├── 4,096 threads per SPC
│   │   └── Mesh connection to L2 cache slices
│   ├── Distributed L2 Cache (150 MB per die → 300 MB total)
│   └── HBM2e Controller (2 stacks × 16 GB = 32 GB per die)
│
└── Die 1 (symmetric to Die 0)

Die-to-die link: 896 GB/s via CoWoS-S silicon interposer
```

### Peak Performance

| Metric | BR100 | BR104 | NVIDIA A100 (ref) |
|--------|-------|-------|-------------------|
| FP32 | 256 TFLOPS | 128 TFLOPS | 312 TFLOPS |
| TF32+ | 512 TFLOPS | 256 TFLOPS | 312 TFLOPS (TF32) |
| BF16 | 1,024 TFLOPS | ~512 TFLOPS | 312 TFLOPS |
| INT8 | 2,048 TOPS | ~1,024 TOPS | 624 TOPS |
| FP64 | **0** | **0** | 19.5 TFLOPS |

### Compute Engine — later generations (added 2026-08-08)

| Generation | Compute organization | Numerics | Peak performance |
|---|---|---|---|
| BR106 / BR110 | SPC + EU, C-Warp SIMT (BR10X, same as BR100 family); SPC and EU counts **not disclosed** | FP32 / TF32+ / BF16 / FP16 / INT8; no FP64 disclosed | **not disclosed** |
| BR166 | Two BR106 dies co-packaged; compute scales with the die count, exact organization **not disclosed** | Same BR10X numerics | **not disclosed** — Biren claims "compute and memory performance doubled versus the prior generation" (vendor claim, unverified) |
| BR20X | Second-generation architecture, chiplet; unit organization **not disclosed** | **Native FP8 and FP4** — the first Biren parts with sub-8-bit numerics | **not disclosed** — "increased compute density" is the only stated delta |

The **FP8/FP4 addition in BR20X is the single most significant compute-engine change** in the Biren line since BR100: the BR10X generation tops out at INT8 for quantized inference and has no low-precision floating-point path at all.

---

## Memory Hierarchy

```
Per EU: L1 cache + configurable shared memory
          ↓
Per Die: 150 MB distributed L2 cache (hardware-managed)
          ↓
Off-chip: HBM2e
  BR100: 64 GB total / 2.3 TB/s bandwidth
  BR104: 32 GB / 819 GB/s
```

### Memory Hierarchy — later generations (added 2026-08-08)

| Generation | On-chip | Off-chip memory | Capacity | Bandwidth |
|---|---|---|---|---|
| BR106 / BR110 | Distributed L2 + EU L1/shared memory (BR10X); capacity **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** |
| BR166 | Two BR106 dies' worth of distributed L2; total **not disclosed** | **not disclosed** | **not disclosed** — the circulating "128 GB HBM" figure is aggregator-only and is **not** adopted here | **not disclosed** |
| BR20X | **not disclosed** | **not disclosed** | **not disclosed** — vendor states "increased memory capacity" | **not disclosed** — vendor states "increased memory bandwidth" |

Biren's product pages for 166M / 166L / 166C / 106M publish **form factor and peak power only**; there is no memory table on any of them. The HBM supply constraint documented for the BR100 era (US export controls on SK Hynix / Samsung HBM) remains the standing structural risk for the line, but Biren has disclosed no memory sourcing detail for the shipping 壁砺 parts.

---

## Six Data-Flow Architecture Features (BiLi)

| # | Name | Description |
|---|------|-------------|
| 1 | TF32+ | Extended dynamic range TF32 for training throughput |
| 2 | TDA | Data-flow Tensor Data Access acceleration |
| 3 | C-Warp | Data-flow concurrent warp execution model |
| 4 | NME | Near-Memory Engine — reduces inter-die/inter-chip data movement |
| 5 | NUMA/UMA | Memory topology optimization for multi-die configuration |
| 6 | SVI | Stream Virtualization Isolation for multi-tenancy |

---

## Host / Packaging / Interconnect

| Layer | BR100 | BR104 |
|-------|-------|-------|
| Host PCIe | Gen5 x16 | Gen4/5 x16 |
| CXL | Yes (v1.1/2.0) | Limited |
| Packaging | CoWoS-S 2.5D | Monolithic |
| Die-to-die | 896 GB/s (interposer) | N/A |
| Scale-up | BLink (8 links × 64 GB/s = 512 GB/s) | BLink (limited) |
| Max GPUs/server | 8 | 4–8 |
| Scale-out | IB/ETH via host NIC | IB/ETH via host NIC |

### Packaging and form factors — 壁砺 generation (added 2026-08-08)

| SKU | Packaging | Form factor | Peak power | Cooling |
|---|---|---|---|---|
| 壁砺™106M | Monolithic BR106 die | OAM module | 400 W | Air |
| 壁砺™106B | Monolithic BR106 die | FHFL dual-width PCIe | not disclosed | Air |
| 壁砺™166M | **2.5D chiplet: 2× BR106 dies + die-to-die interconnect** | 4U OAM V1.1 module | **550 W** | Air |
| 壁砺™166L | 2.5D chiplet: 2× BR106 dies | OAM module | **600 W** | **Cold-plate liquid** |
| 壁砺™166C | 2.5D chiplet: 2× BR106 dies | FHFL (290 mm) dual-width PCIe inference card | **300 W** | Air |

The BR166 chiplet approach is architecturally the same move as BR100 (two dies co-packaged with a high-bandwidth die-to-die link), but reached **volume production** where BR100 did not. Die-to-die bandwidth for BR166 is **not disclosed**; the 896 GB/s figure belongs to BR100's CoWoS-S interposer and must not be carried over. The 166L liquid-cooled SKU is the first Biren part above the 550 W OAM envelope.

---

## Scale-up Fabric Roadmap — BLink 2.0, NPO Optics and the Distributed Decoupled Supernode (announced WAIC 2026, added 2026-08-08)

Announced at **WAIC 2026 (Shanghai, 2026-07-17 to 07-20)**. This is an **architecture/roadmap announcement, not a shipping product** — no availability date, no per-link bandwidth, and no customer were disclosed.

### Three-tier supernode matrix

| Tier | Scale | Interconnect | Physical organization |
|---|---|---|---|
| Standard server | 16 cards | Electrical | Conventional server chassis |
| High-density cabinet | 128 cards | Electrical | Cabinet-scale — the prior electrical scale-up ceiling |
| **Distributed decoupled supernode** | **1,024 cards** | **NPO optical (BLink 2.0)** | **GPU nodes and switch nodes physically separated, optically linked; GPU nodes remain in standard server form factors** |

### BLink 2.0 — stated capabilities

| Capability | Description |
|---|---|
| Memory-semantic interconnect | Up to **1,024 GPUs sharing a single memory space** |
| In-network computing | Collectives offloaded into the switch nodes (implies a BLink switch ASIC — a departure from BR100's switchless point-to-point BLink 1.x) |
| Intelligent congestion control | Fabric-level congestion management |
| Multi-layer link self-healing | Recovery spanning the physical layer through the framework layer |

**Per-link and aggregate BLink 2.0 bandwidth: not disclosed.** Biren's stated *motivation* is a per-GPU scale-up bandwidth requirement **exceeding 1 TB/s**; that is a requirement statement, not a product spec. (A "224 Gbps port rate" attributed to this announcement elsewhere does **not** appear in Biren's release.)

### NPO — near-packaged optics

Biren's rationale, verbatim from its release: copper signalling *"attenuates severely within 3 metres."* NPO fuses the optical engine into the GPU module, **removes the high-power DSP chip** (retimer), and extends optical reach to *"several hundred metres."* This physical decoupling is what allows the switch nodes to leave the GPU cabinet.

### Prior-generation dOCS deployment (qualified)

Biren's FY2025 report states it delivered multiple thousand-card clusters *"including a 2,048-card optical-interconnect / optical-switching GPU supernode cluster."* Independent WAIC coverage attributes this to the **previous-generation dOCS** (distributed optical circuit switching) supernode — a **32-card / 4-chassis building block**, aggregated into a 2,048-card cluster at a national-level computing platform reported as Shanghai INESA (上海仪电).

**This is not a 2,048-card scale-up / shared-memory domain and it is not BLink 2.0.** Biren's own release describes the national-platform dOCS deployment without stating a card count; the 2,048 figure comes from the annual report and media.

---

## Export Control and Production Timeline

| Date | Event |
|------|-------|
| Aug 2022 | BR100 announced at Hot Chips 34 |
| Oct 2022 | US export controls; TSMC suspends advanced node for Biren — BR100 never reaches volume |
| Jan 2023 | **BR106 (壁砺106) enters mass production** — first commercially shipping Biren silicon |
| Oct 2024 | **BR110 (壁砺110) enters mass production** — edge-inference positioning |
| Aug 2025 | **BR166 (壁砺166) enters mass production**; ships at scale 2H2025 |
| Dec 17, 2025 | HKEX listing hearing cleared; prospectus cites cumulative BR106+BR110 sales "exceeding 12,000 units" |
| Jan 2026 | Hong Kong IPO; HK$5.58B raised; +76% on debut |
| Mar 30, 2026 | FY2025 results: revenue RMB 1.0346 B (+207.2% YoY), gross margin 53.8%, funding reserve > RMB 8.5 B |
| Jul 17–20, 2026 | **WAIC 2026: NPO optical interconnect, BLink 2.0, 16/128/1,024-card supernode tiers, BR2xx FP8/FP4** |
| Planned 2026 | BR20X commercial launch (stated plan; chip in physical design / tape-out verification as of the FY2025 disclosure) |
| Targeted 2028 | BR30X (cloud training/inference) and BR31X (edge inference) commercialization |

---

## Not Disclosed / Not Verified (as of 2026-08-08)

- FLOPS, HBM type / capacity / bandwidth, and process node for **BR106, BR110, BR166 and BR20X**
- Per-link and aggregate **BLink 2.0** bandwidth; BLink bandwidth for the shipping 壁砺 generation
- **BR166 die-to-die bandwidth** and packaging vendor
- BR20X die count, chiplet composition, and tape-out date
- No Biren paper found at **Hot Chips 2026 / ISCA 2026 / ISSCC 2026** — for Hot Chips 38 (2026-08-23 to 08-25) the program page returned HTTP 403, so this is *"could not verify"*, not *"absent"*. In any case Hot Chips 38 is still in the future and no slides or abstracts exist.
- No **MLPerf** submission found for any Biren part

---

## Sources

- [Hot Chips 34 Slide Deck (official)](https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf)
- [Chips and Cheese analysis](https://chipsandcheese.com/p/hot-chips-34-birens-br100-a-machine-learning-gpu-from-china)
- [ServeTheHome BR100 review](https://www.servethehome.com/biren-br100-gpu-for-datacenter-compute-and-ai-workloads/)
- [VideoCardz specifications](https://videocardz.com/newz/chinas-biren-br100-is-7nm-hpc-gpu-with-77b-transistors-and-64gb-hbm2e-memory)
- [Zhihu deep dive (Chinese)](https://zhuanlan.zhihu.com/p/551888300)

### Added 2026-08-08
- [birentech.com — hardware product menu (壁砺™166L / 166M / 166C / 106M / 106B)](https://www.birentech.com/)
- [壁砺™166M — 4U OAM V1.1 air-cooled, 550 W peak](https://www.birentech.com/product/hardware/166m/)
- [壁砺™166L — cold-plate liquid-cooled OAM, 600 W peak](https://www.birentech.com/product/hardware/166l/)
- [壁砺™166C — FHFL 290 mm dual-width PCIe inference card, 300 W peak](https://www.birentech.com/product/hardware/166c/)
- [壁砺™106M — air-cooled OAM, 400 W peak](https://www.birentech.com/product/hardware/106m/)
- [Biren WAIC 2026 press release — NPO, BLink 2.0, three-tier supernode, BR2xx FP8/FP4](https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/)
- [Jiemian — WAIC 2026 BLink 2.0 1,024-GPU shared memory space](https://www.jiemian.com/article/14787272.html)
- [NetEase 163.com — WAIC 2026; prior-gen dOCS supernode at national platform "reaching 2,048 cards"](https://www.163.com/tech/article/L22F1NH700098IEO.html)
- [IT之家 via Sohu — WAIC 2026 three tiers and BR2xx chiplet FP8/FP4](https://www.sohu.com/a/1052133469_114760)
- [Sohu — FY2025 annual report coverage (2026-03-30): 2,048-card optical-switching supernode cluster; BR20X planned 2026](https://www.sohu.com/a/1003843604_313745)
- [Baidu Baike — BR106 (dev 2020, tape-out 2021, mass production Jan 2023, 7nm BR10X)](https://baike.baidu.com/item/BR106/67163892)
- [Baidu Baike — 壁砺166系列 (co-packages two 壁砺106 dies via chiplet + die-to-die interconnect)](https://baike.baidu.com/item/%E5%A3%81%E7%A0%BA166%E7%B3%BB%E5%88%97/67163995)
