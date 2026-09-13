# Iluvatar CoreX (天数智芯) TianGai GPU Hardware Architecture Reference

*as_of: 2026-08-08*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*

---

## Chip Identity

| Attribute | TianGai-100 (BI-V100) | TianGai-150 (BI-V150) | Zhikai 100 (智铠100 / MR-V100) | **TianGai 300 (天垓300)** |
|-----------|----------------------|----------------------|-----------------|-----------------|
| Architecture brand | Big Island Gen1 | Big Island Gen2 | Zhikai inference | **New-generation self-developed architecture (unnamed publicly)** |
| Purpose | AI Training | AI Training | AI Inference | **AI training + inference (general-purpose GPU flagship)** |
| Process | TSMC 7nm | TSMC 7nm | Undisclosed | **Not disclosed** |
| Packaging | 2.5D CoWoS | 2.5D CoWoS | Standard PCIe | **Not disclosed** |
| Memory | 32 GB HBM2 | 64 GB HBM2 | 32 GB GDDR6 (est.) | **Not disclosed (type, capacity and bandwidth all undisclosed)** |
| Host interface | PCIe Gen4 x16 | PCIe Gen4 x16 FHFL | PCIe x16 | **Not disclosed** |
| Data types | FP32/FP16/BF16/INT8 | FP32/FP16/BF16/INT8 | FP16/INT8 | **+ FP8 and FP4 (first Iluvatar part with either)** |
| Released / announced | Jan 2021 (GA) | ~2022 (GA) | ~2022 (GA) | **Announced 2026-07-19 (WAIC 2026); not confirmed shipping** |

> **TianGai 300 disclosure boundary (as of 2026-08-08).** The vendor lists 天垓300 in its product navigation (`cpjs-yj-xlxl-tg300`) but publishes **no specification sheet**, and its news feed has not been updated since 2021. Process node, foundry, die size, transistor count, packaging, memory type/capacity/bandwidth, peak FLOPS/TOPS at every dtype, scale-up link bandwidth, host interface generation, TDP, form factor and price are **all not disclosed** by the vendor or by any press outlet. They are written as "not disclosed" throughout this document and must not be estimated.

---

## Compute Engine

```
TianGai GPU (SIMT GPGPU, conceptual)
├── Multiple Shader Clusters (exact count not disclosed)
│   └── Each cluster contains:
│       ├── SIMT execution pipelines (warp-based)
│       ├── Tensor Compute Units (FP16/BF16/INT8 matrix acceleration)
│       ├── FP32 / FP16 / BF16 ALUs
│       └── Local shared memory (L1 scratchpad)
├── On-chip L2 cache (HW-managed; capacity undisclosed)
├── HBM2 memory controllers (multi-stack, training series)
├── GDDR6 memory controller (inference series)
└── CLIF (Cluster Interconnect Fabric) — proprietary multi-GPU links
```

### TianGai 300 compute engine (announced 2026-07-19) — what is and is not known

```
TianGai 300 (new-generation architecture, conceptual)
├── SIMT general-purpose compute model RETAINED
│   ├── Scalar execution path
│   ├── Vector execution path
│   └── Tensor execution path
├── Reduced precision: FP8 and FP4 (new for Iluvatar)
├── Instruction classes: MMA / DPX / FMA
├── ISA extensions (named, function stated, encoding undisclosed):
│   ├── ixSMEX  — raises data reuse, cuts redundant memory traffic
│   ├── ixDPX   — compresses a 6-instruction dynamic-programming sequence to 1
│   └── ixTrans — lossless matrix transpose; fewer bank conflicts, less VRAM overhead
├── Compute-unit count ........ not disclosed
├── Tensor-unit dimensions .... not disclosed
├── Clock ..................... not disclosed
├── On-chip SRAM / L2 ......... not disclosed
└── Memory subsystem .......... not disclosed (type, capacity, bandwidth)
```

The three ISA extensions are the **only** concrete microarchitectural detail published. ixDPX is notable as an explicit dynamic-programming accelerator in a general-purpose GPU ISA (NVIDIA's DPX instructions in Hopper are the nearest analogue); ixTrans addresses transpose bank conflicts in hardware rather than in the kernel library.

### Peak Performance

| Metric | TianGai-100 | TianGai-150 | **TianGai 300** | NVIDIA A100 (for ref.) |
|--------|------------|------------|------------|----------------------|
| Process | TSMC 7nm | TSMC 7nm | **Not disclosed** | TSMC 7nm |
| FP32 | ~40 TFLOPS | ~50 TFLOPS | **Not disclosed** | 312 TFLOPS |
| FP16 / BF16 | ~80–100 TFLOPS | ~120–150 TFLOPS | **Not disclosed** | 312 TFLOPS |
| FP8 | — | — | **Supported; peak not disclosed** | — |
| FP4 | — | — | **Supported; peak not disclosed** | — |
| INT8 | ~160–200 TOPS | ~240–300 TOPS | **Not disclosed** | 624 TOPS |
| TPP density (co. metric) | 2,352 | 3,040 | **Not disclosed** | — |
| Memory | 32 GB HBM2 | 64 GB HBM2 | **Not disclosed** | 80 GB HBM2e |
| Memory BW | ~1.2 TB/s | ~1.2–1.6 TB/s | **Not disclosed** | 2.0 TB/s |
| TDP | ~350W | ~400W | **Not disclosed** | 400W |

Note: TianGai-100/150 FP performance figures are estimates inferred from company TPP density metrics and architecture assumptions; no independent benchmark results have been publicly disclosed. For **TianGai 300 the vendor has released no absolute performance figure of any kind** — only relative percentages against an unspecified Hopper solution (see below). Any absolute TFLOPS/TOPS number for TianGai 300 circulating elsewhere is aggregator-invented.

### TianGai 300 vendor performance claims (marketing, unverified)

All claims are first-party, relative-only, and benchmarked against **NVIDIA Hopper** — *not* Blackwell. No MLPerf submission and no third-party benchmark exists for any Iluvatar part.

| Claim | Vendor figure | Basis |
|---|---|---|
| Attention efficiency | >90% across precisions; ~10% above a Hopper solution at 64k context | Company briefing, WAIC 2026 |
| MoE compute efficiency | >70%; ~10% higher average MoE throughput | DeepSeek-V4-class workload, company-run |
| Time-to-first-token | ~20% lower | Qwen / GLM / Kimi / DeepSeek |
| Decode efficiency | ~10% higher | Company-run |
| Inter-card communication latency | ~13% lower average | Company-run |
| Energy efficiency | "above mainstream Hopper solutions" | No figure given |

Reported workload-level optimizations named in coverage: attention/FFN (AF) separation and prefill/decode (PD) separation. Implementation details not disclosed.

---

## Memory Hierarchy

```
Per-CU: Local shared memory / L1 scratchpad
         (capacity undisclosed — typical GPGPU range 32–128 KB/CU)
         ↓
On-chip: L2 cache (hardware-managed)
         (capacity undisclosed — likely 8–32 MB range for 7nm GPU)
         ↓
Off-chip: HBM2 (training series)
  TianGai-100: 32 GB total, ~1.2 TB/s
  TianGai-150: 64 GB total, ~1.2–1.6 TB/s
  Zhikai 100 (智铠100 / MR-V100): 32 GB GDDR6, ~512 GB/s (est.)

TianGai 300: memory type, capacity and bandwidth ALL NOT DISCLOSED.
  Neither the vendor nor any press outlet states HBM generation,
  capacity, stack count, or bandwidth. Do not estimate.
  On-chip SRAM / L1 scratchpad / L2 capacity: not disclosed.
```

---

## Host Interface and Scale-up

| Layer | TianGai-150 (BI-V150) | **TianGai 300** | Notes |
|-------|----------------------|------------|-------|
| Host PCIe | Gen 4 x16 FHFL | **Not disclosed** | ~64 GB/s bidir on BI-V150 |
| Packaging | 2.5D CoWoS (TSMC) | **Not disclosed** | CoWoS enables HBM2 stacking on Big Island |
| Scale-up fabric | CLIF (proprietary) | **Not disclosed whether CLIF or a new fabric** | Bandwidth undisclosed in both cases |
| Max accelerators per scale-up domain | 8 cards (single server) | **144 chips (天数超节点 "Tianshu supernode", claimed)** | Supernode at trial / partnership stage |
| Scale-out | Ethernet / InfiniBand | **Not disclosed** | Via host NIC on Big Island |

---

## CLIF — Cluster Interconnect Fabric

**CLIF** is Iluvatar CoreX's proprietary GPU-to-GPU direct interconnect (NVLink/MTLink analog):
- Enables ring/tree topologies for collective operations within a server node
- Bandwidth and topology not publicly disclosed
- Deployed in production 8-GPU server configurations for cloud AI training
- No dedicated switch ASIC mentioned (direct point-to-point assumed)

---

## 天数超节点 — Tianshu Supernode (announced 2026-07-19)

Announced alongside TianGai 300 at WAIC 2026, the **天数超节点 ("Tianshu supernode")** is the first scale-up domain size Iluvatar CoreX has ever stated publicly. It moves the company's scale-up story from a single 8-GPU server to a rack-scale domain, matching the direction taken by NVIDIA (NVL72), Huawei (CloudMatrix) and other Chinese vendors.

| Attribute | Value |
|---|---|
| Claimed scale-up domain | **144 chips, "high-speed full interconnect"** |
| Topology | **Not disclosed** (full mesh vs. switched fabric unstated) |
| Per-link / per-chip bandwidth | **Not disclosed** |
| Switch ASIC | **Not disclosed** (no switch chip named) |
| Single coherent memory domain? | **Not disclosed** |
| Relationship to CLIF | **Not disclosed** — whether the supernode extends the existing CLIF fabric or introduces a new one is unstated |
| Rack power / cooling | **Not disclosed** |
| Status | **Trial / partnership-discussion stage** — explicitly pre-production |

Because neither link bandwidth nor topology is published, the supernode cannot be compared quantitatively to NVL72 or CloudMatrix384. Only the domain size (144) is on the record, and only as a company claim.

---

## Architecture Generations

### Gen 1 — Big Island (TianGai-100, 2021)
- TSMC 7nm, 2.5D CoWoS
- 32 GB HBM2
- First mass-produced Chinese 7nm GPGPU
- TPP density 2,352
- Products: BI-V100

### Gen 2 — Big Island Gen2 (TianGai-150, ~2022)
- TSMC 7nm, 2.5D CoWoS
- 64 GB HBM2 (doubled capacity)
- TPP density 3,040
- Products: BI-V150

### Zhikai (Inference, ~2022)
- Dedicated inference architecture
- Enhanced integer compute units
- Optimized data paths for low-latency inference
- Products: 智铠100 (Zhikai 100), product code MR-V100

### Gen 3 — New-generation self-developed architecture (TianGai 300, announced 2026-07-19)

- **Announced 2026-07-19** at WAIC 2026 (Shanghai, July 17–20); day 3 of the conference
- Company describes it as the **first product on a new-generation self-developed architecture** — a generation change, not a Big Island derivative
- **SIMT retained**, with scalar, vector and tensor execution paths
- **FP8 and FP4** added; **MMA / DPX / FMA** instruction classes
- **ixSMEX / ixDPX / ixTrans** ISA extensions (see Compute Engine above)
- Companion **144-chip 天数超节点** scale-up system (see dedicated section above)
- Process, packaging, memory, FLOPS, TDP, interconnect bandwidth: **all not disclosed**
- Status: **announced**; company states only "已具备规模化应用条件" ("meets the conditions for large-scale deployment"). No tape-out, sampling, mass-production or named-deployment disclosure.
- Products: 天垓300 (no card/board name published)

### Edge / endpoint line — 彤央 (Tongyang) TY series *(survey gap, noted 2026-08-08)*

The vendor site now lists an edge/endpoint product family absent from this survey: **TY1000, TY1100, TY1100-NX, TY1100-NX-PRO, TY1200**. No specifications were retrieved for any of them. Recorded here as a known gap rather than left unmentioned.

### Tianshu / Tianxuan / Tianji / Tianquan (Roadmap codenames)

| Arch | Year | vs NVIDIA | Note |
|------|------|-----------|------|
| Tianshu (天枢) | 2025 | Claims >Hopper | 300+ benchmark clients |
| Tianxuan (天璇) | 2026 | Targets Blackwell | H200-class performance goal — ⚠️ unverified; no shipped or announced product tied to this name |
| Tianji (天玑) | 2026 | Claims to surpass Blackwell | Aggressive target — ⚠️ unverified |
| Tianquan (天权) | 2027 | Targets Rubin | Company stated goal |

Architecture names derived from the Big Dipper (北斗七星) constellation, and sourced from January 2026 press coverage of an investor-facing roadmap.

> **⚠️ Codename mapping unresolved (2026-08-08).** The product actually launched in 2026 is **天垓300**, named from the 天垓 *product* line, and its vendor benchmarks are all against **Hopper**, not Blackwell. **No retrieved source ties TianGai 300 to 天璇 (Tianxuan), 天玑 (Tianji), or any other Big-Dipper architecture codename.** This document keeps the product line and the roadmap codenames as separate namespaces and does not assume TianGai 300 = Tianxuan.

---

## Export Control Context

| Factor | Status |
|--------|--------|
| TSMC 7nm access | Maintained (7nm permitted for non-HPC GPU variants) |
| 2.5D CoWoS packaging | Maintained at TSMC |
| HKEX IPO Jan 2026 | Indicates confidence in continued manufacturing access (ticker 09903.HK) |
| US Entity List | Not listed as of Aug 2026 |
| TianGai 300 foundry | **Not disclosed** — no source states whether it is TSMC or a domestic foundry, so no export-control inference can be drawn |
| Comparison to Biren | Biren blocked post-Oct 2022; Iluvatar continues production |

---

## Sources

### TianGai 300 / WAIC 2026 (added 2026-08-08)

- [Iluvatar CoreX official site](https://www.iluvatar.com/) — product navigation lists 天垓300 (node `cpjs-yj-xlxl-tg300`) in the 天垓 training series; **no spec sheet**. Strongest primary artifact; the vendor news feed has not been updated since 2021, so there is no launch press release.
- [Tencent News, 2026-07-19](https://news.qq.com/rain/a/20260719A08MB800) — WAIC day-3 launch; new self-developed architecture; SIMT; ixSMEX/ixDPX/ixTrans; FP4/FP8 + MMA/DPX/FMA; 144-chip supernode; "已具备规模化应用条件"; 340+ customers as of end-2025
- [Tencent News, second independent piece, 2026-07-19](https://news.qq.com/rain/a/20260719A08J8N00) — ixDPX compresses six instructions to one; Hopper-relative TTFT / attention / MoE / decode claims; confirms no process/memory/TDP disclosure
- [IT之家](https://www.ithome.com/0/978/781.htm) — 2026-07-19 WAIC launch; SIMT general-purpose architecture; Hopper-relative claims
- [Tencent Cloud developer news](https://cloud.tencent.com/developer/news/4280275) — 144-chip full-interconnect supernode at trial/partnership stage; ~100 acceleration libraries; 340+ customers, 30+ industries, 1,000+ day cluster uptime, >10k chips online
- [半导体行业观察 (semi-insights)](https://www.semi-insights.com/s/bdt/15/50538.shtml) — July 19 launch; SIMT scalar/vector/tensor; attention/MoE/AF-separation/PD-separation optimizations
- AAStocks — 09903.HK company news: "released its new-generation general-purpose GPU flagship product, Tiangai 300" (https://www.aastocks.com)
- WAIC 2026 official dates July 17–20, 2026, Shanghai — WAIC official site and Shanghai municipal government English release
- [TrendForce, 2026-01-12](https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/) — prior roadmap context (Tianshu/Tianxuan/Tianji/Tianquan); **no source maps TianGai 300 onto these codenames**

### Baseline

- [Iluvatar CoreX Wikipedia](https://en.wikipedia.org/wiki/Iluvatar_CoreX)
- [Tom's Hardware — Iluvatar GPU roadmap vs Rubin](https://www.tomshardware.com/pc-components/gpus/chinas-iluvatar-corex-unveils-four-generation-gpu-roadmap-aimed-at-surpassing-nvidia-rubin)
- [TrendForce — 2026–28 roadmap targeting H200, B200](https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/)
- [TrendForce — Iluvatar eyeing Rubin 2027](https://www.trendforce.com/news/2026/01/28/news-chinas-illuvatar-corex-unveils-bold-gpu-roadmap-reportedly-eyeing-nvidias-rubin-by-2027/)
- [Caixin — HKEX IPO USD 475M](https://www.caixinglobal.com/2025-12-30/chinese-gpu-maker-iluvatar-corex-seeks-475-million-in-hong-kong-listing-102398766.html)
- [Caixin — HKEX debut USD 5.3B valuation](https://www.caixinglobal.com/2026-01-08/chinese-gpu-maker-iluvatar-corex-climbs-in-hong-kong-debut-with-53-billion-valuation-102401708.html)
- [BAAI — Aquila2-70B mixed-cluster training (BI-V100 + BI-V150)](https://www.elecfans.com/d/2328248.html)
- [CSDN — TianGai-150 specifications overview](https://blog.csdn.net/2402_84466582/article/details/139412485)
- [Leiphone — Iluvatar GPGPU architecture overview](https://m.leiphone.com/category/chips/fD16hBivUTdFgdUP.html)
