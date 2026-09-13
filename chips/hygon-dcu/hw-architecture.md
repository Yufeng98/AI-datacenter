# Hygon DCU Hardware Architecture Reference

*as_of: 2026-08-08*
*chip: hygon-dcu*
*device_class: GPU/DCU (China, 海光)*
*generations: 深算一号 (Y100) / 深算二号 (Z100, Z100L) / K100 / K100_AI / **深算三号 (BW1000)** / 深算四号 (in R&D)*

---

## Overview

Hygon DCU (Deep Computing Unit / 深算处理器) is a GPGPU-class accelerator derived from AMD's GCN/Vega architecture lineage. It uses the same 64-thread wavefront, CU-based compute structure, LDS scratchpad, and xGMI scale-up interconnect as AMD Instinct cards — but manufactured and sold by Hygon independently after the 2019 US Entity List restriction. The software stack (DTK/HIP) remains ROCm-compatible.

This document covers all 7 hardware layers with quantitative specifications for the **Z100, Z100L, K100, and K100_AI** generations.

> ⚠️ **Generation-scope warning (2026-08-08).** The current flagship is **深算三号 (BW1000)**, but Hygon has disclosed **no** compute, memory, process, TDP, or interconnect specification for it — and its own product page (`hygon.cn/product/accelerator`) publishes no DCU model names or specs at all. **Every number in sections 1–7 below describes Y100 / Z100 / Z100L / K100 / K100_AI, not BW1000.** Numbers for BW1000 do circulate in Chinese retail stock forums; they are mutually contradictory and are deliberately excluded — see "深算三号 (BW1000)" below.

---

## Chip Generations

| Product | CUs | Memory | BW | TDP | Process |
|---------|-----|--------|-----|-----|---------|
| Z100 | ~60 | 16–32 GB HBM2 | ~1 TB/s | ~300 W | TSMC 7nm (est.) |
| Z100L | ~60 | 32 GB HBM2 | ~1 TB/s | ~300 W | TSMC 7nm (est.) |
| K100 | ~64 | 32–64 GB HBM2e | ~1.2 TB/s | ~350 W | TSMC 7nm (est.) |
| K100_AI | ~64+ | 64 GB HBM2e | ~1.2 TB/s | 400 W | 7nm (est.) |
| 深算一号 (Y100) | 4,096 SPs / ~64 CUs | 32 GB HBM2 | 1 TB/s | — | — |
| 深算二号 (Z100) | — | — | — | 350 W | — |
| **深算三号 (BW1000)** — current flagship | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** |
| **深算四号** — in R&D (Hygon, 2025-12-11: "进展顺利") | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** |

---

## 深算三号 (BW1000) — Current Flagship, Specifications Undisclosed

*Added 2026-08-08. This corrects the 2026-04-05 baseline, which recorded K100_AI as the current flagship. BW1000 launched in 2025 and was publicly deployed by December 2025 — a repo miss, not a new announcement.*

**Status.** Hygon's only public status statement is its 2025-12-11 reply on the SSE investor-interaction platform: *"深算三号已经投入市场，受到客户认可"* (in market, accepted by customers). Hygon has **not** stated "mass production from Q3 2025" or "large-scale shipment in H1 2026" — those, along with "10k → 30k wafers/month capacity", "R&D began mid-2022" and "tape-out verification completed 2024", trace only to Chinese retail-investor channel-check posts and are treated as rumor. "Primary volume driver of H1 2026 growth" is brokerage inference; Hygon's 业绩预告 discloses only consolidated revenue and profit ranges.

**Confirmed deployment.** 中科南京信息高铁研究院 (CAS-affiliated) announced the **国内首发** of 深算三号 BW1000 on its 信息高铁智算算力网 AI development platform in December 2025, echoed by the Nanjing Qilin Park government newsroom on 2025-12-19/22.

**Disclosed hardware specifications: none.**

| Layer | BW1000 |
|---|---|
| 1. Compute Engine — CU count, clock, peak FP32/TF32/BF16/FP16/INT8 | not disclosed |
| 2. Data Path — wavefront width, SIMD structure, matrix/tensor unit presence | not disclosed (GCN/Vega DCU lineage presumed from unchanged DTK/HIP toolchain; not confirmed) |
| 3. On-chip Memory — LDS, VGPR, L1, L2 | not disclosed |
| 4. Off-chip Memory — HBM generation, capacity, bandwidth | not disclosed |
| 5. Host Interface / Package — PCIe generation, form factor, TDP, process node | not disclosed |
| 6. Scale-up — xGMI generation, per-card bandwidth, cards/node | not disclosed |
| 7. Scale-out — NIC, fabric | not disclosed (RCCL over Ethernet/RoCE presumed unchanged) |

> **Excluded figures and why.** A spec set of *FP32 49 TFLOPS / TF32 96 TFLOPS / BF16-FP16 192 TFLOPS / INT8 392 TOPS* circulates for BW1000. It originates from an Eastmoney 股吧 retail message-board post and a user-uploaded Baidu Wenku file — no datasheet, slide, or filing. Other circulating numbers for the same part contradict it outright: ~20 TFLOPS FP32 (Zhihu), FP64 ~30 TFLOPS (51CTO blog), "FP16 300T" for an alleged "Alibaba BW1000 AI variant" (Sept-2025 retail note). None is publishable.

**深算四号.** Hygon stated in the same 2025-12-11 investor reply that 深算四号 R&D is **"进展顺利"**. No timeline, node, or specification disclosed. A retail claim of "明年回片" (silicon back next year) is rumor.

**Negative results.** No Hygon/DCU MLPerf submission; no Hygon paper at Hot Chips 2026, ISCA 2026, or ISSCC 2026.

---

## 1. Compute Engine

### CU (Compute Unit) Microstructure

```
DCU (~60–64 Compute Units total)
├── Each CU at ~1.7 GHz:
│   ├── 4× SIMD16 vector execution units
│   │   ├── 16 FP32 lanes (64 FP32/CU)
│   │   ├── 16 FP64 lanes at 1/4 rate (64 FP64/CU)
│   │   └── 16 INT32 lanes
│   ├── 1× Scalar Unit (branch, control flow, addr calc)
│   ├── Branch + Message Unit
│   ├── LDS (Local Data Share): 64 KB
│   └── VGPR file: 64 KB per SIMD (4× = 256 KB/CU)
│
└── DPP (Data Parallel Processor) top-level cluster
    ├── CU groups
    ├── L2 Cache slices (~4–8 MB total)
    └── Command Processor (wavefront dispatch from host)
```

### Wavefront (vs CUDA Warp)

| Attribute | DCU Wavefront | NVIDIA Warp |
|-----------|--------------|-------------|
| Width | 64 threads | 32 threads |
| SIMD execution | 4 cycles × 16 lanes | 1 cycle × 32 lanes |
| Max in-flight per CU | ~40 wavefronts | ~64 warps |
| Max VGPRs per thread | 256× 32-bit | 255× 32-bit |

### Peak Performance

| Metric | DCU-Y100 | DCU-Z100 / K100_AI | 深算三号 (BW1000) |
|--------|----------|-------------------|-------------------|
| FP64 | ~22 TFLOPS (est.) | ~45 TFLOPS (est.) | not disclosed |
| FP32 | ~45 TFLOPS (est.) | 90 TFLOPS | not disclosed |
| TF32 | — | — | not disclosed |
| FP16 | ~90 TFLOPS (est.) | 180 TFLOPS | not disclosed |
| BF16 | Supported | Supported | not disclosed |
| INT8 | Supported | Supported | not disclosed |
| CU count / clock | ~64 CUs | ~60–64 CUs @ 1.7 GHz | not disclosed |

The Y100/Z100 FP32 and FP16 figures come from Chinese analyst reports (未来智库), not a Hygon datasheet. For BW1000 not even an analyst-report figure exists with a traceable primary source — see the exclusion note above.

---

## 2. Data Path

### SIMT Pipeline

```
Wavefront Dispatch
   │
   ├── Scalar Unit (branch, address, uniform ops)
   │
   └── 4× SIMD16 (vector ALU)
         ├── FP32 / FP64 / INT ops
         ├── LDS access (shared memory read/write)
         └── Global memory (buffer load/store → L1/L2/HBM)
```

- **Coalesced global loads**: 64-thread wavefront → 128-byte cache line (ideal case)
- **LDS bank conflicts**: 32 banks × 4-byte; 64-thread WF = up to 2-way serialization without padding
- **Latency hiding**: ~40 wavefronts in-flight per CU tolerate ~200–500 cycle HBM access latency
- **OpenMP/OpenACC**: Supported via DTK LLVM; same kernel compilation pipeline

---

## 3. On-chip Memory

| Level | Size | Scope | Managed By | Latency |
|-------|------|-------|-----------|---------|
| VGPR (per SIMD) | 64 KB (256 KB/CU) | Per-thread | Hardware register allocator | Sub-cycle |
| SGPR | ~16 KB/CU (est.) | Per-wavefront | Hardware register allocator | Sub-cycle |
| LDS (shared mem) | 64 KB/CU | Workgroup | Programmer (`__shared__` in HIP) | ~few cycles |
| L1 Cache | ~16–32 KB/CU (est.) | Per-CU | Hardware | ~10 cycles |
| L2 Cache | ~4–8 MB total | Chip-global | Hardware | ~100 cycles |

LDS is the primary high-bandwidth scratchpad for tiled algorithms (GEMM tiling, attention block processing). Explicit staging required via HIP `__shared__` declarations.

---

## 4. Off-chip Memory

### HBM Specifications

| Product | HBM Gen | Capacity | Bandwidth | Channels |
|---------|---------|----------|-----------|----------|
| Z100 | HBM2 | 16–32 GB | ~1.0 TB/s | 4-stack |
| Z100L | HBM2 | 32 GB | ~1.0 TB/s | 4-stack |
| K100_AI | HBM2e (est.) | 64 GB | ~1.2 TB/s (est.) | ~4-stack |
| 深算一号 (Y100) | HBM2 | 32 GB | 1.0 TB/s | 4× HBM2 |
| **深算三号 (BW1000)** | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** |

HBM is integrated on the same 2.5D interposer as the DCU die. Package manufacturing uses TSMC CoWoS or similar 2.5D integration (specific details not publicly disclosed by Hygon).

> **Scope (2026-08-08):** the HBM2 / HBM2e, 16–64 GB, ~1.0–1.2 TB/s figures above belong to **深算一号/二号 (Y100 / Z100 / Z100L / K100 / K100_AI) only**. They must not be presented as the memory subsystem of the current BW1000 flagship, for which Hygon has published no HBM generation, capacity, or bandwidth.

---

## 5. Host Interface / Package

| Attribute | Value |
|-----------|-------|
| Host interface | PCIe Gen4 x16 |
| Peak host BW | ~32 GB/s bidirectional |
| Form factor | PCIe FHFL dual-slot |
| TDP | 300–400 W (by generation) |
| Package | 2.5D interposer + HBM stacks |
| Process (est.) | TSMC 7nm |
| OEM servers | H3C R4900/R5300, Sugon GPU servers, Alibaba Cloud ECS |

> **Scope (2026-08-08):** PCIe Gen4 x16, FHFL dual-slot, 300–400 W and TSMC 7nm apply to **Y100 / Z100 / Z100L / K100 / K100_AI**. For **深算三号 (BW1000)** the PCIe generation, form factor, TDP, and process node are all **not disclosed**.

The Hygon C86 (x86 Zen-derived) CPU + DCU pairing is sold as a complete domestic AI server solution — unique in that both CPU and GPU accelerator come from the same Chinese vendor. Note that the planned Hygon absorption merger with Sugon (中科曙光), which would have brought the Sugon server line in-house, was **terminated on 2025-12-09**; Sugon remains an independent OEM partner (603019.SH).

---

## 6. Scale-up Interconnect (xGMI)

| Attribute | Value |
|-----------|-------|
| Protocol | xGMI (AMD Infinity Fabric derivative) |
| 深算一号 bandwidth | 184 GB/s multi-card |
| Link type | Peer-to-peer (no external switch ASIC required) |
| Topology | Fully connected within node |
| Max cards/node | 8 (typical server; exact max not disclosed) |
| SW protocol | HIP peer access (`hipDeviceEnablePeerAccess`) |
| **深算三号 (BW1000)** | **xGMI generation and scale-up bandwidth not disclosed** |

> **Scope (2026-08-08):** the 184 GB/s figure is a **深算一号 (Y100)** number from an analyst report. It is not a BW1000 figure and must not be carried forward as the current-generation scale-up bandwidth.

xGMI is the same protocol used in AMD Instinct MI-series cards. Hygon DCU implements a compatible version. Direct HBM-to-HBM transfers bypass PCIe, providing low-latency access for model-parallel training and inference.

---

## 7. Scale-out Interconnect

| Attribute | Value |
|-----------|-------|
| Network | Standard Ethernet (25/100 GbE) or InfiniBand via host NIC |
| Protocol | RoCE v2 |
| Communication library | RCCL (DTK port) |
| Collective ops | AllReduce (ring), AllGather, Broadcast, ReduceScatter |
| Multi-node training | Supported via PyTorch DDP / PaddlePaddle Fleet + RCCL |
| **深算三号 (BW1000)** | **not disclosed** (no BW1000-specific fabric, NIC, or RCCL change documented) |

No proprietary scale-out NIC is bundled with DCU. Standard data-center network infrastructure is used. RCCL handles topology-aware communication across nodes.

---

## Architecture: Key Comparisons

| Attribute | Hygon DCU (K100_AI) | AMD MI300X | NVIDIA H100 |
|-----------|--------------------|-----------|--------------| 
| CUs / SMs | ~64 CUs | 304 CUs | 132 SMs |
| Wavefront/warp | 64 threads | 64 threads | 32 threads |
| Memory | 64 GB HBM2e | 192 GB HBM3 | 80 GB HBM3 |
| Memory BW | ~1.2 TB/s | 5.3 TB/s | 3.35 TB/s |
| Peak FP16 | 180 TFLOPS | 1,307 TFLOPS | 989 TFLOPS |
| Scale-up | xGMI 184 GB/s | xGMI 896 GB/s | NVLink4 900 GB/s |
| Process | 7nm (est.) | TSMC 5nm | TSMC 4N |
| HIP compat | Yes (DTK) | Yes (ROCm) | No (CUDA) |

The K100_AI column above is a **2024-generation** comparison and is retained for architectural history. An equivalent row for the current **深算三号 (BW1000)** cannot be built: none of CU count, memory, memory bandwidth, peak FP16, scale-up bandwidth, or process node is disclosed.

DCU occupies a lower performance tier than AMD MI300X but has the advantage of domestic China availability, x86 CPU ecosystem pairing, and ROCm-compatible HIP programming — making it the most accessible GPU-class accelerator for Chinese AI developers using open-source frameworks.

---

## Sources

- [Optimizing Depthwise Separable Conv on DCU — CCF Springer 2024](https://link.springer.com/article/10.1007/s42514-024-00200-3)
- [Optimizing Sparse GEMM for DCUs — Journal of Supercomputing 2024](https://link.springer.com/article/10.1007/s11227-024-06234-2)
- [DCU Architecture Internal Block Diagram — ResearchGate](https://www.researchgate.net/figure/Internal-block-diagram-of-the-Hygon-DCU-architecture-the-DCU-relies-on-its-DPP-Data_fig5_393802594)
- [Z100 Hardware Specs — USTC CJCP](http://cjcp.ustc.edu.cn/hxwlxb/en/supplement/90419368-fad1-4c1f-9bee-84f971f833b1)
- [HAMi DCU Support — GitHub](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md)
- [深算一号/二号规格 — 未来智库](https://www.vzkoo.com/read/2024051540a9a146aec194e8412980de.html)
- [Hygon DCU Dual-Chip Roadmap — Digitimes Dec 2025](https://www.digitimes.com/news/a20251223VL208/ai-chip-china-cpu.html)
- [AMD xGMI Bandwidth — ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)

### Added 2026-08-08 (深算三号 / BW1000)

- [海光信息 accelerator product page — publishes no DCU model names or specifications](https://www.hygon.cn/product/accelerator)
- [中科南京信息高铁研究院 — 深算三号 BW1000 国内首发 (Dec 2025)](https://www.ictnj.ac.cn/newsinfo/10874852.html)
- [南京麒麟科创园 — 深算三号 首发部署新闻 (2025-12-22)](https://qilinpark.nanjing.gov.cn/xwzx/202512/t20251222_5748374.html)
- [海光信息投资者互动回复 (2025-12-11): 深算三号已投入市场；深算四号研发进展顺利 — 新浪财经](https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml)
- [深算三号 — 百度百科 (encyclopedia-grade; Hunyuan Hy3 adaptation claim)](https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890)
- [海光信息终止吸收合并中科曙光 — 财新 (2025-12-09)](https://www.caixin.com/2025-12-09/102391669.html) — merger termination; affects OEM/packaging context
- ⚠️ *Excluded as unusable spec sources (recorded so future passes do not re-adopt them):* [Eastmoney 股吧 BW1000 spec post](https://guba.eastmoney.com/news,688041,1483072345.html), [Zhihu BW1000 discussion](https://zhuanlan.zhihu.com/p/2066563534897652881), [集思录 thread](https://www.jisilu.cn/question/510441), [雪球 note](https://xueqiu.com/2453283973/400298837) — retail message-board / forum origin, mutually contradictory figures, no primary source.
