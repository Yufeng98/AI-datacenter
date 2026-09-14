# Hygon DCU Hardware Architecture Investigation

*as_of: 2026-09-13*
*chip: hygon-dcu*
*device_class: GPU/DCU (China, 海光)*
*resource: hw-architecture*

> **2026-08-08:** the body below is the 2026-04-05 investigation, retained for architectural history. Two of its statements are corrected by the dated update appended at the end of this file: (1) the flagship is **深算三号 (BW1000)**, not K100_AI; (2) the Sugon merger was **terminated 2025-12-09**. All specifications below are scoped to Y100 / Z100 / Z100L / K100 / K100_AI.

---

## Summary

Hygon DCU (Deep Computing Unit / 深算处理器) is a GPGPU-class accelerator designed for AI training and HPC. It is architecturally derived from AMD's Vega/GCN lineage (via the AMD–Hygon JV), with Hygon developing independently after the 2019 Entity List addition. The execution model uses 64-thread wavefronts (vs AMD's own 64-thread wavefronts in RDNA/CDNA), CU-based compute clusters, LDS scratchpad, and HBM off-chip memory. Software runs in the DTK/HIP environment directly compatible with ROCm. Current AI flagship is the **K100_AI** (64 GB, 400 W). The **深算一号 (DCU-Y100)** and **深算二号 (DCU-Z100)** are the marketed product names.

---

## Product Generations

| Generation | Product Name | CUs / Cores | Memory | BW | TDP | Process |
|-----------|-------------|------------|--------|-----|-----|---------|
| Z100 (Gen 1) | DCU Z100 | ~60 CUs (est.) | 16–32 GB HBM2 | ~1 TB/s | ~300 W | TSMC 7nm (est.) |
| Z100L | DCU Z100L | ~60 CUs | 32 GB HBM2 | ~1 TB/s | ~300 W | TSMC 7nm (est.) |
| K100 (Gen 2) | DCU K100 | ~64 CUs (est.) | 32–64 GB HBM2e | ~1.2 TB/s | ~350 W | TSMC 7nm (est.) |
| K100_AI | 深算二号 | ~64+ CUs | 64 GB | — | 400 W | 7nm (est.) |
| 深算一号 (Y100) | DCU-Y100 | 4,096 CUs (per analyst report) | 32 GB HBM2 | 1 TB/s | — | — |
| 深算二号 (Z100) | DCU-Z100 | — | — | — | 350 W | — |

> Note: "4,096 CUs" in analyst reports likely refers to 4,096 shader cores / stream processors, not CUs. At 64 SPs/CU that equals ~64 CUs — consistent with 60-CU academic references.

---

## 1. Compute Engine

### CU (Compute Unit) Microstructure

```
Hygon DCU — Compute Architecture (per academic literature)
├── ~60 Compute Units (CUs) at 1.7 GHz (Z100-class)
│   └── Each CU contains:
│       ├── 4× SIMD16 execution units (64 FP32 ALUs/CU)
│       │   └── Each SIMD16: 16 FP32 + 16 FP64 + 16 INT ops
│       ├── 1× Scalar Unit (SCA — branch, control flow)
│       ├── 4× SIMD Schedulers (one per SIMD16)
│       ├── 64 KB LDS (Local Data Share / shared memory)
│       ├── 64 KB VGPR per SIMD (256× 32-bit regs per thread max)
│       └── 1× Branch+Message Unit
├── DPP (Data Parallel Processor) top-level cluster
│   └── Contains multiple CU groups + L2 cache slices
├── Fixed-function units: texture samplers, ROP units
└── Command Processor: dispatches wavefronts from host
```

### Wavefront Model

| Attribute | Value | vs NVIDIA |
|-----------|-------|-----------|
| Wavefront size | 64 threads | 32-thread CUDA warp |
| SIMD width per SIMD unit | 16 lanes | — |
| Cycles to execute 1 wavefront | 4 (64÷16) | 1 |
| Max VGPRs per thread | 256× 32-bit | 255× 32-bit |
| VGPR file per SIMD | 64 KB | — |

The 64-thread wavefront gives 2× register pressure headroom vs NVIDIA but requires explicit consideration for bank conflicts and occupancy tuning.

### Peak Performance (Z100 / K100 class)

| Metric | 深算一号 (Y100) | 深算二号 (Z100) | Notes |
|--------|--------------|--------------|-------|
| FP32 | ~45 TFLOPS (est.) | 90 TFLOPS | Analyst report |
| FP16 | ~90 TFLOPS (est.) | 180 TFLOPS | Analyst report |
| BF16 | Supported | Supported | DTK-level support |
| INT8 | Supported | Supported | via tensor ops |
| FP64 | Supported (1/4 FP32 rate, est.) | Supported | GCN heritage |

---

## 2. Data Path

### SIMT Execution Pipeline

```
Wavefront Dispatch → SIMD16 Issue → ALU Execute → Writeback
     ↓                    ↓
Scalar Unit (branch)   LDS access (shared memory)
     ↓
L1 / L2 Cache / HBM (global memory)
```

- **Wavefront scheduler** in each CU selects among up to 40 in-flight wavefronts for latency hiding
- **VGPR** (Vector General Purpose Registers) hold per-thread data
- **SGPR** (Scalar General Purpose Registers) hold wavefront-uniform data (addresses, constants)
- **LDS** (Local Data Share) = scratchpad shared among all threads in a workgroup
- Supports **OpenMP, OpenACC** on top of HIP via DTK LLVM compiler

### Memory Access Model

- **Coalesced global loads**: 64-thread wavefront coalesced into 128-byte cache lines
- **Bank conflicts in LDS**: 32 banks; with 64-thread WF, can cause 2× serialization if not padded
- **Texture cache**: L1-level read-only path for sampler-based access

---

## 3. On-chip Memory

| Level | Size | Scope | Access |
|-------|------|-------|--------|
| VGPR register file | 64 KB per SIMD (256 KB total/CU) | Per-thread | Sub-cycle |
| SGPR register file | ~16 KB per CU | Per-wavefront | Sub-cycle |
| LDS (shared memory) | 64 KB per CU | Workgroup | ~few cycles |
| L1 Cache | ~16–32 KB per CU (est.) | Per-CU | ~10 cycles |
| L2 Cache | ~4–8 MB total (est.) | Chip-global | ~100 cycles |

LDS is SW-managed via `__shared__` in HIP (same syntax as CUDA). The hardware does not automatically place data in LDS — the programmer (or compiler) must explicitly stage data.

---

## 4. Off-chip Memory

| Product | Memory Type | Capacity | Bandwidth | Interface |
|---------|------------|----------|-----------|-----------|
| Z100 | HBM2 | 16–32 GB | ~1 TB/s | HBM2 4-stack |
| Z100L | HBM2 | 32 GB | ~1 TB/s | HBM2 |
| 深算一号 (Y100) | HBM2 | 32 GB | 1 TB/s | 4× HBM2 channels |
| 深算二号 / K100_AI | HBM2e (est.) | 64 GB | ~1.2 TB/s (est.) | HBM2e multi-stack |

HBM is integrated on the same package/interposer as the DCU die. The HBM2 use is consistent with Vega-era architecture lineage. HBM2e/HBM3 upgrades are roadmap-level.

---

## 5. Host Interface / Package

| Attribute | Value |
|-----------|-------|
| Host interface | PCIe Gen4 x16 |
| Form factor | PCIe FHFL dual-slot card |
| Package | 2.5D interposer with HBM stacks |
| TDP | 300 W (Z100) → 400 W (K100_AI) |
| OEM servers | H3C R4900/R5300, Sugon (曙光) server series |

Hygon DCU acts as a PCIe accelerator attached to the host Hygon C86 CPU (x86, Zen-derived). The CPU+DCU pairing is Hygon's go-to-market strategy for domestic AI servers.

---

## 6. Scale-up Interconnect

| Attribute | Value |
|-----------|-------|
| Protocol | xGMI (AMD Infinity Fabric derivative) |
| 深算一号 bandwidth | 184 GB/s multi-card |
| Max cards per node | 8 (est., typical server config) |
| Topology | Peer-to-peer fully connected (within node) |

xGMI (External Global Memory Interconnect) is the same protocol used in AMD Instinct MI-series GPUs. Hygon DCU uses a compatible implementation for GPU–GPU direct memory access within a node. The Z100 and K100 cards support peer-to-peer HIP transfers via xGMI links without PCIe traversal.

---

## 7. Scale-out Interconnect

| Attribute | Value |
|-----------|-------|
| Protocol | Standard Ethernet / RoCE v2 |
| Library | DTK RCCL (ROCm Collective Communication Library port) |
| Collective ops | AllReduce, AllGather, Broadcast, ReduceScatter |
| Cluster evidence | Domestic cloud providers (Ali Cloud, Tencent Cloud, Inspur, Sugon) |

No dedicated scale-out NIC is bundled — standard network infrastructure (25/100 GbE or InfiniBand from domestic vendors like Mellanox alternatives) is used. RCCL over RoCE/Ethernet provides multi-node training support.

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────┐
│                    Hygon DCU (K100_AI)                      │
│                                                             │
│  ┌──────────────── DPP (Data Parallel Processor) ────────┐  │
│  │  CU0  CU1  CU2  ... CU63 (~64 Compute Units total)   │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │ Each CU: 4×SIMD16 | 64KB LDS | 64KB VGPR/SIMD  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  ┌──────────── L2 Cache (~4-8 MB) ──────────────┐    │  │
│  └──┤                                               ├────┘  │
│     └───────────────────────────────────────────────┘       │
│                         │                                   │
│  ┌──────────── HBM2e Memory (64 GB, ~1.2 TB/s) ──────────┐  │
│  └────────────────────────────────────────────────────────┘  │
│                         │                                   │
│      PCIe Gen4 x16 ←──  │  ──→ xGMI (184 GB/s multi-DCU)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Sources

- [Optimizing Depthwise Separable Conv on DCU — CCF Springer 2024](https://link.springer.com/article/10.1007/s42514-024-00200-3)
- [Optimizing Sparse GEMM for DCU — Journal of Supercomputing 2024](https://link.springer.com/article/10.1007/s11227-024-06234-2)
- [DCU Architecture Internal Block Diagram — ResearchGate](https://www.researchgate.net/figure/Internal-block-diagram-of-the-Hygon-DCU-architecture-the-DCU-relies-on-its-DPP-Data_fig5_393802594)
- [Z100 Hardware Specs Reference — USTC CJCP](http://cjcp.ustc.edu.cn/hxwlxb/en/supplement/90419368-fad1-4c1f-9bee-84f971f833b1)
- [HAMi DCU Support Doc — GitHub](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md)
- [深算一号/二号 规格 — 未来智库](https://www.vzkoo.com/read/2024051540a9a146aec194e8412980de.html)
- [Hygon Information Technology — Wikipedia](https://en.wikipedia.org/wiki/Hygon_Information_Technology)
- [AMD–Chinese Joint Venture — Wikipedia](https://en.wikipedia.org/wiki/AMD%E2%80%93Chinese_joint_venture)
- [Hygon DCU Dual-Chip Roadmap — Digitimes Dec 2025](https://www.digitimes.com/news/a20251223VL208/ai-chip-china-cpu.html)

---

## Investigation Update — 2026-08-08: 深算三号 (BW1000) and the Sugon merger termination

*Appended 2026-08-08. Scope: flagship correction, specification availability, corporate structure. This update corrects **two pre-existing errors** in the 2026-04-05 investigation above.*

### E1 — Pre-existing error: flagship is one generation stale

The summary above states "Current AI flagship is the **K100_AI** (64 GB, 400 W)". That was already wrong at the time of the 2026-04-05 baseline.

Hygon's current DCU flagship is **深算三号 (Shensuan No.3)**, marketed model **海光深算三号 BW1000**; the series reference "DCU 8300" appears in secondary listings only. BW1000 launched in **2025** and was publicly deployed by **mid-December 2025** — before the baseline date. This is a **missed item, not new news** in the Apr–Aug 2026 window, and the framing matters: it should not be recorded as a 2026 product launch.

### E2 — Pre-existing error: Sugon merger recorded as live

The 2026-04-05 investigation's `key_differentiators` lists "Merger with Sugon announced May 2025". The transaction was **formally terminated**. See "Corporate structure" below.

### Status verbs — what Hygon actually said

The strongest first-party statement about BW1000 is Hygon's **2025-12-11 reply on the Shanghai Stock Exchange investor-interaction platform (上证e互动)**:

> 深算三号已经投入市场，受到客户认可

i.e. **in market, shipping to customers, with customer acceptance**. That is the entirety of Hygon's public status disclosure. Explicitly, Hygon has **not** stated:

| Circulating status claim | Provenance | Verdict |
|---|---|---|
| "Mass production initiated Q3 2025" | Toutiao / Xueqiu retail channel-check posts | LOW — rumor |
| "Initial capacity 10k wafers/month → 30k in Q4 2025" | Same | LOW — rumor |
| "R&D began mid-2022" | Same | LOW — rumor |
| "Tape-out verification completed 2024" | Same | LOW — rumor |
| "Entered large-scale shipment in H1 2026; primary volume driver of compute growth" | Brokerage / financial-media inference layered on the 业绩预告 | Analyst attribution, **not** company disclosure |

Hygon's H1 2026 业绩预告 discloses only consolidated revenue and profit ranges and does not attribute growth to 深算三号 by name.

### Deployment evidence (solid)

The CAS-affiliated **中科南京信息高铁研究院** announced in December 2025 the **国内首发** of 深算三号 BW1000 on its **信息高铁智算算力网** AI development platform. Echoed by the Nanjing Qilin Park government newsroom on **2025-12-19/22**. This establishes real deployment rather than announcement-only status, and is the evidentiary basis for calling BW1000 the shipping flagship.

### Hardware specifications: none disclosed

| Layer | BW1000 status |
|---|---|
| Compute — CU count, clock, FP64/FP32/TF32/BF16/FP16/INT8 peaks | **not disclosed** |
| Data path — wavefront width, SIMD structure, matrix-unit presence | **not disclosed** (GCN/Vega DCU lineage presumed only because the DTK/HIP toolchain is unchanged; not confirmed) |
| On-chip memory — LDS, VGPR, SGPR, L1, L2 | **not disclosed** |
| Off-chip memory — HBM generation, capacity, bandwidth | **not disclosed** |
| Package / host — process node, TDP, form factor, PCIe generation | **not disclosed** |
| Scale-up — xGMI generation, per-card bandwidth, cards/node | **not disclosed** |
| Scale-out — fabric, NIC | **not disclosed** |

Hygon's own accelerator product page (`hygon.cn/product/accelerator`) publishes **no DCU model names or specifications at all**. There is therefore **no vendor-primary spec source** for BW1000 anywhere.

### Rejected specification figures — recorded so they are not re-adopted

A figure set of **FP32 49 TFLOPS / TF32 96 TFLOPS / BF16-FP16 192 TFLOPS / INT8 392 TOPS** circulates widely for BW1000. Provenance and disposition:

- **Actual origin**: an Eastmoney 股吧 retail stock message-board post (`guba.eastmoney.com/news,688041,1483072345.html`) and a user-uploaded Baidu Wenku document. **Not** a "Chinese financial-data aggregator" as sometimes described, and not a datasheet, conference slide, or filing.
- **Mutually contradicted** by other circulating figures for the same part:
  - ~20 TFLOPS FP32 (Zhihu long-form piece)
  - FP64 ~30 TFLOPS, "H100-class" (51CTO blog)
  - FP16 300T for an alleged "Alibaba BW1000 AI variant" (Sept-2025 retail note)

Because these conflict and none has a primary source, **no BW1000 throughput number is admitted to the survey**. Confidence: `rejected`.

### Consequence for the tables above

The memory and interconnect specifications in sections 4 and 6 of the 2026-04-05 investigation (HBM2/HBM2e 32–64 GB, ~1.0–1.2 TB/s; xGMI 184 GB/s; PCIe Gen4) are **scoped to 深算一号/二号 (Y100 / Z100 / Z100L / K100 / K100_AI)** and must not be presented as current-generation. In particular, 184 GB/s is a **Y100** analyst-report number.

### 深算四号 (Shensuan No.4)

In the **same 2025-12-11** investor-platform reply, Hygon stated 深算四号 R&D is **"进展顺利"** (progressing smoothly). No timeline, node, or specification disclosed. Note the date: this is a **December 2025** statement, roughly eight months before this update and before the repo's own 2026-04-05 baseline — it is **not** an H1 2026 disclosure. A retail note claiming "明年回片" (silicon back next year) is rumor.

### Corporate structure — Sugon absorption merger TERMINATED

| Date | Event |
|---|---|
| 2025-05-25 | Hygon announces plan to absorb 中科曙光 (Sugon) via share exchange |
| 2025-05-26 | Hygon shares halted |
| 2025-06-09 | Formal **重组预案** disclosed — exchange ratio **1 Sugon : 0.5525 Hygon**; Sugon priced 79.26 RMB/sh (120-day VWAP +10%), Hygon 143.46 RMB/sh; headline value **~RMB 115.9–116.0 B**; described as the first case under China's revised merger regulations |
| 2025-06-10 | Trading resumes |
| **2025-12-09** | **Hygon board resolves to terminate** the 换股吸收合并 and the associated matched-funds raise: *"市场环境较本次交易筹划之初发生较大变化，本次实施重大资产重组的条件尚不成熟"*, compounded by the deal's scale and counterparty count. Hygon committed to plan no further major asset restructuring for ≥1 month, stated termination would not materially harm operations, and that industrial cooperation with Sugon continues. |

Confirmed independently by **Caixin, Yicai, Jiemian, Eastmoney, 观察者网, 同花顺** (2025-12-09/10). **Sugon did not delist** and remains independently listed as **603019.SH**; a July 2026 sell-side-style deep-dive frames Sugon's post-termination "independent path". No source shows the deal revived through 2026-08-08.

Architecture consequence: the Sugon server line remains a **third-party OEM** channel for DCU, not an in-house one.

### H1 2026 financials (disclosed 2026-07-16) — confirmed

| Metric | Range | YoY |
|---|---|---|
| Revenue | RMB 8.5–9.3 B (85–93 亿) | +55.56% – +70.20% |
| Net profit | RMB 1.70–1.83 B (17–18.3 亿) | +41.50% – +52.32% |
| Non-GAAP (扣非) net profit | RMB 1.51–1.70 B | +38.53% – +55.96% — *not independently re-verified* |

Consolidated CPU+DCU; no DCU-only breakout. Growth percentages are the safest citation.

### Negative results

- **No** Hygon/DCU MLPerf Training or Inference submission found.
- **No** Hygon paper at Hot Chips 2026, ISCA 2026, or ISSCC 2026.
- **No** vendor block diagram, die shot, or microarchitecture disclosure for BW1000.

### Sources — 2026-08-08 update

- [海光信息投资者互动回复 (2025-12-11) — 新浪财经](https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml)
- [中科南京信息高铁研究院 — 深算三号 BW1000 国内首发](https://www.ictnj.ac.cn/newsinfo/10874852.html)
- [南京麒麟科创园新闻 (2025-12-22)](https://qilinpark.nanjing.gov.cn/xwzx/202512/t20251222_5748374.html)
- [海光信息 accelerator product page (no DCU specs published)](https://www.hygon.cn/product/accelerator)
- [终止吸收合并中科曙光 — 财新 (2025-12-09)](https://www.caixin.com/2025-12-09/102391669.html)
- [终止换股吸收合并 — 第一财经](https://www.yicai.com/news/102948899.html)
- [终止合并 — 界面新闻](https://www.jiemian.com/article/13741594.html)
- [终止重组 — 东方财富 (2025-12-09)](https://finance.eastmoney.com/a/202512093586757039.html)
- [终止重组后续 — 东方财富 (2025-12-10)](https://finance.eastmoney.com/a/202512103587902138.html)
- [终止合并 — 观察者网 (2025-12-10)](https://www.guancha.cn/economy/2025_12_10_799933.shtml)
- [终止吸收合并 — 同花顺 (2025-12-09)](https://news.10jqka.com.cn/20251209/c673085520.shtml)
- [曙光后续报道 — 第一财经](https://www.yicai.com/news/102952855.html)
- [深算三号 — 百度百科](https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890)
- [IT之家 相关报道](https://www.ithome.com/0/977/747.htm)
- ⚠️ **Rejected spec sources** (retail forums, contradictory, no primary basis): [Eastmoney 股吧](https://guba.eastmoney.com/news,688041,1483072345.html) · [Zhihu](https://zhuanlan.zhihu.com/p/2066563534897652881) · [集思录](https://www.jisilu.cn/question/510441) · [雪球](https://xueqiu.com/2453283973/400298837)

## Investigation Update — 2026-09-13: Audited H1 2026 results only

*classification: Roadmap (financial) — no new hardware facts found*

The 2026-07-16 H1 2026 pre-announcement recorded in the 2026-08-08 update was superseded by Hygon's **audited 半年报, published 2026-08-13**:

| Metric | Audited H1 2026 | YoY |
|---|---|---|
| Revenue | RMB 9.099 B (90.99亿元) | +66.52% |
| Q2 2026 revenue alone | RMB 5.065 B (50.65亿元) | +65.32% |
| Net profit | RMB 1.798 B (17.98亿元) | +49.69% |
| Gross margin | down 4.94 pp YoY | — |

Both figures land inside the pre-announced ranges already on record, so this is a narrowing/confirmation, not a correction. Still consolidated CPU+DCU — DCU is not broken out. Market snapshot (2026-08-21, not a disclosure): share price RMB 246.21, market cap ≈ RMB 572.3B.

**No new hardware finding.** Searches for 深算三号/BW1000 spec disclosures and 深算四号 status in this window returned only restatements of the 2025-12 "进展顺利" (progressing smoothly) language already on record; no tape-out, sampling, or spec news located. No MLPerf submission and no Hot Chips 38 / ISCA 2026 paper found (Hot Chips 38 has now passed with nothing located, consistent with the task brief's premise for this batch).

## Sources — added 2026-09-13
- Sina Finance (2026-08-17): https://finance.sina.com.cn/tech/roll/2026-08-17/doc-ininruyf6769682.shtml
- Eastmoney (2026-08-13): https://finance.eastmoney.com/a/202608133840707684.html
- Eastmoney Fund (2026-08-13): https://fund.eastmoney.com/a/202608133840708424.html
- Securities Star / MSN — Q2 revenue detail: https://www.msn.cn/zh-cn/money/技术/每周股票复盘-海光信息-688041-q2收入50-65亿增65-32/ar-AA2adHfu
