# Biren Technology BR10X (壁砺 / BR100 · BR104 · BR106 · BR110 · BR166) Software and Hardware Stack Summary

*as_of: 2026-08-08*
*prior revision: 2026-04-05*
*chip: biren*
*device_class: GPU-like AI Accelerator (China)*

---

## Overview

Biren Technology (壁仞科技), founded in Shanghai in 2019 by GPU industry veterans (ex-AMD, ex-Qualcomm), designed China's most powerful general-purpose GPU as of its August 2022 announcement at Hot Chips 34. The **BR100** is a dual-die GPGPU built on TSMC 7nm using CoWoS-S 2.5D packaging — 77 billion transistors, 64 GB HBM2e at 2.3 TB/s bandwidth, 300 MB on-chip L2 cache, PCIe 5.0 + CXL host interface, and **BLink** proprietary 8-GPU interconnect. Peak rated performance: **256 TFLOPS FP32 / 1,024 TFLOPS BF16 / 2,048 TOPS INT8**.

Two months after the announcement, US export controls (October 2022) forced TSMC to suspend advanced-node production for Biren, blocking mass production of the BR100. The **BR104** (monolithic single-die, 300W, 128 TFLOPS FP32) was positioned as the more producible domestic variant.

> **Generation caveat (added 2026-08-08).** Everything in the BR100/BR104 sections below describes the **Hot Chips 34 (2022) parts, which never reached volume production**. Biren's *commercial* portfolio is the **壁砺™ (Bili) family — BR106, BR110 and BR166** — the same BR10X first-generation architecture and the same naming series (壁砺106 = BR106, 壁砺110 = BR110, 壁砺166 = BR166), not a separate product line. Biren's site publishes **only form factor and peak power** for these SKUs: no FLOPS, memory, or process figures. The second-generation **BR20X** (native FP8/FP4, chiplet, BLink 2.0) is in physical design and tape-out verification. See **"壁砺 Product Line, WAIC 2026 and BR20X Update (2026-08-08)"** below; where that section and the BR100-era text disagree, the update section is current.

Despite the production setback, Biren executed a landmark **Hong Kong IPO on January 2, 2026**, raising **HK$5.58 billion (~$717M USD)** — the first Chinese GPU startup to go public. Shares surged 76% on debut (HK$19.60 → HK$34.46), signaling strong investor confidence in China's domestic AI chip ecosystem.

Biren's software platform, **BIRENSUPA**, is explicitly designed as a CUDA alternative. The SUPA stack maps directly to CUDA's layer structure: BRCC compiler (nvcc), BILA library (cuDNN/cuBLAS), SUPA runtime (cudart), BCCL (NCCL), and framework backends for PyTorch, TensorFlow, and PaddlePaddle.

---

## Software Stack

### Framework Integration

- **PyTorch SUPA backend**: Registers BR100/BR104 as `torch.device("supa")`. Dispatches ATen operations to Biren's BILA deep learning library. Distributed training via BCCL (NCCL analog). Supports TF32+/BF16 autocast.
- **TensorFlow**: XLA backend or custom device plugin integration.
- **PaddlePaddle**: Custom operator plugins; critical for Chinese enterprise AI market (Baidu ecosystem).
- **LLM inference serving**: SUPA-accelerated inference for LLaMA, Qwen, Baichuan models via vLLM-style serving frameworks.

### Compiler / IR

- **BRCC (Biren Runtime Compiler)**: nvcc analog. Compiles SUPA C++ GPU kernels (using `__global__`, thread/block/grid model, `<<<>>>` launch syntax) to Biren's native GPU ISA binary. Provides CUDA C++ migration mode for porting existing CUDA code with minimal changes.
- **SUPA MLIR inference compiler**: TensorRT analog. Accepts ONNX/PyTorch/TF models, performs graph fusion + quantization (INT8/BF16/TF32+), outputs compiled BR100 engine. Exploits 300 MB L2 cache for weight residency.
- **cuda2supa migration tool**: Automated CUDA → SUPA namespace substitution. Library shims for cuBLAS → BILA, cuDNN → BILA, NCCL → BCCL.

### Op Library

- **BILA (Biren DL Library)**: cuDNN + cuBLAS combined equivalent. Implements GEMM, Conv2D, Batched MatMul, Flash-Attention-style SDPA, Normalization, Pooling, Activation in TF32+/BF16/FP16/INT8 precisions. Exploits BR100's massive 300 MB L2 cache for weight/KV-cache residency.
- **General computing library**: BLAS L1–L3 equivalents; Reduce, Scan, Sort primitives.

### Kernel Library

- **SUPA kernel templates** (CUTLASS analog): GEMM/Conv template library for custom kernel authoring. Tiled for SPC (32 per die) and EU hierarchy. L2-cache-aware tiling for 300 MB shared L2.

### Runtime

- **SUPA Runtime API (`libsupa.so`)**: cudart analog. Provides `supaMalloc`/`supaFree`, `supaStreamCreate`/`Sync`, `supaEventCreate`/`ElapsedTime`, `supaMemcpyAsync`, and `<<<grid, block, shmem, stream>>>` kernel launch syntax. API designed for minimal CUDA migration friction.
- **SUPA Driver API**: Lower-level API for explicit context management, module loading, fine-grained memory control.

### Driver / Firmware

- **Linux kernel driver**: PCIe BAR mapping, IOCTL dispatch, DMA engine, interrupt handling.
- **On-GPU firmware**: Resource management and power control (NVIDIA GSP analog).
- **Kubernetes device plugin**: `birentech.com/gpu` resource advertising; health monitoring; SVI (Stream Virtualization Isolation) for multi-tenant workloads.

### Communication

- **BCCL (Biren Collective Communication Library)**: NCCL analog. AllReduce, AllGather, ReduceScatter over BLink (intra-node 8-GPU ring/tree). Falls back to InfiniBand/Ethernet for inter-node scale-out.

### Assembler / ISA

- **Biren GPU ISA (BGISA)**: Proprietary SIMT ISA. C-Warp based (warp execution with hardware divergence handling). Supports FP32/TF32+/BF16/FP16/INT8 natively. **Not publicly documented** — unlike NVIDIA PTX, there is no public virtual ISA. All development uses SUPA C++ + BRCC compiler.

---

## Hardware Architecture

### Compute Engine

The BR100 is a GPU-class SIMT accelerator organized around **SPC (Streaming Processing Clusters)**:

- **SPC**: 32 per die (64 total in dual-die BR100). Each contains 16 EUs and 4,096 threads.
- **EU (Execution Unit)**: Atom-level compute block with FP32/TF32+/BF16/INT8 pipelines and tensor units. Analogous to NVIDIA SM.
- **C-Warp**: SIMT warp execution model; threads execute same instruction in lock-step with predication for divergence.
- **No FP64**: The BR100 has zero FP64 compute (unlike NVIDIA H100 or AMD MI300X) — an AI-first design choice.

**TF32+** is Biren's extended TF32 format with higher dynamic range than NVIDIA TF32, delivering 512 TFLOPS on the BR100 at higher accuracy than standard TF32 for training workloads.

### Data Path

The BR100 is described as a **data-flow-centric architecture** with six proprietary features:
1. TF32+ precision
2. TDA — Data-flow access acceleration (optimized tensor streaming)
3. C-Warp — Data-flow parallel execution
4. NME — Near-memory engine (reduces die-to-die data movement)
5. NUMA/UMA — Memory access topology optimization
6. SVI — Stream Virtualization Isolation (multi-tenancy)

The on-chip network uses a **mesh topology** connecting SPCs to distributed L2 cache slices, similar to Intel's Sapphire Rapids server CPU design.

### On-chip Memory

| Level | Capacity | Scope | Managed By |
|-------|----------|-------|-----------|
| L1 / Shared Memory | ~few MB total | Per EU | Programmer (configurable) |
| L2 Cache (distributed) | **300 MB** | All SPCs on die | Hardware |

The 300 MB distributed L2 is the BR100's most distinctive on-chip memory feature — approximately 6× NVIDIA A100's ~40 MB L2, and larger than H100's 50 MB. This enables large model weight tiles and KV-cache to remain on-chip across many tokens/layers.

### Off-chip Memory

| Product | Memory | Capacity | Bandwidth | Interface |
|---------|--------|----------|-----------|-----------|
| BR100 | HBM2e | 64 GB | 2,300 GB/s | 4,096-bit |
| BR104 | HBM2e | 32 GB | 819 GB/s | 2,048-bit |

HBM supply from SK Hynix and Samsung is restricted by US export controls for Chinese chipmakers. This has been the critical production bottleneck for Biren since October 2022.

### Host Interface / Package

| Component | BR100 | BR104 |
|-----------|-------|-------|
| Host interface | PCIe Gen5 x16 + CXL | PCIe Gen4/5 |
| Form factor | OAM (550W) | PCIe FHFL (300W) |
| Packaging | TSMC CoWoS-S 2.5D | Monolithic |
| Die-to-die BW | 896 GB/s (via Si interposer) | N/A |

CXL support is a distinguishing feature vs contemporaries — allows BR100 HBM2e to appear as coherent CXL memory to the host CPU.

### Scale-up Interconnect (BLink)

**BLink** is Biren's proprietary GPU-to-GPU interconnect (NVLink analog). Figures below are the **BLink 1.x / BR100 (2022)** disclosure:
- 8 bidirectional BLink links per BR100 GPU
- 64 GB/s per link → **512 GB/s aggregate** bandwidth
- Up to **8 GPUs per server** node (DGX A100-class 8-GPU configuration)
- No external BLink switch ASIC disclosed at the time of the BR100 announcement (no NVSwitch equivalent)

> Superseded in part (2026-08-08): at WAIC 2026 Biren announced **BLink 2.0**, which does assume switch nodes (in-network computing / collective offload into switches) and a 1,024-GPU shared memory space over NPO optics. Per-link bandwidth for BLink 2.0 — and for the shipping 壁砺166 generation — is **not disclosed**. See the update section below.

### Scale-out Interconnect

- Standard InfiniBand or Ethernet via host NICs
- BCCL handles inter-node collective communication
- No proprietary scale-out network ASIC

---

## Performance vs NVIDIA A100

| Metric | BR100 | NVIDIA A100 SXM4 (80GB) |
|--------|-------|--------------------------|
| Peak FP32 | 256 TFLOPS | 312 TFLOPS |
| Peak BF16 | 1,024 TFLOPS | 312 TFLOPS (tensor) |
| Peak INT8 | 2,048 TOPS | 624 TOPS |
| Memory | 64 GB HBM2e | 80 GB HBM2e |
| Memory BW | 2.3 TB/s | 2.0 TB/s |
| L2 Cache | 300 MB | ~40 MB |
| TDP | 550W | 400W |
| Host Interface | PCIe 5.0 + CXL | PCIe 4.0 |
| Process | TSMC 7nm | TSMC 7nm |

Biren's claimed **2.6× average speedup** and **2.8× maximum speedup** over A100 in AI workloads is based on their own internal benchmarks (not independently verified). The large L2 cache and higher BF16/INT8 theoretical throughput support the claim for memory-bound transformer inference.

---

## Business Context

| Event | Date | Details |
|-------|------|---------|
| Founded | 2019 | Shanghai; ex-AMD/Qualcomm founders; focus on GPGPU for AI |
| BR100 announced | August 2022 | Hot Chips 34; 77B transistors; 2 PFLOPS AI |
| TSMC suspension | October 2022 | US export controls; advanced node production blocked; BR100 never reached volume |
| BR106 mass production | January 2023 | First commercially shipping Biren silicon (single die, BR10X arch) |
| BR110 mass production | October 2024 | Same architecture as BR106; edge-inference positioning |
| BR166 mass production | August 2025 | 2.5D chiplet of two BR106 dies; shipped at scale 2H2025 |
| HKEX listing hearing | December 17, 2025 | Prospectus: combined BR106+BR110 sales "exceeded 12,000 units" (cumulative) |
| IPO pricing | January 2026 | HK$19.60/share; raised HK$5.58B (~$717M) |
| Trading debut | January 2, 2026 | +76% to HK$34.46; institutional demand 26× oversubscribed; retail 2,348× |
| Market cap at debut | January 2026 | ~HK$46B (~$5.9B USD) |
| FY2025 annual results | March 30, 2026 | Revenue RMB 1.0346 B (+207.2% YoY); gross margin 53.8%; funding reserve > RMB 8.5 B |
| WAIC 2026 announcements | July 17–20, 2026 | NPO optical interconnect; BLink 2.0; 16/128/1,024-card supernode tiers; BR2xx FP8/FP4; Token Factory |

The IPO was the first Chinese GPU startup to go public and occurred amid a broader wave of Chinese AI investment following DeepSeek R1 (January 2025), which demonstrated competitive LLM performance from domestic Chinese AI infrastructure.

---

## 壁砺 Product Line, WAIC 2026 and BR20X Update (2026-08-08)

*Updated 2026-08-08. Primary sources: birentech.com product pages and newsroom (WAIC 2026 release); Biren FY2025 annual results (published 2026-03-30); BIRENSUPA / BirenTechnology GitHub org APIs. Secondary: Jiemian, NetEase (163.com), IT之家 via Sohu, 10jqka, Baidu Baike, HKEX listing coverage.*

Two things changed relative to the 2026-04-05 baseline. First, a **baseline miss**: the repo documented only the 2022 BR100/BR104 parts and never recorded the commercially shipping 壁砺 SKUs or the FY2025 results (which were published 2026-03-30, i.e. *before* the baseline scan). Second, **genuinely new material**: the WAIC 2026 announcements (July 2026) and the BIRENSUPA GitHub activity (May–Aug 2026).

### The shipping product line is 壁砺 (BR10X), not BR100/BR104

The naming is the *same family*, not a different one — 壁砺106 = BR106, 壁砺110 = BR110, 壁砺166 = BR166, all on the **BR10X first-generation architecture, TSMC 7nm**. What is stale in the sections above is the *generation*: BR100/BR104 were announced at Hot Chips 34 and then blocked by the October 2022 TSMC suspension; they never reached volume. birentech.com's hardware menu lists exactly five SKUs, and **no BR100 or BR104 page exists on the site**.

| SKU | Form factor | Peak power | Silicon | Status |
|---|---|---|---|---|
| 壁砺™166M | 4U OAM V1.1 module, air-cooled | 550 W | 2.5D chiplet: two BR106 dies co-packaged | Mass production from Aug 2025 |
| 壁砺™166L | OAM module, cold-plate liquid-cooled | 600 W | 2.5D chiplet: two BR106 dies | Shipping (2H2025) |
| 壁砺™166C | FHFL (290 mm) dual-width PCIe inference card | 300 W | 2.5D chiplet: two BR106 dies | Shipping (2H2025) |
| 壁砺™106M | OAM module, air-cooled | 400 W | Single BR106 die | Mass production from Jan 2023 |
| 壁砺™106B | FHFL dual-width PCIe card | not disclosed | Single BR106 die | Shipping |

- **BR106**: development started 2020, tape-out 2021, mass production January 2023. Single die.
- **BR110**: same architecture as BR106, single die, mass production October 2024, positioned for edge inference. It has **no product page on birentech.com**.
- **BR166 series** (2025): 2.5D chiplet co-packaging of **two BR106 dies** with a die-to-die interconnect. Mass production began **August 2025**; shipped at scale in 2H2025. Vendor claim: *"compute and memory performance doubled versus the prior generation."*
- Combined BR106 + BR110 sales **"exceeded 12,000 units"** — a *cumulative* figure from the HK IPO prospectus (listing hearing 2025-12-17, IPO January 2026), reported via secondary Chinese media. It is a pre-baseline, cut-off-dated number, not a current run-rate.

> **⚠️ Do not publish "800 TFLOPS BF16 / 128 GB HBM" for the 166 series.** Those numbers circulate widely on Chinese aggregators (Toutiao and downstream). All four Biren product pages (166M / 166L / 166C / 106M) were fetched directly and publish **only form factor and peak power** — no FLOPS table, no memory table. **No public datasheet gives FLOPS, HBM type/capacity/bandwidth, per-link BLink bandwidth, or process node for the 166 series or for BR20X.** Treat the aggregator figures as low confidence and unpublished.

### WAIC 2026 — NPO optical interconnect, BLink 2.0, and the distributed decoupled supernode

At **WAIC 2026 (Shanghai, 2026-07-17 to 07-20)** Biren announced, for the first time, an **NPO (near-packaged optics)** interconnect together with a **"distributed decoupled" supernode architecture** in which GPU nodes and switch nodes are *physically separated* and optically linked — letting the GPU nodes remain in standard server form factors instead of a bespoke rack. This is an **architecture and roadmap announcement, not a shipping product**: no availability date, no per-link bandwidth, and no customer were disclosed.

Three-tier supernode matrix:

| Tier | Scale | Interconnect |
|---|---|---|
| Standard server | 16 cards | Electrical |
| High-density cabinet | 128 cards | Electrical |
| Distributed decoupled supernode | **1,024 cards** | **NPO optical** |

The 1,024-card tier breaks the prior **128-card electrical scale-up ceiling**.

**BLink™ 2.0** (new; the repo previously covered only BLink 1.x) is stated to provide four capabilities:
1. **Memory-semantic interconnect** — up to **1,024 GPUs sharing one memory space**
2. **In-network computing** — collectives offloaded into the switches
3. **Intelligent congestion control**
4. **Multi-layer link self-healing**, from the physical layer through the framework layer

Biren's stated motivation: per-GPU scale-up bandwidth requirements **exceeding 1 TB/s**, while copper signalling *"attenuates severely within 3 metres."* NPO fuses the optical engine into the GPU module, **removes the high-power DSP chip**, and reaches *"several hundred metres."* (A "224 Gbps port rate" attributed to this announcement elsewhere is **not present** in Biren's release — do not repeat it.)

### BR20X — second-generation architecture, in development

| Attribute | BR20X ("BR2xx" in WAIC-day press) |
|---|---|
| Generation | 2nd-generation Biren architecture (successor to BR10X) |
| Numerics | **Native FP8 and FP4** |
| Packaging | Chiplet |
| Interconnect | Native supernode interconnect — BLink 2.0, ~1,000-card scale |
| Compute / memory | "Increased compute density, memory capacity and memory bandwidth" — absolute figures **not disclosed** |
| Process node | **not disclosed** |
| Status | R&D from 2024; **architecture design complete; in physical design and tape-out verification (流片验证)**. It has **not taped out**, is **not sampling**, and is **not shipping**. |
| Planned launch | 2026 (some reports say 2H26 / autumn 2026) — a *stated plan*, not a committed date |
| Follow-ons | **BR30X** (cloud training/inference) and **BR31X** (edge inference), targeted for 2028 commercialization |

Note on sourcing: WAIC-day press coverage refers to the part generically as the **"BR2xx series."** The specific string "BR20X" comes from prospectus / annual-report language, analyst notes and Baidu Baike — secondary sources, not a datasheet.

### Deployed at scale — with an important qualification

Biren's FY2025 report states it delivered multiple thousand-card intelligent computing clusters, *"including a 2,048-card optical-interconnect / optical-switching GPU supernode cluster."* Independent WAIC coverage attributes this to the **previous-generation dOCS** (distributed optical circuit switching) supernode — a **32-card / 4-chassis building block** aggregated into a 2,048-card cluster at a national-level computing platform reported as **Shanghai INESA (上海仪电)**.

**It is not a 2,048-card single scale-up / shared-memory domain, and it is not BLink 2.0.** Biren's own WAIC press release mentions the national-platform dOCS deployment *without stating a card count*; the 2,048 figure comes from the annual report and media coverage.

### FY2025 financial results (published 2026-03-30 — a baseline miss, not new news)

| Metric | FY2025 | YoY |
|---|---|---|
| Revenue | RMB **1.0346 B** (10.346 亿元) | **+207.2%** (from RMB 337 M) |
| Gross profit | RMB **557.0 M** | +210.8% |
| Gross margin | **53.8%** | +63 bps |
| R&D expense | RMB **1.476 B** | +78.5% |
| Adjusted net loss | ~RMB **874 M** | — |
| Cash + financial assets (end-2025) | RMB **2.896 B** | — |
| IPO net proceeds (early 2026) | RMB **5.631 B** | — |
| Total funding reserve | **> RMB 8.5 B** | — |

Two reading hazards: (1) all source figures are denominated in **亿元** — RMB 10.35亿 is RMB **1.035 billion**, not RMB 10.35 billion; (2) the **IFRS reported loss is far larger** than the adjusted loss, dominated by non-cash fair-value movements on convertible preferred shares — do not quote the headline loss without that caveat.

### Software — additive changes since baseline

The **BIRENSUPA** GitHub org has become the live one; the older **BirenTechnology** org (ModelZoo, k8s-device-plugin, go-brml) is **fully archived**, last push 2024-12-17.

| Repo (BIRENSUPA org) | Created | Last push | Note |
|---|---|---|---|
| `Mooncake` (fork) | 2026-07-20 | 2026-08-08 | **Actively developed.** KV-cache disaggregation / prefill-decode separation |
| `mmcv` (fork) | 2026-08-05 | 2026-08-05 | Newly forked |
| `biren-driver-management-tools-skill` | 2026-06-24 | — | Driver management tooling |
| `sglang` (fork) | 2026-05-06 | 2026-05-06 | **Static fork — zero pushes since creation.** Not active development |
| `PaddleCustomDevice` | 2023 | — | Pre-existing Paddle backend |

The Mooncake fork is the substantive signal: it puts **disaggregated prefill/decode with a KV-cache transfer engine** into Biren's serving path, which the repo's software section did not previously cover. The SGLang fork should **not** be described as an active port.

Biren also publishes **Day-0 model enablement** claims: MiniMax M3 (2026-06-16), Zhipu GLM-5.2 (2026-06-26), MiniMax H3 (2026-08-03).

**Token Factory** (vendor claim, primary-sourced). Biren's own WAIC 2026 release describes a "Token Factory" application framework claiming a **95%+ cache hit rate via five-level caching**, and a **cross-vendor heterogeneous co-inference** scheme co-developed with **China Telecom (中国电信)** claiming **~20% throughput gain**. This is unaudited vendor marketing, but it is on Biren's own newsroom rather than a third-party rumour.

### Not confirmed / not found

- **No Biren paper found at Hot Chips 2026, ISCA 2026 or ISSCC 2026.** For Hot Chips 38 specifically the program page returned HTTP 403, so this is *"could not verify"*, not *"absent"*. (Hot Chips 38 runs 2026-08-23 to 08-25 — 15 days after this update; no slides or abstracts exist yet in any case.)
- **No MLPerf submission found** for any Biren part.
- **No public datasheet** for the 166 series or BR20X: FLOPS, HBM type/capacity/bandwidth, per-link BLink 2.0 bandwidth, and process node are all **not disclosed**.
- Verification for this update ran with an exhausted web-search budget and relied on direct primary fetches plus DuckDuckGo HTML; **English-language coverage is thinner than ideal** and Baidu Baike was readable only via search snippets.

---

## Resources

### Official
- [Biren Technology Official Product Page — BR10X](https://www.birentech.com/BR10X.html)
- [Hot Chips 34 BR100 Slide Deck](https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf)

### Official — added 2026-08-08
- [birentech.com homepage — hardware menu lists exactly 壁砺™166L / 166M / 166C / 106M / 106B](https://www.birentech.com/)
- [壁砺™166M product page — 4U OAM V1.1 air-cooled, 峰值功耗 550 W (no FLOPS/memory published)](https://www.birentech.com/product/hardware/166m/)
- [壁砺™166L product page — cold-plate liquid-cooled OAM, 600 W peak](https://www.birentech.com/product/hardware/166l/)
- [壁砺™166C product page — FHFL (290 mm) dual-width PCIe inference card, 300 W peak](https://www.birentech.com/product/hardware/166c/)
- [壁砺™106M product page — air-cooled OAM, 400 W peak](https://www.birentech.com/product/hardware/106m/)
- [Biren newsroom index (WAIC 2026 item; Day-0 enablement items)](https://www.birentech.com/news/)
- [Biren WAIC 2026 press release — NPO, BLink 2.0, 16/128/1024-card tiers, BR2xx FP8/FP4, Token Factory](https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/)
- [BIRENSUPA GitHub organization (Mooncake / sglang / mmcv forks, driver tooling, PaddleCustomDevice)](https://github.com/BIRENSUPA)
- [BIRENSUPA developer documentation index](https://developer.birentech.com/Document_Hardware.html)
- [BIRENSUPA software product page](https://www.birentech.com/product/software/birensupa/)

### Analysis
- [Chips and Cheese — Hot Chips 34 BR100 Deep Dive](https://chipsandcheese.com/p/hot-chips-34-birens-br100-a-machine-learning-gpu-from-china)
- [ServeTheHome — Biren BR100 GPU for Datacenter](https://www.servethehome.com/biren-br100-gpu-for-datacenter-compute-and-ai-workloads/)
- [Tom's Hardware — 77B Transistors 2 PFLOPS](https://www.tomshardware.com/news/chinese-biren-rolls-out-new-gpus-with-77-billion-transistors-2-pflops-of-ai-performance)
- [WCCFTech — 2.8x faster than Ampere A100](https://wccftech.com/birentech-china-most-powerful-gpu-biren-br100-architecture-disclosed-2-8x-faster-than-nvidia-ampere/)
- [HPCwire — BIRENSUPA SDK overview](https://www.hpcwire.com/2022/08/22/chinese-startup-biren-details-br100-gpu/)
- [VideoCardz — Hardware specifications](https://videocardz.com/newz/chinas-biren-br100-is-7nm-hpc-gpu-with-77b-transistors-and-64gb-hbm2e-memory)

### Business / IPO
- [Bloomberg — $717M Hong Kong IPO](https://www.bloomberg.com/news/articles/2026-01-02/ai-chip-designer-biren-to-debut-after-717-million-hong-kong-ipo)
- [SCMP — Biren HK IPO bookbuilding](https://www.scmp.com/business/banking-finance/article/3337272/chinese-ai-chipmaker-biren-kicks-bookbuilding-us624-million-hong-kong-ipo)
- [IndraStra — TSMC suspends Biren](https://www.indrastra.com/2022/10/sources-tsmc-suspend-work-for-chinese.html)

### Added 2026-08-08 — WAIC 2026 and FY2025
- [IT之家 via Sohu (2026-07-19) — WAIC 2026: NPO, BLink 2.0 1024-GPU shared memory, three tiers, BR2xx FP8/FP4](https://www.sohu.com/a/1052133469_114760)
- [Jiemian — WAIC 2026: "BLink2.0…最多1024张GPU共享同一个内存空间"; notes no tape-out date given for BR2xx](https://www.jiemian.com/article/14787272.html)
- [NetEase 163.com (2026-07-17) — NPO upgrade from dOCS; prior-gen dOCS supernode at national platform "reaching 2,048 cards"](https://www.163.com/tech/article/L22F1NH700098IEO.html)
- [10jqka (2026-07-18) — 首次推出NPO光互连、分布式解耦架构超节点，单超节点1024卡](https://www.10jqka.com.cn)
- [WAIC 2026 official dates (Shanghai, July 17–20, 2026)](https://english.shanghai.gov.cn/en-DWAIC2026/index.html)
- [Tencent News (2026-03-30) — FY2025 revenue RMB 10.35亿, +207.2% YoY](https://news.qq.com/rain/a/20260330A08PTQ00)
- [Jiemian — FY2025 revenue RMB 10.346亿 (+207.2%), gross profit RMB 5.570亿 (+210.8%)](https://www.jiemian.com/article/14185757.html)
- [Futu News — FY2025 R&D expense RMB 14.76亿 (+78.5%)](https://news.futunn.com/post/70839289/)
- [Xueqiu — gross margin 53.8% (+63 bps); adjusted loss ~RMB 8.74亿](https://xueqiu.com/9983210953/381860568)
- [Sohu (2026-03-30) — first annual report: cash RMB 28.96亿, IPO net proceeds RMB 56.31亿, 2,048-card optical-switching supernode cluster, BR20X planned 2026 with FP8/FP4](https://www.sohu.com/a/1003843604_313745)
- [Baidu Baike — BR106: development 2020, tape-out 2021, mass production Jan 2023, 7nm, BR10X architecture](https://baike.baidu.com/item/BR106/67163892)
- [Baidu Baike — 壁砺166系列: co-packages two 壁砺106 dies via chiplet + die-to-die interconnect; launched 2025](https://baike.baidu.com/item/%E5%A3%81%E7%A0%BA166%E7%B3%BB%E5%88%97/67163995)
- [ChinaBizInsider — HKEX listing hearing cleared 2025-12-17; prospectus 12,000+ combined BR106/BR110 units](https://chinabizinsider.com/chinese-gpu-chip-designer-biren-technology-clears-hong-kong-ipo-hurdle-eyes-listing/)
- ⚠️ [Toutiao aggregator listing "BF16 800 TFLOPS / 128 GB HBM" for the 166 series — **only** source for these numbers; contradicted by the absence of any such table on birentech.com. Recorded for traceability; **not** treated as a spec](https://www.toutiao.com/w/1865240228792329/)
