# Enflame Technology (燧原科技) Hardware Architecture

*as_of: 2026-08-08*
*chip: enflame*
*device_class: Data Transfer Unit (China)*
*Representative products: CloudBlazer T10 (DTU 1.0), CloudBlazer T20/i20 (DTU 2.0), CloudBlazer S60 (DTU 3.0), CloudBlazer L600 (DTU 4.0)*
*System products: SmartCluster (rack); 云燧 ESL64-O / ESL64-C supernodes (with ZTE, announced 2026-07-18)*

---

## Overview

Enflame's **Deep Thinking Unit (DTU)** — branded externally as the **General Compute Unit (GCU)** — is a reconfigurable spatial-dataflow AI accelerator designed for large-scale cloud datacenter training and inference workloads. Unlike GPU SIMT architectures, the DTU exposes a **static dataflow compilation model**: computation graphs are lowered by the **TopsCC compiler** into the GCU-CARE (Compute Architecture Reconfigurable Engine) execution fabric at compile time, with no dynamic thread scheduling on the hardware. This deterministic execution model is analogous in spirit to Groq TSP and SambaNova RDU, though Enflame's implementation is closer to a conventional MIMD core structure than a pure spatial array.

The fundamental compute hierarchy is: **SIP** (Scalable Intelligent Processor, ~equivalent to a GPU SM) → **SIC** (Scalable Intelligent Cluster, 8 SIPs) → **Die** (4 SICs = 32 SIPs). Each generation reuses this naming while substantially upgrading the SIP microarchitecture. The GCU-LARE (Local Area Reconfigurable Engine) interconnect handles inter-chip scale-up.

---

## 1. Compute Engine

### SIP — Scalable Intelligent Processor

The fundamental compute building block. Each SIP contains:

| Component | Description |
|-----------|-------------|
| Tensor ALU | Matrix/vector operations; supports mixed-precision (FP32/FP16/BF16/INT8/INT16/INT32) |
| Data Transfer Engine | On-chip DMA for moving data between local scratchpad and shared memory / HBM |
| Local SRAM Scratchpad | Private to the SIP; compiler-managed activation and weight buffers |
| Sparsity Engine | Hardware-level zero-skipping for weights and activations (key design differentiator) |

### SIC — Scalable Intelligent Cluster

Eight SIPs form one SIC. The SIC provides:
- Shared inter-SIP communication fabric (GCU-DARE, Data-path Architecture Reconfigurable Engine)
- Shared SRAM staging buffer accessible by all 8 SIPs within the cluster
- DMA coordination layer managing traffic between HBM and SIP scratchpads

### Product Specifications by Generation

| Product | Code | Process | SIPs | Peak FP32 | Peak TF32 | Peak INT8 | HBM | BW |
|---------|------|---------|------|----------|----------|---------|-----|-----|
| CloudBlazer T10 | 邃思1.0 | GF 12LP | 32 | ~8 TFLOPS (est.) | — | — | HBM2, 32 GB | ~512 GB/s |
| CloudBlazer i20 | 邃思2.0 (infer) | 12 nm | 32 | 32 TFLOPS | 128 TFLOPS | 256 TOPS | HBM2e, 16 GB | 819 GB/s |
| CloudBlazer T20 | 邃思2.0 (train) | 12 nm | 32+ | 40 TFLOPS | 160 TFLOPS | 320 TOPS | HBM2e, 64 GB | 1.8 TB/s |
| CloudBlazer S60 | 邃思3.0 (SCORPIO) | TSMC N6 | TBD | ~128 TFLOPS (est.) | — | ~512 TOPS (est.) | HBM2e | TBD |
| CloudBlazer L600 | 邃思4.0 | TSMC 5 nm (rumored) | Not disclosed | Not disclosed | Not disclosed | Not disclosed | HBM3, 144 GB | 3.6 TB/s |

*T20 packaging: 9-die MCM (5 compute + 4 HBM2e), 57.5 × 57.5 mm — China's largest AI chip at announcement (2021).*

*L600 additional card spec (added 2026-08-08, backfilled from the 2025-07-27 WAIC launch): **800 GB/s interconnect bandwidth**, native FP8. Status as of the June 2026 IPO filing: silicon returned (已回片), early commercialization; scale production and supernode delivery still described prospectively — **not** in mass production. Training products were 1.15% of AI-accelerator-card revenue in 2025.*

### Roadmap generations (no silicon announced)

| Generation | Evidence | Status as of 2026-08-08 |
|---|---|---|
| DTU 5.0 (5th-gen series) | IPO use-of-proceeds line 基于五代AI芯片系列产品研发及产业化项目, RMB 1.503 B | No tape-out, silicon, or specification announced |
| DTU 6.0 (6th-gen series) | IPO use-of-proceeds line 基于六代AI芯片系列产品研发及产业化项目, RMB 1.197 B | No tape-out, silicon, or specification announced |

### Precision Support

FP32, TF32, FP16, BF16, INT32, INT16, INT8 (all generations). FP8 added from DTU 4.0 (L600).

---

## 2. Data Path

### GCU-CARE — Compute Architecture Reconfigurable Engine

The core dataflow execution model. Key characteristics:

| Aspect | NVIDIA GPU (SIMT) | Enflame DTU (GCU-CARE) |
|--------|------------------|----------------------|
| Dispatch model | Dynamic warp scheduling | Static compile-time graph mapping |
| Threading unit | 32-thread warps | SIP (Scalable Intelligent Processor) |
| Memory model | HW caches + shared memory | Compiler-managed scratchpad tiling |
| Divergence handling | Predicated execution per thread | Not applicable (no runtime branching) |
| Synchronization | `__syncthreads()`, barriers | Compiler-inserted data-flow tokens |
| Sparsity | Software (NVIDIA 2:4 structured) | Hardware zero-skip engine |

### GCU-DARE — Data-path Architecture Reconfigurable Engine

On-chip data movement engine within a SIC. Responsibilities:
- Routes tensor tiles between SIP scratchpads and shared cluster SRAM
- Overlaps data prefetch with SIP compute (double-buffering)
- Handles reduction and broadcast patterns within the SIC
- Feeds the GCU-LARE inter-chip link at the chip boundary

### Sparsity Exploitation

One of Enflame's key architectural differentiators (highlighted at Hot Chips 33). The hardware:
- Detects zero weights / activations at tile granularity
- Skips the multiply-accumulate operations for zero entries
- Does not require explicit structured sparsity pruning (unstructured sparsity supported)

This yields super-linear throughput gains on sparse models (e.g., MoE, pruned BERT).

---

## 3. On-chip Memory

### Memory Hierarchy

| Level | Name | Scope | Managed By | Primary Use |
|-------|------|-------|-----------|-------------|
| 1st | SIP Local SRAM | Per SIP (private) | TopsCC compiler | Activation tiling, weight staging |
| 2nd | Cluster Shared SRAM | Per SIC (8 SIPs share) | TopsCC compiler | Inter-SIP tile exchange, staging |
| Off-chip | HBM stack(s) | Chip-level | HW DMA + compiler | Model weights, activations, optimizer state |

Unlike Cambricon's programmer-visible NRAM/WRAM qualifiers, Enflame's memory hierarchy is entirely managed by the TopsCC compiler and is not directly user-addressable. This simplifies programming but constrains low-level customisation.

---

## 4. Off-chip Memory

### Memory Configurations by Product

| Product | Memory Type | Capacity | Bandwidth | Notes |
|---------|------------|----------|-----------|-------|
| T10 (DTU 1.0) | HBM2 | 32 GB | ~512 GB/s | 2.5D CoWoS MCM, GF 12LP |
| i20 (DTU 2.0 infer) | HBM2e | 16 GB | 819 GB/s | Inference-optimised, 150W |
| T20 (DTU 2.0 train) | HBM2e × 4 | 64 GB | 1.8 TB/s | Training card, 9-die MCM |
| S60 (DTU 3.0) | HBM2e | TBD | TBD | TSMC N6, TechInsights reverse-engineered |
| L600 (DTU 4.0) | HBM3 | 144 GB | 3.6 TB/s | Native FP8; unveiled WAIC 2025-07-27; card also specified at 800 GB/s interconnect BW |

### Export Control Context

Enflame's S60 chip (TSMC N6) was scrutinised by NBC News / TechInsights in 2025 for potential US export control violations. The L600 (targeting ~H100-class performance with 144 GB > H20's 96 GB) positions Enflame directly against the Nvidia H20 in the Chinese domestic market post-export-restriction tightening.

---

## 5. Host Interface / Package

### PCIe Interface

| Product | PCIe | Form Factor | TDP |
|---------|------|-------------|-----|
| T10 | PCIe 4.0 x16 | FHFL | ~300 W |
| T20 | PCIe 4.0 x16 | FHFL | ~300 W |
| S60 | PCIe 4.0 x16 | FHFL | TBD |
| L600 | PCIe 5.0 (likely) | FHFL | TBD |

### Advanced Packaging

All *shipping* DTU generations use **2.5D MCM (Multi-Chip Module)** packaging:
- Compute chiplet(s) + HBM stack(s) connected via silicon interposer
- T20: 9-die MCM (5 compute die + 4 HBM2e stacks), 57.5 × 57.5 mm
- S60: SCORPIO-AO compute chiplet (TSMC N6NTO-HPC) + HBM2e stacks (TechInsights confirmed)

#### CoPoS panel-level packaging — sample only (added 2026-08-08)

On **2026-07-18** at WAIC, Enflame and Shanghai **先封科技 (XianFeng Technologies)** released a **CoPoS (Chip-on-Panel-on-Substrate) glass-substrate panel-level advanced-packaging sample** adapted to an Enflame high-end AI compute die. Reported as 国内首款面向AI算力芯片的玻璃基板CoPoS先进封装样品 — China's first public pairing of a domestic high-end AI compute die with domestic CoPoS panel-level packaging.

| Property | Claim |
|---|---|
| Substrate | Glass, panel-level format |
| Cited advantages | Tunable CTE, good planarity, low signal loss |
| Maturity | **Sample only.** Commentary explicitly states 不等同于台积电最终量产规格 — engineering validation of a domestic CoPoS route, not a production package |
| Applied to | "Enflame high-end AI compute chip" — the specific die is **not disclosed** |

This does not change the packaging of any shipping product; it is recorded as a forward-looking packaging direction.

---

## 6. Scale-up Interconnect (GCU-LARE)

**GCU-LARE** (Local Area Reconfigurable Engine) is Enflame's proprietary chip-to-chip scale-up interconnect, analogous to NVIDIA NVLink.

| Generation | BW (bidirectional) | Max Direct Connect | Notes |
|------------|--------------------|--------------------|-------|
| LARE 1.0 (T10) | Not disclosed | 4 chips direct | Non-cache-coherent; proprietary protocol |
| LARE 1.5 (T20) | ~300 GB/s bidir | 8 chips | GCU-LARE bridge card for 4-card full-mesh in-server |
| LARE 2.0 (T20+) | 300 GB/s bidir | 1,000s of cards | Scales to full SmartCluster |
| L600 (DTU 4.0) card spec | **800 GB/s** | Not disclosed | Announced with L600 at WAIC 2025-07-27; **no source attributes this figure to a named LARE generation**, so it is recorded as an L600 card-level interconnect spec rather than "LARE 3.0" |

**GCU-LARE Bridge Card**: A hardware adapter card that provides 3× GCU-LARE ports enabling full-mesh interconnect among 4 cards within a single server chassis without a switch ASIC.

### 云燧 ESL64 Supernodes (announced 2026-07-18, with ZTE) — added 2026-08-08

Enflame's first named supernode product line, launched jointly with **ZTE (中兴通讯)** at WAIC 2026 (Shanghai, 2026-07-17 to 07-20).

| System | Interconnect construction | Stated cluster scale | Notes |
|---|---|---|---|
| **云燧 ESL64-O** | **OEX 正交无背板** — orthogonal, backplane-free chassis achieving a "0 线缆" zero-cable card-to-card interconnect | Not disclosed | Marketed on reduced interconnect cost, low latency, signal integrity, thermals, and serviceability |
| **云燧 ESL64-C** | Conventional copper **Cable-tray** design | 万卡级以上 — 10,000+ card cluster networking | The 10,000-card claim is attributed by sources **specifically to ESL64-C**; no source makes it for ESL64-O |

**Not disclosed for either system:** card count per node (the "64" in the product name is *inferred from the name only* and is not confirmed by any source), per-link or per-card interconnect bandwidth, topology degree/diameter, and the silicon populating the nodes — sources describe it only as 自研AI芯片, and **no source confirms it is L600**.

Architecturally, the orthogonal backplane-free construction is the notable point: cards mate directly at right angles through the chassis midplane rather than through cabled connectors, which is what allows the "zero cable" claim. This is a chassis/mechanical scale-up strategy rather than a new link protocol — no new GCU-LARE generation was named at launch.

### NPO Optical Interconnect Prototype (WAIC 2026)

Separately from the ESL64 launch, Enflame demonstrated a **near-package optics (NPO)** optical-interconnect prototype intended to break copper reach limits at the supernode tier. Claim: 已实现对512张加速卡以上超节点架构的稳定支持 — stable support for supernode architectures of **512+ accelerator cards**. This is a prototype demonstration; no product, link rate, wavelength count, or optical-engine supplier is disclosed.

Industry context: both Enflame and Biren showed NPO optical scale-up at WAIC 2026, indicating a broader domestic shift from copper to near-package optics at the supernode tier.

---

## 7. Scale-out Interconnect

| Component | Description |
|-----------|-------------|
| ECCL (Enflame Collective Communication Library) | NCCL analog; AllReduce, AllGather for multi-GCU multi-node training over Ethernet/RoCE |
| Standard Ethernet / RoCE | Standard networking for inter-node communication in SmartCluster deployments |
| SmartCluster | Enflame's full-rack AI computing system integrating GCU cards, GCU-LARE, and ECCL |
| 云燧 ESL64-O / ESL64-C (2026-07-18) | Supernode systems co-developed with ZTE; ESL64-C is credited with 10,000+ card cluster networking. See §6 for construction detail and undisclosed parameters |
| NPO optical prototype (2026-07) | Near-package optics demonstration; claimed stable support for 512+ accelerator-card supernodes; prototype only |

No proprietary scale-out NIC or switching ASIC (analogous to NVIDIA NVSwitch for external switching) has been publicly announced. The ESL64 supernodes are chassis/mechanical scale-up products built with ZTE, not a new Enflame switching silicon.

---

## Update Log

### 2026-08-08 — WAIC 2026 systems, packaging sample, L600 interconnect backfill

*Window covered: 2026-04-05 → 2026-08-08.*

Added in this revision:
- **§5 Packaging:** CoPoS glass-substrate panel-level sample with 先封科技 (2026-07-18) — sample only; shipping products remain 2.5D MCM on silicon interposer.
- **§6 Scale-up:** L600 800 GB/s card interconnect spec (backfilled from the 2025-07-27 launch — a repo gap, not a 2026 development); 云燧 ESL64-O / ESL64-C supernodes with ZTE; NPO optical prototype.
- **§7 Scale-out:** ESL64 and NPO rows.
- **§1:** DTU 5.0 / DTU 6.0 recorded as IPO use-of-proceeds line items with no announced silicon; L600 status corrected to "silicon returned, early commercialization" (**not** mass production).

Not changed, because nothing new was disclosed: SIP/SIC counts for DTU 3.0 and DTU 4.0 (still undisclosed), S60 HBM capacity/bandwidth, L600 peak FLOPS, L600 process node (still only a "TSMC 5 nm" rumour), and all GCU-CARE/GCU-DARE ISA detail.

No Enflame paper or talk appeared at ISCA 2026 or any other venue in this window. Hot Chips 38 (2026-08-23/25) has not occurred and is not cited.

---

## Sources

- [GF press release: DTU 1.0 on GF 12LP (Dec 2019)](https://gf.com/gf-press-release/enflame-technology-announces-cloudblazer-dtu-chip-globalfoundries-12lp-finfet/)
- [Enflame DTU 1.0 at Hot Chips 33 — ServeTheHome (Aug 2021)](https://www.servethehome.com/enflame-dtu-1-0-ai-compute-chip-at-hot-chips-33/)
- [AI Compute Chip from Enflame — IEEE HC33 (2021)](https://ieeexplore.ieee.org/document/9567224)
- [智东西: 邃思2.0 中国最大AI芯片 (2021)](https://zhidx.com/p/281331.html)
- [云燧T20 百度百科](https://baike.baidu.com/item/%E4%BA%91%E7%87%A7T20/59243461)
- [Oceanpine Capital: i20 announcement](https://oceanpine.com/news/20220118451.html)
- [TechInsights: Enflame S60 Floorplan Analysis (2024)](https://www.techinsights.com/blog/enflame-s60-ai-accelerator-processor-floorplan-analysis)
- [TrendForce: Enflame L600 + MetaX at WAIC July 2025](https://www.trendforce.com/news/2025/07/29/news-chinese-ai-chip-unicorns-enflame-metax-unveil-next-gen-chips-shortly-after-nvidias-h20-return/)
- [T10 Product Manual — 燧原硬件文档中心](https://support.enflame-tech.com/onlinedoc_hw/3-t1x/t10/product_manual/content/source/T10_product_manual.html)
- [T20 Product Manual — 燧原软件栈文档中心](https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/product_manual/content/source/T20_product_manual.html)
- [GCU-LARE Bridge Card Manual](https://support.enflame-tech.com/onlinedoc_hw/3-t1x/t10/link_card/content/source/index.html)
- [NBCNews: TSMC Enflame S60 export scrutiny (2025)](https://www.nbcnews.com/tech/tech-news/ai-chip-tsmc-enflame-techinsights-rcna259342)
- [Enflame STAR IPO filing — Pandaily (Jan 2026)](https://pandaily.com/enflame-s-star-market-ipo-accepted-signaling-strong-momentum-in-china-s-ai-chip-sector)

### Added 2026-08-08

- [C114: Enflame + ZTE 云燧 ESL64-O launch, OEX orthogonal zero-cable; CoPoS with 先封科技; IPO registration 2026-07-09](https://www.c114.net.cn/industry/101611.html)
- [IT之家: ESL64-O launch detail — confirms no card count, bandwidth, or chip model disclosed (2026-07-18)](https://www.ithome.com/0/978/716.htm)
- [Sina Finance: ESL64-O vs ESL64-C positioning; NPO optical prototype supporting 512+ accelerator-card supernodes (2026-07-21)](https://finance.sina.com.cn/tech/shenji/2026-07-21/doc-iniipxxe9278635.shtml)
- [Sohu: ESL64-C Cable-tray design meeting 万卡级以上 cluster networking](https://www.sohu.com/a/1051829208_313745)
- [ZOL: NPO optical prototype, stable support for 512+ card supernode architectures](https://ai.zol.com.cn/1219/12193958.html)
- [Sina Finance: CoPoS glass-substrate panel-level sample with 先封科技 (2026-07-18)](https://finance.sina.com.cn/stock/t/2026-07-18/doc-iniifanf8343506.shtml)
- [腾讯新闻: L600 144 GB / 3.6 TB/s / 800 GB/s at WAIC (2025-07-27)](https://news.qq.com/rain/a/20250727A06S1L00)
- [DRAMeXchange: L600 spec coverage (2025-07-28)](https://www.dramx.com/News/IC/20250728-38864.html)
- [百度百科 燧原L600](https://baike.baidu.com/item/%E7%87%A7%E5%8E%9FL600/67178547)
- [Sohu: IPO prospectus wording on L600 scale production (forward-looking); training = 1.15% of accelerator-card revenue](https://www.sohu.com/a/1037503510_237556)
- [DRAMeXchange: IPO use-of-proceeds breakdown — 5th-gen RMB 15.03亿, 6th-gen 11.97亿, AI HW/SW co-innovation 33亿 (2026-06-16)](https://www.dramx.com/News/IC/20260616-40620.html)
- [WAIC 2026 official site (event held 2026-07-17 to 07-20)](https://www.worldaic.com.cn/)
