# Enflame HW Architecture Investigation Report

*investigator: search-chip-toolchain*
*as_of: 2026-04-05*
*source: Web search (6 queries) + official Enflame documentation + IEEE HC33 + TechInsights*

---

## Investigation Summary

Enflame's hardware architecture is well-documented for DTU 1.0 (Hot Chips 33, 2021) and DTU 2.0 (multiple Chinese-language sources). DTU 3.0 (S60) has reverse-engineered die-level floorplan from TechInsights. DTU 4.0 (L600) announced at WAIC July 2025 with high-level specs only.

---

## Compute Engine

**Primary source**: IEEE HC33 paper (ieeexplore.ieee.org/document/9567224), GF press release (Dec 2019).

The DTU compute hierarchy:
- **SIP (Scalable Intelligent Processor)**: Atomic compute unit with Tensor ALU (multi-precision matrix/vector ops), a local Data Transfer Engine (DTE, on-chip DMA), local SRAM scratchpad, and a hardware sparsity engine.
- **SIC (Scalable Intelligent Cluster)**: 8 SIPs grouped together with a shared SRAM layer and GCU-DARE routing fabric.
- **DTU die**: 4 SICs = 32 SIPs for DTU 1.0 and DTU 2.0.

Key differentiator: **hardware unstructured sparsity** — the DTU can skip zero-valued weight/activation multiplications without requiring NVIDIA-style structured 2:4 pruning.

**Precision support**: FP32, TF32, FP16, BF16, INT32, INT16, INT8 (all generations). FP8 from DTU 4.0.

**Peak performance (T20, DTU 2.0)**: 40 TFLOPS FP32, 160 TF32, 320 INT8 TOPS.

---

## Data Path

**GCU-CARE** (Compute Architecture Reconfigurable Engine) is the static dataflow execution model. TopsCC compiler lowers computation graphs to GCU-CARE at compile time — no dynamic warp scheduling exists. This is the fundamental architectural distinction from GPU SIMT.

**GCU-DARE** (Data-path Architecture Reconfigurable Engine) handles intra-SIC tile routing, DMA overlap, and reduction/broadcast patterns. Fully compiler-managed — no user-visible DMA call API.

---

## On-chip Memory

Both SIP local SRAM and SIC shared SRAM are compiler-managed and are not user-addressable (contrast: Cambricon BANG C `__nram__`/`__wram__`, CUDA `__shared__`). TopsCC handles all tiling and data movement scheduling.

---

## Off-chip Memory

| Product | Memory | Capacity | BW |
|---------|--------|----------|-----|
| T10 (DTU 1.0) | HBM2 | 32 GB | ~512 GB/s |
| i20 (DTU 2.0) | HBM2e | 16 GB | 819 GB/s |
| T20 (DTU 2.0) | HBM2e × 4 | 64 GB | 1.8 TB/s |
| S60 (DTU 3.0) | HBM2e | TBD | TBD |
| L600 (DTU 4.0) | HBM3 | 144 GB | 3.6 TB/s |

T20 uses 9-die MCM (57.5×57.5 mm) — largest Chinese AI chip at announcement. S60 uses TSMC N6NTO-HPC SCORPIO-AO chiplet (TechInsights confirmed).

---

## Host Interface / Package

All generations: 2.5D MCM advanced packaging (compute chiplet + HBM stacks on silicon interposer). PCIe Gen4 x16 (T10/T20/S60), PCIe Gen5 likely for L600. FHFL server card form factor.

---

## Scale-up Interconnect

**GCU-LARE** (Local Area Reconfigurable Engine):
- LARE 1.0: up to 4-chip direct, non-cache-coherent
- LARE 2.0: 300 GB/s bidirectional, scalable to 1,000s of cards via SmartCluster topology
- **GCU-LARE Bridge Card**: 3 LARE ports per card → 4-card full-mesh intra-server, no external switch ASIC needed

---

## Scale-out Interconnect

ECCL (Enflame Collective Communication Library) over standard Ethernet/RoCE. SmartCluster product line packages full-rack AI compute with ECCL + networking. No proprietary NIC or switching ASIC.

---

## Confidence Assessment

| Component | Confidence | Basis |
|-----------|------------|-------|
| SIP/SIC hierarchy | High | IEEE HC33 paper, GF press release |
| GCU-CARE execution model | High | HC33 slides, ServeTehHome coverage |
| T20 peak performance | High | Multiple Chinese-language tech sources |
| S60 process node (TSMC N6) | High | TechInsights reverse engineering |
| L600 specs (144 GB HBM3, 3.6 TB/s) | Medium | TrendForce announcement coverage |
| DTU 3.0/4.0 SIP counts | Low | Not publicly disclosed |
| GCU-CARE ISA details | Low | Not publicly documented |

---

# Update Investigation — 2026-08-08

*investigator: update-chip-landscape (scan + round-3 adversarial verification)*
*window: 2026-04-05 → 2026-08-08*
*sources: WAIC 2026 Chinese-language trade press (C114, IT之家, Sina Finance, Sohu, ZOL), DRAMeXchange IPO coverage, 2025 L600 launch coverage (backfill)*

## Summary of this window

One genuinely new hardware-adjacent product line (the 云燧 ESL64 supernodes with ZTE), one packaging *sample*, one interconnect spec backfilled from before the baseline, and a status-verb correction on L600. **No new silicon generation.** No DTU 5.0 tape-out, no MLPerf submission, no conference paper.

## Scale-up Interconnect — new: 云燧 ESL64 supernodes

Announced 2026-07-18 at WAIC 2026 (Shanghai, 2026-07-17 to 07-20), jointly with **ZTE (中兴通讯)**. First named Enflame supernode product line; prior coverage stopped at the SmartCluster rack.

| System | Construction | Stated scale |
|---|---|---|
| 云燧 ESL64-O | OEX 正交无背板 — orthogonal, backplane-free chassis; "0 线缆" zero-cable card-to-card interconnect. Marketed on interconnect cost, latency, signal integrity, thermals, serviceability | Not disclosed |
| 云燧 ESL64-C | Conventional copper Cable-tray design | 万卡级以上 (10,000+ card) cluster networking — attributed by sources **specifically to ESL64-C** |

**Undisclosed (do not estimate):** card count per node — the "64" in the product names is inferred from the name only and is confirmed by no source; per-link and per-card interconnect bandwidth; topology degree/diameter; the silicon populating the nodes, described only as 自研AI芯片 (no source confirms L600).

Architecturally this is a chassis/mechanical scale-up strategy — cards mate at right angles through the midplane instead of through cabled connectors — not a new link protocol. No new GCU-LARE generation was named.

## Scale-up Interconnect — new: NPO optical prototype

Separate from ESL64, Enflame demonstrated a **near-package optics (NPO)** optical-interconnect prototype at WAIC 2026, claimed to have 已实现对512张加速卡以上超节点架构的稳定支持 — stable support for supernode architectures of 512+ accelerator cards. Prototype only; no product, link rate, wavelength count, or optical-engine supplier disclosed. Notable as an industry pattern: Biren showed NPO optical scale-up at the same event.

## Scale-up Interconnect — backfill: L600 800 GB/s

L600 is consistently specified at **144 GB / 3.6 TB/s / 800 GB/s interconnect**, native FP8. **This is not a 2026 disclosure** — it was published with the L600 launch at WAIC on 2025-07-27, before this survey's 2026-04-05 baseline. It is a repo gap being closed, not a development in this window. No source ties 800 GB/s to a named GCU-LARE generation, so it is recorded as an L600 card-level spec and does **not** supersede the LARE 2.0 300 GB/s bidirectional figure for T20-class parts.

## Packaging — CoPoS glass-substrate sample (2026-07-18)

With Shanghai 先封科技 (XianFeng Technologies), Enflame released a **CoPoS (Chip-on-Panel-on-Substrate) glass-substrate panel-level advanced-packaging sample** adapted to an Enflame high-end AI compute die — reported as 国内首款面向AI算力芯片的玻璃基板CoPoS先进封装样品. Cited advantages: tunable CTE, good planarity, low signal loss.

Maturity is explicit in the coverage: a **sample**, framed as engineering validation of a domestic CoPoS route, 不等同于台积电最终量产规格. The survey's existing claim that all shipping generations use 2.5D MCM on silicon interposers **remains correct**; CoPoS is recorded only as a forward-looking direction. The specific die is not disclosed.

## L600 status — correction to a superseded characterization

Do **not** write "in mass production." The prospectus sentence often cited — 随着公司第四代训推一体产品L600规模化量产及超节点系统的交付，公司将持续拓展训练领域 — is forward-looking, not a statement of present fact. Independent June-2026 coverage of the listing-committee filing describes L600 as 已回片但尚未大规模量产交付 (silicon back from fab, not yet in large-scale mass-production delivery), and reports training products at **1.15% of AI-accelerator-card revenue in 2025**.

Correct status verb: **silicon returned, early commercialization; scale production and supernode delivery still forward-looking as of the June 2026 filing.**

## Roadmap — DTU 5.0 / DTU 6.0

These exist only as IPO use-of-proceeds line items: RMB 1.503 B for 基于五代AI芯片系列产品研发及产业化项目 and RMB 1.197 B for the 6th-gen equivalent (of a RMB 6.0 B raise; the remaining RMB 3.3 B goes to 先进人工智能软硬件协同创新项目). No tape-out, silicon, or specification is announced for either.

## Still undisclosed after this window

- SIP/SIC counts for DTU 3.0 (S60) and DTU 4.0 (L600)
- S60 HBM capacity and bandwidth
- L600 peak FLOPS at any precision; L600 process node (only a "TSMC 5 nm" rumour)
- ESL64 card count, per-card bandwidth, host silicon
- All GCU-CARE / GCU-DARE ISA detail

## Confidence Assessment (2026-08-08 additions)

| Component | Confidence | Basis |
|-----------|------------|-------|
| ESL64-O / ESL64-C existence, launch date, partner (ZTE) | High | Four independent Chinese outlets (C114, IT之家, Sina, Sohu) |
| ESL64-O orthogonal backplane-free / zero-cable design | High | C114 + IT之家 |
| 10,000+ card claim attaches to ESL64-C specifically | Medium-High | Sohu; no source makes the claim for ESL64-O |
| ESL64 card count = 64 | **Not confirmed** — name inference only | No source |
| NPO prototype, 512+ card supernode support | Medium | Sina Finance, ZOL |
| CoPoS sample (existence, glass substrate, partner, sample status) | High | IT之家, 艾邦半导体网, Sina |
| L600 800 GB/s interconnect | High (but pre-baseline) | QQ News + DRAMeXchange, both 2025-07 |
| L600 "not in mass production" | High | Prospectus wording + independent June-2026 filing coverage |
| DTU 5.0 / 6.0 as funded projects only | High | DRAMeXchange IPO project breakdown |
