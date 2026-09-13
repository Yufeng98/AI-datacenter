# d-Matrix Corsair — Hardware Architecture Investigation

*chip: d-matrix*
*investigation: hw-architecture*
*date: 2026-04-05*
*sources: Hot Chips 2025, Chips & Cheese analysis, d-Matrix technical white paper, IEEE paper, ServeTheHome HC37 coverage*

---

## 1. Architecture Paradigm

d-Matrix Corsair is a **Digital In-Memory Compute (DIMC)** inference accelerator. Unlike conventional accelerators that shuttle weights from external DRAM to on-chip compute units, Corsair **embeds multipliers directly inside SRAM bit-cells**, turning memory into the compute plane. This eliminates the memory wall for weight-stationary LLM inference where reads dominate.

The evolution from the research prototype to Corsair 3DIMC™ involves 3D-stacking DIMC chiplets to increase SRAM density and bandwidth.

---

## 2. DIMC Core

The fundamental building block is the **DIMC core**:
- Integrates a **64×64 MAC array** (matmul with INT8) or 64×128 with INT4 directly in SRAM
- Weight bits stored in SRAM; activations streamed in; multiply-accumulate occurs at the storage site
- Reduces energy cost vs. reading weights from HBM then computing in a separate GEMM engine
- Supports **OCP MX formats**: MXINT4, MXINT8, MXINT16 (block floating point — blocks share one exponent)

---

## 3. Chiplet Architecture

```
Chiplet
└── 4 Quads
    └── 4 Slices each  →  16 slices / chiplet
        └── 16 DIMC cores / slice  →  256 DIMC cores / chiplet
```

All 256 DIMC cores within a chiplet are connected via an **all-to-all intra-chiplet interconnect**, making them act as a single logical entity.

**Per chiplet**:
- 256 DIMC cores
- 0.5 GB integrated SRAM (512 MB)
- Interfaces with external LPDDR5X channels

---

## 4. Chip and Card Organization

| Level | Count | Details |
|-------|-------|---------|
| DIMC cores / chiplet | 256 | 4 quads × 4 slices × 16 cores |
| Chiplets / chip | 4 | |
| Chips / Corsair card | 2 | |
| DIMC cores / Corsair card | 2,048 | single card |
| DIMC cores / dual-Corsair | 4,096 | via DMX Bridge |
| SRAM / Corsair card | 2 GB | at 150 TB/s aggregate |
| SRAM / dual-Corsair | 4 GB | |
| LPDDR5X / Corsair card | 256 GB | ~400 GB/s |

Two PCIe Corsair cards can be merged via a **DMX Bridge card** into a single logical 4 GB / 4,096-core unit.

---

## 5. Memory Hierarchy

| Tier | Type | Capacity | Bandwidth | Notes |
|------|------|----------|-----------|-------|
| Integrated (Performance Memory) | SRAM (in DIMC cores) | 2 GB / card | 150 TB/s | Weight storage + compute |
| Off-chip (Capacity Memory) | LPDDR5X | 256 GB / card | ~400 GB/s | KV-cache, large model overflow |

The 150 TB/s figure is the aggregate **internal SRAM bandwidth** accessible by all DIMC cores simultaneously — far exceeding HBM bandwidth. LPDDR5X is used for capacity (model weights that exceed SRAM, KV-cache).

---

## 6. Die-to-Die Interconnect: DMX Link

- Custom in-house IP called **DMX Link**
- Connects chiplets within a chip and between chips on a card
- Total bandwidth: **1 TB/s** die-to-die
- Enables all chiplets to behave as one large accelerator
- **DMX Bridge card**: passive bridge PCIe card allowing 2 Corsair cards to be merged, yielding 4x chiplet count

---

## 7. Host Interface and System Integration

- **PCIe Gen5 x16** (full-height half-length card)
- Standard server slot — no custom motherboard or HBM interposer needed
- Process node: **TSMC 6nm**
- Power: not publicly disclosed; targets inference efficiency

---

## 8. Scale-Out: JetStream I/O Card

Announced September 2025:
- Separate **JetStream 400G** PCIe Gen5 card (full-height)
- 400 Gbps Ethernet NIC purpose-built for AI inference traffic
- Pairs with Corsair cards in same server
- Enables rack-scale deployments without expensive InfiniBand
- d-Matrix also acquired **GigaIO SuperNODE** PCIe fabric technology for intra-rack Corsair scaling

---

## 9. Compute Specifications

| Metric | Value |
|--------|-------|
| Peak compute (INT8) | 2,400 TOPS / card |
| Numerical formats | MXINT4, MXINT8, MXINT16 |
| SRAM bandwidth | 150 TB/s |
| LPDDR5X bandwidth | ~400 GB/s |
| LPDDR5X capacity | 256 GB |
| Host interface | PCIe Gen5 x16 |
| Process node | TSMC 6nm |

---

## 10. Performance Claims

- Llama3 8B: **60,000 tokens/s** at **1 ms/token** on a single server
- Llama3 70B: **30,000 tokens/s** at **2 ms/token** in a single rack
- vs. GPU: 10× faster interactive speed, 3× better TCO, 3× greater energy efficiency
- Target: real-time interactive inference (1-token/batch streaming, low latency)

---

## 11. Key Architectural Insights

1. **Weight-stationary DIMC**: weights live in SRAM permanently during inference; activations are the streaming operands
2. **No HBM**: deliberate choice — HBM adds cost and CoWoS complexity; LPDDR5X provides capacity at lower cost
3. **MXINT4 efficiency**: block FP reduces data movement and quantization overhead simultaneously
4. **All-to-all chiplet mesh**: avoids bandwidth bottlenecks between chiplets unlike ring topologies
5. **PCIe-native**: positions as a disaggregated inference card alongside existing GPU infrastructure

---

## Sources

- [d-Matrix Corsair at Hot Chips 2025 — ServeTheHome](https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/)
- [d-Matrix Corsair: 256GB of LPDDR for AI Models — Chips & Cheese](https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for)
- [d-Matrix Technical White Paper](https://d-matrix.ai/pdf/d-Matrix-WhitePaper-Technical-FINAL.pdf)
- [Corsair: An In-memory Computing Chiplet Architecture — IEEE Xplore](https://ieeexplore.ieee.org/iel8/40/5210076/11108245.pdf)
- [d-Matrix Takes On AI Memory Wall — HPCwire](https://www.hpcwire.com/2025/09/02/d-matrix-takes-on-ai-memory-wall-with-3d-stacked-in-memory-compute/)
- [d-Matrix JetStream announcement](https://www.d-matrix.ai/announcements/jetstream/)

---

# Update — 2026-08-08 Investigation

*investigation: hw-architecture (update)*
*date: 2026-08-08*
*baseline: 2026-04-05 section above (retained unchanged)*
*sources: ISCA 2026 Raptor early-silicon paper; d-Matrix 2026-06-09 full-production announcement; d-Matrix × Alchip 2025-11-18 announcement; d-matrix.ai/product; GigaIO BusinessWire release 2026-04-02; ServeTheHome Hot Chips 2025 coverage*

---

## U1. Corsair productization status

| Item | Value | Confidence | Note |
|---|---|---|---|
| Status | **In full production** as of **2026-06-09**; "products to begin shipping in volume to priority customers" | high | Vendor announcement; corroborated by HPCwire and CNBC (2026-06-09) |
| Availability | Gated — "now available for select, qualified customers" as of 2026-07 | high | Parasail release |
| Customers | **not disclosed** (described only as "hyperscalers, neoclouds and frontier AI labs") | high | |
| First reference systems | **SquadRack**, "slated for deployment this summer" (2026) | medium | Secondary coverage (techedgeai) |
| Manufacturing partner | **Alchip Technologies** (design/production); process **TSMC N6** | high | New vs baseline, which said only "TSMC 6nm" |
| Packaging | Organic substrate; **no HBM, no CoWoS**; LPDDR5X for capacity | high | |
| Capacity | Multi-year TSMC/Alchip capacity secured (vendor statement) | medium | Vendor claim, no volumes given |

Correct status verb is **IN FULL PRODUCTION / BEGINNING VOLUME SHIPMENT** — *not* "deployed at scale."

---

## U2. Raptor — next-generation silicon (3DIMC)

The 2026-04-05 baseline treated "3DIMC" as an architecture trademark only. That is superseded. **Raptor** is the named successor to Corsair and the commercial debut vehicle for 3DIMC; **Pavehawk** is the lab-validated test silicon that preceded it. Announced jointly with Alchip on **2025-11-18** (vendor claim: "up to 10× faster inference than HBM4-based solutions").

**ISCA 2026** (2026-06-27 – 2026-07-01, Raleigh Convention Center) published *"Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference"* — Nair, Hadidi, Ganesh, Kodge, Rathore, Thanawala, Reddy, Saharia, Patankar, Tiruvur, Kurella, Bhoja (d-Matrix Inc. / UBC; includes CTO Sudeep Bhoja). The paper reports **on-silicon characterization, not simulation**.

### U2.1 Verified Raptor architecture

| Parameter | Value |
|---|---|
| Logic process | TSMC **N4P** |
| 3D integration | Logic die **face-to-face bonded** onto a 3D-DRAM die, **36 µm µbump pitch** |
| Chiplet organization | **4 gangs × 4 slices**; each slice = **4×4 tensor-engine array + 1 SIMD core** |
| 3D-DRAM per chiplet | **840 banks** mapped to **256 independent channels** (16 per slice) |
| MCM | **4 chiplets @ 1.2 GHz** |
| Card | **up to 4 MCMs** |
| Substrate / interposer | **9-4-9 organic substrate onto a 3D CoWoS interposer** |
| Power / thermals | **~422 W per MCM**, Tj up to 105 °C |
| Secondary memory tier | **8 on-package LPDDR5X-9600 devices = 128 GB per MCM** |
| Host / inter-MCM | **PCIe Gen7**; Gen-2 D2D at **32 Gbps/lane** |
| Peak TFLOPS/TOPS; numeric formats; on-die SRAM; 3D-DRAM capacity | **not disclosed** |
| Launch date / sampling date / pricing | **not disclosed** — status is **EARLY SILICON / PRE-PRODUCTION** |

### U2.2 Measured results (the only measured figures in this file)

- **~105 TB/s of 3D-DRAM bandwidth per card at 700 MHz**, with **2.5 ns average flit latency**. The paper frames this as **~12.5× an HBM3 card**.
- Paper throughput claim: **4.71× vs HBM** and **2.44× vs SRAM** across Llama-3.1 70B, DeepSeek-V3, Kimi K2, GPT-OSS, Whisper and Canary.

### U2.3 Architectural reading

Raptor keeps the in-memory-compute thesis but changes what the compute sits in. Corsair scaled SRAM (2 GB/card at 150 TB/s) and pushed capacity out to LPDDR5X. Raptor bonds logic directly to a 3D-DRAM die, buying DRAM density while preserving the wide-parallel access pattern DIMC depends on — 256 independent channels per chiplet, 16 privately owned per slice. Notably it **accepts CoWoS**, reversing the cost argument that explicitly ruled CoWoS out for Corsair; the 3D stack requires it.

---

## U3. Rack-scale specifications (vendor, d-matrix.ai/product)

Additive to the card-level figures in the baseline sections above, and internally consistent with them (2 GB @ 150 TB/s and 256 GB LPDDR5X per card).

| Level | Dense compute | Performance memory | Capacity memory | Other |
|---|---|---|---|---|
| Dual card (2 cards + DMX Bridge) | 4,800 TFLOPS MXINT8 / 19,200 TFLOPS MXINT4 | 4 GB @ 300 TB/s | up to 512 GB | 6,400 mm² total silicon; **512 GB/s card-to-card DMX Bridge** |
| Inference server (8 cards) | 19.2 PFLOPS MXINT8 / 76.8 PFLOPS MXINT4 | 16 GB @ 1,200 TB/s | up to 2 TB | PCIe-based scale-up; Llama3 8B 60,000 tok/s @ 1 ms/token |
| Inference rack (8 servers / 64 cards) | — | 128 GB @ 9.6 PB/s | up to 16.4 TB | up to 100B params "Performance Mode", 1T+ "Capacity Mode"; Llama3 70B 30,000 tok/s @ 2 ms/token |

All are **vendor marketing figures** with no disclosed batch size, accuracy-vs-precision tradeoff, or measurement methodology.

---

## U4. DMX Link vs DMX Bridge — do not conflate

A circulating correction claiming the repo's 1 TB/s DMX Link figure is wrong was **checked and rejected**. There are two distinct interconnects:

| Link | Scope | Bandwidth | Source |
|---|---|---|---|
| **DMX Link** | die-to-die / chiplet-to-chiplet | **~1 TB/s**, 115 ns D2D latency | ServeTheHome, Hot Chips 2025 |
| **DMX Bridge** | card-to-card | **512 GB/s** | d-matrix.ai/product |

The `die_to_die_bandwidth_TB_per_s: 1.0` field in the YAML sidecar is **correct** and must not be changed to 512 GB/s.

---

## U5. Intra-rack fabric — GigaIO assets (prior-window item, now fully specified)

Acquisition closed **2026-04-02** (predates the 2026-04-05 baseline; the baseline recorded it in one line). Full detail:

- d-Matrix acquired GigaIO's **datacenter business**: the **SuperNODE** system (up to 32 accelerators), the **FabreX** PCIe Gen5 memory fabric (**sub-200 ns cross-server memory access**), and the **Carlsbad engineering team**.
- Terms **not disclosed**.
- GigaIO, Inc. continues as an independent company focused on edge. Its own site footer now reads "FabreX is a trademark of d-Matrix, Inc.", independently corroborating the transfer.
- CEO Sid Sheth: *"Inference is bigger than any one chip. It's now a systems problem."*

## U6. SquadRack — rack blueprint (gap in the baseline)

Announced **2025-10-14 at the OCP Global Summit** with **Arista, Broadcom and Supermicro**: d-Matrix's "first blueprint for disaggregated standards-based rack-scale" inference. It is the named vehicle for the summer 2026 first deployments. It was absent from all prior revisions of this file.

---

## U7. Negative / non-findings (recorded so they are not re-searched)

- **MLPerf**: confirmed negative — no d-Matrix/Corsair submission in any MLPerf Inference round.
- **Hot Chips 38** (2026-08-23 – 2026-08-25, Stanford): d-Matrix **not listed in the advance program**. Conference is after this update's date; recheck afterwards.
- **Funding**: no 2026 round confirmed. The $275M on record is the **November 2025 Series C at ~$2B valuation** (~$450M total raised; Bullhound Capital, Triatomic Capital, Temasek leading; QIA, EDBI and Microsoft's M12 participating). M12's participation explains "Microsoft backing" headlines in June 2026 — not a new investment.
- **"Corsair's 32 compute units"** (vendor phrasing appearing in third-party Infinity coverage): could **not** be independently verified and does not reconcile with the documented 2,048 DIMC cores / 16 chiplets per dual card. **Deliberately not recorded as a spec.**
- **Parasail (2026-07)**: an announced partnership and stated **intent**, not a deployment. Corsair is **not** currently running across 40+ datacenters; Parasail's own post says they "plan to explore expanded integration." Intended architecture: NVIDIA Hopper/Blackwell for compute-bound **prefill**, Corsair for latency-sensitive **decode**. "Up to 10× faster interactive inference" and "up to 3× better energy efficiency" are unqualified vendor claims.
- **Gimlet Labs (2026-03-12)**: out of window (predates the 2026-04-05 baseline); vendor-relayed, workload unspecified, framed illustratively by Gimlet itself. Marketing, not a benchmark.

---

## Update Sources (2026-08-08)

- [Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference — ISCA 2026](https://ramyadhadidi.github.io/files/dMatrix-Raptor-ISCA.pdf)
- [ISCA 2026 conference site (dates: 2026-06-27 – 2026-07-01, Raleigh)](https://iscaconf.org/isca2026/)
- [d-Matrix × Alchip: world's first 3D-DRAM solution (Raptor announcement, 2025-11-18)](https://www.d-matrix.ai/announcements/d-matrix-and-alchip-announce-collaboration-on-worlds-first-3d-dram-solution-to-supercharge-ai-inference/)
- [Tom's Hardware — independent coverage of the 3DIMC claims](https://www.tomshardware.com/pc-components/ram/new-3d-stacked-memory-tech-seeks-to-dethrone-hbm-in-ai-inference-d-matrix-claims-3dimc-will-be-10x-faster-and-10x-more-efficient)
- [Corsair AI Inference Platform enters full production (2026-06-09)](https://www.d-matrix.ai/announcements/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/)
- [HPCwire — full-production coverage, "transition from sampling to scale manufacturing"](https://www.hpcwire.com/off-the-wire/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/)
- [techedgeai — TSMC N6 + Alchip, organic substrate vs CoWoS, SquadRack summer deployment](https://techedgeai.com/d-matrix-launches-corsair-inference-accelerator-to-slash-ai-latency/)
- [d-Matrix product page — dual card / server / rack specs, 512 GB/s DMX Bridge](https://www.d-matrix.ai/product/)
- [GigaIO Sells Datacenter Technology and Assets to d-Matrix — BusinessWire, 2026-04-02](https://secure.businesswire.com/news/home/20260402375332/en/GigaIO-Sells-Groundbreaking-Datacenter-Technology-and-Assets-to-d-Matrix)
- [supercomputing.news — SuperNODE up to 32 accelerators, FabreX sub-200 ns, Carlsbad team](https://www.supercomputing.news/ai/d-matrix-acquires-gigaios-data-center-business)
- [gigaio.com — footer now reads "FabreX is a trademark of d-Matrix, Inc."](https://gigaio.com/)
- [ServeTheHome Hot Chips 2025 — ~1 TB/s D2D, 115 ns, 275 W @ 800 MHz / 550 W @ 1.2 GHz, 38 TOPS/W](https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/)
- [Parasail × d-Matrix — "select, qualified customers", "plan to explore expanded integration"](https://www.parasail.io/blogs/parasail-d-matrix-accelerators)
- [MLCommons Inference (datacenter) — no d-Matrix submission](https://mlcommons.org/benchmarks/inference-datacenter/)
