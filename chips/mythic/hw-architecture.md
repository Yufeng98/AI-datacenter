# Mythic AMP — Hardware Architecture

*chip: mythic*
*as_of: 2026-08-08*
*device_class: Analog In-Memory Compute*
*primary sources: https://mythic.ai/technology/, https://mythic.ai/products/m1076-analog-matrix-processor/, https://fuse.wikichip.org/news/5727/mythic-rolls-out-m1000-series-analog-ai-accelerators-raises-70m-along-the-way/, https://fuse.wikichip.org/news/2755/analog-ai-startup-mythic-to-compute-and-scale-in-flash/*
*2026-08-08 update sources: https://www.videantis.com/mythic-acquires-videantis.html, https://global.honda/en/topics/2026/c_2026-02-04eng.html, https://mythic.ai/whats-new/, https://www.taylorwessing.com/en/insights-and-events/news/media-centre/press-releases/2026/06/mythic-inc-acquires-videantis*

---

## Company Status (Corrected)

The prompt asserts Mythic shut down in 2024. This is **incorrect based on public records**. The accurate timeline:

- **November 2022**: Mythic ran out of operating cash; near shutdown.
- **March 2023**: Rescued by a $13 M bridge round (Atreides, DCVC, Lux Capital, Catapult Ventures, Hermann Hauser Investment); Dave Fick became CEO.
- **June 2024**: Named NVIDIA veteran Dr. Taner Ozcelik as CEO to expand AI inference market.
- **December 2025**: Raised $125 M oversubscribed round led by DCVC (joined by NEA, SoftBank KR, Honda Motor, Lockheed Martin, Atreides, others). Claims 100× energy advantage vs. GPU. Targets 1T-parameter LLMs and datacenter deployment. Total funding: ~$290 M across 9+ rounds.
- **February 2026**: Honda + Mythic announce **joint development** of an analog-AI SoC for software-defined vehicles; Honda R&D **licenses Mythic APU technology**. Honda is now both an investor and a co-development partner. *(Backfill — predates the 2026-04-05 baseline scan.)*
- **May 2026**: **Acquired Videantis GmbH** (Hannover, Germany), a licensable digital processor-IP vendor; transaction closed. eCAPITAL joined the Mythic cap table as part of the deal. Mythic's stated forward architecture becomes **analog CIM + programmable digital core**. See "2026 Update" below.
- **HQ note (unresolved)**: repo records Austin, TX; Honda's 2026-02-04 release says "a Texas, U.S.-based technology company"; Mythic's 2026-05-19 dateline reads "Palo Alto, CA". Treat as dual-site.

---

## Fundamental Architecture: Analog Compute in Flash Memory

Mythic's core innovation is the **Analog Matrix Processor (AMP)**: matrix-vector multiplication performed entirely inside embedded NOR flash memory cells, in the analog domain.

Standard digital accelerators must (1) load weights from DRAM/HBM into compute units, (2) multiply, (3) accumulate. Mythic's approach:

- **Weights are permanently programmed** as analog charge levels into flash cells (multi-level NOR flash stores multiple bits per cell as a continuous charge level).
- When input activation voltages are applied to wordlines, current flows through each cell proportional to both the input voltage and the stored conductance (weight).
- Summation of currents on bitlines computes a **dot product in the analog domain** — one pass through the array, no separate DRAM access.
- ADCs convert the resulting analog current sum back to digital.

This is compute-in-memory (CIM) using the **analog domain** of flash — distinct from digital CIM (SRAM-based, e.g., d-Matrix Corsair) and from photonic CIM (Lightmatter Envise).

---

## AMP Tile — The Fundamental Unit

```
┌──────────────────────────────────────────────────────┐
│                    AMP Tile                          │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  Mythic Analog Compute Engine (ACE™)        │    │
│  │  NOR Flash Array (weight storage)           │    │
│  │  Input DAC: activation → voltage            │    │
│  │  Bitline ADC: current sum → digital result  │    │
│  │  250 GOPS/tile @ 0.25 pJ/MAC               │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ 32-bit RISC-V │  │ SIMD Vector  │                 │
│  │ nano-processor│  │   Engine     │                 │
│  │ (Codasip CPU) │  │              │                 │
│  └──────────────┘  └──────────────┘                 │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │   64 KB      │  │   NoC Router │                 │
│  │    SRAM      │  │  (2D mesh)   │                 │
│  └──────────────┘  └──────────────┘                 │
└──────────────────────────────────────────────────────┘
```

Each tile is a complete, self-contained compute + control unit. The RISC-V nano-processor (licensed from Codasip) manages tile-local execution, SIMD handles non-matrix ops, SRAM buffers intermediate activations, and the NoC router connects to adjacent tiles.

**Key ACE numbers:**
- ~250 GOPS per tile (250 billion MACs/second)
- 0.25 pJ per MAC (extremely low vs. digital: typical digital SRAM MAC ~1–4 pJ)
- Weights stored as multi-level NOR flash charge → no off-chip memory access for weight reads during inference

---

## Chip Hierarchy

```
Mythic AMP Chip
├── Compute tiles (76 or 108, depending on SKU)
│   └── Each tile: ACE (flash MMA) + RISC-V + SIMD + 64 KB SRAM + NoC router
├── Control tiles (4 per chip)
│   └── PCIe 2.0 host interface
│   └── Global scheduling + DMA
├── 2D mesh NoC connecting all tiles
└── No external DRAM required for weight storage
    (all weights live in flash cells of ACE arrays)
```

```
Mythic M1108 (108-tile chip):
  ┌───────────────────────────────┐
  │  Compute Tiles (108)          │
  │  [T][T][T][T] ... [T][T][T]  │  ← 2D mesh, weights in flash
  │  [T][T][T][T] ... [T][T][T]  │
  │         ...                   │
  ├───────────────────────────────┤
  │  Control Tiles (4)            │
  │  [C][C][C][C]                 │  ← PCIe 2.0 host I/F
  └───────────────────────────────┘
  Process: 40nm embedded flash (TSMC or similar)
  Package: 19×19 mm² BGA

Mythic M1076 (76-tile chip):
  Same architecture, 76 compute tiles, smaller die
  Package: smaller BGA
```

---

## Product SKUs

| Product | AMP Tiles | Peak TOPS | Typical Power | Weight Params | Process | Launch |
|---------|-----------|-----------|---------------|---------------|---------|--------|
| M1108 AMP | 108 tiles | 35 TOPS | 4 W | ~100 M | 40nm eFlash | Nov 2020 |
| M1076 AMP | 76 tiles | 25 TOPS | 3 W | ~80 M | 40nm eFlash | Jun 2021 |
| M1076 ×16 PCIe card | 16× M1076 | ~400 TOPS | 75 W (card) | ~1.28 B | 40nm eFlash | 2022 |
| *Next-gen datacenter APU (announced intent only)* | Not disclosed | Vendor claim: 100× GPU energy efficiency; no TOPS figure | Not disclosed | Stated target 1T+ param LLMs | **Not disclosed** | **No date announced** |
| *Honda co-developed automotive SoC (announced 2026-02-04/06)* | Not disclosed | Mythic marketing target: **100,000+ TOPS** (not measured; not datacenter) | Not disclosed | Not disclosed | **Not disclosed** | Prototype **vehicle testing late 2020s / early 2030s** (Mythic target); production after trials |
| *Starlight analog co-processor (name first surfaced 2026-05-19)* | Not disclosed | Not disclosed | Not disclosed | Not disclosed | **Not disclosed** | **No date announced** — sensor-embedded edge part, not datacenter |

> **2026-08-08 correction.** The previous "Advanced node (est.)" entry for the next-gen part was an estimate with no primary source and has been replaced with **not disclosed**. No process node has been published for *any* forthcoming Mythic part. Nothing in the lower three rows is designed, taped out, sampling, or shipping — all are at the "announced" status verb.

---

## Memory Hierarchy

| Level | Technology | Role | Bandwidth | Notes |
|-------|-----------|------|-----------|-------|
| Flash weight storage | Embedded NOR multi-level flash (in ACE) | Permanent weight storage per tile | No DRAM BW needed during compute | Weights read as analog current — no digital load |
| Tile SRAM | 64 KB SRAM per tile | Activation buffers, intermediate results | High on-tile | SW-managed scratchpad |
| Host DRAM | External DRAM (host-side) | Model weight programming (flash write), OS, host activations | Standard host BW | Flash cells written once at model load time |
| *Hybrid platform (2026 direction)* | **Not disclosed** | — | **Not disclosed** | No memory tier (SRAM size, DRAM/HBM attach, or capacity) has been published for any analog+digital Mythic part; a 1T-parameter datacenter target would require an off-chip weight tier that Mythic has not described |

**Key distinction**: Weights are programmed into flash at model-load time and persist indefinitely. During inference, weight access consumes zero DRAM bandwidth — inputs activate flash wordlines; currents flow and accumulate as the MAC result directly.

---

## Interconnect

| Link | Technology | BW | Purpose |
|------|-----------|----|---------| 
| Tile-to-tile | 2D mesh NoC (on-chip) | High (not disclosed) | Activation routing between tiles |
| Chip-to-host | PCIe 2.0 | ~4 GB/s | Host CPU ↔ AMP control, activation I/O |
| Multi-chip | PCIe card (up to 16 chips) | via PCIe fabric | Scale-up on single PCIe card |
| Form factors | M.2 M-key, M.2 A+E-key, PCIe card | varies | Embedded to server deployment |
| *Hybrid analog+digital platform (2026 direction)* | **Not disclosed** | **Not disclosed** | No NoC, die-to-die, or scale-out fabric has been described for a combined analog-CIM + v-MP6000UDX part |

---

## Compute Performance and Efficiency

| Metric | Value | vs. GPU | Notes |
|--------|-------|---------|-------|
| Energy per MAC | 0.25 pJ | ~10× better than digital | Analog current accumulation vs. digital adder tree |
| M1108 single-chip | 35 TOPS @ 4 W | ~8 TOPS/W | Single 40nm chip; no external DRAM power |
| M1076 single-chip | 25 TOPS @ 3 W | ~8 TOPS/W | Same architecture, 76 tiles |
| 16× M1076 card | ~400 TOPS @ 75 W | ~5 TOPS/W (card) | PCIe card; AMP-to-AMP communication overhead |
| Next-gen claim (2025) | 750× tokens/s/W on 1T LLM vs. NVIDIA best | — | **Unverified.** Internal benchmark, unnamed architecture. Re-checked 2026-08-08: **no retrievable source repeats or substantiates this figure**; keep flagged |
| Honda automotive target (2026-02-06) | **100,000+ TOPS** and "100×" energy efficiency | — | **Mythic marketing target for an automotive part that does not exist.** Not measured, not datacenter. Honda's own 2026-02-04 release contains **no** TOPS number and **no** efficiency multiple |
| Videantis-release claim (2026-05-19) | "100× energy efficiency advantage" / "one percent of the energy used by today's state-of-the-art GPUs" | — | Vendor marketing in the acquisition press release; no product, node, workload, or benchmark behind it |

---

## Process and Packaging

- **Process**: 40nm embedded flash (eFlash) — mature node chosen deliberately for NOR flash reliability and cost, not performance.
- **Package**: 19×19 mm² BGA (M1108); smaller BGA (M1076)
- **No HBM, no GDDR, no CoWoS**: entirely self-contained; weights in flash eliminate off-chip DRAM for inference weights.
- **Form factors**: Standalone chip, PCIe M.2 (A+E-key, M-key), PCIe expansion card (up to 16 chips).
- **Forward parts (2026-08-08)**: **no process node is disclosed** for the next-gen datacenter APU, the Honda automotive SoC, or Starlight. The Videantis acquisition release claims dual **US + Europe manufacturing** — a marketing statement with no foundry, node, or site named.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 40nm eFlash (mature node) | NOR flash reliability; cost advantage vs. advanced nodes; dense, non-volatile weight storage |
| Analog domain MAC | ~10× lower energy per operation vs. digital; current summation is physical law, not transistor switching |
| No external DRAM for weights | Eliminates memory wall for weight access; edge deployment without DRAM components |
| RISC-V per-tile | Flexible programmability of non-matrix ops without custom ISA royalties; Codasip licensed CPU |
| SIMD engine per-tile | Handles activation functions, BatchNorm, pooling digitally — analog MAC handles only linear algebra |
| 2D mesh NoC | Scalable to many tiles; avoids global bus bottleneck; compiler-partitions DNN layers across tiles |
| INT8 quantization | Converts FP32 weights to 8-bit charge levels; well within NOR flash analog dynamic range |

---

## Analog Compute Limitations (Documented)

- **Precision**: Analog noise limits effective dynamic range; Mythic uses INT8 quantization (not BF16/FP16). Higher-precision inference requires noise calibration.
- **Weight reprogramming**: NOR flash write cycles (~100K–1M) limit how often models can be swapped. Edge deployment assumes stable model(s).
- **Temperature sensitivity**: NOR flash charge can drift with temperature; requires calibration tables per operating range.
- **Scale**: 40nm process limits transistor density vs. advanced-node digital accelerators. *(Corrected 2026-08-08: the prior claim "next-gen chip targets advanced node" is **not disclosed** — no node has been published for any forthcoming part.)*
- **Supported ops**: Only linear (matrix) ops are computed in analog; all nonlinear activations (ReLU, GELU, Softmax, LayerNorm) are computed digitally by the SIMD engine — same split as Lightmatter. The May 2026 Videantis rationale is an explicit acknowledgement of this limit: the acquisition is framed as buying a *programmable digital backbone* for control, attention, non-max suppression, SLAM, and video codecs that the analog array cannot execute.

---

## 2026 Update — Stated Move to a Hybrid Analog + Digital Platform

*Added 2026-08-08. Change class: roadmap (corporate/IP acquisition), **not** a silicon disclosure.*

### What actually changed

Mythic acquired **Videantis GmbH** (Hannover, Germany) on **2026-05-19**; the transaction has **closed** (Taylor Wessing advisory notice, 2026-06-02, satisfaction of all regulatory requirements). Videantis continues as a **wholly owned subsidiary**; all five founders, led by **Dr. Hans-Joachim Stolberg**, joined Mythic. **Terms not disclosed.** European deep-tech VC **eCAPITAL** became a Mythic shareholder as part of the transaction (cap-table change, not a priced round).

The architectural consequence for this survey: **the repo's pure-analog characterization of Mythic is now incomplete.** Mythic's *stated* forward architecture is analog compute-in-NOR-flash for the matmul **plus a licensed programmable digital core** for everything else. No such part exists, has been named, or has been specified.

### The acquired IP block — v-MP6000UDX

| Property | Value |
|---|---|
| Name | **v-MP6000UDX** "unified processor platform" |
| Type | Licensable digital processor IP (soft IP for chip makers) |
| Microarchitecture | **VLIW + SIMD array of identical cores** |
| Workload coverage | Deep-learning inference, classical computer vision, signal/image processing, **video encode/decode** — all on one architecture |
| Core count / clock / area / node | **Not disclosed** |
| Throughput (TOPS/TFLOPS) | **Not disclosed** |
| Data types | **Not disclosed** |
| Software | Claimed decade-hardened *unified* software stack across all four workload classes |

**Videantis credentials are vendor marketing claims and could not be corroborated outside the press release.** Record them as claims, never as specs: ">25 million chips shipped worldwide" using the IP; "zero field defects across its deployed production base"; "deep penetration across all top 3 European automotive manufacturers"; shipping "inside chips from a top three global semiconductor company"; AEB deployed in 20 mm × 20 mm camera modules; selection in the "top 1% of companies by the European Innovation Council".

### Division of labour as Mythic describes it (vendor framing)

| Function | Executed on |
|---|---|
| Matrix-vector multiply / GEMM | Analog CIM (ACE, NOR flash) |
| Control flow, attention, non-max suppression, SLAM, video codecs | Programmable digital backbone (v-MP6000UDX-derived) |

CEO Dr. Taner Ozcelik: *"we effectively double our architectural advantage and accelerate the development of the hybrid computer the world needs for the AI era."* Stated scaling range, verbatim: *"from single camera drones, to factory robots, to autonomous platforms, all the way to data center deployments."*

### Product names surfaced in the 2026 releases

- **Analog Processing Unit (APU)** — Mythic's current umbrella term for its analog parts (supersedes "AMP" in 2026 copy). Honda R&D **licenses APU technology**.
- **Starlight** — analog co-processors intended for embedding *inside sensors*. Edge, not datacenter. No specifications published.

### Automotive program (Honda) — backfill, predates the 2026-04-05 baseline

Honda's own newsroom (**2026-02-04**, the primary source) states co-development of a **system-on-a-chip for software-defined vehicles** with Mythic, citing neuromorphic / analog compute-in-memory for compute performance and energy efficiency, with **no numbers and no timeline**. Mythic's release (**2026-02-06**) adds that **Honda R&D licenses Mythic's APU technology**, targets **"100×" energy efficiency and 100,000+ TOPS**, and expects **prototype chips for vehicle testing in the late 2020s / early 2030s** with production after successful trials. The quantitative figures are **Mythic marketing targets only** — they are absent from Honda's release and describe a part that does not yet exist.

### What is still not disclosed

No datacenter product name · no process node for any new part · no datacenter TOPS or TDP · no tapeout · no sampling · no shipping · no deployment at scale · no merged SDK, ISA, or compiler for the hybrid platform · no memory hierarchy, NoC, die-to-die, or scale-out fabric for a combined part. The only quantitative performance target found anywhere is the automotive 100,000+ TOPS aspiration. Mythic does not appear on the Hot Chips 38 advance program (Aug 23–25, 2026, Stanford).

### Evidence-quality caveat

Essentially all acquisition coverage is verbatim republication of the same BusinessWire release (edge-ai-vision, pulse2, TMCnet, AFP feed, hw.dev). The only located confirmation independent of both parties is the Taylor Wessing advisory notice, whose page returns **HTTP 403** to direct fetch and was read only via its search-result snippet. No trade-press independent reporting, deal value, headcount, or roadmap detail was found.

### Figure note

`chips/mythic/hw-architecture.dot` was **deliberately not modified**. It depicts shipping M1076/M1108 silicon; the hybrid analog+digital platform has no disclosed engine, memory tier, or fabric to draw, and rendering an announced intention as an architecture block would misrepresent the evidence.

---

## Analog vs. Digital vs. Photonic CIM Comparison

| Aspect | Mythic AMP (Analog Flash) | d-Matrix Corsair (Digital SRAM) | Lightmatter Envise (Photonic) |
|--------|--------------------------|--------------------------------|-------------------------------|
| Compute medium | Analog current (flash cells) | Digital logic in SRAM | Photons (MZI mesh) |
| Weight storage | NOR flash (non-volatile, permanent) | SRAM (volatile, loaded per session) | MZI phase voltages (semi-persistent) |
| Precision | INT8 (analog noise-limited) | INT8/MXINT4 | ABFP16 (custom analog FP) |
| Weight BW during inference | Zero (in-cell analog) | 150 TB/s SRAM | Zero (in-mesh phase states) |
| Process node | 40nm (mature eFlash) | TSMC 6nm | Silicon photonics + 12nm CMOS |
| Target market | Edge AI (cameras, drones, robotics) shipping; automotive SoC co-development with Honda (announced Feb 2026); datacenter LLMs = **stated intent only, no silicon disclosed** | Datacenter inference | Datacenter inference |
| Digital datapath | Per-tile SIMD today; **2026 direction adds a licensed VLIW+SIMD digital core (v-MP6000UDX, Videantis acquisition, May 2026)** — no hybrid part specified | Fully digital by construction | Digital SIMD/vector alongside photonic mesh |
| Non-volatile weights | Yes (flash) | No (SRAM) | Partial (phase drift) |
