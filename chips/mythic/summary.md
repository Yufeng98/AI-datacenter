# Mythic AMP — Research Summary

*chip: mythic*
*device_class: Analog In-Memory Compute*
*as_of: 2026-04-05 · last updated: 2026-08-08*

---

## Company Status Correction

The research brief stated "Mythic shut down in 2024." This is **incorrect**. Accurate history:

- Mythic (founded 2012, originally Isocline, co-founders Dave Fick and Mike Henry, University of Michigan) ran out of operating cash in **November 2022** and nearly shut down.
- Rescued by a **$13 M bridge round in March 2023** (Atreides, DCVC, Lux Capital); Dave Fick became CEO.
- **June 2024**: Named NVIDIA veteran Dr. Taner Ozcelik as CEO to scale commercialization.
- **December 2025**: Raised **$125 M** oversubscribed round led by DCVC, joined by NEA, SoftBank KR, Honda Motor, Lockheed Martin, Atreides, and others.
- **February 2026**: Honda and Mythic announced **joint development** of an analog-AI SoC for software-defined vehicles (Honda newsroom 2026-02-04; Mythic release 2026-02-06). *Missed by the 2026-04-05 baseline scan — see the update section below.*
- **May 2026**: Acquired **Videantis GmbH** (Hannover, Germany), a licensable digital processor-IP vendor — Mythic's forward architecture is now stated as **analog CIM + a programmable digital core**, not pure analog. *See the update section below.*
- Company as of August 2026 is **active**, stating an intent to move from edge AI to **datacenter-scale 1T-parameter LLM inference**, claiming 100× GPU energy efficiency and 750× tokens/s/W on 1T models (internal benchmark; **unverified — no retrievable source substantiates the 750× figure as of 2026-08-08**). No datacenter product name, process node, TOPS, TDP, tapeout, or sampling has been disclosed; the pivot remains a stated intention with no silicon disclosure behind it.
- Total funding raised: ~$290 M across 9+ rounds. No priced round after the Dec 2025 $125 M, though European deep-tech VC **eCAPITAL** became a Mythic shareholder as part of the Videantis transaction (cap-table change, not a funding round).

---

## What Was Built: Products Shipped

### M1108 AMP (November 2020)
- Industry's first commercial analog matrix processor.
- 108 AMP tiles, 35 TOPS, 4 W typical, 40nm embedded NOR flash.
- 19×19 mm² BGA. PCIe M.2 form factor. Up to ~100 M weight parameters on-chip.

### M1076 AMP (June 2021)
- Optimized smaller SKU: 76 AMP tiles, 25 TOPS, 3 W, same 40nm eFlash process.
- Form factors: standalone chip, PCIe M.2 A+E-key, PCIe M.2 M-key.
- 16× M1076 PCIe card: ~400 TOPS @ 75 W for higher-performance edge.

### Shipping in 2023–2024
- Commercial products shipped to customers in edge AI verticals: surveillance cameras, drones, robotics, machine vision, automotive ADAS.

---

## Core Technology

**Analog compute-in-NOR-flash**: Weights are programmed as analog charge levels into NOR flash cells. When input voltages are applied to wordlines, cell currents flow proportional to both input and stored weight conductance; bitline current summation computes the dot product **in the analog domain** — achieving MAC operations with zero DRAM bandwidth and extremely low energy (0.25 pJ/MAC vs. ~1–4 pJ/MAC for digital SRAM-based accelerators).

This is distinct from:
- **Digital CIM** (d-Matrix Corsair): SRAM-based, digital MAC
- **Photonic CIM** (Lightmatter Envise): optical MZI mesh
- **PIM in DRAM** (SK Hynix AiMX, Samsung Aquabolt): compute added to HBM/GDDR

Key architectural properties:
- Non-volatile weight storage (flash persists across power cycles)
- No external DRAM required for inference weight access
- Limited to INT8 precision (analog noise floor)
- Nonlinear ops (ReLU, GELU, Softmax, LayerNorm) executed digitally by per-tile SIMD
- 40nm mature process (not cutting-edge) — chosen for eFlash reliability and cost

---

## Software Stack

The **MAPP SDK** (closed-source, Python) handles:
1. Import from PyTorch / TensorFlow / ONNX / TensorRT
2. **Mythic Optimization Suite**: INT8/INT4 post-training quantization with analog range calibration
3. **Mythic Graph Compiler**: tile partitioning, weight packing to flash addresses, NoC activation routing, per-tile RISC-V codegen, SIMD scheduling, cycle-accurate simulation
4. **MAPP Runtime**: one-time flash programming at model-load, PCIe activation transfer, tile orchestration, ADC result collection

For the shipping M1076/M1108 products, no user-visible ISA or kernel API exists. The analog ACE has no programmable instruction set — weights are conductance states, inputs are voltages, outputs are ADC-converted currents.

> **Hedge added 2026-08-08.** The acquired Videantis **v-MP6000UDX** is a licensable *programmable* VLIW+SIMD processor that ships with its own toolchain, so a future hybrid Mythic part would plausibly expose a digital programming interface. **Mythic has published nothing about a merged SDK, ISA, or compiler**, so the developer-facing story is unchanged today. Do not read the acquisition as evidence that a user-visible ISA now exists.

---

## Key Numbers

| Metric | M1076 | M1108 | Notes |
|--------|-------|-------|-------|
| AMP tiles | 76 | 108 | Each tile: ACE + RISC-V + SIMD + 64KB SRAM + NoC |
| Peak TOPS | 25 | 35 | Single chip |
| Typical power | 3 W | 4 W | During inference |
| Energy/MAC | 0.25 pJ | 0.25 pJ | ~10× better than digital |
| Weight capacity | ~80 M params | ~100 M params | Stored in flash |
| Off-chip DRAM | None | None | No DRAM required |
| Process | 40nm eFlash | 40nm eFlash | Mature node |
| Precision | INT8 / INT4 | INT8 / INT4 | Analog noise-limited |
| Host interface | PCIe 2.0 | PCIe 2.0 | ~4 GB/s |

---

## Corporate & Roadmap Update (Feb–May 2026)

*Added 2026-08-08. Change class: roadmap (corporate/IP), not silicon. Sources: Videantis/BusinessWire acquisition release 2026-05-19; Taylor Wessing advisory notice 2026-06-02; Honda global newsroom 2026-02-04; Mythic release 2026-02-06; mythic.ai news index; Hot Chips 2026 advance program.*

**Bottom line: two corporate moves, zero new silicon.** Mythic has still disclosed no datacenter product name, no process node for any new part, no datacenter TOPS or TDP, no tapeout, no sampling, no shipping, and no deployment. Everything below is at the "announced" status verb. Mythic's own news index carries exactly three items between Dec 2025 and 2026-08-08 (the $125 M round, Honda, Videantis) and nothing after 2026-05-19. Mythic does **not** appear on the Hot Chips 38 advance program (Aug 23–25, 2026, Stanford).

### 1. Videantis acquisition — announced 2026-05-19, closed

| Item | Detail |
|---|---|
| Target | **Videantis GmbH**, Hannover, Germany — licensable digital processor-IP vendor |
| Announced | 2026-05-19 (BusinessWire; Palo Alto CA / Hannover DE dateline) |
| Terms | **Not disclosed** (no deal value, no headcount figure) |
| Structure | Continues as a **wholly owned subsidiary**; all five founders, led by **Dr. Hans-Joachim Stolberg**, joined Mythic |
| Cap table | European deep-tech VC **eCAPITAL became a Mythic shareholder** as part of the transaction — an equity change, **not** a priced funding round |
| Closing | Taylor Wessing (Mythic's counsel) states on 2026-06-02 that the transaction was "successfully completed … following the satisfaction of all regulatory requirements" — a completed acquisition, not a pending LOI |

**The acquired asset — v-MP6000UDX.** A "unified processor platform": a **VLIW + SIMD array of identical cores** covering deep-learning inference, classical computer vision, signal/image processing, and video encode/decode on a single architecture, with a claimed decade-hardened unified software stack.

**Videantis credentials are vendor marketing claims, independently unverified** — none could be corroborated outside the press release: ">25 million chips shipped worldwide" using the IP, "zero field defects across its deployed production base", "deep penetration across all top 3 European automotive manufacturers", shipping "inside chips from a top three global semiconductor company", AEB deployed in 20 mm × 20 mm camera modules, and selection in the "top 1% of companies by the European Innovation Council".

**Strategic framing (vendor).** Analog CIM for the matmul plus a programmable unified digital backbone for control, attention, non-max suppression, SLAM, and codecs. CEO Dr. Taner Ozcelik: *"we effectively double our architectural advantage and accelerate the development of the hybrid computer the world needs for the AI era."* Stated scaling range, verbatim: *"from single camera drones, to factory robots, to autonomous platforms, all the way to data center deployments."* The release also claims a "100× energy efficiency advantage" / "one percent of the energy used by today's state-of-the-art GPUs" and dual US + Europe manufacturing — **all unsubstantiated marketing, with no product, node, or benchmark behind them.**

### 2. Honda joint development — announced 2026-02-04 / 2026-02-06 (backfill)

This **predates the repo's own 2026-04-05 baseline scan and was simply missed** — it is a gap-fill, not news from the Apr–Aug 2026 window. Only the Videantis acquisition is genuinely new since baseline.

- **Honda global newsroom, 2026-02-04** (primary): Honda will co-develop a **system-on-a-chip for software-defined vehicles** with Mythic (which Honda describes as "a Texas, U.S.-based technology company"), citing neuromorphic / analog compute-in-memory for compute performance and energy efficiency. **Honda's release states no numbers and no timeline.**
- **Mythic release, 2026-02-06**: **Honda R&D licenses Mythic's Analog Processing Unit (APU) technology**; targets **"100×" energy efficiency and 100,000+ TOPS**; **prototype chips for vehicle testing in the late 2020s / early 2030s**, production after successful trials.
- **Attribution matters**: the 100× and 100,000+ TOPS figures come **solely from Mythic's marketing copy**, are forward-looking automotive targets rather than datacenter or measured-silicon numbers, and appear nowhere in Honda's own release.
- Honda is now **both** a Dec-2025 investor **and** a co-development partner — the repo previously recorded only the investor relationship.
- **HQ inconsistency to flag**: Honda's newsroom calls Mythic "a Texas, U.S.-based technology company"; Mythic's own May 2026 dateline reads "Palo Alto, CA". The repo records Austin, TX. Both Austin and Palo Alto appear in Mythic materials; treat HQ as dual-site / unresolved.

### 3. Product names newly surfaced (absent from the prior repo snapshot)

- **Analog Processing Unit (APU)** — Mythic's current umbrella term for its analog parts, superseding "AMP" in 2026 marketing copy.
- **Starlight** — analog co-processors intended for embedding *in sensors*. Edge, **not** datacenter. No specs published.

### 4. Survey implications

1. **The pure-analog characterization is now incomplete.** Mythic's stated forward architecture is analog CIM **plus** an acquired programmable VLIW/SIMD digital core. No hybrid part has been designed, taped out, or specified.
2. **The programming-model line is hedged, not rewritten** (see "Software Stack" above). Nothing about a merged toolchain has been published.
3. **The 750× tokens/s/W on 1T models claim remains unsubstantiated** by any retrievable source and stays flagged as unverified.

### 5. Evidence-quality caveat

Essentially all acquisition coverage is verbatim republication of the same BusinessWire release (edge-ai-vision, pulse2, TMCnet, AFP wire feed, hw.dev). The **only genuinely independent confirmation located** is the Taylor Wessing advisory notice confirming closing — and that page returns HTTP 403 to direct fetch and was read only via its search-result snippet. **No trade-press independent reporting, deal value, headcount, or roadmap detail was found.**

---

## Deliverables

| File | Contents |
|------|----------|
| `chips/mythic/hw-architecture.md` | Full hardware architecture documentation |
| `chips/mythic/hw-architecture.yaml` | Machine-readable hw architecture |
| `chips/mythic/software-stack.md` | Full software stack documentation |
| `chips/mythic/software-stack.yaml` | Machine-readable software stack |
| `chips/mythic/layer-table.md` | Per-layer comparison table row |
| `chips/mythic/hw-architecture.dot` | Graphviz architecture diagram source (filename corrected 2026-08-08; unchanged by this update — see figure note in `hw-architecture.md`) |
| `chips/mythic/hw-architecture.png` | Rendered architecture diagram (150 DPI) |
| `research/mythic/search-results.md` | Resource index (created 2026-08-08) |
| `chips/mythic/summary.md` | This file |
