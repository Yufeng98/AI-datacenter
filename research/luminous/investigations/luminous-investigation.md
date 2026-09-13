# Luminous Computing — Investigation Report

*investigator: research pipeline*
*as_of: 2026-08-08*
*chip: luminous*
*device_class: Photonic-Electronic Hybrid*
*status: DEFUNCT — operations ceased May 2023; patent portfolio sold to AMD, USPTO-recorded 2026-02-28*

---

## STATUS (verified 2026-08-08): DEFUNCT

Luminous Computing, Inc. is **not an operating company** and has not been one since 2023.

| Item | Value |
|------|-------|
| Status | Defunct / non-operating; formal dissolution **not disclosed** (not confirmable from free public records) |
| Operations ceased | 2023-05 (~50% layoffs, photonics team disbanded); surplus lab assets auctioned 2023-07 |
| Product shipped | None, ever |
| IP — channel 1 | 2023-10 Enosemi *committed commercial license* — **license only, Luminous retained title**. AMD acquired Enosemi **2025-05-28/29**. |
| IP — channel 2 | **2026-02-28** USPTO recordation: Luminous Computing, Inc. → Advanced Micro Devices, Inc., **outright patent assignment**. Verified on US11500410B1, US12379543B2, US12613811B2. No press coverage. |
| SDK / software | Nothing survived; nothing was ever released or open-sourced |
| Web presence | `luminous.com` is a third-party domain-sale listing (Top Notch Domains / Embrace.com), not a company site |

**Two corrections to prior revisions of this report:**
1. The Oct 2023 Enosemi arrangement was a **license, not a transfer of ownership**. Luminous kept title to its patents until Feb 2026 — which is exactly why it was able to sell them to AMD then.
2. AMD acquired Enosemi in **May 2025**, not October 2023. The earlier "Oct 2023" date was wrong.

---

## Investigation Summary

Luminous Computing was an AI hardware company that attempted to build a photonic-electronic hybrid AI supercomputer. It raised ~$123M across three rounds (2018–2022), disbanded its photonics team in May 2023, and licensed its silicon photonics design IP to Enosemi in October 2023 (AMD acquired Enosemi in May 2025). It sold its own patent portfolio outright to AMD with USPTO recordation on 2026-02-28. No chip reached production.

**Confidence levels:**
- Defunct status and 2026-02-28 AMD patent assignment: HIGH (USPTO records on three separate filings)
- May 2025 AMD/Enosemi acquisition date: HIGH (Electronics Weekly 2025-05-29, quoting AMD SVP Brian Amick)
- Company history, funding, pivot, wind-down: HIGH (multiple sources)
- Architecture direction (interconnect-at-every-scale): HIGH (NextPlatform 2022, VentureBeat 2022)
- Memory type (DDR, no HBM): HIGH (Enosemi/Luminous press release, Oct 2023)
- Gen 1/1.X/2 product generations: MEDIUM (single press release source; product never shipped)
- Software stack: LOW (no public artifacts; claims only)
- Chip-level specs (process node, die count, performance): NOT FOUND

---

## Key Findings by Layer

### Hardware

**Compute Engine**
- CMOS ASIC performs all compute (matrix ops, activations)
- Pre-pivot (2018–2020): also targeted photonic MVM using MZI meshes
- CTO Nahmias IEEE JSTQE 2020 paper: photonic MACs superior to electronic in energy/speed/density
- Process node: not public
- Performance: 3,000× TPU v3 board claimed (pre-pivot, unverified, never shipped)
- Patent: US20200284984A1 — "System for photonic computing," Nahmias → Luminous Computing

**Data Path**
- Silicon photonic WDM interconnect at every scale: die-to-memory, chip-to-chip, board-to-board, rack-to-rack
- Claimed 10–100× bandwidth improvement vs electrical at each scale
- Distance-independent bandwidth (optical vs copper degradation with length)
- SOI waveguides, WDM multiplexing, electro-optic modulators

**Off-chip Memory** (**CONFIRMED: DDR, no HBM**)
- Multi-terabyte DDR memory banks (confirmed Enosemi press release Oct 2023)
- No HBM used: "accomplished without using High Bandwidth Memory"
- Gen 1 card: capacity not specified
- Gen 1.X card: 2 TB DDR (planned, never shipped)
- Gen 2 card: with networking (planned, never shipped)
- Architecture rationale: photonic links provide HBM-equivalent bandwidth to distant DDR

**Scale-up Interconnect**
- Silicon photonic WDM chip-to-chip links (SOI waveguides + electro-optic modulators)
- WDM: multiple wavelength channels per waveguide
- 10–100× bandwidth vs electrical SerDes (claimed, no spec published)

**Scale-out Interconnect**
- Rack-to-rack photonic fiber WDM (described qualitatively, never shipped)
- Gen 2 card planned with networking for large-scale applications

### Software

**All layers: not public.** The only confirmed software claim is "native PyTorch integration" for Gen 1 inference cards (Aware partnership announcement, Sep 2023; Enosemi press release, Oct 2023). No compiler, runtime, SDK, driver, or kernel library was ever released.

---

## IP Disposition

| IP Component | Route | Current Owner |
|-------------|-------|---------------|
| Silicon photonics design libraries | Luminous → Enosemi (**license only**, Oct 2023) → AMD (Enosemi acquired **2025-05-28/29**) | Licensed rights at AMD; Luminous held title until Feb 2026 |
| Electro-optic modulator designs | Same route | Same |
| WDM component designs | Same route | Same |
| PDK elements (GlobalFoundries SiPh process) | Same route | Same |
| US11500410B1 — parallel photonic computation | Outright assignment, USPTO recorded **2026-02-28** | **Advanced Micro Devices, Inc.** |
| US12379543B2 — photonic IC system and fabrication | Outright assignment, USPTO recorded **2026-02-28** | **Advanced Micro Devices, Inc.** |
| US12613811B2 — disaggregated memory / high-BW interconnect architecture (granted 2026-04-28) | Outright assignment, USPTO recorded **2026-02-28** | **Advanced Micro Devices, Inc.** |
| Patent application US20200284984A1 (photonic computing system) | Whether it was included in the Feb 2026 assignment is **not disclosed** — not individually verified | Not disclosed |
| Software / SDK / compiler source | Never released, never open-sourced, never transferred | Does not exist publicly |

---

## Sources Consulted

| Priority | Source | URL |
|----------|--------|-----|
| PRIMARY | USPTO/Google Patents US11500410B1 — Luminous → AMD, recorded 2026-02-28 | https://patents.google.com/patent/US11500410B1/en |
| PRIMARY | USPTO/Google Patents US12379543B2 — Luminous → AMD, recorded 2026-02-28 | https://patents.google.com/patent/US12379543B2/en |
| PRIMARY | USPTO/Google Patents US12613811B2 — Luminous → AMD, recorded 2026-02-28; granted 2026-04-28 | https://patents.google.com/patent/US12613811B2/en |
| HIGH | Electronics Weekly: AMD buys Enosemi — establishes **May 2025** acquisition date | https://www.electronicsweekly.com/news/business/amd-buys-enosemi-2025-05/ |
| HIGH | Enosemi/Luminous press release (Oct 2023) — committed commercial **license** | https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html |
| HIGH | Enosemi launch announcement | https://www.enosemi.com/news/2023-10-12_enosemi_luminous_press_release/ |
| HIGH | luminous.com — third-party domain-sale listing, retrieved 2026-08-08 (company no longer controls its domain) | https://www.luminous.com/ |
| NEGATIVE | Searches for press coverage of the AMD/Luminous Feb 2026 patent acquisition returned zero results — USPTO records are the sole evidence | — |
| HIGH | NextPlatform: Optical Architecture (Mar 2022) | https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/ |
| HIGH | VentureBeat: $105M Series A (Mar 2022) | https://venturebeat.com/ai/luminous-computing-which-is-developing-a-light-based-ai-accelerator-chip-raises-105m |
| HIGH | Nahmias IEEE JSTQE 2020 | https://ieeexplore.ieee.org/ielaam/2944/8764697/8844098-aam.pdf |
| MEDIUM | Aware + Luminous partnership (Sep 2023) | https://www.businesswire.com/news/home/20230921189058/en/Aware-and-Luminous-Accelerate-Adoption-of-Generative-AI-Across-Secure-Enterprises |
| MEDIUM | TechCrunch: Photonics tough nut (Apr 2023) | https://techcrunch.com/2023/04/21/light-powered-ai-chips-future/ |
| MEDIUM | Teamblind: company dead discussion | https://www.teamblind.com/post/Luminous-Computing-is-dead-dhKjvYd4 |
| MEDIUM | SVDisposition surplus assets auction (Jul 2023) | https://www.svdisposition.com/auction-detail?id=594 |
| LOW | Patent US20200284984A1 | https://patents.google.com/patent/US20200284984A1/en |
| LOW | MIT Technology Review 2019 | https://www.technologyreview.com/2019/06/13/867/ai-chips-uses-optical-semiconductor-machine-learning/ |
| LOW | Optics.org: Hochberg joins (2021) | https://optics.org/news/13/3/36 |
