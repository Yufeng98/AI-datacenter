# Luminous Computing — Summary

*as_of: 2026-08-08*
*chip: luminous*
*device_class: Photonic-Electronic Hybrid*
*status: DEFUNCT — operations wound down May 2023; residual patent portfolio sold to AMD, USPTO-recorded 2026-02-28*

---

## STATUS: DEFUNCT (verified 2026-08-08)

> **Luminous Computing, Inc. is not an operating company and has not been one since 2023.**
> It is retained in this survey as an architectural data point only. Nothing in this entry
> describes a product, vendor, or roadmap a reader can engage with today.

| Item | Value |
|------|-------|
| Current status | Defunct / non-operating. Formal dissolution is **not disclosed** (not confirmable from free public records). |
| Date operations ceased | **2023-05** — ~50% layoffs, entire photonics team disbanded; surplus lab assets auctioned 2023-07. |
| Product ever shipped | **None.** ~$123M raised (2018–2022); no silicon reached production or customers. |
| Fate of the IP — channel 1 (license) | **2023-10:** Enosemi launched with a *committed commercial license* to Luminous silicon-photonics design IP. This was a **license, not a transfer of ownership** — Luminous retained title to its patents. AMD then acquired **Enosemi on 2025-05-28/29** (not Oct 2023, as an earlier revision of this entry incorrectly stated). |
| Fate of the IP — channel 2 (outright sale) | **2026-02-28 (USPTO recordation date):** Luminous Computing, Inc. assigned its own patent portfolio **outright to Advanced Micro Devices, Inc.** Verified on three filings: US11500410B1, US12379543B2, US12613811B2. The transaction received **no press coverage**; USPTO assignment records are the sole evidence. |
| Fate of the SDK / software | **Nothing survived.** No compiler, runtime, SDK, or driver was ever released, open-sourced, or transferred. The claimed "native PyTorch integration" for the Gen 1 inference card never became public in any form. |
| Corporate web presence | **Gone.** `luminous.com` is no longer company-controlled — it is a parked broker listing ("Luminous.com - Domain Name for Sale", Top Notch Domains, LLC via Embrace.com, stated "7 figure USD valuation"). |
| Residual legal existence | The entity retained enough legal existence to prosecute patents into 2026 (US12613811B2 granted 2026-04-28) and to execute the Feb 2026 assignment. The Feb 2026 patent sale plus the domain sale are consistent with final liquidation of the estate in H1 2026. |

**Primary sources for this status block:**

- [US11500410B1 — "System and method for parallel photonic computation"](https://patents.google.com/patent/US11500410B1/en) — assignment Luminous Computing, Inc. → Advanced Micro Devices, Inc., recorded **2026-02-28**.
- [US12379543B2 — "Photonic integrated circuit system and method of fabrication"](https://patents.google.com/patent/US12379543B2/en) — same assignee change, recorded 2026-02-28.
- [US12613811B2 — "Computer architecture with disaggregated memory and high-bandwidth communication interconnects"](https://patents.google.com/patent/US12613811B2/en) — granted 2026-04-28; same assignee change recorded 2026-02-28. Three filings establish a portfolio-level sale, not a one-off.
- [Electronics Weekly: "AMD buys Enosemi" (2025-05-29)](https://www.electronicsweekly.com/news/business/amd-buys-enosemi-2025-05/) — establishes the **May 2025** Enosemi acquisition date and that Enosemi *licensed* its technology from Luminous.
- [luminous.com](https://www.luminous.com/) — retrieved 2026-08-08: broker domain-sale listing, not a company site.

**Negative evidence:** searches for press coverage of the AMD/Luminous 2026 patent acquisition returned zero results. The transaction appears entirely unreported outside USPTO records.

---

## One-Line Summary

Luminous Computing was a Mountain View photonic-electronic hybrid startup (2018–2023) that aimed to solve AI interconnect bottlenecks at every scale using silicon photonic WDM links; it raised ~$123M, never shipped a product, disbanded its photonics team in May 2023, licensed its silicon-photonics design IP to Enosemi in Oct 2023 (AMD acquired Enosemi in May 2025), and finally sold its own patent portfolio outright to AMD in Feb 2026.

---

## What They Built (Intended)

Luminous's architecture inserted silicon photonic waveguide interconnects at every level of an AI supercomputer — from processor-to-memory up through rack-to-rack — claiming 10–100× bandwidth improvement at each scale vs. electrical alternatives. The CMOS compute die handled all actual matrix operations electronically; photonics was exclusively the data movement fabric.

The founding vision (2018–2020) also included photonic matrix-vector multiply (MVM) using Mach-Zehnder interferometer meshes, based on CTO Mitchell Nahmias's Princeton neuromorphic photonics research (IEEE JSTQE 2020). This was deprioritized after the seed round in favor of the interconnect-only approach.

---

## Key Technical Claims (Unverified — Never Shipped)

All figures below were company marketing claims made while Luminous was operating. None was independently confirmed, and no silicon ever shipped against them.

| Metric | Claim | Status |
|--------|-------|--------|
| Performance vs TPU v3 board | 3,000× | Company claim; unverified; pre-pivot |
| Bandwidth improvement | 10–100× at every scale | Company claim; unverified; no specs public |
| Memory per card | 12–25× vs competitors at HBM BW | Company claim; unverified; no product shipped |
| Process node | Not disclosed | — |

---

## Company Timeline

| Date | Event |
|------|-------|
| 2018 | Founded; $1M pre-seed |
| 2019 | $9M seed; early photonic MAC compute research |
| 2020 | Nahmias IEEE JSTQE paper on photonic MACs |
| 2021 | Michael Hochberg (silicon photonics pioneer) joined; pivot from compute to interconnect |
| 2022-03 | $105M Series A; "world's most powerful AI supercomputer" announcement |
| 2023-05 | ~50% layoffs; photonics team disbanded; **operations wound down** |
| 2023-07 | Surplus lab assets auctioned (SVDisposition) |
| 2023-10 | Enosemi launched with a **committed commercial license** to Luminous silicon photonics design IP (license only — Luminous retained title) |
| **2025-05-28/29** | **AMD acquired Enosemi** (16 employees) for its co-packaged optics program — corrects the earlier "Oct 2023" error in this entry |
| **2026-02-28** | **Luminous Computing, Inc. assigned its patent portfolio outright to Advanced Micro Devices, Inc.** (USPTO recordation date; unreported in press) |
| 2026-04-28 | US12613811B2 granted, already showing AMD as assignee — last observed activity connected to the estate |
| 2026-08-08 | `luminous.com` observed as a third-party domain-sale listing; company web presence gone |

---

## Software Stack

**Nothing public was ever released.** The company wound down before productizing any compiler, runtime, SDK, or driver. A "native PyTorch integration" was claimed for a Gen 1 inference card that was never shipped publicly. Nothing was open-sourced, and no software was transferred in either IP transaction — both concerned photonic *design* IP and patents, not a software stack.

---

## What Survived (Downstream)

Two distinct channels carried Luminous IP into AMD:

1. **License channel (2023-10 → 2025-05).** Enosemi built its silicon-photonics business on a committed commercial license to Luminous's design IP — waveguide libraries, electro-optic modulator designs, WDM components, PDK elements. AMD acquired Enosemi in **May 2025** for its co-packaged optics (CPO) program, so that licensed IP reached AMD indirectly and second-hand. Luminous never transferred ownership through this channel.
2. **Direct sale channel (2026-02-28).** Luminous sold its own patents outright to AMD, recorded at USPTO on 2026-02-28. The subject matter spans parallel photonic computation, photonic integrated circuit fabrication, and disaggregated-memory system architecture — i.e. the pre-pivot photonic *compute* work as well as the interconnect work.

Whether any Luminous-derived design is present in shipping AMD CPO silicon is **not disclosed**. AMD has made no public statement connecting the Feb 2026 patent purchase to any product.

---

## Comparison to Lightmatter

Both Luminous and Lightmatter were photonic-electronic hybrids targeting AI, but with opposite orientations:

| Aspect | Luminous (defunct) | Lightmatter |
|--------|----------|-------------|
| Photonics role | Interconnect only | Compute (MZI matmul) + Interconnect (Passage) |
| Electronic role | All compute | ADC/DAC, activation, control |
| Capital raised | ~$123M | ~$850M |
| Product shipped | None | Envise (Nature 2025), Passage M1000/L200 |
| Status (2026-08) | **Defunct; patents sold to AMD Feb 2026** | Active, scaling |
| IP outcome | License → Enosemi → AMD (May 2025); patents → AMD (Feb 2026) | Proprietary |

Lightmatter had ~7× more capital, solved the analog noise problem with ABFP16, and published peer-reviewed results in Nature. Luminous ran out of runway before solving the same fundamental challenges.

---

## Why This Entry Is Retained

Luminous is the clearest case in this survey of an architecture outliving its company: the design IP was valuable enough to license in 2023 and the patents valuable enough for AMD to buy outright in 2026, while the company itself never shipped a unit. The thesis — that optical links let you hang multi-terabyte DDR off a compute die and skip HBM entirely — was never falsified by a product. It was defeated by capital intensity, silicon-photonics manufacturing immaturity, and the rapid improvement of electrical NVLink/NVSwitch bandwidth that removed the urgency. See `hw-architecture.md` for the architecture and the wind-down analysis.

---

## Resources

- [NextPlatform: Luminous Optical Architecture (Mar 2022)](https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/)
- [VentureBeat: $105M Series A (Mar 2022)](https://venturebeat.com/ai/luminous-computing-which-is-developing-a-light-based-ai-accelerator-chip-raises-105m)
- [Enosemi launch with committed commercial license to Luminous IP (Oct 2023)](https://www.enosemi.com/news/2023-10-12_enosemi_luminous_press_release/)
- [Electronics Weekly: AMD buys Enosemi (May 2025)](https://www.electronicsweekly.com/news/business/amd-buys-enosemi-2025-05/)
- [Google Patents / USPTO: US11500410B1 — Luminous → AMD, recorded 2026-02-28](https://patents.google.com/patent/US11500410B1/en)
- [Google Patents / USPTO: US12379543B2 — Luminous → AMD, recorded 2026-02-28](https://patents.google.com/patent/US12379543B2/en)
- [Google Patents / USPTO: US12613811B2 — Luminous → AMD, recorded 2026-02-28; granted 2026-04-28](https://patents.google.com/patent/US12613811B2/en)
- [Nahmias IEEE JSTQE 2020 paper](https://ieeexplore.ieee.org/ielaam/2944/8764697/8844098-aam.pdf)
- [TechCrunch: Photonics tough nut to crack (Apr 2023)](https://techcrunch.com/2023/04/21/light-powered-ai-chips-future/)
- [SVDisposition: Surplus assets auction (Jul 2023)](https://www.svdisposition.com/auction-detail?id=594)
- [luminous.com — domain-sale broker listing (retrieved 2026-08-08)](https://www.luminous.com/)
