# Luminous Computing — Software Stack & Hardware Resources

*as_of: 2026-08-08*
*device_class: Photonic-Electronic Hybrid*
*status: DEFUNCT — see status summary below*
*seeds: https://nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/ (note: the original seed https://www.luminous.com is no longer a company site — it is a domain-sale listing as of 2026-08-08)*

---

## Company Status Summary (verified 2026-08-08): DEFUNCT

Luminous Computing (founded 2018, Mountain View CA) raised a total of ~$123M across three rounds (pre-seed $1M in 2018, seed $9M in 2019, Series A $105M in March 2022). Key investors: Bill Gates, Gigafund, 8090 Partners, Neo, Third Kind Venture Capital, Alumni Ventures Group. It never shipped a product.

In **May 2023** the company laid off ~50% of staff including the entire photonics team and wound down operations; surplus assets were auctioned by Silicon Valley Disposition in July 2023. In **October 2023** Enosemi launched with a *committed commercial license* to Luminous's silicon photonics design IP — a **license, not a sale**; Luminous retained title to its patents. **AMD acquired Enosemi on 2025-05-28/29** (not October 2023, as an earlier revision of this file implied). Separately, **Luminous Computing, Inc. assigned its own patent portfolio outright to Advanced Micro Devices, Inc. with USPTO recordation date 2026-02-28** (verified on US11500410B1, US12379543B2, US12613811B2); this transaction received no press coverage.

The company is **defunct**. Formal dissolution is **not disclosed** — it is not confirmable from free public records — but the Feb 2026 patent sale, the fact that `luminous.com` is now a third-party domain-sale listing, and the absence of any operations since 2023 are consistent with final liquidation of the estate in H1 2026. All links below are historical records, not live vendor resources.

**Co-founders:** Marcus Gomez (CEO), Mitchell Nahmias (CTO — Princeton neuromorphic photonics PhD), Michael Hochberg (silicon photonics pioneer, joined 2021).

---

## Software Stack

### Framework Integration
- [Aware + Luminous partnership (BusinessWire, Sep 2023)](https://www.businesswire.com/news/home/20230921189058/en/Aware-and-Luminous-Accelerate-Adoption-of-Generative-AI-Across-Secure-Enterprises) — confirms "native PyTorch integration" for Gen 1 AI inference cards (strongest sourced claim)
- [Enosemi/Luminous press release (PRWeb, Oct 2023)](https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html) — "Gen 1 AI inference cards offer native PyTorch integration"; "compute and memory bandwidth in-line with or better than competing hardware"; confirms multi-TB DDR without HBM
- [Luminous Computing LinkedIn](https://www.linkedin.com/company/luminous-computing) — Gen 1 AI inference cards; no public SDK released
- ~~[Luminous Computing Website](https://www.luminous.com)~~ — **DEAD as a company site.** Historically referenced "generative AI inference" hardware with no public software stack; as of 2026-08-08 the domain is a third-party sale listing (Top Notch Domains, LLC via Embrace.com)

### Compiler / IR
- *Not public* — no public compiler toolchain was released before the 2023 wind-down; weight-to-phase-angle compilation for photonic MZI meshes was in internal development

### Op Library
- *Not public* — photonic matrix-vector multiply (the core op) executed in the optical domain; no public op library

### Kernel Library
- *Not public*

### Runtime
- *Not public* — runtime architecture not disclosed; limited evidence it was ever productized

### Driver / Firmware
- *Not public* — chip-to-chip optical interconnect firmware not disclosed

### Communication
- *Not public* — communication layer for photonic interconnect not publicly specified

### Assembler / ISA
- *Not public* — photonic hardware has no conventional ISA; MZI phase programming would have been the "instruction" primitive; no public ISA specification

---

## Hardware Architecture

### Compute Engine
- [NextPlatform: Luminous Shines A Light On Optical Architecture](https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/) — core architecture: digital chips optically connected; photonic interconnect at every hierarchy level (chip-to-chip, board-to-board, rack-to-rack); originally also pursued photonic MVM compute
- [VentureBeat Series A](https://venturebeat.com/ai/luminous-computing-which-is-developing-a-light-based-ai-accelerator-chip-raises-105m) — claimed 3,000× performance over TPU v3 board; "aiming to build the whole computer"
- [Pulse2: How Luminous Computing Plans To Use Light](https://pulse2.com/luminous-computing-light-ai-faster-and-more-efficient/) — Mitchell Nahmias research: photonic multiply-accumulate (MAC) ops; IEEE JSTQE paper "Photonic Multiply-Accumulate Operations for Neural Networks" (2020); neuromorphic photonics foundation
- [Photonic MAC paper — IEEE JSTQE 2020 (Nahmias et al.)](https://ieeexplore.ieee.org/ielaam/2944/8764697/8844098-aam.pdf) — academic basis for Luminous's photonic compute architecture; photonic hardware significantly outperforms electronic in energy, speed, and compute density for neural networks

### Data Path
- [NextPlatform 2022](https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/) — photonic interconnect inserted at every communication bottleneck: memory↔processor, board↔board, box↔box, rack↔rack; bandwidth 10-100× improvement at every scale
- [Bill Gates / MIT Technology Review 2019](https://www.technologyreview.com/2019/06/13/867/ai-chips-uses-optical-semiconductor-machine-learning/) — early description: optical semiconductor using light to process AI data; silicon photonics waveguides

### On-chip Memory
- *Not public* — no disclosed SRAM or on-chip memory specification

### Off-chip Memory
- [Enosemi/Luminous press release (Oct 2023)](https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html) — **DDR memory, no HBM**: "proprietary compute and memory architecture enables high bandwidth access to multi-Terabyte banks of DDR memory"; Gen 1 card, Gen 1.X card (2 TB of memory), Gen 2 card (with networking); "accomplished without using High Bandwidth Memory"
- [VentureBeat / SiliconAngle 2022](https://siliconangle.com/2022/03/03/luminous-computing-raises-105m-build-photonics-powered-ai-supercomputer/) — photonic interconnect enables access across a "much larger memory space"; DRAM type/capacity not public at time

### Host Interface / Package
- [Enosemi/Luminous press release (Oct 2023)](https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html) — Gen 2 card described as having "networking" capability; PCIe form factor implied; die count and process node not public
- [Luminous Computing patent US20200284984A1](https://patents.google.com/patent/US20200284984A1/en) — system for photonic computing; inventor Mitchell A. Nahmias, assigned to Luminous Computing Inc.; input module, computation module, control module with filter banks and detectors

### Scale-up Interconnect
- [NextPlatform 2022](https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/) — silicon photonics chip-to-chip optical links; WDM (wavelength division multiplexing) channels; optical waveguides integrated into package; 10–100× BW at every scale
- [VentureBeat 2022](https://venturebeat.com/ai/luminous-computing-which-is-developing-a-light-based-ai-accelerator-chip-raises-105m) — "increasing bandwidth by 10 times to 100 times at every distance scale"

### Scale-out Interconnect
- [NextPlatform 2022](https://www.nextplatform.com/2022/03/17/luminous-shines-a-light-on-optical-architecture-for-future-ai-supercomputer/) — rack-to-rack photonic fiber links; distance-independent bandwidth; full supercomputer-scale topology described qualitatively
- [Enosemi/Luminous press release (Oct 2023)](https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html) — Gen 2 card described with "networking" for large-scale applications; scale-out product intent confirmed though never shipped

---

## IP Disposition

### Channel 1 — Enosemi license (Oct 2023), Enosemi acquired by AMD (May 2025)
- [Enosemi launch press release — PRWeb Oct 2023](https://www.prweb.com/releases/semiconductor-startup-enosemi-launches-with-a-committed-commercial-license-to-key-silicon-photonics-design-ip-created-by-luminous-computing-301956542.html) — Enosemi founded May 2023 by ex-Elenion/Nokia engineers (Ari Novack, Matt Streshinsky, Shahab Ardalan); **committed commercial license** to Luminous's silicon photonics design IP. A license, not a sale — Luminous retained title.
- [Electronics Weekly: AMD buys Enosemi (2025-05-29)](https://www.electronicsweekly.com/news/business/amd-buys-enosemi-2025-05/) — **CORRECTS the date.** AMD acquired Enosemi (16 employees) in **May 2025**, not October 2023, to accelerate co-packaged optics (CPO). Quotes AMD SVP Brian Amick; confirms Enosemi licensed its technology from Luminous.
- [Enosemi launch announcement page](https://www.enosemi.com/news/2023-10-12_enosemi_luminous_press_release/) — the Oct 2023 launch item; it does **not** report an AMD acquisition, and an earlier revision of this file mis-cited it as such.

### Channel 2 — Luminous patent portfolio sold outright to AMD (USPTO recorded 2026-02-28)
- [US11500410B1 — "System and method for parallel photonic computation"](https://patents.google.com/patent/US11500410B1/en) — PRIMARY. Luminous Computing, Inc. → Advanced Micro Devices, Inc., recorded 2026-02-28.
- [US12379543B2 — "Photonic integrated circuit system and method of fabrication"](https://patents.google.com/patent/US12379543B2/en) — PRIMARY. Same assignment, 2026-02-28.
- [US12613811B2 — "Computer architecture with disaggregated memory and high-bandwidth communication interconnects"](https://patents.google.com/patent/US12613811B2/en) — PRIMARY. Granted 2026-04-28; same assignment recorded 2026-02-28. Three filings establish a portfolio-level sale.
- Negative evidence: no press coverage of this transaction was found. USPTO assignment records are the sole evidence.

---

## Other Resources

- [TechCrunch: Photonics is a tough nut to crack (Apr 2023)](https://techcrunch.com/2023/04/21/light-powered-ai-chips-future/) — industry context; photonic chip challenges: larger footprint, difficult mass production, immature fab ecosystem, electronic control bottlenecks
- [The Register: Luminous $105M Series A](https://www.theregister.com/2022/03/05/luminous-ai-supercomputer-photonics/) — pivot description: shifted from compute to chip-to-chip communications where photonics provides most benefit
- [Teamblind: "Luminous Computing is dead?"](https://www.teamblind.com/post/Luminous-Computing-is-dead-dhKjvYd4) — employee discussion confirming May 2023 layoffs and photonics team disbanding
- [SVDisposition: Surplus Assets Auction (July 2023)](https://www.svdisposition.com/auction-detail?id=594) — physical lab equipment auctioned, confirming wind-down of hardware operations
- [Futuriom: Luminous Powers AI Optics at the Edge (May 2023)](https://www.futuriom.com/articles/news/luminous-computing-points-optics-at-the-edge/2023/05) — post-pivot description; "edge AI inference" framing
- [Optics.org: Hochberg joins Luminous (2021)](https://optics.org/news/13/3/36) — silicon photonics pioneer Michael Hochberg joined Luminous in 2021
- [Ryan Hamerly Google Scholar](https://scholar.google.com/citations?user=PUUW6h4AAAAJ&hl=en) — academic context: NTT Research / MIT visiting scientist, photonic processor research; not formally affiliated with Luminous but work cited in photonic computing field
- [Photonic Multiply-Accumulate IEEE paper — Nahmias et al. 2020](https://ieeexplore.ieee.org/ielaam/2944/8764697/8844098-aam.pdf) — academic foundation from Luminous CTO
