# Tesla Dojo — Hardware Architecture Investigation

*as_of: 2026-08-08*
*Sources: Hot Chips 34 (HC34, August 2022), IEEE Xplore, tech press analysis; roadmap addendum 2026-08-08 (see final section)*

---

## Investigation Summary

This document synthesizes the hardware architecture of Tesla Dojo v1 from public disclosures, primarily the Hot Chips 34 presentation "The Microarchitecture of Tesla's Exa-Scale Computer" (August 2022) and the associated IEEE Xplore journal paper. It is organized by the system hierarchy: D1 chip → Training Tile → System Tray → Cabinet → ExaPOD.

---

## 1. D1 Chip

### Die Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Fabrication process | TSMC 7 nm | HC34 |
| Die size | 645 mm² | HC34 |
| Transistors | ~50 billion | HC34 |
| Compute nodes per die | 354 training nodes | HC34 |
| Clock speed | 2 GHz | HC34 |
| TDP | 400 W | HC34 |

### Compute Performance

| Precision | Throughput |
|-----------|-----------|
| BF16 | 376 TFLOPS |
| CFP8 (custom FP8) | 376 TFLOPS |
| FP32 | 22 TFLOPS |

The D1 chip is the fundamental compute die. It departs from conventional GPU design: instead of a small number of large, cache-coherent SIMT multiprocessors, it contains 354 relatively small "training nodes" tightly coupled in an on-chip mesh.

### Compute Node (per node)

Each of the 354 training nodes contains:
- A 64-byte SIMD vector unit with BF16/CFP8/CFloat16 mixed-precision support
- A scalar unit executing the custom 64-bit Dojo ISA
- 1.25 MB of local SRAM (software-managed, no hardware cache coherence)
- Load bandwidth: ~400 GB/s per node
- Store bandwidth: ~270 GB/s per node

This SRAM-first design eliminates DRAM and cache hierarchy inside the chip: there is no virtual memory, no L1/L2 cache, no global coherence directory. The software compiler is responsible for all data movement between nodes and between on-chip SRAM and off-chip HBM.

### ISA

- Custom 64-bit scalar ISA (name not publicly disclosed)
- 64-byte SIMD vector operations
- Supports BF16, CFP8 (Tesla's custom 8-bit float), CFloat16, FP32
- LLVM-based compiler backend (confirmed by HC34 and job postings)
- No public ISA specification document

### On-chip Memory

| Parameter | Value |
|-----------|-------|
| Total on-chip SRAM | 440 MB |
| Per-node SRAM | 1.25 MB |
| Node count | 354 |
| Memory management | Software-managed (no cache hierarchy) |

### On-chip Interconnect (Mesh)

- Nodes are arranged in a 2D mesh topology within the die
- ~10 TB/s aggregate directional on-chip mesh bandwidth
- Bidirectional ring-bus connecting nodes in rows and columns
- Deterministic, software-scheduled data movement (no hardware coherence)

### Chip-to-Chip (C2C) SerDes

| Parameter | Value |
|-----------|-------|
| SerDes lanes per chip | 576 |
| SerDes speed | 112 GT/s |
| Aggregate chip-to-chip BW | ~8 TB/s |
| Channels | 96 high-bandwidth bi-directional SerDes links |

The D1 chip tiles directly to adjacent D1 chips without any external switching silicon — the SerDes interfaces are exposed on the chip edges, and the chips butt-join on a PCB substrate.

---

## 2. Training Tile

The Training Tile is a 5×5 grid of D1 chips on a single PCB module, the fundamental deployment unit.

| Parameter | Value |
|-----------|-------|
| D1 chips per tile | 25 (5×5 arrangement) |
| BF16/CFP8 throughput | 9 PFLOPS |
| On-tile SRAM | 11 GB |
| Thermal design power | ~15 kW |
| Aggregate bandwidth | 36 TB/s |

- Chips communicate directly over the SerDes mesh — no switching chips required
- Each tile edge connects to DIP (Dojo Interface Processor) cards for HBM access
- 5 DIP cards per tile edge provide external HBM at 13 TB/s aggregate per tile
- Each DIP card also handles host-side connectivity at 160 GB/s

### Memory Hierarchy at Tile Level

| Level | Type | Capacity | Bandwidth | Managed By |
|-------|------|----------|-----------|-----------|
| On-chip SRAM | SRAM | 11 GB (25 × 440 MB) | ~250 TB/s (25 × on-chip mesh) | Software compiler |
| HBM (via DIP) | HBM | Not specified per tile | 13 TB/s | DMA, TTP protocol |
| Host server DRAM | DRAM | External | 160 GB/s host link | TTP data ingestion |

---

## 3. System Tray and Cabinet

| Level | Unit | Key Metrics |
|-------|------|-------------|
| System Tray | 6 Training Tiles | 6× tile compute |
| Cabinet | 2 System Trays | 300 D1 chips, 106,200 compute nodes |

- Two trays per cabinet, bolted together in a standard rack unit
- Cabinet-level power distribution and liquid cooling
- Cabinet-level TTPoE switch fabric connects to scale-out network

---

## 4. ExaPOD Cluster

| Parameter | Value |
|-----------|-------|
| Cabinets | 10 |
| D1 chips | 3,000 |
| Compute nodes | 1,062,000 |
| BF16/CFP8 throughput | 1 EFLOPS |
| Total on-chip SRAM | ~1.3 TB |
| HBM (via DIP cards) | ~13 TB |

The ExaPOD is the full-scale deployment unit. Ten cabinets are networked together using TTPoE (Tesla Transport Protocol over Ethernet) over commodity Ethernet switches. The name reflects the 1 ExaFLOP (1 EFLOPS) peak compute figure.

---

## 5. Scale-out Interconnect: TTPoE

TTPoE (Tesla Transport Protocol over Ethernet) is Tesla's custom Layer-2/transport protocol for scale-out networking, open-sourced at Hot Chips 36 (August 2024).

| Feature | Detail |
|---------|--------|
| Physical layer | Commodity Ethernet switches (no PFC, no special RDMA hardware) |
| Latency | Microsecond one-way (hardware-offloaded NIC) |
| Loss handling | Anticipates packet loss; built-in hardware retry |
| NIC | "Dumb-NIC" (Tesla custom, hardware offload, no CPU involvement) |
| Open-source | `teslamotors/ttpoe` on GitHub |
| Industry body | Tesla joined UltraEthernet Consortium (UEC) |

TTPoE contrasts with InfiniBand (RDMA, requires lossless fabric with PFC) and RoCEv2 (also requires PFC). Tesla's approach is closer to the spirit of the emerging UltraEthernet standard: accept packet loss and handle it efficiently in hardware rather than engineering losslessness into the fabric.

---

## 6. Power and Cooling

| Level | Power |
|-------|-------|
| D1 chip | 400 W |
| Training Tile (25 D1) | ~15 kW |
| ExaPOD (3000 D1) | ~1.2 MW (estimated, not officially stated) |

- Liquid cooling assumed at cabinet and tile level given power density

---

## Key Architectural Distinctions

1. **No external switching silicon**: D1 chips tile directly — the chips are the fabric.
2. **Software-managed memory**: No cache hierarchy; compiler schedules all data movement explicitly.
3. **Custom ISA**: Not x86, not ARM, not RISC-V — a bespoke design for neural network workloads.
4. **Flat power delivery**: Single 400W chip rather than massive multi-chip modules, simplifying thermal management at the die level.
5. **SRAM-centric on-chip memory**: 440 MB per chip is extraordinarily large for SRAM (H100 has ~36 MB L2, plus 228 KB/SM shared memory).

---

## Addendum — Roadmap Investigation, 2026-08-08

*Scope: what changed between the 2026-04-05 baseline and 2026-08-08. Conclusion up front: **no Dojo hardware fact in sections 1–6 above changed.** Everything in this addendum is program/roadmap context, verified against primary filings and contemporaneous reporting.*

### A. Dojo 3 — program confirmed, architecture entirely undisclosed

| Question | Finding | Confidence |
|---|---|---|
| Is Dojo 3 an active program? | Yes. Publicly confirmed 2026-01-19/20 (Teslarati 2026-01-19 "Tesla confirms that work on Dojo 3 has officially resumed"; TechCrunch and Tom's Hardware 2026-01-20; Bloomberg 2026-01-17). Restated in Musk's 2026-04-15 X post | high |
| Is the April 2026 news a status upgrade? | **No.** The revival predates this survey's 2026-04-05 baseline by ~3 months; the repo already recorded "January 2026 — Restart announced by Musk." The 2026-04-15 X post ("Congrats to the @Tesla_AI chip design team on taping out AI5! AI6, Dojo3 & other exciting chips in work.") is an incidental restatement | high (that it is not new); medium (exact post wording — verified only via secondary outlets, not the post itself) |
| Process node | not disclosed | — |
| Die size / transistor count / compute-unit type or count | not disclosed | — |
| On-chip SRAM, off-chip memory technology/capacity/bandwidth | not disclosed | — |
| Scale-up topology; is the 5×5 Training Tile retained? | not disclosed; **unknown either way** | — |
| Scale-out fabric; is TTPoE reused? | not disclosed; no new TTPoE release since HC36 | — |
| Tape-out date / performance target | not disclosed | — |
| Stated purpose | Space-based / orbital AI compute. Musk, January 2026: "AI7/Dojo3 will be for space-based AI compute." Corroborated by SpaceX's Form S-1 (2026-05-20): Terafab supports "two kinds of chips— one type optimized for terrestrial edge and inference to be used primarily in Tesla's Optimus robots and vehicles, and another type optimized for the space environment to be used in our orbital compute infrastructure" | high |
| Engineering leadership | Anant Nivarti rejoined Tesla to lead silicon engineering including AI6 and Dojo 3 (Data Center Dynamics, 2026-04-07) | medium |

**Superseded framing — do not treat as current.** The Q2 2025 earnings call (2025-07-23) contains, verbatim: *"Thinking about Dojo 3 and the AI6 inference chip, it seems like intuitively, we want to try to find convergence there where it's basically the same chip, but it's used where, say, 2 of them in a car or an Optimus and maybe a larger number on a board, kind of 5, 12 on a board or something like that."* This is often paraphrased as a converged terrestrial "5–12 chips per server board" architecture that abandons wafer-scale tiling. Two cautions: (1) it is a July 2025 statement, superseded by the January 2026 space-compute repositioning; (2) it is **intent, not a design disclosure** — it does not establish that Dojo 3 abandons (or keeps) the Training Tile topology. Note also that the commonly circulated version of this quote inserts bracketed text ("[converged architecture designs]", "[server]") that is not in the transcript and drops Musk's own characterization of AI6 as an *inference* chip.

### B. Adjacent Tesla silicon (context for why Dojo 3 has no silicon news)

| Chip | Status 2026-08-08 | Source | Confidence |
|---|---|---|---|
| AI5 | Taped out 2026-04-15; reported in wafer production at Samsung Taylor, TX on 2nm (Electrek, 2026-07-13). Volume/qualification unverified. Musk (2026-04-15): initial focus is "Optimus and our supercomputer clusters"; vehicles ~2027; "AI4 is enough to achieve much better than human safety for FSD" | Electrek 2026-07-13; Tom's Hardware / Not a Tesla App 2026-04-15/16 | medium |
| AI6 | In design; Musk (2026-03-19): Tesla "might be able to tape out AI6 in December." Described as a self-driving/FSD chip **in addition to** Optimus and datacenter use — it is AI5, not AI6, that skips vehicles first. Backed by the $16.5B Samsung foundry agreement announced 2025-07-27/28 (pre-baseline) | Reuters 2026-03-19; Teslarati 2026-03-21 | medium-high |
| MEGAPOD | Tesla trademark filing, USPTO SN 99893717 (~2026-06-18), covering "computer servers, computer hardware for artificial intelligence processing, computer networking hardware, electrical power distribution units, and cooling systems, sold as a unit." **Trademark only** — no product, no chip, no specification | USPTO; Teslarati | medium |

### C. Terafab — foundry context, no Dojo linkage

| Item | Finding | Confidence |
|---|---|---|
| Announced | **March 2026** (not April). SpaceX S-1: "We announced a collaboration with Tesla in March 2026 to build the Terafab initiative with a long-term goal of producing one terawatt of compute hardware each year." "Intel joined the project in April 2026" | high (primary filing) |
| Process | **Intel 14A**, disclosed by Musk 2026-04-22/23 (Reuters, Tom's Hardware 04-22; DCD 04-23) | high |
| Site | Confirmed 2026-08-06: new greenfield SpaceX-led site in **Grimes County, near College Station, TX** — *not* an existing SpaceX campus. A separate prototype "Advanced Technology Fabrication" facility is at Tesla's Giga Texas | high |
| Capital | **$16.8B initial** investment; up to ~100M sq ft of manufacturing space (Reuters 2026-08-06). ⚠️ SpaceX's own May 2026 Texas filings put phase one at ~$55B and total build-out at up to ~$119B. Report $16.8B only as an initial-stage figure | high for $16.8B; medium for the $55B/$119B filings |
| Equipment | ASML CEO Christophe Fouquet confirmed direct talks with Musk, calling him "very serious" (Tom's Hardware 2026-05-21; Bloomberg 2026-06-17); ASML CFO said guidance includes Terafab (Reuters 2026-07-15) | medium-high |
| Tesla's commitment | **Framework agreement only.** S-1: "While we have a framework agreement with Tesla, neither Tesla nor Intel are obligated to remain a part of the project, and we may not enter into any such definitive agreements." Also "a general framework" whose specific projects "will be subject to separate negotiations and agreements … and have not yet been determined" | high (primary filing) |
| Execution risk | S-1: "While we expect to construct Terafab to address such supply constraints, Terafab may not be successful, in which case we may not have other sources of sufficient AI chips to meet our orbital AI compute demands" | high (primary filing) |
| Dojo linkage | **None.** The S-1 contains no mention of "Dojo," "AI5," "AI6," or "14A" | high |

### D. Negative findings (checked, nothing found)

- **No Tesla or Dojo talk in the Hot Chips 38 advance program** (Aug 23–25, 2026), checked 2026-08-08. Note that Hot Chips 38 is 15 days in the future as of this writing; even for listed talks, no slides or abstracts exist.
- No Dojo 3 tape-out announcement; no D2/D3 silicon disclosure.
- No new TTPoE release; `teslamotors/ttpoe` remains the only open-source artifact.
- Methodological note: a vendor/publisher *tag page* showing no recent items is not valid evidence of absence and was not relied on here.

### E. Spec point re-verified, no change

D1 BF16/CFP8 throughput stays at **376 TFLOPS** (Hot Chips 34). A May 2026 Tom's Hardware custom-ASIC survey lists 362 TFLOPS; the HC34 figure is the better primary source and is independently corroborated. **362 is not adopted.**

### F. Pre-baseline items recorded for completeness

The shutdown quote (Musk, August 2025: "once it became clear that all paths converged to AI6, I had to shut down Dojo and make some tough personnel choices"), Peter Bannon's departure, and ~20 Dojo engineers leaving to found DensityAI are all Bloomberg/Reuters reports of 2025-08-07 — 8–12 months older than the survey baseline and already covered by the existing "August 2025 — Dojo team disbanded" entry. The "retired" framing for Bannon is **not independently confirmed**; contemporaneous coverage says he left and the team was disbanded.

---

## Sources

| Resource | URL |
|----------|-----|
| HC34 PDF — Microarchitecture | https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf |
| IEEE Xplore — HC34 paper | https://ieeexplore.ieee.org/document/10078146/ |
| Chips & Cheese — HC34 analysis | https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture |
| The Next Platform — inside Dojo | https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/ |
| SemiAnalysis — D1 chip critique | https://newsletter.semianalysis.com/p/the-tesla-dojo-chip-is-impressive |
| Wikipedia — Tesla Dojo | https://en.wikipedia.org/wiki/Tesla_Dojo |

### Added 2026-08-08 (roadmap addendum)

| Resource | URL |
|----------|-----|
| SpaceX Form S-1, filed 2026-05-20 (CIK 0001181412, accession 0001628280-26-036936) | https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm |
| SEC full-text search — "Terafab" in S-1 filings | https://efts.sec.gov/LATEST/search-index?q=%22Terafab%22&forms=S-1 |
| SEC submissions index — SpaceX (CIK 0001181412) | https://data.sec.gov/submissions/CIK0001181412.json |
| TechCrunch — Dojo3 for space-based AI compute (2026-01-20) | https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute/ |
| Tom's Hardware — Dojo3 space supercomputer restart (2026-01-20) | https://www.tomshardware.com/tech-industry/supercomputers/elon-musk-restarts-dojo3-space-supercomputer-project-as-ai5-chip-design-gets-in-good-shape-will-be-first-tesla-built-supercomputer-to-feature-all-in-house-hardware-with-no-help-from-nvidia |
| Tesla Q2 2025 earnings call transcript (2025-07-23) | https://www.insidermonkey.com/blog/tesla-inc-nasdaqtsla-q2-2025-earnings-call-transcript-1575336/ |
| Tom's Hardware — Terafab: 100M sq ft, $16.8B initial capital (2026-08-07) | https://www.tomshardware.com/tech-industry/semiconductors/terafab-starts-to-take-shape-100-million-square-feet-of-manufacturing-space-and-usd16-8b-initial-capital-investment |
| Tom's Hardware — SpaceX IPO risk factors / Terafab may not succeed | https://www.tomshardware.com/tech-industry/artificial-intelligence/spacex-admits-it-cant-find-enough-chips-for-orbital-ai-yet-requires-significantly-more-than-are-currently-available-to-us-firms-risk-factors-in-ipo-paperwork-also-says-ambitious-terafab-project-may-not-be-successful |
| Tom's Hardware — Musk demonstrates first AI5 sample (2026-04-15/16) | https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc-rather-than-tsmc-claims-40x-performance-boost-over-the-predecessor |
| Tom's Hardware — custom AI ASICs examined (May 2026; source of the non-adopted 362 TFLOPS figure) | https://www.tomshardware.com/tech-industry/semiconductors/custom-ai-asics-examined-from-broadcom-to-mtia |
| Not a Tesla App — AI5 done, not coming to vehicles first | https://www.notateslaapp.com/news/3974/musk-says-teslas-ai5-chip-is-done-not-coming-to-vehicles-first |
| Teslarati — Tesla finalizes AI5 chip design | https://www.teslarati.com/tesla-finalizes-ai5-chip-design-elon-musk-makes-bold-claim-capability/ |
| Teslarati — AI6 self-driving chip expectations (2026-03-21) | https://www.teslarati.com/elon-musk-teases-expectations-tesla-ai6-self-driving-chip/ |
| Teslarati — Tesla trademarks MEGAPOD (June 2026) | https://www.teslarati.com/tesla-just-trademarked-megapod-heres-what-it-is/ |
| American Bazaar — Musk confirms progress on AI6, Dojo3 (2026-04-16) | https://www.americanbazaaronline.com/2026/04/16/tesla-ai5-chip-elon-musk-confirms-progress-on-ai6-dojo3-systems/ |
| Wikipedia — Terafab | https://en.wikipedia.org/wiki/Terafab |
| Hot Chips 38 advance program (Aug 23–25, 2026) — no Tesla/Dojo talk, checked 2026-08-08 | https://hotchips.org/advance-program/ |
