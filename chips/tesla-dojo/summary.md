# Tesla Dojo — Software and Hardware Stack Summary

*as_of: 2026-08-08*

---

## Project Status

> **Tesla disbanded the Dojo team in August 2025** after declaring Dojo 2 "an evolutionary dead end." A **Dojo 3** program was publicly confirmed as resumed in **January 2026**, and Musk has since restated it in passing (X post, 2026-04-15). But as of 2026-08-08 there is **no disclosed silicon, process node, memory system, interconnect, tape-out date, or performance target for Dojo 3** — every architectural statement about it would be invention. Tesla's near-term AI silicon investment is in the **AI5** (Samsung Taylor, 2nm) and **AI6** (Samsung, $16.5B foundry agreement) chips, and Tesla continues to rely on NVIDIA hardware for current training needs. The **D1** chip and the Dojo v1 architecture remain the latest *disclosed* Dojo hardware and the basis for this summary. See the [2026-08-08 roadmap update](#roadmap-update--dojo-3-ai5ai6-and-terafab-2026-08-08) below.

---

## Overview

Tesla Dojo is one of the most architecturally distinctive AI training systems ever disclosed publicly. Conceived to train Tesla's Autopilot neural networks on massive video datasets, Dojo departs from the GPU paradigm in fundamental ways: it replaces general-purpose streaming multiprocessors with hundreds of small, tightly-coupled training nodes on each die, replaces cache hierarchies with explicit software-managed SRAM, and replaces industry-standard networking (InfiniBand) with a custom lossy Ethernet protocol (TTPoE).

The result is a system optimized for a specific workload — large-scale neural network training on internally-generated video data — at the cost of programmability, ecosystem openness, and toolchain portability. The D1 chip reached 376 TFLOPS at BF16/CFP8 on TSMC 7nm in 2022, and an ExaPOD of 3,000 D1 chips achieves 1 EFLOPS at that precision. Despite the architectural novelty, the software stack was never publicly released, and the project's cancellation in August 2025 leaves TTPoE as the only lasting open-source artifact.

---

## Hardware Stack

### D1 Chip

The **D1** is a 645 mm², 50-billion-transistor monolithic die on TSMC 7nm. It contains **354 training nodes** arranged in a 2D mesh. Each node has:
- A 64-byte SIMD vector unit (BF16/CFP8/CFloat16)
- A scalar unit running a custom 64-bit ISA
- 1.25 MB of local SRAM (software-managed)

The chip contains **440 MB of total on-chip SRAM** — extraordinarily large by any standard (H100 has ~36 MB L2 total). There is deliberately **no cache hierarchy, no virtual memory, and no coherence directory**. The software compiler is responsible for all data placement and movement. The chip runs at 2 GHz and dissipates 400 W.

Chip-to-chip connectivity uses 576 SerDes lanes at 112 GT/s, providing ~8 TB/s aggregate chip-to-chip bandwidth. Chips tile directly without switching silicon — the SerDes interfaces butt-join on a shared PCB.

### Training Tile → ExaPOD Hierarchy

| Level | Unit | Compute | Memory |
|-------|------|---------|--------|
| Die | D1 chip | 376 TFLOPS BF16 | 440 MB SRAM |
| Module | Training Tile (5×5) | 9 PFLOPS BF16 | 11 GB SRAM + HBM via DIP |
| Rack Unit | System Tray (6 tiles) | 54 PFLOPS | — |
| Cabinet | 2 trays | 300 D1 chips | — |
| Cluster | ExaPOD (10 cabinets) | 1 EFLOPS BF16 | 1.3 TB SRAM + 13 TB HBM |

The **ExaPOD** — ten cabinets totaling 3,000 D1 chips and 1,062,000 compute nodes — achieves 1 ExaFLOP at BF16/CFP8. HBM is not on the D1 die itself; instead, **DIP (Dojo Interface Processor) cards** at tile edges provide 13 TB/s HBM access per tile.

### Scale-out: TTPoE

Tesla's **Tesla Transport Protocol over Ethernet (TTPoE)** connects ExaPODs over commodity Ethernet switches. Unlike InfiniBand (which requires a lossless fabric) or RoCEv2 (which requires Priority Flow Control), TTPoE is designed for a **lossy fabric**: hardware-offloaded retry handles packet loss without complex switch-level flow control. Tesla open-sourced TTPoE at Hot Chips 36 (August 2024) and joined the UltraEthernet Consortium. The GitHub repository (`teslamotors/ttpoe`) remains active and is the only open-source artifact from the Dojo program.

---

## Software Stack

Tesla's Dojo software stack is built entirely in-house and has never been publicly released. The user-facing interface is **standard PyTorch** — a trained model can be submitted for Dojo training without writing custom kernels. Beneath that:

1. **Custom ML Compiler**: Captures the PyTorch computation graph, optimizes it, and generates code via an **LLVM backend** targeting the Dojo custom ISA. Handles data parallelism, model parallelism, and graph parallelism automatically. Compiles on first execution; caches and reuses kernels.
2. **Custom ISA**: A 64-bit scalar + 64-byte SIMD vector instruction set supporting BF16, CFP8, CFloat16, and FP32. No public specification.
3. **Runtime**: Fault-tolerant — reroutes mesh traffic around failed links. Manages SRAM allocation across nodes (no hardware virtual memory).
4. **TTP data ingestion**: Video frames flow from host servers to DIP cards at ~160 GB/s per tile edge using the Tesla Transport Protocol.

There is no public compiler, no public ISA document, no public SDK, and no public kernel library. This is the sharpest contrast to the CUDA/ROCm ecosystems where every layer is documented and most are open-source.

---

## Key Architectural Distinctions

| Feature | Dojo | H100 GPU |
|---------|------|----------|
| Compute units per die | 354 training nodes | 132 SMs |
| On-chip SRAM | 440 MB (software-managed) | ~36 MB L2 + 228 KB SMEM/SM |
| Cache hierarchy | None (by design) | 4-level (registers → SMEM → L2 → HBM) |
| Memory management | Compiler-explicit | Hardware (caches, virtual memory, TLB) |
| Off-chip memory on die | None (HBM via separate DIP cards) | HBM on-package (CoWoS) |
| Chip-to-chip | Direct SerDes butt-join | NVLink + NVSwitch |
| Scale-out | TTPoE (lossy Ethernet) | InfiniBand / RoCEv2 |
| ISA | Custom, not public | PTX (virtual) + SASS (native) |
| Software openness | Not public (TTPoE only) | Mostly open (CUDA, CUTLASS, NCCL) |

---

## Timeline

| Date | Event |
|------|-------|
| August 2021 | D1 chip announced at Tesla AI Day |
| August 2022 | Architecture revealed at Hot Chips 34 |
| October 2022 | First Dojo supercomputer powered on |
| July 2023 | Dojo v1 entered production |
| August 2024 | TTPoE open-sourced; presented at Hot Chips 36 |
| August 2025 | Dojo team disbanded; D2/Dojo 2 cancelled |
| 2026-01-17/20 | Dojo 3 restart publicly confirmed (Bloomberg 01-17; Teslarati 01-19; TechCrunch and Tom's Hardware 01-20). Musk: "AI7/Dojo3 will be for space-based AI compute" |
| March 2026 | Terafab collaboration between SpaceX and Tesla announced (per SpaceX Form S-1) |
| 2026-03-19 | Musk: Tesla "might be able to tape out AI6 in December" (Reuters) |
| 2026-04-07 | Anant Nivarti rejoins Tesla to lead silicon engineering, including AI6 and Dojo 3 (Data Center Dynamics) |
| 2026-04-15 | AI5 taped out; Musk X post: "AI6, Dojo3 & other exciting chips in work" |
| 2026-04-22/23 | Musk discloses Terafab will use **Intel 14A**; Intel joined the project in April 2026 |
| 2026-05-20 | SpaceX Form S-1 filed — flags Terafab execution risk; Tesla participation is a *framework agreement* only |
| June 2026 | Tesla files "MEGAPOD" trademark (USPTO SN 99893717) for modular AI-datacenter hardware — trademark only, no product or chip disclosed |
| 2026-07-13 | AI5 reported in wafer production at Samsung Taylor, TX on 2nm (Electrek) |
| 2026-08-06 | Terafab site made official: Grimes County (near College Station), TX; $16.8B *initial* capital investment |
| 2026-08-08 | No Dojo 3 silicon disclosure; no Tesla/Dojo talk in the Hot Chips 38 advance program (Aug 23–25, 2026) |

---

## Roadmap Update — Dojo 3, AI5/AI6, and Terafab (2026-08-08)

*Updated 2026-08-08. Verified against SpaceX's Form S-1 (CIK 0001181412, accession 0001628280-26-036936, filed 2026-05-20), the Tesla Q2 2025 earnings-call transcript, the Hot Chips 38 advance program, and contemporaneous Reuters / Bloomberg / TechCrunch / Tom's Hardware / Electrek reporting. Nothing in this section is a hardware disclosure.*

### What is actually new since the 2026-04-05 baseline

**Nothing about Dojo silicon.** No Dojo 3 tape-out, no D2/D3 chip disclosure, no new TTPoE release, and **no Tesla or Dojo talk in the Hot Chips 38 advance program** (Aug 23–25, 2026). The D1 / Training Tile / ExaPOD specifications in this survey remain the latest disclosed Dojo hardware and are unchanged.

**Corrections to the previous baseline text:**

- The prior summary's "Musk announced a restart in January 2026, but … the project remains in transition" understated the situation in one direction and the April news overstated it in the other. Dojo 3's revival was **publicly confirmed on 2026-01-19/20**, three months before this survey's baseline. The 2026-04-15 X post — "Congrats to the @Tesla_AI chip design team on taping out AI5! AI6, Dojo3 & other exciting chips in work." — is an **incidental restatement, not a status upgrade**. (Post wording verified only through secondary outlets, not the post itself.)
- The prior line "AI6 (Samsung-fabbed) is new training direction" conflated two chips. It is **AI5**, not AI6, that is not going to vehicles first: Musk said on 2026-04-15 that AI5's initial focus is "Optimus and our supercomputer clusters," with vehicles expected around 2027, and that "AI4 is enough to achieve much better than human safety for FSD." **AI6 is described as a self-driving/FSD chip** in addition to Optimus and datacenter use.

### Dojo 3's stated purpose: space-based compute

The most recent public statement of Dojo 3's purpose is Musk's January 2026 X post: **"AI7/Dojo3 will be for space-based AI compute"** — Dojo 3 is now paired with AI7 and aimed at *orbital* compute. This is corroborated by SpaceX's Form S-1, which describes Terafab as supporting "two kinds of chips— one type optimized for terrestrial edge and inference to be used primarily in Tesla's Optimus robots and vehicles, and another type optimized for the space environment to be used in our orbital compute infrastructure."

An **older and now superseded** framing is often quoted as if it were current: on the Q2 2025 earnings call (2025-07-23) Musk said, verbatim, *"Thinking about Dojo 3 and the AI6 inference chip, it seems like intuitively, we want to try to find convergence there where it's basically the same chip, but it's used where, say, 2 of them in a car or an Optimus and maybe a larger number on a board, kind of 5, 12 on a board or something like that."* That July 2025 "converged chip, 5–12 on a board" statement predates the January 2026 space-compute repositioning and should not be presented as Tesla's current direction.

> **Both statements are intent, not design disclosure.** Nothing public confirms whether Dojo 3 retains or abandons the wafer-scale-style 5×5 Training Tile topology, the 440 MB-SRAM/no-cache memory model, the direct-SerDes butt-join, or TTPoE. Treat all Dojo 3 architecture as **not disclosed**.

### AI5 / AI6 status

| Chip | Status (2026-08-08) | Foundry / node | Stated target |
|---|---|---|---|
| AI5 | Taped out 2026-04-15; reported in **wafer production** at Samsung Taylor, TX on **2nm** (Electrek, 2026-07-13; volume/qualification unverified) | Samsung Taylor, 2nm | Optimus and Tesla's supercomputer clusters first; vehicles ~2027 |
| AI6 | In design; Musk (2026-03-19, Reuters) said Tesla "might be able to tape out AI6 in December" | Samsung, under the $16.5B agreement announced 2025-07-27/28 | Self-driving/FSD, Optimus, and datacenter |
| AI7 / Dojo 3 | Program acknowledged; **no silicon, node, memory, interconnect, tape-out date, or performance figure disclosed** | not disclosed | Space-based / orbital AI compute (Musk, Jan 2026) |

### Terafab — new supply-chain context

Terafab is a fab venture linking SpaceX, Tesla, and Intel; it is the first supply-chain development material to the Dojo line since the D1 era, so it is recorded here even though it discloses nothing about Dojo silicon.

| Item | Detail |
|---|---|
| Announced | **March 2026** (SpaceX–Tesla collaboration, per the S-1). Intel joined **April 2026** |
| Process | **Intel 14A** — disclosed by Musk 2026-04-22/23 (Reuters, Tom's Hardware, DCD) |
| Site | Confirmed 2026-08-06: a new greenfield SpaceX-led site in **Grimes County, near College Station, TX** — *not* an existing SpaceX campus. A separate prototype "Advanced Technology Fabrication" facility sits at Tesla's Giga Texas |
| Capital | **$16.8B initial** capital investment (Reuters, 2026-08-06), up to ~100M sq ft of manufacturing space. ⚠️ Read against SpaceX's own May 2026 Texas filings of **~$55B for phase one** and up to **~$119B** total build-out — $16.8B is an initial-stage figure only |
| Goal | "a long-term goal of producing one terawatt of compute hardware each year" (S-1) |
| Equipment | ASML CEO Christophe Fouquet publicly confirmed direct talks with Musk and called him "very serious" (2026-05-21, 2026-06-17); ASML's CFO said guidance includes Terafab (Reuters, 2026-07-15) |

**Risk, verbatim from the primary filing.** SpaceX's Form S-1 states: *"While we expect to construct Terafab to address such supply constraints, Terafab may not be successful, in which case we may not have other sources of sufficient AI chips to meet our orbital AI compute demands,"* and *"While we have a framework agreement with Tesla, neither Tesla nor Intel are obligated to remain a part of the project, and we may not enter into any such definitive agreements."* The S-1 further calls the Tesla arrangement "a general framework" whose specific projects "will be subject to separate negotiations and agreements … and have not yet been determined." **The S-1 contains no mention of "Dojo," "AI5," "AI6," or "14A."**

### Shutdown background (pre-baseline, recorded for completeness)

Musk, August 2025: *"once it became clear that all paths converged to AI6, I had to shut down Dojo and make some tough personnel choices."* Lead architect **Peter Bannon** departed and roughly **20 Dojo engineers left to found DensityAI** (Bloomberg/Reuters, 2025-08-07). Contemporaneous reporting says he left / the team was disbanded; the "retired" framing is **not independently confirmed**.

### Spec note — D1 throughput

A May 2026 Tom's Hardware custom-ASIC survey lists D1 as "362 TFLOPS BF16." This survey **keeps 376 TFLOPS BF16/CFP8**, the Hot Chips 34 primary figure, which is independently corroborated. The 362 figure is not adopted.

---

## Sources

- [Hot Chips 34 — "The Microarchitecture of Tesla's Exa-Scale Computer"](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf)
- [Hot Chips 36 — TTPoE presentation](https://hc2024.hotchips.org/assets/program/conference/day2/17_HC2024_Tesla_TTPoE_v5.pdf)
- [IEEE Xplore — DOJO Microarchitecture](https://ieeexplore.ieee.org/document/10078146/)
- [IEEE Xplore — TTPoE paper](https://ieeexplore.ieee.org/document/10664947)
- [teslamotors/ttpoe GitHub](https://github.com/teslamotors/ttpoe)
- [The Next Platform — Inside Tesla's Dojo](https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/)
- [Chips & Cheese — HC34 Dojo Analysis](https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture)
- [TechCrunch — Rise and Fall of Dojo](https://techcrunch.com/2025/09/02/tesla-dojo-the-rise-and-fall-of-elon-musks-ai-supercomputer/)

### Added 2026-08-08 (roadmap update)

- [SpaceX Form S-1 (filed 2026-05-20) — CIK 0001181412, accession 0001628280-26-036936](https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm) — primary source for the Terafab risk factors, the March 2026 announcement date, and the two-chip-type description
- [TechCrunch — "Elon Musk says Tesla's restarted Dojo3 will be for space-based AI compute" (2026-01-20)](https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute/)
- [Tom's Hardware — Musk restarts Dojo3 space supercomputer project (2026-01-20)](https://www.tomshardware.com/tech-industry/supercomputers/elon-musk-restarts-dojo3-space-supercomputer-project-as-ai5-chip-design-gets-in-good-shape-will-be-first-tesla-built-supercomputer-to-feature-all-in-house-hardware-with-no-help-from-nvidia)
- [Tesla Q2 2025 earnings call transcript (2025-07-23)](https://www.insidermonkey.com/blog/tesla-inc-nasdaqtsla-q2-2025-earnings-call-transcript-1575336/) — verbatim "Dojo 3 and the AI6 inference chip" quote
- [Tom's Hardware — Terafab: 100M sq ft, $16.8B initial capital (2026-08-07)](https://www.tomshardware.com/tech-industry/semiconductors/terafab-starts-to-take-shape-100-million-square-feet-of-manufacturing-space-and-usd16-8b-initial-capital-investment)
- [Tom's Hardware — SpaceX IPO risk factors on Terafab (2026-05)](https://www.tomshardware.com/tech-industry/artificial-intelligence/spacex-admits-it-cant-find-enough-chips-for-orbital-ai-yet-requires-significantly-more-than-are-currently-available-to-us-firms-risk-factors-in-ipo-paperwork-also-says-ambitious-terafab-project-may-not-be-successful)
- [Tom's Hardware — Musk demonstrates first AI5 sample (2026-04-15/16)](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc-rather-than-tsmc-claims-40x-performance-boost-over-the-predecessor)
- [Teslarati — AI6 self-driving chip expectations (2026-03-21)](https://www.teslarati.com/elon-musk-teases-expectations-tesla-ai6-self-driving-chip/)
- [Teslarati — Tesla trademarks MEGAPOD (June 2026)](https://www.teslarati.com/tesla-just-trademarked-megapod-heres-what-it-is/)
- [Wikipedia — Terafab](https://en.wikipedia.org/wiki/Terafab)
- [Hot Chips 38 advance program (Aug 23–25, 2026)](https://hotchips.org/advance-program/) — checked 2026-08-08; no Tesla/Dojo talk listed
