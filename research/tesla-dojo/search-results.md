# Tesla Dojo — Search Results

**Date:** 2026-04-05 (roadmap resources appended 2026-08-08 — see §10)
**Status Note:** Tesla disbanded the Dojo team in August 2025 after declaring Dojo 2 "an evolutionary dead end." A **Dojo 3** program was publicly confirmed as resumed on 2026-01-19/20 and restated by Musk on 2026-04-15, but as of 2026-08-08 **no Dojo 3 silicon, node, memory, interconnect, tape-out date, or performance target has been disclosed**. The D1 chip and original Dojo v1 architecture remain the best-documented public artifacts and the latest disclosed Dojo hardware.

---

## 1. Official Documentation / Whitepapers

| Resource | URL | Notes |
|---|---|---|
| Hot Chips 34 (HC34) — "The Microarchitecture of Tesla's Exa-Scale Computer" (PDF) | https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf | Primary public technical reference; presented August 2022 |
| Hot Chips 36 (HC2024) — TTPoE presentation (PDF) | https://hc2024.hotchips.org/assets/program/conference/day2/17_HC2024_Tesla_TTPoE_v5.pdf | Network layer deep-dive; presented August 2024 |
| IEEE Xplore — "The Microarchitecture of DOJO, Tesla's Exa-Scale Computer" | https://ieeexplore.ieee.org/document/10078146/ | Peer-reviewed journal version of HC34 content |
| IEEE Xplore — "Tesla Transport Protocol Over Ethernet (TTPoE): A New Lossy, Exa-Scale Fabric" | https://ieeexplore.ieee.org/document/10664947 | Peer-reviewed paper on TTPoE networking |

---

## 2. Open-Source Repositories

| Repo | URL | Notes |
|---|---|---|
| `teslamotors/ttpoe` | https://github.com/teslamotors/ttpoe | Tesla Transport Protocol over Ethernet — open-sourced at HC2024 |

- *No public compiler, runtime, ISA toolchain, or SDK repositories found.*
- *No public PyTorch backend or MLIR dialect repositories found.*

---

## 3. Chip Architecture (D1)

**Key specifications (from HC34 and public disclosures):**

- **Process:** TSMC 7 nm
- **Die size:** 645 mm²
- **Transistors:** ~50 billion
- **Compute nodes per die:** 354 training nodes
- **Clock speed:** 2 GHz
- **Compute:** 376 TFLOPS at BF16/CFP8; 22 TFLOPS at FP32
- **On-chip SRAM:** 440 MB total (1.25 MB per node × 354 nodes)
- **Off-chip bandwidth:** 576 SerDes lanes at 112 GT/s → ~8 TB/s aggregate chip-to-chip
- **On-chip mesh bandwidth:** ~10 TB/s directional
- **Power (TDP):** 400 W per chip

| Resource | URL |
|---|---|
| Tom's Hardware — D1 chip deep-dive | https://www.tomshardware.com/news/tesla-d1-ai-chip |
| Hot Chips 34 analysis (Chips & Cheese) | https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture |
| Data Center Dynamics — training tile reveal | https://www.datacenterdynamics.com/en/news/tesla-details-dojo-supercomputer-reveals-dojo-d1-chip-and-training-tile-module/ |
| Tesla Wikipedia article | https://en.wikipedia.org/wiki/Tesla_Dojo |

---

## 4. Training Tile / ExaPOD System Hierarchy

| Level | Unit | Key Specs |
|---|---|---|
| Die | D1 chip | 376 TFLOPS BF16, 440 MB SRAM, 400 W |
| Module | Training Tile (5×5 D1 array) | 9 PFLOPS BF16/CFP8, 11 GB SRAM, 15 kW, 36 TB/s aggregate BW |
| Rack unit | System Tray (6 tiles) | 6× tile compute; 2 trays per cabinet |
| Cabinet | Cabinet (2 system trays) | 300 D1 chips, 106,200 cores |
| Cluster | ExaPOD (10 cabinets) | 3,000 D1 chips, 1,062,000 cores, 1 EFLOPS at BF16/CFP8, ~1.3 TB SRAM + 13 TB HBM |

| Resource | URL |
|---|---|
| Next Platform — inside Dojo | https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/ |
| Tesla Wikipedia | https://en.wikipedia.org/wiki/Tesla_Dojo |

---

## 5. Memory Hierarchy

- **Primary compute memory:** Distributed SRAM (software-managed, no cache coherence)
  - 1.25 MB per compute node; load ~400 GB/s, store ~270 GB/s per node
  - No virtual memory, no global cache directories by design
- **External memory:** HBM (high-bandwidth memory) attached to DIP (Dojo Interface Processor) chips
  - Each training tile has 13 TB/s HBM access via 5 DIP cards per tile edge (160 GB/s host-side)
- **Two-tier NUMA:** Local SRAM (fast, software-managed) + HBM (higher latency, DMA-accessed)

| Resource | URL |
|---|---|
| Hackaday — Dojo as interesting CPU design | https://hackaday.com/2022/09/06/teslas-dojo-is-an-interesting-cpu-design/ |
| SemiAnalysis newsletter — technical critique | https://newsletter.semianalysis.com/p/the-tesla-dojo-chip-is-impressive |
| Chips & Cheese — HC34 analysis | https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture |

---

## 6. Interconnect / Networking

### On-tile mesh (chip-to-chip within tile)
- 96 high-bandwidth bi-directional SerDes links per D1
- ~2 TB/s aggregate chip-to-chip bandwidth per D1
- No external "glue" silicon required — chips tile directly

### Scale-out: TTPoE (Tesla Transport Protocol over Ethernet)
- Custom Layer-2/transport protocol replacing TCP for AI cluster traffic
- Runs on commodity Ethernet switches (no specialized RDMA/PFC switches needed)
- Hardware-offloaded on "Dumb-NIC"; microsecond-range one-way latency
- Handles lossy fabric with built-in retry; anticipates packet loss rather than avoiding it
- Open-sourced at HC2024; Tesla joined UltraEthernet Consortium (UEC)

| Resource | URL |
|---|---|
| `teslamotors/ttpoe` GitHub | https://github.com/teslamotors/ttpoe |
| ServeTheHome — TTPoE explainer | https://www.servethehome.com/tesla-dojo-exa-scale-lossy-ai-network-using-the-tesla-transport-protocol-over-ethernet-ttpoe/ |
| Chips & Cheese — TTPoE at HC2024 | https://chipsandcheese.com/p/teslas-ttpoe-at-hot-chips-2024-replacing-tcp-for-low-latency-applications |
| wcollins.io — TTPoE technical writeup | https://wcollins.io/posts/2024/tesla-adapts-ethernet-for-with-modified-transport-for-dojo/ |

---

## 7. Software Stack / Compiler / Programming Model

**What is known (from HC34 and job postings):**

- **Framework interface:** PyTorch extensions (no CUDA, no manual C/C++ kernels required for users)
- **Compiler:** Custom in-house ML compiler with LLVM backend
  - Translates PyTorch computation graphs to Dojo ISA
  - Generates code on-the-fly; reuses compiled kernels for subsequent executions
  - Handles data parallelism, model parallelism, and graph parallelism automatically
  - Supports loop-level control flow
- **ISA:** Custom 64-bit scalar + 64-byte SIMD vector ISA; BF16/CFP8/CFloat16 mixed precision
- **Runtime:** Fault-tolerant; reroutes mesh traffic around failed links; auto-scales across tiles
- **Data ingestion:** Tesla Transport Protocol (TTP) for high-throughput video data loading from host servers

**What is NOT public:**
- No public compiler source code
- No public ISA specification document
- No public SDK or toolchain releases
- No public kernel library (equivalent to cuDNN or XLA ops)

| Resource | URL |
|---|---|
| Next Platform — software stack overview | https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/ |
| Tweaktown — AI Day D1 transcript | https://www.tweaktown.com/news/81229/teslas-insane-new-dojo-d1-ai-chip-full-transcript-of-its-unveiling/index.html |
| Dice.com — ML Compiler job listing (archived) | https://www.dice.com/job-detail/709f33c4-12a1-41a6-bd1c-b790d9502039 |

---

## 8. Project Status & Timeline

| Date | Event |
|---|---|
| August 2021 | D1 chip and Dojo project announced at Tesla AI Day |
| August 2022 | Full architecture revealed at Hot Chips 34 (HC34) |
| October 2022 | First Dojo supercomputer powered on; briefly tripped the power grid |
| July 2023 | Dojo v1 went into production |
| August 2024 | TTPoE open-sourced; presented at Hot Chips 36 |
| August 2025 | Team disbanded; Dojo 2 / D2 chip program cancelled |
| 2026-01-17/20 | Dojo 3 restart publicly confirmed; Musk: "AI7/Dojo3 will be for space-based AI compute" |
| March 2026 | SpaceX–Tesla Terafab collaboration announced (per SpaceX Form S-1) |
| 2026-04-15 | AI5 taped out; Musk restates "AI6, Dojo3 & other exciting chips in work" |
| 2026-04-22/23 | Terafab process disclosed as Intel 14A; Intel joined the project in April 2026 |
| 2026-05-20 | SpaceX Form S-1 filed — Terafab risk factors; Tesla participation is a framework agreement only |
| 2026-07-13 | AI5 reported in wafer production at Samsung Taylor, TX on 2nm |
| 2026-08-06 | Terafab site confirmed: Grimes County near College Station, TX; $16.8B initial capital |
| 2026-08-08 | No Dojo 3 hardware disclosure; no Tesla/Dojo talk in the Hot Chips 38 advance program |

| Resource | URL |
|---|---|
| TechCrunch — rise and fall of Dojo | https://techcrunch.com/2025/09/02/tesla-dojo-the-rise-and-fall-of-elon-musks-ai-supercomputer/ |
| electrive.com — Tesla ends supercomputer project | https://www.electrive.com/2025/08/11/tesla-ends-supercomputer-project/ |
| eWeek — Dojo ends, next-gen AI chips | https://www.eweek.com/news/tesla-dojo-ai-supercomputer/ |

---

## 9. Other Resources

| Type | Title | URL |
|---|---|---|
| Press/analysis | Chips & Cheese — HC34 Dojo Microarchitecture | https://chipsandcheese.com/p/hot-chips-34-teslas-dojo-microarchitecture |
| Press/analysis | SemiAnalysis — D1 chip technical critique | https://newsletter.semianalysis.com/p/the-tesla-dojo-chip-is-impressive |
| Press/analysis | James Hamilton (Amazon) — Dojo overview | https://perspectives.mvdirona.com/2021/08/tesla-project-dojo-overview/ |
| Press | notateslaapp.com — HC34 summary | https://www.notateslaapp.com/news/935/a-look-at-tesla-s-dojo-supercomputer-shared-at-hot-chips-34 |
| Press | Electrek — Dojo deep-dive presentations | https://electrek.co/2022/08/24/tesla-deep-dive-presentations-dojo-ai-supercomputer/ |
| Press | The Register — Dojo overview | https://www.theregister.com/2022/08/24/tesla_supercomputer_dojo/ |
| Encyclopedia | Wikipedia — Tesla Dojo | https://en.wikipedia.org/wiki/Tesla_Dojo |
| Blog | Towards Data Science — AI Day 2021 Dojo review | https://towardsdatascience.com/tesla-ai-day-2021-review-part-3-project-dojo-teslas-new-supercomputer-715d102dbb29/ |

---

## 10. Roadmap Resources Added 2026-08-08

*None of these disclose Dojo hardware. They are program, corporate, and supply-chain sources for the Dojo 3 / AI5 / AI6 / Terafab picture.*

### 10.1 Primary filings

| Resource | URL | Notes |
|---|---|---|
| SpaceX Form S-1 (filed 2026-05-20; CIK 0001181412, accession 0001628280-26-036936) | https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm | **Best primary source in this batch.** Terafab risk factors verbatim; March 2026 announcement date; Intel joined April 2026; "two kinds of chips" (terrestrial edge/inference vs space). Contains **no** mention of "Dojo," "AI5," "AI6," or "14A" |
| SEC full-text search — "Terafab" in S-1 filings | https://efts.sec.gov/LATEST/search-index?q=%22Terafab%22&forms=S-1 | Filing-level discovery |
| SEC submissions index — SpaceX | https://data.sec.gov/submissions/CIK0001181412.json | Filing history |
| USPTO SN 99893717 — Tesla "MEGAPOD" trademark (~2026-06-18) | (USPTO TSDR, search by serial number) | Trademark filing only; hardware goods description; no chip or software named |

### 10.2 Dojo 3 program

| Resource | URL | Notes |
|---|---|---|
| TechCrunch — "Elon Musk says Tesla's restarted Dojo3 will be for space-based AI compute" (2026-01-20) | https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute/ | Current statement of Dojo 3's purpose |
| Tom's Hardware — Musk restarts Dojo3 space supercomputer project (2026-01-20) | https://www.tomshardware.com/tech-industry/supercomputers/elon-musk-restarts-dojo3-space-supercomputer-project-as-ai5-chip-design-gets-in-good-shape-will-be-first-tesla-built-supercomputer-to-feature-all-in-house-hardware-with-no-help-from-nvidia | Restart coverage |
| American Bazaar — Musk confirms progress on AI6, Dojo3 (2026-04-16) | https://www.americanbazaaronline.com/2026/04/16/tesla-ai5-chip-elon-musk-confirms-progress-on-ai6-dojo3-systems/ | Secondary record of the 2026-04-15 X post wording |
| Tesla Q2 2025 earnings call transcript (2025-07-23) | https://www.insidermonkey.com/blog/tesla-inc-nasdaqtsla-q2-2025-earnings-call-transcript-1575336/ | Verbatim "Dojo 3 and the AI6 inference chip … 5, 12 on a board" — **superseded framing** |

### 10.3 AI5 / AI6

| Resource | URL | Notes |
|---|---|---|
| Tom's Hardware — Musk demonstrates first AI5 sample (2026-04-15/16) | https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc-rather-than-tsmc-claims-40x-performance-boost-over-the-predecessor | AI5 tape-out; 40× claim is a **vendor marketing claim**, not independently confirmed |
| Not a Tesla App — AI5 done, not coming to vehicles first | https://www.notateslaapp.com/news/3974/musk-says-teslas-ai5-chip-is-done-not-coming-to-vehicles-first | Establishes it is AI5, not AI6, that skips vehicles first |
| Teslarati — Tesla finalizes AI5 chip design | https://www.teslarati.com/tesla-finalizes-ai5-chip-design-elon-musk-makes-bold-claim-capability/ | AI5 design status |
| Teslarati — AI6 self-driving chip expectations (2026-03-21) | https://www.teslarati.com/elon-musk-teases-expectations-tesla-ai6-self-driving-chip/ | AI6 as an FSD chip |
| Teslarati — Tesla trademarks MEGAPOD | https://www.teslarati.com/tesla-just-trademarked-megapod-heres-what-it-is/ | Trademark coverage |

### 10.4 Terafab

| Resource | URL | Notes |
|---|---|---|
| Tom's Hardware — Terafab takes shape: 100M sq ft, $16.8B initial capital (2026-08-07) | https://www.tomshardware.com/tech-industry/semiconductors/terafab-starts-to-take-shape-100-million-square-feet-of-manufacturing-space-and-usd16-8b-initial-capital-investment | $16.8B is **initial-stage only**; SpaceX's May 2026 Texas filings imply ~$55B phase one, up to ~$119B total |
| Tom's Hardware — SpaceX IPO risk factors; Terafab may not be successful | https://www.tomshardware.com/tech-industry/artificial-intelligence/spacex-admits-it-cant-find-enough-chips-for-orbital-ai-yet-requires-significantly-more-than-are-currently-available-to-us-firms-risk-factors-in-ipo-paperwork-also-says-ambitious-terafab-project-may-not-be-successful | Secondary coverage of the S-1 risk factors |
| Wikipedia — Terafab | https://en.wikipedia.org/wiki/Terafab | Overview / date reconciliation |

### 10.5 Negative checks

| Resource | URL | Notes |
|---|---|---|
| Hot Chips 38 advance program (Aug 23–25, 2026) | https://hotchips.org/advance-program/ | Checked 2026-08-08: **no Tesla or Dojo talk**. Conference is 15 days in the future — no slides or abstracts exist for any listed talk |
| `teslamotors/ttpoe` | https://github.com/teslamotors/ttpoe | Re-checked 2026-08-08: no new release since the Hot Chips 36 open-sourcing |
| Tom's Hardware — custom AI ASICs examined (May 2026) | https://www.tomshardware.com/tech-industry/semiconductors/custom-ai-asics-examined-from-broadcom-to-mtia | Source of a **362 TFLOPS BF16** D1 figure. **Not adopted** — the survey keeps the Hot Chips 34 figure of 376 TFLOPS |

> **Methodology note:** a publisher tag page (e.g. a `/tag/dojo` index) showing no recent items is *not* valid evidence of absence and was not used as one here.

---

## Summary Assessment

Tesla Dojo is one of the **most architecturally distinctive** AI training systems ever disclosed publicly, but also one of the **least open** in terms of tooling:

- **Well documented:** D1 chip microarchitecture (HC34 PDF), system hierarchy (ExaPOD/tile/cabinet), TTPoE networking (HC2024 PDF + open-source repo).
- **Partially documented:** Software stack principles (PyTorch-based, custom compiler with LLVM backend) from HC34 talks and job postings.
- **Not public:** Compiler source, ISA spec, SDK, kernel libraries, runtime source.
- **Project risk:** As of April 2026, the Dojo program is effectively dead (D1/D2 line cancelled in Aug 2025). The TTPoE GitHub repo (`teslamotors/ttpoe`) is the only open-source artifact of ongoing value.

**Update 2026-08-08.** A Dojo 3 program is acknowledged and staffed (Anant Nivarti rejoined Tesla in April 2026 to lead silicon engineering including AI6 and Dojo 3), and its stated target is space-based/orbital compute. But the status of this survey's *technical* content is unchanged: the D1 / Training Tile / ExaPOD specifications are still the latest disclosed Dojo hardware, TTPoE is still the only open-source artifact, and **every Dojo 3 architectural question remains "not disclosed."** The main new material is corporate and supply-chain context (Terafab / Intel 14A, framework agreement only) rather than anything about the accelerator itself.
