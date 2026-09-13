# OpenAI-Broadcom Jalapeño — Search Results

**Device class:** Custom AI Accelerator ("Intelligence Processor", OpenAI's term)
**Research date:** 2026-08-08 (first consolidated search-results file for this chip; baseline research 2026-04-05)

This chip previously had no `search-results.md`. This file consolidates the resource set discovered
during the 2026-08-08 scan of the 2026-06-24 Jalapeño unveiling, plus the baseline sources already
cited in `investigations/`.

Every entry is tagged by evidence class. Nothing here is a spec source unless marked **primary**.

---

## Layer 1 — Device Overview

**Jalapeño** is OpenAI's first custom silicon, co-developed with Broadcom (ASIC implementation +
Tomahawk Ethernet networking) and Celestica (board / rack system integration), manufactured by TSMC.
Unveiled 2026-06-24 as an LLM-inference-optimized ASIC. Silicon status as of 2026-08-08: **sampling**
(engineering samples running ML workloads in the lab, including GPT-5.3-Codex-Spark, at production
target frequency and power). Not in mass production; not deployed at scale.

The "Project Titan" codename used in earlier revisions of this survey has **no primary source** and
should not be cited.

---

## Layer 2 — Primary Sources (vendor / official)

| Resource | URL | Notes |
|---|---|---|
| **Broadcom IR — "OpenAI and Broadcom Unveil LLM-Optimized Intelligence Processor"** (2026-06-24) | https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-unveil-llm-optimized-intelligence-processor | **Durable primary source for the unveiling.** Page timed out on direct fetch; content corroborated via Nasdaq and Yahoo mirrors |
| Nasdaq press-release mirror (2026-06-24) | https://www.nasdaq.com/press-release/openai-and-broadcom-unveil-llm-optimized-intelligence-processor-2026-06-24 | Mirror of the joint release |
| Yahoo Finance carry (full verbatim joint release) | https://finance.yahoo.com/technology/ai/articles/openai-broadcom-unveil-llm-optimized-130000020.html | The most complete retrievable copy of the release text |
| OpenAI announcement page — "OpenAI and Broadcom unveil LLM-optimized inference chip" | https://openai.com/index/openai-broadcom-jalapeno-inference-chip/ | **Returns HTTP 403 to automated fetch**; title and snippet confirmed via search index only. Do not cite a Wayback snapshot as the source of record — use the Broadcom IR / Nasdaq release |
| OpenAI-Broadcom strategic collaboration (2025-10-13) | https://openai.com/index/openai-and-broadcom-announce-strategic-collaboration/ | Original 10 GW partnership announcement |
| Broadcom IR — strategic collaboration (2025-10-13) | https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-announce-strategic-collaboration-deploy-10 | Original 10 GW partnership announcement |
| Broadcom IR — Jericho4 shipping | https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-jericho4-enabling-distributed-ai-computing-across | Portfolio context only. **Jericho4 is not named in any OpenAI/Broadcom Jalapeño material** |
| OpenAI careers — Triton compiler engineer | https://openai.com/careers/software-engineer-triton-compiler-san-francisco/ | Software-stack signal: team "advancing Triton and its backend" for custom silicon |
| Triton compiler repository | https://github.com/triton-lang/triton | No Jalapeño backend present as of 2026-08-08 |

---

## Layer 3 — Independent Press (unveiling, 2026-06-24 onward)

| Resource | URL | What it independently confirms |
|---|---|---|
| Engadget | https://www.engadget.com/2201045/openai-broadcom-jalapeno-inference-processor-ai-accelerator/ | "First Intelligence Processor"; ~nine-month design cycle; perf-per-watt claim only; technical report promised; late-2026 deployment; multi-generation platform |
| TechCrunch | https://techcrunch.com/2026/06/24/openai-unveils-its-first-custom-chip-built-by-broadcom/ | 2026-06-24 unveiling; chip **still in testing**, "early results"; inference-focused; no numbers |
| CNBC | https://www.cnbc.com/2026/06/24/openai-and-broadcom-reveal-jalapeno-first-ai-chip-in-partnership.html | Name and date (body returned HTTP 403; headline/snippet only) |
| Forbes (Jon Markman, 2026-06-25) | https://www.forbes.com/sites/jonmarkman/2026/06/25/meet-jalapeo-openais-first-custom-ai-chip-built-with-broadcom/ | Name, partnership, inference targeting |
| Tom's Hardware — unveiling + package/die analysis | https://www.tomshardware.com/tech-industry/artificial-intelligence/broadcom-and-openai-unveil-custom-built-jalapeno-inference-processor-openais-first-chip-is-a-massive-reticle-sized-asic-built-in-an-ultra-fast-nine-month-development-cycle | **Third-party estimate (medium confidence)**: compute chiplet ~25.46 × 33 mm ≈ 840 mm² vs ~858 mm² reticle limit; six HBM stacks; separate I/O chiplet; two structural dummy dies; regular columnar wafer floorplan. Journalist inference from photos, **not a vendor spec** |
| Tom's Hardware — custom AI ASIC survey (Broadcom to MTIA) | https://www.tomshardware.com/tech-industry/semiconductors/custom-ai-asics-examined-from-broadcom-to-mtia | Cross-vendor custom-ASIC context; node reporting on the 10 GW program |

---

## Layer 4 — Baseline Press (Oct 2025 – Apr 2026)

| Resource | URL | Notes |
|---|---|---|
| TechCrunch (Oct 2025) | https://techcrunch.com/2025/10/14/openai-and-broadcom-partner-on-ai-hardware/ | Partnership announcement coverage |
| CNBC (Oct 2025) | https://www.cnbc.com/2025/10/13/openai-partners-with-broadcom-custom-ai-chips-alongside-nvidia-amd.html | Partnership announcement coverage |
| Tom's Hardware — chip finalization / architecture | https://www.tomshardware.com/tech-industry/artificial-intelligence/openai-and-broadcom-to-finalize-custom-ai-processor-in-the-coming-months-say-industry-sources | Source of the "systolic array" architecture reporting (industry sources, not vendor) |
| DCD — first custom inference chip | https://www.datacenterdynamics.com/en/news/openai-building-first-custom-ai-inference-chip-with-tsmc-and-broadcom-report/ | TSMC N3 reporting |
| DCD — mass production 2026 | https://www.datacenterdynamics.com/en/news/openai-to-start-mass-producing-its-custom-ai-chip-in-2026-report/ | Superseded by the 2026-06-24 sampling status |
| TrendForce — N3 Gen 1 / A16 Gen 2 roadmap (2026-01-15) | https://www.trendforce.com/news/2026/01/15/news-openai-reportedly-to-deploy-custom-ai-chip-on-tsmc-n3-by-end-2026-second-gen-planned-for-a16/ | **Reported, not confirmed.** The A16/2nm Gen 2 element could not be corroborated in primary sources |

---

## Layer 5 — Conference / Scheduled Disclosure

| Resource | URL | Notes |
|---|---|---|
| Hot Chips 38 advance program | https://www.hotchips.org/advance-program/ | Conference 2026-08-23 to 2026-08-25. **OpenAI**: Richard Ho, Ravi Narayanaswami, Chris Leary — "You Can Just Build Things … Chips", AI 2 session (chair Brucek Khailany), Tue 2026-08-25 4:45–6:15 PM PDT. **Broadcom**: Hemal Shah — "Thor Ultra: An Ethernet NIC Chip Optimized for AI & HPC", same day. *Disclosure scheduled — content not yet public.* No slides, abstract or spec exists as of 2026-08-08; nothing in this survey is sourced from either talk |

**Follow-up action:** re-scan after 2026-08-25. This is the most likely venue for the first real
Jalapeño microarchitecture disclosure.

---

## Layer 6 — Low-Confidence / Rumor

| Resource | URL | Disposition |
|---|---|---|
| The Information — Arm server CPU for the OpenAI/Broadcom rack | https://www.theinformation.com/articles/openai-working-softbanks-arm-broadcom-ai-chip-effort | **Rumor.** Unnamed sources; never confirmed by Arm, OpenAI or Broadcom; publication date not independently verified |
| Tom's Hardware carry of the Arm CPU report | https://www.tomshardware.com/pc-components/cpus/openai-arm-partner-on-custom-cpu-for-broadcom-chip | Same rumor; article body truncated on fetch |
| Medium — "Titan chip: the $500 billion bet" | https://medium.com/hardware-for-agi/openais-titan-chip-the-500-billion-bet-to-break-free-from-nvidia-3e077eca3be9 | Blog commentary; not a spec source |

---

## Layer 7 — Excluded / Do Not Cite

| Item | Why excluded |
|---|---|
| tech-insider.org — "OpenAI Titan chip Samsung HBM4" (https://tech-insider.org/openai-titan-chip-samsung-hbm4-custom-ai-chip-2026/) | Aggregator/SEO source. Sole basis for the "Samsung HBM4 exclusive supply" line, which the repo previously recorded as *confirmed*. **Downgraded to reported-not-confirmed on 2026-08-08.** Neither company has ever stated the HBM generation |
| "Project Titan" codename (opentools.ai 2025-09-08, tech-insider.org, xagi-labs blog) | No primary source. Not used by OpenAI or Broadcom. **Unsourced** |
| "50% cheaper than Nvidia" (tech-insider.org, 2026-07-25) | Absent from every primary source; no benchmark of any kind exists for Jalapeño. **Aggregator noise** |
| web.archive.org capture 20260624175604 of the OpenAI page | Unreachable from the scanning environment and not a durable citation. Use the Broadcom IR / Nasdaq release instead |
| "Jalapeño will be sold to third parties" | Never stated. Inferred by some coverage from OpenAI's "works with all LLMs" language, which is an architecture statement only |

---

## Layer 8 — Known Gaps After This Scan

- Microarchitecture: PE array dimensions, tile size, dataflow, sparsity, attention handling
- Peak throughput at any precision; supported data types
- On-chip SRAM capacity and bandwidth
- HBM generation, supplier, capacity, bandwidth
- Vendor-stated process node, transistor count, TDP, clock
- Scale-up fabric silicon and bandwidth; topology; host interface generation
- Chips per rack, rack power, rack topology
- ISA, compiler IR, runtime and driver architecture; no SDK exists
- Any independent or standardized benchmark result
- Exact wording of the "built from the ground up for current and future LLMs" passage (search budget
  exhausted before this could be closed out)
- The May 2026 process-node survey referenced in secondary coverage (could not be retrieved)
