# OpenAI-Broadcom Jalapeño — Summary

*as_of: 2026-08-08*
*Device class: Custom AI Accelerator ("Intelligence Processor", OpenAI's term)*
*Status: Sampling — engineering samples running ML workloads in the lab (announced 2026-06-24); initial deployment targeted by end of 2026; NOT in mass production*

---

## One-Line Summary

Jalapeño is OpenAI's first custom LLM-inference accelerator, co-developed with Broadcom (ASIC implementation + Tomahawk Ethernet networking) and Celestica (board/rack integration), unveiled on 2026-06-24 with engineering samples already running GPT-5.3-Codex-Spark in the lab; it anchors the 10 GW OpenAI-Broadcom compute program, with initial deployment targeted for late 2026 and no performance figures disclosed.

---

## Partnership Overview

| Parameter | Value |
|-----------|-------|
| Chip designer | OpenAI (VP Hardware: Richard Ho, ex-Google TPU) |
| ASIC partner | Broadcom (design services + silicon implementation + Tomahawk networking) |
| System partner | Celestica (board, rack system integration, scalable production systems) — named 2026-06-24 |
| Foundry | TSMC |
| Partnership announced | October 13, 2025 |
| Chip unveiled | June 24, 2026 (name: Jalapeño) |
| Working together since | Early 2024 (~18 months before the Oct 2025 announcement) |
| Scale commitment | 10 GW of OpenAI-designed AI accelerators |
| Deployment timeline | Initial deployment targeted by end of 2026 → 2029 completion (stated target, not an achieved milestone) |
| Named deployment partner | Microsoft (per Broadcom CEO Hock Tan, 2026-06-24) |
| Estimated cost | $350B–$500B (Financial Times estimate) |

---

## Chip Specifications (Jalapeño, Gen 1)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Official name | **Jalapeño** — "OpenAI's first Intelligence Processor" | Confirmed (OpenAI + Broadcom joint release, 2026-06-24) |
| Prior repo codename | "Project Titan" — **unsourced**; appears only on aggregator/SEO sites, never in a primary source. Retained here only as an alias readers may encounter | Unsourced — do not cite |
| Architecture | Systolic / tiled matrix engine (reported; consistent with the regular columnar wafer floorplan seen in released die photography) | Reported, not vendor-confirmed |
| Process | Not disclosed by either company on 2026-06-24. Prior reporting on the 10 GW program consistently references a 3nm-class (TSMC N3) first deployment | Reported, not confirmed |
| Compute die | ~25.46 mm × 33 mm ≈ **840 mm²** (against the ~858 mm² EUV reticle limit) | Third-party estimate — Tom's Hardware analysis of released package/wafer photos; medium confidence |
| Package composition | One large compute chiplet + a separate I/O chiplet + **six HBM stacks** + two structural dummy dies | Third-party estimate (same photo analysis); medium confidence |
| Memory generation | **Not disclosed.** Neither HBM3E nor HBM4 was stated at unveiling. A Samsung HBM4 exclusive-supply arrangement was *reported* in early 2026 but is neither confirmed nor contradicted by the June 2026 disclosure | Reported, not confirmed |
| Peak TFLOPS / TOPS | Not disclosed (any precision) | — |
| Data types | Not disclosed (BF16/FP8 inferred from the architecture class and inference target) | Inferred |
| TDP | Not disclosed | — |
| Networking | Broadcom all-Ethernet. **Tomahawk explicitly confirmed** by both companies; Jericho4's role remains inferred and unconfirmed | Mixed (see Interconnect) |
| Host interface | Not disclosed (PCIe Gen5/6 inferred) | Inferred |
| Design cycle | Initial design → manufacturing tape-out in **nine months**, claimed as "the fastest ASIC development cycle ever achieved in high-performance advanced semiconductors" | **Vendor marketing claim** — unverifiable |
| Status | **Sampling.** Engineering samples running ML workloads in the lab at production target frequency and power, including GPT-5.3-Codex-Spark. Not mass production, not deployed at scale | Confirmed (vendor statement, 2026-06-24) |

---

## Roadmap

| Generation | Process | Memory | Status | Evidence |
|------------|---------|--------|--------|----------|
| Jalapeño (Gen 1) | Not disclosed; 3nm-class (TSMC N3) per prior reporting | Not disclosed (6 stacks estimated from photos) | Engineering samples in lab as of 2026-06-24; initial deployment targeted end of 2026 | Confirmed status; node reported-not-confirmed |
| Gen 2 ("Titan 2" in earlier reporting) | TSMC A16 (1.6nm) | Not disclosed | Reported development start H2 2026, deploy 2027+ | **Unconfirmed roadmap reporting** — the 2nm/A16 element was not corroborated in primary sources and is not vendor-stated |
| Beyond | Not disclosed | Not disclosed | Hock Tan: "This is just the beginning of a multi-generation roadmap" | Vendor statement (no specifics) |

---

## Architecture Overview

Jalapeño is reported to use a **systolic / tiled matrix engine** — the same fundamental class as Google TPU, AWS Trainium, and Microsoft Maia. Unlike NVIDIA GPUs (SIMT) or Etched Sohu (fixed-function transformer ASIC), systolic arrays are compiler-programmed: the compiler tiles operations into the array's regular data flow. The wafer/die photography released on 2026-06-24 shows a "very regular, repeated, columnar" floorplan, which third-party analysts read as consistent with this class — but OpenAI and Broadcom have not stated the microarchitecture.

```
PyTorch / Model Definition
         ↓
torch.compile + Triton compiler (Jalapeño backend — not public)
         ↓
Jalapeño compute chiplet (~840 mm², third-party estimate; node not disclosed)
  ├── Matrix multiply engine (tiled/systolic — dimensions not public)
  ├── Vector/activation units (not public)
  └── On-chip SRAM buffers (not public)
         ↓
Separate I/O chiplet (third-party estimate from package photos)
         ↓
HBM — six stacks (third-party estimate); generation NOT disclosed
         ↓
Broadcom all-Ethernet fabric
  ├── Broadcom Tomahawk Ethernet (CONFIRMED by OpenAI + Broadcom, 2026-06-24)
  └── Broadcom Jericho4 AI switch (inferred — role NOT confirmed; absent from the announcement)
         ↓
Celestica board / rack system integration
```

**Key design choices:**
- **All-Ethernet networking**: OpenAI and Broadcom chose Ethernet over InfiniBand (Nvidia's preferred fabric), aligned with the Ultra Ethernet Consortium (UEC). Tomahawk silicon is now named explicitly by both companies
- **Reticle-scale compute chiplet + separate I/O die**: the package is multi-die (compute chiplet, I/O chiplet, six HBM stacks, two structural dummy dies) per third-party photo analysis — not a monolithic accelerator
- **Memory generation undisclosed**: HBM3E vs HBM4 has never been stated by either company; the earlier "Samsung HBM4 exclusive" report remains uncorroborated by any primary source
- **Process node undisclosed**: 3nm-class first deployment is consistent with prior reporting on the 10 GW program but was not restated at unveiling; the A16/2nm Gen 2 element is unconfirmed
- **Triton compiler**: OpenAI's own ML compiler extended for Jalapeño silicon (internal; not open-sourced)
- **Inference-first, model-agnostic**: OpenAI describes Jalapeño as built for "current and future LLMs across the industry" rather than locked to its own models. This is an architecture/flexibility statement — **OpenAI has made no commitment to sell Jalapeño to third parties**, and no such claim should be read into it
- **Training still relies on NVIDIA**: Jalapeño targets LLM inference (OpenAI's API revenue driver)

---

## Software Stack

The Jalapeño software stack is almost entirely not public. Key signals:

| Layer | Status | Notes |
|-------|--------|-------|
| Framework | PyTorch (inferred) | OpenAI's entire infrastructure uses PyTorch |
| Compiler | Triton + Jalapeño backend (confirmed in job postings; not public) | OpenAI team "advancing Triton and its backend" for custom silicon |
| Kernel library | Not public | Written as Triton kernels targeting Jalapeño |
| Runtime | Not public | Internal OpenAI inference infrastructure |
| Driver | Not public | PCIe kernel-mode driver (inferred) |
| Networking | Broadcom Ethernet (not public) | Broadcom collective ops library (NCCL-equivalent) |
| SDK publicly released | No | Internal only; no developer access |

**Bring-up milestone (2026-06-24)**: engineering samples are running ML workloads in the lab — including **GPT-5.3-Codex-Spark** — at production target frequency and power. This is the first evidence that the compiler, runtime and driver path exist end-to-end well enough to execute a real production-class OpenAI model on the silicon. No detail about any of those layers was released, and no SDK exists.

---

## Strategic Significance

**Why OpenAI is building custom silicon:**
1. **Cost reduction**: 90% inference cost reduction vs. equivalent GPU workloads (stated goal, Richard Ho)
2. **Architecture specialization**: Embed knowledge from frontier model development directly into hardware
3. **Nvidia dependency reduction**: 10 GW of custom compute reduces long-term Nvidia reliance
4. **Competitive moat**: Custom silicon optimized for OpenAI's specific model architectures and serving patterns

**Why Broadcom (not just TSMC)?**
Broadcom brings:
- Custom ASIC design expertise (largest custom silicon design shop outside the hyperscalers)
- End-to-end Ethernet networking (Jericho4 AI fabric + Tomahawk switches + optical)
- PCIe and connectivity IP
- Existing relationships: Google (TPU), Meta (MTIA), and now OpenAI all use Broadcom ASIC services

**Why Ethernet over InfiniBand?**
- No per-port Nvidia tax
- Standards-based (Ultra Ethernet Consortium)
- Broadcom's core strength
- Potentially lower cost at scale

---

## Limitations and Risks

| Risk | Assessment |
|------|-----------|
| Software stack maturity | Highest risk: the Triton backend for Jalapeño + inference runtime must scale from lab bring-up (GPT-5.3-Codex-Spark on engineering samples) to fleet-scale serving by end of 2026 |
| Training still needs NVIDIA | Jalapeño is inference-focused; OpenAI needs NVIDIA H100/B200 for training frontier models |
| No independent benchmarks | **Still true as of 2026-08-08.** No numbers of any kind were disclosed at unveiling — only unquantified perf-per-watt and utilization claims, with a technical report promised "in the coming months". No MLPerf submission, no third-party benchmark. A "50% cheaper than Nvidia" figure circulating on aggregator sites does not appear in any primary source and is excluded from this survey |
| Memory sourcing unknown | The HBM generation and supplier were never stated by either company; the "Samsung HBM4 exclusive" report is uncorroborated, so supply-concentration risk cannot be assessed from public information |
| Ethernet for scale-up | Less proven than NVLink for tightly coupled training at scale; only Tomahawk (scale-out class) is confirmed — the scale-up fabric silicon has not been named |
| Timeline slip risk | Silicon is at engineering-sample stage as of June 2026. End-2026 initial deployment is a stated target, not a de-risked milestone |
| Broadcom revenue dependence | Broadcom's Q1 2026 AI revenue +106% YoY ($8.4B) driven partly by custom silicon — mutual dependency |
| Reticle-limit die | If the ~840 mm² compute-chiplet estimate is right, Jalapeño sits within ~2% of the EUV reticle limit — yield and cost exposure are correspondingly high. This is a third-party photo estimate, not a disclosed figure |

---

## Jalapeño Unveiling Update (June 2026)

*Updated 2026-08-08. Primary sources: joint OpenAI/Broadcom press release of 2026-06-24 (Broadcom IR / Nasdaq mirrors); independent coverage from Engadget, TechCrunch, CNBC, Forbes; package/die analysis from Tom's Hardware. Prior-generation content above is preserved; this section records what changed.*

### What was announced

On **2026-06-24** OpenAI and Broadcom jointly unveiled **Jalapeño**, described by OpenAI as its "first Intelligence Processor" and by the joint release as an "LLM-Optimized Intelligence Processor". This is the first time the chip has had a public name and the first release of any package or die imagery.

| Item | Before (repo baseline, 2026-04-05) | After (2026-06-24 disclosure) |
|---|---|---|
| Name | "Project Titan" (internal codename) | **Jalapeño** — official, in both companies' releases. "Titan" has no primary source and is demoted to unsourced |
| Silicon status | "Not yet shipped; mass production targeted H2 2026" | **Sampling** — engineering samples running ML workloads in the lab at production target frequency and power |
| Workload proof point | None | **GPT-5.3-Codex-Spark** running on engineering samples in the lab |
| System partners | Broadcom, TSMC | + **Celestica** (board, rack system integration, scalable production systems) |
| Scale-out networking | Tomahawk (inferred) | **Tomahawk confirmed** by both companies |
| Scale-up networking | Jericho4 (inferred) | **Still inferred** — Jericho4 is not mentioned in the announcement |
| Deployment partners | OpenAI facilities + unnamed partner DCs | + **Microsoft** named by Broadcom CEO Hock Tan |
| Die / package | Not public | Third-party estimate: ~840 mm² compute chiplet + I/O chiplet + 6 HBM stacks + 2 dummy dies |
| Process node | "TSMC N3 (3nm)" stated as fact | **Not disclosed** by either company; 3nm-class is prior reporting, not confirmation |
| Memory | "Samsung HBM4 (exclusive supply agreement)" stated as confirmed | **Not disclosed** — generation never stated; the HBM4 report is downgraded to reported-not-confirmed |

### Development cycle (vendor marketing claim)

The companies state that Jalapeño went "from initial design to manufacturing tape-out in just nine months", which they call "what we believe to be the fastest ASIC development cycle ever achieved in high-performance advanced semiconductors". OpenAI adds that it used its own models to accelerate parts of the design and optimization process. Both statements are unverifiable vendor claims and are recorded as such.

### Performance — nothing quantitative was released

The only performance language in the announcement is:
- "performance per watt substantially better than current state-of-the-art"
- realized utilization "much closer to theoretical peak"
- a detailed technical report on performance promised "in the coming months"

There is **no** disclosed peak throughput, memory bandwidth, TDP, clock, or efficiency figure, **no** MLPerf submission, and **no** third-party benchmark. TechCrunch independently characterises the chip as still in testing with "early results".

### Package and die — third-party estimate, medium confidence

Tom's Hardware analysed the released package and wafer photography and estimates:

| Element | Estimate | Nature of evidence |
|---|---|---|
| Compute chiplet | ~25.46 mm × 33 mm ≈ **840 mm²** (EUV reticle limit ≈ 858 mm²) | Journalist measurement from photos |
| HBM stacks | **Six**, surrounding the compute chiplet | Photo count |
| I/O chiplet | One, separate from the compute die | Photo identification |
| Structural dummy dies | Two | Photo identification |
| Wafer floorplan | "Very regular, repeated, columnar" — read as consistent with a tiled/systolic matrix engine | Photo inference |

None of this comes from a vendor spec sheet. It must be attributed as an estimate everywhere it is reused.

### Scheduled disclosure — Hot Chips 38

Richard Ho, Ravi Narayanaswami and Chris Leary (OpenAI) are scheduled to present **"You Can Just Build Things … Chips"** in the AI 2 session at Hot Chips 38, **Tuesday 2026-08-25, 4:45–6:15 PM PDT** (conference Aug 23–25; session chair Brucek Khailany). Broadcom separately presents "Thor Ultra: An Ethernet NIC Chip Optimized for AI & HPC" (Hemal Shah) the same day.

*Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* No slides, abstract, or specification from either talk exists as of 2026-08-08, and nothing in this survey is sourced from them. This is the most likely venue for the first real Jalapeño microarchitecture disclosure; a follow-up scan after 2026-08-25 is warranted.

### Unconfirmed / excluded

| Item | Disposition |
|---|---|
| "Project Titan" codename | Unsourced — aggregator sites only; not used by OpenAI or Broadcom |
| "50% cheaper than Nvidia" | Aggregator noise; absent from all primary sources — excluded |
| Jalapeño sold to third parties | **Not stated.** OpenAI's "works with all LLMs" language is an architecture statement; no sales commitment exists. Excluded as speculation |
| TSMC A16 / 2nm Gen 2 | Unconfirmed roadmap reporting; not corroborated in primary sources |
| Samsung HBM4 exclusive supply | Reported (aggregator-sourced), not confirmed; neither confirmed nor contradicted by the June 2026 disclosure |
| Arm server CPU pairing | **Rumor.** The Information reported that Arm is designing a server-class CPU to anchor OpenAI's next-generation racks alongside the Broadcom accelerator, potentially pairable with NVIDIA and AMD parts too. Unnamed sources; never confirmed by Arm, OpenAI or Broadcom; publication date not independently verified. Low confidence |

---

## Resources

### Jalapeño unveiling (2026-06-24)

- Broadcom IR press release (primary): https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-unveil-llm-optimized-intelligence-processor
- Nasdaq press-release mirror: https://www.nasdaq.com/press-release/openai-and-broadcom-unveil-llm-optimized-intelligence-processor-2026-06-24
- Yahoo Finance carry (full verbatim joint release): https://finance.yahoo.com/technology/ai/articles/openai-broadcom-unveil-llm-optimized-130000020.html
- OpenAI announcement page (returns HTTP 403 to automated fetch; title/snippet confirmed via search index): https://openai.com/index/openai-broadcom-jalapeno-inference-chip/
- Engadget: https://www.engadget.com/2201045/openai-broadcom-jalapeno-inference-processor-ai-accelerator/
- TechCrunch: https://techcrunch.com/2026/06/24/openai-unveils-its-first-custom-chip-built-by-broadcom/
- CNBC: https://www.cnbc.com/2026/06/24/openai-and-broadcom-reveal-jalapeno-first-ai-chip-in-partnership.html
- Forbes: https://www.forbes.com/sites/jonmarkman/2026/06/25/meet-jalapeo-openais-first-custom-ai-chip-built-with-broadcom/
- Tom's Hardware — package/die photo analysis (~840 mm², six HBM stacks, I/O chiplet): https://www.tomshardware.com/tech-industry/artificial-intelligence/broadcom-and-openai-unveil-custom-built-jalapeno-inference-processor-openais-first-chip-is-a-massive-reticle-sized-asic-built-in-an-ultra-fast-nine-month-development-cycle
- Hot Chips 38 advance program (scheduled talk; content not yet public): https://www.hotchips.org/advance-program/
- The Information — Arm server CPU report (rumor, unconfirmed): https://www.theinformation.com/articles/openai-working-softbanks-arm-broadcom-ai-chip-effort

### Partnership announcement (2025-10-13) and earlier

- OpenAI announcement: https://openai.com/index/openai-and-broadcom-announce-strategic-collaboration/
- Broadcom announcement: https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-announce-strategic-collaboration-deploy-10
- TechCrunch (Oct 2025): https://techcrunch.com/2025/10/14/openai-and-broadcom-partner-on-ai-hardware/
- CNBC (Oct 2025): https://www.cnbc.com/2025/10/13/openai-partners-with-broadcom-custom-ai-chips-alongside-nvidia-amd.html
- TrendForce N3/A16 roadmap: https://www.trendforce.com/news/2026/01/15/news-openai-reportedly-to-deploy-custom-ai-chip-on-tsmc-n3-by-end-2026-second-gen-planned-for-a16/
- Samsung HBM4 deal: https://tech-insider.org/openai-titan-chip-samsung-hbm4-custom-ai-chip-2026/
- Tom's Hardware — architecture details: https://www.tomshardware.com/tech-industry/artificial-intelligence/openai-and-broadcom-to-finalize-custom-ai-processor-in-the-coming-months-say-industry-sources
- Triton compiler (OpenAI): https://github.com/triton-lang/triton
- Triton job posting (stack signal): https://openai.com/careers/software-engineer-triton-compiler-san-francisco/
