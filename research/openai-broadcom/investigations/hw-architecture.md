# OpenAI-Broadcom Jalapeño — Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: openai-broadcom*
*product name: Jalapeño (official, unveiled 2026-06-24)*
*prior repo codename: "Project Titan" — unsourced, see the 2026-08-08 section*
*confidence: low-to-medium (silicon at engineering-sample stage; microarchitecture still undisclosed)*

> **Sections 1–10 below are the April 2026 baseline investigation and are preserved as written.**
> Several of their statements were superseded or downgraded on 2026-06-24; see
> **"Jalapeño Unveiling Investigation (2026-08-08)"** at the end of this document, which is authoritative
> where the two disagree.

---

## Overview

OpenAI and Broadcom announced a strategic collaboration on October 13, 2025 to co-develop and deploy 10 gigawatts of OpenAI-designed custom AI accelerators. The chip, internally codenamed "Project Titan," is OpenAI's first custom silicon program. OpenAI leads chip architecture design; Broadcom provides ASIC design services, packaging integration, and end-to-end Ethernet networking. TSMC manufactures the silicon.

The program is led by Richard Ho, VP of Hardware at OpenAI, a former Google TPU engineer. The stated goal is a 90% reduction in inference costs versus equivalent GPU workloads.

---

## 1. Chip Compute Architecture

### 1.1 Architecture Class

The chip is reported to use a **systolic array architecture** — the same class as Google TPU, AWS Trainium, and Microsoft Maia — with design optimizations informed by OpenAI's model workloads. This is not a GPU (no SIMT), not a fixed-function ASIC (unlike Etched Sohu), and not a dataflow/reconfigurable architecture (unlike SambaNova).

| Attribute | Value |
|-----------|-------|
| Architecture type | Systolic array (reported) |
| Primary target | AI inference (primary), training (secondary, limited scale) |
| Programmability | Yes — compiler-programmed systolic array (unlike Etched Sohu) |
| ISA | Not public |
| Precision | Not disclosed (BF16/FP8 standard for this class, likely supported) |
| Peak TFLOPS | Not public |
| Die count | Not public (single die assumed for gen 1) |

### 1.2 Compute Unit Details

Not publicly disclosed. The systolic array configuration (array dimensions, tile size, sparsity support, attention accelerator presence) is not public as of April 2026.

---

## 2. Memory Architecture

### 2.1 Off-chip Memory

| Attribute | Value |
|-----------|-------|
| Memory type | HBM4 (Samsung exclusive supply agreement confirmed, Mar 2026) |
| Capacity | Not public |
| Bandwidth | Not public |
| HBM stacks | Not public |
| Packaging | Not public (CoWoS-L or SoIC inferred for HBM4 integration) |

**Note**: Samsung confirmed an exclusive HBM4 supply agreement for OpenAI's Titan chip program in early 2026. HBM4 provides higher bandwidth and capacity per stack than HBM3E (used in NVIDIA B200), enabling improved memory-bandwidth-bound inference throughput.

### 2.2 On-chip SRAM

Not publicly disclosed. Systolic arrays typically include:
- Weight-stationary accumulator buffers (local to PEs)
- Activation SRAM (unified buffer for compiler-managed tiling)
- Scratchpad for operator fusion

Specific capacity and bandwidth: not public.

---

## 3. Process Technology and Packaging

| Attribute | Value |
|-----------|-------|
| Foundry | TSMC |
| Gen 1 node | N3 (3nm) — reported by multiple industry sources |
| Gen 2 node | A16 (1.6nm, TSMC nanosheet + Super Power Rail) — reported |
| Gen 2 timeline | Development starts H2 2026; deployment 2027+ |
| Die size | Not public |
| Packaging | Not public (CoWoS-L inferred for HBM4 + systolic array) |
| Transistor count | Not public |
| TDP / Power | Not public |

**TSMC N3**: OpenAI's first gen chip targets TSMC's 3nm (N3/N3E) process, the same generation as Apple M3/A17 and AMD EPYC Genoa successors. N3 offers ~18% speed uplift and ~35% power reduction vs N5 at same transistor count.

**TSMC A16**: The second-generation chip (Titan 2) targets TSMC A16 (1.6nm-class), which uses nanosheet transistors and backside power delivery (Super Power Rail). A16 offers 8-10% speed gain and 15-20% power reduction vs N2P.

---

## 4. Scale-up Interconnect (Chip-to-Chip)

The rack system integrates compute, memory, and networking on Broadcom's Ethernet stack. Unlike NVIDIA (NVLink proprietary), OpenAI-Broadcom chose **all-Ethernet** for both scale-up (intra-rack) and scale-out (inter-rack).

| Attribute | Value |
|-----------|-------|
| Scale-up protocol | Ethernet (Broadcom Ultra Ethernet Consortium) |
| Scale-up chip | Broadcom Jericho4 (AI fabric) — inferred from Broadcom portfolio |
| Scale-up topology | Not public |
| Scale-up bandwidth | Not public |
| Inter-chip interconnect | Not public (dedicated chip-to-chip links vs. switched Ethernet: not confirmed) |

**Broadcom Ethernet context**: Broadcom's Jericho3-AI / Jericho4 switching chips are designed for AI fabrics, supporting AI collective operations (AllReduce, AllGather) natively over Ethernet with congestion-free operation. The OpenAI partnership uses "Broadcom's end-to-end portfolio of Ethernet, PCIe and optical connectivity solutions."

---

## 5. Scale-out Networking

| Attribute | Value |
|-----------|-------|
| Protocol | Ethernet (not InfiniBand) |
| NIC | Broadcom Tomahawk/Jericho-based switching + Broadcom Ethernet NICs (inferred) |
| Bandwidth per port | Not public (likely 400 GbE / 800 GbE at deployment) |
| Optical interconnect | Broadcom PCIe and optical solutions included per announcement |

**Strategic choice**: OpenAI and Broadcom selected Ethernet over InfiniBand (Nvidia's preferred AI fabric). This aligns with the Ultra Ethernet Consortium (UEC) initiative, which Google, Meta, AMD, Intel, and others support as an open alternative to Nvidia's proprietary networking stack.

---

## 6. Host Interface

Not publicly disclosed. PCIe Gen5 or Gen6 inferred from TSMC N3 timeline and 2026 deployment target.

---

## 7. Rack System Architecture

| Attribute | Value |
|-----------|-------|
| Deployment target | H2 2026 (start) → 2029 (full 10 GW deployment) |
| Rack contents | OpenAI Titan accelerators + Broadcom Ethernet fabric + Broadcom optical/PCIe |
| Scale | 10 GW of compute across OpenAI facilities and partner data centers |
| Estimated cost | $350B–$500B (Financial Times estimate for full 10 GW deployment) |
| Chips per rack | Not public |
| Power per rack | Not public |

---

## 8. Competitive Context

| Dimension | OpenAI Titan (Gen 1) | NVIDIA B200 | Google TPU v6 | AWS Trainium 2 |
|-----------|---------------------|-------------|---------------|----------------|
| Architecture | Systolic array (reported) | SIMT GPU | Systolic array | Systolic array |
| Process | TSMC N3 | TSMC 4NP | Not disclosed | TSMC 5nm (est.) |
| Memory | HBM4 (Samsung exclusive) | HBM3e | HBM3e | HBM2e |
| Networking | Ethernet (Broadcom) | NVLink/InfiniBand | ICI torus | NeuronLink/EFA |
| Target | Inference-first | Training + inference | Training + inference | Training + inference |
| Availability | H2 2026 (target) | Shipping | Shipping (TPUv5e/v6) | Shipping |
| Peak TFLOPS | Not public | 2,250 (FP16) | ~450 BF16 (v7) | 1,264 (cFP8) |
| Open software | Triton (compiler) + PyTorch | CUDA ecosystem | XLA/JAX | Neuron SDK |

---

## 9. Hardware Trade-off Analysis

| Design Choice | Advantage | Risk / Disadvantage |
|---------------|-----------|---------------------|
| Systolic array (not GPU) | Higher FLOPS/watt for matrix multiply; no warp scheduling overhead | Requires mature compiler; less flexible than GPU |
| All-Ethernet networking | Open standard; no Nvidia NVLink dependency; commodity switches | Higher latency than NVLink/InfiniBand for collective ops; less proven at scale |
| TSMC N3 (Gen 1) | Competitive process; available now for 2026 deployment | More expensive per wafer than N5; yield ramp risk |
| TSMC A16 (Gen 2) | Leading-edge nanosheet + backside power; 8-10% speed gain | Gen 2 not yet in mass production (2027+) |
| HBM4 (Samsung exclusive) | Highest bandwidth/capacity per stack | Exclusive supply creates single-vendor memory risk |
| Inference-first deployment | Cost-optimized for OpenAI's primary revenue workload | Training capabilities secondary; NVIDIA still needed for training |
| Broadcom partnership (not in-house fab) | ASIC expertise accelerates design; time-to-market | Dependency on Broadcom for future generations |

---

## 10. What Is Not Public

- Die floorplan, PE array dimensions, systolic tile size
- On-chip SRAM capacity and bandwidth
- Peak TFLOPS / TOPS (any precision)
- HBM4 stack count and capacity per chip
- Inter-chip scale-up bandwidth and topology details
- Host interface speed (PCIe generation and lane count)
- TDP / rack power density
- Chips per rack / racks per 10 GW deployment
- Software ISA or compiler IR specification
- Any independent benchmark results (chip not yet shipped)

---

## Jalapeño Unveiling Investigation (2026-08-08)

*Scan date 2026-08-08. Trigger event: joint OpenAI/Broadcom unveiling on 2026-06-24. This section is
authoritative where it conflicts with sections 1–10 above.*

### 11.1 Product identity

| Attribute | Value | Confidence |
|---|---|---|
| Official name | **Jalapeño** | confirmed — used in both the OpenAI announcement page ("OpenAI and Broadcom unveil LLM-optimized inference chip") and the joint press release ("OpenAI and Broadcom Unveil LLM-Optimized Intelligence Processor") |
| Positioning | "OpenAI's first Intelligence Processor" | confirmed (OpenAI's own framing) |
| Unveiling date | 2026-06-24 | confirmed |
| "Project Titan" | **Unsourced.** Traces only to aggregator/SEO sites (opentools.ai, tech-insider.org, xagi-labs); appears in no primary source | repo defect — corrected 2026-08-08 |

### 11.2 Silicon status — sampling, not production

> "Engineering samples of the Jalapeño chip are running ML workloads in the lab at production target
> frequency and power, including GPT-5.3-Codex-Spark."

This is a real advance over the April 2026 baseline ("not yet shipped"), but the correct status verb is
**sampling**. It is not shipping, not mass production, and not deployed at scale. TechCrunch independently
characterises the chip as still in testing with "early results".

The GPT-5.3-Codex-Spark data point is significant beyond hardware: it means the compiler, runtime and
driver path are functional enough end-to-end to execute a production-class OpenAI model on real silicon.
No detail about any of those layers was released.

### 11.3 Development cycle — vendor marketing claim

> "Jalapeño was co-developed from initial design to manufacturing tape-out in just nine months, and the
> custom AI accelerator program represents what we believe to be the fastest ASIC development cycle ever
> achieved in high-performance advanced semiconductors."

OpenAI further states it used its own models to accelerate parts of the design and optimization process.
Both are unverifiable vendor claims. Recorded as claims; not treated as findings.

### 11.4 Package and die — third-party estimate (medium confidence)

Tom's Hardware analysed the package and wafer photography released on 2026-06-24:

| Element | Estimate |
|---|---|
| Compute chiplet | ~25.46 mm × 33 mm ≈ **840 mm²**, against the ~858 mm² EUV reticle limit |
| HBM stacks | Six, surrounding the compute chiplet |
| Additional dies | One separate I/O chiplet + two structural dummy dies |
| Wafer floorplan | "Very regular, repeated, columnar" — consistent with a tiled/systolic accelerator |

This is journalist inference from photos, **not a vendor spec sheet**. It must be attributed as an
estimate wherever reused. Its architectural significance: the package is **multi-die**, not monolithic —
the only genuinely new structural fact from this disclosure.

### 11.5 Memory — generation not disclosed

Neither company stated the HBM generation on 2026-06-24 (HBM3E vs HBM4). The April 2026 baseline recorded
"Samsung HBM4 (exclusive supply agreement confirmed, Mar 2026)" as **confirmed**; that sourcing is an
aggregator (tech-insider.org), and the June 2026 disclosure neither confirms nor contradicts it.

**Correction applied 2026-08-08:** the Samsung HBM4 exclusive is downgraded from `confirmed` to
`reported-not-confirmed`, and the HBM generation itself is recorded as `not disclosed`.

### 11.6 Process node — not disclosed

The process node was **not disclosed** in the 2026-06-24 announcement. Prior reporting on the 10 GW
program consistently references a 3nm-class first deployment starting H2 2026, which is consistent with
the repo's N3 Gen 1 line but is not confirmation of it. The **2nm / TSMC A16 Gen 2** element could not be
substantiated in primary sources and remains unconfirmed roadmap reporting, not fact.

**Correction applied 2026-08-08:** Gen 1 node downgraded from stated fact to `reported-not-confirmed`;
Gen 2 A16 downgraded to `unconfirmed-reporting`.

### 11.7 Interconnect

| Attribute | Disposition after 2026-06-24 |
|---|---|
| Tomahawk | **Upgraded to confirmed.** "Broadcom's silicon implementation and networking technologies, including Tomahawk networking silicon, help bring the platform to large-scale production" |
| Jericho4 | **Still inferred.** Not mentioned anywhere in the announcement; the scale-up attribution gains no support |
| Scale-up bandwidth / topology | Not disclosed |
| Host CPU | Not disclosed. The Information reported Arm is designing a server-class CPU to anchor OpenAI's next-generation racks, possibly pairable with NVIDIA and AMD accelerators too. Unnamed sources; never confirmed by Arm, OpenAI or Broadcom; publication date not independently verified. **Rumor, low confidence** |

### 11.8 Partners and deployment

| Attribute | Value | Confidence |
|---|---|---|
| Celestica | Newly named partner — "chip implementation, board, rack system integration, high-performance networking, and scalable production systems". Absent from the April 2026 baseline | confirmed, new |
| Initial deployment | "Designed for initial deployment by the end of 2026 and expanding in the years ahead" | confirmed as a **target**, not an achieved milestone |
| Microsoft | Hock Tan (Broadcom CEO): "we are enabling the deployment of gigawatt scale data centers with Microsoft and other partners beginning in 2026". New relative to the baseline | confirmed (vendor statement) |
| Multi-generation roadmap | Hock Tan: "This is just the beginning of a multi-generation roadmap" — no nodes, generations or dates given | confirmed (no specifics) |

### 11.9 Performance — nothing quantitative

The companies disclosed **no** quantitative performance data. The complete set of claims:

- "performance per watt substantially better than current state-of-the-art"
- realized utilization "much closer to theoretical peak"
- a detailed technical report on performance promised "in the coming months"

No MLPerf submission and no third-party benchmark exists. All performance language is unquantified vendor
marketing. A **"50% cheaper than Nvidia"** figure circulating on aggregator sites (tech-insider.org,
2026-07-25) appears in no primary source and is excluded.

### 11.10 Explicitly rejected inferences

| Inference | Why rejected |
|---|---|
| "Jalapeño is potentially sellable to third parties" | Overreach. OpenAI's "designed with flexibility to work with all LLMs" / "built from the ground up for current and future LLMs across the industry" is an **architecture** statement. Neither company has said the chip will be sold externally |
| "2nm / A16 confirmed for Gen 2" | Not substantiated in any retrievable primary source |
| "Samsung HBM4 confirmed" | Aggregator-sourced; not restated at unveiling |
| "50% cheaper than Nvidia" | Aggregator noise; absent from primary sources |

### 11.11 Scheduled disclosure — Hot Chips 38

Confirmed on the live Hot Chips 38 advance program (conference 2026-08-23 to 2026-08-25):

- **OpenAI** — Richard Ho, Ravi Narayanaswami, Chris Leary: "You Can Just Build Things … Chips", AI 2
  session (chair Brucek Khailany), **Tuesday 2026-08-25, 4:45–6:15 PM PDT**
- **Broadcom** — Hemal Shah: "Thor Ultra: An Ethernet NIC Chip Optimized for AI & HPC", same day

*Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* No slides, abstract or
specification exists as of 2026-08-08 and nothing in this investigation derives from either talk. This is
the most likely venue for the first real Jalapeño microarchitecture disclosure. **Action: re-scan after
2026-08-25.**

### 11.12 Sourcing note

The OpenAI announcement page returns HTTP 403 to automated fetch and web.archive.org is unreachable from
the scanning environment. The full press-release text was recovered via the Yahoo Finance carry and
corroborated by Broadcom IR, Nasdaq, Engadget, TechCrunch, CNBC and Forbes. **Cite the Broadcom IR /
Nasdaq press release as the durable primary source**, not a Wayback snapshot.

---

## Resources

### Jalapeño unveiling (2026-06-24) and follow-on

- Broadcom IR press release (durable primary): https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-unveil-llm-optimized-intelligence-processor
- Nasdaq mirror (2026-06-24): https://www.nasdaq.com/press-release/openai-and-broadcom-unveil-llm-optimized-intelligence-processor-2026-06-24
- Yahoo Finance carry (full verbatim joint release): https://finance.yahoo.com/technology/ai/articles/openai-broadcom-unveil-llm-optimized-130000020.html
- OpenAI page (HTTP 403 to automated fetch; title/snippet only): https://openai.com/index/openai-broadcom-jalapeno-inference-chip/
- Engadget: https://www.engadget.com/2201045/openai-broadcom-jalapeno-inference-processor-ai-accelerator/
- TechCrunch: https://techcrunch.com/2026/06/24/openai-unveils-its-first-custom-chip-built-by-broadcom/
- CNBC: https://www.cnbc.com/2026/06/24/openai-and-broadcom-reveal-jalapeno-first-ai-chip-in-partnership.html
- Forbes: https://www.forbes.com/sites/jonmarkman/2026/06/25/meet-jalapeo-openais-first-custom-ai-chip-built-with-broadcom/
- Tom's Hardware — package/die photo analysis: https://www.tomshardware.com/tech-industry/artificial-intelligence/broadcom-and-openai-unveil-custom-built-jalapeno-inference-processor-openais-first-chip-is-a-massive-reticle-sized-asic-built-in-an-ultra-fast-nine-month-development-cycle
- Tom's Hardware — custom AI ASIC survey (Broadcom to MTIA): https://www.tomshardware.com/tech-industry/semiconductors/custom-ai-asics-examined-from-broadcom-to-mtia
- Hot Chips 38 advance program (scheduled talks; content not yet public): https://www.hotchips.org/advance-program/
- The Information — Arm server CPU (rumor, unconfirmed): https://www.theinformation.com/articles/openai-working-softbanks-arm-broadcom-ai-chip-effort
- Tom's Hardware carry of the Arm CPU report: https://www.tomshardware.com/pc-components/cpus/openai-arm-partner-on-custom-cpu-for-broadcom-chip

### Baseline (April 2026 and earlier)

- OpenAI official announcement (Oct 2025): https://openai.com/index/openai-and-broadcom-announce-strategic-collaboration/
- Broadcom investor announcement: https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-announce-strategic-collaboration-deploy-10
- TechCrunch (Oct 2025): https://techcrunch.com/2025/10/14/openai-and-broadcom-partner-on-ai-hardware/
- CNBC (Oct 2025): https://www.cnbc.com/2025/10/13/openai-partners-with-broadcom-custom-ai-chips-alongside-nvidia-amd.html
- Tom's Hardware — TSMC N3 / finalize chip design: https://www.tomshardware.com/tech-industry/artificial-intelligence/openai-and-broadcom-to-finalize-custom-ai-processor-in-the-coming-months-say-industry-sources
- DCD — TSMC N3 report: https://www.datacenterdynamics.com/en/news/openai-building-first-custom-ai-inference-chip-with-tsmc-and-broadcom-report/
- TrendForce — N3 Gen1 / A16 Gen2 roadmap (Jan 2026): https://www.trendforce.com/news/2026/01/15/news-openai-reportedly-to-deploy-custom-ai-chip-on-tsmc-n3-by-end-2026-second-gen-planned-for-a16/
- Samsung HBM4 deal: https://tech-insider.org/openai-titan-chip-samsung-hbm4-custom-ai-chip-2026/
- Broadcom Jericho4 AI fabric: https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-jericho4-enabling-distributed-ai-computing-across
- OpenAI Triton compiler job posting (software stack signal): https://openai.com/careers/software-engineer-triton-compiler-san-francisco/
