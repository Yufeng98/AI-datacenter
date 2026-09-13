# Groq LPU — HW Architecture Investigation

## Summary

Groq's LPU (TSP) is the most radically different architecture in the AI accelerator space. It replaces all dynamic hardware (caches, branch predictors, out-of-order execution) with compiler-scheduled static execution and all-SRAM weight storage. The result is extremely high, predictable memory bandwidth (80 TB/s) and deterministic latency.

---

## Compute Units

### Matrix Multiplication Unit
- 320×320 fused dot product array
- Handles GEMM (the dominant LLM operation)

### Vector ALUs
- 5,120 vector ALUs
- Handles activation functions, layer norm, element-wise ops

### No GPU-style SM/warp scheduler
- No out-of-order execution
- No speculative execution
- No cache hierarchy (SRAM is primary, not cache)

---

## Memory Architecture

### On-Chip SRAM
- ~220–230 MB per die
- Acts as weight storage AND activation buffer
- Internal bandwidth: 80 TB/s
- Compare to H100 HBM3: ~3.35 TB/s — Groq SRAM is ~24× faster

### No Off-Chip DRAM
- A single LPU die has no DRAM
- All data must fit in 230 MB SRAM
- For large models: weights striped across chips in GroqRack

---

## Execution Model

**Software-defined, compiler-scheduled:**
1. Compiler receives model (PyTorch/ONNX)
2. Compiler generates a cycle-exact schedule for every op
3. Schedule includes: which spatial unit, which cycle, which SRAM address
4. At runtime: LPU executes the schedule deterministically
5. Zero branching, zero cache misses, zero scheduling overhead

**Result:** Every inference run takes exactly the same time — zero jitter.

---

## Spatial Architecture

- Functional units arranged in 2D spatial grid
- Compiler maps tensor flows to physical routes across the grid
- "Conveyor belt" data streaming: data flows through units rather than being loaded/stored

---

## Chip Physical

| Parameter | Value |
|---|---|
| Process | Samsung 14 nm (v1) |
| Die size | 25×29 mm (~725 mm²) |
| Clock | 900 MHz |
| INT8 TOPS | 750 |
| FP16 TFLOPS | 188 |

---

## Chip-to-Chip (GroqRack)

- Protocol: Plesiosynchronous (near-synchronous, known drift)
- No external switches
- Compiler schedules inter-chip packets to exact clock cycles
- 576 chips in GroqRack: ~130 GB SRAM, ~640 TB/s chip-to-chip bandwidth
- 576 chips appear as single coherent memory to application

---

## Key Design Decisions

1. **All-SRAM**: eliminates DRAM bandwidth bottleneck entirely; 80 TB/s vs 3.35 TB/s HBM3
2. **Compiler-scheduled**: deterministic latency, zero jitter
3. **No cache/predictor**: frees transistor area for more compute and SRAM
4. **Plesiosynchronous interconnect**: scales to hundreds of chips without clock distribution hell
5. **Spatial dataflow**: matches streaming nature of transformer token generation

---

# Investigation Update — 2026-08-08

**Scope:** corrections to the NVIDIA Groq 3 LPX (LP30) entry and to the corporate framing of the Groq programme; no new *Groq*-designed silicon exists in the window.

**Method:** primary-source retrieval (groq.com newsroom/about-us, NVIDIA developer blog, nvidia.com LPX product page, nvidianews Vera Rubin release, hotchips.org advance program) plus secondary corroboration. Note a tooling limitation in the underlying scan: reuters.com and cnbc.com article bodies returned 403, so their content is cited from search-result snippets and Wikipedia's reference list, not full articles. The ~$20B attribution chain should be re-verified against the CNBC original when search access permits.

## 1. Corporate framing — the repo's prior claim was wrong

The December 2025 NVIDIA–Groq transaction was **not an acquisition**. Groq's own newsroom post (2025-12-24) states it "has entered into a **non-exclusive licensing agreement** with Nvidia for Groq's inference technology," that Groq "will continue to operate as an independent company," and that "GroqCloud will continue to operate without interruption." Founder/CEO Jonathan Ross, president Sunny Madra and other employees left to join NVIDIA "to help advance and scale the licensed technology." Structurally: a non-exclusive IP license plus a large acqui-hire.

The **~$20 billion** figure originated with CNBC on 2025-12-24 and was carried by Reuters, TechCrunch, EE Times and IBD. It is **not confirmed by NVIDIA or Groq**; Reuters notes "Groq did not disclose financial details of the deal." No SEC filing disclosing a value was found. This is well-sourced journalism, not a vendor figure — the survey must label it as such.

Leadership as of 2026-08-08: **Adam Winter, CEO** (groq.com/about-us: "became CEO in 2026"; some outlets say interim), Matt Eng CFO, Alan Rice COO, Sinclair Schuller CTO, Rakesh Malhotra CPO; Alex Davis named Chairman in coverage of the June 2026 raise. Simon Edwards, who became CEO at the time of the December 2025 transaction, **left in April 2026** for Bloom Energy (confirmed by Bloom Energy's own bio). Wikipedia's Groq infobox still lists Jonathan Ross and is stale.

## 2. Groq corporate trajectory — $650M raise, pivot to inference-cloud operator (2026-06-22)

$650M growth capital led by **Disruptive** and **Infinitum** with participation from existing investors; **valuation not disclosed** (prior round reportedly ~$6.9B). Stated footprint: **13 data centers** across North America, Europe, the Middle East and APAC; **"more than five million developers"**; expects to **scale toward 200 MW by end of 2027**. Corroborated by Bloomberg and DataCenterDynamics (both 2026-06-22).

Architecturally significant: Groq's own release says it will fit out that footprint "with Groq's latest inference technology, **including the new LPX system from NVIDIA**." Groq is therefore now a *customer* for NVIDIA LPX silicon derived from its own licensed IP, not solely the originator. For a hardware survey this changes the attribution model: LP30 is NVIDIA silicon; GroqCloud is a Groq service that will run on it.

Adjacent: Groq announced a **U.S. DOE partnership on 2025-12-18**, six days before the NVIDIA agreement.

## 3. NVIDIA Groq 3 LPX (LP30) — corrected and expanded specifications

Announced at **GTC 2026 on 2026-03-16**. Confirmed by NVIDIA's developer blog and the nvidia.com/en-us/data-center/lpx product page:

| Parameter | Value | Change vs. prior repo state |
|---|---|---|
| On-chip SRAM per chip | **500 MB** | corrected from 512 MB |
| SRAM bandwidth per chip | 150 TB/s | unchanged |
| Chip-to-chip links | **96 @ 112 Gbps** | new |
| Aggregate C2C BW per chip | **2.5 TB/s bidirectional** | new |
| FP8 dense compute | **1.2 PFLOPS/chip; 9.6 PFLOPS/tray** | new |
| Tray | **1U liquid-cooled, 8 LPUs** | new |
| Rack | **256 LPUs = 32 trays** | chip count confirmed |
| Rack SRAM | **128 GB @ 40 PB/s** | 40 PB/s already present |
| Rack scale-up BW | **640 TB/s** | new |
| Rack DDR5 | **12 TB** | new |
| Efficiency claim | "up to 35x higher inference throughput per megawatt" | reclassified as **vendor marketing, unaudited** |
| Process node | **Not disclosed** | prior "Samsung 4nm" is unconfirmed (secondary blogs only) |
| TDP | **Not disclosed** | new explicit gap |
| Non-FP8 datatype throughput | **Not disclosed** | new explicit gap |
| Availability | **ANNOUNCED**; H2 2026 Vera Rubin partner availability guidance; no LPX-specific ship date | downgraded from "ships Q3 2026" |

**Arithmetic anomaly in NVIDIA's own material:** the developer blog quotes 315 PFLOPS AI inference compute per LPX rack, but 9.6 PFLOPS FP8/tray × 32 trays = 307.2 PFLOPS. Unexplained; do not treat 315 PFLOPS as a dense FP8 number without the caveat.

**Corrected prefill/decode split.** The repo previously said "Rubin GPUs handle prefill, Groq LPUs handle token generation." NVIDIA's blog describes the opposite split *within decode*: GPUs take throughput-bound work such as full-context attention over the accumulated KV cache; LPX accelerates latency-sensitive execution within decode such as sparse MoE expert feed-forward networks.

**"No DRAM" no longer a system-wide absolute.** The LP30 die still has no attached DRAM, but the LPX rack carries **12 TB of DDR5**. The DDR5 tier's bandwidth, attachment point, and the compiler/runtime policy governing movement between it and LPU SRAM are all **not disclosed**.

## 4. Flagged unverified: GroqRack "~640 TB/s aggregate"

No Groq primary source for a 640 TB/s GroqRack chip-to-chip figure could be retrieved. 640 TB/s is *exactly* NVIDIA's published rack-scale scale-up figure for the 256-LPU LPX rack, so the repo's legacy attribution is probably a conflation. Marked **unverified** in `chips/groq/hw-architecture.md`, `summary.md` and `layer-table.md` pending a Groq source.

## 5. No new Groq silicon, April–August 2026

Groq published exactly two items in 2026: "GroqCloud: Expanding to Meet Demand" (2026-02-16) and the 2026-06-22 fundraise. No LPU v3 / next-generation TSP was announced.

## 6. Scheduled disclosure — Hot Chips 38 (not yet presented)

Hot Chips 38, Aug 23–25 2026, Memorial Auditorium, Stanford. **Session AI 1, Tuesday 2026-08-25, 2:15–4:15 PM PDT: "Think Fast: LPU Accelerator for Heterogeneous Compute" — Igor Arsovski & Santosh Raghavan, NVIDIA.** First Hot Chips LPU disclosure under NVIDIA branding. *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public* as of 2026-08-08; no slides, abstract or specs exist. Do not cite the talk title as the source of any spec. Re-scan after 2026-08-25 for process node, TDP and per-datatype throughput.

## Sources (2026-08-08 update)

- https://groq.com/newsroom
- https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale
- https://groq.com/newsroom/groq-raises-usd650m-to-scale-its-ai-inference-cloud-business
- https://groq.com/about-us
- https://groq.com/blog
- https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/
- https://www.nvidia.com/en-us/data-center/lpx/
- https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform
- https://www.bloomenergy.com/team/simon-edwards/
- https://hotchips.org/advance-program/
- https://www.storagereview.com/news/nvidia-groq-3-lpx-everything-we-know
- https://www.crn.com/news/components-peripherals/2026/nvidia-puts-groq-lpu-vera-cpu-and-bluefield-4-dpu-into-new-data-center-racks
- https://www.reuters.com/business/nvidia-buy-ai-chip-startup-groq-about-20-billion-cnbc-reports-2025-12-24/ (snippet only; body 403)
- https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html (snippet only; body 403)
- https://www.bloomberg.com/news/articles/2026-06-22/groq-raises-650-million-to-help-startup-pivot-after-nvidia-deal
- https://www.datacenterdynamics.com/en/news/groq-secures-650m-in-new-growth-capital-for-ai-cloud-expansion/
- https://en.wikipedia.org/wiki/Groq (secondary; infobox CEO field is stale)
