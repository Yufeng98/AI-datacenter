# Cerebras WSE-3 — Summary

**Device class:** Wafer-Scale Engine
**Manufacturer:** Cerebras Systems (founded 2016; S-1 filed 2024; IPO priced 2026-05-13, began trading on Nasdaq as **CBRS** 2026-05-14)
**System:** CS-3 AI supercomputer (production); **CS-4 "Nexus"** disclosed 2026-08-25 at Hot Chips 38 — early access now, GA targeted late Q3 2026 (see 2026-09-13 update below)
**Research date:** 2026-04-05 · **Last updated:** 2026-09-13
**Key sources:** https://www.cerebras.ai/chip, https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf, https://arxiv.org/html/2503.11698v1, https://github.com/Cerebras/modelzoo, https://sdk.cerebras.net/

---

## What It Is

The Cerebras WSE-3 (Wafer-Scale Engine 3) is the world's largest semiconductor device — a single 300mm TSMC 5nm silicon wafer measuring 46,225 mm² that serves as one processor. Rather than dicing a wafer into ~80 individual GPU dies, Cerebras uses the entire wafer as a unified chip. The WSE-3 integrates 900,000 AI-optimized Processing Elements (PEs) in a 2D rectangular mesh, 44 GB of distributed on-chip SRAM delivering 21 PB/s bandwidth, and 4 trillion transistors achieving 125 FP16 petaflops. The architecture deliberately avoids HBM — co-locating 48 KB of SRAM with every PE to eliminate the memory wall. For models exceeding 44 GB parameters, an external MemoryX appliance (DRAM+Flash, 4TB–2.4PB) stores weights and streams them layer-by-layer to the WSE via the SwarmX scale-out fabric. Up to 2,048 CS-3 systems can be clustered via SwarmX for pure data-parallel training.

---

## Key Specifications

| Parameter | WSE-3 | H100 SXM5 (comparison) |
|---|---|---|
| Process | TSMC 5nm | TSMC 4N |
| Die area | 46,225 mm² | ~814 mm² |
| Transistors | 4 trillion | 80 billion |
| Compute units | 900,000 PEs | 16,896 CUDA cores (132 SMs) |
| Per-PE SRAM | 48 KB | — |
| FP16 TFLOPS | 125,000 | 1,979 |
| On-chip SRAM | 44 GB | ~50 MB L2 |
| SRAM bandwidth | 21 PB/s | — |
| Off-chip memory | MemoryX (4TB–2.4PB, weights only) | 80 GB HBM3 @ 3.35 TB/s |
| System TDP | ~23 kW | ~700 W |
| Max model params | 24T per CS-3 (1.2PB MemoryX @ FP16); 120T per MemoryX unit (multi-CS cluster) | ~1–2T (in practice) |
| FP16/BF16 SIMD width | 8-wide per PE | — |
| INT8 SIMD width | 16-wide per PE | — |
| Scale-up fabric | Swarm 2D mesh (100 Pb/s on-chip) | NVLink 4 (900 GB/s) |
| Scale-out fabric | SwarmX (up to 2,048 CS-3) | NVSwitch + IB |

> **2026-09-13:** this table describes WSE-3/CS-3, still the shipping production baseline. **WSE-3T** (disclosed Hot Chips 38, 2026-08-25) is the same die at ~2× clock (2.8 vs 1.4 GHz), ships inside **CS-4 "Nexus"** racks (3 wafers/rack, direct wafer-to-wafer interconnect), and is in early access now with GA targeted late Q3 2026. See "Cerebras Update — WSE-3T / CS-4 'Nexus' (Hot Chips 38) (2026-09-13)" below for the full spec set; peak FLOPS for WSE-3T is not disclosed.

---

## Software Stack

```
User Training Script (Python) or YAML Trainer config
           │
cerebras.pytorch (cstorch) — PyTorch 2.0 Lazy Tensor Core (LTC) backend
           │  [captures full computation graph; no eager execution]
           │
CIRH Compiler (MLIR/ATen-based IR)
  • Operator fusion: FlashAttention, LayerNorm+dropout, GeLU+proj
  • Constant folding, dead code elimination
  • Memory layout optimization across 44 GB SRAM
  • AutoGen: auto-generate CSL kernels for standard ops
  • Execution mode selection: Layer Pipelined vs Weight Streaming
           │
Per-PE CSL Binaries + Execution Schedule
           │
Cerebras Runtime (SdkRuntime / Appliance API)
  • csctl CLI job scheduling, Grafana monitoring, Slurm integration
  • MemDataLoader: batch prefetch pipeline
  • Weight Streaming Scheduler: MemoryX → SwarmX → PE SRAM dispatch
           │
WSE-3 Hardware (900K PEs, 44GB SRAM, Swarm 2D mesh)
```

**Model Zoo** (open source, Apache 2.0): Llama 2/3, GPT-2/3, Mistral, Mixtral, BERT, T5, DINOv2, LLaVA.
**Low-level SDK**: CSL (Cerebras Software Language) — Zig-inspired dataflow language for custom PE kernels; `<collectives_2d>`, `<message_passing>` libraries.

---

## Two Execution Modes

### Layer Pipelined (models ≤ 44GB parameters)
- All weights and activations reside on-chip SRAM throughout training
- No off-chip memory access during compute
- Maximum compute utilization; best performance per model

### Weight Streaming (models > 44GB parameters, up to 120T)
- Weights stored in MemoryX (DRAM+Flash, 4TB–2.4PB)
- MemoryX broadcasts one layer's weights via SwarmX to all CS-3s
- CS-3 executes forward pass for that layer; activations held in SRAM
- Repeat for all layers; backward pass mirrors in reverse
- Gradient all-reduce over SwarmX; MemoryX performs weight update
- Near-linear data-parallel scaling demonstrated to 192 CS-2 systems

---

## Key Strengths

1. **125 PetaFLOPS FP16** — theoretically equivalent to ~62 H100 GPUs in one system footprint
2. **44 GB on-chip SRAM at 21 PB/s** — 7,000× H100 HBM bandwidth; activations never leave silicon
3. **No HBM bottleneck** — PE-local SRAM eliminates the memory wall for activation storage
4. **Unique defect tolerance** — 0.05mm² PE size + 1–1.5% spare PEs + dynamic routing = viable wafer-scale yield
5. **Weight streaming enables trillion-parameter training** — MemoryX+SwarmX extends single-system capacity to 24T parameters
6. **PyTorch-native LTC backend** — standard torch.nn.Module with minimal code changes; no XLA fork
7. **Fast inference** — 969+ tokens/s for Llama 3.1 405B; Cerebras Inference API deployed at AWS

---

## Limitations

1. **Cost** — CS-3 systems are priced at enterprise/hyperscale tier; not commodity hardware
2. **Power** — ~23 kW TDP per CS-3; requires significant datacenter power and water cooling
3. **Pure data parallelism only** — weight streaming does not support tensor/pipeline model parallelism across CS-3s
4. **Proprietary compiler** — CIRH compilation pipeline is closed; custom ops require CSL SDK path
5. **Not in MLPerf Training leaderboard** — Cerebras has not submitted official MLPerf training benchmark results. *Last verified 2026-04-05; not re-verified in the 2026-08-08 scan (no MLCommons results table could be retrieved for MLPerf Training v6.0 / Inference v6.0). Pending re-check.*
6. **CS-3 is a single-system design** — no NVLink-style peer-to-peer GPU tiling within one node. *Qualified 2026-08-08: Cerebras and AMD announced (2026-07-23) a disaggregated inference product in which AMD Helios rack-scale Instinct systems perform prefill and the WSE performs decode. That is a heterogeneous two-tier system, not peer-to-peer WSE tiling, and it is **announced only** — expected availability H2 2026 via Cerebras Cloud. See the 2026-08-08 update section.* **Further qualified 2026-09-13: this is no longer true for CS-4.** The CS-4 "Nexus" rack (disclosed Hot Chips 38, 2026-08-25) houses **3 WSE-3T wafers per rack with direct wafer-to-wafer links** — 2.4 Tb/s aggregate bandwidth per wafer, ~2µs latency. This is a genuine intra-rack peer-to-peer wafer interconnect, architecturally new relative to CS-1 through CS-3. See the 2026-09-13 update section.

---

## Inference Cloud

- **Cerebras Inference API**: OpenAI-compatible REST API for fast inference
- Supports: Llama 3.1/3.3, Qwen 3 (32B, 235B), GPT-OSS-120B, ZAI GLM-4.7, Llama 4
- Speed: >3,000 tokens/s; 969 tokens/s for 405B model; Llama 4 at 18× faster than GPU baselines
- Partnership: Meta Llama API uses Cerebras for fast inference backend
- AWS Marketplace: Cerebras Fast Inference Cloud available as managed service

---

## Cerebras Update — Corporate, Roadmap and Disaggregated Inference (2026-08-08)

*Update window 2026-04-01 → 2026-08-08, plus one pre-baseline item (the January 2026 OpenAI agreement) that the 2026-04-05 baseline missed. Prior-generation content above is unchanged except where explicitly corrected.*

### Headline: no new silicon generation

**There is no WSE-4 and no CS-4.** As of 2026-08-08, cerebras.ai/chip still describes only the WSE-3, and no vendor announcement of a fourth-generation wafer or a CS-4 system exists in any retrieved source. A third-party wiki describing a WSE-4 as "anticipated" is speculation, not an announcement. The WSE-3 / CS-3 baseline documented above remains current.

**One baseline correction:** the WSE-3 die area is **46,225 mm², not 46,255 mm²** (cerebras.ai/chip) — the repo previously carried a transposed digit, now fixed here, in `hw-architecture.md`, and in the research YAML. Everything else in the baseline (900,000 PEs, 4T transistors, 125 PF FP16, TSMC 5nm, 44 GB SRAM @ 21 PB/s) is unchanged and re-confirmed.

### AMD Helios + Cerebras WSE disaggregated inference — ANNOUNCED 2026-07-23

Announced at AMD's *Advancing AI 2026* event and verified against AMD's own newsroom (an independent primary source, not only the Cerebras press release):

| Aspect | Detail |
|---|---|
| Split | **AMD Helios rack-scale (Instinct) handles prefill** — prompt and large-context processing; **the Cerebras WSE handles decode / token generation** |
| Vendor performance claim | "expected to deliver up to **5× higher tokens per second per watt**" |
| Basis of that claim | **Modelling, not measurement.** AMD's footnote: "Based on modelling by AMD Performance Labs and Cerebras in July 2026 to determine tokens per second per kilowatt (TPS/kW) at a comparable interactivity point with Kimi 2.6 1T Model comparing an AMD Helios rackscale solution with Cerebras WSE to a Cerebras WSE-only configuration." |
| Baseline | **Cerebras-WSE-only**, *not* a GPU-only baseline |
| Status | **Announced only.** "Expected to become available initially through Cerebras Cloud in the second half of 2026"; Cerebras "plans to deploy AMD Helios systems in its data centers." Not sampling, not shipping, not deployed at scale. |
| Interconnect between the two tiers | **not disclosed** |

> **Two cautions on the 5× figure.** (1) The headline unit is tokens/s/W while the substantiating footnote measures tokens/s/kW — the units in AMD's own materials do not agree. (2) It is a vendor projection against Cerebras's own prior configuration, so it says nothing about WSE-vs-GPU efficiency. Treat it as a marketing claim throughout.

Architecturally this is the most significant item in the window: it partially undercuts the long-standing framing of the CS-3 as a self-contained system (Limitation #6 above) and introduces a heterogeneous prefill/decode split in which the wafer is specialised for the memory-bandwidth-bound decode phase and a GPU rack absorbs the compute-bound prefill phase. A parallel AWS-Trainium-prefill / CS-3-decode arrangement appears only in secondary trade press and is **not carried here** (low confidence).

### Rack-scale architecture disclosure — scheduled, content not yet public

**"The Cerebras Rack-Scale Architecture for Wafer Scale Engine"** — Jean-Philippe (J.P.) Fricker, Cerebras — is on the official Hot Chips 38 advance program, Session AI 1, **Tuesday 2026-08-25, 2:15–4:15 PM PDT**, Stanford Memorial Auditorium. **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** As of this update the talk is 17 days in the future; nothing about its technical content is known. Only the existence of the scheduled talk is confirmed, and it must not be cited as the source of any specification. The title implies a rack-scale system model beyond the CS-3 + MemoryX + SwarmX structure documented above; re-scan after 2026-08-25 for slides.

> **Source hygiene note.** The repo's existing source `cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai` is a **2021** post about WSE-2 / CS-2 weight streaming, MemoryX and SwarmX. It is valid support for the weight-streaming architecture but is **not** evidence of any 2026 rack-scale product.

### Corporate and capacity events in the window

| Date | Event | Status |
|---|---|---|
| 2026-01-14 | **OpenAI compute agreement** — OpenAI committed to purchase **up to 750 MW of compute over three years, running through 2028**, in a deal reported as worth **more than $10 billion** (Reuters, CNBC, TechCrunch, WSJ). | Signed commercial purchase commitment — **not deployed capacity** |
| 2026-05-13 | **IPO priced** at **$185/share**, 30 million Class A shares, **~$5.55B gross**, implying **~$56.4B fully diluted** at the offer price (Reuters). | Completed |
| 2026-05-14 | **Began trading on Nasdaq as CBRS.** Opened $350, intraday high ~$385, closed **$311.07 (+68%)** → ~**$106.75B** fully diluted. Reported as the largest US tech IPO since Uber (2019). | Completed |
| 2026-06-23 | **Q1 FY2026 results:** GAAP revenue **$193.4M**; record core revenue **$191.3M, up 92% YoY**. | Reported |
| 2026-07-09 | **Flex manufacturing expansion** — new Milpitas, California lines supporting an anticipated **~7× increase in CS-3 production capacity through 2026**. | Announced capacity expansion **for CS-3** — further evidence no new generation is imminent |
| 2026-07-09 | **European capacity** — CEO Andrew Feldman at the RAISE Summit (Paris): target of **200 MW of AI compute in Europe by end of 2027**, first European capacity online by end of 2026, sited in France and the Nordics. | Stated target/plan, **not built capacity** |
| 2026-07-23 | AMD Helios + WSE disaggregated inference (above). | Announced, expected H2 2026 |
| 2026-08-25 | Hot Chips 38 rack-scale talk (above). | Scheduled — content not yet public |

**Figures deliberately excluded.** A "$9.5 billion IPO" figure circulating in a headline slug is wrong by roughly 6× against the $56.4B offer-price valuation and does not match the $5.55B raised — do not use it. A "$25B OpenAI backlog" figure from one aggregator around the Q1 FY2026 report could not be confirmed against the primary release and conflicts in magnitude with the January $10B figure — excluded.

**Not re-verified in this pass** (do not treat as confirmed): the minor product/blog items Multi-LoRA (2026-05-06), trillion-parameter Kimi K2.6 inference (2026-05-19), Gemma 4 multimodal inference (2026-06-29), the Upstage/South Korea partnership (2026-07-10), the sovereign-AI post (2026-05-26), and a G42 India–UAE piece (2026-06-01).

---

## Cerebras Update — WSE-3T / CS-4 "Nexus" (Hot Chips 38) (2026-09-13)

*Update window 2026-08-08 → 2026-09-13. Primary source: Cerebras's own Hot Chips 38 deep-dive blog, https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026 (2026-08-25), fetched directly. Secondary sources for detail the primary post omits: ServeTheHome (https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/, 2026-08-25) and The Next Platform (https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400, 2026-08-19). This resolves the disclosure the 2026-08-08 pass had flagged as scheduled but not yet public.*

### Headline: a new SKU, not a new die

Cerebras disclosed **WSE-3T** and the **CS-4 "Nexus"** rack at Hot Chips 38. WSE-3T is the **same physical die as WSE-3** — same TSMC 5nm process, same 900,000 cores, same 44 GB on-wafer SRAM, same 46,225 mm² die area — run at roughly **2× the clock** (1.4 GHz → 2.8 GHz, per The Next Platform). The 2026-08-08 finding "no WSE-4, no CS-4" for a *new die design* still holds; what's new is a higher-clocked SKU and a new rack system, not a fourth-generation wafer.

### What's new and sourced

- **On-wafer bandwidth (new figures):** WSE-3T's on-chip SRAM bandwidth is **43,000 TB/s** (~2× WSE-3's already-recorded 21 PB/s), and Cerebras's own blog states **53.5 PB/s of aggregate on-wafer fabric bandwidth** — "more than 200 times the NVL72 rack's scale-up bandwidth" (260 TB/s).
- **Wafer-to-wafer interconnect (new capability):** CS-4 "Nexus" racks hold **3 WSE-3T wafers**, each in its own "compute backpack," connected by **direct wafer-to-wafer links**: **2.4 Tb/s aggregate bandwidth per wafer**, latency as low as **~2 microseconds**. This is a materially new scale-up capability — see the updated Limitation #6 above.
- **Rack architecture:** power delivery via AC/DC converters placed **0.5 mm from the wafer** (vs. ~50 mm on a GPU rack), up to 30 modules/backpack at up to 277 VAC → 54.5 VDC; per-backpack water conditioning; "three times the compute per rack" vs. CS-1–CS-3, "50% fewer components," deployable "up to 3× faster."
- **Performance claims (vendor, unverified):** "2x more tokens, at 10x more tokens per watt" vs. CS-3; "30x faster than a GPU" (baseline unspecified in retrieved coverage).
- **Availability:** early access to select customers **now**; GA targeted **later in Q3 2026** (Next Platform, 2026-08-19).
- **Roadmap — CS-5 (targeted 2027):** up to 10,000 output tokens/s/user (mid-size/open models); up to 5,000 tokens/s/user and 3M tokens/s/MW for frontier models; support for >50T-parameter models.
- **Roadmap — CS-6 (no date):** wafer-scale SRAM and compute integrated with **3D-stacked DRAM**, targeting an order-of-magnitude smaller inference footprint.

### Explicitly checked and not confirmed

- **Peak FLOPS for WSE-3T** — not stated in either secondary source retrieved. (A naive 2× extrapolation from WSE-3's 125 PF FP16 would suggest ~250 PF, but this is the survey's own arithmetic, not a vendor figure, and is not recorded as a spec.)
- **"6× 200GbE per system"** — this figure, which circulated in pre-scan notes, traces to The Next Platform author's own explicit speculation ("I *think* the wafer I/O module has six Ethernet ports... we have tried to confirm... with Cerebras but have not heard back"). **Not recorded as confirmed.**
- CS-4 pricing; WSE-3T transistor count (presumed unchanged, not restated); exact wafer-to-wafer link protocol.
- No new corporate/capacity events beyond what the 2026-08-08 update already recorded were found in this pass; the IPO figures already on record ($5.55B raised, 2026-05-13/14, Nasdaq: CBRS) were spot-checked against an independent source (Wikipedia) with no discrepancy.

---

## Programming Model Rationale

The Cerebras software stack is shaped almost entirely by two hardware constraints: (1) the on-chip SRAM limit of 44 GB, and (2) the dataflow execution model of the 2D PE mesh.

**Why PyTorch + LTC instead of a custom DSL?** The 900K PEs look like a spatial dataflow machine, but Cerebras chose to meet researchers at PyTorch rather than require them to learn a spatial programming model. The Lazy Tensor Core backend solves the mismatch by capturing the full computation graph lazily (no eager execution) and handing it to the CIRH compiler, which performs the spatial mapping. Users write standard `torch.nn.Module` code; the compiler handles the wafer-specific layout.

**Why CIRH instead of XLA/HLO?** XLA's HLO representation is optimized for TPU's systolic array execution model — it has strong assumptions about tensor shapes and tiling that don't translate well to Cerebras's irregular PE layout. CIRH is MLIR-based and stays close to ATen semantics, giving the compiler more flexibility to perform WSE-specific optimizations (PE assignment, SRAM partitioning, color routing table generation) without fighting a foreign IR.

**Why two execution modes?** The 44 GB SRAM constraint is hard. Models like GPT-3 (175B params, ~350 GB in FP16) cannot fit on-chip. Weight Streaming was the architectural answer: decouple compute (WSE) from storage (MemoryX). Layer Pipelined mode is the "ideal" path — everything on-chip, maximum throughput. Weight Streaming trades throughput for capacity, enabling 24T-parameter models at the cost of synchronizing each layer's weights through MemoryX and SwarmX.

**Why CSL instead of CUDA/PTX?** CUDA's thread/warp/block model is fundamentally mismatched to the WSE PE architecture. Each PE has its own PC, 48KB SRAM, and a mesh router — there are no warps, no shared memory in the GPU sense, no SMs. CSL's dataflow model (tasks activated by wavelet arrivals on colors) maps directly to how PE routers work: a wavelet arrives, the hardware fires the associated task, the task reads from DSD-addressed SRAM, and sends output wavelets to neighbors. The programming model is the hardware model.

**Why pure data parallelism at scale?** The MemoryX+SwarmX architecture stores one copy of weights centrally and broadcasts to all CS-3s. This means tensor parallelism (splitting one model's layers across CS-3s) is not supported in weight streaming mode — there is no direct CS-3-to-CS-3 weight communication, only MemoryX-to-CS-3. This is a deliberate simplification: near-linear data-parallel scaling is harder to achieve with tensor parallelism, and for training, data parallelism provides linear throughput scaling with minimal code complexity.

**Why AutoGen and closed compiler?** Writing CSL for 900K heterogeneous PEs with a 2D mesh routing constraint is not something researchers should do manually. AutoGen generates CSL kernels from CIRH op patterns automatically, making the platform usable without CSL expertise. The compiler remains closed because the PE layout and routing optimization algorithms are core IP. The tradeoff: no custom kernel authoring at training scale (requires the SDK path), but zero barrier for standard LLM training.

---

## Sources

- https://www.cerebras.ai/chip
- https://www.cerebras.ai/system
- https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf
- https://arxiv.org/html/2503.11698v1
- https://ieeexplore.ieee.org/document/10123162/
- https://dl.acm.org/doi/abs/10.1109/MM.2024.3386628
- https://github.com/Cerebras/modelzoo
- https://sdk.cerebras.net/
- https://training-api.cerebras.ai/
- https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning
- https://www.cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem
- https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai *(2021 — WSE-2/CS-2 weight streaming; not evidence for any 2026 product)*

### Added 2026-09-13

- https://www.cerebras.ai/blog/ultrafast-frontier-inference-cerebras-deep-dive-at-hot-chips-2026 — primary source, Hot Chips 38 (2026-08-25): 53.5 PB/s aggregate on-wafer fabric bandwidth (WSE-3T), CS-5/CS-6 roadmap
- https://www.servethehome.com/cerebras-talks-going-rack-scale-with-their-wses-at-hot-chips-2026/ — 43,000 TB/s SRAM bandwidth, 2.4 Tb/s / 2µs wafer-to-wafer interconnect, CS-4 Nexus rack detail, 2x tokens/10x tokens-per-W vs CS-3
- https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400 — WSE-3T clock speed (1.4→2.8 GHz), unchanged die/core/SRAM, early access + late-Q3-2026 GA target

### Added 2026-08-08

- https://newsroom.amd.com/news/aai-2026-cerebras-inference/ — AMD primary source for the Helios-prefill / WSE-decode announcement and the 5× TPS/kW modelling footnote (2026-07-23)
- https://www.cerebras.ai/press-release/amd-and-cerebras-announce-industry-leading-ultra-low-latency-and-high-throughput-ai-inference — Cerebras side of the same announcement
- https://hotchips.org/advance-program/ — HC38 advance program (rack-scale talk, 2026-08-25; not yet presented)
- https://www.reuters.com/legal/government/cerebras-prices-ipo-185-per-share-raise-555-billion-sources-say-2026-05-13/ — IPO pricing, $185/share, $5.55B, ~$56.4B fully diluted
- https://www.reuters.com/legal/transactional/cerebras-set-debut-stock-market-gripped-by-ai-mania-2026-05-14/ — day-one trading, close $311.07, ~$106.75B
- https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html — Nasdaq listing, ticker CBRS
- https://www.nasdaqprivatemarket.com/company/cerebras/ — exchange/ticker corroboration
- https://www.reuters.com/technology/openai-buy-compute-capacity-startup-cerebras-around-10-billion-wsj-reports-2026-01-14/ — OpenAI ~$10B / 750 MW commitment through 2028
- https://www.cnbc.com/2026/01/14/cerebras-scores-openai-deal-worth-over-10-billion.html
- https://investors.flex.com/news/news-details/2026/Flex-and-Cerebras-Expand-Partnership-to-Scale-American-Manufacturing-of-Cerebras-AI-Supercomputers/default.aspx — ~7× CS-3 capacity, Milpitas, 2026-07-09
- https://investors.cerebras.ai/news-releases/news-release-details/cerebras-systems-accelerates-european-expansion-200mw-ai-compute — 200 MW Europe by end-2027 target
- https://investors.cerebras.ai/news-releases/news-release-details/cerebras-systems-announces-strong-first-quarter-2026-results — Q1 FY2026, $193.4M GAAP revenue
