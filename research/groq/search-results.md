# Groq LPU (TSP) — Search Results

**Device class:** Deterministic TSP (Tensor Streaming Processor / Language Processing Unit)
**Research date:** 2026-04-05 · **Last updated:** 2026-08-08

> ⚠️ **CORRECTION (2026-08-08).** The note that previously stood here — "Groq was acquired by NVIDIA in 2026 for ~$20 billion. The architecture and IP are now part of NVIDIA" — is **wrong**. The 2025-12-24 transaction was a **non-exclusive inference-technology licensing agreement** plus an acqui-hire. Groq remains an **independent company**, GroqCloud continues to operate, and Groq raised **$650M on 2026-06-22**. The ~$20B figure is CNBC reporting, **not confirmed by either company**. See "Layer 16" at the end of this file.

---

## Layer 1 — Device Overview

Groq's LPU (Language Processing Unit), originally named TSP (Tensor Streaming Processor), is a deterministic inference accelerator. Its defining features:
- **All-SRAM weight storage** (no DRAM on-chip): ~230 MB SRAM per die
- **Compiler-scheduled execution**: no branch prediction, no cache, no out-of-order execution
- **Plesiosynchronous chip-to-chip**: 576 chips in GroqRack act as a single coherent memory

Key references:
- https://groq.com/lpu-architecture
- http://pkamath.com/publications/papers/tsp-isca20.pdf (ISCA 2020)
- https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed

---

## Layer 2 — Chip Specifications

| Parameter | LPU v1 | LPU v2 |
|---|---|---|
| Process | Samsung 14 nm | Samsung 4 nm |
| Die size | 25×29 mm | — |
| Clock | 900 MHz | — |
| INT8 TOPS | 750 | improved |
| FP16 TFLOPS | 188 | improved |
| On-chip SRAM | ~220–230 MB | — |
| SRAM bandwidth | 80 TB/s (internal) | — |
| MAC array | 320×320 fused dot product | — |
| Vector ALUs | 5,120 | — |
| DRAM | None (SRAM-only) | — |

---

## Layer 3 — TSP Microarchitecture

- **Compute:** 320×320 fused dot product matrix multiplication unit + 5,120 Vector ALUs
- **Memory:** All weight storage is on-chip SRAM (~230 MB), not DRAM
- **Execution:** Purely compiler-scheduled — zero reactive hardware (no branch predictor, no cache, no out-of-order buffer)
- **Data flow:** Tensor streaming — activations flow through a conveyor-belt style pipeline
- **Spatial layout:** Functional units arranged in a 2D spatial grid; compiler maps tensor flows to physical routes

---

## Layer 4 — Memory Model

- No off-chip DRAM on a single chip
- 220–230 MB on-chip SRAM per die
- 80 TB/s internal SRAM bandwidth
- For large models: model weights are distributed across multiple chips (each chip holds a shard)
- 576-chip GroqRack = 576 × 230 MB ≈ 130 GB total SRAM memory space

---

## Layer 5 — Chip-to-Chip Interconnect

- **Protocol:** Plesiosynchronous (near-synchronous with known drift)
- **Topology:** Direct mesh / Dragonfly variant — no external switches
- **Compiler awareness:** Compiler schedules packet delivery to exact clock cycles across chips
- GroqRack: 576 chips, up to 640 TB/s aggregate chip-to-chip bandwidth
- Model weights striped across chips; inference pipeline spans all chips concurrently

---

## Layer 6 — Software Stack

```
PyTorch / TensorFlow / ONNX / CoreML / Keras
         ↓
GroqFlow (automated compiler workflow, open source)
         ↓
torch-MLIR or ONNX → GTen ops (Groq dialect)
         ↓
Groq Compiler (static spatial scheduler)
  - Maps tensor operations to chip spatial units
  - Schedules to cycle-exact timing
  - Multi-chip routing pre-computed
         ↓
Groq DevTools package (groq-devtools)
         ↓
Groq Runtime (groq-runtime)
         ↓
LPU hardware
```

---

## Layer 7 — Compiler

- Input: torch-MLIR, ONNX, GTen dialect
- **Static compiler scheduling**: assigns every operation to specific spatial unit and clock cycle
- Inter-chip data movement pre-computed by compiler
- No dynamic runtime scheduling
- GitHub: https://github.com/groq/groqflow (open source)
- Also supports assembler for bare-metal programming

---

## Layer 8 — Workload Fit

**Best for:**
- LLM inference (autoregressive generation) — high token/second, ultra-low latency
- Workloads that benefit from deterministic, zero-jitter execution
- Models that fit within GroqRack SRAM budget

**Weaknesses:**
- Training (SRAM-only; no large activation memory)
- Very large models requiring >130 GB (576-chip rack limit)
- Dynamic/irregular computation shapes (compiler must pre-schedule)

---

## Layer 9 — Precision

- INT8
- FP16
- BF16 (LPU v2)

---

## Layer 10 — Deployment

- GroqRack: 576 LPUs, ~130 GB effective memory
- Groq Cloud: public API inference service (groq.com) — 13 data centers, >5M developers, ~200 MW target by end of 2027 (2026-06-22)
- Following the Dec 2025 non-exclusive licensing agreement (*not* an acquisition): **NVIDIA** Groq 3 LPX for the NVIDIA Vera Rubin platform; Groq itself plans to deploy LPX systems in GroqCloud

---

## Layer 11 — Open Source

- GroqFlow: open source (GitHub: groq/groqflow)
- mlagility: model benchmark tools (open source)
- Compiler backend: partially open (MLIR dialect)
- Chip design: closed

---

## Layer 12 — LLM Performance

- ~500 tokens/sec on Llama 3 70B (per public benchmarks)
- Groq Cloud consistently ranks fastest public inference API
- 10× lower latency vs GPU HBM (SRAM BW vs HBM BW argument)

---

## Layer 13 — Competing Chips

- NVIDIA H100/H200 (much higher throughput, higher latency)
- Cerebras CS-3 (wafer-scale SRAM approach, different scale)
- Tesla FSD (edge only)

---

## Layer 14 — Roadmap / Status

- LPU v2: Samsung 4 nm (2025)
- **No new Groq-designed silicon announced Apr–Aug 2026** (no LPU v3 / next-gen TSP)
- NVIDIA Groq 3 LPX (LP30) for Vera Rubin — **ANNOUNCED** GTC 2026-03-16; H2 2026 partner availability guidance; process node and TDP **not disclosed**
- GroqCloud continues operation; Groq remains independent and raised $650M (2026-06-22)
- Hot Chips 38 (2026-08-25), Session AI 1: "Think Fast: LPU Accelerator for Heterogeneous Compute" (NVIDIA) — *disclosure scheduled, content not yet public*

---

## Layer 15 — Key Publications

- ISCA 2020 paper: http://pkamath.com/publications/papers/tsp-isca20.pdf
- Hot Chips 2022: https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2022/11/ATPESC-2022-Track-1-Talk-4-Ling-Groq.pdf
- Hot Chips 34: https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf
- Zellic deep dive: https://www.zellic.io/blog/groq-tsp-whitepapers/

---

## Layer 16 — Resources added 2026-08-08

### Groq primary (corporate status, leadership, GroqCloud)

| Resource | Date | What it establishes |
|---|---|---|
| [Groq newsroom index](https://groq.com/newsroom) | — | Exactly three items Dec 2025 – Aug 2026: DOE partnership (2025-12-18), NVIDIA licensing agreement (2025-12-24), $650M raise (2026-06-22) |
| [Groq & NVIDIA non-exclusive inference technology licensing agreement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) | 2025-12-24 | **Primary.** "non-exclusive licensing agreement"; Groq "will continue to operate as an independent company"; "GroqCloud will continue to operate without interruption"; Ross & Madra to NVIDIA; **no dollar amount stated** |
| [Groq raises $650M to scale its AI inference cloud business](https://groq.com/newsroom/groq-raises-usd650m-to-scale-its-ai-inference-cloud-business) | 2026-06-22 | **Primary.** $650M led by Disruptive and Infinitum; 13 data centers; >5M developers; ~200 MW by end of 2027; "the new LPX system from NVIDIA" |
| [Groq About Us](https://groq.com/about-us) | — | **Primary.** Adam Winter CEO ("became CEO in 2026"); Matt Eng CFO, Alan Rice COO, Sinclair Schuller CTO, Rakesh Malhotra CPO |
| [Groq blog index](https://groq.com/blog) | — | **Primary.** Only two 2026 posts: "GroqCloud: Expanding to Meet Demand" (2026-02-16) and the 2026-06-22 fundraise — **no new Groq silicon** |

### NVIDIA primary (Groq 3 LPX / LP30)

| Resource | Date | What it establishes |
|---|---|---|
| [Inside NVIDIA Groq 3 LPX (developer blog)](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) | 2026-03-16 | **Primary.** 500 MB SRAM/LPU, 150 TB/s, 96 C2C links @112 Gbps = 2.5 TB/s, 256 LPUs / 32 trays of 8, 128 GB rack SRAM, 40 PB/s, 640 TB/s scale-up, 315 PFLOPS, "up to 35x higher inference throughput per megawatt"; LP30 codename. **No process node, no TDP, no ship date.** Corrects the prefill/decode split |
| [NVIDIA LPX product page](https://www.nvidia.com/en-us/data-center/lpx/) | — | **Primary.** 500 MB SRAM/LPU, 128 GB & 40 PB/s per rack, 640 TB/s scale-up, 256 LPUs, **12 TB DDR5 per rack**; no node, no power, no ship date |
| [NVIDIA Vera Rubin platform release](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform) | — | **Primary.** Vera Rubin-based products "will be available from partners starting the second half of this year" |

### Conference

| Resource | What it establishes |
|---|---|
| [hotchips.org](https://hotchips.org/) | Hot Chips 38: Aug 23–25 2026, Memorial Auditorium, Stanford. **Not yet held as of 2026-08-08** |
| [Hot Chips 38 advance program](https://hotchips.org/advance-program/) | Session AI 1, Tue 2026-08-25 2:15–4:15 PM PDT: "Think Fast: LPU Accelerator for Heterogeneous Compute" — Igor Arsovski & Santosh Raghavan, NVIDIA. *Disclosure scheduled — content not yet public. Do not cite as the source of any spec.* |

### Secondary / corroborating

| Resource | Note |
|---|---|
| [StorageReview — NVIDIA Groq 3 LPX: Everything We Know](https://www.storagereview.com/news/nvidia-groq-3-lpx-everything-we-know) | 1.2 PFLOPS FP8/chip, 9.6 PFLOPS FP8/tray, 500 MB @150 TB/s, "available in the second half of 2026"; explicitly states node and TDP are unknown |
| [CRN — NVIDIA puts Groq LPU, Vera CPU and BlueField-4 DPU into new data center racks](https://www.crn.com/news/components-peripherals/2026/nvidia-puts-groq-lpu-vera-cpu-and-bluefield-4-dpu-into-new-data-center-racks) | LPX announced at GTC 2026 alongside Vera Rubin NVL72 |
| [Reuters — Nvidia to license Groq technology, hire executives](https://www.reuters.com/business/nvidia-buy-ai-chip-startup-groq-about-20-billion-cnbc-reports-2025-12-24/) | 2025-12-24. "Groq did not disclose financial details of the deal." **Body returned 403 during the scan — snippet only; re-verify.** |
| [CNBC — Nvidia buying Groq for about $20 billion](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html) | **Originating source of the ~$20B figure.** Body returned 403 — snippet only; re-verify |
| [CNBC — Nvidia-Groq deal is structured to keep fiction of competition alive](https://www.cnbc.com/2025/12/26/nvidia-groq-deal-is-structured-to-keep-fiction-of-competition-alive.html) | Analyst commentary describing the deal as non-exclusive licensing |
| [Bloomberg — Groq raises $650 million](https://www.bloomberg.com/news/articles/2026-06-22/groq-raises-650-million-to-help-startup-pivot-after-nvidia-deal) | Independent corroboration of the 2026-06-22 raise |
| [DataCenterDynamics — Groq secures $650M](https://www.datacenterdynamics.com/en/news/groq-secures-650m-in-new-growth-capital-for-ai-cloud-expansion/) | Independent corroboration; leads named |
| [FinSMEs — Groq raises $650M](https://finsmes.com/2026/06/groq-raises-650m-in-growth-funding.html) | Names Adam Winter as newly appointed CEO and Alex Davis as Chairman |
| [Bloom Energy — Simon Edwards bio](https://www.bloomenergy.com/team/simon-edwards/) | Confirms Edwards "later served as CEO [of Groq] following a strategic licensing transaction with NVIDIA" and has since departed |
| [Wikipedia — Groq](https://en.wikipedia.org/wiki/Groq) | Characterizes the deal as non-exclusive licensing, not acquisition. **Infobox still lists Jonathan Ross as CEO — stale, do not use** |

**Scan tooling caveat (2026-08-08):** WebSearch quota was exhausted during the verification pass; independent sourcing used direct WebFetch and DuckDuckGo-lite result pages. reuters.com and cnbc.com article bodies were blocked (403). The ~$20B attribution chain should be re-verified against the CNBC original when full search access is restored.
