# Etched Sohu — Summary

*as_of: 2026-08-08*
*Device class: Transformer-Specific ASIC*
*Status: A0 silicon returned (H1 2026); first rack-scale product in customer validation; production started; NOT shipping and NOT deployed at scale as of 2026-08-08*

> **Naming note (2026-08-08).** The repo slug remains `etched-sohu`, but the string "Sohu" no longer appears
> anywhere on etched.com (homepage, `/progress`, `/progress/frontier-inference-clusters`,
> `/progress/accelerating-inference`, checked 2026-08-08). Etched now markets only "frontier inference
> clusters"/"frontier inference systems". **No vendor or press source states that Sohu was retired, renamed, or
> that the transformer-specialized approach was abandoned** — this is confirmed *de-emphasis of the product name
> in marketing*, not a confirmed cancellation or architectural pivot. The 2024-era Sohu content below is
> preserved as the historical baseline; where it describes silicon behaviour it should be read as
> **unverified for the current (A0) part**. See "2026 Update" below.

---

## One-Line Summary

The Etched Sohu is the world's first transformer-specific ASIC: a chip that hardcodes the entire transformer computation graph — attention, FFN, normalization — directly in silicon rather than executing transformer kernels on a general-purpose programmable engine.

---

## Company

| Parameter | Value |
|-----------|-------|
| Company | Etched |
| Founded | 2022 |
| Founders | Gavin Uberti (Co-Founder & CEO), Robert Wachen (Co-Founder & President), Chris Zhu (Co-Founder) — "three Harvard dropouts" |
| Other leadership | Mark Ross (CTO, ex-CTO of Cypress); Saptadeep Pal (VP ASIC & Architecture, ex-NVIDIA H100/A100/V100 architecture team, co-founded Auradine); Brian Loiler (VP Platform); Wayne Cao (VP Production); David Munday (VP Software); Tim Perevozchikov (VP Finance) |
| HQ | San Jose / Cupertino, CA |
| Funding | ~$1.1B raised to date — ~$800M cumulative through Jun 2026 (four unannounced financings incl. VentureTech Alliance; most recent prior round $500M closed Dec 2025, reported Jan 2026) + $300M Series C (Jul 2026). **Do not add $800M to the older $620M figure — $800M is a cumulative total, not a new round.** |
| Valuation | $10.3B post-money (Series C, 2026-07-23); previously $5B post-money ($500M round, Dec 2025) |
| Lead investors | Sequoia Capital (Series C lead), with a16z, Jane Street, Diffusion, Argo, SK Hynix; Stripes led the $500M Dec 2025 round; VentureTech Alliance (TSMC's venture arm); returning backers incl. Peter Thiel, Andrej Karpathy, Dylan Field, Amjad Masad |
| Headcount | 400+ engineers (vendor; TechCrunch says ~400 employees) from NVIDIA, Google TPU, Broadcom, SK Hynix, TSMC |

---

## Chip Specifications

*The table below is the 2024–Q1 2026 Sohu disclosure set. None of these figures appear on etched.com as of
2026-08-08, and none has been restated for the A0 silicon. Treat as historical, not as current-part specs.*

| Parameter | Value |
|-----------|-------|
| Chip name | Sohu (name no longer used in Etched marketing as of mid-2026) |
| Process | TSMC 4nm (2024 disclosure). For the A0 silicon Etched states **TSMC N4P** — vendor-sourced only; see 2026 update |
| Die size | Reticle-limit (~800 mm²), single monolithic die |
| Off-chip memory | 144 GB HBM3E per chip (2024 disclosure; **superseded as the architectural story** by "Cluster Scale Memory" — no per-chip capacity disclosed for the new part) |
| Peak TFLOPS | Not disclosed |
| FLOPS utilization | Claimed 90%+ |
| Data types | Not disclosed (likely BF16/FP16/FP8) |
| Inference throughput | 500,000+ tok/s Llama 70B on 8× server (claimed, 2024; no longer stated by Etched) |
| Training support | No |
| Host interface | Not disclosed |
| Status | A0 silicon returned H1 2026; rack-scale product in customer validation; not shipping as of 2026-08-08 |

---

## 2026 Update — A0 Silicon, "Frontier Inference Clusters", LVI + CSM (June–July 2026)

*Updated 2026-08-08. Primary sources: etched.com homepage, `/progress`, `/progress/frontier-inference-clusters`
(2026-06-30), `/progress/accelerating-inference` (2026-07-23), `/join` — all retrieved 2026-08-08. Independent
corroboration: TechCrunch 2026-06-30 and TechCrunch 2026-07-23. Nothing prior to this section is deleted; the
2024–Q1 2026 Sohu material above is the historical baseline.*

Etched published its first substantive technical update in roughly two years on 2026-06-30, followed by a funding
post on 2026-07-23. The two posts change the record on silicon status, funding, and marketed architecture. They
do **not** supply a single absolute performance, power, capacity, or bandwidth number.

### Silicon status

Verbatim from `/progress/frontier-inference-clusters` (2026-06-30):

> "Earlier this year our A0 silicon came back from TSMC N4P, and today we are busy validating our first
> rack-scale product with customers to fulfill $1B in demand."

- **First silicon (A0) returned in H1 2026.** TechCrunch independently reported the same day that "TSMC
  successfully manufactured its chip"; Yahoo Finance headlined "Etched Emerges From Stealth With Working Chip".
- **Node: TSMC N4P — vendor-stated only.** No independent source names the node; TechCrunch confirms TSMC
  manufactured the chip but not the process. Keep the vendor attribution attached.
- **A0 is a first tapeout revision, not production silicon.** No third party has verified function, clocks,
  yield, or power. (Etched's own CTO bio frames A0 as a repeatable milestone: "Shipped 5 systems generating
  >$1B in revenue, all on A0 silicon.")

### Repositioning — marketing-level; an architecture change is NOT confirmed

Etched now markets "a new category of AI hardware: **frontier inference systems**", co-designing "chips, racks,
software, and manufacturing methods" for **both prefill and decode**, aimed at many-trillion-parameter sparse
MoEs, long context, and agentic workloads. The old framing (transformer-only ASIC, 144 GB HBM3E, 500,000+ tok/s
Llama-70B, "one 8× Sohu server replaces 160 H100s") is absent from the site. As stated in the naming note above,
this is confirmed de-emphasis, not a confirmed pivot — no source says the transformer-specialized approach was
abandoned.

### Low Voltage Inference (LVI) — **vendor marketing claim, zero absolute numbers**

No third-party technical analysis of LVI exists (no Tom's Hardware, EE Times, SemiAnalysis, HPCwire, DCD, or
ServeTheHome coverage surfaced as of 2026-08-08). Etched's claims:

| Claim | Etched's wording | Independent status |
|---|---|---|
| Voltage | math blocks run "at under half the voltage of most AI chips" | TechCrunch paraphrases only as "at a much lower voltage than any other AI chip" — no figure |
| Density | "multiple times the FLOPs density of AI chips today" | Unverified; no FLOPS or mm² figure disclosed |
| Sustained utilization | "trillion-parameter sparse MoEs at 80%+ Peak FLOPs without thermal throttling" | Unverified; no TDP or peak-FLOPS baseline disclosed |
| Enablers | "splittable math arrays", a new PDN/VRM architecture, advanced packaging, cold plates | Named by vendor only; no design detail public |

Absolute voltage, FLOPS, TDP, and die figures: **not disclosed**.

### Cluster Scale Memory (CSM) — **vendor marketing claim, zero absolute numbers**

- Described as an "HBM/SRAM hybrid design [that] solves both memory capacity and mem2mem latency".
- Joined by "a proprietary ultra-low-latency, high-bandwidth interconnect" that creates "a much lower-latency
  shared memory pool across our scale-up domain", targeting **SRAM-level decode speeds**.
- Explicitly contrasted by Etched against SRAM-only chips, 3D DRAM, and optics.
- TechCrunch paraphrase: an interconnect letting many chips "use a shared memory pool at a very, very fast, low
  latency."
- **Capacity: not disclosed. Bandwidth: not disclosed. Latency: not disclosed.** CSM supersedes "144 GB HBM3E
  per chip" as the architectural story, but no per-chip capacity has been stated for the new part and none
  should be inferred.

### Scale-up domain

The 2026-07-23 `/progress` post ("Accelerating Inference") closes a hiring pitch with "thousand-chip scale-up
domains, in-house SMT lines, and new RL environments for recursive kernel generation." This implies a scale-up
domain on the order of ~1,000 chips. **Directional vendor signal, not a spec.** (Note: this phrase is *not* on
`/join`, which contains only bios, values, and job listings.)

### Funding — do not double-count

| Event | Amount | Valuation | Date | Lead / participants |
|---|---|---|---|---|
| Series A | $120M | — | Jun 2024 | — |
| Prior round | $500M | $5B post | closed **Dec 2025** (reported Jan 2026) | Stripes lead; Jane Street, Hudson River Trading, Two Sigma, Ribbit Capital |
| Cumulative through Jun 2026 | **~$800M total raised** | — | stated 2026-06-30 | "four unannounced financings, including a strategic investment from VentureTech Alliance" (TSMC's venture arm) |
| Series C | $300M | **$10.3B post** | 2026-07-23 | **Sequoia Capital** lead; a16z, Jane Street, Diffusion, Argo, SK Hynix; returning: Thiel, Karpathy, Dylan Field, Amjad Masad |

**Running total ≈ $1.1B** (~$800M cumulative through June 2026 plus the $300M Series C). The $800M figure is
TechCrunch-reported as Etched's *cumulative* total at 2026-06-30 — it is **not** a new raise to be added to the
repo's stale $620M, and the correct total is neither $1.42B nor $1.72B. The repo's previous "Series B Jan 2026"
date is corrected to a December 2025 close.

### Commercial status — do not upgrade the verb

As of 2026-08-08: **A0 silicon back; first rack-scale product in customer validation; production/fabrication
started; NOT shipping, NOT deployed at scale.**

- "we've kicked off production to fulfill over $1B in customer contracts" (2026-06-30)
- "We've kicked off fabrication of hundreds of millions of dollars worth of inference clusters" (2026-07-23)
- "Our first racks ship this summer" is a **forward-looking vendor statement made 2026-06-30**. No independent
  evidence that any rack has shipped was found as of 2026-08-08, and **no customer has been named**.
- TechCrunch independently reports "$1 billion in booked orders/contracts", but that figure originates with the
  company.

### Performance

Still **no third-party benchmarks and no absolute vendor numbers**. Etched says only: "Early customer tests show
us achieving SOTA throughput, latency, and power efficiency on inference workloads", and promises "more updates
on our performance and roadmap this summer". An MLPerf Inference v6.0 submitter list could not be retrieved, so
"no Etched MLPerf submission" is recorded as **unverified**, not as a confirmed absence.

### Manufacturing and facilities

| Item | Source | Detail |
|---|---|---|
| Taiwan factory | vendor (`/progress`) | opened; no location, size, or capacity disclosed |
| San Jose office | vendor | data center, test house, and NPI prototyping lab built in-house |
| San Jose HQ datacenter | TechCrunch | 2 MW |
| New lab | vendor: "a new 10-Megawatt lab fifteen minutes from our office"; TechCrunch: ~80,000 sq ft, 10 MW, **Milpitas** | both agree on 10 MW |
| In-house SMT lines | vendor (hiring copy) | mentioned; no detail |

### Hot Chips 2026

Etched is a **Rhodium-level (top-tier) sponsor** of Hot Chips 2026 (Aug 23–25, 2026) and has **no talk in the
advance program**. A sponsorship is not a disclosure and must not be treated as evidence of a forthcoming
architecture reveal — though the vendor did promise a performance update "this summer".

---

## Architecture

*Historical baseline (2024–Q1 2026 disclosures). **Unverified for the current A0 silicon** — Etched's 2026 posts
describe "frontier inference systems" for both prefill and decode and never restate the fixed-pipeline framing.
Retained because no source contradicts it and none says transformer specialization was dropped.*

Sohu implements the transformer computation as a **fixed hardware pipeline**:

```
HBM3E Weights
      ↓
[Input Embedding / Linear Projection]
      ↓
[Attention Engine] ← QKV projection (proprietary non-GEMM hardware)
   - Computes softmax(QKᵀ/√d)V without standard GEMM engine
   - Multi-head, grouped-query, multi-query attention variants
      ↓
[Residual Add]
      ↓
[Layer Norm / RMS Norm]
      ↓
[FFN Engine] ← two linear layers + GELU/SiLU activation
      ↓
[Residual Add + Norm]
      ↓
[Output Projection / Unembedding]
      ↓
Token Output
```

**Key insight**: On a GPU, each of these blocks is a separately dispatched kernel. On Sohu, they are permanent silicon blocks — data flows through them as a fixed pipeline. There is no instruction fetch, no kernel dispatch, no warp scheduling, and no cache miss. Every transistor is always doing transformer work.

---

## Programming Model Rationale: Why Build a Transformer-Only ASIC

The decision to build a transformer-only ASIC is a bet on three simultaneous conditions all remaining true:

**1. Transformers dominate AI workloads indefinitely.**
Since the 2017 "Attention Is All You Need" paper, transformers have consumed AI: GPT, BERT, LLaMA, Claude, Gemini, Stable Diffusion (DiT), vision transformers (ViT), and virtually every frontier model. If this architectural dominance holds, the transformer-only bet is correct and the silicon efficiency advantage is permanent.

**2. FLOPS utilization is the binding constraint.**
GPU transformer inference achieves only ~30-40% of peak FLOPS. This gap exists because the GPU's general-purpose machinery — warp schedulers, instruction decoders, cache hierarchy, memory arbiters — consumes significant silicon area and power without contributing to transformer computation. A transformer ASIC eliminates this overhead. Etched claims 90%+ utilization — a 2-3× efficiency multiplier from the same transistor count.

**3. Silicon specialization is commercially viable at frontier model scale.**
Frontier LLM inference is dominated by a small number of architectures (GPT family, Llama family, MoE variants). The market is large enough that serving only transformers does not limit the addressable opportunity. Etched's bet is that the LLM inference market justifies a dedicated ASIC even with the 3-year re-spin risk if architectures change.

**The fundamental trade-off:**
- **GPU**: 30-40% utilization × enormous addressable workloads = good business
- **Sohu**: 90%+ utilization × transformer-only workloads = potentially great business IF transformers remain dominant

For pure LLM inference at hyperscaler scale, Sohu's architecture is theoretically optimal. The risk is that transformer dominance ends before Sohu ships in volume — or before Etched reaches a second chip generation.

---

## Software Stack

Etched's software stack is intentionally minimal:

```
PyTorch / ONNX (model definition)
         ↓
Etched Transformer Compiler
  (validates architecture, packs weights, maps to pipeline)
         ↓
Sohu Execution Binary
         ↓
Etched Runtime (thin loader + I/O)
         ↓
PCIe Driver
         ↓
Sohu Hardware
```

- **No user ISA** (no CUDA/PTX equivalent)
- **No kernel library** (no cuDNN/CUTLASS equivalent)
- **No dynamic dispatch** (hardware pipeline is fixed)
- SDK still not publicly released as of 2026-08-08

**2026 caveat.** Etched's 2026-07-23 hiring copy mentions "new RL environments for **recursive kernel
generation**", and the company lists a VP of Software (David Munday) and describes co-designing "chips, racks,
**software**, and manufacturing methods". The word "kernel" is in tension with the "no kernel library"
characterization above. This is hiring-copy phrasing with no technical detail attached — recorded as a signal,
not as evidence that a kernel-programming layer exists.

---

## Performance Claims

All figures are from Etched's own marketing. No third-party benchmarks exist. **These 2024-era figures no longer
appear on etched.com as of 2026-08-08 and have not been restated for the A0 silicon** — see the 2026 Update
above, where Etched's only performance language is the unquantified "SOTA throughput, latency, and power
efficiency".

| System | Config | Llama 70B tok/s | vs Sohu |
|--------|--------|-----------------|---------|
| Etched Sohu | 8× server | 500,000+ | 1× |
| NVIDIA B200 | 8× server | ~45,000 | 11× slower |
| NVIDIA H100 | 8× server | ~23,000 | 22× slower |

One 8× Sohu server claimed to replace 160 H100 GPUs.

---

## Strategic Position

**Strengths (updated 2026-08-08):**
- Deep specialization enables maximum FLOPS utilization for transformers
- Simple software stack reduces deployment friction
- 144 GB HBM3E sufficient for frontier LLMs (70B in FP16) — *2024 disclosure; not restated for A0 silicon*
- TSMC 4nm reticle-limit die maximizes transistor budget; A0 silicon vendor-stated on TSMC N4P
- **A0 silicon is back from the foundry (H1 2026)** — the single largest de-risking event in the company's history
- **~$1.1B raised at a $10.3B valuation** with SK Hynix (memory) and VentureTech Alliance (TSMC) both strategic
  investors — supply-chain alignment for HBM and leading-edge wafers
- Vertical integration: Taiwan factory, in-house SMT lines, 2 MW HQ datacenter + 10 MW Milpitas lab

**Risks (updated 2026-08-08):**
- **Still not shipping.** June 2024 announcement → 2026-08-08: A0 silicon back, rack-scale product in customer
  validation, production started, but no shipped rack independently confirmed and no named customer
- All performance claims remain unverified by third parties; the 2026 posts contain **zero absolute numbers**
- LVI and CSM are vendor marketing terms with **no third-party technical analysis whatsoever**
- A0 is a first tapeout revision — function, yield, clocks, and power are unvalidated externally
- Transformer architecture dominance is assumed (SSMs, diffusion, hybrids are alternatives)
- ~3-year re-spin cycle if transformer dominance ends
- Concentrated bet vs. NVIDIA's flexible roadmap
- Valuation moved $5B → $10.3B in ~7 months on a product that has not shipped
- The $1B booked-order figure originates with the company; TechCrunch reports but does not independently verify it

---

## Resources

**2026 update sources (retrieved 2026-08-08)**
- Etched, "Frontier Inference Clusters" (2026-06-30): https://www.etched.com/progress/frontier-inference-clusters
- Etched, "Accelerating Inference" (2026-07-23): https://www.etched.com/progress/accelerating-inference
- Etched progress index: https://www.etched.com/progress
- Etched homepage: https://www.etched.com/
- Etched leadership / careers: https://www.etched.com/join
- TechCrunch (2026-07-23), $300M Series C at $10.3B: https://techcrunch.com/2026/07/23/ai-chip-startup-etched-defies-skeptics-hits-10-3b-valuation-from-big-name-investors/
- TechCrunch (2026-06-30), TSMC manufactured the chip; $800M cumulative: https://techcrunch.com/2026/06/30/nvidia-competitor-etched-hits-5b-valuation-1b-in-sales-for-ai-chip/
- Hot Chips 2026 advance program (Etched = Rhodium sponsor, **no talk**): https://www.hotchips.org/advance-program/

**Historical (2024 – Q1 2026)**
- Official site: https://etched.com
- TechCrunch launch (June 2024): https://techcrunch.com/2024/06/25/etched-is-building-an-ai-chip-that-only-runs-transformer-models/
- Tom's Hardware: https://www.tomshardware.com/tech-industry/artificial-intelligence/sohu-ai-chip-claimed-to-run-models-20x-faster-and-cheaper-than-nvidia-h100-gpus
- Bloomberg ($500M round, reported Jan 2026; round closed Dec 2025): https://www.bloomberg.com/news/articles/2026-01-13/ai-chip-startup-etched-raises-500-million-to-take-on-nvidia
- LessWrong technical analysis: https://www.lesswrong.com/posts/qhpB9NjcCHjdNDsMG/new-fast-transformer-inference-asic-sohu-by-etched
- Wikipedia: https://en.wikipedia.org/wiki/Etched_(company) — *returned 404 on 2026-08-08*
