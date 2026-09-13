# Etched Sohu Hardware Architecture

*as_of: 2026-08-08*
*generations: Sohu (announced 2024, TSMC 4nm, reticle-limit, 144 GB HBM3E) / "frontier inference cluster" A0 silicon (2026, TSMC N4P per vendor)*

---

## Generation Overview

| Generation | Marketed name | Announced | Process | Compute story | Memory story | Scale-up | Status (2026-08-08) |
|---|---|---|---|---|---|---|---|
| Gen 1 | **Sohu** | Jun 2024 | TSMC 4nm (N4) | Hardwired transformer pipeline; non-GEMM attention engine; claimed 90%+ FLOPS utilization | 144 GB HBM3E per chip; no described on-chip SRAM cache | 8-chip reference server | Never independently confirmed to have shipped; name absent from etched.com since mid-2026 |
| Gen 2 | **"frontier inference cluster" / "frontier inference system"** (no product name disclosed) | 2026-06-30 | **TSMC N4P — vendor-stated only** | **Low Voltage Inference (LVI)** — vendor claim: math blocks at "under half the voltage of most AI chips", "multiple times the FLOPs density"; splittable math arrays + new PDN/VRM. No absolute numbers | **Cluster Scale Memory (CSM)** — vendor claim: HBM/SRAM hybrid + proprietary ultra-low-latency, high-bandwidth interconnect forming a shared memory pool. **No capacity or bandwidth disclosed** | "thousand-chip scale-up domains" (directional vendor signal, not a spec); rack-scale product | **A0 silicon returned H1 2026**; rack-scale product in customer validation; production started; **not shipping** |

**Gen 2 caveats that govern every row above.** (1) The A0 designation and the N4P node come only from Etched's
2026-06-30 post; TechCrunch independently confirms only that "TSMC successfully manufactured its chip". (2) A0
is a first tapeout revision, not production silicon — no third party has verified function, clocks, yield, or
power. (3) LVI and CSM are vendor marketing terms with **zero third-party technical analysis and zero absolute
numbers**. (4) No source states that Sohu was retired or that the transformer-specialized approach was
abandoned; the name is simply absent from Etched's marketing. Everything below that describes Gen 1 is retained
as the historical baseline and should be read as **unverified for the A0 part**.

---

## Overview

The Etched Sohu hardware architecture is built on a single principle: **the transformer is the chip**. Rather than building a programmable compute substrate and then writing transformer kernels on top of it, Sohu bakes every component of the transformer computation graph directly into silicon as fixed-function hardware blocks.

This approach inverts the standard AI chip design philosophy. NVIDIA, AMD, and Google build programmable arrays (SIMT, SIMD, systolic) and achieve transformer inference at ~30-40% FLOPS utilization. Etched hardwires the transformer and claims 90%+ FLOPS utilization — delivering more effective compute per transistor by eliminating all programmable overhead.

---

## 1. Compute Engine: Hardwired Transformer Pipeline

Unlike GPUs (which schedule transformer ops as software kernels on programmable SMs) or systolic arrays (which execute matrix multiply operations via compiler-generated data flows), Sohu implements each transformer operation as a dedicated hardware block arranged in a fixed pipeline:

### Attention Engine

The most distinctive hardware block in Sohu is its **proprietary attention engine**. On a GPU, multi-head attention requires:
- GEMM kernel for Q, K, V projections
- FlashAttention kernel (or equivalent) for `softmax(QKᵀ/√d)V`
- GEMM kernel for output projection

Sohu replaces this with a single hardwired block that:
- Computes QKV projections through dedicated matrix units
- Computes the attention score pattern (`QKᵀ`) using a **non-GEMM hardware path** — proprietary circuitry designed specifically for the attention dot product (where Q and K are both functions of the same input sequence)
- Applies softmax in dedicated hardware
- Produces the attended output `softmax(QKᵀ/√d)V` directly

According to Etched, this non-GEMM attention implementation uses **less silicon area** than a GEMM-based approach, freeing transistor budget for more execution units.

### FFN Engine

The feed-forward network (the dominant compute operation at large batch sizes) is implemented as a dedicated pipeline:
- Linear projection 1 (W₁ x + b₁)
- Activation function (GELU or SiLU — hardwired)
- Linear projection 2 (W₂ x + b₂)

Both linear projections use dedicated matrix multiplication hardware.

### Layer Normalization Block

Layer normalization and RMS normalization (used in modern LLMs like LLaMA) are implemented as dedicated normalization hardware, not as general-purpose vector operations.

### Residual Add

Element-wise residual addition (the skip connections that are added after both attention and FFN blocks in every transformer layer) is handled by dedicated ALU hardware.

### Linear Projection Block

Input embedding projection and output unembedding (the final logit projection) have dedicated hardware.

### 2026 addition — Low Voltage Inference (LVI) — vendor claim, unverified

*Source: etched.com/progress/frontier-inference-clusters, 2026-06-30. No third-party technical analysis of LVI
exists as of 2026-08-08.*

LVI is Etched's name for a **power-delivery and circuit strategy**, not a new arithmetic engine. It is the first
compute-side architectural disclosure since the 2024 Sohu material, and it is entirely qualitative:

| Element | Etched's wording | What is disclosed |
|---|---|---|
| Operating voltage | math blocks run "at under half the voltage of most AI chips" | **No voltage figure.** "Most AI chips" is undefined |
| Density benefit | "multiple times the FLOPs density of AI chips today" | **No FLOPS figure, no mm², no perf/W** |
| Sustained utilization | "We can run trillion-parameter sparse MoEs at 80%+ Peak FLOPs without thermal throttling" | **No peak-FLOPS baseline, no TDP** |
| Circuit enabler | "splittable math arrays" | Named only; no microarchitecture detail |
| Power delivery | a new PDN/VRM architecture | Named only |
| Physical enablers | advanced packaging; cold plates | Named only; packaging type not identified |

The engineering logic implied (not stated by Etched) is the standard one: dynamic power scales with V², so
halving supply voltage buys a large power budget that can be spent on more parallel math, provided the arrays
can be split to tolerate the reduced voltage margin and the PDN can deliver the correspondingly higher current
at low voltage. **Etched has published no data supporting any step of this.** TechCrunch paraphrases the whole
claim as chips running "at a much lower voltage than any other AI chip".

Relation to the Gen 1 story: LVI is a *different* efficiency argument from the 2024 one. Gen 1 argued
utilization (90%+ of peak vs 30–40% on GPUs) via specialization; LVI argues **FLOPs density and sustained
throughput under a thermal ceiling** via voltage. The two are not contradictory, but Etched no longer states the
90%+ utilization figure, and the "80%+ Peak FLOPs" number in the 2026 post is a *sustained-under-thermal* claim
about MoE workloads, not the same metric.

---

## 2. Pipeline Dataflow

The transformer layer computation flows through a fixed hardware pipeline:

```
Input Activations (from HBM3E or previous layer)
         ↓
Layer Norm Block (pre-norm)
         ↓
Attention Engine
  ├── QKV projection (matrix units)
  ├── Attention dot product (proprietary non-GEMM hardware)
  ├── Softmax (dedicated)
  └── Output projection (matrix units)
         ↓
Residual Add
         ↓
Layer Norm Block (pre-norm for FFN)
         ↓
FFN Engine
  ├── Linear 1 (matrix units)
  ├── Activation: GELU / SiLU (dedicated)
  └── Linear 2 (matrix units)
         ↓
Residual Add
         ↓
Output Activations → next layer or token prediction
```

Weights for each operation are streamed from HBM3E on demand. The pipeline timing is fixed by hardware — there is no runtime scheduler determining when each stage executes.

---

## 3. Memory Architecture

### Off-chip Memory: HBM3E

| Parameter | Sohu | H100 SXM5 | B200 |
|-----------|------|-----------|------|
| Type | HBM3E | HBM3 | HBM3e |
| Capacity per chip | 144 GB | 80 GB | 192 GB |
| vs H100 | 1.8× | baseline | — |
| vs B200 | 0.75× | — | baseline |
| Estimated BW | ~4-5 TB/s | 3.35 TB/s | 8 TB/s |

HBM3E is integrated via CoWoS 2.5D packaging (inferred from TSMC 4nm + HBM3E combination standard in this generation).

The 144 GB capacity is sufficient to hold:
- LLaMA 70B in FP16 (~140 GB) with modest KV-cache
- LLaMA 13B in FP16 (~26 GB) with large KV-cache or batch
- GPT-4 class models require multi-chip tensor parallelism

### No Described On-chip SRAM Cache

Unlike GPUs (which have L1/L2 caches + shared memory) or Groq LPU (which has all-SRAM weight storage), Sohu's fixed pipeline reads weights directly from HBM3E with timing governed by the hardwired dataflow. There is no described on-chip cache hierarchy.

### 2026 addition — Cluster Scale Memory (CSM) — vendor claim, unverified

*Source: etched.com/progress/frontier-inference-clusters, 2026-06-30. No third-party technical analysis exists
as of 2026-08-08.*

CSM is the memory story for the A0-silicon part and it **supersedes "144 GB HBM3E per chip" as the architectural
framing** — while disclosing no numbers of its own. Two components:

| Component | Etched's wording | Disclosed |
|---|---|---|
| On-package memory | "Our HBM/SRAM hybrid design solves both memory capacity and mem2mem latency" | HBM present, SRAM present. **HBM generation, stack count, per-chip capacity, SRAM capacity, and bandwidth are all not disclosed** |
| Scale-up fabric | "a proprietary ultra-low-latency, high-bandwidth interconnect"; "We created a much lower-latency shared memory pool across our scale-up domain" | Existence and role only. **Protocol, topology, link rate, per-chip bandwidth, and latency are all not disclosed** |
| Target | "SRAM-level decode speeds" | Qualitative target, no tok/s or ns figure |

Etched explicitly positions CSM **against three named alternatives**:

- **SRAM-only chips** (the Groq/Cerebras approach) — implied criticism: insufficient capacity for
  many-trillion-parameter MoEs.
- **3D DRAM** — implied criticism: not available/adequate at the required capacity-latency point.
- **Optics** — implied criticism: latency and/or power for a memory-semantic fabric.

Architecturally, what CSM describes is a **memory-semantic scale-up domain**: a pooled address space spanning
many chips at latencies low enough that decode (which is latency-bound and KV-cache-bound, not FLOP-bound) sees
something close to local-SRAM behaviour. If real, this is a genuinely new memory tier relative to Gen 1's flat
per-chip HBM3E. **It is asserted, not demonstrated.** TechCrunch's independent paraphrase goes no further than
"an interconnect [letting many chips] use a shared memory pool at a very, very fast, low latency."

**Do not record a per-chip capacity for the Gen 2 part.** None has been disclosed, and the Gen 1 figure of
144 GB must not be carried forward to it.

### Memory summary across generations

| Parameter | Gen 1 (Sohu, 2024) | Gen 2 (A0 / frontier inference cluster, 2026) |
|---|---|---|
| On-chip SRAM | Not described | Present (part of "HBM/SRAM hybrid") — **capacity not disclosed** |
| Off-chip memory | HBM3E | HBM (generation not disclosed) |
| Capacity per chip | 144 GB | **Not disclosed** |
| Bandwidth | ~4–5 TB/s (estimated, never vendor-confirmed) | **Not disclosed** |
| Cross-chip memory | None described | Shared memory pool across scale-up domain via proprietary interconnect (**no numbers**) |
| Memory model | HW-managed via fixed pipeline dataflow | Not disclosed |

---

## 4. Process and Package

| Parameter | Gen 1 (Sohu, 2024) | Gen 2 (A0 silicon, 2026) |
|-----------|--------------------|--------------------------|
| Foundry | TSMC | TSMC (independently confirmed by TechCrunch) |
| Node | 4nm (N4) | **N4P — vendor-stated only**; no independent source names the node |
| Silicon revision | — | **A0** (first tapeout revision), returned H1 2026 |
| Die size | Reticle-limit (~800 mm²) | Not disclosed |
| Die count | 1 (single monolithic die) | Not disclosed |
| Packaging | CoWoS 2.5D (inferred for HBM3E) | "advanced packaging" (vendor wording; type not identified) |
| Transistors | ~80-100B (estimated from 4nm reticle-limit die) | Not disclosed |
| Host interface | Not disclosed (likely PCIe Gen5) | Not disclosed |
| Power / TDP | Not disclosed | Not disclosed |
| Cooling | Not disclosed | Cold plates (vendor-stated, as an LVI enabler) |

**Die count note**: NVIDIA B200 uses two reticle-limit dies (NV-HBI interconnect). Sohu uses a single reticle-limit die — simpler inter-die interconnect but capped at one die's transistor budget.

**A0 note**: A0 denotes the first tapeout revision. Etched's own CTO bio frames A0 as a milestone that can carry
a product ("Shipped 5 systems generating >$1B in revenue, all on A0 silicon"), but no third party has verified
function, clocks, yield, or power for Etched's A0 part.

---

## 5. Scale-up: 8× Sohu Server

| Parameter | Value |
|-----------|-------|
| Chips per reference server | 8 |
| Claimed throughput | 500,000+ tok/s on Llama 70B |
| vs 8× H100 | ~22× faster (claimed) |
| vs 8× B200 | ~11× faster (claimed) |
| GPU equivalent | ~160 H100s (claimed) |
| Inter-chip protocol | Not disclosed |
| Inter-chip topology | Not disclosed |
| Power (server) | Not disclosed |

For a 70B model, 8 chips × 144 GB = 1.152 TB aggregate memory, enabling very large KV-cache (large context windows and/or large batch sizes).

### 2026 — rack-scale system and the "thousand-chip scale-up domain"

*Sources: etched.com/progress (2026-07-23 post "Accelerating Inference"), /progress/frontier-inference-clusters
(2026-06-30).*

The unit of product has moved from an 8-chip server to a **rack**. Etched now markets "frontier inference
clusters" and describes co-designing "chips, racks, software, and manufacturing methods" for **both prefill and
decode**, targeting many-trillion-parameter sparse MoEs, long context, and agentic workloads.

| Parameter | Value (2026-08-08) |
|---|---|
| Product unit | Rack-scale "frontier inference cluster" (no SKU, chip count per rack, or power figure disclosed) |
| Scale-up domain size | **"thousand-chip scale-up domains"** — directional vendor signal from hiring copy, **not a spec** |
| Scale-up protocol | "a proprietary ultra-low-latency, high-bandwidth interconnect" — name, protocol, link rate, and topology **not disclosed** |
| Scale-up semantics | Memory-semantic: shared memory pool across the scale-up domain (see CSM, §3) |
| Scale-out | Not disclosed |
| Rack power / cooling | Not disclosed; cold plates mentioned as an LVI enabler |
| Workload split | Both prefill and decode on the same platform (vendor) |

Note on provenance: the phrase "thousand-chip scale-up domains" appears in the closing hiring pitch of the
2026-07-23 `/progress` post — **not** on `/join`, which contains only leadership bios, values, and job listings.

If the ~1,000-chip figure is directionally right, Etched's scale-up domain would be comparable in chip count to
Google's TPU v8t superpod tier and well above an NVL72-class NVLink domain — but Etched has published no
bisection bandwidth, no topology, and no latency, so **no comparison table entry should carry a number**.

---

## 6. MoE Architecture Variant

Mixture-of-Experts models (DeepSeek, Mixtral, etc.) add a routing layer that selects K experts from N for each token. This creates conditional data flow (different tokens take different paths) that is architecturally different from dense attention.

Etched has announced a separate Sohu variant/configuration for MoE. Details are not publicly disclosed, but the MoE hardware must accommodate:
- Router computation (softmax over expert logits)
- Expert selection (top-K gating)
- Sparse expert dispatch (only K/N experts active per token)
- Expert aggregation (weighted sum of K expert outputs)

**2026 update.** MoE has moved from a side variant to the headline workload. The 2026-06-30 post targets
"many-trillion-parameter MoEs, long context, and agentic workloads" and claims Etched "can run trillion-parameter
sparse MoEs at 80%+ Peak FLOPs without thermal throttling". No routing hardware, expert-placement, or
all-to-all mechanism is described. Etched still discloses no MoE microarchitecture; the only structural hint is
that CSM's cross-chip shared memory pool is the mechanism by which expert weights and KV state are reachable
across a thousand-chip domain.

---

## 7. Hardware Trade-off Analysis

| Design Choice | Advantage | Disadvantage |
|---------------|-----------|--------------|
| Hardwired transformer pipeline | 90%+ FLOPS utilization; no programmable overhead | Cannot execute non-transformer operations |
| Proprietary non-GEMM attention | Less silicon per attention FLOP than GEMM approach | Specific to attention pattern; not reusable for other ops |
| Single reticle-limit die | No die-to-die interconnect overhead | Single die yield risk; no scaling to larger transistor counts without re-spin |
| HBM3E (not on-chip SRAM) | 144 GB capacity — fits 70B models | Dependent on HBM3E latency vs. SRAM-based approach |
| Inference-only | Hardware optimized for forward pass | No training, no fine-tuning on-chip |
| Fixed pipeline | Zero dispatch latency; deterministic execution | Zero flexibility; any new architecture requires 3-year chip re-spin |

---

## 8. Comparison: Sohu vs. Other Inference-Specialized Chips

| Dimension | Etched Sohu | Groq LPU | Cerebras CS-3 |
|-----------|-------------|----------|---------------|
| Architecture | Hardwired transformer pipeline | Compiler-scheduled tensor streaming | Wafer-scale dataflow |
| Compute specialization | Transformer only | General GEMM + vector | General sparse/dense |
| Memory | HBM3E (144 GB) | SRAM only (~230 MB) | On-wafer SRAM (44 GB) |
| Memory BW | ~4-5 TB/s | 80 TB/s (SRAM) | ~20 PB/s (on-wafer) |
| FLOPS utilization | 90%+ (claimed) | High (all-SRAM, pre-scheduled) | High (dataflow) |
| Peak TFLOPS | Not disclosed | 188 FP16 | Not disclosed |
| Scale | 8-chip server | GroqRack 576 chips | 1 wafer = 1 CS-3 |
| Training | No | No | Yes |
| Status (2026-08-08) | A0 silicon back; rack-scale product in customer validation; production started; **not shipping** | Shipping | Shipping |

**2026 note on the memory axis.** The Sohu column above reflects Gen 1. For Gen 2, Etched's Cluster Scale Memory
explicitly repositions the chip *between* the Groq/Cerebras all-SRAM extreme and a conventional HBM-only
accelerator — an HBM/SRAM hybrid with a memory-semantic scale-up fabric. Capacity and bandwidth for that design
are **not disclosed**, so the row cannot yet be filled in.

---

## 9. Disclosure Watch (as of 2026-08-08)

| Item | Status |
|---|---|
| Hot Chips 2026 (Aug 23–25, 2026) | Etched is a **Rhodium-level sponsor** but has **no talk in the advance program**. A sponsorship is not a disclosure |
| Promised vendor update | Etched promised "more updates on our performance and roadmap this summer" (2026-06-30). Not yet published as of 2026-08-08 |
| MLPerf Inference v6.0 | An Etched submission could not be confirmed or ruled out — the submitter list could not be retrieved. Record as **unverified**, not as a confirmed absence |
| Third-party benchmarks | None exist |
| Third-party analysis of LVI / CSM | None found (Tom's Hardware, EE Times, SemiAnalysis, HPCwire, DCD, ServeTheHome all searched) |

---

## Resources

**2026 (retrieved 2026-08-08)**
- Etched, "Frontier Inference Clusters" (2026-06-30) — primary source for A0/N4P, LVI, CSM: https://www.etched.com/progress/frontier-inference-clusters
- Etched, "Accelerating Inference" (2026-07-23) — Series C, "thousand-chip scale-up domains": https://www.etched.com/progress/accelerating-inference
- Etched progress index: https://www.etched.com/progress
- Etched homepage: https://www.etched.com/
- TechCrunch (2026-06-30) — TSMC manufactured the chip; $800M cumulative raised: https://techcrunch.com/2026/06/30/nvidia-competitor-etched-hits-5b-valuation-1b-in-sales-for-ai-chip/
- TechCrunch (2026-07-23) — $300M at $10.3B; LVI/CSM paraphrase; Milpitas 80,000 sq ft / 10 MW: https://techcrunch.com/2026/07/23/ai-chip-startup-etched-defies-skeptics-hits-10-3b-valuation-from-big-name-investors/
- Hot Chips 2026 advance program: https://www.hotchips.org/advance-program/

**Historical**
- TechCrunch: https://techcrunch.com/2024/06/25/etched-is-building-an-ai-chip-that-only-runs-transformer-models/
- Tom's Hardware: https://www.tomshardware.com/tech-industry/artificial-intelligence/sohu-ai-chip-claimed-to-run-models-20x-faster-and-cheaper-than-nvidia-h100-gpus
- LessWrong technical deep-dive: https://www.lesswrong.com/posts/qhpB9NjcCHjdNDsMG/new-fast-transformer-inference-asic-sohu-by-etched
- Bloomberg ($500M round, reported Jan 2026; round closed Dec 2025): https://www.bloomberg.com/news/articles/2026-01-13/ai-chip-startup-etched-raises-500-million-to-take-on-nvidia
- Wikipedia: https://en.wikipedia.org/wiki/Etched_(company) — *404 on 2026-08-08*
