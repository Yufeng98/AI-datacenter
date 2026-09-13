# Etched Sohu — HW Architecture Investigation

*as_of: 2026-08-08*
*Baseline investigation dated 2026-04-05 retained below in full; see "§11. 2026-08-08 Update" at the end.*

## Summary

The Etched Sohu is the world's first transformer-specific ASIC. It abandons general-purpose programmable compute entirely, instead hardwiring every major operation in the transformer computation graph — multi-head attention, feed-forward network, layer normalization, and linear projections — as dedicated silicon blocks. The key hypothesis: by eliminating all instruction fetch/decode, thread scheduling, memory arbitration, and programmable kernel overhead, Sohu achieves 90%+ FLOPS utilization vs. ~30-40% on GPUs, delivering equivalent effective throughput from a much smaller and cheaper chip.

---

## 1. Core Architectural Concept: Hardcoded Transformer Pipeline

### What "Hardcoding Attention" Means

On a GPU, transformer inference proceeds as a sequence of programmable kernel launches: a GEMM kernel for QKV projection, a FlashAttention kernel for `softmax(QKᵀ/√d)V`, another GEMM for output projection, and so on. Each step requires the GPU's programmable execution engine — SIMT warp schedulers, instruction decoders, L1/L2 cache hierarchy — to interpret and execute software.

On Sohu, the transformer computation graph IS the hardware. The architecture contains:

1. **Hardwired Attention Block**: Dedicated circuitry that computes multi-head attention without using standard GEMM primitives. The Q, K, V projections flow through purpose-built matrix units; the attention score computation (`QKᵀ`) uses a proprietary hardware path designed specifically for the attention dot product pattern. According to Etched, this attention hardware uses "less surface area" to compute attention than a GEMM engine approach, freeing silicon budget for more execution units.

2. **Hardwired FFN Block**: Dedicated pipeline for the two-layer feed-forward network (two linear projections with GELU/SiLU activation between them). This is the most FLOP-intensive part of transformer inference at large batch sizes.

3. **Hardwired Normalization Block**: Layer normalization and RMS normalization executed in dedicated silicon without requiring a programmable kernel.

4. **Hardwired Linear Projection Block**: Input embedding projections and output (unembedding) projection layers.

5. **Dataflow Pipeline**: These blocks are arranged as a fixed pipeline. Activations flow through the pipeline stage-by-stage: attention → FFN → norm → next layer, with HBM3E supplying weights.

### Non-Attention Ops

Non-attention operations (FFN, residual add, normalization, unembedding) are handled by their own dedicated hardware blocks in the same pipeline. There is no "fallback" to a programmable engine — if an operation is not in the hardcoded pipeline, Sohu cannot execute it. This is the fundamental constraint: Sohu can only run architectures that fit the standard transformer blueprint.

---

## 2. Compute Architecture

| Block | Operation | Architecture |
|-------|-----------|-------------|
| Attention Engine | QKV projection + `softmax(QKᵀ/√d)V` + output projection | Proprietary, not GEMM-based |
| FFN Engine | Two linear layers + activation (GELU/SiLU) | Dedicated matrix pipeline |
| Layer Norm Block | Layer norm / RMS norm | Dedicated normalization hardware |
| Residual Add | Element-wise addition | Dedicated ALU |

**FLOPS utilization claim**: 90%+ (vs. ~30-40% for GPU transformer inference)
This is the central performance argument: the 2-3× compute efficiency gain per transistor comes entirely from eliminating programmable overhead.

**No disclosed peak FLOPS figure**: Etched has not published a peak TFLOPS number; they market Sohu by task throughput (tokens/second) rather than raw compute.

---

## 3. Memory Architecture

| Parameter | Sohu | H100 SXM5 | B200 |
|-----------|------|-----------|------|
| Off-chip memory | 144 GB HBM3E | 80 GB HBM3 | 192 GB HBM3e |
| Memory ratio vs H100 | 1.8× capacity | baseline | — |
| Memory ratio vs B200 | 0.75× capacity | — | baseline |
| HBM3E bandwidth per stack | ~1.2 TB/s | — | — |
| Estimated total BW | ~4-5 TB/s (multi-stack) | 3.35 TB/s | 8 TB/s |

**Memory model**: HBM3E for weight storage and KV-cache. No on-chip SRAM cache hierarchy described — the fixed pipeline reads weights and activations directly from HBM3E with timing controlled by the hardwired dataflow.

**KV-cache**: A 144 GB HBM3E budget per chip enables large context windows and large batch sizes. For a 70B model (FP16 = ~140 GB), a single Sohu chip has sufficient capacity for the full model plus KV-cache headroom.

---

## 4. Silicon Budget Argument

The silicon efficiency argument is central to Etched's design thesis:

- A GPU die allocates large area to programmable execution infrastructure: warp schedulers, instruction caches, L1/L2 cache hierarchy, memory controllers, coherence logic, branch prediction.
- A transformer ASIC eliminates this infrastructure entirely. The transistors freed go directly into more transformer-specific execution units.
- Etched claims that per mm² of silicon, Sohu delivers substantially more transformer FLOPS than TSMC 4nm GPU dies of comparable area.
- The reticle-limit die size (~800 mm²) on TSMC 4nm is fully packed with transformer execution units, attention hardware, FFN pipelines, and HBM3E controllers.

---

## 5. Process & Package

| Parameter | Value |
|-----------|-------|
| Foundry | TSMC 4nm |
| Die size | Reticle-limit (~800 mm²), single monolithic die |
| Memory | 144 GB HBM3E (multi-stack, 2.5D CoWoS packaging) |
| Cooling | Not disclosed |
| Host interface | Not disclosed (PCIe or custom) |

Note: B200 uses two dies at the reticle limit; Sohu uses a single reticle-limit die, which simplifies inter-die interconnect but limits maximum transistor count to ~80-100B at 4nm.

---

## 6. Scale-Up System: 8× Sohu Server

| Parameter | Value |
|-----------|-------|
| Chips per server | 8 |
| Llama 70B tok/s | 500,000+ (claimed) |
| H100 equivalent | ~160 H100s (claimed) |
| Interconnect | Not disclosed |
| TDP / Power | Not disclosed |

The server-level interconnect between 8 Sohu chips is not publicly documented. Given the transformer-specific nature, inter-chip communication likely focuses on tensor parallelism (splitting attention heads and FFN across chips) and pipeline parallelism.

---

## 7. MoE Variant

Etched has mentioned a separate Sohu variant or configuration for Mixture-of-Experts models. MoE adds expert routing logic (top-K gating) and sparse expert selection. The hardwired approach for MoE must accommodate dynamic routing patterns, which is architecturally different from dense attention. Specific details not publicly disclosed.

---

## 8. Hardware Design Trade-offs

| Design Decision | What is Gained | What is Given Up |
|-----------------|----------------|------------------|
| Hardcoded transformer pipeline | 90%+ FLOPS utilization; simplicity; no programmable overhead | Cannot run any non-transformer architecture |
| No programmable ISA | Chip is entirely the compute function; zero dispatch overhead | No user-programmable kernels; black-box execution |
| Reticle-limit single die | Maximum transistor count in single die; no die-to-die interconnect | Die yield risk; less redundancy than multi-die |
| HBM3E 144 GB | Sufficient capacity for 70B models + KV-cache | No on-chip SRAM cache; dependent on HBM latency |
| Inference-only | Hardware perfectly tuned for forward pass | No training support (no backward pass hardware) |
| Transformer-only | Deep specialization enables 90%+ utilization | ~3-year re-spin cycle if architectures change |

---

## 9. Comparison: Sohu vs. Groq LPU (two inference ASIC approaches)

| Dimension | Etched Sohu | Groq LPU |
|-----------|-------------|----------|
| Computation model | Hardcoded transformer pipeline | Compiler-scheduled tensor streaming |
| Memory | HBM3E (144 GB) | All-SRAM (~230 MB) |
| General ML ops | No (transformer only) | Limited (must pre-schedule) |
| Attention implementation | Custom hardwired attention | GEMM-based (standard matrix ops) |
| Programmability | None (hardware is the function) | Compiler-scheduled via .iop binary |
| Training | No | No |
| Workload scope | Transformer inference | General-ish inference (but SRAM-constrained) |

---

## 10. Key Uncertainties (as of Q1 2026)

- Exact FLOPS figures not published
- Scale-up interconnect topology not disclosed
- Software stack / SDK not released
- No independent benchmarks
- Chip has not shipped to customers; all claims are from Etched marketing
- MoE variant details undisclosed
- Power consumption undisclosed

---

## 11. 2026-08-08 Update — A0 Silicon, Low Voltage Inference, Cluster Scale Memory

*Investigated 2026-08-08. Everything in §1–§10 above is the 2024–Q1 2026 baseline and is retained unchanged.
Where §1–§10 describes silicon behaviour it must now be read as **unverified for the A0 part**.*

**Evidence base.** Primary: etched.com homepage, `/progress`, `/progress/frontier-inference-clusters`
(2026-06-30), `/progress/accelerating-inference` (2026-07-23), `/join` — all retrieved 2026-08-08. Independent:
TechCrunch 2026-06-30 and TechCrunch 2026-07-23; hotchips.org advance program. Corroboration for this update
rests mainly on those two TechCrunch articles — the verification session's WebSearch budget was exhausted, a
Wikipedia page for Etched returned 404, and Yahoo Finance article bodies were unreachable.

### 11.1 What changed in the record

| Item | Repo baseline (2026-04-05) | Corrected (2026-08-08) | Confidence |
|---|---|---|---|
| Silicon status | "Not shipped as of Q1 2026" | **A0 silicon returned H1 2026**; rack-scale product in customer validation; production started; still **not shipping**, no named customer | confirmed (vendor) + partially corroborated |
| Node | TSMC 4nm | **TSMC N4P** for the A0 part — **vendor-stated only**; TechCrunch confirms only that TSMC manufactured the chip | claimed-unverified (vendor-only) |
| Product name | Sohu | "Sohu" absent from all etched.com pages since mid-2026; marketed only as "frontier inference clusters"/"frontier inference systems". **No source says Sohu was retired or renamed** | confirmed-absence (marketing), NOT a confirmed cancellation |
| Memory story | 144 GB HBM3E per chip | **Cluster Scale Memory** — HBM/SRAM hybrid + proprietary interconnect. **No Gen 2 capacity or bandwidth disclosed**; 144 GB must not be carried forward | claimed-unverified |
| Compute story | 90%+ FLOPS utilization via specialization | **Low Voltage Inference** — FLOPs density via sub-half-voltage math blocks. **No absolute numbers** | claimed-unverified |
| Scale-up | 8-chip server | Rack-scale cluster; "thousand-chip scale-up domains" (hiring copy, directional only) | claimed-unverified |
| Funding | $620M total, $5B valuation, "Series B Jan 2026" | ~**$1.1B** total (~$800M cumulative through Jun 2026 + $300M Series C Jul 2026); **$10.3B** post-money; prior round **closed Dec 2025**, reported Jan 2026 | confirmed (two independent sources) |
| Founders | Uberti, Zhu | **Uberti, Wachen, Zhu** — three co-founders | confirmed |

### 11.2 A0 silicon

Verbatim (etched.com/progress/frontier-inference-clusters, 2026-06-30): *"Earlier this year our A0 silicon came
back from TSMC N4P, and today we are busy validating our first rack-scale product with customers to fulfill $1B
in demand."*

Analysis:
- A0 is the **first tapeout revision**. It is not production silicon and does not imply yield, clock, or power
  targets were met. Etched's own CTO bio treats A0-shipping as an achievement worth naming ("Shipped 5 systems
  generating >$1B in revenue, all on A0 silicon"), which is consistent with A0 being unusual to ship from, not
  routine.
- TechCrunch (2026-06-30) independently states "TSMC successfully manufactured its chip", and Yahoo Finance
  headlined "Etched Emerges From Stealth With Working Chip". Neither names a node or the A0 designation.
- **N4P is therefore single-sourced to the vendor.** Keep the attribution attached everywhere it appears.
- No third party has verified function, clock frequency, yield, or power.

### 11.3 Low Voltage Inference (LVI)

LVI is a **power-delivery/circuit** claim, not a new arithmetic engine, and it carries **zero absolute numbers**.
Vendor wording, in full:

- "We've designed a new architecture to run our chip's math blocks at under half the voltage of most AI chips."
- yields "multiple times the FLOPs density of AI chips today"
- "We can run trillion-parameter sparse MoEs at 80%+ Peak FLOPs without thermal throttling."
- Enablers named: **splittable math arrays**, a new **PDN/VRM architecture**, **advanced packaging**, **cold plates**.

What is *not* disclosed: supply voltage, peak FLOPS, FLOPs/mm², perf/W, TDP, die area, packaging type, clock.
"Most AI chips" is an undefined baseline. TechCrunch's independent paraphrase adds nothing quantitative — only
that the chips run "at a much lower voltage than any other AI chip".

Methodological note for the survey: **the 80%+ figure is not the same metric as the 2024 "90%+ FLOPS
utilization" claim.** The 2024 claim was architectural utilization (fraction of peak achieved because there is
no programmable overhead). The 2026 claim is sustained throughput under a thermal ceiling on MoE workloads.
Etched no longer states the 90%+ figure. Do not present them as a progression.

No third-party technical analysis of LVI exists as of 2026-08-08 (Tom's Hardware, EE Times, SemiAnalysis,
HPCwire, DCD, ServeTheHome all checked; nothing surfaced).

### 11.4 Cluster Scale Memory (CSM)

Vendor wording:
- "Our HBM/SRAM hybrid design solves both memory capacity and mem2mem latency."
- "…a proprietary ultra-low-latency, high-bandwidth interconnect."
- "We created a much lower-latency shared memory pool across our scale-up domain."
- Target: **SRAM-level decode speeds**.
- Explicitly contrasted against **SRAM-only chips**, **3D DRAM**, and **optics**.

What CSM describes architecturally is a **memory-semantic scale-up domain**: a pooled address space spanning many
chips, at latency low enough that decode — which is latency- and KV-cache-bound rather than FLOP-bound — behaves
closer to local SRAM. The competitive framing is coherent: Groq/Cerebras get SRAM latency but not capacity;
HBM-only accelerators get capacity but pay HBM latency and cannot pool across chips; 3D DRAM and optical
memory-semantic fabrics are the two other routes Etched is implicitly rejecting.

**Nothing is quantified.** Not disclosed: HBM generation, stack count, per-chip HBM capacity, SRAM capacity,
on-package bandwidth, fabric link rate, fabric topology, fabric latency, pooled-capacity ceiling, coherence
model. TechCrunch's paraphrase ("use a shared memory pool at a very, very fast, low latency") confirms the
concept exists in the company's pitch but adds no figure.

**Explicit instruction for downstream tables: do not record any per-chip memory capacity for the Gen 2 part.**
The 144 GB HBM3E figure belongs to the 2024 Sohu disclosure only.

### 11.5 Scale-up domain and system packaging

- The unit of product moved from an **8-chip server** to a **rack-scale "frontier inference cluster"**.
- "Thousand-chip scale-up domains" appears in the closing hiring pitch of the 2026-07-23 `/progress` post
  ("Accelerating Inference"), alongside "in-house SMT lines, and new RL environments for recursive kernel
  generation". **Directional signal, not a spec.** It is *not* on `/join` (a prior draft of this finding cited
  the wrong URL; `/join` contains only bios, values, and job listings).
- Chips per rack, rack power, rack cooling design, and scale-out fabric: **not disclosed**.
- Workload framing now covers **both prefill and decode** explicitly, aimed at many-trillion-parameter sparse
  MoEs, long context, and agentic workloads.

### 11.6 Manufacturing and facilities

| Item | Vendor | Independent (TechCrunch) |
|---|---|---|
| Taiwan factory | opened (no location/size/capacity) | not mentioned |
| San Jose office | data center + test house + NPI prototyping lab built in-house | 2 MW datacenter at San Jose HQ |
| New lab | "a new 10-Megawatt lab fifteen minutes from our office" | ~80,000 sq ft, 10 MW, **Milpitas** |
| In-house SMT lines | mentioned in hiring copy | not mentioned |
| Headcount | "400+ engineers from NVIDIA, Google TPUs, Broadcom, SK Hynix, TSMC" | ~400 employees |

Vertical integration of test, NPI, and board assembly is the most concretely verifiable architectural signal in
the 2026 material: it is consistent with a rack-scale product where the chip, the PDN/VRM, the cold plates, and
the scale-up fabric are co-designed and cannot be sourced from a merchant ODM.

### 11.7 Performance and disclosure watch

- **No absolute vendor performance numbers** in either 2026 post. The only language is "Early customer tests
  show us achieving SOTA throughput, latency, and power efficiency on inference workloads."
- Etched promised "more updates on our performance and roadmap this summer" (2026-06-30). Not published as of
  2026-08-08.
- **No third-party benchmarks exist.**
- **MLPerf Inference v6.0**: the submitter list could not be retrieved. Record Etched's absence as
  **unverified**, not as a confirmed absence.
- **Hot Chips 2026 (Aug 23–25, 2026)**: Etched is a **Rhodium-level (top-tier) sponsor** with **no talk in the
  advance program**. A sponsorship is not a disclosure and must not be cited as evidence of a forthcoming
  architecture reveal. (Inference-related talks in that program are Intel "Crescent Island" and Microsoft
  "MAIA 200".)

### 11.8 Remaining uncertainties (2026-08-08)

- Whether the A0 part is a Sohu derivative or a new design — **not disclosed**, and no source addresses it.
- Whether the hardwired-transformer-pipeline microarchitecture survives in the A0 part — **not disclosed**.
- Every LVI and CSM quantity (voltage, FLOPS, capacity, bandwidth, latency, TDP) — **not disclosed**.
- Chips per rack, rack power, scale-out fabric — **not disclosed**.
- Whether any rack has actually shipped — no independent evidence as of 2026-08-08.
- Customer identities — none named.
- MLPerf participation — unverified.
