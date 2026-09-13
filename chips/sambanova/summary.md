# SambaNova RDU Software and Hardware Stack Summary

*as_of: 2026-08-08* (base research 2026-04-05; see "Update — 2026-08-08" below)
*chip: sambanova*
*device_class: Dataflow / Reconfigurable*

---

## Overview

SambaNova Systems builds the **Reconfigurable Dataflow Unit (RDU)** — a spatial, dataflow processor whose architecture is fundamentally different from GPUs (SIMT), systolic arrays (TPUs), and traditional CPUs. The current production chip is the **SN40L** (4th generation, TSMC 5 nm, 2023); the **SN50** (5th generation, TSMC 3 nm) was announced 2026-02-24 with general availability targeted for H2 2026 — **not confirmed shipped to any customer as of 2026-08-08**. What is publicly demonstrated is a single 16-RDU SambaRack SN50 running benchmark workloads.

The defining characteristic: the **SambaFlow compiler maps the entire ML model computation graph onto the PCU/PMU array at compile time**, generating a static binary (PEF) that encodes every operation placement, every data route, and every memory allocation. At runtime, the RDU executes this pre-compiled spatial program — there is no runtime scheduler, no kernel launch overhead, no thread blocks. Data flows from PCU to PCU on a 3D switching fabric without returning to off-chip memory between operators.

The second defining characteristic: a **three-tier memory hierarchy** (on-chip SRAM → HBM → DDR) where all three tiers are directly attached to the accelerator — not the host CPU. The DDR tier (up to 1.5 TiB on SN40L) enables serving trillion-parameter models that cannot fit in GPU HBM.

---

## Software Stack

> **Sourcing note (added 2026-08-08).** Everything in this section describes the **SN40L-era SambaFlow SDK**. As of 2026-08-08 the public SambaFlow developer documentation tree has been **withdrawn** — `docs.sambanova.ai/developer/latest/*`, `/api-reference/` and `/resources/latest/glossary.html` all return HTTP 404, and the current documentation index (174 entries) contains no SambaFlow, compiler, PEF-authoring, or `samba.session` pages of any kind. The material below is therefore re-attributed to the SN40L Hot Chips paper and arXiv 2405.07518 rather than to vendor documentation. **Whether the compiler itself changed for SN50 is unknown**; no source supports claiming it did. The currently documented products are SambaCloud (inference API) and SambaStack (on-prem Kubernetes platform) — see "Update — 2026-08-08" below.

### Framework Integration

- **PyTorch** is the primary ML framework. Users write standard `torch.nn.Module` code; the SambaFlow conversion layer replaces `torch.Tensor` with `SambaTensor` via `samba.from_torch_model()`. Most model code is **unchanged**.
- **TensorFlow** is secondarily supported.
- **Hugging Face Transformers**: natively supported. GPT-2, GPT-J, Llama, BERT variants, and others are documented and tested. Weights load from HF Hub; `samba.session.compile()` takes the HF model directly.
- No kernel authoring required or possible — the hardware is programmed only through the compiler. *(Qualified 2026-08-08: no user-facing kernel language has ever been published, and none appears in the current documentation. But parallelism strategy is a deployment-time choice, not a compiler-hidden one — see "Update — 2026-08-08 §5".)*

### Compiler / IR: SambaFlow Dataflow Compiler

The SambaFlow compiler is a **whole-graph, ahead-of-time (AOT) spatial compiler** — one of the most aggressive compilation strategies in the industry. Its pipeline:

1. **Graph tracing**: `samba.from_torch_model()` traces the PyTorch computation graph
2. **High-level transforms**: operator fusion into sections, meta-pipelining, data/tensor/pipeline parallelism mapping
3. **Hardware-aware lowering**: assigns each operator to a specific PCU; assigns each tensor to a specific PMU or memory tier; routes switching fabric connections; configures AGCUs
4. **Code generation**: produces the **PEF (Program Execution File)** — a binary encoding the complete RDU configuration

**Operator fusion modes:**
- `o0`: one section per operator (debug mode)
- `o1`: maximum fusion (production) — hundreds of operators fused into a single section that executes entirely in on-chip SRAM

Fusion speedup: **2x–13x** vs unfused baseline (per SN40L Hot Chips paper).

**Parallelism**: the compiler maps data parallelism, tensor parallelism, and pipeline parallelism across multiple RDU sockets — the user never writes sharding code. *(Revised 2026-08-08: the compiler maps the chosen parallelism strategy; the strategy itself is a deployment-time choice. SambaNova's 2026-07-30 SN50 post describes an operator selecting between TP16 across all 16 RDUs of a rack (≈800 t/s, highest interactivity) and TP8+DP8 (≈400 t/s, more concurrent users) on the same hardware, and SambaStack ships a "high-throughput vs. high-interactivity" deployment-configuration page. Parallelism is therefore a per-PEF-bundle deployment knob, not something wholly hidden inside the compiler.)*

### Sections and PEF

A **section** is the basic execution unit: a fused subgraph that runs to completion in on-chip SRAM before returning control. The PEF encodes all sections, their routing, and their memory layouts. Compilation happens **once**; the PEF is reused indefinitely.

### Runtime

The **SambaNova Runtime** executes through `samba.session`:
- `samba.session.compile()` — AOT compilation to PEF (minutes, one-time)
- `samba.session.load()` — loads PEF to RDU hardware (seconds, once per process start)
- `samba.session.run()` — executes inference or training (milliseconds per call)

Multi-RDU collective communication (AllReduce, etc.) is compiled into the PEF; there is no separate user-visible communication library (unlike NCCL for GPUs). *(Qualified 2026-08-08: still true in the sense that no SambaNova collectives library is published, but the collective pattern follows from the parallelism strategy selected at deployment time, so the PEF is built per chosen strategy rather than being strategy-agnostic.)*

---

## Hardware Architecture

### Compute Engine: PCU (Pattern Compute Unit)

The PCU is a **multi-stage, reconfigurable SIMD pipeline** — one PCU per innermost-parallel loop body. SN40L has 1040 PCUs per socket (638 BF16 TFLOPS); SN50 roughly doubles this count with 5x more FP8 throughput (3.2 PFLOPS).

Unlike GPU Tensor Cores (which require software to explicitly schedule MMA tiles), PCUs execute their assigned operations continuously once programmed — no scheduling latency.

### Memory: PMU (Pattern Memory Unit) — Distributed Scratchpad

1040 PMUs on SN40L provide **520 MiB of on-chip SRAM** — approximately 10x more than an NVIDIA H100's total shared memory + L1. Crucially, this SRAM is **not a hardware cache**: it is software-managed by the SambaFlow compiler, with exact data placement determined at compile time. PMUs are physically interleaved with PCUs, minimizing wire lengths.

### Three-Tier Memory Hierarchy

| Tier | SN40L | SN50 | Notes |
|------|-------|------|-------|
| On-chip SRAM (PMU) | 520 MiB | 432 MiB | Hundreds of TB/s; compiler-managed |
| HBM (co-packaged) | 64 GiB | 64 GiB @ 1.8 TB/s | Software-managed caching tier |
| DDR (directly attached) | up to 1.5 TiB | up to 2 TiB DDR5 | Model weight reservoir; not host CPU's DDR |

The DDR is the unique differentiator: GPUs have no DDR tier. SambaNova attaches DDR directly to the accelerator package, enabling models like Samba-CoE (150 × 7B = ~1T params) that cannot fit in GPU HBM.

### 3D Switching Fabric

PCUs and PMUs are connected by three parallel on-chip networks:
- **Scalar network** (word-level): loop counters, scalar control values
- **Vector network** (multi-word): bulk tensor data between PCUs/PMUs
- **Control network** (bit-level): predicates, enables, barriers

All routes are statically determined at compile time. No dynamic routing decisions at runtime.

### Scale-up Interconnect

**SN40L (SambaRack)**: 16 RDU sockets connected via P2P network (RDU-Connect); racks interconnected via InfiniBand. Aggregate 10.2 BF16 PFLOPS per rack.

**SN50**: Up to 256 RDUs via a **switched fabric at 2.2 TB/s bidirectional per chip** — 4x more BW than SN40L. Supports 10T-parameter models and 10M-token context.

---

## Programming Model Rationale

**1. Spatial execution eliminates the runtime scheduling bottleneck.** On a GPU, every kernel launch, every NCCL collective, every cuBLAS dispatch involves runtime overhead: ~5-10 µs per launch at the low end. The RDU's static PEF eliminates this entirely — once loaded, the hardware runs at the speed of silicon, not the speed of the scheduler. This is why SambaNova's token-per-second and tokens-per-watt benchmarks are strong despite lower raw FLOPS than H100.

**2. The whole-graph compiler enables fusions impossible on GPUs.** GPU compilers (TorchInductor, XLA) fuse operators within a single dispatch, but cross-kernel fusion requires explicit FlashAttention-style custom kernels. SambaFlow fuses an entire transformer block — or multiple blocks — into a single PEF section that touches DRAM only at boundaries. This eliminates the bandwidth bottleneck that makes LLM inference memory-bound on GPUs.

**3. The three-tier memory architecture solves the memory wall for trillion-parameter models.** The single biggest constraint on LLM inference is model size vs available memory. GPUs are limited to HBM capacity (80-192 GiB), requiring expensive tensor parallelism across many GPUs for large models. SambaNova's DDR tier (1.5 TiB, directly attached) enables storing entire large model ensembles locally, with HBM acting as a software-managed streaming cache.

**4. Composition of Experts requires the DDR tier.** SambaNova's flagship Samba-CoE (150 × 7B experts, ~2.1 TiB total) fits on a single SambaRack because DDR holds all expert weights. The dataflow compiler creates a PEF covering all 150 experts; the router selects which section to activate per request. This architecture is not practical on standard GPU hardware without extremely large multi-GPU clusters.

**5. Reconfigurability means recompilation per model, not per batch.** The RDA is reconfigurable — it can be re-programmed for any model. But unlike a GPU (which dispatches any kernel at runtime), changing the model requires recompiling and loading a new PEF. This is acceptable for deployment scenarios (where the model is fixed) but limits dynamic model switching.

**6. No kernel programming is both a strength and a constraint.** The strength: zero friction for PyTorch users — no CUDA knowledge required. The constraint: advanced users cannot write custom low-level kernels for edge cases, as is possible with Triton/CUDA on GPUs. The compiler must handle all operations.

**7. Intel collaboration (2026) is now an announced heterogeneous-inference product blueprint, not a hypothesis.** *(Corrected 2026-08-08.)* The February 2026 round was a **$350M+ strategic Series E led by Vista Equity Partners and Cambium Capital, with Intel Capital participating** — Intel neither led nor "backed" it in the sense the earlier text implied. What the collaboration produced is concrete: on **2026-04-08** SambaNova and Intel announced a *Blueprint for Heterogeneous Inference* — **GPUs for prefill, SambaNova RDUs for decode, Intel Xeon 6 CPUs as host and "action CPU" for agentic tool/API execution**, availability stated as H2 2026. This is a disaggregated prefill/decode architecture, the same structural pattern being pursued by Cerebras+AMD and NVIDIA+Groq-LPX, and it is the clearest statement to date of where SambaNova believes the RDU belongs in a datacenter: as dedicated token-generation capacity rather than as a general-purpose GPU substitute.

**8. Disaggregation reframes the three-tier memory hierarchy.** *(Added 2026-08-08.)* SambaNova's 2026-04-16 "Decode Era" post maps the tiers onto decode-phase roles explicitly: **SRAM for the hottest local working set, HBM for weights and KV cache, DDR for prompt caching and multi-model/agentic workflows**. Under this framing the DDR tier's value shifts from "hold a 1T-parameter CoE model" (the SN40L pitch) to "hold prompt caches and many models for agentic serving" — the same hardware, a different economic argument.

---

## Chip Specifications

> ⚠️ **Sourcing warning (2026-08-08).** The SN50 column below is largely **extrapolated**. Re-verification on 2026-08-08 confirmed that *none* of the SN50 die-level numbers — 432 MiB SRAM, 64 GiB HBM @ 1.8 TB/s, 2.2 TB/s bidirectional chip-to-chip, 1,600 BF16 TFLOPS / 3.2 FP8 PFLOPS, 256 GiB–2 TiB DDR5, ~2× SN40L PCU count — appear in either the 2026-02-24 press release or the SN50 introduction blog post. SambaNova's own primary material states only: "256 accelerators", "multi-terabyte-per-second interconnect", "four times more network bandwidth than the previous generation", "10T+ parameter models", "10M+ context", "20 kW in a SambaRack", "5X faster", "3X lower TCO". Rows marked *(extrapolated)* below should be treated as unsourced pending a primary disclosure. The first detailed SN50 architecture disclosure is **scheduled** for Hot Chips 38 (Session AI 2, "Dataflow at Scale: the SN50 RDU", Raghu Prabhakar, 2026-08-25) — **content not yet public**; do not cite the talk as a spec source. Re-scan early September 2026.

| Spec | SN40L (Gen 4) | SN50 (Gen 5) |
|------|--------------|-------------|
| Generation | 4th | 5th |
| Release | 2023 | Announced 2026-02-24; GA targeted H2 2026; **not confirmed shipped as of 2026-08-08** |
| Process | TSMC 5 nm | TSMC 3 nm (N3) |
| Transistors | ~102 B | not disclosed |
| Package | Dual-die + HBM + DDR | Dual-chiplet |
| PCU count | 1040 | ~2x SN40L *(extrapolated)* |
| PMU count | 1040 | not disclosed |
| BF16 TFLOPS/socket | 638 | 1,600 *(extrapolated)* |
| FP8 PFLOPS/socket | — | 3.2 *(extrapolated)* |
| On-chip SRAM | 520 MiB | 432 MiB *(extrapolated)* |
| HBM capacity | 64 GiB | 64 GiB *(extrapolated)* |
| HBM bandwidth | ~1 TB/s (node) | 1.8 TB/s *(extrapolated)* |
| DDR capacity | up to 1.5 TiB | 256 GiB – 2 TiB DDR5 *(extrapolated)* |
| Max RDU scale-up | 16 (SambaRack) | **16 demonstrated** (SambaRack SN50, 2026-07); 256 vendor-claimed |
| Chip-to-chip BW | P2P | 2.2 TB/s bidir *(extrapolated;* vendor says only "multi-TB/s" and 4× SN40L*)* |
| Rack power | — | 20 kW per SambaRack (vendor-stated) |
| Max model params | — | 10T+ (vendor claim) |
| Max context tokens | — | 10M+ (vendor claim) |

---

## Update — 2026-08-08 (scan window 2026-04-05 → 2026-08-08)

*Change class: **major**. Verified 2026-08-08. Prior-generation content above is preserved; this section records what changed and corrects superseded statements in place.*

### 1. The SambaFlow developer documentation is gone; docs are now SambaCloud + SambaStack

All five documentation links this summary previously cited return **HTTP 404** as of 2026-08-08 (verified by direct request, not inferred): `docs.sambanova.ai/developer/latest/`, `/developer/latest/porting-overview.html`, `/developer/latest/compiler-o1.html`, `/api-reference/`, `/resources/latest/glossary.html`.

`docs.sambanova.ai` now serves a Mintlify site rooted at `/docs/en/...` covering exactly two products:

| Product | Scope |
|---|---|
| **SambaCloud** | Get Started, Models, Features (OpenAI compatibility, Anthropic compatibility, prompt caching, responses, vision/audio/video), Build, Integrations, API Reference |
| **SambaStack v2.0.2** | Getting started, hardware administration (incl. "RDU module administration", SambaRack Manager command reference), service administration, model deployment, performance, observability, platform administration |

The full index (`docs.sambanova.ai/docs/llms.txt`, 174 entries) contains **no SambaFlow, no compiler, no PEF-authoring, and no `samba.session` API documentation of any kind**. Public SambaFlow SDK documentation is therefore **withdrawn as of ~mid-2026**. Whether the compiler itself changed is **unknown** and no source supports claiming it did.

**PEF survives as the deployment artifact.** SambaStack v2.0.2 exposes a `Pef` Kubernetes custom resource alongside `Model`, `ModelProfile`, `ModelDeployment` and `ModelBundle`; the SambaWiz tool generates "PEF settings" and Kubernetes manifests. What changed is the surface around it: **Helm + Kubernetes CRDs, not a Python SDK.**

### 2. SambaStack version history (from the official release notes)

| Version | Date |
|---|---|
| v0.4.8 | 2026-03-10 |
| v0.5.14 | 2026-04-01 |
| v0.5.17 | 2026-04-08 |
| v1.0.57 | 2026-04-30 |
| v1.1.1 | 2026-05-27 |
| v1.2.0 | 2026-07-07 |
| **v2.0.2** | **2026-08-05** |

v2.0.2 is a **breaking** release: a new four-CRD resource model replaces `BundleTemplate`/`Bundle`/`BundleDeployment` (old format functional until 2026-09-30); Kubernetes (RKE2) moves 1.30 → 1.35; models split into a standalone `sambastack-models` Helm chart (install order `sambastack-base` → `sambastack` → `sambastack-models`); structured output is enforced for constrained-decoding models. The reference architecture documents **LiteLLM** as the gateway and **Prometheus / Grafana / OpenSearch / Fluent Bit** for observability.

Note: the on-prem hardware reference documents remain **SN40L-16**. **No SN50 SambaRack documentation is published.**

### 3. SN50 status: previewed and benchmarked, NOT confirmed shipping

Shipping language has not advanced since the announcement. The 2026-02-24 press release says SN50 "will start shipping to customers later this year"; the 2026-04-08 Intel release states H2 2026 availability. **No source confirms an SN50 delivery to a customer as of 2026-08-08.** The 2026-07-08 press release notes JPMorganChase "will deploy SN40 and SN50 systems" — a stated intent, not a delivery.

What is demonstrated is a **single 16-RDU SambaRack SN50**. The "256 RDU" scale-up figure remains a vendor claim: SambaNova's own 2026-07-30 post frames 64- and 256-chip configurations as future ("As SN50 scales to 64 and 256 chips…").

### 4. Disaggregated prefill/decode — an announced Intel product blueprint

**2026-04-08 press release:** "SambaNova and Intel Announce Blueprint for Heterogeneous Inference: GPUs For Prefill, SambaNova RDUs for Decode, and Intel Xeon 6 CPUs for Agentic Tools." GPUs handle prefill (prompt → KV cache); SN50 RDUs are "the dedicated inference fabric for high-throughput, low-latency decode"; Xeon 6 acts as host and "action CPU" for agentic task coordination and tool/API execution. Availability H2 2026. The release makes **no mention of vLLM, Kubernetes, or any open-source serving stack.**

The supporting narrative is the 2026-04-16 blog post "The Decode Era of AI: Why Dataflow Matters More Than Ever" (SRAM = hottest working set, HBM = weights + KV, DDR = prompt caching and multi-model workflows). Cross-reference for the survey taxonomy: this is the same disaggregated prefill/decode pattern pursued by **Cerebras+AMD** and **NVIDIA+Groq-LPX**.

### 5. vLLM appears as the SN50 serving path — narrowly sourced

Both SN50 performance posts state the results were produced on vLLM. Verbatim, 2026-07-30: "SN50 is deployed using vLLM appearing with ~800 tokens/second at the fastest interactivity"; 2026-07-08: "This demonstration was built and measured on vLLM."

**Hedge this.** There is **no public SambaNova vLLM hardware plugin** — the `sambanova` GitHub org contains no vLLM repository and no `vllm-sambanova` plugin was found — and vLLM is mentioned **nowhere** in the SambaStack v2.0.2 documentation index or in the complete SambaStack release-notes history from Sept 2025 through Aug 2026. The defensible statement is: *SambaNova reports serving SN50 through vLLM in its 2026 benchmark demonstrations; the integration is not open-source and is not referenced in the shipping SambaStack product documentation, so its depth is unverified.*

Consequence for the programming model (applied in place above): the July 30 post describes the operator choosing between two chip-parallelism strategies on the same 16-chip rack — **TP16 across all 16 RDUs → ≈800 t/s**, versus **TP8 + DP8 → ≈400 t/s** with more concurrent users. Parallelism is a deployment-time configuration selected per PEF bundle, corroborated by SambaStack's "high-throughput vs. high-interactivity" deployment page.

### 6. Performance — three numbers, three provenances

| Date | Configuration | Result | Provenance |
|---|---|---|---|
| 2026-07-08 | 1 NVIDIA H200 rack (4 GPUs, prefill) + 1 SambaRack SN50 (16 RDUs, decode), MiniMax M2.7 | up to **850 t/s** short-context, **>450 t/s** long-context | SambaNova-hosted RAISE Summit **preview/demonstration**; attributed to Artificial Analysis |
| 2026-07-30 | Single 16-chip SambaRack SN50, MiniMax M2.7, deployed with vLLM | **TP16 ≈ 800 t/s** at fastest interactivity; **TP8+DP8 ≈ 400 t/s** at a throughput point "starting to approach" B200 on the same model | **Vendor-reported**, from a pre-official run. SambaNova calls it "early validation" |
| 2026-08-08 | Public SambaCloud endpoint, MiniMax-M2.7 | **392.4 output t/s**, rank **#1 of 5** providers (Fireworks 262.1, MiniMax 61.7, Novita FP8 59.3) | **Independent** — Artificial Analysis public provider comparison |

> **Provenance caveat that must travel with the 2026-07-30 figures.** SemiAnalysis has published nothing on SambaNova through its own channels: its newsletter archive for Jun–Aug 2026 contains no such article, and its InferenceX dashboard (`inferencemax.semianalysis.com` now 301-redirects to `inferencex.semianalysis.com`) covers NVIDIA, AMD and TPU v7 Ironwood but lists **no SambaNova hardware**. SambaNova's CEO is quoted in the same post saying the company "look[s] forward to participating in the official InferenceX with SN50." Cite ~800/~400 t/s as **vendor-reported from a pre-official run**, not as an independent third-party benchmark.
>
> **Use the 392.4 t/s figure wherever the survey compares production serving.** Reserve 800–850 t/s for demo-configuration comparisons and label it as such — the demo number is more than 2× the deliverable speed of the public endpoint.

### 7. Corporate

- **2026-07-08:** "SambaNova Completes First Close of $1B Financing at $11B Valuation" — first close of a **Series F**, **$1B**, **$11B post-money**, led by **General Atlantic** with significant participation from Seligman Ventures and T. Rowe Price Associates. Proceeds earmarked to "expand capacity, accelerate product innovation, and scale deployments." Names **JPMorganChase** as an enterprise customer intending to deploy SN40 and SN50 systems.
- **2026-02-24 (resolves the repo's previously unverified claim):** the earlier raise was a **$350M+ strategic Series E led by Vista Equity Partners and Cambium Capital, with Intel Capital participating** as part of a planned multi-year strategic collaboration. Intel Capital was a *participant*, not a lead or "backer" — rationale #7 above is corrected accordingly.
- **2026-04-21:** partnership with TEPCO Systems for Japan's power sector. **2026-06-12** and **2026-08-04:** executive hires (engineering/finance leaders; Mohsen Moazami as Vice Chair of Global Strategy and Partnerships). **2026-07-22:** joined the Genesis Mission Consortium.
- No acquisition, no restructuring, no layoffs found in the window.

### 8. Open-source tooling

`github.com/sambanova/sambastack-tools` — TypeScript, **created 2026-01-15** (pre-baseline: new *to this survey*, not new *in the window*), actively pushed through 2026-08-07, 2 stars, not archived. Two components:

- **SambaEval** — "a local workbench for evaluating LLMs against CSV datasets across multiple models and providers… heuristic and LLM-as-judge scoring, per-row token usage and latency metrics, and pluggable Python output generators for tool use or agentic workflows."
- **SambaWiz** — "a GUI wizard that accelerates the creation and deployment of model bundles on SambaStack… selecting models, configuring PEF settings, generating Kubernetes manifests, and deploying bundles to your cluster." Releases: v2.0.0 (2026-07-31, V3 bundle support), v2.1.0 (2026-08-06, vision + ASR/TTS playground), v2.1.1 (2026-08-07). The SambaStack release notes instruct operators to pull the matching SambaWiz version with each release, so SambaWiz is effectively part of the supported deployment path.

Also in the org and pushed within the window: `sambanova-inference-api-spec` (created 2026-07-21), `sambanova-plugin-cc` (created 2026-03-31), plus SDK/integration repos. There is **no public SambaFlow, model-zoo, or vLLM repository**.

Minor SambaCloud items: Anthropic Messages API support (2026-07-01), prompt caching (2026-07-16), Gemma 4 31B (2026-06-10), MiniMax M2.7 launch (2026-05-05).

### 9. Explicitly NOT confirmed

- **All SN50 die-level specs currently tabulated in this repo** — see the sourcing warning above the Chip Specifications table.
- **Any SN50 architectural detail.** Hot Chips 38 runs 2026-08-23 to 08-25; Session AI 2, Tuesday 2026-08-25, "Dataflow at Scale: the SN50 RDU", Raghu Prabhakar (SambaNova) is confirmed on the advance program — **disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public**. No slides, abstract, or specification exists. It is evidence of nothing about SN50 internals. **Re-scan early September 2026**, at which point the SN50 spec table can likely be put on a primary footing.
- **Whether the SambaFlow compiler and AOT/PEF model changed for SN50**, or whether the vLLM path replaces or wraps it. Unknown.

---

## Resources

### Documentation — current (verified live 2026-08-08)
- [SambaNova documentation root (SambaCloud + SambaStack)](https://docs.sambanova.ai/)
- [Full documentation index — `llms.txt`, 174 entries](https://docs.sambanova.ai/docs/llms.txt)
- [SambaStack release notes (v0.4.8 → v2.0.2)](https://docs.sambanova.ai/docs/en/release-notes/sambastack.md)

### Documentation — WITHDRAWN (all HTTP 404 as of 2026-08-08)

These were the repo's SambaFlow SDK sources through the 2026-04-05 baseline. They are retained here for provenance only; they no longer resolve, and no replacement SambaFlow documentation is published.

- ~~[SambaNova Developer Docs (SambaFlow)](https://docs.sambanova.ai/developer/latest/)~~ — 404
- ~~[Model Conversion Overview](https://docs.sambanova.ai/developer/latest/porting-overview.html)~~ — 404
- ~~[Compiler Optimization Modes](https://docs.sambanova.ai/developer/latest/compiler-o1.html)~~ — 404
- ~~[SambaFlow API Reference](https://docs.sambanova.ai/api-reference/)~~ — 404
- ~~[SambaNova Glossary](https://docs.sambanova.ai/resources/latest/glossary.html)~~ — 404

### Technical Papers
- [SambaNova SN40L: Scaling the AI Memory Wall (arXiv 2405.07518)](https://arxiv.org/html/2405.07518v1)
- [Accelerated Computing with a Reconfigurable Dataflow Architecture (Whitepaper)](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [Evaluating Emerging AI/ML Accelerators: IPU, RDU, and NVIDIA/AMD GPUs](https://arxiv.org/html/2311.04417v3)

### Product
- [SN40L RDU Product Page](https://sambanova.ai/products/rdu-ai-chips)
- [Dataflow Architecture Overview](https://sambanova.ai/products/dataflow-architecture)
- [SambaRack Product Page](https://sambanova.ai/products/sambarack)
- [Introducing the SN50 RDU](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference)

### Press and Blog (2026 update window)
- [SambaNova press index](https://sambanova.ai/press)
- [SambaNova Completes First Close of $1B Financing at $11B Valuation (2026-07-08)](https://sambanova.ai/press/sambanova-completes-first-close-of-1b-financing-at-11b-valuation)
- [SambaNova + Intel: Blueprint for Heterogeneous Inference (2026-04-08)](https://sambanova.ai/press/sambanova-announces-collaboration-with-intel-on-ai-solution)
- [SN50 unveiling / $350M Series E (2026-02-24)](https://sambanova.ai/press/sambanova-unveils-fastest-chip-for-agentic-ai-collaborates-with-intel-and-raises-350m)
- [The Decode Era of AI: Why Dataflow Matters More Than Ever (2026-04-16)](https://sambanova.ai/blog/why-dataflow-matters-more-than-ever)
- [SN50 runs fastest MiniMax speeds in the world (2026-07-08)](https://sambanova.ai/blog/sn50-runs-fastest-minimax-speeds-in-the-world)
- [SemiAnalysis benchmarks SambaRack SN50 on MiniMax M2.7 (2026-07-30)](https://sambanova.ai/blog/semianalysis-benchmarks-sambarack-sn50-with-fast-inference-on-minimax-m2.7)

### Open Source
- [sambanova/sambastack-tools](https://github.com/sambanova/sambastack-tools) — SambaEval + SambaWiz (TypeScript; created 2026-01-15)
- [SambaNova GitHub org](https://github.com/sambanova) — no SambaFlow, model-zoo, or vLLM repository exists publicly

### Third-party measurement
- [Artificial Analysis — MiniMax-M2.7 provider comparison](https://artificialanalysis.ai/models/minimax-m2-7/providers) — SambaNova 392.4 output t/s, rank #1 of 5 providers (2026-08-08)
