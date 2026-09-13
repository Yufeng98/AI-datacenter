# SambaNova RDU Hardware Architecture

*as_of: 2026-04-05*
*chip: sambanova*
*Architectures: SN40L (Gen 4, 2023) and SN50 (Gen 5, 2026)*

---

## Overview

The SambaNova Reconfigurable Dataflow Unit (RDU) is a **spatial, dataflow processor** — the antithesis of the von Neumann/SIMT model. Rather than fetching and dispatching instructions at runtime, the SambaFlow compiler **statically maps the entire model computation graph onto PCU/PMU arrays at compile time**. Each PCU is assigned its operation for the life of the inference or training run; data flows through the chip on a 3D switching fabric without returning to memory between operator boundaries.

This architecture eliminates:
- Runtime thread scheduling overhead
- Kernel launch latency
- Redundant DRAM round-trips between operator calls

The tradeoff: the hardware is only as effective as the compiler. A given PEF binary encodes one specific model-to-hardware mapping.

---

## 1. Compute Engine: Pattern Compute Unit (PCU)

### Architecture

The **PCU** is the fundamental compute cell. It implements a **multi-stage, reconfigurable SIMD pipeline** designed to execute a single innermost-parallel loop body — typically a reduction, accumulation, or elementwise operation.

Key characteristics:
- Reconfigurable: pipeline stages are configured at compile time via the PEF
- SIMD: executes across multiple data lanes in parallel
- Pipelined: multiple stages overlap execution (pipeline parallelism across stages)
- Independent: each PCU executes its assigned portion of the model graph autonomously

### SN40L Compute Specifications

| Spec | Value |
|------|-------|
| PCU count per socket | 1040 |
| Peak BF16 TFLOPS per socket | 638 |
| FP16 TFLOPS per socket (TechInsights) | ~688 |
| Transistors | ~102 billion |
| Process | TSMC 5 nm |
| Die config | Dual-die (two logic dies) |
| TDP (estimated) | — |

### SN50 Compute Specifications

| Spec | Value |
|------|-------|
| Process | TSMC 3 nm (N3) |
| Die config | Dual-chiplet |
| BF16 PFLOPS per socket | 1.6 |
| FP8 PFLOPS per socket | 3.2 |
| Compute vs SN40L | 5x more at FP8, 2.5x more at BF16 |
| Native FP8 support | Yes (added in SN50) |

### SambaRack Compute (SN40L-16)

| Spec | Value |
|------|-------|
| RDU sockets per rack | 16 |
| Aggregate BF16 TFLOPS | ~10.2 PFLOPS |
| Aggregate PCUs | 16,640 |
| Aggregate PMUs | 16,640 |

---

## 2. Memory Unit: Pattern Memory Unit (PMU)

### Architecture

The **PMU** is a distributed, software-managed scratchpad SRAM — not a hardware cache. PMUs are interspersed with PCUs across the RDU tile array, minimizing data movement by placing memory physically close to compute.

Distinguishing features vs GPU shared memory:
- Much larger total capacity (520 MiB on SN40L vs ~30 MB total SMEM on H100)
- Compiler manages placement — data location is statically determined at compile time
- Supports specialized access patterns including:
  - Sequential streaming
  - Arbitrary strides
  - Sparse access via AGCU
  - Bank-level parallelism within and across PMUs

### PMU Specifications

| Spec | SN40L | SN50 |
|------|-------|------|
| PMU count per socket | 1040 | — |
| Total on-chip SRAM | 520 MiB | 432 MiB |
| Aggregate on-chip BW | Hundreds of TB/s | — |
| Management | Software (compiler-controlled) | Software |

Note: SN50's on-chip SRAM is slightly lower than SN40L (432 vs 520 MiB) per socket, with the increase in HBM bandwidth compensating.

---

## 3. Switching Fabric

### 3D On-Chip Network

PCUs and PMUs are connected through a three-dimensional switching fabric consisting of **three independent parallel networks**:

| Network | Granularity | Purpose |
|---------|-------------|---------|
| Scalar network | Word-level (32-bit) | Control values, loop counters, scalar operands |
| Vector network | Multi-word | Bulk tensor data movement between PCUs/PMUs |
| Control network | Bit-level | Predicates, enables, synchronization signals |

These networks run **in parallel** alongside the PCUs and PMUs. The 3D topology enables point-to-point routing between any PCU/PMU pair without congestion bottlenecks, as the compiler statically routes all data paths at compile time.

---

## 4. Address Generation and Coalescing Unit (AGCU)

The **AGCU** sits at the boundary between on-chip and off-chip memory, providing:
- **Address generation**: computes off-chip memory addresses for streaming data
- **Coalescing**: batches multiple RDU memory requests into efficient DRAM bursts
- **Sparse data support**: handles irregular access patterns for sparse tensors and graph datasets
- **P2P interface**: connects the RDU to other RDUs in a multi-chip configuration
- **Host interface**: connects to the host processor for control-plane operations

---

## 5. Three-Tier Memory Hierarchy

The SN40L's defining memory innovation is **three tiers of directly-attached memory**, with the HBM and DDR physically co-packaged with (or directly attached to) the RDU die — not connected via PCIe to a host CPU.

### Tier 1: On-Chip PMU SRAM

| Parameter | SN40L | SN50 |
|-----------|-------|------|
| Capacity | 520 MiB | 432 MiB |
| Bandwidth | Hundreds of TB/s | — |
| Latency | Sub-cycle (single cycle per PMU) | — |
| Management | Compiler-managed (static) | Compiler-managed |
| Role | Active computation data, intermediate tensors | Same |

### Tier 2: Co-Packaged HBM

| Parameter | SN40L | SN50 |
|-----------|-------|------|
| Capacity | 64 GiB | 64 GiB |
| Bandwidth | ~1 TB/s (per socket) | 1.8 TB/s |
| Standard | HBM (gen unspecified in SN40L) | HBM2E |
| Role | Software-managed caching tier: model weights for active layers | Same |
| Function | Temporal/spatial locality exploitation for generative inference | Same |

The HBM acts as a **programmer-visible staging buffer** between DDR and on-chip SRAM. The SambaFlow compiler determines what lives where.

### Tier 3: Directly-Attached DDR DRAM

| Parameter | SN40L | SN50 |
|-----------|-------|------|
| Capacity | Up to 1.5 TiB (pluggable DIMMs) | 256 GiB – 2 TiB |
| Standard | DDR4/DDR5 | DDR5 |
| Transfer rate to HBM | >1 TB/s per SN40L Node (16-socket) | — |
| Role | Model weight reservoir; entire multi-hundred-billion-parameter models stored here | Same |
| Attachment | Direct to RDU die package (not to host CPU) | Direct |

The DDR being attached to the accelerator (not the host) is key: it enables the SambaFlow compiler to reason about data movement entirely within the accelerator domain, without PCIe transfers.

### Memory Hierarchy Summary Comparison

| Feature | SambaNova SN40L | NVIDIA H100 | Google TPU v4 |
|---------|----------------|-------------|---------------|
| On-chip memory | 520 MiB SRAM (software) | 30 MB SMEM + 50 MB L2 (hardware cache) | 96 MiB (VMEM) |
| Near-chip memory | 64 GiB HBM | 80 GiB HBM3 | 32 GiB HBM2 |
| Extended memory | 1.5 TiB DDR (attached) | None (must use host RAM over PCIe) | None |
| Memory management | Fully compiler-managed | HW cache + programmer-managed SMEM | Compiler (XLA) |

---

## 6. Package and Die Configuration

### SN40L Package

- **Two logic dies** on a single package
- HBM stacks co-packaged with logic dies on an interposer
- Pluggable DDR DIMMs directly attached to the board/socket
- Process: TSMC 5 nm
- ~102 billion transistors

### SN50 Package

- **Dual-chiplet** design (analogous to Blackwell's dual-die, but from a different foundational approach)
- Process: TSMC 3 nm (N3)
- Roughly doubles PCU/PMU count of SN40L
- HBM2E co-packaged (64 GiB, 1.8 TB/s)
- DDR5 directly attached (256 GiB – 2 TiB)

---

## 7. Scale-up Interconnect

### SN40L: P2P + InfiniBand

- 16 RDU sockets per SambaRack connected via **peer-to-peer (P2P) network** (RDU-Connect)
- Racks interconnected via **InfiniBand**
- Collective communication (AllReduce, AllGather) built on top of P2P primitives
- All-to-all model parallelism across the 16 sockets

### SN50: High-Speed Switched Fabric

| Spec | Value |
|------|-------|
| Chip-to-chip bandwidth | 2.2 TB/s bidirectional per RDU |
| Max scale-up | 256 RDU chips |
| Network BW vs SN40L | 4x |
| Max model size supported | 10 trillion parameters |
| Max context length | 10 million tokens |
| Interconnect fabric | Switched (vs P2P in SN40L) |

The SN50's switched fabric enables non-blocking all-to-all communication between 256 RDUs — critical for serving trillion-parameter Composition of Experts models.

---

## 8. Dataflow vs GPU/TPU: Architectural Comparison

| Dimension | SambaNova RDU | NVIDIA GPU | Google TPU |
|-----------|--------------|------------|------------|
| Execution model | Spatial dataflow (compile-time scheduled) | SIMT (runtime scheduled) | Systolic array (runtime dispatched) |
| Compute mapping | Static: compiler places ops on PCUs | Dynamic: scheduler assigns warps | Static: XLA lowers to systolic tiles |
| Memory management | Fully static (compiler) | Mixed: HW cache + programmer SMEM | Compiler-controlled (XLA + VMEM) |
| On-chip memory type | Software-managed SRAM (PMU) | HW-managed L1/L2 + programmer SMEM | Software-managed VMEM |
| Inter-op data movement | Direct PCU-to-PCU on switching fabric (no DRAM) | Explicit kernel-to-kernel DRAM round-trips (unless fused) | Systolic forwarding within array |
| Runtime overhead | Near-zero (configuration loaded at session start) | Per-kernel launch, cuBLAS dispatch (~5-10 µs) | XLA-scheduled dispatch |
| Programmability | Model-level (PyTorch → PEF) | Kernel-level (CUDA C++, Triton) | Framework-level (JAX/XLA) |
| Reconfigurability | Per-model (recompile for different model) | Per-kernel (runtime) | Per-graph (XLA compilation) |
| Model size support | Up to 10T params (SN50) via DDR tier | Limited by HBM capacity | Limited by HBM capacity |

---

## Sources

- [SambaNova SN40L: Scaling the AI Memory Wall (arXiv 2405.07518)](https://arxiv.org/html/2405.07518v1)
- [Accelerated Computing with a Reconfigurable Dataflow Architecture (Whitepaper)](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [SN40L RDU Product Page](https://sambanova.ai/products/rdu-ai-chips)
- [Dataflow Architecture Overview](https://sambanova.ai/products/dataflow-architecture)
- [Introducing the SN50 RDU](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference)
- [Evaluating Emerging AI/ML Accelerators: IPU, RDU, and NVIDIA/AMD GPUs](https://arxiv.org/html/2311.04417v3)
- [Accelerating Scientific Applications with SambaNova RDA](https://sambanova.ai/blog/accelerating-scientific-applications-with-sambanova-reconfigurable-dataflow-architecture)
- [SambaNova Pits Its Engineering Against Nvidia For Agentic AI (NextPlatform)](https://www.nextplatform.com/ai/2026/02/25/sambanova-pits-its-engineering-against-nvidia-for-agentic-ai/4092613)
- [SambaRack Datasheet](https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf)

---

# Investigation Update — 2026-08-08

*Scan window 2026-04-05 → 2026-08-08. Change class: **major** (driven mainly by software/product changes; see the software-stack investigation for those). This section is additive — nothing above is deleted. Where this section and the material above disagree, this section wins.*

## A. SN50 status correction

The document above dates SN50 as "Gen 5, 2026" and the chip summary said "shipping H2 2026". Re-verified position as of 2026-08-08:

| Question | Answer | Evidence |
|---|---|---|
| Announced? | Yes, 2026-02-24 | Press release "SambaNova Unveils Fastest Chip for Agentic AI…" |
| Shipping to customers? | **No confirmation.** The 2026-02-24 release says SN50 "will start shipping to customers later this year"; the 2026-04-08 Intel release states H2 2026 availability | Both press releases |
| Any named customer? | JPMorganChase "will deploy SN40 and SN50 systems" — stated **intent**, not a delivery | 2026-07-08 Series F press release |
| Largest configuration publicly demonstrated? | **One 16-RDU SambaRack SN50** | Vendor blog posts 2026-07-08 and 2026-07-30 |
| 256-RDU scale-up? | Vendor claim only. SambaNova's own 2026-07-30 post writes "As SN50 scales to 64 and 256 chips…", i.e. future tense | 2026-07-30 blog post |
| On-prem hardware documentation? | Only **SN40L-16**. No SN50 SambaRack documentation is published in SambaStack v2.0.2 | docs.sambanova.ai index (174 entries) |

## B. SN50 die-level specs are extrapolated, not sourced

This is a correction of pre-existing repo content. The SN50 numbers carried in this document and in `chips/sambanova/` — **432 MiB SRAM, 64 GiB HBM @ 1.8 TB/s, 2.2 TB/s bidirectional chip-to-chip, 1,600 BF16 TFLOPS / 3.2 FP8 PFLOPS, 256 GiB–2 TiB DDR5, ~2× SN40L PCU count** — do **not** appear in either the 2026-02-24 press release or the SN50 introduction blog post, and could not be corroborated in this scan.

What SambaNova's own primary material actually states about SN50:

- "256 accelerators"
- "multi-terabyte-per-second interconnect"
- "four times more network bandwidth than the previous generation"
- "10T+ parameter models"
- "10M+ context"
- "20 kW in a SambaRack"
- "5X faster", "3X lower TCO"

Everything else is extrapolation and is now labelled as such throughout `chips/sambanova/`. **Do not promote these to confirmed without a primary disclosure.**

## C. Demonstrated SN50 deployment configurations (16-RDU rack)

The 2026-07-30 post is architecturally informative independent of its performance numbers, because it names the parallelism configurations:

| Configuration | Mapping across the 16 RDUs | Reported throughput (MiniMax M2.7) |
|---|---|---|
| Fastest interactivity | TP16 — tensor parallel across all 16 RDUs | ≈800 output tokens/sec |
| More concurrent users | TP8 + DP8 — tensor parallel across 8, data parallel across the other 8 | ≈400 output tokens/sec |

Implication for the dataflow programming model: the parallelism *strategy* is chosen at deployment time per PEF bundle; the compiler maps the chosen strategy. This qualifies the long-standing repo claim that sharding is entirely compiler-hidden.

## D. Performance provenance (three distinct numbers)

| Date | Setup | Figure | Status |
|---|---|---|---|
| 2026-07-08 | 1 NVIDIA H200 rack (4 GPUs, prefill) + 1 SambaRack SN50 (16 RDUs, decode), MiniMax M2.7 | up to 850 t/s short-context; >450 t/s long-context | Vendor-hosted RAISE Summit **preview/demonstration**, attributed to Artificial Analysis |
| 2026-07-30 | 1 × 16-chip SambaRack SN50, MiniMax M2.7, served with vLLM | TP16 ≈800 t/s; TP8+DP8 ≈400 t/s ("starting to approach" B200 on the same model) | **Vendor-reported, pre-official run.** SambaNova calls it "early validation" |
| 2026-08-08 | Public SambaCloud endpoint, MiniMax-M2.7 | **392.4 output t/s**, rank #1 of 5 providers (Fireworks 262.1, MiniMax 61.7, Novita FP8 59.3) | **Independent** — Artificial Analysis provider comparison |

**Provenance caveat.** The "SemiAnalysis benchmark" framing does not survive checking. SemiAnalysis has published nothing about SambaNova on its own channels — its newsletter archive for Jun–Aug 2026 contains no such article, and its InferenceX dashboard (`inferencemax.semianalysis.com` 301-redirects to `inferencex.semianalysis.com`) lists NVIDIA, AMD and TPU v7 Ironwood but no SambaNova hardware. SambaNova's CEO is quoted in the same post looking forward to "participating in the official InferenceX with SN50." Treat ~800/~400 t/s as vendor-reported.

**Survey guidance:** use **392.4 t/s** for production-serving comparisons; reserve 800–850 t/s for demo-configuration comparisons and label them.

## E. System-level architecture change: disaggregated prefill/decode

On **2026-04-08** SambaNova and Intel announced a "Blueprint for Heterogeneous Inference":

| Stage | Hardware | Role (per the release) |
|---|---|---|
| Prefill | GPUs | Prompt → KV cache |
| Decode | SN50 RDUs | "The dedicated inference fabric for high-throughput, low-latency decode" |
| Agentic tools | Intel Xeon 6 | Host and "action CPU": agentic task coordination, tool/API execution |

Availability stated as H2 2026. The release makes **no mention of vLLM, Kubernetes, or any open-source serving stack**. The 2026-07-08 demonstration is the concrete instance of this blueprint.

The supporting narrative, "The Decode Era of AI: Why Dataflow Matters More Than Ever" (2026-04-16), re-maps the three memory tiers onto decode roles: SRAM = hottest local working set, HBM = weights and KV, DDR = prompt caching and multi-model workflows. The hardware is unchanged; the argument for it is not.

**Taxonomy cross-reference for the survey:** this is the same disaggregated prefill/decode structure as Cerebras+AMD and NVIDIA+Groq-LPX. Three independent non-GPU vendors converging on "GPU prefill + specialist decode" is a taxonomy-level observation, not a per-chip detail.

## F. Corporate context bearing on hardware roadmap credibility

- **2026-07-08:** first close of a **Series F** — **$1B at $11B post-money**, led by **General Atlantic**, with significant participation from Seligman Ventures and T. Rowe Price Associates. Proceeds earmarked to "expand capacity, accelerate product innovation, and scale deployments."
- **2026-02-24 (corrects a previously unverified repo claim):** the earlier raise was a **$350M+ strategic Series E led by Vista Equity Partners and Cambium Capital, with Intel Capital participating**. Intel Capital was a participant, not the lead or the "backer".
- 2026-04-21 TEPCO Systems partnership (Japan power sector); 2026-07-22 joined the Genesis Mission Consortium; executive hires 2026-06-12 and 2026-08-04. No acquisition, restructuring, or layoffs found.

## G. Open items for the next scan

1. **Hot Chips 38, Session AI 2, Tuesday 2026-08-25: "Dataflow at Scale: the SN50 RDU", Raghu Prabhakar (SambaNova).** Confirmed on the advance program. The conference runs 2026-08-23 to 08-25 — **15 days after this scan. Disclosure scheduled; content not yet public. It is evidence of nothing about SN50 internals and must not be cited as a spec source.** Re-scan early September 2026; at that point most of section B can likely be put on a primary footing.
2. Whether an SN50 SambaRack appears in SambaStack hardware documentation (currently SN40L-16 only).
3. Whether SN50 enters the official SemiAnalysis InferenceX dataset.
4. Whether any SN50 customer delivery is confirmed.

## Sources added 2026-08-08

- https://sambanova.ai/press — press index (the 2026-04-05-baseline research covered only /blog and missed both the Intel blueprint and the Series F)
- https://sambanova.ai/press/sambanova-completes-first-close-of-1b-financing-at-11b-valuation
- https://sambanova.ai/press/sambanova-announces-collaboration-with-intel-on-ai-solution
- https://sambanova.ai/press/sambanova-unveils-fastest-chip-for-agentic-ai-collaborates-with-intel-and-raises-350m
- https://sambanova.ai/blog/why-dataflow-matters-more-than-ever
- https://sambanova.ai/blog/sn50-runs-fastest-minimax-speeds-in-the-world
- https://sambanova.ai/blog/semianalysis-benchmarks-sambarack-sn50-with-fast-inference-on-minimax-m2.7
- https://artificialanalysis.ai/models/minimax-m2-7/providers
- https://newsletter.semianalysis.com/archive?sort=new — negative evidence
- https://inferencex.semianalysis.com/ — negative evidence (no SambaNova hardware listed)
- https://docs.sambanova.ai/docs/llms.txt — 174-entry index; SN40L-16 is the only documented on-prem rack
- https://hotchips.org/advance-program/ — SN50 talk scheduled 2026-08-25; not usable as a spec source
