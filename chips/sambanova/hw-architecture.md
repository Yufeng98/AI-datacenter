# SambaNova RDU Hardware Architecture

*as_of: 2026-08-08*
*Architectures: SN40L (Gen 4, 2023) and SN50 (Gen 5, announced 2026-02-24; GA targeted H2 2026; not confirmed shipped as of 2026-08-08)*

---

## Generation Overview

| Generation | Status (2026-08-08) | Process | Package | Compute (per socket) | On-chip SRAM | HBM | DDR | Scale-up | Primary source basis |
|---|---|---|---|---|---|---|---|---|---|
| **SN40L (Gen 4, 2023)** | Shipping; SambaRack SN40L-16 is the documented on-prem reference platform in SambaStack v2.0.2 | TSMC 5 nm | Dual-die + HBM + DDR, ~102 B transistors | 638 BF16 TFLOPS (~688 FP16) | 520 MiB (1040 PMUs) | 64 GiB, ~1 TB/s | up to 1.5 TiB DDR4/5 | 16 RDUs / SambaRack via RDU-Connect P2P; InfiniBand rack-to-rack | arXiv 2405.07518 (SN40L Hot Chips paper), RDA whitepaper, SambaRack datasheet |
| **SN50 (Gen 5)** | **Announced 2026-02-24; GA targeted H2 2026; no customer delivery confirmed as of 2026-08-08.** A single 16-RDU SambaRack SN50 has been publicly benchmarked (2026-07) | TSMC 3 nm (N3) | Dual-chiplet | 1.6 BF16 PFLOPS / 3.2 FP8 PFLOPS *(extrapolated — see warning)* | 432 MiB *(extrapolated)* | 64 GiB @ 1.8 TB/s *(extrapolated)* | 256 GiB – 2 TiB DDR5 *(extrapolated)* | **16 RDUs demonstrated**; 256 vendor-claimed; "multi-TB/s" fabric, 4× SN40L network BW | Vendor announcement only: "256 accelerators", "multi-terabyte-per-second interconnect", "4× network bandwidth", "10T+ params", "10M+ context", "20 kW per SambaRack", "5X faster", "3X lower TCO" |

> ⚠️ **SN50 sourcing warning (added 2026-08-08).** Re-verification on 2026-08-08 could not corroborate *any* SN50 die-level figure in this document — 432 MiB SRAM, 64 GiB HBM @ 1.8 TB/s, 2.2 TB/s bidirectional chip-to-chip, 1,600 BF16 TFLOPS / 3.2 FP8 PFLOPS, 256 GiB–2 TiB DDR5, ~2× SN40L PCU count. None appear in the 2026-02-24 press release or the SN50 introduction blog post. Every SN50 row below marked *(extrapolated)* is unsourced pending primary disclosure. The first detailed SN50 architecture disclosure is **scheduled** for Hot Chips 38, Session AI 2, 2026-08-25 ("Dataflow at Scale: the SN50 RDU", Raghu Prabhakar) — **disclosure scheduled, Hot Chips 38, Aug 2026; content not yet public.** It cannot be cited as a spec source. Re-scan early September 2026.

---

## Overview

The SambaNova Reconfigurable Dataflow Unit (RDU) implements a **spatial, dataflow execution model**. The SambaFlow compiler maps an entire ML model computation graph onto the PCU/PMU tile array at compile time. Data flows through the chip on a 3D switching fabric; PCUs execute their statically assigned operations without a runtime scheduler. There are no warps, no thread blocks, no kernel launches.

This document covers all hardware layers for SN40L and SN50 generations.

---

## 1. Compute Engine

### Pattern Compute Unit (PCU)

The PCU is the fundamental compute cell — a **multi-stage reconfigurable SIMD pipeline**:

| Component | Description |
|-----------|-------------|
| Architecture | Multi-stage SIMD pipeline; stages configured at compile time |
| Parallelism | Lane parallelism (SIMD across multiple data elements) + pipeline parallelism (stages) |
| Reconfigurability | Fully reconfigurable per model (via PEF load) |
| Operation granularity | One innermost-parallel loop body per PCU |
| Runtime scheduling | None — PCU executes its fixed assignment continuously |

### SN40L Compute Specifications

| Spec | Value |
|------|-------|
| PCU count per socket | 1040 |
| Peak BF16 TFLOPS per socket | 638 |
| Peak FP16 TFLOPS per socket | ~688 |
| Transistors | ~102 billion |
| Process | TSMC 5 nm |
| Die configuration | Dual-die (two logic dies per package) |
| SambaRack aggregate BF16 | 10.2 PFLOPS (16 sockets) |

### SN50 Compute Specifications

| Spec | Value | Sourcing (2026-08-08) |
|------|-------|-----------------------|
| Process | TSMC 3 nm (N3) | Vendor-stated |
| Die configuration | Dual-chiplet | Vendor-stated |
| PCU count per socket | ~2× SN40L | *Extrapolated* — no primary source |
| PMU count per socket | not disclosed | — |
| BF16 PFLOPS per socket | 1.6 | *Extrapolated* — no primary source |
| FP8 PFLOPS per socket | 3.2 | *Extrapolated* — no primary source |
| FP8 improvement vs SN40L | 5x | Consistent with vendor "5X faster", but the absolute PFLOPS it is derived from are extrapolated |
| BF16 improvement vs SN40L | 2.5x | *Extrapolated* |
| Native FP8 | Yes (new in SN50) | *Extrapolated* — implied by vendor FP8 messaging, not stated in the press release |
| Rack power | 20 kW per SambaRack | Vendor-stated |
| Status | Announced 2026-02-24; GA targeted H2 2026; **no customer delivery confirmed as of 2026-08-08** | 2026-02-24 press release ("will start shipping to customers later this year"); 2026-04-08 Intel release (H2 2026 availability) |
| Demonstrated deployment | Single 16-RDU SambaRack SN50, benchmarked 2026-07-08 and 2026-07-30 | Vendor blog posts |
| First architecture disclosure | **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public** (Session AI 2, 2026-08-25, "Dataflow at Scale: the SN50 RDU", Raghu Prabhakar) | Hot Chips advance program |

---

## 2. Data Path

### Spatial Dataflow Execution

The RDU's defining data path characteristic:

1. **Compile-time assignment**: each PCU receives its operation assignment when the PEF loads
2. **Direct streaming**: output from one PCU flows directly to the next on the vector switching network — no DRAM round-trips
3. **Operator fusion**: consecutive operations (Linear → LayerNorm → GELU → Linear) are fused into a section that streams entirely through on-chip SRAM
4. **No dynamic scheduling**: the switching fabric routes are predetermined; no arbitration needed at runtime

This contrasts with GPU execution:
- GPU: each kernel writes to HBM; next kernel reads from HBM; the DRAM bandwidth is a bottleneck between every operator
- RDU: operators within a section pass data through the switching fabric; HBM is accessed only at section boundaries

### AGCU (Address Generation and Coalescing Unit)

The AGCU is the RDU's memory controller and external interface:

| Function | Description |
|----------|-------------|
| Address generation | Computes off-chip memory addresses for streaming patterns |
| Request coalescing | Batches multiple RDU requests into efficient DRAM bursts |
| Sparse access | Handles irregular access patterns for sparse tensors |
| P2P inter-RDU | Peer-to-peer DMA to/from other RDU sockets |
| Host interface | Control-plane connection to host CPU |

---

## 3. On-chip Memory

### Pattern Memory Unit (PMU)

The PMU is a **software-managed distributed scratchpad SRAM** — not a hardware cache:

| Spec | SN40L | SN50 |
|------|-------|------|
| PMU count per socket | 1040 | not disclosed |
| Total SRAM capacity | 520 MiB | 432 MiB *(extrapolated)* |
| Aggregate bandwidth | Hundreds of TB/s | not disclosed |
| Management | Compiler-managed (static placement) | Compiler-managed |
| Data placement | Fixed at PEF compile time | Fixed at PEF compile time |
| Bank-level parallelism | High (within and across PMUs) | High |

PMUs are **physically interleaved with PCUs** across the tile array, minimizing data movement distances.

Comparison vs GPU on-chip memory:

| Feature | SambaNova PMU (SN40L) | NVIDIA H100 SMEM+L1 |
|---------|----------------------|---------------------|
| Total capacity | 520 MiB | ~30 MB (228 KB/SM × 132 SMs) |
| Management | Software (compiler) | Mixed (HW cache + programmer SMEM) |
| Placement control | Static (compile time) | Dynamic (programmer allocates SMEM; L1 is HW-managed) |
| Bandwidth | Hundreds of TB/s | ~33 TB/s (SMEM aggregate) |

### 3D Switching Fabric

Three parallel on-chip networks connecting all PCUs and PMUs:

| Network | Granularity | Traffic |
|---------|-------------|---------|
| Scalar network | Word (32-bit) | Loop counters, scalars, control values |
| Vector network | Multi-word | Bulk tensor data (activations, weights) |
| Control network | Bit | Predicates, enables, synchronization |

All routes are **statically determined at compile time**. No dynamic routing decisions, no arbitration — the compiler owns the full switching fabric configuration.

---

## 4. Off-chip Memory

### Three-Tier Memory Architecture

The SN40L's key architectural innovation: **all three memory tiers are directly attached to the RDU** (not the host CPU).

| Tier | SN40L Capacity | SN40L BW | SN50 Capacity | SN50 BW | Role |
|------|----------------|----------|---------------|---------|------|
| On-chip SRAM (PMU) | 520 MiB | Hundreds of TB/s | 432 MiB *(extrapolated)* | not disclosed | Active compute data, intermediate tensors |
| Co-packaged HBM | 64 GiB | ~1 TB/s | 64 GiB *(extrapolated)* | 1.8 TB/s *(extrapolated)* | Software-managed caching tier (HBM2E on SN50 — *extrapolated*) |
| Directly-attached DDR | up to 1.5 TiB | — | 256 GiB – 2 TiB *(extrapolated)* | not disclosed | Model weight reservoir (DDR4/5 on SN40L; DDR5 on SN50) |

**Decode-era tier roles (2026-04-16 vendor framing, added 2026-08-08).** SambaNova's "The Decode Era of AI" post assigns each tier an explicit role in disaggregated decode serving, which is a change of *argument* rather than of hardware:

| Tier | SN40L-era pitch | Decode-era pitch (2026) |
|---|---|---|
| SRAM (PMU) | Fused-section working set | The hottest local working set |
| HBM | Staging buffer between DDR and SRAM | **Model weights + KV cache** |
| DDR | Reservoir for 1T-param Composition-of-Experts weights | **Prompt caching and multi-model / agentic workflows** |

Data movement pattern during inference:

```
DDR (full model weights)
  → HBM (active model weights for current layer range, software-staged by compiler)
    → PMU SRAM (active tile, fed to PCUs)
      → PCU compute
    → PMU SRAM (output tile)
  → HBM (next stage)
→ DDR (KV cache / persistent state, if any)
```

### Why the DDR Tier Matters

GPU clusters lack a DDR tier attached to the accelerator:
- NVIDIA H100: 80 GiB HBM only; DDR lives on host CPU across PCIe
- To serve a 70B model (BF16 = ~140 GiB), you need 2+ H100s for weights alone
- Samba-CoE (150 × 7B = ~2.1 TiB total): requires many GPU nodes; fits on one SambaRack

### HBM as Software-Managed Cache

Unlike GPU HBM (which is the only "fast" off-chip memory and holds both model weights and KV cache):
- SambaNova HBM acts as a **staging buffer between DDR and SRAM**
- The SambaFlow compiler determines what lives in HBM vs DDR at any moment
- This allows models far larger than HBM capacity to run efficiently

---

## 5. Package and Die Configuration

### SN40L Package

| Component | Details |
|-----------|---------|
| Logic dies | 2 (dual-die) |
| HBM stacks | Co-packaged on interposer |
| DDR | Pluggable DIMMs directly attached to board (not host) |
| Process | TSMC 5 nm |
| Transistors | ~102 billion |
| Interconnect to host | PCIe |

### SN50 Package

| Component | Details | Sourcing (2026-08-08) |
|-----------|---------|-----------------------|
| Logic dies | Dual-chiplet | Vendor-stated |
| HBM | HBM2E, 64 GiB, 1.8 TB/s | *Extrapolated* — no primary source |
| DDR | DDR5, 256 GiB – 2 TiB, directly attached | *Extrapolated* — no primary source |
| Process | TSMC 3 nm (N3) | Vendor-stated |
| Transistors | not disclosed | — |
| Host interface | not disclosed for SN50 | — |

---

## 6. Scale-up Interconnect

### SN40L: RDU-Connect + InfiniBand

| Spec | Value |
|------|-------|
| RDU sockets per SambaRack | 16 |
| Intra-rack protocol | P2P (RDU-Connect) |
| Inter-rack protocol | InfiniBand |
| Collective support | AllReduce, AllGather via P2P primitives |
| Aggregate BF16 (SambaRack) | 10.2 PFLOPS |
| Aggregate on-chip SRAM | 16 × 520 MiB = ~8 GiB |
| Aggregate HBM | 16 × 64 GiB = 1 TiB |
| Aggregate DDR | 16 × 1.5 TiB = 24 TiB |

### SN50: Switched Fabric

| Spec | Value | Sourcing (2026-08-08) |
|------|-------|-----------------------|
| **RDUs demonstrated** | **16 (a single SambaRack SN50)** | Vendor blog posts 2026-07-08 and 2026-07-30 |
| Max RDU chips | 256 | **Vendor claim only.** SambaNova's own 2026-07-30 post frames 64- and 256-chip configurations as future: "As SN50 scales to 64 and 256 chips…" |
| Chip-to-chip bandwidth | 2.2 TB/s bidirectional per RDU | *Extrapolated.* Vendor material states only "multi-terabyte-per-second interconnect" |
| Improvement vs SN40L | 4x network bandwidth | Vendor-stated |
| Fabric type | Switched (non-blocking) | Vendor-stated ("switched"); "non-blocking" is *extrapolated* |
| Max model parameters | 10T+ | Vendor claim |
| Max context length | 10M+ tokens | Vendor claim |
| Max rack span | Multiple racks | Vendor claim; not demonstrated |
| Per-rack power | 20 kW | Vendor-stated |

### SN50 Deployment Configurations Actually Demonstrated (2026-07)

Both public SN50 demonstrations used one 16-RDU SambaRack. Notably, the chip-parallelism strategy is selected at **deployment time**, not fixed by the compiler:

| Configuration | Parallelism across the 16 RDUs | Reported result (MiniMax M2.7) |
|---|---|---|
| Highest interactivity | **TP16** (tensor parallel across all 16 RDUs) | ≈800 output tokens/sec |
| More concurrent users | **TP8 + DP8** (tensor parallel across 8, data parallel across the other 8) | ≈400 output tokens/sec |

These are **vendor-reported** figures from a pre-official run (see the summary's Update §6 for the full provenance caveat, including that SemiAnalysis has published nothing on SambaNova through its own channels and that SN50 does not appear in the InferenceX dataset). The independently measured public SambaCloud endpoint delivers **392.4 output tokens/sec** on the same model (Artificial Analysis, 2026-08-08).

### System-Level: Disaggregated Prefill/Decode (2026-04-08, announced with Intel)

Not a change to the RDU die, but a change to the system the RDU is designed to sit in. The SambaNova–Intel "Blueprint for Heterogeneous Inference" assigns:

| Stage | Hardware | Role |
|---|---|---|
| Prefill | GPUs | Prompt ingestion → KV cache generation |
| Decode | **SN50 RDUs** | "The dedicated inference fabric for high-throughput, low-latency decode" |
| Agentic tools | Intel Xeon 6 | Host plus "action CPU": task coordination, tool/API execution |

Availability stated as H2 2026. The 2026-07-08 demonstration is the concrete instance: one NVIDIA H200 rack (4 GPUs) for prefill paired with one 16-RDU SambaRack SN50 for decode. This is the same structural pattern as Cerebras+AMD and NVIDIA+Groq-LPX.

---

## 7. Scale-out Interconnect

| Protocol | Generation | Usage |
|----------|-----------|-------|
| InfiniBand | Standard (EDR/HDR/NDR) | SN40L rack-to-rack |
| SN50 switched fabric | Proprietary high-speed | SN50 multi-rack scale-up |

SambaNova's approach: the switched fabric handles scale-up (RDU-to-RDU) natively; InfiniBand is used for scale-out to broader data center infrastructure.

---

## Sources

- [SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts (arXiv 2405.07518)](https://arxiv.org/html/2405.07518v1)
- [Accelerated Computing with a Reconfigurable Dataflow Architecture (Whitepaper)](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [SN40L RDU Product Page](https://sambanova.ai/products/rdu-ai-chips)
- [Introducing the SN50 RDU](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference)
- [SambaRack Datasheet](https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf)
- [Evaluating Emerging AI/ML Accelerators: IPU, RDU, and NVIDIA/AMD GPUs](https://arxiv.org/html/2311.04417v3)
- [SambaNova Pits Its Engineering Against Nvidia For Agentic AI (NextPlatform)](https://www.nextplatform.com/ai/2026/02/25/sambanova-pits-its-engineering-against-nvidia-for-agentic-ai/4092613)

### Added 2026-08-08

- [SambaNova + Intel: Blueprint for Heterogeneous Inference (2026-04-08)](https://sambanova.ai/press/sambanova-announces-collaboration-with-intel-on-ai-solution)
- [The Decode Era of AI: Why Dataflow Matters More Than Ever (2026-04-16)](https://sambanova.ai/blog/why-dataflow-matters-more-than-ever)
- [SN50 runs fastest MiniMax speeds in the world (2026-07-08)](https://sambanova.ai/blog/sn50-runs-fastest-minimax-speeds-in-the-world)
- [SemiAnalysis benchmarks SambaRack SN50 on MiniMax M2.7 (2026-07-30)](https://sambanova.ai/blog/semianalysis-benchmarks-sambarack-sn50-with-fast-inference-on-minimax-m2.7) — vendor-reported; see provenance caveat
- [SN50 unveiling press release (2026-02-24)](https://sambanova.ai/press/sambanova-unveils-fastest-chip-for-agentic-ai-collaborates-with-intel-and-raises-350m)
- [Artificial Analysis — MiniMax-M2.7 provider comparison](https://artificialanalysis.ai/models/minimax-m2-7/providers) — independent production measurement, 392.4 output t/s
- [SambaStack release notes](https://docs.sambanova.ai/docs/en/release-notes/sambastack.md) — SN40L-16 remains the documented on-prem hardware reference; no SN50 rack documentation published
- Hot Chips 38 advance program (https://hotchips.org/advance-program/) — "Dataflow at Scale: the SN50 RDU", Session AI 2, 2026-08-25. **Disclosure scheduled; content not yet public. Not a spec source.**
