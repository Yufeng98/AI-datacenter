# Groq LPU (TSP) Software and Hardware Stack Summary

*as_of: 2026-04-05 · **Last updated: 2026-08-08*** — see "Corporate Status & LP30 Corrections Update (2026-08-08)" below.

---

## Overview

Groq's Language Processing Unit (LPU), originally named the Tensor Streaming Processor (TSP), is the most architecturally distinctive inference accelerator in the AI chip landscape. Its core thesis: **eliminate all dynamic hardware in favor of a compiler that knows everything at compile time**. No caches. No branch predictors. No out-of-order execution. No DRAM. Every operation is assigned to a specific physical unit at a specific clock cycle by the compiler — making the hardware a pure deterministic executor of a pre-computed plan.

The flagship system is the **GroqRack**: 576 LPU chips interconnected via a plesiosynchronous fabric, appearing to software as a single coherent ~130 GB SRAM memory. A single chip holds 230 MB SRAM at 80 TB/s — roughly 24x the bandwidth of H100 HBM3. (The "640 TB/s aggregate" GroqRack figure carried in earlier revisions of this survey is **unverified** — see the 2026-08-08 update.)

On **2025-12-24 Groq and NVIDIA entered a non-exclusive inference-technology licensing agreement**. This was **not an acquisition**: Groq continues to operate as an independent company and GroqCloud continues without interruption, while founder/CEO Jonathan Ross, president Sunny Madra and other employees joined NVIDIA to advance the licensed technology. The resulting **NVIDIA Groq 3 LPX** (LP30 chip; 500 MB SRAM/die at 150 TB/s) was announced at GTC on 2026-03-16 as a decode-phase co-processor in NVIDIA's Vera Rubin platform; NVIDIA guides Vera Rubin partner availability to H2 2026 and has published no LPX-specific ship date. Full corrections — deal structure, the unconfirmed ~$20B price, the LP30 spec fixes, the prefill/decode split, and Groq's June 2026 $650M raise — are in the dated update section below.

---

## Software Stack

### Framework Integration

Groq supports PyTorch as the primary framework via **GroqFlow** (open source, Apache 2.0, GitHub: groq/groqflow). The interface is a single function call:

```python
from groqflow import groqit
gmodel = groqit(my_pytorch_model, inputs)  # compiles model to .iop binary
result = gmodel(inputs)                    # runs on LPU hardware
```

ONNX is a first-class alternative entry. TensorFlow and CoreML are supported via ONNX export. There is no eager-mode PyTorch backend (no `PrivateUse1`) — all execution is ahead-of-time compiled.

The **groq-python** cloud SDK provides an OpenAI-compatible REST client for GroqCloud inference:

```python
from groq import Groq
client = Groq()
response = client.chat.completions.create(model="llama-3.3-70b-versatile", ...)
```

GroqCloud delivered 750-900 tokens/sec on Llama 3.3 70B in 2025, and over 1,660 tokens/sec with Speculative Decoding on Llama 3 70B — consistently the fastest public inference API.

### Compiler / IR

The compiler is the most critical component of the Groq stack. It implements static spatial scheduling across three stages:

1. **Frontend**: torch-MLIR or ONNX-MLIR converts PyTorch/ONNX models into standard MLIR IR
2. **GTen lowering**: Lowers MLIR to Groq's GTen dialect — a small set of DL/HPC ops that map directly to LPU functional units
3. **Spatial scheduling**: The Groq Compiler assigns every GTen op to:
   - A specific functional unit (Matrix Mult Unit or Vector ALU) in the 2D spatial grid
   - A specific clock cycle (cycle-exact)
   - A specific SRAM bank address (for weights and activations)
   - For multi-chip: specific inter-chip packet delivery cycles

Output is an `.iop` (Instruction Operation Program) binary. No custom kernels are written — the compiler handles all fusion automatically, unlike CUTLASS (NVIDIA) or TBE-TIK (Ascend) which require hand-tuned kernel variants.

Groq contributed to the upstream MLIR ecosystem: torch-mlir and ONNX-MLIR (bxing-groq GitHub). The Hot Chips 34 (2022) paper describes the second-generation compiler that schedules not just on-chip but inter-chip packet delivery cycles across the 576-chip GroqRack.

### Op Library

Not applicable. The Groq Compiler handles all operator fusion on GTen dialect ops. There is no cuDNN/MIOpen equivalent — no separately versioned op dispatch library exists.

### Kernel Library

Not applicable. No user-facing kernel library (no CUTLASS/CK analog). The hardware's deterministic execution model makes per-kernel tuning unnecessary; the compiler performs all placement and scheduling globally.

### Runtime

**groq-runtime** is minimal by design:
- Loads `.iop` binary onto the chip
- Performs one-time SRAM weight loading at startup
- Executes inference by replaying the pre-compiled instruction stream exactly

There is no dynamic scheduler, no stream/event API, no CUDA-Graph equivalent — because all of that complexity is encoded statically in the `.iop` binary.

The GroqWare SDK bundles both `groq-devtools` (the compiler) and `groq-runtime` (driver + runtime library).

### Driver / Firmware

PCIe Linux kernel module: BAR mapping, command queue management, interrupt handling. No on-chip firmware resource manager (unlike NVIDIA's GSP). The LPU executes its pre-compiled instruction stream without dynamic management.

### Communication

No NCCL equivalent. All inter-chip communication within GroqRack is **pre-scheduled by the compiler** — packet delivery times are assigned at compile time over the plesiosynchronous fabric. For GroqCloud, clients communicate via HTTPS REST.

### Assembler / ISA

No user-visible virtual ISA (no PTX equivalent). The `.iop` binary format is proprietary. A bare-metal assembler is available for research use. The ISCA 2020 paper describes the TSP's VLIW-like instruction format at a high level.

---

## Hardware Architecture

### Compute Engine

The LPU contains two types of compute units:

- **Matrix Multiplication Unit**: 320x320 fused dot product array. Handles GEMM — the dominant operation in transformer inference. Delivers 750 INT8 TOPS / 188 FP16 TFLOPS (LPU v1, 14nm).
- **Vector ALUs**: 5,120 vector ALUs for activation functions (GeLU, SiLU), layer norm, element-wise operations.

There are no SIMT warps, no warp schedulers, no out-of-order buffers, no branch predictors. The compute units execute exactly what the compiler scheduled.

### Data Path: Tensor Streaming

The TSP's defining data movement pattern is **tensor streaming**: data flows through a conveyor-belt pipeline of functional units arranged in a 2D spatial grid. The compiler maps tensor operations to physical routes across this grid. Activations never idle — they stream from one unit to the next per the pre-computed schedule. This eliminates the cache-miss stalls and scheduling overhead that plague GPU inference.

### On-chip Memory (All-SRAM)

| Memory | LPU v1 | NVIDIA H100 (for comparison) |
|--------|--------|-------------------------------|
| Type | On-chip SRAM | HBM3 |
| Capacity | ~230 MB | 80 GB |
| Bandwidth | 80 TB/s | 3.35 TB/s |
| BW ratio | **24x faster** | — |

The SRAM is not a cache — it is the primary weight storage and activation buffer. The compiler places weights into specific SRAM banks and schedules reads/writes to exact cycles. There are no cache misses because there is no cache.

### Off-chip Memory

A single LPU v1 die has **no DRAM**. For large models:
- Weights are striped across multiple chips in GroqRack
- Each chip holds a model shard in its 230 MB SRAM
- 576-chip GroqRack: ~130 GB total SRAM (576 x 230 MB)

> **The "no DRAM anywhere" absolute no longer holds at rack level for the LPX generation.** NVIDIA's LPX product page lists **12 TB of DDR5 per 256-LPU rack**. The LP30 die itself still has no attached DRAM; the DDR5 is a rack-level tier. See the 2026-08-08 update.

### Host Interface / Package

- LPU v1: Samsung 14nm, 25x29 mm (~725 mm²), 900 MHz
- LPU v2: Samsung 4nm (2025)
- NVIDIA Groq 3 LPX (LP30): **process node not disclosed** (the "Samsung 4nm" attribution is *not confirmed* — it appears only in low-quality secondary blogs); **500 MB** SRAM/die, 150 TB/s, 1.2 PFLOPS FP8/chip; TDP not disclosed; announced GTC 2026-03-16, H2 2026 partner availability guidance
- PCIe host interface

### Scale-up Interconnect (GroqRack)

| Parameter | Value |
|-----------|-------|
| Protocol | Plesiosynchronous (near-synchronous, known drift) |
| Topology | Direct mesh / dragonfly variant (no external switches) |
| Chips | 576 per GroqRack |
| Total SRAM | ~130 GB |
| Chip-to-chip BW | **unverified** (earlier revisions said "~640 TB/s aggregate"; no Groq primary source found — 640 TB/s is exactly NVIDIA's published *LPX* rack scale-up figure, so this is probably a conflation) |
| Scheduling | Compiler-assigned per packet delivery cycle |
| Appearance to software | Single coherent memory |

The plesiosynchronous protocol periodically synchronizes chip clocks via software to cancel crystal-based drift, enabling cycle-exact scheduling across hundreds of chips without a central clock distribution tree.

### Scale-out Interconnect

GroqRack systems connect to datacenter networks via standard Ethernet/InfiniBand host NICs. No proprietary scale-out fabric.

---

## Programming Model Rationale

The Groq LPU's design philosophy is a deliberate inversion of the GPU design philosophy:

**1. The compiler is the scheduler, not the hardware.** GPUs use hardware warps, schedulers, and out-of-order execution to hide latency dynamically. The LPU eliminates all of this and requires the compiler to produce a perfect schedule. This trades flexibility for predictability — and for inference workloads with fixed model shapes, it works extremely well.

**2. SRAM bandwidth > DRAM bandwidth.** Transformer token generation is memory-bandwidth-bound (weight load per token, low arithmetic intensity at batch=1). At 80 TB/s SRAM bandwidth vs 3.35 TB/s HBM3, the LPU achieves ~24x higher weight-read bandwidth per chip — directly translating to ~24x higher token throughput at batch=1.

**3. Determinism eliminates jitter.** Every LPU inference takes exactly the same time. There are no cache cold-start penalties, no scheduler stalls, no interrupt latency. For latency-SLA inference APIs (like GroqCloud), this is critical.

**4. No kernel authoring means faster deployment.** GPU users spend significant effort writing CUTLASS kernels, selecting cuDNN engines, and tuning TensorRT. Groq's compiler handles all of this automatically — `groqit(model, inputs)` is genuinely one line.

**5. Plesiosynchronous fabric enables rack-scale coherence.** Rather than building complex distributed memory protocols, Groq's compiler knows exactly when each chip will send and receive each packet. The plesiosynchronous protocol periodically corrects crystal drift — simple software, not complex hardware.

**6. Weakness: training and dynamic shapes.** SRAM-only storage limits capacity for large activation tensors (needed for backprop). Static scheduling requires fixed input shapes — dynamic batching requires recompilation. Very large models exceeding GroqRack SRAM capacity (~130 GB for 576 chips) cannot run.

---

## Corporate Status & LP30 Corrections Update (2026-08-08)

*Updated 2026-08-08. Primary sources: groq.com/newsroom (2025-12-18, 2025-12-24, 2026-06-22), groq.com/about-us, NVIDIA developer blog 2026-03-16, nvidia.com/en-us/data-center/lpx, nvidianews Vera Rubin platform release, hotchips.org/advance-program. Secondary: CNBC/Reuters/TechCrunch/EE Times 2025-12-24, Bloomberg & DataCenterDynamics 2026-06-22, StorageReview, CRN.*

This section corrects material errors carried in earlier revisions of this file and adds developments through 2026-08-08. Prior-generation content above is retained deliberately; this section supersedes it where they conflict.

### 1. NVIDIA did NOT acquire Groq — it licensed the technology

| Claim in earlier revisions | Corrected (2026-08-08) |
|---|---|
| "NVIDIA acquired Groq" | **Non-exclusive licensing agreement**, announced 2025-12-24. Groq's own release: it "has entered into a non-exclusive licensing agreement with Nvidia for Groq's inference technology," Groq "will continue to operate as an independent company," and "GroqCloud will continue to operate without interruption." |
| "for ~$20 billion" | **Reported at ~$20B by CNBC (2025-12-24)** and carried by Reuters, TechCrunch, EE Times and IBD. **Explicitly not confirmed by NVIDIA or Groq**; Reuters notes "Groq did not disclose financial details of the deal." No SEC filing disclosing a value was found. Treat as well-sourced journalism, not a vendor figure. |

Structurally the deal is **a non-exclusive IP license plus a large acqui-hire**: founder/CEO Jonathan Ross, president Sunny Madra and other employees left to join NVIDIA "to help advance and scale the licensed technology."

**Leadership as of 2026-08-08.** Simon Edwards (previously CFO) became CEO at the time of the December 2025 transaction but **left Groq in April 2026** for Bloom Energy. Groq's own About Us page now lists **Adam Winter as CEO** ("became CEO in 2026"; several outlets describe him as interim), with Matt Eng (CFO), Alan Rice (COO), Sinclair Schuller (CTO) and Rakesh Malhotra (CPO). Alex Davis is named as Chairman in coverage of the June 2026 raise. (Wikipedia's Groq infobox still lists Jonathan Ross as CEO and is stale.)

### 2. Groq raised $650M (2026-06-22) — the pivot to inference-cloud operator

| Item | Value |
|---|---|
| Amount | $650M growth capital |
| Lead investors | Disruptive and Infinitum, with participation from existing investors |
| Valuation | **Not disclosed** (prior round reportedly ~$6.9B) |
| Footprint | 13 data centers across North America, Europe, the Middle East and APAC |
| Developers | "more than five million" |
| Capacity target | "expects to scale toward 200 MW by the end of 2027" |

Materially, **Groq has pivoted from chip designer to inference-cloud operator**. Its own release says it will fit out that footprint "with Groq's latest inference technology, including the new LPX system from NVIDIA" — i.e. Groq is now a **buyer of NVIDIA LPX silicon derived from its own licensed IP**, not solely the originator of it. Corroborated by Bloomberg and DataCenterDynamics (both 2026-06-22).

Adjacent context: Groq also announced a **U.S. Department of Energy partnership on 2025-12-18** ("to Advance AI Inference and Next-Generation Computing Infrastructure"), six days before the NVIDIA agreement.

### 3. NVIDIA Groq 3 LPX / LP30 — corrected specifications

Announced at **GTC 2026 (2026-03-16)**. All figures below are from NVIDIA's developer blog and the nvidia.com LPX product page.

> **Naming caveat (checked 2026-08-08).** NVIDIA's Rubin product page says only *"NVIDIA Groq 3 LPX is the inference accelerator for NVIDIA Vera Rubin"* — the strings **"Groq 3 LPU" and "LP30" do not appear there**. `LP30` is used in this repo as the die designation because it appears in NVIDIA's developer-blog coverage, but the claim that LPU (chip) and LPX (rack) are formally distinct product names is **not confirmed by any NVIDIA primary source**. NVIDIA itself uses "LPX" for both the accelerator and the 256-unit rack, and "LPU" generically for the SRAM-based accelerator class.

| Parameter | Corrected value | Note |
|---|---|---|
| On-chip SRAM per LP30 | **500 MB** | earlier revisions said 512 MB — wrong; 500 MB is the figure NVIDIA publishes |
| SRAM bandwidth per LPU | 150 TB/s | unchanged |
| Chip-to-chip links | **96 links @ 112 Gbps** | new |
| Aggregate C2C BW per chip | **2.5 TB/s bidirectional** | new |
| FP8 compute per chip | **1.2 PFLOPS** | new |
| FP8 compute per tray | **9.6 PFLOPS** (8 LPUs) | new |
| Rack composition | **256 LPUs in 32 liquid-cooled 1U trays of 8** | new |
| Rack SRAM | **128 GB total, 40 PB/s aggregate** | 40 PB/s already present |
| Rack scale-up BW | **640 TB/s** | new |
| Rack DDR5 | **12 TB per rack** | new — see the off-chip-memory caveat above |
| Process node | **Not disclosed** | earlier "Samsung 4nm" is *not confirmed*; appears only in low-quality secondary blogs |
| TDP | **Not disclosed** | |
| Non-FP8 datatype throughput | **Not disclosed** | |
| Efficiency claim | "up to 35x higher inference throughput per megawatt" | **vendor marketing claim, unaudited** |

> **Arithmetic caveat in NVIDIA's own material:** the developer blog states 315 PFLOPS of AI inference compute per LPX rack, but 9.6 PFLOPS FP8/tray × 32 trays = 307.2 PFLOPS. The discrepancy is unexplained; do not present 315 PFLOPS as a dense FP8 figure without this caveat.

**Availability — downgraded.** Earlier revisions said LPX "ships in Q3 2026." NVIDIA has published **no LPX-specific ship date**. NVIDIA's Vera Rubin platform release states Vera Rubin-based products "will be available from partners starting the second half of this year," and secondary coverage (StorageReview, CRN) describes LPX as available in H2 2026. Correct status verb: **ANNOUNCED** (GTC, March 2026), with **H2 2026 partner availability guidance**. "Q3 2026" was an analyst inference. There is no evidence of sampling, shipping, or at-scale deployment as of 2026-08-08.

### 4. Corrected prefill/decode division of labour

Earlier revisions said "Rubin GPUs handle prefill, Groq LPUs handle token generation." **This misstates NVIDIA's own description.** Per the NVIDIA developer blog, the split is *within decode*:

- **Rubin GPUs** take the **throughput-bound** decode work — notably full-context attention over the accumulated KV cache.
- **Groq 3 LPX** accelerates the **latency-sensitive** execution within decode — notably sparse MoE expert feed-forward networks.

### 5. No new Groq silicon, April–August 2026

Groq published exactly two items in 2026: "GroqCloud: Expanding to Meet Demand" (2026-02-16) and the 2026-06-22 fundraise. **No LPU v3 / next-generation TSP was announced by Groq in the window.**

### 6. Forward-looking: Hot Chips 38 (disclosure scheduled, not yet public)

Hot Chips 38 runs **Aug 23–25, 2026** at Memorial Auditorium, Stanford. **Session AI 1, Tuesday 2026-08-25, 2:15–4:15 PM PDT: "Think Fast: LPU Accelerator for Heterogeneous Compute" — Igor Arsovski & Santosh Raghavan, NVIDIA.** This is the first Hot Chips LPU disclosure under NVIDIA branding. *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public* as of 2026-08-08. Re-scan after Aug 25 for process node, TDP and per-datatype throughput, none of which are currently disclosed.

---

## Resources

### Documentation
- [Groq LPU Architecture](https://groq.com/lpu-architecture)
- [Inside the LPU: Deconstructing Groq's Speed](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- [What is a Language Processing Unit?](https://groq.com/blog/the-groq-lpu-explained)
- [GroqCloud API Docs](https://console.groq.com/docs/overview)
- [Groq Compiler Solutions](https://groq.com/meet-groq-compiler-solutions-for-a-symbiotic-software-hardware-ecosystem/)
- [NVIDIA Groq 3 LPX Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) (2026-03-16)
- [NVIDIA LPX product page](https://www.nvidia.com/en-us/data-center/lpx/)
- [NVIDIA Vera Rubin platform release](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform)

### Corporate / Status (added 2026-08-08)
- [Groq–NVIDIA non-exclusive inference technology licensing agreement (2025-12-24)](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale)
- [Groq raises $650M to scale its AI inference cloud business (2026-06-22)](https://groq.com/newsroom/groq-raises-usd650m-to-scale-its-ai-inference-cloud-business)
- [Groq About Us — current leadership](https://groq.com/about-us)
- [Groq newsroom index](https://groq.com/newsroom)
- [Hot Chips 38 advance program](https://hotchips.org/advance-program/) — Session AI 1, 2026-08-25, "Think Fast: LPU Accelerator for Heterogeneous Compute" (NVIDIA); *not yet presented*

### Academic Papers
- [ISCA 2020: Think Fast — A Tensor Streaming Processor (TSP)](http://pkamath.com/publications/papers/tsp-isca20.pdf)
- [ISCA 2022: A Software-Defined Tensor Streaming Multiprocessor](https://dl.acm.org/doi/10.1145/3470496.3527405)
- [Hot Chips 34 (2022): Software-Defined Scale-out TSP](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf)

### Open-Source Repositories
- [GroqFlow](https://github.com/groq/groqflow) — Automated compilation pipeline
- [groq-python](https://github.com/groq/groq-python) — Cloud SDK
- [bxing-groq/onnx-mlir](https://github.com/bxing-groq/onnx-mlir) — Groq's ONNX-MLIR fork
