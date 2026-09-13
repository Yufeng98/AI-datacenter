# Tesla Dojo — Software Stack Investigation

*as_of: 2026-08-08*
*Sources: Hot Chips 34 (HC34, August 2022), Tesla job postings (archived), tech press; roadmap addendum 2026-08-08 (see final section)*

---

## Investigation Summary

The Dojo software stack is the least publicly documented aspect of the system. Tesla never released a public SDK, compiler source, ISA specification, or kernel library. What is known comes from:

1. HC34 slides and presentation (August 2022) — disclosed framework integration and compiler strategy at a high level
2. Tesla job postings (archived) — confirmed LLVM backend, PyTorch frontend, and ML compiler team composition
3. Tech press analysis (The Next Platform, Chips & Cheese) — interpretive coverage of HC34 content
4. The `teslamotors/ttpoe` GitHub repository — the only open-source software artifact

This document summarizes what is confirmed, what is inferred, and what is not public.

---

## 1. Framework Integration

### PyTorch Frontend

Tesla uses **PyTorch** as the user-facing interface for Dojo training workloads. The goal, as stated at HC34, was to allow existing PyTorch-trained models to run on Dojo with minimal code changes — the user writes standard PyTorch, and the Tesla compiler handles the rest.

| Feature | Status |
|---------|--------|
| PyTorch model execution | confirmed (HC34) |
| torch.compile / dynamo integration | not confirmed; predates torch.compile release |
| Autograd support | inferred (PyTorch frontend implies gradient computation) |
| Manual C/C++ kernel authoring required | No — explicit goal to avoid this (HC34) |
| Public PyTorch backend repository | Not public |

### No JAX / TensorFlow Support Mentioned

HC34 presentations focused exclusively on PyTorch. There is no public evidence of TensorFlow or JAX integration.

---

## 2. Compiler / IR

### Custom ML Compiler with LLVM Backend

Tesla built a custom end-to-end ML compiler. The pipeline is:

```
PyTorch computation graph
        |
   Graph capture / tracing
        |
   Tesla ML Compiler (IR not public)
        |
   Code generation (LLVM backend)
        |
   Dojo custom ISA (binary)
        |
   D1 hardware execution
```

| Feature | Status |
|---------|--------|
| Frontend | PyTorch computation graph capture | confirmed |
| Intermediate representation (IR) | custom; not public | confirmed existence |
| Backend | LLVM | confirmed (HC34, job postings) |
| Parallelism strategies | Data, model, graph parallelism — all handled automatically | confirmed |
| On-the-fly compilation | Yes — generates code on first execution; reuses on subsequent | confirmed |
| Loop-level control flow | Supported | confirmed |
| Mixed-precision handling | BF16 / CFP8 / CFloat16 selection | confirmed |
| Source code public | No | confirmed |
| MLIR-based | Unknown; not stated | unknown |

### Key Design Philosophy

- No user-facing kernel authoring API (no equivalent of CUDA C++ or Triton)
- The compiler handles data layout, tiling, scheduling, and inter-node data movement
- Software-managed SRAM placement means the compiler must explicitly schedule all on-chip memory accesses
- "Automatic partitioning" cited in HC34: the compiler tiles the model across the 354-node mesh without user directives

### Kernel / Op Library

There is no public evidence of a standalone kernel library (equivalent to cuDNN or cuBLAS). All op implementations are believed to be internal to the compiler's code generation path or are compiled on-demand.

| Feature | Status |
|---------|--------|
| GEMM library | not public |
| Convolution library | not public |
| Attention / transformer ops | not public |
| Normalization ops | not public |

---

## 3. ISA

| Feature | Value |
|---------|-------|
| ISA type | Custom (Tesla proprietary) |
| Scalar word width | 64-bit |
| Vector width | 64 bytes (512 bits) |
| Supported precisions | BF16, CFP8 (custom 8-bit float), CFloat16, FP32 |
| ISA documentation | Not public |
| Assembler | Not public |
| Debug tools | Not public |

---

## 4. Runtime

### Execution Model

- The runtime manages distribution of compiled compute graphs across the mesh
- Fault-tolerant: reroutes mesh traffic around failed links without halting training
- Auto-scales dynamically across tiles in an ExaPOD
- No virtual memory in the hardware; the runtime coordinates SRAM allocation across nodes

| Feature | Status |
|---------|--------|
| Fault tolerance (link rerouting) | confirmed (HC34) |
| Dynamic scaling across tiles | confirmed |
| Virtual memory support | No — explicitly absent |
| Source code | Not public |

### TTP Data Ingestion

The **Tesla Transport Protocol (TTP)** — the precursor to TTPoE — handles high-throughput video data loading from host servers into Dojo:

- Purpose: feed video frames (Tesla's primary training data) into Dojo at ~160 GB/s per tile edge
- Implemented in hardware (DIP cards)
- The only publicly known data-plane protocol within the cluster

---

## 5. Communication / Networking Library

### TTPoE (Scale-out)

The only open-source software component of the Dojo stack is the **Tesla Transport Protocol over Ethernet** (`teslamotors/ttpoe`), open-sourced at HC2024 (August 2024).

| Feature | Detail |
|---------|--------|
| Layer | Custom Layer-2/transport, runs over commodity Ethernet |
| Purpose | Scale-out collective communication (equivalent to NCCL's role in GPU clusters) |
| Hardware offload | Yes — in the Dumb-NIC |
| Collective operations | Not specified; suspected AllReduce or point-to-point |
| Open-source | https://github.com/teslamotors/ttpoe |

No equivalent of NCCL (a high-level collective communication library) for Dojo has been publicly described. The TTPoE repo contains the kernel-space driver and protocol implementation, not a high-level ML communication API.

---

## 6. What Is Not Public

| Category | Status |
|----------|--------|
| Compiler source code | Not public |
| ISA specification document | Not public |
| Assembler / disassembler | Not public |
| SDK / toolchain | Not public |
| Kernel library (cuDNN equivalent) | Not public |
| Runtime source code | Not public |
| Profiling / debugging tools | Not public |
| PyTorch backend plugin | Not public |
| MLIR dialect for Dojo | Not public |
| Model parallelism API | Not public |

---

## 7. Project Status Implications for Software Stack

As of August 2025, the Dojo team was disbanded. The software stack described above was an internal Tesla system — never intended for external use or open-source release. With the team disbanded and the D2/Dojo 2 program cancelled, the toolchain is effectively frozen. Tesla's new AI training direction (AI6, Samsung-fabbed) will require a new software stack.

The only lasting open-source contribution from Dojo's software work is the TTPoE protocol and its GitHub repository.

---

## Addendum — Software-Stack Status, 2026-08-08

*Conclusion up front: **no software-stack fact in sections 1–7 above changed.** No SDK, compiler, runtime, ISA document, kernel library, or framework backend has been released or announced for Dojo, Dojo 3, AI5, AI6, or AI7 as of 2026-08-08.*

### A. What was checked and what was found

| Question | Finding | Confidence |
|---|---|---|
| Any Dojo SDK / compiler / runtime release? | **No.** Nothing since the 2026-04-05 baseline | high |
| Any Dojo ISA specification published? | **No** | high |
| New TTPoE release or repo activity constituting a release? | **No new release** since the Hot Chips 36 (August 2024) open-sourcing. `teslamotors/ttpoe` remains the only open-source artifact of the program | high |
| Is TTPoE stated to be Dojo 3's fabric? | **Not stated.** No public source says Dojo 3 reuses TTPoE | high (as a negative on the claim) |
| Does the D1-era PyTorch + custom-compiler + LLVM stack carry forward to Dojo 3? | **Unknown.** No statement either way | not disclosed |
| Dojo 3 software toolchain (frontend, IR, codegen, runtime, collectives) | **not disclosed** — every layer | not disclosed |
| AI5 / AI6 / AI7 software toolchain | **not disclosed.** These are Tesla-internal inference/training SoCs; no external SDK exists | high |
| Tesla talk at Hot Chips 38 (Aug 23–25, 2026) that might disclose a stack? | **None in the advance program**, checked 2026-08-08 | high |

### B. Why the stack is likely to be discontinuous

Two facts make continuity of the D1 software stack doubtful, without establishing it either way:

1. **The team that built it was disbanded** (August 2025). Lead architect Peter Bannon departed and roughly 20 Dojo engineers left to found DensityAI (Bloomberg/Reuters, 2025-08-07). The compiler, runtime, and ISA tooling described above have no known maintaining organization.
2. **Dojo 3's stated target changed.** Musk's January 2026 statement is that "AI7/Dojo3 will be for space-based AI compute," corroborated by SpaceX's Form S-1 describing a chip type "optimized for the space environment to be used in our orbital compute infrastructure." A radiation/power-constrained orbital target has different software requirements from the D1-era video-training stack.

Silicon leadership did change: Anant Nivarti rejoined Tesla in April 2026 to lead silicon engineering including AI6 and Dojo 3 (Data Center Dynamics, 2026-04-07). No software-organization details accompanied that.

### C. Not a software disclosure

Tesla's **MEGAPOD** trademark filing (USPTO SN 99893717, ~2026-06-18) covers hardware goods — "computer servers, computer hardware for artificial intelligence processing, computer networking hardware, electrical power distribution units, and cooling systems, sold as a unit." It names no software, no chip, and no toolchain, and is a trademark filing rather than a product announcement.

### D. Corrections to circulating claims

- Statements that AI6 is "reserved for Optimus and data centers, not vehicles" are a **conflation with AI5**. AI6 is described as a self-driving/FSD chip as well (Teslarati, 2026-03-21). This matters for stack scoping: an FSD-targeted AI6 implies a vehicle-inference toolchain lineage (closer to the Tesla FSD chip stack) rather than a Dojo training-compiler lineage.
- The July 2025 "converged Dojo 3 / AI6 … 5, 12 on a board" earnings-call remark is sometimes read as implying a single unified software stack across car and datacenter. It is **stated intent from before the January 2026 space repositioning**, and no programming-model claim was ever attached to it.

---

## Sources

| Resource | URL |
|----------|-----|
| HC34 PDF | https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf |
| Next Platform — software overview | https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/ |
| teslamotors/ttpoe GitHub | https://github.com/teslamotors/ttpoe |
| HC2024 TTPoE PDF | https://hc2024.hotchips.org/assets/program/conference/day2/17_HC2024_Tesla_TTPoE_v5.pdf |
| IEEE — TTPoE paper | https://ieeexplore.ieee.org/document/10664947 |
| Tweaktown — AI Day transcript | https://www.tweaktown.com/news/81229/teslas-insane-new-dojo-d1-ai-chip-full-transcript-of-its-unveiling/index.html |

### Added 2026-08-08 (software-stack addendum)

| Resource | URL |
|----------|-----|
| `teslamotors/ttpoe` — re-checked 2026-08-08, no new release since HC36 | https://github.com/teslamotors/ttpoe |
| TechCrunch — Dojo3 for space-based AI compute (2026-01-20) | https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute/ |
| SpaceX Form S-1, filed 2026-05-20 (two chip types: terrestrial edge/inference vs space) | https://www.sec.gov/Archives/edgar/data/1181412/000162828026036936/spaceexplorationtechnologi.htm |
| Tesla Q2 2025 earnings call transcript (2025-07-23) | https://www.insidermonkey.com/blog/tesla-inc-nasdaqtsla-q2-2025-earnings-call-transcript-1575336/ |
| Teslarati — AI6 self-driving chip expectations (2026-03-21) | https://www.teslarati.com/elon-musk-teases-expectations-tesla-ai6-self-driving-chip/ |
| Teslarati — Tesla trademarks MEGAPOD (June 2026) | https://www.teslarati.com/tesla-just-trademarked-megapod-heres-what-it-is/ |
| Hot Chips 38 advance program — no Tesla/Dojo talk, checked 2026-08-08 | https://hotchips.org/advance-program/ |
