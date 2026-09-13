# Esperanto ET-SoC-1 Software and Hardware Stack Summary

*as_of: 2026-08-08*

---

## Overview

The Esperanto ET-SoC-1 is the most radically different approach to AI acceleration in the chip landscape from a software-ecosystem perspective: **run 1,088 standard 64-bit RISC-V CPUs in parallel**, each augmented with a vector/tensor unit for ML math, all on a single TSMC 7nm die consuming less than 20 watts. Where NVIDIA builds a few massively powerful SMs, Cerebras builds a wafer-scale mesh, and Groq builds a deterministic SRAM processor, Esperanto builds a **many-core RISC-V cluster on a chip** — essentially a 1,000-node CPU cluster in a 570 mm² package drawing less power than a typical laptop charger.

The chip's deepest insight is a software one: because ET-Minion cores implement standard RISC-V RV64GCV, the full open-source RISC-V toolchain — GCC, LLVM, Linux, gdb, valgrind — works natively. Instead of building a bespoke ISA and toolchain (as NVIDIA did with PTX/CUDA, as Google did with XLA/TPU), Esperanto inherits a rich existing ecosystem and adds ML extensions on top. This dramatically lowers the software bring-up burden.

**Corporate note**: Esperanto Technologies closed in July 2025 after staff attrition. esperanto.ai now carries a permanent banner — "Esperanto Technologies has ceased operations" — linking to Ainekko as the IP acquirer; the site does not redirect and the historical product/technology pages remain up as an archive (newest news item 2025-04-09). Ainekko (self-branded "AiNEKKO"; nekko.ai, ainekko.io) announced the acquisition of Esperanto's intellectual property on **2025-11-19** — the repo previously dated this to October 2025, which conflated it with Ainekko's separate 2025-10-22 "AI Foundry" launch. Ainekko redirected the architecture from data-center inference toward open-hardware edge AI; the software stack (Glow-based compiler, RISC-V extensions, ONNXRuntime EP) was also transferred. **Datacenter relevance of this chip is now effectively zero** — the live trajectory is edge/embedded inference. See "Ainekko / CORE-ET Silicon Platform Update (2026-08-08)" below for the current state.

---

## Hardware Architecture

### Dual Core Class Design

| Core Type | Count | Type | Primary Role |
|-----------|-------|------|-------------|
| ET-Minion | 1,088 | In-order, multithreaded RV64GCV | ML inference compute |
| ET-Maxion | 4 | Quad-issue out-of-order RV64GCV | OS, control, coordination |

**ET-Minion**: Minimal in-order pipeline optimized for high MHz at low voltage. The integer core is small — what makes ET-Minion powerful for ML is the **per-core vector/tensor unit**: up to 512-cycle tensor instructions performing 32,000 MACs in a single instruction issue, plus a vector transcendental unit for activation functions (exp, log, tanh, sigmoid, GeLU, SiLU) in hardware.

**ET-Maxion**: Four out-of-order cores handle Linux, PCIe host communication, DMA management, and coordination of the ET-Minion pool. The chip can operate as a standalone compute node (Linux server), not just a PCIe accelerator.

### Performance and Power

| Metric | Value |
|--------|-------|
| Process | TSMC 7nm |
| Die area | 570 mm² |
| Transistors | 24+ billion |
| TDP | <20W |
| Peak INT8 TOPS | 100–200 (frequency-dependent) |
| TOPS/W (INT8) | ~5–10 |
| On-chip SRAM | >160 MB |
| Off-chip memory | LPDDR4x up to 32 GB (~137 GB/s) |
| Host interface | PCIe Gen4 x8 |

### Glacier Point v2 Card

Six ET-SoC-1 chips on a single PCIe card:

| Card Metric | Value |
|-------------|-------|
| Chips | 6 |
| Total RISC-V cores | 6,558 ET-Minion + 24 ET-Maxion |
| Total DRAM | 192 GB LPDDR4x |
| Total DRAM bandwidth | ~822 GB/s |
| Peak INT8 TOPS | ~600–1,200 |

### Key Architecture Trade-offs

The ET-SoC-1's defining strength is **TOPS per watt** at under 20W — roughly 3–6x more efficient than A100-class hardware for supported workloads. The defining weakness is **memory bandwidth**: LPDDR4x at ~137 GB/s per chip is 15–20x lower than HBM3. This made the chip excellent for embedding-table-dominated DLRM recommendation inference (where the bottleneck is memory capacity and random access, not bandwidth) but limited for bandwidth-intensive large transformer inference. The planned ET-SoC-2x/3x roadmap (previewed Nov 2024 with NEC) was to add HBM and FP64 — but the company closed before tapeout. *As of 2026-08-08 no ET-SoC-2x/3x revival, and no HBM or FP64 successor, appears in any Ainekko or OpenHW Foundation source; the 16nm part now in tapeout (below) is an edge/embedded design with MRAM, not an HBM datacenter part.*

---

## Software Stack

### Framework Integration

Esperanto's primary model entry path is **ONNX** — PyTorch models are exported to ONNX via `torch.onnx.export()`, TensorFlow via tf2onnx. The key user-facing runtime interface is a custom **ONNXRuntime Execution Provider (EP)**: users load a model with the ET-SoC-1 EP, call `sess.run()`, and hardware dispatch is transparent. The programming API requires no hardware-specific code.

For the HPC use case (General Purpose SDK, launched May 2023), users write **standard C/C++** with OpenMP-style thread pragmas targeting all 1,088 ET-Minion cores — the closest analog to writing parallel code for a many-core CPU cluster.

### Compiler / IR

The ML compiler is built on **Meta's Glow** (open source, `github.com/pytorch/glow`):

1. **ONNX → Glow graph IR**: Standard ops (Conv, GEMM, Attention, Softmax, Norm)
2. **Glow optimization passes**: Op fusion, quantization, constant folding
3. **Lowering to RISC-V via LLVM**: Glow → LLVM IR → RISC-V backend with Esperanto's custom vector/tensor intrinsics
4. **Output**: Per-core compiled RISC-V ELF binaries + interleaved weight data for SRAM locality

Unlike GPU compilers (TensorRT, XLA), which generate a small number of large GPU kernels, the Glow/LLVM pipeline generates **1,088 separate RISC-V binaries** — one per ET-Minion partition. This scales compilation time with core count but achieves fine-grained parallelism.

### Runtime

The runtime is thin: initialize ET-Minion SRAM tiles, load compiled per-core binaries via PCIe DMA (or from DRAM directly via ET-Maxion), release ET-Minion resets, collect results. No dynamic scheduling occurs — the compiled partition assignment is fully static, similar to Groq's `.iop` model. However, unlike Groq's all-SRAM design, the ET-SoC-1 runtime must manage DMA pipelining between LPDDR4x and the on-chip SRAM tiles during inference.

### Programming Model Rationale

The ET-SoC-1 programming model has a unique advantage: **standard RISC-V tools all work**. A developer can:
- Use `gdb` to debug a running ET-Minion
- Profile with standard RISC-V perf counters
- Write custom kernels in C with RISC-V vector intrinsics
- Compile any RISC-V-compatible code directly to the chip

This contrasts with CUDA (proprietary), PTX (proprietary virtual ISA), and even Tensix ISA (Tenstorrent — RISC-V base but with custom SFPU). The openness of RISC-V as the base ISA is the ET-SoC-1's single largest software differentiator.

---

## Target Workloads

### Initial Target: DLRM / Recommendation Inference
- Embedding table lookups: random DRAM access pattern, memory-capacity-bound
- Low arithmetic intensity — ET-SoC-1's <20W power suits always-on servers
- Samsung SDS reported strong results on recommendation workloads
- Cambrian AI Research: "rock solid" ResNet-50, DLRM, Transformer performance (2022)

### Expanded Target (2023–2025): GenAI + HPC
- Pivoted to transformer inference (LLMs, Whisper ASR)
- Small language models with KV-cache via ONNX
- General-purpose HPC via C/C++ SDK (Monte Carlo, genomics, physics)
- DLRM pivot was necessary due to bandwidth limitations for large LLMs on LPDDR4x

---

## Corporate Timeline and Current Status

| Date | Event |
|------|-------|
| 2017 | Esperanto Technologies founded by Dave Ditzel (Intel, Transmeta, SiFive alum) |
| Aug 2021 | ET-SoC-1 unveiled at Hot Chips 33 |
| Apr 2022 | Silicon in customer hands; Samsung SDS evaluation begins |
| May 2023 | General Purpose HPC SDK launched |
| Nov 2024 | NEC HPC partnership; ET-SoC-2x/3x roadmap previewed |
| Jul 2025 | Esperanto Technologies closes due to staff attrition |
| 2025-10-22 | Ainekko launches its "AI Foundry" platform — **not** the Esperanto acquisition (the repo previously misdated the acquisition to this event) |
| 2025-11-19 | Ainekko announces acquisition of Esperanto's intellectual property. Release describes "more than 1,000 Minion™ cores on a single die"; says only that Ainekko is "exploring foundation-based governance"; names no license and does not name ET-SoC-1 |
| 2026-01-29 | Ainekko merges with MRAM startup **Veevx** ("breakthrough Memory and Embedded AI Capabilities"). MRAM described as having "SRAM-like performance" for edge/embedded inference. Douglas Smith joins as an "executive team member" who "for many years managed the intellectual property at Broadcom" |
| 2026-04-20 | Public repositories **github.com/openhwgroup/core-et** and **core-et-erbium** created — a clean-SystemVerilog re-translation of the IP, not Esperanto's production RTL. `core-et`'s LICENSE file is verbatim Apache License 2.0 |
| 2026-05-05 | Manifesto "The Next Thousand Chips" published by Tanya Dadasheva (nekko.ai only; not in any press release) |
| 2026-06-02 | **"CORE-ET Silicon Platform (ETSP)" accepted as an OpenHW Foundation project.** Release describes "a scalable network-on-chip (NoC) architecture with integrated MRAM-based memory" and states the design "is being taped out on a 16nm node". Names the license as "Solderpad Hardware License v2.1 ... building on the well-known Apache 2.0 license" |

*(Superseded rows removed from this table: "Oct 2025 — Ainekko acquires all IP", "Nov 2025 — open-sources ET-SoC-1 RTL (Apache 2.0); merges with Veevx; plans 8-core tapeout on TSMC shuttle wafer". The acquisition is Nov 2025, the Veevx merger is Jan 2026, the public code dates to Apr 2026, and the 8-core shuttle plan is superseded by the 16nm tapeout.)*

---

## Ainekko / CORE-ET Silicon Platform Update (2026-08-08)

*Updated 2026-08-08. Primary sources: Ainekko GlobeNewswire releases 2025-11-19, 2026-01-29, 2026-06-02; github.com/openhwgroup/core-et (+ core-et-erbium) and its LICENSE file; openhwfoundation.org; esperanto.ai closure banner; nekko.ai (vendor site, labeled where used).*

Nothing about the ET-SoC-1 silicon itself has changed — it remains the 2021 TSMC 7nm, 1,088-ET-Minion part described above, and no new ET-SoC-1 hardware has shipped. What changed since the 2026-04-05 baseline is the **disposition of the IP**: it is now public code under foundation governance, and a physically different successor part is in tapeout.

### 1. What is actually public — and what is not

The code is locatable and is hosted by the **OpenHW Foundation**, not by Ainekko's own GitHub:

| Item | Detail |
|---|---|
| Repositories | `github.com/openhwgroup/core-et`, `github.com/openhwgroup/core-et-erbium` |
| Created | 2026-04-20 (both) |
| Last activity observed | `core-et` pushed 2026-05-07; `core-et-erbium` pushed 2026-06-23 (~4.9 MB, 3 stars) |
| Checked-in license | `core-et/LICENSE` is **verbatim Apache License 2.0** |
| License named in press release | **"Solderpad Hardware License v2.1 ... building on the well-known Apache 2.0 license"** (2026-06-02) — a genuine discrepancy with the checked-in file |
| Project / governance | **"CORE-ET Silicon Platform (ETSP)"**, accepted as an OpenHW Foundation project 2026-06-02; Ainekko is listed as an OpenHW member |

> **Scope caveat — do not say "the ET-SoC-1 manycore was open-sourced."** `core-et` describes itself as "an Ainekko project for collecting hardware IP in a form that can be translated, verified, documented, and integrated by agentic hardware-development workflows," working from "CORE-ET modules into clean SystemVerilog," with the original kept on a separate branch. It is a **re-translation**, not Esperanto's production ET-SoC-1 RTL. Esperanto lineage is visible only indirectly (e.g. `docs/minion_vpu_standalone_plan.md` referencing the "real-VPU Minion path"). Neither the 2025-11-19 nor the 2026-06-02 release names ET-SoC-1.

> **Governance wording.** Third-party summaries describing the release as "under the Eclipse Foundation" are loose but not false: OpenHW Foundation (formerly OpenHW Group; openhwgroup.org now 302-redirects to openhwfoundation.org) is "an Eclipse Foundation global initiative." The precise home is the OpenHW Foundation.

> **Do not cite `github.com/ainekko`.** The GitHub API confirms it is an unrelated personal **User** account (created 2023-08-15, 4 public repos, no company or website set) — not the code host.

Net effect on the repo's prior claim: **"Apache 2.0" was substantially correct and should not be marked unverified** — only the date was wrong (public repos date to 2026-04-20, not November 2025), and the press-release license name differs from the checked-in file.

### 2. A 16nm successor is in tapeout — with MRAM and a NoC

The 2026-06-02 release states the design **"is being taped out on a 16nm node"** and describes **"a scalable network-on-chip (NoC) architecture with integrated MRAM-based memory."** The 2026-01-29 Veevx release independently confirms MRAM "with SRAM-like performance" targeted at edge/embedded inference.

| Attribute | Value |
|---|---|
| Process | **16nm** (foundry not disclosed) |
| Status | **Tapeout in progress** — not sampling, not shipping, not deployed; no silicon demonstrated |
| Memory | Integrated **MRAM**-based on-die memory — capacity, bandwidth, density: **not disclosed** |
| Fabric | Scalable **NoC** — topology, width, bandwidth: **not disclosed** |
| Core count / peak performance | **Not disclosed** for the 16nm part |
| Target | Edge / embedded ("Physical AI") inference — not datacenter |

MRAM is a genuinely new memory tier relative to ET-SoC-1, which used >160 MB SRAM plus LPDDR4x. This supersedes the repo's earlier "plans 8-core tapeout on TSMC shuttle wafer" note.

> **Vendor branding, not press-release fact.** The names **"iRAM"** (for the MRAM tier) and **"Erbium"** appear only on nekko.ai and as a repository name; both the January and June 2026 press releases say only "MRAM-based embedded AI" / "integrated MRAM-based memory." Likewise vendor-site-only and single-sourced: the tagline "Model-to-Silicon for Physical AI," the claim of "a tape-out every 9 months" via reusable platform blocks, the attribution of "1,088 RISC-V cores" to ET-SoC-1 on nekko.ai, and the manifesto "The Next Thousand Chips."

> **Titles.** "Co-founder and Head of Silicon" and "former Broadcom IP architect" for Douglas Smith appear only on nekko.ai. The 2026-01-29 press release says only "executive team member" who "for many years managed the intellectual property at Broadcom."

### 3. What did *not* happen

- **No ET-SoC-2x/3x revival**, no HBM successor, no FP64 successor — absent from every retrieved source.
- **No new ET-SoC-1 silicon**, no new card, no new deployment.
- **Hot Chips 38 (Aug 23–25, 2026) presence: not confirmed** — hotchips.org/program returned HTTP 403 and could not be checked. This is "not confirmed", not a negative finding.
- Datacenter relevance remains effectively zero. The chip stays in this survey as an *architectural* data point (many-core RISC-V inference at <20W), not as a deployable datacenter part.

---

## Resources

### Primary Sources
- [IEEE Micro 2022: Esperanto ET-SoC-1 Chip (Dave Ditzel)](https://www.esperanto.ai/wp-content/uploads/2022/05/Dave-IEEE-Micro.pdf)
- [Hot Chips 33 (2021) Presentation](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Esperanto.Dave_Ditzel.presentation.v1submitted.pdf)
- [IEEE Xplore: Accelerating ML Recommendation with 1000+ RISC-V Processors](https://ieeexplore.ieee.org/document/9566904/)

### Architecture Analysis
- [WikiChip Deep Dive: ET-SoC-1](https://fuse.wikichip.org/news/4911/a-look-at-the-et-soc-1-esperantos-massively-multi-core-risc-v-approach-to-ai/)
- [STH: Esperanto at HC33](https://www.servethehome.com/esperanto-et-soc-1-1092-risc-v-ai-accelerator-solution-at-hot-chips-33/)
- [Cambrian AI: Esperanto RISC-V in AI and HPC](https://cambrian-ai.com/esperanto-sees-a-bright-future-for-risc-v-in-ai-and-hpc/)

### Software Documentation
- [Esperanto Technology Page](https://www.esperanto.ai/technology/)
- [SDK Launch Press Release](https://www.esperanto.ai/News/esperanto-technologies-launches-general-purpose-sdk/)
- [Blog: Compiling SLMs with ONNXRuntime](https://www.esperanto.ai/blog/compiling-and-executing-small-language-models-with-onnxruntime-2/)
- [Blog: Adapting Whisper to ET-SoC-1](https://www.esperanto.ai/blog/adapting-whisper-models-to-et-soc-1-architecture-and-exporting-them-to-onnx/)
- [Esperanto GitHub](https://github.com/esperantotech)

### Corporate News
- [EE Times: Esperanto Pivots to HPC and GenAI](https://www.eetimes.com/esperanto-pivots-to-hpc-and-generative-ai/)
- [XPU.pub: Esperanto Exits AI Chips](https://xpu.pub/2025/07/05/esperanto/)
- [EE Times: Ainekko Buys Esperanto IP, Open-Sources It](https://www.eetimes.com/ainekko-buys-esperanto-hardware-ip-open-sources-it/)
- [GlobeNewswire: Ainekko Acquires Esperanto IP (2025-11-19)](https://www.globenewswire.com/news-release/2025/11/19/3191016/0/en/Ainekko-Acquires-Esperanto-Technologies-Intellectual-Property-to-Power-Open-Source-Edge-AI-Platform.html)
- [EE Times: Ainekko + Veevx MRAM Merger](https://www.eetimes.com/ainekko-merges-with-veevx-adds-mram-to-open-source-stack/)

### Post-Acquisition Primary Sources (added 2026-08-08)
- [GlobeNewswire: Ainekko Merges with Veevx (2026-01-29)](https://globenewswire.com/news-release/2026/01/29/3228632/0/en/ainekko-merges-with-veevx-expands-open-silicon-platform-with-breakthrough-memory-and-embedded-ai-capabilities.html)
- [GlobeNewswire: Ainekko's Edge AI Silicon Platform is now an OpenHW Foundation Open-Source Project (2026-06-02)](https://globenewswire.com/news-release/2026/06/02/3305225/0/en/ainekko-s-edge-ai-silicon-platform-is-now-an-openhw-foundation-open-source-project.html)
- [GlobeNewswire: all AiNekko releases](https://www.globenewswire.com/search/organization/AINekko)
- [github.com/openhwgroup/core-et](https://github.com/openhwgroup/core-et) — CORE-ET Silicon Platform source (created 2026-04-20)
- [core-et LICENSE (verbatim Apache-2.0)](https://raw.githubusercontent.com/openhwgroup/core-et/main/LICENSE)
- [github.com/openhwgroup/core-et-erbium (API record)](https://api.github.com/repos/openhwgroup/core-et-erbium)
- [OpenHW Foundation](https://openhwfoundation.org/)
- [Ainekko / AiNEKKO company site](https://nekko.ai/) — vendor source; "iRAM", "Model-to-Silicon for Physical AI", "The Next Thousand Chips" appear only here
- [esperanto.ai (closure banner, archived product pages)](https://www.esperanto.ai/)
