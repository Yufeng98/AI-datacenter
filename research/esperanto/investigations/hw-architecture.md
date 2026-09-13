# Esperanto ET-SoC-1 Hardware Architecture Investigation

*as_of: 2026-08-08 (baseline sections dated 2026-04-05; see appended investigation of 2026-08-08 at the end)*
*chip: esperanto*
*sources: IEEE Micro 2022, Hot Chips 33 (2021), WikiChip deep dive, STH HC33 writeup, RISC-V International blog*

---

## Overview

The Esperanto ET-SoC-1 is a **massively parallel RISC-V AI accelerator** built around a fundamentally different premise than GPU or systolic array chips: instead of one or a few large, complex compute units, deploy **over 1,000 small, energy-efficient RISC-V cores**, each augmented with a vector/tensor unit for ML math. The architectural bet is that the aggregate parallelism of many low-power in-order cores can match or exceed GPU throughput-per-watt on ML inference workloads — especially DLRM/recommendation and transformer models.

The chip is implemented in **TSMC 7nm** using **24+ billion transistors** across a **570 mm² die** with 89 metal layers.

---

## 1. Core Taxonomy: Two-Class Architecture

### 1.1 ET-Minion (1,088 cores — the compute workhorse)

The ET-Minion is the namesake of ET-SoC-1's defining feature: a custom 64-bit in-order RISC-V core designed to maximize operations-per-milliwatt for ML workloads.

| Parameter | Value |
|-----------|-------|
| Count per chip | 1,088 |
| ISA | RV64GCV (64-bit RISC-V + G base + Compressed + Vector extensions) |
| Pipeline | In-order, fine-grained multithreaded |
| Clock | Low-voltage operation optimized (few gates/stage) |
| INT8 peak | 128 GOPS per GHz per core |
| Key optimization | Minimal pipeline stages for high MHz at low voltage |

**Vector/Tensor Unit (per ET-Minion)**

Each ET-Minion carries a vector/tensor unit that substantially exceeds the integer core in die area — the linear and nonlinear math of ML is the dominant workload:

- **Wide vector execution**: Full vector-width operations every cycle
- **Tensor instructions**: Run for up to 512 cycles; a single tensor instruction can perform up to **32,000 multiply-accumulate operations** — amortizing instruction fetch overhead across thousands of MACs
- **Vector transcendental unit**: Hardware acceleration for exp(), log(), sin(), cos(), sigmoid, tanh — directly accelerating activation functions (softmax numerics, GeLU) without software lookup tables
- **Data types**: INT8, FP16, BF16, FP32 (exact precision support per datatype not fully disclosed)

**Energy Efficiency Rationale**

In-order cores have substantially fewer transistors than out-of-order cores (no reorder buffer, no branch predictor, no instruction window). This lets Esperanto pack 1,088 cores into 570 mm² at 7nm while keeping power below 20W — a power density that is physically impossible with GPU SM-class compute units at comparable TOPS.

### 1.2 ET-Maxion (4 cores — control/host)

| Parameter | Value |
|-----------|-------|
| Count per chip | 4 |
| ISA | RV64GCV (same ISA family) |
| Pipeline | Quad-issue out-of-order |
| Features | Branch prediction, sophisticated prefetch, high single-thread perf |
| Role | OS execution, driver tasks, coordination, serial host operations |

The ET-Maxion provides the high single-thread performance needed to boot Linux, manage DMA operations, coordinate the ET-Minion cores, and handle control-plane logic that cannot be easily parallelized. Four cores are sufficient because the control plane is not the bottleneck for inference throughput.

---

## 2. On-Chip Memory

| Parameter | Value |
|-----------|-------|
| Total on-chip SRAM | >160 MB (160+ million bytes) |
| Distribution | Distributed across core clusters; each ET-Minion has local SRAM |
| Access model | Software-managed; no hardware cache snooping required |
| Role | Weight tiles, activations, intermediate tensors |

The 160 MB of on-chip SRAM is substantial for an inference accelerator but still requires external DRAM for typical model weights. The architecture uses a **tiling / software-managed data movement** approach: model weights are streamed from LPDDR4x DRAM into the on-chip SRAM tiles, processed by the cores, and results returned.

This differs from Groq (pure on-chip SRAM, no DRAM) and from GPU (hardware caches with LRU eviction). The ET-SoC-1 approach is closer to the Cerebras / Tenstorrent style: large per-core scratchpad, software-controlled data placement.

---

## 3. Off-Chip Memory

| Parameter | Value |
|-----------|-------|
| Memory type | LPDDR4x DRAM |
| Maximum capacity | 32 GB per chip |
| DRAM bandwidth | ~137 GB/s per chip (estimated from LPDDR4x spec at 32-bit width × 4267 MT/s) |
| Storage | eMMC Flash interface (for weights/model persistence) |
| Memory model | Traditional load/store from RISC-V cores |

LPDDR4x was chosen for its **low power** (the "LP" in LPDDR), consistent with the <20W TDP target. The tradeoff: LPDDR4x bandwidth (~137 GB/s) is far lower than HBM2e (~1.2 TB/s) or HBM3 (~3.35 TB/s). This was acceptable for the initial DLRM/recommendation target workload (embedding table lookups are memory-capacity-bound, not bandwidth-bound), but limited the chip for bandwidth-intensive transformer inference.

---

## 4. Host Interface and Packaging

| Parameter | Value |
|-----------|-------|
| PCIe | Gen4 x8 (8 lanes) |
| Form factor | PCIe accelerator card |
| Glacier Point v2 card | 6 ET-SoC-1 chips per card |
| Card aggregate compute | ~600–1,200 TOPS (6 × 100–200 TOPS) |
| Card aggregate DRAM | 192 GB (6 × 32 GB LPDDR4x) |
| Card aggregate DRAM BW | ~822 GB/s (6 × 137 GB/s) |
| Card aggregate cores | 6,558 ET-Minion + 24 ET-Maxion RISC-V cores |

The Glacier Point v2 card demonstrates the scalability of the approach: 6 chips on a single PCIe card, all accessible as a unified pool of RISC-V compute. The host system (x86 server) communicates via PCIe Gen4; the ET-Maxion cores handle host-facing control while ET-Minion cores execute the inference workload.

---

## 5. Execution Model: Massively Parallel SPMD on RISC-V

The fundamental programming model is **SPMD (Single Program, Multiple Data)** across 1,088 cores — the same paradigm as GPU CUDA but executed on standard RISC-V CPUs rather than SIMT hardware warps:

- Each ET-Minion runs an **independent RISC-V thread**
- Threads cooperate via shared SRAM and explicit synchronization primitives
- No hardware warp scheduler — scheduling is the programmer's / compiler's responsibility
- Vector/tensor units run long-latency tensor instructions (up to 512 cycles), allowing the integer pipeline to overlap with tensor unit execution

This model has a key advantage: **the full RISC-V software ecosystem applies**. RISC-V compilers (GCC, LLVM), operating systems (Linux), debuggers, profilers — all work natively on ET-Minion without custom toolchain development. Esperanto extended the ISA for ML but the base toolchain is standard.

---

## 6. Inter-Chip Communication (Glacier Point v2)

On a 6-chip Glacier Point v2 card, inter-chip communication is not explicitly described in public sources as using a specialized fabric (unlike Groq's plesiosynchronous protocol or Tenstorrent's Ethernet mesh). The architecture appears to use **host-mediated PCIe** for inter-chip coordination at the card level, with DRAM accessible to all chips for data sharing. This is a simpler but lower-bandwidth approach compared to peers with dedicated chip-to-chip links.

---

## 7. Key Architecture Trade-offs

| Design Decision | What is Gained | What is Given Up |
|-----------------|----------------|-----------------|
| 1,088 in-order cores vs. few large OoO cores | Low power (<20W), fine-grained parallelism, RISC-V ecosystem | Single-thread performance; complex kernel parallelization required |
| Per-core vector/tensor unit with 32K-op tensor instructions | High ML math throughput per watt; long tensor ops amortize fetch overhead | Large ISA extension area per core |
| LPDDR4x vs. HBM | Low power, low cost | Limited bandwidth (~137 GB/s vs. HBM's TB/s range) — bottleneck for transformer inference |
| RISC-V base ISA | Full open toolchain (GCC, LLVM, Linux); C/C++ programmability | Custom ML-specific paths need hand-tuning in compiler/runtime |
| 160 MB on-chip SRAM | Large working set; fewer DRAM round-trips | Still requires DRAM for typical model weights |
| <20W TDP | Edge/embedded deployment potential; low cooling cost | Peak throughput limited vs. data-center accelerators |

---

## 8. Comparison: ET-SoC-1 vs. Contemporaries

| Dimension | Esperanto ET-SoC-1 | NVIDIA A100 (80GB) | Tenstorrent Blackhole |
|-----------|---------------------|--------------------|-----------------------|
| Compute units | 1,088 RISC-V cores + VTU | 6,912 CUDA cores + Tensor Cores | 140 Tensix (5 RISC-V per Tensix) |
| Peak TOPS INT8 | 100–200 | 624 | 745 (FP8) |
| TDP | <20W | 400W | ~300W |
| TOPS/W | 5–10 INT8 | 1.56 INT8 | ~2.5 FP8 |
| Off-chip memory | LPDDR4x 32GB | HBM2e 80GB | GDDR6 32GB |
| Memory BW/chip | ~137 GB/s | 2 TB/s | 512 GB/s |
| ISA | RISC-V RV64GCV + Esperanto extensions | CUDA PTX + SASS | RISC-V + Tensix ISA |
| Programming model | C/C++ SPMD on RISC-V; Glow-based ML stack | CUDA; cuDNN; TensorRT | TT-Metal; TTNN |

The ET-SoC-1's dominant strength is **TOPS per watt** at a <20W power envelope — roughly 3–6x better efficiency than A100 class hardware. Its weakness is raw bandwidth: LPDDR4x limits large transformer models.

---

## 9. Resources

- [IEEE Micro 2022 — Dave Ditzel: Esperanto ET-SoC-1](https://www.esperanto.ai/wp-content/uploads/2022/05/Dave-IEEE-Micro.pdf)
- [Hot Chips 33 (2021) Presentation](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Esperanto.Dave_Ditzel.presentation.v1submitted.pdf)
- [IEEE Xplore: Accelerating ML Recommendation with 1000+ RISC-V Processors](https://ieeexplore.ieee.org/document/9566904/)
- [WikiChip Deep Dive: ET-SoC-1](https://fuse.wikichip.org/news/4911/a-look-at-the-et-soc-1-esperantos-massively-multi-core-risc-v-approach-to-ai/)
- [STH: Esperanto ET-SoC-1 at HC33](https://www.servethehome.com/esperanto-et-soc-1-1092-risc-v-ai-accelerator-solution-at-hot-chips-33/)
- [RISC-V International: ET-SoC-1 ML Recommendation](https://riscv.org/blog/accelerating-ml-recommendation-with-over-1000-risc-v-tensor-processors-on-esperantos-et-soc-1-chip-david-r-ditzel-esperanto-technologies-inc/)

---

# Appended Investigation — 2026-08-08: Ainekko / CORE-ET Silicon Platform (ETSP)

*Scope: post-acquisition hardware disposition. Sections 1–9 above are unchanged and remain the authority on ET-SoC-1 silicon, which has not been revised.*

*Primary sources consulted this pass: Ainekko GlobeNewswire releases 2025-11-19, 2026-01-29, 2026-06-02; the GlobeNewswire AiNekko organization index; github.com/openhwgroup/core-et and its raw LICENSE file; the GitHub API records for openhwgroup repos, core-et-erbium, and the user `ainekko`; openhwfoundation.org; esperanto.ai; nekko.ai (vendor site — labeled at every point of use).*

## A. Correction of two dates already committed in this repo

| Fact | Repo said (as_of 2026-04-05) | Correct | Evidence |
|---|---|---|---|
| Ainekko acquires Esperanto IP | October 2025 | **2025-11-19** | GlobeNewswire acquisition release dated 2025-11-19. October 22, 2025 was Ainekko's separate "AI Foundry" launch — the repo conflated the two events |
| Ainekko merges with Veevx | November 2025 | **2026-01-29** | GlobeNewswire merger release dated 2026-01-29 |
| Code open-sourced | November 2025 | Public repos **created 2026-04-20** | GitHub API: `openhwgroup/core-et` and `core-et-erbium` both created 2026-04-20. The 2025-11-19 release names no license and says only that Ainekko is "exploring foundation-based governance" |

The repo's **"Apache 2.0" license claim is substantially correct and should NOT be marked unverified**: `core-et/LICENSE` is verbatim Apache License 2.0. Only the date was wrong.

## B. What is published, precisely

| Item | Observation |
|---|---|
| Repos | `github.com/openhwgroup/core-et`, `github.com/openhwgroup/core-et-erbium` |
| Created / last push | Both created 2026-04-20; `core-et` pushed 2026-05-07; `core-et-erbium` pushed 2026-06-23 (~4.9 MB, 3 stars) |
| Checked-in license | `core-et/LICENSE` = verbatim Apache License 2.0 |
| Press-release license | 2026-06-02 names "Solderpad Hardware License v2.1 ... building on the well-known Apache 2.0 license" — **unresolved discrepancy with the checked-in file**; both recorded, neither suppressed |
| Project name / governance | **"CORE-ET Silicon Platform (ETSP)"**, accepted as an **OpenHW Foundation** project 2026-06-02. OpenHW Foundation (formerly OpenHW Group; openhwgroup.org 302-redirects to openhwfoundation.org) is "an Eclipse Foundation global initiative." Ainekko is listed as an OpenHW member. Descriptions of the release as being "under the Eclipse Foundation" are loose but not false |

**Scope caveat (important, and easy to get wrong):** this is **not** Esperanto's production ET-SoC-1 RTL. `core-et` self-describes as "an Ainekko project for collecting hardware IP in a form that can be translated, verified, documented, and integrated by agentic hardware-development workflows," working "CORE-ET modules into clean SystemVerilog," with the original kept on a separate branch. Esperanto lineage is visible only indirectly — e.g. `docs/minion_vpu_standalone_plan.md` referencing the "real-VPU Minion path." **Neither the 2025-11-19 nor the 2026-06-02 release names ET-SoC-1.** The 2025-11-19 release characterizes the acquired IP as "more than 1,000 Minion™ cores on a single die."

`github.com/ainekko` is an **unrelated personal User account** (GitHub API: type User, created 2023-08-15, 4 public repos, no company or website set). It must not be cited as the code host.

## C. Successor hardware — CORE-ET Silicon Platform (16nm)

The one substantive *hardware* development: the 2026-06-02 release states the design **"is being taped out on a 16nm node"** and describes **"a scalable network-on-chip (NoC) architecture with integrated MRAM-based memory."**

| Parameter | Value |
|---|---|
| Process | 16nm; foundry **not disclosed** |
| Status | **Tapeout in progress** — not sampling, not shipping, not deployed; no silicon demonstrated |
| On-die memory | Integrated **MRAM**-based memory; the 2026-01-29 Veevx release independently confirms MRAM "with SRAM-like performance" for edge/embedded inference |
| MRAM capacity / bandwidth / density | **Not disclosed** |
| Fabric | Scalable **NoC**; topology, width, bandwidth **not disclosed**. Intra-die only — no chip-to-chip fabric disclosed |
| Core count, peak TOPS, TDP, SRAM, off-chip DRAM, host interface | **Not disclosed** |

MRAM is a genuine architectural delta: ET-SoC-1 had >160 MB SRAM plus LPDDR4x and no non-volatile on-die tier. The 16nm node is a step *back* from ET-SoC-1's TSMC 7nm — consistent with an edge/embedded cost target rather than datacenter density.

This supersedes the repo's prior "plans 8-core tapeout on TSMC shuttle wafer" note.

## D. Provenance discipline — vendor-only claims

The following appear **only on nekko.ai** and in no press release retrieved this pass. They are recorded as vendor self-description and must not be presented as confirmed specifications:

- "iRAM" as the brand for the MRAM tier, and "Erbium" as a design name (the latter also exists only as a repo name)
- Tagline "Model-to-Silicon for Physical AI"
- "A tape-out every 9 months" via reusable, "programmable and composable" platform blocks
- Attribution of "1,088 RISC-V cores" to ET-SoC-1 on the Ainekko site (the figure itself is independently confirmed for ET-SoC-1 by IEEE Micro / HC33 — the point is that Ainekko's site is not the source of record)
- The manifesto "The Next Thousand Chips," dated 2026-05-05, by Tanya Dadasheva
- Douglas Smith's titles "co-founder and Head of Silicon" and "former Broadcom IP architect." The 2026-01-29 press release says only "executive team member" who "for many years managed the intellectual property at Broadcom"

## E. Negative findings and open items

- **No ET-SoC-2x/3x revival**, no HBM successor, no FP64 successor — absent from every source retrieved. Treat as permanently cancelled.
- **No new ET-SoC-1 silicon, card, or deployment.**
- **Hot Chips 38 (2026-08-23 to 2026-08-25) presence: not confirmed.** hotchips.org/program returned HTTP 403 and could not be checked. Recorded as *not confirmed*, not as a negative finding. Hot Chips 38 is 15 days in the future as of this writing; no slides or abstracts exist.
- **esperanto.ai**: permanent banner "Esperanto Technologies has ceased operations" linking to nekko.ai as IP acquirer. No redirect; historical pages remain as archive; newest news item 2025-04-09 — consistent with the July-2025 closure already recorded.
- **Datacenter relevance: effectively zero.** The live trajectory is edge/embedded inference.

## F. Sources added this pass

- https://www.globenewswire.com/news-release/2025/11/19/3191016/0/en/Ainekko-Acquires-Esperanto-Technologies-Intellectual-Property-to-Power-Open-Source-Edge-AI-Platform.html
- https://globenewswire.com/news-release/2026/01/29/3228632/0/en/ainekko-merges-with-veevx-expands-open-silicon-platform-with-breakthrough-memory-and-embedded-ai-capabilities.html
- https://globenewswire.com/news-release/2026/06/02/3305225/0/en/ainekko-s-edge-ai-silicon-platform-is-now-an-openhw-foundation-open-source-project.html
- https://www.globenewswire.com/search/organization/AINekko
- https://github.com/openhwgroup/core-et
- https://raw.githubusercontent.com/openhwgroup/core-et/main/LICENSE
- https://api.github.com/orgs/openhwgroup/repos?per_page=100&sort=created&direction=desc
- https://api.github.com/repos/openhwgroup/core-et-erbium
- https://api.github.com/users/ainekko
- https://openhwfoundation.org/
- https://www.esperanto.ai/
- https://nekko.ai/ (vendor)
