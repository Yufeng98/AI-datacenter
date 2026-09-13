# Esperanto ET-SoC-1 Hardware Architecture

*as_of: 2026-08-08*
*generations: ET-SoC-1 (TSMC 7nm, 2021–2025, shipped) / CORE-ET Silicon Platform — ETSP (16nm, tapeout in progress 2026, Ainekko + OpenHW Foundation)*

---

## Overview

The Esperanto ET-SoC-1 is a **massively parallel RISC-V AI accelerator** that takes an unconventional bet: pack **1,088 energy-efficient in-order RISC-V cores** (ET-Minion), each with a dedicated vector/tensor unit, onto a single TSMC 7nm die consuming less than 20 watts. Four out-of-order ET-Maxion cores handle the control plane and run Linux.

The result is a chip that achieves 100–200 INT8 TOPS at under 20W — approximately 5–10 TOPS/W — roughly 3–6x more efficient than contemporaneous GPU hardware for supported workloads. The architectural premise: many small, energy-efficient cores with local compute beat a few large, power-hungry ones when the workload (ML inference) is naturally data-parallel.

Esperanto Technologies closed in July 2025. Its IP was acquired by **Ainekko** (announced 2025-11-19) and is now published as the **CORE-ET Silicon Platform (ETSP)**, an OpenHW Foundation project (accepted 2026-06-02). A 16nm successor with integrated MRAM and a scalable NoC is in tapeout. Sections 1–8 below describe the shipped ET-SoC-1; section 9 covers the successor.

---

## Generation Overview

| Generation | Year | Process | Compute | On-chip memory | Off-chip memory | Fabric | Status |
|---|---|---|---|---|---|---|---|
| **ET-SoC-1** | 2021 (silicon Q1 2022) | TSMC 7nm, 570 mm², 24B+ transistors, 89 mask layers | 1,088 ET-Minion (in-order RV64GCV + per-core VTU) + 4 ET-Maxion (quad-issue OoO) | >160 MB distributed SRAM (SW-managed) | LPDDR4x, 32 GB, ~137 GB/s | Host PCIe Gen4 x8; no chip-to-chip fabric | Shipped; company closed Jul 2025 |
| *ET-SoC-2x / 3x (planned)* | previewed Nov 2024 (NEC) | not disclosed | not disclosed; FP64 added | not disclosed | **HBM** (planned) | not disclosed | **Never fabricated**; no revival as of 2026-08-08 |
| **CORE-ET Silicon Platform (ETSP)** | announced 2026-06-02 | **16nm** (foundry not disclosed) | not disclosed (RISC-V "Minion"-lineage IP re-translated to clean SystemVerilog) | **Integrated MRAM-based memory** — capacity/BW not disclosed | not disclosed | **Scalable NoC** — topology/BW not disclosed | **Tapeout in progress**; not sampling, not shipping |

*The ETSP row is deliberately sparse: the 2026-06-02 press release gives no core count, capacity, bandwidth, or performance figure, and no silicon has been demonstrated. Do not fill these cells with ET-SoC-1 numbers — ETSP is an edge/embedded design, not an ET-SoC-1 shrink.*

---

## 1. Compute Engine

### ET-Minion Core (×1,088)

The ET-Minion is Esperanto's key innovation: a minimal in-order RISC-V pipeline extended with a large vector/tensor unit that dominates the die area.

| Parameter | Value |
|-----------|-------|
| Architecture | In-order, fine-grained multithreaded |
| ISA | RV64GCV (RISC-V 64-bit, G base + Vector extensions) |
| Pipeline depth | Minimal (few gates/stage for high MHz at low voltage) |
| Integer peak | 128 INT8 GOPS per GHz per core |
| Chip-wide peak | 100–200 TOPS (sum of all ET-Minions at operating frequency) |

**Vector/Tensor Unit (per core)**

| Feature | Detail |
|---------|--------|
| Tensor instructions | Up to 512 cycles; 32,000 MACs per instruction |
| Vector transcendental unit | Hardware exp(), log(), sin(), cos(), tanh(), sigmoid() — direct activation function support |
| Data types | INT8, FP16, BF16, FP32 |
| Area | Substantially larger than integer pipeline — ML math dominates chip area |

The 32,000-MAC tensor instruction is the key to ET-Minion efficiency: it amortizes instruction fetch, decode, and dispatch overhead across 32K operations. With 1,088 such cores, the aggregate MAC throughput scales with core count, achieving effective TOPS at remarkably low power.

### ET-Maxion Core (×4)

| Parameter | Value |
|-----------|-------|
| Architecture | Quad-issue out-of-order |
| ISA | RV64GCV (same family) |
| Features | Branch prediction, sophisticated prefetch |
| Role | Linux OS, PCIe DMA management, ET-Minion coordination |

### No GPU-style Control Logic

Unlike GPU SMs, ET-Minion cores have:
- No hardware warp scheduler (SPMD parallelism is software-managed)
- No shared memory controller (SRAM is directly addressed)
- No speculative execution beyond the ET-Maxion OoO pipeline

---

## 2. Data Path: Massively Parallel SPMD

The execution model is **SPMD** (Single Program, Multiple Data) over 1,088 independent RISC-V threads:

1. The Glow/LLVM compiler partitions the ML computation graph across 1,088 cores
2. Each ET-Minion receives its own compiled RISC-V ELF binary with per-core weight tiles
3. At runtime, all 1,088 ET-Minions execute independently; synchronization via SRAM barriers where needed
4. Long tensor instructions (512 cycles) allow the integer pipeline to prefetch data while tensor units compute

This is fundamentally different from GPU SIMT: there is no lockstep execution within a warp, no divergence penalty, no shared memory bank conflict protocol. Each core is truly independent.

---

## 3. On-chip Memory

| Parameter | Value |
|-----------|-------|
| Total on-chip SRAM | >160 MB |
| Distribution | Distributed near ET-Minion cores (scratchpad model) |
| Access model | Software-managed; explicit load/store from RISC-V |
| Cache | No hardware caches — SRAM is primary managed storage |
| Role | Weight tiles, activations, intermediate results |

160 MB is large for a <20W chip and enables meaningful working-set residency for DLRM embedding tables and KV-caches of small language models. For large transformers (LLaMA 7B+), weights must tile through LPDDR4x DRAM.

**New on-die memory tier in the successor (2026).** The CORE-ET Silicon Platform adds **MRAM-based on-die memory** alongside (not necessarily replacing) SRAM — see section 9. This is the first non-volatile on-die memory tier in the lineage; the ET-SoC-1 had none. Capacity, bandwidth, and density are **not disclosed**.

---

## 4. Off-chip Memory

| Parameter | Value |
|-----------|-------|
| Memory type | LPDDR4x (Low-Power DRAM) |
| Max capacity | 32 GB per chip |
| Bandwidth | ~137 GB/s per chip (estimated) |
| Storage | eMMC Flash (model weight persistence) |
| Memory access | Standard RISC-V load/store from ET-Minion/Maxion |

**LPDDR4x choice rationale**: The "LP" (low-power) designation aligns with Esperanto's <20W TDP target. LPDDR4x consumes 30–40% less power than GDDR6 or server DDR4 at comparable data rates. The bandwidth tradeoff (137 GB/s vs. HBM's TB/s range) was acceptable for DLRM (capacity-bound) but became a bottleneck for large-model transformer inference — motivating the ET-SoC-2x/3x roadmap's planned HBM adoption.

**Successor (2026):** no off-chip memory type, capacity, or bandwidth is disclosed for the 16nm CORE-ET Silicon Platform. The disclosed memory story for that part is on-die MRAM (section 9); the HBM plan of ET-SoC-2x/3x has **not** been revived.

---

## 5. Host Interface / Package

| Parameter | Value |
|-----------|-------|
| PCIe | Gen4 x8 |
| Chip process | TSMC 7nm |
| Die area | 570 mm² |
| Transistors | 24+ billion |
| Metal layers | 89 |
| TDP | <20W |
| Operating modes | PCIe accelerator mode OR standalone Linux server mode |

The dual operating mode is unique: because the ET-Maxion cores run Linux, an ET-SoC-1 can be provisioned either as a standard PCIe accelerator card (host x86 drives workloads) or as an independent compute node (the chip runs its own inference service). The Glacier Point v2 card fits 6 chips in a standard PCIe form factor.

---

## 6. Scale-up: Glacier Point v2 Card

| Metric | Value |
|--------|-------|
| Chips per card | 6 |
| Total ET-Minion cores | 6,558 |
| Total ET-Maxion cores | 24 |
| Total DRAM | 192 GB LPDDR4x |
| Total DRAM bandwidth | ~822 GB/s |
| Peak INT8 TOPS | ~600–1,200 |
| Inter-chip fabric | Host PCIe mediated (no dedicated chip-to-chip links) |

The absence of a dedicated chip-to-chip interconnect (unlike NVLink, IPU-Link, or Groq's plesiosynchronous fabric) is a design simplification that lowers hardware cost and power but limits inter-chip bandwidth for workloads requiring tight synchronization.

**Successor (2026):** the CORE-ET Silicon Platform introduces an on-die **"scalable network-on-chip (NoC) architecture"** (2026-06-02 press release wording). This is an *intra-die* fabric, not a chip-to-chip interconnect — no scale-up link, topology, width, or bandwidth is disclosed. ET-SoC-1 itself used an Arteris NoC IP integration; the 2026 wording is the first time a NoC is presented as a headline platform feature.

---

## 7. Scale-out

Standard x86 server Ethernet/PCIe for multi-card deployments. No proprietary scale-out fabric. Cloud access was offered via Esperanto's cloud evaluation program.

No scale-out story is disclosed for the CORE-ET Silicon Platform; its stated target is edge/embedded inference rather than multi-node deployment.

---

## 8. Planned Next Generation (not realized)

ET-SoC-2x and ET-SoC-3x were previewed in November 2024 at an NEC partnership announcement:

| Feature | ET-SoC-2x/3x plan |
|---------|-------------------|
| Memory | HBM (replacing LPDDR4x) |
| FP64 | Added (for HPC/supercomputer market) |
| Target | Generative AI transformers + HPC |
| Partners | NEC Corporation (next-gen RISC-V supercomputer roadmap) |
| Status | Not fabricated; company closed July 2025 |

*As of 2026-08-08, no ET-SoC-2x/3x revival appears in any Ainekko or OpenHW Foundation source, and neither HBM nor FP64 is mentioned for the successor platform. Treat ET-SoC-2x/3x as permanently cancelled.*

---

## 9. Successor: CORE-ET Silicon Platform (ETSP) — Ainekko / OpenHW Foundation, 2026

*Added 2026-08-08. Primary sources: Ainekko GlobeNewswire releases 2025-11-19, 2026-01-29, 2026-06-02; github.com/openhwgroup/core-et; openhwfoundation.org.*

After Esperanto closed (July 2025), Ainekko acquired the IP (announced **2025-11-19**) and merged with MRAM startup **Veevx** (**2026-01-29**). The resulting design is published as the **CORE-ET Silicon Platform (ETSP)**, accepted as an **OpenHW Foundation** project on **2026-06-02**.

### 9.1 Disclosed hardware attributes

| Parameter | Value | Source status |
|---|---|---|
| Process | **16nm** node | Press release 2026-06-02; foundry not named |
| Status | **Being taped out** — not sampling, not shipping, not deployed | Press release 2026-06-02 |
| On-die memory | **Integrated MRAM-based memory**; MRAM described as having "SRAM-like performance" | Press releases 2026-06-02 and 2026-01-29 |
| MRAM capacity / bandwidth / density | **Not disclosed** | — |
| Interconnect | **Scalable network-on-chip (NoC) architecture** | Press release 2026-06-02 |
| NoC topology / bandwidth | **Not disclosed** | — |
| Core count | **Not disclosed** for the 16nm part. (The 2025-11-19 acquisition release describes the acquired IP as "more than 1,000 Minion™ cores on a single die"; that describes the Esperanto design, not the 16nm part) | — |
| Peak performance / TOPS / TDP | **Not disclosed** | — |
| SRAM, off-chip DRAM, host interface | **Not disclosed** | — |
| Target | Edge / embedded inference | Press releases; vendor tagline "Model-to-Silicon for Physical AI" is nekko.ai-only |

### 9.2 What the published code actually is

| Item | Detail |
|---|---|
| Repositories | `github.com/openhwgroup/core-et` and `github.com/openhwgroup/core-et-erbium` |
| Created | 2026-04-20 (both); `core-et` pushed 2026-05-07, `core-et-erbium` pushed 2026-06-23 (~4.9 MB, 3 stars) |
| Checked-in license | `core-et/LICENSE` is **verbatim Apache License 2.0** |
| License named in press release | **"Solderpad Hardware License v2.1 ... building on the well-known Apache 2.0 license"** (2026-06-02) — discrepancy with the checked-in file, recorded as-is |
| Governance | OpenHW Foundation (formerly OpenHW Group; openhwgroup.org 302-redirects to openhwfoundation.org), itself "an Eclipse Foundation global initiative". Ainekko is an OpenHW member |

**Scope caveat.** `core-et` is *not* Esperanto's production ET-SoC-1 RTL. It self-describes as "an Ainekko project for collecting hardware IP in a form that can be translated, verified, documented, and integrated by agentic hardware-development workflows," working from "CORE-ET modules into clean SystemVerilog," with the original kept on a separate branch. Esperanto lineage is visible only indirectly (e.g. `docs/minion_vpu_standalone_plan.md` referencing the "real-VPU Minion path"). Neither the 2025-11-19 nor the 2026-06-02 release names ET-SoC-1.

**Naming hygiene.** The MRAM tier brand **"iRAM"** and the name **"Erbium"** appear only on nekko.ai and as a repository name — neither press release uses them. `github.com/ainekko` is an unrelated personal GitHub **User** account (created 2023-08-15, 4 public repos), not the code host.

### 9.3 Architectural delta vs ET-SoC-1

| Dimension | ET-SoC-1 (2021) | CORE-ET Silicon Platform (2026) |
|---|---|---|
| Process | TSMC 7nm | 16nm (foundry not disclosed) — a *deliberate step back* in node, consistent with edge cost targets rather than datacenter density |
| On-die memory | >160 MB SRAM, volatile, SW-managed | SRAM (unspecified) **+ integrated MRAM** — first non-volatile on-die tier in the lineage |
| Off-chip memory | LPDDR4x 32 GB, ~137 GB/s | Not disclosed |
| Fabric | Arteris NoC internally; no chip-to-chip links | "Scalable NoC" as a headline platform feature; still no chip-to-chip fabric disclosed |
| Deployment target | Datacenter DLRM/SLM inference | Edge / embedded inference |
| Availability | Silicon shipped Q1 2022 | Tapeout in progress; no silicon demonstrated |

**Datacenter relevance: effectively zero.** ETSP is retained in this survey as the terminus of the many-core-RISC-V-inference thread, not as a deployable datacenter part.

---

## 10. Key Hardware Metrics Summary

| Metric | ET-SoC-1 | NVIDIA A100 80GB | Cerebras WSE-3 |
|--------|-----------|-----------------|----------------|
| Compute units | 1,088 in-order RISC-V | 6,912 CUDA cores | 900,000 PE cores |
| Process | TSMC 7nm | TSMC 7nm | TSMC 5nm |
| Die area | 570 mm² | 826 mm² | 46,225 mm² (wafer) |
| TDP | <20W | 400W | 23,000W (CS-3 system) |
| Peak INT8 TOPS | 100–200 | 624 | — (PetaFLOPS scale) |
| TOPS/W | ~5–10 | ~1.56 | — |
| On-chip SRAM | >160 MB | ~40 MB (L2+SMEM) | 44 GB |
| Off-chip memory | LPDDR4x 32 GB | HBM2e 80 GB | MemoryX 1.2 PB |
| Off-chip BW | ~137 GB/s | 2 TB/s | — |
| Target | DLRM/SLM inference | Training + inference | LLM training |

---

## Resources

- [IEEE Micro 2022: ET-SoC-1 Chip](https://www.esperanto.ai/wp-content/uploads/2022/05/Dave-IEEE-Micro.pdf)
- [Hot Chips 33 Presentation](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Esperanto.Dave_Ditzel.presentation.v1submitted.pdf)
- [IEEE Xplore: ML Recommendation with 1000+ RISC-V Processors](https://ieeexplore.ieee.org/document/9566904/)
- [WikiChip Deep Dive](https://fuse.wikichip.org/news/4911/a-look-at-the-et-soc-1-esperantos-massively-multi-core-risc-v-approach-to-ai/)
- [RISC-V International Blog](https://riscv.org/blog/accelerating-ml-recommendation-with-over-1000-risc-v-tensor-processors-on-esperantos-et-soc-1-chip-david-r-ditzel-esperanto-technologies-inc/)
- [EE Times: Esperanto Pivots to GenAI](https://www.eetimes.com/esperanto-pivots-to-hpc-and-generative-ai/)
- [HPCwire: NEC + Esperanto HPC Partnership](https://www.hpcwire.com/off-the-wire/esperanto-and-nec-partner-to-advance-next-gen-risc-v-for-hpc/)

### Successor / Post-Acquisition (added 2026-08-08)
- [GlobeNewswire: Ainekko Acquires Esperanto IP (2025-11-19)](https://www.globenewswire.com/news-release/2025/11/19/3191016/0/en/Ainekko-Acquires-Esperanto-Technologies-Intellectual-Property-to-Power-Open-Source-Edge-AI-Platform.html)
- [GlobeNewswire: Ainekko Merges with Veevx — MRAM (2026-01-29)](https://globenewswire.com/news-release/2026/01/29/3228632/0/en/ainekko-merges-with-veevx-expands-open-silicon-platform-with-breakthrough-memory-and-embedded-ai-capabilities.html)
- [GlobeNewswire: Ainekko Edge AI Silicon Platform → OpenHW Foundation project; 16nm tapeout (2026-06-02)](https://globenewswire.com/news-release/2026/06/02/3305225/0/en/ainekko-s-edge-ai-silicon-platform-is-now-an-openhw-foundation-open-source-project.html)
- [github.com/openhwgroup/core-et](https://github.com/openhwgroup/core-et)
- [core-et LICENSE — verbatim Apache-2.0](https://raw.githubusercontent.com/openhwgroup/core-et/main/LICENSE)
- [OpenHW Foundation](https://openhwfoundation.org/)
- [nekko.ai (vendor site — "iRAM", "Erbium", "Model-to-Silicon for Physical AI" appear only here)](https://nekko.ai/)
