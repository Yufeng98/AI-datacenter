# Intel Gaudi 3 — Hardware Architecture

*as_of: 2026-09-13*
*chip: intel-gaudi*
*sources: Gaudi 3 White Paper (Intel, 2024), Hot Chips 2024, docs.habana.ai v1.24.0; Crescent Island: Intel Hot Chips 38 talk "Crescent Island: GPU Designed for Agentic AI Inference" (2026-08-24, via ServeTheHome and Chips and Cheese coverage) — confirmed architecture; Computex 2026 trade coverage (2026-06-01/02) for items not restated at Hot Chips 38*

---

## Overview

Intel Gaudi 3 (HL-325L) is a dual-die AI accelerator integrating heterogeneous compute (MME + TPC), high-bandwidth on-die SRAM, HBM2e, and 24 on-die RoCE v2 NIC ports on a single package. The design philosophy is that networking, compute, and memory are co-equal first-class resources — each die dedicates significant area to NIC SerDes alongside compute engines.

---

## Compute Engine (Dual Engine: MME + TPC)

### Matrix Multiplication Engine (MME)

| Property | Gaudi 3 |
|----------|---------|
| MME count | 8 (4 per die) |
| Supported ops | GEMM, batched-GEMM, convolution, attention dot-products |
| Peak throughput | 1835 BF16 TFLOPs (OAM) |
| Tile management | SynapseAI compiler (static) |
| Precision | BF16, FP8 (via TPC-assisted conversion) |

The MME is a fixed-function engine: it accepts GEMM descriptor packets from the SynapseAI compiler specifying tile sizes, strides, precisions, and operand addresses in SRAM. It does not execute other operation types.

### Tensor Processing Core (TPC)

| Property | Gaudi 3 |
|----------|---------|
| TPC count | 64 (32 per die) |
| Vector width | 256 bytes / cycle |
| VLIW slots | 4: Vector (SIMD ALU), Scalar, Load, Store |
| Addressing | 5D tensor addressing in Load/Store AGU |
| Precision | FP32, BF16, FP16, FP8 (E4M3, E5M2), INT32, INT16, INT8 |
| Programming | TPC-C (C99 + intrinsics), compiled by tpc-clang (LLVM fork) |

Each TPC is independently programmable. The SynapseAI compiler assigns operator kernels from the 1400+ TPC Kernel Library to TPC subsets. Custom kernels (TPC-C, open-source LLVM toolchain) can replace or augment library kernels.

### Crescent Island — Xe3P compute engine (disclosed Hot Chips 38, 2026-08-24)

| Property | Crescent Island |
|----------|-----------------|
| GPU architecture | Xe3P |
| Xe core count | 32 |
| XMX (matrix) engines | 256 (32 Xe cores × 8 XMX each) |
| XMX design | "3-way extended Xe matrix"; FP4 precision co-issue; FP64 support |
| Systolic array depth | 16-deep |
| Per-core register file (GRF) | 1 MB |
| Per-core L1 | 512 KB |
| L2 cache | 32 MB unified |
| Datatypes | FP4 and MXFP4 through FP64 |
| Peak throughput (any datatype) | Not Intel-disclosed. Chips and Cheese **estimates** (assumed 2.5 GHz clock, quadrupled XMX rate vs. Xe2/Xe3): FP64 vector 10.2 TFLOP/s, FP32 vector 20.5 TFLOP/s, FP16 vector ~41 TFLOP/s, TF32 XMX 328 TFLOP/s, FP16/BF16 XMX 655 TFLOP/s, FP8 XMX 1.3 PFLOP/s, FP4/MXFP4 XMX 2.6 PFLOP/s |

This is Crescent Island's first architectural (not just capacity/power) disclosure — replacing the "not disclosed" placeholder carried since the 2026-06 Computex spec sheet. Crescent Island has no MME/TPC equivalent: the Xe3P XMX engines are a fixed-function-plus-programmable GPU matrix path, not a dual-engine dataflow design like Gaudi's MME+TPC split.

*Source: Intel Hot Chips 38 talk "Crescent Island: GPU Designed for Agentic AI Inference" (Sumit Mohan, Hong Jiang), 2026-08-24, per ServeTheHome; TFLOP/s figures are Chips and Cheese estimates, 2026-08-27, not Intel-disclosed.*

---

## Data Path

The canonical execution pipeline:

```
HBM2e
  ↕  (async DMA, double-buffering)
SRAM (96 MB, 12.8 TB/s)
  ↕              ↕
MME engines    TPC engines
(GEMM →SRAM)   (read SRAM, write SRAM)
```

- **MME → SRAM → TPC**: MME writes GEMM output to SRAM; TPC reads for post-GEMM ops (e.g., activation functions). This producer-consumer pipeline keeps intermediate activations in SRAM without HBM2e round-trips.
- **DMA double-buffering**: While one SRAM buffer is consumed by compute, another is filled by DMA from HBM2e — overlapping memory movement with computation.
- **NIC parallelism**: Integrated RoCE NIC engines operate independently from MME and TPC; during AllReduce, NICs perform RDMA on HBM2e concurrently with the next forward/backward pass computation.
- **All-parallel execution**: MME, TPC, DMA, and NIC engines each have independent command queues and can execute simultaneously when graph-scheduled appropriately.

---

## On-chip Memory (SRAM)

| Property | Gaudi 3 |
|----------|---------|
| Total capacity | 96 MB |
| Per-die capacity | 48 MB |
| Bandwidth | 12.8 TB/s aggregate |
| Management | Software-managed (SynapseAI compiler) — no hardware prefetch |
| Users | MME engines + TPC engines (shared per die) |
| Role | Activation buffer, operand staging, GEMM tile prefetch |

SRAM bandwidth (12.8 TB/s) exceeds HBM2e bandwidth (3.7 TB/s) by 3.5×. The compiler's SRAM placement strategy is the primary performance lever: operators whose working sets fit in SRAM run at 3.5× the effective memory bandwidth of HBM-backed operations.

TPC processors also have small per-TPC local scratchpad memory in addition to shared SRAM access.

---

## Off-chip Memory (HBM2e)

| Property | Gaudi 3 | Gaudi 2 |
|----------|---------|---------|
| Capacity | 128 GB | 96 GB |
| Stacks | 8 × HBM2e | 6 × HBM2e |
| Bandwidth | 3.7 TB/s | 2.45 TB/s |
| BW improvement | — | ~51% |

HBM2e stores model weights, optimizer states, full-size activations that overflow SRAM, and gradient tensors. It is directly accessible via RDMA by the integrated RoCE NICs for collective communication operations, bypassing the host CPU entirely.

### Crescent Island — LPDDR5X (successor product, sampling H2 2026)

Crescent Island breaks the Gaudi memory lineage: it uses **LPDDR5X rather than HBM**, trading peak bandwidth for capacity per dollar in inference serving.

| Property | Crescent Island |
|----------|-----------------|
| Memory type | LPDDR5X (not HBM) |
| Capacity — reference design | 160 GB |
| Capacity — partner/ODM configurations | up to 480 GB |
| Bandwidth | **Still not Intel-disclosed as of Hot Chips 38 (2026-08-24)** — ServeTheHome notes Intel explicitly withheld this figure. Chips and Cheese now **estimates over 1.5 TB/s**, reverse-engineered from PCB photos (20 LPDDR5X modules — 12 front, 8 back — on an assumed 1280-bit bus at LPDDR5X-9600); this analyst estimate **supersedes** the earlier, much lower Computex-era press guess of ~0.6–0.7 TB/s ("~684 GB/s"). Neither figure is an Intel number |
| Module organization | Chips and Cheese's photo-based count: **20 LPDDR5X modules** (12 front + 8 back) on the reference 160 GB card, implying ~8 GB/module — analyst-derived, not Intel-stated. (The previously-excluded "20 × 24 GB" breakdown would only reconcile with the 480 GB ODM configuration, not the 160 GB reference design.) |
| On-die cache | **32 MB unified L2** (Intel-disclosed, Hot Chips 38) |
| RDMA/NIC access to memory | Not disclosed; no on-die RoCE equivalent has been announced |

*Source class: memory type/capacity are Intel-disclosed (Computex 2026); L2 size is Intel-disclosed (Hot Chips 38, 2026-08-24). Bandwidth and module organization remain analyst estimates (Chips and Cheese, 2026-08-27), not Intel figures.*

---

## Host Interface / Package

| Property | Gaudi 3 | Gaudi 2 |
|----------|---------|---------|
| PCIe generation | Gen5 x16 | Gen4 x16 |
| Host BW (bidir) | ~128 GB/s | ~64 GB/s |
| Form factor | OAM (HL-325L) + PCIe (HL-338) | OAM + PCIe |
| Cards per tray | 8 (OAM) | 8 (OAM) |
| Package | Dual-die (2 compute dies) | Single die |
| TDP (OAM) | 900 W (air) / 1200 W (liquid) | 600 W |

The dual-die package (Gaudi 3) connects two compute dies on-package, each with 4 MMEs + 32 TPCs + 48 MB SRAM. The HBM2e stacks and NIC SerDes are shared across both dies via on-package interconnect. The PCIe Gen5 upgrade doubles host bandwidth, reducing bottlenecks for large batch loading and parameter server patterns.

### Crescent Island — packaging and power (successor product)

| Property | Crescent Island |
|----------|-----------------|
| Form factor | PCIe add-in card (no OAM/mezzanine variant announced) |
| PCIe generation | **PCIe Gen5 x16** — Intel-disclosed at Hot Chips 38 (2026-08-24), resolving the prior "not disclosed" |
| Power | **350 W**, air-cooled PCIe GPU — reaffirmed at Hot Chips 38 |
| Active-idle power | **50 W or less** (G0 state) — new disclosure, Hot Chips 38 |
| Low-power idle | ~10 W (G8 state) — new disclosure, Hot Chips 38 |
| Cooling | **Air-cooled**, explicitly not liquid-cooled; targets standard enterprise servers |
| Reliability | ECC + parity across key memory; error checking on every hop of the internal IP fabric; dynamic page offlining; hard post-package repair — new disclosure, Hot Chips 38 |
| Die/package configuration | Still not disclosed (die count, tile count, process node all unstated) |

The 350 W air-cooled PCIe envelope is the sharpest architectural contrast with Gaudi 3 (900 W air / 1200 W liquid OAM): Crescent Island is designed to drop into existing air-cooled enterprise racks rather than to require an OAM baseboard and liquid loop. The active-idle (50 W) and reliability disclosures are new at Hot Chips 38 and read as inference-fleet-economics and RAS messaging aimed at "agentic AI inference" duty cycles.

*Source class: PCIe generation, idle power, and reliability features are Intel-disclosed (Hot Chips 38, 2026-08-24, via ServeTheHome). Power/cooling reaffirm Computex 2026 trade coverage (2026-06-01/02).*

---

## Scale-up Interconnect (Integrated RoCE)

| Property | Gaudi 3 |
|----------|---------|
| Port count | 21 × 200 GbE |
| Protocol | RoCE v2 (RDMA over Converged Ethernet) |
| Implementation | On-die SerDes (48 pairs, 112 Gbps PAM4 Tx/Rx) |
| Peak BW/card | 4.2 Tbps unidirectional |
| Connectivity | Card-to-card (direct) or via standard Ethernet switches |
| RDMA target | HBM2e physical addresses |
| Switch requirement | Standard 200G/400G Ethernet (no proprietary ASIC required) |

The 21 scale-up ports are the primary intra-node and intra-rack communication fabric. Within an 8-Gaudi server tray, ports can be cabled in a direct mesh or connected through a standard Ethernet ToR switch. No NVLink-equivalent proprietary switching fabric is required.

---

## Scale-out Interconnect

| Property | Gaudi 3 |
|----------|---------|
| Port count | 3 × 200 GbE |
| Protocol | RoCE v2 (same as scale-up) |
| Implementation | On-die SerDes (shared technology with scale-up) |
| Target | External Ethernet switches (Cisco Nexus, Arista, etc.) |
| Fallback | Host NIC with PCIe peer-direct GDR |

The 3 scale-out ports connect to the external cluster Ethernet fabric for cross-rack multi-node training. HCCL routes inter-rack AllReduce traffic through these ports; the application sees no API difference between intra-node and inter-node collectives.

---

## Generation Comparison

| Specification | Gaudi 1 | Gaudi 2 | Gaudi 3 | Crescent Island | Jaguar Shores |
|---|---|---|---|---|---|
| Die count | 1 | 1 | 2 (dual-die) | Not disclosed | Multi-tile (quad-tile rumored) |
| Architecture | Gaudi ASIC | Gaudi ASIC | Gaudi ASIC | Xe3P GPU: 32 Xe cores, 256 XMX engines, 16-deep systolic array | Gaudi-branded, 18A |
| MME engines | 2 | 2 | 8 | — (XMX units instead) | Not disclosed |
| TPC engines | 8 | 24 | 64 | — (GPU shaders) | Not disclosed |
| SRAM / cache | ~32 MB | 48 MB | 96 MB @ 12.8 TB/s | 32 MB unified L2 + 1 MB GRF/512 KB L1 per Xe core (Hot Chips 38) | Not disclosed |
| Memory | 32 GB HBM2 | 96 GB HBM2e @ 2.45 TB/s | 128 GB HBM2e @ 3.7 TB/s | 160 GB LPDDR5X reference; up to 480 GB LPDDR5X (partner/ODM configs) | HBM4 (SK Hynix); HBM4E rumored |
| Memory bandwidth | — | 2.45 TB/s | 3.7 TB/s | **Still not Intel-disclosed** (Chips and Cheese estimate: >1.5 TB/s, up from an earlier ~0.6–0.7 TB/s press guess — neither is an Intel figure) | Not disclosed |
| BF16/FP16 TFLOPs | ~95 | ~432 | ~1835 | Not Intel-disclosed; Chips and Cheese estimate ~655 TFLOP/s (XMX) | Not disclosed |
| FP8 TFLOPs | — | — | ~1835 | Not Intel-disclosed; Chips and Cheese estimate ~1.3 PFLOP/s (XMX) | Not disclosed |
| FP4 TFLOPs | — | — | — | Not Intel-disclosed; Chips and Cheese estimate ~2.6 PFLOP/s (XMX) | Not disclosed |
| Datatypes | BF16/FP32 | BF16/FP32/FP16/INT8 | FP32/BF16/FP16/FP8 (E4M3,E5M2)/INT32/INT16/INT8 | FP4/MXFP4 through FP64 (Intel-confirmed, Hot Chips 38) | Not disclosed |
| NIC ports (scale-up) | 10 × 100 GbE | 21 × 100 GbE | 21 × 200 GbE | Not disclosed | Silicon photonics (optical) |
| NIC ports (scale-out) | 0 (external) | 3 × 100 GbE | 3 × 200 GbE | Not disclosed | Silicon photonics (optical) |
| PCIe | Gen4 | Gen4 | Gen5 x16 | **PCIe Gen5 x16** (Hot Chips 38 disclosure) | Not disclosed |
| TDP | ~300 W (OAM, air) | ~600 W (OAM, air) | ~900 W air / 1200 W liquid (OAM) | **350 W air-cooled PCIe AIC**; 50 W active-idle (G0), ~10 W low-power idle (G8) (Hot Chips 38) | Not disclosed |
| Process node | TSMC | TSMC | TSMC | Not disclosed (Xe3P microarchitecture; fab/node not stated) | Intel 18A |
| Software path | SynapseAI | SynapseAI | SynapseAI (graph compiler + TPC-C + HCCL) | oneAPI/SYCL + Level Zero Compute Runtime (**not** SynapseAI) | Not disclosed |
| Status | Legacy | Legacy | GA (shipping) | Customer sampling H2 2026; GA 2027 (unrestated at Hot Chips 38) | ~2027 (pre-announcement) |

*Crescent Island column: architecture (32 Xe cores/256 XMX engines, 16-deep systolic array, 32 MB L2, 1 MB GRF + 512 KB L1/core), PCIe Gen5 x16, active-idle power and reliability features are Intel-disclosed at the Hot Chips 38 talk "Crescent Island: GPU Designed for Agentic AI Inference" (Sumit Mohan, Hong Jiang), 2026-08-24, per ServeTheHome coverage. Memory/power figures not restated at Hot Chips 38 (480 GB / 350 W / FP4–FP64 / GA-2027) still trace to Computex 2026 trade coverage (2026-06-01/02). TFLOP/s and memory-bandwidth figures are Chips and Cheese analyst **estimates** (2026-08-27), not Intel numbers — see the Compute Engine and Off-chip Memory sections above.*

---

## Next-Generation Roadmap

### Falcon Shores — Cancelled (January 2025)

Falcon Shores was Intel's planned Gaudi 3 successor: a multi-chiplet hybrid combining Xe-HPC GPU chiplets with Gaudi ASIC elements, targeting late 2025 launch. Intel cancelled it following the Q4 2024 earnings call, citing customer feedback and inability to compete meaningfully in cloud AI data center markets. It was demoted to an internal test chip only and will never ship commercially.

### Crescent Island — Inference GPU (sampling H2 2026)

Crescent Island is an inference-optimized data center GPU announced at OCP Global Summit in October 2025 and given a fuller spec sheet at Computex 2026 (2026-06-01/02), then given its first Intel architectural disclosure at Hot Chips 38 (2026-08-24). It represents a break from the Gaudi ASIC lineage — it is a GPU based on the Xe3P (performance-optimized Xe3) microarchitecture, the same architecture used in Panther Lake mobile CPUs, built around **32 Xe cores / 256 XMX engines** on a 16-deep systolic array with 32 MB unified L2 and 1 MB GRF + 512 KB L1 per core. Key design choice: LPDDR5X prioritizes memory capacity per dollar over peak bandwidth, targeting "tokens-as-a-service" inference providers — 160 GB in the reference design and **up to 480 GB in partner/ODM configurations**. It is a **350 W air-cooled PCIe Gen5 x16 add-in card** (50 W active-idle), and the confirmed datatype range is **FP4/MXFP4 through FP64** (FP64 would put the part in scientific/HPC reach rather than pure inference). Memory bandwidth, process node and die/package configuration remain **not disclosed** by Intel; Chips and Cheese estimates bandwidth over 1.5 TB/s and XMX throughput at 655 TFLOP/s (BF16), 1.3 PFLOP/s (FP8) and 2.6 PFLOP/s (FP4) — analyst estimates, not Intel figures. Customer sampling H2 2026; general availability 2027 (unrestated at Hot Chips 38). Software path is oneAPI/SYCL over Level Zero, not SynapseAI. *Architecture, PCIe generation and idle-power figures are Intel-disclosed (Hot Chips 38, 2026-08-24); capacity/power/datatype-range figures not restated there still trace to Computex 2026 trade coverage (medium confidence).*

### Jaguar Shores — Rack-scale Gaudi successor (~2027)

Jaguar Shores carries the Gaudi brand and is designed as a rack-scale disaggregated AI platform built on Intel 18A (gate-all-around) process with HBM4 memory from SK Hynix (HBM4E also rumored). Package footprint reported at ~92.5 mm × 92.5 mm with quad-tile configuration and octal HBM stacks. It pairs with Diamond Rapids Xeon CPUs and uses silicon photonics for optical inter-rack interconnects — replacing Gaudi 3's copper RoCE approach at scale. Expected earliest availability: late 2027. Detailed specs have not been officially disclosed. **No new primary-source information appeared in the 2026-04-05 → 2026-08-08 window; all timing detail remains LOW confidence and Jaguar Shores has no Hot Chips 38 slot.**

---

## Update — 2026-08-08

*Window covered: 2026-04-05 → 2026-08-08.*

**Gaudi 3 silicon is unchanged.** No new Gaudi hardware disclosure, no revised Gaudi 3 specification, and no Intel first-party end-of-life notice appeared in this window. All Gaudi 1/2/3 figures above stand as previously recorded.

What changed in this revision:

1. **Crescent Island specification expanded** (Computex 2026, 2026-06-01/02, trade-press-sourced): memory capacity ceiling raised from a flat "160 GB LPDDR5X" to "160 GB reference / up to 480 GB partner configurations"; first-ever power figure (350 W, air-cooled PCIe add-in card); datatype range widened from "FP8/FP4/MXFP4/MXFP8" to "FP4 through FP64"; general availability dated to 2027. Recorded in the Generation Comparison table and in new Crescent Island subsections under Off-chip Memory and Host Interface / Package.
2. **Memory bandwidth deliberately recorded as "not disclosed."** The widely circulated "~684 GB/s" figure is a press back-calculation from an assumed LPDDR5X bus width, not an Intel statement — TechSpot writes that bandwidth "is expected to reach 684 GB/s", and other coverage notes explicitly that Intel gave capacity, TDP and architecture but no headline bandwidth number. It is not entered as a spec anywhere in this repo. The "20 × 24 GB module" breakdown is likewise a press reconstruction and is excluded.
3. **Confidence marking.** Intel's own Computex 2026 press kit and AI newsroom posts contain no mention of Crescent Island. Corroboration across Tom's Hardware, VideoCardz, TechSpot, TechTimes, Neowin and Wccftech is broad but single-origin (coverage of one Intel briefing/slide). All Crescent Island Computex figures are marked medium confidence.
4. **Software path recorded as an architectural fact.** Crescent Island does not inherit SynapseAI/HCCL; it runs Intel's unified Xe stack (oneAPI/SYCL over the Level Zero Compute Runtime). Enablement landed upstream ahead of silicon: Intel Compute Runtime 26.01.36711.4 (2026-01-14), Intel Graphics Compiler v2.27.10 (Jan 2026), Compute Runtime support promoted 2026-04-20.
5. **Driver status correction (software, recorded here for completeness).** Mainline `drivers/accel/habanalabs/` has `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/` and **no `gaudi3/`** as of 2026-08-08; the Gaudi 3 driver pull request targeting v6.19 (dri-devel, Dec 2025) was rejected and an out-of-tree driver is still required for Gaudi 3.

**Pending disclosure:** Intel, "Crescent Island: GPU Designed for Agentic AI Inference" (Sumit Mohan, Hong Jiang) — disclosure scheduled, Hot Chips 38, Mon 2026-08-24, 4:45–6:45 PM, Day-1 GPU session; content not yet public. Expected to settle TFLOPS, Xe3P core/XMX counts, process node and real memory bandwidth. Re-scan after 2026-08-25.

**Sources for this update:** https://acceleratedcomputing.ai/news/2026-06-01-intel-crescent-island/ · https://codeoxi.com/blog/intel-crescent-island-gpu · https://videocardz.com/newz/intel-crescent-island-gpu-officially-supports-up-to-480gb-lpddr5x-memory · https://www.techspot.com/news/112608-intel-crescent-island-gpu-support-up-480gb-lpddr5x.html · https://newsroom.intel.com/press-kit/press-kit-intel-at-computex-2026 · https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs · https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html · https://www.phoronix.com/news/Intel-CR-26.01.36711.4 · https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler · https://hotchips.org/program/conference/

---

## Update — 2026-09-13

*Window covered: 2026-08-08 → 2026-09-13.* **Gaudi 1/2/3 silicon is unchanged.** The pending Hot Chips 38 Crescent Island talk flagged in the prior update has now happened (2026-08-24) and is reflected throughout the Compute Engine, Off-chip Memory, Host Interface/Package, and Generation Comparison sections above. Summary of what changed:

1. **Crescent Island architecture disclosed for the first time**: Xe3P, 32 Xe cores feeding 256 XMX engines on a 16-deep systolic array, 1 MB GRF + 512 KB L1 per core, 32 MB unified L2. This was previously entirely "not disclosed."
2. **PCIe generation confirmed**: PCIe Gen5 x16 (previously "not disclosed").
3. **New power disclosures**: 50 W or less active-idle (G0), ~10 W low-power idle (G8), plus RAS features (ECC/parity, per-hop fabric error checking, dynamic page offlining, hard post-package repair). The 160 GB/480 GB capacity, 350 W TDP, air-cooling and FP4-through-FP64 datatype range are unchanged from Computex 2026 but are now reinforced by an Intel-hosted conference talk rather than resting solely on trade coverage of a slide.
4. **Memory bandwidth remains undisclosed by Intel.** Chips and Cheese now estimates over 1.5 TB/s (from a photo-derived 20-module, 1280-bit-bus, LPDDR5X-9600 reconstruction) — up from an earlier, much lower ~0.6–0.7 TB/s press guess. Both are analyst estimates, not Intel numbers.
5. **New analyst TFLOP/s estimates** (Chips and Cheese, assumed 2.5 GHz clock): FP64 vector 10.2, FP32 vector 20.5, FP16 vector ~41, TF32 XMX 328, FP16/BF16 XMX 655, FP8 XMX 1,300, FP4/MXFP4 XMX 2,600 (all TFLOP/s except the last two, which are PFLOP/s-scale as noted in the tables above). None of these are Intel-disclosed.
6. **Sampling/GA schedule (customer sampling H2 2026, GA 2027) was not restated** at Hot Chips 38 — it remains a Computex-era, medium-confidence figure.
7. **Process node, die/package configuration and NIC/scale-up architecture remain undisclosed.**

**Sources for this update:** https://www.servethehome.com/intel-crescent-island-160gb-to-480gb-lpddr5x-ai-gpu-at-hot-chips-2026/ · https://chipsandcheese.com/p/hot-chips-2026-intels-crescent-island
