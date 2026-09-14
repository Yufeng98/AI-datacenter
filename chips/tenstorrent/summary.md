# Tenstorrent Software and Hardware Stack Summary

*as_of: 2026-09-13*
*chip: tenstorrent*
*device_class: Tensix RISC + SFPU*
*devices: Wormhole (n150/n300/Galaxy Wormhole), Blackhole (p100a/p150a/p150b/Galaxy Blackhole), Quasar (bring-up target in the open-source stack; no public spec sheet)*

---

## Overview

Tenstorrent is a RISC-V AI accelerator company founded by Jim Keller (Intel, AMD, Apple, Tesla CPU lead). Its hardware architecture centers on the **Tensix Processor** — a 2D mesh of **Tensix cores** connected by an explicit Network-on-Chip (NoC) with no hardware cache hierarchy. Every byte that moves between cores or to DRAM must be explicitly requested by software. This is the defining architectural characteristic that distinguishes Tenstorrent from GPU-based accelerators.

The current product lineup (revised 2026-08-08 against the vendor product pages — see the dated update section below):
- **Wormhole** (n150/n300): 72–80 Tensix cores, 12 GB GDDR6, 262–328 TOPS FP8, GFP 12nm
- **Blackhole** (p100a / p150a / p150b PCIe cards): **120 Tensix cores enabled** on all three shipping SKUs (140 is the full-die figure) + 16 Big RISC-V host cores, 28–32 GB GDDR6, **664 TFLOPS BLOCKFP8** per card, PCIe 5.0 x16, 300 W, TSMC 6nm. The dual-die **p300** is no longer listed on the Blackhole product page as of this scan; it is retained in the historical tables below.
- **Blackhole Galaxy**: 6U server with 32 Blackhole ASICs; **23 PFLOPS Block FP8**, 6.2 GB accelerator SRAM @ 2.9 PB/s, 1 TB GDDR6 @ 16 TB/s. **Generally available since 2026-04-28**, list price from $110,000.
- **Quasar**: the next-generation Tenstorrent target being brought up in the open-source stack (tt-llk / tt-metal) since **August 2025**. No public spec sheet, core count, or memory configuration exists; `tt-isa-documentation` still covers only Wormhole B0 and Blackhole A0.

The entire software stack — from ISA documentation to ML framework integration — is **fully open-source** (Apache 2.0), a strategic differentiator from NVIDIA's partially-closed ecosystem.

---

## Software Stack

### Framework Integration

Tenstorrent supports all major ML frameworks through multiple frontend paths:

- **TT-XLA** (`tenstorrent/tt-xla`): PJRT-based bridge; JAX and `torch.compile` → StableHLO → TT-MLIR → hardware. Recommended for new projects.
- **TT-Torch** (`tenstorrent/tt-torch`): PyTorch `torch.export` → StableHLO → TT-MLIR
- **TT-Forge-FE** (`tenstorrent/tt-forge-fe`): ONNX, TensorFlow, and other frameworks → TT-MLIR
- **TT-Forge** (`tenstorrent/tt-forge`): Top-level compiler repo combining all frontends; CI-validates GPT-OSS 120B, Llama 3 70B, Stable Diffusion XL, Whisper large-v3, YOLOv12

Framework access is primarily through standard PyTorch and JAX interfaces — models run unchanged when using `torch.compile` or JAX's JIT with the TT-XLA backend.

### Compiler / IR

- **TT-MLIR** (`tenstorrent/tt-mlir`): MLIR-based compiler framework with Tenstorrent-specific dialects:
  - **TTIR**: High-level tensor ops (hardware-independent; analogous to StableHLO/TOSA)
  - **TTNN**: Models the TT-NN op library API as an MLIR dialect
  - **TTMetal**: Host-side TT-Metalium operations (buffer allocation, program dispatch)
  - **TTKernel**: Compute kernel code representation for code generation
- **Compiler passes**: op fusion, tensor sharding, memory layout assignment (DRAM vs. L1), TTIR → TTNN → TTMetal lowering
- **Compilation path**: `StableHLO → TTIR → TTNN → TTMetal → TT-NN C++ + RISC-V kernel binaries`

### Op Library

- **TT-NN** (in `tenstorrent/tt-metal`, `ttnn/` directory): PyTorch-like Python + C++ neural network operator library
  - ~800+ model variants validated in CI
  - Provides `ttnn.matmul`, `ttnn.linear`, `ttnn.softmax`, `ttnn.conv2d`, `ttnn.attention`, fused ops
  - `MeshDevice` abstraction for multi-chip tensor-parallel operations
  - Tensor layout system: `ROW_MAJOR_LAYOUT`, `TILE_LAYOUT` (native 32×32), height/width/block sharding
  - Built-in CCL (Collective Communication Library) over Ethernet: AllReduce, AllGather, ReduceScatter

### Kernel Library

- **TT-LLK** (`tenstorrent/tt-llk`): Header-only Low Level Kernel library
  - Implements the **Unpack → Math → Pack** pipeline at the hardware intrinsic level
  - Architecture-specific variants: `tt-llk-wh` (Wormhole), `tt-llk-bh` (Blackhole)
  - LLK primitives: `llk_unpack_A`, `llk_math_eltwise_unary`, `llk_pack`, etc.
  - Used internally by TT-Metalium's compute kernel API; seldom called directly
- **Matmul kernel header** (`tt_metal/include/compute_kernel_api/matmul.h`): High-level compute API calling LLK
- **FlashAttention** (`tech_reports/FlashAttention/`): SRAM-resident FlashAttention implementation on Tensix

### Runtime

- **TT-Metalium** (in `tenstorrent/tt-metal`): Low-level, bare-metal SDK for direct hardware programming
  - Key abstractions: `Device`, `Program`, `CircularBuffer`, `KernelHandle`, `Buffer`, `CommandQueue`
  - Three kernel types per Tensix core: **Reader** (BRISC/RISCV_0), **Compute** (TRISC0-2), **Writer** (NCRISC/RISCV_1)
  - Kernels are C++ files JIT-compiled to RISC-V binaries at `CreateKernel()` time
  - **Fast Dispatch**: Lightweight command queue via ARC management core; microsecond dispatch latency
  - **SPMD**: Same kernel program across all targeted Tensix cores with per-core runtime args

The most important Metalium API concept is the **CircularBuffer** — a ring buffer in L1 SRAM that enables lock-free producer-consumer synchronization between the reader, compute, and writer kernels running concurrently on a single Tensix core.

> **Update 2026-08-08.** CircularBuffer is being systematically replaced by the **DataflowBuffer (DFB)** as part of the "Metal 2.0 / Device 2.0" API rearchitecture (underway in tt-metal since at least 2025-11-16, still in progress at v0.75.0). See the dated update section below.
>
> **Update 2026-09-13.** tt-metal v0.77.0 (2026-08-18) and v0.78.0 (2026-09-05) continue the Metal 2.0 migration and add **DeepSeek V4 HCA functional prefill support** and **Kimi-K3 MLA bring-up** at the tt-metal/TT-NN layer — though tt-forge's published model-coverage matrix (through 2026-08-31 dev builds) does not yet list DeepSeek V4. v0.78.0 also adds **Fabric2D and TorusXY topology support for prefill operations** — a software routing/scheduling feature, not new interconnect hardware. See the 2026-09-13 update section below.

### Driver / Firmware

- **TT-KMD** (`tenstorrent/tt-kmd`): Linux kernel module; registers `/dev/tenstorrent/%d`; PCIe BAR mapping, IRQ handling
- **TT-SMI** (`tenstorrent/tt-smi`): Console hardware monitor (temperatures, utilization, power)
- **TT-Installer** (`tenstorrent/tt-installer`): One-command full stack installer via Podman/Docker
- **TT-NPE** (`tenstorrent/tt-npe`): NoC Performance Estimator for kernel optimization

### Communication

- **TT-NN CCL**: Built into TT-NN; AllReduce, AllGather, ReduceScatter, AllToAll over chip-to-chip Ethernet
- **MeshDevice API**: `ttnn.MeshDevice` + `ttnn.ShardTensor2D` for automatic tensor-parallel sharding
- Ethernet tiles appear in the NoC grid — cross-chip NoC operations use the same API as intra-chip operations

### Assembler / ISA

- **TT-ISA-Documentation** (`tenstorrent/tt-isa-documentation`): Complete, publicly available ISA reference
  - Baby RISC-V ISA (Wormhole B0 + Blackhole A0)
  - Matrix Unit (FPU) ISA: `MVMUL`, `GMPOOL`, `ELWMUL` instructions
  - Vector Unit (SFPU) ISA: 32-lane SIMD instruction set
  - Scalar Unit (ThCon) ISA
- No PTX-equivalent virtual ISA — kernels compile directly to native RISC-V binaries per chip generation
- Corsix community series provides deep-dive ISA analysis for Wormhole

---

## Hardware Architecture

### Compute Engine

The **Tensix core** is the fundamental compute unit. Each contains:

| Component | Specification |
|-----------|--------------|
| Baby RISC-V cores | 5: BRISC, NCRISC, TRISC0 (Unpack), TRISC1 (Math), TRISC2 (Pack) |
| L1 SRAM | 1.5 MB (software-managed scratchpad) |
| Matrix unit (FPU) | 32×32 tile matmul; BF16/FP16/FP8(E4M3,E5M2)/INT8 |
| Vector unit (SFPU) | 32-lane SIMD; Exp, Log, Sqrt, Tanh, GeLU, custom |
| NoC routers | 2 (NoC 0 + NoC 1), each 32 bytes/cycle |
| Native tile format | 32×32 |

Wormhole has 80 Tensix cores (72 active in n150 PCIe cards); Blackhole has 140. Blackhole also integrates 16 "Big RISC-V" 64-bit dual-issue cores for on-chip Linux hosting.

### Data Path (NoC)

The chip is organized as a 2D NoC mesh with **wraparound torus topology** and **no hardware cache**:
- **NoC 0** (east+south) + **NoC 1** (west+north): dual unidirectional rings
- **32 bytes/cycle** per link; full torus connectivity
- **All data movement is explicit**: programmer (or compiler) calls `noc_async_read`/`noc_async_write` for all transfers
- DRAM, Ethernet, and PCIe tiles appear as addressable NoC nodes — accessed via the same API

This is the central architectural philosophy: **no implicit data movement, no cache coherence protocol, no hidden state**. Every transfer is software-initiated and software-tracked.

### On-chip Memory

1.5 MB per Tensix core = **210 MB total** on Blackhole (140 cores), **~120 MB** on Wormhole (80 cores). This SRAM is a software-managed scratchpad with no eviction policy. Data persists until explicitly overwritten. Tensor sharding distributes model parameters and activations across the distributed L1 pool for zero-DRAM-traffic operation on hot data.

### Off-chip Memory

Tenstorrent uses **GDDR6** (not HBM):
- **Wormhole**: 12 GB GDDR6, ~336 GB/s, 6 controllers
- **Blackhole**: 32 GB GDDR6, 512 GB/s, 24 controllers

GDDR6 provides lower peak bandwidth than HBM (8 TB/s on B200) but avoids HBM supply chain constraints and CoWoS interposer manufacturing complexity. For inference workloads that can keep hot weights in the distributed on-chip SRAM, GDDR6 bandwidth is sufficient.

### Host Interface / Package

- Wormhole: PCIe 4.0 x16, GFP 12nm, 670 mm²
- Blackhole: PCIe 5.0, TSMC 6nm; standalone mode via 16 Big RISC-V host cores

### Scale-up Interconnect

Tenstorrent uses **standard Ethernet** for chip-to-chip connectivity:
- **Wormhole**: 16 Ethernet tiles × 100 Gbps = 1.6 Tbps per chip
- **Blackhole**: 10 × 400 Gbps = 4 Tbps per chip; 4× QSFP-DD external ports

Ethernet tiles appear in the NoC grid at fixed coordinates. Cross-chip data movement uses the same `noc_async_read/write` NoC API as intra-chip movement — the programmer abstraction is transparent to chip boundaries.

Multi-chip configurations: n300 (2× Wormhole), T3000 (8× Wormhole in 4×2 mesh), Blackhole Galaxy (32× Blackhole in 4×8 mesh).

### Scale-out Interconnect

Standard commodity Ethernet switches (400G/800G Ethernet infrastructure) for rack and multi-rack deployments. No proprietary switching ASIC required. Blackhole Galaxy (GA 2026-04-28): 32 chips, **23 PFLOPS Block FP8**, 1 TB GDDR6 at **16 TB/s memory bandwidth**, an intra-node accelerator fabric of 10× 400 GbE per ASIC (vendor-stated **32 TB/s**, bidirectional math), and up to 56× 800 GbE QSFP-DD of scale-out (vendor-stated **11.2 TB/s**, bidirectional). Vendor states the architecture scales to **144 nodes / >4,000 chips**.

---

## Programming Model Rationale

**1. Explicit data movement enables predictable bandwidth utilization.** The absence of hardware caches means there are no cache miss penalties, no cache thrashing, and no cache coherence traffic. A programmer who correctly models the data access pattern achieves bandwidth close to theoretical DRAM limits. The cost is programming complexity — all data movement must be explicitly orchestrated.

**2. The circular buffer is the key synchronization primitive.** Rather than GPU-style barrier synchronization (`__syncthreads()`), TT-Metalium uses circular buffers as lock-free producer-consumer queues between concurrently-running reader, compute, and writer kernels. This enables natural pipeline overlap: the reader fetches tile N+1 while the compute kernel processes tile N.

**3. The Unpack→Math→Pack pipeline maps directly to hardware.** The three TRISC cores (TRISC0, TRISC1, TRISC2) correspond exactly to the Unpack, Math, and Pack hardware units. A compute kernel is compiled three times — once for each core — and the cores run as a pipeline. This is hardware-level software pipelining made explicit in the programming model.

**4. SPMD with per-core runtime args is the parallelism model.** Like GPU SIMT, the same program runs on all Tensix cores. Unlike GPU SIMT (where threads are warp-scheduled), each Tensix core is an independent MIMD processor with its own 5 RISC-V cores. Parallelism across the mesh is achieved by partitioning tensor data across cores, not by warp-level SIMD within a core.

**5. Ethernet-based scale-out requires no proprietary fabric.** The decision to use standard 100G/400G Ethernet for both on-chip interconnect and rack-scale connectivity eliminates the need for NVSwitch-style proprietary switching. This reduces system cost and increases flexibility but requires the software stack to tolerate Ethernet latency in collective operations.

**6. The fully open stack removes barriers to hardware optimization.** Every layer — ISA, LLK, runtime, op library, compiler — is open-source. Researchers can optimize a custom attention kernel from TRISC instruction level up to TT-MLIR dialect, without reverse-engineering any component. This is the primary ecosystem strategy to compete against CUDA's locked-in ecosystem.

---

## Galaxy Blackhole GA and Stack Update (April–August 2026)

*Updated 2026-08-08. Sources: Tenstorrent newsroom (2026-04-28 GA release; 2026-05-04 TT-Deploy; 2026-06-30 TT-Ascalon S / Japan), tenstorrent.com/en/hardware/galaxy and /blackhole product pages, Galaxy Blackhole User Guide v1.6 (docs.tenstorrent.com, dated 2026-08-04), The Register 2026-04-28, tt-metal v0.75.0 release notes (2026-07-30), GitHub commit search on tenstorrent/tt-metal and tenstorrent/tt-llk.*

Prior-generation content above is retained; this section records what changed and corrects figures that were wrong in the 2026-04-05 baseline.

### 1. Galaxy Blackhole is generally available (2026-04-28)

Tenstorrent's wording is "announces today general availability of Tenstorrent Galaxy Blackhole deployed at scale." **Status should be read as "GA, sold at list price, deployed with named partners"** — not "volume production." Neither the press release nor the product page uses "volume" or any production-status verb, and the widely-repeated "production commenced January 2026" line traces only to an aggregator blog, not to Tenstorrent. The HPCwire/AIwire pickup (2026-05-01) and the TT-Deploy event (2026-05-04) are downstream of the single 2026-04-28 announcement, not separate launches.

**Hardened Galaxy Blackhole system specification** (vendor product page, corroborated by The Register 2026-04-28 and Galaxy Blackhole User Guide v1.6, 2026-08-04):

| Parameter | Value | Baseline value (superseded) |
|---|---|---|
| Form factor | 6U rackmount, air-cooled | — |
| Accelerators | 32 × Blackhole ASIC | 32 (unchanged) |
| Peak compute | **23 PFLOPS Block FP8** | ~25 PFLOPS FP8 (summary), ~24 PFLOPS (hw-arch) |
| Accelerator SRAM | **6.2 GB @ 2.9 PB/s** | ~6.7 GB, bandwidth not recorded |
| Off-chip memory | 1 TB GDDR6 **@ 16 TB/s** | 1 TB, "~16 TB/s aggregate" (unlabeled) |
| Accelerator fabric | 10 × 400 GbE per ASIC — vendor-stated **32 TB/s** | not recorded |
| Scale-out | up to 56 × 800 GbE QSFP-DD — vendor-stated **11.2 TB/s** | "~1 TBps aggregate" (wrong) |
| Host | 1 × AMD EPYC 9004 (Zen 4, ≤32 cores, ≤280 W TDP) | not recorded |
| Host memory | up to 576 GB (6 × 96 GB) DDR5-4800 ECC RDIMM | not recorded |
| Power | **8–10 kW average, 12 kW max; max system power configurable up to 14.5 kW** | not recorded |
| Weight | 262 lbs / 119 kg | not recorded |
| Max system scale | vendor-stated **144 nodes / >4,000 chips** | not recorded |

**Bandwidth accounting caveat.** The 32 TB/s and 11.2 TB/s figures are vendor *bidirectional* math and should be halved for unidirectional comparison: 32 ASICs × 10 × 400 Gb/s = 128 Tb/s = **16 TB/s each way**; 56 × 800 Gb/s = 44.8 Tb/s = **5.6 TB/s each way**. The Register separately quotes ~100 Tbps aggregate intra-node Ethernet, which does not reconcile cleanly with 128 Tb/s; the discrepancy is unexplained in public material.

**Public list pricing** (vendor page, confirmed by The Register) — the first time Tenstorrent has published datacenter system prices:

| SKU | List price |
|---|---|
| Galaxy Blackhole (1 server) | from **$110,000** |
| Galaxy Blackhole 4-system base supercluster | from **$440,000** |
| Galaxy Wormhole (1 server) | from **$70,000** |
| Blackhole p100a PCIe card | $999 |
| Blackhole p150a / p150b PCIe card | $1,399 |

At TT-Deploy (2026-05-04) Tenstorrent described superclusters of **36 Galaxy units** networked as a single computer already in operation.

**GA-named customers and partners:** Prodia, Equinix, OrionVM, BetterBrain, Virtu Financial, Turiyam, Cirrascale, and ai&.

### 2. Shipping Blackhole PCIe SKU correction

The 2026-04-05 baseline recorded "140 Tensix cores / 745 TOPS FP8" for Blackhole. That is the **full-die** figure. The vendor product page lists only three shipping cards, all with 120 Tensix cores enabled:

| SKU | Tensix cores | Big RISC-V | GDDR6 | Memory BW | Peak | Host I/F | TDP | Cooling | External ports | Price |
|---|---|---|---|---|---|---|---|---|---|---|
| p100a | 120 | 16 | 28 GB | **448 GB/s** | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | active | — | $999 |
| p150a | 120 | 16 | 32 GB | 512 GB/s | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | active | 4 × QSFP-DD 800G | $1,399 |
| p150b | 120 | 16 | 32 GB | 512 GB/s | 664 TFLOPS BLOCKFP8 | PCIe 5.0 x16 | 300 W | passive | 4 × QSFP-DD 800G | $1,399 |

**p300** (dual-die, 64 GB) no longer appears on the Blackhole product page. It is kept in the historical tables in `hw-architecture.md`, flagged as delisted.

The Galaxy's 23 PFLOPS across 32 ASICs works out to ~719 TFLOPS per ASIC (repo arithmetic, not vendor-stated) — between the 664 TFLOPS shipping-card figure and the 745 TOPS full-die figure. Tenstorrent has not published the per-ASIC configuration used in Galaxy, so the reconciliation is **not disclosed**.

### 3. Quasar — next-generation target, longer-running than it appears

Quasar is a genuine next-generation Tenstorrent target under active bring-up in the open-source stack, but it is **not** a new-window event:

- First code: `feat: initial commit of Quasar LLK` in `tenstorrent/tt-llk` (PR #593) on **2025-08-15**, pulled into tt-metal on 2025-08-17 — roughly eight months before this repo's 2026-04-05 baseline.
- GitHub commit search returns **662** Quasar-mentioning commits in tt-metal and **85** in tt-llk.
- tt-metal v0.75.0 (2026-07-30) and the August 2026 dev builds add *incremental* items: "Enable trace capture and replay on Quasar", "Enabling DFBs in Quasar fast-dispatch flow", "Add Quasar fast dispatch stress tests", "Fix Quasar unicast and iDMA VC assignments", "make Quasar-unsupported APIs fail at compile time", Quasar SFPU/matmul LLK perf tests.
- **Process node:** Tenstorrent announced in **October 2023** that the Quasar chiplet — then described as a "low-cost, low-power chiplet for machine learning" — would be fabbed on **Samsung Foundry SF4X (4 nm-class)** at Taylor, Texas. Whether the 2025–26 tt-metal "Quasar" architecture target is that same silicon has **not** been restated by Tenstorrent, so treat the node attribution as *medium confidence*.
- Still **not disclosed**: Quasar core count, SRAM/memory configuration, peak throughput, packaging, or any spec sheet. `tt-isa-documentation` contains only `WormholeB0` and `BlackholeA0` top-level directories, and docs.tenstorrent.com/aibs lists only Wormhole and Blackhole cards plus Wormhole/Blackhole QuietBox, LoudBox and Galaxy systems.

### 4. Software: Metal 2.0 / Device 2.0 and the CircularBuffer → DataflowBuffer migration

This is a **genuine gap in the prior repo coverage** rather than a new-window development — the rearchitecture has been underway in tt-metal since at least **2025-11-16** (`[DM]: Porting DM tests to Device 2.0 APIs first pass`, #32121; 168 "Metal 2.0"-mentioning commits) — but neither `summary.md` nor the repo's `METALIUM_GUIDE.md` coverage mentioned it.

Confirmed in tt-metal **v0.75.0** (published 2026-07-30):

| Item | Description |
|---|---|
| **Metal 2.0 API** | Kernel scratchpad support; NOC-distinctness legality check added to the Metal 2.0 API |
| **Device 2.0 API** | Broad TT-NN migration — normalization and reduction ops migrated; CNN ops in progress |
| **DataflowBuffer (DFB)** | Systematic replacement for `CircularBuffer` across eltwise, conv/pool, data-movement and Moreh kernels; DFBs enabled in the Quasar fast-dispatch flow |
| **Multi-config layer support** | New feature in v0.75.0 |
| **emule multichip fiber engine** | Multichip fiber engine + fabric/CCL teleport, demonstrated on an 8-chip LoudBox |

**Architectural significance for this survey:** the repo previously presented `CircularBuffer` as *the* Metalium synchronization primitive (see "Programming Model Rationale" item 2 above). DFB supersedes it in new kernels; both coexist during the migration.

**TT-Lang** is now listed as a software component on docs.tenstorrent.com/aibs and was absent from the repo's stack coverage. Its scope and relationship to TT-Metalium/TT-NN are **not disclosed** in material retrieved for this scan.

### 5. Performance and model-coverage claims (vendor marketing — none independently reproduced)

| Date | Claim | Caveat |
|---|---|---|
| 2026-04-28 (GA) | DeepSeek-R1-0528 671B at **350+ tok/s/user**, batch 8–64, up to 128K context | The Register reported **~300 tok/s/user actually measured**, with 350 only *expected* via software improvements, and flagged that Tenstorrent "doesn't specify the batch size"; also noted prior Tenstorrent testing showed "generally poor performance scaling" |
| 2026-05-04 (TT-Deploy) | Same 350+ figure restated: batch 32 across 16 Galaxies, sub-4 s TTFT at 100K context | Not a new record; same generation of claim as GA |
| 2026-06-30 | DeepSeek-R1-0528 671B at **over 400 tok/s/user**; roadmap target **500 tok/s/user at ~$6/M tokens** | Vendor only |
| 2026-06-30 | Kimi K2.6 at **900 tok/s/user**, "3× faster than GPUs" | Vendor only; comparison GPU unspecified |
| 2026-06-30 | LTX 2.3 Fast: ~**6 s** for a 144-frame 1080p video with audio and lip-sync, "4× faster than GPUs" | Vendor only |
| 2026-04-28 / 2026-05-04 | Prodia video generation: **720p, 81-frame video in 2.4 s**, "10× faster than leading GPU systems" (GA); restated as **2.5 s** at TT-Deploy | The bare "2.5 s" figure circulating in coverage drops the resolution and frame count |
| 2026-05-04 | "Roughly **2.5 million** Hugging Face models running on Tenstorrent" at a **~90%** pass rate | The Register explicitly flags this as awaiting independent verification |

**tt-forge 1.5.0.dev** builds (August 2026) carry DeepSeek-V3.1/V3.2, GLM-4.7, Kimi K2/K2.6, Qwen 2.5/3 (incl. Qwen 3 embeddings), Phi-4 and Falcon-3 entries, with single-device / data-parallel / tensor-parallel matrices on n150, n300 and p150. Reported per-model rates in those builds are modest — e.g. DeepSeek-V3.1 at 2–3 tok/s/user on n150, Falcon-3-10B at 40–41 tok/s/user on p150. There is **no** DeepSeek-V4 entry.

### 6. Adjacent products and deployments

- **TT-Ascalon S** (2026-06-30): RISC-V **CPU IP** for "mixed, branch-heavy, tool-connected" agentic workloads; ~50% the area of TT-Ascalon X at ~140% performance per mm². This is CPU IP, **not** a datacenter accelerator. Process node **not disclosed**.
- **Japan** (2026-06-30): **ai&** deploying **120+ Galaxy systems**, described as the largest scale of sovereign AI compute **in Japan**; Turing demonstrated Blackhole in vehicles; Tenstorrent contributing a RISC-V CPU chiplet to Japan's national **2 nm** program via **Rapidus**; Osaka AI datacenter already active; plan to bring up to 200 Japanese silicon engineers into design teams.
- **Stealthium** runtime-observability partnership (2026-07-30).
- *Out of window:* the Tenstorrent / **Infinia Technologies** sovereign-AI partnership is dated **2026-01-27**, before this repo's baseline.

### 7. Explicitly not confirmed

- **No Tenstorrent talk appears in the Hot Chips 38 (Aug 23–25, 2026) advance program** — confirmed absent, not merely unfound.
- No Tenstorrent submission found in MLPerf Inference v6.0 (2026-04-01) or in MLPerf Training.
- No LG deal confirmed in this scan window.
- Quasar spec sheet, core count, memory configuration: **not disclosed**.
- TT-Ascalon S process node: **not disclosed**.
- Galaxy per-ASIC Blackhole configuration (which reconciles 23 PFLOPS / 32): **not disclosed**.

---

## Software Update — tt-metal v0.77/v0.78, JapanFold, Qualcomm rumor re-check (2026-09-13)

*Updated 2026-09-13. Scan window 2026-08-08 → 2026-09-13. Classification: **Moderate** (tt-metal version bumps)
+ **Minor** (JapanFold) + **Roadmap negative result** (Qualcomm rumor). WebSearch was unavailable this session
(budget exhausted); sourced from direct GitHub releases/tags fetches, the Tenstorrent newsroom, and Bing's
news-search surface as a fallback for the Qualcomm-rumor check.*

**tt-metal v0.77.0 (2026-08-18)** and **v0.78.0 (2026-09-05)** shipped inside the window (plus ongoing
`v0.79.0-dev*` nightly builds through at least 2026-09-13). Headline items: **DeepSeek V4 HCA functional prefill
support** and **Kimi-K3 MLA bring-up** (v0.77.0) — the first DeepSeek V4 reference found in tt-metal, though
tt-forge's own model-coverage builds (through 2026-08-31) still omit it, so treat as "functional at the
tt-metal/TT-NN layer, not yet an end-to-end supported model"; continued Metal 2.0 / Device 2.0 migration; TT-NN
silent-data-corruption fixes (`topk`, `tilize`, `sort`); and **Fabric2D / TorusXY topology support for prefill
operations** (v0.78.0) — a software routing/scheduling feature for prefill workloads, **not new interconnect
hardware**.

**JapanFold (2026-09-03):** ai& and Tenstorrent launched a sovereign drug-discovery platform on ai&'s
Tenstorrent-Galaxy-powered Japan infrastructure — a new named application/customer on the already-tracked ai&
Japan deployment (120+ Galaxy systems, recorded 2026-06-30). No new hardware.

**Qualcomm–Tenstorrent acquisition rumor — re-checked, still unresolved.** The existing "2026-06-16 report... ~$10B
deal... rumor only... deliberately excluded" finding was re-checked (prompted by a hint referencing a similar
2026-07-01 report). The same story cluster (Reuters/Seeking Alpha/Datacenter Dynamics/MSN, "$8 billion to $10
billion," mid-June to early-July 2026) is the only coverage found — **no confirmation, denial, or resolution as of
2026-09-13**. Status unchanged: rumor only, not adopted as fact.

**"Grendel" naming — checked.** SemiAnalysis's "Blackhole, Grendel and..." newsletter describes Grendel as an
earlier-disclosed third-generation-chip codename (TSMC 4nm, 64 in-house RISC-V cores, 16×400G Ethernet, tape-out
originally ~2023–24). No 2025/2026 primary source uses the name; all current next-gen bring-up activity uses
**Quasar** exclusively. Working assessment: Grendel is most plausibly an earlier/retired codename for what is now
called Quasar, or a distinct plan that did not proceed as named — **not confirmed either way**.

**Checked, no change found:** no Tenstorrent talk in Hot Chips 38 (occurred within this window; not re-verified
against a post-event archive this cycle); no MLPerf submission; no new Galaxy/Blackhole/Quasar hardware spec; no
LG deal.

Sources: https://github.com/tenstorrent/tt-metal/releases/tag/v0.77.0 · https://github.com/tenstorrent/tt-metal/releases/tag/v0.78.0 · https://tenstorrent.com/en/newsroom · https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and

---

## Resources

### Documentation
- [Understanding the Tenstorrent Software Stack](https://docs.tenstorrent.com/getting-started/tt-software-stack.html)
- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- [TT-MLIR Documentation](https://docs.tenstorrent.com/tt-mlir/)
- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Wormhole Specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)

### Open-Source Repositories
- [tt-metal](https://github.com/tenstorrent/tt-metal) — TT-Metalium + TT-NN
- [tt-forge](https://github.com/tenstorrent/tt-forge) — End-to-end compiler
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) — MLIR compiler framework
- [tt-xla](https://github.com/tenstorrent/tt-xla) — JAX / torch.compile PJRT bridge
- [tt-llk](https://github.com/tenstorrent/tt-llk) — Low-level kernel library
- [tt-kmd](https://github.com/tenstorrent/tt-kmd) — Linux kernel driver
- [tt-npe](https://github.com/tenstorrent/tt-npe) — NoC performance estimator
- [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) — ISA reference

### Hardware
- [Wormhole Product Page](https://tenstorrent.com/en/hardware/wormhole)
- [Blackhole Product Page](https://tenstorrent.com/en/hardware/blackhole)
- [Galaxy Product Page](https://tenstorrent.com/en/hardware/galaxy) — Galaxy Blackhole / Galaxy Wormhole specs and list pricing
- [Galaxy Blackhole System Docs](https://docs.tenstorrent.com/systems/galaxy-blackhole/index.html) — User Guide v1.6 (2026-08-04)
- [Tensix Neo IP Page](https://tenstorrent.com/en/ip/tensix-neo)

### Newsroom (April–August 2026)
- [Tenstorrent Enables AI at Scale (Galaxy Blackhole GA, 2026-04-28)](https://tenstorrent.com/en/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance)
- [TT-Deploy (2026-05-04)](https://tenstorrent.com/en/newsroom/tt-deploy)
- [Tenstorrent Sets New Performance Records, Launches TT-Ascalon S (2026-06-30)](https://tenstorrent.com/newsroom/tenstorrent-sets-new-performance-records-launches-tt--ascalon-s)
- [The Register: Tenstorrent's Galaxy Blackhole AI servers are finally out (2026-04-28)](https://www.theregister.com/software/2026/04/28/tenstorrents-galaxy-blackhole-ai-servers-are-finally-out/5229759)
- [HPCwire/AIwire: Galaxy Blackhole GA (2026-05-01)](https://www.hpcwire.com/aiwire/2026/05/01/tenstorrent-announces-general-availability-of-galaxy-blackhole-ai-system/)
- [The Register: Samsung to fab RISC-V chips for Tenstorrent (2023-10-04)](https://www.theregister.com/on-prem/2023/10/04/samsung-to-fab-risc-v-chips-for-tenstorrent/) — Quasar chiplet on Samsung SF4X

### Newsroom / releases (added 2026-09-13)
- [ai& and Tenstorrent Launch JapanFold (2026-09-03)](https://tenstorrent.com/en/newsroom)
- [tt-metal v0.77.0 release (2026-08-18)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.77.0)
- [tt-metal v0.78.0 release (2026-09-05)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.78.0)
- [tt-forge releases (1.5.0.dev builds through 2026-08-31)](https://api.github.com/repos/tenstorrent/tt-forge/releases?per_page=5)

### Analysis
- [HotChips 2024: Blackhole & TT-Metalium](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf)
- [ASPLOS 2025: Dissecting Blackhole](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [SemiAnalysis: Wormhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-wormhole-analysis-a-scale)
- [SemiAnalysis: Blackhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and)
- [Corsix: Wormhole Deep-Dives](https://www.corsix.org/content/tt-wh-part1)
