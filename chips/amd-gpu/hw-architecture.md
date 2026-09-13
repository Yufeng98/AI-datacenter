# AMD GPU Hardware Architecture

*as_of: 2026-08-08*
*Architectures: CDNA3 (MI300X/MI300A/MI325X), CDNA4 (MI350X/MI355X/MI350P) and CDNA 5 (MI455X/MI430X)*

---

## Overview

AMD Instinct datacenter GPUs are organized around arrays of Compute Units (CUs), each an independent SIMD processor, packaged in a multi-chiplet MCM (Multi-Chip Module) design with XCDs (Accelerator Complex Dies) 3D-stacked on IODs (I/O Dies) via SoIC hybrid bonding, all sitting on a CoWoS silicon interposer alongside HBM stacks. CUs connect through a multi-level memory hierarchy (LDS, L1, L2, Infinity Cache, HBM) and Infinity Fabric interconnect (XGMI for GPU-to-GPU). This document covers all 7 hardware layers for the CDNA3, CDNA4 and CDNA 5 generations.

**CDNA 5 (MI455X, launched 2026-07-23) breaks three of the invariants above**: the shader building block is described as a **WGP** with four dual-issue **Wave32** SIMD32 units (not Wave64 over SIMD16); the separate Infinity Cache tier is replaced by a **192 MB global L2 on two Fabric-and-Cache Dies**; and GPU-to-GPU scale-up moves from an 8-GPU XGMI mesh to a switched **72-GPU UALink-over-Ethernet domain**. Per-generation detail is carried in each layer section below, with the full narrative in the [CDNA 5 section](#cdna-5--instinct-mi455x--helios-2026).

---

## Generation Overview

| Generation | Lead SKU | Shader block | Blocks/chip | Wave width | Process (compute / IO) | HBM | HBM BW | Last-level on-package cache | Scale-up | Peak low-precision |
|---|---|---|---|---|---|---|---|---|---|---|
| CDNA3 | MI300X | CU (4x SIMD16 + 1 Matrix Core) | 304 CUs (8 XCDs x 38) | 64 | TSMC N5 / N6 | HBM3 192 GB | 5.3 TB/s | 256 MB Infinity Cache (4 IODs) | XGMI, 7 links, 8-GPU mesh | ~2.6 PFLOPS FP8 |
| CDNA3 refresh | MI325X | CU | 304 CUs | 64 | TSMC N5 / N6 | HBM3e 288 GB | ~6.0 TB/s | 256 MB Infinity Cache | XGMI, 8-GPU mesh | ~2.6 PFLOPS FP8 |
| CDNA4 | MI350X / MI355X | CU (2x throughput/CU) | 256 CUs (8 XCDs x 32) | 64 | TSMC N3P / N6 | HBM3e 288 GB | ~8 TB/s | 256 MB Infinity Cache (2 IODs) | XGMI, 8-GPU mesh | 20 PFLOPS FP4 |
| **CDNA 5** | **MI455X** | **WGP (4 dual-issue Wave32 SIMD32 + 4 matrix units)** | **256 WGPs (8 XCDs x 32 active; 34 physical/XCD)** | **32** | **TSMC N2 (XCD) / N3 (IOD + FCD)** | **HBM4 432 GB** | **23.3 TB/s** | **192 MB global L2 (2 x 96 MB on Fabric-and-Cache Dies), 54 TB/s** | **UALoE, 36 x 400 Gb/s, 72-GPU domain** | **40.26 PFLOPS MXFP4** |

*CDNA 5 row added 2026-08-08. 256 WGPs corresponds to 512 CUs in RDNA-style nomenclature — it is **not** directly comparable to MI350X's 256 CUs. MI455X transistor count and TBP are **not disclosed**; the N2/N3 split is corroborated only at keynote-slide level.*

---

## 1. Compute Engine

### Compute Unit (CU)

The CU is the fundamental compute building block. Each CU contains:

| Component | Per CU |
|-----------|--------|
| SIMD Units | 4x SIMD16 (64-wide total) |
| Matrix Core (MFMA) | 1 |
| Wavefront Scheduler | 1 (round-robin across SIMDs) |
| Max Resident Wavefronts | 40 (10 slots x 4 pools) |
| Vector Registers (VGPR) | 128 KB (512 x 32-bit x 64 lanes) |
| Scalar Registers (SGPR) | 12.5 KB |
| Accumulator Registers (AGPR) | Up to 256 KB (CDNA2+) |
| LDS (Local Data Share) | 64 KB |
| L1 Vector Cache | 32 KB |
| L1 Scalar Cache | 16 KB |
| Special Function Units (SFU) | Yes |
| Branch Unit | 1 |

**GPU-level CU counts:**

| SKU | CUs | XCDs | CUs/XCD (enabled) | Process |
|-----|-----|------|-------------------|---------|
| MI300X | 304 | 8 | 38 (of 40 physical) | TSMC N5 |
| MI300A (APU) | 228 | 6 | 38 | TSMC N5 |
| MI325X | 304 | 8 | 38 | TSMC N5 |
| MI350X | 256 | 8 | 32 | TSMC N3P |
| **MI455X (CDNA 5)** | **512 CU-equivalents = 256 WGPs** | **8** | **32 active WGPs (of 34 physical)** | **TSMC N2** |

**CDNA 5 shader organization (MI455X).** CDNA 5 is described in WGP terms rather than CU terms:

| Attribute | MI455X |
|---|---|
| XCDs | 8 |
| Shader engines per XCD | 2 |
| WGPs per shader engine | 16 |
| WGPs per XCD | 32 active (34 physical, 2 fused off) |
| WGPs per GPU | 256 |
| SIMD units per WGP | 4 dual-issue SIMD32 (Wave32) |
| Matrix units per WGP | 4 |
| Max engine clock | 2.4 GHz |
| Transistors / TBP | not disclosed |

Common error to avoid: 256 WGPs is **512 CUs** in RDNA-style nomenclature. Reports of "8 shader engines per XCD with 16 WGPs each" are arithmetically impossible (that would give 1,024 WGPs) and are not correct.

### Matrix Core (MFMA)

| Gen | Architecture | Supported Formats | Peak Throughput (per GPU) |
|-----|-------------|-------------------|--------------------------|
| CDNA3 | MI300X | FP64, FP32, FP16, BF16, FP8 (E4M3/E5M2), INT8 | ~2.6 PFLOPS FP8, ~1.3 PFLOPS BF16, ~163 TFLOPS FP64 |
| CDNA4 | MI350X | FP64, FP32, FP16, BF16, FP8, FP6, FP4 (OCP MX), INT8 | ~20 PFLOPS FP4 (4x gen-on-gen) |
| **CDNA 5** | **MI455X** | **FP32, FP16/BF16, FP8, MXFP6, MXFP4; new E5M3 scale format; 4-bit Tensor LUT instruction** | **40.26 PFLOPS MXFP4 and 315 TFLOPS FP32 (confirmed); 20.13 PFLOPS MXFP6/FP8 and 5.03 PFLOPS FP16/BF16 (arithmetically consistent, not independently confirmed)** |
| **CDNA 5 (HPC SKU)** | **MI430X** | adds hardware FP64 emphasis | **up to 288 TFLOPS FP64** (AMD press release) |

**CDNA 5 numeric / ISA additions** (AMD CDNA 5 blog, 2026-08-04) with no CDNA3/CDNA4 counterpart:

- **E5M3 scale format** — an additional exponent bit relative to the usual E8M0 micro-scaling exponent, giving a wider scale range for MX-style block formats.
- **4-bit Tensor Lookup Table (LUT) instruction** — a table-driven path for low-precision compute.
- **Tensor Data Movers** — dedicated data-movement hardware. AMD publishes no microarchitectural detail; describing them as an analogue of NVIDIA's TMA is interpretation, not vendor language.

AMD's own performance framing is "up to 4x greater AI compute throughput for key low-precision formats" — a **vendor marketing claim**, not an independently measured MI355X comparison.

MFMA compute rates per CU per clock cycle:

| Operation | FLOPS/clock/CU (CDNA3) |
|-----------|------------------------|
| Matrix FP64 | 256 |
| Matrix FP32 | 256 |
| Matrix FP16/BF16 | 2,048 |
| Matrix FP8/INT8 | 4,096 |

CDNA4 doubles throughput per CU for FP16/BF16/FP8/INT8 and adds native FP4/FP6 support (OCP MX micro-scale formats) with hardware support for independent A/B matrix type selection.

---

## 2. Data Path

### Wavefront Execution

- **Wavefront size**: 64 threads (CDNA1-CDNA4); 32 threads (RDNA consumer GPUs); **32 threads (CDNA 5 — native Wave32)**
- **SIMD execution**: One 64-thread wavefront executes across 4 SIMD16 pipelines over **4 clock cycles** (16 threads/cycle/SIMD)
- **Scheduler**: Round-robin across the 4 SIMDs; issues up to one instruction per wavefront per selected SIMD per cycle
- **Occupancy**: Up to 40 resident wavefronts per CU; actual occupancy limited by VGPR/LDS/SGPR pressure
- **Latency hiding**: The 4-cycle SIMD execution structure allows the scheduler to issue to other wavefronts/SIMDs in intervening cycles

### Dual-Issue and Pipeline Structure

- **CDNA3 dual-issue**: Supports double-issue FP32 -- a vector instruction and a scalar/branch instruction can issue simultaneously
- **MFMA independence**: Matrix Core instructions run on a dedicated unit, overlapping with VALU/VMEM/LDS operations in other wavefronts
- **Pipeline units**: VALU (vector ALU), SALU (scalar ALU), VMEM (vector memory), LDS, Branch, MFMA -- up to 5 instructions/cycle dispatched by the issue arbiter

### CDNA 5 Wave32 Execution (MI455X)

CDNA 5 changes the fundamental SIMD execution width:

- **Wave width**: native **Wave32**. AMD's CDNA 5 blog lists "Wave32 compute execution"; Chips and Cheese's MI455X deep-dive describes **4 dual-issue Wave32 SIMD32 units per WGP**, replacing CDNA3/CDNA4's Wave64-over-4x-SIMD16 arrangement.
- **Wave64 status**: **not disclosed.** No source states that Wave64 is removed from the ISA — only that CDNA 5 executes Wave32 natively. Do not assert removal.
- **Software consequence**: occupancy, register-pressure and LDS-bank-conflict reasoning derived for 64-lane waves does not transfer. AITER, CK-Tile and Triton-ROCm tuning tables for CDNA 5 must be re-derived rather than scaled from gfx942/gfx950 configurations. The CDNA 5 gfx target itself is not yet public (see Layer 5 notes and the CDNA 5 section).
- **Registers**: 128 KB vector registers per SIMD (reported; not independently confirmed).

### XCD (Accelerator Complex Die)

Each XCD is a self-contained GPU chiplet:
- CDNA3: 40 physical CUs (38 enabled), 4 MB L2, TSMC 5nm
- CDNA4: 32 enabled CUs, L2 cache, TSMC N3P (3nm)
- **CDNA 5: 34 physical WGPs (32 enabled) in 2 shader engines, TSMC N2; L2 moves off the XCD entirely onto two Fabric-and-Cache Dies**
- 8 XCDs per MI300X/MI350X/MI455X package
- XCDs are 3D-stacked on IODs; intra-XCD communication uses local interconnect, cross-XCD uses on-package Infinity Fabric

---

## 3. On-chip Memory

| Structure | Size | Scope | Notes |
|-----------|------|-------|-------|
| LDS (Local Data Share) | 64 KB | Per CU | Programmer-managed scratchpad; 32 banks, 4 bytes/bank; ~100x HBM bandwidth |
| L1 Vector (Data) Cache | 32 KB | Per CU | Caches VGPR-sourced loads; write-through |
| L1 Scalar (Constant) Cache | 16 KB | Per CU | Read-only; shared by scalar units |
| L1 Instruction Cache | 64 KB | Per XCD | Shared among CUs in a shader array |
| L2 Cache | 4 MB | Per XCD | Coherence point for atomics and cross-CU accesses; 32 MB total across 8 XCDs |
| Infinity Cache | 256 MB | Package-wide | SRAM on IODs (4 x 64 MB); last-level on-package cache before HBM |

**Key design points:**
- GPU L1 caches are write-through; coherence between CUs requires explicit `__threadfence` / `s_barrier` / `s_mem_realtime` instructions
- The L2 cache is the coherence point for all atomics and cross-CU accesses
- The 256 MB Infinity Cache provides a bandwidth multiplier by absorbing repeated accesses before they reach HBM

### CDNA 5 On-chip Memory (MI455X) — structural change

| Structure | Size | Scope | Notes |
|-----------|------|-------|-------|
| LDS | **320 KB** | Per WGP | Chips and Cheese MI455X deep-dive; the largest single change for CK-Tile / AITER tile sizing |
| L1 | **128 KB** | Per WGP | Chips and Cheese. A "64 KB per WGP" figure appears in some coverage and is **not confirmed** |
| Vector registers | 128 KB | Per SIMD | reported; not independently confirmed |
| **Global L2** | **192 MB** | Package-wide | **2 x 96 MB across two Fabric-and-Cache Dies (FCDs)**; **27 TB/s per FCD = 54 TB/s aggregate** |
| Infinity Cache | — | — | **No separate Infinity Cache tier.** CDNA3's "4 MB L2 per XCD + 256 MB Infinity Cache on IODs" is replaced by one global L2 on dedicated cache dies |

AMD's own CDNA 5 blog says only "larger L2 caches"; the 192 MB / 2x96 MB / 54 TB/s figures come from the Chips and Cheese deep-dive, i.e. third-party analysis rather than an AMD spec sheet. The architectural point for the software stack: the last-level on-package cache is no longer a separate Infinity Cache tier behind per-XCD L2s, so the CDNA3-era NUMA reasoning ("cross-XCD accesses fall through 4 MB L2 into 256 MB Infinity Cache") no longer describes this generation.

---

## 4. Off-chip Memory

### MI300X (CDNA3)

- **Capacity**: 192 GB HBM3 (8 stacks x 24 GB)
- **Peak bandwidth**: 5.3 TB/s
- **Bus width**: 1024-bit per stack x 8 stacks = 8192-bit total
- **Memory clock**: ~2.4 Gbps per pin (HBM3 specification)
- **HBM stacks**: Co-packaged on the CoWoS substrate alongside XCDs and IODs

### MI325X (CDNA3 refresh)

- **Capacity**: 288 GB HBM3e (8 stacks x 36 GB)
- **Peak bandwidth**: ~6.0 TB/s
- Released Q4 2024 as a memory-capacity upgrade to MI300X

### MI350X / MI355X (CDNA4)

- **Capacity**: 288 GB HBM3e (8 stacks x 36 GB; 12-high stacks)
- **Peak bandwidth**: ~8 TB/s (~1.3x better bandwidth per watt vs MI300X)
- **Pin speed**: 8 Gbps per pin (vs 2.4 Gbps on MI300X)

### MI455X / MI430X (CDNA 5)

- **Capacity**: **432 GB HBM4** (12 stacks x 36 GB)
- **Bus**: **2,048-bit per stack**, 192 channels
- **Peak bandwidth**: **23.3 TB/s** — AMD frames this as **2.91x** the prior generation (23.3 / 8.0 ≈ 2.91, internally consistent)
- **Pin speed**: not disclosed
- MI430X carries the same 432 GB HBM4 at the same bandwidth
- *Repo correction: the 19.6 TB/s figure previously carried for "MI400" was a pre-launch estimate and is superseded.*

---

## 5. Host Interface / Package

### MI300X Package

- **Die composition**: 8 XCDs (TSMC 5nm) + 4 IODs (TSMC 6nm) + 8 HBM3 stacks
- **3D integration**: XCDs 3D-stacked on IODs using TSMC SoIC hybrid bonding; entire assembly on CoWoS silicon interposer
- **IOD functions**: Each IOD contains 64 MB Infinity Cache slice, PCIe/Infinity Fabric controllers, and HBM memory controllers
- **Host connectivity**: PCIe Gen 5 x16 (128 GB/s bidirectional)
- **Total transistors**: ~146 billion
- **TBP**: 750 W

### MI300A APU

- **Die composition**: 3 CCDs (Zen 4, 8 cores each = 24 CPU cores) + 6 XCDs (228 GPU CUs) + 4 IODs
- **Unified memory**: CPU and GPU share 128 GB HBM3 pool; eliminates PCIe copy between CPU and GPU memory
- **Process**: TSMC 5nm (CCD/XCD) + 6nm (IOD)

### MI350X Package

- **Die composition**: 8 XCDs (TSMC N3P) + 2 IODs (TSMC N6) -- reduced from 4 IODs for power efficiency
- **Packaging**: CoWoS + SoIC 3D stacking with hybrid bonding
- **Total transistors**: 185 billion
- **TBP**: 1400 W (MI355X configuration)

### MI455X Package (CDNA 5)

- **Die composition**: **8 XCDs + 2 I/O Dies + 2 Fabric-and-Cache Dies (FCDs)** + 12 HBM4 stacks. The FCD is a new die type: it carries the 96 MB L2 slice and fabric that previously lived as per-XCD L2 plus IOD Infinity Cache.
- **Process**: **TSMC N2 for the compute chiplets, TSMC N3 for the I/O and Fabric-and-Cache dies.** Corroborated at keynote-slide level ("2nm compute chiplets, 3nm elsewhere"), not from a die-level teardown.
- **Packaging**: **TSMC CoWoS-L** (confirmed by Chips and Cheese). Claims that it is the "largest chip yet built on CoWoS-L" are a marketing superlative and are not verified.
- **Host connectivity**: **256 GB/s bidirectional over a dedicated 16-lane Infinity Fabric link to the EPYC "Venice" CPU** — not PCIe. This is the first Instinct part whose primary host attach is Infinity Fabric rather than PCIe.
- **Total transistors**: **not disclosed** (a 320B figure circulates in aggregator coverage and is unconfirmed).
- **TBP**: **not disclosed.**
- **gfx ISA target**: **not disclosed** — the public ROCm 7.14.0 compatibility matrix stops at gfx950 for Instinct parts.

---

## 6. Scale-up Interconnect

### Infinity Fabric / XGMI

- **Technology**: AMD 4th-generation Infinity Fabric
- **Links per MI300X GPU**: 7 XGMI links
- **Per-link specification**: 16 lanes x 32 Gbps/lane = 512 Gbps = 64 GB/s unidirectional; 128 GB/s bidirectional
- **Practical bandwidth**: ~48 GB/s per link unidirectional (~75% efficiency after CRC overhead)
- **8-GPU aggregate**: Each GPU connects to all 7 peers; 896 GB/s aggregate per GPU

### Topology

| Configuration | GPUs | Links per GPU | Notes |
|---------------|------|---------------|-------|
| Single link | 2 | 1 | Minimal 2-GPU config |
| Dual link | 4 | 2 | Quad-link ring or partial mesh |
| Full mesh | 8 | 7 | Standard OAM baseboard (MI300X) |

- The 8-GPU OAM baseboard uses a **full mesh** topology: every GPU has a direct XGMI link to every other GPU
- Within the MI300X package, the 8 XCDs and 4 IODs communicate via on-package Infinity Fabric (direct die-to-die, not XGMI)
- No central switch ASIC (unlike NVIDIA's NVSwitch); all-to-all connectivity is achieved by the mesh topology itself

### Comparison to NVLink

XGMI's full-mesh topology differs structurally from NVIDIA's NVLink + NVSwitch:
- NVSwitch provides non-blocking all-to-all via a central crossbar ASIC with hardware reduction (SHARP)
- XGMI requires the mesh links themselves to carry all traffic; no in-network reduction hardware
- This shapes RCCL's algorithm selection: ring algorithms on XGMI mesh vs NVLS on NVSwitch

### CDNA 5 Scale-up: UALink-over-Ethernet (MI455X)

CDNA 5 abandons the 8-GPU XGMI mesh for a switched fabric:

| Attribute | MI355X (CDNA4) | **MI455X (CDNA 5)** |
|---|---|---|
| Scale-up technology | XGMI / Infinity Fabric mesh | **UALink-over-Ethernet (UALoE)** |
| Per-GPU interfaces | 7 XGMI links | **36 x 400 Gb/s UALoE interfaces** |
| Per-GPU scale-up BW | 896 GB/s aggregate | **1.8 TB/s each way = 3.6 TB/s bidirectional** |
| Scale-up domain | 8 GPUs (OAM baseboard) | **72 GPUs (single Helios rack domain)** |
| Rack scale-up BW | — | **260 TB/s** |
| Topology | direct full mesh, no switch | **switched** (AMD + Broadcom hardware) |
| In-network reduction | none | **not disclosed** |

Notes and evidence limits:
- The "36 x 400 Gb/s" figure is from the Chips and Cheese deep-dive; a "72 lanes" phrasing appears in some coverage and is **not confirmed**.
- The AMD–Broadcom partnership for Helios switching is corroborated (ServeTheHome's Helios deep-dive is explicitly "AMD + Broadcom hardware combined"), but the specific claim of **12x Broadcom Tomahawk 6 ASICs in a 12-plane topology** is **not confirmed**.
- The 9x growth of the scale-up domain (8 -> 72 GPUs) plus the move from mesh to switch is the single most consequential change for RCCL: the pre-computed 8-GPU ring orderings and `rcclRomeModel`-style topology templates do not describe this generation.

---

## 7. Scale-out Interconnect

### Network Stack

```
Application (PyTorch / JAX / vLLM)
    |
RCCL (collective communications)
    |
UCX (transport layer)
    |
InfiniBand (NDR/HDR) or RoCEv2 (400GbE via Pollara 400)
    |
Top-of-Rack Switch -> Spine -> Leaf (fat-tree or dragonfly)
```

### AMD Pensando NIC Portfolio

- **Pensando Pollara 400**: Up to 400 Gbps Ethernet; hardware-programmable P4 engine (3rd generation); first UEC (Ultra Ethernet Consortium)-ready AI NIC. Up to 25% RCCL performance improvement vs commodity NICs. Validated for 16-1024 node clusters; UEC-based Ethernet claimed to scale beyond 1 million GPUs.
- **Salina 400 DPU**: Offloads networking/storage functions; pairs with Pollara for full DPU functionality.
- **Pensando "Vulcano"** (CDNA 5 / Helios generation): the scale-out NIC used in Helios racks. **43 TB/s scale-out bandwidth per rack** is confirmed (AMD blog + ServeTheHome). A per-GPU figure of 2,400 Gb/s is arithmetically consistent with that rack total (72 x 2,400 Gb/s ≈ 43.2 TB/s) but is not independently stated. Claims of "3x UALink128 ports per GPU" and "12x Pensando Salina 400G DPUs, one per compute tray" are **not confirmed**; only the general use of Pensando NICs/DPUs and the "Vulcano" NIC name are.

### GPU-Direct RDMA

- AMD kernel driver exposes PeerDirect RDMA interfaces
- NICs can directly DMA to/from GPU HBM without CPU involvement
- HMM (Heterogeneous Memory Management) support for unified address space
- Enabled via `MPICH_GPU_SUPPORT_ENABLED=1` or equivalent for GPU-aware MPI

### Cluster Networking

- InfiniBand remains dominant for highest-performance clusters
- RoCE/UEC gaining traction for hyperscale deployments
- RCCL uses XGMI for intra-node (fast path) and IB/RoCE/TCP for inter-node
- Rail-optimized tree topology in RCCL routes through NIC-local GPUs first

---

## Next-Generation: MI400 and UDNA Architecture

*Added: 2026-04-05. **Partially superseded 2026-08-08** — the MI400-series shipped as **CDNA 5**, not "CDNA Next" and not UDNA. AMD's press release says only that "Next-generation AMD Instinct MI500 Series GPUs are coming in 2027"; keynote coverage attaches **CDNA 6** to MI500. This section is kept for the UDNA strategy background; shipping MI455X specifications are in the [CDNA 5 section](#cdna-5--instinct-mi455x--helios-2026).*

### UDNA — Unified Architecture Strategy

UDNA (Unified DNA) is AMD's announced convergence of the RDNA (gaming) and CDNA (datacenter) GPU lineages into a single unified microarchitecture. Announced at IFA 2024, UDNA replaces both the RDNA and CDNA naming after their current generations (RDNA 4 / CDNA 4 are the last before the merge).

**Motivation:** Post-2019 split created two diverging ISAs (`gfxRDNA*` and `gfxCDNA*`). Developers targeting AMD hardware had to choose between architectures; porting between them required full recompilation, and optimization tuning (MFMA assembly, CK kernels) could not be shared. UDNA collapses this back to a single target — analogous to returning to the unified GCN era — while retaining the modern compute capabilities AMD built in CDNA.

**ALU design:** UDNA uses an ALU structure reminiscent of the original GCN architecture (common ancestor of RDNA and CDNA), modernized with:
- Dedicated matrix/tensor cores (MFMA-equivalent) in **all** UDNA GPU products, including gaming SKUs
- Forward and backward ISA compatibility guaranteed across at minimum UDNA 6 and UDNA 7 generations
- Unified wavefront model (exact width TBD; likely converged between RDNA's 32-wide and CDNA's 64-wide)

### MI400-Series Specifications (as shipped)

*Table corrected 2026-08-08 against the launched part.*

| Attribute | MI300X (CDNA3) | MI350X (CDNA4) | **MI455X (CDNA 5, 2026)** |
|-----------|---------------|----------------|--------------------------|
| Peak FP4 | N/A | 20 PFLOPS | **40.26 PFLOPS MXFP4** |
| Peak FP8 | ~2.6 PFLOPS | ~10 PFLOPS | **20.13 PFLOPS MXFP6/FP8** (not independently confirmed) |
| Peak FP16/BF16 | ~1.3 PFLOPS | — | **5.03 PFLOPS** (not independently confirmed) |
| Peak FP32 | — | — | **315 TFLOPS** |
| Memory | 192 GB HBM3 | 288 GB HBM3e | **432 GB HBM4** |
| Memory BW | 5.3 TB/s | ~8 TB/s | **23.3 TB/s** (was listed as 19.6 TB/s pre-launch) |
| Scale-up BW | 896 GB/s (7 XGMI links) | 8-GPU XGMI mesh | **3.6 TB/s bidirectional UALoE; 72-GPU domain** |
| Scale-out BW | — | — | **43 TB/s per 72-GPU rack** (the pre-launch "300 GB/s/GPU" figure is superseded) |
| Process | TSMC 5nm (XCD) | TSMC N3P (XCD) | **TSMC N2 (XCD) + N3 (IOD/FCD)** |
| TBP | 750 W | 1400 W | **not disclosed** |

**HBM4 is a significant jump**: 432 GB capacity and 23.3 TB/s bandwidth, which AMD frames as 2.91x the prior generation — critical for large-model LLM inference where memory bandwidth is the primary bottleneck.

### Helios Rack-Scale Platform

AMD's "Helios" system architecture pairs the MI400 series with the full AMD datacenter stack:

```
Helios Rack (2026, in production; shipping from Q3 2026)
  ├── 72x MI455X (Instinct, CDNA 5)  — 40.26 PFLOPS MXFP4 each, 432 GB HBM4 each
  ├── 18x EPYC "Venice" CPUs         — 4 GPUs per compute node
  ├── UALoE scale-up fabric          — 72-GPU domain, 260 TB/s rack scale-up
  └── Pensando "Vulcano" AI NICs     — 43 TB/s rack scale-out

Rack totals: 2.9 EFLOPS peak MXFP4 | 31 TB HBM4 | up to 36 TB DDR5 | ~1.7 PB/s aggregate HBM BW
```

*The earlier "3 AI exaflops per rack" target resolved to **2.9 EFLOPS MXFP4** at launch. Aggregate HBM bandwidth of ~1.7 PB/s is the arithmetically consistent figure (72 x 23.3 TB/s ≈ 1.68 PB/s); a circulating 1.4 PB/s figure is inconsistent and discarded. **Rack power, rack format and bus-bar voltage are not disclosed** — the "225–245 kW", "Open Rack Wide 1.2 m x 1.3 m / 44 OU" and "50 V liquid-cooled bus bar" figures are not confirmed against a primary AMD source; AMD's press-release footnote references a "100 kW power envelope per rack" only as comparison methodology.*

Deployment commitments as of 2026-08-08 (corporate/roadmap announcements, not deployed capacity):
- **Anthropic**: to deploy up to **2 GW** of AMD Instinct MI455X GPUs
- **OpenAI**: 6 GW agreement dating to October 2025; Helios expected online **beginning Q4 2026**
- **Meta**: "has begun testing and validating workloads on AMD Helios racks" — no capacity figure. *(Repo correction: the previously recorded "AMD + Meta: 6 GW" was wrong; the 6 GW figure belongs to the OpenAI agreement.)*
- **Microsoft**: named among Helios adopters, no capacity figure

### MI500 Outlook (2027)

- Architecture: **CDNA 6** (per keynote coverage; AMD's press release says only "Next-generation AMD Instinct MI500 Series GPUs are coming in 2027")
- Process, memory, performance: **not disclosed**
- *The previously recorded "TSMC 2nm / HBM4E / ~1,000x vs MI300X" line has been removed: none of it had a primary source. AMD's on-stage line was a "2000x performance improvement in just 4 years" marketing claim, not a spec.*
- MI600 series is described as in development for 2028; specifications not disclosed

---

## CDNA 5 / Instinct MI455X / Helios (2026)

*Added: 2026-08-08.*

### Launch and status

AMD launched the MI400-series flagship at **Advancing AI 2026 on 2026-07-23**; AMD's ROCm blog "Introducing AMD CDNA™ 5 and the AMD Helios™ Rackscale Solution" followed on **2026-08-04**. The hardware itself was not brand-new: MI450/Helios were public from October 2025 and MI455X + Venice + Helios hardware was physically displayed at CES 2026. What is new after this repo's 2026-04-05 baseline is the **CDNA 5 architecture disclosure, the production declaration, and the detailed specifications**.

Status language, precisely: Helios is "now in production to be deployed by leading AI companies at gigawatt scale" (AMD press release); on stage AMD said shipments start in Q3 2026; OpenAI expects Helios online from Q4 2026. The correct characterization is **in production / shipping from Q3 2026, ramping into Q4 2026 — not deployed at scale.**

### Layer-by-layer delta vs CDNA4

| Layer | CDNA4 (MI355X) | CDNA 5 (MI455X) | Evidence strength |
|---|---|---|---|
| 1 Compute Engine | 256 CUs, 1 Matrix Core/CU | 256 WGPs (8 XCDs x 32 of 34), 4 matrix units/WGP, 2.4 GHz | third-party deep-dive |
| 2 Data Path | Wave64 over 4x SIMD16 | **Wave32**, 4 dual-issue SIMD32/WGP | AMD blog ("Wave32 compute execution") + deep-dive |
| 3 On-chip Memory | 64 KB LDS/CU; 4 MB L2/XCD + 256 MB Infinity Cache | **320 KB LDS/WGP; 128 KB L1/WGP; 192 MB global L2 (2x96 MB on FCDs), 54 TB/s; no separate Infinity Cache tier** | third-party deep-dive; AMD says only "larger L2 caches" |
| 4 Off-chip Memory | 288 GB HBM3e, ~8 TB/s | **432 GB HBM4, 23.3 TB/s**, 12 stacks x 2,048-bit, 192 channels | AMD ("2.91x") |
| 5 Host / Package | 8 XCD + 2 IOD, PCIe Gen5 host | **8 XCD (N2) + 2 IOD + 2 FCD (N3), CoWoS-L; host = 256 GB/s bidirectional 16-lane Infinity Fabric to EPYC** | keynote-slide + deep-dive |
| 6 Scale-up | XGMI mesh, 8 GPUs | **UALoE, 36 x 400 Gb/s, 3.6 TB/s bidir/GPU, 72-GPU domain, 260 TB/s/rack** | AMD blog + ServeTheHome + deep-dive |
| 7 Scale-out | Pollara 400 | **Pensando "Vulcano"; 43 TB/s per rack** | AMD blog + ServeTheHome |

### SKU segmentation

| SKU | Architecture | Position | Status |
|---|---|---|---|
| **MI455X** | CDNA 5 | Flagship rack-scale training/inference | In production; shipping from Q3 2026 |
| **MI430X** | CDNA 5 | HPC / sovereign; **up to 288 TFLOPS hardware FP64**; 432 GB HBM4 | Announced; **H1 2027** availability |
| MI350P | **CDNA 4 (gfx950)** | Dual-slot PCIe enterprise-inference card, announced 2026-05-07, re-promoted at Advancing AI 2026 | Shipping; in the ROCm 7.14.0 compatibility matrix. **Not** a CDNA 5 part; capacity not confirmed |

"MI450 / MI450-series" naming coexists with MI455X in AMD's own materials. **MI440X** appears in some aggregator coverage, is absent from AMD's press release, and is deliberately not entered here.

### Software-visible consequences

1. **No public gfx target.** The ROCm 7.14.0 compatibility matrix's newest Instinct entries are gfx950 / gfx942 / gfx90a / gfx908. Neither the CDNA 5 blog nor the ROCm 7.14 blog names a gfx target for CDNA 5. **The CDNA 5 gfx ISA identifier is not disclosed** — production hardware is ahead of public ISA-target disclosure, so there is no published `--offload-arch` value, no CDNA 5 `hsa/gfx*` AITER assembly directory, and no CDNA 5 CK-Tile tuning CSVs in the open tree.
2. **Wave32 invalidates carried-over tuning.** Occupancy/register/LDS-bank reasoning derived on 64-lane waves does not transfer.
3. **320 KB LDS per WGP** changes the tile-size search space for CK-Tile and AITER far more than the L1 change does.
4. **72-GPU scale-up domain** replaces the 8-GPU mesh assumptions baked into RCCL's pre-computed topology templates. ROCm 7.14.0's hierarchical AllGather (inter-node vs intra-node phase separation) and direct reduce-scatter are the visible software-side response.
5. **E5M3 scale format, 4-bit Tensor LUT and Tensor Data Movers** have no counterpart in the CDNA3/CDNA4 MFMA description and no public ISA documentation yet.

---

## Summary: Key Architectural Abstractions

| Abstraction | Definition | CDNA3 Value | CDNA4 Change | CDNA 5 Change |
|-------------|-----------|-------------|--------------|---------------|
| **CU (Compute Unit)** | Basic programmable compute block: 4x SIMD16, Matrix Core, LDS, caches, scheduler | 304 CUs (MI300X) | 256 CUs, 2x throughput/CU | Superseded by the **WGP** as the described block |
| **WGP (Workgroup Processor)** | CDNA 5 shader block: 4 dual-issue Wave32 SIMD32 units + 4 matrix units + 320 KB LDS + 128 KB L1 | — | — | **256 WGPs (8 XCDs x 32 of 34 physical); 2 shader engines/XCD; 2.4 GHz** |
| **Matrix Core (MFMA)** | Dedicated tensor accelerator; executes v_mfma_* instructions using VGPR/AGPR tiles | FP64-FP8 support | Adds FP6, FP4 (OCP MX) | 4 matrix units/WGP; **E5M3 scale format, 4-bit Tensor LUT, Tensor Data Movers**; 40.26 PFLOPS MXFP4 |
| **Wavefront** | SIMD execution unit scheduled by the shader | 64 threads (fixed) | 64 threads (unchanged) | **32 threads (native Wave32)**; Wave64 availability not disclosed |
| **XCD** | GPU compute chiplet: shader blocks + local caches + schedulers; stackable unit in MCM packages | 8 XCDs, TSMC 5nm, 38 CUs each | 8 XCDs, TSMC N3P, 32 CUs each | 8 XCDs, **TSMC N2**, 32 active WGPs each; **L2 moved off-XCD** |
| **FCD (Fabric-and-Cache Die)** | New CDNA 5 die type carrying the global L2 slice and fabric | — | — | **2 FCDs (TSMC N3), 96 MB L2 each, 27 TB/s each** |
| **Infinity Fabric** | On-chip and inter-chip coherent interconnect (XGMI for GPU-GPU) | 7 links x 128 GB/s bidir | Carried forward; 2-IOD design | On-package + **256 GB/s host link**; GPU-GPU replaced by **UALoE** |
| **Infinity Cache** | Package-wide last-level on-package SRAM cache on IODs | 256 MB (4 x 64 MB) | Carried forward | **Replaced by 192 MB global L2 on FCDs (54 TB/s)** |
| **UALoE** | UALink-over-Ethernet scale-up fabric | — | — | **36 x 400 Gb/s per GPU, 3.6 TB/s bidir, 72-GPU domain, 260 TB/s/rack** |

---

## Sources

- [AMD CDNA 3 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Instinct MI300 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300 Series Microarchitecture -- ROCm Docs](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [AMD CDNA 3 Compute Architecture -- Chips and Cheese](https://chipsandcheese.com/p/amds-cdna-3-compute-architecture)
- [AMD CDNA 4 Architecture Announcement -- Chips and Cheese](https://chipsandcheese.com/p/amds-cdna-4-architecture-announcement)
- [AMD MI300X Hot Chips 2024 Presentation](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf)
- [AMD Dives Deep on CDNA 4 at Hot Chips 2025 -- ServeTheHome](https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/)
- [Understanding RCCL Bandwidth and xGMI on MI300X -- AMD ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)
- [AMD Pensando Pollara 400 AI NIC](https://www.amd.com/en/products/network-interface-cards/pensando.html)
- [MI300X Platform Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4 -- AMD ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [AMD Instinct MI400 — 40 PFLOPS, 432 GB HBM4 — WccfTech](https://wccftech.com/amd-instinct-mi400-accelerator-doubles-compute-40-pflops-432-gb-hbm4-memory-2026-launch/)
- [AMD Launches MI350, Confirms MI400 in 2026 — VideoCardz](https://videocardz.com/newz/amd-launches-instinct-mi350-series-confirms-mi400-in-2026-with-432gb-hbm4-memory)
- [AMD Announces UDNA — Brings RDNA and CDNA Together — Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)
- [AMD RDNA and CDNA Merge into UDNA — PC Gamer](https://www.pcgamer.com/hardware/graphics-cards/from-the-developers-standpoint-they-love-this-strategyamds-plan-to-merge-its-rdna-and-cdna-gpu-architectures-to-a-unified-system-called-udna/)
- [AMD 2026-2027 AI Roadmap: MI400 and MI500 — WccfTech](https://wccftech.com/amd-to-battle-nvidia-ai-dominance-instinct-mi400-accelerators-2026-mi500-2027/)
- [AMD Instinct MI350 Series and Beyond — AMD Official Blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html)

### CDNA 5 / MI455X / Helios (added 2026-08-08)

- [Introducing AMD CDNA 5 and the AMD Helios Rackscale Solution — AMD ROCm Blogs (2026-08-04)](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)
- [AAI 2026: AMD Delivers Full-Stack Compute for the Agentic AI Era — AMD Investor Relations press release (2026-07-23)](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era)
- [AMD's Instinct MI455X: Aiming for the Top — Chips and Cheese (2026-07-23)](https://chipsandcheese.com/p/amds-instinct-mi455x-aiming-for-the)
- [AMD Helios Architecture Deep Dive: AMD + Broadcom Hardware Combined — ServeTheHome](https://www.servethehome.com/amd-helios-architecture-deep-dive-amd-broadcom-hardware-combined/)
- [AMD Advancing AI 2026 Keynote Live Coverage — ServeTheHome](https://www.servethehome.com/amd-advancing-ai-2026-keynote-live-coverage/)
- [AMD EPYC Venice, Instinct MI455X and Helios Hardware on Display at CES 2026 — ServeTheHome (2026-01-07)](https://www.servethehome.com/amds-epyc-venice-instinct-mi455x-helios-hardware-on-display-for-first-time-at-ces-2026/)
- [ROCm Compatibility Matrix (gfx targets)](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Oracle Leads AI Innovation with AMD "Altair" MI450 GPUs and Helios Racks — The Next Platform (2025-10-14)](https://www.nextplatform.com/compute/2025/10/14/oracle-leads-ai-innovation-with-amd-altair-mi450-gpus-and-helios-racks/1632443)
- [AMD Instinct — Wikipedia (SKU/date cross-check)](https://en.wikipedia.org/wiki/AMD_Instinct)
