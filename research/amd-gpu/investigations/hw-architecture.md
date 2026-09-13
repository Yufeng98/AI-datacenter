# AMD GPU Hardware Architecture Investigation
## CDNA3 (MI300X) and CDNA4 (MI350/MI355X)

**Chip:** amd-gpu  
**Device Class:** GPU  
**Architectures Covered:** CDNA3 (MI300-series), CDNA4 (MI350-series)  
**Investigation Date:** 2026-04-05  

---

## Layer 1: Compute Engine

### CDNA3 — MI300X

Each **Compute Unit (CU)** in CDNA3 contains:
- **4 SIMD units**, each 16-wide (SIMD16), for 64-wide vector execution per CU
- **Wavefront size:** 64 threads (one wavefront spans all 4 SIMDs over 4 cycles at 16 wide each)
- **Matrix Cores:** One Matrix Core unit per CU; supports MFMA (Matrix Fused Multiply-Add) instructions
- **Scalar unit + SGPR:** Scalar general-purpose registers (12.5 KB/CU), handles control flow, address computation
- **Vector registers (VGPR):** 128 KB per CU; 512 × 32-bit × 64 lanes
- **Local Data Share (LDS):** 64 KB per CU (shared scratchpad for threads in a workgroup)
- **L1 Vector (Data) Cache:** 32 KB per CU (sometimes cited as 16 KB for separate L1 scalar + 16 KB vector in CDNA2; CDNA3 documents confirm 32 KB L1 data cache per CU)
- **L1 Scalar (Constant) Cache:** 16 KB per CU (read-only constant/instruction cache)

**Supported data types per CU:**
| Operation | FLOPS/clock/CU |
|-----------|----------------|
| Matrix FP64 | 256 |
| Matrix FP32 | 256 |
| Matrix FP16 | 2048 |
| Matrix BF16 | 2048 |
| Matrix FP8  | 4096 |
| Matrix INT8 | 4096 |

**MI300X aggregate compute:**
- 304 CUs total (8 XCDs × 38 enabled CUs/XCD; each XCD has 40 physical CUs)
- Peak FP8: ~2.6 PFLOPS; Peak BF16: ~1.3 PFLOPS; Peak FP64: ~163 TFLOPS

### CDNA4 — MI350 / MI355X

Key architectural improvements over CDNA3:
- **Process node:** TSMC N3P (3nm) for XCDs (vs. 5nm in MI300X); IOD remains on TSMC N6 (6nm)
- **CU count:** 256 CUs total (8 XCDs × 32 enabled CUs/XCD) — fewer but significantly denser CUs
- **2× throughput per CU** for FP16/BF16 and FP8/INT8 operations vs. CDNA3
- **New data types:** FP6 and FP4 added (OCP MX micro-scale formats); hardware support for independent A/B matrix type selection
- FP4/FP6 delivers same compute rate — MI350 achieves **20 PFLOPS FP4**, a **4× gen-on-gen** improvement
- **New vector ALU:** supports 2-bit operations; can accumulate BF16 results into FP32
- **Transistor count:** 185 billion (vs. ~146 billion for MI300X)
- **Package change:** 2 IODs (reduced from 4 IODs in MI300X) enabling wider, lower-clocked die-to-die connections for power efficiency

---

## Layer 2: Data Path

### Wavefront Scheduling

- The **scheduler** arbitrates and issues instructions for all active wavefronts on a CU
- On every clock cycle the scheduler selects wavefronts from one SIMD unit at a time in **round-robin** across the 4 SIMDs
- Issues **up to one instruction per wavefront** per selected SIMD per cycle
- Multiple wavefronts can be resident simultaneously on a CU (occupancy limited by VGPR/LDS usage)

### Instruction Pipeline

- Each SIMD is **SIMD16-wide**; executing a 64-thread wavefront takes **4 cycles** (16 threads × 4 cycles)
- This 4-cycle structure hides latency: the scheduler can issue instructions to other wavefronts on other SIMDs in the intervening cycles
- **Dual-issue capability:** CDNA3 supports double-issue FP32 execution — a vector instruction and a scalar/branch instruction can be issued simultaneously when the instruction stream allows it

### Matrix Core Pipeline

- MFMA instructions target the dedicated Matrix Core unit within each CU
- MFMA operates on accumulator register tiles (up to 32×32 for FP16, 16×16 for FP64)
- Latency is hidden by overlapping matrix computation with vector/memory operations in other wavefronts

---

## Layer 3: On-chip Memory

| Structure | Size | Scope | Notes |
|-----------|------|-------|-------|
| LDS (Local Data Share) | 64 KB | Per CU | Programmer-managed scratchpad; shared across threads in a workgroup |
| L1 Vector (Data) Cache | 32 KB | Per CU | Private to CU; caches VGPR-sourced loads |
| L1 Scalar (Constant) Cache | 16 KB | Per CU | Read-only; shared by scalar units |
| L1 Instruction Cache | 64 KB | Per XCD (shared among CUs in a shader array) | |
| L2 Cache | 4 MB | Per XCD | Shared across all CUs within one XCD; 8 XCDs = 32 MB total L2 |
| Infinity Cache | 256 MB | Package-wide | SRAM on IODs; acts as last-level on-package cache before HBM; shared across all 8 XCDs |

**Notes:**
- Some AMD docs cite 8 MB L2 per XCD for certain configurations; the MI300X confirmed figure is **4 MB L2 per XCD** (32 MB total across 8 XCDs)
- The 256 MB Infinity Cache resides on the 4 IODs and is accessed via the on-package Infinity Fabric

---

## Layer 4: Off-chip Memory

### MI300X (CDNA3)

- **Capacity:** 192 GB HBM3 (8 stacks × 24 GB per stack)
- **Peak bandwidth:** 5.3 TB/s (5,300 GB/s)
- **HBM stacks:** 8, co-packaged on the CoWoS substrate alongside the XCDs and IODs
- **Bus width:** 1024-bit per stack × 8 stacks = 8192-bit total memory bus
- **Memory clock:** ~2.4 Gbps per pin (HBM3 specification)

### MI325X (CDNA3 refresh)

- **Capacity:** 288 GB HBM3e (8 stacks × 36 GB)
- **Peak bandwidth:** ~6.0 TB/s
- Released Q4 2024 as a memory-capacity upgrade to MI300X

### MI350X / MI355X (CDNA4)

- **Capacity:** 288 GB HBM3e (8 stacks × 36 GB; 12-high 24 Gbit devices at full 8 Gbps/pin)
- **Bandwidth:** ~8 TB/s (approximately 1.3× better bandwidth per watt vs. MI300X)
- **Stack height:** 12-hi HBM3e vs. 8-hi HBM3 in MI300X

---

## Layer 5: Host Interface / Package

### MI300X Package

- **Die composition:**
  - 8 × XCD (Accelerator Complex Die) — TSMC 5nm, each with 40 CUs (38 enabled) + 4 MB L2
  - 4 × IOD (I/O Die) — TSMC 6nm, each with 64 MB Infinity Cache slice + PCIe, Infinity Fabric controllers, memory controllers
- **3D integration:** XCDs are 3D-stacked on top of IODs using TSMC SoIC (System on Integrated Chip) hybrid bonding; entire assembly sits on a CoWoS silicon interposer with HBM3 stacks
- **Host connectivity:** PCIe Gen 5 ×16 (128 GB/s host bandwidth)
- **Total transistors:** ~146 billion
- **TBP (Total Board Power):** 750 W

### MI300A APU

- **Die composition:** 3 × CCD (CPU Core Complex Die, Zen 4, 8 cores each = 24 CPU cores) + 6 × XCD (GPU, 38 CUs each = 228 CUs total) + 4 × IOD
- **Unified Memory:** CPU and GPU share the same 128 GB HBM3 pool; eliminates PCIe copy between CPU and GPU memory
- **Process:** Same TSMC 5nm (CCD/XCD) + 6nm (IOD)

### MI350 Package

- **Die composition:** 8 × XCD (TSMC N3P, 32 enabled CUs each) + 2 × IOD (TSMC N6, reduced from 4)
- **Packaging:** Same CoWoS + SoIC 3D stacking; hybrid bonding between XCDs and IODs
- **TBP:** 1400 W (MI355X configuration)

---

## Layer 6: Scale-up Interconnect

### Infinity Fabric / XGMI (Cross GPU Memory Interconnect)

- **Technology:** AMD 4th-generation Infinity Fabric
- **Links per MI300X GPU:** 7 XGMI links
- **Per-link specification:** 16 lanes × 32 Gbps/lane = 512 Gbps = **64 GB/s per link** (unidirectional); **128 GB/s bidirectional**
- **Practical bandwidth:** ~48 GB/s per link (unidirectional, accounting for CRC overhead ~75% efficiency)
- **8-GPU node aggregate:** Each GPU connects to all 7 peers; aggregate xGMI bandwidth = **896 GB/s** (7 links × 128 GB/s)

### Topology Tiers

| Configuration | GPUs | Links per GPU | Notes |
|---------------|------|---------------|-------|
| Single link   | 2    | 1 link each   | Minimal 2-GPU config |
| Dual link     | 4    | 2 links each  | Quad-link ring or partial mesh |
| Quad link     | 8    | 7 links each  | Full mesh (8-GPU OAM baseboard) |

- The standard AMD MI300X OAM (Open Accelerator Module) 8-GPU baseboard uses a **full mesh** topology where every GPU has a direct XGMI link to every other GPU
- Within the MI300X package, the 8 XCDs and 4 IODs communicate via on-package Infinity Fabric (no XGMI; direct die-to-die)

---

## Layer 7: Scale-out Interconnect

### Overview

For inter-node (cluster-level) GPU-to-GPU communication, AMD MI300X clusters rely on:

1. **RCCL (ROCm Collective Communication Library):** AMD's equivalent of NCCL; uses XGMI for intra-node communication and InfiniBand/RoCE/TCP for inter-node
2. **UCX (Unified Communication X):** Standard transport layer for InfiniBand and RoCE
3. **GPU-aware MPI:** AMD kernel driver exposes PeerDirect RDMA interfaces; NICs can directly DMA to/from GPU HBM without CPU involvement

### AMD Pensando DPU / NIC Portfolio

- **Pensando Pollara 400 AI NIC:** Up to 400 Gbps Ethernet; hardware-programmable P4 engine (3rd generation); first UEC (Ultra Ethernet Consortium)-ready AI NIC
- **Salina 400 DPU:** Offloads networking/storage functions; pairs with Pollara for full DPU functionality
- **Validated cluster designs:** 16–1024 nodes; up to 25% RCCL performance improvement vs. commodity NICs
- **Scale demonstrated:** UEC-based Ethernet claimed to scale beyond 1 million GPUs

### Cluster Network Stack

```
Application (PyTorch / JAX / TensorFlow)
    |
RCCL (collective communications)
    |
UCX (transport layer)
    |
InfiniBand (NDR/HDR) or RoCEv2 (400GbE via Pollara 400)
    |
Top-of-Rack Switch → Spine → Leaf (fat-tree or dragonfly)
```

- **GPU-aware MPI** is enabled via `MPICH_GPU_SUPPORT_ENABLED=1` or equivalent; ROCm exposes HMM (Heterogeneous Memory Management) for unified address space
- InfiniBand remains dominant for highest-performance clusters; RoCE/UEC gaining traction for hyperscale deployments

---

## Summary: Key Architectural Abstractions

| Abstraction | Definition | CDNA3 Value | CDNA4 Change |
|-------------|-----------|-------------|--------------|
| **CU (Compute Unit)** | Basic programmable compute block containing SIMDs, Matrix Core, LDS, caches, scheduler | 304 CUs (MI300X) | 256 CUs (MI350X), 2× throughput/CU |
| **Matrix Core** | Dedicated tensor/matrix accelerator within each CU; executes MFMA instructions | FP64–FP8 support | Adds FP6, FP4 (OCP MX) |
| **Wavefront** | Execution group of 64 threads; basic SIMD scheduling unit | 64 threads fixed | 64 threads (unchanged) |
| **Infinity Fabric** | AMD's on-chip and inter-chip coherent interconnect fabric (XGMI for GPU-GPU) | 7 links × 128 GB/s bi-directional | Carried forward; 2-IOD design lowers power |
| **XCD (Accelerator Complex Die)** | GPU chiplet containing CUs, L2 cache, local schedulers; stackable unit in MCM packages | 8 XCDs/pkg, TSMC 5nm | 8 XCDs/pkg, TSMC N3P |

---

## MI400 / UDNA Roadmap

*Added: 2026-04-05. **SUPERSEDED 2026-08-08** — the MI400-series shipped as **CDNA 5** with 23.3 TB/s HBM4 and a UALoE 72-GPU scale-up domain. Every figure in the roadmap table and the "MI400 confirmed specifications" list below is a pre-launch estimate; the "~1,000x vs MI300X" MI500 figure had no primary source, and the "AMD and Meta 6 GW" line is wrong (6 GW is the OpenAI agreement). This section is retained as the record of what was believed on 2026-04-05 — see the CDNA 5 investigation update at the end of this file for the verified figures.*

---

### What UDNA Is

**UDNA (Unified DNA)** is AMD's strategic architectural merger of its two previously separate GPU lineages:

- **RDNA** — consumer gaming graphics architecture (Radeon RX series)
- **CDNA** — datacenter/HPC compute architecture (Instinct series)

AMD announced UDNA at IFA 2024. The motivation is developer friction: since the 2019 GCN split into RDNA and CDNA, developers targeting AMD hardware faced two incompatible ISAs, separate toolchains, and no cross-architecture forward compatibility. NVIDIA's CUDA by contrast presents a single unified programming model across all product lines (gaming, workstation, datacenter). UDNA is AMD's direct response — one architecture, one ISA family, one software stack for gaming GPUs, datacenter accelerators, and console APUs (including the Sony PlayStation 6).

**Key technical changes UDNA introduces:**

| Feature | RDNA (gaming) | CDNA (datacenter) | UDNA (unified) |
|---------|--------------|-------------------|----------------|
| Matrix/Tensor Cores | No (shader-based emulation) | Yes (MFMA) | Yes — matrix cores in all products |
| Wavefront width | 32 (RDNA) | 64 (CDNA) | TBD (likely converged) |
| ISA compatibility | gfxRDNA family | gfxCDNA family | Single UDNA ISA family |
| ALU design | RDNA shader ALU | CDNA SIMD + MFMA | GCN-inspired unified ALU |
| Forward/backward ISA compat. | Generation breaks | Generation breaks | Planned full forward + backward compat. across UDNA 6/7+ |

UDNA uses an ALU design reminiscent of the original GCN (Graphics Core Next) architecture — the common ancestor of both RDNA and CDNA — but modernized with dedicated matrix cores in all SKUs. AMD has stated a goal of forced full forward and backward ISA compatibility across the first several UDNA generations (UDNA 6 and UDNA 7) to preserve the kernel optimization work developers invest.

---

### AMD Instinct Roadmap: MI350 → MI400 → MI500

| Generation | Architecture | Process | Expected | Key Specs |
|------------|-------------|---------|----------|-----------|
| MI300X | CDNA3 | TSMC 5nm (XCD) + 6nm (IOD) | Shipping | 304 CUs, 192 GB HBM3, 5.3 TB/s, ~2.6 PFLOPS FP8 |
| MI325X | CDNA3 refresh | TSMC 5nm + 6nm | Shipping (Q4 2024) | 288 GB HBM3e, ~6 TB/s |
| MI350X / MI355X | CDNA4 | TSMC N3P (XCD) + N6 (IOD) | Shipping (2025) | 256 CUs, 288 GB HBM3e, ~8 TB/s, ~20 PFLOPS FP4 |
| **MI400** | **CDNA "Next" / UDNA transition** | TBD (est. 2nm-class) | **2026** | **40 PFLOPS FP4, 20 PFLOPS FP8, 432 GB HBM4, 19.6 TB/s, 300 GB/s scale-out/GPU** |
| MI500 | CDNA6 / UDNA | TSMC 2nm | 2027 | Up to 1,000x vs MI300X (FP4); HBM4E |

**MI400 confirmed specifications (AMD official):**

- **Peak FP4:** 40 PFLOPS (2x MI350X)
- **Peak FP8:** 20 PFLOPS
- **HBM4 capacity:** 432 GB (upgrade from HBM3e)
- **HBM4 bandwidth:** 19.6 TB/s (~2.4x MI350X's 8 TB/s)
- **Scale-out bandwidth:** 300 GB/s per GPU (Infinity Fabric / XGMI next-gen)
- **Rack solution:** "Helios" rack-scale platform — pairs MI400 GPUs with EPYC "Venice" CPUs and Pensando "Vulcano" AI NICs; targets 3 AI exaflops per rack

---

### Helios Rack-Scale Platform (2026)

AMD's "Helios" reference design is the system-level counterpart to MI400:

- Unifies MI400 GPUs + EPYC Venice CPUs + Pensando Vulcano NICs in a single rack-scale architecture
- Targets full-rack training and distributed inference at scale
- Pensando Vulcano NIC delivers next-generation scale-out networking beyond Pollara 400's 400 Gbps
- AMD and Meta announced a 6 GW deployment commitment; AMD and OpenAI announced a strategic partnership for 6 GW of AMD GPU deployments in 2026

---

### UDNA Implications for the Software Stack

1. **Single ISA target:** Future ROCm toolchains can target one UDNA GFX family instead of maintaining separate gfxRDNA and gfxCDNA targets. `--offload-arch=gfxUDNA_X` (exact naming TBD) will cover both gaming and datacenter.
2. **Matrix cores everywhere:** AITER, CK, MIOpen, and hipBLASLt MFMA kernels will be portable to consumer Radeon UDNA GPUs — opening AMD's AI kernel ecosystem to the entire Radeon product line.
3. **Forward ISA compatibility:** Compiled `.co` HSACO blobs should be reusable across UDNA generations without recompilation (a major pain point today between gfx942 and gfx950).
4. **Developer ecosystem:** The CUDA-mirroring strategy accelerates. A single ROCm install covers gaming, workstation, and datacenter GPU targets.

---

## Sources

- [AMD CDNA 3 White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Instinct MI300 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300 Series Microarchitecture — ROCm Docs](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [AMD CDNA 3 Compute Architecture — Chips and Cheese](https://chipsandcheese.com/p/amds-cdna-3-compute-architecture)
- [AMD CDNA 4 Architecture Announcement — Chips and Cheese](https://chipsandcheese.com/p/amds-cdna-4-architecture-announcement)
- [AMD MI300X Hot Chips 2024 Presentation](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf)
- [AMD Dives Deep on CDNA 4 at Hot Chips 2025 — ServeTheHome](https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/)
- [Understanding RCCL Bandwidth and xGMI on MI300X — AMD ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)
- [AMD Pensando Pollara 400 AI NIC](https://www.amd.com/en/products/network-interface-cards/pensando.html)
- [AMD MI300X Platform Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4 — AMD ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [AMD Instinct MI350 Launch Coverage — WccfTech](https://wccftech.com/amd-instinct-mi350-mi355x-launched-3nm-185-billion-transistors-288-gb-hbm3e-fp4-fp6-2-2x-faster-blackwell-b200/)
- [Understanding Data Movement in AMD Multi-GPU Systems](https://arxiv.org/html/2410.00801v1)
- [ROCm Pipeline Descriptions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html)
- [AMD Instinct MI400 — 40 PFLOPS, 432 GB HBM4 — WccfTech](https://wccftech.com/amd-instinct-mi400-accelerator-doubles-compute-40-pflops-432-gb-hbm4-memory-2026-launch/)
- [AMD Launches MI350, Confirms MI400 — VideoCardz](https://videocardz.com/newz/amd-launches-instinct-mi350-series-confirms-mi400-in-2026-with-432gb-hbm4-memory)
- [AMD Announces UDNA — Brings RDNA and CDNA Together — Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)
- [AMD RDNA and CDNA Merge into UDNA — PC Gamer](https://www.pcgamer.com/hardware/graphics-cards/from-the-developers-standpoint-they-love-this-strategyamds-plan-to-merge-its-rdna-and-cdna-gpu-architectures-to-a-unified-system-called-udna/)
- [AMD 2026-2027 AI Roadmap: MI400 and MI500 — WccfTech](https://wccftech.com/amd-to-battle-nvidia-ai-dominance-instinct-mi400-accelerators-2026-mi500-2027/)
- [AMD Instinct MI350 Series and Beyond — AMD Blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html)

---

## CDNA 5 / Instinct MI455X / Helios — Investigation Update

*Added: 2026-08-08. Scan date 2026-08-08; adversarially verified. Supersedes parts of the "MI400 / UDNA Roadmap" section above.*

### Event and status

- AMD launched the MI400-series flagship at **Advancing AI 2026**. The keynote and the AMD press release both carry the date **2026-07-23**; a "July 22–23" two-day framing is **not confirmed** by any retrieved source.
- The AMD ROCm blog "Introducing AMD CDNA™ 5 and the AMD Helios™ Rackscale Solution" is dated **2026-08-04**.
- The product was not new at launch: MI450/Helios were public from October 2025 (The Next Platform, "Altair" MI450 + Helios, Oracle first in line, 2025-10-14) and MI455X + EPYC "Venice" + Helios hardware was physically displayed at **CES 2026** (ServeTheHome, 2026-01-07). This repo's 2026-04-05 scan missed it. What is genuinely new after the baseline is the **CDNA 5 architecture disclosure, the production declaration, and the detailed specifications**.
- **Availability wording**: AMD's press release says Helios is "now in production to be deployed by leading AI companies at gigawatt scale"; on stage AMD said "Helios is in full production. Shipments will start in Q3." OpenAI "expects to bring Helios online beginning in the fourth quarter of 2026." Correct verb: **in production / shipping from Q3 2026, ramping into Q4 2026 — not deployed at scale.**

### Architecture naming (corrects a repo error)

The shipping MI400-series architecture is **CDNA 5** — not "CDNA Next" and not UDNA. UDNA/CDNA 6 attaches to **MI500 (2027)**: AMD's press release says only "Next-generation AMD Instinct MI500 Series GPUs are coming in 2027", and ServeTheHome's keynote coverage attaches **CDNA 6** to MI500. The repo's previous "~1,000x vs MI300X" MI500 figure had no primary source and has been deleted; AMD's on-stage line was a "2000x performance improvement in just 4 years" marketing claim, not a spec.

### Layer 1 — Compute Engine (CDNA 5, MI455X)

| Attribute | Value | Evidence |
|---|---|---|
| XCDs | 8 | Chips and Cheese deep-dive (2026-07-23) |
| Shader engines per XCD | 2 | Chips and Cheese |
| WGPs per shader engine | 16 | Chips and Cheese |
| WGPs per XCD | 32 active of 34 physical (2 fused off) | Chips and Cheese |
| WGPs per GPU | 256 (= 512 CUs in RDNA-style nomenclature) | Chips and Cheese |
| SIMDs per WGP | 4 dual-issue Wave32 SIMD32 | Chips and Cheese + AMD blog |
| Matrix units per WGP | 4 | Chips and Cheese |
| Max engine clock | 2.4 GHz | Chips and Cheese |
| Transistors | **not disclosed** (320B is an unverified aggregator figure) | — |
| TBP | **not disclosed** | — |

Arithmetic caution recorded for future scans: "8 shader engines per XCD, 16 WGPs each" is impossible (1,024 WGPs); and "256 active CUs / 256 WGPs" conflates CU with WGP.

**Peak throughput per GPU**: 40.26 PFLOPS MXFP4 (confirmed) and 315 TFLOPS FP32 (confirmed). 20.13 PFLOPS MXFP6/FP8 and 5.03 PFLOPS FP16/BF16 are arithmetically consistent halvings but were **not independently confirmed**. AMD's own wording is "up to 4x greater AI compute throughput for key low-precision formats" — a **vendor marketing claim**, not a specific "4x MI355X in MXFP8/MXFP4" measurement.

**New numeric / ISA features** (AMD CDNA 5 blog, no CDNA3/CDNA4 counterpart):
- **E5M3 floating-point scale format** — an extra exponent bit for wider scale range.
- **4-bit Tensor Lookup Table (LUT) instruction** for low-precision compute.
- **Tensor Data Movers** — AMD gives no microarchitectural detail; the "analogue of NVIDIA's TMA" reading is interpretation, not vendor language.

### Layer 2 — Data Path (CDNA 5)

- CDNA 5 executes **native Wave32**. AMD's CDNA 5 blog lists "Wave32 compute execution"; Chips and Cheese describes 4 dual-issue Wave32 SIMD32 units per WGP, an explicit shift from CDNA3/CDNA4's Wave64-over-SIMD16.
- **Hedge recorded**: no retrieved source states that Wave64 is *removed from the ISA*. "Drops Wave64" is **not confirmed**; only native Wave32 execution is.
- Software impact: invalidates the repo's Programming Model Rationale point #2 as an AMD-wide invariant and changes occupancy / register-pressure / LDS-bank-conflict reasoning for AITER, CK-Tile and Triton-ROCm tuning.
- Vector registers: 128 KB per SIMD (reported; not independently confirmed).

### Layer 3 — On-chip Memory (CDNA 5) — structural change

- **192 MB global L2**, split as **2 x 96 MB across two Fabric-and-Cache Dies (FCDs)**, **27 TB/s per FCD = 54 TB/s aggregate**. This replaces CDNA3's structure of 4 MB L2 per XCD plus 256 MB Infinity Cache on the IODs — there is **no separate Infinity Cache tier** in CDNA 5.
- **320 KB LDS per WGP** (Chips and Cheese) — the item that matters most for CK/AITER tiling.
- **128 KB L1 per WGP** (Chips and Cheese). A "64 KB L1 per WGP" figure circulating elsewhere is **not confirmed**.
- Evidence caveat: these are from a third-party deep-dive, not an AMD spec sheet. AMD's own blog says only "larger L2 caches".

### Layer 4 — Off-chip Memory (CDNA 5)

- **432 GB HBM4 at 23.3 TB/s per GPU**; 12 HBM4 stacks x 36 GB on a **2,048-bit bus, 192 channels**.
- AMD frames it as **"2.91x memory bandwidth"** vs the prior generation (23.3 / 8.0 ≈ 2.91 — internally consistent).
- **Repo correction**: the previously carried 19.6 TB/s figure was a pre-launch estimate and is stale.

### Layer 5 — Host Interface / Package (CDNA 5)

- **8 XCDs + 2 I/O Dies + 2 Fabric-and-Cache Dies + 12 HBM4 stacks.** The FCD is a new die type.
- Process: **TSMC N2 for compute chiplets, TSMC N3 elsewhere** — corroborated only at keynote-slide level ("2nm compute chiplets, 3nm elsewhere", ServeTheHome), not by a teardown.
- Packaging: **CoWoS-L** (Chips and Cheese). "Largest chip yet built on CoWoS-L" is a **marketing superlative, not verified**.
- **Host link: 256 GB/s bidirectional over a dedicated 16-lane Infinity Fabric to the EPYC CPU** (Chips and Cheese) — not PCIe.
- **gfx ISA target: not disclosed** (see software-stack notes).

### Layer 6 — Scale-up Interconnect (CDNA 5)

- Moves off the 8-GPU XGMI mesh to **UALink-over-Ethernet (UALoE)**.
- **36 x 400 Gb/s UALoE interfaces per GPU** = 1.8 TB/s each way = **3.6 TB/s bidirectional** (Chips and Cheese). The "72 lanes" phrasing in some coverage is **not confirmed**.
- **72-GPU single scale-up domain** (vs 8 on MI355X); **260 TB/s rack-level scale-up bandwidth** (AMD blog + ServeTheHome).
- Switched rather than meshed; the AMD–Broadcom hardware partnership is corroborated (ServeTheHome's Helios deep-dive is explicitly "AMD + Broadcom hardware combined"), but the specific **"12x Broadcom Tomahawk 6 ASICs in a 12-plane topology"** count is **not confirmed**.
- Whether UALoE switching performs in-network reduction: **not disclosed**.

### Layer 7 — Scale-out Interconnect (CDNA 5)

- **43 TB/s scale-out per rack** — confirmed (AMD blog + ServeTheHome).
- Per-GPU 2,400 Gb/s is arithmetically consistent (72 x 2,400 Gb/s ≈ 43.2 TB/s) but was **not independently stated**.
- **"3x UALink128 ports per GPU"** and **"12x Pensando Salina 400G DPUs (one per compute tray)"** are **not confirmed**; only the general use of Pensando NICs/DPUs and the **"Vulcano"** NIC name are.

### Helios rack

| Attribute | Value | Status |
|---|---|---|
| GPUs | 72x MI455X | confirmed |
| CPUs | 18x EPYC "Venice" | confirmed |
| Aggregate HBM4 | 31 TB | confirmed |
| Aggregate DDR5 | up to 36 TB | confirmed |
| Peak MXFP4 | 2.9 EFLOPS | confirmed |
| Aggregate HBM BW | ~1.7 PB/s | consistent (72 x 23.3 TB/s ≈ 1.68 PB/s); a 1.4 PB/s aggregator figure is inconsistent and discarded |
| Rack power | **not disclosed** | the "225–245 kW" figure is refuted/unsourced; AMD's press-release footnote references a "100 kW power envelope per rack" only as comparison methodology |
| Rack format, bus bar | **not disclosed** | Open Rack Wide (1.2 m x 1.3 m, 44 OU) and a 50 V liquid-cooled bus bar are reported but not confirmed against a primary AMD spec sheet |

### SKU segmentation

- **MI455X** — flagship rack-scale training/inference. Confirmed.
- **MI430X** — HPC/sovereign, **up to 288 TFLOPS hardware FP64** (AMD press release), same 432 GB HBM4; availability **H1 2027** per keynote coverage, i.e. announced, not shipping.
- **MI440X** — **not mentioned in AMD's press release** and not confirmed by any retrieved source. Deliberately not entered in the repo.
- **MI350P** — **CDNA 4 / gfx950** PCIe part (listed as "AMD Instinct MI350P (gfx950)" in the ROCm 7.14.0 compatibility matrix; Wikipedia dates its announcement to 2026-05-07). It is a CDNA 4 enterprise-inference card re-promoted at Advancing AI 2026, **not** an MI400-series/CDNA 5 SKU. The "144 GB" figure is unconfirmed.
- "MI450 / MI450-series" naming coexists with MI455X in AMD's own materials.

### Customer commitments (corporate/roadmap, not deployed capacity)

- **Anthropic**: to "deploy up to 2 gigawatts of AMD Instinct MI455X GPUs". A reported **$5B AMD investment in Anthropic is not confirmed**.
- **OpenAI**: Helios online from Q4 2026. The **6 GW** figure is a **pre-existing October 2025 agreement** restated in keynote coverage, not a new July 2026 disclosure.
- **Meta**: "has begun testing and validating workloads on AMD Helios racks" — no capacity figure. *(Repo correction: the previously recorded "AMD + Meta: 6 GW of AMD GPU deployments" was wrong; 6 GW belongs to OpenAI.)*
- **Microsoft**: named among Helios adopters, no capacity figure in the press release.

### Method caveat

The adversarial verification for this update ran with the session's WebSearch budget exhausted, so confirmation came from directly fetched pages (ServeTheHome, AMD IR, AMD ROCm blogs, ROCm docs, GitHub, Chips and Cheese, Wikipedia, The Next Platform) rather than broad search. Items marked "not confirmed" may still be true and are worth one more pass with search available.

### Sources (CDNA 5 update)

- [Introducing AMD CDNA 5 and the AMD Helios Rackscale Solution — AMD ROCm Blogs (2026-08-04)](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html)
- [AAI 2026 press release — AMD Investor Relations (2026-07-23)](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era)
- [AMD's Instinct MI455X: Aiming for the Top — Chips and Cheese (2026-07-23)](https://chipsandcheese.com/p/amds-instinct-mi455x-aiming-for-the)
- [AMD Helios Architecture Deep Dive — ServeTheHome](https://www.servethehome.com/amd-helios-architecture-deep-dive-amd-broadcom-hardware-combined/)
- [AMD Advancing AI 2026 Keynote Live Coverage — ServeTheHome](https://www.servethehome.com/amd-advancing-ai-2026-keynote-live-coverage/)
- [AMD EPYC Venice, Instinct MI455X and Helios Hardware at CES 2026 — ServeTheHome (2026-01-07)](https://www.servethehome.com/amds-epyc-venice-instinct-mi455x-helios-hardware-on-display-for-first-time-at-ces-2026/)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Oracle Leads AI Innovation with AMD "Altair" MI450 GPUs and Helios Racks — The Next Platform (2025-10-14)](https://www.nextplatform.com/compute/2025/10/14/oracle-leads-ai-innovation-with-amd-altair-mi450-gpus-and-helios-racks/1632443)
- [AMD Instinct — Wikipedia](https://en.wikipedia.org/wiki/AMD_Instinct)
