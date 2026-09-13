# AWS Neuron Hardware Architecture — Investigation Report

**Chip:** AWS Neuron (Trainium/Inferentia)
**Investigation Focus:** NeuronCore generations, memory hierarchy, interconnects, host interface
**Date:** 2026-04-05

---

## Overview

AWS has produced four generations of custom ML silicon under the Trainium and Inferentia brands, each centered on an increasingly capable NeuronCore compute unit. The architecture follows a consistent design philosophy: software-managed SRAM (no hardware cache), heterogeneous compute engines (Tensor/Vector/Scalar/GPSIMD), HBM off-chip memory, and a proprietary scale-up interconnect (NeuronLink) complemented by EFA for scale-out. Trainium chips target training; Inferentia chips target inference (and increasingly support both). The progression spans NeuronCore-v2 (Trainium1/Inferentia2, 2022-2023), NeuronCore-v3 (Trainium2, 2024), NeuronCore-v4 (Trainium3, 2025), with Trainium4/NeuronCore-v5 announced for the future roadmap.

---

## Architecture

### NeuronCore-v2 (Trainium1 / Inferentia2)

**Process node:** 7nm (TSMC)
**Chips using this core:** Trainium1 (16 NeuronCore-v2 per chip), Inferentia2 (2 NeuronCore-v2 per chip)

**Compute engines (per NeuronCore):**
- **Tensor Engine** — Systolic array optimized for GEMM/CONV/Transpose. >90 TFLOPS FP16/BF16 (6× NeuronCore-v1). Mixed precision: FP16, BF16, TF32, INT8, FP32.
- **Vector Engine** — Parallelized for reduction operations and activation functions. ~2.3 TFLOPS FP32 (10× NeuronCore-v1).
- **Scalar Engine** — Element-wise scalar operations. ~2.9 TFLOPS FP32 (3× NeuronCore-v1).
- **GPSIMD Engine** — 8× 512-bit wide fully programmable vector processors running general-purpose C code; can access on-chip SRAM directly for custom operators.

**On-chip memory:**
- SBUF (State Buffer): 24 MiB software-managed SRAM; 128 partitions
- PSUM (Partial Sum Buffer): 2 MiB software-managed SRAM; holds intermediate Tensor Engine outputs

**Off-chip memory (per NeuronDevice):**
- Trainium1: 32 GiB HBM2e per chip, 820 GB/s HBM bandwidth
- Inferentia2: 32 GiB HBM per chip, 380 INT8 TOPS throughput, 4× throughput over Inf1

**Instance context:**
- `trn1.32xlarge`: 16 NeuronCore-v2, 128 GiB NeuronDevice memory, NeuronLink-v2
- `inf2.48xlarge`: 12 Inferentia2 chips, 384 GiB device memory

---

### NeuronCore-v3 (Trainium2)

**Process node:** 7nm (TSMC)
**Chips using this core:** Trainium2 (8 NeuronCore-v3 per chip)

**Compute engines (per NeuronCore):**
- **Tensor Engine** — 128×128 BF16 systolic array. 158 cFP8 TFLOPS (FP8 compressed), 79 BF16/FP16/TF32 TFLOPS, 20 FP32 TFLOPS.
- **Vector Engine** — Expanded parallel width vs. v2.
- **Scalar Engine** — Retained from v2 design.
- **GPSIMD Engine** — 8× 512-bit programmable vector processors (same count as v2, improved ISA).
- **DGE (Dynamic Grouping Engine)** — New hardware block on v3. Provides dynamic data grouping and rearrangement to support structured sparsity; feeds Tensor Engine to enable sparse computation at hardware speed.

**On-chip memory:**
- SBUF: **28 MiB** (up from 24 MiB in v2); 128 partitions × 224 KiB per partition
- PSUM: 2 MiB (unchanged from v2)

**Off-chip memory (per Trainium2 chip):**
- **96 GiB HBM2e**, 2.9 TB/s HBM bandwidth per chip (per AWS documentation)

**Trainium2 chip aggregate:** 8 NeuronCore-v3 → 1.3 PF FP8 compute per chip

**Instance context:**
- `trn2.48xlarge`: 16 Trainium2 chips, 1.5 TiB total HBM, 46 TB/s aggregate HBM bandwidth
- `trn2u.48xlarge` (UltraServer): 64 Trainium2 chips via 4× trn2.48xlarge in NeuronLink 4×4 2D Torus topology

---

### NeuronCore-v4 (Trainium3)

**Process node:** 3nm N3P (TSMC)
**Chips using this core:** Trainium3 (8 NeuronCore-v4 per chip, dual-chiplet design)

**Compute engines (per NeuronCore):**
- **Tensor Engine** — 512×128 MXFP8/MXFP4 systolic array (OCP compliant data formats). **315 MXFP8/MXFP4 TFLOPS** per NeuronCore (≈2× NeuronCore-v3). Supports FP16, BF16, TF32, FP32, MXFP8, MXFP4 mixed precision.
- **Vector Engine** — Enhanced.
- **Scalar Engine** — Enhanced.
- **GPSIMD Engine** — 8× 512-bit programmable vector processors.
- **DGE** — Retained and enhanced from v3; structured sparsity support.

**On-chip memory:**
- SBUF: **32 MiB** (up from 28 MiB in v3); 128 partitions
- PSUM: 2 MiB (unchanged)

**Off-chip memory (per Trainium3 chip):**
- **144 GiB HBM3e** (4 stacks), **4.9 TB/s** HBM bandwidth per chip

**Trainium3 chip aggregate:** 8 NeuronCore-v4 → **2.52 PFLOPs MXFP8** per chip

**Instance context:**
- Trn3 UltraServer Gen1: 64 Trainium3 chips, NeuronSwitch-v1
- Trn3 UltraServer Gen2: 144 Trainium3 chips (NL72×2 switched), up to 20 TB HBM

---

### NeuronCore Generation Comparison Table

| Feature | NeuronCore-v2 (Trn1/Inf2) | NeuronCore-v3 (Trn2) | NeuronCore-v4 (Trn3) |
|---|---|---|---|
| Process | 7nm | 7nm | 3nm N3P |
| Tensor Engine | 128×128 BF16 | 128×128 BF16 | 512×128 MXFP8/MXFP4 |
| Peak FP8 (per core) | ~79 BF16 TFLOPS | 158 cFP8 TFLOPS | 315 MXFP8 TFLOPS |
| SBUF | 24 MiB | 28 MiB | 32 MiB |
| PSUM | 2 MiB | 2 MiB | 2 MiB |
| SBUF partitions | 128 | 128 | 128 |
| DGE | No | Yes | Yes (enhanced) |
| Sparsity | No | Structured | Structured |
| HBM (per chip) | 32 GiB | 96 GiB | 144 GiB |
| HBM BW (per chip) | 820 GB/s | 2.9 TB/s | 4.9 TB/s |
| Cores per chip | 16 (Trn1) / 2 (Inf2) | 8 | 8 |

---

## Data Path

### SBUF / PSUM Memory Model

The NeuronCore uses a **software-managed scratchpad architecture** — there is no hardware cache or cache coherency protocol. All data movement between HBM and SBUF/PSUM is explicitly managed by the compiler (or NKI programmer).

```
HBM (off-chip)
    │  DMA Engine (explicit load/store)
    ▼
SBUF (State Buffer — on-chip SRAM)
    │  128 independent partitions
    │  Data tiles staged here for compute
    ▼
Tensor Engine (systolic array)
    │  Matrix multiply output
    ▼
PSUM (Partial Sum Buffer — 2 MiB)
    │  Accumulates partial results across tiles
    ▼
Vector/Scalar Engine (activation, normalization)
    │
    ▼
SBUF (result tile — for next layer or store back to HBM)
```

**DMA engines** handle all HBM↔SBUF and HBM↔PSUM transfers, operating independently from the compute engines. The NKI compiler (and neuronx-cc) overlaps DMA prefetch with compute to hide HBM latency — a double-buffering pattern where one tile is being computed while the next is being DMA'd in.

### Pipelining

The four compute engines (Tensor, Vector, Scalar, GPSIMD) execute **asynchronously** in a producer-consumer pipeline. A typical transformer layer executes as:

1. DMA: load weight tile from HBM to SBUF.
2. Tensor Engine: GEMM on activation × weight → PSUM.
3. DMA: pre-load next tile while Tensor Engine runs.
4. Vector Engine: read from PSUM → apply activation function → write to SBUF.
5. Scalar Engine: normalization.
6. DMA: store output tile from SBUF back to HBM.

---

## On-Chip Memory Details

| Buffer | v2 Size | v3 Size | v4 Size | Type | Managed by |
|---|---|---|---|---|---|
| SBUF | 24 MiB | 28 MiB | 32 MiB | SRAM | Software (compiler / NKI) |
| PSUM | 2 MiB | 2 MiB | 2 MiB | SRAM | Hardware (Tensor Engine output) |

SBUF is organized as 128 partitions. NKI's `nl.ndarray` maps tile dimensions to partition indices. The compiler statically allocates SBUF partitions to avoid bank conflicts during concurrent DMA and compute.

---

## Off-Chip Memory (HBM)

HBM resides within the NeuronDevice (not per-NeuronCore). On Trainium2, 96 GiB HBM2e is shared across all 8 NeuronCore-v3 instances in the chip, accessed via shared DMA engines. On Trainium3, 144 GiB HBM3e (4 stacks) with 4.9 TB/s per chip.

Weights for large models that exceed SBUF are streamed from HBM on demand. The NEFF artifact stores weights in HBM-resident format; libnrt.so handles initial transfer from host DRAM to HBM at model load time.

---

## Host Interface

- **PCIe:** NeuronDevices connect to the host CPU via PCIe. The `aws-neuronx-dkms` kernel mode driver handles PCIe BAR mapping, command queue management, and interrupt routing.
- **Nitro integration:** EC2 Trainium instances use the AWS Nitro system. The Nitro hypervisor provides direct device passthrough to the instance, enabling near-bare-metal PCIe throughput. Device enumeration and memory-mapped I/O are managed via the Nitro device model.
- **Host DRAM to HBM:** Model weights are transferred from host DRAM through PCIe to HBM at model load time via DMA. This is orchestrated by libnrt.so.

---

## Scale-Up: NeuronLink

NeuronLink is AWS's proprietary chip-to-chip interconnect for intra-node scale-up.

| Generation | Chip | Bandwidth | Topology |
|---|---|---|---|
| NeuronLink-v2 | Trainium1 | 768 GB/s bidirectional per link | 2D Torus (trn1.32xlarge: 16 chips) |
| NeuronLink-v3 | Trainium2 | **1 TB/s** bidirectional | 4×4 2D Torus (Trn2: 16 chips per node); 3D Torus in UltraServer (64 chips) |
| NeuronLink-v4 | Trainium3 | **2.5 TB/s** bidirectional | Switched via NeuronSwitch-v1 (all-to-all) |

**Trn1 (NeuronLink-v2):** 16 chips in a 2D Torus. EFAv2 up to 1,600 Gbps per instance.

**Trn2 (NeuronLink-v3):** 16 chips per node in 4×4 2D Torus. Trn2 UltraServer: 64 chips via NL32×2 3D Torus (≈6,100 copper cables); 1 TiB/s NeuronLink bandwidth within UltraServer. EFAv3 at 3.2 Tbps per trn2 instance, 12.8 Tbps per UltraServer.

**Trn3 (NeuronLink-v4 + NeuronSwitch-v1):**
- NeuronSwitch-v1 is an all-to-all switched fabric replacing the 2D/3D Torus of previous generations.
- Doubles intra-chip bandwidth vs. Trn2 UltraServer.
- Reduces latency for all-to-all patterns, optimized for MoE expert routing and autoregressive inference.
- Trn3 NL32×2 Switched: 64 chips. Trn3 NL72×2 Switched: **144 chips** per UltraServer (≈5,100 copper cables).
- Total compute per Trn3 UltraServer (144 chips): up to 20 TB HBM, 706 TB/s HBM bandwidth.

---

## Scale-Out: EFA and UltraClusters

- **EFAv2** (Trn1): up to 1,600 Gbps per instance for inter-node collective communication.
- **EFAv3** (Trn2/Trn3): 3.2 Tbps per trn2 instance; 12.8 Tbps per trn2 UltraServer; higher bandwidth for Trn3 UltraServers.
- **UltraClusters:** Multiple UltraServers are interconnected with **petabit-scale EFA networking**, enabling scale-out to thousands of chips for LLM training at the scale of Project Rainier (≈500,000 Trainium2 chips for Anthropic workloads).
- Collective communication across nodes routes through EFA; intra-node collectives route through NeuronLink — this bifurcation shapes parallelism strategy (tensor parallelism within node, data/pipeline parallelism across nodes).

---

## Key Findings

1. **SBUF growth tracks model size:** SBUF has grown from 24 MiB (v2) to 28 MiB (v3) to 32 MiB (v4), enabling larger tile sizes and reducing HBM accesses per layer — critical for attention head computation and MoE expert activation.
2. **DGE enables sparse MoE:** The Dynamic Grouping Engine on v3/v4 provides hardware-accelerated token routing for Mixture-of-Experts models; NeuronSwitch-v1 on Trn3 provides the interconnect complement for expert parallelism at scale.
3. **NeuronSwitch replaces Torus:** The shift from 2D/3D Torus (Trn1/Trn2) to an all-to-all switched fabric (Trn3 via NeuronSwitch-v1) halves worst-case hop count for all-to-all collectives; important for expert parallelism in MoE.
4. **3nm chiplet design on Trn3:** Trainium3 is a dual-chiplet design on TSMC 3nm N3P, enabling die area scaling beyond monolithic reticle limits while NeuronLink-v4 at 2.5 TB/s maintains chiplet coherency bandwidth.
5. **MXFP8/MXFP4 on v4:** NeuronCore-v4 adopts OCP MicroScaling floating point formats, providing a standardized microscaling approach for quantization that is hardware-accelerated in the Tensor Engine.
6. **No hardware cache:** The software-managed SRAM design eliminates cache coherency overhead and enables predictable, compiler-scheduled memory access patterns — a design tradeoff that requires the NKI programming model for custom kernels.

---

## Relation to Software

| Hardware Component | Software Interface |
|---|---|
| Tensor Engine (systolic array) | `nki.isa.tensor_tensor`, `nl.matmul` |
| Vector Engine | `nki.isa.activation`, `nl.add` |
| Scalar Engine | `nki.isa.scalar` |
| GPSIMD Engine | Custom C code via GPSIMD NKI API |
| DGE (Dynamic Grouping Engine) | Compiler-managed; NKI MoE kernels |
| SBUF partitions | `nl.ndarray(shape, buffer=nl.sbuf)` |
| PSUM | Implicit output of `nki.isa.tensor_tensor` |
| HBM | `nl.load` / `nl.store` DMA ops |
| NeuronLink | `nki.collectives` / NCCom library |
| EFA | NCCom library (inter-node path) |
| PCIe / Nitro | `aws-neuronx-dkms` kernel driver |

---

## Update — 2026-08-08 (window 2026-04-05 → 2026-08-08)

**Investigation focus:** Trn3 UltraServer specification table, Trn3 scale-up fabric substrate, deployment scale.
**Method:** direct fetch of primary sources — GitHub API release metadata, `raw.githubusercontent.com` revisions of `about-neuron/arch/neuron-hardware/trn3-arch.rst` at four commits, the live rendered AWS doc page, the EC2 Trn3 product page, and the Anthropic newsroom. WebSearch was unavailable during verification, so absence claims below are "no evidence found", not "verified absent".

### 1. Trn3 UltraServer spec table — a repo research gap, not a window change

The prior revision of this survey recorded Trn3 EFA as "higher per-instance BW (exact GA specs pending)". That was wrong: the complete Gen1/Gen2 spec table has been published since the **2025-12-02 re:Invent commit** (`6e694cbb`) and was unchanged through the 2026-01-29 commit (`b741ad5b`). Commit history for the file:

| Commit | Date | Note |
|---|---|---|
| `db1586c3` | 2025-12-02 | Initial re:Invent version |
| `6e694cbb` | 2025-12-02 | Already contains the full Gen1/Gen2 spec table (EFA 12,800 / 28,800 Gbps etc.) |
| `b741ad5b` | 2026-01-29 | 83 lines; no PCIe connectivity section |
| `539a73c5` | 2026-04-09 | 136 lines; **adds** the "Trn3 UltraServer Connectivity and Networking" section; byte-identical to the current `master` version |

Full table as published:

| Metric | Gen1 | Gen2 |
|---|---|---|
| Trainium3 devices | 64 (4 servers) | 144 (36 servers) |
| Switching | NeuronLink-v4 + NeuronSwitch-v1 | First-level NeuronSwitch-v1 in server; two second-level NeuronSwitch-v1 across servers |
| HBM3e | 9,216 GiB | 20,736 GiB |
| HBM bandwidth | 313.6 TB/s | 705.6 TB/s |
| MXFP8 / MXFP4 | 161,088 TFLOPS | 362,448 TFLOPS |
| FP16 / BF16 / TF32 | 42,944 TFLOPS | 96,624 TFLOPS |
| FP32 | 11,712 TFLOPS | 26,352 TFLOPS |
| EFA | 12,800 Gbps | 28,800 Gbps |
| Host | 768 vCPU / 8,192 GiB | 2,304 vCPU / 27,648 GiB |

Per device: 144 GiB HBM3e @ 4.9 TB/s, 2,517 TFLOPS MXFP8/MXFP4, NeuronLink-v4 at 2,048 GiB/s. Dividing the Gen1 and Gen2 aggregates by their stated device counts yields identical per-device values, which is a useful internal consistency check on the table.

**EFA reading:** AWS reports Trn3 EFA at **UltraServer granularity**, not per instance. 12,800 / 64 = 28,800 / 144 = **200 Gbps of EFA per Trainium3 device** in both configurations. A per-instance Trn3 EFA figure is **not disclosed**.

### 2. Genuinely new in the window: Trn3 chip-to-chip is a PCIe Gen6 switch fabric

The 2026-04-09 commit (shipped with Neuron SDK 2.29.0) appended a new "Trn3 UltraServer Connectivity and Networking" section stating that Trn3 uses a PCIe switch-based interconnect for all chip-to-chip communication, "replacing the point-to-point NeuronLink topology used in previous generations (Trn1, Trn2)".

| Scope | Links per chip | Aggregate bidirectional |
|---|---|---|
| Intra-server (4-chip sled → intra-server switch) | 4 × PCIe Gen6 ×8 | 256 GB/s |
| Inter-server, within rack | 5 × PCIe Gen6 ×8 | 320 GB/s |
| Inter-rack | 2 × PCIe Gen6 ×8 direct | 128 GB/s |

- **Routing:** each chip carries a (rack, server, chip) tuple encoded in the upper bits of the outbound PCIe address; switches select the output port by BAR address matching. Transparent to workloads; configured by the runtime and compiler.
- **Synchronization:** hardware semaphores; data and its completion semaphore are guaranteed to take the same physical path, so ordering does not require software fences.

**Why this matters for the survey.** NeuronSwitch-v1 is a PCIe Gen6 switch fabric and NeuronLink-v4 rides PCIe Gen6 ×8 lane groups. This qualifies the survey's framing of NeuronLink as bespoke vendor interconnect IP: with Trn3, both Neuron interconnect tiers run on industry-standard transports (PCIe Gen6 scale-up, EFA scale-out), and the differentiation sits in topology, address encoding, and the semaphore ordering model.

### 3. Unreconciled bandwidth figures — flagged, not resolved

Three numbers for Trn3 per-device scale-up bandwidth are in circulation and none reconciles with the others:

| Figure | Source | Status |
|---|---|---|
| 2,048 GiB/s per device | AWS Trn3 architecture page, spec table | Vendor spec-table figure — cite this |
| 704 GB/s per chip (256 + 320 + 128) | Same AWS page, connectivity section link budget | AWS-internal inconsistency; record, do not reconcile |
| 2.5 TB/s bidirectional | SemiAnalysis Trainium3 deep dive | Third-party; matches neither AWS number; was previously carried in this survey without attribution |

No primary source reconciles them. Do not average, pick silently, or carry the 2.5 TB/s figure as if it were vendor-stated.

### 4. Deployment scale correction

Anthropic's 2026-04-20 newsroom post states Anthropic **currently uses over one million Trainium2 chips** to train and serve Claude. This supersedes the ~500,000-chip Project Rainier figure previously carried in three places in this survey. Also stated: up to 5 GW of new capacity; new Trn2 capacity online Q2 2026 with scaled Trainium3 later in 2026; nearly 1 GW of combined Trn2 + Trn3 by end of 2026; Amazon investing $5B immediately plus up to $20B milestone-based on top of $8B previously; Anthropic committing $100B+ to AWS over ten years. These are the parties' own statements, not audited counts.

### 5. Could not confirm

- **Trn3 instance availability.** The EC2 Trn3 product page gives only UltraServer aggregates and states neither GA nor preview; no instance sizes, no regions. Keep hedged — **not disclosed**.
- **Trainium4.** No new primary-source confirmation between 2026-04-05 and 2026-08-08. The re:Invent Dec-2025 preview claims (≥6× FP4, 3× FP8, 4× memory bandwidth, 2× capacity vs Trn3, 8 HBM4 stacks, NVLink 6 / NVLink Fusion participation, late 2026–2027) remain vendor preview only. Process node and NeuronCore-v5 microarchitecture **not disclosed**.
- **Hot Chips 38 (Aug 23–25, 2026):** AWS/Amazon/Annapurna absent from the program.
- **MLPerf Training v6.0 (2026-06-16, 24 submitters):** no AWS submission. Record no Trainium MLPerf result.
