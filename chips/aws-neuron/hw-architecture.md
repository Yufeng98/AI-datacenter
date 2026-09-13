# AWS Neuron — Hardware Architecture

*as_of: 2026-08-08*
*device_class: Systolic Array Accelerator*

---

## NeuronCore Design Philosophy

AWS NeuronCore is built around three hardware commitments that differentiate it from GPU architectures:

1. **Software-managed SRAM scratchpad, no hardware cache.** SBUF (State Buffer) and PSUM (Partial Sum Buffer) are explicitly addressed by the compiler or NKI programmer. There is no cache coherency protocol. Deterministic memory access latency enables tight, latency-hiding compiler schedules.

2. **Heterogeneous fixed-function compute engines in a static producer-consumer pipeline.** Tensor, Vector, Scalar, and GPSIMD engines are independently pipelined and compiler-scheduled. GPSIMD serves as a general-purpose escape hatch.

3. **Scale-up interconnect (NeuronLink) + cloud EFA for scale-out.** The NeuronLink topology has evolved each generation, reshaping parallelism strategy. EFA decouples scale-out networking from chip vendor IP. As of Trainium3 the scale-up side is also standards-based: AWS documentation added 2026-04-09 discloses that Trn3 chip-to-chip traffic runs over a **PCIe Gen6 switch fabric**, replacing the point-to-point NeuronLink topology of Trn1/Trn2 (see "Trn3 Scale-Up Fabric" below).

---

## NeuronCore Generation Comparison

| Feature | NeuronCore-v2 (Trn1/Inf2) | NeuronCore-v3 (Trn2) | NeuronCore-v4 (Trn3) |
|---------|--------------------------|---------------------|---------------------|
| Process | TSMC 7nm | TSMC 7nm | TSMC 3nm N3P |
| Package | Monolithic | Monolithic | Dual-chiplet |
| Tensor Engine | 128×128 BF16 | 128×128 BF16 | 512×128 MXFP8/MXFP4 (OCP) |
| Peak FP8/TFLOPS per core | ~79 BF16 TFLOPS | 158 cFP8 / 79 BF16 TFLOPS | 315 MXFP8 TFLOPS |
| Vector Engine | ~2.3 TFLOPS FP32 | Expanded | Enhanced |
| Scalar Engine | ~2.9 TFLOPS FP32 | Retained | Enhanced |
| GPSIMD | 8× 512-bit | 8× 512-bit (improved ISA) | 8× 512-bit |
| DGE | No | Yes | Yes (enhanced) |
| SBUF | 24 MiB, 128 partitions | 28 MiB, 128 partitions | 32 MiB, 128 partitions |
| PSUM | 2 MiB | 2 MiB | 2 MiB |
| HBM per chip | 32 GiB HBM2e | 96 GiB HBM2e | 144 GiB HBM3e (4 stacks) |
| HBM BW per chip | 820 GB/s | 2.9 TB/s | 4.9 TB/s |
| Cores per chip | 16 (Trn1) / 2 (Inf2) | 8 | 8 |
| Chip aggregate compute | ~1.4 PF FP16 (Trn1) | 1.3 PF FP8 | 2,517 TFLOPS MXFP8/MXFP4 (AWS Trn3 spec table) |
| Chip aggregate FP16/BF16/TF32 | — | — | 671 TFLOPS *(derived: 42,944 ÷ 64 devices)* |
| Chip aggregate FP32 | — | — | 183 TFLOPS *(derived: 11,712 ÷ 64 devices)* |
| Scale-up fabric substrate | Point-to-point NeuronLink (copper) | Point-to-point NeuronLink (copper) | **PCIe Gen6 switch fabric** (NeuronSwitch-v1) — added to AWS docs 2026-04-09 |
| Scale-up domain | 16 chips (trn1.32xlarge) | 64 chips (Trn2 UltraServer) | 64 chips (UltraServer Gen1) / 144 chips (UltraServer Gen2) |

*Derived rows divide the AWS Trn3 UltraServer Gen1 spec-table aggregates by the stated 64-device domain size; Gen2 aggregates divided by 144 devices give the same per-device values.*

---

## Compute Engine

### Tensor Engine

The Tensor Engine is a systolic matrix multiplication array. On NeuronCore-v2 and v3 it is 128×128 with BF16/FP16/TF32 data types. NeuronCore-v4 expands to 512×128 and adopts OCP MicroScaling data formats (MXFP8, MXFP4), approximately doubling per-core throughput at the same precision.

The NKI intrinsic `nki.isa.tensor_tensor` dispatches directly to this engine. The compiler (`neuronx-cc`) generates Tensor Engine instructions during its NeuronCore back-end pass from MLIR lowering of HLO matmul ops.

### Vector Engine

Handles parallel vector operations: element-wise reductions, activation functions (ReLU, GELU, Swish), softmax, and layer normalization. Reads from PSUM (Tensor Engine output) and writes results to SBUF. Dispatched via `nki.isa.activation`.

### Scalar Engine

Element-wise scalar operations on individual tensor elements. Used for control-flow-light normalization steps and scalar arithmetic. Dispatched via `nki.isa.scalar`.

### GPSIMD Engine

Eight 512-bit wide programmable vector processors per NeuronCore, capable of running general-purpose C code compiled offline. Can access SBUF directly. GPSIMD provides an escape hatch for custom operations not efficiently expressed in Tensor/Vector/Scalar engine instructions (e.g., custom quantization kernels, non-standard activation variants).

### DGE — Dynamic Grouping Engine (NeuronCore-v3/v4)

Introduced on NeuronCore-v3 and enhanced on v4. Performs dynamic data grouping and token rearrangement in hardware, enabling:
- **Structured sparsity**: groups sparse activations into dense tiles for Tensor Engine consumption.
- **MoE token routing**: routes tokens to expert weight tiles without software scatter/gather, critical for efficient Mixture-of-Experts training and inference.

DGE feeds prepared token groups directly into the Tensor Engine's input port, eliminating a CPU/software overhead that would otherwise bottleneck large MoE models.

---

## Data Path

### SBUF / PSUM Memory Model

The NeuronCore's data path is a **software-directed staged pipeline**:

```
HBM (off-chip, per NeuronDevice)
    │  DMA Engine: explicit nl.load / nl.store
    ▼
SBUF (State Buffer — software-managed SRAM scratchpad)
    │  128 partitions; NKI nl.ndarray maps tiles to partition addresses
    │  Compiler statically allocates partitions (no bank conflicts)
    ▼
Tensor Engine (systolic array GEMM)
    │  128×128 (v2/v3) or 512×128 (v4) matmul
    ▼
PSUM (Partial Sum Buffer — 2 MiB SRAM)
    │  Accumulates partial results across tile-level GEMM iterations
    ▼
Vector Engine (activation, normalization, reduction)
    │
    ▼
SBUF (result tile — ready for next layer or DMA store to HBM)
    │  DMA Engine
    ▼
HBM (store output activations / weight gradients)
```

### DMA Engine

The DMA engine handles all HBM↔SBUF and HBM↔PSUM transfers, operating independently from the compute engines. This independence enables **double-buffering**: while the Tensor Engine computes on tile N in SBUF, the DMA engine prefetches tile N+1 from HBM. Both `neuronx-cc` and NKI kernel authors leverage this to overlap memory latency with computation.

### Engine Pipeline Execution

The four compute engines execute **asynchronously** in a producer-consumer chain. A typical transformer attention layer executes as:

1. DMA: load weight tile from HBM → SBUF
2. Tensor Engine: GEMM (activation × weight) → PSUM
3. DMA (concurrent): prefetch next weight tile
4. Vector Engine: PSUM → apply attention scale + softmax → SBUF
5. Scalar Engine: layer normalization → SBUF
6. DMA: store output activation SBUF → HBM

---

## On-chip Memory

| Buffer | v2 Size | v3 Size | v4 Size | Partitions | Managed By |
|--------|---------|---------|---------|-----------|------------|
| SBUF (State Buffer) | 24 MiB | 28 MiB | 32 MiB | 128 (all gens) | Software (compiler / NKI `nl.ndarray`) |
| PSUM (Partial Sum Buffer) | 2 MiB | 2 MiB | 2 MiB | N/A | Hardware (direct Tensor Engine output) |

**SBUF growth rationale**: Each 4 MiB increase per generation reduces HBM refetch frequency for large attention heads and growing MoE expert weight tiles. At 32 MiB (v4), the scratchpad can hold a full 128-head attention tile computation without intermediate HBM round-trips.

**PSUM invariance**: The 2 MiB PSUM has not grown across generations — it is sized for the worst-case partial sum accumulation depth of the Tensor Engine's systolic array, which has not required expansion even as the array grew from 128×128 to 512×128 on v4.

---

## Off-chip Memory (HBM)

HBM is a **NeuronDevice-level resource shared across all NeuronCore instances** in the chip. There is no per-NeuronCore HBM partition — all cores access HBM through shared DMA engines.

| Generation | HBM Type | Per-Chip Capacity | Per-Chip Bandwidth | Instance Context |
|------------|----------|-------------------|-------------------|-----------------|
| NeuronCore-v2 | HBM2e | 32 GiB | 820 GB/s | trn1.32xlarge: 128 GiB total |
| NeuronCore-v3 | HBM2e | 96 GiB | 2.9 TB/s | trn2.48xlarge: 1.5 TiB total; UltraServer: 6 TiB |
| NeuronCore-v4 | HBM3e (4 stacks) | 144 GiB | 4.9 TB/s | Trn3 UltraServer Gen1 (64 chips): 9,216 GiB @ 313.6 TB/s; Gen2 (144 chips): 20,736 GiB @ 705.6 TB/s |

Model weights stored in NEFF are transferred from host DRAM through PCIe to HBM by `libnrt.so` at model load time. During inference/training, the DMA engine streams weight tiles from HBM to SBUF on demand, overlapping with compute.

---

## Host Interface / Package

### PCIe and Nitro

NeuronDevices connect to the EC2 instance host CPU via PCIe. The `aws-neuronx-dkms` kernel module provides the software interface (BAR mapping, command queue management, interrupt routing). AWS Nitro integration provides direct device passthrough, eliminating hypervisor overhead and achieving near-bare-metal PCIe throughput.

### Package Evolution

| Generation | Process | Package | Notes |
|------------|---------|---------|-------|
| Trn1/Inf2 | TSMC 7nm | Monolithic | NeuronCore-v2 |
| Trn2 | TSMC 7nm | Monolithic | NeuronCore-v3; 8 cores per chip |
| Trn3 | TSMC 3nm N3P | Dual-chiplet | NeuronCore-v4; NeuronLink-v4 provides chiplet coherency |

The Trainium3 dual-chiplet design on 3nm N3P enables scaling beyond monolithic reticle area limits while NeuronLink-v4 maintains sufficient inter-chiplet memory bandwidth. AWS's Trn3 spec table gives NeuronLink-v4 as **2,048 GiB/s per device**; the "2.5 TB/s" figure previously carried here is a SemiAnalysis (third-party) number and is not reconciled with AWS's own figures — see the bandwidth note under "Trn3 Scale-Up Fabric" below.

---

## Scale-up Interconnect: NeuronLink

NeuronLink is AWS's proprietary chip-to-chip interconnect for intra-node and intra-UltraServer communication. It carries collective communication traffic (AllReduce, AllGather, AllToAll) dispatched by NCCom without CPU involvement.

| Generation | Chip | BW | Topology | Max Chips | Physical |
|------------|------|----|----------|-----------|---------|
| NeuronLink-v2 | Trainium1 | 768 GB/s bidir | 2D Torus | 16 (trn1.32xlarge) | Copper backplane |
| NeuronLink-v3 | Trainium2 | 1 TB/s bidir | 4×4 2D Torus (node) / 3D Torus (UltraServer) | 64 (Trn2 UltraServer) | ~6,100 copper cables |
| NeuronLink-v4 + NeuronSwitch-v1 | Trainium3 | 2,048 GiB/s per device (AWS spec table); 2.5 TB/s bidir (SemiAnalysis) — unreconciled | All-to-all **PCIe Gen6 switched** | 64 (UltraServer Gen1) / 144 (UltraServer Gen2) | PCIe Gen6 ×8 link groups over copper (~5,100 cables reported for the 144-chip domain) |

### Topology Impact on Collective Performance

**2D/3D Torus (Trn1/Trn2)**: Ring-AllReduce traverses torus dimensions at full NeuronLink bandwidth, making Tensor Parallelism ring AllReduce hardware-efficient. All-to-all patterns (MoE expert dispatch) require O(√N) hops through intermediate nodes — expensive at 64-chip scale.

**All-to-all switched fabric (Trn3/NeuronSwitch-v1)**: Every chip is directly reachable from every other chip within the UltraServer domain via NeuronSwitch-v1. AllToAll, AllReduce, and AllGather all operate at O(1) hop count. This eliminates the torus bandwidth asymmetry between near and far neighbors and makes MoE expert parallelism hardware-efficient for the first time in the Neuron lineage.

**Inferentia2 NeuronLink**: Connects up to 12 Inferentia2 chips at 192 GB/s for multi-chip model sharding in inference (tensor parallelism across chips for large LLMs).

### Trn3 Scale-Up Fabric — PCIe Gen6 Switching (documented 2026-04-09)

*Added 2026-08-08. Source: the "Trn3 UltraServer Connectivity and Networking" section appended to the AWS Trn3 architecture page on 2026-04-09 (shipped with Neuron SDK 2.29.0); still the live page as of 2026-08-08.*

AWS discloses that Trainium3 uses a **PCIe switch-based interconnect for all chip-to-chip communication**, explicitly "replacing the point-to-point NeuronLink topology used in previous generations (Trn1, Trn2)". NeuronSwitch-v1 is therefore a PCIe Gen6 switch fabric, and NeuronLink-v4 is carried over PCIe Gen6 ×8 lane groups.

| Scope | Links per chip | Aggregate bidirectional | Function |
|---|---|---|---|
| Intra-server | 4 × PCIe Gen6 ×8 → intra-server switch | 256 GB/s | 4-chip sled |
| Inter-server (within rack) | 5 × PCIe Gen6 ×8 → inter-server switches | 320 GB/s | Server-to-server inside the UltraServer domain |
| Inter-rack | 2 × PCIe Gen6 ×8 direct | 128 GB/s | Rack-to-rack (Gen2 UltraServer spans 36 servers) |

**Addressing and routing.** Each chip is identified by a **(rack, server, chip) tuple encoded in the upper bits of the outbound PCIe address**. Switches perform BAR address matching to select the output port. Routing is therefore address-based rather than table-driven at the message level, and is transparent to workloads — the Neuron runtime and compiler configure it.

**Ordering.** The fabric provides **hardware semaphores** for cross-fabric synchronization. Data and its completion semaphore are guaranteed to traverse the same physical path, so ordering is enforced by the fabric rather than by software fences — the hardware basis for NCCom's CPU-bypass collectives on Trn3.

**Architectural significance.** With Trn3, both tiers of the Neuron interconnect run on industry-standard transports: PCIe Gen6 for scale-up and EFA for scale-out. AWS's differentiation moves to topology, address encoding, and the semaphore ordering model rather than a bespoke link layer — the sharpest available contrast with NVLink/NVSwitch, which remains a custom link layer end to end.

> ⚠️ **Unreconciled bandwidth figures.** The per-chip link budget above sums to 256 + 320 + 128 = **704 GB/s**, which cannot be reconciled with the same AWS page's spec-table figure of **2,048 GiB/s per device** for NeuronLink-v4. The **2.5 TB/s bidirectional** figure this survey previously carried matches neither AWS number and appears to originate with SemiAnalysis. We cite 2,048 GiB/s/device as the AWS vendor spec-table value and record the 704 GB/s link-level sum and the 2.5 TB/s third-party figure as unreconciled. No primary source reconciles them; do not average or silently pick one.

### Trn3 UltraServer Configurations

*Note: these figures have been published on the AWS Trn3 architecture page since 2025-12-02 (re:Invent). Their absence from earlier revisions of this survey was a research gap, not a pending disclosure.*

| Metric | UltraServer Gen1 | UltraServer Gen2 |
|---|---|---|
| Trainium3 devices | 64 (4 servers) | 144 (36 servers) |
| Switch hierarchy | NeuronSwitch-v1 within the scale-up domain | First-level NeuronSwitch-v1 within server + two second-level NeuronSwitch-v1 across servers |
| HBM3e capacity | 9,216 GiB | 20,736 GiB |
| HBM bandwidth | 313.6 TB/s | 705.6 TB/s |
| MXFP8 / MXFP4 | 161,088 TFLOPS | 362,448 TFLOPS |
| FP16 / BF16 / TF32 | 42,944 TFLOPS | 96,624 TFLOPS |
| FP32 | 11,712 TFLOPS | 26,352 TFLOPS |
| EFA | 12,800 Gbps | 28,800 Gbps |
| Host vCPUs | 768 | 2,304 |
| Host memory | 8,192 GiB | 27,648 GiB |

AWS states the all-to-all connectivity is optimized for MoE and autoregressive inference serving. **Trn3 instance availability (GA vs preview), instance sizes, and regions are not disclosed** — the EC2 Trn3 product page publishes only UltraServer aggregates.

---

## Scale-out Interconnect: EFA + UltraClusters

| Version | Instance | Per-Instance BW | UltraServer BW |
|---------|----------|-----------------|----------------|
| EFAv2 | Trn1 | 1,600 Gbps | N/A |
| EFAv3 | Trn2 | 3.2 Tbps | 12.8 Tbps |
| EFAv3 | Trn3 | not disclosed per instance (AWS reports Trn3 EFA at UltraServer granularity) | 12,800 Gbps (Gen1, 64 devices) / 28,800 Gbps (Gen2, 144 devices) — a consistent **200 Gbps of EFA per Trainium3 device** |

NCCom automatically selects EFA for collectives spanning different physical nodes (ranks not reachable via NeuronLink). EFA uses the SRD (Scalable Reliable Datagram) protocol with a Neuron-specific transport plugin.

**EC2 UltraClusters** interconnect multiple UltraServers with petabit-scale EFA networking. The bifurcated communication strategy — NeuronLink for fast intra-node TP/SP collectives, EFA for inter-node DP/PP gradient synchronization — is the architectural basis of NxD Core's 3D parallelism process group hierarchy.

**Project Rainier scale**: AWS deployed ~500,000 Trainium2 chips for Anthropic's LLM training workloads in 2025 using UltraCluster EFA networking, validating petabit-scale EFA viability for hundreds of thousands of chips. **Updated 2026-08-08**: Anthropic's 2026-04-20 newsroom post states it now uses **over one million Trainium2 chips** to train and serve Claude, with significant new Trn2 capacity online in Q2 2026 and scaled Trainium3 later in 2026 — nearly 1 GW of combined Trn2 + Trn3 by end of 2026, within an announced envelope of up to 5 GW of new capacity. These are the parties' own statements, not independently audited deployment counts.

---

## Roadmap and Disclosure Status (2026-08-08)

**Trainium4 / NeuronCore-v5 — roadmap unchanged.** The re:Invent (December 2025) preview claims remain the only public description: ≥6× FP4, 3× FP8, 4× memory bandwidth and 2× memory capacity versus Trainium3 via 8 stacks of HBM4, plus participation in a shared scale-up fabric through NVIDIA NVLink 6 / NVLink Fusion; availability reported as late 2026–2027. These are **vendor preview claims**, not independently confirmed specifications, and no new primary-source confirmation was found between 2026-04-05 and 2026-08-08. Process node, die configuration, and NeuronCore-v5 microarchitecture are **not disclosed**.

**Hot Chips 38 (Aug 23–25, 2026).** AWS / Amazon / Annapurna Labs does **not** appear in the HC38 program; no Trainium or Neuron talk is scheduled.

**MLPerf.** No AWS/Amazon/Trainium submission appears in MLPerf Training v6.0 (published 2026-06-16, 24 submitters). No Trainium MLPerf result is recorded in this survey.
