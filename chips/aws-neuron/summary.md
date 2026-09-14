# AWS Neuron (Trainium / Inferentia) — Software & Hardware Stack Summary

*as_of: 2026-09-13*
*device_class: Systolic Array Accelerator*

---

## Overview

AWS Neuron is Amazon's custom AI silicon program, comprising the Trainium (training) and Inferentia (inference) chip lines and the unified Neuron SDK that programs them. The architecture is built around the NeuronCore compute unit — a heterogeneous, software-managed accelerator that pairs a systolic-array Tensor Engine with Vector, Scalar, and GPSIMD auxiliary engines, backed by an explicit SRAM scratchpad (no hardware cache) and HBM off-chip memory.

The chip line spans four NeuronCore generations: NeuronCore-v2 (Trainium1/Inferentia2, 7nm, 2022–2023), NeuronCore-v3 (Trainium2, 7nm, 2024), NeuronCore-v4 (Trainium3, 3nm N3P, 2025), and NeuronCore-v5 (Trainium4, announced roadmap). Each generation roughly doubles peak FP8 throughput per core and grows the on-chip SRAM scratchpad. The scale-up interconnect has evolved from a 2D Torus (NeuronLink-v2) to a 3D Torus (NeuronLink-v3) to an all-to-all switched fabric (NeuronLink-v4 + NeuronSwitch-v1) — which AWS documentation (added 2026-04-09) discloses is physically a **PCIe Gen6 switch fabric**, not a point-to-point proprietary link (see the 2026-08-08 update section). EFA (Elastic Fabric Adapter) provides the scale-out interconnect. Project Rainier reached ~500,000 Trainium2 chips in 2025; as of April 2026 Anthropic states it uses **over one million Trainium2 chips** across AWS.

---

## Software Stack

### Framework Integration

AWS Neuron supports two primary ML frameworks:

**PyTorch** is the primary target. Two integration paths coexist:

- **torch-neuronx** (current, PyTorch ≤ 2.9): Implemented on top of PyTorch/XLA. Trainium/Inferentia appears as an XLA device. `xm.mark_step()` triggers lazy evaluation and JIT graph compilation. `torch_neuronx.trace()` performs ahead-of-time (AOT) compilation to a NEFF artifact. The TorchDynamo backend (`torch.compile(backend="openxla_eval")`) captures FX graphs and routes them through XLA HLO before the Neuron compiler back-end.

- **TorchNeuron** (planned PyTorch ≥ 2.10, private beta as of Neuron SDK 2.27 — *status unverified since 2.27; no TorchNeuron/torch-neuronx entry appears in the 2.31.0 component release notes, which is not evidence either way*): Registers Trainium/Inferentia as a native PyTorch device using the **PrivateUse1** device backend mechanism, entirely eliminating the PyTorch/XLA dependency. TorchNeuron supports eager-mode tensor execution, `torch.compile` via TorchDynamo with a custom Neuron backend, and standard distributed APIs (`torch.distributed`, FSDP, DDP, DTensor) with no Neuron-specific wrappers.

**JAX** support was introduced in September 2024 (Neuron SDK 2.20). JAX traces computation graphs to StableHLO, which the Neuron compiler's XLA frontend ingests. `jax.jit` triggers compilation through the Neuron XLA backend. NKI kernels integrate into JAX graphs via `custom_call` bindings.

**NKI (Neuron Kernel Interface)** provides a framework-agnostic bare-metal tile programming API. NKI kernels can be registered as custom operators via `nki.baremetal` (PyTorch) or `custom_call` (JAX), or executed standalone on Trainium/Inferentia. NKI reached **Stable** (leaving Beta) at NKI 0.3.0 / Neuron SDK 2.29.0 on 2026-04-09; the current release is **NKI 0.6.0 / Neuron SDK 2.32.0 (2026-08-17)** — see Update (2026-09-13) below.

### Compiler / IR

The Neuron compiler (`neuronx-cc`) is a multi-phase pipeline:

1. **XLA Frontend** — accepts framework graph inputs (PyTorch FX via XLA, JAX StableHLO). Applies hardware-agnostic optimizations: constant propagation, operator fusion, re-materialization. Emits XLA HLO / StableHLO IR.

2. **MLIR Middle-end** — lowers HLO ops to Neuron dialect MLIR ops. Makes tiling, memory placement, and pipelining decisions specific to the target NeuronCore generation. Schedules SBUF partition allocation and DMA overlap.

3. **NeuronCore Back-end** — generates NeuronISA machine instructions from MLIR. Schedules the four compute engines (Tensor, Vector, Scalar, GPSIMD). Emits NEFF.

**NKI kernels** bypass the XLA frontend entirely and enter the pipeline at the MLIR back-end via the separately open-sourced **NKI Compiler** (Apache 2.0, released Neuron 2.27). This gives kernel authors direct control over SBUF tile layout and instruction scheduling.

As of Neuron SDK 2.31.0 (2026-07-07), `neuronx-cc` ships a **redesigned code-generation backend** — improved instruction scheduling and memory prefetch — which is now the **default for Trn2 and Trn3**.

**NEFF (Neuron Executable File Format)** is the canonical compiled artifact. It is a single-file container encoding NeuronISA instructions, model weights (in HBM-resident format), tensor metadata, and execution graph. All compilation paths produce NEFF. NEFF supports offline compilation, checksum-validated loading, and multi-NeuronCore execution scheduling — enabling reproducible cluster deployments.

### Op Library

AWS does not expose a separate cuDNN/MIOpen-equivalent op library layer. High-level operator fusion and graph optimization occur inside `neuronx-cc` during the XLA and MLIR compilation phases. Operator coverage is expressed through the compiler's HLO op lowering table rather than a separately versioned op library.

### Kernel Library

**NKI (nki.lang / nki.isa)** is the primary kernel programming interface:

- `nki.lang` — high-level NumPy/Triton-like tile API. `nl.ndarray` allocates SBUF tiles; `nl.load` / `nl.store` emit DMA instructions; `nl.matmul` dispatches the Tensor Engine; `nl.add` / activation functions dispatch the Vector Engine.

- `nki.isa` — low-level ISA intrinsics for direct engine dispatch (`tensor_tensor`, `activation`, `bn_stats`), explicit SBUF/PSUM partition addressing, and instruction-level DMA/compute pipelining.

- `nki.collectives` (added Neuron 2.27) — collective communication ops callable from within a NKI kernel (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `collective_permute`, `rank_id`; variable-length `all_to_all_v` added in 2.29.0). Enables fused compute+communication patterns without returning to the framework.

- `nki-stdlib` (**NKI Standard Library**, added 2.29.0) — developer-visible source for all NKI APIs, plus native NKI language objects (`NkiTensor`). `NkiTensor` gained composable zero-cost view methods (slice / select / permute / rearrange) in NKI 0.5.0.

The **nki-samples** GitHub repository provides official reference implementations: Flash Attention (io-aware), GEMM, RoPE embedding, and MoE routing kernels. The separate **NKI Library** of packaged kernels grew substantially across 2.29–2.31 (see the 2026-08-08 update section).

### Collective Communication Runtime

**NCCom (Neuron Collective Communication)** is the low-level collective runtime embedded in `libnrt.so`. Key properties:

- **CPU-bypass execution**: Collectives are dispatched from NeuronCore programs directly to NeuronLink or EFA DMA engines without a PCIe round-trip through the host CPU.
- **Automatic transport selection**: Routes to NeuronLink (intra-node, within UltraServer) or EFA (inter-node, across nodes) based on process group topology — the same distributed code works at any scale.
- **Supported ops**: AllReduce, AllGather, ReduceScatter, AllToAll, AllToAllV, Broadcast, ReduceScatter.

### Distributed Training Libraries

**NxD Core (neuronx-distributed)** provides XLA-based PyTorch distributed primitives:
- Tensor Parallelism (TP): `ColumnParallelLinear` / `RowParallelLinear` with AllGather/ReduceScatter at layer boundaries.
- Pipeline Parallelism (PP): `NxDPPModel` with 1F1B and interleaved schedules.
- Sequence Parallelism (SP): Partitions the sequence axis, reducing peak activation memory by the TP degree.
- Data Parallelism (DP) + ZeRO-1: Optimizer state sharding across DP ranks.
- Full 3D parallelism (TP × PP × DP) with a process group hierarchy mapping TP to NeuronLink-connected chips.

**NxD Training (neuronx-distributed-training)** is the high-level 3D parallelism training library. It integrates with NVIDIA NeMo Megatron configs (`neuronx-nemo-megatron`) for GPT/LLaMA pretraining with minimal code changes. Supports activation checkpointing, continuous batching, and has been demonstrated at Project Rainier scale (~500,000 Trainium2 chips in 2025; Anthropic states over one million Trainium2 chips fleet-wide as of April 2026).

**NxD Inference (neuronx-distributed-inference)** provides production inference features: continuous batching, speculative decoding, vLLM integration, and TP/PP for large LLM serving on Trn2/Trn3. **Breaking change in Neuron SDK 2.29.0**: NxD Inference no longer supports NKI kernels on Trn1/Inf2 — because NKI 0.3.0 itself does not support Trn1/Inf2 — and the component release note states NxD Inference models are now supported only on **Trn2 and newer**. Trn1/Inf2 users must pin to Neuron SDK 2.28. (The 2.31.0 release-notes *index card* still lists NxD Inference against Inf2/Trn1/Trn1n/Trn2/Trn3; the component release note is the authoritative statement.) **Update (2026-09-13): NxD Inference entered maintenance mode as of Neuron SDK 2.32.0 (2026-08-17)** — "no new feature releases are planned for this component"; AWS directs users to migrate to **vLLM Neuron** (Beta, upgraded to v0.24.0 in 2.32.0), which is now the primary forward-looking inference-serving path.

**Neuron Agentic Development** (first documented as an SDK component in 2.30.0) ships AI coding-agent skills bundled in all Neuron DLAMIs and DLCs: `neuron-framework-autoport` (ports HuggingFace transformer models to NxD Inference end-to-end, including compilation and greedy-token-match accuracy validation) and `neuron-framework-equivalence` (numerical-equivalence validation via progressive 3-tensor R-ratio analysis with fault localization).

### Runtime

**Neuron Runtime (libnrt.so)** is a shared library installed at `/opt/aws/neuron/lib/` and dynamically linked into the ML framework process. There is no separate runtime daemon. Responsibilities:
- NEFF loading, checksum validation, and NeuronCore memory management.
- DMA scheduling for weight streaming (host DRAM → PCIe → HBM at model load; HBM → SBUF during inference).
- Collective communication orchestration via NCCom.
- Metrics and telemetry APIs.
- **Zero-copy host↔device transfers, enabled by default since 2.30.0**; async event APIs; contiguous shared scratchpad support (2.31.0).

**aws-neuronx-dkms** is the kernel mode driver providing PCIe BAR mapping, command queue management, and interrupt routing to NeuronDevices.

### Driver / Firmware

The `aws-neuronx-dkms` kernel module serves as the full driver layer — there is no separate firmware subsystem analogous to NVIDIA's GSP. AWS Nitro integration provides direct device passthrough to the EC2 instance, yielding near-bare-metal PCIe throughput. The Nitro hypervisor handles device enumeration and memory-mapped I/O via the Nitro device model.

### Assembler / ISA

NeuronISA is AWS's proprietary machine instruction set for NeuronCores. It is not publicly documented beyond what is exposed through NKI's `nki.isa` intrinsics. The NKI Compiler (open-sourced Apache 2.0 in Neuron 2.27) compiles NKI kernels via an MLIR pipeline to NeuronISA machine code embedded in NEFF artifacts. There is no intermediate virtual ISA equivalent to PTX.

---

## Hardware Architecture

### Compute Engine

Each NeuronCore contains four independently-pipelined execution units:

- **Tensor Engine** — Systolic array for GEMM, convolution, and transpose. Dimensions have grown per generation: 128×128 BF16 (v2/v3) → 512×128 MXFP8/MXFP4 (v4). Peak throughput: >90 BF16 TFLOPS (v2), 158 cFP8 TFLOPS / 79 BF16 TFLOPS (v3), 315 MXFP8 TFLOPS (v4). NeuronCore-v4 adopts OCP MicroScaling (MXFP8/MXFP4) data formats.

- **Vector Engine** — Parallel vector compute for reductions, activation functions, and layer normalization. ~2.3 TFLOPS FP32 on v2; scaled up on v3/v4.

- **Scalar Engine** — Element-wise scalar operations. ~2.9 TFLOPS FP32 on v2.

- **GPSIMD Engine** — 8× 512-bit wide fully programmable vector processors per NeuronCore. Run general-purpose C code compiled offline. Can access on-chip SRAM directly; serve as a "software escape hatch" for non-standard ops not covered by the other engines.

- **DGE (Dynamic Grouping Engine)** — Introduced on NeuronCore-v3, enhanced on v4. Provides hardware-accelerated dynamic data grouping and token rearrangement for structured sparsity and MoE token routing. Feeds prepared token groups directly into the Tensor Engine's systolic array, eliminating software-managed scatter/gather overhead.

### Data Path

The NeuronCore implements a **software-managed scratchpad architecture** — there is no hardware cache or cache coherency protocol. All data movement between HBM and SBUF/PSUM is explicitly managed by the compiler or the NKI programmer via DMA instructions.

The execution pipeline follows a producer-consumer chain:

```
HBM → [DMA] → SBUF → [Tensor Engine] → PSUM → [Vector Engine] → SBUF → [Scalar Engine] → SBUF → [DMA] → HBM
```

DMA engines operate independently from compute engines, enabling double-buffering: one tile is computed while the next tile is pre-fetched via DMA. The `neuronx-cc` compiler (and NKI programmer) statically schedule this overlap.

### On-chip Memory

| Buffer | v2 Size | v3 Size | v4 Size | Type | Management |
|--------|---------|---------|---------|------|------------|
| SBUF (State Buffer) | 24 MiB | 28 MiB | 32 MiB | SRAM | Software (compiler / NKI) |
| PSUM (Partial Sum Buffer) | 2 MiB | 2 MiB | 2 MiB | SRAM | Hardware (Tensor Engine output) |

SBUF is organized as 128 partitions across all generations. The NKI `nl.ndarray` API maps Python tile index expressions to physical partition addresses. The compiler statically allocates partitions to prevent bank conflicts during concurrent DMA and compute.

PSUM receives the direct output of the Tensor Engine's systolic array. It accumulates partial sums across tiles before the Vector Engine reads it for post-processing (activation, normalization).

### Off-chip Memory

HBM resides at the NeuronDevice level (not per-NeuronCore) and is shared by all NeuronCore instances in a chip via shared DMA engines.

| Generation | HBM Type | Per-Chip Capacity | Per-Chip BW |
|------------|----------|-------------------|-------------|
| NeuronCore-v2 (Trn1) | HBM2e | 32 GiB | 820 GB/s |
| NeuronCore-v3 (Trn2) | HBM2e | 96 GiB | 2.9 TB/s |
| NeuronCore-v4 (Trn3) | HBM3e (4 stacks) | 144 GiB | 4.9 TB/s |

### Host Interface / Package

NeuronDevices connect to the host CPU via PCIe. The `aws-neuronx-dkms` kernel module handles BAR mapping, command queues, and interrupts. AWS Nitro provides direct device passthrough, giving near-bare-metal PCIe throughput. Host DRAM-to-HBM weight transfer at model load is orchestrated by `libnrt.so` via DMA.

Trainium3 (NeuronCore-v4) introduces a **dual-chiplet design** on TSMC 3nm N3P, enabling die area scaling beyond monolithic reticle limits. NeuronLink-v4 provides the inter-chip bandwidth; AWS's own spec table gives **2,048 GiB/s per device**, while the widely-cited "2.5 TB/s bidirectional" figure is third-party (SemiAnalysis) — see the bandwidth reconciliation note in the 2026-08-08 update section.

### Scale-up Interconnect (NeuronLink)

| Generation | Chip | BW | Topology | Scale-up Domain |
|------------|------|----|----------|-----------------|
| NeuronLink-v2 | Trainium1 | 768 GB/s bidir | 2D Torus | 16 chips (trn1.32xlarge) |
| NeuronLink-v3 | Trainium2 | 1 TB/s bidir | 4×4 2D Torus (node) / 3D Torus (UltraServer) | 64 chips (Trn2 UltraServer, ~6,100 copper cables) |
| NeuronLink-v4 + NeuronSwitch-v1 | Trainium3 | 2,048 GiB/s per device (AWS spec table); 2.5 TB/s bidir (SemiAnalysis) — figures unreconciled | All-to-all **PCIe Gen6 switch** fabric | 64 chips (UltraServer Gen1) or 144 chips (UltraServer Gen2) |

The shift from 2D/3D Torus to an all-to-all switched fabric (NeuronSwitch-v1) eliminates the hop-count asymmetry between near and far neighbors, reducing worst-case all-to-all collective latency from O(√N) to O(1) hops — critical for MoE expert dispatch. AWS documentation added on 2026-04-09 discloses that this fabric is built from **PCIe Gen6 switches**, explicitly replacing the point-to-point NeuronLink topology of Trn1/Trn2.

### Scale-out Interconnect (EFA + UltraClusters)

| Version | Instance | Bandwidth |
|---------|----------|-----------|
| EFAv2 | Trn1 | up to 1,600 Gbps per instance |
| EFAv3 | Trn2 | 3.2 Tbps per instance; 12.8 Tbps per Trn2 UltraServer |
| EFAv3 | Trn3 | 12,800 Gbps per Trn3 UltraServer Gen1 (64 devices); 28,800 Gbps per Trn3 UltraServer Gen2 (144 devices) — i.e. a consistent **200 Gbps of EFA per Trainium3 device** in both configurations |

Note: AWS reports Trn3 EFA bandwidth at **UltraServer granularity**, not per instance; the repo's earlier "exact GA specs pending" was a research gap, not a missing disclosure — these numbers have been in the AWS Trn3 architecture page since 2025-12-02.

Multiple UltraServers are interconnected via petabit-scale EFA networking in **EC2 UltraClusters**, enabling scale-out to hundreds of thousands of chips. Project Rainier reached ~500,000 Trainium2 chips for Anthropic training workloads in 2025; as of 2026-04-20 Anthropic states it uses **over one million Trainium2 chips** across AWS. NCCom routes inter-node collective traffic over EFA using the SRD (Scalable Reliable Datagram) protocol without CPU involvement.

---

## Programming Model Rationale

The AWS Neuron programming model is a direct expression of its hardware constraints and design philosophy.

### Tile Programming from Software-Managed SRAM (SBUF/PSUM)

The NeuronCore's elimination of hardware cache forces a tile-based programming model. Because there is no automatic data reuse mechanism, the programmer (or compiler) must explicitly manage which tiles reside in SBUF, when to evict them, and how to overlap DMA prefetch with compute. NKI's tile API (`nl.ndarray`, `nl.load`, `nl.store`) is the natural consequence: it surfaces SBUF partitions as first-class Python objects and makes DMA a programmer-controlled operation. This design trades hardware complexity for compiler predictability — memory access latency is deterministic, and the compiler can produce tight, latency-hiding instruction schedules. The cost is that custom kernels require explicit tile management that GPU programmers do not need to perform.

### XLA Graph Compilation and Static Scheduling

The Tensor Engine's systolic array requires static data flow to achieve peak utilization — dynamic control flow (data-dependent branching, irregular sparsity) cannot be efficiently scheduled at compilation time. The XLA frontend, which operates on static computation graphs (HLO), is a natural fit: XLA's operator fusion, constant propagation, and re-materialization passes all assume a static graph that can be globally analyzed. This is why the original PyTorch integration was built on PyTorch/XLA rather than a dynamic dispatching model. The transition to TorchNeuron (PrivateUse1) introduces eager mode, but `torch.compile` with TorchDynamo remains the recommended training path — it still captures static subgraphs for compilation.

### NeuronLink Topology Drives Parallelism Strategy

The NeuronLink interconnect topology directly determines which parallelism strategies are hardware-efficient:

- **Trn1/Trn2 (2D/3D Torus)**: Ring-AllReduce traverses torus dimensions efficiently, making Tensor Parallelism (TP) the preferred intra-node strategy. All-to-all patterns (required for MoE expert dispatch) require O(√N) hops — inefficient on a torus. NxD Core accordingly maps TP ranks to NeuronLink-connected chips and uses EFA for data-parallel gradient AllReduce across nodes.

- **Trn3 (NeuronSwitch-v1 all-to-all fabric)**: O(1)-hop any-to-any connectivity makes MoE expert parallelism (all-to-all dispatch) hardware-efficient for the first time in the Neuron lineage. The DGE hardware block (v3/v4) complements this: it handles token grouping for MoE routing in hardware, removing a software scatter/gather bottleneck that would otherwise defeat the switched fabric's latency advantage.

### EFA + UltraClusters Enable Massive Scale-Out Without Proprietary Interconnect

Unlike NVIDIA's NVLink-C2C / NVSwitch ecosystem (which requires NVIDIA-specific networking end-to-end), Neuron's inter-node communication uses EFA — a standard high-performance network fabric available to all EC2 instance types. This means the scale-out interconnect is cloud infrastructure, not chip vendor IP. AWS can scale EFA bandwidth independently of chip generations (EFAv2 → EFAv3 across Trn1 → Trn2/Trn3 generations) and deploy UltraClusters at planetary scale without deploying a proprietary network switch. The trade-off is that EFA bandwidth per chip is lower than NVSwitch-based GPU clusters at equivalent chip counts — but AWS compensates with high NeuronLink intra-node bandwidth and petabit-scale EFA fabric for inter-node gradients.

**Qualification (2026-08-08): the scale-*up* fabric is also becoming standards-based.** With Trainium3, the "proprietary interconnect" framing weakens on the scale-up side as well. AWS documentation added 2026-04-09 states that Trn3 chip-to-chip communication runs over **PCIe Gen6 switches** — a commodity switching substrate — with NeuronLink-v4 carried on PCIe Gen6 ×8 link groups and routing performed by encoding a (rack, server, chip) tuple in the upper bits of the outbound PCIe address (BAR address matching selects the output port). Under this reading, AWS has now pushed *both* tiers of its interconnect onto industry-standard transports (PCIe Gen6 scale-up, EFA/Ethernet-class scale-out), and the differentiation lives in topology, addressing, and the hardware-semaphore ordering model rather than in a bespoke SerDes link layer. This is the sharpest available contrast with NVLink/NVSwitch, which remains a custom link layer end-to-end.

### PrivateUse1 Backend: Shift Toward Standard PyTorch Integration

The transition from torch-neuronx (XLA-based) to TorchNeuron (PrivateUse1) reflects a broader vendor trend: minimizing proprietary wrapper surface area to reduce porting friction. XLA's lazy evaluation model (requiring `xm.mark_step()` calls, XLA-specific distributed APIs) is a significant behavioral difference from standard PyTorch. TorchNeuron eliminates this: standard `torch.distributed`, FSDP, DTensor, and `torch.nn.parallel` APIs all work without modification on Trainium/Inferentia once the PrivateUse1 backend is registered. The NKI Compiler's open-sourcing (Apache 2.0, Neuron 2.27) extends this philosophy to the kernel layer, enabling community development of custom Neuron kernels without AWS-controlled compiler toolchain access. NKI's promotion from Beta to **Stable** in Neuron 2.29.0 (2026-04-09) completes that arc: the kernel layer is now a supported, versioned public API rather than a preview surface.

---

## Neuron SDK 2.29–2.31 Update and Trn3 Fabric Disclosure (2026-08-08)

*Updated 2026-08-08. Window covered: 2026-04-05 → 2026-08-08. Primary sources: aws-neuron/aws-neuron-sdk GitHub release tags and `release-notes/components/*.rst`; the AWS Trn3 architecture page and its git history; Anthropic newsroom 2026-04-20. Release dates below are GitHub `published_at` values.*

### Release timeline

The repo's previous "Neuron SDK 2.27" narrative was already stale at the 2026-04-05 baseline: **2.28.0 (2026-02-25)** and **2.28.1 (2026-03-13)** shipped before it. Four releases landed inside the window:

| Release | Date | NKI version | Headline |
|---|---|---|---|
| 2.29.0 | 2026-04-09 | NKI 0.3.0 | **NKI Beta → Stable**; NKI Standard Library; CPU Simulator; Neuron Explorer Beta → Stable; breaking Trn1/Inf2 drop in NxD Inference |
| 2.29.1 | 2026-05-01 | — | Patch (GitHub tag date; the readthedocs release index shows 04/09/26 — GitHub tag metadata is the stronger source) |
| 2.30.0 | 2026-05-21 | NKI 0.4.0 | Trn3-specific ISA (`activate2`, OCP FP8 matmul inputs); tile size doubled to 1024; zero-copy transfers default; Neuron DRA driver; **Neuron Agentic Development** first appears as an SDK component |
| 2.31.0 | 2026-07-07 | NKI 0.5.0 | Tensor indirection (gather/scatter); redesigned `neuronx-cc` codegen backend default on Trn2/Trn3; UltraServer Operator public beta |

### 2.29.0 (2026-04-09) — NKI reaches Stable

The single most consequential software-stack event in the window. NKI 0.3.0 graduates from **Beta to Stable**, converting the kernel layer from a preview API into a supported one.

- **NKI Standard Library (`nki-stdlib`)** — developer-visible source for all NKI APIs plus native NKI language objects (`NkiTensor`).
- **CPU Simulator (experimental)** — executes NKI kernels entirely on the host CPU, with no Trainium hardware required; enabled by environment variable or API. This is the first local-development path for Neuron kernels that does not require access to an accelerator instance.
- `nki.collectives.all_to_all_v` (variable-length all-to-all) added.
- 7 new experimental NKI-Lib kernels (Conv1D, Transformer TKG, communication-compute fusion).
- **Neuron Explorer** (profiling/debugging suite) also moves Beta → Stable, now published on the VS Code Extension Marketplace.
- **Breaking:** NKI 0.3.0 does not support Trn1/Inf2, so NxD Inference no longer supports NKI kernels on those devices, and NxD Inference models are stated to be supported only on **Trn2 and newer**. Trn1/Inf2 users must pin to 2.28.
- API churn: `SbufManager` renamed `BufferManager`; MoE TKG boolean flags replaced by an `LNCShardingStrategy` enum.

### 2.30.0 (2026-05-21) — Trn3 hardware surfaces in NKI

- **NKI 0.4.0 Trn3 ISA exposure:** `nki.isa.activate2` (fused preprocess + activate on the Scalar Engine, Trn3-only), OCP FP8 inputs to matrix-multiply, Vector-Engine absolute-value reductions, and bytes-aware tile-size properties (`sbuf_size_bytes`, `sbuf_fmax_bytes`).
- **Graph compiler:** rewritten memory-liveness analysis (compile-time reduction); **maximum computation tile size doubled to 1024**; coalesced reduce-scatter optimization.
- **Runtime:** zero-copy host↔device transfers **on by default**; async event APIs; Collectives support for the **Trn3 Gen2 UltraServer ring topology**.
- **NKI Library:** +3 core kernels (segmented attention, KV-parallel prefill, FP8 quantization) and +19 experimental (context parallelism, MXFP8 training, state-space models, fused optimizers, MoE dispatch); PyTorch reference implementations shipped for 29 kernels.
- **Neuron DRA Driver** — Kubernetes Dynamic Resource Allocation with topology-aware scheduling of Trainium devices and EFA interfaces.
- DLAMIs rebased on Ubuntu 24.04; JAX-NeuronX 0.10.0.

### Neuron Agentic Development — a stack layer the survey did not previously document

`neuron-agentic-development` first appears as a first-class, separately release-noted Neuron SDK component in **2.30.0**, bundled in all DLAMIs and DLCs by default. It ships two AI coding-agent skills:

- **`neuron-framework-autoport`** — ports HuggingFace transformer models to NxD Inference end-to-end, including compilation and greedy-token-match accuracy validation.
- **`neuron-framework-equivalence`** — numerical-equivalence validation of a ported model via progressive 3-tensor R-ratio analysis with fault localization.

2.31.0 only refreshes these skills for NKI 0.5.0 compatibility. *Scope note:* the component release-notes file begins at 2.30.0, so "first documented SDK component in 2.30.0" is the defensible claim; whether informal NKI-authoring or debugging agent skills circulated before 2.30 is **not disclosed**. Source: github.com/aws-neuron/neuron-agentic-development.

### 2.31.0 (2026-07-07, current latest as of 2026-08-08)

- **NKI 0.5.0:** **tensor indirection (gather/scatter) via `.indirect()`** — a new on-chip addressing mode for compute operations; OCP MX scale format (`float8_e8m0fnu`); composable `NkiTensor` view methods (slice / select / permute / rearrange) for zero-cost layout transforms; 8192-element bf16 destination output for `nc_matmul`.
- **`neuronx-cc` redesigned code-generation backend, now default for Trn2 and Trn3** — improved instruction scheduling and memory prefetch.
- **Runtime:** contiguous shared scratchpad support.
- **UltraServer Operator** for Amazon EKS enters public beta.
- NKI Library: **14 new experimental kernels** (deformable attention, MoE training collectives, indexed gather/scatter, DeepSeek MLA projection, ring attention) — experimental, not GA library kernels.
- Deprecations: the SPMD launch-grid requirement for LNC2 NKI kernels is deprecated and will be removed in a future release (behavior unchanged in 0.5.0); `neuronxcc.nki.*` namespace use is now a hard compile error; `nki.jit(platform_target=...)` and `nki.jit(mode=...)` deprecated.
- Supported instances listed: Inf1, Inf2, Trn1, Trn1n, Trn2, Trn3.

### Trn3 UltraServer specifications (correcting a repo research gap, not a window change)

The complete Gen1/Gen2 specification table has been on the AWS Trn3 architecture page since the **2025-12-02 re:Invent commit** — four months before the repo's 2026-04-05 baseline. The repo's earlier "EFAv3 | Trn3 | higher per-instance BW (exact GA specs pending)" was therefore a gap in our own research, not a pending disclosure.

| Metric | Trn3 UltraServer Gen1 | Trn3 UltraServer Gen2 |
|---|---|---|
| Trainium3 devices in scale-up domain | 64 (4 servers) | 144 (36 servers) |
| Switching | NeuronLink-v4 + NeuronSwitch-v1 | First-level NeuronSwitch-v1 within server + two second-level NeuronSwitch-v1 across servers |
| HBM3e capacity | 9,216 GiB | 20,736 GiB |
| HBM bandwidth | 313.6 TB/s | 705.6 TB/s |
| MXFP8 / MXFP4 | 161,088 TFLOPS | 362,448 TFLOPS |
| FP16 / BF16 / TF32 | 42,944 TFLOPS | 96,624 TFLOPS |
| FP32 | 11,712 TFLOPS | 26,352 TFLOPS |
| EFA | 12,800 Gbps | 28,800 Gbps |
| Host | 768 vCPUs, 8,192 GiB host memory | 2,304 vCPUs, 27,648 GiB host memory |

Per chip: 144 GiB HBM3e at 4.9 TB/s, 2,517 TFLOPS MXFP8/MXFP4, NeuronLink-v4 at 2,048 GiB/s per device. AWS states the all-to-all connectivity is optimized for MoE and autoregressive inference serving.

**Trn3 instance availability remains hedged.** The EC2 Trn3 product page publishes only UltraServer aggregates and states neither GA nor preview, and lists no instance sizes or regions — availability status is **not disclosed**.

### New in the window: the Trn3 scale-up fabric is PCIe Gen6

The 2026-04-09 commit to the Trn3 architecture page (shipped alongside 2.29.0; still the live page as of 2026-08-08) added a **"Trn3 UltraServer Connectivity and Networking"** section disclosing that Trn3 uses a **PCIe switch-based interconnect for all chip-to-chip communication**, explicitly "replacing the point-to-point NeuronLink topology used in previous generations (Trn1, Trn2)":

| Scope | Links per chip | Aggregate bidirectional |
|---|---|---|
| Intra-server (4-chip sled → intra-server switch) | 4 × PCIe Gen6 ×8 | 256 GB/s |
| Inter-server, within rack (→ inter-server switches) | 5 × PCIe Gen6 ×8 | 320 GB/s |
| Inter-rack (direct links) | 2 × PCIe Gen6 ×8 | 128 GB/s |

- **Address-based routing:** each chip is identified by a (rack, server, chip) tuple encoded in the upper bits of the outbound PCIe address; switches use BAR address matching to select the output port. This is transparent to workloads — the runtime and compiler configure it.
- **Hardware semaphores** provide cross-fabric synchronization: data and its completion semaphore are guaranteed to traverse the same physical path, giving ordering without software fences.

This means NeuronSwitch-v1 is a **PCIe Gen6 switch fabric** and NeuronLink-v4 is carried over PCIe Gen6 ×8 lane groups — materially qualifying the framing of NeuronLink as bespoke vendor interconnect IP (see "Programming Model Rationale" above).

> ⚠️ **Unreconciled bandwidth figures — AWS's own page is internally inconsistent.** The per-chip link budget in the new connectivity section sums to 256 + 320 + 128 = **704 GB/s**, which cannot be reconciled with the same page's spec-table figure of **2,048 GiB/s per device** for NeuronLink-v4. The repo's previously carried **2.5 TB/s bidirectional** matches neither and appears to originate with SemiAnalysis. This survey cites 2,048 GiB/s/device as the AWS vendor spec-table figure and records the 704 GB/s link-level sum and the 2.5 TB/s third-party figure as unreconciled. No reconciliation is offered by any primary source.

### Deployment scale: Anthropic–Amazon expansion (2026-04-20)

Anthropic's newsroom post of 2026-04-20 states:

- Up to **5 GW** of new compute capacity.
- Anthropic **currently uses over one million Trainium2 chips** to train and serve Claude — superseding the ~500,000-chip Project Rainier figure previously carried throughout this survey.
- Significant new Trn2 capacity online in **Q2 2026**, with scaled **Trainium3** later in 2026; **nearly 1 GW** of combined Trn2 + Trn3 by end of 2026.
- Amazon investing **$5B immediately plus up to $20B milestone-based**, on top of **$8B** previously; Anthropic committing **$100B+** to AWS over ten years.

These are the parties' own statements, not independently audited deployment counts.

### Roadmap and negative results

- **Trainium4 / NeuronCore-v5:** roadmap-unchanged. The re:Invent (Dec 2025) preview claims — ≥6× FP4, 3× FP8, 4× memory bandwidth and 2× capacity vs Trainium3 via 8 stacks of HBM4, plus NVIDIA NVLink 6 / NVLink Fusion participation in a shared scale-up fabric, availability late 2026–2027 — remain **vendor preview claims** with no new primary-source confirmation found in this window.
- **Hot Chips 38 (Aug 23–25, 2026):** AWS / Amazon / Annapurna is **absent** from the program. No Trainium or Neuron talk is scheduled.
- **MLPerf Training v6.0 (published 2026-06-16, 24 submitters):** no AWS/Amazon/Trainium submission. No Trainium MLPerf result is recorded in this survey.
- **TorchNeuron (PrivateUse1):** no status change found. Neither `torch-neuronx` nor TorchNeuron appears as an updated component in the 2.31.0 release-notes index, but that index explicitly notes that some components may not be updated in a given release — so this is evidence of neither GA nor deprecation. Status remains "private beta as of Neuron SDK 2.27, unverified since".

---

## Update (2026-09-13)

*Scan window 2026-08-08 → 2026-09-13. Classification: **Moderate** (SDK/compiler/runtime version update; no new NeuronCore generation or spec disclosure). Sources: aws-neuron/aws-neuron-sdk GitHub releases; readthedocs release-notes pages; component `.rst` changelogs. WebSearch was unavailable this session (budget exhausted); findings are from direct primary-source URL fetches only.*

Two releases shipped in the window: **2.31.1** (2026-08-12, patch/bugfix) and **2.32.0** (2026-08-17, NKI 0.6.0).

- **NKI 0.6.0**: on-device top-K reduction (`nisa.topk`); variable-length collective `all_gather_v`; new runtime loop constructs `fori_loop`/`while_loop` (replacing `nl.dynamic_range`); relaxed DMA transpose constraints. +13 new NKI Library kernels (DeepSeek-V3.2 sparse-MLA context encoding, MXFP8 MoE training); PyTorch reference impls extended to 22 more kernels.
- **Graph compiler (neuronx-cc v2.27.5334.0)**: `--native-int64` / `--implicit-integer-downcast` flags; complex64 op support expanded to 30 ops; embedding-lookup-as-gather optimization — AWS claims up to 64% faster compiles and up to 96% smaller NEFFs for affected workloads **(vendor claim)**.
- **Runtime/driver**: variable-size collectives (AllGatherV, ReduceScatterV, AllToAllV) for uneven per-rank data on Trn2/Trn3; one-rank-per-die topology on Trn3 Gen2 UltraServer; max NCCL communicators per NEFF raised 12 → 16.
- **NxD Inference → maintenance mode**: no further feature releases planned; AWS directs migration to vLLM Neuron (upgraded to v0.24.0). This is the culmination of the 2.29.0 Trn1/Inf2 drop and Trn2-and-newer scoping recorded in the prior update — vLLM Neuron is now the primary forward path for inference serving.
- **Neuron Agentic Development**: new `neuron-framework-autoport-vllm-neuron` skill (HuggingFace → vLLM Neuron porting).

**Checked, no change found**: the Trn3 architecture page (UltraServer spec table, the unreconciled NeuronLink-v4 bandwidth figures, and the absence of Trn3 GA/preview status) is byte-for-byte consistent with the 2026-04-09 revision already recorded — no new hardware disclosure this window. No new Trainium4 primary-source material found. Hot Chips 38 (Aug 23–25, 2026) is within the window but was already recorded as having no AWS/Neuron talk; not independently re-verified this cycle.

Sources: https://github.com/aws-neuron/aws-neuron-sdk/releases/tag/v2.32.0 · https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nki.rst · https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nxd-inference.rst

---

## Resources

| Resource | URL |
|----------|-----|
| About NKI | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/about/index.html |
| Neuron Graph Compiler | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html |
| Native PyTorch for AWS Trainium | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html |
| NeuronX Runtime Documentation | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/index.html |
| NeuronCore-v2 Architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html |
| NeuronCore-v3 Architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html |
| NeuronCore-v4 Architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v4.html |
| Trainium2 Architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html |
| Trainium3 Architecture | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html |
| Trn2 Architecture (NeuronLink-v3) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html |
| Trn3 Architecture (NeuronSwitch-v1) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html |
| NeuronX Distributed (NxD Core) | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html |
| Neuron Collective Communication | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html |
| aws-neuron/nki-samples (GitHub) | https://github.com/aws-neuron/nki-samples |
| aws-neuron/neuronx-distributed (GitHub) | https://github.com/aws-neuron/neuronx-distributed |
| Announcing AWS Neuron SDK 2.27.0 | https://aws.amazon.com/about-aws/whats-new/2025/12/announcing-aws-neuron-2-27/ |
| Neuron SDK release notes index | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html |
| Neuron SDK 2.31.0 release notes | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.31.0.html |
| aws-neuron-sdk GitHub releases (authoritative release dates) | https://github.com/aws-neuron/aws-neuron-sdk/releases |
| NKI component release notes (0.3.0 / 0.4.0 / 0.5.0) | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nki.rst |
| NxD Inference component release notes (Trn1/Inf2 drop wording) | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nxd-inference.rst |
| Neuron Agentic Development component release notes | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/agentic-development.rst |
| aws-neuron/neuron-agentic-development (GitHub) | https://github.com/aws-neuron/neuron-agentic-development |
| Announcing AWS Neuron 2.29 | https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/ |
| Announcing AWS Neuron 2.30.0 | https://aws.amazon.com/about-aws/whats-new/2026/05/aws-announce-neuron-2-30-0/ |
| Trn3 architecture page — source (PCIe Gen6 connectivity section) | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/about-neuron/arch/neuron-hardware/trn3-arch.rst |
| Amazon EC2 Trn3 instances (product page) | https://aws.amazon.com/ec2/instance-types/trn3/ |
| Anthropic — Amazon compute expansion (2026-04-20) | https://www.anthropic.com/news/anthropic-amazon-compute |
| SemiAnalysis — Trainium3 Deep Dive | https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential |
| SemiAnalysis — Trainium2 Architecture | https://newsletter.semianalysis.com/p/amazons-ai-self-sufficiency-trainium2-architecture-networking |
| Neuron SDK 2.32.0 release notes | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.32.0.html |
| Neuron SDK v2.32.0 GitHub release | https://github.com/aws-neuron/aws-neuron-sdk/releases/tag/v2.32.0 |
| NKI component release notes (0.6.0 added 2026-09-13) | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nki.rst |
| NxD Inference component release notes (maintenance-mode notice) | https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nxd-inference.rst |
