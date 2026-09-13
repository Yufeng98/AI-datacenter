# AWS Neuron Communication & Distributed Training — Investigation Report

**Chip:** AWS Neuron (Trainium/Inferentia)
**Investigation Focus:** NCCom collectives, NeuronLink, EFA, NxD parallelism strategies
**Date:** 2026-04-05

---

## Overview

AWS Neuron's communication and distributed training stack is built around a two-tier interconnect architecture: NeuronLink for intra-node chip-to-chip communication and EFA (Elastic Fabric Adapter) for inter-node communication. The software layer that orchestrates collective operations across both tiers is NCCom (Neuron Collective Communication), a library embedded in the Neuron Runtime that dispatches collectives without CPU involvement. Above NCCom, the NxD (NeuronX Distributed) libraries provide framework-level distributed training APIs: NxD Core for low-level tensor/pipeline parallelism primitives and NxD Training for high-level 3D parallelism workflows compatible with NeMo Megatron.

The NeuronLink topology has evolved across chip generations in ways that directly constrain which parallelism strategies are efficient: the ring-friendly 2D/3D Torus of Trn1/Trn2 favors AllReduce-heavy tensor parallelism, while the NeuronSwitch-v1 all-to-all fabric of Trn3 unlocks efficient expert parallelism (MoE all-to-all) within a UltraServer domain.

---

## Architecture

### NCCom (Neuron Collective Communication)

NCCom is the low-level collective communication runtime built into Neuron Runtime (libnrt.so). Its key architectural properties:

- **CPU-bypass execution:** Collective operations are dispatched from NeuronCore programs directly to NeuronLink or EFA without routing through the host CPU. This eliminates PCIe round-trips for synchronization.
- **Supported collectives:** AllReduce, AllGather, ReduceScatter, AllToAll, AllToAllV (variable-length), Broadcast, ReduceScatter.
- **Transport selection:** NCCom automatically routes operations to NeuronLink (intra-node, same UltraServer) or EFA (inter-node, across nodes) based on process group topology.
- **nki.collectives integration:** As of Neuron 2.27, NKI kernels can invoke NCCom operations directly within a kernel function (`nki.collectives.all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `collective_permute`, `rank_id`), enabling custom attention and MoE kernels to perform cross-chip reductions without returning to the framework layer.
- **nccom-test tool:** A standalone benchmarking utility for measuring collective performance; supports AllToAllV for variable-sized benchmarks.

### NeuronLink (Intra-Node Scale-Up)

NeuronLink is the proprietary chip-to-chip interconnect that connects NeuronCores within a server node or UltraServer domain. It carries NCCom collective traffic without traversing the host CPU or PCIe bus.

**Generation evolution and topology:**

| Generation | Chip | BW | Topology | Max Chips (scale-up domain) |
|---|---|---|---|---|
| NeuronLink-v2 | Trainium1 | 768 GB/s | 2D Torus | 16 (trn1.32xlarge) |
| NeuronLink-v3 | Trainium2 | 1 TB/s | 4×4 2D Torus (node) / 3D Torus (UltraServer) | 64 (Trn2 UltraServer) |
| NeuronLink-v4 + NeuronSwitch-v1 | Trainium3 | 2.5 TB/s | All-to-all switched | 144 (Trn3 NL72×2 UltraServer) |

**Topology impact on collective performance:**

- **2D/3D Torus (Trn1/Trn2):** AllReduce is most efficient as a ring reduction traversing the torus dimension. Worst-case hop count is `O(√N)` for 2D Torus or `O(∛N)` for 3D Torus. All-to-all patterns (needed for MoE expert dispatch) require up to `O(N)` hops through intermediate nodes — inefficient.
- **All-to-all switched fabric (Trn3/NeuronSwitch-v1):** Every chip is directly connected to every other chip within the domain via the switch fabric. AllReduce, AllToAll, and expert dispatch collectives all operate at O(1) hops. This halves worst-case all-to-all latency and removes the bandwidth asymmetry between near/far neighbors in a torus.

**Physical implementation:** NeuronLink is implemented via copper cables on backplanes internal to the UltraServer chassis. Trn2 UltraServer (64 chips, 3D Torus): ~6,100 copper cables across 4 NeuronLink backplanes. Trn3 NL72×2 UltraServer (144 chips, switched): ~5,100 copper cables (denser switch fabric is more cable-efficient than torus at high chip counts).

**Inferentia2 NeuronLink:** Inf2 instances connect up to 12 Inferentia2 chips at 192 GB/s via NeuronLink for multi-chip model sharding in inference.

### EFA (Elastic Fabric Adapter) for Inter-Node

EFA is AWS's custom network fabric for high-performance inter-instance MPI/collective traffic.

| Version | Instance | Bandwidth | Usage |
|---|---|---|---|
| EFAv2 | Trn1 | up to 1,600 Gbps per instance | Inter-node AllReduce (data parallelism gradient sync) |
| EFAv3 | Trn2 | 3.2 Tbps per instance; 12.8 Tbps per UltraServer | Inter-node AllReduce, AllGather (pipeline/data parallelism) |
| EFAv3 | Trn3 | higher per-instance bandwidth (exact GA specs pending) | All inter-node collective traffic |

NCCom selects EFA for any collective between ranks that span different physical nodes (i.e., not reachable via NeuronLink). EFA is accessed through the standard RDMA/SRD (Scalable Reliable Datagram) protocol, with NCCom providing the Neuron-specific transport plugin.

**UltraClusters:** Multiple UltraServers in an EC2 UltraCluster are connected via petabit-scale EFA networking. This enables scale-out to tens of thousands of chips — AWS Project Rainier deployed ~500,000 Trainium2 chips for Anthropic, with plans to scale beyond 1 million chips.

---

## NxD Core (neuronx-distributed)

NxD Core is an XLA-based PyTorch library providing distributed training and inference primitives for Neuron devices.

**Supported parallelism strategies:**
- **Tensor Parallelism (TP):** Partitions weight matrices across NeuronCores; linear layers split column-wise or row-wise with corresponding AllGather/ReduceScatter at layer boundaries. `ColumnParallelLinear` and `RowParallelLinear` are the key APIs.
- **Pipeline Parallelism (PP):** Splits the model depth-wise across NeuronCore groups; supports 1F1B (one-forward-one-backward) schedule and interleaved pipeline schedules to reduce pipeline bubble fraction.
- **Sequence Parallelism (SP):** Partitions the sequence dimension of activations to reduce peak activation memory. Enabled by `sequence_parallel_enabled=True` on parallel linear layers; inserts AllGather before and ReduceScatter after sequence-split layers.
- **Data Parallelism (DP):** Standard gradient accumulation and AllReduce; integrates with ZeRO-1 for optimizer state sharding.
- **ZeRO-1:** Shards optimizer states (not gradients or parameters) across data-parallel ranks to reduce per-device memory.

**3D Parallelism:** NxD Core supports full 3D parallelism (TP × PP × DP) with a process group hierarchy that maps TP ranks to NeuronLink-connected chips within a node, and PP/DP ranks across nodes via EFA.

**Key APIs:**
- `neuronx_distributed.parallel_layers.ColumnParallelLinear`, `RowParallelLinear`
- `neuronx_distributed.pipeline.NxDPPModel` for pipeline stages
- `neuronx_distributed.parallel_state` for process group management
- Distributed optimizer wrappers for ZeRO-1

---

## NxD Training (neuronx-distributed-training)

NxD Training is the high-level distributed training library built on NxD Core, targeting LLM pretraining and fine-tuning workflows.

**Features:**
- **NeMo compatibility:** Direct integration with NVIDIA NeMo framework recipes (via `neuronx-nemo-megatron`), enabling GPT/LLaMA pretraining with Neuron-specific optimizations via familiar NeMo YAML configs.
- **3D parallelism out of the box:** TP + PP + DP configured via training config with no manual process group management.
- **Activation checkpointing:** Selective and full recomputation strategies to trade compute for memory.
- **Sequence parallelism:** Built-in SP support for long-context training.
- **Continuous batching and gradient accumulation.**
- **Scale demonstrated:** LLaMA-2 13B/70B, LLaMA-3 70B training on Trn1/Trn2 documented; Project Rainier runs at 500,000 Trainium2 chip scale.

**Standard API compatibility:** With TorchNeuron (PrivateUse1, PyTorch ≥ 2.10), standard `torch.distributed`, FSDP, DDP, and DTensor APIs work without Neuron-specific wrappers — lowering the porting cost for existing PyTorch distributed code.

---

## Data Flow: Distributed Training Collective Path

```
Transformer Layer (forward pass)
  │
  ├── [Tensor Parallel] ColumnParallel GEMM
  │     └── Input: AllGather over TP ranks via NeuronLink
  │     └── Output: local partial result
  │
  ├── [Sequence Parallel] Activation
  │     └── ReduceScatter over TP/SP ranks via NeuronLink
  │
  ├── [Pipeline Parallel] Stage boundary
  │     └── Send activation to next stage via NeuronLink (intra-node)
  │         or EFA (inter-node stage boundary)
  │
  └── [Data Parallel] Gradient sync after backward
        └── AllReduce over DP ranks via EFA (inter-node)
            or NeuronLink (if DP ranks co-located)

NCCom dispatch path:
  1. NeuronCore program issues collective instruction
  2. NCCom runtime resolves transport (NeuronLink vs. EFA)
  3. DMA engines move data without CPU involvement
  4. Completion signal returned to NeuronCore
```

---

## Hardware Interface

| Communication Type | Hardware | Software API |
|---|---|---|
| Intra-node AllReduce (TP gradient) | NeuronLink | NCCom / `nki.collectives.all_reduce` |
| Intra-node AllGather (TP activation) | NeuronLink | NCCom / `nki.collectives.all_gather` |
| Intra-node AllToAll (MoE dispatch) | NeuronLink / NeuronSwitch-v1 | NCCom / `nki.collectives.all_to_all` |
| Inter-node AllReduce (DP gradients) | EFAv2/v3 | NCCom (EFA transport) |
| Inter-node AllGather (PP activation) | EFAv3 | NCCom (EFA transport) |
| Pipeline stage activation (inter-node) | EFAv3 | NxD Core PP send/recv |
| Intra-chip compute pipeline | NeuronLink-internal / SBUF | Not directly user-facing |

---

## Key Findings

1. **Two-tier transport with automatic selection:** NCCom transparently selects NeuronLink (fast, low-latency) for intra-node collectives and EFA (high-bandwidth, lower latency than standard TCP) for inter-node, allowing the same distributed training code to run efficiently at any scale without topology-aware changes.

2. **NeuronLink topology constrains parallelism strategy:** The 2D/3D Torus of Trn1/Trn2 makes TP ring-AllReduce efficient (bounded by ring bandwidth) but penalizes all-to-all (MoE expert routing). The NeuronSwitch-v1 all-to-all switched fabric on Trn3 removes this constraint, making MoE expert parallelism first-class within a UltraServer domain.

3. **nki.collectives enables kernel-level collective fusion:** By allowing NKI kernels to issue collectives without returning to the framework, AWS enables fused kernel patterns where compute and communication overlap at the instruction level — a capability analogous to NCCL in-kernel collectives on GPUs but with explicit SBUF tile management.

4. **NxD 3D parallelism maps TP to NeuronLink, DP to EFA:** The process group hierarchy in NxD Core is designed so TP ranks occupy NeuronLink-connected chips within a single node (taking advantage of 1 TB/s NeuronLink bandwidth) while DP and PP ranks span nodes via EFA. This maximizes use of the fastest interconnect for the most communication-intensive operations (TP AllReduce).

5. **Project Rainier demonstrates petascale EFA viability:** AWS's deployment of ~500,000 Trainium2 chips for Anthropic's training workloads — using petabit-scale UltraCluster EFA networking — validates that EFA can sustain collective communication at unprecedented chip counts. This architecture uses data parallelism across UltraServer boundaries and relies on EFA for gradient synchronization at global scale.

6. **Standard distributed API transition (TorchNeuron):** The move to PrivateUse1 native backend means `torch.distributed`, FSDP, DDP, and DTensor will work on Neuron devices without any Neuron-specific wrappers, significantly lowering the barrier to porting distributed training code from GPU clusters.

---

## Relation to Hardware

| Software Component | Hardware Dependency |
|---|---|
| NCCom AllReduce (intra-node) | NeuronLink-v2/v3/v4 chip-to-chip links |
| NCCom AllToAll (MoE dispatch) | NeuronSwitch-v1 all-to-all fabric (Trn3) |
| NCCom AllReduce (inter-node) | EFAv2 (Trn1) / EFAv3 (Trn2/Trn3) |
| NxD TP ColumnParallelLinear AllGather | NeuronLink (intra-node TP group) |
| NxD PP stage boundary send/recv | EFAv3 (inter-node stages) or NeuronLink |
| nki.collectives.all_to_all | NeuronLink / NeuronSwitch-v1 |
| DGE token routing for MoE | NeuronCore-v3/v4 DGE hardware block |
| AWS ParallelCluster scheduler | EFAv3 / MPI transport layer |

---

## Update — 2026-08-08

*Window 2026-04-05 → 2026-08-08. See `hw-architecture.md` (2026-08-08 update) for the fabric evidence and `neuron-sdk-nki.md` (2026-08-08 update) for the SDK evidence; this section records only the communication-layer consequences.*

### Trn3 scale-up fabric is PCIe Gen6 switched (documented 2026-04-09)

AWS documentation added on 2026-04-09 (with Neuron SDK 2.29.0) discloses that Trn3 chip-to-chip communication runs over **PCIe Gen6 switches**, explicitly replacing the point-to-point NeuronLink topology of Trn1/Trn2. Per chip: 4 × PCIe Gen6 ×8 intra-server (256 GB/s), 5 × PCIe Gen6 ×8 inter-server within rack (320 GB/s), 2 × PCIe Gen6 ×8 inter-rack (128 GB/s). Routing is address-based — a (rack, server, chip) tuple in the upper bits of the outbound PCIe address, resolved by BAR address matching at the switch — and is transparent to workloads.

For the collectives layer the significant property is **ordering**: the fabric provides hardware semaphores, and data and its completion semaphore are guaranteed to traverse the same physical path. NCCom therefore gets ordering from the fabric rather than from software fences, which is the hardware basis of its CPU-bypass collective dispatch on Trn3.

Finding 2 above ("NeuronLink topology constrains parallelism strategy") stands, but its mechanism should now be stated as: Trn3's all-to-all property comes from a PCIe Gen6 switch topology, not from a bespoke link layer.

> ⚠️ Per-device scale-up bandwidth is **unreconciled**: AWS's spec table says 2,048 GiB/s/device, the same page's link budget sums to 704 GB/s, and the 2.5 TB/s figure used in finding 4 above is SemiAnalysis's. Do not treat 2.5 TB/s as vendor-stated.

### Trn3 EFA figures (resolving the earlier "specs pending" gap)

AWS reports Trn3 EFA at **UltraServer granularity**: 12,800 Gbps per UltraServer Gen1 (64 devices) and 28,800 Gbps per Gen2 (144 devices) — a consistent **200 Gbps of EFA per Trainium3 device**. A per-instance Trn3 EFA figure is **not disclosed**. These numbers were published on 2025-12-02 and their earlier absence here was a research gap.

### Collectives and runtime changes in the window

- **`nki.collectives.all_to_all_v`** (variable-length all-to-all) added in Neuron SDK 2.29.0 — extends the kernel-level collective set recorded above.
- **Collectives support for the Trn3 Gen2 UltraServer ring topology** added in 2.30.0.
- **Coalesced reduce-scatter** optimization in the graph compiler (2.30.0).
- **Zero-copy host↔device transfers enabled by default** in the runtime (2.30.0); async event APIs.
- **Neuron DRA Driver** (2.30.0): Kubernetes Dynamic Resource Allocation with topology-aware scheduling of both Trainium devices and **EFA interfaces** — the first Neuron scheduling component that treats the scale-out NIC as a first-class allocatable resource. **UltraServer Operator** for Amazon EKS entered public beta in 2.31.0.
- NKI Library gained communication-relevant kernels: comm-compute fusion (2.29.0), context parallelism and MoE dispatch (2.30.0, experimental), MoE training collectives and ring attention (2.31.0, experimental).

### Deployment scale correction to finding 5

Project Rainier reached ~500,000 Trainium2 chips in 2025. Anthropic's 2026-04-20 newsroom post states it now uses **over one million Trainium2 chips** to train and serve Claude, with nearly 1 GW of combined Trn2 + Trn3 planned by end of 2026 inside an announced envelope of up to 5 GW. Party-stated figures, not audited counts.
