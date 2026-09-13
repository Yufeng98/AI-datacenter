# HCCL & Distributed Training Investigation

*chip: intel-gaudi*
*as_of: 2026-04-05*
*sources: docs.habana.ai v1.23.0 (HCCL API Reference, Scaling Guide), github.com/HabanaAI/Megatron-DeepSpeed, github.com/HabanaAI/hccl_demo*

---

## Overview

HCCL (Habana Collective Communications Library) is Intel Gaudi's NCCL-compatible collective communication library, included as a core component of the SynapseAI software stack. What distinguishes HCCL from NCCL is its direct integration with Gaudi's on-die RoCE v2 NICs: where NCCL relies on NVLink or an external InfiniBand/Ethernet NIC, HCCL programs the 24 integrated 200 GbE ports on Gaudi 3 (21 scale-up + 3 scale-out) directly, achieving RDMA transfers into and out of HBM2e without CPU involvement. This allows HCCL to implement a unified scale-up and scale-out topology over standard Ethernet—the same physical ports serve both intra-node and inter-node traffic. PyTorch distributed training uses HCCL as the `hccl` process group backend (a drop-in replacement for `nccl`), and both DeepSpeed and Megatron-DeepSpeed have Gaudi forks that switch the communication backend to HCCL.

---

## Architecture

### Component Structure

```
Application (PyTorch / DeepSpeed / Megatron)
         |
torch.distributed (process group API)
         | backend="hccl"
HCCL Library (libhccl.so — part of SynapseAI)
         ├── Collective scheduler (AllReduce, AllGather, ReduceScatter, Broadcast, Barrier)
         ├── Point-to-point (Send/Recv)
         ├── Scale-up path: → programs integrated RoCE QP descriptors
         │                    → RDMA READ/WRITE directly on peer HBM2e
         │                    (21 × 200 GbE ports per Gaudi 3 card)
         └── Scale-out path: → same API, routes to 3 × 200 GbE external ports
                               → or Host NIC (GDR / peer-direct, optional)

habanalabs kernel driver
         └── RoCE engine DMA ← programs NIC send/recv queue pairs
```

### Key Abstractions

| Abstraction | Description |
|---|---|
| `hccl` process group | Drop-in replacement for `nccl` in `torch.distributed.init_process_group(backend="hccl")`; wraps HCCL collective ops |
| Integrated RoCE QP | Queue Pair (RDMA primitive) programmed by HCCL directly against Gaudi's on-die NIC; target address is a peer card's HBM2e physical address |
| Scale-up topology | Network of 21 × 200 GbE on-die ports per card used for intra-node/intra-rack communication; can connect card-to-card or through standard Ethernet switch |
| Scale-out topology | 3 × 200 GbE on-die ports per card for cross-rack multi-node clusters; same Ethernet fabric, no protocol boundary |
| HCCL Communicator | Handle encapsulating a group of ranks, their QP mappings, and collective algorithm selection; created via `hcclCommInitRank` |
| Host NIC scale-out | Optional alternative: uses server's Ethernet/InfiniBand NIC with GDR (GPU Direct RDMA) for scale-out when integrated ports are insufficient |

### Supported Collective Primitives

- **AllReduce** (primary primitive for data-parallel gradient synchronization)
- **AllGather** (used in ZeRO-3 parameter reconstruction, Megatron tensor parallelism)
- **ReduceScatter** (ZeRO-3 gradient sharding)
- **Broadcast** (initial weight synchronization)
- **Reduce** (partial gradient collection)
- **Barrier** (synchronization fence)
- **Send / Recv** (point-to-point for pipeline parallelism)

---

## Data Flow

### Data-Parallel AllReduce (gradient synchronization across 8 Gaudi 3 cards in one server)

1. **Backward pass complete**: Each rank's gradient tensor resides in its HBM2e
2. **HCCL AllReduce call**: PyTorch distributed triggers `hcclAllReduce` on the `hccl` communicator
3. **Algorithm selection**: HCCL selects ring-allreduce or recursive halving-doubling based on message size and rank count
4. **QP programming**: HCCL programs RoCE Queue Pair descriptors on the integrated NIC, targeting peer cards' HBM2e addresses (obtained during communicator initialization)
5. **RDMA transfer**: Integrated NIC engines execute RDMA READ/WRITE operations directly on HBM2e — no CPU copy, no host memory bounce buffer
6. **Reduction**: Each card accumulates received gradients into its local HBM2e buffer (TPC or dedicated reduction hardware handles the add operation)
7. **Completion**: HCCL signals completion event; SynapseAI runtime resumes optimizer step

### Multi-Node AllReduce (scale-out, e.g., 16 nodes × 8 cards)

- HCCL routes intra-node traffic through 21 × 200 GbE scale-up ports (direct or via ToR switch)
- HCCL routes inter-node traffic through 3 × 200 GbE scale-out ports to the multi-rack Ethernet fabric
- From the application's perspective, the API call is identical — HCCL manages the topology split transparently

### DeepSpeed ZeRO-3 with HCCL

1. Optimizer states, gradients, parameters are sharded across ranks (using `torch.distributed` with `backend="hccl"`)
2. `AllGather` called before forward pass to reconstruct full-precision weight tensors (HCCL AllGather → integrated NIC RDMA)
3. `ReduceScatter` called after backward pass to shard gradients (HCCL ReduceScatter → integrated NIC RDMA)
4. Overlap: DeepSpeed prefetch engine triggers `AllGather` for layer N+1 while layer N is computing — HCCL NIC ops run in parallel with TPC/MME compute

---

## Hardware Interface

- **Integrated RoCE NIC (on-die)**: HCCL programs the 24 NIC ports' queue pairs via the habanalabs driver. Each port implements full RoCEv2: reliable connection (RC) mode, remote memory access to any mapped HBM2e region.
- **HBM2e as RDMA target**: HCCL registers HBM2e tensor buffers as RDMA memory regions (MR); peer cards read/write directly. No host DRAM copy required.
- **DMA engine coordination**: The SynapseAI runtime coordinates HCCL RDMA DMA with compute DMA (SRAM↔HBM) to avoid contention on HBM2e bandwidth.
- **Host NIC fallback (scale-out)**: If 3 on-die scale-out ports are insufficient, HCCL can use the host server's NIC with `Peer Direct` (RDMA from host NIC directly to Gaudi HBM2e via PCIe peer-to-peer).
- **No external switch required for scale-up**: Within an 8-card server tray, the 21 scale-up ports can be cabled card-to-card in a full-mesh or partial-mesh topology — no Ethernet switch needed for intra-node communication.

---

## Key Findings

1. **Unique among AI accelerators: on-die Ethernet NICs**: Gaudi is the only major AI training accelerator with RoCE NICs integrated on the compute die. HCCL exploits this to deliver collective communications without any discrete NIC chip, reducing system cost and latency.
2. **Unified scale-up and scale-out over Ethernet**: Unlike NVIDIA's dual-fabric model (NVLink for scale-up, InfiniBand for scale-out), Gaudi uses a single Ethernet fabric for both, simplifying cluster networking but requiring careful port allocation (21 vs. 3 split on Gaudi 3).
3. **NCCL API compatibility enables drop-in replacement**: `backend="hccl"` is a one-line change in PyTorch distributed setup; existing NCCL-based training scripts (DeepSpeed, Megatron, FSDP) work with no further modification.
4. **Overlap of communication and compute is explicit**: HCCL operations are non-blocking by default; the SynapseAI runtime schedules NIC DMA and TPC/MME execution on separate engines, allowing genuine compute-communication overlap.
5. **Host NIC scale-out is a supported escape hatch**: For very large clusters where 3 × 200 GbE scale-out ports are insufficient, HCCL supports an alternative path through the host server NIC using GDR, preserving flexibility without code changes.
6. **Megatron-DeepSpeed fork required**: HabanaAI maintains a dedicated fork (`github.com/HabanaAI/Megatron-DeepSpeed`) with HCCL backend substitution, HPU kernel calls, and 3D parallelism (tensor + pipeline + data) for large-scale LLM pretraining on Gaudi clusters.

---

## Relation to Hardware Architecture

HCCL's design is inseparable from Gaudi's on-die integrated RoCE architecture. The 21 + 3 port split on Gaudi 3 directly shapes HCCL's two-level topology: scale-up uses the 21 ports to saturate intra-node and intra-rack bandwidth (up to 4.2 Tbps/card unidirectional), while scale-out uses the 3 ports for cross-rack traffic. Because the NICs share die area with the MME and TPC engines, HCCL communication truly overlaps with compute — both the NIC engines and the TPC/MME engines issue commands from independent scheduler queues and operate on independent memory regions of HBM2e. This is the primary architectural advantage Gaudi offers over accelerators that use external NIC chips connected via PCIe.
