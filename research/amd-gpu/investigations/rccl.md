# RCCL Investigation Report

**Resource:** ROCm Communication Collectives Library (RCCL)
**URL:** https://github.com/ROCm/rccl
**Investigated:** 2026-04-05
**Commit:** 57e5868
**Note:** The standalone RCCL repo is retired; development has moved to [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems). The codebase investigated here represents the last canonical state.

---

## Overview

RCCL (pronounced "Rickle") is AMD's collective communication library for ROCm GPUs — the direct functional equivalent of NVIDIA's NCCL. It implements the standard collective operations (AllReduce, AllGather, Reduce, Broadcast, ReduceScatter, Scatter, Gather, AlltoAll, Send/Recv) and is consumed by PyTorch's distributed backend, DeepSpeed, and other multi-GPU frameworks on AMD hardware. Critically, RCCL is not a clean-room rewrite: it was forked from NCCL and retains the NCCL internal namespace (`nccl*` struct names, `NCCL_PARAM`, `ncclResult_t`), with AMD-specific extensions patched in under `#if defined(__HIP_PLATFORM_AMD__)` guards and marked `// [RCCL]` comments. The key differentiators introduced by AMD are: (1) native topology awareness for XGMI/Infinity Fabric links (replacing NVLink awareness with per-generation bandwidth tables for MI200, MI300X/gfx942, and gfx950); (2) a library of pre-computed `rcclRomeModel` topology templates for known AMD server configurations (EPYC Rome/Milan/Genoa nodes with 8-GPU configurations); (3) rail-optimized ring and tree algorithms for multi-node AMD clusters; (4) optional MSCCL++ integration for custom collective algorithms; and (5) ROCSMIsupport for hardware telemetry during initialization.

---

## Architecture

### Module Structure

```
rccl/
  src/
    collectives.cc        # Public API dispatch (ncclAllReduce, ncclAllGather, …)
    enqueue.cc            # Kernel planning, algorithm selection (getAlgoInfo), launch
    init.cc               # ncclCommInitRank, topology build, channel setup
    transport.cc          # Transport abstraction (selects P2P / NET / SHM)
    transport/
      p2p.cc              # P2P transport: direct XGMI/PCIe GPU-to-GPU via HIP IPC
      net.cc              # Network transport abstraction
      net_ib.cc           # InfiniBand Verbs (ibverbs) transport
      net_socket.cc       # TCP/IP socket fallback transport
      shm.cc              # Shared-memory transport (same-process, CPU-assisted)
      coll_net.cc         # CollNet (in-network compute) transport
    graph/
      topo.cc             # Topology XML parse, XGMI link enumeration, path computation
      topo.h              # ncclTopoSystem, ncclTopoNode, path/link type constants
      rome_models.cc      # Pre-computed rcclRomeModel templates + rail-opt selection
      rome_models.h       # rcclRomeModel struct definition
      search.cc           # Ring/tree graph search over topology
      connect.cc          # Channel-level connection setup, rail-optimized tree build
      tuning.cc           # Per-collective algorithm/protocol selection (getAlgoInfo impl)
      rings.cc            # Ring permutation generation
      trees.cc            # Tree (balanced/split) generation
      paths.cc            # All-pairs shortest path over topology graph
    device/
      all_reduce.h        # GPU kernel: runRing / runTree for AllReduce (HIP __device__)
      primitives.h        # Primitives<T,RedOp,Fan,…> — core data movement + reduce
      prims_simple.h      # ProtoSimple ring protocol primitives
      prims_ll.h          # ProtoLL (low-latency, 8B header+data) primitives
      prims_ll128.h       # ProtoLL128 (128B packet) primitives
      common.h            # ncclShmem, channel shared memory layout
    bootstrap.cc          # Out-of-band rank rendezvous (sockets)
    channel.cc            # Channel struct lifecycle
    proxy.cc              # CPU proxy thread for network operations
    msccl.cc              # MSCCL compatibility shim (deprecated)
  ext-src/                # External plugin/tuner API headers
  ext-tuner/              # Custom tuner plugin hook
```

### Key Abstractions

| Abstraction | File | Description |
|---|---|---|
| `ncclTopoSystem` | `src/graph/topo.h` | Complete node-level topology graph: GPU, PCI, CPU (NUMA), NIC, NET nodes; pre-computed all-pairs paths with bandwidth |
| `ncclTopoXGMISpeed(gcn)` | `src/graph/topo.h:311` | Per-arch XGMI bandwidth: gfx90a=36 GB/s, gfx942=48 GB/s, gfx95x=48 GB/s — used to annotate XGMI links at topology build time |
| `rcclRomeModel` | `src/graph/rome_models.h` | Pre-computed topology template: GPU bus IDs, NUMA affinity, XGMI connectivity matrix, GDR levels, and optimal `ringBase`/`treeRail` string patterns |
| `Primitives<T,RedOp,Fan,Proto>` | `src/device/primitives.h` | Templated GPU-side data-movement class; implements `send`, `recv`, `recvReduceSend`, `recvReduceCopy` using one of three wire protocols (Simple/LL/LL128) |
| `getAlgoInfo` | `src/enqueue.cc:338` | Runtime algorithm + protocol selector: queries topology bandwidth, message size, nRanks to pick (Ring/Tree/CollNet/NVLS) × (Simple/LL/LL128) |
| `ncclTopoAddXGMI` | `src/graph/topo.cc:584` | AMD-specific routine that reads `<xgmi>` XML nodes and adds `LINK_NVL` (reusing NVLink slot) edges between GPU nodes, scaled by per-arch speed |
| `useRailOptimizedTrees` | `src/graph/topo.h:218` | Boolean flag on `ncclTopoSystem`; when set, `connect.cc` builds multi-rail trees that route across NIC-local GPUs first, matching AMD's inter-node rail topology |

### Dependency Graph (simplified)

```
PyTorch / MPI / DeepSpeed
        |
   ncclAllReduce()          [collectives.cc]
        |
   ncclEnqueueCheck()       [enqueue.cc]
        |
   getAlgoInfo()  <-------> ncclTopoSystem [graph/topo.h]
        |                        ^
   taskAppend()           ncclTopoAddXGMI / rcclRomeModel
        |
   ncclGroupEndInternal()  → kernel launch via hipLaunchKernelGGL
        |
   ncclDevKernel_Generic_N()  [device/common.cu]
        |
   Primitives<T,RedOp,Fan,Proto>  [device/primitives.h]
        |
   Transport layer: p2p.cc / net_ib.cc / net_socket.cc / shm.cc
```

---

## Data Flow — AllReduce Trace

The following traces the ring-AllReduce path for a multi-GPU single-node call on a MI300X (gfx942) system.

### Step 1 — API Entry (`collectives.cc:246`)

```cpp
ncclResult_t ncclAllReduce_impl(sendbuff, recvbuff, count, datatype, op, comm, stream) {
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
    ALLREDUCE_CHUNKSTEPS,
    comm->rcclUseOneSlice ? ALLREDUCE_SLICESTEPS_SINGLE_NODE : ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);   // or mscclEnqueueCheck if MSCCL is active
}
```

The `rcclUseOneSlice` flag is an AMD addition that reduces pipeline depth on single-node XGMI systems where latency is low.

### Step 2 — Algorithm Selection (`enqueue.cc` → `graph/tuning.cc`)

`ncclEnqueueCheck` calls `taskAppend`, which calls `getAlgoInfo`. This function iterates over algorithm candidates (Ring, Tree, CollNet, NVLS) and protocols (LL, LL128, Simple), computing expected bandwidth from `comm->topo->maxBw` (set during init from XGMI/PCIe link speeds). The winner is stored in `task->algorithm` and `task->protocol`.

Key AMD influence: because `ncclTopoAddXGMI` has registered 48 GB/s `LINK_NVL` edges for MI300X GPUs, `maxBw` reflects full Infinity Fabric bandwidth, biasing the selector toward Ring (high throughput) over Tree (lower latency) at large message sizes.

### Step 3 — Work Dispatch (`enqueue.cc`)

`ncclTasksRegAndEnqueue` builds `ncclDevWorkColl` structs and calls `ncclKernelPlanner`, then submits via `hipLaunchKernelGGL` (HIP equivalent of `cudaLaunchKernel`):

```cpp
hipLaunchKernelGGL(ncclDevKernel_Generic_N, grid, block, 0, stream, args);
```

The generic kernel is parameterized by unroll factor (1/2/4). AMD adds `ncclDevKernelDebug_*` variants gated by `ENABLE_COLLTRACE` for latency profiling.

### Step 4 — GPU Kernel (`device/all_reduce.h`)

```cpp
// ring AllReduce inner loop (device/all_reduce.h)
Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0, false, RCCLMetadata, ...> prims
  (tid, nthreads, &ring->prev, &ring->next, sendbuff, recvbuff, ...);

for (int slice = 0; slice < nranks-1; slice++) {
  prims.recvReduceSend(offset, nelem);  // reduce-scatter phase
}
for (int slice = 0; slice < nranks-1; slice++) {
  prims.recvCopySend(offset, nelem);    // all-gather phase
}
```

`Primitives` writes/reads through per-channel ring buffers in GPU memory. For XGMI peers the buffer pointer is a direct HIP IPC handle into the remote GPU's VRAM — data moves over the physical Infinity Fabric links without host involvement.

### Step 5 — Transport Layer

For intra-node XGMI: `transport/p2p.cc` allocates `ncclP2pBuff` via `hipIpcGetMemHandle` / `hipIpcOpenMemHandle`. The `p2pResources.type` is set to `P2P_DIRECT` when XGMI direct access is confirmed, or `P2P_IPC` when going through the HIP IPC layer.

For inter-node IB: `transport/net_ib.cc` posts ibverbs RDMA `ibv_post_send` / `ibv_post_recv`. The CPU proxy thread (`proxy.cc`) drives the ibverbs completion queue asynchronously and signals GPU-side step counters.

---

## Hardware Interface — Infinity Fabric / XGMI

### Bandwidth Constants (topo.h)

```c
#define VEGA_XGMI_WIDTH   24.0   // GB/s  (MI50, MI60)
#define MI200_XGMI_WIDTH  36.0   // GB/s  (MI200 / gfx90a)
#define GFX94X_XGMI_WIDTH 48.0   // GB/s  (MI300X / gfx942)
#define GFX95X_XGMI_WIDTH 48.0   // GB/s  (MI350X / gfx950)
```

These are per-link bandwidths. MI300X has 7 XGMI links per GPU (6 to peer GPUs + 1 to CPU die in the MCM), each running at 48 GB/s.

### XGMI vs NVLink Design Choice (topo.cc)

RCCL uses the `LINK_NVL` constant (shared with NCCL's NVLink) for XGMI edges in the topology graph. This is intentional: at the graph layer, XGMI is treated as equivalent to NVLink — a direct high-bandwidth GPU-to-GPU link that bypasses PCIe. The difference is detected at runtime via `gpu.gcn` (the GCN architecture string from HIP) inside `ncclTopoXGMISpeed`, which dispatches to the per-arch bandwidth constant.

### XML Topology Scan

During `ncclCommInitRank`, RCCL queries the system PCIe/XGMI topology through `rocm-smi` or AMD SMI (`amdsmi_wrap.cc`), serializes it to an XML string, then feeds it to `ncclTopoGetSystemFromXml`. The XML contains `<xgmi count="N" target="BUSID" tclass="class_0x0302"/>` nodes for each XGMI link. `ncclTopoAddXGMI` recursively walks this XML, resolves bus IDs to `ncclTopoNode` structs, and calls `ncclTopoConnectNodes` with the bandwidth from `ncclTopoXGMISpeed`.

### Rome/EPYC Model Database (rome_models.cc)

For known AMD server platforms, RCCL maintains ~30 pre-computed `rcclRomeModel` entries (e.g., `rome_model_22` for 8-GPU 4-NUMA-domain Rome servers with 2 XGMI links/GPU). Each model contains:

- `connMatrix[8x8]`: which GPU pairs have direct XGMI links
- `gdrLevel[N]`: whether GPU-Direct RDMA is usable for each GPU-NIC pair
- `ringBase`: optimal ring order string (e.g., `"7 4 5 3 1 0 6 2|4 7 3 5 0 1 2 6"`)
- `treeRail`: multi-rail tree assignment for inter-node collective optimization

When `rcclGetRomeSystem` matches a live system against the database, the optimal pre-computed ring/tree is used directly instead of running the full graph search, which is computationally expensive for large GPU counts.

### Rail-Optimized Trees

AMD clusters often use a "rail" NIC topology where each GPU is associated with a local NIC on the same PCIe switch. RCCL detects this via `RCCL_TOPO_GDR_ALL` flag and builds trees where intra-rail GPUs exchange data before forwarding to network — minimizing the number of GPU-NIC crossings and maximizing GDR utilization.

---

## Key Findings

1. **RCCL is a maintained NCCL fork, not a rewrite.** The NCCL-inherited code is large; AMD modifications are additive and clearly marked. This means RCCL benefits from NCCL algorithm improvements upstream but also inherits NCCL assumptions (e.g., NVLink topology concepts mapped 1:1 onto XGMI).

2. **XGMI is RCCL's primary performance lever.** The library is architected around the assumption that intra-node peers have direct XGMI (Infinity Fabric) connections. P2P transport selects `P2P_DIRECT` mode which writes directly into peer GPU VRAM over XGMI — no PCIe bus involved. Bandwidth constants show MI300X at 48 GB/s/link, competitive with NVLink 4.

3. **The Rome model database is a hard-coded topology oracle.** Unlike NCCL's pure graph-search approach, RCCL pre-computes optimal rings for known AMD platform configurations. This improves collective startup time but means unexpected platform configurations (new server BOM, different NIC placement) may fall back to the slower generic search.

4. **Three wire protocols trade off latency vs throughput.** ProtoLL (low-latency) uses 8-byte in-band headers for small messages; ProtoLL128 uses 128-byte packets; ProtoSimple maximizes pipelining for large tensors. The selection boundary is message-size dependent and tuned per-topology. AMD's XGMI bandwidth makes ProtoSimple the winner for large AllReduces typical in training.

5. **MSCCL is deprecated.** The MSCCL scheduler (Microsoft-contributed custom collective algorithm engine) exists in the codebase but is marked `deprecated` in the 2024 codebase. MSCCL++ is the preferred replacement but remains optional (`--enable-mscclpp` build flag).

6. **ROCm-SMI / AMD SMI integration.** `init.cc` optionally links against `amdsmi` or `rocm_smi` to query GPU topology at communicator creation. This is required for XGMI link detection on multi-GPU nodes.

7. **HIP replaces CUDA throughout.** All `cudaStream_t` types are aliased to `hipStream_t`, `cudaLaunchKernel` → `hipLaunchKernelGGL`, and CUDA device properties → HIP equivalents. The HIP portability layer means RCCL can theoretically run on NVIDIA GPUs via HIP-on-CUDA, though this is not a production use case.

---

## Relation to Hardware Architecture

RCCL's design decisions map directly to AMD GPU hardware characteristics:

| Hardware Feature | RCCL Mechanism | File |
|---|---|---|
| XGMI / Infinity Fabric links | `LINK_NVL` topology edges, `ncclTopoAddXGMI`, per-arch BW constants | `topo.cc`, `topo.h` |
| MI300X MCM die-to-die fabric | 7-link topology; XGMI to CPU die treated as C2C path | `topo.h:LINK_C2C`, `topo.cc` |
| EPYC NUMA topology | CPU nodes in topology graph; inter-NUMA QPI/infinity paths shape ring order | `topo.cc`, `rome_models.cc` |
| GPU Direct RDMA (GDR) | `gdrLevel` per GPU-NIC pair; `PATH_PIX` / `PATH_PHB` determines GDR eligibility | `rome_models.cc`, `net_ib.cc` |
| HIP IPC / VRAM peer access | `P2P_DIRECT` / `P2P_IPC` transport modes in `p2p.cc` | `transport/p2p.cc` |
| InfiniBand / RoCE NICs | ibverbs `net_ib.cc`; ROCm GDR path via `net_ib_rocm.cc` | `transport/net_ib*.cc` |
| AMD GCN architecture string | `gpu.gcn` field drives XGMI speed dispatch and arch-specific tuning | `topo.h:ncclTopoNode` |

The overall picture is that RCCL is the "glue" between the collective communication API expected by ML frameworks and the AMD hardware interconnect hierarchy. It abstracts away the difference between XGMI topologies (intra-node fast path) and PCIe+network topologies (inter-node) using the same ring/tree algorithm framework — with AMD-specific topology databases and bandwidth constants filling in the hardware-specific knowledge.
