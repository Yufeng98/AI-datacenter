# NCCL Investigation Report

**Resource:** https://github.com/NVIDIA/nccl  
**as_of:** 2026-04-05  
**commit:** 49839df

---

## Overview

NCCL (NVIDIA Collective Communications Library, pronounced "Nickel") is a standalone C/CUDA library that provides optimized collective communication primitives for multi-GPU and multi-node GPU clusters. It sits in the **communication layer** of the AI datacenter software stack, directly above the hardware transport fabric (NVLink, NVSwitch, InfiniBand, PCIe) and below framework-level distributed training libraries such as `torch.distributed` / `c10d`, Horovod, and JAX.

NCCL implements: AllReduce, AllGather, Reduce, Broadcast, ReduceScatter, and arbitrary point-to-point Send/Recv. The library is consumed via a plain C API (`ncclComm_t`, `ncclAllReduce(...)`, etc.) and linked as `libnccl.so`. It is the de-facto standard collective layer used by virtually every serious GPU training framework.

**Stack position:**

```
  Framework (PyTorch / JAX / PaddlePaddle)
        │  torch.distributed / c10d / horovod
        ▼
      NCCL  ◄── this library
        │
  Transport fabric
  (NVLink / NVSwitch / InfiniBand / PCIe / GPUDirect RDMA)
        │
      GPU Hardware (SM, HBM, NIC)
```

---

## Architecture

### Module Structure

```
nccl/
├── src/
│   ├── nccl.h.in              # Public C API template
│   ├── init.cc                # Communicator lifecycle
│   ├── enqueue.cc             # Collective dispatch & kernel launch
│   ├── collectives.cc         # Collective op registration
│   ├── proxy.cc               # CPU-side proxy thread (net I/O)
│   ├── bootstrap.cc           # Out-of-band rank rendezvous
│   ├── channel.cc             # Channel management
│   ├── transport.cc           # Transport abstraction
│   ├── transport/
│   │   ├── p2p.cc             # NVLink / PCIe peer-to-peer
│   │   ├── shm.cc             # Shared-memory (same-host, same-process)
│   │   ├── net.cc             # Network transport (IB / socket)
│   │   ├── net_ib/            # InfiniBand verbs, GDR, GDAKI
│   │   ├── nvls.cc            # NVLink SHARP (multicast)
│   │   └── coll_net.cc        # CollNet (in-network compute)
│   ├── graph/
│   │   ├── topo.cc / topo.h   # Hardware topology discovery
│   │   ├── search.cc          # Optimal channel graph search
│   │   ├── rings.cc           # Ring algorithm assignment
│   │   ├── trees.cc           # Tree algorithm assignment
│   │   ├── connect.cc         # Transport connection setup
│   │   └── tuning.cc          # Algorithm / protocol selection
│   ├── device/
│   │   ├── all_reduce.h       # AllReduce GPU kernel (ring + tree)
│   │   ├── primitives.h       # Send/recv primitive templates
│   │   ├── prims_simple.h     # SIMPLE protocol primitive
│   │   ├── prims_ll.h         # LL (low-latency) protocol
│   │   ├── prims_ll128.h      # LL128 (128-byte NVLink) protocol
│   │   ├── reduce_kernel.h    # Reduction operations (Sum, Prod, Min, Max)
│   │   └── symmetric/        # Symmetric (NVLS / multicast) kernels
│   ├── plugin/
│   │   ├── net/               # Versioned network plugin interface (v6–v11)
│   │   ├── tuner/             # Tuner plugin interface (v2–v5)
│   │   ├── profiler/          # Profiler plugin interface (v1–v6)
│   │   └── gin/               # GIN (GPU Initiated Networking) plugin
│   ├── register/              # User buffer registration (zero-copy)
│   ├── rma/                   # Remote Memory Access
│   ├── scheduler/             # AllGatherV / symmetric scheduling
│   ├── gin/                   # GIN host-side implementation
│   ├── nccl_device/           # Device-side shared data structures
│   ├── ras/                   # Reliability / availability / serviceability
│   └── misc/                  # ibvwrap, gdrwrap, cudawrap, socket, utils
├── bindings/
│   └── nccl4py/               # Python (Cython) bindings
└── plugins/                   # External plugin examples
    ├── net/, tuner/, profiler/, env/, mixed/
```

### Dependency Graph (simplified)

```
  Public API (nccl.h)
       │
  init.cc / enqueue.cc / collectives.cc
       │
  ┌────┴──────────────────────────────────────────────┐
  │  graph/ (topo discovery, channel search, tuning)  │
  │  transport/ (p2p, shm, net, nvls, coll_net)       │
  │  proxy.cc  (CPU-side network I/O thread)          │
  │  register/ (buffer pinning / zero-copy)           │
  └────────────────────────────────────────────────────┘
       │
  device/ (CUDA kernels: collectives + primitives)
       │
  Hardware (NVLink/PCIe/IB/GPUDirect)
```

### Key Abstractions

1. **`ncclComm_t` — Communicator**  
   The central opaque handle (`struct ncclComm` in `src/include/comm.h`). Encapsulates: rank membership, topology graph, transport connections, channel array, proxy thread state, tuner config, CUDA stream binding, and per-rank buffer registrations. Created via `ncclCommInitRank()` / `ncclCommInitRankConfig()`.

2. **Channel (`ncclDevChannel` / `ncclChannel`)**  
   A logical data-transfer lane mapped onto a physical interconnect path (one or more NVLinks, one PCIe path, or one network device). Multiple channels run concurrently to saturate bandwidth. Each channel carries either a ring topology or a tree topology (and may carry both for different algorithms). Defined in `src/include/channel.h` and the device-side view in `src/include/nccl_device/comm.h`.

3. **Transport Abstraction (`ncclTransport`)**  
   A vtable-based interface (`src/include/transport.h`) with four concrete implementations:
   - `TRANSPORT_P2P (0)` — direct GPU memory access over NVLink or PCIe via CUDA IPC / cuMem
   - `TRANSPORT_SHM (1)` — CPU shared memory for same-process, same-node ranks
   - `TRANSPORT_NET (2)` — InfiniBand Verbs or TCP sockets for inter-node traffic
   - `TRANSPORT_COLLNET (3)` — In-network compute (SHARP, collective offload)
   Each transport provides `canConnect`, `send.*`, and `recv.*` function pointers for setup, connect, free, proxy, and progress operations.

4. **Communication Primitive Templates (`Primitives<T, RedOp, Fan, Direct, Proto>`)**  
   C++ template classes in `src/device/primitives.h` and `prims_*.h` that run inside GPU kernels. They implement the low-level send/receive ring slots with hardware-specific protocols:
   - **ProtoSimple** — bulk data, optimal for large messages and NVLink
   - **ProtoLL** — low-latency, 8-byte data + 8-byte flag per 16-byte unit, for small messages
   - **ProtoLL128** — 128-byte NVLink cache-line optimized, best for medium messages over NVLink
   Fan templates (`FanSymmetric<N>`, `FanAsymmetric<Recv, Send>`) parametrize the ring/tree arity.

5. **Topology System (`ncclTopoSystem`)**  
   Discovered at `ncclCommInitRank()` time by parsing the system's PCI tree and querying NVML for NVLink fabric topology (`src/graph/topo.cc`). Nodes are typed as `GPU`, `PCI`, `NVS` (NVSwitch), `CPU` (NUMA domain), `NIC`, `NET`, `GIN`. Bandwidth constants are hardcoded per SM generation (e.g., `SM90_NVLINK_BW = 20.6 GB/s/link`, `SM100_NVLINK_BW = 40.1 GB/s/link`). Path types are classified as `PATH_LOC`, `PATH_NVL`, `PATH_NVB` (through intermediate GPU), `PATH_C2C`, `PATH_PIX`, `PATH_PXB`, `PATH_PXN`, `PATH_PHB`, `PATH_SYS`, `PATH_NET`. This topology drives channel search and algorithm selection.

6. **Proxy Thread**  
   A per-communicator CPU thread (`src/proxy.cc`) that drives network I/O (IB send/recv, socket send/recv) on behalf of the GPU. The GPU kernel enqueues network work items to a host FIFO; the proxy thread drains the FIFO, posts IB work requests or socket sends, and writes completion flags back into GPU-accessible memory. For NVLink-only intra-node collectives, no proxy thread work is needed.

7. **NVLS (NVLink SHARP) Transport**  
   Added in CUDA 12.1+, implemented in `src/transport/nvls.cc`. Uses CUDA Multicast Objects (`cuMulticastCreate`) to create a virtually shared buffer across multiple GPUs on the same NVSwitch fabric. This allows a reduce-scatter or all-gather to be performed by writing once to the multicast address and having all GPUs receive reduced data simultaneously, matching hardware-level SHARP semantics.

---

## Data Flow — AllReduce Traced Path

**Scenario:** 8-GPU ring AllReduce, large message, NVLink fabric, ProtoSimple protocol.

### Step 1: User Call
```c
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
```
Enters `src/enqueue.cc` → `ncclEnqueueCheck()` validates arguments → `ncclSetupAllReduce()` selects algorithm (Ring vs Tree vs NVLS) and protocol (Simple/LL/LL128) by consulting the tuner plugin and topology bandwidths.

### Step 2: Work Descriptor Construction
`enqueue.cc` builds an `ncclDevWorkColl` descriptor (algorithm, protocol, data type, reduce op, buffer pointers, element count, channel assignments) and enqueues it per channel onto the communicator's `devWorkFifo`.

### Step 3: Kernel Launch
`enqueue.cc` calls `cudaLaunchKernel` (or `cudaGraphLaunch` if graph capture is active) for the appropriate NCCL device kernel — selected from a table indexed by `(algorithm, protocol, datatype, reduceOp)`. For AllReduce + Ring + Simple + float + sum, this dispatches `ncclKernel_AllReduce_RING_SIMPLE_Sum_float`.

### Step 4: Ring Reduce-Scatter (GPU kernel, `src/device/all_reduce.h: runRing()`)
Inside the kernel, each CTA owns one channel's ring step. The ring has `nranks` steps:

- **Step 0 (push):** Each rank pushes its own data chunk to its ring-next neighbor via `prims.directSend(offset, offset, nelem)`. "Direct" means writing straight to the peer's CUDA buffer using P2P (NVLink) or to a staging buffer for NET.
- **Steps 1 … nranks-2 (reduce-forward):** Each rank receives the incoming chunk from its ring-prev, reduces it with its local chunk, and forwards the result to ring-next via `prims.directRecvReduceDirectSend(...)`.
- **Step nranks-1 (final reduce):** Each rank performs the last local reduction and produces the final sum for its owned chunk via `prims.directRecvReduceCopyDirectSend(...)`.

At this point each rank holds one fully-reduced chunk.

### Step 5: Ring All-Gather (GPU kernel, continuing `runRing()`)
- **Steps nranks … 2*nranks-3 (copy-forward):** Each rank forwards its reduced chunk around the ring via `prims.directRecvCopyDirectSend(...)`.
- **Final step:** Each rank receives the last missing chunk via `prims.directRecv(...)`.

All ranks now hold the complete AllReduce result.

### Step 6: Protocol-level Data Movement (inside `Primitives`)
For **ProtoSimple**: `directSend` writes to the remote peer's receive buffer using NVLink P2P write (cuMem / CUDA IPC mapping), then updates a flag in a shared `ncclConnFifo` slot. The receiver spins on the flag (NCCL_STEPS slots deep), then reads directly from its own buffer.

For **ProtoLL128**: data and inline flags are packed into 128-byte cache-line units and written atomically to NVLink, reducing the round-trips needed for fence/flag synchronization.

### Step 7: Network Path (inter-node)
For ranks on different nodes, the primitive falls through to NET transport. The GPU kernel writes data to a staging buffer accessible by the proxy thread, posts a request to the proxy FIFO, and the proxy thread calls `ibv_post_send` (InfiniBand) or `send()` (TCP). On the receiver side, the proxy polls `ibv_poll_cq`, copies data into the GPU buffer via GDRCopy (RDMA into GPU memory), and writes a completion flag that the GPU kernel polls.

---

## Hardware Interface

### NVLink / NVSwitch

- Detected at init via `NVML` and PCIe topology enumeration (`src/graph/topo.cc`). NVLink bandwidth constants are per-SM-generation: Pascal (18 GB/s/link), Volta/Ampere (20 GB/s/link), Hopper (20.6 GB/s/link), Blackwell (40.1 GB/s/link).
- P2P transport uses `cuMemAddressReserve` / `cuMemMap` (or legacy `cudaIpcGetMemHandle`) to map peer GPU memory so that GPU kernels can issue NVLink load/store directly — no CPU involvement.
- The topology graph classifies NVSwitch-connected paths as `LINK_NVL` / `PATH_NVL`. Paths traversing an intermediate GPU (without a physical NVSwitch) are `PATH_NVB`.
- NVLS (NVLink SHARP) uses `cuMulticastCreate` / `cuMulticastAddDevice` to form a multicast group. Kernels write to a UC (unicast) alias that hardware broadcasts across all GPUs' MC (multicast) mappings simultaneously, enabling hardware-reduced AllReduce without ring/tree software steps.

### Multi-Node NVLink (MNNVL)

- Detected via `nvmlGpuFabricInfoV_t` (`src/mnnvl.cc`). When GPUs across nodes are connected via NVSwitch fabric (e.g., DGX SuperPOD / GB200 NVL72), NCCL sets `comm->MNNVL = 1` and uses FABRIC-type `cuMemGenericAllocationHandle` to enable direct cross-node NVLink P2P writes, bypassing InfiniBand entirely.

### InfiniBand / GPUDirect RDMA

- Implemented in `src/transport/net_ib/` using libibverbs (wrapped via `src/misc/ibvwrap.cc` and dynamic symbol loading).
- GPUDirect RDMA: `src/include/gdrwrap.h` wraps the `gdrcopy` kernel module, enabling the IB HCA to DMA data directly from/to GPU HBM. The receive buffer is registered with `ibv_reg_mr` using GPU virtual addresses; `ibv_post_recv` / `ibv_post_send` then transfer data without staging through CPU DRAM.
- **GDAKI (GPU Direct Accelerated Kernel Initiated):** A newer path in `src/transport/net_ib/gdaki/` and `src/include/nccl_device/gin/gdaki/` using DOCA GPUNetIO, where GPU kernels post IB work requests directly from the SM, completely bypassing the CPU proxy for IB messaging.
- `src/include/gdrwrap.h` also handles a WC (write-combining) fence needed for PCIe/IB paths on x86 vs POWER.

### GIN (GPU Initiated Networking)

- Defined in `src/include/plugin/nccl_gin.h` and implemented in `src/gin/`. Allows GPU SM threads to directly initiate network operations (send/recv) into InfiniBand or NVLink without a CPU proxy round-trip. Requires supported HW (e.g., ConnectX-7 with GPUNetIO support or MNNVL fabric).

### PCIe

- Used when GPUs are on different PCIe segments with no NVLink. P2P writes go via the PCIe root complex; bandwidth constants show `PCI_BW = 12.0 GB/s` (Gen3 x16). Intel CPUs incur a 20% overhead because they convert 512-bit GPU P2P TLPs into 64-byte chunks (`INTEL_P2P_OVERHEAD(bw) = bw*6/5`).
- When P2P is not available, NCCL falls back to `TRANSPORT_SHM` (CPU shared memory bounce buffer) or `TRANSPORT_NET`.

---

## Key Findings

1. **Protocol trinity:** NCCL's three communication protocols (Simple, LL, LL128) are chosen at runtime per collective based on message size and interconnect type. LL128 is specifically designed to saturate NVLink's 128-byte cache-line width, giving it a structural advantage over generic approaches for medium-sized messages on NVLink.

2. **NVLS is a hardware-software co-design:** The NVLink SHARP transport (`nvls.cc`) requires both NVSwitch hardware with multicast capability and CUDA 12.1+ `cuMulticastObject` API. It lets NCCL offload the reduce tree into the switch fabric, achieving near-wire-speed AllReduce for intra-node collectives on DGX H100/H200/B200 systems.

3. **Topology-driven algorithm selection:** At communicator init, NCCL builds a full system topology graph and searches for optimal ring/tree channel assignments using bandwidth-weighted graph search (`src/graph/search.cc`, `tuning.cc`). The selected algorithm (Ring, Tree, CollnetChain, CollnetDirect, NVLS, NVLSTree) and protocol directly depend on the physical interconnect discovered — not a static default.

4. **Plugin extensibility:** Network transport, tuning policy, profiling, and environment variable parsing are all exposed as versioned plugin interfaces (`src/include/plugin/`). This allows vendors (e.g., AWS EFA, Google TCPx) to inject custom transports without forking NCCL.

5. **GIN / GDAKI removes CPU bottleneck:** The proxy-less GPU Initiated Networking path (`src/gin/`, `src/transport/net_ib/gdaki/`) enables GPU SMs to directly post IB work requests, which is critical for extremely large GPU clusters where the proxy CPU thread becomes a throughput bottleneck.

6. **MNNVL expands NVLink beyond a single node:** With GB200 NVL72 (72 GPUs across 9 nodes over NVSwitch fabric), NCCL detects the FABRIC handle type and treats cross-node GPUs as if they were on local NVLink, allowing AllReduce to run entirely over NVSwitch at ~900 GB/s aggregate bandwidth.

7. **Buffer registration for zero-copy:** `src/register/` allows the application to pre-pin send/recv buffers with `ncclCommRegister()`. NCCL then maps them directly into the transport (IB MR, NVLS multicast) without per-operation registration overhead, which is material for small-message-dominated workloads.

---

## Relation to Hardware Architecture

NCCL's design choices are inseparable from the GPU interconnect topology that NVIDIA has shipped across generations:

- **Volta (V100) introduced NVLink 2.0 with NVSwitch** — NCCL gained the concept of full-mesh topologies and introduced the LL128 protocol tuned to NVLink cache lines.
- **Ampere (A100) on DGX A100** — 8 GPUs fully connected via 3rd-gen NVLink + NVSwitch. NCCL's ring and tree algorithms already exploit this; `PATH_NVL` paths dominate intra-node routing.
- **Hopper (H100) introduced NVLS (NVLink SHARP / NVSwitch collective offload)** — the hardware can perform reductions inside the switch. NCCL 2.18+ added the NVLS transport path to exploit this, enabling single-pass AllReduce at hardware speeds.
- **Blackwell (B200) and GB200 NVL72** — NVLink 5.0 at 40.1 GB/s/link, and the MNNVL fabric that spans multiple nodes. NCCL's `mnnvl.cc` and FABRIC handle support were added to cover this, collapsing the inter-node/intra-node distinction for the first time.
- The proxy thread design (CPU-side IB progress engine) reflects the historical reality that GPU SM threads could not initiate RDMA; GIN / GDAKI removes this constraint on modern ConnectX-7 + H100/B200 hardware.

In summary, NCCL is a thin but highly hardware-aware library whose internal algorithm branches, protocol widths, bandwidth constants, and transport implementations directly mirror the successive generations of NVIDIA's GPU interconnect hardware.
