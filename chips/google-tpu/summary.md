# Google TPU — Software & Hardware Stack Summary

*as_of: 2026-09-13*
*device_class: domain-specific AI accelerator (training + inference)*

---

## Overview

Google's Tensor Processing Units (TPUs) are purpose-built AI accelerators organized around a deterministic, systolic array execution model. Unlike NVIDIA's SIMT GPU architecture — which exposes thousands of general-purpose threads and a rich user-visible ISA — TPUs expose a small number of very wide VLIW-scheduled compute engines optimized exclusively for dense matrix algebra and structured collective communication. Every design decision in the TPU hardware and software stack flows from this premise: predictability over flexibility, compiler control over programmer control, vertical integration over open interfaces.

The deployed datacenter lineup spans TPU v4, v5e, v5p, v6e (Trillium), v7 (Ironwood), and the v8 generation. The eighth generation, announced at Google Cloud Next '26 on 2026-04-22, splits the TPU lineup into two architecturally distinct chips for the first time: **TPU v8t (codename Sunfish)** for pre-training and large-scale RL, and **TPU v8i (codename Zebrafish)** for inference, post-training, and agentic reasoning. Google credits the design to work "in partnership with Google DeepMind" and does **not** disclose a process node or an ASIC design partner for either chip; press reporting (Wccftech and others) attributes Sunfish to Broadcom, Zebrafish to MediaTek, and both to TSMC 2nm — treat those three attributions as press-reported and unconfirmed by Google. Both chips host on Google Axion (Arm Neoverse-V2) servers and add a new FP4 path (OCP MX microscaling). They diverge at the on-chip SRAM, memory, and interconnect tiers: v8t carries 216 GB HBM3e (6,528 GB/s) with **128 MB on-chip SRAM (Vmem)** — flat against v7 — and stays on the 3D torus topology (extended by the new Virgo optical scale-out fabric to ~1M chips multi-DC), while v8i carries 288 GB HBM3e at 8,601 GB/s with **384 MB on-chip SRAM (3× v7)** and adopts a new high-radix **Boardfly** topology that cuts the 1,024-chip network diameter from 16 hops to 7 — a structural choice tuned for the small, latency-bound collectives of decode-time reasoning. Google's specialized-feature split is also asymmetric: v8t lists *SparseCore (embeddings) and an **LLM Decoder Engine*** (function not detailed), while v8i lists the on-chip **Collectives Acceleration Engine (CAE)**, which lowers small-tensor reduction latency by up to 5×. Google describes v8i as a chiplet part — "two Tensor Cores on-core dies and one CAE on the chiplet die" per chip.

For reference, v7 Ironwood — still the deployed flagship, and as of 2026-08-08 still the newest generation in the Cloud TPU supported-version list — provides two TensorCores per chip, 256×256 MXU, 192 GB HBM3e at 7.4 TB/s, 4 SparseCores, FP8 hardware support, and a 9,216-chip 3D torus pod delivering 42.5 ExaFLOPS aggregate FP8 performance. v8t roughly triples that pod-level peak at 121 FP4 ExaFLOPS over 9,600 chips (12.6 PFLOPS peak FP4/chip) and 2 PB shared HBM, with **2.7×** advertised training price/performance vs Ironwood. v8i targets pods of up to 1,152 physically connected chips / **1,024 active chips** (10.1 PFLOPS peak FP4/chip; ~11.6 **FP4** ExaFLOPS at 1,152 chips, 331.8 TB total HBM) with 1.8× advertised inference price/performance (Google: "up to 80% performance-per-dollar improvement"). Google claims up to 2× better performance-per-watt for both chips. **Availability:** Google states "both chips will be generally available later this year" (i.e. calendar 2026); as of 2026-08-08 neither chip is orderable and neither appears in the Cloud TPU release notes or supported-version list.

---

## Software Stack

### Framework Integration

JAX is Google's primary ML framework for TPU. It was co-designed with the TPU execution model: JAX functions compiled for TPU must be pure (no Python side effects) and statically traceable to a jaxpr, because the TPU's static VLIW execution requires a complete compiled binary before any computation can begin. The main transforms are:

- `jax.jit` — traces Python to jaxpr, lowers to StableHLO, triggers XLA compilation, and dispatches to TPU. Not optional on TPU.
- `jax.grad` — reverse-mode automatic differentiation via jaxpr adjoint computation.
- `jax.vmap` — vectorization over a new batch axis without explicit loops.
- `jax.shard_map` — explicit SPMD with `lax.all_reduce`, `lax.all_gather` etc. for expert users needing finer-grained control than Shardy's automatic propagation.
- `jax.Array` + `NamedSharding` — unified array type with mesh-based sharding annotations that Shardy propagates automatically through the computation graph.

The production JAX AI Stack adds: **Flax** (neural network modules via NNX/Linen), **Optax** (composable gradient optimizers; Adam, Adafactor, LAMB), **Orbax** (async checkpointing to GCS), **Grain** (deterministic data loading from GCS/ArrayRecord), **Tokamax** (production Pallas kernels: FlashAttention, MoE routing, RoPE), and **MaxText** (open-source LLM reference implementation for Gemma, Llama, DeepSeek, Mistral).

PyTorch accesses TPU via **PyTorch/XLA** (`torch_xla`), which lowers PyTorch ops to StableHLO for XLA compilation. As of 2025–2026, vLLM uses JAX as the lowering path for all TPU inference, including PyTorch-authored models. TensorFlow is supported up to TPU v6e; it is not supported on v7 Ironwood.

### Compiler / IR

The compilation chain from Python to TPU binary passes through five distinct representations:

```
Python function
    | jax.jit tracing
jaxpr (JAX functional IR)
    | JAX lowering
StableHLO (MLIR opset, ~100 ops, versioned portability layer)
    | Shardy SPMD partitioning (sharding propagation + collective insertion)
HLO (XLA internal SSA graph)
    | XLA optimization passes:
    |   operator fusion, layout assignment, memory space assignment,
    |   buffer assignment, CSE/DCE, rematerialization, dot tiling
Optimized HLO
    | XLA TPU backend
LLO (Low Level Operations — TPU-specific IR)
    | TPU assembler
VLIW instruction bundles (8 ops/cycle → TPU binary)
```

**XLA** is the primary compiler, packaged inside `libtpu.so`. Its highest-impact pass is **operator fusion**: adjacent element-wise ops, activations, and norms are fused into the output of matrix multiplies, eliminating HBM roundtrips for intermediate tensors.

**StableHLO** is the portability seam. It is an MLIR dialect with ~100 operations providing backward and forward compatibility guarantees. JAX, PyTorch/XLA, and TensorFlow all serialize to StableHLO, allowing the compiler and hardware to evolve independently. This is the TPU analogue of PTX's role in NVIDIA's stack — with the critical difference that StableHLO is not user-visible and carries no promise of execution-time portability across hardware generations.

**Shardy** (completed migration in JAX March 2026, replacing GSPMD) is the MLIR-based SPMD partitioner. Given sharding annotations on input/output tensors expressed as `NamedSharding` with logical mesh axes (e.g., `('batch', 'model', 'pipeline')`), Shardy propagates sharding through the entire computation graph and inserts `AllReduce`, `AllGather`, `AllToAll`, `ReduceScatter`, and `CollectivePermute` collective operations wherever sharding changes.

### Op Library

TPU has no separate Op Library layer in the NVIDIA/AMD sense (no cuDNN-equivalent). Standard deep learning operations (convolution, attention, normalization) are expressed as compositions of XLA HLO primitives and fused by the XLA operator fusion pass. The XLA compiler itself subsumes the op library role, producing optimized fused kernels directly from the HLO graph without a separate library dispatch path.

For operations requiring custom tiling or memory orchestration not achievable via automatic fusion, **Pallas** kernels provide the escape hatch (see Kernel Library).

### Kernel Library

**Pallas** is JAX's kernel DSL — the TPU analogue of Triton for NVIDIA GPUs. Pallas kernels are Python functions using `jax.numpy` semantics, decorated with `pallas_call`, specifying:
- A `grid` of logical blocks over the output space.
- `BlockSpec` objects defining how input/output tensors are tiled across the grid.
- `MemorySpace` annotations (VMEM, CMEM) for on-chip placement.

Inside the kernel body, all operands are VMEM-resident references (not HBM pointers); HBM-to-VMEM DMA is scheduled by Mosaic.

**Mosaic** is the TPU kernel compiler backend that translates Pallas IR (jaxpr) to VLIW instruction bundles. Mosaic:
- Maps `BlockSpec` tile dimensions to MXU input requirements (aligned to 128×128 or 256×256).
- Generates double-buffered DMA commands (N-stage prefetch) so tile N+1 is in-flight from HBM while tile N is computing on the MXU.
- Schedules VLIW instruction slots for MXU + VPU + DMA + scalar ops in parallel.
- Emits LLO for final instruction packing.

Pallas kernels appear as `custom_call` HLO ops embedded in the main XLA program. **Tokamax** is Google's production Pallas kernel library: FlashAttention variants, MoE routing, RoPE embeddings, automatically selecting the right variant for the target TPU generation.

### Runtime

**libtpu.so** is the monolithic shared library on every Cloud TPU VM. It bundles three subsystems in a single deployable unit:

1. **XLA Compiler** — full AOT/JIT compilation pipeline from StableHLO to VLIW binaries, with Shardy partitioner integrated. On-disk binary cache keyed by content hash eliminates the 30–120 second compile latency for repeated runs with the same model architecture.
2. **TPU Kernel Driver** — PCIe command queue submission, DMA management, interrupt handling, HBM buffer allocation/deallocation, memory-mapped I/O to TPU hardware.
3. **ICI Runtime** — implementations of ring AllReduce (each torus axis), AllGather, AllToAll, ReduceScatter, CollectivePermute; 3D/2D torus routing tables; multi-pod rendezvous (DCN coordination).

libtpu versions are tightly coupled to JAX releases and upgraded atomically. There is no separate driver package, compiler package, or communication library — all three are upgraded together, which makes TPU software updates atomic but requires matched JAX + libtpu versions.

### Driver / Firmware

The TPU kernel driver is part of libtpu.so rather than a standalone kernel module in the NVIDIA/AMD sense. From a deployment standpoint, libtpu includes the userspace driver component; the minimal kernel-mode portion manages PCIe DMA and interrupt routing. Google does not publish a standalone open-source kernel module for TPU.

### Communication

Distributed communication is expressed via `jax.lax` collective primitives, inserted automatically by Shardy during XLA compilation:

| JAX Primitive | XLA HLO Op | ICI Implementation |
|---|---|---|
| `lax.psum` | AllReduce | Bidirectional ring along each torus axis |
| `lax.all_gather` | AllGather | ICI-routed gather along torus axis |
| `lax.all_to_all` | AllToAll | Full data exchange via ICI |
| `lax.psum_scatter` | ReduceScatter | Combined reduce + scatter |
| `lax.ppermute` | CollectivePermute | Neighbor-to-neighbor direct send |

Intra-pod collectives use ICI. Inter-pod collectives (multi-pod Multislice) use DCN via Titanium IPU offload. XLA schedules DCN AllReduce during windows when intra-pod MXU compute can proceed concurrently.

### Assembler / ISA

TPU has no publicly exposed ISA. Google controls the full compilation stack from Python to VLIW instruction bundles, with LLO as the internal low-level IR. There is no PTX-equivalent virtual ISA accessible to users or third-party compilers. This is a deliberate design choice: by keeping the ISA private, Google retains the ability to change the microarchitecture between generations without maintaining binary compatibility guarantees. Code compiled for v5p does not run optimally on v6e without recompilation; there is no forward-compatibility mechanism at the binary level.

---

## Hardware Architecture

### Compute Engine

Each TPU chip contains one or more **TensorCores** — Google's term for the full processing die unit (distinct from NVIDIA's "Tensor Core" functional unit). Each TensorCore contains:

- **MXU (Matrix Multiply Unit)**: weight-stationary systolic array, 128×128 (v4/v5) or 256×256 (v6e/v7/v8), executing 16,384 or 65,536 MACs/cycle respectively.
- **VPU (Vector Processing Unit)**: element-wise operations (activations, norms, transposes), executing concurrently with MXU in separate VLIW slots.
- **ScalarCore**: control flow, loop iteration, address generation.
- **VMEM**: per-TensorCore SRAM scratchpad (~16–32 MB on v4–v6e; ~64 MB on v7, ~128 MB/chip; **128 MB per chip on v8t, 384 MB per chip on v8i**; compiler-managed). Google publishes v8 Vmem only as a per-chip figure — the per-TensorCore breakdown for v8t/v8i is **not disclosed**.
- **CMEM**: scalar memory for loop counters, DMA descriptors, small LUTs.

TPU v7 Ironwood has 2 TensorCores per chip (131,072 MACs/chip) and is the first generation with FP8 hardware support in both TensorCores and MXU. It also has 4 SparseCores per chip for embedding table lookup acceleration.

TPU v8t and v8i both retain 2 TensorCores per chip and add native **FP4** support (OCP MX microscaling, with BF16 accumulation). MXU dimensions for v8 are **not stated** in Google's deep-dive; the 256×256 array is carried forward from v6e/v7 and should be read as an assumption, not a disclosure. The two chips diverge at the on-chip SRAM tier (v8t: 128 MB Vmem/chip; v8i: 384 MB Vmem/chip), at the memory tier (v8t: 216 GB HBM3e at 6,528 GB/s; v8i: 288 GB HBM3e at 8,601 GB/s), at the interconnect (v8t: 3D torus; v8i: Boardfly high-radix), and in their specialized blocks. Google's spec table lists v8t's specialized features as **SparseCore (embeddings) plus an LLM Decoder Engine** — a previously unrecorded named block whose function Google does not detail — and v8i's as the **Collectives Acceleration Engine (CAE)**, which handles the per-token small-tensor reductions of autoregressive decoding in dedicated silicon, reducing on-chip collective latency by up to 5×. Google does **not** publish SparseCore counts for v8, and does not attribute a SparseCore to v8i at all. v8i is a chiplet part: two TensorCore on-core dies plus one CAE chiplet die per chip.

### Data Path

TPU uses a **VLIW** execution model with 8-operation-per-cycle instruction packets. Each packet schedules one operation per functional unit: MXU (two slots for load/execute pipeline), VPU (two slots), DMA read (HBM→VMEM), DMA write (VMEM→HBM), scalar ALU, and scalar load/store (CMEM). All scheduling is static; there is no dynamic issue logic, branch predictor, or out-of-order execution hardware. This eliminates significant silicon area and power, enabling larger systolic arrays and more HBM capacity per die area.

The fundamental data movement pattern is **weight-stationary**:
```
HBM → [DMA] → VMEM (weights, staged ahead of compute)
HBM → [DMA] → VMEM (activations, one tile at a time)
VMEM → MXU (weights held stationary in array cells)
VMEM → MXU (activations stream through rows each cycle)
MXU → VMEM (output partial sums accumulated)
VMEM → [DMA] → HBM (output tensors)
```
DMA engines transfer HBM tiles asynchronously with double-buffering, so tile N+1 is prefetched while tile N computes on the MXU.

### On-chip Memory

VMEM is the primary on-chip SRAM (~16 MB per TensorCore for v4/v5, ~32 MB for v6e, ~64 MB for v7 = ~128 MB/chip; 128 MB/chip on v8t and 384 MB/chip on v8i). It is explicitly compiler-managed — no hardware cache. Workloads that spill from VMEM to HBM suffer approximately 100× bandwidth reduction. XLA's operator fusion pass exists primarily to keep intermediate tensors in VMEM and avoid spills.

CMEM is a separate small SRAM for scalar operands (loop counters, DMA metadata, ICI control registers), keeping scalar and vector memory accesses on independent bandwidth paths.

### Off-chip Memory

HBM capacity has grown from 16 GB (v5e) through 95 GB (v5p), 144 GB HBM3 (v6e), 192 GB HBM3e at 7.4 TB/s (v7 Ironwood), to **216 GB HBM3e at 6,528 GB/s on v8t** and **288 GB HBM3e at 8,601 GB/s on v8i**. The 9,216-chip Ironwood superpod aggregates 1.77 PB total HBM; v8t's 9,600-chip superpod aggregates 2 PB; v8i's 1,152-chip pod aggregates 331.8 TB (Google also describes the v8i pod as "up to 1,024 active chips"). The weight-stationary MXU design reduces weight read amplification: for large-batch inference, weights are loaded once into VMEM per layer per batch step (not per sequence), making TPU HBM bandwidth spending proportionally dominated by activations and KV cache traffic rather than weight reloads. v8i's HBM-bandwidth advantage over v8t (8.6 vs 6.5 TB/s) reflects this directly: inference workloads are bandwidth-bound on KV-cache reads and weight reloads at small batch, while v8t's training workloads are bound by ICI bandwidth and FLOPS throughput, allowing v8t to spend less of the chip's pin budget on HBM.

### Host Interface / Package

TPU VMs run the user's Python/JAX process on a host CPU co-located with the TPU on the same board. libtpu.so manages all host-to-TPU communication. PCIe Gen 4 x16 (~64 GB/s) is the host-to-HBM transfer path, used for model parameter loading at initialization. Training and inference operate entirely from HBM during execution.

### Scale-up Interconnect

ICI (Inter-Chip Interconnect) is Google's proprietary high-speed intra-pod fabric, implementing 3D torus (v4/v5p/v7/v8t), 2D torus (v5e/v6e), or — new in v8i — the **Boardfly** high-radix topology. v7 Ironwood provides 9,600 Gbps (1.2 TB/s) per chip; v8t and v8i both **double** this to 19.2 Tbps (2.4 TB/s) per chip. v7 connects 9,216 chips in 64-chip 4×4×4 cube racks (copper DAC intra-cube, optical fiber inter-cube); v8t scales the same 3D-torus arrangement to 9,600 chips per superpod.

**Boardfly (v8i).** v8i adopts a high-radix topology in which each *board* presents an internal all-to-all (full mesh) of its on-board chips, and boards are aggregated into board-of-boards groups. The diameter of a 1,024-chip pod drops from 16 hops on a 3D torus to 7 hops on Boardfly — a structural choice motivated by the latency-bound, small-tensor collectives of autoregressive decoding and MoE routing. v8i pods scale to 1,152 physically connected chips with 331.8 TB total HBM; Google separately describes a pod as "up to 1,024 active chips" (36 groups of 8 boards), so 1,152 is best read as the physical/aggregate count and 1,024 as the addressable one.

TPU v4 introduced **OCS (Optical Circuit Switches)** — 136×136 port MEMS-based optical switches — to make the 3D torus physically realizable at 4,096-chip scale. OCS are reconfigurable, enabling different logical torus shapes from the same physical hardware and twisted torus configurations that improve bisection bandwidth by up to 70%. The OCS approach is retained on v8t for shape flexibility within a 9,600-chip superpod.

### Scale-out Interconnect

**Multislice** (v4–v7) connects multiple ICI pods via datacenter network (DCN) for training scales beyond a single pod. **Titanium IPUs** (Google in-house SmartNIC/DPU) offload DCN AllReduce and RDMA operations. The full v7-era AI Hypercomputer abstraction can aggregate up to 147,456 chips across 16 pods.

**Virgo Network (v8t).** v8 introduces a new optical scale-out fabric, **Virgo**, supporting up to **134,000 chips** in a single-datacenter fabric and up to **~1,000,000 chips** across multiple datacenters in a single training cluster. Virgo replaces the DCN-routed Multislice path for training-scale workloads on v8t, with Titanium IPUs continuing to offload cross-pod reductions; XLA collective insertion treats Virgo as a high-bandwidth, moderate-latency outermost mesh axis.

---

## Programming Model Rationale

### Why JAX's Functional Transform Model Maps to XLA/Systolic Arrays

JAX's insistence on functional purity (`jax.jit` requires traceable, side-effect-free functions) is not an aesthetic choice — it is a hardware requirement. The TPU's VLIW execution model requires a completely static instruction schedule before any computation can begin. A program with Python side effects, dynamic shapes, or data-dependent control flow cannot be statically compiled to a VLIW binary. JAX's tracing mechanism (producing a jaxpr from pure functions) is the only way to extract a static computation graph from Python.

`jax.grad` composes with `jax.jit` because automatic differentiation operates on the jaxpr (a static graph), not on the Python execution. The gradient of a pure function is itself a pure function; it compiles to a VLIW binary of its own via the same pipeline. This composability is unique to JAX's design: PyTorch's eager autograd engine cannot produce a fully static computation for TPU without `torch.compile`, which must approximate static tracing.

`jax.vmap` batching and `jax.shard_map` parallelism both work at the jaxpr level, adding batch or shard axes to the computation graph before XLA sees it. This means the compiler always receives a single, fully specified computation graph representing the entire batch or the per-shard computation, which it can tile optimally for the MXU. There is no concept of "dynamically dispatching to different batches at runtime."

### Why Pallas/Mosaic Give Kernel-Level Control for MXU's Weight-Stationary Dataflow

XLA's automatic tiling is optimal for standard GEMM and convolution patterns where the access pattern is regular and the XLA compiler can predict the optimal tile dimensions. It cannot match hand-tuned kernels for:
- **Attention** — the softmax over scores requires VPU operations interleaved with MXU dots at tile granularity; XLA fusion cannot produce the precise double-buffering pattern that FlashAttention-style kernels require.
- **MoE routing** — sparse gather of expert activations requires explicit control over which VMEM buffers hold which expert tiles.
- **Custom quantization** — FP8 dequantization inserted at exactly the right memory stage to avoid VMEM blowup.

Pallas/Mosaic's `BlockSpec` mechanism gives kernel authors direct control over VMEM tile placement. The kernel author specifies: "each block of my output uses this tile from VMEM; Mosaic should double-buffer the HBM→VMEM DMA for this input." Mosaic then generates the DMA schedule that overlaps the HBM fetch for tile N+1 with the MXU computation of tile N. This is the hardware-level optimization that matters for weight-stationary TPU performance: keeping the MXU fed without stalling on HBM latency.

### Why libtpu Is a Monolithic .so — Google's Vertically Integrated Design

NVIDIA's software stack is composed of separately versioned, separately installable components (CUDA toolkit, cuBLAS, NCCL, driver). This composability serves NVIDIA's ecosystem model: third-party software vendors can ship cuBLAS or NCCL versions independently of the GPU driver.

Google has no such ecosystem constraint. The TPU is a closed proprietary system; Google ships both the hardware and the primary software. libtpu.so is monolithic because:

1. **ISA privacy**: There is no public TPU ISA. libtpu is the sole compiler from user code to TPU binaries. There is no third-party compiler that needs to emit TPU code independently of the driver, so separate compiler and driver packages provide no benefit.
2. **Co-versioning**: The XLA compiler must know the exact hardware capabilities (MXU size, VMEM capacity, VLIW slot assignments) to generate optimal code. When Google ships a new TPU generation, the compiler, driver, and ICI runtime must all change together. Bundling them as a single .so makes this atomicity explicit and enforced.
3. **ICI topology coupling**: The ICI runtime inside libtpu uses routing tables that are specific to the pod topology (3D vs 2D torus, OCS configuration). These tables are generated by the same software that configures the ICI fabric, making it natural to colocate ICI runtime and driver.
4. **Operational simplicity**: Cloud TPU VMs are managed infrastructure. Google controls when libtpu is upgraded on its infrastructure; users are expected to match the libtpu version advertised by their TPU software stack version. This atomic upgrade model eliminates the versioning mismatches that frequently afflict NVIDIA CUDA users.

### Why ICI 3D Torus Topology Shapes GSPMD/Shardy's Sharding Decisions

On a 3D torus, each chip has 6 neighbors (±1 in each of X, Y, Z). Ring AllReduce along any one axis uses every link along that axis at full bidirectional bandwidth with zero contention. Three independent ring AllReduces (one per axis, sequenced) achieve near-peak efficiency for data-parallel gradient synchronization.

Shardy's collective insertion is topology-aware: it maps logical mesh axes (batch, model, pipeline) to physical ICI axes and selects collective operations that match the physical topology. A model-parallel AllReduce that crosses the X axis of the torus uses only X-axis links, leaving Y and Z axis bandwidth free for concurrent pipeline or data-parallel communication.

The OCS reconfigurability of TPU v4 extends this: by reprogramming the optical switches, the 4,096-chip pod can present as a 16×16×16 cube or an 8×16×32 slab. Shardy can exploit whichever shape minimizes communication overhead for a specific model's parallelism dimensions. This is why GSPMD/Shardy's "automatic" sharding is not generic graph partitioning — it is topology-matched graph partitioning that exploits specific ICI properties.

### Why There Is No CUDA-Like Low-Level ISA Exposed to Users

NVIDIA exposes PTX as a stable, versioned virtual ISA that users can write directly or compile Triton to. This serves NVIDIA's ecosystem: hardware vendors, academic researchers, and performance engineers can write optimized code at the ISA level without access to SASS documentation.

Google made the opposite trade-off. By not exposing a user-visible ISA:
- Google can change the TPU microarchitecture between generations without any binary compatibility burden. The v6e's 256×256 MXU rendered v5p code suboptimal, but because users cannot write ISA-level code, this was a recompilation problem, not a binary compatibility crisis.
- The compiler retains complete control over VLIW instruction packing, enabling aggressive optimization that user-written assembly would likely subvert.
- Security: no user-mode exploits based on undocumented ISA behavior.

The downside is that Pallas/Mosaic is the floor of the stack. Users who want kernel-level control must express it through Mosaic's tile/pipeline abstractions, not through raw instruction scheduling. For the ML workloads Google targets (transformer training and inference), this is an acceptable constraint: Mosaic's pipeline scheduling is expressive enough to implement FlashAttention, MoE routing, and other production kernels at near-peak performance.

---

## TPU v8 Correction & Software-Stack Update (2026-08-08)

*Window covered: 2026-04-27 → 2026-08-08. Primary sources: Google's TPU 8t/8i technical deep-dive and eighth-generation announcement (both 2026-04-22), the Cloud TPU release notes and supported-version list (both retrieved 2026-08-08). Prior-generation content above is unchanged; this section records corrections to v8 figures the repo already carried, plus the one substantive stack change in the window.*

**No new TPU silicon in this window.** v8t (Sunfish) / v8i (Zebrafish) remain the newest generation. There is **no TPU v9 information**, **no v8 pricing**, and **no MLPerf result attributable to a TPU** (Google is one of 24 submitting organizations for MLPerf Training v6.0, published 2026-06-16 with two new MoE benchmarks — DeepSeek V3 671B and GPT-OSS 20B — but MLCommons does not attribute hardware to individual submitters).

### Corrections to previously recorded v8 figures

| Item | Previously in this repo | Corrected (Google primary source) |
|---|---|---|
| On-chip SRAM (Vmem), v8t | 384 MB/chip (192 MB/TensorCore) | **128 MB/chip** — flat vs v7's ~128 MB/chip |
| On-chip SRAM (Vmem), v8i | 384 MB/chip (192 MB/TensorCore) | **384 MB/chip** — the "3× previous generation" claim applies to **8i only** |
| Per-TensorCore Vmem split, v8 | 192 MB/TensorCore | **not disclosed** — Google publishes a per-chip figure only |
| Training price/performance, v8t | 2.8× vs Ironwood | **2.7×** ("up to 2.7x performance-per-dollar improvement … for large-scale training") |
| v8i pod aggregate | ~11.6 **FP8** ExaFLOPS | ~11.6 **FP4** ExaFLOPS (1,152 × 10.1 PFLOPS peak FP4) |
| v8i pod size | 1,152 chips | 1,152 physically connected / **up to 1,024 active** chips per pod |
| HBM bandwidth | ~6.5 TB/s (8t), 8.6 TB/s (8i) | **6,528 GB/s** (8t), **8,601 GB/s** (8i) — the rounded figures were right, these are exact |
| SparseCore on v8 | 4 SparseCores on both v8t and v8i | SparseCore listed for **v8t only**; **count not published** for v8. v8i's specialized feature is the CAE |
| Process node / design partners | "TSMC 2nm; v8t with Broadcom, v8i with MediaTek" stated as fact | **Press-reported, unconfirmed by Google** — none of Google's three Next '26 posts mention a node, TSMC, Broadcom, or MediaTek |

Two facts previously absent from the repo are added: v8t carries a named **LLM Decoder Engine** alongside SparseCore (Google gives no functional detail), and **v8i is a chiplet design** — "two Tensor Cores on-core dies and one CAE on the chiplet die" per chip.

### Availability status (as of 2026-08-08)

Google's announcement post states verbatim that "both chips will be generally available later this year", and the Next '26 infrastructure post says they "will be available to Cloud customers soon". Neither chip appears in the Cloud TPU supported-version list (which still tops out at TPU7x / Ironwood) and no v8 availability entry appears in the Cloud TPU release notes. The correct status is therefore **announced, vendor-stated GA target of calendar 2026, not yet available to customers**. There is no customer preview program for v8 silicon — the only "preview" in Google's deep-dive is for native PyTorch support on TPU. Any "external availability late 2027" figure in circulation is **not** supported by a retrievable source and is not recorded here.

### Software stack — Compute Engine-native TPU provisioning (2026-06-01, GA)

The one substantive stack change in the window, from the Cloud TPU release notes:

- **2026-06-01 (GA): Compute Engine natively supports TPUs.** TPU VMs and TPU slices can now be provisioned and managed through the standard Compute Engine instance and managed-instance-group APIs, with custom OS images and configurable boot-disk sizing, across all consumption options (on-demand, spot, reservation). This is a control-plane convergence: TPU fleets stop being a bespoke resource type managed through TPU-specific APIs (`gcloud compute tpus`) and become ordinary GCE instances, inheriting MIG autohealing/autoscaling, image management, and disk configuration. It changes provisioning and fleet operations, not the compiled execution path — JAX/XLA/libtpu behaviour on the device is unaffected.
- **2026-04-27 (GA):** "Cloud TPU now offers TPU availability in AI zones." Additive; same date as the prior refresh baseline.
- **No libtpu or JAX version entries** appear in the Cloud TPU release notes for May–August 2026, so **no SDK version numbers are recorded for this window**.

### Scheduled disclosure to re-scan (resolved 2026-09-13 — see below)

**Hot Chips 38** (Stanford, Aug 24–25 2026) session AI 2 (Tue 2026-08-25, 4:45–6:15 PM PDT, chair Brucek Khailany) lists *"The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era"* — Norman Jouppi & Sridhar Lakshmanamurthy, Google. As of the 2026-08-08 scan this was a scheduled disclosure with no public content. It has since been delivered — see "Update (2026-09-13)" below for what it added.

### Commercial capacity note (pre-baseline, recorded for completeness)

On **2026-04-06** Anthropic, Google, and Broadcom announced multiple gigawatts of next-generation TPU capacity coming online starting 2027, the vast majority sited in the US, described by Anthropic CFO Krishna Rao as the company's "most significant compute commitment to date"; it builds on the October 2025 up-to-1M-TPU agreement. This predates the 2026-04-27 refresh baseline and is a commercial-capacity fact rather than a hardware or stack fact.

### Still not disclosed

Per-TensorCore Vmem split for v8t/v8i; SparseCore counts for v8; MXU dimensions for v8 (the deep-dive never states 256×256); FP8 support on v8 (the deep-dive discusses FP4 only); v8 pricing; any TPU v9 information; any Meta TPU agreement beyond pre-baseline press reports.

---

## TPU v8 Hot Chips 38 Update (2026-09-13)

*Window covered: 2026-08-08 → 2026-09-13. Primary talk: "The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era" (Norman Jouppi, Sridhar Lakshmanamurthy), delivered at Hot Chips 38 on 2026-08-25. Source: ServeTheHome's session writeup (https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/, 2026-08-25) — a detailed secondary account attributed to the talk; no independently-retrieved transcript or slide deck. Still no TPU v9 information, no v8 pricing, and no new MLPerf attribution.*

This is the first detailed architectural disclosure for v8t/v8i beyond the April 2026 Cloud Next material. It resolves several previously "not disclosed" items and adds new facts; it does **not** change any figure the 2026-08-08 pass had already sourced to Google's own blog posts.

**New/resolved facts:**
- **HBM stack count**: v8t (Sunfish) has **6 HBM stacks**; v8i (Zebrafish) has **8 HBM stacks**. This corrects an unsourced "4 stacks" estimate previously carried for v8t.
- **Superpod compute, directly stated**: v8t's 9,600-chip superpod delivers **121 EFLOPS of FP4 compute** — matching the figure this repo had already derived arithmetically (12.6 PFLOPS/chip × 9,600), now independently sourced to the talk itself.
- **Virgo network**: at its 134,000-chip single-domain scale, Virgo delivers **47 Pb/s of aggregate bandwidth**, over a **two-layer switching topology**. Neither figure was previously recorded.
- **Physical layout**: v8t superpods are built from **300 racks of 4-TPU trays** (300 × 32 = 9,600 chips). v8i pods use **8 trays of 4 TPUs per group, 36 groups** (1,152 chips) — a more granular version of the April 2026 "36 groups of 8 boards" language.
- **Perf/W**: re-confirmed "around twice" Ironwood's perf/W for v8t, consistent with the existing "up to 2×" figure.
- **Design process**: Google states it used AI assistance in the v8t/v8i design process for power and area optimization (no further detail).

**Checked and not found — do not treat as confirmed:** the ServeTheHome Hot Chips coverage gives **no GA/availability date** for either chip. A "GA late 2027" figure circulating in some secondary commentary about TPU 8i was specifically checked against this source and **could not be corroborated**; it is not recorded. The April 2026 vendor statement ("both chips will be generally available later this year", i.e. calendar 2026) remains the only sourced availability figure. The Cloud TPU release notes and supported-version list were **not re-checked** in this pass (last checked 2026-08-08, no v8 entry).

**Still not disclosed:** per-TensorCore Vmem split; SparseCore counts for v8; MXU array dimensions for v8; FP8 support on v8; process node and ASIC design partners (Broadcom/MediaTek/TSMC 2nm remain press-reported only — the available Hot Chips coverage did not address fab or partner questions); v8 pricing; TPU v9.

---

## Resources

### Compiler / Runtime
- [OpenXLA/XLA Repository](https://github.com/openxla/xla)
- [StableHLO Specification](https://openxla.org/stablehlo/spec)
- [Shardy Overview — OpenXLA](https://openxla.org/shardy/overview)
- [Shardy Guide for JAX Users](https://openxla.org/shardy/getting_started_jax)
- [Shardy JAX Migration](https://docs.jax.dev/en/latest/shardy_jax_migration.html)
- [From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
- [libtpu on PyPI](https://pypi.org/project/libtpu/)
- [TPU Software Versions Matrix](https://docs.cloud.google.com/tpu/docs/runtimes)

### Pallas / Mosaic
- [Pallas Design Document — JAX Documentation](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [MaxText Pallas Kernels Guide](https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html)

### JAX / Framework
- [JAX Repository](https://github.com/jax-ml/jax)
- [JAX Documentation](https://docs.jax.dev/en/latest/)
- [JAX JIT Compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [Building Production AI on Cloud TPUs with JAX](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)
- [MaxText Repository](https://github.com/AI-Hypercomputer/maxtext)
- [Flax Repository](https://github.com/google/flax)
- [Optax Repository](https://github.com/google-deepmind/optax)
- [Orbax Repository](https://github.com/google/orbax)
- [Grain Repository](https://github.com/google/grain)
- [JAX Scaling Book — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)

### Hardware
- [TPU v8 (Sunfish/Zebrafish) Announcement — Google Blog (2026-04-22)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [TPU 8t and TPU 8i Technical Deep Dive — Google Cloud Blog](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
- [AI Infrastructure at Next '26 — Google Cloud Blog](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26)
- [TPU 8t/8i and Virgo Network Analysis — fundaai (Substack)](https://fundaai.substack.com/p/researchtpu-8t8i-and-virgo-network)
- [TPU v7 (Ironwood) Documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Ironwood TPU Announcement Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/)
- [Inside the Ironwood TPU Codesigned AI Stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [TPU v6e (Trillium) Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [TPU v5p Documentation](https://docs.cloud.google.com/tpu/docs/v5p)
- [TPU v4 Documentation](https://docs.cloud.google.com/tpu/docs/v4)
- [TPU v4 Whitepaper — arXiv:2304.01433 (ISCA 2023)](https://arxiv.org/abs/2304.01433)
- [TPU Architecture Overview — Google Cloud](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Multislice Training — Google Cloud](https://docs.cloud.google.com/tpu/docs/v5e-training)
- [AI Hypercomputer Overview](https://cloud.google.com/blog/products/compute/updates-to-ai-hypercomputer-software-stack/)
- [GSPMD Paper — arXiv:2105.04663](https://arxiv.org/abs/2105.04663)
- [Google TPU Architecture: 7 Generations Explained — Introl Blog](https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations)
- [TPU Deep Dive — Henry Ko](https://henryhmko.github.io/posts/tpu/tpu.html)

### Added 2026-08-08
- [Cloud TPU Release Notes](https://docs.cloud.google.com/tpu/docs/release-notes) — 2026-06-01 GA: Compute Engine native TPU support; 2026-04-27 GA: TPU availability in AI zones
- [Cloud TPU Supported Versions / System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — retrieved 2026-08-08; newest documented generation is TPU7x (Ironwood), no v8 entry
- [Hot Chips 38 Program](https://hotchips.org/program/conference/) — Aug 24–25 2026; session AI 2 lists Google's eighth-generation TPU family talk (disclosure scheduled, content not yet public)
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 24 submitters incl. Google; hardware not attributed per submitter
- [Anthropic / Google / Broadcom Compute Partnership (2026-04-06)](https://www.anthropic.com/news/google-broadcom-partnership-compute) — multi-gigawatt next-generation TPU capacity from 2027

### Added 2026-09-13
- [Google's TPUv8s for Training and Inference at Hot Chips 2026 — ServeTheHome](https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/) (2026-08-25) — Hot Chips 38 session writeup for "The Eighth Generation TPU Family" talk (Jouppi, Lakshmanamurthy); HBM stack counts, 121 EFLOPS FP4 superpod figure, Virgo 47 Pb/s bandwidth, rack/tray physical layout; no GA date given
