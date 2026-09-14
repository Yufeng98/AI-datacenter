# Google TPU Hardware Architecture Investigation

**Layer**: Hardware Architecture (all 7 layers)
**Chip**: Google TPU (v4 / v5e / v5p / v6e Trillium / v7 Ironwood / v8t Sunfish / v8i Zebrafish)
**as_of**: 2026-04-27

---

## Overview

Google's Tensor Processing Units (TPUs) are purpose-built AI accelerators organized around a deterministic, systolic array execution model fundamentally different from NVIDIA's SIMT GPU architecture. Where GPUs expose thousands of general-purpose SIMT threads, TPUs expose a small number of very wide VLIW-scheduled compute engines optimized for dense matrix algebra. This design choice trades flexibility for predictability: a TPU's execution timeline is compiler-determined, eliminating dynamic scheduling hardware and enabling extremely high utilization of the matrix units.

The current datacenter TPU lineup spans four generations actively deployed on Google Cloud:

| Generation | Codename | MXU Size | HBM Capacity | ICI Topology | Pod Scale |
|---|---|---|---|---|---|
| v4 | — | 128×128 | 32 GB | 3D torus + OCS | 4,096 chips |
| v5e | — | 128×128 | 16 GB | 2D torus | 256 chips |
| v5p | — | 128×128 | 95 GB | 3D torus | 8,960 chips |
| v6e | Trillium | 256×256 | 144 GB | 2D torus | 256 chips |
| v7 | Ironwood | 256×256 | 192 GB | 3D torus | 9,216 chips |

TPU v7 (Ironwood) is the current flagship, delivering 4,614 FP8 TFLOPS per chip and scaling to a 9,216-chip superpod delivering 42.5 ExaFLOPS aggregate FP8 performance.

---

## 1. Compute Engine

### 1.1 TensorCore Architecture

Each TPU chip contains one or more **TensorCores** — the fundamental compute die unit. A TensorCore is not the same as NVIDIA's "Tensor Core" (which is a functional unit within an SM); instead, Google's TensorCore is a full processing element containing:

- **MXU (Matrix Multiply Unit)**: The systolic array for matrix multiply-accumulate
- **VPU (Vector Processing Unit)**: For element-wise operations (activations, norms, transposes)
- **ScalarCore**: For control flow, loop iteration, address generation
- **VMEM (Vector Memory / Scratchpad SRAM)**: Per-TensorCore on-chip working memory
- **CMEM (Control Memory / Scalar Memory)**: Small fast memory for scalar variables and DMA metadata

TensorCore counts per chip:
- **TPU v4, v5e, v5p**: 1 TensorCore per chip
- **TPU v6e (Trillium)**: 1 TensorCore per chip (with wider MXU)
- **TPU v7 (Ironwood)**: 2 TensorCores per chip

### 1.2 MXU (Matrix Multiply Unit) — Systolic Array

The MXU is a weight-stationary systolic array — the defining compute structure of every TPU generation. Data flows through a grid of multiply-accumulate cells in a systolic (rhythmic, pipelined) fashion:

**Weight-stationary operation:**
1. Weights are loaded into the systolic array cells and held stationary
2. Input activations stream through rows of the array left-to-right
3. Each cell multiplies its stationary weight by the passing activation and adds to an accumulated sum
4. Partial sums propagate downward through columns
5. Final accumulated outputs exit at the bottom

This design maximizes weight reuse — each weight multiply participates in multiple dot products without re-fetching from VMEM — making it ideally suited for the multiply-accumulate-heavy patterns of transformer inference and training.

**MXU dimensions by generation:**

| Generation | MXU Size | MACs/cycle | Notes |
|---|---|---|---|
| TPU v1 | 256×256 | 65,536 | INT8 only (inference-only) |
| TPU v2 | 128×128 | 16,384 | BF16 training |
| TPU v3 | 128×128 | 16,384 | 2× v2 FLOPs via HBM2 + overclocking |
| TPU v4 | 128×128 | 16,384 | Sparse embedding support added |
| TPU v5e | 128×128 | 16,384 | Efficiency-optimized variant |
| TPU v5p | 128×128 | 16,384 | High-memory variant for training |
| TPU v6e (Trillium) | 256×256 | 65,536 | 4× MACs/cycle vs v5e at same clock |
| TPU v7 (Ironwood) | 256×256 | 65,536 per TensorCore | 2 TensorCores/chip = 131,072 MACs/chip |

**Data type support progression:**
- TPU v1: INT8 only
- TPU v2/v3: BF16 (Google invented BF16 for ML training; 8-bit exponent, 7-bit mantissa)
- TPU v4/v5: BF16, INT8, FP32 accumulation
- TPU v6e (Trillium): Added FP16
- TPU v7 (Ironwood): Added FP8 (first TPU with hardware FP8 support in MXU and TensorCores)

### 1.3 SparseCore

SparseCore is a specialized accelerator within the TPU chip for sparse embedding table lookups — the dominant operation in large-scale ranking and recommendation systems (ads, search, YouTube).

**SparseCore presence by generation:**
- TPU v4: First generation with SparseCore (1 per chip)
- TPU v5p: SparseCore for large-scale embedding training
- TPU v6e (Trillium): 3rd-generation SparseCore (improved bandwidth and capacity)
- TPU v7 (Ironwood): 4 SparseCores per chip (2× v6e); expanded to support financial, scientific, and graph workloads beyond recommendation systems

**SparseCore operation:**
SparseCore accelerates the "gather" pattern: given a large embedding table (potentially 100s of GB) and a list of sparse integer indices, look up and aggregate the corresponding rows. This is irregular memory access — the antithesis of the MXU's dense systolic computation — requiring a separate hardware unit with dedicated SRAM, address translation, and aggregation logic.

### 1.4 VPU (Vector Processing Unit)

The VPU handles element-wise vector operations that cannot be expressed as matrix multiplies:
- Activation functions (ReLU, GELU, SiLU, softmax denominator)
- Layer normalization (reduction + multiply + add)
- Element-wise multiply/add (residual connections)
- Transcendental functions

VPU and MXU execute concurrently (in different VLIW slots), so well-structured kernels overlap activation computation with the next tile's matrix multiply.

---

## 2. Data Path

### 2.1 Weight-Stationary Dataflow

The MXU's weight-stationary architecture defines the fundamental data movement pattern on TPU:

```
HBM → [DMA] → VMEM (weight tiles, staged)
HBM → [DMA] → VMEM (activation tiles, streamed)
VMEM → MXU (weights loaded once, held stationary)
VMEM → MXU (activations streamed through rows each cycle)
MXU → VMEM (output partial sums accumulated, written back)
VMEM → [DMA] → HBM (output tensors)
```

DMA engines transfer data between HBM and VMEM asynchronously, overlapping with MXU computation. This is the hardware foundation for the double-buffering / pipeline prefetch techniques that Mosaic and XLA use.

### 2.2 VLIW Execution Model

TPU uses a **VLIW (Very Long Instruction Word)** execution model. A single VLIW instruction packet schedules up to 8 operations per clock cycle across independent functional units:

| VLIW Slot | Functional Unit |
|---|---|
| 0 | MXU: matrix multiply dispatch |
| 1 | MXU: matrix multiply pipeline |
| 2 | VPU: vector element-wise op |
| 3 | VPU: vector reduction |
| 4 | DMA read (HBM → VMEM) |
| 5 | DMA write (VMEM → HBM) |
| 6 | Scalar ALU (address gen, loop control) |
| 7 | Scalar load/store (CMEM access) |

Because all instruction scheduling is static (compiler-determined), the TPU has no dynamic issue logic, no branch predictor, no out-of-order execution hardware. This eliminates significant silicon area and power, enabling larger systolic arrays and more HBM capacity per die area.

### 2.3 HBM → VMEM → MXU Pipeline

The full data path for a matrix multiply:

1. **Host → HBM**: Model weights and activations are loaded into HBM from the host PCIe connection during initialization.
2. **HBM → VMEM (weight prefetch)**: DMA engines move the next tile of weights from HBM into a VMEM staging buffer. With double-buffering, tile N+1 is fetched while tile N is computing.
3. **HBM → VMEM (activation stream)**: Input activations are DMA'd into a separate VMEM buffer in tile-sized chunks.
4. **VMEM → MXU (weight load)**: The weight tile is written into the MXU's systolic array cells. This is a one-time cost per tile.
5. **VMEM → MXU (activation feed)**: Activation rows stream through the MXU array one row per clock cycle.
6. **MXU → VMEM (accumulate)**: Output partial sums exit the array and accumulate in VMEM output buffers.
7. **VMEM → HBM (output DMA)**: Completed output tiles are written back to HBM.

---

## 3. On-chip Memory

### 3.1 VMEM (Vector Memory / Scratchpad)

VMEM is the primary on-chip SRAM scratchpad, analogous to shared memory on NVIDIA GPUs. It is explicitly managed — the compiler (XLA/Mosaic) decides what data is in VMEM at each point in execution. VMEM is per-TensorCore.

Approximate VMEM capacities:
- **TPU v4**: ~16 MB per TensorCore
- **TPU v5e / v5p**: ~16–32 MB per TensorCore
- **TPU v6e (Trillium)**: ~32 MB per TensorCore
- **TPU v7 (Ironwood)**: Larger VMEM to match expanded 256×256 MXU bandwidth demands

VMEM bandwidth to the MXU is extremely high (order of TB/s), which is why keeping frequently accessed tensors in VMEM rather than HBM is the dominant performance optimization.

### 3.2 CMEM (Control Memory / Scalar Memory)

CMEM is a small, fast SRAM bank used for scalar operands:
- Loop counters and iteration variables
- DMA address descriptors and metadata
- Small lookup tables (e.g., activation function LUTs)
- ICI communication control registers

CMEM is separate from VMEM to avoid bandwidth contention: vectorized tile accesses and scalar control-flow reads do not compete for the same SRAM ports.

### 3.3 Unified Buffer (Early TPUs)

TPU v1–v3 used a **Unified Buffer (UB)** — a 24–32 MB SRAM pool shared between the weight FIFO, activation staging, and output accumulation. Starting from v4, this was reorganized into VMEM + CMEM with more explicit memory space semantics.

---

## 4. Off-chip Memory

### 4.1 HBM Generations and Capacities

TPU off-chip memory has used HBM since v2, with progressively larger stacks:

| Generation | HBM Gen | Capacity (per chip) | Bandwidth (per chip) | Notes |
|---|---|---|---|---|
| TPU v1 | GDDR5 | 8 GB | 34 GB/s | Inference-only |
| TPU v2 | HBM | 16 GB | 700 GB/s | Training capable |
| TPU v3 | HBM2 | 32 GB | 900 GB/s | |
| TPU v4 | HBM2e | 32 GB | ~1.2 TB/s | |
| TPU v5e | HBM2e | 16 GB | 819 GB/s | Efficiency variant |
| TPU v5p | HBM2e | 95 GB | 2,765 GB/s | Training flagship |
| TPU v6e (Trillium) | HBM3 | 144 GB | ~1.6 TB/s | 2× HBM capacity vs v5e |
| TPU v7 (Ironwood) | HBM3e | 192 GB | 7.4 TB/s | 8 HBM3e stacks; 1.77 PB per 9,216-chip superpod |

**TPU v7 (Ironwood) memory scale:**
- 192 GB HBM3e per chip (8 stacks)
- 7.4 TB/s peak HBM bandwidth per chip
- 9,216-chip superpod: 1.77 PB total HBM, ~68 TB/s aggregate HBM bandwidth

This places Ironwood's per-chip HBM capacity on par with NVIDIA B200 (192 GB HBM3e) with slightly lower bandwidth (7.4 TB/s vs 8.0 TB/s for B200), but at a 9,216-chip pod scale rather than per-node.

### 4.2 HBM Access Patterns

The weight-stationary MXU architecture creates a distinct HBM access profile compared to NVIDIA GPUs:
- **Weights**: Loaded once into VMEM per forward/backward pass layer; reused for all activations in the batch → low HBM read amplification for weights
- **Activations**: Streamed from HBM one tile at a time; large activations dominate HBM bandwidth for large-batch training
- **KV cache (inference)**: Large KV caches for long-context inference stress HBM capacity; Ironwood's 192 GB enables longer context windows without offloading

---

## 5. Host Interface

### 5.1 TPU VM Architecture

Google Cloud TPUs are accessed via **TPU VMs** — virtual machines that run directly on the TPU host board (not as a paravirtualized PCIe device). The TPU VM architecture:

```
User Code (Python / JAX)
      |
      v
libtpu.so (XLA compiler + TPU driver + ICI runtime)
      |
      v  [kernel driver]
TPU hardware (via PCIe or on-board interconnect)
```

The TPU is an on-board PCIe device relative to the host CPU in the VM. However, unlike cloud GPU instances where the GPU is a peripheral to a general-purpose server, the TPU VM host CPU is co-located on the TPU board, providing very low PCIe latency.

### 5.2 libtpu and Host Communication

`libtpu.so` is the monolithic host-side library that bundles:
- **XLA compiler**: Full ahead-of-time (AOT) and just-in-time (JIT) compilation to TPU binaries
- **TPU kernel driver**: Kernel-mode driver interface for DMA, interrupt, and command queue management
- **ICI runtime logic**: AllReduce, AllGather, AllToAll coordination across the ICI topology
- **Memory management**: HBM buffer allocation, pinned host memory for zero-copy transfers

Host-to-TPU data transfer for model loading:
1. JAX / libtpu allocates a pinned host buffer and copies model parameters from Python.
2. libtpu issues a DMA command to transfer the pinned buffer to HBM via PCIe.
3. The TPU acknowledges DMA completion via interrupt.
4. XLA-compiled binary execution begins; the host CPU polls for completion via a memory-mapped status register.

### 5.3 PCIe Interface

TPU v4+ chips interface to the host via PCIe Gen 4 x16 (~64 GB/s bidirectional). This is substantially slower than the chip's HBM bandwidth, meaning model parameter loading is a one-time cost (amortized over many forward passes) while inference/training compute operates entirely from HBM without returning to the host.

---

## 6. Scale-up Interconnect (ICI)

### 6.1 ICI (Inter-Chip Interconnect)

ICI is Google's proprietary high-speed chip-to-chip interconnect, used to build TPU pods where all chips share a unified high-bandwidth, low-latency fabric for collective communication. ICI is strictly for intra-pod communication; inter-pod communication uses DCN.

**ICI specifications by generation:**

| Generation | Topology | Per-chip BW | Max pod size |
|---|---|---|---|
| TPU v4 | 3D torus + OCS | ~600 Gbps | 4,096 chips |
| TPU v5e | 2D torus | — | 256 chips |
| TPU v5p | 3D torus | 4,800 Gbps | 8,960 chips |
| TPU v6e (Trillium) | 2D torus | ~4,800 Gbps (2× v5e) | 256 chips |
| TPU v7 (Ironwood) | 3D torus | 9,600 Gbps (1.2 TB/s) | 9,216 chips |

### 6.2 3D Torus Topology (v4, v5p, v7)

The 3D torus is Google's preferred high-bandwidth topology for large training pods. Each chip connects to 6 immediate neighbors (±1 in each of X, Y, Z dimensions) with wraparound links, forming a toroidal surface in 3D space.

**Structural unit: 4×4×4 cube**
The basic building block is a **64-chip 4×4×4 cube**:
- Copper DAC (Direct Attach Copper) cables connect chips within the same cube (short reach, low cost)
- Optical links (via OCS) connect cubes to other cubes (longer reach for torus wraparound)

For TPU v7 Ironwood:
- 64 chips per rack (4×4×4 cube)
- Intra-cube: copper DAC
- Inter-cube: optical fiber
- 9,216 chips = 144 racks arranged in a 3D torus

**Bandwidth on v7**:
- 200 GB/s per axis per chip (bidirectional)
- 4 ICI links per chip
- 9.6 Tbps aggregate bidirectional per chip

### 6.3 Optical Circuit Switching (OCS) — TPU v4

TPU v4 introduced **Optical Circuit Switches (OCS)** to make the 3D torus ICI physically realizable at pod scale. Each OCS is a MEMS-based optical switch (136×136 port, 128 active + 8 spares) that steers optical beams between fiber paths:

- OCS act as a programmable patch panel for the torus topology
- Configuration: 48 OCS units connect 48 pairs of 4×4×4 blocks, yielding 4,096 chips in a 3D torus
- **Reconfigurability**: OCS topology can be dynamically reconfigured, allowing the same 4,096 chips to present as different 3D torus shapes for different job sizes
- **Power advantage**: OCS eliminate electrical-to-optical conversion at each hop, reducing switching power by ~40% vs equivalent electrical switches
- **Twisted topologies**: OCS reconfigurability enables "twisted" torus configurations that increase bisection bandwidth by up to 70% for certain collective patterns

### 6.4 2D Torus Topology (v5e, v6e)

TPU v5e and v6e (Trillium) use a simpler 2D torus topology:
- Each chip connects to 4 neighbors (±X, ±Y with wraparound)
- Pod: 16×16 grid = 256 chips maximum
- Lower network diameter than 3D torus at this scale; simpler topology reduces routing complexity
- Trillium: ICI bandwidth more than doubled vs v5e, supporting the larger MXU throughput

---

## 7. Scale-out Interconnect

### 7.1 Multislice: DCN-Based Multi-Pod Training

For training scales beyond a single ICI pod, Google uses **Multislice** — connecting multiple TPU pods (slices) via a standard datacenter network (DCN). In this topology:

- **ICI** handles all-to-all communication within each pod (tight collective bandwidth)
- **DCN** handles gradient synchronization between pods (data-parallel AllReduce)
- **Titanium IPUs** (network offload processors) accelerate the DCN collectives

Multislice architecture supports:
- Up to 16 ICI pods per DCN aggregation block
- 4 aggregation blocks × 4 ICI pods each = up to 147,456 chips in a full deployment (v7 scale)
- DCN provides petabit/s aggregate bandwidth across pods

### 7.2 Titanium IPUs

Titanium is Google's in-house **Infrastructure Processing Unit** — a SmartNIC/DPU that offloads network functions from the host CPU:

- Accelerates DCN AllReduce for multi-pod gradient synchronization
- Handles RDMA-style data movement between pods without burdening the host CPU
- Runs software-defined network (SDN) protocols for topology management
- Enables the "AI Hypercomputer" abstraction: TPU pods + Titanium IPUs + GCS storage presented as a unified programmable infrastructure

### 7.3 Scale Comparison

| Scale | Configuration | Chips | Peak FP8 |
|---|---|---|---|
| Single chip | v7 Ironwood | 1 | 4,614 TFLOPS |
| Pod / Superpod | v7 9,216-chip | 9,216 | 42.5 ExaFLOPS |
| Multislice (max) | 16 pods | 147,456 | ~680 ExaFLOPS (theoretical) |

---

## Key Findings

### How Hardware Architecture Drives the Software Stack

1. **Systolic array determinism → compiler-centric model**: TPU's fully static VLIW execution means there is no runtime dynamic scheduling. The XLA compiler must produce a complete instruction schedule at compile time. This is why JAX's `jax.jit` compilation is not optional on TPU — it is the only execution model.

2. **MXU size jump (128×128 → 256×256) → recompilation required**: Trillium's 4× MACs/cycle improvement is not free. Code compiled for v5p's 128×128 MXU produces half-width tiles on v6e; XLA must recompile with 256×256 tile shapes to achieve peak utilization. This is unlike NVIDIA GPUs where PTX code runs (suboptimally) on newer architectures.

3. **Weight-stationary dataflow → HBM bandwidth efficiency**: For large-batch inference, TPU's weight-stationary design dramatically reduces HBM weight reads — weights are loaded once per layer per batch step, not once per sequence. This is TPU's fundamental advantage over output-stationary or flow-through GPU architectures for inference.

4. **VMEM as the performance bottleneck**: Unlike NVIDIA's L1/L2 cache hierarchy, TPU's VMEM is explicitly managed with no cache. An XLA kernel that cannot fit its working set in VMEM causes HBM spills, which are catastrophic for performance (100× bandwidth reduction). Operator fusion exists primarily to keep intermediate tensors in VMEM.

5. **ICI 3D torus → AllReduce topology match**: GSPMD/Shardy's collective insertion is topology-aware. On a 3D torus, ring AllReduce along each axis can achieve near-optimal bandwidth utilization by exploiting bidirectional ICI links simultaneously. The topology shapes which parallelism strategies are efficient.

6. **OCS reconfigurability → flexible parallelism dimensions**: TPU v4's OCS allows the 4,096-chip pod to present as different logical 3D grid shapes (e.g., 16×16×16 or 8×16×32). This lets GSPMD/Shardy choose pipeline, tensor, and data parallelism dimension splits that match the physical topology, maximizing ICI bandwidth utilization for each model.

7. **SparseCore → recommendation systems on TPU**: Without SparseCore, embedding lookup (gather) on TPU would bottleneck on the MXU which cannot accelerate sparse, irregular access patterns. SparseCore enables production-scale recommendation systems (>1T parameter embedding tables) on TPU, beyond pure transformer workloads.

8. **FP8 on v7 → inference era design**: Ironwood is the first TPU with MXU/TensorCore FP8 support, explicitly targeting inference workloads. Combined with 192 GB HBM (large KV caches) and 4 SparseCores, v7 is designed for the retrieval-augmented generation (RAG) and recommendation-heavy workloads of Google's production AI serving infrastructure.

---

## Sources

- [TPU v7 (Ironwood) Documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Ironwood TPU Announcement Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/)
- [Inside the Ironwood TPU Codesigned AI Stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [TPU v5p Documentation](https://docs.cloud.google.com/tpu/docs/v5p)
- [TPU v6e (Trillium) Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [TPU v4 Documentation](https://docs.cloud.google.com/tpu/docs/v4)
- [TPU Architecture Overview — Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [TPU v4 Whitepaper — arXiv:2304.01433 (ISCA 2023)](https://arxiv.org/abs/2304.01433)
- [Google TPU Architecture: 7 Generations Explained — Introl Blog](https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations)
- [Introducing Trillium (TPU v6e) — Google Cloud Blog](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [OCS Architecture — FiberMall](https://www.fibermall.com/blog/unveiling-google-tpu-architecture.htm)
- [How to Think About TPUs — JAX Scaling Book](https://jax-ml.github.io/scaling-book/tpus/)
- [TPU Deep Dive — Henry Ko](https://henryhmko.github.io/posts/tpu/tpu.html)
- [SemiAnalysis: TPUv7 Analysis](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)
- [Tensor Processing Unit — Wikipedia](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [Multislice Training — Google Cloud Documentation](https://docs.cloud.google.com/tpu/docs/v5e-training)
- [AI Hypercomputer Overview — Google Cloud Blog](https://cloud.google.com/blog/products/compute/updates-to-ai-hypercomputer-software-stack/)

---

## Update — TPU v8: Sunfish (8t) and Zebrafish (8i) (2026-04-27)

### Announcement context

Google announced its eighth-generation TPU at Google Cloud Next '26 on 2026-04-22. Unlike all prior generations, which exposed a single chip family with optional memory- and pod-shape SKUs (e.g., v5e vs v5p, v6e vs v6e-a-side), v8 splits the TPU lineup into two architecturally distinct, pin-incompatible chips:

| Chip | Codename | Design partner | Process | Workload | Codesign focus |
|---|---|---|---|---|---|
| TPU v8t (8t) | Sunfish | Broadcom | TSMC 2nm | Pre-training, large-scale RL | Compute throughput, scale-up bandwidth, multi-pod scale-out |
| TPU v8i (8i) | Zebrafish | MediaTek | TSMC 2nm | Inference, post-training, agentic reasoning | On-chip SRAM, HBM bandwidth, low-latency collectives |

Both chips host on **Axion** (Google's in-house Arm Neoverse-V2 server CPU) rather than x86 hosts, completing the JAX-on-Arm transition Google previewed in 2025.

### Per-chip specifications

| Spec | TPU v7 (Ironwood) | TPU v8t (Sunfish) | TPU v8i (Zebrafish) |
|---|---|---|---|
| Process | TSMC N3P | TSMC 2nm | TSMC 2nm |
| TensorCores per chip | 2 | 2 (compute-balanced) | 2 (memory-balanced) |
| MXU size | 256×256 | 256×256 | 256×256 |
| Native data types | FP8, BF16, INT8 | FP4, FP8, BF16, INT8 | FP4, FP8, BF16, INT8 |
| Peak FP4 (chip) | — (no FP4) | ~12.6 PFLOPS | ~10.1 PFLOPS |
| Peak FP8 (chip) | 4,614 TFLOPS | ~6.3 PFLOPS | ~5.0 PFLOPS |
| HBM capacity | 192 GB HBM3e | 216 GB HBM3e | 288 GB HBM3e |
| HBM bandwidth | 7.4 TB/s | ~6.5 TB/s | 8.6 TB/s |
| On-chip SRAM (VMEM) | ~128 MB | 384 MB (3× v7) | 384 MB (3× v7) |
| ICI per-chip | 9.6 Tb/s (1.2 TB/s) | 19.2 Tb/s (2.4 TB/s) | 19.2 Tb/s (2.4 TB/s) |
| Scale-out NIC | Titanium IPU | Titanium IPU + 400 Gb/s scale-out | Titanium IPU |
| Perf/W vs v7 | 1.0× | up to 2× | up to 2× |
| Cooling | 3rd-gen liquid | 4th-gen liquid | 4th-gen liquid |

The chips share the same MXU geometry (256×256 weight-stationary systolic, identical to v6e/v7) and the same first-class FP4 + FP8 path, so the kernel compiler (Mosaic / Pallas) targets both with shared lowering rules. The split is at the **memory and interconnect** level, not the compute level.

### TPU v8t (Sunfish) — training-optimized

**Per-chip:** 12.6 PFLOPS FP4, 216 GB HBM3e, 384 MB on-chip SRAM. 19.2 Tb/s ICI bandwidth doubles Ironwood's 9.6 Tb/s and is allocated heavily to scale-up — the training workload's all-reduce and all-gather bandwidth requirement.

**Topology — 3D torus (continued).** Sunfish keeps the 3D torus topology of v4/v5p/v7 because training collectives (ring AllReduce along each torus axis) are bandwidth-bound, not latency-bound, and the torus's bisection bandwidth scales linearly with chip count. OCS-style optical reconfigurability is retained for logical-shape flexibility.

**Pod scale.** A single v8t superpod scales to **9,600 chips** (up from 9,216 on Ironwood) delivering **121 FP4 ExaFLOPS** of peak compute and **2 PB** of shared HBM. Per-chip peak (~12.6 FP4 PFLOPS) × 9,600 ≈ 121 ExaFLOPS confirms a near-flat-utilization claim under bandwidth-balanced workloads.

**Multi-pod / scale-out — Virgo.** Google's **Virgo Network** is a new optical scale-out fabric replacing the DCN path used by Multislice on prior generations. Virgo supports up to **134,000 chips in a single datacenter** and up to **~1,000,000 chips across multiple datacenters** in a single training fabric. Inter-pod AllReduce is offloaded to Titanium IPUs, but Virgo's optical core gives roughly an order of magnitude lower diameter than DCN-routed Multislice.

**Price/performance claim.** Google quotes **2.8× training price/performance** vs Ironwood, attributed to the combination of (a) FP4 hardware, (b) doubled scale-up bandwidth, (c) Virgo's lower scale-out latency, and (d) 2nm-process power efficiency.

### TPU v8i (Zebrafish) — inference- and agent-optimized

**Per-chip:** 10.1 PFLOPS FP4, **288 GB HBM3e** (1.5× v7), **8.6 TB/s** HBM bandwidth (1.16× v7), **384 MB** on-chip SRAM (3× v7). 19.2 Tb/s ICI matches v8t. Compared to v8t, v8i trades raw peak compute (10.1 vs 12.6 PFLOPS) for HBM capacity and HBM bandwidth — the parameters that determine inference throughput when KV caches and weights together exceed on-chip SRAM.

**Topology — Boardfly (new).** v8i replaces the 3D torus with a high-radix topology Google calls **Boardfly**: tightly-coupled fully-connected boards (each board is an all-to-all of its on-board chips) aggregated into board-of-boards groups. The result is a flat-diameter network — Google reports the diameter of a 1,024-chip pod drops from **16 hops (3D torus)** to **7 hops (Boardfly)**, a ~56% reduction.

The motivation is structural: agentic and reasoning workloads (chain-of-thought decoding, multi-turn RL rollouts, MoE routing) generate small, latency-bound all-to-all and tree-reduction patterns that are bandwidth-underutilized on a 3D torus and hop-count-limited on its long axis. Boardfly exchanges per-link bandwidth for shorter critical paths and uniform pair-wise latency.

**Pod scale.** A single v8i pod scales to **1,152 chips** delivering **~11.6 FP8 ExaFLOPS** and **331.8 TB** of total HBM. The chip count (1,152 = 24 × 48, or 9 × 128, or 18 × 64) is chosen to fit the Boardfly's high-radix structure rather than a torus's regular grid.

**Collectives Acceleration Engine (CAE).** A new on-chip block in v8i specifically targeting the autoregressive-decode bottleneck. During chain-of-thought generation, every decoded token triggers a small reduction (across model-parallel shards or expert-parallel groups) on a tiny tensor (typically a single hidden-state vector). On v7, this reduction round-tripped through the ICI-managed AllReduce path with millisecond-class latency. CAE handles these reductions in dedicated on-chip silicon, achieving up to **5× lower on-chip latency**, and exposes them to XLA as a new HLO op (`cae_reduce`, currently undocumented in OpenXLA — Google-internal pass).

**Price/performance claim.** Google quotes **1.8× inference price/performance** vs Ironwood, dominated by (a) the 3× SRAM keeping more of the model and KV cache on-chip, (b) Boardfly's lower decode-time collective latency, (c) CAE, and (d) the 2nm process.

### Implications for the programming model

1. **Same MXU + same kernels, different memory specs.** Pallas/Mosaic kernels target the same 256×256 MXU on both v8t and v8i. The compiler must, however, retune tile sizes: with 384 MB SRAM (3× v7) the optimal tile gets larger, and Mosaic's autotuner exposes a new generation tag (`tpu_8t`, `tpu_8i`) so that XLA can dispatch the right tile-size schedule.

2. **FP4 is now first-class.** Both chips natively execute FP4 GEMM (likely OCP MX-FP4 microscaling format, matching Microsoft Maia and NVIDIA Blackwell). XLA's quantization passes have a new `--xla_tpu_enable_fp4` flag (default on for v8). FP4 weight-stationary computation still uses BF16 accumulation in the MXU, mirroring the FP8 pipeline on v7.

3. **Topology-aware Shardy partitioning becomes asymmetric.** Shardy's collective-insertion pass on v7 assumed a 3D torus on every TPU. On v8 it must dispatch on chip type: v8t pods get 3D-torus-shaped logical meshes, v8i pods get Boardfly-shaped meshes (effectively flat 2D from the Shardy point of view, since the per-board all-to-all hides the on-board topology). User-level mesh axis names (`('batch', 'model', 'pipeline')`) stay unchanged, but the physical mapping diverges.

4. **CAE introduces a new HLO opcode pathway for v8i only.** XLA emits `cae_reduce` in place of certain small-tensor `AllReduce`/`ReduceScatter` instructions when the chip type is v8i; on v8t the same op lowers to the conventional ICI ring path. This is the first time the TPU stack has had chip-type-conditional HLO lowering — the sign that v8 is more than a refresh.

5. **Disaggregated training and serving.** With v8t and v8i pin-incompatible, Google operationally separates training pods and inference pods. Cross-pod weight transfer for newly-trained models uses Virgo's scale-out fabric (or external object storage on cooler paths). Continuous-learning loops that once ran on a single Ironwood pod now bridge a v8t pod (training) and a v8i pod (rollout/eval), connected via Virgo.

### Why two chips at all (vs one with options)

Three reasons emerge from the announcement materials:

- **Workload divergence.** Pre-training and inference now have orthogonal bottlenecks: pre-training is FLOP- and ICI-bandwidth-bound (huge dense matmuls, gradient AllReduce); decoding is HBM-bandwidth-, on-chip-SRAM-, and collective-latency-bound (tiny matmuls, KV reads, per-token reductions). A single die optimized for both would be Pareto-dominated by either specialist on its own axis.

- **Process economics at 2nm.** A single die that is both compute-heavy (v8t-class) and memory-/SRAM-heavy (v8i-class) would exceed reticle budgets at 2nm. By splitting, each chip stays within reticle and yields well — Sunfish and Zebrafish appear to be near-reticle-limit single dies rather than chiplet stacks.

- **Vendor strategy.** By contracting Broadcom for Sunfish and MediaTek for Zebrafish, Google distributes risk and exploits each vendor's IP strengths (Broadcom: high-radix optical, custom SerDes; MediaTek: dense-SRAM 2nm experience, mobile-class power efficiency). Both chips are TSMC-fabbed on the same 2nm node, but the design partners differ.

### Resolved generational table

| Generation | Codename | MXU | HBM | On-chip SRAM | ICI Topology | ICI/chip | Pod Scale |
|---|---|---|---|---|---|---|---|
| v4 | — | 128×128 | 32 GB | ~32 MB | 3D torus + OCS | — | 4,096 |
| v5e | — | 128×128 | 16 GB | ~32 MB | 2D torus | — | 256 |
| v5p | — | 128×128 | 95 GB | ~96 MB | 3D torus | — | 8,960 |
| v6e | Trillium | 256×256 | 144 GB | ~96 MB | 2D torus | — | 256 |
| v7 | Ironwood | 256×256 | 192 GB HBM3e | ~128 MB | 3D torus | 9.6 Tb/s | 9,216 |
| **v8t** | **Sunfish** | 256×256 | **216 GB HBM3e** | **384 MB** | **3D torus + Virgo scale-out** | **19.2 Tb/s** | **9,600** (1M multi-DC) |
| **v8i** | **Zebrafish** | 256×256 | **288 GB HBM3e** | **384 MB** | **Boardfly (high-radix)** | **19.2 Tb/s** | **1,152** |

### Sources (v8 update)

- [Our eighth generation TPUs: two chips for the agentic era — Google Blog (2026-04-22)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [TPU 8t and TPU 8i technical deep dive — Google Cloud Blog](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
- [AI infrastructure at Next '26 — Google Cloud Blog](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26)
- [Google TPU 8i for Inference and TPU 8t for Training Announced — ServeTheHome](https://www.servethehome.com/google-tpu-8i-for-inference-and-tpu-8t-for-training-announced/)
- [Google dual tracks TPU 8 to conquer training and inference — The Register (2026-04-22)](https://www.theregister.com/2026/04/22/google_tpu8_dual_track_training_inference/)
- [Google splits TPUv8 strategy into two chips, Broadcom training and MediaTek inference — Wccftech](https://wccftech.com/google-splits-tpuv8-strategy-two-chips-broadcom-training-mediatek-inference-duties/)
- [Research: TPU 8t/8i and Virgo Network — fundaai (Substack)](https://fundaai.substack.com/p/researchtpu-8t8i-and-virgo-network)
- [Google launches training and inference TPUs — CNBC](https://www.cnbc.com/2026/04/22/google-launches-training-and-inference-tpus-in-latest-shot-at-nvidia.html)
- [Two new TPUs to power the next wave of AI training and inference — SiliconANGLE](https://siliconangle.com/2026/04/22/google-unveils-new-tpus-power-next-wave-ai-training-inference/)
- [Google Cloud Next 2026: TPU 8t and 8i architectures — Hyperframe Research](https://hyperframeresearch.com/2026/04/22/google-cloud-next-2026-google-cloud-bifurcates-the-ai-future-specialized-tpu-8t-and-8i-architectures-signal-the-end-of-general-purpose-silicon/)

---

## Dated Investigation — 2026-08-08: v8 spec corrections, no new silicon

**Window:** 2026-04-27 → 2026-08-08.
**Primary sources:** Google's *TPU 8t and TPU 8i technical deep dive* and *Our eighth generation TPUs* announcement (both 2026-04-22); Cloud TPU release notes and Cloud TPU system-architecture/supported-version page (both retrieved 2026-08-08).
**Headline:** no new TPU silicon in the window. v8t (Sunfish) / v8i (Zebrafish) remain the newest generation; there is no TPU v9 information and no v8 pricing. The material output of this pass is a set of corrections to v8 figures previously recorded in this repo, plus two v8 facts that had never been recorded.

### C1 — On-chip SRAM (Vmem): the repo was wrong for TPU 8t

Google's v8 spec table gives **On-Chip SRAM (Vmem) = 128 MB for TPU 8t and 384 MB for TPU 8i**. The "3× the previous generation" SRAM claim belongs to **8i alone**: the announcement post states it in exactly those terms — "TPU 8i pairs 288 GB of high-bandwidth memory with 384 MB of on-chip SRAM — 3x more than the previous generation" — and the Next '26 infrastructure post repeats the tripling for 8i only.

Internal-consistency check against the repo's own v7 figure of ~128 MB/chip: 384 MB is exactly 3× v7 for 8i, and 8t holds flat at v7's capacity. Both halves of the correction check out against a figure the repo already carried independently.

Earlier revisions recorded **384 MB/chip for both** v8 chips and further asserted **192 MB per TensorCore**. Google publishes v8 Vmem as a **per-chip** figure only; the per-TensorCore breakdown for v8t/v8i is **not disclosed**, and the 192 MB/TensorCore figure was an unsupported division.

Architecturally this is the cleanest expression of the v8 workload split. Decode is bound by on-chip capacity, KV-cache HBM reads, and collective latency — tripling Vmem on the inference chip directly buys resident context and is what makes KV-cache-resident decoding practical. Training is FLOP- and ICI-bandwidth-bound, so v8t spends the equivalent die area on compute and interconnect and takes no SRAM increase over v7.

### C2 — Training price/performance: 2.7×, not 2.8×

Google: "TPU 8t delivers up to **2.7x** performance-per-dollar improvement over Ironwood TPU for large-scale training." The repo said 2.8×. The inference figure previously recorded (1.8×) is correct — Google states "up to 80% performance-per-dollar improvement". Both chips: "up to 2x better performance-per-watt".

### C3 — v8i pod aggregate was mislabeled FP8

1,152 × 10.1 PFLOPS = 11.6 EFLOPS, but Google's 10.1 PFLOPS is **peak FP4**, not FP8. The pod figure is therefore ~11.6 **FP4** ExaFLOPS. (The v8t equivalent checks out unchanged: 9,600 × 12.6 PFLOPS = 121 FP4 EFLOPS, matching Google's "121 exaflops".)

### C4 — v8i pod size: both numbers are Google's

Google gives both. Boardfly connects "up to 1,152 of these chips together", while a pod is described as "up to **1,024 active** chips" (36 groups of 8 boards; 7-hop worst-case diameter vs 16 on a 3D torus of the same size). Keep 1,152 as the physical/aggregate figure — it is what yields the 331.8 TB pod HBM total at 288 GB/chip — and record 1,024 as the active count alongside it.

### C5 — Specialized-feature split: the repo overstated SparseCore on v8i

Google's spec table lists TPU 8t's specialized features as "SparseCore (Embeddings) **& LLM Decoder Engine**" and TPU 8i's as "**CAE** (Collectives Acceleration Engine)". The repo credited **4 SparseCores to both** v8 chips. Google attributes SparseCore only to 8t and publishes **no per-chip SparseCore count for v8**; the "4" was carried over from v7 without a source. The **LLM Decoder Engine** on 8t is a named block that the repo had never recorded — Google gives no functional description, count, or performance figure for it.

### C6 — v8i is a chiplet design

Google: "For each TPU 8i chip, there are two Tensor Cores (TC) on-core dies and one CAE on the chiplet die." v8i is a multi-die part with the collective engine on a separate die — the first TPU generation for which Google describes a chiplet organization. This explains how the CAE was added without displacing TensorCore or SRAM area on the compute dies. The repo did not previously say this.

### C7 — HBM bandwidth: exact figures

**6,528 GB/s (8t)** and **8,601 GB/s (8i)**. The repo's "~6.5 TB/s" and "8.6 TB/s" round correctly; the exact GB/s figures now appear in the hardware tables.

### C8 — Fab and partner attributions must be hedged

**None** of Google's three Next '26 posts mentions a process node, TSMC, Broadcom, or MediaTek; the announcement credits design work "in partnership with Google DeepMind". The repo asserted "Both v8 chips fabbed on TSMC 2nm; v8t designed with Broadcom; v8i designed with MediaTek" as fact, sourced to a single Wccftech piece. These are now recorded as **press-reported and unconfirmed by Google**.

### C9 — Availability: announced, GA promised for 2026, not yet shipping

Google's announcement post states verbatim: "**Both chips will be generally available later this year**, and can be used as part of Google's AI Hypercomputer." The Next '26 post says "TPU 8t and TPU 8i will be available to Cloud customers soon." As of 2026-08-08 neither chip appears in the Cloud TPU supported-version list (still topping out at TPU7x / Ironwood) and no v8 availability entry appears in the release notes.

Correct status verb: **announced, vendor-stated GA target of calendar 2026, not yet available to customers.** Explicitly rejected: that v8 is in a customer "preview" (the only "preview" in Google's deep-dive is native PyTorch support on TPU), that Google's blog states no date, and that external availability is targeted for late 2027 (that claim traces to a The Next Platform URL returning HTTP 404 and is not recorded).

### Forward-looking — Hot Chips 38 (not evidence)

HC38 runs **Aug 24–25, 2026** at Stanford. Session AI 2 (Tue 2026-08-25, 4:45–6:15 PM PDT, chair Brucek Khailany) lists "The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era" — Norman Jouppi & Sridhar Lakshmanamurthy, Google. **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** No slides or abstracts posted. Recorded only as a venue to re-scan after 2026-08-25; it must not be cited as the source of any specification. It is expected to be the first detailed architectural disclosure for v8t/v8i beyond the Cloud Next '26 material.

### Explicitly not confirmed in this window

- Per-TensorCore Vmem split for v8t/v8i.
- SparseCore counts for v8 (either chip).
- MXU dimensions for v8 — the deep-dive never states 256×256; the repo's 256×256 is a carried-forward assumption.
- FP8 support on v8 — the deep-dive discusses FP4 only.
- Process node, foundry, and ASIC design partners for v8 (vendor-side).
- v8 pricing; any TPU v9 information; any Meta TPU agreement beyond pre-baseline press reports.
- Any TPU MLPerf Training v6.0 result. Google is one of 24 submitting organizations (results published 2026-06-16; new benchmarks DeepSeek V3 671B and GPT-OSS 20B), but MLCommons does not attribute hardware to individual submitters. **No TPU MLPerf number is recorded.**

### Scope note — commercial capacity

Anthropic / Google / Broadcom announced (2026-04-06, anthropic.com; CFO Krishna Rao) multiple gigawatts of next-generation TPU capacity coming online **starting 2027**, vast majority sited in the US, building on the October 2025 up-to-1M-TPU agreement. Verified, but it predates this chip's 2026-04-27 refresh baseline and is a commercial-capacity fact rather than a hardware fact.

### Corrected v8 rows

| Generation | Codename | MXU | HBM | HBM BW | On-chip SRAM (Vmem) | Specialized blocks | ICI Topology | Pod Scale | Peak FP4/chip |
|---|---|---|---|---|---|---|---|---|---|
| **v8t** | **Sunfish** | 256×256 (assumed) | 216 GB HBM3e | **6,528 GB/s** | **128 MB/chip** | SparseCore (count n/d) + **LLM Decoder Engine** | 3D torus + Virgo scale-out | 9,600 (134K single-DC, ~1M multi-DC) | **12.6 PFLOPS** |
| **v8i** | **Zebrafish** | 256×256 (assumed) | 288 GB HBM3e | **8,601 GB/s** | **384 MB/chip** | **CAE** (on separate chiplet die); no SparseCore attributed | Boardfly (high-radix) | **1,152 connected / 1,024 active** | **10.1 PFLOPS** |

### Sources (2026-08-08 pass)

- [TPU 8t and TPU 8i technical deep dive — Google Cloud Blog (2026-04-22)](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — primary 8t/8i spec table
- [Our eighth generation TPUs: two chips for the agentic era — Google Blog (2026-04-22)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — availability wording, 8i-only 3× SRAM claim
- [AI infrastructure at Next '26 — Google Cloud Blog (2026-04-22)](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26)
- [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes) — retrieved 2026-08-08
- [Cloud TPU system architecture / supported versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — retrieved 2026-08-08; newest documented generation is TPU7x (Ironwood)
- [Hot Chips 38 program](https://hotchips.org/program/conference/) — retrieved 2026-08-08
- [MLPerf Training v6.0 results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)

---

## Update — 2026-09-13: Hot Chips 38 disclosure

*Window: 2026-08-08 → 2026-09-13. Primary talk: "The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era" (Norman Jouppi, Sridhar Lakshmanamurthy), delivered Hot Chips 38, 2026-08-25 — the disclosure the 2026-08-08 pass flagged as scheduled. Source used: ServeTheHome's session writeup, https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/ (2026-08-25). This is a detailed secondary account of the talk; no slide deck or transcript was independently retrieved. WebSearch budget for this scan was largely consumed on the Cerebras and SambaNova legs, so verification here rests on this one article plus the pre-existing baseline.*

### A. HBM stack counts (new)

The talk states v8t (Sunfish) has **6 HBM stacks** and v8i (Zebrafish) has **8 HBM stacks**. The repo's off-chip memory table previously carried an unsourced estimate of "4 HBM stacks" for v8t (apparently half of Ironwood's 8); that estimate is now corrected. No stack count had previously been recorded for v8i at all.

### B. Superpod FP4 aggregate — now a directly-cited figure

The 2026-08-08 pass computed v8t's superpod aggregate as ~121 FP4 ExaFLOPS by multiplying Google's per-chip peak (12.6 PFLOPS) by the 9,600-chip superpod size, and cross-checked it against Google's own "121 exaflops" quote from the April 2026 material. The Hot Chips 38 talk restates the same figure directly ("121 EFLOPS of FP4 compute"), so this is now doubly-sourced rather than resting on one arithmetic check.

### C. Virgo network — bandwidth and topology detail (new)

Two facts not in any prior source for this repo:
- **47 Pb/s aggregate bandwidth** at Virgo's 134,000-chip single-datacenter scale.
- A **two-layer switching topology** for Virgo. The talk (per ServeTheHome) does not detail what each layer does.

### D. Physical rack/tray configuration (new)

- v8t: **300 racks per superpod, 4 TPUs per tray** → 300 × 32 chips/rack = 9,600 chips, consistent with the already-recorded superpod size.
- v8i: **8 trays of 4 TPUs per group, 36 groups** → 8 × 4 × 36 = 1,152 chips, consistent with the already-recorded pod size and a refinement of Google's April 2026 "36 groups of 8 boards" language (each board = a 4-chip tray).

### E. Perf/W — reconfirmed, not new

ServeTheHome quotes the talk as "around twice the perf-per-watt as the TPUv7 Ironwood" — this is v8t-specific language but is consistent with, and does not change, the "up to 2×" figure already recorded for both chips from the April 2026 announcement.

### F. Design process (new, minor)

Google states it used AI to help design the TPU 8t/8i chips, specifically for power and area optimization. No further detail (methodology, tooling, which design stages) is given in the available coverage.

### G. Availability — explicitly checked, nothing found

This scan specifically checked the Hot Chips 38 coverage for a GA/availability statement, because a "GA late 2027" figure for TPU 8i has been circulating in some secondary commentary (flagged for verification at the start of this pass). **The ServeTheHome article contains no availability language for either chip at all** — no GA date, no shipping timeline, nothing. The "late 2027" figure is therefore **not corroborated by this source** and is **not recorded**. This is consistent with the 2026-08-08 pass's finding that an "external availability late 2027" claim traced to a dead URL and was excluded. The only sourced availability figure remains Google's April 2026 statement ("both chips will be generally available later this year"). The Cloud TPU release notes and supported-version list were **not re-fetched** in this pass; their last-checked state (2026-08-08, no v8 entry) is carried forward unverified.

### H. Price/performance and specialized-block claims — no update

No new price/performance figures, SparseCore counts, MXU dimensions, FP8-on-v8 confirmation, or process-node/design-partner confirmation appear in the available Hot Chips 38 coverage. All "not disclosed" items from the 2026-08-08 pass (§ "Explicitly not confirmed in this window") remain not disclosed.

### Sources (2026-09-13 pass)

- [Google's TPUv8s for Training and Inference at Hot Chips 2026 — ServeTheHome (2026-08-25)](https://www.servethehome.com/googles-tpuv8s-for-training-and-inference-at-hot-chips-2026/) — primary source for this update: HBM stack counts, 121 EFLOPS FP4, Virgo 47 Pb/s + two-layer topology, rack/tray physical layout, perf/W reconfirmation, AI-assisted design note; explicitly checked and found no GA/availability statement
- [Anthropic / Google / Broadcom compute partnership (2026-04-06)](https://www.anthropic.com/news/google-broadcom-partnership-compute)
