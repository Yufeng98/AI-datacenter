# Google TPU — Hardware Architecture

*as_of: 2026-08-08*
*generations: v4 / v5e / v5p / v6e (Trillium) / v7 (Ironwood) / v8t (Sunfish) / v8i (Zebrafish)*

---

## Generation Overview

| Generation | Codename | MXU Size | MACs/chip/cycle | HBM | HBM BW | On-chip SRAM | ICI Topology | ICI/chip | Pod Scale | TC | SC | FP8 | FP4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | — | 128×128 | 16,384 | HBM2e 32 GB | ~1.2 TB/s | ~32 MB | 3D torus + OCS | ~600 Gbps | 4,096 | 1 | 1 | No | No |
| v5e | — | 128×128 | 16,384 | HBM2e 16 GB | 819 GB/s | ~32 MB | 2D torus | — | 256 | 1 | 0 | No | No |
| v5p | — | 128×128 | 16,384 | HBM2e 95 GB | 2,765 GB/s | ~96 MB | 3D torus | 4.8 Tbps | 8,960 | 1 | 1 | No | No |
| v6e | Trillium | 256×256 | 65,536 | HBM3 32 GB (per spec — see note) | ~1.6 TB/s | ~96 MB | 2D torus | ~4.8 Tbps | 256 | 1 | 2 | No | No |
| v7 | Ironwood | 256×256 | 131,072 | HBM3e 192 GB | 7.4 TB/s | ~128 MB | 3D torus | 9.6 Tbps | 9,216 | 2 | 4 | Yes | No |
| **v8t** | **Sunfish** | 256×256 † | 131,072 † | **HBM3e 216 GB** | **6,528 GB/s** | **128 MB** | **3D torus + Virgo scale-out** | **19.2 Tbps** | **9,600** | 2 | n/d ‡ | n/d ‡ | **Yes** |
| **v8i** | **Zebrafish** | 256×256 † | 131,072 † | **HBM3e 288 GB** | **8,601 GB/s** | **384 MB** | **Boardfly (high-radix)** | **19.2 Tbps** | **1,152 phys / 1,024 active** | 2 | — ‡ | n/d ‡ | **Yes** |

† *MXU dimensions and MACs/chip/cycle for v8 are **not stated** by Google; 256×256 is carried forward from v6e/v7 and is an assumption, not a disclosure.*
‡ *Google's v8 spec table lists specialized features as "SparseCore (Embeddings) & LLM Decoder Engine" for **8t** and "CAE (Collectives Acceleration Engine)" for **8i**. No SparseCore count is published for either v8 chip, and no SparseCore is attributed to 8i. FP8 support on v8 is **not disclosed** — Google's deep-dive discusses FP4 only; the FP8 path is assumed carried over from v7 but is unconfirmed.*

**v8 announcement:** Google Cloud Next '26 (2026-04-22). Hosted on Google Axion (Arm Neoverse-V2) servers, 4th-generation liquid cooling, up to 2× perf/W vs Ironwood. Google credits the design to work "in partnership with Google DeepMind" and discloses **no process node and no ASIC design partner**. Press reporting (Wccftech and others, unconfirmed by Google) attributes *Sunfish* to Broadcom, *Zebrafish* to MediaTek, and both to TSMC 2nm — recorded here as press-reported only.

**v8 SRAM asymmetry (corrected 2026-08-08).** Google's spec table gives On-Chip SRAM (Vmem) = **128 MB for TPU 8t** and **384 MB for TPU 8i**. The widely quoted "3× the previous generation" SRAM claim applies to **8i alone**; against v7's ~128 MB/chip, 8i triples the capacity and 8t holds flat. The repo previously recorded 384 MB for both chips — that was wrong for 8t.

**v8i packaging.** Google states: "For each TPU 8i chip, there are two Tensor Cores (TC) on-core dies and one CAE on the chiplet die" — v8i is a chiplet part with the CAE on a separate die.

**Per-chip peak (Google):** 12.6 PFLOPS peak FP4 (8t), 10.1 PFLOPS peak FP4 (8i).

**Availability (as of 2026-08-08):** announced; Google states "both chips will be generally available later this year" (calendar 2026); not yet available to customers, not in the Cloud TPU supported-version list, no release-notes entry. No customer preview program exists for v8 silicon.

---

## 1. Compute Engine

### TensorCore Architecture

Google's **TensorCore** is the fundamental compute die unit within a TPU chip — not to be confused with NVIDIA's "Tensor Core" functional unit. Each Google TensorCore is a complete processing element containing:

- **MXU (Matrix Multiply Unit)** — the weight-stationary systolic array
- **VPU (Vector Processing Unit)** — element-wise operations, executing concurrently with MXU
- **ScalarCore** — control flow, loop iteration, address generation
- **VMEM** — per-TensorCore on-chip SRAM scratchpad
- **CMEM** — scalar memory separate from VMEM

TPU v7 Ironwood is the first generation with **2 TensorCores per chip**, enabling parallel matrix pipeline execution. All previous generations have 1 TensorCore per chip.

### MXU — Weight-Stationary Systolic Array

The MXU is the defining structure of every TPU generation. It operates as a weight-stationary systolic array:

1. Weights are loaded into the systolic array cells and held stationary
2. Input activations stream through rows left-to-right, one row per clock cycle
3. Each cell multiplies its stationary weight by the passing activation and accumulates
4. Partial sums propagate downward through columns
5. Final accumulated outputs exit at the bottom of the array

This maximizes weight reuse per HBM load — each weight participates in multiple dot products without being re-fetched.

**MXU evolution:**

| Generations | Array Size | MACs/cycle (per TensorCore) | Key change |
|---|---|---|---|
| v2/v3 | 128×128 | 16,384 | Introduced BF16 (Google's invention for ML) |
| v4/v5e/v5p | 128×128 | 16,384 | INT8 + sparse embedding support |
| v6e (Trillium) | 256×256 | 65,536 | 4× MACs/cycle vs v5e at same clock; required XLA retiling |
| v7 (Ironwood) | 256×256 | 65,536/TC × 2 TC = 131,072/chip | Added FP8; 2 TensorCores/chip |
| v8t (Sunfish) | 256×256 (assumed — not disclosed) | 131,072/chip (assumed) | Added FP4 (OCP MX-FP4 microscaling); 12.6 PFLOPS peak FP4 per chip; process node not disclosed |
| v8i (Zebrafish) | 256×256 (assumed — not disclosed) | 131,072/chip (assumed) | Added FP4; 10.1 PFLOPS peak FP4 per chip; lower peak than v8t despite more SRAM/HBM (clock/area trade not disclosed) |

> Google's TPU 8t/8i deep-dive never states an MXU array size for v8. The 256×256 entries above are carried forward from v6e/v7 for continuity and are flagged as assumptions.

**Data type support:**
- v2/v3: BF16 (Google invented BF16 for ML training: 8-bit exponent, 7-bit mantissa)
- v4/v5: BF16, INT8, FP32 accumulation
- v6e: BF16, INT8, FP16, FP32 accumulation
- v7 (Ironwood): FP8, BF16, FP16, INT8, FP32 accumulation (FP8 is first-generation MXU hardware support)
- v8t / v8i: FP4 (OCP MX microscaling) confirmed; FP8/BF16/FP16/INT8/FP32-accumulation assumed carried over from v7 but **not disclosed** for v8 (Google's deep-dive discusses FP4 only). FP4 uses BF16 accumulation in the MXU; the FP4 path mirrors the FP8 dataflow added in v7.

### VPU — Vector Processing Unit

The VPU handles element-wise operations that the systolic MXU cannot efficiently express:
- Activation functions: ReLU, GELU, SiLU, sigmoid
- Softmax denominator computation (reduction over sequence dimension)
- Layer normalization (reduction + scale + shift)
- Residual connections (element-wise add)
- Transcendental functions (exp, log, rsqrt)

Critically, the VPU and MXU execute concurrently in separate VLIW slots. Well-structured kernels overlap activation computation (on VPU) with the next tile's matrix multiply (on MXU), achieving near-peak utilization of both functional units simultaneously.

### SparseCore

SparseCore is a dedicated on-chip accelerator for **sparse embedding table lookups** — the dominant operation in large-scale ranking and recommendation systems. The MXU's dense weight-stationary dataflow is poorly suited to the irregular random-access patterns of embedding lookups (given a table of 10B × 128-dim embeddings and a list of random row indices, fetch and aggregate the corresponding rows).

SparseCore evolution:
- v4: First generation (1 SparseCore/chip) — enabled production-scale recommendation systems on TPU
- v5p: SparseCore for 1T+ parameter embedding training
- v6e: 2nd-gen SparseCore (2/chip, improved bandwidth)
- v7 (Ironwood): 4 SparseCores/chip — expanded to accelerate financial, scientific, and graph sparse workloads beyond recommendation
- v8t (Sunfish): SparseCore present ("SparseCore (Embeddings)" in Google's spec table); **count not published**
- v8i (Zebrafish): **no SparseCore attributed by Google** — the chip's listed specialized feature is the CAE. The repo previously credited 4 SparseCores to both v8 chips; that is not supported by any primary source and has been withdrawn.

### LLM Decoder Engine (v8t)

Google's v8 spec table lists TPU 8t's specialized features as "SparseCore (Embeddings) **& LLM Decoder Engine**". The LLM Decoder Engine is a named on-chip block exclusive to the *training* chip; Google publishes **no functional description, no count, and no performance figure** for it. Recorded here as a named block only — its role (plausibly accelerating the decode/rollout phase of large-scale RL and post-training, which runs on the training chip) is **not disclosed**. This block was absent from earlier revisions of this document.

---

## 2. Data Path — VLIW Execution Model

TPU uses a **VLIW (Very Long Instruction Word)** execution model. Each instruction packet is issued every clock cycle and can schedule up to 8 independent operations simultaneously:

| VLIW Slot | Functional Unit | Example operation |
|---|---|---|
| 0 | MXU dispatch | Initiate systolic matrix multiply |
| 1 | MXU pipeline | Continue in-progress multiply |
| 2 | VPU element-wise | Apply GELU activation |
| 3 | VPU reduction | Compute softmax denominator |
| 4 | DMA read | HBM → VMEM tile prefetch |
| 5 | DMA write | VMEM → HBM output writeback |
| 6 | Scalar ALU | Increment loop counter |
| 7 | Scalar LD/ST | Load DMA descriptor from CMEM |

Because all instruction scheduling is static (compiler-determined by XLA and Mosaic), the TPU has no:
- Dynamic issue logic
- Branch predictor
- Out-of-order execution hardware
- Warp scheduler

This eliminates significant silicon area and power, enabling larger systolic arrays, more HBM capacity, and greater energy efficiency per FLOP compared to a general-purpose SIMT processor.

### Weight-Stationary Data Movement

The full data path for a GEMM tile:

```
Host RAM → PCIe (64 GB/s) → HBM [initialization, one-time cost]

HBM → DMA read (VLIW slot 4) → VMEM weight staging buffer
HBM → DMA read (VLIW slot 4) → VMEM activation buffer
      [double-buffered: tile N+1 prefetches while tile N computes]

VMEM → MXU systolic cells → weights loaded and held stationary
VMEM → MXU row feeds → activations stream through rows

MXU output → VMEM accumulator buffer
VMEM → DMA write (VLIW slot 5) → HBM [output tensors]
```

The DMA double-buffering pattern is the critical performance mechanism: Mosaic and XLA generate instruction schedules where the DMA read for weight tile N+1 occupies VLIW slot 4 while the MXU processes weight tile N in slots 0–1. This hides HBM latency (~100+ ns) behind continuous MXU computation.

---

## 3. On-chip Memory

### VMEM (Vector Memory / Scratchpad SRAM)

VMEM is the primary on-chip SRAM scratchpad, analogous to shared memory on NVIDIA GPUs. Key properties:

- **Explicitly compiler-managed**: No hardware cache. All data movement to/from VMEM is scheduled by DMA instructions compiled by XLA or Mosaic. There is no implicit cache fill or eviction.
- **Per-TensorCore**: Each TensorCore has its own VMEM; v7 has 2 separate VMEM banks (one per TensorCore).
- **Extremely high bandwidth to MXU**: VMEM bandwidth to the MXU systolic inputs is on the order of TB/s, matching the MXU's data consumption rate.
- **Capacity bottleneck**: VMEM capacity determines the maximum fused kernel size. Workloads that overflow VMEM spill intermediate tensors to HBM, incurring ~100× bandwidth penalty. XLA's operator fusion pass is bounded by this constraint.

Approximate VMEM capacities per TensorCore:
- v4: ~16 MB
- v5e / v5p: ~16–32 MB
- v6e (Trillium): ~32 MB
- v7 (Ironwood): ~64 MB per TensorCore (~128 MB per chip; sized to sustain 256×256 MXU bandwidth)
- v8t (Sunfish): **128 MB per chip** — flat against v7. Google publishes only a per-chip Vmem figure for v8; the **per-TensorCore split is not disclosed**.
- v8i (Zebrafish): **384 MB per chip** — 3× v7, and the largest single-generation SRAM jump in TPU history. Google attaches the "3× more than the previous generation" claim to 8i alone.

**Corrected 2026-08-08.** Earlier revisions of this document recorded 192 MB/TensorCore = 384 MB/chip for *both* v8 chips. Google's spec table gives On-Chip SRAM (Vmem) = 128 MB for TPU 8t and 384 MB for TPU 8i, and never publishes a per-TensorCore breakdown for v8.

The SRAM asymmetry is the clearest expression of the v8 workload split. The 3× jump is **v8i-only** and is what makes KV-cache-resident decoding practical on the inference chip: decode is bound by on-chip capacity and by HBM reads of the KV cache, so tripling Vmem directly buys resident context. v8t gains no SRAM over v7 — training is FLOP- and ICI-bandwidth-bound, so the die area goes to compute and interconnect instead. Software-side, the divergence means the Mosaic autotuner cannot emit one v8 tile-size schedule: `tpu_8t` and `tpu_8i` generation tags select materially different VMEM budgets (128 MB vs 384 MB per chip), where previously a single per-generation schedule sufficed.

### CMEM (Control Memory / Scalar Memory)

CMEM is a small, fast SRAM bank separate from VMEM, used for:
- Loop counters and iteration variables
- DMA address descriptors and transfer metadata
- Small lookup tables (activation function LUTs, quantization scales)
- ICI communication control registers and rendezvous state

CMEM is isolated from VMEM to avoid bandwidth contention: vectorized tile accesses (TB/s) and scalar control-flow reads (KB/s) do not compete for the same SRAM ports or address crossbars.

### No Cache Hierarchy

Unlike NVIDIA GPUs (which have L1 (SMEM+data cache), L2, register file) and AMD GPUs (LDS, L2, Infinity Cache), TPU has no hardware cache hierarchy. There is no L1 or L2 data cache. All on-chip data management is explicit and compiler-directed. This is a fundamental architectural choice: the TPU trades cache hardware complexity for larger VMEM capacity and simpler (area-efficient) chip design.

---

## 4. Off-chip Memory (HBM)

### HBM Generations by TPU Version

| Version | HBM Gen | Per-chip Capacity | Per-chip BW | Notes |
|---|---|---|---|---|
| v1 | GDDR5 | 8 GB | 34 GB/s | Inference-only; no training |
| v2 | HBM | 16 GB | 700 GB/s | First training-capable TPU |
| v3 | HBM2 | 32 GB | 900 GB/s | |
| v4 | HBM2e | 32 GB | ~1.2 TB/s | |
| v5e | HBM2e | 16 GB | 819 GB/s | Efficiency variant |
| v5p | HBM2e | 95 GB | 2,765 GB/s | Training flagship |
| v6e (Trillium) | HBM3 | 32 GB | ~1.6 TB/s | 2× capacity vs v5e |
| v7 (Ironwood) | HBM3e | 192 GB | 7.4 TB/s | 8 stacks; 1.77 PB/superpod |
| **v8t (Sunfish)** | **HBM3e** | **216 GB** | **6,528 GB/s** | Training-tuned: more compute per HBM byte than v8i; 4 HBM stacks |
| **v8i (Zebrafish)** | **HBM3e** | **288 GB** | **8,601 GB/s** | Inference-tuned: more HBM capacity *and* more HBM BW than v8t; sized for KV-cache-resident decoding (paired with 384 MB Vmem) |

### v7 Ironwood Superpod Memory Scale

- 192 GB HBM3e per chip (8 stacks)
- 7.4 TB/s peak HBM bandwidth per chip
- 9,216-chip superpod: **1.77 PB total HBM**, ~68 TB/s aggregate HBM bandwidth

### v8t Sunfish Superpod Memory Scale

- 216 GB HBM3e per chip
- 6,528 GB/s peak HBM bandwidth per chip
- 128 MB on-chip SRAM (Vmem) per chip — flat vs v7
- 9,600-chip superpod: **2 PB total HBM**, ~62 TB/s aggregate HBM bandwidth

### v8i Zebrafish Pod Memory Scale

- 288 GB HBM3e per chip
- 8,601 GB/s peak HBM bandwidth per chip
- 384 MB on-chip SRAM (Vmem) per chip — 3× v7
- 1,152-chip pod (physical/aggregate): **331.8 TB total HBM**, ~9.9 PB/s aggregate HBM bandwidth. Google separately describes a v8i pod as "up to **1,024 active** chips" (36 groups of 8 boards); 331.8 TB is the 1,152 × 288 GB figure.

### HBM Access Pattern (Weight-Stationary)

The weight-stationary MXU architecture creates a distinct HBM access profile:

- **Weights**: Loaded once per layer per forward-pass batch step into VMEM. Reused for all sequences/activations in the batch. For large-batch inference, weight HBM reads are amortized across many activations — fundamentally different from output-stationary or flow-through GPU architectures that re-load weights per token.
- **Activations**: Streamed from HBM in tiles for each layer. For large sequence lengths (long-context inference), activations and KV caches dominate HBM bandwidth.
- **KV cache (inference)**: Long-context inference (128K–1M tokens) requires large KV caches in HBM. v7's 192 GB capacity is sized to hold substantial KV caches for serving long-context models without offloading; v8i pushes this to 288 GB HBM plus 384 MB of on-chip Vmem, moving the hottest slice of the KV cache on-die.

---

## 5. Host Interface / Package

### TPU VM Architecture

Google Cloud TPUs are accessed via **TPU VMs**: virtual machines whose host CPU is co-located with the TPU on the same physical board. The user's Python/JAX process runs directly on this host VM.

```
User Process (Python + JAX)
      |
      v
libtpu.so (XLA compiler + TPU kernel driver + ICI runtime)
      |
      v  [PCIe Gen 4 x16, ~64 GB/s]
TPU ASIC (HBM, VMEM, MXU, ICI)
```

Unlike cloud GPU instances where the GPU is a peripheral in a datacenter server, the TPU VM host CPU is on the same board, providing lower PCIe latency for command submission and interrupt delivery.

### PCIe Interface

TPU v4+ uses PCIe Gen 4 x16 (~64 GB/s bidirectional). This is the model-loading path: at initialization, libtpu DMAs model parameters from host DRAM (or GCS via the network) across PCIe into HBM. During training and inference, the TPU operates entirely from HBM.

The 64 GB/s PCIe bandwidth is adequate for one-time model loading but would be a severe bottleneck for per-step data transfer. The TPU execution model — where a compiled binary runs for many training steps on a fixed HBM-resident model — avoids this bottleneck entirely.

---

## 6. Scale-up Interconnect (ICI)

### ICI Specifications

| Version | Topology | BW/chip | Max pod chips |
|---|---|---|---|
| v4 | 3D torus + OCS | ~600 Gbps | 4,096 |
| v5e | 2D torus | — | 256 |
| v5p | 3D torus | 4,800 Gbps | 8,960 |
| v6e (Trillium) | 2D torus | ~4,800 Gbps | 256 |
| v7 (Ironwood) | 3D torus | 9,600 Gbps (1.2 TB/s) | 9,216 |
| **v8t (Sunfish)** | **3D torus + Virgo scale-out** | **19.2 Tbps (2.4 TB/s)** | **9,600** (single fabric: 134K; multi-DC: ~1M) |
| **v8i (Zebrafish)** | **Boardfly (high-radix, fully-connected boards)** | **19.2 Tbps (2.4 TB/s)** | **1,152 physical / up to 1,024 active** |

### 3D Torus Topology (v4, v5p, v7)

Each chip connects to 6 immediate neighbors (±1 in X, Y, Z) with wraparound links. The **4×4×4 cube** (64 chips) is the physical building block:
- **Intra-cube links**: Copper DAC (Direct Attach Copper) — short reach, low cost, low latency
- **Inter-cube links**: Optical fiber — longer reach for torus wraparound and inter-rack connections

For v7 Ironwood: 64 chips per rack, 144 racks in 3D torus, 9,216 chips total. Each chip has 4 ICI links (200 GB/s per axis per chip bidirectional).

**Why 3D torus for AllReduce?**
Ring AllReduce along any one torus axis achieves near-peak bandwidth: all chips participate, all ICI links are in use, and the algorithm is pipelined. Three sequential ring AllReduces (one per axis) complete a full AllReduce with optimal bandwidth utilization. Shardy's collective insertion exploits this: it assigns different parallelism dimensions (batch, model, pipeline) to different torus axes to enable concurrent communication.

### OCS — Optical Circuit Switching (v4)

TPU v4 introduced **Optical Circuit Switches** to make the 3D torus physically realizable at 4,096-chip scale:

- Each OCS: 136×136 ports (128 active + 8 spares), MEMS-based optical beam steering
- 48 OCS units connect 48 pairs of 4×4×4 cubes
- **Reconfigurable**: OCS can be reprogrammed within ~10 ms to present different 3D torus shapes — a 16×16×16 cube, an 8×16×32 slab, or a "twisted" torus
- **Twisted torus**: A non-standard torus configuration enabled by OCS that increases bisection bandwidth by up to 70% for certain collective communication patterns (e.g., AllToAll-heavy pipeline parallelism)
- **Power advantage**: ~40% power reduction vs equivalent electrical cross-connect switches by eliminating E/O conversion at each hop

### 2D Torus Topology (v5e, v6e)

v5e and v6e use simpler 2D torus (16×16 = 256 chips max), with each chip having 4 neighbors (±X, ±Y). Lower maximum pod scale in exchange for simpler routing and lower cost per chip for efficiency-optimized workloads.

### Boardfly Topology (v8i)

TPU v8i replaces the 3D torus with **Boardfly**, a high-radix topology designed for the small, latency-bound collective patterns of agentic and reasoning workloads (chain-of-thought decoding, multi-turn RL rollouts, MoE routing).

**Construction.** Chips are organized into *boards* with full mesh (all-to-all) connectivity inside each board. Boards are then aggregated into board-of-boards groups via a high-radix inter-board switching layer. Google describes a pod as **36 groups of 8 boards**, "up to 1,024 active chips", with Boardfly connecting "up to 1,152 of these chips together". The result is a flat-diameter network: at the 1,024-chip scale Google quotes a worst-case diameter of **7 hops** vs **16 hops** on a 3D torus of the same size — a ~56% reduction in network diameter. Read 1,152 as the physical/aggregate chip count (it is the figure that yields the 331.8 TB pod HBM total at 288 GB/chip) and 1,024 as the active count.

**Why not 3D torus on the inference chip?** Decode-time collectives are *small-tensor, latency-bound*. A single decoded token triggers a per-head reduction across model-parallel shards on a tensor of a few KB. On a 3D torus, this reduction's latency is dominated by hop count (each hop adds a switch crossing and SerDes traversal latency), not bandwidth. Boardfly trades the torus's bandwidth-optimal property (ring AllReduce saturates every link along an axis) for fewer hops on the critical path of every decoded token.

**Topology asymmetry between v8t and v8i.** XLA's Shardy partitioner now dispatches on chip type. On v8t (3D torus) it generates the same axis-aligned collective decomposition used since v4. On v8i (Boardfly) it treats each board as a single all-to-all node, then handles inter-board communication separately. The user-visible mesh axis names (`('batch', 'model', 'pipeline')`) stay the same, but the physical mapping diverges.

### Collectives Acceleration Engine — CAE (v8i only)

CAE is a new on-chip block introduced in v8i to accelerate the small-tensor reductions that dominate autoregressive decoding latency. Mechanism:

- During chain-of-thought generation, every decoded token requires a small reduction (typically a per-head attention-output sum across model-parallel or expert-parallel shards) on a tensor of a few KB.
- On v7, this reduction round-trips through the ICI-managed AllReduce path with millisecond-class latency dominated by ICI control-plane and inter-board hops.
- CAE handles these reductions in dedicated on-chip silicon: a hardware tree-reduce engine with dedicated SRAM buffers, exposed to XLA as a new HLO op (`cae_reduce`). Up to **5× lower on-chip collective latency** vs the equivalent ICI AllReduce.
- The XLA TPU backend lowers small-tensor `AllReduce`/`ReduceScatter` to `cae_reduce` automatically when the chip type is v8i; on v8t the same op continues to lower to ICI ring AllReduce.

CAE is the first instance in TPU history of a chip-type-conditional HLO lowering rule, indicating that v8 is more than a simple shrink and refresh.

**CAE lives on its own die.** Google states that "for each TPU 8i chip, there are two Tensor Cores (TC) on-core dies and one CAE on the chiplet die" — v8i is a **chiplet design**, with the collective engine physically separated from the TensorCore dies. This is the first TPU generation for which Google describes a multi-die chip organization, and it explains why the CAE could be added to the inference chip without displacing TensorCore or SRAM area on the compute dies.

---

## 7. Scale-out Interconnect (Multislice + Titanium + Virgo)

### Multislice

**Multislice** is Google's multi-pod training topology:
- Each ICI pod (slice) communicates internally via ICI (high bandwidth, low latency)
- Inter-pod communication for gradient AllReduce uses the standard Datacenter Network (DCN)
- XLA schedules inter-pod DCN AllReduce during periods when intra-pod MXU compute can proceed concurrently, hiding DCN latency

Scale:
- Up to 16 ICI pods per DCN aggregation block
- Up to 4 aggregation blocks per full Multislice deployment
- Maximum theoretical scale: 147,456 chips (v7 generation)

### Titanium IPU

**Titanium** is Google's in-house Infrastructure Processing Unit (IPU) — a SmartNIC/DPU that offloads network functions:
- Accelerates DCN AllReduce for inter-pod gradient synchronization without burdening the host CPU
- Handles RDMA-style data movement between pods
- Runs Google's software-defined network (SDN) protocols for topology management
- Enables the **AI Hypercomputer** abstraction: TPU pods + Titanium IPUs + GCS storage as unified programmable AI infrastructure

### Virgo Network (v8t scale-out)

**Virgo** is Google's new optical scale-out fabric introduced with v8t, replacing the DCN-routed Multislice path for training-scale workloads. Properties:

- Optical core with high-radix optical switches; lower diameter than DCN-routed Multislice.
- Connects v8t pods (9,600 chips each) into a unified scale-out fabric.
- **Single-datacenter scale**: up to **134,000 chips** in a single Virgo fabric.
- **Multi-datacenter scale**: up to **~1,000,000 chips** in a single training cluster spanning multiple datacenters (the "Multi-DC training" envelope Google previewed at I/O 2024 with the Pathways stack).
- Inter-pod AllReduce is offloaded to Titanium IPUs; XLA collective insertion treats Virgo as a high-bandwidth, moderate-latency outermost mesh axis.

Virgo is the largest single-step scale-out increase in TPU history (Multislice on v7 topped out at 147,456 chips; Virgo extends that by ~7× in single-DC and ~7× again across DCs).

---

## Hardware-Software Interface Summary

| Hardware Feature | Software Mechanism | Exposed via |
|---|---|---|
| MXU 128×128 | XLA tiles GEMM to 128-wide K chunks | XLA compiler (automatic) |
| MXU 256×256 (v6e+) | XLA retiles to 256-wide blocks; 4× MACs/cycle | XLA recompilation required |
| VMEM scratchpad | XLA memory space assignment; Mosaic BlockSpec | XLA fusion pass + Pallas |
| CMEM scalar memory | Mosaic places loop indices and DMA metadata in CMEM | Mosaic (automatic) |
| VLIW 8-op/cycle | Mosaic / XLA TPU backend packs MXU+VPU+DMA+scalar | Compiler (not user-visible) |
| DMA double-buffering | Mosaic N-stage prefetch pipeline | Pallas pipeline_stages param |
| ICI 3D torus AllReduce | Shardy inserts lax.psum → AllReduce along torus axis | NamedSharding mesh axes |
| OCS topology (v4) | GSPMD/Shardy selects optimal parallelism dimension splits | Shardy mesh axis mapping |
| FP8 (v7) | jax.numpy BF16→FP8 cast ops; Tokamax FP8 kernels | JAX dtype + Pallas |
| SparseCore | TPU embedding ops (jax.lax.gather via SparseCore path) | JAX + libtpu (automatic) |
| FP4 (v8t/v8i) | OCP MX-FP4 microscaling; XLA `--xla_tpu_enable_fp4` pass; FP4 GEMM in MXU with BF16 accumulation | JAX dtype + Pallas (jax.numpy.float4_e2m1 / e3m0) |
| 384 MB on-chip SRAM (v8i) / 128 MB (v8t) | Mosaic autotuner emits a different tile-size schedule per generation tag (`tpu_8t` = 128 MB budget, `tpu_8i` = 384 MB budget) | Mosaic / Pallas |
| LLM Decoder Engine (v8t) | No published software interface — function not detailed by Google | Not disclosed |
| Chiplet packaging (v8i: 2 TensorCore dies + 1 CAE die) | Transparent to XLA; CAE reached via the `cae_reduce` lowering path | Compiler-only |
| Boardfly topology (v8i) | Shardy lowers Boardfly mesh axes to per-board all-to-all + inter-board reductions | NamedSharding mesh axes (chip-type dispatch) |
| CAE — Collectives Acceleration Engine (v8i) | XLA lowers small-tensor `AllReduce`/`ReduceScatter` to `cae_reduce` HLO op | Compiler-only (chip-type conditional) |
| Virgo Network (v8t scale-out) | Outermost mesh axis in Shardy; Titanium IPU offload for cross-pod reductions | NamedSharding mesh axes |

---

## Update — 2026-08-08 (v8 spec corrections; no new silicon)

*Window: 2026-04-27 → 2026-08-08. Prior-generation content above is unchanged.*

No new TPU silicon was announced in this window; v8t/v8i remain the newest generation and there is no TPU v9 information. This revision corrects v8 figures the document already carried and adds two previously unrecorded v8 facts. Changes applied above:

| § | Change |
|---|---|
| Generation Overview | v8t on-chip SRAM 384 MB → **128 MB**; v8i stays 384 MB. HBM BW to exact **6,528 / 8,601 GB/s**. v8i pod scale annotated **1,152 physical / 1,024 active**. SparseCore column marked not-published (v8t) / not-attributed (v8i); FP8 column marked not-disclosed for v8. TSMC 2nm / Broadcom / MediaTek demoted to press-reported. |
| 1. Compute Engine | MXU 256×256 for v8 flagged as an assumption (never stated by Google). Data-type list hedged: FP4 confirmed, FP8 and the rest not disclosed for v8. SparseCore evolution extended with the v8 split. New **LLM Decoder Engine (v8t)** subsection. |
| 3. On-chip Memory | v8 Vmem rewritten: 128 MB/chip (8t), 384 MB/chip (8i), per-TensorCore split not disclosed; the 3× jump is v8i-only and is the KV-cache-resident-decoding enabler. |
| 4. Off-chip Memory | Exact per-chip HBM bandwidths; v8i pod memory scale annotated with the 1,024-active figure. |
| 6. Scale-up Interconnect | Boardfly construction detail (36 groups of 8 boards; 1,152 connected / 1,024 active). New note: **v8i is a chiplet part** — 2 TensorCore dies + 1 CAE die. |
| HW–SW Interface | SRAM row split per chip; LLM Decoder Engine and chiplet packaging rows added. |

**Not disclosed / withdrawn:** per-TensorCore Vmem split for v8; SparseCore counts for v8; MXU dimensions for v8; FP8 support on v8; process node and ASIC design partners for v8; v8 pricing.

**Scheduled disclosure — do not cite as a source for any spec.** Hot Chips 38 (Stanford, Aug 24–25 2026), session AI 2, Tue 2026-08-25 4:45–6:15 PM PDT, chair Brucek Khailany: "The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era", Norman Jouppi & Sridhar Lakshmanamurthy (Google). *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* Re-scan after 2026-08-25.

---

## Sources

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
- [Google TPU Architecture: 7 Generations Explained](https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations)
- [JAX Scaling Book — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)
- [TPU Deep Dive — Henry Ko](https://henryhmko.github.io/posts/tpu/tpu.html)
- [OCS Architecture — FiberMall](https://www.fibermall.com/blog/unveiling-google-tpu-architecture.htm)
- [Tensor Processing Unit — Wikipedia](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)
- [Eighth-Generation TPUs: Two Chips for the Agentic Era — Google Blog (2026-04-22)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [TPU 8t and TPU 8i Technical Deep Dive — Google Cloud Blog](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
- [AI Infrastructure at Next '26 — Google Cloud Blog](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26)
- [TPU 8t/8i and Virgo Network Analysis — fundaai (Substack)](https://fundaai.substack.com/p/researchtpu-8t8i-and-virgo-network)
- [Google TPU 8i and TPU 8t Announced — ServeTheHome](https://www.servethehome.com/google-tpu-8i-for-inference-and-tpu-8t-for-training-announced/)
- [Google Splits TPUv8 Strategy: Sunfish/Broadcom + Zebrafish/MediaTek — Wccftech](https://wccftech.com/google-splits-tpuv8-strategy-two-chips-broadcom-training-mediatek-inference-duties/) — *press reporting; the TSMC 2nm / Broadcom / MediaTek attributions rest on this piece and are unconfirmed by Google*
- [Cloud TPU Supported Versions / System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — retrieved 2026-08-08; newest documented generation is TPU7x (Ironwood); no v8 entry
- [Cloud TPU Release Notes](https://docs.cloud.google.com/tpu/docs/release-notes) — retrieved 2026-08-08; no v8 availability entry
- [Hot Chips 38 Program](https://hotchips.org/program/conference/) — Aug 24–25 2026; scheduled v8 architecture talk (content not yet public)
