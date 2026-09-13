# JAX Framework & Runtime Investigation

**Layer**: Framework Integration + Runtime + Communication
**Chip**: Google TPU (v4 / v5e / v5p / v6e / v7 Ironwood)
**as_of**: 2026-04-05

---

## Overview

JAX is Google's primary ML framework for TPU — the preferred authoring layer for research, production training, and increasingly inference. Unlike PyTorch, which started as an eager-execution imperative framework, JAX was designed from the ground up around functional transformations and ahead-of-time compilation. This functional model aligns perfectly with TPU's compiler-centric execution model: in JAX, all code destined for TPU must be traceable (no Python side effects) and compilable (expressible as XLA HLO programs).

Three components form the complete runtime path:

1. **JAX** — The Python-level framework providing numerical APIs (`jax.numpy`), functional transforms (`jit`, `grad`, `vmap`, `pmap`, `shard_map`), and the programming model for distributed computation.
2. **libtpu** — The single deployable shared library (`libtpu.so`) present on every Cloud TPU VM that bundles the XLA compiler, TPU kernel driver, and ICI communication runtime. JAX calls into libtpu for all TPU operations.
3. **Ecosystem libraries** — Flax (neural networks), Optax (optimization), Orbax (checkpointing), Grain (data loading), and Tokamax (high-performance kernels) compose the production JAX AI stack.

---

## Architecture

### JAX: Functional Transforms and Tracing

#### Core Design Principles

JAX's design rests on two foundational ideas:

1. **Functional purity**: JAX functions that are JIT-compiled must be pure (no Python side effects, no random global state). This is enforced by JAX's tracing mechanism — a Python function is evaluated with abstract placeholder values (tracers), producing a computation graph (jaxpr) that captures only the mathematical operations, not the Python control flow context.

2. **Composable transformations**: Transforms like `jit`, `grad`, `vmap`, and `pmap` are higher-order functions that accept functions and return functions. They compose: `jax.jit(jax.grad(loss_fn))` traces the gradient of `loss_fn` and then JIT-compiles the gradient function for TPU.

#### Primary Transforms

| Transform | Purpose | Mechanism |
|---|---|---|
| `jax.jit` | Compile and cache a function for accelerator execution | Traces to jaxpr → lowers to StableHLO → XLA compiles → dispatches to TPU |
| `jax.grad` | Reverse-mode automatic differentiation | Augments the jaxpr with adjoint computation via linearize + transpose |
| `jax.vmap` | Vectorize (batch) a function over a new axis | Broadcasts operations across a new batch dimension in the jaxpr |
| `jax.pmap` | Data-parallel sharding across devices | Maps function over device axes; inserts pmapped collectives (deprecated in favor of `jax.Array` + `shard_map`) |
| `jax.shard_map` | Fine-grained SPMD with explicit collective insertion | Newer API allowing per-shard computation with explicit `lax.all_reduce`, `lax.all_gather`, etc. |
| `jax.lax.scan` | Efficient loop unrolling / RNN-style computation | Compiles fixed-iteration loops as XLA `while` ops rather than unrolling N copies of the loop body |

#### jax.numpy: Drop-in NumPy API

`jax.numpy` mirrors the NumPy API (`jnp.matmul`, `jnp.einsum`, `jnp.where`, etc.) but returns JAX arrays that can be traced, differentiated, and compiled. On TPU, `jnp.matmul` traces to an HLO `dot_general` op which XLA lowers to a tiled MXU computation.

#### jax.Array and Sharding

Modern JAX (v0.4+) uses `jax.Array` — a unified array type that can be:
- **Replicated** across devices (same data on all chips)
- **Sharded** across devices (different slices on different chips)
- **Addressable** only on the local device slice

Sharding is expressed via `jax.sharding.NamedSharding` (mesh-based) or `jax.sharding.PartitionSpec`. XLA (via Shardy since March 2026) propagates sharding annotations from specified inputs through the entire computation graph, eliminating the need to annotate every intermediate tensor.

Example of named sharding:
```python
mesh = jax.make_mesh((8, 16), ('batch', 'model'))
x_sharding = NamedSharding(mesh, P('batch', None))  # shard batch dim across 8 devices
w_sharding = NamedSharding(mesh, P(None, 'model'))  # shard model dim across 16 devices
x = jax.device_put(x_np, x_sharding)
w = jax.device_put(w_np, w_sharding)
# XLA/Shardy automatically inserts AllReduce at the dot product
y = jnp.dot(x, w)
```

---

### libtpu: Unified Runtime

`libtpu.so` is the monolithic shared library deployed on every Cloud TPU VM. It is the software boundary between the JAX Python layer and the TPU hardware. Every TPU operation initiated by JAX passes through libtpu.

#### libtpu Components

```
libtpu.so
├── XLA Compiler (libtpu-XLA)
│   ├── StableHLO → HLO importer
│   ├── HLO optimization passes (fusion, layout, memory scheduling)
│   ├── Shardy SPMD partitioner
│   ├── TPU backend (HLO → LLO → VLIW binary)
│   └── Compilation cache (content-hash keyed, on-disk)
├── TPU Kernel Driver Interface
│   ├── DMA command submission
│   ├── Interrupt handling (compilation complete, execution complete)
│   ├── HBM buffer allocation/deallocation
│   └── PCIe memory-mapped I/O to TPU hardware
└── ICI Runtime
    ├── AllReduce, AllGather, AllToAll, ReduceScatter implementations
    ├── 3D/2D torus routing tables
    ├── Rendezvous and barrier synchronization
    └── Multislice DCN coordination (inter-pod collective scheduling)
```

#### libtpu Versioning

libtpu follows a strict versioning scheme tied to JAX releases. As of 2026:
- JAX and libtpu are installed as a matched pair (both from PyPI)
- `libtpu-nightly` packages were replaced by the stable `libtpu` package
- The TPU software versions matrix specifies which JAX version is compatible with which libtpu version and which TPU hardware generation
- Breaking API changes in the XLA compiler ABI require matched libtpu upgrades

#### Compilation Dispatch Flow

When JAX calls `jax.jit(fn)(x, W)`:

1. **Python tracing**: JAX traces `fn` with abstract shaped arrays for `x` and `W`, producing a jaxpr.
2. **StableHLO lowering**: The jaxpr is lowered to a StableHLO MLIR module.
3. **Shardy partitioning**: If `x` or `W` carry sharding annotations, Shardy propagates sharding through the StableHLO module and inserts collective ops.
4. **libtpu compile call**: JAX calls into libtpu's XLA compilation interface, passing the StableHLO module and target device info.
5. **XLA optimization + TPU codegen**: libtpu-XLA runs all HLO optimization passes and emits a TPU binary (VLIW instruction bundles).
6. **Binary caching**: The compiled binary is cached on disk keyed by the StableHLO module hash + device configuration.
7. **Dispatch**: libtpu submits the binary and HBM input buffer addresses to the TPU command queue via the kernel driver.
8. **Execution**: The TPU executes the VLIW program; libtpu polls for completion.
9. **Result**: Output HBM buffers are wrapped as `jax.Array` objects and returned to Python.

---

### Framework Ecosystem: Flax, Optax, Orbax, Grain

The JAX AI Stack is a curated set of libraries layered above JAX and below application-level code:

#### Flax — Neural Network Library

Flax provides object-oriented neural network module authoring on top of JAX's functional primitives:

- **NNX module system** (2024+): Stateful object model with explicit `nnx.Param`, `nnx.Variable` types. The module class holds parameter state; the forward method is a pure function given that state.
- **Linen** (legacy): Functional module system using `@nn.compact` and module `init`/`apply` patterns.
- **Layer library**: Dense, Conv, MultiHeadDotProductAttention, LayerNorm, Embed, etc.
- **MaxText integration**: Google's reference LLM implementation (Gemma, Llama, DeepSeek, Mistral) is written in Flax/JAX and uses Pallas for custom attention kernels.

Flax does not interact with libtpu directly — all computation flows through `jax.jit`-compiled JAX primitives. Flax parameters are regular `jax.Array` objects; the framework just provides structure and utilities.

#### Optax — Gradient Processing and Optimization

Optax provides composable optimizer implementations built on JAX:

- **Optimizer transforms**: `optax.adam`, `optax.sgd`, `optax.adafactor`, `optax.lamb`, etc.
- **Composability**: `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))` composes transforms into a single optimizer.
- **Gradient clipping, weight decay, schedule**: Built-in wrappers for common training techniques.
- **Sharding-aware**: Optax optimizer state (Adam's first/second moments) is stored as `jax.Array`, which participates in JAX's sharding propagation automatically.

#### Orbax — Checkpointing and Persistence

Orbax handles model serialization:

- **`orbax.checkpoint.CheckpointManager`**: Saves/restores `jax.Array` shards to/from Google Cloud Storage (GCS) with automatic background serialization.
- **Async checkpointing**: Continues training while the previous checkpoint is written to GCS, eliminating the checkpoint overhead from the training critical path.
- **Sharding-aware restoration**: Restores a checkpoint saved with one device mesh to a different device mesh without reshape operations (useful for scaling runs up/down).

#### Grain — Data Pipeline

Grain provides deterministic, reproducible data loading:

- **`grain.python.DataLoader`**: Iterator-based API that maps transformations over datasets with deterministic shuffling (seeded random permutation reproducible across restarts).
- **Dataset sources**: Google Cloud Storage, local filesystem, ArrayRecord format.
- **Prefetching**: Grain runs data loading on host CPU threads, prefetching batches into pinned host memory ready for DMA to TPU HBM.

#### Tokamax — High-Performance Kernel Library

Tokamax is Google's production-quality Pallas kernel library:

- Implements state-of-the-art Flash Attention, MoE routing kernels, RoPE embeddings, and other transformer primitives using Pallas/Mosaic.
- Automatically selects the right kernel for the target TPU generation (v5p vs v6e vs v7) and data type.
- Used by MaxText and production Gemini training/serving infrastructure.

---

## Data Flow

### Training Step Data Flow (Large-Scale LLM)

A single gradient update step on a 9,216-chip Ironwood pod running MaxText:

**Phase 0 — Data loading (host side)**
Grain loads a tokenized batch from GCS → host RAM → pinned host memory. Batch is sharded: each chip's host buffer holds 1/9216 of the global batch.

**Phase 1 — Forward pass**
1. `jax.jit(train_step)(state, batch)` is invoked. On first call, JAX traces the full transformer forward + backward pass to jaxpr.
2. Shardy propagates tensor sharding through the 100+ layer transformer (tensor-parallel + data-parallel + pipeline-parallel annotations).
3. XLA compiles the sharded jaxpr; libtpu distributes the binary to all 9,216 TPU chips.
4. Each chip receives its shard of the batch via DMA from host HBM.
5. Forward pass executes: embedding lookup (SparseCore), attention (MXU for QKV projections, custom Pallas attention kernel for softmax+score), FFN (MXU for weight projections, VPU for activations).
6. **AllReduce at attention output**: After each attention head's partial sum, a Shardy-inserted ICI AllReduce aggregates partial results across tensor-parallel chips.

**Phase 2 — Backward pass**
7. XLA-compiled backward pass computes gradients via reverse-mode AD on the same chips.
8. **Gradient AllReduce**: Data-parallel gradient synchronization via ICI AllReduce (within pod) or DCN + Titanium IPU AllReduce (multi-pod Multislice). This is the dominant collective communication cost.
9. Gradients are accumulated in each chip's HBM gradient buffer.

**Phase 3 — Optimizer update**
10. Optax's Adam step is applied: weight tensors and optimizer states (m, v) updated in-place.
11. Updated model state is returned as new `jax.Array` objects.

**Phase 4 — Checkpointing (async)**
12. Every N steps, Orbax serializes model shards from each chip's HBM to host RAM to GCS asynchronously, not blocking the next training step.

---

## Hardware Interface

### JAX → libtpu → TPU Interface

The interface boundary between JAX and the TPU hardware:

```
JAX Python layer
  jax.Array (logical sharded array)
  jax.jit (compilation trigger)
  jax.lax.* primitives (HLO-mapped ops)
        |
        |  [StableHLO serialization via MLIR]
        v
libtpu.so
  XLA compiler (StableHLO → VLIW binary)
  TPU driver (PCIe command queue submission)
  ICI runtime (AllReduce/AllGather coordination)
        |
        |  [PCIe DMA + command queue]
        v
TPU Hardware
  HBM (model weights, activations, gradients)
  VMEM (tiles staged from HBM by DMA)
  MXU (systolic array, 128×128 or 256×256)
  ICI (chip-to-chip collective fabric)
```

### Collective Communication Implementation

JAX expresses distributed computation via `jax.lax` collective primitives:

| JAX Primitive | XLA HLO Op | ICI Implementation |
|---|---|---|
| `lax.psum` | `AllReduce` | Ring AllReduce along ICI torus axis |
| `lax.all_gather` | `AllGather` | Bidirectional ICI gather along torus axis |
| `lax.all_to_all` | `AllToAll` | ICI-routed full data exchange |
| `lax.psum_scatter` | `ReduceScatter` | Combined reduce + scatter |
| `lax.ppermute` | `CollectivePermute` | Direct ICI neighbor-to-neighbor send |

These primitives are inserted by Shardy during XLA compilation when sharding annotations change between operations. libtpu's ICI runtime handles the low-level 3D torus routing, synchronization barriers, and timing for all collectives.

For multi-pod Multislice operations:
- Intra-pod collectives use ICI (high bandwidth)
- Inter-pod gradient AllReduce uses DCN via Titanium IPU (lower bandwidth, higher latency)
- XLA schedules DCN AllReduce during periods when intra-pod MXU compute can proceed concurrently

---

## Key Findings

1. **JAX and TPU are co-designed**: JAX's functional, pure-function programming model was designed with TPU's compiler-centric execution in mind. Python imperative code that cannot be traced to a jaxpr (code with untraced Python side effects) cannot run on TPU — the hardware architecture drives the framework design.

2. **libtpu is the entire software-hardware interface**: Unlike NVIDIA's stack (CUDA runtime + cuBLAS + NCCL + driver as separate components), TPU bundles everything into a single `libtpu.so`. This makes TPU upgrades atomic — a new libtpu version brings a new compiler, new driver, and new ICI runtime simultaneously — but also means the versioning is tightly coupled.

3. **Shardy completes SPMD automation**: With Shardy's full adoption in JAX (March 2026), users specify sharding intent via `NamedSharding` annotations on inputs and outputs; Shardy propagates sharding through the entire transformer and inserts collectives automatically. This removes the fragile per-op annotation style of older `with_sharding_constraint` usage.

4. **The Flax/Optax/Orbax stack is production-validated**: All of Google DeepMind's Gemini models, Google Brain's research, and most Google Cloud AI products train using the JAX AI stack (JAX + Flax + Optax + Orbax). MaxText's open-source LLM codebase is the reference implementation, handling trillion-parameter models across 9,216+ chips.

5. **PyTorch/XLA as a second entry point**: The TorchTPU initiative (Google/Meta collaboration) aims to make JAX's XLA backend fully accessible from PyTorch via `torch.compile`. As of 2025-2026, vLLM uses JAX as the lowering path for all TPU inference, even for PyTorch-defined models — demonstrating JAX/XLA's role as the universal TPU compilation substrate.

6. **libtpu compilation caching enables production throughput**: XLA compilation for a 100B+ parameter model takes 30–120 seconds. libtpu's on-disk binary cache (keyed by StableHLO module hash) ensures that repeated training runs with the same model architecture skip recompilation entirely, making compilation cost amortizable over thousands of training steps.

7. **Grain's deterministic data loading enables reproducibility**: Distributed ML training at 9,000+ chips is notoriously hard to reproduce. Grain's deterministic shuffle seeds and checkpointable data loader state mean that a training run can be exactly reproduced from any checkpoint, including the exact data order seen by each chip — a property critical for debugging and compliance.

8. **GSPMD → Shardy migration eliminates a major sharp edge**: GSPMD's opaque sharding propagation was a common source of user confusion (unexpected collective insertion, poor bisection bandwidth choices). Shardy's explicit mesh-axis representation and MLIR-based debugging tools make sharding strategies inspectable and predictable, reducing the expertise barrier for large-scale JAX training.

---

## Relation to Hardware

JAX's framework abstractions are shaped by TPU hardware constraints:

- **`jax.jit` is mandatory, not optional**: TPU's static VLIW execution requires a complete compile before any computation can run. `jax.jit` is the user-visible knob that triggers this. Unlike GPU frameworks where eager execution is valid, TPU programs that omit `jax.jit` fall back to slow, op-by-op dispatch.

- **`jax.Array` sharding ↔ ICI topology**: The named mesh dimensions in `NamedSharding` (`('batch', 'model', 'pipeline')`) map to physical ICI axes. Shardy chooses collective types (AllReduce vs AllGather vs ReduceScatter) and scheduling to maximize ICI bandwidth utilization for the given mesh topology.

- **Optax sharding transparency**: Because optimizer state is stored as `jax.Array` with the same sharding as model weights, Optax updates run in-place on each chip's VMEM without inter-chip communication for the update itself. Only gradient synchronization (AllReduce) requires ICI traffic.

- **Orbax async checkpointing ↔ HBM bandwidth**: Writing a 100B+ parameter model to GCS from 9,216 chips in parallel requires careful orchestration of HBM reads, PCIe transfers, and GCS writes. Orbax's async serialization pipeline overlaps these transfers with the next training iteration's MXU compute.

- **libtpu ICI AllReduce ↔ 3D torus routing**: libtpu's AllReduce implementation uses bidirectional ring AllReduce along each ICI torus dimension sequentially. The 3D torus allows three independent ring AllReduces (one per dimension), each using the full per-axis ICI bandwidth, achieving near-peak efficiency for data-parallel gradient synchronization.

---

## Sources

- [JAX Repository — GitHub](https://github.com/jax-ml/jax)
- [JAX Documentation](https://docs.jax.dev/en/latest/)
- [Building Production AI on Cloud TPUs with JAX](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)
- [libtpu on PyPI](https://pypi.org/project/libtpu/)
- [TPU Software Versions Matrix](https://docs.cloud.google.com/tpu/docs/runtimes)
- [JAX JIT Compilation — JAX Documentation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [Shardy Guide for JAX Users — OpenXLA Project](https://openxla.org/shardy/getting_started_jax)
- [Shardy JAX Migration — JAX Documentation](https://docs.jax.dev/en/latest/shardy_jax_migration.html)
- [Flax Repository — GitHub](https://github.com/google/flax)
- [Optax Repository — GitHub](https://github.com/google-deepmind/optax)
- [Orbax Repository — GitHub](https://github.com/google/orbax)
- [Grain Repository — GitHub](https://github.com/google/grain)
- [MaxText Repository — GitHub](https://github.com/AI-Hypercomputer/maxtext)
- [vLLM TPU Backend Blog](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)
- [PyTorch/XLA SPMD Blog](https://pytorch.org/blog/pytorch-xla-spmd/)
- [JAX Scaling Book — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/)
- [A Developer's Guide to Debugging JAX on Cloud TPUs](https://developers.googleblog.com/a-developers-guide-to-debugging-jax-on-cloud-tpus-essential-tools-and-techniques/)

---

## Dated Investigation — 2026-08-08: software stack, window 2026-04-27 → 2026-08-08

*This chip's software-stack investigation lives in this file and `xla-pallas.md`; there is no separate `software-stack.md` for google-tpu. Primary source for this pass: the Cloud TPU release notes, retrieved 2026-08-08.*

### The one substantive change: Compute Engine-native TPU provisioning (GA 2026-06-01)

**Cloud TPU release notes, 2026-06-01, GA — Compute Engine now natively supports TPUs.** TPU VMs and TPU slices can be provisioned and managed through the standard **Compute Engine instance and managed-instance-group APIs**, with custom OS images and configurable boot-disk sizing, across all consumption options (on-demand, spot, reservation).

Why this matters to the stack rather than to the marketing page: TPU has historically been a bespoke GCE resource type reached through TPU-specific APIs (`gcloud compute tpus`, the TPU Node / TPU VM resource model), with its own lifecycle, its own image handling, and no access to the MIG machinery that the rest of Compute Engine uses for autohealing, rolling updates, and autoscaling. Making a TPU slice an ordinary Compute Engine instance is a **control-plane convergence**: fleet operations, image management, and disk configuration for TPU converge on the same primitives as CPU and GPU fleets.

What it does **not** change: nothing in the compiled execution path. JAX → jaxpr → StableHLO → Shardy → HLO → LLO → VLIW is untouched, libtpu is unchanged, and ICI/collective behaviour on the device is unaffected. This is a provisioning and fleet-management change that sits strictly above `libtpu.so`.

### Other release-notes entries in the window

- **2026-04-27, GA:** "Cloud TPU now offers TPU availability in AI zones." Same date as the repo's prior refresh baseline; additive and low-value.

### Explicitly not found in this window

- **No libtpu version entries and no JAX version entries** appear in the Cloud TPU release notes for May–August 2026. **No SDK version numbers are recorded for this window.**
- **TPU v8 does not yet appear in the Cloud TPU supported-version list**, which still tops out at TPU7x (Ironwood) as of 2026-08-08. There is consequently no v8 runtime, libtpu, or JAX compatibility entry to record. The only "preview" feature named in Google's v8 deep-dive is **native PyTorch support on TPU** — a software preview, not a silicon preview.
- No changes to Shardy, Pallas/Mosaic, Tokamax, MaxText, or the vLLM TPU backend surfaced in the window's release notes.

### Sources (2026-08-08 pass)

- [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes) — retrieved 2026-08-08; 2026-06-01 and 2026-04-27 GA entries
- [Cloud TPU system architecture / supported versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — retrieved 2026-08-08
- [TPU 8t and TPU 8i technical deep dive — Google Cloud Blog (2026-04-22)](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — native PyTorch support in preview
