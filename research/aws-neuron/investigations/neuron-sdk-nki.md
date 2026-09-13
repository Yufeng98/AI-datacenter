# AWS Neuron SDK & NKI — Investigation Report

**Chip:** AWS Neuron (Trainium/Inferentia)
**Investigation Focus:** Compiler Pipeline, NKI, Framework Integrations, Runtime
**Date:** 2026-04-05

---

## Overview

The AWS Neuron SDK is a comprehensive software stack for compiling, optimizing, and executing machine learning workloads on AWS custom silicon (Trainium and Inferentia). The stack spans from high-level framework APIs (PyTorch, JAX) down to bare-metal kernel programming via NKI, and terminates at a thin runtime library (libnrt.so) embedded inside each ML framework process. The compiler pipeline is the central abstraction: it accepts model graphs from multiple frontend paths and produces NEFF (Neuron Executable File Format) binaries that the runtime loads onto NeuronCores.

At re:Invent 2025 (Neuron SDK 2.27), AWS open-sourced the NKI compiler under Apache 2.0 and announced TorchNeuron — a native PyTorch backend planned for PyTorch 2.10+ that replaces the XLA dependency for PyTorch workflows while preserving compatibility with existing torch-neuronx code.

---

## Architecture

### Compiler Pipeline

The Neuron compiler (`neuronx-cc`) operates in a multi-phase pipeline:

```
Framework Graph (PyTorch FX / JAX HLO / XLA)
        │
        ▼
[1] XLA Frontend — hardware-agnostic graph optimizations
    • constant propagation, re-materialization, operator fusion
    • produces StableHLO / XLA HLO IR
        │
        ▼
[2] MLIR Lowering — Neuron-specific middle-end
    • maps HLO ops to Neuron dialect ops
    • tiling, memory placement, pipelining decisions
        │
        ▼
[3] NeuronCore Back-end — hardware code generation
    • NKI IR → NeuronISA machine instructions
    • schedules Tensor/Vector/Scalar/GPSIMD engines
        │
        ▼
[4] NEFF (Neuron Executable File Format)
    • single-file container: instructions + weights + tensor metadata
    • loaded by libnrt.so at model load time
```

NKI kernels enter the pipeline at phase 3 (they bypass XLA graph passes and are compiled directly via the MLIR-based NKI compiler), giving kernel authors direct control over instruction scheduling and memory placement.

### Neuron Kernel Interface (NKI)

NKI is a Python-based bare-metal tile programming API with NumPy/Triton-like syntax. It exposes two API layers:

- **`nki.lang`** — High-level tile operations: memory allocation (`nl.ndarray`), load/store (`nl.load`, `nl.store`), compute (`nl.matmul`, `nl.add`), synchronization, and control flow. Provides access abstractions over SBUF and PSUM memory.
- **`nki.isa`** — Low-level ISA intrinsics: direct engine dispatch (`tensor_tensor`, `activation`, `bn_stats`), explicit SBUF/PSUM partition addressing, and instruction-level pipelining. Enables overlapping DMA and compute.
- **`nki.collectives`** (added Neuron 2.27) — Multi-NeuronCore collective operations within a kernel: `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `collective_permute`, `rank_id`.

NKI kernels compile with the open-source MLIR-based NKI Compiler. The compiler preserves developer-specified execution order and memory allocation while applying backend-level optimization passes.

**Integration points:**
- NKI kernels can be called directly from Python as standalone functions on Trainium/Inferentia.
- NKI kernels can be registered as framework custom operators via `nki.baremetal` (PyTorch) or JAX `custom_call` bindings, integrating into larger model graphs.
- The `nki-samples` GitHub repository provides reference implementations: Flash Attention, matrix multiply, RoPE, and MoE routing kernels.

### TorchNeuron (PyTorch Neuron)

The Neuron SDK provides two PyTorch integration paths:

**Path 1 — torch-neuronx (current, PyTorch ≤ 2.9):**
Implemented on top of PyTorch/XLA. The device is presented as an XLA device; `xm.mark_step()` triggers lazy evaluation and JIT compilation. `torch_neuronx.trace()` AOT-compiles a model graph to NEFF. TorchDynamo backend (`torch.compile(backend="openxla_eval")`) captures FX graphs and routes through XLA HLO before neuronx-cc back-end.

**Path 2 — TorchNeuron (planned PyTorch ≥ 2.10, private beta as of Neuron 2.27):**
Registers Trainium/Inferentia as a native PyTorch device using the **PrivateUse1** device backend mechanism. This eliminates the PyTorch/XLA dependency. Features:
- **Eager mode execution** — tensors live natively on the Neuron device; ops are dispatched eagerly.
- **torch.compile** — TorchDynamo captures FX graphs; TorchNeuron's custom backend transforms them into NeuronISA instructions.
- **Standard distributed APIs** — `torch.distributed`, FSDP, DDP, DTensor work with no Neuron-specific modifications.
- **torch.nn.parallel** compatibility.

### JAX Neuron Integration

Introduced September 2024 (Neuron SDK 2.20). JAX workloads compile via XLA's HLO path: JAX traces to StableHLO, which is then ingested by neuronx-cc's XLA frontend. Key integration:
- `jax.jit` triggers compilation through the Neuron XLA backend.
- NKI kernels integrate via JAX's `custom_call` mechanism.
- Supports both training (via JAX `grad`) and inference.
- NxD Core's JAX APIs provide tensor-parallel primitives for multi-NeuronCore JAX workloads.

### Neuron Runtime (libnrt.so)

The runtime is a shared library (`libnrt.so`) installed at `/opt/aws/neuron/lib/`. It is dynamically linked into the ML framework process — there is no separate runtime daemon. Architecture:

```
ML Framework Process
    ├── libnrt.so  (Neuron Runtime Library)
    │       ├── NEFF loading & validation (checksum)
    │       ├── NeuronCore memory management
    │       ├── DMA scheduling & execution
    │       ├── Collective communication orchestration
    │       └── Metrics / telemetry APIs
    └── aws-neuronx-dkms  (kernel mode driver)
            └── PCIe / Nitro device access
```

The runtime loads NEFF binaries onto one or more NeuronCores, manages the SBUF/PSUM allocation declared in the NEFF, schedules DMA transfers for weight streaming, and handles multi-NeuronCore collective communication. It communicates with the kernel driver (aws-neuronx-dkms) which provides PCIe access to the NeuronDevice on Trainium/Inferentia hardware.

---

## Data Flow

A typical training step (torch-neuronx path):

1. Python script calls `optimizer.step()` / `xm.mark_step()`.
2. PyTorch/XLA accumulates a lazy graph of operations.
3. At `mark_step`, the graph is lowered to XLA HLO and sent to `neuronx-cc`.
4. `neuronx-cc` runs XLA → MLIR → NeuronISA passes; emits a NEFF.
5. `libnrt.so` loads the NEFF: streams weights via DMA to HBM, places activation buffers in SBUF.
6. The Tensor/Vector/Scalar engines execute the forward pass; PSUM holds intermediate results.
7. AllReduce gradients are dispatched via NeuronLink (intra-node) or EFA (inter-node) without CPU involvement.
8. Updated weights are written back to HBM.

For NKI kernels the path is shorter: Python → NKI compiler (MLIR) → NEFF → libnrt load → direct engine execution.

---

## Hardware Interface

- **Engine dispatch:** `nki.isa` instructions directly address Tensor Engine (systolic array), Vector Engine, Scalar Engine, and GPSIMD Engine via ISA opcodes.
- **Memory model:** SBUF partitions (128 partitions × 224 KiB on v3) and PSUM (2 MiB) are explicitly addressed; the NKI tile API maps Python index expressions to partition addresses.
- **DMA:** `nl.load` / `nl.store` emit DMA instructions that transfer tiles between HBM and SBUF/PSUM without CPU involvement.
- **Collectives:** `nki.collectives` targets NeuronLink (intra-node) or EFA (inter-node) depending on topology.
- **NEFF:** output artifact consumed by `libnrt.so`; contains NeuronISA instructions, parameter tensors, and execution metadata.

---

## Key Findings

1. **Two-track PyTorch strategy:** AWS is transitioning from XLA-based torch-neuronx to a native PrivateUse1 backend (TorchNeuron) for PyTorch ≥ 2.10, reducing the dependency surface and enabling standard PyTorch distributed APIs without wrappers.
2. **NKI bypasses graph compiler:** NKI kernels enter the pipeline at the MLIR back-end, providing tile-level control unavailable through the standard XLA frontend — critical for Flash Attention and MoE routing where memory layout matters.
3. **Open-source MLIR NKI compiler (Neuron 2.27):** The NKI compiler is now Apache 2.0 licensed, enabling community contributions and independent compilation toolchain development.
4. **libnrt.so in-process architecture:** The absence of a sidecar daemon simplifies deployment; the runtime is co-versioned with the framework package and managed per-process.
5. **NEFF as the canonical artifact:** All compilation paths produce NEFF, enabling model caching, offline compilation, and checksum-validated loading — important for reproducible cluster deployments.
6. **nki.collectives bridges kernel and network layers:** Adding collective ops inside NKI kernels lets custom attention or MoE kernels perform cross-chip reductions without returning to the framework layer.

---

## Relation to Hardware

| Software Layer | Hardware Target |
|---|---|
| `nki.isa.tensor_tensor` | Tensor Engine (128×128 BF16 systolic array, v2/v3) |
| `nki.isa.activation` | Vector Engine |
| `nki.isa.scalar` | Scalar Engine |
| `nki.isa.gpsimd` | GPSIMD Engine (8× 512-bit vector processors) |
| `nl.load` / `nl.store` | DMA Engine → SBUF / PSUM ↔ HBM |
| `nki.collectives.all_reduce` | NeuronLink (intra-node) / EFA (inter-node) |
| NEFF weight streaming | HBM → SBUF via DMA at model load |
| `libnrt.so` device access | PCIe via aws-neuronx-dkms kernel driver |

---

## Update — 2026-08-08 (Neuron SDK 2.29 → 2.31)

**Investigation focus:** SDK releases, NKI stability status, compiler backend, runtime, and new SDK components in the window 2026-04-05 → 2026-08-08.
**Method:** GitHub release-tag metadata (`published_at`) plus the component release-notes sources in `aws-neuron/aws-neuron-sdk` (`release-notes/components/*.rst`). WebSearch was unavailable during verification; all confirmation came from direct primary-source fetches.

### Release timeline

The "Neuron SDK 2.27" framing used throughout the prior revision of this report was **already stale at the 2026-04-05 baseline** — 2.28.0 (2026-02-25) and 2.28.1 (2026-03-13) predate it. Four releases fall inside the window:

| Release | Date (GitHub `published_at`) | NKI |
|---|---|---|
| 2.29.0 | 2026-04-09 | NKI 0.3.0 |
| 2.29.1 | 2026-05-01 (patch) | — |
| 2.30.0 | 2026-05-21 | NKI 0.4.0 |
| 2.31.0 | 2026-07-07 (current latest as of 2026-08-08) | NKI 0.5.0 |

*Date discrepancy:* the readthedocs release index reportedly shows 04/09/26 for 2.29.1, while the GitHub tag `v2.29.1` is dated 2026-05-01/02. GitHub tag metadata is the stronger source; cite 2026-05-01.

### 2.29.0 (2026-04-09) — NKI Beta → Stable

The most consequential software-stack event in the window for this survey, whose kernel-layer narrative previously described NKI only as a Beta tile API with an open-sourced compiler.

- **NKI 0.3.0 promoted from Beta to Stable.**
- **NKI Standard Library (`nki-stdlib`)** — developer-visible source for all NKI APIs and native NKI language objects (`NkiTensor`).
- **CPU Simulator (experimental)** — executes NKI kernels entirely on host CPU; no Trainium hardware required; enabled by environment variable or API. First local-dev path for Neuron kernels without accelerator access.
- `nki.collectives.all_to_all_v` (variable-length all-to-all).
- 7 new experimental NKI-Lib kernels (Conv1D, Transformer TKG, comm-compute fusion).
- **Neuron Explorer** profiling/debugging suite: Beta → Stable, with full Device widget support; published on the VS Code Extension Marketplace.
- **Breaking hardware-support change:** NKI 0.3.0 does not support Trn1/Inf2. Consequently NxD Inference no longer supports NKI kernels on Trn1/Inf2, and the component note states NxD Inference models are supported only on **Trn2 and newer**. Trn1/Inf2 users must pin to 2.28. *Caveat:* the 2.31.0 release-notes index card still lists NxD Inference against Inf2/Trn1/Trn1n/Trn2/Trn3 — cite the component release note, not the index card.
- API renames: `SbufManager` → `BufferManager`; MoE TKG boolean flags → `LNCShardingStrategy` enum.

### 2.30.0 (2026-05-21) — Trn3 hardware exposed through NKI

- **NKI 0.4.0:** `nki.isa.activate2` (fused preprocess + activate on the Scalar Engine, Trn3-specific), OCP FP8 inputs to matrix-multiply, Vector-Engine absolute-value reductions, bytes-aware tile-size properties (`sbuf_size_bytes`, `sbuf_fmax_bytes`).
- **Graph compiler:** rewritten memory-liveness analysis (compile-time improvement); **maximum computation tile size doubled to 1024**; coalesced reduce-scatter optimization.
- **Runtime:** zero-copy host↔device transfers **enabled by default**; async event APIs; Collectives support for the **Trn3 Gen2 UltraServer ring topology**.
- **NKI Library:** +3 core kernels (segmented attention, KV-parallel prefill, FP8 quantization); +19 experimental (context parallelism, MXFP8 training, state-space models, fused optimizers, MoE dispatch); PyTorch reference implementations for 29 kernels.
- **Neuron DRA Driver** — Kubernetes Dynamic Resource Allocation with topology-aware scheduling of Trainium devices and EFA interfaces.
- DLAMIs rebased on Ubuntu 24.04; JAX-NeuronX 0.10.0.

### Neuron Agentic Development — a stack layer this survey did not document

`neuron-agentic-development` first appears as a first-class, separately release-noted Neuron SDK component in **2.30.0**, bundled in all DLAMIs and DLCs by default. Two skills:

- **`neuron-framework-autoport`** — ports HuggingFace transformer models to NxD Inference end-to-end, including compilation and greedy-token-match accuracy validation.
- **`neuron-framework-equivalence`** — numerical-equivalence validation via progressive 3-tensor R-ratio analysis with fault localization.

2.31.0 only refreshes these skills for NKI 0.5.0 compatibility.

*Scope caveat:* the component release-notes file begins at 2.30.0. The defensible claim is "first documented SDK component in 2.30.0", **not** "new subsystem" and **not** a beta→default transition (no such transition is documented). Whether informal NKI-authoring / debug / profile agent skills shipped before 2.30 is **not disclosed**.

### 2.31.0 (2026-07-07) — compiler backend redesign

- **NKI 0.5.0:** tensor indirection (gather/scatter) via `.indirect()` as a new on-chip addressing mode on compute operations; OCP MX scale format (`float8_e8m0fnu`); composable `NkiTensor` view methods (slice / select / permute / rearrange) for zero-cost layout transforms; 8192-element bf16 destination output for `nc_matmul`.
- **`neuronx-cc` redesigned code-generation backend, now default for Trn2 and Trn3** — improved instruction scheduling and memory prefetch. This is a change to the compiler layer this report documents as "NeuronCore back-end".
- **Runtime:** contiguous shared scratchpad support.
- **UltraServer Operator** for Amazon EKS: public beta.
- **NKI Library:** 14 new **experimental** kernels (deformable attention, MoE training collectives, indexed gather/scatter, DeepSeek MLA projection, ring attention) — experimental, not GA library kernels.
- Deprecations: SPMD launch-grid requirement for LNC2 kernels deprecated, to be removed in a future release (behavior unchanged in 0.5.0); `neuronxcc.nki.*` namespace use is now a hard compile error; `nki.jit(platform_target=...)` and `nki.jit(mode=...)` deprecated.
- Supported instances: Inf1, Inf2, Trn1, Trn1n, Trn2, Trn3.

### Could not confirm

- **TorchNeuron / PrivateUse1.** No status change found. Neither `torch-neuronx` nor TorchNeuron appears as an updated component in the 2.31.0 release-notes index — but that index explicitly notes that some components may not be updated in a given release, so this is evidence of neither GA nor deprecation. Keep "private beta as of Neuron SDK 2.27", flagged unverified since 2.27.

### Revised key findings (superseding items 1 and 3 above)

1. **NKI is now a Stable API (2.29.0), not a Beta preview.** The survey's kernel-layer narrative should treat NKI as a supported, versioned public surface, with `nki-stdlib` exposing its own source and `NkiTensor` as a first-class language object.
2. **Trn1/Inf2 have been dropped from the forward path.** NKI 0.3.0 targets Trn2 and newer only, which propagated into NxD Inference as a breaking change. Trn1/Inf2 remain supported only on the 2.28 line.
3. **The compiler back-end was redesigned in 2.31.0** and is default on Trn2/Trn3 — the first codegen-level rewrite recorded in this survey.
4. **A new SDK layer exists above the frameworks:** agentic porting and equivalence-checking skills shipped as SDK components, bundled in DLAMIs/DLCs. No other chip in this survey currently documents an equivalent layer.
5. **Local development no longer requires accelerator hardware** for kernel authoring, via the experimental CPU Simulator.
