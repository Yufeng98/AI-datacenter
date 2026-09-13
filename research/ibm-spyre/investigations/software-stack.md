# IBM Spyre Accelerator — Software Stack Investigation

*as_of: 2026-08-08*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*

---

## Overview

The Spyre stack is unusual among enterprise accelerators. IBM stood up a **public GitHub org
(`github.com/torch-spyre`) with 14 mostly Apache-2.0 repositories**, published **design RFCs and
compiler↔device interface specifications in the open**, and hosts full developer documentation at
**torch-spyre.readthedocs.io**. Very little of this is marketing prose — the docs describe pass pipelines,
scratchpad solvers, stick constraints and tensor-layout algebra at a level normally found only in internal
compiler design documents.

But the stack is **split open/closed at a very specific seam**. Everything from PyTorch down to the
SuperDSC/KTIR interface is open; the **back-end compiler (DeepTools), the runtime (DeepRT/Flex), the
collectives library, the driver and the firmware are proprietary**, and `torch_sendnn` is *"only available
pre-installed in a base environment."* The torch-spyre installation page still says instructions will be
published *"once the runtime reaches public availability."*

IBM's own summary of the split:

> **Open source** — PyTorch Inductor (extended for Spyre), the Torch-Spyre front-end compiler.
> **Proprietary** — the DeepTools back-end compiler and the SuperDSC format specification.

(Note the second half of that sentence is now partly out of date in IBM's favour: the SuperDSC *bundle
format* is in fact published as an open interface spec, even though the DeepTools consumer is closed.)

**Honest framing for the survey:** the front end and the middle IRs are genuinely open and unusually well
documented — including a hardware-free CPU interpreter for the tile IR — but the final lowering, runtime,
driver and ISA are a closed box, and the runtime is not even publicly downloadable.

---

## Layer 1 — Framework Integration

| Component | Detail | Open source |
|-----------|--------|-------------|
| **PyTorch 2.x** (2.9.1 → ~2.11; `torch_sendnn` 1.2.0 pairs with torch 2.10.0) | Primary and effectively only framework. `spyre` is registered as a first-class device via **`PrivateUse1`**: `torch.utils.rename_privateuse1_backend("spyre")` + `torch._register_device_module("spyre", …)`. Both eager and `torch.compile(model, backend="spyre")` paths exist | ✅ Apache-2.0 (`torch-spyre/torch-spyre`) |
| **sendnn-inference** | The production **vLLM plugin**, ~638 commits. Formerly `vllm-project/vllm-spyre`, which now redirects to `torch-spyre/sendnn-inference`. Hosted docs at `docs.vllm.ai/projects/spyre`. Community Slack channel `#sig-spyre` | ✅ Apache-2.0 |
| **spyre-inference** | Second-generation vLLM plugin built on `torch-spyre` (the older `sendnn-inference` sits on `torch_sendnn`) | ✅ Apache-2.0 |
| **hf-adapters** | HuggingFace Transformers enablement by **monkey-patching at load time** — replaces only the operations Spyre cannot run natively (RoPE, RMSNorm, KV-cache management, generation loop); weights, tokenizers and configs are untouched. 30 adapters, 46 verified checkpoints, 100+ models (Llama, Qwen, Granite, Mistral, BERT, XLM-RoBERTa, VLMs) | ✅ Apache-2.0 |
| **sglang-spyre** | Early SGLang port | 🟡 no license file |
| **foundation-model-stack / aiu-fms-testing-utils** | The older FMS-based path driving `torch_sendnn` directly. Documents the environment-variable surface: `FLEX_COMPUTE=SENTIENT`, `FLEX_DEVICE=PF`, `DTLOG_LEVEL`, `TORCH_SENDNN_LOG`, `DT_DEEPRT_VERBOSE`, `TORCH_SENDNN_CACHE_ENABLE`, `DTCOMPILER_KEEP_EXPORT`, `SENCORES` | ✅ (utils repo) |
| **aiu-bench** | Self-hosted performance suite for testing AIU compilers | ✅ Apache-2.0 |

### vLLM feature status

| Supported | Partial | Not supported |
|-----------|---------|---------------|
| chunked prefill, automatic prefix caching, guided decoding, logprobs, beam search, **tensor parallel**, embedding models | multimodality, quantization | LoRA, speculative decoding, encoder-decoder, prompt logprobs, pipeline parallel, expert parallel, data parallel, PD disaggregation, sleep mode |

### Negative finding worth recording

The **IBM Z Deep Learning Compiler (zDLC)** and **zDNN** — IBM's ONNX-MLIR-based flow for AI on IBM Z —
target the **Telum / Telum II zAIU via the NNPA instruction only**. The zDNN README references Telum I/II
and never mentions Spyre or AIU-as-Spyre; the zDLC 5.0 announcement is explicitly about z17's Telum II.
**No public ONNX-based or z/OS-native (MLz/WMLz) path to Spyre is documented.** Spyre's public software path
is PyTorch/vLLM on Linux on Z, Linux on Power, and Red Hat OpenShift.

---

## Layer 2 — Graph Capture

**TorchDynamo + AOTAutograd** trace the program into an **FX graph of ATen operations**; decomposition passes
reduce it to core ATen.

- **Ahead-of-time compilation with static shapes** is the supported mode.
- `torch.compile(..., dynamic=True)` is handled by **static binary specialization** — compile once per
  shape, reuse the binary across equivalent geometries.
- `torch.compile` artifact caching cuts startup time.
- **Eager mode** is implemented by AOT-compiling each operation individually (e.g.
  `@torch.library.register_kernel("aten::mm", ["spyre"])` wrapping `torch.compile(torch.mm, dynamic=False)`)
  — correct but slow.
- A **single graph break in the hot path erases the gains** of the surrounding compiled region. On an
  architecture with no caches and no runtime scheduler, falling back to host execution is far more
  expensive than on a GPU.

---

## Layer 3 — Graph Compiler (the deepest open layer)

**TorchInductor with an out-of-tree Spyre backend.** Three components are registered:

| Component | File | Role |
|-----------|------|------|
| `SuperDSCScheduling` | `scheduler.py` | Replaces Inductor's Triton scheduling; decides op grouping and ordering |
| `SpyrePythonWrapperCodegen` | `wrapper.py` | Python wrapper codegen for kernel dispatch |
| `SpyreDeviceOpOverrides` | `device/op_overrides.py` | Device-specific op implementations |

Six upstream extension points are used — `CustomPreGradPasses`, `CustomPrePasses`, `CustomPostPasses`,
`CustomPreFusionPasses`, `CustomPostFusionPasses`, `CustomPreSchedulingPasses` — with `enable_spyre_context`
as the entry point.

**Pipeline:**

```
PyTorch
  → FX graph (ATen)
  → LoopLevelIR
  → OpSpec
  → SuperDSC JSON bundle
  → [DeepTools — proprietary]
  → SpyreCode device binary
```

### The 16-pass LoopLevelIR pre-scheduling pipeline

This is the part that encodes the hardware model directly into the compiler:

1. deadcode elimination
2. split multi-ops
3. **propagate tensor layouts** (assign `FixedTiledLayout`)
4. validate ops (shared `ElementArrangement`)
5. optimize restickify locations
6. finalize layouts
7. **insert restickify**
8. post-mutation restickify
9. BMM padding
10. dedup constants
11. propagate named dims
12. assign dim hints
13. **coarse tile**
14. **span reduction** (enforce the 256 MB per-core device-memory span)
15. **cost-model matmul division + work distribution across the 32 cores**
16. **scratchpad (LX) planning**

### Work-division planning (three passes)

Span reduction → cost-model matmul division over `(b, m, n, k)` with
`cost = compute + hbm + psum + shape_penalties + batch` → work distribution across output dimensions and
then at most one reduction dimension, enforcing equal stick counts per core. Splits are recorded in
`op.op_it_space_splits` as index coefficients. The core budget is configurable via `SENCORES` (default 32).

### Scratchpad (LX) planning

Four solvers: **greedy** (default, chronological), **first-fit**, **best-fit**, and a **CP-SAT** solver that
models placement as 2-D non-overlapping rectangles over (lifetime × address), minimizing DDR traffic.
Buffers align to 128 B; the budget is ~1.6 MB of the 2 MB LX. **There is no eviction** by design. A
"core-division mismatch" between adjacent operations disqualifies a buffer from LX and routes it through
device DRAM.

Other documented passes: indirect access (gather), working-set reduction, coarse-tiling loop IR, and
span-overflow hint analysis.

Key modules: `views.py`, `op_spec.py`, `work_division.py`, `scratchpad/`, `coarse_tile.py`,
`propagate_named_dims.py`, `memory_planning.py`, `customops.py`, `decompositions.py`, `lowering.py`,
`spyre_kernel.py`, `codegen/`. **All Apache-2.0.**

---

## Layer 4 — Compiler↔Backend Interface IRs (published as open specs)

### SuperDSC / SDSC — the current production interface

"Super Design Space Config." Published openly as an interface spec even though its consumer (DeepTools) is
closed. A bundle is **MLIR + JSON**:

- **MLIR file** — control flow via `sdscbundle.sdsc_execute` (naming an `sdsc.json` file plus `symbol_ids`),
  `scf.for` loops, `arith.constant`, `affine.apply`.
- **`sdsc.json`** per operation:
  - `OpFunc` + data format
  - input/output tensor descriptors with memory location (**DDR/"HBM" vs. LX**)
  - **core work mapping**: `numWkSlicesPerDim_`, `coreIdToWkSlice_`, `dataStageParam_`
  - tensor placement with `isStartAddrSymbolic_`
  - **data layout**: stick configuration (which dims live in a stick and how many elements), layout beyond
    sticks, per-dim/per-core **back-gaps**, and scale factors (−1 reduction, −2 broadcast)
  - **folds**: affine `alpha*index + beta` parameterizations that compactly describe all 32 cores
- **Stick constraints** are operation-class-specific and ripple through the graph: e.g. a BatchMatMul output
  stick holds `generated_dim=64`; a DF16 input1 stick holds `reduction_dim=64`; an FP8/INT8 input2 stick
  holds `reduction_dim=2, generated_dim=64`.
- JSON was chosen because *"SuperDSC artifacts have to be diffable and inspectable during development."*

### KTIR — KernelTile IR, the announced successor

An **MLIR dialect (`ktdp`)** that IBM explicitly positions as a **community-aligned spec generalizable
across dataflow accelerators**, not Spyre-only. **RFC 0682 merged March 2026**; *"the spec is stable, the
reference interpreter is up, and the backend lowering path is in development"* — production still routes
through SuperDSC.

- **Hardware abstraction**: *"the accelerator contains multiple cores, with each core comprising a compute
  engine and an on-chip scratchpad memory,"* connected by an on-chip fabric to off-chip memory banks; each
  compute tile has a global view of all memory via the fabric.
- **Three-step memory access** — the core design idea:
  1. `construct_memory_view` / `construct_distributed_memory_view` — name a region (sizes, strides,
     coordinate sets, memory space)
  2. `construct_access_tile` / `construct_indirect_access_tile` — which slice this core touches
     (IntegerSets + AffineMaps; gather/scatter via index tensors)
  3. `ktdp.load` / `ktdp.store`

  This separates layout, work division and data movement so each can be reasoned about independently.
- Other ops: `get_compute_tile_id`, `runtime_arg_extract`, `inter_tile_produce` / `inter_tile_reduce`
  (all-reduce patterns).
- `SpyreMemorySpaceAttr` annotates memrefs with the memory level (**HBM** vs. **`LX, core=N`**), carrying
  compute/memory affinity.
- Reuses upstream `arith`, `math`, `linalg`, `scf` — computation and control flow are *not* reinvented.
- Distinctions from GPU models that IBM states explicitly: **persistent, compile-time-partitioned cores**
  (not thread blocks) and **explicit scratchpad management** (no implicit cache hierarchy).

**Companion open implementations:**

- **`ktir-mlir-frontend`** — MLIR parser + Python bindings (`mlir_ktdp`). CMake ≥3.20, C++17, LLVM/MLIR.
  Apache-2.0.
- **`ktir-cpu`** — CPU **interpreter, validator and latency model** for KTIR. Simulates a multi-core grid
  with two memory spaces per core (HBM 128 GB shared; **LX 2 MB per core**), NumPy execution, registry-based
  op dispatch. Explicitly framed as *"an environment and reward model for AI-driven compiler development."*
  **This makes hardware-free study of the Spyre programming model possible** — rare and valuable for a
  survey. Apache-2.0.
- **`dataflow-scheduler` + `dataflow-scheduler-mlir-dialects`** — C++/MLIR infrastructure that takes
  **KTIR + an architecture description file (MLIR)** and produces **DFIR (Dataflow IR)** through
  intermediate **KTDF / KTDFLow** schedule IRs. Strategy: partition at memory boundaries → fuse via
  `linalg.generic` → materialize a load→compute→store pipeline → refine (hierarchical routing through the
  memory hierarchy, loop tiling, hierarchical/sibling pipelines, LICM, **double buffering**) → parallelize
  across compute instances → normalize to a physical 1-D grid of hardware units → split into per-engine
  dataflow units. Single execution model: *"pipeline + parallel."* **Architecture-driven, not
  hardware-hardcoded**, with pluggable codegen backends. Requires LLVM `llvmorg-22.1.3`. Tool:
  `dataflow-scheduler -kEmitDFIR --device=<device.mlir> <input.mlir>`. Apache-2.0.

---

## Layer 5 — Kernel Language and Kernel Compiler

**Triton is the kernel language.** IBM maintains a **fork at `github.com/torch-spyre/triton`** (MIT,
tracking upstream via `sync-*` branches), because the Spyre lowering is not upstream.

**`spyre-kernels`** is *"the canonical home for authoring, validating, and tracking Triton kernels targeting
IBM Spyre hardware."* Flow: **Triton → TTIR (Triton IR) → KTIR → machine code**, driven by
`scripts/gen_ktir.py`. The repo commits the `.ttir` and `.ktir` artifacts alongside each `triton_kernel.py`,
organized per model (e.g. `kernels/models/Meta-Llama-3.1-8B-Instruct/torch.matmul.*`) and per vLLM
operation (`decode_softmax_reducev`, `embedding`, `log_softmax`, `matmul`, `merge_attn_states`, `mrope`,
`prefill_attention`, `reshape_and_cache`, `rms_norm`, `silu_and_mul`, `ranks`).

**Four validation tiers:**

| Tier | Check |
|------|-------|
| T0 | Numerical equivalence vs. PyTorch on GPU |
| T1 | Spyre-shape compliance — **tiles fit the scratchpad, grid fits 32 cores** |
| T2 | Correctness on the `ktir_cpu` simulator **and** on real hardware |
| T3 | Expert review |

The invariants are literally checked as documents in the repo: `grid-fits-32-cores.md`,
`tile-fits-scratchpad.md`, `runtime-arg-agnostic.md`. The repo also ships agent skills for LLM-assisted
kernel conversion. 🟡 No license file.

---

## Layer 6 — Back-end Compiler: DeepTools (proprietary)

*"A proprietary component called DeepTools, developed by IBM."* It runs **out-of-process** as
`dxp_standalone -d <output_dir>`, with outputs cached under the Inductor cache.

- **Input**: SuperDSC JSON bundle (future: KTIR).
- **Responsibilities**: dataflow mapping (SuperDSC ops → Spyre dataflows), **core scheduling across all 32
  cores**, and binary generation including the load/store sequences that stage data from LPDDR5 into LX.
- **Output: SpyreCode** — a container that is itself an **open spec** (`interface-specs/0277-SpyreCode`)
  even though the producer is closed. Four parts:
  1. **Job Execution Plan** — ordered runtime commands: `ComputeOnHost`, `ComputeOnDevice` (firmware control
     message → compute control blocks), `DataTransfer` (DMAI/DMAO control blocks)
  2. **Job Preparation Plan** — one-time `Allocate` (into SegmentId=7) and `InitTransfer` commands
  3. **Job binaries** (`init.bin`) — the programs for the Spyre compute cores
  4. **Host compute metadata** — drives **program correction**: symbol resolution and just-in-time binary
     patching of loop counts and addresses before each launch
- Internal passes, instruction formats and control-block encodings: **not disclosed**.

---

## Layer 7 — Tensor API and Layout Model

- **`SpyreTensorImpl`** — a C++ `at::TensorImpl` subclass carrying Spyre layout metadata beyond size/stride,
  including DCI translation data for CPU↔Spyre conversion.
- **`SpyreTensorLayout`** — `device_size` (the PyTorch size extended with padding and extra tiling dims),
  `stride_map` (host stride per device dim), `device_dtype`, and `element_arrangement` (packing within a
  stick; default `STANDARD`). Host↔device address identity:
  `offset = dot(device_coordinates, stride_map)`.
- **`FixedTiledLayout`** — the Inductor `Layout` subclass wrapping `SpyreTensorLayout`.
- **Stickification** — *"the transformation from a host-strided PyTorch layout to a tiled Spyre device
  layout."* A `(1024, 256)` fp16 tensor physically becomes `(4, 1024, 64)` on device — **not expressible
  with PyTorch strides**. `restickify` reconciles adjacent operations that disagree on tile structure.
- **DCI (Data Conversion Information)** — loop ranges, strides and dtype fed to the DMA engine for host-side
  transfers.
- Python/API surface: `.to("spyre")` (optionally with an explicit `SpyreTensorLayout`), `new_empty`,
  `new_empty_strided`, `device_tensor_layout()`. Default dtype is fp16.

---

## Layer 8 — Runtime

| Component | Role | Open source |
|-----------|------|-------------|
| **torch_spyre (C++/Python)** | `SpyreAllocator` (lazy chunked, CUDA-caching-allocator-style; virtual sub-allocation within large chunks so VF mode's limited backend handles are not exhausted); `SpyreGuardImpl` (`c10::impl::DeviceGuardImplInterface`); **`SpyreStream`** (PyTorch `Stream` interface, FIFO within a stream, no cross-stream ordering, **sticky error model** — a failed stream is unusable and must be recreated); **`JobPlan`** (cached, owns device allocations + pinned host buffers); `RuntimeOperation{H2D, D2H, Compute, HostCallback}`; `CompositeAddress`/`LogicalAddress` (`region_id` + 128 B-aligned offset). Entry points `PrepareKernel` (once per SDSC) and `LaunchKernel` (per invocation); `SPYRE_ALLOW_TILED_LAUNCH` enables transparent multi-iteration tiled dispatch when a tensor dim exceeds the compiled tile dim | ✅ Apache-2.0 |
| **DeepRT** | IBM device runtime that compiles/prepares operations into SpyreCode directories consumed by the JobPlan translator | ❌ proprietary |
| **Flex runtime** | The actual execution engine — `RuntimeStream.launchOperation()`, `FlexAllocator`, kernel launch, device communication, dispatch to the driver. PF/VF modes via `FLEX_DEVICE`; `FLEX_COMPUTE=SENTIENT` selects hardware | ❌ proprietary |
| **`torch_sendnn`** | The umbrella device stack. *"The full `torch_sendnn` stack is only available pre-installed in a base environment"* — installed system-wide, reached with `--system-site-packages`. A **CPU-only development path** exists (`uv pip install sendnn-inference` + CPU PyTorch) so the front end can be developed without hardware | ❌ not publicly distributed |
| **`spyre_comms` / `spyreccl`** | Collectives. `spyreccl` is a PyTorch `torch.distributed` backend (`init_process_group` → `dist.broadcast`, `dist.all_reduce`); blocking mode; **allreduce prioritized for tensor parallel**; **on-node (single server) only**, multi-node "may be added in the future"; up to 8 cards. Functional collectives currently compile through `torch.inductor`, with migration to `torch.distributed`/`torch.comms` planned | ❌ closed library, open backend shim |

---

## Layer 9 — Driver, Firmware, ISA

- **Kernel driver and card firmware are proprietary.** The driver exposes PF and VF (SR-IOV) functions; in VF
  mode memory is addressed via a firmware `region_id` lookup rather than physical addresses.
  `ComputeOnDevice` is realized as a **firmware control message** that generates compute control blocks;
  DMA is expressed as **DMAI/DMAO control blocks**.
- **ISA: not public.** No instruction set, encoding, or assembler is published. The `init.bin` job binaries
  produced by DeepTools are the only artifact, and their internal format is undocumented. The nearest public
  "contract" is the **SpyreCode** container spec, which describes *commands*, not *instructions*.
- **Export control** is cited in the RFCs as the reason the metrics tooling reads a vendor API rather than
  raw hardware counters — a hint at why the low layers stay closed.

---

## Layer 10 — Observability

| Layer | Tool | Open source |
|-------|------|-------------|
| Application | Spyre extension to the **PyTorch Profiler** via `REGISTER_PRIVATEUSE1_PROFILER` (`record`, `elapsed`, `synchronize`, `onEachDevice`); `kineto-spyre` wheel. *"Event completion"* on a dataflow architecture is defined as *"when all output tokens of a kernel have been written to their destination"* | ✅ |
| Compiler front end | Enhanced Inductor **provenance tracking** (pass-level) | ✅ |
| Compiler back end | IR-instrumentation profiler (intra-kernel) | mixed |
| Runtime | **`libaiupti`** — AIU Profiling Tools Interface (kernel + memory tracking) | ❌ |
| Hardware | **`aiu-smi`** over the **`spyremetrics`** API — the `nvidia-smi` analogue. Reports power, temperature, device-busy %, memory read/write bandwidth, PCIe ingress/egress, RDMA ops, request rates, reserved/actual/peak device memory, **PT-array utilization estimated from power**, and per-segment reserved memory. Works in PF and VF modes on **x86, Power and Z** | ❌ (open RFC) |
| Post-processing | **`aiu-trace-analyzer`** — derived metrics | ✅ |

Profiler metrics with no von Neumann analogue: pipeline utilization, DMA overhead, reconfiguration latency,
inter-core communication efficiency, **stick alignment overhead**, three-level parallelism analysis, and —
for the compiler-managed LX — **peak/average scratchpad utilization, fragmentation ratio, allocation
efficiency**. RFC 0601 explicitly contrasts the **dual memory hierarchy**: device DRAM is
*runtime-allocator-managed, observable at runtime only*; the scratchpad is *managed by the compiler's LX
planner, observable at compile time and runtime*.

---

## Layer 11 — Cluster / Orchestration

The `github.com/ibm-aiu` org, all Apache-2.0 and written in Go, all actively pushed as of August 2026:

`spyre-operator` (OpenShift operator for Spyre cards) · `spyre-device-plugin` (Kubernetes device plugin) ·
**`dra-driver-spyre`** (Kubernetes Dynamic Resource Allocation driver) · `spyre-health-checker` ·
`spyre-scheduler-plugins` (out-of-tree scheduler framework plugins) · `spyre-webhook-validator` ·
`certified-operators` (Red Hat certified operator bundles) · `spyre-operator-docs`.

This is the Red Hat OpenShift deployment story IBM's press release gestures at, and it is fully in the open.

---

## Open-Source Summary

| Open (Apache-2.0 unless noted) | Closed |
|--------------------------------|--------|
| `torch-spyre` (PyTorch PrivateUse1 backend + Inductor front end), `sendnn-inference` & `spyre-inference` (vLLM plugins), `hf-adapters`, `sglang-spyre` (no license), `aiu-bench`, `ktir-mlir-frontend`, `ktir-cpu`, `dataflow-scheduler`, `dataflow-scheduler-mlir-dialects`, `triton` fork (MIT), `spyre-kernels` (no license), `interface-specs`, `RFCs` (no license), the entire `ibm-aiu` Kubernetes/OpenShift tooling, `aiu-trace-analyzer` | **DeepTools** back-end compiler (`dxp_standalone`), **DeepRT**, **Flex** runtime, **`torch_sendnn`** distribution, **`spyre_comms`**, **`libaiupti`**, **`spyremetrics`/`aiu-smi`** implementation, kernel **driver**, card **firmware**, the **ISA** |

---

## Not Disclosed

- DeepTools internals: pass list, dataflow-mapping algorithms, core-scheduling heuristics, source
- Flex runtime and DeepRT source, APIs and internal architecture
- Kernel driver and card firmware source, ioctl/ABI surface, firmware control-message format
- `spyre_comms` collective implementation, algorithms and topology awareness
- The Spyre ISA, instruction encoding, and the `init.bin` job-binary format
- Whether the `torch_sendnn` runtime will ever be publicly distributed — the docs say installation
  instructions arrive "once the runtime reaches public availability"
- Whether Spyre is reachable natively from z/OS (e.g. via Machine Learning for z/OS, zDLC, or any ONNX
  path) — no public documentation of such a flow exists; zDLC/zDNN target Telum/Telum II NNPA only

---

## Sources

- [torch-spyre developer docs (root)](https://torch-spyre.readthedocs.io/en/latest/)
- [torch-spyre docs — Compiler stack overview (open vs proprietary)](https://torch-spyre.readthedocs.io/en/latest/compiler/architecture.html)
- [torch-spyre docs — Inductor front end](https://torch-spyre.readthedocs.io/en/latest/compiler/inductor_frontend.html)
- [torch-spyre docs — Back-end compiler (DeepTools)](https://torch-spyre.readthedocs.io/en/latest/compiler/backend.html)
- [torch-spyre docs — KTIR in the pipeline](https://torch-spyre.readthedocs.io/en/latest/compiler/ktir.html)
- [torch-spyre docs — Scratchpad (LX) planning](https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html)
- [torch-spyre docs — Work-division planning](https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html)
- [torch-spyre docs — Tensor layouts](https://torch-spyre.readthedocs.io/en/latest/user_guide/tensors_and_layouts.html)
- [torch-spyre docs — Runtime overview](https://torch-spyre.readthedocs.io/en/latest/runtime/index.html)
- [torch-spyre docs — Installation](https://torch-spyre.readthedocs.io/en/latest/getting_started/installation.html)
- [torch-spyre/torch-spyre](https://github.com/torch-spyre/torch-spyre)
- [torch-spyre/sendnn-inference](https://github.com/torch-spyre/sendnn-inference)
- [torch-spyre/spyre-inference](https://github.com/torch-spyre/spyre-inference)
- [torch-spyre/hf-adapters](https://github.com/torch-spyre/hf-adapters)
- [torch-spyre/triton](https://github.com/torch-spyre/triton)
- [torch-spyre/spyre-kernels](https://github.com/torch-spyre/spyre-kernels)
- [torch-spyre/ktir-mlir-frontend](https://github.com/torch-spyre/ktir-mlir-frontend)
- [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu)
- [torch-spyre/dataflow-scheduler](https://github.com/torch-spyre/dataflow-scheduler)
- [torch-spyre/interface-specs](https://github.com/torch-spyre/interface-specs)
- [SuperDSC Bundle spec](https://github.com/torch-spyre/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md)
- [SpyreCode spec](https://github.com/torch-spyre/interface-specs/blob/main/0277-SpyreCode/0277-SpyreCodeSpec.md)
- [ProgramExecution spec](https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/ProgramExecutionSpec.md)
- [RFC 0099 — Multi-Device](https://github.com/torch-spyre/RFCs/blob/main/0099-MultiDevice/0099-MultiDeviceRFC.md)
- [RFC 0171 — Spyre Device](https://github.com/torch-spyre/RFCs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md)
- [RFC 0601 — Spyre Profiling Toolkit](https://github.com/torch-spyre/RFCs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md)
- [RFC 0682 — KTIR Spec](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)
- [vLLM Spyre plugin docs](https://docs.vllm.ai/projects/spyre/en/latest/)
- [vLLM Spyre — supported features](https://docs.vllm.ai/projects/spyre/en/latest/user_guide/supported_features.html)
- [foundation-model-stack/aiu-fms-testing-utils](https://github.com/foundation-model-stack/aiu-fms-testing-utils)
- [ibm-aiu org (Kubernetes/OpenShift tooling)](https://github.com/ibm-aiu)
- [IBM/zDNN — Telum-only, negative finding](https://github.com/IBM/zDNN)
- [IBM/zDLC — Telum-only, negative finding](https://github.com/IBM/zDLC)
- [IBM Research — PyTorch support for IBM Spyre](https://research.ibm.com/blog/pytorch-support-ibm-spyre)
