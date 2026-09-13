# Cerebras WSE Software Stack — Investigation Report

*as_of: 2026-04-05*
*chip: cerebras*
*device_class: Wafer-Scale Engine*
*primary sources: https://training-api.cerebras.ai/, https://sdk.cerebras.net/, https://github.com/Cerebras/modelzoo, https://www.cerebras.ai/blog/supporting-pytorch-on-the-cerebras-wafer-scale-engine, https://training-api.cerebras.ai/en/rel-2.3.1/original/compiler-reports/compile-report.html*

---

## Overview

The Cerebras software stack is a vertically integrated, PyTorch-centric system for training and inference on the WSE-3. At the top sits a standard PyTorch 2.0 API with a Cerebras-specific Lazy Tensor Core (LTC) backend that captures model computation graphs without eager execution. These graphs are lowered through CIRH (Cerebras Intermediate Representation — High), an MLIR-based IR close to ATen, where operator fusion, constant folding, and memory optimization passes are applied. The compiler then partitions the graph across 900K PEs, generates per-PE CSL binaries, and embeds a weight-streaming schedule determining whether to use Layer Pipelined or Weight Streaming execution. The runtime handles the data-loading pipeline (MemDataLoader), weight dispatch from MemoryX via SwarmX, and gradient collection. Below the PyTorch path, the Cerebras SDK exposes a lower-level environment where developers write CSL (Cerebras Software Language) kernels directly — a Zig-inspired, compile-time-powerful language where computation is dataflow-driven: tasks activate when wavelets arrive on colors. The Model Zoo provides YAML-driven reference implementations of major LLMs including Llama, Mixtral, GPT-3, and multimodal models, making the stack immediately usable for large-scale training without custom kernel development.

---

## Architecture

### Stack Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    USER / RESEARCHER LAYER                           │
│  Python training script  OR  YAML config (Trainer class)            │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               FRAMEWORK INTEGRATION                                  │
│  cerebras.pytorch (cstorch)  — PyTorch 2.0 LTC backend              │
│  • cstorch.compile(model, backend="CSX")                            │
│  • cstorch.Trainer(params, optimizer, lr_scheduler, ...)            │
│  • Lazy tensor capture: no eager execution; builds computation graph │
│  • Sparsity Library: GMP, SRigL unstructured sparse training         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  PyTorch ATen ops → CIRH lowering
┌──────────────────────────▼───────────────────────────────────────────┐
│               COMPILER / IR                                          │
│  CIRH (Cerebras IR — High-level, MLIR-based, close to ATen)         │
│  Passes:                                                             │
│  • Operator fusion (fused attention, fused LayerNorm+dropout, etc.)  │
│  • Constant folding, dead code elimination                           │
│  • Memory layout optimization for 44 GB SRAM distribution           │
│  • AutoGen: auto-generates CSL kernels for standard ops              │
│  • Execution mode selection: Layer Pipelined vs Weight Streaming     │
│  Output: per-PE CSL binaries + execution schedule                   │
│                                                                      │
│  Compile Report: layer-by-layer analysis, utilization, projections  │
│  Incremental Compile: re-uses prior artifacts when structure same   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  CSL binaries → PE array
┌──────────────────────────▼───────────────────────────────────────────┐
│               OP LIBRARY / MODEL ZOO                                 │
│  github.com/Cerebras/modelzoo                                        │
│  • Llama 2/3, GPT-2/3, Mistral, Mixtral, T5, BERT, DINOv2, LLaVA   │
│  • YAML-driven Trainer configs (dataset, model, optimizer, lr)      │
│  • checkpoint conversion tools (Cerebras ↔ HuggingFace)            │
│  • Sparsity extensions: GMP, SRigL pruning in training loop         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               KERNEL LIBRARY (CSL / SDK)                             │
│  sdk.cerebras.net — Cerebras SDK 1.4.0                               │
│  CSL (Cerebras Software Language):                                   │
│  • Zig-inspired; compile-time metaprogramming for PE kernels         │
│  • Dataflow model: tasks activate on wavelet arrival on a color      │
│  • 24 colors per PE (virtual channels); 32-bit wavelets              │
│  • DSDs: Data Structure Descriptors — typed, strided SRAM views      │
│  • <collectives_2d>: reduce/broadcast across 2D PE mesh              │
│  • <message_passing>: explicit PE-to-PE point-to-point (WSE-3 only) │
│  • github.com/Cerebras/sdk-examples: GEMM, stencils, custom ops     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               RUNTIME                                                │
│  SdkRuntime (sdk.cerebras.net/api-docs/sdkruntime-api)              │
│  • Compile + load CSL binary onto CS-3                               │
│  • Launch execution; manage symbol I/O                               │
│  • memcpy to/from device (host ↔ PE SRAM)                           │
│                                                                      │
│  Cerebras Runtime (training path):                                   │
│  • MemDataLoader: prefetch + batch data pipeline                     │
│  • Weight streaming scheduler: dispatches layer weights from MemoryX │
│  • Gradient collection: streams gradients out via SwarmX             │
│  • SDK Appliance API: multi-job deployment, CS-3 lifecycle mgmt      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               DRIVER / FIRMWARE                                      │
│  • csctl CLI: job submission, queue management, monitoring           │
│  • Grafana dashboards: cluster health, job progress, utilization     │
│  • Slurm integration: HPC cluster scheduler support                  │
│  • Appliance API: device allocation, program lifecycle               │
│  • Compile server: dedicated host for graph compilation (separate)  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               COMMUNICATION                                          │
│  On-chip (Swarm 2D mesh via CSL):                                    │
│  • <collectives_2d>: all-reduce, broadcast, scatter/gather           │
│  • <message_passing>: point-to-point PE messaging (WSE-3)           │
│                                                                      │
│  Multi-replica scale-out (SwarmX):                                   │
│  • cerebras.pytorch.distributed — multi-replica data parallel API   │
│  • Gradient all-reduce semantics over SwarmX fabric                 │
│  • Near-linear scaling: demonstrated to 192 CS-2s                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│               HARDWARE (WSE-3)                                       │
│  900,000 PEs | 44 GB SRAM | 125 PF FP16 | Swarm 2D mesh            │
│  46,225 mm² | TSMC 5nm | 4T transistors                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Abstractions

| Abstraction | Description |
|---|---|
| **cerebras.pytorch (cstorch)** | PyTorch 2.0 LTC backend API. `cstorch.compile()` captures the full model graph. `cstorch.Trainer` wraps training loop with weight streaming management. |
| **CIRH** | Cerebras Intermediate Representation (High-level). MLIR-based, ATen-aligned IR. Compiler input for optimization and PE-assignment passes. Not user-visible. |
| **AutoGen** | Automatic kernel generation pass: derives CSL kernels from CIRH op patterns. Eliminates manual CSL writing for standard neural network ops. |
| **CSL (Cerebras Software Language)** | Zig-inspired low-level kernel language for PE programming. Dataflow model: tasks + colors + wavelets. Exposes DSDs, collectives, message passing. |
| **Data Structure Descriptor (DSD)** | CSL type for defining strided memory views over PE-local SRAM. All SRAM access in CSL kernels goes through DSDs. Enables efficient N-D tensor iteration without pointer arithmetic. |
| **Task / Color / Wavelet** | CSL execution primitives: a Task is a function activated by wavelet arrival on its Color; a Wavelet is a 32-bit message containing data + color tag. Fundamental CSL dataflow unit. |
| **Trainer (YAML config)** | High-level training orchestrator: instantiated from a YAML file specifying model, dataset, optimizer, lr_scheduler, callbacks. Abstracts compile/run/checkpoint lifecycle. |
| **MemDataLoader** | Cerebras runtime data pipeline: prefetches batches from storage, formats for streaming into PE memory. Must be used instead of standard PyTorch DataLoader. |
| **Weight Streaming Scheduler** | Runtime component in Cerebras Runtime that coordinates layer-weight dispatch from MemoryX → SwarmX → CS-3 PEs, pipelined with compute. |

---

## Data Flow

### PyTorch Training — Weight Streaming Mode (Large LLM)

```python
# User code (simplified Cerebras training script)
import cerebras.pytorch as cstorch

model = LlamaForCausalLM(config)                    # standard torch.nn.Module
compiled_model = cstorch.compile(model, backend="CSX")   # LTC graph capture
optimizer = cstorch.optim.AdamW(...)
trainer = cstorch.Trainer(model=compiled_model, optimizer=optimizer, ...)

# Launch training loop
trainer.train(train_dataloader=MemDataLoader(...))
```

### Compiler Pipeline (inside `cstorch.compile`)

```
1. LTC (Lazy Tensor Core) intercepts PyTorch ATen ops
   - Model runs lazily: no computation yet, builds IR graph
   - All ops recorded as ATen dialect nodes

2. ATen → CIRH lowering
   - Custom XLA/HLO wrappers replaced by direct ATen → CIRH mapping
   - CIRH is MLIR-based, stays close to PyTorch abstraction level

3. CIRH optimization passes:
   a. Operator fusion: FlashAttention pattern, LayerNorm+dropout, GeLU+proj
   b. Constant folding + dead code elimination
   c. Memory placement: assign activation tensors to PE SRAM regions
   d. Execution mode analysis: Layer Pipelined if total params ≤ 44GB, else Weight Streaming
   e. AutoGen: generate CSL kernels for unfused ops automatically

4. PE assignment:
   - Compiler maps CIRH graph nodes to PE regions (compile-time spatial layout)
   - Routing table generated: which colors carry which tensors between PE groups

5. Output:
   - Per-PE CSL binaries (one binary per PE region type)
   - Execution schedule: layer order, weight streaming timeline, gradient collection points
   - Compile report: layer-by-layer utilization, projected throughput, memory usage

6. Binary loaded to CS-3 via SdkRuntime / Appliance API
```

### CSL Kernel Dataflow (one PE, during forward pass)

```csl
// Conceptual CSL task for a single PE in a linear layer
const input_color: color = @get_color(0);   // color 0 carries input activations
const output_color: color = @get_color(1);  // color 1 carries output activations

// Task activates when a wavelet arrives on input_color
task forward_task(wavelet: f16) void {
  // Accumulate into local SRAM using DSD
  acc_dsd += @as(f16, wavelet) * weight_dsd;  // FP16 MAC with SIMD
  if (last_input) {
    // Send result to next-layer PE via fabric
    @mov32(output_queue, @as(u32, acc + bias));
  }
}

comptime { @bind_task(forward_task, input_color); }  // bind task to color
```

---

## Hardware Interface Points

| Software Component | Hardware Target | Interface Mechanism |
|---|---|---|
| `cstorch.compile()` (LTC) | PE array (via compiler) | Produces PE-specific CSL binary; compiler embeds SRAM layout |
| `SdkRuntime.load()` | CS-3 host PCIe | Binary transferred via PCIe to CS-3; loaded into PE instruction memory |
| `SdkRuntime.memcpy_h2d()` | PE SRAM via host interface | Batch data injected as wavelet stream at mesh boundary |
| Weight Streaming Scheduler | MemoryX → SwarmX → PE SRAM | RoCE RDMA DMA; layer weights broadcast to PE SRAM before each layer |
| `<collectives_2d>` (CSL) | Swarm 2D mesh (hardware routers) | Uses mesh colors for reduce/broadcast; no software switch |
| `cerebras.pytorch.distributed` | SwarmX fabric | Gradient all-reduce over SwarmX tree; managed by Cerebras Runtime |
| `csctl` | CS-3 appliance daemon | REST API + Slurm hooks; manages job queues and device allocation |

---

## Key Findings

1. **LTC over XLA was a deliberate long-term compatibility choice.** Cerebras migrated from XLA (shared with JAX/TF) to PyTorch's Lazy Tensor Core in release 2.0. LTC is part of standard PyTorch core, so Cerebras can track upstream PyTorch releases without maintaining a fork. The migration also removed the dependency on XLA's HLO representation, switching to a direct ATen → CIRH path.

2. **CIRH is MLIR-based and ATen-aligned — not a custom IR dialect.** Cerebras explicitly chose to stay close to ATen semantics at the IR level to minimize the operator lowering surface. The compiler performs fusions (FlashAttention, GeLU+Linear, LayerNorm+dropout) as CIRH-level graph rewrites before CSL code generation.

3. **AutoGen eliminates manual CSL writing for training users.** For standard ops (matmul, attention, normalization), the compiler automatically generates optimized CSL kernels via AutoGen. Direct CSL programming is only needed for custom scientific computing workloads (stencils, PDEs, custom ops) accessed through the Cerebras SDK path.

4. **Two programming entry points with different abstraction levels.** (a) High-level: PyTorch + `cerebras.pytorch` + Model Zoo YAML configs — zero WSE-specific code needed for standard LLM training. (b) Low-level: CSL via Cerebras SDK — full access to PE memory, wavelet colors, and mesh routing for custom kernels. The two paths are not directly composable; custom CSL ops must be wrapped and imported.

5. **YAML-driven Trainer class is the production training interface.** `cstorch.Trainer` with a YAML config file specifying model, dataset, optimizer, lr_scheduler, and mixed precision settings is the recommended path for large-scale LLM training. This matches the Model Zoo's 40+ model configurations.

6. **Compile time is significant and must be planned for.** Cerebras compilation maps the entire model graph onto 900K PEs at compile time — this is not a JIT cache. The Incremental Compile feature re-uses prior compilation artifacts when model structure is unchanged (same graph, different hyperparameters). The Compile Report provides per-layer analysis to guide architectural decisions before long training runs.

7. **`cerebras.pytorch.distributed` provides multi-replica data parallelism via SwarmX.** Unlike NCCL (which coordinates GPU-to-GPU all-reduce via NVLink/IB), Cerebras distributed training routes gradients through SwarmX's tree broadcast-reduce. The API is intentionally similar to `torch.distributed` to ease migration.

8. **Stack is mostly closed above CSL.** The Model Zoo is open-source (Apache 2.0). The PyTorch API surface is documented as an SDK release. The CIRH compiler and hardware binaries are proprietary. CSL itself is a documented public language with a public SDK but the CSL-to-hardware code generation backend is closed.

---

## Openness Assessment

| Component | Status | Notes |
|---|---|---|
| Model Zoo (modelzoo) | Open source (Apache 2.0) | github.com/Cerebras/modelzoo; 40+ model configs |
| Cerebras SDK / CSL | Public SDK (docs + toolchain) | sdk.cerebras.net; sdk-examples open source |
| cerebras.pytorch API | SDK release (documented, pip-installable) | pypi: cerebras-pytorch |
| CIRH compiler | Closed | Internal to Cerebras; outputs not user-visible |
| Hardware binary format | Closed | PE binaries are proprietary |
| Inference API | Cloud service (OpenAI-compatible REST) | inference-docs.cerebras.ai |

---

## Relation to Software Stack Layers

| Layer | Cerebras Component | Notes |
|---|---|---|
| Framework Integration | cerebras.pytorch (cstorch), PyTorch 2.0 LTC backend | Standard torch.nn.Module; cstorch.compile() entry point |
| Compiler / IR | CIRH (MLIR/ATen-based), AutoGen, Compile Report | Operator fusion, PE assignment, weight-streaming schedule generation |
| Op Library | Model Zoo (40+ LLMs/vision), Sparsity Library | YAML-driven; GMP+SRigL sparse training; HF checkpoint conversion |
| Kernel Library | CSL (tasks/colors/wavelets/DSDs), sdk-examples | Low-level PE kernel language; `<collectives_2d>`, `<message_passing>` |
| Runtime | SdkRuntime API, Appliance API, MemDataLoader, Weight Streaming Scheduler | Load/launch/memcpy lifecycle + weight dispatch coordination |
| Driver / Firmware | csctl, Grafana, Slurm integration, compile server | Job submission, cluster monitoring, HPC scheduler hooks |
| Communication | `<collectives_2d>`, `<message_passing>`, `cerebras.pytorch.distributed` | On-chip mesh collectives + multi-replica SwarmX gradient reduction |
| Assembler / ISA | CSL Language Guide, DSDs, color/wavelet micro-ISA | Wavelet = 32-bit message; color = virtual channel; task = activation handler |

---

# Investigation Update — 2026-08-08 (roadmap scan, window 2026-04-01 → 2026-08-08)

*as_of: 2026-08-08*
*scan type: roadmap / landscape refresh, adversarially verified*

## Headline: no verified change to the SDK, compiler, or runtime in this window

The 2026-04-05 stack baseline above — `cerebras.pytorch` (cstorch) on PyTorch 2.0 LTC, CIRH (MLIR/ATen) with AutoGen, CSL + `<collectives_2d>` / `<message_passing>`, SdkRuntime / Appliance API, MemDataLoader, the Weight Streaming Scheduler, csctl — **stands unchanged**. No new software release, compiler IR, kernel language feature, or runtime component was confirmed in the 2026-04-01 → 2026-08-08 window. The documented versions remain rel-2.7.0 / SDK 1.4.0 as recorded above; no newer release was verified in this pass (this is "not re-verified", not "confirmed unchanged upstream").

## The one software-relevant architectural item: disaggregated prefill/decode serving

The AMD Helios + WSE disaggregated inference product (**announced 2026-07-23**, expected availability H2 2026 initially through Cerebras Cloud — see the hardware investigation update for full detail and sourcing) has an obvious software surface, but **none of it has been published**:

| Software question | Status |
|---|---|
| API/SDK for submitting a request that spans both tiers | **not disclosed** — presumably behind the existing OpenAI-compatible Cerebras Inference REST API, but this is not stated by either vendor |
| Scheduler / placement policy deciding prefill vs decode residency | **not disclosed** |
| KV-cache serialization format handed from Helios to the WSE | **not disclosed** |
| Whether ROCm and the Cerebras stack interoperate directly, or are joined only by a serving-layer protocol | **not disclosed** |
| Compiler involvement (does CIRH participate in partitioning, or is the split purely a serving-layer decision?) | **not disclosed** |

Because the entire mechanism is undisclosed, this update adds only a *placeholder* row to `chips/cerebras/layer-table.md` under Runtime, explicitly marked "not disclosed", rather than describing a stack that has not been published. Nothing here should be treated as an implementation claim.

Note also that the workload used in AMD's performance footnote — **Kimi 2.6 1T** — is a trillion-parameter model, consistent with the decode tier operating in weight-streaming mode rather than layer-pipelined mode, but neither vendor states the execution mode. Do not infer it.

## Product/blog items seen but NOT verified in this pass

The following surfaced in the raw scan of cerebras.ai/blog but were **not independently corroborated** during verification. They are recorded here as leads for the next scan, and are deliberately **not** written into `chips/cerebras/summary.md` or `layer-table.md` as confirmed capability:

| Date | Item | Potential stack layer touched |
|---|---|---|
| 2026-05-06 | Multi-LoRA on Cerebras Inference | Runtime / serving (adapter multiplexing) |
| 2026-05-19 | Trillion-parameter inference with Kimi K2.6 for enterprises | Runtime / weight streaming at inference |
| 2026-06-29 | Gemma 4 multimodal inference | Op library / Model Zoo coverage |
| 2026-07-10 | Upstage partnership (South Korea) | Ecosystem / model availability |
| 2026-05-26 | Sovereign-AI positioning post | Deployment, not stack |
| 2026-06-01 | Rest of World piece on an India–UAE AI partnership involving G42 and Cerebras | Deployment, not stack |

Of these, **Multi-LoRA** is the one with a genuine layer-table implication if confirmed (a runtime-level adapter-multiplexing capability on the inference path). It should be the first target of the next software scan.

## Follow-ups

1. Re-scan `sdk.cerebras.net` and `training-api.cerebras.ai` for a release newer than rel-2.7.0 / SDK 1.4.0 — the current version record is stale-by-default, not confirmed-current.
2. Verify Multi-LoRA and the Kimi K2.6 trillion-parameter inference path against cerebras.ai/blog and inference-docs.cerebras.ai.
3. Watch `inference-docs.cerebras.ai` in H2 2026 for any API surface exposing the AMD Helios prefill tier.
4. The Hot Chips 38 rack-scale talk (2026-08-25, J.P. Fricker) may carry software/runtime content — *disclosure scheduled, content not yet public*. Re-scan after 2026-08-25.
