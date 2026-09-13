# Etched Sohu — Software Stack Investigation

*as_of: 2026-08-08*
*Baseline investigation dated 2026-04-05 retained below in full; see "§12. 2026-08-08 Update" at the end.*

## Summary

The Etched Sohu software stack is intentionally minimal by design. Because the hardware only runs transformer inference, the software only needs to translate a transformer model into the fixed Sohu execution pipeline. There is no general-purpose compiler, no kernel programming interface, and no user-facing ISA. Etched has described "a few lines of code" to port a model — but the full SDK has not been publicly released as of Q1 2026.

---

## 1. Programming Model Philosophy

Sohu's programming model reflects its hardware philosophy: **the hardware IS the computation, so the software's job is model mapping, not kernel programming**.

On NVIDIA, a developer programs at multiple levels:
- Framework (PyTorch) → compiler (torch.compile, TensorRT) → kernel library (cuDNN, CUTLASS) → runtime (CUDA RT) → driver → GPU

On Sohu, the stack collapses:
- Framework (PyTorch/ONNX) → **Transformer compiler** → hardware

There is no kernel library, no user-facing runtime API, no instruction set to program against. The transformer compiler translates the model directly to the Sohu execution pipeline.

---

## 2. Framework Integration (Inferred)

**PyTorch**
- Models defined in PyTorch are exported (likely via torch.export or torch.onnx.export) for the Sohu compiler
- The compiler traces the computation graph and maps it to Sohu's hardwired pipeline stages
- Confidence: inferred from Etched's public statements about "a few lines of code to port models"

**ONNX**
- ONNX is the natural interoperability format; likely accepted as an alternative entry point
- Confidence: inferred

**TensorFlow / JAX**
- Not mentioned; likely supported via ONNX export path if needed
- Confidence: speculative

---

## 3. Transformer-Specific Compiler

**What the Compiler Does**
Since Sohu's hardware directly implements the transformer computation graph, the compiler's job is:

1. **Model ingestion**: Accept PyTorch or ONNX model definition
2. **Architecture validation**: Verify the model conforms to a supported transformer variant (standard MHA, grouped-query attention, MoE, etc.)
3. **Weight packing**: Reshape and pack weight tensors to match Sohu's internal memory layout and HBM3E access patterns
4. **Pipeline mapping**: Map model layers to Sohu's hardwired execution pipeline stages
5. **KV-cache configuration**: Configure attention hardware for target context length and batch size
6. **Binary generation**: Produce a Sohu execution binary (format undisclosed)

**What the Compiler Does NOT Do**
- It does not schedule individual operations (operations are hardwired)
- It does not generate instruction streams (there are no user-programmable instructions)
- It does not perform operator fusion (hardware already performs all transformer ops in a fused pipeline)
- It cannot handle non-transformer operations

**Confidence**: Inferred from architectural constraints and Etched's public statements. No compiler SDK has been released.

---

## 4. Supported Transformer Variants

| Variant | Status |
|---------|--------|
| Dense transformer (standard MHA) | Supported (primary target) |
| Grouped-query attention (GQA) | Likely supported (required for modern LLMs like Llama) |
| Multi-query attention (MQA) | Likely supported |
| Mixture-of-Experts (MoE) | Supported via separate variant/configuration |
| Sliding window attention | Uncertain |
| Non-transformer architectures (SSMs, Mamba, CNNs) | Not supported |

---

## 5. Runtime

The Sohu runtime is likely a thin layer that:
- Loads the compiled model binary
- Manages HBM3E weight loading
- Handles batching and token streaming
- Interfaces with host CPU / server orchestration

Given the hardware's fixed pipeline, there is no dynamic dispatch, no kernel selection, and no scheduling overhead in the runtime. The runtime is essentially a model loader and I/O manager.

---

## 6. Serving / API Layer

- Likely: OpenAI-compatible HTTP endpoint for LLM inference (industry standard)
- Etched's revenue model is selling server systems; customers would integrate via standard API
- No public API disclosed as of Q1 2026

---

## 7. Driver / Host Interface

- PCIe host interface likely (standard server form factor assumed)
- Kernel-mode driver for host-to-chip communication (not disclosed)
- No firmware described (unlike NVIDIA's GSP; a fixed-pipeline ASIC has no on-chip firmware runtime)

---

## 8. Development / Debugging Tools

- No public SDK or developer toolchain released
- No profiler disclosed
- The hardware's fixed pipeline means traditional profiling (kernel-by-kernel timing) does not apply
- Developer-facing tools likely focus on model compatibility checking and performance estimation

---

## 9. Software Stack Layers (Summary)

```
User code (Python, REST API)
         ↓
Model definition (PyTorch / ONNX)
         ↓
Etched Transformer Compiler
  - Architecture validation
  - Weight packing
  - Pipeline mapping
  - Binary generation
         ↓
Sohu Execution Binary
         ↓
Etched Runtime (thin: model loader + I/O)
         ↓
PCIe Host Driver
         ↓
Sohu Hardware (hardwired transformer pipeline)
  - Attention engine
  - FFN engine
  - Layer norm
  - Linear projection
  [All running on HBM3E]
```

---

## 10. What Does Not Exist (Intentionally)

| GPU Component | Sohu Equivalent |
|---------------|-----------------|
| CUDA / HIP / PTX (user ISA) | None — no user-programmable ISA |
| cuDNN / MIOpen (op library) | None — ops are hardware, not library |
| CUTLASS / GEMM kernels | None — attention is hardwired, not a kernel |
| NCCL (collective comm lib) | None disclosed — inter-chip comms handled by hardware |
| Device memory API | None — memory managed by compiler/runtime |

---

## 11. Key Open Questions (as of Q1 2026)

- Has any public SDK or documentation been released?
- What is the exact compiler input format?
- What transformer variants beyond dense MHA and MoE are supported?
- How does the compiler handle models with non-standard ops (custom activations, etc.)?
- Is there a model compatibility checker tool?
- What is the inter-chip communication model for 8-chip server?
- What is the serving API / endpoint format?

---

## 12. 2026-08-08 Update — What the June/July 2026 Posts Say About Software

*Investigated 2026-08-08. §1–§11 above are the 2024–Q1 2026 baseline and are retained unchanged.*

**Headline: almost nothing changed, and that is itself the finding.** Etched's two 2026 posts
(`/progress/frontier-inference-clusters`, 2026-06-30; `/progress/accelerating-inference`, 2026-07-23) are
hardware- and business-focused. There is still **no public SDK, no compiler, no runtime documentation, no
profiler, no API reference, and no open-source repository** as of 2026-08-08.

### 12.1 Confirmed as still true

| Baseline claim | Status 2026-08-08 |
|---|---|
| No public SDK or developer toolchain | Still true — site checked 2026-08-08 |
| No open-source compiler, runtime, or driver | Still true |
| No published serving API / endpoint format | Still true |
| No profiler or debugging tooling disclosed | Still true |
| No user-programmable ISA disclosed | Still true (no contrary statement) |

### 12.2 New software signals (all weak, all vendor-sourced)

**1. Software is now named as a co-design axis.** Etched markets co-designing "chips, racks, **software**, and
manufacturing methods". This elevates software from "intentionally minimal" (the 2024 framing) to a named pillar
— but no component is described. *Source: /progress/frontier-inference-clusters, 2026-06-30. Confidence:
confirmed as vendor wording; zero technical content.*

**2. "Recursive kernel generation" — the one genuinely surprising phrase.** The 2026-07-23 post closes its
hiring pitch with: *"thousand-chip scale-up domains, in-house SMT lines, and new RL environments for recursive
kernel generation."*

This matters for the survey because the repo's software-stack model for this chip is built on the assertion that
there are **no kernels** — the layer table records "Op Library: not applicable", "Kernel Library: not
applicable", "Assembler / ISA: not applicable", on the grounds that transformer ops are hardwired silicon rather
than dispatched software. A company hiring people to build RL environments for *generating kernels* implies
some programmable or compiler-scheduled surface exists.

Three readings, none confirmed:
- (a) "Kernel" is used loosely for compiler-emitted schedules/tilings, not for GPU-style programmable kernels.
- (b) The Gen 2 part is more programmable than Sohu was described to be, consistent with the broader
  "frontier inference systems" framing covering both prefill and decode.
- (c) It refers to using LLMs/RL to generate kernels for *other* platforms as an internal tool, unrelated to the
  product.

**Recorded as a signal, not as evidence.** Do not change the "no kernel library" layer rows on the strength of a
hiring sentence; flag the tension in the layer table (done) and revisit when Etched publishes anything technical.
*Source: /progress (2026-07-23). Confidence: claimed-unverified.*

**3. A VP of Software exists.** `etched.com/join` lists **David Munday, VP Software**, alongside Brian Loiler
(VP Platform) and Wayne Cao (VP Production). A named VP-level software org is consistent with more than a thin
loader, but no product, tool, or stack layer is described. *Source: /join, retrieved 2026-08-08. Confidence:
confirmed (roster), inferential (implication).*

**4. Communication layer now has a named hardware substrate.** Cluster Scale Memory's "proprietary
ultra-low-latency, high-bandwidth interconnect" exposes "a shared memory pool across our scale-up domain". If
the pool is memory-semantic, the software consequence is significant: a thousand-chip domain with load/store
reachability changes what a collectives library needs to do, and could explain the absence of any NCCL-analogue
in Etched's public material. **Etched describes no API, no collectives library, no memory model, and no
programming interface for this.** *Sources: /progress/frontier-inference-clusters (2026-06-30); TechCrunch
2026-07-23 paraphrase. Confidence: claimed-unverified for the hardware; the software implication is inference
only.*

**5. Workload scope widened.** The platform is now marketed for **both prefill and decode**, many-trillion-
parameter sparse MoEs, long context, and agentic workloads. The 2024 software model (compile a transformer
graph, pack weights, run a fixed pipeline) has no account of MoE expert routing at thousand-chip scale, KV-cache
management for long context, or the multi-turn/tool-calling patterns implied by "agentic". Whatever handles
those is undisclosed. *Source: /progress/frontier-inference-clusters (2026-06-30).*

### 12.3 Not disclosed (unchanged or newly relevant)

- Compiler name, input format, IR, and output binary format
- Whether a kernel-level or ISA-level programming surface exists on the Gen 2 part
- Runtime architecture, scheduler, batching/continuous-batching policy
- Collectives / communication library for the scale-up domain
- Memory model and coherence semantics of the CSM shared pool
- Serving API and endpoint format
- Model coverage list (which architectures are supported)
- Quantization / data-type support in the toolchain
- Profiler, debugger, model-compatibility checker
- Any open-source component whatsoever

### 12.4 Disclosure watch

Etched is a **Rhodium-level sponsor of Hot Chips 2026 (Aug 23–25, 2026)** with **no talk in the advance
program** — no software disclosure is scheduled there. Etched separately promised "more updates on our
performance and roadmap this summer" (2026-06-30), unpublished as of 2026-08-08.
