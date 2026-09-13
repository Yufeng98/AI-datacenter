# SambaNova Software Stack

*as_of: 2026-04-05*
*chip: sambanova*
*Stack: SambaFlow (compiler + runtime + SDK)*

---

## Overview

The SambaNova software stack is named **SambaFlow**. It is a vertically integrated compilation and runtime environment that takes a standard **PyTorch model** (or TensorFlow) and:

1. Traces the computation graph via PyTorch's tracing machinery
2. Applies spatial mapping, operator fusion, and memory placement
3. Outputs a **PEF (Program Execution File)** — a static binary encoding the entire model-to-hardware mapping
4. Executes via the SambaNova Runtime, which loads the PEF to the RDU and calls it via `samba.session.run()`

The stack philosophy: **the compiler does what a GPU's runtime scheduler does at compile time**, enabling zero-overhead inference. The programmer never writes a kernel.

---

## 1. Framework Integration (PyTorch)

### Entry Point

SambaFlow integrates with PyTorch at the **module tracing layer** — the same level as `torch.fx` or `torch.jit.trace`, but specific to SambaNova. The user writes a standard `torch.nn.Module`.

### Conversion API

```python
import sambaflow.samba as samba

# Convert PyTorch model to SambaFlow model
samba_model = samba.from_torch_model(model)
# Converts: torch.Tensor → SambaTensor
#           nn.Parameter → SambaParameter
```

Key points:
- **Most model code is unchanged** — only the session layer changes
- `samba.from_torch_model()` walks the module, replacing tensor operations with SambaFlow implementations that log the computation graph instead of executing it
- `SambaTensor` and `SambaParameter` are duck-type compatible with their torch counterparts during tracing

### Hugging Face Integration

SambaFlow explicitly supports Hugging Face Transformers models:
- `samba.session.compile(model, dummy_inputs)` accepts any `transformers.PreTrainedModel`
- Documented examples: GPT-2, GPT-J, various BERT variants, Llama
- Model weights downloaded from HF Hub, converted in-place, compiled to PEF

### Training Integration

For training:
- `samba.session.run()` replaces the standard `loss.backward()` + `optimizer.step()` calls
- Gradients are computed by the RDU within the single `run()` call — no separate backward pass dispatch
- SambaFlow uses `samba.optim.*` wrappers for optimizers (SGD, Adam, AdamW)

### Comparison vs GPU Frameworks

| Feature | SambaFlow (RDU) | PyTorch + CUDA (GPU) |
|---------|----------------|----------------------|
| Kernel authoring | None required | Optional (CUDA C++, Triton) |
| Backward pass | Automatic via compiler | autograd engine |
| Dispatch mechanism | `samba.session.run()` — static | Dynamic dispatch per op |
| JIT vs AOT | AOT only (compile to PEF first) | Both (eager + compile) |
| Operator customization | Limited (compiler-controlled) | Full (custom CUDA kernels) |

---

## 2. Compiler: SambaFlow Dataflow Compiler

### Architecture

The SambaFlow compiler is a **whole-graph, ahead-of-time (AOT) compiler**. It receives the traced PyTorch computation graph and performs:

1. **Graph-level transformations**
   - Operator fusion into "sections"
   - Meta-pipelining (pipeline parallelism across sections)
   - Multi-section support (splitting large models into segments that fit on-chip)
   - Parallelism mapping (data, tensor, pipeline parallelism)

2. **Hardware-aware lowering**
   - PCU assignment: maps each operator subgraph to a PCU or set of PCUs
   - PMU assignment: allocates on-chip SRAM for tensors; determines which tier (SRAM/HBM/DDR) holds each tensor across the memory hierarchy
   - Switching fabric routing: statically routes data paths between PCUs/PMUs
   - Address generation configuration for AGCUs

3. **Output: PEF (Program Execution File)**
   - Binary file encoding the complete hardware configuration
   - Contains PCU pipeline configurations, PMU data layouts, switching fabric routes, AGCU descriptors
   - The RDU is "programmed" by loading the PEF at session start

### Operator Fusion: o0 vs o1 Modes

The compiler's most impactful optimization is **operator fusion**:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `--compiler-mode o0` | One section per operator (no fusion) | Debugging, profiling, verifying correctness |
| `--compiler-mode o1` | Maximum operator fusion | Production inference/training |

In `o1` mode, the compiler fuses hundreds of operators (e.g., Linear → LayerNorm → GELU → Linear → Dropout) into a **single section** that executes entirely within on-chip SRAM, with no intermediate DRAM writes. This is the spatial equivalent of FlashAttention — but applied to the entire model graph, not just attention.

Speedups from fusion: **2x–13x** vs unfused baseline on various workloads (per SN40L paper).

### Sections

A **section** is the fundamental execution unit in SambaFlow:
- A contiguous subgraph of the model, fused into one hardware configuration
- Executes completely before the system returns to memory
- Sections are named and called by name in `samba.session.run()`
- Multiple sections in a model are **pipelined** by the runtime (meta-pipelining): while section N executes on the RDU, section N+1's data is pre-staged in HBM

### Compilation Pipeline

```
PyTorch Model (.py)
    ↓  samba.from_torch_model()  [graph tracing]
Annotated Dataflow Graph
    ↓  High-level transforms      [fusion, parallelism, pipelining]
Optimized Graph
    ↓  Hardware-aware lowering    [PCU/PMU placement, fabric routing]
Hardware Configuration
    ↓  Code generation
PEF File (.pef)
    ↓  samba.session.load()
RDU Hardware Execution
```

### Parallelism Strategies

The SambaFlow compiler supports nested combinations of:
- **Data parallelism**: replicate model across multiple RDU sockets; split batch
- **Tensor parallelism**: split weight matrices across PCU arrays within/across sockets
- **Pipeline parallelism**: different model stages on different sockets

These are configured via compiler flags and the `samba.session.compile()` API, not by manual model sharding (unlike GPU tensor parallel with Megatron-LM).

---

## 3. Runtime: SambaNova Runtime

### Session API

The session object (`samba.session`) is the primary interface to the RDU at runtime:

```python
# Compile (AOT — done once)
samba.session.compile(
    model=samba_model,
    inputs=dummy_inputs,
    name="my_model",
    output_file="my_model.pef",
    compile_op_mode="o1"
)

# Load PEF to RDU
samba.session.load("my_model.pef")

# Run (inference or training)
outputs = samba.session.run(
    inputs=real_inputs,
    section_types=["RDU"]  # which sections to execute
)
```

### Execution Model

- `samba.session.run()` is **synchronous** from the host perspective
- Inside the RDU, execution is fully pipelined (PCU stages overlap)
- No thread blocks, no warps, no kernels — the PEF is the program
- Multiple sections execute sequentially with data-pipelining between them

### Data Movement at Runtime

1. Weights loaded from DDR → HBM at model load time (>1 TB/s on 16-socket node)
2. Input activations transferred host → RDU (PCIe or direct attach)
3. During inference: HBM → on-chip SRAM as needed (compiler-determined schedule)
4. PCUs consume from and produce to PMUs; PMU-to-PMU data moves on switching fabric
5. Output activations transferred RDU → host

### Multi-RDU Execution

For models spanning multiple RDU sockets:
- Collective communication primitives (AllReduce, AllGather, ReduceScatter) are compiled into the PEF
- The runtime executes these using the P2P network (SN40L) or switched fabric (SN50)
- No separate communication library needed (unlike NCCL for GPUs)

---

## 4. Memory Management

Unlike GPU programming (where the programmer explicitly allocates and manages CUDA device memory), SambaFlow memory management is **entirely compiler-driven**:

| Memory Tier | Who decides placement | Programmer control |
|-------------|----------------------|--------------------|
| PMU SRAM | Compiler (static, per-PEF) | None |
| HBM | Compiler (static caching policy) | None |
| DDR | Compiler (model weight placement) | Indirectly via model structure |
| Host DRAM | User (via numpy/torch tensors) | Full |

This means:
- No `cudaMalloc`, no explicit tensor placement, no `.to(device)` calls beyond model conversion
- No memory fragmentation issues
- No OOM errors from dynamic allocation (model must fit in the three tiers)
- The PEF encodes the full memory layout — recompile to change it

---

## 5. Operator Support and Model Compatibility

### Supported Operations

SambaFlow supports the full PyTorch `nn` module set for transformer models:
- Linear / GEMM
- Attention (multi-head, grouped-query)
- LayerNorm, RMSNorm
- GELU, SiLU, ReLU activations
- Embedding lookup
- Softmax
- Convolutions (CNN support from earlier generations)
- Custom user operations via `samba.op` annotations

### Unsupported Patterns

Some PyTorch patterns require model modifications for RDU:
- Dynamic control flow (Python `if`/`while` over tensor values) — must be unrolled or eliminated
- In-place operations that conflict with tracing
- Some uses of `BatchNorm` (recommended to use `LayerNorm` instead on RDU)
- Variable-length sequence batching without padding

### Model Classes

SambaFlow has tested and documented support for:
- GPT-2, GPT-J, GPT-NeoX
- Llama, Llama 2, Llama 3
- BERT, RoBERTa
- ResNet, VGG (CNN inference)
- Custom Composition of Experts (CoE) architectures

---

## 6. Composition of Experts (CoE) — Flagship Use Case

CoE is SambaNova's primary innovation enabled by the three-tier memory architecture and dataflow compiler:

**Concept**: Instead of a single large monolithic model (like GPT-4), run an ensemble of specialized small models (experts) and route queries to the best expert at inference time.

**SambaNova implementation**:
- **Samba-CoE**: 150 expert models × 7B parameters = ~1T total parameter ensemble
- Each expert fits in HBM; all experts' weights reside in DDR
- At inference time: selected expert's weights stream DDR → HBM → on-chip SRAM → compute
- The dataflow compiler creates a single PEF covering all 150 experts; the router selects which section to execute

**Why this works on RDU and not GPU**:
- GPU (H100): 80 GiB HBM, no DDR. Each 7B model is ~14 GiB in BF16; only ~5 experts fit in HBM simultaneously. No DDR tier
- RDU (SN40L): 1.5 TiB DDR stores all 150 × ~14 GiB = ~2.1 TiB models; HBM acts as a fast cache; SRAM handles active compute. With DDR compression, feasible on a single node

---

## 7. SDK and Tooling

### Core SDK (`sambaflow`)

| Component | Purpose |
|-----------|---------|
| `sambaflow.samba` | Core session API (compile, load, run) |
| `sambaflow.samba.optim` | Optimizer wrappers (Adam, SGD, AdamW) |
| `sambaflow.samba.metrics` | Performance benchmarking utilities |
| `sambaflow.nn` | Custom nn layers optimized for RDU |
| Model analyzer | Analyzes PyTorch model; reports compatibility issues before compilation |

### Development Workflow

```
1. Write/load PyTorch model
2. Run samba.session.compile() to generate PEF  [one-time, minutes]
3. Run samba.session.load(pef) to initialize RDU  [seconds]
4. samba.session.run() in inference/training loop  [milliseconds per call]
```

Compilation is **amortized**: PEF is generated once and cached. The same PEF is used for all subsequent runs of the same model with the same configuration.

### Compiler Flags

Key `samba.session.compile()` options:
- `--num-tiles N`: how many PCU/PMU tiles to use (controls resource allocation)
- `--pef-name name`: PEF output filename
- `--compiler-mode {o0,o1}`: fusion mode
- `--data-parallel N`: number of data-parallel replicas
- `--weight-sharing`: enable weight sharing across RDU sockets

---

## Sources

- [Model Conversion Overview](https://docs.sambanova.ai/developer/latest/porting-overview.html)
- [Compiler Optimization Modes](https://docs.sambanova.ai/developer/latest/compiler-o1.html)
- [SambaFlow API Reference](https://docs.sambanova.ai/api-reference/)
- [Compilation, Training, and Inference (LeNet)](https://docs.sambanova.ai/developer/latest/lenet-end2end.html)
- [Compile/Fine-tune/Inference with HuggingFace GPT](https://docs.sambanova.ai/developer/latest/hf-compile-run.html)
- [Code Elements of the Inference Program](https://docs.sambanova.ai/developer/latest/hf-model-inference.html)
- [Data Parallel Mode](https://docs.sambanova.ai/developer/latest/data-parallel.html)
- [SambaNova Glossary](https://docs.sambanova.ai/resources/latest/glossary.html)
- [SambaFlow Release Notes (legacy)](https://docs-legacy.sambanova.ai/developer/latest/release-notes.html)
- [Accelerated Computing with a Reconfigurable Dataflow Architecture (Whitepaper)](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [SambaNova SN40L Paper (arXiv 2405.07518)](https://arxiv.org/html/2405.07518v1)

---

# Investigation Update — 2026-08-08

*Scan window 2026-04-05 → 2026-08-08. Change class: **major**. This section is additive; nothing above is deleted. Where the two disagree, this section wins.*

## A. The SambaFlow developer documentation has been withdrawn

**This is the headline change for the SambaNova software stack.** Every SambaFlow documentation URL cited in this investigation and in `chips/sambanova/summary.md` returns **HTTP 404** as of 2026-08-08, verified by direct request rather than inferred:

| URL | Status 2026-08-08 |
|---|---|
| `https://docs.sambanova.ai/developer/latest/` | 404 |
| `https://docs.sambanova.ai/developer/latest/porting-overview.html` | 404 |
| `https://docs.sambanova.ai/developer/latest/compiler-o1.html` | 404 |
| `https://docs.sambanova.ai/api-reference/` | 404 |
| `https://docs.sambanova.ai/resources/latest/glossary.html` | 404 |

`docs.sambanova.ai` now serves a Mintlify site rooted at `/docs/en/...`. The full index (`docs.sambanova.ai/docs/llms.txt`, **174 entries**) covers exactly two products:

- **SambaCloud** — Get Started, Models, Features (OpenAI compatibility, Anthropic compatibility, prompt caching, responses, vision/audio/video), Build, Integrations, API Reference
- **SambaStack v2.0.2** — getting started, hardware administration (including "RDU module administration" and the SambaRack Manager command reference), service administration, model deployment, performance, observability, platform administration

The index contains **no SambaFlow, no compiler, no PEF-authoring, and no `samba.session` API documentation of any kind**.

### Consequences for this investigation

1. Everything above — SambaFlow tracing via `samba.from_torch_model()`, `o0`/`o1` fusion modes, `samba.session.compile/load/run`, the compiler-flag list, the SDK module table — is now sourced entirely to documentation that is **no longer published**. It is re-attributed to the **SN40L Hot Chips paper and arXiv 2405.07518** and re-dated as describing the **SN40L-era SambaFlow SDK**, not the currently documented product.
2. Public SambaFlow SDK documentation is marked **withdrawn as of ~mid-2026**.
3. **Whether the compiler itself changed is unknown.** No source supports claiming it did. Withdrawal of documentation is not evidence of a compiler rewrite, of a compiler deprecation, or of anything about SN50's toolchain.

### PEF survives

The deployment artifact is unchanged. SambaStack v2.0.2 exposes a **`Pef` Kubernetes custom resource** alongside `Model`, `ModelProfile`, `ModelDeployment` and `ModelBundle`, and the SambaWiz tool generates "PEF settings" plus Kubernetes manifests. What changed is the *surface around* the PEF: **Helm charts and Kubernetes CRDs, not a Python SDK.**

## B. SambaStack — the currently documented deployment path

### Verified version history (official release notes)

| Version | Date |
|---|---|
| v0.4.8 | 2026-03-10 |
| v0.5.14 | 2026-04-01 |
| v0.5.17 | 2026-04-08 |
| v1.0.57 | 2026-04-30 |
| v1.1.1 | 2026-05-27 |
| v1.2.0 | 2026-07-07 |
| **v2.0.2** | **2026-08-05** |

### v2.0.2 — breaking release (three days old at scan time)

- **New four-CRD resource model** — `Model`, `ModelProfile`, `ModelDeployment`, `ModelBundle` — replacing `BundleTemplate` / `Bundle` / `BundleDeployment`. The old format remains functional until **2026-09-30**.
- **Kubernetes (RKE2) 1.30 → 1.35**
- **Models split into a standalone `sambastack-models` Helm chart**; install order is `sambastack-base` → `sambastack` → `sambastack-models`
- **Structured output enforced** for constrained-decoding models
- Reference architecture documents **LiteLLM** as the gateway and **Prometheus / Grafana / OpenSearch / Fluent Bit** for observability

Note: the on-prem hardware reference documents remain **SN40L-16**. **No SN50 SambaRack documentation is published.**

## C. vLLM as the reported SN50 serving path — narrowly sourced, hedge carefully

Both SN50 performance posts state the results were produced on vLLM:

- 2026-07-30, verbatim: *"SN50 is deployed using vLLM appearing with ~800 tokens/second at the fastest interactivity"*
- 2026-07-08, verbatim: *"This demonstration was built and measured on vLLM."*

Countervailing evidence, all negative and all checked in this scan:

| Check | Result |
|---|---|
| Public SambaNova vLLM hardware plugin | **None found.** The `sambanova` GitHub org contains no vLLM repository; no `vllm-sambanova` plugin found |
| vLLM in the SambaStack v2.0.2 documentation index | **Absent** |
| vLLM in the complete SambaStack release-notes history (Sept 2025 – Aug 2026) | **Absent** |
| vLLM in the 2026-04-08 Intel blueprint press release | **Absent** (the release mentions no open-source serving stack at all) |

**The defensible statement:** *SambaNova reports serving SN50 through vLLM in its 2026 benchmark demonstrations; the integration is not open-source and is not referenced in the shipping SambaStack product documentation, so its depth is unverified.* Do not write that "the serving stack has shifted to vLLM" — that is over-read from two blog sentences.

## D. Parallelism is a deployment-time choice — qualifying two long-standing repo claims

The 2026-07-30 post describes an operator selecting between two chip-parallelism strategies on the same 16-chip rack:

| Configuration | Mapping | Reported throughput |
|---|---|---|
| Fastest interactivity | TP16 across all 16 RDUs | ≈800 t/s |
| More concurrent users | TP8 + DP8 | ≈400 t/s |

SambaStack ships a **"High-throughput vs. high-interactivity" deployment-configuration page**, corroborating that this is a supported operator-facing knob.

Two repo statements are therefore qualified (edits applied in `chips/sambanova/summary.md`):

- *"No kernel authoring required or possible"* — still true in the sense that no user-facing kernel language has ever been published, but the operator does make architecturally consequential choices.
- *"Multi-RDU collectives are compiled into the PEF; there is no separate communication library"* — still true that no collectives library is published, but the collective pattern follows from the deployment-time parallelism strategy, so the PEF is built *per chosen strategy* rather than being strategy-agnostic.
- *"no manual model sharding needed"* → **"the compiler maps the chosen parallelism strategy; the strategy itself is a deployment-time choice."**

## E. Open-source tooling: `sambanova/sambastack-tools`

TypeScript. **Created 2026-01-15** — i.e. *before* the 2026-04-05 baseline, so it is new **to this survey**, not new **in the window**. Actively pushed through 2026-08-07. 2 stars. Not archived.

| Component | Description (verbatim from README) |
|---|---|
| **SambaEval** | "a local workbench for evaluating LLMs against CSV datasets across multiple models and providers… heuristic and LLM-as-judge scoring, per-row token usage and latency metrics, and pluggable Python output generators for tool use or agentic workflows" |
| **SambaWiz** | "a GUI wizard that accelerates the creation and deployment of model bundles on SambaStack… selecting models, configuring PEF settings, generating Kubernetes manifests, and deploying bundles to your cluster" |

SambaWiz releases: **v2.0.0** (2026-07-31, V3 bundle support), **v2.1.0** (2026-08-06, vision + ASR/TTS playground), **v2.1.1** (2026-08-07). The SambaStack release notes instruct operators to pull the matching SambaWiz version with each release, so **SambaWiz is effectively part of the supported deployment path**, not a side project.

Also in the org and pushed within the window: `sambanova-inference-api-spec` (created 2026-07-21), `sambanova-plugin-cc` (created 2026-03-31), plus the SDK/integration repos. There is **no public SambaFlow, model-zoo, or vLLM repository**.

## F. SambaCloud feature additions in the window

| Date | Feature |
|---|---|
| 2026-05-05 | MiniMax M2.7 launched on SambaCloud |
| 2026-06-10 | Gemma 4 31B |
| 2026-07-01 | Anthropic Messages API compatibility |
| 2026-07-16 | Prompt caching |

## G. Open items for the next scan

1. Whether any replacement SambaFlow / compiler documentation is republished.
2. Whether a SambaNova vLLM plugin is open-sourced, or vLLM appears in SambaStack documentation or release notes.
3. Whether SambaStack gains SN50 rack documentation (SN40L-16 only today).
4. Hot Chips 38, 2026-08-25, "Dataflow at Scale: the SN50 RDU" (Raghu Prabhakar) — **disclosure scheduled; content not yet public.** It may address the compiler question in §A.3. Re-scan early September 2026.

## Sources added 2026-08-08

- https://docs.sambanova.ai/ — current documentation root (SambaCloud + SambaStack only)
- https://docs.sambanova.ai/docs/llms.txt — full 174-entry index; no SambaFlow/compiler/PEF-authoring pages
- https://docs.sambanova.ai/docs/en/release-notes/sambastack.md — authoritative version history; `Pef` CRD; K8s 1.30→1.35; no vLLM mention anywhere
- Direct HTTP status checks (`curl -L`) on the five previously cited SambaFlow doc URLs — all HTTP 404 on 2026-08-08
- https://github.com/sambanova/sambastack-tools — SambaEval + SambaWiz
- https://raw.githubusercontent.com/sambanova/sambastack-tools/main/README.md — verbatim component descriptions
- https://api.github.com/repos/sambanova/sambastack-tools — created_at 2026-01-15T17:22:22Z, pushed_at 2026-08-07, TypeScript, 2 stars
- https://api.github.com/orgs/sambanova/repos?sort=pushed — confirms no vLLM, SambaFlow, or model-zoo repository
- https://sambanova.ai/blog/sn50-runs-fastest-minimax-speeds-in-the-world — "built and measured on vLLM"
- https://sambanova.ai/blog/semianalysis-benchmarks-sambarack-sn50-with-fast-inference-on-minimax-m2.7 — TP16 vs TP8+DP8 deployment configurations
- https://sambanova.ai/press/sambanova-announces-collaboration-with-intel-on-ai-solution — no mention of vLLM, Kubernetes, or any open-source serving stack
