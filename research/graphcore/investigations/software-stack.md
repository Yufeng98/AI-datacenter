# Graphcore IPU — Software Stack Investigation

**Chip:** Graphcore IPU (GC200 / Bow)
**Device Class:** MIMD / BSP (IPU)
**Investigator:** AI Chip Research Pipeline
**Date:** 2026-04-05

---

## 1. Overview

The Graphcore software stack is unified under the **Poplar SDK**, which provides everything from low-level tile-vertex programming to high-level PyTorch/TensorFlow integration. The Poplar SDK is Graphcore's answer to CUDA — a complete, vertically integrated toolchain that must be used to compile and run any workload on the IPU.

**Latest release:** Poplar SDK 3.4.0 (March 4, 2024)
**Key insight:** Unlike CUDA where kernels are explicitly launched, Poplar uses a **graph-based** compilation model — the programmer describes a computation graph, and the Poplar compiler statically schedules every tile's compute and exchange program.

---

## 2. Stack Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  User / Framework Layer                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  PyTorch     │  │ TensorFlow 2 │  │  ONNX / PopART       │ │
│  │  (standard)  │  │  (standard)  │  │  (import+execute)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                  │                     │             │
├─────────▼──────────────────▼─────────────────────▼────────────┤
│  IPU Framework Extensions                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  PopTorch    │  │  TF-IPU      │  │  PopART (Advanced    │ │
│  │  (PyTorch    │  │  backend     │  │  Runtime)            │ │
│  │   extension) │  │              │  │                      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                  │                     │             │
├─────────▼──────────────────▼─────────────────────▼────────────┤
│  Poplar Graph Compiler & Runtime                               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Poplar Graph API (C++)                                │   │
│  │  ├── Graph construction (addVertex, addCodelets, etc.) │   │
│  │  ├── Program IR (Sequence, Repeat, Switch, Copy)       │   │
│  │  └── poplar::Engine (compile, load, run)               │   │
│  └────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PopLibs (built-in op libraries)                       │   │
│  │  ├── poplin  — linear algebra (GEMM, Conv)             │   │
│  │  ├── popnn   — neural net layers (RNN, NLP, norm)      │   │
│  │  ├── popops  — elementwise ops, reductions, scatter    │   │
│  │  ├── poprand — random number generation                │   │
│  │  └── popsparse — sparse operations                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  Codelet / Vertex Layer (Tile ISA)                             │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  Codelets (C++ or assembly)                            │   │
│  │  ├── Compiled by Poplar SDK (host → IPU tile binary)   │   │
│  │  ├── Each vertex runs on one tile worker thread        │   │
│  │  └── Direct access to tile SRAM only                   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  Runtime / Driver Layer                                        │
│  ┌─────────────────┐  ┌────────────────────────────────────┐  │
│  │  poplar::Engine  │  │  V-IPU (Virtualized IPU)          │  │
│  │  (host runtime) │  │  Multi-tenant IPU resource manager │  │
│  └─────────────────┘  └────────────────────────────────────┘  │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  PCIe Driver / HAL                                             │
│  └── Host OS kernel driver → IPU-Machine PCIe interface        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  IPU Hardware                                                  │
│  └── 1,472 tiles × (6 worker threads + 624 KB SRAM) + Exchange│
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Poplar SDK (Core Compiler & Runtime)

### 3.1 Poplar Graph API

The central abstraction is a **computation graph**:

```cpp
#include <poplar/Graph.hpp>

poplar::Graph graph(device);
auto a = graph.addVariable(poplar::FLOAT, {1024}, "a");
graph.setTileMapping(a, 0);

graph.addCodelets("my_kernel.cpp");
auto cs = graph.addComputeSet("main");
auto v  = graph.addVertex(cs, "MyVertex");
graph.setTileMapping(v, 0);

poplar::program::Sequence prog;
prog.add(poplar::program::Execute(cs));
prog.add(poplar::program::Copy(src, dst));

poplar::Engine engine(graph, prog);
engine.load(device);
engine.run(0);
```

### 3.2 Compilation Model

1. **Graph construction** (host, C++ or Python bindings)
2. **Compilation**: Poplar compiler analyzes graph, assigns tensors to tiles, generates per-tile compute and exchange schedules
3. **Executable**: fully static schedule — each tile executes exactly instruction sequence X, then sends/receives exactly bytes Y
4. **Load**: executable transferred to IPU via PCIe
5. **Run**: IPU executes autonomously; host waits

Unlike CUDA: no JIT launch, no dynamic dispatch, no warp divergence handling, memory layout fully determined at compile time.

---

## 4. PopLibs — Built-in Operation Libraries

| Library | Domain | Key Functions |
|---------|--------|---------------|
| `poplin` | Linear algebra | GEMM, convolution (fwd/bwd/upd), grouped conv |
| `popnn` | Neural network | RNN/LSTM/GRU, layer norm, batch norm, softmax |
| `popops` | Element-wise + reduce | Add, mul, exp, log, reduce (sum/max/mean), cast |
| `poprand` | Randomness | Uniform, normal, Bernoulli sampling |
| `popsparse` | Sparse ops | Sparse GEMM, sparse embedding lookup |

---

## 5. Codelets — Tile-Level Kernels

```cpp
// my_vertex.cpp — compiled for IPU tile
#include <poplar/Vertex.hpp>

class ReLUVertex : public poplar::Vertex {
public:
    poplar::Input<poplar::Vector<float>>  in;
    poplar::Output<poplar::Vector<float>> out;

    bool compute() {
        for (unsigned i = 0; i < in.size(); ++i)
            out[i] = std::max(0.0f, in[i]);
        return true;
    }
};
```

- Written in **C++ or assembly** (ASM for maximum performance)
- Each vertex runs on **one tile worker thread**
- Accesses only **tile-local SRAM**
- Tile ISA publicly documented: https://docs.graphcore.ai/projects/isa/en/latest/

---

## 6. PopART — Poplar Advanced Runtime

PopART is the **ONNX-based** training and inference runtime:

```python
import popart

session = popart.TrainingSession(
    fnModel=builder.getModelProto(),
    dataFlow=popart.DataFlow(batches_per_step, anchors),
    loss=loss_id,
    optimizer=popart.SGD({"defaultLearningRate": (0.01, False)}),
    deviceInfo=popart.DeviceManager().acquireAvailableDevice(1)
)
session.prepareDevice()
session.run(stepio)
```

Key features: ONNX import/export, gradient accumulation, pipeline parallelism, replication.

---

## 7. PopTorch — PyTorch for IPU

```python
import poptorch

model = MyModel()
poptorch_model = poptorch.trainingModel(model)

for batch in dataloader:
    loss = poptorch_model(batch)  # compiles on first call, then runs on IPU
```

**IPU-specific features:**
- `poptorch.Options()`: replication factor, gradient accumulation, precision
- `poptorch.BeginBlock()` / `poptorch.EndBlock()`: pipeline stage annotations
- `poptorch.ipu_print_tensor()`: debug print from IPU
- `poptorch.recomputationCheckpoint()`: activation recomputation for memory savings
- Mixed precision: FP16 + stochastic rounding (FP16.SR)

---

## 8. TensorFlow for IPU

```python
from tensorflow.python import ipu

cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
ipu.config.configure_ipu_system(cfg)

strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    model = keras.Model(...)
    model.compile(...)
    model.fit(dataset)
```

---

## 9. V-IPU — Virtualized IPU

Resource manager for multi-tenant IPU deployments:
- Abstracts physical IPUs into virtual partitions
- Supports Slurm-based HPC cluster integration
- Manages IPU allocation, firmware state, network topology

---

## 10. PopVision — Profiling

- **Graph Analyser:** tile mapping, memory usage, Poplar graph visualization
- **System Analyser:** BSP superstep timeline, compute vs exchange ratio, cycle counts
- Identifies: memory overflows, exchange imbalance, underutilized tiles

---

## 11. Triton Inference Server Backend

Poplar SDK ships a Triton backend for serving PopTorch/PopART/TF-IPU models via gRPC/HTTP.

---

## 12. Software Layer Mapping

| Layer | Component | Analogous CUDA Component |
|-------|-----------|--------------------------|
| Framework | PopTorch / TF-IPU | PyTorch (CUDA) / TF GPU |
| Model Runtime | PopART | TensorRT / ONNX Runtime |
| Graph Compiler | Poplar | XLA / TorchInductor |
| Op Library | PopLibs (poplin/popnn/popops) | cuDNN / cuBLAS |
| Kernel Language | Codelets (C++ / ASM) | CUDA C++ |
| ISA | Tile Vertex ISA | PTX + SASS |
| Runtime | poplar::Engine | CUDA Runtime |
| Driver | PCIe HAL | nvidia.ko |
| Virtualization | V-IPU | MIG / MPS |
| Profiling | PopVision | Nsight |

---

## 13. Precision Support

- **FP16.16**: FP16 multiply, FP16 accumulate with stochastic rounding — primary AI mode
- **FP16.SR**: hardware stochastic rounding — unique IPU training accuracy feature
- **FP32**: full precision
- **INT8**: limited support

---

## 14. Openness

| Component | Status |
|-----------|--------|
| PopTorch | Open source (GitHub) |
| PopLibs | Open source (GitHub) |
| Poplar SDK docs | Publicly available |
| Poplar compiler | Closed binary |
| Chip design | Closed |
| PopART | SDK release (public, not fully open) |

---

## 15. Key Software Insights

1. **Static graphs are mandatory:** Poplar requires full graph knowledge at compile time.
2. **Compilation is slow:** Can take minutes for large models; executable caching is essential.
3. **Stochastic rounding:** Built-in FP16.SR for higher training accuracy at FP16 — unique IPU feature.
4. **Gradient accumulation is first-class:** PopTorch handles micro-batching transparently.
5. **Pipelining:** Model pipeline parallelism across IPUs well-supported via BeginBlock/EndBlock.
6. **SDK lifecycle concern:** Last release Poplar SDK 3.4.0 (March 2024); post-SoftBank pace unclear.

---

## 16. Re-scan — 2026-08-08 (SDK / toolchain status)

**Scan window:** 2026-04-01 → 2026-08-08. **Prior investigation date:** 2026-04-05.
**Verdict: no software change.** Poplar SDK 3.4.0 (2024-03-04) remains the latest release. Sections 1–15 above stand unmodified.

### 16.1 SDK release status — verified via GitHub API, not the docs site

The docs site is **not** a usable version oracle any more: `docs.graphcore.ai`'s SDK-overview page **does not state a current SDK version** (it only references behaviour "since Poplar SDK 3.1"). The check was therefore done against the GitHub release/tag API:

| Repository | Newest release tag | Last code push |
|---|---|---|
| `graphcore/poplibs` | `sdk/poplar/3.4.0` | October 2023 |
| `graphcore/poptorch` | `sdk/poptorch/3.4.0` | October 2023 |

**Conclusion: the open-source Poplar stack is effectively dormant.** No 3.5, no 4.x, no post-3.4.0 tag exists. The repo's "Poplar SDK 3.4.0 (March 2024)" baseline stands unchanged, and the §15 "SDK lifecycle concern" is now stronger, not weaker.

### 16.2 GitHub org activity — real, but not a stack shift

The `graphcore` GitHub org has recent pushes, which is easy to misread as new toolchain work. It is not:

| Repo | Last push | What is actually in it |
|---|---|---|
| `graphcore/pytorch-fork` | 2026-08-05 | Fork of upstream PyTorch |
| `graphcore/triton-fork` | 2026-06-08 | **Stock upstream README; no IPU backend, no mention of IPU support** — reads as a plain upstream mirror |
| `graphcore/vllm-fork` | 2025-09-22 | **Stock upstream README; no IPU backend, no mention of IPU support** — reads as a plain upstream mirror |
| `graphcore/llvm-project-fork` | 2026-03-11 | Does carry a Colossus IPU backend (the long-standing LLVM path used by the Poplar codelet compiler) |

⚠️ **Do not report a Triton or vLLM IPU inference path.** Both forks were opened and inspected; neither contains IPU support. Only the LLVM fork carries IPU-specific code, and that is the pre-existing Colossus backend, not new work.

### 16.3 Precision / feature surface

Unchanged. No FP8, no BF16, no INT4 in the SDK. The one 2026 Graphcore technical post, *"Stochastic Rounding: How randomness helps us build better models"* (2026-05-26), re-explains the **existing** FP16.SR feature documented in §13; it introduces no new API, datatype, or SDK capability.

### 16.4 Corporate context

Not a software fact, but it is the mechanism behind the dormancy: an ongoing SoftBank recapitalisation (a ~$457M share issue recorded on the UK register in April 2026, with further allotments through 2026-08-04) and the departure of both technical co-founders (Simon Knowles 2025-08-20; Nigel Toon 2026-07-31). Details in `research/graphcore/investigations/hw-architecture.md` §12.3 and `chips/graphcore/summary.md`.

### 16.5 Re-scan sources

- https://api.github.com/repos/graphcore/poplibs/tags
- https://api.github.com/repos/graphcore/poptorch/tags
- https://api.github.com/repos/graphcore/poptorch/releases
- https://api.github.com/orgs/graphcore/repos
- https://github.com/graphcore/triton-fork
- https://github.com/graphcore/vllm-fork
- https://docs.graphcore.ai/projects/sdk-overview/en/latest/overview.html
- https://www.graphcore.ai/posts

---

## Sources

- [Poplar SDK Overview](https://docs.graphcore.ai/projects/sdk-overview/en/latest/overview.html)
- [Poplar SDK 3.4.0 Release Notes](https://docs.graphcore.ai/_/downloads/release-notes/en/latest/pdf/)
- [Poplar and PopLibs User Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html)
- [PopLibs API Reference](https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html)
- [PopTorch User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/intro.html)
- [PopTorch GitHub](https://github.com/graphcore/poptorch)
- [PopLibs GitHub](https://github.com/graphcore/poplibs)
- [Creating Custom Operations for the IPU](https://docs.graphcore.ai/projects/custom-ops/en/latest/custom-ops.html)
- [Tile Vertex ISA](https://docs.graphcore.ai/projects/isa/en/latest/)
- [IPU Programmer's Guide — Programming Model](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/programming_model.html)
- [Memory and Performance Optimisation](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/understand-ipu-programming-model.html)
