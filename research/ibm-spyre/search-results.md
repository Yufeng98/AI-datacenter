# IBM Spyre Accelerator — Search Results

*as_of: 2026-08-08*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*

---

## Summary

The **IBM Spyre Accelerator** is a discrete 5 nm inference ASIC on a single-slot 75 W PCIe card, sold as a
priced system option for **IBM z17 / LinuxONE 5** (GA 2025-10-28) and **IBM Power11** (GA "early December
2025"). It is the productized descendant of the IBM Research **AIU** (Artificial Intelligence Unit) line, and
the software stack still carries the AIU name throughout (`torch_sendnn`, `aiu-smi`, `libaiupti`, the
`ibm-aiu` GitHub org).

Two things make it worth a survey entry beyond "another inference card":

1. **No hardware cache anywhere in the compute path.** IBM's own compiler documentation states it flatly —
   the compiler emits explicit load/store instructions to stage 128-byte "sticks" between LPDDR5 and a
   2 MB per-core LX scratchpad, and there is no eviction mechanism by design.
2. **An unusually open middle of the stack.** IBM stood up a public GitHub org (`github.com/torch-spyre`,
   14 mostly Apache-2.0 repos) publishing the PyTorch backend, the TorchInductor front end, the
   compiler↔device **interface specs**, design **RFCs**, a Triton fork, and a **CPU interpreter for the tile
   IR** — while keeping the back-end compiler, runtime, driver, firmware and ISA closed.

**Do not conflate with the Telum II on-die AI unit.** Telum II's zAIU is a CPU-integrated accelerator reached
via the NNPA instruction and served by zDNN/zDLC. Spyre is a separate discrete PCIe ASIC with a separate
software path; zDNN and zDLC do **not** target Spyre.

**Hot Chips 38 (2026-08-23/25) is in the future.** Nothing in this file depends on it. All evidence is from
ISSCC 2026 (past), IBM Research blogs, IBM Newsroom, IBM Docs, and live public repositories.

---

## Search Queries

1. "IBM Spyre Accelerator ISSCC 2026 inference-optimized scalable AI accelerator enterprise workloads"
2. "IBM Spyre Accelerator 128GB LPDDR5 card memory capacity"
3. "IBM Z Deep Learning Compiler Spyre accelerator zDNN onnx-mlir"
4. "Spyre accelerator z/OS AI Toolkit Machine Learning for z/OS Watson Machine Learning ONNX inference"
5. "aiu-fms foundation-model-stack AIU Spyre github IBM deep learning compiler torch_sendnn"
6. "github torch-spyre org repositories list" (`api.github.com/orgs/torch-spyre/repos`)
7. "github ibm-aiu org repositories list" (`api.github.com/orgs/ibm-aiu/repos`)
8. "torch-spyre KernelTile IR KTIR MLIR dataflow scheduler IBM"
9. Direct fetch of the full `torch-spyre.readthedocs.io` doc tree (architecture / compiler / runtime / user guide)
10. Direct fetch of the `torch-spyre/RFCs` and `torch-spyre/interface-specs` git trees
11. "vllm-project/vllm-spyre repository status and redirect target"
12. "IBM/zDNN README hardware target Telum NNPA vs Spyre"
13. `research.ibm.com` Spyre blog series: lifting the cover / building / PyTorch support / Spyre for Z / AIU

---

## Resources Found

### Vendor Primary — Architecture and Product

| Resource | URL | Category |
|----------|-----|----------|
| IBM Research — "Lifting the cover on the IBM Spyre Accelerator" (2026-02-18; deepest public microarchitecture description) | https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator | Hardware Spec |
| IBM Research — "Building the IBM Spyre Accelerator" | https://research.ibm.com/blog/building-the-ibm-spyre-accelerator | Product / Lineage |
| IBM Research — "PyTorch support for IBM Spyre" (2026-03-06) | https://research.ibm.com/blog/pytorch-support-ibm-spyre | Software |
| IBM Research — "Spyre for Z" (Hot Chips 2024 preview) | https://research.ibm.com/blog/spyre-for-z | Preview (stale figures) |
| IBM Research — "IBM Artificial Intelligence Unit (AIU)" (2022 lineage chip) | https://research.ibm.com/blog/ibm-artificial-intelligence-unit-aiu | Lineage |
| IBM Newsroom — Spyre commercial availability (2025-10-07) | https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability | Availability |
| IBM Docs — "Introduction to the Spyre Accelerator" (IBM z 9175-ME1) | https://www.ibm.com/docs/en/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator | Product Docs |
| IBM Research AI-hardware blog index | https://research.ibm.com/blog?tag=ai-hardware | Overview |

### Academic

| Resource | URL | Category |
|----------|-----|----------|
| ISSCC 2026 — "Spyre: An Inference-Optimized Scalable AI Accelerator for Enterprise Workloads" (abstract + author list) | https://research.ibm.com/publications/spyre-an-inference-optimized-scalable-ai-accelerator-for-enterprise-workloads | Research |
| DBLP record (DOI 10.1109/ISSCC49663.2026.11409090, pp. 52–54) | https://dblp.org/rec/conf/isscc/CohenKVSVZCRSGWLHKMNSHG26 | Bibliographic |

### Developer Documentation (IBM-authored)

| Resource | URL | Category |
|----------|-----|----------|
| torch-spyre developer docs — root | https://torch-spyre.readthedocs.io/en/latest/ | SDK Docs |
| IBM Spyre device (cores, corelets, 2 MB LX, 128 B sticks, ring 128 B/cycle/dir, >300 TOPS) | https://torch-spyre.readthedocs.io/en/latest/architecture/spyre_accelerator.html | Hardware Spec |
| Dataflow accelerator architecture ("there is no hardware cache") | https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html | Execution Model |
| Key concepts | https://torch-spyre.readthedocs.io/en/latest/getting_started/key_concepts.html | SDK Docs |
| Glossary (LX, PT, SFU, stick, SDSC, DeepTools, flex, SENCORES) | https://torch-spyre.readthedocs.io/en/latest/getting_started/glossary.html | SDK Docs |
| Installation ("instructions to be released once the runtime reaches public availability") | https://torch-spyre.readthedocs.io/en/latest/getting_started/installation.html | SDK Docs |
| Compiler stack overview (open vs proprietary split) | https://torch-spyre.readthedocs.io/en/latest/compiler/architecture.html | Compiler Docs |
| Inductor front-end deep dive (16-pass pipeline, modules) | https://torch-spyre.readthedocs.io/en/latest/compiler/inductor_frontend.html | Compiler Docs |
| Back-end compiler (DeepTools, `dxp_standalone`) | https://torch-spyre.readthedocs.io/en/latest/compiler/backend.html | Compiler Docs |
| KTIR in the pipeline | https://torch-spyre.readthedocs.io/en/latest/compiler/ktir.html | Compiler Docs |
| Scratchpad (LX) planning (2 MB, ~1.6 MB usable, four solvers, no eviction) | https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html | Compiler Docs |
| Work-division planning (256 MB span, cost model, SPMD encoding) | https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html | Compiler Docs |
| Coarse-tiling loop IR | https://torch-spyre.readthedocs.io/en/latest/compiler/coarse_tiling_loops.html | Compiler Docs |
| Indirect access (gather) | https://torch-spyre.readthedocs.io/en/latest/compiler/indirect_access.html | Compiler Docs |
| Working-set reduction | https://torch-spyre.readthedocs.io/en/latest/compiler/working_set_reduction.html | Compiler Docs |
| Span-overflow hint analysis | https://torch-spyre.readthedocs.io/en/latest/compiler/span_overflow_hint_analysis.html | Compiler Docs |
| Tensor layouts (SpyreTensorLayout, stickification, DCI) | https://torch-spyre.readthedocs.io/en/latest/user_guide/tensors_and_layouts.html | SDK Docs |
| Supported operations | https://torch-spyre.readthedocs.io/en/latest/user_guide/supported_operations.html | SDK Docs |
| Profiling | https://torch-spyre.readthedocs.io/en/latest/user_guide/profiling/index.html | SDK Docs |
| Runtime overview (DeepRT, Flex, PF/VF, spyreccl) | https://torch-spyre.readthedocs.io/en/latest/runtime/index.html | Runtime Docs |

### Open-Source GitHub Repositories — `torch-spyre` org

| Resource | URL | Category |
|----------|-----|----------|
| torch-spyre org (14 repos) | https://github.com/orgs/torch-spyre/repositories | Overview |
| torch-spyre/torch-spyre — PyTorch PrivateUse1 backend + Inductor front end (Apache-2.0) | https://github.com/torch-spyre/torch-spyre | Framework / Compiler |
| torch-spyre/sendnn-inference — production vLLM plugin (formerly `vllm-project/vllm-spyre`; Apache-2.0) | https://github.com/torch-spyre/sendnn-inference | Serving |
| torch-spyre/spyre-inference — 2nd-gen vLLM plugin built on `torch-spyre` (Apache-2.0) | https://github.com/torch-spyre/spyre-inference | Serving |
| torch-spyre/hf-adapters — HuggingFace enablement by load-time monkey-patching (Apache-2.0) | https://github.com/torch-spyre/hf-adapters | Framework |
| torch-spyre/sglang-spyre — early SGLang port (no license file) | https://github.com/torch-spyre/sglang-spyre | Serving |
| torch-spyre/triton — Triton fork with the Spyre TTIR→KTIR target (MIT) | https://github.com/torch-spyre/triton | Kernel Language |
| torch-spyre/spyre-kernels — Triton kernels + committed `.ttir`/`.ktir`; T0–T3 validation tiers | https://github.com/torch-spyre/spyre-kernels | Kernels |
| torch-spyre/ktir-mlir-frontend — KernelTile IR (`ktdp` MLIR dialect) parser + Python bindings (Apache-2.0) | https://github.com/torch-spyre/ktir-mlir-frontend | IR |
| torch-spyre/ktir-cpu — CPU interpreter / validator / latency model for KTIR (Apache-2.0) | https://github.com/torch-spyre/ktir-cpu | Simulator |
| torch-spyre/dataflow-scheduler — KTIR → KTDF/KTDFLow → DFIR scheduling infrastructure (Apache-2.0) | https://github.com/torch-spyre/dataflow-scheduler | Compiler |
| torch-spyre/dataflow-scheduler-mlir-dialects — MLIR dialects for the scheduler (Apache-2.0) | https://github.com/torch-spyre/dataflow-scheduler-mlir-dialects | Compiler |
| torch-spyre/aiu-bench — self-hosted AIU compiler performance suite (Apache-2.0) | https://github.com/torch-spyre/aiu-bench | Benchmarks |
| torch-spyre/interface-specs — compiler↔runtime↔device interface specifications | https://github.com/torch-spyre/interface-specs | Open Spec |
| torch-spyre/RFCs — public design RFCs | https://github.com/torch-spyre/RFCs | Open Spec |

### Open Specifications (published even where the consumer is closed)

| Resource | URL | Category |
|----------|-----|----------|
| SuperDSC Bundle spec (MLIR + `sdsc.json`; 70+ OpFuncs; stick constraints; folds) | https://github.com/torch-spyre/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md | IR Spec |
| SpyreCode spec (job execution/preparation plans, `init.bin`, 128 GB / 8×16 GB segments, program correction) | https://github.com/torch-spyre/interface-specs/blob/main/0277-SpyreCode/0277-SpyreCodeSpec.md | Binary Container |
| ProgramExecution spec (SpyreStream, RuntimeStream, JobPlan, CompositeAddress, PF/VF) | https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/ProgramExecutionSpec.md | Runtime Spec |
| StreamSynchronization spec | https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/StreamSynchronizationSpec.md | Runtime Spec |
| RFC 0047 — Tiled Tensors | https://github.com/torch-spyre/RFCs/blob/main/0047-TiledTensors/0047-TiledTensorsRFC.md | Tensor API |
| RFC 0099 — Multi-Device (spyreccl, Spyre Comms, on-node only) | https://github.com/torch-spyre/RFCs/blob/main/0099-MultiDevice/0099-MultiDeviceRFC.md | Collectives |
| RFC 0171 — Spyre Device (PrivateUse1, sticks, allocator, VF mode) | https://github.com/torch-spyre/RFCs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md | Runtime |
| RFC 0601 — Spyre Profiling Toolkit (dual memory hierarchy, libaiupti, AIU SMI) | https://github.com/torch-spyre/RFCs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md | Profiling |
| RFC 0682 — KTIR Spec (merged March 2026) | https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md | IR Spec |
| RFC 1069 — Spyre Tensor Layout Extraction | https://github.com/torch-spyre/RFCs/blob/main/1069-SpyreTensorLayoutExtraction/1069-SpyreTensorLayoutExtraction.md | Layouts |
| RFC 1358 — Coarse Tiling | https://github.com/torch-spyre/RFCs/blob/main/1358-CoarseTiling/1358-CoarseTiling.md | Compiler |
| RFC 2676 — Spyre Metrics API Extension | https://github.com/torch-spyre/RFCs/blob/main/2676-SpyreMetricsApiExtension/2676-SpyreMetricsApiExtensionRFC.md | Telemetry |
| RFC 2696 — AIU SMI Extension (nvidia-smi analogue; PF/VF; x86/Power/Z) | https://github.com/torch-spyre/RFCs/blob/main/2696-AiuSmiExtension/2696-AiuSmiExtensionRFC.md | Telemetry |
| RFC 2971 — FP32 Element Arrangement | https://github.com/torch-spyre/RFCs/blob/main/2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md | Datatypes |

### Serving Documentation

| Resource | URL | Category |
|----------|-----|----------|
| vLLM Spyre plugin documentation | https://docs.vllm.ai/projects/spyre/en/latest/ | Serving Docs |
| vLLM Spyre — installation (CPU-only dev path; "full `torch_sendnn` stack only available pre-installed") | https://docs.vllm.ai/projects/spyre/en/latest/getting_started/installation.html | Serving Docs |
| vLLM Spyre — supported-features matrix | https://docs.vllm.ai/projects/spyre/en/latest/user_guide/supported_features.html | Serving Docs |

### Cluster / Orchestration — `ibm-aiu` org (Go, Apache-2.0)

| Resource | URL | Category |
|----------|-----|----------|
| ibm-aiu org | https://github.com/ibm-aiu | Overview |
| ibm-aiu/spyre-operator — OpenShift operator for Spyre cards | https://github.com/ibm-aiu/spyre-operator | Orchestration |
| ibm-aiu/spyre-device-plugin — Kubernetes device plugin | https://github.com/ibm-aiu/spyre-device-plugin | Orchestration |
| ibm-aiu/dra-driver-spyre — Kubernetes Dynamic Resource Allocation driver | https://github.com/ibm-aiu/dra-driver-spyre | Orchestration |
| ibm-aiu/spyre-health-checker | https://github.com/ibm-aiu/spyre-health-checker | Orchestration |
| ibm-aiu/spyre-scheduler-plugins | https://github.com/ibm-aiu/spyre-scheduler-plugins | Orchestration |
| ibm-aiu/spyre-webhook-validator | https://github.com/ibm-aiu/spyre-webhook-validator | Orchestration |
| ibm-aiu/spyre-operator-docs | https://github.com/ibm-aiu/spyre-operator-docs | Docs |

### Legacy / Adjacent (including negative findings)

| Resource | URL | Category |
|----------|-----|----------|
| foundation-model-stack/aiu-fms-testing-utils — legacy FMS/`torch_sendnn` path; documents `FLEX_*`/`DT_*` env vars | https://github.com/foundation-model-stack/aiu-fms-testing-utils | Framework |
| IBM/zDNN — Telum / Telum II zAIU NNPA library (**does NOT target Spyre**) | https://github.com/IBM/zDNN | Negative finding |
| IBM/zDLC — IBM Z Deep Learning Compiler, ONNX-MLIR based (**Telum/NNPA only**) | https://github.com/IBM/zDLC | Negative finding |
| IBM zDLC documentation | https://ibm.github.io/zDLC/ | Negative finding |
| IBM Z Deep Learning Compiler 5.0.0 announcement (z17 / Telum II via MLz on z/OS) | https://community.ibm.com/community/user/blogs/sunny-anand/2025/06/10/ibm-z-deep-learning-compiler-500 | Adjacent |

### Secondary Technical Analysis

| Resource | URL | Category |
|----------|-----|----------|
| Dr. Ian Cutress (More Than Moore) — "IBM's Spyre AI Accelerator Deep Dive" (die size, per-dtype TOPS, PCIe Gen5, RDMA, dual-loop power, design timeline; ISSCC-attributed) | https://morethanmoore.substack.com/p/ibms-spyre-ai-accelerator-deep-dive | Analysis |
| The Register — Hot Chips 2024 coverage (origin of the **stale** "8 cards / 256 cores" figure) | https://www.theregister.com/on-prem/2024/08/27/ibm-details-upcoming-chips-to-support-ai-on-mainframes/ | Stale secondary |

---

## Key Findings

- **Status: shipping (GA), sold as a priced system option.** Announced 2025-10-07; GA 2025-10-28 on IBM z17
  and LinuxONE 5; GA "early December 2025" on IBM Power11. **No named external at-scale production
  deployment is public** — pre-GA validation is documented only at IBM Yorktown Heights and the University
  at Albany Center for Emerging AI Systems. Record as *shipping (GA)*, **not** *deployed at scale*.
- **Compute**: 32 active AI cores (34 physical, 2 spares for yield), 2 corelets per core, each corelet an
  8×8 SIMD-systolic PE array (64 "low-precision math engines") plus 1D vector/SFU units — 4,096 math
  engines per card. fp16 / fp8 / int8 / int4 on the 2D arrays; fp32 added on the 1D vector path.
  IBM primary states only **">300 TOPS per card at 75 W"**; the per-datatype breakdown is secondary.
- **Memory: software-managed, no hardware cache.** 2 MB LX scratchpad per core (~1.6 MB usable),
  64 MB on-chip total (arithmetic), 128 GB LPDDR5 per card over 16 channels @ 6.4 Gbps ≈ 204 GB/s.
  Transfers happen in 128-byte "sticks" — a direct inheritance of the zAIU/zDNN "stickified tensor" layout.
  There is **no eviction mechanism** for LX: "This is deliberate."
- **Interconnect**: bidirectional on-chip ring at 128 B/cycle/direction (aggregate GB/s not derivable —
  clock not disclosed); **no proprietary chip-to-chip fabric** — cards scale out over a standard PCIe switch
  fabric with direct card-to-card RDMA. Chassis limits: **48 cards on z17/LinuxONE 5, 16 on Power11**;
  **single-model ensembles cap at 8 cards / ~1 TB**. The widely repeated "8 cards / 256 cores" is IBM's
  **August 2024 Hot Chips preview**, not the shipping product.
- **Execution model**: statically scheduled dataflow, SPMD across cores, static shapes, compile-time work
  division and data staging. No caches, no out-of-order execution, no warp scheduler — the entire
  performance model rests on the compiler.
- **Software stack is split at a precise seam.** Open: PyTorch PrivateUse1 backend, TorchInductor front end
  (16-pass LoopLevelIR pipeline), Triton fork, KTIR MLIR dialect + CPU interpreter, dataflow scheduler,
  vLLM plugins, all Kubernetes/OpenShift tooling, and the SuperDSC / SpyreCode / ProgramExecution
  **interface specs**. Closed: **DeepTools** back-end compiler, **DeepRT**/**Flex** runtime, `torch_sendnn`
  distribution, `spyre_comms`, `libaiupti`, `spyremetrics`, kernel driver, firmware, and the **ISA**.
  The runtime is not even publicly downloadable — installation docs say instructions arrive "once the
  runtime reaches public availability."
- **Negative finding**: there is **no public ONNX-based or z/OS-native path to Spyre**. zDLC and zDNN target
  the Telum / Telum II zAIU via the NNPA instruction only. Spyre's public path is PyTorch/vLLM on Linux on
  Z, Linux on Power, and Red Hat OpenShift.
- **Rare for a survey**: `ktir-cpu` is an open CPU interpreter, validator and latency model for the tile IR,
  explicitly framed as "an environment and reward model for AI-driven compiler development" — making the
  Spyre programming model studiable without hardware.
