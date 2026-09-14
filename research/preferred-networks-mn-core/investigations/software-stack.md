# Preferred Networks MN-Core Software Stack Investigation

*as_of: 2026-09-13*
*chip: preferred-networks-mn-core*
*device_class: Compiler-Scheduled SIMD Accelerator (Japan)*

---

## Overview

The MN-Core software stack is **MLSDK** (machine learning) plus **HPCSDK** (general-purpose C/C++, alpha). Both live inside a container at `/opt/pfn/pfcomp/{fx2onnx,pfvm,mncl,codegen}`.

**Bottom line on openness:** the stack is **overwhelmingly proprietary**. Everything load-bearing — PFVM, the codegen graph compiler and code emitter, the runtime, the operator library, the assembler, the emulator, the user-space driver and the kernel module — ships as **binary `.deb` packages** from a private Google Artifact Registry APT repository, under an **End-User License Agreement** (`MN-Core_SDK_End-User_License_Agreement.pdf`, shipped inside the container at `/opt/pfn/licenses/`). The only Apache-2.0 artifact is a thin repository of Dockerfiles and examples.

**Documentation, however, is unusually good and completely open** — including a full ~130-page ISA manual, a freely downloadable assembler and cycle-faithful emulator, an educational from-scratch graph compiler, and a public assembly-optimization contest. That combination — closed toolchain, open ISA — makes MN-Core the most *reproducible* proprietary stack in this registry.

**Pipeline in one line:**

```
PyTorch (custom torch build)
  → fx2onnx (torch.fx + FakeTensor) → Exported ONNX
  → PFVM Compiler                    → Compiled ONNX
  → codegen Graph Compiler (L3IR)    → MNGraph  (Dtype / Location / Layout assigned)
  → codegen Code Emitter             → GPFNApp  (FlatBuffers-packed VSM + relocation info)
  → codegen runtime                  → libgpfn3 → gpfn3.ko → MN-Core 2 board
```

---

## Layer 1: Framework Integration

| Component | Detail | Open? |
|---|---|---|
| **PyTorch** | The one supported framework. MLSDK ships a **custom-built `torch`** (2.9.0 in SDK 0.7, with torchvision 0.24.0), installed from PyTorch's CPU wheel index and then patched with MLSDK extensions. The Dockerfile warns: *"the SDK won't work with PyTorch's official Python packages. It uses our own custom build"* — you cannot substitute a different torch version or use a user venv in the same environment. | ❌ proprietary build |
| **`pytorch-pfn-extras` 0.9.0** | PFN's own OSS training-loop/extensions library, pulled in as a dependency. | ✅ Apache-2.0 |
| **HuggingFace `transformers`** | Used by PFN's own LLM examples (Llama / Qwen2 / PLaMo pipeline-parallel inference, SLM SFT). | ✅ external |
| **JAX** | PFN's marketing pages claim the compiler ingests graphs "defined with high-level languages such as PyTorch and **JAX**". **JAX appears nowhere in the MLSDK 0.7 documentation, the GitHub repo, or the PFCP docs.** Treat JAX support as an unsubstantiated marketing statement. | — (unverified) |
| **No vLLM / TensorRT-LLM / ONNX Runtime EP** | LLM inference is an **experimental** workload category; PFN's own path is hand-rolled pipeline parallelism over MPI + gloo. | — |

### Programming model

You do not get an eager device. You isolate a **pure function** of type `Callable[[Dict[str, Tensor]], Dict[str, Tensor]]`, register parameters and buffers with a `Context`, and compile it. The whitepaper's canonical diff from a CPU ResNet-50 script is one import plus one line:

```python
import mncore
train_step = mncore.compile(model_with_loss, backward=True, optimizer=optimizer)
```

(the current API is `mlsdk.Context(...).compile(fn, sample, storage.path(...))`).

**Forward, backward and the optimizer step are all compiled into a single device program** — a direct consequence of there being no host-side control flow per op and no device-side branching.

### Portability story

A device string, same source:

| Device string | Meaning | Use |
|---|---|---|
| `pfvm:cpu` | PFVM Runtime executing Compiled ONNX with LibTorch on CPU | Reference execution of the *rewritten* graph |
| `pfvm:cuda` | Same, on GPU | Faster numerical verification |
| `emu2` | MN-Core 2 software emulator | Recommended development target |
| `mncore2:auto` | Real hardware (index or `auto`) | Production |

PFN explicitly recommends validating on `pfvm:cpu` / `pfvm:cuda` first, because the PFVM graph rewrite is the hardest stage to debug. The four-way split (PyTorch graph vs Compiled graph) × (LibTorch vs codegen/layers) gives four checkable combinations to bisect a numerical regression.

### Coverage evidence

PFN publishes a HuggingFace `timm` backbone survey (SDK v0.4): **378 compilable, 369 inference-ready, 156 training-ready** models, filterable by CNN / ViT / Hybrid.

Advertised workload categories: image classification (ResNet / ViT / ConvNeXt / EfficientNet), detection (SSD, DETR, FCOS, YOLACT), Stable Diffusion (training and eval examples in-repo), LLM training/fine-tuning (LLaMA 8B, Qwen2, PLaMo), **LLM inference (experimental)**, speech (SqueezeFormer, Whisper — "example coming soon"), robotics diffusion policy ("coming soon").

---

## Layer 2: Graph Capture

| Component | Detail | Open? |
|---|---|---|
| **FX2ONNX Exporter** | PFN's own exporter at `/opt/pfn/pfcomp/fx2onnx`. Uses `torch.fx` symbolic tracing with `FakeTensor`s (no real compute during trace) to emit **Exported ONNX**. Documented limitation: control flow that branches on tensor presence may not trace correctly. | ❌ binary |
| **`fx2onnx.linter`** | Public API (`fx2onnx.linter.lint`, `LintLevel`, `LintResult`) added in SDK v0.6 that pre-flights a model for exportability and reports actionable issues. A genuinely unusual and useful stack feature. | ❌ binary, documented API |
| **Legacy `torch.onnx` path** | Selectable with `MNCORE_USE_LEGACY_ONNX_EXPORTER=1`; **deprecated**, limited support, still used by some shipped examples (e.g. the pipeline-parallel Llama script). | — |
| **Static shapes only** | "we currently do not support graphs that have Dynamic Shapes in their input/output structures." By the time the graph reaches MNGraph, all shapes are static. | — |

**The IR spine is ONNX all the way down.** Exported ONNX → Compiled ONNX → MNGraph are all ONNX-based structures that can be dumped to `.onnx` files. This is unusual — most modern stacks use MLIR; PFN standardized on ONNX as the interchange format in 2021 and stayed there.

---

## Layer 3: Graph Compiler #1 — PFVM

`/opt/pfn/pfcomp/pfvm`. PFVM is *both a compiler and a runtime* for Exported ONNX.

- **PFVM Compiler**: constant propagation, common-subexpression elimination, operator fusion, replacement with backend-specific specialized operators; it also **injects things the exported graph lacks**, notably weight-update handling as custom ONNX operators. Output = **Compiled ONNX**.
- **PFVM Runtime**: executes Compiled ONNX directly with each operator implemented in **LibTorch** — this is what `pfvm:cpu` and `pfvm:cuda` are.
- **Design rationale (PFN's own)**: the PFVM rewrite is large and hard to validate, so the pipeline is deliberately split into PyTorch Graph vs Compiled Graph, and LibTorch vs codegen/layers implementations, giving four checkable combinations.

❌ proprietary. Background: PFN tech blog on the PFVM ONNX exporter.

---

## Layer 4: Graph Compiler #2 — codegen Graph Compiler (L3IR)

`/opt/pfn/pfcomp/codegen`. Input **Compiled ONNX** → output **MNGraph**, an ONNX extension composed of **MNNode** and **MNValue** objects. Dumped as `l3ir.txt` (nodes in scheduling order) and `l3ir_stripped.onnx` — the historical name **L3IR** survives in the filenames.

### The three MNValue properties — this trio *is* the MN-Core programming model

1. **Dtype** — numeric precision. Mixed precision managed graph-wide; GEMM/conv may be demoted to Half, BatchNorm restored to higher precision.
2. **Location** — **DRAM or LM0/LM1 only.** GRF and L1BM/L2BM are never Locations.
3. **Layout** — how the tensor is mapped across the physical memory *tree*.

The Layout notation is the most distinctive artifact in this stack:

```
(64,128)/((8_L2B:1, 8:2), (16_MAB:1, 2:1, 4_PE:1); B@[L1B,W])
```

Read as: tensor shape (64,128); axis 0 split 8-ways across **L2B** with address stride 2; axis 1 split 16-ways across **MAB** and 4-ways across **PE**; **broadcast** across L1B; `B@[…,W]` marks 64-bit word packing. Levels are `{PE, W, Addr, MAB, L1B, L2B}` plus `Time` (see Time-Slice below). Sub-terms are called "Axis" / "Subaxis".

### Pass pipeline (PFN's documented order)

| # | Pass | What it decides |
|---|---|---|
| 1 | **Dtype Planner** | Assign precisions; rough LM-consumption estimate |
| 2 | **Location Planner (initial)** | Pin things that must be in DRAM (e.g. parameters) |
| 3 | **Layout Planner** | Assign Layouts to satisfy per-node requirements; inserts **`MNCoreLayoutSwitch`** on conflict. **Layout Planner Z (`lpz`)** is the recommended variant and costs LayoutSwitch operations into its objective |
| 4 | **Location Planner** | Finalize now that Layout determines LM footprint; inserts **`MNCoreUpload` / `MNCoreDownload`** (LM↔DRAM) on conflict |
| 5 | **Time-Slice** | For MNValues whose `num_lw` exceeds LM capacity (**2048 LW on MN-Core 2**), add a `Time` subaxis to the Layout and split the node, inserting `MNCoreInputSplit` / `MNCoreOutputConcat` |
| 6 | **Location Planner (sliced)** | Re-adjust after slicing |
| 7 | **Node Simulation** | Actually compile each MNNode under multiple configurations to **measure real cycle counts and memory usage**. Modes: `fake` (estimate only) / `default` / `fast` / `best` (try all patterns) / `full`. Cached via `mlsdk.CacheOptions(enable_codegen_cache=...)`; dumped to `simulation_result.json` |
| 8 | **Scheduler** | Topological order plus DRAM spill/refill placement, LM0↔LM1 moves, `Forget` operations, and **recomputation** decisions. Options: `always_from_dram` (debug), `reuse_consecutive` (**default**), `spill_opt`, **`auto_recompute_sa`** (simulated annealing over recompute + schedule). Env knobs: `CODEGEN_SA_STEPS`, `CODEGEN_NUM_SA_THREADS`, `CODEGEN_SCHEDULE_PP_BEAM_WIDTH` |
| 9 | **Address Planner** | Assign concrete addresses once lifetimes are known |
| 10 | **Gene propagation** | Carry layout constraints from strongly-constrained nodes (matmul, conv) to distant nodes |

PFN's rationale for recompute: *"particularly effective for the MN-Core architecture where SRAM capacity is limited and data transfer costs between LM and DRAM are significant."*

### Optimization presets and tuning surface

`/opt/pfn/pfcomp/codegen/preset_options/`: `O0.json` (just make it compile) → `O1` (conservative, same scheduler as O2+) → `O2` (simulated annealing) → `O3` (wider search + **L1Merge**) → `O4` (wider still) → `debug.json`. Plus `CODEGEN_FIND_BEST_COMPILE_OPTIONS` / `CODEGEN_FIND_BETTER_COMPILE_OPTIONS` autotuners.

There are **~35 documented `CODEGEN_*` environment variables**, including model-specific hacks such as **`CODEGEN_AUTO_RECOMPUTE_HACK_FOR_QWEN`** — a candid signal of how much per-model tuning this compiler still needs.

Compile options exposed through the Python API: `option_json`, `float_dtype` (`float`/`mixed`), `layout_planner`, `scheduler`, `simulation_mode`, **`sram_budget`** (0–1 fraction of LM the compiler may use), `sa_expected_run_iters`, `out_onnx`, `layout_spec`, `gemm_layout_spec`, `num_threads`.

❌ proprietary. (See Layer 6(d) for the *educational* open reimplementation.)

---

## Layer 5: Kernel Compiler — codegen Code Emitter

MNGraph → **GPFNApp**. Two steps: compile each MNNode independently, then link.

- **Concat strategy** — because the Address Planner already made addresses globally consistent, linking is literal concatenation of per-node assembly.
- **L1Merge strategy** — merges two instruction sequences into one when they use disjoint resources (e.g. a DRAM↔LM-heavy sequence and a compute-heavy sequence), shortening the stream. Applied repeatedly. Blocked by in-place I/O, in which case `MNCoreReorderAddress` is inserted.
- **Operator implementations live in `codegen/layers` and are not shipped as source** — "The implementations in `codegen/layers` are not yet included in the MN-Core SDK image and are instead bundled with the pre-built codegen libraries." **Adding an unimplemented operator is therefore not currently possible for users**; PFN says it is "actively preparing the framework to enable feature expansions", with no timeline.
- Historically this layer was described as **L2IR / Generic Conv / Layer** (MNTensor, PEVector, Generic Move Impl) and **L1IR** (Layer Impl, L1-level instruction merger, L1IR scheduling graph) — see the 2021 PFN blog.

❌ proprietary.

---

## Layer 6: Kernel Language / Low-Level Programming

Four distinct entry points, plus a contest.

### (a) MN-Core 2 assembly (VSM) — fully public ISA

The **MN-Core 2 Software Developer Manual** (EN/JA, revised **2026-06-02**, first edition 2024-11-15) is a complete ~130-page ISA reference covering:

- Statement / immediate / tag syntax; `quit`, `d get`, `d set` control statements
- **MV instruction** statements: ~24 documented transfer modes (PDM↔DRAM↔L2BM; individual / parallel / intra-group broadcast / inter-group broadcast / reduce / collect-reduce / distribute / collect), each with documented LW/cycle throughput, DRAM indirection, priority, and inter-MV constraints
- **PE instruction** statements: operand namespaces (`$p`, `$d`, `$c`, `$b`, `$m`, `$n`, `$r`, `$s`, `$t`, `omr`, `x`/`y` matrix registers, `dar` DRAM address register, forwarding operands `mauf` / `aluf` / `lbf` / `mreadf` / `nowrite`), mask registers, **hazard-avoidance rules**, L1BM/L2BM instruction expressions (broadcast, 4×4 reductions, distribute/collect), **MAU instructions** (`dmfma` / `dmmul` / `fmfma` / `fmmul` / `gmfma` / `gmmul` / `hmfma` / `hmmul` matrix ops; `dvfma` / `fvfma` / `hvfma` / `dvadd` / `fvpassa`… vector ops), matrix-register write and transposed-read instructions, ALU instructions (incl. `imm`, inter-PE circular shift `msl` / `msr`), and `wait`

✅ **Documentation is public and free.** ❌ The toolchain binaries are closed.

### (b) `assemble3` + `gpfn3_package_main` — assembler and emulator

Distributed **free, no login**, as `mncore2_emuenv_20240826.tar.xz`, and also bundled inside the SDK at `/opt/pfn/pfcomp/codegen/build/tools/` (since SDK v0.6). Verified: both are **stripped x86-64 ELF binaries**, shipped with a bare "AS IS" DISCLAIMER — no source, no permissive licence.

Workflow:

```
assemble3 --instruction-mode flat out.vsm > out.asm
gpfn3_package_main -i out.asm -d dump.txt
```

The emulator relaxes the instruction-packing requirement and supports `d get` debug dumps of any memory element, which is why it is the recommended development target.

### (c) HPCSDK / MNCL — general-purpose C/C++

`/opt/pfn/pfcomp/mncl`. "Provides a general-purpose programming environment in C/C++ with **OpenCL-like and directive-based** programming models. Currently, only **MNCL**, an OpenCL-like environment, is provided."

**Status: alpha. No published documentation** — the repo says "please refer to the header files included with the SDK." The 2023 whitepaper described a **subset of OpenACC** (with MN-Core-specific `l2(...)` / `l1(...)` data clauses on `#pragma acc data`) and a **subset of OpenCL**, both "currently under development"; as of SDK 0.7 the OpenACC path still has no public docs. ❌ proprietary, ⚠️ immature.

### (d) `mncore_simple_graph_compiler_for_education`

✅ **The one substantive open code artifact.** A from-scratch teaching graph compiler (`github.com/pfnet/mncore_simple_graph_compiler_for_education`) that takes a PyTorch MLP through `torch.fx` → ONNX → MN-Core 2 assembly and trains MNIST on the emulator. Contains `fx_export/{export,compile,mncore_transform,vsm_converter,mncore_utils,train,traincpp}.py`, a C++ `matrix_operations.hpp` shim, and ~40 hand-written `.vsm` test cases (matmul with/without transpose, DRAM up/download at various layouts, row/col reductions, `relu_grad`, `gather_sum`).

⚠️ **No LICENSE file is present in the repository** — describe it as *publicly readable*, never as "open-source-licensed".

### (e) MN-Core Challenge

A public competitive-programming contest (2024) where entrants hand-optimize MN-Core 2 assembly against a judge, with an official tips section and errata against the SDM. https://mncore-challenge.preferred.jp/

---

## Layer 7: Tensor API

The `mlsdk` Python package. Documented public surface (MLSDK 0.7 API Reference):

| Object | Surface |
|---|---|
| **`MNDevice(device_name)`** | Colon-separated device string: `mncore2:auto`, `emu2`, `pfvm:cuda`, `pfvm:cpu`; index or `auto` |
| **`Context`** | `compile()`, `compile_automap()`, `load_codegen_dir()`, `register_param()`, `register_buffer()`, `register_optimizer_buffers()`, `get_registered_value_proxy()`, `switch_context()`, `synchronize()` |
| **`CompiledFunction`** | Callable replacement for the original Python function; `allocate_input_proxy()` |
| **`TensorProxy` / `TensorLike`** | *The device tensor type.* `TensorProxy.cpu()` pulls back to host; `TensorProxy.load_from()` pushes a `torch.Tensor` (or another proxy) into device memory, with an option to copy so the source can be mutated. This is how you avoid host round-trips between iterations |
| **`CacheOptions`** | Persist Node Simulation and GPFNApp artifacts across runs (essential — O2+ compiles are slow) |
| **Optimizers** | `MNCoreOptimizer`, **`MNCoreSGD`**, **`MNCoreAdam`**, **`MNCoreAdamW`**, `MNCoreLRScheduler` — device-side fused optimizers compiled into the same program as fwd+bwd. PFN warns `MNCoreAdamW` is **not bit-compatible with LibTorch** |
| **Naming** | `set_tensor_name()`, `set_tensor_name_in_module()`, `set_buffer_name_in_optimizer()`, `get_tensor_name()` — required so the Context can distinguish parameters of different models |
| **Profiling** | `trace_event()`, `trace_scope()` → `trace.json`, viewable in **Perfetto UI** / Chrome Tracing |
| **Storage** | `mlsdk.storage.path()` / `mlsdk.path()` |

❌ proprietary.

---

## Layer 8: Runtime

### GPFNApp — the deployable artifact

The **codegen runtime** loads and executes **GPFNApp**, a **FlatBuffers**-packaged object containing:

- the **VSM** (pre-compiled assembly, a.k.a. GPFNBin)
- input/output node metadata (name, Dtype, Layout, Address)
- the original model parameters
- **relocation info** — emitted code has address fields left blank and is relocated against the live Context

Inspectable with the shipped `dump_gpfnapp` tool.

Because compilation (Node Simulation + SA scheduling) is expensive, the runtime is explicitly **designed around GPFNApp reuse**: "particularly suitable for workloads that repeatedly execute the same computations, such as training loops or batch inference." PFN states the corollary weakness outright: "it performs poorly with operations requiring extensive indexing… it generally excels at operations with high spatial locality."

### Host↔device data movement

All traffic goes through **Group 0's PDM** over PCIe. Host-side **transposition into the device Layout happens on the CPU during upload** — PFN flags this as a performance consideration for large inputs. Runtime errors documented in the FAQ reveal implementation detail: pinned-memory "IDMA chunk" allocation, a device **lock** (one program per board; `gpfn3-smi reset` breaks it), and a requirement for **`CAP_SYS_NICE`** to set CPU affinity (`docker run --cap-add=SYS_NICE`).

### Debugging and observability

`codegen_dir` collects `report.json`, `out.txt` / `out.json`, `model.onnx`, `model.app`, `model.vsm`, per-planner `layout.*` dumps, `l3ir.txt`, `l3ir_stripped.onnx`, `simulation_result.json` and `trace.json` — browsable in a shipped **Codegen Dashboard** with a Netron graph view, per-node view, logs, and Perfetto UI / Chrome Tracing panes.

### Communication

**There is no collectives library.** No NCCL / RCCL / CNCL analogue exists. Multi-board scaling uses **OpenMPI** (`libopenmpi3` / `openmpi-bin` are in the SDK image) with `torch.distributed` on the **`gloo`** backend and host-memory send/recv — see `sdk/examples/run_pp_llama.py` (pipeline-parallel Llama prefill+decode across N boards) and `run_llm_infer_pp.sh`. Whether a device-side collective library is planned is **not disclosed**.

❌ proprietary.

---

## Layer 9: Driver / Firmware / Platform

### APT packages

Repository **`mncore-packages`**, hosted at `https://asia-northeast1-apt.pkg.dev/projects/mncore-packages` (Google Artifact Registry), added via `apt/add_mncore_packages.sh` in the public GitHub repo:

| Package | Role |
|---|---|
| **`gpfn3-dkms`** | Linux **kernel module** (DKMS-built), module name `gpfn3`. Recognizes MN-Core 2 as a PCIe device (vendor `0ccd`), creates `/dev/mnc2p<bus>s<n>` nodes, and **applies clock/MAB configuration at load time** (settings are volatile across power cycles). Supports Secure Boot via MOK enrollment on Ubuntu 24.04; Ubuntu 22.04 requires disabling Secure Boot |
| **`libgpfn3-0`, `libgpfn3-dev`** | User-space driver and headers |
| **`gpfn3-smi`** | Management/monitoring CLI (the `nvidia-smi` analogue): `list`, `reset <dev>`, `clear <dev>`, `config <dev> clock --core 750 --gddr6 15000`, `config <dev> mab`, `mtest --forward --dmode h --l1 64 --l2 2048 --l3 64 --dram 3968 <dev>` (~1-minute self-test). MLSDK docs say users should need nothing beyond `list` |
| **`gpfn3-loader`** | Loader utility |
| **`mncore-sdk`** | The entire SDK (pinned, `apt-mark hold mncore-sdk`) |

Supported host OS: **Ubuntu 24.04 LTS and 22.04 LTS only** (latest LTS + one previous). A container engine (Docker) is required. Deployment is container-first: `mncore-sdk-minimal:<ver>` and `mncore-sdk-full:<ver>` (adds JupyterLab, VS Code CLI, build tools) built from the public Dockerfiles; `create_dev_ctr.sh` starts a dev container with `/dev/<devname>` mapped in.

### Cloud platform — PFCP

**Preferred Computing Platform (PFCP)**, operated by **Preferred Computing Infrastructure, Inc. (PFCI)** — a PFN / Mitsubishi Corporation / IIJ JV announced 2024-12-23:

- Multi-tenant **managed Kubernetes**; MN-Core 2 exposed as the extended resource **`preferred.jp/mncore2`**
- Node types: **Reserved** (monthly fixed) and **Shared** (pay-as-you-go, PriorityClasses `shared-standard` / `shared-best-effort`, HRQ quotas). Per-MN-Core-2 resource cap: **7000m CPU, 125Gi memory** (consistent with 8 boards per MN-Server 2 with 2×Xeon 8480+ and 1 TB RAM)
- Custom resources: `ReservedNode`, `ClusterWorkspacePreset`, `ParallelJob`, `SealedSecret`; Headlamp dashboard; Grafana / Prometheus / Alertmanager monitoring; OIDC workload identity federation to AWS/GCP; RWO/RWX persistent storage with snapshots
- User guide: https://docs.pfcomputing.com/ (Japanese authoritative; English machine-translated by PLaMo Translate)
- **Pricing not published** (contact-gated)

**Playground**: https://playground.mn-core.com/ — a live LLM fine-tuning service running on PFCP that anyone can try.

---

## Layer 10: ISA

Public and complete — see Layer 6(a). Characteristics that matter for this survey:

- VLIW PE instruction, fixed width, always carries a `wait` field
- **4-cycle "step" granularity, chosen because 1-cycle issue would saturate the host PCIe link**
- Two packing modes (auto-stride 3 PE + 1 MV; flat 2 PE + 1 MV)
- **No branches**; mask-register predication
- **No hazard interlocks**
- **Register-memory, not load-store** — local memories are named directly as operands
- Untyped operands; big-endian storage
- Explicit forwarding operands (`mauf`, `aluf`, `lbf`, `mreadf`, `nowrite`)
- An MV/PE instruction split that maps exactly onto the memory tree (MV above L2B, PE at L2B and below)

**Not disclosed:** the number of ONNX operators supported by codegen, and the authoritative supported-op list.

---

## SDK Release History — Note How Recent Public Availability Is

| Version | Date | Highlights |
|---|---|---|
| v0.2 | 2025 (via PFCP changelog) | Early PFCP release |
| **v0.4** | 2026-02-27 | Compiler perf improvements, MLSDK migration guide, sample projects |
| **v0.5** | 2026-04-28 | "SDK source now public at github.com/pfnet/mncore" — ⚠️ **misleading wording**: what became public is the Dockerfiles + examples, not the compiler/runtime source. User-side container build became the recommended flow |
| **v0.6** | 2026-06-05 | **MN-Core 2 emulator and assembler bundled into MLSDK**; `fx2onnx.linter.lint` API; Stable Diffusion advanced example |
| v0.7 | 2026-07-15 | — |
| — | 2026-06-22 | **MN-Core SDK Hub** developer portal launch (dev.mn-core.com) |
| **v0.8** | **2026-08-26** | **Current.** dev.mn-core.com/news/en/ lists only "MN-Core SDK v0.8 has been released" with no changelog/diff page — the Hub does not publish a version-to-version release-notes document (unlike a GitHub Releases page). The v0.8 Technical Notes, Getting Started, and index pages were checked directly and contain no explicit "what's new in 0.8" content and no dependency version table (torch/torchvision versions not stated, unlike the v0.7-era "2.9.0" figure this survey recorded). **Specific content changes in v0.8 are therefore not disclosed to this survey** beyond the version bump and date itself |

**The public, no-login MN-Core developer story is roughly six months old as of August 2026.** Before 2026 the SDK was effectively PFCP-internal.

---

## Honest Assessment

### Strengths

MN-Core is the cleanest published example of the "compiler owns everything" thesis. The compiler does not merely schedule — it performs **cache replacement** (Location Planner + Scheduler spill/refill/`Forget`), **data layout across a physical topology** (Layout Planner with an explicit `B@[L2B,L1B,MAB,PE]` notation), **capacity management** (Time-Slice), **rematerialization** (`auto_recompute_sa`) and **instruction packing** (VLIW + L1Merge) — decisions a GPU hands to hardware.

The three-property MNValue model (Dtype / Location / Layout) is a genuinely reusable abstraction, and PFN documents both it and the full ISA far more openly than most vendors. The free emulator plus the public SDM plus the educational graph compiler make this the most *reproducible* proprietary stack in this registry.

### Weaknesses

No source for anything load-bearing. **No user extensibility** — you cannot add an operator, because `codegen/layers` is not shipped. One framework, one custom torch build, **static shapes only**, no dynamic-shape or paged-attention story, LLM inference labelled experimental, HPCSDK in alpha with no docs, OpenACC promised in 2023 and still undocumented in 2026. No collectives, no device fabric on MN-Core 2 — multi-board goes through host gloo/MPI. Compile times are long enough that caching is a first-class API. And the presence of `CODEGEN_AUTO_RECOMPUTE_HACK_FOR_QWEN` says plainly that the "compiler handles it" promise still needs per-model rescue.

### Comparison anchor

Groq's TSP is the obvious neighbour, but MN-Core goes further: Groq's compiler schedules a chip that still fetches its own instructions; MN-Core's PEs have **no program counter and no decoder at all**, and the instruction rate is bounded by the PCIe link to the host. The MTIA and Sophgo layer models in this repo both have a hardware-managed or partially-hardware-managed memory path somewhere; MN-Core has none, at any tier, in any generation.

---

## Explicitly Not Disclosed

- MLSDK / HPCSDK licence terms — the EULA PDF ships inside the container at `/opt/pfn/licenses/` and is not published on the web in indexable form
- Licence for `pfnet/mncore_simple_graph_compiler_for_education` (no LICENSE file in the repository)
- HPCSDK / MNCL: any public documentation, API surface, supported OpenCL subset, or OpenACC status beyond "alpha, see the headers"
- Source availability or an extension mechanism for `codegen/layers` operator implementations (PFN says the framework is "being prepared" but gives no timeline)
- The number of ONNX operators supported by codegen, and the authoritative supported-op list
- Whether the MLSDK stack supports JAX (claimed on PFN marketing pages, absent from all SDK documentation)
- Whether a device-side collective communication library is planned
- PFCP pricing (contact-gated; no public rate card)
- MN-Core Technology Conference 25 (2025-12-16, Tokyo Midtown) presentation materials — announced on the MN-Core Challenge site but no public slide archive was located

---

## Update — 2026-09-13

*Window covered: 2026-08-08 → 2026-09-13. Change class: **moderate** (one SDK point release; no hardware spec change). Queries run: `"Preferred Networks new AI chip 2026"`, `"MN-Core next generation"`, `"Preferred Networks AI accelerator announcement 2026"`, `"Preferred Networks SDK release 2026"`, `"MN-Core benchmark"`, `"MN-Core whitepaper"`, `"Preferred Networks IPO"`, `"Preferred Networks Rapidus 2026"`.*

### MN-Core SDK v0.8 released (2026-08-26)

Confirmed directly via https://dev.mn-core.com/news/en/ (checked 2026-09-13): "2026-08-26 — MN-Core SDK v0.8 Released — RELEASE — MN-Core SDK v0.8 has been released." This is the only SDK release in the window (v0.7, 2026-07-15, predates the 2026-08-08 baseline and was already on file). No dedicated changelog page exists on the SDK Hub for either release — checked `technical_notes.html`, `getting_started.html`, and `index.html` under `/sdk/0.8/MLSDK/docs/en/`, none of which contain version-diff content, so **specific v0.8 changes are not disclosed** to this survey beyond the version number and date. The MN-Core 2 hardware itself is unaffected by an SDK release.

### Corporate: IPO reporting (early September 2026) — recorded here as context, not a software-stack change

Independent press (Bloomberg, Crypto Briefing, MarketScreener, DIGITIMES — all dated approximately 2026-09-06/07/08) reports that **Preferred Networks is pursuing an IPO**, with CEO Daisuke Okanohara quoted characterizing the move as necessary to fund MN-Core chip mass production given "the rising cost and scale needed to stay competitive in the global AI race." No funding amount, exchange, or timeline was found in the excerpts retrieved (WebSearch was unavailable this session — quota exhausted; this rests on WebFetch summaries of Bing News results, which is narrower than a full search and did not surface the original Bloomberg article text directly). PFN's own news index (preferred.jp/en/news/) does **not** carry an IPO announcement as of 2026-09-13 — this is being reported by financial press, not yet confirmed by a PFN press release. **Treat as reported-but-not-vendor-confirmed.** No PFN/Rapidus news was found in the window (the only Rapidus-adjacent item found, "Rapidus and Cadence Partner on Agentic AI for Advanced SoC Design," does not mention PFN or MN-Core).

### Not independently re-verified this pass

MN-Core L1000/L1100/L1400 status, MN-Core 2 availability/pricing, and the roadmap graphic were spot-checked (PFN AI Chips business page, fetched 2026-09-13) and appear **unchanged** from the 2026-08-08 baseline — MN-Core L1000 still described as "under development" / prototype stage, MN-Core 2 pricing unchanged (MN-Server 2 ¥20M, Devkit ¥2M). No MN-Core 3 name was found anywhere (the Hot Chips 36 "MN-Core Next" vs. the PFN/Rapidus basic-agreement "new model in the MN-Core series" naming ambiguity recorded at baseline remains unreconciled).

---

## Sources

- [MLSDK 0.7 — Technical Notes](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/technical_notes.html)
- [MLSDK 0.7 — Advanced Features](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/advanced_features.html)
- [MLSDK 0.7 — API Reference](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/api_reference.html)
- [MLSDK 0.7 — Hardware Specification](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/hardware_specification.html)
- [MLSDK 0.7 — Getting Started](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/getting_started.html)
- [MLSDK 0.7 — Porting Tutorial](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/porting_tutorial.html)
- [MLSDK 0.7 — FAQ](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/faq.html)
- [MN-Core SDK Hub](https://dev.mn-core.com/) · [Architecture](https://dev.mn-core.com/architecture/) · [Models/Workloads](https://dev.mn-core.com/models/) · [News](https://dev.mn-core.com/news/en/)
- [MN-Core 2 Software Developer Manual (EN), rev. 2026-06-02](https://projects.preferred.jp/mn-core/assets/mncore2_dev_manual_en.pdf)
- [MN-Core 2 White Paper](https://projects.preferred.jp/mn-core/assets/MN-Core_2_whitepaper_en.pdf)
- [MN-Core Emulator Environment tarball](https://projects.preferred.jp/mn-core/assets/mncore2_emuenv_20240826.tar.xz)
- [github.com/pfnet/mncore](https://github.com/pfnet/mncore)
- [github.com/pfnet/mncore_simple_graph_compiler_for_education](https://github.com/pfnet/mncore_simple_graph_compiler_for_education)
- [github.com/pfnet/pytorch-pfn-extras](https://github.com/pfnet/pytorch-pfn-extras)
- [PFN tech blog — Accelerating Deep Learning Workloads with the MN-Core Compiler (2021)](https://tech.preferred.jp/en/blog/mncore-compiler-1/)
- [PFN tech blog — MN-Core tensor layout (JA)](https://tech.preferred.jp/ja/blog/mn-core-tensor-layout/)
- [PFN tech blog — compiler optimization with recompute (JA)](https://tech.preferred.jp/ja/blog/mncore-compiler-optimization-with-recompute/)
- [PFN tech blog — PFVM ONNX exporter (JA)](https://tech.preferred.jp/ja/blog/pfvm-onnx-exporter/)
- [PFN tech blog — build an MN-Core graph compiler yourself (JA, 2025-12)](https://tech.preferred.jp/ja/blog/mn-core2_graphcompiler_scratch/)

### Added 2026-09-13
- [MN-Core SDK Hub News (EN) — v0.8 release entry, 2026-08-26](https://dev.mn-core.com/news/en/)
- [MLSDK 0.8 — Documentation Index](https://dev.mn-core.com/sdk/0.8/MLSDK/docs/en/index.html)
- [MLSDK 0.8 — Technical Notes](https://dev.mn-core.com/sdk/0.8/MLSDK/docs/en/technical_notes.html)
- [MLSDK 0.8 — Getting Started](https://dev.mn-core.com/sdk/0.8/MLSDK/docs/en/getting_started.html)
- [PFN AI Chips business page (re-checked 2026-09-13)](https://www.preferred.jp/en/business/chips/)
- [PFN News index (EN), re-checked 2026-09-13 — no IPO press release found](https://www.preferred.jp/en/news/)
- Bing News search results (via WebFetch) for Preferred Networks IPO coverage — Bloomberg, Crypto Briefing, MarketScreener, DIGITIMES (all ~2026-09-06/07/08) — original articles not directly fetched; summarized via search-result snippets only
- [MN-Core Challenge](https://mncore-challenge.preferred.jp/)
- [MN-Core Playground](https://playground.mn-core.com/)
- [PFCP User Guide](https://docs.pfcomputing.com/en/)
- [MN-Core 2 Devkit / MN-Server 2 installation & operation manual](https://projects.preferred.jp/mn-core/assets/MN-Core2-Devkit-MN-Server-2-installation-operation-manual.pdf)
