# SynapseAI SDK & TPC Investigation

*chip: intel-gaudi*
*as_of: 2026-08-08 (baseline investigation 2026-04-05; see dated update section at end)*
*sources: docs.habana.ai v1.24.0, github.com/HabanaAI/tpc_llvm, github.com/HabanaAI/gaudi-pytorch-bridge*

> **Note (2026-08-08):** this file is the software-stack investigation for intel-gaudi. See the
> "Update — software stack (2026-08-08)" section at the end for driver-upstreaming status, repository
> archival scope, the v1.24.0 documentation bump, and the Crescent Island (oneAPI/SYCL + Level Zero)
> successor stack.

---

## Overview

SynapseAI is Intel Gaudi's end-to-end software stack, serving as the bridge between deep learning frameworks (primarily PyTorch) and the heterogeneous Gaudi hardware. It consists of four tightly integrated layers: the graph compiler (which performs ahead-of-time optimization and binary code generation for the MME and TPC engines), the runtime (which manages device execution, memory, and host-device communication), the TPC kernel library (1400+ pre-built VLIW SIMD kernels covering all non-GEMM operators), and the TPC-C SDK (which enables custom kernel development via an LLVM-based C compiler extended with Gaudi-specific intrinsics). The PyTorch integration is implemented as a loadable device backend—`gaudi-pytorch-bridge`—which registers a `hpu` device with PyTorch and intercepts operator dispatch to route tensors through the SynapseAI lowering layer.

---

## Architecture

### Module Structure

```
habana_frameworks/
  torch/
    core/                  # HPU device registration, tensor storage
    hpex/                  # Extended ops: optimizers, mixed-precision, fused kernels
    distributed/           # HCCL distributed backend wrapper

gaudi-pytorch-bridge/
  pytorch_helpers/         # Op registration (ATEN dispatch keys)
  habana_lazy/             # Lazy IR accumulation and graph flushing
  habana_eager/            # Eager op dispatch
  synapse_shim/            # SynapseAI C API bindings

SynapseAI (closed-source runtime):
  graph_compiler/          # TF/PT graph → SynapseAI IR → MME/TPC binaries
  tpc_kernel_lib/          # 1400+ precompiled TPC kernel binaries
  runtime/                 # Device memory manager, stream scheduler, DMA engine
  synapse_api.h            # Public C API surface
```

### Key Abstractions

| Abstraction | Description |
|---|---|
| `SynapseAI Graph` | Directed acyclic graph of nodes; each node maps to an MME op or TPC kernel; compiled AOT to device binary |
| `TPC Kernel` | A VLIW SIMD program targeting a single TPC processor (256-byte vector lane, VLIW with 4 slots: Vector/Scalar/Load/Store); compiled from TPC-C via LLVM |
| `Habana Bridge (gaudi-pytorch-bridge)` | PyTorch device backend that accumulates ops into a SynapseAI graph (lazy mode) or dispatches individually (eager mode) |
| `Lowering Module` | Translates PyTorch ATen ops to SynapseAI graph nodes; handles type promotion, layout conversion (NCHW→NHWC), and op fusion hints |
| `habana_frameworks.torch` | Python-level API surface: `mark_step()` triggers graph flush; `hpex` exposes Gaudi-optimized fused ops and FP8 transformer engine |

### Dependency Graph

```
User Python Script
       |
habana_frameworks.torch (import)
       |
gaudi-pytorch-bridge (C++ device backend, registered via torch plugin)
       ├── eager path: op → synapse_shim → SynapseAI runtime (immediate execution)
       └── lazy path:  op → IR node accumulation → mark_step() →
                           SynapseAI graph compiler → optimized binary → runtime
                                          |
                              TPC-C kernel library (1400+ kernels, LLVM compiled)
                              MME scheduling pass
```

---

## Data Flow

### PyTorch Lazy Mode: `model(input)` through to hardware execution

1. **Op dispatch**: `torch.nn.Linear(input)` triggers ATen dispatch; bridge intercepts at `HPUBackend` key.
2. **IR accumulation**: `habana_lazy` module creates a lazy IR node (`HloInstruction`) for the matmul. Tensor remains a stub; no computation occurs.
3. **`mark_step()` trigger**: Called explicitly or automatically (by optimizer step / loss.backward() boundary). Flushes the accumulated IR graph to the SynapseAI graph compiler.
4. **Graph compilation**: The SynapseAI compiler performs:
   - Operator fusion (e.g., Linear + ReLU fused into a single TPC kernel call)
   - Data layout transformation (to hardware-preferred formats)
   - GEMM ops dispatched to MME engine nodes
   - Elementwise/non-GEMM ops dispatched to TPC nodes
   - Memory allocation planning (SRAM vs HBM)
5. **Binary generation**: MME configuration words and TPC kernel binaries (selected from the precompiled TPC kernel library) are emitted.
6. **Runtime execution**: SynapseAI runtime schedules MME ops and TPC ops concurrently on hardware; DMA engines move tensors HBM↔SRAM; result tensors returned to host when accessed.

### TPC Custom Kernel Development Flow

1. Write kernel in **TPC-C** (C99 + Gaudi intrinsics, e.g., `v_f32_add_b`, `v_convert_bf16_to_f32`)
2. Compile with `tpc-clang` (LLVM fork: `github.com/HabanaAI/tpc_llvm`); verify with TPC simulator
3. Register kernel via SynapseAI Graph API (`synNodeCreateWithId`)
4. Graph compiler integrates the custom kernel binary into the compiled graph at the appropriate TPC node

---

## Hardware Interface

- **MME interface**: SynapseAI graph compiler emits GEMM configuration descriptors consumed by the Matrix Multiplication Engine; tile sizes determined by compiler.
- **TPC interface**: Each TPC op is a compiled VLIW binary assigned to one or more TPC processors in the 64-TPC cluster (Gaudi 3); the VLIW instruction encodes concurrent Vector, Scalar, Load, and Store operations.
- **SRAM**: Graph compiler manages placement of intermediate activations in 96 MB shared SRAM (12.8 TB/s); both MME and TPC access this shared pool.
- **DMA**: Async DMA engines move tensors between HBM2e (128 GB, 3.7 TB/s) and SRAM; double-buffering enables overlap of compute and data movement.
- **Host↔Device**: PCIe Gen5 (x16, ~128 GB/s bidirectional) used for tensor transfers and command submission from host CPU.

---

## Key Findings

1. **Lazy mode is the primary performance path**: Eager mode is convenient for debugging but gives the compiler no fusion opportunity. `mark_step()` is the critical boundary that triggers graph-level optimization.
2. **TPC is a programmable fallback for all non-GEMM ops**: Every operator that cannot be expressed as matrix multiplication (activations, norms, elementwise) maps to a TPC kernel from the 1400+ precompiled library—or a user-written TPC-C kernel.
3. **LLVM toolchain is open-source**: `github.com/HabanaAI/tpc_llvm` is publicly available, enabling custom kernel development without NDA toolchain access.
4. **FP8 support in Transformer Engine**: `habana_frameworks.torch.hpex` includes a `ModuleExtension` for FP8 linear layers, directly targeting the Gaudi 3 TPC's E4M3/E5M2 FP8 units.
5. **habana_frameworks.torch.core must be imported before any HPU op**: The import side-effect registers the HPU device backend with PyTorch's dispatcher; missing this import causes silent CPU fallback.

---

## Relation to Hardware Architecture

SynapseAI is the only software path to Gaudi hardware. There is no CUDA-equivalent low-level API exposed to users; all access flows through the SynapseAI graph API or the PyTorch/TF bridges. This means the graph compiler's decisions (which ops go to MME vs TPC, how SRAM is partitioned) are not user-visible, making profiler tools (Habana Profiler, hl-smi) essential for performance debugging. The TPC-C SDK provides an escape hatch for custom kernels but requires deep familiarity with the VLIW ISA.

---

## Update — software stack (2026-08-08)

*Window covered: 2026-04-05 → 2026-08-08. Several items below are pre-baseline public facts that were missing from this repo; they are recorded here as gap closures, and dated accordingly.*

### 1. Documentation version bump

`docs.habana.ai` now serves **Intel Gaudi software v1.24.0** (this file and the chip Resources table previously cited v1.23.0). Software releases are still being published in 2026 — Gaudi is not a dead product line. Source: https://docs.habana.ai/en/latest/

### 2. Gaudi 3 kernel driver is NOT in mainline Linux (correction)

Previously the repo said the habanalabs driver was "open-source, upstreamed in Linux kernel" without qualification. That holds for **Goya, Gaudi 1 and Gaudi 2 only**.

- Verified against mainline on 2026-08-08: `drivers/accel/habanalabs/` contains `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/` — **there is no `gaudi3/` directory**.
- Intel posted Gaudi 3 driver code for upstreaming in late Nov 2025 and sent the pull request "accel/habanalabs: Gaudi3 support and updates for v6.19" on dri-devel (Dec 2025). It was **rejected** on code-quality grounds (reverts and build artifacts inside a ~300k-line series) and now targets a later cycle.
- Practical consequence: **Gaudi 3 deployment requires the out-of-tree habanalabs driver.**

Sources: https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs · https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html · https://www.phoronix.com/news/Intel-SynapseAI-Stops

### 3. Archived repositories — scope matters

`github.com/HabanaAI/SynapseAI_Core`, the open **reference implementation** of the SynapseAI API, was archived **2025-02-03** with the notice: *"This project will no longer be maintained by Intel. Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates."* Also archived: `Gaudi-tutorials` (2025-09-18) and `Model-References` (2026-01-08).

This is **not** "the Gaudi user-space stack is dead". Verified against the HabanaAI org (GitHub REST API and org repo listing):

| Repository | Archived? | Last push |
|---|---|---|
| `SynapseAI_Core` | **yes** (2025-02-03) | — |
| `Gaudi-tutorials` | **yes** (2025-09-18) | — |
| `Model-References` | **yes** (2026-01-08) | — |
| `gaudi-pytorch-bridge` | no | 2026-07-13 |
| `vllm-fork` | no | 2026-07-27 |
| `optimum-habana-fork` | no | 2026-07-17 |
| `gaudi-*` K8s operator / device-plugin / exporter suite | no | 2026-07-23 |
| `Megatron-LM` | no | 2026-07-10 |
| `Setup_and_Install` | no | 2026-07-10 |

The accurate statement: **the archived reference implementation removes the accompanying open user-space that Linux's `accel` subsystem expects from a driver submission — a material obstacle to Gaudi 3 driver upstreaming — while the production PyTorch/vLLM enablement path continues to ship.**

Sources: https://github.com/orgs/HabanaAI/repositories?type=all&sort=updated · https://www.phoronix.com/news/Intel-SynapseAI-Stops

### 4. Successor stack — Crescent Island does not inherit SynapseAI

The prior investigation flagged SynapseAI/HCCL continuity across Gaudi → Crescent Island as an open question. Evidence now says the SynapseAI graph-compiler stack (SynapseAI + TPC-C + tpc_llvm + HCCL) is a **terminal branch**. Crescent Island rides Intel's unified Xe path — **oneAPI/SYCL over the Level Zero Compute Runtime** — with enablement landed in public source trees ahead of the hardware:

| Component | Version / date | Crescent Island status |
|---|---|---|
| Intel Compute Runtime (Level Zero / OpenCL) | 26.01.36711.4 — 2026-01-14 | early Crescent Island support added |
| Intel Graphics Compiler (IGC) | v2.27.10 — Jan 2026 | initial Crescent Island support added |
| Intel Compute Runtime | 2026-04-20 | Crescent Island support **promoted** |

Intel stated at OCP (Oct 2025) that this unified stack is being developed and validated on Arc Pro B-series GPUs and will extend to Xe3P. Implication for this survey: none of Gaudi's compiler/runtime/collective tooling investment (graph compiler heuristics, TPC-C kernels, HCCL) transfers to Intel's next datacenter AI product.

Sources: https://www.phoronix.com/news/Intel-CR-26.01.36711.4 · https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler

### 5. Not disclosed

- Whether Crescent Island exposes any HCCL-compatible collective API, or what its multi-card collective path is.
- Whether Jaguar Shores inherits SynapseAI, the Xe/oneAPI stack, or something else — **no software stack details disclosed**.
- Any Gaudi 3 MLPerf v6.x software-optimized result (Gaudi was absent from MLPerf Inference v6.0 on 2026-04-01 and Intel was absent entirely from MLPerf Training v6.0 on 2026-06-16).
