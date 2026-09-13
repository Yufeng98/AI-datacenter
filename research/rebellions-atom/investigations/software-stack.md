# Rebellions ATOM — Software Stack Investigation

*chip: rebellions-atom*
*as_of: 2026-08-08*
*sources: RBLN SDK docs, RBLN compiler blog, optimum-rbln GitHub, rbln-model-zoo GitHub, vLLM RFC; (2026-08-08 update) docs.rbln.ai release notes, Rebellions newsroom 2026-06-30, TheElec 2026-06-30*

---

## Overview

Rebellions' software stack is the **RBLN SDK**, a compile-once, deploy-anywhere toolchain centered on an AI compiler that transforms ML framework models into static executables for ATOM/REBEL NPUs. The programming model is fundamentally different from CUDA: there is no kernel authoring, no runtime JIT, and no user-visible ISA. The user provides a trained model; the RBLN compiler produces a binary that runs deterministically on hardware.

---

## 1. Framework Integration

The RBLN SDK ingests models from three major ML frameworks:

| Framework | Integration Path | Notes |
|---|---|---|
| PyTorch | `torch.export` / `rebel.compile()` | Primary path; traces computation graph |
| TensorFlow | SavedModel / TF2 concrete function | Graph capture |
| HuggingFace Transformers | `optimum-rbln` plugin | `RBLNAutoModelFor*` drop-in replacements |
| HuggingFace Diffusers | `optimum-rbln` plugin | Stable Diffusion, SDXL |

The `optimum-rbln` library provides `RBLNAutoModelForCausalLM`, `RBLNAutoModelForSeq2SeqLM`, `RBLNStableDiffusionPipeline`, etc., which are drop-in replacements for their HuggingFace counterparts, handling model export and compilation transparently.

---

## 2. Compiler / IR

### RBLN Compiler Architecture

The RBLN Compiler is a two-phase static compiler:

```
Model (PyTorch/TF/HF)
        |
        v
  [Frontend Compiler]
   - Framework graph capture
   - Operator lowering to RBLN IR
   - Graph-level optimization (fusion, constant folding)
        |
        v
    [RBLN IR]
   - Pure tensor operations
   - No memory operations exposed at this level
        |
        v
  [Dependency Analysis]
   - Analyzes DRAM/SRAM spaces per compute and memory op
   - Finds and preserves inter-operation dependencies
        |
        v
  [Backend Compiler]
   - Memory allocation (DRAM, SRAM)
   - Scheduling (compute + memory overlap)
   - Tiling decisions for CGRA/Neural Core
   - RISC ISA code generation via Compute Library
        |
        v
  [RBLN Binary (.rbln)]
   - Static executable for target ATOM/REBEL NPU
```

### Compute Library

The RBLN Compute Library generates **RISC ISA programs** for Neural Engines based on:
- Operation type (GEMM, convolution, normalization, activation, etc.)
- Tensor shapes
- Target hardware (ATOM vs REBEL)

For each operation, the Compute Library determines:
- **Tiling**: How the computation is tiled to fit on-chip SRAM
- **Required SRAM size**: Static reservation
- **Estimated compute time**: Used by the scheduler for pipeline optimization

The Compute Library supports:
- Hundreds of GEMM variants (matrix multiplication)
- Normalization operations (LayerNorm, RMSNorm)
- Nonlinear activations (GELU, SiLU, ReLU, Softmax)
- Convolutions (CNN workloads)
- Attention patterns (including Flash Attention-style tiled attention)

### Key Compiler Properties

- **Global optimization**: The compiler analyzes the entire computational graph at once, enabling cross-operation scheduling, memory reuse, and parallel execution — not possible with per-kernel compilation
- **Static compilation**: No runtime JIT; the binary is fully determined at compile time
- **Compile-once, deploy anywhere**: A compiled `.rbln` binary runs on any ATOM/REBEL device without recompilation
- **Quantization support**: INT8, FP8 quantization handled at compile time

---

## 3. Op / Kernel Library

There is no user-visible kernel library in the CUDA sense. The Compute Library is an internal compiler component that generates neural engine programs. Users do not write kernels.

For inference serving:
- **RBLN Model Zoo**: Pre-compiled, ready-to-run model binaries for common architectures (ResNet, BERT, T5, LLaMA, SDXL, etc.)
- **Optimum-RBLN**: Pre-integrated HuggingFace model compilations

---

## 4. Runtime

### RBLN Runtime (`rbln` Python package)

- Loads `.rbln` compiled binaries onto ATOM/REBEL devices
- Manages device memory allocation (host ↔ device transfers)
- Handles multi-instance scheduling (up to 16 hardware-isolated instances on ATOM)
- Provides `rbln.Runtime` Python API for inference execution
- Supports NVIDIA Triton Inference Server integration (gRPC/REST serving)

### Serving Integrations

| System | Integration | Notes |
|---|---|---|
| NVIDIA Triton Inference Server | Native backend plugin | ResNet50 tutorial in official docs |
| vLLM | Official plugin | RFC #7247; enables LLM token streaming |
| HuggingFace pipelines | Via optimum-rbln | TransformersCompatibility |

---

## 5. Driver / Firmware

Rebellions does not publicly document its kernel driver in detail. Known characteristics:
- Linux kernel module for PCIe device management and BAR mapping
- Device firmware for Neural Engine command processing
- Multi-instance isolation implemented in hardware + driver
- PCIe Gen5 DMA engine for host-device data transfers

---

## 6. Programming Model Summary

The RBLN programming model is a **framework-to-binary compilation model**:

```
User writes:         PyTorch / HuggingFace model (standard Python)
                           |
SDK compiles:        rebel.compile(model, input_shapes) → .rbln binary
                           |
Runtime executes:    rbln.Runtime.load(".rbln") → inference()
```

There is no GPU-style kernel launch, no stream management, no shared memory management, and no ISA exposure to users. This is similar to AWS Neuron's compile-once model but with a different hardware substrate (CGRA/Neural Cores vs systolic arrays).

---

## 7. Quantization and Precision

| Format | ATOM | REBEL-Quad |
|---|---|---|
| FP16 | Yes (32 TFLOPS) | Yes |
| INT8 | Yes (128 TOPS) | Yes |
| FP8 | REBEL-Quad only | Yes (2,048 TFLOPS) |
| INT4 / FP4 | Not confirmed | Not confirmed |

---

## 8. Multi-Model / Multi-Instance

- **ATOM Multi-Instance**: 16 hardware-isolated instances per card; enables concurrent multi-tenant inference
- Each instance has dedicated Neural Engine resources and SRAM allocation
- The RBLN runtime arbitrates instance assignments

---

## Investigation Update — 2026-08-08 (window 2026-04-06 → 2026-08-08)

*Sections 1–8 above describe the stack as of 2026-04-05 and remain accurate. This section records what changed. The single most consequential item is a new stack tier acquired wholesale.*

### 9. New tier — SqueezeBits acquisition (2026-06-30)

Rebellions acquired **SqueezeBits**, a Korean AI model-compression / inference-optimization firm (founded March 2022), through an **all-stock swap** making it a 100%-owned subsidiary; existing SqueezeBits investors received Rebellions shares. Cash terms were not disclosed. TheElec reports prior-year figures of KRW 2.646B revenue, KRW 897M operating profit, KRW 5.3B total assets. The two firms had co-developed since 2024 and jointly contributed to vLLM.

SqueezeBits brings three products:

| Product | Role | Where it sits in this stack |
|---|---|---|
| **OwLite** | Quantization / model compression | Above the RBLN frontend compiler — a *pre-compilation* model-transformation tier that did not previously exist in Rebellions' first-party toolchain |
| **Fits on Chips** | LLM benchmarking / LLMOps toolkit | Cross-cutting: measurement and operations, no analogue in the prior stack |
| **Yetter** | Inference engine | Serving tier — relationship to the existing vLLM plugin and Triton backend is undisclosed |

**Why this matters architecturally.** Section 2 above describes a stack whose only optimization surface is the compiler: quantization is "handled at compile time," and there is no user-facing model-transformation stage. OwLite and Fits on Chips introduce an explicit model-optimization tier *above* `rebel.compile()`. That is a structural change to the stack shape, not a feature addition.

**What is not disclosed:** whether or how OwLite/Fits on Chips/Yetter integrate with the RBLN SDK, whether they ship in the monthly SDK release train, whether Yetter supersedes or complements the vLLM plugin, and whether any of them will support non-Rebellions targets going forward. Nothing in the SDK release notes through 2026-07-31 references any SqueezeBits product.

### 10. RBLN SDK release cadence and breaking changes

The SDK moved to monthly calendar releases. Four releases fell inside the window. **Two carry breaking changes.** Version strings below are taken from the release notes and the `.post1` suffixes are load-bearing.

| Release | Compiler | Driver | Contents |
|---|---|---|---|
| 2026-04-30 | v0.10.3 | v3.0.0 | Parallel compilation; RSD support for vision transformers and Qwen3-VL; adds Qwen3-VL-32B, StableFast3D, TripoSR. **BREAKING — RBLN NPU Operator v0.3.x → v0.4.0**: Helm values restructured (`spec.daemonsets` → `spec.podDefaults`), auto device detection replaces the static ConfigMap, generic resource mode `rebellions.ai/npu` becomes the default |
| 2026-05-29 | v0.10.4.post1 | v3.2.2 | New `rbln-vs` hardware diagnostic tool; `rbln_daemon` renamed `rbln-smd`; `rbln-smi` gains `mknod` for container use; automatic vLLM compilation removes the separate precompile step; `max_seq_lens` renamed `max_seq_len` for VLMs |
| 2026-06-26 | v0.11.0.post1 | v3.2.2 | **BREAKING — first release supporting Transformers v5**: the compiled model format changed; models compiled with earlier SDKs are incompatible and must be recompiled. `tensor_parallel_size` renamed `num_devices` across compiler APIs. Adds Gemma4 and EXAONE-4.5 |
| 2026-07-31 | v0.11.1.post1 | v3.2.2 | Adds `index_copy` / `index_add` / `grouped_conv1d`; **fixes numerical mismatches caused by an inadvertently bundled mimalloc allocator in v0.11.1** — plain v0.11.1 is a known-bad build; `RBLN_VISIBLE_DEVICES` alias for `RBLN_DEVICES`; new `torch.rbln.explain()` (diagnoses CPU fallback) and `torch.rbln.synchronize()`; compile-time reduction via op fusion |

Immediately preceding the window, for cadence context: 2026-01-30 (compiler v0.10.0, driver v3.0.0, modular driver architecture, MoE model support), 2026-02-27 (vLLM v0.13.0 support), 2026-03-27 (public PyTorch-RBLN release, initial Dynamic Resource Allocation driver support).

**Observations relevant to the survey's programming-model chapter:**

1. **The compile-once artifact is not stable across SDK majors.** Section 6 above frames "compile once, deploy anywhere" as the defining property of this stack. The 2026-06-26 Transformers-v5 change invalidates every previously compiled `.rbln` binary. The portability guarantee is across *devices*, not across *toolchain versions* — worth stating precisely.
2. **Kubernetes is now a first-class deployment surface** with its own breaking-change surface (NPU Operator v0.4.0). This layer was absent from the 2026-04-05 layer table.
3. **New diagnosability tooling.** `rbln-vs` (hardware diagnostics) and `torch.rbln.explain()` (CPU-fallback diagnosis) are the first user-facing tools for answering "why is this slow / why did this not run on the NPU" — a recurring gap in static-compilation stacks.
4. **No new hardware family entered the SDK during 2026.** Release notes still cover only the ATOM and REBEL series — corroborating the hardware-side finding of no new silicon.

### Not confirmed / open

- Integration path between the SqueezeBits products and the RBLN SDK: **not disclosed**.
- Whether Yetter replaces, wraps, or coexists with the vLLM plugin: **not disclosed**.
- Multi-instance/partitioning behaviour on REBEL-Quad: still **not publicly specified** (unchanged from the baseline).

---

## Sources

- [RBLN SDK User Guide](https://docs.rbln.ai/latest/index.html)
- [Understanding RBLN Compiler](https://rebellions.ai/understanding-rbln-compiler/)
- [optimum-rbln GitHub](https://github.com/rebellions-sw/optimum-rbln)
- [rbln-model-zoo GitHub](https://github.com/rebellions-sw/rbln-model-zoo)
- [vLLM RFC: RBLN NPU support](https://github.com/vllm-project/vllm/issues/7247)
- [RBLN SDK Release Note (PDF)](https://rebellions.ai/wp-content/uploads/2025/07/250704_SDK_Release_Note.pdf)
- [Rebellions Developers Page](https://rebellions.ai/developers/)

### Added 2026-08-08

- [RBLN SDK Release Notes](https://docs.rbln.ai/latest/supports/release_note.html) — authoritative for compiler/driver version strings and breaking changes
- [Rebellions acquires SqueezeBits — 2026-06-30](https://rebellions.ai/newsroom/rebellions_squeezebits_acquisition_260630/)
- [TheElec: SqueezeBits all-stock acquisition, 100% subsidiary — 2026-06-30](https://www.thelec.net/news/articleView.html?idxno=11826)
- [Storage Newsletter: Rebellions/SqueezeBits coverage — 2026-07-10](https://www.storagenewsletter.com/2026/07/10/)
