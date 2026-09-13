# Esperanto ET-SoC-1 Software Stack Investigation

*as_of: 2026-08-08 (baseline sections dated 2026-04-05; see appended investigation of 2026-08-08 at the end)*
*chip: esperanto*
*sources: Esperanto Technology page, SDK press release (May 2023), Esperanto blogs, ONNX/Glow integration docs*

---

## Overview

The Esperanto software stack is built on a key architectural insight that mirrors the hardware philosophy: **leverage the open RISC-V software ecosystem rather than building a proprietary one from scratch**. Because ET-SoC-1's cores run standard RISC-V, tools like GCC, LLVM, Linux, gdb, and valgrind work natively. Esperanto extended this with ML-specific layers on top, reducing the software bring-up cost compared to chips with entirely custom ISAs.

The stack has two independent front-ends:
1. **Primo AI/ML SDK** — inference-oriented; ingests ONNX/PyTorch models and compiles to RISC-V
2. **General Purpose HPC SDK** — C/C++ direct programming of 1,000+ RISC-V cores for non-AI workloads

---

## 1. Framework Integration

### PyTorch
- PyTorch models are exported to ONNX using `torch.onnx.export()` or `torch.export`
- ONNX is the canonical input to the Esperanto ML compiler
- No native PyTorch eager backend exists — all execution is ahead-of-time via ONNX

### TensorFlow / Keras
- Supported via TensorFlow → ONNX conversion path (tf2onnx)
- No native TF integration

### ONNX (First-class)
- ONNX is the primary and preferred model input format
- Esperanto built a custom **ONNXRuntime Execution Provider (EP)** for ET-SoC-1
- Users interact via standard ONNXRuntime APIs (Python, C, C++, Java) — zero hardware-specific code at the inference call site

```python
import onnxruntime as ort
# ET-SoC-1 EP loads and dispatches to hardware transparently
sess = ort.InferenceSession("model.onnx", providers=["ETSoC1ExecutionProvider"])
output = sess.run(None, {"input": input_data})
```

### Whisper / Small Language Models (documented use cases)
- Whisper ASR models adapted to ET-SoC-1 via ONNX export with custom operator handling
- Small language models (SLMs) exported to ONNX with **KV-cache support** for autoregressive generation
- Demonstrates iterative attention with cached key/value states — ET-SoC-1's 160 MB on-chip SRAM is large enough for KV-cache of SLMs at moderate context lengths

---

## 2. Compiler / IR

### Glow Compiler (Meta Open Source)
- **Core ML compiler**: Meta's [Glow](https://github.com/pytorch/glow) open-source graph compiler
- Glow accepts ONNX (or PyTorch via Glow's loader) and compiles to optimized RISC-V code
- Esperanto uses Glow as the graph-level optimization and lowering pass, with custom ET-SoC-1 backends for:
  - Lowering standard ops (Conv, GEMM, Attention, Softmax) to ET-Minion tensor instructions
  - Partitioning the computation graph across 1,088 cores
  - Scheduling data movement between on-chip SRAM tiles and LPDDR4x

### LLVM / RISC-V Backend
- Glow lowers to LLVM IR, which targets the RISC-V backend
- LLVM's RISC-V codegen handles the standard integer operations; Esperanto's custom vector/tensor intrinsics extend the LLVM backend for the ET-Minion vector/tensor ISA extensions
- This is the layer where the 32K-op tensor instructions and vector transcendental operations are emitted

### Compiler Output
- A **package of compiled RISC-V ELF binaries** — one per ET-Minion tile or partition
- Includes the weight data interleaved with code for SRAM locality
- The runtime loads these per-core binaries, initializes SRAM tile data, and launches all 1,088 ET-Minions simultaneously

### Primo AI/ML SDK
- Umbrella SDK wrapping the Glow + LLVM pipeline
- Provides Python APIs for model compilation, profiling, and deployment
- Preview software; not released for production as of 2025

---

## 3. Op Library

There is no separate cuDNN/MIOpen-equivalent op library for ET-SoC-1. Instead:

- Standard RISC-V vector intrinsics handle element-wise ops
- The Glow compiler lowers GEMM and convolution to ET-Minion tensor instructions
- Activation functions (GeLU, SiLU, softmax, sigmoid, tanh) are accelerated by the **vector transcendental unit** — no software lookup table or polynomial approximation needed
- This is essentially the same model as Groq (compiler owns all op fusion) rather than the GPU model (cuDNN/cuBLAS as separately versioned libraries)

---

## 4. Kernel Library

No user-facing kernel library (no CUTLASS equivalent). The RISC-V C/C++ programming model means:

- **For ML**: Glow compiler generates all kernel code; users do not write custom RISC-V kernels
- **For HPC** (General Purpose SDK): Users write standard C/C++ with OpenMP-style parallel pragmas or direct RISC-V thread spawning; the compiler handles vectorization with RISC-V vector extensions

The HPC SDK is the closest analog to a "kernel library" — it provides:
- RISC-V vector intrinsics headers
- OpenMP thread management targeting ET-Minion cores
- Math library optimized for the vector transcendental unit

---

## 5. Runtime

### ONNXRuntime Custom EP
- The ET-SoC-1 execution provider for ONNXRuntime intercepts operator dispatch
- Compiled graph partitions are loaded onto chip; ONNXRuntime manages I/O

### Native Runtime (Primo stack)
- Lightweight RISC-V runtime managing:
  - ET-Minion core initialization and barrier synchronization
  - SRAM tile allocation and DMA from LPDDR4x
  - PCIe DMA for host-to-chip input data movement
  - Result collection and PCIe DMA back to host
- No dynamic scheduling at runtime — the compiled partition assignment is static

### Linux on ET-Maxion
- The 4 ET-Maxion cores run a full Linux OS
- The chip is therefore a complete compute node, not just an accelerator — it can host its own processes, memory-map DRAM, and manage the ET-Minion pool
- This enables ET-SoC-1 to operate in both PCIe accelerator mode (host drives inference) and standalone server mode (chip runs its own inference service)

---

## 6. Driver / Firmware

- **Linux kernel driver**: Standard PCIe kernel module for BAR mapping, DMA, and interrupt handling
- **ET-Maxion firmware**: The ET-Maxion runs Linux; there is no separate microcontroller firmware layer (unlike NVIDIA's GSP). The OS manages all resource allocation
- **ET-Minion bootloader**: The ET-Maxion OS loads compiled RISC-V binaries into ET-Minion SRAM tiles and releases reset — simple loader, not a runtime firmware

---

## 7. Communication

- **Single-chip**: ET-Minion cores communicate via shared on-chip SRAM (standard multiprocessor shared memory)
- **Multi-chip (Glacier Point v2)**: 6 chips on one PCIe card; host x86 server coordinates via PCIe; no dedicated chip-to-chip fabric described in public sources
- **No collective library**: No NCCL/RCCL equivalent; multi-chip parallelism is handled at the Glow compiler layer by partitioning the model graph across chips

---

## 8. Assembler / ISA

| Aspect | Detail |
|--------|--------|
| Base ISA | RISC-V RV64GCV — fully open standard; standard RISC-V assembler works |
| Extensions | Esperanto custom vector/tensor ISA extensions (not publicly specified at instruction encoding level) |
| Toolchain | GCC + LLVM RISC-V backends; Esperanto patches for custom extensions |
| Programmer access | RISC-V assembly is user-accessible (unlike GPU PTX being the lowest public layer) |
| No virtual ISA needed | RISC-V IS the virtual ISA — standard across cores; no PTX equivalent required |

---

## 9. General Purpose SDK (May 2023)

Announced May 2023 as an extension beyond AI inference:

- Enables standard C/C++ development targeting all 1,088 ET-Minion cores
- OpenMP parallelism model; RISC-V vector intrinsics for SIMD/tensor ops
- Use cases: Monte Carlo simulation, physics simulation, genomics, HPC kernels
- Represents Esperanto's pivot toward broader HPC market beyond ML inference

---

## 10. Post-Acquisition Software Status

> ⚠️ **Superseded 2026-08-08.** This section as originally written contained two date errors and overstated what was open-sourced. It is preserved below for history; see section 11 for the corrected account.

*Original text (as_of 2026-04-05):* After Ainekko acquired Esperanto's IP in October 2025: hardware RTL open-sourced under Apache 2.0; software toolchain (Glow-based compiler, RISC-V extensions, SDK) also transferred to Ainekko; Ainekko's AI Foundry platform makes these available for community development; Ainekko plans to use Esperanto's RISC-V cores (initially 8 cores) in a new tapeout with MRAM from Veevx startup; the existing ET-SoC-1 software stack provides the starting point for Ainekko's open-hardware edge AI platform.

*Corrections:* the acquisition was announced **2025-11-19** (not October — 2025-10-22 was the separate "AI Foundry" launch); the Veevx merger was **2026-01-29** (not November 2025); the code became public on **2026-04-20**; the "8-core tapeout on a TSMC shuttle wafer" plan is superseded by a **16nm tapeout in progress**; and what is published is a **re-translation**, not ET-SoC-1's production RTL.

---

## 11. Appended Investigation — 2026-08-08: Source Release, License, and Governance

*Primary sources: Ainekko GlobeNewswire releases 2025-11-19, 2026-01-29, 2026-06-02; github.com/openhwgroup/core-et and its raw LICENSE; GitHub API records for openhwgroup repos, core-et-erbium, and the user `ainekko`; openhwfoundation.org.*

### 11.1 Where the code actually lives

| Item | Detail |
|---|---|
| Repositories | `github.com/openhwgroup/core-et`, `github.com/openhwgroup/core-et-erbium` |
| Created | 2026-04-20 (both) |
| Last push observed | `core-et` 2026-05-07; `core-et-erbium` 2026-06-23 (~4.9 MB, 3 stars) |
| Host | **OpenHW Foundation** GitHub org — *not* Ainekko's own GitHub |
| Do not cite | `github.com/ainekko` — GitHub API confirms an unrelated personal **User** account (created 2023-08-15, 4 public repos, no company or website set) |

### 11.2 License — two answers, both recorded

- The **checked-in file** `core-et/LICENSE` is **verbatim Apache License 2.0**.
- The **2026-06-02 press release** instead names **"Solderpad Hardware License v2.1 ... building on the well-known Apache 2.0 license"**.
- The **2025-11-19 acquisition release names no license at all** and says only that Ainekko is "exploring foundation-based governance."

This discrepancy is unresolved and is recorded as such. The practical consequence for this survey: the repo's pre-existing "Apache 2.0" claim is **substantially correct and should not be flagged as unverified** — only its November-2025 date was wrong.

### 11.3 Governance

**"CORE-ET Silicon Platform (ETSP)"** was accepted as an **OpenHW Foundation** project on **2026-06-02**. OpenHW Foundation (formerly OpenHW Group; openhwgroup.org 302-redirects to openhwfoundation.org) is "an Eclipse Foundation global initiative", so descriptions of the release as being "under the Eclipse Foundation" are loose but not false — the precise home is the OpenHW Foundation. Ainekko is listed as an OpenHW member.

This is the first foundation-governed home for any part of the Esperanto lineage; prior to this, `github.com/esperantotech` was a vendor-controlled org.

### 11.4 What the release is *for* — an unusual development model

`core-et` self-describes as "an Ainekko project for collecting hardware IP in a form that can be **translated, verified, documented, and integrated by agentic hardware-development workflows**," working from "CORE-ET modules into clean SystemVerilog," with the original kept on a separate branch.

Two consequences for the software-stack layer table:

1. **It is not the ET-SoC-1 production RTL.** Esperanto lineage is visible only indirectly (e.g. `docs/minion_vpu_standalone_plan.md` references the "real-VPU Minion path"). Neither the 2025-11-19 nor the 2026-06-02 press release names ET-SoC-1.
2. **The stated consumer of the source is a toolchain, not a human RTL engineer.** The release is framed as a substrate for LLM-driven ("agentic") hardware development — a materially different rationale from the usual open-hardware motivation of community review and reuse. This is the one genuinely novel software-side item in this update.

### 11.5 What is *not* disclosed on the software side

No SDK, compiler, runtime, driver, or ISA documentation has been published or announced for the 16nm CORE-ET Silicon Platform. In particular:

- Whether the **Glow-based ML compiler**, the **LLVM RISC-V backend extensions**, the **ONNXRuntime Execution Provider**, the **Primo AI/ML SDK**, or the **General Purpose HPC SDK** carry forward to ETSP is **not disclosed**.
- No replacement stack, no framework integration, and no programming model for ETSP has been described.
- The `esperantotech` GitHub org and the Esperanto blog/documentation pages remain reachable only as archive material (esperanto.ai newest news item: 2025-04-09).

Accordingly the software layer table's ETSP rows read "not disclosed" rather than inheriting the ET-SoC-1 stack.

### 11.6 Sources added this pass

- https://www.globenewswire.com/news-release/2025/11/19/3191016/0/en/Ainekko-Acquires-Esperanto-Technologies-Intellectual-Property-to-Power-Open-Source-Edge-AI-Platform.html
- https://globenewswire.com/news-release/2026/01/29/3228632/0/en/ainekko-merges-with-veevx-expands-open-silicon-platform-with-breakthrough-memory-and-embedded-ai-capabilities.html
- https://globenewswire.com/news-release/2026/06/02/3305225/0/en/ainekko-s-edge-ai-silicon-platform-is-now-an-openhw-foundation-open-source-project.html
- https://github.com/openhwgroup/core-et
- https://raw.githubusercontent.com/openhwgroup/core-et/main/LICENSE
- https://api.github.com/repos/openhwgroup/core-et-erbium
- https://api.github.com/users/ainekko
- https://openhwfoundation.org/
- https://nekko.ai/ (vendor site)

---

## Resources

- [Esperanto Technology Page](https://www.esperanto.ai/technology/)
- [Esperanto Products Page](https://www.esperanto.ai/products/)
- [SDK Launch Press Release (May 2023)](https://www.esperanto.ai/News/esperanto-technologies-launches-general-purpose-sdk/)
- [Esperanto Blog: Compiling SLMs with ONNXRuntime](https://www.esperanto.ai/blog/compiling-and-executing-small-language-models-with-onnxruntime-2/)
- [Esperanto Blog: Adapting Whisper to ET-SoC-1](https://www.esperanto.ai/blog/adapting-whisper-models-to-et-soc-1-architecture-and-exporting-them-to-onnx/)
- [Esperanto Blog: SLMs with KV-cache ONNX](https://www.esperanto.ai/blog/exporting-slms-to-onnx-with-kv-cache-support/)
- [Esperanto GitHub](https://github.com/esperantotech)
- [EE Times: Esperanto Pivots to HPC and GenAI](https://www.eetimes.com/esperanto-pivots-to-hpc-and-generative-ai/)
- [Ainekko: Open-Sources Esperanto IP](https://www.eetimes.com/ainekko-buys-esperanto-hardware-ip-open-sources-it/)
