# Microsoft Maia Software Stack — Investigation Report

*as_of: 2026-04-05*
*Primary sources: Azure Blog "From silicon to software to systems"; Microsoft Tech Community "Inside Maia 100"; Maia 200 announcement and deep-dive blogs*

---

## Overview

Microsoft's Maia software stack is designed around the principle of **developer agility with hardware-optimized performance paths**. The stack offers two programming models — Triton (portable, GPU-compatible) and NPL (Maia-native, maximum performance) — fronted by a first-class PyTorch backend. The Maia SDK is in preview for Azure customers; no components are open-sourced.

The primary deployment target is **Azure OpenAI Services** (Copilot, Azure AI Studio), where Maia runs Microsoft-first-party models. External developer access is via the preview SDK.

---

## 1. Framework Integration

### PyTorch Backend
- **First-class PyTorch backend** — the primary user-facing interface
- Supports both **eager mode** (dynamic dispatch) and **graph mode** (torch.compile-style capture)
- Standard PyTorch model code runs without modification; Maia-specific optimizations are below the framework boundary
- Enables rapid deployment of existing models from Azure AI model catalog

### Azure OpenAI Services Integration
- Maia serves production inference for Azure OpenAI models (GPT-4 class, Copilot)
- Models are optimized for Maia using the SDK compiler pipeline before serving
- The end-user does not interact with Maia directly; it is Azure-internal infrastructure

**Confidence: confirmed (from Azure Blog and Microsoft Tech Community)**

---

## 2. Compiler / IR

### Two Programming Models

Microsoft exposes two compilation paths for Maia:

#### Path 1: Triton Programming Model
- Uses **OpenAI Triton** (open-source Python DSL) as the authoring language
- Provides **agility and portability** — same Triton kernel can run on NVIDIA GPUs and Maia
- **Triton compiler for Maia:** Maia-specific backend lowering Triton IR to Maia ISA/hardware
- The Maia Triton backend itself is **not open-sourced**

#### Path 2: Maia API / NPL (Nested Parallel Language)
- **NPL (Nested Parallel Language)** — Microsoft's custom programming model
- Provides **explicit control** of:
  - Data movement (DMA scheduling)
  - SRAM placement (tile/cluster SRAM tier selection)
  - Parallel execution (tile/cluster parallelism mapping)
- Achieves near-peak hardware utilization
- Used internally by Microsoft engineers to author the optimized kernel library
- **Not publicly released or documented**

**Confidence: confirmed (from Azure Blog SDK overview)**

---

## 3. Op Library

Microsoft has developed a set of **highly optimized ML compute and communication kernels** using the SDK compilers. These include:

- GEMM / batched GEMM (matrix multiplication for attention and FFN layers)
- Attention kernels (likely flash-attention style, SRAM-tiling aware)
- Collective communication kernels (all-gather, scatter-reduce, all-to-all matching Maia network primitives)
- Activation and normalization ops

**Status: not public** — the kernel library is an Azure-internal artifact. External developers can author custom kernels using NPL or Triton.

**Confidence: confirmed existence, not public (inferred from Azure Blog)**

---

## 4. Kernel Library

Custom kernel authoring is **supported** via both programming models:
- Triton: familiar syntax for ML researchers
- NPL: more control for performance-critical inner loops

A set of pre-built kernels is included in the SDK (not publicly available). These are the production kernels used to run Azure OpenAI workloads on Maia.

**Status: not public**

---

## 5. Runtime

The Maia runtime (not publicly named) handles:
- Kernel dispatch and scheduling
- Memory management (SRAM allocation, HBM buffer management)
- DMA queue management
- Synchronization (hardware semaphores)

The runtime is delivered as part of the **Maia SDK** (preview), accessible to Azure customers. No public source, no public documentation of the runtime API.

**Status: not public (SDK preview only)**

---

## 6. Driver / Firmware

### Driver
- Maia kernel driver (Azure-internal): manages PCIe BAR, command submission, interrupt handling
- Not publicly released; not upstreamed to Linux kernel

### Firmware
- Maia 100 is explicitly stated to support **firmware updates that "unlock new capabilities"** (per TechRadar)
- Suggests a substantial on-chip firmware layer (analogous to NVIDIA's GSP firmware)
- Firmware is Azure-managed; not accessible to end users

**Status: not public**

---

## 7. Communication

### Maia 100: Custom RoCE-like Protocol
- Custom transport layer over Ethernet, extending standard RoCE
- Provides enhanced reliability and load balancing (beyond standard RoCE)
- **AES-GCM encryption** natively built into the transport — enables **confidential compute**
- Collective operations: all-gather + scatter-reduce (4800 Gbps), all-to-all (1200 Gbps)
- Unified fabric: same network handles both scale-up (intra-pod) and scale-out (inter-pod)

### Maia 200: Integrated On-die NIC
- Transport logic integrated directly on the Maia 200 die (not a separate NIC)
- 2.8 TB/s bidirectional bandwidth
- 2-tier topology supporting clusters of up to 6,144 accelerators
- Advanced congestion control and reliability built in

**Confidence: confirmed (HC2024, Azure Blog, Maia 200 deep-dive)**

---

## 8. Assembler / ISA

### Vector Processor ISA
- Custom ISA powering the loosely-coupled superscalar vector processor
- Supports FP32 and BF16 data types
- **Not publicly documented** — no PTX equivalent, no public assembler

### Tensor Unit ISA
- The 16×R×16 tensor unit on Maia 100 (and TTU on Maia 200) has microarchitecture-specific instructions
- **Not publicly documented**

### MX Data Format (OCP Standard)
- Microsoft co-developed the **MX (Microscaling) format** with AMD, Arm, Intel, Meta, Qualcomm, NVIDIA under the Open Compute Project (OCP)
- MX is a hardware-native format on Maia (MX4/MX6/MX9 on Maia 100; FP4/FP8 on Maia 200)
- The OCP MX specification is public (opencomputeproject.org); the Maia ISA implementation is not

**Status: not public**

---

## 9. SDK Developer Tools

| Tool | Description |
|------|-------------|
| Triton compiler for Maia | Compiles Triton kernels to Maia; portability path |
| NPL compiler | Compiles NPL (Nested Parallel Language) to Maia; performance path |
| PyTorch backend | Framework integration; eager + graph mode |
| Maia simulator | Software emulator for pre-silicon and pre-deployment validation |
| Cost calculator | Estimates performance and utilization before running on hardware |
| Debugger | Kernel-level debugging on Maia |
| Profiler | Execution timeline and bottleneck analysis |
| Visualizer | Kernel/graph visualization |
| Quantization tools | Model quantization to MX/FP8/FP4 formats |
| Validation suite | Numerical correctness checking |

---

## 10. What Is Public vs. Not Public

| Component | Status | Notes |
|-----------|--------|-------|
| PyTorch backend | confirmed-public (concept) | Exists; implementation not open-sourced |
| Triton DSL | confirmed-public | openai/triton is open; Maia backend is not |
| NPL language | confirmed-exists, not public | Described in Azure Blog; no spec or repo |
| Compiler source | not public | Azure-internal |
| Kernel library | not public | Azure-internal |
| Runtime | not public | SDK preview only |
| Driver | not public | Azure-internal |
| Firmware | not public | Azure-managed |
| ISA specification | not public | No PTX equivalent |
| SDK | preview | Available to Azure customers, not open-sourced |
| MX data format spec | public (OCP) | opencomputeproject.org; not Maia-specific |

---

## Sources

- [Azure Blog — from silicon to software to systems](https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/)
- [Microsoft Tech Community — Inside Maia 100](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118)
- [Microsoft Blog — Maia 200 announcement](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/)
- [Microsoft Tech Community — Maia 200 deep dive](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312)
- [TechRadar — HBM2e and firmware update capability](https://www.techradar.com/pro/microsoft-deliberately-chose-to-use-old-tech-for-its-nvidia-gpu-rival-maia-100-ai-accelerator-uses-hbm2e-memory-and-the-mysterious-ability-to-unlock-new-capabilities-via-firmware-update)
- [HC2024 PDF — Inside Maia 100](https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf)

---

## Update investigation — window 2026-04-05 → 2026-08-08 (dated 2026-08-08)

*Verdict: **no software-stack change.** No SDK GA, no version number, no changelog, no open-sourcing, no public ISA document, no public kernel library, no upstream driver. Every row in sections 1–10 above stands. The one new fact is a **workload** fact, not a stack fact.*

### 1. Maia SDK — still preview, re-confirmed

| Attribute | Status 2026-08-08 |
|---|---|
| Availability | Preview only, sign-up-gated (selected academics, developers, frontier labs, open-source contributors) |
| GA | **Not announced.** No GA date, no GA timeline |
| Version number / changelog / release notes | **None public** |
| Public documentation tree | **None** — `https://learn.microsoft.com/en-us/azure/maia/` returns HTTP 404 |
| Component list | **Unchanged**: PyTorch backend, Triton compiler for Maia, NPL compiler, Maia simulator + cost calculator, debugger/profiler/visualizer, quantization + validation tools, pre-built kernel library |
| Openness below the PyTorch/Triton API | **Still fully closed**: no compiler source, no ISA spec, no kernel library, no runtime source, no upstreamed driver, no firmware |

The 404 on the `learn.microsoft.com/azure/maia` docs tree is a useful ongoing signal: Microsoft publishes a docs tree for GA Azure hardware surfaces, and its continued absence is consistent with the SDK remaining gated.

### 2. New workload class — Microsoft's own MAI models on Maia 200

This is the survey-relevant change in the window. Before Build 2026, the recorded Maia workload was Azure OpenAI Services inference (Copilot, Azure AI Studio). At Build 2026 (2026-06-02) Microsoft launched **seven MAI models across five families** and stated that it is *"co-designing our models with our own silicon, optimizing MAI-Thinking-1 on our Maia 200 chip."*

| MAI model | Notes |
|---|---|
| **MAI-Thinking-1** | Microsoft's **first reasoning model**; reported as a ~35B-active-parameter MoE with a 256K context window; explicitly named as the model co-designed with and optimized on Maia 200, and benchmarked head-to-head against GB200 |
| MAI-Image-2.5 | image |
| MAI-Transcribe-1.5 | speech-to-text |
| MAI-Voice-2 | speech |
| MAI-Code-1-Flash | code |

**Software-stack significance.** This is Microsoft's **first public claim of model↔accelerator co-design on Maia** — the compiler/kernel stack is being tuned against a first-party model family rather than only against OpenAI models served through Azure. **However: no mechanism was disclosed.** Nothing was published about which parts of the stack were changed (NPL kernels, Triton schedules, quantization recipe, KV-cache placement across TSRAM/CSRAM, collective schedules), so the co-design claim is recorded as a workload/positioning fact, not as an architectural or toolchain fact.

The associated "further 1.4× performance-per-watt" figure is a **vendor claim with no published methodology** (no workload, sequence length, batch size, precision, cluster size, or power-measurement boundary) and is stated as incremental on top of the January 2026 "30% better performance per dollar" claim — which is a perf/**dollar** claim against Microsoft's own fleet. The primary transcript does **not** state the 1.4× baseline; "vs GB200" is press inference. It must not enter any comparison table.

### 3. Negative results

- **No Maia compiler, runtime, driver, kernel-library or ISA artifact became public** in the window. No new GitHub organization, repository, or PyPI/NuGet package.
- **No Triton-for-Maia backend upstreaming** into `openai/triton`.
- **No public NPL specification, tutorial, or sample corpus.**
- **No MLPerf submission** to legitimise any Maia software-stack performance number (MLPerf Training v6.0, 2026-06-16 — Azure's entry was 8,192 NVIDIA GB200 GPUs).
- **MRC (Multipath Reliable Connection)** — an OCP Ethernet transport co-developed by Microsoft (arXiv:2606.18170, 2026-06-16) — **does not mention Maia**. It is not part of the Maia communication-layer stack on any available evidence; recorded as adjacent ecosystem context only.

### 4. What to re-check after 2026-08-25

Hot Chips 38 session AI 1 ("MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng, Microsoft) is scheduled for 2026-08-25 and **its content is not yet public**. Historically the Hot Chips Maia talk (HC2024) was the single richest source for stack detail — the NPL/Triton split and the DMA/semaphore programming model both come from it. Re-run this investigation once slides are released.

### Sources consulted in this window

- https://microsoft.ai/news/microsoft-build-2026-mai-keynote-transcript/ — 2026-06-02, primary
- https://microsoft.ai/news/building-a-hillclimbing-machine-launching-seven-new-mai-models/ — MAI model family
- https://learn.microsoft.com/en-us/azure/maia/ — HTTP 404, negative control on SDK GA
- https://hotchips.org/program/conference/ — scheduled talk, no content
- https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ — no Maia entry
- https://arxiv.org/abs/2606.18170 — MRC transport, does not mention Maia
- https://techcommunity.microsoft.com/category/azure/blog/azureinfrastructureblog — Apr–Aug 2026 silicon posts are Cobalt-only
