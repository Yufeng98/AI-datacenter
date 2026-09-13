# OpenAI-Broadcom Jalapeño — Software Stack Investigation

*as_of: 2026-08-08*
*chip: openai-broadcom*
*product name: Jalapeño (official, unveiled 2026-06-24)*
*prior repo codename: "Project Titan" — unsourced*
*confidence: low (very limited public disclosure; no SDK; silicon at engineering-sample stage)*

> **Sections 1–9 below are the April 2026 baseline and are preserved as written** (they use the old
> "Titan" name throughout). See **"Jalapeño Software-Stack Update (2026-08-08)"** at the end for what
> changed after the 2026-06-24 unveiling; that section is authoritative where the two disagree.

---

## Overview

The OpenAI-Broadcom Titan chip has a software stack that is almost entirely not public as of April 2026. The chip has not been shipped to customers and the SDK has not been publicly released. However, several signals from OpenAI job postings, public statements, and the company's existing open-source infrastructure (Triton compiler) allow informed inference about the stack design.

---

## 1. Framework Integration Layer

### PyTorch (Primary)
OpenAI's entire research and production infrastructure is built on PyTorch. The Titan chip will necessarily support PyTorch as its primary entry point for model developers. The integration path is inferred to follow the standard custom accelerator pattern:

```
torch.compile → custom backend → Triton dialect → Titan runtime
```

- **Status**: Not publicly released. Inferred from company-wide PyTorch dependency.
- **Confidence**: High (inferred)

### Other Frameworks
JAX, TensorFlow: not disclosed. OpenAI's internal use of these frameworks is minimal compared to PyTorch. Support is plausible via ONNX but not confirmed.

---

## 2. Compiler / IR Layer

### Triton (Confirmed Signal)

The most important public signal about the Titan software stack comes from OpenAI's internal job postings (active as of Q1 2026):

> "The Compiler/Kernels team is responsible for the performance-critical software stack that brings OpenAI's custom silicon to life. They design and build the compilers, languages, and high-performance kernels that allow researchers to fully exploit their first-party accelerators. This includes **advancing Triton and its backend**, developing new compiler passes, and creating the tooling needed to write fast, correct, and deeply optimized kernels for brand-new hardware."

**Implication**: OpenAI is extending Triton — the Python GPU programming language they created and open-sourced — to support the Titan hardware backend. Triton already has backends for NVIDIA (PTX/SASS), AMD (GCDNative), and Intel GPUs. A Titan ASIC backend would follow the same pattern.

#### Triton Architecture (for context)
```
Python kernel (@triton.jit decorated function)
         ↓
Triton frontend (Python AST → Triton-IR, SSA form)
         ↓
Triton-IR (MLIR dialect, tile-based blocked representation)
         ↓
MLIR optimization passes (auto tiling, shared memory promotion, loop unrolling)
         ↓
[Backend: NVIDIA → LLVM → PTX | AMD → LLVM → GCNISA | Intel → LLVM → SPIR-V]
         ↓
[OpenAI Titan backend: LLVM → Titan ISA — not public]
```

- **Triton repository**: https://github.com/triton-lang/triton
- **Titan backend**: Not public. Would be added as a new LLVM codegen target.

### torch.compile Integration

`torch.compile` (TorchDynamo + TorchInductor) is the standard PyTorch compilation pathway. For Titan, the expected flow:

```
torch.compile(model, backend="openai_titan")
         ↓
TorchDynamo (graph capture)
         ↓
TorchInductor or custom inductor backend
         ↓
Triton kernels (Titan ISA target)
         ↓
Titan runtime
```

- **Status**: Not public. Standard pattern for custom accelerator integration.

### MLIR / XLA
Not confirmed. OpenAI has not publicly discussed XLA or HLO as compilation pathways for Titan. Triton/MLIR is the more likely compiler IR given OpenAI's ownership of Triton.

---

## 3. Op Library / Kernel Library

For a systolic array ASIC, the op library is typically:

- **Matrix multiply (GEMM)**: Handled by the systolic array hardware directly; no separate op library needed
- **Attention / FlashAttention**: Either hardwired (unlikely for systolic) or written as Triton kernels targeting the Titan backend
- **Layer norm, softmax, elementwise ops**: Likely Triton kernels or a bundled kernel library (not public)

**Status**: No public op library, no public kernel library. OpenAI job postings reference writing "high-performance kernels" for custom silicon, implying kernel authorship (Triton-based) rather than a pre-built cuDNN-equivalent library.

---

## 4. Runtime Layer

| Component | Status | Notes |
|-----------|--------|-------|
| Inference runtime | Not public | OpenAI runs its own inference infrastructure (internal); not a public cloud service |
| Model loading / weight management | Not public | HBM4 weight tiling and streaming strategy: not public |
| Batching and scheduling | Not public | Continuous batching (standard for LLM serving): inferred |
| Quantization support | Not public | FP8/INT8 quantization for inference: inferred from industry practice |
| vLLM / SGLang integration | Not public | OpenAI likely uses internal serving system, not vLLM |
| Serving API | Not public | Internal; OpenAI exposes public API at api.openai.com |

---

## 5. Driver / Firmware Layer

| Component | Status | Notes |
|-----------|--------|-------|
| Kernel-mode driver | Not public | PCIe device driver (inferred) |
| On-chip firmware | Not public | Whether systolic array requires on-chip firmware (vs. NVIDIA GSP model) is not public |
| DMA engine | Not public | Host-to-HBM4 and HBM4-to-PE weight streaming: not public |
| Management interface | Not public | |

---

## 6. Communication / Collective Layer

| Component | Status | Notes |
|-----------|--------|-------|
| AllReduce / AllGather library | Not public | Broadcom Ethernet collective ops library (analog of NCCL): not public |
| Transport | Ethernet (Broadcom stack) | Confirmed in partnership announcement |
| Collective algorithm | Not public | Broadcom Jericho4 supports in-network computing for AI collectives |
| RDMA | Not public | RoCE v2 likely given Broadcom Ethernet stack |

---

## 7. Software Stack Comparison

| Layer | OpenAI Titan | NVIDIA GPU | Google TPU | AWS Trainium |
|-------|-------------|------------|------------|--------------|
| Framework | PyTorch (inferred) | PyTorch/JAX/TF | JAX/TF | PyTorch/JAX/TF |
| Compiler | Triton (Titan backend, inferred) | nvcc/Triton/XLA | XLA (HLO) | Neuron Compiler (XLA fork) |
| Kernel library | Not public (Triton kernels) | CUTLASS/cuDNN | Built into XLA | NeuronCore built-ins |
| Runtime | Not public (internal) | CUDA Runtime | TPU Runtime | Neuron Runtime |
| Driver | Not public | nvidia.ko + GSP | — | Neuron Driver |
| Networking | Broadcom Ethernet (not public) | NCCL + NVLink | DCN/ICI | EFA + NeuronLink |
| ISA | Not public | PTX + SASS | Not public | Not public |
| SDK publicly released | No | Yes (CUDA) | Partial (JAX) | Partial (Neuron SDK) |

---

## 8. What Is Not Public

- Titan software SDK (not released)
- Triton backend for Titan (not open-sourced)
- Kernel library or op library for Titan
- Inference runtime implementation
- Driver source code
- ISA / instruction encoding
- Quantization toolchain
- Model compilation and deployment workflow
- Any developer documentation

---

## 9. Strategic Software Stack Observations

**Triton as the unifying compiler**: OpenAI's choice to build on Triton rather than a proprietary compiler IR (like Qualcomm's AIC100 or Huawei's ACL) is strategically important. Triton is already familiar to ML engineers who write GPU kernels. Extending it to Titan reduces the ecosystem gap vs. CUDA.

**Inference-first design**: The stack prioritizes inference serving (the business-critical workload for OpenAI's API). Training will still rely on NVIDIA GPUs for the near term, per public statements.

**Closed ecosystem (for now)**: Unlike NVIDIA (open CUDA), Google (open JAX/XLA), or even Tenstorrent (open tt-metal), OpenAI's Titan stack is fully internal. No developer ecosystem outside OpenAI is being enabled. This limits Titan to internal OpenAI workloads — the 10 GW deployment is entirely for OpenAI's own inference infrastructure and partner data centers (not a general-purpose cloud service in the traditional sense).

---

## Jalapeño Software-Stack Update (2026-08-08)

*Scan date 2026-08-08. Trigger: the 2026-06-24 OpenAI/Broadcom unveiling. The unveiling was a hardware
announcement — it released **no** SDK, compiler, runtime, driver or ISA detail. What it did release is a
single but consequential bring-up data point.*

### 10.1 Naming

The chip's official name is **Jalapeño**. Everywhere the baseline sections above say "Titan backend",
read "Jalapeño backend". "Project Titan" has no primary source and should not be cited.

### 10.2 The one real software signal: end-to-end bring-up on silicon

> "Engineering samples of the Jalapeño chip are running ML workloads in the lab at production target
> frequency and power, including GPT-5.3-Codex-Spark."

Read as a software-stack finding, this is the first evidence that the whole toolchain exists and works
on real silicon:

| Layer | What the GPT-5.3-Codex-Spark run implies | Confidence |
|---|---|---|
| Framework / graph capture | A model-ingest path from OpenAI's PyTorch infrastructure to the accelerator exists | inferred (high) |
| Compiler (Triton + Jalapeño backend) | A working codegen backend emitting executable Jalapeño code exists | inferred (high) |
| Kernel library | Kernels sufficient for a production-class LLM (attention, norm, softmax, GEMM) exist | inferred (high) |
| Runtime | Weight loading, batching and token streaming work well enough to serve a real model in-lab | inferred (high) |
| Driver / firmware | Host↔device enumeration and DMA work at production target frequency and power | inferred (high) |
| Collectives / multi-chip | **No evidence either way** — the statement describes lab operation, not fleet or multi-node serving | unknown |

None of these layers had any implementation detail disclosed. Nothing here should be recorded as a
confirmed software fact beyond "the stack ran a model on engineering samples".

### 10.3 What did NOT change

| Item | Status as of 2026-08-08 |
|---|---|
| Public SDK | **Still none.** No developer access, no documentation, no download |
| Triton Jalapeño backend | Still not open-sourced; not present in the public triton-lang/triton repository |
| Kernel / op library | Still not public |
| Inference runtime | Still not public |
| Driver source | Still not public |
| ISA / instruction encoding | Still not public |
| Quantization toolchain | Still not public |
| Collective-communication library | Still not public and still unnamed. Only the *switch silicon* (Tomahawk) was named, not the software |
| Public benchmark | Still none — no MLPerf, no third-party result |

### 10.4 Model-scope statement (do not over-read)

OpenAI describes Jalapeño as "designed with flexibility to work with all LLMs" and "built from the ground
up for current and future LLMs across the industry". This is a statement about **architectural
generality**, and at the software level it implies only that the compiler/runtime path is not
hard-specialized to one model family. **It is not a commitment to release an SDK, to open the stack, or
to sell the chip to third parties**, and none of those should be inferred.

### 10.5 Design-time use of OpenAI models (vendor claim)

OpenAI states it used its own models "to accelerate parts of the design and optimization process" for
Jalapeño. This concerns EDA/design tooling rather than the deployed software stack, and is an
unverifiable vendor claim. Recorded for completeness only.

### 10.6 Scheduled disclosure — Hot Chips 38

Richard Ho, Ravi Narayanaswami and **Chris Leary** (OpenAI) present "You Can Just Build Things … Chips"
in the AI 2 session on **Tuesday 2026-08-25, 4:45–6:15 PM PDT**. Leary's presence is notable from a
software-stack perspective (compiler background), which makes this the most likely venue for the first
public description of the Jalapeño compiler/runtime path.

*Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* No slides or abstract exist as
of 2026-08-08; nothing in this investigation derives from the talk. **Action: re-scan after 2026-08-25.**

---

## Resources

### Jalapeño unveiling (2026-06-24)

- Broadcom IR press release (durable primary): https://investors.broadcom.com/news-releases/news-release-details/openai-and-broadcom-unveil-llm-optimized-intelligence-processor
- Yahoo Finance carry (full verbatim joint release): https://finance.yahoo.com/technology/ai/articles/openai-broadcom-unveil-llm-optimized-130000020.html
- TechCrunch (still in testing, "early results"): https://techcrunch.com/2026/06/24/openai-unveils-its-first-custom-chip-built-by-broadcom/
- Engadget: https://www.engadget.com/2201045/openai-broadcom-jalapeno-inference-processor-ai-accelerator/
- Hot Chips 38 advance program (scheduled talk; content not yet public): https://www.hotchips.org/advance-program/

### Baseline (April 2026 and earlier)

- OpenAI-Broadcom partnership announcement: https://openai.com/index/openai-and-broadcom-announce-strategic-collaboration/
- OpenAI Triton compiler job posting (stack signal): https://openai.com/careers/software-engineer-triton-compiler-san-francisco/
- Triton repository: https://github.com/triton-lang/triton
- TrendForce roadmap (N3/A16): https://www.trendforce.com/news/2026/01/15/news-openai-reportedly-to-deploy-custom-ai-chip-on-tsmc-n3-by-end-2026-second-gen-planned-for-a16/
- Medium — Titan chip analysis: https://medium.com/hardware-for-agi/openais-titan-chip-the-500-billion-bet-to-break-free-from-nvidia-3e077eca3be9
- DCD — mass production 2026: https://www.datacenterdynamics.com/en/news/openai-to-start-mass-producing-its-custom-ai-chip-in-2026-report/
