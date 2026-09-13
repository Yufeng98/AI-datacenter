# Untether AI — Software Stack Investigation

*as_of: 2026-08-08*

**SDK:** imAIgine SDK
**GitHub:** https://github.com/untetherai

---

## Status — DISCONTINUED; vendor bankrupt

The imAIgine SDK was **discontinued in June 2025** and has no successor. Untether AI Corporation subsequently
filed an **assignment in bankruptcy on 2025-10-15** (BIA liquidation, Ontario Estate No. 31-3285414, trustee
PricewaterhouseCoopers Inc., LIT).

**Correction to the earlier "AMD acquired the team" framing:** per the trustee's report, the June 2025 "AMD
Transaction" conveyed only the **exclusive right to negotiate new employment with certain Untether employees**
for **US$25M paid to Untether**. **AMD acquired no IP, no products, and no software.** The imAIgine compiler,
allocator, and runtime did not transfer to AMD or to anyone else.

**Nothing was open-sourced.** `github.com/untetherai` contains only forks of third-party tools
(`py-cpp-demangle`, `onnx-simplifier`, `damoyolo`, `onnx2torch2`, `YOLOX`, `turnkeyml`, `yolov3`), newest
activity **May 2025**, with **no imAIgine SDK or compiler source**. The documentation URLs below
(`untether.ai/...`) are dead — the domain fails TLS handshake as of 2026-08-08. Every claim in this file is
therefore reconstructed from vendor announcements and press coverage published while the company operated, and
is **not independently verifiable against source code**.

Primary status source: [PwC Inc. Trustee's Report to the First Meeting of Creditors,
2025-10-30](https://www.pwc.com/ca/en/car/untether/assets/untether-004_311025.pdf).

This file is retained in the past tense as a historical record of the stack's design.

---

## imAIgine SDK Overview

The imAIgine SDK (pronounced "imagine") was Untether AI's end-to-end inference deployment toolkit.
It provided a push-button path from a trained PyTorch/TensorFlow/ONNX model to running inference
on speedAI hardware, handling quantization, compilation, physical mapping, and runtime.

---

## SDK Layer Architecture

```
User-Facing API (Python)
    ↓
Framework Ingestion Layer
    ├── PyTorch (via torch.export / torchscript)
    ├── TensorFlow (SavedModel / concrete functions)
    └── ONNX (direct import)
    ↓
Quantization Engine
    ├── Post-Training Quantization (PTQ) — automated calibration
    ├── Post-Quantization Training (PQT) — fine-tune after quant
    ├── Knowledge Distillation — accuracy recovery
    └── Target datatypes: INT4, INT8, FP8, BF16
    ↓
Generative Compiler (v25.04+)
    ├── Automatic kernel generation for novel NN layers
    ├── 4× model coverage expansion (300+ models, v25.04)
    └── Minutes to support new layer types
    ↓
Graph Optimizer / Lowering
    ├── Operator fusion
    ├── Layout transformation (bank-friendly tiling)
    └── Sparsity exploitation (2:1 structured)
    ↓
Physical Allocator
    ├── Maps ops → specific SRAM banks
    ├── Multi-chip partitioning (4-card system)
    └── Data placement optimization
    ↓
Runtime (imAIgine Runtime)
    ├── Host-side orchestration (x86 / ARM)
    ├── PCIe DMA management
    └── Inference session management
    ↓
speedAI Hardware (PCIe card)
```

---

## Software Components

### 1. Framework Ingestion

| Component | Supported Formats |
|---|---|
| PyTorch | torch.export, TorchScript, ONNX export |
| TensorFlow | SavedModel, concrete functions |
| ONNX | Direct ONNX model import |

### 2. Quantization Engine

- **Automated PTQ:** calibration dataset → per-layer scale/zero-point
- **PQT (Post-Quantization Training):** QAT-style fine-tuning after initial quantization
- **Knowledge Distillation:** float model as teacher, quantized as student
- **Precision targets:** INT4 (extreme compression), INT8 (default), FP8 (accuracy-efficient), BF16

### 3. Compiler

| Release | Key Feature |
|---|---|
| v1.0 (2023) | Basic operator lowering; manual kernel flow option |
| v2.0 (2023) | High-Performance Compute (HPC) flow; bare-metal kernel programming |
| EA (2024) | Early access for speedAI240; ONNX support |
| v25.04 (2025) | Generative compiler — auto-generates kernels for unsupported layers; 300+ model support |

**Generative Compiler (v25.04, March 2025 — final release):** vendor described it as using template synthesis
to automatically produce SRAM-bank-mapped kernels for previously unsupported layer types (e.g., attention
variants, new activations), reducing layer onboarding from weeks to minutes. **Vendor claim; no source or
independent evaluation is available.**

### 4. Bare-Metal / HPC Programming Flow

A second programming path (akin to writing CUDA PTX kernels) for expert users:
- Direct PE array programming within a bank
- Custom RISC-V assembly extensions
- Manual SRAM layout control
- Enabled library developers to implement highly optimized kernels

### 5. Physical Allocator

Mapped the computational graph onto the 729-bank physical topology:
- Bank assignment per tensor
- Intra-bank PE scheduling
- Inter-bank data routing via scratchpad
- Multi-chip (4× card) pipeline partitioning

### 6. Runtime

- Host library (`libimaigine.so`) loaded by Python wrapper
- PCIe data transfer management (DMA engine)
- Inference session lifecycle
- Power mode control (sport vs. eco)

---

## Supported Neural Network Domains (v25.04)

- Image classification (ResNet, EfficientNet, ViT families)
- Object detection (YOLO variants, SSD, FCOS)
- Semantic segmentation (DeepLab, SegFormer)
- Error/anomaly detection
- Speech / audio (limited)
- Early generative model support (limited, pre-shutdown)

---

## Toolchain Summary Table

| Layer | Tool / Component |
|---|---|
| Framework ingestion | PyTorch, TensorFlow, ONNX |
| Quantization | imAIgine PTQ / PQT / KD |
| Compiler | imAIgine Generative Compiler (v25.04) |
| Bare-metal kernels | HPC Flow (RISC-V + PE programming) |
| Physical allocation | imAIgine Allocator |
| Runtime | imAIgine Runtime (Python + C lib) |
| Benchmark integration | MLPerf Loadgen |
| Hardware interface | PCIe BAR + DMA |

---

## Key Differentiators vs. GPU Software Stack

(Comparison describes imAIgine as it stood at its final v25.04 release, March 2025.)

| Aspect | Untether (imAIgine) | NVIDIA (CUDA ecosystem) |
|---|---|---|
| Memory abstraction | Banks were compute+storage; no explicit memcpy | Explicit H2D/D2H transfers |
| Kernel programming | RISC-V extensions + PE arrays | CUDA C++ / PTX |
| Compiler target | Bank physical layout (729 banks) | SM thread hierarchy |
| Quantization | Integrated in SDK | TensorRT / Torch-Quantization |
| Model support (peak) | 300+ (v25.04) | Virtually unlimited |
| Multi-chip | 4-card partitioning built-in | NCCL / NVLink |

---

## Sources

**Corporate status (primary):**

- [PwC Inc. — Trustee's Report to the First Meeting of Creditors, 2025-10-30](https://www.pwc.com/ca/en/car/untether/assets/untether-004_311025.pdf) — bankruptcy 2025-10-15; AMD Transaction was employment-negotiation rights only, US$25M, no IP/products/software acquired.
- [PwC Canada insolvency file — UNTETHER AI CORP.](https://www.pwc.com/ca/en/services/insolvency-assignments/untetherai.html)
- [github.com/untetherai](https://github.com/untetherai) — third-party forks only; last activity May 2025; confirms the SDK and compiler were never open-sourced.
- Negative evidence: `untether.ai` fails TLS handshake as of 2026-08-08; all vendor documentation links below are dead and retained for provenance only.

**Historical vendor material (all links dead as of 2026-08-08):**

- [imAIgine SDK page](https://www.untether.ai/products/software/)
- [SDK early access announcement](https://www.untether.ai/untether-ai-releases-early-access-to-imaigine-software-development-kit-supporting-speedai-inference-acceleration-solutions/)
- [HPC flow announcement](https://www.businesswire.com/news/home/20230117005711/en/Untether-AI-Increases-Developer-Velocity-and-Adds-High-Performance-Compute-Flow-to-the-imAIgine-Software-Development-Kit)
- [Generative compiler v25.04](https://www.businesswire.com/news/home/20250310953294/en/Untether-AI-Dramatically-Expands-AI-Model-Support-and-Speeds-Developer-Velocity-with-New-Generative-Compiler-Technology)
- [SDK bare-metal EE Times](https://www.eetimes.com/untether-ai-sdk-allows-bare-metal-programming/)
- [GitHub](https://github.com/untetherai)
