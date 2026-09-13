# Qualcomm Cloud AI 100/200 — Software Stack Investigation

## Summary

The Cloud AI 100 software stack is well-documented and partially open-source. The Linux kernel driver is upstream. The primary compilation path goes: PyTorch/ONNX → qaic-compile → QNN context binary → libQAic runtime → AIC100. Qualcomm also supports ONNX Runtime's QNN Execution Provider and ExecuTorch backend.

---

## Full Stack

```
PyTorch / TensorFlow / ONNX (model sources)
         |
         v
AI Hub Workbench / qaic-compile CLI
  - ONNX import, TorchScript import
  - Operator lowering to QNN primitives
  - INT8 quantization
  - Fusion (conv+BN, GEMM+activation)
  - Output: QNN context binary
         |
         v
libQAic (Cloud AI SDK — C++/Python runtime)
  - Session management
  - Async inference queuing
  - Multi-SoC tensor sharding
         |
         v
QAIC kernel driver (Linux: drivers/accel/qaic/)
  - DMA buffer management
  - MQ DMA (multi-queue)
  - PCIe communication
         |
         v
AIC100 firmware
         |
         v
AIC100 hardware (AI cores, SRAM, DRAM)
```

---

## Alternative Paths

### ONNX Runtime QNN Execution Provider
```
ONNX model → ONNX Runtime → QNN EP → QNN SDK → AIC100
```

### ExecuTorch (PyTorch edge/server)
```
PyTorch model → torch.export → ExecuTorch → Qualcomm AI Engine Direct backend → AIC100
```

---

## Compiler (qaic-compile)

- Input: ONNX (dynamic shapes supported), TorchScript
- Quantization: post-training INT8 quantization
- Fusion: conv+BN, GEMM+activation, multi-op fusion
- Custom ops: C++ plugin API
- Output: QNN context binary (device-specific)
- CLI tool: `qaic-exec` for benchmark/test

---

## Runtime (libQAic / Cloud AI SDK)

- C++ API with Python bindings
- Session API: load model, queue inputs, retrieve outputs
- Async inference: multiple in-flight requests
- Multi-SoC: shards large models across 4 SoCs on Ultra card
- Available on GitHub: https://github.com/quic/cloud-ai-sdk

---

## Kernel Driver

- Location: `drivers/accel/qaic/` in Linux kernel tree (upstream since kernel 6.3)
- DMA buffer management: IOMMU mapping
- MQ DMA: multi-queue for high-throughput inference serving
- Documented: https://docs.kernel.org/accel/qaic/aic100.html

---

## Openness

| Component | Status |
|---|---|
| QAIC Linux driver | Open (upstream kernel) |
| Cloud AI SDK (libQAic) | Open (GitHub: quic/cloud-ai-sdk) |
| QNN SDK | Public release (not fully open) |
| Compiler (qaic-compile) | Closed binary |
| Chip design / firmware | Closed |

---

## Key Software Design Decisions

1. **Upstream kernel driver**: enables first-class Linux support across all distros
2. **ONNX-first compilation**: maximizes compatibility with training frameworks
3. **QNN as universal abstraction**: same QNN graph targets mobile and cloud Hexagon cores
4. **libQAic async API**: designed for high-throughput inference server (Triton-compatible)
5. **Multi-SoC transparency**: Ultra card 4-SoC sharding transparent to application

---

# Investigation Update — Modular acquisition and SDK releases (2026-08-08)

*Additive. The `qaic-compile` → QNN context binary → `libQAic` → QAIC driver path documented above is unchanged
and remains Qualcomm's production compilation path for Cloud AI 100.*

## Qualcomm acquires Modular — COMPLETED 2026-07-29

This is the largest change to Qualcomm's AI software posture in the survey period.

| Event | Date | Source |
|---|---|---|
| Announced ("Qualcomm to Acquire Modular"), close guided to H2 2026 | 2026-06-24 | Qualcomm PR |
| **Completed** ("Qualcomm Completes Acquisition of Modular") | **2026-07-29** | Qualcomm newsroom **and** Modular's own blog post of the same date |

- **Financial terms: not disclosed.**
- **Mojo, MAX and Modular Cloud continue as products and brands** under Qualcomm.
- **Chris Lattner** became **Qualcomm EVP of Advanced AI Software and Platforms**.
- Qualcomm's June acquisition PR **does not itself name Mojo or MAX**. It describes acquiring "an open, AI-native
  software stack" that runs "across CPU, GPU, NPU, and custom ASIC architectures". The Mojo/MAX identification
  comes from Modular's side.

### Correct framing (important)

Modular adds an **open, MLIR-based, silicon-agnostic compute layer alongside** Qualcomm's existing closed
`qaic-compile` + QNN context binary + `libQAic` path. **Qualcomm has not said Modular replaces that stack**, and
no source describes a Mojo or MAX backend for AI200/AI250/AI300. Describing the acquisition as Qualcomm
"replacing its software stack" is an overstatement and is not written anywhere in this repo.

What it does signal, strategically: Qualcomm is buying a "write once, run anywhere" compiler front end at the same
moment it announces a CPU line (C1000) and two new accelerator generations (AI250, AI300) that it will have to
make programmable. The pairing of an in-house CPU, near-memory accelerators, and an acquired silicon-agnostic
compiler is the coherent read.

## Cloud AI 100 SDK — quic/efficient-transformers

**Date correction.** An earlier scan attributed WAN non-unified execution, `transformer_high`/`transformer_low`
modules, and first-block-cache for WAN/FLUX to "April 2026 additions". **There is no April 2026 release.** The
correct provenance:

| Release | Date | Contents |
|---|---|---|
| v1.21.0 | **2025-12-22** | FLUX.1-schnell and WAN 2.2 diffusion support |
| v1.22.0 | **2026-06-18** | WAN 2.2 dual-stage high/low-noise transformers; first-block-caching infrastructure for Diffusers models; blocked-KV attention; layerwise ONNX export for large MoE; moves to HF Transformers 5.5.4 and Python 3.12 |

Assessment: **incremental and generative-media-focused, not architectural.** Critically, the release notes show
**no AI200/AI250 support** — the SDK remains Cloud AI 100-targeted. There is as of 2026-08-08 no published
toolchain for any Dragonfly accelerator.

## Other software-adjacent announcements (2026-06-24)

- Qualcomm and **Hugging Face** expanded their relationship "from device to cloud".

## Openness table (updated)

| Component | Status | Change |
|---|---|---|
| QAIC Linux driver | Open (upstream kernel) | unchanged |
| Cloud AI SDK (libQAic) | Open (GitHub: quic/cloud-ai-sdk) | unchanged |
| quic/efficient-transformers | Open | v1.22.0, 2026-06-18; Cloud AI 100 only |
| QNN SDK | Public release (not fully open) | unchanged |
| Compiler (qaic-compile) | Closed binary | unchanged |
| **Mojo** | Open-ish, MLIR-based; Modular-governed | **new — Qualcomm-owned since 2026-07-29** |
| **MAX inference platform / Modular Cloud** | Modular products, continuing under Qualcomm | **new — Qualcomm-owned since 2026-07-29** |
| Chip design / firmware | Closed | unchanged |

## Sources for this section

- https://www.qualcomm.com/news/releases/2026/06/qualcomm-to-acquire-modular
- https://www.qualcomm.com/news/releases/2026/07 (listing confirming "Qualcomm Completes Acquisition of Modular", 2026-07-29)
- https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular
- https://www.modular.com/blog
- https://github.com/quic/efficient-transformers/releases
- https://www.qualcomm.com/news/releases
