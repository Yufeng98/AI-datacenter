# Rain AI Software Stack

*as_of: 2026-08-08*
*Architecture: Digital In-Memory Computing (D-IMC) + Andes RISC-V*
*Vendor status: **DEFUNCT** — non-operating; patents assigned to OpenAI Opco, LLC, USPTO-recorded 2025-10-23*

---

## Overview

> **Status notice (2026-08-08).** Rain AI is defunct. **No part of this software stack was ever released**:
> no SDK, no developer documentation, no public API reference, and no open-source repository. Nothing survives
> the wind-down, and the disposition of the compiler/runtime source is **not disclosed** — the only documented
> transfer is the patent assignment to OpenAI Opco, LLC (USPTO-recorded 2025-10-23). This document records
> what Rain *described* building, not software anyone can obtain.

Rain AI's software stack was entirely proprietary and never publicly documented. The company described itself as "co-designing every layer of the AI stack" — hardware, compiler, and runtime — but released no open-source components, no SDK documentation, and no public API references at any point in its existence.

The following is reconstructed from Rain's website (dormant since 2024-06-27), partner press releases, and third-party coverage.

---

## Layer Summary

| Layer | Component | Status |
|-------|-----------|--------|
| Framework Integration | Unspecified (RISC-V generality implied) | Never released |
| Compiler / IR | Proprietary Rain compiler + Andes ACE/COPILOT | Never released |
| Op Library | Proprietary | Never released |
| Kernel Library | Proprietary | Never released |
| Runtime | Proprietary Rain runtime | Never released |
| Driver / Firmware | Proprietary | Never released |
| Communication | Not described | Never disclosed |
| Assembler / ISA | RISC-V RV64GCV (Andes AX45MPV) + custom D-IMC extensions | Partial (RISC-V base is open) |

---

## 1. Framework Integration

Rain stated that its RISC-V-based platform allowed developers "unparalleled flexibility to implement any operator and compile any model," implying PyTorch/ONNX model ingestion. No specific framework connector (torch.compile backend, ONNX Runtime EP, TVM target, etc.) was ever publicly documented or released.

**Status: never released.**

---

## 2. Compiler / IR

Rain described a full-stack compiler co-designed with the D-IMC hardware. Key elements:

- **Host control path**: Targeted Andes AX45MPV (RISC-V RV64GCV + RVV 1.0)
- **D-IMC dispatch path**: Used Andes ACE/COPILOT to define custom instructions that scheduled workloads into D-IMC compute tiles
- **Balanced pipeline design**: Rain claimed the compiler produced a balanced schedule across the RISC-V/D-IMC interface to maximize D-IMC utilization — a vendor claim, never demonstrated publicly

No IR format, compiler toolchain name, or open-source component was ever released.

**Status: proprietary; never released.**

---

## 3. Op Library

No operator library was ever documented. The D-IMC paradigm typically accelerates matrix-multiply-accumulate operations natively; other operators (activations, normalization, attention) would have run on RISC-V vector units or on D-IMC with custom kernels.

**Status: proprietary; never released.**

---

## 4. Kernel Library

No kernel library was ever documented or open-sourced.

**Status: proprietary; never released.**

---

## 5. Runtime

Rain described a proprietary runtime co-designed with the hardware and compiler. Features mentioned:

- Claimed support for training and inference workloads
- Claimed scaling across multiple chips (the IP-licensing pitch implied multi-instance deployment)
- No API surface, scheduling model, or memory management details were ever published

**Status: proprietary; never released.**

---

## 6. Driver / Firmware

No driver or firmware documentation was ever published. The host interface (PCIe/CXL/other) was never disclosed, so the kernel-mode driver architecture is unknown and will remain so.

**Status: never disclosed.**

---

## 7. Communication

No multi-chip or cluster communication library was ever described. Scale-out interconnect was never disclosed.

**Status: never disclosed.**

---

## 8. Assembler / ISA

The **base ISA** is RISC-V RV64GCV (Andes AX45MPV):
- RV64G: base integer + multiply + atomics + FP
- C: compressed instruction extension
- V: RISC-V Vector extension v1.0

Rain added **custom D-IMC instructions** via the Andes ACE/COPILOT framework. These custom instructions dispatched tensor operations to D-IMC tiles. The specific instruction encodings were proprietary and never published.

The RISC-V base ISA is open and documented at [riscv.org](https://riscv.org/specifications/). Rain's extensions were never made public.

---

## 9. IP Licensing (historical)

Rain stated that it offered IP licensing for its D-IMC compute tile and associated software stack, positioning the stack as licensable IP for integration into third-party SoCs rather than as a standalone developer platform. No licensee was ever announced. The only IP transfer confirmed in primary records is the outright assignment of Rain's patents to **OpenAI Opco, LLC**, USPTO-recorded **2025-10-23**; whether any software rights moved with them is **not disclosed**.

---

## Comparison to Industry

Rain AI's column is historical — the vendor is defunct and none of its stack was ever obtainable.

| Dimension | Rain AI (defunct) | NVIDIA | AWS Neuron | Tenstorrent |
|-----------|---------|--------|------------|-------------|
| Framework support | Never documented | PyTorch/JAX/TF native | NeuronSDK (PyTorch/JAX) | tt-forge (PyTorch) |
| Compiler open-source | No | Partial (Triton) | No | Yes (tt-mlir) |
| ISA public | RISC-V base only | PTX/SASS | NeuronCore ISA (partial) | Tensix (partial) |
| Runtime public | No | CUDA Runtime | NeuronRuntime | tt-metal |
| Developer ecosystem | None | Massive | Growing | Growing |
