# Rain AI Layer Table

*as_of: 2026-08-08*
*device_class: Neuromorphic AI*
*architecture: Digital In-Memory Computing (D-IMC) + Andes RISC-V*
*vendor status: **DEFUNCT** — non-operating; patents assigned to OpenAI Opco, LLC, USPTO-recorded 2025-10-23*

> Historical record. Rain never shipped a commercial product and no layer below was ever released to
> developers. Table entries describe what the program had built or licensed, not what is available today.

## Software Layers

| Layer | Component | Open Source | Notes |
|-------|-----------|-------------|-------|
| Framework Integration | Implied: PyTorch/ONNX (via RISC-V generality) | No | No public connector ever released; "any model" was a marketing claim |
| Compiler / IR | Rain Proprietary Compiler | No | Full-stack; co-designed with D-IMC HW |
| Compiler / IR | Andes ACE/COPILOT | No | ISA extension framework for D-IMC dispatch |
| Op Library | Rain Proprietary Op Library | No | Not documented |
| Kernel Library | Rain D-IMC Tile Kernels | No | Not documented |
| Runtime | Rain Proprietary Runtime | No | Claimed training + inference; IP licensing was offered, never consummated publicly |
| Driver / Firmware | Rain Host Driver | No | Host interface not disclosed |
| Communication | Not documented | No | Multi-chip comms not described |
| Assembler / ISA (base) | RISC-V RV64GCV (Andes AX45MPV) | Yes (RISC-V spec) | OoO core + RVV 1.0 vector extension |
| Assembler / ISA (custom) | D-IMC Custom Instructions (ACE/COPILOT) | No | Proprietary encodings for D-IMC dispatch |

## Hardware Layers

| Layer | Component | Specification | Notes |
|-------|-----------|---------------|-------|
| Compute Engine | D-IMC Compute Tiles | Digital; MAC in SRAM; claimed train+infer | Tile dimensions, TOPS: never disclosed |
| Compute Engine | Andes AX45MPV Host Core | RV64GCV, OoO, RVV 1.0 | RISC-V host/control CPU |
| Data Path | Arteris FlexNoC 5 | Advanced mesh topology; physically aware | Connects RISC-V ↔ D-IMC ↔ memory |
| Data Path | RISC-V ↔ D-IMC Proprietary Interconnect | Balanced pipeline (Rain's description) | Not documented |
| On-chip Memory | D-IMC SRAM (compute-in-memory) | Capacity: not disclosed | SRAM = both storage and compute medium |
| Off-chip Memory | Not disclosed | Type, capacity, BW: not disclosed | — |
| Host Interface / Package | Not disclosed | PCIe/CXL/other: not disclosed | — |
| Scale-up Interconnect | Not disclosed | Not documented | — |
| Scale-out Interconnect | Not disclosed | Not documented | — |

## Gap Summary

| Gap | Impact |
|-----|--------|
| Process node never disclosed | Cannot compare transistor density or power |
| Peak TOPS/TFLOPS never disclosed | Cannot benchmark vs. competitors |
| Off-chip memory type/BW never disclosed | Cannot assess memory-bound workload performance |
| SDK/compiler never released | Zero developer adoption path; nothing survives the wind-down |
| No open-source components | No community ecosystem, and no code artifact left behind |
| **Vendor defunct (patents → OpenAI Opco, LLC, USPTO-recorded 2025-10-23)** | No roadmap; gaps are permanent. Entry retained as a historical architecture data point only. |
