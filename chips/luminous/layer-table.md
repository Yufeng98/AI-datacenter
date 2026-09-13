# Luminous Computing — Layer Table

*as_of: 2026-08-08*
*chip: luminous*
*device_class: Photonic-Electronic Hybrid*
*status: DEFUNCT — operations ceased May 2023; no software ever released; patents sold to AMD, USPTO-recorded 2026-02-28*

> **DEFUNCT VENDOR.** Every "not public" entry below is permanent: Luminous ceased operations in
> May 2023 and nothing was subsequently released or open-sourced. See `summary.md` for the dated
> status block.

## Software Layers

| Layer | Component | Status | Notes |
|-------|-----------|--------|-------|
| Framework Integration | PyTorch | Not public | "Native integration" claimed, no SDK |
| Framework Integration | TensorFlow | Not public | — |
| Framework Integration | JAX | Not public | — |
| Compiler / IR | Weight-to-phase compiler | Not public | Internal; converts W→SVD→MZI phase angles |
| Compiler / IR | Graph compiler | Not public | Internal; network partitioning |
| Op Library | Matrix-vector multiply | Not public | Electronic CMOS execution |
| Op Library | Activation functions | Not public | — |
| Kernel Library | All kernels | Not public | — |
| Runtime | Execution runtime | Not public | — |
| Runtime | Memory management | Not public | — |
| Driver / Firmware | PCIe kernel driver | Not public | — |
| Driver / Firmware | Photonic calibration firmware | Not public | MZI/ring thermal drift compensation |
| Driver / Firmware | WDM wavelength control | Not public | Modulator bias control |
| Communication | Multi-chip library | Not public | — |
| Communication | Collective ops | Not public | — |
| Assembler / ISA | Phase voltage programming | Not applicable | No conventional ISA; photonic primitives only |

## Hardware Layers

| Layer | Component | Specification | Source Quality |
|-------|-----------|---------------|----------------|
| Compute Engine | CMOS ASIC matmul engine | Not public | — |
| Compute Engine | Activation unit | Not public | — |
| Data Path | SOI waveguide network | WDM, 10–100× BW claimed | NextPlatform 2022 |
| Data Path | Electro-optic modulators | Type/rate not public | Indirect |
| Data Path | Photodetectors + TIA/ADC | Not public | — |
| On-chip Memory | SRAM / buffers | Not public | — |
| Off-chip Memory | System DRAM | 12–25× vs competitors (claimed) | Company claim; unverified |
| Host Interface / Package | PCIe (assumed) | Not disclosed | — |
| Scale-up Interconnect | Board-to-board WDM optical | 10–100× claimed; spec not public | NextPlatform 2022 |
| Scale-up Interconnect | Chip-to-chip WDM optical | 10–100× claimed; spec not public | NextPlatform 2022 |
| Scale-out Interconnect | Rack-to-rack fiber WDM | 10–100× claimed; spec not public | NextPlatform 2022 |

## IP Disposition

Two distinct channels. The Oct 2023 Enosemi arrangement was a **license only** — Luminous retained title. Ownership of the patents did not move until the outright sale to AMD recorded 2026-02-28.

| IP Component | Transaction | Recipient | Date | Downstream |
|-------------|-------------|-----------|------|------------|
| Silicon photonics design libraries | Committed commercial **license** (no title transfer) | Enosemi | 2023-10 | AMD, after AMD acquired Enosemi **2025-05-28/29** |
| Electro-optic modulator designs | License | Enosemi | 2023-10 | AMD (via Enosemi, May 2025) |
| WDM component designs | License | Enosemi | 2023-10 | AMD (via Enosemi, May 2025) |
| Process design kit (PDK) elements | License | Enosemi | 2023-10 | GlobalFoundries process; AMD (via Enosemi, May 2025) |
| Luminous patent portfolio (US11500410B1, US12379543B2, US12613811B2, …) | **Outright assignment** | Advanced Micro Devices, Inc. | **2026-02-28** (USPTO recordation) | AMD; no press coverage, USPTO records are sole evidence |
| Software / SDK / compiler source | None — never released or transferred | — | — | Nothing survived |
