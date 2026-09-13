# Luminous Computing — Software Stack

*as_of: 2026-08-08*
*chip: luminous*
*device_class: Photonic-Electronic Hybrid*
*status: DEFUNCT — operations ceased May 2023; no software stack was ever released, and none survived the wind-down*

---

## Summary

Luminous Computing wound down its hardware program in May 2023 before releasing any public software stack, and the company is defunct as of this writing. It never published a compiler, runtime, SDK, or driver for its photonic-electronic hybrid architecture. All software stack layers below are marked **not public**; these are permanent gaps, not pending disclosures.

The partial exception is a claim of native PyTorch integration in a "Gen 1 AI inference card" product description (from LinkedIn / company website circa 2023), but no SDK, open-source code, or documentation was ever released. Neither the 2023 Enosemi license nor the 2026 patent sale to AMD included software.

---

## Layer-by-Layer Status

### Framework Integration

| Component | Status | Notes |
|-----------|--------|-------|
| PyTorch | Not public | "Native PyTorch integration" claimed for Gen 1 inference card; no SDK released |
| TensorFlow | Not public | Not mentioned |
| JAX | Not public | Not mentioned |

**Evidence:** Company LinkedIn and website (circa 2023) referenced "Gen 1 AI inference cards" with "native PyTorch integration" and "12–25× memory vs competing products at HBM-level bandwidths." No public code or documentation.

---

### Compiler / IR

| Component | Status | Notes |
|-----------|--------|-------|
| Weight-to-phase compiler | Not public | Required to convert floating-point weight matrices to MZI phase angles (SVD decomposition) — internal development only |
| Graph compiler | Not public | Network partitioning across multiple photonic chips — internal |
| MLIR / LLVM backend | Not public | Likely required but never disclosed |

**Note:** For photonic MVM compute (pre-pivot architecture), the compiler must perform W → SVD(W) → {θᵢ} (phase angles) for each layer, analogous to Lightmatter's idCompile. For the interconnect-focused post-pivot architecture, a conventional CMOS ASIC compiler would suffice.

---

### Op Library

| Component | Status | Notes |
|-----------|--------|-------|
| Matrix-vector multiply | Not public | Would execute on CMOS compute die (electronic) |
| Activation functions | Not public | |
| Attention | Not public | |

---

### Kernel Library

Not public. No kernel library was publicly released.

---

### Runtime

| Component | Status | Notes |
|-----------|--------|-------|
| Execution runtime | Not public | |
| Memory management | Not public | |
| Multi-chip coordination | Not public | |

---

### Driver / Firmware

| Component | Status | Notes |
|-----------|--------|-------|
| Kernel driver | Not public | PCIe driver for CMOS compute die — never released |
| Photonic calibration firmware | Not public | Thermal drift compensation for MZI/ring modulators — critical for photonic systems; details internal |
| WDM control firmware | Not public | Wavelength locking and modulator bias control |

---

### Communication

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-chip communication library | Not public | |
| Collective operations (AllReduce, etc.) | Not public | |

---

### Assembler / ISA

Photonic hardware has no conventional ISA. The fundamental "instruction" would be setting phase voltages on electro-optic modulators (for photonic compute) or modulator bias currents (for photonic interconnect). No public specification exists.

---

## What Was Productized

The only software artifact with any public evidence is:

- **Gen 1 AI inference card product** (2023): Referenced "native PyTorch integration" and large memory capacity (12–25× vs competitors at HBM-level bandwidths). No public SDK, no GitHub repos, no documentation pages found.

---

## IP Transfer (no software involved)

The silicon photonics **design IP** (not software stack) was *licensed* to Enosemi in October 2023 — a license only, with Luminous retaining title. AMD acquired Enosemi on **2025-05-28/29**. Separately, Luminous assigned its own patent portfolio outright to AMD with USPTO recordation date **2026-02-28**.

Neither transaction included a software stack, compiler, or runtime — there was none to transfer. AMD's CPO software integration is separate from anything Luminous built and is **not disclosed** as of 2026-08-08.

---

## References

- [VentureBeat Series A (2022)](https://venturebeat.com/ai/luminous-computing-which-is-developing-a-light-based-ai-accelerator-chip-raises-105m)
- [Enosemi launch with committed commercial license to Luminous IP (Oct 2023)](https://www.enosemi.com/news/2023-10-12_enosemi_luminous_press_release/)
- [Electronics Weekly: AMD buys Enosemi (May 2025)](https://www.electronicsweekly.com/news/business/amd-buys-enosemi-2025-05/)
- [US11500410B1 — Luminous → AMD assignment recorded 2026-02-28](https://patents.google.com/patent/US11500410B1/en)
- [Luminous Computing LinkedIn](https://www.linkedin.com/company/luminous-computing)
- [TechCrunch: Photonics tough nut to crack (Apr 2023)](https://techcrunch.com/2023/04/21/light-powered-ai-chips-future/)
