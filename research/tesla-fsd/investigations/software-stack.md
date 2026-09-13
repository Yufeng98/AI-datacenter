# Tesla FSD Chip — Software Stack Investigation

## Summary

Tesla's FSD software stack is entirely proprietary. PyTorch is used for training; a custom NN compiler handles quantization, tiling, and static scheduling for deployment to the NPU. There is no public SDK.

---

## Layer Map

```
[PyTorch Training (Dojo/GPU cluster)]
        |
        v
[Tesla NN Compiler]
  - Topology mapping (coarse pass)
  - Weight pruning + INT8 quantization (fine pass)
  - Tile partitioning → NPU block assignment
  - Static schedule generation
        |
        v
[Optimized binary (per chip revision: HW3 / HW4)]
        |
        v
[OTA delivery to vehicle]
        |
        v
[Custom embedded Linux (ARM Cortex-A clusters)]
        |
        v
[NPU runtime microcode — static scheduler]
  - DMA feed to MAC array
  - Result writeback to SRAM
        |
        v
[NPU hardware: 96×96 systolic array]
```

---

## Training Stack

| Component | Detail |
|---|---|
| Framework | PyTorch |
| Training cluster | Tesla Dojo D1/D2 supercomputer |
| Data pipeline | Tesla Fleet (video from millions of vehicles) |
| Precision training | FP32 / BF16 |
| Quantization | Post-training / quantization-aware training → INT8 |

---

## Compiler

- Proprietary Tesla NN compiler (not open-sourced)
- Two-pass compilation:
  1. **Coarse pass:** maps model topology to NPU block structure
  2. **Fine pass:** weight pruning, INT8 quantization, generates static schedule
- Layer fusion: conv + scale + activation + pooling → single tiled kernel
- Output: chip-specific binary (HW3 binary ≠ HW4 binary)
- FSD v13 required forced INT8 (16-bit weights compressed to 8-bit to fit HW3)

---

## Runtime

- No dynamic OS scheduler for NPU (static only)
- ARM clusters run embedded Linux (real-time processes)
- Sensor ingestion, camera preprocessing on CPU/GPU
- NPU fed via DMA with pre-scheduled activation and weight streams
- 48 neural networks running in pipelined fashion

---

## Inference Pipeline (FSD v12+)

- **End-to-end neural network:** replaced 300K+ lines of rule-based code
- 8 cameras → Vision Engine feature extraction
- BEV (Bird's Eye View) occupancy network
- Trajectory / ego-motion estimation
- Real-time decision network
- All stages pre-compiled into static NPU schedule

---

## OTA Updates

- Models deployed as versioned OTA packages
- Vehicles download new NN binaries via cellular/WiFi
- No user access to compiler or model internals

---

## Key Software Design Decisions

1. **Closed stack**: no public SDK, no third-party models possible on NPU
2. **Static scheduling**: predictable latency, automotive safety compliance
3. **PyTorch + proprietary compiler**: leverage open training ecosystem, close the deployment gap with custom toolchain
4. **INT8 everywhere**: enables 32 MiB SRAM to hold entire model, avoids DRAM reads during inference

---

## 2026-08-08 Update — Software Stack Status

*Investigation date: 2026-08-08. Prior sections (research date 2026-04-05) remain valid.*

### Headline: no software-stack change is disclosed

The 2026 Tesla silicon disclosures (AI4.5 in January, AI4.1/AI4 Plus and the AI5 tape-out in April, the Samsung
AI5 leg in July) are **entirely hardware and business disclosures.** Across all of them Tesla disclosed:

- **no compiler** name, IR, or architecture for any new generation;
- **no runtime, driver, or firmware** detail;
- **no ISA** for AI4.5, AI4.1 or AI5;
- **no SDK, no API, no developer documentation, no sample code, no open-source release;**
- **no MLPerf submission**;
- **no conference talk** — there is **no Tesla entry in the Hot Chips 38 advance program** (2026-08-23 to
  2026-08-25; Waymo, not Tesla, holds the autonomous-driving keynote and the automotive SoC slot).

The stack described above — PyTorch training → proprietary two-pass Tesla NN compiler → static microcode
schedule → OTA chip-specific binary → embedded Linux on Arm — is still the entire public picture, and it is
still sourced from HW3-era reverse engineering plus Tesla AI Day talks, not from documentation.

**`openness: closed_proprietary` is reaffirmed without qualification.** Tesla remains, alongside a small number
of purely internal ASICs in this survey, one of the chips with **zero** third-party programmability surface.

### The few software-adjacent implications that can be recorded

These are inferences from hardware disclosures, not disclosed software facts, and are labelled as such:

| Implication | Basis | Confidence |
|---|---|---|
| The compiler must now target **at least four** binary variants (HW3, AI4, AI4.5, and eventually AI4.1/AI5), since the stack already ships "chip-specific binaries" and AI4.5 is a distinct part number in the field | Existing OTA model + AI4.5 part number 2261336-02-A | medium |
| AI5's on-die **PCIe** implies a host-attached programming model unlike the vehicle-embedded HW3/HW4 model — a driver/runtime surface that does not exist for prior FSD parts | Musk statement that AI5 integrates Arm CPU cores and PCIe blocks | low-medium (hardware statement only; no runtime disclosed) |
| An Optimus/datacenter-first AI5 would be served by Tesla-internal inference infrastructure rather than the vehicle OTA path | Musk, Q1 2026 call, on AI5 target markets | low (target market itself is unresolved) |
| AI4.1 is a memory/clock refresh with no described ISA or dtype change, so it should require **recompilation, not retargeting** | Musk's AI4.1 description | low-medium |

Nothing above should be written into the survey as a Tesla software disclosure. They are recorded so that a
later scan can tell the difference between "Tesla said this" and "this follows from what Tesla said."

### Training side

No change disclosed. PyTorch remains the training framework; Dojo remains the named training silicon line, with
**Dojo 3** mentioned as "in work" in Musk's 2026-04-15 post and nothing further public — no node, no schedule,
no architecture, and no statement about the Dojo software stack.

### Update sources

- https://electrek.co/2026/04/15/tesla-ai5-chip-taped-out-musk-ai6-dojo3/
- https://electrek.co/2026/04/23/tesla-hw4-plus-upgrade-will-hw4-follow-hw3/
- https://electrek.co/2026/01/26/tesla-quietly-starts-shipping-model-y-with-new-ai4-5-computer/
- https://www.hotchips.org/advance-program/
