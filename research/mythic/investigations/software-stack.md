# Mythic AMP — Software Stack

*chip: mythic*
*as_of: 2026-08-08*
*device_class: Analog In-Memory Compute*
*primary sources: https://mythic.ai/technology/mythic-ai-workflow/, https://medium.com/mythic-ai/a-peek-into-software-engineering-at-mythic-1b0ca5522868, https://mythic.ai/technology/*

---

## Overview

Mythic's software stack converts a standard PyTorch/TensorFlow/ONNX model into a binary that programs the flash cells of the AMP chip and drives inference. The stack is called **MAPP (Mythic AI Processing Platform)** for Gen 1 products, with a full toolchain handling quantization, graph compilation, flash weight programming, and runtime execution.

The key challenge unique to analog CIM: floating-point weights must be quantized to integer (INT8) analog charge levels that can be reliably stored in NOR flash cells and recovered with sufficient precision.

---

## Software Pipeline

```
Trained Model (PyTorch / TensorFlow / ONNX)
        │
        ▼
┌──────────────────────────────────────────────┐
│       Mythic Optimization Suite              │
│  - INT8 post-training quantization           │
│  - Calibration dataset → scale/zero-point    │
│  - Per-layer analog range fitting            │
│  - Supports PyTorch, TF, ONNX natively       │
│  - TensorRT as optional front-end            │
└───────────────────┬──────────────────────────┘
                    │ Quantized model graph
                    ▼
┌──────────────────────────────────────────────┐
│       Mythic Graph Compiler                  │
│  - Layer partitioning → AMP tiles            │
│  - Weight packing: INT8 values → flash addr  │
│  - Activation routing schedule (2D NoC)      │
│  - RISC-V code generation (per-tile)         │
│  - SIMD schedule (activations, norms, pool)  │
│  - Equivalence checking (digital reference)  │
│  - Performance simulation                    │
│  - Generates: flash programming binary +     │
│    tile execution schedule                   │
└───────────────────┬──────────────────────────┘
                    │ Packaged AMP binary
                    ▼
┌──────────────────────────────────────────────┐
│       MAPP Runtime                           │
│  - Flash programming: binary → AMP chip      │
│    (PCIe DMA → ACE flash write)              │
│  - Host activation transfer (PCIe 2.0)       │
│  - Tile execution orchestration              │
│  - ADC result collection + digital assembly  │
│  - Result return to host                     │
└──────────────────────────────────────────────┘
                    │
                    ▼
              Inference result
```

---

## Stack Layers

### Framework Integration

- **PyTorch**: Native model import via ONNX export or direct TorchScript trace.
- **TensorFlow**: Supported via `SavedModel` / `TFLite` export → MAPP import.
- **ONNX**: First-class input format; most common path for edge deployment.
- **TensorRT (NVIDIA)**: Supported as optional front-end for quantization (PTQ via TensorRT calibrator) before MAPP graph compilation — useful when TensorRT's quantization tooling is preferred.
- No model code changes required for supported op types.

### Compiler / IR: Mythic Optimization Suite

- **Post-training quantization (PTQ)**: FP32 weights → INT8 analog charge levels.
- **Per-layer calibration**: Calibration dataset determines scale and zero-point per tensor.
- **Analog range fitting**: Flash cells have a specific dynamic range of conductance; quantization must match the physical range of the NOR flash array.
- **INT4 support**: Aggressive quantization for layers where accuracy permits.
- **Equivalence checker**: Compares quantized digital simulation vs. expected analog result to catch precision loss early.

### Compiler / IR: Mythic Graph Compiler

- **Layer partitioning**: Assigns each DNN layer (or sub-layer) to a set of AMP tiles; large layers span multiple tiles via 2D NoC.
- **Weight packing**: Flattens quantized weight tensors into flash cell address layout for each tile's ACE.
- **Activation routing**: Schedules inter-tile activation movement through the NoC; tile order matches DNN forward pass.
- **RISC-V code generation**: Per-tile RISC-V programs manage local execution flow, ADC reads, SIMD calls.
- **SIMD scheduling**: Non-matrix ops (ReLU, GELU, BatchNorm, Softmax, pooling, residual add) assigned to SIMD engines.
- **Performance simulation**: Cycle-accurate simulator models tile execution, NoC contention, ADC latency before hardware.
- **Output**: Packaged binary containing flash programming data + tile RISC-V executables + host driver metadata.

### Op Library

- **Analog linear ops (ACE)**: GEMM (matrix-matrix → decomposed to MVM), FC layers, Conv2D (via im2col→GEMM), Q/K/V/O attention projections — all executed in analog flash domain.
- **Digital nonlinear ops (SIMD)**: ReLU, GELU, Tanh, Sigmoid, Softmax, LayerNorm, BatchNorm, residual add, pooling — executed digitally by per-tile SIMD engine.
- **Supported model types**: CNN (ResNet, YOLOv3, EfficientDet), Pose estimation (OpenPose), Transformers (attention projected to FC layers), LLMs (2026+ roadmap).

### Kernel Library

No user-programmable kernel API is exposed. The compiler generates all RISC-V and SIMD instructions internally. Developers cannot write custom "kernels" in the GPU/TPU sense.

### Runtime: MAPP Runtime

- **Flash programming**: At model-load time, the runtime DMA-transfers the weight binary to the AMP chip via PCIe 2.0. Flash cells are programmed with INT8 analog charge levels. This is a one-time operation per model.
- **Activation transfer**: Host CPU sends input activation tensors to AMP via PCIe 2.0 per inference call.
- **Tile orchestration**: Issues execution commands to all tiles; tiles execute their RISC-V programs, performing analog MACs and routing activations through NoC.
- **ADC collection**: Reads digital results from each tile's ADC output into host memory.
- **Multi-chip orchestration**: On PCIe cards with up to 16 AMPs, runtime distributes layers across chips.

### Driver / Firmware

- **PCIe BAR management**: Maps AMP registers and flash programming interfaces.
- **Flash write controller**: Manages NOR flash cell programming (voltage ramp, verify, retry) — more complex than DRAM write.
- **Calibration firmware**: Manages temperature-drift compensation tables for flash conductance levels.
- No open-source kernel module disclosed.

### Assembler / ISA

No user-visible ISA for the analog compute path. The ACE performs analog MAC with no instruction fetch — weights are conductance states, inputs are voltages.

The embedded RISC-V per tile uses the standard RISC-V ISA (RV32I + custom extensions for SIMD). This is handled entirely by the Mythic Graph Compiler; users never write RISC-V assembly.

---

## Development Workflow (User-Facing)

```python
# Mythic MAPP SDK — simplified usage (Python)
import mapp

# 1. Load trained model (ONNX, PyTorch, TF)
model = mapp.load_model("yolov5s.onnx")

# 2. Quantize (post-training, with calibration data)
quantized = mapp.quantize(model, calibration_dataset=calib_loader,
                          dtype="int8")

# 3. Compile for target AMP chip
binary = mapp.compile(quantized, target="M1076",
                      tiles=76, batch_size=1)

# 4. Program flash and run inference
amp = mapp.AMP(device_id=0)
amp.load(binary)              # programs flash cells (one-time)
result = amp.infer(input_tensor)   # fast repeated inference
```

---

## Performance Simulation and Debugging

- **Cycle-accurate simulator**: Part of MAPP toolchain; models tile pipeline, NoC routing, ADC latency. Allows performance estimation without hardware.
- **Analog noise model**: Simulator optionally injects modeled flash noise to predict inference accuracy degradation.
- **Equivalence checker**: Runs forward pass in floating-point reference alongside quantized/analog model; flags layers with >threshold precision loss.
- **No GPU-style profiler**: No nvprof/nsight equivalent publicly documented; performance analysis relies on simulation pre-deployment.

---

## Software Stack Summary Table

| Layer | Component | Open? | Notes |
|-------|-----------|-------|-------|
| Framework Integration | PyTorch / TF / ONNX / TensorRT (optional) | Standard OSS front-ends | MAPP imports ONNX or TorchScript |
| Compiler — Quantization | Mythic Optimization Suite | Closed SDK | INT8/INT4 PTQ; analog range calibration |
| Compiler — Graph | Mythic Graph Compiler | Closed SDK | Tile partitioning; weight packing; RISC-V codegen; SIMD schedule |
| Op Library | Analog linear (ACE) + Digital nonlinear (SIMD) | Closed HW | User does not call ops directly; compiler generates |
| Kernel Library | None | — | No user kernel API; analog ops have no programmable ISA |
| Runtime | MAPP Runtime | Closed SDK | Flash programming; activation transfer; tile orchestration |
| Driver / Firmware | PCIe BAR + Flash write controller + Calibration FW | Closed | NOR flash-specific firmware complexity |
| Assembler / ISA | RISC-V (RV32I) per tile — compiler-internal | Standard ISA, compiler-managed | No user assembly; ACE has no user ISA |
| Debugger / Simulator | Cycle-accurate simulator, analog noise model, equivalence checker | Closed SDK | Essential for pre-hardware validation |

---

## Software-Stack Update — 2026-08-08 (Videantis acquisition)

*Change class: roadmap. **No change to any shipping Mythic software component.** Sources: Mythic/Videantis acquisition release 2026-05-19; Taylor Wessing advisory notice 2026-06-02; videantis.com technology page; Mythic news index; Honda newsroom 2026-02-04 / Mythic 2026-02-06.*

### What happened

Mythic acquired **Videantis GmbH** (Hannover, Germany) — announced **2026-05-19**, transaction **closed** — a licensable digital processor-IP vendor. The acquired asset is the **v-MP6000UDX** "unified processor platform", a **VLIW + SIMD array of identical cores** spanning deep-learning inference, classical computer vision, signal/image processing, and video encode/decode, marketed with a claimed **decade-hardened *unified* software stack** across all four workload classes. Terms not disclosed; Videantis continues as a wholly owned subsidiary and all five founders (led by Dr. Hans-Joachim Stolberg) joined Mythic.

### Why it matters for the stack — and how far the claim goes

The repo's standing statement is that **"no user-visible ISA or kernel API exists"** for Mythic. That statement was written about the analog ACE, and it remains **true today**. But the direction of travel has changed:

- A **v-MP6000UDX-derived digital core is a programmable processor** — licensable soft IP that, in its pre-acquisition Videantis form, shipped with its own compiler/toolchain to licensee chip makers. A hybrid Mythic part built around it would plausibly expose a digital programming interface where the analog path never could.
- **Mythic has published nothing** about a merged SDK, a unified IR, a combined ISA, a kernel language, or how MAPP would target a digital core. The v-MP6000UDX toolchain exists in the public record only as **pre-acquisition Videantis IP** — videantis.com's own technology page does not even name an SDK.
- **Therefore: hedge, do not rewrite.** The practical developer story as of 2026-08-08 is unchanged — closed MAPP SDK, Python-level entry, compiler-generated RISC-V/SIMD, no user kernel API. Do not read the acquisition as evidence that a user-visible ISA now exists.

### Layer-by-layer status after the acquisition

| Layer | Status 2026-08-08 | Note |
|---|---|---|
| Framework Integration | Unchanged (PyTorch / TF / ONNX / optional TensorRT) | No statement on whether a hybrid part keeps the same front-ends |
| Compiler / IR | Unchanged (Mythic Optimization Suite → Mythic Graph Compiler, closed) | **Not disclosed** whether/how the compiler would target a VLIW+SIMD digital core |
| Op Library | Unchanged split: analog linear (ACE) / digital nonlinear (per-tile SIMD) | Acquisition rationale names attention, non-max suppression, SLAM, and video codecs as digital-backbone work — implies an expanded digital op surface, but **no op list published** |
| Kernel Library | Still none user-facing | v-MP6000UDX is programmable; **no Mythic kernel API announced** |
| Runtime | Unchanged (MAPP Runtime, closed) | No statement on scheduling across analog + digital engines |
| Driver / Firmware | Unchanged, closed | — |
| Assembler / ISA | Analog: none, by construction. Digital: v-MP6000UDX carries a VLIW+SIMD ISA, **not published by Mythic** | Hedged, not rewritten |
| Debugger / Simulator | Unchanged | No statement on a unified simulator |

### Also relevant to the software story

- **Honda R&D licenses Mythic's Analog Processing Unit (APU) technology** (Mythic release 2026-02-06; Honda's own 2026-02-04 release describes qualitative SoC co-development only). A *technology licensing* relationship implies a third party programming Mythic analog IP, but **no licensee-facing SDK, documentation, or API has been published**.
- Terminology drift to track: Mythic's 2026 copy uses **"Analog Processing Unit (APU)"** as the umbrella term rather than "AMP"; **"Starlight"** names sensor-embedded analog co-processors. Neither has published software documentation.

### Evidence-quality caveat

Nearly all coverage of the acquisition is verbatim republication of one BusinessWire release. The only independent confirmation located is the Taylor Wessing advisory notice (HTTP 403 to direct fetch; read via search snippet). **Nothing in this section is a measured or documented software capability** — it is a corporate change with software implications that Mythic has not yet detailed.

---

## Investigation Log — 2026-08-08 Software-Stack Scan

*Investigator pass appended 2026-08-08. Verdict: **no change to any shipping Mythic software component**; one forward-looking implication recorded as a hedge.*

### Method

1. Re-checked Mythic's news index for any SDK, compiler, runtime, driver, or documentation announcement between 2025-12 and 2026-08-08. **None** — the three items in the index are a funding round, a Honda program, and an acquisition.
2. Read the full Videantis acquisition release for software content. It claims a "decade-hardened unified software stack" spanning DL inference, computer vision, signal/image processing, and video codecs — **a vendor claim with no SDK name, no compiler name, no documentation link, and no public ISA reference**.
3. Checked `videantis.com/technology.html` directly. It confirms Videantis licenses a proprietary processor platform to chip makers but **does not name v-MP6000UDX and does not name any SDK** on that page.
4. Searched for any Mythic statement about a merged toolchain, unified IR, combined ISA, kernel language, or MAPP targeting a digital core. **Nothing published.**
5. Read the Honda pair of releases for licensee-facing software detail. Honda's 2026-02-04 primary is qualitative only; Mythic's 2026-02-06 states Honda R&D **licenses APU technology** but names no SDK, API, or documentation.

### What was written, and how strongly

| Claim considered | How it was recorded |
|---|---|
| "A Videantis-derived digital core comes with its own toolchain, so Mythic now has a user-visible ISA" | **Rejected as overstated.** Written as a hedge on the standing "no user-visible ISA or kernel API exists" line, not as a rewrite. |
| Expanded digital op surface (attention, non-max suppression, SLAM, codecs) | Recorded at `confidence: medium`, `basis: vendor_stated_rationale_only`, with `published_op_list: none`. |
| Videantis software-quality credentials ("zero field defects", ">25 M chips shipped") | Recorded in the hardware YAML as **unverified vendor marketing**, never as capability. |
| Honda licensing implies a licensee-facing SDK | Recorded as an implication with `licensee_facing_sdk_or_docs: not_disclosed`. |

### Deliberate non-changes

- The MAPP pipeline description, the Python workflow example, the op-library split, and the debugger/simulator section are **untouched** — nothing about the shipping stack changed.
- No new layer was added to `layer-table.md`; the Kernel Library and Assembler/ISA rows were **annotated with a hedge** instead.

### Open items for the next scan

1. Any Mythic announcement of a **merged SDK, IR, ISA, or kernel API** for a hybrid analog+digital part.
2. Whether **MAPP** survives as the brand or is superseded alongside the "AMP" → "APU" terminology drift.
3. Whether the **v-MP6000UDX toolchain** is ever published, documented, or made available under Mythic.
4. Any **licensee-facing documentation** arising from the Honda APU licensing agreement.
5. Whether Videantis' claimed "unified software stack" is ever described concretely (compiler, IR, op coverage) rather than as a marketing adjective.
