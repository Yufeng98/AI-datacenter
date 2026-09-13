# Lightmatter — Software Stack Investigation

*as_of: 2026-08-08 (original investigation 2026-04-05; 2026-08-08 update appended at end)*
*chip: lightmatter*
*device_class: Photonic Interconnect + Optics (compute product no longer marketed) — historically Photonic Compute + Interconnect*
*sources: lightmatter.co/products/idiom, research.contrary.com/company/lightmatter, nextplatform.com 2021; 2026-08-08 sources listed in the appended section*

---

## Overview: The Idiom Platform

Lightmatter's software platform is called **Idiom**. It is a closed-source neural network compilation and execution environment that presents a "plug-and-play" interface to ML developers who already use PyTorch or TensorFlow.

The core design philosophy: **hide the photonic substrate entirely**. Developers never write photonic-specific code. They import Idiom libraries and run existing models.

---

## Software Stack Layers

### Layer 1 — Framework Integration

**PyTorch and TensorFlow integration**
- Developers import `lightmatter` (Idiom) Python libraries into existing PyTorch/TensorFlow training/inference scripts
- Idiom intercepts the computational graph at the framework level (similar to a custom dispatch backend)
- Existing model code runs unchanged for supported ops
- No PTX/ISA-level programming required (contrast: CUDA)

---

### Layer 2 — Compiler: idCompile (Graph Compiler)

**idCompile** is Idiom's graph compiler:
- Ingests PyTorch/TensorFlow computational graph (ONNX or framework-native IR)
- Identifies matmul/linear subgraphs → mapped to photonic PTCs
- Identifies nonlinear ops (activation functions, layer norm, softmax) → mapped to DCI electronics
- **Network partitioning**: automatically divides large neural networks into segments fitting across multiple Envise blades (multi-blade parallelism)
- Handles electronic-photonic data handoff scheduling
- Generates execution schedule: which ops run on PTCs, which on DCIs, data movement between them

**Key compiler challenges unique to photonics:**
- MZI weights must be encoded as phase voltages; compiler maps float weights → phase settings
- ABFP16 quantization: compiler selects shared exponents per block to minimize analog noise impact
- Analog noise compensation: phase calibration tables baked into compiled binaries
- Layer-by-layer execution model: weights loaded into MZI mesh per layer (no weight reuse across layers without reprogram)

---

### Layer 3 — Runtime

**Idiom Runtime** manages:
- Device discovery and allocation
- Execution of compiled Envise programs
- Data transfer: host DRAM → DCI DAC buffers → PTC → ADC → DCI activation → host DRAM
- Multi-blade orchestration for partitioned models
- Optical path calibration (phase drift correction)

---

### Layer 4 — Debugging and Profiling

**idBug** (debugger):
- Provides visibility into intermediate activations at electronic-photonic boundaries
- Compares photonic vs. digital reference values for precision debugging

**idProfiler** (profiler):
- Per-layer latency breakdown: PTC compute time, DAC/ADC conversion time, data transfer time
- Identifies bottlenecks in the photonic-electronic pipeline

---

## Software Stack Summary Diagram

```
User Training / Inference Script (Python)
         │
         │  import lightmatter  (Idiom)
         ↓
PyTorch / TensorFlow
         │  (graph capture)
         ↓
idCompile (Idiom Graph Compiler)
  • Partition: matmul → PTC | activations → DCI
  • ABFP16 quantization, phase-voltage encoding
  • Multi-blade partitioning for large models
  • Analog noise compensation calibration
         │
         ↓
Idiom Runtime
  • Device management, data transfer
  • Multi-blade orchestration
  • Optical calibration (phase drift)
         │
  ┌──────┴──────────────────────────┐
  ↓                                 ↓
DCI (12nm CMOS)              Envise DRAM (DDR4 1TB)
  • DAC/ADC                    • model weights
  • Activation funcs           • activations buffer
  • Layer norm, softmax
         │
         ↓
4× PTC (128×128 MZI mesh)
  • Matrix-vector multiply at speed of light
  • ABFP16 analog computation
```

---

## Comparison to Other AI Chip Software Stacks

| Aspect | Lightmatter Idiom | CUDA (NVIDIA) | NeuronSDK (AWS) |
|--------|-------------------|---------------|-----------------|
| Abstraction level | High (framework integration, no photonic primitives exposed) | Low-to-high (PTX/SASS to cuDNN) | Medium (XLA/NeuronCore API) |
| User effort | Minimal (import library) | High (kernels, memory mgmt) | Low-medium (neuron_cc compiler) |
| Kernel customization | Not possible (closed) | Yes (CUDA C++, Triton) | Limited (NKI) |
| Precision | ABFP16 (custom analog) | FP4–FP64 | BF16/FP32/cFP8 |
| Analog calibration | Built-in (transparent) | N/A | N/A |
| Multi-chip partitioning | Automatic (idCompile) | Manual (tensor parallel) | Automatic (neuron_cc) |
| Debugger | idBug | cuda-gdb | neuron-monitor |
| Profiler | idProfiler | Nsight | neuron-profile |

---

## Key Observations

1. **Closed stack**: Idiom is entirely proprietary; no open-source kernel library, no PTX equivalent, no user-programmable photonic primitives. The photonic abstraction is fully encapsulated.

2. **Analog precision management**: ABFP16 is a Lightmatter-specific format designed to maximize photonic dynamic range. The compiler handles quantization transparently, unlike NVIDIA where FP8/INT8 quantization requires explicit user or library intervention.

3. **Phase calibration**: Real photonic systems suffer from manufacturing variation and thermal phase drift in MZI elements. Idiom maintains per-chip calibration tables and corrects phase errors at compile-time and runtime — a concern absent from electronic architectures.

4. **Electronic-photonic co-scheduling**: The compiler must schedule data movement between photonic cores and digital logic carefully, since each ADC/DAC conversion adds latency. This is Idiom's core differentiation over naive graph compilers.

5. **No user-visible ISA**: There is no PTX/SASS equivalent. The lowest user-visible abstraction is the PyTorch/TensorFlow graph. Photonic operation is entirely hidden.

---

# Update — 2026-08-08

*Sources fetched or verified 2026-08-08: lightmatter.co, lightmatter.co/products/, lightmatter.co/products/passage-l20/, lightmatter.co/products/guide/, lightmatter.co/products/vclick-optics/, lightmatter.co/press-release/lightmatter-joins-nvidia-nvlink-fusion/, lightmatter.co/news/, electronicdesign.com (Guide DR press-release carry, 2026-05-21).*

## Headline: no change to Idiom, and Idiom's target device is no longer marketed

**No SDK, compiler, runtime or toolchain change was found in the Apr–Aug 2026 window.** There is no new Idiom release, no documentation refresh, no open-sourcing, no PyTorch/TensorFlow version bump and no public API. Everything in the original investigation above still describes Idiom as last documented.

The material change is contextual rather than technical: **Idiom's target device (Envise) is no longer listed in Lightmatter's product navigation** as of 2026-08-08. The `/products/` page lists Passage L200, Passage L20, Guide 1, Guide DR and four evaluation kits; Idiom and Envise appear only in the footer trademark line. There is no announced end-of-life for either. Confidence: medium — this is evidence of absence on a marketing site.

Consequences for this survey's software-stack framing:

| Stack layer | 2026-04-05 state | 2026-08-08 state |
|---|---|---|
| Framework integration (PyTorch / TensorFlow via `import lightmatter`) | Active, documented | Unchanged in content; target device no longer marketed |
| idCompile graph compiler | Active, documented | Unchanged; no new release found |
| Idiom Runtime | Active, documented | Unchanged; no new release found |
| idBug / idProfiler | Active, documented | Unchanged; no new release found |
| Envise device driver | Active, documented | Unchanged; no new release found |
| **Module management plane** | **Absent from the model** | **New: CMIS-based management for the Guide line** |

## The one genuinely new software surface: CMIS module management

The 2026 Passage/Guide products introduce Lightmatter's first **standards-based** software surface. It is a control/management plane, not a compute programming model:

**Guide 1 (VLSP Light Engine)**
- **CMIS-compliant telemetry**
- **Active stabilization** and **hyper-local thermal tuning**, maintaining sub-GHz wavelength precision without drift
- **N+M redundancy**: the engine autonomously retunes a backup laser onto a failed laser's wavelength — an autonomous firmware behavior, not a host-driven one
- Vendor describes it as "software-defined" with real-time telemetry

**Guide DR (Laser NIC)**
- **CMIS 5.3** management over **I2C / I3C**
- **OCP MHS** (Modular Hardware System) chassis integration

**Passage L20**
- **IEEE 802.3dj**-compliant electrical signaling (a link-layer standard, not a software API)

Architecturally this means a switch or host management stack talks to Guide over **stock CMIS**, the same interface used for pluggable optics, rather than through a Lightmatter-proprietary API. That is a deliberate deployability choice: no new driver is required in the management plane.

**No public SDK, driver, firmware image, API reference or open-source repository has been released for Passage or Guide.** There is nothing to investigate at the code level; the entire software surface of the current product line is vendor firmware plus a standards-defined management interface.

## Ecosystem software / EDA collaborations

Named at the Guide launch: **Cadence**, **Synopsys** and **GUC** — EDA and design-service collaborators for the VLSP photonic IC, not user-facing software. **PHIX** is a separately stated (packaging) collaboration. These are design-time tooling relationships and do not change the deployed stack.

## Correct statements of the negatives

| Item | Correct statement |
|---|---|
| Idiom updates in window | **None found.** Not "discontinued" — simply no release or documentation change observed. |
| Open-source components | **None.** The stack remains entirely closed; no PTX/Triton/kernel-language equivalent, unchanged from the baseline. |
| MLPerf | **Not applicable.** With no marketed compute product, Lightmatter has nothing submittable. Do not record as "no results." |
| Hot Chips 38 | Runs **2026-08-23/25** (Stanford Memorial Auditorium), *after* this document's as_of date. Lightmatter does not appear in the advance program. Re-check afterwards; not a settled negative. |
| Public SDK for Passage / Guide | **Not disclosed / not released.** |
