# Q.ANT Photonic NPU — Software Stack Investigation

*as_of: 2026-04-05*
*chip: q-ant*
*device_class: Photonic NPU (Germany)*

---

## Overview

Q.ANT's software stack is a closed, proprietary system. There is no public GitHub repository or open-source SDK. The stack provides a relatively thin abstraction layer — the NPS (Native Processing Server) ships as a complete Linux server with pre-installed drivers and libraries, and users access it via C/C++/Python APIs that target the Q.PAL algorithm library. Framework integration (PyTorch, TensorFlow, Keras) is supported via operator dispatch bridges.

The stack has no analog to CUDA's PTX ISA, CUTLASS kernel library, or open kernel module — Q.ANT's photonic chips are programmed at a much higher level of abstraction (problem domain → algorithm library), reflecting both the photonic hardware model and the early commercial stage of the product.

---

## Stack Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Framework Integration                                       │
│  PyTorch · TensorFlow · Keras (operator dispatch bridge)    │
├─────────────────────────────────────────────────────────────┤
│  High-Level API                                              │
│  Python API  ·  C/C++ API                                   │
├─────────────────────────────────────────────────────────────┤
│  Q.PAL — Q.ANT Photonic Algorithm Library                   │
│  Nonlinear model primitives · Physics simulation ops ·      │
│  Computer vision primitives · Sensor fusion ops             │
├─────────────────────────────────────────────────────────────┤
│  Runtime / HAL                                              │
│  NPS Device Driver (Linux) · PCIe DMA engine               │
│  DAC controller · Photodetector ADC reader                  │
│  MZI phase programming interface                            │
├─────────────────────────────────────────────────────────────┤
│  Hardware — Q.ANT NPU 2 (TFLN Photonic Chip)               │
│  MZI mesh (optical MVM) · Native TFLN nonlinear ops        │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Hardware Abstraction / Driver

| Component | Description |
|-----------|-------------|
| OS | Linux (pre-installed on NPS server) |
| Driver type | Proprietary kernel driver (not open-source; no upstream analog) |
| PCIe interface | Standard PCIe for host↔NPU data transfer |
| DAC control | Driver manages voltage sequencing for MZI phase programming (weight loading) |
| Photodetector readout | ADC readout via driver; results forwarded to host memory |
| Programming model | No ISA / no kernel launch model; weights are voltage configurations, not code |
| Analogy | Closer to an FPGA driver (configure then run) than a GPU kernel launcher |

**Key difference from GPU**: There is no "kernel" to write. A "program" for Q.ANT is a voltage configuration table (MZI weights) + an input data stream → the photons do the rest.

---

## Layer 2: Q.PAL — Q.ANT Photonic Algorithm Library

Q.PAL is Q.ANT's proprietary algorithm library, analogous in function to cuDNN or MKL-DNN but tailored to photonic nonlinear computation.

| Property | Description |
|----------|-------------|
| Full name | Q.ANT Photonic Algorithm Library |
| Open source | No |
| Language | C++ (native) with Python bindings |
| Purpose | Provides nonlinear network models and algorithm primitives optimized for the TFLN photonic hardware |
| Key capability (NPU 2) | Nonlinear model primitives that exploit native TFLN optical nonlinearity — reduces parameter counts, training depth while improving accuracy |
| Target domains | Image learning & classification; physics-based simulation; robotics & physical AI; sensor fusion; computer vision; scientific discovery |
| Excluded domains | Transformer / LLM attention (NOT a target use case for Q.PAL) |
| Versioning | Not publicly disclosed; evolves with NPU generations |

### Q.PAL Algorithm Categories (from public descriptions)

| Category | Representative Primitives |
|----------|--------------------------|
| Nonlinear network layers | Custom photonic nonlinear activations (native TFLN χ²) |
| Image processing | Convolutional kernels, classification, feature detection |
| Physics AI | Simulation operators (PDE solvers, physics-constrained nets) |
| Scientific computation | Matrix-vector multiply for numerical methods |
| Sensor fusion | Multi-modal data integration operators |

---

## Layer 3: C/C++ and Python APIs

| Component | Description |
|-----------|-------------|
| Languages | C++, C, Python |
| API style | Imperative procedural; not a graph-compilation model |
| Python bindings | Yes; shipped with NPS Linux environment |
| PCIe integration | Exposed via API for data transfer to/from NPU |
| Model loading | API accepts model weights (as phase configurations) and input tensors |
| Documentation | Not publicly available; provided to NPS customers |
| Analogy | Resembles a vendor-specific numpy extension or a thin inference SDK (closer to Qualcomm's libQAic than to the CUDA ecosystem) |

---

## Layer 4: ML Framework Integration

| Framework | Integration Mechanism | Status |
|-----------|----------------------|--------|
| PyTorch | Operator dispatch bridge (C++ extension, custom backend) | Supported |
| TensorFlow | Custom op / plugin | Supported |
| Keras | Backend bridge | Supported |
| ONNX | Not explicitly confirmed | Unknown |
| JAX | Not confirmed | Unknown |
| Custom models | Via Q.PAL / C++/Python API directly | Primary developer interface |

Framework integration is described as **plug-and-play**: existing PyTorch/TF models targeting Q.ANT workload classes can be dispatched to the NPU without rewriting model code.

**Important caveat**: Q.ANT's photonic NPU is **not** a general-purpose GPU replacement. Only workloads in Q.ANT's target domain (nonlinear AI, physics simulation, etc.) benefit from the photonic acceleration. LLM/Transformer inference is not supported as a primary use case.

---

## Layer 5: Compiler / IR

| Component | Status |
|-----------|--------|
| Compiler | Proprietary (details not public) |
| IR / bytecode | Not disclosed |
| Open source | No |
| User-facing compiler | Not exposed to end users; bundled in NPS server stack |
| Kernel authoring | Not applicable — no user-programmable ISA or kernel model |

The Q.ANT NPU has no user-visible ISA or programmable shader/kernel model. End users cannot write custom "photonic kernels." The hardware is configured by Q.PAL primitives which map through the driver to MZI voltage tables.

---

## Layer 6: Deployment and Workflow

### Typical Inference Workflow

```
1. User model (PyTorch / custom Python)
        ↓
2. Q.PAL / framework bridge identifies target ops
        ↓
3. Compiler (internal) maps ops → MZI voltage configuration tables
        ↓
4. Runtime loads weights via PCIe → DAC → MZI phase voltages
        ↓
5. Input data streamed via PCIe → optical modulators (input encoding)
        ↓
6. Photons propagate through MZI mesh → optical MVM + nonlinear ops
        ↓
7. Photodetectors → ADC → results via PCIe → host CPU memory
        ↓
8. Output tensor returned to user model
```

### HPC Integration

The NPS server integrates into HPC environments as a standard Linux compute node. Standard HPC job schedulers (SLURM, PBS) can dispatch workloads to NPS nodes. Integration with LRZ and JSC uses standard HPC infrastructure (InfiniBand or HDR for inter-node; PCIe for intra-node host↔NPU).

---

## Software Stack Summary Table

| Layer | Component | Open Source | Analogy |
|-------|-----------|-------------|---------|
| Framework | PyTorch / TF / Keras bridge | No | torch.cuda device plugin |
| High-level API | Python API | No | — |
| Algorithm library | Q.PAL (Photonic Algorithm Library) | No | cuDNN (narrow domain) |
| Runtime | Proprietary HAL + PCIe DMA engine | No | CUDA Runtime (libcudart) |
| Driver | Linux kernel driver (PCIe) | No | nvidia.ko |
| Compiler | Proprietary (not user-facing) | No | nvcc (internal only) |
| ISA / kernel model | None (no user ISA) | N/A | N/A |
| Hardware | TFLN MZI mesh (LENA arch) | N/A | — |

---

## Key Differentiators vs GPU Software Stack

| Aspect | Q.ANT NPU | NVIDIA GPU |
|--------|-----------|------------|
| Programmability | Algorithm library only (no ISA) | Full CUDA kernel authoring (PTX/SASS/Triton) |
| Precision model | Analog continuous (no digital dtypes) | FP64/FP32/BF16/FP16/FP8/INT8/FP4 |
| Memory programming | Voltage table (MZI config) | SRAM tiling, memory access patterns |
| Kernel model | None (configuration + data stream) | SIMT warp execution |
| Open source | Fully proprietary | Partially open (CUTLASS, NCCL, open kernel module) |
| Framework support | PyTorch / TF / Keras | All major frameworks |
| Target use case | Narrow: nonlinear AI, physics sim | Broad: all AI/HPC workloads |
| Power efficiency | ~30 W chip | 700–1,000 W chip (Hopper/Blackwell) |

---

## Sources

- https://qant.com/photonic-computing/
- https://qant.com/press-releases/q-ant-unveils-its-second-generation-photonic-processor-to-power-the-next-wave-of-ai-and-hpc/
- https://thequantuminsider.com/2025/11/19/qant-next-gen-photonic-npu/
- https://www.allaboutcircuits.com/news/q.ants-new-photonic-processor-pushes-ai-and-hpc-beyond-silicons-limits/
- https://www.hpcwire.com/2025/06/18/q-ant-photonic-computing-shines-at-isc-2025/
- https://quantumzeitgeist.com/qant-npu-ai-photonic-processing/
- https://qant.com/wp-content/uploads/2025/11/20251111_QANT-Photonic-AI-Accelerator-Gen-2.pdf
- https://insidehpc.com/2025/07/q-ant-photonic-ai-processor-goes-into-operation-at-leibniz-supercomputing-centre/

---

# Investigation Update — 2026-08-08

*Window: 2026-04-05 → 2026-08-08. For a software-stack survey this is the **material** change of the window: the compiler layer is no longer "internal only".*

## Headline

**Q.ANT released no SDK, no toolchain, and no documentation in this window, and its own stack remains fully proprietary with no public GitHub.** What changed is that a **third party** — Daisytuner GmbH — demonstrated a working **PyTorch → photonic compilation path** onto Q.ANT NPU Gen 2. Layers 1–6 above are otherwise unchanged.

## Layer 5 (Compiler / IR) — revised

The prior entry read `Compiler | Proprietary (details not public) | User-facing compiler: not exposed to end users`. That is now **incomplete rather than wrong**: Q.ANT's internal compiler is still proprietary and non-user-facing, but a **second, third-party compilation flow now exists alongside it.**

### Daisytuner PyTorch → photonic path (revealed April 2026)

| Property | Detail |
|---|---|
| Owner | **Daisytuner GmbH — a third party, not Q.ANT** |
| Graph capture | **Daisyflow** |
| Compiler | **`docc`** — Daisytuner Optimizing Compiler Collection, **SDFG-based** |
| Model compiled | **Faster R-CNN with a ResNet-50 backbone** (object detection) |
| Source framework | PyTorch, compiled **directly**; "no custom code" |
| Coverage | Pre-processing, inference and post-processing running **end-to-end** |
| Target | Q.ANT **NPU Gen 2** |
| Q.ANT's framing | "the first time an AI model from a standard ML framework has been successfully compiled for photonic hardware" |
| Reveal date | **April 2026** per Q.ANT's 2026-06-23 press release; the exact date could not be pinned (daisytuner.com/news is client-rendered and undated) |

### Caveats that must travel with this item

1. **The compiler is Daisytuner's, not Q.ANT's.** This does not make Q.ANT's stack more open. Q.ANT still has no public GitHub and no published SDK documentation.
2. **`github.com/daisytuner/docc` is public and actively developed** (last push 2026-08-07, ~22 stars as of that date), corroborating that Daisytuner ships a real compiler product — but **a Q.ANT/photonic backend was not located in the open-source tree.** The photonic target may be delivered through Daisytuner's hosted service rather than the open repo. Do not describe the Q.ANT backend as open source.
3. **Do NOT attribute this work to SC'25.** The "SC '25" tag on daisytuner.com belongs to a separate OpenFOAM/Tenstorrent item.
4. **This is not the ISC 2026 demo.** Q.ANT's ISC 2026 (2026-06-23) demos were a diffusion image-to-image model and TiRex (xLSTM, NXAI) — see `hw-architecture.md`. The Daisytuner item is an object-detection pipeline. Merging the two events is a factual error.

## Layer 4 (Framework Integration) — revised

The framework relationship is **no longer only an operator-dispatch bridge / Q.PAL library call**. Two distinct mechanisms now exist:

| Mechanism | Owner | Nature |
|---|---|---|
| Operator dispatch bridge (PyTorch / TensorFlow / Keras) | Q.ANT | Dispatch of supported ops into Q.PAL; imperative, not a compilation flow |
| Daisyflow graph capture → `docc` compilation | Daisytuner (third party) | A genuine **compilation** flow from a standard framework onto the photonic target |

ONNX and JAX support remain **not confirmed**.

## Revised software-stack summary table

| Layer | Component | Open Source | Notes |
|-------|-----------|-------------|-------|
| Framework | PyTorch / TF / Keras bridge (Q.ANT) | No | Operator dispatch |
| Framework (3rd party) | Daisyflow PyTorch graph capture | Compiler repo public; photonic backend not located in it | Daisytuner GmbH |
| High-level API | Python / C / C++ API | No | — |
| Algorithm library | Q.PAL | No | Narrow-domain cuDNN analog |
| Compiler (Q.ANT) | Proprietary, internal, maps Q.PAL ops → MZI voltage tables | No | Not user-facing |
| Compiler (3rd party) | Daisytuner `docc` (SDFG-based) → NPU Gen 2 | `docc` repo public; Q.ANT backend not located in open tree | Revealed Apr 2026 |
| Runtime | Proprietary HAL + PCIe DMA engine | No | — |
| Driver | Linux kernel driver (PCIe) | No | — |
| ISA / kernel model | None | N/A | No user-programmable ISA |
| Hardware | TFLN MZI mesh (LENA arch) | N/A | — |

## Unchanged

Q.PAL, the C/C++/Python API, the runtime/HAL, the Linux kernel driver, the absence of a user-visible ISA, and the deployment workflow are all unchanged. No Q.ANT SDK version, release note, or public documentation appeared in the window. Q.ANT's open-source footprint remains **none**.

## Sources added 2026-08-08

- https://daisytuner.com/ — Daisytuner's own product page: "Faster R-CNN with ResNet-50 backbone compiled directly from PyTorch to photonic hardware" on "Q.ANT NPU Gen 2", end-to-end pipeline, no custom code. Also shows the "SC '25" tag belongs to the OpenFOAM/Tenstorrent item, not the Q.ANT item.
- https://github.com/daisytuner/docc — public Daisytuner Optimizing Compiler Collection repo (last push 2026-08-07); no Q.ANT/photonic backend located in the public tree.
- https://qant.com/press-releases/q-ant-runs-generative-ai-on-photonic-hardware/ — Q.ANT primary release (2026-06-23) dating the Daisytuner compiler reveal to April 2026 and describing it as a first for standard-framework compilation onto photonic hardware.
- https://thequantuminsider.com/2026/06/23/q-ant-runs-generative-ai-on-photonic-hardware/ — independent coverage of the ISC 2026 demos (distinct event).
