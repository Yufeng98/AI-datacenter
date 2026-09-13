# Iluvatar CoreX (天数智芯) Software Stack Investigation

*as_of: 2026-04-05*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*
*resource: software-stack*

---

## Summary

Iluvatar CoreX (天数智芯) has built a full-stack GPU software platform — from instruction set and drivers up to framework backends — branded as **IXUCA** (Iluvatar CoreX Unified Computing Architecture). The stack is explicitly designed to maintain high compatibility with mainstream GPU programming ecosystems (CUDA, OpenCL, ROCm) so customers can migrate existing AI workloads with minimal code changes.

The open-source portion of the stack is published via the **DeepSpark** community platform (`github.com/Deep-Spark` / `deepspark.org.cn`). Key components include the **IxRT** inference runtime, **ix-container-toolkit** for containerized GPU deployment, **ixGDB** GPU debugger, and model benchmark suites. The closed proprietary layer includes the IXUCA compute runtime, ixnn (cuDNN analog), ixblas (cuBLAS analog), and the GPU kernel driver.

---

## Stack Overview

```
Framework Integration
  └── PyTorch (via IXUCA device backend / torch extension)
  └── TensorFlow (via IXUCA compatibility layer)
  └── PaddlePaddle (support via IXUCA)
  └── MindSpore (support via IXUCA)
  └── FlagPerf benchmark framework (BAAI / 智源 benchmark)

Compiler / IR
  └── IXUCA Compiler (kernel compilation pipeline; undisclosed internals)
  └── IxRT Compiler (graph optimizer for inference, TensorRT analog)

Op Library
  └── ixnn (cuDNN analog — Conv, Attention, Norm, Pooling)
  └── ixblas (cuBLAS analog — GEMM, BLAS L1-L3)

Inference Runtime
  └── IxRT (Iluvatar Inference Runtime — TensorRT analog)
  └── IGIE (Iluvatar GPU Inference Engine)

Runtime
  └── IXUCA Runtime (ix-runtime / libcuda shim — cudart analog)
  └── IXUCA Driver API (lower-level context and module management)

Driver / Firmware
  └── Iluvatar GPU Kernel Driver (.ko — Linux PCIe / IOCTL / DMA)
  └── ix-container-toolkit (Docker/containerd GPU container runtime)
  └── ix-device-plugin (Kubernetes DaemonSet for GPU resource exposure)
  └── ix-exporter (Prometheus metrics exporter for Iluvatar GPUs)

Communication
  └── CLIF fabric driver (intra-node GPU-to-GPU; undisclosed details)
  └── Standard NCCL-compatible CCL (inter-node over IB/Ethernet)

Debugging / Profiling
  └── ixGDB (GPU debugger; CUDA-GDB 10.2 derived; open-source)
  └── ixSMI (GPU status monitor; nvidia-smi analog)
  └── ixPROF (profiler)
  └── ixKN (kernel trace utility)
  └── ixSYS (system diagnostics)

Assembler / ISA
  └── Proprietary ISA (full in-house design; not publicly documented)
```

---

## Layer Details

### Framework Integration

**PyTorch**
- Iluvatar CoreX GPUs are usable from PyTorch via an IXUCA device extension
- The IXUCA runtime exposes a `libcuda`-compatible interface, allowing PyTorch's CUDA backend to dispatch to Iluvatar hardware
- The ix-container-toolkit reports `CUDA Version: N/A` with IXUCA version info, showing it uses a CUDA-compatible shim rather than native CUDA
- Supported frameworks: PyTorch, TensorFlow, PaddlePaddle, MindSpore

**FlagPerf**
- BAAI (Beijing Academy of AI / 智源研究院) FlagPerf benchmark framework runs on Iluvatar CoreX BI-V100/BI-V150 clusters
- Used for ResNet50, LLM training benchmarks on TianGai hardware
- Validates Aquila2-70B mixed-cluster (BI-V100 + BI-V150) training at 85.3% of theoretical peak

---

### Compiler / IR

**IXUCA Compiler**
- In-house kernel compilation toolchain (architecture internals not publicly disclosed)
- Full-stack in-house development claim: instruction set → chip architecture → foundational software stack
- Compile target: proprietary Iluvatar GPU ISA

**IxRT Compiler (Inference Graph Optimizer)**
- Part of the IxRT inference runtime
- Performs graph-level optimization for deployment: layer fusion, quantization, precision calibration
- TensorRT analog for Iluvatar hardware
- Open-source components: plugins, deploy tools, sample applications (`github.com/Deep-Spark/iluvatar-corex-ixrt`)

---

### Op Library

**ixnn (cuDNN analog)**
- Convolution, Attention (SDPA), Normalization, Pooling, Activation
- Integrated with IXUCA runtime and IxRT
- Supports FP32 / FP16 / BF16 / INT8 precision

**ixblas (cuBLAS analog)**
- GEMM (General Matrix Multiply), BLAS L1/L2/L3
- Optimized for Iluvatar GPU tensor compute units
- Used by framework backends for linear algebra dispatch

---

### Inference Runtime

**IxRT (Iluvatar Inference Runtime)**
- Open-source inference engine (`github.com/Deep-Spark/iluvatar-corex-ixrt`)
- High-performance AI inference for production deployment
- Components:
  - AI compiler (graph optimization)
  - Inference runtime (execution engine)
  - Development APIs (C++ and Python)
  - Plugin system for custom operators
  - Deploy tools (model conversion, calibration)
  - Sample applications
- Analogous to NVIDIA TensorRT + TensorRT plugins

**IGIE (Iluvatar GPU Inference Engine)**
- Alternative inference stack referenced alongside IxRT in IXUCA documentation
- Used together with IxRT within the IXUCA inference tier
- Provides additional framework interoperability (especially for TensorFlow and PaddlePaddle graphs)

---

### Runtime

**IXUCA Runtime (ix-runtime / libcuda shim)**
- cudart analog for Iluvatar GPUs
- Exposes CUDA-compatible API (`libcuda.so` naming) with IXUCA implementation
- Enables CUDA-written application code to run on Iluvatar hardware with minimal modification
- Memory management: `ixMalloc` / `ixFree` / `ixMemcpy` (or libcuda-shim equivalents)
- Stream/event management for asynchronous execution
- IXUCA version: displayed in container toolkit instead of CUDA version

**IXUCA Driver API**
- Lower-level context, module, and device management
- Fine-grained control for advanced applications

---

### Driver / Firmware

**Iluvatar GPU Kernel Driver (.ko)**
- Linux PCIe kernel module
- BAR mapping, IOCTL dispatch, DMA engines, interrupt handling
- Supports x86 Ubuntu and domestic Linux distributions (Kylin, UOS)

**ix-container-toolkit** (`github.com/Deep-Spark/ix-container-toolkit`)
- Container runtime integration: Docker, containerd, podman
- Automatically configures containers to access Iluvatar GPU resources
- Includes `ix-container-runtime` hook (analogous to nvidia-container-runtime)
- Open-source; Apache 2.0 license

**ix-device-plugin**
- Kubernetes DaemonSet
- Exposes Iluvatar GPU resources to the Kubernetes scheduler
- GPU health monitoring and multi-tenant resource allocation

**ix-exporter** (`github.com/Deep-Spark/ix-exporter`)
- HTTP server exposing Iluvatar GPU node metrics in Prometheus format
- Metrics: utilization, memory, temperature, frequency, power

---

### Communication

**CLIF (Cluster Interconnect Fabric)**
- Proprietary GPU-to-GPU interconnect driver for intra-node scaling (NVLink analog)
- Bandwidth not publicly disclosed
- Used in 8-GPU server configurations

**Collective Communication**
- Inter-node: standard NCCL-compatible protocol over InfiniBand or RoCE Ethernet via host NIC
- No public CCL library source disclosed (unlike NCCL / MCCL / BCCL)

---

### Debugging and Profiling

**ixGDB** (`github.com/Deep-Spark/ixGDB`)
- Open-source GPU debugger for Iluvatar GPUs
- Derived from CUDA-GDB 10.2 (NVIDIA's GPU extension to GNU GDB)
- Source-level GPU kernel debugging
- Licensed under GPLv3 (inherits GDB license)

**ixSMI**
- GPU status monitor (nvidia-smi analog)
- Shows utilization, memory, temperature, power, ECC status

**ixPROF**
- GPU performance profiler (nsight-systems analog)
- Kernel-level timing and occupancy analysis

**ixKN**
- Kernel name and trace utility

**ixSYS**
- System diagnostics and health check tool

---

### Assembler / ISA

**Iluvatar Proprietary SIMT ISA**
- Fully in-house instruction set architecture
- Not publicly documented
- SIMT execution model with warp/wavefront parallelism
- Supports FP32 / FP16 / BF16 / INT8 on TianGai generations
- All GPU kernel development through IXUCA compiler toolchain (not a public ISA like PTX)

---

## CUDA Compatibility Summary

| CUDA Component | IXUCA Equivalent | Compatibility Method |
|---------------|------------------|---------------------|
| nvcc | IXUCA Compiler | Source recompile |
| CUDA Runtime | IXUCA Runtime / libcuda shim | API-compatible shim |
| CUDA Driver | IXUCA Driver API | API-compatible layer |
| cuDNN | ixnn | Source-level drop-in |
| cuBLAS | ixblas | Source-level drop-in |
| NCCL | CLIF + CCL | Partial (intra-node proprietary, inter-node NCCL-compat) |
| TensorRT | IxRT + IGIE | Graph-level compatible |
| nvidia-smi | ixSMI | Functionality parity |
| nvidia-container-toolkit | ix-container-toolkit | Full Container Runtime API parity |
| CUDA-GDB | ixGDB | GDB extension parity |
| nsight | ixPROF | Partial profiling parity |
| K8s device plugin | ix-device-plugin | Full Kubernetes resource API |

---

## DeepSpark Open-Source Platform

DeepSpark (`deepspark.org.cn`, `github.com/Deep-Spark`) is Iluvatar CoreX's open-source community and AI benchmark platform:

- Hundreds of open-source application algorithms and models (vision, NLP, recommendation, etc.)
- Deeply coupled with industrial AI applications
- Provides multi-dimensional evaluation benchmarks calibrated to industry needs
- Hosts all open-source Iluvatar tooling (IxRT, ix-container-toolkit, ixGDB, ix-exporter, etc.)
- Model Zoo with reference implementations for Iluvatar GPU hardware

---

## Ecosystem Maturity

| Dimension | Status |
|-----------|--------|
| Framework support | PyTorch, TensorFlow, PaddlePaddle, MindSpore |
| LLM inference | Supported via IxRT/IGIE (exact model list not fully disclosed) |
| Container / K8s | Full (ix-container-toolkit, ix-device-plugin, ix-exporter) |
| Debugging | Full (ixGDB, ixSMI, ixPROF, ixKN, ixSYS) |
| CUDA migration tooling | Basic (libcuda shim; no CUDA source translator like Musify) |
| Open-source depth | Moderate (IxRT, containers, ixGDB open; compiler/runtime proprietary) |
| Community | DeepSpark platform; BAAI benchmark integration |
| Benchmarked | BAAI FlagPerf ResNet50 + Aquila2-70B 70B training |
| Production deployment | 52,000+ GPUs, 290+ customers, finance/healthcare/transport |

---

## Sources

- [Iluvatar CoreX Wikipedia](https://en.wikipedia.org/wiki/Iluvatar_CoreX)
- [GitHub — Deep-Spark/iluvatar-corex-ixrt](https://github.com/Deep-Spark/iluvatar-corex-ixrt)
- [GitHub — Deep-Spark/ix-container-toolkit](https://github.com/Deep-Spark/ix-container-toolkit)
- [GitHub — Deep-Spark/ix-exporter](https://github.com/Deep-Spark/ix-exporter)
- [GitHub — Deep-Spark/ixGDB](https://github.com/Deep-Spark/ixGDB)
- [GitHub — Deep-Spark/DeepSpark](https://github.com/Deep-Spark/DeepSpark)
- [HAMi — Enable Iluvatar GPU sharing (MR-V100/BI-V150)](https://project-hami.io/docs/userguide/iluvatar-device/enable-illuvatar-gpu-sharing)
- [RiseUnion — Iluvatar GPU Virtualization Guide MR-V100/BI-V150](https://www.theriseunion.com/blog/HAMi-iluvatar-support.html)
- [DeepSpark — IxRT project introduction](https://www.deepspark.org.cn/introduce?code=IxRTxmjs&topicId=711&fullCode=xmjs)
- [Digitimes — Iluvatar CoreX first commercial GPGPU China](https://www.digitimes.com/news/a20251222VL203/gpgpu-commercial-gpu-software-production-china.html)
- [BAAI — Aquila2-70B mixed cluster with Iluvatar](https://www.elecfans.com/d/2328248.html)
- [BAAI FlagPerf — TianGai-100 ResNet50 benchmark](https://hub.baai.ac.cn/view/29875)

---

# Investigation Update — FlyDSL, Developer Portal, and Deep-Spark Activity

*investigated: 2026-08-08*
*scope: software-stack changes in the window Apr 2026 – Aug 2026*
*confidence: medium — these findings come from the raw scan of Deep-Spark repository activity and the WAIC 2026 briefing coverage. The adversarial verification pass focused on the TianGai 300 hardware claims and neither corroborated nor contradicted the software items; they are recorded as `reported` rather than `confirmed`.*

## FlyDSL — a new kernel-authoring DSL layer

**`github.com/Deep-Spark/FlyDSL`** (last updated **2026-08-08**) is the single most significant software-stack change. It is described as *"a Python DSL and MLIR stack for authoring high-performance GPU kernels with explicit layouts and tiling."*

| Attribute | Detail |
|---|---|
| Repository | `github.com/Deep-Spark/FlyDSL`, `iluvatar` branch |
| Commits on `iluvatar` branch | 873 |
| Provenance | Fork of an **AMD-ROCm-origin project** |
| Iluvatar-specific addition | **FlyIXDL backend** targeting Iluvatar GPUs |
| Hardware coverage | `ivcore11`-class parts: **MR-50, MR-100, BI-V150, BI-V150s** |
| Design lineage (cited) | **CUTLASS layout algebra** + **ROCm Composable Kernel tile patterns** |
| Abstraction level | Explicitly **lower-level than Triton** — the programmer states layouts and tiles rather than delegating to an autotuner |

### Why this matters for the survey

The survey's layer table previously recorded **no kernel-authoring DSL** for Iluvatar. The vendor's kernel story was: proprietary IXUCA compiler at the bottom, closed ixnn/ixblas libraries above it, and nothing in between that a third-party engineer could use to write a fused kernel. FlyDSL fills that gap with a CUTLASS/Triton-tier layer, and it is open-source.

Two structural observations:

1. **The ROCm lineage is notable.** Iluvatar's runtime compatibility story has always been CUDA-shaped (`libcuda` shim, `ixMalloc`, ixGDB from CUDA-GDB 10.2). FlyDSL's provenance is the *AMD* ecosystem — CUTLASS layout algebra for the type system and ROCm Composable Kernel for the tile patterns. This is the first significant piece of the Iluvatar stack whose upstream is not NVIDIA.
2. **The hardware coverage list is a disclosure in itself.** `ivcore11` grouping MR-50, MR-100, BI-V150 and BI-V150s is the clearest public statement of which Iluvatar parts share a core generation. Note that **TianGai 300 is not in that list** — consistent with it being a new architecture generation with no public toolchain support yet.

## Developer portal

Iluvatar launched **`developer.iluvatar.com`** around **2026-07-15/19**, aggregating developer documentation, software images, model cases, technical blogs and DeepSpark resources into a single entry point. At the WAIC briefing the company stated it has built **~100 acceleration libraries since 2018**, spanning communication, compilation, drivers, quantization and performance analysis. The libraries are **not individually enumerated publicly**, so this is a company claim without a verifiable inventory.

## Deep-Spark repository activity in-window

| Repository | Status / last update | Note |
|---|---|---|
| `FlyDSL` | 2026-08-08 | New kernel-authoring DSL (above) |
| `DeepSparkInference` | 2026-08-06 | Model zoo now curates **216 inference models** |
| `iluvatar-corex-ixrt` | 2026-08-05 | Inference runtime, actively maintained |
| `DeepSparkHub` | 2026-08-03 | Model/application hub |
| vLLM fork | 2026-06-12 | Iluvatar-targeted vLLM |
| `xllm` | added 2026-05-12 | Multi-accelerator LLM inference engine |
| `LightX2V` | added 2026-05-12 | Video-generation inference |
| `ix-exporter` | 2026-05-20 | Prometheus metrics |
| `ix-device-plugin` | 2026-05-19 | Kubernetes device plugin |

The pattern is a stack broadening from "run frameworks on our GPU" toward serving-engine coverage (vLLM, xllm) and modality coverage (LightX2V for video generation), plus the new low-level kernel-authoring entry point.

## No TianGai 300 toolchain disclosure

Despite the 2026-07-19 hardware announcement, **no software-stack disclosure specific to TianGai 300 was found**: no compiler release notes mentioning FP8/FP4 codegen, no intrinsics documentation for ixSMEX/ixDPX/ixTrans, no FlyDSL backend target for the new architecture, and no SDK version tied to it. How the new ISA extensions are exposed to programmers is **not disclosed**.

## Not found (checked)

- No MLPerf submission by Iluvatar CoreX (zero MLCommons entries through 2025; none in v6.0).
- No Hot Chips / ISCA / ISSCC 2026 presentation. *(Hot Chips 38 runs 2026-08-23/25 — still in the future as of this investigation; program public, content not.)*

## Sources (this update)

- [GitHub — Deep-Spark/FlyDSL](https://github.com/Deep-Spark/FlyDSL) — last updated 2026-08-08; `iluvatar` branch, 873 commits
- [Deep-Spark repositories sorted by last update](https://github.com/orgs/Deep-Spark/repositories?sort=updated) — repository activity snapshot as of 2026-08-08
- [Iluvatar developer portal](https://developer.iluvatar.com) — launched ~2026-07-15/19
- [Tencent Cloud developer news](https://cloud.tencent.com/developer/news/4280275) — ~100 acceleration libraries claim; WAIC 2026 briefing
- [Tencent News, 2026-07-19](https://news.qq.com/rain/a/20260719A08MB800) — WAIC 2026 launch coverage
