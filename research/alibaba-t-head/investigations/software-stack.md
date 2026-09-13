# T-Head (平头哥) Software Stack Investigation

**Chip:** alibaba-t-head  
**Device Class:** RISC-V AI (平头哥)  
**Investigated:** 2026-04-05  
**Sources:** T-Head official site, HC32 2020 presentation, Hot Chips paper, GitHub T-head-Semi, TechNode 2026

---

## 1. Software Ecosystem Overview

T-Head operates two largely separate software stacks:

| Stack | Target Hardware | Openness |
|---|---|---|
| **HGAI SDK** | Hanguang 800 NPU (inference) | Proprietary, cloud-only access |
| **Zhenwu SW Stack** | Zhenwu 810E PPU (training+inference) | Proprietary, PyTorch/ONNX frontend |
| **Xuantie Toolchain** | Xuantie RISC-V CPUs | Open-source (Apache 2.0) |

---

## 2. HGAI SDK — Hanguang 800 NPU

### Overview
HGAI (Hanguang Artificial Intelligence SDK) is T-Head's proprietary SDK for the Hanguang 800. It is exposed to users via Alibaba Cloud ECS instances; no public download/distribution.

### Supported Frontend Frameworks
- TensorFlow
- MXNet
- Caffe
- ONNX (model interchange)

### Pipeline Stages
```
User model (TF/MXNet/Caffe/ONNX)
       │
       ▼
[Graph IR Conversion]
  frontend parser → unified Graph IR
       │
       ▼
[Quantization]
  INT8 post-training quantization
  sparse weight compression
       │
       ▼
[NPU Compilation]
  graph partitioning → TE/PE/ME mapping
  memory tiling for 48 MB SRAM/core
  pipeline scheduling
       │
       ▼
[Binary / Runtime artifact]
       │
       ▼
[Hanguang Runtime (HGRT)]
  PCIe DMA, kernel launch, result collection
       │
       ▼
[NPUSMI monitoring]
  frequency, memory utilization, compute utilization
```

### Key Compiler Features
- Graph-level operator fusion
- Convolution acceleration: standard, deconvolution, dilated, 3D
- Additional ops: interpolation, ROI pooling, matmul
- Sparse compression pass for weight tensors
- INT8 quantization (primary precision)

### Deployment Mode
- Available as cloud inference API on Alibaba Cloud
- Internal Alibaba workloads: product search, auto-translation, recommendations, ads, customer service

---

## 3. Zhenwu 810E Software Stack

### Overview
Fully in-house developed; hardware and software independently owned by T-Head. Designed for Qwen LLM series but supports general models.

### Frontend
- **PyTorch** (primary, via frontend API compatibility layer)
- **ONNX** (model interchange for inference deployment)
- **TensorFlow** (claimed compatibility)

### Compiler
- T-Head proprietary compiler (name not publicly disclosed)
- MLIR-based internal IR (inferred from architecture)
- Supports training and inference graph lowering
- Known limitation: early operator coverage gaps for complex RL workloads

### Communication / Collective
- **ICN (Inter-Chip Network):** 7 links × proprietary protocol, 700 GB/s aggregate
- Collective operations for multi-card training (AllReduce, AllGather equivalents)
- Near-linear scaling demonstrated

### Serving / Inference
- Optimized inference path for Qwen3, DeepSeek V3
- Deployed via Alibaba Cloud Model Studio API
- 50% inference price reduction vs GPU baseline

### Monitoring
- Internal cluster management tools (not publicly documented)

---

## 4. Xuantie RISC-V Open-Source Toolchain

The Xuantie CPU ecosystem has the most mature open-source software support.

### Repositories (github.com/T-head-Semi / github.com/XUANTIE-RV)

| Repo | Description |
|---|---|
| xuantie-gnu-toolchain | GCC + Binutils for Xuantie RISC-V (RVV, extensions) |
| newlib | Lightweight C library for embedded Xuantie targets |
| buildroot | Buildroot customized for Xuantie CPU boards |
| riscv-aosp | Android 10/AOSP port for XuanTie RISC-V boards |
| openc906 | OpenC906 RTL (Verilog, Apache 2.0) |
| OpenC910 | OpenC910 RTL (Verilog, Apache 2.0) |

### Open-Sourced CPU Cores (Apache 2.0)
- E902, E906 (embedded in-order)
- C906, C910 (mid/high-performance; Verilog + toolchain + scripts)

### Linux Ecosystem
- Linux kernel upstreaming for Xuantie boards in progress
- Buildroot + Debian distributions available for dev boards (e.g., Sipeed LicheePi 4A with TH1520 SoC containing C910 cores)

### AI on Xuantie (C950 specific)
- Native support for Qwen3 and DeepSeek V3 (100B+ parameter class)
- C950 targets agentic AI agent workloads at the CPU level
- No separate NPU SDK needed for C950 LLM inference (integrated AI engine)

---

## 5. Software Stack Maturity Assessment

| Layer | Hanguang 800 | Zhenwu 810E | Xuantie CPU |
|---|---|---|---|
| Framework frontend | TF/MXNet/Caffe/ONNX | PyTorch/ONNX/TF | Standard Linux toolchain |
| Compiler | Proprietary HGAI | Proprietary (MLIR-based) | GCC + LLVM (upstream) |
| Runtime | HGRT (cloud-only) | Proprietary | Standard Linux |
| Operator coverage | CNN-focused; limited | General + LLM gap | Full RISC-V vector |
| Open-source | None | None | Extensive (cores + toolchain) |
| Ecosystem maturity | Low (cloud-only) | Medium | Medium-High |
| PyTorch support | No | Yes | Via standard CPU path |

---

## 6. Competitive Positioning

- No equivalent of CUDA, ROCm, or OpenCL — fully proprietary inference APIs
- Hanguang 800 never publicly released; inference consumed internally by Alibaba
- Zhenwu 810E first chips to challenge NVIDIA in China market at cloud scale
- Xuantie RISC-V open-source approach mirrors RISC-V community norms; targets edge/embedded + emerging server market
- C950 AI agent positioning attempts to differentiate from pure NPU approach

---

# Investigation Update — 2026-08-08

**Investigated:** 2026-08-08 (window: 2026-04-05 baseline → 2026-08-08)
**Change class:** major — the "no public SDK / fully proprietary" characterisation above is superseded, with caveats.
**Primary source:** T-Head news post, 2026-07-18 (https://www.t-head.cn/news/newsDetail?id=189)
**Independent corroboration:** SCMP, The Next Web, Nation Press; technical deep-dives on happyrock.cloud and developer.aliyun.com

---

## 7. SAIL Software Stack — announced open-sourced 2026-07-18 (WAIC Shanghai)

T-Head announced at **WAIC Shanghai on 2026-07-18** that it had open-sourced **SAIL**, its full software stack for the Zhenwu accelerator line. This is the first break in the entirely closed Zhenwu software story documented in §3 above.

### What is claimed

| Claim | Value | Standing |
|---|---|---|
| Framework compatibility | **260+ mainstream training and inference frameworks**, explicitly including **PyTorch, TensorFlow, vLLM, SGLang** | Vendor claim; not independently benchmarked |
| CUDA migration effort | **"fewer than seven days"** | Vendor statement, reported as such by SCMP |
| Alternative migration wording | "migrate without modifying existing code" | One technical deep-dive; **not equivalent** to the seven-day claim, and the two have not been reconciled |
| Kernel integration | Native driver adaptation to the **OpenAnolis Anolis Cloud Kernel (ANCK)** | Reported in technical coverage |
| Distribution | Via the **T-Head developer community** | Reported; not a GitHub organisation |

### Layer decomposition — NOT settled

Coverage disagrees on the shape of the stack, and this survey does not adopt a canonical layer count:

- **Five-layer description** (several outlets): kernel drivers → compiler → high-performance operator + communication libraries → SDK/toolchain → performance-analysis/debug tools.
- **Three-layer description** (at least one technical deep-dive): interface / SDK / OS.

Both are recorded; neither is asserted as fact. The raw scan stated five layers as settled — that is an overstatement.

### Openness — vendor-asserted, not verified

**No public Git repository URL and no license could be confirmed from any retrieved source.** Distribution appears to run through the T-Head developer community rather than a public code host. Consequently:

- "Open source" is recorded as **vendor-asserted availability**, not a verified public repo.
- The **Xuantie RISC-V repositories (§4) remain the only confirmed-public T-Head code** — Apache 2.0, on github.com/T-head-Semi and github.com/XUANTIE-RV.
- Nothing in the SAIL announcement changes the Hanguang 800 / HGAI position (§2), which remains cloud-only and closed.

### What this supersedes

| Prior statement (2026-04-05 baseline) | Status |
|---|---|
| "No public SDK exists for either chip; access is exclusively through Alibaba Cloud's Model Studio API" | **Superseded** by the SAIL announcement, subject to the openness caveat above |
| "T-Head's compiler stack is fully proprietary" | **Superseded** — SAIL is announced to include a compiler layer; IR design, passes and internal name remain **not disclosed** |
| "Both Hanguang and Zhenwu use proprietary closed-source PCIe/ICN drivers. Not open-sourced." | **Partially superseded** for Zhenwu — SAIL is announced to include a kernel-driver layer with ANCK adaptation. Hanguang unchanged |

### Serving-platform change

The **Panjiu AL128 supernode** (128 accelerators, see hw-architecture.md §7) is stated by Alibaba as "now available through Alibaba's model service platform, **Bailian**." This is the only availability Alibaba asserts for the M890 generation — the chip itself carries no GA or mass-production claim.

### Not disclosed

- SAIL repository URL, license, version, release cadence
- Compiler IR, pass pipeline, internal compiler name
- Operator library coverage, op list, kernel authoring interface
- Communication library API, collective coverage, NCCL-equivalence
- Profiling/debug tool names and capabilities
- Whether SAIL targets the M890 only, or the 810E and earlier parts as well
- Any FP4 programming interface, despite the M890's "FP32 down to FP4" hardware claim

### Not found in this window

- No MLPerf submission using SAIL or any T-Head part.
- No T-Head software presentation confirmed at ISCA 2026 or ISSCC 2026. Hot Chips 38 runs **2026-08-23–25**, after this investigation date; no T-Head talk is confirmed on its program and no content from that conference exists yet.
- No change to the Xuantie GNU toolchain, OpenC906/OpenC910 RTL, or the C950 native-inference story in §4.
