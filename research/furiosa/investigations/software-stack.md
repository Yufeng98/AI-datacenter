# FuriosaAI Software Stack Investigation

*as_of: 2026-04-05*

---

## Overview

FuriosaAI provides a cohesive open-source SDK called **furiosa-sdk** (Apache 2.0, GitHub: furiosa-ai/furiosa-sdk). The SDK spans the full inference deployment pipeline: quantization, compilation, runtime execution, model serving, and profiling. The stack is Python-first, with ONNX as the canonical model exchange format and a proprietary AOT compiler generating `.enf` binaries (Warboy) or equivalent RNGD binaries.

The RNGD SDK is versioned independently at **FuriosaAI Developer Center 2026.1.0** (`developer.furiosa.ai`), distinct from the older Warboy SDK (0.10.x).

---

## Layer 1: Framework Integration

### PyTorch

FuriosaAI does not implement a PyTorch PrivateUse1 eager backend. Instead, the integration path is:

```
PyTorch Model (torch.nn.Module)
       ↓ torch.onnx.export() or torch.export()
   ONNX / ExportedProgram
       ↓ furiosa-compiler
   .enf binary (Warboy) / RNGD binary
```

For RNGD, FuriosaAI integrates with **PyTorch 2.x `torch.export`** for capturing models as ExportedPrograms, which are then lowered through the compiler.

### ONNX

ONNX is the primary model representation. The `furiosa-compiler` accepts ONNX graphs directly. All supported operator sets (vision: ResNet, EfficientDet, YOLO; LLM: LLaMA, Mistral, EXAONE via RNGD) are accepted as ONNX.

### HuggingFace Transformers (RNGD-specific)

RNGD ships with integration to HuggingFace `transformers` and `text-generation-inference` (TGI) server for LLM deployment:
```
HuggingFace checkpoint (safetensors)
       ↓ furiosa-llm (RNGD-specific)
   Compiled + quantized RNGD model
       ↓ furiosa-serving / TGI backend
   REST API endpoint
```

### furiosa-models

The `furiosa-models` package (PyPI, open source) provides pre-validated model zoo entries for Warboy and RNGD:
- ResNet50, EfficientNetB0 (vision)
- YOLOv5, YOLOv8 (object detection)
- SSD, RetinaFace (face detection)
- Model objects with built-in pre/post-processing pipelines

---

## Layer 2: Compiler / IR

### furiosa-compiler

The FuriosaAI compiler is the central stack component. It accepts ONNX input and produces chip-specific binaries.

**Compilation stages:**
1. **ONNX import**: Parse and validate ONNX graph; check operator coverage
2. **Graph optimization**: Constant folding, dead code elimination, op fusion (Conv+BN+ReLU, GEMM+bias, LayerNorm fusion)
3. **Quantization-aware lowering**: Apply INT8/INT4/FP8 quantization schedules (using calibration data from `furiosa-quantizer`)
4. **Hardware placement**: Assign operators to NPU compute resources vs. CPU fallback
5. **Memory planning**: Allocate SRAM tiles and HBM3 segments (RNGD); tile activation buffers to fit on-chip SRAM
6. **Binary generation**: Produce `.enf` (Engineered NPU Format) for Warboy or equivalent RNGD binary

The compiler is **not open-source** but is distributed as a binary via the FuriosaAI apt/pip repository.

**Key compiler features:**
- **Tactic selection**: Multiple kernel implementation variants per operator; compiler selects the most efficient given tile size and quantization precision
- **Tensor contraction mapping** (RNGD): Maps arbitrary tensor contraction ops to TCP hardware slices; the compiler is aware of the 8 PE × 64 slice grid topology
- **LLM-specific passes** (RNGD): KV-cache allocation, attention fusion (QKV project → scaled dot-product → output project as one TCP operation), RoPE embedding fusion
- **Operator fallback**: Unsupported operators execute on host CPU with automatic data marshaling

### furiosa-quantizer

Separate tool for post-training quantization (PTQ):
```python
from furiosa.quantizer import quantize
quantized_model = quantize(onnx_model, calibration_dataset)
```

Supports:
- INT8 dynamic/static quantization
- INT4 weight-only quantization (RNGD-specific)
- FP8 per-tensor and per-channel quantization (RNGD-specific)
- GPTQ-style block quantization for LLM weights

### Command-Line Tools

```bash
furiosa compile model.onnx --target rngd --output model.rngd.bin
furiosa quantize model.onnx --calib-data cal_data/ --output model_q8.onnx
furiosa perftest model.rngd.bin --batch 1 --loops 1000
furiosa profile model.rngd.bin --output profile.json
```

---

## Layer 3: Runtime

### furiosa-runtime (Warboy)

The Python runtime for Warboy provides a synchronous and async inference API:

```python
import furiosa.runtime as runtime

# Synchronous
with runtime.session.create("model.enf") as session:
    output = session.run([input_tensor])

# Async
async with runtime.session.create("model.enf") as session:
    output = await session.run_async([input_tensor])
```

Internal implementation:
- Loads `.enf` binary onto Warboy via ioctl to the kernel driver
- Manages PCIe DMA transfers for input/output tensors
- Handles multi-session multiplexing on a single Warboy card

### furiosa-runtime (RNGD — 2026.1.0)

RNGD runtime adds:
- **NPU partition management**: Selects 1/2/4/8 virtual NPU partitions at session creation
- **Streaming token generation API**: `generate()` / `generate_async()` for LLM auto-regressive decoding
- **KV-cache management**: Pre-allocates and manages HBM3 KV-cache segments per session
- **Continuous batching**: Multiple inference requests share the same RNGD in a time-multiplexed fashion

### furiosa-llm (RNGD-specific)

High-level LLM inference interface:

```python
from furiosa.llm import LLM, SamplingParams

llm = LLM("meta-llama/Llama-3-8B-Instruct", devices="npu:0:*")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

Supported models:
- LLaMA 2 / 3 (7B–70B)
- Mistral 7B
- EXAONE 3.5 (32B) — validated by LG AI Research
- Gemma 2B / 7B

---

## Layer 4: Serving

### furiosa-serving (Torchserve-style)

HTTP/gRPC inference server with:
- OpenAI-compatible Chat Completions API (`/v1/chat/completions`)
- Triton Inference Server compatible REST API
- Health check and metrics endpoints

### TGI Integration (RNGD)

The NXT RNGD server ships with a HuggingFace Text Generation Inference (TGI) backend plugin, enabling drop-in use of HuggingFace's production serving stack:
```bash
docker run furiosaai/text-generation-inference \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --device npu:0:*
```

---

## Layer 5: Driver / Firmware

### Kernel Driver (Linux)

FuriosaAI ships a Linux kernel module (`furiosa-npu.ko`) distributed via their apt repository (`apt install furiosa-driver`):
- PCIe BAR mapping for MMIO register access
- DMA channel management for weight loading and activation I/O
- NPU partition management (RNGD partition selection)
- No on-chip firmware RM (unlike NVIDIA GSP) — NPU executes compiler-scheduled instruction streams directly

### Firmware

RNGD includes a minimal on-chip microcontroller for:
- Power management (DVFS)
- Thermal monitoring
- PCIe link management

This is distinct from NVIDIA's GSP model — the FuriosaAI firmware does not manage operator scheduling; that is entirely determined by the compiler.

---

## Layer 6: Profiling / Debug Tools

### furiosa-profiler

Post-run trace capture compatible with Chrome Tracing JSON format:
```python
with runtime.session.create("model.enf", profiler=True) as session:
    session.run([input])
session.save_trace("trace.json")
# Open in chrome://tracing or Perfetto
```

Trace shows:
- Per-operator execution time on NPU
- Memory DMA transfers
- CPU-NPU synchronization points

### Jupyter Notebook Examples

The furiosa-sdk repository includes Jupyter notebooks:
- `Image_Classification.ipynb` — end-to-end ResNet50 on Warboy
- `InferenceAccuracyCheck.ipynb` — accuracy validation after quantization

---

## Programming Model Philosophy

FuriosaAI's software stack reflects a deliberate **compiler-first, runtime-thin** philosophy that parallels Groq's approach but retains an open programming surface:

1. **All scheduling is AOT**: No JIT, no dynamic kernel selection at runtime. The compiler produces a complete execution plan including memory tiling, quantization, and operator fusion.

2. **ONNX as lingua franca**: Any framework that can export ONNX is supported. No framework-specific backend required — unlike TensorFlow or JAX which need hardware-specific integrations.

3. **Python SDK for accessibility**: Explicit parallel to CUDA Python — users who know NumPy can write NPU inference applications with minimal new concepts.

4. **LLM-first evolution**: The RNGD SDK (2026.1.0) adds `furiosa-llm` as a first-class API, reflecting the shift from vision to generative AI workloads.

---

## Resources

- [GitHub: furiosa-ai/furiosa-sdk](https://github.com/furiosa-ai/furiosa-sdk)
- [FuriosaAI GitHub Organization](https://github.com/furiosa-ai)
- [FuriosaAI Developer Center 2026.1.0](https://developer.furiosa.ai/latest/)
- [RNGD Overview](https://developer.furiosa.ai/latest/en/overview/rngd.html)
- [Warboy SDK Docs (0.10.2)](http://developer.furiosa.ai/docs/latest/en/npu/warboy.html)
- [Python SDK Guide](http://developer.furiosa.ai/docs/latest/en/software/python-sdk.html)
- [furiosa-models Getting Started](http://developer.furiosa.ai/furiosa-models/latest/getting_started/)
- [LG AI Research EXAONE Benchmark](https://furiosa.ai/blog/lg-ai-research-taps-furiosaai-to-achieve-2-25x-better-llm-inference-in-production-vs-gpus)

---

# Investigation Update — 2026-08-08

*Scan window: 2026-04-10 → 2026-08-06. The 2026-04-05 baseline above is retained; the sections below state what has changed and which baseline claims are now false.*

## A. Headline: FuriosaAI now has a user-facing kernel language

**This falsifies a baseline assertion in this repo.** The 2026-04-05 text above and `chips/furiosa/summary.md` both stated that FuriosaAI has "no kernel library (no CUTLASS equivalent)" and that "users never write custom kernels." SDK **2026.3.0**, released **2026-06-30**, ships **TCL (Tensor Contraction Language)**.

### TCL — Tensor Contraction Language

FuriosaAI's own description: "a declarative Python eDSL in which a kernel author writes a high-level, `@tcl.kernel`-decorated function that says **what** to compute and leaves **how** to run it — tiling, scheduling, fusion, hardware mapping — to the compiler."

Design properties:
- **Tensor contraction is a first-class language primitive**, matching the TCP hardware primitive one-to-one. This is the tightest language/hardware correspondence in the stack: RNGD's compute unit *is* a contraction engine, and TCL's core construct *is* a contraction.
- **Declarative, not imperative.** The author does not write the schedule. This is a materially different design point from Triton (explicit tiling, explicit block indices) and from Pallas (explicit reference semantics over blocks). Placing TCL in the survey's taxonomy: it sits in the Triton/Pallas *tier* (a user-facing kernel-authoring layer above the compiler) but at a **declarative** point in that tier rather than the schedule-explicit point Triton occupies.
- **Padding, sharding, and multi-chip collectives are expressed at the language level**, not by the user hand-rolling communication.
- Kernels ship as reusable blocks (RMSNorm, Linear, MLP) in a new **`furiosa-kernels`** package.

FuriosaAI's own framing of the payoff: *"Enablement now scales with the number of reusable blocks, not the number of models."* The company credits TCL with unlocking Qwen3-VL (e.g. Qwen3-VL-32B), gpt-oss-120b, Solar-Open-100B, Qwen3-30B-A3B, and K-EXAONE-236B-A23B.

### Openness limits — important for the survey's open/closed axis

| Package | Version | Uploaded | License |
|---|---|---|---|
| `furiosa-kernels` | 2026.3.0 | 2026-06-30T07:28:17Z | **`LicenseRef-Proprietary`** |
| `furiosa-llm` | 2026.3.0 | 2026-06-30 | Apache Software License |

- The SDK is now **mixed-license**: the LLM interface is Apache, the kernel layer is proprietary. Any framing of "the FuriosaAI kernel layer" as Apache 2.0 is wrong.
- The public Developer Center at 2026.3.0 has **no TCL or `furiosa-kernels` documentation section** (checked via navigation and the docs search page). There is no public reference documentation for the language as of 2026-08-08.
- Whether TCL is **generally available to all customers or gated to partners could not be confirmed**.

Net characterization for the survey: *a real, shipped, user-facing kernel-authoring layer whose openness is limited — proprietary license, no public reference documentation.*

## B. Other SDK 2026.3 changes

Reported in the vendor release blog; **not independently verified** (no public documentation exists for most of them):

- **FXB (Furiosa Executable Bundle)** — portable compiled artifacts giving zero-recompilation reuse across compatible model variants. This partially relaxes the baseline's "recompilation needed for shape/variant changes" characterization, though the AOT model itself is unchanged.
- **Multimodal serving with chunked prefill.**
- **Overlap scheduling** to cut NPU idle time.
- **Scoring-based data-parallel request routing**, combining prefix locality and token footprint.
- **Breaking changes** upgrading from 2026.2.

## C. FVISA — "Furiosa Virtual ISA"

Named publicly as a stack layer alongside the compiler and TCL at **RENEGADE Summit 2026** (blog dated 2026-05-13). **No further technical detail is disclosed** — no instruction set, no documentation, no assembler, no user-facing tooling.

Effect on the layer table: the baseline row "No user-visible virtual ISA (no PTX equivalent)" is now imprecise. A virtual ISA layer **exists and has been named**; it is simply not documented or exposed. Recorded as *named only*.

## D. SDK distribution channel correction

`github.com/furiosa-ai/furiosa-sdk` is **stale**: latest release tag **0.9.2 (June 2024)**, still the Warboy-era 0.9.x line, with no mention of TCL, `furiosa-kernels`, or 2026.x. The live SDK ships as **PyPI wheels under a calendar-version scheme (2026.x)**, documented at developer.furiosa.ai.

Consequence: the repo's shorthand "furiosa-sdk, Apache 2.0, GitHub" describes a legacy artifact, not the current SDK, and should not be cited as evidence about the 2026 stack's openness.

## E. Model coverage as of 2026.3

Baseline (LLaMA 2/3, Mistral 7B, EXAONE 3.5 32B, Gemma 2B/7B) plus, new in 2026.3: **Qwen3-VL** (incl. Qwen3-VL-32B), **gpt-oss / gpt-oss-120b**, **Solar-Open-100B**, **Qwen3-30B-A3B** (MoE), **K-EXAONE-236B-A23B**. Samsung SDS's NPUaaS serves Qwen3 and gpt-oss 120B in production.

## F. Research output — ICML 2026 (Seoul)

Four papers from FuriosaAI's AI Research Group, announced 2026-08-06. All are **algorithmic/software; no hardware disclosures**:

| Paper | Contribution | Reported result |
|---|---|---|
| **ReJump** | Tree-jump representation for LLM reasoning | up to 9.1% improvement on reasoning tasks |
| **LoSA** | Locality-aware sparse attention for block-wise diffusion LMs | — |
| **AsyncOPD** | Stale on-policy distillation | 1.6×–3.8× higher training throughput |
| **EfficientRollout** | 4-bit quantized drafting + adaptive speculative decoding | up to 19.6% rollout speedup; 12.7% end-to-end training improvement |

## Sources added 2026-08-08

- https://furiosa.ai/blog/furiosa-sdk-2026-3-a-new-kernel-framework-and-the-models-it-unlocks — primary, 2026-06-30
- https://pypi.org/pypi/furiosa-kernels/json — package registry primary (license, upload timestamp)
- https://pypi.org/pypi/furiosa-llm/json — package registry primary
- https://developer.furiosa.ai/latest/en/ — Developer Center 2026.3.0 (negative: no TCL section)
- https://furiosa.ai/blog/experience-renegade-summit-2026 — FVISA named
- https://furiosa.ai/blog/furiosaai-at-icml-2026-advancing-full-stack-software-efficiency
- https://github.com/furiosa-ai/furiosa-sdk/releases — negative: stale at 0.9.2 (June 2024)
