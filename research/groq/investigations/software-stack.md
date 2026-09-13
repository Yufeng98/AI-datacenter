# Groq LPU — Software Stack Investigation

## Summary

Groq's software stack is uniquely simple compared to GPU stacks. Because the hardware is fully compiler-scheduled with no dynamic runtime decisions, the entire "intelligence" lives in the ahead-of-time (AOT) compiler. There are no kernel libraries, no device-specific runtime schedulers, and no user-visible ISA in the traditional sense. The stack is: Framework → GroqFlow (Python automation) → torch-MLIR/ONNX frontend → Groq Compiler (cycle-exact spatial scheduler) → GroqWare (groq-devtools + groq-runtime) → LPU hardware.

Following the **December 2025 non-exclusive licensing agreement** with NVIDIA (*not* an acquisition — corrected 2026-08-08), the NVIDIA Groq 3 LPX integrates with NVIDIA's Vera Rubin platform. Groq's own GroqWare/GroqFlow/GroqCloud stack continues to operate independently. See the 2026-08-08 update section at the end of this file for what is and is not publicly known about the NVIDIA-side software path.

---

## Framework Integration

### Supported Frontends

- **PyTorch**: Primary path via GroqFlow. `groqflow.groqit()` accepts a `torch.nn.Module`. Internally, GroqFlow traces the model using `torch.jit.trace()` or torch-MLIR's `torch-mlir-opt` to produce MLIR. No custom PyTorch backend (`PrivateUse1`) — the model is compiled offline, not run eagerly.
- **ONNX**: Alternative entry point. GroqFlow calls ONNX-MLIR (bxing-groq fork) to convert ONNX models to MLIR before feeding the Groq compiler.
- **TensorFlow / Keras / CoreML**: Supported via intermediate ONNX export (tf2onnx, coremltools), then ONNX → MLIR path.
- **groq-python SDK**: OpenAI-compatible Python client for Groq Cloud inference (GroqCloud). Not related to on-device compilation; it is a REST API wrapper with streaming support. Available at `pip install groq`.

### GroqFlow

- **GitHub**: https://github.com/groq/groqflow (open source, Apache 2.0)
- **Entry point**: `from groqflow import groqit; gmodel = groqit(my_pytorch_model, inputs)`
- Automates the full pipeline: model capture → MLIR lowering → Groq compiler → binary (.iop file) → groq-runtime execution
- Supports build caching: re-uses compiled `.iop` if the model and inputs are unchanged
- Also includes `mlagility` benchmark tooling for evaluating model performance across hardware targets

---

## Compiler / IR

### Compilation Pipeline

```
PyTorch / ONNX / TF / CoreML
        ↓
torch-MLIR or ONNX-MLIR frontend
   (produces standard MLIR with torch/linalg dialect ops)
        ↓
Groq GTen Dialect lowering
   (decomposes torch/linalg ops into ~tens of Groq GTen ops:
    GTen is Groq's MLIR dialect for DL + HPC workloads)
        ↓
Groq Compiler (groq-devtools)
   - Op fusion pass (automatic, no custom kernels)
   - Spatial placement: assigns each op to physical functional unit
     in 2D grid (Matrix Mult Unit or Vector ALU bank)
   - Cycle-exact scheduling: every op assigned to exact clock cycle
   - SRAM address assignment: weight tiles mapped to specific SRAM banks
   - Inter-chip routing: for multi-chip models, packet delivery
     scheduled to exact cycles over plesiosynchronous fabric
   - Output: .iop binary (Instruction Operation Program)
        ↓
.iop binary (contains: instruction stream, weight data, routing tables)
```

### Key Compiler Properties

- **No custom kernel authoring**: Groq explicitly avoids user-written kernels. The compiler fuses ops automatically from GTen ops. This contrasts sharply with CUTLASS (NVIDIA), TBE-TIK (Ascend), and AscendC.
- **Static scheduling — everything decided at compile time**: No speculative execution, no cache management, no runtime dispatching. The compiler is responsible for:
  - Instruction ordering and parallelism extraction
  - Memory layout (SRAM bank assignment)
  - Data movement timing (cycle-exact SRAM reads/writes)
  - Inter-chip packet schedules (for GroqRack multi-chip models)
- **MLIR-based**: Uses upstream MLIR infrastructure. Groq contributed to torch-mlir and ONNX-MLIR (bxing-groq GitHub).
- **Assembler**: For bare-metal/research use, Groq also provides a low-level assembler allowing direct instruction specification.

### Hot Chips 34 (2022) Software-Defined Architecture

The Hot Chips 34 paper "The Groq Software-defined Scale-out Tensor Streaming Multiprocessor" details the second-generation multi-chip architecture where the compiler schedules not just on-chip execution but also inter-chip packet delivery cycles, making the entire 576-chip GroqRack appear as a single programmatically scheduled processor.

---

## Op Library

Not applicable in the traditional sense. Groq does not ship a cuDNN/MIOpen equivalent. All operator fusion is handled by the Groq Compiler's pass pipeline operating on GTen dialect ops. The compiler decomposes GTen ops automatically — no hand-tuned kernel dispatch.

This is architecturally intentional: since execution is static and deterministic, the compiler can always produce optimal schedules without needing hand-tuned kernel variants.

---

## Kernel Library

Not applicable. Groq has no user-facing kernel library (no CUTLASS/CK equivalent). Users never write or select kernels for the LPU. The hardware's spatial layout and deterministic execution model make per-kernel tuning unnecessary — the compiler performs all placement and scheduling globally.

---

## Runtime

### groq-runtime

- Loads compiled `.iop` binary onto GroqChip hardware
- Manages SRAM weight loading (one-time load of weights into chip SRAM at startup)
- Executes inference: feeds input activations, retrieves output activations
- No dynamic scheduling at runtime — execution follows the pre-compiled instruction stream exactly
- Minimal runtime: no context management, no stream/event API, no graph capture (all already encoded in .iop)

### groq-python (Cloud SDK)

- OpenAI-compatible REST client
- `from groq import Groq; client = Groq(); client.chat.completions.create(...)`
- Supports streaming token output, function calling, tool use
- Available models (2025): Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, Gemma 2 9B, Whisper large-v3
- Performance: 750–900 tokens/sec on Llama 3.3 70B (2025)
- With Speculative Decoding: 1,660+ tokens/sec on Llama 3 70B (late 2024)

### GroqNode / GroqRack Runtime

- GroqNode: single-server unit (typically 8 LPU chips)
- GroqRack: full rack of 576 chips, acting as single coherent system
- Runtime launches inference jobs; all 576 chips execute in lockstep per compiled schedule
- Plesiosynchronous sync: runtime applies periodic software phase adjustments to cancel crystal clock drift across chips

---

## Driver / Firmware

- Groq PCIe driver (Linux kernel module): BAR mapping, command queue management, interrupt handling
- No on-chip firmware subsystem equivalent to NVIDIA's GSP — the LPU's execution is controlled entirely by the pre-compiled instruction stream in SRAM; no on-chip processor manages resource allocation dynamically
- GroqWare SDK bundles both `groq-devtools` (compiler) and `groq-runtime` (driver + runtime library)

---

## Communication

- **Plesiosynchronous chip-to-chip protocol**: compiled-in packet schedules; no separate collective communications library
- For GroqCloud deployment: standard HTTPS/REST from client to GroqCloud API
- No NCCL equivalent: all inter-chip communication in GroqRack is pre-scheduled by compiler, not a runtime collective library

---

## Assembler / ISA

- **No user-visible virtual ISA** (no PTX equivalent)
- The `.iop` binary is the output of the compiler; its format is not publicly documented
- For research: Groq has published a bare-metal assembler interface in some academic contexts
- The ISCA 2020 paper describes the ISA at a high level: the TSP uses a VLIW-like instruction format where multiple functional units are scheduled in each instruction word, but the exact encoding is proprietary

---

## Post-Licensing-Agreement (2026) — CORRECTED 2026-08-08

*The heading of this section previously read "Post-NVIDIA-Acquisition". There was no acquisition; see the update section below.*

### NVIDIA Groq 3 LPX — corrected facts

- LP30 chip: **500 MB** on-chip SRAM per die (not 512 MB), 150 TB/s SRAM bandwidth, **1.2 PFLOPS FP8** per chip
- **Process node not disclosed** (the "Samsung 4nm" attribution is unconfirmed); **no LPX-specific ship date** — status ANNOUNCED at GTC 2026-03-16, with H2 2026 Vera Rubin partner-availability guidance
- 256 LPUs per rack in 32 liquid-cooled 1U trays of 8; 128 GB rack SRAM at 40 PB/s aggregate; 640 TB/s rack scale-up; 12 TB DDR5 per rack
- Role in Vera Rubin platform: **latency-sensitive decode co-processor**. Corrected split — Rubin GPUs take throughput-bound decode work (full-context attention over the accumulated KV cache); LPX accelerates latency-sensitive decode work such as sparse MoE expert feed-forward networks
- "Up to 35x higher inference throughput per megawatt" — **NVIDIA marketing claim, unaudited**
- ⚠️ **Unverified software claim carried from the 2026-04-05 revision:** "NVIDIA NIM (Inference Microservice) routing layer dispatches prefill to Rubin, decode to Groq LPX" and "NIM bridges CUDA ecosystem to LPX". Neither NVIDIA's developer blog nor the LPX product page describes a NIM-based dispatch mechanism, and the prefill/decode framing it rests on is itself wrong. **Treat as not confirmed.** The LPX programming model — compiler, runtime, dispatch layer, and whether Groq's `.iop`/GroqWare toolchain is retained on NVIDIA silicon — is **not disclosed**.

---

# Investigation Update — 2026-08-08 (software / SDK / corporate)

**Headline: there is no publicly documented software stack for NVIDIA Groq 3 LPX.** NVIDIA's two primary LPX artifacts (the 2026-03-16 developer blog and the nvidia.com LPX product page) are hardware-only. No compiler, runtime, SDK, container, driver, or ISA documentation for LP30 has been published as of 2026-08-08. Every LPX software statement in this file that is not attributed to a primary source above should be read as **not disclosed**, not as inferred fact.

## 1. Corporate correction affecting stack attribution

The 2025-12-24 transaction was a **non-exclusive inference-technology licensing agreement** plus a large acqui-hire, not an acquisition. Groq "will continue to operate as an independent company" and "**GroqCloud will continue to operate without interruption**." This matters for the software survey: **GroqFlow, GroqWare (groq-devtools + groq-runtime), the GTen dialect, the `.iop` format, the groq-python SDK and the GroqCloud API remain Groq's**, and continue to be developed and operated by Groq. NVIDIA holds a non-exclusive license to the inference *technology*; nothing in the primary sources indicates NVIDIA took ownership of Groq's toolchain.

The widely cited **~$20B** value originated with CNBC (2025-12-24) and is **not confirmed by either company**.

Leadership as of 2026-08-08: **Adam Winter (CEO)**, Matt Eng (CFO), Alan Rice (COO), Sinclair Schuller (CTO), Rakesh Malhotra (CPO). Simon Edwards, CEO at the time of the transaction, left in April 2026.

## 2. GroqCloud — the surviving software product line

Groq's 2026 activity is concentrated on GroqCloud, not on new silicon:

- **"GroqCloud: Expanding to Meet Demand" (2026-02-16)** — the only Groq platform post of 2026.
- **2026-06-22 raise ($650M, Disruptive and Infinitum leading, valuation not disclosed):** GroqCloud serves **"more than five million developers"** across **13 data centers** in North America, Europe, the Middle East and APAC, with a stated target of **~200 MW by end of 2027**.
- Groq states it will fit out that footprint "with Groq's latest inference technology, **including the new LPX system from NVIDIA**." Operationally this means GroqCloud is expected to become a **heterogeneous fleet** — Groq LPU v1/v2 systems plus NVIDIA LPX systems — behind one OpenAI-compatible API. **How the GroqCloud serving layer will target LPX, and whether the Groq compiler emits code for LP30, is not disclosed.**

## 3. No new SDK, compiler, or runtime release found

No new GroqFlow, GroqWare, groq-python or compiler release is documented in the 2026-04 to 2026-08 window by any primary source retrieved. The stack described in the body of this file remains the current public picture.

## 4. Scheduled disclosure — Hot Chips 38 (not yet presented)

**Session AI 1, Tuesday 2026-08-25, 2:15–4:15 PM PDT: "Think Fast: LPU Accelerator for Heterogeneous Compute" — Igor Arsovski & Santosh Raghavan, NVIDIA**, at Hot Chips 38 (Aug 23–25 2026, Memorial Auditorium, Stanford). *Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.* The "heterogeneous compute" framing suggests the GPU/LPU work-partitioning mechanism may be described, which is the single largest gap in the LPX software picture. Do not cite the talk title as the source of any spec. Re-scan after 2026-08-25.

## Sources (2026-08-08 update)

- https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale
- https://groq.com/newsroom/groq-raises-usd650m-to-scale-its-ai-inference-cloud-business
- https://groq.com/about-us
- https://groq.com/blog
- https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/
- https://www.nvidia.com/en-us/data-center/lpx/
- https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform
- https://hotchips.org/advance-program/

---

## Resources

- https://github.com/groq/groqflow (GroqFlow open source)
- https://console.groq.com/docs/overview (GroqCloud API docs)
- https://github.com/groq/groq-python (Python SDK)
- https://groq.com/meet-groq-compiler-solutions-for-a-symbiotic-software-hardware-ecosystem/ (Compiler blog)
- http://pkamath.com/publications/papers/tsp-isca20.pdf (ISCA 2020)
- https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf (Hot Chips 34)
- https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/ (NVIDIA Groq 3 LPX)
