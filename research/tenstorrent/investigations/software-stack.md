# Tenstorrent Software Stack Overview Investigation

*as_of: 2026-04-05*
*device_class: Tensix RISC + SFPU*
*sources: official docs, GitHub repos, search results*

---

## Overview

Tenstorrent's software stack is a fully **open-source**, vertically integrated system spanning from ML framework frontends down to hardware intrinsics. It consists of three primary user-facing layers — **TT-Forge** (compiler), **TT-NN** (op library), and **TT-Metalium** (runtime/kernel SDK) — plus **TT-LLK** (hardware intrinsics), **TT-MLIR** (compiler IR), **TT-KMD** (kernel driver), and **TT-ISA** (documented instruction set).

The open-source nature of the full stack is a deliberate strategic differentiator from NVIDIA's partially closed stack (closed cuDNN, closed driver firmware) and a competitive moat for Tenstorrent's developer ecosystem.

---

## Layer-by-Layer Architecture

### Layer 0: ISA and Hardware Intrinsics (TT-ISA + TT-LLK)

**TT-ISA Documentation** (`tenstorrent/tt-isa-documentation`)
- Complete, public instruction set reference for Wormhole B0 and Blackhole A0
- Documents: Baby RISC-V ISA, Matrix Unit (FPU) ISA, Vector Unit (SFPU) ISA, Scalar Unit (ThCon) ISA
- Matrix unit instructions: `MVMUL` (matrix-vector multiply), `GMPOOL`, `ELWMUL`
- SFPU: 32-lane SIMD operations (Exp, Log, Sqrt, Tanh, Sigmoid, custom)

**TT-LLK** (`tenstorrent/tt-llk`)
- Header-only C++ library: low-level kernel (LLK) primitives
- Implements the **Unpack → Math → Pack** pipeline at the instruction level
- Provides: `llk_unpack_A`, `llk_math_eltwise_unary`, `llk_pack`, etc.
- Architecture-specific implementations for Wormhole and Blackhole (separate repos: `tt-llk-wh`, `tt-llk-bh`)
- Used internally by TT-Metalium's compute kernel API; not typically called directly by users

**Key LLK pipeline stages**:
1. **Unpack**: Converts tiles from SRAM (arbitrary precision) → FPU input registers; handles data format conversion
2. **Math**: Executes FPU/SFPU instructions (MVMUL, ELWMUL, activations)
3. **Pack**: Converts FPU output registers → SRAM tiles; handles precision conversion for output

---

### Layer 1: Runtime SDK (TT-Metalium)

**TT-Metalium** (in `tenstorrent/tt-metal`)
- Low-level, open-source SDK for direct hardware programming
- Exposes RISC-V cores, NoC, FPU/SFPU, and SRAM to the programmer
- Provides a C++ API for kernel authoring, compilation, and execution

**Core APIs**:
```cpp
// Device management
Device* device = CreateDevice(0);
// Buffer allocation
Buffer* dram_buf = CreateBuffer(BufferConfig{.device=device, .page_size=tile_sz, ...});
// Program construction
Program program = CreateProgram();
KernelHandle k = CreateKernel(program, "kernel.cpp", cores, DataMovementConfig{...});
CircularBuffer cb = CreateCircularBuffer(program, cores, CircularBufferConfig{...});
// Execution
CommandQueue& cq = device->command_queue();
EnqueueProgram(cq, program, false);
Finish(cq);
```

**Key abstractions**: `Device`, `Program`, `Kernel`, `CircularBuffer`, `Buffer`, `CommandQueue`

**Kernel compile model**: Kernels are C++ files compiled to RISC-V binaries at `CreateKernel` time using a JIT compiler (via SFPI, a custom RISC-V compiler infrastructure). The Metalium JIT compilation pipeline:
1. C++ source → SFPI RISC-V cross-compiler → ELF binary
2. ELF binary packaged into a Program dispatch descriptor
3. Fast Dispatch path sends binaries to Tensix cores via ARC management core

**Supported hardware**: Grayskull (deprecated), Wormhole, Blackhole

---

### Layer 2: Op Library (TT-NN)

**TT-NN** (in `tenstorrent/tt-metal`, under `ttnn/`)
- PyTorch-like Python + C++ neural network op library built on TT-Metalium
- ~800+ model variants tested in CI
- Provides: `ttnn.matmul`, `ttnn.linear`, `ttnn.softmax`, `ttnn.conv2d`, `ttnn.attention`, etc.

**Key design principles**:
- **Tensor-centric API**: `ttnn.Tensor` carries device placement, memory layout (row-major vs. tile), sharding
- **Automatic kernel selection**: TT-NN dispatches to optimized TT-Metalium kernels based on tensor shape, dtype, and sharding config
- **Fused operations**: Common patterns (Linear+GeLU, Attention, LayerNorm+Residual) are fused into single kernels for maximum efficiency

**Tensor layouts supported**:
- `ROW_MAJOR_LAYOUT`: Standard row-major layout in DRAM
- `TILE_LAYOUT`: Native 32×32 tile format in DRAM or L1
- **Sharded layouts**: Height-sharded, width-sharded, block-sharded across Tensix L1s

**Multi-device programming** (`MeshDevice`):
- `ttnn.MeshDevice` abstracts a 2D mesh of physical chips
- Operations can be automatically sharded across devices (tensor parallelism)
- `ttnn.ReplicateTensor` and `ttnn.ShardTensor2D` control placement

**CCL (Collective Communication Library)**:
- Built into TT-NN for device-to-device communication
- Provides: All-Reduce, All-Gather, Reduce-Scatter over Ethernet
- Used for multi-chip inference (tensor-parallel LLM serving)

```python
import ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.matmul(a, b)
result = ttnn.to_torch(c)
```

---

### Layer 3: Compiler / IR (TT-MLIR)

**TT-MLIR** (`tenstorrent/tt-mlir`)
- MLIR-based compiler framework defining custom dialects for Tenstorrent hardware
- Bridges between frontend ML frameworks (via StableHLO/ONNX) and TT-NN/TT-Metalium

**Dialects**:
1. **TTIR** (Tenstorrent IR): Named ops on tensors (akin to StableHLO/TOSA); hardware-independent
2. **TTNN**: Models the TT-NN Python/C++ API as an MLIR dialect; used for op lowering
3. **TTMetal**: Models host-side TT-Metalium operations (buffer allocation, program dispatch)
4. **TTKernel**: Represents compute kernel code within the compiler (for kernel generation)

**Compiler passes**:
- `ttnn-layout`: Converts input tensors to device memory space + tile layout
- `convert-ttir-to-ttnn`: Lowers TTIR ops → TTNN dialect
- Op fusion passes: Fuse adjacent elementwise ops, linear+activation patterns
- Sharding passes: Distribute tensors across mesh devices
- Memory passes: Assign DRAM vs. L1 placement for tensors

**Compilation pipeline**:
```
StableHLO / ONNX
    ↓  (import)
TTIR dialect
    ↓  (ttir-layout, optimization passes)
TTNN dialect
    ↓  (convert-ttir-to-ttnn)
TTMetal dialect
    ↓  (code generation)
TT-NN C++ / TT-Metalium kernels → device binary
```

**ttrt**: The TT-MLIR runtime tool; loads compiled flatbuffer binaries and executes on device.

---

### Layer 4: End-to-End Compiler (TT-Forge)

**TT-Forge** (`tenstorrent/tt-forge`)
- Top-level MLIR-based compiler combining multiple frontends with the TT-MLIR backend
- Goal: Run any ML model from PyTorch/JAX/ONNX/TensorFlow on Tenstorrent hardware

**Frontend integrations**:
- **TT-XLA** (`tenstorrent/tt-xla`): PJRT-based bridge for JAX and PyTorch/XLA; compiles via StableHLO → TT-MLIR; recommended path for new projects
- **TT-Torch** (`tenstorrent/tt-torch`): PyTorch `torch.compile` backend via `torch.export` → StableHLO → TT-MLIR
- **TT-Forge-FE** (`tenstorrent/tt-forge-fe`): Frontend for ONNX, TensorFlow, and other frameworks
- **TT-Buda** (legacy): Predecessor compiler stack; deprecated in favor of TT-Forge

**Validated models (CI)**:
- GPT-OSS 120B, Llama 3 70B, Mistral 7B, DeepSeek
- Stable Diffusion XL, Whisper large-v3, YOLOv12
- All accessible from PyTorch, JAX, and ONNX frontends

**TT-Alchemist**: Code generation tool that translates compiled models into human-readable TT-NN Python library calls (for debugging and optimization)

---

### Layer 5: Driver and System Tools

**TT-KMD** (`tenstorrent/tt-kmd`)
- Linux kernel module: registers `/dev/tenstorrent/%d` device files
- Handles PCIe BAR mapping, command queue setup, interrupt handling
- Upstreamed to NixOS `nixpkgs`

**TT-SMI** (`tenstorrent/tt-smi`)
- Console-based hardware monitor (analogous to `nvidia-smi`)
- Reports device temperatures, utilization, power consumption

**TT-Installer** (`tenstorrent/tt-installer`)
- One-command installer for the full stack via Podman/Docker containers

**TT-NPE** (`tenstorrent/tt-npe`)
- NoC Performance Estimator: simulates data movement across the NoC for a given kernel pattern
- Useful for predicting and optimizing bandwidth bottlenecks before running on hardware

---

## Stack Topology Diagram

```
User Model (PyTorch / JAX / ONNX / TensorFlow)
    ↓
TT-Forge: TT-XLA (JAX/torch.compile) | TT-Torch | TT-Forge-FE (ONNX/TF)
    ↓
TT-MLIR: TTIR → TTNN → TTMetal dialects + optimization passes
    ↓
TT-NN: PyTorch-like op library (~800+ model variants, MeshDevice, CCL)
    ↓
TT-Metalium: Device/Program/CircularBuffer/Kernel C++ SDK
    ↓
TT-LLK: Header-only Unpack/Math/Pack hardware intrinsics
    ↓
Tensix Hardware: BRISC|NCRISC (NoC) + TRISC0/1/2 (FPU/SFPU)
    ↓
TT-KMD: Linux kernel driver (/dev/tenstorrent)
```

---

## Key Insight: Open-Source Stack as Differentiator

Tenstorrent's **entire software stack is open-source** (Apache 2.0 / MIT):
- TT-Metal, TT-NN, TT-MLIR, TT-Forge, TT-LLK, TT-KMD, TT-SMI, TT-Installer
- ISA documentation publicly available (tt-isa-documentation)

This contrasts with NVIDIA (closed cuDNN, closed GSP firmware, closed TensorRT weights) and AMD (partially open). Tenstorrent's open-source commitment reduces vendor lock-in, enables community contributions, and allows researchers to optimize at every stack layer without reverse engineering.

The explicit data movement model means that programmers who understand the NoC and SRAM architecture can achieve theoretically optimal kernel performance — there are no hidden cache behaviors or undocumented hardware mechanisms to work around.

---

## Sources

- [Understanding the Tenstorrent Software Stack](https://docs.tenstorrent.com/getting-started/tt-software-stack.html)
- [TT-Forge Product Page](https://tenstorrent.com/en/software/tt-forge)
- [TT-MLIR Documentation](https://docs.tenstorrent.com/tt-mlir/)
- [TT-MLIR GitHub](https://github.com/tenstorrent/tt-mlir)
- [TT-Forge GitHub](https://github.com/tenstorrent/tt-forge)
- [TT-LLK GitHub](https://github.com/tenstorrent/tt-llk)
- [TT-XLA GitHub](https://github.com/tenstorrent/tt-xla)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [TT-MLIR Dialects Overview](https://docs.tenstorrent.com/tt-mlir/dialects-overview.html)

---

# Investigation Update — 2026-08-08: Metal 2.0 / Device 2.0, DataflowBuffer, TT-Lang, Quasar backend

*as_of: 2026-08-08*
*scope: tt-metal v0.75.0 (published 2026-07-30) and August 2026 dev builds; tt-forge 1.5.0.dev; docs.tenstorrent.com/aibs product/software index; vendor model-coverage and performance claims*
*sources: tt-metal release notes and GitHub commit search, tt-forge releases API, docs.tenstorrent.com/aibs, Tenstorrent newsroom 2026-04-28 / 2026-05-04 / 2026-06-30, The Register 2026-04-28*

## 1. Metal 2.0 / Device 2.0 — a real gap in prior repo coverage, not a new event

The prior repo documents describe TT-Metalium in terms of `Device` / `Program` / `CircularBuffer` / `KernelHandle` / `Buffer` / `CommandQueue`, with the **CircularBuffer** presented as *the* synchronization primitive. A substantial API rearchitecture branded **"Metal 2.0"** (runtime) and **"Device 2.0"** (TT-NN op-facing) has been underway in `tenstorrent/tt-metal` since at least **2025-11-16** and was entirely absent from the repo's coverage and from METALIUM_GUIDE.md as summarised here.

**Timeline evidence:**

| Item | Date | Reference |
|---|---|---|
| Earliest "Device 2.0" commit | 2025-11-16 | `[DM]: Porting DM tests to Device 2.0 APIs first pass` (#32121) |
| "Metal 2.0"-mentioning commits in tt-metal | as of 2026-08-08 | 168 |

**Important:** this predates the repo's 2026-04-05 baseline. It must be recorded as a *coverage gap now closed*, not as an April–August 2026 development.

**Confirmed in tt-metal v0.75.0 (published 2026-07-30):**

| Feature | Description |
|---|---|
| Kernel scratchpad for Metal 2.0 | New kernel-side scratchpad allocation under the Metal 2.0 API |
| NOC-distinctness legality check | Added to the Metal 2.0 API — the runtime now rejects programs that violate NoC-assignment distinctness at API level |
| Device 2.0 migration | Normalization and reduction ops migrated in TT-NN; CNN ops in progress |
| Multi-config layer support | New feature |
| emule multichip fiber engine | Multichip fiber engine plus fabric/CCL teleport, demonstrated on an 8-chip Blackhole LoudBox |

## 2. CircularBuffer → DataflowBuffer (DFB)

The most architecturally significant software change for this survey. tt-metal v0.75.0 carries a systematic **CircularBuffer → DataflowBuffer (DFB)** migration spanning eltwise, conv/pool, data-movement and Moreh kernels, and DFBs are enabled in the **Quasar fast-dispatch flow**.

Consequences for the survey's programming-model narrative:

- The repo's "the circular buffer is the key synchronization primitive" framing (summary.md, Programming Model Rationale item 2) is now generation-specific. It remains accurate for existing Wormhole/Blackhole kernels but is being superseded in new code.
- Both primitives coexist during the migration; `cb_push_back` / `cb_wait_front` remain in the documented NoC/compute kernel API.
- The DFB naming and its enablement specifically in the Quasar dispatch path suggests DFB is the forward-looking primitive for the next silicon generation. That linkage is visible in the commit stream but has **not** been stated by Tenstorrent in prose; recorded as inference.

## 3. TT-Lang

`docs.tenstorrent.com/aibs` now lists a software component named **TT-Lang** that is absent from this repo's stack coverage and from the layer table's prior contents. Its scope, its relationship to TT-Metalium / TT-NN / TT-MLIR, and whether it is a kernel DSL, a graph language, or a frontend are all **not disclosed** in material retrieved for this scan. Recorded as a listed-only component pending a dedicated investigation.

## 4. Quasar in the software stack

Quasar is the only *new* hardware target visible anywhere in the Tenstorrent open-source stack, and it is visible only in the stack — there is no product page, spec sheet or ISA documentation for it.

- First code: `tenstorrent/tt-llk` PR #593, `feat: initial commit of Quasar LLK`, **2025-08-15**; merged into tt-metal **2025-08-17**.
- Commit counts as of this scan: **662** in tt-metal, **85** in tt-llk.
- v0.75.0 / Aug-2026 dev items: trace capture and replay on Quasar; DFBs in the Quasar fast-dispatch flow; Quasar fast-dispatch stress tests; Quasar unicast and iDMA virtual-channel assignment fixes; Quasar-unsupported APIs made to fail at compile time; Quasar unary/binary SFPU performance tests; Quasar LLK matmul perf coverage; Quasar selection in the LLK ttsim regression script.
- `tt-isa-documentation` still has only `WormholeB0` and `BlackholeA0` — no Quasar ISA reference, so the survey's "ISA fully public" claim now has a generation-scoped caveat: it holds for shipped silicon, not for the next generation.

**Do not repeat the raw-scan error** that Quasar "landed" in the Apr–Aug 2026 window, nor that "no process node is disclosed" (Samsung SF4X was announced for the Quasar chiplet in October 2023 — medium confidence that it is the same silicon).

## 5. Model coverage — tt-forge 1.5.0.dev (August 2026)

tt-forge dev builds carry model entries with single-device / data-parallel / tensor-parallel strategy matrices on **n150, n300 and p150**:

- DeepSeek-V3.1, DeepSeek-V3.2 (**no** DeepSeek-V4 entry exists — the raw scan was wrong)
- GLM-4.7
- Kimi K2 / K2.6
- Qwen 2.5, Qwen 3 (including Qwen 3 embeddings)
- Phi-4
- Falcon-3

Reported per-model throughput in those builds is modest, e.g. DeepSeek-V3.1 at **2–3 tok/s/user on n150** and Falcon-3-10B at **40–41 tok/s/user on p150**. These are CI-reported figures on single PCIe cards and should not be conflated with the Galaxy-class marketing numbers below.

**Vendor claim, not independently verified:** "roughly 2.5 million Hugging Face models running on Tenstorrent" at a **~90% pass rate** (TT-Deploy, 2026-05-04). The Register explicitly flags this as awaiting independent verification.

## 6. Performance claims (all vendor marketing; none independently reproduced)

| Date | Workload | Claim | Independent caveat |
|---|---|---|---|
| 2026-04-28 (GA) | DeepSeek-R1-0528 671B | 350+ tok/s/user, batch 8–64, up to 128K context | The Register measured/reported **~300 tok/s/user**, with 350 only *expected* via software improvements; batch size unspecified by Tenstorrent; prior Tenstorrent testing showed "generally poor performance scaling" |
| 2026-05-04 (TT-Deploy) | DeepSeek-R1-0528 671B | same 350+ figure; batch 32 across 16 Galaxies; sub-4 s TTFT at 100K context | not a new record — same claim generation as GA |
| 2026-06-30 | DeepSeek-R1-0528 671B | over 400 tok/s/user; roadmap target 500 tok/s/user at ~$6/M tokens | vendor only |
| 2026-06-30 | Kimi K2.6 | 900 tok/s/user, "3× faster than GPUs" | vendor only; comparison GPU unspecified |
| 2026-06-30 | LTX 2.3 Fast | ~6 s for a 144-frame 1080p video with audio and lip-sync, "4× faster than GPUs" | vendor only |
| 2026-04-28 / 2026-05-04 | Prodia video generation | GA: "720p, 81-frame video in 2.4 seconds, 10× faster than leading GPU systems"; TT-Deploy restates as 2.5 s | the bare "2.5 s" figure in secondary coverage drops resolution and frame count |

**Timeline correction:** the 350+ tok/s/user DeepSeek figure was already the GA (2026-04-28) number — it is not a TT-Deploy-only result. 400+ arrived 2026-06-30.

## 7. Partnerships in the software/observability layer

- **Stealthium** — runtime-observability partnership, **2026-07-30**.
- *Out of window:* Tenstorrent / **Infinia Technologies** sovereign-AI partnership, dated **2026-01-27**, before the repo baseline.

## 8. Explicitly not confirmed

- No Tenstorrent talk in the **Hot Chips 38** (Aug 23–25, 2026) advance program — confirmed absent.
- No Tenstorrent MLPerf Inference v6.0 or MLPerf Training submission found.
- TT-Lang scope and layer placement: **not disclosed**.
- Whether Metal 2.0 / Device 2.0 will fully replace the 1.0 APIs, and on what schedule: **not disclosed**.

## Sources (added 2026-08-08)

- [tt-metal v0.75.0 release (2026-07-30)](https://api.github.com/repos/tenstorrent/tt-metal/releases/tags/v0.75.0)
- [tt-metal releases index](https://github.com/tenstorrent/tt-metal/releases)
- [GitHub commit search — "Metal 2.0" in tt-metal](https://api.github.com/search/commits?q=repo:tenstorrent/tt-metal+%22Metal+2.0%22&sort=committer-date&order=asc)
- [GitHub commit search — Quasar in tt-metal](https://api.github.com/search/commits?q=repo:tenstorrent/tt-metal+Quasar&sort=committer-date&order=asc)
- [GitHub commit search — Quasar in tt-llk](https://api.github.com/search/commits?q=repo:tenstorrent/tt-llk+Quasar&sort=committer-date&order=asc)
- [tt-forge releases API (1.5.0.dev builds)](https://api.github.com/repos/tenstorrent/tt-forge/releases?per_page=5)
- [tt-isa-documentation repo contents](https://api.github.com/repos/tenstorrent/tt-isa-documentation/contents/)
- [docs.tenstorrent.com/aibs — product and software index (TT-Lang listing)](https://docs.tenstorrent.com/aibs/)
- [Tenstorrent Newsroom: Galaxy Blackhole GA (2026-04-28)](https://tenstorrent.com/en/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance)
- [Tenstorrent Newsroom: TT-Deploy (2026-05-04)](https://tenstorrent.com/en/newsroom/tt-deploy)
- [Tenstorrent Newsroom: performance records / TT-Ascalon S (2026-06-30)](https://tenstorrent.com/en/newsroom/tenstorrent-sets-new-performance-records-launches-tt--ascalon-s)
- [The Register: Galaxy Blackhole AI servers are finally out (2026-04-28)](https://www.theregister.com/software/2026/04/28/tenstorrents-galaxy-blackhole-ai-servers-are-finally-out/5229759)
