# Tecorigin (太初元碁) SDAA Software Stack Investigation

*as_of: 2026-08-08*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*

---

## Overview

Tecorigin's stack is a **complete, deliberate CUDA clone laid over a Sunway-idiom machine**. Almost every CUDA-ecosystem component has a named SDAA counterpart — runtime, virtual ISA, kernel language, cuDNN/cuBLAS/cuRAND analogues, NCCL analogue, nvidia-smi, NVML, CUPTI, NVTX, Nsight, cuda-gdb, c++filt, DCGM-diag. The one thing that is *not* CUDA-shaped is the machine underneath: an SPMD slave-core array with a private per-SPE software-managed scratchpad, explicit DMA/RMA/broadcast, and no data cache.

The entire stack is delivered as exactly **two packages**:

| Package | Contents |
|---|---|
| **TecoDriver** | Kernel driver + firmware + management library + monitoring tools |
| **TecoToolKit** | The SDK — compilers, runtime, acceleration libraries, communication library, frameworks, debug/profile tooling |

They version-lock 1:1 — TecoToolKit vN is forward-compatible with TecoDriver ≤ vN. The authoritative component inventory is the **环境安装手册 v3.2.0** (`getReleaseTree/?group=software_installation`), which enumerates every shipped component with its version. **Current release at time of research: v3.2.0.**

Distribution is `.deb` / `.rpm` / `.runfile` packages and Docker tarballs from `mirrors.tecorigin.com` and `jfrog.tecorigin.net/artifactory`. **Nothing is on PyPI and there is no Hugging Face organization** (see Negative Checks).

---

## Layer 1: Framework Integration

### TecoPyTorch (`torch_sdaa`) v3.2.0

Out-of-tree PyTorch device extension on **PyTorch 2.7.1** (a 2.4.0 line also ships). Exposes `device='sdaa'` and a `torch.sdaa.*` namespace mirroring `torch.cuda`.

- Distributed: `ProcessGroupTCCL` registered as `backend='tccl'`; DDP documented across the 4 SDAA devices per card.
- AMP: FP16 training with weight backup and loss scaling.
- **Inference in TecoPyTorch is FP32-only** (documented limitation).
- Custom ops via **SDAA Extension** (see Layer 8); `torch.sdaa.memory.SDAAPluggableAllocator` gives CUDA-parity pluggable allocation.
- Partially open: [gitee.com/tecorigin/teco-torch](https://gitee.com/tecorigin/teco-torch), BSD-3-Clause, 17★, last push 2025-03. Binaries ship only as Docker images / wheels from vendor mirrors.

### TecoPaddle (`paddle_sdaa`) v3.2.0

PaddlePaddle 3.0.0 plugin via Paddle's **CustomDevice** mechanism. Two sources exist and the second matters more:

- [gitee.com/tecorigin/teco-paddle](https://gitee.com/tecorigin/teco-paddle), Apache-2.0, 19★.
- **[PaddlePaddle/PaddleCustomDevice `backends/sdaa`](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa)**, Apache-2.0 — **upstream and Tecorigin-independent**. ~115 kernel `.cc` files, a `sdaac_ops/` directory of custom `.scpp` kernels, a `pr_ci_sdaa.sh` hardware-CI script, and CMake linkage against `TECODNN_LIB`, `TBLAS_LIB`, `TCCL_LIB`, `SDPTI`, `SDAA_LIB`, `TECOCUSTOM_LIB`. Tecorigin does not control this repo, and its hardware CI implies a real device fleet.

### Teco-vLLM v3.2.0

A vLLM port, and the most feature-complete piece of the stack:

- **Serving features**: PagedAttention, Continuous Batching, Chunked Prefill, Automatic Prefix Caching, **Disaggregated Prefill (PD分离)**, LoRA adapters, Tool Calling, Reasoning Outputs.
- **Parallelism**: TP, PP, EP, DP, **EPLB** (expert-parallel load balancer), and **SBO** (single-batch overlap) for shared-expert compute/communication overlap. Multi-node via **Ray**.
- **Quantization**: **INT8 W8A16** (`quantization="sdaa_wint8"`, online dynamic only — "INT8算子的权重排布需要为私有格式"), **INT4 W4A16**, **KV-Cache-INT8**, GPTQ, AWQ, GGUF.

> **This quantization menu is a hardware tell.** Every entry is **weight-only**. There is no W8A8 path — consistent with a matrix unit whose only input types are FP16 and S16 (see the hardware investigation). Record it as evidence, not just as a feature list.

Binaries are proprietary; the model recipes are open at [Tecorigin/tecovllm-modelzoo](https://github.com/Tecorigin/tecovllm-modelzoo) (BSD-3-Clause).

### Teco-Megatron-LM v3.2.0

Megatron-LM port with Tensor / Sequence / Pipeline / Expert parallelism plus a distributed optimizer. Documented recipes include **DeepSeek-R1-Distill-Llama-70B SFT at TP=4, PP=8 on 1 node × 8 cards** — 32 ranks, a clean independent cross-check on 4 SPAs per card. Proprietary.

### TecoInferenceEngine

Two separate engines ship:

- **TecoInferenceEngine（小模型）v3.1.0** — small-model inference engine. **ONNX is the only front end**; PyTorch/Paddle models must be exported to ONNX first. Python + C++ APIs, a TensorRT-migration path, dynamic shapes, async inference, CPU fallback, 70+ models. Its plugin API is open in teco-ops.
- **TecoInferenceEngine（大模型）** — a separate large-model engine (doc id 83566). Proprietary.

### Vendor-claimed additional support (VENDOR CLAIM — not all corroborated in the manuals)

SGLang, xDiT, DeepSpeed, LLaMA-Factory, Transformers, TensorFlow; "40+ mainstream LLMs" (Qwen, GLM, DeepSeek, MiniMax, InternLM, SD / Wan / FLUX / Open-Sora, Whisper).

---

## Layer 2: Graph Capture

| Mechanism | Notes |
|---|---|
| **TorchDynamo / `torch.compile`** | Standard PyTorch capture; `fullgraph` supported. **`dynamic`, `mode` and `options` are reserved and not yet supported** — so there is no dynamic-shape compiled path today |
| **ONNX** | The only front end for TecoInferenceEngine (小模型) |
| **Stream capture** | `sdaaErrorStreamCaptureUnsupported` / `...Invalidated` exist in the runtime enum — CUDA-Graph-style capture is at least stubbed, but no user-facing API is documented |

---

## Layer 3: Graph Compiler

### `teco_inductor`

TecoPyTorch's TorchInductor backend, invoked as `torch.compile(model, backend='teco_inductor')`.

- **Fuses** Pointwise, Reduction and Foreach ops (with broadcasting).
- **Conv and GEMM fall back** to the hand-written acceleration libraries — the compiler does not generate matrix-unit code.
- Codegen options at `torch_sdaa._inductor.config.teco.*`: `use_simd128`, `higher_performance`, `use_table` (lookup-table transcendentals — both `higher_performance` and `use_table` may reduce accuracy), `enable_kernel_profile`, `debug_sync_graph`.
- Proprietary.

### TVM / Relay IR

TecoInferenceEngine's compiler is **TVM-based**. Confirmed by the [teco-ops README](https://github.com/Tecorigin/teco-ops/blob/main/README.md): custom ops are "通过 **TVM Relay IR** 注册", and plugin sources must be built with `g++` rather than `tecocc` "因为需要兼容 TVM C++ 头文件".

Four-layer architecture:

```
Front end        ONNX parse + FP32→FP16 conversion
     ↓
Graph optimize   fusion, constant folding, CSE, data-layout insertion,
                 fused-kernel codegen, pass pipelines
     ↓
Runtime          Engine / Context, async execution, dynamic shape
     ↓
Device
```

TVM itself is Apache-2.0 upstream; Tecorigin's backend is proprietary.

---

## Layer 4: Kernel Compiler and Kernel Language

### TecoCC (`tecocc`) v3.2.0

The SDAA C compiler driver, and demonstrably **Clang/LLVM-derived**:

- Emits `.bc` bitcode separately for host and device.
- Uses `clang-offload-bundler` bundles (`--sdaa-link`).
- Supports `-flto` link-time optimization, `-O0..-O3`, `-g`.
- Device sources use the extension **`.scpp`**.
- Notable flags: `--sdaa-arch=pcx_100`, `--stack-on-global` (moves the SPM stack to Global), `--sdaa-device-only` / `--sdaa-host-only`, `--sdaa-device-lib=` (`.bc` device libraries).
- Proprietary.

### PCXAC (PCX Advanced Compiler) v1.2.0

Compiles and links PCX virtual-ISA files to machine code. Deliberately lightweight. Ships with static and dynamic memory checkers (**MemChecker**). Proprietary.

### SDAA C v3.2.0 (documentation at v3.3.0, released 2026-07-23)

"Software Defined Accelerator Architecture C" — C/C++ with CUDA-shaped extensions.

| Category | Surface |
|---|---|
| Qualifiers | `__global__`, `__device__`, `__host__`, `__local__`, `__scoped_local__` |
| Launch | `<<<...>>>`, `threadIdx`, `threadDim` |
| Thread groups | `ThreadGroup`, `thread_group_set_mask` / `include` / `exclude`, `sync_threads` |
| SPM allocation | `malloc` / `free`, `get_heap_size`, `get_stack_size`, `get_local_size` |
| DMA | `memcpy`, `memcpy_stride`, `memcpy_async` / `memcpy_wait` (`MemcpyHandle`) |
| **RMA** | `rma_get` / `rma_put`, `rma_async_get` / `rma_async_put`, `rma_complete` / `rma_wait` |
| Broadcast | `broadcast`, `broadcast_async`, `memcpy_broadcast` |
| Atomics | `atomic_inc` / `add` / `sub` / `cas` |
| **Matmul** | blocking `matmul`; non-blocking `matmul_init` / `matmul_load_weight` / `matmul_compute` / `matmul_store` / `matmul_wait` |
| Transpose | blocking + non-blocking |
| SIMD | `simd_load/store/loadu/storeu/load_widen/store_narrow/set/stretch`, plus `*128` 128-bit forms |
| Batched math | `batch_gelu`, `batch_sigmoid`, `batch_tanh`, `sum`, … |
| Restrictions | no exceptions, RTTI, STL, `new`, global ctors/dtors, local statics, file I/O, or native C/C++ atomics |

The language itself is proprietary, but **real `.scpp` kernel source is open** in teco-ops, Teco-AL and PaddleCustomDevice.

---

## Layer 5: ISA — PCX (the biggest software finding of this pass)

**PCX (Parallel Computing eXecution) v1.0.0** is a **hardware-independent virtual instruction set architecture** for Tecorigin's heterogeneous many-core architecture. It is structurally **the PTX of this ecosystem**, and it was completely absent from the seed.

It sits between SDAA C / frameworks and the machine ISA so that "同一版本的PCX指令集可以在太初元碁多种系列的硬件上直接编译并高效执行".

**Three usage modes:**
1. TecoCC auto-generates PCX (recommended path).
2. PCX inline-embedded in SDAA C for targeted tuning.
3. Hand-written PCX for maximum performance.

**Features:** multi-level storage abstraction; thread and thread-group hierarchy (`%tid`, `%gid`, `%ngroup`, `%nthread`); **infinite virtual general registers**; scalar, vector and matrix instruction classes; debug info (`.hint`, `.section`, `.file`, `.loc`); performance-sampling instructions.

**Directives:** `.version`, `.arch`, `.entry`, `.func`, `.visible` / `.extern` / `.weak` / `.alias`; storage qualifiers `.spm` / `.global` / `.const`.

**Instruction highlights:** matrix — `matmul_init` / `matmul_load_weight` / `matmul_compute` / `matmul_store`; data movement — `mov`, `lda`, `load`, `store`, `memcpy`, `memcpy_broadcast`, `rma`, `broadcast`.

**Compatibility table currently lists only T100系列** — while the stated purpose is decoupling from "多种系列" of hardware. That gap strongly implies planned successor silicon; **no successor is named.**

**The actual T1 machine instruction set is not disclosed — PCX exists precisely to hide it.** No opcode listing, machine-ISA reference or disassembly format is published.

PCX is proprietary but **fully documented publicly** at [docs.tecorigin.com/release/pcx](http://docs.tecorigin.com/release/pcx).

---

## Layer 6: Acceleration Libraries (Tensor API)

| Component | Version | Library | CUDA analogue |
|---|---|---|---|
| **TecoDNN** | v3.2.0 | `libtecodnn.so` | cuDNN |
| **TecoBLAS** | v3.2.0 | `libtecoblas.so` | cuBLAS |
| **TecoRAND** | v3.2.0 | `libtecorand.so` | cuRAND |
| **TecoLMK** | v3.2.0 | — | Large-model **inference kernel library** — new, not in the seed |
| **TecoCUSTOM** | v3.2.0 | `libtecodnn_ext.so` (formerly "CustomDNN") | custom-operator library |
| **TecoAL** | — | `libtecoal.so` | see naming note |

All proprietary. **None of them has a public manual** — `getReleaseTree` returns 404 for tecoal, tecodnn, tecoblas, tecolmk, tecocustom and tecorand. Only the install manual names them.

### Naming note — TecoAL vs TecoDNN (resolving a seed ambiguity)

The seed correctly flagged a rename in flux: `libtecoal.so` / `tecoal_ext` appear in teco-ops where the shipping stack uses `libtecodnn.so` / `libtecodnn_ext.so`. The **v3.2.0 install manual lists only TecoDNN / TecoBLAS / TecoCUSTOM — not TecoAL**. So TecoAL looks like the **open-source face** of the operator library rather than a shipped rename. **Document both names.** Open source: [gitee.com/tecorigin/teco-al](https://gitee.com/tecorigin/teco-al), BSD-3-Clause, 31★, **157 forks** (competition-inflated), last push 2025-03.

---

## Layer 7: Communication, Runtime, Driver

### Communication

| Component | Version | Notes |
|---|---|---|
| **TCCL** | v3.2.0 (`libtccl.so`) | NCCL analogue. PyTorch integrates it as `ProcessGroupTCCL` (`backend='tccl'`) |
| **TCCLTests** | v1.1.0 | nccl-tests analogue |
| MLNX_OFED 5.9-0.5.6.0 + SHARP 3.2.0 + Open MPI 4.1.5rc2 | — | **Mandatory** multi-node dependency, redistributed by Tecorigin from `mirrors.tecorigin.com` |

### Runtime — SDAARuntime v3.2.0 (`libsdaart.so`)

A CUDA-Runtime analogue, documented at [docs.tecorigin.com/release/sdaart](http://docs.tecorigin.com/release/sdaart) — **a manual the seed did not find.**

- Device: `sdaaSetDevice`, `sdaaGetDeviceCount`, `sdaaGetDeviceProperties`, `sdaaDeviceProp_t`, `sdaaDeviceAttribute_t` (incl. `sdaaDevAttrArch`).
- Memory: `sdaaMalloc`, `sdaaFree`, `sdaaMemcpy`.
- Streams and events: `sdaaStreamCreate`, `sdaaStreamSynchronize`, `sdaaStreamWaitEvent`.
- P2P, plus a full CUDA-shaped `sdaaError_t` enum (including `sdaaErrorECCNotCorrectable`, `sdaaErrorPeerAccessUnsupported`, `sdaaErrorStreamCaptureUnsupported`).

### Driver / firmware

| Component | Version | Notes |
|---|---|---|
| **SDAADriver** | v3.2.0 | Kernel driver. Devices appear as **`/dev/tcaicardN`** (one node per card; containers pass `--device=/dev/tcaicard0..3`). Package `tecodriver` (deb/rpm per host arch). Firmware images named `aiflash_v<ver>_T1`; VBIOS, MCU and PCB versions all reported |
| **TCML** | v1.15.0 | NVML analogue — management/monitoring **library** |
| **TecoSMI** | v1.15.0 (`teco-smi`) | nvidia-smi analogue: `-L/-q/-d/-l/-lms`, `--query-device`, `--format=csv`, `stats`, `dmon`, `daemon`, `replay`, `pmon`, `topo` (incl. `-p2p r\|w`), device reset `-r`, SPE clock lock/reset `-lsc/-rsc`, low-power mode `-lpm` |
| **TecoExporter** | v1.5.1 | Prometheus-style hardware telemetry collection service |

---

## Layer 8: Debug, Profile, Validate, and Custom Ops

### Debug / profile / validate

| Component | Version | CUDA analogue | Notes |
|---|---|---|---|
| **TecoGDB** | v3.1.0 | cuda-gdb | GNU GDB derivative for `.scpp` device code. Source-level breakpoints, **SPE focus switching** (`switchSPE`), inter-SPE focus mechanism, `target tecocore` core-dump loading, `printException`, `bt`, `info locals/args`, Kernel-PC / `setPc`. Limitation: "目前只支持调试所有SPE运行相同代码的程序" |
| **TecoGDB Visual** | v1.2.0 | — | GUI debugger |
| **SDPTI** | v1.7.0 (`libsdpti.so`) | CUPTI | Activity + Callback APIs, activity buffers, external correlation IDs |
| **TCPX** | v0.3.0 | **NVTX** | Event/range annotation library — not in the seed |
| **TSight CLI / GUI** | v1.12.0 / v1.9.0 | **Nsight** | Host + device parallelism profiling — not in the seed |
| **TCVS** | v1.6.0 | DCGM-diag | Hardware/software validation and stability suite — not in the seed |
| `sdaacfilt` | — | c++filt | Symbol demangler |
| `tcap_dllogger` | — | — | Structured training/inference logger; [gitee.com/tecorigin/tcap_dllogger](https://gitee.com/tecorigin/tcap_dllogger), Apache-2.0 |

### Custom-operator paths

| Path | Mechanism |
|---|---|
| **SDAA Extension** | PyTorch custom-op mechanism mirroring `torch.utils.cpp_extension`: write `.scpp` kernel → C++ wrapper → `TORCH_LIBRARY` registration → setuptools build → optional `torch.compile` and `autograd` support. **`load_inline` support added in v3.2.0** |
| **AbstractPluginOp** | TecoInferenceEngine plugin base class (`InferOutputShape` + `Enqueue`) → `libteco_ops_plugin.so` + `libTecoInferPlugin.so` |
| **TVM Relay registration** | teco-ops registers custom ops through TVM Relay IR; plugin sources built with `g++` for TVM header compatibility |
| `SDAAPluggableAllocator` | `torch.sdaa.memory.SDAAPluggableAllocator` — pluggable device allocator, CUDA-parity |

---

## Layer 9: Cluster Platform

**太初算力管理服务平台** — a multi-tenant cluster scheduler. Documentation mentions Jupyter / VS Code access, Megatron-DeepSpeed training, **TVM + Triton** small-model serving, FasterTransformer large-model serving, and GPFS / GlusterFS storage. **VENDOR CLAIM** — not corroborated by any retrievable technical manual.

---

## Precision and Numerics Controls — they leak hardware constraints

These environment variables are the most useful indirect evidence about the matrix unit and should be read alongside the hardware investigation:

| Variable | Effect |
|---|---|
| `TORCH_SDAA_CONV_USE_FP32` / `TORCH_SDAA_CONV2D_BACKWARD_USE_FP32` | By default Conv "算子内部将FP32输入转FP16进行计算" — FP32 conv inputs are internally down-converted to FP16 |
| `TORCH_SDAA_BF16_CLIP` | Clips bf16 GEMM inputs to **±65407** — essentially the FP16 maximum. **bf16 matmul is emulated on the FP16 unit** |
| `TORCH_SDAA_CONV_WEIGHT_CHWN` | Opt-in **CHWN** weight layout for speed — an unusual layout implying FU data-layout preferences |
| `TORCH_SDAA_FALLBACK_OPS` / `TORCH_SDAA_RUNTIME_AUTOFALLBACK` | Per-op CPU fallback |
| `TORCH_SDAA_LOG_LEVEL`, `TORCH_DEVICE_BACKEND_AUTOLOAD` | Logging / autoload |

Training supports FP32 and FP16 (AMP with weight backup + loss scaling). **Inference in TecoPyTorch is FP32-only**; TecoInferenceEngine converts everything to FP16.

---

## Openness Assessment

### Genuinely open

- **[PaddlePaddle/PaddleCustomDevice `backends/sdaa`](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa)** (Apache-2.0) is the single most valuable independent artifact: ~115 kernel `.cc` files, a `sdaac_ops/` directory of custom `.scpp` kernels, a `pr_ci_sdaa.sh` CI script, CMake linkage against the whole proprietary library set. **Tecorigin does not control this repo**, and its hardware CI implies a real device fleet.
- **[Tecorigin/teco-ops](https://github.com/Tecorigin/teco-ops)** (BSD-3-Clause) is a well-structured operator-development kit: layered `interface`/`ual` architecture, real `.scpp` kernels, `SDAAC_examples` covering DMA / RMA / SIMD / atomics / broadcast / matmul / transpose, a protobuf-driven C++ test harness, PyTorch bindings, TVM Relay plugin examples, and a **CUDA reference implementation directory (`cuda/`) used as the accuracy baseline** — a telling detail about how Tecorigin validates operators.
- Gitee: teco-al (BSD-3), teco-torch (BSD-3), teco-paddle (Apache-2.0), modelzoo-old (BSD-3), teco-generative-ai (BSD-3), tcap_dllogger (Apache-2.0), plus sdcops and thirdparty_llm with **no license declared**.

### Closed

**Every compiler, library, runtime, driver and tool binary.** TecoCC, PCXAC, teco_inductor, TecoDNN/BLAS/RAND/LMK/CUSTOM, TCCL, SDAARuntime, SDAADriver, TCML, TecoSMI, TecoExporter, TecoGDB, SDPTI, TCPX, TSight, TCVS — all proprietary, distributed only as `.deb`/`.rpm`/`.runfile`/Docker tarball from `mirrors.tecorigin.com` and `jfrog.tecorigin.net`.

### Documentation

Unusually thorough and versioned for a Chinese AI-chip vendor: **17+ manuals**, Chinese-only, with per-release changelogs going back to v1.0.0 and embedded instructional video assets. SDAA C is at v3.3.0 (released 2026-07-23). Contrary to the seed, the portal **is** machine-retrievable — via an undocumented REST API returning Yjs CRDT binary (see search-results.md).

### Traction

Small. GitHub repos 0–4★; Gitee 0–31★ (Teco-AL's 157 forks are WAIC-competition artifacts, not adoption). The most active GitHub repos were created **2026-04-13** for a WAIC developer competition ([github.com/tecorigin-waic](https://github.com/tecorigin-waic)), so recent commit velocity **overstates** public SDK cadence.

Meanwhile the *documentation* release cadence (v2.0 → v3.2 across 2025, TecoPyTorch tracking PyTorch 2.7.1, Teco-vLLM gaining PD-separation and EPLB across 2025-09 → 2026) indicates a real, actively engineered internal stack. **The public-ecosystem signal and the internal-engineering signal point in opposite directions; report both.**

### Negative checks

| Check | Result |
|---|---|
| PyPI `torch-sdaa`, `torch_sdaa`, `tecovllm`, `teco-vllm`, `tecoops`, `paddle-custom-sdaa` | **all 404** |
| Hugging Face `?author=tecorigin` | **`[]`** — no organization |
| Public manuals for TecoAL / TecoDNN / TecoBLAS / TCCL / TCVS / TecoLMK / TecoExporter / TecoCUSTOM / TecoRAND / TCML | **404** — named only in the install manual |
| English documentation | **none** — Chinese-only throughout |

---

## Sources

- [环境安装手册 v3.2.0](http://docs.tecorigin.com/release/software_installation) — authoritative component inventory, versions, host CPU/OS matrix, MLNX_OFED / Open MPI
- [SDAA C 编程指南 v3.2.0 (latest v3.3.0)](http://docs.tecorigin.com/release/sdaac) — kernel language, intrinsics, TecoCC options, device restrictions
- [PCX 编程指南 v1.2.0 (PCX ISA v1.0.0)](http://docs.tecorigin.com/release/pcx) — the virtual ISA
- [SDAARuntime 用户手册 v3.2.0](http://docs.tecorigin.com/release/sdaart) — runtime API, device props, error enum
- [TecoPyTorch 用户手册 v3.2.0 (PyTorch 2.7.1)](http://docs.tecorigin.com/release/torch2.7) — teco_inductor, AMP, DDP, SDAA Extension, env vars
- [TecoPyTorch 用户手册 (PyTorch 2.4)](http://docs.tecorigin.com/release/torch_2.4)
- [TecoPaddle 用户手册 v3.2.0](http://docs.tecorigin.com/release/tecopaddle)
- [Teco-vLLM 用户手册 v3.2.0](http://docs.tecorigin.com/release/teco_vllm) — quantization, TP/PP/EP/DP/EPLB, PD separation, Ray
- [Teco-Megatron-LM 用户手册 v3.2.0](http://docs.tecorigin.com/release/teco_megatron_lm) — parallelism, model recipes
- [TecoInferenceEngine（小模型）用户手册 v3.1.0](http://docs.tecorigin.com/release/tecoinferenceengine) — TVM-based four-layer architecture
- [TecoSMI 用户手册 v1.15.0](http://docs.tecorigin.com/release/tecosmi)
- [TecoGDB 用户手册 v3.1.0](http://docs.tecorigin.com/release/tecogdb) / [TecoGDB Visual v1.2.0](http://docs.tecorigin.com/release/tecogdbvisual)
- [SDPTI 用户手册 v1.7.0](http://docs.tecorigin.com/release/sdpti) / [TSight GUI v1.9.0](http://docs.tecorigin.com/release/tsight)
- [teco-ops (BSD-3-Clause)](https://github.com/Tecorigin/teco-ops) — `.scpp` kernels, SDAAC_examples, TVM Relay plugin, CUDA baseline
- [teco-ops README](https://github.com/Tecorigin/teco-ops/blob/main/README.md) / [README_OP.md](https://github.com/Tecorigin/teco-ops/blob/main/doc/README_OP.md) / [README_PLUGIN.md](https://github.com/Tecorigin/teco-ops/blob/main/doc/README_PLUGIN.md)
- [PaddleCustomDevice SDAA backend (INDEPENDENT, Apache-2.0)](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa)
- [Committed run log with full stack banner](https://github.com/Tecorigin/modelzoo/blob/main/PyTorch/contrib/Classification/ACNet-master/scripts/acnet.txt) — library versions and `.so` names
- [Tecorigin/tecovllm-modelzoo](https://github.com/Tecorigin/tecovllm-modelzoo) / [Tecorigin/modelzoo](https://github.com/Tecorigin/modelzoo) / [Tecorigin/teco-modelzoo](https://github.com/Tecorigin/teco-modelzoo)
- [gitee.com/tecorigin/teco-al](https://gitee.com/tecorigin/teco-al) / [teco-torch](https://gitee.com/tecorigin/teco-torch) / [teco-paddle](https://gitee.com/tecorigin/teco-paddle) / [tcap_dllogger](https://gitee.com/tecorigin/tcap_dllogger)
- [Binary mirrors](http://mirrors.tecorigin.com/) and [jfrog artifactory](http://jfrog.tecorigin.net/artifactory/)
