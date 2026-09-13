# Stream Computing NeuralScale — Software Stack Investigation

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*

---

## Overview

The Stream Computing stack is a **deliberate and fairly complete CUDA clone at the low level**
(SHC / `<<< >>>` / `stcc` / `stcMalloc` / `stc-smi` / `stc-gdb` / `stcpti`) sitting under an
**MLIR + IREE + torch-mlir–derived graph compiler (MLTC)** and a **proprietary LLM serving framework
(STC_LLM)**. The **entire product stack is closed source**; the only genuinely open artifacts are adjacent
— the RISC-V matrix-extension spec and toolchain forks, a PaddlePaddle model zoo, and an Apache-2.0 backend
contributed into ByteDance's benchmark.

**Delivery vehicle:** everything ships as **STCRP** (Stream Computing System Reference Platform). The
current public release is **STCRP V1.12.1**. There are no public SDK repositories; packages are obtained
from vendor technical support as `.deb` / `.rpm` repo packages plus Python wheels.

**STCRP V1.12.1 component versions** `confirmed`:
HPE 1.9.7 · stc-dkms 1.9.6 · stc-kernel-common 1.9.6 · hpert 1.4.3 · hpert-dev 1.4.3 · **stcc 1.9.6** ·
stc-smi 1.8.5 · stc-prof 1.2.12 · stc-gdb 1.8.3 · stc-vprof 1.8.0 · hpe-example 1.9.1 ·
**HPE Python 1.4.1** · **MLTC 1.6.1** · SNQ 1.0.0 · **STC_LLM 1.3.2** · **STC_LLM_DNN 1.3.1** ·
**STC_IE 1.6.1** · SNC 1.0.1.
OS support: **Ubuntu 22.04, Ubuntu 25.04, Kylin V10 (银河麒麟)**. Python **3.10** required for MLTC.
Firmware compatibility: MCU 10.0.14 / NPU-ctrl 10.3.7.

> **Generation hazard.** The 2021–2023 stack was **TensorTurbo** (TVM-based, LLVM backend, stcDNN operator
> library) with **STC_DDK**. The current stack is **MLTC** (MLIR/IREE) with **STC_IE** / **STC_LLM**.
> Third-party and pre-2024 sources still describe TensorTurbo; mixing the two produces a wrong stack
> diagram. See "Historical stack" below.

---

## Layer 13 — Framework Integration

| Component | Detail | Open? | Confidence |
|---|---|---|---|
| **PyTorch (primary)** | Two routes: (a) `torch.compile(model, backend=mltc_backend)` from `mltc.mltc.frontend.torch_frontend`; (b) `from_torch(nn.Module \| ExportedProgram, ...)` → `.vmfb`. Supports `torch.export` with `Dim(...)` dynamic shapes on parallel/reduce axes and `import_symbolic_shape_expressions` | closed | confirmed |
| **ONNX** | `OnnxToStc().run(...)` frontend; **156 ONNX ops** supported; opset 19 examples; custom ops via domain `mltc.custom` | closed | confirmed |
| **TensorFlow** | `TfToStc().run(...)`; **106 TF ops** supported (TF1 GraphDef `.pb`) | closed | confirmed |
| **PaddlePaddle** | Level-I compatibility certification with Baidu; **`Paddle-STCNNE`** backend + **`STCPaddleModelZoo`** (open, ~20 models: ResNet / DenseNet / VGG / EfficientNet / ShuffleNet / SE-ResNet / DBNet, with CPU-vs-NPU accuracy deltas). *TensorTurbo-era artifact — requires TensorTurbo ≥1.10 + STC_DDK* | **open (repo)** | confirmed |
| **HuggingFace / LLMs** | Via **STC_LLM** (below) — loads HF `bin` / `safetensors` directly | closed | confirmed |
| **Ad-hoc app integration** | Documented worked example patching **Ultralytics YOLOv5** with a new `vmfb` backend type in `DetectMultiBackend`, alongside TensorRT / OpenVINO / Triton | — | confirmed |
| **Not present** | **No vLLM fork, no SGLang, no Triton (OpenAI) kernel language, no TVM in the current stack, no ONNX Runtime EP** | — | confirmed (absence across all SDK doc pages) |

---

## Layer 12 — Graph Capture

- **PyTorch:** TorchDynamo (`torch.compile` custom backend) and **`torch.export` → ExportedProgram**, then
  imported to a **Torch dialect** MLIR. The `from_torch` signature (`output_type=OutputType.TORCH`,
  `backend_legal_ops`, `extra_library_file_name`, `experimental_support_mutation`,
  `import_symbolic_shape_expressions`) is **byte-for-byte the torch-mlir `fx.export_and_import` API**, and
  the supported-op list is spelled in torch-mlir's internal C++ class names (**141 `Aten*Op` entries** such
  as `AtenConvolutionOp`, `AtenNativeLayerNormOp`, `AtenBmmOp`, `ConvertAtenMaxPool2dOp`). **Strong evidence
  that MLTC's PyTorch frontend is torch-mlir–derived** — though the vendor never states it. `confirmed`
- **TF / ONNX:** dedicated importers (`TfToStc`, `OnnxToStc`) producing the same STC graph-level MLIR; input
  shapes and dtypes are given on the command line (`-i name:1x3x224x224`; dtypes f64/f32/f16/i64/i32/i16/i8/ui*).

---

## Layer 11 — Graph Compiler: MLTC (Multi-Level Tensor Compiler)

MLIR-based, proprietary. `confirmed`

Design principles as stated by the vendor:

1. **Multi-level IR with progressive lowering** — each lowering step handles exactly one concern
   (multi-core execution model, bufferization, operator→instruction selection).
2. **Operator-agnostic tiling system (GOAT)** — hoists the common logic out of operator authoring, tiling
   over **LLB → MC → L1**. The vendor's rationale is the key sentence for this architecture: *"when writing
   operators for NeuralScale, most of the code is managing the multi-level storage hierarchy and the data
   movement between levels."* On a software-managed memory machine, this is the compiler's central job.
3. **Optimisations are independent, individually switchable passes** on specific IR levels.
4. **The trunk after graph optimisation is operator-agnostic** — no per-operator special-casing.
5. **Graph-level ops are atomic — there are deliberately no fused operators in the graph IR** (fusion
   happens below).

Named stages: **Graph Optimizations** (equivalence-preserving rewrites) → **Group Partition**
(locality-driven operator grouping for NeuralScale) → **GOAT tiling** → codegen.

### Dialect / op set

`stc.*` graph dialect with **164 ops**. The AI-specific ones reveal the target: `stc.attention`,
`stc.attention_lm`, `stc.attention_mask_add`, **`stc.kv_cache_load` / `stc.kv_cache_store`**, `stc.matmul`
vs **`stc.matmul_vme`** (VME-based high-precision matmul), `stc.quantize` / `stc.dequantize`,
`stc.im2col` / `stc.col2im`, and a **built-in collective family**: `stc.device_broadcast`,
`stc.device_split`, `stc.device_concat`, `stc.device_allconcat`,
`stc.device_reduce_sum/mean/max/min`, `stc.device_sync`.

### Output artifact and lineage

- Output is **`.vmfb`** ("STC MLTC fatbin"), compiled with `-arch=npu-v1` (the compiler also accepts
  `npu-v2`; see the hardware investigation — that is a flag, not a chip).
- **IREE lineage:** `.vmfb` is IREE's VM FlatBuffer container, and `--graph-partition-factor=N` is
  documented as "splitting into N **`dispatch.workgroup`s**" — IREE Flow-dialect terminology. Combined with
  torch-mlir's `OutputType`, **MLTC is an IREE + torch-mlir derivative**, a materially different lineage
  from Sophgo's TPU-MLIR. *(Inference from public artifacts; the vendor never names the upstreams or
  revisions — that provenance is **not disclosed**.)*

### Notable compile flags

`--high-precision` (swaps `matmul` → `matmul_vme`, Newton iteration for transcendentals),
`--bisection-reduce`, `--bisection-matmul`, `--enable-merge-attention`
(`matmul` → `matmul_batchinner`, attention-specialised, default on), `--attention-shrink-factor`,
`--graph-partition-factor`, **`--pipeline-partition-factor` / `--pipeline-partition-file`** (pipeline
parallelism across stages), `--manual-partition-file`, `--static-subkernel-file=<json>` (manual
operator-split strategy, e.g. `{"s0":[1,4,8]}`), `--dump-ir-{before,after}-all`.

### Extension and debug tooling

- **Custom-operator plug-in path:** author an ONNX node in domain `mltc.custom`, write a **shape-inference
  script** (static shape/dtype/layout inference at compile time) pointed to by `MLTC_CUSTOM_OPS_PATH`, and a
  compute implementation (`.h` + `.py`) registered via `MLTC_PLUGINS_PATH`.
- **CPU reference path:** `Simulator().run("model.mlir", "-i in.bin -o out.bin --dump-each-op-result
  --dump-dir=...")` runs the post-frontend MLIR on the host CPU as the golden reference.
- **Precision debugging:** `run_npu_data_analysis.py` + `run_precision_analysis.py` produce `cpu_vs_npu.csv`
  with per-node `nan/inf`, `errors/total`, **cosine similarity** (flagged below 0.95), first-divergent
  index, plus error scatter/histogram plots. Defaults `--atol 0.005 --rtol 0.2`.
- **Model-level performance:** `run_perf_analysis.py` emits per-core CSVs with `mcu_cycle`, `vme_cycle`,
  `mme_cycle`, `vec_cycle` (native RVV), `syn_cycle`, `mte_total_cycle`, instruction counts, per-pair
  parallel cycles, `mte_{pld,icmov,l12llb,llb2l1}_{cycle,byte}`, `l1_conflict_cycle`, `im_conflict_cycle`,
  `dcache_miss_cnt`, `icache_miss_cnt`.

---

## Layer 10 — Kernel Compiler: `stcc`

- `stcc` — "Stream Computing Heterogeneous C++ Compiler" — compiles **host and device code in one
  invocation** into a single executable, exactly like `nvcc`. `confirmed`
- **LLVM evidence:** the documented invocation is `stcc --rtlib=compiler-rt hello_world.hc -g -o hello_world`
  — `--rtlib=compiler-rt` is a Clang driver flag; the visual profiler integrates a **Clangd** language
  server for `.hc` files; the device runtime lives under `/usr/local/hpe/riscv32npu/` (RV32 target).
  Combined with the company's public **`riscv-stc/llvm-project`** fork carrying matrix-extension support,
  `stcc` is with high confidence a Clang/LLVM downstream. **The vendor does not state this, and no LLVM
  version is disclosed.**
- Device code is compiled to **RV32 + RVV v0.8 + the OP-VE custom extension**; disassembly in `stc-gdb`
  shows ordinary compressed RISC-V (`addi sp,sp,-16`, `auipc`, `jalr`, `c.*`).

---

## Layer 9 — Kernel Language: SHC (Stream Computing Heterogeneous C++), `.hc`

- **Fully C++17-compatible**, extended with CUDA-shaped syntax: `<<< >>>` kernel launch, `__global__`,
  `__device__`, and memory-space qualifiers `__local__` (L1 Buffer) and `__shared__` (LLB). Built-ins
  `CoreID`, `CoreNum`; barrier `sync()`. Headers `<hpe.h>`, `<npurt.h>`, `<asm_macro.h>`. `confirmed`
- **Direct ISA intrinsics and CSR macros** — the programmer writes the custom instructions by name and
  configures shape CSRs explicitly. From the vendor's own `matrix-multiply.hc`:

  ```c
  shape1 = DEFINE_SHAPE(LCOL_RROW, 1);
  shape2 = DEFINE_SHAPE(1, LCOL_RROW);
  CONFIG_VE_BC_CSR(shape1, shape2, 0, 0);
  memul_mm((__fp16 *)IM_BUFFER_START, local_left, local_right);  // MME matmul → Intermediate Buffer
  CONFIG_VE_CSR(DEFINE_SHAPE(1,1), 0, 0, 0);
  mov_m(local_out, (__fp16 *)IM_BUFFER_START);                   // MTE move IM → L1
  ```

  This is a *much* lower abstraction level than Triton or even CUDA C: the kernel author manages the
  L1 / IM / LLB hierarchy and the shape CSRs by hand. `confirmed`
- **HPE-Python** provides the same heterogeneous programming model from Python (separate install, wheel
  `hpe_python-1.4.1-cp310-...`).
- **No Triton, no TVM TensorIR, no auto-scheduler, no DSL** in the current stack. The only automation is
  MLTC's GOAT tiling framework.

---

## Layer 8 — Tensor / Inference API

- **`mltc` Python package**: `TfToStc`, `OnnxToStc`, `from_torch`, `optimize`, `Compiler`, `Executor`,
  `Simulator`.
- **`stc_ie.stc_runtime.executor.Executor(vmfb, device_ids, n)`** — the runtime executor, also used directly
  by LLM code; supports `get_inputs()` / `get_outputs()` and dict-in / dict-out `run()`.
- **`STC_IE`** (v1.6.1) is the packaged "inference engine" wrapping compiled `.vmfb` models.
- Legacy equivalent: **STC_DDK** ("AI Convertor + AI Executor") in the TensorTurbo era.

---

## Layer 7 — Runtime: HPE (Heterogeneous Programming Engine)

- **`hpert`** — host-side runtime library: device management, memory management, execution control,
  **stream management**.
- **`npurt`** — device-side runtime: on-device `printf`, memcpy helpers.
- **API surface is a CUDA runtime clone**, verbatim from profiler output: `stcMalloc`, `stcMallocHigh`,
  `stcFree`, `stcMemcpy` (`stcMemcpyHostToDevice` / `DeviceToHost`), `stcConfigureCall`, `stcLaunchKernel`,
  `stcModuleLoadData`, `stcRegisterFatBinary`, `stcUnregisterFatBinary`, `stcDeviceSynchronize`,
  `stcRuntimeGetVersion`.
- **`stcpti`** — CUPTI analogue: profiling-data collection API exposing MME/MTE/VME-granularity cycle counts.
- **`STCML` (`libstcml.so`, `<stcml.h>`)** — NVML analogue for management: `stcmlInit` / `Shutdown`,
  `stcmlDeviceGetCount`, `GetHandleByIndex` / `ByPciBusId`, `GetName/Uuid/Serial/Power`,
  `Get/SetFrequency` (east/west domains), `Get/SetVoltage`, temperature get / set-alert / set-shutdown,
  `EnableMonitor`, `GetBoardStatus`, `StartBoard`, firmware-version getters, `stcmlErrorString`; 14 error
  codes `STCML_SUCCESS … STCML_ERROR_OTHER`.
- **Resource model:** a card = `device`; a device holds 4 `NPC Cluster`s; a cluster holds NPCs. Kernels are
  addressed as `[device x, cluster y, core z]`. **The cluster is the allocation and isolation unit**
  (`/dev/stc0c0…c3`, `stc-smi -q -c N`, container `--device` per cluster).

### Tooling — all proprietary, all NVIDIA-shaped

| Tool | NVIDIA analogue | Notes |
|---|---|---|
| `stc-smi` | `nvidia-smi` | Summary table; `--query-npu=index,name,serial,power.draw,temperature.npu,utilization.npu,memory.total,memory.used --format=csv`; per-cluster view; DVFS set; reset; thermal monitor; firmware upgrade |
| `stc-topo` | `nvidia-smi topo -m` | Prints a matrix with **`GPU0…GPU7` row labels**; classes LOC / PIX / PXB / PHB / SYS plus CPU and NUMA affinity |
| `stc-gdb` | `cuda-gdb` | Fully GDB-compatible; simultaneous host+device debugging; source- and instruction-level stepping on device; attach; extension commands `stc focus [device D cluster C core N]` and `stc info` (per-NPC PC + status) |
| `stc-prof` | `nsys` | `record` / `dump` (Chrome-tracing JSON, viewable in Perfetto) / `summary` / `detail --sub-module=DEFAULT,VME,MTE,MME,PAL,MCU,MEMORY,MEMORY-COL`. Reports **VME-CU vs VME-VEC** (custom vs native-RVV cycles), sysDMA DDR↔LLB byte counts and bandwidth, and **L1 / IM bank-collision counters** |
| `stc-vprof` | `nsight` | Java / JDK-11 GUI; local + remote (SSH) projects; timeline; `.hc` editor with Clangd; per-NPC parallelism bar charts; CSV export |
| `stc-hpaa` | — | "Half-Precision Accuracy Analysis" (TensorTurbo era) |
| `NPU-Viewer` | — | Standalone hardware qualification: PCIe bandwidth, DDR bandwidth, TOPS, stress test |
| `p2p_perf` | `p2pBandwidthLatencyTest` | Card-to-card DDR/LLB transfer benchmark (tarball named `p2p_perf_rdma`) |
| `stcqual` | — | Factory qualification suite |

---

## Layer 6 — Driver / Firmware

- **`stc-dkms`** builds **`stc.ko`** via DKMS (so it tracks kernel versions without manual rebuilds);
  installed to `/lib/modules/$(uname -r)/updates/dkms/stc.ko`. `confirmed`
- **`stc-kernel-common`** ships udev rules `/lib/udev/rules.d/70-stc.drv.rules` creating the device nodes.
- Device nodes: **`/dev/stc0`** (device), **`/dev/stc0c0 … /dev/stc0c3`** (per-cluster), **`/dev/stc0ctrl`**.
  `lspci` reports kernel driver `stc`.
- **Firmware:** two signed images — **NPU-ctrl firmware** (SoC bring-up) and **MCU firmware** (card power).
  Upgradeable in-band via `stc-smi -u` / `--mcu-upgrade` with `.sdux` files (root required).
- **Virtualisation:** **no SR-IOV** (1 PF, no VFs). PCIe **passthrough to KVM guests is supported**
  (firmware V1.3.3+); host and KVM simultaneously from V1.3.4+.
- **Install layout:** `/usr/local/hpe/{bin,lib,include,conf,example,java,libexec,riscv32npu,share}`.
  Distributed as repo packages (`hpe-repo-ubuntu2204-1-9-local_1.9.7_amd64.deb`, Kylin RPM) with GPG keys;
  `hpe-host` (driver only) vs `hpe-simple` (userspace) split for the container flow.

---

## Layer 5 — Communication / Multi-Device

- **There is no collective-communication library** — no NCCL / RCCL / HCCL / CNCL analogue anywhere in the
  SDK. `confirmed (absence verified across all 22 SDK doc pages)`
- Multi-device parallelism is instead expressed **inside the compiler**: MLTC's `stc.device_broadcast`,
  `stc.device_split`, `stc.device_concat`, `stc.device_allconcat`,
  `stc.device_reduce_{sum,mean,max,min}`, `stc.device_sync`, plus `--pipeline-partition-factor` for
  pipeline parallelism.
- **STC_LLM implements tensor parallelism by pre-splitting weights at conversion time:**
  `convert_weight -n <NPU count>` produces a weight set valid **only** for that card count ("weights
  converted for 4 NPUs cannot be used on 2 NPUs").
- Transport is PCIe peer-to-peer, measured by the vendor's own tool at **~9 GB/s** (LLB destination) and
  **<1 GB/s** (DDR destination) — see the hardware investigation.

---

## Layer 4 — LLM Serving: STC_LLM (proprietary, *not* a vLLM fork)

- **Three backends selected by `--compiler`:**
  - **`stc_llm_dnn`** (default) — the **hand-written-kernel** path; the vendor states it "currently shows
    better inference performance". Mechanically it is a **C++ code generator**: it renders model source into
    `engine_dir` (default `.cache/stc_llm`) as `.cpp`, which is then built into an NPU executable
    (`render_cpp` controls regeneration). Weight conversion:
    `python -m stc_llm_dnn.tools.convert_weight -m <HF id> -p <src> -o <dst> -n <npus>`.
  - **`mltc`** — compiles the model through MLTC to `.vmfb` (`--mltc_weight_dir`, `--mltc_ht_file`).
  - **`IE`** — via STC_IE.
- **Servers:** `stc_llm.entrypoints.openai.api_server_new` (DNN path) and `...openai.api_server` (MLTC path).
  **OpenAI-compatible** `/v1/chat/completions` and `/v1/completions`, SSE streaming, `usage` with
  `cached_tokens`; works with the stock `openai` Python client and `curl`.
- **Features:** dynamic batching (`max_tasks`, default 256, one stream per task), **KV-cache segmentation**
  (`slot_number_per_segment`, default 256 tokens), **prefix caching** (`--enable_prefix_caching`),
  **DMA weight preload** (`dma_preload`, default on — overlaps weight copies with matrix compute),
  host-side offload of embedding (`embed_on_host`) and logits (`output_on_host`), sampling controls
  (temperature / top_p / top_k / logprobs / top_logprobs), `response_format: json_object`,
  **`--reasoning-parser`** (Qwen3, DeepSeek-R1) and **`--enable-auto-tool-choice` + `--tool-call-parser`**
  (Qwen3) for function calling.
- **Serve-time quantisation:** `quant_type ∈ {fp16, w8a8, w8a16}`; `compress_factor ∈ {128,64,32,16,8,1,-1}`
  (−1 = per-channel) for w8a16.
- **Observability:** native **Prometheus** metrics under the `stc_llm:` namespace —
  `avg_prompt_throughput_toks_per_s`, `avg_generation_throughput_toks_per_s`, `time_to_inference_count`,
  `first_token_time`, `task_duration`, `task_decoder_latency`, `avg_generation_latency`,
  `num_requests_running/swapped/waiting`, `prompt_tokens_total`, `generation_tokens_total` — with a
  documented Grafana dashboard flow.

### Layer 4b — Quantisation tooling

- **SNQ** (`stcnq`) — PTQ for small / ONNX models; emits an ONNX quantised model. `confirmed`
- **SNC** (`stcnc`, "Stream Computing Neural Compressor") — **built on Intel Neural Compressor**; PyTorch LLM
  quantisation; implements **SmoothQuant** with an α sweep (`alpha_list: [0.1…0.9]`), calibrated on
  **C-Eval**, emitting per-α activation-scale `.npy` sets (`act_scale_qkv_w.npy`,
  `act_scale_mlp_up_gate_w.npy`, …); CPU-only quantisation supported. `confirmed`

---

## Layer 3 — Deployment / Orchestration

- **`stc-k8s-device-plugin`** (Go; `main.go`, `Dockerfile`, `stc-device-plugin.yaml`) — Kubernetes DaemonSet
  device plugin; supports k8s 1.18 / 1.22 / 1.23 / 1.24 (1.18 recommended); auto-registers NPUs, tracks
  health, auto-enrols new nodes. `confirmed`
- **`npu-exporter`** — Python Prometheus exporter on port **9836**, systemd unit; metrics `npu_temp`,
  `npu_util`, `power_draw`, `power_total`, `memory_used`, `memory_total`, `cluster_count`, `cluster_util`,
  `cluster_memory_used`, `cluster_memory_total`, plus `npu_info` / `cluster_info` label sets. `confirmed`
- **Docker:** documented split — driver (`hpe-host`) on the host, userspace (`hpe-simple` + MLTC + tools) in
  the container; containers get NPUs via `--device /dev/stc0 --device /dev/stc0c0…c3 --device /dev/stc0ctrl`.
  Reference image ~10.7 GB. **No official public registry images.**
- **希姆云平台 (cloud platform)** — multi-cluster / multi-tenant scheduling, quotas, image registry, audit;
  **大模型平台** — model deployment, API gateway, model repository. Documented separately at
  docs.streamcomputing.com, but these pages are largely image-based with little extractable text —
  effectively **not disclosed** at specification level.

---

## Historical stack — TensorTurbo (must not be mixed with the current one)

The 2021–2023 stack was **TensorTurbo**, a **TVM-based** graph compiler with an **LLVM** backend and a
**stcDNN** operator library, plus **STC_DDK** (AI Convertor + AI Executor) and HPE 1.5.x. Its pipeline was:

```
Model Zoo → Inference API → Graph Compiler (Model Parser → customized NPU backend: optimisation + codegen)
          + stcDNN Lib + LLVM → loadable files → HPE (user-space HAL + kernel-mode driver) → NPU
```

It exposed **C++ and Python inference APIs for TensorFlow, PyTorch, MXNet and Keras**, importing framework
graph IRs into a unified TensorTurbo IR, then applying graph scheduling, operator scheduling and
intra-operator tiling. `confirmed` [CARRV'21 §3.3 + Fig. 4; ByteDance STC README naming TensorTurbo 1.11.0 /
STC_DDK 1.1.0 / HPE 1.5.1]

**The company has migrated TVM → MLIR/IREE (TensorTurbo → MLTC).** This is a clean, well-documented
compiler-lineage migration and is worth calling out in the survey — very few vendors have publicly changed
compiler families mid-product. The `STCPaddleModelZoo` repo is a surviving TensorTurbo-era artifact.

---

## Open Source vs Proprietary — precise accounting

**Proprietary / closed — the entire product stack:** HPE (hpert, npurt, stc-dkms / `stc.ko`,
stc-kernel-common), stcc, SHC, STCML, stcpti, stc-smi, stc-gdb, stc-prof, stc-vprof, stc-topo, MLTC,
STC_IE, STC_LLM, STC_LLM_DNN, SNQ, SNC, NPU-Viewer, p2p_perf, stcqual, k8s-device-plugin, npu-exporter, and
both firmware images. No public source repositories; **no published license terms**; distribution via
vendor support contact. `github.com/streamcomputing` is an empty account.

**Genuinely open:**

| Artifact | Org / URL | License | Status |
|---|---|---|---|
| `riscv-matrix-spec` | riscv-stc | **CC-BY-4.0** | 27★, AsciiDoc, last activity **Dec 2024**; v0.5 announced Nov 2024 |
| `riscv-matrix-project` (umbrella) | riscv-stc | BSD-3 (org default) | 39★, last activity **Mar 2024** |
| `llvm-project` fork (matrix support) | riscv-stc | Apache-2.0 w/ LLVM exception | last activity **Oct 2024** |
| `riscv-isa-sim` (Spike fork, matrix ext.) | riscv-stc | BSD-3 | last activity **Feb 2025** |
| `riscv-dnn` (DNN library on RVV + matrix) | riscv-stc | BSD-3 | last activity **Mar 2025** — newest in the org |
| `riscv-pvp` / `riscv-pvp-matrix` (verification platform) | riscv-stc | BSD-3 | Mar 2024 |
| `riscv-opcodes`, `riscv-test-env`, `riscv-openocd-matrix` | riscv-stc | upstream licenses | 2024 |
| `chipyard`, `riscv-boom` (SonicBOOM), `rocket-chip`, `barstools`, `berkeley-hardfloat`, `testchipip`, `cva6-wrapper`, `riscv-sodor` | riscv-stc | upstream (BSD / Apache) | 2022 – Sep 2024 |
| `STCPaddleModelZoo` | Stream-Computing | (unstated) | PaddlePaddle model zoo, TensorTurbo era |
| ByteMLPerf **STC backend** | bytedance/xpu-perf | **Apache-2.0, © 2023 Stream Computing Inc.** | Reproducible benchmark harness + results |

**Note for the survey:** none of the open repos are the shipping toolchain. The `riscv-stc` org is
research/standards work on the *open* RISC-V matrix extension, not the product's OP-VE custom extension, and
**the whole org has been quiet since March 2025**. Any repo-driven code investigation of the actual compiler
or runtime is impossible; the stack must be documented from vendor manuals.

---

## Sources

- [STCRP product overview](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP产品简介)
- [STCRP Release Notes V1.12.1](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP_Release_Notes)
- [STCRP installation guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP安装指南)
- [HPE usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南)
- [MLTC usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/MLTC使用指南)
- [STC_LLM usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/STC_LLM使用指南)
- [Python API (MLTC frontends, compile flags)](https://docs.streamcomputing.com/AI加速卡/开发者资源/Python_API)
- [C++ API (STCML reference)](https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API)
- [MLTC operator support (164 stc.* / 156 ONNX / 106 TF / 141 ATen)](https://docs.streamcomputing.com/AI加速卡/开发者资源/MLTC算子支持说明)
- [Glossary](https://docs.streamcomputing.com/AI加速卡/希姆计算术语表)
- [k8s-device-plugin guide](https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/k8s-device-plugin使用指南)
- [npu-exporter guide](https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/npu-exporter使用指南)
- [CARRV'21 §3.3 — TensorTurbo (historical stack)](https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf)
- [ByteMLPerf / xpu-perf STC backend README](https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md)
- [riscv-stc org repositories](https://github.com/orgs/riscv-stc/repositories)
- [riscv-stc/riscv-matrix-spec](https://github.com/riscv-stc/riscv-matrix-spec)
- [Stream-Computing/STCPaddleModelZoo](https://github.com/Stream-Computing/STCPaddleModelZoo)
