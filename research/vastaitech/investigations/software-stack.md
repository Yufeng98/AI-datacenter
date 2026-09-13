# VastaiTech (瀚博半导体) Software Stack Investigation

*as_of: 2026-08-08*
*chip: vastaitech*
*device_class: GPU-like Inference Accelerator + Video Codec (China, 瀚博 VastaiTech)*

---

## Overview

The compute target is called **VACC** throughout the stack — "Vastai Accelerated Computing"; the vendor never expands the acronym anywhere public. The 2026 delivery vehicle is **VVI (Vastai Versatile Inference)**, currently **VVI-26.02** (with VVI-25.12.SP2 still in circulation), obtainable only from the sales-gated developer centre.

The defining structural fact about this stack is that there are **two parallel runtime paths that share nothing above the driver**:

- **Build_In** — VastaiTech's own AOT-compiled inference stack: VAMC compiler → VastStream / VastStreamX runtime → VastGenX / VastGenServer serving. This is the mature path for CV/NLP and the original LLM path.
- **vLLM** — a PyTorch-shaped path: a `torch_vacc` device backend plus a `vllm_vacc` vLLM fork. This is the path used for all 2026 LLM/VLM work.

They do not share a compiler, an IR, a tensor type, or a serving front end. A model deployed on one path tells you almost nothing about its behaviour on the other.

```
                      Build_In path                    vLLM path
Framework      VastModelZOO / VastGenX          vLLM (vllm_vacc) / Xinference / MinerU
Graph capture  HF-modeling patch + AOT trace    none (--enforce-eager mandatory)
Graph compiler VAMC  (backend: tvm_vacc)        — (eager op dispatch)
Kernel level   VDSP ELF ops + VNNL (closed)     VNNL / torch_vacc kernels (closed)
Tensor API     VastStreamX (vsx) / VACM tensors torch.Tensor on the "vacc" device
Runtime        VACL / VACE / VAME + Stream      vllm_vacc + torch_vacc runtime
Collectives    VCCL                             VCCL
Mgmt           VAML → vasmi / VAProfiler / VASID / valogger; VastCloudNative (k8s)
Driver         vastai_pci.ko  (/dev/vastai0)
ISA            not disclosed
```

> **The single most important, most-often-missed fact about this stack: the graph compiler VAMC is a TVM derivative, not MLIR-based.** Every one of the 344 compile configs in the public model zoo sets `backend.type: tvm_vacc`.

---

## Layer 1 — Framework integration

| Component | What it is | Open? | Source |
|---|---|---|---|
| **vllm_vacc** | Fork/port of vLLM. Versions observed in the wild: **0.7.2**, **0.9.2**, **0.11.0**, **0.17.0**. Shipped as a closed `.whl` and as `harbor.vastaitech.com/ai_deliver/vllm_vacc:{latest,VVI-26.02,VVI-25.12.SP2}`. Ships an OpenAI-compatible server, tool calling (hermes / deepseek_v3 parsers), reasoning parsers, guided/JSON decoding, and MTP speculative decoding for DeepSeek | **Proprietary** (only Dockerfiles and recipes are public) | VastModelZOO/llm/common/docker |
| **torch_vacc** | Out-of-tree PyTorch device backend. Versions 1.3.0 → **1.3.3**. Paired with a **CPU** build of torch (`torch==2.7.0+cpu`, later `2.8.0+cpu`) — torch itself is not built against the device; `torch_vacc` supplies it | **Proprietary** | same |
| **vLLM recipe site** | `vllm-vacc.vastaitech.com` — a vllm-recipes-styled site, "Community-maintained recipes for VASTAI Tech VA16 VA10L VA1L", **31 recipes** (25 Qwen, 6 DeepSeek), vLLM **0.17.0** | Public docs | vllm-vacc.vastaitech.com |
| **VastModelZOO** | The reference model platform. **1000+ models**, 5 domains (CV / AUDIO / NLP / LLM / MLLM), 33 sub-categories, two runtimes. Contains the full VAMC compile YAMLs, per-model deploy docs, quantisation recipes and modified `modeling_*_vacc.py` files | **Apache-2.0** | github.com/Vastai/VastModelZOO |
| **VastStreamX-Samples** | C++/Python samples for the VastStreamX API: 42 sample directories, 20+ CV/NLP tasks, VDSP built-in ops, custom ops, JPEG/H.264/H.265 codec, video capture, AI+codec pipelines, card telemetry | **MIT** | github.com/Vastai/VastStreamX-Samples |
| **xinference_vacc** | Xinference integration — LLM / Embedding / Rerank / VLM, **vLLM engine only**. Adds `VACC_VISIBLE_DEVICES`, `VNNL_CONV1D_DLC=1` (Qwen2-Audio). Upstreamed fixes to Xinference and xoscar | **Apache-2.0** (Dockerfiles) | github.com/Vastai/xinference_vacc |
| **MinerU fork** | Document-parsing VLM pipeline on VACC. Zero code change to upstream MinerU 2.7.3. Notably, **upstream OpenDataLab documents VastAI as a supported accelerator** — the only genuinely third-party-hosted VastaiTech integration doc found | **GPL-3.0** (fork); upstream doc independent | github.com/Vastai/MinerU |
| **VastGenX** | Proprietary LLM/VLM serving tool for the Build_In path: `vastgenx serve --model … --llm_devices "[0..7]" --vit_model … --vit_devices "[4]"`. Devices are **dies**, and VLM vision and language towers are placed on *different dies*. OpenAI-compatible API plus `vastgenx webui` | **Proprietary** | VastModelZOO/tools/vastgenx |
| **VastGenServer** | Proprietary text2vec / embedding serving (Build_In path) | **Proprietary** | VastModelZOO README |
| **FFmpeg + VAAPI plugins** | Standard media interface for the codec blocks; VastStream ships FFmpeg VAAPI plugins | Plugin, closed | vastaitech.com/software/vaststream |
| **Evaluation** | `evalscope` integration scripts for LLM/VLM accuracy and performance; `vllm bench serve`; OmniDocBench for OCR | Apache-2.0 scripts | VastModelZOO/tools/evalscope |

**Compiler frontends accepted** on the Build_In path: HuggingFace transformers, ONNX, PyTorch/TorchScript, **Keras (`.h5`, natively)**, TensorFlow, Darknet, and TVM Relay — from a census of `frontend.type` across 344 shipped configs.

---

## Layer 2 — Graph capture

**Build_In: there is no runtime capture.** The model is traced ahead of time from a **patched** HuggingFace modeling file. Every LLM except LLaMA requires the user to add to the weight directory:

- a `config_vacc.json` with an `auto_map` pointing at a vendor-modified `xxx_modeling_xxx_vacc.py`
- `"_attn_implementation": "eager"`
- `"insert_slice": true`

and the modeling file also gains a `quantize()` method for per-channel INT8. This is a **source-level** capture mechanism — invasive, and the main portability tax of the Build_In path: every new model architecture needs a vendor-authored modeling patch before it can be compiled.

**vLLM: no capture at all.** `--enforce-eager` is required in every documented command, and the MinerU integration states it as a rule: "注意在执行任意与vllm相关命令需追加 `--enforce_eager` 参数". There is no CUDA-Graph or `torch.compile` equivalent. `FUSE_ALL_DECODER_LAYERS=0` is an environment variable used to *disable* a decoder-layer fusion optimisation for long-context runs, implying the fusion that does exist is a runtime option rather than a compiled graph.

---

## Layer 3 — Graph compiler: VAMC

**VAMC = Vastai Model Compiler.** Invoked as `vamc compile <config.yaml>` (and `vamc quant` for the calibration flow). Distributed as a closed wheel (`vamc-xxx.whl`). Python 3.8 is required for the classic CV flow.

### The backend is an Apache TVM derivative

Every one of the 344 compile configs in VastModelZOO sets `backend.type: tvm_vacc` (594 occurrences across the tree), and the calibration dataset loader is `dataset.type: tvm`. The compiled artifact is a TVM-style module prefix — `deploy_weights/<name>/mod` — consumed by the runtime as `--model_prefix .../mod`. This places VastaiTech in the TVM-lineage camp (alongside e.g. Enflame's TopsGraph-era tooling) rather than the MLIR camp that most 2024+ Chinese accelerator compilers occupy.

### Config schema observed across the model zoo

```yaml
name: Qwen2-7B-fp16-tp4-1024-2048
frontend:
  checkpoint: /path/to/Qwen2-7B
  type: huggingface | onnx | pytorch | keras | tensorflow | darknet | tvm
  dtype: fp32
  shape: { input_ids: [[1024],[2048]] }        # multiple STATIC shapes
  model_kwargs:
    tp: 4                 # TENSOR PARALLELISM IS A COMPILE-TIME PROPERTY
    model_arch: vacc      # load modeling_vacc.py / config_vacc.json
    b2s: true             # batch-to-sequence; raises throughput, hurts latency
    align_qkv: true       # when num_key_value_heads % tp != 0
    attention_split_num: 16|32|96
    ffn_split_num: 4|16|48
    build_visual / llm_build            # VLM two-tower builds
graph:
  extra_ops: { type: insert_odma | null } # inserts explicit output-DMA nodes
backend:
  type: tvm_vacc
  dtype: fp16 | int8 | fp8
  merge_params: true
  compile:
    cluster_mode: 0|1                   # hardware cluster partitioning
    data_transport_mode: 1|3
    data_type: 0 (fp16) | 2 (int8)
    opt_level: 2
    mem_inplace: true|false
    output_ddr: [-1, 1]                 # force these outputs to off-chip DDR
    output_layout: ...
    enable_graph_partition: true
    enable_float_to_half: true
    gather_data_vccl_dsp_enable: true   # run VCCL gather on the VDSP engines
    quantize_dsp_ops / nearest_interpol_on_dsp   # push ops to VDSP
    skip_conv_layers / skip_matmul_layers        # keep layers out of quant
    requant_suppress / overflow_adaptive / split_convergence_points
    switch_lhs_rhs / stream_mode
  quantize:
    calibrate_mode: kl_divergence | percentile | mse | max
    quantize_per_channel: true
    weight_scale: power2
    calibrate_range / calibrate_chunk_by
dataset: { type: tvm|huggingface, path: ..., sampler: {...}, process_ops: [...] }
workspace: { path: ./deploy_weights/, workers: 4 }
```

Two things stand out architecturally:

1. **Tensor parallelism is compiled in, not scheduled at runtime** (`model_kwargs.tp`) on the Build_In path — the compiler emits a per-die partitioned model. The vLLM path by contrast takes `--tensor-parallel-size` at launch. Changing TP degree on Build_In means recompiling.
2. **The compiler owns memory placement** (`output_ddr`, `mem_inplace`, `data_transport_mode`, `insert_odma`), which is the strongest available evidence that the on-chip memory is a software-managed scratchpad rather than a hardware cache. See the hardware investigation, §5.1.

**The semantics of `cluster_mode`, `data_transport_mode`, `data_type`, `opt_level`, `output_layout`, `stream_mode`, `split_convergence_points` and `requant_suppress` are not disclosed.** The values are visible in 344 shipped configs; their meanings are documented only in the gated developer centre.

### Quantisation

FP16, BF16, INT8 PTQ (per-channel, four calibration modes — 151 configs use `percentile`, 131 use `kl_divergence`), **W8A16-GPTQ**, W4A16, INT4, **FP8**, and **FP4** (VA16, 2026). Calibration datasets used across the zoo: C4, CEval, Alpaca, ImageNet.

Worth noting: GPTQ calibration in `vamc_quant.yaml` can be run on `device: cuda:5` — i.e. **calibration runs on NVIDIA hardware and deployment on VACC**. `VACC_STACK_SIZE=256` must be exported when compiling models above 32 B.

---

## Layer 4 — Kernel compiler

**There is no public kernel compiler and no public kernel language.** This is a real gap relative to peers: no Triton port, no TileLang, no assembler, no intrinsics header, no disassembler.

What exists instead:

- **VDSP custom operators ship as pre-built ELF binaries** at `/opt/vastai/vaststreamx/data/elf/<op_name>` (e.g. `planar_argmax`, `brightness`, `norma_tensor_3ch`). A user loads one with `vsx::CustomOperator(op_name, elf_file)` and drives it by filling a packed C struct of device addresses and shapes, then calling `run_sync(tensors, config_bytes, output_info)`.
- **The toolchain that produces those ELFs is not distributed and not documented.** The vendor's own VLM work ("visual部分的旋转位置编码在VDSP自定义算子上实现") shows VastaiTech engineers writing new VDSP kernels; customers appear to receive only binaries.
- **VNNL** — inferred to be the low-level neural-network kernel library, a cuDNN analogue. It is visible **only** through two environment variables: `VNNL_MODEL_SYNC` (unset in the container entrypoint) and `VNNL_CONV1D_DLC=1` (required for Qwen2-Audio). No documentation, no header, no public mention by the vendor. **Flag this as inferred, not confirmed.**

---

## Layer 5 — Kernel / operator language as exposed to users

Users do not write kernels; they **compose** them.

- **Built-in VDSP ops**: `cvtcolor`, `resize`, `scale`, `flip`, `warpaffine`, `crop`, `copy_make_border`, `batch_crop_resize`.
- **Fusion ops selected by JSON**, e.g.

  ```json
  {"OpType": "FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR",
   "OpConfig": {"IimageFormat":"BGR888","Mean":[0.485,0.456,0.406],"Std":[0.229,0.224,0.225],
                "NormaType":"DIV255_MINUSMEAN_DIVSTD","ColorCvtCode":"BGR2RGB","ResizeType":"BILINEAR"}}
  ```

  Fusion ops are identified by `GetOpType() >= 100`. There is also a `BERT_EMBEDDING_OP`. These JSON files (`data/configs/*.json`, `build_in/vdsp_params/*.json`) are the "vdsp_params" argument threaded through every sample and every `vamp` invocation.
- **Custom ops** use the ctypes-struct + ELF mechanism described above.

---

## Layer 6 — Tensor API

**VastStreamX** (`import vaststreamx as vsx`; C++ `#include "vaststreamx/vaststreamx.h"`) is the modern high-level API — release **26.04** dated 2026-05-09, updated through 2026-07-20. The samples are MIT-licensed; the library itself ships as a closed `vaststreamx-xxx.whl` / `.bin`.

| Type | Role |
|---|---|
| `vsx.Tensor` / `vsx::Tensor` | Device or host tensor; `Shape()`, `GetContext()`, `GetDataAddress()`, `Clone(ctx)`, `TypeFlag::kUint16/…` |
| `vsx.Image` / `vsx::Image` | Typed image with `Format` (BGR_INTERLEAVE, RGB planar, YUV_NV12), width/height **pitch** |
| `Context::CPU()` / `Context::VACC(device_id)` | Explicit memory space; `dev_type == kVACC` test |
| `vsx.from_numpy(arr, device_id)` / `vsx.as_numpy(t)` | Explicit H2D / D2H |
| `DataManager` | Ref-counted buffer with custom deleter (zero-copy wrap of host memory) |
| `vsx::Model`, `vsx::ModelOperator` | Loaded `mod` artifact and its graph node; `GetInput/OutputShapeByIndex`, `GetMaxBatchSize` |
| `vsx::Operator`, `BuildInOperator`, `CustomOperator` | VDSP ops; `LoadOpsFromJsonFile`, `GetAttribute<AttrKey::…>` |
| `vsx::Graph`, `vsx::Stream` | Dataflow graph + execution stream; `AddOperators`, `RegisterModelOperatorOutput`, `Build`, `RunSync`, `process_async`/`get_output`/`close_input`/`wait_until_done`, `StreamBalanceMode::kBM_RUN`, `GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE` |
| `vsx.Card`, `vsx.Die` | Topology and telemetry: `uuid`, `card_type`, `dies`, `temperature`, `power`, `memory`, `utilization{ai, vdsp[], vdmcu[], vemcu[]}` |

Beneath it sits the original **VastStream** C SDK, documented publicly only as five libraries:

| Lib | Expansion | Function |
|---|---|---|
| **VACM** | Common Lib | Base utilities, log management, **device context management**, data management, **tensor management, memory management**, OS abstraction |
| **VACE** | Compute Engine | The operator library; executes single operators by function call, and loads/executes **VDSP operators** in combination with VACL |
| **VACL** | **Accelerate Language** | The inference API — load models, build a VDSPStream, run the stream |
| **VAME** | Media Engine | Image/video processing: media subsystem init, decode/encode operations |
| **VAML** | Management Library | Hardware monitoring and board status; the backing library for VASMI and VAProfiler |

The vendor's own one-line description of the SDK contents: "视频处理库、图像处理库、算子库、模型量化工具、编译器、运行时系统及硬件驱动" — video library, image library, operator library, quantisation tool, **compiler**, **runtime**, and driver.

**No API reference documentation is public for any of these libraries.** Every documentation route on `developer.vastaitech.com` returns `{"code":4000,"message":"错误的账户或秘钥，请检查"}`.

---

## Layer 7 — Runtime

### Build_In runtime

VACL/VACE plus the VastStreamX stream engine. Model artefacts are directories named by dtype and calibration method — `resnet50-int8-percentile-1_3_224_224-vacc/mod`, `bisenet-int8-kl_divergence-1_3_512_512-vacc/mod`, `mask2former-fp16-none-1_3_1024_1024-vacc/mod`.

**`vamp`** is the profiling/inference CLI, a `trtexec` analogue:

```
vamp -m …/mod --vdsp_params …json -i 8 -p 1 -b 22 -s [3,224,224] \
     --datalist list.txt --path_output out
```

### vLLM runtime environment variables

`VACC_VISIBLE_DEVICES` (the `CUDA_VISIBLE_DEVICES` analogue — selects **dies**), `VACC_LOG_LEVEL`, `VACC_STACK_SIZE`, `VACC_RT_MODELSAVE_EN`, `VACM_LOG_CFG`, `VLLM_VACC_KVCACHE_SPACE=16`, `VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0`, `LLM_MAX_PREFILL_SEQ_LEN`, `FUSE_ALL_DECODER_LAYERS`, `LLM_DEVICES`, `LLM_RECALL`, `VSX_DISABLE_DEEPBIND`, plus a family of `VASTAI_*` op switches: `VASTAI_OP_DEFORM_ATTN_CORE`, `VASTAI_WINDOW_PARTITION_OP`, `VASTAI_WINDOW_REVERSE_OP`, `VASTAI_ROLL_OP`, `VASTAI_PATCH_MERGING_OP`, `VASTAI_OP_MVIT_FOLD/UNFOLD`, `VASTAI_GEN_SINEEMBED_FOR_POSITION`.

The `VASTAI_*` family is worth noting on its own: these are per-operator enable switches for structural ops (window partition/reverse, roll, patch merging, deformable attention). Their existence implies that operator coverage is managed by hand, switch by switch, rather than by a general lowering.

### Collectives

**VCCL** — `VCCL_SOCKET_IFNAME`, `VCCL_MODEL_SYNC`, and the compile flag `gather_data_vccl_dsp_enable` (collective gather executed on the VDSP engines). **No public API, no documentation, no list of supported collectives.**

### Documented runtime limits

From `tools/vllm/usage_limits.md`:

- **max-concurrency 4** for essentially every model — DeepSeek-V3/R1 671B, all Qwen3 sizes, and even BGE-small embedding models.
- DeepSeek-V3: TP32 → 56 K input / 64 K context with MTP; TP32-PP2 → 100 K input / 128 K context without MTP.
- `min_p` sampling is **unsupported and errors out**.
- Requests exceeding the context window are silently intercepted rather than handled.
- Data parallelism (`--data-parallel-size`/`--dp`) is **not supported** (MinerU support matrix).

A max concurrency of 4 on a 671 B serving configuration is a very low figure and is the strongest self-disclosed signal that the vLLM port is functional but not performance-mature.

### Container / orchestration — VastCloudNative

A Kubernetes **Operator** covering driver install, container runtime, a **device plugin** (schedules and health-checks VastaiTech devices), and an **Exporter** (temperature, frequency, power, voltage, memory) for Prometheus. Docker integration is via an **OCI plugin**.

### Fleet management — VastDCManager

Built on VADriver + VAML:

- **VASMI** (`vasmi`) — the `nvidia-smi` analogue: `vasmi list`, `vasmi setconfig dpm=enable -d all`, `vasmi setcardmode <mode> -d <id>`. Reports device type, clock, temperature, per-process memory and codec cost; sets over-current and over-temperature protection.
- **VAProfiler** — runtime/hardware-cost profiler.
- **VASID** (System Integration Diag) — online environment/hardware/performance diagnosis with error reporting.
- **valogger** — logging daemon (`nohup sudo valogger &`).

---

## Layer 8 — Driver

- Installer: `sudo ./vastai_driver_install_xxx.run install --setkernelhook`.
- Kernel module **`vastai_pci`** (`lsmod | grep -i vastai_pci`). Device node **`/dev/vastai0`**; version string at `/dev/vastai0_version`. Card detection uses `lspci -d:0100`, i.e. **PCI device ID 0x0100**.
- Observed driver version string: `00.25.12.30 d3_3_v2_9_a3_1 a76bf37 20251230`.
- **Proprietary, out-of-tree, not upstreamed.** No source anywhere.
- x86_64 and aarch64 both supported; a `Dockerfile.noavx` variant exists for CPUs without AVX.

---

## Layer 9 — ISA

**Not disclosed.** No ISA manual, no assembler, no disassembler, no intrinsics, no public instruction reference. The only ISA-adjacent artefact visible to users is the opaque VDSP custom-op ELF. There is no published paper (ISCA / MICRO / Hot Chips / ISSCC / arXiv) describing the microarchitecture or the instruction set.

---

## Open source versus proprietary — the honest summary

**Open — `github.com/Vastai`, 4 repos, all active in 2026:**

| Repo | Licence | Created | Last push (as of 2026-08-08) | Contents |
|---|---|---|---|---|
| `VastModelZOO` | Apache-2.0 | 2022-12-07 | 2026-08-07 | Compile configs, deploy docs, patched modeling files, benchmarks (30★) |
| `VastStreamX-Samples` | MIT | 2025-08-26 | 2026-08-03 | API samples, VDSP op configs, measured perf tables (3★) |
| `xinference_vacc` | Apache-2.0 | 2025-09-10 | 2026-08-07 | Dockerfiles and integration glue (4★) |
| `MinerU` (fork) | GPL-3.0 | — | 2026-02-06 | Document-parsing pipeline on VACC |

Every one of these is **samples, recipes, configs and Dockerfiles**. **Not one line of the driver, compiler, runtime, kernel library, or collective library is open.**

**Proprietary and access-gated:** the driver; VastStream (VACM / VACE / VACL / VAME / VAML); VAMC; the VastStreamX libraries; VastGenX; VastGenServer; `torch_vacc`; `vllm_vacc`; VNNL; VCCL; `vasmi` / `vamp` / VAProfiler / VASID; the Docker images on `harbor.vastaitech.com`; and **all API reference documentation**.

**The developer centre is genuinely gated — verified.** `developer.vastaitech.com` serves an identical React SPA shell on every route. Its backend API is reachable — `GET /api/home/v0/home` returns solution blurbs — but every product and documentation endpoint (`/api/product/v0/summary`, `/api/product/v0/groups/1/summary`, `/api/asset/v0/files/download`) returns `{"code":4000,"message":"错误的账户或秘钥，请检查"}`. VastModelZOO's README confirms the policy: "需联系销售代表获取瀚博开发者中心版本权限". There is **no public API reference manual, no architecture whitepaper, and no conference paper** for this vendor.

---

## Comparative read

Relative to the Chinese-vendor peer group in this survey:

- **More open than** Cambricon (MagicMind is closed) and Huawei (CANN's kernel layer) in one narrow respect only — the *compile configuration surface* is fully public, so an outsider can read exactly which flags, quantisation modes and parallelism degrees the compiler supports for 344 real models. That is unusual and genuinely useful.
- **Much less open than** Sophgo, whose TPU-MLIR is a complete open compiler. VastaiTech publishes the configs but not the compiler that consumes them.
- **Notably behind on kernel programmability.** Peers offer Triton ports (Meta, several Chinese vendors), TileLang, or at minimum an intrinsics header. VastaiTech offers pre-built ELFs and no authoring path at all. A customer cannot write a new fused kernel for this hardware.
- **Behind on execution maturity for LLM serving.** Mandatory eager mode, no data parallelism, and max-concurrency 4 are all self-disclosed and all point the same way.

---

## Sources

Vendor primary:
- [VastStream SDK — VACM/VACE/VACL/VAME/VAML](https://www.vastaitech.com/software/vaststream)
- [VastStream (English)](https://www.vastaitech.com/en/software/vaststream)
- [VastCloudNative — k8s Operator, device plugin, Exporter](https://www.vastaitech.com/software/vastcloudnative)
- [VastDCManager — VASMI, VAProfiler, VASID](https://www.vastaitech.com/software/vastdcmanager)
- [vLLM × VastAI recipe site (31 recipes, vLLM 0.17.0)](https://vllm-vacc.vastaitech.com/)
- [Developer centre — gated; all product/doc routes return 错误的账户或秘钥](https://developer.vastaitech.com/)

Open source:
- [Vastai/VastModelZOO](https://github.com/Vastai/VastModelZOO)
- [VastModelZOO public model index (1000+ models)](https://vastai.github.io/VastModelZOO/)
- [VAMC compiler config schema (`backend.type: tvm_vacc`)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_config.yaml)
- [VAMC quantisation config (w8a16_gptq, calibration datasets)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_quant.yaml)
- [LLM compile guidance (seq-len multiple of 16, VACC_STACK_SIZE=256)](https://github.com/Vastai/VastModelZOO/blob/main/llm/README.md)
- [Qwen2 Build_In deploy doc (modeling_qwen2_vacc.py, insert_slice)](https://github.com/Vastai/VastModelZOO/blob/main/llm/qwen2/README.md)
- [Qwen2 VAMC compile YAML (tp: 4, b2s, model_arch: vacc)](https://github.com/Vastai/VastModelZOO/blob/main/llm/qwen2/build_in/build/hf_qwen2_fp16.yaml)
- [vLLM Dockerfile (VNNL/VCCL/VACM env vars, torch_vacc, vllm_vacc)](https://github.com/Vastai/VastModelZOO/blob/main/llm/common/docker/Dockerfile)
- [Dockerfile.arm (aarch64 wheels)](https://github.com/Vastai/VastModelZOO/blob/main/llm/common/docker/Dockerfile.arm)
- [vLLM examples and sampling-parameter support matrix](https://github.com/Vastai/VastModelZOO/blob/main/llm/common/examples/README.md)
- [vLLM usage limits — max-concurrency 4, min_p unsupported](https://github.com/Vastai/VastModelZOO/blob/main/tools/vllm/usage_limits.md)
- [DeepSeek-V3/V3.1 deployment — TP32, TP32-PP2, MTP](https://github.com/Vastai/VastModelZOO/blob/main/llm/deepseek_v3/README.md)
- [VastGenX serving tool (--llm_devices / --vit_devices are dies)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vastgenx/README.md)
- [GLM-OCR OmniDocBench — H800 vs VACC-VA16](https://github.com/Vastai/VastModelZOO/blob/main/vlm/glm_ocr/vllm/README.md)
- [ResNet deploy doc — vamc compile, vamp, keras frontend](https://github.com/Vastai/VastModelZOO/blob/main/cv/classification/resnet/source_code/keras.md)
- [Vastai/VastStreamX-Samples (release 26.04, 2026-05-09)](https://github.com/Vastai/VastStreamX-Samples)
- [model_base.hpp — Model/ModelOperator/Graph/Stream](https://github.com/Vastai/VastStreamX-Samples/blob/main/common/model_base.hpp)
- [custom_op_base.hpp — CustomOperator(op_name, elf_file)](https://github.com/Vastai/VastStreamX-Samples/blob/main/common/custom_op_base.hpp)
- [planar_argmax custom op — ELF + ctypes struct](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/vdsp_op/custom_op/argmax/argmax_op.hpp)
- [VDSP fusion-op JSON](https://github.com/Vastai/VastStreamX-Samples/blob/main/data/configs/detr_bgr888.json)
- [Vastai/xinference_vacc](https://github.com/Vastai/xinference_vacc)
- [Vastai/MinerU — driver/version matrix, DP unsupported, --enforce_eager mandatory](https://github.com/Vastai/MinerU)

Third-party:
- [OpenDataLab MinerU — VastAI acceleration-card documentation (independently hosted)](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/VastAI)
