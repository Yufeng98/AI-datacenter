# Sophgo Software Stack Investigation

*as_of: 2026-09-13*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*

> **2026-08-08:** a dated investigation section covering the **TPUv7 software stack** (torch-tpu, tpuv7-runtime, SG2260 collectives, the BM1690 compiler backend, and TPU-MLIR v1.27→v1.29) is appended at the end of this file. It **supersedes §"Layer 6: Communication" below for the BM1690/SG2260 generation.** The 2026-04-05 body is retained unchanged as the record of the BM1684X-generation stack.

---

## Overview

Sophgo's software stack for the BM1684X/BM1688 TPU line is called **SophonSDK** (also known as **SOPHONSDK**). It is a layered system closely analogous to NVIDIA CUDA: a compiler (TPU-MLIR) converts pre-trained models to a hardware binary (`.bmodel`); a runtime (BMRuntime/BMRT) executes `.bmodel` files; and low-level device management is handled by BMLib (wrapped in the `libsophon` package). High-level users interact through **sophon-sail** (Python/C++ wrapper) or directly through **BMRuntime C API**.

The key open-source artifact is **TPU-MLIR** (GitHub: `sophgo/tpu-mlir`), a full MLIR-based compilation pipeline from framework models to hardware binaries. This makes Sophgo's compiler stack more transparent and auditable than most Chinese AI chip vendors.

---

## Layer 1: Framework Integration

### LLM-TPU

- **Repo**: `sophgo/LLM-TPU`
- **Function**: Deploys open-source generative AI models (LLM/VLM) on BM1684X and BM1688 (CV186X) chips
- **Supported models**: Qwen2, LLaMA series, and growing list of HuggingFace models
- **Compilation path**: Uses TPU-MLIR to convert HuggingFace models to `.bmodel`; uses `tpu-runtime` inference engine interface for deployment
- **Deployment modes**: PCIE mode (host-attached BM1684X card) and SoC mode (embedded BM1688)

### vllm-tpu

- **Repo**: `sophgo/vllm-tpu`
- **Function**: Sophgo fork of vLLM for production LLM serving on BM1684X/BM1688
- **Enables**: Continuous batching, KV-cache management, OpenAI-compatible API over Sophgo TPU

### LLM-TPU_Lite

- **Repo**: `sophgo/LLM-TPU_Lite`
- **Function**: Lightweight LLM inference for edge deployment on "lite TPU" variants (BM1688/CV186X)
- **Target**: Embedded inference use cases where full vllm-tpu overhead is undesirable

### llmc-tpu

- **Repo**: `sophgo/llmc-tpu`
- **Function**: Fork of ModelTC/llmc for quantization-aware compilation and calibration targeting Sophgo hardware
- **Use**: INT4/INT8 quantization pipeline before TPU-MLIR compilation

### Framework Model Support (via TPU-MLIR)

The compiler accepts models from:
- **PyTorch** (via ONNX export)
- **ONNX** (native support)
- **TFLite** (native support)
- **Caffe** (native support)
- **TensorFlow, PaddlePaddle, MXNet** (convert to ONNX first)
- **HuggingFace LLMs** (direct: Qwen2, LLaMA series)

---

## Layer 2: Compiler / IR — TPU-MLIR

### Architecture

**TPU-MLIR** (`sophgo/tpu-mlir`) is the central open-source compilation infrastructure for all Sophgo TPU chips. It is built on LLVM/MLIR and provides a complete lowering pipeline from framework model to hardware binary.

**Compilation stages**:

```
Framework model (ONNX / TFLite / Caffe / PyTorch)
      ↓  model_transform.py
  .mlir (Top dialect — framework-neutral IR)
      ↓  model_deploy.py (quantization + lowering)
  .mlir (TPU dialect — chip-specific IR)
      ↓  codegen
  .bmodel (hardware binary for target chip)
```

### Key tools

| Tool | Function |
|------|----------|
| `model_transform.py` | Converts framework model → `.mlir` (Top dialect) |
| `model_deploy.py` | Lowers Top → TPU dialect + quantization + codegen → `.bmodel` |
| `bmrt_test` | Test/benchmark `.bmodel` execution on hardware |
| `bmodel_dis` | Disassemble `.bmodel` for inspection |

### Supported target chips

- BM1684X (cloud/edge, 32 TOPS INT8)
- BM1684 (Gen 3)
- BM1688 / CV186X (edge SoC, 16 TOPS INT8)
- CV18XX series (earlier edge chips)

### Quantization

TPU-MLIR includes built-in INT8/INT4 quantization via calibration tables. Users run `run_calibration.py` on a representative dataset to produce a calibration file, which `model_deploy.py` uses to quantize the model before lowering.

### MLIR Dialect Hierarchy

```
Top Dialect
  └── Canonical framework-neutral graph ops
      (TopConv, TopMatMul, TopRelu, etc.)

TPU Dialect
  └── Chip-specific lowered ops
      (TpuConv, TpuMatMul, TpuLoad, TpuStore)
      with explicit DMA scheduling and SRAM tiling
```

### Legacy Compiler

- **`sophgo/tpu_compiler`**: Earlier CVITEK AI compiler (pre-TPU-MLIR); used for CV18XX series; now deprecated in favor of TPU-MLIR

---

## Layer 3: Runtime — BMRuntime / BMRT

### BMRuntime (BMRT)

- **Library**: `libbmrt.so` (part of libsophon)
- **Function**: Reads `.bmodel` binary files and executes them on Sophgo TPU chips
- **API style**: C API (`bmrt_load_bmodel`, `bmrt_launch_tensor`, `bmrt_memcpy_s2d`, `bmrt_memcpy_d2s`)
- **Handles**: Model loading, tensor memory management, async execution launch, synchronization

### BMLib

- **Library**: `libbmlib.so` (part of libsophon)
- **Function**: Low-level device and memory management
- **Key APIs**: `bm_dev_request` (device handle), `bm_malloc_device_byte` (device memory alloc), `bm_memcpy_s2d` (host→device DMA), `bm_handle_free`, `bm_set_device_mem`

### BMCV

- **Library**: `libbmcv.so` (part of libsophon)
- **Function**: Hardware-accelerated computer vision preprocessing using BM1684X's VPP (Video Processing Pipeline) and JPU hardware
- **Operations**: Color space conversion, scaling/resize, affine transform, JPEG encode/decode, normalization, NMS, sort, BASE64 encoding

### Sophon-SAIL

- **Repo**: `sophon-ai-algo/sophon-inference`
- **Function**: High-level Python/C++ wrapper encapsulating BMRuntime, BMCV, BMDecoder, BMLib
- **Use**: Simplifies model deployment; users work with `sail.Engine` (model runner), `sail.BMImage` (image tensor), `sail.Decoder` (video decoder)
- **Languages**: Python (pybind11) and C++

### SOPHON-MW (Multimedia)

- **Components**: BM-OpenCV (hardware-accelerated OpenCV fork), BM-FFmpeg (hardware-accelerated FFmpeg fork)
- **Function**: Enables hardware H.264/H.265 decode/encode within standard Python/C++ multimedia pipelines
- **Integration**: Seamlessly passes decoded frames to BMCV/BMRuntime without host-side copies

---

## Layer 4: Driver / Firmware

### libsophon

- **Repo**: `sophgo/libsophon`
- **Function**: Unified package containing the kernel driver, BMLib, BMRuntime, BMCV, and header files
- **Platforms**: Linux (x86_64, aarch64); SoC mode for embedded BM1688

### Kernel Driver

- **Module**: `sophon.ko` (or `bmsophon.ko` in some releases)
- **Function**: PCIe BAR mapping, IOCTL dispatch, DMA engine control, interrupt handling
- **PCIe modes**: Standard PCIe device for BM1684X card; SoC-native mode for BM1688 embedded

### Container / Cloud Support

- **Docker**: Official Sophgo Docker images for SophonSDK (`sophgo/tpuc_dev`)
- **K8s**: Device plugin for TPU resource management in Kubernetes clusters

---

## Layer 5: Model Binary — `.bmodel`

The `.bmodel` is Sophgo's hardware binary format, analogous to NVIDIA's `.plan` (TensorRT) or Cambricon's `.cnbin`. It encodes:
- Compiled tensor operations in chip-native instruction format
- DMA command sequences (prefetch schedules)
- Weight data (quantized INT8/INT4 or FP16/BF16)
- Network topology metadata (input/output tensor shapes, names)
- Target chip identifier (BM1684X, BM1688, etc.)

The `.bmodel` is deployed and executed entirely by BMRuntime — users never write chip ISA-level code directly.

---

## Layer 6: Communication

> **⚠ Superseded 2026-08-08 for the BM1690/SG2260 generation.** This section remains accurate for the BM1684X/BM1688 stack (libsophon + BMRuntime). It is **wrong** for TPUv7: `sophgo/torch-tpu` ships a full `torch.distributed` collective backend for SG2260 over the SG-Link fabric. See the appended 2026-08-08 investigation.

Sophgo does not provide a collective communication library equivalent to NCCL, CNCL, or HCCS for multi-chip training on the BM1684X generation. The focus there is inference:

- **Multi-card inference**: Applications use multiple BM1684X cards independently (data parallelism over video streams) rather than model-parallel training
- **Host networking**: Standard TCP/IP or RoCE for distributed applications
- **No dedicated collective fabric**: No SOPHON equivalent of AllReduce/AllGather

---

## Sources

- [sophgo/tpu-mlir GitHub](https://github.com/sophgo/tpu-mlir)
- [sophgo/libsophon GitHub](https://github.com/sophgo/libsophon)
- [sophgo/LLM-TPU GitHub](https://github.com/sophgo/LLM-TPU)
- [sophgo/vllm-tpu GitHub](https://github.com/sophgo/vllm-tpu)
- [TPU-MLIR Introduction (v23.05 docs)](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/quick_start_en/html/01_introduction.html)
- [SophonSDK User Guide (v23.05)](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/SophonSDK_doc/en/html/sdk_intro/1_intro.html)
- [BMRuntime reference docs](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-runtime/reference_en/html/bmruntime/runtime.html)
- [BMLIB reference docs](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/libsophon/reference_en/html/2_bmlib_basic_concept.html)
- [DeepWiki sophgo/tpu-mlir](https://deepwiki.com/sophgo/tpu-mlir)

---

# Investigation 2026-08-08 — TPUv7 software stack (SG2260 / BM1690) and TPU-MLIR v1.27→v1.29

*Investigated 2026-08-08. Two distinct things happened since the 2026-04-05 baseline: (a) the survey discovered an entire second software stack it had missed — the TPUv7 stack for BM1690/SG2260 — and (b) TPU-MLIR shipped four releases in-window.*

## 1. There are two Sophgo runtime stacks, not one

The 2026-04-05 investigation described libsophon/BMRuntime as *the* Sophgo runtime. That is the **BM1684X/BM1688** stack. The BM1690/SG2260 generation has its own:

| Layer | BM1684X / BM1688 generation | BM1690 / SG2260 ("TPUv7") generation |
|---|---|---|
| Driver | `sophon.ko` / `bmsophon.ko` (in libsophon) | **`tpuv7-driver`** (v1.1.3 `.deb`) |
| Runtime | BMRuntime (`libbmrt.so`) + BMLib + BMCV, packaged as libsophon | **`tpuv7-runtime`** (v1.1.3 `.deb`) |
| Firmware path | libsophon-managed | **`/lib/firmware/tpuv7/`**, with per-chip configs `sc11_config.ini` and `sc11_config_chip2.ini` |
| Device management | BMLib API | **`tpu-smi`** |
| Profiling | `bmrt_test`, `bmodel_dis` | **`bigTpuProfile --arch BM1690`** |
| Compiler backend | `libbackend_bm1684x.so` | **`libbackend_bm1690.so`**; `third_party/nntoolchain/tpuv7_sha256.txt` |

The two config files are, incidentally, one of the three independent proofs that the SC11 FP300 card holds two BM1690 chips.

## 2. torch-tpu — the PyTorch path that refutes the "no collectives" claim

`github.com/sophgo/torch-tpu` is a **PyTorch device extension** targeting SG2260. It was not in the 2026-04-05 investigation at all, and it is the single most consequential omission, because it invalidates three standing repo claims at once.

Capabilities found:

| Capability | Evidence |
|---|---|
| **JIT mode (SG2260 only)** plus eager mode | README |
| **DeepSpeed ZeRO-1 and ZeRO-2 with CPU offload** | README |
| **Megatron-DeepSpeed tensor parallelism** | README |
| **Full `torch.distributed` collective backend** | `python/dist_test2260/` — `all_reduce`, `all_gather`, `all_gather_into_tensor`, `all_to_all`, `broadcast`, `reduce`, `gather`, `scatter`, point-to-point, and DDP |
| **Documented chip-to-chip topology** | `docs/quick_start/assets/3_c2c_topology.png`; `docs/developer_manual/source_zh/06_distribute.rst` |
| SG2260 build integration | `firmware_core/sg2260-toolchain.cmake` |

Consequences for the survey text:

1. *"Sophgo does not provide a collective communication library (no NCCL/CNCL/HCCS equivalent)"* — **refuted for SG2260.** A complete collective suite ships in torch-tpu.
2. *"No large-scale training capability"* — **refuted for SG2260.** ZeRO-1/ZeRO-2 plus Megatron tensor parallelism is a training stack, not an inference stack.
3. *"No proprietary scale-up interconnect"* — **refuted.** SG-Link is named in Sophgo's own filing and the C2C topology is documented here (see the hw-architecture investigation).

Caveat worth preserving: the *existence* of these APIs is confirmed from Sophgo's own repo. Their **performance, maturity, and production use are not** — no benchmark, no scaling study, and no named deployment was found.

## 3. vllm-tpu has moved generation

The 2026-04-05 entry described vllm-tpu as "Sophgo's fork of vLLM for BM1684X/BM1688". The current master (pushed Dec 2025) is a **vLLM v0.11.0 fork whose README targets SG2260**:

- "supports running LLaMa, Qwen, DeepSeek on Sophon TPU **SG2260**"
- **DeepSeek-V3 and DeepSeek-R1 in FP8**
- Llama-3.1-70B and Qwen2-72B with `--tp_size 2` (i.e. tensor parallelism across the two chips of one SC11 FP300 card)
- Installs `tpuv7-driver` / `tpuv7-runtime` 1.1.3 and `tpu-smi`
- Documents the per-chip memory carve-out (`share-mem-start=0x1e0000000`, `share-mem-size=0x1e20000000`) used to establish ~128 GB/chip

## 4. TPU-MLIR — BM1690 support is real but less open than BM1684X

The BM1690 backend exists and is exercised:

- `include/tpu_mlir/Backend/BM168x/BM1690.h` (and `BM1690E.h`) define the machine model.
- `libbackend_bm1690.so` and `third_party/nntoolchain/tpuv7_sha256.txt` ship the closed backend blob.
- Codegen is **ppl-based**.
- Checked-in regression artifacts `resnet50_v2_bm1690_f16_core8.bmodel` and `resnet50_v2_bm1690_f8e5m2.bmodel` confirm 8-core codegen and native FP8 E5M2.

But it is **not** as open as the BM1684X path, and the survey should say so:

- The public `llm_convert.py` still advertises only `bm1684x` / `bm1688` / `cv186x` as targets.
- The BM1690 MLIR dialect sits under `experimental/`.

So: the open compiler covers the datacenter part structurally, while the *user-facing LLM conversion flow* still points at the previous generation.

**Also present and undocumented in this survey:** TPU-MLIR carries backends for **SGTPUV8**, **SG2380**, and **BM1684X2**. These are separate parts with no corresponding chip pages. Flagged as an open item.

## 5. TPU-MLIR releases in-window (after the 2026-04-05 baseline)

Dates taken from the **releases atom feed** — the rendered releases page omits the year and is easy to misread.

| Version | Date | Contents |
|---|---|---|
| **v1.27** | 2026-04-29 | — |
| **v1.28.1** | 2026-06-16 | Auto-detection of LLM quantization; `batch_size` support in LLM inference; TPU Lang dump plus a **Rope** operator; **YOLOv26** post-process entry; a **Floor** activation op; mixed CUDA/CPU inference |
| **v1.29-beta.0** | 2026-06-23 | Titled around handling **`A_log` in Qwen3.5** |
| **v1.29** | 2026-07-01 | Narrow **conv2d hardware-margin fix** |

## 6. Repo activity check (2026-08-08)

Sophgo's public GitHub org is active: sophon-tools (2026-08-08), sophon-demo (2026-08-07), tpu-mlir (2026-08-04), LLM-TPU (2026-08-05), libsophon (2026-07-27). **No new BM1690- or SG2044-specific public repo appeared** — BM1690 support arrives inside the existing repos (tpu-mlir, vllm-tpu, torch-tpu, mcu) rather than as a new project.

## 7. Open items

- torch-tpu maturity: no benchmark, scaling study, or production reference found for ZeRO/Megatron on SG2260.
- SG-Link's software interface: the collective backend is visible, the transport spec is not.
- `llm_convert.py` gap: whether BM1690 LLM conversion is available through a non-public flow is unknown.
- SGTPUV8 / SG2380 / BM1684X2 compiler backends — undocumented parts, warrant separate investigation.

## Sources (2026-08-08)

- https://github.com/sophgo/torch-tpu — SG2260 PyTorch extension; JIT/eager; DeepSpeed ZeRO-1/2 CPU offload; Megatron TP; `python/dist_test2260/` collectives; `firmware_core/sg2260-toolchain.cmake`; `docs/quick_start/assets/3_c2c_topology.png`; `docs/developer_manual/source_zh/06_distribute.rst`
- https://github.com/sophgo/vllm-tpu — vLLM v0.11.0 fork for SG2260; DeepSeek-V3/R1 FP8; `--tp_size 2`; tpuv7-driver/tpuv7-runtime 1.1.3; `tpu-smi`; `bigTpuProfile --arch BM1690`; `/lib/firmware/tpuv7/sc11_config.ini` + `sc11_config_chip2.ini`
- https://github.com/sophgo/tpu-mlir — BM1690/BM1690E backends, `libbackend_bm1690.so`, `third_party/nntoolchain/tpuv7_sha256.txt`, FP8/8-core regression bmodels, SGTPUV8 / SG2380 / BM1684X2 backends
- https://github.com/sophgo/tpu-mlir/releases.atom — authoritative release dates
- https://github.com/sophgo/tpu-mlir/releases/tag/v1.28.1 — v1.28.1 release notes
- https://github.com/orgs/sophgo/repositories?sort=updated — org activity check

---

## Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*Verified directly against the GitHub REST API (`api.github.com/repos/sophgo/...`), not the releases atom feed, to avoid the year-parsing risk the 2026-08-08 pass flagged. No hardware/silicon disclosure this window — this is a pure SDK update.*

### TPU-MLIR advances from v1.29 to v1.30.2

The `/releases` endpoint only surfaces tags that carry a full GitHub Release object; several point releases are tags-only and are invisible to that endpoint, but commit dates (via `git/refs/tags` → `commits/{sha}`) confirm the true sequence:

| Version | Commit / publish date | Note |
|---|---|---|
| v1.29 (previously recorded) | 2026-07-01 | Narrow conv2d hardware-margin fix |
| v1.30-beta.0 | 2026-08-24 | tag only, no Release notes object |
| v1.30-beta.1 | 2026-08-26 | tag only |
| v1.30 | 2026-08-26 | tag only |
| v1.30.1 | 2026-08-31 | tag only |
| **v1.30.2** | **2026-08-31** (GitHub Release, `published_at` 04:24 UTC) | Full release notes published — see below |

**v1.30.2 release notes (verbatim structure, summarized):**
- **LLM serving**: added **chunk prefill** (e.g., 64K prefill split into 8K chunks) and **chunked decode** (incl. a Qwen3.5-specific variant); added MoE performance analysis (`llm_analyse.py`); FP8 LLM fixes; auto-generated random input/result comparison in `llm_convert`/`model_deploy`; MoE weight-logic dedup.
- **New model support**: **MiniCPM-V-4.6**, **Step3-VL**, **Falcon-Perception**, **LocateAnything-3B** (all multimodal); UnlimitedOCR converter for DeepSeek-V2-MoE VLM.
- **New ops/patterns**: `A16Gather` (W4A16/W8A16 on-the-fly dequant), `FlexAttention`/`FAttentionLse`, fused attention-decode kernel, `SliceAttentionChainPattern` for attention tiling, float-mix search for mixed-precision quantization.
- **Backend/platform**: continued **BM1684X2** enablement (RQ1 requant, RVTI, multi-core FC path); **updated BM1690/BM1690E backend**; merged **CUDA op support** (PRs 278/279, "plus more"); added c2c ops support; BM1688 user-IO-tag support extended (tags 3–7).
- No process-node, silicon, or spec disclosure — this release is entirely compiler/runtime.

**Assessment**: this is the most substantive TPU-MLIR release since the 2026-04-05 baseline — it is the first release to explicitly touch the BM1690/BM1690E backend again since the 2026-08-08 pass, and the first to mention CUDA-op support (relevant to any future NVIDIA-source-portability story) — but it remains a software-only change; no BM1690 spec value changes as a result.

### Repo activity (verified via `pushed_at`, 2026-09-13)

| Repo | `pushed_at` |
|---|---|
| sophgo/tpu-mlir | 2026-08-31 |
| sophgo/libsophon | 2026-08-10 |
| sophgo/mcu | 2026-08-31 |
| sophgo/LLM-TPU | 2026-09-04 |
| sophgo/sophon-demo | 2026-09-14 (essentially current) |
| sophgo/vllm-tpu | 2025-12-17 (**stale — no change since baseline**) |
| sophgo/torch-tpu | 2026-01-28 (**stale — no change since baseline**) |

vllm-tpu and torch-tpu — the two repos carrying the BM1690/SG2260 vLLM fork and PyTorch DeepSpeed/Megatron support — have **not been pushed to since before the 2026-04-05 research baseline**. This is worth flagging: the LLM-serving and distributed-training story for BM1690 has had no public code movement in over seven months, even while TPU-MLIR itself is actively developed.

### Searched and absent

- No new BM1690-successor or SG2044-specific repo.
- No STAR Market (科创板) filing update found for Sophgo in this window (searches returned only a year-old BIS Entity List story and an unrelated STAR Market listing for a different vendor, 燧原科技/Enflame).
- No Hot Chips 38 (2026-08-23 → 08-25) Sophgo talk.

### Sources added 2026-09-13

- [GitHub API — sophgo/tpu-mlir releases (live query, 2026-09-13)](https://api.github.com/repos/sophgo/tpu-mlir/releases)
- [GitHub — tpu-mlir v1.30.2 release notes](https://github.com/sophgo/tpu-mlir/releases/tag/v1.30.2)
- [GitHub API — sophgo/{tpu-mlir,libsophon,mcu,LLM-TPU,sophon-demo,vllm-tpu,torch-tpu} repo metadata (`pushed_at`, live query 2026-09-13)](https://api.github.com/repos/sophgo/tpu-mlir)
