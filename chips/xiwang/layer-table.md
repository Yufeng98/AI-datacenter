# Xiwang (曦望) Software Stack Layer Table

*as_of: 2026-08-08*
*prior revision: 2026-04-05*
*chip: xiwang*
*device_class: AI Accelerator (China, 曦望)*

**Stack brand: SIRE — "Sunrise Integrated Running Environment"** (软硬协同 · 高度兼容 · 卓越效能). Programming model: **TANG**. Runtime: **TangRT**. Collectives: **PCCL**. All four names became public after the 2026-04-05 baseline, which recorded them as undisclosed.

| Layer | Component | Name / Details | Open Source | Confidence |
|-------|-----------|---------------|-------------|------------|
| Framework | PyTorch integration | PyTorch device backend under SIRE; vendor claims auto-tracking of the latest upstream release | Partial (FlagCX torch plugin, 2026-06-01) | Medium-High |
| Framework | vLLM | Vendor-stated; Sunrise listed as a supported vendor in `vllm-plugin-FL` | Yes (vllm-plugin-FL) | High |
| Framework | SGLang | Vendor-stated | No | Medium |
| Framework | LightLLM | Vendor-stated | No | Medium |
| Framework | LightX2V | Vendor-stated (video/diffusion serving) | No | Medium |
| Framework | Hugging Face transformers | Compatible (claimed >90% ModelScope models) | — | Low |
| Framework | DeepSpeed | Confirmed supported | — | Medium |
| Framework | TensorFlow | Not confirmed | — | Low |
| Framework | PaddlePaddle | Not confirmed | — | Low |
| Compiler | Triton backend | **FlagTree `third_party/sunrise`** — `backend_name='tang'`, target `tang:S2`; Triton 3.4 from 2026-01-23, Triton 3.6 from 2026-06-30 | **Yes** (shim only — links closed-source `sunriseTritonPlugin.so`) | High |
| Compiler | Triton plugin (proprietary core) | `sunriseTritonPlugin.so`; wheels `flagtree==0.4.0+sunrise3.4` / `0.6.0+sunrise3.6` from resource.flagos.net | No | High |
| Compiler | Kernel compiler | Self-developed (nvcc analog); brand-level name still not disclosed | No | Medium |
| Compiler | LLVM backend / triples | `stcu-unknown-tang`; S3 via `stcuv2` triple + `*_S3.bc` libdevice | Visible via FlagTree | High |
| Compiler | Object bundling | `clang-offload-bundler` under `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-<arch>/` | Visible via FlagTree | High |
| Compiler | CUDA migration tool | Present (claimed); vendor claims 零代码无缝迁移 from third-party GPGPU platforms; depth unknown | No | Low |
| Compiler | Inference graph compiler | Assumed present (TensorRT analog); **not confirmed** | No | Low |
| Op Library | DL library | cuDNN/cuBLAS analog — **name still not disclosed** | No | Low |
| Op Library | Device math library | **OCML-based** (`__ocml_*`, AMD-style) in the Triton path | Visible via FlagTree | High |
| Op Library | GEMM / matmul | Present; `min_dot_size` (8,8,16) INT8 / (8,8,4) otherwise; dot input precision `ieee` only | No | Medium-High |
| Op Library | Attention / FlashAttention | Present; vendor claims 98% FlashAttention efficiency for S3 — **仿真实测 (simulation-measured)** | No | Medium |
| Kernel Library | Custom kernel templates | Assumed present (CUTLASS analog); Triton is the public authoring path | Partial (Triton) | Medium |
| Runtime | Runtime API | **TangRT** — `libtangrt_shared`, root `/usr/local/tangrt` (cudart analog) | No (headers/paths visible via FlagTree) | High |
| Runtime | Driver-level library | **`libtang.so`** | No | High |
| Runtime | Env configuration | `TRITON_LIBTANG_PATH`, `TRITON_SUNRISE_LLD_PATH`, `TRITON_SUNRISE_TRANSLATE_TRIPLE` | Visible via FlagTree | High |
| Communication | Collective comms | **PCCL** — "Sunrise Collective Communications Library"; FlagCX adaptor (`USE_SUNRISE=1`, `CCL_HOME=/usr/local/pccl`, `-lpccl`) | Adaptor yes (FlagCX); PCCL itself no | High |
| Communication | Supported ops | send/recv, broadcast, reduce, allreduce, allgather, reducescatter, group ops — homogeneous **and** heterogeneous modes | — | High |
| Communication | Unsupported ops | **gather, scatter, alltoall, alltoallv** (per FlagCX support matrix) | — | High |
| Communication | Scale-up fabric | **SRLink** (proprietary); bandwidth and topology not disclosed | No | Medium |
| Driver | Kernel driver | Linux PCIe driver (present; details not disclosed) | Unknown | Low |
| Driver | On-chip firmware | SIRE's bottom layer is **SoC 固件/系统软件层**; details not disclosed | No | Medium |
| Driver | Kubernetes plugin | Not confirmed | — | Low |
| ISA | Native ISA | Proprietary SIMT; **warp size 32**; no reference manual published | No | Medium-High |
| ISA | FP8 encoding exposed | `fp8e5` (E5M2) only on the Triton path | Visible via FlagTree | High |
| ISA | Virtual ISA (PTX analog) | Not confirmed | — | Low |
| ISA | Assembler / linker | LLD-based (`TRITON_SUNRISE_LLD_PATH`) | Partial | Medium |
| Tooling | Profiler | Assumed present; not disclosed | No | Low |
| Tooling | Debugger | Assumed present; not disclosed | No | Low |
| Tooling | Developer portal | sunrise-ai.com (SIRE is now a first-class product page) | — | Medium |
| Tooling | Wheel index | `https://resource.flagos.net/repository/flagos-pypi-hosted/simple` | Yes | High |
| Platform | MaaS / model engine | SIRE scope tags: 基础工具链 / 编程接口 / 基础库 / 基础平台 / 模型引擎 / MaaS 服务能力 / IP 模块化 | No | Low |

**Overall open-source posture (revised 2026-08-08):** **Partially open.** The baseline's "None identified — fully proprietary stack" is **refuted**. Xiwang participates in the BAAI **FlagOS** ecosystem through three upstream repos — FlagTree (Triton backend), FlagCX (PCCL adaptor) and vllm-plugin-FL — but every one of them is a thin adaptor over closed binaries (`sunriseTritonPlugin.so`, `libtang.so`, `libpccl`). No compiler internals, no ISA documentation and no operator library source are public.

**Vendor-claimed model coverage:** DeepSeek, Qwen, Llama, SD / Flux / Wan / Seko / Hunyuan3D, CNNs and speech models.

**Coverage gaps:** no public ISA documentation, no open compiler internals, no operator-library source or performance data, no capacity/bandwidth figures for either part, no independent benchmark, no MLPerf submission. The FlagTree user manual states the backend is **"Available for S2"** — S3 exists in the open toolchain only as compiler plumbing (`stcuv2` triple, `*_S3.bc` libdevice).

> The vendor news index carries an item 曦望全栈适配 FlagOS 2.1. The FlagOS organisation page advertises "Latest Release v1.5", so the **"FlagOS 2.1" version string is not confirmed** and is not recorded as fact here.
