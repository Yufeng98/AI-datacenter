# Tecorigin (太初元碁) SDAA Layer Mapping Table

*as_of: 2026-08-08*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*

Source keys used in the tables below:

| Key | Source |
|---|---|
| `docs-sdaac` | http://docs.tecorigin.com/release/sdaac — SDAA C 编程指南 v3.2.0 (latest v3.3.0) |
| `docs-pcx` | http://docs.tecorigin.com/release/pcx — PCX 编程指南 v1.2.0 / PCX ISA v1.0.0 |
| `docs-tecosmi` | http://docs.tecorigin.com/release/tecosmi — TecoSMI 用户手册 v1.15.0 |
| `docs-sdaart` | http://docs.tecorigin.com/release/sdaart — SDAARuntime 用户手册 v3.2.0 |
| `docs-install` | http://docs.tecorigin.com/release/software_installation — 环境安装手册 v3.2.0 |
| `docs-torch27` | http://docs.tecorigin.com/release/torch2.7 — TecoPyTorch v3.2.0 (PyTorch 2.7.1) |
| `docs-torch24` | http://docs.tecorigin.com/release/torch_2.4 — TecoPyTorch (PyTorch 2.4) |
| `docs-tecopaddle` | http://docs.tecorigin.com/release/tecopaddle — TecoPaddle v3.2.0 |
| `docs-tecovllm` | http://docs.tecorigin.com/release/teco_vllm — Teco-vLLM v3.2.0 |
| `docs-megatron` | http://docs.tecorigin.com/release/teco_megatron_lm — Teco-Megatron-LM v3.2.0 |
| `docs-infer` | http://docs.tecorigin.com/release/tecoinferenceengine — TecoInferenceEngine（小模型）v3.1.0 |
| `docs-opperf` | http://docs.tecorigin.com/release/op_perf_opt — 性能优化手册-算子篇 v1.1.0 |
| `docs-sdaacperf` | http://docs.tecorigin.com/release/sddac_perf_opt — 性能优化手册-SDAA C篇 v2.0.2 |
| `docs-tecogdb` | http://docs.tecorigin.com/release/tecogdb — TecoGDB v3.1.0 |
| `docs-sdpti` | http://docs.tecorigin.com/release/sdpti — SDPTI v1.7.0 |
| `docs-tsight` | http://docs.tecorigin.com/release/tsight — TSight GUI v1.9.0 |
| `gh-teco-ops` | https://github.com/Tecorigin/teco-ops (+ doc/teco-ops-hardware.md, README.md, doc/README_OP.md, doc/README_PLUGIN.md) |
| `gh-paddle-sdaa` | https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa — **INDEPENDENT** |
| `gh-runlog` | https://github.com/Tecorigin/modelzoo/blob/main/PyTorch/contrib/Classification/ACNet-master/scripts/acnet.txt |
| `gitee-tecorigin` | https://gitee.com/tecorigin — teco-al, teco-torch, teco-paddle, tcap_dllogger, … |
| `vendor-tech` | https://www.tecorigin.com/cn/technology.html |
| `vendor-products` | https://www.tecorigin.com/cn/products.html (marketing) |

---

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Packaging | Entire stack ships as exactly **two** version-locked packages: **TecoDriver** (driver + firmware + TCML + TecoSMI + TecoExporter) and **TecoToolKit** (SDK). TecoToolKit vN forward-compatible with TecoDriver ≤ vN. Current release **v3.2.0** | confirmed | docs-install |
| Packaging | Distribution is `.deb` / `.rpm` / `.runfile` + Docker tarballs from mirrors.tecorigin.com and jfrog.tecorigin.net. **Nothing on PyPI; no Hugging Face org** | confirmed | docs-install, negative checks |
| Framework Integration | **TecoPyTorch** (`torch_sdaa`) v3.2.0 — out-of-tree device extension on **PyTorch 2.7.1** (2.4.0 line also ships); `device='sdaa'`, `torch.sdaa.*` mirroring `torch.cuda`; AMP FP16 training; **inference FP32-only** | confirmed | docs-torch27, docs-torch24, gitee-tecorigin |
| Framework Integration | **TecoPaddle** (`paddle_sdaa`) v3.2.0 — PaddlePaddle 3.0.0 via CustomDevice | confirmed | docs-tecopaddle, gitee-tecorigin |
| Framework Integration | **PaddleCustomDevice `backends/sdaa`** — upstream, **Tecorigin-independent**, Apache-2.0; ~115 kernel `.cc` files, `sdaac_ops/*.scpp`, `pr_ci_sdaa.sh` hardware CI | confirmed | gh-paddle-sdaa |
| Framework Integration | **Teco-vLLM** v3.2.0 — PagedAttention, continuous batching, chunked prefill, prefix caching, **disaggregated prefill (PD分离)**, LoRA, tool calling, reasoning outputs | confirmed | docs-tecovllm |
| Framework Integration | Teco-vLLM parallelism: **TP / PP / EP / DP + EPLB** (expert-parallel load balancer) **+ SBO** (single-batch overlap); multi-node via **Ray** | confirmed | docs-tecovllm |
| Framework Integration | Teco-vLLM quantization: **INT8 W8A16** (`quantization="sdaa_wint8"`, online dynamic only), **INT4 W4A16**, **KV-cache-INT8**, GPTQ, AWQ, GGUF. **All weight-only — no W8A8** | confirmed | docs-tecovllm |
| Framework Integration | **Teco-Megatron-LM** v3.2.0 — tensor / sequence / pipeline / expert parallelism + distributed optimizer; DeepSeek-R1-Distill-Llama-70B SFT at TP=4 × PP=8 on 1 node × 8 cards | confirmed | docs-megatron |
| Framework Integration | **TecoInferenceEngine（小模型）** v3.1.0 — ONNX-only front end, Python + C++ APIs, TensorRT-migration path, dynamic shapes, async inference, CPU fallback, 70+ models | confirmed | docs-infer |
| Framework Integration | **TecoInferenceEngine（大模型）** — separate large-model inference engine | likely | docs portal homepage API (doc id 83566) |
| Framework Integration | Vendor-claimed additional support: SGLang, xDiT, DeepSpeed, LLaMA-Factory, Transformers, TensorFlow; "40+ mainstream LLMs" | unconfirmed | vendor-tech (VENDOR CLAIM, not corroborated in manuals) |
| Graph Capture | **TorchDynamo / `torch.compile`** with `fullgraph`. **`dynamic`, `mode`, `options` are reserved and NOT supported** — no dynamic-shape compiled path today | confirmed | docs-torch27 |
| Graph Capture | **ONNX** — the only front end for TecoInferenceEngine (小模型); PyTorch/Paddle must export to ONNX first | confirmed | docs-infer |
| Graph Capture | Stream capture **stubbed only** — `sdaaErrorStreamCaptureUnsupported` / `...Invalidated` in the runtime enum; no user-facing API documented | likely | docs-sdaart |
| Graph Compiler | **`teco_inductor`** — TorchInductor backend (`torch.compile(model, backend='teco_inductor')`); fuses Pointwise / Reduction / Foreach with broadcasting; **Conv and GEMM fall back to the libraries** | confirmed | docs-torch27 |
| Graph Compiler | `teco_inductor` codegen options at `torch_sdaa._inductor.config.teco.*`: `use_simd128`, `higher_performance`, `use_table` (LUT transcendentals; both may reduce accuracy), `enable_kernel_profile`, `debug_sync_graph` | confirmed | docs-torch27 |
| Graph Compiler | **TVM / Relay IR** — TecoInferenceEngine's compiler; custom ops "通过 TVM Relay IR 注册"; plugins built with `g++` for TVM header compatibility. Four layers: front end (ONNX + FP32→FP16) → graph opt (fusion, const-fold, CSE, layout insertion, fused-kernel codegen) → runtime (Engine/Context, async, dynamic shape) → device | confirmed | gh-teco-ops, docs-infer |
| Kernel Compiler | **TecoCC** (`tecocc`) v3.2.0 — **Clang/LLVM-derived**: `.bc` bitcode host+device, `clang-offload-bundler` (`--sdaa-link`), `-flto`, `-O0..-O3`, `-g`. Device sources `.scpp`. Flags `--sdaa-arch=pcx_100`, `--stack-on-global`, `--sdaa-device-only/--sdaa-host-only`, `--sdaa-device-lib=` | confirmed | docs-sdaac |
| Kernel Compiler | **PCXAC** (PCX Advanced Compiler) v1.2.0 — compiles/links PCX to machine code; ships static + dynamic **MemChecker** | confirmed | docs-pcx, docs-install |
| Kernel Language | **SDAA C** v3.2.0 (docs at v3.3.0, 2026-07-23) — `__global__`/`__device__`/`__host__`/`__local__`/`__scoped_local__`, `<<<...>>>`, `threadIdx`/`threadDim`; intrinsics for thread groups, `sync_threads`, SPM `malloc`/`free`, **DMA**, **RMA**, **broadcast**, atomics, **matmul**, transpose, SIMD (incl. `*128`), batched math | confirmed | docs-sdaac, gh-teco-ops |
| Kernel Language | Device-side restrictions: no exceptions, RTTI, STL, `new`, global ctors/dtors, local statics, file I/O, or native C/C++ atomics — a bare-metal scratchpad target | confirmed | docs-sdaac |
| **ISA** | **PCX (Parallel Computing eXecution) v1.0.0** — **hardware-independent VIRTUAL ISA**, structurally the PTX of this ecosystem. Infinite virtual general registers; scalar/vector/matrix instruction classes; thread + thread-group hierarchy (`%tid`, `%gid`, `%ngroup`, `%nthread`); multi-level storage abstraction; debug info; perf-sampling instructions | confirmed | docs-pcx |
| ISA | PCX directives `.version`, `.arch`, `.entry`, `.func`, `.visible/.extern/.weak/.alias`; storage qualifiers `.spm`/`.global`/`.const`; matrix ops `matmul_init/load_weight/compute/store`; movement `mov`, `lda`, `load`, `store`, `memcpy`, `memcpy_broadcast`, `rma`, `broadcast`. Compatibility table lists **only T100系列** | confirmed | docs-pcx |
| ISA | **T1 machine instruction set — NOT DISCLOSED, by design.** PCX exists to hide it; no opcode listing or disassembly format published | confirmed | docs-pcx |
| Tensor API / Libraries | **TecoDNN** v3.2.0 (`libtecodnn.so`) — cuDNN analogue | confirmed | docs-install, gh-runlog |
| Tensor API / Libraries | **TecoBLAS** v3.2.0 (`libtecoblas.so`) — cuBLAS analogue | confirmed | docs-install, gh-runlog |
| Tensor API / Libraries | **TecoRAND** v3.2.0 (`libtecorand.so`) — cuRAND analogue | confirmed | docs-install, gh-runlog |
| Tensor API / Libraries | **TecoLMK** v3.2.0 — large-model inference kernel library | confirmed | docs-install |
| Tensor API / Libraries | **TecoCUSTOM** v3.2.0 (`libtecodnn_ext.so`, formerly "CustomDNN") — custom-operator library | confirmed | docs-install, gh-runlog |
| Tensor API / Libraries | **TecoAL** (`libtecoal.so`) — the **open-source face** of the operator library, NOT a shipped rename: the v3.2.0 install manual lists only TecoDNN/TecoBLAS/TecoCUSTOM. Document both names | likely | docs-install, gh-teco-ops, gitee-tecorigin |
| Tensor API / Libraries | **No public manual exists for any acceleration library** — `getReleaseTree` 404s for tecoal/tecodnn/tecoblas/tecolmk/tecocustom/tecorand | confirmed | docs portal API probes |
| Communication | **TCCL** v3.2.0 (`libtccl.so`) — NCCL analogue; PyTorch integrates as `ProcessGroupTCCL` (`backend='tccl'`) | confirmed | docs-install, docs-torch27, gh-runlog |
| Communication | **TCCLTests** v1.1.0 — nccl-tests analogue | confirmed | docs-install |
| Communication | **MLNX_OFED 5.9-0.5.6.0** (incl. NVIDIA SHARP 3.2.0) + **Open MPI 4.1.5rc2** — MANDATORY multi-node dependency, vendor-redistributed | confirmed | docs-install |
| Runtime | **SDAARuntime** v3.2.0 (`libsdaart.so`) — CUDA-Runtime analogue: `sdaaSetDevice/GetDeviceCount/GetDeviceProperties`, `sdaaMalloc/Free/Memcpy`, streams (`sdaaStreamCreate/Synchronize/WaitEvent`), events, P2P, `sdaaDeviceProp_t`, `sdaaDeviceAttribute_t` (incl. `sdaaDevAttrArch`), full CUDA-shaped `sdaaError_t` enum | confirmed | docs-sdaart, gh-runlog |
| Driver / Firmware | **SDAADriver** v3.2.0 (package `tecodriver`) — devices appear as **`/dev/tcaicardN`**, one node per card; containers pass `--device=/dev/tcaicard0..3`; firmware images `aiflash_v<ver>_T1`; VBIOS/MCU/PCB versions reported | confirmed | docs-install, docs-tecosmi, gh-paddle-sdaa |
| Driver / Firmware | **TCML** v1.15.0 — NVML analogue (management/monitoring library) | confirmed | docs-install |
| Driver / Firmware | **TecoSMI** v1.15.0 (`teco-smi`) — nvidia-smi analogue: `-L/-q/-d/-l/-lms`, `--query-device`, `--format=csv`, `stats`, `dmon`, `daemon`, `replay`, `pmon`, `topo` (incl. `-p2p r\|w`), `-r` reset, `-lsc/-rsc` SPE clock lock/reset, `-lpm` low-power | confirmed | docs-tecosmi |
| Driver / Firmware | **TecoExporter** v1.5.1 — Prometheus-style telemetry collection service | confirmed | docs-install |
| Debug / Profile | **TecoGDB** v3.1.0 — cuda-gdb analogue for `.scpp`; source-level breakpoints, **SPE focus switching** (`switchSPE`), `target tecocore` core dumps, `printException`, Kernel-PC/`setPc`. Limitation: only programs where all SPEs run identical code | confirmed | docs-tecogdb |
| Debug / Profile | **TecoGDB Visual** v1.2.0 — GUI debugger | confirmed | docs portal |
| Debug / Profile | **SDPTI** v1.7.0 (`libsdpti.so`) — CUPTI analogue: Activity + Callback APIs, activity buffers, external correlation IDs | confirmed | docs-sdpti, gh-runlog |
| Debug / Profile | **TCPX** v0.3.0 — **NVTX analogue**, event/range annotation library | confirmed | docs-install |
| Debug / Profile | **TSight CLI** v1.12.0 + **TSight GUI** v1.9.0 — **Nsight analogue**, host + device parallelism profiling | confirmed | docs-tsight, docs-install |
| Debug / Profile | **TCVS** v1.6.0 — hardware/software validation and stability suite (DCGM-diag analogue) | confirmed | docs-install |
| Debug / Profile | `sdaacfilt` — symbol demangler (c++filt analogue); `tcap_dllogger` — structured training/inference logger (Apache-2.0) | confirmed | docs-install, gitee-tecorigin |
| Custom Ops | **SDAA Extension** — PyTorch custom-op path mirroring `torch.utils.cpp_extension`: `.scpp` kernel → C++ wrapper → `TORCH_LIBRARY` → setuptools; optional `torch.compile` + autograd; **`load_inline` added in v3.2.0** | confirmed | docs-torch27 |
| Custom Ops | **AbstractPluginOp** — TecoInferenceEngine plugin base class (`InferOutputShape` + `Enqueue`) → `libteco_ops_plugin.so` + `libTecoInferPlugin.so` | confirmed | gh-teco-ops |
| Custom Ops | `torch.sdaa.memory.SDAAPluggableAllocator` — pluggable device allocator (CUDA parity) | confirmed | docs-torch27 |
| Precision Controls | `TORCH_SDAA_CONV_USE_FP32` / `..._CONV2D_BACKWARD_USE_FP32` (Conv internally converts FP32→FP16 by default); **`TORCH_SDAA_BF16_CLIP` clips bf16 GEMM inputs to ±65407**; `TORCH_SDAA_CONV_WEIGHT_CHWN`; `TORCH_SDAA_FALLBACK_OPS` / `..._RUNTIME_AUTOFALLBACK`; `TORCH_SDAA_LOG_LEVEL`; `TORCH_DEVICE_BACKEND_AUTOLOAD` | confirmed | docs-torch27 |
| Precision Controls | Training: FP32 + FP16 (AMP with weight backup + loss scaling). **TecoPyTorch inference is FP32-only**; TecoInferenceEngine converts everything to FP16 | confirmed | docs-torch27, docs-infer |
| Platform | 太初算力管理服务平台 — multi-tenant cluster scheduler; mentions Jupyter/VS Code, Megatron-DeepSpeed, TVM + Triton small-model serving, FasterTransformer large-model serving, GPFS/GlusterFS | unconfirmed | vendor-tech (VENDOR CLAIM) |
| Openness | **Open**: teco-ops (BSD-3), Teco-AL (BSD-3), teco-torch (BSD-3), teco-paddle (Apache-2.0), modelzoo family (BSD-3), tcap_dllogger (Apache-2.0), and the independent PaddleCustomDevice SDAA backend (Apache-2.0). **Closed**: every compiler, library, runtime, driver and tool binary | confirmed | gh-teco-ops, gh-paddle-sdaa, gitee-tecorigin, docs-install |
| Openness | Traction is small and partly artificial — 0–4★ GitHub, 0–31★ Gitee; Teco-AL's 157 forks are WAIC-competition artifacts; the most active GitHub repos were created 2026-04-13 for that competition | confirmed | GitHub/Gitee API |

---

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Chip Identity | Chip family name is **T1** (firmware `aiflash_v<ver>_T1`; `sdaaDeviceProp_t.clockRate` = "T1计算核心SPE的频率"). **T100/T110/T111 are card SKUs, not chip names** | confirmed | docs-tecosmi, docs-sdaart |
| Chip Identity | Architecture target **T100 系列**; TecoCC arch flag `--sdaa-arch=pcx_100`; driver reports `TECO_AICARD_01` / `TECO_AI_01` / PCI ID `0x071900A1` | confirmed | docs-pcx, docs-tecosmi |
| Chip Identity | Cards: 元碁 T100 (PCIe), T110 (air-cooled OAM), T111 (liquid-cooled OAM). Systems: workstation 1–4 cards, T1008 4U/8, I1004 2U/4, T1108 6U/8 OAM, T1118 2U/8 liquid, SuperPOD-128 (128 cards/rack) | confirmed | vendor-tech |
| Chip Identity | **Process node, foundry, die size, transistor count, packaging — NOT DISCLOSED** anywhere in ~3.2 MB of vendor docs or on the vendor site | unconfirmed | (absence) |
| Compute Engine | Card = **4 SPAs**; each SPA is an independent SDAA device with its own Global memory; **no shared address space across SPAs** | confirmed | docs-tecosmi, docs-torch27, docs-tecovllm, docs-tecogdb |
| Compute Engine | SPA = **32 SPEs** (`Has 32 GPC`; SPE rows 0…31; "大于31时表示特殊模块"; `CORE_NUM = 32` / "32个从核") | confirmed | docs-tecogdb, gh-teco-ops, docs-sdaacperf |
| Compute Engine | **128 SPEs per card** (4 × 32) | confirmed (derived from two confirmed numbers) | docs-tecosmi, docs-tecogdb |
| Compute Engine | Per SPE: **SU + SREG** (scalar), **VPU + VREG** (vector), **FU** (matrix/MMA, reads and writes SPM only), **private SPM** | confirmed | gh-teco-ops, docs-sdaac, docs-pcx |
| Compute Engine | Matrix unit: **128 × 32 × 32 MMA** per operation; **weight-stationary** with double-buffered weight register; accumulator buffer with `.flush_output`; 8-bit-plus weight-row mask; **64-byte** input stride granularity | confirmed | docs-opperf, docs-pcx, docs-sdaac |
| Compute Engine | Matrix dtypes: **exactly four** — `0x606` FP16→FP16, `0x806` FP16→FP32, `0x404` S16→S16, `0x704` S16→S32. **No BF16, TF32, FP8 or INT8 in the matrix unit** | confirmed | docs-pcx, docs-sdaac |
| Compute Engine | bf16 GEMM is **emulated on the FP16 unit** — `TORCH_SDAA_BF16_CLIP` clips inputs to ±65407; SDAA C bf16 is "仅支持指针操作"; Teco-vLLM quantization is entirely weight-only (no W8A8) | confirmed | docs-torch27, docs-sdaac, docs-tecovllm |
| Compute Engine | **FP64 present in the PCX ISA** (`.f64`) — consistent with HPC + AI positioning | confirmed | docs-pcx |
| Compute Engine | Clocks: `Mpe` 2000/2300 MHz, **`Spe` 2000/3000 MHz** (eFUSE-set initial), `Hbm` 1600/1650 MHz, `Glb` 2200/2200 MHz. Perf manuals benchmark at **2.36 GHz** SPE | confirmed | docs-tecosmi, docs-opperf |
| Compute Engine | An **`Mpe` clock domain exists**, but its role, count, ISA and programmability are **not disclosed**; it is invisible to SDAA C and PCX. Nothing states it is an SW26010-style MPE | unconfirmed | docs-tecosmi, docs-tecogdb |
| Compute Engine | **Peak throughput at any precision — NOT DISCLOSED.** Only a tutorial micro-benchmark exists: **≈29 TFLOPS** FP16 through the matrix unit vs **≈2.5 TFLOPS** through vector instructions at 2.36 GHz. Vendor states it is "远没有达到" peak; scope (per-SPE/SPA/card) not stated | likely (as a measurement); unconfirmed (as any peak) | docs-opperf |
| Compute Engine | SPE pipeline depth, issue width, pipeline count — **not disclosed** ("multi-stage, multi-issue" only) | unconfirmed | docs-opperf |
| Data Path | **SPMD** execution — one program image per SPA; `threadIdx` = SPE ID, `threadDim` = SPE count. One SPA = one device (`sdaaSetDevice`) | confirmed | docs-sdaac, docs-torch27 |
| Data Path | PCX describes intra-thread-group execution as **SIMD**, a stronger claim than SDAA C's SPMD. **The two vendor docs are inconsistent**; TecoGDB's per-SPE focus and independent PCs favor SPMD with independent control flow | likely | docs-pcx, docs-sdaac, docs-tecogdb |
| Data Path | **DMA** Global↔SPM: `memcpy`, `memcpy_stride`, `memcpy_async`/`memcpy_wait` (`MemcpyHandle`) | confirmed | docs-sdaac |
| Data Path | **RMA** SPE↔SPE SPM directly: `rma_get`/`rma_put`, `rma_async_get`/`rma_async_put`/`rma_complete`/`rma_wait`; `RmaNormalMode`/`RmaCustomizeMode` — the Sunway athread fingerprint | confirmed | docs-sdaac, gh-teco-ops |
| Data Path | **Hardware broadcast**: `broadcast`, `broadcast_async`, `memcpy_broadcast`; thread-group scoped | confirmed | docs-sdaac |
| Data Path | Thread groups (`ThreadGroup`, `thread_group_set_mask/include/exclude`) scope sync, broadcast and RMA to SPE subsets | confirmed | docs-sdaac |
| On-chip Memory | **Three-level model, no hardware data cache in the compute path**: registers / SPM / Global, all movement explicit | confirmed | docs-opperf |
| On-chip Memory | **SPM: private per SPE, ≥235 KB usable, hard ceiling 240512 B** (240640 B incl. 128 B wrapper = exactly 235 KiB). Partitioned into heap / stack / local; partition sizes runtime-queried, never published; **physical array size not disclosed** | confirmed | gh-teco-ops, docs-sdaac |
| On-chip Memory | **Instruction cache present, per SPE** — the only cache documented anywhere. Confirmed by the "指令缓存脱靶次数 (Instruction Cache Miss)" perf counter and TecoGDB's per-SPE `GPC` = "指令cache首地址". **Size not disclosed** | confirmed | docs-sdaacperf, docs-tecogdb |
| On-chip Memory | Register file sizes and **physical VREG width — not disclosed**; SDAA C vector types are inconsistent in width (`halfv16` 256 b vs `floatv16` 512 b) and cannot be used to infer it | unconfirmed | docs-sdaac |
| On-chip Memory | Measured (single SPE, 2.36 GHz, 128 KB): Global→SPM DMA **45.45 GB/s** 4 B-aligned, degrading **49×** to **0.92 GB/s** on mismatched mod-4 residues. Ranking SPM→SPM > Global→SPM > SPM→Global > Global→Global | confirmed | docs-sdaacperf |
| Off-chip Memory | **HBM** — confirmed by dedicated `Hbm` power/voltage/current/clock telemetry domains and per-card "设备内存芯片" reporting | confirmed | docs-tecosmi |
| Off-chip Memory | Driver-reported capacity: **15296 MB per SPA**, **65536 MB (64 GB) per card** (61172 MB free). No datasheet capacity is published — treat as "documented driver-reported capacity of the T100-generation card" | confirmed | docs-tecosmi, docs-torch27, docs-tecopaddle |
| Off-chip Memory | **HBM bandwidth — NOT DISCLOSED and NOT DERIVABLE.** Only the HBM clock (1600/1650 MHz) is exposed; generation, stack count and bus width are all secret | unconfirmed | docs-tecosmi |
| Off-chip Memory | ECC present (`sdaaErrorECCNotCorrectable`) | confirmed | docs-sdaart |
| On-chip Interconnect | **Ring network (环网) @ 2200 MHz** — `teco-smi` documents `Glb` as "太初AI加速卡环网的实时频率". The only direct vendor statement of on-chip topology. Bandwidth/bisection **not disclosed** | confirmed | docs-tecosmi |
| On-chip Interconnect | SPE array behaves as an **8-column × 4-row grid with row/column broadcast buses** — derived from vendor cost models: RMA Manhattan distance `D(x,y)=abs(x/8−y/8)+abs(x%8−y%8)`; broadcast fast groups = rows {0–7}…{24–31} and columns {0,8,16,24}…{7,15,23,31}; `sync_threads` fast on shared quotient or remainder mod 8 (~2.2× otherwise); DMA best with distinct `id%8` | likely (derived) | docs-sdaacperf |
| On-chip Interconnect | **Do not** cite the SDAA C "横向、纵向广播" example as topology evidence — it is a software-simulated logical 4×4 grouping | confirmed | docs-sdaac |
| Host Interface / Package | **PCIe Gen4 ×16** (`PCIe Generation Max: 4`, `Link Width Max: 16x`); device nodes `/dev/tcaicardN`, one per card | confirmed | docs-tecosmi, gh-paddle-sdaa |
| Scale-up Interconnect | **No proprietary chip-to-chip link documented; evidence is NEGATIVE.** `teco-smi topo` legend is exactly `SYS/NODE/PHB/PXB/PIX` (the nvidia-smi PCIe set) with **no NVLink-equivalent entry**. **PCIe P2P is the documented card-to-card path**; Teco-vLLM runs TP=8 across 2 cards over it | confirmed | docs-tecosmi, docs-sdaart, docs-tecovllm |
| Scale-up Interconnect | SuperPOD-128 rack interconnect topology and bisection bandwidth — **not disclosed** | unconfirmed | (absence) |
| Scale-out Interconnect | Standard **InfiniBand / RoCE**. TecoToolKit multi-node install **requires MLNX_OFED 5.9-0.5.6.0** (incl. NVIDIA SHARP 3.2.0) and **Open MPI 4.1.5rc2**, vendor-redistributed. Multi-node vLLM uses Ray + `GLOO_SOCKET_IFNAME`. T1108 advertised with "8 IB/RoCE 高速通信" | confirmed | docs-install, docs-tecovllm, vendor-tech |
| Scale-out Interconnect | Deployment link claims: "四链路 400 Gbps" (Yan'an, Lihu), "四链路 200 Gbps 国产 AI 计算网络" (open-source platform), self-developed "200G 高速无损互联技术" | unconfirmed | vendor-products (VENDOR MARKETING) |
| Power / Thermal | **TDP — not disclosed.** Observed idle telemetry: **90 W at 0 % SPE utilization, 35–40 °C**. **Shutdown temperature threshold 70 °C**; a slowdown-threshold field exists | confirmed (as observation), unconfirmed (as TDP) | docs-tecosmi, docs-torch27 |
| Power / Thermal | Rack-level: 太湖之光A+ — 128 cards/rack, **32 PFLOPS**, **100 kW**, "highest compute density in China" | unconfirmed | vendor-products (VENDOR CLAIM, unspecified precision) |
| Host CPU Support | Five validated CPU/OS combinations with full framework stacks: x86_64 (Ubuntu 22.04), **Hygon 海光 7380/7375** (Kylin V10), **Phytium 飞腾 S5000C** ARMv8 (Kylin V10 国防版), **Sunway 申威 8A (SW-64)** (UOS Server 20), **Loongson 龙芯 3C6000 (LoongArch)** (Loongnix Server 23.1) | confirmed | docs-install |
| Lineage | **Sunway lineage = team / ecosystem / idiom, NOT architectural derivation.** Team from NSCC-Wuxi (3× Gordon Bell); SW-64 host support; 太湖之光A+ branding; athread idioms in depth. **No source states SW ISA sharing or SW26010 derivation**; no on-device MPE+CPE core-group structure; host CPU is the master; PCX exists to decouple software from the machine ISA | likely | vendor-about, docs-sdaacperf, docs-install, docs-pcx |
| Benchmarks | **No independent third-party benchmark exists** — no MLPerf, no SPEC, no external review retrievable. (Weak negative: WebSearch was unavailable this pass) | unconfirmed | (absence) |
