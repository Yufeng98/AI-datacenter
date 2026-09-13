# Stream Computing (希姆计算) NeuralScale Layer Mapping Table

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (primary): `torch.compile(model, backend=mltc_backend)` and `from_torch(nn.Module \| ExportedProgram)` after `torch.export`, with `Dim(...)` dynamic shapes on parallel/reduce axes | confirmed | mltc-guide, python-api |
| Framework Integration | ONNX: `OnnxToStc().run(...)`; 156 ONNX ops; custom ops via domain `mltc.custom` | confirmed | mltc-op-support |
| Framework Integration | TensorFlow: `TfToStc().run(...)`; 106 TF ops (TF1 GraphDef `.pb`) | confirmed | mltc-op-support |
| Framework Integration | PaddlePaddle: `Paddle-STCNNE` backend + open `STCPaddleModelZoo` (~20 models, CPU-vs-NPU accuracy deltas) — **TensorTurbo-era artifact, not the current stack** | confirmed | stcpaddlemodelzoo |
| Framework Integration | HuggingFace LLMs via STC_LLM (loads `bin` / `safetensors` directly) | confirmed | stc-llm-guide |
| Framework Integration | Worked example: Ultralytics YOLOv5 patched with a `vmfb` backend type in `DetectMultiBackend` | confirmed | mltc-guide |
| Framework Integration | **Absent**: no vLLM fork, no SGLang, no Triton kernel language, no TVM in the current stack, no ONNX Runtime EP | confirmed (absence) | all-sdk-docs |
| Graph Capture | TorchDynamo custom backend + `torch.export` → ExportedProgram → Torch-dialect MLIR; `from_torch` signature is byte-for-byte torch-mlir's `fx.export_and_import`; op list uses 141 torch-mlir `Aten*Op` C++ class names | confirmed (API surface) / likely (torch-mlir derivation) | python-api, mltc-op-support |
| Compiler / IR | **MLTC** (Multi-Level Tensor Compiler) 1.6.1 — proprietary MLIR compiler; stages Graph Optimizations → Group Partition → GOAT tiling → codegen | confirmed | mltc-guide |
| Compiler / IR | **GOAT** — operator-agnostic tiling framework over **LLB → MC → L1**; vendor rationale: "most of the code is managing the multi-level storage hierarchy and the data movement between levels" | confirmed | mltc-guide |
| Compiler / IR | `stc.*` graph dialect, 164 ops: `stc.attention`/`_lm`, `stc.kv_cache_load`/`_store`, `stc.matmul` vs `stc.matmul_vme`, `stc.quantize`/`dequantize`, `stc.im2col`/`col2im`, collectives `stc.device_*` | confirmed | mltc-op-support |
| Compiler / IR | Graph-level ops are **atomic — no fused operators in the graph IR** (fusion happens below); optimisations are independent switchable passes | confirmed | mltc-guide |
| Compiler / IR | Custom-operator plug-in: ONNX node in `mltc.custom` + shape-inference script (`MLTC_CUSTOM_OPS_PATH`) + compute impl `.h`/`.py` (`MLTC_PLUGINS_PATH`) | confirmed | mltc-guide |
| Compiler / IR | Debug/analysis: `Simulator()` CPU golden path; `run_precision_analysis.py` → `cpu_vs_npu.csv` (cosine similarity flagged <0.95); `run_perf_analysis.py` → per-core cycle/byte CSVs | confirmed | mltc-guide |
| Model Binary | **`.vmfb`** ("STC MLTC fatbin") compiled with `-arch=npu-v1`; the container is **IREE's VM FlatBuffer** and `--graph-partition-factor` is documented in IREE `dispatch.workgroup` terms → **IREE + torch-mlir derivative**; the vendor never names the upstream | confirmed (artifact) / likely (lineage) | mltc-guide, python-api |
| Kernel Language | **SHC** (Stream Computing Heterogeneous C++, `.hc`): full C++17 + `<<< >>>`, `__global__`, `__device__`, `__local__` (L1) / `__shared__` (LLB), `CoreID`/`CoreNum`, `sync()`; direct ISA intrinsics (`memul_mm`, `mov_m`) and shape-CSR macros (`DEFINE_SHAPE`, `CONFIG_VE_CSR`) | confirmed | hpe-guide, glossary |
| Kernel Language | HPE-Python — same heterogeneous model from Python (wheel `hpe_python-1.4.1-cp310`) | confirmed | stcrp-release-notes |
| Kernel Language | **Absent**: no Triton, no TVM TensorIR, no auto-scheduler, no tensor DSL. Only automation is GOAT tiling | confirmed (absence) | all-sdk-docs |
| Kernel Compiler | **`stcc`** — compiles host + device in one invocation (nvcc-shaped); device target RV32 + RVV v0.8 + OP-VE; `--rtlib=compiler-rt`, Clangd integration and `/usr/local/hpe/riscv32npu/` imply a Clang/LLVM downstream (never stated; LLVM version not disclosed) | confirmed (behaviour) / likely (LLVM lineage) | hpe-guide, riscv-stc-llvm |
| Tensor / Inference API | `mltc` Python package (`TfToStc`, `OnnxToStc`, `from_torch`, `optimize`, `Compiler`, `Executor`, `Simulator`); `STC_IE` 1.6.1 with `stc_ie.stc_runtime.executor.Executor(vmfb, device_ids, n)` | confirmed | python-api |
| Tensor / Inference API | Legacy: **STC_DDK** (AI Convertor + AI Executor), TensorTurbo era | confirmed | carrv-2021, bytemlperf-stc |
| Runtime | **HPE** (Heterogeneous Programming Engine): `hpert` host runtime (device/memory/execution/stream mgmt) + `npurt` device runtime (on-device `printf`, memcpy) | confirmed | hpe-guide |
| Runtime | **CUDA runtime clone by name**: `stcMalloc`, `stcMallocHigh`, `stcFree`, `stcMemcpy(HostToDevice/DeviceToHost)`, `stcConfigureCall`, `stcLaunchKernel`, `stcModuleLoadData`, `stcRegisterFatBinary`, `stcDeviceSynchronize` | confirmed | hpe-guide |
| Runtime | **STCML** (`libstcml.so`) — NVML analogue: init/shutdown, device enumeration, name/uuid/serial/power, get/set frequency (east/west), get/set voltage, temperature alerts, board status, firmware versions; 14 error codes | confirmed | cpp-api |
| Runtime | **`stcpti`** — CUPTI analogue; MME/MTE/VME-granularity cycle counts | confirmed | hpe-guide |
| Runtime | Resource model: card = device → 4 NPC Clusters → NPCs; kernels addressed `[device x, cluster y, core z]`; **the cluster is the allocation/isolation unit** (`/dev/stc0c0…c3`) | confirmed | hpe-guide, stcrp-install |
| Tooling | `stc-smi` (nvidia-smi), `stc-topo` (nvidia-smi topo, prints `GPU0…GPU7` labels), `stc-gdb` (cuda-gdb, `stc focus device/cluster/core`), `stc-prof` (nsys; Chrome-tracing JSON; VME-CU vs VME-VEC; L1/IM bank collisions), `stc-vprof` (Nsight, JDK-11 GUI with Clangd `.hc` editor) | confirmed | hpe-guide |
| Tooling | `NPU-Viewer` (PCIe/DDR/TOPS/stress qualification), `p2p_perf` (p2pBandwidthLatencyTest analogue), `stcqual` (factory qualification), `stc-hpaa` (half-precision accuracy, TensorTurbo era) | confirmed | npu-viewer-guide, p2p-perf-guide |
| Driver / Firmware | `stc-dkms` builds `stc.ko` via DKMS; `stc-kernel-common` udev rules; nodes `/dev/stc0`, `/dev/stc0c0…c3`, `/dev/stc0ctrl`; `lspci` driver `stc` | confirmed | stcrp-install |
| Driver / Firmware | Two **signed** firmware images — NPU-ctrl (SoC bring-up) + MCU (card power); in-band upgrade via `stc-smi -u` / `--mcu-upgrade` with `.sdux` | confirmed | firmware-release-notes |
| Driver / Firmware | Virtualisation: **no SR-IOV** (1 PF, no VFs); PCIe passthrough to KVM supported (FW V1.3.3+), host + KVM simultaneously (V1.3.4+) | confirmed | stcp920-manual, firmware-release-notes |
| Driver / Firmware | Packaging: repo `.deb`/`.rpm` with GPG keys; `hpe-host` (driver) vs `hpe-simple` (userspace) split; install layout `/usr/local/hpe/{bin,lib,include,riscv32npu,…}`; **no public source repos, no published license terms** | confirmed | stcrp-install |
| Communication | **No collective library** — no NCCL/RCCL/HCCL/CNCL analogue anywhere in the SDK (absence verified across all 22 SDK doc pages) | confirmed (absence) | all-sdk-docs |
| Communication | Collectives live **in the compiler**: `stc.device_broadcast / split / concat / allconcat / reduce_{sum,mean,max,min} / sync`; pipeline parallelism via `--pipeline-partition-factor` | confirmed | mltc-op-support, python-api |
| Communication | Tensor parallelism via weight pre-splitting at conversion time: `convert_weight -n <NPU count>`; weights for 4 NPUs are unusable on 2 | confirmed | stc-llm-guide |
| Communication | Transport: PCIe peer-to-peer — vendor-measured ~9.0 GB/s (LLB destination), ~0.89 GB/s (DDR destination) | confirmed | p2p-perf-guide |
| LLM Serving | **STC_LLM** 1.3.2 — proprietary, **not a vLLM fork**; OpenAI-compatible `/v1/chat/completions` + `/v1/completions`, SSE streaming, `usage.cached_tokens` | confirmed | stc-llm-guide |
| LLM Serving | Backends via `--compiler`: `stc_llm_dnn` (default; hand-written kernels rendered as `.cpp` into `engine_dir`, vendor says faster), `mltc` (`.vmfb`), `IE` (STC_IE) | confirmed | stc-llm-guide |
| LLM Serving | Features: dynamic batching (`max_tasks` 256), KV-cache segmentation (`slot_number_per_segment` 256 tokens), prefix caching, DMA weight preload, `embed_on_host`/`output_on_host`, reasoning parsers (Qwen3, DeepSeek-R1), tool-call parser (Qwen3) | confirmed | stc-llm-guide |
| LLM Serving | Serve-time quantisation `fp16 / w8a8 / w8a16`; `compress_factor {128,64,32,16,8,1,-1}` (−1 = per-channel) | confirmed | stc-llm-guide |
| LLM Serving | Native Prometheus metrics in the `stc_llm:` namespace + documented Grafana dashboard flow | confirmed | stc-llm-guide |
| Quantisation | **SNQ** (`stcnq`) — ONNX PTQ; **SNC** (`stcnc`) — built on **Intel Neural Compressor**, SmoothQuant with α sweep 0.1–0.9 calibrated on C-Eval, emits per-α activation-scale `.npy` sets; CPU-only quantisation supported | confirmed | stc-llm-guide |
| Deployment / Orchestration | `stc-k8s-device-plugin` (Go DaemonSet; k8s 1.18/1.22/1.23/1.24; auto-registration + health) | confirmed | k8s-plugin-guide |
| Deployment / Orchestration | `npu-exporter` — Python Prometheus exporter on port **9836** (npu_temp, npu_util, power_draw, memory_used/total, cluster_* metrics) | confirmed | npu-exporter-guide |
| Deployment / Orchestration | Docker: `hpe-host` driver on host + `hpe-simple` userspace in container; `--device /dev/stc0 /dev/stc0c0…c3 /dev/stc0ctrl`; reference image ~10.7 GB; **no official public registry images** | confirmed | stcrp-install |
| Deployment / Orchestration | 希姆云平台 (multi-tenant scheduling, quotas, registry, audit) and 大模型平台 — docs are image-only pages, specifications effectively **not disclosed** | unconfirmed | cloud-platform-docs |
| Historical Stack | **TensorTurbo** (2021–2023): TVM-based graph compiler + LLVM backend + stcDNN operator library + STC_DDK + HPE 1.5.x; C++/Python inference APIs for TF/PyTorch/MXNet/Keras. **Migrated TVM → MLIR/IREE (TensorTurbo → MLTC)** — do not mix the two stacks | confirmed | carrv-2021, bytemlperf-stc |
| Open Source | **Product stack entirely closed** (HPE, stcc, SHC, MLTC, STC_IE, STC_LLM, SNQ/SNC, all tools, both firmwares); `github.com/streamcomputing` is an empty account; no published license terms | confirmed | stcrp-install, github-streamcomputing |
| Open Source | Open adjacents: `riscv-stc/riscv-matrix-spec` (CC-BY-4.0, 27★, Dec 2024), `riscv-matrix-project` (39★), LLVM fork (Oct 2024), Spike fork (Feb 2025), `riscv-dnn` (Mar 2025), `riscv-pvp-matrix`, chipyard/BOOM forks; `STCPaddleModelZoo`; ByteMLPerf STC backend (Apache-2.0). **Org quiet since March 2025; none is the shipping toolchain** | confirmed | riscv-stc-org |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | **NeuralScale core (NPC)**: AndesCore N25F RV32G scalar core (5-stage in-order, 64 KB I$/D$) driving three decoupled engines; in-order issue, ≤3 instructions/cycle, stalls only on address overlap. Per-core MIMD, **not SIMT** | confirmed | carrv-2021 |
| Compute Engine | **VME** — 64 FP16 MACs + POLY unit (`exp`/`div`/`sqrt`); runs base RVV *and* custom instructions with L1/IM byte-address operands in GPRs | confirmed | carrv-2021 |
| Compute Engine | **MME** — 64 × 32 = 2048 MAC units; each MAC = 1× FP16 or 2× INT8 per cycle; DIB + Weight Buffer → Intermediate Buffer | confirmed | carrv-2021 |
| Compute Engine | **MTE** — explicit memory-transfer engine: L1↔LLB, L1↔DRAM point-to-point, and LLB broadcast to all corresponding L1 Buffers | confirmed | carrv-2021 |
| Compute Engine | P920 SoC: **32 NeuralScale cores in 4 NPC Clusters**; ARM Cortex-A53 management CPU; **128 TFLOPS FP16 / 256 TOPS INT8 @ 1.0 GHz**; TSMC 12 nm FinFET; 400 mm²; 130 W chip TDP | confirmed | carrv-2021, stcp920-manual |
| Compute Engine | **Datatypes: FP16 and INT8 only** — no BF16, FP8, FP4, INT4, or native FP32 compute path | confirmed | stcp920-manual, model-support |
| Compute Engine | RVV v0.8 config VLEN = 1024 b, ELEN = 16 b; architectural vector-register count **not disclosed** | confirmed | carrv-2021 |
| Compute Engine | Cores per cluster: **not disclosed** (4 clusters and 32 cores are stated; 8/cluster is derived) | derived | stcp920-manual, carrv-2021 |
| ISA | **RV32G + RVV v0.8 + 53 custom instructions** on custom-3 opcode `1111011` (OP-VE), fixed 32-bit encoding `funct6/dmc/rs2/rs1/dm/opm2/rd/opcode`; `opm2` = mm/m/mv/mf shape class; `{dmc,dm}` = direction | confirmed | carrv-2021 |
| ISA | **22 unprivileged vector CSRs** for shape/config (`shape_s1`, `shape_s2`, `conv_FM_in`, `mte_shape`, …), written with base RISC-V CSR instructions | confirmed | carrv-2021 |
| ISA | Vendor-custom, **NOT the ratified RISC-V matrix/IME extension**. Live in shipping silicon per the SDK's `inst_decoder.py` (`0x06ABE8FB → veadd.mv.dimw`) | confirmed | carrv-2021, mltc-guide |
| ISA | Separate **open** RISC-V matrix proposal (`riscv-stc/riscv-matrix-spec`, CC-BY-4.0, v0.5): 8 tile regs + 8 accumulators, ELEN/MLEN/RLEN/AMUL, Zmi4/Zmv/Zmi2c/Zmc2i/Zmsp. **Not implemented by the P920** | confirmed | riscv-matrix-spec |
| Data Path | Statically scheduled three-engine overlap; profiler reports `PAL(MTE/MME)`, `PAL(MTE/VME)`, `PAL(VME/MME)`, `PAL(ALL)` per NPC | confirmed | hpe-guide, mltc-guide |
| Data Path | Measured BERT trace: NPCs 95% of cycles; DMA overlap 96%; MME 78% of in-NPC cycles; MTE overlaps 92%; VME serialises 45% of its time on MME dependencies | confirmed | carrv-2021 |
| Data Path | Shape state in CSRs, not instruction operands (`CONFIG_VE_CSR`, `CONFIG_VE_BC_CSR`, `DEFINE_SHAPE`) | confirmed | hpe-guide |
| On-chip Memory | **Software-managed and non-coherent** — custom VME/MME instructions take L1/IM byte addresses in GPRs; all inter-level movement is explicit MTE/sysDMA. Only the 64 KB scalar I$/D$ are hardware caches (control path) | confirmed | carrv-2021 |
| On-chip Memory | **L1 Buffer 1.25 MiB/NPC** (1 MiB Data IO + 0.25 MiB Weight); bandwidth 512 GB/s from a single third-party-hosted vendor figure — no vendor manual states it | confirmed (capacity) / unconfirmed (BW) | stcp920-manual, bytemlperf-stc |
| On-chip Memory | **Intermediate Buffer 256 KB/NPC** (MME output / VME staging); bandwidth **not disclosed** | confirmed | carrv-2021, glossary |
| On-chip Memory | **LLB 8 MiB/cluster, 32 MiB/chip**, 8 × 4 MB banks on the NoC; bandwidth **contested**: 17 TB/s (CARRV'21) vs 256 GB/s per cluster (vendor table) — ~17× apart, unreconciled | confirmed (capacity) / unconfirmed (BW) | carrv-2021, bytemlperf-stc |
| On-chip Memory | Total on-chip SRAM ≈ **80 MiB** + 4 MiB scalar caches (derived, not vendor-stated) | derived | carrv-2021, stcp920-manual |
| Off-chip Memory | **16 GB LPDDR4X**, 256-bit @ 3733 MT/s, 4 GiB per cluster; **108 GB/s** per card manuals vs 119.4 GB/s (vendor table) vs 119.5 GB/s (derived) vs 136 GB/s theoretical (CARRV'21). Derating reason **not disclosed** | confirmed (type/capacity) / unconfirmed (exact BW) | stcp920-manual, bytemlperf-stc, carrv-2021 |
| Off-chip Memory | 2 DMA controllers per DDR subsystem (`sysdma_0`/`sysdma_1`; C0 = DDR→LLB, C1 = LLB→DDR) | confirmed | carrv-2021, hpe-guide |
| Off-chip Memory | **Bandwidth-bound consequence**: Qwen2-7B 2 cards @ 10.39 tok/s; Qwen2-72B 16 cards @ 4.98 tok/s; DeepSeek-R1-Distill-Llama-70B 16 cards @ 4.42 tok/s (vendor's own table) | confirmed | model-support |
| On-chip Interconnect | **4×4 mesh NoC**, all links bidirectional; separated planes (control 32 b, data 512 b per direction); 64 GB/s per direction per link, 128 GB/s combined @ 1 GHz; explicitly **non-coherent**. (A company blog's "4×6 mesh / 1 TB/s" summary conflicts — use CARRV) | confirmed | carrv-2021 |
| On-chip Interconnect | **HSYNC** — partitions the 32 cores into up to 16 configurable sync groups; SDK exposes only `sync()` and a 4-cluster model; group API and mapping **not disclosed** | confirmed | carrv-2021, hpe-guide |
| Host Interface / Package | PCIe 4.0 ×16; BAR0 16 GB pref / BAR2 32 MB non-pref / BAR4 1 GB pref; 1 PF, no SR-IOV; VID/SVID 0x23e2; DIDs 0x0100/0103/0104/0105/0106; single-width ¾-length full-height, 721.2 g, passive, one 8-pin aux | confirmed | stcp920-manual |
| Host Interface / Package | SMBus OOB management at 0x2A (board/chip power, temperature, PCI IDs, FW version, **power brake**, serial/part number, voltages); thermal alert 91 °C / shutdown 95 °C | confirmed | stcp920-manual |
| Scale-up Interconnect | **None on any shipping card** — PCIe P2P only. SoC has PCIE0/PCIE1 (either configurable as root complex, CARRV describes PCIE1 for chip-to-chip scaling), but no card manual documents such a link; whether it is wired is **not disclosed** | confirmed (absence in product docs) | stcp920-manual, carrv-2021 |
| Scale-up Interconnect | Vendor-measured card-to-card: DDR→DDR 888.6 MB/s, DDR→LLB 9005 MB/s, LLB→DDR 886.5 MB/s, LLB→LLB 9014 MB/s — peak ~9 GB/s, far below the PCIe 4.0 ×16 ceiling | confirmed | p2p-perf-guide |
| Scale-out Interconnect | Standard datacenter Ethernet; **no RDMA fabric, no collective-communication hardware** | confirmed | all-sdk-docs |
| Product Line | STCP920 (128 TFLOPS/256 TOPS, 160 W) · STCP950L (153/309, 160 W) · STCP950P (145/290, 150 W) · STCP980L (179/358, 160 W) · STCP980P (167/335, 150 W) — identical topology, memory, physicals and firmware; **one `npu-v1` compiler target**. Die identity **not disclosed** | confirmed (specs) / likely (same die) | five card manuals |
| Roadmap (not products) | **`npu-v2`**: only public trace is that MLTC 1.6.1 accepts `--arch=npu-v2`. Microarchitecture, cores, node, foundry, memory, datatypes, TDP, tape-out/sampling and name are all **not disclosed** — must not be registered as silicon | unconfirmed | python-api, corporate-site |
| Roadmap (not products) | **"RISC-V AI 超节点"** — marked 研发中 on the vendor site; "CPU+NPU+DPU AI-native computing unit"; no topology, node count, interconnect or date | unconfirmed | corporate-site |

## Source Keys

| Key | URL |
|-----|-----|
| carrv-2021 | https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf |
| stcp920-manual | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP920产品手册 |
| five card manuals | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950L产品手册 (also STCP950P / STCP980L / STCP980P 产品手册) |
| stcrp-overview | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP产品简介 |
| stcrp-install | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP安装指南 |
| stcrp-release-notes | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP_Release_Notes |
| hpe-guide | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南 |
| mltc-guide | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/MLTC使用指南 |
| stc-llm-guide | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/STC_LLM使用指南 |
| glossary | https://docs.streamcomputing.com/AI加速卡/希姆计算术语表 |
| cpp-api | https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API |
| python-api | https://docs.streamcomputing.com/AI加速卡/开发者资源/Python_API |
| mltc-op-support | https://docs.streamcomputing.com/AI加速卡/开发者资源/MLTC算子支持说明 |
| model-support | https://docs.streamcomputing.com/AI加速卡/开发者资源/模型支持说明 |
| p2p-perf-guide | https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/p2p_perf使用指南 |
| npu-viewer-guide | https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/NPU-Viewer使用指南 |
| firmware-release-notes | https://docs.streamcomputing.com/AI加速卡/固件使用手册/Firmware_Release_Notes |
| k8s-plugin-guide | https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/k8s-device-plugin使用指南 |
| npu-exporter-guide | https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/npu-exporter使用指南 |
| cloud-platform-docs | https://docs.streamcomputing.com/智算云平台/希姆云平台使用手册/希姆云平台介绍 |
| bytemlperf-stc | https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md |
| riscv-stc-org | https://github.com/orgs/riscv-stc/repositories |
| riscv-matrix-spec | https://github.com/riscv-stc/riscv-matrix-spec |
| riscv-stc-llvm | https://github.com/riscv-stc/llvm-project |
| stcpaddlemodelzoo | https://github.com/Stream-Computing/STCPaddleModelZoo |
| corporate-site | https://www.streamcomputing.com/ |
| github-streamcomputing | https://github.com/streamcomputing |
| all-sdk-docs | Absence verified by grep across all 22 retrieved SDK doc pages under https://docs.streamcomputing.com/AI加速卡/ |
