# Huawei Ascend Software and Hardware Stack Summary

*as_of: 2026-09-13*
*(baseline sections written 2026-04-05; see "Ascend 950 Generation Update (2026-08-08)" and "Update (2026-09-13)" below)*

---

## Overview

Huawei Ascend NPUs are China's leading domestic AI accelerators, built around the **Da Vinci architecture** — HiSilicon's proprietary heterogeneous AI core with five parallel execution units: a systolic matrix Cube engine, a SIMD Vector engine, a Scalar controller, and two Memory Transfer Engines (MTE1/MTE2). The current production generation is **Ascend 910C** (Da Vinci 3.0, dual-die MCM, ~800 TFLOPS FP16), with the 910B (Da Vinci 2.0, SMIC N+1) widely deployed in Chinese datacenter clusters. At the system level, the **CloudMatrix 384** cluster (384 × Ascend 910C, 6,912 × 400G silicon photonic optical links, 2.8 Tbps per chip) represents Huawei's rack-scale answer to NVIDIA GB200 NVL72.

The software ecosystem is anchored by **CANN** (Compute Architecture for Neural Networks), a multi-layer SDK spanning the AscendCL runtime, ATC offline compiler, TBE/AscendC kernel development framework, and HCCL collective communications library. The two primary framework entry points are **MindSpore** (Huawei's native framework, source-to-source AD, native Ascend optimization) and **torch_npu** (PyTorch PrivateUse1 backend adapter, pip-installable, zero-code-change migration from CUDA). Both ultimately dispatch through CANN to the Da Vinci hardware.

---

## Software Stack

### Framework Integration

Huawei Ascend supports AI model development through two primary framework paths:

- **MindSpore**: Huawei's full-scenario deep learning framework (open-source, Apache 2.0, hosted on Gitee). Uses source-to-source automatic differentiation — Python `nn.Cell` subclasses are traced into a **MindIR** computation graph, optimized through the Graph Engine (GE) layer, then lowered via CANN to TBE/AscendC operators on Da Vinci hardware. Supports `GRAPH_MODE` (AOT, static shapes, maximum optimization) and `PYNATIVE_MODE` (eager for debugging). Auto-parallelism (`set_auto_parallel_context()`) automatically partitions models across HCCS-connected NPUs with tensor, pipeline, and expert (MoE) parallel strategies. The **MindFormers** library provides production transformer training; **MindSpeed** provides Megatron-style large model acceleration.

- **torch_npu (PyTorch Ascend Adapter)**: Implements PyTorch's PrivateUse1 backend mechanism, registering `torch.npu` as a device alongside `torch.cuda`. Install via `pip install torch-npu`; change `device="cuda"` → `device="npu"` for zero-code migration. Supports `torch.compile`, FSDP, DTensor, and (since Jan 2025) Torchtune LLM fine-tuning. Maintained with LTS branches tracking PyTorch 2.1–2.5. Primary source on Gitee (`gitee.com/ascend/pytorch`).

- **ONNX Runtime CANN Execution Provider**: ONNX model execution via AscendCL; C/C++ and Python APIs.

- **OpenCV CANN Backend**: Vision preprocessing pipeline via Ascend DVPP hardware unit.

### Compiler / IR

- **ATC (Ascend Tensor Compiler)**: The offline AOT model compiler, analogous to TensorRT. Converts ONNX, TensorFlow, Caffe, MindIR, and single-operator JSON models to `.om` (Offline Model) format. Optimization passes: operator fusion (Conv+BN+ReLU, MatMul+BiasAdd+Act), INT8/INT4 quantization, memory layout conversion to NC1HWC0 5D format, SRAM placement planning, and AIPP hardware image preprocessing configuration. The `.om` binary is self-contained (operator code + weights + execution graph + memory plan) and loaded at inference time via `aclmdlLoadFromFile()` — no runtime JIT.

- **MindSpore Graph Compiler + Graph Engine (GE)**: JIT/AOT compiler from MindIR to CANN operators. Performs cross-layer fusion and HCCS-aware collective schedule optimization.

- **TileLang-Ascend**: MLIR-based tiled kernel DSL adapter for Ascend; allows writing high-performance kernels from Python tile specifications targeting Da Vinci hardware.

### Op Library

- **CANN Standard Operator Library (Ascend OL)**: 1,000+ pre-built, hardware-optimized operators covering Conv2D, MatMul, BatchNorm, LSTM, Multi-Head Attention, elementwise, and reduction ops. Dispatched via AscendCL `aclopExecuteV2` or transparently through framework dispatch layers.

- **AIPP (AI Pre-Processing Module)**: Hardware image preprocessing pipeline baked into `.om` models at ATC compile time. Handles YUV/NV12 → RGB8 color conversion, mean/variance normalization, crop, resize — offloads preprocessing from CPU and avoids HBM round-trips.

### Kernel Library

- **TBE-DSL (Tensor Boost Engine Domain-Specific Language)**: High-level Python DSL for operator authoring. Developer describes computation; TBE compiler automatically generates tiling, double-buffering, and Da Vinci instruction scheduling. Suitable for regular-shaped new operators.

- **TBE-TIK (Tensor Iterator Kernel)**: Lower-level Python DSL with explicit Da Vinci hardware control: tensor tiling, L1/UB buffer allocation, MTE transfer scheduling, Cube/Vector instruction invocation (`data_move()`, `matmul()`, `vconv()`, `vadd()`). The performance path for hand-tuned operators.

- **AscendC**: Announced 2023. C/C++-compatible kernel language for Da Vinci. Provides typed memory abstractions (`GlobalTensor` for HBM, `LocalTensor` for on-chip), explicit DMA calls (`DataCopy()`), Cube invocation (`Matmul()`), and double-buffer pipeline synchronization (`EnQue`/`DeQue`). Eliminates manual intra-core barrier management vs TIK. Most similar to CUDA C++ in developer experience.

### Runtime

- **AscendCL (Ascend Computing Language)**: The primary C-language runtime API. Manages device contexts (`aclrtSetDevice`), HBM memory (`aclrtMalloc`/`aclrtMemcpy`), asynchronous streams (`aclrtCreateStream`/`aclrtSynchronizeStream`), model loading/execution (`aclmdlLoadFromFile`/`aclmdlExecute`), and custom operator dispatch (`aclopExecuteV2`). Analogous to CUDA Runtime + Driver APIs combined. Python `acl` bindings available.

### Driver / Firmware

- **Ascend NPU Kernel Driver**: Linux kernel module providing PCIe device management, BAR mapping, command queue submission, and IRQ handling. Not fully open-sourced (unlike NVIDIA's open-gpu-kernel-modules). Bundled with CANN SDK.

- **Ascend Firmware**: Closed on-chip firmware for resource management; TEE (Trusted Execution Environment) integration studied in Ascend-CC confidential computing research.

### Communication

- **HCCL (Huawei Collective Communications Library)**: Analogous to NCCL. Provides AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv. Topology-aware: selects HCCS transport (intra-server, high-bandwidth) vs RoCE v2 RDMA (inter-server). Integrated as `torch.distributed backend="hccl"` for PyTorch and via `mindspore.communication.init()` for MindSpore.

- **HCCS (Huawei Compute Communication System)**: Proprietary scale-up intra-server bus. Higher bandwidth than PCIe switch. 8 Ascend NPUs per Atlas 800T server via HCCS. *(Superseded for the 950 generation: the scale-up fabric is now branded **UnifiedBus / 灵衢 (Lingqu) 2.0**, launched at Huawei Connect on 2025-09-18 and published as an open specification. The earlier "HCCS 4.0 → 100,000-chip" framing in this document predates that rebrand — see the 2026-08-08 update section.)*

### Assembler / ISA

- **CCE (Core Compute Engine) Instruction Set**: Proprietary Da Vinci ISA. Five parallel instruction streams: Cube (matrix ops), Vector (element-wise), Scalar (control), MTE1 (input DMA), MTE2 (output DMA). Statically scheduled; no virtual ISA equivalent to PTX. Documentation is partial (Da Vinci architecture paper; no full ISA spec published). Developer access is through AscendC language primitives and TIK Python API.

---

## Hardware Architecture

### Compute Engine

Each Da Vinci AI Core contains: Cube Unit (16×16×16 FP16 systolic = 4,096 MACs/cycle), Vector Unit (128-lane FP16 or 256-lane INT8/cycle), Scalar Unit (micro-CPU), MTE1 (input DMA), MTE2 (output DMA). All five execute concurrently under static CCE scheduling.

Chip generations: 32 cores (Ascend 910, 256 TFLOPS FP16) → 25 cores (910B, 320 TFLOPS) → ~64 effective cores (910C dual-die, ~800 TFLOPS) → 910D (roadmap, HBM3, SMIC advanced).

### Data Path

Static 5-stream CCE pipeline with no dynamic out-of-order execution. Double-buffering (MTE1 prefetches next tile while Cube processes current) is the primary utilization technique. Tiling granularity is constrained by L0A/L0B/L0C buffer sizes (16/16/32 KB).

### On-chip Memory

84 MB total SRAM (Ascend 910): per-core L0A/L0B/L0C/L1/UB buffers + 32 MB shared L2 at 4 TB/s NoC bandwidth. The L2:HBM bandwidth ratio (4 TB/s : 1.2 TB/s) makes L2 reuse ~3.3× more valuable than HBM bandwidth.

### Off-chip Memory

HBM2 32 GB 1,228 GB/s (910) → HBM2e 64 GB ~800 GB/s (910B, SMIC-constrained) → HBM2e 128 GB ~1.6 TB/s (910C dual-die). Roadmap targets HBM3/3e with 910D (2025+).

### Host Interface / Package

PCIe 4.0 × 16 (~64 GB/s). Ascend 910C: organic MCM with 2× SMIC N+1 NPU dies + TSMC 7nm CPU companion + 8× HBM2e stacks. No silicon interposer (cost constraint of SMIC process).

### Scale-up Interconnect

HCCS: proprietary intra-server bus (8 NPUs/server in Atlas 800T). CloudMatrix 384: 56 × 400G SiPh LPO per server, 2.8 Tbps/chip, full-mesh all-to-all optical across 384 chips in 16 racks. For the 950 generation this fabric is branded **UnifiedBus / 灵衢 2.0** (open specification, launched 2025-09-18); the Ascend 950 whitepaper cites cluster scale beyond 128K cards, superseding the older "HCCS 4.0 → 100,000-chip" line.

### Scale-out Interconnect

CloudMatrix 384: 8 × 400G SiPh LPO per server for scale-out; 3,168 total fibers; full-mesh optical between racks (4 switch racks for 12 compute racks). All-optical interconnect (no copper backplane at scale).

---

## Programming Model Rationale

**1. Static CCE scheduling demands compiler-visible pipeline.** The Da Vinci core has no dynamic scheduler. TBE/AscendC must explicitly express MTE1→Cube→MTE2 pipeline overlap via `EnQue`/`DeQue` barriers, or the compiler inserts synchronizations that serialize the five units. This is the primary reason TBE-TIK and AscendC exist: they provide the pipeline expressivity that a pure Python-level framework API cannot.

**2. The 5-level memory hierarchy constrains tile sizes.** The Cube unit reads from 16 KB L0A/L0B and writes to 32 KB L0C. Any matrix tile that doesn't fit these buffers spills to L1/UB with MTE overhead. The 512 KB L1 per core acts as a staging area. Optimal kernels tile GEMM to exactly fill L0A/L0B while double-buffering from L1. This is why AscendC exposes named address spaces (`GlobalTensor`, `LocalTensor` with memory-level tag) rather than a flat address space.

**3. ATC offline compilation decouples deployment from JIT.** Unlike CUDA's fat-binary + PTX JIT fallback, Ascend's production deployment model pre-compiles the full model (including quantization, fusion, AIPP, and memory layout decisions) to a `.om` at deployment time. This eliminates JIT latency but requires re-compilation for shape changes. The tradeoff matches Ascend's inference-heavy deployment profile (Atlas 300 inference cards).

**4. MindSpore S2S AD enables cluster-aware optimization.** Unlike PyTorch's per-layer eager dispatch, MindSpore traces the full forward+backward graph as MindIR before any computation. The GE layer can therefore fuse across the forward/backward boundary, schedule collective communications (AllReduce for data-parallel gradients) to overlap with compute, and plan HCCS vs RoCE transport selection — optimizations impossible in eager mode.

**5. torch_npu PrivateUse1 is the ecosystem bridge.** By implementing PyTorch's PrivateUse1 device backend, Huawei allows the vast PyTorch ecosystem (vLLM, DeepSpeed, HuggingFace Transformers, Torchtune) to run on Ascend with minimal porting effort. The `device="npu"` change is the full user-visible migration surface; CANN handles all hardware dispatch below.

**6. CloudMatrix 384's all-optical fabric reflects HCCS bandwidth ceiling.** HCCS provides high bandwidth within an 8-chip server, but scaling beyond a single server previously required dropping to RoCE 100 GE (Atlas 900). The 910C's CloudMatrix design bypasses this by replacing the HCCS intra-pod fabric with silicon photonic LPO — achieving 2.8 Tbps per chip for 384 chips at the cost of ~6,912 optical transceivers. This is architecturally similar to NVIDIA's NVLink-C2C / NVSwitch strategy but implemented optically due to the lack of a high-volume silicon interposer process at SMIC.

---

## Ascend 950 / Next-Generation Roadmap

*Sources: Huawei Connect September 2025 keynote, TrendForce, Tom's Hardware, MWC 2026 announcements.*

### Chip Roadmap (2025–2028)

All figures below are **chip-level** specifications as announced. The shipping *card* (Atlas 350, built on 950PR) is derated versus the chip spec — see the 2026-08-08 update section.

| Year | Chip | Variant | Memory | BW | FP8 Compute | MXFP4 Compute |
|------|------|---------|--------|----|------------|--------------|
| 2025 | Ascend 910C | (production) | 128 GB HBM2e | ~1.6 TB/s | — | — |
| 2026 Q1 | Ascend 950PR (chip spec) | Prefill & Recommend | 128 GB HiBL 1.0 | 1.6 TB/s | 1 PFLOPS | 2 PFLOPS |
| 2026 Q4 | Ascend 950DT (chip spec) | Decode & Train | 144 GB HiZQ 2.0 | 4 TB/s | 1 PFLOPS | 2 PFLOPS |
| 2027 Q4 | Ascend 960 | TBD | TBD | TBD | 2 PFLOPS | 4 PFLOPS (est.) |
| 2028 | Ascend 970 | TBD | TBD | TBD | target: 4 ZF cluster | — |

### Ascend 950 Key Changes vs 910C

- **Architecture**: Da Vinci 4.0 — new SIMD+SIMT hybrid execution model
- **Precision**: FP8, MXFP8, HiF8 (Huawei-proprietary), MXFP4 — 1 PFLOPS FP8 / 2 PFLOPS MXFP4 per chip
- **Memory independence**: In-house HiBL 1.0 (950PR, 128 GB / 1.6 TB/s) and HiZQ 2.0 (950DT, 144 GB / 4 TB/s), bypassing export-controlled foreign HBM supply
- **Interconnect**: Scale-up bandwidth 2.5× to **2 TB/s** (vs 910C HCCS); the fabric is branded **UnifiedBus / 灵衢 2.0**
- **Atlas 350 card**: the shipping 950PR card is rated **1.56 PFLOPS FP4, up to 112 GB, 1.4 TB/s, 600 W** (Zhang Dixuan, 2026-03-23). The "~2.8× NVIDIA H20" figure is a **Huawei marketing comparison**, not an independently measured result.
- **Atlas 950 SuperPoD**: 8,192 **Ascend 950DT** chips, 16 EFLOPS FP4 (announced MWC 2026; a 1,024-card subset was physically demonstrated at WAIC 2026)
- **Roadmap milestone**: 1 million-chip supernode cluster target by 2027 (百万卡超节点集群)

### CANN SDK Evolution for 950 Series

- New data types registered: FP8, MXFP8, HiF8, MXFP4 in AscendC / TBE operator library
- ATC quantization pass extended to emit HiF8/MXFP4 Cube instructions
- HCCL updated for 2 TB/s interconnect topology and 8,192-chip SuperPoD scale
- torch_npu 2.x: FP8 autocast and MXFP4 quantization paths added
- MindSpore auto-parallelism updated for 8,192-chip cluster scheduling

---

## Ascend 950 Generation Update (2026-08-08)

*Updated 2026-08-08. Sources: CANN 9.0.0 commercial release notes (hiascend.com), MindSpore 2.9.0 release (mindspore.cn, PyPI), Ascend 950 NPU architecture whitepaper (Huawei OBS, ~38 pp), Huawei WAIC 2026 release (2026-07-17), Huawei Cloud INSPIRE 2026 reporting (TrendForce / Huawei Central, 2026-06-06/08), vllm-ascend release notes, GitCode CANN organization.*

Between April and August 2026 the Ascend 950 generation moved from a roadmap entry into shipping silicon with a matching software release. Everything above about the 910 / 910B / 910C generations and CloudMatrix 384 is unchanged and remains current. What follows is what is new — plus two corrections to figures that predate this repo's 2026-04-05 baseline.

### 1. Software: CANN 9.0.0 and MindSpore 2.9.0 (May 2026)

**CANN 9.0.0 (commercial)** was released **2026-05-09** (betas 2026-03-09 and 2026-03-30). This is the first CANN release with first-class Ascend 950 support; the release notes state verbatim *"CANN新增适配Ascend 950PR（Atlas 350加速卡）"*.

| Area | Change in CANN 9.0.0 |
|---|---|
| Data types | FP8, MXFP8, MXFP4 added (HiF8 is part of the 950 precision set) |
| AscendC | *"AscendC支持SIMD+SIMT混合编程"* — hybrid SIMD+SIMT kernel programming with roughly **700 new SIMT API entry points** (warp, atomic, math, type-conversion groups) |
| AscendC | Reg-based (register-level) programming path exposed specifically on Ascend 950PR |
| HCCL | Batched-collective merge via `HcclGroupStart()` / `HcclGroupEnd()`; *"集合通信支持CCU通信加速"* (collective acceleration through the on-chip CCU) |
| CATLASS | Template library upgraded for 950-series tensor/vector co-compute paths and new Fixpipe move modes |
| Packaging | apt and pip installs added (now conda / yum / apt / pip) with a unified download bundle; Huawei **markets** a drop from ~2 h to ~45 min deployment time — a vendor claim, not an independently measured figure |

**MindSpore 2.9.0** was released **2026-05-07**, paired with CANN 9.0.0, published on mindspore.cn and PyPI, with native Ascend 950PR support. (Note for future scans: the Gitee mirror's release list lagged at v2.7.2 / CANN 8.5.0 — treat mindspore.cn and PyPI as the canonical version sources. The v2.7.2 notes also record that the *"CANN 8.5.0 package has completed the upgrade to an open-source architecture"*, which changed CANN naming conventions and install directories.)

**vllm-ascend** — not previously covered in this summary — is now the fastest-moving Ascend serving surface and the place where 950 enablement lands first:

| Release | Ascend-relevant content |
|---|---|
| v0.18.0 / v0.19.1rc1 (2026-04-30) | baseline 950-era branches |
| v0.20.2rc1 (2026-06-03) | upgrades to CANN 9.0.0; MXFP4 quantization on Ascend 950; DeepSeek-V4 DSA attention |
| v0.21.0rc1 (2026-06-16) | full end-to-end DeepSeek-V4 on Ascend 950; `FULL_AND_PIECEWISE` hybrid graph compilation mode |
| v0.22.1rc1 (2026-06-30) | Ascend 950 dynamic quantization; Mooncake connector for DeepSeek-V4 hybrid KV cache; HCCL weight transfer for RL |
| v0.23.0rc1 (2026-07-19) | aligns with upstream vLLM v0.23; W4A16 MXFP4 and all-gather-EP MXFP4. **Requires CANN 9.0.1** (not 9.0.0) for Ascend 950 — CANN has already moved past 9.0.0 |

**CANN open-sourcing is now live**, superseding this document's earlier note that CANN internals are closed. `gitcode.com/cann` hosts a multi-repo organization of ~76 projects actively committed through early August 2026: operator libraries (`ops-transformer`, `ops-math`, `ops-nn`, `ops-cv`), communication (`hccl`, `hcomm`, `shmem`), the Ascend C dev kit (`asc-devkit`), Python tooling (`pypto`, `pyasc`), `runtime`, the graph engine (`ge`), `graph-autofusion`, `cann-recipes-infer` / `cann-recipes-train`, `cann-bench`, and `amct`. The **kernel driver and on-chip firmware remain closed** — only the userspace layers were opened.

### 2. Hardware: Ascend 950 NPU architecture whitepaper (late May / early June 2026)

Huawei published a ~38-page *昇腾950 NPU架构白皮书* on its OBS public-download endpoint. Dating: the PDF carries `Last-Modified: 2026-06-04`, Chinese trade press covered it on 2026-06-11, and an independent analyst digest of its contents is dated 2026-05-22 — so **late May / early June 2026**, not July. The PDF is image-only, so its contents reach this survey through analyst readings rather than extracted vendor text; the confidence split below reflects that.

**Independently corroborated:**

- **Cube and Vector are now separate cores.** The 950 splits the Da Vinci core into independent **AIC (Cube)** and **AIV (Vector)** cores, organized as **1 Cube + 2 Vector per AI subsystem**. This is a real break from the 910-era five-units-in-one-core design.
- **A new 128 MB global L2** spanning the chip — a memory level that does not exist on 910B/910C — with roughly 2× per-access improvement and a per-way cache lock / residency policy.
- Memory: **128 GB / 1.6 TB/s (950PR)**, **144 GB / 4 TB/s (950DT)** — unchanged from the roadmap.
- **~2 TB/s inter-chip** bandwidth over HiLink SerDes; **PCIe 5.0 ×16** host interface (up from PCIe 4.0 on 910C); **dual 400 Gbps UBoE** (UnifiedBus over Ethernet).
- **STARS 2.0** hardware scheduler arbitrating across AIC / AIV / CPU / DVPP / SDMA / UB / CCU.
- **NDDMA** as a lightweight AIC↔AIV data path.

**Naming note:** the whitepaper describes this as the **3rd-generation Da Vinci** architecture, whereas press coverage (and this document's earlier sections) call it **Da Vinci 4.0**. The two labels refer to the same silicon; the discrepancy is in vendor-vs-press numbering, and Huawei has not published a reconciliation.

**Single-source, NOT independently confirmed** — recorded here for future verification, not to be cited as spec: 18 Da Vinci cores per AI die / 36 Cube + 72 Vector per chip; L0A/L0B at 64 KB, L1 at 512 KB, UB at 512 KB; a 512 B / 4×128 B sector cache; BufferID-based synchronization replacing the `EnQue`/`DeQue` idiom; a "Linx816" on-chip AI CPU (4 clusters × 2 ARMv8-A cores, 4 MB L3 per cluster); and the UnifiedBus RTP/CTP port split (2,016 GB/s = 448 GB/s RTP over 4 ports + 1,008 GB/s CTP over 9 ports, 18 × X4 HiLink at 112 Gbps/lane).

**Compute ladder.** Independent sources give only the rounded **1 PFLOPS FP8 / 2 PFLOPS MXFP4** per chip for both 950PR and 950DT — corroborated by Huawei's own 1,024-card = 1 EFLOPS FP8 arithmetic. A single third-party analysis additionally reports **~547 TFLOPS BF16/FP16** and **~273 TFLOPS TF32**; those two rungs are **single-source estimates**, not confirmed specs. The BF16/FP16 and TF32 rungs are otherwise **not disclosed** by Huawei.

### 3. Systems: Atlas 950 SuperPoD physically demonstrated at WAIC 2026

At WAIC 2026 (Shanghai, **2026-07-17 to 07-20**) Huawei showed Atlas 950 SuperPoD hardware for the first time — its own release (2026-07-17) labels it 真机首次公开亮相, "first public physical appearance," and the system took the conference SAIL award.

| Demonstrated configuration (WAIC 2026) | Value |
|---|---|
| Cards shown | 1,024 |
| Compute | 1 EFLOPS FP8 / 2 EFLOPS FP4 |
| Globally unified memory address space | 256 TB |
| Fabric RTT | 3 μs over the 灵衢 (Lingqu) protocol |
| Status | **DEMONSTRATED, not shipping** — commercial delivery remains Q4 2026 |

The full-scale target — **8,192 Ascend 950DT cards at 16 EFLOPS FP4** — is unchanged from the MWC 2026 announcement already recorded above. The same release notes an air-cooled variant, **Atlas 850E**, at 96-card commercial deployment (MWC 2026 material describes it scaling 8 → 1,024 NPUs for conventional datacenters), and 750+ commercial deployments of the older Atlas 384-series supernodes. MWC 2026 also gave the packaging rule for Atlas 950: **64 NPUs per cabinet**, scaling to 8,192.

### 4. Schedule: Ascend 950DT deployment pulled forward to August 2026 (announced)

At the Huawei Cloud 2026 INSPIRE Creators event, Huawei VP **Chen Lin** said Ascend 950DT deployment on Huawei Cloud would be pulled forward from Q4 2026 to **August 2026** (reported 2026-06-06/08). Important qualifications:

- This is an **announced schedule, not an accomplished deployment**. As of 2026-08-08 there is no confirmation the August deployment has gone live.
- **Q4 2026 remains the official commercial release date.**
- Chen Lin's own description was qualitative — "greatly improve vector compute, memory bandwidth, natively support FP8" — not a spec disclosure.
- Restated specs at that event: 144 GB, 4 TB/s, FP8/MXFP8/MXFP4/HiF8.
- **Process node: not disclosed.** "SMIC N+3" is analyst inference only (low confidence); Huawei has never stated the node.
- DeepSeek as an expected early adopter (with DeepSeek V4 already running on the Ascend 950 platform) is **press reporting**, not a Huawei statement.

### 5. Corrections to pre-baseline figures (baseline misses, not new news)

Two items below were public **before** this repo's 2026-04-05 baseline and were missed at the time. They are corrections, not window updates.

**(a) Atlas 350 shipping card vs Ascend 950PR chip spec** — announced 2026-03-23/24 by Huawei Ascend computing president **Zhang Dixuan**:

| Figure | Ascend 950PR (chip spec, as recorded above) | Atlas 350 (shipping card) |
|---|---|---|
| FP4 compute | 2 PFLOPS MXFP4 | **1.56 PFLOPS FP4** |
| Memory | 128 GB HiBL 1.0 | **up to 112 GB** |
| Bandwidth | 1.6 TB/s | **1.4 TB/s** |
| TDP | not previously recorded | **600 W** (~1.5× H20) |

The "**2.8× H20**" headline is a **Huawei marketing comparison**, not an independently confirmed benchmark result.

**(b) UnifiedBus / 灵衢 (Lingqu) 2.0 supersedes the "HCCS 4.0" framing.** The 950-generation scale-up fabric is UnifiedBus, launched by Huawei rotating chairman **Xu Zhijun on 2025-09-18 at Huawei Connect 2025** — roughly seven months before this repo's baseline. Huawei published it as an **open specification** (base spec, firmware spec, software reference designs) at unifiedbus.com, with an openEuler *UB Service Core* software architecture reference design. The Ascend 950 whitepaper cites cluster scale **beyond 128K cards**, superseding this document's earlier "HCCS 4.0 targets 100,000-chip cluster scale" line.

**(c)** Da Vinci 4.0's SIMD+SIMT hybrid execution model was already recorded in this document at baseline. What is new in this window is its **software delivery** in CANN 9.0.0, not the architectural fact.

### 6. Open items and non-findings

- **Hot Chips 38 runs 2026-08-23 to 08-25 — 15 days in the future as of this update.** No Ascend content from it exists yet; any Ascend talk there is *disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public*.
- MLPerf Inference v6.0 results were published 2026-04-01 with 24 submitting organizations. The submitter list could not be enumerated, so "no Huawei Ascend submission" is **plausible but unverified** — it is not recorded here as a confirmed negative.
- No evidence of an **Ascend 960 or 970** schedule change since baseline.
- The **"910D"** name continues to appear only in older press; nothing retrieved contradicts this repo's existing note that it was superseded by 950-series branding.
- Direct text extraction from the Ascend 950 whitepaper PDF (image-only) remains the highest-value next step — it would resolve most of the single-source microarchitecture items in §2.

---

## Update (2026-09-13)

*Scan window 2026-08-08 → 2026-09-13. Sources: Bloomberg (2026-09-04, via secondary coverage), The Decoder (2026-09-04), TechNode (2026-09-07), Huawei Central (2026-09-10), Huawei Connect 2026 event page.*

### 1. Commercial demand signal: DeepSeek orders ≥160,000 Ascend 950DT accelerators

Bloomberg reported (2026-09-04) that **DeepSeek plans to deploy at least 160,000 Huawei Ascend 950DT accelerators** at a new gigawatt-scale data center it is building in **Inner Mongolia**, described by secondary coverage as the largest known Huawei AI chip cluster to date. Key qualifiers, all **press-reported, not vendor-confirmed**:

- The deployment is described as **inference-only**; DeepSeek reportedly continues to use Nvidia hardware for training.
- Bloomberg/The Decoder report Huawei **cannot deliver the full order for over a year**, citing production constraints and HBM memory shortages (with China's CXMT cited as still 3–5 years behind Samsung/SK hynix/Micron in HBM).
- This is a **reported order/plan**, not a confirmed shipped deployment — it does not change the Ascend 950DT's official Q4 2026 commercial-release date recorded above, and it does not resolve whether the Huawei Cloud "August 2026" pull-forward (§4 above) actually went live; no confirmation of that was found in this window either.
- No new hardware specs accompany this report; 144 GB / 4 TB/s / FP8-MXFP8-MXFP4-HiF8 figures are unchanged.

Sources: [Bloomberg, "DeepSeek Plans Big Huawei AI Chip Order to Power New Data Center" (2026-09-04)](https://www.bloomberg.com/news/articles/2026-09-04/deepseek-plans-big-huawei-ai-chip-order-to-power-new-data-center), [The Decoder (2026-09-04)](https://the-decoder.com/deepseek-plans-the-largest-known-huawei-chip-cluster-with-160000-processors-in-inner-mongolia/), [TechNode (2026-09-07)](https://technode.com/2026/09/07/).

### 2. Ascend 950DT price increase (press-reported)

Huawei Central (2026-09-10, citing Bloomberg) reports Huawei notified clients of an **Ascend 950DT price increase of ~60% over three months** — from roughly **¥150,653 to ¥250,000** per unit — attributed to strong demand, limited HBM supply raising production cost, and a stated intent to price closer to Nvidia's B200. This is **press-reported, not independently verified against a Huawei price list**, and no unit is specified (per-card vs. per-chip ambiguous in the secondary reporting). Source: [Huawei Central, "Huawei Ascend 950DT price jumped 60% over past three months" (2026-09-10)](https://www.huaweicentral.com/huawei-ascend-950dt-price-jumped-60-over-past-three-months/).

### 3. Huawei Connect 2026 — not yet held as of this scan

Huawei Connect 2026 is scheduled for **2026-09-17 to 09-19** in Shanghai (World Expo Exhibition & Convention Center / Shanghai Expo Center) — **after** this scan's 2026-09-13 cutoff. No Ascend-specific agenda content was available at fetch time; any HC 2026 disclosures (Ascend 960 roadmap, 950DT GA, etc.) remain for the next scan window. Source: [Huawei Connect 2026 event page](https://www.huawei.com/en/events/huaweiconnect).

### 4. MindSpore 2.10.0 (pre-window release, missed at 2026-08-08 baseline)

**MindSpore 2.10.0** was published on PyPI **2026-07-31** — before the 2026-08-08 baseline cutoff but not captured then. It is the current stable release (2.9.0 is now the prior "maintained" line). Visible release-note highlights: per-rank sharded Distributed Checkpoint (DCP) with cross-strategy reshard loading, AKG compiler integration with "AscendNPU IR" for fused-operator generation on the Ascend backend, and two-level on-chip MPMD parallelism for MoE communication masking. No explicit Ascend 950-only feature was identified; treat as a correction to the record, not new-in-window news. Source: [MindSpore release history — PyPI](https://pypi.org/project/mindspore/#history), [MindSpore 2.10 version notes](https://www.mindspore.cn/version-updates/en/2_10_en).

### 5. No findings

No evidence of a new Ascend chip SKU, CANN version beyond 9.0.x, or Ascend 960/970 schedule change in this window. The CANN GitCode organization (`gitcode.com/cann`) remains under active multi-repo development. Hot Chips 38 (2026-08-23 to 08-25) is now in the past; no Ascend/Huawei talk or slide deck was located in this scan (consistent with the task brief that none of this batch's chips had a Hot Chips 38 talk).

---

## Resources

### Documentation
- [CANN Documentation Portal](https://support.huawei.com/enterprise/en/ascend-computing/cann-pid-251168373)
- [ATC Tool Documentation](https://support.huawei.com/enterprise/en/doc/EDOC1100192457/2a40d134/atc-tool)
- [TBE-TIK Introduction](https://support.huawei.com/enterprise/en/doc/EDOC1100164820/83779921/tik-introduction)
- [AscendCL API Guide](https://developer.huawei.com/consumer/en/doc/hiai-guides/introduction-0000001051486804)
- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html)
- [Ascend AI Processor Architecture (O'Reilly)](https://www.oreilly.com/library/view/ascend-ai-processor/9780128234891/B9780128234884000035.xhtml)

### Open-Source Repositories
- [torch_npu (Ascend/pytorch)](https://github.com/Ascend/pytorch) — PyTorch Ascend adapter
- [MindSpore (Gitee)](https://gitee.com/mindspore/mindspore) — Primary MindSpore source
- [MindSpore Models](https://gitee.com/mindspore/models) — ModelZoo (Ascend/GPU/CPU)
- [Huawei-Ascend/samples](https://github.com/Huawei-Ascend/samples) — AscendCL sample code
- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) — MLIR tiled kernel DSL for Ascend
- [CANN Installer (hpcaitech)](https://github.com/hpcaitech/CANN-Installer) — CANN install helper

### Architecture Papers
- [DaVinci: A Scalable Architecture (CMC PDF)](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf)
- [Performance Modeling on DaVinci AI Core (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S074373152300014X)
- [MindSpore White Paper (PDF)](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/white_paper/MindSpore_white_paper_enV1.1.pdf)
- [Serving LLMs on CloudMatrix 384 (arXiv)](https://arxiv.org/html/2506.12708v3)
- [Ascend-CC: Confidential Computing on NPU (arXiv)](https://arxiv.org/html/2407.11888v1)

### Hardware Analysis
- [Ascend 910B Examined — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910)
- [CloudMatrix 384 Analysis — SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72)
- [TechInsights 910C Teardown — SemiWiki](https://semiwiki.com/forum/threads/techinsights-teardown-huawei-ascend-910c-still-contains-cpu-dies-from-tsmc-from-2020.23737/)
- [Ascend NPU Roadmap — Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-ascend-npu-roadmap-examined-company-targets-4-zettaflops-fp4-performance-by-2028-amid-manufacturing-constraints)

### Ascend 950 Generation (added 2026-08-08)
- [CANN 9.0.0 Commercial Release Notes — hiascend.com](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/release-notes.md) — verbatim Ascend 950PR / Atlas 350 support, SIMD+SIMT AscendC, HCCL group API, packaging changes
- [Ascend 950 NPU Architecture Whitepaper (PDF, image-only, ~38 pp)](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf) — Huawei OBS; `Last-Modified: 2026-06-04`
- [MindSpore Release Notes (canonical)](https://www.mindspore.cn/docs/en/stable/RELEASE.html) and [MindSpore 2.9 version notes](https://www.mindspore.cn/version-updates/en/2_9_en) — 2.9.0, 2026-05-07, paired with CANN 9.0.0
- [vllm-ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html) / [vllm-ascend GitHub releases](https://github.com/vllm-project/vllm-ascend/releases) — MXFP4, DeepSeek-V4, CANN 9.0.1 requirement
- [CANN open-source organization — GitCode](https://gitcode.com/cann) — ~76 repos: ops-*, hccl/hcomm/shmem, asc-devkit, pypto/pyasc, runtime, ge, cann-recipes-*
- [Atlas 950 SuperPoD at WAIC 2026 — Huawei (2026-07-17)](https://www.huawei.com/cn/news/2026/7/atlas-950-superpod) — 1,024-card demo, 1 EFLOPS FP8 / 2 EFLOPS FP4, 256 TB unified address space, 3 μs RTT
- [SuperPoD announcements at MWC 2026 — Huawei](https://www.huawei.com/en/news/2026/3/mwc-superpod-ai) — 64 NPUs/cabinet → 8,192; Atlas 850E air-cooled variant
- [UnifiedBus open specification](https://www.unifiedbus.com/en) — base spec, firmware spec, SW reference designs
- [openEuler UB Service Core SW Architecture Reference Design (PDF)](https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf)
- [Ascend 950DT deployment pulled forward to August — TrendForce (2026-06-08)](https://www.trendforce.com/news/2026/06/08/news-huawei-brings-forward-ascend-950dt-deployment-to-august-deepseek-v4-2-seen-as-potential-early-adopter/) — announced schedule; Q4 2026 remains official
- [Huawei confirms Ascend 950DT to debut in August — Huawei Central](https://www.huaweicentral.com/huawei-confirms-ascend-950dt-ai-chip-to-debut-in-august/)
- [Atlas 350 debut on Ascend 950PR — TrendForce (2026-03-23)](https://www.trendforce.com/news/2026/03/23/news-huawei-debuts-atlas-350-on-ascend-950pr-with-in-house-hbm-touting-2-8x-h20-performance/) — 1.56 PFLOPS FP4, 112 GB, 1.4 TB/s, 600 W shipping-card spec
- [Atlas 350 unveiled — Tom's Hardware (2026-03-24)](https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20)
- [Ascend 950 NPU whitepaper analysis (third-party, single-source)](https://pillumina.github.io/posts/aiinfra/ascend-950-npu/) — source of the unconfirmed microarchitecture and RTP/CTP figures flagged in §2

### Update (added 2026-09-13)
- [Bloomberg — "DeepSeek Plans Big Huawei AI Chip Order to Power New Data Center" (2026-09-04)](https://www.bloomberg.com/news/articles/2026-09-04/deepseek-plans-big-huawei-ai-chip-order-to-power-new-data-center) — ≥160,000 Ascend 950DT, Inner Mongolia, inference-only, >1 year to deliver in full
- [The Decoder (2026-09-04)](https://the-decoder.com/deepseek-plans-the-largest-known-huawei-chip-cluster-with-160000-processors-in-inner-mongolia/)
- [TechNode (2026-09-07)](https://technode.com/2026/09/07/)
- [Huawei Central — "Huawei Ascend 950DT price jumped 60% over past three months" (2026-09-10)](https://www.huaweicentral.com/huawei-ascend-950dt-price-jumped-60-over-past-three-months/) — press-reported ¥150,653 → ¥250,000
- [Huawei Connect 2026 event page](https://www.huawei.com/en/events/huaweiconnect) — 2026-09-17 to 09-19, Shanghai; not yet held as of this scan
- [MindSpore release history — PyPI](https://pypi.org/project/mindspore/#history) — 2.10.0 published 2026-07-31 (pre-window, missed at 2026-08-08 baseline)
