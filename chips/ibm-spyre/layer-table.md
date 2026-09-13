# IBM Spyre Accelerator — Layer Mapping Table

*as_of: 2026-08-08*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch 2.x (2.9.1 → ~2.11): `spyre` registered as a first-class device via **PrivateUse1** (`rename_privateuse1_backend` + `_register_device_module`); eager and `torch.compile(backend="spyre")` paths | confirmed | github-torch-spyre, ibm-pytorch-blog |
| Framework Integration | sendnn-inference: production **vLLM plugin**, ~638 commits, Apache-2.0; formerly `vllm-project/vllm-spyre` (now redirects). Supports chunked prefill, prefix caching, guided decoding, logprobs, beam search, tensor parallel, embeddings | confirmed | github-sendnn-inference, vllm-spyre-docs |
| Framework Integration | vLLM **unsupported**: LoRA, speculative decoding, encoder-decoder, prompt logprobs, pipeline/expert/data parallel, PD disaggregation, sleep mode. Partial: multimodality, quantization | confirmed | vllm-spyre-features |
| Framework Integration | spyre-inference: second-generation vLLM plugin built on `torch-spyre` rather than `torch_sendnn`, Apache-2.0 | confirmed | github-spyre-inference |
| Framework Integration | hf-adapters: HuggingFace enablement by **load-time monkey-patching** of only the unsupported ops (RoPE, RMSNorm, KV-cache mgmt, generation loop); 30 adapters, 46 verified checkpoints, 100+ models | confirmed | github-hf-adapters |
| Framework Integration | sglang-spyre: early SGLang port (🟡 no license file) | confirmed | github-sglang-spyre |
| Framework Integration | aiu-fms-testing-utils: legacy FMS path driving `torch_sendnn` directly; documents `FLEX_COMPUTE`, `FLEX_DEVICE`, `DTLOG_LEVEL`, `TORCH_SENDNN_LOG`, `SENCORES` env vars | confirmed | github-aiu-fms |
| Framework Integration | **Negative finding**: zDLC and zDNN target Telum/Telum II zAIU via NNPA **only**; no public ONNX or z/OS-native (MLz/WMLz) path to Spyre exists | confirmed | github-zdnn, github-zdlc |
| Graph Capture | TorchDynamo + AOTAutograd → FX graph of ATen ops; decomposition to core ATen; AOT compilation with **static shapes** is the supported mode | confirmed | torch-spyre-compiler-docs |
| Graph Capture | `dynamic=True` handled by **static binary specialization** (compile once per shape, reuse across equivalent geometries); artifact caching cuts startup | confirmed | torch-spyre-compiler-docs |
| Graph Capture | Eager mode = AOT-compiling each op individually (`register_kernel("aten::mm", ["spyre"])` wrapping `torch.compile`) — correct but slow. A single graph break in the hot path erases the compiled region's gains | confirmed | torch-spyre-compiler-docs, rfc-0171 |
| Graph Compiler | **TorchInductor with an out-of-tree Spyre backend** (Apache-2.0): `SuperDSCScheduling` (scheduler.py), `SpyrePythonWrapperCodegen` (wrapper.py), `SpyreDeviceOpOverrides` (device/op_overrides.py); six upstream extension points; `enable_spyre_context` entry point | confirmed | torch-spyre-inductor-docs, github-torch-spyre |
| Graph Compiler | Pipeline: PyTorch → FX (ATen) → **LoopLevelIR** → OpSpec → **SuperDSC JSON** → [DeepTools] → SpyreCode binary | confirmed | torch-spyre-compiler-docs |
| Graph Compiler | **16-pass LoopLevelIR pipeline**: DCE · split multi-ops · propagate layouts (FixedTiledLayout) · validate ops · optimize restickify locations · finalize layouts · insert restickify · post-mutation restickify · BMM padding · dedup constants · propagate named dims · assign dim hints · coarse tile · **span reduction** · **matmul division + work distribution across 32 cores** · **LX planning** | confirmed | torch-spyre-inductor-docs |
| Graph Compiler | Work division: cost model `compute + hbm + psum + shape_penalties + batch` over (b,m,n,k); distribute across output dims then ≤1 reduction dim; equal stick counts per core; splits recorded as index coefficients in `op_it_space_splits`; core budget via `SENCORES` (default 32) | confirmed | torch-spyre-workdiv-docs |
| Graph Compiler | Scratchpad (LX) planning: four solvers — greedy (default), first-fit, best-fit, **CP-SAT** (2-D non-overlapping rectangles over lifetime × address, minimizing DDR traffic); 128 B alignment; ~1.6 MB budget; **no eviction**; core-division mismatch disqualifies a buffer from LX | confirmed | torch-spyre-scratchpad-docs |
| Graph Compiler | Additional passes: indirect access (gather), working-set reduction, coarse-tiling loop IR, span-overflow hint analysis | confirmed | torch-spyre-compiler-docs |
| Interface IR | **SuperDSC / SDSC** ("Super Design Space Config") — current production interface, **published as an open spec** although its consumer is closed. MLIR (`sdscbundle.sdsc_execute`, `scf.for`, `affine.apply`) + per-op `sdsc.json`: OpFunc, memory location (DDR/"HBM" vs LX), core work mapping, stick config, back-gaps, scale factors, affine **folds** describing all 32 cores | confirmed | superdsc-spec |
| Interface IR | SuperDSC stick constraints are op-class-specific and ripple: BatchMatMul output stick `generated_dim=64`; DF16 input1 stick `reduction_dim=64`; FP8/INT8 input2 stick `reduction_dim=2, generated_dim=64` | confirmed | superdsc-spec |
| Interface IR | **KTIR (KernelTile IR)** — MLIR dialect `ktdp`, RFC 0682 merged March 2026; positioned by IBM as a **community-aligned spec generalizable across dataflow accelerators**. Three-step memory access: `construct_memory_view` → `construct_access_tile` → `ktdp.load`/`store`. `SpyreMemorySpaceAttr` marks HBM vs `LX, core=N`. Spec stable, backend lowering in development; production still routes through SuperDSC | confirmed | github-ktir-frontend, rfc-0682 |
| Interface IR | KTIR stated distinctions from the GPU model: persistent compile-time-partitioned cores (not thread blocks); explicit scratchpad management (no implicit cache hierarchy). Reuses upstream `arith`/`math`/`linalg`/`scf` | confirmed | github-ktir-frontend |
| Simulator | **ktir-cpu** (Apache-2.0): CPU **interpreter, validator and latency model** for KTIR; simulates a multi-core grid with HBM 128 GB shared + **LX 2 MB per core**; NumPy execution; framed as "an environment and reward model for AI-driven compiler development". Enables hardware-free study of the programming model | confirmed | github-ktir-cpu |
| Scheduler Infra | **dataflow-scheduler** + dataflow-scheduler-mlir-dialects (Apache-2.0): KTIR + architecture description (MLIR) → KTDF/KTDFLow → **DFIR**. Partition at memory boundaries → fuse (`linalg.generic`) → materialize load→compute→store → refine (hierarchical routing, loop tiling, pipelines, LICM, **double buffering**) → parallelize → normalize to a 1-D grid → split per engine. Execution model: "pipeline + parallel". LLVM `llvmorg-22.1.3` | confirmed | github-dataflow-scheduler |
| Kernel Language | **Triton** via IBM fork `torch-spyre/triton` (MIT, tracks upstream via `sync-*` branches) — the Spyre lowering is not upstream | confirmed | github-triton-fork |
| Kernel Language | **spyre-kernels**: Triton → TTIR → KTIR → machine code via `scripts/gen_ktir.py`; commits `.ttir`/`.ktir` alongside each kernel, per model and per vLLM op (prefill_attention, rms_norm, silu_and_mul, mrope, reshape_and_cache, …). 🟡 no license file | confirmed | github-spyre-kernels |
| Kernel Language | Four validation tiers: **T0** numerics vs PyTorch-on-GPU · **T1** shape compliance (tiles fit scratchpad, grid fits 32 cores) · **T2** correctness on `ktir_cpu` **and** real hardware · **T3** expert review. Invariants stored as documents (`grid-fits-32-cores.md`, `tile-fits-scratchpad.md`, `runtime-arg-agnostic.md`) | confirmed | github-spyre-kernels |
| Back-end Compiler | **DeepTools — PROPRIETARY.** "A proprietary component called DeepTools, developed by IBM." Runs out-of-process as `dxp_standalone -d <output_dir>`; outputs cached under the Inductor cache. Input SuperDSC JSON (future KTIR); does dataflow mapping, **core scheduling across all 32 cores**, and binary generation incl. LPDDR5→LX load/store sequences | confirmed | torch-spyre-backend-docs |
| Back-end Compiler | DeepTools internals — pass list, mapping algorithms, scheduling heuristics, source — **not disclosed** | confirmed (as non-disclosure) | torch-spyre-backend-docs |
| Device Binary | **SpyreCode** — container produced by DeepTools; the **container format is an open spec** although the producer is closed. Four parts: Job Execution Plan (`ComputeOnHost`, `ComputeOnDevice`, `DataTransfer`) · Job Preparation Plan (`Allocate` into SegmentId=7, `InitTransfer`) · job binaries (`init.bin`) · host compute metadata driving **program correction** (JIT patching of loop counts and addresses before launch) | confirmed | spyrecode-spec |
| Tensor API / Layout | `SpyreTensorImpl` (at::TensorImpl subclass with layout metadata + DCI translation data); `SpyreTensorLayout` (`device_size`, `stride_map`, `device_dtype`, `element_arrangement`; `offset = dot(device_coords, stride_map)`); `FixedTiledLayout` (Inductor Layout subclass) | confirmed | torch-spyre-tensors-docs, rfc-0047 |
| Tensor API / Layout | **Stickification** — "the transformation from a host-strided PyTorch layout to a tiled Spyre device layout". A (1024, 256) fp16 tensor becomes (4, 1024, 64) on device — **not expressible with PyTorch strides**. `restickify` reconciles adjacent ops that disagree on tile structure | confirmed | torch-spyre-tensors-docs |
| Tensor API / Layout | **DCI (Data Conversion Information)** — loop ranges, strides, dtype fed to the DMA engine for host-side transfers. Python surface: `.to("spyre")`, `new_empty`, `new_empty_strided`, `device_tensor_layout()`. Default dtype fp16 | confirmed | torch-spyre-tensors-docs, rfc-1069 |
| Runtime (open) | `torch_spyre` C++/Python (Apache-2.0): `SpyreAllocator` (lazy chunked, CUDA-caching-allocator-style, virtual sub-allocation for VF handle limits) · `SpyreGuardImpl` · **`SpyreStream`** (FIFO within a stream, no cross-stream ordering, **sticky error model** — failed stream must be recreated) · cached `JobPlan` · `RuntimeOperation{H2D,D2H,Compute,HostCallback}` · `CompositeAddress` = region_id + 128 B offset. Entry points `PrepareKernel` / `LaunchKernel`; `SPYRE_ALLOW_TILED_LAUNCH` for multi-iteration tiled dispatch | confirmed | program-execution-spec, stream-sync-spec |
| Runtime (closed) | **DeepRT — PROPRIETARY**: IBM device runtime that prepares ops into SpyreCode directories consumed by the JobPlan translator | confirmed | torch-spyre-runtime-docs |
| Runtime (closed) | **Flex — PROPRIETARY**: the actual execution engine — `RuntimeStream.launchOperation()`, `FlexAllocator`, kernel launch, device communication, driver dispatch. PF/VF via `FLEX_DEVICE`; `FLEX_COMPUTE=SENTIENT` selects hardware | confirmed | torch-spyre-runtime-docs |
| Runtime (closed) | **`torch_sendnn` — NOT PUBLICLY DISTRIBUTED**: "only available pre-installed in a base environment", reached with `--system-site-packages`. A CPU-only dev path (`uv pip install sendnn-inference` + CPU PyTorch) exists for front-end work without hardware | confirmed | vllm-spyre-install, torch-spyre-install |
| Communication | **`spyre_comms` / `spyreccl` — PROPRIETARY library with an open backend shim**: PyTorch `torch.distributed` backend (`init_process_group` → `broadcast`, `all_reduce`); blocking; allreduce prioritized for tensor parallel; **on-node (single server) only**, multi-node "may be added in the future"; **up to 8 cards**. Functional collectives currently compile through `torch.inductor`; migration to `torch.distributed`/`torch.comms` planned | confirmed | rfc-0099, torch-spyre-runtime-docs |
| Driver / Firmware | **Kernel driver — PROPRIETARY.** Exposes PF and VF (SR-IOV) functions; in VF mode memory is addressed via firmware `region_id` lookup rather than physical addresses. ioctl/ABI surface not disclosed | confirmed | rfc-0171, program-execution-spec |
| Driver / Firmware | **Card firmware — PROPRIETARY.** `ComputeOnDevice` is realized as a firmware control message generating compute control blocks; DMA expressed as DMAI/DMAO control blocks. Control-message format not disclosed | confirmed | spyrecode-spec |
| ISA | **NOT PUBLIC.** No instruction set, encoding or assembler is published; the `init.bin` job binaries are the only artifact and their format is undocumented. The nearest public contract is the SpyreCode container spec, which describes **commands, not instructions**. RFCs cite **export control** as the reason telemetry reads a vendor API rather than raw counters | confirmed | spyrecode-spec, rfc-2696 |
| Observability | PyTorch Profiler extension via `REGISTER_PRIVATEUSE1_PROFILER` (`record`, `elapsed`, `synchronize`, `onEachDevice`); `kineto-spyre` wheel. "Event completion" redefined for dataflow as "when all output tokens of a kernel have been written to their destination" | confirmed | rfc-0601, torch-spyre-profiling-docs |
| Observability | **`aiu-smi`** over the **`spyremetrics`** API — the `nvidia-smi` analogue (❌ closed implementation, open RFC). Power, temperature, device-busy %, memory R/W bandwidth, PCIe ingress/egress, RDMA ops, request rates, reserved/actual/peak device memory, **PT-array utilization estimated from power**, per-segment reserved memory. PF and VF, on x86/Power/Z | confirmed | rfc-2696, rfc-2676 |
| Observability | **`libaiupti`** (❌ closed) — AIU Profiling Tools Interface (kernel + memory tracking); `aiu-trace-analyzer` (✅ open) for derived metrics; Inductor provenance tracking (✅ open) at the compiler front end | confirmed | rfc-0601 |
| Observability | Dataflow-specific metrics with no von Neumann analogue: pipeline utilization, DMA overhead, reconfiguration latency, inter-core communication efficiency, **stick alignment overhead**, and for the compiler-managed LX — peak/average scratchpad utilization, **fragmentation ratio**, allocation efficiency. RFC 0601 contrasts the **dual memory hierarchy**: DRAM runtime-managed/runtime-observable vs. LX compiler-managed/compile-time-and-runtime-observable | confirmed | rfc-0601 |
| Orchestration | `ibm-aiu` org, all Go + Apache-2.0, actively pushed as of Aug 2026: **spyre-operator** (OpenShift), **spyre-device-plugin** (K8s), **dra-driver-spyre** (Dynamic Resource Allocation), spyre-health-checker, spyre-scheduler-plugins, spyre-webhook-validator, certified-operators, spyre-operator-docs | confirmed | github-ibm-aiu |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | **32 active AI cores** (34 physical — 2 spares for yield), 8×4 grid; 2 **corelets** per core sharing one 2 MB LX scratchpad | confirmed | ibm-lifting-the-cover, torch-spyre-device-docs |
| Compute Engine | Per corelet: 2D **8×8 SIMD-systolic PE array** = 64 "low-precision math engines" (the **PT** / Processing Tensor unit) for matmul/bmm/conv + fused epilogues; plus 1D vector array(s) / **SFU** for GELU, softmax, element-wise | confirmed | ibm-lifting-the-cover, torch-spyre-glossary |
| Compute Engine | Total 4,096 math engines per card (32 × 2 × 64) | inferred (arithmetic) | derived |
| Compute Engine | Datatypes: 2D array fp16/fp8/int8/int4; 1D vector adds **fp32** for activations and normalization. Compiler-visible: DL16 (IBM DLFloat16), FP32, FP8, INT8, INT4 | confirmed | ibm-lifting-the-cover, superdsc-spec |
| Compute Engine | Peak throughput, **IBM primary**: ">300 TOPS per card, while consuming just 75 W" | confirmed | torch-spyre-device-docs |
| Compute Engine | Per-datatype peaks — fp16 98 / fp8 157 / int8 315 (4.2 TOPS/W) / int4 629 TOPS | unconfirmed | morethanmoore (ISSCC-attributed; **no IBM primary source**) |
| Compute Engine | **Clock frequency: not disclosed** | not disclosed | — |
| Compute Engine | Op library: 70+ SuperDSC `OpFunc` primitives (BatchMatMul, Conv2D, broadcasts, BatchNorm/LayerNorm, GELU/ReLU/Sigmoid/Exp/Log, reductions, pooling, depthwise conv, quantization/CSQ, TopK) | confirmed | superdsc-spec |
| Data Path | **Statically scheduled dataflow**: "An operation is eligible to execute as soon as all of its input operands are available". All work division, data staging and kernel specification fixed at compile time → IBM claims **deterministic execution latency** | confirmed | torch-spyre-dataflow-docs |
| Data Path | **SPMD across cores** — "cores follow a common program structure but operate on different tiles", selected by core ID | confirmed | torch-spyre-dataflow-docs |
| Data Path | Explicit path: Host → LPDDR5 → LX → PT/SFU → LX → LPDDR5 → Host, with compiler-emitted loads/stores. Multi-input ops (concat, residual add) are explicit synchronization points | confirmed | torch-spyre-dataflow-docs |
| Data Path | **No hardware caches, no out-of-order execution, no warp scheduler, no runtime dispatcher** — the entire performance model rests on the compiler | confirmed | torch-spyre-dataflow-docs |
| Data Path | Constraints: static shapes required (dynamic WIP); default device dtype fp16; int64 silently downcasts to int32; inner dims must be 128 B aligned | confirmed | torch-spyre-key-concepts |
| Data Path | **Program correction**: for kernels with symbolic addresses/dynamic dims, a host callback JIT-patches loop counts and address references into the device binary immediately before launch | confirmed | spyrecode-spec |
| Data Path | **Virtualization**: PF and VF (SR-IOV) modes via `FLEX_DEVICE`; VF mode uses `region_id` + 128 B-aligned offset through firmware lookup instead of physical addresses | confirmed | program-execution-spec, rfc-0171 |
| On-chip Memory | **LX scratchpad: 2 MB per core** (~1.6 MB usable after backend reservations), shared by both corelets → **64 MB total** (arithmetic). **Compiler-managed; no hardware cache; no eviction policy** — "There is no mechanism to move a buffer to HBM and reload it later. This is deliberate." | confirmed (64 MB total inferred) | torch-spyre-dataflow-docs, torch-spyre-scratchpad-docs |
| On-chip Memory | IBM describes a **"2-level programmable SRAM scratchpad microarchitecture"** per core; only the 2 MB LX level is quantified — the inner level feeding the PE array is **not sized** | confirmed (two levels) / not disclosed (sizes) | ibm-lifting-the-cover |
| On-chip Memory | Register files / PE-local storage capacity, and LX↔PT/SFU bandwidth — **not disclosed** | not disclosed | — |
| Off-chip Memory | **128 GB LPDDR5 per card**; **16 channels @ 6.4 Gbps**; **~204 GB/s** peak to the cores; 8 dual-channel LPDDR5 modules on the card (not packaged on the SoC); SECDED ECC (ECC: secondary) | confirmed | ibm-lifting-the-cover, ibm-docs-9175, torch-spyre-device-docs |
| Off-chip Memory | Transfer granularity: **"stick" = 128-byte aligned chunk = 64 fp16 elements** (`BYTES_IN_STICK = 128`), sized to "match the natural bandwidth between LPDDR5 device memory and the per-core LX scratchpad". Direct inheritance of the zAIU/zDNN "stickified tensor" layout on Telum | confirmed | torch-spyre-device-docs, torch-spyre-glossary |
| Off-chip Memory | **256 MB** maximum contiguous device-memory span addressable by any one core (compiler "span reduction" pass); job address space 128 GB in **8 segments of ≤16 GB**, SegmentId=7 reserved for SpyreCode; 128 B alignment | confirmed | torch-spyre-workdiv-docs, spyrecode-spec |
| Off-chip Memory | SuperDSC JSON names device memory `HBM` as **legacy nomenclature** — Spyre uses LPDDR5, not HBM | confirmed | torch-spyre-glossary |
| Off-chip Memory | Whether card configurations other than 128 GB exist — **not disclosed** | not disclosed | — |
| On-chip Interconnect | **Bidirectional ring** across all 32 active cores, **128 B per cycle per direction**. Compiler exploits it: K-split matmul reductions permute core IDs so collaborating cores sit adjacent on the ring, "reducing hop counts from m×n to 1" | confirmed | torch-spyre-device-docs, torch-spyre-workdiv-docs |
| On-chip Interconnect | Aggregate NoC bandwidth in GB/s — **not disclosed** (cannot be derived without the clock) | not disclosed | — |
| Host Interface / Package | Single-slot PCIe card, **75 W, no auxiliary power connector**; IBM primary states only "PCIe card" with "fully pipelined DMA/RDMA support" | confirmed | ibm-lifting-the-cover, ibm-newsroom |
| Host Interface / Package | PCIe **Gen5 ×16, 64 GB/s** | unconfirmed | morethanmoore (**no IBM primary confirmation of generation or lane count**) |
| Scale-up Interconnect | **No proprietary chip-to-chip fabric.** Cards scale over a **standard PCIe switch fabric** with **direct card-to-card RDMA** bypassing the host CPU. ISSCC: "scales over a standard PCIE fabric" | confirmed | ibm-lifting-the-cover, isscc-2026 |
| Scale-up Interconnect | Card-to-card bandwidth 64 GB/s, CRC-protected | unconfirmed | morethanmoore |
| Scale-up Interconnect | **Chassis limits: IBM z17 / LinuxONE 5 — 48 cards** (≈6.1 TB, 1,536 cores); **IBM Power11 — 16 cards** (≈2 TB, 512 cores) | confirmed | ibm-lifting-the-cover, ibm-newsroom |
| Scale-up Interconnect | **Single-model ensemble limit: 8 cards / ~1 TB** — IBM Docs and torch-spyre docs both state "ensembles of up to eight cards delivering 1 TB memory"; `spyreccl` caps at 8. ISSCC says "scales up to 4 or more devices for large generative models". The gap between the 48-card chassis limit and the 8-card ensemble limit is **undocumented** | confirmed | ibm-docs-9175, torch-spyre-runtime-docs, isscc-2026 |
| Scale-up Interconnect | ⚠️ **Stale figure**: "8 cards / 256 cores" is IBM's **August 2024 Hot Chips preview** (one I/O drawer), not the shipping product. ⚠️ **Conflicting secondary**: "48 per tray, 192 per system" — its own 6.1 TB total corresponds to 48 cards; **use 48** | confirmed | theregister-2024, ibm-lifting-the-cover |
| Scale-out Interconnect | **None.** Spyre Comms is explicitly **on-node (single server) only**; multi-node listed as a possible future addition with no roadmap date. No Ethernet/RoCE on the accelerator, no collective offload engine | confirmed | rfc-0099 |
| Physical / Electrical | **5 nm CMOS**, **25.6 billion transistors**, "14 miles" of interconnect wiring | confirmed | ibm-lifting-the-cover, ibm-spyre-for-z |
| Physical / Electrical | Die size **330 mm²** | unconfirmed | morethanmoore (ISSCC-attributed; **no IBM primary confirmation**) |
| Physical / Electrical | **Dual voltage domains**: 0.55 V for the high-activity AI core array; 0.75 V for timing-critical logic, SRAM and third-party IP | confirmed | ibm-lifting-the-cover |
| Physical / Electrical | **Dual-loop power controller**: fast inner loop absorbs peak-current spikes; slower software-controlled outer loop adjusts the average-current target | confirmed | ibm-lifting-the-cover |
| Physical / Electrical | 7T standard-cell library chosen over 6T (6T needed extra buffers at 0.55 V): 9% synthesis-frequency reduction → 7.5% power saving; re-synthesis → +8% power, +6% area. Dual-loop claim: 25% higher inference throughput vs single-loop | unconfirmed | morethanmoore (ISSCC-attributed) |
| Physical / Electrical | Idle power, thermal solution, sustained-vs-peak power behaviour — **not disclosed** | not disclosed | — |
| Maturity | **Shipping (GA), sold as a priced system option.** Announced 2025-10-07; GA 2025-10-28 on z17/LinuxONE 5; GA "early December 2025" on Power11. **No named external at-scale deployment is public** — pre-GA validation only at IBM Yorktown Heights and the University at Albany Center for Emerging AI Systems. Record as *shipping (GA)*, not *deployed at scale* | confirmed | ibm-newsroom, research-ibm-building |
| Vendor Claims | "2-to-3× better power/performance than GPUs on encoder-class models" (ISSCC abstract, no baseline disclosed); ">8M documents/hour" at prompt size 128 (internal testing, 1M-unit dataset, batch 128, single card); "near-linear scaling up to 8 cards". **No independent benchmark (MLPerf or otherwise) exists** | confirmed as claims | isscc-2026, ibm-newsroom |

## Source Keys

| Key | URL |
|-----|-----|
| ibm-lifting-the-cover | https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator |
| research-ibm-building | https://research.ibm.com/blog/building-the-ibm-spyre-accelerator |
| ibm-pytorch-blog | https://research.ibm.com/blog/pytorch-support-ibm-spyre |
| ibm-spyre-for-z | https://research.ibm.com/blog/spyre-for-z |
| isscc-2026 | https://research.ibm.com/publications/spyre-an-inference-optimized-scalable-ai-accelerator-for-enterprise-workloads |
| ibm-newsroom | https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability |
| ibm-docs-9175 | https://www.ibm.com/docs/en/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator |
| torch-spyre-device-docs | https://torch-spyre.readthedocs.io/en/latest/architecture/spyre_accelerator.html |
| torch-spyre-dataflow-docs | https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html |
| torch-spyre-key-concepts | https://torch-spyre.readthedocs.io/en/latest/getting_started/key_concepts.html |
| torch-spyre-glossary | https://torch-spyre.readthedocs.io/en/latest/getting_started/glossary.html |
| torch-spyre-install | https://torch-spyre.readthedocs.io/en/latest/getting_started/installation.html |
| torch-spyre-compiler-docs | https://torch-spyre.readthedocs.io/en/latest/compiler/architecture.html |
| torch-spyre-inductor-docs | https://torch-spyre.readthedocs.io/en/latest/compiler/inductor_frontend.html |
| torch-spyre-backend-docs | https://torch-spyre.readthedocs.io/en/latest/compiler/backend.html |
| torch-spyre-scratchpad-docs | https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html |
| torch-spyre-workdiv-docs | https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html |
| torch-spyre-tensors-docs | https://torch-spyre.readthedocs.io/en/latest/user_guide/tensors_and_layouts.html |
| torch-spyre-runtime-docs | https://torch-spyre.readthedocs.io/en/latest/runtime/index.html |
| torch-spyre-profiling-docs | https://torch-spyre.readthedocs.io/en/latest/user_guide/profiling/index.html |
| github-torch-spyre | https://github.com/torch-spyre/torch-spyre |
| github-sendnn-inference | https://github.com/torch-spyre/sendnn-inference |
| github-spyre-inference | https://github.com/torch-spyre/spyre-inference |
| github-hf-adapters | https://github.com/torch-spyre/hf-adapters |
| github-sglang-spyre | https://github.com/torch-spyre/sglang-spyre |
| github-triton-fork | https://github.com/torch-spyre/triton |
| github-spyre-kernels | https://github.com/torch-spyre/spyre-kernels |
| github-ktir-frontend | https://github.com/torch-spyre/ktir-mlir-frontend |
| github-ktir-cpu | https://github.com/torch-spyre/ktir-cpu |
| github-dataflow-scheduler | https://github.com/torch-spyre/dataflow-scheduler |
| github-aiu-fms | https://github.com/foundation-model-stack/aiu-fms-testing-utils |
| github-ibm-aiu | https://github.com/ibm-aiu |
| github-zdnn | https://github.com/IBM/zDNN |
| github-zdlc | https://github.com/IBM/zDLC |
| superdsc-spec | https://github.com/torch-spyre/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md |
| spyrecode-spec | https://github.com/torch-spyre/interface-specs/blob/main/0277-SpyreCode/0277-SpyreCodeSpec.md |
| program-execution-spec | https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/ProgramExecutionSpec.md |
| stream-sync-spec | https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/StreamSynchronizationSpec.md |
| rfc-0047 | https://github.com/torch-spyre/RFCs/blob/main/0047-TiledTensors/0047-TiledTensorsRFC.md |
| rfc-0099 | https://github.com/torch-spyre/RFCs/blob/main/0099-MultiDevice/0099-MultiDeviceRFC.md |
| rfc-0171 | https://github.com/torch-spyre/RFCs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md |
| rfc-0601 | https://github.com/torch-spyre/RFCs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md |
| rfc-0682 | https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md |
| rfc-1069 | https://github.com/torch-spyre/RFCs/blob/main/1069-SpyreTensorLayoutExtraction/1069-SpyreTensorLayoutExtraction.md |
| rfc-2676 | https://github.com/torch-spyre/RFCs/blob/main/2676-SpyreMetricsApiExtension/2676-SpyreMetricsApiExtensionRFC.md |
| rfc-2696 | https://github.com/torch-spyre/RFCs/blob/main/2696-AiuSmiExtension/2696-AiuSmiExtensionRFC.md |
| vllm-spyre-docs | https://docs.vllm.ai/projects/spyre/en/latest/ |
| vllm-spyre-install | https://docs.vllm.ai/projects/spyre/en/latest/getting_started/installation.html |
| vllm-spyre-features | https://docs.vllm.ai/projects/spyre/en/latest/user_guide/supported_features.html |
| morethanmoore | https://morethanmoore.substack.com/p/ibms-spyre-ai-accelerator-deep-dive |
| theregister-2024 | https://www.theregister.com/on-prem/2024/08/27/ibm-details-upcoming-chips-to-support-ai-on-mainframes/ |
