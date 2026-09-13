# Tsingmicro (清微智能) TX81 Layer Mapping Table

*as_of: 2026-08-08*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | **torch_txda** (closed wheel): PyTorch device plugin registering device `txda` via PrivateUse1; `torch.txda.{is_available, current_device, set_device, current_stream, get_device_properties, get_device_capability}` | confirmed | flagtree-publish-readme |
| Framework Integration | **FlagGems `_tsingmicro`** (open): `VendorDescriptor(vendor_name="tsingmicro", device_name="txda", device_query_cmd="tsm_smi", dispatch_key="PrivateUse1", fp64_enabled=False, int64_enabled=False)`; `tune_configs.yaml` 1313 lines / ~37 op families | confirmed | flaggems-tsingmicro-init, flaggems-tune-configs |
| Framework Integration | **txops** (closed wheel): vendor PyTorch operator package | confirmed | flagtree-publish-readme |
| Framework Integration | **vllm-plugin-FL / sglang-plugin-FL** forks ("support txda"): open but thin; no txda-specific directories visible | likely | tsingmicro-public-e |
| Framework Integration | **FlagScale / Megatron-LM-FL / TransformerEngine-FL / PyTorch-Plugin-FL** forks: training-side ecosystem, maintained to Aug 2026; upstream PyTorch-Plugin-FL has **no** txda backend, so the fork is aspirational | likely | tsingmicro-public-e |
| Framework Integration | Vendor claim of "CUDA 生态兼容 / 零代码迁移" — **marketing.** Real surface is API-*shape* compatibility: HIP-shaped runtime + NCCL-shaped TCCL + Triton | confirmed | tsingmicro-tx8-page, flagcx-device-adaptor |
| Graph Capture | **None public.** No torch.compile / TorchDynamo / FX integration visible; execution is per-kernel Triton JIT plus FlagGems op dispatch | confirmed | flagtree-tsingmicro-backend |
| Graph Compiler | **RAISA graph-optimisation layer** (closed, undocumented): vendor layer-3 description "根据上层需求对计算图进行全局优化". Component name, IR, and whether it is a distinct artifact are **not disclosed** | unconfirmed | tsingmicro-tx8-page |
| Graph Compiler | Maturity signal: `TXDA_FALLBACK_CPU_OPS` lists ~60 ops falling back to CPU (`sort, gather, index, pad, embedding_backward, cat, add, mul, sum`); `TXDA_SKIP_OPS` also present | confirmed | flagtree-run-flaggems-multicards |
| Kernel Compiler | **FlagTree `third_party/tsingmicro`** (open, 609 files, branch `triton_v3.3.x`, integrated 2025-06-06 with CI): class `TXDABackend`, `GPUTarget(backend="txda")`, `binary_ext="so"`; co-authored with **Terapines Technology (Wuhan) / 兆松科技** | confirmed | flagtree-readme, flagtree-compiler-py |
| Kernel Compiler | Pipeline `ttir → coreir → txir → llir → so`; `--triton-to-core-dialects`, `--tle-to-mk`, `--dsa-memory-to-core`, `--linalg-tiling`, `--core-dialects-to-mk=precision-mode={0,1,2}`, `--mk-pipeline`, `--spmd-allocate-shared-memory`, `--mk-to-tx81`, `--tx81-insert-barrier`, `--tx81-resolve-dma-base-addr`, `--tx81-to-llvm` | confirmed | flagtree-compiler-py |
| Kernel Compiler | MLIR dialects: **Address**, **triton-shared** (vendored), **MagicKernel (MK)** + Func + Instr, **Tx81**, **DSA**. `MagicKernelOps.td` 30 KB (~90 ops); `Tx81Ops.td` **47 KB (~150 ops)** — an operation-level model of the accelerator | confirmed | tx81ops-td, magickernelops-td |
| Kernel Compiler | Tools (open, built locally): `tsingmicro-opt`, `tsingmicro-lsp`, `tsingmicro-reduce`, `tsingmicro-llvm-opt`, `tsingmicro-tensor-layout`, `tx-profiler` | confirmed | flagtree-tsingmicro-backend |
| Kernel Compiler | **tsingmicro-llvm21** (closed binary): vendor LLVM 21 fork, bundle `…-x64_v0.6.0`; source on private `gitlab.tsingmicro.com`; internal codename **"ZTC"** in header guards | confirmed | flagtree-wiki-manual |
| Kernel Compiler | Backend codegen: `clang++ --target=riscv64-unknown-elf -march=rv64imfdc -O2 -c` → `riscv64-unknown-elf-gcc -shared -mabi=lp64d -lcommon_util -linstr_rcs1 -llibc_stub -lvr` → `kernel.so`; toolchain **T-Head XuanTie-900 GCC ELF newlib V2.8.0** | confirmed | flagtree-compiler-py |
| Kernel Language | **Triton 3.3** (open) — the public kernel language for TX81; pinned to 3.3 while other FlagTree vendors moved to 3.6 | confirmed | flagtree-readme |
| Kernel Language | **`triton.language.extra.txda.libdevice`** (51 KB, open): maps Triton math builtins to Tx81 intrinsics | confirmed | flagtree-tsingmicro-backend |
| Kernel Language | **TLE — Triton Language Extensions** (`triton.experimental.tle`, open): three tiers TLE-Lite / TLE-Struct / TLE-Raw; Tsingmicro uses TLE-DSA and TLE-distributed | confirmed | flagtree-wiki-tle |
| Kernel Language | **TLE-DSA** (scratchpad): `tle.language.dsa.alloc(shape, dtype, scope=spm)`, `.copy(GM↔SPM, SPM↔SPM)`, `.local_ptr()`; upstream docstring names "the on-chip SRAM exposed by TsingMicro TX8" and calls it the analogue of NVIDIA shared memory | confirmed | tle-dsa-types |
| Kernel Language | **TLE-distributed** (mesh): `tle.device_mesh`, `tle.sharding`, `tle.ShardedTensor`, `tle.reshard`, `tle.shard_id`, `tle.remote`, `tle.distributed_barrier`, `tle.distributed_dot` — this is how the tile/chip mesh is programmed from Triton | confirmed | tle-distributed, tle-noc-gemm-example |
| Kernel Language | Vendor-claimed "自研 C/C++ 接口语言" — **not public**; only the Triton path exists publicly | unconfirmed | tsingmicro-tx8-page |
| Compiler Runtime (CRT) | `third_party/tsingmicro/crt/` (open, ~130 C files): `__Gemm`, `__Rdma`/`__Wdma`/`__Rdma4d`, `__Send`, `op_gelu.c` (31 KB), `pow.c` (21 KB), all dtype conversions incl. MXFP, DMA bounds checking with client-pointer magic `0x54445841` ("TDXA") | confirmed | crt-gemm-c, crt-send-c |
| Op Libraries | **FlagGems** (open) — the primary public operator library (Triton kernels) | confirmed | flaggems-tsingmicro-init |
| Op Libraries | **oplib_rcs1 / tx8be-oplib / txops / 自研高性能算子库** (closed) — referenced from `CMakeLists.txt` and `build_tx8_deps.sh` | confirmed | flagtree-cmakelists, build-tx8-deps |
| Runtime (host) | **Kuiper host SDK — `tx_runtime.h` / `libhpgr`**, installed at `/usr/local/kuiper` (closed). **HIP-shaped C API**: `txSetDevice`, `txGetDeviceByPCIBusId`, `txMalloc`, `txMemcpyAsync`, `txStreamCreate`, `txEventRecord`, `txIpcGetMemHandle`, **`txLaunchKernelGGL`**; types `txStream_t`, `txEvent_t`, `txError_t`, `TX_SUCCESS` | confirmed | flagcx-device-adaptor, txda-tools-py |
| Runtime (device) | **tx81fw / rcs1fw-rtt** firmware (closed) on **RT-Thread SMP** via T-Head **YoC** (`tx8-yoc-rt-thread-smp`) | confirmed | build-tx8-deps |
| Runtime (intrinsics) | **`libinstr_rcs1.a`** + `instr_def.h`, `instr_adapter.h`, `instr_operator.h`, `common_base.h` (closed) — the accelerator command/instruction ABI | confirmed | flagtree-cmakelists |
| Runtime (simulation) | `triton_cmodel`, `tx8be_op_cmodel`, **`neuralcore_qemu`** (closed): `USE_SIM_MODE=1` builds against a C-model + QEMU neural-core simulator | confirmed | flagtree-driver-py, build-tx8-deps |
| Profiling | **tx-profiler** LLVM IR pass (open) injecting `addOrderProfile` / `TsmWaitfinish` / `printOrderByEvent`; **profiling_tool v5.5.0/v5.6.0**, `rcs_profiling`, `hrt_profiler`, `tsm_profiler.h` (closed); env `ENABLE_PROFILING`, `TSM_PROFILER_EN`, `TRACE_POINTS` | confirmed | flagtree-profiler-cpp |
| Communication | **TCCL — TsingMicro Communication Collectives Library** (closed, `tccl.h`): NCCL-shaped `tcclComm_t`, `tcclResult_t`, `tcclDataType_t`, `tcclRedOp_t`. Open FlagCX adaptors under `USE_TSM_ADAPTOR`; vendor string `"TSMICRO"` | confirmed | flagcx-readme, flagcx-tccl-adaptor |
| Communication | FlagCX matrix: TCCL supports **send, recv, broadcast, gather, scatter, reduce, allreduce, allgather, reducescatter, alltoall, alltoallv, group ops** in **both homogeneous and heterogeneous** modes. Algorithms, topology awareness and achieved bandwidth **not disclosed** | confirmed | flagcx-readme |
| Driver | Kernel driver **not public** — ships inside the Kuiper SDK. Module name and IOCTL interface **not disclosed**. Management CLI **`tsm_smi`** (nvidia-smi analogue). Vendor image `hub.tsingmicro.com/tx8/ubuntu/v5.7.0.0524:tsingmicro_release` | confirmed | flagtree-publish-readme, flaggems-tsingmicro-init |
| Driver | Virtualization / partitioning: claimed in RAISA layer 4 ("驱动、容器、虚拟化") but **no technical detail public** | unconfirmed | tsingmicro-tx8-page |
| ISA | **Two-level.** (a) Scalar: **RISC-V RV64IMFDC / lp64d**, T-Head XuanTie-900 (open standard). (b) Accelerator: coarse-grained **command descriptors** — classes `I_NEUR` (GEMM/conv), `I_RDMA` (DMA); dispatch `RcsExecute(&inst)`, wait `RcsWaitfinish()` / `TsmWaitfinish()` | confirmed | crt-gemm-c, flagtree-compiler-py |
| ISA | Descriptor builder API: `TsmNewGemm/TsmNewConv`, `AddInput/AddWeight/AddBias/AddOutput`, `ConfigMKN`, `ConfigBatch`, `SetPsum`, `SetTransflag`, `SetQuant`, `SetSparse`, `SetPads/SetUnPads/SetKernelStrides/SetDilations`, `SetNegativeAxisScale/SetPositiveAxisScale`, `EnableRelu/EnableLeakyRelu`. **Bit encodings, opcode list and register map not disclosed** | confirmed | tx81ops-td, crt-gemm-c |
| Distribution | `pip install flagtree===0.6.0+tsingmicro3.3` from `resource.flagos.net`; source build via `FLAGTREE_BACKEND=tsingmicro` + `TX8_DEPS_ROOT` + `LLVM_SYSPATH`; bundles `tx8_depends_dev_20260507_104051_v0.6.0`; Python 3.10 / PyTorch 2.7.0 / Ubuntu 22.04 / x86_64 | confirmed | flagtree-wiki-manual |
| CI | `.github/workflows/tsingmicro3.3-build-and-test.yml`, self-hosted runner on **real hardware**, 14 tests. **The TLE NoC ring-GEMM test is commented out with `# TODO: fix`** — the multi-tile distributed path is not CI-green | confirmed | flagtree-ci-workflow |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | **TX81 device: 16 tiles** ("multiprocessors") in a **4 × 4 2-D mesh**; device capability reported as (8,1) = TX81 | confirmed | flaggems-tsingmicro-init, crt-send-c, tle-noc-gemm-example |
| Compute Engine | **Per-tile scalar core**: RISC-V **RV64IMFDC**, ABI lp64d, T-Head **XuanTie 900** — **no RVV**; all vector/matrix work goes to engines | confirmed | flagtree-compiler-py, flagtree-cmakelists |
| Compute Engine | **Neural engine** (`I_NEUR`): GEMM + convolution with fused bias, per-axis scale, ReLU/LeakyReLU, and SPM-resident psum accumulation | confirmed | crt-gemm-c, tx81ops-td |
| Compute Engine | **Hardware backward convolution**: `ConvOp.op_type` = {0 conv, 1 depthwise, 2 **backward conv**, 3 gemm} — consistent with training positioning | confirmed | tx81ops-td |
| Compute Engine | **Structured sparsity in hardware**: `en_sparse` + `src_sparse` on ConvOp. **Pattern, ratio and sparse peak not disclosed** | confirmed | tx81ops-td |
| Compute Engine | Additional engines: vector/elementwise/transcendental, layout (`img2col`, transpose, mirror, rotate, NCHW↔NHWC, pad, concat, gather_scatter, bilinear, lut16/lut32, randgen), reduction/sort (`reduce_*`, cumsum, argmax/argmin, sort, count, histogram) | confirmed | tx81ops-td |
| Compute Engine | **Native compute formats bf16 / fp16 / tf32 / fp32** (explicit dialect header statement); INT8 via `src_fmt`/`dst_fmt` + per-channel INT8 bias; FP8 E4M3/E5M2 (+UZ) and **MX / FP4 block-scaled** in the conversion path (`MXFPScale*`, `FP4E2M1To*`, CRT `mxfp_*.c`) | confirmed | tx81ops-td, tx81types-td |
| Compute Engine | **No FP64, no native INT64** — `fp64_enabled=False`, `int64_enabled=False`; INT64 split into two INT32 lanes by CRT `legalizeMemoryOpAttribute` | confirmed | flaggems-tsingmicro-init, crt-tx81-c |
| Compute Engine | **PE-array dimensions, MAC-lane count, systolic-vs-vector organisation: not disclosed** | confirmed (as undisclosed) | — |
| Compute Engine | **Reconfiguration granularity**: observable granularity is one command descriptor per whole tiled tensor op. Configuration-memory size, context switching, partial reconfiguration: **not disclosed** | likely | tx81ops-td, crt-gemm-c |
| Compute Engine | Peak: **512 TFLOPS FP16 per RPU module**; **4 PFLOPS per REX1032 node**; **>500 PFLOPS for the 4,096-chip REX81 supernode** (千万亿 = 10¹⁵, so PFLOPS not exaflops). **Per-chip TFLOPS not disclosed**; INT8/FP8/FP4/TF32/sparse peaks not disclosed; clock not disclosed | likely (vendor claim) | eefocus, sina, tsingmicro-tx8-page |
| Compute Engine | **Derived only**: ~128 TFLOPS FP16/chip, 4 chips/module, 32 chips/node, 128 nodes/supernode — arithmetic from 4 PFLOPS ÷ 512 TFLOPS and 2 TB ÷ 64 GB. **Never cite as a published spec** | unconfirmed (derived) | flaggems-tsingmicro-init, eefocus |
| Data Path | **SPMD over tiles**: `txLaunchKernelGGL(..., dim3{gridX,gridY,gridZ}, dim3{1,1,1}, ...)` — block dim hard-wired to 1×1×1, so the grid indexes tiles directly, one program instance per tile | confirmed | flagtree-driver-py, tle-noc-gemm-example |
| Data Path | **Command-descriptor dispatch**: RISC-V core fills `RcsNeInstr {I_NEUR}` / `RcsRdmaInstr {I_RDMA}` and calls `RcsExecute(&inst)` — "Dispatch the command to accelerator"; awaits `RcsWaitfinish()` / `TsmWaitfinish()` | confirmed | crt-gemm-c, crt-rdma-c |
| Data Path | **Async DMA with dependency-driven barriers**: `tx81-insert-barrier` emits `tx.barrier` only for RISC-V aliasing hazards and WDMA→RDMA RAW through DDR; between compute commands **"hardware handles ordering"** (implies hardware scoreboarding / in-order engine issue) | confirmed | tx81-passes-td |
| Data Path | **Software pipelining limited to double buffering**: `mk-pipeline` builds (num_stages − 1) SPM prefetch buffers with `num_stages` **default and hard clamp = 2** | confirmed | mkpipeline-passes-td |
| Data Path | Sync primitives: `tx.barrier`, `tx.distribute_barrier(mesh_physical_ids, mesh_shape)` → `__BarrierSubgroup()`, `tx.atomic_barrier_in/out`, SPM-flag spin-wait between neighbouring tiles | confirmed | tx81ops-td, crt-send-c |
| Data Path | Device firmware: `tx81fw` / `rcs1fw-rtt` on RT-Thread SMP via T-Head YoC. Internal subsystem codename **RCS1** (expansion not disclosed) | confirmed | build-tx8-deps |
| On-chip Memory | **Two address spaces only** — `typedef enum { UNKNOWN=0, SPM=1, DDR=2 } MemorySpace`. All compute operands live in SPM; DDR↔SPM exclusively via RDMA/WDMA | confirmed | crt-tx81-def-h, tx81-passes-td |
| On-chip Memory | **No hardware data cache in the compute path.** FlagGems' `L2_cache_size = 3 MB` is literally `SPM_SIZE` re-exported as a Triton API shim, not a cache | confirmed | flaggems-tsingmicro-init |
| On-chip Memory | **SPM 3 MiB per tile** (`SPM_SIZE = 3*1024*1024`); 64 KiB system-reserved, 256 B op-reserved; **3,014,656 B visible to a Triton kernel**; SPM base `spmMappingOffset = 0x30400000` | confirmed | flaggems-tsingmicro-init, flagtree-driver-py, crt-tx81-c |
| On-chip Memory | **Aggregate on-chip SPM 48 MiB** (16 × 3 MiB) — derived. **SPM bandwidth not disclosed**; engine-internal registers/accumulators not disclosed | likely (derived) | — |
| On-chip Memory | Compiler management: `--spmd-allocate-shared-memory` liveness/interference allocator (`Allocation.cpp` 29 KB, `Membar.cpp` 14 KB, `Alias.cpp`); usage recorded as `triton_tsm.spm_use`, surfaced as `metadata["shared"]` | confirmed | flagtree-tsingmicro-backend |
| Off-chip Memory | **64 GB per TX81 device** (`total_memory`, comment `# 64GB` in the vendor-maintained FlagGems backend). Whether this is the shipping configuration or a placeholder is **unverified by any datasheet** | likely | flaggems-tsingmicro-init |
| Off-chip Memory | **Memory technology not disclosed** — the code names the space "DDR(dram)", a generic SDK label; vendor says only "大容量显存超高显存带宽". **Bandwidth not disclosed** | confirmed (as undisclosed) | tx81ops-td, tsingmicro-tx8-page |
| Off-chip Memory | Node memory: **2 TB standard, up to 4 TB** on REX1032 (vendor claim) | likely | eefocus |
| Off-chip Memory | **Roadmap caveat**: "基于国产DRAM的三维存算融合技术" applies to the **next generation**, not TX81. **Do not classify TX81 as PIM or 3-D-stacked** | confirmed | tsingmicro-tx8-page |
| Host Interface / Package | **PCIe confirmed** — FlagCX adaptor uses `txGetDeviceByPCIBusId` and formats BDF `"%04x:%02x:%02x.0"`; **DMA-buf supported** (`tsmicroAdaptorDmaSupport → true`), enabling GPUDirect-RDMA-style paths. **Generation and lane width not disclosed** | confirmed | flagcx-device-adaptor |
| Host Interface / Package | **Process node, foundry, die size, transistor count, package type, card form factor, TDP, clock: all not disclosed.** Dies per package **not disclosed** (`remote_die_id` present in IR but unused — multi-die suggested, not confirmed) | confirmed (as undisclosed) | crt-send-c |
| On-chip Interconnect | **2-D mesh NoC** over the 4×4 tile grid; config registers at `KUIPER_ADDR_MAP_REG_BASE = 0x6A0000`, hardware tile ID at `0x6A0058`. "Kuiper" (柯伊伯) is the platform codename (also `/usr/local/kuiper`) | likely | crt-send-c |
| On-chip Interconnect | **DTE async block transfer**: `__Send(...)` → `direct_dte_send_async` / `direct_fsm_monitor_receive` / `direct_dte_wait_done`, with hardware FSM flow-control monitors | confirmed | crt-send-c |
| On-chip Interconnect | **Globally addressable peer SPM**: `get_tile_spm_addr_base(tile, x, y)` lets a tile directly read/write another tile's SPM (sync flags at `SINGLE_SPM_SYNC_ADDR`, DTE destination). **A distributed shared address space, not message-passing only** | confirmed | crt-send-c |
| On-chip Interconnect | **NoC link width, bandwidth, hop latency: not disclosed** | confirmed (as undisclosed) | — |
| Scale-up Interconnect | **Switchless proprietary chip-to-chip mesh/torus.** Compiler sees a four-level address hierarchy `(remote_chip_id_x, remote_chip_id_y, remote_die_id, remote_tile_id)` via `Tx81_RemoteBufferOp` / `RemoteLoadOp` / `RemoteStoreOp` — **remote load/store semantics, not just DMA copy** | confirmed | tx81ops-td |
| Scale-up Interconnect | `Tx81_DistributeBarrierOp`: subgroup barrier carrying the TLE `device_mesh` topology through the compiler, lowering to `__BarrierSubgroup()` — **collective sync across a chip mesh is a first-class primitive** | confirmed | tx81ops-td |
| Scale-up Interconnect | Vendor claims: 无交换机线性扩展, Mesh/Torus 千卡级集群, 千卡直接互联无需交换机成本, 自研算力网格技术 | likely (vendor claim) | tsingmicro-tx8-page |
| Scale-up Interconnect | **Link bandwidth, SerDes rate, lanes per link, radix, hop latency and physical medium: ALL not disclosed** — the largest evidence gap, since switchless scale-up is the headline claim | confirmed (as undisclosed) | — |
| Scale-out Interconnect | **TCCL** collectives over the chip fabric; FlagCX reports all 12 collective ops in homogeneous and heterogeneous modes. Implementation, algorithms and achieved bandwidth **not disclosed** | confirmed | flagcx-readme |
| Systems | **TX81 RPU module** 512 TFLOPS FP16; **REX1032** 4 PFLOPS / 2 TB (max 4 TB), runs DeepSeek-R1 671B full-precision single-node with stable 128K context; **REX81 supernode** 4,096 chips / >500 PFLOPS — **announced only** | likely (vendor claim) | eefocus, sina, tsingmicro-tx8-page |
| Systems | REX1032 rack units, PSU, cooling, network ports **not disclosed**; REX81 rack count, torus dimensions, cabling **not disclosed** | confirmed (as undisclosed) | — |
| Maturity | **Shipping at modest scale** — Aug-2025 China Unicom Inner Mongolia tenders ≈ RMB 47M (~US$6.5M) total; corroborated by a Sept-2025 RMB 47.9M figure across two China Unicom / 中贝通信 projects | confirmed | eefocus, sina |
| Maturity | **Unverified vendor claims**: 30000+ cumulative card orders, 200+ adapted models, "first-tier domestic cloud-chip shipments 1H2025", ">50% energy reduction at equal compute", thousand-card centres in 东北/浙江/北京/安徽 | unconfirmed | tsingmicro-home, sina |
| Maturity | **Corporate**: not publicly listed; Series C > RMB 2B (Dec 2025, Beijing Energy Group); ChiNext IPO **tutoring / 辅导验收 as of 2026-06-16** (Huatai United Securities) — pre-filing, no prospectus, no audited revenue | confirmed | sina-finance-ipo, qq-ipo |
| Maturity | **No architecture disclosure of any kind** — no ISSCC / ISCA / Hot Chips / MICRO / JSSC paper and no whitepaper for TX8/TX81 as of 2026-08-08 | confirmed | — |

---

## Source Key

| Key | URL |
|-----|-----|
| tsingmicro-home | https://www.tsingmicro.com/ |
| tsingmicro-tx8-page | https://www.tsingmicro.com/products/tx8/series |
| flagtree-readme | https://github.com/FlagTree/flagtree/blob/main/README.md |
| flagtree-tsingmicro-backend | https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro |
| flagtree-compiler-py | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py |
| flagtree-driver-py | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/driver.py |
| txda-tools-py | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/txda_tools.py |
| tx81ops-td | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td |
| tx81types-td | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Types.td |
| tx81-passes-td | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Transforms/Passes.td |
| magickernelops-td | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Dialect/IR/MagicKernelOps.td |
| mkpipeline-passes-td | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/MKPipeline/Passes.td |
| crt-tx81-def-h | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/include/Tx81/tx81_def.h |
| crt-gemm-c | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/gemm.c |
| crt-rdma-c | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/rdma.c |
| crt-send-c | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/send.c |
| crt-tx81-c | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/tx81.c |
| tle-noc-gemm-example | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_4096.py |
| tle-dsa-types | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/dsa/types.py |
| tle-distributed | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/distributed.py |
| flagtree-cmakelists | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/CMakeLists.txt |
| build-tx8-deps | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/build_tx8_deps.sh |
| flagtree-publish-readme | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/README.md |
| flagtree-run-flaggems-multicards | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/run_flaggems_on_multicards.sh |
| flagtree-profiler-cpp | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/profiler/profiler.cpp |
| flagtree-ci-workflow | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/.github/workflows/tsingmicro3.3-build-and-test.yml |
| flagtree-wiki-manual | https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tsingmicro |
| flagtree-wiki-tle | https://github.com/flagos-ai/FlagTree/wiki/TLE |
| flaggems-tsingmicro-init | https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py |
| flaggems-tune-configs | https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/tune_configs.yaml |
| tsingmicro-public-e | https://github.com/tsingmicro-public-e |
| flagcx-readme | https://github.com/FlagOpen/FlagCX/blob/main/README.md |
| flagcx-tccl-adaptor | https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/ccl/tccl_adaptor.cc |
| flagcx-device-adaptor | https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc |
| eefocus | https://www.eefocus.com/article/1888048.html |
| sina | https://news.sina.com.cn/sx/2025-09-15/detail-infqpxhk9790648.shtml |
| sina-finance-ipo | https://finance.sina.com.cn/roll/2026-06-16/doc-inicqnru6292806.shtml |
| qq-ipo | https://news.qq.com/rain/a/20260616A07PQG00 |
