# Tsingmicro TX81 Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*

---

## Overview

**Beijing Tsingmicro Intelligence Technology** (北京清微智能科技股份有限公司, founded 2018-07-26) is a Chinese fabless AI chip company with a Tsinghua reconfigurable-computing lineage. It ships two architecturally distinct product lines:

1. **TX5 series** (TX510 vision, TX210 voice) — edge/embedded NPUs served by the **TS.Knight → RNE** toolchain. **Out of scope for this entry.**
2. **TX8 series** — the cloud/datacenter line. The datacenter part is the **TX81**, marketed as an **RPU (Reconfigurable Processing Unit)** built on a self-developed "**可重构 2.0**" (Reconfigurable 2.0) architecture and described by the vendor as its "第一代高性能云端 AI 芯片" (first-generation high-performance cloud AI chip).

Systems: **TX81 计算模组** (RPU compute module) → **REX1032** server node → **REX81** supernode.

### The evidence situation — read this before using any number below

**There is no ISSCC, ISCA, Hot Chips, MICRO or JSSC paper for TX8/TX81, and no architecture whitepaper.** The vendor publishes no datasheet and operates no developer portal (`developer/doc/docs/open.tsingmicro.com` all fail to resolve).

Almost everything microarchitectural in this document is therefore reconstructed from **vendor-authored open source** — the Tsingmicro Triton backend upstreamed into BAAI's FlagTree (`third_party/tsingmicro`, 609 files, `triton_v3.3.x` branch), plus the vendor-maintained FlagGems and FlagCX backends. The 47 KB `Tx81Ops.td` MLIR dialect is effectively a published operation-level model of the accelerator, and the C runtime (`crt/`) exposes the command-descriptor dispatch interface directly.

Confidence tags used below:

| Tag | Meaning |
|-----|---------|
| `[primary-vendor]` | Vendor site / spec statement |
| `[vendor-claim]` | Marketing or trade-press repetition of a vendor figure, unverified |
| `[code]` | Extracted from vendor-authored open-source code |
| `[derived]` | Arithmetic performed here, explicitly flagged as inference |
| `[not disclosed]` | Not public — do not estimate |

The vendor uses "TX81" for both the die and the module in marketing copy. Trade press distinguishes them ("TX81 单个 RPU 模组" = one RPU *module*; "4096 颗 TX81 芯片" = 4096 TX81 *chips*). This document keeps chip and module separate.

---

## 1. Compute Engine

### 1.1 Positioning and reconfiguration model

The vendor's own definition of RPU, as reproduced in its PR: "基于计算需求和数据流特性，在运行过程中利用动态重构技术实现**计算单元、互连结构、数据通路**的动态按需配置" — runtime dynamic reconfiguration of compute units, interconnect structure and datapath. `[vendor-claim]`

The vendor further claims "全球首款商用可重构计算芯片 … 在同等算力下将能耗降低 50% 以上" (world's first commercial reconfigurable computing chip; >50% energy reduction at equal compute). `[vendor-claim]` — no measured basis is published.

**The actual reconfiguration granularity visible in public code is coarse: one command descriptor configures an engine for an entire tiled tensor operation.** The finer CGRA properties — configuration-memory size, context switching, partial reconfiguration, per-cycle datapath reconfiguration — are **not described in any public source**. `[not disclosed]`

### 1.2 Tile organisation

| Property | Value | Confidence |
|---|---|---|
| Tiles ("multiprocessors") per TX81 device | **16** | `[code]` — FlagGems `_tsingmicro/__init__.py`: `multi_processor_count=16`, `name="TX81"`; TLE examples `TILE_NUM = 16` |
| Tile topology | **4 × 4 2-D mesh** | `[code]`, inferred — `send.c`: `initTileId(coreIndex, /*rowLength=*/4)`, `get_tile_spm_addr_base(tile, 4, 4)`; the TLE example's `TILE_PHYSICAL_RELATION = [0,1,2,3,7,11,15,14,13,12,8,9,10,6,5,4]` is exactly a Hamiltonian cycle over a 4×4 grid (every hop is a nearest-neighbour step) |
| Device capability reported to Triton | major=8, minor=1 (i.e. "TX81") | `[code]` |
| Warp size reported to Triton | 16 — **vestigial**; block dim is hard-wired to 1×1×1 | `[code]` — `GPUTarget("txda", capability, warp_size=16)` |

### 1.3 Inside a tile

Each tile is a **RISC-V control core plus a set of command-driven fixed-function / reconfigurable engines**:

- **Scalar control core ("kcore")** — RISC-V **RV64IMFDC**, ABI **lp64d**, built with the **T-Head XuanTie 900-series** bare-metal SDK (`Xuantie-900-gcc-elf-newlib-x86_64-V2.8.0`, `riscv64-unknown-elf-gcc` 10.4.0). Note the **absence of the RVV vector extension** in `-march`: all vector and matrix work goes to the engines, not to RISC-V vector instructions. `[code]`
- **Neural engine** — issues `I_NEUR`-class instructions; serves both GEMM and convolution. `ActFuncMode { None, ENRelu, ENLeakRelu }` is documented in the CRT header as "Neural engine activate mode". `[code]`
- **DMA engines** — `RDMA` (DDR→SPM) and `WDMA` (SPM→DDR), `I_RDMA` class; 1-D, 4-D and general strided rank≤8 forms. `[code]`
- **DTE (Direct Transfer Engine)** — tile-to-tile transfer with hardware FSM flow-control monitors (`direct_dte_attach`, `direct_dte_send_async`, `direct_dte_wait_done`, `direct_fsm_monitor_init/receive/deinit`). `[code]`
- **Vector / elementwise / transcendental / activation units** — a large vector op set exposed directly as hardware ops in the Tx81 dialect. `[code]`
- **Data-movement / layout engines** — `img2col`, `transpose`, `mirror`, `rotate90/180/270`, `nchw2nhwc`, `nhwc2nchw`, `pad`, `concat`, `gather_scatter` (with async variant), `bilinear`, `lut16`/`lut32`, `randgen`. `[code]`
- **Reduction / sort units** — `reduce_sum/avg/max/min/mul`, `cumsum`, `argmax`, `argmin`, `sort`, `count`, `histogram`. `[code]`

**PE array dimensions, number of MAC lanes, and systolic-vs-vector organisation inside the neural engine are `[not disclosed]`.**

### 1.4 The engine configuration surface (the concrete "reconfiguration" interface)

The RISC-V core builds a command descriptor and dispatches it with `RcsExecute()`. One descriptor configures the engine for an entire tensor operation. This is the most direct public evidence of the coarse-grained reconfiguration model.

**`Tx81_GemmOp`** wraps the intrinsic sequence `TsmNewGemm, TsmDeleteGemm, AddInput, ConfigMKN, AddOutput, SetPsum, SetTransflag, SetQuant, ConfigBatch, EnableRelu, EnableLeakyRelu, DisableRelu, DisableLeakyRelu, AddBias, SetNegativeAxisScale, SetPositiveAxisScale`.
Operands: A/B/bias/psum addresses **in SPM**, `dims = {M,K,N}`, `en_psum` (accumulate into an SPM partial-sum buffer), `trans_src_a/b`, `batch_src_a/b`, `relu_mode`, per-channel INT8 bias, negative/positive axis scale, `src_fmt`/`dst_fmt`. `[code]`

**`Tx81_ConvOp`** wraps `TsmNewConv, AddInput, AddWeight, AddBias, AddOutput, SetOpType, SetNegativeAxisScale, SetPositiveAxisScale, SetSparse, SetPsum, SetPads, SetUnPads, SetKernelStrides, SetDilations, EnableRelu, EnableLeakyRelu, SetQuant`.
`op_type`: **0 = conv, 1 = depthwise conv, 2 = backward conv, 3 = gemm** — i.e. a **backward-convolution mode exists in hardware**, consistent with the training positioning. Activations are NHWC; weight dims are (Kx, Ky, Sx, Sy). `[code]`

**Structured sparsity**: `ConvOp` carries `en_sparse` + `src_sparse` (a sparse-matrix address in SPM). Sparsity is a hardware feature of the conv/GEMM engine. **The supported pattern and ratio are `[not disclosed]`**, and no sparse peak throughput is published. `[code]`

### 1.5 Data types

| Class | Support |
|---|---|
| Float (type system) | F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64 |
| TF32 | Present throughout the conversion op set and CRT (`tf32_fp32.c`, `fp32_tf32.c`) |
| Integer | I1, I4, I8, I16, I32, I64 |
| Block-scaled / micro-scaling | `FP4E2M1ToBF16Op`, `FP4E2M1ToFP16Op`, `MXFPScaleBF16Op`, `MXFPScaleFP16Op`; CRT `mxfp_bf16.c` (11 KB), `mxfp_fp16.c` (10 KB), `mxfp_scale_*`; 16 KB `test_dot_scaled.py`. **MX / FP4 block-scaled support is real in the hardware conversion path.** |
| Conversions | ~50 hardware conversion ops covering the INT8/16/32 ↔ FP16/BF16/FP32/TF32 cross-product plus FP8/FP4 → BF16/FP16 |

**Explicit header statement** (`Tx81Ops.td`, lines 10–16): *"Data format supported by Tx81 ML accelerator are: f16, fp16, tf32, fp32. For Tx81 accelerator unsupported data type, we can either convert it by using `TsmConvert`, or lower the operations to run on RISC-V controller instead."* (The first "f16" is almost certainly a typo for bf16.) So the accelerator's **native compute formats are the 16/19/32-bit float family**; INT8 GEMM is reached via the `src_fmt`/`dst_fmt` fields and per-channel INT8 bias, and narrower formats arrive by conversion. `[code]`

**No FP64 and no native INT64**: the FlagGems vendor descriptor sets `fp64_enabled=False, int64_enabled=False`, and the CRT `legalizeMemoryOpAttribute` splits INT64 into two INT32 lanes. `[code]`

### 1.6 Peak throughput

| Metric | Value | Confidence |
|---|---|---|
| Per **RPU module** | **512 TFLOPS FP16** | `[vendor-claim]` |
| Per **REX1032 node** | **4 PFLOPS** | `[vendor-claim]` |
| **REX81 supernode** | **4,096 TX81 chips; "算力突破每秒 500 千万亿次" = >500 PFLOPS** (NOT exaflops — 千万亿 = 10¹⁵) | `[primary-vendor]` |
| Per-**chip** TFLOPS | **not disclosed** | — |
| INT8 / FP8 / FP4 / TF32 / sparse peak | **not disclosed** | — |
| Clock frequency | **not disclosed** | — |

> **Derived decomposition — this is arithmetic, NOT a vendor figure.**
> 4 PFLOPS ÷ 512 TFLOPS = **8 RPU modules per REX1032 node**. Vendor-maintained code reports **64 GB per TX81 device**, and the vendor states **2 TB per node** → **32 TX81 chips per node** (2048 ÷ 64) → **4 chips per module** → **~128 TFLOPS FP16 per TX81 chip**.
> Cross-checks: 4096 × 128 TFLOPS = **524 PFLOPS**, matching "突破 500 千万亿次"; 4096 ÷ 32 = **128 REX1032 nodes** per REX81 supernode; the "最高 4T 显存" option corresponds to 128 GB per chip.
> Every step is self-consistent, but **the per-chip figure is inference and must never be cited as a published spec.** `[derived]`

---

## 2. Data Path

### 2.1 Execution model

1. **SPMD over tiles.** A Triton kernel launches as `txLaunchKernelGGL(name, binary, size, dim3{gridX,gridY,gridZ}, dim3{1,1,1}, args, argsize, 0, stream)` — **block dimension hard-wired to 1×1×1**, so the grid indexes tiles directly, one program instance per tile. `tle.shard_id(MESH, axis=0)` returns the tile's physical id inside the kernel. `[code]`
2. **Per tile, the RISC-V core issues coarse-grained engine commands.** The compiled kernel is a **RISC-V shared object** running on the tile's kcore. For each tensor op it fills a descriptor (`RcsNeInstr inst = {I_NEUR, ...}` for GEMM/conv, `RcsRdmaInstr inst = {I_RDMA, ...}` for DMA), populates it through the builder API, and calls `RcsExecute(&inst)` — documented as "Dispatch the command to accelerator". Completion is awaited with `RcsWaitfinish()` / `TsmWaitfinish()`. `[code]`
3. **Asynchrony and ordering.** DMA is asynchronous. The `tx81-insert-barrier` pass inserts `tx.barrier` **only** when (a) a non-`tx` (RISC-V) op may alias a buffer with a pending async write, or (b) an RDMA reads DDR that a preceding WDMA may have written (RAW through DDR between two async DMA engines). Between compute-engine commands, **"hardware handles ordering"** — implying hardware scoreboarding / in-order issue within the engine pipeline. `[code]`
4. **Software pipelining.** The `mk-pipeline` pass builds `(num_stages − 1)` SPM prefetch buffers around loops containing a DDR→SPM copy plus an `mk.dot`, with **default and hard clamp `num_stages = 2`** — ping-pong double buffering only. `[code]`
5. **Synchronisation primitives** — `tx.barrier` (all work items), `tx.distribute_barrier(mesh_physical_ids, mesh_shape)` (subgroup, lowers to `__BarrierSubgroup`), `tx.atomic_barrier_in` / `tx.atomic_barrier_out`, plus SPM-flag spin-wait sync between neighbouring tiles (`tile_sync_by_spm_single_direction`). `[code]`
6. **Firmware.** Each tile runs vendor firmware `tx81fw` / `rcs1fw-rtt` built on **RT-Thread SMP** via T-Head's **YoC** framework (`tx8-yoc-rt-thread-smp`). `[code]`

`RCS1` is the vendor's internal name for the compute subsystem generation, appearing throughout (`rcs1fw-rtt`, `instr_rcs1`, `oplib_rcs1`, `rcs1_spm.h`, `RcsGemm/RcsConv/RcsRdma/RcsExecute`, `rcs_profiling`). **The expansion is `[not disclosed]`** — "Reconfigurable Computing Subsystem" is plausible but unsourced.

### 2.2 Tile data flow

```
DDR (device memory, 64 GB)
    ↓ RDMA  (I_RDMA descriptor, async)
SPM (3 MiB per tile, software-managed)
    ↓ operand addresses in GEMM/Conv descriptor
Neural engine / vector / reduction / layout engines  (I_NEUR class)
    ↓ psum accumulation stays in SPM (en_psum)
SPM (result)
    ↓ WDMA  (I_RDMA descriptor, async)
DDR
```

Tile-to-tile traffic bypasses DDR entirely: the DTE moves SPM→peer-SPM directly, and peer SPM is also globally addressable for loads/stores and sync flags.

---

## 3. On-chip Memory

### 3.1 The critical fact: fully software-managed, no hardware data cache

The Tx81 memory model has exactly **two address spaces**:

```c
typedef enum { UNKNOWN = 0, SPM = 1, DDR = 2 } MemorySpace;
```
`[code]` — `crt/include/Tx81/tx81_def.h`

- **All compute-engine operands live in SPM.** Every `ConvOp` / `GemmOp` operand is documented as "addr in SPM"; the output is "Output matrix C addr in SPM".
- **DDR↔SPM movement is exclusively explicit DMA.** The barrier-insertion pass states it outright: *"DDR↔SPM data movement is exclusively through RDMA (DDR→SPM) and WDMA (SPM→DDR). All other NPU compute ops operate on SPM and need no barriers between them (hardware handles ordering)."* `[code]`
- **There is no evidence of a hardware data cache anywhere in the compute path.** FlagGems reports an `L2_cache_size = 3 MB` field, but that value is literally `SPM_SIZE` re-exported to satisfy Triton/PyTorch APIs that expect a cache-size attribute. It is an API shim, not a cache.

TX81 therefore belongs in the **software-managed scratchpad, compiler-scheduled** class — with TPU, Groq, and Sophgo — rather than the GPU class.

### 3.2 Capacities

| Level | Capacity | Confidence |
|---|---|---|
| Register file (scalar) | RV64IMFDC GPR/FPR set | `[code]` |
| Engine-internal registers / accumulators | **not disclosed** | — |
| **SPM per tile** | **3 MiB** (`SPM_SIZE = 3 * 1024 * 1024`); 64 KiB system-reserved (`SYS_SPM_RESERVED_SIZE`), 256 B op-reserved | `[code]` |
| SPM visible to a Triton kernel | **3,014,656 B** = 3 MiB − 0x10000 − 0x10000 | `[code]` — `driver.py`: `{"max_shared_mem": 1024*1024*3 - 0x10000 - 0x10000}` |
| **Aggregate on-chip SPM per device** | **48 MiB** (16 × 3 MiB) | `[derived]` |
| SPM address-map base in the RISC-V core | `spmMappingOffset = 0x30400000` | `[code]` |
| SPM bandwidth | **not disclosed** | — |

### 3.3 SPM management by the compiler

- Allocation is an MLIR pass (`--spmd-allocate-shared-memory`) with a liveness/interference allocator (`lib/Analysis/Allocation.cpp` 29 KB, `Membar.cpp` 14 KB, `Alias.cpp`). Total usage is recorded as module attribute `triton_tsm.spm_use` and returned to the runtime as `metadata["shared"]`. `[code]`
- Double buffering only: `mk-pipeline`, `num_stages` default and hard clamp 2. `[code]`

---

## 4. Off-chip Memory

| Property | Value | Confidence |
|---|---|---|
| Capacity per TX81 device | **64 GB** (`total_memory` field, comment `# 64GB`) | `[code]`, vendor-maintained FlagGems backend |
| Memory **technology** | **not disclosed** — code names the space "DDR(dram)", a generic SDK label; vendor says only "大容量显存超高显存带宽" | — |
| Memory **bandwidth** | **not disclosed** | — |
| Node memory (REX1032) | **2 TB standard, up to 4 TB** | `[vendor-claim]` |

> **Whether the 64 GB `total_memory` value is the shipping configuration or a placeholder is unverified by any datasheet.** It is consistent with the vendor's 2 TB/node figure under the derived 32-chips-per-node decomposition, which is the only cross-check available.

> **Roadmap caveat — do not attribute to TX81.** The vendor's TX8 page states that the *next* generation ("新一代") will adopt "基于国产DRAM的三维存算融合技术" (domestic-DRAM 3-D memory-compute-fusion) to raise memory bandwidth. This is a **future-generation** statement. **TX81 must not be classified as PIM or 3-D-stacked.** `[primary-vendor]`

---

## 5. Host Interface / Package

| Property | Value |
|---|---|
| Host interface | **PCIe** — confirmed. The FlagCX device adaptor uses `txGetDeviceByPCIBusId` and formats device IDs as `"%04x:%02x:%02x.0"` (domain:bus:device.function) from `txDeviceProperty.devProp.{domainId,busId,deviceId}`. DMA-buf is supported (`tsmicroAdaptorDmaSupport → true`), enabling GPUDirect-RDMA-style paths. `[code]` |
| PCIe generation / lane width | **not disclosed** |
| Process node | **not disclosed** |
| Foundry | **not disclosed** |
| Die size / transistor count | **not disclosed** |
| Dies per package | **not disclosed** — `remote_die_id` exists in the IR but is unused in the public `__Send` implementation (`(void)dieId;`), so multi-die is *suggested, not confirmed* |
| Package type | **not disclosed** |
| Card form factor (FHFL / OAM / proprietary module) | **not disclosed** |
| Chip TDP / card power | **not disclosed** |
| Clock frequency | **not disclosed** |

---

## 6. On-chip Interconnect (tile ↔ tile)

**2-D mesh NoC over the 4×4 tile grid.** `[code, inferred]`

Two mechanisms are visible in code:

1. **DTE async block transfer** with FSM completion monitors:
   `__Send(chipX, chipY, dieId, tileId, dst, src, elem_bytes, data_size, physical_ids, mesh_size, ring_size)` → `direct_dte_send_async` / `direct_fsm_monitor_receive` / `direct_dte_wait_done`. `[code]`
2. **Globally addressable peer SPM.** `get_tile_spm_addr_base(tile, x, y)` returns a base address through which one tile directly reads and writes another tile's SPM — used for spin-wait sync flags at `SINGLE_SPM_SYNC_ADDR` and as the DTE destination (`nextTileBaseAddr + dst`). **This is a distributed shared address space, not message-passing only.** `[code]`

Config registers live at `KUIPER_ADDR_MAP_REG_BASE = 0x6A0000`; `SCFG_TILE_ID_ADDR 0x6A0058` holds the hardware 2-D logical tile ID. "Kuiper" (柯伊伯) is the SoC/platform codename that also names the host SDK install path `/usr/local/kuiper`; **what hardware generation Kuiper maps to is `[not disclosed]`**.

**NoC link width, bandwidth and hop latency: `[not disclosed]`.**

---

## 7. Scale-up Interconnect (chip ↔ chip) — the switchless fabric

The Tx81 dialect exposes a **four-level physical address hierarchy** to the compiler:

```
remote_chip_id_x, remote_chip_id_y, remote_die_id, remote_tile_id
```

carried by `Tx81_RemoteBufferOp`, `Tx81_RemoteLoadOp`, `Tx81_RemoteStoreOp`. `[code]`

- Semantics are **remote load/store**, not merely DMA copy: `remote_store` "Store data from the current tile to a destination tile"; `remote_load` "Receive data from a source tile and write it into the given destination buffer". Both carry optional `mesh_physical_ids` / `mesh_shape` attributes so the compiler can route.
- `Tx81_DistributeBarrierOp` — "Subgroup barrier with mesh topology … carries the TLE `device_mesh` topology through the compiler pipeline and eventually lowers to `__BarrierSubgroup()`". **Collective synchronisation across a chip mesh is a first-class compiler/hardware primitive.** `[code]`
- **`remote_die_id` is present in the IR but unused in the current open-source `__Send`** (`(void)dieId;`). Multi-die packaging is suggested but not confirmed.

Vendor description: "**无交换机线性扩展**" (switchless linear scaling), "支持 **Mesh/Torus** 拓扑组建千卡级高速智算集群", "兼容传统交换机组网架构", "**千卡直接互联，无需交换机成本**", "自研算力网格技术". `[primary-vendor]` / `[vendor-claim]`

**Link bandwidth, SerDes rate, lanes per link, radix, hop latency, and physical medium (cable / backplane / on-PCB): all `[not disclosed]`.** This is the single most important undisclosed number for a chip whose main architectural selling point is switchless scale-up.

---

## 8. Scale-out and Collectives

- **TCCL — TsingMicro Communication Collectives Library** is a real, named product, listed in FlagCX alongside NCCL/HCCL/CNCL. The API is NCCL-shaped: `tcclComm_t`, `tcclResult_t`, `tcclDataType_t` (int8/uint8/int32/uint32/int64/uint64/float16/float32/float64/bfloat16), `tcclRedOp_t` (sum/prod/max/min/avg). `[code]`
- FlagCX's backend-support matrix marks **TCCL as supporting send, recv, broadcast, gather, scatter, reduce, allreduce, allgather, reducescatter, alltoall, alltoallv and group ops in BOTH homogeneous and heterogeneous modes** — one of the more complete rows in that table. `[code]`
- **TCCL implementation details — algorithms, topology awareness, achieved bus bandwidth — are `[not disclosed]`.** Only the API surface is visible, through the open FlagCX adaptor.

---

## 9. Systems and Products

| Product | Description | Confidence |
|---|---|---|
| **TX81 计算模组** (RPU compute module) | 512 TFLOPS FP16 per module | `[vendor-claim]` |
| **REX1032 高性能智算服务器 / 训推一体服务器** (also "TX81-1032") | 4 PFLOPS/node; 2 TB node memory (max 4 TB); runs DeepSeek-R1 671B full-precision single-node; stable 128K long context; vendor claims >4× the concurrency of comparable products at 128K. Rack units, PSU, cooling, network ports: **not disclosed** | `[vendor-claim]` |
| **REX81 Supernode 超节点** | 4,096 TX81 chips, >500 PFLOPS, "可灵活重组的国产超节点系统"; announced at the 2025 Bund Conference; "入选 2026 中关村论坛重大科技成果". **Announced only — no deployed instance evidence.** Rack count, cabling, torus dimensions: **not disclosed** | `[primary-vendor]` (announcement) / `[vendor-claim]` (award) |

---

## 10. Maturity and Deployment Evidence

| Status | Evidence |
|---|---|
| **Shipping / deployed at modest scale** | **Aug-2025 China Unicom Inner Mongolia tenders**: GPU-card procurement ≈ **RMB 32M** plus server procurement > **RMB 15M**, total "四千七百多万元" (~RMB 47M / ~US$6.5M). Independently reported by 与非网 |
| Corroborating | Sept-2025 vendor statement: two China Unicom / 中贝通信 intelligent-computing private-cloud projects totalling **RMB 47.9M** |
| Deployment footprint | Thousand-card intelligent-computing centres in 东北, 浙江, 北京, 安徽 and "国内多个省份" `[vendor-claim]` |
| Order volume (escalating) | "算力卡订单总量突破 **20000** 枚" (Sept 2025) → "累计算力卡订单量超 **30000+**" (2026 site) `[vendor-claim, unverified]` |
| Model coverage | "已适配 **200+** 模型/应用"; DeepSeek V3.1 adapted within days of release `[vendor-claim]` |
| Market position | "云端算力芯片出货量居**第一梯队**" (2025 H1) `[vendor-claim, unverified]` |
| Corporate | **NOT publicly listed.** Series C > RMB 2B (Dec 2025, led by Beijing Energy Group); ChiNext (创业板) IPO **tutoring / 辅导验收 stage as of 2026-06-16** with Huatai United Securities — pre-filing, no prospectus, no audited revenue public |
| REX81 supernode | **Announced only** |
| Tape-out / sampling / volume-production dates | **not disclosed** |

---

## Sources

- [Tsingmicro TX8 series product page](https://www.tsingmicro.com/products/tx8/series)
- [Tsingmicro corporate homepage](https://www.tsingmicro.com/)
- [Tsingmicro About page](https://www.tsingmicro.com/about)
- [FlagTree `third_party/tsingmicro` backend root](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro)
- [`Tx81Ops.td` — Tx81 hardware dialect, ~150 ops](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td)
- [`Tx81Types.td` — supported type system](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Types.td)
- [`Transforms/Passes.td` — barrier insertion / memory-model statement](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Transforms/Passes.td)
- [`crt/include/Tx81/tx81_def.h` — MemorySpace enum, Neural engine modes](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/include/Tx81/tx81_def.h)
- [`crt/lib/Tx81/gemm.c` — I_NEUR descriptor + RcsExecute](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/gemm.c)
- [`crt/lib/Tx81/send.c` — DTE, FSM monitors, 4×4 tile addressing](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/send.c)
- [`crt/lib/Tx81/rdma.c` — I_RDMA async DMA](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/rdma.c)
- [`crt/lib/Tx81/tx81.c` — SPM mapping offset, INT64 legalisation](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/tx81.c)
- [`examples/tle/test_tle_dsa_noc_gemm_4096.py` — 16-tile ring GEMM](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_4096.py)
- [`MKPipeline/Passes.td` — SPM double buffering](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/MKPipeline/Passes.td)
- [`backend/driver.py` — launch API, max_shared_mem, warp size](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/driver.py)
- [`backend/compiler.py` — RISC-V toolchain and pipeline](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py)
- [FlagGems `_tsingmicro/__init__.py` — TX81 device descriptor (16 tiles, 3 MiB SPM, 64 GB)](https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py)
- [FlagCX `device/tsmicro_adaptor.cc` — PCIe BDF, DMA-buf, tx* runtime](https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc)
- [FlagCX README — TCCL support matrix](https://github.com/FlagOpen/FlagCX/blob/main/README.md)
- [与非网 — 512 TFLOPS module, 4 PFLOPS node, China Unicom tender](https://www.eefocus.com/article/1888048.html)
- [新浪 — Bund Conference Sept 2025 announcement](https://news.sina.com.cn/sx/2025-09-15/detail-infqpxhk9790648.shtml)
- [新浪财经 — ChiNext IPO tutoring status 2026-06-16](https://finance.sina.com.cn/roll/2026-06-16/doc-inicqnru6292806.shtml)
