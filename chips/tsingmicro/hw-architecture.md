# Tsingmicro (清微智能) TX81 Hardware Architecture

*as_of: 2026-08-08*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*
*Representative products: TX81 (RPU compute module), REX1032 (server node), REX81 (supernode)*

---

## Overview

The **TX81** is Tsingmicro's cloud/datacenter accelerator, self-branded an **RPU (Reconfigurable Processing Unit)** on a "**可重构 2.0**" (Reconfigurable 2.0) architecture. The vendor describes it as its "第一代高性能云端 AI 芯片" (first-generation high-performance cloud AI chip).

Structurally, the device is:

- **16 tiles in a 4 × 4 2-D mesh.**
- Each tile = a **RISC-V RV64IMFDC control core** (T-Head XuanTie 900, no RVV) + a set of **command-driven engines** (neural/GEMM/conv, DMA, tile-to-tile DTE, vector, layout, reduction).
- **3 MiB of software-managed scratchpad (SPM) per tile**, 48 MiB aggregate; **no hardware data cache anywhere in the compute path**.
- **64 GB of off-chip device memory** ("DDR" in the SDK's own naming — the actual technology is undisclosed).
- A **switchless chip-to-chip mesh/torus fabric** exposed to the compiler as a four-level `(chip_x, chip_y, die, tile)` **remote load/store** address space.

The defining architectural characteristic is that **the memory model has exactly two address spaces, `SPM` and `DDR`, and every DDR↔SPM transfer is an explicit compiler-emitted DMA**. This places TX81 in the compiler-scheduled scratchpad class with TPU, Groq and Sophgo, not the GPU class.

> **Evidence basis.** There is **no ISSCC / ISCA / Hot Chips / MICRO / JSSC paper, no architecture whitepaper, and no developer portal** for TX8/TX81. The microarchitecture below is reconstructed from **vendor-authored open source** — principally the Tsingmicro Triton backend in FlagTree (`third_party/tsingmicro`, `triton_v3.3.x`, 609 files) whose 47 KB `Tx81Ops.td` MLIR dialect is an operation-level model of the accelerator, plus the vendor-maintained FlagGems and FlagCX backends.
>
> **Nothing physical is disclosed** — process node, foundry, die size, dies per package, memory technology, memory bandwidth, TDP, clock, PCIe generation, form factor, and every chip-to-chip fabric figure. These are recorded as "not disclosed" throughout, never estimated.

---

## 1. Compute Engine

### 1.1 Device-level organisation

| Component | Specification | Basis |
|-----------|---------------|-------|
| Tiles per device | **16** | vendor-maintained code (`multi_processor_count=16`, `name="TX81"`) |
| Tile topology | **4 × 4 2-D mesh** | inferred from `initTileId(pid, rowLength=4)`, `get_tile_spm_addr_base(tile, 4, 4)`, and a Hamiltonian-cycle physical-ID ring in the vendor's own example |
| Device capability | major=8, minor=1 ("TX81") | vendor-maintained code |
| Warp size reported to Triton | 16 — **vestigial**; block dim hard-wired to 1×1×1 | vendor-maintained code |
| Dies per package | **not disclosed** | `remote_die_id` exists in the IR but is unused (`(void)dieId;`) |

### 1.2 Inside a tile

| Unit | Role |
|------|------|
| **Scalar control core ("kcore")** | RISC-V **RV64IMFDC**, ABI lp64d, T-Head **XuanTie 900**. **No RVV** — all vector and matrix work is offloaded to engines |
| **Neural engine** | GEMM and convolution; `I_NEUR` instruction class; fused ReLU/LeakyReLU (`ActFuncMode`) |
| **RDMA / WDMA** | DDR→SPM and SPM→DDR; `I_RDMA` class; 1-D, 4-D and general strided rank≤8 forms |
| **DTE (Direct Transfer Engine)** | Tile-to-tile block transfer with hardware FSM flow-control monitors |
| **Vector / transcendental / activation** | Large vector op set exposed directly as hardware ops |
| **Layout / data movement** | `img2col`, `transpose`, `mirror`, `rotate90/180/270`, `nchw2nhwc`, `nhwc2nchw`, `pad`, `concat`, `gather_scatter` (async variant), `bilinear`, `lut16`/`lut32`, `randgen` |
| **Reduction / sort** | `reduce_sum/avg/max/min/mul`, `cumsum`, `argmax`, `argmin`, `sort`, `count`, `histogram` |

**PE-array dimensions, MAC-lane count, and whether the neural engine is systolic or vector-organised are `not disclosed`.**

### 1.3 The reconfiguration interface

The vendor's RPU definition is runtime dynamic reconfiguration of "计算单元、互连结构、数据通路". **The only publicly observable reconfiguration is coarse-grained**: the RISC-V core builds a command descriptor and dispatches it with `RcsExecute()`, and one descriptor configures the engine for an entire tiled tensor operation.

The GEMM configuration surface:

| Field group | Contents |
|---|---|
| Operands | A / B / bias / psum addresses **in SPM** |
| Shape | `dims = {M, K, N}`, `ConfigBatch` |
| Accumulation | `en_psum` — accumulate into an SPM partial-sum buffer |
| Transforms | `trans_src_a` / `trans_src_b`, `batch_src_a` / `batch_src_b` |
| Fusion | `AddBias` (per-channel for INT8), `SetNegativeAxisScale` / `SetPositiveAxisScale`, `EnableRelu` / `EnableLeakyRelu` |
| Format | `src_fmt` / `dst_fmt`, `SetQuant` |

The convolution surface adds `SetOpType` — **`0 = conv, 1 = depthwise, 2 = backward conv, 3 = gemm`**. A **hardware backward-convolution mode exists**, consistent with the training positioning. Activations are NHWC; weight dims are (Kx, Ky, Sx, Sy). `SetPads`, `SetUnPads`, `SetKernelStrides`, `SetDilations` complete the descriptor.

**Structured sparsity** is a hardware feature of the conv/GEMM engine — `ConvOp` carries `en_sparse` plus `src_sparse` (a sparse-matrix address in SPM). **The supported pattern, the ratio, and any sparse speedup are `not disclosed`.**

**The finer CGRA properties — configuration-memory size, context switching, partial reconfiguration, per-cycle datapath reconfiguration — are described nowhere public.**

### 1.4 Data types

| Class | Support |
|---|---|
| **Native accelerator compute formats** | **bf16 / fp16 / tf32 / fp32** — per an explicit statement in the dialect header: *"Data format supported by Tx81 ML accelerator are: f16, fp16, tf32, fp32. For Tx81 accelerator unsupported data type, we can either convert it by using `TsmConvert`, or lower the operations to run on RISC-V controller instead."* |
| Float type system | F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64 |
| Integer type system | I1, I4, I8, I16, I32, I64 |
| INT8 GEMM | Supported via `src_fmt`/`dst_fmt` plus per-channel INT8 bias on the descriptor |
| **Block-scaled (MX / FP4)** | **Real in the hardware conversion path** — `FP4E2M1ToBF16Op`, `FP4E2M1ToFP16Op`, `MXFPScaleBF16Op`, `MXFPScaleFP16Op`, plus CRT `mxfp_bf16.c`, `mxfp_fp16.c`, `mxfp_scale_*` and a 16 KB `test_dot_scaled.py` |
| Conversions | ~50 hardware conversion ops covering INT8/16/32 ↔ FP16/BF16/FP32/TF32 plus FP8/FP4 → BF16/FP16 |
| **Not supported** | **FP64** (`fp64_enabled=False`) and **native INT64** (`int64_enabled=False`; INT64 split into two INT32 lanes by the CRT) |

### 1.5 Throughput

| Level | Value |
|---|---|
| TX81 **RPU module** | **512 TFLOPS FP16** (vendor claim) |
| **REX1032 node** | **4 PFLOPS** (vendor claim) |
| **REX81 supernode** | **4,096 TX81 chips, >500 PFLOPS** (vendor product page) — note: 千万亿 = 10¹⁵, so **PFLOPS, not exaflops** |
| **Per TX81 chip** | **not disclosed** |
| INT8 / FP8 / FP4 / TF32 / sparse peak | **not disclosed** |
| Clock frequency | **not disclosed** |

> **Derived decomposition — arithmetic, not a vendor figure.** 4 PFLOPS ÷ 512 TFLOPS = 8 modules per node; 2 TB node memory ÷ 64 GB per device = 32 chips per node → 4 chips per module → **~128 TFLOPS FP16 per chip**. Cross-checks: 4096 × 128 TFLOPS = 524 PFLOPS (matches ">500 PFLOPS"); 4096 ÷ 32 = 128 nodes per supernode; the "max 4 TB" node option maps to 128 GB per chip. Self-consistent, but **inference — never cite as a published spec.**

---

## 2. Data Path

### 2.1 Execution model

**SPMD over tiles.** A kernel launches as `txLaunchKernelGGL(name, binary, size, dim3{gridX,gridY,gridZ}, dim3{1,1,1}, ...)` — the **block dimension is hard-wired to 1×1×1**, so the grid indexes tiles directly: one program instance per tile. `tle.shard_id(MESH, axis=0)` returns the tile's physical id inside the kernel.

**Per tile, the compiled kernel is a RISC-V shared object.** For each tensor operation it fills a command descriptor (`RcsNeInstr {I_NEUR}` for GEMM/conv, `RcsRdmaInstr {I_RDMA}` for DMA), populates it through the builder API, and calls `RcsExecute(&inst)` — documented as "Dispatch the command to accelerator". Completion is awaited with `RcsWaitfinish()` / `TsmWaitfinish()`.

**Ordering.** DMA is asynchronous. The compiler's `tx81-insert-barrier` pass inserts `tx.barrier` **only** when (a) a non-`tx` (RISC-V) op may alias a buffer with a pending async write, or (b) an RDMA reads DDR that a preceding WDMA may have written. Between compute-engine commands the pass documents that **"hardware handles ordering"**, implying hardware scoreboarding or in-order issue inside the engine pipeline.

**Software pipelining.** DDR→SPM prefetch is overlapped with `mk.dot` by the `mk-pipeline` pass, which builds `(num_stages − 1)` SPM buffers with **`num_stages` default and hard clamp = 2** — ping-pong double buffering only, no deeper pipelines.

**Firmware.** Each tile runs vendor firmware `tx81fw` / `rcs1fw-rtt` on **RT-Thread SMP** via T-Head's **YoC** framework.

### 2.2 Per-tile data flow

```
DDR (device memory, 64 GB)
    ↓ RDMA (I_RDMA descriptor, async)
SPM (3 MiB per tile, software-managed, no cache)
    ↓ operand addresses carried in the GEMM/Conv descriptor
Neural engine / vector / reduction / layout engines (I_NEUR class)
    ↓ psum accumulation stays resident in SPM (en_psum)
SPM (result)
    ↓ WDMA (I_RDMA descriptor, async)
DDR
```

Tile-to-tile traffic never touches DDR: the DTE moves SPM → peer SPM directly, and peer SPM is additionally globally addressable for loads, stores and spin-wait sync flags.

---

## 3. On-chip Memory

### 3.1 Two address spaces, no cache

```c
typedef enum { UNKNOWN = 0, SPM = 1, DDR = 2 } MemorySpace;
```

- **All compute-engine operands live in SPM** — every `GemmOp` / `ConvOp` operand is documented as "addr in SPM".
- **DDR↔SPM movement is exclusively explicit DMA.** The barrier pass states it directly: *"DDR↔SPM data movement is exclusively through RDMA (DDR→SPM) and WDMA (SPM→DDR). All other NPU compute ops operate on SPM and need no barriers between them (hardware handles ordering)."*
- **There is no hardware data cache in the compute path.** FlagGems reports `L2_cache_size = 3 MB`, but that value is literally `SPM_SIZE` re-exported to satisfy Triton/PyTorch APIs expecting a cache-size attribute — an API shim, not a cache.

### 3.2 Capacities

| Level | Type | Capacity | Managed by |
|-------|------|----------|-----------|
| Scalar registers | RV64IMFDC GPR/FPR | architectural | hardware |
| Engine-internal registers / accumulators | — | **not disclosed** | — |
| **SPM per tile** | SRAM scratchpad | **3 MiB** (64 KiB system-reserved, 256 B op-reserved) | **compiler** |
| SPM visible to a Triton kernel | — | **3,014,656 B** (3 MiB − 0x10000 − 0x10000) | compiler |
| **Aggregate SPM per device** | — | **48 MiB** (derived: 16 × 3 MiB) | compiler |
| SPM bandwidth | — | **not disclosed** | — |

SPM address-map base in the RISC-V core: `spmMappingOffset = 0x30400000`.

### 3.3 Compiler management of SPM

- Allocation is an MLIR pass (`--spmd-allocate-shared-memory`) backed by a liveness/interference allocator (`Allocation.cpp` 29 KB, `Membar.cpp` 14 KB, `Alias.cpp`). Total usage is recorded as module attribute `triton_tsm.spm_use` and returned to the runtime as `metadata["shared"]`.
- Only **double buffering** is available (`num_stages` clamped to 2), so SPM residency and tiling choices dominate achievable overlap.

Because there is no cache and only two-stage pipelining, **the SPM allocator and the tiling heuristics are the performance-critical parts of the stack** — the same structural property as TPU/XLA and Sophgo/TPU-MLIR.

---

## 4. Off-chip Memory

| Property | Value |
|---|---|
| Capacity per TX81 device | **64 GB** — `total_memory` field with comment `# 64GB` in the vendor-maintained FlagGems backend |
| Technology | **not disclosed.** The code names the space "DDR(dram)", a generic SDK label; the vendor says only "大容量显存超高显存带宽" |
| Bandwidth | **not disclosed** |
| REX1032 node memory | **2 TB standard, up to 4 TB** (vendor claim) |

> **Whether 64 GB is the shipping configuration or a placeholder is unverified by any datasheet.** The only cross-check available is that it is consistent with the vendor's 2 TB/node figure under the derived 32-chips-per-node decomposition.

> **Roadmap caveat — do not attribute to TX81.** The vendor's TX8 page says the **next** generation ("新一代") will adopt "基于国产DRAM的三维存算融合技术" (domestic-DRAM 3-D memory-compute fusion) to raise bandwidth. That is a future-generation statement. **TX81 must not be classified as PIM or 3-D-stacked.**

---

## 5. Host Interface / Package

| Property | Value |
|---|---|
| Host interface | **PCIe** — confirmed. The FlagCX device adaptor uses `txGetDeviceByPCIBusId` and formats device IDs as `"%04x:%02x:%02x.0"` (domain:bus:device.function); DMA-buf is supported (`tsmicroAdaptorDmaSupport → true`), enabling GPUDirect-RDMA-style paths |
| PCIe generation / lane width | **not disclosed** |
| Process node | **not disclosed** |
| Foundry | **not disclosed** |
| Die size / transistor count | **not disclosed** |
| Dies per package | **not disclosed** (multi-die suggested by an unused `remote_die_id` field, not confirmed) |
| Package type | **not disclosed** |
| Card form factor | **not disclosed** |
| Chip TDP / card power | **not disclosed** |
| Clock frequency | **not disclosed** |
| Tape-out / sampling / volume-production dates | **not disclosed** |

---

## 6. On-chip Interconnect (tile ↔ tile)

**2-D mesh NoC over the 4 × 4 tile grid.** Two mechanisms are visible in code:

| Mechanism | API | Nature |
|---|---|---|
| **DTE async block transfer** | `__Send(chipX, chipY, dieId, tileId, dst, src, elem_bytes, data_size, physical_ids, mesh_size, ring_size)` → `direct_dte_send_async` / `direct_fsm_monitor_receive` / `direct_dte_wait_done` | Hardware-FSM-monitored bulk copy |
| **Globally addressable peer SPM** | `get_tile_spm_addr_base(tile, x, y)` | A tile directly reads and writes another tile's SPM — used for spin-wait sync flags at `SINGLE_SPM_SYNC_ADDR` and as the DTE destination. **This is a distributed shared address space, not message-passing only** |

Config registers sit at `KUIPER_ADDR_MAP_REG_BASE = 0x6A0000`; `SCFG_TILE_ID_ADDR 0x6A0058` holds the hardware 2-D logical tile ID. "Kuiper" (柯伊伯) is the SoC/platform codename that also names the host SDK path `/usr/local/kuiper`; the hardware generation it maps to is **not disclosed**.

**NoC link width, bandwidth and hop latency: not disclosed.**

---

## 7. Scale-up Interconnect (chip ↔ chip)

The Tx81 dialect exposes a **four-level physical address hierarchy** to the compiler:

```
remote_chip_id_x, remote_chip_id_y, remote_die_id, remote_tile_id
```

carried by `Tx81_RemoteBufferOp`, `Tx81_RemoteLoadOp` and `Tx81_RemoteStoreOp`.

- **Semantics are remote load/store, not merely DMA copy.** `remote_store` "Store data from the current tile to a destination tile"; `remote_load` "Receive data from a source tile and write it into the given destination buffer". Both carry optional `mesh_physical_ids` / `mesh_shape` attributes so the compiler can route.
- **`Tx81_DistributeBarrierOp`** is a subgroup barrier with mesh topology, carrying the TLE `device_mesh` through the pipeline and lowering to `__BarrierSubgroup()` — **collective synchronisation across a chip mesh is a first-class compiler/hardware primitive.**

Vendor description of the fabric:

| Claim | Chinese |
|---|---|
| Switchless linear scaling | 无交换机线性扩展 |
| Mesh/Torus thousand-card clusters | 支持 Mesh/Torus 拓扑组建千卡级高速智算集群 |
| Also compatible with conventional switched networks | 兼容传统交换机组网架构 |
| Thousand-card direct interconnect, no switch cost | 千卡直接互联，无需交换机成本 |
| Self-developed compute-mesh technology | 自研算力网格技术 |

| Fabric property | Value |
|---|---|
| Link bandwidth | **not disclosed** |
| SerDes rate | **not disclosed** |
| Lanes per link | **not disclosed** |
| Link radix | **not disclosed** |
| Hop latency | **not disclosed** |
| Physical medium (cable / backplane / on-PCB) | **not disclosed** |

> This is the largest evidence gap in the entry: **switchless scale-up is the chip's headline architectural claim, and not one quantitative property of it is public.**

---

## 8. Scale-out and Collectives

- **TCCL — TsingMicro Communication Collectives Library** is a real, named product, listed in FlagCX alongside NCCL/HCCL/CNCL. Closed source; the API surface is visible through the open FlagCX adaptor.
- API is **NCCL-shaped**: `tcclComm_t`, `tcclResult_t`, `tcclDataType_t` (int8/uint8/int32/uint32/int64/uint64/float16/float32/float64/bfloat16), `tcclRedOp_t` (sum/prod/max/min/avg).
- FlagCX's support matrix marks TCCL as covering **send, recv, broadcast, gather, scatter, reduce, allreduce, allgather, reducescatter, alltoall, alltoallv and group ops — in both homogeneous and heterogeneous modes**, one of the more complete rows in that table.
- **Algorithms, topology awareness and achieved bus bandwidth: not disclosed.**

---

## 9. Product Portfolio

| Product | Composition | Peak | Memory | Status |
|---|---|---|---|---|
| **TX81 计算模组** (RPU module) | chips per module **not disclosed** (4 is derived only) | 512 TFLOPS FP16 | — | Shipping |
| **REX1032** 高性能智算服务器 (a.k.a. TX81-1032) | modules per node **not disclosed** (8 is derived only) | 4 PFLOPS | 2 TB (max 4 TB) | Shipping; runs DeepSeek-R1 671B full-precision single-node, stable 128K context |
| **REX81 Supernode** 超节点 | **4,096 TX81 chips** | **>500 PFLOPS** | — | **Announced only** (Bund Conference, Sept 2025) |

REX1032 rack units, PSU rating, cooling and network ports are **not disclosed** (the vendor advertises only a qualitative "低整机功耗"). REX81 rack count, torus dimensions and cabling are **not disclosed** beyond the chip count.

---

## 10. Maturity

| Signal | Evidence |
|---|---|
| **Shipping at modest scale** | **Aug-2025 China Unicom Inner Mongolia tenders** — ~RMB 32M compute cards + >RMB 15M servers ≈ RMB 47M (~US$6.5M), independently reported |
| Corroborating | Sept-2025 vendor statement of RMB 47.9M across two China Unicom / 中贝通信 intelligent-computing private-cloud projects |
| Deployment footprint | Thousand-card centres in 东北, 浙江, 北京, 安徽 — vendor claim, unverified |
| Order volume | "20000+" (Sept 2025) → "30000+ cumulative" (2026 site) — vendor claim, escalating, unverified |
| REX81 supernode | Announced only; no deployed instance evidence |
| Corporate | Not publicly listed; Series C > RMB 2B (Dec 2025, Beijing Energy Group); ChiNext IPO tutoring / 辅导验收 as of 2026-06-16, pre-filing |
| Architecture disclosure | **None** — no ISSCC/ISCA/Hot Chips/MICRO/JSSC paper, no whitepaper |

---

## Sources

- [Tsingmicro TX8 series product page](https://www.tsingmicro.com/products/tx8/series)
- [Tsingmicro corporate homepage](https://www.tsingmicro.com/)
- [Tsingmicro About page](https://www.tsingmicro.com/about)
- [`Tx81Ops.td` — the Tx81 hardware dialect (~150 ops)](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td)
- [`Tx81Types.td` — supported type system](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Types.td)
- [`Transforms/Passes.td` — barrier insertion and the SPM/DDR memory-model statement](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Transforms/Passes.td)
- [`crt/include/Tx81/tx81_def.h` — MemorySpace enum, neural-engine activation modes](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/include/Tx81/tx81_def.h)
- [`crt/lib/Tx81/gemm.c` — I_NEUR descriptor, builder API, RcsExecute](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/gemm.c)
- [`crt/lib/Tx81/rdma.c` — I_RDMA async DMA](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/rdma.c)
- [`crt/lib/Tx81/send.c` — DTE, FSM monitors, 4×4 tile addressing, Kuiper register base](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/send.c)
- [`crt/lib/Tx81/tx81.c` — SPM mapping offset, INT64 legalisation](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/tx81.c)
- [`MKPipeline/Passes.td` — SPM double buffering, num_stages clamp](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/MKPipeline/Passes.td)
- [`backend/driver.py` — launch API, max_shared_mem, warp size](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/driver.py)
- [`backend/compiler.py` — RISC-V toolchain and pipeline](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py)
- [`examples/tle/test_tle_dsa_noc_gemm_4096.py` — 16-tile Hamiltonian ring GEMM](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_4096.py)
- [FlagGems `_tsingmicro/__init__.py` — TX81 device descriptor](https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py)
- [FlagCX `device/tsmicro_adaptor.cc` — PCIe BDF, DMA-buf, tx* runtime](https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc)
- [FlagCX README — TCCL support matrix](https://github.com/FlagOpen/FlagCX/blob/main/README.md)
- [与非网 — 512 TFLOPS module, 4 PFLOPS node, China Unicom tender](https://www.eefocus.com/article/1888048.html)
- [新浪 — Bund Conference Sept 2025 announcement](https://news.sina.com.cn/sx/2025-09-15/detail-infqpxhk9790648.shtml)
