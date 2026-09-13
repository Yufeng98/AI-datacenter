# Tecorigin (太初元碁) SDAA Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*

---

## Overview

Tecorigin (brand 太初元碁; legal entity 太初（杭州）集成电路有限公司, founded November 2019, HQ Hangzhou, R&D in Wuxi/Beijing/Shanghai, systems-integration center in Yancheng) builds the **SDAA** (Software Defined Accelerator Architecture) datacenter AI accelerator line.

The machine is **not a GPU and not a systolic-array TPU**. It is a *heterogeneous many-core* accelerator in the Sunway/Cell idiom, wrapped in a CUDA-shaped software surface:

- The card is a **host/device machine** — "主从异构的物理架构，即 Host-Device 运行模式". The **host CPU is the master**; there is no documented on-device master core.
- The device contains multiple **SPA** (Synergistic Processor Element **Array**). Each SPA owns its own **Global** memory and is exposed to software as an **independent device** — "每个SPA类似于GPU的一张卡".
- Each SPA contains **32 SPEs** (Synergistic Processor Elements, called 从核 "slave cores" in Tecorigin's own kernel comments). An SPE is a fully independent core with its own compute, memory-access and control logic.
- There is **no hardware data cache in the compute path**. The vendor states a strictly three-level, software-managed model: **registers / SPM / Global**, with all movement between levels performed by explicit, programmer- or compiler-issued **DMA** (Global↔SPM), **RMA** (SPE↔SPE scratchpad, direct), and **hardware broadcast**.

The single most important framing for the survey: **Tecorigin publishes no datasheet.** There is no peak TFLOPS figure, no HBM bandwidth, no process node, no die size, and no TDP anywhere in ~3.2 MB of vendor documentation or on the vendor website. Every quantitative fact below comes from *documentation of the programming interface, the driver telemetry tool, the debugger, or a published performance cost model* — that is, from what the software must know about the machine. Those are strong sources for structure and weak-to-absent sources for peak performance. **Do not back-derive peak throughput from anything here.**

### Chip identity

| Item | Value | Provenance |
|---|---|---|
| **Chip family name** | **T1** | Firmware images are `aiflash_v<ver>_T1` (TecoSMI manual); `sdaaDeviceProp_t.clockRate` is documented as "**T1**计算核心SPE的频率" (SDAARuntime manual) |
| Architecture target | **T100 系列** | PCX v1.0.0 compatibility table: "T100系列加速卡: 支持（默认）"; TecoCC flag `--sdaa-arch=pcx_100` |
| Card as reported by driver | `TECO_AICARD_01`; brand `TECO_AI_01`; PCI device ID `0x071900A1` | TecoSMI sample output |
| Card SKUs | 元碁 **T100** (PCIe), **T110** (air-cooled OAM), **T111** (liquid-cooled OAM) | Vendor technology page |
| Systems | AI workstation (1–4 cards), **T1008** (4U/8-card), **I1004** (2U/4-card inference), **T1108** (6U/8-card OAM, "8 IB/RoCE 高速通信"), **T1118** (2U/8-card liquid-cooled), **SuperPOD-128** (128 cards/rack) | Vendor technology page |
| Process node / die size / transistor count / package | **Not disclosed** — absent from all 17 manuals and the entire vendor site | — |

**T100/T110/T111 are card SKUs. T1 is the chip.** This resolves the seed's "no die/chip codename" gap.

---

## 1. Compute Engine

### Hierarchy

```
Card  =  4 × SPA            (4 independent SDAA devices, separate Global memories)
SPA   = 32 × SPE            (SPMD slave-core array, one program image per SPA)
SPE   =  SU + SREG          (scalar unit + scalar registers)
       + VPU + VREG         (vector unit + vector registers)
       + FU                 (matrix / MMA unit — reads and writes SPM only)
       + SPM                (private software-managed scratchpad)
⇒ 128 SPEs per card
```

**4 SPAs per card — CONFIRMED, over-determined.** The seed listed this as an inference from launch scripts. It is in fact stated directly and repeatedly:

- TecoSMI `-q` reports `Minor Number : 0 1 2 3` for one card, and `teco-smi -c` shows four device rows sharing one Bus-Id.
- TecoPyTorch DDP docs: "每张加速卡上有4个可用的SDAA计算设备".
- Teco-vLLM: "1张太初AI加速卡有4个SDAA计算设备", deriving `tensor_parallel_size=8` for 2 cards.
- TecoGDB prints `Switching to SPA 3, SPE 6`.
- Cross-check: Teco-Megatron-LM's DeepSeek-R1-Distill-Llama-70B recipe is TP=4 × PP=8 = 32 ranks on **1 node × 8 cards** = 32 SDAA devices.

**32 SPEs per SPA — CONFIRMED.** TecoGDB's core-dump output prints exactly `Has 32 GPC` with SPE rows 0…31, and documents "SPE：发生异常的SPE，**大于31时表示特殊模块**" (indices above 31 denote special hardware modules, not SPEs). Independently, every perf-optimization example enumerates SPE IDs 0–31 and shipped kernels use `CORE_NUM = 32` / "32个从核".

**128 SPEs per card is DERIVED** (4 × 32) — but from two independently confirmed vendor-stated numbers.

### Per-SPE units

| Unit | Role |
|---|---|
| **SU** + **SREG** | Scalar unit and scalar register file |
| **VPU** + **VREG** | Vector unit: vector arithmetic, type conversion, compare/select, vector load/store |
| **FU** | The matrix / MMA engine. It **reads and writes SPM directly and can touch nothing else** |
| **SPM** | Private on-chip scratchpad, partitioned into heap / stack / local |

Source: `teco-ops/doc/teco-ops-hardware.md`, SDAA C guide §硬件架构, PCX guide §硬件架构.

The perf manual states each core "supports multi-stage, multi-issue pipelines" but gives **no pipeline depth, issue width, or pipeline count**.

### Matrix unit (FU)

| Property | Value | Provenance |
|---|---|---|
| Operation tile | **128 × 32 × 32 MMA per operation** | "由于矩阵乘接口每次进行的是一个 128×32×32 的 MMA" (operator perf manual). Corroborated: PCX `matmul_compute` / `matmul_store` accept `row` = 1…128; SDAA C `matmul` alignment rules single out `k == 32 && n == 32` as the zero-overhead case |
| Dataflow | **Weight-stationary**, double-buffered weight register (`matmul_load_weight`; `.update_weight` switches the load buffer) | PCX guide |
| Accumulation | Accumulator buffer; `.flush_output` controls drain to SPM | PCX guide |
| Weight masking | Individual weight rows maskable / zero-paddable via an 8-bit-plus mask | PCX guide |
| Input stride granularity | **64 bytes** | PCX guide |
| Internal MAC-array shape, cycles per MMA | **Not disclosed** | — |

### Matrix-unit data types — the sharpest architectural finding

PCX `matmul_init` encodes exactly **four** input→output combinations (identical enum in SDAA C as `MatmulDataType`):

| Encoding | Input | Output |
|---|---|---|
| `0x606` | FP16 | FP16 |
| `0x806` | FP16 | FP32 |
| `0x404` | S16 | S16 |
| `0x704` | S16 | S32 |

**There is no BF16, no TF32, no FP8 and no INT8 path in the matrix unit.** Three independent lines of evidence say this is a real hardware limit rather than a documentation gap:

1. `TORCH_SDAA_BF16_CLIP` clips bf16 matmul inputs to **[-65407, 65407]** — essentially the FP16 maximum. bf16 GEMM is *emulated by down-converting into the FP16 unit*.
2. SDAA C states bfloat16 "目前仅支持指针操作" — storage/pointer only, converted via `simd_load_widen` / `simd_store_narrow`.
3. Teco-vLLM's entire quantization menu is **weight-only**: INT8 **W8A16**, INT4 **W4A16**, GPTQ, AWQ, KV-cache-INT8. There is **no W8A8** — exactly what you expect when activations must enter a 16-bit matrix unit.

Consistently, `TORCH_SDAA_CONV_USE_FP32` documents that Conv "算子内部将FP32输入转FP16进行计算" by default.

### Scalar / vector / ISA data types

- **SDAA C**: `half` (IEEE-754 binary16), `bfloat16` (storage only), plus C/C++ base types. Vector types `intv16`, `uintv16`, `floatv16` (16 × 32-bit), `shortv32`, `ushortv32` (32 × 16-bit), `halfv16`. A separate 128-bit SIMD form exists (`simd_round128`, `simd_floor128`, …, plus the `use_simd128` compiler option).
- **PCX virtual ISA**: `.s8/.s16/.s32/.s64`, `.u8/.u16/.u32/.u64`, `.bool`, **`.f16/.f32/.f64`**, `.v<Type><N>` vectors, `.b<N>` byte arrays, `.str`.
- **FP64 is in the ISA** — consistent with the vendor's HPC + AI positioning.
- **Physical VREG width is not disclosed.** Do not infer it from the vector types, which are internally inconsistent (`halfv16` = 256 bit vs `floatv16` = 512 bit).

### Clocks

TecoSMI `-q` exposes four clock domains (sample output):

| Domain | Current | Max | Meaning |
|---|---|---|---|
| `Mpe` | 2000 MHz | 2300 MHz | Management/master processing element — **role undisclosed** |
| **`Spe`** | **2000 MHz** | **3000 MHz** | SPE compute clock; initial frequency is **eFUSE-set** |
| `Hbm` | 1600 MHz | 1650 MHz | HBM |
| `Glb` | 2200 MHz | 2200 MHz | "太初AI加速卡**环网**的实时频率" — the on-chip **ring network** |

Perf manuals benchmark consistently at an **SPE clock of 2.36 GHz**. `teco-smi -lsc/-rsc` lock/reset SPE clocks; `-lpm` enables a low-power mode.

**The `Mpe` domain is a genuine partial correction to the seed's "no on-device management core is documented".** A management/master processing-element clock domain *is* exposed by the driver, and TecoGDB's ">31 = special modules" points the same way. **However**, its role, count, ISA and programmability are **not disclosed**, it is invisible to SDAA C and PCX, and nothing states it is an SW26010-style MPE forming a core group. Report it as an observed clock/telemetry domain only.

### Measured throughput — a micro-benchmark, not a peak

The operator perf-optimization manual reports, for an FP16 `128×32×32` MMA loop resident in SPM at an SPE clock of 2.36 GHz:

- **≈29 TFLOPS** through the matrix unit
- **≈2.5 TFLOPS** through general-purpose vector instructions

Two caveats must survive into the survey:

1. Tecorigin explicitly writes "上述实验**远没有达到**太初AI加速卡乘加运算的浮点性能峰值" — this is a tutorial micro-benchmark, not peak.
2. **The document does not state the scope.** The kernel is `__global__` with no `threadIdx` guard, so it runs on all 32 SPEs of **one SPA**, which makes per-SPA the most likely reading — but that is inference, not vendor statement.

**Peak TFLOPS/TOPS at any precision is not disclosed. Do not scale this number.**

---

## 2. Data Path

### Execution model

- **SPMD**: "同一计算核心阵列SPA内所有计算核心SPE运行同一份应用程序". `threadIdx` = SPE ID within the SPA; `threadDim` = SPE count in the SPA.
- **One SPA = one device**: `sdaaSetDevice(n)` selects the SPA; in TecoPyTorch, "一个SDAA设备对应一个计算核心阵列SPA".
- **Thread groups** (`ThreadGroup`, `thread_group_set_mask` / `include` / `exclude`) scope synchronization, broadcast and RMA to SPE subsets.
- PCX describes intra-thread-group execution as **SIMD** ("每个线程组中的线程以SIMD方式，执行同样的机器指令") — a stronger lock-step claim than SDAA C's SPMD framing. The two vendor documents are **not perfectly consistent**. TecoGDB's per-SPE focus switching and independent PCs suggest the practical model is SPMD with independent control flow (TecoGDB: "目前只支持调试所有SPE运行相同代码的程序").

### Data movement primitives

| Path | Primitive | Forms |
|---|---|---|
| Host ↔ Global | `sdaaMemcpy` | blocking / stream-async |
| Global ↔ SPM | **DMA** | blocking `memcpy`, `memcpy_stride`; non-blocking `memcpy_async` / `memcpy_wait` with `MemcpyHandle` |
| SPE SPM ↔ SPE SPM | **RMA** (direct, no round trip through Global) | `rma_get` / `rma_put`; non-blocking `rma_async_get` / `rma_async_put` / `rma_complete` / `rma_wait`; `RmaNormalMode` and `RmaCustomizeMode` |
| One → many SPEs | **hardware broadcast** | `broadcast`, `broadcast_async`, `memcpy_broadcast`; thread-group scoped |

**RMA plus hardware broadcast between slave cores is the Sunway `athread` idiom** and is the clearest architectural fingerprint of the lineage.

### Async / overlap

Non-blocking DMA / RMA / broadcast / transpose / matmul handles on the device side; host-side **streams and events** with multi-stream concurrency. `sdaaErrorStreamCaptureUnsupported` / `...Invalidated` exist in the runtime error enum, so CUDA-Graph-style capture is at least stubbed.

### Device-side language restrictions (a bare-metal scratchpad target)

No exceptions, no RTTI, no STL, no `new`, no global constructors/destructors, no local statics, no file I/O, no native C/C++ atomics (device atomics are provided as intrinsics: `atomic_inc/add/sub/cas`).

---

## 3. On-chip Memory

The vendor states a **three-level model with no hardware data cache in the compute path**: "太初AI加速卡设备端的存储器模型可以分为3层：Global存储、SPM存储和寄存器" (operator perf manual).

| Level | Scope | Capacity | Notes |
|---|---|---|---|
| SREG / VREG | per SPE | **not disclosed** | Scalar and vector register files; exchange data with SPM and Global |
| **SPM** | **private per SPE** | **≥ 235 KB usable**; hard ceiling **240512 B** for the wrapped allocator | teco-ops README: "SPM 内存申请不超过 235KB"; README_OP.md: "容量有限（约 235KB）… 上限为 240512B". 128 B of wrapper overhead implies 240640 B = exactly 235 KiB. Subdivided by SDAA C into **heap** (`malloc`/`free`, `get_heap_size`), **stack** (`get_stack_size`; movable to Global with `--stack-on-global`) and **local** (`__local__` / `__scoped_local__`, `get_local_size`). Partition sizes are runtime-queried and **never published**. **Physical SPM size is not disclosed** — 235 KB is the usable-allocation ceiling |
| **Instruction cache** | per SPE | **not disclosed** | The **only** cache documented anywhere. SDAA C perf sampling exposes an "指令缓存脱靶次数 (Instruction Cache Miss)" counter via `perf_start`/`perf_stop`/`perf_print`; TecoGDB prints per-SPE `GPC` as "指令cache首地址" |

**No hardware data cache is documented at any level.** The vendor never states this as an explicit negative, but the consistent three-level framing plus mandatory explicit DMA/RMA implies it strongly.

### Measured DMA characteristics (single SPE, 2.36 GHz, 128 KB transfers — SDAA C perf manual)

- Global → SPM blocking DMA: **45.45 GB/s** with 4 B-aligned source *and* destination.
- The same transfer degrades to **0.92 GB/s** — a **49× cliff** — when source and destination have *different* mod-4 residues.
- Direction ranking, fastest to slowest: **SPM→SPM > Global→SPM > SPM→Global > Global→Global**. Global→Global requires SPM as a staging buffer and transiently consumes 20 KB of SPM heap.

These are **per-SPE micro-benchmarks**. Aggregate SPM↔Global bandwidth at full SPA occupancy is **not disclosed**.

---

## 4. Off-chip Memory

| Property | Value | Provenance |
|---|---|---|
| Technology | **HBM** | TecoSMI exposes dedicated `Power` / `Voltage` / `Current` / `Clocks` domains named **`Hbm`** alongside `Chip`, plus per-card "设备内存芯片" telemetry |
| Capacity **per SPA** | **15296 MB** (≈14.94 GB) | Real captured `teco-smi -c` output in the TecoPyTorch and TecoPaddle FAQ sections: `0MB / 15296MB` on each of 4 devices, `35C 90W` |
| Capacity **per card** | **65536 MB = 64 GB** total, 61172 MB free | `teco-smi -q`: `Total : 65536 MB`. 4 × 15296 = 61184 MB usable; the ~4.3 GB gap is plausibly ECC/reserve |
| **Bandwidth** | **NOT DISCLOSED** | Only the HBM clock (1600 MHz, max 1650) is exposed. Generation (HBM2/2E/3), stack count and bus width are all undisclosed, so bandwidth **cannot be derived** |
| ECC | Present | `sdaaErrorECCNotCorrectable` in the SDAARuntime error enum |
| Datasheet capacity | **Not disclosed** | Tecorigin publishes no capacity spec. Treat 64 GB as "documented driver-reported capacity of the T100-generation card"; other configurations may exist |

**There is no shared address space across SPAs.** Each SPA is a separate device with its own Global memory; frameworks treat SPAs as separate ranks (TP=4 within one card). **Whether the 4 SPAs sit on one die, on 4 chiplets in a package, or on 4 separate dies on the card is not disclosed.**

---

## 5. On-chip Interconnect

### Ring network (vendor-stated)

TecoSMI documents the `Glb` clock domain as "太初AI加速卡**环网**的实时频率" — **环网 = ring network**, 2200 MHz. This is the only *direct* vendor statement of on-chip network topology found anywhere. **No bisection bandwidth, link width, or topology diagram is published.**

### SPE array is an 8-wide × 4-deep grid — DERIVED, but from vendor cost models

This is inference, but from an unusually strong source: Tecorigin's own **published performance cost models**, not marketing copy.

1. **RMA distance model.** "令 **D(x, y) = abs(x/8 − y/8) + abs(x%8 − y%8)**，则 D(x, y) 的值越小，性能越好" — a Manhattan distance on a grid whose row index is `id/8` and column index is `id%8`.
2. **Broadcast fast groups** are exactly the rows **{0–7}, {8–15}, {16–23}, {24–31}** and the columns **{0,8,16,24} … {7,15,23,31}** — i.e. row and column broadcast buses.
3. **`sync_threads`** on a thread group is fast when members share a quotient (same row) *or* a remainder (same column) mod 8, and ~2.2× slower otherwise.
4. **DMA bandwidth** is best when the participating SPEs have *distinct* `id % 8` — i.e. `id % 8` selects a memory-side port or bank.

Together these describe an **8 column × 4 row** SPE grid with row and column buses — architecturally the same idea as the Sunway SW26010's 8×8 CPE mesh, at 8×4.

> **Do not cite the SDAA C "横向、纵向广播" example as evidence of topology.** That example builds a *software-simulated* logical 4×4 grouping. The cost models above are the real evidence.

---

## 6. Host Interface / Package

| Property | Value | Provenance |
|---|---|---|
| Host interface | **PCIe Gen4 ×16** | TecoSMI: `PCIe Generation Max: 4`, `Link Width Max: 16x` |
| Device nodes | `/dev/tcaicardN` — one node per **card** (containers pass `--device=/dev/tcaicard0..3`) | PaddleCustomDevice SDAA README; TecoDriver docs |
| Form factors | PCIe card (T100); OAM air-cooled (T110); OAM liquid-cooled (T111) | Vendor technology page |
| Telemetry fields | VBIOS version, MCU version, PCB version | TecoSMI |
| Package / process | **Not disclosed** | — |

---

## 7. Scale-up Interconnect

**No proprietary chip-to-chip link is documented anywhere, and the available evidence is negative:**

- `teco-smi topo` reports only PCIe-class paths. Its legend contains exactly **`SYS / NODE / PHB / PXB / PIX`** — the nvidia-smi PCIe set — with **no NVLink-equivalent entry**. The sample matrix shows two devices connected via `NODE`.
- Peer-to-peer read/write capability is queryable (`topo -p2p r|w`), and the runtime carries `sdaaErrorPeerAccessUnsupported` plus "启动P2P所需的硬件资源已经用尽".
- **PCIe P2P is the documented card-to-card path.** Teco-vLLM runs TP=8 across 2 cards over it.

This is a material architectural limitation relative to NVLink / HCCS / MLU-Link peers, and it is consistent with the SPA-as-independent-device model: the software never assumes a shared address space wider than one SPA.

**SuperPOD-128 rack interconnect topology: not disclosed.**

---

## 8. Scale-out Interconnect

Standard **InfiniBand / RoCE**, using upstream NVIDIA networking software:

- TecoToolKit's multi-node install **requires MLNX_OFED 5.9-0.5.6.0** (including NVIDIA SHARP 3.2.0) and **Open MPI 4.1.5rc2**, redistributed by Tecorigin from `mirrors.tecorigin.com`.
- The T1108 server is advertised with "8 IB/RoCE 高速通信".
- Multi-node vLLM uses **Ray** plus `GLOO_SOCKET_IFNAME`.

Deployment-level figures (**VENDOR CLAIM**, products page, unspecified precision): Yan'an and Lihu use "四链路 400 Gbps 计算网络互联"; the open-source platform uses "四链路 200 Gbps 国产 AI 计算网络"; Tecorigin claims self-developed "200G 高速无损互联技术".

---

## 9. Physical / Power / Thermal

| Property | Value |
|---|---|
| **TDP** | **Not disclosed** |
| Observed idle telemetry | **90 W at 0 % SPE utilization, 35–40 °C** (captured `teco-smi -c` output — *not* a TDP figure) |
| Shutdown temperature threshold | **70 °C** (a slowdown/throttle threshold field also exists) |
| Cooling | Air (T100 PCIe, T110 OAM) / liquid (T111 OAM) |
| Rack-level (**VENDOR CLAIM**, unspecified precision) | 太湖之光A+ — 128 cards per self-designed rack, **32 PFLOPS**, **100 kW**, claimed highest compute density in China |

---

## 10. Host CPU Support — a distinguishing datapoint

TecoDriver and TecoToolKit ship for **five** CPU/OS combinations (environment installation manual v3.2.0), with full framework stacks (TecoPyTorch, TecoPaddle, Teco-vLLM) built and validated for all five:

| Host CPU | ISA | OS |
|---|---|---|
| Intel / AMD | x86_64 | Ubuntu 22.04 |
| **Hygon 海光 7380 / 7375** | x86_64 | Kylin V10 |
| **Phytium 飞腾 S5000C** | ARMv8 | Kylin V10 国防版 |
| **Sunway 申威 8A** | **SW-64** | UOS Server 20 |
| **Loongson 龙芯 3C6000** | **LoongArch** | Loongnix Server 23.1 |

Shipping a full AI framework stack on an **SW-64 host** is close to unique in this registry and is the strongest *concrete* evidence of the Sunway ecosystem tie.

---

## 11. On the Sunway Lineage — refined, not overturned

**No new evidence was found** that the SDAA/T1 chip derives from the SW26010 or shares the SW ISA, and it was explicitly looked for. The seed's downgrade to **team / ecosystem / idiom lineage** stands.

**What the new documentation strengthens:** the athread-style idiom set is now confirmed in far more depth — a slave-core array with a *documented 2-D-mesh cost model*, an LDM-like private scratchpad with heap/stack/local partitions, explicit DMA plus inter-core RMA plus hardware row/column broadcast, SPMD with per-core IDs, FP64 in the ISA, HPC positioning, and first-class SW-64 host support.

**What the new documentation adds against a simple derivation claim:** the programming stack is deliberately CUDA-shaped, and **PCX is an explicitly PTX-like abstraction layer whose entire purpose is to decouple software from the machine ISA**. There is **no on-device MPE+CPE core-group structure documented**; the host CPU is the master. The `Mpe` clock domain and ">31 = special modules" hint at undocumented on-die management logic, but nothing more.

Company provenance (vendor About page): core team from Tsinghua University and the **National Supercomputing Center in Wuxi** (home of Sunway TaihuLight); **three Gordon Bell Prizes**. The flagship deployment is branded **太湖之光A+** ("TaihuLight A+").

---

## 12. Not Disclosed — render as "not disclosed", never estimate

- Process node, foundry, die size, transistor count, monolithic vs chiplet
- Whether the 4 SPAs are 4 dies, 4 chiplets, or 4 on-die clusters
- **Peak dense throughput at any precision** (FP16 / FP32 / FP64 / INT16 / INT8)
- **HBM bandwidth** (per SPA or per card); HBM generation, stack count, bus width
- Official datasheet memory capacity (only driver telemetry is available)
- **Card TDP** (only an idle observation and a 70 °C shutdown threshold)
- Physical SPM size per SPE; heap/stack/local partition sizes
- Scalar and vector register file sizes; physical VREG width
- Aggregate Global bandwidth and SPM↔Global bandwidth at full SPA occupancy
- On-chip NoC bandwidth and precise topology (only "环网" at 2200 MHz + the derived 8×4 grid)
- Role, count, ISA and programmability of the **MPE**
- Existence/bandwidth/topology of any proprietary card-to-card scale-up link (evidence is *negative*)
- SuperPOD-128 rack interconnect topology and bisection bandwidth
- **The T1 machine instruction set** — not disclosed *by design*; PCX exists to hide it
- SPE pipeline depth, issue width, instruction-pipeline count
- Instruction-cache size per SPE
- Whether any hardware data cache exists between SPM and Global (absence strongly implied, never stated)
- Matrix-unit internal MAC-array shape and cycles per 128×32×32 MMA
- **Independent third-party benchmarks — none exist.** No MLPerf, no SPEC, no external review retrievable
- Independent corroboration of the deployment claims (Yancheng 206P, 太湖之光A+ 200P/32 PFLOPS/100 kW, Yan'an 200P, Lihu 300P) — **still OPEN**. These come solely from Tecorigin's own products page; the Yancheng site is self-described as only ">60% domestic content", so it is not necessarily all-Tecorigin silicon
- **Press coverage, analyst reports, funding records, customer references, export-control listings — NOT CHECKED.** WebSearch was unavailable for this entire pass, for the second consecutive research pass
- Unit shipment volumes, production capacity, total deployed card count
- Roadmap beyond the T100 series. PCX's compatibility table lists only "T100系列" while its stated purpose is decoupling software from "多种系列" of Tecorigin hardware — strongly implying successor silicon is planned, but **no successor is named**

---

## Sources

- [SDAA C 编程指南 v3.2.0 (latest v3.3.0)](http://docs.tecorigin.com/release/sdaac) — hardware architecture, matmul, DMA/RMA/broadcast, data types, TecoCC options
- [PCX 编程指南 v1.2.0 (PCX ISA v1.0.0)](http://docs.tecorigin.com/release/pcx) — virtual ISA, `matmul_init` dtype enum, hardware architecture
- [TecoSMI 用户手册 v1.15.0](http://docs.tecorigin.com/release/tecosmi) — clocks, HBM domains, memory totals, PCIe, `topo` legend, T1 firmware naming
- [SDAARuntime 用户手册 v3.2.0](http://docs.tecorigin.com/release/sdaart) — device properties, error enum, "T1计算核心SPE的频率"
- [性能优化手册-算子篇 v1.1.0](http://docs.tecorigin.com/release/op_perf_opt) — 128×32×32 MMA, 3-level memory model, 29 / 2.5 TFLOPS micro-benchmark
- [性能优化手册-SDAA C篇 v2.0.2](http://docs.tecorigin.com/release/sddac_perf_opt) — RMA Manhattan cost model, broadcast row/column groups, DMA bandwidth, instruction-cache counter
- [TecoGDB 命令行工具用户手册 v3.1.0](http://docs.tecorigin.com/release/tecogdb) — "Has 32 GPC", SPE > 31 = special modules, SPA 0–3
- [环境安装手册 v3.2.0](http://docs.tecorigin.com/release/software_installation) — host CPU/OS matrix, MLNX_OFED / Open MPI requirements
- [teco-ops hardware doc](https://github.com/Tecorigin/teco-ops/blob/main/doc/teco-ops-hardware.md) — SPA/SPE/SU/VPU/FU/SPM definitions
- [teco-ops README](https://github.com/Tecorigin/teco-ops/blob/main/README.md) — "SPM 内存申请不超过 235KB"
- [teco-ops operator dev guide](https://github.com/Tecorigin/teco-ops/blob/main/doc/README_OP.md) — SPM ≈235 KB / 240512 B ceiling
- [PaddleCustomDevice SDAA backend (INDEPENDENT, Apache-2.0)](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa) — `/dev/tcaicard0..3`, hardware CI
- [Tecorigin technology / product matrix](https://www.tecorigin.com/cn/technology.html) — T100/T110/T111, T1008/I1004/T1108/T1118, SuperPOD-128
- [Tecorigin solutions & deployments (marketing)](https://www.tecorigin.com/cn/products.html) — 太湖之光A+ 128 cards / 32 PFLOPS / 100 kW, 400 Gbps links
- [Tecorigin about (company facts)](https://www.tecorigin.com/cn/about.html) — founded 2019-11, HQ Hangzhou, NSCC-Wuxi team, 3× Gordon Bell Prize
