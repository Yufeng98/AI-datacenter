# Tecorigin (太初元碁) SDAA Hardware Architecture

*as_of: 2026-08-08*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*
*Representative products: 元碁 T100 (PCIe card), T110 (air-cooled OAM), T111 (liquid-cooled OAM) — all carrying the **T1** chip*

---

## Overview

Tecorigin's SDAA (Software Defined Accelerator Architecture) accelerator is a **heterogeneous many-core** machine, not a GPU and not a systolic-array TPU. Its structure is the Sunway/Cell idiom — an array of independent slave cores, each with a private software-managed scratchpad, connected by explicit DMA, inter-core RMA and hardware broadcast — presented through a deliberately CUDA-shaped software surface.

```
Host CPU (the master)
   │  PCIe Gen4 ×16
   ▼
Card  ( = /dev/tcaicardN )
   ├── SPA 0 ──┐
   ├── SPA 1   │  4 SPAs per card. Each SPA is an INDEPENDENT SDAA device
   ├── SPA 2   │  with its OWN Global memory. No shared address space
   └── SPA 3 ──┘  across SPAs.
        │
        ├── Global memory (HBM), 15296 MB per SPA
        ├── ring network (环网) @ 2200 MHz
        └── 32 × SPE, organized as an 8-column × 4-row grid
                 ├── SU  + SREG        scalar unit
                 ├── VPU + VREG        vector unit
                 ├── FU                matrix / MMA engine — SPM-only
                 ├── SPM               PRIVATE scratchpad, ≥235 KB usable
                 └── instruction cache (the ONLY cache on the device)

⇒ 128 SPEs per card
```

### The evidentiary situation — read this first

**Tecorigin publishes no datasheet.** There is no peak TFLOPS/TOPS at any precision, no HBM bandwidth, no process node, no die size, and no TDP anywhere in ~3.2 MB of vendor documentation or on the vendor website.

Everything in this document was recovered from four kinds of source:

| Source kind | What it is good for | What it cannot tell you |
|---|---|---|
| Programming manuals (SDAA C, PCX) | Structure, dtypes, tile shapes, primitives | Peak throughput, physical sizes |
| Driver telemetry (`teco-smi`) | Clocks, memory totals, PCIe, topology class, power/thermal observations | TDP, bandwidth |
| Debugger (TecoGDB) | Core counts, SPA/SPE indexing | Microarchitecture |
| Published performance cost models | Relative costs → inferred on-chip topology | Absolute bandwidths |

These are strong on *structure* and silent on *peak performance*. **Do not back-derive peak throughput from anything here.**

> **Retrieval note.** The docs portal serves content as **Yjs CRDT binary** via an undocumented REST API (`http://docs.tecorigin.com/api/api/…`). Byte-scraping the CRDT interleaves edit history and produces **scrambled digits** — an early pass produced "2 TFLOPS" where the text reads "29 TFLOPS". All numbers below were recovered by proper CRDT decoding with `pycrdt`.

---

## 0. Chip Identity and Product Line

| Item | Value | Provenance |
|---|---|---|
| **Chip family name** | **T1** | Firmware images `aiflash_v<ver>_T1` (TecoSMI); `sdaaDeviceProp_t.clockRate` = "**T1**计算核心SPE的频率" (SDAARuntime) |
| Architecture target | **T100 系列** | PCX compatibility table "T100系列加速卡: 支持（默认）"; TecoCC `--sdaa-arch=pcx_100` |
| Driver-reported card | `TECO_AICARD_01`, brand `TECO_AI_01`, PCI device ID `0x071900A1` | TecoSMI sample output |
| Card SKUs | 元碁 **T100** (PCIe), **T110** (air-cooled OAM), **T111** (liquid-cooled OAM) | Vendor technology page |
| Systems | AI workstation (1–4 cards); **T1008** 4U/8-card; **I1004** 2U/4-card inference; **T1108** 6U/8-card OAM ("8 IB/RoCE 高速通信"); **T1118** 2U/8-card liquid-cooled; **SuperPOD-128** 128 cards/rack | Vendor technology page |
| Process node / foundry | **Not disclosed** | — |
| Die size / transistor count / package | **Not disclosed** | — |
| SPA construction (1 die? 4 chiplets? 4 dies?) | **Not disclosed** | — |

**T100/T110/T111 are card SKUs; T1 is the chip.** Earlier revisions of this repo's seed recorded "no die/chip codename"; that gap is now closed.

---

## 1. Compute Engine

### 1.1 Hierarchy — 4 SPAs × 32 SPEs

| Level | Count | Confirmation |
|---|---|---|
| **SPA** per card | **4** | `teco-smi -q`: `Minor Number : 0 1 2 3`; `teco-smi -c` shows 4 device rows sharing one Bus-Id; TecoPyTorch DDP: "每张加速卡上有4个可用的SDAA计算设备"; Teco-vLLM: "1张太初AI加速卡有4个SDAA计算设备"; TecoGDB: `Switching to SPA 3, SPE 6` |
| **SPE** per SPA | **32** | TecoGDB core dump prints `Has 32 GPC`, SPE rows 0…31, and "SPE：发生异常的SPE，**大于31时表示特殊模块**"; all perf examples enumerate SPE 0–31; shipped kernels use `CORE_NUM = 32` / "32个从核" |
| **SPE** per card | **128** | **DERIVED** — 4 × 32, from two independently confirmed vendor-stated numbers |

An independent cross-check: Teco-Megatron-LM's DeepSeek-R1-Distill-Llama-70B SFT recipe is TP=4 × PP=8 = 32 ranks on **1 node × 8 cards** = 32 SDAA devices.

Each SPA is exposed to software as a **separate device** — "每个SPA类似于GPU的一张卡" — with its own Global memory. `sdaaSetDevice(n)` selects the SPA. **There is no shared address space across SPAs**, so even a single card is used model-parallel (TP=4).

### 1.2 Per-SPE units

| Unit | Description |
|---|---|
| **SU** + **SREG** | Scalar unit and scalar register file |
| **VPU** + **VREG** | Vector unit — vector arithmetic, type conversion, compare/select, vector load/store |
| **FU** | Matrix / MMA engine. **Reads and writes SPM directly and can touch nothing else** |
| **SPM** | Private on-chip software-managed scratchpad (heap / stack / local) |

Register file sizes and physical VREG width are **not disclosed**. The perf manual says only that each core "supports multi-stage, multi-issue pipelines"; pipeline depth, issue width and pipeline count are **not disclosed**.

### 1.3 Matrix unit (FU)

| Property | Value |
|---|---|
| Operation tile | **128 × 32 × 32 MMA** |
| Dataflow | **Weight-stationary**, double-buffered weight register (`matmul_load_weight`; `.update_weight` switches the load buffer) |
| Accumulation | Accumulator buffer; `.flush_output` controls the drain to SPM |
| Weight masking | Individual weight rows maskable / zero-paddable via an 8-bit-plus mask |
| Input stride granularity | **64 bytes** |
| Internal MAC-array shape / cycles per MMA | **Not disclosed** |

The tile shape is stated directly ("由于矩阵乘接口每次进行的是一个 128×32×32 的 MMA") and corroborated twice: PCX `matmul_compute` and `matmul_store` accept `row` = 1…128, and SDAA C's `matmul` alignment rules single out `k == 32 && n == 32` as the zero-overhead case.

### 1.4 Matrix-unit data types — FP16 and S16 only

PCX `matmul_init` encodes exactly four input→output combinations (the same enum appears in SDAA C as `MatmulDataType`):

| Encoding | Input | Output |
|---|---|---|
| `0x606` | FP16 | FP16 |
| `0x806` | FP16 | FP32 |
| `0x404` | S16 | S16 |
| `0x704` | S16 | S32 |

**There is no BF16, no TF32, no FP8 and no INT8 path in the matrix unit.**

Three independent corroborations that this is a hardware limit and not a documentation gap:

1. **`TORCH_SDAA_BF16_CLIP` clips bf16 matmul inputs to [−65407, 65407]** — essentially the FP16 maximum. bf16 GEMM is *emulated by down-converting into the FP16 unit*.
2. **SDAA C states bfloat16 "目前仅支持指针操作"** — storage/pointer only, converted via `simd_load_widen` / `simd_store_narrow`.
3. **Teco-vLLM's entire quantization menu is weight-only** — INT8 **W8A16**, INT4 **W4A16**, GPTQ, AWQ, KV-cache-INT8. There is **no W8A8**, exactly what you expect if activations must enter a 16-bit matrix unit.

Consistently, `TORCH_SDAA_CONV_USE_FP32` documents that Conv "算子内部将FP32输入转FP16进行计算" by default.

This is a material competitive datapoint. In a 2026 field where FP8 (and increasingly FP4) is table stakes for large-model serving, the T1 generation's matrix unit tops out at 16-bit inputs, and all low-precision gains must come from weight-only compression.

### 1.5 Scalar / vector / ISA data types

| Layer | Types |
|---|---|
| SDAA C scalar | `half` (IEEE-754 binary16), `bfloat16` (**storage only**), C/C++ base types |
| SDAA C vector | `intv16`, `uintv16`, `floatv16` (16 × 32-bit); `shortv32`, `ushortv32` (32 × 16-bit); `halfv16` |
| SDAA C SIMD128 | `simd_round128`, `simd_floor128`, … plus the `use_simd128` compiler option |
| PCX ISA | `.s8/.s16/.s32/.s64`, `.u8/.u16/.u32/.u64`, `.bool`, **`.f16/.f32/.f64`**, `.v<Type><N>`, `.b<N>`, `.str` |

**FP64 is in the ISA** — consistent with the vendor's HPC + AI positioning and Gordon Bell provenance.

> **Physical VREG width is not disclosed.** Do not infer it from the vector types; they are internally inconsistent (`halfv16` = 256 bit vs `floatv16` = 512 bit).

### 1.6 Clocks

| Domain | Current | Max | Meaning |
|---|---|---|---|
| `Mpe` | 2000 MHz | 2300 MHz | Management/master processing element — **role not disclosed** |
| **`Spe`** | **2000 MHz** | **3000 MHz** | SPE compute clock; initial frequency is **eFUSE-set** |
| `Hbm` | 1600 MHz | 1650 MHz | HBM |
| `Glb` | 2200 MHz | 2200 MHz | "太初AI加速卡**环网**的实时频率" — the on-chip **ring network** |

Perf manuals benchmark consistently at an **SPE clock of 2.36 GHz**. `teco-smi -lsc/-rsc` lock and reset SPE clocks; `-lpm` enables a low-power mode.

**On the `Mpe` domain.** Its existence is a partial correction to the earlier "no on-device management core is documented" — a management/master clock domain *is* exposed by the driver, and TecoGDB's ">31 = special modules" points the same way. But its **role, count, ISA and programmability are not disclosed**, it is invisible to SDAA C and PCX, and nothing states it is an SW26010-style MPE forming a core group. Treat it as an observed clock/telemetry domain only.

### 1.7 Performance — what is and is not known

| Metric | Value |
|---|---|
| Peak FP16 / FP32 / FP64 / INT16 throughput | **Not disclosed** |
| Peak INT8 | **Not disclosed** (and no INT8 matrix path exists) |
| Published micro-benchmark, matrix unit | **≈29 TFLOPS** FP16 |
| Published micro-benchmark, vector instructions | **≈2.5 TFLOPS** |
| Micro-benchmark conditions | 128×32×32 MMA loop resident in SPM, SPE clock 2.36 GHz |
| Micro-benchmark scope | **Not stated.** The kernel is `__global__` with no `threadIdx` guard, so it runs on all 32 SPEs of **one SPA** — per-SPA is the most likely reading, but this is inference |
| Vendor's own caveat | "上述实验**远没有达到**太初AI加速卡乘加运算的浮点性能峰值" — far below peak |

**Do not scale the 29 TFLOPS figure and do not present it as a peak.** The ~11.6× ratio between matrix-unit and vector throughput is, however, a legitimate and useful architectural datapoint.

---

## 2. Data Path

### 2.1 Execution model

- **SPMD** — "同一计算核心阵列SPA内所有计算核心SPE运行同一份应用程序". `threadIdx` = SPE ID within the SPA; `threadDim` = SPE count in the SPA.
- CUDA-shaped surface: `__global__` kernels with `<<<...>>>` launch, `__device__`, `__local__`, `__scoped_local__`, `__host__`, `sdaaMalloc` / `sdaaFree` / `sdaaMemcpy`, streams and events.
- **Thread groups** (`ThreadGroup`, `thread_group_set_mask` / `include` / `exclude`) scope synchronization, broadcast and RMA to SPE subsets.
- PCX describes intra-thread-group execution as **SIMD** ("每个线程组中的线程以SIMD方式，执行同样的机器指令") — a stronger lock-step claim than SDAA C's SPMD framing. **The two vendor documents are not perfectly consistent.** TecoGDB's per-SPE focus switching and independent PCs suggest the practical model is SPMD with independent control flow.
- Device code is a **bare-metal target**: no exceptions, RTTI, STL, `new`, global constructors/destructors, local statics, file I/O, or native C/C++ atomics (device atomics exist as intrinsics).

### 2.2 Data movement primitives

| Path | Primitive | Forms |
|---|---|---|
| Host ↔ Global | `sdaaMemcpy` | blocking / stream-async |
| Global ↔ SPM | **DMA** | `memcpy`, `memcpy_stride`; `memcpy_async` / `memcpy_wait` (`MemcpyHandle`) |
| SPE SPM ↔ SPE SPM | **RMA** — direct, no round trip through Global | `rma_get` / `rma_put`; `rma_async_get` / `rma_async_put` / `rma_complete` / `rma_wait`; `RmaNormalMode`, `RmaCustomizeMode` |
| One → many SPEs | **hardware broadcast** | `broadcast`, `broadcast_async`, `memcpy_broadcast`; thread-group scoped |

**RMA plus hardware broadcast between slave cores is the Sunway `athread` idiom** and is the clearest architectural fingerprint of the lineage.

Async and overlap: non-blocking DMA / RMA / broadcast / transpose / matmul handles on the device; host-side streams and events with multi-stream concurrency. CUDA-Graph-style stream capture is *stubbed* (`sdaaErrorStreamCaptureUnsupported` / `...Invalidated` exist in the runtime enum) but has no documented user-facing API.

---

## 3. On-chip Memory

The vendor states a **three-level model with no hardware data cache in the compute path**: "太初AI加速卡设备端的存储器模型可以分为3层：Global存储、SPM存储和寄存器".

| Level | Scope | Capacity | Managed by |
|---|---|---|---|
| SREG / VREG | per SPE | **not disclosed** | compiler |
| **SPM** | **private per SPE** | **≥235 KB usable**; hard ceiling **240512 B** | programmer / compiler (`malloc`/`free`) |
| **Instruction cache** | per SPE | **not disclosed** | hardware |
| Data cache | — | **NONE DOCUMENTED** | — |

### SPM detail

The usable-allocation ceiling is stated twice in teco-ops: "SPM 内存申请不超过 235KB" (README) and "容量有限（约 235KB）… 上限为 240512B" (README_OP.md). With 128 B of wrapper overhead, 240640 B = exactly 235 KiB.

SDAA C subdivides SPM into three partitions:

| Partition | API | Notes |
|---|---|---|
| heap | `malloc` / `free`, `get_heap_size` | dynamic |
| stack | `get_stack_size` | movable to Global with `--stack-on-global` |
| local | `__local__` / `__scoped_local__`, `get_local_size` | static |

**Partition sizes are runtime-queried and never published, and the physical SPM array size is not disclosed** — 235 KB is a usable ceiling, not an array size.

### Instruction cache — the only cache on the device

Its existence is confirmed by an SDAA C performance counter, "指令缓存脱靶次数 (Instruction Cache Miss)", exposed via `perf_start` / `perf_stop` / `perf_print`, and by TecoGDB printing per-SPE `GPC` as "指令cache首地址". **Size is not disclosed.**

**No hardware data cache is documented at any level.** The vendor never states this as an explicit negative, but the consistent three-level framing plus mandatory explicit DMA/RMA implies it strongly. Functionally this places SDAA in the same class as Cambricon MLU, Tenstorrent Tensix, Sophgo TPU and the Sunway CPE array: **all data locality is the programmer's or compiler's problem.**

### Measured DMA characteristics (single SPE, 2.36 GHz, 128 KB transfers)

| Case | Bandwidth |
|---|---|
| Global → SPM, 4 B-aligned src **and** dst | **45.45 GB/s** |
| Global → SPM, src/dst with different mod-4 residues | **0.92 GB/s** — a **49× cliff** |

Direction ranking, fastest to slowest: **SPM→SPM > Global→SPM > SPM→Global > Global→Global**. Global→Global needs SPM as a staging buffer and transiently consumes 20 KB of SPM heap.

These are **per-SPE micro-benchmarks**. Aggregate SPM↔Global bandwidth at full SPA occupancy is **not disclosed**.

---

## 4. Off-chip Memory

| Property | Value |
|---|---|
| Technology | **HBM** |
| Capacity per SPA | **15296 MB** (≈14.94 GB) |
| Capacity per card | **65536 MB = 64 GB** total (61172 MB reported free) |
| **Bandwidth** | **NOT DISCLOSED** |
| HBM generation / stacks / bus width | **Not disclosed** |
| ECC | Present (`sdaaErrorECCNotCorrectable`) |
| Cross-SPA address space | **None** |

HBM is confirmed by `teco-smi` exposing dedicated `Power` / `Voltage` / `Current` / `Clocks` domains named **`Hbm`** alongside `Chip`, plus per-card "设备内存芯片" telemetry.

Capacities come from **real captured `teco-smi` output** in the TecoPyTorch and TecoPaddle FAQ sections (`0MB / 15296MB` on each of four devices at 35 °C / 90 W) and from `teco-smi -q` (`Total : 65536 MB`). 4 × 15296 = 61184 MB usable of 65536 MB physical; the ~4.3 GB gap is plausibly ECC/reserve.

> **Tecorigin publishes no datasheet capacity.** Treat 64 GB as "documented driver-reported capacity of the T100-generation card". Other configurations may exist.

**Bandwidth cannot be derived.** Only the HBM clock (1600 MHz, max 1650) is exposed; without generation, stack count and bus width there is no valid arithmetic path to GB/s. This is one of the most consequential gaps in the entry — for a 64 GB-class accelerator, bandwidth determines almost everything about LLM decode performance.

---

## 5. On-chip Interconnect

### 5.1 Ring network — vendor-stated

`teco-smi` documents the `Glb` clock domain as "太初AI加速卡**环网**的实时频率" — **环网 = ring network**, running at 2200 MHz. This is the **only direct vendor statement of on-chip network topology** found anywhere. Bandwidth, bisection and link width are **not disclosed**, and no topology diagram is published.

### 5.2 SPE array behaves as an 8 × 4 grid — derived from vendor cost models

This is inference, but from an unusually strong source: Tecorigin's own **published performance cost models**, not marketing copy.

| Evidence | Implication |
|---|---|
| RMA cost model: "令 **D(x, y) = abs(x/8 − y/8) + abs(x%8 − y%8)**，则 D(x, y) 的值越小，性能越好" | Manhattan distance on a grid with **row = id/8, column = id%8** |
| Broadcast fast groups are exactly rows **{0–7}, {8–15}, {16–23}, {24–31}** and columns **{0,8,16,24} … {7,15,23,31}** | **Row and column broadcast buses** |
| `sync_threads` is fast when group members share a quotient (same row) *or* a remainder (same column) mod 8, ~2.2× slower otherwise | Same 8-wide grid |
| DMA bandwidth is best when participating SPEs have **distinct `id % 8`** | `id % 8` selects a memory-side port/bank |

Together: an **8-column × 4-row SPE grid with row/column buses** — architecturally the same idea as the Sunway SW26010's 8×8 CPE mesh, at 8×4.

> **Do not cite the SDAA C "横向、纵向广播" example as topology evidence.** That example constructs a *software-simulated* logical 4×4 grouping. The cost models above are the real evidence.

---

## 6. Host Interface / Package

| Property | Value |
|---|---|
| Host interface | **PCIe Gen4 ×16** (`PCIe Generation Max: 4`, `Link Width Max: 16x`) |
| Device nodes | **`/dev/tcaicardN`** — one node per **card**; containers pass `--device=/dev/tcaicard0..3` |
| Form factors | PCIe FHFL card (T100); OAM air-cooled (T110); OAM liquid-cooled (T111) |
| Reported firmware/board versions | VBIOS, MCU, PCB |
| Package construction | **Not disclosed** |

---

## 7. Scale-up Interconnect

**No proprietary chip-to-chip or card-to-card link is documented anywhere, and the available evidence is negative.**

| Evidence | Reading |
|---|---|
| `teco-smi topo` legend contains exactly **`SYS / NODE / PHB / PXB / PIX`** — the nvidia-smi PCIe set — with **no NVLink-equivalent entry**; the sample matrix shows two devices connected via `NODE` | The management tool has no concept of a proprietary fabric |
| P2P read/write is queryable (`topo -p2p r\|w`); runtime carries `sdaaErrorPeerAccessUnsupported` and "启动P2P所需的硬件资源已经用尽" | **PCIe P2P is the documented card-to-card path** |
| Teco-vLLM runs TP=8 across 2 cards | over PCIe P2P |

This is a material limitation versus NVLink / HCCS / MLU-Link peers, and it is architecturally consistent with the SPA-as-independent-device model: the software never assumes a shared address space wider than one SPA.

**SuperPOD-128 rack interconnect topology and bisection bandwidth: not disclosed.**

---

## 8. Scale-out Interconnect

Standard **InfiniBand / RoCE**, built on upstream NVIDIA networking software:

| Component | Requirement |
|---|---|
| MLNX_OFED | **5.9-0.5.6.0**, including **NVIDIA SHARP 3.2.0** |
| MPI | **Open MPI 4.1.5rc2** |
| Distribution | Redistributed by Tecorigin from `mirrors.tecorigin.com` |
| Multi-node frameworks | **Ray** + `GLOO_SOCKET_IFNAME` for vLLM |
| Server | T1108 advertised with "8 IB/RoCE 高速通信" |

Note the strategic tension: a domestic-substitution accelerator whose only high-bandwidth scale-out path depends on an NVIDIA networking software stack.

Deployment-level link claims (**VENDOR MARKETING**, unspecified precision): Yan'an and Lihu use "四链路 400 Gbps 计算网络互联"; the open-source platform uses "四链路 200 Gbps 国产 AI 计算网络"; Tecorigin claims self-developed "200G 高速无损互联技术".

---

## 9. Physical / Power / Thermal

| Property | Value |
|---|---|
| **TDP** | **Not disclosed** |
| Observed idle telemetry | **90 W at 0 % SPE utilization, 35–40 °C** — captured `teco-smi -c` output, **not a TDP figure** |
| Shutdown temperature threshold | **70 °C** (a slowdown/throttle threshold field also exists) |
| Cooling | Air (T100, T110) / liquid (T111) |
| Rack-level (**VENDOR CLAIM**) | 太湖之光A+: 128 cards per self-designed rack, **32 PFLOPS**, **100 kW**; claimed highest compute density in China |

---

## 10. Host CPU Support

| Host CPU | ISA | OS |
|---|---|---|
| Intel / AMD | x86_64 | Ubuntu 22.04 |
| **Hygon 海光 7380 / 7375** | x86_64 | Kylin V10 |
| **Phytium 飞腾 S5000C** | ARMv8 | Kylin V10 国防版 |
| **Sunway 申威 8A** | **SW-64** | UOS Server 20 |
| **Loongson 龙芯 3C6000** | **LoongArch** | Loongnix Server 23.1 |

Full framework stacks (TecoPyTorch, TecoPaddle, Teco-vLLM) are built and validated for **all five**. Shipping a complete AI stack on an **SW-64 host** is close to unique in this registry and is the strongest *concrete* evidence of the Sunway ecosystem tie.

---

## 11. Sunway Lineage Assessment

**Verdict: team / ecosystem / idiom lineage — NOT architectural derivation.**

| Supports the lineage | Cuts against derivation |
|---|---|
| Core team from Tsinghua and the **National Supercomputing Center in Wuxi**; three Gordon Bell Prizes | **No source states** the T1 shares the SW ISA or derives from SW26010 — explicitly searched for, not found |
| First-class **Sunway SW-64 host CPU** support in the shipping stack | The programming stack is deliberately **CUDA-shaped** |
| Flagship deployment branded **太湖之光A+** | **PCX** is an explicitly PTX-like layer whose whole purpose is to *decouple* software from the machine ISA |
| athread idiom set in depth: slave-core array (SPEs literally 从核), LDM-like private scratchpad, DMA + RMA + row/column broadcast, SPMD with per-core IDs, FP64 in the ISA, documented 2-D-mesh cost model | **No on-device MPE+CPE core-group structure** is documented; the **host CPU is the master** |

The `Mpe` clock domain and TecoGDB's ">31 = special modules" hint at undocumented on-die management logic — but that is a hint, not an MPE+CPE architecture, and nothing more can be said.

---

## 12. Comparison Notes for the Survey

| Axis | Tecorigin T1 | Nearest analogues in this registry |
|---|---|---|
| Core model | 128 independent slave cores/card with private scratchpads, SPMD | Sunway SW26010 CPE array; Cell SPE; Tenstorrent Tensix (though Tensix is dataflow-routed) |
| Memory model | 3-level, **no data cache**, explicit DMA + inter-core RMA + HW broadcast | Sunway athread; Cambricon MLU; Sophgo TPU (compiler-managed SRAM) |
| Matrix engine | Per-core 128×32×32 MMA, weight-stationary, **FP16/S16 only** | Far behind FP8-capable 2026 peers |
| ISA strategy | Public **virtual ISA (PCX)**, secret machine ISA | NVIDIA PTX/SASS — the same split, deliberately imitated |
| Scale-up | **PCIe P2P only** | Sophgo BM1684X (pre-SG-Link); weaker than Ascend HCCS, Cambricon MLU-Link, NVLink |
| Memory capacity | ~14.9 GB per device, 64 GB per card, **bandwidth undisclosed** | Capacity comparable to mid-range parts; bandwidth unknown, which blocks any LLM-decode comparison |
| Disclosure level | **Lowest in the registry** — no peak FLOPS, no bandwidth, no node, no TDP | Comparable to the least-disclosed Chinese vendors; worse than Sophgo, which at least has a vendor filing |

---

## 13. Not Disclosed — never estimate these

Process node · foundry · die size · transistor count · monolithic vs chiplet · whether the 4 SPAs are dies/chiplets/clusters · **peak throughput at any precision** · **HBM bandwidth** · HBM generation/stacks/bus width · datasheet memory capacity · **TDP** · physical SPM size · SPM partition sizes · SREG/VREG sizes · physical VREG width · aggregate Global bandwidth · NoC bandwidth and precise topology · MPE role/count/ISA/programmability · any proprietary scale-up link (evidence is negative) · SuperPOD-128 topology and bisection bandwidth · **the T1 machine ISA** (hidden by design) · SPE pipeline depth/issue width · instruction-cache size · existence of any data cache (absence implied, never stated) · matrix-unit MAC-array shape and cycles per MMA · **independent third-party benchmarks (none exist)** · independent corroboration of deployment claims · press/analyst/funding/export-control status (**not checked — WebSearch unavailable for two consecutive passes**) · shipment volumes · roadmap beyond T100 (implied by PCX's "多种系列" wording, but **no successor is named**)

---

## Sources

- [SDAA C 编程指南 v3.2.0 (latest v3.3.0)](http://docs.tecorigin.com/release/sdaac) — hardware architecture, matmul, DMA/RMA/broadcast, data types
- [PCX 编程指南 v1.2.0 / PCX ISA v1.0.0](http://docs.tecorigin.com/release/pcx) — virtual ISA, `matmul_init` dtype enum, hardware architecture
- [TecoSMI 用户手册 v1.15.0](http://docs.tecorigin.com/release/tecosmi) — clocks, HBM domains, memory totals, PCIe, `topo` legend, T1 firmware naming, power/thermal
- [SDAARuntime 用户手册 v3.2.0](http://docs.tecorigin.com/release/sdaart) — device properties, error enum, "T1计算核心SPE的频率"
- [性能优化手册-算子篇 v1.1.0](http://docs.tecorigin.com/release/op_perf_opt) — 128×32×32 MMA, 3-level memory model, 29 / 2.5 TFLOPS micro-benchmark
- [性能优化手册-SDAA C篇 v2.0.2](http://docs.tecorigin.com/release/sddac_perf_opt) — RMA Manhattan cost model, broadcast row/column groups, DMA bandwidth, instruction-cache counter
- [TecoGDB 命令行工具用户手册 v3.1.0](http://docs.tecorigin.com/release/tecogdb) — "Has 32 GPC", SPE > 31 = special modules, SPA 0–3
- [环境安装手册 v3.2.0](http://docs.tecorigin.com/release/software_installation) — host CPU/OS matrix, MLNX_OFED / Open MPI requirements
- [TecoPyTorch 用户手册 v3.2.0](http://docs.tecorigin.com/release/torch2.7) — 4 devices per card, `TORCH_SDAA_BF16_CLIP`, conv FP32→FP16
- [Teco-vLLM 用户手册 v3.2.0](http://docs.tecorigin.com/release/teco_vllm) — weight-only quantization menu, TP across cards
- [teco-ops hardware doc](https://github.com/Tecorigin/teco-ops/blob/main/doc/teco-ops-hardware.md) — SPA/SPE/SU/VPU/FU/SPM
- [teco-ops README](https://github.com/Tecorigin/teco-ops/blob/main/README.md) — "SPM 内存申请不超过 235KB"
- [teco-ops operator dev guide](https://github.com/Tecorigin/teco-ops/blob/main/doc/README_OP.md) — SPM ≈235 KB / 240512 B ceiling
- [PaddleCustomDevice SDAA backend (INDEPENDENT, Apache-2.0)](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa) — `/dev/tcaicard0..3`, hardware CI
- [Tecorigin technology / product matrix](https://www.tecorigin.com/cn/technology.html)
- [Tecorigin solutions & deployments (marketing)](https://www.tecorigin.com/cn/products.html)
- [Tecorigin about (company facts)](https://www.tecorigin.com/cn/about.html)
