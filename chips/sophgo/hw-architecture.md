# Sophgo (算能) Hardware Architecture

*as_of: 2026-08-08*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*
*Representative products: BM1690 / SG2260 "TPUv7" (datacenter, on SC11 FP300 card), BM1684X (cloud/edge inference), BM1688 (edge SoC), SG2042 / SG2044 (RISC-V server CPUs), SG2380 (RISC-V SoC + AI)*

---

## Overview

Sophgo produces two architecturally distinct silicon families:

1. **TPU line** (inference-focused through Gen 4; large-model-capable from TPUv7): BM1684 → **BM1684X** (Gen 4) → **BM1688/CV186X** (edge SoC variant) → **BM1690 / SG2260** (TPUv7, datacenter). BM1684/BM1684X/BM1688 all use a high data-width SIMD tensor processor, compiler-managed SRAM scratchpad, ARM Cortex-A53 management CPU, and LPDDR4X memory. BM1690 keeps the compiler-managed, statically scheduled model but scales it out to 8 TPU cores with a large L2 SRAM tier, native FP8, a proprietary SG-Link scale-up fabric, and ~128 GB of low-power DRAM per chip.

2. **RISC-V line** (compute-focused): **SG2042** (64-core RISC-V server), its successor **SG2044**, and **SG2380** (16-core SoC with integrated 20 TOPS AI accelerator). These use T-Head C920 (SG2042/SG2044) or SiFive P670 (SG2380) RISC-V cores and target HPC, developer platforms, and edge Linux.

The defining architectural characteristic of the TPU line is **static compiler scheduling**: TPU-MLIR compiles the entire model graph into a fixed command sequence stored in the `.bmodel` binary, which the runtime streams to the hardware command processor. There is no runtime dynamic scheduling or speculative execution on the TPU. This holds for BM1690 as well; BM1690 adds an on-die RISC-V core as a fallback execution path for long-tail operators the compiler cannot map.

---

## 0. TPU Generation Overview

| Generation | Chip | Silicon name | TPU cores | On-chip SRAM | Off-chip memory (per chip) | Peak (per chip) | Scale-up fabric | Process |
|---|---|---|---|---|---|---|---|---|
| Gen 3 | BM1684 | — | 1 | not disclosed | LPDDR4X | ~17.6 TOPS INT8 | none | 12nm |
| Gen 4 | **BM1684X** | — | 1 | not disclosed | LPDDR4X 16 GB, ~68 GB/s | 32 TOPS INT8 (35.2 Winograd); 16 TFLOPS FP16 | none (PCIe P2P) | 12nm (Samsung) |
| Gen 4 (edge) | **BM1688** / CV186X | — | 1 | not disclosed | LPDDR4X 8 GB, ~34 GB/s | 16 TOPS INT8 / 32 TOPS INT4 | none | 12nm |
| **TPUv7** | **BM1690** | **SG2260** | **8** | **16 MB LMEM/core (128 MB total) + 128 MB L2** | **~128 GB low-power DRAM, ~546 GB/s (derived)** | **~200 TOPS INT8/FP8; ~100 TFLOPS FP16/BF16** (half of the vendor's per-card claim) | **SG-Link** (multi-chip + multi-card) | **not disclosed** |
| TPUv7 (cut-down) | BM1690E | SG2260 | 8 | 16 MB LMEM/core + **16 MB L2** | not disclosed | not disclosed | SG-Link | not disclosed |

*Added 2026-08-08.* BM1690 per-chip figures are derived by halving the vendor's SC11 FP300 card claims, because **the SC11 FP300 card carries two BM1690 chips** (`sophgo/mcu`, `SC11FP300/common.h`: `#define SOC_NUM 2`; the vLLM driver emits `sc11_config.ini` and `sc11_config_chip2.ini`). Microarchitectural values come from Sophgo's own compiler backend, not a datasheet. **No process node or foundry is disclosed for BM1690/SG2260** — do not infer one; Sophgo has been on the US BIS Entity List since 2025-01-16.

---

## 1. Compute Engine

### BM1690 / SG2260 TPU (TPUv7) — added 2026-08-08

Sophgo does not publish a BM1690 datasheet. Everything in the microarchitecture table below is read out of Sophgo's **own open-source compiler backend** — `include/tpu_mlir/Backend/BM168x/BM1690.h` and `python/profile_helper/bm1690_defs.py` in `sophgo/tpu-mlir`. These are the values the compiler's cost model and code generator use, so they describe the machine the compiler believes it is targeting. **They are not a datasheet, and peak TOPS must not be back-derived from them.**

| Component | Specification | Provenance |
|-----------|---------------|-----------|
| TPU cores per chip | **8** | `bm1690_defs.py` Core Num 8; `resnet50_v2_bm1690_f16_core8.bmodel` |
| NPU lanes per core | **64** | `BM1690.h` `NPU_NUM = 64` |
| EU vector width | **512 bit / 64 byte** | `BM1690.h` `EU_BYTES = 64` |
| Local memory (LMEM) | **256 KB per lane × 16 banks = 16 MB per core**; 128 MB across 8 cores | `BM1690.h` `LMEM_BYTES`, `LMEM_BANKS` |
| Modeled TIU / DMA clock | 1000 MHz | `bm1690_defs.py` |
| Native data types | INT4 / INT8 / **FP8** / NF4 / TF32 / FP16 / BF16 / FP32 | vendor filing; FP8 E5M2 codegen confirmed by `resnet50_v2_bm1690_f8e5m2.bmodel` |
| Embedded RISC-V core | Present — fallback execution path for long-tail operators | vendor filing |
| Sort / top-K units | Dedicated hardware units, aimed at billion-scale vector search | vendor filing |
| Process node | **Not disclosed** | — |

**Per-chip peak performance is not directly published.** The vendor publishes only card-level figures for the two-chip SC11 FP300 card:

| Metric | SC11 FP300 card (2 × BM1690) — **vendor claim** | Implied per BM1690 chip |
|---|---|---|
| INT8 / FP8 | >400 TOPS | ~200 TOPS |
| FP16 / BF16 | >200 TFLOPS | ~100 TFLOPS |
| TF32 | >100 TFLOPS | ~50 TFLOPS |
| FP32 | >25 TFLOPS | ~12.5 TFLOPS |
| Power | 300 W whole-card TDP (整卡功耗 300 W) | ~150 W |

Source: China Security & Protection Industry Association innovation-product filing, 2024-09-30, filed by 厦门算能科技有限公司. This is a **vendor-authored document republished by an industry association — marketing, not independent measurement.** No MLPerf or other third-party benchmark of BM1690 is confirmed to exist.

Architecturally, BM1690 is a multi-core scale-out of the same statically scheduled, compiler-managed design used since BM1684X, not a new execution model: 8 independent TPU cores each with a 64-lane, 512-bit-wide SIMD datapath and its own 16 MB scratchpad, sharing a large on-chip L2 SRAM (see §3) and a common off-chip pool. The two genuinely new *kinds* of block are the on-die RISC-V fallback core and the sort/top-K units.

### BM1684X TPU Core

| Component | Specification |
|-----------|---------------|
| Architecture | High data-width SIMD tensor processor |
| Generation | 4th-gen (vs BM1684 Gen 3: 2× performance) |
| Peak INT8 | 32 TOPS |
| Peak INT8 (Winograd) | 35.2 TOPS |
| Peak FP16 / BF16 | 16 TFLOPS |
| Peak FP32 | 2 TFLOPS |
| Supported precisions | INT4, INT8, FP16, BF16, FP32 |
| Process node | 12nm (Samsung) |

The "high data-width SIMD" design is Sophgo's philosophy of using very wide SIMD lanes to amortize instruction decode overhead across many multiply-accumulate operations per instruction. This maximizes compute density per unit area for regular DNN operations (convolutions, matrix multiplications). The internal microarchitecture detail (lane width, number of MAC units) is not publicly disclosed.

### SoC Management CPU (BM1684X)

| Component | Specification |
|-----------|---------------|
| CPU | Octa-core ARM Cortex-A53 |
| Frequency | Up to 2.3 GHz |
| Role | OS/Linux host, network I/O, non-compute tasks |

The ARM cores handle everything except tensor computation: Linux OS management, network serving, pre/post-processing coordination, and BMLib device management calls. All AI inference runs on the TPU core.

### BM1688 / CV186X (Edge SoC)

| Component | Specification |
|-----------|---------------|
| Peak INT8 | 16 TOPS |
| Peak INT4 | 32 TOPS |
| Peak FP16 / BF16 | 4 TFLOPS |
| Peak FP32 | 0.5 TFLOPS |
| CPU | Octa-core ARM Cortex-A53 @ 1.6 GHz |
| Video decode | 16× 1080p30 H.264/H.265 |
| Video encode | 10× 1080p30 H.264/H.265 |
| Process | 12nm |

BM1688 is the power-optimized edge variant, targeting drone vision systems, industrial cameras, and edge compute boxes requiring multi-channel video plus AI inference.

### SG2042 RISC-V Server CPU

| Component | Specification |
|-----------|---------------|
| Core count | 64 (T-Head C920 OoO RISC-V) |
| Organization | 16 clusters × 4 cores |
| L1-D / L1-I per core | 64 KB / 64 KB |
| L2 per cluster | 1 MB |
| L3 system cache | 64 MB |
| Frequency | 2 GHz |
| Process | TSMC 6nm |
| TDP | 120W |
| AI accelerator | None (pure CPU) |

The SG2042 is Sophgo's bet on RISC-V for server/HPC workloads. It delivers 5–10× better performance per core than previous widely available RISC-V hardware, though x86 server CPUs are still 4–8× faster on multi-threaded benchmarks.

### SG2044 RISC-V Server CPU — added 2026-08-08

*Reported by the 2026-08-08 scan; held at lower confidence than the BM1690 material, which was independently re-verified.*

| Component | Specification |
|-----------|---------------|
| Role | Second-generation 64-core RISC-V server CPU, successor to SG2042 |
| Vector extension | **RVV 1.0** (new vs SG2042) |
| Memory subsystem | Improved vs SG2042; controller count/type **not disclosed** |
| Linux support | Platform support merged into **mainline Linux 6.16** |
| Server SKUs | SRA3-40 (compute), SRB3-40 (storage), SRM3-40 (converged) |
| Frequency / process / cache / TDP | **Not disclosed** |

The first comprehensive HPC evaluation is arXiv:2508.13840 (August 2025), comparing SG2044 against SG2042 plus x86 and Arm server parts.

### SG2380 RISC-V SoC

| Component | Specification |
|-----------|---------------|
| Performance cores | 12 × SiFive P670 OoO @ 2.5 GHz |
| Efficiency cores | 4 × 1.6 GHz |
| AI accelerator | 20 TOPS (integrated) |
| GPU | Imagination (3D graphics) |
| VPU | 4Kp60 H.265/H.264/AV1/VP9 |
| Max RAM | 96 GB (192-bit interface) |
| PCIe | x16 |
| Ethernet | Up to 25 GbE |

The SG2380 is Sophgo's closest analogy to an Apple M-series chip: high-performance RISC-V CPU + integrated AI accelerator + GPU + VPU in one SoC.

---

## 2. Data Path

### BM1684X Command Execution Model

The BM1684X TPU is driven by a **static command stream**, not a dynamic GPU-style kernel dispatch. TPU-MLIR compiles the entire model into a `.bmodel` binary encoding a pre-scheduled sequence of tensor operations and DMA commands. BMRuntime's role is to:

1. Load the `.bmodel` (parse network topology + weights)
2. Allocate device memory for inputs/outputs/intermediates
3. Stream the pre-compiled command sequence to the TPU command processor
4. Synchronize on completion

There is no JIT compilation, no PTX/SASS equivalent, and no runtime kernel fusion — all of that happens in TPU-MLIR at compile time.

### DMA Engine

A dedicated DMA engine runs independently of the TPU compute pipeline, enabling compute-DMA overlap (double-buffering):

```
Compute tile N       → TPU processing
DMA prefetch tile N+1 → LPDDR4X → on-chip SRAM
(simultaneous)
```

The DMA is programmed by the `.bmodel`'s embedded DMA command sequence; users do not write DMA code directly.

### BMCV / VPP Pipeline

The BM1684X includes a hardware **VPP (Video Processing Pipeline)** for image preprocessing and a **JPU (JPEG Processing Unit)** for JPEG encode/decode:

```
Camera / video → H.264/H.265 VPU decode
              → JPU JPEG decode
              → VPP: resize, CSC, normalize, crop
              → TPU-accessible DDR buffer
              → BMRuntime: copy to device, launch inference
```

This pipeline enables multi-channel video analysis (up to 32-ch HD decode, 12-ch encode on BM1684X) with minimal CPU involvement.

### BM1690 Multi-Core Data Path — added 2026-08-08

BM1690 preserves the static command-stream model but replicates it across 8 cores. TPU-MLIR's BM1690 backend exposes an explicit multi-core interface (`BM1690.h`) and emits per-core `.bmodel` variants (`..._core8.bmodel`), so core count is a compile-time codegen parameter rather than a runtime scheduling decision. The data path adds one level relative to BM1684X:

```
Off-chip DRAM (~128 GB/chip)
    ↕ DMA
L2 SRAM (128 MB, chip-shared)        ← new tier vs BM1684X
    ↕ DMA
LMEM (16 MB per core, 256 KB × 64 lanes × 16 banks)
    ↕ TPU core compute (64 lanes × 512-bit EU)
```

Two escape hatches sit outside the tensor pipeline: the **on-die RISC-V core** executes long-tail operators the compiler cannot lower, and **dedicated sort / top-K units** service vector-search workloads that map poorly onto a SIMD tensor datapath. Runtime profiling of this path uses `bigTpuProfile --arch BM1690`.

---

## 3. On-chip Memory

| Level | Chip | Type | Managed by | Capacity |
|-------|------|------|-----------|----------|
| Local SRAM (LMEM) | **BM1690** | TPU scratchpad | Compiler (TPU-MLIR) | **256 KB/lane × 16 banks = 16 MB per core; 128 MB per 8-core chip** |
| **L2 SRAM** | **BM1690** | Chip-shared on-die SRAM | Compiler / DMA | **128 MB** (`L2_SRAM_SIZE = 0x8000000`); modeled max BW **128 GB/s** |
| L2 SRAM | BM1690E | Chip-shared on-die SRAM | Compiler / DMA | **16 MB** (`L2_SRAM_SIZE = 0x1000000`) |
| Local SRAM | BM1684X / BM1688 | TPU scratchpad | Compiler (TPU-MLIR) | Capacity not publicly disclosed; no HW cache in compute path |
| L1/L2/L3 | SG2042 | Standard CPU cache hierarchy | Hardware | 64KB/1MB/64MB per spec |

The absence of a hardware data cache in the TPU compute path (similar to Cambricon MLU, Tenstorrent Tensix) means that all data locality must be managed by the compiler. TPU-MLIR's SRAM tiling and DMA scheduling is therefore performance-critical.

*Added 2026-08-08.* **BM1690's 128 MB of chip-shared L2 SRAM is the largest single architectural change in the TPU line since BM1684X**, and it is the first Sophgo TPU with an explicit intermediate SRAM tier between DRAM and the per-core scratchpad. The 128 GB/s figure is the compiler's modeled L2 bandwidth, not a measured or datasheet number; it is notably *lower* than the derived aggregate DRAM bandwidth (~546 GB/s), which suggests it models a specific access class rather than peak L2 throughput. Do not cite it as peak on-chip bandwidth.

---

## 4. Off-chip Memory

| Chip | Type | Capacity | Bandwidth |
|------|------|----------|-----------|
| **BM1690** (per chip) | **Low-power DRAM — exact type not confirmed** (see note) | **~128 GB** | **~546 GB/s (derived)** |
| **SC11 FP300 card** (2 × BM1690) | same | **up to 256 GB** (vendor claim) | **>1.1 TB/s** (vendor claim) |
| BM1684X | LPDDR4X | 16 GB | ~68 GB/s |
| BM1688 | LPDDR4X | 8 GB | ~34 GB/s |
| SG2042 | DDR4-3200 × 4 ch | Up to 128+ GB (DIMM) | ~102 GB/s |
| SG2380 | LPDDR5 / custom | Up to 96 GB | ~150 GB/s (192-bit) |

**BM1690 memory notes (added 2026-08-08).**
- **Capacity** comes from Sophgo's own vllm-tpu configuration: `share-mem-start=0x1e0000000` (7.5 GiB) plus `share-mem-size=0x1e20000000` (~120.6 GiB) per chip → **~128 GB per BM1690**, consistent with the vendor's 256 GB per two-chip card.
- **Bandwidth** is **derived, not published**: TPU-MLIR's model gives a DDR clock of 8533 MT/s and 68.264 GB/s per core; × 8 cores = **~546 GB/s per chip**, consistent with the vendor's ">1.1 TB/s" per two-chip card.
- **Memory type is not confirmed.** LPDDR5X is asserted only by Baidu Baike (user-editable) and downstream copies of it. Sophgo's own filing says only "低功耗内存技术" (low-power memory technology) and "最佳带宽-容量-成本比例内存". LPDDR5X is *consistent* with an 8533 MT/s clock, but no retrievable Sophgo datasheet states it. **Do not print LPDDR5X unhedged.**

The BM1684X's LPDDR4X at ~68 GB/s is a significant constraint vs. HBM-equipped accelerators (Cambricon MLU290: 1,228 GB/s; Ascend 910B: ~1,200 GB/s). This limits the BM1684X to **compute-bound inference workloads** where the model fits in SRAM or where batch-level reuse amortizes DRAM access. BM1690 changes the shape of that tradeoff rather than removing it: it buys **capacity** (~128 GB/chip, 8× BM1684X) at DRAM-class bandwidth (~546 GB/s, still roughly half an HBM-equipped Chinese peer). The design point is large-model / KV-cache-resident serving where capacity per watt matters more than bytes per FLOP.

---

## 5. Host Interface / Package

| Product | Chips per card | Form Factor | Host Interface |
|---------|---------------|------------|---------------|
| BM1684X SoC (SE5/SE7) | 1 | Embedded SoC module | — (runs standalone Linux) |
| SC7PRO (八芯卡) | 8 × BM1684X | PCIe card | PCIe Gen3/4 x16 |
| SC7FP150 (六芯卡) | 6 × BM1684X | PCIe card | PCIe Gen3/4 x16 |
| SC7HP75 (三芯卡) | 3 × BM1684X | PCIe FHFL card | PCIe Gen3/4 x16 |
| SC7HP75_1 (单芯卡) | 1 × BM1684X | PCIe card | PCIe Gen3/4 x16 |
| **SC11 FP300** | **2 × BM1690** | **PCIe card, 300 W whole-card TDP** | **PCIe (generation/width not disclosed)** |
| **SC11E FP300** | **2 × BM1690E** | **PCIe card** | **not disclosed** |
| BM1690EVB | 1 × BM1690 | Evaluation board | — |
| SG6-10-B22 | 8 × BM1684X | 1U rackmount server | Multi-card chassis |

> **Correction (2026-08-08).** This table previously listed a BM1684X card called **"SC7 FP300"**. No such product exists. Sophgo's own MCU firmware board-type table (`sophgo/mcu`, `BoardType.md`) enumerates the BM1684X cards as SC7PRO / SC7FP150 / SC7HP75 / SC7HP75_1, and they are 8-, 6-, 3- and 1-chip cards respectively. The "FP300" suffix belongs only to the SC11 card built on BM1690. The same table maps board type `0xB2` → "Chip: BM1690, BM1690EVB" and `0xB3` → "Chip: BM1690, SC11".

---

## 6. Scale-up Interconnect

| Chip | Fabric | BW | Scale-up domain |
|------|--------|----|-----------------|
| **BM1690** | **SG-Link** (proprietary; multi-chip *and* multi-card) | **Not disclosed** (per-link BW, lane count and topology are all undisclosed) | **8 cards = 16 chips = up to 2 TB card memory in one 4U chassis** (confirmed); ~1000-card clusters over commodity Ethernet beyond that |
| BM1684X | None (PCIe P2P only) | PCIe Gen4 limited (~64 GB/s bidir) | Single chassis, data-parallel only |
| SG2042 | CCIX (2-chip scale-up) | Not disclosed | 2 sockets |

The BM1684X's lack of a proprietary chip-to-chip high-speed fabric is its most significant limitation for LLM inference at scale. Large models requiring tensor parallelism across chips must use PCIe peer-to-peer or host-mediated transfers, making BM1684X unsuitable for multi-chip model-parallel inference at high throughput.

### SG-Link (BM1690) — added 2026-08-08

**Superseding the prior claim that Sophgo has no proprietary scale-up fabric.** Sophgo's SC11 FP300 filing names **SG-Link** as a multi-chip and multi-card high-speed interconnect, and `sophgo/torch-tpu` documents a **chip-to-chip (C2C) topology** and ships a full `torch.distributed` collective backend against it (`python/dist_test2260/`), which is what makes tensor-parallel and ZeRO-sharded execution possible on this generation.

What is **confirmed**: the scale-up domain is **8 SC11 FP300 cards = 16 BM1690 chips = up to 2 TB of card memory in a single 4U chassis**. Beyond the chassis, board-level Ethernet expansion ports are stated to support clusters of roughly 1000 cards over commodity switches — i.e. Ethernet, not SG-Link, is the scale-out path.

What is **not disclosed**: SG-Link per-link bandwidth, lane count, signaling, and topology. No number should be printed for any of these.

### 128-chip "supernode" (WAIC 2025) — announced demonstration

At WAIC in Shanghai in late July 2025, Sophgo showed a **128 × BM1690 超节点** server described as 16 chips per layer across 8 layers, with Chinese media reporting "up to 8 TB" of memory and large FP8 compute. This is **not** the SG-Link scale-up domain and should not be presented as a shipping system:

- No product name, interconnect specification, or availability date was disclosed.
- The reported **8 TB does not reconcile** with ~128 GB/chip — 128 chips × 128 GB = 16 TB. Either the memory configuration differs, the number is a reporting error, or the layer count was conflated. **Treat 8 TB as unverified.**

---

## 7. Scale-out Interconnect

Standard GbE/10GbE/25GbE host networking on all Sophgo server products through the BM1684X generation. No proprietary RDMA or collective fabric on that generation; applications targeting distributed inference use standard TCP or RoCE-capable NICs through the host.

*Added 2026-08-08.* For **BM1690 / SC11 FP300**, scale-out is explicitly Ethernet-based: the card carries **board-level network expansion ports** stated to support clusters of roughly **1000 cards over commodity switches**. Port count and per-port rate are **not disclosed**. Sophgo has not announced a proprietary scale-out fabric.

---

## 8. Product Portfolio

| SKU | Chip | INT8 TOPS | Memory | Form Factor |
|-----|------|-----------|--------|-------------|
| **SC11 FP300** | **BM1690 × 2** | **>400 TOPS INT8/FP8 per card (vendor claim)** | **up to 256 GB (vendor claim)** | **PCIe card, 300 W** |
| **SC11E FP300** | **BM1690E × 2** | **not disclosed** | **not disclosed** | **PCIe card** |
| **BM1690EVB** | **BM1690 × 1** | **~200 TOPS (implied)** | **~128 GB (implied)** | **Evaluation board** |
| **4U SG-Link chassis** | **BM1690 × 16 (8 cards)** | **>3.2 POPS (8 × card claim)** | **up to 2 TB** | **4U server** |
| SE5 mini server | BM1684X × 1 | 32 | 16 GB | Desktop box |
| SE7-32 micro server | BM1684X × 1 | 32 | 16 GB | Ultra-compact |
| SC7PRO (八芯卡) | BM1684X × 8 | 32 per chip | 16 GB per chip | PCIe card |
| SC7FP150 (六芯卡) | BM1684X × 6 | 32 per chip | 16 GB per chip | PCIe card |
| SC7HP75 (三芯卡) | BM1684X × 3 | 32 per chip | 16 GB per chip | PCIe FHFL card |
| SC7HP75_1 (单芯卡) | BM1684X × 1 | 32 | 16 GB | PCIe card |
| SG6-10-B22 | BM1684X × 8 | 256 (8-card) | 128 GB | 1U server |
| BPI-SM9 module | BM1688 | 16 | 8 GB | Compute module |
| Milk-V Pioneer | SG2042 | — | Up to 128 GB DDR4 | ATX server board |
| SRA3-40 / SRB3-40 / SRM3-40 | SG2044 | — | Not disclosed | Compute / storage / converged servers |
| SG2380 board (OASIS) | SG2380 | 20 (AI accel.) | Up to 96 GB | Developer SBC |

**Catalog status (2026-08-08).** SC11 FP300 and BM1690 do **not** appear in Sophgo's public product catalog — the catalog API returns 38 SKUs, all BM1684X/BM1688/CV18xx/SG2042-class — although the SC11 FP300 product page URL remains search-indexed. This is consistent with restricted-channel sales. No shipment volume, revenue, or named datacenter deployment for BM1690 has been confirmed.

---

## Sources

- [SOPHON BM1684X product page](https://www.sophon.ai/product/introduce/bm1684x.html)
- [SOPHGO BM1684X IEEE paper](https://ieeexplore.ieee.org/document/10764438/)
- [SC7 HP75 product page](https://sophon.ai/product/introduce/sc7-hp75.html)
- [BM1688 AIMORELOGY spec page](https://aimorelogy.com/en/products/sophgo/bm1688/)
- [Banana Pi BPI-SM9 BM1688 module](https://www.electronics-lab.com/banana-pi-bpi-sm9-16-enc-a3-som-features-sophgo-bm1688-ai-soc/)
- [SG2042 processor spec (Milk-V Pioneer)](https://milkv.io/docs/pioneer/getting-started/processor)
- [SG2042 HPC paper (arXiv)](https://ui.adsabs.harvard.edu/abs/2023arXiv230900381B/abstract)
- [SG2380 CNX-Software](https://www.cnx-software.com/2023/10/21/sophgo-sg2380-16-core-sifive-p670-risc-v-processor-20-tops-ai-accelerator/)
- [SG2380 96GB update — Phoronix](https://www.phoronix.com/news/SG2380-RISC-V-SoC-Upgrade)
- [SiFive + Sophgo licensing](https://www.sifive.com/press/sophgo-licenses-sifive-risc-v-processor-cores-to-drive)

### Added 2026-08-08 — BM1690 / SG2260 (TPUv7), SC11 FP300, SG2044

Primary (Sophgo's own code/firmware):
- [sophgo/mcu — BoardType.md](https://github.com/sophgo/mcu/blob/master/BoardType.md) — `0xB2` = BM1690/BM1690EVB, `0xB3` = BM1690/SC11; BM1684X cards = SC7PRO / SC7FP150 / SC7HP75 / SC7HP75_1
- [sophgo/mcu — SC11FP300/common.h](https://github.com/sophgo/mcu/blob/master/SC11FP300/common.h) — `#define SOC_NUM 2` (two BM1690 per SC11 FP300 card)
- [tpu-mlir — include/tpu_mlir/Backend/BM168x/BM1690.h](https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Backend/BM168x/BM1690.h) — `NPU_NUM=64`, `EU_BYTES=64`, `LMEM_BYTES=256KB`, `LMEM_BANKS=16`, `L2_SRAM_SIZE=0x8000000`
- [tpu-mlir — python/profile_helper/bm1690_defs.py](https://github.com/sophgo/tpu-mlir/blob/master/python/profile_helper/bm1690_defs.py) — `"Chip Arch": "sg2260"`, Core Num 8, TIU/DMA 1000 MHz, DDR 8533 MT/s, DDR max BW 68.264 GB/s/core, L2 max BW 128 GB/s
- [sophgo/vllm-tpu](https://github.com/sophgo/vllm-tpu) — vLLM v0.11.0 fork for SG2260; tpuv7-driver/tpuv7-runtime 1.1.3; `sc11_config.ini` + `sc11_config_chip2.ini`; `share-mem-size=0x1e20000000`
- [sophgo/torch-tpu](https://github.com/sophgo/torch-tpu) — SG2260 PyTorch extension; C2C topology docs; `python/dist_test2260/` collectives

Vendor marketing / secondary:
- [SC11 FP300 innovation-product filing (China Security & Protection Industry Association, 2024-09-30)](https://xh.21csp.com.cn/cxcp_2024/202409/2056.html) — card-level 300 W / >400 TOPS / >1.1 TB/s / 256 GB / SG-Link / 4U 2 TB / ~1000-card Ethernet clusters / embedded RISC-V / sort-topK units
- [SCMP — Sophgo adapts compute card for DeepSeek (CTTL verification, 2025-06-30)](https://www.scmp.com/tech/tech-trends/article/3316363/chinese-chipmaker-sophgo-adapts-compute-card-deepseek-beijings-self-reliance-push)
- [WAIC 2025 — 128 × BM1690 超节点 (Chinese media)](https://finance.sina.com.cn/tech/2025-07-28/doc-infhywrc0366459.shtml)

Regulatory / other:
- [BIS Entity List additions, effective 2025-01-16 (Federal Register 2025-00480)](https://www.federalregister.gov/documents/2025/01/16/2025-00480/additions-to-the-entity-list)
- [SG2044 HPC evaluation — arXiv:2508.13840](https://arxiv.org/abs/2508.13840)
