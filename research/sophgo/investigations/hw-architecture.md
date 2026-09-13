# Sophgo Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*

> **2026-08-08:** a dated investigation section covering the BM1690 / SG2260 (TPUv7) datacenter generation, the SC11 FP300 card, SG-Link, and SG2044 is appended at the end of this file. The 2026-04-05 body below is retained unchanged except for one factual correction (the non-existent "SC7 FP300" card, §6).

---

## Overview

Sophgo (算能, formerly Bitmain's AI division, incorporated ~2021) produces two distinct silicon families:

1. **TPU line** — tensor processor units for AI inference: BM1684 (Gen 3), **BM1684X** (Gen 4, cloud/edge), **BM1688** / CV186X (Gen 4, edge/SoC)
2. **RISC-V CPU line** — high-core-count RISC-V processors for HPC and agentic compute: **SG2042** (64-core server), **SG2380** (16-core SoC with integrated 20 TOPS AI accelerator)

The BM1684X is Sophgo's primary datacenter-grade chip. The key architectural insight is that the **TPU is not a co-processor bolted onto a CPU** — it is the primary compute tile, with the ARM Cortex-A53 cluster serving as the SoC management CPU. This is distinct from, e.g., NVIDIA Jetson (GPU primary) or Qualcomm AI 100 (DSP primary).

---

## 1. Compute Engine — BM1684X TPU

### TPU Architecture

The BM1684X TPU uses a **SIMD architecture with very large data width** (Sophgo's terminology: "high data-width SIMD") to maximize compute density per unit area. This reduces instruction-unit overhead and achieves high compute density, particularly for convolution and matrix operations.

| Spec | Value |
|------|-------|
| Peak INT8 | 32 TOPS |
| Peak INT8 (Winograd conv) | 35.2 TOPS |
| Peak FP16 / BF16 | 16 TFLOPS |
| Peak FP32 | 2 TFLOPS |
| Generation | 4th-gen tensor processor |
| Process node | 12nm (Samsung) |
| Precision support | INT4, INT8, FP16, BF16, FP32 |

### ARM CPU Cluster (SoC Management)

| Spec | Value |
|------|-------|
| CPU | Octa-core ARM Cortex-A53 |
| Frequency | Up to 2.3 GHz |
| Role | SoC management, OS, host tasks |

### BM1688 / CV186X (Edge TPU SoC)

| Spec | Value |
|------|-------|
| Peak INT8 | 16 TOPS |
| Peak INT4 | 32 TOPS |
| Peak FP16 / BF16 | 4 TFLOPS |
| Peak FP32 | 0.5 TFLOPS |
| CPU | Octa-core ARM Cortex-A53, up to 1.6 GHz |
| Process | 12nm |
| Video decode | H.264/H.265: up to 16× 1080p @ 30 fps |
| Video encode | H.264/H.265: up to 10× 1080p @ 30 fps |

---

## 2. RISC-V CPU Line

### SG2042 (64-core RISC-V Server CPU)

| Spec | Value |
|------|-------|
| Core architecture | 64 × RISC-V (T-Head C920 OoO cores) |
| Organization | 16 clusters × 4 cores each |
| L1-D / L1-I per core | 64 KB / 64 KB |
| L2 per cluster | 1 MB shared |
| L3 system cache | 64 MB |
| Frequency | Up to 2 GHz |
| Memory controllers | 4 × DDR4-3200 (RDIMM/ECC/UDIMM) |
| PCIe | 32 × PCIe Gen4.0 |
| Process | TSMC 6nm |
| TDP | 120W |
| Connectivity | CCIX (2-chip scale-up), GbE, SDIO, SPI, I2C, UART |

The SG2042 delivers 5–10× better performance per core than prior widely available RISC-V hardware, though it still lags x86 server CPUs by 4–8× on multi-threaded workloads.

### SG2380 (16-core RISC-V SoC with AI)

| Spec | Value |
|------|-------|
| Core architecture | 12 × SiFive P670 performance + 4 × efficiency cores |
| Performance frequency | 2.5 GHz |
| Efficiency frequency | 1.6 GHz |
| AI accelerator | 20 TOPS integrated |
| GPU | Imagination GPU (3D graphics) |
| VPU | 4Kp60 H.265, H.264, AV1, VP9 |
| Memory | Up to 96 GB (192-bit interface) |
| Storage | UFS 3.2, SATA 3.0 |
| PCIe | Up to x16 |
| Ethernet | Up to 25 GbE |

SG2380 is Sophgo's hybrid architecture flagship: high-performance RISC-V CPU cores (licensed from SiFive) paired with an integrated AI accelerator — a closer analog to Apple M-series than a pure TPU chip.

---

## 3. Data Path

### BM1684X TPU Pipeline

The BM1684X TPU data path is organized around a **compiler-scheduled command-driven pipeline**. There is no GPU-style speculative execution or out-of-order issue in the TPU core. The model compiler (TPU-MLIR) statically schedules all tensor operations, producing a `.bmodel` binary that encodes a fixed command sequence executed by the TPU command processor.

Key data path characteristics:
- **DMA engine**: Dedicated engine for DDR ↔ local SRAM data movement, overlapping with compute
- **BMCV hardware accelerator**: Separate video processing pipeline (VPP) for image preprocessing (resize, crop, color convert, JPEG decode) with direct output to TPU-accessible memory
- **Video codec**: Dedicated H.264/H.265 encode/decode hardware units for computer vision pipelines

### BM1684X Memory Data Path

```
DDR memory
    ↕ DMA engine (BMLib bmlib_mem_alloc / bm_mem_get_device_addr)
Local SRAM (TPU on-chip)
    ↕ TPU compute pipeline
Local SRAM (output)
    ↕ DMA engine
DDR memory (result)
```

---

## 4. On-chip Memory

The BM1684X on-chip memory architecture is not fully disclosed by Sophgo. Based on the SIMD-wideness design philosophy and the bmodel binary format:

- **Local SRAM**: Scratchpad memory for TPU computation; size undisclosed in public documentation
- **NRAM / SRAM**: Compiler-managed (no hardware cache in the primary compute path)
- **No hardware cache**: TPU compute path relies on compiler-managed DMA to prefetch data into on-chip SRAM before compute begins

---

## 5. Off-chip Memory

### BM1684X

| Spec | Value |
|------|-------|
| Memory type | LPDDR4X |
| Capacity | 16 GB |
| Bandwidth | ~68 GB/s |
| Host interface | PCIe Gen3/Gen4 x16 (SoC and PCIe card variants) |

### SC7 HP75-I (PCIe Accelerator Card)

The SC7 HP75-I is Sophgo's cloud inference PCIe card based on BM1684X. Supports 96-channel HD video decoding and 96-channel HD video analysis concurrently.

| Spec | Value |
|------|-------|
| Chip | BM1684X |
| Peak INT8 | 32 TOPS |
| Form factor | PCIe FHFL |
| Interface | PCIe Gen3/4 x16 |

---

## 6. Host Interface / Package

| Spec | BM1684X |
|------|---------|
| Host PCIe | Gen3/4 x16 |
| SoC variants | Standalone SoC (SE5, SE7 mini servers) |
| Card variants | SC7PRO (8 chips), SC7FP150 (6 chips), SC7HP75 (3 chips), SC7HP75_1 (1 chip) |
| Server | SG6-10-B22 intelligent server |

> **Correction 2026-08-08.** This row previously read "SC7 FP300, SC7 HP75-I (PCIe FHFL)". **No BM1684X card named "SC7 FP300" exists.** Sophgo's own MCU firmware board-type table (`sophgo/mcu`, `BoardType.md`) enumerates the BM1684X cards as SC7PRO (八芯卡), SC7FP150 (六芯卡), SC7HP75 (三芯卡) and SC7HP75_1 (单芯卡) — 8-, 6-, 3- and 1-chip cards respectively. The "FP300" suffix belongs only to the **SC11** card, which is built on BM1690. Because the SC-series is multi-chip-per-card throughout, per-card TOPS and memory are the per-chip figures times the chip count.

---

## 7. Scale-up Interconnect

The BM1684X does not have a proprietary chip-to-chip high-speed interconnect equivalent to NVLink, MLU-Link, or HCCS. Multi-chip configurations rely on:
- **PCIe peer-to-peer** within a server
- No dedicated high-bandwidth scale-up fabric

This is a key architectural limitation vs. GPU-class accelerators for training workloads. Sophgo's primary positioning is inference, where single-chip or multi-card (PCIe) configurations suffice.

---

## 8. Scale-out Interconnect

Standard host networking (1/10/25 GbE on SE/server products). No proprietary RDMA or collective communication fabric.

---

## Sources

- [SOPHON BM1684X product page](https://www.sophon.ai/product/introduce/bm1684x.html)
- [SOPHGO BM1684X IEEE paper (IEEEXplore)](https://ieeexplore.ieee.org/document/10764438/)
- [SG2042 processor documentation (Milk-V Pioneer)](https://milkv.io/docs/pioneer/getting-started/processor)
- [SG2380 CNX-Software article](https://www.cnx-software.com/2023/10/21/sophgo-sg2380-16-core-sifive-p670-risc-v-processor-20-tops-ai-accelerator/)
- [SiFive + Sophgo licensing announcement](https://www.sifive.com/press/sophgo-licenses-sifive-risc-v-processor-cores-to-drive)
- [SG2042 HPC evaluation paper (arXiv)](https://ui.adsabs.harvard.edu/abs/2023arXiv230900381B/abstract)
- [SG2380 96GB RAM update — Phoronix](https://www.phoronix.com/news/SG2380-RISC-V-SoC-Upgrade)
- [SC7 HP75-I product page](https://sophon.ai/product/introduce/sc7-hp75.html)

---

# Investigation 2026-08-08 — BM1690 / SG2260 "TPUv7", SC11 FP300, SG-Link, SG2044

*Investigated 2026-08-08. Scope: the datacenter-class Sophgo generation that the 2026-04-05 investigation missed entirely, plus one pre-existing factual error (§6 above).*

## Method and evidence quality

Sophgo publishes no BM1690 datasheet, and the chip is absent from its public product catalog. The investigation therefore leaned on three tiers of evidence, which must be kept distinct when citing:

| Tier | What it is | How to cite |
|---|---|---|
| **A — Sophgo's own source code and firmware** | `sophgo/mcu` (MCU/board firmware), `sophgo/tpu-mlir` (compiler backend + profiler machine model), `sophgo/vllm-tpu` (driver configs), `sophgo/torch-tpu` | Strongest available. Structural facts (core counts, SRAM sizes, chips-per-card) are effectively confirmed. But the compiler's machine model is **not a datasheet** — do not back-derive peak TOPS from it |
| **B — vendor-authored filing republished by a third party** | China Security & Protection Industry Association innovation-product filing, 2024-09-30, filed by 厦门算能科技有限公司 | All performance numbers here are **vendor marketing**. Label them as such. No independent benchmark exists |
| **C — Chinese media / user-editable encyclopedias** | Sina/x-techcon WAIC coverage; Baidu Baike | Use only for existence-of-event claims. Never for specs. The single LPDDR5X assertion lives only here |

Independent, non-Sophgo-sourced corroboration is thin: essentially only SCMP/Reuters-tier coverage of the CTTL DeepSeek-R1 result repeats the 256 GB / 1.1 TB/s card figures.

## 1. Naming — BM1690 = SG2260 = "TPUv7"

Three names for one thing, which is why the part reads as three products in secondary coverage:

- **BM1690** — product/board name. `sophgo/mcu` `BoardType.md` maps `0xB2` → "Chip: BM1690, BM1690EVB" and `0xB3` → "Chip: BM1690, SC11".
- **SG2260** — silicon/architecture name. TPU-MLIR's profiler literally sets `"Chip Arch": "sg2260"` for BM1690 (`python/profile_helper/bm1690_defs.py`). `torch-tpu` and `vllm-tpu` both address the part as SG2260.
- **TPUv7** — software-generation brand. Runtime packages are `tpuv7-driver` / `tpuv7-runtime`; firmware lives in `/lib/firmware/tpuv7/`; the compiler ships `third_party/nntoolchain/tpuv7_sha256.txt`.

A cut-down **BM1690E** sibling exists (`BM1690E.h`: `L2_SRAM_SIZE = 0x1000000` = 16 MB, vs 128 MB) with a RISC-V kernel-module variant; its board is **SC11E FP300**. Recommendation adopted in the survey: use **BM1690** as canonical, cross-reference SG2260 and TPUv7.

## 2. Chip microarchitecture (Tier A)

From `include/tpu_mlir/Backend/BM168x/BM1690.h` and `python/profile_helper/bm1690_defs.py`:

| Parameter | Value | Symbol |
|---|---|---|
| TPU cores per chip | 8 | `Core Num` |
| NPU lanes per core | 64 | `NPU_NUM = 64` |
| EU vector width | 512 bit (64 byte) | `EU_BYTES = 64` |
| LMEM per lane | 256 KB | `LMEM_BYTES` |
| LMEM banks | 16 | `LMEM_BANKS` |
| LMEM per core | 16 MB (`TPU Lmem` = 16,777,216 B) | — |
| L2 SRAM | 128 MB | `L2_SRAM_SIZE = 0x8000000` |
| Modeled TIU / DMA clock | 1000 MHz | `TIU/DMA Frequency` |
| Modeled DDR clock | 8533 MT/s | `DDR Frequency` |
| Modeled DDR BW | 68.264 GB/s **per core** | `DDR Max BW` |
| Modeled L2 BW | 128 GB/s | `L2 Max BW` |
| Kernel module | `libbm1690_kernel_module.so` | — |

**Native FP8 is confirmed structurally**, not just claimed: `test/TDB/bmodel/resnet50_v2_bm1690_f8e5m2.bmodel` exists as a checked-in regression artifact, alongside `resnet50_v2_bm1690_f16_core8.bmodel` which confirms 8-core codegen.

Two blocks with no BM1684X analog appear in the Tier-B filing: an **embedded RISC-V core** used as a fallback for long-tail operators, and **dedicated sort / top-K hardware units** targeting billion-scale vector search. Both are plausible given the compiler's operator coverage gaps, but neither is visible in the open compiler backend.

**Process node and foundry are not disclosed by any source retrieved.** Do not print a node. Sophgo was added to the **US BIS Entity List effective 2025-01-16** (Federal Register 2025-00480) over TSMC dies allegedly routed to Huawei, which makes BM1690's foundry supply a live open question rather than a detail.

## 3. The card/chip conflation — the single biggest trap

**SC11 FP300 carries two BM1690 chips.** Evidence (Tier A, three independent artifacts):

1. `sophgo/mcu` `SC11FP300/common.h`: `#define SOC_NUM 2`, plus `BM1690_I2C0_IRQ` / `BM1690_I2C_IRQ`.
2. `SC11FP300/chip.c` and `pin.h`: separate `BM0_` and `BM1_` reset GPIOs.
3. `vllm-tpu` driver install emits **two** firmware config files: `sc11_config.ini` and `sc11_config_chip2.ini`.

Therefore every headline SC11 FP300 number is per card, not per chip:

| Metric | Per card (Tier B, vendor claim) | Per BM1690 chip |
|---|---|---|
| INT8 / FP8 | >400 TOPS | ~200 TOPS |
| FP16 / BF16 | >200 TFLOPS | ~100 TFLOPS |
| TF32 | >100 TFLOPS | ~50 TFLOPS |
| FP32 | >25 TFLOPS | ~12.5 TFLOPS |
| Memory | up to 256 GB | ~128 GB |
| Bandwidth | >1.1 TB/s | ~546 GB/s |
| Power | 300 W (整卡功耗, whole-card TDP) | ~150 W |
| Data types | INT4/INT8/FP8/NF4/TF32/FP16/BF16/FP32 | same |

Note also that "<300 W" as circulated is imprecise — the filing says 整卡功耗 300 W / 控制在300W以内, i.e. a **300 W whole-card TDP**.

The per-chip figures cross-check against Tier A independently, which is what gives confidence in the halving:

- **Capacity:** `vllm-tpu` per-chip DDR carve-out `share-mem-start = 0x1e0000000` (7.5 GiB) + `share-mem-size = 0x1e20000000` (~120.6 GiB) → **~128 GB per chip**, ×2 = 256 GB per card. ✓
- **Bandwidth:** 68.264 GB/s per core × 8 cores = **~546 GB/s per chip** (derived), ×2 = ~1.09 TB/s per card, matching ">1.1 TB/s". ✓

## 4. Off-chip memory type — deliberately left unresolved

**LPDDR5X is not confirmed.** It is asserted by Baidu Baike (user-editable, and the page 403s to direct fetch — reached only via search snippet) and by downstream copies of that page. Sophgo's own filing says only 低功耗内存技术 ("low-power memory technology") and 最佳带宽-容量-成本比例内存. An 8533 MT/s DDR clock is *consistent* with LPDDR5X, but consistency is not confirmation and no retrievable Sophgo datasheet states it. Written into the survey as "low-power DRAM — exact type not confirmed", with the LPDDR5X claim and its weak provenance recorded rather than adopted.

## 5. SG-Link — the fabric that refutes the repo's standing claim

The 2026-04-05 investigation concluded "no proprietary chip-to-chip high-speed interconnect". That is correct for BM1684X and **wrong for BM1690**.

- The Tier-B filing names **SG-Link** as a 多卡高速互联 fabric spanning both multi-chip and multi-card.
- `sophgo/torch-tpu` ships `docs/quick_start/assets/3_c2c_topology.png` and `docs/developer_manual/source_zh/06_distribute.rst`, i.e. a documented chip-to-chip topology, plus a working collective suite on top of it (see the software-stack investigation).

**Confirmed scale-up domain: 8 cards = 16 BM1690 chips = up to 2 TB of card memory in one 4U chassis** (单台4U最大2TB显存). Beyond the chassis the filing describes 网络扩展端口 supporting 千卡集群 — roughly 1000-card clusters over commodity Ethernet switches. So Ethernet, not SG-Link, is the scale-out path.

**Not disclosed anywhere retrieved: SG-Link per-link bandwidth, lane count, signaling, and topology.** No number should be printed for any of these.

## 6. The 128-chip "supernode" — demoted to announced demonstration

Chinese coverage of WAIC (Shanghai, late July 2025) describes a Sophgo **超节点** server of **128 BM1690 chips, 16 per layer × 8 layers**, with "up to 8 TB 显存" and large FP8 compute.

Problems that keep this out of the spec tables:

- **The memory figure does not reconcile.** 8 TB / 128 chips = 64 GB/chip, against ~128 GB/chip established from Tier A. The expected total is 16 TB. Either the config differs, the reporter erred, or the layer count was conflated.
- No product name, interconnect specification, or availability date was disclosed.
- It is **not** the SG-Link scale-up domain, which is 16 chips. Presenting 128 as "Sophgo's scale-up scale" would be wrong.

Recorded as an announced demonstration with an unresolved memory figure.

## 7. Status verbs

Announced and lab-certified; **not** confirmed deployed at scale.

| Date | Event | Tier |
|---|---|---|
| Sept 2024 | SC11 FP300 launched at 2024 中国算力大会; top innovation award at 2024 Beijing Security Expo | B |
| 2025-01-16 | Sophgo added to US BIS Entity List (FR 2025-00480) | A (primary regulatory) |
| 2025-06-30 | SC11 FP300 passed DeepSeek-R1 (671B) adaptation verification at CTTL (under CAICT/MIIT) | independent (SCMP et al.) |
| late Jul 2025 | 128 × BM1690 supernode shown at WAIC | C |
| 2026-01-06 | CSRC filing update consistent with a STAR Market listing process, per Chinese reporting. Filing itself not retrievable; circulating valuation unverified → **"filing reported, outcome not confirmed"** | C |
| 2026-08-08 | Sophgo's catalog API (`POST getProductList`) returns 38 SKUs, **none** of them SC11 FP300, BM1690, or any SC7 card; the SC11 FP300 product URL stays search-indexed → consistent with restricted-channel sales | A |

No shipment volume, revenue, or named datacenter deployment was found. No MLPerf submission by Sophgo is confirmed. No Hot Chips / ISCA / ISSCC 2026 Sophgo presentation is confirmed. No BM1690 successor announcement in 2026 is confirmed.

## 8. SG2044 — lower-confidence addendum

Reported by the 2026-08-08 raw scan; **not independently re-verified in the adversarial pass**, so held below the BM1690 material.

- Second-generation 64-core RISC-V server CPU, successor to SG2042.
- Adds **RVV 1.0** vector support and an improved memory subsystem.
- Platform support merged into **mainline Linux 6.16**.
- Productized as **SRA3-40** (compute), **SRB3-40** (storage), **SRM3-40** (converged) servers.
- First comprehensive HPC evaluation: **arXiv:2508.13840** (Aug 2025), vs SG2042 plus x86 and Arm server parts.
- Frequency, process node, cache sizes and TDP: **not disclosed** here.

## 9. Open items

- SG-Link per-link bandwidth, lane count, topology — not disclosed.
- BM1690 process node and foundry — not disclosed; materially uncertain given the Entity List status.
- BM1690 off-chip DRAM type — LPDDR5X unconfirmed.
- Per-chip peak TOPS/TFLOPS — only derivable by halving vendor card claims; no primary per-chip figure exists.
- The WAIC supernode's 8 TB memory figure — unresolved contradiction.
- TPU-MLIR carries backends for **SGTPUV8**, **SG2380** and **BM1684X2**; none of these parts is documented in the survey. Worth a separate investigation.
- STAR Market listing outcome — filing reported, not confirmed.

## Sources (2026-08-08)

Tier A — Sophgo's own code/firmware:
- https://github.com/sophgo/mcu/blob/master/BoardType.md
- https://github.com/sophgo/mcu/blob/master/SC11FP300/common.h
- https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Backend/BM168x/BM1690.h
- https://github.com/sophgo/tpu-mlir/blob/master/python/profile_helper/bm1690_defs.py
- https://github.com/sophgo/vllm-tpu
- https://github.com/sophgo/torch-tpu
- https://www.sophgo.com/basic-api/business/product/getProductList (POST {"pageNo":1,"pageSize":300})

Tier B — vendor-authored filing:
- https://xh.21csp.com.cn/cxcp_2024/202409/2056.html

Independent / regulatory:
- https://www.scmp.com/tech/tech-trends/article/3316363/chinese-chipmaker-sophgo-adapts-compute-card-deepseek-beijings-self-reliance-push
- https://www.chinastrategy.org/2025/06/30/chinas-sophgo-adapts-chip-product-for-deepseek-in-self-reliance-push/
- https://www.federalregister.gov/documents/2025/01/16/2025-00480/additions-to-the-entity-list
- https://arxiv.org/abs/2508.13840

Tier C — media / weak:
- https://finance.sina.com.cn/tech/2025-07-28/doc-infhywrc0366459.shtml
- https://www.x-techcon.com/article/23851.html
- https://baike.baidu.com/item/SC11%20FP300/67686300 (user-editable; only source naming LPDDR5X)
