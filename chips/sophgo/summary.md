# Sophgo (算能) Software and Hardware Stack Summary

*as_of: 2026-09-13*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*

---

## Overview

**Sophgo** (算能科技, formerly Bitmain's AI division, incorporated ~2021) is a Chinese AI chip company producing two distinct silicon families: a **TPU line** for AI inference (BM1684/BM1684X/BM1688, and from the TPUv7 generation **BM1690**) and a **RISC-V CPU line** for general compute and edge AI (SG2042, SG2044, SG2380). The company's Bitmain heritage gives it deep semiconductor manufacturing knowledge. Through the BM1684X generation its technology direction was firmly AI inference for the domestic Chinese market rather than large-scale training; the BM1690 / SC11 FP300 generation changes that framing — see the **BM1690 / SC11 FP300 Update (2026-08-08)** section below.

The current flagship datacenter part is the **BM1690** (silicon name **SG2260**; the runtime/software generation is branded **TPUv7**), shipped on the two-chip **SC11 FP300** PCIe card. The previous-generation **BM1684X** is a fourth-generation tensor processor delivering 32 TOPS INT8 at 12nm; it is sold as both an embedded SoC (in SE5/SE7 mini servers) and as PCIe accelerator cards. Per Sophgo's own MCU firmware board table the BM1684X PCIe cards are **SC7PRO** (八芯卡, 8 chips), **SC7FP150** (六芯卡, 6 chips), **SC7HP75** (三芯卡, 3 chips) and **SC7HP75_1** (单芯卡, 1 chip) — the SC-series is multi-chip-per-card, so per-card TOPS and memory are the per-chip figures multiplied by the chip count. The edge variant, **BM1688**, targets drone vision, industrial inspection, and smart cameras at 16 TOPS INT8 in a low-power SoC form factor.

> **Correction (2026-08-08).** Earlier revisions of this page listed a BM1684X PCIe card called **"SC7 FP300"**. No such product exists in Sophgo's own board-type table; the name conflated the SC7 (BM1684X) family with the SC11 **FP300** (BM1690) card. All "SC7 FP300" references have been removed or replaced. The related 32 TOPS / 16 GB figures previously attached to SC7 card rows were per-chip figures presented as per-card.

Sophgo's most significant open-source contribution is **TPU-MLIR** (`sophgo/tpu-mlir`), a fully open MLIR-based compilation pipeline that converts ONNX/Caffe/TFLite/PyTorch models to `.bmodel` hardware binaries. This makes Sophgo's compiler stack more transparent than most Chinese AI chip vendors, enabling third-party tool integration and academic study.

The RISC-V line is architecturally separate: **SG2042** (64-core T-Head C920 server CPU), its successor **SG2044**, and **SG2380** (16-core SiFive P670 SoC with integrated 20 TOPS AI accelerator) target HPC, embedded Linux, and developer platforms (Milk-V Pioneer). These chips compete on RISC-V ecosystem growth rather than AI inference density.

---

## Software Stack

### Framework Integration

Sophgo's LLM inference stack is organized around three repos:

- **LLM-TPU** (`sophgo/LLM-TPU`): Deploys generative AI models (Qwen2, LLaMA-series, HuggingFace LLMs) on BM1684X and BM1688. Uses TPU-MLIR for compilation and `tpu-runtime` for inference. Supports both PCIE mode (server-attached card) and SoC mode (embedded BM1688).
- **vllm-tpu** (`sophgo/vllm-tpu`): Sophgo's fork of vLLM for production LLM serving — continuous batching, KV-cache management, OpenAI-compatible API.
- **LLM-TPU_Lite**: Lightweight variant for edge BM1688 where full vllm overhead is unacceptable.

For traditional CV/inference workloads, users interact through:
- **sophon-sail**: Python/C++ high-level wrapper around BMRuntime, BMCV, and BMLib. The `sail.Engine` class loads and runs `.bmodel` files; `sail.Decoder` handles video; `sail.BMImage` manages tensors.
- **SOPHON-MW**: Hardware-accelerated OpenCV (BM-OpenCV) and FFmpeg (BM-FFmpeg) forks that route H.264/H.265 encode/decode through the chip's JPU/VPP hardware.

Framework model support flows through TPU-MLIR's conversion utilities: PyTorch, TensorFlow, PaddlePaddle, MXNet, and Caffe all target ONNX as an intermediate step before entering the MLIR pipeline. ONNX, TFLite, and Caffe are natively supported without an intermediate conversion.

### Compiler / IR — TPU-MLIR

TPU-MLIR is the crown jewel of Sophgo's open-source ecosystem. The compilation pipeline is two-stage:

1. **`model_transform.py`**: Converts a framework model to a `.mlir` file in the **Top dialect** — a framework-neutral intermediate representation of the network graph.
2. **`model_deploy.py`**: Lowers the Top dialect to the **TPU dialect** (chip-specific ops with explicit DMA scheduling and SRAM tiling), runs quantization (INT8/INT4) using a user-provided calibration table, and produces the `.bmodel` binary.

The `.bmodel` is Sophgo's hardware binary format — analogous to NVIDIA's TensorRT `.plan` or Cambricon's `.cnbin`. It encodes compiled tensor operations, DMA command sequences, quantized weights, and target chip metadata. Users never write ISA-level code directly.

The earlier `sophgo/tpu_compiler` (CVITEK AI compiler) handles older CV18XX series chips and is now superseded by TPU-MLIR for current hardware.

**Update (2026-09-13).** TPU-MLIR advanced from **v1.29** (2026-07-01) to **v1.30.2** (2026-08-31, verified via the GitHub API). New in v1.30.2: LLM **chunk prefill** and **chunked decode** (incl. a Qwen3.5-specific path), new multimodal model support (**MiniCPM-V-4.6, Step3-VL, Falcon-Perception, LocateAnything-3B**), a further-updated **BM1690/BM1690E backend**, continued **BM1684X2** enablement, and merged **CUDA op support**. Software-only — no new BM1690 spec value. Two repos in the BM1690 LLM-serving/training path — `vllm-tpu` and `torch-tpu` — have **not been pushed since before the 2026-04-05 baseline** (last push 2025-12-17 and 2026-01-28 respectively); flagged as a stalled area. See `research/sophgo/investigations/software-stack.md` → "Update — 2026-09-13".

### Runtime — libsophon

The runtime stack is packaged as **libsophon** (`sophgo/libsophon`), a unified package containing:

- **BMRuntime (BMRT / `libbmrt.so`)**: Loads `.bmodel` files and executes them on the TPU. The primary inference C API: `bmrt_load_bmodel`, `bmrt_launch_tensor`, `bmrt_memcpy_s2d`/`_d2s` for host↔device data movement.
- **BMLib (`libbmlib.so`)**: Low-level device and memory management. `bm_dev_request` opens a device handle; `bm_malloc_device_byte` allocates device memory; `bm_memcpy_s2d`/`d2s` are the DMA primitives.
- **BMCV (`libbmcv.so`)**: Hardware-accelerated CV preprocessing using the BM1684X's VPP (Video Processing Pipeline) and JPU (JPEG processing unit). Covers resize, color space conversion, affine transform, JPEG encode/decode, normalization, NMS, and sorting.

### Driver / Firmware

The kernel driver (`sophon.ko` or `bmsophon.ko`) provides PCIe BAR mapping, IOCTL dispatch, DMA engine control, and interrupt handling. Sophgo provides official Docker images (`sophgo/tpuc_dev`) and Kubernetes device plugins for cloud deployment.

### Communication

**Superseded 2026-08-08 — corrected below.** For the **BM1684X generation** the statement holds: libsophon ships no collective communication library, and multi-card configurations rely on application-level data parallelism (e.g., routing independent video streams to different cards over PCIe) rather than model-parallel execution with AllReduce.

For the **BM1690 / SG2260 (TPUv7) generation** it is wrong. Sophgo's `torch-tpu` PyTorch extension ships a full `torch.distributed` collective backend for SG2260 (`python/dist_test2260/`: `all_reduce`, `all_gather`, `all_gather_into_tensor`, `all_to_all`, `broadcast`, `reduce`, `gather`, `scatter`, point-to-point, and DDP), documents a **chip-to-chip (C2C) topology**, and supports DeepSpeed ZeRO-1/ZeRO-2 with CPU offload plus Megatron-DeepSpeed tensor parallelism. The underlying fabric is **SG-Link**. See the BM1690 update section.

---

## Hardware Architecture

### BM1684X TPU

The BM1684X uses a **high data-width SIMD architecture** for the TPU core — Sophgo's terminology for a wide vector/tensor unit that minimizes instruction overhead per multiply-accumulate operation. The compiler statically schedules all tensor operations; there is no runtime speculative execution. Key specs: 32 TOPS INT8 (35.2 TOPS with Winograd conv acceleration), 16 TFLOPS FP16/BF16, 2 TFLOPS FP32, in a 12nm process.

On-chip SRAM is compiler-managed (no hardware data cache in the primary compute path). The chip includes a dedicated DMA engine that prefetches weights and activations from LPDDR4X (16 GB, ~68 GB/s) into on-chip SRAM before each compute phase, enabling double-buffered compute-DMA overlap when the compiler schedules it.

The BM1684X SoC packages an octa-core ARM Cortex-A53 at 2.3 GHz for SoC management alongside the TPU. The ARM CPU runs Linux and handles non-compute tasks; all AI inference runs on the TPU core.

### RISC-V Line

**SG2042**: 64-core server CPU (16 clusters × 4 T-Head C920 OoO RISC-V cores), TSMC 6nm, 2 GHz, 64 MB L3 cache, 4 × DDR4-3200 controllers, 32 × PCIe Gen4.0, 120W TDP. CCIX allows 2-chip scale-up. HPC evaluation shows 5–10× better per-core performance than prior RISC-V hardware, but 4–8× behind x86 server CPUs on multi-threaded workloads.

**SG2380**: 16-core SoC using SiFive P670 cores (licensed), 2.5 GHz peak, integrated 20 TOPS AI accelerator, Imagination GPU, up to 96 GB RAM (192-bit interface), PCIe x16, 25 GbE — targeting developer platforms and edge AI with a more complete SoC feature set.

### Product Line

| Product | Chip | Target | INT8 TOPS |
|---------|------|--------|-----------|
| SE5 mini server | BM1684X × 1 | Cloud edge inference | 32 |
| SE7 micro server | BM1684X × 1 | Cloud edge inference | 32 |
| SC7PRO (八芯卡) | BM1684X × 8 | PCIe inference card | 32 per chip |
| SC7FP150 (六芯卡) | BM1684X × 6 | PCIe inference card | 32 per chip |
| SC7HP75 (三芯卡) | BM1684X × 3 | PCIe inference card (96ch video) | 32 per chip |
| SC7HP75_1 (单芯卡) | BM1684X × 1 | PCIe inference card | 32 |
| SG6-10-B22 server | BM1684X × N | Multi-card inference server | 32N |
| **SC11 FP300** | **BM1690 × 2** | **Datacenter AI card (PCIe)** | **>400 TOPS INT8/FP8 per card (vendor claim)** |
| **SC11E FP300** | **BM1690E × 2** | **Datacenter AI card, cut-down L2 variant** | **not disclosed** |
| Milk-V Pioneer | SG2042 | RISC-V developer server | — |
| SRA3-40 / SRB3-40 / SRM3-40 | SG2044 | RISC-V compute / storage / converged servers | — |
| SG2380 boards | SG2380 | RISC-V edge AI SoC | 20 (integrated) |

Card names and chip counts for the SC7 and SC11 families are from Sophgo's own MCU firmware board-type table (`sophgo/mcu`, `BoardType.md`), which maps board type `0xB2` → "Chip: BM1690, BM1690EVB" and `0xB3` → "Chip: BM1690, SC11".

---

## BM1690 / SC11 FP300 Update (2026-08-08)

*Updated 2026-08-08. This section adds Sophgo's datacenter-class TPUv7 generation, which the survey previously omitted entirely. Primary evidence is Sophgo's own open-source firmware and compiler (`sophgo/mcu`, `sophgo/tpu-mlir`, `sophgo/vllm-tpu`, `sophgo/torch-tpu`); the headline performance numbers come from a vendor-authored innovation-product filing and are labelled as vendor claims. Prior-generation BM1684X/BM1688 content above is retained unchanged.*

### Naming — one chip, three names

| Name | Where it comes from |
|---|---|
| **BM1690** | Product/board name. Sophgo MCU firmware `BoardType.md`: `0xB2` = BM1690/BM1690EVB, `0xB3` = BM1690/SC11 |
| **SG2260** | Silicon/architecture name. TPU-MLIR's profiler sets `"Chip Arch": "sg2260"` for BM1690 (`python/profile_helper/bm1690_defs.py`) |
| **TPUv7** | Software-generation brand. Runtime packages are `tpuv7-driver` / `tpuv7-runtime`; firmware lives in `/lib/firmware/tpuv7/` |

A cut-down sibling, **BM1690E**, exists with 16 MB of L2 SRAM instead of 128 MB (`BM1690E.h`, `L2_SRAM_SIZE = 0x1000000`) and a RISC-V kernel-module variant; its board is **SC11E FP300**. This survey uses **BM1690** as the canonical name.

### Chip-level microarchitecture (from Sophgo's own compiler backend)

The following are read directly out of TPU-MLIR (`include/tpu_mlir/Backend/BM168x/BM1690.h`, `python/profile_helper/bm1690_defs.py`). They are the **compiler's machine model**, not a datasheet — peak TOPS must not be back-derived from them.

| Parameter | Value |
|---|---|
| TPU cores per chip | **8** |
| NPU lanes per core | **64** |
| EU vector width | **512 bit (64 byte)** |
| Local memory (LMEM) | **256 KB per lane × 16 banks = 16 MB per core** → 128 MB across 8 cores |
| On-chip L2 SRAM | **128 MB** (`L2_SRAM_SIZE = 0x8000000`); **16 MB** on BM1690E |
| Modeled TIU / DMA clock | 1000 MHz |
| Modeled L2 max bandwidth | 128 GB/s |
| Modeled DDR clock | 8533 MT/s; 68.264 GB/s per core → **~546 GB/s per 8-core chip (derived)** |
| Off-chip capacity per chip | **~128 GB** (vllm-tpu `share-mem-start=0x1e0000000` + `share-mem-size=0x1e20000000`) |
| Memory type | **Not confirmed.** LPDDR5X is asserted only by Baidu Baike (user-editable); Sophgo's own filing says only "低功耗内存技术" (low-power memory). Consistent with the 8533 MT/s model clock, but not on any retrievable Sophgo datasheet |
| Process node / foundry | **Not disclosed** by any retrievable source |

**Native FP8.** TPU-MLIR's regression artifacts include `resnet50_v2_bm1690_f16_core8.bmodel` and `resnet50_v2_bm1690_f8e5m2.bmodel`, independently confirming 8-core codegen and native **FP8 E5M2** support.

### Card level — SC11 FP300 carries TWO BM1690 chips

This is the single most important correction to make when reading Sophgo's marketing: **every headline SC11 FP300 number is per card, and a card holds two BM1690 chips.** Sophgo's MCU firmware for the board defines `#define SOC_NUM 2` with separate `BM0_`/`BM1_` reset GPIOs (`sophgo/mcu`, `SC11FP300/common.h`, `chip.c`, `pin.h`), and the vLLM driver install emits two config files, `sc11_config.ini` and `sc11_config_chip2.ini`.

| Metric | Per SC11 FP300 card (vendor claim) | Implied per BM1690 chip |
|---|---|---|
| INT8 / FP8 | **>400 TOPS** | ~200 TOPS |
| FP16 / BF16 | **>200 TFLOPS** | ~100 TFLOPS |
| TF32 | **>100 TFLOPS** | ~50 TFLOPS |
| FP32 | **>25 TFLOPS** | ~12.5 TFLOPS |
| Memory capacity | **up to 256 GB** | ~128 GB |
| Memory bandwidth | **>1.1 TB/s** | ~546 GB/s (matches the compiler model) |
| Power | **300 W whole-card TDP** (整卡功耗 300 W) | ~150 W |
| Data types | INT4 / INT8 / FP8 / NF4 / TF32 / FP16 / BF16 / FP32 | same |

Source for the card table: the China Security & Protection Industry Association 2024 innovation-product filing dated 2024-09-30, filed by 厦门算能科技有限公司 (Xiamen Sophgo). It is a **vendor-supplied document republished by an industry association — treat all performance figures as vendor marketing.** No third-party benchmark exists (no MLPerf submission by Sophgo is confirmed).

The same filing documents two accelerator blocks not present in the BM1684X: an **embedded RISC-V core** used as a fallback path for long-tail operators, and **dedicated sort / top-K hardware units** aimed at billion-scale vector search.

### SG-Link — Sophgo does have a scale-up fabric

The prior claim that Sophgo has "no proprietary scale-up interconnect" is **wrong for this generation**. Sophgo's filing names **SG-Link** as a multi-chip *and* multi-card high-speed interconnect, and `torch-tpu` documents a C2C topology on top of it.

- **Confirmed scale-up domain: 8 cards = 16 BM1690 chips = up to 2 TB of card memory in one 4U chassis.**
- Beyond the chassis, board-level Ethernet expansion ports are stated to support clusters of **~1000 cards over commodity switches**.
- **SG-Link per-link bandwidth, lane count, and topology are not disclosed** in any retrievable source.

### The 128-chip "supernode" — announced demo, unresolved numbers

At **WAIC (Shanghai, late July 2025)** Sophgo showed a **128 × BM1690 超节点** server, described as 16 chips per layer across 8 layers, with Chinese media reporting "up to 8 TB" of memory and large FP8 compute. Treat this as an **announced demonstration, not a shipping SKU**:

- No product name, interconnect specification, or availability date was disclosed.
- The reported 8 TB does **not** reconcile with ~128 GB/chip (128 × 128 GB = 16 TB). Either the memory configuration differs, the figure is a reporting error, or the layer count was conflated. **8 TB is unverified.**
- The confirmed SG-Link scale-up domain remains 8 cards / 16 chips / 2 TB, not 128 chips.

### Status — announced and lab-certified, not confirmed at scale

| Date | Event |
|---|---|
| **Sept 2024** | SC11 FP300 launched at the 2024 China Computing Power Conference (中国算力大会); top innovation award at the 2024 Beijing Security Expo |
| **2025-01-16** | **Sophgo added to the US BIS Entity List** (Federal Register 2025-00480), in the tranche tied to a TSMC-fabricated die found in a Huawei processor. Material to BM1690's foundry supply; current effect not confirmed |
| **2025-06-30** | Sophgo announced SC11 FP300 **passed adaptation verification with DeepSeek-R1 (671B)** at China Telecommunication Technology Labs (CTTL, under CAICT/MIIT) — reported by SCMP and others |
| **late July 2025** | 128-chip BM1690 supernode shown at WAIC |
| **2026-01-06** | A CSRC filing update consistent with a STAR Market (科创板) listing process is reported by Chinese media. The filing itself could not be retrieved and the circulating valuation figure is unverified — treat as **filing reported, outcome not confirmed** |
| **2026-08-08 (this scan)** | SC11 FP300 and BM1690 are **absent from Sophgo's public product catalog** — the catalog API returns 38 SKUs, all BM1684X/BM1688/CV18xx/SG2042-class — although the SC11 FP300 product URL remains search-indexed. Consistent with restricted-channel sales |

No shipment volumes, revenue, or named datacenter deployment were found. Nothing was confirmed for a BM1690 successor, and no Hot Chips / ISCA / ISSCC 2026 Sophgo presentation was confirmed.

### Software-stack changes for this generation

- **torch-tpu** (`sophgo/torch-tpu`) — PyTorch device extension targeting SG2260, with **JIT mode (SG2260 only)** and eager mode, **DeepSpeed ZeRO-1 / ZeRO-2 with CPU offload**, and **Megatron-DeepSpeed tensor parallelism**. Ships the `torch.distributed` collective suite listed in the Communication section above.
- **tpuv7-runtime / tpuv7-driver** (v1.1.3 `.deb` packages) with the **`tpu-smi`** management tool are the runtime for this generation — a separate stack from libsophon/BMRuntime, which continues to serve BM1684X/BM1688.
- **vllm-tpu** is now a **vLLM v0.11.0 fork** (pushed Dec 2025) supporting LLaMA, Qwen and DeepSeek on SG2260, including **DeepSeek-V3 and DeepSeek-R1 in FP8**, and Llama-3.1-70B / Qwen2-72B with `--tp_size 2`. Profiling uses `bigTpuProfile --arch BM1690`.
- **TPU-MLIR** carries a BM1690/bm1690e backend (`libbackend_bm1690.so`, `third_party/nntoolchain/tpuv7_sha256.txt`) with ppl-based codegen — but the *public* `llm_convert.py` still advertises only `bm1684x`/`bm1688`/`cv186x`, and the BM1690 MLIR dialect sits under `experimental/`. **BM1690 compiler support is real but less open and less mature than the BM1684X path.**
- TPU-MLIR also carries backends for parts this survey does not yet document: **SGTPUV8**, **SG2380**, and **BM1684X2**.
- **TPU-MLIR releases since the 2026-04-05 baseline** (dates from the GitHub releases atom feed; the rendered releases page omits the year): **v1.27 (2026-04-29)**, **v1.28.1 (2026-06-16)**, **v1.29-beta.0 (2026-06-23)**, **v1.29 (2026-07-01)**. v1.28.1 adds auto-detection of LLM quantization, `batch_size` support in LLM inference, a TPU Lang dump plus a Rope operator, a YOLOv26 post-process entry, a Floor activation op, and mixed CUDA/CPU inference. v1.29-beta.0 is titled around handling `A_log` in Qwen3.5. v1.29 is a narrow conv2d hardware-margin fix.
- Repo activity through August 2026 is healthy: sophon-tools (2026-08-08), sophon-demo (2026-08-07), tpu-mlir (2026-08-04), LLM-TPU (2026-08-05), libsophon (2026-07-27). No new BM1690- or SG2044-specific public repo appeared.

### SG2044 — second-generation RISC-V server CPU

*Reported by the 2026-08-08 scan; not independently re-verified in the adversarial pass, so held at lower confidence than the BM1690 material above.*

**SG2044** is the successor to SG2042: a second-generation 64-core RISC-V server CPU adding **RVV 1.0 vector support** and an improved memory subsystem. Platform support was merged into **mainline Linux 6.16**. It is productized as the **SRA3-40** (compute), **SRB3-40** (storage) and **SRM3-40** (converged) servers. The first comprehensive HPC evaluation is **arXiv:2508.13840** (August 2025), which compares it against SG2042 plus x86 and Arm server parts. Clock, process node, cache sizes and TDP are **not disclosed** here.

---

## Competitive Position

Sophgo's BM1684X/BM1688 occupies a distinct niche: **affordable edge-to-cloud inference for Chinese industrial AI**, primarily competing with NVIDIA Jetson (edge) and older H20/L20 in cost-sensitive inference deployments. The BM1690 / SC11 FP300 generation moves Sophgo out of that niche and into large-model territory, but on announced-and-lab-certified rather than confirmed-at-scale footing. Key differentiators:

- **Open-source compiler**: TPU-MLIR is more open than Cambricon MagicMind or Huawei CANN; enables academic research and third-party integrations. Caveat: the BM1690 path is less open than the BM1684X path (dialect under `experimental/`, `llm_convert.py` still advertises only BM1684X/BM1688/CV186X)
- **RISC-V + TPU hybrid portfolio**: Unique among Chinese AI chip vendors; positions Sophgo for long-term domestic RISC-V ecosystem bets (China's RISC-V push). BM1690 also embeds a RISC-V core on-die as a long-tail-operator fallback path
- **Bitmain heritage**: Deep ASIC manufacturing expertise; strong supply-chain relationships
- **Capacity-first memory strategy**: ~128 GB per BM1690 / 256 GB per SC11 FP300 card from low-power DRAM rather than HBM — large capacity at ~1.1 TB/s per card, far below HBM peers on bandwidth but well above them on GB-per-watt and GB-per-dollar. Attractive for KV-cache-heavy long-context serving; weak for bandwidth-bound decode
- **Limitations (revised 2026-08-08)**: Per-chip compute remains below Huawei Ascend 910B or any H100-class part, and bandwidth per chip (~546 GB/s) is roughly half of an HBM-equipped Chinese peer. No third-party benchmark of any kind exists. Process node and foundry are undisclosed, and Sophgo has been on the **US BIS Entity List since 2025-01-16**, making its leading-edge supply a live open question. SC11 FP300 is absent from the public product catalog

> **Superseded 2026-08-08.** The former bullet "*No large-scale training capability; no proprietary scale-up interconnect*" was true of BM1684X but is wrong for the BM1690 generation: **SG-Link** is a proprietary multi-chip/multi-card scale-up fabric, and `torch-tpu` ships DeepSpeed ZeRO-1/ZeRO-2 and Megatron tensor parallelism with a full collective backend on SG2260.

---

## Sources

- [sophgo/tpu-mlir GitHub](https://github.com/sophgo/tpu-mlir)
- [sophgo/libsophon GitHub](https://github.com/sophgo/libsophon)
- [sophgo/LLM-TPU GitHub](https://github.com/sophgo/LLM-TPU)
- [SOPHON BM1684X product page](https://www.sophon.ai/product/introduce/bm1684x.html)
- [SOPHGO BM1684X IEEE paper](https://ieeexplore.ieee.org/document/10764438/)
- [SG2042 processor spec (Milk-V)](https://milkv.io/docs/pioneer/getting-started/processor)
- [SG2380 CNX-Software article](https://www.cnx-software.com/2023/10/21/sophgo-sg2380-16-core-sifive-p670-risc-v-processor-20-tops-ai-accelerator/)
- [SophonSDK User Guide v23.05](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/SophonSDK_doc/en/html/sdk_intro/1_intro.html)
- [TPU-MLIR Introduction docs](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/quick_start_en/html/01_introduction.html)

### Added 2026-08-08 (BM1690 / SC11 FP300 / SG2044)

- [sophgo/mcu — BoardType.md (board-type ↔ chip mapping)](https://github.com/sophgo/mcu/blob/master/BoardType.md)
- [sophgo/mcu — SC11FP300/common.h (`#define SOC_NUM 2`)](https://github.com/sophgo/mcu/blob/master/SC11FP300/common.h)
- [tpu-mlir — python/profile_helper/bm1690_defs.py (Chip Arch = sg2260; 8 cores; 64 NPU; 8533 MT/s)](https://github.com/sophgo/tpu-mlir/blob/master/python/profile_helper/bm1690_defs.py)
- [tpu-mlir — include/tpu_mlir/Backend/BM168x/BM1690.h (NPU_NUM/EU_BYTES/LMEM/L2_SRAM_SIZE)](https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Backend/BM168x/BM1690.h)
- [sophgo/vllm-tpu — vLLM v0.11.0 fork for SG2260; tpuv7-driver/runtime; sc11_config.ini + sc11_config_chip2.ini](https://github.com/sophgo/vllm-tpu)
- [sophgo/torch-tpu — PyTorch extension for SG2260; DeepSpeed ZeRO, Megatron TP, dist_test2260 collectives, C2C topology](https://github.com/sophgo/torch-tpu)
- [SC11 FP300 innovation-product filing, China Security & Protection Industry Association (2024-09-30, vendor-supplied)](https://xh.21csp.com.cn/cxcp_2024/202409/2056.html)
- [SCMP — Sophgo adapts compute card for DeepSeek (CTTL verification, 2025-06-30)](https://www.scmp.com/tech/tech-trends/article/3316363/chinese-chipmaker-sophgo-adapts-compute-card-deepseek-beijings-self-reliance-push)
- [WAIC 2025 — 128 × BM1690 超节点 server (Chinese media report)](https://finance.sina.com.cn/tech/2025-07-28/doc-infhywrc0366459.shtml)
- [US BIS Entity List addition, effective 2025-01-16 (Federal Register 2025-00480)](https://www.federalregister.gov/documents/2025/01/16/2025-00480/additions-to-the-entity-list)
- [tpu-mlir releases atom feed (v1.27 / v1.28.1 / v1.29-beta.0 / v1.29 dates)](https://github.com/sophgo/tpu-mlir/releases.atom)
- [SG2044 HPC evaluation — arXiv:2508.13840 (Aug 2025)](https://arxiv.org/abs/2508.13840)
