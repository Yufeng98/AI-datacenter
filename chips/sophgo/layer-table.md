# Sophgo (算能) Layer Mapping Table

*as_of: 2026-08-08*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | LLM-TPU (sophgo/LLM-TPU): deploy Qwen2/LLaMA/HuggingFace LLMs on BM1684X/BM1688; TPU-MLIR compiled .bmodel; PCIE+SoC modes | confirmed | github-llm-tpu |
| Framework Integration | vllm-tpu (sophgo/vllm-tpu): vLLM **v0.11.0** fork for production LLM serving; continuous batching, KV-cache, OpenAI API. Targets **SG2260/BM1690**; DeepSeek-V3 + DeepSeek-R1 in **FP8**; Llama-3.1-70B / Qwen2-72B with `--tp_size 2` | confirmed | github-vllm-tpu |
| Framework Integration | **torch-tpu (sophgo/torch-tpu): PyTorch device extension for SG2260/BM1690; JIT mode (SG2260 only) + eager mode; DeepSpeed ZeRO-1/ZeRO-2 with CPU offload; Megatron-DeepSpeed tensor parallelism** | confirmed | github-torch-tpu |
| Framework Integration | LLM-TPU_Lite (sophgo/LLM-TPU_Lite): lightweight edge LLM inference for BM1688/CV186X | confirmed | github-llm-tpu-lite |
| Framework Integration | llmc-tpu (sophgo/llmc-tpu): quantization-aware model calibration (INT4/INT8) before TPU-MLIR compilation | confirmed | github-llmc-tpu |
| Framework Integration | sophon-sail: Python/C++ high-level wrapper (sail.Engine, sail.BMImage, sail.Decoder) over BMRuntime+BMCV+BMLib | confirmed | sophon-sdk-docs |
| Framework Integration | SOPHON-MW: BM-OpenCV + BM-FFmpeg (hardware H.264/H.265 encode/decode via JPU/VPP hardware) | confirmed | sophon-sdk-docs |
| Framework Integration | Direct model support (TPU-MLIR): ONNX (native), TFLite (native), Caffe (native), PyTorch (via ONNX), TF/Paddle/MXNet (via ONNX) | confirmed | tpu-mlir-docs |
| Compiler / IR | TPU-MLIR (sophgo/tpu-mlir): open-source MLIR-based pipeline; model_transform.py → Top dialect .mlir → model_deploy.py → .bmodel | confirmed | github-tpu-mlir, tpu-mlir-docs |
| Compiler / IR | Top Dialect: framework-neutral MLIR IR (TopConv, TopMatMul, etc.) | confirmed | tpu-mlir-docs |
| Compiler / IR | TPU Dialect: chip-specific lowered IR with explicit DMA scheduling and SRAM tiling | confirmed | tpu-mlir-docs |
| Compiler / IR | INT8/INT4 quantization (calibration table via run_calibration.py; integrated in model_deploy.py) | confirmed | tpu-mlir-docs |
| Compiler / IR | **TPU-MLIR BM1690/bm1690e backend: `libbackend_bm1690.so` + `third_party/nntoolchain/tpuv7_sha256.txt`; ppl-based codegen; native FP8 E5M2 and 8-core codegen (`resnet50_v2_bm1690_f8e5m2` / `..._f16_core8` regression bmodels). Less open/mature than the BM1684X path: BM1690 dialect under `experimental/`, and public `llm_convert.py` still advertises only bm1684x/bm1688/cv186x** | confirmed | github-tpu-mlir |
| Compiler / IR | **TPU-MLIR also carries backends for undocumented parts: SGTPUV8, SG2380, BM1684X2** | confirmed | github-tpu-mlir |
| Compiler / IR | **TPU-MLIR releases since 2026-04-05 baseline: v1.27 (2026-04-29), v1.28.1 (2026-06-16), v1.29-beta.0 (2026-06-23), v1.29 (2026-07-01). v1.28.1: LLM-quantization auto-detect, batch_size in LLM inference, TPU Lang dump + Rope op, YOLOv26 post-process, Floor activation, mixed CUDA/CPU inference. v1.29-beta.0: A_log handling for Qwen3.5. v1.29: conv2d hw-margin fix** | confirmed | tpu-mlir-releases-atom |
| Compiler / IR | tpu_compiler (legacy, sophgo/tpu_compiler): earlier CVITEK compiler for CV18XX; superseded by TPU-MLIR | confirmed | github-tpu-compiler |
| Model Binary | .bmodel: Sophgo hardware binary (compiled ops + DMA cmds + quantized weights + topology + target chip ID) | confirmed | tpu-mlir-docs, bmrt-docs |
| Runtime | BMRuntime (BMRT / libbmrt.so, in libsophon): load+execute .bmodel; bmrt_load_bmodel, bmrt_launch_tensor, bmrt_memcpy_s2d/d2s | confirmed | bmruntime-docs |
| Runtime | BMLib (libbmlib.so, in libsophon): low-level device+memory mgmt; bm_dev_request, bm_malloc_device_byte, bm_memcpy_s2d | confirmed | bmlib-docs |
| Runtime | BMCV (libbmcv.so, in libsophon): hardware CV preprocessing via VPP+JPU; resize, CSC, affine, JPEG, normalize, NMS | confirmed | sophon-sdk-docs |
| Driver / Firmware | libsophon (sophgo/libsophon): unified package (kernel driver + BMLib + BMRuntime + BMCV + headers) | confirmed | github-libsophon |
| Driver / Firmware | sophon.ko / bmsophon.ko: Linux kernel driver; PCIe BAR, IOCTL dispatch, DMA engine, interrupt handling | confirmed | github-libsophon |
| Driver / Firmware | Docker: sophgo/tpuc_dev official images; Kubernetes device plugin for TPU resource management | confirmed | sophon-sdk-docs |
| Runtime | **tpuv7-runtime / tpuv7-driver (v1.1.3 .deb) + `tpu-smi` management tool — the runtime for the BM1690/SG2260 (TPUv7) generation; firmware in `/lib/firmware/tpuv7/`. Separate stack from libsophon/BMRuntime, which continues to serve BM1684X/BM1688** | confirmed | github-vllm-tpu |
| Runtime | **`bigTpuProfile --arch BM1690` — profiling tool for the TPUv7 generation** | confirmed | github-vllm-tpu |
| Communication | BM1684X generation: no collective library (no NCCL/CNCL equivalent); inference-only positioning; multi-card via app-level data parallelism over PCIe | confirmed | sophon-sdk-docs |
| Communication | **BM1690/SG2260 generation: torch-tpu ships a full `torch.distributed` collective backend (`python/dist_test2260/`: all_reduce, all_gather, all_gather_into_tensor, all_to_all, broadcast, reduce, gather, scatter, p2p, DDP) plus a documented C2C topology, over the SG-Link fabric. Supersedes the "Sophgo has no collective library" claim for this generation** | confirmed | github-torch-tpu |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | **BM1690 (silicon name SG2260; software generation TPUv7): 8 TPU cores/chip; 64 NPU lanes/core; 512-bit (64 B) EU vector width; 16 MB LMEM/core (256 KB × 16 banks); on-die RISC-V core for long-tail-operator fallback; dedicated sort/top-K units. Native FP8 (E5M2) codegen confirmed** | confirmed (from Sophgo's own compiler backend — machine model, not datasheet) | github-tpu-mlir-bm1690h, github-tpu-mlir-bm1690defs |
| Compute Engine | **BM1690 peak: per-chip figures are NOT published. SC11 FP300 card (= 2 × BM1690) vendor claim: >400 TOPS INT8/FP8, >200 TFLOPS FP16/BF16, >100 TFLOPS TF32, >25 TFLOPS FP32, 300 W whole-card TDP → implied per chip ~200 TOPS / ~100 TFLOPS / ~150 W. Data types INT4/INT8/FP8/NF4/TF32/FP16/BF16/FP32** | vendor claim (industry-association filing of a vendor-authored document; no third-party benchmark exists) | csp-sc11-filing |
| Compute Engine | **BM1690 process node and foundry: not disclosed by any retrievable source. Sophgo added to US BIS Entity List effective 2025-01-16** | not disclosed | federal-register-2025-00480 |
| Compute Engine | **BM1690E: cut-down BM1690 sibling — 16 MB L2 SRAM instead of 128 MB; RISC-V kernel-module variant; board SC11E FP300** | confirmed | github-tpu-mlir-bm1690eh |
| Compute Engine | BM1684X TPU: high data-width SIMD tensor processor (4th gen); 32 TOPS INT8 / 35.2 TOPS Winograd; 16 TFLOPS FP16/BF16; 2 TFLOPS FP32 | confirmed | sophon-bm1684x-page, ieee-bm1684x |
| Compute Engine | BM1684X SoC CPU: Octa-core ARM Cortex-A53 @ 2.3 GHz (management CPU; runs Linux; non-compute tasks) | confirmed | sophon-bm1684x-page |
| Compute Engine | BM1688 TPU: 16 TOPS INT8 / 32 TOPS INT4; 4 TFLOPS FP16/BF16; 0.5 TFLOPS FP32; Cortex-A53 @ 1.6 GHz | confirmed | aimorelogy-bm1688 |
| Compute Engine | SG2042 RISC-V: 64 cores (16 clusters × 4 T-Head C920 OoO); 2 GHz; TSMC 6nm; 120W TDP | confirmed | milkv-sg2042 |
| Compute Engine | **SG2044 RISC-V: second-gen 64-core server CPU (SG2042 successor); adds RVV 1.0 vector support + improved memory subsystem; mainline Linux 6.16 platform support; SRA3-40 / SRB3-40 / SRM3-40 servers. Frequency, process, cache, TDP not disclosed** | reported (raw scan; not re-verified in adversarial pass) | arxiv-2508.13840 |
| Compute Engine | SG2380 RISC-V SoC: 16-core SiFive P670 (12 perf @ 2.5 GHz + 4 eff @ 1.6 GHz) + 20 TOPS AI accelerator | confirmed | cnxsoftware-sg2380 |
| Compute Engine | Process: BM1684X/BM1688 — 12nm Samsung; SG2042 — TSMC 6nm; **BM1690/SG2260 and SG2044 — not disclosed** | confirmed | multiple |
| Data Path | Compiler-scheduled (static) command execution: TPU-MLIR statically schedules all ops; no OoO execution in TPU | confirmed | tpu-mlir-docs |
| Data Path | **BM1690: multi-core static scheduling — core count is a compile-time codegen parameter (`..._core8.bmodel`), not a runtime scheduling decision; explicit multi-core interface in the TPU-MLIR backend. Three-tier data path DRAM → 128 MB L2 SRAM → 16 MB LMEM/core → 64-lane 512-bit EU** | confirmed | github-tpu-mlir-bm1690h |
| Data Path | Dedicated DMA engine: async LPDDR4X ↔ on-chip SRAM prefetch; enables compute-DMA double-buffering | confirmed | bmlib-docs |
| Data Path | BMCV VPP pipeline: hardware image preprocessing (resize/CSC/affine) feeding directly into TPU-accessible memory | confirmed | sophon-sdk-docs |
| Data Path | Video codec hardware: BM1684X — 32-ch HD decode, 12-ch HD encode; BM1688 — 16-ch decode, 10-ch encode | confirmed | sophon-bm1684x-page |
| On-chip Memory | BM1684X/BM1688: compiler-managed SRAM scratchpad (no HW data cache in primary compute path); capacity undisclosed | inferred | tpu-mlir-docs |
| On-chip Memory | **BM1690: two-tier on-chip SRAM — 16 MB LMEM per core (128 MB across 8 cores) + 128 MB chip-shared L2 SRAM (`L2_SRAM_SIZE = 0x8000000`); modeled L2 max BW 128 GB/s. First Sophgo TPU with an explicit intermediate SRAM tier. BM1690E: 16 MB L2** | confirmed (compiler machine model) | github-tpu-mlir-bm1690h, github-tpu-mlir-bm1690defs |
| Off-chip Memory | BM1684X: LPDDR4X 16 GB ~68 GB/s; BM1688: LPDDR4X 8 GB | confirmed | sophon-bm1684x-page |
| Off-chip Memory | **BM1690: ~128 GB per chip (vllm-tpu `share-mem-start=0x1e0000000` + `share-mem-size=0x1e20000000`); ~546 GB/s per chip derived from 8533 MT/s DDR clock × 68.264 GB/s/core × 8 cores. SC11 FP300 card (2 chips): up to 256 GB, >1.1 TB/s (vendor claim)** | capacity confirmed; bandwidth derived; card figures vendor claim | github-vllm-tpu, github-tpu-mlir-bm1690defs, csp-sc11-filing |
| Off-chip Memory | **BM1690 memory type: NOT CONFIRMED. LPDDR5X asserted only by Baidu Baike (user-editable); Sophgo's own filing says only "低功耗内存技术". Consistent with 8533 MT/s but on no retrievable Sophgo datasheet — do not state flat** | not disclosed | baidu-baike-sc11 (weak), csp-sc11-filing |
| Off-chip Memory | SG2042: DDR4-3200 × 4 controllers (RDIMM/ECC/UDIMM); SG2380: up to 96 GB (192-bit interface) | confirmed | milkv-sg2042, phoronix-sg2380 |
| Host Interface / Package | BM1684X: PCIe Gen3/4 x16 (SoC mode + PCIe card mode) | confirmed | sophon-bm1684x-page |
| Host Interface / Package | **BM1684X PCIe cards are SC7PRO (8 chips), SC7FP150 (6 chips), SC7HP75 (3 chips), SC7HP75_1 (1 chip) — multi-chip cards. *Correction 2026-08-08: the previously listed "SC7 FP300" does not exist; "FP300" belongs only to the SC11/BM1690 card.*** | confirmed | github-mcu-boardtype |
| Host Interface / Package | **SC11 FP300 = 2 × BM1690 on one PCIe card, 300 W whole-card TDP (`#define SOC_NUM 2`; driver emits sc11_config.ini + sc11_config_chip2.ini). MCU board types: 0xB2 = BM1690/BM1690EVB, 0xB3 = BM1690/SC11. PCIe generation and width not disclosed** | confirmed | github-mcu-sc11fp300, github-mcu-boardtype, github-vllm-tpu |
| Scale-up Interconnect | BM1684X generation: no proprietary chip-to-chip interconnect; PCIe peer-to-peer only; inference-only positioning | confirmed | sophon-sdk-docs |
| Scale-up Interconnect | **BM1690: SG-Link — proprietary multi-chip AND multi-card high-speed fabric. Confirmed scale-up domain: 8 cards = 16 chips = up to 2 TB card memory in one 4U chassis. Per-link bandwidth, lane count and topology NOT disclosed. Supersedes the repo's prior "Sophgo has no proprietary scale-up interconnect" claim** | confirmed (existence + domain); BW/topology not disclosed | csp-sc11-filing, github-torch-tpu |
| Scale-up Interconnect | **128 × BM1690 "超节点" shown at WAIC July 2025 (16 chips/layer × 8 layers): announced demonstration only — no product name, interconnect spec, or ship date. Media-reported "8 TB memory" does not reconcile with ~128 GB/chip (128 × 128 GB = 16 TB) and is unverified** | announced demo; figure unverified | sina-waic-2025 |
| Scale-up Interconnect | SG2042: CCIX chip-to-chip (2-socket scale-up); 32 × PCIe Gen4.0 | confirmed | milkv-sg2042 |
| Scale-out Interconnect | Standard Ethernet (1/10/25 GbE on SE/SG server products); no RDMA fabric | confirmed | sophon-sdk-docs |
| Scale-out Interconnect | **SC11 FP300: board-level Ethernet expansion ports stated to support ~1000-card clusters over commodity switches. Port count and per-port rate not disclosed. Ethernet, not SG-Link, is the scale-out path** | vendor claim | csp-sc11-filing |
| Deployment Status | **BM1690/SC11 FP300: announced + lab-certified, NOT confirmed deployed at scale. Launched Sept 2024 (中国算力大会); CTTL/CAICT DeepSeek-R1 671B adaptation verification announced 2025-06-30; WAIC supernode demo July 2025. No shipment volume, revenue, or named datacenter deployment found. Absent from Sophgo's public catalog API (38 SKUs, none BM1690/SC11) though the product URL stays search-indexed — consistent with restricted-channel sales** | confirmed | scmp-sophgo-deepseek, sophgo-catalog-api |

## Source Keys Added 2026-08-08

| Key | URL |
|-----|-----|
| github-mcu-boardtype | https://github.com/sophgo/mcu/blob/master/BoardType.md |
| github-mcu-sc11fp300 | https://github.com/sophgo/mcu/blob/master/SC11FP300/common.h |
| github-tpu-mlir-bm1690h | https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Backend/BM168x/BM1690.h |
| github-tpu-mlir-bm1690eh | https://github.com/sophgo/tpu-mlir (include/tpu_mlir/Backend/BM168x/BM1690E.h) |
| github-tpu-mlir-bm1690defs | https://github.com/sophgo/tpu-mlir/blob/master/python/profile_helper/bm1690_defs.py |
| github-torch-tpu | https://github.com/sophgo/torch-tpu |
| tpu-mlir-releases-atom | https://github.com/sophgo/tpu-mlir/releases.atom |
| csp-sc11-filing | https://xh.21csp.com.cn/cxcp_2024/202409/2056.html (vendor-authored, republished by industry association) |
| scmp-sophgo-deepseek | https://www.scmp.com/tech/tech-trends/article/3316363/chinese-chipmaker-sophgo-adapts-compute-card-deepseek-beijings-self-reliance-push |
| sina-waic-2025 | https://finance.sina.com.cn/tech/2025-07-28/doc-infhywrc0366459.shtml |
| federal-register-2025-00480 | https://www.federalregister.gov/documents/2025/01/16/2025-00480/additions-to-the-entity-list |
| sophgo-catalog-api | https://www.sophgo.com/basic-api/business/product/getProductList (POST; 38 SKUs, none BM1690/SC11) |
| baidu-baike-sc11 | https://baike.baidu.com/item/SC11%20FP300/67686300 (user-editable; only source naming LPDDR5X — weak) |
| arxiv-2508.13840 | https://arxiv.org/abs/2508.13840 |
