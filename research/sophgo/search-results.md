# Sophgo (算能) — Search Results

*as_of: 2026-08-08*
*chip: sophgo*
*device_class: RISC-V + TPU Hybrid (China, 算能)*

---

## Search Queries

1. "Sophgo BM1684X BM1688 AI chip architecture specifications 2024 2025"
2. "算能 BM1684X TPU 架构 规格 深度学习"
3. "Sophgo SDK TPU-MLIR toolchain compiler open source"
4. "Sophgo RISC-V SG2042 SG2380 processor specifications performance"
5. "Sophgo open source GitHub repositories sophgo LLM TPU runtime"
6. "Sophgo Bitmain AI accelerator cloud SC7 FP300 HP75 inference server specifications TOPS"

---

## Resources Found

### Official Product Pages

| Resource | URL | Category |
|----------|-----|----------|
| SOPHON BM1684X product page | https://www.sophon.ai/product/introduce/bm1684x.html | Hardware Spec |
| ~~SC7 FP300 accelerator card~~ | ~~https://sophon.ai/product/introduce/sc7.html~~ | **REMOVED 2026-08-08 — dead URL for a product that does not exist. No BM1684X card named "SC7 FP300" appears in Sophgo's own board-type table; the real BM1684X cards are SC7PRO / SC7FP150 / SC7HP75 / SC7HP75_1. "FP300" belongs only to the SC11 (BM1690) card.** |
| SC7 HP75-I accelerator card | https://sophon.ai/product/introduce/sc7-hp75.html | Hardware Spec |
| SOPHON SDK product page | https://www.sophon.ai/product/introduce/bmnn-sdk.html | Software |
| SOPHGO main site | https://sophon.ai/ | Overview |

### Open-Source GitHub Repositories

| Resource | URL | Category |
|----------|-----|----------|
| sophgo/tpu-mlir — ML compiler (MLIR-based) | https://github.com/sophgo/tpu-mlir | Compiler |
| sophgo/libsophon — driver + runtime library | https://github.com/sophgo/libsophon | Runtime/Driver |
| sophgo/LLM-TPU — LLM deployment on BM1684X/BM1688 | https://github.com/sophgo/LLM-TPU | Framework |
| sophgo/LLM-TPU_Lite — LLM on lite TPU | https://github.com/sophgo/LLM-TPU_Lite | Framework |
| sophgo/vllm-tpu — vLLM fork for Sophgo TPUs | https://github.com/sophgo/vllm-tpu | Framework |
| sophgo/llmc-tpu — ModelTC/llmc for Sophgo | https://github.com/sophgo/llmc-tpu | Framework |
| sophgo/tpu_compiler — legacy CVITEK AI compiler | https://github.com/sophgo/tpu_compiler | Compiler |
| sophgo org repositories | https://github.com/orgs/sophgo/repositories | Overview |

### Documentation

| Resource | URL | Category |
|----------|-----|----------|
| TPU-MLIR Introduction (v23.05) | https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/quick_start_en/html/01_introduction.html | Compiler Docs |
| SophonSDK User Guide (v23.05) | https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/SophonSDK_doc/en/html/sdk_intro/1_intro.html | SDK Docs |
| BMRuntime reference | https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-runtime/reference_en/html/bmruntime/runtime.html | Runtime Docs |
| BMLIB reference | https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/libsophon/reference_en/html/2_bmlib_basic_concept.html | Runtime Docs |
| DeepWiki sophgo/tpu-mlir | https://deepwiki.com/sophgo/tpu-mlir | Analysis |

### RISC-V Processors

| Resource | URL | Category |
|----------|-----|----------|
| SG2042 processor spec (Milk-V Pioneer) | https://milkv.io/docs/pioneer/getting-started/processor | Hardware Spec |
| SG2042 spec PDF | https://ftp.radix-linux.su/3pp/Sophgo/doc/Milk-V/SG2042_Draft_Spec_V1.0.pdf | Hardware Spec |
| SG2380 (OASIS) CNX-Software article | https://www.cnx-software.com/2023/10/21/sophgo-sg2380-16-core-sifive-p670-risc-v-processor-20-tops-ai-accelerator/ | Hardware Spec |
| SiFive+Sophgo license announcement | https://www.sifive.com/press/sophgo-licenses-sifive-risc-v-processor-cores-to-drive | Partnership |
| SG2042 HPC evaluation (arXiv) | https://ui.adsabs.harvard.edu/abs/2023arXiv230900381B/abstract | Research |

### IEEE / Academic

| Resource | URL | Category |
|----------|-----|----------|
| SOPHGO BM1684X IEEE paper | https://ieeexplore.ieee.org/document/10764438/ | Research |

---

## Key Findings

- **Hardware family**: TPU line (BM1684/BM1684X/BM1688 for inference) + RISC-V CPU line (SG2042 64-core, SG2380 16-core)
- **TPU compiler**: TPU-MLIR — open-sourced MLIR-based compiler converting ONNX/Caffe/TFLite/PyTorch to `.bmodel` binary
- **Runtime stack**: libsophon (BMLib + BMRuntime/BMRT + BMCV) → kernel driver; sophon-sail Python/C++ high-level wrapper
- **LLM support**: LLM-TPU + vllm-tpu for Qwen2, LLaMA-series on BM1684X/BM1688
- **Bitmain heritage**: Sophgo is the AI division spun off from Bitmain (Bitcoin ASIC maker) ~2021; shares manufacturing supply-chain expertise
- **Domestic positioning**: Positioned as Jetson/edge-inference alternative in Chinese industrial AI market; BM1688 targets drone/vision/edge; BM1684X targets cloud inference

---

# Search Update 2026-08-08 — BM1690 / SG2260 (TPUv7), SC11 FP300, SG2044

## Additional Search Queries

7. "Sophgo BM1690 SG2260 TPUv7 specifications cores memory bandwidth"
8. "算能 SC11 FP300 算力卡 规格 256GB 400TOPS SG-Link"
9. "Sophgo SC11 FP300 DeepSeek CTTL 适配验证"
10. "算能 BM1690 超节点 WAIC 2025 128颗"
11. "sophgo torch-tpu SG2260 DeepSpeed Megatron distributed collectives"
12. "sophgo tpuv7-runtime tpuv7-driver tpu-smi"
13. "Sophgo SG2044 RISC-V RVV 1.0 Linux 6.16 SRA3-40"
14. "Sophgo Entity List BIS January 2025"

**Search-method note.** `sophgo.com` product pages return near-empty content to a fetcher, and there is no retrievable BM1690 datasheet. The productive route was **Sophgo's own open-source repos** (`mcu`, `tpu-mlir`, `vllm-tpu`, `torch-tpu`), which encode board topology, core counts, SRAM sizes and memory carve-outs directly in firmware and compiler code. Treat those as the primary sources for this generation; treat everything with a performance number on it as vendor marketing until a third party measures it.

## Resources Found — Tier A: Sophgo's own code and firmware (strongest)

| Resource | URL | Category | What it establishes |
|----------|-----|----------|---------------------|
| sophgo/mcu — BoardType.md | https://github.com/sophgo/mcu/blob/master/BoardType.md | Hardware Spec (primary) | `0xB2` = BM1690/BM1690EVB, `0xB3` = BM1690/SC11; BM1684X cards = SC7PRO (8), SC7FP150 (6), SC7HP75 (3), SC7HP75_1 (1) — and **no "SC7 FP300"** |
| sophgo/mcu — SC11FP300/common.h | https://github.com/sophgo/mcu/blob/master/SC11FP300/common.h | Hardware Spec (primary) | `#define SOC_NUM 2` — SC11 FP300 carries **two** BM1690 chips |
| tpu-mlir — BM1690.h | https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Backend/BM168x/BM1690.h | Hardware Spec (primary) | `NPU_NUM=64`, `EU_BYTES=64`, `LMEM_BYTES=256KB`, `LMEM_BANKS=16`, `L2_SRAM_SIZE=0x8000000` (128 MB) |
| tpu-mlir — bm1690_defs.py | https://github.com/sophgo/tpu-mlir/blob/master/python/profile_helper/bm1690_defs.py | Hardware Spec (primary) | `"Chip Arch": "sg2260"`, Core Num 8, TIU/DMA 1000 MHz, DDR 8533 MT/s, DDR max BW 68.264 GB/s/core, L2 max BW 128 GB/s |
| sophgo/tpu-mlir (repo tree) | https://github.com/sophgo/tpu-mlir | Compiler | `resnet50_v2_bm1690_f16_core8.bmodel`, `resnet50_v2_bm1690_f8e5m2.bmodel`; `libbackend_bm1690.so`; `tpuv7_sha256.txt`; SGTPUV8 / SG2380 / BM1684X2 backends |
| sophgo/vllm-tpu | https://github.com/sophgo/vllm-tpu | Framework | vLLM v0.11.0 fork for SG2260; DeepSeek-V3/R1 FP8; `--tp_size 2`; tpuv7-driver/runtime 1.1.3; `tpu-smi`; `share-mem-size=0x1e20000000` (~128 GB/chip); `bigTpuProfile --arch BM1690` |
| sophgo/torch-tpu | https://github.com/sophgo/torch-tpu | Framework | PyTorch extension for SG2260; JIT + eager; DeepSpeed ZeRO-1/2 CPU offload; Megatron TP; `python/dist_test2260/` collectives; `3_c2c_topology.png`; `06_distribute.rst` |
| tpu-mlir releases atom feed | https://github.com/sophgo/tpu-mlir/releases.atom | Compiler | Authoritative release dates (the rendered page omits the year) |
| tpu-mlir v1.28.1 release notes | https://github.com/sophgo/tpu-mlir/releases/tag/v1.28.1 | Compiler | LLM-quantization auto-detect; batch_size; Rope op; YOLOv26; Floor; mixed CUDA/CPU |
| Sophgo product catalog API | https://www.sophgo.com/basic-api/business/product/getProductList | Overview | POST returns 38 SKUs — **none** BM1690/SC11/SC7. Evidence of restricted-channel sales |

## Resources Found — Tier B: vendor-authored, third-party-republished (marketing)

| Resource | URL | Category | Caveat |
|----------|-----|----------|--------|
| SC11 FP300 innovation-product filing (China Security & Protection Industry Association, 2024-09-30, 厦门算能科技有限公司) | https://xh.21csp.com.cn/cxcp_2024/202409/2056.html | Hardware Spec | **Vendor-authored.** Card-level 300 W / >400 TOPS INT8-FP8 / >200 TFLOPS FP16-BF16 / >100 TFLOPS TF32 / >25 TFLOPS FP32 / 256 GB / >1.1 TB/s; SG-Link; 4U = 2 TB; ~1000-card Ethernet clusters; embedded RISC-V; sort/top-K units |
| SC11 FP300 product page (search-indexed, near-empty to fetchers) | https://www.sophgo.com/sophon-u/product/introduce/sc11_fp300.html | Hardware Spec | Still indexed but absent from the catalog API |

## Resources Found — Independent / regulatory

| Resource | URL | Category |
|----------|-----|----------|
| SCMP — Sophgo adapts compute card for DeepSeek (CTTL, 2025-06-30) | https://www.scmp.com/tech/tech-trends/article/3316363/chinese-chipmaker-sophgo-adapts-compute-card-deepseek-beijings-self-reliance-push | Deployment |
| China Strategy — same CTTL verification | https://www.chinastrategy.org/2025/06/30/chinas-sophgo-adapts-chip-product-for-deepseek-in-self-reliance-push/ | Deployment |
| BIS Entity List additions effective 2025-01-16 (FR 2025-00480) | https://www.federalregister.gov/documents/2025/01/16/2025-00480/additions-to-the-entity-list | Supply Chain |
| SG2044 HPC evaluation (arXiv:2508.13840, Aug 2025) | https://arxiv.org/abs/2508.13840 | Research |

## Resources Found — Tier C: media / user-editable (existence claims only)

| Resource | URL | Category | Caveat |
|----------|-----|----------|--------|
| Sina Finance — WAIC 2025 Sophgo 超节点 (128 × BM1690) | https://finance.sina.com.cn/tech/2025-07-28/doc-infhywrc0366459.shtml | Event | "8 TB" figure unverified and inconsistent with ~128 GB/chip |
| x-techcon — WAIC 2025 coverage | https://www.x-techcon.com/article/23851.html | Event | same |
| elecfans — WAIC 2025 coverage | https://www.elecfans.com/d/6900141.html | Event | same |
| Baidu Baike — SC11 FP300 | https://baike.baidu.com/item/SC11%20FP300/67686300 | Hardware Spec | **User-editable; 403s to direct fetch.** The *only* source naming LPDDR5X. Do not cite as a spec source |
| Tencent News — Sophgo STAR Market filing report (Jan 2026) | https://news.qq.com/rain/a/20260112A06P1100 | Corporate | Filing itself not retrievable; valuation unverified |

## Key Findings — Update 2026-08-08

- **The survey's flagship framing was an order of magnitude wrong.** Sophgo's datacenter part is **BM1690** (silicon **SG2260**, software generation **TPUv7**), not BM1684X. Per chip: 8 TPU cores, 64 lanes/core, 512-bit EU, 16 MB LMEM/core, **128 MB L2 SRAM**, ~128 GB DRAM at ~546 GB/s (derived), native FP8 E5M2.
- **Card vs chip is the central trap.** 256 GB / >400 TOPS / >1.1 TB/s / 300 W are all **per SC11 FP300 card, which holds TWO BM1690 chips**. Halve for per-chip.
- **Sophgo does have a scale-up fabric — SG-Link.** Confirmed domain 8 cards = 16 chips = 2 TB in one 4U chassis; ~1000-card Ethernet clusters beyond. Per-link bandwidth and topology **not disclosed**.
- **Sophgo does ship collectives.** `torch-tpu` has a full `torch.distributed` backend for SG2260 plus DeepSpeed ZeRO-1/2 and Megatron TP. The repo's "no collective library / no training capability" claims are refuted for this generation.
- **Two runtime stacks now coexist**: libsophon/BMRuntime (BM1684X/BM1688) and tpuv7-driver/tpuv7-runtime + `tpu-smi` (BM1690).
- **Not disclosed, do not estimate**: BM1690 process node and foundry; off-chip DRAM type (LPDDR5X is Baike-only); SG-Link per-link BW, lane count and topology; per-chip peak figures.
- **Status is announced + lab-certified, not deployed at scale**: launched Sept 2024, CTTL DeepSeek-R1 671B verification 2025-06-30, WAIC supernode demo July 2025. No volume, revenue, deployment, or third-party benchmark found; the card is absent from the public catalog.
- **Supply-chain constraint**: Sophgo on the **US BIS Entity List since 2025-01-16**.
- **New RISC-V part**: **SG2044** (SG2042 successor, RVV 1.0, mainline Linux 6.16, SRA3-40/SRB3-40/SRM3-40) — reported by the scan, lower confidence.
- **Repo error corrected**: the "SC7 FP300" BM1684X card does not exist.
- **Leads not yet followed**: tpu-mlir carries backends for **SGTPUV8**, **SG2380** and **BM1684X2**.
