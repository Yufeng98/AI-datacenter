# VastaiTech (瀚博半导体) — Search Results

*as_of: 2026-09-13*
*chip: vastaitech*
*device_class: GPU-like Inference Accelerator + Video Codec (China, 瀚博 VastaiTech)*

---

## Scope

Datacenter PCIe cards only: **VA1, VA10, VA1V, VA1L, VA12, VA16, VA10L** and the **南禺 / Nanyu VG-series** datacenter graphics cards. The edge/embedded parts (VE1S, VE1M, VE1V, VS1000) are out of scope except where their published numbers illuminate the shared silicon.

> **The single biggest research trap for this vendor:** `vastaitech.com` is stale. Its milestone timeline and awards list both stop in 2023, and the Nanyu nav entry is a dead placeholder. The **current 2026 product line — VA16, VA10L, VA1L — appears nowhere on the marketing site.** It is documented only in `github.com/Vastai`, the `vllm-vacc.vastaitech.com` recipe site, and 2025–2026 Chinese press. Anyone researching this chip from the website alone will document the wrong product generation.

---

## Search Queries

Chinese-language queries were run first and carry most of the yield; English coverage of this vendor is thin and garbles specs.

1. 瀚博半导体 VA16 加速卡 规格 显存 带宽
2. 瀚博半导体 SG100 GPU 工艺 制程 参数
3. 瀚博半导体 上市辅导备案 中信建投 / 中信证券
4. 瀚博 SV100 达景 芯片 7nm 工艺
5. 瀚博半导体 乾元 SG100 GPU 发布
6. 瀚博半导体 载天 VA16 算力 TOPS 带宽
7. 瀚博半导体 载天 VA10L VA1L 产品
8. 瀚博半导体 VA10 数据中心推理卡 400TOPS 150W
9. 瀚博半导体 VA1 推理卡 INT8 TOPS 7nm SV100
10. 瀚博 VA1 32GB LPDDR4x 或 GDDR6 显存 带宽 PCIe4.0
11. 瀚博半导体 南禺 数据中心显卡 SG100
12. 瀚博半导体 2023 WAIC SG100 VG1600 VG1800 VA1L VA12 发布
13. 瀚博 VA1L 200TOPS 72TFLOPS 512GB 一体机 1750亿
14. 瀚博半导体 VGX VA16 AIGC大模型一体机 DeepSeek 671B 部署
15. 瀚博半导体 2024中国算力大会 VGX VA16 一体机 2TB 405B
16. "VA16" 瀚博 显存 128GB 四芯 OR 四颗 OR 多Die  *(no results)*
17. 瀚博半导体 第三代 芯片 SV200 OR SG200 OR 新一代GPU 2025  *(no results)*
18. 瀚博 SV100 达景 架构 张量 引擎 解码器 芯片 特点 芯东西
19. 瀚博半导体 钱军 对话 首款云端GPU 芯东西 / zhidx 专访
20. 快手 瀚博半导体 投资 客户 VA1 部署
21. 瀚博半导体 IPO 辅导备案 上海证监局 2025年7月 中信证券
22. 瀚博半导体 2026 中标 智算中心 部署 客户 千卡
23. 瀚博半导体 海马云 千卡集群 战略合作
24. Vastai VastaiTech SV100 SG100 GPU China inference accelerator 7nm specifications
25. 瀚博 VA1 加速卡 参数 规格表 内存带宽 GB/s  *(nothing found — confirms non-disclosure)*

Direct-crawl passes, in addition to search:

- `vastaitech.com` Next.js `__NEXT_DATA__` extraction over `/product/*`, `/software/*`, `/solution/*`, `/company/about` (both ZH and EN trees)
- `developer.vastaitech.com` SPA bundle + `/api/*` endpoint enumeration
- `git clone` of all four `github.com/Vastai` repos + full-tree grep for compiler flags, environment variables, device APIs, and VDSP / VCCL / VNNL references

---

## Resources Found

### Official Product / Company Pages (vendor primary)

| Resource | URL | Category |
|----------|-----|----------|
| About — company history and milestone timeline (SV tape-out 2021-03, VA1 2021-07, VUCA + VA10 + VE1 2022-09, SG100 silicon 2023-02, funding rounds) | https://www.vastaitech.com/company/about | Company |
| 载天 VA1 product page (70 W HHHL, 120 ch 1080p30, 8K codec, VUCA description) | https://www.vastaitech.com/product/general/va1 | Hardware Spec |
| 载天 VA10 product page (150 W FH 3/4-length, 240 ch 1080p30) | https://www.vastaitech.com/product/general/va10 | Hardware Spec |
| 载天 VA1V video card (H.264/H.265/AV1, 8K 10-bit, 2× 8K HDR@60+fps, FFmpeg) | https://www.vastaitech.com/product/video/va1v | Hardware Spec |
| 载天 VE1V video card (1× 8K HDR@60+fps) — edge, for silicon cross-reference | https://www.vastaitech.com/product/video/ve1v | Hardware Spec |
| 智能一体机 (AI appliance) solution page | https://www.vastaitech.com/solution/all-in-one | Solution |
| Support page — 3-year hardware + software warranty, formal RMA process | https://www.vastaitech.com/support | Maturity evidence |

### Software Pages (vendor primary)

| Resource | URL | Category |
|----------|-----|----------|
| VastStream SDK — VACM / VACE / VACL / VAME / VAML decomposition, FFmpeg VAAPI plugins | https://www.vastaitech.com/software/vaststream | Software |
| VastStream (English) — same content, useful for the library-name expansions | https://www.vastaitech.com/en/software/vaststream | Software |
| VastCloudNative — Kubernetes Operator, device plugin, Exporter, Docker OCI plugin | https://www.vastaitech.com/software/vastcloudnative | Software |
| VastDCManager — VASMI, VAProfiler, VASID; built on VADriver + VAML; Prometheus | https://www.vastaitech.com/software/vastdcmanager | Software |
| Developer centre — **gated**; API returns `{"code":4000,"message":"错误的账户或秘钥，请检查"}` on every product/documentation route | https://developer.vastaitech.com/ | Negative result |

### Open-Source Repositories (github.com/Vastai — 4 repos, all active in 2026)

| Resource | URL | Category |
|----------|-----|----------|
| VastModelZOO — Apache-2.0; 1000+ models, VVI install flow, VAMC configs, deploy docs | https://github.com/Vastai/VastModelZOO | Model Zoo |
| VastModelZOO public model index (5 domains, 33 sub-categories, 2 runtimes) | https://vastai.github.io/VastModelZOO/ | Docs |
| VAMC compiler config schema (`backend.type: tvm_vacc`, `tp`, `gather_data_vccl_dsp_enable`) | https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_config.yaml | Compiler |
| VAMC quantisation config (`w8a16_gptq`, calibration datasets) | https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_quant.yaml | Compiler |
| LLM compile guidance (seq-len multiple of 16, `VACC_STACK_SIZE=256`, `b2s`, `align_qkv`) | https://github.com/Vastai/VastModelZOO/blob/main/llm/README.md | Compiler |
| Qwen2 Build_In deploy doc (`modeling_qwen2_vacc.py`, `config_vacc.json`, `insert_slice`, eager attn) | https://github.com/Vastai/VastModelZOO/blob/main/llm/qwen2/README.md | Graph capture |
| Qwen2 VAMC compile YAML (`tp: 4`, `b2s`, `model_arch: vacc`, `tvm_vacc`) | https://github.com/Vastai/VastModelZOO/blob/main/llm/qwen2/build_in/build/hf_qwen2_fp16.yaml | Compiler |
| DeepSeek-V3/V3.1 deployment (single VA16 server, TP32, TP32-PP2, MTP, max-concurrency 4) | https://github.com/Vastai/VastModelZOO/blob/main/llm/deepseek_v3/README.md | Runtime |
| vLLM backend usage limits (max-concurrency 4 across all models; `min_p` unsupported) | https://github.com/Vastai/VastModelZOO/blob/main/tools/vllm/usage_limits.md | Runtime limits |
| vLLM Dockerfile (VNNL / VCCL / VACM env vars, `torch_vacc` 1.3.0, `vllm_vacc` 0.7.2) | https://github.com/Vastai/VastModelZOO/blob/main/llm/common/docker/Dockerfile | Runtime |
| Dockerfile.arm (aarch64 wheels) | https://github.com/Vastai/VastModelZOO/blob/main/llm/common/docker/Dockerfile.arm | Runtime |
| vLLM examples + sampling-parameter support matrix (torch 2.7.0+cpu, vLLM 0.9.2, Py 3.12) | https://github.com/Vastai/VastModelZOO/blob/main/llm/common/examples/README.md | Runtime |
| VastGenX serving tool (`--llm_devices` / `--vit_devices` are **dies**; OpenAI API + WebUI) | https://github.com/Vastai/VastModelZOO/blob/main/tools/vastgenx/README.md | Serving |
| **GLM-OCR vLLM README — NVIDIA H800 vs VACC-VA16 OmniDocBench head-to-head timings** | https://github.com/Vastai/VastModelZOO/blob/main/vlm/glm_ocr/vllm/README.md | Vendor-published benchmark |
| ResNet deploy doc (`vamc compile`, `vamp`, keras/onnx frontends, int8 vs fp16 accuracy) | https://github.com/Vastai/VastModelZOO/blob/main/cv/classification/resnet/source_code/keras.md | Compiler flow |
| VastStreamX-Samples — MIT; release 26.04 (2026-05-09), VVI-26.02, OCLK/DCLK/ECLK clocks | https://github.com/Vastai/VastStreamX-Samples | Samples |
| card_info sample — **`VA1-16G`, Die ID 0,1; 8 GB/die; `util{ai, vdsp[], vdmcu[], vemcu[]}`** | https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/card_info/card_info.cpp | Hardware telemetry |
| model_base.hpp (Model / ModelOperator / Operator / Graph / Stream / StreamBalanceMode) | https://github.com/Vastai/VastStreamX-Samples/blob/main/common/model_base.hpp | Tensor API |
| custom_op_base.hpp (`CustomOperator(op_name, elf_file)`) | https://github.com/Vastai/VastStreamX-Samples/blob/main/common/custom_op_base.hpp | Kernel interface |
| planar_argmax custom op (ELF at `/opt/vastai/…/elf/`, ctypes struct, ≤96 channels) | https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/vdsp_op/custom_op/argmax/argmax_op.hpp | Kernel interface |
| video_decode sample (per-die H.264 1547 fps / H.265 1861 fps @ DCLK=650 MHz) | https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/video_decode/README.md | Measured perf |
| VDSP fusion-op JSON (`FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR`) | https://github.com/Vastai/VastStreamX-Samples/blob/main/data/configs/detr_bgr888.json | Op config |
| xinference_vacc — Xinference integration, vLLM engine only, `VACC_VISIBLE_DEVICES`, `VNNL_CONV1D_DLC` | https://github.com/Vastai/xinference_vacc | Framework |
| Vastai/MinerU fork — driver `00.25.12.30`, `torch-vacc` 1.3.3.777, `vllm-vacc` 0.11.0.777, Hygon C86; TP supported / **DP not supported**; `--enforce_eager` mandatory | https://github.com/Vastai/MinerU | Framework |

### Third-Party / Independent Documentation

| Resource | URL | Category |
|----------|-----|----------|
| OpenDataLab MinerU — upstream, third-party-hosted VastAI acceleration-card docs (the only genuinely independent integration doc found) | https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/VastAI | Third-party docs |
| 通泰易 (2025-06-13) — server compatibility certification: TG657V2 / TG658V3 / TG659V2 with the **VA16 "训推一体" card** | http://ttyinfo.com/News/info/id/154.html | OEM certification |

### vLLM Recipe Site (vendor primary — the current-product source of truth)

| Resource | URL | Category |
|----------|-----|----------|
| vLLM × VastAI recipes — "Community-maintained recipes for VASTAI Tech VA16 VA10L VA1L", 31 recipes, vLLM 0.17.0 | https://vllm-vacc.vastaitech.com/ | Framework |
| DeepSeek-V3 (8× VA16, 128 GB per card = 4 × 32 G, TP32, FP8) | https://vllm-vacc.vastaitech.com/deepseek-ai/DeepSeek-V3 | Hardware config |
| DeepSeek-V3.1 (8× VA16, TP32, FP8, 131072 ctx, MTP) | https://vllm-vacc.vastaitech.com/deepseek-ai/DeepSeek-V3.1 | Hardware config |
| **Qwen3-32B — "VA16128G (4×32G) / VA10L128G (4×32G)"**, BF16 / FP8 / GPTQ-INT8 / INT4 | https://vllm-vacc.vastaitech.com/Qwen/Qwen3-32B | Hardware config |
| Qwen3-235B-A22B (4× VA16, TP16, FP8) | https://vllm-vacc.vastaitech.com/Qwen/Qwen3-235B-A22B | Hardware config |
| Qwen3-Embedding (VA16128G / VA10L128G, TP1/2/4/8) | https://vllm-vacc.vastaitech.com/Qwen/Qwen3-Embedding | Hardware config |
| Qwen3-0.6B (1× VA16 or VA10L128G, TP1–8) | https://vllm-vacc.vastaitech.com/Qwen/Qwen3-0.6B | Hardware config |

### Press (primary launch coverage)

| Resource | URL | Category |
|----------|-----|----------|
| 腾讯新闻 / QQ (2021-07-08) — SV100 + VA1 launch: >200 TOPS INT8, FP16/BF16/INT8, 32 GB, 75 W, PCIe 4.0 ×16, 64+ ch 1080p H.264/H.265/AVS2 | https://news.qq.com/rain/a/20210708A032OI00 | Launch coverage |
| 界面新闻 (2021) — SV100 is a **DSA**; VA1 based on **SV102**; founders' AMD background; Kuaishou as customer + investor | https://m.jiemian.com/article/6342672.html | Press |
| 芯东西 zhidx (2022-09-08) — CEO interview; **best public description of the on-die block structure** | https://zhidx.com/p/344936.html | Architecture |
| 36Kr (2022-09-05) — VA10 launch: 400 TOPS INT8, 150 W, 100 ch 1080p30 transcode; VE1 100 TOPS 40–65 W | https://www.36kr.com/p/1901732567984512 | Launch coverage |
| 量子位 QbitAI (2023-07) — WAIC 2023: SG100 7 nm; 南禺 VG1600/VG1800/VG14; VA1L 200 TOPS / 72 TFLOPS; VA12 250 W, 512 TOPS / 160 TFLOPS | https://www.qbitai.com/2023/07/66614.html | Launch coverage |
| 36Kr (2023-07-06) — independent confirmation of the same WAIC 2023 numbers | https://www.36kr.com/p/2332635957528066 | Launch coverage |
| eeNews Europe — "China's Vastai launches 7nm GPU for AI, visual apps" (SG100) | https://www.eenewseurope.com/en/chinas-vastai-launches-7nm-gpu-for-ai-visual-apps/ | Press (English) |
| 上海证券报 via Sina Finance (2026-04-27) — **载天 VA16 128 GB, FP4 + FP8, DeepSeek-V4 adaptation, up to 2 TB appliance** | https://finance.sina.com.cn/roll/2026-04-27/doc-inhvxtrc2314774.shtml | Current product |
| 同花顺 10jqka (2026-04-27) — same VA16 / DeepSeek-V4 story, Chinese quotes verbatim | https://news.10jqka.com.cn/20260427/c676308817.shtml | Current product |
| iCloudNews (2024-04-10) — 海马云 × 瀚博 strategic partnership, **thousand-card** cloud-rendering cluster on domestic ARM + VastaiTech GPUs | https://www.icloudnews.net/a/79464.html | Partnership |

### Regulatory

| Resource | URL | Category |
|----------|-----|----------|
| **CITIC Securities IPO counselling progress report (Period 1) — PRIMARY FILING**: agreement 2025-07-11, counselling period from 2025-07-18, A-share target, >5% shareholders enumerated | https://www.cs.ecitic.com/newsite/tzgg/ipoqyfdgg/202510/P020251023518196738538.pdf | Primary filing |
| SSE STAR Market IPO review database — **NEGATIVE RESULT**: no 瀚博 record among all 1,046 filings (consistent with counselling-stage-only status) | https://query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB | Negative result |

---

## Key Findings

- **Two chip generations in mass production, both 7 nm.** **SV100** (达景 "Dajing"; the VA1 SKU is **SV102**) is a server-class AI-inference **DSA — explicitly not a GPGPU**; tape-out 2021-03, mass production 2022-Q1. **SG100** (乾元 "Qianyuan") is a **full-function GPU** with graphics (DX11/OpenGL/Vulkan), AI and video on one die; silicon back 2023-02, mass production 2023-04. A third generation is reported to be in development; nothing about it is disclosed.
- **Foundry is never named for either 7 nm part.** Do not assume TSMC.
- **VUCA** (*Vast(ai) Unified Compute Architecture*, Sept 2022) is the umbrella name. The best public block-level description is a 2022 CEO interview: a high-performance compute engine, a high-performance AI engine, a **programmable vector compute engine (the VDSP)**, dedicated video-decode and graphics render/display cores, and unified memory management with coherent interfaces and low-latency interconnect.
- **A card is N independent dies, each with its own memory pool.** `vsx::Card::GetAllDies()` enumerates them; the `card_info` sample shows `VA1-16G` = **2 dies × 8 GB**, and the vLLM recipes state **`VA16128G (4×32G)`** and **`VA10L128G (4×32G)`**. This is why DeepSeek-V3 on 8× VA16 runs at **TP=32**. Whether a "die" is a separate silicon die in an MCM or a partition of a monolithic die is **not disclosed**.
- **The SDK telemetry corroborates the block structure**: `util.ai` (scalar), `util.vdsp[]`, `util.vdmcu[]` (decode), `util.vemcu[]` (encode) — one AI domain per die, arrays of VDSP cores and codec MCUs. The array sizes come from a non-public header.
- **Three clock domains**: OCLK (operator/AI, 835–880 MHz in samples), DCLK (decode, 650 MHz), ECLK (encode, 200 MHz), with user-controllable DPM.
- **Zero absolute memory specs exist.** Memory **technology** (HBM/GDDR/LPDDR/DDR) and memory **bandwidth** are undisclosed for **every VastaiTech part ever shipped**, in Chinese and English alike. Only capacities are public.
- **The graph compiler VAMC is TVM-based, not MLIR-based** — all 344 compile configs in the model zoo set `backend.type: tvm_vacc`. This is the most-often-missed fact about the stack.
- **Two parallel runtime paths that share nothing above the driver**: the AOT-compiled **Build_In** path (VAMC → VastStream/VastStreamX → VastGenX) and the **vLLM** path (`torch_vacc` + `vllm_vacc`), which is where all 2026 LLM/VLM work happens and which mandates `--enforce-eager`.
- **No public kernel compiler, kernel language, ISA, or intrinsics.** VDSP custom operators ship as prebuilt ELF binaries; the toolchain that produces them is not distributed.
- **The vendor published its own unfavourable benchmark.** The GLM-OCR OmniDocBench table shows VACC-VA16 matching H800 on accuracy (95.6 vs 95.4) while taking **13 h 26 min vs 15.5 min** at TP1 — roughly 50× slower, ~25× at TP4. Single workload, but it is apples-to-apples and vendor-published.
- **Self-disclosed maturity limit**: `max-concurrency 4` across essentially every vLLM model, including DeepSeek-V3 and small BGE embedding models; no data parallelism; mandatory eager mode.
- **Maturity verdict: shipping / commercially available, NOT verified deployed at scale.** No named end customer with a disclosed deployment size, no third-party benchmark, no ISCA / MICRO / Hot Chips / ISSCC / arXiv publication of any kind.
- **IPO status**: entered **IPO counselling** (辅导备案) with CITIC Securities on 2025-07-18, targeting a domestic A-share listing. This is the pre-application stage — **do not write that VastaiTech has filed an IPO application.** The SSE STAR Market negative result is consistent with, not contradictory to, this.

---

## Resources Added 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*No hardware/silicon disclosure this window. Two items: a new VastStreamX SDK release that predates the window but was absent from the prior pass, and a re-verification of IPO status.*

- **[new 2026-09-13]** [Vastai/VastStreamX-Samples release `26.08`](https://github.com/Vastai/VastStreamX-Samples/releases/tag/26.08) — published **2026-08-03** (verified via GitHub API `releases`; 5 days before the 2026-08-08 baseline scan and missed at that pass). Supersedes the previously recorded `26.04` (2026-05-27, not 2026-05-09 as the baseline stated — corrected here against the live API). Commit activity in the repo continued steadily through the window (bug fixes: `dbnet_detector` max_candidates, mini-box classification, rotate-angle init; a performance-improvement commit 2026-09-11; docs updates through 2026-09-14) — routine maintenance, no new hardware surface.
- **[new 2026-09-13]** [Vastai/VastModelZOO — repo activity](https://github.com/Vastai/VastModelZOO) — `pushed_at` **2026-08-31** (verified via GitHub API); recent commits are model-doc fixes (GLM-OCR eval doc, vamp data format) — no new model family added in-window.
- **[new 2026-09-13]** [SSE STAR Market IPO review database — re-queried live (`query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB`)](https://query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB) — now **1,048 total filings** (up from 1,046 at the 2026-08-08 pass); **no 瀚博/VastaiTech record found**, confirming the counselling-stage-only status is still current as of 2026-09-13.

## Searched and Not Found (2026-09-13)

- No VA22/VA1L successor or third-generation-chip specification.
- No new VastStream (legacy C SDK) release.
- No STAR Market filing progression beyond counselling.
- No Hot Chips 38 (2026-08-23 → 08-25) VastaiTech talk.
