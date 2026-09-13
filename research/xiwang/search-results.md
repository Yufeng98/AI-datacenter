# Xiwang (曦望 / Sunrise) — Search Results

**Device class:** AI Accelerator (China, 曦望)
**Research date:** 2026-04-05 (baseline; no search-results file was produced at the time)
**This file created:** 2026-08-08, from the major update scan

---

## Note on this file

The 2026-04-05 research pass did not leave a `search-results.md` for Xiwang. This file is created at the 2026-08-08 update and catalogues **all** resources currently known for the chip, with the baseline-era sources listed first for continuity and the newly discovered ones marked. The single most consequential finding is that the baseline's conclusion "no open-source repositories identified" was **wrong** — the FlagTree `sunrise` backend had been public since 2026-01-23.

---

## Layer 1 — Vendor primary sources

| Resource | Type | Date | Confidence | Notes |
|---|---|---|---|---|
| https://sunrise-ai.com/ | Vendor homepage | current (CMS assets republished 2026-08-06) | high for claims-as-claims | Five hardware product lines (启望 GPUs / 智望 cards / 辰望 servers / 寰望 supernodes / 熙望 workstations) plus **SIRE** software stack. Claims: "5X 单芯片性能提升", "100% Host-Device 互连带宽提升", 98–99% operator efficiency, 90% token cost reduction, 国内首款挂载 LPDDR6 的 GPU, 国内首用 PCIe Gen6 的 GPU. Footer: **浙江曦望智能科技股份有限公司**, 浙ICP备2025183180号-1. |
| https://sunrise-ai.com/products/s3-product | Vendor product page | current | high for vendor statements | **NEW 2026-08-08.** Primary source for: LPDDR6 (向下兼容 LPDDR5X); PCIe Gen6; precisions FP32/FP16/FP8/INT8/FP4; FP4 peak "P 级" only; **仿真实测** GEMM 99% / FlashAttention 98%; 寰望 SC3-256 超节点 (liquid-cooled, PD-disaggregated, large-EP, 万亿乃至更大参数量 multimodal MoE, ">20× throughput" footnoted 数据来源于曦望实验室); SIRE four-layer diagram. **No schedule, no status, no chip count for SC3-256.** |
| https://sunrise-ai.com/products/s2-product | Vendor product page | current | high for vendor statements | **NEW 2026-08-08.** Primary source for: 自研 **TANG 编程模型** + **SIRE** software stack; **SRLink** scale-up (自研 SRLink 等高速互联技术); 智望 **S2-X1** PCIe card and 智望 **S2-M1 OAM module**; 辰望 servers A4-2N1 / A4-2N2 / N8-2G1 / N8-2G2 / A16-2G1; frameworks PyTorch / vLLM / SGLang / LightLLM / LightX2V; 规模化量产 / 万片级量产; S2 repositioned as inference + fine-tuning GPGPU. |
| https://sunrise-ai.com/news | Vendor news index | undated items | medium | **NEW 2026-08-08.** Carries 曦望完成近30亿融资 (Jan 2026) and 推理 GPU 独角兽曦望再获超 10 亿元融资 (**undated**; feed position implies H1 2026). Also carries 曦望全栈适配 FlagOS 2.1 — **version string unconfirmed, do not publish**. |

---

## Layer 2 — Open source (all NEW at 2026-08-08; the baseline recorded none)

| Resource | Type | Date | Confidence | Notes |
|---|---|---|---|---|
| https://github.com/FlagTree/flagtree | Open-source repo | backend added **2026-01-23**; Triton 3.6 **2026-06-30** | high | BAAI FlagOS's unified Triton fork. README vendor table lists "Sunrise（曦望芯科）" → `third_party/sunrise`. **Was public before the 2026-04-05 baseline and was missed.** |
| https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/compiler.py | Source file | current | high | `backend_name='tang'`; target `tang:S2`; LLVM triple `stcu-unknown-tang`; S3 via `stcuv2` + `*_S3.bc`; `warp_size=32`; `num_warps=4`, `num_stages=3`; `supported_fp8_dtypes=('fp8e5',)`; `min_dot_size` (8,8,16) INT8 / (8,8,4); dot input precision `ieee`; `clang-offload-bundler` under `/usr/local/tangrt/toolchains/llvm/`. **First inspectable microarchitecture evidence for this chip.** |
| https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/driver.py | Source file | current | high | `libtang.so`, `libtangrt_shared`, runtime root `/usr/local/tangrt`. |
| https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/language/tang/libdevice.py | Source file | current | high | Device math library built on AMD-style **OCML** (`__ocml_*`). |
| https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-sunrise | Vendor/project docs | current | high | Sunrise Triton plugin v0.4.0 (Triton 3.4) / v0.6.0 (Triton 3.6); wheels `flagtree===0.6.0+sunrise3.6` from `resource.flagos.net`; backend **"Available for S2"**; links closed-source `sunriseTritonPlugin.so`. |
| https://raw.githubusercontent.com/FlagOpen/FlagCX/main/README.md | Open-source repo README | current | high | Names **PCCL — "Sunrise Collective Communications Library"** (links sunrise-ai.com). Support matrix: send/recv, broadcast, reduce, allreduce, allgather, reducescatter, group ops in homogeneous **and** heterogeneous modes; **gather / scatter / alltoall / alltoallv NOT supported**. |
| https://raw.githubusercontent.com/FlagOpen/FlagCX/main/makefiles/sunrise.mk | Build file | current | high | `USE_SUNRISE`, `DEVICE_HOME=/usr/local/tangrt`, `-ltangrt_shared`, `CCL_HOME=/usr/local/pccl`, `-lpccl`, `-DUSE_SUNRISE_ADAPTOR`. |
| https://github.com/FlagOpen/FlagCX/commits/main | Commit history | 2026-06-01, 2026-06-10 | high | 2026-06-01 "[PAL] Add torch plugin support for Sunrise"; 2026-06-10 runtime vendor detection replacing the `USE_SUNRISE_ADAPTOR` ifdef. |
| https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/README.md | Open-source repo README | undated | medium-high | Sunrise listed as a supported chip vendor. |

---

## Layer 3 — Archive snapshots (dating control, NEW 2026-08-08)

| Resource | Date | Notes |
|---|---|---|
| http://web.archive.org/web/20251211220612id_/https://sunrise-ai.com/ | 2025-12-11 | **No SIRE, no LPDDR6, no PCIe Gen6, no SC3-256.** S3 claim reads 推理性能 x3 倍+ — this is the source of the repo baseline's "+300%". |
| http://web.archive.org/web/20260521020029id_/https://sunrise-ai.com/ | 2026-05-21 | SC3-256 appears in the navigation as a **placeholder link**; still no SIRE, no LPDDR6, no PCIe Gen6. |

These two snapshots establish that the SIRE branding and the LPDDR6 / PCIe Gen6 disclosures are genuinely post-baseline (appearing between 2026-05-21 and 2026-08), while the FlagTree integration was already public on 2026-01-23.

---

## Layer 4 — Chinese trade press (baseline-era; attribute, do not treat as spec)

| Resource | Type | Date | Confidence | Notes |
|---|---|---|---|---|
| https://www.qbitai.com/2026/01/373113.html | Trade press (量子位) | 2026-01 | medium | S3 announcement coverage. |
| https://www.qbitai.com/2025/06/303355.html | Trade press (量子位) | 2025-06 | medium | S2 financing coverage (近10亿元). |
| https://caifuhao.eastmoney.com/news/20260123202536017296970 | Trade press (东方财富) | 2026-01-23 | medium | Titled 近**30亿**元融资 = nearly RMB **3 billion**. **This is the source the repo baseline misread as ¥30 billion.** |
| https://www.ofweek.com/ai/2026-01/ART-201700-8420-30680307.html | Trade press (OFweek) | 2026-01 | medium | Same ~RMB 3B funding story. |
| https://zhidx.com/p/489508.html | Trade press (智东西) | 2025 | medium | 杭州GPU黑马 profile. |
| https://finance.sina.com.cn/tech/roll/2025-07-01/doc-infcxsmt5575943.shtml | Trade press (新浪财经) | 2025-07-01 | medium | S2 reveal coverage. |
| https://finance.ifeng.com/c/8l5eKqTHNhb | Trade press (凤凰网财经) | 2025 | medium | 2026年量产第三代 framing. |
| https://blog.csdn.net/suanlix/article/details/149097721 | Blog aggregation | 2025 | low | S2-vs-A100 summary; aggregator, not primary. |
| https://zhuanlan.zhihu.com/p/1924148927080432369 | Community post | 2025 | low | Funding commentary. |
| https://36kr.com/p/3661162854146696 | Trade press (36氪) | 2025 | low | Sector-wide piece. |

---

## Layer 5 — Searches that returned nothing

- **MLPerf** — no Xiwang submissions found in any round.
- **Hot Chips / ISCA / ISSCC / MICRO / ASPLOS** — no Xiwang papers or talks found. Hot Chips 38 (2026-08-23…25) is 15 days in the future as of this scan and has **no scheduled Xiwang talk**; its program is not evidence of anything for this vendor.
- **Independent benchmarks** — none. Every performance number for either part is a company claim, and the S3 operator-efficiency figures are explicitly simulation-derived.
- **ISA documentation / programming guide** — none published. The FlagTree backend is the only public source of ISA-adjacent parameters.
- **Company registry evidence** for the 浙江曦望智能科技股份有限公司 vs 杭州曦望芯科智能科技有限公司 entity question — not obtained.
- **WeChat-hosted executive interviews** (the source for the S3 tape-out schedule) — access-gated; could not be verified.

---

## Layer 6 — Newly named components and terms to track

- **SIRE** — Sunrise Integrated Running Environment; the full-stack brand.
- **TANG** — the programming model (`backend_name='tang'`, target `tang:S2`).
- **TangRT** — the CUDA-runtime analog; `/usr/local/tangrt`, `libtangrt_shared`.
- **PCCL** — Sunrise Collective Communications Library; `/usr/local/pccl`.
- **SRLink** — proprietary scale-up interconnect (S2).
- **stcu / stcuv2** — LLVM target triple families (S2 / S3).
- **sunriseTritonPlugin.so** — the closed compiler core behind the open FlagTree shim.
- **寰望 SC3-256 超节点** — S3 rack-scale product.
- **启望 / 智望 / 辰望 / 寰望 / 熙望** — the five hardware product families.

---

## Layer 7 — Open items for the next scan

1. **Did S3 tape out?** The January 2026 executive statement said mid-2026. No source after April 2026 confirms it. The vendor's S3 page still publishes no status.
2. **PCCL alltoall/alltoallv.** Absent from FlagCX's support matrix, yet the SC3-256 is marketed for large-EP MoE. Watch for the ops appearing, or for evidence that EP dispatch uses a different path.
3. **S3 numbers.** Any capacity, bandwidth, TDP, node or FLOPS disclosure would be the single most valuable addition — none currently exist.
4. **SC3-256 chip count.** Confirm or refute the "256" implied by the product name; the vendor page never states it.
5. **Corporate entity.** Registry check on 浙江曦望智能科技股份有限公司 vs 杭州曦望芯科智能科技有限公司.
6. **The undated >¥1B round.** Establish its date and the resulting cumulative total.
7. **"FlagOS 2.1".** The vendor claims full-stack adaptation to a FlagOS version the FlagOS org page does not advertise. Resolve the version string before publishing it.
8. **FlagTree S3 enablement.** The backend currently advertises S2 only; watch for the manual to add S3, which would be the first third-party signal that S3 silicon exists.
