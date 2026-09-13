# Tecorigin (太初元碁) SDAA — Search Results

*as_of: 2026-08-08*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*

---

## Methodological Note — read this before trusting any coverage claim

Two constraints shaped this scan and both must travel with the results:

1. **WebSearch was unavailable for the entire research pass.** The budget reported 200/200 consumed on the first call, so **zero keyword searches ran**. Everything catalogued below was reached by *direct URL retrieval*. Consequently there is **no press coverage, no analyst report, no funding record, and no export-control check** in this scan. That evidence class is unexamined for the second consecutive pass and is the top priority for any follow-up.
2. **The vendor documentation portal is machine-retrievable after all** — correcting the prior seed, which recorded `docs.tecorigin.com` as a Vue SPA that is "NOT agent-fetchable". The SPA is backed by a REST API at `http://docs.tecorigin.com/api/api/…` (the `api` segment is doubled because the axios client sets `baseURL:"/api"` and then requests relative `"api/…"` paths). Content is returned as **Yjs CRDT binary updates**, not HTML. Naive byte-scraping of the CRDT interleaves edit history and yields **scrambled digits** — an early heuristic pass produced "2 TFLOPS" where the true text reads "29 TFLOPS". All numbers in this research set were recovered by decoding the CRDT with `pycrdt`.

Via that route, **17 complete manuals (~3.2 MB of Chinese-language primary vendor documentation)** were extracted, including six documents the seed did not know existed: the **PCX virtual ISA guide**, **TecoSMI**, **SDAARuntime**, **TSight**, **SDPTI**, and **TecoGDB**.

---

## Search Queries

**WebSearch returned no results — the two intended Chinese-language queries were blocked by an exhausted budget and are recorded here as attempted, not run.**

1. `太初元碁 SDAA 加速卡 T100 参数 算力` — **ATTEMPTED, BLOCKED** (WebSearch budget exhausted)
2. `太初元碁 元碁T110 OAM 显存 HBM 功耗` — **ATTEMPTED, BLOCKED** (WebSearch budget exhausted)

### Direct-retrieval probes run instead

| # | Probe | Outcome |
|---|-------|---------|
| 1 | `curl http://docs.tecorigin.com/release/sdaac/v3.1.0/` | Vue SPA shell only — confirms the seed's surface observation |
| 2 | `curl http://docs.tecorigin.com/assets/index-8cb1dfea.js` | Extracted the API route table from the JS bundle |
| 3 | grep bundle for `api/` template literals | Found `getReleaseTree`, `getReleasedContent`, `docContentList`, `getReleasedVersionsPublic`, `homepage`, `searchAllDoc` |
| 4 | `/api/checkConnect/` vs `/api/api/checkConnect/` | Discovered the **doubled-prefix** base path |
| 5 | `assets/DocContentView-139b7cb7.js` | Resolved `getReleaseTree` params `{group, version}` |
| 6 | `/api/api/homepage/?page_size=100` | Enumerated all documentation sets, versions and FAQ entries |
| 7 | `getReleaseTree` across candidate group names | **18 hits**: sdaac, pcx, tecosmi, tecogdb, tecogdbvisual, software_installation, sdaac_beginner_guide, sddac_perf_opt, op_perf_opt, torch_2.4, torch2.7, tecopaddle, teco_vllm, tecoinferenceengine, teco_megatron_lm, sdaart, sdpti, tsight. **404** for tecoal, tecodnn, tecoblas, tccl, tcvs, tecolmk, tecoexporter, tecocustom, tecorand, tsight_cli, tcml |
| 8 | `docContentList/<root_id>/` per group → `pycrdt` decode | Recovered exact text and numerals from the Yjs payloads |
| 9 | grep of decoded corpus | Targets: TFLOPS, GB/s, 带宽, bfloat/bf16, 数据类型, Memory-Usage, 环网, Mpe, 拓扑, topo, P2P, 缓存, CORE_NUM, 从核, 235KB, matmul, MMA, nm工艺/台积电/中芯 |
| 10 | `raw.githubusercontent.com/Tecorigin/teco-ops/main/{README.md, doc/teco-ops-hardware.md, doc/README_OP.md, doc/README_PLUGIN.md, doc/QA.md}` | Retrieved |
| 11 | `raw.githubusercontent.com/PaddlePaddle/PaddleCustomDevice/develop/backends/sdaa/README.md` | Retrieved (independent source) |
| 12 | `gitee.com/tecorigin` HTML + `gitee.com/api/v5/repos/tecorigin/<repo>` × 8 | Retrieved star/fork/license/push metadata |
| 13 | `api.github.com/orgs/Tecorigin/repos` | **429 rate-limited** — fell back to raw.githubusercontent.com |
| 14 | `www.tecorigin.com/cn/{technology,products,about,developer,news}.html` | Retrieved; `/en/technology.html` → **404** (no English site) |
| 15 | `pypi.org/pypi/{torch-sdaa,torch_sdaa,tecovllm,teco-vllm,tecoops,paddle-custom-sdaa}/json` | **all 404** |
| 16 | `huggingface.co/api/models?author=tecorigin` and `?search=tecorigin` | **empty** |

---

## Resources Found

### Documentation Portal — the richest primary source for this vendor

| Resource | URL | Category |
|----------|-----|----------|
| **Docs portal REST API** (undocumented; discovered this scan) | `http://docs.tecorigin.com/api/api/getReleaseTree/?group=<g>&version=latest` → root id, then `http://docs.tecorigin.com/api/api/docContentList/<id>/` → Yjs CRDT binary | Method / Primary access |
| Docs portal (browser UI) | http://docs.tecorigin.com/ | Documentation |
| Docs homepage config API (enumerates all doc sets, videos, FAQ) | http://docs.tecorigin.com/api/api/homepage/?page_size=100 | Discovery |
| SDAA C 编程指南 v3.2.0 (latest v3.3.0, 2026-07-23) | http://docs.tecorigin.com/release/sdaac | Programming Model |
| **PCX 编程指南 v1.2.0 (PCX ISA v1.0.0)** — virtual ISA | http://docs.tecorigin.com/release/pcx | ISA |
| **TecoSMI 用户手册 v1.15.0** | http://docs.tecorigin.com/release/tecosmi | Hardware Spec (telemetry) |
| **SDAARuntime 用户手册 v3.2.0** | http://docs.tecorigin.com/release/sdaart | Runtime Docs |
| 环境安装手册 v3.2.0 | http://docs.tecorigin.com/release/software_installation | SDK Docs (component inventory) |
| TecoPyTorch 用户手册 v3.2.0 (PyTorch 2.7.1) | http://docs.tecorigin.com/release/torch2.7 | Framework Docs |
| TecoPyTorch 用户手册 (PyTorch 2.4) | http://docs.tecorigin.com/release/torch_2.4 | Framework Docs |
| TecoPaddle 用户手册 v3.2.0 | http://docs.tecorigin.com/release/tecopaddle | Framework Docs |
| Teco-vLLM 用户手册 v3.2.0 | http://docs.tecorigin.com/release/teco_vllm | Framework Docs |
| Teco-Megatron-LM 用户手册 v3.2.0 | http://docs.tecorigin.com/release/teco_megatron_lm | Framework Docs |
| TecoInferenceEngine（小模型）用户手册 v3.1.0 | http://docs.tecorigin.com/release/tecoinferenceengine | Compiler / Runtime Docs |
| 性能优化手册-算子篇 v1.1.0 | http://docs.tecorigin.com/release/op_perf_opt | Hardware Spec (MMA shape, 3-level memory, measured TFLOPS) |
| 性能优化手册-SDAA C篇 v2.0.2 | http://docs.tecorigin.com/release/sddac_perf_opt | Hardware Spec (RMA cost model, DMA bandwidth, I-cache counter) |
| SDAA C 零基础入门 v1.1.0 | http://docs.tecorigin.com/release/sdaac_beginner_guide | Programming Model |
| TecoGDB 命令行工具用户手册 v3.1.0 | http://docs.tecorigin.com/release/tecogdb | Debug Docs ("Has 32 GPC", SPA 0–3) |
| TecoGDB 可视化工具用户手册 v1.2.0 | http://docs.tecorigin.com/release/tecogdbvisual | Debug Docs |
| SDPTI 用户手册 v1.7.0 | http://docs.tecorigin.com/release/sdpti | Profiling Docs (CUPTI analogue) |
| TSight GUI 用户手册 v1.9.0 | http://docs.tecorigin.com/release/tsight | Profiling Docs (Nsight analogue) |

### Open-Source Repositories — GitHub

| Resource | URL | Category |
|----------|-----|----------|
| Tecorigin/teco-ops (BSD-3-Clause) — `.scpp` kernels, SDAAC_examples, TVM Relay plugin, PyTorch bindings, CUDA baseline | https://github.com/Tecorigin/teco-ops | Kernel Library |
| teco-ops hardware doc — SPA/SPE/SU/VPU/FU/SPM | https://github.com/Tecorigin/teco-ops/blob/main/doc/teco-ops-hardware.md | Hardware Spec |
| teco-ops README — "SPM 内存申请不超过 235KB", TVM Relay registration | https://github.com/Tecorigin/teco-ops/blob/main/README.md | Hardware Spec |
| teco-ops operator dev guide — SPM ≈235 KB / 240512 B ceiling | https://github.com/Tecorigin/teco-ops/blob/main/doc/README_OP.md | Hardware Spec |
| teco-ops plugin guide — tecocc vs g++, TVM headers | https://github.com/Tecorigin/teco-ops/blob/main/doc/README_PLUGIN.md | Compiler |
| **PaddleCustomDevice `backends/sdaa` (INDEPENDENT, Apache-2.0)** | https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa | Independent corroboration |
| PaddleCustomDevice SDAA README — `/dev/tcaicard0..3`, Docker source | https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/sdaa/README.md | Independent corroboration |
| Tecorigin/modelzoo (BSD-3-Clause) | https://github.com/Tecorigin/modelzoo | Model Zoo |
| Tecorigin/teco-modelzoo (BSD-3-Clause) | https://github.com/Tecorigin/teco-modelzoo | Model Zoo |
| Tecorigin/tecovllm-modelzoo (BSD-3-Clause) | https://github.com/Tecorigin/tecovllm-modelzoo | Model Zoo |
| Committed run log with full stack banner (`.so` names + versions) | https://github.com/Tecorigin/modelzoo/blob/main/PyTorch/contrib/Classification/ACNet-master/scripts/acnet.txt | Runtime / Stack evidence |
| WAIC 2026 competition org (private per-team repos + cluster access) | https://github.com/tecorigin-waic | Ecosystem |

### Open-Source Repositories — Gitee

| Resource | URL | Category |
|----------|-----|----------|
| teco-al (BSD-3-Clause, 31★, 157 forks — competition-inflated) | https://gitee.com/tecorigin/teco-al | Kernel Library |
| teco-torch (BSD-3-Clause, 17★) | https://gitee.com/tecorigin/teco-torch | Framework |
| teco-paddle (Apache-2.0, 19★) | https://gitee.com/tecorigin/teco-paddle | Framework |
| modelzoo-old (BSD-3-Clause, 26★, 91 forks) | https://gitee.com/tecorigin/modelzoo-old | Model Zoo |
| teco-generative-ai (BSD-3-Clause) | https://gitee.com/tecorigin/teco-generative-ai | Model Zoo |
| tcap_dllogger (Apache-2.0) — structured training logger | https://gitee.com/tecorigin/tcap_dllogger | Tooling |
| sdcops (no license declared) | https://gitee.com/tecorigin/sdcops | Kernel Library |
| thirdparty_llm (no license declared) | https://gitee.com/tecorigin/thirdparty_llm | Model Zoo |
| Gitee org index | https://gitee.com/tecorigin | Discovery |

### Vendor Site

| Resource | URL | Category |
|----------|-----|----------|
| Technology / product matrix — T100/T110/T111, T1008/I1004/T1108/T1118, SuperPOD-128 | https://www.tecorigin.com/cn/technology.html | Hardware Spec (marketing) |
| Solutions & deployments — Yancheng 206P, 太湖之光A+, Yan'an, Lihu | https://www.tecorigin.com/cn/products.html | Deployment (marketing) |
| About — founded 2019-11, HQ Hangzhou, NSCC-Wuxi team, 3× Gordon Bell Prize | https://www.tecorigin.com/cn/about.html | Company |
| Developer / model zoo | https://www.tecorigin.com/cn/developer.html | Ecosystem (marketing) |
| News index (2021–2026); legal entity 太初（杭州）集成电路有限公司 | https://www.tecorigin.com/cn/news.html | Company |
| Binary mirrors (Docker tarballs, wheels, MLNX_OFED bundle) | http://mirrors.tecorigin.com/ , http://jfrog.tecorigin.net/artifactory/ | Distribution |

### Negative Results

| Check | Result |
|-------|--------|
| PyPI: `torch-sdaa`, `torch_sdaa`, `tecovllm`, `teco-vllm`, `tecoops`, `paddle-custom-sdaa` | **all 404** — nothing on PyPI |
| Hugging Face: `?author=tecorigin` | **`[]`** — no Hugging Face organization |
| English vendor site `www.tecorigin.com/en/technology.html` | **404** — Chinese-only |
| `getReleaseTree` for tecoal / tecodnn / tecoblas / tccl / tcvs / tecolmk / tecoexporter / tecocustom / tecorand / tcml | **404** — the acceleration libraries and the management library have **no public manual**; only the install manual names them |
| MLPerf / SPEC / third-party benchmark | **none found** (note: no WebSearch, so this is a weak negative) |

---

## Key Findings

- **Chip identity resolved.** T100 / T110 / T111 are **card SKUs**; the **chip is "T1"** — firmware images are `aiflash_v<ver>_T1` (TecoSMI) and `sdaaDeviceProp_t.clockRate` is documented as "T1计算核心SPE的频率" (SDAARuntime). The compiler architecture flag is `--sdaa-arch=pcx_100`, the PCX compatibility table lists only "T100系列". This closes the seed's "no die/chip codename" gap.
- **Topology of the machine.** One card = **4 SPAs** (each an independent SDAA device with its own Global memory) × **32 SPEs per SPA** = **128 SPEs per card**. Both factors are now vendor-stated, not inferred: TecoSMI reports `Minor Number : 0 1 2 3`, TecoPyTorch/Teco-vLLM say "每张加速卡上有4个可用的SDAA计算设备", and TecoGDB prints `Has 32 GPC` with SPE rows 0…31 plus "SPE：大于31时表示特殊模块".
- **Sunway idiom, not Sunway derivation.** The athread-style fingerprint is now confirmed in depth — slave-core array (SPEs are literally 从核 in vendor comments), LDM-like private scratchpad, explicit DMA + inter-core **RMA** + hardware row/column **broadcast**, SPMD with per-core IDs, FP64 in the ISA, and first-class **申威 SW-64 host CPU** support. But **no source states the T1 shares the SW ISA or derives from SW26010**, and there is **no on-device MPE+CPE core-group structure** documented. Keep the seed's downgrade to team/ecosystem/idiom lineage.
- **Matrix unit is 16-bit-only.** PCX `matmul_init` encodes exactly four dtype combinations: FP16→FP16, FP16→FP32, S16→S16, S16→S32. **No BF16, no TF32, no FP8, no INT8 in the matrix unit.** Corroborated three ways: `TORCH_SDAA_BF16_CLIP` clips bf16 GEMM inputs to ±65407 (the FP16 max), SDAA C says bf16 is "仅支持指针操作", and Teco-vLLM's quantization menu is entirely **weight-only** (W8A16 / W4A16 / GPTQ / AWQ / KV-INT8) with no W8A8 path.
- **PCX is the biggest software finding** — a fully documented, PTX-analogous **hardware-independent virtual ISA** with infinite virtual registers, scalar/vector/matrix instruction classes, and a stated purpose of decoupling software from the machine ISA across "多种系列的硬件". Completely absent from the seed. The **T1 machine ISA is not disclosed by design.**
- **Interconnect is conventional.** PCIe Gen4 ×16 host; **no proprietary scale-up link is documented anywhere** — `teco-smi topo`'s legend is exactly the nvidia-smi PCIe set (SYS/NODE/PHB/PXB/PIX) with no NVLink-equivalent entry, and card-to-card is PCIe P2P. Scale-out is standard IB/RoCE, with the toolkit *requiring* MLNX_OFED 5.9-0.5.6.0 + Open MPI 4.1.5rc2 redistributed from `mirrors.tecorigin.com`.
- **Two-package stack.** Everything ships as exactly **TecoDriver** + **TecoToolKit**, version-locked 1:1, currently **v3.2.0**. Built and validated for **five host CPU/OS combinations**: x86_64, Hygon 海光, Phytium 飞腾 (ARMv8), **Sunway 申威 8A (SW-64)**, and Loongson 龙芯 (LoongArch). Shipping a full AI stack on an SW-64 host is close to unique in this registry.
- **Zero peak-performance disclosure.** No TFLOPS/TOPS at any precision, no HBM bandwidth, no process node, no die size, no TDP. The only public throughput number is a **tutorial micro-benchmark** (≈29 TFLOPS FP16 through the matrix unit vs ≈2.5 TFLOPS through vector instructions at a 2.36 GHz SPE clock) which Tecorigin itself states is "远没有达到" peak and whose scope (per-SPE / per-SPA / per-card) is not stated.
- **Openness is bimodal.** The *kernel-source* layer is genuinely open (teco-ops BSD-3, Teco-AL, and the Tecorigin-independent PaddleCustomDevice SDAA backend under Apache-2.0 with a hardware CI script). Every compiler, library, runtime, driver and tool **binary** is closed and distributed only as `.deb`/`.rpm`/`.runfile`/Docker tarball from vendor mirrors.
- **Traction is small and partly artificial.** 0–4★ on GitHub, 0–31★ on Gitee; Teco-AL's 157 forks are WAIC-competition artifacts. The three most active GitHub repos were created 2026-04-13 for that competition. The *documentation* cadence (v2.0 → v3.2 across 2025, PyTorch 2.7.1 tracking, Teco-vLLM gaining PD-separation and EPLB) tells a very different and more credible story about internal engineering activity.
