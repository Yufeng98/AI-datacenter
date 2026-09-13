# Iluvatar CoreX (天数智芯) Search Results

*as_of: 2026-08-08*
*chip: tianshu-zhixin*
*device_class: GPU (天数智芯 / Iluvatar CoreX)*

This file was created on 2026-08-08 during the TianGai 300 update wave. The chip's original
investigation (Apr 2026) predates the per-chip `search-results.md` convention, so the baseline
resources below are back-filled from the investigation reports' source lists; the resources under
"Discovered 2026-08-08" are new.

---

## Discovered 2026-08-08 — TianGai 300 / WAIC 2026

### Primary (vendor)

| Resource | URL | Layer | Note |
|----------|-----|-------|------|
| Iluvatar CoreX product navigation | https://www.iluvatar.com/ | Hardware | Lists 天垓300 (node `cpjs-yj-xlxl-tg300`), 天垓150, 天垓100, 智铠100, and the 彤央 TY1000/TY1100/TY1100-NX/TY1100-NX-PRO/TY1200 edge line. **No specification sheet for 天垓300.** Vendor news feed unchanged since 2021 — no launch press release exists, so this listing is the strongest primary artifact. |
| Iluvatar developer portal | https://developer.iluvatar.com | Software | Launched ~2026-07-15/19; docs, software images, model cases, tech blogs, DeepSpark resources |

### Press coverage of the 2026-07-19 launch

⚠️ **Independence caveat:** these outlets appear to derive from a common company briefing. Source
count is high; true independence is low. Nothing in the coverage is independently measured.

| Resource | URL | Layer | What it adds |
|----------|-----|-------|--------------|
| Tencent News (piece 1), 2026-07-19 | https://news.qq.com/rain/a/20260719A08MB800 | Hardware | WAIC day-3 launch; new self-developed architecture; SIMT; ixSMEX/ixDPX/ixTrans; FP4/FP8 + MMA/DPX/FMA; 144-chip supernode; "已具备规模化应用条件"; 340+ customers as of end-2025 |
| Tencent News (piece 2), 2026-07-19 | https://news.qq.com/rain/a/20260719A08J8N00 | Hardware | "ixDPX compresses six instructions to one"; Hopper-relative TTFT / attention / MoE / decode claims; confirms no process/memory/TDP disclosure |
| IT之家 | https://www.ithome.com/0/978/781.htm | Hardware | 2026-07-19 WAIC launch; SIMT general-purpose architecture; Hopper-relative claims |
| Tencent Cloud developer news | https://cloud.tencent.com/developer/news/4280275 | Hardware / Business | 144-chip full-interconnect supernode at trial/partnership stage; ~100 acceleration libraries; 340+ customers, 30+ industries, 1,000+ day cluster uptime, >10k chips online |
| 半导体行业观察 (semi-insights) | https://www.semi-insights.com/s/bdt/15/50538.shtml | Hardware | July 19 launch; SIMT scalar/vector/tensor; attention/MoE/AF-separation/PD-separation optimizations |
| 经济参考报, 2026-07-20 | https://www.jjckb.cn/20260720/3cb0e7d7378745b58c576c51ee79b65a/c.html | Business | WAIC 2026 coverage |
| 观察者网, 2026-07-18 | https://www.guancha.cn/GongSi/2026_07_18_824207.shtml | Business | Pre-launch coverage. ⚠️ Note: the launch itself is dated 2026-07-19 by every retrieved source; "July 18" is not supported. |
| AAStocks — 09903.HK company news | https://www.aastocks.com | Business | "released its new-generation general-purpose GPU flagship product, Tiangai 300" |
| datayuan (Substack) | https://datayuan.substack.com/p/iluvatar-corex-waic-has-released | Hardware | English-language aggregation of the WAIC announcement |
| pioneersat | https://www.pioneersat.com/cyzx/115894.html | Hardware | Secondary Chinese coverage |
| WAIC 2026 official site / Shanghai municipal government release | — | Context | Confirms WAIC 2026 dates: July 17–20, 2026, Shanghai |

### Software (new)

| Resource | URL | Layer | Note |
|----------|-----|-------|------|
| Deep-Spark/FlyDSL | https://github.com/Deep-Spark/FlyDSL | Kernel-authoring DSL | **New layer for this vendor.** Python DSL + MLIR stack; explicit layouts and tiling; fork of an AMD-ROCm-origin project; FlyIXDL backend for `ivcore11`-class parts (MR-50, MR-100, BI-V150, BI-V150s); CUTLASS layout algebra + ROCm Composable Kernel lineage; lower-level than Triton; 873 commits on `iluvatar` branch; last updated 2026-08-08 |
| Deep-Spark repositories by last update | https://github.com/orgs/Deep-Spark/repositories?sort=updated | Software | Activity snapshot 2026-08-08: DeepSparkInference 216 models (08-06); iluvatar-corex-ixrt (08-05); DeepSparkHub (08-03); vLLM fork (06-12); ix-exporter (05-20); ix-device-plugin (05-19); xllm and LightX2V added (05-12) |

### Roadmap context (pre-existing, re-checked)

| Resource | URL | Note |
|----------|-----|------|
| TrendForce, 2026-01-12 | https://www.trendforce.com/news/2026/01/12/news-chinas-iluvatar-corex-reportedly-to-unveil-2026-28-gpu-roadmap-targeting-nvidia-h200-b200/ | Big-Dipper codenames (Tianshu/Tianxuan/Tianji/Tianquan). ⚠️ **No retrieved source maps TianGai 300 onto any of these codenames.** |

### Searched and NOT found (2026-08-08)

| Query target | Result |
|---|---|
| MLPerf / MLCommons entry for Iluvatar CoreX | **None.** A Chinese analysis reports zero entries through 2025; none in v6.0. |
| Hot Chips / ISCA / ISSCC 2026 presentation by Iluvatar | **None.** (Hot Chips 38 runs 2026-08-23/25 — still future as of this scan; program public, content not.) |
| H1 2026 financial disclosure | **None** post-dating the FY2025 RMB 1.03B figure. |
| TianGai 300 specification sheet (any source) | **None.** Process, memory, FLOPS, TDP, interconnect BW, form factor all unpublished. |
| TianGai 300 toolchain / SDK disclosure | **None.** No FP8/FP4 codegen notes, no ixSMEX/ixDPX/ixTrans intrinsics docs, no FlyDSL target. |

---

## Baseline resources (back-filled from the Apr 2026 investigations)

### Official

| Resource | URL | Layer |
|----------|-----|-------|
| Iluvatar CoreX official site | https://www.iluvatar.com/ | Company |
| DeepSpark Open Platform | https://www.deepspark.org.cn/ | Software |
| Deep-Spark GitHub organization | https://github.com/Deep-Spark | Software |
| TianGai-100 product page | https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-tlxltt-zk100 | Hardware |
| DeepSpark — IxRT project introduction | https://www.deepspark.org.cn/introduce?code=IxRTxmjs&topicId=711&fullCode=xmjs | Inference Runtime |

### Open source

| Resource | URL | Layer |
|----------|-----|-------|
| iluvatar-corex-ixrt | https://github.com/Deep-Spark/iluvatar-corex-ixrt | Inference Runtime |
| ix-container-toolkit | https://github.com/Deep-Spark/ix-container-toolkit | Driver / Container |
| ixGDB | https://github.com/Deep-Spark/ixGDB | Debugging |
| ix-exporter | https://github.com/Deep-Spark/ix-exporter | Monitoring |
| DeepSpark | https://github.com/Deep-Spark/DeepSpark | Model zoo |

### Hardware analysis

| Resource | URL | Layer |
|----------|-----|-------|
| Wikipedia — Iluvatar CoreX | https://en.wikipedia.org/wiki/Iluvatar_CoreX | Hardware |
| Tom's Hardware — four-generation roadmap | https://www.tomshardware.com/pc-components/gpus/chinas-iluvatar-corex-unveils-four-generation-gpu-roadmap-aimed-at-surpassing-nvidia-rubin | Roadmap |
| TrendForce — Rubin 2027 roadmap | https://www.trendforce.com/news/2026/01/28/news-chinas-illuvatar-corex-unveils-bold-gpu-roadmap-reportedly-eyeing-nvidias-rubin-by-2027/ | Roadmap |
| SCMP — roadmap vs Rubin | https://www.scmp.com/tech/big-tech/article/3341368/iluvatar-corex-targets-nvidias-rubin-gpu-road-map-amid-china-chip-push | Roadmap |
| CSDN — TianGai-150 specifications | https://blog.csdn.net/2402_84466582/article/details/139412485 | Hardware |
| Leiphone — GPGPU architecture overview | https://m.leiphone.com/category/chips/fD16hBivUTdFgdUP.html | Hardware |
| Cloudhin — BI-V150 listing | http://www.cloudhin.com/xk/showproduct.php?id=275 | Hardware |

### Software / ecosystem

| Resource | URL | Layer |
|----------|-----|-------|
| HAMi — Iluvatar GPU sharing | https://project-hami.io/docs/userguide/iluvatar-device/enable-illuvatar-gpu-sharing | K8s / Virtualization |
| RiseUnion — Iluvatar GPU virtualization guide | https://www.theriseunion.com/blog/HAMi-iluvatar-support.html | K8s / Virtualization |
| Digitimes — first commercial GPGPU in China | https://www.digitimes.com/news/a20251222VL203/gpgpu-commercial-gpu-software-production-china.html | Company |
| BAAI — Aquila2-70B mixed-cluster training | https://www.elecfans.com/d/2328248.html | Framework / Benchmark |
| BAAI FlagPerf — TianGai-100 ResNet50 | https://hub.baai.ac.cn/view/29875 | Benchmark |

### Business / IPO

| Resource | URL | Layer |
|----------|-----|-------|
| Caixin — HKEX IPO USD 475M | https://www.caixinglobal.com/2025-12-30/chinese-gpu-maker-iluvatar-corex-seeks-475-million-in-hong-kong-listing-102398766.html | Business |
| Caixin — HKEX debut USD 5.3B valuation | https://www.caixinglobal.com/2026-01-08/chinese-gpu-maker-iluvatar-corex-climbs-in-hong-kong-debut-with-53-billion-valuation-102401708.html | Business |
| Startupnews — 2025 revenue +92% (RMB 1.03B) | https://startupnews.fyi/2026/04/02/chinese-gpu-designer-iluvatar-corex-reports-92-jump-in-annual-revenue/ | Business |
