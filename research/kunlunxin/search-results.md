# Kunlunxin XPU Software Stack & Hardware Resources

*as_of: 2026-08-08*
*chip: kunlunxin*
*device_class: AI Accelerator (百度昆仑芯)*
*seeds: https://www.kunlunxin.com/, https://github.com/baidu/vLLM-Kunlun, https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/index_cn.html*

> **File created 2026-08-08** during the Kunlunxin update wave. The 2026-04-05 baseline research for this
> chip was recorded directly in `research/kunlunxin/summary.md` and
> `research/kunlunxin/investigations/`; the baseline-era resources are relisted here so this file is a
> complete catalog, and newly discovered resources are marked **[new 2026-08-08]**.
>
> **Retrieval caveat:** `kunlunxin.com` returns **HTTP 403** to automated retrieval. Vendor-primary pages
> are listed below for human follow-up but could not be fetched during this pass, which is why every
> supernode performance figure in this chip's documents traces to Chinese-media reporting rather than a
> vendor spec sheet.

---

## Software Stack

### Framework Integration

- [PaddlePaddle XPU hardware support](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/index_cn.html) — Official PaddlePaddle documentation for the native XPU backend; install matrix and verified model list
- [PaddlePaddle 2.4 Release Note](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/release_note_en.html) — Introduces the XPU Plugin custom-operator mechanism
- [Kunlun XPU PaddlePaddle Installation Guide — PaddleX](https://paddlepaddle.github.io/PaddleX/main/en/other_devices_support/paddlepaddle_install_XPU.html) — End-to-end install walkthrough for XPU
- [KunlunXin XPU — FastDeploy LLM Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/kunlunxin_xpu/) — FastDeploy LLM serving path on XPU
- [Kunlunxin XPU Backend — FastDeploy DeepWiki](https://deepwiki.com/PaddlePaddle/FastDeploy/8.3-kunlunxin-xpu-backend) — `XpuWorker`, XVLLM, and device-init internals

### LLM Serving

- **[new 2026-08-08]** [baidu/vLLM-Kunlun (GitHub)](https://github.com/baidu/vLLM-Kunlun) — **Out-of-tree hardware plugin** (NOT a fork) registering XPU as a vLLM platform via Python entry points, per the vLLM "RFC: Hardware Pluggable" interface; prerequisites list Kunlun3 P800 only
- **[new 2026-08-08]** [vLLM-Kunlun documentation (readthedocs)](https://vllm-kunlun.readthedocs.io/en/latest/) — Install path (stock vLLM + plugin), prerequisites (Ubuntu 20.04, Python ≥3.10, PyTorch ≥2.5.1 via KL3 `xpytorch`), 20+ supported models
- **[new 2026-08-08]** [vLLM-Kunlun releases (GitHub API)](https://api.github.com/repos/baidu/vLLM-Kunlun/releases) — Verified release history: v0.10.1.1 (2025-12-24), v0.11.0rc1 (2025-12-26), v0.11.0 stable (2026-03-13, 154 commits / 26 contributors)

### Debugging / Profiling

- **[new 2026-08-08]** [vLLM-Kunlun developer guide — operator accuracy](https://vllm-kunlun.readthedocs.io/en/latest/developer_guide/evaluation/accuracy/accuracy_kernel.html) — Documents **`torch_xray`** (operator precision dump, automatic GPU-vs-P800 layer-by-layer comparison, auto-generated operator unit tests) and **`xpu_profiler`** (nsys-like operator call timelines). Both are documented ecosystem tools, **not** present in the repo's top-level tree

### Compiler / IR and Op Library

- [昆仑芯 XTCL — PaddlePaddle Lite](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/kunlunxin_xtcl.html) — TVM-based XPU graph compiler (AOT/JIT), ONNX + Paddle IR import
- [昆仑芯 XPU — PaddlePaddle Lite (v2.12)](https://www.paddlepaddle.org.cn/lite/v2.12/demo_guides/kunlunxin_xpu.html) — XPU deployment path for Paddle-Lite
- [Paddle-Lite Kunlunxin XPU demo guide (GitHub)](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/kunlunxin_xpu.md) — Buildable reference for the XPU backend
- [核心技术 — 昆仑芯官网](https://www.kunlunxin.com/technology) — Vendor overview of XRE / XTDK / XDNN / XTCL *(HTTP 403 to automated retrieval)*
- [分享：昆仑芯×飞桨适配方案 — 昆仑芯官网](https://www.kunlunxin.com/news/813.html) — Vendor writeup of the PaddlePaddle adaptation *(HTTP 403 to automated retrieval)*

### Community / Explainers

- [百度昆仑 XPU 详解 — 知乎](https://zhuanlan.zhihu.com/p/646793342) — Community deep-dive on the XPU architecture and stack
- [Kunlunxin P800 Virtualization Guide — RiseUnion](https://www.theriseunion.com/blog/HAMi-kunlunxin-p800-support.html) — HAMi-based P800 virtualization/partitioning

---

## Hardware Architecture

### Compute Engine

- [Kunlun: A 14nm High-Performance AI Processor for Diversified Workloads — Hot Chips 2020 (IEEE)](https://ieeexplore.ieee.org/document/9366056) — The only peer-reviewed primary architecture source for the XPU family (Gen 1)
- [Hot Chips 2020 Kunlun slide deck (PDF)](https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Inference_Baidu_Kunlun_v5.pdf) — SDNN / XVME cluster structure, 230 TOPS @ 900 MHz, 348 mm², 160 W
- [Baidu Kunlun II — Tom's Hardware](https://www.tomshardware.com/news/baidu-unveils-kunlun-ii-processor-for-ai) — Gen 2 (R200/R300) 7nm specs
- [昆仑芯P800前世今生 — CSDN](https://blog.csdn.net/Rong_Toa/article/details/151322568) — P800 spec compilation (secondary)
- [昆仑芯2代AI芯片 — 昆仑芯官网](https://www.kunlunxin.com/product/2873.html) — Vendor Gen 2 product page *(HTTP 403 to automated retrieval)*

### Packaging / Memory

- [Baidu Kunlun1-T I-Cube 2.5D package analysis — TechInsights](https://www.techinsights.com/blog/baidu-kunlun1-t-ai-processor-samsung-i-cube-25d-package-technology-advanced-packaging-analysis) — Teardown-grade confirmation of the Gen 1 Samsung I-Cube 2.5D HBM2 package

### Scale-up Interconnect / Supernodes

- **[new 2026-08-08]** [Baidu Create 2026 — Shen Dou on Tianchi 256 (腾讯新闻, 2026-05-13)](https://news.qq.com/rain/a/20260513A0688800) — Primary dating anchor: Tianchi 256 "lit up" (点亮) April 2026, stated to go on sale June 2026; +25% throughput, +50% inference efficiency
- **[new 2026-08-08]** [WAIC 2026 wrap-up (腾讯新闻, 2026-07-22)](https://news.qq.com/rain/a/20260722A04PTE00) — XPU-Link protocol with programmable switching, self-designed copper cabling, liquid-cooling CDU, 77% claimed bandwidth efficiency; thousand-card / 4,000-card roadmap intent; still calls Tianchi 256/512 "即将全面上市"
- **[new 2026-08-08]** [WAIC 2026 supernode coverage (新浪财经, 2026-07-22)](https://finance.sina.cn/tech/2026-07-22/detail-iniistty7951726.d.html) — Parallel account of the same WAIC disclosures
- **[new 2026-08-08]** [Baidu World 2025 — original Tianchi 256 figures (新浪财经, 2025-11-13)](https://finance.sina.com.cn/tech/2025-11-13/doc-infxfnye5573223.shtml) — Establishes that "4× inter-card bandwidth" and "3.5× single-card token throughput" are 2025 announcement figures, **not** 2026 results
- **[new 2026-08-08]** [WAIC 2026 supernode coverage (腾讯新闻, 2026-07-20)](https://news.qq.com/rain/a/20260720A07PE200) — 32/64-card Kunlun Super Node shown alongside Tianchi 256
- **[new 2026-08-08]** [WAIC 2026 China supernode push — TrendForce (2026-07-20)](https://www.trendforce.com/news/2026/07/20/news-waic-2026-highlights-chinas-supernode-push-led-by-huawei-minimax-m3-unitree-robots-in-focus/) — English-language industry context for the WAIC supernode wave
- **[new 2026-08-08]** [天池256卡超节点 — 百度百科](https://baike.baidu.com/item/%E5%A4%A9%E6%B1%A0256%E5%8D%A1%E8%B6%85%E8%8A%82%E7%82%B9/67795862) — **User-editable wiki**; used only as a launch-month cross-check, never as a primary source
- [昆仑芯3代P800万卡点亮 — 新浪科技](https://finance.sina.com.cn/tech/roll/2025-02-05/doc-ineimcyh2167561.shtml) — Ten-thousand-card P800 cluster bring-up
- [Baidu 3rd-gen Kunlun cluster — Digitimes](https://www.digitimes.com/news/a20250208PD210/baidu-ai-chip-chips-2024-production.html) — 30,000-card cluster reporting

### Gen 4 (M100 / M300)

- **[new 2026-08-08]** [芯智讯 WAIC show-floor report (腾讯新闻, 2026-07-20)](https://news.qq.com/rain/a/20260720A03JPT00) — **Key M100 source**: first physical public appearance, *no* official introduction materials released, carrier boards "正在制作中"; booth-staff positioning vs NVIDIA H20 with lower HBM capacity
- **[new 2026-08-08]** [M100 at WAIC 2026 (同花顺, 2026-07-17)](https://stock.10jqka.com.cn/20260717/c678258367.shtml) — Independent confirmation of the physical debut
- **[new 2026-08-08]** [Kunlunxin WAIC preview (同花顺, 2026-07-16)](https://stock.10jqka.com.cn/20260716/c678228712.shtml) — Pre-show framing
- **[new 2026-08-08]** [WAIC M100 coverage (雪球)](https://xueqiu.com/2156146731/401079693) — Third independent account of the M100 showing
- **[new 2026-08-08]** [昆仑芯M100 — 百度百科](https://baike.baidu.com/item/%E6%98%86%E4%BB%91%E8%8A%AFM100/66989616) — **User-editable wiki**; the origin of the unverified "January 2026 commercial ramp" claim. Recorded as low confidence only
- [M100/M300 roadmap — TrendForce (2025-11-13)](https://www.trendforce.com/news/2025/11/13/news-baidu-rolls-out-kunlun-roadmap-m100-m300-ai-chips-arrive-2026-2027/) — Original Gen 4 roadmap announcement
- **[new 2026-08-08]** [Baidu launches Kunlun M100 and M300 — DataCenterDynamics](https://www.datacenterdynamics.com/en/news/baidu-launches-kunlun-m100-and-m300-ai-chips/) — English-language roadmap coverage

---

## Corporate / Market

- **[new 2026-08-08]** [Baidu spin-off and separate listing of Kunlunxin (HKEXnews PDF, doc 2026010200013, 2026-01-02)](https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/2026010200013.pdf) — Primary filing announcement
- **[new 2026-08-08]** [Baidu IR — proposed spin-off and separate listing of Kunlunxin](https://ir.baidu.com/news-releases/news-release-details/baidu-announces-proposed-spin-and-separate-listing-kunlunxin/) — Issuer confirmation of the confidential Form A1 submission
- **[new 2026-08-08]** [Reuters — Kunlunxin targets USD 50B Hong Kong IPO (2026-06-28)](https://www.reuters.com/world/asia-pacific/baidus-ai-chip-unit-kunlunxin-targets-50-billion-hong-kong-ipo-information-2026-06-28/) — **Press-reported leak** (citing The Information), not a filed figure
- **[new 2026-08-08]** [CNBC — Baidu Kunlunxin Hong Kong IPO (2026-06-29)](https://www.cnbc.com/2026/06/29/baidu-kunlunxin-hong-kong-ipo.html) — Follow-up; Baidu HK shares +6%
- **[new 2026-08-08]** [Kunlunxin IPO context (网易, 2026)](https://m.163.com/local/article/KSQN6KLH05129QAF.html) — Secondary Chinese coverage of the listing plan
- **[new 2026-08-08]** [Kunlunxin valuation reporting (腾讯新闻, 2026-06-14)](https://news.qq.com/rain/a/20260614A021ST00) — Earlier valuation reporting, useful for showing how fast the leaked figure moved
- [Kunlunxin files Hong Kong IPO — SCMP](https://www.scmp.com/news/china-future-tech/semiconductors/article/3338471/baidu-chip-unit-kunlunxin-files-hong-kong-ipo-amid-chinas-push-tech-self-reliance) — January 2026 filing coverage
- [Baidu explores Kunlunxin spinoff — TrendForce (2025-12-09)](https://www.trendforce.com/news/2025/12/09/news-baidu-reportedly-explores-kunlunxin-spinoff-and-listing-amid-a-surge-in-china-chip-ipos/) — Pre-filing reporting
- [Kunlunxin — Wikipedia](https://en.wikipedia.org/wiki/Kunlunxin) — Company overview

---

## Searched and Not Found (2026-08-08)

Recorded as **absent**, not as negative evidence:

- No MLPerf Training v5.x or Inference v6.x submission by Kunlunxin or Baidu XPU
- No Hot Chips 2026 / ISCA 2026 / ISSCC 2026 paper or talk on Kunlunxin — Hot Chips 38 runs 2026-08-23 → 08-25 and lists no Kunlunxin session
- No new public XRE / XTDK / XDNN / XTCL release notes in the 2026-04-01 → 2026-08-08 window
- No published M100 specification of any kind (process, memory, throughput, TDP)
- No vendor-primary Tianchi supernode spec sheet retrievable (kunlunxin.com HTTP 403)
