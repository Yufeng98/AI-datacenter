# Xiwang (曦望) GPU Software and Hardware Stack Summary

*as_of: 2026-08-08*
*prior revision: 2026-04-05 (baseline; content preserved below, corrections marked inline)*
*chip: xiwang*
*device_class: AI Accelerator (China, 曦望)*

---

## Overview

**曦望 (Xiwang Sunrise)** — recorded at baseline as 杭州曦望芯科智能科技有限公司 — is a Hangzhou-based GPU startup spun out of **SenseTime (商汤科技)** in late 2024, built around SenseTime's internal large-chip (大芯片) R&D team that had been developing GPU silicon since ~2019. The company is led by **Xu Bing (徐冰)**, a SenseTime co-founder. Within approximately one year of formal independence, Xiwang raised **≈¥3 billion (~US$0.4B)** — ⚠️ *corrected 2026-08-08: the baseline "~¥30 billion (~$4.1B USD)" was a ~10× unit error; the cited sources read 近30亿元 = nearly RMB 3 billion. See the 2026-08-08 update section.*

⚠️ *Corporate entity, flagged 2026-08-08:* the vendor site footer now reads **浙江曦望智能科技股份有限公司** (Zhejiang, joint-stock; ICP 浙ICP备2025183180号-1), while FlagTree still labels the vendor 曦望芯科. A rename/restructuring is **not asserted** here — no company-registry evidence was obtained.

Xiwang's product roadmap spans three generations:

- **S1** — DSA inference chip; taped out ~2019; >20,000 cumulative units; architecture not public
- **S2** — Full GPGPU; TSMC 7nm CoWoS, 64 GB memory, 350–450W TDP; first light July 2023; mass production from 2024 (~1,000 units that year); vendor now claims 规模化量产 / 万片级量产 (10,000-unit-class volume, company claim); performance claimed ≥A100, approaching H100. *As of 2026-08 the vendor markets S2 as an inference + fine-tuning GPGPU rather than a training part.*
- **S3** — Inference-specialized GPU; announced January 2026; FP32/FP16/FP8/INT8/FP4; **LPDDR6 memory and PCIe Gen6 host interface** (disclosed mid-2026); targets 90% per-token cost reduction; the vendor's single-chip performance claim vs. S2 rose from "3×+" (Dec 2025) to "5×" (current site). **Announced only — no public evidence of tape-out, sampling or shipment.**

The company differentiates on **inference economics** (per-token cost) rather than raw FLOPS, positioning itself as a "chip company that better understands AI" by virtue of SenseTime's production AI heritage. Its S3 memory choice — **LPDDR6 instead of HBM** — is the clearest architectural expression of that strategy, and also sidesteps the HBM export-control bottleneck that constrains Biren, Enflame and MetaX.

**Product family branding** (recorded 2026-08-08): 启望 (Qiwang) GPUs S1/S2/S3 · 智望 (Zhiwang) compute cards and OAM modules · 辰望 (Chenwang) servers and AI-agent appliances · 寰望 (Huanwang) supernodes/clusters · 熙望 (Xiwang) workstations · **SIRE** software stack.

---

## Software Stack

Xiwang's software platform is designed as a **CUDA replacement stack** — full compatibility with CUDA APIs is the primary user-facing claim. The stack is branded **SIRE (Sunrise Integrated Running Environment)** and the programming model is branded **TANG** — ⚠️ *both names became public after the 2026-04-05 baseline, which recorded "the specific stack brand name is not publicly disclosed."*

### Framework Integration
- **PyTorch**: Confirmed; training, fine-tuning, inference; DeepSpeed distributed training confirmed. SIRE claims PyTorch support that auto-tracks the latest upstream release.
- **Serving frameworks** (vendor-stated, 2026-08): **SGLang, vLLM, LightLLM, LightX2V**
- **Hugging Face**: Claimed compatible; >90% of mainstream LLM architectures on ModelScope
- **CUDA migration**: Source-level migration tooling confirmed; vendor claims 零代码无缝迁移 (zero-code migration) from third-party GPGPU platforms; binary compatibility not confirmed
- ⚠️ *Corrected 2026-08-08:* the baseline's "No open-source repos identified" is **wrong**. Xiwang has an upstreamed Triton backend in **FlagTree** (`third_party/sunrise`), public since **2026-01-23** and upgraded to Triton 3.6 on **2026-06-30**, plus **PCCL** support in **FlagCX** and a Sunrise entry in **vllm-plugin-FL**.

### Compiler / IR
- Fully self-developed compiler toolchain; vendor brand **SIRE**, programming model **TANG**
- Open path: **FlagTree** (BAAI FlagOS's unified Triton fork) `third_party/sunrise` — `backend_name = 'tang'`, target strings `tang:S2`, LLVM triple `stcu-unknown-tang` (S3 uses a `stcuv2` triple with separate `*_S3.bc` libdevice). The backend links a **closed-source `sunriseTritonPlugin.so`**, so it is a shim around a proprietary compiler, not a full open compiler.
- Input: CUDA C++ (via migration path) + native Xiwang kernel language + Triton (via FlagTree)
- Output: Xiwang GPU ISA binary; object bundling via `clang-offload-bundler` under `/usr/local/tangrt/toolchains/llvm/`
- Inference graph compiler (TensorRT analog) assumed present; not publicly confirmed

### Op Library
- cuDNN/cuBLAS analog — name still **not public**
- GEMM, attention, normalization, pooling confirmed present (inferred from LLM compatibility claims)
- Device math library in the open Triton path is **OCML-based** (`__ocml_*` entry points, AMD-style)
- Flash-Attention style attention kernels: vendor claims 98% FlashAttention operator efficiency for S3 (仿真实测 / simulation-measured)
- S3 adds FP8/FP4 precision support

### Runtime
- **TangRT** — the CUDA-runtime analog; `libtang.so` / `libtangrt_shared`, default install root `/usr/local/tangrt`
- Env knobs exposed by the open backend: `TRITON_LIBTANG_PATH`, `TRITON_SUNRISE_LLD_PATH`, `TRITON_SUNRISE_TRANSLATE_TRIPLE`

### Communication Library
- **PCCL** — "Sunrise Collective Communications Library", integrated into FlagCX (`USE_SUNRISE=1`, `CCL_HOME=/usr/local/pccl`, `-lpccl`, `-DUSE_SUNRISE_ADAPTOR`)
- FlagCX support matrix: send/recv, broadcast, reduce, allreduce, allgather, reducescatter and group ops, in both homogeneous and heterogeneous modes; **gather / scatter / alltoall / alltoallv are not supported**
- Topology details and bandwidth still not public

### ISA
- Fully self-developed SIMT ISA; LLVM target triple `stcu-unknown-tang` (`stcuv2` for S3)
- **Warp size 32**; FlagTree defaults `num_warps=4`, `num_stages=3`; `min_dot_size` = (8,8,16) for INT8 else (8,8,4); dot input precision restricted to `ieee`; only **`fp8e5` (E5M2)** exposed among FP8 types on the Triton path
- No ISA reference manual published; no PTX-equivalent virtual ISA announced

---

## Hardware Architecture

### S2 GPGPU (Current Flagship)

| Attribute | Value |
|-----------|-------|
| Process | TSMC 7nm |
| Packaging | 2.5D CoWoS |
| Memory | 64 GB (type **not disclosed**) |
| Host interface | PCIe Gen5 x16 (128 GB/s) |
| TDP | 350–450 W |
| Compute architecture | SIMT GPGPU (fully self-developed); warp size 32 (from FlagTree backend) |
| Compute units | Not public |
| Peak FP32 | Claimed > NVIDIA A100 (312 TFLOPS ref); not independently verified |
| Peak BF16 | Claimed approaching H100; not independently verified |
| Scale-up interconnect | **SRLink** (proprietary; 自研 SRLink 等高速互联技术) — bandwidth and topology not public |
| Form factors | 智望 S2-X1 PCIe FHFL card; 智望 S2-M1 **OAM module** |

**Key unknowns:** transistor count, compute unit count, on-chip SRAM capacity/type, memory type, memory bandwidth figure, SRLink bandwidth/topology.

### S3 (announced January 2026)

| Attribute | Value |
|-----------|-------|
| Process | Not disclosed |
| Memory | **LPDDR6**, backward-compatible with LPDDR5X (vendor claims 国内首款挂载 LPDDR6 的 GPU). Capacity and bandwidth **not disclosed** |
| Host interface | **PCIe Gen6** (vendor claims 国内首用 PCIe Gen6 的 GPU; "100% host-device bandwidth improvement" claim) |
| Precision | FP32, FP16, FP8, INT8, FP4 (all vendor-stated) |
| Peak FP4 | Characterised only as "P 级" (petaFLOPS-class) — **no TFLOPS figure published** |
| Operator efficiency | ~99% GEMM, ~98% FlashAttention — **仿真实测 (simulation-measured)** company claim, not silicon measurement |
| Target performance vs S2 | "5× single-chip performance" (company claim, current site; was "3×+" in Dec 2025 — see update section) |
| Target cost vs S2 | −90% per-token (company claim) |
| Positioning | 百万Token一分钱 (1M tokens for ¥0.01) |
| Rack-scale product | 寰望 **SC3-256 超节点** ("Rise SC3-256 SuperPOD"), liquid-cooled, PD-disaggregated, large-EP; chip count **not stated by the vendor** |
| Status | **Announced only** — no public evidence of tape-out, sampling or shipment |

---

## Funding and Business Context

| Date | Event |
|------|-------|
| Late 2024 | Spun out of SenseTime |
| Early 2025 | Angel round (SenseTime + LianChuang Yongxuan) |
| May 2025 | A-round: ¥250M (Beijing Lier lead) |
| June–July 2025 | Pre-A round: ~¥1B; SANY Group fund (recorded as 华胥基金 in 2026 reporting, 华旭基金 in the baseline — character discrepancy unresolved), Fourth Paradigm, Youzu Network, Beijing Lier, Songhe Capital, Haitong Kaiyuan |
| January 2026 | S3 announced; cumulative funding **≈¥3B (~US$0.4B)** — vendor news item 曦望完成近30亿融资. ⚠️ *Corrected 2026-08-08 from the baseline's "~¥30B (~$4.1B)", a ~10× unit error* |
| H1 2026 (exact date **not disclosed**) | A further round of **>¥1B** — vendor news item 推理 GPU 独角兽曦望再获超 10 亿元融资, undated on the vendor site; cumulative total after this round is **not confirmed** |
| 2026 | S2 at 万片级 (10,000-unit-class) volume per company claim; S3 schedule is an unverified January 2026 executive statement (see update section) |

---

## Competitive Positioning

Xiwang occupies a distinct niche among Chinese GPU startups:

- **vs. Biren BR100**: Similar 7nm CoWoS specs but Biren blocked by US export controls; Xiwang currently shipping
- **vs. Mthreads S4000**: Xiwang is 7nm vs. Mthreads 12nm; higher performance claims; similar CUDA-compat strategy
- **vs. Kunlunxin P800**: Kunlunxin has 96 GB HBM3 and 30,000-unit cluster at Baidu; Xiwang is smaller volume but faster funding velocity
- **vs. Hygon K100_AI**: Hygon on Entity List (US sanctions); Xiwang not yet sanctioned

The SenseTime heritage gives Xiwang a credible path to production LLM inference deployments through SenseTime's own AI workloads, providing a natural first customer and optimization target.

---

## Xiwang Update — SIRE Stack Named, FlagOS Open-Source Integration, S3 LPDDR6/PCIe Gen6 (2026-08-08)

*Updated 2026-08-08 (supersedes the 2026-04-05 baseline). Change class: **MAJOR**. Primary sources: sunrise-ai.com S2/S3 product pages and news index (CMS assets republished 2026-08-06); FlagTree `third_party/sunrise`; FlagOpen/FlagCX; flagos-ai/vllm-plugin-FL. Dating control: the 2025-12-11 Wayback snapshot of sunrise-ai.com has no SIRE, no LPDDR6, no PCIe Gen6 and no SC3-256; the 2026-05-21 snapshot has SC3-256 in the navigation (placeholder link) but still no SIRE and no LPDDR6 — so those disclosures are genuinely post-baseline. The FlagTree integration, by contrast, was public on 2026-01-23 and was simply **missed** by the baseline research pass.*

### 1. The software stack is now named — and partly open source (largest change)

**SIRE — "Sunrise Integrated Running Environment"** is now a first-class product line in the site navigation, tagline 软硬协同 · 高度兼容 · 卓越效能. The vendor diagrams SIRE as four layers over the GPU:

```
框架层与生态应用   (frameworks + ecosystem applications)
编译器与算子库层   (compiler + operator libraries)
驱动与运行时层     (driver + runtime)
SoC 固件/系统软件层 (SoC firmware / system software)
────────────────────────────────────────────
Xiwang GPU (S2 / S3)
```

Scope tags claimed for SIRE: 基础工具链 / 编程接口 / 基础库 / 基础平台 / 模型引擎 / MaaS 服务能力 / IP 模块化.

Named components (all new to this repo):

| Component | Name | Evidence |
|---|---|---|
| Programming model | **TANG** | S2 product page: 依托自研 TANG 编程模型及 SIRE 软件栈; corroborated by `backend_name = 'tang'` in FlagTree |
| Runtime | **TangRT** | `libtang.so`, `libtangrt_shared`, install root `/usr/local/tangrt` (FlagTree driver, FlagCX makefile) |
| Collectives | **PCCL** ("Sunrise Collective Communications Library") | FlagCX README (links sunrise-ai.com); `CCL_HOME=/usr/local/pccl`, `-lpccl` |
| Triton compiler backend | FlagTree `third_party/sunrise` | FlagTree README vendor table: "Sunrise（曦望芯科）" |
| Op library / graph compiler | **still not disclosed** | — |

**Open-source integration timeline** (refutes the baseline's "no open-source repos identified"):

| Date | Event |
|---|---|
| **2026-01-23** | FlagTree adds the `sunrise` backend on **Triton 3.4** with CI/CD (*public before the 2026-04-05 baseline; missed*) |
| **2026-06-01** | FlagCX: "[PAL] Add torch plugin support for Sunrise" |
| **2026-06-10** | FlagCX: runtime vendor detection replaces the `#ifdef USE_SUNRISE_ADAPTOR` build-time switch |
| **2026-06-30** | FlagTree upgrades the `sunrise` backend to **Triton 3.6** with CI/CD |
| (undated) | `vllm-plugin-FL` lists Sunrise as a supported chip vendor |

Prebuilt wheels: `flagtree==0.4.0+sunrise3.4` and `flagtree==0.6.0+sunrise3.6` from `https://resource.flagos.net/repository/flagos-pypi-hosted/simple`. **Caveat:** the backend links a closed-source `sunriseTritonPlugin.so`, so this is an open shim around a proprietary compiler. The published FlagTree user manual states the backend is **"Available for S2"** — the open toolchain targets S2 silicon; S3 exists in it only as compiler plumbing.

> The vendor news index carries an item 曦望全栈适配 FlagOS 2.1. The FlagOS organisation page advertises "Latest Release v1.5", so the **"FlagOS 2.1" version string is not confirmed and is deliberately not published here**.

### 2. First inspectable microarchitecture evidence (from the open backend)

These are the first hard, third-party-inspectable architecture facts for this chip. They come from compiler configuration, not from a vendor spec sheet, and describe what the Triton path exposes — not necessarily the full hardware capability.

| Fact | Value |
|---|---|
| LLVM target triple | `stcu-unknown-tang` |
| S3 differentiation | `stcuv2` triple + separate `*_S3.bc` libdevice (source comment: "libdivice库,需要区分S2/S3") |
| Warp size | **32** |
| Defaults | `num_warps = 4`, `num_stages = 3` |
| FP8 types exposed | **`fp8e5` (E5M2) only** |
| `min_dot_size` | (8, 8, 16) for INT8; (8, 8, 4) otherwise |
| Dot input precision | restricted to `ieee` |
| Device math library | AMD-style **OCML** (`__ocml_*`) |
| Object bundling | `clang-offload-bundler` under `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-<arch>/` |

### 3. S3 hardware — confirmed direction, still no numbers

- **Memory: LPDDR6**, backward-compatible with LPDDR5X ("LPDDR6/5X … 向下兼容 LPDDR5X"); homepage claims 国内首款挂载 LPDDR6 的 GPU. This is the architecturally significant item: Xiwang is chasing inference cost-per-token with commodity LPDDR rather than HBM, which also sidesteps the HBM export-control bottleneck. **Capacity and bandwidth are not disclosed.** This applies to **S3 only** — S2's 64 GB memory type remains undisclosed and must not be back-filled as LPDDR.
- **Host interface: PCIe Gen6** (国内首用 PCIe Gen6 的 GPU), with a "100% host-device interconnect bandwidth improvement" claim.
- **Precisions: FP32, FP16, FP8, INT8, FP4.** Peak FP4 is given only as "P 级" (petaFLOPS-class); no TFLOPS figure exists.
- **Operator efficiency 99% GEMM / 98% FlashAttention is 仿真实测 — simulation-measured**, which matters for a part with no public tape-out.
- **Vendor performance claim inflated since the baseline:** the December 2025 site said 推理性能 x3 倍+ (the repo's "+300%"); the current site says **"5X 单芯片性能提升"**, with the 90% per-token cost reduction unchanged. Both are unverified company claims — the change and its date are the recordable fact, not the number.
- Process node, TDP, die area, foundry and compute-unit counts remain **not disclosed**.

### 4. 寰望 SC3-256 SuperPOD (new rack-scale product)

An S3-based rack-scale system: liquid-cooled, **PD-disaggregated** (prefill/decode separation), built for **large Expert-Parallel (EP)** deployment, targeting **万亿乃至更大参数量** (trillion-parameter and larger) multimodal MoE inference. Vendor claims: TCO down "an order of magnitude" at equal performance, communication latency down an order of magnitude, and ">20× throughput advantage" — the last footnoted 数据来源于曦望实验室 (vendor lab data).

> **The vendor page never states a chip count.** "256" appears only in the product name. Do not record "256 S3 chips" as a specification. Link technology and bandwidth for the supernode are also undisclosed.

### 5. Scale-up interconnect is now partially named: SRLink

The S2 page states S2 scales across cards and nodes via 自研 **SRLink** 等高速互联技术. This replaces the baseline's flat "Scale-up interconnect: Not public". Link technology, topology and bandwidth remain undisclosed.

### 6. Product and form-factor expansion

- **S2 gains an OAM module** (智望 S2-M1) alongside the PCIe card (智望 S2-X1).
- **辰望 servers:** A4-2N1 / A4-2N2 / N8-2G1 / N8-2G2 / A16-2G1. The 16-card A16-2G1 is claimed to run full DeepSeek in a single node using a 计算感知通信拓扑 ("compute-aware communication topology"), claimed +400% vs general-purpose platforms — **unverified**.
- **熙望 workstations** are a fifth hardware line, new to this repo.
- S2 is now marketed as an **inference + fine-tuning** GPGPU (大模型推理 GPU) rather than as a training part.

### 7. Status verbs — deliberately not upgraded

- **S2: volume production.** Vendor states 已实现规模化量产 and 万片级量产 (10,000-unit class). The unit figure remains a company claim.
- **S3: announced only.** There is no public evidence that S3 has taped out, sampled or shipped, and the vendor's own S3 page carries no schedule and no status. A January 2026 executive statement (attributed to co-CEO 王勇, 2026-01-28) described S3 R&D as complete with tape-out planned for mid-2026, mass production by end-2026, and a three-year roadmap of **S3 (2026) → S4 high-performance chip (2027) → S5 security chip (2028)**. That statement **could not be independently verified** (WeChat sources are access-gated) and **no source dated after April 2026 confirms the mid-2026 tape-out occurred.** Treat the whole schedule as roadmap-grade.

### 8. Funding correction

The baseline's "**~¥30 billion (~$4.1B USD)**" is a ~10× unit error. The repo's own cited sources are titled 近**30亿**元融资 = nearly **RMB 3 billion ≈ US$0.4B**, matching the vendor news item 曦望完成近30亿融资 (January 2026). A subsequent round, 推理 GPU 独角兽曦望**再获超 10 亿元**融资 (>RMB 1B), is listed **undated** on the vendor news page; its position in the reverse-chronological feed places it between January 2026 and an April 2026 item, so H1 2026 is the most that can be said. **The date of that round and the resulting cumulative total are not confirmed.**

---

## Data Quality Notes

*Revised 2026-08-08.*

This summary is based on **limited public information** — vendor product pages, Chinese-language media, and (new as of this revision) an open-source Triton/collectives integration in the FlagOS ecosystem. The following critical technical details remain **not disclosed**:

- Exact compute unit count and hierarchy
- Transistor count and die area
- On-chip SRAM capacity and type
- **S2** memory type (64 GB capacity confirmed; type never stated)
- **S3** memory capacity and bandwidth (LPDDR6 type confirmed)
- S3 process node, TDP and foundry
- Any FLOPS/TOPS figure for S3 (FP4 is "P 级" only)
- SRLink bandwidth and topology
- SC3-256 chip count and link technology
- Op-library and graph-compiler component names
- Any independent or third-party benchmark result

**Resolved since the baseline:** software stack brand (SIRE), programming model (TANG), runtime (TangRT), collectives library (PCCL), open-source presence (FlagTree / FlagCX / vllm-plugin-FL), S3 memory type (LPDDR6), S3 host interface (PCIe Gen6), scale-up interconnect name (SRLink).

**No MLPerf submissions and no Hot Chips / ISCA / ISSCC papers were found for Xiwang.** (Hot Chips 38, 2026-08-23…25, is 15 days in the future as of this revision and contains no scheduled Xiwang talk.)

Confidence in architecture details: **LOW-MEDIUM** (raised from LOW — the FlagTree backend supplies inspectable warp size, triples and dot shapes, but no capacity, bandwidth or FLOPS figure). Confidence in software-stack component naming: **HIGH** (vendor page + open source agree). Confidence in funding/business facts: **MEDIUM** (lowered from MEDIUM-HIGH — the baseline carried a 10× funding error and the latest round is undated).

---

## Resources

- [曦望官网 — sunrise-ai.com](https://sunrise-ai.com/)
- [量子位 — S2 financing (June 2025)](https://www.qbitai.com/2025/06/303355.html)
- [量子位 — S3 announcement (January 2026)](https://www.qbitai.com/2026/01/373113.html)
- [智东西 — 杭州GPU黑马](https://zhidx.com/p/489508.html)
- [东方财富 — 30亿融资](https://caifuhao.eastmoney.com/news/20260123202536017296970)
- [OFweek — 从商汤拆出来的AI芯片公司](https://www.ofweek.com/ai/2026-01/ART-201700-8420-30680307.html)
- [新浪财经 — 新国产GPU曦望](https://finance.sina.com.cn/tech/roll/2025-07-01/doc-infcxsmt5575943.shtml)
- [CSDN — S2追平A100综述](https://blog.csdn.net/suanlix/article/details/149097721)

### Added 2026-08-08

- [曦望 S3 product page — LPDDR6 / PCIe Gen6 / FP4 / SC3-256 / SIRE diagram](https://sunrise-ai.com/products/s3-product)
- [曦望 S2 product page — TANG programming model, SIRE, SRLink, S2-X1 / S2-M1, 辰望 servers](https://sunrise-ai.com/products/s2-product)
- [曦望 news index — 完成近30亿融资; 再获超10亿元融资 (undated)](https://sunrise-ai.com/news)
- [FlagTree — unified Triton fork; `third_party/sunrise` backend](https://github.com/FlagTree/flagtree)
- [FlagTree sunrise backend compiler.py — triples, warp size, dot shapes](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/compiler.py)
- [FlagTree sunrise user manual — wheels, "Available for S2"](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-sunrise)
- [FlagCX — PCCL "Sunrise Collective Communications Library" and op-support matrix](https://raw.githubusercontent.com/FlagOpen/FlagCX/main/README.md)
- [vllm-plugin-FL — Sunrise listed as a supported chip vendor](https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/README.md)
- [Wayback 2025-12-11 sunrise-ai.com — pre-SIRE / pre-LPDDR6 control snapshot](http://web.archive.org/web/20251211220612id_/https://sunrise-ai.com/)
- [Wayback 2026-05-21 sunrise-ai.com — SC3-256 in nav, still no SIRE/LPDDR6](http://web.archive.org/web/20260521020029id_/https://sunrise-ai.com/)
