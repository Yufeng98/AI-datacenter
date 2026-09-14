# Xiwang (曦望) GPU Hardware Architecture

*as_of: 2026-09-13*
*chip: xiwang*
*device_class: AI Accelerator (China, 曦望)*
*Representative products: S1 (DSA, 2022), S2 (GPGPU, 7nm, 2024), S3 (inference GPU, 2026 target)*

---

## Overview

**曦望 Xiwang Sunrise** (full legal name: 杭州曦望芯科智能科技有限公司) is a Hangzhou-based GPU startup spun out of **SenseTime (商汤科技)** at the end of 2024. The company was formed around SenseTime's internal large-chip (大芯片) R&D team, which had been developing GPU silicon since approximately 2019. The founding CEO is **Xu Bing (徐冰)**, a SenseTime co-founder. SenseTime participated in the angel-round financing and retains a strategic relationship with the company.

Xiwang's mission is framed as "rewriting the P&L of the Chinese AI industry" — targeting a dramatic reduction in per-token inference cost rather than competing on raw FLOPS. The company describes itself as a "chip company that better understands AI" due to its origin inside a major AI application firm.

The product roadmap spans three generations:

| Product | Type | Process | Status |
|---------|------|---------|--------|
| S1 | DSA cloud/edge inference chip | Not public | Taped out ~2019; cumulative sales >20,000 units |
| S2 | Training + inference GPGPU | TSMC 7nm CoWoS | First light July 2023; mass production 2024 (1,000 units); near-term ramp to 10,000 units |
| S3 | Inference-specialized GPU | Not public (advanced node expected) | Announced January 2026; mass production target 2026 |

---

## 1. S1 — DSA Inference Chip

### Architecture

S1 is a **Domain-Specific Architecture (DSA)** chip targeting cloud and edge inference. It is a first-generation product developed before the company's formal spinout. Specific microarchitectural details (compute units, memory hierarchy, die area, transistor count) are **not publicly disclosed**.

| Attribute | Value |
|-----------|-------|
| Architecture | DSA (not GPGPU) |
| Target workload | Cloud and edge inference |
| Tapeout year | ~2019 |
| Cumulative sales | >20,000 units (company claim, 2025) |
| Process node | Not public |
| Memory | Not public |
| Peak performance | Not public |

The S1 established Xiwang's commercial track record and silicon execution capability prior to the GPGPU pivot in S2.

---

## 2. S2 — Training + Inference GPGPU

The S2 is Xiwang's flagship product as of 2025–2026. It is a **full GPGPU** — fully programmable parallel processor with SIMT execution model — designed for both training and inference of large language models. Key positioning: performance comparable to NVIDIA A100 SXM, approaching H100 SXM, at equivalent 7nm process.

### Process and Packaging

| Attribute | Value |
|-----------|-------|
| Process node | TSMC 7nm |
| Packaging | 2.5D CoWoS (TSMC) |
| Form factor | PCIe FHFL acceleration card |
| TDP | 350–450 W |
| Host interface | PCIe Gen5 (128 GB/s) |

### Compute Engine

Xiwang states the S2 is a fully self-developed GPGPU from instruction set to microarchitecture. The internal architecture name is **not publicly disclosed**. Number of compute units, SM/CU count, tensor core configuration, and transistor count are **not public**. (⚠️ *Updated 2026-08-08: **warp size is 32**, read from the FlagTree `sunrise` backend — see the 2026-08-08 update, section 4.*)

| Attribute | Value |
|-----------|-------|
| Compute architecture | SIMT GPGPU (fully self-developed) |
| Architecture brand | Not public |
| Compute unit type | Not public (GPGPU SM-analog) |
| Compute unit count | Not public |
| Tensor/matrix units | Present (inferred; A100-class performance claim) |
| Supported precision | FP32, TF32 (inferred), BF16, FP16, INT8; FP8 not confirmed for S2 |
| ISA | Proprietary (not publicly documented) |
| Warp/wavefront size | Not public |

**Performance claims (company-stated, not independently verified):**

| Metric | S2 | NVIDIA A100 SXM | Notes |
|--------|-----|-----------------|-------|
| FP32 | Exceeds A100 | 312 TFLOPS | Company claim |
| TF32 | Exceeds A100 | 312 TFLOPS (tensor) | Company claim |
| FP16/BF16 | Approaches H100 | A100: 312; H100: 1,979 TFLOPS | Company claim |
| INT8 | Not disclosed | A100: 624 TOPS | Not public |
| Memory BW | "High" (higher than A100) | 2.0 TB/s | Qualitative claim only |

### Memory Hierarchy

| Level | Details |
|-------|---------|
| On-chip SRAM | Capacity not public; type (cache vs scratchpad) not public |
| Off-chip memory | 64 GB (type **not disclosed**; CoWoS packaging *suggests* HBM2e or HBM3 — ⚠️ *2026-08-08: this is an inference, not a vendor statement; do not record it as a spec, and do not back-fill S3's LPDDR6 onto S2*) |
| Memory bandwidth | Not publicly quantified; described as "higher than A100" |
| Memory interface | Not public (CoWoS 2.5D implies HBM stacks) |

The 64 GB memory capacity is confirmed. Whether HBM2e (SK Hynix/Samsung) or domestic HBM alternatives are used is **not public** — this is strategically sensitive given US export controls on HBM supply to Chinese chipmakers.

### Scale-up Interconnect

| Attribute | Value |
|-----------|-------|
| Proprietary interconnect | ⚠️ *Updated 2026-08-08:* named **SRLink** (自研); bandwidth and topology still not public |
| Multi-GPU scale-up | Supported (inferred; multi-GPU cluster deployments mentioned) |
| Host interconnect | PCIe Gen5 x16 (128 GB/s) |

---

## 3. S3 — Next-Generation Inference GPU

S3 was formally announced in **January 2026** at the same event as Xiwang's cumulative financing disclosure (⚠️ *corrected 2026-08-08: ≈RMB 3 billion, not 30 billion*). It is positioned as an all-in inference-specialized GPU for large model serving.

| Attribute | Value |
|-----------|-------|
| Architecture | GPGPU with inference-optimized extensions (details not public) |
| Process node | Not publicly disclosed (advanced node expected, likely 5nm or 7nm with new process variant) |
| Packaging | Not public |
| Target TDP | Not public |
| Memory | Not public |
| Target performance | "300%+ inference performance vs S2" (company claim) |
| Target cost | "90% reduction in per-token inference cost vs S2" (company claim) |
| Precision | FP8 and FP4 native support confirmed (company announcement) |
| Mass production target | ⚠️ *2026-08-08: unverified January 2026 executive statement (tape-out mid-2026, MP end-2026). No public evidence of tape-out. S3 remains announced-only.* |

The "百万Token一分钱" (1 million tokens for 1 RMB cent) positioning targets DeepSeek R1-class models at a dramatically lower serving cost point than H100-based infrastructure.

---

## 4. Business and Funding Timeline

| Date | Event |
|------|-------|
| ~2019 | SenseTime internal large-chip team begins GPU silicon development; S1 taped out |
| Late 2024 | Formal spinout from SenseTime as 杭州曦望芯科智能科技有限公司 |
| Early 2025 | Angel round financing (SenseTime + 联创永宣 LianChuang Yongxuan investors) |
| May 2025 | A-round: ¥250 million (Beijing Lier lead) |
| June–July 2025 | Pre-A round: ~¥1 billion (¥10 亿); investors: SANY 三一集团 (华旭基金), Fourth Paradigm (第四范式), Youzu Network (游族网络), Beijing Lier, Songhe Capital, Haitong Kaiyuan |
| July 2025 | S2 chip publicly revealed; S3 roadmap announced |
| H2 2025 | Continued fundraising rounds (company described as 2025's most-funded AI startup in multiple reports) |
| January 2026 | S3 announced; cumulative funding disclosed as ~~**~¥30 billion** (~$4.1 billion USD)~~ — ⚠️ **CORRECTED 2026-08-08: ≈¥3 billion (~US$0.4B)**; the cited sources are titled 近30亿元 = nearly RMB 3 billion. See the 2026-08-08 update, section 1 |
| 2026 | S3 mass production target; S2 ramp to 10,000+ units |

---

## 5. Comparison with Peer Chinese GPUs

| Chip | Process | Memory | BF16 Performance | Status |
|------|---------|--------|-----------------|--------|
| Xiwang S2 | TSMC 7nm CoWoS | 64 GB | ~A100 class (company claim) | Mass production 2024 (small volume) |
| Mthreads MTT S4000 | TSMC 12nm | 48 GB GDDR6 | 200 TFLOPS | STAR IPO Nov 2025; 4,096 SP |
| Biren BR100 | TSMC 7nm CoWoS | 64 GB HBM2e | 1,024 TFLOPS | US export controls blocked production |
| Kunlunxin P800 | 7nm | 96 GB HBM3 | 345 TFLOPS | 30,000 unit cluster at Baidu |
| Hygon K100_AI | TSMC 7nm (est.) | 64 GB HBM2e | 180 TFLOPS (FP16) | GCN-derived; Entity List 2019 |

Xiwang's differentiation: **inference cost** focus (not raw FLOPS), **CUDA compatibility** (lower adoption barrier), **SenseTime AI application heritage** (optimized for production LLM serving from day one).

---

## Sources

- [量子位 — 曦望融资近10亿融资 (June 2025)](https://www.qbitai.com/2025/06/303355.html)
- [智东西 — 杭州GPU黑马融资近10亿](https://zhidx.com/p/489508.html)
- [凤凰网财经 — 曦望融资近10亿元，2026年量产第三代](https://finance.ifeng.com/c/8l5eKqTHNhb)
- [量子位 — 曦望发布推理GPU S3 (January 2026)](https://www.qbitai.com/2026/01/373113.html)
- [新浪科技 — 新国产GPU曦望，刚融了10个亿](https://finance.sina.com.cn/tech/roll/2025-07-01/doc-infcxsmt5575943.shtml)
- [东方财富 — 曦望1年内累计获得近30亿元融资](https://caifuhao.eastmoney.com/news/20260123202536017296970)
- [OFweek — 从商汤拆出来的AI芯片公司融了30亿](https://www.ofweek.com/ai/2026-01/ART-201700-8420-30680307.html)
- [CSDN — 曦望S2规格综述](https://blog.csdn.net/suanlix/article/details/149097721)
- [知乎 — 如何看待国产GPU曦望完成近10亿融资](https://zhuanlan.zhihu.com/p/1924148927080432369)
- [36氪 — 国产AI芯片疯狂秀肌肉](https://36kr.com/p/3661162854146696)

---

# Update — 2026-08-08: S3 memory/host disclosures, SRLink, SC3-256, and the first compiler-visible microarchitecture evidence

*Appended 2026-08-08. Supersedes the 2026-04-05 baseline above where marked. Change class: **major**.*

*Primary sources: sunrise-ai.com S2 and S3 product pages plus the news index (site CMS assets republished 2026-08-06); FlagTree `third_party/sunrise`; FlagOpen/FlagCX. Dating control: Wayback snapshots 2025-12-11 (no SIRE / LPDDR6 / PCIe Gen6 / SC3-256) and 2026-05-21 (SC3-256 in nav as a placeholder link; still no SIRE, no LPDDR6).*

## 1. What changed since the baseline

| Item | Baseline (2026-04-05) | Now (2026-08-08) |
|---|---|---|
| S3 memory | "Not public" | **LPDDR6**, backward-compatible with LPDDR5X. Capacity and bandwidth still not disclosed |
| S3 host interface | "Not public" | **PCIe Gen6** |
| S3 precision | FP8, FP4 confirmed; BF16/FP16/INT8 inferred | Vendor-stated list is **FP32, FP16, FP8, INT8, FP4**. BF16 is *not* on the S3 list — the baseline's inference should be dropped, not carried |
| S3 peak throughput | — | FP4 given only as "P 级" (petaFLOPS-class). **No TFLOPS figure exists** |
| S3 perf claim vs S2 | "+300%" | Vendor now claims **"5× single-chip performance"**; the Dec 2025 snapshot said 推理性能 x3 倍+. The claim was raised; nothing was measured |
| Scale-up interconnect | "Not public" | **SRLink** (自研), named on the S2 page. Bandwidth and topology still not disclosed |
| Rack-scale system | — | **寰望 SC3-256 超节点** (S3), liquid-cooled, PD-disaggregated, large-EP |
| S2 form factor | PCIe FHFL | 智望 **S2-X1** PCIe FHFL **and 智望 S2-M1 OAM module** |
| S2 status | "Limited MP (1,000 units 2024; 10,000 ramp)" | Vendor claims 规模化量产 / **万片级量产** (10,000-unit class) |
| S2 memory type | "likely HBM2e/HBM3 via CoWoS" | Restated as **not disclosed**. CoWoS is an inference basis, not a vendor statement. The LPDDR6 disclosure applies to S3 only |
| Funding | "~¥30B (~$4.1B)" | **≈¥3B (~US$0.4B)** as of Jan 2026 — a ~10× unit error in the baseline — plus a further **>¥1B** round announced H1 2026, undated |

## 2. LPDDR6 instead of HBM — why it is the architecturally significant item

The S3 page states LPDDR6 with backward compatibility to LPDDR5X, and the homepage claims 国内首款挂载 LPDDR6 的 GPU (first domestic GPU to attach LPDDR6). This is a deliberate divergence from every other Chinese datacenter accelerator in this survey:

- **Cost structure.** LPDDR is commodity mobile DRAM. It supports the vendor's entire positioning (90% per-token cost reduction, 百万Token一分钱) far more directly than any FLOPS claim.
- **Supply chain.** HBM from SK Hynix and Samsung is export-restricted for Chinese customers; that restriction is the binding constraint on Biren, Enflame and MetaX. LPDDR6 is not subject to it.
- **The unquantifiable trade-off.** LPDDR6 per-pin rates are far above LPDDR5X, but total bandwidth is a function of channel count, which the vendor does not publish. With no capacity figure and no bandwidth figure, it is **not possible** to say whether S3's memory system can feed a petaFLOPS-class FP4 datapath. Any bandwidth number attributed to S3 is invented.

**Do not back-fill LPDDR onto S2.** S2's 64 GB is confirmed; its type has never been stated by the vendor.

## 3. 寰望 SC3-256 SuperPOD

| Attribute | Value |
|---|---|
| Names | 寰望 SC3-256 超节点 / "Rise SC3-256 SuperPOD" |
| Base chip | S3 |
| Cooling | Liquid |
| Serving architecture | PD-disaggregated (prefill/decode separation) + large Expert-Parallel (EP) deployment |
| Target workload | 万亿乃至更大参数量 multimodal MoE (trillion-parameter **and larger**) |
| Chip count | **Not stated by the vendor.** "256" appears only in the product name |
| Link technology / bandwidth | Not disclosed |
| Vendor claims | TCO down an order of magnitude at equal performance; comms latency down an order of magnitude; ">20× throughput advantage", footnoted 数据来源于曦望实验室 |

The PD-disaggregation and large-EP framing is the same architectural direction as Huawei CloudMatrix and NVIDIA's disaggregated-serving work: separate the bandwidth-bound prefill phase from the latency-bound decode phase onto differently-provisioned resources. For a part with LPDDR-class memory bandwidth, PD disaggregation is not a differentiator so much as a necessity — decode is capacity-bound and tolerates lower bandwidth, which is exactly where LPDDR6 is competitive.

## 4. First compiler-visible microarchitecture evidence

The FlagTree `third_party/sunrise` backend is the first artifact about this chip that a third party can inspect. It reveals the following. These describe **what the Triton path exposes**, which is a lower bound on hardware capability, not a specification.

| Property | Value | File |
|---|---|---|
| Backend name | `tang` | `backend/compiler.py` |
| Target string | `tang:S2` | `backend/compiler.py` |
| LLVM triple | `stcu-unknown-tang` | `backend/compiler.py` |
| S3 differentiation | `stcuv2` triple + `*_S3.bc` libdevice (comment: "libdivice库,需要区分S2/S3") | `backend/compiler.py` |
| Warp size | 32 | `backend/compiler.py` |
| Defaults | `num_warps=4`, `num_stages=3` | `backend/compiler.py` |
| FP8 dtypes | `('fp8e5',)` — E5M2 only | `backend/compiler.py` |
| `min_dot_size` | (8,8,16) for INT8; (8,8,4) otherwise | `backend/compiler.py` |
| Dot input precision | `ieee` only | `backend/compiler.py` |
| Runtime libs | `libtang.so`, `libtangrt_shared`; root `/usr/local/tangrt` | `backend/driver.py` |
| Device math | OCML (`__ocml_*`), AMD-style | `language/tang/libdevice.py` |
| Bundler | `clang-offload-bundler` under `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-<arch>/` | `backend/compiler.py` |

Architectural reading: warp size 32 and an `ieee`-only dot precision put this closer to an NVIDIA-style SIMT machine than to AMD's 64-wide wavefronts, despite the OCML math library being AMD-derived. The `min_dot_size` triple of (8,8,16) for INT8 versus (8,8,4) otherwise implies a matrix unit whose K-depth quadruples at INT8 — consistent with a tensor core that packs four INT8 operands per FP32-width lane. **This is inference from compiler constants, not a vendor statement**, and is recorded as such.

## 5. Status and schedule — deliberately not upgraded

- **S2: volume production** (vendor: 已实现规模化量产, 万片级量产). Unit count remains a company claim.
- **S3: announced only.** The vendor's own S3 page carries no schedule and no status line. No public evidence exists that S3 has taped out, sampled or shipped.
- A January 2026 executive statement (co-CEO 王勇, 2026-01-28) described S3 R&D as complete, tape-out planned mid-2026, MP by end-2026, and a roadmap of S4 (2027, high-performance) and S5 (2028, security chip). **This could not be independently verified** — the WeChat sources are access-gated — and **no source dated after April 2026 confirms the mid-2026 tape-out occurred.**

## 6. Still not disclosed

Compute-unit count and hierarchy; transistor count; die area; on-chip SRAM; S2 memory type; S2 and S3 memory bandwidth; S3 memory capacity; S3 process node, foundry, packaging and TDP; SRLink bandwidth and topology; SC3-256 chip count and link technology; any FLOPS or TOPS figure for S3; any independent or third-party benchmark. No MLPerf submissions. No Hot Chips, ISCA or ISSCC papers. (Hot Chips 38 runs 2026-08-23…25 and has no scheduled Xiwang talk.)

## 7. Corporate entity — flagged, not asserted

The vendor site footer now reads **浙江曦望智能科技股份有限公司** (Zhejiang; joint-stock company), ICP filing 浙ICP备2025183180号-1. The baseline records 杭州曦望芯科智能科技有限公司, and FlagTree's vendor table still says 曦望芯科. A restructuring or rename is plausible but **no company-registry evidence was obtained**, so it is recorded as an open question.

## Sources added 2026-08-08

- [S3 product page — LPDDR6/5X, PCIe Gen6, FP32/FP16/FP8/INT8/FP4, "P 级" FP4, 仿真实测 99%/98%, 寰望 SC3-256, SIRE layer diagram](https://sunrise-ai.com/products/s3-product)
- [S2 product page — TANG programming model, SIRE, SRLink, 智望 S2-X1 / S2-M1, 辰望 server family, 规模化量产](https://sunrise-ai.com/products/s2-product)
- [sunrise-ai.com homepage and news index — 5× single-chip claim, five hardware lines incl. 熙望 workstations, 曦望完成近30亿融资, 再获超10亿元融资 (undated), footer 浙江曦望智能科技股份有限公司](https://sunrise-ai.com/)
- [FlagTree — README vendor table "Sunrise（曦望芯科）"; changelog 2026/01/23 and 2026/06/30](https://github.com/FlagTree/flagtree)
- [FlagTree sunrise `backend/compiler.py` — triples, warp size, FP8 dtypes, min_dot_size](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/compiler.py)
- [FlagTree sunrise `backend/driver.py` — libtang.so, libtangrt_shared, /usr/local/tangrt](https://raw.githubusercontent.com/FlagTree/flagtree/main/third_party/sunrise/backend/driver.py)
- [Wayback 2025-12-11 sunrise-ai.com — control snapshot ("x3 倍+", no SIRE/LPDDR6/Gen6/SC3-256)](http://web.archive.org/web/20251211220612id_/https://sunrise-ai.com/)
- [Wayback 2026-05-21 sunrise-ai.com — SC3-256 in nav, still no SIRE/LPDDR6/Gen6](http://web.archive.org/web/20260521020029id_/https://sunrise-ai.com/)

# Update — 2026-09-13: Funding chronology resolved (Roadmap)

*scan window: 2026-08-08 → 2026-09-13; classification: Roadmap (financial) — no new hardware facts*

Eastmoney (2026-08-28) reports a new Xiwang funding round — **¥2B raised, post-money valuation ≈¥20B (200亿元)**, nearly double the prior valuation within ~4 months — and, in doing so, retroactively resolves two items the 2026-08-08 update left open:

- The previously **undated** ">¥1B" round (2026-08-08 update, §8) is now dated to **April 2026**, reaching a **>¥10B valuation**.
- **Cumulative funding since the late-2024 SenseTime spin-off is now ≈¥6B (接近60亿元)**.

Investors in the August round: PICC Equity, China Construction Bank Equity, Orient International Assets, Janchor Partners, 中科创星, 同创伟业, 进化论资本, 临芯投资, 毅达资本, 湖畔基金, 弘晖基金; corporate investors CP Group, 九安医疗, 盈峰环境, 三七互娱, 同程旅行. The same report describes the company as operating **three product generations (S1/S2/S3)** with **~400 employees, >80% in R&D**.

**Unit-conversion caution:** an initial automated read of this article rendered the valuation as "¥200B"; the correct figure, cross-checked against the ¥2B raise itself, is **¥20B (200亿元)**.

**No S3 tape-out, sampling, or new hardware spec was found in this window** — the S3 status recorded in the 2026-08-08 update (announced only, no confirmed tape-out) is unchanged.

## Sources — added 2026-09-13
- Eastmoney — Xiwang ¥2B round, ≈¥20B valuation (2026-08-28): https://finance.eastmoney.com/a/202608283858780525.html
