# FuriosaAI Hardware Architecture Investigation

*as_of: 2026-09-13*

---

## Overview

FuriosaAI is a South Korean AI semiconductor startup (founded 2017, Seoul) that has shipped two generations of inference NPU silicon: **Warboy** (Gen 1, Samsung 14nm, 2021) and **RNGD** (Gen 2, TSMC 5nm, mass production January 2026). Both chips target data-center inference workloads; RNGD specifically targets LLM/multimodal inference and is FuriosaAI's flagship product heading into a 2027 IPO.

---

## Generation 1: Warboy

### Compute Engine

| Parameter | Value |
|-----------|-------|
| Process | Samsung 14nm (mass-produced by Samsung Foundry) |
| Die area | 180 mm² |
| Transistors | ~5 billion |
| Clock | 2.0 GHz |
| Peak INT8 | 64 TOPS |
| Compute unit | PE array with explicit SIMD vector engine |

Warboy is a **vision-focused NPU** designed for image classification, object detection, and segmentation (ResNet, EfficientDet, YOLO variants). Its microarchitecture is a PE (Processing Element) systolic-array-style GEMM engine combined with a vector/element-wise unit for activation and normalization. The design emphasizes compiler-managed execution — no caches, SW-managed SRAM.

### Memory

| Parameter | Value |
|-----------|-------|
| On-chip SRAM | 32 MB |
| Off-chip memory | 16 GB LPDDR4X |
| Memory bandwidth | 66 GB/s |

The LPDDR4X choice (vs HBM) reflects Warboy's origin as a vision NPU where model weight sizes are modest (50–200 MB). The on-chip SRAM holds activation buffers and frequently reused kernel tiles.

### Package / Host Interface

| Parameter | Value |
|-----------|-------|
| Form factor | PCIe x8 card |
| Host interface | PCIe Gen4 x8 |
| TDP | ~50W |
| Cooling | Passive / light active |

Warboy was designed by FuriosaAI and **taped out via SemiFive's ASIC platform** (SemiFive is a subsidiary of SK Telecom providing chiplet/SoC design services). Samsung Foundry manufactured the chip in 14nm LPP.

---

## Generation 2: RNGD ("Renegade")

RNGD is FuriosaAI's major architectural departure — from a vision NPU to a full LLM/multimodal inference accelerator. It introduced the **Tensor Contraction Processor (TCP)** architecture, presented at Hot Chips 2024 and published at MICRO 2025.

### Tensor Contraction Processor (TCP) Architecture

The fundamental insight behind TCP: modern deep learning — including attention, GEMM, convolution, and batched operations — is reducible to **tensor contraction**, a higher-dimensional generalization of matrix multiplication. Rather than specializing hardware for GEMM (as most NPUs do), RNGD builds a more flexible primitive that handles arbitrary-rank tensor contractions.

This is architecturally significant because:
1. Attention (QKV projections, softmax, attention scores) maps directly as tensor contractions — no decomposition into matrix-vector products needed
2. Convolutions are a special case of tensor contraction — same hardware handles CNN and transformer layers
3. Future model architectures (mixture-of-experts, multi-head latent attention) fit naturally without ISA extensions

### Compute Engine

| Parameter | Value |
|-----------|-------|
| Process | TSMC 5nm |
| Transistors | ~40 billion |
| Clock | 1.0 GHz |
| Peak INT8 | 512 TOPS |
| Peak INT4 | 1,024 TOPS |
| Peak FP8 | 512 TOPS (FP8-compatible) |
| Processing Elements | 8 PEs × 64 "slices" each (512 slices total) |
| Architecture | Tensor Contraction Processor (TCP) |

The **8 PEs, each with 64 slices** form the physical instantiation of the TCP. Each slice is a multi-dimensional MAC unit capable of contracting tensor dimensions — not merely a 2D dot product. The compiler maps operator tensor indices to hardware slice dimensions.

### Memory

| Parameter | Value |
|-----------|-------|
| Off-chip memory | 2× HBM3 modules |
| HBM3 capacity | 24 GB (2×12 GB) |
| Memory bandwidth | 1.5 TB/s |
| On-chip SRAM | Not publicly disclosed (compiler-managed) |

The jump from Warboy's 66 GB/s LPDDR4X to RNGD's 1.5 TB/s HBM3 (23× bandwidth increase) enables LLM inference at reasonable batch sizes. The 24 GB HBM3 capacity holds 7B–13B parameter models in FP8 and supports tensor-parallelism across two RNGD cards for 70B parameter models.

### Multi-Tenancy: NPU Partitioning

RNGD supports hardware-level partitioning into **2, 4, or 8 isolated virtual NPUs** per physical chip:
- Each partition has its own dedicated PE slices and memory bandwidth allocation
- Full hardware isolation — suitable for Kubernetes multi-tenant workloads
- Enables mixing multiple models on a single card (e.g., 4 smaller models on 1 RNGD)

### Host Interface / Package

| Parameter | Value |
|-----------|-------|
| Host interface | PCIe Gen5 x16 |
| TDP | 180W |
| Form factor | PCIe accelerator card |
| Production start | January 2026 |

### NXT RNGD Server

FuriosaAI also announced the **NXT RNGD server**, a complete inference system:
- 4 RNGD cards per server
- Total server TDP: ~3 kW (vs. >10 kW for NVIDIA DGX H100)
- Benchmark (LG AI Research, EXAONE 3.5 32B): 60 tokens/sec at 4K context, 50 tokens/sec at 32K context
- Power efficiency: 3.75× more tokens per rack watt vs. H100-based GPU rack (same power budget)

---

## Architecture Comparison: Warboy vs. RNGD

| Dimension | Warboy | RNGD |
|-----------|--------|------|
| Process | Samsung 14nm | TSMC 5nm |
| Transistors | 5B | 40B |
| Die area | 180 mm² | Not disclosed (~600+ mm² est.) |
| Peak INT8 | 64 TOPS | 512 TOPS |
| Memory type | LPDDR4X | HBM3 |
| Memory BW | 66 GB/s | 1.5 TB/s |
| Memory capacity | 16 GB | 24 GB |
| Host interface | PCIe Gen4 x8 | PCIe Gen5 x16 |
| TDP | ~50W | 180W |
| Architecture | PE array (vision-first) | TCP (LLM-first) |
| Target workload | CNN inference | LLM/multimodal inference |

---

## Foundry / Supply Chain

| Generation | Foundry | Node | Notes |
|------------|---------|------|-------|
| Warboy | Samsung Foundry | 14nm LPP | Via SemiFive ASIC platform |
| RNGD | TSMC | 5nm N5 | Shifted from Samsung; mass prod. Jan 2026 |

FuriosaAI's shift from Samsung 14nm to TSMC 5nm for RNGD reflects an industry trend (TrendForce 2024 reported multiple Korean AI IC design firms adopting dual-foundry models). TSMC's 5nm gave FuriosaAI access to higher density (40B transistors) and the HBM3 interposer ecosystem.

---

## Resources

- [FuriosaAI RNGD Product Page](https://furiosa.ai/rngd)
- [FuriosaAI Warboy Product Page](https://furiosa.ai/warboy)
- [FuriosaAI Developer Center — RNGD Overview](https://developer.furiosa.ai/latest/en/overview/rngd.html)
- [Hot Chips 2024: RNGD Architecture (Chips & Cheese)](https://chipsandcheese.com/p/furiosaais-rngd-at-hot-chips-2024-accelerating-ai-with-a-more-flexible-primitive)
- [MICRO 2025: FuriosaAI RNGD TCP Paper](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-micro25.pdf)
- [HPCwire: Tensor Contraction Architecture Deep-Dive](https://www.hpcwire.com/2025/09/30/the-fast-and-the-furiosaai-korean-chip-startup-takes-aim-at-nvidia-gpus-with-tensor-contraction-architecture/)
- [Warboy Specs Page](https://furiosa.ai/warboy/specs)
- [SemiFive + Warboy Partnership](https://www.eenewseurope.com/en/semifive-helps-furiosaai-warboy-processor-get-to-market/)

---

# Investigation Update — 2026-08-08

*Scan window: 2026-04-10 → 2026-08-06. Supersedes parts of the 2026-04-05 baseline above. Prior-generation text is retained unchanged for historical traceability; corrections are stated explicitly here.*

## A. Baseline corrections to RNGD (pre-existing repo errors)

These are **not news**. They are errors in the 2026-04-05 baseline recorded above, found while verifying the 2026 update against FuriosaAI's own product page (https://furiosa.ai/rngd) and Developer Center 2026.3.0.

| Parameter | 2026-04-05 baseline (wrong) | Corrected value | Source |
|---|---|---|---|
| HBM3 capacity | 24 GB (2 × 12 GB) | **48 GB** (2 stacks via CoWoS-S @ 6.0 Gbps) | furiosa.ai/rngd |
| On-chip SRAM | "Not publicly disclosed" | **256 MB @ 384 TB/s** | furiosa.ai/rngd |
| 512 headline figure | "512 TOPS INT8" | **512 TFLOPS FP8**, derived by the vendor as 64 TFLOPS FP8 × 8 PEs | furiosa.ai/rngd |
| Partition table | 24/12/6/3 GB per vNPU | **48/24/12/6 GB** per vNPU | recomputed |
| TDP | 180 W (single figure) | **180 W** (product page + all 2026 press) vs **150 W** (developer docs) — unreconciled vendor discrepancy | furiosa.ai/rngd; developer.furiosa.ai |

Confidence notes:
- The **FP8 512 TFLOPS** figure is confirmed against a vendor primary source.
- **256 TFLOPS BF16 / 512 TOPS INT8 / 1,024 TOPS INT4** appear in the 2026.3.0 developer documentation but were **not independently confirmed** in the verification pass. They are retained at reduced confidence. The prior repo baseline's 1,024 INT4 TOPS was derived from the mislabelled 512-as-INT8 figure and should not be treated as independently sourced.
- SR-IOV partitioning into 2/4/8 instances is **confirmed**.
- PCIe Gen5 x16 with card-to-card P2P, 1.0 GHz clock: confirmed.

## B. Third-generation accelerator — Broadcom co-development (announced 2026-05-27)

**Status verb, precisely: announced partnership / roadmap item.** No product name, no codename, no tape-out, no silicon, no sampling, no shipping.

FuriosaAI's verbatim description: the platform is "incorporating HBM4/4E, 2nm process technology, and high-speed inter-chip networking," built "by pairing Furiosa's TCP architecture with Broadcom's market-leading XPU Technology and IP Platform, Ethernet scale-up and fabric switches." Charlie Kawwas (President, Broadcom Semiconductor Solutions Group) is quoted in the release. Broadcom's contribution: XPU IP, advanced multi-die packaging, and Ethernet scale-up/fabric switching. The design is described as a **multi-die chiplet system-in-package**; the scale-up fabric is described as supporting an **all-to-all-capable topology** for MoE routing traffic.

Korean-language coverage on 2026-05-28 (Yonhap AKR20260528119900017; Digital Today; The AI) adds two facts absent from the English post and consistently reported across outlets:
1. The compute die is on **TSMC** 2nm.
2. **Sampling is targeted for H1 2028.**

**Explicitly NOT disclosed for Gen 3:** peak throughput at any dtype, HBM capacity, memory bandwidth, TDP, interconnect bandwidth, scale-up domain size (chips/nodes/rack), die count, die sizes, mass-production date, host interface, form factor, on-chip SRAM.

**Rejected as unreliable.** theoutpost.ai (AI-generated aggregator) asserts "12 memory sites, potentially 432 GB", "two massive 2nm compute chiplets plus two IO controllers", and "3.5D XDSiP". It self-attributes the sampling date to Wccftech; none of these specs appear in vendor or wire coverage. **Do not record.**

**Sourcing caveat.** No Broadcom-issued press release confirming the partnership was located on broadcom.com. The announcement is FuriosaAI-side with a Broadcom executive quote. DataCenterDynamics coverage returned HTTP 403 on direct fetch; only its indexed snippet was retrieved, so it is listed for completeness rather than as independent verification.

**Architectural significance for the survey.** Gen 3 is the first FuriosaAI part with a scale-up fabric of any kind. Warboy and RNGD are PCIe-only; multi-card parallelism is host-mediated. This places FuriosaAI in the same structural category as Google TPU v8t (Broadcom) and OpenAI Titan (Broadcom): an Ethernet-based scale-up domain co-designed with Broadcom, positioned around data movement and memory access rather than peak FLOPS.

## C. RNGD production and deployment status

- **Mass production formally declared** at RENEGADE Summit 2026 (2026-05-13), TSMC 5nm. Supply chain named: TSMC (foundry), SK hynix (HBM3).
- **NXT RNGD Server capacity revised**: FuriosaAI's 2026-07-07 Equinix announcement describes the server as holding **up to 8 RNGD accelerators in a 3 kW-class system**. The prior baseline records 4 cards / 3 kW. Both figures are FuriosaAI's own; the 8-card configuration is the newer one.
- **Sweden (2026-08-04)** — largest disclosed deployment. FuriosaAI + I/ONX HPC + Velox, Stockholm; **15 MW total** site. Phase 1 = **1,800 RNGD accelerators**, 2 MW online early 2027; further 8 MW later in 2027; **7,000+ additional accelerators** planned in subsequent phases.
- **Equinix Lisbon LS2 (2026-07-07)**: RNGD servers installed for European enterprise evaluation, announced at RAISE Summit Paris. European flagship office established in Portugal, 2026-04-10.
- **Samsung SDS NPUaaS (2026-07-20)**: Korea's first domestic NPU-as-a-Service (Samsung SDS's characterization), on Samsung Cloud Platform, Dongtan data center, subscribable in **1/2/4/8-card configurations** — a direct commercialization of RNGD's hardware partition modes. Vendor-quoted specs: 512 TFLOPS FP8, 1.5 TB/s, 180 W. Serves Qwen3 and gpt-oss 120B. Partnership work began September 2025.
- **Deployment commitments** announced at RENEGADE Summit 2026: LG AI Research, Samsung SDS, LG U+, Upstage, MegazoneCloud.
- **Daum (Kakao portal) search overviews (2026-07-31)**: RNGD reported serving AI-generated search overviews at millions-of-users scale. Headline-level only — low confidence.

## D. New benchmark (vendor-published, 2026-08-06)

White paper: RNGD vs **4× NVIDIA RTX PRO 6000 Blackwell Server Edition** (bare metal), Qwen3-32B FP8, via Lablup **Backend.AI**. Claims: 1.3×–1.5× higher throughput/watt at all concurrency levels; 95% of the GPU setup's peak throughput at 256-request concurrency; 30–44% lower power draw; TTFT under 1 s through concurrency 32 vs over 2.9 s for the GPU control. **Vendor claim — not independent, not a standardized benchmark.**

## E. Negative findings (recorded deliberately)

1. **No MLPerf results exist for RNGD.** FuriosaAI is not among the 24 submitting organizations for MLPerf Inference v6.0 (results published 2026-04-01).
2. **No FuriosaAI talk appears on the Hot Chips 38 program** (Aug 23–25, 2026, Stanford).
3. **"RNGD-S" is not a confirmed product.** No such variant appears on furiosa.ai/rngd, in the 2026.3.0 developer docs, or in any 2026 press release. Current product names: RNGD (PCIe card), NXT RNGD Server, Warboy / Vision NPU.

## F. Corporate (hardware-adjacent)

**Series D is NOT closed** as of late July 2026. A secondary Korean trade report (The Bell, relayed 2026-07-21 by ai-market-watch.com) says the round is "nearing the close" at **KRW 800B (~$600M)** — above the $500M target in the baseline — at a **KRW 4 trillion (~$3B)** valuation, with TS Investment named. **Confidence: LOW.** Secondary aggregation of a Korean trade report; no FuriosaAI press release confirms a close.

## Sources added 2026-08-08

- https://furiosa.ai/rngd — vendor product page (48 GB HBM3, 256 MB SRAM @ 384 TB/s, 512 TFLOPS FP8 = 64 × 8 PEs, 180 W, PCIe P2P)
- https://developer.furiosa.ai/latest/en/ — Developer Center 2026.3.0 (150 W, 1.0 GHz, BF16/INT8/INT4 figures)
- https://furiosa.ai/blog/furiosaai-partners-with-broadcom-to-build-next-generation-inference-platform-for-the-agentic-era
- https://www.newstheai.com/news/articleView.html?idxno=20758 — The AI (Korean), 2026-05-28
- https://www.digitaltoday.co.kr/news/articleView.html?idxno=669758 — Digital Today (Korean), 2026-05-28
- https://www.yna.co.kr/view/AKR20260528119900017 — Yonhap wire, 2026-05-28 (indexed snippet only; host not directly fetchable)
- https://www.datacenterdynamics.com/en/news/furiosaai-partners-with-broadcom-for-development-of-ai-inference-chips/ — HTTP 403 on direct fetch; snippet only, NOT independently verified
- https://furiosa.ai/blog/experience-renegade-summit-2026
- https://furiosa.ai/blog/furiosaai-and-samsung-sds
- https://furiosa.ai/blog/furiosaai-equinixs-lisbon-data-center-press-release
- https://furiosa.ai/blog/furiosaai-partners-with-i-onx-and-velox-for-new-15-mw-ai-data-center-in-sweden
- https://furiosa.ai/blog/white-paper-benchmarking-rngd-on-backend-ai
- https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ — negative result
- https://hotchips.org/ — negative result
- https://www.ai-market-watch.com/news/furiosaai-nears-completion-of-800-billion-won-series-d-valuation-at-4-trillion-w-eawtpx — LOW confidence

---

## G. Investigation Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*No new silicon or spec this window (Gen 3 Broadcom co-development remains announced-only, H1 2028 sampling target unchanged). Two items: a Korean regulatory/business development inside the window, and a pre-baseline correction (a Sept-2025 OpenAI demo event that the repo had never recorded).*

### G.1 Korean export-control change and "living showroom" ask (2026-09-10)

Crypto Briefing (2026-09-10) reports that Korean fabless AI chip startups — **explicitly naming both FuriosaAI and Rebellions** — are lobbying Seoul to "orchestrate large-scale domestic deployments that can serve as living showrooms," i.e., government-brokered reference deployments they can cite when selling internationally.

- **New regulatory fact**: effective **2026-09-01**, Korea added high-performance AI chips to its **strategic goods export-control list**, requiring government approval for overseas shipment — a compliance burden the article frames as disproportionately affecting small startups versus incumbents like NVIDIA.
- **Scale context (vendor-agnostic, collective figure)**: Korean AI-chip startups' overseas contracts collectively total only **~$30 million**, against NVIDIA's tens-of-billions-per-quarter datacenter revenue; cited example deals are small (a ~$0.5M wheelchair-platform contract, a $2.5M water-monitoring system) — **not attributed specifically to FuriosaAI**, so do not read this as a FuriosaAI-specific revenue figure.
- **Government response already in motion**: Ministry of Science and ICT's **K-AI Semiconductor Growth Forum** and the **K-NPU Project** (initiated late 2025), plus a stated plan for **8.4 GW of AI datacenter capacity**.
- This item is **corporate/regulatory context, not a hardware or shipment fact**, and is equally relevant to the `rebellions-atom` entry (both companies are named in the same article).

### G.2 Pre-baseline correction: FuriosaAI × OpenAI demo, 2025-09-11 (newly recorded)

Not previously in this repo. At the **grand opening of OpenAI's Seoul office (2025-09-11)**, FuriosaAI and OpenAI staged a **live demo**: OpenAI's open-weight **gpt-oss-120b** model running on just **two RNGD cards** in **MXFP4** precision (Yahoo Finance, republishing a 2025-09-17 piece; total funding cited there as $246M, i.e. pre-Series-C-close figures — stale, superseded by the Series C $125M July 2025 close already on record). **This was a technical demonstration staged at a partner event, not a shipment, sale, or supply agreement with OpenAI** — the pre-verified research-pass claim of "shipments to OpenAI" is **not supported by this source and should not be written as a shipment/customer relationship.** It does, however, corroborate the already-recorded fact that gpt-oss-120b is one of the models the TCL/2026.3 stack targets.

### G.3 Series D — still not confirmed closed

No August/September 2026 source was found confirming a Series D close. The KRW 800B (~$600M) figure relayed 2026-07-21 (The Bell, via ai-market-watch.com, LOW confidence) remains the most recent figure on record. **Note**: this repo's own pre-verified-fact list for this scan cited "750B-won Series D" — no source matching that specific figure was found in this pass; it is close to but does not match the KRW 800B figure already on record. Do not treat either figure as confirmed.

### Searched and absent

- No confirmed FuriosaAI RNGD 2026 cumulative unit-volume figure (e.g., "~20,000 units") — the only production-rate figures on record remain the pre-window ~1,000/month → 2,000–3,000/month by end-2026 target.
- No Series D close announcement.
- No Gen 3 (Broadcom) tape-out/sampling update.
- No Hot Chips 38 FuriosaAI talk; no MLPerf submission.

### Sources added 2026-09-13

- [Crypto Briefing — South Korean AI chipmakers ask government for domestic deployment references (2026-09-10)](https://cryptobriefing.com/south-korean-ai-chipmakers-deployment-references/)
- [Yahoo Finance — AI Unicorn FuriosaAI With $246M In Funding Teams With OpenAI To Run 120B Model On Just 2 Cards (republished; underlying event 2025-09-11)](https://finance.yahoo.com/news/ai-unicorn-furiosaai-246m-funding-123111208.html)
- REJECTED: https://theoutpost.ai/news-story/broadcom-partners-with-furiosa-ai-on-2nm-ai-accelerator-chip-with-hbm-4e-memory-for-inference-26665/ — AI-generated aggregator; specs traced to Wccftech speculation
