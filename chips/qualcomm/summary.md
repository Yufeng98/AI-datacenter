# Qualcomm Cloud AI 100/200 — Summary

**Device class:** Cloud AI Inference Accelerator
**Manufacturer:** Qualcomm
**Deployment:** Data center inference servers (PCIe card); rack-scale from AI200 onward
**Research date:** 2026-04-05
**Last updated:** 2026-09-13

NOTE: This summary covers Cloud AI 100 and Cloud AI 200 data center products only, not Hexagon DSP or mobile NPU.

NOTE (2026-08-08): Since June 2026 Qualcomm brands its entire data center line **Qualcomm Dragonfly**. The
Dragonfly portfolio named in Qualcomm's own press release is four products — the **C1000** CPU and the
**AI200 / AI250 / AI300** accelerators (AI200 and AI250 retroactively rebranded from their October 2025 launch).
See the *Dragonfly Portfolio Update (June–July 2026)* section at the end of this document.

---

## What It Is

The Qualcomm Cloud AI 100 is a data center inference PCIe card built on the AIC100 SoC, which contains 16 Hexagon AI cores each with a Tensor Unit (HMX matrix engine), Vector Unit (HVX), and Scalar Processor. The Ultra card puts 4 SoCs on one PCIe card for 870 TOPS and 128 GB LPDDR4X — competitive memory capacity for large LLMs.

---

## Key Specifications

| SKU | AI Cores | INT8 TOPS | SRAM | DRAM | TDP |
|---|---|---|---|---|---|
| Standard | 16 | 400 | 144 MB | 32 GB, 136 GB/s | 75 W |
| Ultra | 64 (4 SoC) | 870 | 576 MB | 128 GB, 548 GB/s | 150 W |
| AI200 (Dragonfly) | not disclosed | not disclosed | not disclosed | 768 GB/card LPDDR (Oct 2025 launch figure — see conflict note) | 160 kW rack (Oct 2025 launch figure) |
| AI250 (Dragonfly) | not disclosed | not disclosed | not disclosed | HBC Gen 1; claimed 133 TB/s per card (unaudited) | not disclosed |
| AI300 (Dragonfly) | not disclosed | not disclosed | not disclosed | HBC Gen 2 (3D-stacked near-memory); capacity not disclosed | not disclosed |

*AI200/AI250/AI300 status as of 2026-08-08: AI200 "sampling in fiscal 2026 on LPDDR5x" (Investor-Day-reported);
AI250 commercial sampling expected mid-2027; AI300 commercial sampling expected 2028. Qualcomm has not disclosed
peak FLOPS/TOPS for any of the three. The 768 GB/card and 160 kW rack figures date from the October 2025 AI200/AI250
launch and were neither restated nor contradicted in June 2026; note that The Register (2026-06-30) attributes
768 GB/card on LPDDR5x to **AI250**, so per-card capacity attribution is an unresolved source conflict.*

---

## AI Core Architecture

Each core has three execution units:
- **Tensor (HMX):** 8192 INT8 ops/cycle, 4096 FP16 ops/cycle
- **Vector (HVX):** 512 INT8 ops/cycle, 700+ instructions
- **Scalar (Q6 VLIW):** 4-way, 6 threads — control and dispatch

Three NoCs separate compute, memory, and configuration traffic.

---

## Software Stack

```
PyTorch / ONNX / TensorFlow
  → qaic-compile (closed) → QNN context binary
  → libQAic (open SDK) → QAIC driver (open, upstream Linux)
  → AIC100 hardware
```

Also: ONNX Runtime QNN EP, ExecuTorch backend.

---

## Key Strengths

1. **128 GB DRAM per Ultra card** — enables large LLM inference without model sharding across cards
2. **Upstream Linux driver** — broad enterprise server compatibility
3. **ONNX Runtime integration** — easy migration from GPU inference
4. **Strong TOPS/W** — 400 TOPS at 75 W for standard SKU (5.3 TOPS/W)
5. **Large on-chip SRAM** — 576 MB Ultra card reduces DRAM pressure for KV-cache

---

## Sources

- https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Architecture/
- https://www.qualcomm.com/artificial-intelligence/data-center/cloud-ai-100-ultra
- https://docs.kernel.org/accel/qaic/aic100.html
- https://arxiv.org/html/2507.00418v1

---

## Dragonfly Portfolio Update (June–July 2026)

*Updated 2026-08-08. Primary sources: Qualcomm press releases of 2026-06-24 (Dragonfly data center roadmap;
diversification strategy; Meta CPU agreement; Modular acquisition) and 2026-07-29 (Q3 FY2026 results; completion
of the Modular acquisition); Modular blog 2026-07-29. Independent: The Register 2026-06-24 and 2026-06-30.
Analyst-only: Futurum Group Investor Day writeup.*

### Branding

At its Investor Day on **2026-06-24** Qualcomm rebranded its data center line **Qualcomm Dragonfly**. The four
products named in the Dragonfly press release are the **C1000** CPU and the **AI200**, **AI250** and **AI300**
accelerators; AI200 and AI250 are the October 2025 parts under new branding.

A *separate* four-way sequence also appeared at Investor Day — connectivity, custom silicon, AI accelerator, CPU.
That is **revenue-ramp sequencing from the financial deck, not the Dragonfly product taxonomy**. Of its fiscal
timings only "custom silicon revenue from Q1 FY2027" and "AI accelerator revenue in H2 FY2027" are attested (via
Futurum), and C1000 production in H2 2028 is corroborated by the Qualcomm/Meta press release.

### HBC — High Bandwidth Compute (the architectural bet)

HBC is Qualcomm's 3D-stacked **near-memory computing** scheme: DRAM stacked directly over XPU logic and connected
by TSVs (mechanism described independently by The Register, 2026-06-30). It is an explicit bet *against* HBM.

Qualcomm-stated, unaudited technology claims:

- **6× bandwidth per watt versus HBM**
- **200× capacity per watt versus SRAM**

| Part | Memory architecture | Qualcomm bandwidth claim | Status |
|---|---|---|---|
| AI200 | LPDDR5x (no HBM) | baseline | Sampling in FY2026 (Investor-Day-reported); no production-shipment claim |
| **AI250** | **HBC Gen 1** | "industry-leading 133 TB/s per card"; "18× increase in effective memory bandwidth compared to AI200" | Commercial sampling expected **mid-2027** |
| **AI300** | **HBC Gen 2** (3D-stacked near-memory) | "54× increase over AI200" in effective memory bandwidth; 4×–8× better performance-per-watt vs existing GPU-based architectures | Commercial sampling expected **2028** |

**CRITICAL CAVEAT — treat all HBC bandwidth numbers as unaudited vendor marketing.** These are "effective"
bandwidth multipliers with **undisclosed methodology**. The Register (2026-06-30) reports that Qualcomm declined
to disclose peak FLOPS for either AI250 or AI300 and declined to explain how the multipliers are computed, and
argues the figures are inflated by definition rather than physical: Qualcomm's implied ~414 TB/s aggregated across
56 AI200 chips would require an implausible ~6,720-bit bus on standard LPDDR5x. No independently measured
bandwidth or throughput figure exists for AI200, AI250 or AI300.

**Baseline conflict:** Qualcomm's own PR states the AI300 figure as 54× versus **AI200**; The Register renders the
same claim as 54× versus **AI250**. Prefer the PR.

AI300 is described as an air- and direct-liquid-cooled **rack-level platform** that scales over UALink and ESUN.

### Dragonfly C1000 — new data center CPU line (announced only)

Qualcomm's first datacenter CPU line, announced 2026-06-24, commercial availability expected **2028**:

| Attribute | Value |
|---|---|
| Cores | 250+ custom Qualcomm Oryon cores |
| Packaging | Chiplet design |
| Clock | >5 GHz (Oryon) |
| Perf/W | ">2× better performance per watt" vs competitive server CPUs (Qualcomm estimate); The Register additionally reports a "30 percent more speed" claim |
| Host/fabric I/O | >2 TB/s PCIe Gen 7 connectivity, plus CXL |
| Configurations | agentic · general-purpose virtualization · AI head node |
| Process, TDP, cache, memory channels | not disclosed |

**Meta is a named C1000 customer.** "Qualcomm and Meta Announce Strategic Multi-Generation Agreement on Data
Center CPUs" (2026-06-24) has Qualcomm "in production starting in the second half of 2028". No volumes, pricing or
binding terms were disclosed — this is a supply agreement, not a deployment.

### Interconnect (new for Qualcomm in this survey)

- **Scale-up:** UALink and **ESUN** (Ethernet for Scale-Up Networking) — Qualcomm is using open standards rather
  than a proprietary fabric.
- **Scale-out:** copper and optical Ethernet, 800G and 1.6T, with stated reach from intra-data-center up to a
  **20 km campus** span.
- Over **35 named ecosystem supporters**, including Meta, Arista, Supermicro, Lenovo, Samsung SDS and SK hynix America.
- Per-chip scale-up bandwidth, per-rack chip counts and maximum system scale: **not disclosed**.

### Software stack — Qualcomm acquires Modular (COMPLETED)

Announced 2026-06-24 ("Qualcomm to Acquire Modular", close guided to H2 2026) and **completed 2026-07-29**,
confirmed by Qualcomm's newsroom release "Qualcomm Completes Acquisition of Modular" and by Modular's own blog
post of the same date. Financial terms not disclosed.

- **Mojo, MAX and Modular Cloud continue as products and brands** under Qualcomm.
- **Chris Lattner** became Qualcomm EVP of Advanced AI Software and Platforms.
- Qualcomm's June acquisition PR does **not** itself name Mojo or MAX; it describes "an open, AI-native software
  stack" running "across CPU, GPU, NPU, and custom ASIC architectures". The Mojo/MAX identification comes from
  Modular's side.

**Framing:** this adds an open, MLIR-based, silicon-agnostic compute layer *alongside* Qualcomm's existing closed
`qaic-compile` → QNN context binary → `libQAic` path. Qualcomm has **not** said Modular replaces that stack, and
no source describes a Mojo/MAX backend for AI200/AI250/AI300. Calling it a replacement is an overstatement.

Separately, Qualcomm and Hugging Face expanded their relationship "from device to cloud" on 2026-06-24.

### Cloud AI 100 SDK (quic/efficient-transformers)

Incremental and generative-media-focused, **not** architectural, and still **Cloud AI 100-targeted** — the release
notes show no AI200/AI250 support:

| Release | Date | Contents |
|---|---|---|
| v1.21.0 | 2025-12-22 | FLUX.1-schnell and WAN 2.2 diffusion support |
| v1.22.0 | 2026-06-18 | WAN 2.2 dual-stage high/low-noise transformers; first-block-caching infrastructure for Diffusers models; blocked-KV attention; layerwise ONNX export for large MoE; moves to HF Transformers 5.5.4 and Python 3.12 |
| v1.22.8.0 | 2026-08-26 | Qwen3.5/Qwen3.6/Gemma4/GLM4 model support; layerwise API cleanup (`CustomLoader`); MoE export RAM reduction via weight aliasing; CCL support extended to more MoE/VLM models. Still Cloud AI 100-only — no AI200/AI250 support |

### Financial framing (attribution matters)

- **Confirmed by Qualcomm's own PR** ("Qualcomm Accelerates Diversification…", 2026-06-24): data center revenue
  **">$15 billion by fiscal 2029"**, against a combined **~$1.7T addressable market by 2030** across all segments.
- **Investor-Day-reported via Futurum only (medium confidence, not in any Qualcomm release):** a $5B FY2027
  data-center target; ≥$1B of it from two unnamed global-scale hyperscaler custom-silicon customers; and an FY2029
  TAM split of $680B accelerators / $200B CPU / $115B custom silicon.
- **Q3 FY2026** (reported 2026-07-29): revenue **"$9.9 billion"** per Qualcomm. The release guides non-handset
  revenue *including Data Center* to accelerate from 24% YoY growth in FY2026 to **>60% in FY2027**. Data center is
  not yet a materially reported segment. Any "down 4% YoY" or "first shipments December 2026" claim is **not**
  supported — the latter is an inference from "meaningful revenue starting in Q1 FY2027" (Qualcomm FQ1'27 =
  Oct–Dec 2026), not a Qualcomm statement.

### Negative findings (both independently confirmed)

- **Qualcomm has no talk on the Hot Chips 2026 (HC38) program.** The program page lists the conference as
  **August 24–25, 2026**.
- **Qualcomm is not among the 24 MLPerf Inference v6.0 submitters** (results published 2026-04-01).
- Consequently there is **no new independent benchmark data** for Cloud AI 100 / AI200 / AI250 in this window.

### Still not disclosed / unresolved

- AI300 memory capacity, process node, TDP and rack power.
- AI250 per-card capacity — Qualcomm attributes 768 GB/card to AI200 (Oct 2025); The Register attributes 768 GB
  LPDDR5x to AI250. Unresolved.
- Peak FLOPS/TOPS for AI200, AI250 and AI300 (Qualcomm explicitly declined).
- Whether AI200 has shipped to any customer beyond sampling.
- Identities of the two hyperscaler custom-silicon customers.
- Any 2026 confirmation that the HUMAIN 200 MW Riyadh deployment (announced Oct/Nov 2025) has begun. **Partially resolved 2026-09-13**: Adobe confirmed migrating "regional AI data captioning workloads" onto HUMAIN's Dragonfly-accelerated infrastructure (2026-08-31) — the first confirmed production workload, though total facility scale/utilization remains undisclosed. See the 2026-09-13 update section below.
- Whether the June 2026 "commercial sampling expected mid-2027" wording for AI250 represents a slip from the
  October 2025 messaging is **plausible but unverified** — the October 2025 launch PR could not be retrieved, and
  The Register still describes AI250 as "launching 2027". Not asserted here.
- A 2026-06-16 report that Qualcomm was circling Tenstorrent in a ~$10B deal is **rumor only** and is deliberately
  excluded from this survey.

### Update sources (2026-08-08 update)

- https://www.qualcomm.com/news/releases/2026/06/qualcomm-unveils-comprehensive-data-center-roadmap-for-the-agent
- https://www.qualcomm.com/news/releases/2026/06/qualcomm-to-acquire-modular
- https://www.qualcomm.com/news/releases/2026/07 (listing confirming "Qualcomm Completes Acquisition of Modular", 2026-07-29)
- https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular
- https://www.qualcomm.com/news/releases/2026/06/qualcomm-accelerates-diversification-with-comprehensive-strategy
- https://www.qualcomm.com/news/releases/2026/06/qualcomm-and-meta-announce-strategic-multi-generation-agreement-
- https://www.qualcomm.com/news/releases/2026/07/qualcomm-announces-third-quarter-fiscal-2026-results
- https://www.theregister.com/systems/2026/06/30/qualcomms-proposed-solution-to-catch-up-in-ai-infra-bury-the-compute-under-the-dram/5264071
- https://www.theregister.com/systems/2026/06/24/qualcomm-claims-its-not-too-late-for-dragonfly-to-land-in-datacenters/5261758
- https://github.com/quic/efficient-transformers/releases
- https://futurumgroup.com/insights/qualcomms-data-center-reentry-at-investor-day-2026-arrives-just-in-time-for-the-inference-decode-prize/
- https://hotchips.org/program/conference/
- https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/

---

## Update (2026-09-13) — HUMAIN deployment confirmed, Qualcomm–AWS collaboration, SDK bump

*Scan window 2026-08-08 → 2026-09-13. Classification: **Roadmap** (deployment/customer confirmations) + **Moderate**
(SDK version bump). No new AI200/AI250/AI300 hardware specs — core count, clock, process node, TDP and peak
FLOPS/TOPS remain **not disclosed**. WebSearch was unavailable this session (budget exhausted); sourced from
direct fetches of Qualcomm newsroom listings and the quic/efficient-transformers GitHub releases page.*

**HUMAIN — first confirmed production workload (2026-08-31).** "Adobe Becomes First Global Software Company to
Migrate AI Workloads onto HUMAIN Platform, Accelerated by Qualcomm" (Qualcomm newsroom, Riyadh) confirms Adobe is
running **"regional AI data captioning workloads"** on HUMAIN's **Qualcomm Dragonfly™** infrastructure, for data/
compute-residency reasons ("keep compute and data in the Kingdom"). The release does not say which Dragonfly part
(AI200/AI250/AI300) is in service — given AI250/AI300 sampling timelines (mid-2027 / 2028), **AI200 is the
plausible inference, not a confirmed fact**. Framed as an initial pilot, "expected to be followed by additional
migrations from the wider industry." No throughput, chip count, or utilization figures disclosed. This is the
first concrete evidence the HUMAIN Riyadh deployment (announced Oct/Nov 2025, 200 MW figure previously
unconfirmed) carries live production workload.

**Qualcomm–AWS custom-silicon and optical-connectivity collaboration (2026-09-08).** Qualcomm announced a
multi-generation collaboration to supply Amazon/AWS with **custom AI inference silicon** and **optical
connectivity solutions** ("extending up to 1.6T and future-generation solutions," using Qualcomm's "advanced
SerDes and optical DSP technologies") for AWS AI data centers. Qualcomm is the **supplier** here (verbatim: "a
multi-generation collaboration with Amazon to enable customized silicon at scale for large-scale AI data
centers"). No product names (Dragonfly/AI200/AI250/AI300/UALink/ESUN), generation count, timeline, or volume
disclosed. Reciprocally, Qualcomm said it will deepen its own use of AWS infrastructure (Bedrock) for EDA
workloads. **This may be one of the "two unnamed global-scale hyperscaler custom-silicon customers" reported via
Futurum from the June 2026 Investor Day — no source confirms this explicitly; recorded as a plausible hypothesis
only.**

**SDK: quic/efficient-transformers v1.22.8.0 (2026-08-26).** Incremental model-support release (Qwen3.5/Qwen3.6/
Gemma4/GLM4, MoE export RAM reduction, CCL extensions) — see the version table above. Still Cloud AI 100-only; no
AI200/AI250 toolchain support published.

**Checked, no change found:** no new AI200/AI250/AI300 spec disclosure; no MLPerf submission found this window; no
re-verification of the Hot Chips 38 absence (prior finding stands, not independently re-checked this cycle); no
resolution of the AI200-vs-AI250 768 GB/card capacity conflict.

Sources: https://www.qualcomm.com/news/releases/2026/08/adobe-becomes-first-global-software-company-to-migrate-ai-worklo · https://www.qualcomm.com/news/releases/2026/09/qualcomm-announces-multi-generational-product-collaboration-with · https://github.com/quic/efficient-transformers/releases
