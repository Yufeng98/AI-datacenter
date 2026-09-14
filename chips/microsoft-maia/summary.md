# Microsoft Maia — Software and Hardware Stack Summary

*as_of: 2026-09-13*
*Last substantive new information: 2026-08-25 (Hot Chips 38 system-architecture disclosure — arXiv paper + Tech Community blog). See "Maia 200 Update — Hot Chips 38" below for the major update; see "Maia 200 Update — Build 2026" further down for the prior (2026-08-08) window, which had no silicon change.*

---

## Overview

Microsoft Maia is Azure's custom AI accelerator family, developed to reduce dependence on NVIDIA GPUs for large-scale AI inference and training inside Microsoft's datacenters. The architecture is designed **end-to-end** — from custom silicon to custom racks to custom networking — co-optimized for transformer-based workloads running on Azure OpenAI Services.

Two generations have shipped:

| | Maia 100 | Maia 200 |
|--|----------|----------|
| Announced | November 2023 | January 2026 |
| Presented | Hot Chips 2024 | Microsoft Blog deep-dive (Jan 2026); **Hot Chips 38 + arXiv:2608.24664 (2026-08-25)** |
| Process | TSMC N5 (5nm) | TSMC 3nm (sub-node unspecified — not confirmed N3P) |
| Transistors | ~105B | >140B (exactly confirmed, arXiv:2608.24664) |
| Die | ~820 mm² (reticle-limited) | **26×33 mm (~858 mm², arXiv:2608.24664); ServeTheHome reports "~820 mm² altogether" — close, not identical** |
| Clock | — | **2 GHz (arXiv:2608.24664)** |
| TDP | 500–700 W | 750 W (19 metal layers — arXiv:2608.24664) |
| Primary use | Training + inference | Inference-optimized |
| Off-chip memory | 64 GB HBM2e / 1.8 TB/s | 216 GB (secondary-sourced) HBM3e / 7 TB/s (bandwidth Microsoft-primary-confirmed 2026-09-13; 6× HBM3e stacks) |
| On-chip SRAM | ~500 MB | 272 MB (component detail added 2026-09-13: 3 MiB/tile TSRAM, 35 MB/cluster CSRAM) |
| Peak compute | 3 POPS MX6 / 0.8 POPS BF16 | **10,145 TFLOPS FP4 / 5,072 TFLOPS FP8** (arXiv:2608.24664 headline; supersedes ">10 / >5 PFLOPS" rounding) |
| Scale-up BW | 4800 Gbps (12×400GbE) | 2.8 TB/s (on-die NIC); **primary detail added 2026-09-13: 28× 400 Gbps ANC = 1.4 TB/s full-duplex, HammingMesh-variant topology** |

Maia powers Azure OpenAI Services inference (Copilot, Azure AI Studio) and, since Build 2026, Microsoft's own first-party **MAI** model family (MAI-Thinking-1 and siblings) — Microsoft's first public claim of model↔silicon co-design on Maia. Maia 200 is **in production in both US Central (Des Moines, Iowa) and US West 3 (Phoenix, Arizona)** as of June 2026, with **Italy, Australia and South Korea announced as next** (no dates, no named Azure regions). Microsoft's own arXiv:2608.24664 paper (2026-08-25) additionally states Maia 200 "is in production in the fleet today" system-wide. *A secondary-sourced claim that Maia 200 specifically serves "GPT-5.2-class workloads" at the Des Moines site was checked against both new primary sources this pass (the arXiv paper and the 2026-08-25 Tech Community blog) and found in neither — treat as unverified.*

---

## Hardware Stack

### Maia 100 Chip

The Maia 100 is a **~820 mm² SoC on TSMC 5nm**, reticle-limited, packaged on a CoWoS-S interposer with 4× HBM2e stacks (64 GB, 1.8 TB/s). It contains ~105 billion transistors.

**Compute subsystem:**
- **Tensor unit (16×R×16):** High-throughput matrix multiply for MX-format data (Microsoft co-developed the OCP MX Microscaling standard). Supports MX6 (3 POPS), MX9 (1.5 POPS), and BF16 (0.8 POPS).
- **Vector processor:** Loosely coupled superscalar engine with custom ISA. Handles FP32 and BF16 element-wise operations (activations, norms, reductions).
- **DMA engine:** Supports tensor sharding and asynchronous data movement with hardware semaphores.

**Memory:**
- ~500 MB on-chip SRAM (L1/L2 software-managed scratchpads; no hardware cache hierarchy)
- 64 GB HBM2e at 1.8 TB/s (Microsoft chose HBM2e deliberately — supply chain maturity and cost vs. HBM3)
- Host interface: PCIe Gen5 ×8 (32 GB/s)

**Scale-up networking:**
- 12× 400 GbE ports → 4800 Gbps all-gather/scatter-reduce, 1200 Gbps all-to-all
- Custom RoCE-like transport with AES-GCM encryption (confidential compute)
- Unified fabric for scale-up and scale-out (single network)

### Maia 200 Chip

Maia 200 introduces a **hierarchical tile → cluster architecture** optimized for LLM inference, where the bottleneck is memory bandwidth, not raw compute. Microsoft names this architecture class **Software Defined Locally Accessed Dataflow Architecture (SDLA)** — added 2026-09-13, arXiv:2608.24664.

- **Tile:** Smallest compute unit — Tile Tensor Unit (TTU) + Tile Vector Processor (TVP) + Tile SRAM (TSRAM, **3 MiB/tile**) + tile DMA
- **Cluster:** Multiple tiles sharing Cluster SRAM (CSRAM, **35 MB/cluster**) + cluster DMA (stages HBM ↔ CSRAM traffic); **4 clusters/chip, 9-or-10 tiles/cluster (36 active tiles benchmarked)** — added 2026-09-13
- 272 MB total on-die SRAM (TSRAM at 10–20× HBM bandwidth for hot data paths)
- 216 GB (secondary-sourced) HBM3e at 7 TB/s (bandwidth and **6-stack count** Microsoft-primary-confirmed 2026-09-13)
- Integrated on-die NIC at 2.8 TB/s bidirectional; 2-tier scale-up network supports 6,144 accelerators — **added 2026-09-13: 28× 400 Gbps ANC = 1.4 TB/s full-duplex per chip, HammingMesh-variant topology (20 fixed + 8 switched-across-4-planes links), 6,144-chip max via 48-SoC Tier-0 switches × up to 32 Tier-1 switches**
- **10,145 TFLOPS FP4, 5,072 TFLOPS FP8** (arXiv:2608.24664 headline, confirms the prior ">10 / >5 PFLOPS" rounding); native FP8/FP4 tensor cores; **2 GHz clock, 26×33 mm die, CoWoS-S 75×75mm package, 750 W across 19 metal layers** — all added 2026-09-13

### Ares Rack System (Maia 100)

Microsoft designed a custom rack ("Ares") specifically for Maia 100:
- 8 Maia servers, 32 Maia chips per rack
- ~40 kW per rack (mandatory liquid cooling; wider than standard 19" OCP racks)
- Dual-sourced top-of-rack switches (Arista + Cisco); 3 switch SKUs per rack
- Azure-integrated dynamic power optimization

---

## Software Stack

### Programming Model

Microsoft exposes two paths for kernel development:

1. **Triton (portability path):** OpenAI Triton Python DSL; same kernel can target NVIDIA GPUs and Maia. Provides agility. Maia Triton backend exists but is not open-sourced.

2. **NPL — Nested Parallel Language (performance path):** Microsoft's custom language for explicit SRAM placement, DMA scheduling, and parallel execution control. Enables near-peak utilization. Used internally for the production kernel library. Not publicly released.

### Framework Integration

- **PyTorch backend (first-class):** eager mode and graph mode (torch.compile-compatible)
- Standard PyTorch models can be deployed on Maia via the SDK without kernel authoring
- Primary deployment targets: Azure OpenAI Services (internal Microsoft models) and, from June 2026, Microsoft's own MAI model family on Maia 200

### Maia SDK (Preview)

The SDK bundles compilers, tools, and a pre-built kernel library. Available to Azure customers in preview. **Status re-checked 2026-08-08: still preview** — no GA, no announced GA timeline, no public version number or changelog, and no public documentation tree (`learn.microsoft.com/azure/maia` returns HTTP 404). Access remains sign-up-gated for selected academics, developers, frontier labs and open-source contributors. The component list below is unchanged since the January 2026 baseline:

| Component | Status |
|-----------|--------|
| PyTorch backend | confirmed (preview) |
| Triton compiler for Maia | confirmed (preview) |
| NPL compiler | confirmed (preview) |
| Maia simulator + cost calculator | confirmed (preview) |
| Debugger, profiler, visualizer | confirmed (preview) |
| Quantization + validation tools | confirmed (preview) |
| Pre-built kernel library | confirmed (preview, not open-source) |

### What Is Not Public

The entire software stack below the PyTorch/Triton API is closed:
- No public compiler source
- No public ISA specification (custom vector processor ISA; tensor unit ISA)
- No public kernel library
- No public runtime source
- No public driver (not upstreamed to Linux)
- No public firmware

---

## Key Architectural Distinctions

| Feature | Maia 100 | H100 GPU | Google TPU v5p |
|---------|----------|----------|----------------|
| Process | TSMC 5nm | TSMC 4nm | TSMC 7nm |
| On-chip SRAM | ~500 MB | ~36 MB L2 | ~16 MB VMEM |
| Memory | 64 GB HBM2e, 1.8 TB/s | 80 GB HBM3, 3.35 TB/s | 95 GB HBM2e, ~2.8 TB/s |
| Scale-up | 4800 Gbps Ethernet | 900 GB/s NVLink | 4800 Gbps ICI |
| ISA | Custom (not public) | PTX + SASS | Not public (VLIW) |
| Software openness | Closed (SDK preview) | Mostly open (CUDA) | Closed (libtpu) |
| Precision | MX6/MX9/BF16 | FP8/BF16/FP4 | BF16/FP8 (v5p) |
| OCP MX native | Yes (co-inventor) | No (Hopper) | No |
| Encryption in transport | AES-GCM native | No | No |
| Rack design | Custom (Ares, 40kW liquid) | Standard OAM/DGX | Google proprietary |

---

## Maia 200 Update — Build 2026 Deployment & Model Co-Design (2026-08-08)

*Update window 2026-04-05 → 2026-08-08. Primary sources: Microsoft Build 2026 MAI keynote transcript (microsoft.ai, 2026-06-02); Maia 200 announcement blog (2026-01-26); Hot Chips 38 program (hotchips.org, retrieved 2026-08-08).*

**Change class: minor.** **No new Maia silicon and no change to any Maia 100 or Maia 200 specification.** Every Maia 200 hardware figure recorded above and in `hw-architecture.md` — TSMC N3, >140B transistors, 750 W, 216 GB HBM3e @ 7 TB/s, 272 MB on-die SRAM, >10 PFLOPS FP4 / >5 PFLOPS FP8, 2.8 TB/s on-die NIC, 2-tier scale-up to 6,144 accelerators — remains the January 2026 figure. What changed is **deployment status**, one **new vendor efficiency claim**, and one **new first-party workload class**.

### 1. Deployment: US West 3 moved from "planned" to "in production"

The January 2026 baseline sentence, verbatim from the Maia 200 announcement blog — "Maia 200 is deployed in our US Central datacenter region near Des Moines, Iowa, with the US West 3 datacenter region near Phoenix, Arizona, coming next and future regions to follow" — is **superseded for Arizona**. At Build 2026 (2026-06-02) Microsoft stated Maia 200 is running in production in **both** US Central (Iowa) and US West 3 (Arizona), with **Italy, Australia and South Korea** announced as next.

Corroborated by DataCenterDynamics (2026-06-03), VentureBeat and Analytics India Magazine (2026-06-03), all reporting the same formulation from the keynote. Caveat retained: the international list is **announced intent with no dates and no named Azure regions**, and traces to the Build keynote as relayed by press — no first-party Microsoft blog restates it. (The 2026-06-02 Azure Cobalt 200 VM blog covers Cobalt regions only and does not mention Maia.)

### 2. New vendor efficiency claim — 1.4× perf/W on first-party MAI models

Mustafa Suleyman, Build 2026 MAI keynote (microsoft.ai transcript, 2026-06-02), verbatim: *"we're also co-designing our models with our own silicon, optimizing MAI-Thinking-1 on our Maia 200 chip and benchmarking it head-to-head against the GB200"*, and *"On top of the 30% improvement that Satya just mentioned, we're seeing a further 1.4x performance-per-watt gain when running our MAI models on the Maia 200 end to end."*

Three precisions the survey must carry:

- The 1.4× is stated as **incremental on top of the pre-existing 30% figure**, and that 30% is the **January 2026** claim of *"30% better performance per dollar than the latest generation hardware in our fleet today"* — a perf/**dollar** claim against **Microsoft's own fleet**, not a perf/watt claim and not explicitly against GB200. The 30% predates this update window and is not new.
- The transcript **does not state that the 1.4× perf/W is measured against GB200.** Press (latent.space AINews 2026-06-03, and others) uniformly renders it as "1.4× perf/W vs GB200", but that is inference from the adjacent head-to-head sentence. Attribute as: *vendor claim, baseline not specified in the primary transcript; widely reported as vs GB200*.
- **Unfalsifiable vendor marketing.** No workload, sequence length, batch size, precision, cluster size, power-measurement boundary, or baseline configuration was published. This figure must **not** enter any comparison table as a measured number.

### 3. New workload class — Maia 200 now serves Microsoft's own MAI models

Previously the survey recorded Maia's workload as Azure OpenAI Services inference. Build 2026 added Microsoft's own **MAI family** — seven models across five families, including **MAI-Thinking-1**, Microsoft's first reasoning model (reported as a ~35B-active-parameter MoE with a 256K context window), plus MAI-Image-2.5, MAI-Transcribe-1.5, MAI-Voice-2 and MAI-Code-1-Flash — as an explicitly silicon-co-designed Maia 200 workload. This is the survey-relevant part of the window: it is Microsoft's **first public claim of model↔accelerator co-design on Maia**.

### 4. Hot Chips 38 — realized 2026-08-25; see the dedicated update section below

The HC38 program listed, on **Tuesday 2026-08-25**, session **AI 1 (2:15–4:15 PM PDT, chairs Sherry Xu and John Wright)**: **"MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator"**, Prashant Ranjan & Jackson Peng (Microsoft). **This talk has now happened**, accompanied by a companion Tech Community blog and a full arXiv paper (2608.24664) — see "Maia 200 Update — Hot Chips 38 System Architecture Disclosure (2026-09-13)" below for the complete technical content. It was, as anticipated in the 2026-08-08 pass, the first *system-level* public disclosure venue for Maia 200, as Hot Chips 2024 was for Maia 100.

### 5. Explicitly not changed / not confirmed

| Item | Status as of 2026-08-08 |
|---|---|
| Maia SDK | Still **preview**. No GA, no GA timeline, no version number, no changelog, no public docs tree (`learn.microsoft.com/azure/maia` → HTTP 404). Component table unchanged. Everything below the PyTorch/Triton API remains closed. |
| MLPerf | **No Maia result.** MLPerf Training v6.0 (2026-06-16) lists Azure among 24 submitters, but Microsoft's own Azure HPC blog states that submission was **8,192 NVIDIA GB200 GPUs** on Llama 3.1 405B. Maia appears nowhere in v6.0. |
| Maia 280 / Braga / Braga-R / Clea | **Nothing new in the window.** The dual-chiplet Maia 280 (two Braga dies, 2027) and Braga-R/Clea slipping to 2028+ come solely from 2025 supply-chain reporting (TrendForce 2025-07-04 and downstream aggregators) that predates the repo baseline. Rumor/analyst class, low confidence — not promoted. |
| New Microsoft Maia technical blog | **None** since the January 2026 deep-dive. The Azure Infrastructure blog's April–August 2026 silicon posts are Cobalt-only (Rowhammer protection in Cobalt 200, 2026-06-25; Cobalt workload-aware power management, 2026-07-16). |
| MRC (Multipath Reliable Connection) | **Adjacent ecosystem context — do NOT attribute to Maia.** An open Ethernet transport extending RoCEv2 with per-packet multipath, sender-based congestion control and fast loss/path-failure recovery, released via OCP ~2026-05-05 by OpenAI with AMD, Broadcom, Intel, Microsoft and NVIDIA; paper arXiv:2606.18170 (submitted 2026-06-16) has Microsoft co-authors (Adrian Caulfield, Michael Papamichael among ~40). **The paper does not mention Maia**, and secondary claims that Maia implements MRC are unsourced. Recorded only as evidence of Microsoft's networking direction. |
| Fairwater (Mount Pleasant, Wisconsin) | **Not a Maia deployment.** The campus, reported fully operational ~2026-06-23 as a single coherent cluster over an 800 Gbps Ethernet fabric, is an **NVIDIA GB200** deployment. Its capacity, fabric and coherent-cluster claims must not be attributed to Maia. |

---

## Maia 200 Update — Hot Chips 38 System Architecture Disclosure (2026-09-13)

*Primary sources: arXiv:2608.24664, "Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration" (Sherry Xu et al., Microsoft, submitted 2026-08-25 — full text extracted via `pdftotext -layout`); Microsoft Tech Community, "Maia 200: Software-defined dataflow and all-Ethernet networking for efficient inference on Azure" (Xu/Ranjan/Hoefler, 2026-08-25 — body extracted from the page's embedded Next.js JSON, since WebFetch's rendered-page extraction returned only the title). Secondary: ServeTheHome Hot Chips 38 coverage, 2026-08-25.*

**Change class: major.** The Hot Chips 38 talk flagged "scheduled, content not yet public" on 2026-08-08 has now happened, and — unusually for this vendor — is backed by a full academic paper, not just slides. Die size, tile/cluster counts, clock speed, and network topology, all "not disclosed" as of 2026-08-08, are now Microsoft-primary-confirmed. Full detail: `research/microsoft-maia/investigations/hw-architecture.md` (research notes) and `chips/microsoft-maia/hw-architecture.md` §9 (chip-level doc).

**Architecture name:** Microsoft names the design class **Software Defined Locally Accessed Dataflow Architecture (SDLA)**.

**Process/package/die (new):** TSMC 3nm (sub-node unspecified, not confirmed N3P); >140B transistors (exact match to the January 2026 figure); **26×33 mm monolithic die** (~858 mm²; ServeTheHome's independent "~820 mm² altogether" is close but not identical); CoWoS-S packaging, 75×75 mm package; 750 W across 19 metal layers; **2 GHz clock**. Direct quote: "it is a full system including tray, rack, and network architecture scalable to thousands of accelerators in a single cluster and it is **in production in the fleet today**."

**Compute hierarchy (new):** 4 clusters × 9-or-10 tiles/cluster (36 active tiles in the paper's own benchmark); each tile = TTU + TVP + 3 MiB TSRAM; each cluster = 35 MB CSRAM. Headline chip peak **10,145 TFLOPS FP4 / 5,072 TFLOPS FP8** (confirms the prior ">10 / >5 PFLOPS" rounding almost exactly); a separate benchmark-section figure states 1,180 TFLOPS BF16 / 4,785 TFLOPS FP8 for the specific 36-tile configuration — both paper-stated, not reconciled.

**Network — HammingMesh topology (new):** 28× 400 Gbps integrated Ethernet NICs (ANCs) per chip = **1.4 TB/s full-duplex**; topology is a "special case of a 2×2 1D Hamming Mesh"; of the 28 ANCs, 20 are fixed intra-tray links and 8 are switched across 4 planes (2 ANCs/plane/chip) — this is the primary confirmation of the previously analyst-only "8 Ethernet lanes / 4 network planes" figure. Two-tier fabric: each T0 switch (51.2T, 128×400G) serves 48 chips/12 trays; up to 32 T1 switches at 1:3 oversubscription; **48×128 = 6,144 max chips**, confirming the existing cluster-size figure with explained arithmetic. Transport is Microsoft's in-house **ATLv2**, which the paper says "later influenced the standardization of Ultra Ethernet." *Open item:* this 1.4 TB/s full-duplex figure and the pre-existing "2.8 TB/s bidirectional" figure describe the same fabric (2.8 ≈ 2×1.4) but neither source states the reconciliation explicitly.

**Measured performance (new):** BF16 matmul up to 99.69% peak (compute-bound) / 51.4% peak bandwidth (memory-bound); FP8 matmul up to 96% / 56%; 8-chip Allgather achieves 78% of the latency bound and 94% of the 1.4 TiB/s bandwidth bound; single-chip Qwen 2.5 7B decode demo reaches 2,434 tokens/s at 16,384-token context (>70% of estimated max). **ServeTheHome's separately reported "1.65 PFLOPS attention effective peak," "~1.3 TB/s BF16 AllReduce," and "655 GB/s All2All" were not found in the arXiv paper's extracted text** — recorded as analyst/HC38-slide-sourced only, not primary-confirmed.

**Two new vendor claims, both unfalsifiable, both excluded from any comparison table:** (1) paper: "Maia 200 saves 30% cost (TCO) and 15% energy... compared to any other AI accelerator in Microsoft's fleet" (internal data, no methodology); (2) blog: **">40% higher token generation" at equal rack power** running MAI-Thinking-1 vs. "other leading accelerators in the Azure fleet" (no chip named) — a new, distinct figure from the Build-2026 "1.4× perf/W" claim already in this file.

**Des Moines / GPT-5.2 — checked, not found.** Both new primary sources were searched for "Des Moines" and "GPT-5"; neither term appears in either document. The Des Moines/Iowa deployment itself remains well-sourced from other material; the specific GPT-5.2-workload pairing is unconfirmed.

---

## Timeline

| Date | Event |
|------|-------|
| November 2023 | Maia 100 announced at Microsoft Ignite alongside Cobalt 100 (Arm CPU) |
| August 2024 | Maia 100 detailed at Hot Chips 2024 (HC2024) — first full public architecture disclosure |
| Late 2024 | Maia 100 deployed for Azure OpenAI Services (Copilot, Azure AI Studio) |
| January 26, 2026 | Maia 200 announced — inference-optimized, TSMC 3nm, tile/cluster architecture; claim of "30% better performance per dollar than the latest generation hardware in our fleet today" (perf/$, vs Microsoft's own fleet) |
| January 27, 2026 | Maia 200 deployed in US Central (Des Moines, Iowa); US West 3 (Phoenix, Arizona) expansion announced as next |
| April 2026 | Maia SDK in preview; Maia 200 serving Azure OpenAI production workloads |
| June 2, 2026 | Build 2026: Maia 200 in production in **both** US Central (Iowa) and US West 3 (Arizona); Italy, Australia, South Korea announced as next (no dates). Microsoft claims a further **1.4× perf/W** running its own MAI models end-to-end on Maia 200, on top of the January 2026 30% perf/$ claim — vendor claim, no methodology published. MAI-Thinking-1 cited as co-designed with Maia 200 and benchmarked head-to-head against GB200. |
| June 16, 2026 | MLPerf Training v6.0 published — **no Maia submission**; Azure's entry was 8,192 NVIDIA GB200 GPUs |
| August 25, 2026 | Hot Chips 38 session AI 1: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft) — **realized**; accompanied by arXiv:2608.24664 and a Tech Community blog disclosing die size (26×33mm), clock (2 GHz), tile/cluster counts (4×9-10), full HammingMesh network topology, and new vendor claims (30% TCO + 15% energy; >40% token-gen/W for MAI-Thinking-1). First system-level public disclosure venue for Maia 200. |

---

## Sources

- [HC2024 — Inside Maia 100 (PDF)](https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf)
- [Microsoft Tech Community — Inside Maia 100](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118)
- [Azure Blog — Maia from silicon to service](https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/)
- [Microsoft Blog — Maia 200 announcement](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/)
- [Microsoft Tech Community — Maia 200 deep dive](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312)
- [Tom's Hardware — Maia 200](https://www.tomshardware.com/pc-components/cpus/microsoft-introduces-newest-in-house-ai-chip-maia-200-is-faster-than-other-bespoke-nvidia-competitors-built-on-tsmc-3nm-with-216gb-of-hbm3e)
- [ServeTheHome — Maia 100 system](https://www.servethehome.com/microsoft-maia-100-ai-accelerator-for-azure/)
- [TechRadar — HBM2e choice](https://www.techradar.com/pro/microsoft-deliberately-chose-to-use-old-tech-for-its-nvidia-gpu-rival-maia-100-ai-accelerator-uses-hbm2e-memory-and-the-mysterious-ability-to-unlock-new-capabilities-via-firmware-update)
- [TrendForce — Maia 200 HBM3e](https://www.trendforce.com/news/2026/01/27/news-microsoft-unveils-maia-200-ai-chip-on-tsmc-3nm-sk-hynix-reportedly-sole-hbm3e-supplier/)

### Added 2026-08-08 (update window 2026-04-05 → 2026-08-08)

- [microsoft.ai — Microsoft Build 2026 MAI keynote transcript](https://microsoft.ai/news/microsoft-build-2026-mai-keynote-transcript/) — 2026-06-02; primary source for the 1.4× perf/W claim and the MAI-Thinking-1 / Maia 200 co-design statement
- [microsoft.ai — Launching seven new MAI models](https://microsoft.ai/news/building-a-hillclimbing-machine-launching-seven-new-mai-models/) — MAI family launched at Build 2026
- [Hot Chips 38 program](https://hotchips.org/program/conference/) — session AI 1, Tue 2026-08-25, "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator" (**scheduled; no content public as of 2026-08-08**)
- [latent.space AINews — Microsoft Build / MAI-Thinking-1](https://www.latent.space/p/ainews-microsoft-build-mai-thinking) — 2026-06-03; independent coverage, renders the claim as "1.4× perf/W versus GB200" (press inference)
- [Analytics India Magazine — Everything Microsoft announced at Build 2026](https://analyticsindiamag.com/ai-trends/everything-microsoft-announced-at-build-2026) — 2026-06-03; Iowa+Arizona production, Italy/Australia/South Korea next
- [DataCenterDynamics — Cobalt 200 VMs in preview, Maia 200 already in production](https://www.datacenterdynamics.com/en/news/microsoft-launches-vms-based-on-cobalt-200-chip-in-preview-maia-200-already-in-production/) — 2026-06-03 (direct fetch returns HTTP 403; verified via search snippet only)
- [MLCommons — MLPerf Training v6.0 results](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 2026-06-16; negative control, no Maia entry
- [Azure HPC blog — Llama 3.1 405B MLPerf Training on Azure at 8K GPU scale](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inside-llama-3-1-405b-mlperf-training-on-azure-system-level-insights-at-8k-gpu-s/4529296) — Azure's v6.0 submission was 8,192 GB200 GPUs, not Maia
- [arXiv:2606.18170 — The Multipath Reliable Connection (MRC) Transport](https://arxiv.org/abs/2606.18170) — 2026-06-16; Microsoft co-authors; **does not mention Maia** (adjacent ecosystem context only)

### Added 2026-09-13 (Hot Chips 38 system architecture disclosure)

- [arXiv:2608.24664 — Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration](https://arxiv.org/abs/2608.24664) — Sherry Xu et al., Microsoft, submitted 2026-08-25 (primary; full paper)
- [Microsoft Tech Community — Maia 200: Software-defined dataflow and all-Ethernet networking for efficient inference on Azure](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/maia-200-software-defined-dataflow-and-all-ethernet-networking-for-efficient-inf/4548198) — Sherry Xu, Prashant Ranjan, Torsten Hoefler, 2026-08-25 (primary)
- [ServeTheHome — Microsoft's Maia 200 Accelerator at Hot Chips 2026](https://www.servethehome.com/microsofts-maia-200-accelerator-at-hot-chips-2026/) — 2026-08-25 (analyst; sole source for the attention-effective-peak, AllReduce, and All2All figures flagged as unconfirmed above)
- [Hot Chips 38 program](https://hotchips.org/program/conference/) — talk realized 2026-08-25
