# Microsoft Maia — Software and Hardware Stack Summary

*as_of: 2026-08-08*
*Last substantive new information: 2026-06-02 (Microsoft Build 2026). No Maia silicon or software-stack change in the 2026-04-05 → 2026-08-08 window; see "Maia 200 Update" below.*

---

## Overview

Microsoft Maia is Azure's custom AI accelerator family, developed to reduce dependence on NVIDIA GPUs for large-scale AI inference and training inside Microsoft's datacenters. The architecture is designed **end-to-end** — from custom silicon to custom racks to custom networking — co-optimized for transformer-based workloads running on Azure OpenAI Services.

Two generations have shipped:

| | Maia 100 | Maia 200 |
|--|----------|----------|
| Announced | November 2023 | January 2026 |
| Presented | Hot Chips 2024 | Microsoft Blog deep-dive |
| Process | TSMC N5 (5nm) | TSMC N3 (3nm) |
| Transistors | ~105B | >140B |
| TDP | 500–700 W | 750 W |
| Primary use | Training + inference | Inference-optimized |
| Off-chip memory | 64 GB HBM2e / 1.8 TB/s | 216 GB HBM3e / 7 TB/s |
| On-chip SRAM | ~500 MB | 272 MB |
| Peak compute | 3 POPS MX6 / 0.8 POPS BF16 | >10 PFLOPS FP4, >5 PFLOPS FP8 |
| Scale-up BW | 4800 Gbps (12×400GbE) | 2.8 TB/s (on-die NIC) |

Maia powers Azure OpenAI Services inference (Copilot, Azure AI Studio) and, since Build 2026, Microsoft's own first-party **MAI** model family (MAI-Thinking-1 and siblings) — Microsoft's first public claim of model↔silicon co-design on Maia. Maia 200 is **in production in both US Central (Des Moines, Iowa) and US West 3 (Phoenix, Arizona)** as of June 2026, with **Italy, Australia and South Korea announced as next** (no dates, no named Azure regions).

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

Maia 200 introduces a **hierarchical tile → cluster architecture** optimized for LLM inference, where the bottleneck is memory bandwidth, not raw compute.

- **Tile:** Smallest compute unit — Tile Tensor Unit (TTU) + Tile Vector Processor (TVP) + Tile SRAM (TSRAM) + tile DMA
- **Cluster:** Multiple tiles sharing Cluster SRAM (CSRAM) + cluster DMA (stages HBM ↔ CSRAM traffic)
- 272 MB total on-die SRAM (TSRAM at 10–20× HBM bandwidth for hot data paths)
- 216 GB HBM3e at 7 TB/s
- Integrated on-die NIC at 2.8 TB/s bidirectional; 2-tier scale-up network supports 6,144 accelerators
- >10 PFLOPS FP4, >5 PFLOPS FP8; native FP8/FP4 tensor cores

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

### 4. Hot Chips 38 — scheduled disclosure, no content yet

The HC38 program lists, on **Tuesday 2026-08-25**, session **AI 1 (2:15–4:15 PM PDT, chairs Sherry Xu and John Wright)**: **"MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator"**, Prashant Ranjan & Jackson Peng (Microsoft). Hot Chips 38 runs 2026-08-23 (tutorials) to 2026-08-25; as of this update it is **15 days in the future**. **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** No slides, abstract, or specifications exist; **zero technical content in this survey derives from it.** It will be the first *system-level* public disclosure venue for Maia 200, as Hot Chips 2024 was for Maia 100.

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
| August 25, 2026 | *(scheduled, not yet presented)* Hot Chips 38 session AI 1: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft) — disclosure scheduled, content not yet public; first system-level public disclosure venue for Maia 200 |

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
