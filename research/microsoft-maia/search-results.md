# Microsoft Maia — Search Results

**Date:** 2026-04-05
**Status Note:** Microsoft Maia 100 was announced November 2023 and presented at Hot Chips 2024 (HC2024). Maia 200, the second-generation inference-focused accelerator, was announced and deployed in January 2026. Both generations are Azure-internal; the Maia SDK is in preview with limited public detail on the kernel/driver internals.

---

## 1. Official Documentation / Whitepapers

| Resource | URL | Notes |
|---|---|---|
| Hot Chips 2024 — "Inside Maia 100" (PDF) | https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf | Primary public technical reference; presented August 2024 |
| Azure Blog — "Azure Maia for the era of AI: From silicon to software to systems" | https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/ | Official systems overview including SDK, rack, and networking |
| Microsoft Tech Community — "Inside Maia 100: Revolutionizing AI Workloads" | https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118 | Detailed Maia 100 architecture blog post |
| Microsoft Blog — "Maia 200: The AI accelerator built for inference" | https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/ | Official Maia 200 announcement |
| Microsoft Tech Community — "Deep dive into the Maia 200 architecture" | https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312 | Tile/cluster hierarchy, SRAM, NoC, networking deep dive |
| Microsoft Source — "Microsoft introduces Maia 200" | https://news.microsoft.com/source/emea/2026/01/microsoft-introduces-maia-200-new-inference-accelerator-enhances-ai-performance-in-azure/ | Press announcement with performance details |

---

## 2. Open-Source Repositories

*No public compiler, runtime, driver, ISA toolchain, or SDK repositories found for Maia.*

- The Maia SDK is available to Azure customers in **preview** only; no public GitHub repositories exist.
- Triton (openai/triton) is referenced as the portability programming model, but any Maia-specific Triton backend is not open-sourced.
- No public PyTorch backend, no public MLIR dialect, no public kernel library.

---

## 3. Chip Architecture — Maia 100

**Key specifications (from HC2024 and public disclosures):**

- **Process:** TSMC N5 (5nm)
- **Package:** CoWoS-S interposer
- **Die size:** ~820 mm² (reticle-limited)
- **Transistors:** ~105 billion
- **TDP:** 500 W nominal, up to 700 W supported
- **Compute:** Peak Dense Tensor POPS — 6-bit: 3 POPS, 9-bit: 1.5 POPS, BF16: 0.8 POPS
- **Tensor unit:** 16×R×16 high-speed tensor unit; supports MX (Microscaling) data format (OCP standard co-developed by Microsoft)
- **Vector processor:** Loosely coupled superscalar engine; custom ISA; supports FP32, BF16
- **On-chip SRAM (L1/L2):** ~500 MB (software-managed scratchpads)
- **Off-chip memory:** 4× HBM2e dies → 64 GB capacity, 1.8 TB/s bandwidth

| Resource | URL |
|---|---|
| HC2024 PDF | https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf |
| TechSpot — Maia 100 deep dive | https://www.techspot.com/news/104514-microsoft-maia-100-looks-bring-customers-cost-effective.html |
| ServeTheHome — Maia 100 accelerator overview | https://www.servethehome.com/microsoft-maia-100-ai-accelerator-for-azure/ |
| Neowin — Maia 100 details | https://www.neowin.net/news/microsoft-shares-more-details-on-maia-100-its-first-custom-ai-chip/ |
| Glenn Klockwood — Maia 100 analysis | https://www.glennklockwood.com/garden/processors/Maia-100 |

---

## 4. Chip Architecture — Maia 200

**Key specifications (from January 2026 announcements):**

- **Process:** TSMC N3 (3nm)
- **Transistors:** >140 billion
- **TDP:** 750 W
- **Compute:** >10 PFLOPS FP4, >5 PFLOPS FP8
- **Tensor cores:** Native FP8/FP4 tensor cores (Tile Tensor Unit, TTU)
- **Vector processor:** Tile Vector Processor (TVP) — programmable SIMD engine
- **On-chip SRAM:** 272 MB (hierarchical: Tile SRAM + Cluster SRAM)
- **Off-chip memory:** 216 GB HBM3e, 7 TB/s bandwidth
- **Scale-up NIC:** Integrated on-die NIC, 2.8 TB/s bidirectional; 2-tier scale-up network supporting 6,144 accelerators
- **Architecture:** Tile → Cluster hierarchy; each tile has TTU + TVP + TSRAM + tile DMA; clusters share CSRAM + cluster DMA

| Resource | URL |
|---|---|
| Microsoft Blog — Maia 200 | https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/ |
| Tech Community — Maia 200 deep dive | https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312 |
| Tom's Hardware — Maia 200 specs | https://www.tomshardware.com/pc-components/cpus/microsoft-introduces-newest-in-house-ai-chip-maia-200-is-faster-than-other-bespoke-nvidia-competitors-built-on-tsmc-3nm-with-216gb-of-hbm3e |
| TrendForce — SK Hynix HBM3e supplier | https://www.trendforce.com/news/2026/01/27/news-microsoft-unveils-maia-200-ai-chip-on-tsmc-3nm-sk-hynix-reportedly-sole-hbm3e-supplier/ |

---

## 5. Memory Hierarchy

### Maia 100
- **On-chip SRAM:** ~500 MB L1/L2 (software-managed scratchpads; no hardware cache)
- **HBM2e:** 64 GB / 1.8 TB/s (4 stacks, CoWoS-S package)
- Design deliberately chose HBM2e (older generation) for cost optimization and supply chain maturity

### Maia 200
- **Tile SRAM (TSRAM):** per-tile; hot data path; ~10–20× bandwidth of HBM
- **Cluster SRAM (CSRAM):** per-cluster shared; second-tier locality
- **Total on-die SRAM:** 272 MB
- **HBM3e:** 216 GB / 7 TB/s (SK Hynix sole supplier reported)

| Resource | URL |
|---|---|
| TechRadar — HBM2e design choice | https://www.techradar.com/pro/microsoft-deliberately-chose-to-use-old-tech-for-its-nvidia-gpu-rival-maia-100-ai-accelerator-uses-hbm2e-memory-and-the-mysterious-ability-to-unlock-new-capabilities-via-firmware-update |
| TrendForce — HBM3e supply | https://www.trendforce.com/news/2026/01/27/news-microsoft-unveils-maia-200-ai-chip-on-tsmc-3nm-sk-hynix-reportedly-sole-hbm3e-supplier/ |

---

## 6. Interconnect / Networking

### Maia 100 Scale-up
- Ethernet-based with custom RoCE-like transport protocol
- Up to **4800 Gbps** all-gather / scatter-reduce bandwidth
- Up to **1200 Gbps** all-to-all bandwidth
- Backend network BW: **600 GB/s (12× 400 GbE ports)**
- Supports AES-GCM encryption (confidential compute capable)
- Unified backend: same network used for scale-up and scale-out

### Maia 100 Host Interface
- **PCIe Gen5 ×8:** 32 GB/s host bandwidth

### Maia 100 System / Rack ("Ares")
- Rack: 8 Maia servers, 32 Maia chips per rack
- ~40 kW per rack (liquid-cooled mandatory; wider than standard 19" OCP racks)
- Top-of-rack: Arista + Cisco dual-sourced switches; 3 switch SKUs per rack
- Custom rack-level power distribution + Azure dynamic power optimization

### Maia 200 Scale-up
- Integrated on-die NIC: **2.8 TB/s bidirectional**
- 2-tier scale-up network topology; supports clusters of 6,144 accelerators
- Advanced transport protocol (enhanced reliability and custom congestion control)

| Resource | URL |
|---|---|
| Azure Blog — silicon-to-service overview | https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/ |
| ServeTheHome — networking and rack | https://www.servethehome.com/microsoft-maia-100-ai-accelerator-for-azure/ |
| Microsoft Tech Community — Maia 100 blog | https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118 |

---

## 7. Software Stack / Compiler / Programming Model

### Framework Integration
- **First-class PyTorch backend** — supports both eager mode and graph mode
- Target deployment: Azure OpenAI Services (Copilot, Azure AI Studio)

### Maia SDK (Preview)
- Available to Azure customers; not publicly released
- Components:
  - **Triton programming model** — agility and portability; runs on GPUs and Maia
  - **Maia API (NPL — Nested Parallel Language)** — custom programming model for maximum performance; explicit control of data movement, SRAM placement, and parallel execution
  - **Compilers:** Triton compiler + Maia API (NPL) compiler
  - **Developer tools:** debugger, profiler, visualizer, model quantization + validation tools
  - **Maia simulator:** pre-silicon cost calculator and model validation
  - **Kernel library:** Microsoft-developed ML compute and communication kernels (built using the compilers); custom kernel authoring also supported

### Compute Precision / Data Types
- MX format (Microscaling; OCP standard) — low-precision MX4/MX6/MX9 via tensor unit
- BF16, FP32 — via vector processor
- FP8, FP4 — native tensor cores on Maia 200

| Resource | URL |
|---|---|
| Azure Blog — SDK overview | https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/ |
| Microsoft Tech Community — Maia 100 blog | https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118 |
| Microsoft Maia 200 announcement | https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/ |

---

## 8. Project Status & Timeline

| Date | Event |
|---|---|
| November 2023 | Maia 100 announced alongside Cobalt 100 (Arm CPU) at Microsoft Ignite |
| August 2024 | Maia 100 presented at Hot Chips 2024 (HC2024) with full architecture details |
| Late 2024 | Maia 100 deployed for Azure OpenAI Services (Copilot and internal workloads) |
| January 26, 2026 | Maia 200 announced publicly |
| January 27, 2026 | Maia 200 deployed in US Central data center (Des Moines, Iowa); US West 3 (Phoenix) expansion announced |
| April 2026 | Maia SDK in preview; Maia 200 serving Azure OpenAI production inference |

| Resource | URL |
|---|---|
| InfoQ — November 2023 announcement | https://www.infoq.com/news/2023/11/azure-custom-chips-cobalt-maia/ |
| Microsoft Source — silicon-to-service | https://news.microsoft.com/source/features/ai/in-house-chips-silicon-to-service-to-meet-ai-demand/ |
| The Register — Maia 200 launch | https://www.theregister.com/2026/01/26/microsoft_maia_200/ |

---

## 9. 15-Layer Stack Assessment

| Layer | Availability | Notes |
|---|---|---|
| Framework Integration | confirmed | PyTorch (eager + graph mode); Azure OpenAI Services |
| Compiler / IR | confirmed | Triton compiler (open-source, Maia backend); NPL/Maia API compiler (not public) |
| Op Library | not public | Microsoft-built kernel library (not released); inferred GEMM/Attention/collective kernels |
| Kernel Library | not public | Custom kernels via NPL; no public kernel repo |
| Runtime | not public | Maia runtime (SDK preview only; no public source) |
| Driver / Firmware | not public | Maia driver (Azure-internal; HBM firmware upgradeable per TechRadar report) |
| Communication | confirmed | Custom RoCE-like protocol (Maia 100); integrated on-die NIC (Maia 200); AES-GCM encryption |
| Assembler / ISA | not public | Custom ISA for vector processor; tensor unit ISA not documented publicly |
| Compute Engine | confirmed | Maia 100: 16×R×16 tensor unit; Maia 200: TTU + TVP tile hierarchy |
| Data Path | confirmed | DMA subsystem; hierarchical NoC; tile→cluster data movement |
| On-chip Memory | confirmed | Maia 100: ~500 MB SRAM; Maia 200: 272 MB TSRAM+CSRAM |
| Off-chip Memory | confirmed | Maia 100: 64 GB HBM2e 1.8 TB/s; Maia 200: 216 GB HBM3e 7 TB/s |
| Host Interface / Package | confirmed | PCIe Gen5 ×8 (32 GB/s); CoWoS-S package; custom Ares rack |
| Scale-up Interconnect | confirmed | Maia 100: 4800 Gbps via 12×400GbE custom RoCE-like; Maia 200: 2.8 TB/s on-die NIC |
| Scale-out Interconnect | confirmed | Custom RoCE-like over Ethernet; Arista/Cisco ToR switches; unified scale-up/scale-out fabric |

---

## 10. Other Resources

| Type | Title | URL |
|---|---|---|
| Analysis | Glenn Klockwood — Maia 100 processor notes | https://www.glennklockwood.com/garden/processors/Maia-100 |
| Analysis | SemiAnalysis — Microsoft Custom Silicon | https://newsletter.semianalysis.com/p/microsoft-infrastructure-ai-and-cpu |
| Press | Data Center Dynamics — Maia/Cobalt announcement | https://www.datacenterdynamics.com/en/news/microsoft-announces-in-house-arm-cpu-and-ai-accelerator-chips-custom-racks/ |
| Press | Tom's Hardware — Maia 200 specs | https://www.tomshardware.com/pc-components/cpus/microsoft-introduces-newest-in-house-ai-chip-maia-200-is-faster-than-other-bespoke-nvidia-competitors-built-on-tsmc-3nm-with-216gb-of-hbm3e |
| Press | Data Center Knowledge — Maia 200 deployment | https://www.datacenterknowledge.com/infrastructure/microsoft-unveils-maia-200-in-house-inference-chip |
| Press | The Register — Maia 200 cost angle | https://www.theregister.com/2026/01/26/microsoft_maia_200/ |
| Press | AI Magazine — Maia 200 bottleneck analysis | https://aimagazine.com/news/microsoft-how-maia-200-accelerator-addresses-ai-bottlenecks |
| Press | AI Unfiltered — Maia 200 vs competitors | https://www.arturmarkus.com/microsoft-maia-200-cuts-ai-token-costs-30-with-140-billion-transistors-3nm-chip-deployed-january-27-in-us-data-centers-outperforms-amazon-trainium-and-google-tpu-on-inference/ |

---

## Summary Assessment

Microsoft Maia is a **well-disclosed hardware architecture with a largely closed software stack**:

- **Well documented:** Chip specifications (HC2024 for Maia 100; Microsoft Blog for Maia 200), memory hierarchy, interconnect bandwidth, system rack design, SDK programming model overview.
- **Partially documented:** Software stack concepts (PyTorch backend, Triton model, NPL language, SDK tools) — confirmed to exist but implementation not public.
- **Not public:** Compiler source, ISA specification, kernel library, runtime source, driver source. SDK is preview-only for Azure customers.
- **Unique aspects:** OCP MX data format co-inventor; custom RoCE-like transport with encryption; Ares rack system optimized for liquid cooling and high power density; firmware-upgradeable capabilities on Maia 100.

---

## Addendum — search wave 2026-08-08 (window 2026-04-05 → 2026-08-08)

**Outcome:** no new primary technical source for Maia. No new silicon, no SDK GA, no open-source release, no MLPerf entry. The window's material is deployment status, one vendor efficiency claim, one new first-party workload class, and one **scheduled but not yet delivered** Hot Chips talk.

### New primary sources

| Resource | URL | Notes |
|---|---|---|
| microsoft.ai — Microsoft Build 2026 MAI keynote transcript | https://microsoft.ai/news/microsoft-build-2026-mai-keynote-transcript/ | **2026-06-02.** Primary source for the "further 1.4x performance-per-watt" claim and the MAI-Thinking-1 / Maia 200 co-design statement. Page shows 2026-06-02 with **no visible update stamp** (an "updated 2026-06-08" date could not be confirmed and is not recorded). |
| microsoft.ai — "Building a hillclimbing machine": seven new MAI models | https://microsoft.ai/news/building-a-hillclimbing-machine-launching-seven-new-mai-models/ | MAI family launched at Build 2026: MAI-Thinking-1, MAI-Image-2.5, MAI-Transcribe-1.5, MAI-Voice-2, MAI-Code-1-Flash and siblings |
| Hot Chips 38 program | https://hotchips.org/program/conference/ | Session AI 1, Tue **2026-08-25**, 2:15–4:15 PM PDT, chairs Sherry Xu & John Wright: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft). **Disclosure scheduled — content not yet public as of 2026-08-08.** Highest-value future source; re-scan after the event. |
| Hot Chips 38 site | https://hotchips.org/ | HC38 runs 2026-08-23 (tutorials) → 2026-08-25; slides released to registrants after the event |
| Microsoft Source EMEA — Maia 200 feature | https://news.microsoft.com/source/emea/features/maia-200-microsoft-ai-accelerator-azure/ | Regional feature piece on Maia 200 |

### New secondary / corroborating sources

| Resource | URL | Notes |
|---|---|---|
| latent.space AINews — Microsoft Build / MAI-Thinking-1 | https://www.latent.space/p/ainews-microsoft-build-mai-thinking | 2026-06-03. Renders the claim as "30% better performance per dollar and 1.4x performance-per-watt gain versus GB200" — the "versus GB200" attribution is **press inference**, not in the transcript. Also reports MAI-Thinking-1 as a 35B-active MoE with 256K context. |
| Analytics India Magazine — Everything Microsoft announced at Build 2026 | https://analyticsindiamag.com/ai-trends/everything-microsoft-announced-at-build-2026 | 2026-06-03. Corroborates Iowa + Arizona in production, Italy/Australia/South Korea next; also describes MRC co-development. |
| DataCenterDynamics — Cobalt 200 VMs in preview, Maia 200 already in production | https://www.datacenterdynamics.com/en/news/microsoft-launches-vms-based-on-cobalt-200-chip-in-preview-maia-200-already-in-production/ | 2026-06-03. **Direct fetch returns HTTP 403** and web.archive.org unreachable from this environment; verified via search-engine snippet only. |
| VentureBeat — Microsoft AI chief on superintelligence | https://venturebeat.com/technology/microsoft-ai-chief-says-company-was-set-free-from-openai-to-pursue-superintelligence | Snippet-level corroboration only (article fetch returned HTTP 429): Maia 200 "already running in production across data centers in Iowa and Arizona, with deployments planned for Italy, Australia, and South Korea." |

### Negative controls (searched, nothing found — record these so the gap is not re-searched blindly)

| Check | URL | Result |
|---|---|---|
| Maia SDK public documentation | https://learn.microsoft.com/en-us/azure/maia/ | **HTTP 404.** No public docs tree, no version numbers — consistent with the SDK remaining a gated preview. |
| Azure Cobalt 200 VM preview blog | https://azure.microsoft.com/en-us/blog/new-azure-cobalt-200-vms-deliver-50-performance-improvement-fully-optimized-for-modern-agentic-ai-workloads/ | 2026-06-02 official post; covers Cobalt 200 VM regions only and **does not mention Maia 200 regions** — so the Italy/Australia/South Korea list has **no first-party blog backing**. |
| MLPerf Training v6.0 | https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ | 2026-06-16; 24 submitters including Azure, but **Maia is not mentioned anywhere**. |
| Azure HPC blog — Azure's MLPerf v6.0 submission | https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inside-llama-3-1-405b-mlperf-training-on-azure-system-level-insights-at-8k-gpu-s/4529296 | Microsoft's own account: the submission was **8,192 NVIDIA GB200 GPUs** on Llama 3.1 405B, not Maia. |
| Azure Infrastructure blog, Apr–Aug 2026 | https://techcommunity.microsoft.com/category/azure/blog/azureinfrastructureblog | Silicon posts in the window are **Cobalt-only** (Cobalt 200 Rowhammer protection 2026-06-25; Cobalt workload-aware power management 2026-07-16). No new Maia technical post since the January 2026 deep-dive. |
| GitHub / package registries | — | No Maia compiler, runtime, driver, kernel-library or NPL repository. No Triton-for-Maia backend upstreamed to `openai/triton`. |

### Adjacent — found in-window but NOT Maia sources

| Resource | URL | Why it is not a Maia source |
|---|---|---|
| arXiv:2606.18170 — "The Multipath Reliable Connection (MRC) Transport" | https://arxiv.org/abs/2606.18170 | Submitted 2026-06-16; ~40 authors including Microsoft's Adrian Caulfield and Michael Papamichael; RoCEv2 extension with per-packet multipath and sender-based congestion control, released via OCP ~2026-05-05 by OpenAI with AMD, Broadcom, Intel, Microsoft, NVIDIA. **The paper does not mention Maia.** Secondary claims that Maia implements MRC are unsourced. Ecosystem context only. |
| Data Center Frontier — Fairwater blueprint | https://www.datacenterfrontier.com/hyperscale/article/55317925/inside-microsofts-global-ai-infrastructure-the-fairwater-blueprint-for-distributed-supercomputing | Mount Pleasant, Wisconsin campus fully operational ~2026-06-23, single coherent cluster over an 800 Gbps Ethernet fabric. **This is an NVIDIA GB200 deployment, not Maia** — do not attribute its capacity or fabric to Maia. |

### Out-of-window / not promoted

| Resource | URL | Why excluded |
|---|---|---|
| TrendForce — Microsoft 2027 interim AI chip reporting | https://www.trendforce.com/news/2025/07/04/ | 2025-07-04, **predates the repo baseline**. Origin of the dual-chiplet Maia 280 (two Braga dies, 2027) / Braga-R and Clea 2028+ roadmap. Rumor / analyst class, low confidence, nothing newer found April–August 2026. Not promoted into any table. |

### Next search trigger

**After 2026-08-25**, re-scan for Hot Chips 38 session AI 1 slides and video. That talk is expected to be the first *system-level* Maia 200 disclosure and the most likely source for the currently **not disclosed** figures: die size, tile/cluster counts, clock, rack and pod design, cooling, and full network topology.
