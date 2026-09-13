# Microsoft Maia Hardware Architecture — Investigation Report

*as_of: 2026-04-05*
*Primary sources: HC2024 "Inside Maia 100" (Sherry Xu, Microsoft); Microsoft Tech Community blog; Maia 200 deep-dive blog (January 2026)*

---

## Overview

Microsoft Maia is Azure's custom AI accelerator family, designed end-to-end for large-scale AI inference and training workloads on Azure. The first generation (Maia 100) was announced November 2023 and presented at Hot Chips 2024; the second generation (Maia 200) was announced and deployed in January 2026.

Both generations share a philosophy of **co-design across silicon, system, and software**: custom SoC + custom rack + custom networking + custom SDK, optimized for Azure OpenAI Services workloads.

---

## 1. Compute Engine

### Maia 100

| Component | Specification |
|-----------|---------------|
| Process | TSMC N5 (5nm) |
| Die size | ~820 mm² (reticle-limited) |
| Transistors | ~105 billion |
| Package | CoWoS-S interposer with 4× HBM2e stacks |
| TDP | 500 W nominal; up to 700 W |

**Tensor Unit (16×R×16):**
- High-speed matrix multiply engine
- Supports MX (Microscaling) data format — Microsoft-co-designed OCP standard
- Supported precisions: MX6-bit (3 POPS), MX9-bit (1.5 POPS), BF16 (0.8 POPS)
- "R" dimension varies with format; designed for transformer GEMMs and attention

**Vector Processor:**
- Loosely coupled superscalar engine
- Custom ISA (not publicly documented)
- Supports FP32 and BF16
- Handles element-wise operations, activations, normalization, reduction

**DMA Engine:**
- Supports different tensor sharding schemes
- Hardware semaphores for asynchronous programming
- Decouples memory movement from compute

### Maia 200

| Component | Specification |
|-----------|---------------|
| Process | TSMC N3 (3nm) |
| Transistors | >140 billion |
| TDP | 750 W |
| Peak FP4 | >10 PFLOPS |
| Peak FP8 | >5 PFLOPS |

**Tile Tensor Unit (TTU):**
- High-throughput matrix multiply and convolution engine
- Native FP8 and FP4 tensor cores
- One TTU per tile

**Tile Vector Processor (TVP):**
- Highly programmable SIMD engine
- Handles non-GEMM operations (activations, norms, gating)
- One TVP per tile, coupled with TTU

**Hierarchical Tile → Cluster Architecture:**
- **Tile:** Smallest autonomous compute + storage unit; TTU + TVP + TSRAM (Tile SRAM) + tile DMA
- **Cluster:** Multiple tiles sharing CSRAM (Cluster SRAM) + cluster DMA (HBM↔CSRAM staging)

---

## 2. Data Path

### Maia 100 Data Flow

1. DMA engine fetches weights/activations from HBM2e into on-chip SRAM scratchpads
2. Tensor unit performs matrix operations using MX-format inputs from SRAM
3. Vector processor applies post-GEMM operations (activation, bias, normalization) on SRAM outputs
4. DMA writes results back to HBM2e
5. Hardware semaphores enable double-buffering and producer-consumer pipelining

The data path is **compiler-scheduled** — no hardware cache hierarchy. The compiler (NPL or Triton) explicitly manages SRAM placement, DMA scheduling, and tensor tiling.

### Maia 200 Data Movement Fabric

- **Hierarchical DMA subsystem:** tile-level DMA (TSRAM ↔ CSRAM) + cluster-level DMA (CSRAM ↔ HBM)
- **Hierarchical NoC (Network-on-Chip):** interconnects tiles within a cluster and clusters within the chip
- Data locality is the primary design goal: hot activations stay in TSRAM (10–20× bandwidth of HBM); weights and KV-cache in CSRAM or HBM

---

## 3. On-chip Memory

### Maia 100

| Feature | Detail |
|---------|--------|
| Total on-chip SRAM | ~500 MB (L1 + L2 scratchpads) |
| Management | Software-managed (no hardware cache) |
| Bandwidth | Much higher than HBM2e (~1.8 TB/s); not precisely disclosed |
| Scope | Global SRAM pool accessible by tensor unit and vector processor via DMA |

Large scratchpad was a deliberate design choice: reduces HBM bandwidth pressure and improves power efficiency for transformer workloads where weight matrices are reused across token steps.

### Maia 200

| Feature | Detail |
|---------|--------|
| Total on-die SRAM | 272 MB |
| Tile SRAM (TSRAM) | Per-tile; hot activation / output buffer |
| Cluster SRAM (CSRAM) | Per-cluster shared; second tier; stages HBM traffic |
| Management | Software-managed via explicit DMA |
| Bandwidth advantage | TSRAM ~10–20× HBM3e bandwidth for hot data |

The two-level SRAM hierarchy directly targets LLM inference bottlenecks: auto-regressive decoding is memory-bandwidth-bound; keeping KV cache and weight slices in CSRAM/TSRAM dramatically reduces HBM access frequency.

---

## 4. Off-chip Memory

### Maia 100

| Feature | Detail |
|---------|--------|
| Memory type | HBM2e |
| Capacity | 64 GB |
| Bandwidth | 1.8 TB/s |
| Stacks | 4× HBM2e dies |
| Package | Co-packaged on CoWoS-S interposer |

Microsoft deliberately chose HBM2e (vs. HBM3) for Maia 100 — a cost and supply-chain decision acknowledged publicly, offsetting bandwidth with the large on-chip SRAM.

### Maia 200

| Feature | Detail |
|---------|--------|
| Memory type | HBM3e |
| Capacity | 216 GB |
| Bandwidth | 7 TB/s |
| Supplier | SK Hynix (reported sole supplier) |

---

## 5. Host Interface / Package

### Maia 100

| Feature | Detail |
|---------|--------|
| Host interface | PCIe Gen5 ×8 |
| Host bandwidth | 32 GB/s |
| Package | CoWoS-S 2.5D interposer (SoC die + 4× HBM2e stacks) |
| Form factor | Custom OAM-like card in Ares rack |
| Firmware | Upgradeable (confirmed — "new capabilities via firmware update") |

### Maia 200

| Feature | Detail |
|---------|--------|
| Host interface | not publicly disclosed |
| Package | TSMC N3 + advanced packaging (details not disclosed) |

### Ares Rack System (Maia 100)

| Feature | Detail |
|---------|--------|
| Rack name | "Ares" |
| Maia chips per rack | 32 (8 servers × 4 chips per server) |
| Rack power | ~40 kW |
| Form factor | Custom wider-than-standard (not 19" OCP) |
| Cooling | Mandatory liquid cooling (closed-loop, rack-level) |
| Networking | 3× top-of-rack switches (Arista + Cisco dual-sourced) |
| Power management | Azure-integrated dynamic power optimization |

---

## 6. Scale-up Interconnect

### Maia 100

| Feature | Detail |
|---------|--------|
| Protocol | Custom RoCE-like transport |
| All-gather / scatter-reduce BW | 4800 Gbps |
| All-to-all BW | 1200 Gbps |
| Physical | 12× 400 GbE ports → 600 GB/s backend BW |
| Encryption | AES-GCM (confidential compute) |
| Topology | Unified backend: single network for scale-up and scale-out |

### Maia 200

| Feature | Detail |
|---------|--------|
| NIC location | Integrated on-die |
| Bandwidth | 2.8 TB/s bidirectional |
| Topology | 2-tier scale-up network |
| Scale | Supports clusters of 6,144 accelerators |
| Protocol | Advanced transport protocol (enhanced reliability, custom congestion) |

---

## 7. Scale-out Interconnect

### Maia 100
- Same Ethernet fabric used for scale-up and scale-out (unified network)
- External Arista/Cisco Ethernet switches at top-of-rack
- Custom RoCE-like protocol with reliability enhancements
- AES-GCM encryption native in the transport

### Maia 200
- 2-tier scale-up NIC provides both intra-cluster and inter-cluster communication
- No separate dedicated scale-out NIC disclosed; unified fabric assumed

---

## 8. Comparison: Maia 100 vs. Maia 200

| Feature | Maia 100 | Maia 200 |
|---------|----------|----------|
| Process | TSMC 5nm (N5) | TSMC 3nm (N3) |
| Transistors | ~105B | >140B |
| TDP | 500–700 W | 750 W |
| Compute (FP4/FP8) | ~3 POPS (6-bit MX) | >10 / >5 PFLOPS FP4/FP8 |
| On-chip SRAM | ~500 MB | 272 MB |
| Off-chip memory | 64 GB HBM2e, 1.8 TB/s | 216 GB HBM3e, 7 TB/s |
| Scale-up BW | 4800 Gbps (12×400GbE) | 2.8 TB/s on-die NIC |
| Primary use case | Training + inference | Inference-optimized |
| Architecture | Monolithic SoC | Tile→Cluster hierarchy |

---

## Sources

- [HC2024 PDF — Inside Maia 100](https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf)
- [Microsoft Tech Community — Inside Maia 100 blog](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118)
- [Microsoft Blog — Maia 200 announcement](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/)
- [Microsoft Tech Community — Maia 200 deep dive](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312)
- [Azure Blog — silicon to service](https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/)
- [Tom's Hardware — Maia 200](https://www.tomshardware.com/pc-components/cpus/microsoft-introduces-newest-in-house-ai-chip-maia-200-is-faster-than-other-bespoke-nvidia-competitors-built-on-tsmc-3nm-with-216gb-of-hbm3e)
- [TechRadar — HBM2e design choice](https://www.techradar.com/pro/microsoft-deliberately-chose-to-use-old-tech-for-its-nvidia-gpu-rival-maia-100-ai-accelerator-uses-hbm2e-memory-and-the-mysterious-ability-to-unlock-new-capabilities-via-firmware-update)
- [ServeTheHome — Maia 100 rack/system](https://www.servethehome.com/microsoft-maia-100-ai-accelerator-for-azure/)
- [TrendForce — HBM3e SK Hynix](https://www.trendforce.com/news/2026/01/27/news-microsoft-unveils-maia-200-ai-chip-on-tsmc-3nm-sk-hynix-reportedly-sole-hbm3e-supplier/)

---

## Update investigation — window 2026-04-05 → 2026-08-08 (dated 2026-08-08)

*Verdict: **no hardware change.** Investigated Microsoft first-party channels (blogs.microsoft.com, microsoft.ai, Azure blog, Azure Infrastructure Tech Community), the Hot Chips 38 program, MLCommons, arXiv, and independent press. Zero Maia 100 or Maia 200 specification moved. Every figure in sections 1–7 above remains the HC2024 / January 2026 figure and is carried forward unchanged.*

### What was checked and found unchanged

| Subsystem | Baseline figure | Result of 2026-08-08 re-check |
|---|---|---|
| Compute (Maia 200) | TSMC N3, >140B transistors, 750 W, >10 PFLOPS FP4, >5 PFLOPS FP8 | unchanged; no new figure published |
| On-chip memory (Maia 200) | 272 MB, TSRAM + CSRAM tiers, ~10–20× HBM bandwidth for TSRAM | unchanged |
| Off-chip memory (Maia 200) | 216 GB HBM3e @ 7 TB/s, SK Hynix reported sole supplier | unchanged |
| Scale-up (Maia 200) | on-die NIC, 2.8 TB/s bidirectional, 2-tier, 6,144 accelerators | unchanged |
| Compute / memory / network (Maia 100) | all HC2024 figures | unchanged |
| Die size, tile count, cluster count, clock (Maia 200) | never disclosed | still **not disclosed** |
| Maia 200 rack / pod / cooling | never disclosed | still **not disclosed** — no Maia 200 analogue of the Ares disclosure exists |

### What is new (non-specification)

1. **Deployment status change.** Build 2026 (2026-06-02): Maia 200 is in production in **both** US Central (Des Moines, Iowa) and US West 3 (Phoenix, Arizona). The January 2026 baseline had Arizona as "coming next". Italy, Australia and South Korea are announced as next, with **no dates and no named Azure regions**; that list traces to the Build keynote as relayed by press (DCD 2026-06-03, VentureBeat, Analytics India Magazine 2026-06-03) with **no first-party Microsoft blog restating it**. The 2026-06-02 official Azure blog on Cobalt 200 VM preview is a useful negative control: it lists Cobalt regions and does not mention Maia.

2. **Vendor efficiency claim, not a measured figure.** Build 2026 MAI keynote (Mustafa Suleyman): "a further 1.4x performance-per-watt gain when running our MAI models on the Maia 200 end to end", explicitly "on top of the 30% improvement". The 30% is the January 2026 claim of "30% better performance per dollar than the latest generation hardware in our fleet today" — perf/**dollar**, against **Microsoft's own fleet**. The transcript never states the 1.4× perf/W baseline; the widely circulated "vs GB200" rendering is **press inference** from an adjacent sentence about benchmarking MAI-Thinking-1 head-to-head against GB200. No workload, sequence length, batch size, precision, cluster size, or power-measurement boundary was published. **Recorded as an unfalsifiable vendor claim; excluded from every comparison table.**

3. **Scheduled disclosure.** Hot Chips 38, session AI 1, Tuesday 2026-08-25, 2:15–4:15 PM PDT (chairs Sherry Xu, John Wright): "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft). **The event is 15 days in the future as of this investigation; no slides, abstract or specifications exist.** Nothing in this report derives from it. It is expected to be the first *system-level* Maia 200 disclosure — the most likely venue for rack, pod, cooling, tile-count and topology detail — and should be the first thing re-investigated after 2026-08-25.

### Negative results (important for the survey)

- **MLPerf Training v6.0 (2026-06-16): no Maia entry.** Azure is among the 24 submitters, but per Microsoft's own Azure HPC blog its submission was 8,192 NVIDIA GB200 GPUs on Llama 3.1 405B. Maia is not named anywhere in v6.0.
- **No new Microsoft Maia technical blog** since the January 2026 deep-dive. April–August 2026 Azure Infrastructure silicon posts are Cobalt-only (Cobalt 200 Rowhammer protection 2026-06-25; Cobalt workload-aware power management 2026-07-16).
- **Maia 280 / Braga / Braga-R / Clea: no in-window source.** All reporting is 2025 supply-chain (TrendForce 2025-07-04 and downstream aggregators) predating the repo baseline: dual-chiplet Maia 280 from two Braga dies for 2027, Braga-R and Clea pushed to 2028+. Rumor/analyst class; **not promoted into any table**.
- **MRC (Multipath Reliable Connection).** Open RoCEv2 extension with per-packet multipath, sender-based congestion control and fast loss/path-failure recovery; released via OCP ~2026-05-05 by OpenAI with AMD, Broadcom, Intel, Microsoft and NVIDIA; paper arXiv:2606.18170 submitted 2026-06-16 with Microsoft co-authors (Adrian Caulfield, Michael Papamichael among ~40). **The paper does not mention Maia.** Secondary claims that Maia implements MRC are unsourced. Ecosystem context only.
- **Fairwater (Mount Pleasant, Wisconsin), reported fully operational ~2026-06-23** as a single coherent cluster over an 800 Gbps Ethernet fabric, is an **NVIDIA GB200 deployment, not Maia.** Do not attribute its capacity or fabric to Maia.

### Sources consulted in this window

- https://hotchips.org/program/conference/ — HC38 program (retrieved 2026-08-08); future event, no slides
- https://microsoft.ai/news/microsoft-build-2026-mai-keynote-transcript/ — 2026-06-02, primary transcript (no visible update stamp; an "updated 2026-06-08" date could not be confirmed and is not recorded)
- https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/ — establishes the pre-baseline 30% perf/$ figure and the original Iowa-deployed / Arizona-next wording
- https://analyticsindiamag.com/ai-trends/everything-microsoft-announced-at-build-2026 — 2026-06-03
- https://www.datacenterdynamics.com/en/news/microsoft-launches-vms-based-on-cobalt-200-chip-in-preview-maia-200-already-in-production/ — 2026-06-03 (HTTP 403 on direct fetch; verified via search snippet only)
- https://venturebeat.com/technology/microsoft-ai-chief-says-company-was-set-free-from-openai-to-pursue-superintelligence — snippet-level corroboration (HTTP 429 on direct fetch)
- https://azure.microsoft.com/en-us/blog/new-azure-cobalt-200-vms-deliver-50-performance-improvement-fully-optimized-for-modern-agentic-ai-workloads/ — 2026-06-02, negative control (Cobalt only, no Maia regions)
- https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ — 2026-06-16
- https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inside-llama-3-1-405b-mlperf-training-on-azure-system-level-insights-at-8k-gpu-s/4529296
- https://techcommunity.microsoft.com/category/azure/blog/azureinfrastructureblog — Apr–Aug 2026 silicon posts are Cobalt-only
- https://arxiv.org/abs/2606.18170 — MRC transport, 2026-06-16
- https://www.trendforce.com/news/2025/07/04/ — 2025-07-04, pre-baseline Maia 280 / Braga roadmap origin
