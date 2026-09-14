# Microsoft Maia Hardware Architecture — Investigation Report

*as_of: 2026-09-13*
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

---

## Update investigation — Hot Chips 38 system-level disclosure (dated 2026-09-13)

*Verdict: **major disclosure.** The Hot Chips 38 talk flagged "scheduled, content not yet public" on 2026-08-08 has now happened (2026-08-25, session AI 1: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng). It is accompanied by a Microsoft Tech Community blog post (2026-08-25, authors Sherry Xu, Prashant Ranjan, Torsten Hoefler) and — critically — a full academic paper on arXiv (2608.24664, "Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration", Sherry Xu et al., Microsoft Corporation, submitted 2026-08-25). Both were fetched and read in full (the arXiv PDF via `pdftotext -layout`, since WebFetch could not extract the PDF's decompressed text directly; the blog via its embedded Next.js JSON payload, since the rendered page also would not yield body text to WebFetch). This is the first system-level, silicon-detail disclosure of Maia 200 — the die size, tile/cluster counts, clock, and network topology that were "not disclosed" as of 2026-08-08 are now Microsoft-primary-confirmed.*

### A. Process, package, physical (arXiv:2608.24664, primary)

| Parameter | Value | Note |
|---|---|---|
| Process | **TSMC's 3nm process** | Paper says only "3nm" — does **not** specify N3 vs N3P vs N3E. The pre-existing repo entry "TSMC N3" should be read as "3nm, sub-node unconfirmed," not as a confirmed N3P claim. |
| Transistors | **>140 billion** | Confirms the January 2026 figure exactly |
| Die | **Near-reticle-sized monolithic die, 26×33 mm** | New. Arithmetic ⇒ ~858 mm². ServeTheHome's Hot Chips 38 coverage separately states "~820 mm² altogether" — close but not identical; the paper's dimensioned figure is treated as primary here, ServeTheHome's rounded total as a roughly-corroborating secondary figure. |
| Packaging | **CoWoS-S**, HBM co-located on a silicon interposer | New (confirms Maia 100's packaging choice carries to Maia 200) |
| Package size | **75 mm × 75 mm** | New |
| TDP | **750 W SoC TDP**, distributed via **19 metal layers** | Confirms 750 W; "19 metal layers" is new |
| Deployment | **"It is a full system including tray, rack, and network architecture scalable to thousands of accelerators in a single cluster and it is in production in the fleet today."** | Direct quote, Microsoft-primary — confirms in-production status independent of the Build 2026 deployment-status reporting already in this file |

### B. Compute architecture — SDLA, clusters, tiles (arXiv, primary)

Maia 200 is described as the first instance of a new architecture class Microsoft names **Software Defined Locally Accessed Dataflow Architecture (SDLA)** — software has explicit control over data movement between HBM and localized SRAMs, in contrast to implicit cache-based designs. This is new vocabulary; it supersedes no prior figure but gives the "tile→cluster hierarchy" already in this file (from the January 2026 deep-dive blog) a formal architectural name and taxonomy (the paper frames SDLA as a data-centric sibling to Flynn's instruction-centric taxonomy).

| Parameter | Value |
|---|---|
| Clusters per SoC | **4** |
| Tiles per cluster | **"nine or ten"** — i.e., 10 physical tiles per cluster with one held in reserve for yield, mirroring the redundancy pattern Meta uses on MTIA 300's PE grid. Benchmark section explicitly runs "a Maia 200 system with 9 Tiles per cluster enabled" (36 active tiles total: 4×9). |
| Per-tile compute units | One **Tile Tensor Unit (TTU)** + one **Tile Vector Processor (TVP)**, specialized tile memory, DMA engines, Sync engines, a **Tile Control Processor (TCP)** |
| TSRAM (per tile) | **3 MiB** |
| CSRAM (per cluster) | **35 MB** |
| Clock | **2 GHz** |
| TTU MAC counts (per cycle) | FP4: **65,536**; FP8/FP6: **32,768**; BF16: **8,192** |
| TTU per-tile peak (FP4, 2 GHz) | **262.14 Tflop/s** |
| TTU input matrix shape | 32×K×32 (K=64 for FP4, K=32 for FP8); output/accumulate into 32×32 FP32 or BF16/FP16 |
| TVP throughput | **256 lanes** for ≤16-bit types at **3.07 Tflop/s**; **128 lanes** FP32 at **1.54 Tflop/s** (both at 2 GHz) |
| Chip-level peak (36 active tiles, unthrottled, benchmark section) | **BF16 1,180 Tflop/s; FP8 4,785 Tflop/s** |
| Chip-level peak (abstract/headline) | **FP4 10,145 Tflop/s; FP8 5,072 Tflop/s** within 750 W (**13.3 / 6.7 Tflop/W**) |

**Note the headline (10,145 FP4 / 5,072 FP8) and the benchmark-section figure (4,785 FP8, no FP4 stated at chip level in that passage) are not identical** — plausibly different tile-count or clock assumptions between the abstract's top-line spec and the specific 36-tile benchmark configuration. Both are quoted from the same primary paper; report both rather than silently reconciling them. The **10,145 TFLOPS FP4 headline confirms** the >10 PFLOPS FP4 figure already in this repo (from the January 2026 blog) almost exactly, and the **5,072 TFLOPS FP8 headline confirms** the >5 PFLOPS FP8 figure almost exactly.

Datatype support (TTU/TVP): BF16, FP8 (E4M3/E5M2), FP6 (E2M3/E3M2), FP4 (E1M2/E2M1); OCP-compliant MXFP composition with E8M0 scaling factors, group size 32. Datatype lane counts for the Reshaper/vector path: INT8/UINT8 512 lanes, INT16/UINT16 256, INT32/UINT32 128, BFP16 256, FP16 256, FP4 256, FP6 256, FP8 256, FP32 128.

### C. Memory (arXiv, primary)

- **6× HBM3e stacks** (HBM0–HBM5 in the SoC diagram), confirming the pre-existing "6 HBM stacks" ServeTheHome-sourced figure now with a Microsoft-primary citation.
- Bandwidth stated as **"7 TB/s"** in the abstract but **"7 TiB/s"** in the introduction — an internal inconsistency in the paper itself (TiB/s ≈ 7.7 TB/s decimal). Flagged here; downstream tables should keep using "7 TB/s" (the more conservative, more widely corroborated figure) and note the paper's own unit ambiguity rather than silently pick one.
- **Absolute HBM capacity in GB is still not stated in the sections of this paper that were extracted.** The repo's existing "216 GB" figure remains TrendForce/Tom's Hardware-sourced (secondary) — this pass did **not** find a Microsoft-primary restatement of 216 GB specifically, though it is plausible the number appears in a table/figure not captured by text extraction. Treat 216 GB as still secondary-sourced pending a cleaner primary confirmation.
- PCIe: **PCIe 6×8, 64 GB/s** (new — host link generation now disclosed)

### D. Networking — HammingMesh topology, now fully disclosed (arXiv, primary)

This is the most significant new disclosure relative to the 2026-08-08 baseline, which had "on-die NIC, 2.8 TB/s bidirectional, 2-tier topology, 6,144 accelerators" with no topology detail.

- **28 integrated 400 Gbps Ethernet-based AI Network Controllers (ANC)** per SoC = **1.4 TB/s full-duplex** total per-chip network bandwidth. This is a materially different figure from, and should not be merged with, the pre-existing repo's "2.8 TB/s bidirectional" (2.8 TB/s ≈ 2× 1.4 TB/s full-duplex, so the two may be reconciled as one counting each direction separately and the other counting combined — flagged for a future pass to reconcile explicitly rather than guessed here).
- Transport: Microsoft's in-house **AI Transport Layer v2 (ATLv2)**, running over lossless (PFC) Ethernet L2 with L3 IP routing, **end-to-end AES-GCM-256 encryption**, ECMP + entropy-vector per-packet load balancing, selective retransmit. ATLv2 "later influenced the standardization of Ultra Ethernet" — Microsoft states it contributed this direction to the **Ultra Ethernet Consortium (UEC) AI base transport profile**.
- Topology: **"a special case of a 2×2 1D Hamming Mesh"** (citing the HammingMesh paper, Belk/Goel/Castro-Miguel) with cross-links added per physical tray for full intra-tray connectivity. Of the 28 ANCs per SoC: **20 use fixed (unswitched) links** within the tray; **8 connect to a switched network** organized as **4 identical planes**, with each SoC connecting **2× 400G ANCs to each plane** — this is the Microsoft-primary confirmation of ServeTheHome's shorthand "8 Ethernet lanes, split into four network planes."
- Blade/board bandwidth asymmetry: north-south links **300 GB/s**, east-west/diagonal **350 GB/s**; the switched portion (400 GB/s) can rebalance this to a **fully balanced 350 GB/s per direction, 1.4 TB/s total balanced network bandwidth**.
- Two-tier switched network, current design point **maximum 6,144 SoCs** (confirms the existing repo figure with a primary source and now explains the arithmetic): each **Tier-0 (T0) switch is 51.2 T, 128× 400G ports**, connecting to **48 Maia 200 SoCs across 12 trays with two links each**; the remaining 32 T0 ports connect to up to **32 Tier-1 (T1) switches at a 1:3 oversubscription ratio**. Total SoCs = 48 × 128 = **6,144**. "Smaller subset configurations are possible and deployed in the field."
- Deployment note: **liquid-cooled by default**; can be deployed in air-cooled datacenters via an **integrated heat exchanger** (Fig. 9) — this is the first disclosure of a Maia 200 cooling option, filling the "Maia 200 rack/pod/cooling design — not disclosed" gap noted on 2026-08-08.

### E. Measured performance (arXiv, primary)

- **BF16 matrix multiply**: up to **99.69%** of peak in the compute-bound regime; **>90%** of peak for multiplications >58 TFLOP of compute; memory-bound regime up to **51.4%** of peak bandwidth (rising above 50% for combined input+output operand size >113.5 MiB), benchmarked across **6,143 relevant matrix shapes**.
- **FP8 matrix multiply**: up to **96%** of peak in the compute-bound regime; up to **56%** in the memory-bound regime; roofline ridge point at arithmetic intensity 674 (vs. 169 for BF16).
- **Allgather** (8 Maia 200 chips across two trays, production-representative workloads): speed-of-light model SoL = max(4 µs, R / 1.4 TiB/s); measured **78% of the latency bound** and **94% of the bandwidth bound**. **This is the paper's only explicitly benchmarked collective** — direct-connect and ring algorithms compared, with direct favored for small messages and ring for large.
- **Cross-check against ServeTheHome's HC38-slide figures** ("~1.3 TB/s BF16 AllReduce", "655 GB/s All2All", "1.65 PFLOPS attention effective peak"): **none of these three figures were found in the extracted arXiv text.** 94% of the paper's 1.4 TiB/s Allgather bandwidth bound ≈ 1.45 TB/s, in the same neighborhood as ServeTheHome's "~1.3 TB/s AllReduce" but not an exact match, and AllReduce ≠ Allgather algorithmically. All2All and the attention-effective-peak figure appear nowhere in the extracted paper text — they may be in a Hot Chips 38 slide chart not reproduced in the arXiv paper, or a figure whose image content the text extraction could not read. **Treat all three ServeTheHome figures as analyst/HC38-slide-sourced only, not confirmed against the primary paper text in this pass.**

### F. Vendor efficiency claims — two distinct claims, both flagged

1. **Paper-stated (arXiv, primary):** "Internal data suggests that Maia 200 saves **30% cost (TCO) and 15% energy** compared to any other AI accelerator in Microsoft's fleet." This refines, and is a different framing from, the January 2026 "30% better performance per dollar" claim — now explicitly TCO + energy, still against Microsoft's own fleet, still internal/unaudited, no methodology published.
2. **Blog-stated (Tech Community, 2026-08-25, primary):** "Maia 200 delivers over **40% higher token generation** under the same rack power budget... when running the MAI-Thinking-1 model than other leading accelerators in the Azure fleet." This is a **new figure**, distinct from the Build-2026-era "1.4× perf/W" claim already recorded in this file (2026-06-02) — it is not stated whether the two are the same underlying measurement re-expressed or a new benchmark. Baseline is "other leading accelerators in the Azure fleet" — **no specific chip (e.g., GB200) is named** in either the blog or the arXiv paper. Both figures are **unfalsifiable vendor claims**: no workload spec, sequence length, batch size, or power-measurement boundary is published beyond what is quoted above. Exclude both from any comparison table as measured facts.

### G. Des Moines / GPT-5.2 claim — actively checked, not found in either primary source

The task brief for this pass flagged a secondary-sourced claim that Maia 200 uses 216 GB HBM3e and serves "GPT-5.2-class workloads at a Des Moines datacenter." **Both primary sources for this update (arXiv:2608.24664 full text, and the 2026-08-25 Tech Community blog full text) were searched for "Des Moines" and "GPT-5" — neither term appears in either document.** This claim remains **unconfirmed by any primary source found in this pass** and should continue to be treated as unverified/secondary. (The pre-existing repo record of Maia 200 running in the US Central/Des Moines, Iowa region comes from the January 2026 announcement blog and Build 2026, and is independently well-sourced — only the specific "GPT-5.2-class workload" pairing is unconfirmed.)

### H. End-to-end demonstration (arXiv, primary) — new

The paper includes a complete inference demo on a **single Maia 200 chip**: **Qwen 2.5 7B** (public model, 28 layers, inner dim 3584, FFN intermediate 18,944, 28 attention heads / 4 KV heads, GQA), memory-bound decode phase, 16,384-token context generating the 16,385th token. Achieved **2,434 tokens/s**, stated as **>70% of estimated maximum performance**; correctness validated against existing GPU inference solutions. This is the first third-party-model, single-chip, primary-sourced performance number for Maia 200 found in this repo's research to date.

### I. Related-work framing (arXiv, primary)

The paper explicitly places Maia alongside "AWS's Trainium and Inferentia, Google's TPUs, and Meta's MTIA" as comparable datacenter-provider-designed accelerators — useful confirmation that Microsoft's own competitive frame for Maia 200 is the custom-silicon cohort already tracked in this repo, not merely NVIDIA/AMD GPUs.

### Sources consulted in this window (added 2026-09-13)

- https://arxiv.org/abs/2608.24664 and https://arxiv.org/pdf/2608.24664 — "Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration", Sherry Xu, Marco Heddes, Jackson Peng, Tom Savell, Monica Tang, Prashant Ranjan, Jesse Benson, Ofer Dekel, Saurabh Dighe, Anupama Kurpad, Artour Levin, Matthew Mattina, George Petre, Cheng Tang, Yuan Yu, Li Zhang, Torsten Hoefler; Microsoft Corporation; submitted 2026-08-25 (primary; PDF fetched and converted with `pdftotext -layout`, ~1,077 lines of extracted text reviewed)
- https://techcommunity.microsoft.com/blog/azureinfrastructureblog/maia-200-software-defined-dataflow-and-all-ethernet-networking-for-efficient-inf/4548198 — Microsoft Tech Community, "Maia 200: Software-defined dataflow and all-Ethernet networking for efficient inference on Azure", Sherry Xu / Prashant Ranjan / Torsten Hoefler, 2026-08-25 (primary; body text extracted from the page's embedded Next.js `__NEXT_DATA__` JSON payload after WebFetch's rendered-text extraction returned only the page title)
- https://www.servethehome.com/microsofts-maia-200-accelerator-at-hot-chips-2026/ — ServeTheHome, Hot Chips 38 coverage, 2026-08-25 (analyst; sole source for the "1.65 PFLOPS attention effective peak", "~1.3 TB/s BF16 AllReduce", and "655 GB/s All2All" figures, none of which were found in the primary arXiv text)
- https://hotchips.org/program/conference/ — Hot Chips 38 program (talk realized 2026-08-25, was "scheduled" as of 2026-08-08)
- https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ — 2026-06-16
- https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inside-llama-3-1-405b-mlperf-training-on-azure-system-level-insights-at-8k-gpu-s/4529296
- https://techcommunity.microsoft.com/category/azure/blog/azureinfrastructureblog — Apr–Aug 2026 silicon posts are Cobalt-only
- https://arxiv.org/abs/2606.18170 — MRC transport, 2026-06-16
- https://www.trendforce.com/news/2025/07/04/ — 2025-07-04, pre-baseline Maia 280 / Braga roadmap origin
