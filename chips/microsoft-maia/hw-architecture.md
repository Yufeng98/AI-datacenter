# Microsoft Maia Hardware Architecture

*as_of: 2026-09-13*
*Primary sources: HC2024 "Inside Maia 100" (Sherry Xu, Microsoft); Maia 200 deep-dive blog (January 2026); arXiv:2608.24664 "Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration" (Xu et al., Microsoft, 2026-08-25); Microsoft Tech Community Maia 200 Hot Chips 38 blog (2026-08-25)*
*Review note (2026-08-08): re-verified against the 2026-04-05 → 2026-08-08 window. **No Maia 100 or Maia 200 hardware specification changed**, and no new Maia silicon was disclosed. Every figure below remains the January 2026 / HC2024 figure. The only new material is deployment status and a scheduled future disclosure — see §8.*
*Review note (2026-09-13): **major update.** The Hot Chips 38 talk scheduled as of 2026-08-08 has now happened, accompanied by a full Microsoft-primary arXiv paper. Die size, tile/cluster counts, clock, network topology, and several performance figures that were previously "not disclosed" are now Microsoft-confirmed — see §9.*

---

## Overview

Microsoft Maia is a Cloud AI Accelerator family custom-designed by Microsoft for Azure AI workloads. The hardware is developed in close co-design with the software stack and system (rack + network), targeting large-scale transformer inference and training. Two generations have shipped: **Maia 100** (TSMC 5nm, 2024) for training and inference, and **Maia 200** (TSMC 3nm, 2026) optimized for inference.

---

## 1. Compute Engine

### Maia 100

| Parameter | Value |
|-----------|-------|
| Process | TSMC N5 (5nm) |
| Die size | ~820 mm² (reticle-limited) |
| Transistors | ~105 billion |
| Package | CoWoS-S (SoC die + 4× HBM2e) |
| TDP nominal | 500 W |
| TDP max | 700 W |
| Peak MX6 | 3 POPS (peta operations per second) |
| Peak MX9 | 1.5 POPS |
| Peak BF16 | 0.8 POPS |

**Tensor Unit (16×R×16):**

The tensor unit is the primary compute engine for matrix-multiply workloads (GEMM, batched GEMM, attention). It is designed around the **OCP MX (Microscaling) format**, which Microsoft co-developed with AMD, Arm, Intel, Meta, NVIDIA, and Qualcomm. The "R" dimension of the 16×R×16 format varies by data type — smaller R for lower-bit precisions (MX6 = 3 POPS; MX9 = 1.5 POPS; BF16 = 0.8 POPS).

**Vector Processor:**

A loosely coupled superscalar engine running a **custom ISA** (not publicly documented). Handles:
- Element-wise operations (activation functions: GELU, SiLU, ReLU)
- Normalization (RMSNorm, LayerNorm)
- Residual additions
- FP32 and BF16 data types

**DMA Engine:**

Decoupled from the compute engines; manages data movement between HBM and on-chip SRAM. Supports multiple tensor sharding schemes and **hardware semaphores** for asynchronous producer-consumer pipelines (double-buffering without software polling).

### Maia 200

| Parameter | Value |
|-----------|-------|
| Process | TSMC 3nm (paper says only "3nm"; N3 vs N3P vs N3E **not specified** by Microsoft) |
| Transistors | >140 billion (arXiv:2608.24664 confirms exactly) |
| Die | **26×33 mm**, near-reticle-sized monolithic die (~858 mm² arithmetic; ServeTheHome separately reports "~820 mm² altogether") — added 2026-09-13, arXiv:2608.24664 |
| Package | **CoWoS-S, 75×75 mm**, HBM co-located on silicon interposer — added 2026-09-13 |
| Clock | **2 GHz** — added 2026-09-13, arXiv:2608.24664 |
| TDP | 750 W SoC TDP, distributed via **19 metal layers** (metal-layer count added 2026-09-13) |
| Peak FP4 | >10 PFLOPS (headline confirmed as **10,145 TFLOPS**, arXiv:2608.24664, 2026-09-13) |
| Peak FP8 | >5 PFLOPS (headline confirmed as **5,072 TFLOPS**, arXiv:2608.24664, 2026-09-13; a separate benchmark-section figure states 4,785 TFLOPS FP8 for a specific 36-active-tile configuration — both are paper-stated, not reconciled here) |
| Efficiency | **13.3 Tflop/W (FP4) / 6.7 Tflop/W (FP8)** — added 2026-09-13, arXiv:2608.24664 |

**Architecture name (added 2026-09-13):** Microsoft names this class of design **Software Defined Locally Accessed Dataflow Architecture (SDLA)** — software has explicit control over data movement between HBM and localized SRAMs, as opposed to implicit hardware caching. Source: arXiv:2608.24664.

**Compute hierarchy (added 2026-09-13, arXiv:2608.24664):** SoC → **4 Clusters** → **9 or 10 Tiles per cluster** (10 physical, 1 held in reserve for yield in at least some configurations — the paper's own benchmark uses "9 Tiles per cluster enabled," i.e. 36 active tiles chip-wide). This fills the "tile count, cluster count" gap flagged "not disclosed" as of 2026-08-08.

**Tile Tensor Unit (TTU):**
- Native FP4 and FP8 tensor cores (plus FP6 and BF16)
- High-throughput matrix multiply and convolution
- One TTU per tile
- **Added 2026-09-13 (arXiv:2608.24664):** MAC counts per cycle — FP4 65,536 / FP8,FP6 32,768 / BF16 8,192; reads 32×K×32 input matrices (K=64 FP4, K=32 FP8), outputs/accumulates 32×32 FP32 or BF16/FP16; per-tile peak at 2 GHz = **262.14 Tflop/s (FP4)**

**Tile Vector Processor (TVP):**
- Highly programmable SIMD engine
- Post-GEMM operations (activations, norms, gating)
- One TVP per tile, tightly coupled with TTU
- **Added 2026-09-13 (arXiv:2608.24664):** 256 lanes for ≤16-bit types at 3.07 Tflop/s; 128 lanes FP32 at 1.54 Tflop/s (both @ 2 GHz)

---

## 2. Data Path

### Maia 100 Data Flow

The Maia 100 data path is **compiler-scheduled** — the NPL/Triton compiler generates explicit DMA transfer sequences. There is no hardware cache; all data movement between HBM and SRAM is explicit.

```
HBM2e
  |-- (cluster DMA) --> On-chip SRAM (L1/L2 scratchpad)
                              |
                    +---------+---------+
                    |                   |
              Tensor Unit         Vector Processor
            (MX GEMM/Attn)      (FP32/BF16 post-ops)
                    |                   |
                    +---------+---------+
                              |
                    (DMA writeback) --> HBM2e
```

Hardware semaphores enable **double-buffering**: while the tensor unit processes tile N, DMA prefetches tile N+1 from HBM. This hides HBM latency and sustains high utilization.

### Maia 200 Hierarchical Data Flow

The tile → cluster hierarchy adds a second SRAM tier specifically for LLM inference:

```
HBM3e
  |-- (cluster DMA) --> Cluster SRAM (CSRAM) [shared per cluster]
                              |
                   (tile DMA) --> Tile SRAM (TSRAM) [per tile]
                                        |
                              +---------+---------+
                              |                   |
                         Tile TTU               Tile TVP
                       (FP4/FP8 GEMM)         (SIMD post-ops)
                              +---------+---------+
                                        |
                             (DMA writeback) --> TSRAM --> CSRAM --> HBM3e
```

Cluster SRAM is optimized for KV-cache staging (hot attention key-value tensors). TSRAM provides the hottest per-tile compute buffer at ~10–20× the bandwidth of HBM3e.

---

## 3. On-chip Memory

### Maia 100

| Feature | Detail |
|---------|--------|
| Total on-chip SRAM | ~500 MB |
| Tiers | L1 + L2 scratchpads (software-managed) |
| Management | Explicit DMA (no hardware cache, no TLB) |
| Primary use | Weight tiles, activation buffers, KV-cache segments |

The 500 MB on-chip SRAM is unusually large (compare: H100 has ~36 MB L2 + 228 KB SMEM/SM). This was a deliberate Microsoft design choice to offset the use of HBM2e (lower bandwidth than HBM3) — keeping hot data on-chip reduces HBM traffic.

### Maia 200

| Feature | Detail |
|---------|--------|
| Total on-die SRAM | 272 MB |
| Tier 1: TSRAM | Per-tile; ~10–20× HBM3e bandwidth; hot compute buffer; **3 MiB per tile** (added 2026-09-13, arXiv:2608.24664) |
| Tier 2: CSRAM | Per-cluster shared; stages HBM → TSRAM traffic; KV-cache; **35 MB per cluster** (added 2026-09-13, arXiv:2608.24664) |
| Management | Software-managed DMA (no hardware cache) |

*Component-level arithmetic check (2026-09-13):* 4 clusters × 35 MB CSRAM = 140 MB; up to 4×10×3 MiB = 120 MiB of TSRAM. These do not cleanly sum to the previously-disclosed 272 MB total (the two figures come from different documents — the January 2026 deep-dive gave the 272 MB total, the arXiv paper gives per-tile/per-cluster sizes) — treated as two independently sourced figures rather than force-reconciled.

---

## 4. Off-chip Memory

### Maia 100

| Feature | Detail |
|---------|--------|
| Type | HBM2e |
| Capacity | 64 GB |
| Bandwidth | 1.8 TB/s |
| Stacks | 4× HBM2e |
| Packaging | Co-packaged on CoWoS-S interposer |
| Generation choice | Deliberate cost/supply-chain decision (HBM2e over HBM3) |

### Maia 200

| Feature | Detail |
|---------|--------|
| Type | HBM3e |
| Capacity | 216 GB (still TrendForce/Tom's Hardware-sourced secondary figure — arXiv:2608.24664 was checked 2026-09-13 and does not restate an absolute GB capacity in the extracted text) |
| Bandwidth | 7 TB/s — Microsoft-primary confirmed 2026-09-13 (arXiv:2608.24664 abstract says "7 TB/s", the paper's own introduction separately says "7 TiB/s" — an internal inconsistency in the source; 7 TB/s is used here as the more conservative figure) |
| Stacks | **6× HBM3e** (HBM0–HBM5) — Microsoft-primary confirmed 2026-09-13, arXiv:2608.24664 |
| Host/PCIe link | **PCIe 6×8, 64 GB/s** — added 2026-09-13, arXiv:2608.24664 |
| Supplier | SK Hynix (reported sole supplier) |

---

## 5. Host Interface / Package

### Maia 100

| Feature | Detail |
|---------|--------|
| Host interface | PCIe Gen5 ×8 |
| Host bandwidth | 32 GB/s |
| Package | CoWoS-S 2.5D interposer |
| Firmware | Upgradeable (new capabilities unlockable post-deployment) |

### Ares Rack System

Microsoft built a custom rack ("Ares") co-designed with Maia 100:

| Feature | Detail |
|---------|--------|
| Name | Ares rack |
| Maia chips per rack | 32 (8 servers × 4 chips) |
| Rack power | ~40 kW |
| Form factor | Custom wider-than-19" (not standard OCP) |
| Cooling | Mandatory closed-loop liquid cooling (rack-level) |
| ToR switches | Arista + Cisco dual-sourced; 3 switch SKUs per rack |
| Power management | Azure-integrated dynamic optimization |

---

## 6. Scale-up Interconnect

### Maia 100

| Feature | Detail |
|---------|--------|
| Protocol | Custom RoCE-like (enhanced reliability + load balancing) |
| All-gather / scatter-reduce | 4800 Gbps aggregate |
| All-to-all | 1200 Gbps |
| Physical links | 12× 400 GbE → 600 GB/s backend bandwidth |
| Encryption | AES-GCM (confidential compute native) |
| Topology | Unified: same network for scale-up and scale-out |

### Maia 200

| Feature | Detail |
|---------|--------|
| NIC location | Integrated on-die |
| Bandwidth | 2.8 TB/s bidirectional (pre-existing figure — see 2026-09-13 note below on reconciling with the newly disclosed 1.4 TB/s full-duplex ANC figure) |
| Topology | 2-tier scale-up network; **added 2026-09-13 (arXiv:2608.24664, Microsoft-primary):** a special case of a **2×2 1D Hamming Mesh** with cross-links added per physical tray for full intra-tray connectivity |
| NIC count / per-chip bandwidth | **Added 2026-09-13:** 28 integrated 400 Gbps Ethernet-based AI Network Controllers (ANC) per SoC = **1.4 TB/s full-duplex**. Of the 28: **20 fixed (unswitched) intra-tray links**, **8 switched links** across **4 identical planes** (2× 400G ANC per plane per SoC) — this is the primary confirmation of the previously analyst-only "8 Ethernet lanes, 4 network planes" shorthand |
| Board bandwidth asymmetry | **Added 2026-09-13:** north-south links 300 GB/s, east-west/diagonal 350 GB/s; switched portion (400 GB/s) can rebalance to a fully balanced 350 GB/s/direction, 1.4 TB/s total |
| Max cluster | 6,144 accelerators — **arithmetic now disclosed (2026-09-13):** each Tier-0 switch (51.2T, 128×400G ports) serves 48 SoCs across 12 trays (2 links each); remaining 32 T0 ports connect to up to 32 Tier-1 switches at 1:3 oversubscription; 48 × 128 = 6,144. "Smaller subset configurations are possible and deployed in the field." |
| Protocol | Advanced transport (custom congestion + reliability); **named 2026-09-13:** Microsoft's in-house **AI Transport Layer v2 (ATLv2)**, over lossless (PFC) Ethernet L2 + L3 IP routing, end-to-end **AES-GCM-256** encryption, ECMP/entropy-vector load balancing, selective retransmit. Microsoft states ATLv2 "later influenced the standardization of Ultra Ethernet" and was contributed to the **Ultra Ethernet Consortium (UEC) AI base transport profile**. |

**Reconciling 2.8 TB/s vs. 1.4 TB/s (open item, 2026-09-13):** the January 2026 deep-dive blog states "2.8 TB/s bidirectional"; the arXiv paper states "1.4 TB/s full duplex" for the same 28-ANC fabric. 2.8 ≈ 2×1.4, consistent with one figure counting each direction separately and the other counting combined duplex capacity, but neither source states this explicitly — flagged for a future pass rather than resolved by inference here.

---

## 7. Scale-out Interconnect

### Maia 100
- Unified Ethernet fabric serves both scale-up (intra-rack/intra-pod) and scale-out (inter-pod)
- External Arista/Cisco switches provide connectivity beyond the rack
- AES-GCM transport encryption enables inter-datacenter confidential compute paths

### Maia 200
- 2-tier on-die NIC provides intra-cluster and inter-cluster connectivity without a separate scale-out NIC
- Topology supports 6,144-accelerator clusters in a single logical domain
- **Added 2026-09-13:** "Maia 200 uses standard Ethernet cabling and switches for all networking" (arXiv:2608.24664) — no proprietary optics or cabling; Microsoft frames this standard-Ethernet choice as a direct contributor to total-cost-of-ownership savings

---

## 8. Deployment & Disclosure Status (added 2026-08-08)

No hardware change; this section records where the silicon runs and what is scheduled to be disclosed.

| Item | Status | Date / source |
|---|---|---|
| Maia 100 | In production for Azure OpenAI Services | Late 2024; HC2024 disclosure Aug 2024 |
| Maia 200 — US Central (Des Moines, Iowa) | In production | 2026-01-27 (announcement blog), reaffirmed at Build 2026 |
| Maia 200 — US West 3 (Phoenix, Arizona) | **In production** (was "coming next" in the January 2026 baseline) | Build 2026, 2026-06-02 |
| Maia 200 — Italy, Australia, South Korea | **Announced as next; no dates, no named Azure regions** | Build 2026 keynote as relayed by press (DCD, VentureBeat, Analytics India Magazine, 2026-06-03); no first-party blog restates it |
| Maia 200 system-level architecture | **Disclosed 2026-08-25** (was "scheduled" as of 2026-08-08). Session AI 1: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft), plus companion Tech Community blog and arXiv:2608.24664 paper. See §9 for full detail | arXiv:2608.24664; techcommunity.microsoft.com blog, 2026-08-25 |
| Maia 200 rack / pod / cooling design | **Partially disclosed 2026-09-13.** No Maia 200 equivalent of a named "Ares"-style rack, but the SoC package is confirmed **liquid-cooled by default**, deployable in air-cooled datacenters via an **integrated heat exchanger** | arXiv:2608.24664 |
| Maia 200 die size, tile count, cluster count, clock | **Disclosed 2026-09-13:** die 26×33 mm (~858 mm²); 4 clusters × 9-or-10 tiles/cluster; clock 2 GHz | arXiv:2608.24664 |
| MLPerf | **No Maia submission.** MLPerf Training v6.0 (2026-06-16) includes Azure, but that entry was 8,192 NVIDIA GB200 GPUs | mlcommons.org; Azure HPC blog |
| Maia 280 / Braga / Braga-R / Clea | **Not confirmed.** 2025 supply-chain reporting only (dual-chiplet Maia 280 from two Braga dies for 2027; Braga-R and Clea 2028+). Rumor/analyst class, predates the repo baseline, not promoted into any table above | TrendForce 2025-07-04 |

**Networking, adjacent — MRC.** The **Multipath Reliable Connection (MRC)** transport, an open extension of RoCEv2 with per-packet multipath, sender-based congestion control and fast loss/path-failure recovery, was released via OCP around 2026-05-05 by OpenAI together with AMD, Broadcom, Intel, Microsoft and NVIDIA (paper arXiv:2606.18170, submitted 2026-06-16, with Microsoft co-authors Adrian Caulfield and Michael Papamichael among ~40). **The paper does not mention Maia**, and secondary claims that Maia implements MRC are unsourced. It is recorded here as ecosystem context for Microsoft's Ethernet-transport direction only — the Maia 100 "custom RoCE-like transport" and Maia 200 "advanced transport" entries in §6 are **not** MRC unless and until Microsoft says so.

**Fairwater is not Maia.** Microsoft's Mount Pleasant, Wisconsin Fairwater campus (reported fully operational ~2026-06-23, run as a single coherent cluster over an 800 Gbps Ethernet fabric using a protocol co-developed with OpenAI and NVIDIA) is an **NVIDIA GB200 Blackwell deployment**. Its capacity and fabric figures must not be attributed to Maia.

**Des Moines / GPT-5.2 claim — checked 2026-09-13, not found.** A secondary-sourced claim that Maia 200 (216 GB HBM3e) serves "GPT-5.2-class workloads" at a Des Moines datacenter was checked against both primary sources for this update (arXiv:2608.24664 full text and the 2026-08-25 Tech Community blog full text). **Neither document mentions "Des Moines" or "GPT-5"** — the Des Moines/Iowa deployment itself is independently well-sourced (see rows above), but the specific GPT-5.2-workload pairing is **unconfirmed by any primary source found in this pass** and should be treated as unverified secondary content.

---

## System Summary Table

| Level | Unit | Chips | Compute | Memory | Network |
|-------|------|-------|---------|--------|---------|
| Die (Maia 100) | Single chip | 1 | 3 POPS MX6 | 64 GB HBM2e + 500 MB SRAM | — |
| Rack (Maia 100) | Ares rack | 32 | ~96 POPS MX6 | ~2 TB HBM2e | — |
| Die (Maia 200) | Single chip | 1 | 10,145 TFLOPS FP4 / 5,072 TFLOPS FP8 (arXiv:2608.24664, 2026-09-13; supersedes ">10 / >5 PFLOPS" rounding) | 216 GB (secondary) HBM3e + 272 MB SRAM | 1.4 TB/s full-duplex (28× 400G ANC) |
| Cluster (Maia 200) | 6,144 chips | 6,144 | **62 exaflop/s FP4** (arXiv:2608.24664, 2026-09-13; supersedes the prior ~61,440 PFLOPS derived estimate with a Microsoft-primary figure) | **43 PiB/s** aggregate memory bandwidth (arXiv:2608.24664) | **8.6 PiB/s** aggregate Ethernet network bandwidth (arXiv:2608.24664) |

---

## 9. Hot Chips 38 System Architecture Disclosure (2026-09-13)

*Primary sources: arXiv:2608.24664, "Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration" (Sherry Xu et al., Microsoft, submitted 2026-08-25); Microsoft Tech Community, "Maia 200: Software-defined dataflow and all-Ethernet networking for efficient inference on Azure" (Xu/Ranjan/Hoefler, 2026-08-25). Both fetched and read in full — see `research/microsoft-maia/investigations/hw-architecture.md` for the complete per-parameter breakdown; this section is the condensed version for the chip-level doc.*

### A. Architecture name and taxonomy

Microsoft names this design class **Software Defined Locally Accessed Dataflow Architecture (SDLA)**: software has explicit control over data movement between HBM and localized SRAMs (as opposed to implicit hardware caching). Framed as a data-movement-centric sibling to Flynn's instruction-centric taxonomy. This gives the pre-existing "tile → cluster hierarchy" description a formal name; it does not change any prior figure.

### B. Process, package, die — newly disclosed

TSMC 3nm (sub-node unspecified — not confirmed as N3P), **>140B transistors**, **26×33 mm monolithic die** (~858 mm² arithmetic; ServeTheHome's independent "~820 mm² altogether" is a close but non-identical secondary figure), **CoWoS-S packaging**, **75×75 mm package**, **750 W SoC TDP across 19 metal layers**, **2 GHz clock**. Microsoft states directly: "it is a full system including tray, rack, and network architecture scalable to thousands of accelerators in a single cluster and it is **in production in the fleet today**."

### C. Compute hierarchy — newly disclosed

**4 clusters**, each with **9 or 10 tiles** (10 physical, redundancy pattern similar to MTIA 300's spare PE row; the paper's own benchmark runs "9 Tiles per cluster enabled" = 36 active tiles). Each tile: 1 TTU + 1 TVP + 3 MiB TSRAM + DMA/Sync engines + Tile Control Processor. Each cluster: 35 MB CSRAM + Cluster Control Processor. TTU MACs/cycle: FP4 65,536 / FP8,FP6 32,768 / BF16 8,192; per-tile FP4 peak 262.14 Tflop/s at 2 GHz. TVP: 256 lanes ≤16-bit @ 3.07 Tflop/s, 128 lanes FP32 @ 1.54 Tflop/s.

**Chip-level peak, two figures from the same paper, not reconciled:** headline (abstract) **10,145 TFLOPS FP4 / 5,072 TFLOPS FP8** within 750 W (13.3 / 6.7 Tflop/W); benchmark-section (36 active tiles, unthrottled) **1,180 TFLOPS BF16 / 4,785 TFLOPS FP8**. The 10,145/5,072 headline figures closely confirm the prior repo's ">10 PFLOPS FP4 / >5 PFLOPS FP8."

### D. Network — HammingMesh topology, fully disclosed

**28× 400 Gbps integrated Ethernet ANCs per SoC = 1.4 TB/s full-duplex.** Topology: special case of a **2×2 1D Hamming Mesh** with per-tray cross-links. Of the 28 ANCs: **20 fixed intra-tray links + 8 switched links across 4 planes** (2× 400G ANC/plane/SoC) — this is the primary confirmation of the analyst-reported "8 Ethernet lanes / 4 network planes" shorthand. Transport: Microsoft's in-house **ATLv2** over lossless PFC Ethernet, **AES-GCM-256** end-to-end, later influencing the **Ultra Ethernet Consortium's AI base transport profile**. Two-tier switch fabric: each **T0 switch (51.2T, 128×400G) serves 48 SoCs/12 trays**; up to **32 T1 switches** at 1:3 oversubscription; **48×128 = 6,144 max SoCs**, confirming the pre-existing cluster-size figure with an explained arithmetic. Standard Ethernet cabling and switches throughout — no proprietary optics.

*Open reconciliation item:* the January 2026 blog's "2.8 TB/s bidirectional" and the paper's "1.4 TB/s full duplex" describe the same 28-ANC fabric; 2.8 ≈ 2×1.4 suggests a per-direction vs. combined-duplex accounting difference, but neither source states this explicitly.

### E. Measured performance — newly disclosed

BF16 matmul up to **99.69%** peak (compute-bound), **51.4%** peak bandwidth (memory-bound); FP8 matmul up to **96%** / **56%**; **Allgather** (8 chips) achieves **78%** of the latency bound and **94%** of the 1.4 TiB/s bandwidth bound. Single-chip **Qwen 2.5 7B** decode demo: **2,434 tokens/s** at 16,384-token context, >70% of estimated max. **ServeTheHome's separately reported "1.65 PFLOPS attention effective peak," "~1.3 TB/s BF16 AllReduce," and "655 GB/s All2All" were not found in the extracted arXiv text** — treated as analyst/HC38-slide-sourced only, not primary-confirmed this pass.

### F. Vendor claims (both unfalsifiable, both excluded from comparison tables)

Paper: "Maia 200 saves **30% cost (TCO) and 15% energy**... compared to any other AI accelerator in Microsoft's fleet" (internal data, no methodology). Blog: Maia 200 delivers **">40% higher token generation" under equal rack power** running MAI-Thinking-1 vs. "other leading accelerators in the Azure fleet" (no chip named, no methodology) — a new, distinct figure from the Build-2026-era "1.4× perf/W" claim already in this repo.

### G. Des Moines / GPT-5.2 — checked, not found

Neither the arXiv paper nor the Tech Community blog mentions "Des Moines" or "GPT-5" in any form. The specific "Maia 200 serves GPT-5.2-class workloads at Des Moines" claim remains unconfirmed by any primary source found in this pass.

---

## Sources

- [HC2024 PDF — Inside Maia 100](https://hc2024.hotchips.org/assets/program/conference/day2/81_HC2024.Microsoft.Xu.Ramakrishnan.final.v2.pdf)
- [Microsoft Tech Community — Inside Maia 100](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100-revolutionizing-ai-workloads-with-microsofts-custom-ai-accelerat/4229118)
- [Microsoft Blog — Maia 200](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/)
- [Microsoft Tech Community — Maia 200 deep dive](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312)
- [Azure Blog — silicon to service](https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/)
- [ServeTheHome — Maia 100 rack](https://www.servethehome.com/microsoft-maia-100-ai-accelerator-for-azure/)
- [TechRadar — HBM2e choice](https://www.techradar.com/pro/microsoft-deliberately-chose-to-use-old-tech-for-its-nvidia-gpu-rival-maia-100-ai-accelerator-uses-hbm2e-memory-and-the-mysterious-ability-to-unlock-new-capabilities-via-firmware-update)
- [Tom's Hardware — Maia 200](https://www.tomshardware.com/pc-components/cpus/microsoft-introduces-newest-in-house-ai-chip-maia-200-is-faster-than-other-bespoke-nvidia-competitors-built-on-tsmc-3nm-with-216gb-of-hbm3e)
- [TrendForce — Maia 200 HBM3e](https://www.trendforce.com/news/2026/01/27/news-microsoft-unveils-maia-200-ai-chip-on-tsmc-3nm-sk-hynix-reportedly-sole-hbm3e-supplier/)

### Added 2026-08-08 (§8 only — no specification source)

- [Hot Chips 38 program](https://hotchips.org/program/conference/) — scheduled Maia 200 system talk, 2026-08-25; **no content public as of 2026-08-08**
- [microsoft.ai — Build 2026 MAI keynote transcript](https://microsoft.ai/news/microsoft-build-2026-mai-keynote-transcript/) — 2026-06-02; deployment status and the 1.4× perf/W vendor claim
- [MLCommons — MLPerf Training v6.0](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 2026-06-16; negative control
- [arXiv:2606.18170 — MRC Transport](https://arxiv.org/abs/2606.18170) — 2026-06-16; adjacent networking context, does not mention Maia

### Added 2026-09-13 (§9 — Hot Chips 38 system disclosure)

- [arXiv:2608.24664 — Maia 200: A Software Defined Dataflow System for Large-scale AI Acceleration](https://arxiv.org/abs/2608.24664) — Sherry Xu et al., Microsoft, submitted 2026-08-25 (primary; full paper text extracted via `pdftotext -layout`)
- [Microsoft Tech Community — Maia 200: Software-defined dataflow and all-Ethernet networking](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/maia-200-software-defined-dataflow-and-all-ethernet-networking-for-efficient-inf/4548198) — Sherry Xu, Prashant Ranjan, Torsten Hoefler, 2026-08-25 (primary)
- [ServeTheHome — Microsoft's Maia 200 Accelerator at Hot Chips 2026](https://www.servethehome.com/microsofts-maia-200-accelerator-at-hot-chips-2026/) — 2026-08-25 (analyst; sole source for the attention-effective-peak, AllReduce, and All2All figures noted above as unconfirmed against the primary paper)
- [Hot Chips 38 program](https://hotchips.org/program/conference/) — talk realized 2026-08-25
