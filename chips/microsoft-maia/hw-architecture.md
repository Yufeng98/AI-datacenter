# Microsoft Maia Hardware Architecture

*as_of: 2026-08-08*
*Primary sources: HC2024 "Inside Maia 100" (Sherry Xu, Microsoft); Maia 200 deep-dive blog (January 2026)*
*Review note (2026-08-08): re-verified against the 2026-04-05 → 2026-08-08 window. **No Maia 100 or Maia 200 hardware specification changed**, and no new Maia silicon was disclosed. Every figure below remains the January 2026 / HC2024 figure. The only new material is deployment status and a scheduled future disclosure — see §8.*

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
| Process | TSMC N3 (3nm) |
| Transistors | >140 billion |
| TDP | 750 W |
| Peak FP4 | >10 PFLOPS |
| Peak FP8 | >5 PFLOPS |

**Tile Tensor Unit (TTU):**
- Native FP4 and FP8 tensor cores
- High-throughput matrix multiply and convolution
- One TTU per tile

**Tile Vector Processor (TVP):**
- Highly programmable SIMD engine
- Post-GEMM operations (activations, norms, gating)
- One TVP per tile, tightly coupled with TTU

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
| Tier 1: TSRAM | Per-tile; ~10–20× HBM3e bandwidth; hot compute buffer |
| Tier 2: CSRAM | Per-cluster shared; stages HBM → TSRAM traffic; KV-cache |
| Management | Software-managed DMA (no hardware cache) |

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
| Bandwidth | 2.8 TB/s bidirectional |
| Topology | 2-tier scale-up network |
| Max cluster | 6,144 accelerators |
| Protocol | Advanced transport (custom congestion + reliability) |

---

## 7. Scale-out Interconnect

### Maia 100
- Unified Ethernet fabric serves both scale-up (intra-rack/intra-pod) and scale-out (inter-pod)
- External Arista/Cisco switches provide connectivity beyond the rack
- AES-GCM transport encryption enables inter-datacenter confidential compute paths

### Maia 200
- 2-tier on-die NIC provides intra-cluster and inter-cluster connectivity without a separate scale-out NIC
- Topology supports 6,144-accelerator clusters in a single logical domain

---

## 8. Deployment & Disclosure Status (added 2026-08-08)

No hardware change; this section records where the silicon runs and what is scheduled to be disclosed.

| Item | Status | Date / source |
|---|---|---|
| Maia 100 | In production for Azure OpenAI Services | Late 2024; HC2024 disclosure Aug 2024 |
| Maia 200 — US Central (Des Moines, Iowa) | In production | 2026-01-27 (announcement blog), reaffirmed at Build 2026 |
| Maia 200 — US West 3 (Phoenix, Arizona) | **In production** (was "coming next" in the January 2026 baseline) | Build 2026, 2026-06-02 |
| Maia 200 — Italy, Australia, South Korea | **Announced as next; no dates, no named Azure regions** | Build 2026 keynote as relayed by press (DCD, VentureBeat, Analytics India Magazine, 2026-06-03); no first-party blog restates it |
| Maia 200 system-level architecture | **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** Session AI 1, Tue 2026-08-25 2:15–4:15 PM PDT: "MAIA 200: A Data Center Scale AI system – MAIA-200 Accelerator", Prashant Ranjan & Jackson Peng (Microsoft) | hotchips.org program, retrieved 2026-08-08 |
| Maia 200 rack / pod / cooling design | **Not disclosed.** No Maia 200 equivalent of the Ares rack disclosure exists; expected to be the substance of the HC38 talk | — |
| Maia 200 die size, tile count, cluster count, clock | **Not disclosed** | — |
| MLPerf | **No Maia submission.** MLPerf Training v6.0 (2026-06-16) includes Azure, but that entry was 8,192 NVIDIA GB200 GPUs | mlcommons.org; Azure HPC blog |
| Maia 280 / Braga / Braga-R / Clea | **Not confirmed.** 2025 supply-chain reporting only (dual-chiplet Maia 280 from two Braga dies for 2027; Braga-R and Clea 2028+). Rumor/analyst class, predates the repo baseline, not promoted into any table above | TrendForce 2025-07-04 |

**Networking, adjacent — MRC.** The **Multipath Reliable Connection (MRC)** transport, an open extension of RoCEv2 with per-packet multipath, sender-based congestion control and fast loss/path-failure recovery, was released via OCP around 2026-05-05 by OpenAI together with AMD, Broadcom, Intel, Microsoft and NVIDIA (paper arXiv:2606.18170, submitted 2026-06-16, with Microsoft co-authors Adrian Caulfield and Michael Papamichael among ~40). **The paper does not mention Maia**, and secondary claims that Maia implements MRC are unsourced. It is recorded here as ecosystem context for Microsoft's Ethernet-transport direction only — the Maia 100 "custom RoCE-like transport" and Maia 200 "advanced transport" entries in §6 are **not** MRC unless and until Microsoft says so.

**Fairwater is not Maia.** Microsoft's Mount Pleasant, Wisconsin Fairwater campus (reported fully operational ~2026-06-23, run as a single coherent cluster over an 800 Gbps Ethernet fabric using a protocol co-developed with OpenAI and NVIDIA) is an **NVIDIA GB200 Blackwell deployment**. Its capacity and fabric figures must not be attributed to Maia.

---

## System Summary Table

| Level | Unit | Chips | Compute | Memory |
|-------|------|-------|---------|--------|
| Die (Maia 100) | Single chip | 1 | 3 POPS MX6 | 64 GB HBM2e + 500 MB SRAM |
| Rack (Maia 100) | Ares rack | 32 | ~96 POPS MX6 | ~2 TB HBM2e |
| Die (Maia 200) | Single chip | 1 | >10 PFLOPS FP4 | 216 GB HBM3e + 272 MB SRAM |
| Cluster (Maia 200) | 6,144 chips | 6,144 | ~61,440 PFLOPS FP4 | ~1.3 PB HBM3e |

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
