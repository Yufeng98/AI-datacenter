# d-Matrix Corsair — Summary

*chip: d-matrix*
*device_class: Digital In-Memory Compute (DIMC)*
*as_of: 2026-08-08*
*generations: Corsair (in full production, 2026-06-09) / Raptor (3DIMC — early silicon, ISCA 2026; pre-production)*

---

## One-Paragraph Summary

d-Matrix Corsair is a **Digital In-Memory Compute (DIMC)** inference accelerator built by a VC-backed startup that raised a $275M Series C in November 2025 (~$450M total). The core innovation is embedding **64×64 MAC arrays directly inside SRAM bit-cells**, making the memory itself the compute engine. A single Corsair PCIe Gen5 card contains 2 chips with 4 chiplets each (2,048 DIMC cores), 2 GB of integrated SRAM at 150 TB/s internal bandwidth, and 256 GB of LPDDR5X for model weight capacity. Two cards merge via a DMX Bridge card to form a 4,096-core, 4 GB SRAM logical unit. The stack supports OCP MX block-FP quantization (MXINT4/8/16), is programmed through the proprietary **Aviator** software stack (PyTorch + Triton front-ends, MLIR-based compiler), and scales out via dedicated **JetStream 400G** Ethernet NICs. Vendor performance claims: 10× faster than GPU, 3× better TCO for LLM inference, 60K tokens/s at 1 ms/token for Llama3 8B in a single server. **Corsair entered full production on 2026-06-09** (TSMC N6, with Alchip as design/production partner) and is beginning volume shipment to select qualified customers. **The successor, Raptor, is the commercial debut vehicle for 3DIMC** — a TSMC N4P logic die face-to-face bonded onto a 3D-DRAM die — and was characterized on real early silicon in an ISCA 2026 paper; see the update section below.

---

## Key Specifications

| Property | Value |
|----------|-------|
| Paradigm | Digital In-Memory Compute (DIMC) |
| Generation | Corsair (shipping); Raptor / 3DIMC™ (early silicon, pre-production) |
| Status | **In full production since 2026-06-09**; beginning volume shipment to select qualified customers |
| Process node | TSMC N6 (6nm); Alchip Technologies is the design/production partner |
| DIMC cores / card | 2,048 (single card) / 4,096 (dual card via DMX Bridge) |
| Integrated SRAM | 2 GB @ 150 TB/s per card (4 GB @ 300 TB/s dual card) |
| Capacity memory | 256 GB LPDDR5X @ ~400 GB/s per card |
| Peak compute | 2,400 TOPS INT8 / card |
| Numeric formats | MXINT4, MXINT8, MXINT16 |
| Host interface | PCIe Gen5 x16 |
| Die-to-die | DMX Link @ ~1 TB/s (chiplet-to-chiplet, 115 ns latency) |
| Card-to-card | DMX Bridge @ 512 GB/s (distinct from DMX Link — do not conflate) |
| Intra-rack | FabreX PCIe memory fabric / SuperNODE (GigaIO assets acquired 2026-04-02) |
| Scale-out | JetStream 400G Ethernet NIC |
| Rack blueprint | SquadRack (OCP Global Summit, 2025-10-14; with Arista, Broadcom, Supermicro) |
| Software stack | Aviator (PyTorch, Triton, MLIR compiler) + Wallaroo serving/control plane (acquired 2026-08-03) |

*Note on counting: 2,048 cores / 2 GB SRAM are **per card**; 4,096 cores / 4 GB SRAM are the **dual-card** (DMX Bridge) figures.*

---

## Positioning

- **Target**: Datacenter LLM inference (not training)
- **Vs GPU** *(vendor claims, no disclosed baseline or methodology)*: 10× latency, 3× TCO, 3× energy
- **Key insight**: Weight-stationary DIMC — weights never move; activations stream in
- **No HBM**: Deliberate cost choice; LPDDR5X for capacity; SRAM for performance
- **PCIe-native**: Incremental deployment alongside existing GPU infrastructure

---

## Software Ecosystem

| Layer | Tool | Status |
|-------|------|--------|
| Deployment / orchestration | Wallaroo serving runtime + control plane (acquired 2026-08-03) | not public |
| Framework | PyTorch, Triton DSL | Public (standard) |
| Model toolkit | Model Factory | not public |
| Quantization | Compressor (MX formats) | not public |
| Compiler | MLIR-based AOT | not public |
| Runtime | Aviator Runtime | not public |
| Scale-out NIC | JetStream driver | not public |

---

## References

- [Hot Chips 2025 coverage — ServeTheHome](https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/)
- [Chips & Cheese deep-dive](https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for)
- [d-Matrix Technical White Paper](https://d-matrix.ai/pdf/d-Matrix-WhitePaper-Technical-FINAL.pdf)
- [IEEE paper — Corsair chiplet architecture](https://ieeexplore.ieee.org/iel8/40/5210076/11108245.pdf)
- [JetStream announcement](https://www.d-matrix.ai/announcements/jetstream/)
- [Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference — ISCA 2026](https://ramyadhadidi.github.io/files/dMatrix-Raptor-ISCA.pdf)
- [d-Matrix × Alchip 3D-DRAM collaboration (Raptor announcement, 2025-11-18)](https://www.d-matrix.ai/announcements/d-matrix-and-alchip-announce-collaboration-on-worlds-first-3d-dram-solution-to-supercharge-ai-inference/)
- [Corsair enters full production (2026-06-09)](https://www.d-matrix.ai/announcements/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/)
- [d-Matrix acquires GigaIO datacenter business (2026-04-02)](https://www.d-matrix.ai/announcements/acquisition-of-gigaio/)
- [d-Matrix acquires Wallaroo.ai (2026-08-03)](https://www.d-matrix.ai/announcements/d-matrix-acquires-wallaroo/)

---

## Corsair Production + Raptor Update (2026-08-08)

*Updated 2026-08-08. Baseline for this section is the repo's 2026-04-05 state. Prior-generation content above is retained. Every throughput, TFLOPS and tokens/s figure below that is not explicitly labelled "measured" is a **vendor marketing claim** published without batch size, accuracy-vs-precision disclosure, or benchmark methodology.*

### 1. Corsair is in full production (2026-06-09)

d-Matrix announced that the Corsair AI Inference Platform "is in full production, with products to begin shipping in volume to priority customers." Secondary coverage describes it as ready for volume shipment to hyperscalers, neoclouds and frontier AI labs, with the first **SquadRack** reference systems slated for deployment in summer 2026.

- **Correct status verb: IN FULL PRODUCTION / BEGINNING VOLUME SHIPMENT.** Not "deployed at scale." Availability is still gated — the July 2026 Parasail release still describes Corsair as "now available for select, qualified customers."
- **Customers: unnamed.**
- **Manufacturing (newly confirmed):** TSMC **N6** with **Alchip Technologies** as design/production partner; organic substrate, **no HBM and no CoWoS**; LPDDR5X for capacity. d-Matrix says it secured multi-year TSMC/Alchip capacity.
- This supersedes the repo's earlier implicit "announced/sampling" framing.

### 2. Raptor — the next silicon generation (3DIMC), with measured early silicon

The repo previously carried "3DIMC" only as an architecture trademark. It has a named successor product and real characterized silicon.

- **Announced 2025-11-18** jointly with Alchip: **Raptor**, successor to Corsair and the commercial debut vehicle for 3DIMC. Vendor claim: "up to 10× faster inference than HBM4-based solutions." **Pavehawk** is the lab-validated test silicon that preceded it.
- **ISCA 2026** (2026-06-27 – 2026-07-01, Raleigh) published *"Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference"* (Nair et al., d-Matrix Inc. + UBC; includes CTO Sudeep Bhoja). This reports **on-silicon characterization, not simulation.**

| Raptor parameter | Value (ISCA 2026 paper) |
|---|---|
| Logic process | TSMC **N4P** |
| Integration | Logic die **face-to-face bonded** directly onto a 3D-DRAM die, **36 µm µbump pitch** |
| Chiplet organization | 4 gangs × 4 slices; each slice = 4×4 tensor-engine array + 1 SIMD core |
| 3D-DRAM per chiplet | **840 DRAM banks** mapped to **256 independent channels** (16 per slice) |
| MCM | **4 chiplets @ 1.2 GHz** per MCM; **up to 4 MCMs per card** |
| Packaging | 9-4-9 organic substrate onto a **3D CoWoS** interposer |
| Power / thermals | **~422 W per MCM**, Tj up to 105 °C |
| Secondary memory tier | **8 on-package LPDDR5X-9600 devices = 128 GB per MCM** |
| Host / inter-MCM | **PCIe Gen7**; Gen-2 D2D at 32 Gbps/lane |
| **Measured** 3D-DRAM bandwidth | **~105 TB/s per card at 700 MHz**, **2.5 ns average flit latency** (paper: ~12.5× an HBM3 card) |
| Paper throughput claim | **4.71×** vs HBM and **2.44×** vs SRAM across Llama-3.1 70B, DeepSeek-V3, Kimi K2, GPT-OSS, Whisper, Canary |
| Launch / sampling date / pricing | **not disclosed** — status is EARLY SILICON / PRE-PRODUCTION |

### 3. Rack-scale specifications (additive to the card-level specs above)

Published on d-matrix.ai/product. These are internally consistent with the per-card figures already in this document (2 GB @ 150 TB/s, 256 GB LPDDR5X), so they extend rather than contradict them. All are **vendor** figures.

| Level | Compute (dense) | Performance memory | Capacity memory | Notes |
|---|---|---|---|---|
| **Dual card** (2 cards, DMX Bridge) | 4,800 TFLOPS MXINT8 / 19,200 TFLOPS MXINT4 | 4 GB @ 300 TB/s | up to 512 GB | 6,400 mm² total silicon; **512 GB/s card-to-card DMX Bridge** |
| **Inference server** (8 cards) | 19.2 PFLOPS MXINT8 / 76.8 PFLOPS MXINT4 | 16 GB @ 1,200 TB/s | up to 2 TB | PCIe-based scale-up; Llama3 8B 60,000 tok/s @ 1 ms/token |
| **Inference rack** (8 servers / 64 cards) | — | 128 GB @ 9.6 PB/s | up to 16.4 TB | up to 100B params in "Performance Mode", 1T+ frontier models in "Capacity Mode"; Llama3 70B 30,000 tok/s @ 2 ms/token |

> **Do not conflate the two interconnects.** **DMX Link** is the *die-to-die* / chiplet link at **~1 TB/s** (Hot Chips 2025; 115 ns D2D latency). The **512 GB/s DMX Bridge** figure on the product page is the *card-to-card* link. Both numbers are correct and describe different things.

### 4. SquadRack — rack-scale blueprint (2025-10-14)

Announced at the **OCP Global Summit** with **Arista, Broadcom and Supermicro**: d-Matrix's "first blueprint for disaggregated standards-based rack-scale" inference. SquadRack is the named vehicle for the summer 2026 first deployments. It was absent from earlier revisions of this document.

### 5. Acquisitions — two in four months

- **GigaIO datacenter business, 2026-04-02** (terms undisclosed). d-Matrix acquired the **SuperNODE** system (up to 32 accelerators), the **FabreX** PCIe Gen5 memory fabric (sub-200 ns cross-server memory access), and the Carlsbad engineering team. GigaIO, Inc. continues independently focused on edge; its site footer now reads "FabreX is a trademark of d-Matrix, Inc." CEO Sid Sheth: *"Inference is bigger than any one chip. It's now a systems problem."* This deal predates the repo's 2026-04-05 baseline and was already recorded in the research files — it simply had not been propagated to this summary.
- **Wallaroo.ai, 2026-08-03** (terms undisclosed) — the strongest net-new item after Raptor. Brings platform, IP and engineering/product/go-to-market staff. Wallaroo provides a **serving runtime and control plane** for deploying models across heterogeneous hardware (x86, Arm, GPU) in cloud, on-prem, edge and air-gapped environments, with **vLLM and SGLang** support. This extends the stack past Aviator into deployment/orchestration — a layer the software table previously lacked. Bowen Inc. was Wallaroo's exclusive financial advisor.

### 6. Partnerships and software collaborations

- **Parasail (2026-07)** — an **announced partnership and stated intent, not a deployment.** Parasail's own post says the companies "plan to explore expanded integration" across Parasail's fleet of 40+ datacenters in 15 countries and will share results "following the first series of deployments." The intended architecture is a heterogeneous **disaggregated split**: NVIDIA Hopper/Blackwell for compute-bound **prefill**, Corsair for latency-sensitive **decode**. "Up to 10× faster interactive inference" and "up to 3× better energy efficiency" are unqualified vendor claims — no model, baseline, batch size or benchmark disclosed.
- **Infinity (infinity.inc), 2026-07-22** — an independent company ($15M raised) automating kernel development for AI chips via its "Ignition" agent. Infinity reports generating a full inference stack running **Qwen3, Qwen3.5 and Gemma4 end-to-end in 10 days at up to 92% of speed-of-light**, and reaching **92% of Corsair's theoretical peak in 10 hours**. Vendor language elsewhere referring to "Corsair's 32 compute units" could not be independently verified and does not reconcile with the documented 2,048 DIMC cores / 16 chiplets per dual card; it is deliberately **not** recorded as a spec here.
- **Gimlet Labs (2026-03-12)** — predates the update window. Vendor-relayed claim of response time falling from ~24 s to under 2 s for Corsair+GPU vs GPU-only; workload unspecified, and Gimlet's own blog frames the number illustratively. Treat as marketing, not a benchmark.

### 7. Funding, benchmarks, conferences

- **Funding:** no 2026 round confirmed. The $275M already recorded is the **November 2025 Series C at a ~$2B valuation** (~$450M raised to date; Bullhound Capital, Triatomic Capital and Temasek leading, with QIA, EDBI and Microsoft's **M12** participating). M12's participation is why some June 2026 coverage carries "Microsoft backing" headlines — that is **not** a new investment.
- **MLPerf:** confirmed negative — **no d-Matrix/Corsair submission in any MLPerf Inference round.**
- **Hot Chips 38 (2026-08-23 – 2026-08-25, Stanford):** d-Matrix is **not listed in the advance program**. The conference is in the future relative to this update; recheck afterwards.
