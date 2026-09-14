# Rebellions ATOM / REBEL Hardware and Software Stack Summary

*chip: rebellions-atom*
*device_class: Inference Accelerator (Korea)*
*as_of: 2026-09-13*

---

## Overview

Rebellions is a South Korean AI chip startup (founded 2020, merged with SAPEON Korea in December 2024 to form Korea's first AI chip unicorn) that builds inference-optimized NPUs for datacenter deployment. The company has shipped two product generations — **ATOM** (Samsung 5nm, CGRA, PCIe) and **REBEL** / **REBEL-Quad** (Samsung 4nm, Neural Cores, UCIe chiplet) — and is deploying rack-scale systems under the **RebelRack** and **RebelPOD** brands, with **RebelCard** (Rebel100 module card) and **RebelServer** added as system-level products in 2026. Rebellions is backed by Arm and Samsung with a $2.3B valuation (March 2026 pre-IPO; ~KRW 3.4T in Korean reporting).

As of 2026-08-08 the silicon line-up is unchanged since the ISSCC 2026 Rebel100 disclosure — the 2026 news flow is software, systems, and corporate (see *Update — 2026-08-08* below).

The defining architectural choice is **static compilation over dynamic dispatch**: the RBLN compiler transforms a full ML model graph into a static binary at compile time, performing all memory allocation, tiling, and scheduling offline. There are no user-written kernels, no runtime JIT, and no user-visible ISA. This compile-once model sacrifices dynamic flexibility for deterministic, low-overhead inference execution.

---

## Hardware Architecture

### ATOM (Gen 1 — Samsung 5nm)

The ATOM is built on a **CGRA (Coarse-Grained Reconfigurable Array)** with 8 Neural Engines and a Command Processor, manufactured on Samsung 5nm. PE tiles are reprogrammed by the compiler for different AI operators. The RBLN-CA12 card provides 16 GB GDDR6 at 256 GB/s via PCIe Gen5 x16 in a single-slot FHFL form factor at 60–130 W TDP.

| Spec | ATOM | ATOM-Max |
|---|---|---|
| Process | Samsung 5nm | Samsung 5nm |
| Neural Engines | 8 | 8 (higher clock) |
| Peak FP16 | 32 TFLOPS | higher |
| Peak INT8 | 128 TOPS | higher |
| On-chip SRAM | 64 MB | 64 MB |
| Off-chip Memory | 16 GB GDDR6, 256 GB/s | 16 GB GDDR6 |
| Host Interface | PCIe Gen5 x16 | PCIe Gen5 x16 |
| TDP | 60–130 W | higher |
| Multi-instance | 16 hardware-isolated | 16 hardware-isolated |

**Deployments**: KT Cloud (Korea, May 2023), SK Telecom AI services, Japan, Saudi Arabia, US. Nearly 3 years of large-scale production inference experience.

### REBEL-Quad / Rebel100 (Gen 2 Chiplet — Samsung 4nm, Hot Chips 2025)

The REBEL-Quad is the world's first AI accelerator to adopt **UCIe-Advanced** for die-to-die interconnect. Four 320 mm² compute ASIC dies are connected at 16 Gbps per link (4 TB/s aggregate) on a single package alongside four 12Hi HBM3e stacks (144 GB total, 4.8 TB/s) and four integrated silicon capacitors for power supply.

| Spec | REBEL-Quad (Rebel100) |
|---|---|
| Process | Samsung 4nm |
| Compute dies | 4 × 320 mm² |
| Die-to-die | UCIe-Advanced, 16 Gbps, 4 TB/s aggregate |
| HBM3e | 4 × 36 GB = 144 GB |
| Memory bandwidth | 4.8 TB/s |
| Peak FP8 | 2,048 TFLOPS (2.048 PFLOPS) |
| Package TDP | ~600 W |
| vs NVIDIA H200 | Equal performance, lower power (ISSCC 2026) |
| vs top-tier GPU | 1.6× throughput, 50% less power, 3.2× TPS/W (Llama 3.3 70B FP8) |

The REBEL Single die architecture uses 16 Neural Cores arranged in clusters of 8, with 128 MB SW-managed SRAM (64 MB L1 distributed + 64 MB L2 shared), a RISC ISA for Neural Engines, and mesh interconnect between clusters.

### Rack Systems

| System | Nodes | Accelerators | FP8 | HBM3e | Memory BW | Network |
|---|---|---|---|---|---|---|
| RebelRack ⚠️ | 4 | 32 | 64 PFLOPS | 4.6 TB | 153.6 TB/s | quad-400GbE/node |
| RebelPOD ⚠️ | 8–128 | 64–1,024 | — | — | — | 800GbE |
| RebelCard | — | 1 (Rebel100 module card) | not disclosed | not disclosed | not disclosed | not disclosed |
| RebelServer | 1 | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed |

> ⚠️ **Sourcing caveat (added 2026-08-08).** The RebelRack/RebelPOD numbers above are **secondary-sourced** (The Register, 2026-03-30). Rebellions' own 2026-03-30 release that launched RebelRack and RebelPOD published **no technical specifications**, and no later primary source corroborates the 32-accelerator / 64 PFLOPS / 4.6 TB / 153.6 TB/s figures. Treat them as reported-but-unconfirmed.
>
> **RebelCard** (announced 2026-04-10) is a module-type accelerator card carrying Rebel100 — four NPU chiplets with 5th-generation HBM (HBM3E) — described as air-cooled. **RebelServer** (first named 2026-07-23) is the server product built around it. Neither has published specs, and no ship date is confirmed in any primary source.

---

## Software Stack

### Programming Model

The RBLN programming model is **framework-to-binary static compilation**. Users work entirely in PyTorch/TensorFlow/HuggingFace — there is no kernel authoring, no stream management, and no user-visible ISA.

```
[PyTorch / HuggingFace / TensorFlow model]
              |
     rebel.compile(model, input_shapes)
              |
      [RBLN Frontend Compiler]
       Graph capture, IR lowering
              |
     [Dependency Analysis]
      DRAM/SRAM access patterns
              |
      [RBLN Backend Compiler]
       Tiling, scheduling, RISC ISA codegen
       via Compute Library
              |
        [.rbln binary]
              |
      rbln.Runtime.load + .run()
              |
       [ATOM / REBEL NPU]
```

### Key Software Components

| Component | Role |
|---|---|
| RBLN SDK | Top-level toolchain; includes compiler + runtime |
| RBLN Compiler (frontend) | Graph capture from PyTorch/TF/HF; operator lowering |
| RBLN Compiler (backend) | Tiling, scheduling, RISC ISA code generation |
| Compute Library | Per-operation RISC ISA program generation |
| RBLN Runtime | Loads `.rbln` binaries; manages device memory and multi-instance |
| optimum-rbln | HuggingFace Transformers + Diffusers integration |
| rbln-model-zoo | Pre-compiled models (LLaMA, SDXL, BERT, T5, ResNet) |
| vLLM plugin | Official vLLM backend for LLM token streaming |
| Triton Inference Server | Production serving via gRPC/REST |
| RBLN NPU Operator | Kubernetes device plugin / Helm chart (breaking v0.3.x → v0.4.0, 2026-04-30) |
| `rbln-vs` / `rbln-smi` / `rbln-smd` | Hardware diagnostics, monitoring CLI, and device daemon (added/renamed 2026-05-29) |
| **OwLite** (SqueezeBits) | Quantization / model-compression toolkit — acquired 2026-06-30; integration into RBLN SDK not disclosed |
| **Fits on Chips** (SqueezeBits) | LLM benchmarking / LLMOps toolkit — acquired 2026-06-30 |
| **Yetter** (SqueezeBits) | Inference engine — acquired 2026-06-30 |

The three SqueezeBits entries are a **new tier** in this stack — model compression / inference serving sitting above the RBLN compiler — that did not exist in Rebellions' first-party toolchain before the 2026-06-30 acquisition. Rebellions has not published how (or whether) these products fold into the RBLN SDK release train.

---

## Architecture Roadmap

| Generation | Product | Process | Memory | Compute | Status |
|---|---|---|---|---|---|
| Gen 1 | ATOM | Samsung 5nm | 16 GB GDDR6, 256 GB/s | 32 TFLOPS FP16 | Production (2022+) |
| Gen 1+ | ATOM-Max | Samsung 5nm | 16 GB GDDR6 | Higher | Production |
| Gen 2 | REBEL Single | Samsung 4nm | HBM (per die) | ~est. 200-400 TFLOPS | Production |
| Gen 2 Chiplet | REBEL-Quad | Samsung 4nm | 144 GB HBM3e | 2,048 TFLOPS FP8 | Production / shipping 2025 |
| Gen 2 Rack | RebelRack | — | 4.6 TB HBM3e ⚠️ secondary-sourced | 64 PFLOPS ⚠️ | Available 2026 |
| Gen 2 Card | RebelCard (Rebel100 module card) | Samsung 4nm | HBM3E, capacity not disclosed | not disclosed | Announced 2026-04-10; entering validation in SK Telecom's datacenter — **not shipping**, no confirmed ship date |
| Gen 2 Server | RebelServer | — | not disclosed | not disclosed | Announced 2026-07-23 (A.X K1 single-server run); no specs disclosed |
| Future | REBEL-IO, REBEL-CPU | Samsung 4nm | — | Trillion-param MoE | Roadmap only — no tape-out, sampling, or spec disclosure as of 2026-08-08 |

---

## Update — 2026-04 to 2026-08 (recorded 2026-08-08)

*Window: 2026-04-06 → 2026-08-08. Sources: Rebellions newsroom (2026-04-10, 2026-06-17, 2026-06-30, 2026-07-23), docs.rbln.ai release notes, TheElec 2026-06-30, Seoul Economic Daily EN 2026-07-23 and 2026-05-27, Reuters/CNBC 2026-07-08, MLPerf Training v6.0 (2026-06-16). All prior-generation content above is retained unchanged.*

### No new silicon

**No new chip generation was announced in this window.** No REBEL Gen-3, and no tape-out, sampling, or spec disclosure for REBEL-IO / REBEL-CPU — those remain roadmap-only exactly as the roadmap table states. There is likewise **no cancellation or delay signal** for ATOM, ATOM-Max, or REBEL-Quad/Rebel100. Every REBEL-Quad figure in the tables above still traces to Hot Chips 2025 / ISSCC 2026 and is unchanged; no new architectural or performance numbers were published.

Negative results worth recording for a survey:
- Rebellions did **not** submit to MLPerf Training v6.0 (results published 2026-06-16; 24 submitters, Rebellions absent). It was also absent from MLPerf Inference v6.0 (2026-04-01).
- Rebellions has **no talk on the Hot Chips 38 program** (Aug 23–25 2026, Stanford). HC38 is still in the future as of this writing; nothing on its program is evidence of anything. Arm does have an AGI CPU talk scheduled there — *disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.*

### SqueezeBits acquisition (2026-06-30) — a new software-stack tier

Rebellions acquired **SqueezeBits**, a Korean AI model-compression / inference-optimization firm, via an **all-stock swap** that makes SqueezeBits a 100%-owned subsidiary (existing SqueezeBits investors received Rebellions shares). Cash terms were not disclosed. TheElec reports SqueezeBits' prior-year figures as KRW 2.646B revenue, KRW 897M operating profit, KRW 5.3B total assets.

SqueezeBits brings three products: **Fits on Chips** (LLM benchmarking / LLMOps), **OwLite** (quantization and compression), and **Yetter** (inference engine). The two firms had co-developed since 2024 and jointly contributed to vLLM. The stated goal is an end-to-end stack spanning hardware, model optimization, and inference serving.

This is **the most stack-relevant change in the window**: it adds a model-compression / inference-serving tier that Rebellions' first-party toolchain did not previously have. How these tools integrate with the RBLN compiler and its monthly release train has **not been disclosed**.

### Arm AGI CPU + RebelCard pairing (2026-04-10) — MOU

A three-way **MOU** between Rebellions, SK Telecom, and Arm targeting sovereign-AI and telecom-datacenter inference:

| Element | Detail |
|---|---|
| Host CPU | Arm **AGI CPU**, described in the primary release as built on **Arm Neoverse CSS V3** and as "the first Arm-designed data center CPU" |
| Accelerator | **RebelCard** — module-type card carrying **Rebel100** (four NPU chiplets, 5th-generation HBM / HBM3E) |
| Cooling | Air-cooled (vendor statement) |
| Deployment path | Co-development → validation in SK Telecom's AI datacenter → global telecom / public-sector sales |
| Vendor claim | Performance "comparable to current flagship GPUs" with better power efficiency — **no** TFLOPS, benchmark, or power figure given |
| Status | MOU / in validation. **Not** a design win, **not** sampling to customers, **not** shipping |

**Do not record a RebelCard ship date.** Aggregator coverage cites "Q3 2026"; the primary Rebellions release contains no launch date at all and the date could not be corroborated in any primary source.

### Largest model run to date (2026-07-23)

A **single RebelServer** ran SK Telecom's sovereign LLM **A.X K1** — over 500 billion parameters, **Mixture-of-Experts** architecture — serving concurrent real-time requests behind a map-based agent service. Rebellions claims serving efficiency "on par with global high-performance GPU servers."

**No quantitative figures were published**: no latency, throughput, TPS/W, batch size, or card count, and the number of Rebel100/ATOM devices in the server was not disclosed. The architecturally interesting part is that a 500B+ MoE model was served from a single box built on a 144 GB-per-package inference part — but without a device count that observation cannot be turned into a capacity claim. Rebellions separately states it has run SKT's "A. Phone Call Summary" high-traffic production service for approximately one year.

### OEM channel (2026-06-17) — MOU

**MOU with Giga Computing** (GIGABYTE's server arm) to co-develop AI servers and rack-scale systems around RebelCard / Rebel100, signed at Computex 2026 in Taipei by Marshall Choy (Rebellions CBO) and Vincent Wang (Giga Computing CCO). No timeline, no specs, no product SKU. Again: intent to co-develop, not a shipping product.

### RBLN SDK — four monthly releases in the window

The SDK is on monthly calendar releases. Two of the four carry **breaking changes**:

| Release | Compiler | Driver | Notable |
|---|---|---|---|
| 2026-04-30 | v0.10.3 | v3.0.0 | Parallel compilation; RSD support for vision transformers and Qwen3-VL; adds Qwen3-VL-32B, StableFast3D, TripoSR. **BREAKING:** RBLN NPU Operator v0.3.x → v0.4.0 — Helm values restructure `spec.daemonsets` → `spec.podDefaults`, auto device detection replaces the static ConfigMap, generic resource mode `rebellions.ai/npu` becomes default |
| 2026-05-29 | v0.10.4.post1 | v3.2.2 | New `rbln-vs` hardware diagnostic tool; `rbln_daemon` renamed `rbln-smd`; `rbln-smi` gains `mknod` for containers; automatic vLLM compilation removes the separate precompile step; `max_seq_lens` → `max_seq_len` for VLMs |
| 2026-06-26 | v0.11.0.post1 | v3.2.2 | **BREAKING:** first release supporting Transformers v5 — the compiled model format changed, so models compiled with earlier SDKs are incompatible and must be recompiled. `tensor_parallel_size` → `num_devices` across compiler APIs. Adds Gemma4 and EXAONE-4.5 |
| 2026-07-31 | v0.11.1.post1 | v3.2.2 | Adds `index_copy` / `index_add` / `grouped_conv1d`; **fixes numerical mismatches caused by an inadvertently bundled mimalloc allocator in v0.11.1** (so plain v0.11.1 is a known-bad build — the `.post1` suffix is load-bearing); `RBLN_VISIBLE_DEVICES` alias for `RBLN_DEVICES`; new `torch.rbln.explain()` (CPU-fallback diagnosis) and `torch.rbln.synchronize()`; compile-time reduction via op fusion |

For context immediately before the window: 2026-01-30 (compiler v0.10.0, driver v3.0.0, modular driver architecture, MoE support), 2026-02-27 (vLLM v0.13.0 support), 2026-03-27 (public PyTorch-RBLN release, initial Dynamic Resource Allocation driver support).

**No new hardware family was added to the SDK during 2026** — release notes still cover only the ATOM and REBEL series. The Transformers-v5 binary-format break is notable for a vendor whose entire value proposition is "compile once, deploy anywhere": the compiled artifact is not stable across SDK majors.

### Not confirmed / open

1. The "Q3 2026" RebelCard release date (aggregator-only; absent from the primary release).
2. Any device count, card count, or quantitative performance figure for RebelServer or the A.X K1 run.
3. The RebelRack spec row (32 accelerators / 64 PFLOPS / 4.6 TB HBM3e / 153.6 TB/s / quad-400GbE per node) — still uncorroborated by any Rebellions primary source; marked secondary-sourced above.
4. A UPI item dated 2026-08-07 ("SK Telecom, Rebellions expand Korean AI chip infrastructure") surfaced in search but returned HTTP 403 on fetch. Anything from the last ~week of this window is unverified.

---

## Key Differentiators vs Competitors

1. **Energy efficiency**: 3.2× TPS/W vs H100-class GPU on Llama 3.3 70B; 44% greater power efficiency vs A100 on T5-3B and SDXL-Turbo (ATOM)
2. **UCIe-Advanced first mover**: Industry's first AI accelerator using UCIe-Advanced die-to-die interconnect; OCP-listed
3. **Static compilation model**: Zero runtime overhead; deterministic latency — differentiates from GPU's dynamic dispatch model
4. **Samsung fab partnership**: Access to Samsung 4/5nm and strategic alignment for future scaling; Arm architecture license for CPU pairing
5. **Production deployment track record**: 3+ years large-scale inference across Korea, Japan, US, Saudi Arabia
6. **Sapeon merger synergy**: Combines Rebellions CGRA/REBEL expertise with SAPEON's SK Telecom enterprise relationships and SAPEON X220/X330 data center NPU experience

---

## Corporate

- Founded: 2020, Seoul, South Korea
- CEO: Park Sung-hyun (ex-Qualcomm)
- Merger: SAPEON Korea (SK Telecom spinoff), December 2024, equity ratio 2.4:1
- Acquisition: **SqueezeBits** (Korean model-compression / inference-optimization firm, founded March 2022), 2026-06-30 — all-stock swap, now a 100%-owned subsidiary; cash terms not disclosed
- Valuation: $2.3B (March 2026 pre-IPO). Korean reporting gives the same round as **KRW 640B raised at a KRW 3.4T valuation** — the won and dollar figures describe one round, the spread being FX drift across reporting dates (~1,450–1,600 KRW/USD), not separate tranches. Total funding to date ~$850M.
- Key investors: Samsung Electronics, Arm, SK Telecom, KT, SoftBank Ventures, **Aramco**, **SK hynix**; March 2026 pre-IPO participants named include the National Growth Fund (KRW 250B), Korea Development Bank (KRW 50B), Mirae Asset Group (KRW 300B)
- Revenue: KRW 32B (2025, as reported to CNBC 2026-07-08)
- IPO: preparing a Korean listing, CEO leaning **KOSPI** over KOSDAQ; preliminary-review filing targeted **within 2026** (previously August 2026 — pushed back per Seoul Economic Daily 2026-05-27), listing targeted H1 2027, possible later US listing; J.P. Morgan named as underwriter. *Timing is MEDIUM confidence and has moved more than once (a June-2026 filing on KOSDAQ was the plan as recently as Feb 2026).*
- **New 2026-09-13**: NVIDIA reported (Bloomberg, 2026-08-21) in **early-stage talks** with Rebellions covering "a technical partnership, an investment or perhaps even an acquisition" — Jensen Huang met CEO Sung-hyun Park at NVIDIA's Santa Clara HQ. Secondary coverage (eWeek, 2026-08-24) draws an explicit parallel to NVIDIA's late-2025 Groq deal (nonexclusive tech license + engineering-staff absorption) as a likely template, **not a confirmed structure**. No investment amount, stake, or scope disclosed; talks may not conclude. See the Update (2026-09-13) section below.
- Fab partner: Samsung Electronics
- Design partner: Synopsys (EDA)
- Partners: Pegatron (manufacturing), DOCOMO Innovations (Japan), SK Telecom; **Arm + SK Telecom** three-way MOU (2026-04-10) and **Giga Computing** MOU (2026-06-17) — both are intent-to-co-develop, not design wins

---

## Resources

- [Rebellions Developer Portal](https://rebellions.ai/developers/)
- [RBLN SDK Documentation](https://docs.rbln.ai/latest/index.html)
- [ATOM Architecture White Paper](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf)
- [REBEL-Quad Hot Chips 2025](https://rebellions.ai/newsroom/rebellions-debuts-rebel-quad-at-hot-chips-2025-breaking-ais-energy-tax-with-high-performance-chiplet-innovation/)
- [optimum-rbln GitHub](https://github.com/rebellions-sw/optimum-rbln)
- [rbln-model-zoo GitHub](https://github.com/rebellions-sw/rbln-model-zoo)
- [Tom's Hardware: ISSCC 2026 Rebel100](https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100)
- [ServeTheHome: REBEL-Quad 144GB HBM3E](https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144tb-hbm3e-accelerator-at-hot-chips-2025/)
- [The Register: RebelRack Rack-Scale (2026)](https://www.theregister.com/2026/03/30/rebellions_ai_rackscale/)
- [Next Platform: REBEL Architecture](https://www.nextplatform.com/2025/12/23/rebellions-ai-puts-together-an-hbm-and-arm-alliance-to-take-on-nvidia/)
- [OCP REBEL-Quad Listing](https://www.opencompute.org/chiplets/76/rebel-quad-ai-accelerator-ai-soc)

### Added 2026-08-08

- [RBLN SDK Release Notes](https://docs.rbln.ai/latest/supports/release_note.html) — monthly compiler/driver releases; authoritative for version strings
- [Rebellions acquires SqueezeBits (2026-06-30)](https://rebellions.ai/newsroom/rebellions_squeezebits_acquisition_260630/)
- [TheElec: SqueezeBits all-stock acquisition (2026-06-30)](https://www.thelec.net/news/articleView.html?idxno=11826)
- [Rebellions + SK Telecom + Arm MOU / RebelCard (2026-04-10)](https://rebellions.ai/newsroom/rebellions-collaborates-with-sk-telecom-and-arm-targeting-sovereign-ai-and-telecom-infrastructure/)
- [Rebellions + Giga Computing MOU (2026-06-17)](https://rebellions.ai/newsroom/rebellions-and-giga-computing-sign-mou-to-develop-next-generation-ai-server-and-rack-scale-solutions/)
- [RebelServer runs SKT A.X K1 (2026-07-23)](https://rebellions.ai/newsroom/rebelserver_run_k1/)
- [Seoul Economic Daily EN: RebelServer runs A.X K1 (2026-07-23)](https://en.sedaily.com/technology/2026/07/23/rebellions-runs-skts-sovereign-model-ax-k1-on-npu-server)
- [Seoul Economic Daily EN: KOSPI IPO preparation (2026-04-06)](https://en.sedaily.com/technology/2026/04/06/rebellions-gears-up-for-kospi-ipo-as-ai-chip-listing-race)
- [Reuters: Rebellions targets IPO next year (2026-07-08)](https://www.reuters.com/world/asia-pacific/s-koreas-rebellions-targets-ipo-next-year-followed-by-potential-us-listing-ceo-2026-07-08/)
- [CNBC: Rebellions IPO, KOSPI over KOSDAQ (2026-07-08)](https://www.cnbc.com/2026/07/08/rebellions-ipo-south-korea-ai-chips.html)
- [Rebellions $400M pre-IPO + RebelRack/RebelPOD launch (2026-03-30)](https://rebellions.ai/newsroom/rebellions-closes-400-million-pre-ipo-and-launches-rebelrack-and-rebelpod-to-accelerate-global-expansion/) — note: contains **no** technical specs
- [MLPerf Training v6.0 results (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — Rebellions absent from the 24 submitters

### Added 2026-09-13

- [Bloomberg — Nvidia in Talks With Chip Startup Rebellions for Potential Deal (2026-08-21)](https://www.bloomberg.com/news/articles/2026-08-21/nvidia-in-talks-with-chip-startup-rebellions-for-potential-deal)
- [eWeek — Nvidia Eyes Rebellions Deal as AI Inference Competition Heats Up (2026-08-24)](https://www.eweek.com/news/nvidia-rebellions-ai-chip-potential-deal-apac-south-korea/)

---

## Update (2026-09-13)

*Scan window 2026-08-08 → 2026-09-13. Classification: Roadmap (corporate) — no new silicon, no new spec, no new performance figure.*

- **NVIDIA–Rebellions talks (2026-08-21, Bloomberg; corroborated by eWeek 2026-08-24)**: NVIDIA is reported in early discussions with Rebellions covering a technical partnership, investment, or acquisition. Jensen Huang met CEO Sung-hyun Park at NVIDIA's Santa Clara HQ. Explicit (unconfirmed) parallel drawn to NVIDIA's late-2025 Groq deal structure. No amount, stake, or scope disclosed; may not conclude.
- **Pre-verified-fact check**: "Rebel100 → Saudi Aramco deployment target H1 2027" could **not be confirmed** in this pass. The only H1 2027 date on record anywhere in this research thread is the Korea IPO listing target (Reuters/CNBC, 2026-07-08); Aramco appears only as an investor, never tied to a deployment date. Treat the Aramco-deployment claim as unverified.
- Coverage-gap note: the 2026-08-07 UPI item on SK Telecom/Rebellions was re-attempted and still could not be retrieved.
- Searched and absent: no Rebel100/RebelCard ship date, no Sapeon-merger status change, no new ISSCC/Hot Chips 38/MLPerf disclosure.

New sources: see `research/rebellions-atom/search-results.md` → "Resources Added 2026-09-13" and `research/rebellions-atom/investigations/hw-architecture.md` → "Investigation Update — 2026-09-13".
