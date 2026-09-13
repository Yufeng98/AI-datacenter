# Tesla FSD Chip — Search Results

**Device class:** Edge Inference SoC
**Research date:** 2026-04-05

---

## Layer 1 — Device Overview

Tesla's Full Self-Driving (FSD) chip is a custom ASIC SoC designed in-house by Tesla for autonomous driving inference. Two generations are deployed at scale:

- **HW3 (FSD Chip):** Samsung 14 nm, dual NPU, 73.7 TOPS INT8, released 2019
- **HW4 (FSD Computer 2 / AI4):** Samsung 7 nm, triple NPU, ~121 TOPS INT8, released 2023

Each vehicle carries two FSD computers for redundancy.

**Key sources:**
- https://fuse.wikichip.org/news/2707/inside-teslas-neural-processor-in-the-fsd-chip/
- https://en.wikichip.org/wiki/tesla_(car_company)/fsd_chip
- https://www.autopilotreview.com/tesla-hardware-4-rolling-out-to-new-vehicles/

---

## Layer 2 — Chip Specifications

| Parameter | HW3 | HW4 |
|---|---|---|
| Process | Samsung 14 nm | Samsung 7 nm |
| NPU count | 2 | 3 |
| NPU clock | 2.0 GHz | 2.2 GHz |
| NPU TOPS (INT8) | 36.86 per NPU / 73.7 total | ~40.5 per NPU / 121.6 total |
| MAC array | 96×96 per NPU | 96×96 per NPU |
| NPU SRAM | 32 MiB per NPU | 32 MiB per NPU |
| CPU | 12× ARM Cortex-A72 @ 2.6 GHz | 20× ARM cores @ 2.35 GHz |
| GPU | Mali G71 MP12 @ 1 GHz | improved GPU |
| DRAM | 8 GB LPDDR4 (68 GB/s) | 16 GB GDDR6 (224 GB/s) |
| Storage | — | 256 GB NVMe |
| TDP | ~100 W (board) | ~160 W (board) |

---

## Layer 3 — NPU Microarchitecture

Each NPU is a systolic MAC array:
- 96×96 = 9,216 MACs → 18,432 INT8 ops/cycle
- 8-bit × 8-bit multiply, 32-bit accumulate
- Per cycle: 256 bytes activation + 128 bytes weight from SRAM
- **ISA:** 8 instructions — 2 DMA (read/write), 3 dot-product variants, scale, element-wise add, no-op parameter slot
- In-order execution with out-of-order memory subsystem
- Static scheduling — workload mapped at compile time, no runtime dispatch unit

---

## Layer 4 — Memory Subsystem

- Per-NPU: 32 MiB on-chip SRAM for activations and weights
- HW3: 8 GB LPDDR4 @ 4266 Mbps, 128-bit bus → 68 GB/s
- HW4: 16 GB GDDR6 @ 14 Gbps, 128-bit bus → 224 GB/s (3.3× improvement)
- HW4: 256 GB NVMe for map/model storage

---

## Layer 5 — Compiler / Toolchain

Tesla's proprietary NN compiler:
- Input: PyTorch models (trained on Dojo / GPU cluster)
- Quantization-aware training → INT8 models shipped to vehicles
- Two-stage compile: coarse topology mapping → fine weight pruning + quantization
- Layer fusion: conv-scale-act-pooling combined into single tiled workload
- Static scheduling: tile-based partitioning of weight matrices → specific NPU block assignments
- Output: optimized binary for target chip revision (HW3 vs HW4 differ in binary)
- FSD v13 required extreme INT8 optimization to run on HW3 (16→8 bit quantization)

---

## Layer 6 — Software / Framework

- **Training:** PyTorch (GPU cluster / Dojo D1)
- **Compiler:** Tesla proprietary (closed source)
- **Runtime:** Custom microcode static scheduler on NPU
- **OS:** Custom embedded Linux on Cortex-A clusters
- **Inference pipeline:** 48 neural networks, camera-first (Tesla Vision), BEV occupancy networks

---

## Layer 7 — Inference Pipeline

1. Vision Engine: 8× cameras → feature representations (1 billion pixels/s capable)
2. Primary NPU inference: vehicles, pedestrians, lane boundaries, traffic signs
3. Ego-motion + trajectory network
4. Decision network: fuses sensor outputs for collision avoidance
5. FSD v12+: end-to-end neural network replacing rule-based 300K-line code

---

## Layer 8 — Communication / IO

- Camera bus (proprietary high-speed serial)
- Ethernet for OTA updates
- Dual FSD computer redundancy with cross-check

---

## Layer 9 — Precision / Data Types

- INT8 (8×8 multiply, 32-bit accumulate) — primary inference
- FP32 used in training pipeline
- INT8 quantization at deployment (model weights pre-quantized before OTA)

---

## Layer 10 — Deployment / Scale

- Deployed in all Tesla vehicles since HW3 (2019)
- HW4 in vehicles from 2023 onward
- OTA model updates via Tesla proprietary binary format
- Robotaxi (Cybercab) uses AI4 platform

---

## Layer 11 — Programming Model

- Not user-programmable (closed ASIC)
- Tesla internal team uses PyTorch → proprietary compiler pipeline
- No public SDK

---

## Layer 12 — Open Source Components

- None (fully proprietary stack)
- Training uses open-source PyTorch but compiler/runtime are closed

---

## Layer 13 — Competing / Related Chips

- NVIDIA DRIVE Orin (262 TOPS, used by many OEMs)
- Qualcomm Snapdragon Ride (Cloud AI 100 class for automotive)
- Mobileye EyeQ6

---

## Layer 14 — Roadmap

- AI5 chip (announced concept 2025): next-gen inference engine
- Tesla shifting focus toward AI inference ASICs for Robotaxi fleet
- Optimus robot uses same AI4 platform

---

## Layer 15 — Key Publications / Talks

- WikiChip deep dive: https://fuse.wikichip.org/news/2707/inside-teslas-neural-processor-in-the-fsd-chip/
- Tesla AI Day 2021, 2022 (YouTube) — architecture walkthroughs
- Andrej Karpathy / PyTorch at Tesla talks
- ArXiv: https://arxiv.org/html/2411.16007v1 (multi-chiplet NPC analysis)

---

# Additional Resources — 2026-08-08 Scan

*Appended 2026-08-08. Covers material published between 2026-01-26 and 2026-08-08. Layer 14 (Roadmap) above,
which listed "AI5 chip (announced concept 2025)", is superseded by the detail below.*

## Note on source quality

Tesla published **no** primary material in this window: no datasheet, no whitepaper, no developer docs, no
conference talk. Every resource below is secondary reporting on a Musk statement, a photograph, an earnings
call, a deleted LinkedIn post, or owner/firmware sightings. They are grouped by evidentiary weight.

## A — Multi-outlet corroborated (announcement-level facts)

| Resource | Date | What it establishes |
|---|---|---|
| https://electrek.co/2026/04/15/tesla-ai5-chip-taped-out-musk-ai6-dojo3/ | 2026-04-15 | AI5 tape-out **announcement**; reproduces the X post text; names AI6 and Dojo3 as "in work" |
| https://www.morningstar.com/news/marketwatch/20260415530/mw-is-tesla-a-chip-stock-now-investors-are-cheering-a-semiconductor-milestone | 2026-04-15 | Independent confirmation of the announcement; volume manufacturing 2027 |
| https://www.benzinga.com/markets/prediction-markets/26/04/51885219/elon-musk-shows-off-first-physical-tesla-ai5-chip-tsla-up-14-in-5-days | 2026-04-17 | Independent confirmation; package description (half-reticle die, surrounding memory packages) |
| https://www.digitimes.com/news/a20260417VL202/tesla-ic-design-development-sk-hynix.html | 2026-04-17 | Independent confirmation; **SK hynix named as AI5 memory supplier** |
| https://electrek.co/2026/04/23/tesla-hw4-plus-upgrade-will-hw4-follow-hw3/ | 2026-04-23 | **AI4.1 / "AI4 Plus"** verbatim Musk quote (16→32 GB/SoC, 64 GB total, ~10% compute and bandwidth); HW3 retrofit; Electrek's contested bandwidth figures (HW3 ~48 GB/s, AI4 ~384 GB/s, Orin ~205, Thor ~273) |
| https://electrek.co/2026/01/26/tesla-quietly-starts-shipping-model-y-with-new-ai4-5-computer/ | 2026-01-26 | **AI4.5 / "AP45"**, part number 2261336-02-A, Fremont 2026 Model Y; hedged three-SoC firmware inference by @greentheonly |
| https://www.hotchips.org/advance-program/ | live | **Confirms no Tesla talk at Hot Chips 38** (2026-08-23 to 25); Waymo holds the AV keynote and automotive SoC slot |

## B — Samsung / Taylor 2 nm leg (relayed from a deleted LinkedIn post; no official confirmation)

| Resource | Date | Note |
|---|---|---|
| https://www.koreatimes.co.kr/business/tech-science/20260713/samsung-taylor-fab-enters-production-phase-for-teslas-ai5-chip | 2026-07-13 | Most useful of the set: states **both foundries received the AI5 design in April 2026**, which is what refutes any "Samsung taped out months later" sequencing |
| https://www.upi.com/Top_News/World-News/2026/07/13/samsung-tesla-foundry-texas-ai5-chip/8251783995948/ | 2026-07-13 | Independent relay; notes Samsung declined to comment |
| https://www.asiae.co.kr/en/article/2026071314023594058 | 2026-07-13 | Independent relay |
| https://www.techtimes.com/articles/320427/20260714/tesla-ai5-locks-samsung-2nm-taylor-flipping-node-assumption.htm | 2026-07-14 | Independent relay |
| https://xenospectrum.com/en/tesla-ai5-samsung-2nm-tapeout-dual-foundry/ | 2026-07 | Dual-foundry framing |
| https://hothardware.com/news/elon-musk-taps-samsung--tsmc-for-teslas-next-gen-ai5-chip | 2025-10 | Establishes that AI5 dual-sourcing predates 2026 |

**Handling rule:** cite these for *"Samsung Taylor 2 nm is reported as the second AI5 source"*, never for a
tape-out date, a node derivative (SF2/SF2P/SF2T), or a production start.

## C — Single-source; use only with explicit attribution, never as spec

| Resource | Date | Single-source content |
|---|---|---|
| https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc-rather-than-tsmc-claims-40x-performance-boost-over-the-predecessor | 2026-04 | **"KR 2613" package date code**; **384-bit bus**; **~768 GB/s – 1.536 TB/s** — all Tom's Hardware's own reading/arithmetic, uncorroborated elsewhere |
| https://www.tomshardware.com/tech-industry/artificial-intelligence/teslas-ai5-with-2nm-class-node-tapes-out-at-samsung-foundry-production-starts-soon-months-after-tsmc-tape-out | 2026-07 | Headline asserts "production starts soon" and "months after TSMC tape-out" — **both overstate the underlying evidence**; useful only as a record of the claim |
| https://247wallst.com/investing/2026/04/22/live-will-tesla-soar-after-announcing-q1-earnings/liveupdates/10/ | 2026-04-22 | Q1 2026 call live blog; **contains a transcription error rendering AI4.1 production as "mid-2025"** — do not use for dates |
| https://www.msn.com/en-us/news/technology/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc/ar-AA20Ynws | 2026-04 | Syndication of the Tom's Hardware piece — **not an independent source**, do not count it as corroboration |

## D — Marketing-claim provenance (for dating the "40×" figure correctly)

| Resource | Date | Note |
|---|---|---|
| https://www.autoevolution.com/news/tesla-hypes-ai5-autopilot-computer-as-neglected-hw3-owners-grow-angrier-259718.html | 2025-10-23 | Establishes the **"40×" claim originates at the Q3 2025 earnings call**, not April 2026 |
| https://www.tweaktown.com/news/108461/elon-musk-says-teslas-next-gen-ai5-chip-is-40x-faster-than-ai4-calls-it-a-beautiful-chip/index.html | 2025-10-24 | Same; corroborates the October 2025 origin |
| https://www.techbuzz.ai/articles/tesla-ai5-chips-challenge-nvidia-with-half-size-design | 2026-04 | Half-reticle framing |
| https://en.wikipedia.org/wiki/Tesla_Autopilot_hardware | live | Aggregated generation table; **secondary and partly derived from the same contested third-party bandwidth figures** — use as an index, not as a source |

## E — Consequences and context (not chip specs)

| Resource | Date | Note |
|---|---|---|
| https://www.theglobeandmail.com/investing/markets/stocks/NVDA/pressreleases/2504304/teslas-ai5-chip-recently-completed-tape-out-heres-why-this-could-be-the-most-important-development-in-the-companys-transition-from-automaker-to-ai-giant/ | 2026-04 | Press release / investor framing — **not a technical source** |
| https://electrek.co/2026/08/06/tesla-spacex-terafab-grimes-county-16-8-billion/ | 2026-08-06 | **Tesla/SpaceX "Terafab" semiconductor site, Grimes County TX, $16.8 B first phase.** Adjacent to this chip's story (Tesla fab strategy) but not an AI5/AI4 spec source; flagged for the landscape scan |

## Still absent as of 2026-08-08

- Any Tesla-published datasheet, whitepaper or developer documentation for any FSD generation
- Any AI5 peak TOPS/FLOPS, dtype list, TDP, die area, memory capacity or memory bandwidth
- Any tape-out date for either AI5 implementation
- Any TSMC node for AI5, or any Samsung node derivative (SF2 / SF2P / SF2T)
- Any AI4.5 specification of any kind
- Any MLPerf result, SDK, ISA document, or conference talk
