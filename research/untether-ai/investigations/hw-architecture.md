# Untether AI — Hardware Architecture Investigation

*as_of: 2026-08-08*

**Chip Family:** runAI200 (Gen 1) / speedAI240 (Gen 2)
**Device Class:** At-Memory Compute
**HQ:** Toronto, Ontario, Canada (former)

---

## Status — DEAD (bankrupt, 2025-10-15)

Untether AI Corporation filed an **assignment in bankruptcy on 2025-10-15** under Canada's Bankruptcy and
Insolvency Act — Ontario Estate/Court No. 31-3285414, **PricewaterhouseCoopers Inc., LIT** as trustee, Goodmans
LLP as independent counsel. Liquidation, not restructuring. **Zero employees** at the date of bankruptcy (52
staff terminated beforehand). Statement of affairs: **~CAD $25.0M assets** (almost entirely cash) against
**~CAD $128.63M** owed to **~71 unsecured creditors**; FY2024 loss ~CAD $68.38M. Trustee-stated causes: failure
to raise additional funding, competitive pressure, and *"a late pivot to generative AI markets resulting in lost
opportunities."* The trustee expects some distribution to unsecured creditors.

**CORRECTION — the "AMD acqui-hired the team" framing in earlier revisions of this file is wrong.** Per the
trustee's report, the June 2025 **"AMD Transaction"** gave AMD only the **exclusive right to negotiate new
employment with certain Untether employees**, and paid **US$25M to Untether**. **AMD acquired no IP, no
products, and no silicon.** Proceeds repaid Untether's secured debt in full in June 2025, corroborated by
National Bank of Canada's release of its patent security interest recorded at USPTO on **2025-06-17**.

**IP:** the ~34-patent portfolio remains registered to **Untether AI Corporation inside the bankrupt estate**.
No USPTO reassignment is recorded as of 2026-08-08, and prosecution continued through bankruptcy
(US12591633B2 *"Computational memory"*, granted 2026-03-31). The trustee pays IP counsel to maintain renewals,
carries intangibles at **CAD $1** realizable, and is *"considering options and alternatives to potentially
monetize the IP"* after a non-binding LOI from the pre-bankruptcy TD Securities marketing process. No trustee
filings have been posted since 2025-10-31 — **no IP sale is publicly confirmed**.

`untether.ai` fails TLS handshake as of 2026-08-08; the company web presence is gone.

Everything below documents discontinued hardware, retained as a historical architectural data point.

---

## Core Concept: At-Memory Compute

Untether AI's foundational insight: ~90% of AI inference energy is consumed by data movement
(weights + activations shuttled between off-chip DRAM, L2/L3 cache, and compute units).
Their solution was to place processing elements *inside* each SRAM bank, targeting the memory wall.

All performance and efficiency numbers in this file are **vendor-claimed** peaks from Untether product
materials and the Hot Chips 2022 disclosure, except where explicitly attributed to MLPerf submissions.

---

## Generation 1 — runAI200

| Parameter | Value |
|---|---|
| Process Node | TSMC 16nm (N16) |
| Memory Banks | 511 per chip |
| SRAM per Bank | 385 KB |
| On-Chip SRAM Total | ~200 MB |
| Processing Elements per Bank | 512 (2D array) |
| Peak Throughput (Sport mode) | 502 TOPS (INT8) |
| Peak Throughput (Eco mode) | ~8 TOPS/W |
| Peak System Throughput | 2 POPS (2× cards in PCIe form factor) |
| Form Factor | PCIe card |

Architecture: each bank was an autonomous compute unit — 512 PEs directly attached to 385 KB SRAM.
No data ever left the bank for arithmetic.

---

## Generation 2 — speedAI240 (Codename: Boqueria)

Presented at Hot Chips 2022. Manufactured at **TSMC 7nm**. Discontinued June 2025.

| Parameter | Value |
|---|---|
| Process Node | TSMC 7nm |
| Memory Banks | 729 per chip |
| SRAM per Bank | ~327 KB |
| On-Chip SRAM Total | 238 MB |
| RISC-V Cores per Bank | 2 × 1.35 GHz custom RISC-V |
| Total RISC-V Cores | 1,456 (datasheet) / 1,435 (product brief) |
| Processing Elements per Bank | 512 (at-memory PEs) |
| Peak Throughput (FP8) | 2 PetaFLOPS |
| Peak Throughput (BF16) | 1 PetaFLOP |
| Energy Efficiency | 30 TFLOPS/W |
| L1 SRAM Bandwidth | ~1 PB/s aggregate |
| Scratchpads | 4 × 1 MB on-chip |
| External DRAM | 2 × 64-bit LPDDR5 ports, up to 32 GB |
| Form Factor | Low-profile PCIe, 75 W TDP (speedAI240 Slim) |
| Datatypes | INT4, INT8, FP8, BF16 |
| Sparsity | 2:1 structured sparsity + zero-detect circuitry |

### Memory Hierarchy

```
L1: 238 MB distributed SRAM (512 PEs × 729 banks — primary compute & storage)
  └── ~1 PB/s aggregate bandwidth (zero-transport: PEs read their own SRAM)
L2: 4 × 1 MB on-chip scratchpads (inter-bank staging)
L3: Up to 32 GB LPDDR5 (model parameter overflow, activations)
```

### RISC-V Augmentation (Gen 2 vs Gen 1)

Gen 2 added two RISC-V scalar cores per bank to handle (per Hot Chips 2022 disclosure):
- Control flow (branches, loops in RNN / Transformer layers)
- Non-linear activations (GELU, SiLU, Softmax) not suited to PE arrays
- Custom kernel dispatch

---

## MLPerf Inference Benchmark Results

| Benchmark | Metric | Untether speedAI240 | NVIDIA H200 | Efficiency Ratio |
|---|---|---|---|---|
| ResNet-50 Server | Queries/sec @ 986 W | 309,752 | ~similar throughput higher power | ~3× more efficient |
| ResNet-50 (vs L40s) | Throughput | 60–65% of H200 | 100% | at ~20% of power |
| Power Draw | TDP | 150 W (Slim card) | 350 W (L40s) | 2.3× |

Note: Untether only submitted ResNet-50 image classification in first MLPerf rounds;
did not submit BERT or LLM benchmarks before shutdown.

---

## Competitive Position

(Position as of the product's active life, 2024–mid-2025.)

- **vs NVIDIA GPU:** 3× better power efficiency on image classification; absolute throughput lower
- **vs Groq:** both avoided HBM; Groq streams from SRAM, Untether embedded PEs in SRAM
- **vs D-Matrix:** both at/near-memory; D-Matrix uses analog CIM, Untether used digital SRAM PEs
- **vs Cerebras:** different scale — Cerebras uses wafer-scale SRAM; Untether used chiplet-scale banks

**Post-mortem.** The trustee's stated cause — *"a late pivot to generative AI markets resulting in lost
opportunities"* — matches the architecture's failure mode rather than any implementation defect. The at-memory
design assumed the working set fits in on-chip SRAM (238 MB), which held for CNN inference and collapsed for
LLM weights; the LPDDR5 fallback (≤32 GB) sat far below the ~1 PB/s in-bank bandwidth that constituted the
entire advantage. Untether won the benchmark it targeted (ResNet-50 efficiency) as that benchmark ceased to
represent the market.

---

## Company Timeline

| Year | Event |
|---|---|
| 2018 | Founded in Toronto by Ran Chaudhuri, Martin Snelgrove |
| 2020 | runAI200 announced — "PetaOps Era" press release |
| 2022 | speedAI240 (Boqueria) unveiled at Hot Chips; TSMC 7nm |
| 2023 | imAIgine SDK v1 — bare-metal kernel programming support |
| 2024 | speedAI240 Slim shipped; MLPerf submission; imAIgine SDK early access |
| Mar 2025 | imAIgine SDK v25.04 announced — generative compiler, 300+ models (final release) |
| Jun 2025 | **"AMD Transaction"**: AMD paid US$25M for the exclusive right to negotiate new employment with certain Untether employees. **No IP, products, or silicon acquired.** Proceeds repaid secured debt in full |
| Jun 2025 | speedAI products and imAIgine SDK discontinued; customers lost support |
| 2025-06-17 | National Bank of Canada patent security interest released (USPTO record) |
| 2025-10-15 | **Assignment in bankruptcy** filed under the BIA; Ontario Estate No. 31-3285414; PwC Inc. trustee; zero employees |
| 2025-10-31 | First meeting of creditors; trustee's preliminary report filed (latest public filing) |
| 2026-03-31 | US12591633B2 "Computational memory" granted — estate still owns and prosecutes the portfolio |
| 2026-08-08 | No USPTO reassignment recorded; no IP sale publicly confirmed; untether.ai offline |

---

## Sources

### Corporate status (primary)

- [PwC Inc. — Trustee's Report to the First Meeting of Creditors on Preliminary Administration, 2025-10-30](https://www.pwc.com/ca/en/car/untether/assets/untether-004_311025.pdf) — **PRIMARY.** Signed Tyler Ray, LIT, VP, PwC Inc.; Estate No. 31-3285414. Source for AMD Transaction terms, CAD $128.63M / 71 creditors, FY2024 loss CAD $68.38M, zero employees, IP retained by estate, TD Securities LOI, intangibles at $1, secured debt repaid June 2025.
- [PwC Canada insolvency file — UNTETHER AI CORP.](https://www.pwc.com/ca/en/services/insolvency-assignments/untetherai.html) — assignment 2025-10-15; trustee appointment; first meeting 2025-10-31.
- [PwC trustee's reports index](https://www.pwc.com/ca/en/services/insolvency-assignments/untetherai/trustees-reports.html) — only the 2025-10-31 report posted; confirms no publicly filed IP sale since.
- [BetaKit — Untether AI files for bankruptcy](https://betakit.com/untether-ai-files-for-bankruptcy-following-amd-acquihire/) — independent secondary; $103.6M deficiency; creditors incl. Middlefield Ventures $33M, Radical Ventures $28.5M, GM Ventures $18.3M, CPPIB $14.8M.
- [US11614947B2](https://patents.google.com/patent/US11614947B2/en) — Untether AI Corporation still current assignee; NBC security interest released 2025-06-17; no reassignment.
- [US12591633B2](https://patents.google.com/patent/US12591633B2/en) — granted 2026-03-31, still assigned to Untether AI Corporation.
- Negative evidence: `untether.ai` and `www.untether.ai` fail TLS handshake (curl error 35) as of 2026-08-08.

### Architecture (historical)

- [Untether AI Products](https://www.untether.ai/products/)
- [At-Memory Architecture](https://www.untether.ai/products/technology/)
- [Hot Chips 2022 announcement](https://www.untether.ai/untether-ai-unveils-its-second-generation-at-memory-compute-architecture-at-hot-chips-2022/)
- [speedAI240 IEEE paper](https://ieeexplore.ieee.org/document/10066167/)
- [speedAI240 Slim ships](https://www.untether.ai/untether-ai-ships-speedai240-slim-worlds-fastest-most-energy-efficient-ai-inference-accelerator-for-cloud-to-edge-applications/)
- [MLPerf results](https://www.untether.ai/mlperf-results/)
- [MLPerf 4.1 analysis — XPU.pub](https://xpu.pub/2024/09/20/mlperf-4-1/)
- [Tom's Hardware — "AMD scoops entire Untether AI chip team"](https://www.tomshardware.com/tech-industry/amd-scoops-entire-untether-ai-chip-team-canada-ai-inference-outfit-will-cease-product-support) — **SECONDARY, framing superseded.** Correctly reports the end of product support, but the "acquisition/acqui-hire of the team" characterization is contradicted by the trustee's report, which describes only exclusive employment-negotiation rights for US$25M.
- [EE Times — "Untether AI shuts down, engineering team joins AMD"](https://www.eetimes.com/untether-ai-shuts-down-engineering-team-joins-amd/) — **SECONDARY, framing superseded.** Same correction as above.

Note: all `untether.ai` links above are dead as of 2026-08-08 (TLS handshake failure); they are retained for
provenance only and are accessible, if at all, through web archives.
