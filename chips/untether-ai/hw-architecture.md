# Untether AI — Hardware Architecture

*as_of: 2026-08-08*

**Device Class:** At-Memory Compute (Canada)
**HQ:** Toronto, Ontario, Canada (former)

---

## Status — DEAD (bankrupt, 2025-10-15)

> Untether AI Corporation filed an **assignment in bankruptcy on 2025-10-15** under Canada's Bankruptcy and
> Insolvency Act (Ontario Estate No. 31-3285414; trustee PricewaterhouseCoopers Inc., LIT). Zero employees at
> the date of bankruptcy. All silicon (runAI200, speedAI240/Slim) and the imAIgine SDK were discontinued in
> June 2025 with no successor. The ~34-patent portfolio remains registered to Untether AI Corporation inside the
> estate — no USPTO reassignment as of 2026-08-08.
>
> The June 2025 **"AMD Transaction" was not an IP or asset acquisition**: per the trustee's report it conveyed
> only the exclusive right to negotiate new employment with certain Untether employees, for US$25M paid to
> Untether (used to repay secured debt in full). **AMD acquired no IP, no products, and no silicon**, and is not
> a successor to this architecture.
>
> Primary source: [PwC Inc. Trustee's Report to the First Meeting of Creditors, 2025-10-30](https://www.pwc.com/ca/en/car/untether/assets/untether-004_311025.pdf).
>
> Everything below describes discontinued hardware and is retained as a historical architectural data point.

---

## Key Innovation

Untether AI placed 512 processing elements *inside* each SRAM bank.
Arithmetic happened where data lived — no movement across a memory bus.
This targeted the data-movement bottleneck that consumes ~90% of GPU inference energy.

---

## Chip Generations

### Gen 1: runAI200 (2020, TSMC 16nm)

- 511 SRAM banks × 385 KB × 512 PEs = ~200 MB SRAM
- 502 TOPS INT8 (sport) / 8 TOPS/W (eco)
- 2 POPS peak (two PCIe cards)

All figures below are **vendor-claimed** peak numbers from Untether product materials and the Hot Chips 2022
disclosure, unless noted otherwise.

### Gen 2: speedAI240 / speedAI240 Slim (2022-2024, TSMC 7nm)

- 729 SRAM banks × ~327 KB × 512 PEs + 2 RISC-V cores = 238 MB SRAM
- 1,456 custom RISC-V cores @ 1.35 GHz
- **2 PetaFLOPS FP8 / 1 PetaFLOP BF16**
- **30 TFLOPS/W** (industry-leading efficiency)
- ~1 PB/s aggregate SRAM bandwidth
- 4 × 1 MB scratchpads + up to 32 GB LPDDR5
- 75 W TDP (Slim); PCIe low-profile
- Datatypes: INT4, INT8, FP8, BF16; 2:1 structured sparsity

---

## Memory Hierarchy

| Level | Size | Bandwidth | Notes |
|---|---|---|---|
| L1 SRAM (in-bank) | 238 MB | ~1 PB/s | PEs accessed directly |
| L2 Scratchpad | 4 MB | not disclosed | Inter-bank staging |
| L3 LPDDR5 | up to 32 GB | not disclosed | Overflow / parameters |

---

## MLPerf Highlights (historical submissions)

- ResNet-50 Server: 309,752 QPS at 986 W — **~3× more power-efficient** than nearest competitor
- vs NVIDIA L40s: 60–65% throughput at ~20% power (~2.3× lower TDP)

No BERT or LLM benchmark was ever submitted.

---

## Company Timeline

| Date | Milestone |
|---|---|
| 2018 | Founded, Toronto |
| 2020 | runAI200 — "PetaOps Era" |
| 2022 | speedAI240 at Hot Chips |
| 2024 | speedAI240 Slim shipped; MLPerf results |
| 2025-03 | imAIgine SDK v25.04 generative compiler (final release) |
| 2025-06 | "AMD Transaction": AMD paid US$25M for exclusive rights to negotiate employment with certain staff; no IP/products/silicon transferred. speedAI and imAIgine discontinued; secured debt repaid in full |
| 2025-06-17 | National Bank of Canada patent security interest released (USPTO record) |
| 2025-10-15 | Assignment in bankruptcy filed (BIA, Ontario Estate No. 31-3285414; PwC Inc. trustee); zero employees |
| 2025-10-31 | First meeting of creditors; ~CAD $25.0M assets vs ~CAD $128.63M unsecured claims (~71 creditors) |
| 2026-03-31 | US12591633B2 "Computational memory" granted — portfolio still prosecuted and owned by the estate |
| 2026-08-08 | No USPTO reassignment of the ~34-patent portfolio; no IP sale publicly confirmed |
