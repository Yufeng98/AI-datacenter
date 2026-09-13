# Untether AI — Summary

*as_of: 2026-08-08*

**Device Class:** At-Memory Compute (Canada)

---

## Status — DEAD (bankrupt, 2025-10-15)

> **Untether AI Corporation (Toronto, Ontario) filed an assignment in bankruptcy on 2025-10-15** under Canada's
> Bankruptcy and Insolvency Act (Ontario Estate/Court No. 31-3285414), with **PricewaterhouseCoopers Inc., LIT**
> as licensed insolvency trustee (Goodmans LLP as independent counsel). This is a **liquidation, not a
> restructuring**. The company had **zero employees** on the date of bankruptcy; 52 staff had been terminated
> beforehand. The statement of affairs shows **~CAD $25.0M in assets** (almost entirely cash) against
> **~CAD $128.63M owed to ~71 unsecured creditors**, on an FY2024 loss of ~CAD $68.38M. The trustee attributes
> the failure to an inability to raise further funding, competitive pressure, and *"a late pivot to generative
> AI markets resulting in lost opportunities."*
>
> **Correction to the widely repeated "AMD acqui-hired the team" framing.** Per the trustee's report, the
> June 2025 "AMD Transaction" gave AMD only the **exclusive right to negotiate new employment with certain
> Untether employees**, in exchange for a payment of **US$25M to Untether**. **AMD acquired no IP, no products,
> and no silicon.** The proceeds were used to repay Untether's secured debt in full in June 2025 — corroborated
> by National Bank of Canada's release of its patent security interest, recorded at USPTO on 2025-06-17. AMD is
> not a successor to the speedAI architecture.
>
> **Products and SDK:** nothing survived commercially. The **speedAI** and **runAI** silicon and the **imAIgine
> SDK** were discontinued with no successor and were **never open-sourced** — `github.com/untetherai` contains
> only forks of third-party tools (onnx-simplifier, YOLOX, turnkeyml, onnx2torch2), last activity May 2025, with
> no compiler or SDK source. Existing customers lost product support. `untether.ai` no longer serves (TLS
> handshake failure, checked 2026-08-08).
>
> **IP:** the ~34-patent portfolio **remains registered to Untether AI Corporation inside the bankrupt estate**.
> No reassignment is recorded at USPTO as of 2026-08-08, and prosecution continued through the bankruptcy
> (US12591633B2, *"Computational memory"*, granted 2026-03-31). The trustee is paying IP counsel to maintain
> renewals, carries intangibles at **CAD $1** realizable on the statement of affairs, and is *"considering
> options and alternatives to potentially monetize the IP"* after reviewing a non-binding letter of intent from
> the pre-bankruptcy TD Securities marketing process. No trustee filings have been posted since 2025-10-31, so
> **no IP sale is publicly confirmed**. The trustee expects some distribution to unsecured creditors.
>
> **Primary source:** [Trustee's Report to the First Meeting of Creditors on Preliminary Administration,
> PwC Inc., LIT, dated 2025-10-30](https://www.pwc.com/ca/en/car/untether/assets/untether-004_311025.pdf)
> (Estate No. 31-3285414). Insolvency file index:
> [pwc.com/ca — UNTETHER AI CORP.](https://www.pwc.com/ca/en/services/insolvency-assignments/untetherai.html)

The architecture documented below is retained as a historical data point. All of it describes a product line
that no longer exists; every performance figure is a past, vendor-reported or MLPerf-submitted result, not a
current offering.

---

## What Made It Unique

Untether AI embedded 512 processing elements directly inside each SRAM bank —
so arithmetic happened where weights and activations were stored.
This "at-memory compute" approach targeted the memory wall that costs GPUs ~90% of their inference energy.

## Products (all discontinued)

| Product | Node | SRAM | Peak Perf | Efficiency | Form |
|---|---|---|---|---|---|
| runAI200 (2020) | 16nm | 200 MB | 502 TOPS / 2 POPS | 8 TOPS/W | PCIe |
| speedAI240 (2024) | 7nm | 238 MB | 2 PFLOPS FP8 | 30 TFLOPS/W | PCIe 75W |

Peak-performance and efficiency figures are **vendor-claimed**, from Untether's product materials and its
Hot Chips 2022 disclosure. Neither part is purchasable.

## Software (discontinued, never open-sourced)

- **imAIgine SDK** — PyTorch/TF/ONNX → quantize → compile → deploy
- **Generative Compiler (v25.04, Mar 2025)** — auto-generated bank kernels, claimed 300+ models
- **HPC Bare-Metal Flow** — RISC-V PE-level programming (analogous to CUDA/PTX for GPUs)

v25.04 was the final release — three months before the AMD transaction and seven months before bankruptcy.
No SDK, compiler, or runtime source was ever published.

## MLPerf

In MLPerf Inference submissions (ResNet-50 Server), speedAI240 Slim reported ~3× better power efficiency than
the nearest competitor, and 60–65% of H200 throughput at ~20% of the power. These were MLCommons-submitted
results for a part that is no longer available; Untether never submitted BERT or any LLM benchmark.

## Why It Died — and Why That Matters Analytically

The trustee's stated causes make Untether a case study in workload-market mismatch rather than architectural
failure. The at-memory design was optimized for small-activation CNN inference, where 238 MB of on-chip SRAM
could hold an entire model. Generative workloads inverted that assumption: weights no longer fit on-chip, and
the LPDDR5 fallback path (up to 32 GB, at a bandwidth far below the ~1 PB/s aggregate in-bank figure) removed
the architecture's whole advantage. The trustee's phrase — *"a late pivot to generative AI markets resulting in
lost opportunities"* — names exactly this. The silicon worked and benchmarked well; the benchmark it won had
stopped being the market.

## Legacy

- **AMD:** obtained, for US$25M paid to Untether, the exclusive right to negotiate new employment with certain
  Untether employees (June 2025). How many were ultimately hired is **not disclosed** in the trustee's report.
  No Untether technology, IP, product, or silicon transferred to AMD.
- **IP:** ~34 US patents remain in the bankrupt estate, unsold and unassigned as of 2026-08-08.
- **Open source:** none. The imAIgine compiler and runtime are absent from the public record.
- **Customers:** lost product support on discontinuation in June 2025.
