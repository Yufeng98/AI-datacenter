# SK Hynix AiM / AiMX — Hardware Architecture Investigation

*chip: sk-hynix-aim*
*investigation: hw-architecture*
*date: 2026-04-05*
*sources: Hot Chips 34 poster, IEEE ISCA 2021, ASPLOS 2023 tutorial, Hot Chips 2024 coverage, Tom's Hardware, SK Hynix newsroom*

---

## 1. Architecture Paradigm

SK Hynix AiM is **Processing-in-Memory (PIM)** technology embedded inside standard GDDR6 DRAM. Rather than building a discrete compute accelerator, AiM integrates **FP16 SIMD MAC units** at every DRAM bank boundary within the DRAM die itself. This eliminates weight transfer between memory and a separate compute die, exploiting the massive **internal DRAM bandwidth** (4–8× the external I/O bandwidth).

AiM targets **matrix-vector multiplication** (GEMV), the dominant operation in LLM decoding (KV-cache attention, embedding lookups).

---

## 2. Key Design Principles

1. **All-bank operation**: All 16 banks operate in parallel simultaneously (vs. standard DRAM one-bank-at-a-time). This unleashes the full internal DRAM bandwidth.
2. **Extended DRAM command set**: New AiM-specific commands (issued by host via AiM ISR — Instruction Set Register) trigger in-DRAM compute without modifying JEDEC bus signaling
3. **MAC at bank boundary**: Each pseudo-channel has its own MAC unit positioned between even/odd bank pairs

---

## 3. GDDR6-AiM Device Architecture

### 3.1 DRAM Bank Organization
A GDDR6-AiM chip has:
- 16 banks (standard GDDR6 organization)
- 2 pseudo-channels (PC0, PC1)
- 8 banks per pseudo-channel
- **1 MAC unit per pseudo-channel** = 2 MAC units per GDDR6-AiM package

### 3.2 MAC Unit
- **16-lane FP16 SIMD array** per pseudo-channel
- Each lane: 1× FP16 multiplier + 1× FP16 adder
- All lanes execute in lock-step (SIMD)
- Total per package: 2 × 16-lane = 32 FP16 MAC units
- Accumulator register files per lane (3 register files)

### 3.3 Internal vs External Bandwidth
| Path | Bandwidth | Notes |
|------|-----------|-------|
| External I/O (GDDR6 pins) | ~64 GB/s | Standard GDDR6 host transfer |
| Internal DRAM bandwidth | ~4 TB/s | All-bank parallel array access |
| AiM compute bandwidth utilization | up to ~4 TB/s | MAC units consume internal BW |

The internal bandwidth advantage (~64×) is the core efficiency gain of AiM.

---

## 4. AiMX Accelerator Card

AiMX is SK Hynix's multi-chip GDDR6-AiM accelerator card:

| Generation | AiM packages | Memory capacity | Interface |
|-----------|-------------|----------------|-----------|
| AiMX Gen1 | 16 GDDR6-AiM | 16 GB | PCIe (host attached) |
| AiMX Gen2 (2024) | 32 GDDR6-AiM | 32 GB | PCIe; roadmap to 256 GB |

- Pairs with CPU or GPU as an **attention/embedding co-processor**
- Host GPU handles compute-bound operations (Attention QK softmax, output projection); AiMX handles memory-bound GEMV (K/V matmul with query vectors)
- AiMX-xPU: heterogeneous CPU+AiMX system (Hot Chips 2024) targeting data center LLM inference

---

## 5. Performance Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| Speedup vs. CPU+DRAM | up to 16× | Memory-bound GEMV |
| Power reduction | up to 80% | vs. GPU+GDDR6 baseline |
| Throughput vs GPU (LLM attention) | 10× faster | Llama 3 70B demo |
| Power vs GPU | 1/5 (20%) | AiMX 2024 demo |
| Operating voltage | 1.25 V | vs. 1.35 V standard GDDR6 |
| GDDR6 speed grade | 16 Gbps/pin | Standard GDDR6 |

---

## 6. Deployment Topology

```
CPU / Host                    GPU
  │                            │
  │ PCIe                       │ PCIe
  ▼                            ▼
  ┌────────────────────────────────┐
  │         AiMX Card              │
  │  [GDDR6-AiM × 16 or × 32]     │
  │  Each package: 2 FP16 MAC      │
  │  All-bank parallel operation   │
  └────────────────────────────────┘
```

AiMX is positioned as a **heterogeneous co-processor** — not a standalone accelerator. The host GPU/CPU handles attention score computation (QK^T, softmax) while AiMX handles the GEMV (matmul with V matrix stored in PIM DRAM).

---

## 7. Technical Notes

- **JEDEC compatibility**: GDDR6-AiM maintains backward compatibility with GDDR6 memory controllers; standard read/write still work; AiM operations added via command extension
- **No external ISA**: AiM compute is triggered by AiM ISR instructions, not a programmer-visible ISA like PTX
- **No caches**: Like all DRAM-based PIM, there is no cache hierarchy; all data lives in the DRAM array itself
- **Limited control flow**: Processors execute SIMD; no branch prediction; compute is fixed MAC + accumulate patterns

---

## 8. Re-investigation — 2026-08-08 (window 2026-04-05 → 2026-08-08)

*Investigated 2026-08-08. Verdict: **no in-window hardware development**; two pre-baseline facts backfilled. Verification note: the verifying pass exhausted its WebSearch budget and corroborated via DuckDuckGo HTML result pages plus direct page fetches; GitHub's REST API returned 403, so the simulator's last-commit date was read off the rendered commits page.*

### 8.1 What was checked and found negative

| Venue / artifact | Date | AiM/AiMX content | Confidence |
|---|---|---|---|
| **Hot Chips 38** program, Stanford | Aug 23–25, **2026** — *future event; program public, no slides/abstracts/specs exist* | **None.** No SK Hynix AiM/AiMX talk. SK Hynix appears only in Tutorial 1 (Memory Technology, Sun Aug 23) with an HBM base-die talk. PIM content in the Tuesday Memory session belongs to Samsung (LPDDR5X-PIM) and to the XCENA/Samsung MX1 CXL computational-memory device. | high (absence); the tutorial's exact **title and speaker are not confirmed** — two independent reads of hotchips.org returned different strings, so neither is recorded |
| **FMS 2026** | Aug 4–6, 2026 | **None.** SK Hynix covered HBF (with Sandisk), HBM4/HBM4E, SOCAMM2, DDR5 RDIMMs, QLC eSSD, and CXL (CMM-Ax, CMM-Hybrid, Pooled Memory). | high |
| **ISSCC 2026** | Feb 2026 | **None.** SK Hynix papers were LPDDR6 (16 Gb, ~14.4 Gbps/pin) and GDDR7. | high |
| `arkhadem/aim_simulator` (Ramulator 2.0, ~70 stars, 66 commits) | last commit **2025-07-22** ("LPDDR5 debugged and tested") | No activity in window. | high |
| news.skhynix.com site search for "AiM" | window | Single hit: educational post "[Tech Note EP.1] It's the Memory — The Future of AI Will Be Decided by Memory" (**2026-05-12**), restating the GDDR6-AiM bottleneck-absorption argument. **No specs, no roadmap, no product news.** | high |

**Explicitly not confirmed (treat as nonexistent):** AiM Gen3 · GDDR7-AiM · HBM-based AiM · AiM logic in an HBM4 custom base die · any MLPerf submission using AiMX · any public AiM SDK release · any cancellation or discontinuation announcement.

### 8.2 Backfill — productization status is still "prototype"

SK Hynix's CES 2026 press release (**2026-01-05**; show Jan 6–9) describes AiMX verbatim as "SK hynix's accelerator card **prototype** featuring a GDDR6-AiM chip which is specialized for large language models (LLMs)." Status has not advanced past prototype: **not sampling, not shipping, not deployed at scale, no named customer**.

**Rejected claim.** TrendForce (2026-03-10) states "SK hynix has already deployed its PIM-based accelerator, AiM, in real-world applications." Single secondary source summarizing a Korean outlet (Global Economic News); names no customer, date, or volume; contradicts SK Hynix's own "prototype" wording; no independent corroboration found. **Do not upgrade the status verb.** Confidence in the rejection: high.

### 8.3 Backfill — first public multi-card AiMX system topology

At the **AI Infra Summit 2025** (SK Hynix post **2025-10-01**; summit held ~Sept 2025 — the post date is the citable one) and again at the **2025 OCP Global Summit** (post **2025-10-31**), SK Hynix ran a live AiMX demo in a **Supermicro GPU SuperServer SYS-421GE-TNRT3 with 2 × NVIDIA H100 and 4 × AiMX cards**, described as optimizing LLM attention / memory-bound workloads. This is the first publicly described multi-card AiMX configuration. **No capacity, bandwidth, or throughput figures were disclosed**; cards attach over standard PCIe with **no proprietary card-to-card link** mentioned. Independently corroborated by TechPowerUp #341520.

### 8.4 Adjacent SK Hynix compute-memory concepts (not AiM)

CES 2026 also showed **CuD** (Compute-using-DRAM — simple computation within the memory cells), **CMM-Ax** (CXL memory module with added compute), and **cHBM** (custom HBM integrating GPU/ASIC functions into the base die). None is AiM; recorded as context for SK Hynix's shifting compute-memory emphasis. The headline CES 2026 memory product was a **16-layer 48 GB HBM4** — the "1.5 TB/s" figure circulating with it is **not confirmed by any primary source** and is deliberately omitted.

### 8.5 Direction of travel

SK Hynix's 2026 compute-memory messaging emphasizes CXL (CMM-Ax, CMM-Hybrid, Pooled Memory), cHBM, and HBF over AiM. On the public record AiM/AiMX is in maintenance/positioning mode rather than productization. This is an interpretation, not a vendor statement — confidence: medium.

---

## Sources

*Added 2026-08-08:*

- [SK hynix Showcases Next-Generation AI Memory Innovation at CES 2026 — PRNewswire, 2026-01-05](https://www.prnewswire.com/news-releases/sk-hynix-showcases-next-generation-ai-memory-innovation-at-ces-2026-302653161.html)
- [SK hynix newsroom — CES 2026](https://news.skhynix.com/en/sk-hynix-showcases-next-generation-ai-memory-innovations-at-ces-2026/)
- [SK hynix newsroom — AI Infra Summit 2025 (2025-10-01)](https://news.skhynix.com/en/ai-infra-summit-2025/)
- [SK hynix newsroom — 2025 OCP Global Summit (2025-10-31)](https://news.skhynix.com/en/sk-hynix-showcases-full-stack-ai-memory-portfolio-at-2025-ocp-global-summit/)
- [SK hynix newsroom — Tech Note EP.1 (2026-05-12)](https://news.skhynix.com/en/tech-note-series-ep1/)
- [SK hynix newsroom — FMS 2026 recap (no AiM content)](https://news.skhynix.com/en/fms-2026/)
- [SK hynix newsroom — HBF at FMS 2026](https://news.skhynix.com/en/hbf-at-fms-2026/)
- [Hot Chips 38 program — future event, fetched 2026-08-08](https://www.hotchips.org/)
- [aim_simulator commit history — last commit 2025-07-22](https://github.com/arkhadem/aim_simulator/commits/main)
- [TechPowerUp #341520 — independent corroboration of the Supermicro 2×H100 + 4×AiMX demo](https://www.techpowerup.com/341520/)
- [All About Circuits — SK hynix LPDDR6 at ISSCC 2026](https://www.allaboutcircuits.com/news/sk-hynix-details-low-power-ddr6-sdram-isscc-2026/)
- [TrendForce, 2026-03-10 — "already deployed" claim, **rejected on verification**](https://www.trendforce.com/news/2026/03/10/news-beyond-hbm-samsung-sk-hynix-reportedly-explore-next-gen-ai-memory/)

*Original (2026-04-05):*

- [HC34 SK Hynix AiM Poster — Hot Chips 34](https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf)
- [Hardware Architecture and Software Stack for PIM — IEEE ISCA 2021](https://ieeexplore.ieee.org/document/9499894)
- [System Architecture and Software Stack for GDDR6-AiM — IEEE](https://ieeexplore.ieee.org/document/9895629/)
- [ASPLOS 2023 PIM Tutorial — SK Hynix AiM slides](https://events.safari.ethz.ch/asplos-pim-tutorial/lib/exe/fetch.php?media=system_architecture_and_software_stack_for_gddr6-aim_sk_hynix_pim_tutorial_asplos23_sent.pdf)
- [SK Hynix AiMX-xPU at Hot Chips 2024 — ServeTheHome](https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/)
- [SK Hynix AiMX 32GB upgrade — newsroom 2024](https://news.skhynix.com/sk-hynix-presents-upgraded-aimx-solution-at-ai-hw-edge-ai-summit-2024/)
