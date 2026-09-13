# SK Hynix AiM / AiMX — Summary

*chip: sk-hynix-aim*
*device_class: Processing-in-Memory (PIM)*
*as_of: 2026-08-08*
*research_baseline: 2026-04-05*

---

## One-Paragraph Summary

SK Hynix **AiM (Accelerator-in-Memory)** is a **Processing-in-Memory** product family built on standard GDDR6 DRAM. AiM embeds **FP16 SIMD MAC units** at the bank boundary of each pseudo-channel inside the DRAM die, enabling all 16 banks to operate simultaneously (all-bank operation) and exploiting the **internal DRAM bandwidth (~4 TB/s)** rather than the limited external I/O bandwidth (~64 GB/s). The **AiMX** accelerator card combines 16–32 GDDR6-AiM packages and acts as a **heterogeneous co-processor** alongside a CPU or GPU: the host handles compute-bound ops while AiMX handles memory-bound GEMV (LLM KV-cache attention, embedding lookups). Presented at Hot Chips 34 (2022), ASPLOS 2023, and Hot Chips 2024 (AiMX-xPU), SK Hynix demonstrated 10× lower latency and 5× lower power vs. GPU for LLM attention. The software stack uses an extended DRAM command set (AiM ISR), a host-side PIM runtime, and a BLAS Level 2 library, with transparent PyTorch/TensorFlow integration and — as of the 2025 conference demos — **vLLM** as the serving frontend; no public SDK has been released. **As of 2026-08-08 AiMX is still described by SK Hynix itself as a card *prototype*** (CES 2026 wording): not sampling, not shipping, and with no named customer or deployment. See "Update — 2026-08-08" below.

---

## Key Specifications

| Property | Value |
|----------|-------|
| Paradigm | Processing-in-Memory (PIM) in GDDR6 |
| Product | GDDR6-AiM / AiMX |
| MAC units / GDDR6-AiM package | 2 (one per pseudo-channel) |
| FP16 SIMD lanes / MAC unit | 16 |
| Total FP16 lanes / package | 32 |
| Internal DRAM bandwidth | ~4 TB/s |
| External I/O bandwidth | ~64 GB/s |
| AiMX Gen1 capacity | 16 GB (16 packages) |
| AiMX Gen2 capacity (2024) | 32 GB (32 packages) |
| Numeric formats | FP16 |
| GDDR6 speed grade | 16 Gbps/pin |
| JEDEC compatible | Yes (backward compatible) |
| Speedup vs. CPU+DRAM | up to 16× |
| Power reduction | up to 80% |
| Latest AiMX generation (as of 2026-08-08) | AiMX Gen2 (2024) — no Gen3 announced |
| Productization status (as of 2026-08-08) | **Prototype** — SK Hynix's own CES 2026 wording ("accelerator card prototype"); not sampling, not shipping, no named customer |

---

## Positioning

- **Target**: LLM inference co-acceleration (attention GEMV, embedding lookup)
- **Not standalone**: Paired with CPU or GPU; handles memory-bound ops only
- **Heterogeneous model**: Host handles compute-bound; AiMX handles memory-bound
- **JEDEC compatible**: Drop-in GDDR6 channel; no memory controller changes

---

## Software Ecosystem

| Layer | Tool | Status |
|-------|------|--------|
| Serving framework | vLLM (demo integration) | demonstrated publicly, not released |
| Framework | PyTorch, TensorFlow | not public (transparent offload path described) |
| BLAS library | AiM BLAS (GEMV) | not public |
| Runtime | AiM Runtime Library | not public |
| HW command interface | AiM ISR (extended DRAM cmds) | partially in papers |
| Research simulator | aim_simulator (Ramulator 2.0) | Public (GitHub) |

---

## Update — 2026-08-08 (window 2026-04-05 → 2026-08-08)

*Prior-generation content above is retained unchanged. This section records what the 2026-08-08 landscape scan verified, including two pre-baseline facts the earlier entry missed.*

### In-window developments: none

Nothing material changed for AiM/AiMX between the 2026-04-05 research baseline and 2026-08-08. The following negatives were independently verified:

- **Hot Chips 38 (Aug 23–25, 2026, Stanford)** — a **future** event as of this update; its program is public but no slides, abstracts, or specs exist, so nothing on the program is evidence of any specification. The published program lists **no SK Hynix AiM/AiMX talk**. SK Hynix appears only in Tutorial 1 (Memory Technology, Sun Aug 23) with an HBM base-die talk — the exact tutorial title and speaker are **not confirmed** (two independent reads of hotchips.org disagreed) and are therefore not recorded here. The PIM content in the Tuesday Memory session belongs to Samsung (LPDDR5X-PIM) and to the XCENA/Samsung MX1 CXL computational-memory device, not to SK Hynix.
- **FMS 2026 (Aug 4–6, 2026)** — SK Hynix's FMS messaging covered HBF (High Bandwidth Flash, co-developed with Sandisk), HBM4/HBM4E, SOCAMM2, DDR5 RDIMMs, QLC eSSD, and CXL (CMM-Ax, CMM-Hybrid, Pooled Memory). **AiM/AiMX was not mentioned.**
- **ISSCC 2026 (Feb 2026)** — SK Hynix's papers were LPDDR6 (16 Gb, ~14.4 Gbps/pin) and GDDR7. **No SK Hynix PIM/AiM paper.**
- **Research simulator** `github.com/arkhadem/aim_simulator` (Ramulator 2.0-based, ~70 stars, 66 commits) — last commit **2025-07-22** ("LPDDR5 debugged and tested"). No activity in the window.
- The only in-window SK Hynix item touching AiM is an **educational blog post**, "[Tech Note EP.1] It's the Memory — The Future of AI Will Be Decided by Memory" (news.skhynix.com, **2026-05-12**), which restates the GDDR6-AiM bottleneck-absorption argument. **No specs, no roadmap, no product news.**

**Not confirmed — treat as nonexistent:** AiM Gen3, GDDR7-AiM, HBM-based AiM, AiM inside an HBM4 custom base die, any MLPerf submission using AiMX, any public AiM SDK release, and any cancellation or discontinuation announcement.

### Backfill 1 — status is still "prototype" (CES 2026)

In its CES 2026 press release (**2026-01-05**; show Jan 6–9) SK Hynix described AiMX verbatim as "SK hynix's accelerator card **prototype** featuring a GDDR6-AiM chip which is specialized for large language models (LLMs)." The status has **not** advanced past prototype: not sampling, not shipping, not deployed at scale.

Alongside AiMX, SK Hynix showed three adjacent compute-memory concepts this survey does not otherwise track:

| Concept | What SK Hynix says it is | Relation to AiM |
|---|---|---|
| **CuD** (Compute-using-DRAM) | Simple computation performed within the memory cells themselves | Different, more primitive PIM point than AiM's bank-boundary MAC units |
| **CMM-Ax** (CXL Memory Module – Accelerator) | CXL memory expansion module with added compute | CXL-attached, not GDDR6-resident |
| **cHBM** (custom HBM) | HBM whose base die absorbs GPU/ASIC functions | Base-die logic, not in-DRAM MACs |

The headline CES 2026 memory product was a **16-layer 48 GB HBM4**. A "1.5 TB/s" bandwidth figure circulating alongside it could not be confirmed against any primary source and is deliberately **not recorded**.

### Backfill 2 — vLLM is a real, public part of the AiMX software story

At the **AI Infra Summit 2025** (SK Hynix post **2025-10-01**; summit held ~Sept 2025) and again at the **2025 OCP Global Summit** (post **2025-10-31**), SK Hynix ran a live AiMX demo in a **Supermicro GPU SuperServer SYS-421GE-TNRT3 with 2× NVIDIA H100 and 4× AiMX cards**, serving LLM/reasoning workloads under the **vLLM framework** and claiming more concurrent requests and longer input prompts than a GPU-only configuration. **No numeric performance figures were disclosed.** The Software Ecosystem table above has gained a `Serving framework | vLLM (demo integration) | demonstrated publicly, not released` row accordingly.

### Claim rejected on verification — do not record AiM as "deployed"

TrendForce (2026-03-10) writes that "SK hynix has already deployed its PIM-based accelerator, AiM, in real-world applications." This is a single secondary source summarizing a Korean outlet; it names no customer, date, or volume, and it directly contradicts SK Hynix's own CES 2026 "prototype" wording. Independent searching surfaced no named customer, no shipment, and no deployment evidence. **AiMX's status verb stays at "prototype / publicly demonstrated."**

### Direction of travel

SK Hynix's 2026 compute-memory messaging has shifted emphasis away from AiM and toward CXL (CMM-Ax, CMM-Hybrid, Pooled Memory), cHBM, and HBF. On the public record AiM/AiMX is in **maintenance and positioning mode rather than productization**.

*Sources for this section: SK Hynix CES 2026 press release (PRNewswire, 2026-01-05); news.skhynix.com AI Infra Summit 2025 (2025-10-01), OCP Global Summit 2025 (2025-10-31), Tech Note EP.1 (2026-05-12), FMS 2026 recap (2026-08-07); hotchips.org Hot Chips 38 program (fetched 2026-08-08); github.com/arkhadem/aim_simulator commit history; TechPowerUp #341520; TrendForce 2026-03-10 (rejected).*

---

## References

- [HC34 AiM Poster — Hot Chips 34](https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf)
- [ISCA 2021 — Hardware Architecture and Software Stack for PIM](https://ieeexplore.ieee.org/document/9499894)
- [ASPLOS 2023 PIM Tutorial — SK Hynix AiM](https://events.safari.ethz.ch/asplos-pim-tutorial/lib/exe/fetch.php?media=system_architecture_and_software_stack_for_gddr6-aim_sk_hynix_pim_tutorial_asplos23_sent.pdf)
- [Hot Chips 2024 — AiMX-xPU IEEE](https://ieeexplore.ieee.org/document/10664793/)
- [aim_simulator — GitHub](https://github.com/arkhadem/aim_simulator)

*Added 2026-08-08:*

- [SK hynix Showcases Next-Generation AI Memory Innovation at CES 2026 — PRNewswire, 2026-01-05](https://www.prnewswire.com/news-releases/sk-hynix-showcases-next-generation-ai-memory-innovation-at-ces-2026-302653161.html)
- [SK hynix at AI Infra Summit 2025 (AiMX + vLLM on Supermicro 2×H100 + 4×AiMX) — newsroom, 2025-10-01](https://news.skhynix.com/en/ai-infra-summit-2025/)
- [SK hynix Showcases Full-Stack AI Memory Portfolio at 2025 OCP Global Summit — newsroom, 2025-10-31](https://news.skhynix.com/en/sk-hynix-showcases-full-stack-ai-memory-portfolio-at-2025-ocp-global-summit/)
- [Tech Note EP.1 — It's the Memory: The Future of AI Will Be Decided by Memory — newsroom, 2026-05-12](https://news.skhynix.com/en/tech-note-series-ep1/)
- [SK hynix at FMS 2026 (no AiM/AiMX content) — newsroom](https://news.skhynix.com/en/fms-2026/)
- [Hot Chips 38 program (Aug 23–25, 2026 — future event; no SK Hynix AiM talk)](https://www.hotchips.org/)
