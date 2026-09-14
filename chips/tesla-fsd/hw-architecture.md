# Tesla FSD Chip — Hardware Architecture

*as_of: 2026-09-13*
*generations: HW3 / HW4 (AI4) / AI4.5 "AP45" (unannounced) / AI4.1 "AI4 Plus" (announced only) / AI5 (taped out, delayed to mid-2027)*

---

## Generation Overview

| Generation | Year | Status (2026-08-08) | Foundry / Process | NPUs | Peak INT8 TOPS | On-chip SRAM | DRAM | Board TDP |
|---|---|---|---|---|---|---|---|---|
| HW3 (FSD Chip) | 2019 | Deployed; **declared insufficient for unsupervised FSD 2026-04-22** | Samsung 14 nm | 2 | 73.7 | 32 MiB/NPU | 8 GB LPDDR4, 68 GB/s † | ~100 W |
| HW4 / AI4 | 2023 | Deployed; current shipping baseline | Samsung 7 nm | 3 | ~121.6 | 32 MiB/NPU | 16 GB GDDR6, 224 GB/s † | ~160 W |
| **AI4.5 / "AP45"** | 2025-12 → | **Shipping in some vehicles; never announced by Tesla.** Update 2026-09-13: a JPMorgan analyst note (via Electrek, 2026-08-20) attributes "~10% more compute and about twice the memory" (vs AI4) to a chip it calls "AI4.5" — **thirdhand (Tesla → JPMorgan → Electrek) and possibly a mislabeling of AI4.1/"AI4 Plus,"** whose announced spec is identically worded. Not promoted to a confirmed spec — see Update section below. | Not disclosed (assumed Samsung 7 nm) | Not disclosed (a three-SoC board is a *low-confidence* firmware inference) | Not disclosed | Not disclosed (**~+10% compute claimed 2026-08-20, unconfirmed / possibly AI4.1 conflation**) | Not disclosed (**~2× AI4's per-SoC memory claimed 2026-08-20, unconfirmed / possibly AI4.1 conflation**) | Not disclosed |
| **AI4.1 / "AI4 Plus"** | target 2027 | **Announced 2026-04-22; no silicon** | Samsung 7 nm, **modified process** | Not disclosed (no NPU-count change stated) | Not disclosed (**~+10 % vs AI4**, Musk) | Not disclosed | **32 GB/SoC, 64 GB/board**, "newer generation RAM"; **~+10 % bandwidth** (Musk) | Not disclosed |
| **AI5** | tape-out announced 2026-04-15 | **Taped out; one packaged engineering sample shown.** Volume targeted 2027 (mid-2027 automotive) | **Dual-sourced: TSMC (node not disclosed) + Samsung SF2-class 2 nm, Taylor TX** | Not disclosed | **Not disclosed** — only marketing multipliers exist (see below) | Not disclosed | **Not disclosed.** Photo shows **12 SK hynix discrete DRAM packages** on an organic substrate (GDDR6 or GDDR7 — grade unestablished); **not HBM/CoWoS** | Not disclosed |

† Contested. WikiChip's per-SoC figures are shown. Electrek (2026-04) instead reports HW3 ~48 GB/s and AI4
~384 GB/s with 32 GB total across the dual-SoC board. **Neither set is Tesla-published** and the two are mutually
inconsistent; both are recorded rather than reconciled.

**AI5 disclosure state.** Tesla has published **no** peak TOPS/FLOPS, dtype list, TDP, die area, memory capacity
or memory bandwidth for AI5. The only performance figures in circulation are conflicting Musk marketing claims:
"up to 40×"/"40 times" better than AI4 (Q3 2025 earnings call, 2025-10-22/23), "up to 10×" more powerful,
"maybe by a factor of 10" perf-per-dollar, and "5× the memory bandwidth of AI4." These are vendor marketing,
not specifications. Musk has also stated AI5 fits "half a reticle with good margin" and integrates the Arm CPU
cores and PCIe blocks alongside the accelerators.

---

## Compute

### NPU (Neural Processing Unit) — Core Block
- Architecture: systolic MAC array, 96×96 grid
- MACs per NPU: 9,216
- INT8 ops per cycle: 18,432 (2 ops per MAC)
- Clock: HW3 @ 2.0 GHz → 36.86 TOPS; HW4 @ 2.2 GHz → ~40.5 TOPS
- HW3: 2 NPUs = 73.7 TOPS; HW4: 3 NPUs = ~121.6 TOPS
- Datapath: 256 B activations + 128 B weights per cycle from SRAM → MACs → 32-bit accumulators
- **AI4.5 / AI4.1 / AI5 (2026):** NPU count, array geometry, clock and dtype support are **not disclosed** for
  any of the three. For AI4.1 Musk stated only "probably a 10% increase in compute" over AI4 — no structural
  change was described, so AI4.1 is best read as a memory-and-clock refresh of the AI4 NPU rather than a new
  microarchitecture. For AI5 nothing about the compute engine is public; the systolic-array description below
  applies to HW3/HW4 and **must not be assumed to carry forward to AI5.**

### CPU Cluster
- HW3: 3 × quad-core ARM Cortex-A72 = 12 cores @ 2.6 GHz
- HW4: 20 ARM cores @ 2.35 GHz
- AI4.5 / AI4.1: not disclosed
- AI5: Musk states the design integrates "the Arm CPU cores and PCIe blocks" on-die alongside the accelerators.
  Core count, microarchitecture and clock are **not disclosed**. The presence of PCIe is itself notable — it is
  the first FSD-line part described as carrying a host interface rather than only automotive IO.

### GPU
- HW3: ARM Mali G71 MP12 @ 1 GHz
- HW4: upgraded (not disclosed)
- AI4.5 / AI4.1 / AI5: not disclosed
- Purpose: camera preprocessing, ISP pipeline, visualization

## Memory

| Level | Capacity | Bandwidth |
|---|---|---|
| NPU SRAM (per NPU) | 32 MiB | very high (local, cycle-level) |
| DRAM (HW3) | 8 GB LPDDR4 | 68 GB/s (WikiChip) / ~48 GB/s (Electrek) † |
| DRAM (HW4) | 16 GB GDDR6 (32 GB per dual-SoC board) | 224 GB/s (WikiChip) / ~384 GB/s (Electrek) † |
| NVMe (HW4) | 256 GB | — |
| DRAM (AI4.1 / AI4 Plus, announced) | **32 GB per SoC, 64 GB per board**, "newer generation RAM" (type not disclosed) | **~+10 % vs AI4** (Musk; no absolute figure) |
| DRAM (AI4.5) | Not disclosed | Not disclosed |
| SRAM / DRAM (AI5) | **Not disclosed.** Package photo shows **12 discrete SK hynix DRAM packages** on an organic substrate | **Not disclosed.** See inference note below |

† **The bandwidth figures for HW3 and HW4 are contested and none is Tesla-published.** WikiChip gives 68 GB/s
(HW3, LPDDR4 @ 4266 Mbps on a 128-bit bus) and 224 GB/s (HW4, GDDR6 @ 14 Gbps on a 128-bit bus). Electrek gives
~48 GB/s and ~384 GB/s respectively, and frames the AI4 figure as board-level with 32 GB total across two SoCs.
The pairs are mutually inconsistent and cannot be reconciled from public data — they are recorded side by side
with attribution. Note that Electrek's "HW3 has one-eighth the memory bandwidth of HW4" line holds only against
its own 48/384 pair.

### AI5 memory organization — what is observed vs what is inferred

**Observed (multi-outlet, from one photograph of one packaged engineering sample; vendor-unconfirmed):**
- A compute ASIC die roughly **half a standard reticle**, ringed by **twelve SK hynix memory packages**.
- The memory is marked like standard discrete DRAM and sits on an **organic substrate** — i.e. **no silicon
  interposer, no CoWoS, no HBM.** Reported by Benzinga, DigiTimes (naming SK hynix as supplier), VideoCardz and
  Tom's Hardware.
- The DRAM grade (GDDR6 vs GDDR7) is **not established by any source.**

**Inferred, single-source — do not cite as a spec:** Tom's Hardware derives a **384-bit external bus** and a
**~768 GB/s – 1.536 TB/s** range purely from the package count and assumed DRAM grade. No other outlet
corroborates the bus width, the grade, or the bandwidth. Tesla has disclosed none of it.

Architecturally the significant, well-supported point is the *class* of memory system, not its width: AI5 keeps
Tesla on **discrete DRAM on organic substrate** rather than moving to HBM-on-interposer, even for a
datacenter/robotics-targeted part. That is a cost- and volume-oriented packaging choice, and it preserves the
same memory *tier structure* as HW4 (on-die SRAM + off-package graphics-class DRAM), scaled up in package count.
(A widely quoted Musk line calling AI5 "one of the most produced AI chips ever" appears in only one outlet's
rendering of the X post and is **not independently corroborated**; it is not relied on here.)

## ISA (8 Instructions)
1. DMA Read (activation/weight load)
2. DMA Write (result store)
3. Dot-product variant A
4. Dot-product variant B
5. Dot-product variant C
6. Scale
7. Element-wise addition (residual connections)
8. Parameter slot (instruction modifier)

This ISA is documented for HW3 (WikiChip reverse-engineering) and is believed to carry into HW4. **No ISA
information of any kind exists for AI4.5, AI4.1 or AI5** — Tesla has never published an FSD ISA, and none of the
2026 disclosures touched the instruction set.

## Execution Model
- In-order instruction execution
- Out-of-order memory subsystem (DMA hides latency)
- Static schedule: generated at compile time, no runtime dispatch
- Compile-time tile partitioning: each weight matrix tile → fixed NPU block assignment

## Chip Integration
- HW3: Samsung 14 nm SoC, ~100 W board TDP
- HW4: Samsung 7 nm SoC, ~160 W board TDP
- Two FSD computers per vehicle (safety redundancy)
- Camera bus, Ethernet, ISP all integrated
- **AI4.5 / "AP45" (from 2025-12):** a distinct FSD computer part number (**2261336-02-A**) found in Fremont-built
  2026 Model Y AWD Premium vehicles and matching Tesla's Electronic Parts Catalog. Tesla has never announced it.
  A **three-SoC** board (vs AI4's two) is reported, but only as a hedged firmware-analysis inference by
  @greentheonly via Electrek — **low confidence, not a teardown die count and not a Tesla statement.** Process,
  compute and memory are not disclosed.
- **AI4.1 / "AI4 Plus" (announced 2026-04-22, production target 2027):** stays on Samsung 7 nm but requires
  Samsung to complete **modifications to that 7 nm process** before production; also reported for Samsung's Texas
  operations. Board organization is unchanged in the announcement (two SoCs → 64 GB total).
- **AI5 (tape-out announced 2026-04-15):** **dual-sourced across two foundries and two nodes** — TSMC (node not
  disclosed by any source) and Samsung **SF2-class 2 nm at Taylor, Texas**. The Samsung derivative
  (SF2 / SF2P / SF2T) is not disclosed. The Samsung leg was disclosed 2026-07-13 by a Samsung Foundry principal
  engineer on LinkedIn; **the post was deleted, Samsung declined to comment, and no official confirmation
  exists.** Korea Times reports both foundries received the design in April 2026, so **no multi-month gap between
  the two tape-outs is supported** — neither tape-out date is public. Correct status: **taped out with a first
  packaged engineering sample shown** — not sampling externally, not shipping. Volume manufacturing targeted 2027
  (mid-2027 for automotive per Electrek).
  - Dual-chip-per-vehicle redundancy is an automotive-platform property and is **not** established for AI5, whose
    near-term priority is Optimus and datacenter/inference clusters (Musk, Q1 2026 call). Automotive use is
    **deferred, not excluded** — Musk allowed that AI5 may eventually go into cars; July 2026 coverage lists
    vehicles among its targets. Sources conflict and Tesla has not settled it.

## Design Philosophy
- Fixed-function for maximum TOPS/W at edge
- Closed/proprietary — no external programmability
- Safety: deterministic latency, dual-chip cross-validation
- **2026 shift:** with AI4 declared sufficient for unsupervised FSD and AI5 aimed first at Optimus and
  datacenter inference, the FSD silicon line is no longer purely an automotive-edge program. The AI5 package —
  half-reticle die, twelve discrete DRAM packages, on-die PCIe — is a datacenter/robotics inference part in
  packaging terms, while the vehicle roadmap advances through incremental AI4 derivatives (AI4.5, AI4.1).
- Conversely, HW3 was **formally conceded insufficient** for unsupervised FSD on 2026-04-22, with a retrofit
  programme ("micro factories") announced for the installed base of roughly 4 million vehicles. Memory bandwidth
  is cited as the choke point, though the specific "one-eighth of HW4" ratio traces to Electrek's write-up
  rather than to a confirmed Tesla statement.

## Disclosure Status (2026-08-08)
- No Tesla talk in the **Hot Chips 38** advance program (2026-08-23 to 2026-08-25); Waymo, not Tesla, holds the
  autonomous-driving keynote and the automotive SoC slot.
- No MLPerf submission for any FSD generation.
- No public SDK, no ISA documentation, no die-area or transistor-count disclosure for any generation.

---

## Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*Classification: **Major**, heavily hedged — see the parallel update in `public/research/tesla-fsd/investigations/hw-architecture.md` for full sourcing detail. Single source: Electrek (2026-08-20), reporting a JPMorgan analyst note (analyst Rajat Gupta) written after a Fremont factory visit/briefing — i.e. Tesla → JPMorgan → Electrek, not a transcript or filing.*

- **AI4.5 — first figures attributed to this name ("~10% more compute, ~2× memory" vs AI4), but likely a conflation with AI4.1/"AI4 Plus"** (whose Musk-stated spec is nearly identical wording). Both readings — genuine AI4.5 spec, or mislabeled AI4.1 — are recorded; neither is adopted as confirmed. See the Generation Overview row above.
- **AI5 delay to mid-2027 reaffirmed** (Electrek's own April 2026 reporting already carried "mid-2027 for automotive"; the August note frames it explicitly as a "delay," which is new framing, not a new date).
- **Cybercab hardware target clarified**: per the same note, Cybercab was originally planned to launch on **AI4** hardware — the first explicit statement found of which chip generation Cybercab targets; consistent with AI5's near-term Optimus/datacenter priority and deferred automotive use.
- **HW3 retrofit reality**: HW3 vehicles now receive a stripped-down **"FSD v14 Lite"** release rather than full FSD, with reported **rising HW3 hardware failure rates** — a new concrete symptom of the 2026-04-22 "HW3 insufficient" finding already in this survey. Thirdhand, not independently corroborated by a second outlet this cycle.
- No new AI5 compute, memory, process-node, or performance figures. No new HW3/HW4 hardware facts.

**Checked, no change found**: no primary Tesla source (earnings call, filing, datasheet) issued new AI4.5/AI5 specs this window; Hot Chips 38 occurred within the window but the "no Tesla talk" finding was not re-verified against a post-event archive this cycle; no MLPerf submission; no public SDK or ISA disclosure.

Source: https://electrek.co/2026/08/20/tesla-jpmorgan-fremont-fsd-v15-hw4-optimus-2027/
