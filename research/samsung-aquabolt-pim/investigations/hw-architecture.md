# Samsung Aquabolt-XL HBM-PIM — Hardware Architecture Investigation

*chip: samsung-aquabolt-pim*
*investigation: hw-architecture*
*date: 2026-04-05*
*sources: HC33 Samsung slides, IEEE HC33 paper, The Memory Guy blog, ServeTheHome HC33, ISSCC 2021, Samsung newsroom*

---

## 1. Architecture Paradigm

Samsung Aquabolt-XL is **Processing-in-Memory (PIM)** embedded inside an **HBM2 memory stack**. The approach is distinct from SK Hynix AiM (GDDR6-based): Samsung works with **HBM2**, the high-bandwidth memory used by GPUs and FPGAs.

Samsung replaced the **bottom 4 dies** of a standard 8-die HBM2 Aquabolt stack with **PIM-DRAM dies** — custom dies that retain full DRAM storage but add **FP16 SIMD processors** at each bank boundary.

The key property: **drop-in HBM2 replacement** — no changes to GPU or FPGA memory controllers required.

---

## 2. Die Stack Organization

```
Standard HBM2 Stack (Aquabolt)   vs.   Aquabolt-XL (HBM-PIM)
─────────────────────────────         ─────────────────────────
  Die 7 (top): DRAM                     Die 7: Standard DRAM
  Die 6: DRAM                           Die 6: Standard DRAM
  Die 5: DRAM                           Die 5: Standard DRAM
  Die 4: DRAM                           Die 4: Standard DRAM
  Die 3: DRAM                           Die 3: PIM-DRAM ★
  Die 2: DRAM                           Die 2: PIM-DRAM ★
  Die 1: DRAM                           Die 1: PIM-DRAM ★
  Die 0 (base): DRAM                    Die 0: PIM-DRAM ★
  Logic Die (base)                      Logic Die (base)
```

- 4 standard DRAM dies + 4 PIM-DRAM dies
- **Total capacity**: same as standard HBM2 (16 GB typical)
- **Logic die**: unchanged — still handles JEDEC HBM2 protocol

---

## 3. PIM-DRAM Die Architecture

### 3.1 Bank and Pseudo-Channel Organization
Each PIM-DRAM die (HBM2 standard):
- 16 banks
- 16 pseudo-channels (PC0–PC15)
- 1 bank per pseudo-channel

### 3.2 Processor Placement
- **1 processor per pseudo-channel** → 16 processors per PIM-DRAM die (32 per some variants)
- Processor is placed at the **bank boundary** between even and odd banks
- Position enables simultaneous access to both banks without going through I/O pins

### 3.3 Processor Design (per pseudo-channel)
```
Processor unit:
  ├── 2× FP16 multipliers
  ├── 2× FP16 adders
  ├── 3× register files
  └── Control unit (micro-RISC instruction decoder)
```

All processors execute the **same instruction simultaneously** (SIMD / SPMD model — "Single Instruction Multiple Processors").

---

## 4. Compute Specifications

| Metric | Per PIM die | Full Stack (4 PIM dies) |
|--------|-------------|------------------------|
| Processors | 32 | 128 |
| Peak FP16 FLOPS | ~300 GFLOPS | ~1.2 TFLOPS |
| Internal bandwidth | ~1.23 TB/s per die | 4.92 TB/s total |
| External I/O bandwidth | ~307 GB/s per die | ~1.23 TB/s |

Note: Internal:external bandwidth ratio is ~4×, exposing the memory bandwidth advantage.

---

## 5. Memory Architecture

| Path | Bandwidth | Notes |
|------|-----------|-------|
| Processor ↔ DRAM arrays (internal) | 4.92 TB/s | All processors access own pseudo-channel banks |
| External HBM2 I/O pins | 1.23 TB/s | Standard JEDEC HBM2 off-stack transfer |
| Processor ↔ External (via I/O) | 1.23 TB/s | For data not in PIM local banks |

The processors operate on data **stored in their local pseudo-channel** — no cross-die data movement needed for PIM compute.

---

## 6. JEDEC Compatibility and Integration

- **Drop-in HBM2 replacement**: Same ball map, same package dimensions, same JEDEC timing specs
- **No memory controller changes**: Aquabolt-XL responds to all standard HBM2 commands
- **PIM activation**: Done via special PIM mode enable sequence (not documented publicly)
- **Target hosts**: GPU (AMD MI60 tested), FPGA (Xilinx Alveo U280 tested)

---

## 7. Instruction Architecture

- Typical **RISC 32-bit instructions** (simplified)
- Micro-code resident in local register/instruction memory per processor
- Host programs the instruction sequence; processors execute it on their local data
- No branch prediction, no OoO execution
- Think: a simple SIMD controller inside DRAM

---

## 8. Performance Validation Results

### With Xilinx Alveo U280 FPGA
- Application: **RNN-T speech recognition**
- Performance: **2.49× speedup** vs standard HBM2 + Alveo
- Energy: **62% reduction**

### With GPU (AMD MI60 inferred)
- Performance: **>2× speedup** (ML workloads)
- Energy: **>70% reduction**

---

## 9. Packaging and Process

- Manufactured by Samsung Foundry
- **3D stack TSV (Through-Silicon Via)** connecting all dies — standard HBM2 packaging
- Logic die at bottom handles protocol conversion
- Process node: 20nm class (Samsung HBM2 generation)

---

## 10. Limitations and Design Tradeoffs

1. **SIMD only**: All processors run same instruction — no per-processor independent control flow
2. **Local data only**: PIM compute works only on data in the processor's own pseudo-channel banks; cross-channel data requires normal I/O path
3. **FP16 only**: No INT8, no BF16, no FP32 in this generation
4. **No cache**: Data must reside in the local DRAM array
5. **Limited compute**: 1.2 TFLOPS total — not a replacement GPU; effective only for memory-bound GEMV

---

## Sources

- [Aquabolt-XL HC33 Samsung Official Slides](https://www.hc33.hotchips.org/assets/program/conference/day1/20210813_HC33_Aquabolt-XL_PIM_Jin_Kim_slide.pdf)
- [Aquabolt-XL IEEE HC33 Paper](https://ieeexplore.ieee.org/document/9567191/)
- [Samsung's Aquabolt-XL PIM Part 1 — The Memory Guy](https://thememoryguy.com/samsungs-aquabolt-xl-processor-in-memory-part-1/)
- [Samsung HBM2-PIM at HC33 — ServeTheHome](https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/)
- [Samsung HBM-PIM Newsroom](https://news.samsung.com/global/samsung-develops-industrys-first-high-bandwidth-memory-with-ai-processing-power)
- [Samsung New HBM2 1.2 TFLOPS — Tom's Hardware](https://www.tomshardware.com/news/samsung-hbm2-hbm-pim-memory-tflops)

---
---

# Investigation Update — LPDDR5X-PIM (2026-08-08)

*investigation: hw-architecture (generation 2)*
*date: 2026-08-08*
*primary source: arXiv:2606.00636v1 (Samsung Electronics), read in full*
*secondary: hotchips.org program; Samsung Semiconductor newsroom FMS 2026; StorageReview; SamMobile; ETNews; Seoul Economic Daily*

## U1. Why this update exists

The 2026-04-05 investigation above covers only Samsung's 2021 HBM2-based Aquabolt-XL. A repo-wide grep for "lpddr5x-pim", "lp5x-pim", and "lpddr5x pim" returned zero hits before this update — Samsung's current PIM generation was entirely missing from the survey.

**Dating discipline.** LPDDR5X-PIM is *not* a new-in-window program. The arXiv note cites its own architecture reference as B. Kim et al., *IEEE Micro* 44(3):40–48, 2024, and trade press (WinBuzzer, 18 Feb 2026 — aggregator, low confidence) reported Samsung "advancing LPDDR5X-PIM" before the repo's 2026-04-05 baseline. What is new in-window is the **substantive disclosure**: the Samsung-authored arXiv note (30 May 2026), the FMS 2026 showcase (4–6 Aug 2026), and a scheduled Hot Chips 2026 talk (25 Aug 2026). The correct framing is "a program the survey missed, now substantively disclosed," not "resurfaced."

## U2. Primary source

**arXiv:2606.00636v1**, cs.AR, submitted 30 May 2026 — *"LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM"* — SangHoon Cha, Jaewan Choi, Byeongho Kim, Yoonah Paik, Sukhan Lee, Kyomin Sohn. The paper states "All authors are with Samsung Electronics, South Korea," making this a vendor primary source. All architecture, numeric-format, and speedup figures below were read directly from the PDF, not from secondary coverage.

## U3. Architecture — what is disclosed

### U3.1 PIM organization

> "Each PIM block is deployed in a 1-to-1 mapping with a corresponding DRAM bank, allowing direct data transfer through internal compute units and specialized PIM registers."

This is a **per-bank** granularity, structurally different from Aquabolt-XL's **per-pseudo-channel** processor placement at a bank boundary.

Named registers: **IRF** (instruction register file, carrying "IRF code") and **SRF** (Source Register File).

### U3.2 Operating modes

| Mode | Behavior |
|---|---|
| Single-Bank (SB) | Standard DRAM operation |
| Multi-Bank (MB) | Parallel PIM execution across multiple banks |

Mode transitions are managed in software by the runtime's PIM Control component.

### U3.3 Numeric formats — the substantive change vs. Aquabolt-XL

| Class | Combinations evaluated (weight-bits / activation-bits) |
|---|---|
| Integer | W8A8, W4A4, W8A16, W4A8, W4A16 |
| Floating point | W8A8(FP), W8A16(FP) |

Aquabolt-XL was **FP16-only**. LPDDR5X-PIM supports 4-bit and 8-bit weights against 4/8/16-bit activations, plus a floating-point path.

### U3.4 Reference memory system

- **LPDDR5X-9600**
- **Strictly JEDEC-compliant** timing (JESD209-5C, 2022)
- **Four DRAM channels** in all published experiments

### U3.5 What is explicitly NOT disclosed

Samsung's own words: *"Further technical details regarding the specific architecture and circuit design of the LPDDR5X-PIM will be disclosed in future publications."*

Consequently the following are **not disclosed** and must not be estimated:
- MAC lanes per PIM block
- Peak throughput (TFLOPS / TOPS)
- Internal-vs-external bandwidth figures
- Device capacity
- Process node
- Package / form factor
- Die area, power, TDP

## U4. Performance — vendor simulation output, not measured silicon

All figures are output from Samsung's own cycle-accurate **LP5X-PIM Sim**, normalized to a **self-defined non-PIM sequential-weight-read baseline with four DRAM channels**. No silicon measurement and no third-party reproduction exists.

| Configuration (baseline weight-tile dimension 4096) | GEMV speedup |
|---|---|
| W8A8, W4A4, W8A8(FP) — larger tile shapes | 6.0× – 6.2× |
| W8A16, W4A16, W8A16(FP) — smaller tile shapes | 5.7× – 5.8× |
| With 150 ns static memory-fence latency modeled | > 5.0× for most configs; low of 4.1× for W4A16 |
| Reshape column-partitioning optimization, W < 2048 | additional up to 1.65× |

The 150 ns fence value is described in the paper as "a representative value for high-performance mobile application processors" — the paper's only application-domain hint.

## U5. Simulator (hardware-modeling tool, not the accelerator)

**LP5X-PIM Sim** is cycle-accurate and built on **DRAMSim3** and **Ramulator**.

⚠️ **Availability not confirmed.** The paper contains no open-sourcing statement and no repository link. Do **not** conflate it with Samsung's earlier **PIMSimulator** (https://github.com/SAITPublic/PIMSimulator, 2023), which the paper cites separately and which *is* public. Merging the two would wrongly imply LP5X-PIM Sim is downloadable.

## U6. Disclosure status

- **Hot Chips 2026** (verified on hotchips.org): conference runs Sun 23 – Tue 25 August 2026 at Memorial Auditorium, Stanford. Memory session, **Tue 25 Aug 2026, 09:30–10:30** (chair Jae W. Lee): *"Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory (PIM) Solution for AI Inference,"* presenter **Karam Hwang (Samsung)**. As of 2026-08-08 the talk has **not been given** — disclosure scheduled, content not yet public. Do not cite the talk title as a source of any spec.
- **Date warning**: Seoul Economic Daily (5 Aug 2026) misdates this to "the 23rd of next month" (September 2026). Independently reproduced. Trust hotchips.org.

## U7. Productization status

Announced and showcased only. At **FMS 2026** (4–6 Aug 2026) Samsung showcased LPDDR5X-PIM as "the industry's first LPDDR memory with processing-in-memory technology," alongside HBM4E, HBM5, zHBM, zNAND-O and PM1763. Samsung's own post gives **no availability or production date** — corroborated independently by StorageReview and SamMobile (both 4 Aug 2026).

Correct status verb: **announced / showcased**. Not supported: sampling, shipping, mass production, deployment, named customers.

Trade press (label low confidence, no Samsung confirmation, no customer named):
- ETNews (23 Jul 2026): repeats the "up to 6.2× faster than conventional memory" simulation figure; says the technology has "moved well beyond the research stage"; aims at near-term on-device AI; mentions possible pairing with Samsung's GAIA 4 nm NPU.
- Seoul Economic Daily (5 Aug 2026): Samsung has "entered a full-fledged sales phase to attract Big Tech customers."

## U8. Target market — mixed; do not state exclusively

| Source | Framing |
|---|---|
| ETNews (23 Jul 2026) | Smartphones and laptops (on-device AI) |
| arXiv:2606.00636 | Only hint: 150 ns is "a representative value for high-performance mobile application processors" |
| Samsung FMS 2026 post | "Charting the future of AI infrastructure" |
| SamMobile (4 Aug 2026) | AI data centers and edge AI devices |

**Defensible statement**: primarily an on-device/edge AI inference play, marketed by Samsung under an AI-infrastructure banner. For this datacenter survey, its relevance is as the **successor of Samsung's PIM program** and as a **contrast case** to Aquabolt-XL's GPU/FPGA-attached HBM2 targeting — not as a confirmed datacenter part.

## U9. HBM-PIM / Aquabolt-XL successor — not confirmed either way

No HBM3/HBM4-based Samsung PIM or Aquabolt-XL successor surfaced in the Hot Chips 2026 program, in FMS 2026 coverage, or on Samsung's PIM technology page. **This is absence of evidence, not a verified negative.**

Samsung's Hot Chips 2026 Tutorial 1 talk, *"HBM Base Die: How will HBM evolve in the future by utilizing the logic process?"* (Sangwook Han, Samsung, 23 Aug 2026), covers logic-process base dies and is **not** described as PIM — and, as a future talk, its content is not public.

⚠️ **Deliberately omitted**: peripheral Samsung HBM4 specifications circulating alongside this story (a 12 Feb 2026 shipping announcement, 11.7 Gbps rated / up to 13 Gbps, 4 nm logic base die, 1c DRAM, HBM4E samples May 2026, an ISSCC 2026 "36GB 3.3TB/s HBM4" paper) **could not be independently verified** — news.samsung.com timed out on repeated attempts, and the only aggregator that loaded garbles the data rate as "11.7 GB/s" (a unit error). These figures are not published here.

## U10. Comparison: Aquabolt-XL vs. LPDDR5X-PIM

| Axis | Aquabolt-XL (2021) | LPDDR5X-PIM (disclosed 2026) |
|---|---|---|
| Memory base | HBM2, 8-die 3D TSV stack | LPDDR5X-9600, discrete device |
| PIM granularity | 1 processor / pseudo-channel (32/die, 128/stack) | 1 PIM block / DRAM bank (1:1) |
| Compute unit detail | 2× FP16 mul + 2× FP16 add + 3 RF + RISC-32 µcontroller | not disclosed |
| Numeric formats | FP16 only | INT W8A8/W4A4/W8A16/W4A8/W4A16 + FP W8A8(FP)/W8A16(FP) |
| Peak throughput | ~1.2 TFLOPS FP16 / stack | not disclosed |
| Internal BW | 4.92 TB/s | not disclosed |
| External BW | 1.23 TB/s (HBM2 I/O) | not disclosed |
| Mode model | Vendor-private PIM mode enable sequence | First-class SB / MB two-mode model |
| Host integration | In-package with GPU/FPGA over interposer | Standard LPDDR5X memory interface |
| Evidence class | Measured on real hardware (MI60, Alveo U280) | Cycle-accurate simulation only |
| Target | GPU/FPGA memory-bound datacenter/HPC workloads | Primarily on-device/edge AI; marketed under AI-infrastructure banner |

## U11. Methodological caveat carried from verification

The adversarial verification pass that confirmed these figures had an exhausted WebSearch budget and relied on direct WebFetch of primary sources plus a full page-by-page read of the arXiv PDF. That is strong for confirming the specifics recorded above, but weak for discovering sources not already cited — in particular, an HBM-PIM announcement that was never searched for cannot be exhaustively ruled out.

## U12. Repo hygiene note

An independent fetch of https://semiconductor.samsung.com/us/technologies/memory/pim/ (cited at `research/samsung-aquabolt-pim/search-results.md`, row 10) returned only generic Samsung Semiconductor navigation with no HBM-PIM/Aquabolt-XL content, consistent with the page having been retired or restructured. Flagged in search-results.md; verify manually before treating as a dead link.

## Sources (LPDDR5X-PIM update)

- [arXiv:2606.00636 — LP5X-PIM Sim (Samsung, 30 May 2026)](https://arxiv.org/abs/2606.00636) · [PDF](https://arxiv.org/pdf/2606.00636)
- [Hot Chips 2026 program](https://www.hotchips.org/)
- [Samsung Semiconductor newsroom — FMS 2026](https://news.samsungsemiconductor.com/global/samsung-unveils-next-gen-3d-memory-vision-at-fms-2026-charting-the-future-of-ai-infrastructure/)
- [StorageReview — Samsung 3D memory roadmap at FMS 2026](https://www.storagereview.com/news/samsung-outlines-3d-memory-roadmap-for-ai-infrastructure-at-fms-2026)
- [SamMobile — zNAND-O / LPDDR5X-PIM / PM1763](https://www.sammobile.com/news/samsung-znand-o-lpddr5x-pim-pm1763-memory-chips-ssd-ai-data-centers/)
- [ETNews (23 Jul 2026)](https://en.etnews.com/20260723200002) — low confidence
- [Seoul Economic Daily (5 Aug 2026)](https://en.sedaily.com/finance/2026/08/05/samsung-sk-push-pim-and-cxl-as-us-china-japan-challenge-hbm) — low confidence; misdates Hot Chips
- [WinBuzzer (18 Feb 2026)](https://www.winbuzzer.com/2026/02/18/samsung-lpddr5x-pim-hbm4-memory-ai-computing-xcxwbn/) — aggregator; establishes pre-baseline public mention only
