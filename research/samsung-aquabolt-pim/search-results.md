# Samsung Aquabolt-XL HBM-PIM — Search Results

*chip: samsung-aquabolt-pim*
*device_class: Processing-in-Memory*
*search_date: 2026-04-05 (Aquabolt-XL) · 2026-08-08 (LPDDR5X-PIM update)*

## Summary

Samsung developed **Aquabolt-XL**, the world's first HBM with AI processing capability. It is an HBM2 stack augmented with **PIM-DRAM dies** that contain 32 FP16 SIMD processors per die (16 pseudo-channels × 2 MAC units). A full stack has 4 PIM-DRAM dies + 4 standard Aquabolt dies = 128 total processors. Announced at **ISSCC February 2021** and presented at **Hot Chips 33 (August 2021)**. Samsung targeted GPU and FPGA HPC/AI accelerators as drop-in HBM2 replacements.

## Resources Found

### Primary Technical Sources

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 1 | Aquabolt-XL HC33 slides (official Samsung) | https://www.hc33.hotchips.org/assets/program/conference/day1/20210813_HC33_Aquabolt-XL_PIM_Jin_Kim_slide.pdf | Conference Slides | High |
| 2 | Aquabolt-XL: Samsung HBM2-PIM with in-memory processing (IEEE Xplore HC33) | https://ieeexplore.ieee.org/document/9567191/ | Peer-reviewed | High |
| 3 | Samsung's Aquabolt-XL Processor-In-Memory Part 1 (The Memory Guy blog) | https://thememoryguy.com/samsungs-aquabolt-xl-processor-in-memory-part-1/ | Expert Blog | High |
| 4 | Samsung HBM2-PIM and Aquabolt-XL at Hot Chips 33 (ServeTheHome) | https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/ | Conference Coverage | High |
| 5 | HC33 Software Stack image (ServeTheHome) | https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/hc33-samsung-hbm2-pim-aquabolt-xl-software-stack/ | Conference Coverage | High |
| 6 | ISSCC 2021: HBM with integrated AI processor (Electronics Weekly) | https://www.electronicsweekly.com/news/business/isscc-2021-hbm-integrated-ai-processor-2021-02/ | Press | Medium |
| 7 | Samsung HBM-PIM announcement — industry first (Samsung global newsroom) | https://news.samsung.com/global/samsung-develops-industrys-first-high-bandwidth-memory-with-ai-processing-power | Vendor | Medium |
| 8 | Samsung Brings In-Memory Processing to Wider Applications | https://news.samsung.com/global/samsung-brings-in-memory-processing-power-to-wider-range-of-applications | Vendor | Medium |
| 9 | HBM-PIM: Cutting-edge memory for AI (Samsung semiconductor blog) | https://semiconductor.samsung.com/news-events/tech-blog/hbm-pim-cutting-edge-memory-technology-to-accelerate-next-generation-ai/ | Vendor Blog | Medium |
| 10 | PIM Technologies page (Samsung Semiconductor US) — ⚠️ *flagged 2026-08-08: an independent fetch returned only generic Samsung Semiconductor navigation with no HBM-PIM/Aquabolt-XL content, consistent with the page having been retired or restructured. Verify manually before treating as dead.* | https://semiconductor.samsung.com/us/technologies/memory/pim/ | Vendor | Low |
| 11 | Samsung New HBM2 with 1.2 TFLOPS embedded (Tom's Hardware) | https://www.tomshardware.com/news/samsung-hbm2-hbm-pim-memory-tflops | Press | Medium |
| 12 | Samsung Demos In-Memory Processing (TechInsights) | https://www.techinsights.com/microprocessor-report/samsung-demos-memory-processing | Analyst | Medium |
| 13 | Aquabolt-XL: Illinois Experts publication record | https://experts.illinois.edu/en/publications/aquabolt-xl-samsung-hbm2-pim-with-in-memory-processing-for-ml-acc | Academic Index | Low |
| 14 | PIMSys virtual prototype (ACM MemSys 2024) | https://dl.acm.org/doi/10.1145/3695794.3695797 | Peer-reviewed | Medium |

### Key Findings

- **Technology**: 4× PIM-DRAM dies + 4× standard Aquabolt HBM2 dies in one stack
- **Processors**: 32 per PIM die × 4 dies = 128 total processors per HBM stack
- **Processor design**: Each processor = 2× FP16 multiplier + 2× FP16 adder + register files; SIMD all execute same instruction
- **Internal bandwidth**: 4.92 TB/s (vs 1.23 TB/s external I/O)
- **System tested**: Xilinx Alveo U280 FPGA → 2.49× speedup, 62% energy reduction
- **GPU test**: AMD MI60 → ~2× performance, >70% energy reduction
- **Peak compute**: ~1.2 TFLOPS FP16 per stack
- **Drop-in**: JEDEC HBM2 compatible; no memory controller changes required
- **Software**: PIM SW stack (TF + PyTorch transparent offload), PIM BLAS library (Samsung proprietary)
- **Programming model**: SIMD — all 128 processors execute same instruction simultaneously

---

## Additional Resources — LPDDR5X-PIM (search_date: 2026-08-08)

Samsung's PIM program continued on a **non-HBM base**. These resources cover **LPDDR5X-PIM**, a distinct generation absent from the 2026-04-05 search above. Note on dating: LPDDR5X-PIM was publicly named before that baseline (WinBuzzer, 18 Feb 2026) and its architecture reference is IEEE Micro 44(3), 2024 — the 2026 items below are new *disclosure*, not a new program.

### Primary Technical Sources

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 15 | **LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM** — Cha, Choi, Kim, Paik, Lee, Sohn (all Samsung Electronics), arXiv:2606.00636v1, cs.AR, 30 May 2026. **Vendor primary source**; sole source for LPDDR5X-PIM architecture, numeric formats, software stack, and speedup figures | https://arxiv.org/abs/2606.00636 | Vendor primary paper (arXiv) | High |
| 16 | LP5X-PIM Sim — full PDF (read page-by-page during verification) | https://arxiv.org/pdf/2606.00636 | Vendor primary paper (PDF) | High |
| 17 | Hot Chips 2026 program — Memory session, Tue 25 Aug 2026 09:30–10:30 (chair Jae W. Lee): "Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory (PIM) Solution for AI Inference," Karam Hwang (Samsung). ⚠️ **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public. Never cite the talk title as the source of a spec.** | https://www.hotchips.org/ | Conference program | High |
| 18 | Samsung PIMSimulator (SAITPublic, 2023) — Samsung's **public** PIM simulator, cited separately by the arXiv note. ⚠️ **This is NOT LP5X-PIM Sim**, which has no stated open-source release and no repo link | https://github.com/SAITPublic/PIMSimulator | Open-source repo | High |

### Vendor and Trade Coverage

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 19 | Samsung unveils next-gen 3D memory vision at FMS 2026 (4 Aug 2026) — LPDDR5X-PIM showcased as "the industry's first LPDDR memory with processing-in-memory technology," alongside HBM4E, HBM5, zHBM, zNAND-O, PM1763. **No availability or production date given** | https://news.samsungsemiconductor.com/global/samsung-unveils-next-gen-3d-memory-vision-at-fms-2026-charting-the-future-of-ai-infrastructure/ | Vendor newsroom | High |
| 20 | Samsung outlines 3D memory roadmap for AI infrastructure at FMS 2026 (StorageReview, 4 Aug 2026) — independent corroboration of the FMS showcase | https://www.storagereview.com/news/samsung-outlines-3d-memory-roadmap-for-ai-infrastructure-at-fms-2026 | Trade press | Medium |
| 21 | Samsung zNAND-O, LPDDR5X-PIM, PM1763 (SamMobile, 4 Aug 2026) — describes AI-datacenter *and* edge-AI targeting, contradicting a mobile-only framing | https://www.sammobile.com/news/samsung-znand-o-lpddr5x-pim-pm1763-memory-chips-ssd-ai-data-centers/ | Trade press | Medium |
| 22 | ETNews (23 Jul 2026) — repeats the "up to 6.2× faster" **simulation** figure; claims the tech has "moved well beyond the research stage"; positions it for smartphones/laptops; mentions possible pairing with Samsung's GAIA 4 nm NPU. ⚠️ Unconfirmed by Samsung; no customer named | https://en.etnews.com/20260723200002 | Trade press | Low |
| 23 | Seoul Economic Daily (5 Aug 2026) — claims Samsung "entered a full-fledged sales phase to attract Big Tech customers"; names none. ⚠️ **Misdates the Hot Chips talk to September 2026** — hotchips.org says 23–25 Aug 2026. Do not cite for the date | https://en.sedaily.com/finance/2026/08/05/samsung-sk-push-pim-and-cxl-as-us-china-japan-challenge-hbm | Trade press | Low |
| 24 | WinBuzzer (18 Feb 2026) — aggregator. Useful **only** to establish that LPDDR5X-PIM was publicly named before the repo's 2026-04-05 baseline. Contains a unit error ("11.7 GB/s" for HBM4) | https://www.winbuzzer.com/2026/02/18/samsung-lpddr5x-pim-hbm4-memory-ai-computing-xcxwbn/ | Aggregator | Low |

### Key Findings — LPDDR5X-PIM

- **Memory base**: LPDDR5X-9600, strictly JEDEC-compliant (JESD209-5C, 2022); 4 DRAM channels in all published experiments
- **Organization**: one PIM block per DRAM bank (1-to-1 mapping); named registers IRF (instruction register file) and SRF (Source Register File)
- **Modes**: Single-Bank (SB, standard DRAM) and Multi-Bank (MB, parallel PIM execution)
- **Numeric formats**: integer W8A8, W4A4, W8A16, W4A8, W4A16; floating-point W8A8(FP), W8A16(FP) — a real widening vs. Aquabolt-XL's FP16-only datapath
- **Not disclosed**: MAC lanes per PIM block, peak TFLOPS/TOPS, internal-vs-external bandwidth, capacity, process node, package. Samsung defers these to "future publications"
- **Performance (Samsung simulation, NOT silicon)**: GEMV vs. a self-defined non-PIM sequential-weight-read baseline with 4 channels — 6.0×–6.2× at tile dim 4096 for W8A8/W4A4/W8A8(FP); 5.7×–5.8× for W8A16/W4A16/W8A16(FP); >5.0× for most configs with a 150 ns fence modeled, low of 4.1× for W4A16; Reshape optimization adds up to 1.65× for W < 2048
- **Software**: "PIM Kernel" layer = Data Mapper (offline bank placement via a "PIM Tile Configuration") + PIM Executor (PIM Device Code Gen / PIM Control / GEMV Kernel). Address mappings: Vertical, Horizontal, Reshape. Not public
- **Simulator**: LP5X-PIM Sim, cycle-accurate, built on DRAMSim3 and Ramulator; **open-source status not confirmed**
- **Status**: announced/showcased only — no production date, no availability, no named customers, no measured silicon
- **Target market**: mixed — mobile/edge framing (ETNews, paper's "mobile application processors" hint) vs. AI-infrastructure/datacenter framing (Samsung FMS post, SamMobile). Do not state exclusively
- **HBM-PIM successor**: not confirmed either way — absence of evidence across the Hot Chips 2026 program, FMS 2026 coverage, and Samsung's PIM page, not a verified negative

### Search Gaps and Failed Retrievals (2026-08-08)

- Verification ran with an exhausted WebSearch budget; discovery relied on direct WebFetch of already-cited primary sources plus a page-by-page read of the arXiv PDF. Sources not already cited could not be exhaustively discovered — an HBM-PIM announcement never searched for cannot be ruled out.
- `https://news.samsung.com/global/samsung-ships-industry-first-commercial-hbm4-with-ultimate-performance-for-ai-computing` — timed out twice; Samsung HBM4 peripheral specs therefore **not published** anywhere in this chip's deliverables
- `https://hotchips.org/program/` — HTTP 403 (the root https://www.hotchips.org/ did load)
- `https://ieeexplore.ieee.org/document/10502321` — empty body
