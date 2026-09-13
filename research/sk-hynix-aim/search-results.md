# SK Hynix AiM (Accelerator-in-Memory) — Search Results

*chip: sk-hynix-aim*
*device_class: Processing-in-Memory*
*search_date: 2026-04-05*
*last_rescan: 2026-08-08 — see "Re-scan — 2026-08-08" below*

## Summary

SK Hynix developed **AiM (Accelerator-in-Memory)**, a PIM product family that integrates compute units inside GDDR6 DRAM chips. The GDDR6-AiM embeds FP16 MAC units inside each bank, leveraging the massive internal DRAM bandwidth (not the limited I/O bandwidth). The **AiMX** is a multi-chip accelerator card combining multiple GDDR6-AiM packages. It was presented at Hot Chips 34 (2022), ASPLOS 2023 tutorial, IEEE ISCA workshop, and Hot Chips 2024.

## Resources Found

### Primary Technical Sources

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 1 | System Architecture and Software Stack for GDDR6-AiM (Hot Chips 34 poster) | https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf | Conference Paper | High |
| 2 | System Architecture and Software Stack for GDDR6-AiM (IEEE Xplore) | https://ieeexplore.ieee.org/document/9895629/ | Peer-reviewed | High |
| 3 | Hardware Architecture and Software Stack for PIM Based on Commercial DRAM (ISCA 2021) | https://ieeexplore.ieee.org/document/9499894 | Peer-reviewed | High |
| 4 | SK hynix Debuts First GDDR6-AiM Accelerator Card 'AiMX' | https://news.skhynix.com/sk-hynix-debuts-first-gddr6-aim-accelerator-card-aimx-for-generative-ai/ | Vendor Press | Medium |
| 5 | Upgraded AiMX Solution at AI HW Summit 2024 | https://news.skhynix.com/sk-hynix-presents-upgraded-aimx-solution-at-ai-hw-edge-ai-summit-2024/ | Vendor Press | Medium |
| 6 | SK Hynix AiMX-xPU at Hot Chips 2024 (ServeTheHome) | https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/ | Conference Coverage | High |
| 7 | SK Hynix AiMX-xPU at Hot Chips 2024 (IEEE) | https://ieeexplore.ieee.org/document/10664793/ | Peer-reviewed | High |
| 8 | System Architecture and Software Stack ASPLOS 2023 Tutorial | https://events.safari.ethz.ch/asplos-pim-tutorial/lib/exe/fetch.php?media=system_architecture_and_software_stack_for_gddr6-aim_sk_hynix_pim_tutorial_asplos23_sent.pdf | Tutorial Slides | High |
| 9 | AiM PIM use case: LLM accelerator (SK Hynix ICOS slides) | https://icos-semiconductors.eu/wp-content/uploads/2024/04/2-3-High-level-speaker_Euicheol-Lim-Vice-president-of-SK-Hynix.pdf | Industry Talk | Medium |
| 10 | aim_simulator GitHub (Ramulator 2.0 based) | https://github.com/arkhadem/aim_simulator | Open Source Simulator | High |
| 11 | SK hynix Develops PIM (newsroom) | https://news.skhynix.com/sk-hynix-develops-pim-next-generation-ai-accelerator/ | Vendor | Low |
| 12 | SK Hynix GDDR6-AiM 16x faster Tom's Hardware | https://www.tomshardware.com/news/sk-hynix-next-generation-ai-accelerator | Press | Low |
| 13 | AI Infra Summit 2025 SK hynix | https://news.skhynix.com/ai-infra-summit-2025/ | Vendor | Low |

### Key Findings

- **AiM DRAM device**: GDDR6 DRAM with embedded MAC units at each bank group
- **All-bank operation**: All 16 banks operate simultaneously, exposing internal bandwidth
- **In-DRAM compute**: FP16 MAC array per pseudo-channel; no movement to external GPU/CPU for matmul
- **AiMX card**: Multi-chip card with 16 GDDR6-AiM packages (16 GB) → upgraded 32 GDDR6-AiM packages (32 GB)
- **Performance**: Up to 16× faster than CPU/GPU+DRAM for memory-bound workloads
- **Power reduction**: Up to 80% power reduction vs standard GDDR6 + GPU
- **Applications**: LLM attention (KV-cache matmul), embedding lookup, RNN inference
- **Software stack**: Host-side runtime with AiM ISR (Instruction Set Register); extended DRAM command set; BLAS-like library; framework integration (PyTorch, TensorFlow paths inferred)
- **Simulator**: Open-source aim_simulator based on Ramulator 2.0 available on GitHub

---

## Re-scan — 2026-08-08 (window 2026-04-05 → 2026-08-08)

*No in-window AiM/AiMX development found. Resources below were newly located: two are pre-baseline items the 2026-04-05 scan missed, the rest are the negative-evidence and context sources used to establish that nothing changed.*

### Newly Found Resources

| # | Title | URL | Type | Date | Quality | Why it matters |
|---|-------|-----|------|------|---------|----------------|
| 14 | SK hynix Showcases Next-Generation AI Memory Innovation at CES 2026 (PRNewswire) | https://www.prnewswire.com/news-releases/sk-hynix-showcases-next-generation-ai-memory-innovation-at-ces-2026-302653161.html | Vendor Press | 2026-01-05 | High | **Primary source for AiMX still being a "prototype."** Also names CuD, CMM-Ax, cHBM; headline product 16-layer 48 GB HBM4 |
| 15 | SK hynix newsroom — CES 2026 | https://news.skhynix.com/en/sk-hynix-showcases-next-generation-ai-memory-innovations-at-ces-2026/ | Vendor | 2026-01 | High | Newsroom mirror of #14 |
| 16 | SK hynix at AI Infra Summit 2025 (English newsroom) | https://news.skhynix.com/en/ai-infra-summit-2025/ | Vendor | post 2025-10-01 (summit ~Sept 2025) | Medium-High | **Primary source for the vLLM integration** and the Supermicro SYS-421GE-TNRT3 + 2× H100 + 4× AiMX demo. Supersedes item #13 above (non-`/en/` URL) |
| 17 | SK hynix Showcases Full-Stack AI Memory Portfolio at 2025 OCP Global Summit | https://news.skhynix.com/en/sk-hynix-showcases-full-stack-ai-memory-portfolio-at-2025-ocp-global-summit/ | Vendor | 2025-10-31 | Medium-High | Repeat of the same 2× H100 + 4× AiMX demo; no new figures |
| 18 | TechPowerUp #341520 — AI Infra Summit 2025 AiMX demo | https://www.techpowerup.com/341520/ | Press | 2025-09 | Medium | Independent corroboration of the demo configuration |
| 19 | [Tech Note EP.1] It's the Memory — The Future of AI Will Be Decided by Memory | https://news.skhynix.com/en/tech-note-series-ep1/ | Vendor blog (educational) | 2026-05-12 | Low | **Only in-window SK hynix item mentioning AiM.** Argument only — no specs, no roadmap |
| 20 | SK hynix at FMS 2026 (recap) | https://news.skhynix.com/en/fms-2026/ | Vendor | 2026-08-07 | High | **Negative evidence:** HBF, HBM4/HBM4E, SOCAMM2, DDR5 RDIMM, QLC eSSD, CXL — no AiM/AiMX |
| 21 | SK hynix / Sandisk HBF at FMS 2026 | https://news.skhynix.com/en/hbf-at-fms-2026/ | Vendor | 2026-08 | Medium | Context for the shift of emphasis away from AiM |
| 22 | Hot Chips 38 program | https://www.hotchips.org/ | Conference program | fetched 2026-08-08 | High | **Negative evidence.** Aug 23–25, 2026 — a FUTURE event: program public, no slides/abstracts/specs. No SK hynix AiM/AiMX talk; PIM entries are Samsung LPDDR5X-PIM and XCENA/Samsung MX1 |
| 23 | aim_simulator commit history | https://github.com/arkhadem/aim_simulator/commits/main | Open Source | last commit 2025-07-22 | High | **Negative evidence:** simulator dormant; no in-window activity |
| 24 | SK hynix newsroom site search for "AiM" | https://news.skhynix.com/?s=AiM | Vendor search | fetched 2026-08-08 | Medium | **Negative evidence:** only 2026 AiM hit is Tech Note EP.1 |
| 25 | All About Circuits — SK hynix LPDDR6 at ISSCC 2026 | https://www.allaboutcircuits.com/news/sk-hynix-details-low-power-ddr6-sdram-isscc-2026/ | Press | 2026-02 | Medium | **Negative evidence:** SK hynix's ISSCC 2026 papers were LPDDR6/GDDR7, no PIM paper |
| 26 | TweakTown — SK hynix 16 Gb LPDDR6 14.4 Gbps at ISSCC 2026 | https://www.tweaktown.com/news/110149/ | Press | 2026-02 | Low-Medium | Corroborates #25 |
| 27 | Kisaco Research — AiM/AiMX: SK hynix's PIM Solution Unleashed | https://www.kisacoresearch.com/content/efficient-inferencing-infrastructure-track-aimaimx-sk-hynixs-pim-solution-unleashed | Conference session listing | 2025 | Low | Session abstract for the AI Infra Summit AiMX talk |
| 28 | TrendForce — Beyond HBM: Samsung, SK hynix reportedly explore next-gen AI memory | https://www.trendforce.com/news/2026/03/10/news-beyond-hbm-samsung-sk-hynix-reportedly-explore-next-gen-ai-memory/ | Aggregator | 2026-03-10 | **Low — claim REJECTED** | Asserts AiM is "already deployed in real-world applications." No customer, date, or volume; contradicts SK hynix's own "prototype" wording. **Do not cite as a status change.** |

### Re-scan Findings

- **No in-window development.** Nothing changed for AiM/AiMX between 2026-04-05 and 2026-08-08.
- **Status backfill:** AiMX is still a **prototype** in SK hynix's own words (CES 2026, 2026-01-05) — not sampling, not shipping, no named customer.
- **Software backfill:** **vLLM** is a real, publicly demonstrated part of the AiMX stack (2025 AI Infra Summit and OCP Global Summit), run on a Supermicro SYS-421GE-TNRT3 with 2× H100 + 4× AiMX. No numeric performance figures disclosed.
- **Not confirmed, treat as nonexistent:** AiM Gen3, GDDR7-AiM, HBM-based AiM, AiM in an HBM4 custom base die, any MLPerf submission using AiMX, any public AiM SDK release, any cancellation announcement.
- **Direction of travel:** SK hynix's 2026 compute-memory messaging emphasizes CXL (CMM-Ax, CMM-Hybrid, Pooled Memory), cHBM, and HBF over AiM.
- **Search caveat:** the verifying pass exhausted its WebSearch budget; corroboration came from DuckDuckGo HTML result pages plus direct fetches, and GitHub's REST API returned 403 (simulator date read from the rendered commits page).
