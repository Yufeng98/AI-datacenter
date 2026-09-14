# SK Hynix AiM (Accelerator-in-Memory) — Search Results

*chip: sk-hynix-aim*
*device_class: Processing-in-Memory*
*search_date: 2026-04-05*
*last_rescan: 2026-09-13 — see "Re-scan — 2026-08-08" and "Resources Added 2026-09-13" below*

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

---

## Resources Added 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*Classification: **Minor**. No AiM/AiMX product, spec, or status change found in this window — AiMX remains a "prototype" per SK hynix's own last statement (CES 2026, unchanged). This scan resolves one previously-flagged ambiguity (the Hot Chips 38 Tutorial 1 speaker/title) and adds negative evidence on two specific questions this round of research was asked to check: an SK hynix LPDDR-based PIM response to Samsung, and an SK hynix HBM4 logic base die. Method note: this session's WebSearch budget was exhausted (200/200) before this chip was investigated; DuckDuckGo returned a CAPTCHA wall on every query, so discovery relied on WebFetch against primary URLs (hotchips.org program page, news.skhynix.com, and direct navigation) plus Bing's rendered HTML, which returned only generic/irrelevant snippets for open-ended queries.*

| # | Title | URL | Type | Date | Quality | Why it matters |
|---|-------|-----|------|------|---------|----------------|
| 29 | Hot Chips 2026 (38) full program — hc2026.hotchips.org | https://hc2026.hotchips.org/program/ | Conference program | fetched 2026-09-13 | High | **Resolves prior ambiguity.** Confirms Tutorial 1 (Memory Technology, Sun Aug 23, 9:00–11:00 AM, chair Suresh Rajgopal) includes **"Advanced packaging for High Bandwidth Memory (HBM)" — Jaesik Lee, SK Hynix**. This is SK hynix's only Hot Chips 38 appearance; it is HBM *packaging* technology, not an AiM/PIM product disclosure, consistent with how the 2026-08-08 entry already treated it (then unconfirmed) |
| 30 | Hot Chips 2026 program — Samsung LPDDR5X-PIM talk | https://hc2026.hotchips.org/program/ | Conference program | fetched 2026-09-13 | High | Confirms the Tuesday Memory session PIM talk, **"Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory"** (Karam Hwang, Samsung), is Samsung's, not SK hynix's — **SK hynix has no competing LPDDR-based PIM disclosure at HC38** |
| 31 | Hot Chips 2026 program — Samsung HBM base-die talk | https://hc2026.hotchips.org/program/ | Conference program | fetched 2026-09-13 | High | Confirms Tutorial 1 also includes **"HBM Base Die: How HBM Will Evolve Using Advanced Logic Processes"** (Sangwook Han, **Samsung**) — the HBM4-logic-base-die disclosure at HC38 belongs to Samsung, not SK hynix |
| 32 | SK hynix Presents a Full Lineup of Memory Solutions Optimized for AI Infrastructure at DTF 2026 | https://news.skhynix.com/en/dtf-2026/ | Vendor | 2026-08-26 | High | **Negative evidence.** Full DTF 2026 product lineup (HBM3E/HBM4/HBM4E, RDIMM/MRDIMM/SOCAMM2/CSODIMM/LPCAMM2, LPDDR5X, GDDR7, QLC eSSD/cSSD lines, CMM-DDR5) — **no AiM/AiMX/PIM mention**; an HBM4 wafer was displayed with no base-die architecture detail given |
| 33 | [Tech Note] Hybrid Bonding: Evolving into a Foundational Technology for Improving Semiconductor Performance | https://news.skhynix.com/en/tech-note-series-ep2/ | Vendor blog (educational) | 2026-08-25 | Medium | **Relevant negative for the HBM4-logic-base-die question.** States HBM4 I/O count doubles to 2,048 and package height grows to 775 µm, but describes hybrid bonding as likely arriving at **HBM4E or HBM5** (stack heights >20 layers) — i.e. HBM4 itself is described as still using conventional TCB. No mention of a logic-process base die, a foundry base-die partner, or AiM/PIM |
| 34 | SK hynix Charts Its Business and Technology Direction at the 2026 Future Forum | https://news.skhynix.com/en/future-forum-2026/ | Vendor | 2026-09-09 (event 2026-09-08) | Medium | **Roadmap-level, not AiM-specific.** "Full-Stack AI Memory Strategy" (3D DRAM + HBM + HBF combined per-workload) and a forward-looking "3D Memory Technology" direction described as "converting DRAM peripheral circuits into logic foundry capabilities" — vague and not tied to AiM/PIM by name; recorded as context only, not as an AiM development |
| 35 | SK hynix Holds Groundbreaking Ceremony for HBM Production Base in Indiana | https://news.skhynix.com/en/groundbreaking-ceremony-in-indiana/ | Vendor | 2026-08-27/28 | Medium | **Negative evidence.** HBM manufacturing capacity news; no AiM/PIM content |
| 36 | SK hynix's technology roadmap for co-packaged optics features in 'Nature Electronics' | https://news.skhynix.com/en/cpo-in-nature-electronics/ | Vendor | 2026-08-20 | Low | **Negative evidence.** CPO/networking technology; no AiM/PIM content |
| 37 | news.skhynix.com/en/ newsroom index, re-checked for the full window | https://news.skhynix.com/en/ | Vendor index | fetched 2026-09-13 | High | **Negative evidence.** Every item dated 2026-08-08 → 2026-09-13 checked (DTF 2026, Future Forum, Hybrid Bonding tech note, Indiana groundbreaking ×3, CPO/Nature Electronics, share buyback, two AI Ecosystem/AI Infrastructure Insight posts, one analyst interview) — **none mention AiM, AiMX, or PIM** |

### Findings

- **No AiM/AiMX product, spec, or status change in the window.** AiMX remains a prototype per SK hynix's own last public statement (CES 2026); nothing in this scan updates that.
- **Hot Chips 38 Tutorial 1 ambiguity resolved:** SK hynix's talk is confirmed as "Advanced packaging for High Bandwidth Memory (HBM)" (Jaesik Lee) — packaging technology, correctly not treated as an AiM/PIM disclosure.
- **No SK hynix LPDDR-based PIM response to Samsung's HC38 "World's First" LPDDR5X-PIM disclosure was found** — at Hot Chips 38 or in SK hynix's own newsroom for the window.
- **No SK hynix HBM4 logic-base-die disclosure was found** — SK hynix's own Hybrid Bonding tech note describes HBM4 as still using conventional TCB packaging (hybrid bonding deferred to HBM4E/HBM5), with no base-die-logic-process or foundry-partner content. This is a direct contrast with Samsung's dedicated HC38 talk on exactly this topic ("HBM Base Die: How HBM Will Evolve Using Advanced Logic Processes").
- **Direction of travel reconfirmed:** SK hynix's August–September 2026 messaging (DTF 2026, Future Forum, Hybrid Bonding) continues to emphasize HBM/HBM4/HBM4E, packaging, and capacity (Indiana fab) — not AiM/PIM.
