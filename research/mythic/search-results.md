# Mythic — Search Results

*as_of: 2026-08-08*
*chip: mythic*
*device_class: Analog In-Memory Compute*

> **File origin note.** No `search-results.md` existed for Mythic before 2026-08-08 — the chip's 2026-04-05 baseline was written directly into `chips/mythic/` and `research/mythic/investigations/` without a resource index. This file starts with the 2026-08-08 roadmap scan. Baseline-era primary sources are listed first for completeness, taken from the `*primary sources:*` headers of the existing deliverables.

---

## Baseline-Era Sources (recorded retroactively, 2026-04-05 scan)

| Resource | Type | Why it matters |
|---|---|---|
| https://mythic.ai/technology/ | Vendor technology page | Analog compute-in-NOR-flash overview; ACE description |
| https://mythic.ai/products/m1076-analog-matrix-processor/ | Vendor product page | M1076: 76 tiles, 25 TOPS, 3 W, form factors |
| https://mythic.ai/technology/mythic-ai-workflow/ | Vendor workflow page | MAPP SDK pipeline: quantization → graph compile → flash program → run |
| https://fuse.wikichip.org/news/5727/mythic-rolls-out-m1000-series-analog-ai-accelerators-raises-70m-along-the-way/ | Trade analysis | M1108 / M1000-series architecture detail; tile internals |
| https://fuse.wikichip.org/news/2755/analog-ai-startup-mythic-to-compute-and-scale-in-flash/ | Trade analysis | Analog flash MAC mechanism; 0.25 pJ/MAC; ADC/DAC path |
| https://medium.com/mythic-ai/a-peek-into-software-engineering-at-mythic-1b0ca5522868 | Vendor engineering blog | Software-engineering practice behind the MAPP toolchain |

---

# Update Scan — 2026-08-08

**Change class:** roadmap (corporate + IP acquisition). **No silicon disclosed.** Two events postdate or were missed by the 2026-04-05 baseline: the Videantis acquisition (2026-05-19, genuinely new) and the Honda joint-development program (2026-02-04/06, a backfill that predates the baseline).

## New Primary Sources

| Resource | Type | Date | Why it matters |
|---|---|---|---|
| https://www.videantis.com/mythic-acquires-videantis.html | Vendor press release (full text) | 2026-05-19 | **The** primary source for the acquisition. BUSINESS WIRE, dateline "Palo Alto, CA and Hannover, Germany". Contains: Videantis GmbH identity, **v-MP6000UDX** VLIW+SIMD unified processor platform, ">25 million chips shipped", "zero field defects", "all top 3 European automotive manufacturers", "top three global semiconductor company", 20 mm × 20 mm AEB camera modules, European Innovation Council "top 1%", both Ozcelik quotes, the verbatim "from single camera drones, to factory robots, to autonomous platforms, all the way to data center deployments" range, **eCAPITAL becoming a Mythic shareholder**, wholly-owned-subsidiary structure, five founders joining under Dr. Hans-Joachim Stolberg, and the **Starlight** / **APU** boilerplate. Terms **not disclosed**. |
| https://global.honda/en/topics/2026/c_2026-02-04eng.html | Vendor newsroom (Honda's own) | 2026-02-04 | **Primary for the Honda program** — cite this date, not Mythic's 02-06. Co-development of a **system-on-a-chip for software-defined vehicles**; describes Mythic as "a Texas, U.S.-based technology company"; cites neuromorphic / analog compute-in-memory qualitatively. **States no TOPS figure, no efficiency multiple, and no timeline.** |
| https://mythic.ai/whats-new/honda-and-mythic-announce-joint-development-of-100x-energy-efficient-analog-ai-chip-for-next-generation-vehicles/ | Vendor press release | 2026-02-06 | Mythic's side of the Honda announcement. Adds that **Honda R&D licenses Mythic's Analog Processing Unit (APU) technology**; states the **"100×" efficiency** and **100,000+ TOPS** targets and **prototype vehicle testing in the late 2020s / early 2030s** with production after trials. **All quantitative figures originate here, not with Honda — attribute accordingly.** |
| https://mythic.ai/whats-new/ | Vendor news index | ongoing | Negative-evidence source. **Exactly three items** between Dec 2025 and 2026-08-08: $125 M round (2025-12-17), Honda (2026-02-06), Videantis (2026-05-19). **Nothing after 2026-05-19.** No chip name, no node, no new funding round. |
| https://www.taylorwessing.com/en/insights-and-events/news/media-centre/press-releases/2026/06/mythic-inc-acquires-videantis | Law-firm advisory notice | 2026-06-02 | **The only located source independent of both parties.** Taylor Wessing (advised Mythic) states the transaction was "successfully completed earlier this year following the satisfaction of all regulatory requirements" — i.e. **closed**, not a pending LOI. **Page returns HTTP 403 to direct fetch; content read via search-result title/snippet only.** Retry from another environment. |
| https://www.videantis.com/ | Vendor site | 2026 | Banner reads "videantis is now part of Mythic" — corroborates the acquisition from the target's own domain. |
| https://www.videantis.com/technology.html | Vendor technology page | — | Confirms Videantis licenses a proprietary processor platform to chip manufacturers (licensable soft IP). **Does not name v-MP6000UDX and does not name any SDK on that page** — relevant negative evidence for the software-stack update. |
| https://hotchips.org/advance-program/ | Conference program | Hot Chips 38, 2026-08-23…25, Stanford | **Mythic does not appear.** Full session/presenter list carries d-Matrix, Cerebras, SambaNova, Meta, Microsoft MAIA 200, Google TPU v8, OpenAI, NVIDIA, AMD, Intel, Samsung, SK Hynix, XCENA, BOS Semiconductors, Waymo, Oxmiq, Mojo Vision. |

## Secondary / Republication (attribute, do not treat as independent)

| Resource | Type | Date | Confidence | Notes |
|---|---|---|---|---|
| https://www.edge-ai-vision.com/2026/05/mythic-acquires-videantis-to-build-the-worlds-most-energy-efficient-ai-compute-platform/ | Trade republication | 2026-05-19 | low (not independent) | Corroborates Hannover HQ, v-MP6000UDX, 25 M chips, zero field defects, top-3 European automakers, price not disclosed, Honda Feb 2026 context — but is **verbatim BusinessWire copy**. |
| https://pulse2.com/mythic-videantis-acquisition-expands-hybrid-ai-compute-platform-with-100x-energy-efficiency-goal/ | Trade republication | 2026-05-20 | low (not independent) | Confirms deal value not disclosed, five founders joining, wholly owned subsidiary. **Explicitly contains no independent reporting.** |
| https://www.businesswire.com/news/home/20260519255958/en/Mythic-Acquires-Videantis-One-of-Europes-Leading-Digital-Processor-IP-Companies-To-Build-the-Worlds-Most-Energy-Efficient-AI-Compute-Platform | Wire copy | 2026-05-19 | — | The origin release itself (indexed). |
| https://www.businesswire.com/news/home/20260206468111/en/Honda-and-Mythic-Announce-Joint-Development-of-100x-Energy-Efficient-Analog-AI-Chip-for-Next-Generation-Vehicles | Wire copy | 2026-02-06 | — | Honda/Mythic release (indexed; direct fetch timed out). |
| https://convergedigest.com/honda-and-mythic-to-co-develop-analog-ai-soc/ | Trade | 2026-02 | low | Corroborates vehicle testing late 2020s, production early 2030s. Fetch returned **403**; read via search snippet. |
| TMCnet / AFP wire feed / hw.dev copies of the acquisition release | Republication | 2026-05 | none | Verbatim BusinessWire copy. Listed only so a future scan does not mistake volume of hits for corroboration. |

## Newly Named Entities and Terms to Track

- **Videantis GmbH** — Hannover, Germany; licensable digital processor-IP vendor; now a wholly owned Mythic subsidiary.
- **v-MP6000UDX** — the acquired "unified processor platform": VLIW + SIMD array of identical cores spanning DL inference, classical CV, signal/image processing, and video encode/decode. **No node, core count, clock, area, throughput, or data types disclosed.**
- **Dr. Hans-Joachim Stolberg** — Videantis co-founder leading the five founders joining Mythic.
- **eCAPITAL** — European deep-tech VC that became a Mythic shareholder via the transaction (cap-table change, **not** a priced round).
- **Analog Processing Unit (APU)** — Mythic's 2026 umbrella term for its analog parts, superseding "AMP" in current copy; the technology Honda R&D licenses.
- **Starlight** — Mythic analog co-processors intended for embedding inside sensors. **Edge, not datacenter.** No specs published.

## Negative Findings (record so the next scan does not re-litigate them)

1. **No datacenter product name, process node, TOPS, TDP, tapeout, sampling, or shipping** for any forthcoming Mythic part.
2. **No new funding round** after the Dec 2025 $125 M — but note the eCAPITAL cap-table change.
3. **No merged SDK, ISA, IR, compiler, or kernel API** for a hybrid analog + digital platform.
4. **The "750× tokens/s/W on 1T models" claim remains unsubstantiated** — no retrieved source repeats or supports it. Keep flagged.
5. **No trade-press independent reporting** on the acquisition; no deal value, no headcount, no roadmap detail.
6. **Mythic is absent from Hot Chips 38.**

## Open Items for the Next Scan

1. Any Mythic news after **2026-05-19** — the index has been static since.
2. First **hybrid analog+digital part** disclosure: name, node, engines, memory hierarchy, fabric.
3. Any **merged toolchain** announcement folding v-MP6000UDX into MAPP.
4. Whether the **datacenter** intent ever attaches to a named product.
5. **Deal value / headcount / roadmap detail** for Videantis — watch for a filing or a genuine trade-press follow-up.
6. Retry **taylorwessing.com** (currently 403) and **convergedigest.com** (403) from an unblocked environment.
7. Resolve the **HQ** question: Austin, TX (repo) vs. "Texas, U.S.-based" (Honda) vs. "Palo Alto, CA" (Mythic's own 2026 dateline).
