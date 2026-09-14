# Tesla FSD Chip — Summary

**Device class:** Edge Inference SoC
**Manufacturer:** Tesla (in-house design), fabricated by Samsung
**Deployment:** Automotive edge (Tesla vehicles, Cybercab Robotaxi, Optimus robot)
**Research date:** 2026-04-05
**Last updated:** 2026-09-13 — see the "Tesla FSD 2026-09-13 Update" section at the end of this file (prior: "Tesla FSD 2026 Update", 2026-08-08)

---

## What It Is

The Tesla FSD (Full Self-Driving) chip is a purpose-built ASIC SoC for real-time autonomous driving inference at the vehicle edge. It is not a general-purpose AI accelerator — it runs Tesla's proprietary neural networks on a fixed-function NPU with no public programming interface.

Two generations are deployed at volume:
- **HW3 (2019):** Samsung 14 nm, 2 NPUs, 73.7 TOPS INT8
- **HW4 / AI4 (2023):** Samsung 7 nm, 3 NPUs, ~121 TOPS INT8, 3.3× memory bandwidth improvement

Three further variants entered the public record after the original research date and are covered in the 2026
update section below: **AI4.5 / "AP45"** (shipping in some vehicles, never formally announced), **AI4.1 /
"AI4 Plus"** (announced 2026-04-22, no silicon), and **AI5** (tape-out announced 2026-04-15, one packaged
engineering sample shown, dual-sourced at TSMC and Samsung). AI4 remains the volume shipping baseline as of
2026-08-08; AI4.1 and AI5 are not deployed.

---

## Key Architecture Features

| Feature | HW3 | HW4 |
|---|---|---|
| NPUs | 2 | 3 |
| TOPS (INT8) | 73.7 | ~121.6 |
| MAC array | 96×96 per NPU | 96×96 per NPU |
| NPU SRAM | 32 MiB per NPU | 32 MiB per NPU |
| DRAM | 8 GB LPDDR4, 68 GB/s † | 16 GB GDDR6, 224 GB/s † |
| Board TDP | ~100 W | ~160 W |
| Process | Samsung 14 nm | Samsung 7 nm |

† **Memory-bandwidth figures are contested and none is Tesla-published.** The values above are WikiChip's
per-SoC figures. Electrek (2026-04) instead reports ~48 GB/s for HW3 and ~384 GB/s of GDDR6 bandwidth for AI4
with 32 GB total across the dual-SoC board (16 GB per SoC). The two sets are mutually inconsistent and cannot be
reconciled from public data — they may differ in per-SoC vs per-board accounting, or one may simply be wrong.
Both are reported here with attribution rather than picking a winner. See the 2026 update section.

---

## Software Stack

- **Training:** PyTorch on Dojo/GPU cluster
- **Compiler:** Proprietary two-pass NN compiler (closed source)
  - Coarse topology mapping → fine INT8 quantization + static schedule
- **Runtime:** Static microcode scheduler (compile-time scheduled, no dynamic dispatch)
- **OS:** Embedded Linux on ARM clusters
- **Deployment:** OTA binary updates (chip-specific binaries)

---

## Distinguishing Design Choices

1. **Static scheduling** — no runtime dispatch, deterministic latency (safety critical)
2. **All-SRAM weight buffer** — 32 MiB per NPU avoids DRAM weight reads during inference
3. **INT8 throughout** — 4× memory savings, full model fits in SRAM
4. **Dual-chip redundancy** — two FSD computers per vehicle, cross-validate
5. **Closed ecosystem** — no third-party access; entire stack from training to deployment is Tesla-internal

---

## Limitations

- Not programmable by customers or researchers
- No public SDK or compiler
- Limited transistor/area detail (no die shot measurements published)
- HW3 reaching limits: FSD v13 required extreme quantization compression. As of 2026-04-22 Tesla has publicly
  conceded HW3 cannot reach unsupervised FSD at all and has committed to a retrofit programme (see 2026 update).
- Memory bandwidth for both deployed generations is known only from third-party estimates that disagree with
  each other; Tesla has never published a bandwidth figure.

---

## Sources

- https://fuse.wikichip.org/news/2707/inside-teslas-neural-processor-in-the-fsd-chip/
- https://en.wikichip.org/wiki/tesla_(car_company)/fsd_chip
- https://www.autopilotreview.com/tesla-hardware-4-rolling-out-to-new-vehicles/
- https://arxiv.org/html/2411.16007v1

---

## Tesla FSD 2026 Update — AI4.5, AI4.1/AI4 Plus, AI5

*Updated 2026-08-08. Prior-generation content above is unchanged. Sources: Musk X post and follow-on coverage
(2026-04-15), Tesla Q1 2026 earnings call (2026-04-22), Electrek (2026-01-26, 2026-04-15, 2026-04-22, 2026-04-23),
Korea Times / UPI / Asiae / TechTimes / XenoSpectrum (2026-07-13/14), Tom's Hardware, DigiTimes, Benzinga.*

**Evidence health warning for this whole section.** Tesla publishes no datasheets, gives no conference talks on
FSD silicon, and files no MLPerf results. Everything below traces to (a) Musk statements on X or on earnings
calls, (b) one photograph of one packaged part, (c) a since-deleted LinkedIn post by a Samsung engineer, or
(d) owner/firmware sightings. **No peak TOPS, FLOPS, dtype list, TDP, die area, memory capacity or memory
bandwidth has been disclosed for AI5.** Where this section gives a number, it says who said it.

### Status board (as of 2026-08-08)

| Variant | Status | First public evidence | Silicon exists? | In vehicles? |
|---|---|---|---|---|
| HW3 | Deployed since 2019; **declared insufficient for unsupervised FSD 2026-04-22** | — | Yes | Yes (~4M cars) |
| HW4 / AI4 | Deployed since 2023; current shipping baseline | — | Yes | Yes |
| AI4.5 / "AP45" | Shipping in some vehicles, **never formally announced by Tesla**. Update 2026-09-13: JPMorgan-via-Electrek attributes "~10% compute, ~2× memory" to "AI4.5" — thirdhand, possibly a mislabeling of AI4.1 (see below) | 2026-01-26 (Electrek, owner sightings) | Yes (part 2261336-02-A) | Yes (2026 Model Y, Fremont) |
| AI4.1 / "AI4 Plus" | **Announced only**, no silicon | 2026-04-22 (Q1 2026 earnings call) | No | Target 2027 production |
| AI5 | **Taped out; one packaged engineering sample shown; delayed to mid-2027 (reaffirmed 2026-08-20)** | 2026-04-15 (Musk X post) | Yes (engineering sample) | No — volume targeted mid-2027; Cybercab originally planned on AI4 |
| AI6, Dojo 3 | Named as "in work"; nothing else public | 2026-04-15 (same X post) | Not disclosed | No |

### AI5 — tape-out announced 2026-04-15

On 2026-04-15 Musk posted on X: *"Congrats to the @Tesla_AI chip design team on taping out AI5! AI6, Dojo3 &
other exciting chips in work,"* alongside a photograph of a packaged AI5 sample; follow-on coverage notes he also
thanked the foundries, mistyping TSMC's handle as "TSC."

- **2026-04-15 is the announcement date, not the tape-out date.** A physically packaged part already existed when
  the post went up, so tape-out necessarily predates it. **Tesla has never published a tape-out date.**
- A reported package date code ("KR 2613," read as assembly week 13 of 2026) appears in **a single outlet**
  (Tom's Hardware) and is not independently corroborated — low confidence.
- The announcement itself is independently confirmed by Electrek (2026-04-15), MarketWatch/Morningstar
  (2026-04-15), Benzinga (2026-04-17) and DigiTimes (2026-04-17).

**Package (observed from one photograph; vendor-unconfirmed).** Multiple independent outlets describe:

| Observation | Confidence | Note |
|---|---|---|
| Compute ASIC die roughly **half a standard reticle** | Medium — multi-outlet, photo-derived; matches Musk's own "half a reticle with good margin" remark | Not a measured die area |
| **Twelve SK hynix memory packages** ringing the die | Medium — Benzinga, DigiTimes, VideoCardz, Tom's Hardware; DigiTimes names SK hynix as supplier | Package count from a photo |
| **Discrete DRAM on an organic substrate — GDDR6 or GDDR7, not HBM/CoWoS** | Medium | Grade (GDDR6 vs GDDR7) not established by any source |
| 384-bit external bus; ~768 GB/s – 1.536 TB/s | **Low — single-source inference** | Tom's Hardware's arithmetic from counting packages. **Not a Tesla spec; do not cite as one.** |

**Performance: nothing is disclosed.** The only public figures are Musk marketing claims, and they do not agree
with each other:

| Claim | Where | Notes |
|---|---|---|
| "up to 40×" / "40 times" better than AI4 | Q3 2025 earnings call (2025-10-22/23), repeated in 2026 coverage | Originates October 2025, not April 2026. Largest of the circulating figures. |
| "up to 10×" more powerful than AI4 | Musk, via Electrek | Conflicts with 40× |
| "maybe by a factor of 10" on performance per dollar | Musk, via Electrek | Different metric again |
| "5× the memory bandwidth of AI4" | Musk, via Electrek | Only bandwidth-specific claim; unquantified in absolute terms |

Citing 40× alone cherry-picks the largest number. Treat the whole cluster as **unverified vendor marketing**,
not as a spec. Musk also stated AI5 integrates the Arm CPU cores and PCIe blocks alongside the accelerators.

**Dual-foundry sourcing — disclosed 2026-07-13, tape-out dates not stated.** A Samsung Foundry principal
engineer (James Kim) posted on LinkedIn on or about 2026-07-13 that *"the Tesla-Samsung AI5 chip has reached
tape-out"* and would be *"manufactured at the Taylor fab using our latest 2nm process."* **The post was
subsequently deleted, Samsung declined to comment, and there is no official Samsung or Tesla confirmation.**
Reported independently by Korea Times, UPI, Asiae, TechTimes and XenoSpectrum (2026-07-13/14).

- AI5 is therefore a **dual-source, dual-node part: TSMC (node not disclosed by anyone) plus Samsung
  SF2-class 2 nm at Taylor, Texas.** The specific Samsung derivative (SF2 / SF2P / SF2T) is **not disclosed**.
- **Do not claim the Samsung implementation taped out months after the TSMC one.** Korea Times reports both
  foundries received the design in April 2026; no source gives either tape-out date. The gap may be weeks.
- Dual-sourcing itself is well supported and predates 2026 — Musk had engaged both foundries for AI5 by
  October 2025.
- Headlines stating Samsung production "starts soon" or that AI5 "is about to enter mass production" overstate
  a deleted LinkedIn post.

**Correct status verb: taped out, with a first packaged engineering sample shown.** Not sampling to third
parties, not shipping, not deployed. Volume manufacturing is targeted for **2027** (Electrek: mid-2027 for
automotive, requiring several hundred thousand completed AI5 boards line-side; MarketWatch: volume
manufacturing 2027).

**Target market is unsettled and sources conflict — the survey does not pick one.** On the Q1 2026 earnings
call (2026-04-22) Musk said AI5 goes into Optimus and the data center *"because it's looking like we'll be able
to achieve unsupervised self-driving with AI4 that is far greater than human safety levels,"* and added that
AI5 may eventually go into cars but there is no near-term cost or urgency reason to switch. July 2026 Samsung
coverage instead describes AI5 as destined for Tesla vehicles, robots and data centers, and April coverage lists
vehicles, Optimus and xAI data centers. **The accurate framing: near-term AI5 priority is Optimus and
datacenter/inference clusters, with automotive use deferred rather than excluded.** Musk did *not* say AI5 will
not go into cars.

Architecturally, AI5 is the first Tesla FSD-line part that is **not primarily an automotive edge SoC** — a
half-reticle die with twelve discrete DRAM packages and integrated PCIe is a datacenter/robotics inference part
in packaging terms, even though its compute microarchitecture is entirely undisclosed.

### AI4.1 / "AI4 Plus" — announced 2026-04-22, no silicon

Announced on the Q1 2026 earnings call. Musk, verbatim: *"We are planning an AI4 upgrade to use newer generation
RAM. So it'll go from 16 gigabytes to I think 32 gigabytes per SoC. So 64 gigabytes total, and probably a 10%
increase in compute and in memory bandwidth."* He called it "AI4.1 or AI4 Plus."

| Parameter | AI4 (HW4) | AI4.1 / AI4 Plus (announced) |
|---|---|---|
| DRAM per SoC | 16 GB | **32 GB** ("newer generation RAM") |
| DRAM per board (dual SoC) | 32 GB | **64 GB** |
| Compute | baseline | **~+10%** (Musk, unqualified) |
| Memory bandwidth | baseline | **~+10%** (Musk, unqualified) |
| Process | Samsung 7 nm | Samsung 7 nm, **modified** — production gated on Samsung completing process modifications |
| Production | shipping | **"next year" = 2027** |

This is a **memory-and-clock refresh, not a new microarchitecture** — no NPU count, MAC array, ISA or dtype
change was described. Status: announced only; no silicon, no tape-out claim. (One aggregator live blog garbled
the production year as "mid-2025"; that is a transcription error — 2027 is correct.)

### AI4.5 / "AP45" — reported 2026-01-26, never announced

Owners of Fremont-built 2026 Model Y AWD Premium vehicles delivered December 2025 – January 2026 found FSD
computers labelled **"AP45" / "AP4.5," part number 2261336-02-A**, matching Tesla's Electronic Parts Catalog.
Tesla has made no statement about it.

The widely repeated **"three-SoC instead of AI4's two-SoC board" detail is low confidence**: it is a
firmware-analysis inference by @greentheonly reported by Electrek and explicitly hedged as *"may feature"* — it
is not a teardown die count and not a Tesla statement. No compute, memory or process figures exist for AI4.5.

### HW3 formally declared insufficient — 2026-04-22

On the Q1 2026 call Musk conceded that millions of HW3 vehicles will not receive unsupervised FSD, and that
Tesla will build dedicated "micro factories" to retrofit them. Independently confirmed by The Verge and Electrek
(both 2026-04-22).

Two widely quoted lines — HW3 *"simply does not have the capability"* and HW3 having *"only one-eighth the
memory bandwidth of HW4"* — **trace to Electrek's write-up rather than to a verifiable transcript, and are
attributed to Electrek here, not to Tesla.** Note that the one-eighth ratio is internally consistent only with
Electrek's own 48 GB/s ÷ 384 GB/s pair, not with the WikiChip 68 / 224 GB/s pair used in the table above — a
further reason to treat all FSD bandwidth figures as contested.

Litigation followed: a US class action (2026-06-29) and a Dutch collective action (2026-06-18).

### Ecosystem: unchanged

- **No Tesla talk in the Hot Chips 38 advance program** (conference runs 2026-08-23 to 2026-08-25; Waymo, not
  Tesla, holds the autonomous-driving keynote and the automotive SoC slot). Verified against hotchips.org.
- No MLPerf submission, no public SDK, no ISA disclosure for any generation beyond HW3/HW4.
- The "closed ecosystem" characterization in this summary continues to hold in full.

### Update sources

- https://electrek.co/2026/04/15/tesla-ai5-chip-taped-out-musk-ai6-dojo3/
- https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-demonstrates-first-sample-of-tesla-ai5-processor-accidentally-thanks-tsc-rather-than-tsmc-claims-40x-performance-boost-over-the-predecessor
- https://www.tomshardware.com/tech-industry/artificial-intelligence/teslas-ai5-with-2nm-class-node-tapes-out-at-samsung-foundry-production-starts-soon-months-after-tsmc-tape-out
- https://www.koreatimes.co.kr/business/tech-science/20260713/samsung-taylor-fab-enters-production-phase-for-teslas-ai5-chip
- https://www.upi.com/Top_News/World-News/2026/07/13/samsung-tesla-foundry-texas-ai5-chip/8251783995948/
- https://xenospectrum.com/en/tesla-ai5-samsung-2nm-tapeout-dual-foundry/
- https://www.techtimes.com/articles/320427/20260714/tesla-ai5-locks-samsung-2nm-taylor-flipping-node-assumption.htm
- https://electrek.co/2026/04/23/tesla-hw4-plus-upgrade-will-hw4-follow-hw3/
- https://electrek.co/2026/01/26/tesla-quietly-starts-shipping-model-y-with-new-ai4-5-computer/
- https://www.digitimes.com/news/a20260417VL202/tesla-ic-design-development-sk-hynix.html
- https://www.benzinga.com/markets/prediction-markets/26/04/51885219/elon-musk-shows-off-first-physical-tesla-ai5-chip-tsla-up-14-in-5-days
- https://www.tweaktown.com/news/108461/elon-musk-says-teslas-next-gen-ai5-chip-is-40x-faster-than-ai4-calls-it-a-beautiful-chip/index.html
- https://www.autoevolution.com/news/tesla-hypes-ai5-autopilot-computer-as-neglected-hw3-owners-grow-angrier-259718.html
- https://www.morningstar.com/news/marketwatch/20260415530/mw-is-tesla-a-chip-stock-now-investors-are-cheering-a-semiconductor-milestone
- https://hothardware.com/news/elon-musk-taps-samsung--tsmc-for-teslas-next-gen-ai5-chip
- https://www.hotchips.org/advance-program/

---

## Tesla FSD 2026-09-13 Update — AI4.5 spec claim, AI5 delay, HW3 "v14 Lite"

*Updated 2026-09-13. Scan window 2026-08-08 → 2026-09-13. Classification: **Major**, heavily hedged. WebSearch
was unavailable this session (budget exhausted); this update relies on a single Electrek article (2026-08-20)
reporting a JPMorgan analyst note (analyst Rajat Gupta) written after a Fremont factory visit/briefing — i.e.
Tesla → JPMorgan → Electrek, thirdhand and not independently corroborated by a second outlet.*

**AI4.5 — first figures attributed to this name, likely conflated with AI4.1.** Electrek, paraphrasing the
JPMorgan note: *"Tesla is rolling out AI4.5, a new version of the chip with roughly 10% more compute and about
twice the memory."* This wording is nearly identical to Musk's 2026-04-22 description of **AI4.1/"AI4 Plus"**
("probably a 10% increase in compute and in memory bandwidth," 16→32 GB/SoC). This survey has tracked AI4.5/AP45
and AI4.1/AI4-Plus as **separate** variants; the new figures may (a) be a genuine AI4.5 spec, or (b) be a
mislabeling of the already-known AI4.1 figures by Electrek/JPMorgan. **Both readings are recorded; neither is
adopted as a confirmed spec.**

**AI5 delay to mid-2027 — reaffirmed, not new.** Electrek/JPMorgan: *"Tesla delayed its next-gen AI5 chip to
mid-2027."* This matches Electrek's own April 2026 "mid-2027 for automotive" estimate already in this survey;
the new element is the explicit "delay" framing, not a new date.

**Cybercab hardware target — first explicit statement found.** Per the same note, Cybercab was **originally
planned to launch on AI4** hardware — sharpening (not contradicting) the existing "AI5 near-term priority is
Optimus/datacenter, automotive deferred" finding.

**HW3 retrofit reality — new concrete detail.** HW3 vehicles now receive a stripped-down **"FSD v14 Lite"**
release rather than full FSD, with reported **rising HW3 hardware failure rates**. This continues, with a new
symptom, the 2026-04-22 "HW3 declared insufficient" finding already in this survey.

**Context only:** Tesla told JPMorgan that FSD v15 is "a major jump in capability, built on seven 'core
technologies,'" ~40% already operational in the Austin robotaxi fleet, and reasserted HW4/AI4 is sufficient for
v15/unsupervised FSD — the same claim made and later walked back for HW3, a parallel Electrek itself draws.

**Checked, no change found:** no primary Tesla source (earnings call, filing, datasheet) issued new AI4.5/AI5
specs this window. Hot Chips 38 (Aug 23–25, 2026) fell within this window; the prior "no Tesla talk" finding was
not re-verified against a post-event archive this cycle. No MLPerf submission, no public SDK, no ISA disclosure.

Source: https://electrek.co/2026/08/20/tesla-jpmorgan-fremont-fsd-v15-hw4-optimus-2027/
