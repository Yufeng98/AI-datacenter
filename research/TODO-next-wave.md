# Next Wave — Investigation Queue

*Created: 2026-04-05*
*Last scan: 2026-09-13 (`update-chip-landscape`, mode full)*

## Status of the 2026-04-05 Wave — COMPLETE

All 22 chips queued on 2026-04-05 have been investigated and have full deliverables under
`chips/<chip>/` and `research/<chip>/`. See `research/status.yaml` for per-chip phase state.

## Added 2026-08-08 (8 chips)

Discovered by the 2026-08-08 landscape scan, adversarially verified, and approved for investigation.
All eight now have complete deliverables.

| Chip | Device Class | Why it was added |
|------|-------------|------------------|
| ibm-spyre | Inference Accelerator (SIMD-systolic, scratchpad-managed) | Shipping since 2025-10-28 on z17/LinuxONE 5 with an ISSCC 2026 paper. IBM had **zero** registry presence — the largest single coverage gap found. |
| nextsilicon-maverick | Reconfigurable Dataflow (runtime-JIT spatial grid) | Distinct programming model: unmodified C/C++/Fortran lowered to a spatial grid, then continuously re-mapped at runtime by a telemetry loop. No kernel language at all. |
| spinncloud-spinnaker2 | Event-Driven Neuromorphic Manycore | Only datacenter-scale neuromorphic system actually deployed (Sandia NNSA, TU Dresden). Fills a paradigm the registry described but did not cover. |
| preferred-networks-mn-core | Compiler-Scheduled SIMD (Japan) | The purest compiler-scheduled point in the registry — no hardware cache **and** no hardware scheduler. Strong contrast case for the programming-model comparison. |
| tsingmicro | Reconfigurable Dataflow / CGRA "RPU" (China) | Only CGRA in the registry; Tsinghua reconfigurable-computing lineage. |
| stream-computing | Programmable NPU — RISC-V + vector/matrix ISA ext. (China) | RISC-V ISA-extension approach to NPU design, documented in a CARRV 2021 paper. |
| tecorigin | Heterogeneous Many-Core, SPA/SPE + SPM (China) | Sunway-lineage master-slave architecture — architecturally unlike every other China entry. Upstream PaddlePaddle CI backend independently corroborates working silicon. |
| vastaitech | GPU-like Inference + Video Codec (China) | Codec-integrated inference is a distinct datacenter design point. |

## Queued — Investigable, Deferred (12)

Verified as real and datacenter-relevant but not approved in the 2026-08-08 wave. Ready to promote at any time.

| Chip | Company | Device Class | Maturity | Note |
|------|---------|-------------|----------|------|
| xcena-mx1 | XCENA (Korea) | CXL computational memory / near-data processing | sampling | Not an accelerator — a CXL Type 3 memory device whose controller embeds NDP cores. Hot Chips 38 talk scheduled. Decide whether the registry covers near-data memory devices. |
| hyperaccel | HyperAccel (Korea) | LLM-specific dataflow "LPU" | sampling | Peer-reviewed architecture (ISCA/HPCA lineage). FPGA product ships today; the Samsung 4nm ASIC is announced only. HyperDex stack is documented but closed. |
| zhonghao-xinying | 中昊芯英 (Hangzhou) | Systolic array / TPU-like "GPTPU" | deployed | Deployed at 1,024-chip scale at China Telecom/Tianjin Mobile, but **no public SDK, compiler, or ISA docs** — cannot run the software half of the pipeline. Chip name is 刹那 "Chana", not "Chala". |
| axelera-ai | Axelera AI (Netherlands) | Digital in-memory compute (D-IMC) | split | Metis ships but is edge-class. Datacenter part **Titania** is announced only, targeting 2028. Revisit when Titania tapes out. |
| lightelligence | Lightelligence / 上海曦智 | Photonic compute + optical interconnect | split | Optical interconnect ships; optical compute (PACE 2) is pilot-only, ~US$2.3M FY2025 revenue. HKEX-listed as a Chapter 18C pre-commercial company. PACE 3 is pre-silicon. |
| celestial-ai | Celestial AI → Marvell | Optical interconnect / optical memory fabric | acquired | Acquisition closed 2026-02-02. Interconnect, not compute — belongs in a fabric section if one is ever added, not the compute registry. |
| fujitsu-monaka | Fujitsu (Japan) | Armv9-A datacenter CPU with SVE2 | sampling | A CPU, not an accelerator. See the scope question below. |
| edgecortix-sakura | EdgeCortix (Tokyo) | Runtime-reconfigurable dataflow | production | Primarily edge; confirm a datacenter SKU exists before promoting. |
| li-auto-m100 | Li Auto 理想汽车 (China) | Orchestrated dataflow | production | Automotive-first, like tesla-fsd. Promote only if the survey keeps covering edge inference SoCs. |
| mobilint | Mobilint (Korea) | NPU (edge / on-prem inference) | production | Edge-class; likely out of scope. |
| extropic-tsu | Extropic (US) | Thermodynamic / probabilistic sampling | taped-out | Genuinely novel paradigm, real test silicon. Not a conventional accelerator — no framework story yet. |
| lumai | Lumai (UK, Oxford spinout) | Free-space optical matrix accelerator | unknown | Low confidence; re-verify before promoting. |

## Watch List — Announced or Taped Out, Not Yet Investigable

Not enough public technical documentation to run the pipeline. Re-check each scan.

**Hyperscaler / lab custom silicon:** anthropic-silicon, xai-x1, bytedance-asic — all announced only, no disclosure.

**Novel paradigms:** taalas (model weights cast into metal; **AMD announced acquisition 2026-08-06**),
normal-computing-cn101 (thermodynamic), snowcap-compute (superconducting), vaire-computing (reversible/adiabatic),
optalysys (photonic compute-in-transit), olix-dx1, ibm-northpole.

**Inference ASICs:** matx-one, positron-asimov, vsora-jotunn8, tensordyne-napier (logarithmic number system),
neureality-nr1, euclyd-craftwerk.

**In-memory compute:** fractile, encharge-ai (charge-domain analog), sagence-ai, houmo-ai (China), moffett-ai (dual-sparsity).

**China:** denglin, corerain (CAISA dataflow).

**RISC-V / IP:** rivos, sifive-datacenter-ai, semidynamics-cervell, tachyum-prodigy.

**Korea:** panmnesia (CXL fabric), deepx (edge).

**Other:** mediatek-madic (diffusion accelerator), arm-agi-cpu.

## Existing Chips — Update Status

All 42 previously-investigated chips were re-scanned on 2026-08-08. 39 had material changes and were updated;
3 (untether-ai, rain-ai, luminous) were corrected to **defunct** status. The 2026-04-05 `huawei-ascend`
pending item is resolved — Ascend 950 is documented.

## Open Scope Questions for the Maintainer

Raised by the 2026-08-08 completeness critic. These are **scope decisions, not research gaps** — no action taken.

1. **Datacenter CPUs are 0 of 57 rows.** Hot Chips 38 devotes two sessions to CPUs with AI acceleration
   (IBM Z + AI inference chipset, NVIDIA Vera, Arm AGI, Fujitsu MONAKA, Intel Diamond Rapids). If the survey's
   subject is "the stack you program to run a model", host CPUs may belong.
2. **Interconnect / scale-up fabric has no registry representation.** `stack-comparison-by-layer.md` has
   scale-up and scale-out rows, but Broadcom (Thor Ultra, Tomahawk/Jericho), NVIDIA BlueField-4/Spectrum-X,
   UALink, ESUN, Astera Labs, and Credo appear nowhere. Qualcomm's AI300 explicitly commits to ESUN.
3. **MLPerf is not tracked as a category.** It is the only apples-to-apples cross-vendor evidence available,
   and submission tables reveal which software stacks are actually functional per accelerator.
   MLPerf Training v6.0 landed 2026-06-16; Inference v6.1 submissions opened 2026-07-09.
4. **Memory-tier innovations** (HBF, LPDDR-PIM, 3D-DRAM, CXL computational memory) are surfacing as compute
   architecture, blurring the accelerator/memory boundary the registry currently assumes.

## Update — 2026-09-13 (mode full)

Scan window 2026-08-08 → 2026-09-13, covering Hot Chips 38 (Aug 23–25, 2026) plus every other source
type (vendor press releases, SDK/GitHub releases, exchange filings, analyst coverage, papers). All 50
registry chips were re-checked; sources were not restricted to Hot Chips.

**Chips with confirmed Hot Chips 38 technical disclosures, now reflected in the corpus:** nvidia-gpu
(Rubin, Vera CPU, Groq 3 LPX/LPU rack specs — first real numbers for a component that had zero), amd-gpu
(MI455X/Helios — MXFP6/FP8 peak now confirmed, `gfx1250` ISA target now public, rack physical format
confirmed), google-tpu (TPU 8t "Sunfish" / 8i "Zebrafish" — HBM stack counts, superpod/fabric bandwidth),
cerebras (WSE-3T / CS-4 "Nexus" — new row, ~2× clock over WSE-3, wafer-to-wafer interconnect), sambanova
(SN50 — HBM generation and network fabric now confirmed), meta-mtia (MTIA 300 process node, MTIA 400 —
first specs at all, new row), microsoft-maia (Maia 200 — first-ever rows in the comparison tables, sourced
to a genuine Microsoft-primary arXiv paper, arXiv:2608.24664), intel-gaudi (Crescent Island — Xe3P core/
cache counts, LPDDR5X capacity range), ibm-spyre (a distinct, pre-announced "AI Inference Acceleration
Chipset" — new row, not conflated with shipping Spyre), samsung-aquabolt-pim (LPDDR5X-PIM — new row with
real bandwidth/capacity numbers), d-matrix (Raptor 3D-DRAM accelerator — new row, distinct from Corsair),
groq (cross-referenced to the NVIDIA-fabbed LPX hardware; GroqCloud's own $350M round and NVIDIA-partner
status recorded separately).

**Chips with material non-Hot-Chips changes:** etched-sohu (first customer delivery — Jane Street,
$700M Series D at $21B), tesla-dojo (a ~$2B "unnamed AI hardware company" acquisition backfilled from a
10-Q), tesla-fsd (AI5 delay to mid-2027 reaffirmed), qualcomm (first confirmed AI200/Dragonfly production
deployment — Adobe on HUMAIN; a new Qualcomm–AWS silicon collaboration), rebellions-atom (NVIDIA reported
in early talks with Rebellions), huawei-ascend (DeepSeek reportedly ordering 160,000+ Ascend 950DT
accelerators), enflame (STAR Market IPO debut, +188–234%), several other Chinese vendors' H1 2026
financials and IPO/funding status (cambricon, biren, muxi, hygon-dcu, mthreads, xiwang, kunlunxin,
tianshu-zhixin). sk-hynix-aim, stream-computing, graphcore, esperanto, luminous, mythic, rain-ai,
untether-ai, q-ant, and spinncloud-spinnaker2 had no material change in the window.

**Data-quality note:** MLPerf Inference v6.1 is **not yet published** as of 2026-09-13 (latest is v6.0,
2026-04-01) — the "submissions opened 2026-07-09" line above refers to the submission window only, not
results; do not report v6.1 results until they actually exist.

## New Chips Discovered (2026-09-13) — Awaiting User Approval

Four parallel discovery agents ran the full 41-query sweep (news, conferences, architecture type, every
region, market segment, analyst coverage, curated lists), explicitly not limited to Hot Chips. This list
is curated down from ~35 raw hits to the candidates with genuine technical substance; per the skill's
Mode 1 rule, **none of these have been investigated — approval is required before any pipeline work
starts.**

| # | Chip | Company | Device Class | Maturity | Public docs | Add? |
|---|------|---------|-------------|----------|--------------|------|
| 1 | Napier | Tensordyne (US; formerly Recogni) | Log-number-system inference ASIC | **Taped out** TSMC 3nm, whitepaper published 2026-09-03 | Whitepaper (tensordyne.ai), 138B transistors, 2.1 PFLOPS FP8 disclosed | ? |
| 2 | Atlas / Asimov | Positron (US) | SRAM-heavy inference accelerator | Atlas **in production** at OCI (>50 racks); Asimov tapes out end-2026 | positron.ai product pages; $875M Series C (2026-09-11) | ? |
| 3 | Epoch series | EVAS Intelligence (China) | RISC-V/TPU-style train+infer accelerator | Vendor claims mass production / "large-scale deployment" | Co-authored ISCA 2026 paper (TISA scheduling); no vendor whitepaper found | ? |
| 4 | DF1000 | Dongfang Suanxin "Orient Silicon" (China, Shanghai) | 3D hybrid-bonded near-memory LLM-decode chip | Taped out, 128-card cluster demonstrated | Press only (CGTN, SCMP); **source disagreement on the company's own Chinese name across three outlets** — verify before adding |
| 5 | Homodyne photonic crossbar | Opticore (US, Berkeley/MIT) | Photonic tensor processor | Research prototype (2025 demo tape-outs); HC38 poster | arXiv:2604.18496 + Hot Chips 38 poster — best-documented photonic candidate found | ? |
| 6 | DX-1 | Olix (US) | SRAM-only inference accelerator | Pre-silicon; first customer delivery H2 2027 | Vendor site only; $312M Series B at $3.3B (2026-08-03) | ? |
| 7 | T100 OPU | Neurophos (US, Austin) | Metamaterial photonic OPU | Pre-silicon; systems targeted ~2028 | Whitepaper referenced on vendor site; $110M Series A | ? |

Rows 1–2 clear the skill's "at least sampling, with public technical documentation" bar cleanly. Rows
3–7 are weaker on one axis each (thin vendor documentation, unresolved naming, or pre-silicon status) —
flagged rather than pre-filtered, since maturity often changes fast for these.

**Not brought forward** (real financing/press activity, but too early or out of scope): Unconventional AI
($475M seed, no product), Majestic Labs (Prometheus server, tape-out still pending), Jiangyuan Technology
D20 (mass-production claim but zero public technical documentation), Arago JEF (test chip only), Openchip
BER10 (a CPU+vector core today; AI accelerators are roadmap-only), Neuchips Raptor N3000 (shipping but
edge/on-prem class, not datacenter-scale), GSI Technology Gemini (edge-focused per the vendor's own 2026
strategy). Full detail on all ~35 raw candidates from this scan is preserved in the session's working
notes if any of these should be revisited.

**Watch-list status changes** (already-tracked but deferred; documentation status improved — do not
require approval to note, only to promote): xcena-mx1 (Hot Chips 38 disclosure + $135M Series B),
tensordyne-napier (now a numbered candidate above, supersedes the old watch-list entry), fujitsu-monaka
and arm-agi-cpu (Hot Chips 38 talks — CPU-only, out of registry scope by existing convention), celestial-ai
and rivos (both acquired — Marvell and Meta respectively; recommend removing from the AI-accelerator watch
list since neither is independent AI silicon anymore).

## Next Scheduled Scan

**After Hot Chips talk slides fully post and MLPerf Inference v6.1 publishes** — re-check the handful of
items this scan flagged as HC38-slide-gated (attendee-only PDFs for NVIDIA and AMD in particular). Absent
a specific trigger, follow the standard monthly/quarterly cadence.

## How to Process

Use `/ai-chip-research add-chip "<chip>" --seeds "url1, url2"` then run the pipeline, or:
`/search-chip-toolchain <chip>` → `/investigate-chip-resources <chip>` → `/summarize-chip-stack <chip>`
