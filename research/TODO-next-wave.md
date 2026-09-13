# Next Wave — Investigation Queue

*Created: 2026-04-05*
*Last scan: 2026-08-08 (`update-chip-landscape`, mode full)*

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

## Next Scheduled Scan

**Early September 2026**, after **Hot Chips 38 (Aug 23–25, 2026)**. That conference has scheduled disclosure
talks for MTIA, Maia 200, Rubin, AMD MI400-series (two talks), Intel Crescent Island, TPU v8, SambaNova SN50,
Cerebras rack-scale WSE, NVIDIA LPU, Samsung LPDDR5X-PIM, XCENA MX1, and OpenAI's chip. Many "not disclosed"
cells written on 2026-08-08 should become real specs once slides post (~Aug 25–Sept 2026).

## How to Process

Use `/ai-chip-research add-chip "<chip>" --seeds "url1, url2"` then run the pipeline, or:
`/search-chip-toolchain <chip>` → `/investigate-chip-resources <chip>` → `/summarize-chip-stack <chip>`
