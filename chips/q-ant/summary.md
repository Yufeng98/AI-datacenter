# Q.ANT Photonic NPU — Summary

*as_of: 2026-08-08*
*device_class: Photonic NPU (Germany)*
*chip: q-ant*

---

## One-Line Summary

Q.ANT builds all-photonic AI coprocessors using Thin-Film Lithium Niobate (TFLN) integrated circuits and the LENA (Light Empowered Native Arithmetics) architecture, targeting nonlinear AI and physics simulation workloads. Q.ANT quotes ~30 W for the photonic chip and separately "Power consumption of NPU 150 W" for the NPU module (the two scopings are Q.ANT's own and are never reconciled by the vendor), against 700–1,000 W for GPU equivalents.

---

## Key Facts

| Property | Value |
|----------|-------|
| Company | Q.ANT GmbH, Stuttgart, Germany |
| Founded | 2018 (TRUMPF spin-off) |
| Architecture | LENA — Light Empowered Native Arithmetics |
| Platform | TFLN (Thin-Film Lithium Niobate on Insulator, 600 nm) |
| Core compute element | MZI (Mach-Zehnder Interferometer) mesh |
| Current product | NPU 2 (announced SC25 Nov 2025). NPS described by Q.ANT as "available for evaluation in select data center environments today"; first commercial deployment (IONOS) planned for later in 2026 — had not occurred as of 2026-08-08 |
| Server form | NPS (Native Processing Server) — 19" turnkey Linux server |
| Chip power | ~30 W (Q.ANT chip-level figure). Q.ANT's product page separately states "Power consumption of NPU 150 W" — most plausibly PIC-only vs NPU-module, but Q.ANT does not say so |
| Energy efficiency claim | 30× lower energy vs CMOS; 50× higher performance (nonlinear AI) — Q.ANT internal benchmarking by its own description; in the ISC 2026 framing the 30× is narrowed to a *target* "at the photonic circuit level" for equivalent matrix operations |
| Compute precision | Analog continuous — no TOPS/TFLOPS metric. Q.ANT does publish "Throughput of NPU 8 GOPS" / "Speed-up to 8 GOPS" (nonlinear-function throughput, Gen 2) |
| Off-chip memory | Host DRAM via PCIe (no HBM/GDDR on NPU) |
| Host interface | PCIe (generation not disclosed) |
| Software | Proprietary; Q.PAL library; C/C++/Python; PyTorch/TF/Keras bridges. Third-party Daisytuner PyTorch→photonic compilation path (Apr 2026) |
| Open source | None from Q.ANT. Daisytuner's `docc` compiler is public, but no Q.ANT/photonic backend was located in its open tree |
| Commercial customers | IONOS (signed 2026-05-19; first commercial customer; rollout planned later in 2026) |
| HPC deployments | LRZ Munich (NPU 1 Jul 2025; NPU 2 Mar 2026); JSC Jülich (2025) |
| Funding | ~$80M USD (Series A, Jul–Oct 2025) — unchanged as of 2026-08-08 |
| Manufacturing | Stuttgart pilot line operated jointly with **IMS CHIPS**; Q.ANT is a TRUMPF spin-off and production is in Stuttgart (not a CMOS foundry) |
| US presence | U.S. headquarters, Austin, Texas (opened 2026-04-23) |
| Export controls | None known |

---

## Architecture Highlights

1. **All-photonic compute**: Unlike Lightmatter (photonic + CMOS co-packaging), Q.ANT performs the full computation path optically — no transistor switching during matrix-vector multiply.

2. **TFLN advantage over Silicon Photonics**: TFLN has a native Pockels (χ¹) electro-optic effect and χ² nonlinear optics. Silicon photonics lacks both — requiring carrier injection/depletion and off-chip nonlinear ops. TFLN is faster, lower-energy, and enables native optical nonlinear activations (NPU 2 key advance).

3. **NPU 2 nonlinear breakthrough**: NPU 2 introduces enhanced nonlinear processing capabilities in the optical domain, enabling new classes of AI models (physical AI, nonlinear networks) that achieve orders-of-magnitude lower parameter counts vs conventional neural nets on the same tasks.

4. **NPS server model**: Sold as a complete turnkey Linux server with PCIe host integration — no custom firmware or driver development required from customers.

5. **Not a GPU replacement**: Q.ANT explicitly targets nonlinear AI, physics simulation, robotics, and computer vision — not LLM inference or large-scale training. Q.ANT complements, not replaces, GPU nodes.

---

## Software Stack Summary

| Layer | Component |
|-------|-----------|
| Framework | PyTorch / TensorFlow / Keras (operator bridge); PyTorch also via third-party Daisytuner graph capture (Apr 2026) |
| API | C/C++/Python |
| Algorithm library | Q.PAL (Q.ANT Photonic Algorithm Library) |
| Runtime / HAL | Proprietary (PCIe DMA, DAC controller, ADC reader) |
| Driver | Linux kernel driver (proprietary) |
| Compiler | Q.ANT internal compiler (not user-facing) **plus** a third-party path: Daisytuner `docc` (SDFG-based) compiling PyTorch models to NPU Gen 2 |
| ISA | None (no user-programmable ISA) |

---

## Competitive Position

| Dimension | Q.ANT | Lightmatter |
|-----------|-------|-------------|
| Product type | All-photonic compute coprocessor | Photonic interconnect (Passage, shipping) + compute (Envise, announced) |
| Platform | TFLN | Silicon photonics |
| Shipping compute | YES (NPU 1 2024; NPU 2 deployed at LRZ + JSC, NPS available for evaluation; no general commercial availability as of Aug 2026) | NO (Envise not yet shipping) |
| HPC deployments | LRZ + JSC (named, paid, operational) | None confirmed |
| Commercial customers | IONOS (signed May 2026; not yet deployed) | None confirmed |
| Power | ~30 W chip / 150 W NPU (both Q.ANT figures, scoping unreconciled) | ~similar per package (not disclosed for Envise) |
| Target | Nonlinear AI, physics simulation | General DNN (transformers, ResNet) |
| Scale-out fabric | Standard HPC networking | Passage optical interconnect |
| Funding | ~$80M | ~$400M+ |

---

## Timeline

| Date | Event |
|------|-------|
| 2018 | Q.ANT founded as TRUMPF spin-off |
| 2024 | NPU 1 first commercial product launches; Stuttgart production begins |
| Jul 2025 | LRZ Munich deploys NPU 1 (world's first photonic AI processor at a supercomputing center) |
| Jul 2025 | €62M Series A announced (Cherry Ventures, UVC Partners, imec.xpand, TRUMPF, L-Bank) |
| Aug–Oct 2025 | Series A second close; Duquesne Family Office investment; total funding ~$80M |
| Nov 2025 (SC25) | NPU 2 announced at Supercomputing 2025, St. Louis |
| Q1 2026 | 2 additional unnamed HPC centers receive NPS servers |
| Mar 2026 | LRZ deploys NPU 2 (second-gen photonic processors at LRZ) |
| H1 2026 | *Previously recorded as "NPU 2 general customer shipments begin". As of 2026-08-08 there is no evidence of general commercial availability, and equally no evidence that Q.ANT retracted an H1 2026 date — see the 2026-08-08 update section.* |
| Apr 2026 | Daisytuner (third party) compiles a PyTorch model directly onto NPU Gen 2 — reveal dated "April" by Q.ANT; exact date not pinnable |
| 2026-04-23 | U.S. headquarters opened in Austin, Texas; Bruno Spruth (ex-IBM VP, POWER Processor Development) appointed CTO |
| 2026-05-19 | IONOS signed as first commercial customer for the NPS (announced at re:publica 2026, Berlin); rollout planned "later this year" |
| 2026-06-23 | ISC High Performance 2026 (Hamburg): diffusion image-to-image model and TiRex (xLSTM, NXAI) time-series model run on NPU Gen 2 |

---

## Q.ANT Update — First Commercial Customer and a Third-Party Compiler Path (2026-08-08)

*Updated 2026-08-08. Window covered: 2026-04-05 → 2026-08-08. Primary sources: Q.ANT press releases of 2026-04-23, 2026-05-19 and 2026-06-23; qant.com/photonic-computing/ (plus Wayback snapshots of 2026-01-22 and 2026-04-10); daisytuner.com. Independent corroboration: The Quantum Insider, 2026-05-21 and 2026-06-23.*

**Net assessment: moderate.** No new silicon, no new published hardware spec, and no Q.ANT SDK release in the window. What genuinely changed is (a) the first commercial (non-research) customer and (b) the compiler layer — which, for a software-stack survey, is the material item. Prior-generation content above is retained unchanged; corrections are called out explicitly below.

### 1. IONOS — first commercial customer (2026-05-19)

Q.ANT announced an agreement making **IONOS**, a European cloud/hosting provider serving ~6.8M customers across 17 markets in Europe and North America, the **first commercial customer** for the Native Processing Server (NPS), which carries the second-generation NPU. Announced at **re:publica 2026, Berlin**; the press release dateline is "Austin, Texas / Stuttgart, Germany — May 19, 2026".

The status verb matters: the agreement is **signed, not deployed**. Both the vendor release and independent coverage state the rollout is planned for "later this year" (2026); Q.ANT's boilerplate reads "first commercial deployment planned for 2026." Independently corroborated by The Quantum Insider (2026-05-21).

### 2. Third-party PyTorch → photonic compilation path (Daisytuner, revealed April 2026)

**Daisytuner GmbH** — a third party, *not* Q.ANT — compiled and deployed an object-detection model directly from PyTorch onto the **Q.ANT NPU Gen 2**. Daisytuner's own site specifies **Faster R-CNN with a ResNet-50 backbone**, with pre-processing, inference and post-processing running end-to-end and "no custom code", via its **Daisyflow** graph capture plus **`docc`** (Daisytuner Optimizing Compiler Collection, SDFG-based) toolchain. Q.ANT describes this as "the first time an AI model from a standard ML framework has been successfully compiled for photonic hardware" and dates the reveal to April 2026.

**Repo impact.** The layer-table row `Compiler | Internal only (not user-facing)` is now incomplete: a third-party compilation flow exists alongside Q.ANT's internal compiler, and the framework relationship is no longer only an operator bridge / Q.PAL library call.

**Caveats that must travel with this item:**
- The compiler is Daisytuner's, not Q.ANT's. Q.ANT's own stack remains proprietary with no public GitHub.
- `github.com/daisytuner/docc` is a public, actively developed repo (~22 stars as of 2026-08-07), but **a Q.ANT/photonic backend was not located in the open-source tree** — the photonic target may be delivered through Daisytuner's hosted service.
- The precise April date could not be pinned: daisytuner.com/news is client-rendered and undated.
- Do **not** attribute this work to SC'25 — the "SC '25" label on daisytuner.com belongs to a separate OpenFOAM/Tenstorrent item.
- This is **not** the ISC 2026 demo below. The two events must not be merged: Daisytuner's is an object-detection pipeline, not a diffusion model.

### 3. ISC High Performance 2026 demos (2026-06-23, Hamburg)

On the second-generation NPU, Q.ANT ran (a) a **diffusion model** for image-to-image synthesis and (b) **TiRex**, a time-series prediction model built on the **xLSTM** architecture from **NXAI** (Austria). Q.ANT claims these are the first complex, production-relevant AI workloads and the first diffusion model of this complexity on photonic hardware.

**No parameter counts, throughput, latency, energy measurements, or accuracy figures were published**, and neither the release nor independent coverage states which layers remained on CPU/GPU. The demonstration is qualitative and **not independently quantifiable**. The Quantum Insider (2026-06-23) explicitly notes the absence of comparative benchmarks.

### 4. US expansion and CTO hire (2026-04-23)

Q.ANT opened a **U.S. headquarters in Austin, Texas** and appointed **Bruno Spruth** as CTO. Spruth spent 16 years at IBM, most recently as **Vice President of POWER Processor Development**. Q.ANT plans to grow U.S. headcount to **20 employees over six months** across software development, photonics, and digital system design. (The May IONOS release refers to the same site more modestly as a "U.S. office in Austin, Texas.")

### 5. Corrections to previously recorded repo content

| Repo statement (pre-2026-08-08) | Correction |
|---|---|
| `Manufacturing \| TRUMPF photonic fab, Stuttgart` | Q.ANT manufactures at a Stuttgart pilot line **"operated jointly with IMS CHIPS"**, and "operates its own TFLN chip pilot line with IMS CHIPS." IMS CHIPS is a distinct named partner and belongs in the manufacturing entry. (A repo correction, not necessarily a change in the window.) |
| `Chip power \| ~30 W` (used bare in summary and hw-architecture) | Q.ANT's own product page states **"Power consumption of NPU 150 W"**. The 30 W and 150 W figures are most plausibly PIC-only versus NPU-module, but **Q.ANT does not say so**; both are now carried with that qualification rather than the bare 30 W. |
| `Compute precision \| Analog continuous (no TOPS/TFLOPS metric)` | Contradicted by Q.ANT's own published **8 GOPS** throughput figure. The precision statement stands (there is no digital dtype), but "no throughput metric at all" was wrong. |
| `NPU 2 shipping H1 2026` | Replaced with: NPU 2 is **deployed and operational at LRZ Munich and JSC Jülich**; the NPS is described by Q.ANT as "available for evaluation in select data center environments today"; the **first commercial deployment (IONOS) is planned for later in 2026** and had not occurred as of 2026-08-08. There is no evidence of general commercial availability, and equally no evidence that Q.ANT publicly retracted an H1 2026 date. |

### 6. Published specs that were public *before* this window (repo scan gaps, not new developments)

The 2026-01-22 Wayback snapshot of qant.com/photonic-computing/ already carried all of the following verbatim — they are **gaps in the repo's original scan**, recorded here with their true pre-baseline provenance, **not** developments in the 2026-04-05 → 2026-08-08 window:

| Spec | Vendor wording | Provenance |
|---|---|---|
| Throughput | "Speed-up to 8 GOPS" / "Throughput of NPU 8 GOPS" | qant.com/photonic-computing/, public by 2026-01-22 |
| NPU power | "Power consumption of NPU 150 W" | same, public by 2026-01-22 |
| PIC platform | **z-cut Lithium Niobate on Insulator (LNoI)** | same, public by 2026-01-22 |
| Operating temperature | 15–35 °C | same, public by 2026-01-22 |
| Roadmap | 0.1 GOps (2024) → 100,000 GOps (2028) | same, public by 2026-01-22 |

The only post-April wording change observed on that page is "Setting the conditions for multidimensional optical scaling" becoming **"Foundation for wavelength multiplexing"** — a marketing rephrase, not a new capability, and not evidence that wavelength multiplexing is implemented.

### 7. Contested — do not assert

- **Gen-1 → Gen-2 speedup: 100× or 50×?** Q.ANT's 2026-05-19 release states "In independent evaluation at Germany's Leibniz Supercomputing Centre (LRZ), the second-generation NPU demonstrated up to a **100x** performance increase over Q.ANT's first generation." The Quantum Insider's 2026-05-21 report of the *same* announcement states **50x**. No LRZ-authored evaluation, workload description, or methodology has been published. Record as: *Q.ANT claims an LRZ evaluation showed up to 100× Gen-1→Gen-2 improvement; secondary coverage of the same release reports 50×; the discrepancy is unresolved and no primary LRZ measurement is public.*
- The separate **"up to 30× higher energy efficiency and up to 50× greater performance per application versus conventional processors"** pair is, by Q.ANT's own wording, **internal benchmarking** — and in the ISC framing the 30× is narrowed to a target "at the photonic circuit level" for equivalent matrix operations, a materially weaker claim than a system-level number. This pair is distinct from the LRZ-attributed Gen-1→Gen-2 figure; do not conflate them.
- Q.ANT's statement that LRZ processors are "actively running workloads in climate modeling, medical imaging, and fusion energy research" is **vendor-sourced only**; no LRZ or JSC publication corroborating those specific workloads was found.

### 8. Still not disclosed

MZI count, channel/matrix dimensions, clock rate, ENOB / effective analog precision, PCIe generation, NPU count per NPS, die size, and any NPU 3 timeline all remain **not disclosed**. Funding is unchanged at ~$80M (Series A, Jul–Oct 2025); no new round in the window.

Q.ANT is **not** on the Hot Chips 38 program (Aug 23–25, 2026). Q.ANT is registered for SC 2026 (Nov 15–20, 2026, Chicago) with no content posted — a calendar entry only, and deliberately not recorded above as a fact about the chip.
