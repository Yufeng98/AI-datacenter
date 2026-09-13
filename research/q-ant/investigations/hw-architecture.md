# Q.ANT Photonic NPU — Hardware Architecture Investigation

*as_of: 2026-04-05*
*chip: q-ant*
*device_class: Photonic NPU (Germany)*
*generations: NPU 1 (2024, first commercial), NPU 2 (2025/2026, second-gen)*

---

## Overview

Q.ANT is a Stuttgart-based photonic computing company founded in 2018 as a spin-off from TRUMPF (a world-leading laser and photonics manufacturer). Q.ANT builds what it describes as the world's first commercially shipping all-photonic AI coprocessors, branded as **Native Processing Units (NPUs)**. The architecture is called **LENA — Light Empowered Native Arithmetics**.

Rather than performing computation in silicon transistors, Q.ANT's NPUs perform matrix-vector multiplication and nonlinear mathematical operations directly in the optical domain using **Thin-Film Lithium Niobate (TFLN)** photonic integrated circuits. This eliminates the transistor-based switching bottleneck that dominates digital CMOS power consumption.

---

## 1. Photonic Compute Core

### Material Platform: Thin-Film Lithium Niobate on Insulator (TFLNoI)

| Property | Description |
|----------|-------------|
| Material | LiNbO₃ (lithium niobate) thin film, 600 nm thick, on insulator substrate |
| Substrate | Silicon wafer with bonded TFLNoI layer |
| Electro-optic coefficient | High (r₃₃ ~30 pm/V) — enables fast, low-voltage light modulation |
| Optical loss | Low — preserves signal integrity across chip |
| Thermal properties | Thermally stable; low active cooling requirement |
| Nonlinear optics | Native second-order (χ²) and electro-optic effects; enables on-chip nonlinear math |

TFLNoI is a key differentiator over silicon photonics (SiPh): SiPh lacks a second-order electro-optic effect (centrosymmetric crystal), requiring carrier injection or depletion for modulation. TFLN's Pockels effect allows faster, lower-energy, linear phase control.

### Core Optical Element: Mach-Zehnder Interferometer (MZI) Mesh

The fundamental compute element is the **Mach-Zehnder Interferometer (MZI)**:

```
Input waveguide
       |
   [Splitter 50:50]
   /              \
[Phase shifter 1] [Phase shifter 2]  ← voltage-controlled refractive index
   \              /
   [Combiner 50:50]
       |
Output (amplitude = cos²(Δφ/2))
```

- Refractive index of waveguides is controlled by applying a voltage (electro-optic Pockels effect)
- An array of MZIs implements a programmable **optical matrix-vector multiplication (MVM)**
- Weight encoding: voltage values → phase shifts → MZI transmission coefficients
- Computation speed: photons travel at ~2/3 the speed of light in waveguide, completing operations in picoseconds
- No transistor switching during a matrix-vector multiply — the light "computes" passively

### LENA Architecture (Light Empowered Native Arithmetics)

LENA is Q.ANT's proprietary compute architecture. Key characteristics:

| Property | Description |
|----------|-------------|
| Compute paradigm | Analog optical matrix-vector multiplication |
| Weight storage | Phase voltages applied to MZI electro-optic modulators |
| Nonlinear activation | Native to optical domain via TFLN nonlinear optical properties (NPU 2) |
| Precision | Analog (not digital TOPS); company does not publish digital-equivalent TOPS figures |
| Energy per operation | Sub-femtojoule optical operations (not counting DAC/ADC overhead) |
| Numerical format | Continuous analog amplitude; approximates FP-like precision at system level |

**NPU 1 (2024)**: First commercial generation. Linear MVM in optical domain. Nonlinear operations offloaded to co-processor or host.

**NPU 2 (2025/2026)**: Enhanced nonlinear processing natively in optical domain via TFLN χ² nonlinearities. This is the key NPU 2 differentiator — enables optical nonlinear activations, not just linear layers.

---

## 2. Hardware Specifications

### NPU 2 Performance Claims (Published by Q.ANT, SC25 Nov 2025)

| Metric | Q.ANT NPU 2 | Notes |
|--------|-------------|-------|
| Power consumption | ~30 W (chip) | vs 700–1,000 W for modern Nvidia GPUs |
| Energy efficiency claim | 30× lower energy per workload vs CMOS | Q.ANT marketing claim for targeted workload class |
| Performance claim | 50× higher performance vs conventional silicon | For nonlinear AI / physics simulation workload class |
| System TDP (NPS server) | Not disclosed (includes x86 host + cooling) | Full 19" server |
| Compute precision | Analog optical — no standard TOPS/TFLOPS metric | Company explicitly avoids digital TOPS comparison |
| Target workloads | Nonlinear AI (physics AI, robotics, computer vision, simulation) | NOT for LLM inference or large-model training |
| Clock frequency | N/A (photonic; no clock; ops in-flight at light speed) | — |

### Chip Physical Properties

| Property | Value |
|----------|-------|
| Substrate | Silicon wafer |
| Active layer | TFLN 600 nm thick on insulator |
| Process node | Proprietary TRUMPF photonic process; not CMOS foundry |
| Die size | Not disclosed |
| Manufacturing location | Stuttgart, Germany (TRUMPF facilities) |
| Transistor count | N/A — photonic chip, not CMOS transistors |

### Memory / Signal Interface

| Component | Description |
|-----------|-------------|
| Weight encoding | DAC (digital-to-analog) → voltage → MZI phase shift |
| Activation readout | Photodetector array → ADC (analog-to-digital) |
| On-chip signal buffer | Waveguide delay lines (limited; weights are phase-static during inference) |
| Off-chip interface | PCIe (host system) — weights/inputs transferred from x86 host |
| Memory hierarchy | No SRAM/HBM; model inputs streamed via host PCIe; weights loaded as voltage configs |

---

## 3. System / Server Form Factor

### Native Processing Server (NPS)

| Component | Description |
|-----------|-------------|
| Form factor | 19-inch 1U or 2U rack-mount server |
| Processor | Multiple NPU Gen 2 photonic chips |
| Host processor | x86 CPU (Intel/AMD; model not specified) |
| Operating system | Linux (standard distribution) |
| Host interface | PCIe (standard) |
| Connectivity | Standard data center networking (Ethernet) |
| Customer interface | Turnkey appliance — ordered as complete server |

The NPS is an **add-in accelerator server**, not a standalone GPU replacement. It plugs into existing HPC/data-center infrastructure alongside conventional CPU/GPU nodes.

---

## 4. Interconnect and Scaling

| Level | Mechanism | Bandwidth |
|-------|-----------|-----------|
| Chip-to-host | PCIe (standard) | PCIe Gen4/5 (not specified) |
| Multi-NPU (intra-server) | Multiple NPU dies in single NPS server | Not disclosed |
| Multi-server scale-out | Standard Ethernet / HPC fabric (InfiniBand or OmniPath) | Via host NIC |
| Photonic interconnect | None (unlike Lightmatter Passage) — Q.ANT is compute-only | N/A |

Q.ANT does **not** productize photonic interconnects (unlike Lightmatter whose Passage product is a photonic interconnect). Q.ANT's photonic technology is purely in the compute path.

---

## 5. Manufacturing and Production

| Property | Description |
|----------|-------------|
| Fab location | Stuttgart, Germany |
| Fab type | TRUMPF photonic fab (not semiconductor foundry like TSMC/Samsung) |
| Production status (NPU 1) | Full production capacity reached 2024–2025 |
| Production status (NPU 2) | Customer shipments H1 2026 |
| Wafer process | TFLN on silicon wafer; TRUMPF proprietary photonic process |
| Export controls | None known — European photonic product, no US semiconductor restrictions |

TRUMPF is a global leader in industrial lasers and photonic manufacturing equipment. Q.ANT benefits from TRUMPF's in-house photonic fab capability, which is a supply-chain moat.

---

## 6. Deployed Installations (HPC/Supercomputing)

| Site | Country | Deployment | Generation |
|------|---------|------------|------------|
| Leibniz Supercomputing Centre (LRZ), Munich | Germany | Jul 2025 (NPU 1); Mar 2026 (NPU 2) | NPU 1 → NPU 2 |
| Jülich Supercomputing Centre (JSC), FZ Jülich | Germany | 2025 (research collaboration) | NPU 1 / NPU 2 |
| Additional HPC center #1 | Not named | Q1 2026 | NPU 2 |
| Additional HPC center #2 | Not named | Q1 2026 | NPU 2 |
| Commercial cloud partner | Not disclosed | Imminent (as of Apr 2026) | NPU 2 |

Q.ANT is the first photonic AI processor company to achieve paid HPC deployment at named national supercomputing centers.

---

## 7. Competitive Positioning vs Other Photonic Approaches

| Dimension | Q.ANT (NPU 2) | Lightmatter (Envise) | Lightmatter (Passage) |
|-----------|---------------|----------------------|----------------------|
| Product type | All-photonic compute coprocessor | Photonic compute + CMOS | Photonic interconnect |
| Photonic platform | TFLN (Thin-Film Lithium Niobate) | Silicon photonics (SiPh) | Silicon photonics (SiPh) |
| Shipping status | YES — NPU 1 (2024), NPU 2 (H1 2026) | Announced; Nature paper Apr 2025 | YES — Passage M1000 (data center) |
| Nonlinear ops | Native TFLN optical nonlinearity (NPU 2) | Via co-packaged CMOS DCI | N/A |
| Memory | Host DRAM via PCIe (no HBM) | DDR4 + NVMe (no HBM) | N/A |
| Target workload | Nonlinear AI, physics simulation | General DNN (transformers shown) | General AI interconnect |
| Scale-out | Standard networking | Passage optical interconnect | Passage itself |
| HPC deployments | LRZ, JSC (Germany, named, paid) | None publicly confirmed | Hyperscalers (unnamed) |
| Funding | ~$80M (Series A, Jul–Oct 2025) | ~$400M+ (Series C 2023) | Same company |

---

## 8. Company Background

| Property | Detail |
|----------|--------|
| Founded | 2018 |
| Origin | Spin-off from TRUMPF GmbH + Co. KG (Stuttgart) |
| Founder / CEO | Michael Förtsch |
| HQ | Stuttgart, Germany |
| Total funding | ~$80M USD (€62M Series A + Duquesne second close, Jul–Oct 2025) |
| Lead investors | Cherry Ventures, UVC Partners, imec.xpand; strategic: TRUMPF, L-Bank |
| Other investors | Verve Ventures, Grazia Equity, EXF Alpha, LEA Partners, Onsight Ventures, Duquesne Family Office LLC |
| Largest photonic computing round in Europe | Yes (as of Oct 2025) |
| Public GitHub | None identified (closed-source; proprietary stack) |

---

## Sources

- https://qant.com/photonic-computing/
- https://qant.com/press-releases/q-ant-unveils-its-second-generation-photonic-processor-to-power-the-next-wave-of-ai-and-hpc/
- https://qant.com/press-releases/higher-performance-less-energy-q-ant-deploys-second-generation-photonic-processors-at-supercomputing-center-lrz/
- https://qant.com/press-releases/leibniz-supercomputing-centre-computes-with-light-worlds-first-photonic-ai-processor-from-q-ant-goes-into-operation/
- https://qant.com/press-releases/qant-raises-62-million-euro-to-transform-the-future-of-computing-with-photonic-processing/
- https://www.eetimes.com/q-ant-raises-series-a-debuts-second-gen-tfln-photonic-chip/
- https://www.eetimes.com/production-of-q-ant-photonic-ai-accelerators-begins-in-stuttgart/
- https://www.eetimes.com/q-ant-hits-full-production-capacity-for-photonic-ai-processors/
- https://www.allaboutcircuits.com/news/q.ants-new-photonic-processor-pushes-ai-and-hpc-beyond-silicons-limits/
- https://www.hpcwire.com/2025/06/18/q-ant-photonic-computing-shines-at-isc-2025/
- https://www.fz-juelich.de/en/news/archive/press-release/2025/computing-with-light-forschungszentrum-julich-and-q-ant-launch-collaboration-on-photonic-computing
- https://thequantuminsider.com/2025/11/19/qant-next-gen-photonic-npu/

---

# Investigation Update — 2026-08-08

*Window: 2026-04-05 → 2026-08-08. Investigator note: the original (2026-04-05) investigation drew almost entirely on qant.com and vendor-syndicated coverage. This update was verified against Wayback snapshots and independent reporting, and two of the original scan's framings did not survive.*

## Headline: no new silicon

**No new NPU generation, no new published hardware specification, and no NPU 3 timeline appeared in this window.** The hardware description in sections 1–5 above stands. What changed is deployment status, a named manufacturing partner, and — the material item for a software-stack survey — the compiler layer (documented in `software-stack.md`).

## 1. Deployment status corrected

| Prior repo statement | Verified status as of 2026-08-08 |
|---|---|
| "NPU 2 customer shipments H1 2026" | NPU 2 is **deployed and operational at LRZ Munich and JSC Jülich**. The NPS is described by Q.ANT as **"available for evaluation in select data center environments today"**. The **first commercial deployment (IONOS) is planned for later in 2026** and had **not** occurred as of 2026-08-08. |
| "Commercial cloud partner — imminent (as of Apr 2026)" | Resolved: the partner is **IONOS**, agreement signed 2026-05-19. |

Two things must both be said, because the evidence supports both and only both: there is **no evidence of general commercial availability**, and there is **no evidence that Q.ANT publicly retracted an H1 2026 date**. The word "slipped" implies a withdrawn commitment and is not supported by any source.

### IONOS (2026-05-19)

- European cloud/hosting provider; **~6.8M customers across 17 markets** in Europe and North America.
- **First commercial customer** for the Native Processing Server (NPS), which carries the second-generation NPU.
- Announced at **re:publica 2026, Berlin**. Press-release dateline: "Austin, Texas / Stuttgart, Germany — May 19, 2026".
- **Signed, not deployed.** Rollout planned "later this year" (2026); Q.ANT boilerplate: "first commercial deployment planned for 2026."
- Independently corroborated by The Quantum Insider, 2026-05-21.

## 2. Manufacturing — IMS CHIPS (correction to section 5 above)

Q.ANT manufactures its photonic chips at a pilot line in Stuttgart **"operated jointly with IMS CHIPS"**, and "operates its own TFLN chip pilot line with IMS CHIPS." Section 5's `Fab type | TRUMPF photonic fab` is incomplete: **IMS CHIPS is a distinct named partner and the joint operator of the pilot line.** Q.ANT remains a TRUMPF spin-off and TRUMPF remains a strategic investor; that is a separate fact from who runs the line.

This is a **repo correction**, not necessarily a change that occurred in the window.

## 3. Published specs the original scan missed (all pre-baseline)

The 2026-01-22 Wayback snapshot of `qant.com/photonic-computing/` already contained the following **verbatim**, and the 2026-04-10 snapshot confirms they straddle the baseline date. They are **gaps in the repo's original scan, not developments in this window** — recording them as new specs would be a pre-baseline-as-new error.

| Spec | Vendor wording | Public by |
|---|---|---|
| Throughput | "Speed-up to 8 GOPS" / "Throughput of NPU 8 GOPS" (nonlinear functions, Gen 2) | 2026-01-22 |
| NPU power | "Power consumption of NPU 150 W" | 2026-01-22 |
| PIC platform | **z-cut Lithium Niobate on Insulator (LNoI)** | 2026-01-22 |
| Operating temperature | 15–35 °C | 2026-01-22 |
| Roadmap | 0.1 GOps (2024) → 100,000 GOps (2028) | 2026-01-22 |

### Two conflicts this creates inside the repo

1. **`Compute precision | Analog continuous (no TOPS/TFLOPS metric)`** is contradicted by Q.ANT's own published 8 GOPS figure. The *precision* statement stands — there is no digital dtype and no TOPS-equivalent — but "no throughput metric at all" was wrong.
2. **`~30 W chip`** (asserted bare in both `summary.md` and `hw-architecture.md`) is contradicted by the published **150 W NPU** figure. The two are **most plausibly PIC-only versus NPU-module**, but **Q.ANT does not say so**. Both figures are now carried with that qualification rather than the bare 30 W.

### One wording change, not a capability change

The only post-April diff observed on `qant.com/photonic-computing/` is "Setting the conditions for multidimensional optical scaling" → **"Foundation for wavelength multiplexing"**. This is a marketing rephrase. **Wavelength multiplexing is not claimed as implemented** in NPU Gen 2 and must not be recorded as a capability or as a roadmap commitment with a date.

## 4. Contested figure: Gen 1 → Gen 2 speed-up

| Source | Date | Figure |
|---|---|---|
| Q.ANT press release (primary) | 2026-05-19 | "In independent evaluation at Germany's Leibniz Supercomputing Centre (LRZ), the second-generation NPU demonstrated up to a **100x** performance increase over Q.ANT's first generation." |
| The Quantum Insider (independent), reporting the *same* announcement | 2026-05-21 | "up to a **50x** performance increase over Q.ANT's first generation" |

**No LRZ-authored evaluation, workload description, or methodology has been published.** Two published sources disagree by 2× on the headline figure. Record as: *Q.ANT claims an LRZ evaluation showed up to 100× Gen-1→Gen-2 improvement; secondary coverage of the same release reports 50×; the discrepancy is unresolved and no primary LRZ measurement is public.* Do not assert either number.

**Do not conflate this with** the separate "up to 30× higher energy efficiency and up to 50× greater performance per application versus conventional processors" pair, which is **Q.ANT internal benchmarking by its own admission**, and which the ISC/June framing narrows to a *target* "at the photonic circuit level" for equivalent matrix operations. The product page states it more loosely as "up to 30x energy efficiency and 50x performance gains per application". Neither scoping is independently measured.

## 5. Workload demonstrations — ISC High Performance 2026 (2026-06-23, Hamburg)

On the second-generation NPU, Q.ANT ran:
- a **diffusion model** for image-to-image synthesis;
- **TiRex**, a time-series prediction model built on the **xLSTM** architecture from **NXAI** (Austria).

Q.ANT claims these are the first complex, production-relevant AI workloads and the first diffusion model of this complexity on photonic hardware.

**Nothing quantitative was published**: no parameter counts, throughput, latency, energy measurements, or accuracy figures, and neither the release nor independent coverage states which layers remained on CPU/GPU. The Quantum Insider (2026-06-23) explicitly notes the absence of comparative benchmarks. **These demos yield no hardware numbers and none may be inferred from them.**

This is **not** the Daisytuner work (see `software-stack.md`); the two events must not be merged.

## 6. Corporate — US expansion and CTO (2026-04-23)

- **U.S. headquarters in Austin, Texas** (the May IONOS release refers to the same site more modestly as a "U.S. office").
- **Bruno Spruth** appointed **CTO** — 16 years at IBM, most recently **Vice President of POWER Processor Development**; remit covers technology strategy, product development, and North American operations.
- Plan to grow to **20 U.S. employees over six months** across software development, photonics, and digital system design.
- Q.ANT states its processors are "actively running workloads in climate modeling, medical imaging, and fusion energy research" — **vendor-sourced only**; no LRZ or JSC publication corroborating those specific workloads was found.

## 7. Unchanged / still not disclosed

Funding is unchanged at **~$80M** (Series A, Jul–Oct 2025); no new round in the window.

Still **not disclosed**: MZI count, channel/matrix dimensions, clock rate, ENOB / effective analog precision, PCIe generation, NPU count per NPS, die size, and any NPU 3 timeline. The product spec pages do not publish them.

Q.ANT is **not** on the Hot Chips 38 program (Aug 23–25, 2026 — in the future as of this investigation, and no source of specs regardless). Q.ANT is registered for **SC 2026** (Nov 15–20, 2026, Chicago) with **no content posted**; this is a forward-looking calendar entry and is deliberately not recorded as a fact about the chip.

## Sources added 2026-08-08

- https://qant.com/press-releases/q-ant-takes-photonic-ai-computing-commercial-as-ais-power-demand-surges/ — primary; IONOS, re:publica venue, IMS CHIPS joint pilot line, the 100× wording
- https://qant.com/press-releases/q-ant-brings-commercial-photonic-computing-to-the-united-states-appoints-bruno-spruth-as-cto/ — primary (datePublished 2026-04-23); Austin HQ, Spruth, 20-employee target
- https://qant.com/press-releases/q-ant-runs-generative-ai-on-photonic-hardware/ — primary (datePublished 2026-06-23); NXAI/TiRex/xLSTM, dates the Daisytuner reveal to April 2026
- https://thequantuminsider.com/2026/05/21/qant-photonic-ai-computing-commercial-ionos/ — independent; reports **50×**, not 100×
- https://thequantuminsider.com/2026/06/23/q-ant-runs-generative-ai-on-photonic-hardware/ — independent; confirms no benchmarks published
- https://web.archive.org/web/20260122130900/https://qant.com/photonic-computing/ — decisive pre-baseline spec evidence
- https://web.archive.org/web/20260410180106/https://qant.com/photonic-computing/ — second snapshot across the baseline
- https://qant.com/photonic-computing/ — current page (fetched 2026-08-08), used only to diff against the snapshots
