# Q.ANT Photonic NPU Hardware Architecture

*as_of: 2026-08-08*
*Generations: NPU 1 (2024, first commercial) and NPU 2 / NPU Gen 2 (announced Nov 2025, deployed at LRZ + JSC from Mar 2026)*
*No new silicon generation was disclosed in the 2026-04-05 → 2026-08-08 window; see the dated update section at the end.*

---

## Overview

Q.ANT is a Stuttgart, Germany photonic computing company founded in 2018 as a TRUMPF spin-off. Its **Native Processing Units (NPUs)** implement the **LENA (Light Empowered Native Arithmetics)** architecture — performing matrix-vector multiplication and nonlinear mathematical operations directly in the optical domain using **Thin-Film Lithium Niobate (TFLN)** photonic integrated circuits.

Q.ANT is the first company to achieve paid commercial deployment of all-photonic AI coprocessors at named national supercomputing centers (LRZ Munich, JSC Jülich).

---

## 1. Compute Engine

### Material Platform: TFLN (Thin-Film Lithium Niobate on Insulator)

| Property | Value |
|----------|-------|
| Material | LiNbO₃ thin film, 600 nm, on insulator. Q.ANT's product page specifies the PIC as **z-cut Lithium Niobate on Insulator (LNoI)** (public by 2026-01-22) |
| Substrate | Silicon wafer |
| Electro-optic effect | Pockels (r₃₃ ~30 pm/V) — fast, low-voltage phase control |
| Optical loss | Low — preserves signal fidelity across chip |
| Nonlinear optics | Native χ² nonlinearity (enables on-chip optical nonlinear math, NPU 2) |
| Thermal stability | High — minimal active cooling needed |
| vs Silicon Photonics | SiPh lacks Pockels effect; TFLN is faster, lower-energy, nonlinear-capable |

### Core Element: MZI Mesh

The fundamental compute element is the **Mach-Zehnder Interferometer (MZI)**:

- Voltage applied → refractive index change → phase shift → output amplitude = cos²(Δφ/2)
- An array of MZIs implements programmable optical matrix-vector multiplication (MVM)
- Weights are encoded as phase voltages (static during an inference pass)
- Computation time: picosecond-scale (no clock, photons travel at ~2/3 speed of light)
- No transistor switching during MVM — passive optical propagation

### NPU Generations

| Generation | Nonlinear Ops | Status | Key Site |
|------------|--------------|--------|----------|
| NPU 1 (2024) | Offloaded to host | Commercially shipping; full production | LRZ (Jul 2025) |
| NPU 2 / Gen 2 (announced Nov 2025) | Native TFLN χ² optical nonlinearity | *Originally recorded as "customer shipments H1 2026". Corrected 2026-08-08:* deployed and operational at LRZ and JSC; NPS "available for evaluation in select data center environments today"; first commercial deployment (IONOS) planned for later in 2026 and not yet occurred | LRZ (Mar 2026); JSC Jülich |
| NPU 3 | — | **Not disclosed** — no timeline, no specification published | — |

*No third silicon generation has been announced. Q.ANT publishes a throughput roadmap (0.1 GOps in 2024 → 100,000 GOps by 2028) but does not attach generation names, dates, or architectures to the intermediate points.*

---

## 2. Performance Specifications

| Metric | NPU 2 Value | Notes |
|--------|-------------|-------|
| Chip power | ~30 W | vs 700–1,000 W for Hopper/Blackwell. **Qualified 2026-08-08:** this is Q.ANT's chip/PIC-level figure; see the NPU-module figure below |
| NPU module power | **150 W** ("Power consumption of NPU 150 W") | qant.com/photonic-computing/; public by 2026-01-22, i.e. a repo scan gap, not a window change. The 30 W vs 150 W scoping (PIC-only vs NPU module) is *plausible but never stated by Q.ANT* |
| Throughput | **8 GOPS** ("Speed-up to 8 GOPS" / "Throughput of NPU 8 GOPS"), sustained on nonlinear functions | qant.com/photonic-computing/; public by 2026-01-22. This is the only published throughput number and is not a TOPS/TFLOPS-equivalent |
| Throughput roadmap | 0.1 GOps (2024) → 100,000 GOps (2028) | Vendor roadmap chart; public by 2026-01-22. No generation names attached to intermediate points |
| Operating temperature | 15–35 °C | qant.com/photonic-computing/; public by 2026-01-22 |
| Energy efficiency | "up to 30× lower per workload" claim | Q.ANT vs CMOS; **Q.ANT's own internal benchmarking**. In the ISC 2026 framing narrowed to a *target* "at the photonic circuit level" for equivalent matrix operations |
| Performance | "up to 50× higher" claim | Nonlinear AI / physics simulation only; Q.ANT internal benchmarking |
| Gen 1 → Gen 2 speed-up | **Contested: 100× (Q.ANT PR, 2026-05-19, attributed to an LRZ evaluation) vs 50× (The Quantum Insider, 2026-05-21, same announcement)** | No LRZ-authored evaluation, workload description, or methodology has been published. Unresolved — do not assert either figure |
| Precision | Analog continuous (no digital TOPS) | Company avoids TOPS comparisons. ENOB / effective analog precision: **not disclosed** |
| Clock | N/A — photonic, no clock domain | — |
| Transistor count | N/A — TFLN photonic, not CMOS | — |
| Die size | Not disclosed | — |
| MZI count / matrix dimensions | Not disclosed | — |
| Process | Stuttgart TFLN pilot line operated jointly with **IMS CHIPS** | Not a CMOS foundry node |

---

## 3. Memory Hierarchy

| Level | Description |
|-------|-------------|
| On-chip | None (photonic; no SRAM/HBM) |
| Weight storage | MZI phase voltages (static during inference) |
| Input/activation | Photodetector → ADC → host DRAM |
| Off-chip | Host DRAM via PCIe (no dedicated HBM/GDDR) |
| Weight loading | DAC → voltage → MZI phase (from host PCIe) |
| Memory hierarchy | Flat: host DRAM → PCIe → DAC/ADC → chip |

---

## 4. System Form Factor (NPS Server)

| Component | Description |
|-----------|-------------|
| Product name | NPS (Native Processing Server) |
| Form factor | 19-inch rack-mount server (1U/2U) |
| NPU count per server | Multiple (exact count not disclosed) |
| Host CPU | x86 (Intel/AMD) |
| OS | Linux |
| Host interface | PCIe (standard; generation not disclosed) |
| Networking | Standard Ethernet / HPC fabric via host NIC |
| Turnkey | Yes — complete server delivered |
| Availability (as of 2026-08-08) | Q.ANT: "available for evaluation in select data center environments today". No general commercial availability; first commercial deployment (IONOS) planned for later in 2026 |

---

## 5. Interconnect

| Level | Mechanism |
|-------|-----------|
| Chip-to-host | PCIe (standard) — **generation not disclosed** (Gen4/Gen5 unconfirmed) |
| Multi-NPU intra-server | Multiple dies in NPS (topology not disclosed) |
| Multi-server scale-out | Standard HPC networking (IB/Ethernet via host NIC) |
| Photonic interconnect | None (Q.ANT is compute-only; no Passage-equivalent) |

---

## 6. HPC and Commercial Deployments

| Site | Deployment | Generation |
|------|------------|------------|
| LRZ (Leibniz Supercomputing Centre), Munich | Jul 2025; upgraded Mar 2026; operational | NPU 1 → NPU 2 |
| JSC (Jülich Supercomputing Centre), FZ Jülich | 2025 (research collaboration); operational | NPU 1 / NPU 2 |
| 2× unnamed HPC centers | Q1 2026 | NPU 2 |
| **IONOS** (European cloud/hosting; ~6.8M customers, 17 markets) | **Agreement signed 2026-05-19** (announced at re:publica 2026, Berlin). First commercial customer for the NPS. **Signed, not deployed** — rollout planned "later this year" (2026) | NPU 2 (in NPS) |

*The "Commercial cloud partner — imminent (as of Apr 2026)" line previously recorded here is resolved: the partner is IONOS.*

Q.ANT states its LRZ processors are "actively running workloads in climate modeling, medical imaging, and fusion energy research". This is **vendor-sourced only** — no LRZ or JSC publication corroborating those specific workloads was found.

---

## 7. Target Workloads

**Targeted:**
- Physics-based AI and simulation (PDEs, physics-constrained networks)
- Advanced robotics and physical AI
- Computer vision and industrial intelligence
- Audio/video processing
- Sensor fusion
- Scientific discovery

**Not targeted:**
- LLM / Transformer inference
- Large-scale model training

---

## Sources

- https://qant.com/photonic-computing/
- https://qant.com/press-releases/q-ant-unveils-its-second-generation-photonic-processor-to-power-the-next-wave-of-ai-and-hpc/
- https://qant.com/press-releases/higher-performance-less-energy-q-ant-deploys-second-generation-photonic-processors-at-supercomputing-center-lrz/
- https://www.eetimes.com/q-ant-raises-series-a-debuts-second-gen-tfln-photonic-chip/
- https://thequantuminsider.com/2025/11/19/qant-next-gen-photonic-npu/

---

## Update — 2026-08-08 (window 2026-04-05 → 2026-08-08)

*Sources: Q.ANT press releases 2026-04-23, 2026-05-19, 2026-06-23; qant.com/photonic-computing/ current page plus Wayback snapshots 2026-01-22 and 2026-04-10; The Quantum Insider 2026-05-21 and 2026-06-23; daisytuner.com.*

**No new silicon and no new published hardware specification appeared in this window.** The hardware sections above are unchanged in substance; what follows records the deployment/manufacturing changes, the newly-surfaced (but pre-baseline) vendor specs now folded into the tables above, and the workload demonstrations.

### Hardware-relevant changes in the window

| Item | Detail |
|---|---|
| First commercial customer | IONOS, agreement **signed** 2026-05-19; rollout planned later in 2026; not deployed as of 2026-08-08 |
| Manufacturing partner | Stuttgart TFLN pilot line is **"operated jointly with IMS CHIPS"**; Q.ANT "operates its own TFLN chip pilot line with IMS CHIPS". Replaces the bare "TRUMPF photonic fab" attribution |
| Marketing wording change | "Setting the conditions for multidimensional optical scaling" → **"Foundation for wavelength multiplexing"** on qant.com/photonic-computing/, observed after April 2026. A rephrase only — wavelength multiplexing is **not** claimed as implemented in NPU Gen 2 |
| Corporate | U.S. headquarters opened in Austin, Texas (2026-04-23); Bruno Spruth (16 years at IBM, latterly VP of POWER Processor Development) appointed CTO; plan to reach 20 U.S. employees within six months across software, photonics, and digital system design |

### Workload demonstrations — ISC High Performance 2026, Hamburg (2026-06-23)

On the second-generation NPU, Q.ANT ran:
- a **diffusion model** for image-to-image synthesis;
- **TiRex**, a time-series prediction model built on the **xLSTM** architecture from **NXAI** (Austria).

Q.ANT claims these are the first complex, production-relevant AI workloads and the first diffusion model of this complexity on photonic hardware. **No parameter counts, throughput, latency, energy, or accuracy figures were published**, and neither the release nor independent coverage states which layers remained on CPU/GPU. These demos yield **no hardware numbers** and must not be used to infer any. The Quantum Insider (2026-06-23) explicitly notes the absence of comparative benchmarks.

Separately and **not the same event**: Daisytuner (a third party) compiled **Faster R-CNN with a ResNet-50 backbone** from PyTorch onto NPU Gen 2, revealed April 2026 — see `chips/q-ant/summary.md` and the software-stack investigation for detail.

### Specs added above that were public *before* the window

The following came from the 2026-01-22 Wayback snapshot of qant.com/photonic-computing/ and are **repo scan gaps, not developments in this window**: 8 GOPS throughput, 150 W NPU power, z-cut LNoI PIC platform, 15–35 °C operating range, and the 0.1 → 100,000 GOps 2024–2028 roadmap. They persist in the 2026-04-10 snapshot, confirming they straddle the baseline date.

### Still not disclosed (unchanged)

MZI count, channel/matrix dimensions, clock rate, ENOB / effective analog precision, PCIe generation, NPU count per NPS, die size, and any NPU 3 timeline. Q.ANT is not on the Hot Chips 38 program (Aug 23–25, 2026).

### Additional sources (2026-08-08)

- https://qant.com/press-releases/q-ant-takes-photonic-ai-computing-commercial-as-ais-power-demand-surges/ — IONOS, 2026-05-19
- https://qant.com/press-releases/q-ant-brings-commercial-photonic-computing-to-the-united-states-appoints-bruno-spruth-as-cto/ — Austin HQ + CTO, 2026-04-23
- https://qant.com/press-releases/q-ant-runs-generative-ai-on-photonic-hardware/ — ISC 2026 demos, 2026-06-23
- https://thequantuminsider.com/2026/05/21/qant-photonic-ai-computing-commercial-ionos/ — independent; reports 50× (not 100×)
- https://thequantuminsider.com/2026/06/23/q-ant-runs-generative-ai-on-photonic-hardware/ — independent; confirms no benchmarks published
- https://web.archive.org/web/20260122130900/https://qant.com/photonic-computing/ — pre-baseline spec evidence
- https://web.archive.org/web/20260410180106/https://qant.com/photonic-computing/ — second snapshot across the baseline
