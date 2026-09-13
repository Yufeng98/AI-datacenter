# Lightmatter — Hardware Architecture Investigation

*as_of: 2026-08-08 (original investigation 2026-04-05; 2026-08-08 update appended at end)*
*chip: lightmatter*
*device_class: Photonic Interconnect + Optics (compute product no longer marketed) — historically Photonic Compute + Interconnect*
*sources: Nature 2025 (doi:10.1038/s41586-025-08854-x), lightmatter.co/products/envise, lightmatter.co/products/passage, lightmatter.co/blog/seeing-is-believing; 2026-08-08 sources listed in the appended section*

---

## Q1: How does photonic matrix multiply work (MZI arrays)?

### The Physics

Photonic matrix multiplication exploits the wave nature of light. A Mach-Zehnder interferometer (MZI) is a Y-junction splitter that:
1. Splits an incoming optical signal into two waveguide paths
2. Applies a phase shift to one or both paths via an electro-optic material (voltage → refractive index change via Pockels effect)
3. Recombines the two beams at a second Y-junction

The output amplitudes at the two output ports are:
- `out_0 = cos(θ/2) · in_0 + i·sin(θ/2) · in_1`
- `out_1 = i·sin(θ/2) · in_0 + cos(θ/2) · in_1`

A single MZI implements a 2×2 unitary rotation parameterized by phase θ.

### From MZI to Matrix

A triangular or rectangular mesh of N×(N-1)/2 MZIs implements an N×N unitary matrix U. Any arbitrary matrix W can be decomposed as W = U · Σ · V† (singular value decomposition), where:
- U and V† are unitary matrices → implemented by MZI meshes
- Σ is a diagonal scaling matrix → implemented by optical attenuators

For Lightmatter's Envise, each Photonic Tensor Core (PTC) is a **128×128 MZI mesh**, enabling 128×128 matrix-vector products. The Nature 2025 paper confirms 4 PTCs per package.

### The Complete Compute Path

```
Host CPU / DRAM
    ↓ (data transfer)
Digital Control Interface (DCI) — 12nm CMOS
    ↓ DAC (digital → optical amplitude)
Laser source → optical modulator
    ↓ (512 optical channels, WDM)
Photonic Tensor Core (PTC): 128×128 MZI mesh
    • Light propagates through MZI network in nanoseconds
    • MZI phase settings encode the weight matrix (set once per layer)
    • Input vector encoded as optical amplitudes
    • Matrix-vector product completes at speed of light
    ↓
Photodetectors → ADC (optical → digital)
    ↓
DCI (12nm CMOS): nonlinear activation, accumulation, control
    ↓
Next layer / output
```

### Key Constraints
- **Precision**: Photonic operations are inherently analog. Lightmatter uses Adaptive Block Floating-Point (ABFP16) — a custom 16-bit format with shared exponents per block — to achieve near-digital accuracy while tolerating analog noise
- **Weight update**: Reprogramming MZI phases requires updating electro-optic voltages; faster than re-fabbing but slower than SRAM writes
- **Accumulation**: Multi-bit precision for large matrices requires multiple optical passes or ADC accumulation in the digital domain

---

## Q2: What operations can it accelerate? (matmul only? attention?)

### What Photonic Cores Accelerate Directly
- **Dense matrix-vector multiply (MVM)**: The core native operation. Each 128×128 PTC performs one MVM per optical pass
- **Batched GEMM / matrix-matrix multiply**: achieved by sequential MVMs with different input batches
- **Linear layers in transformers**: fully-connected projections (Q, K, V, O, FFN up/down projections) map directly
- **Conv layers** (as GEMM via im2col): supported via Idiom compiler transformation
- **ResNet, BERT, Atari RL**: demonstrated in Nature 2025 at near-FP32 accuracy

### What Requires Electronic Assistance
- **Attention score computation** (softmax(QKᵀ/√d) · V): QKᵀ and PV matmuls run on photonic cores; **softmax (nonlinear) runs on DCI electronics**
- **Layer normalization, ReLU, GELU**: all nonlinear activations run on 12nm DCI chips
- **Residual adds, element-wise ops**: electronics
- **Sparse / irregular ops**: not accelerated; fall back to DCI or host CPU

### Operational Model
Lightmatter's architecture is a **systolic analog accelerator for dense linear algebra** with a co-packaged digital controller. The 4:2 ratio of photonic-to-digital chips (4 PTCs + 2 DCIs per package) reflects the compute-heavy nature of matmul vs. the lighter nonlinear activation workload.

### Supported Models (as of Nature 2025 / Idiom)
- ResNet-50, ResNet-101 (image classification)
- BERT-base, BERT-large (language understanding)
- Deep Q-Network (Atari RL)
- Transformer inference (LLM forward pass via idiom compiler partitioning)

---

## Q3: What is the electronic-photonic interface?

### Package Architecture (Nature 2025)

The Envise chip is a **6-die vertically integrated package**:

```
┌─────────────────────────────────────────────┐
│         Envise Package (multi-chip module)  │
│                                             │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│  │PTC-0 │ │PTC-1 │ │PTC-2 │ │PTC-3 │      │
│  │128×  │ │128×  │ │128×  │ │128×  │      │
│  │128   │ │128   │ │128   │ │128   │      │
│  │MZI   │ │MZI   │ │MZI   │ │MZI   │      │
│  │mesh  │ │mesh  │ │mesh  │ │mesh  │      │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │
│     │        │        │        │            │
│  ┌──┴────────┴────────┴────────┴───────┐   │
│  │     DCI-0 (12nm CMOS)               │   │
│  │   DAC/ADC | Activation | Control    │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │     DCI-1 (12nm CMOS)               │   │
│  │   DAC/ADC | Activation | Control    │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Laser source (on-package or fiber-coupled) │
└─────────────────────────────────────────────┘
```

### Interface Details
- **DAC (DCI → PTC)**: converts digital activation values to optical modulator drive voltages; encodes input vector as optical amplitudes on 512 wavelength-division-multiplexed (WDM) channels
- **ADC (PTC → DCI)**: photodetectors convert output optical power to photocurrent; transimpedance amplifiers (TIA) + ADC digitize; ~78 W total electrical power for the package
- **MZI weight programming**: DCI writes phase voltages to MZI electrodes via a high-speed DAC array; weights are loaded per-layer execution
- **Memory**: no on-chip SRAM for activations in the photonic cores; DCI holds activations in registers / LPDDR buffer between layers; Envise 4S server has 1 TB DDR4 + 3 TB NVMe flash per blade for model weights

### Envise 4S Server Specifications
| Parameter | Value |
|-----------|-------|
| Photonic chips per package | 4 PTCs + 2 DCIs |
| Packages per server | 4 (16 Envise chips total) |
| Form factor | 4U rack server |
| Host CPU | 2× AMD EPYC |
| System DRAM | 1 TB DDR4 |
| Storage | 3 TB NVMe flash |
| Throughput (reported) | 65.5 TOPS (ABFP16) per package |
| System power | ~3 kW |
| Scale-out interconnect | 6.4 Tbps optical (Passage) |
| Inference efficiency | ~240 ResNet-50 inferences/sec/W |

---

## Q4: How does Passage interconnect differ from NVLink?

### Passage Architecture

Passage is a **3D photonic interposer** platform — chips are stacked ON TOP of the Passage die, which provides optical I/O to the die complex. Two products:

**Passage M1000** (March 2025):
- 4,000+ mm² multi-reticle active photonic interposer (largest photonic chip ever)
- 114 Tbps total optical bandwidth
- 256 optical fiber connections
- Built-in solid-state optical circuit switching (OCS) — reconfigures connectivity without mechanical switching
- Enables thousands of GPUs in a single optical domain
- 1.5 kW+ power delivery to stacked die complex

**Passage L200** (March 2025):
- 3D co-packaged optics (CPO) — integrates directly with GPU/XPU die
- 32 Tbps and 64 Tbps versions
- >200 Tbps total I/O bandwidth per chip package
- 5–10× improvement over existing CPO bandwidth density

### Passage vs. NVLink — Key Differences

| Dimension | NVLink 5 (NVIDIA) | Passage M1000 (Lightmatter) |
|-----------|-------------------|------------------------------|
| **Medium** | Electrical copper (SerDes) | Photons (optical fiber / waveguide) |
| **Topology** | Fixed point-to-point + NVSwitch hub | Reconfigurable OCS — any-to-any |
| **Bandwidth/chip** | 1.8 TB/s (72-GPU NVL72) | 114 Tbps total per interposer (shared domain) |
| **I/O placement** | Edge of chip (pad-limited) | Any surface of die complex (3D interposer) |
| **Distance** | Sub-meter (within rack) | Multi-rack, data-center scale (optical fiber) |
| **Latency** | ~100 ns | ~1 ns/m propagation (fiber) |
| **Scale** | 576 GPUs max (NVL72 × N) | Thousands of XPUs in single domain |
| **Circuit switching** | No (fixed NVSwitch fabric) | Yes — solid-state OCS reconfigures domain |
| **Power efficiency** | ~5 pJ/bit | < 1 pJ/bit (photonic) |
| **Standard** | NVIDIA proprietary | XPO MSA (founding member, March 2026) |
| **Business model** | GPU-bundled | Third-party interposer (sold to hyperscalers/OEMs) |

### Strategic Distinction
NVLink is a **scale-up** technology (tightly coupling GPUs within a node/rack). Passage operates as both **scale-up** (via M1000 interposer for die-complex integration) and **scale-out** (via optical fiber to multi-rack clusters), blurring the traditional boundary between on-board interconnect and network fabric.

Passage's solid-state OCS enables **dynamic reconfiguration** of GPU clusters at nanosecond timescales — impossible with NVSwitch's fixed circuit topology.

---

# Update — 2026-08-08

*Sources fetched or verified 2026-08-08: lightmatter.co (homepage), lightmatter.co/products/, lightmatter.co/products/passage-l20/, lightmatter.co/press-release/lightmatter-expands-photonic-interconnect-roadmap-with-passage-l20-unified-optical-engine-for-npo-and-obo-applications/, photonicsonline.com (independent carry of the same release), theregister.com/2026/03/11/lightmatter_passge_l20_fiber/, lightmatter.co/products/guide/, electronicdesign.com (independent carry of the Guide DR release, 2026-05-21), lightmatter.co/products/vclick-optics/, lightmatter.co/press-release/lightmatter-joins-nvidia-nvlink-fusion/, lightmatter.co/news/, spectrum.ieee.org/nvlink-fusion-optics.*

**Global caveat.** Nothing described below has shipped. Passage L20 samples from late 2026; Guide DR samples from Q4 2026; the vendor's products page states only that "Evaluation kits are sampling today." Every performance figure is a vendor marketing claim carried by press release or product page; none is independently verified, and no third-party benchmark or deployment exists.

---

## Q5: What changed in Lightmatter's product architecture between 2026-04-05 and 2026-08-08?

### Q5a — Portfolio composition

As of 2026-08-08, lightmatter.co's homepage and `/products/` page list:

- **Passage L200** — CPO, 32–64 Tbps aggregate, 112G PAM4
- **Passage L20** — NPO/OBO unified optical engine, 12.8 Tbps aggregate
- **Guide 1** — VLSP Light Engine
- **Guide DR** — liquid-cooled Laser NIC
- **Evaluation kits**: Passage M1000 EVK, Passage EVK100, Passage EVK50, Guide 1 EVK

**Envise appears nowhere** in product navigation or descriptive content. It survives only in the footer trademark line: "…Idiom, Guide, Passage, Envise, Edgeless I/O, eClick, vClick, A Giant Leap, and VLSP are all trademarks of Lightmatter, Inc." CRN's July 2026 semiconductor-startup roundup likewise lists Lightmatter's products as Passage and Guide only.

This is **evidence of absence on a marketing site, not an announced discontinuation** — there is no EOL notice and no press release. Confidence: medium. The correct survey statement is that Lightmatter's public positioning has shifted to photonic interconnect and optics and Envise is no longer marketed — not that Envise was discontinued.

M1000, previously documented here as a product tier, is now presented as **Passage M1000 EVK**, a rack-level 3D-photonic-interposer reference/evaluation platform (114 Tbps).

### Q5b — Announcement timeline, and which items were repo misses

| Date | Item | Coverage status |
|---|---|---|
| 2026-01-26/27 | Guide VLSP Light Engine unveiled | **Repo miss** — predates the 2026-04-05 baseline |
| 2026-03-11 | Passage L20 announced; vClick dFAU announced alongside | **Repo miss** — predates the baseline |
| 2026-03-15/19 | L20 demonstrated at OFC, Los Angeles | Repo miss |
| 2026-05-21 | Guide DR liquid-cooled Laser NIC announced | New in window |
| ~2026-06-03 | Lightmatter joins NVIDIA NVLink Fusion (own press release) | New in window |
| 2026-06-25 | AI Breakthrough "AI Semiconductor Innovation" award | New in window (marketing only) |
| — | Envise absent from product navigation | Observed 2026-08-08 |

Three of the four new products predate this survey's baseline; they are coverage gaps rather than fresh news.

---

## Q6: What is Passage L20 architecturally, and where does it sit relative to L200 and pluggables?

L20 is a **new tier**, not a refresh: near-package optics (NPO) sits between pluggable transceivers and full co-packaged optics. One part number serves two deployment styles:

- **NPO**: mounted on the switch board adjacent to the switch ASIC (a couple of inches from the die)
- **OBO**: mounted via flexible-PCB or mezzanine card, adding a retimer for extended reach

Vendor-stated specifications:

| Parameter | Value |
|---|---|
| Bandwidth | 6.4 Tbps per direction / 12.8 Tbps aggregate |
| Optical ports / fibers | 32, bidirectional (BiDi) |
| Per-lane rate | 200 Gbps |
| SerDes | 212.5 Gbps PAM4 |
| Electrical signaling | IEEE 802.3dj-compliant |
| Max TDP | 30 W |
| Fiber coupling | Corning CPO FlexConnect |
| Thermal | A2 ASHRAE-compliant cold plate |
| Availability | Sampling begins late 2026 |

**On the "6.4 vs 12.8 Tbps discrepancy": there is none.** Lightmatter's press release states "6.4 Tbps (each direction)"; its homepage states "12.8 Tbps of aggregate bandwidth and 4x pluggable density." 32 ports × 200 Gbps × 2 directions = 12.8 Tbps. These are the same figure in two conventions and must be stated once as "6.4 Tbps/direction, 12.8 Tbps aggregate." Presenting them as duelling numbers would manufacture a false controversy.

**Two figures ARE genuinely inconsistent across Lightmatter's own materials** and require dual attribution:

| Parameter | Press release (2026-03-11) | Product page (2026-08-08) |
|---|---|---|
| Energy efficiency | 3.0 pJ/bit | 5 pJ/bit |
| Package | 2000-pin BGA | 1827-ball BGA, 37.5 × 26.4 mm |

**BiDi multiplexing** is the distinctive architectural property: upstream and downstream traffic share one fiber, which the vendor says cuts fiber count ~50% versus duplex/unidirectional DR optics. Vendor density claims: "4× pluggable density" and "88% smaller by volume compared to OSFP."

**Attribution correction.** The often-quoted line "just 16 L20s would be enough to replace 512, 200 Gbps pluggables in a 102.4 Tbps switch, while also reducing power consumption considerably" is **The Register's** framing. It does not appear in the vendor press release.

**Roadmap positioning.** CEO Nick Harris to The Register: *"I don't think that it's going to be a super-long roadmap for near package optics, because, of course, CPO is coming, and we think that's like a 2028 high-volume ramp."* The 2028 date is the load-bearing part of the quote.

---

## Q7: What is the Guide line, and why does it constitute a new subsystem tier?

CPO/NPO/OBO deployments need continuous-wave laser light supplied from outside the ASIC package, conventionally from racks of discrete **ELSFP** modules ("laser farms") that cost faceplate area, power and serviceability. Guide replaces that with an integrated light-source tier — a subsystem this survey had not previously modelled for Lightmatter at all.

### Guide 1 — "Very Large Scale Photonics" (VLSP) Light Engine (announced 2026-01-26/27)

- Integrates **hundreds of lasers and photonic components on a single chip**
- Built in an **HVM CMOS fab** with **on-PIC wavelength tunability**
- **100 mW+ optical power per fiber**
- **16 wavelengths** at 100 / 200 / 400+ GHz spacing, with a roadmap to **64** ("4 to 64 wavelengths with zero increase in assembly complexity")
- **CMIS-compliant telemetry, active stabilization and hyper-local thermal tuning** maintaining **sub-GHz wavelength precision without drift**
- **N+M redundancy**: on a laser failure the engine autonomously retunes a backup laser onto the failed wavelength
- Supports **CPO, NPO and OBO** deployments
- Named ecosystem collaborators at launch: **Cadence, Synopsys, GUC**; **PHIX** is a separately stated collaboration
- Guide 1 EVK is among the kits described as "sampling today"

### Guide DR — liquid-cooled Laser NIC (LNIC) (announced 2026-05-21)

Vendor bills it as the **industry's first liquid-cooled Laser NIC**, targeted at CPO scale-up:

| Parameter | Value |
|---|---|
| Form factor | OCP NIC 3.0, in-chassis, **zero front-panel footprint** |
| Fibers | up to 64 |
| Optical power | 200 mW per fiber |
| Lanes | 256 at 200G |
| Aggregate | 51.2 Tbps per module |
| 1RU configuration | up to four modules → **204.8 Tbps** CPO scale-up switching bandwidth |
| Rack density | ~4× vs conventional ELSFPs |
| Management | CMIS 5.3 over I2C / I3C; OCP MHS integration |
| Thermal | ASHRAE A2 compliant; liquid cooled |
| Availability | **sampling begins Q4 2026** |

**Correction to an earlier scan:** these per-unit specs *are* publicly retrievable (vendor press release, carried by Electronic Design and Morningstar/BusinessWire). The earlier "specs not retrievable" conclusion was an HTTP-403 access artifact, not an absence of evidence.

**Not disclosed:** Guide 1 per-unit fiber count and aggregate bandwidth; Guide DR wavelength count and electrical power draw.

---

## Q8: What is vClick dFAU and what problem does it solve?

Announced 2026-03-11 alongside L20. Trademark spelling **vClick**; product name **vClick dFAU**. It addresses optical-packaging *yield*, not bandwidth: fiber attach is conventionally permanent and late, so a defective optical engine is only discovered after full package assembly.

- Industry's first **detachable** fiber array unit (FAU) with **vertical / expanded-beam coupling** designed to survive advanced-packaging **mold-and-grind** flows
- Expanded-beam interface permits **passive alignment**
- **< 1.5 dB** insertion loss, sustained across insertion *and* reinsertion
- Developed with **SENKO** (SEAT and Metallic PIC Coupler integration)
- Enables **known-good optical engine verification at the wafer level** before final assembly
- Vendor calls it "production-validated"; **no GA date given**

Characterizing vClick as a "surface-attach fiber array" is wrong and inverts the headline property, which is detachability.

---

## Q9: What is Lightmatter's ecosystem and partner position as of 2026-08-08?

### NVIDIA NVLink Fusion

Announced in Lightmatter's own press release dated **~2026-06-03** (not, as sometimes reported, by IEEE Spectrum on 2026-07-09 — that is downstream coverage over a month later). Stated role: deliver Passage **CPO and NPO** products optically and electrically compatible with NVIDIA's optical and SerDes technologies for semi-custom AI factories, with a claimed **~50% reduction in fiber and connector count**. Harris: "This is what the next era of AI infrastructure looks like… combining the industry's most advanced AI platform and the world's leading interconnect."

Attribution discipline:
- The release **names neither Ayar Labs nor Marvell**. The "Lightmatter alongside Ayar Labs and Marvell" grouping is IEEE Spectrum's.
- Ayar Labs is independently confirmed as a separate NVLink Fusion joiner (2026-06-02). **Marvell's inclusion is not independently confirmed.**
- The Roy Kim "packaging is no longer the bottleneck" remark and the 72 → 576 GPU scale-up-domain figures are attributable to **IEEE Spectrum only**, not independently verified.
- The ~2028 high-volume optical scale-up expectation **is** independently corroborated by Harris's own "2028 high-volume ramp" remark to The Register.

### Manufacturing / packaging partners

| Partner | Status |
|---|---|
| GlobalFoundries, ASE, Amkor | Confirmed (long-standing, pre-baseline) |
| GUC, Cadence, Synopsys | Confirmed — Guide/VLSP design ecosystem |
| SENKO | Confirmed — vClick dFAU |
| Corning | Confirmed — CPO FlexConnect on L20 |
| PHIX | Confirmed — stated collaboration |
| **TSMC** | **Not confirmed** as a named Lightmatter manufacturing partner from any primary source in this window (only a third-party LinkedIn post surfaced). The survey's generic reference to TSMC silicon photonics as an industry enabler must not be upgraded to a stated partnership. |
| **Tower Semiconductor** | **Not a partner — dropped.** The only Tower announcement in the period is Tower + LightIC Technologies (an unrelated FMCW-LiDAR silicon-photonics company, 2026-01-05), which appears to be the origin of the conflation. |

### Funding

Unchanged: **$850M total raised at a $4.4B valuation**, from the October 2024 $400M Series D (T. Rowe Price-led). No new round in the Apr–Aug 2026 window. Still reported at these figures by CRN in July 2026.

---

## Q10: Negatives, and how to state them correctly

| Item | Correct statement |
|---|---|
| MLPerf | **Not applicable.** With no marketed compute product, Lightmatter has nothing submittable. Do not record this as "no results." |
| Hot Chips 38 | The conference runs **2026-08-23/25** at Stanford Memorial Auditorium — *after* this document's as_of date. Lightmatter **does not appear in the advance program**. "No Hot Chips 2026 talk" is a premature negative; re-check after the conference. |
| New Envise silicon | None found in the window. |
| Nature follow-up paper | None found in the window. |
| Shipping product | **None.** L20 samples late 2026; Guide DR samples Q4 2026; only evaluation kits are described as sampling today. |

---

## Methodological caveat on the 2026-08-08 verification pass

The verification session's WebSearch quota was exhausted at the first call; independent discovery was performed by fetching DuckDuckGo HTML/Lite result pages and then retrieving primary pages directly. Several outlets (Converge Digest, Data Center Dynamics, TechPowerUp, BusinessWire, hotchips.org) returned HTTP 403 or timed out. The Amkor, Morningstar, Tower/LightIC and funding datapoints rest partly on search-result snippets rather than full-page retrieval and are labelled `snippet` confidence in the YAML sidecar.
