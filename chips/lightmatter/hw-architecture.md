# Lightmatter Envise + Passage + Guide — Hardware Architecture

*as_of: 2026-08-08*
*chip: lightmatter*
*device_class: Photonic Interconnect + Optics (compute product no longer marketed) — historically Photonic Compute + Interconnect*
*generations: Envise (photonic compute, Nature 2025 — no longer marketed) / Passage M1000 (2025, now an EVK) / Passage L200 CPO (2025) / Passage L20 NPO+OBO (2026) / Guide 1 VLSP light engine (2026) / Guide DR Laser NIC (2026) / vClick dFAU (2026)*
*primary sources: https://www.nature.com/articles/s41586-025-08854-x, https://lightmatter.co/products/, https://lightmatter.co/products/passage-l20/, https://lightmatter.co/products/guide/, https://lightmatter.co/products/vclick-optics/, https://lightmatter.co/blog/seeing-is-believing*

---

## Generation / Product Overview

All Passage and Guide figures below are **vendor-stated and independently unverified**. Nothing in the 2026 product line has shipped: Passage L20 samples from late 2026, Guide DR from Q4 2026, and Lightmatter's products page states only that "Evaluation kits are sampling today."

| Product | Announced | Class | Headline bandwidth / optical power | Fibers | Power | Key architectural property | Status 2026-08-08 |
|---|---|---|---|---|---|---|---|
| Envise (Nature 2025 package) | 2025-04 (paper) | Photonic compute (MZI MVM) | 65.5 TOPS ABFP16/package | — | ~78 W elec + 1.6 W optical | 4× 128×128 PTC + 2× 12nm DCI, 6-die MCM | **No longer marketed** (absent from vendor product nav) |
| Passage M1000 | 2025-03-31 | 3D multi-reticle active photonic interposer | 114 Tbps total | 256 | 1.5 kW+ delivery to stacked dies | Solid-state OCS; any-surface electro-optical I/O; 4,000+ mm² | Repositioned as **Passage M1000 EVK** (reference platform) |
| Passage L200 | 2025-03-31 | Co-packaged optics (CPO) | 32–64 Tbps aggregate | not disclosed | not disclosed | 112G PAM4; >200 Tbps total I/O per chip package | Listed product |
| **Passage L20** | **2026-03-11** | **Near-package optics (NPO) + on-board optics (OBO)** | **6.4 Tbps/direction, 12.8 Tbps aggregate** | **32 ports at 200 Gbps/lane, bidirectional** | **30 W max TDP** | **BiDi multiplexing on a single fiber; 212.5 Gbps PAM4; IEEE 802.3dj electrical signaling; Corning CPO FlexConnect** | **Announced; sampling late 2026** |
| **Guide 1 (VLSP Light Engine)** | **2026-01-26/27** | **Integrated laser / external light source** | **100 mW+ optical power per fiber; 16 wavelengths (roadmap to 64)** | not disclosed | not disclosed | **Hundreds of lasers on one chip in an HVM CMOS fab; on-PIC wavelength tunability; N+M laser redundancy** | **Guide 1 EVK sampling** |
| **Guide DR (Laser NIC)** | **2026-05-21** | **Liquid-cooled Laser NIC (LNIC) for CPO scale-up** | **200 mW/fiber; 51.2 Tbps per module; 204.8 Tbps per 1RU (4 modules)** | **up to 64** | not disclosed | **OCP NIC 3.0, in-chassis, zero front-panel footprint; 256 lanes at 200G; ~4× ELSFP rack density** | **Announced; sampling Q4 2026** |
| **vClick dFAU** | **2026-03-11** | **Detachable fiber array unit (packaging component)** | — | — | — | **Vertical / expanded-beam coupling; survives mold-and-grind; < 1.5 dB insertion and reinsertion loss; wafer-level known-good optical engine test** | **"Production-validated"; no GA date** |

**Architectural significance of the 2026 additions.** Two of them are genuinely new tiers rather than refreshes of existing ones:

1. **A near-package optics tier (L20)** sits between pluggable transceivers and full CPO — mounted on the switch board adjacent to the ASIC, or on a flexible-PCB/mezzanine card for OBO (which adds a retimer for extended reach). CEO Nick Harris frames it as a deliberately short-lived bridge: *"I don't think that it's going to be a super-long roadmap for near package optics, because, of course, CPO is coming, and we think that's like a 2028 high-volume ramp."*
2. **An external light-source tier (Guide)** did not exist in this survey's prior model of Lightmatter's architecture at all. Co-packaged and near-package optics need laser light supplied from outside the compute/switch package; Guide replaces the discrete ELSFP "laser farm" with an integrated, tunable, telemetered light engine.

---

## Fundamental Architecture: Light as the Compute Medium

> **Historical section (retained deliberately).** Envise is no longer listed in Lightmatter's product navigation as of 2026-08-08 — see the Generation / Product Overview above and the 2026 update in `summary.md`. There is no announced discontinuation; the product is simply no longer marketed. The Nature 2025 result described below is published, peer-reviewed science and stands on its own. Sections up to and including "Precision: ABFP16" describe the Envise compute architecture as of that paper.

Lightmatter's Envise chip performs matrix-vector multiplication using **photons instead of electrons**. The central insight: when light propagates through a cascade of beam-splitters and phase shifters (a Mach-Zehnder interferometer mesh), the optical amplitude at each output is a linear combination of the input amplitudes — exactly a matrix-vector product.

This is not a simulation of matmul in photonics. It is **direct physical embodiment of linear algebra**: the matrix is encoded in the physical geometry and electrical phase settings of the waveguides, and the result is the light that emerges from the other end.

---

## Mach-Zehnder Interferometer (MZI) — The Primitive

```
                  ┌─ Phase shifter (θ) ─┐
Input 0 ──────── ┤ Y-splitter           ├── Y-combiner ──── Output 0
                  │                     │   cos(θ/2)·in0 + i·sin(θ/2)·in1
Input 1 ──────── ┤                      ├── Y-combiner ──── Output 1
                  └─────────────────────┘   i·sin(θ/2)·in0 + cos(θ/2)·in1
```

- A single MZI implements a **2×2 unitary rotation** parameterized by phase θ
- Phase θ is set by applying voltage to an electro-optic (Pockels-effect) electrode
- The operation is reversible, passive, and requires **zero switching energy** for the matrix weight itself — only the laser and detector consume power

### From MZI to Full Matrix

Any N×N complex matrix W decomposes via SVD: **W = U · Σ · V†**

- **U** (unitary): triangular MZI mesh (N×(N-1)/2 MZIs)
- **Σ** (diagonal): optical attenuators (variable optical amplitude)
- **V†** (unitary): second triangular MZI mesh

A **128×128 Photonic Tensor Core** in Envise implements this full 128×128 matrix-vector product in one optical pass (~1 ns traversal time).

---

## Package Architecture (Nature 2025)

```
┌─────────────────────────────────────────────────────────────┐
│                  Envise Package (MCM, 6 dies)                │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  PTC-0   │  │  PTC-1   │  │  PTC-2   │  │  PTC-3   │  │
│  │ 128×128  │  │ 128×128  │  │ 128×128  │  │ 128×128  │  │
│  │ MZI mesh │  │ MZI mesh │  │ MZI mesh │  │ MZI mesh │  │
│  │ + U,Σ,V† │  │ + U,Σ,V† │  │ + U,Σ,V† │  │ + U,Σ,V† │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │ optical     │              │              │         │
│  ┌────┴─────────────┴──────────────┴──────────────┴─────┐  │
│  │         DCI-0 (12nm CMOS Digital Control)            │  │
│  │  DAC arrays → modulator drivers (512 WDM channels)   │  │
│  │  Photodetector arrays → ADC → digital activations    │  │
│  │  Nonlinear activations, LayerNorm, Softmax            │  │
│  │  Phase calibration control (MZI drift compensation)  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         DCI-1 (12nm CMOS Digital Control)            │  │
│  │  (redundant/parallel DCI for second PTC pair)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Laser source (on-package or fiber-coupled)                 │
│  WDM multiplexer: 512 optical channels                      │
└─────────────────────────────────────────────────────────────┘
        │ PCIe / Passage interconnect
        ↓
    Host CPU + DDR4 DRAM + NVMe
```

**Key numbers (Nature 2025):**
- 65.5 TOPS (ABFP16) per package
- ~78 W electrical + 1.6 W optical
- Near-FP32 accuracy on ResNet, BERT, Atari RL

---

## Compute Execution Flow

```
1. Host CPU loads weight matrix W (float32)
        ↓
2. idCompile (offline): W → SVD → MZI phase angles θ_i
   (one-time compilation per model layer)
        ↓
3. Runtime loads phase voltages to MZI electrodes via DCI DAC array
   (once per layer; ~microseconds)
        ↓
4. Input activation vector x encoded as optical amplitudes
   via DCI DAC → laser modulator → 512 WDM channels
        ↓
5. Light traverses 128×128 MZI mesh
   → matrix-vector product y = W·x computed at speed of light (~1 ns)
        ↓
6. Photodetectors measure output optical power
   → TIA + ADC in DCI → digital result vector y
        ↓
7. DCI applies nonlinear activation σ(y) in CMOS electronics
        ↓
8. Result fed back to Step 4 for next layer
```

**Bottleneck**: DAC/ADC conversion (~10s of ns) and weight reprogramming between layers, not the photonic compute itself.

---

## Memory Hierarchy

| Level | Type | Capacity | Bandwidth | Location |
|-------|------|----------|-----------|----------|
| MZI weight register | Electro-optic voltage | 128×128 per PTC = 16K weights | Instantaneous (optical) | On photonic die |
| DCI activation buffer | SRAM / registers | Small (inter-layer activations) | Tbps | On 12nm DCI die |
| DDR4 system DRAM | DRAM | 1 TB (4S server) | ~300 GB/s | Off-package |
| NVMe flash | Flash | 3 TB (4S server) | ~10 GB/s | Off-package |

**Note**: Envise has no HBM. The photonic architecture does not require the massive DRAM bandwidth that GPU architectures need, because the compute (linear layer) happens in the optical domain with weights stored in MZI phase states rather than loaded from memory per operation.

---

## Passage M1000 — 3D Photonic Interposer Architecture

```
┌────────────────────────────────────────────────────────┐
│          Passage M1000 (4,000+ mm² interposer)         │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Die complex stacked on top (GPUs/XPUs/ASICs)   │  │
│  │  Micro-bumped onto M1000 active photonic surface │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Active photonic layer (waveguides + grating couplers) │
│  ├── Electro-optical I/O: any surface (not edge-only)  │
│  ├── Solid-state Optical Circuit Switch (OCS)          │
│  │   • Nanosecond reconfiguration                      │
│  │   • Any-to-any connectivity within domain           │
│  ├── 256 fiber connections (edge + surface)            │
│  ├── 114 Tbps total optical bandwidth                  │
│  └── 1.5 kW+ power delivery to die complex            │
│                                                        │
│  Scale: connects thousands of XPUs in single domain   │
└────────────────────────────────────────────────────────┘
           │ 256 optical fibers
           ↓
   Multi-rack / data-center optical network
```

**Status note (2026-08-08):** M1000 is now presented by the vendor as **Passage M1000 EVK**, a rack-level 3D-photonic-interposer *reference/evaluation platform* (114 Tbps), alongside Passage EVK100 and EVK50, rather than as a shipping product tier.

---

## Passage L20 — Near-Package Optics (NPO) and On-Board Optics (OBO)

*Announced 2026-03-11; demonstrated at OFC, Los Angeles, 2026-03-15/19. Sampling begins late 2026 — announced only, not shipping. All figures vendor-stated.*

L20 is a **unified optical engine**: one part number serving two deployment styles that previously required different hardware.

```
        NPO deployment                          OBO deployment
┌──────────────────────────────┐      ┌──────────────────────────────────┐
│  Switch board                │      │  Switch board                    │
│  ┌────────┐   ┌──────────┐   │      │  ┌────────┐                      │
│  │ Switch │◄─►│ Passage  │   │      │  │ Switch │◄── 212.5G PAM4 ──┐   │
│  │  ASIC  │   │   L20    │───┼─►    │  │  ASIC  │                  │   │
│  └────────┘   └──────────┘   │      │  └────────┘        ┌─────────▼─┐ │
│   ~a couple of inches apart  │      │      flex-PCB /    │ retimer + │ │
│   (adjacent, on-board)       │      │      mezzanine ───►│ Passage   │─┼─►
└──────────────────────────────┘      │      card          │   L20     │ │
                                      │   (extended reach) └───────────┘ │
                                      └──────────────────────────────────┘
```

| Parameter | Value | Source note |
|---|---|---|
| Aggregate bandwidth | 6.4 Tbps per direction / **12.8 Tbps aggregate** | Press release states per-direction; homepage states aggregate. **Not a discrepancy** — 32 × 200 Gbps × 2 = 12.8 Tbps |
| Optical ports / fibers | 32, bidirectional (BiDi) | Each fiber carries upstream and downstream simultaneously |
| Per-lane rate | 200 Gbps | |
| SerDes line rate | 212.5 Gbps PAM4 | |
| Electrical signaling | IEEE 802.3dj-compliant | |
| Max TDP | 30 W | |
| Energy efficiency | **3.0 pJ/bit** (press release) / **5 pJ/bit** (product page) | Genuine internal inconsistency in vendor materials — quote both |
| Package | **2000-pin BGA** (press release) / **1827-ball BGA, 37.5 × 26.4 mm** (product page) | Genuine internal inconsistency — quote both |
| Fiber coupling | Corning CPO FlexConnect | |
| Thermal | A2 ASHRAE-compliant cold plate | |
| Density claims | "4× pluggable density"; "88% smaller by volume compared to OSFP" | Vendor |
| Fiber-count claim | BiDi cuts fiber count ~50% vs duplex/unidirectional DR optics | Vendor |
| Third-party framing | "Just 16 L20s would be enough to replace 512, 200 Gbps pluggables in a 102.4 Tbps switch" | **The Register's** framing, *not* a vendor claim |

**Why BiDi matters architecturally.** Conventional DR optics dedicate one fiber per direction. L20 multiplexes both directions onto a single fiber, so a given switch faceplate/board fiber budget carries twice the traffic. At datacenter scale the binding constraint on optical scale-up is increasingly fiber and connector count, not raw Tbps — which is also the metric Lightmatter cites for its NVLink Fusion work (~50% reduction in fiber and connector count).

---

## Guide — VLSP Light Engine and Laser NIC (New Subsystem Tier)

*Guide 1 announced 2026-01-26/27; Guide DR announced 2026-05-21 (sampling Q4 2026). All figures vendor-stated and independently unverified.*

Co-packaged and near-package optics require laser light delivered from outside the ASIC package, conventionally via racks of discrete **ELSFP** (external laser small form-factor pluggable) modules — a "laser farm" that consumes faceplate area, power and serviceability budget. Guide replaces that farm with an integrated light-source tier.

```
┌─────────────────────────────────────────────────────────────────┐
│  Guide 1 — "Very Large Scale Photonics" (VLSP) Light Engine     │
│                                                                 │
│  Hundreds of lasers + photonic components on a single chip      │
│  built in an HVM CMOS fab, with on-PIC wavelength tunability    │
│                                                                 │
│  • 100 mW+ optical power per fiber                              │
│  • 16 wavelengths @ 100 / 200 / 400+ GHz spacing                │
│    (roadmap: 4 → 64 wavelengths, "zero increase in              │
│     assembly complexity")                                       │
│  • CMIS-compliant telemetry + active stabilization              │
│    + hyper-local thermal tuning → sub-GHz wavelength precision  │
│  • N+M redundancy: autonomously retunes a backup laser          │
│    onto a failed laser's wavelength                             │
│  • Deployment targets: CPO, NPO and OBO                         │
└─────────────────────────────────────────────────────────────────┘
                     │ fiber-delivered CW light
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Guide DR — liquid-cooled Laser NIC (LNIC), OCP NIC 3.0         │
│                                                                 │
│  • In-chassis, **zero front-panel footprint** (contrast ELSFP)  │
│  • Up to 64 fibers, 200 mW optical power per fiber              │
│  • Supports 256 lanes @ 200G → 51.2 Tbps aggregate per module   │
│  • Up to 4 modules in 1RU → 204.8 Tbps CPO scale-up bandwidth   │
│  • ~4× rack density vs conventional ELSFPs                      │
│  • CMIS 5.3 management over I2C / I3C; OCP MHS integration      │
│  • ASHRAE A2 thermal compliance; liquid cooled                  │
└─────────────────────────────────────────────────────────────────┘
```

Design-ecosystem collaborators named at the Guide launch: **Cadence, Synopsys, GUC**. **PHIX** is a separately stated collaboration.

**Not disclosed:** Guide 1 per-unit fiber count and aggregate bandwidth; Guide DR wavelength count and per-module power draw.

---

## vClick dFAU — Detachable Fiber Array Unit (Packaging)

*Announced 2026-03-11 alongside L20. Trademark spelling: **vClick**; product name **vClick dFAU**. Vendor describes it as "production-validated"; no GA date given.*

vClick addresses an optical-packaging yield problem rather than a bandwidth problem. Fiber attach is conventionally permanent and happens late, so a defective optical engine is discovered only after the full package has been assembled.

- Industry's first **detachable** fiber array unit (FAU) with **vertical / expanded-beam coupling** designed to survive advanced-packaging **mold-and-grind** flows
- Expanded-beam interface permits **passive alignment** (no active alignment step)
- **< 1.5 dB** insertion loss, maintained across both insertion *and* reinsertion
- Developed with **SENKO** (SEAT and Metallic PIC Coupler integration)
- Enables **known-good optical engine verification at the wafer level**, before final assembly

The headline property is *detachability*: connect for wafer-level test, detach, complete packaging, reconnect. Describing it as a "surface-attach" array inverts the point.

---

## Interconnect Ecosystem Position (2026)

| Item | Detail |
|---|---|
| **NVIDIA NVLink Fusion** | Lightmatter announced joining the NVLink Fusion ecosystem in its own press release ~2026-06-03. Stated role: deliver Passage **CPO and NPO** products optically and electrically compatible with NVIDIA's optical and SerDes technologies for semi-custom AI factories; claimed ~50% reduction in fiber and connector count. The press release names neither Ayar Labs nor Marvell — the "alongside Ayar Labs and Marvell" grouping is IEEE Spectrum's (2026-07-09) framing. Ayar Labs is independently confirmed as a separate joiner (2026-06-02); **Marvell's inclusion is not independently confirmed**. |
| **XPO MSA** | Founding member, March 2026 (unchanged) |
| **Standards touched by the 2026 line** | IEEE 802.3dj (L20 electrical signaling); CMIS / CMIS 5.3 (Guide, Guide DR); OCP NIC 3.0 and OCP MHS (Guide DR); ASHRAE A2 (L20, Guide DR) |
| **Roadmap** | NPO explicitly a bridge; CEO Harris expects a **~2028 high-volume CPO ramp**. IEEE Spectrum separately reports an industry expectation of high-volume optical scale-up around 2028 with scale-up domains growing from 72 toward 576 GPUs by 2027 — attributable to Spectrum only, not independently verified. |
| **Confirmed manufacturing / packaging partners** | GlobalFoundries, ASE, Amkor (pre-baseline); Corning (L20 fiber coupling); SENKO (vClick); PHIX; GUC, Cadence, Synopsys (Guide design ecosystem) |
| **Not confirmed** | **TSMC** as a named Lightmatter manufacturing partner (no primary source in this window). **Tower Semiconductor** — no Lightmatter relationship exists; the only Tower announcement in the period is Tower + LightIC (FMCW-LiDAR, 2026-01-05). |

---

## Precision: ABFP16 (Adaptive Block Floating-Point 16-bit)

Photonic analog compute introduces noise: laser shot noise, MZI phase errors, thermal drift. Standard FP16 or INT8 quantization maps poorly to photonic dynamic range.

Lightmatter's ABFP16 format:
- Groups matrix elements into blocks
- Uses a **shared exponent** per block (like block floating-point) to maximize the range used by photonic amplitude modulation
- Individual elements use 16-bit mantissa within the block
- Compiler selects block boundaries to minimize quantization error relative to photonic noise floor

Result: near-FP32 accuracy (confirmed in Nature 2025 across ResNet, BERT, Atari RL) without the digital-equivalent energy cost of FP32.

---

## Key Architecture Distinctions vs. Electronic Accelerators

| Aspect | Lightmatter Envise | NVIDIA H100 | Google TPU v4 |
|--------|-------------------|-------------|---------------|
| Compute medium | Photons (MZI mesh) | Electrons (transistors) | Electrons (systolic array) |
| Matmul energy | Sub-linear in matrix dimension | Linear in ops count | Linear in ops count |
| Weight storage | MZI phase voltages (in-mesh) | HBM (off-chip DRAM) | HBM (off-chip DRAM) |
| Weight load bandwidth needed | None during compute (weights in mesh) | 8 TB/s HBM BW | 4.4 TB/s HBM BW |
| Nonlinear ops | Electronic (12nm CMOS DCI) | Electronic (CUDA cores) | Electronic (Vector units) |
| Precision format | ABFP16 (custom analog) | FP4–FP64 | BF16/FP8 |
| Reconfigurability | Layer-by-layer (phase reload) | Per-instruction (CUDA) | Compiler-scheduled |
| Interconnect | Passage (photonic, OCS) | NVLink (copper SerDes) | ICI (copper+optical torus) |

*This comparison table describes Envise as documented in Nature 2025. It is retained as architectural history; Envise is no longer a marketed product (2026-08-08), and Lightmatter's current lineup contains no compute device. Consequently MLPerf is **not applicable** to Lightmatter rather than merely absent.*

**Hot Chips 38** (Aug 23–25, 2026, Stanford Memorial Auditorium) has not yet occurred as of this document's `as_of` date. Lightmatter does not appear in the advance program; this should be re-checked after the conference rather than recorded as a settled negative.
