# Lightmatter — Summary

**Device class:** Photonic Interconnect + Optics (compute product no longer marketed, as of 2026-08-08) — historically classified in this survey as *Photonic Compute + Interconnect*
**Manufacturer:** Lightmatter (founded 2017; MIT spinout; private)
**Products (as listed by the vendor on 2026-08-08):** Passage L200 (CPO), Passage L20 (NPO/OBO), Guide 1 and Guide DR (VLSP laser light engines), vClick dFAU (detachable fiber array unit), plus four evaluation kits (Passage M1000 EVK, Passage EVK100, Passage EVK50, Guide 1 EVK). **Envise** (photonic AI accelerator) no longer appears in vendor product navigation — see the 2026 update section below.
**Research date:** 2026-04-05 · **Last updated:** 2026-08-08
**Key sources:** https://lightmatter.co/products/, https://lightmatter.co/products/passage-l20/, https://lightmatter.co/products/guide/, https://lightmatter.co/products/vclick-optics/, https://www.nature.com/articles/s41586-025-08854-x, https://lightmatter.co/products/idiom/

---

## What It Is

> **Superseded framing — read with the 2026 update below.** Through the 2026-04-05 baseline this survey described Lightmatter as a two-product photonic company spanning compute (Envise) and interconnect (Passage). As of 2026-08-08 Lightmatter's public product portfolio is interconnect and optics only (Passage + Guide + vClick); Envise is no longer marketed. The Envise material in this section is retained as architectural and scientific history — the Nature 2025 result stands as published science — not as a description of the current lineup.

Lightmatter builds the world's first commercial photonic AI computing platform — a two-product portfolio that replaces electrons with photons at two critical AI datacenter bottlenecks: compute (Envise) and interconnect (Passage).

**Envise** is a photonic AI accelerator. Dense matrix multiplications — the dominant operation in transformer and CNN inference — are executed by arrays of Mach-Zehnder interferometers (MZIs) etched into silicon photonic chips. Light passing through cascaded MZI meshes computes matrix-vector products at the speed of light, consuming orders of magnitude less energy per operation than CMOS transistors for equivalent linear algebra. The April 2025 Nature paper demonstrated a 6-die package (4 photonic tensor cores + 2 digital control interface chips) achieving 65.5 TOPS at 78 W — running ResNet, BERT, and Atari RL at near-FP32 accuracy.

**Passage** is a 3D photonic interposer and co-packaged optics platform that replaces copper SerDes interconnects with photonic waveguides, enabling chip-to-chip and rack-to-rack connectivity at bandwidth densities impossible with electrical I/O. The March 2025 M1000 is a 4,000+ mm² multi-reticle active interposer delivering 114 Tbps, while the L200 provides 32–64 Tbps co-packaged optics with 200+ Tbps total I/O per chip package.

---

## Key Specifications

| Parameter | Envise Package (Nature 2025) | Envise 4S Server |
|-----------|------------------------------|------------------|
| Process | Silicon photonics + 12nm CMOS DCI | — |
| Photonic tensor cores | 4× PTC (128×128 MZI mesh) | 16 PTCs (4 packages) |
| Digital control | 2× DCI (12nm CMOS) | 8 DCIs |
| Peak throughput | 65.5 TOPS (ABFP16) | ~262 TOPS (est.) |
| Electrical power | ~78 W | ~3,000 W |
| Optical power | ~1.6 W | ~6.4 W |
| System DRAM | — | 1 TB DDR4 |
| Storage | — | 3 TB NVMe |
| Host CPU | — | 2× AMD EPYC |
| Scale-out | — | 6.4 Tbps (Passage) |
| Inference eff. | — | ~240 ResNet-50 inf/sec/W |

| Parameter | Passage M1000 | Passage L200 | Passage L20 *(new, 2026)* |
|-----------|---------------|--------------|---------------------------|
| Type | 3D multi-reticle photonic interposer | 3D co-packaged optics (CPO) | Unified optical engine for near-package optics (NPO) and on-board optics (OBO) |
| Die area | 4,000+ mm² | — | not disclosed (package: see note below) |
| Total bandwidth | 114 Tbps | 32/64 Tbps versions | 6.4 Tbps per direction / 12.8 Tbps aggregate |
| Total I/O/package | — | >200 Tbps | — |
| Fiber connections | 256 | — | 32 optical ports/fibers at 200 Gbps per lane, bidirectional (BiDi) |
| SerDes | — | 112G PAM4 | 212.5 Gbps PAM4; IEEE 802.3dj-compliant electrical signaling |
| Power | 1.5 kW+ delivery to stacked die | — | 30 W maximum TDP |
| Key features | Solid-state OCS, any-surface I/O | 5–10× vs prior CPO | BiDi multiplexing (vendor claims ~50% fiber-count reduction); A2 ASHRAE cold plate; Corning CPO FlexConnect fiber coupling; "88% smaller by volume compared to OSFP" (vendor) |
| Announced | March 31, 2025 | March 31, 2025 | March 11, 2026 (demoed at OFC, Los Angeles, March 15–19, 2026) |
| Status (2026-08-08) | Repositioned as **Passage M1000 EVK** — a rack-level reference/evaluation platform, not a product tier | Listed product | Announced only; **sampling begins late 2026** |

*All Passage figures are vendor-stated and independently unverified.*

**L20 internal spec inconsistencies in Lightmatter's own materials** (quote both, do not pick one):

| Parameter | Press release (2026-03-11) | Product page (fetched 2026-08-08) |
|-----------|----------------------------|-----------------------------------|
| Energy efficiency | 3.0 pJ/bit | 5 pJ/bit |
| Package | 2000-pin BGA | 1827-ball BGA, 37.5 × 26.4 mm |

*Note: the widely repeated "6.4 vs 12.8 Tbps discrepancy" is **not** a discrepancy — Lightmatter's press release states "6.4 Tbps (each direction)" and its homepage states "12.8 Tbps of aggregate bandwidth"; 32 ports × 200 Gbps × 2 directions = 12.8 Tbps. Same figure, two conventions.*

### Guide — VLSP Laser Light Engines (new product line, 2026)

| Parameter | Guide 1 (VLSP Light Engine) | Guide DR (Laser NIC) |
|-----------|-----------------------------|----------------------|
| Type | "Very Large Scale Photonics" integrated laser/light engine | Liquid-cooled Laser NIC (LNIC) for CPO scale-up — billed by vendor as an industry first |
| Announced | January 26–27, 2026 | May 21, 2026 |
| Optical power | 100 mW+ per fiber | 200 mW per fiber |
| Fibers | not disclosed per-unit | up to 64 |
| Wavelengths | 16 at 100/200/400+ GHz spacing; roadmap to 64 ("4 to 64 wavelengths with zero increase in assembly complexity") | not disclosed |
| Aggregate bandwidth | not disclosed | 51.2 Tbps per module; 256 lanes at 200G; 204.8 Tbps with four modules in 1RU |
| Form factor | on-chip integration of hundreds of lasers and photonic components; HVM CMOS fab with on-PIC wavelength tunability | OCP NIC 3.0, in-chassis, zero front-panel footprint; OCP MHS integration |
| Management | CMIS-compliant telemetry, active stabilization, hyper-local thermal tuning for sub-GHz wavelength precision | CMIS 5.3 over I2C/I3C |
| Reliability | N+M redundancy: autonomously retunes a backup laser on failure | — |
| Thermal | supports CPO, NPO and OBO deployments | ASHRAE A2 compliant; liquid cooled |
| Density claim | replaces discrete ELSFP "laser farms" | ~4× rack density vs conventional ELSFPs |
| Status (2026-08-08) | Guide 1 EVK listed among "evaluation kits sampling today" | Announced only; **sampling begins Q4 2026** |

*All Guide figures are vendor marketing claims, independently unverified.*

---

## Software Stack

```
User Python Script (PyTorch / TensorFlow)
         │  import lightmatter (Idiom)
         ↓
idCompile (Idiom Graph Compiler)
  • matmul → PTC | nonlinear → DCI electronics
  • ABFP16 quantization + phase-voltage encoding
  • Multi-blade partitioning, calibration tables
         ↓
Idiom Runtime
  • Device management, optical drift correction
  • Multi-blade orchestration
         ↓
DCI (12nm CMOS) ←→ 4× PTC (128×128 MZI)
  DAC/ADC + activation       Photonic matrix multiply
```

**No user-visible ISA.** No kernel library. No PTX equivalent. The photonic substrate is fully abstracted. Developers interact only with PyTorch/TensorFlow and an import statement.

---

## Programming Model Rationale: Why Photonic Compute for AI

### The Core Problem

AI inference and training are dominated by matrix multiplications (GEMM). As models scale, GEMM FLOPs grow as O(n²) or O(n³) while memory bandwidth grows only linearly with die area/stacking. Electronic CMOS accelerators (GPUs, TPUs) hit a wall: the energy cost per multiply-accumulate (MAC) is bounded by the physical energy of moving charges through transistors (~1–10 fJ/MAC), which fundamentally limits efficiency.

### The Photonic Advantage

In a Mach-Zehnder interferometer mesh:
- **Multiply** = optical interference (light passing through a waveguide takes zero switching energy for the matrix weights once set)
- **Add** = photodetector summing multiple optical signals (coherent superposition)
- **Energy cost** = dominated by DAC/ADC conversions and laser power, not the multiply itself
- **Speed** = photons traverse the chip in ~1 ns regardless of matrix dimension

For a 128×128 matrix-vector multiply:
- **Electronic GPU**: ~16,384 multiply-accumulate operations at ~10 fJ each = ~164 nJ
- **Photonic PTC**: one optical pass through MZI mesh, energy dominated by laser + photodetector = ~O(1 nJ), largely independent of matrix dimension

This **dimension-independent energy cost** means photonic efficiency improves quadratically relative to electronics as matrix size grows — a fundamental physical advantage, not an engineering one.

### Why Now?

1. **Silicon photonics maturity**: telecom-driven investment over 20 years produced manufacturing processes (e.g. TSMC Silicon Photonics — cited here as an industry-wide enabler, **not** as a named Lightmatter foundry partnership; Lightmatter's confirmed partners are GlobalFoundries, ASE and Amkor) capable of integrating 200,000+ optical components on a chip
2. **AI workload concentration**: transformers concentrate >80% of FLOPs in GEMM → photonic cores address the dominant operation
3. **ABFP16 precision**: Lightmatter's adaptive block floating-point format achieves near-FP32 accuracy on photonic analog hardware, closing the precision gap that previously blocked photonic deployment
4. **3D integration**: heterogeneous packaging (photonic + 12nm CMOS in one package) eliminates off-chip bandwidth as the bottleneck between analog and digital domains

### The Passage Interconnect Rationale

Electronic I/O (SerDes) is limited to chip edges (pad pitch ~50–100 µm). As chips scale, edge bandwidth does not scale with die area. Passage breaks this constraint by routing optical signals through the substrate in 3D, enabling I/O from any surface at bandwidth densities (>10 Tbps/mm²) impossible with copper.

Passage's built-in optical circuit switching also enables **dynamic cluster reconfiguration** — AI training cluster topology can change at nanosecond timescales, optimizing for different model parallelism strategies without physical rewiring.

---

## 2026 Update — Repositioning from Photonic Compute to Photonic Interconnect

*Updated 2026-08-08. Sources: lightmatter.co homepage and /products/ (both fetched 2026-08-08); Passage L20 press release and product page (2026-03-11); vClick Optics product page; Guide product page (announced 2026-01-26/27); Guide DR press release as carried by Electronic Design (2026-05-21); Lightmatter "Joins NVIDIA NVLink Fusion" press release (~2026-06-03); The Register (2026-03-11); IEEE Spectrum (2026-07-09).*

> **Status caveat governing this entire section: nothing described below has shipped or been deployed at scale.** Passage L20 sampling begins late 2026; Guide DR sampling begins Q4 2026. Lightmatter's own products page states only that "Evaluation kits are sampling today." Every performance number in this section is a vendor claim carried by press release; none is independently confirmed, and there are no third-party benchmarks, customer deployments, or MLPerf results.

### Timeline (and a note on repo coverage)

| Date | Event | In this survey's prior baseline? |
|---|---|---|
| 2026-01-26/27 | Guide "Very Large Scale Photonics (VLSP) Light Engine" unveiled | No — **repo miss** (predates the 2026-04-05 baseline) |
| 2026-03-11 | Passage L20 unified optical engine announced; vClick dFAU announced alongside | No — **repo miss** (predates the baseline) |
| 2026-03-15/19 | Passage L20 demonstrated at OFC, Los Angeles | No — repo miss |
| 2026-05-21 | Guide DR liquid-cooled Laser NIC announced | New |
| ~2026-06-03 | Lightmatter joins NVIDIA NVLink Fusion (own press release) | New |
| 2026-06-25 | AI Breakthrough "AI Semiconductor Innovation" award (marketing award, no technical content) | New |
| ongoing | Envise disappears from vendor product navigation | New (observed 2026-08-08) |

Three of the four new products predate this survey's 2026-04-05 baseline and are coverage gaps rather than fresh news; they are recorded here for completeness.

### Portfolio change — Envise is no longer a marketed product

As of 2026-08-08 the lightmatter.co homepage and `/products/` page list only Passage L200, Passage L20, Guide 1, Guide DR and four evaluation kits. **Envise appears nowhere in product navigation or descriptive content**; it survives only in the footer trademark line ("…Idiom, Guide, Passage, Envise, Edgeless I/O, eClick, vClick, A Giant Leap, and VLSP are all trademarks of Lightmatter, Inc."). The CRN semiconductor-startup roundup of July 2026 likewise lists Lightmatter's products as Passage and Guide.

**This is an absence on a marketing site, not an announced end-of-life.** There is no discontinuation notice, no EOL bulletin, and no press release. The accurate statement for the survey is that *Lightmatter's public positioning has shifted to photonic interconnect and optics, and Envise is no longer marketed as an active product* — **not** that Envise was discontinued. Confidence: medium (evidence of absence).

Consequences for this survey's taxonomy:
- Lightmatter's `device_class` moves from **Photonic Compute + Interconnect** to **Photonic Interconnect + Optics (compute product no longer marketed)**.
- The Nature 2025 Envise result (65.5 TOPS ABFP16 at ~78 W, near-FP32 accuracy on ResNet/BERT/Atari RL) **remains valid published science** and is retained above as architectural history.
- MLPerf is now **not applicable** rather than merely absent: with no marketed compute product, Lightmatter has nothing that could be submitted.

### Passage L20 — near-package optics as a bridge to CPO

Passage L20 is a new *tier* in the Passage line, not a refresh: a unified optical engine that sits between pluggable transceivers and full co-packaged optics. It mounts adjacent to the switch ASIC on the switch board (NPO) or via flexible-PCB / mezzanine integration (OBO, which adds a retimer for extended reach). Specifications are tabulated in the Key Specifications section above.

The architecturally distinctive claim is **bidirectional (BiDi) multiplexing**: each fiber carries upstream and downstream traffic simultaneously, which Lightmatter says cuts fiber count roughly 50% versus duplex/unidirectional DR optics. Vendor density claims are "4× pluggable density" and "88% smaller by volume compared to OSFP."

*Attribution note:* the frequently cited framing that "just 16 L20s would be enough to replace 512, 200 Gbps pluggables in a 102.4 Tbps switch" is **The Register's**, not a vendor press-release claim, and should be attributed as such.

**Roadmap framing.** CEO Nick Harris told The Register: *"I don't think that it's going to be a super-long roadmap for near package optics, because, of course, CPO is coming, and we think that's like a 2028 high-volume ramp."* L20 is explicitly positioned as a bridge product ahead of a ~2028 CPO volume ramp — the 2028 date is the load-bearing part of the quote for roadmap purposes.

### vClick dFAU — detachable fiber array unit

Announced 2026-03-11 alongside L20 (trademark spelling: **vClick**; product name **vClick dFAU**). Lightmatter bills it as the industry's first **detachable** fiber array unit with vertical / expanded-beam coupling, engineered to survive advanced-packaging mold-and-grind flows:

- Vertically expanded-beam interface enabling **passive alignment**
- **< 1.5 dB** insertion loss, sustained across insertion *and* reinsertion
- Developed with **SENKO** (SEAT and Metallic PIC Coupler integration)
- Enables **known-good optical engine verification at the wafer level**, before final assembly — the manufacturing-yield argument for high-volume CPO
- Vendor describes it as "production-validated"; **no general-availability date given**

"Detachable" is the headline property; describing vClick as a surface-attach array inverts the point.

### Guide — an on-package light-source tier

Guide introduces a subsystem tier that did not exist in this survey's prior model of Lightmatter's architecture: the **external light source** for co-packaged and near-package optics. Guide 1 integrates hundreds of lasers and photonic components on a single chip built in an HVM CMOS fab with on-PIC wavelength tunability, replacing the discrete ELSFP "laser farms" that CPO deployments otherwise require. Guide DR packages the same idea as a liquid-cooled OCP NIC 3.0 Laser NIC that lives inside the chassis with zero front-panel footprint. Specifications for both are tabulated above.

Named ecosystem collaborators at the Guide launch: **Cadence, Synopsys and GUC**; **PHIX** is a separately stated collaboration.

### NVIDIA NVLink Fusion

Lightmatter announced joining NVIDIA's **NVLink Fusion** ecosystem in its own press release dated ~2026-06-03. Its stated role is to deliver Passage CPO and NPO products that are optically and electrically compatible with NVIDIA's optical and SerDes technologies for semi-custom AI factories, with a claimed ~50% reduction in fiber and connector count.

Attribution discipline:
- The IEEE Spectrum article of 2026-07-09 is **downstream coverage**, not the announcement.
- Lightmatter's press release names **neither Ayar Labs nor Marvell**. The "Lightmatter alongside Ayar Labs and Marvell" grouping is IEEE Spectrum's framing. Ayar Labs is independently confirmed as a separate NVLink Fusion joiner (announced 2026-06-02); **Marvell's inclusion is not independently confirmed here**.
- The Roy Kim remark that packaging is no longer the bottleneck, and the figures on scale-up domains growing from 72 toward 576 GPUs by 2027, are attributable to **IEEE Spectrum only** and were not independently verified.
- The ~2028 high-volume optical scale-up expectation **is** independently corroborated by Harris's own "2028 high-volume ramp" remark to The Register.

### Manufacturing and ecosystem partners

| Relationship | Status |
|---|---|
| GlobalFoundries, ASE, Amkor | Confirmed (long-standing, pre-baseline) |
| GUC, Cadence, Synopsys | Confirmed — Guide/VLSP design ecosystem |
| SENKO | Confirmed — vClick dFAU |
| Corning | Confirmed — CPO FlexConnect fiber coupling on L20 |
| PHIX | Confirmed — stated collaboration |
| TSMC | **Not confirmed** as a named Lightmatter manufacturing partner from any primary source in this window. This survey's generic reference to TSMC silicon photonics as an industry enabler must not be upgraded to a stated partnership. |
| Tower Semiconductor | **Not a partner — removed.** The only Tower announcement in this period is Tower + LightIC Technologies (an unrelated FMCW-LiDAR silicon-photonics company, 2026-01-05), which appears to be the origin of the conflation. |

### Not yet known

- **Hot Chips 38 (Aug 23–25, 2026, Stanford Memorial Auditorium) has not yet occurred.** Lightmatter does not appear in the advance program. Re-check after the conference; "no Hot Chips 2026 talk" is a premature negative until then.
- **MLPerf: not applicable** (no marketed compute product), rather than "no results."
- No new Envise silicon and no Nature follow-up paper found in this window.
- Guide 1 per-unit fiber count, aggregate bandwidth and wavelength count for Guide DR: **not disclosed**.

---

## Company Status

| Parameter | Value |
|-----------|-------|
| Founded | 2017 |
| Origin | MIT spinout |
| CEO | Nicholas Harris (co-founder) |
| Headquarters | Boston, MA |
| Total raised | ~$850M |
| Last round | Series D — $400M (Oct 2024) |
| Valuation | $4.4B (Oct 2024) |
| Key investors | T. Rowe Price, Fidelity, GV (Google Ventures) |
| Status | Private (CEO: "probably last private round") — no new round in the Apr–Aug 2026 window; $850M / $4.4B still current as of July 2026 (CRN) |
| Standards | XPO MSA founding member (March 2026); IEEE 802.3dj-compliant electrical signaling (L20); CMIS 5.3 / OCP NIC 3.0 / OCP MHS (Guide DR) |
| Ecosystem | NVIDIA **NVLink Fusion** partner (announced ~2026-06-03) |
| Recognition | AI Breakthrough "AI Semiconductor Innovation" award, 2026-06-25 (marketing award; no technical content) |
