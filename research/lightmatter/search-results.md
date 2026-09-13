# Lightmatter — Search Results

*as_of: 2026-08-08 (original scan 2026-04-05; 2026-08-08 results appended at end)*
*chip: lightmatter*
*device_class: Photonic Interconnect + Optics (compute product no longer marketed) — historically Photonic Compute + Interconnect*

---

## Query Results

### Query 1: "Lightmatter Envise photonic AI chip architecture"

**Key findings:**
- Envise is the world's first general-purpose photonic AI accelerator
- Architecture: photonic cores for matrix operations + electronic control and memory
- 4 photonic chips per Envise package manipulating 512 light beams through 200,000+ optical components
- Large arrays of Mach-Zehnder interferometers (MZIs) perform matrix multiplication
- Envise 4S: 16 Envise chips in a 4-U server at only 3 kW power

**Sources:**
- https://lightmatter.co/products/envise/
- https://www.azooptics.com/Article.aspx?ArticleID=2371
- https://research.contrary.com/company/lightmatter
- https://lightmatter.co/blog/a-new-kind-of-computer/
- https://tspasemiconductor.substack.com/p/lightmatter-transforming-ai-infrastructure

---

### Query 2: "Lightmatter Passage photonic interconnect"

**Key findings:**
- Passage is a 3D photonic interposer / interconnect platform
- **Passage M1000** (announced March 31, 2025): 4,000+ mm² multi-reticle active photonic interposer; 114 Tbps total optical bandwidth; 256 optical fibers; 1.5 kW+ power delivery; world's first built-in solid-state optical circuit switching; connects thousands of GPUs in a single domain
- **Passage L200** (announced March 31, 2025): first 3D co-packaged optics (CPO) product; 32 Tbps and 64 Tbps versions; >200 Tbps total I/O bandwidth per chip package; 5–10× improvement over existing CPO solutions; up to 8× faster AI training time
- Key innovation: overcomes edge-limited electrical I/O by enabling electro-optical I/O virtually anywhere on the chip surface

**Sources:**
- https://lightmatter.co/press-release/lightmatter-unveils-passage-m1000-photonic-superchip-worlds-fastest-ai-interconnect/
- https://lightmatter.co/products/m1000/
- https://lightmatter.co/products/passage/
- https://www.datacenterdynamics.com/en/news/lightmatter-unveils-m1000-and-l200-passage-photonic-interconnects/
- https://lightmatter.co/blog/seeing-is-believing-a-technical-deep-dive-into-lightmatters-hardware/
- https://tspasemiconductor.substack.com/p/lightmatter-passage-a-comprehensive

---

### Query 3: "Lightmatter Nature paper 2025 photonic accelerator"

**Key findings:**
- Paper: "Universal photonic artificial intelligence acceleration" published in *Nature*, April 2025 (doi: 10.1038/s41586-025-08854-x)
- Companion paper: "An integrated large-scale photonic accelerator with ultralow latency" also in *Nature* 2025 (doi: 10.1038/s41586-025-08786-6)
- Architecture: 6-chip vertically-integrated package — 4× 128×128 Photonic Tensor Cores (PTCs) + 2× 12nm digital control interface (DCI) chips
- Performance: 65.5 TOPS (ABFP16) at ~78 W electrical + 1.6 W optical power
- Demonstrated: ResNet (image classification), BERT (NLP), Atari deep RL — achieving near-FP32 accuracy
- Physics Review coverage: "Photonic Computing Takes a Step Toward Fruition" (April 2025)

**Sources:**
- https://www.nature.com/articles/s41586-025-08854-x
- https://www.nature.com/articles/s41586-025-08786-6
- https://link.aps.org/doi/10.1103/Physics.18.84
- https://lightmatter.co/blog/a-new-kind-of-computer/

---

### Query 4: "Lightmatter SDK programming model software stack"

**Key findings:**
- **Idiom** software platform: compiler + debugger + profiler for Envise
  - **idCompile**: graph compiler; partitions large networks across multiple Envise blades
  - **idBug**: debugger
  - **idProfiler**: profiler
- Integrates with PyTorch and TensorFlow; developers import Lightmatter libraries and run existing models
- Plug-and-play philosophy: no model rewriting required for supported workloads
- Idiom handles network partitioning and blade parallelism transparently

**Sources:**
- https://lightmatter.co/products/idiom/
- https://research.contrary.com/company/lightmatter
- https://www.nextplatform.com/2021/03/17/lightmatter-normalizing-silicon-photonics-for-ai/

---

### Query 5: "Lightmatter specifications Mach-Zehnder interferometer photonic tensor core"

**Key findings:**
- MZI operation: splits incoming light into two beams on different waveguide paths; phase controlled by electro-optic voltage; recombines beams; output amplitude encodes dot product
- MZI array computes unitary matrix-vector product; cascaded layers implement general matrices via singular value decomposition (SVD)
- Nature paper: 128×128 PTC = 128×128 MZI mesh per photonic core
- Lightmatter "Mars" (earlier chip): demonstrated first silicon-photonic neural net on a single chip
- Matrix multiply happens at the speed of light; only ADC/DAC and activation function require electronics
- IEEE Spectrum coverage: "Lightmatter's Mars Chip Performs Neural-Network Calculations at the Speed of Light"

**Sources:**
- https://spectrum.ieee.org/lightmatter-mars-photonic-chip-neural-network-calculations-speed-of-light
- https://www.nature.com/articles/s41377-022-00717-8

---

### Query 6: "Lightmatter funding valuation 2025 2026"

**Key findings:**
- **Founded:** 2017 (MIT spinout); founders: Nicholas Harris (CEO), Darius Bunandar, Prineha Narang, Thomas Graham
- **Total raised:** ~$850M
- **Series D** (October 2024): $400M; valuation $4.4B (4× prior $1.1B valuation); led by T. Rowe Price; existing: Fidelity, GV (Google Ventures)
- CEO Nick Harris: "This is probably our last private funding round"
- Customers/partners: unnamed hyperscalers and AI companies (Series D press release notes cloud provider participation)
- XPO MSA founding member (March 2026): standardizing high-density optical interconnects for AI data centers

**Sources:**
- https://www.businesswire.com/news/home/20241016498931/en/Lightmatter-Raises-400M-Series-D-Quadruples-Valuation-to-4.4B-as-Photonics-Leader-for-Next-Gen-AI-Data-Centers
- https://lightmatter.co/press-release/lightmatter-raises-400m-series-d-quadruples-valuation-to-4-4b-as-photonics-leader-for-next-gen-ai-data-centers/
- https://techcrunch.com/2024/10/16/lightmatters-400m-d-round-has-ai-hyperscalers-hyped-for-photonic-datacenters/

---

## Resource Inventory

| Category | Resource | URL | Priority |
|----------|----------|-----|----------|
| Product page | Envise photonic AI platform | https://lightmatter.co/products/envise/ | HIGH |
| Product page | Passage M1000 photonic superchip | https://lightmatter.co/products/m1000/ | HIGH |
| Product page | Passage L200 co-packaged optics | https://lightmatter.co/press-release/lightmatter-announces-passage-l200-the-fastest-co-packaged-optics-for-ai/ | HIGH |
| Product page | Idiom software platform | https://lightmatter.co/products/idiom/ | HIGH |
| Nature paper | Universal photonic AI acceleration | https://www.nature.com/articles/s41586-025-08854-x | HIGH |
| Nature paper | Integrated large-scale photonic accelerator | https://www.nature.com/articles/s41586-025-08786-6 | HIGH |
| Blog | A New Kind of Computer | https://lightmatter.co/blog/a-new-kind-of-computer/ | MEDIUM |
| Blog | Technical deep dive into hardware | https://lightmatter.co/blog/seeing-is-believing-a-technical-deep-dive-into-lightmatters-hardware/ | HIGH |
| Analysis | Contrary Research business breakdown | https://research.contrary.com/company/lightmatter | MEDIUM |
| Analysis | TSPASemiconductor: Passage 3D photonic interposer | https://tspasemiconductor.substack.com/p/lightmatter-passage-a-comprehensive | MEDIUM |
| Analysis | TSPASemiconductor: Lightmatter infrastructure | https://tspasemiconductor.substack.com/p/lightmatter-transforming-ai-infrastructure | MEDIUM |
| News | MIT News: startup accelerates light-speed computing | https://news.mit.edu/2024/startup-lightmatter-accelerates-progress-toward-light-speed-computing-0301 | MEDIUM |
| News | IEEE Spectrum: Mars chip neural net | https://spectrum.ieee.org/lightmatter-mars-photonic-chip-neural-network-calculations-speed-of-light | LOW |
| Press release | M1000 announcement (BusinessWire) | https://www.businesswire.com/news/home/20250331220170/en/Lightmatter-Unveils-Passage-M1000-Photonic-Superchip-Worlds-Fastest-AI-Interconnect | MEDIUM |

---

# Appended Search Results — 2026-08-08

> **Retrieval caveat.** The 2026-08-08 session's WebSearch quota was exhausted at the first call; independent discovery was done by fetching DuckDuckGo HTML/Lite result pages and then retrieving primary pages directly. Converge Digest, Data Center Dynamics, TechPowerUp, BusinessWire and hotchips.org returned HTTP 403 or timed out — entries below marked *(snippet only)* rest on search-result snippets rather than full-page retrieval.

### Query 7: "Lightmatter Passage L20 near-package optics"

**Key findings:**
- **Passage L20** announced 2026-03-11; demonstrated at OFC, Los Angeles, 2026-03-15/19. A unified optical engine for **near-package optics (NPO)** and **on-board optics (OBO)** — a new tier between pluggables and full CPO
- 6.4 Tbps per direction / 12.8 Tbps aggregate; 32 optical ports at 200 Gbps per lane, **bidirectional**; 212.5 Gbps PAM4 SerDes; 30 W max TDP; IEEE 802.3dj-compliant electrical signaling; A2 ASHRAE cold plate; Corning CPO FlexConnect fiber coupling
- Internal vendor inconsistencies: energy efficiency 3.0 pJ/bit (press release) vs 5 pJ/bit (product page); package 2000-pin BGA (press release) vs 1827-ball BGA 37.5 × 26.4 mm (product page)
- The "6.4 vs 12.8 Tbps" pair is **not** a discrepancy — per-direction vs aggregate of the same figure
- The "16 L20s replace 512 pluggables in a 102.4 Tbps switch" line is **The Register's** framing, not the vendor's. Vendor's own claims: "4× pluggable density", "88% smaller by volume compared to OSFP"
- Availability: **sampling begins late 2026** — announced only
- Full Harris quote (The Register): "I don't think that it's going to be a super-long roadmap for near package optics, because, of course, CPO is coming, and we think that's like a 2028 high-volume ramp."

**Sources:**
- https://lightmatter.co/press-release/lightmatter-expands-photonic-interconnect-roadmap-with-passage-l20-unified-optical-engine-for-npo-and-obo-applications/
- https://lightmatter.co/products/passage-l20/
- https://www.photonicsonline.com/doc/lightmatter-expands-photonic-interconnect-roadmap-with-passage-l-unified-optical-engine-for-npo-and-obo-applications-0001
- https://www.theregister.com/2026/03/11/lightmatter_passge_l20_fiber/

---

### Query 8: "Lightmatter Guide VLSP light engine laser"

**Key findings:**
- **Guide "Very Large Scale Photonics (VLSP) Light Engine"** unveiled 2026-01-26/27 — a new **external light-source tier** for CPO/NPO/OBO
- Integrates hundreds of lasers and photonic components on a single chip, built in an HVM CMOS fab with on-PIC wavelength tunability
- 100 mW+ optical power per fiber; 16 wavelengths at 100/200/400+ GHz spacing with a roadmap to 64 ("4 to 64 wavelengths with zero increase in assembly complexity")
- CMIS-compliant telemetry, active stabilization and hyper-local thermal tuning maintaining sub-GHz wavelength precision without drift
- N+M redundancy by autonomously retuning a backup laser; replaces discrete ELSFP "laser farms"
- Design-ecosystem collaborators named at launch: Cadence, Synopsys, GUC; PHIX separately *(snippet only for the Photonics.com carry)*
- Two variants: **Guide 1** and **Guide DR**

**Sources:**
- https://lightmatter.co/products/guide/
- https://lightmatter.co/news/
- https://www.photonics.com/Articles/a71892 *(snippet only)*

---

### Query 9: "Lightmatter Guide DR liquid-cooled Laser NIC"

**Key findings:**
- **Guide DR** announced 2026-05-21 — billed by the vendor as the industry's first **liquid-cooled Laser NIC (LNIC)** for CPO scale-up
- OCP NIC 3.0 form factor, in-chassis with **zero front-panel footprint**; up to 64 fibers; 200 mW optical power per fiber; supports 256 lanes at 200G; **51.2 Tbps aggregate per module**; up to four modules in 1RU for **204.8 Tbps** of CPO scale-up switching bandwidth; ~4× rack density vs conventional ELSFPs
- CMIS 5.3 management over I2C/I3C; OCP MHS integration; ASHRAE A2 thermal compliance
- Availability: **sampling begins Q4 2026** *(snippet only — Morningstar/BusinessWire full page returned 403)*
- All figures are vendor marketing, independently unverified

**Sources:**
- https://www.electronicdesign.com/directory/semiconductors/power-semiconductors/press-release/55379808/
- https://lightmatter.co/products/guide/
- https://www.morningstar.com/news/business-wire/20260521715044/ *(snippet only; 403 on full page)*

---

### Query 10: "Lightmatter vClick detachable fiber array unit"

**Key findings:**
- **vClick dFAU** announced 2026-03-11 alongside L20 (trademark spelling **vClick**)
- "The first **detachable** Fiber Array Unit (FAU) with vertical coupling designed to survive" advanced-packaging mold-and-grind flows
- Vertically expanded-beam interface enabling **passive alignment**; **< 1.5 dB** insertion loss across insertion *and* reinsertion
- SENKO SEAT and Metallic PIC Coupler integration
- "Enables known-good optical engine verification at the wafer level" before final assembly — a high-volume CPO manufacturing enabler
- Described as "production-validated"; **no GA date given**
- Note: characterizing it as a "surface-attach" array is wrong — detachability is the headline property

**Sources:**
- https://lightmatter.co/products/vclick-optics/

---

### Query 11: "Lightmatter product lineup 2026 / Envise status"

**Key findings:**
- As of 2026-08-08 the homepage and `/products/` page list only: **Passage L200, Passage L20, Guide 1, Guide DR**, plus **Passage M1000 EVK, Passage EVK100, Passage EVK50, Guide 1 EVK**
- Products page states "**Evaluation kits are sampling today**" — nothing else is sampling or shipping
- **Envise appears nowhere** in product navigation or descriptive content; it survives only in the footer trademark line ("…Idiom, Guide, Passage, Envise, Edgeless I/O, eClick, vClick, A Giant Leap, and VLSP are all trademarks of Lightmatter, Inc.")
- No discontinuation notice, no EOL bulletin, no press release. This is **evidence of absence on a marketing page**, not an announced discontinuation (confidence: medium)
- L200 homepage text: "Supports 32 to 64 Tbps of aggregate bandwidth through co-packaged optics, using 112G PAM4 signaling"
- Passage M1000 (114 Tbps, 4,000+ mm², 256 fibers) is now presented as an **EVK / reference platform**, not a product tier

**Sources:**
- https://lightmatter.co/
- https://lightmatter.co/products/

---

### Query 12: "Lightmatter NVIDIA NVLink Fusion"

**Key findings:**
- Lightmatter announced joining NVIDIA's **NVLink Fusion** ecosystem in its own press release dated **~2026-06-03** (Ayar Labs announced separately on 2026-06-02)
- Stated role: deliver Passage **CPO and NPO** products optically and electrically compatible with NVIDIA's optical and SerDes technologies for semi-custom AI factories; claimed **~50% reduction in fiber and connector count**
- Harris: "This is what the next era of AI infrastructure looks like… combining the industry's most advanced AI platform and the world's leading interconnect."
- The release **names neither Ayar Labs nor Marvell**. The "Lightmatter alongside Ayar Labs and Marvell" grouping comes from IEEE Spectrum (2026-07-09), which is **downstream coverage over a month later**, not the announcement. **Marvell's inclusion is not independently confirmed.**
- IEEE Spectrum also carries Lightmatter VP Roy Kim's argument that packaging is no longer the bottleneck, and an industry expectation of high-volume optical scale-up around 2028 with scale-up domains growing from 72 toward 576 GPUs by 2027 — **attributable to Spectrum only**, not independently verified. The 2028 date *is* corroborated by Harris's own remark to The Register.

**Sources:**
- https://lightmatter.co/press-release/lightmatter-joins-nvidia-nvlink-fusion/
- https://spectrum.ieee.org/nvlink-fusion-optics

---

### Query 13: "Lightmatter manufacturing partners foundry packaging 2026"

**Key findings:**
- **Confirmed**: GlobalFoundries, ASE, Amkor (long-standing, pre-baseline) *(snippet only)*; GUC, Cadence, Synopsys (Guide design ecosystem) *(snippet only)*; SENKO (vClick); Corning (L20 fiber coupling); PHIX *(snippet only)*
- **TSMC is NOT confirmed** as a named Lightmatter manufacturing partner from any primary source in this window — only a third-party LinkedIn post surfaced. The survey's generic reference to TSMC silicon photonics must not be upgraded to a stated partnership
- **Tower Semiconductor is NOT a partner.** The only Tower announcement in the period is *Tower Semiconductor Partners with LightIC to Expand Silicon Photonics Beyond AI Infrastructure* (2026-01-05) — LightIC being an unrelated FMCW-LiDAR company. This appears to be the origin of an erroneous Lightmatter–Tower attribution and should be dropped

**Sources:**
- https://towersemi.com/2026/01/05/01052026/ (the LightIC announcement — evidence *against* a Lightmatter relationship)
- https://lightmatter.co/products/guide/
- https://lightmatter.co/products/vclick-optics/
- https://lightmatter.co/products/passage-l20/

---

### Query 14: "Lightmatter funding 2026 / Hot Chips 2026 / MLPerf"

**Key findings:**
- **Funding unchanged**: $850M total raised at a $4.4B valuation, from the October 2024 $400M Series D (T. Rowe Price-led). No new round in the window; still reported at these figures by CRN in July 2026 *(snippet only for Sacra / Gunderson Dettmer corroboration)*
- **Hot Chips 38 runs Aug 23–25, 2026** at Memorial Auditorium, Stanford — *after* this scan date. Lightmatter does **not** appear in the advance program (48 entries; AMD, Intel, NVIDIA Rubin/BlueField-4 and Google TPU v8 referenced). "No Hot Chips 2026 talk" is a premature negative; re-check after the conference. hotchips.org returned 403 — dates and program scope obtained via search
- **MLPerf**: no submissions, and **not applicable** — with Envise no longer marketed, Lightmatter has no compute product that could be submitted
- No new Envise silicon and no Nature follow-up paper found in the window
- Lightmatter won an AI Breakthrough "AI Semiconductor Innovation" award on 2026-06-25 (marketing award, no technical content)

**Sources:**
- https://www.crn.com/news/computing/2026/the-10-coolest-semiconductor-startups-of-2026-so-far
- https://lightmatter.co/news/
- https://hotchips.org/ *(403; dates/scope via search)*

---

## Resource Inventory — added 2026-08-08

| Category | Resource | URL | Priority |
|----------|----------|-----|----------|
| Product page | Lightmatter products index (current lineup; "Evaluation kits are sampling today") | https://lightmatter.co/products/ | HIGH |
| Product page | Passage L20 (NPO / OBO unified optical engine) | https://lightmatter.co/products/passage-l20/ | HIGH |
| Product page | Guide (VLSP Light Engine; Guide 1 and Guide DR) | https://lightmatter.co/products/guide/ | HIGH |
| Product page | vClick Optics (vClick dFAU detachable fiber array unit) | https://lightmatter.co/products/vclick-optics/ | HIGH |
| Press release | Passage L20 — photonic interconnect roadmap expansion (2026-03-11) | https://lightmatter.co/press-release/lightmatter-expands-photonic-interconnect-roadmap-with-passage-l20-unified-optical-engine-for-npo-and-obo-applications/ | HIGH |
| Press release | Lightmatter Joins NVIDIA NVLink Fusion (~2026-06-03) | https://lightmatter.co/press-release/lightmatter-joins-nvidia-nvlink-fusion/ | HIGH |
| Press carry | Electronic Design — Guide DR liquid-cooled Laser NIC specs (2026-05-21) | https://www.electronicdesign.com/directory/semiconductors/power-semiconductors/press-release/55379808/ | HIGH |
| Press carry | Photonics Online — independent carry of the L20 release | https://www.photonicsonline.com/doc/lightmatter-expands-photonic-interconnect-roadmap-with-passage-l-unified-optical-engine-for-npo-and-obo-applications-0001 | MEDIUM |
| News | The Register — Passage L20 bidirectional fiber; full Harris "2028 high-volume ramp" quote (2026-03-11) | https://www.theregister.com/2026/03/11/lightmatter_passge_l20_fiber/ | HIGH |
| News | IEEE Spectrum — NVLink Fusion optics (2026-07-09; downstream coverage, contains unverified framing) | https://spectrum.ieee.org/nvlink-fusion-optics | MEDIUM |
| News | Lightmatter press-coverage index (authoritative 2026 timeline) | https://lightmatter.co/news/ | HIGH |
| Analysis | CRN — 10 Coolest Semiconductor Startups of 2026 (funding + product listing) | https://www.crn.com/news/computing/2026/the-10-coolest-semiconductor-startups-of-2026-so-far | MEDIUM |
| Press carry | Morningstar/BusinessWire — Guide DR "sampling in Q4 2026" (snippet only; 403 on full page) | https://www.morningstar.com/news/business-wire/20260521715044/ | LOW |
| Disconfirming | Tower Semiconductor + LightIC (2026-01-05) — the unrelated announcement behind the erroneous Lightmatter–Tower attribution | https://towersemi.com/2026/01/05/01052026/ | LOW |
| Conference | Hot Chips 38 (Aug 23–25, 2026, Stanford) — Lightmatter absent from advance program; re-check after the event | https://hotchips.org/ | LOW |
