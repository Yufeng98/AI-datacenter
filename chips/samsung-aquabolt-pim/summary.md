# Samsung PIM (Aquabolt-XL HBM-PIM / LPDDR5X-PIM) — Summary

*chip: samsung-aquabolt-pim*
*device_class: Processing-in-Memory (PIM)*
*generations: Aquabolt-XL HBM2-PIM (2021) / LPDDR5X-PIM (disclosed 2026)*
*as_of: 2026-09-13*
*(Aquabolt-XL content researched 2026-04-05; LPDDR5X-PIM section added 2026-08-08; Hot Chips 38 disclosure added 2026-09-13)*

---

## One-Paragraph Summary

Samsung **Aquabolt-XL** is the world's first HBM with integrated AI processing capability, announced at ISSCC February 2021 and detailed at Hot Chips 33 (HC33). It is an **HBM2 stack** where the bottom 4 standard DRAM dies are replaced with **PIM-DRAM dies** — each containing 32 **FP16 SIMD processors** (one per pseudo-channel) placed at bank boundaries. A full 8-die stack contains **128 processors** with combined peak compute of **~1.2 TFLOPS FP16** and access to **4.92 TB/s** of internal DRAM bandwidth (vs. 1.23 TB/s external I/O). The design is a **JEDEC-compatible HBM2 drop-in replacement** requiring no memory controller changes. Samsung validated it on an AMD MI60 GPU (>2× performance, >70% energy reduction) and Xilinx Alveo U280 FPGA (2.49× faster, 62% less energy) for RNN and ML workloads. The software stack includes a PIM BLAS library and transparent TensorFlow/PyTorch integration, though no public SDK has been released. Samsung positioned Aquabolt-XL as research/prototype and no commercial production of the HBM2-PIM part was confirmed post-2021.

**Samsung's PIM program did not stop there — it moved off HBM.** The current disclosed generation is **LPDDR5X-PIM**, a PIM block placed 1-to-1 with each DRAM bank of an LPDDR5X-9600 device, with **mixed-precision INT/FP GEMV** rather than Aquabolt-XL's FP16-only datapath. It was substantively disclosed in a Samsung-authored arXiv note (30 May 2026), showcased at FMS 2026 (4–6 Aug 2026), and given a full architecture and performance disclosure at Hot Chips 2026 on 25 Aug 2026 (16 PIM blocks, 614 GB/s internal bandwidth, 2.4 TOPS SINT4, 16 GB/rank, plus a Llama-3.1-8B benchmark). **No HBM3/HBM4-based Samsung PIM or Aquabolt-XL successor is confirmed either way**, though a Hot Chips 38 tutorial describes a related "processing elements in the base die" direction for future HBM4/HBM5. See the LPDDR5X-PIM update sections below (2026-08-08 and 2026-09-13).

---

## Key Specifications — Aquabolt-XL HBM-PIM (2021)

| Property | Value |
|----------|-------|
| Paradigm | Processing-in-Memory (PIM) in HBM2 |
| Product | Aquabolt-XL / HBM-PIM |
| Memory base | HBM2 (Samsung Aquabolt) |
| Die stack | 4× PIM-DRAM + 4× standard DRAM + logic die |
| Processors / PIM-DRAM die | 32 |
| Total processors / stack | 128 |
| Processor type | FP16 SIMD (2 multipliers + 2 adders) |
| Peak FP16 TFLOPS | ~1.2 |
| Internal bandwidth | 4.92 TB/s |
| External HBM2 I/O | 1.23 TB/s |
| Total capacity | 16 GB (standard HBM2) |
| JEDEC HBM2 compatible | Yes (drop-in) |
| Numeric formats | FP16 |
| Validated on | AMD MI60 GPU, Xilinx Alveo U280 |
| Announced | ISSCC 2021 / HC33 Aug 2021 |

*This table describes the HBM2-era part only. For LPDDR5X-PIM see the update section below.*

---

## Positioning — Aquabolt-XL

- **Target**: GPU/FPGA memory-bound ML and HPC workloads (GEMV, RNN, attention)
- **Drop-in**: No hardware changes needed in GPU/FPGA memory controller
- **Not standalone**: HBM2 stack attached to GPU/FPGA die as usual
- **Research/prototype**: No confirmed commercial production of the HBM2-PIM part post-2021. Samsung's PIM program continued, but on an LPDDR base (below), not on HBM.

---

## Software Ecosystem — Aquabolt-XL

| Layer | Tool | Status |
|-------|------|--------|
| Framework | TensorFlow, PyTorch | not public (transparent offload described) |
| BLAS library | PIM BLAS (BLAS L2) | not public |
| Runtime | PIM Runtime / Driver | not public |
| HW interface | PIM mode enable commands | not public |
| Research tool | PIMSys (gem5 + DRAMSys) | Public (ACM MemSys 2024) |
| Samsung simulator (earlier gen) | PIMSimulator ([SAITPublic/PIMSimulator](https://github.com/SAITPublic/PIMSimulator), 2023) | Public — note: this is **not** LP5X-PIM Sim (see below) |

---

## LPDDR5X-PIM Update (2026-08-08)

*Updated 2026-08-08. Primary source: **arXiv:2606.00636v1**, "LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM," SangHoon Cha, Jaewan Choi, Byeongho Kim, Yoonah Paik, Sukhan Lee, Kyomin Sohn (all Samsung Electronics), cs.AR, submitted 30 May 2026 — https://arxiv.org/abs/2606.00636. Secondary: Samsung Semiconductor newsroom FMS 2026 post (4 Aug 2026); StorageReview and SamMobile FMS 2026 coverage (4 Aug 2026); hotchips.org program; ETNews (23 Jul 2026); Seoul Economic Daily (5 Aug 2026).*

### Dating and framing

LPDDR5X-PIM is **not a new-in-2026 program**: the arXiv note cites its own architecture reference as B. Kim et al., *IEEE Micro* 44(3):40–48, 2024, and trade press reported Samsung "advancing LPDDR5X-PIM" on 18 Feb 2026 (WinBuzzer, aggregator — low confidence). What is new is the **substantive disclosure**: the arXiv note (30 May 2026), the FMS 2026 showcase (4–6 Aug 2026), and a Hot Chips 2026 talk (25 Aug 2026). This entry closes a gap in the survey rather than recording a resurrection.

### What it is

| Property | Value |
|----------|-------|
| Paradigm | Processing-in-Memory in **LPDDR5X** (not HBM) |
| Product | Samsung LPDDR5X-PIM / "LP5X-PIM" |
| Memory base | LPDDR5X-9600, strictly JEDEC-compliant (JESD209-5C, 2022); **561-ball JEDEC package** (Hot Chips 38) |
| PIM organization | **One PIM block per DRAM bank (1-to-1 mapping)** (arXiv); Hot Chips 38: **16 PIM blocks** in the DRAM banks, MAC trees running in parallel |
| Operating modes | **Single-Bank (SB)** — standard DRAM operation; **Multi-Bank (MB)** — parallel PIM execution across banks |
| Named registers | IRF (instruction register file), SRF (Source Register File); Hot Chips 38 adds a **Vector Register File** (1-kbit, max 64 sequential reads) |
| Peak TOPS / TFLOPS | **2.4 TOPS SINT4** (SINT8 act./SINT4 wt.); **~1.2 TFLOPs FP8 per package** (Hot Chips 38 — resolves prior "not disclosed") |
| Internal vs. external bandwidth | **614 GB/s internal** (PIM-side, x64 @ 9600 Mbps) vs. **76.8 GB/s external** (host-side) — an "eight times" ratio, internally consistent (Hot Chips 38) |
| Numeric formats | Integer **W8A8, W4A4, W8A16, W4A8, W4A16** + more; floating-point **W8A8(FP), W8A16(FP)** + more — Hot Chips 38 states **"fifteen combinations"** total |
| Reference channel count (arXiv experiments) | 4 DRAM channels |
| Process node | **still not disclosed** |
| Capacity | **16 GB across four dies per rank** (Hot Chips 38) |
| Status | Disclosed at Hot Chips 38 (2026-08-25); simulator and datasheet **available upon request**; SDK with reference tooling mentioned. **Still no production date, no availability commitment, no named customers, no confirmed measured-silicon results.** |
| Disclosure venue | arXiv 30 May 2026; FMS 2026 (4–6 Aug 2026); Hot Chips 2026 Memory session, Tue 25 Aug 2026, presenter Karam Hwang (Samsung) — **delivered as scheduled** |

Samsung's arXiv paper had stated: *"Further technical details regarding the specific architecture and circuit design of the LPDDR5X-PIM will be disclosed in future publications."* The Hot Chips 38 talk (2026-08-25) is that disclosure — see the 2026-09-13 update section below for full detail. Process node remains the one major architectural item still undisclosed.

### Performance — vendor simulation output, not silicon

All figures below are Samsung's own cycle-accurate simulator output against a **self-defined non-PIM sequential-weight-read baseline with four DRAM channels**. There is no measured-silicon result in the public record.

| Configuration (weight-tile dim 4096) | GEMV speedup vs. non-PIM baseline |
|---|---|
| Larger tile shapes: W8A8, W4A4, W8A8(FP) | **6.0× – 6.2×** |
| Smaller tile shapes: W8A16, W4A16, W8A16(FP) | **5.7× – 5.8×** |
| With a static 150 ns memory-fence latency modeled | **> 5.0×** for most configs; low of **4.1×** for W4A16 |
| "Reshape" column-partitioning software optimization, small matrices (W < 2048) | additional **up to 1.65×** |

ETNews (23 Jul 2026) repeats the "up to 6.2× faster than conventional memory" figure — it is the same simulation number, not independent corroboration.

### Software stack (new — absent from the Aquabolt-XL stack)

A **"PIM Kernel"** layer sits between the application and the memory device:

| Component | Role |
|---|---|
| **Data Mapper** (offline) | PIM-aware placement of weights and scale factors into DRAM banks using a predefined **"PIM Tile Configuration"**; preloaded so no runtime rearrangement is needed |
| **PIM Executor** (runtime) | Three sub-components: **PIM Device Code Gen** (synthesizes IRF code + hardware config code from matrix shapes and dtypes), **PIM Control** (manages SB ↔ MB mode transitions), **GEMV Kernel** (executes per-tile GEMV on a specialized PIM ISA; handles pipeline flush-out) |

Address-mapping strategies: **Vertical Mapping** (rows interleaved across Channel / Rank / Bank Group / Bank), **Horizontal Mapping** (adjacent sub-matrices in one bank for row-buffer hits), **Reshape Optimization** (column-based partitioning for near-100% intra-PIM efficiency).

**Simulator**: *LP5X-PIM Sim* is cycle-accurate and built on **DRAMSim3** and **Ramulator**. The paper makes **no open-sourcing statement and gives no repository link** — treat availability as not confirmed. Do not conflate it with Samsung's earlier, genuinely public **PIMSimulator** (github.com/SAITPublic/PIMSimulator, 2023), which the paper cites separately.

### Target market — mixed; do not state exclusively

ETNews positions LPDDR5X-PIM for smartphones and laptops, and the paper's only application hint is that the 150 ns fence latency is "a representative value for high-performance mobile application processors." But Samsung showcased it at FMS 2026 under an explicit "future of AI infrastructure" framing, and SamMobile describes it as aimed at AI data centers and edge AI devices. The defensible statement: **primarily an on-device/edge AI inference play, marketed by Samsung under an AI-infrastructure banner.** Its relevance to this datacenter survey is as the successor of Samsung's PIM program and as a contrast case to Aquabolt-XL's GPU/FPGA-attached HBM2 targeting — **not** as a confirmed datacenter part.

### Status verbs — what is and is not supported

- **Supported**: announced, showcased (FMS 2026), simulated (arXiv), scheduled for Hot Chips 2026 disclosure.
- **Not supported**: sampling, shipping, mass production, deployment, named customers. Samsung's own FMS post gives no availability or production date.
- Korean trade-press characterizations — ETNews "moved well beyond the research stage," Seoul Economic Daily "entered a full-fledged sales phase to attract Big Tech customers" — are **low confidence**, unconfirmed by Samsung, and name no customer.
- **Date warning**: Seoul Economic Daily (5 Aug 2026) misdates the Hot Chips talk to "the 23rd of next month" (September 2026). hotchips.org lists Hot Chips 2026 as 23–25 August 2026. Do not cite sedaily for the date.

### HBM-PIM / Aquabolt-XL successor — not confirmed either way

No HBM3/HBM4-based Samsung PIM or Aquabolt-XL successor surfaced in the Hot Chips 2026 program, in FMS 2026 coverage, or on Samsung's PIM technology page. This is **absence of evidence, not a verified negative**. Samsung's Hot Chips 2026 Tutorial 1 talk, "HBM Base Die: How will HBM evolve in the future by utilizing the logic process?" (Sangwook Han, Samsung, 23 Aug 2026), covers logic-process base dies and is **not** described as PIM — and, being a future talk, its content is not yet public. Peripheral Samsung HBM4 specifications circulating alongside this story (shipping date, data rate, base-die node, HBM4E sampling, an ISSCC 2026 HBM4 paper) **could not be independently verified** and are deliberately omitted here.

*Update 2026-09-13: the Tutorial 1 talk referenced above has now been delivered — see the LPDDR5X-PIM Update (2026-09-13) section below for what it disclosed.*

---

## LPDDR5X-PIM Update (2026-09-13)

*Updated 2026-09-13. Window covered: 2026-08-08 → 2026-09-13. Primary source: ServeTheHome coverage of Samsung's Hot Chips 38 talk "Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory (PIM) Solution for AI Inference" (Karam Hwang), delivered Tue 2026-08-25 — the disclosure flagged as pending throughout the 2026-08-08 section above. Also: ServeTheHome coverage of Samsung's Sunday tutorial "HBM Base Die: How HBM Will Evolve Using Advanced Logic Processes" (Sangwook Han), 2026-08-23.*

**Bottom line: the Hot Chips 38 talk delivered on nearly everything the 2026-08-08 section flagged as pending.** The "What it is" table above has been updated in place with the new figures; this section explains what changed and what is still open.

1. **Architecture disclosed**: 16 PIM blocks sit in the DRAM banks with MAC trees running in parallel (vs. the arXiv paper's "1-to-1 per bank" description — the two are not necessarily in conflict, but the exact reconciliation of "16 blocks" against per-bank placement is not spelled out in the reporting). ALU supports both FP and INT. A Vector Register File (1-kbit, max 64 sequential reads) is named for the first time.
2. **Peak throughput disclosed**: 2.4 TOPS SINT4 (SINT8 activations / SINT4 weights); ~1.2 TFLOPs FP8 per package. Previously entirely "not disclosed."
3. **Bandwidth disclosed**: 614 GB/s PIM-side (internal) vs. 76.8 GB/s host-side (external, standard LPDDR5X-9600 x64) — an "eight times" ratio that checks out arithmetically (614/76.8 ≈ 8.0).
4. **Capacity and package disclosed**: 16 GB across four dies per rank; JEDEC-standard 561-ball package.
5. **Precision set widened**: "fifteen combinations" of datatype pairs stated at Hot Chips 38, versus the arXiv paper's published subset of 7.
6. **New named-model benchmark**: Llama-3.1-8B inference (320-token context, SINT8 act./SINT4 weights) shows **2.28× lower latency** (5.4s vs. 12.3s) and **3.01× higher throughput** (81.3 vs. 27.0 tok/s) versus a non-PIM baseline. This supersedes the pure-GEMV-tile evidence from the arXiv paper as the headline performance claim, though it is **not confirmed to be measured silicon** — treat as likely continued simulator output pending an explicit statement otherwise.
7. **Software/tooling clarified**: LP5X-PIM Sim and a datasheet are **available upon request** (resolves the prior "availability not confirmed"); an SDK with reference tooling is now mentioned.
8. **Forward roadmap disclosed**: **LPDDR6-PIM is in development toward a finalized JEDEC LP6-PIM specification** — the first confirmed next step for the LPDDR-PIM line.
9. **HBM base-die tutorial delivered (2026-08-23)**: Samsung describes evolving the HBM base die from a passive interposer into an active co-processor, including **"processing elements in the base die"** to reduce die-to-die bandwidth demand, applying "D1c DRAM process and logic 4nm to HBM4" and roadmapping toward **"zHBM"** (3D vertical integration) across HBM4/HBM5. This does **not** use the terms "PIM" or "Aquabolt," and is **not** confirmed as an Aquabolt-XL/HBM-PIM successor — but it is the same underlying idea (compute co-located with DRAM) aimed at a different memory base. Record as an adjacent roadmap signal, not a confirmed successor.
10. **Target market broadened**: Hot Chips 38 states server, mobile, *and* client AI inference — wider than the arXiv-era "primarily mobile application processor" framing (still consistent with the "mixed, not exclusively on-device" reading from the 2026-08-08 section).
11. **Updated program roadmap**: HBM-PIM/Aquabolt-XL (2021, HBM2) → **LPDDR5X-PIM (2026, disclosed at Hot Chips 38)** → LPDDR6-PIM (JEDEC LP6-PIM, in development) and/or HBM4–HBM5 base-die processing elements (roadmapped, unconfirmed lineage to the PIM line).
12. **Still not disclosed**: process node; production date; availability commitment; named customers; independent (non-Samsung) benchmark confirmation for any figure, including the new Llama-3.1-8B result.

**Sources for this update:** https://www.servethehome.com/samsung-lpddr5x-pim-at-hot-chips-2026/ · https://www.servethehome.com/samsung-evolving-hbm-base-die-at-hot-chips-2026/

---

## References

- [HC33 Official Samsung Aquabolt-XL Slides](https://www.hc33.hotchips.org/assets/program/conference/day1/20210813_HC33_Aquabolt-XL_PIM_Jin_Kim_slide.pdf)
- [IEEE HC33 Paper](https://ieeexplore.ieee.org/document/9567191/)
- [Samsung HBM-PIM Announcement](https://news.samsung.com/global/samsung-develops-industrys-first-high-bandwidth-memory-with-ai-processing-power)
- [The Memory Guy — Aquabolt-XL Part 1](https://thememoryguy.com/samsungs-aquabolt-xl-processor-in-memory-part-1/)
- [ServeTheHome HC33 Coverage](https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/)
- [PIMSys Virtual Prototype — ACM MemSys 2024](https://dl.acm.org/doi/10.1145/3695794.3695797)

**LPDDR5X-PIM (added 2026-08-08)**

- [arXiv:2606.00636 — "LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM" (Samsung, 30 May 2026)](https://arxiv.org/abs/2606.00636) — primary source for all architecture, format, and speedup figures above
- [Hot Chips 2026 program (hotchips.org)](https://www.hotchips.org/) — Memory session, Tue 25 Aug 2026, "Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory (PIM) Solution for AI Inference," Karam Hwang (Samsung) — delivered as scheduled; see below for coverage
- [Samsung Semiconductor newsroom — FMS 2026 3D memory vision (4 Aug 2026)](https://news.samsungsemiconductor.com/global/samsung-unveils-next-gen-3d-memory-vision-at-fms-2026-charting-the-future-of-ai-infrastructure/)
- [StorageReview — Samsung 3D memory roadmap at FMS 2026 (4 Aug 2026)](https://www.storagereview.com/news/samsung-outlines-3d-memory-roadmap-for-ai-infrastructure-at-fms-2026)
- [SamMobile — zNAND-O, LPDDR5X-PIM, PM1763 (4 Aug 2026)](https://www.sammobile.com/news/samsung-znand-o-lpddr5x-pim-pm1763-memory-chips-ssd-ai-data-centers/)
- [ETNews (23 Jul 2026)](https://en.etnews.com/20260723200002) — trade press, low confidence
- [Seoul Economic Daily (5 Aug 2026)](https://en.sedaily.com/finance/2026/08/05/samsung-sk-push-pim-and-cxl-as-us-china-japan-challenge-hbm) — trade press, low confidence; **misdates the Hot Chips talk to September 2026**
- [Samsung PIMSimulator (SAITPublic, 2023)](https://github.com/SAITPublic/PIMSimulator) — earlier Samsung PIM simulator; **not** LP5X-PIM Sim

**Hot Chips 38 disclosure (added 2026-09-13)**

- [ServeTheHome — Samsung LPDDR5X-PIM at Hot Chips 2026 (2026-08-25)](https://www.servethehome.com/samsung-lpddr5x-pim-at-hot-chips-2026/) — primary source for all 2026-09-13 architecture, bandwidth, throughput, and Llama-3.1-8B benchmark figures
- [ServeTheHome — Samsung Evolving HBM Base Die at Hot Chips 2026 (2026-08-23)](https://www.servethehome.com/samsung-evolving-hbm-base-die-at-hot-chips-2026/) — HBM base-die tutorial coverage
