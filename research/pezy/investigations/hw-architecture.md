# PEZY Computing — HW Architecture Investigation

*as_of: 2026-08-08 (original investigation 2026-04-05; see section 12 — Investigation Update 2026-08-08)*

## Summary

PEZY Computing is a small Japanese fabless chip designer that has spent 15 years building MIMD (Multiple Instruction Multiple Data) manycore processors for energy-efficient FP64 supercomputing. The latest generation, PEZY-SC4s (presented at Hot Chips 37, August 2025), is a 2,048-PE MIMD processor on TSMC 5nm with 4× HBM3 (96 GB, 3.2 TB/s) and a simulated FP64 energy efficiency of ~91 GF/W — nearly double NVIDIA H200's ~49 GF/W at FP64. The SC4s adds BF16 support as a bow to the AI era, a quad-core RISC-V management processor enabling host-less Linux operation, and a new 5-level memory hierarchy culminating in 64 MB per-state L3.

PEZY's architectural thesis is the inverse of GPU-era design: run thousands of small in-order MIMD cores at modest clocks (~1.5 GHz) rather than wide SIMD at high frequencies. This minimizes branch-divergence waste and maximizes FP64 energy efficiency for irregular scientific workloads.

As of Q1 2026, SC4s is pre-production (simulated results only). The SC3-equipped ZettaScaler 3.0 remains the shipping product.

---

## 1. Company and Background

| Parameter | Value |
|-----------|-------|
| Founded | 2010, Tokyo, Japan |
| Type | Fabless semiconductor (Japanese domestic) |
| CEO | Motoaki Saito (arrested Dec 2017 for NEDO subsidy fraud, ¥653M; company continued R&D) |
| Partner | ExaScaler Inc. — manufactures ZettaScaler liquid-immersion-cooled supercomputer systems |
| Focus | MIMD manycore HPC processors; FP64 energy efficiency; Japan domestic HPC |
| Foundry | TSMC (all generations: 28nm → 16nm → 7nm → 5nm) |

### Key Corporate Events
- 2017: CEO arrested; Gyoukou supercomputer contract canceled; government subsidy fraud case
- 2018: PEZY GM received suspended sentence; Saito still on trial
- 2019–2022: Company rebuilt, continued SC3 development with ZettaScaler 3.0
- 2025: SC4s presented at Hot Chips 37; test system (90 nodes, 8.6 PF FP64) planned

---

## 2. Product Line Evolution

| Generation | Year | PEs | Process | Clock | Peak FP64 | Notable System | Green500 |
|------------|------|-----|---------|-------|----------|----------------|---------|
| PEZY-SC | 2014 | 1,024 | TSMC 28nm | 733 MHz | ~1.5 TFLOPS | Shoubu (RIKEN) | #1, Jun/Nov 2015 (7.03 GF/W) |
| PEZY-SC2 | 2017 | 2,048 | TSMC 16FF+ | 1.0 GHz | 4.1 TFLOPS | Gyoukou (JAMSTEC); Shoubu System B | #1, Nov 2017 (17 GF/W) |
| PEZY-SC3 | 2020 | 4,096 | TSMC 7nm | 1.2 GHz | ~19.7 TFLOPS | ZettaScaler3.0 | #12, Nov 2021 (24.6 GF/W) |
| PEZY-SC3s | 2020+ | 512 | TSMC 7nm (109 mm²) | — | — | Small module variant | — |
| PEZY-SC4 | ~2024 | 4,096 | — | — | — | Intermediate/unreleased | — |
| PEZY-SC4s | 2025 | 2,048 | TSMC 5nm | 1.5 GHz | ~24.6 TFLOPS (sim.) | ZettaScaler4.0 (planned) | Target ~91 GF/W FP64 |

**Note on SC4s core count**: SC4s halves PE count vs SC3 (2,048 vs 4,096) but substantially increases per-PE complexity: wider FP64 units, BF16 support, larger caches, and HBM3 vs HBM2. The reduction in PE count is traded for higher per-PE throughput, better cache efficiency, and a simpler coherence domain.

---

## 3. PEZY-SC4s Compute Architecture

### 3.1 Design Philosophy: MIMD vs GPU SIMD

PEZY's central differentiation is MIMD execution at the core level. Each PE executes fully independent instruction streams — there is no warp/wavefront SIMD grouping constraint. This contrasts sharply with GPU execution where all threads in a warp (NVIDIA 32-wide) or wavefront (AMD 64-wide) must execute the same instruction each cycle.

| Aspect | GPU (SIMD) | PEZY (MIMD) |
|--------|-----------|------------|
| Branch divergence | Expensive (threads serialized or masked) | No divergence penalty |
| Vector width | 1,024-bit (wave32) to 2,048-bit (wave64) | 256-bit per PE |
| Best workload | Dense regular matrix ops | Irregular scientific (n-body, sparse, graph) |
| FP64 efficiency | Typically low (many GPUs downsample to FP32) | High (FP64 is first-class) |

### 3.2 Processing Element (PE)

The PE is the fundamental compute unit:

| Feature | PEZY-SC4s PE |
|---------|-------------|
| Execution model | In-order scalar + SIMD |
| Hardware threads | 8 (SMT8) — two groups of 4 |
| Thread scheduling | Fine-grained within group (different thread each cycle); coarse-grained between groups (swap on long-latency instruction) |
| FP64 unit | 4-wide SIMD (256-bit) |
| BF16 support | Yes (new in SC4s; wider effective vector for BF16) |
| FP32 | Yes |
| FP16 | Yes |
| INT8 | Yes |
| L1 I-cache | 4 KB per PE |
| L1 D-cache | 4 KB per PE |
| Scratchpad | 24 KB per PE (software-managed) |

**SMT design rationale**: 8 hardware threads per PE hide memory latency. When a thread stalls on a cache miss or memory operation, the hardware scheduler selects a different thread from the active group (fine-grained, every cycle). For longer latency events (e.g., HBM3 access), the entire 4-thread group swaps out (coarse-grained multithreading via instruction flag). This achieves high PE utilization without out-of-order hardware.

### 3.3 Memory Hierarchy: Five-Level Village/City/Prefecture/State

PEZY uses a geographic naming convention for its hierarchy:

```
PE (1 core, 8 threads)
  │ L1: 4 KB I + 4 KB D + 24 KB scratchpad
  ▼
Village (4 PEs share scratchpad caches)
  ▼
City (4 villages = 16 PEs)
  │ Shared: 32 KB L2 I-cache + 64 KB L2 D-cache
  ▼
Prefecture (18 cities = up to 288 PEs; 16 active in SC4s)
  ▼
State (8 prefectures = 2,048 active PEs)
  │ Shared: 64 MB L3 cache
  ▼
HBM3 (4 stacks, 96 GB, 3.2 TB/s)
```

| Level | Sharing | Cache |
|-------|---------|-------|
| PE | Individual | 4 KB I + 4 KB D + 24 KB scratchpad |
| Village (4 PEs) | Shared scratchpad | — |
| City (16 PEs) | Shared | 32 KB L2 I + 64 KB L2 D |
| Prefecture (~288 PEs) | — | — |
| State (2,048 PEs) | Shared | 64 MB L3 |

**Total on-chip SRAM estimate**: 2,048 × (4+4+24) KB = ~64 MB L1/scratchpad + per-city L2 (128 cities × 96 KB = ~12 MB) + 64 MB L3 ≈ ~140 MB total on-chip SRAM.

### 3.4 Off-chip Memory: HBM3

| Parameter | Value |
|-----------|-------|
| Technology | HBM3 |
| Stacks | 4 |
| Total capacity | 96 GB |
| Total bandwidth | 3.2 TB/s |
| Per-stack BW | 800 GB/s |
| Bus width (inferred) | 1,024-bit per stack @ 8 GT/s |

**Comparison with peers (FP64 HPC context)**:

| Chip | Off-chip Memory | Capacity | BW |
|------|----------------|----------|-----|
| PEZY-SC4s | HBM3 | 96 GB | 3.2 TB/s |
| NVIDIA H100 SXM | HBM3 | 80 GB | 3.35 TB/s |
| AMD MI300X | HBM3 | 192 GB | 5.3 TB/s |
| PEZY-SC3 | HBM2 | 32 GB | ~512 GB/s |

HBM3 represents a major bandwidth leap over SC3's HBM2, enabling the larger FP64 working sets of modern scientific codes.

---

## 4. Process and Package

| Parameter | Value |
|-----------|-------|
| Foundry | TSMC 5nm |
| Die size | ~556 mm² (single monolithic die) |
| TDP | ~600 W (estimated) |
| Packaging | 2.5D (4× HBM3 stacks + SC4s die on silicon interposer) |
| Host interface | PCIe Gen 5 × 16 |

The 556 mm² die is large for a 5nm process — reflecting the on-chip SRAM budget (64 MB L3 alone) and 2,048 PE array with HBM3 controllers. PEZY uses a single monolithic die (no chiplet disaggregation).

---

## 5. Management Processor: RISC-V (New in SC4s)

A key new feature distinguishing SC4s from all prior generations is an integrated quad-core RISC-V management processor:

| Parameter | Value |
|-----------|-------|
| Core | Rocket Core (open-source, in-order, scalar) |
| Count | 4 cores |
| Clock | 1.5 GHz |
| Role | Runs Linux OS; manages PE initialization and scheduling |
| Impact | Host-less operation — no external x86 CPU required to boot/manage the chip |

Prior to SC4s, all PEZY systems required an AMD EPYC host processor running Linux to manage the PEZY coprocessors. The SC4s RISC-V management processor can run Linux natively on-chip, making it theoretically possible to build PEZY-only HPC nodes. In practice, the ZettaScaler 4.0 reference system still includes an AMD EPYC 9555P (Zen 5, 64-core Turin) for host compute and system management.

---

## 6. ZettaScaler 4.0 System

The reference system board and supercomputer design:

| Parameter | Value |
|-----------|-------|
| Accelerators per node | 4 × PEZY-SC4s |
| Host CPU | AMD EPYC 9555P (Zen 5, 64 cores, 1.5 GHz) |
| Host connection | PCIe Gen 5 x16 per SC4s chip |
| Scale-up interconnect | 400 Gb/s NDR InfiniBand per node |
| Cooling | Liquid immersion cooling (fluorine-type inert liquid) |
| Target test cluster | 90 nodes × 4 chips = 360 SC4s chips |
| Total PEs (test) | 737,280 |
| Peak FP64 (test) | 8.6 PFLOPS |

### ZettaScaler Liquid Immersion Cooling
PEZY's ZettaScaler systems use fluorine-type inert liquid immersion cooling (not water cooling). The liquid:
- Has high electrical insulation
- Is non-combustible, non-toxic, odorless
- Zero ODP (Ozone Depletion Potential)
- High boiling point — no sealing required
- Enables silent, compact supercomputer deployments (laboratory/office-grade)

ZettaScaler 3.0 (SC3 era): also offered optional air cooling variant.

---

## 7. FP64 Energy Efficiency — Core Design Argument

PEZY's primary market argument is FP64 GF/W (gigaflops per watt), and this has been the metric that got them onto Green500 lists:

| Chip | FP64 GF/W | Year |
|------|-----------|------|
| PEZY-SC (Shoubu) | 7.0 | 2015 |
| PEZY-SC2 (Shoubu System B) | 17.0 | 2017 |
| PEZY-SC3 (ZettaScaler3.0) | 24.6 | 2021 |
| PEZY-SC4s (simulated) | ~91 | 2025 |
| NVIDIA H200 | ~49 | 2024 |
| AMD MI300A | ~110 | 2024 |

The simulated ~91 GF/W for SC4s would put PEZY between H200 and MI300A — a substantial improvement driven by:
1. TSMC 5nm voltage/frequency scaling
2. Reduced PE count (2,048 vs 4,096) focused on higher per-PE throughput at better efficiency
3. 64 MB L3 shared cache reducing HBM3 access frequency
4. MIMD vs SIMD: no warp-divergence waste on scientific workloads

**Important caveat**: SC4s has not shipped; all performance and efficiency numbers are from PEZY simulations, not independent benchmarks.

---

## 8. Interconnect and Scaling

### Intra-chip
The 5-level hierarchy (PE → village → city → prefecture → state) uses a hierarchical on-chip network (NoC) with a level of shared caches at each grouping boundary. PEZY has not published detailed NoC topology.

### Chip-to-host
PCIe Gen 5 × 16: ~128 GB/s bidirectional per chip (standard PCIe slot).

### Node-to-node (InfiniBand)
400 Gb/s NDR InfiniBand per node provides 50 GB/s node-to-node bandwidth for MPI collective operations in multi-node HPC clusters.

### System Scale
- Reference test system: 90 nodes × 4 SC4s = 360 chips
- No proprietary scale-up fabric (unlike NVIDIA NVLink or AMD xGMI); scale-up within a node is via PCIe; scale-out is standard InfiniBand

---

## 9. Comparison with Key FP64 HPC Competitors

| Dimension | PEZY-SC4s | NVIDIA H100 SXM | AMD MI300X |
|-----------|-----------|-----------------|------------|
| Architecture | MIMD manycore | SIMT GPU | SIMT GPU |
| PE/Core count | 2,048 MIMD PEs | 132 SMs (8,448 CUDA cores) | 304 CUs |
| Process | TSMC 5nm | TSMC 4nm | TSMC 5nm |
| Clock | 1.5 GHz | 1.83 GHz | 1.9 GHz |
| FP64 TFLOPS | ~24.6 (sim.) | 66.9 | 163.4 |
| FP16 TFLOPS | — | 1,979 | 1,307 |
| BF16 TFLOPS | Supported | 1,979 | 1,307 |
| HBM | HBM3, 96 GB | HBM3, 80 GB | HBM3, 192 GB |
| Memory BW | 3.2 TB/s | 3.35 TB/s | 5.3 TB/s |
| FP64 GF/W | ~91 (sim.) | ~49 | ~110 |
| Primary market | FP64 HPC | Training + HPC + Inf | Training + HPC |
| Status (2025) | Pre-production | Shipping | Shipping |

PEZY's FP64 TFLOPS are modest by GPU standards (~24.6 vs H100's 66.9), but the efficiency-per-watt argument at FP64 is competitive. The chip is not designed for AI training at scale; BF16 support is an add-on for opportunistic AI workloads.

---

## 10. Key Design Trade-offs

| Decision | Gained | Given Up |
|----------|--------|----------|
| MIMD not SIMD | No branch-divergence penalty; FP64-first | Narrow SIMD (256-bit) limits BF16/INT8 throughput vs GPU |
| 2,048 PEs (vs SC3's 4,096) | Higher per-PE cache, better FP64 efficiency | Lower absolute PE count |
| 64 MB L3 shared cache | Reduces HBM3 access; improves reuse for HPC | Large silicon area cost |
| RISC-V management | Host-less Linux; no x86 dependency | Additional die area; still paired with EPYC in practice |
| Liquid immersion cooling | Extreme density, silent, efficient | Proprietary deployment infrastructure |
| No proprietary scale-up fabric | Standard IB ecosystem | Limited NVLink/xGMI-class bandwidth scaling |
| OpenCL 1.2 (PZCL) | Portable; familiar for HPC developers | Far behind CUDA ecosystem; no ML compiler integration |

---

## 11. Key Uncertainties (Q1 2026)

- SC4s is pre-production; all numbers from PEZY simulations
- No independent benchmark results
- ZettaScaler 4.0 system availability timeline not announced
- AI/ML workload performance not characterized (BF16 added but no INT4/FP8/matrix-dense AI benchmark published)
- Company remains small and Japan-domestic; supply scale unknown post-fraud recovery
- No disclosed multi-chip coherence architecture or NUMA-aware software for multi-SC4s nodes

---

## 12. Investigation Update — 2026-08-08

*Window covered: 2026-04-05 → 2026-08-08. Change class: **moderate** (no hardware specification changed; the edits are a status refresh, a newly identified schedule slip, one new primary citation, and two SC3-based vendor application results).*

**Headline: nothing changed for the SC4s silicon.** Sections 1–11 above stand as written, with the status qualifications recorded below.

### 12.1 SC4s status re-verified: still pre-production

Independent retrieval of PEZY's Japanese news index, English news index, English products page and English homepage on 2026-08-08:

| Check | Result |
|---|---|
| News items in all of calendar 2026 | **Exactly two** — 2026-05-11 and 2026-05-13. Neither mentions SC4s, ZettaScaler 4.0, tape-out, sampling, shipping, or mass production. Next item back is 2025-11-21 (a board appointment) |
| SC4s on products page | **No.** Page lists ZettaScaler 3.0, PEZY-SC3 Processor and Module, ZettaVEGA, PZLAST, Photo Real 3D, 3D Viewer for Medical, AI-based Image Analysis, ZettaScaler-2.0, PEZY-SC2 |
| ZettaScaler 4.0 on products page | **No** |
| Homepage highlights | ZettaScaler 3.0 / PEZY-SC3 / ZettaVEGA only |
| Measured SC4s silicon results anywhere | **None** |

The correct phrasing for the survey is: *still pre-production as of 2026-08-08; SC4s has never appeared on PEZY's product page or in any PEZY press release.*

### 12.2 Newly identified: undeclared schedule slip

PEZY's press release of **2025-06-06** stated that SC4s would "release at the end of the year" — end of 2025. Fourteen months later there is no SC4s product page, no availability announcement, and no silicon results.

**SC4s has missed its publicly stated end-2025 release target by at least 8 months, with no revised date announced.**

This is an *inference from absence*. PEZY has published no cancellation, delay, funding, or corporate-status notice, so the slip is undeclared rather than vendor-confirmed. It should not be written as an acknowledged delay.

### 12.3 New primary publication (IEEE Micro), no spec change

> Hatta, Tsunoda, Uchida, Ishitani, Koizumi, Shioya, Ishii, "PEZY-SC4s: The Fourth-Generation MIMD Many-Core Processor with High Energy Efficiency and Flexibility for HPC and AI," *IEEE Micro*, pp. 1–8. DOI **10.1109/MM.2026.3698804**. Crossref record created 2026-06-01, deposited 2026-06-02 (online ~June 2026, inside the window).

An extended journal version of the Hot Chips 37 talk, **distinct** from the HC37 proceedings paper already cited (DOI 10.1109/HCS66204.2025.11154388, 2025-08-24). This is now the citable primary reference for SC4s.

**Contents not retrieved.** IEEE Xplore returns HTTP 418/403 to automated fetch; neither abstract nor body could be read. It is therefore **not disclosed** whether the Micro version reports measured silicon results or still only the ~91 GF/W simulation. **No SC4s figure in §2–§9 has been upgraded on the strength of this citation.**

### 12.4 What PEZY silicon is actually doing in 2026 — SC3, not SC4s

Both in-window vendor announcements are bioinformatics/genomics applications on the previous-generation PEZY-SC3. Neither is AI, neither is FP64 HPC, and neither involves SC4s.

**2026-05-11 — pzMutect2 (somatic-variant calling for cancer genome analysis)**

| Item | Value |
|---|---|
| Claim | 139× speedup vs Mutect2 in GATK 4.2.6.1 |
| Wall clock | 62 h 36 m 05 s → 26 m 57 s |
| Accuracy | >99.99% concordance, verified via `bcftools isec` |
| System | ZettaVEGA genome-analysis system |
| Processor | **Not named on the release page.** SC3 attribution is inferred from the release slug and PEZY's 2024-12-26 ZettaVEGA announcement |
| Evidence status | Vendor-reported, single workload, no independent replication |

**2026-05-13 — PZLAST-MAG (public protein-sequence search server)**

| Item | Value |
|---|---|
| What | Free public high-speed protein-sequence search over metagenome-assembled genomes; no registration |
| Processor | PEZY-SC3 |
| Partners | National Institute of Genetics (Mori, Kurokawa); ROIS Data Science Common Use Platform Bio-generative AI R&D Center (Higashi) |
| Scale | 210,000+ MAG sources, ~400 M sequences, ~100 B amino acids |
| Latency | Large-scale searches complete in ~5–15 minutes |
| Accuracy | PEZY describes it as comparable to DIAMOND and MMseqs2 (vendor characterization) |
| Hosting | https://pzlast.nig.ac.jp/pzlast/mag |
| Paper | *Bioinformatics Advances*, DOI 10.1093/bioadv/vbag129 |

**Strategic read**: PEZY's visible 2026 commercial activity is ZettaVEGA/PZLAST genome analysis on the shipping 7 nm part, not the SC4s AI/HPC accelerator.

### 12.5 Negative findings, independently re-checked

**Green500, June 2026 — no PEZY in the top 20.** The top 20 runs KAIROS (CALMIP/CNRS, 73.282 GF/W), ROMEO-2025 (70.912), Levante GPU extension (69.426), Isambard-AI phase 1 (68.835), Otus (68.177) … down to Frontier TDS (62.684), and is entirely NVIDIA GH200/H100 and AMD MI300A/MI250X. The SC4s ~91 GF/W FP64 figure therefore has **no measured list entry behind it** and is **not directly comparable** to these mixed-precision-optimized HPL efficiency numbers.

> ⚠️ Scope limit: only the top 20 was verified. "PEZY appears nowhere in the June 2026 Green500" is a claim about all 500 entries that this check cannot support and must **not** be written.

**Hot Chips 38.** No PEZY talk on the program (Aug 23–25, 2026, Stanford Memorial Auditorium) — no SC4s follow-up to HC37. HC38 is in the future as of this writing; its program is not a valid source for any specification claim in either direction.

**Corporate status.** No cancellation, delay, funding, acquisition, or corporate-status news for PEZY or ExaScaler in the window. ⚠️ Caveat: ExaScaler's own site could not be checked — www.exascaler.co.jp serves a TLS certificate valid only for pezy.co.jp/pezy.jp and is unreachable to automated fetch — so this negative rests on PEZY's index alone.

### 12.6 Third-party academic porting activity on SC3s

- Zhu, Cui, Wang, Liu, "PEZY-DPM: Porting and optimization of the DPM Monte Carlo dose calculation code on the PEZY-SC3s MIMD processor," *Computer Physics Communications* **328**:110319 (print Nov 2026; Crossref-indexed 2026-08-07). DOI 10.1016/j.cpc.2026.110319.
- (Pre-baseline, context) "Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s," PACT 2025, DOI 10.1109/PACT65351.2025.00045 (2025-11-03) — an AI convolution kernel on SC3s.

**Naming note.** Both papers use **PEZY-SC3s**, the 512-PE / 109 mm² / TSMC 7 nm small-module variant (recorded in §2 of this document), which the survey's chip-level docs did not previously distinguish from the 4,096-PE PEZY-SC3. That distinction has been made explicit in `chips/pezy/hw-architecture.md` §1.

### 12.7 Correction propagated from the software-stack investigation

The claim "no ML framework backends exist for PEZY" — carried in earlier revisions of `chips/pezy/summary.md`, `chips/pezy/layer-table.md`, `chips/pezy/programming-model.dot`, and `research/pezy/investigations/software-stack.*` — is **wrong** and has been retracted. PEZY announced PyTorch support for the PEZY-SC series on 2025-06-06 (a pre-baseline item the survey should already have carried). See §12 of the software-stack investigation for the corrected, hedged wording.

### 12.8 Updated uncertainties (superseding §11 where they overlap)

- SC4s tape-out / sampling / mass-production status — **not disclosed**
- Revised SC4s release date — **not disclosed**
- Whether the IEEE Micro 2026 article contains measured silicon results — **not disclosed** to this survey
- Whether PEZY appears below rank 20 in the June 2026 Green500 — **unverified**
- ExaScaler corporate status — **unverifiable** (TLS certificate mismatch on exascaler.co.jp)
- Still open from §11: no independent benchmark results; ZettaScaler 4.0 availability timeline unannounced; AI/ML workload performance uncharacterized on SC4s; no disclosed multi-chip coherence architecture

### Sources for this update

- https://www.pezy.co.jp/news/ · https://www.pezy.co.jp/en/news/ (news indexes, fetched 2026-08-08)
- https://www.pezy.co.jp/en/news/news20250606-pezysc-pytorch-generativeai/ (end-2025 SC4s release target; PyTorch announcement)
- https://www.pezy.co.jp/news/news20260511-humangenome-zettavega-pzmutect2/
- https://www.pezy.co.jp/en/news/news20260513-pzlastmag/
- https://www.pezy.co.jp/en/products/ · https://www.pezy.co.jp/en/
- https://api.crossref.org/works/10.1109/mm.2026.3698804
- https://api.crossref.org/works/10.1016/j.cpc.2026.110319
- https://top500.org/lists/green500/list/2026/06/
- https://www.hotchips.org/
