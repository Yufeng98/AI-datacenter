# PEZY Computing (PEZY-SC4s) Software and Hardware Stack Summary

*as_of: 2026-09-13 (original research 2026-04-05; see "PEZY Update — 2026-08-08" and "Update (2026-09-13)" below)*

---

## Overview

PEZY Computing is a Tokyo-based fabless chip startup (founded 2010) that has spent 15 years building MIMD (Multiple Instruction Multiple Data) manycore processors for energy-efficient FP64 supercomputing. Their central architectural bet: **thousands of small in-order MIMD cores at modest clocks deliver better FP64 energy efficiency than wide SIMD GPU architectures** for the irregular, divergent codes that dominate scientific HPC.

The company has validated this bet with a consistent Green500 track record: PEZY-powered systems topped the Green500 (most energy-efficient supercomputers) in 2015 and 2017, and ranked #12 in 2021. The latest generation, **PEZY-SC4s** (Hot Chips 37, August 2025), extends this with TSMC 5nm, 4× HBM3 (96 GB, 3.2 TB/s), a new 64 MB L3 cache, BF16 support, and a novel RISC-V management processor enabling host-independent Linux operation. Simulated FP64 efficiency is ~91 GF/W — approximately 1.9× NVIDIA H200's ~49 GF/W at FP64.

The chip is **still pre-production as of 2026-08-08**: SC4s has never appeared on PEZY's product page or in any PEZY press release, and PEZY's own 2025-06-06 release stated it would "release at the end of the year" (end-2025) — a target now missed by 8+ months with no revised date announced. The partner company ExaScaler manufactures the ZettaScaler liquid-immersion-cooled supercomputer systems that house PEZY accelerators. See "PEZY Update — 2026-08-08" below.

---

## Hardware Architecture

### PEZY-SC4s: Core Design

The SC4s contains **2,048 MIMD PEs** running at 1.5 GHz. The PE count is halved vs SC3 (4,096), but each PE is substantially more capable: larger caches, 8-thread SMT, HBM3, BF16 support, and a 64 MB shared L3 cache.

Each PE contains:
- **8 hardware threads** (SMT8): two groups of 4 threads. Fine-grained multithreading rotates threads every cycle within a group; coarse-grained swaps full groups on long-latency events (e.g., HBM3 access)
- **4-wide FP64 SIMD** (256-bit): FP64 is first-class, unlike GPUs that typically halve FP64 throughput
- **BF16 support** (new in SC4s): added for AI-era workloads; wider effective SIMD for low-precision
- **L1**: 4 KB I + 4 KB D + 24 KB SW-managed scratchpad

The hierarchy uses a geographic naming scheme: PE → Village (4 PEs sharing scratchpad) → City (16 PEs + 32 KB/64 KB L2) → Prefecture (~256 PEs) → State (2,048 PEs + 64 MB shared L3). Total on-chip SRAM is estimated ~140 MB.

Off-chip: 4× HBM3 stacks at 96 GB, 3.2 TB/s — a major upgrade from SC3's HBM2 (32 GB, ~512 GB/s).

### MIMD vs GPU SIMD

The key architectural differentiation:

| Aspect | GPU (H100, SIMD) | PEZY-SC4s (MIMD) |
|--------|-----------------|-----------------|
| Branch divergence | Expensive (warp masking/serialization) | None (independent per PE) |
| FP64 per watt | ~49 GF/W | ~91 GF/W (simulated) |
| FP64 absolute | 66.9 TFLOPS | ~24.6 TFLOPS |
| AI throughput | 989 FP16 TFLOPS | Not primary target |
| Vector width | 1,024–2,048 bits | 256 bits |

PEZY wins on FP64 per watt. GPU wins on absolute throughput (all precisions) and AI workload support.

### RISC-V Management Processor (SC4s New Feature)

An on-chip quad-core RISC-V Rocket Core cluster (1.5 GHz) runs Linux natively on the SC4s die. Prior PEZY generations required an external AMD EPYC host to boot and manage the coprocessor. SC4s can operate standalone — though the ZettaScaler 4.0 reference system still pairs it with an AMD EPYC 9555P (Zen 5, 64-core Turin) for host compute.

### ZettaScaler System

ZettaScaler 4.0 node:
- 4× PEZY-SC4s + AMD EPYC 9555P
- 400 Gb/s NDR InfiniBand
- Liquid immersion cooling (fluorine-type inert liquid: non-toxic, non-combustible, high electrical insulation, no sealing required)
- Planned test cluster: 90 nodes / 360 chips / 737,280 PEs / 8.6 PF FP64

Liquid immersion cooling enables compact, silent supercomputer deployments suitable for offices and labs — a differentiated packaging capability vs GPU-based systems.

---

## Software Stack

### Programming Model: PZCL (OpenCL 1.2 base)

PZCL (PEZY Computing Language) is the sole programming interface:
- Based on OpenCL 1.2 (2011 standard)
- Host code (C/C++) runs on AMD EPYC or RISC-V management processor
- Kernel code runs on PEZY PEs
- Access-controlled: API documentation requires contacting PEZY directly

The OpenCL 1.2 base is familiar to HPC developers but old by AI toolchain standards (2011 specification). No public MLIR/LLVM path. No tensor operation library.

**ML framework support (corrected 2026-08-08).** PEZY announced a PyTorch backend for the PEZY-SC series in June 2025, with Accelerate/DeepSpeed/Transformers/vLLM/TGI/Diffusers reported functional on PEZY-SC3 and inference demonstrated for Gemma3, Llama3, Qwen2, Stable Diffusion 2, HuBERT and ViT. This is a vendor claim only: no public repository, release tag, version number, benchmark, or documentation for this backend could be located, and the PZCL SDK remains access-controlled. Treat the PyTorch path as announced-but-unverifiable rather than as an open ecosystem. (This retracts the earlier statement in this document that "No ML framework backends (PyTorch, JAX, TensorFlow) exist" — that was a pre-existing error.)

### Software Stack (Simplified)

```
Scientific Application (Fortran/C/C++ + MPI)
        ↓
PZCL Host API (OpenCL 1.2 base; access-controlled)
        ↓
PZCL Compiler (OpenCL-C → proprietary PEZY PE binary)
        ↓
PZCL Runtime (queue, DMA, scratchpad management)
        ↓
Linux PCIe Driver (kernel module)
        ↓
PCIe Gen 5 → PEZY-SC4s (RISC-V management → 2,048 PEs + HBM3)
        ↓
InfiniBand (MPI → multi-node HPC cluster)
```

### Multi-chip Communication

Within a node: PCIe peer-to-peer (no proprietary chip-to-chip fabric). No NVLink/xGMI equivalent — each SC4s has independent HBM3. Multi-chip coordination is via AMD EPYC host or MPI.

Between nodes: Standard InfiniBand (400 Gb/s NDR), standard MPI collectives.

---

## Position in the AI Chip Landscape

PEZY occupies a narrow, differentiated niche:

**Strengths:**
- Best-in-class Green500 FP64 energy efficiency track record (Japan domestic)
- Liquid immersion cooling enables compact deployments
- MIMD removes branch-divergence waste for irregular scientific codes
- RISC-V management adds supply-chain independence from x86 CPU vendors
- Consistent process technology advancement (28nm → 16nm → 7nm → 5nm)

**Weaknesses:**
- Pre-production chip (SC4s) as of 2026-08-08; all efficiency numbers are simulations, and the publicly stated end-2025 release target has been missed by 8+ months with no revised date
- Small company; supply scale and availability limited
- Access-controlled proprietary SDK with no public ML ecosystem
- AI framework support is announced (PyTorch on the SC series, June 2025) but unverifiable — no public repo, release tag, version number, or benchmark; no independent AI training or inference result exists
- No proprietary high-bandwidth scale-up fabric (limits AI model parallelism)
- Historical corporate instability (2017 fraud arrest; recovery underway)
- OpenCL 1.2 base significantly behind CUDA/ROCm/XLA ecosystem sophistication

**Target customer**: Japan domestic scientific HPC centers (RIKEN, JAMSTEC, KEK, national labs) that need energy-efficient FP64 supercomputers and can deploy liquid-immersion-cooled ZettaScaler systems.

**2026 commercial reality check**: PEZY's only visible commercial activity in 2026 to date is bioinformatics/genomics on the *previous-generation* PEZY-SC3 (ZettaVEGA, PZLAST) — not AI, and not FP64 HPC on SC4s. See the update section below.

---

## PEZY Update — 2026-08-08 (No SC4s Silicon News; Undeclared Schedule Slip; PyTorch Retraction)

*Updated 2026-08-08. Window covered: 2026-04-05 → 2026-08-08. Primary sources: PEZY Japanese and English news indexes, PEZY English products page and homepage (all fetched 2026-08-08); PEZY press release 2025-06-06; Crossref records for IEEE Micro DOI 10.1109/MM.2026.3698804 and Computer Physics Communications DOI 10.1016/j.cpc.2026.110319; June 2026 Green500 list.*

**Nothing changed for the SC4s silicon in this window.** The prior-generation content above stands; the items below are corrections and additions.

### 1. SC4s status: still pre-production (confirmed independently)

- PEZY's news index carries **exactly two items for all of calendar 2026** — 2026-05-11 and 2026-05-13 — and neither mentions SC4s, ZettaScaler 4.0, tape-out, sampling, shipping, or mass production. The next item back is 2025-11-21 (a board appointment).
- The English products page as of 2026-08-08 lists ZettaScaler 3.0, PEZY-SC3 Processor and Module, ZettaVEGA, PZLAST, Photo Real 3D, 3D Viewer for Medical, AI-based Image Analysis, ZettaScaler-2.0, and PEZY-SC2. **There is no PEZY-SC4s entry and no ZettaScaler 4.0 entry.** The homepage highlights ZettaScaler 3.0 / PEZY-SC3 / ZettaVEGA only.

### 2. Undeclared schedule slip (new)

PEZY's 2025-06-06 press release stated that SC4s was planned for "release at the end of the year" — i.e. end of 2025. Fourteen months later there is still no SC4s product page, no availability announcement, and no silicon results. **SC4s has missed its publicly stated end-2025 release target by at least 8 months, with no revised date announced.** This is an inference from absence: PEZY has published no cancellation, delay, funding, or corporate-status notice, so the slip is *undeclared*, not vendor-confirmed.

### 3. New primary citation: IEEE Micro journal version of the SC4s paper

An extended journal version of the Hot Chips 37 talk appeared in IEEE Micro in ~June 2026 (Crossref record created 2026-06-01, deposited 2026-06-02):

> N. Hatta, Tsunoda, Uchida, Ishitani, Koizumi, Shioya, Ishii, "PEZY-SC4s: The Fourth-Generation MIMD Many-Core Processor with High Energy Efficiency and Flexibility for HPC and AI," *IEEE Micro*, pp. 1–8. DOI [10.1109/MM.2026.3698804](https://doi.org/10.1109/MM.2026.3698804).

This is a **distinct publication** from the HC37 proceedings paper (DOI 10.1109/HCS66204.2025.11154388, 2025-08-24) already cited below. **Whether the Micro version reports measured silicon or still only the ~91 GF/W simulation is not disclosed** — IEEE Xplore refuses automated retrieval, so no SC4s number in this document has been upgraded on the strength of this citation.

### 4. Correction: PEZY *has* announced a PyTorch backend (retraction of a pre-existing repo error)

Earlier revisions of this document stated "No ML framework backends (PyTorch, JAX, TensorFlow) exist" and "No AI training support; BF16 added but no ML framework integration." **Both statements were wrong and have been retracted in place** (see Software Stack and Weaknesses above). PEZY announced on **2025-06-06** that the PEZY-SC series supports PyTorch, along with Accelerate, DeepSpeed, Transformers, vLLM, Text Generation Inference and Diffusers, running on PEZY-SC3 / ZettaScaler 3.0, with Gemma3, Llama3, Qwen2, Stable Diffusion 2, HuBERT and Vision Transformer reported working; SC4s support was said to follow with the SC4s release.

The correct characterization is *announced-but-unverifiable*: no PyTorch fork, PZCL/OpenCL PrivateUse1 backend, MLIR/LLVM path, package, or SDK version number could be located from any independent source. The "no public ML ecosystem / access-controlled SDK" weakness stands unchanged.

### 5. 2026 vendor announcements — both SC3-based genomics, neither about SC4s

| Date | Item | Claim | Evidence status |
|---|---|---|---|
| 2026-05-11 | **pzMutect2** on ZettaVEGA | 139× speedup of somatic-variant calling vs Mutect2 in GATK 4.2.6.1 — wall clock **62 h 36 m 05 s → 26 m 57 s**, >99.99% concordance verified via `bcftools isec` | Vendor claim, single workload, no independent replication. The release page does not name the processor; the SC3/ZettaVEGA attribution is inferred from the release slug and PEZY's 2024-12-26 ZettaVEGA announcement. **Record as an SC3 result, not an SC4s result.** |
| 2026-05-13 | **PZLAST-MAG** | Free public protein-sequence search server for metagenome-assembled genomes, running on PEZY-SC3. Indexes 210,000+ MAGs, ~400 M sequences, ~100 B amino acids; searches complete in ~5–15 min with accuracy PEZY describes as comparable to DIAMOND and MMseqs2 | Vendor claim; built with the National Institute of Genetics (Mori, Kurokawa) and the ROIS Data Science Common Use Platform Bio-generative AI R&D Center (Higashi). Hosted at pzlast.nig.ac.jp, no registration. Associated paper: *Bioinformatics Advances*, DOI 10.1093/bioadv/vbag129 |

**Strategic read**: both of PEZY's 2026 announcements are bioinformatics applications on the previous-generation SC3, not AI and not FP64 HPC on SC4s.

### 6. Negative findings (independently re-checked)

- **Green500, June 2026**: no PEZY / ZettaScaler / ExaScaler system appears in the **top 20**, which is led by KAIROS (CALMIP/CNRS, 73.282 GFlops/W), ROMEO-2025 (70.912), Levante GPU extension (69.426), Isambard-AI phase 1 (68.835), Otus (68.177), down to Frontier TDS (62.684). The SC4s ~91 GF/W FP64 figure remains a **simulation with no measured list entry behind it**, and it is not directly comparable to these mixed-precision-optimized HPL efficiency numbers. *Only the top 20 was verified* — "PEZY appears nowhere in the June 2026 Green500" is not proven.
- **Hot Chips 38** (Aug 23–25, 2026, Stanford Memorial Auditorium): the advance program contains no PEZY talk. HC38 is in the future at the time of writing and is not a valid source for any spec claim either way.
- **No cancellation, delay, funding, acquisition, or corporate-status news** for PEZY or ExaScaler in the window. Caveat: ExaScaler's own site could not be checked — www.exascaler.co.jp presents a TLS certificate valid only for pezy.co.jp/pezy.jp — so this negative rests on PEZY's index alone.
- **No PZCL/SDK release, no version number, and no public MLIR/LLVM path** found in the window.

### 7. Third-party academic porting activity (additive)

- Zhu, Cui, Wang, Liu, "PEZY-DPM: Porting and optimization of the DPM Monte Carlo dose calculation code on the PEZY-SC3s MIMD processor," *Computer Physics Communications* **328**:110319 (print Nov 2026; Crossref-indexed 2026-08-07).
- (Pre-baseline, for context) "Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s," PACT 2025, DOI 10.1109/PACT65351.2025.00045 (2025-11-03) — an AI convolution kernel on SC3s.

Both papers use the designation **PEZY-SC3s** (the 512-PE, 109 mm² small-module 7 nm variant), which this survey has not consistently distinguished from the 4,096-PE PEZY-SC3. Third-party porting work — largely from Chinese academic groups — continues on SC3s while SC4s remains unreleased.

---

## Update (2026-09-13)

*Change class: major. Window: 2026-08-08 → 2026-09-13. Full detail: `research/pezy/investigations/hw-architecture.md` §13; `chips/pezy/hw-architecture.md` §10.*

PEZY's news index gained exactly one item in this window: **2026-08-31**, announcing publication of the IEEE Micro journal paper on SC4s (the same paper this survey flagged on 2026-08-08 as "content not retrievable"). PEZY's own summary of the paper discloses a new FP64 efficiency figure — **"115 GFLOPS/W achieved in double precision matrix multiplication, confirming 2.2× improvement versus prior generation"** — plus a first-time quantified **BF16 peak of 576 TFLOPS**, and a refined system-scale figure of **8.9 PFLOPS FP64** for the planned 90-node test cluster (previously 8.6 PF). All figures are **(vendor claim)**; PEZY's "2.2× improvement" does not arithmetically reconcile against either the previously known 24.6 GF/W (SC3, measured) or 91 GF/W (SC4s, simulated) figures, so it is recorded as stated rather than derived. **Whether 115 GFLOPS/W reflects a silicon measurement or a refined simulation is not disclosed** — IEEE Xplore continues to block automated retrieval of the paper itself (re-confirmed 2026-09-13). Peak FP64 (24.6 TFLOPS), HBM3 capacity/bandwidth (96 GB / 3.2 TB/s), and process (TSMC 5nm) are unchanged. **SC4s remains absent from PEZY's products page** (re-checked 2026-09-13); the end-2025 release target is now missed by 9+ months with no revised date. No other PEZY news, and no corporate-status, funding, or ExaScaler change, was found in the window. Sources: https://www.pezy.co.jp/en/news/news20260831-pezysc4s-ieeemicro/, https://www.pezy.co.jp/en/products/, https://api.crossref.org/works/10.1109/MM.2026.3698804

---

## Resources

### Primary
- **[IEEE Micro (2026) — "PEZY-SC4s: The Fourth-Generation MIMD Many-Core Processor with High Energy Efficiency and Flexibility for HPC and AI", Hatta et al., pp. 1–8, DOI 10.1109/MM.2026.3698804](https://doi.org/10.1109/MM.2026.3698804)** — extended journal version of the HC37 talk; published in the July–August 2026 issue (PEZY announced this 2026-08-31). Citable primary reference. Full text still not retrievable to this survey (Xplore blocks automated fetch as of 2026-09-13); the 115 GFLOPS/W and 576 TFLOPS BF16 figures below come from PEZY's own news-page summary of the paper, not the paper text itself.
- [PEZY news post announcing the IEEE Micro publication (2026-08-31) — source of the 115 GFLOPS/W and 576 TFLOPS BF16 figures](https://www.pezy.co.jp/en/news/news20260831-pezysc4s-ieeemicro/)
- [PEZY-SC4s Hot Chips 2025 Slides (official PDF)](https://www.pezy.co.jp/wp-content/uploads/2025/09/HC2025.PEZYComputing.NaoyaHatta.v06.pdf)
- [IEEE Xplore — PEZY-SC4s HC37 proceedings paper (DOI 10.1109/HCS66204.2025.11154388)](https://ieeexplore.ieee.org/document/11154388/)
- [PEZY Computing website](https://www.pezy.co.jp/en/)
- [PEZY products page](https://www.pezy.co.jp/en/products/) — as of 2026-09-13 still lists no PEZY-SC4s and no ZettaScaler 4.0
- [ZettaScaler 3.0 product page](https://www.pezy.co.jp/en/products/zettascaler-3-0/)

### Vendor announcements (2025–2026)
- [2025-06-06 — PyTorch / generative-AI support on the PEZY-SC series](https://www.pezy.co.jp/en/news/news20250606-pezysc-pytorch-generativeai/) — also the source of the "SC4s release at the end of the year" (end-2025) target
- [2026-05-11 — pzMutect2 on ZettaVEGA: 139× vs GATK 4.2.6.1 Mutect2](https://www.pezy.co.jp/news/news20260511-humangenome-zettavega-pzmutect2/) (Japanese)
- [2026-05-13 — PZLAST-MAG public MAG protein-search server on PEZY-SC3](https://www.pezy.co.jp/en/news/news20260513-pzlastmag/) · [service](https://pzlast.nig.ac.jp/pzlast/mag)
- [2026-08-31 — IEEE Micro publication of the PEZY-SC4s paper announced](https://www.pezy.co.jp/en/news/news20260831-pezysc4s-ieeemicro/)
- [PEZY news index (English)](https://www.pezy.co.jp/en/news/) · [PEZY news index (Japanese)](https://www.pezy.co.jp/news/)

### Analysis
- [Chips and Cheese — PEZY-SC4s at Hot Chips 2025](https://chipsandcheese.com/p/pezy-sc4s-at-hot-chips-2025)
- [Next Platform — Why Is Japan Still Investing In Custom FP Accelerators?](https://www.nextplatform.com/2025/09/04/why-is-japan-still-investing-in-custom-floating-point-accelerators/)
- [ServeTheHome — HC2025 Report](https://www.servethehome.com/pezy-computings-pezy-sc4s-a-mimd-many-core-architecture-at-hot-chips-2025/)
- [TechRadar feature — PEZY manycore deep dive](https://www.techradar.com/pro/security/states-prefectures-cities-and-villages-how-one-tiny-japanese-cpu-maker-is-taking-a-radically-different-route-to-making-processors-with-thousands-of-cores)

### Academic
- [PEZY-SC3 arXiv (2023)](https://arxiv.org/abs/2301.07510)
- Zhu, Cui, Wang, Liu — "PEZY-DPM: Porting and optimization of the DPM Monte Carlo dose calculation code on the PEZY-SC3s MIMD processor," *Computer Physics Communications* 328:110319 (2026). DOI [10.1016/j.cpc.2026.110319](https://doi.org/10.1016/j.cpc.2026.110319)
- "Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s," PACT 2025. DOI [10.1109/PACT65351.2025.00045](https://doi.org/10.1109/PACT65351.2025.00045)
- [PZLAST-MAG paper — *Bioinformatics Advances*, DOI 10.1093/bioadv/vbag129](https://doi.org/10.1093/bioadv/vbag129)
- [WikiChip PEZY-SCx family](https://en.wikichip.org/wiki/pezy/pezy-scx)

### Benchmark lists
- [Green500, June 2026](https://top500.org/lists/green500/list/2026/06/) — no PEZY/ZettaScaler system in the top 20 (top-20 verified only)

### Programming
- [PEZY Systems Tutorial (PZCL)](https://pezy-mokumoku.github.io/tutorial/tutorial/)
