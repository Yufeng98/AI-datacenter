# PEZY Computing — Search Results

*as_of: 2026-09-13 (original scan 2026-04-05; see "Update Scan — 2026-08-08" and "Update Scan — 2026-09-13" at the end)*
*device_class: Manycore HPC/FP64 (Japan)*

## Search Queries Executed

1. "PEZY Computing SC4S architecture Hot Chips 2025"
2. "PEZY SC2 SC3 manycore processor specifications FP64 HPC"
3. "PEZY Computing supercomputer Green500 Shoubu Suiren energy efficiency"
4. "PEZY-SC4S 4096 cores FP64 performance specifications TSMC 5nm 2025"
5. "PEZY Computing programming model OpenCL SDK PZCL software stack"
6. "PEZY Computing Japan HPC chip history SC1 SC2 SC3 SC4 product line"
7. "PEZY-SC4s RISC-V cores Linux host-less SMT threads village city prefecture hierarchy"
8. "PEZY Computing ZettaScaler liquid immersion cooling supercomputer 2025 2026"
9. "PEZY-SC4s peak FP64 TFLOPS energy efficiency gigaflops per watt HBM3 bandwidth"
10. "PEZY-SC4s PE architecture MIMD 8-thread SMT FP64 FP32 BF16 INT8 compute units"
11. "PEZY Computing ZettaScaler 4.0 SC4s system board AMD EPYC 2025 2026"
12. "PEZY Computing fraud charges Motoaki Saito 2017 2018 recovery status 2024 2025"

## Key Resources Discovered

### Primary Technical Documentation
| Resource | URL | Type |
|----------|-----|------|
| HC2025 Slide Deck — PEZY-SC4s (official PDF) | https://www.pezy.co.jp/wp-content/uploads/2025/09/HC2025.PEZYComputing.NaoyaHatta.v06.pdf | Conference presentation |
| PEZY-SC4s IEEE Xplore (Hot Chips 37, 2025) | https://ieeexplore.ieee.org/document/11154388/ | Peer-reviewed |
| Chips and Cheese — PEZY-SC4s at Hot Chips 2025 | https://chipsandcheese.com/p/pezy-sc4s-at-hot-chips-2025 | Deep-dive analysis |
| ServeTheHome — PEZY-SC4s Hot Chips 2025 | https://www.servethehome.com/pezy-computings-pezy-sc4s-a-mimd-many-core-architecture-at-hot-chips-2025/ | Conference report |
| Next Platform — Why Is Japan Still Investing In Custom FP Accelerators? | https://www.nextplatform.com/2025/09/04/why-is-japan-still-investing-in-custom-floating-point-accelerators/ | Analysis |
| TechRadar — PEZY manycore deep dive | https://www.techradar.com/pro/security/states-prefectures-cities-and-villages-how-one-tiny-japanese-cpu-maker-is-taking-a-radically-different-route-to-making-processors-with-thousands-of-cores | Feature |

### Product Pages
| Resource | URL | Type |
|----------|-----|------|
| PEZY Computing official website | https://www.pezy.co.jp/en/ | Vendor |
| PEZY-SC2 product page | https://www.pezy.co.jp/en/products/pezy-sc2module-processor/ | Vendor |
| PEZY-SC3 product page | https://www.pezy.co.jp/en/products/pezy-sc3/ | Vendor |
| ZettaScaler 3.0 product page | https://www.pezy.co.jp/en/products/zettascaler-3-0/ | Vendor |

### Academic / Reference
| Resource | URL | Type |
|----------|-----|------|
| PEZY-SC3 arXiv paper (2023) | https://arxiv.org/abs/2301.07510 | Paper |
| WikiChip PEZY-SCx family | https://en.wikichip.org/wiki/pezy/pezy-scx | Wiki |
| WikiChip PEZY-SC2 | https://en.wikichip.org/wiki/pezy/pezy-scx/pezy-sc2 | Wiki |
| WikiChip PEZY-SC4 | https://en.wikichip.org/wiki/pezy/pezy-scx/pezy-sc4 | Wiki |
| PEZY Computing Wikipedia | https://en.wikipedia.org/wiki/PEZY_Computing | Wiki |
| ZettaScaler WikiChip | https://en.wikichip.org/wiki/zettascaler | Wiki |

### Programming Model
| Resource | URL | Type |
|----------|-----|------|
| PEZY Systems Tutorial (PZCL) | https://pezy-mokumoku.github.io/tutorial/tutorial/ | Tutorial |

### TOP500 / Green500 References
| Resource | URL | Type |
|----------|-----|------|
| ZettaScaler3.0 TOP500 entry (NA-J2) | https://top500.org/system/180049/ | Database |
| ZettaScaler3.0 TOP500 entry (NA-IT1) | https://top500.org/system/180043/ | Database |

## Summary of Key Findings

### Company
- PEZY Computing: Tokyo-based Japanese fabless chip startup, founded 2010
- CEO Motoaki Saito arrested Dec 2017 for NEDO subsidy fraud (~¥653M/$3.8M); GM received suspended sentence 2018; company recovered and continued R&D
- Focus: MIMD manycore processors for HPC, targeting energy-efficient FP64 compute
- Partner: ExaScaler — manufactures ZettaScaler liquid-immersion-cooled supercomputer systems

### Product Line Evolution
| Generation | Core Count | Process | Clock | FP64 TF | Notable System |
|------------|-----------|---------|-------|---------|----------------|
| PEZY-SC (2014) | 1,024 PEs | TSMC 28nm | 733 MHz | ~1.5 | Shoubu (Green500 #1, 2015, 7 GF/W) |
| PEZY-SC2 (2017) | 2,048 PEs | TSMC 16FF+ | 1.0 GHz | 4.1 | Gyoukou (TOP500 #4, Nov 2017) |
| PEZY-SC3 (2020) | 4,096 PEs | TSMC 7nm | 1.2 GHz | ~19.7 | ZettaScaler3.0, Green500 top-12 |
| PEZY-SC3s | 512 PEs | TSMC 7nm | — | — | Small-form variant (109mm²) |
| PEZY-SC4s (2025) | 2,048 PEs | TSMC 5nm | 1.5 GHz | ~24.6 | ZettaScaler4.0 (90-node test sys) |

### PEZY-SC4s Key Specs (HC2025, pre-production simulation)
- Die: TSMC 5nm, 556 mm², ~600W TDP
- PEs: 2,048 MIMD cores, 8-thread SMT, 4-wide FP64 SIMD
- Memory: 4× HBM3, 96 GB, 3.2 TB/s
- L1: 4 KB I + 4 KB D + 24 KB scratchpad/PE
- L2: 32 KB I + 64 KB D / city (4 villages = 16 PEs)
- L3: 64 MB shared / state (8 prefectures)
- Hierarchy: PE → village (4 PEs) → city (4 villages, 16 PEs) → prefecture (18 cities, ~288 PEs; 16 active) → state (8 prefectures = 2,048 active PEs)
- RISC-V management processor: quad-core Rocket Core @ 1.5 GHz (host-independent Linux)
- Host interface: PCIe Gen 5 x16
- Host system: AMD EPYC 9555P (Zen 5, 64-core Turin)
- Scale-up: 400 Gb/s NDR InfiniBand per node
- Energy efficiency (simulated): ~91 GF/W FP64 (cf. H200 ~49, MI300A ~110)
- Target test cluster: 90 nodes, 737,280 PEs, 8.6 PF FP64

### Software Stack
- PZCL: proprietary OpenCL 1.2-based programming model (PEZY's GPU-analog SDK)
- APIs restricted; access via PEZY contact
- Target workloads: scientific HPC (n-body, lattice QCD, genomics), increasingly AI/ML via BF16 support

---

## Update Scan — 2026-08-08

*Window: 2026-04-05 → 2026-08-08. Outcome: **no SC4s silicon news**. Two SC3-based vendor announcements, one new primary publication, one retracted repo claim.*

### Queries / retrievals executed

1. PEZY news index (Japanese) — full 2026 listing
2. PEZY news index (English) — full listing with per-item hrefs
3. PEZY English products page — current product line
4. PEZY English homepage — highlighted products
5. Crossref bibliographic sweep: "PEZY-SC4s Fourth-Generation MIMD Many-Core Processor"
6. Crossref DOI lookups: 10.1109/MM.2026.3698804, 10.1016/j.cpc.2026.110319
7. Green500 June 2026 list — PEZY / ZettaScaler / ExaScaler presence (top 20)
8. Hot Chips 38 advance program — PEZY presence
9. Independent index sweep: "PEZY SC4s 2026"

### Newly discovered resources

#### Primary — publications
| Resource | URL / DOI | Type | Note |
|---|---|---|---|
| **IEEE Micro (2026) — "PEZY-SC4s: The Fourth-Generation MIMD Many-Core Processor with High Energy Efficiency and Flexibility for HPC and AI", Hatta, Tsunoda, Uchida, Ishitani, Koizumi, Shioya, Ishii, pp. 1–8** | [10.1109/MM.2026.3698804](https://doi.org/10.1109/MM.2026.3698804) | Journal (peer-reviewed) | Extended version of the HC37 talk; Crossref created 2026-06-01. **Distinct** from the HC37 proceedings paper (10.1109/HCS66204.2025.11154388). **Body not retrievable** — Xplore returns HTTP 418/403 to automated fetch; no spec sourced from it |
| Crossref metadata record for the above | https://api.crossref.org/works/10.1109/mm.2026.3698804 | Metadata | Basis for the citation |

#### Vendor announcements
| Resource | URL | Type | Note |
|---|---|---|---|
| PEZY 2025-06-06 — PyTorch / generative-AI support on the PEZY-SC series | https://www.pezy.co.jp/en/news/news20250606-pezysc-pytorch-generativeai/ | Vendor PR | **Key correction source.** Announces PyTorch + Accelerate/DeepSpeed/Transformers/vLLM/TGI/Diffusers on PEZY-SC3; Gemma3/Llama3/Qwen2/SD2/HuBERT/ViT reported working. Also the source of the "SC4s release at the end of the year" (end-2025) target |
| PEZY 2026-05-11 — pzMutect2 on ZettaVEGA | https://www.pezy.co.jp/news/news20260511-humangenome-zettavega-pzmutect2/ | Vendor PR (JA) | 139× vs GATK 4.2.6.1 Mutect2; 62h36m05s → 26m57s; >99.99% concordance via `bcftools isec`. Processor not named on page |
| PEZY 2026-05-13 — PZLAST-MAG | https://www.pezy.co.jp/en/news/news20260513-pzlastmag/ | Vendor PR | Public MAG protein-search server on PEZY-SC3; 210,000+ MAGs, ~400M sequences, ~100B amino acids |
| PZLAST-MAG service | https://pzlast.nig.ac.jp/pzlast/mag | Live service | Free, no registration |
| PEZY news index (English) | https://www.pezy.co.jp/en/news/ | Vendor index | Exactly two 2026 items; prior item 2025-11-21 |
| PEZY news index (Japanese) | https://www.pezy.co.jp/news/ | Vendor index | Same |
| PEZY products page (English) | https://www.pezy.co.jp/en/products/ | Vendor | **No PEZY-SC4s, no ZettaScaler 4.0** as of 2026-08-08 |

#### Academic — third-party
| Resource | URL / DOI | Type | Note |
|---|---|---|---|
| Zhu, Cui, Wang, Liu — "PEZY-DPM: Porting and optimization of the DPM Monte Carlo dose calculation code on the PEZY-SC3s MIMD processor", *Computer Physics Communications* 328:110319 | [10.1016/j.cpc.2026.110319](https://doi.org/10.1016/j.cpc.2026.110319) | Journal | Print Nov 2026; Crossref-indexed 2026-08-07 |
| "Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s", PACT 2025 | [10.1109/PACT65351.2025.00045](https://doi.org/10.1109/PACT65351.2025.00045) | Conference | 2025-11-03; pre-baseline. Only public **AI kernel** optimization work on PEZY hardware located to date |
| PZLAST-MAG paper, *Bioinformatics Advances* | [10.1093/bioadv/vbag129](https://doi.org/10.1093/bioadv/vbag129) | Journal | Associated with the 2026-05-13 announcement |

#### Benchmark lists
| Resource | URL | Note |
|---|---|---|
| Green500, June 2026 | https://top500.org/lists/green500/list/2026/06/ | No PEZY/ZettaScaler/ExaScaler in the **top 20** (KAIROS 73.282 → Frontier TDS 62.684 GF/W; all NVIDIA GH200/H100 and AMD MI300A/MI250X). Only the top 20 was verified |
| Hot Chips 38 | https://www.hotchips.org/ | Aug 23–25 2026, Stanford Memorial Auditorium. No PEZY talk on the advance program. **Future event — not a valid source for any spec claim** |

### Dead ends / unreachable

| Resource | Status |
|---|---|
| IEEE Xplore article page for DOI 10.1109/MM.2026.3698804 | HTTP 418/403 to automated fetch — abstract and body unavailable |
| www.exascaler.co.jp | Unreachable — serves a TLS certificate valid only for pezy.co.jp / pezy.jp. ExaScaler corporate-status negative therefore rests on PEZY's index alone |

### Net effect on the survey

- SC4s remains **pre-production**; no spec above changed.
- Newly recorded: an **undeclared 8+ month slip** past the stated end-2025 release target.
- **Retracted**: the repo's "no ML framework backends (PyTorch/JAX/TensorFlow) exist" claim.
- Added: IEEE Micro primary citation; two SC3 genomics vendor claims; SC3 vs SC3s naming distinction.

---

## Update Scan — 2026-09-13

*Window: 2026-08-08 → 2026-09-13. Outcome: **major** — one new PEZY news item discloses a new vendor-claimed FP64 efficiency figure for the previously-unread IEEE Micro paper. Queries: `"PEZY new AI chip 2026"`, `"PEZY-SC4s next generation"`, `"PEZY AI accelerator announcement 2026"`, `"PEZY SDK release 2026"`, `"PEZY SC4s benchmark"`, `"PEZY SC4s whitepaper"`, plus direct re-fetch of PEZY's own news/products pages and the Crossref/IEEE Xplore DOI redirect.*

| Resource | URL | Type | Note |
|---|---|---|---|
| PEZY news post — IEEE Micro publication announced | https://www.pezy.co.jp/en/news/news20260831-pezysc4s-ieeemicro/ | Vendor PR (2026-08-31) | **New primary finding.** States "115 GFLOPS/W achieved in double precision matrix multiplication, confirming 2.2× improvement versus prior generation"; 576 TFLOPS BF16 peak; 8.9 PFLOPS/90-node system plan (refines 8.6 PF). No shipping/availability info. Provenance (measured vs. simulated) not stated |
| PEZY news index (English), re-fetched | https://www.pezy.co.jp/en/news/ | Vendor index | Exactly **one** new item since 2026-08-08 (the above) |
| PEZY products page (English), re-fetched | https://www.pezy.co.jp/en/products/ | Vendor | **Still no PEZY-SC4s, no ZettaScaler 4.0** as of 2026-09-13 — unchanged from 2026-08-08 |
| IEEE Xplore document page (via DOI redirect) | https://ieeexplore.ieee.org/document/11543438/ | Journal | Still not retrievable to automated fetch — re-confirmed dead end |
| Crossref metadata, re-queried | https://api.crossref.org/works/10.1109/MM.2026.3698804 | Metadata | Confirms IEEE Micro, Vol. 46 Issue 4, July 2026; no abstract field populated |

**Not independently re-verified this pass** (carried forward from 2026-08-08 unchanged): Green500 June 2026 top-20 standing; Hot Chips 38 absence; ExaScaler corporate/TLS status.

**Net effect on the survey**: a new vendor-claimed FP64 efficiency number (115 GFLOPS/W) and a new BF16 peak (576 TFLOPS) are added alongside — not in place of — the existing ~91 GF/W simulated figure; SC4s shipping/product status is unchanged (still pre-production, now 9+ months past its stated end-2025 target).
