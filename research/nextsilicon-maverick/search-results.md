# NextSilicon Maverick-2 — Search Results

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*
*device_class: Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)*

---

## Scope Note (read first)

Maverick-2 is **datacenter-class hardware** (OAM, 750 W, HBM3E, PCIe Gen5, deployed in a DOE/NNSA
supercomputer) but it is **HPC-first, not an AI accelerator**. The published peak-performance table lists
**FP64, FP32 and FP16 only** — no BF16, no FP8, no INT8, no sparsity. There is **no MLPerf result, no LLM
throughput number, and no shipping framework integration** of any kind. It is included in this survey for
its *programming model* (compile-once, runtime-JIT projection onto a reconfigurable grid), which is
architecturally distinct from every other device in the registry.

**Research constraint for this pass:** the `WebSearch` tool budget was exhausted at session start. All
results below were obtained by direct `WebFetch`/`curl` of primary sources, the Google Patents XHR API, and
the GitHub REST API. DuckDuckGo HTML, DDG Lite and Mojeek scraping endpoints were attempted and returned
empty. Query strings below therefore record *retrieval targets*, not search-engine queries.

---

## Search Queries / Retrieval Targets

1. `site:nextsilicon.com sitemap.xml` — full vendor URL inventory
2. `nextsilicon.com/maverick` — product spec table and language-support claims
3. `nextsilicon.com/tech/ + /faq + /risc-v/` — ICA description, dev tools, Arbel CPU, contradictions
4. `docs.nextsilicon.com` — SDK documentation accessibility check (returns **HTTP 401**)
5. `nextsilicon.com/insights/*` — Spectra acceptance, Cornelis partnership, Arbel productization, ODISSEE, BYOC
6. `nextsilicon.com/careers/compiler-engineer` — stack disclosure via job postings ("MLIR compiler team")
7. The Next Platform Maverick-2 launch deep-dive (full HTML + all embedded figures)
8. `image.nextplatform.com` figure IDs 214595–214601 — OCR of the vendor spec table, flow diagram, block diagram, die shot, benchmark chart
9. `chipsandcheese.com/p/nextsilicon-putting-hpc-first` — Elad Raz interview on the memory model and NoC
10. `sandia.gov/labnews/2026/01/29/...` — Spectra system configuration and Vanguard programme context
11. GitHub API: `orgs/nextsilicon/repos`, `repos/kokkos/kokkos/git/trees/develop?recursive=1` — locate all NextSilicon paths
12. `raw.githubusercontent.com/kokkos/kokkos/develop/core/src/NextSilicon/*` — full backend source read (19 files)
13. `raw.githubusercontent.com/kokkos/kokkos/develop/scripts/nextsilicon-test-wrapper.sh` — the two-pass execution flow
14. `github.com/nextsilicon/llvm-project/branches` — 41 branch names (MLIR import / inliner / mem2reg / SROA)
15. `patents.google.com/?assignee=Next+Silicon+Ltd` — 41-patent enumeration via the XHR query API

---

## Resources Found

### Official Product / Vendor Pages

| Resource | URL | Category |
|----------|-----|----------|
| Maverick-2 product page (specs, language support) | https://www.nextsilicon.com/maverick | Hardware Spec |
| NextSilicon Tech page (ICA; Profiler / Chip Viewer / Projection Viewer) | https://www.nextsilicon.com/tech/ | Overview |
| NextSilicon FAQ (contradicts product page on languages) | https://www.nextsilicon.com/faq | Overview (marketing) |
| Arbel RISC-V CPU page (64c, 3.4 GHz, RVA23, SPEC disclaimer) | https://www.nextsilicon.com/risc-v/ | Hardware Spec (companion CPU) |
| Launch press release, 2025-10-22 | https://www.businesswire.com/news/home/20251022712360/en/ | Vendor PR wire |
| Insights index (all vendor blog/news posts) | https://www.nextsilicon.com/insights/ | Overview |
| SDK documentation portal — **HTTP 401 Unauthorized** (re-verified 2026-08-08) | https://docs.nextsilicon.com/ | SDK Docs (gated) |
| Vendor sitemap (full URL inventory) | https://www.nextsilicon.com/sitemap.xml | Overview |

### Vendor Blog / Deployment / Roadmap Posts

| Resource | URL | Category |
|----------|-----|----------|
| Spectra @ Sandia achieves full system acceptance (2026-05-18) | https://www.nextsilicon.com/insights/spectra-supercomputer-at-sandia-achieves-full-system-acceptance/ | Deployment |
| ODISSEE consortium at CERN; 2 servers / 4 Maverick-2 cards (2026-02-17) | https://www.nextsilicon.com/insights/elads-blog-odissee-annual-consortium-meeting-cern/ | Deployment |
| Cornelis Networks joint reference architectures, CN5000/CN6000 (2026-06-22) | https://www.nextsilicon.com/insights/cornelis-nextSilicon-to-build-joint-reference-architectures-for-ai-and-hpc/ | Partnership |
| Arbel productized to 64/128-core, Q1 2028 target (2026-06-10) | https://www.nextsilicon.com/insights/nextsilicon-productize-arbel-risc-v-core-into-64core-enterprise-processor-for-ai-hpc/ | Roadmap |
| Maverick-2 launch blog (benchmarks; overclaims Python/CUDA) | https://www.nextsilicon.com/insights/elads-blog-Maverick2-launch/ | Marketing |
| "BYOC — bring your own code" programming-model blog | https://www.nextsilicon.com/insights/BYOC_maverick2_blog/ | Programming Model |
| Beyond von Neumann / dataflow blog | https://www.nextsilicon.com/insights/elads-blog-beyond-von-neumann-dataflow/ | Conceptual |
| SC25 reflections (HPCwire Readers' Choice 2025 awards, 2025-12-03) | https://www.nextsilicon.com/insights/elads-blog-sc25-reflections/ | Marketing |
| Compiler Engineer job posting ("innovative MLIR compiler team") | https://www.nextsilicon.com/careers/compiler-engineer/ | Stack Evidence |
| Runtime Core Software Engineer posting (Linux + RTOS) | https://www.nextsilicon.com/careers/runtime-core-software-engineer/ | Stack Evidence |

### Third-Party Technical Coverage

| Resource | URL | Category |
|----------|-----|----------|
| The Next Platform launch deep-dive (224 blocks, 4 regions, 32 E-cores, 54B transistors, mill cores) | https://www.nextplatform.com/compute/2025/10/22/nextsilicon-takes-aim-at-cpus-and-gpus-with-maverick-2-dataflow-engine/1639749 | Analysis (deepest) |
| Chips and Cheese, "NextSilicon: Putting HPC First" (Elad Raz interview) | https://chipsandcheese.com/p/nextsilicon-putting-hpc-first | Interview |
| Sandia Lab News: Spectra (64 nodes, 128 OAM, Penguin Solutions, Chilldyne) | https://www.sandia.gov/labnews/2026/01/29/not-the-largest-supercomputer-but-maybe-the-most-interesting/ | Primary lab source |
| Hackaday commentary (VLIW/FPGA/Cell framing; Itanium skepticism) | https://hackaday.com/2026/02/20/nextsilicons-maverick-2-the-future-of-high-performance-computing/ | Commentary |

### Vendor Artifacts Reproduced by Third Parties (figures)

These are NextSilicon's own slides, and are the **only public source** for peak FLOPS and HBM bandwidth.

| Resource | URL | Category |
|----------|-----|----------|
| **Vendor spec table** (peak FP64/FP32/FP16; 3.2/6.4 TB/s; 128/256 MB; 32/64 RISC-V; 100 GbE) | https://image.nextplatform.com/214601.webp?imageId=214601&width=1412&height=1200&format=jpg | Hardware Spec |
| Three-stage flow figure (Identify Likely Flow → Optimized Graph → Project onto Grid; TELEMETRY loop; "top 1% of code flows"; patent US16/053,382) | https://image.nextplatform.com/214600.webp?imageId=214600&width=1412&height=598&format=jpg | Programming Model |
| Compute-block block diagram (Dispatch, Reservation Station, MMU, TLB, MEP) | https://image.nextplatform.com/214597.webp?imageId=214597&width=1412&height=978&format=jpg | Microarchitecture |
| Maverick-2 die shot (four-quadrant compute-block array) | https://image.nextplatform.com/214596.webp?imageId=214596&width=1412&height=1230&format=jpg | Die Photo |
| Benchmark figure (GUPS, HPCG GFLOPS/W, STREAM, PageRank) | https://image.nextplatform.com/214599.webp?imageId=214599&width=1412&height=874&format=jpg | Performance (vendor) |
| ISA-vs-dataflow silicon-allocation figure ("~2% dedicated to compute") | https://image.nextplatform.com/214595.webp?imageId=214595&width=1412&height=946&format=jpg | Conceptual |

### Open-Source Code (the strongest evidence in this chip)

| Resource | URL | Category |
|----------|-----|----------|
| **Upstream Kokkos NextSilicon backend (19 files, Apache-2.0 WITH LLVM-exception)** | https://github.com/kokkos/kokkos/tree/develop/core/src/NextSilicon | Portability Layer |
| `Kokkos_NextSiliconSpace.cpp` — UVM page migration; full page-size list 4 KiB → 16 GiB | https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSiliconSpace.cpp | Memory Model |
| `Kokkos_NextSilicon_ThreadSpaceGuard.hpp` — training vs handoff semantics | https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_ThreadSpaceGuard.hpp | Execution Model |
| `Kokkos_NextSilicon_Intrinsics.hpp` — `__ns_immutable_thread_invariant_parameter_struct`, "the MLIR compiler stack" | https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_Intrinsics.hpp | Compiler Evidence |
| `Kokkos_NextSilicon_ParallelFor_Range.hpp` — `nextapi::parallel_for`, device mutex, driver cloning | https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_ParallelFor_Range.hpp | Dispatch API |
| `Kokkos_NextSilicon_PageAlignedData.hpp` — `nextapi_mem_migrate`, `PageLocation` enum | https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp | Memory Model |
| **`scripts/nextsilicon-test-wrapper.sh`** — the two-pass training→projection→device flow | https://github.com/kokkos/kokkos/blob/develop/scripts/nextsilicon-test-wrapper.sh | Runtime Model |
| Kokkos SNL CI workflow — `nextcxx`, `$NEXT_HOME/llvm/bin`, runner `ns-1.2.0` | https://github.com/kokkos/kokkos/blob/develop/.github/workflows/snl-ci.yml | Toolchain |
| `FindTPLNEXTAPI.cmake` — `libnextapi`, `$NEXT_HOME`, default `/opt/nextsilicon` | https://github.com/kokkos/kokkos/blob/develop/cmake/Modules/FindTPLNEXTAPI.cmake | SDK Layout |
| Kokkos CHANGELOG — 5.2.0 NextSilicon entry | https://raw.githubusercontent.com/kokkos/kokkos/develop/CHANGELOG.md | Release Proof |
| Kokkos issue #9032 — NextSilicon Backend Tracking, 16 levels | https://github.com/kokkos/kokkos/issues/9032 | Maturity |
| Kokkos PR #8998 — execution + memory spaces | https://github.com/kokkos/kokkos/pull/8998 | PR |
| Kokkos PR #9100 — RangePolicy `parallel_for`, DeepCopy | https://github.com/kokkos/kokkos/pull/9100 | PR |
| Kokkos PR #9162 — enable persistent optimizer cache | https://github.com/kokkos/kokkos/pull/9162 | Runtime |
| Kokkos PR #9378 — LockPolicy; device clock counter unavailable | https://github.com/kokkos/kokkos/pull/9378 | Limitation |
| NextSilicon GitHub org (14 repos, **all forks**; no first-party SDK repo) | https://github.com/orgs/nextsilicon/repositories | Overview |
| `nextsilicon/llvm-project` fork (41 branches; MLIR import / inliner / mem2reg / SROA) | https://github.com/nextsilicon/llvm-project | Compiler |
| `mlir-to-aiir` branch — 2 commits, 2026-04-01, bulk MLIR→AIIR rename (**April Fools; NOT a real IR**) | https://github.com/nextsilicon/llvm-project/commits/mlir-to-aiir | Seed Correction |
| `nextsilicon/kokkos-public` (staging branches `nextsilicon-backend-pr-1/2/2b/2c`) | https://github.com/nextsilicon/kokkos-public | Staging |
| `nextsilicon/kokkos-kernels-public` (BLAS/sparse) | https://github.com/nextsilicon/kokkos-kernels-public | Libraries |
| `nextsilicon/kokkos-tools-public` (profiling) | https://github.com/nextsilicon/kokkos-tools-public | Tooling |
| `nextsilicon/ns-gem5` (gem5 fork; RISC-V PMP, cache/prefetcher work) | https://github.com/nextsilicon/ns-gem5 | Simulator |

### Patents (the only public microarchitecture disclosure)

| Resource | URL | Category |
|----------|-----|----------|
| WO2019055675A1 — "Directed and interconnected grid dataflow architecture" (priority 2017-09-13; Raz, Tayari) | https://patents.google.com/patent/WO2019055675A1/en | Foundational |
| US20190042282A1 — "Runtime optimization of configurable hardware" (app. US16/053,382 — the number printed on the vendor flow figure) | https://patents.google.com/patent/US20190042282A1/en | Optimizer Loop |
| US11875153B1 — "Executing concurrent threads on a reconfigurable processing grid" | https://patents.google.com/patent/US11875153B1/en | Threading |
| US11995419B1 — "GUI for code to dataflow graph representation" (Projection Viewer; LEUs/GCUs) | https://patents.google.com/patent/US11995419B1/en | Dev Tools |
| EP4668102A1 — "Dynamic software interface translation…" (does **not** mention CUDA) | https://patents.google.com/patent/EP4668102A1/en | Portability |
| US20190042427A1 — "Reconfigurable cache architecture and methods for cache coherency" | https://patents.google.com/patent/US20190042427A1/en | Memory |
| US12056376B2 — "Interconnected memory grid with bypassable units" | https://patents.google.com/patent/US12056376B2/en | Memory |
| US20260086816A1 — "Optimizing execution of code on reconfigurable hardware using likely data…" | https://patents.google.com/patent/US20260086816A1/en | Optimizer |
| Google Patents assignee query — 41 Next Silicon Ltd patents | https://patents.google.com/?assignee=Next+Silicon+Ltd | Index |

---

## Key Findings

- **Vendor**: NextSilicon Ltd (Israel, founded 2017; ~$303M raised over 8 years per The Next Platform).
  Launched Maverick-2 on **2025-10-22**.
- **Device class**: non-von-Neumann "Intelligent Compute Architecture" (ICA) — a runtime-reconfigurable
  spatial dataflow grid. Standard C/C++/Fortran IR is *projected* onto the grid at run time; there is **no
  kernel language and no device-side source dialect at all**.
- **Two SKUs**: PCIe Gen5 x16 FHDW card (single die, 96 GB HBM3E @ 3.2 TB/s, 128 MB, 400 W, air-cooled) and
  OAM (dual die, 192 GB HBM3E @ 6.4 TB/s, 256 MB, 750 W, liquid-cooled). TSMC 5 nm, 2.5D, 1.5 GHz.
- **Per die**: 224 compute blocks in 4 compute regions (7 columns × 8 blocks × 4 regions), each holding
  "hundreds of interlinked ALUs"; 32 embedded RISC-V E-cores at the die edges; ~54B transistors
  (~108B for the dual-die OAM).
- **Datatype verdict**: FP64 / FP32 / FP16 only. FP32 and FP16 have *identical* scalar/vector throughput —
  there is no low-precision multiplier. **No BF16, FP8, INT8 or sparsity figure exists publicly.**
- **Programming model is the product**: compile once with `nextcxx` → run a **training pass** under
  `nextsystemd` telemetry → the runtime identifies the "top 1% of code flows", builds an optimized compute
  graph, and **projects** it onto the grid as *mill cores* → re-run on device. Projections are cached
  persistently across runs. Observable via `nextcli application status` (`IDLE` / `OPTIMIZING` / `IMPROVED`).
- **Only open-source component**: the **Kokkos `NextSilicon` execution space**, merged **upstream** in
  Kokkos 5.2.0. Maturity is early and publicly tracked — only `RangePolicy parallel_for` + DeepCopy/View
  have landed of 16 tracked levels.
- **MLIR is confirmed** as the compiler stack (Kokkos intrinsic comment + job posting + LLVM fork branch
  work). **"AIIR" is not a NextSilicon IR** — the `mlir-to-aiir` branch is a two-commit April Fools rename.
- **Deployment**: Sandia National Laboratories **"Spectra"** — 64 nodes × 2 OAM = **128 accelerators**,
  second platform of the NNSA/ASC **Vanguard** programme, full system acceptance **2026-05-18**. Plus a
  small EU **ODISSEE** evaluation (2 servers / 4 cards). "Dozens of customer sites" is uncorroborated.
- **Every published benchmark is HPC and vendor-supplied with unnamed baselines**: GUPS 32.6, HPCG
  0.8 GFLOPS/W (600 GFLOPS @ 750 W), STREAM 5.2 TB/s, PageRank ~40 GSTEPS. **Zero AI/ML results.**
- **SDK is gated**: `docs.nextsilicon.com` returns HTTP 401; the architecture whitepaper is "COMING SOON".
  There is no ISCA/MICRO/Hot Chips publication for this part.
