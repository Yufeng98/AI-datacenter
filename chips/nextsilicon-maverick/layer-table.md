# NextSilicon Maverick-2 Layer Mapping Table

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*
*device_class: Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)*

> **Scope flag.** Maverick-2 is datacenter-class HPC hardware, not an AI accelerator. Several conventional
> AI-stack layers are marked **N/A by design** below — that is the architecture, not missing research.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | **None shipping.** Vendor product page places CUDA, HIP/ROCm and "leading AI frameworks" under "upcoming integrations planned" | confirmed | nextsilicon-maverick-page |
| Framework Integration | Vendor FAQ and launch blog claim Python / CUDA / TensorFlow / oneAPI / OpenCL / ROCm work today — **contradicted by the vendor's own product page**; no independent corroboration exists. Record as roadmap | confirmed (as a contradiction) | nextsilicon-faq, nextsilicon-launch-blog, nextsilicon-maverick-page |
| Graph Capture | **N/A by design** — there is no ML graph in this programming model | confirmed | nextsilicon-maverick-page |
| Graph Compiler | **N/A by design** — no ML graph compiler exists | confirmed | nextsilicon-maverick-page |
| Kernel Language | **N/A by design** — no kernel language, no DSL, no CUDA analogue, no device-side source dialect. Vendor slogan "BYOC — bring your own code"; spec-table row reads "Any program" | confirmed | byoc-blog, spec-table-figure |
| Host Language | Shipping: **ISO C / C++ / Fortran + OpenMP + Kokkos**. Fortran is vendor-stated only (no Flang-specific artifact found; front end not disclosed) | confirmed | nextsilicon-maverick-page, kokkos-upstream |
| Tensor API | **N/A by design** — the unit of work is `parallel_for` over an index range, not a tensor | confirmed | kokkos-nextsilicon-parallelfor |
| Compiler / IR | **`nextcxx`** — LLVM/MLIR-based C++ compiler driver; bundled LLVM binutils at `$NEXT_HOME/llvm/bin/`; CI uses `-DCMAKE_CXX_STANDARD=20`; **no `Kokkos_ARCH_*` flag exists** (unique among Kokkos device backends) | confirmed | kokkos-snl-ci, kokkos-arch-cmake |
| Compiler / IR | **MLIR confirmed as the stack**: Kokkos intrinsic `__ns_immutable_thread_invariant_parameter_struct` documented as communicating "to the MLIR compiler stack"; vendor job posting advertises an "innovative MLIR compiler team" | confirmed | kokkos-ns-intrinsics, nextsilicon-compiler-job |
| Compiler / IR | Public LLVM fork `nextsilicon/llvm-project` (41 branches, active to 2026-08-01): LLVM-IR→MLIR import with metadata preservation (`loop-info*`, `parameter_attribute*`, `access-group-attr`, `passthrough-import`), scalar opt in MLIR (`mlir-mem2reg`, `mlir-sroa`), inliner work | likely (inferred from branch names) | nextsilicon-llvm-fork |
| Compiler / IR | **"AIIR" is NOT a NextSilicon IR** — the `mlir-to-aiir` branch has two commits, both 2026-04-01, performing a bulk "MLIR"→"AIIR" find-and-replace with a rocket emoji. April Fools' joke | confirmed | nextsilicon-llvm-aiir-branch |
| Compiler / IR | MLIR → grid lowering, and the grid configuration encoding, are **proprietary and not disclosed** | confirmed | (absence verified) |
| Device Binary | **N/A** — one host binary; no fat binary, no separate device object, no `.bmodel`/`.plan` analogue. The device "binary" is a runtime-generated projection, never materialized to a public format | confirmed | kokkos-test-wrapper, kokkos-snl-ci |
| Runtime | **`nextsystemd`** — optimizer + telemetry daemon; identifies hot flows, builds the optimized compute graph, projects it onto the grid. Config tree `optimizer-pi: { enable-telemetry-less, mlc: { acceleration-threshold } }` (key expansions **not disclosed**) | confirmed | kokkos-test-wrapper |
| Runtime | **`libnextapi.so` (NextAPI)** — host API. Public surface reachable from Kokkos: `nextapi::parallel_for` (the entire dispatch API), `__next_is_in_handed_off_code()`, `nextapi_memory_copy/fill`, `nextapi_mem_migrate`. Full API surface **not disclosed** (no reference manual; docs are 401-gated) | confirmed | findtpl-nextapi, kokkos-nextsilicon-src |
| Runtime | **`nextcli`** — control CLI. `nextcli application status` exposes `Optimization state: IDLE \| OPTIMIZING \| IMPROVED`; `nextcli system status` reports the device as "Maverick 2" | confirmed | kokkos-test-wrapper, kokkos-snl-ci |
| Runtime | **Persistent optimizer cache** under `XDG_CACHE_HOME` — amortizes the training pass across runs (Kokkos CI maps a volume to `$HOME/optimizer-cache`) | confirmed | kokkos-pr-9162 |
| Runtime semantics | Dispatch is **synchronous and blocking** (no streams/queues/async); `fence()` is a **no-op**; kernel launch takes a **device mutex** (one kernel at a time); functor copied into a **4 MB minimum** shared-space heap buffer; driver cloned to prevent stack migration to device | confirmed | kokkos-nextsilicon-parallelfor |
| Runtime semantics | `KOKKOS_IF_ON_HOST/DEVICE` **cannot be resolved at compile time** — a runtime thread-local is used, because "NextSilicon kernels may run on the host during training/telemetry collection while still being 'within' the device execution space" | confirmed | kokkos-threadspaceguard |
| Runtime limitations | `printf` (CS-515) and `abort` (CS-694) from grid code do not work; `fork()` unsupported; device-side cycle counters unavailable (PR #9378) | confirmed | kokkos-nextsilicon-src, kokkos-pr-9378 |
| Portability Layer | **Kokkos `NextSilicon` execution space + `NextSiliconSharedSpace`** — merged **upstream** in Kokkos 5.2.0 (PRs #8998, #9100); 19 files, Apache-2.0 WITH LLVM-exception; space factory `"180_NextSilicon"`; `LayoutLeft`; `concurrency() = 65536` (**FIXME placeholder, not a hardware spec**) | confirmed | kokkos-changelog, kokkos-nextsilicon-src, kokkos-pr-8998 |
| Portability Layer | Maturity: tracking issue #9032 lists 16 levels; **only `RangePolicy parallel_for` + DeepCopy/View have landed**. Outstanding: `parallel_reduce`, MDRange, TeamPolicy, nested parallelism, scratch memory, `parallel_scan`, SIMD, Containers, Algorithms, Benchmarks, device-only `NextSiliconSpace` | confirmed | kokkos-issue-9032 |
| Portability Layer | Vendor staging fork `nextsilicon/kokkos-public` contains the **same file set** as upstream — nothing more advanced is public | confirmed | nextsilicon-kokkos-public |
| Profiling / Debug | **Profiler** (real-time telemetry: performance patterns and bottlenecks) — proprietary | confirmed | nextsilicon-tech-page |
| Profiling / Debug | **Chip Viewer** (live hardware-level information) — proprietary | confirmed | nextsilicon-tech-page |
| Profiling / Debug | **Projection Viewer** (visualizes what happens inside each mill core in real time). Patent US11995419B1 describes source↔graph GUI with control-flow branch counters, cache-bin and HBM occupancy, across logical → compute → hardware → projection levels; patent vocabulary is "logical element units (LEUs)" in "grid compute units (GCU)" | confirmed | nextsilicon-tech-page, patent-us11995419b1 |
| Profiling / Debug | Kokkos Tools: `Kokkos::Profiling::Experimental::DeviceType::NextSilicon` (open); device_id reporting stubbed pending NextAPI support (CS-611) | confirmed | kokkos-nextsilicon-src |
| Driver / Firmware | Kernel driver exists (PCIe Gen5 device) but is **not named publicly**; source closed; device node not disclosed. SDK installs to `$NEXT_HOME`, default **`/opt/nextsilicon`**. SDK versions seen in CI: 1.2.0, 1.3.0-395 | likely | findtpl-nextapi, kokkos-snl-ci |
| ISA | **No published ISA, bitstream format, or configuration-word specification.** Grid configuration is a projected dataflow graph, not an instruction stream. Fallback tiers execute RISC-V (E-cores) and x86-64 (host) | confirmed | (absence verified) |
| Communication | **No collective communication library** (no NCCL/RCCL analogue) named in any source | confirmed | (absence verified) |
| Documentation | `docs.nextsilicon.com` returns **HTTP 401 Unauthorized** (verified 2026-08-08); architecture whitepaper "COMING SOON"; GitHub org has 14 repos, **all forks**, no first-party SDK repo | confirmed | docs-401, nextsilicon-github-org |

## Execution Model (the distinguishing mechanism)

| Step | Detail | Confidence | Sources |
|------|--------|------------|---------|
| 1. AOT compile | One binary via `nextcxx`. No device object, no fat binary, no arch flag | confirmed | kokkos-snl-ci |
| 2. Training run | Program executes **on the host** while `nextsystemd` collects telemetry | confirmed | kokkos-test-wrapper, kokkos-threadspaceguard |
| 3. Optimize / project | `nextsystemd` identifies the "top 1% of code flows", builds an optimized compute graph, projects it onto the grid as mill cores. States: `IDLE` ("no mills found") / `OPTIMIZING` / `IMPROVED`. CI budget: 5 minutes | confirmed | kokkos-test-wrapper, flow-figure |
| 4. Device run | Re-execute; hot regions "hand off" to the grid | confirmed | kokkos-test-wrapper |
| 5. Continuous re-optimization | Telemetry keeps flowing; mapping keeps changing. "The longer the code runs, the better it gets" | confirmed | nextplatform-deepdive |
| 6. Persistent cache | Projections cached across runs, amortizing the training pass | confirmed | kokkos-pr-9162 |
| Alt. telemetry-less mode | Static projection path (`enable-telemetry-less: true`) where "any projection error counts as a failure" — **currently DISABLED upstream**: "FIXME_NEXTSILICON: re-enable when telemetry-less mode is working" | confirmed | kokkos-test-wrapper |
| Telemetry observable | **Branch / logic-path execution counts**, not conventional performance counters | confirmed | flow-figure, patent-wo2019055675a1, patent-us11995419b1 |
| Optimizer actions | **Relocation** (co-locate frequently communicating groups), **duplication** (replicate bottlenecked groups), **in-lining** (fuse groups), **shrinking** (release unused duplicates), **host demotion** (move a function to the CPU) | confirmed | patent-us20190042282a1, nextsilicon-faq |
| Optimizer internals | Projection/mapping algorithm, telemetry sampling rate, telemetry record format and cache format: **not disclosed** | confirmed | (absence verified) |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Non-von-Neumann spatial dataflow grid ("Intelligent Compute Architecture"); no instruction stream, no branch predictor, no speculation | confirmed | nextplatform-deepdive, chipsandcheese |
| Compute Engine | **224 compute blocks per die** in 4 regions (7 columns × 8 blocks × 4). Third-party count from the vendor die shot, not a vendor disclosure | likely | nextplatform-deepdive, die-shot-figure |
| Compute Engine | ALUs per compute block: **not disclosed** — "hundreds of interlinked ALUs" (vendor); TNP states explicitly that the number is not being released. Their ~196/block estimate is **not** a spec | confirmed (as non-disclosure) | nextplatform-deepdive |
| Compute Engine | Per compute block: **Dispatcher** (fires on full operand arrival), **Reservation Station** (register-file equivalent — there is no architected register file), **MMU/TLB** (distributed translation), **MEPs** (memory-access "instructions") | confirmed | block-diagram-figure |
| Compute Engine | **Mill core = software-defined projection**, not a physical core. Many fit on one compute block "Tetris style"; loaded/deleted in claimed nanoseconds. Per-block capacity, max graph size, measured reconfig latency: **not disclosed** | confirmed | nextplatform-deepdive, nextsilicon-faq |
| Compute Engine | **32 RISC-V E-cores per die** (64 dual-die) on the outside edges; Arbel-lineage. Microarchitecture **not disclosed** | confirmed | spec-table-figure, arbel-productization-post |
| Compute Engine | Three-tier fallback: **GRID → RISC-V E-CORES → HOST CPU** (vendor's own diagram labels). Vendor claims ~80% of instruction runtime on the grid | confirmed | flow-figure, nextplatform-deepdive |
| Compute Engine | Threading: vendor claims "hundreds of threads per mill core"; mechanism is a **context storage table** spill on inconsistent-latency operations (patent US11875153B1). Total capacity **not disclosed**; Kokkos' 65536 is a FIXME placeholder | unconfirmed | patent-us11875153b1, kokkos-nextsilicon-src |
| Numerics | **FP64 / FP32 / FP16 only.** Single die: 10.8/20.2, 20.3/28.2, 20.3/28.2 TFLOPS (scalar-vector / matrix-tensor). Dual die: 21.6/40.3, 40.7/56.1, 40.7/56.4. **FP32 and FP16 scalar/vector throughput are identical** — no low-precision multiplier | confirmed | spec-table-figure |
| Numerics | **BF16, FP8, INT8, INT4 and structured sparsity: not disclosed** — absent from the published table; no source states whether the hardware supports them at all | confirmed (absence verified) | spec-table-figure |
| Numerics | How the grid is configured as a "matrix/tensor" unit: **not explained by any source** | confirmed | nextplatform-deepdive |
| Data Path | Dataflow-triggered: MEP → MMU/TLB → Dispatcher (fires on operand arrival) → Reservation Station → ALUs. No static command stream, no pre-scheduled binary | confirmed | block-diagram-figure |
| On-chip Memory | **Tile-local SRAM: software-managed, explicit DMA, NOT cache coherent** (CEO: "you need to do a DMA and move data around"). Per-tile capacity **not disclosed** | confirmed | chipsandcheese |
| On-chip Memory | **Chip-level hardware cache: 128 MB (single die) / 256 MB (dual die)**. Vendor spec table labels the row "L1 cache"; vendor product page labels it "Cache Coherence". **Coherence domain and protocol not disclosed**; boundary with the non-coherent tile SRAM unspecified | likely | spec-table-figure, nextsilicon-maverick-page, patent-us20190042427a1 |
| Off-chip Memory | **HBM3E**: 96 GB @ **3.2 TB/s** (single die) / 192 GB @ **6.4 TB/s** (dual die); flavors up to 288 GB. Stack count and per-stack config **not disclosed** | confirmed | spec-table-figure, chipsandcheese |
| Memory Model | **UVM**: host and device share one virtual address space; allocation is a plain `std::aligned_alloc` (no device-alloc call); **exception-based demand page migration**; page sizes 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB, 4 GiB, **16 GiB**; pinning via `nextapi_mem_migrate` with `PageLocation {Host, Device, Any}` | confirmed | kokkos-nextsiliconspace, kokkos-pagealigneddata |
| Host Interface / Package | **PCIe Gen5 x16** both SKUs. Card: PCIe FHDW, single die, air-cooled, **400 W**. OAM: dual die, liquid-cooled, **750 W**. TSMC **5 nm**, **2.5D**, **1.5 GHz**. Host CPU "External" | confirmed | spec-table-figure, nextsilicon-maverick-page |
| Host Interface / Package | Transistors: **~54 B per die**, **~108 B** dual-die OAM. **Die area: not disclosed** by any source | confirmed | nextplatform-deepdive, chipsandcheese |
| On-chip Interconnect | Custom NoC, **non-uniform by design**: "NOC barriers" between tiles impose a **latency** (not throughput) penalty. This is what makes the optimizer's *relocation* action worthwhile. Topology, width, bandwidth and penalty magnitude **not disclosed** | confirmed | chipsandcheese |
| Scale-up Interconnect | **No proprietary chip-to-chip fabric** in any public source (no NVLink/Infinity Fabric analogue). Multi-device via PCIe Gen5 x16 plus network | confirmed | (absence verified) |
| Scale-up Interconnect | Die-to-die on the OAM: exists, but technology, width, bandwidth, latency and **whether the two dies present as one coherent device or two** are all **not disclosed** | unconfirmed | (absence verified) |
| Scale-out Interconnect | **On-package Ethernet: 2 × 100 GbE (OAM) / 1 × 100 GbE (card)** integrated on the accelerator — unusual, and the primary scale-out path | confirmed | spec-table-figure, chipsandcheese |
| Scale-out Interconnect | **Cornelis Networks** joint reference architectures with **CN5000 400 Gbps** (announced 2026-06-22), extending to **CN6000 800 Gbps** testing in H2 2026. Attachment details unspecified | confirmed | cornelis-post |
| Deployment | Sandia **"Spectra"**: 64 nodes × 2 OAM = **128 accelerators**; Penguin Solutions Tundra; Chilldyne liquid cooling; HPCG/LAMMPS/SPARTA; NNSA/ASC Vanguard programme; full system acceptance **2026-05-18**. Evaluation testbed, **not deployment at scale**. Host CPU / fabric / storage / system software / total power **not disclosed** | confirmed | sandia-labnews, spectra-acceptance-post |
| Deployment | **ODISSEE** (EU, incl. CERN openlab): 2 servers / 4 cards, 2026-02-17. Vendor-sourced | likely | odissee-post |
| Deployment | "Dozens of customer sites worldwide" and "product volumes in Q4 2025": **uncorroborated vendor marketing**; no independent source names another customer; volumes not disclosed | unconfirmed | nextsilicon-faq |
| Performance | All published results are **HPC and vendor-supplied with unnamed baselines**: GUPS 32.6 @ 460 W; HPCG 0.8 GFLOPS/W (600 GFLOPS @ 750 W); STREAM 5.2 TB/s (83.9% of peak); PageRank ~40 GSTEPS to 40 GB graphs | confirmed (as vendor claims) | benchmark-figure, businesswire-pr |
| Performance | **No AI/ML result of any kind** — no MLPerf, no LLM throughput, no training or inference number, no PyTorch benchmark | confirmed (absence verified) | (absence verified) |
| Companion CPU | **Arbel** RISC-V (64c, 3.4 GHz, 10-wide, 480-entry ROB, RVA23 + custom): **test chip**, production 64/128-core parts targeted **Q1 2028**. Vector config **contested** (vendor page 3 × 256-bit vs TNP and vendor's own June 2026 post: four 128-bit). SPEC figures explicitly **not reviewed by SPEC**. Not an AI accelerator; **not in the registry on its own** | likely | nextsilicon-riscv-page, arbel-productization-post, nextplatform-deepdive |
