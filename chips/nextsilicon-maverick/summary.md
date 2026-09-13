# NextSilicon Maverick-2 — Summary

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*

**Device class:** Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)
**Manufacturer:** NextSilicon Ltd (Israel, founded 2017), fabricated by TSMC (5 nm)
**Launch:** 2025-10-22 · **Maturity:** Shipping — limited deployment (1 named supercomputer, 1 EU evaluation)
**Deployment:** Sandia National Laboratories "Spectra" (128 accelerators); ODISSEE consortium (4 cards)
**Research date:** 2026-08-08

---

## Scope Note — read this before citing Maverick-2 as an AI chip

Maverick-2 is **datacenter-class hardware** — OAM form factor, 750 W, 192 GB HBM3E, PCIe Gen5, liquid
cooling, deployed in a DOE/NNSA supercomputer. It is **not an AI accelerator**:

- The published peak-performance table lists **FP64, FP32 and FP16 only**. There is **no BF16, no FP8, no
  INT8, no INT4, no structured sparsity** figure in any public source.
- FP32 and FP16 have **identical** scalar/vector throughput (20.3 / 40.7 TFLOPS). There is no low-precision
  throughput multiplier of the kind every AI accelerator has.
- **No AI/ML benchmark of any kind has been published** — no MLPerf, no LLM throughput, no training or
  inference number, no PyTorch result.
- **No framework integration ships.** The vendor product page places CUDA, HIP/ROCm and "leading AI
  frameworks" under "upcoming integrations planned."
- Every published result is HPC (HPCG, STREAM, GUPS, PageRank), vendor-supplied, with unnamed baselines.

It is included in this survey because its **programming model** — compile once with a standard C++
compiler, then let a runtime telemetry loop JIT the spatial mapping onto a reconfigurable grid — is
architecturally distinct from every other device in the registry.

---

## What It Is

Maverick-2 implements NextSilicon's non-von-Neumann **"Intelligent Compute Architecture" (ICA)**. Each die
carries **224 compute blocks** arranged in four regions, each block holding "hundreds of interlinked ALUs",
plus **32 embedded RISC-V E-cores** at the die edges. There is no instruction stream on the grid, no
architected register file, no branch predictor and no speculation. Instead, the toolchain lowers ordinary
compiled C/C++/Fortran IR through MLIR, and the runtime **projects** the resulting dataflow graph onto the
ALUs as **mill cores** — software-defined, transient graph objects that can be laid down "Tetris style" on
a compute block and torn down again.

Code that does not suit dataflow falls back to the on-die RISC-V E-cores, and failing that to the external
x86 host CPU. The vendor's own flow diagram labels these three tiers **GRID / RISC-V E-CORES / HOST CPU**
and claims ~80% of instruction runtime lands on the grid.

---

## Key Architecture Features

| Feature | Single Die (PCIe card) | Dual Die (OAM) |
|---|---|---|
| Process / packaging | TSMC 5 nm, 2.5D | TSMC 5 nm, 2.5D dual-die |
| Frequency | 1.5 GHz | 1.5 GHz |
| Compute blocks | 224 (4 regions × 7 cols × 8) | 448 (2 dies) |
| ALUs per block | **not disclosed** ("hundreds") | **not disclosed** |
| RISC-V E-cores | 32 | 64 |
| Transistors | ~54 B | ~108 B |
| Die area | **not disclosed** | **not disclosed** |
| Chip-level cache | 128 MB | 256 MB |
| HBM | 96 GB HBM3E @ 3.2 TB/s | 192 GB HBM3E @ 6.4 TB/s (up to 288 GB flavors) |
| Peak FP64 | 10.8 TF s/v · 20.2 TF m/t | 21.6 TF s/v · 40.3 TF m/t |
| Peak FP32 | 20.3 TF s/v · 28.2 TF m/t | 40.7 TF s/v · 56.1 TF m/t |
| Peak FP16 | 20.3 TF s/v · 28.2 TF m/t | 40.7 TF s/v · 56.4 TF m/t |
| BF16 / FP8 / INT8 / sparsity | **not disclosed — absent from the table** | **not disclosed** |
| TDP / cooling | 400 W, air | 750 W, liquid |
| Host interface | PCIe Gen5 x16 | PCIe Gen5 x16 |
| On-package network | 1 × 100 GbE | 2 × 100 GbE |

*s/v = scalar/vector, m/t = matrix/tensor. Figures are transcribed from NextSilicon's own spec table as
reproduced by The Next Platform — the vendor's web page does not publish peak FLOPS or HBM bandwidth.*

---

## Software Stack

There is **no kernel language and no device-side source dialect**. The vendor slogan is "BYOC — bring your
own code"; the spec-table programming-model row reads "Any program."

```
Standard C / C++ / Fortran + OpenMP + Kokkos
  → nextcxx (LLVM/MLIR-based compiler driver) — ONE binary, no arch flag, no device object
  → training run on the HOST, under nextsystemd telemetry
  → nextsystemd: identify "top 1% of code flows" → optimized compute graph → PROJECT onto grid
     (state observable via `nextcli application status`: IDLE / OPTIMIZING / IMPROVED)
  → device run — hot regions "hand off" to the grid
  → continuous re-optimization; projections cached persistently across runs
       ↑ telemetry (branch / logic-path execution counts) feeds back continuously
  → libnextapi.so (NextAPI) → kernel driver (unnamed) → Maverick-2 grid
```

- **Only open-source component:** the **Kokkos `NextSilicon` execution space**, merged **upstream** in
  Kokkos 5.2.0 (Apache-2.0 WITH LLVM-exception). Maturity is early and publicly tracked in
  `kokkos/kokkos#9032` — of 16 levels, only `RangePolicy parallel_for` plus DeepCopy/View have landed.
- **MLIR is confirmed** as the compiler stack, from a Kokkos intrinsic comment ("the MLIR compiler stack"),
  a vendor job posting ("innovative MLIR compiler team"), and the public LLVM fork's branch work on
  LLVM-IR→MLIR import, mem2reg/SROA and inlining.
- **Dispatch is synchronous.** `nextapi::parallel_for` is the entire dispatch API — no streams, no queues,
  no kernel launch. `fence()` is a no-op; launch takes a device mutex; one kernel at a time.
- **Memory is UVM.** Host and device share one virtual address space; allocation is a plain
  `std::aligned_alloc`; migration is exception-driven demand paging with page sizes enumerated in the
  Kokkos source from 4 KiB up to **16 GiB**.
- **SDK is gated:** `docs.nextsilicon.com` returns HTTP 401; the architecture whitepaper is "COMING SOON".
  The NextSilicon GitHub org has 14 repos, **all forks** — there is no first-party SDK repository.

---

## Distinguishing Design Choices

1. **No kernel language at all.** Not a DSL, not a CUDA analogue — the input is unmodified ISO C/C++/Fortran.
   The absence is the product.
2. **Runtime JIT of the *spatial mapping*, not of instructions.** Every other accelerator compiles ahead of
   time to a fixed binary; Maverick-2 recompiles its own physical layout while the program runs.
3. **Telemetry observes branches, not counters.** The optimizer works from logic-path execution counts and
   acts by relocation, duplication, in-lining, shrinking and host demotion (patent US20190042282A1).
4. **Deliberately non-uniform NoC.** "NOC barriers" between tiles impose a latency penalty by design, which
   is precisely what makes the optimizer's *relocation* action worth performing.
5. **Hybrid memory model.** Chip-level hardware cache + host/device UVM on top; software-managed,
   non-coherent, explicitly DMA'd SRAM local to each tile underneath.
6. **On-package Ethernet.** 100 GbE integrated on the accelerator itself — unusual, and the only scale-out
   path, since there is no NVLink-class scale-up fabric.
7. **FP64-first numerics.** The design intent is visible in the peak table: FP64 matrix throughput is
   ~72% of FP32's, and FP16 buys nothing over FP32.

---

## Corrections Applied Against Earlier Notes

> - **Die area "~615 mm²" is removed.** No vendor page, spec table, press release, The Next Platform, Chips
>   and Cheese or Hackaday source states any die area. Only transistor counts are public.
> - **"AIIR" is not a NextSilicon IR.** The `mlir-to-aiir` branch in the public LLVM fork has exactly two
>   non-upstream commits, both dated **2026-04-01**, performing a bulk find-and-replace of "MLIR" → "AIIR"
>   with a rocket emoji. It is an April Fools' joke. MLIR itself remains well supported by other evidence.
> - **"Vanguard-II" is not the system name.** Sandia's machine is **Spectra**, the second platform under the
>   Vanguard programme. The vendor's own FAQ and blog use the incorrect shorthand.
> - **Python and CUDA are roadmap, not shipping.** The vendor FAQ and launch blog claim Python, CUDA,
>   TensorFlow, oneAPI, OpenCL and ROCm work today; the vendor **product page** places all of them under
>   "upcoming integrations planned", and all independent corroboration supports the product page only.
> - **Transistor counts must carry their SKU.** ~54 B is per die (The Next Platform); ~108 B is the dual-die
>   OAM (Chips and Cheese). They are reconcilable, not contradictory.
> - **ALU counts are third-party estimates.** The Next Platform states explicitly: "NextSilicon is not
>   releasing the specific number of ALUs per compute block." Their ~196/block and "tens of thousands to
>   close to a hundred thousand" figures are inferred from an illustration and are **not** recorded as specs.

---

## Scale and Deployment

| Site | Scale | Status |
|---|---|---|
| **Sandia National Laboratories — "Spectra"** | 64 nodes × 2 Maverick-2 OAM = **128 accelerators**; Penguin Solutions Tundra integration; Chilldyne negative-pressure liquid cooling | Deployed Jan 2026; **full system acceptance 2026-05-18**. Second platform of the NNSA/ASC **Vanguard** programme (after Astra, 2018). Day-one apps: HPCG, LAMMPS, SPARTA. Mission is fluid-dynamics simulation for nuclear deterrence assessment — **not AI**. A first-of-kind evaluation testbed, **not deployment at scale**. |
| **ODISSEE** (EU Horizon Europe, incl. CERN openlab) | 2 servers / 4 Maverick-2 cards | Small evaluation deployment, 2026-02-17. Vendor-sourced only. |
| "Dozens of customer sites worldwide" | — | **Uncorroborated vendor marketing.** No independent source names any other customer. Unit volumes not disclosed. |

Partnership: **Cornelis Networks** joint reference architectures with the CN5000 400 Gbps fabric
(announced 2026-06-22), extending to CN6000 800 Gbps testing in H2 2026.

Recognition: HPCwire Readers' Choice 2025 — Best HPC Server Product or Technology; Top New Products or
Technologies to Watch.

---

## Performance — all vendor-supplied, all HPC, baselines never named

| Benchmark | Maverick-2 | "GPU" | "CPU" |
|---|---|---|---|
| GUPS | 32.6 (at 460 W) | 5.6 | 1.4 |
| HPCG | 0.8 GFLOPS/W (600 GFLOPS at 750 W) | 0.428 | 0.21 |
| STREAM | 5.2 TB/s = 83.9% of peak | — | — |
| PageRank | ~40 GSTEPS, scales to 40 GB graphs | falls off past 25 GB | — |

Umbrella claims: "up to 10× the performance of leading GPUs while consuming as much as 60% less power";
"4× greater performance-per-watt". No result has been independently reproduced; The Next Platform is openly
skeptical of the STREAM figure ("we do not believe this is possible").

---

## Companion CPU: Arbel (context only — not in the registry)

Unveiled alongside Maverick-2: 64 cores, 3.4 GHz, 10-wide, 480-entry ROB, RVA23 + custom extensions.
Vendor SPEC claims are explicitly "not reviewed or endorsed by SPEC". Vector-unit configuration is
**contested** — the vendor product page says 3 × 256-bit; The Next Platform and the vendor's own June 2026
post both say four 128-bit. Arbel is a **test chip** in "customer and partner silicon evaluations", with
64- and 128-core production parts targeted for **Q1 2028**. The vendor states Arbel was "originally
designed as the control processor within Maverick-2" — i.e. the on-die E-cores are Arbel-lineage. Arbel is
a general-purpose server CPU and should not be entered into a registry of AI accelerators on its own.

---

## Open Questions / Not Disclosed

Die area · ALUs and FPUs per compute block · total ALU count · per-tile SRAM capacity · reservation-station
capacity · coherence domain and protocol of the 128/256 MB cache · NoC topology, width and bandwidth ·
die-to-die link on the OAM (technology, bandwidth, coherence) · HBM stack count · any scale-up fabric
beyond PCIe · BF16/FP8/INT8/sparsity support · mill-core capacity limits and measured reconfiguration
latency · thread/context capacity · E-core microarchitecture · kernel driver name · grid configuration
encoding · `optimizer-pi` / `mlc` config-key expansions · NextAPI's full API surface · Spectra host CPU,
fabric and system software · shipment volumes · any Maverick-3 successor.

---

## Sources

- https://www.nextsilicon.com/maverick
- https://www.nextsilicon.com/tech/
- https://www.nextsilicon.com/faq
- https://www.nextsilicon.com/risc-v/
- https://www.businesswire.com/news/home/20251022712360/en/
- https://www.nextplatform.com/compute/2025/10/22/nextsilicon-takes-aim-at-cpus-and-gpus-with-maverick-2-dataflow-engine/1639749
- https://image.nextplatform.com/214601.webp?imageId=214601&width=1412&height=1200&format=jpg (vendor spec table)
- https://chipsandcheese.com/p/nextsilicon-putting-hpc-first
- https://www.sandia.gov/labnews/2026/01/29/not-the-largest-supercomputer-but-maybe-the-most-interesting/
- https://www.nextsilicon.com/insights/spectra-supercomputer-at-sandia-achieves-full-system-acceptance/
- https://github.com/kokkos/kokkos/tree/develop/core/src/NextSilicon
- https://github.com/kokkos/kokkos/blob/develop/scripts/nextsilicon-test-wrapper.sh
- https://github.com/kokkos/kokkos/issues/9032
- https://github.com/nextsilicon/llvm-project
- https://patents.google.com/patent/US20190042282A1/en
- https://patents.google.com/patent/WO2019055675A1/en
- https://docs.nextsilicon.com/ (HTTP 401 — gated, verified 2026-08-08)
