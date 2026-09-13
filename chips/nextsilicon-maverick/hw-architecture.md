# NextSilicon Maverick-2 Hardware Architecture

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*
*device_class: Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)*
*Representative products: Maverick-2 PCIe card (single die, 400 W), Maverick-2 OAM (dual die, 750 W)*

---

## Overview

Maverick-2 is a **non-von-Neumann spatial dataflow accelerator** — NextSilicon's "Intelligent Compute
Architecture" (ICA). Each die is a grid of **224 compute blocks** in four regions, each block holding
"hundreds of interlinked ALUs", with **32 embedded RISC-V E-cores** on the outside edges. There is no
instruction stream on the grid, no architected register file, no branch predictor and no speculation.
Instead, a compiled program's dataflow graph is **projected** onto the ALUs at run time as **mill cores**,
and a telemetry-driven runtime continuously re-optimizes that projection while the program executes.

Two SKUs ship, both TSMC 5 nm, 2.5D-packaged and clocked at 1.5 GHz: a single-die PCIe Gen5 x16 card
(400 W, air-cooled) and a dual-die OAM module (750 W, liquid-cooled).

> **Scope caveat.** This is a **HPC-first part, not an AI accelerator.** The published peak table lists
> FP64/FP32/FP16 only, with FP32 and FP16 at identical scalar/vector throughput. No BF16, FP8, INT8 or
> sparsity figure exists publicly, and no AI/ML benchmark has ever been published. Peak FP64 dominates the
> design intent.

The defining architectural characteristic is therefore the inverse of a statically compiled accelerator:
where a TPU or an NPU fixes its dataflow at compile time in a binary, Maverick-2 fixes only the *program*
at compile time and JITs the *physical mapping* at run time, continuously.

---

## 1. Compute Engine

### Grid organization (per die)

| Component | Specification |
|-----------|---------------|
| Architecture | Spatial dataflow grid ("Intelligent Compute Architecture") |
| Compute regions | 4 |
| Compute blocks per region | 56 (7 columns × 8 blocks) |
| Compute blocks per die | **224** |
| ALUs per compute block | **not disclosed** — vendor says "hundreds of interlinked ALUs" |
| Total ALUs per die | **not disclosed** |
| FPUs per compute block | **not disclosed** |
| Embedded RISC-V E-cores | 32 per die, placed on the left and right outside edges |
| Transistors | ~54 B per die; ~108 B for the dual-die OAM |
| Die area | **not disclosed** |
| Process | TSMC 5 nm |
| Frequency | 1.5 GHz |

The 224 / 4-region / 7×8 counts are The Next Platform's own count from the vendor die shot, not a vendor
disclosure. The Next Platform states explicitly that "NextSilicon is not releasing the specific number of
ALUs per compute block"; their ~196/block estimate and their "tens of thousands to close to a hundred
thousand ALUs" total are inferred from an illustration and are **not recorded here as specifications**.

> **Die area:** no vendor page, spec table, press release, The Next Platform, Chips and Cheese or Hackaday
> source states any die area for Maverick-2. Only transistor counts are public.

### Compute-block microarchitecture

Per the vendor block diagram ("NextSilicon Dataflow Architecture"), each compute block is:
**Memory Bus → Dispatch + Reservation Station → Compute Block (ALUs)**, with **MMU, TLB and MEP** attached.

| Unit | Function (vendor annotation, verbatim) |
|------|----------------------------------------|
| Dispatcher | "Triggers computation after verifying the arrival of all data" — the dataflow firing rule |
| Reservation Station | "Analogous to the general-purpose register state of instruction set processors" |
| MMU / TLB | "translate virtual addresses for local memory operations" (one per compute block) |
| MEP — Memory Entry Point | "Also contained on compute blocks and act like memory access instructions" |

Two architectural consequences follow. First, there is **no conventional architected register file** — the
reservation station plays that role, and its capacity and organization are not disclosed. Second, **address
translation is distributed**: every compute block carries its own MMU and TLB, used "sparingly and only
when an ALU calls for specific data. There is no speculation or prediction, just fetching."

NextSilicon's patent claims cover the combination of reservation station + dispatcher + dataflow compute
block.

### Mill cores (not physical cores)

A **mill core** is a *software-defined, dynamically instantiated projection* of a compiled dataflow
subgraph onto ALUs — the vendor FAQ's own term is "software-defined core". Many mill cores can be laid down
on the same compute block "Tetris style", and loaded or deleted "in a matter of nanoseconds". The grid is a
grid of **compute blocks**; mill cores are the transient objects projected onto them.

Not disclosed: how many mill cores fit per compute block, their maximum graph size, and the measured
reconfiguration latency (the "nanoseconds" figure is a vendor marketing claim).

### Threading and latency tolerance

The vendor claims "a mill core can support hundreds of threads at once" against a CPU's two and a GPU's
32–64. The mechanism is disclosed in patent US11875153B1: on an "inconsistent-latency operation" (e.g. a
memory access), a thread's runtime context is spilled into a **context storage table** so the same logical
elements can be reused immediately by another thread, and the original thread resumes later — the dataflow
analogue of GPU warp switching, done by context table rather than warp scheduler. *The patent is not tied to
Maverick-2 silicon; total thread/context capacity is not disclosed.*

Kokkos reports `concurrency() = 65536`, but the source marks this a FIXME placeholder — **not** a hardware
figure.

### Three-tier execution fallback

| Tier | Executes | Share (vendor claim) |
|------|----------|----------------------|
| 1 | Dataflow grid (compute blocks / mill cores) | ~80% of instruction runtime |
| 2 | 32 (64 dual-die) on-die RISC-V E-cores | remainder |
| 3 | External x86 host CPU | remainder |

The vendor's three-stage flow diagram labels these regions **GRID / RISC-V E-CORES / HOST CPU** on a
branch-likelihood histogram — the tri-tier split is the vendor's own framing. E-core microarchitecture
(pipeline, ISA extensions, clock, cache) is **not disclosed**; they are described only as Arbel-lineage
control processors.

---

## 2. Peak Performance

Transcribed from NextSilicon's own spec table as reproduced by The Next Platform. This is the **only public
source** for peak FLOPS — the vendor's web page does not publish them.

| Precision | Single Die scalar/vector | Single Die matrix/tensor | Dual Die scalar/vector | Dual Die matrix/tensor |
|-----------|--------------------------|--------------------------|------------------------|------------------------|
| FP64 | 10.8 TFLOPS | 20.2 TFLOPS | 21.6 TFLOPS | 40.3 TFLOPS |
| FP32 | 20.3 TFLOPS | 28.2 TFLOPS | 40.7 TFLOPS | 56.1 TFLOPS |
| FP16 | 20.3 TFLOPS | 28.2 TFLOPS | 40.7 TFLOPS | 56.4 TFLOPS |
| BF16 / FP8 / INT8 / INT4 / sparsity | **not disclosed — absent from the table** | | | |

**Read this table as an architectural statement.** FP32 and FP16 have identical scalar/vector throughput
and near-identical matrix throughput: there is no low-precision multiplier. FP64 matrix throughput is ~72%
of FP32's. This is an FP64-first design. How the grid is configured as a "matrix/tensor" unit at all is
**not explained** by any source.

---

## 3. Data Path

Execution on the grid is triggered by data arrival, not by an instruction pointer:

```
Memory Bus
    ↓  MEPs issue memory accesses (the "load/store instructions" of the block)
MMU / TLB (per compute block — distributed translation, no speculation)
    ↓
Dispatcher — fires only once ALL operands for a node have arrived
    ↓
Reservation Station — holds operand state (the register-file equivalent)
    ↓
Compute Block: hundreds of interlinked ALUs configured as mill-core subgraphs
    ↓  results flow to the next node in the projected graph
```

Above this, the *mapping itself* is a live object:

```
telemetry (branch / logic-path execution counts)
    ↓
nextsystemd optimizer  — relocation | duplication | in-lining | shrinking | host demotion
    ↓
re-projection of mill cores onto compute blocks (claimed: nanoseconds)
```

There is no static command stream, no `.bin`/`.bmodel`-style pre-scheduled binary, and no published
bitstream format.

---

## 4. On-chip Memory

| Level | Type | Managed by | Notes |
|-------|------|-----------|-------|
| Reservation station (per compute block) | Operand state | Hardware (dataflow-triggered) | Capacity **not disclosed**; replaces the architected register file |
| Tile-local SRAM | Scratchpad | **Software — explicit DMA; NOT coherent** | Per-tile capacity **not disclosed**. CEO: "you need to do a DMA and move data around"; described as CUDA-shared-memory-like |
| Chip-level cache | Hardware cache | Hardware | **128 MB** single die / **256 MB** dual die. Vendor spec table labels this row "L1 cache"; the vendor product page labels it "Cache Coherence" |

This hierarchy is a genuine hybrid and the public sources are in partial tension: the vendor labels the
128/256 MB level coherent while the CEO states the tile-local SRAM is not. **The coherence domain and
protocol of the 128/256 MB level, and the boundary between the coherent and non-coherent regions, are not
publicly specified.** NextSilicon does hold several cache-coherency patents (US20190042427A1,
US12505046B1, US12130736B2), so a coherent hardware tier certainly exists.

---

## 5. Off-chip Memory and Unified Virtual Memory

| SKU | Type | Capacity | Bandwidth |
|-----|------|----------|-----------|
| Single die (card) | HBM3E | 96 GB | 3.2 TB/s |
| Dual die (OAM) | HBM3E | 192 GB (flavors up to 288 GB per Chips and Cheese) | 6.4 TB/s |

HBM stack count and per-stack configuration are **not disclosed** — only the aggregates are public.

**The host↔device model is UVM with exception-based demand paging**, documented directly in upstream
Kokkos source (the best primary evidence available on this chip, since it lives in a non-vendor-controlled
repository):

- Host and device **share one virtual address space**. `NextSiliconSharedSpace::allocate` is a plain
  `std::aligned_alloc` on the host heap — **there is no explicit device allocation call**.
- The "NextSilicon UVM migration runtime" chooses a page size from the alignment of the allocation.
- Migration is fault-driven: "fewer/larger pages improves **exception-based page migration**".
- **Complete supported page-size list, enumerated in code:** 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB,
  16 MiB, 64 MiB, 256 MiB, 1 GiB, 4 GiB, **16 GiB**. This is a hard MMU spec available nowhere else.
- Pages may be **pinned**: `nextapi_mem_migrate(ptr, size, NEXTAPI_PAGE_LOC_HOST, true)`, with a
  `PageLocation { Host, Device, Any }` enum. Kokkos pins its thread-local flags to host to prevent
  migration; and clones its driver object "to prevent the stack from getting migrated to device."

**Synthesis:** at the *allocation* level memory is hardware-managed and transparently migrated; at the
*tile* level it is software-managed, explicitly DMA'd and non-coherent.

---

## 6. Host Interface / Package

| Attribute | Single Die (card) | Dual Die (OAM) |
|-----------|-------------------|----------------|
| Form factor | PCIe full-height, double-width | OAM |
| Host interface | PCIe Gen5 x16 | PCIe Gen5 x16 |
| Host CPU | External | External |
| Packaging | 2.5D | 2.5D dual-die |
| TDP | 400 W | 750 W |
| Cooling | Air | Liquid |

NextSilicon's own **Arbel** RISC-V CPU is intended to fill the host-CPU role eventually; the vendor states
Arbel was "originally designed as the control processor within Maverick-2", making the on-die E-cores
Arbel-lineage. Arbel is a test chip with 64/128-core production parts targeted for Q1 2028 and is not an
AI accelerator in its own right.

---

## 7. On-chip Interconnect (NoC)

The NoC is custom and **non-uniform by design**. Elad Raz (Chips and Cheese): there are "NOC barriers…
barriers in between" tiles; crossing one costs "a penalty… measured in latency rather than in throughput";
"you don't want that one side of the core will communicate to the other side. You want to keep everything
localized."

This is architecturally load-bearing: the runtime optimizer's **relocation** action ("two computation groups
that often call each other are allocated topologically close to each other", patent US20190042282A1) exists
precisely to exploit this non-uniformity.

**Not disclosed:** topology, link width, per-link and bisection bandwidth, and the magnitude of the
cross-barrier latency penalty.

---

## 8. Die-to-Die and Scale-up Interconnect

| Aspect | Status |
|--------|--------|
| Die-to-die (OAM) | Exists — the two dies present as 64 E-cores / 192 GB / 6.4 TB/s combined. Technology, width, bandwidth, latency: **not disclosed**. Whether the two dies form one coherent device or two devices in a package: **not disclosed** |
| Proprietary scale-up fabric | **None in any public source** — no NVLink / Infinity Fabric analogue |
| Multi-device path | PCIe Gen5 x16 plus network |

The absence of a scale-up fabric is consistent with the HPC-first positioning: the target workloads
(HPCG, LAMMPS, SPARTA, graph analytics) are MPI-over-network codes, not tensor-parallel model shards.

---

## 9. Scale-out Interconnect

| Aspect | Value |
|--------|-------|
| On-package Ethernet | **2 × 100 GbE** (OAM) / **1 × 100 GbE** (card), integrated on the accelerator itself |
| Other named fabrics | InfiniBand, Ethernet, RDMA (CEO, system level) |
| Partnership | **Cornelis Networks**, announced 2026-06-22: joint reference architectures with **CN5000 400 Gbps**, extending to **CN6000 800 Gbps** testing in H2 2026. Physical attachment details not specified |

On-package Ethernet on an accelerator is unusual and, given the absence of a scale-up fabric, is the
device's distinguishing system-level interconnect feature.

---

## 10. Product / Deployment Portfolio

| Deployment | Configuration | Status |
|------------|---------------|--------|
| Sandia "Spectra" | 64 nodes × 2 Maverick-2 OAM = **128 accelerators**; Penguin Solutions Tundra; Chilldyne liquid cooling | Deployed Jan 2026; **full system acceptance 2026-05-18**. NNSA/ASC Vanguard programme, second platform. Apps: HPCG, LAMMPS, SPARTA. Evaluation testbed, not scale deployment |
| ODISSEE (EU, incl. CERN openlab) | 2 servers / 4 Maverick-2 cards | Evaluation, 2026-02-17. Vendor-sourced |
| "Dozens of customer sites worldwide" | — | Uncorroborated vendor marketing |

Spectra's host CPU, node interconnect, storage, system software stack and total system power are **not
disclosed** by either Sandia or NextSilicon.

---

## 11. Performance (vendor-supplied; baselines never named)

| Benchmark | Maverick-2 | "GPU" | "CPU" |
|-----------|-----------|-------|-------|
| GUPS | 32.6 (at 460 W) | 5.6 | 1.4 |
| HPCG | 0.8 GFLOPS/W (600 GFLOPS at 750 W) | 0.428 | 0.21 |
| STREAM | 5.2 TB/s = 83.9% of 6.2–6.4 TB/s peak | — | — |
| PageRank | ~40 GSTEPS, scales to 40 GB graphs | falls off past 25 GB | — |

Umbrella claims: "up to 10× the performance of leading GPUs while consuming as much as 60% less power";
"4× greater performance-per-watt". No independent reproduction exists; The Next Platform doubts the STREAM
figure.

**There is no published AI/ML result of any kind.**

---

## 12. Not Disclosed

Die area · ALUs per compute block · total ALU count · FPUs per compute block · per-tile SRAM capacity ·
reservation-station capacity · coherence domain and protocol of the 128/256 MB cache · NoC topology, link
width, bisection bandwidth and cross-barrier penalty · die-to-die technology, bandwidth, latency and
coherence · HBM stack count and per-stack configuration · any scale-up fabric beyond PCIe and 100 GbE ·
BF16/FP8/INT8/INT4/sparsity support · mill-core capacity limits, maximum graph size and measured
reconfiguration latency · thread/context capacity · E-core microarchitecture · grid configuration encoding
(no published ISA or bitstream format) · Spectra host CPU, fabric, storage, system software and total power
· shipment volumes and customer identities beyond Sandia and ODISSEE · any Maverick-3 successor.

---

## Sources

- [Maverick-2 product page](https://www.nextsilicon.com/maverick)
- [NextSilicon Tech page](https://www.nextsilicon.com/tech/)
- [Vendor spec-table figure (via The Next Platform)](https://image.nextplatform.com/214601.webp?imageId=214601&width=1412&height=1200&format=jpg)
- [Compute-block block diagram figure](https://image.nextplatform.com/214597.webp?imageId=214597&width=1412&height=978&format=jpg)
- [Maverick-2 die shot](https://image.nextplatform.com/214596.webp?imageId=214596&width=1412&height=1230&format=jpg)
- [Three-stage flow figure (GRID / RISC-V E-CORES / HOST CPU)](https://image.nextplatform.com/214600.webp?imageId=214600&width=1412&height=598&format=jpg)
- [Vendor benchmark figure](https://image.nextplatform.com/214599.webp?imageId=214599&width=1412&height=874&format=jpg)
- [The Next Platform launch deep-dive](https://www.nextplatform.com/compute/2025/10/22/nextsilicon-takes-aim-at-cpus-and-gpus-with-maverick-2-dataflow-engine/1639749)
- [Chips and Cheese — "NextSilicon: Putting HPC First"](https://chipsandcheese.com/p/nextsilicon-putting-hpc-first)
- [Kokkos_NextSiliconSpace.cpp — UVM page sizes](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSiliconSpace.cpp)
- [Kokkos_NextSilicon_PageAlignedData.hpp — pinning API](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp)
- [Sandia Lab News — Spectra](https://www.sandia.gov/labnews/2026/01/29/not-the-largest-supercomputer-but-maybe-the-most-interesting/)
- [Spectra full system acceptance](https://www.nextsilicon.com/insights/spectra-supercomputer-at-sandia-achieves-full-system-acceptance/)
- [Cornelis Networks partnership](https://www.nextsilicon.com/insights/cornelis-nextSilicon-to-build-joint-reference-architectures-for-ai-and-hpc/)
- [Patent US11875153B1 — concurrent threads on a reconfigurable grid](https://patents.google.com/patent/US11875153B1/en)
- [Patent US20190042282A1 — runtime optimization actions](https://patents.google.com/patent/US20190042282A1/en)
- [Patent US20190042427A1 — reconfigurable cache / coherency](https://patents.google.com/patent/US20190042427A1/en)
- [Patent WO2019055675A1 — grid dataflow architecture](https://patents.google.com/patent/WO2019055675A1/en)
