# NextSilicon Maverick-2 Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*
*device_class: Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)*

---

## Overview

**NextSilicon Ltd** (Israel, founded 2017; ~$303M raised over eight years per The Next Platform) launched
**Maverick-2** on **2025-10-22**. It is a non-von-Neumann **"Intelligent Compute Architecture" (ICA)**: a
spatial grid of compute blocks holding large numbers of interlinked ALUs, onto which the toolchain and
runtime *project* a dataflow graph derived from ordinary compiled C/C++/Fortran. There is no instruction
stream on the grid, no branch predictor, and no speculation.

Two SKUs ship: a **single-die PCIe Gen5 x16 card** (400 W, air-cooled) and a **dual-die OAM module**
(750 W, liquid-cooled). Both are TSMC 5 nm, 2.5D-packaged, and clocked at 1.5 GHz.

**Scope caveat — this part is HPC-first, not an AI accelerator.** The published peak-performance table
lists FP64, FP32 and FP16 only, with FP32 and FP16 at *identical* scalar/vector throughput. No BF16, FP8,
INT8 or structured-sparsity figure exists in any public source, and no AI/ML benchmark of any kind has been
published. Peak FP64 dominates the design intent. Maverick-2 is included in this survey for the novelty of
its runtime-JIT programming model, not for AI throughput.

---

## 1. Canonical Specification Table

The single most authoritative artifact is NextSilicon's own product spec table, reproduced as a figure in
The Next Platform's launch coverage. It is **more complete than the vendor's own web page** — it is the
only public source giving peak FLOPS and HBM bandwidth. Rows are transcribed verbatim.

| Spec | Maverick-2 Single Die (PCIe card) | Maverick-2 Dual Die (OAM) |
|------|-----------------------------------|---------------------------|
| Production year | 2024 | 2024 |
| Form factor | PCIe card, full-height, double-width | OAM |
| Host CPU | External | External |
| RISC-V E-cores | 32 | 64 |
| Max chip power (TDP) | 400 W | 750 W |
| Host interface | PCI Express 5.0 x16 | PCI Express 5.0 x16 |
| Process | 5 nm (TSMC) | 5 nm (TSMC) |
| Frequency | 1.5 GHz | 1.5 GHz |
| HBM capacity | HBM3E 96 GB | HBM3E 192 GB |
| HBM bandwidth | **3.2 TB/s** | **6.4 TB/s** |
| "L1 cache" (vendor row label) | 128 MB | 256 MB |
| On-package network | 1 × 100 GbE | 2 × 100 GbE |
| Thermal solution | Air-cooled | Liquid |
| Peak FP64 | **10.8 TFLOPS** scalar/vector; **20.2 TFLOPS** matrix/tensor | **21.6 TFLOPS** scalar/vector; **40.3 TFLOPS** matrix/tensor |
| Peak FP32 | **20.3 TFLOPS** scalar/vector; **28.2 TFLOPS** matrix/tensor | **40.7 TFLOPS** scalar/vector; **56.1 TFLOPS** matrix/tensor |
| Peak FP16 | **20.3 TFLOPS** scalar/vector; **28.2 TFLOPS** matrix/tensor | **40.7 TFLOPS** scalar/vector; **56.4 TFLOPS** matrix/tensor |
| Programming model (vendor row) | "Any program" | "Any program" |

**Cross-checks.** The vendor product page independently confirms TSMC 5 nm, 2.5D packaging, 1.5 GHz,
96/192 GB HBM3E, 128/256 MB, 400/750 W and PCIe Gen5 x16 — but labels the 128/256 MB row **"Cache
Coherence"** rather than "L1 cache". The vendor launch deck independently states "ICA SINGLE DIE 96 GB
3.2 TB/S" and "ICA DUAL DIE 192 GB 6.4 TB/S". Chips and Cheese confirms 6.4 TB/s, 192 GB and two 100 GbE
ports, and adds that capacities "up to 288 GB" exist as flavors.

### Datatype note — the AI-relevance verdict

FP32 and FP16 scalar/vector throughput are **identical** (20.3 / 40.7 TFLOPS), and their matrix/tensor
figures differ by 0.3 TFLOPS on the dual die. There is no low-precision throughput multiplier of the kind
every AI accelerator has. Combined with the absence of BF16/FP8/INT8 from the table entirely, this is
decisive hardware-side evidence that Maverick-2 is an HPC part. How the grid is configured as a
"matrix/tensor" unit at all is **not explained** by any source (The Next Platform: "We are not sure how to
configure the Maverick-2 as a matrix/tensor unit").

---

## 2. Compute Engine

### Execution paradigm

Non-von-Neumann dataflow. Co-founder/CTO Ilan Tayari states that traditional CPUs "dedicate 98 percent of
their silicon to overhead, traffic management, data shuffling – not actual computation", and that ICA
"pivot[s] the silicon allocation ratio… We're not trying to hide latency, but to tolerate and minimize it
by design." CEO Elad Raz to Chips and Cheese: "there is no branch predictor whatsoever. There is a data
flow." *(vendor claims, quoted by third parties)*

### Physical organization (per die)

| Element | Value | Evidence class |
|---------|-------|----------------|
| Compute regions | 4 | third-party count from vendor die shot |
| Compute blocks per region | 56 (7 columns × 8 blocks) | third-party count |
| Compute blocks per die | **224** | third-party count |
| ALUs per compute block | "hundreds of interlinked ALUs" — **exact count not disclosed** | vendor claim / explicit withholding |
| Total ALUs per die | **not disclosed** | — |
| FPUs per compute block | **not disclosed** | — |
| RISC-V E-cores per die | 32 (on the left and right outside edges) | vendor spec table + TNP |
| Transistors per die | ~54 billion | TNP |
| Transistors, dual-die OAM | ~108 billion | Chips and Cheese |
| Die area | **not disclosed** | verified absent across all retrieved sources |

The 224-block figure is The Next Platform's own count from the die photograph, not a vendor disclosure:
"By our count, there is a grid of seven columns of compute blocks that each have eight compute blocks, for
a total of 224 compute blocks on the die." TNP additionally *estimates* ~196 ALUs/block from a 14×14
illustration and totals "many tens of thousands to close to a hundred thousand ALUs" — these are
**third-party estimates and are not recorded here as specifications**. TNP states plainly: "NextSilicon is
not releasing the specific number of ALUs per compute block."

> **Die area: do not record a number.** No vendor page, spec table, press release, The Next Platform,
> Chips and Cheese or Hackaday article states any die area for Maverick-2. Only transistor counts are
> public. A "~615 mm²" figure circulating in earlier notes is unsupported.

### Compute-block microarchitecture

The vendor block diagram ("NextSilicon Dataflow Architecture") shows, per compute block:
**Memory Bus → Dispatch + Reservation Station → Compute Block (ALUs)**, with **MMU, TLB and MEP** attached
on the side. Verbatim annotations from the figure:

| Unit | Vendor annotation |
|------|-------------------|
| **Dispatcher** | "Triggers computation after verifying the arrival of all data" — the dataflow firing rule |
| **Reservation Station** | "Analogous to the general-purpose register state of instruction set processors" |
| **MMU / TLB** | "translate virtual addresses for local memory operations" |
| **MEP** (Memory Entry Point) | "Also contained on compute blocks and act like memory access instructions" |

The Next Platform adds: "Like regular CPUs, the Maverick ICA uses memory management units and a table
lookaside buffer, but these are used sparingly and only when an ALU calls for specific data. **There is no
speculation or prediction, just fetching.**" And: "It is this combination of reservation station,
dispatcher, and dataflow compute block that NextSilicon has patented."

Note the architectural consequence: there is **no conventional architected register file**. The
reservation station plays that role, and its capacity and organization are not disclosed. Address
translation is **distributed** — every compute block carries its own MMU and TLB.

### "Mill core" — critical clarification

A mill core is **not a physical core**. It is a *software-defined, dynamically instantiated projection* of
a compiled dataflow subgraph onto ALUs. The Next Platform, verbatim: "When an application is compiled for
the dataflow engine, it is literally mapped onto it, into something called a mill core (it looks like a
graph). It looks like the intermediate representation graph of a program before it is compiled, and it is
laid down on the ALUs. **Many mill cores can be laid down on the same compute block, Tetris style, and the
mill cores can be loaded up and deleted as needed in a matter of nanoseconds.**"

The vendor FAQ calls the same object a "software-defined core". The grid is therefore a grid of **compute
blocks**; mill cores are the transient software objects projected onto them. How many mill cores fit per
compute block, their maximum graph size, and the measured reconfiguration latency are all **not
disclosed** (the "nanoseconds" figure is a vendor marketing claim).

### Threading and latency tolerance

Tayari: "a typical CPU has two threads, a GPU has between 32 and 64 threads, but a mill core can support
hundreds of threads at once." TNP extrapolates "maybe tens of mill cores per compute block and 224 compute
blocks… easily up to thousands of threads" — *third-party extrapolation, not a spec*.

The mechanism is disclosed in patent **US11875153B1** ("Executing concurrent threads on a reconfigurable
processing grid"): when a thread hits an "inconsistent-latency operation" (e.g. a memory access), its
runtime context is spilled into a **context storage table** (organized as rows × columns) so the same
logical elements can immediately be reused by another thread; the first thread resumes later. This is the
dataflow analogue of GPU warp switching, implemented by a context table rather than a warp scheduler.
*The patent is not tied to Maverick-2 silicon and the table dimensions are not a Maverick-2 spec.*

Kokkos reports `concurrency() = 65536` for the NextSilicon execution space, but the source marks this
**FIXME placeholder** — it is **not** a hardware thread-capacity figure.

### Fallback tiers

Code that does not suit dataflow drops to (a) the on-die RISC-V E-cores, then (b) the external x86 host
CPU. TNP reports that "~80 percent of instruction runtime" is offloaded to the Maverick ALU blocks, the
remainder running on E-cores or the host *(vendor claim)*. The vendor's own three-stage flow diagram plots
a branch-likelihood histogram partitioned into three labelled regions — **GRID**, **RISC-V E-CORES**,
**HOST CPU** — so the tri-tier split is the vendor's own framing, not an interpretation.

E-core microarchitecture (pipeline, ISA extensions, clock, cache) is **not disclosed**; they are described
only as Arbel-lineage control processors.

---

## 3. Memory Hierarchy

This is genuinely a hybrid, and public sources are in partial tension. Each tier is presented with its
evidence rather than reconciled into a single claim.

| Tier | Capacity / BW | Management | Evidence |
|------|---------------|------------|----------|
| Reservation station (per compute block) | not disclosed | hardware, dataflow-triggered | vendor block diagram: "analogous to the general-purpose register state" |
| Distributed on-chip SRAM (per tile/block) | not disclosed individually | **software-managed; explicit DMA; NOT coherent** | Chips and Cheese (CEO interview) |
| Chip-level cache ("L1 cache" / "Cache Coherence") | 128 MB single die / 256 MB dual die | **hardware cache**, labelled coherent by vendor | vendor spec table; vendor product page |
| HBM3E | 96 GB @ 3.2 TB/s / 192 GB @ 6.4 TB/s (up to 288 GB flavors) | hardware-managed, paged | vendor spec table; Chips and Cheese |
| Host DRAM ↔ device HBM | — | **UVM with exception-based demand page migration** | upstream Kokkos source |

### The software-managed part

Elad Raz on Chips and Cheese states the tile-local SRAM is **not cache coherent** and that moving data
between tiles requires explicit DMA — "you need to do a DMA and move data around" — describing it as
CUDA-shared-memory-like ("there is a notion of shared memory… localized memory"). *(confirmed — CEO,
third-party venue)*

### The hardware-managed part

The 128/256 MB level is presented by the vendor as a cache, and the vendor product page's label for the
row is literally "Cache Coherence". NextSilicon holds multiple cache-coherency patents: US20190042427A1
("Reconfigurable cache architecture and methods for cache coherency"), US12505046B1 ("dynamic
cluster-based cache coherency for multi-core"), US12130736B2 ("sharing a cache line between non-contiguous
memory areas"). The Projection Viewer patent US11995419B1 describes visualizing "cache bins" and HBM
allocation. A hardware cache tier therefore certainly exists — but its **coherence domain and protocol are
not publicly specified**, and the boundary between the coherent and non-coherent regions is undefined in
public sources.

### The UVM part — the best primary evidence on this chip

Upstream Kokkos `Kokkos_NextSiliconSpace.cpp` documents Maverick-2's paging behaviour directly, in a
non-vendor-controlled repository:

- Host and device **share one virtual address space**. `NextSiliconSharedSpace::allocate` is a plain
  `std::aligned_alloc` on the host heap — **no explicit device allocation call exists**.
- "The **NextSilicon UVM migration runtime** chooses a page size based on the alignment of the allocation."
- Migration is **fault/exception-driven**: "fewer/larger pages improves **exception-based page
  migration**"; "fewer/larger pages reduces the number of required TLB entries."
- Maverick-2's **complete supported page-size list is enumerated in code**: 4 KiB, 16 KiB, 64 KiB,
  256 KiB, 1 MiB, 4 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB, 4 GiB, **16 GiB**. This is a hard MMU spec
  available nowhere else.
- Pages can be **pinned** to host or device:
  `nextapi_mem_migrate(ptr, size, NEXTAPI_PAGE_LOC_HOST, true)`, with a
  `PageLocation { Host, Device, Any }` enum. Kokkos pins its thread-local flags to host "to prevent
  migration to device memory."
- Stack migration is a live hazard the backend works around: "Clone the driver to prevent the stack from
  getting migrated to device."

### Synthesis

At the **allocation** level, memory is hardware-managed and transparently migrated (UVM + demand paging +
a large chip-level cache); at the **tile** level, the SRAM local to a compute block is software-managed,
explicitly DMA'd, and non-coherent. Address translation is distributed across per-block MMUs/TLBs with
MEPs acting as the memory-access "instructions". *(Each component confirmed separately; the synthesis is
inferred.)*

Per-tile SRAM capacity, HBM stack count and per-stack configuration are **not disclosed** — only the
aggregate figures are public.

---

## 4. Interconnect

### On-chip NoC

Custom, and **non-uniform by design**. Raz (Chips and Cheese): there are "NOC barriers… barriers in
between" tiles; crossing them costs "a penalty… measured in latency rather than in throughput"; "you don't
want that one side of the core will communicate to the other side. You want to keep everything localized."
Topology, link width, per-link and bisection bandwidth, and the magnitude of the cross-barrier penalty are
all **not disclosed**. Locality is also the explicit objective of the runtime optimizer's *relocation*
action (see the software-stack investigation).

### Die-to-die (OAM)

The OAM is a 2.5D dual-die package whose two dies present as 64 RISC-V cores / 192 GB / 6.4 TB/s combined.
The die-to-die link technology, width, bandwidth and latency are **not disclosed**, and **no source states
whether the two dies form a single coherent device or two devices in one package**.

### Scale-up

There is **no proprietary chip-to-chip scale-up fabric** analogous to NVLink or Infinity Fabric in any
public source. Multi-device configurations rely on PCIe Gen5 x16 plus the network.

### Scale-out

**2 × 100 GbE** ports on the OAM (1 × 100 GbE on the card), integrated on the device itself — notable,
because on-package Ethernet is unusual for an accelerator. Raz mentions "Infiniband, Ethernet, RDMA" for
system networking.

### Fabric partnership

On **2026-06-22** NextSilicon and **Cornelis Networks** announced joint reference architectures pairing
Maverick-2 with the **CN5000 400 Gbps** fabric, with testing to extend to **CN6000 800 Gbps** in H2 2026.
Physical attachment details are not specified. *(vendor announcement)*

---

## 5. Physical / Packaging

| Attribute | Value |
|-----------|-------|
| Process | TSMC 5 nm |
| Packaging | 2.5D |
| Card SKU | PCIe full-height double-width, single die, air-cooled, 400 W |
| OAM SKU | Dual die, liquid-cooled, 750 W |
| Die size | **not disclosed** |
| Host CPU | "External" per the spec table; NextSilicon's own Arbel RISC-V CPU is intended to fill this role eventually |

---

## 6. Companion CPU: Arbel (context only — not an AI accelerator)

Unveiled alongside Maverick-2. Per the vendor page: 64 performance cores, 3.4 GHz, 10-wide pipeline,
480-entry ROB, 3 × 256-bit vector units, RVA23 (Hypervisor) plus custom extensions. Claimed
2.6 SPECint2017/GHz and 3.4 SPECfp2017/GHz against a competitor's 2.3/3.0 — carrying the explicit vendor
disclaimer that all figures "are estimates and **have not been reviewed or endorsed by SPEC** or any third
party."

The Next Platform reports the test chip's cache hierarchy as 64 KB L1I + 64 KB L1D, 1 MB L2 and 2 MB L3
per core, with 6 integer ALUs and **four 128-bit FPUs** — this **conflicts** with the vendor page's
"3 × 256-bit vector units". The vendor's own 2026-06-10 Arbel productization post also says "four 128-bit
vector units". **Unresolved inconsistency; do not record a single value.**

Architectural link to Maverick-2: that post states Arbel was "**originally designed as the control
processor within Maverick-2**, where it handles the serial logic and data movement that the dataflow engine
cannot parallelize" — i.e. the 32 on-die E-cores are Arbel-lineage.

Status: test chip on TSMC 5 nm, in "customer and partner silicon evaluations"; 64- and 128-core production
parts targeted for **Q1 2028** on an unnamed "advanced" node. **Arbel should not be entered into a compute
registry of AI accelerators on its own.**

---

## 7. Performance — all vendor-supplied, all HPC

Every published result is HPC. Baselines are **never named** ("leading GPUs", "GPU", "CPU"), and no result
has been independently reproduced.

| Benchmark | Maverick-2 | "GPU" | "CPU" |
|-----------|-----------|-------|-------|
| GUPS | 32.6 (at 460 W) | 5.6 | 1.4 |
| HPCG | 0.8 GFLOPS/W (600 GFLOPS at 750 W) | 0.428 | 0.21 |
| STREAM | 5.2 TB/s achieved = 83.9% of 6.2–6.4 TB/s peak | — | — |
| PageRank | ~40 GSTEPS, scales to 40 GB graphs; "10× better" | falls off past 25 GB | — |

Umbrella marketing claims: "up to 10× the performance of leading GPUs while consuming as much as 60% less
power"; "4× greater performance-per-watt." The Next Platform's own caution on the STREAM figure: "we do
not believe this is possible. It could be close, but it can't be perfect."

**There is no published AI/ML result of any kind** — no MLPerf, no LLM throughput, no training or inference
number, no PyTorch benchmark. *(verified absent across all retrieved sources)*

---

## 8. Deployments

### Sandia National Laboratories — "Spectra" (primary, most credible)

Second platform in Sandia's **Vanguard** programme (NNSA Advanced Simulation and Computing; consortium with
LLNL and LANL; the first platform was Astra, 2018).

| Attribute | Value |
|-----------|-------|
| Compute nodes | 64 |
| Accelerators | 2 × Maverick-2 dual-die OAM per node = **128 accelerators** |
| Integrator | Penguin Solutions (Tundra infrastructure) |
| Cooling | Chilldyne negative-pressure liquid cooling |
| Day-one applications | HPCG, LAMMPS, SPARTA |
| Mission | Advanced fluid-dynamics simulation for nuclear deterrence assessment — **not AI** |
| Deployed | January 2026 (Sandia Lab News, 2026-01-29) |
| Full system acceptance | **2026-05-18** (vendor post) |
| Host CPU, node fabric, storage, system software, total power | **not disclosed** |

Vanguard programme lead James H. Laros III: "The Vanguard program exists to put new architectures through
rigorous evaluation against workloads that are directly relevant to our mission." This is a **first-of-kind
evaluation testbed, not deployment at scale**.

> *Naming note:* the vendor FAQ and launch blog call this system "**Vanguard-II**". Sandia's own name is
> **Spectra**, the second Vanguard *platform*. Use Spectra; record Vanguard-II only as the vendor's
> (inaccurate) shorthand.

### ODISSEE (EU)

Online Data Intensive Solutions for Science in the Exabytes Era — a Horizon Europe consortium including
CERN openlab and SKA-related partners. NextSilicon "delivered two servers with four Maverick-2 cards" for
training and development (vendor post, 2026-02-17). Small evaluation deployment; vendor-sourced only.

### Unverified

"Dozens of customer sites worldwide" and "available in product volumes now in Q4 2025" are **uncorroborated
vendor marketing** — no independent source names any customer other than Sandia and the ODISSEE consortium.
Unit shipment volumes are not disclosed.

Recognition: HPCwire Readers' Choice 2025 — Best HPC Server Product or Technology; Top New Products or
Technologies to Watch.

---

## 9. Explicitly Not Disclosed

Recorded so that future passes do not re-derive or invent these:

- Die area (mm²) for the Maverick-2 die
- ALUs per compute block; total ALU count per die or package
- FPUs per compute block, and whether every ALU has an associated FPU
- Per-compute-block / per-tile SRAM capacity (only the 128/256 MB aggregate is public)
- Reservation-station capacity and organization
- Coherence domain and protocol of the 128/256 MB cache level
- NoC topology, link width, per-link and bisection bandwidth, cross-barrier latency penalty
- Die-to-die interconnect on the OAM: technology, width, bandwidth, latency, coherence
- Number of HBM3E stacks and per-stack configuration
- Whether any scale-up chip-to-chip fabric exists beyond PCIe Gen5 x16 and on-package 100 GbE
- BF16, FP8, INT8, INT4 and structured-sparsity support — absent from the published table; no source
  states whether the hardware supports them at all
- Mill-core capacity limits, maximum graph size, measured reconfiguration latency
- Thread/context capacity (Kokkos' 65536 is a FIXME placeholder)
- E-core microarchitecture
- Grid configuration encoding — there is no published ISA or bitstream format
- Spectra host CPU, node interconnect, storage, system software stack, total system power
- Maverick-3 or any successor — no roadmap part is named in any retrieved source

---

## Sources

- [Maverick-2 product page](https://www.nextsilicon.com/maverick)
- [NextSilicon Tech page](https://www.nextsilicon.com/tech/)
- [NextSilicon FAQ](https://www.nextsilicon.com/faq)
- [Arbel RISC-V CPU page](https://www.nextsilicon.com/risc-v/)
- [Launch press release, 2025-10-22 (Business Wire)](https://www.businesswire.com/news/home/20251022712360/en/)
- [The Next Platform launch deep-dive](https://www.nextplatform.com/compute/2025/10/22/nextsilicon-takes-aim-at-cpus-and-gpus-with-maverick-2-dataflow-engine/1639749)
- [Vendor spec-table figure (via The Next Platform)](https://image.nextplatform.com/214601.webp?imageId=214601&width=1412&height=1200&format=jpg)
- [Compute-block block diagram figure](https://image.nextplatform.com/214597.webp?imageId=214597&width=1412&height=978&format=jpg)
- [Maverick-2 die shot](https://image.nextplatform.com/214596.webp?imageId=214596&width=1412&height=1230&format=jpg)
- [Vendor benchmark figure](https://image.nextplatform.com/214599.webp?imageId=214599&width=1412&height=874&format=jpg)
- [Three-stage flow figure](https://image.nextplatform.com/214600.webp?imageId=214600&width=1412&height=598&format=jpg)
- [Chips and Cheese — "NextSilicon: Putting HPC First"](https://chipsandcheese.com/p/nextsilicon-putting-hpc-first)
- [Sandia Lab News — Spectra (2026-01-29)](https://www.sandia.gov/labnews/2026/01/29/not-the-largest-supercomputer-but-maybe-the-most-interesting/)
- [Spectra full system acceptance (2026-05-18)](https://www.nextsilicon.com/insights/spectra-supercomputer-at-sandia-achieves-full-system-acceptance/)
- [ODISSEE consortium post (2026-02-17)](https://www.nextsilicon.com/insights/elads-blog-odissee-annual-consortium-meeting-cern/)
- [Cornelis Networks partnership (2026-06-22)](https://www.nextsilicon.com/insights/cornelis-nextSilicon-to-build-joint-reference-architectures-for-ai-and-hpc/)
- [Arbel productization post (2026-06-10)](https://www.nextsilicon.com/insights/nextsilicon-productize-arbel-risc-v-core-into-64core-enterprise-processor-for-ai-hpc/)
- [Kokkos_NextSiliconSpace.cpp (UVM page sizes)](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSiliconSpace.cpp)
- [Patent WO2019055675A1 — grid dataflow architecture](https://patents.google.com/patent/WO2019055675A1/en)
- [Patent US11875153B1 — concurrent threads on a reconfigurable grid](https://patents.google.com/patent/US11875153B1/en)
- [Patent US20190042427A1 — reconfigurable cache / coherency](https://patents.google.com/patent/US20190042427A1/en)
- [Patent US12056376B2 — interconnected memory grid with bypassable units](https://patents.google.com/patent/US12056376B2/en)
- [Hackaday commentary (2026-02-20)](https://hackaday.com/2026/02/20/nextsilicons-maverick-2-the-future-of-high-performance-computing/)
