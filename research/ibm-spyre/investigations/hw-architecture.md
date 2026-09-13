# IBM Spyre Accelerator — Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*

---

## Overview

The **IBM Spyre Accelerator** is a discrete inference ASIC on a single-slot, 75 W, PCIe card carrying 128 GB
of LPDDR5. It is a system option for **IBM z17**, **IBM LinuxONE 5** and **IBM Power11** — not a merchant
part sold into general datacenter racks. It is the productization of the IBM Research **AIU** (Artificial
Intelligence Unit) line, and the AIU name survives everywhere in the software (`torch_sendnn`, `aiu-smi`,
`libaiupti`, the `ibm-aiu` GitHub org); `torch-spyre` itself describes its purpose as the "PyTorch backend
for IBM's Spyre **AIU**".

**Do not conflate Spyre with the Telum II on-die AI unit.** Telum II's zAIU is an accelerator integrated into
the mainframe CPU and reached by the NNPA instruction, served by the zDNN library and the IBM Z Deep
Learning Compiler. Spyre is a separate discrete ASIC with a completely separate software path.

Architecturally the defining property is what Spyre *lacks*: **there is no hardware cache anywhere in the
compute path**. IBM's own compiler documentation states this outright. All data movement between LPDDR5 and
the per-core LX scratchpad is emitted by the compiler as explicit load/store instructions, in 128-byte
"stick" granules, with no eviction mechanism. Everything — work division across the 32 cores, scratchpad
residency, transfer scheduling — is decided at compile time. The result IBM advertises is *deterministic
execution latency*, in contrast to GPU scheduling jitter and cache effects.

**Status: shipping (GA), sold as a priced system option.** Announced 2025-10-07; GA 2025-10-28 on z17 and
LinuxONE 5; GA "early December 2025" on Power11. No named external at-scale production deployment is
public — pre-GA validation is documented only at IBM Yorktown Heights and the University at Albany Center
for Emerging AI Systems. Record as *shipping (GA)*, **not** *deployed at scale*.

---

## 0. Lineage — the IBM AIU line

| Year | Milestone | Confidence |
|------|-----------|------------|
| 2018 | 14 nm half-core prototype (VLSI) | confirmed-secondary |
| 2020 | 14 nm full core, two-corelet structure | confirmed-secondary |
| 2021 | 7 nm four-core chip (ISSCC 2021) | confirmed-secondary |
| 2022 | **IBM AIU** research chip: 32 cores, 23 B transistors, 5 nm; described by IBM as "the scaled version of an already proven AI accelerator built into our Telum chip", with cores that "closely resemble the AI core embedded in the Telum chip" of z16 | confirmed-primary |
| Aug 2024 | 5 nm 32-core SoC with full power management previewed at Hot Chips 2024 | confirmed-primary |
| Oct 2025 | Spyre Accelerator commercially announced (2025-10-07); GA on z17 / LinuxONE 5 (2025-10-28) | confirmed-primary |
| Dec 2025 | GA on IBM Power11 ("early December 2025") | confirmed-primary |
| Feb 2026 | ISSCC 2026 paper presented (pp. 52–54) | confirmed-primary |

**ISSCC 2026 paper.** Cohen, Kar, Venkataramani, Srinivasan, Veraa, Ziegler, Cao, Ranjan, Silberman,
Guillorn, … Chang, *"Spyre: An Inference-Optimized Scalable AI Accelerator for Enterprise Workloads,"*
ISSCC 2026, pp. 52–54, DOI 10.1109/ISSCC49663.2026.11409090. Abstract:

> "Spyre is a scalable, power-efficient AI accelerator product for enterprise workloads. Featuring 32 AI
> cores, mixed-precision support, and LPDDR5 memory, it fits in a single-slot PCIe form factor and scales
> over a standard PCIE fabric. Optimized for inference workloads, Spyre achieves 2-to-3× better
> power/performance than GPUs on encoder-class models and scales up to 4 or more devices for large
> generative models."

This is a **3-page circuits paper, not a full architecture paper**. The deepest public microarchitecture
description is IBM Research's 2026-02-18 blog post plus the IBM-authored `torch-spyre` developer
documentation.

---

## 1. Compute Engine

| Item | Value | Confidence |
|------|-------|------------|
| AI cores | **32 active**, 34 physical (2 spares for yield), laid out as an 8×4 grid | confirmed-primary (grid layout: confirmed-secondary) |
| Per core | 2 **corelets**, sharing one 2 MB LX scratchpad | confirmed-primary |
| Per corelet — matrix unit | 2D **8×8 SIMD-systolic PE array** = "64 low-precision math engines"; exposed in the toolchain as the **PT** ("Processing Tensor") execution unit; handles matmul, bmm, conv and fused epilogues | confirmed-primary |
| Per corelet — vector unit | 1D vector array / **SFU** (Special Function Unit, a.k.a. SFP) for non-linear activations (GELU, softmax) and element-wise ops. IBM's blog states *two* 1D vector arrays per corelet | confirmed-primary |
| Total math engines | 32 cores × 2 corelets × 64 = **4,096 low-precision math engines** | inferred (arithmetic from confirmed values) |
| Datatypes — 2D array | **fp16, fp8, int8, int4** | confirmed-primary |
| Datatypes — 1D vector | adds **fp32** for activations / normalization | confirmed-primary |
| Compiler-visible formats | `DL16` (IBM DLFloat16), `FP32`, `FP8`, `INT8`, `INT4` in the SuperDSC op spec | confirmed-primary |
| Peak throughput (IBM primary) | **">300 TOPS per card, while consuming just 75 W"** | confirmed-primary |
| Peak throughput per datatype | fp16 98 TOPS · fp8 157 TOPS · **int8 315 TOPS (4.2 TOPS/W)** · int4 629 TOPS | confirmed-secondary (ISSCC-attributed; not in any IBM primary source) |
| Clock frequency | **not disclosed** | — |

### Operation library

The SuperDSC operation spec catalogs **70+ `OpFunc` primitives**: BatchMatMul and Conv2D in
FP8/FP16/INT4/INT8, broadcast add/sub/mul, BatchNorm/LayerNorm, GELU/ReLU/Sigmoid/Exp/Log, sum/max/min/
mean/absmax reductions, average and max pooling, depthwise convolution, quantization/CSQ conversions, and
TopK. This is the practical definition of "what the hardware can do" in the absence of a published ISA.

---

## 2. On-chip Memory — software-managed, no hardware cache

This is the architecturally load-bearing fact of the whole design. IBM's compiler documentation states it
without hedging:

> "compiler-emitted load/store instructions to move tiles between LPDDR5 and the LX scratchpad; **there is
> no hardware cache**"

> "There is no hardware cache. The compiler decides which tensors reside in LX at each point in the
> computation and emits explicit load/store instructions to move data."

| Level | Capacity | Management | Confidence |
|-------|----------|-----------|------------|
| **LX scratchpad (on-chip SRAM)** | **2 MB per core**, shared by the core's two corelets → **64 MB total** across 32 cores; ~**1.6 MB usable** per core after backend reservations | Software / compiler. Allocated by the Inductor "LX planning" pass. **No eviction policy** — "There is no mechanism to move a buffer to HBM and reload it later. This is deliberate." | confirmed-primary (64 MB total is arithmetic) |
| Scratchpad microarchitecture | IBM describes a **"2-level programmable SRAM scratchpad microarchitecture"** per core. The split between the two levels — the inner level feeding the PE array vs. the 2 MB LX — is **not disclosed** | software-managed | confirmed-primary for the two-level claim; sizes not disclosed |
| Register files / PE-local storage | **not disclosed** | — | — |
| LX ↔ PT/SFU bandwidth | **not disclosed** (only the 128 B stick granularity is public) | — | — |

---

## 3. Off-chip Memory

| Item | Value | Confidence |
|------|-------|------------|
| Type / capacity | **128 GB LPDDR5 per card** | confirmed-primary |
| Organization | **16 channels @ 6.4 Gbps**; physically 8 dual-channel LPDDR5 modules on the PCIe card (not packaged on the SoC) | confirmed-primary |
| Peak bandwidth | **~204 GB/s** to the cores | confirmed-primary |
| ECC | SECDED on all DRAM | confirmed-secondary |
| Management | Runtime-allocated (`SpyreAllocator` / `FlexAllocator`); observable at runtime only | confirmed-primary |
| Transfer granularity | **"stick" = 128-byte aligned chunk = 64 fp16 elements**; compiler constant `BYTES_IN_STICK = 128`. IBM: this "matches the natural bandwidth between LPDDR5 device memory and the per-core LX scratchpad, enabling full-stick transfers in single operations" | confirmed-primary |
| Per-core addressing limit | **256 MB** maximum contiguous device-memory span addressable by any one core (distinct from the 2 MB LX limit); enforced by the compiler's "span reduction" pass | confirmed-primary |
| Job virtual address space | 128 GB per job, split into **8 segments of ≤16 GB**; SegmentId=7 reserved for SpyreCode (binaries, correction data). Addresses 128 B-aligned | confirmed-primary |
| Other capacities | Whether any card configuration other than 128 GB exists is **not disclosed** | — |

Two naming notes worth recording:

- The "stick" concept is a direct inheritance from IBM's zAIU/zDNN **"stickified tensor"** data layout on
  Telum — the same AIU tensor-layout philosophy, re-exposed here at 128 B granularity.
- The SuperDSC JSON still uses the field name **`HBM`** for device memory as legacy nomenclature. Spyre uses
  LPDDR5, not HBM.

---

## 4. Data Path and Execution Model

Spyre is a **statically scheduled dataflow** engine. IBM's definition: "An operation is eligible to execute
as soon as all of its input operands are available." All work division, data staging and kernel
specification are fixed at compile time; there is no runtime dispatcher, no out-of-order execution, no warp
scheduler, and no hardware caches. IBM's claimed payoff is *deterministic execution latency* versus GPU
scheduling jitter and cache effects.

**Explicit data path:**

```
Host
  ↕ PCIe (DMA / RDMA, DCI-described transfers)
LPDDR5 device memory  (128 GB, ~204 GB/s, runtime-allocated)
  ↕ compiler-emitted load/store, 128 B sticks
LX scratchpad         (2 MB per core, ~1.6 MB usable, compiler-planned, no eviction)
  ↕
PT (8×8 SIMD-systolic array)  +  SFU (1D vector / special functions)
  ↕
LX scratchpad → LPDDR5 → Host
```

Multi-input operations (concat, residual add) are explicit synchronization points in this schedule.

**SPMD across cores.** "Cores follow a common program structure but operate on different tiles", selected by
core ID. Work division is encoded by the compiler as index coefficients, not chosen at run time.

**Program correction / JIT patching.** For kernels with symbolic addresses or dynamic dimensions, a host
callback patches loop counts and address references into the device binary immediately before launch.

**Constraints exposed to the programmer.** Static shapes are required (dynamic shapes are WIP); the default
device dtype is fp16; int64 silently downcasts to int32; inner dimensions must be 128 B aligned.

**Virtualization.** The runtime exposes **PF (physical function)** and **VF (virtual function)** modes via
`FLEX_DEVICE=PF|VF`. In VF mode physical addresses are unavailable, so memory is referenced by `region_id`
plus a 128 B-aligned offset resolved through firmware lookup.

---

## 5. On-chip Interconnect

| Item | Value | Confidence |
|------|-------|------------|
| Topology | **Bi-directional ring** connecting all 32 active cores | confirmed-primary |
| Link width | **128 B per cycle per direction** | confirmed-primary |
| Aggregate bandwidth | **not disclosed** — 128 B/cycle/dir cannot be converted to GB/s without the clock frequency, which IBM has not published | — |

The compiler is ring-topology-aware: for K-split matmul reductions it applies a "codegen-side core-ID
permutation that places collaborating cores on adjacent ring positions, reducing hop counts from m×n to 1."
This is one of the clearest examples in the survey of a compiler pass written directly against a NoC
topology.

---

## 6. Host Interface and Package

| Item | Value | Confidence |
|------|-------|------------|
| Form factor | Single-slot PCIe card, **no auxiliary power connector** | confirmed-primary |
| Host link | **PCIe Gen5 ×16, 64 GB/s** | confirmed-**secondary** — IBM primary sources say only "PCIe card" plus "fully pipelined DMA/RDMA support" |
| DMA | Fully pipelined DMA/RDMA support; transfers described by DCI (Data Conversion Information: loop ranges, strides, dtype) | confirmed-primary |

---

## 7. Scale-up / Scale-out Interconnect

There is **no proprietary chip-to-chip fabric**. Spyre cards scale out over a **standard PCIe switch
fabric** with **direct card-to-card RDMA that bypasses the host CPU**. The ISSCC abstract phrases it as
"scales over a standard PCIE fabric".

| Scope | Value | Confidence |
|-------|-------|------------|
| Card-to-card fabric | Standard PCIe switch fabric, direct card-to-card RDMA (host bypass) | confirmed-primary |
| Card-to-card bandwidth | **64 GB/s, CRC-protected** | confirmed-**secondary** only |
| Chassis limit — IBM z17 / LinuxONE 5 | **up to 48 cards** (≈6.1 TB accelerator memory, = 1,536 AI cores) | confirmed-primary (IBM Research blog and IBM Newsroom both state 48) |
| Chassis limit — IBM Power11 | **up to 16 cards** (≈2 TB, = 512 AI cores) | confirmed-primary |
| Single-model "ensemble" limit | **8 cards / ~1 TB**. IBM Docs and the torch-spyre docs both say "ensembles of up to eight cards delivering 1 TB memory"; the `spyreccl` collective backend likewise supports **up to 8 cards, on-node only**. The ISSCC abstract says Spyre "scales up to 4 or more devices for large generative models" | confirmed-primary |
| Multi-node (cross-server) | **Not supported.** Spyre Comms is explicitly on-node only; multi-node is listed as a possible future addition with no roadmap date | confirmed-primary |

**Two figures to handle carefully:**

- **Stale**: the widely repeated "8 cards / 256 cores" is IBM's **August 2024 Hot Chips preview** (one I/O
  drawer), not the shipping product. Do not cite it as a current spec.
- **Conflicting**: one secondary source states "48 cards per tray, 192 per system" for z17 — but its own
  memory total (6.1 TB) corresponds to 48 cards × 128 GB, and both IBM primary sources say 48. **Use 48.**
- The gap between the 48-card chassis limit and the 8-card single-model ensemble limit is **undocumented**.

---

## 8. Physical and Electrical

| Item | Value | Confidence |
|------|-------|------------|
| Process | **5 nm CMOS** (standard node) | confirmed-primary |
| Transistors | **25.6 billion** | confirmed-primary |
| Die size | **330 mm²** | confirmed-**secondary** only (ISSCC-attributed; no IBM primary confirmation) |
| Interconnect wire length | "14 miles" | confirmed-primary |
| Standard cells | 7T library chosen over denser 6T (6T required extra buffers at 0.55 V); 9% synthesis-frequency reduction → 7.5% power saving; re-synthesis → +8% power, +6% area | confirmed-secondary (ISSCC-attributed) |
| Voltage domains | **0.55 V** for the high-activity AI core array; **0.75 V** for timing-critical logic, SRAM and third-party IP | confirmed-primary |
| Power management | **Dual-loop controller**: a fast inner loop absorbs peak-current spikes; a slower software-controlled outer loop adjusts the average-current target | confirmed-primary |
| Power-management benefit | 25% higher inference throughput vs. a single-loop controller | confirmed-secondary, ISSCC-attributed — **vendor claim** |
| TDP | **75 W** | confirmed-primary |
| Card memory | 128 GB LPDDR5, SECDED ECC | confirmed-primary (ECC: secondary) |
| Idle power / thermal solution / sustained-vs-peak behaviour | **not disclosed** | — |

---

## 9. Vendor Performance Claims

All figures below are **vendor claims from internal testing**. There are **no independent benchmark results
for Spyre — MLPerf or otherwise — in the public record.**

| Claim | Source | Label |
|-------|--------|-------|
| "2-to-3× better power/performance than GPUs on encoder-class models" | ISSCC 2026 abstract | vendor claim, no baseline configuration disclosed |
| ">8 million documents/hour" ingestion for knowledge-base builds at prompt size 128 (internal testing, 1M-unit dataset, batch size 128, single card) | IBM Newsroom | vendor claim |
| "Near-linear scaling up to 8 cards" | ISSCC 2026 via secondary source | vendor claim |
| 4.2 TOPS/W at int8 | secondary / ISSCC-attributed | vendor claim |

---

## 10. Explicitly Not Disclosed

Recorded here so downstream tables do not silently fill these in:

- SoC clock frequency (and therefore aggregate ring NoC bandwidth in GB/s)
- The sizes of the two levels of the "2-level programmable SRAM scratchpad microarchitecture"
- Register-file capacity / SIMD width of each of the 64 math engines per corelet
- Bandwidth between the LX scratchpad and the PT/SFU units
- IBM confirmation of die size (330 mm² is secondary only)
- IBM confirmation of PCIe generation / lane count (Gen5 ×16 is secondary only)
- IBM confirmation of card-to-card RDMA bandwidth (64 GB/s is secondary only)
- Per-datatype peak throughput from an IBM primary source (only ">300 TOPS" is primary)
- Total on-chip SRAM as a published figure (64 MB is arithmetic)
- The Spyre ISA — no instruction set, encoding, assembler, or `init.bin` format is published
- Idle power, thermal solution, sustained-vs-peak power behaviour
- Whether card memory configurations other than 128 GB exist
- IBM feature codes and list price
- Named external production customers or any at-scale deployment figures
- Any training capability — Spyre is positioned as inference-only, and IBM does not state whether the
  hardware could support training

---

## Sources

- [IBM Research — Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research — Building the IBM Spyre Accelerator](https://research.ibm.com/blog/building-the-ibm-spyre-accelerator)
- [IBM Research — Spyre for Z (Hot Chips 2024 preview)](https://research.ibm.com/blog/spyre-for-z)
- [IBM Research — IBM Artificial Intelligence Unit (AIU)](https://research.ibm.com/blog/ibm-artificial-intelligence-unit-aiu)
- [ISSCC 2026 — Spyre: An Inference-Optimized Scalable AI Accelerator for Enterprise Workloads](https://research.ibm.com/publications/spyre-an-inference-optimized-scalable-ai-accelerator-for-enterprise-workloads)
- [DBLP record, DOI 10.1109/ISSCC49663.2026.11409090](https://dblp.org/rec/conf/isscc/CohenKVSVZCRSGWLHKMNSHG26)
- [IBM Newsroom — Spyre commercial availability (2025-10-07)](https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability)
- [IBM Docs — Introduction to the Spyre Accelerator (9175-ME1)](https://www.ibm.com/docs/en/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator)
- [torch-spyre docs — IBM Spyre device](https://torch-spyre.readthedocs.io/en/latest/architecture/spyre_accelerator.html)
- [torch-spyre docs — Dataflow accelerator architecture](https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html)
- [torch-spyre docs — Key concepts](https://torch-spyre.readthedocs.io/en/latest/getting_started/key_concepts.html)
- [torch-spyre docs — Glossary](https://torch-spyre.readthedocs.io/en/latest/getting_started/glossary.html)
- [torch-spyre docs — Scratchpad (LX) planning](https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html)
- [torch-spyre docs — Work-division planning](https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html)
- [SuperDSC Bundle spec](https://github.com/torch-spyre/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md)
- [SpyreCode spec](https://github.com/torch-spyre/interface-specs/blob/main/0277-SpyreCode/0277-SpyreCodeSpec.md)
- [ProgramExecution spec](https://github.com/torch-spyre/interface-specs/blob/main/ProgramExecution/ProgramExecutionSpec.md)
- [RFC 0171 — Spyre Device](https://github.com/torch-spyre/RFCs/blob/main/0171-SpyreDevice/0171-SpyreDeviceRFC.md)
- [More Than Moore — IBM's Spyre AI Accelerator Deep Dive (secondary)](https://morethanmoore.substack.com/p/ibms-spyre-ai-accelerator-deep-dive)
- [The Register — Hot Chips 2024 coverage (stale 8-card figure)](https://www.theregister.com/on-prem/2024/08/27/ibm-details-upcoming-chips-to-support-ai-on-mainframes/)
