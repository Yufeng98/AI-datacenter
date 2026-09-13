# IBM Spyre Accelerator — Hardware Architecture

*as_of: 2026-08-08*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*
*Representative product: IBM Spyre Accelerator (single-slot PCIe card, 5 nm, 32 AI cores, 128 GB LPDDR5, 75 W)*
*Host systems: IBM z17, IBM LinuxONE 5, IBM Power11*

---

## Overview

The IBM Spyre Accelerator is a **discrete inference ASIC** that plugs into IBM's own enterprise servers. One
part number, one configuration: 32 active AI cores at 5 nm, 128 GB of LPDDR5 on the card, 75 W in a
single PCIe slot with no auxiliary power connector. There is no edge variant, no training variant, and no
merchant-market SKU.

The defining architectural characteristic is **total compiler ownership of the memory hierarchy**. Spyre has
no hardware cache in the compute path — IBM's compiler documentation states this without hedging — so every
byte that reaches a compute unit was placed there by a load instruction the compiler emitted, in 128-byte
"stick" granules, into a 2 MB per-core scratchpad that has **no eviction mechanism at all**. Work division
across the 32 cores, scratchpad residency and transfer scheduling are decided at compile time. The stated
payoff is **deterministic execution latency**, in contrast to GPU scheduling jitter and cache effects — the
right trade for regulated transactional workloads where tail latency matters more than peak throughput.

**Lineage.** Spyre descends from the IBM Research AIU line: a 14 nm half-core prototype (2018), a 14 nm full
core with the two-corelet structure (2020), a 7 nm four-core chip (ISSCC 2021), and the 2022 32-core, 23 B
transistor **IBM AIU** research chip that IBM described as "the scaled version of an already proven AI
accelerator built into our Telum chip." A 5 nm 32-core SoC with full power management was previewed at
Hot Chips 2024; the product was announced 2025-10-07 and detailed at ISSCC 2026.

**Do not conflate with Telum II's on-die AI unit.** Telum II's zAIU is integrated into the mainframe CPU and
reached by the NNPA instruction, served by zDNN and the IBM Z Deep Learning Compiler. Spyre is a separate
discrete ASIC on a separate software path.

---

## 1. Compute Engine

### Core organization

| Component | Specification |
|-----------|---------------|
| AI cores | **32 active**, 34 physical (2 spares for yield), 8×4 grid |
| Corelets per core | 2, sharing one 2 MB LX scratchpad |
| Matrix unit per corelet | 2D **8×8 SIMD-systolic PE array** = 64 "low-precision math engines"; the **PT** (Processing Tensor) execution unit |
| Vector unit per corelet | 1D vector array / **SFU** (Special Function Unit); IBM's blog states two 1D vector arrays per corelet |
| Total math engines | 4,096 (32 × 2 × 64) — arithmetic, not a published figure |
| Clock frequency | **not disclosed** |
| Register files / PE-local storage | **not disclosed** |

The PT array handles matmul, batched matmul, convolution and fused epilogues. The SFU handles non-linear
activations (GELU, softmax) and element-wise operations — the work that does not map onto a systolic array.

### Datatypes

| Unit | Formats |
|------|---------|
| 2D SIMD-systolic array | fp16, fp8, int8, int4 |
| 1D vector / SFU | adds **fp32** for activations and normalization |
| Compiler-visible (SuperDSC) | `DL16` (IBM DLFloat16), `FP32`, `FP8`, `INT8`, `INT4` |

### Throughput

| Metric | Value | Source class |
|--------|-------|--------------|
| Peak, IBM primary | **">300 TOPS per card, while consuming just 75 W"** | IBM primary |
| fp16 | 98 TOPS | secondary, ISSCC-attributed |
| fp8 | 157 TOPS | secondary, ISSCC-attributed |
| int8 | 315 TOPS (4.2 TOPS/W) | secondary, ISSCC-attributed |
| int4 | 629 TOPS | secondary, ISSCC-attributed |

**IBM has never published a per-datatype breakdown.** The four rows above appear only in secondary technical
coverage attributed to ISSCC 2026. Cite ">300 TOPS" when an IBM-sourced number is required.

### Operation library

In the absence of a published ISA, the practical definition of what the hardware can do is the SuperDSC op
catalog: **70+ `OpFunc` primitives** — BatchMatMul and Conv2D in FP8/FP16/INT4/INT8, broadcast add/sub/mul,
BatchNorm/LayerNorm, GELU/ReLU/Sigmoid/Exp/Log, sum/max/min/mean/absmax reductions, average and max pooling,
depthwise convolution, quantization/CSQ conversions, and TopK.

---

## 2. Data Path and Execution Model

Spyre is a **statically scheduled dataflow** machine. IBM's rule: "An operation is eligible to execute as
soon as all of its input operands are available." There is no runtime dispatcher, no out-of-order execution,
no warp scheduler, and no hardware cache. Everything is decided by the compiler.

```
Host
  ↕ PCIe — DMA / RDMA, transfers described by DCI (loop ranges, strides, dtype)
LPDDR5 device memory   128 GB · ~204 GB/s · runtime-allocated
  ↕ compiler-emitted load/store, 128-byte sticks
LX scratchpad          2 MB per core · ~1.6 MB usable · compiler-planned · no eviction
  ↕
PT (8×8 SIMD-systolic array)   +   SFU (1D vector / special functions)
  ↕
LX scratchpad → LPDDR5 → Host
```

- **SPMD across cores.** "Cores follow a common program structure but operate on different tiles", selected
  by core ID. Splits are baked into the binary as index coefficients.
- **Explicit synchronization.** Multi-input operations (concat, residual add) are synchronization points in
  the static schedule.
- **Program correction.** For kernels with symbolic addresses or dynamic dimensions, a host callback patches
  loop counts and address references into the device binary immediately before launch.
- **Programmer-visible constraints.** Static shapes required (dynamic shapes are WIP); default device dtype
  fp16; int64 silently downcasts to int32; inner dimensions must be 128 B aligned.
- **Virtualization.** PF and VF (SR-IOV) modes via `FLEX_DEVICE`. In VF mode physical addresses are
  unavailable, so memory is referenced by `region_id` plus a 128 B-aligned offset resolved through firmware.

---

## 3. On-chip Memory

| Level | Type | Capacity | Managed by | Notes |
|-------|------|----------|-----------|-------|
| LX scratchpad | SRAM | 2 MB per core (~1.6 MB usable), **64 MB total** across 32 cores | **Compiler** (Inductor LX planning pass) | Shared by the core's two corelets. **No eviction policy** |
| Inner scratchpad level | SRAM | **not disclosed** | Compiler | IBM confirms a "2-level programmable SRAM scratchpad microarchitecture" per core but sizes only the LX level |
| Register files / PE-local | — | **not disclosed** | — | — |
| Hardware cache | — | **none** | — | Explicitly absent |

IBM's compiler documentation, verbatim:

> "There is no hardware cache. The compiler decides which tensors reside in LX at each point in the
> computation and emits explicit load/store instructions to move data."

> "There is no mechanism to move a buffer to HBM and reload it later. This is deliberate."

The 64 MB total is arithmetic (32 × 2 MB), not an IBM-published figure. The bandwidth between LX and the
PT/SFU units is not disclosed — only the 128 B stick transfer granularity is public.

This is the strongest form of scratchpad management in the survey. Cambricon MLU, Tenstorrent Tensix and
Sophgo's TPU line all rely on compiler-managed SRAM, but Spyre removes the spill path entirely: if the
compiler cannot fit a buffer in LX, that buffer lives in device DRAM for the whole computation. A
"core-division mismatch" between adjacent operations is enough to disqualify a buffer from LX.

---

## 4. Off-chip Memory

| Spec | Value |
|------|-------|
| Type | LPDDR5 (**not** HBM) |
| Capacity | **128 GB per card** |
| Channels | **16 @ 6.4 Gbps** |
| Peak bandwidth | **~204 GB/s** |
| Physical organization | 8 dual-channel LPDDR5 modules on the PCIe card (not packaged on the SoC) |
| ECC | SECDED on all DRAM (*secondary source*) |
| Management | Runtime allocator (`SpyreAllocator` / `FlexAllocator`); observable at runtime only |
| Other capacities | **not disclosed** |

### Sticks — the transfer quantum

| Property | Value |
|----------|-------|
| Size | **128 bytes**, aligned |
| Elements | 64 fp16 |
| Compiler constant | `BYTES_IN_STICK = 128` |
| IBM rationale | "matches the natural bandwidth between LPDDR5 device memory and the per-core LX scratchpad, enabling full-stick transfers in single operations" |
| Heritage | Direct inheritance of the zAIU/zDNN "stickified tensor" layout on Telum |

### Address-space limits

| Limit | Value | Enforced by |
|-------|-------|-------------|
| Per-core contiguous device-memory span | **256 MB** | Compiler "span reduction" pass |
| Job virtual address space | 128 GB in **8 segments of ≤16 GB**; SegmentId=7 reserved for SpyreCode | Runtime |
| Address alignment | 128 B | Runtime / compiler |

Naming note: the SuperDSC JSON uses the field name `HBM` for device memory as legacy nomenclature. Spyre
uses LPDDR5.

At ~204 GB/s, Spyre's memory bandwidth is roughly an order of magnitude below HBM-class accelerators
(H100 ≈ 3.35 TB/s, MTIA 300 ≈ 6.1 TB/s) and comparable to LPDDR5-era MTIA v2 (204.8 GB/s). The design
compensates with capacity (128 GB per card at 75 W) and with a compiler that plans every transfer, rather
than with raw bandwidth.

---

## 5. On-chip Interconnect

| Spec | Value |
|------|-------|
| Topology | **Bidirectional ring** connecting all 32 active cores |
| Width | **128 B per cycle per direction** |
| Aggregate bandwidth (GB/s) | **not disclosed** — cannot be derived without the clock frequency |

The ring is not just a hardware fact; the compiler targets it directly. For K-split matmul reductions the
backend applies a "codegen-side core-ID permutation that places collaborating cores on adjacent ring
positions, reducing hop counts from m×n to 1." This is one of the clearest cases in the survey of a compiler
pass written against a specific NoC topology.

---

## 6. Host Interface and Package

| Spec | Value | Source class |
|------|-------|--------------|
| Form factor | Single-slot PCIe card, **no auxiliary power connector** | IBM primary |
| Host link | **PCIe Gen5 ×16, 64 GB/s** | **secondary only** — IBM primary says only "PCIe card" plus "fully pipelined DMA/RDMA support" |
| DMA description | DCI (Data Conversion Information): loop ranges, strides, dtype | IBM primary |
| Card memory | 128 GB LPDDR5 | IBM primary |
| TDP | 75 W | IBM primary |

---

## 7. Scale-up Interconnect

**There is no proprietary chip-to-chip fabric.** Spyre cards communicate over a **standard PCIe switch
fabric** using **direct card-to-card RDMA that bypasses the host CPU**. The ISSCC 2026 abstract phrases it
as "scales over a standard PCIE fabric."

| Scope | Value | Source class |
|-------|-------|--------------|
| Fabric | Standard PCIe switch fabric, direct card-to-card RDMA | IBM primary |
| Card-to-card bandwidth | 64 GB/s, CRC-protected | **secondary only** |
| Chassis limit — IBM z17 / LinuxONE 5 | **48 cards** ≈ 6.1 TB accelerator memory = 1,536 AI cores | IBM primary (blog + newsroom) |
| Chassis limit — IBM Power11 | **16 cards** ≈ 2 TB = 512 AI cores | IBM primary |
| Single-model ensemble limit | **8 cards / ~1 TB** — IBM Docs and torch-spyre docs both say "ensembles of up to eight cards delivering 1 TB memory"; `spyreccl` supports up to 8 cards | IBM primary |
| ISSCC wording | "scales up to 4 or more devices for large generative models" | IBM primary |

**Two figures to handle carefully:**

- **Stale.** "8 cards / 256 cores" is IBM's **August 2024 Hot Chips preview** figure for one I/O drawer, not
  the shipping product. It is widely repeated in secondary coverage. Do not cite it as a current spec.
- **Conflicting.** One secondary source states "48 cards per tray, 192 per system" for z17 — but its own
  memory total (6.1 TB) corresponds to 48 cards × 128 GB, and both IBM primary sources say 48. **Use 48.**

The gap between the **48-card chassis limit** and the **8-card single-model ensemble limit** is
**undocumented**. Whether a single model can span more than 8 cards on a shipping system is not stated
anywhere public.

---

## 8. Scale-out Interconnect

**Not supported.** Spyre Comms is explicitly **on-node (single server) only**; multi-node is listed as a
possible future addition with no roadmap date. There is no Ethernet/RoCE integration on the accelerator, no
collective offload engine, and no cross-server topology. This is a per-server inference part.

---

## 9. Physical and Electrical

| Spec | Value | Source class |
|------|-------|--------------|
| Process | **5 nm CMOS** (standard node) | IBM primary |
| Transistors | **25.6 billion** | IBM primary |
| Die size | **330 mm²** | **secondary only**, ISSCC-attributed |
| Interconnect wire length | "14 miles" | IBM primary |
| Standard-cell library | **7T** chosen over denser 6T (6T needed extra buffers at 0.55 V): 9% synthesis-frequency reduction → 7.5% power saving; re-synthesis → +8% power, +6% area | secondary, ISSCC-attributed |
| Voltage domain — AI core array | **0.55 V** (high activity) | IBM primary |
| Voltage domain — logic/SRAM/IP | **0.75 V** (timing-critical) | IBM primary |
| Power management | **Dual-loop controller**: fast inner loop absorbs peak-current spikes; slower software-controlled outer loop adjusts the average-current target | IBM primary |
| Power-management benefit | 25% higher inference throughput vs. single-loop | secondary, ISSCC-attributed — **vendor claim** |
| TDP | **75 W** | IBM primary |
| Idle power / thermal solution / sustained-vs-peak | **not disclosed** | — |

The 7T-over-6T cell choice and the dual-loop controller are the substance of the ISSCC 2026 circuits paper.
They are also the reason a 32-core 5 nm accelerator fits in a bus-powered 75 W slot at all: the AI array runs
at 0.55 V, which is only viable if the cell library tolerates it without buffer insertion, and the current
transients that a 4,096-engine systolic array produces at that voltage are what the fast inner loop exists to
absorb.

---

## 10. Vendor Performance Claims

All internal testing. **No independent benchmark exists for Spyre — MLPerf or otherwise.**

| Claim | Source | Note |
|-------|--------|------|
| "2-to-3× better power/performance than GPUs on encoder-class models" | ISSCC 2026 abstract | No baseline GPU or configuration disclosed |
| ">8 million documents/hour" ingestion for knowledge-base builds at prompt size 128 | IBM Newsroom | 1M-unit dataset, batch size 128, single card, internal testing |
| "Near-linear scaling up to 8 cards" | ISSCC 2026 via secondary | — |
| 4.2 TOPS/W at int8 | secondary / ISSCC-attributed | Not an IBM-published figure |

---

## 11. Not Disclosed

- SoC clock frequency — and therefore aggregate ring NoC bandwidth in GB/s
- Sizes of the two levels of the "2-level programmable SRAM scratchpad microarchitecture"
- Register-file capacity / SIMD width of each of the 64 math engines per corelet
- Bandwidth between the LX scratchpad and the PT/SFU execution units
- IBM primary confirmation of die size, PCIe generation/lane count, and card-to-card RDMA bandwidth
- Per-datatype peak throughput from an IBM primary source
- Total on-chip SRAM as a published figure (64 MB is arithmetic)
- The Spyre ISA — no instruction set, encoding, assembler or `init.bin` format is published
- Idle power, thermal solution, sustained-vs-peak power behaviour
- Whether card memory configurations other than 128 GB LPDDR5 exist
- IBM feature codes and list price
- Named external production customers or at-scale deployment figures
- Any training capability, or whether the hardware could support it
- How many cards a single model can span beyond 8

---

## Sources

- [IBM Research — Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research — Building the IBM Spyre Accelerator](https://research.ibm.com/blog/building-the-ibm-spyre-accelerator)
- [IBM Research — Spyre for Z (Hot Chips 2024 preview)](https://research.ibm.com/blog/spyre-for-z)
- [IBM Research — IBM Artificial Intelligence Unit (AIU)](https://research.ibm.com/blog/ibm-artificial-intelligence-unit-aiu)
- [ISSCC 2026 — Spyre: An Inference-Optimized Scalable AI Accelerator for Enterprise Workloads](https://research.ibm.com/publications/spyre-an-inference-optimized-scalable-ai-accelerator-for-enterprise-workloads)
- [DBLP — ISSCC 2026 record, DOI 10.1109/ISSCC49663.2026.11409090](https://dblp.org/rec/conf/isscc/CohenKVSVZCRSGWLHKMNSHG26)
- [IBM Newsroom — Spyre commercial availability](https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability)
- [IBM Docs — Introduction to the Spyre Accelerator (9175-ME1)](https://www.ibm.com/docs/en/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator)
- [torch-spyre docs — IBM Spyre device](https://torch-spyre.readthedocs.io/en/latest/architecture/spyre_accelerator.html)
- [torch-spyre docs — Dataflow accelerator architecture](https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html)
- [torch-spyre docs — Scratchpad (LX) planning](https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html)
- [torch-spyre docs — Work-division planning](https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html)
- [torch-spyre docs — Glossary](https://torch-spyre.readthedocs.io/en/latest/getting_started/glossary.html)
- [SuperDSC Bundle spec](https://github.com/torch-spyre/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md)
- [SpyreCode spec](https://github.com/torch-spyre/interface-specs/blob/main/0277-SpyreCode/0277-SpyreCodeSpec.md)
- [RFC 0099 — Multi-Device (on-node only)](https://github.com/torch-spyre/RFCs/blob/main/0099-MultiDevice/0099-MultiDeviceRFC.md)
- [More Than Moore — IBM's Spyre AI Accelerator Deep Dive (secondary)](https://morethanmoore.substack.com/p/ibms-spyre-ai-accelerator-deep-dive)
- [The Register — Hot Chips 2024 coverage (stale 8-card figure)](https://www.theregister.com/on-prem/2024/08/27/ibm-details-upcoming-chips-to-support-ai-on-mainframes/)
