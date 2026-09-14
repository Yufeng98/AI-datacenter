# IBM Spyre Accelerator — Hardware and Software Stack Summary

*as_of: 2026-09-13*
*chip: ibm-spyre*
*device_class: Inference Accelerator (SIMD-Systolic Dataflow, scratchpad-managed)*
*Representative product: IBM Spyre Accelerator — single-slot 75 W PCIe card for IBM z17 / LinuxONE 5 / Power11*
*Next-generation preview (pre-announcement, Hot Chips 38, 2026-08-24): a next-gen dual-ISA IBM Z/LinuxONE CPU plus a separate, unnamed "AI Inference Acceleration Chipset" — see the 2026-09-13 update section*

---

## Overview

The **IBM Spyre Accelerator** is a 5 nm inference ASIC on a single-slot, 75 W PCIe card carrying 128 GB of
LPDDR5. It is sold as a **priced system option** for **IBM z17**, **IBM LinuxONE 5** and **IBM Power11** —
not as a merchant part for general datacenter racks. It is the productization of IBM Research's **AIU**
(Artificial Intelligence Unit) line, whose 2022 research chip IBM described as "the scaled version of an
already proven AI accelerator built into our Telum chip." The AIU name survives throughout the software
(`torch_sendnn`, `aiu-smi`, `libaiupti`, the `ibm-aiu` GitHub org).

**Status: shipping (GA).** Announced 2025-10-07; GA 2025-10-28 on z17 and LinuxONE 5; GA "early December
2025" on Power11. **No named external at-scale production deployment is public** — pre-GA validation is
documented only at IBM Yorktown Heights and the University at Albany Center for Emerging AI Systems. This is
genuine product availability, not an eval board, but it is *shipping (GA)*, **not** *deployed at scale*.

**Do not conflate Spyre with the Telum II on-die AI unit.** Telum II's zAIU is a CPU-integrated accelerator
reached via the NNPA instruction and served by zDNN and the IBM Z Deep Learning Compiler. Spyre is a
separate discrete PCIe ASIC with a completely separate software path — and neither zDNN nor zDLC targets it.

Two properties make Spyre interesting for a survey about programming models:

1. **There is no hardware cache anywhere in the compute path.** IBM's own compiler documentation states it
   outright. Data moves between LPDDR5 and a 2 MB per-core LX scratchpad only via compiler-emitted
   load/store instructions, in 128-byte "stick" granules, with **no eviction mechanism** — "This is
   deliberate." Work division across the 32 cores, scratchpad residency, and transfer scheduling are all
   fixed at compile time. IBM's claimed payoff is deterministic execution latency versus GPU scheduling
   jitter and cache effects.
2. **The middle of the software stack is unusually open.** IBM runs a public GitHub org
   (`github.com/torch-spyre`, 14 mostly Apache-2.0 repos) publishing the PyTorch backend, the TorchInductor
   front end, design RFCs, the compiler↔device **interface specifications**, a Triton fork, and — rare
   among accelerator vendors — a **CPU interpreter and latency model for the tile IR** that makes the
   programming model studiable without hardware.

The catch is where the stack closes. The back-end compiler (**DeepTools**), the runtime (**DeepRT** /
**Flex**), the collectives library, the driver, the firmware and the **ISA** are all proprietary, and
`torch_sendnn` is "only available pre-installed in a base environment" — it cannot be downloaded at all.

---

## Hardware Architecture

### Compute

32 **active** AI cores (34 physical — 2 spares for yield) in an 8×4 grid. Each core holds two **corelets**
that share one 2 MB LX scratchpad. Each corelet contains a 2D **8×8 SIMD-systolic PE array** — IBM's "64
low-precision math engines", exposed in the toolchain as the **PT** (Processing Tensor) unit — plus 1D
vector / **SFU** units for non-linear activations and element-wise work. That is **4,096 math engines per
card** (arithmetic, not a published figure).

| Spec | Value |
|---|---|
| Process | 5 nm CMOS |
| Transistors | 25.6 billion |
| AI cores | 32 active (34 physical, 2 yield spares) |
| Per core | 2 corelets sharing 2 MB LX scratchpad |
| Per corelet | 8×8 SIMD-systolic array (64 math engines) + 1D vector/SFU |
| Datatypes (2D array) | fp16, fp8, int8, int4 |
| Datatypes (1D vector) | adds fp32 for activations/normalization |
| Peak throughput (IBM primary) | **">300 TOPS per card at 75 W"** |
| Clock frequency | **not disclosed** |
| Die size | 330 mm² — *secondary source only* |
| TDP | 75 W, no auxiliary power connector |

A per-datatype breakdown (fp16 98 / fp8 157 / int8 315 / int4 629 TOPS, 4.2 TOPS/W at int8) circulates in
secondary technical coverage attributed to ISSCC 2026. **IBM has never published it**; treat it as
unconfirmed.

### Memory — software-managed, no cache

| Level | Capacity | Bandwidth | Managed by |
|---|---|---|---|
| LX scratchpad | 2 MB per core (~1.6 MB usable), 64 MB total (arithmetic) | not disclosed | **Compiler** — no eviction, by design |
| Inner scratchpad level | **not disclosed** (IBM confirms a "2-level programmable SRAM scratchpad microarchitecture" but sizes only the LX level) | not disclosed | Compiler |
| LPDDR5 device memory | 128 GB per card, 16 channels @ 6.4 Gbps, 8 dual-channel modules, SECDED ECC | **~204 GB/s** | Runtime allocator |

Transfers happen in **"sticks"** — 128-byte aligned chunks, 64 fp16 elements, compiler constant
`BYTES_IN_STICK = 128`. IBM says the size "matches the natural bandwidth between LPDDR5 device memory and
the per-core LX scratchpad." The stick is a direct inheritance of the zAIU/zDNN **"stickified tensor"**
layout from Telum — the same AIU tensor philosophy re-exposed at 128 B granularity. Two further limits are
compiler-enforced: a **256 MB** maximum contiguous device-memory span per core, and a job address space of
128 GB in 8 segments of ≤16 GB (SegmentId=7 reserved for the device binary).

Note: the SuperDSC JSON still names device memory `HBM` as legacy nomenclature. Spyre uses LPDDR5.

### Interconnect

On-chip, a **bidirectional ring** joins all 32 cores at **128 B per cycle per direction**. Aggregate
bandwidth in GB/s is **not derivable** — the clock is not disclosed. The compiler is ring-aware: K-split
matmul reductions apply a core-ID permutation that places collaborating cores on adjacent ring positions,
"reducing hop counts from m×n to 1."

Off-chip, **there is no proprietary chip-to-chip fabric**. Cards scale out over a **standard PCIe switch
fabric with direct card-to-card RDMA** that bypasses the host CPU. The ISSCC abstract puts it as "scales
over a standard PCIE fabric."

| Scope | Value |
|---|---|
| Host interface | Single-slot PCIe card; **PCIe Gen5 ×16 / 64 GB/s is secondary-source only** (IBM says only "PCIe card" with "fully pipelined DMA/RDMA support") |
| Card-to-card | Standard PCIe switch fabric + RDMA; 64 GB/s CRC-protected is *secondary only* |
| Chassis limit — z17 / LinuxONE 5 | **48 cards** (≈6.1 TB, 1,536 cores) |
| Chassis limit — Power11 | **16 cards** (≈2 TB, 512 cores) |
| Single-model ensemble | **8 cards / ~1 TB** — the `spyreccl` cap and IBM's own docs |
| Multi-node (cross-server) | **Not supported** — on-node only |

The widely repeated **"8 cards / 256 cores" is stale** — it is IBM's August 2024 Hot Chips preview figure
for one I/O drawer, not the shipping product. Separately, one secondary source claims "48 per tray, 192 per
system" for z17, but its own 6.1 TB memory total corresponds to 48 cards × 128 GB, and both IBM primary
sources say 48. **Use 48.** The gap between the 48-card chassis limit and the 8-card single-model ensemble
limit is undocumented.

### Execution model

Statically scheduled dataflow: "an operation is eligible to execute as soon as all of its input operands are
available." Cores run **SPMD** — a common program structure over different tiles, selected by core ID. There
is **no hardware cache, no out-of-order execution, no warp scheduler and no runtime dispatcher**. Static
shapes are required; default device dtype is fp16; int64 silently downcasts to int32; inner dimensions must
be 128 B aligned. For kernels with symbolic addresses, a host callback performs **program correction** —
JIT-patching loop counts and addresses into the device binary immediately before launch. The runtime
supports **PF and VF (SR-IOV)** modes; in VF mode memory is referenced by `region_id` + 128 B-aligned offset
through firmware lookup.

### Power and packaging

Dual voltage domains — **0.55 V** for the high-activity AI core array, **0.75 V** for timing-critical logic,
SRAM and third-party IP. A **dual-loop power controller** pairs a fast inner loop that absorbs peak-current
spikes with a slower software-controlled outer loop that adjusts the average-current target; the claimed 25%
throughput benefit over a single-loop controller is an ISSCC-attributed vendor claim carried by secondary
coverage. IBM cites "14 miles" of interconnect wiring.

---

## Software Stack

```
PyTorch 2.x  (spyre registered via PrivateUse1; eager + torch.compile)
  → TorchDynamo / AOTAutograd → FX graph (ATen)
  → TorchInductor with out-of-tree Spyre backend
        LoopLevelIR (16 passes: layouts → restickify → coarse tile →
        span reduction → work division across 32 cores → LX planning)
  → OpSpec → SuperDSC JSON bundle  (open interface spec)
        [ Triton fork → TTIR → KTIR — the announced successor path ]
  → DeepTools back-end compiler  (PROPRIETARY, dxp_standalone)
  → SpyreCode container  (open spec: job plans, init.bin, program correction)
  → DeepRT / Flex runtime  (PROPRIETARY)
  → kernel driver + firmware  (PROPRIETARY)
  → Spyre hardware  (ISA not public)
```

### Framework and serving

PyTorch is the primary and effectively the only framework, with `spyre` registered as a first-class device
through **PrivateUse1**. Serving runs through **vLLM**: `sendnn-inference` (the production plugin, formerly
`vllm-project/vllm-spyre`, ~638 commits) and the newer `spyre-inference` built on `torch-spyre`. Supported
vLLM features include chunked prefill, prefix caching, guided decoding, beam search and **tensor
parallelism**; LoRA, speculative decoding, and pipeline/expert/data parallelism are not supported.
`hf-adapters` enables HuggingFace models by **monkey-patching at load time** — replacing only the operations
Spyre cannot run natively (RoPE, RMSNorm, KV-cache management, generation loop) across 30 adapters and 46
verified checkpoints. An early SGLang port exists.

**Negative finding:** there is no public ONNX-based or z/OS-native path to Spyre. zDLC and zDNN target the
Telum/Telum II zAIU via NNPA only. Spyre's public path is PyTorch/vLLM on Linux on Z, Linux on Power, and
Red Hat OpenShift.

### Compiler — the deepest open layer

The Spyre TorchInductor backend registers `SuperDSCScheduling`, `SpyrePythonWrapperCodegen` and
`SpyreDeviceOpOverrides`, and hooks six upstream extension points. Its **16-pass LoopLevelIR pipeline** is
where the hardware model lives: layout propagation and `restickify` insertion, coarse tiling, **span
reduction** (enforcing the 256 MB per-core span), **cost-model matmul division and work distribution across
the 32 cores** (`cost = compute + hbm + psum + shape_penalties + batch`, equal stick counts per core), and
finally **LX scratchpad planning** with four solvers — greedy, first-fit, best-fit, and a **CP-SAT** solver
that models allocation as 2-D non-overlapping rectangles over (lifetime × address) minimizing DRAM traffic.

The compiler↔backend interface is published even though its consumer is not. **SuperDSC** ("Super Design
Space Config") bundles are MLIR + JSON describing per-op memory location (DDR vs LX), core work mapping,
stick configuration, back-gaps and affine "folds" that compactly parameterize all 32 cores. IBM chose JSON
because "SuperDSC artifacts have to be diffable and inspectable during development."

**KTIR** (KernelTile IR, MLIR dialect `ktdp`, RFC 0682 merged March 2026) is the announced successor, and
IBM explicitly positions it as a **community-aligned spec generalizable across dataflow accelerators**. Its
central idea is a three-step separation of concerns: `construct_memory_view` (name a region) →
`construct_access_tile` (which slice this core touches) → `ktdp.load`/`ktdp.store`. It states its
distinctions from the GPU model outright: persistent, compile-time-partitioned cores instead of thread
blocks, and explicit scratchpad management instead of an implicit cache hierarchy. Production still routes
through SuperDSC.

**Triton is the kernel language**, via an IBM fork (MIT). `spyre-kernels` commits the `.ttir` and `.ktir`
artifacts alongside each kernel and validates them in four tiers: T0 numerical equivalence vs. PyTorch on
GPU, T1 shape compliance (*tiles fit the scratchpad, grid fits 32 cores*), T2 correctness on the `ktir_cpu`
simulator **and** real hardware, T3 expert review.

### Tensor layout

**Stickification** is "the transformation from a host-strided PyTorch layout to a tiled Spyre device
layout." A `(1024, 256)` fp16 tensor physically becomes `(4, 1024, 64)` on device — a layout **not
expressible with PyTorch strides**, which is why `SpyreTensorImpl` and `SpyreTensorLayout` exist alongside
the standard size/stride metadata, and why `restickify` passes are needed wherever adjacent operations
disagree on tile structure.

### Runtime, driver, ISA

`torch_spyre` (open) provides the allocator, device guard, `SpyreStream` (FIFO within a stream, no
cross-stream ordering, **sticky error model** — a failed stream must be recreated) and cached `JobPlan`s.
Below that everything is closed: **DeepRT** prepares SpyreCode directories, **Flex** actually launches work
and talks to the driver, `spyre_comms`/`spyreccl` implements collectives (allreduce prioritized for tensor
parallel, **on-node only, up to 8 cards**), and the kernel driver and firmware are proprietary. **No ISA,
instruction encoding, assembler or `init.bin` format is published**; the nearest public contract is the
SpyreCode container spec, which describes *commands*, not *instructions*. RFCs cite **export control** as
the reason telemetry reads a vendor API rather than raw hardware counters.

### Observability and orchestration

`aiu-smi` over the `spyremetrics` API is the `nvidia-smi` analogue (closed implementation, open RFC),
reporting power, temperature, busy %, memory and PCIe bandwidth, RDMA ops, and **PT-array utilization
estimated from power**. The PyTorch Profiler is extended via `REGISTER_PRIVATEUSE1_PROFILER`, with "event
completion" redefined for a dataflow machine as "when all output tokens of a kernel have been written to
their destination." RFC 0601 tracks the **dual memory hierarchy** explicitly: device DRAM is
runtime-allocator-managed and observable only at runtime; the LX scratchpad is compiler-planned and
observable at compile time *and* runtime, with fragmentation ratio and allocation efficiency as first-class
metrics. Cluster deployment is fully open — the `ibm-aiu` org ships an OpenShift operator, a Kubernetes
device plugin, a **DRA driver**, scheduler plugins, a health checker and a webhook validator, all Apache-2.0
Go and actively maintained.

---

## Vendor Performance Claims

All figures are **vendor claims from internal testing**. **No independent benchmark — MLPerf or otherwise —
exists for Spyre in the public record.**

| Claim | Source |
|---|---|
| "2-to-3× better power/performance than GPUs on encoder-class models" | ISSCC 2026 abstract (no baseline configuration disclosed) |
| ">8 million documents/hour" ingestion at prompt size 128 (1M-unit dataset, batch 128, single card) | IBM Newsroom |
| "Near-linear scaling up to 8 cards" | ISSCC 2026 via secondary coverage |
| 4.2 TOPS/W at int8 | Secondary / ISSCC-attributed |

---

## Competitive Position

Spyre is not competing with H100-class silicon and does not pretend to. It is an **enterprise-attach
inference part**: it goes into a mainframe or Power server that a bank or insurer already owns, in a
single-slot 75 W envelope with no auxiliary power, so it can be added to existing chassis without changing
the power and cooling story. Against that yardstick:

- **Deterministic latency as the product.** No caches, no dynamic scheduling, everything decided at compile
  time. For regulated transactional workloads — fraud scoring in the transaction path — predictable tail
  latency is worth more than peak FLOPs.
- **Compiler-managed memory taken to its logical end.** Many accelerators have scratchpads; Spyre has a
  scratchpad with **no eviction path at all**, and IBM says that is deliberate. That is a stronger position
  than Cambricon MLU, Tenstorrent Tensix or Sophgo's TPU line, and it puts the entire performance burden on
  the 16-pass Inductor pipeline.
- **A genuinely open middle.** Published interface specs, public RFCs, an MLIR tile IR IBM wants
  standardized across dataflow accelerators, and an open CPU interpreter for it. Among enterprise
  accelerator vendors this is exceptional.
- **Limitations.** ~204 GB/s of LPDDR5 is an order of magnitude below HBM-class parts; single-model
  ensembles cap at 8 cards; there is no multi-node scale-out and no proprietary fabric; the runtime cannot
  be downloaded; the ISA is not public; and there is no independent benchmark or named external at-scale
  deployment.

---

## Update (2026-09-13)

*Window covered: 2026-08-08 → 2026-09-13. Primary event: IBM, "The future IBM Z & LinuxONE Processor and AI Inference Acceleration Chipset" (Christian Zoellin), Hot Chips 38, 2026-08-24; corroborated in part by IBM Newsroom, "IBM Unveils Next-Generation Dual-Architecture Processor for IBM Z and LinuxONE" (2026-08-24, IBM primary).*

**The shipping Spyre Accelerator described above is unchanged.** No new benchmark, productization, or spec news appeared for it in this window. What's new is a Hot Chips 38 **pre-announcement** of two distinct, not-yet-shipping pieces of silicon — a next-generation host CPU and a separate AI accelerator chipset. Neither is Spyre; do not conflate them with the product documented above.

**1. Next-gen IBM Z / LinuxONE CPU** — IBM-confirmed (Newsroom primary): 11 cores at 5.7+ GHz on 2 nm, natively executing both z/Architecture and AArch64 (v9.3, SVE2, 2,792 instructions implemented in hardware, not translated) concurrently — the first processor milestone from the IBM–Arm collaboration announced April 2026. Cache hierarchy (36 MB private L2/core, 432 MB virtual L3, 3.5 GB virtual L4) is reported by ServeTheHome's Hot Chips 38 coverage; IBM's own press release does not itemize cache sizes.

**2. AI Inference Acceleration Chipset** — a separate accelerator die, reported only via ServeTheHome's Hot Chips 38 coverage (no IBM primary document located for this specific piece): 16 active AI cores + 1 redundant, FP4/MXFP4 datatypes, "up to 4x TOPS" (baseline unstated), 96 GB HBM3e at up to ~4 TB/s, PCIe Gen6 low-latency peer-to-peer host interface, plus confidential-computing and quantum-safe-crypto security features.

**Is this a Spyre successor?** IBM did not say so, and gave the chipset no product name. This entry's own arithmetic — applying the disclosed "~20x the memory bandwidth of the current generation" claim to Spyre's documented ~204 GB/s LPDDR5 bandwidth yields ~4.08 TB/s, closely matching the disclosed ~4 TB/s HBM3e figure — is *suggestive* of Spyre continuity but is not an IBM statement; "current generation" could equally mean Telum II's zAIU. Record this chipset as an architecturally distinct, unnamed product pending IBM's own lineage statement, not as "Spyre 2."

**Status: pre-announcement / architecture preview only.** No availability, sampling, or shipping date was disclosed for either piece in either source.

Full detail: see the new §12 in `hw-architecture.md` and the corresponding update in `public/research/ibm-spyre/investigations/hw-architecture.md`.

**Sources for this update:** https://www.servethehome.com/ibm-z-and-linuxone-dual-isa-processor-and-ai-acceleration-at-hot-chips-2026/ · https://newsroom.ibm.com/2026-08-24-ibm-unveils-next-generation-dual-architecture-processor-for-ibm-z-and-linuxone

---

## Sources

- [IBM Research — Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research — Building the IBM Spyre Accelerator](https://research.ibm.com/blog/building-the-ibm-spyre-accelerator)
- [IBM Research — PyTorch support for IBM Spyre](https://research.ibm.com/blog/pytorch-support-ibm-spyre)
- [IBM Research — IBM Artificial Intelligence Unit (AIU)](https://research.ibm.com/blog/ibm-artificial-intelligence-unit-aiu)
- [ISSCC 2026 — Spyre: An Inference-Optimized Scalable AI Accelerator for Enterprise Workloads](https://research.ibm.com/publications/spyre-an-inference-optimized-scalable-ai-accelerator-for-enterprise-workloads)
- [IBM Newsroom — Spyre commercial availability (2025-10-07)](https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability)
- [IBM Docs — Introduction to the Spyre Accelerator (9175-ME1)](https://www.ibm.com/docs/en/systems-hardware/zsystems/9175-ME1?topic=introduction-spyre-accelerator)
- [torch-spyre developer docs](https://torch-spyre.readthedocs.io/en/latest/)
- [torch-spyre docs — Dataflow accelerator architecture](https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html)
- [torch-spyre docs — Scratchpad (LX) planning](https://torch-spyre.readthedocs.io/en/latest/compiler/scratchpad_planning.html)
- [torch-spyre docs — Work-division planning](https://torch-spyre.readthedocs.io/en/latest/compiler/work_division_planning.html)
- [torch-spyre/torch-spyre](https://github.com/torch-spyre/torch-spyre)
- [torch-spyre/interface-specs](https://github.com/torch-spyre/interface-specs)
- [torch-spyre/ktir-cpu](https://github.com/torch-spyre/ktir-cpu)
- [vLLM Spyre plugin docs](https://docs.vllm.ai/projects/spyre/en/latest/)
- [ibm-aiu — Kubernetes/OpenShift tooling](https://github.com/ibm-aiu)
- [IBM/zDNN — Telum-only (negative finding)](https://github.com/IBM/zDNN)
- [More Than Moore — IBM's Spyre AI Accelerator Deep Dive (secondary)](https://morethanmoore.substack.com/p/ibms-spyre-ai-accelerator-deep-dive)

**Next-generation preview (added 2026-09-13)**

- [ServeTheHome — IBM Z and LinuxONE Dual-ISA Processor and AI Acceleration at Hot Chips 2026 (2026-08-24)](https://www.servethehome.com/ibm-z-and-linuxone-dual-isa-processor-and-ai-acceleration-at-hot-chips-2026/)
- [IBM Newsroom — IBM Unveils Next-Generation Dual-Architecture Processor for IBM Z and LinuxONE (2026-08-24)](https://newsroom.ibm.com/2026-08-24-ibm-unveils-next-generation-dual-architecture-processor-for-ibm-z-and-linuxone)
