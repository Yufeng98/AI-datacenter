# Intel Gaudi — Software & Hardware Stack Summary

*as_of: 2026-04-05*
*last_updated: 2026-08-08*
*device_class: Mixed (GEMM + VLIW TPC)*

---

## Overview

Intel Gaudi 3 (HL-325L) is Intel's flagship AI training and inference accelerator, designed as a direct architectural competitor to NVIDIA H100. Unlike GPU architectures built around a single programmable compute engine, Gaudi 3 is heterogeneous at its core: a dedicated Matrix Multiplication Engine (MME) handles all GEMM/convolution workloads, while 64 Tensor Processing Cores (TPC) — each a VLIW SIMD processor — handle all non-GEMM operations. The entire stack is unified under SynapseAI, a closed-source graph compiler and runtime that schedules across both engines.

Gaudi 3's most architecturally distinctive feature is its 24 on-die RoCE v2 NIC ports (21 scale-up + 3 scale-out), implemented as on-die SerDes rather than a discrete NIC chip. This means distributed training collective communications (AllReduce, AllGather, ReduceScatter) run over standard Ethernet infrastructure at up to 4.2 Tbps per card, without requiring proprietary NVLink-style switching hardware. HCCL (Habana Collective Communications Library) provides a drop-in NCCL-compatible API surfacing this capability to PyTorch distributed training workflows.

---

## Software Stack

### Framework Integration

The primary framework integration is PyTorch via the open-source **gaudi-pytorch-bridge** package and the **habana_frameworks.torch** Python module. Importing `habana_frameworks.torch.core` registers a `hpu` device backend with PyTorch's dispatcher, routing all operations through SynapseAI.

The bridge supports two execution modes:
- **Lazy mode** (primary performance path): ATen ops are accumulated into a lazy IR graph; `mark_step()` triggers the SynapseAI graph compiler, enabling cross-op fusion, layout optimization, and SRAM placement — the critical path for production training throughput.
- **Eager mode**: ops dispatch immediately to SynapseAI, forgoing fusion; useful for debugging but not for production performance.

Beyond raw PyTorch, the ecosystem includes:
- **optimum-habana**: HuggingFace Transformers integration, wrapping Trainer and inference pipelines for HPU
- **DeepSpeed (Intel Gaudi fork)**: ZeRO stages 1/2/3 with HCCL replacing NCCL
- **Megatron-DeepSpeed for Gaudi** (`github.com/HabanaAI/Megatron-DeepSpeed`): 3D parallelism (tensor + pipeline + data) for large-scale LLM pretraining
- **PyTorch Lightning** (lightning-Habana HPUAccelerator): maps Lightning training loops onto HPU

### Compiler / IR

**SynapseAI Graph Compiler** is the central software component — and the only path to Gaudi hardware. It receives operator graphs from the framework bridge (or TensorFlow's XLA-based integration) and performs:
1. Operator fusion (e.g., Linear + ReLU into a single TPC kernel invocation)
2. Data layout transformation (NCHW → NHWC for hardware-preferred tensor formats)
3. Dispatch decisions: GEMM-reducible ops → MME configuration descriptors; all others → TPC kernel selection
4. SRAM placement planning: assignment of intermediate activations to the 96 MB on-die SRAM vs. HBM2e
5. DMA scheduling: double-buffering descriptors for async HBM↔SRAM movement

For TPC custom kernel development, the **TPC-C LLVM Compiler** (`tpc-clang`, from `github.com/HabanaAI/tpc_llvm`) compiles C99 + Gaudi intrinsics to VLIW binaries targeting the TPC ISA. This is the only user-facing low-level compilation pathway.

### Op Library

SynapseAI exposes an **operator graph API** (`synNodeCreateWithId` and friends) through which all ops are registered as graph nodes. This API is not the typical user-facing surface — the PyTorch bridge, TF integration, and custom kernel SDK all sit above it.

The **Habana Transformer Engine** (exposed via `habana_frameworks.torch.hpex`) provides FP8 linear layer support with E4M3/E5M2 formats, targeting the TPC's native FP8 hardware units on Gaudi 3. This is the primary interface for mixed-precision LLM training.

### Kernel Library

The **TPC Kernel Library** contains 1400+ precompiled VLIW SIMD kernels covering every non-GEMM deep learning operator — activations, normalization, elementwise, reductions, scatter/gather, and type conversion operations. Kernels are precompiled via `tpc-clang` and selected by the SynapseAI graph compiler at graph compilation time.

Users requiring operators not in the library can write custom kernels in TPC-C and register them via the SynapseAI Graph API; the compiler integrates the custom binary into the compiled graph at the appropriate TPC node.

### Runtime

**SynapseAI Runtime** (`libsynapse.so`, closed-source) is the execution layer responsible for:
- Device memory management (HBM2e allocation, SRAM block assignment)
- Stream scheduling: ordering MME, TPC, DMA, and NIC engine command queues
- DMA engine programming: HBM↔SRAM transfers with double-buffering for compute/memory overlap
- PCIe command submission: dispatching compiled graph binaries to the device via PCIe Gen5

The runtime also manages HCCL communicator lifecycle and coordinates NIC engine DMA with compute DMA to prevent HBM2e bandwidth contention.

### Driver / Firmware

The **habanalabs kernel driver** (open-source; upstreamed in the mainline Linux kernel for Goya / Gaudi 1 / Gaudi 2. **Gaudi 3 support is *not* merged as of 2026-08-08 — an out-of-tree driver is required**; mainline `drivers/accel/habanalabs/` contains `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/` and no `gaudi3/`. See the 2026-08-08 update section below.) manages:
- PCIe BAR mappings and command queue submission
- Device MMU (SMMU) configuration for HBM2e address translation
- RoCE NIC Queue Pair (QP) management for HCCL's RDMA operations
- Power management and thermal monitoring (exposed via `hl-smi`)

### Communication

**HCCL** (Habana Collective Communications Library) is Intel Gaudi's drop-in NCCL replacement. Key properties:
- API compatible: `torch.distributed.init_process_group(backend="hccl")` is a one-line migration from NCCL
- Programs the 24 integrated on-die RoCE v2 NIC ports directly via the habanalabs driver
- RDMA operations target HBM2e physical addresses of peer cards — no CPU copies, no host DRAM bounce buffers
- Supports: AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, Barrier, Send/Recv
- Two-level topology: 21 scale-up ports for intra-node/rack traffic, 3 scale-out ports for inter-rack
- Non-blocking by default: NIC engines operate independently from MME/TPC, enabling genuine compute-communication overlap

For large clusters, HCCL routes scale-up traffic through the 21 × 200 GbE ports (direct card-to-card or through standard Ethernet switches) and scale-out traffic through the 3 × 200 GbE ports to multi-rack Ethernet fabric — the same RoCEv2 protocol throughout, without a protocol boundary between intra-node and inter-node communication.

### Assembler / ISA

The TPC ISA is a VLIW architecture with 4 pipeline slots per instruction: Vector (256-byte SIMD), Scalar, Load, and Store. The Load and Store slots include Address Generation Units (AGU) with native 5-dimensional tensor addressing. This ISA is not user-visible in normal use — it is the compilation target of `tpc-clang`. There is no PTX-equivalent virtual ISA; TPC binaries are architecture-generation specific.

---

## Hardware Architecture

### Compute Engine (Dual Engine: MME + TPC)

Gaudi 3 uses a heterogeneous dual-engine compute model:

**MME (Matrix Multiplication Engine):** 8 MME engines (4 per compute die), each dedicated to GEMM and GEMM-reducible operations (dense matrix multiply, batched GEMM, convolution, attention dot-products). The compiler statically tiles all GEMM work across the 8 MMEs. Gaudi 3 delivers 1835 BF16 matrix TFLOPs (OAM).

**TPC (Tensor Processing Core):** 64 TPC VLIW SIMD processors (32 per compute die). Each TPC has:
- 256-byte vector width: 64 × FP32, 128 × BF16/FP16, or 256 × FP8 lanes per cycle
- 4-slot VLIW instruction: Vector (SIMD ALU), Scalar, Load (with AGU), Store (with AGU)
- Native data type support: FP32, BF16, FP16, FP8 (E4M3/E5M2), INT32, INT16, INT8
- Each TPC is independently programmable; the compiler assigns operator kernels to TPC subsets

### Data Path

The canonical execution pipeline is MME → SRAM → TPC:
1. MME reads weight tiles from SRAM, computes GEMM results, writes to SRAM
2. TPC reads MME output from SRAM, applies activation/normalization/elementwise, writes result to SRAM
3. Async DMA engines move tensors between HBM2e and SRAM; double-buffering enables compute and memory movement to overlap

All four engine classes (MME, TPC, NIC, DMA) can execute in parallel when scheduled appropriately by the SynapseAI graph compiler.

### On-chip Memory (SRAM)

- **96 MB total** (48 MB per compute die)
- **12.8 TB/s aggregate bandwidth** — 3.5× higher than HBM2e bandwidth (3.7 TB/s)
- Software-managed scratchpad: the SynapseAI compiler (not hardware prefetch logic) determines data placement
- Shared between MME and TPC engines on each die
- Functions as: activation buffer between pipeline stages, operand staging for TPC kernels, GEMM tile prefetch buffer
- TPC processors also have small per-TPC local scratchpad in addition to shared SRAM access

### Off-chip Memory (HBM2e)

- **128 GB HBM2e**, 8 stacks (Gaudi 3)
- **3.7 TB/s bandwidth** (~51% improvement over Gaudi 2's 2.45 TB/s)
- Stores: model weights, optimizer states, activations that overflow SRAM, gradient tensors
- Directly accessible via RDMA by the integrated RoCE NICs for collective communications — no CPU intermediate

### Host Interface / Package

- **PCIe Gen5 x16**: ~128 GB/s bidirectional peak (doubled from Gaudi 2's Gen4)
- **OAM (Open Accelerator Module)** form factor (HL-325L): 8 cards per HGX-compatible server tray; cards can be directly cabled for scale-up
- **PCIe card** form factor (HL-338): single-card deployment option
- **Dual-die package**: two compute dies on one package, each with 4 MMEs, 32 TPCs, 48 MB SRAM; dies share the HBM2e pool and NIC ports

### Scale-up Interconnect (Integrated RoCE)

- **21 × 200 GbE RoCE v2 ports** per card, implemented as on-die SerDes (48 pairs of 112 Gbps PAM4 Tx/Rx)
- No discrete NIC chip; the SerDes are on the compute die alongside MME and TPC engines
- Total scale-up bandwidth: **4.2 Tbps per card** (unidirectional)
- Supports direct card-to-card cabling within a server tray or routing through standard Ethernet switches
- RDMA operations write directly to HBM2e physical addresses — no CPU involvement in data path
- Standard Ethernet means any switch vendor's 200G/400G fabric works; no proprietary switching ASICs required

### Scale-out Interconnect

- **3 × 200 GbE RoCE v2 ports** per card for inter-rack, multi-node cluster connectivity
- Same on-die SerDes technology as scale-up ports
- Connect to external Ethernet switches (Cisco Nexus 9000, Arista, etc.) for multi-rack deployment
- HCCL transparently routes collective operations across scale-up and scale-out ports; API is identical from the application perspective
- Host NIC (PCIe peer-direct GDR) is a supported escape hatch for very large clusters requiring more than 3 × 200 GbE scale-out bandwidth

---

## Programming Model Rationale

Intel Gaudi's software stack reflects five fundamental hardware constraints and design choices:

**1. Dual-engine architecture requires a graph compiler as the scheduler.**
MME and TPC cannot be directly programmed by the user — there is no CUDA equivalent. The graph compiler (SynapseAI) is the necessary intermediary that decides which ops go to which engine and in what order. This is why there is no user-visible low-level GPU kernel API: the op-to-engine dispatch is a compiler decision, not a programmer decision. The practical implication is that SynapseAI's fusion and scheduling quality is the primary determinant of performance, and the lazy mode `mark_step()` boundary is the critical design decision that gives the compiler a large enough graph to optimize.

**2. 96 MB SRAM at 12.8 TB/s is the performance lever — SRAM placement strategy is critical.**
SRAM bandwidth is 3.5× HBM bandwidth. An operator that keeps its data in SRAM (by fitting within a graph segment's working set) runs at 12.8 TB/s; one that spills to HBM runs at 3.7 TB/s. This is why the graph compiler's SRAM placement strategy is the most important optimization — it is the difference between memory-bandwidth-bound operations hitting 3.5× better performance. This also explains why large-batch lazy mode outperforms small-batch eager mode even at low arithmetic intensity: the compiler can tile and pipeline to keep intermediates in SRAM.

**3. Integrated RoCE NICs eliminate external networking hardware — unique TCO advantage.**
No other major AI training accelerator (as of 2024) integrates Ethernet NICs on the compute die. This design choice means Gaudi clusters need only standard Ethernet switches (commodity infrastructure) rather than proprietary NVLink switches or InfiniBand HCAs. HCCL's NCCL-compatible API ensures existing distributed training code migrates with minimal effort. The cost of this is the fixed 21:3 scale-up to scale-out port split — NVIDIA's model allows more flexible fabric allocation via NVSwitch.

**4. TPC's VLIW SIMD + TPC-C SDK provides flexibility NVIDIA's fixed-function Tensor Cores do not.**
NVIDIA Tensor Cores are fixed-function hardware: they execute one operation (matrix multiply-accumulate) at high efficiency and cannot be reprogrammed for other operators. TPC processors are general VLIW SIMD units: they can execute any operator expressible in TPC-C. This means Gaudi can natively support novel operators (FP8 quantization schemes, custom activations, experimental attention variants) by writing new TPC-C kernels, without hardware changes. The 1400+ precompiled TPC Kernel Library represents Intel's investment in pre-building the most common operator set. The TPC-C LLVM compiler (`tpc_llvm`, open-source) is the programmability escape hatch.

**5. NCCL-compatible HCCL API ensures ecosystem compatibility despite different hardware.**
Changing the collective communications library would require modifying every distributed training framework. HCCL's decision to mirror NCCL's API exactly (including `backend="hccl"` in `torch.distributed`) means DeepSpeed, Megatron-DeepSpeed, FSDP, and PyTorch Lightning all work on Gaudi with minimal code changes. The underlying hardware is radically different (on-die Ethernet NICs vs. external InfiniBand), but the software interface is identical. This is a deliberate ecosystem compatibility choice.

---

---

## Roadmap: Post-Gaudi 3 Strategy

*as_of: 2026-04-05*

Intel's post-Gaudi 3 roadmap has undergone significant strategic restructuring since 2025. There is no product called "Gaudi 4." The three-phase trajectory is as follows:

### Falcon Shores — Cancelled (Jan 2025)

Originally planned as a multi-chiplet hybrid (Xe-HPC GPU + Gaudi ASIC) targeting late 2025, Falcon Shores was cancelled on Intel's Q4 2024 earnings call. Customer feedback indicated the product was not competitive, and Intel publicly acknowledged it was not participating meaningfully in cloud AI data center markets. Falcon Shores was relegated to an internal test chip and will not ship commercially.

### Crescent Island — Inference GPU (customer sampling H2 2026)

Announced at OCP Global Summit (October 2025). Crescent Island is an inference-oriented GPU based on the Xe3P architecture — a performance-optimized variant of Intel's Xe3 GPU used in Panther Lake mobile. The reference design carries **160 GB of LPDDR5X** memory (capacity-optimized over bandwidth, unlike Gaudi's HBM), with **partner/ODM configurations up to 480 GB LPDDR5X** per the Computex 2026 disclosure. Positioned for "tokens-as-a-service" inference operators in air-cooled enterprise servers. **350 W air-cooled PCIe add-in card**; datatype range reported as **FP4 through FP64**. Memory bandwidth and TFLOPs are **not disclosed**. **Note:** This product diverges from the Gaudi ASIC architecture; it is not a training accelerator and does not appear to carry Gaudi's on-die RoCE NIC design. Computex figures are trade-press-sourced (medium confidence) — see the 2026-08-08 update section.

### Jaguar Shores — Gaudi-branded rack-scale platform (~2027)

Confirmed by Intel to carry the Gaudi brand. Built on Intel 18A (gate-all-around) process with HBM4 memory from SK Hynix (HBM4E also rumored). Reported package: ~92.5 mm × 92.5 mm, quad-tile configuration, octal HBM stacks. Designed as a rack-scale disaggregated platform paired with Diamond Rapids Xeon CPUs and using silicon photonics for optical interconnects — a departure from Gaudi 3's copper-based on-die RoCE model. Earliest expected availability: late 2027. Full specifications not yet disclosed.

### Roadmap Summary Table

| Product | Architecture | Memory | Status | Target |
|---------|-------------|--------|--------|--------|
| Gaudi 3 (HL-325L) | Gaudi ASIC, dual-die | 128 GB HBM2e @ 3.7 TB/s | GA, shipping | Training + inference |
| Falcon Shores | Xe-HPC + Gaudi hybrid | — | **Cancelled** (Jan 2025) | Was training + inference |
| Crescent Island | Xe3P GPU | 160 GB LPDDR5X reference, up to 480 GB (partner configs); bandwidth not disclosed | Sampling H2 2026, GA 2027; 350 W air-cooled PCIe | Inference (FP4–FP64) |
| Jaguar Shores | Gaudi-branded, Intel 18A, rack-scale | HBM4 / HBM4E (SK Hynix) | ~2027 pre-announcement | Training + inference |

*Crescent Island figures per Computex 2026 trade coverage (2026-06-01/02); no Intel first-party page carrying these figures was located. Hot Chips 38 talk 2026-08-24 pending.*

### Strategic Implications

- Intel has no commercially available training-class successor to Gaudi 3 until at least 2027 (Jaguar Shores), creating a multi-year portfolio gap for large-scale LLM training use cases.
- Crescent Island fills the inference market near-term but with a different architecture (GPU vs. ASIC). **The stack-continuity question is now effectively answered: SynapseAI/HCCL does not carry forward.** Crescent Island rides Intel's unified Xe path — oneAPI/SYCL over the Level Zero Compute Runtime — with enablement already landed upstream of the hardware (Intel Compute Runtime 26.01.36711.4, 2026-01-14; Intel Graphics Compiler v2.27.10, Jan 2026; Compute Runtime support promoted 2026-04-20).
- Jaguar Shores' rack-scale disaggregated design and silicon photonics interconnects align Intel more closely with next-generation datacenter network architectures (co-packaged optics, optical switching fabrics), but remain unproven at production scale.
- Software maturity remains Intel's most cited competitive weakness. Jaguar Shores has disclosed no software stack details; Crescent Island's stack is the Arc/Xe oneAPI stack, which Intel said at OCP (Oct 2025) is being developed and validated on Arc Pro B-series GPUs before extending to Xe3P.
- The discontinuity is therefore two-sided: Gaudi's graph-compiler stack (SynapseAI + TPC-C + HCCL) has no successor role in Intel's datacenter AI roadmap, and Crescent Island customers inherit none of the Gaudi-specific tooling investment.

---

## Intel Gaudi Update (2026-08-08)

*Updated 2026-08-08. Window covered: 2026-04-05 → 2026-08-08. Sources listed per item; Crescent Island specs are trade-press-sourced and marked as such.*

**Bottom line: Gaudi 3 hardware is unchanged.** Everything material in this window concerns (a) the Crescent Island successor and (b) two factual corrections to text previously committed in this entry. Prior-generation content above is preserved as written except where a statement was factually wrong.

### 1. Correction — Gaudi 3 is NOT upstreamed in Linux

The Software Stack section previously said the habanalabs driver was "open-source, upstreamed in Linux kernel" without qualification. That is true for **Goya, Gaudi 1 and Gaudi 2 only**. Verified against mainline: `drivers/accel/habanalabs/` contains `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/` — **there is no `gaudi3/` directory in Linux master as of 2026-08-08**. Intel posted Gaudi 3 driver code for upstreaming in late Nov 2025 and sent a pull request targeting v6.19 (dri-devel, Dec 2025); it was rejected on code-quality grounds (reverts and build artifacts in a ~300k-line series) and now targets a later cycle. **Gaudi 3 deployment requires the out-of-tree habanalabs driver.**

Sources: https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs · https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html · https://www.phoronix.com/news/Intel-SynapseAI-Stops

### 2. Correction of scope — one archived repository, not a dead user-space stack

`github.com/HabanaAI/SynapseAI_Core`, the open **reference implementation** of the SynapseAI API, was archived **2025-02-03** carrying the notice: *"This project will no longer be maintained by Intel. Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates."* Two other repos are also archived: `Gaudi-tutorials` (2025-09-18) and `Model-References` (2026-01-08).

This does **not** mean the Gaudi user-space stack is dead. Verified against the HabanaAI org: `gaudi-pytorch-bridge` is not archived (pushed 2026-07-13), `vllm-fork` (2026-07-27), `optimum-habana-fork` (2026-07-17), the `gaudi-*` Kubernetes operator/device-plugin/exporter suite (2026-07-23), `Megatron-LM` and `Setup_and_Install` (2026-07-10). `docs.habana.ai` currently serves **Intel Gaudi software v1.24.0**.

The accurate framing: the archived reference implementation removes the accompanying open user-space that Linux's `accel` subsystem expects from a driver submission, which is a material obstacle to Gaudi 3 driver upstreaming — while the production PyTorch/vLLM enablement path continues to ship. All of this predates the 2026-04-05 baseline; it is a repo gap being closed, not new news.

Sources: https://github.com/orgs/HabanaAI/repositories?type=all&sort=updated · https://docs.habana.ai/en/latest/

### 3. Benchmark absence — Gaudi has gone quiet

- **MLPerf Inference v6.0**, published **2026-04-01** (four days *before* the prior baseline, so not a window event): 24 submitting organizations. Intel submitted **Xeon 6 CPUs and Arc Pro B70/B65/B60 GPUs only**. Intel's own newsroom post does not mention Gaudi once.
- **MLPerf Training v6.0**, published **2026-06-16** (in window): 24 submitting organizations — **Intel is not among them at all.**

Gaudi 3 has now been absent from consecutive MLPerf rounds. Recorded as an observation, not as an Intel statement.

Sources: https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ · https://newsroom.intel.com/artificial-intelligence/intel-delivers-ai-performance-mlperf-inference-v6-0 · https://mlcommons.org/2026/06/mlperf-training-v6-0-results/

### 4. No Gaudi 3 EOL — NOT CONFIRMED

No Intel first-party product-discontinuance notice for Gaudi 3 exists. Trade commentary asserting "Gaudi will be discontinued in 2026–2027" is third-party. Gaudi 3 remains GA through cloud/enterprise channels and software releases continue (v1.24). Characterize Gaudi 3 as **strategically deprioritized, not end-of-life**.

### 5. Crescent Island — Computex 2026 disclosure (the only substantive new hardware information)

Around **2026-06-01/02** at Computex, Intel gave Crescent Island a fuller spec sheet, consistently reported by Tom's Hardware, VideoCardz, TechSpot, TechTimes, Neowin and Wccftech:

| Property | Value |
|---|---|
| Memory (reference design) | 160 GB LPDDR5X |
| Memory (partner/ODM configs) | up to 480 GB LPDDR5X |
| Memory bandwidth | **not disclosed** |
| Power / cooling | 350 W, air-cooled PCIe add-in card (explicitly not liquid-cooled) |
| Datatypes | FP4 through FP64 |
| Peak throughput (any datatype) | **not disclosed** |
| Xe3P core / XMX counts, process node, package | **not disclosed** |
| Schedule | customer sampling H2 2026 (unchanged), general availability 2027 |

**Confidence caveat.** Intel's own Computex 2026 materials — the press kit, "Intel Announces New AI Innovations at Computex" (2026-06-01) and "Computex 2026: An Intelligent World Built on Silicon" (2026-06-02) — **do not mention Crescent Island at all**. No Intel-hosted page carrying 480 GB / 350 W / FP4–FP64 could be located. Corroboration is broad but single-origin (trade coverage of an Intel briefing/slide). These figures are **medium confidence, trade-press-sourced**.

**Explicitly not disclosed, and deliberately not recorded here as specs:**
- **Memory bandwidth.** The widely circulated "~684 GB/s" is a **press estimate**, not an Intel figure (TechSpot: bandwidth "is expected to reach 684 GB/s"; MLQ.ai calls it "estimated"; other coverage notes Intel gave capacity, TDP and architecture but no headline bandwidth number). Press estimates of roughly 0.6–0.7 TB/s exist but are back-calculated from an assumed LPDDR5X bus width.
- The "20 × 24 GB module" capacity breakdown — a press reconstruction with no Intel attribution.
- TFLOPS at any datatype, Xe3P core/XMX counts, process node, die/package configuration.

Sources: https://acceleratedcomputing.ai/news/2026-06-01-intel-crescent-island/ · https://codeoxi.com/blog/intel-crescent-island-gpu · https://videocardz.com/newz/intel-crescent-island-gpu-officially-supports-up-to-480gb-lpddr5x-memory · https://www.techspot.com/news/112608-intel-crescent-island-gpu-support-up-480gb-lpddr5x.html · https://newsroom.intel.com/press-kit/press-kit-intel-at-computex-2026 (no Crescent Island mention)

### 6. Crescent Island software stack — the open question is largely answered

This entry previously flagged SynapseAI/HCCL continuity across Gaudi → Crescent Island as an open question, and said "neither Crescent Island nor Jaguar Shores has disclosed software stack details." The evidence says **SynapseAI/HCCL does not carry forward**. Crescent Island rides Intel's unified Xe path: **oneAPI/SYCL over the Level Zero Compute Runtime**. Intel stated at OCP (Oct 2025) that this stack is being developed and validated on Arc Pro B-series GPUs and will extend to Xe3P. Concrete enablement landed upstream well ahead of the hardware:

- **Intel Compute Runtime 26.01.36711.4** (2026-01-14) — early Crescent Island support
- **Intel Graphics Compiler v2.27.10** (Jan 2026) — initial Crescent Island support
- **Intel Compute Runtime** — Crescent Island support **promoted 2026-04-20**

Practical implication for this survey: the Gaudi programming model documented above (SynapseAI graph compiler, TPC-C VLIW kernels, HCCL) is a **terminal branch**. Nothing in Intel's disclosed datacenter AI roadmap inherits it.

Sources: https://www.phoronix.com/news/Intel-CR-26.01.36711.4 · https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler

### 7. Jaguar Shores — nothing new

No primary-source information in the window. Trade reporting continues to place it at ~2027 (rack-scale, silicon photonics, HBM4, Intel 18A). Claims of "design finalization by mid-2026" are supply-chain sourcing, not Intel statements. Jaguar Shores has no Hot Chips 38 slot. **All timing detail remains LOW confidence; the roadmap text above is unchanged.**

### 8. Hot Chips 38 — scheduled, not evidence

Hot Chips 38 runs **2026-08-23/25**. The program lists, in the Day-1 GPU session (Mon **2026-08-24**, 4:45–6:45 PM), Intel's "**Crescent Island: GPU Designed for Agentic AI Inference**" (Sumit Mohan, Hong Jiang). **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** No slides, abstract or specs exist; this talk must not be cited as the source of any number. It is the expected settling source for TFLOPS, Xe3P core counts, process node and real memory bandwidth. Re-scan after 2026-08-25. No Gaudi or Jaguar Shores talk is on the program.

Source: https://hotchips.org/program/conference/

### 9. Adjacent, non-Gaudi (context only)

At Computex 2026 Intel announced production rack-scale AI infrastructure pairing Xeon 6+ (Intel 18A) with **SambaNova SN-50 RDUs**, integrated by Foxconn (confirmed via Intel's own investor press release). This is relevant to the survey only as evidence that Intel is positioning as a host/rack integrator alongside third-party accelerators; it is **not** a Gaudi development and does not enter the Gaudi roadmap table. Source: https://www.intc.com/news-events/press-releases/detail/1771/intel-announces-new-ai-innovations-at-computex-chip-to

---

## Resources

| Resource | URL |
|----------|-----|
| Intel Gaudi Software Documentation (v1.24.0, current as of 2026-08-08) | https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html |
| Gaudi 3 White Paper (Intel, 2024) | https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf |
| Gaudi 3 OAM Product Brief | https://cdrdv2-public.intel.com/817487/gaudi-3-ai-accelerator-hl-325l-oam-mezzanine-card-product-brief.pdf |
| Gaudi 3 PCIe Product Brief | https://cdrdv2-public.intel.com/817488/Gaudi%203%20PCIe%20Product%20Brief_RB_1_V6.pdf |
| Gaudi 3 Cluster Reference Design | https://cdrdv2-public.intel.com/833842/gaudi-3-ai-accelerator-cluster-ref-design-white-paper.pdf |
| Hot Chips 2024 Presentation | https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf |
| gaudi-pytorch-bridge (GitHub) | https://github.com/HabanaAI/gaudi-pytorch-bridge |
| TPC LLVM Compiler (GitHub) | https://github.com/HabanaAI/tpc_llvm |
| Megatron-DeepSpeed for Gaudi (GitHub) | https://github.com/HabanaAI/Megatron-DeepSpeed |
| HCCL API Reference | https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html |
| TPC User Guide | https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html |
| PyTorch Theory of Operations | https://docs.habana.ai/en/latest/PyTorch/PyTorch_Gaudi_Theory_of_Operations.html |
| habanalabs in mainline Linux (no `gaudi3/` as of 2026-08-08) | https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs |
| Gaudi 3 upstreaming pull request, dri-devel (Dec 2025, rejected) | https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html |
| SynapseAI_Core archived Feb 2025 (Phoronix, 2025-12-15) | https://www.phoronix.com/news/Intel-SynapseAI-Stops |
| HabanaAI GitHub org — repository activity / archive status | https://github.com/orgs/HabanaAI/repositories?type=all&sort=updated |
| MLPerf Inference v6.0 results (2026-04-01) | https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ |
| MLPerf Training v6.0 results (2026-06-16; no Intel submission) | https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ |
| Crescent Island Computex 2026 coverage (480 GB, 350 W, FP4–FP64) | https://videocardz.com/newz/intel-crescent-island-gpu-officially-supports-up-to-480gb-lpddr5x-memory |
| Intel Compute Runtime 26.01.36711.4 — early Crescent Island support | https://www.phoronix.com/news/Intel-CR-26.01.36711.4 |
| Intel Graphics Compiler v2.27.10 — initial Crescent Island support | https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler |
| Hot Chips 38 program (Intel Crescent Island talk, 2026-08-24 — not yet public) | https://hotchips.org/program/conference/ |
