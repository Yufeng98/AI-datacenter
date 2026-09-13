# Meta MTIA — Summary

**Device class:** Inference Accelerator (v1 / v2 / 2i); Training + Inference Accelerator (MTIA 300 onward)
**Manufacturer:** Meta (in-house design), fabricated by TSMC
**Deployment:** Meta data centers (Facebook, Instagram, WhatsApp ranking/ads/LLM inference)
**Research date:** 2026-04-05 · **Last updated:** 2026-08-08

---

## What It Is

MTIA (Meta Training and Inference Accelerator) is Meta's custom inference ASIC built to replace GPU inference for recommendation and ranking workloads at scale. Six chip generations have shipped in ~2 years. The architecture centers on a grid of RISC-V + fixed-function processing elements with large on-chip SRAM. Through MTIA 2i the off-chip memory is LPDDR5 (not HBM) — optimized for DLRM cost-efficiency. **From MTIA 300 (first silicon disclosure at ISCA 2026) the line pivots: HBM3E, chiplet packaging, FP8 training throughput, and NIC chiplets integrated into the accelerator package** — see the MTIA 300 update section below.

---

## Key Architecture Features

| Feature | MTIA v1 | MTIA v2 |
|---|---|---|
| Process | TSMC 7 nm | TSMC 5 nm |
| PE grid | 8×8 = 64 PEs | 8×8 = 64 PEs |
| INT8 TOPS | 102.4 | ~357 (3.5×) |
| INT8 TOPS sparse | — | ~714 (7×) |
| On-chip SRAM | 128 MB | 256 MB |
| DRAM | 64 GB LPDDR5, 176 GB/s | 128 GB LPDDR5, 204.8 GB/s |
| TDP | 25 W | 90 W |
| Die area | 373 mm² | 421 mm² |
| PCIe | Gen4 | Gen5 ×8 |

MTIA 2i (2025): 256 MB SRAM, 2.7 TB/s SRAM bandwidth

*The table above covers the LPDDR5-era parts only. For MTIA 300 silicon specifications see "MTIA 300 Update — ISCA 2026 Silicon Disclosure" below.*

---

## Software Stack

```
PyTorch (eager + torch.compile + torch.export)
  → TorchDynamo / TorchFX IR
  → TorchInductor MTIA graph compiler
  → Triton-MTIA kernel compiler (MLIR + LLVM)
  → MTIA Streaming Interface (async runtime)
  → MTIA Firmware Driver
  → Hardware
```

- Triton used as hardware-agnostic kernel language (GPU + MTIA)
- vLLM support for LLM inference
- KernelEvolve: agentic LLM-based kernel generation

---

## Distinguishing Design Choices

1. **RISC-V control cores** — no ISA royalties, full customization
2. **LPDDR5 not HBM** — DLRM is not as bandwidth-starved as training; saves cost
3. **Triton-first kernel interface** — portability across GPU and MTIA
4. **PyTorch eager mode support** — rare for ASICs, enables developer flexibility
5. **Asynchronous dataflow** — hides memory latency for embedding lookups
6. **Sparsity hardware** — 7× boost for pruned recommendation models

---

## Scale

- Hundreds of thousands of MTIA 2i deployed at Meta (confirmed by the ISCA 2026 paper)
- MTIA v2 / 2i: 72 chips per rack, ~50 PetaFLOPS INT8 per rack (sparse)
- MTIA 300: 16 accelerators per rack (1 accelerator + 1 host CPU per compute blade), liquid-cooled
- Used alongside NVIDIA H100/A100/H200 and AMD GPUs

---

## Sources

- https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/
- https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/
- https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf
- https://arxiv.org/html/2512.23236v1

---

## MTIA v2 Update (2025–2026)

*Updated 2026-04-05. Sources: ISCA 2025 paper, Meta engineering blog (August 2024), Meta roadmap announcement (March 2026).*

### Corrected v2 Specifications (from ISCA 2025)

Two parameters in the original summary table were underspecified:

- **PE-local SRAM**: 384 KB/PE in v2 (not 128 KB — that is v1's figure)
- **NoC bandwidth**: 3.3× over v1 (not 2×)
- **PCIe**: ~~Gen5 ×16 (128 GB/s), not ×8~~ — *this 2026-04-05 "correction" was itself wrong and was reverted on 2026-08-08. The ISCA 2025 spec table says **8× PCIe Gen5 (32 GB/s)** for MTIA 2i; the original ×8 reading was right.*

Updated comparison row for the table above:

| Feature | MTIA v1 | MTIA v2 (corrected) |
|---|---|---|
| PE-local SRAM | 128 KB/PE | **384 KB/PE** (3×) |
| NoC BW | baseline | **3.3× v1** |
| PCIe | 8× Gen4 (16 GB/s) | **8× Gen5 (32 GB/s)** (re-corrected 2026-08-08) |

### Production Results (ISCA 2025)

The ISCA 2025 paper ("Meta's Second Generation AI Chip: Model-Chip Co-Design and Productionization Experiences") documents production outcomes for MTIA 2i (the MTIA v2 silicon revision, also called MTIA 200):

- **44% TCO reduction** vs GPU inference for production recommendation workloads
- **23 firmware-bundle releases** fleet-wide in 2024 — firmware as primary software delivery mechanism
- Model-chip co-design methodology: pruning schedules, embedding sharding, and static shape compilation co-designed with hardware to achieve the 7× sparse compute figure

### MTIA 300/400/450/500 — Announced March 2026

Meta announced a four-chip roadmap co-designed with Broadcom, shifting to ~6-month chip cadence:

| Chip | Status (April 2026) | Workload | Memory | Headline Improvement |
|---|---|---|---|---|
| MTIA 300 | Production | Ranking/rec training | 216 GB HBM | First training-capable MTIA; HBM debut |
| MTIA 400 | Lab testing | GenAI inference | HBM capacity **not disclosed** | +400% FP8 FLOPS, +51% HBM BW vs MTIA 300 |
| MTIA 450 | ~early 2027 | GenAI inference | — | **+75% MX4 FLOPS vs MTIA 400**; 2× HBM BW |
| MTIA 500 | 2027 | GenAI inference | — | +50% HBM BW, up to +80% HBM capacity, +43% MX4 FLOPS vs 450 |

> **Corrections applied 2026-08-08** (against Meta's own 2026-03-11 blog):
> - The earlier "MTIA 450: 6× MX4 FLOPS vs MTIA 400" was a transcription error. Meta's blog states MTIA 450 increases MX4 FLOPS **by 75% over MTIA 400**; the "6×" figure is MX4 FLOPS relative to FP16/BF16 *on the same chip*.
> - The earlier "MTIA 400: 288 GB HBM" is **not supported by any primary source** — Meta publishes only relative deltas for MTIA 400. 288 GB appears only in aggregators and has been removed.
> - MTIA 500 is "scheduled for mass deployment in 2027" per the blog — the earlier "~late 2027" was an unattributed inference.
> - **No process node is disclosed for MTIA 300/400/450/500.** Any "2nm" figure in circulation is aggregator-only.

**Roadmap headline figures** (MTIA 300 → MTIA 500): HBM bandwidth +4.5×, compute FLOPs +25× (matches the blog).

**Key architectural shift**: MTIA 300+ moves from LPDDR5 to HBM, reflecting a workload pivot from DLRM-only to LLM/GenAI inference and training. Modular chiplet design allows MTIA 400/450/500 to drop into the same chassis/rack/network infrastructure.

---

## MTIA 300 Update — ISCA 2026 Silicon Disclosure (2026-08-08)

*Updated 2026-08-08. Primary source: "MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines", MTIA Team, Meta Platforms, **ISCA 2026 Industry Track**, Session 5B "Industry Track 2", Tuesday 2026-06-30, 12:00–12:20 EDT, Room 302 (corresponding author Chunqiang Tang) — https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf. Meta's publications index renders the title as "MTIA-300: Meta's Training Chip with Embedded NIC Chiplets and Communication Offloading Engine". Secondary: Meta newsroom Broadcom PR 2026-04-14; Reuters/TechCrunch 2026-07-09.*

This is the **first silicon-level disclosure of MTIA 300**. The March 11, 2026 roadmap blog recorded above gave only "216 GB HBM" and "first training-capable". All figures below are **Meta-stated**, from the paper (Table I unless noted).

### Why MTIA 300 matters architecturally

MTIA 300 is the first MTIA that is a *training* chip, and the first accelerator Meta claims — "to our knowledge" — that combines **built-in NIC chiplets** with **general-purpose collective offloading engines** on the accelerator package. The network is not a peripheral: 12 RoCE NICs live on two network chiplets in the same package as the compute die, and 16 on-die Message Engines execute Reduce / AllReduce / ReduceScatter without occupying the compute PEs.

### MTIA 300 headline specifications

| Parameter | MTIA 2i (a.k.a. MTIA 200) | **MTIA 300** |
|---|---|---|
| Role | Inference | **Training (ranking/recommendation) + inference** |
| Clock | 1.35 GHz | **1.9 GHz** |
| Supply voltage | — | **0.85 V** |
| TDP | 85 W (65 W typical) | **912 W (667 W typical)** — ~10.7× |
| Cooling | Air | **Liquid** |
| Instances (Table I wording) | "2.35B gates, 103M FLOPS" | **"7.2B gates, 511M FLOPS"** |
| Packaging | Monolithic | **2.5D chiplet: 1 compute chiplet + 2 network chiplets on a 3.2×-reticle interposer** |
| Peak GEMM FP16/BF16 | 177 TFLOP/s | **560 TFLOP/s** |
| Peak GEMM FP8 | — (INT8: 354 TOPS) | **1,120 TFLOP/s** |
| On-chip SRAM / LLC | 256 MB @ 2.7 TB/s | **192 MB @ 11.4 TB/s (R+W)** |
| Off-chip memory | 128 GB LPDDR5 @ 204.8 GB/s | **216 GB HBM3E @ 6.1 TB/s (read *or* write)** |
| Per-PE local memory | 384 KB @ 1.0 TB/s | **512 KB @ 1.9 TB/s (R+W)** |
| Host interface (Table I) | 8× PCIe Gen5 (32 GB/s) | **16× PCIe Gen5 (64 GB/s)** |
| Built-in networking | None | **12 × 800 Gbps RoCE NICs = 1.2 TB/s** |

> **Note on "511M FLOPS".** ISCA 2026 Table I literally prints "7.2B gates, 511M FLOPS" in the *Instances* row. In context "FLOPS" is the paper's abbreviation for **flip-flops** (instance counts), not floating-point throughput. The paper's wording is quoted here so the figure is traceable; do not read it as a performance number.

> **Note on bandwidth units.** 6.1 TB/s HBM is stated as read-**or**-write. The 11.4 TB/s LLC and 1.9 TB/s local-memory figures are read**+**write. The qualifiers are not interchangeable.

> **Process node: not disclosed.** The paper gives die and package dimensions but no node for MTIA 300.

### Compute

- **12 × 6 = 72 PEs**, plus **one redundant PE row** (6 additional PEs) for yield: each PE column can tolerate one faulty PE, configured at boot, transparent to software and with no NoC performance impact.
- **16 Message Engines** at the PE-grid edges (see Collective offload below).
- Per PE: two 64 B-wide **RISC-V vector cores**, Command Processor, Memory Layout Unit, **Dot Product Engine** (two 32×64B×32 MAC tiles; **7.82 TFLOPS/PE** with FP16/BF16 inputs and FP32 output; also FP8 S1E4M3/S1E5M2 and TF32), Reduction Engine, SIMD/SFU, Fabric Interface, Memory Bridge, and **512 KB software-managed local memory** split into circular buffers.
- Control core is a **RISC-V quad SMP core**.
- **INT8 GEMM removed, FP8 added** for training. SIMD width raised **32 → 128 elements/cycle**, cutting the GEMM:SIMD ratio to **16:1** from MTIA 2i's 32:1. SIMD throughput: 42.5 TOPS on the SIMD engine (FP8/FP16/BF16/FP32) plus 42.5 (INT8/FP16) / 21.3 (BF16/FP32) on the RISC-V vector cores.
- NoC uses **cluster routers, L-routing (X then Y), and virtual lanes** for deadlock avoidance. Unlike MTIA 2i the compute chiplet has **no memory crossbar**.

### Memory

- **216 GB HBM3E in 6 stacks** — the paper states "each side connects to 3 twelve-high HBM3E stacks" at 8.0 Gbps. (36 GB per stack is arithmetic, 216/6, not a stated figure.) **6.1 TB/s read-or-write.**
- **192 MB on-chip LLC**: the die figure shows "48 LLC (96MB)" on **each side** — 48 banks and 96 MB per side, **96 banks / 192 MB total**, at 11.4 TB/s R+W. This is *lower capacity* than MTIA 2i's 256 MB but **4.2× its 2.7 TB/s bandwidth**.
- **Measured 5.57 TB/s (91% of peak)** on a BF16-add kernel, plotted against H100 (2.4 TB/s peak) and H200 (4.8 TB/s peak).

### Built-in NIC chiplets (the headline novelty)

- **Two network chiplets**, each containing **six custom 800 Gbps (100 GB/s) RDMA/RoCE IP blocks** based on third-party NIC IP → **600 GB/s per chiplet, 12 NICs / 1.2 TB/s per accelerator**, attached over a die-to-die interface with 112G SerDes (8×112G per NIC in Fig. 4).
- NIC optimizations: **"express doorbells"** — the work request itself serves as the doorbell write, avoiding an extra HBM ring-buffer read that cost ~800 ns per transaction; **24,576 outstanding work requests across 1,024 QPs per IP block**; **QP caching removed** to save area, limiting each *NIC chiplet* to 1,100 active QPs; simplified packet-processing pipeline.

### Collective offload — Message Engines

- **16 Message Engines** placed at the PE-grid edges next to HBM, cache and I/O.
- Each ME: one scalar RISC-V core (**CPU-M**), **256 KB context RAM**, a single shared Completion Queue, and a **Near Memory Compute** block doing 128 B/cycle for reduction or DMA (96 B/cycle when all are active concurrently).
- MEs deliver up to **2.8 TB/s of reduction throughput** — over 2× the 1.2 TB/s I/O bandwidth — while using only **one-third of the chip area of the compute engines**. They accelerate Reduce, AllReduce and ReduceScatter.

### System and rack

- Host interface **16× PCIe Gen5 (64 GB/s)**; **1 CPU with 512 GB RAM per MTIA 300** (1:1 host:accelerator, vs 8 accelerators per host on the H100/H200 testbeds) — chosen to avoid PCIe contention.
- **Scale-up: 16 accelerators = 1 rack at 800 GB/s per accelerator** (option up to 1,000 GB/s). Multiple racks can be combined into larger scale-up domains.
- **Scale-out: 200 GB/s per accelerator**, 4,096-node first-level domain ("L2 unlimited" in Table I; "16K nodes or more if needed" in the text).
- Chassis: **16 compute blade slots + 6 network blade slots** on a pair of cable backplanes (not all network blades need be installed; compute blades mounted vertically to shrink the backplane). Two network blade types — **scale-up** (ASIC chosen for low latency and power) and **scale-out** (disaggregated scheduled fabric with packet spray, in-order delivery, fabric-level end-to-end credit). Both blade types liquid-cooled.

### Measured results — read the H100 caveat

- **1.42× higher Perf/TCO than H100** on a production DLRM of approximately 150 billion parameters (99% of parameters in embeddings) at local batch 10,240 on 24 MTIA 300; **1.39×** at batch 6,144 on 40 MTIA 300.
- **Communication/collective time 3.9× better than H100**; **embedding operators up to 2.5× H100**.
- ⚠️ **The H100 baseline is a custom 500 W-power-capped configuration** delivering 780 TF/s BF16 instead of 1,000 TFLOPS at 700 W (Table III). Every citation of the 1.42× / 1.39× / 3.9× figures must carry this qualifier or it materially overstates the advantage. Table III testbeds: MTIA 300 = 560 TF/s BF16, 216 GB, 6.1 TB/s, 912 W accel / 1,500 W host, 1 accel/host, 16-accel scale-up @800 GB/s, 200 GB/s scale-out; H100 = 780 TF/s, 96 GB, 2.4 TB/s, 500 W / 6,500 W, 8 accel/host, 8-accel scale-up @450 GB/s, 50 GB/s scale-out; H200 = 1,000 TF/s, 141 GB, 4.8 TB/s, 700 W / 8,850 W.

### LLM inference

Evaluated on **DeepSeek-R1 via vLLM under the InferenceMax benchmark**, 8-accelerator TP8-TP8 and DP8-EP8 configurations, BF16 attention/KV-cache with FP8 MoE, short-prompt online scenario with input and output lengths drawn independently and uniformly from [0.8×1024, 1024] tokens, concurrency swept 4 → 256. Meta reports MTIA 300 "outperforms H200 overall on the InferenceMax benchmark", specifically **at concurrency above 64** where execution is decode-dominant and HBM bandwidth dominates. At low concurrency it loses to NVLink-connected H200 because of higher small-message communication overhead.

### Stated limitations (Meta's own)

- **MX4 and NVFP4 with row-wise/block-wise scaling are not natively supported** on MTIA 300 and fall back to RISC-V execution. Meta states native MX4 support was added in **MTIA 400**.
- **Eager mode is host-bottlenecked** (Python interpreter, dynamic dispatch, device-host communication) and does not scale with faster silicon.
- Numerical-parity tooling for cross-platform convergence debugging exists but "still require[s] time to mature".

### Software stack changes (see also layer table)

PyTorch-native with **FSDP2, DTensor, TorchRec, XFormers**; TorchDynamo forward-graph capture, **AOTAutograd** backward generation, TorchInductor/Triton codegen; a **CUDA-like runtime API** supporting both eager and graph modes; kernels in C++ or Triton; coding agents used for automated kernel generation; compiler performs **ILP-based graph scheduling** for peak-memory reduction and **activation rematerialization**.

New named component: **HCCL**, Meta's collective communications library. HCCL constructs work packets and subgraphs of WQEs (SEND / RECV / WRITE / WAIT / SET / REDUCE) dispatched by the control core (**CPU-C**) to the 16 Message Engines, and drives the NIC chiplets through RDMA verbs (`rdma_core`, `ibv_create_qp`). Collectives traced via `torch.export` / `torch.compile` can live in the **same graph as compute**. Dynamic-shape collectives and device-resident AllToAll with dynamic send/recv counts remain WIP.

### Ratified by primary source

The ISCA 2026 paper states MTIA 2i (a.k.a. MTIA 200) is **"now deployed at a scale of hundreds of thousands of chips"** — confirming the scale figure already recorded in this summary.

### Status verbs as of 2026-08-08

| Generation | Status |
|---|---|
| MTIA 2i / 200 | **Deployed at scale** — hundreds of thousands of chips (primary-confirmed) |
| MTIA 300 | **In production** for ranking/recommendation training, with published production-scale measured results; not claimed at MTIA-2i fleet scale |
| MTIA 400 | **Lab-tested only** — "finished testing MTIA 400 in our labs and are on the path to deploying it"; no confirmed in-datacenter deployment |
| MTIA 450 / 500 | **Announced and scheduled only** (early 2027 / 2027) |

### Other developments since the 2026-04-05 baseline

**Broadcom partnership formalized 2026-04-14** (Meta newsroom). Broadcom will work with Meta "across chip design, advanced packaging, and networking", built on Broadcom's XPU platform, with Broadcom Ethernet for cluster networking, under "a commitment that exceeds 1GW, which is the first phase of a sustained, multi-gigawatt rollout". The release states **no process node and no end year** — the widely circulated "2nm" and "through 2029" figures appear only in aggregators and are **not published here**.

**Production-timing signal (secondary, medium-low confidence).** Reuters, 2026-07-09, citing an internal Meta memo, reports Meta plans to begin manufacturing an in-house AI chip code-named **"Iris"** in September 2026, with testing completed in roughly six weeks and no major issues, as part of a plan to reach about 7 GW of compute by end-2026 and 14 GW the following year (TechCrunch and Yahoo Finance carried the same memo). Several secondary outlets map Iris to MTIA 400; that **generation mapping is aggregator-level and not confirmed by any primary source**.

**Additional Meta ISCA 2026 industry-track papers** (Session 4B "Industry Track 1", Monday 2026-06-29, 16:30–18:10 EDT): "KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta" (17:50–18:10) — previously cited in this repo only as an uncited bullet — plus "Vistara: Making CXL Real" and "From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs". Meta's publications index also lists "LoKA: Low-precision Kernel Applications for Recommendation Models At Scale" [ISCA'26] and "Triton for MTIA: Bridging the Programming Model Gaps for Custom AI Accelerators" [IEEE Micro 2026].

**Upcoming — disclosure scheduled, not yet public.** Hot Chips 38, Session "AI 1", Tuesday **2026-08-25**, 2:15–4:15 PM — Meta, "Meta's Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI" (Srinagesh Loke, Cindy Chen, Jatinder Singh). **Content not yet public**; the listed title does not contain "MTIA", so MTIA coverage is a reasonable inference but unconfirmed until the talk. Re-scan after 2026-08-25.

### Resolved: MTIA v2 / 2i PCIe width

**There was no conflict between the papers — only between the papers and this repo.** The **ISCA 2025** spec table gives *Host connection: 8× PCIe Gen5 (32 GB/s)* for MTIA 2i and *8× PCIe Gen4 (16 GB/s)* for MTIA v1, and describes each accelerator module as two MTIA 2i chips "connected via two 8× Gen5 PCIe links". **ISCA 2026** Table I says the same thing. This repo's 2026-04-05 entry of "PCIe Gen5 ×16 (128 GB/s)" was wrong on both lane count and bandwidth — 128 GB/s matches no Gen5 configuration Meta describes — and has been corrected throughout on 2026-08-08. MTIA 300's 16× Gen5 (64 GB/s) is a real doubling of host lanes over v2.

### New sources (2026-08-08)

- https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf — ISCA 2026 MTIA 300 paper (primary)
- https://aisystemcodesign.github.io/ — Meta AI system co-design publications index
- https://iscaconf.org/isca2026/program/ — ISCA 2026 program (Sessions 4B, 5B)
- https://www.computer.org/csdl/proceedings-article/isca/2026/506500b084/2iG0QI12UeY — IEEE CSDL proceedings entry
- https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/ — Meta blog "Four MTIA Chips in Two Years", 2026-03-11
- https://about.fb.com/news/2026/04/meta-partners-with-broadcom-to-co-develop-custom-ai-silicon/ — Meta newsroom, 2026-04-14
- https://techcrunch.com/2026/07/09/metas-new-ai-chips-will-begin-production-in-september/ — "Iris" production report, 2026-07-09 (secondary)
- https://hotchips.org/program/conference/ — Hot Chips 38 program (scheduled talk, content not yet public)
