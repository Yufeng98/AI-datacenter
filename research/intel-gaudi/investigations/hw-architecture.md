# Intel Gaudi Hardware Architecture Investigation

*chip: intel-gaudi*
*as_of: 2026-09-13 (baseline investigation 2026-04-05; see dated update sections at end)*
*sources: Gaudi 3 White Paper (Intel, 2024), Hot Chips 2024 presentation, docs.habana.ai v1.24.0, VideoCardz, OAM Product Brief*

---

## Overview

Intel Gaudi 3 (HL-325L) is a dual-die AI accelerator architected specifically for large language model training and inference. Each accelerator integrates eight Matrix Multiplication Engines (MME) and 64 Tensor Processing Cores (TPC) for compute, 96 MB of on-die SRAM (12.8 TB/s bandwidth) serving as a shared scratchpad between MME and TPC engines, 128 GB of HBM2e (3.7 TB/s) for model weights and activation storage, and 24 integrated 200 GbE RoCE v2 NIC ports (21 scale-up + 3 scale-out) implemented on-die—eliminating the need for a discrete networking card. The resulting architecture is explicitly designed to keep all execution engines (MME, TPC, NIC, DMA) active in parallel, with the SRAM acting as the high-bandwidth coordination point between them.

---

## Architecture

### Generation Comparison

| Specification | Gaudi 1 | Gaudi 2 | Gaudi 3 |
|---|---|---|---|
| Die count | 1 | 1 | 2 (dual-die) |
| MME engines | 2 | 2 | 8 |
| TPC engines | 8 | 24 | 64 |
| SRAM | ~32 MB | 48 MB | 96 MB (12.8 TB/s) |
| HBM | 32 GB HBM2 | 96 GB HBM2e (2.45 TB/s) | 128 GB HBM2e (3.7 TB/s) |
| BF16 TFLOPs (matrix) | ~95 | ~432 | ~1835 |
| FP8 TFLOPs | — | — | ~1835 |
| Integrated NIC ports | 10 × 100 GbE | 21 × 100 GbE | 24 × 200 GbE |
| Scale-up ports | 10 | 21 | 21 × 200 GbE |
| Scale-out ports | 0 (external NIC) | 3 | 3 × 200 GbE |
| PCIe | Gen4 | Gen4 | Gen5 x16 |
| OAM TDP (air/liquid) | ~300 W | ~600 W | 900 W / 1200 W |

### Hardware Layers

#### 1. Compute Engine (Dual Engine: MME + TPC)

**MME (Matrix Multiplication Engine):**
- Handles all operations reducible to matrix multiplication: GEMM, convolution, batched-GEMM, attention dot-products
- Gaudi 3 has **8 MME engines** (4 per compute die)
- Each MME operates on configurable tile sizes; data layouts managed by the SynapseAI compiler
- Delivers 1835 BF16 matrix TFLOPs per card (OAM)

**TPC (Tensor Processing Core):**
- **VLIW SIMD processor** purpose-built for non-GEMM deep learning operations
- **256-byte vector width** (64 × FP32 or 128 × BF16/FP16 or 256 × FP8 lanes per cycle)
- VLIW instruction has **4 pipeline slots**: Vector (SIMD), Scalar, Load, Store
- Load and Store slots include an **Address Generation Unit (AGU)** supporting 5-dimensional tensor addressing natively
- Supports: FP32, BF16, FP16, FP8 (E4M3 and E5M2), INT32, INT16, INT8
- Gaudi 3 has **64 TPC engines** (32 per compute die), each independently programmable
- Programmed via TPC-C (C99 + intrinsics) compiled with the LLVM-based `tpc-clang`

#### 2. Data Path

- **MME → SRAM → TPC pipeline**: MME writes GEMM results to SRAM; TPC reads from SRAM for subsequent elementwise operations (e.g., Linear + activation = one pipeline stage)
- **DMA engines**: Asynchronous DMA transfers between HBM2e and SRAM; support double-buffering so compute and data movement overlap
- **NIC ↔ HBM**: Integrated RoCE NICs read/write directly from HBM2e for collective communication (RDMA); no host CPU involvement required
- All engines (MME, TPC, NIC, DMA) can execute **fully in parallel** when the workload allows; the SynapseAI graph compiler is responsible for scheduling this parallelism

#### 3. On-chip Memory (SRAM)

- **96 MB total** (48 MB per compute die), organized as L2/L3 cache hierarchy on Gaudi 3
- **12.8 TB/s aggregate bandwidth** — the primary high-bandwidth scratchpad
- Shared between MME engines and TPC engines on each die
- Serves as: activation buffer between pipeline stages, operand staging for TPC kernels, prefetch buffer for upcoming GEMM tiles
- TPC processors also have small **local memory** (per-TPC scratchpad) in addition to shared SRAM access

#### 4. Off-chip Memory (HBM2e)

- **128 GB HBM2e** (Gaudi 3), comprising 8 HBM2e stacks
- **3.7 TB/s bandwidth** (vs. 2.45 TB/s on Gaudi 2 — ~51% improvement)
- Stores: model weights, optimizer states, activations that don't fit in SRAM, gradient tensors
- Gaudi 2 reference: 96 GB HBM2e at 2.45 TB/s

#### 5. Host Interface

- **PCIe Gen5 x16**: ~128 GB/s bidirectional peak bandwidth (Gaudi 3)
- Gaudi 2 used PCIe Gen4 x16 (~64 GB/s bidirectional)
- **OAM (Open Accelerator Module)** form factor (HL-325L): the primary multi-card configuration; 8 Gaudi 3 OAMs per HGX-compatible tray
- **PCIe card** form factor (HL-338) also available for single-card deployments
- Commands submitted from host via PCIe; DMA descriptors programmed through PCIe-mapped registers

#### 6. Scale-up Interconnect (Integrated RoCE)

- **21 × 200 GbE RoCE v2 ports** dedicated to scale-up (intra-node and rack-level) on Gaudi 3
- Ports implemented as **on-die SerDes** (48 pairs of 112 Gbps PAM4 Tx/Rx) — no discrete NIC chip needed
- Supports **direct chip-to-chip connections** (within a server tray) or connections through standard Ethernet switches
- Total scale-up bandwidth: **21 × 200 Gbps = 4.2 Tbps** per card (unidirectional); full bisection possible with switch
- RoCE v2 enables RDMA directly to/from HBM2e, bypassing host CPU for collective communication
- For an 8-Gaudi server tray, the 21 scale-up ports are typically partitioned: some for all-to-all intra-node connectivity, the remainder for top-of-rack switch uplinks

#### 7. Scale-out Interconnect

- **3 × 200 GbE RoCE v2 ports** dedicated to scale-out (inter-rack, multi-node clusters) on Gaudi 3
- Same integrated on-die SerDes technology as scale-up ports
- Connect to external Ethernet switches (e.g., Cisco Nexus 9000, Arista) for multi-rack clusters
- HCCL transparently routes collective operations across scale-up and scale-out ports
- Host NIC (via GDR / peer-direct) is also supported as an alternative scale-out path

---

## Data Flow

### Training Forward Pass (single Gaudi 3 card)

1. **Weight load**: DMA fetches weight tensors from HBM2e → SRAM (pre-staged by SynapseAI runtime)
2. **MME execution**: GEMM tile computation reads from SRAM, writes result to SRAM
3. **TPC execution**: Activation function kernel (e.g., GELU) reads GEMM output from SRAM, applies SIMD computation, writes result to SRAM
4. **SRAM eviction**: If activation doesn't fit in 96 MB SRAM, DMA writes it back to HBM2e
5. **NIC activity** (if data-parallel): HCCL AllReduce triggers RDMA operations directly from HBM2e across RoCE ports — overlapped with compute of next batch

### AllReduce Data Path

```
TPC computes gradient → stored in HBM2e
HCCL runtime → programs integrated RoCE NIC descriptors
RoCE NIC → RDMA READ/WRITE directly on peer HBM2e (no CPU copy)
Reduced gradient → written back to local HBM2e
Optimizer step → reads from HBM2e, updates weight in HBM2e
```

---

## Hardware Interface

- **SynapseAI compiler → MME**: Emits GEMM descriptor packets (tile sizes, precision, strides) consumed by MME controller
- **SynapseAI compiler → TPC**: Emits VLIW binary + tensor descriptor (address, stride, 5D dims) for each TPC invocation
- **SynapseAI runtime → DMA**: Programs DMA transfer descriptors for HBM↔SRAM movement
- **HCCL → integrated NIC**: Programs RoCE QP (Queue Pair) descriptors for RDMA operations; NIC directly accesses HBM2e physical addresses
- **habanalabs kernel driver**: Manages PCIe BAR mappings, command queues, device MMU (SMMU for HBM address translation)

---

## Key Findings

1. **Dual-die design doubles TPC and MME count**: Gaudi 3's two compute dies give 8 MMEs and 64 TPCs vs. Gaudi 2's 2 MMEs and 24 TPCs — a ~4× compute scaling at 2× the die area.
2. **On-die RoCE is architecturally unique**: No other major AI accelerator (H100, MI300X as of 2024) integrates Ethernet NICs on-die. This eliminates NVLink/InfiniBand dependency and allows any Ethernet switch fabric to serve as the scale-up interconnect.
3. **SRAM bandwidth exceeds HBM bandwidth by 3.5×**: At 12.8 TB/s vs. 3.7 TB/s, SRAM is the performance-critical tier; the compiler's SRAM management strategy is the primary lever for compute utilization.
4. **FP8 support is native in TPC**: The TPC supports E4M3 and E5M2 FP8 formats natively, enabling direct FP8 training without software emulation.
5. **PCIe Gen5 reduces host-transfer bottleneck**: The jump from Gen4 to Gen5 doubles host bandwidth to ~128 GB/s, relevant for large batch loading and frequent parameter server patterns.

---

## Relation to Hardware Architecture

Gaudi 3 is designed around the principle that networking, compute, and memory should all be co-equal first-class resources on the die. The 24 integrated NIC ports consume a significant fraction of die area alongside the MME and TPC clusters, reflecting Intel's view that inter-chip communication is as performance-critical as raw FLOPs. The shared SRAM scratchpad (rather than per-engine private caches) reflects a software-managed memory philosophy where the graph compiler—not hardware prefetch logic—determines data placement.

---

## Gaudi 4 / Falcon Shores Roadmap

*as_of: 2026-04-05*
*sources: Intel Q4 2024 earnings call (Jan 2025), Tom's Hardware, TechRadar, Tweaktown, TrendForce, HPCwire, Intel Newsroom*

### Summary of Intel's Post-Gaudi 3 Strategy

There is no product called "Gaudi 4." After Gaudi 3 (current generation), Intel's publicly known roadmap progresses through three distinct phases, each representing a significant strategic shift from the prior generation.

### Phase 1: Falcon Shores — Cancelled (announced Jan 2025)

Falcon Shores was Intel's originally planned successor to Gaudi 3. It was described as a multi-chiplet design combining Xe-HPC (Xe3-HPC) GPU chiplets with Gaudi-derived AI acceleration in a single product — Intel's attempt at a hybrid GPU+ASIC accelerator. The plan was to ship Falcon Shores in late 2025.

**Cancellation:** Intel CEO Pat Gelsinger's replacement, Lip-Bu Tan, along with acting CEO Michelle Johnston Holthaus, announced on the Q4 2024 earnings call (January 2025) that Falcon Shores would not come to market. The decision was driven by customer feedback indicating the product was not what enterprises wanted, and Intel's acknowledged inability to compete in cloud-based AI data center markets meaningfully. Falcon Shores was demoted to an internal test chip only.

**Quote (Intel, Jan 2025):** *"We're not yet participating in the cloud-based AI data center market in a meaningful way."*

### Phase 2: Crescent Island — Inference GPU, sampling H2 2026

Announced at OCP Global Summit (October 2025), Crescent Island is Intel's near-term successor to Gaudi 3 for inference workloads. It is a departure from the Gaudi ASIC architecture — it is a GPU product based on the Xe3P (performance-optimized) microarchitecture, the same Xe3 used in Panther Lake mobile processors.

| Property | Crescent Island |
|----------|----------------|
| Codename | Crescent Island |
| Architecture | Xe3P (GPU, not Gaudi ASIC) |
| Memory | 160 GB LPDDR5X |
| Memory choice rationale | Cost per GB optimized over bandwidth — targets inference capacity |
| Compute | XMX (Xe Matrix eXtensions) with FP8, FP4, MXFP4, MXFP8 support |
| Cooling | Air-cooled enterprise servers (no liquid cooling required) |
| Target market | "Tokens-as-a-service" inference providers |
| Customer sampling | H2 2026 |
| TDP / TFLOPs | Not disclosed as of April 2026 |
| Relationship to Gaudi | Separate GPU product line; does not carry the Gaudi ASIC architecture |

**Key architectural divergence from Gaudi:** Crescent Island uses LPDDR5X instead of HBM, targets inference-only workloads, and is positioned as a cost-optimized part — not a training-capable accelerator. It does not appear to carry Gaudi's on-die RoCE NIC architecture.

### Phase 3: Jaguar Shores — Rack-scale Gaudi successor, ~2027

Jaguar Shores is Intel's next Gaudi-branded AI accelerator, revealed informally in mid-2025 and confirmed by Intel to carry the Gaudi brand. It represents a full architectural reset toward rack-scale disaggregation.

| Property | Jaguar Shores |
|----------|--------------|
| Codename | Jaguar Shores |
| Process node | Intel 18A |
| Memory | HBM4 (SK Hynix), rumored HBM4E variant |
| Memory BW | ~2 TB/s per stack (HBM4 class) |
| Package | ~92.5 mm × 92.5 mm; quad-tile configuration; octal HBM stacks |
| Interconnect | Silicon photonics for optical rack-level interconnects |
| CPU pairing | Designed to be combined with Diamond Rapids Xeon CPU |
| Architecture scope | Rack-scale disaggregated compute + memory (not a single-chip design) |
| Expected availability | 2027 at earliest; possible late 2027 or 2028 |
| TFLOPs / NIC details | Not yet disclosed |

**Architectural significance:** Jaguar Shores disaggregates compute and memory resources across an entire rack, rather than integrating everything on a single package as Gaudi 3 does. The use of silicon photonics for inter-rack connectivity marks a departure from Gaudi 3's copper RoCE approach. The 18A process node (Intel's most advanced gate-all-around node) would be Intel's first return to leading-edge fabrication for an AI accelerator since Ponte Vecchio (Intel 7).

### Roadmap Timeline

| Year | Product | Architecture | Status |
|------|---------|-------------|--------|
| 2024 (shipping) | Gaudi 3 (HL-325L) | Gaudi ASIC, dual-die, HBM2e | GA |
| 2025 (cancelled) | Falcon Shores | Xe-HPC + Gaudi hybrid, multi-chiplet | Cancelled — internal test chip only |
| H2 2026 (sampling) | Crescent Island | Xe3P GPU, LPDDR5X, inference-only | Customer sampling |
| ~2027 | Jaguar Shores | Gaudi-branded, 18A, HBM4, rack-scale | Pre-announcement / rumor stage |

### Strategic Context

Intel's post-Gaudi 3 trajectory reflects repeated strategy pivots under significant competitive pressure from NVIDIA and AMD:

1. **Falcon Shores cancellation** eliminated Intel's only planned training-class GPU successor to Gaudi 3, leaving a multi-year gap in Intel's high-end training portfolio.
2. **Crescent Island** addresses the nearer-term inference market (where Gaudi 3 lacked competitive positioning against NVIDIA H100/H200 for inference TCO), but does so with a GPU architecture rather than a specialized ASIC.
3. **Jaguar Shores** is positioned as the long-term Gaudi lineage successor for training-class workloads, but at 2027+ timing and with limited public technical detail, it remains speculative from a competitive standpoint.
4. **Software continuity risk**: Whether SynapseAI / HCCL / the Gaudi software stack extends to Crescent Island (Xe3P GPU) and Jaguar Shores (rack-scale) is not publicly confirmed. Crescent Island's GPU architecture suggests it may use a different software path (potentially Xe GPU drivers rather than SynapseAI).

### Sources

- [Intel cancels Falcon Shores GPU — Jaguar Shores successor (Tom's Hardware)](https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-cancels-falcon-shores-gpu-for-ai-workloads-jaguar-shores-to-be-successor)
- [Intel kills off Falcon Shores, moves to rack-scale with Jaguar Shores (Tweaktown)](https://www.tweaktown.com/news/102912/intel-kills-off-falcon-shores-ai-chip-moves-to-rack-scale-solution-with-jaguar/index.html)
- [Intel unveils Crescent Island inference GPU (Tom's Hardware)](https://www.tomshardware.com/pc-components/gpus/intel-unveils-crescent-island-an-inference-only-gpu-with-xe3p-architecture-and-160gb-of-memory)
- [Intel to Expand AI Accelerator Portfolio with New GPU (Intel Newsroom)](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu)
- [Jaguar Shores 18A + HBM4 rack-scale (TrendForce)](https://www.trendforce.com/news/2025/08/21/news-intels-jaguar-shores-reportedly-breaks-cover-showcasing-18a-and-hbm4-on-rack-scale-ai-solution/)
- [Jaguar Shores next-gen Gaudi with HBM4E (Tweaktown)](https://www.tweaktown.com/news/109493/intels-next-gen-jaguar-shores-gaudi-ai-accelerator-rumored-to-use-new-hbm4e-memory/index.html)
- [Intel quietly adds Jaguar Shores to Gaudi roadmap (TechRadar)](https://www.techradar.com/pro/intel-quietly-adds-jaguar-shores-to-its-gaudi-ai-accelerator-roadmap-as-it-seeks-to-compete-more-fiercely-against-amd-and-nvidia)
- [Intel roadmaps 2026-2028 (Tom's Hardware)](https://www.tomshardware.com/tech-industry/semiconductors/intel-chip-roadmap-2026-2028)
- [Intel Crescent Island at OCP 2025 (StorageReview)](https://www.storagereview.com/news/intel-targets-ai-inference-at-ocp-2025-with-crescent-island-gpu-and-gaudi-3-racks)
- [Intel axes Falcon Shores amid market challenges (TechTarget)](https://www.techtarget.com/searchdatacenter/news/366618595/Intel-axes-Falcon-Shores-amid-market-challenges)

---

## Update — Crescent Island Computex 2026 disclosure and roadmap re-verification (2026-08-08)

*Window covered: 2026-04-05 → 2026-08-08. Verification pass 2026-08-08.*

### Scope of change

**Gaudi 3 silicon is unchanged.** No Gaudi hardware disclosure, no revised Gaudi 3 specification, and no Intel first-party discontinuance notice appeared in this window. Every Gaudi 1 / Gaudi 2 / Gaudi 3 figure in the sections above stands as previously recorded. The material items are (a) an expanded Crescent Island spec sheet and (b) two corrections to previously recorded claims.

### Crescent Island — Computex 2026 (2026-06-01/02)

| Property | Prior record (2026-04-05) | Updated (2026-08-08) | Confidence |
|---|---|---|---|
| Memory capacity | 160 GB LPDDR5X | 160 GB LPDDR5X reference design; **up to 480 GB LPDDR5X** in partner/ODM configurations | medium (trade press) |
| Memory bandwidth | not recorded | **not disclosed** | — |
| Power | not disclosed | **350 W** | medium (trade press) |
| Cooling / form factor | "air-cooled enterprise servers" | **air-cooled PCIe add-in card**, explicitly not liquid-cooled | medium (trade press) |
| Datatypes | FP8 / FP4 / MXFP4 / MXFP8 | **FP4 through FP64** | medium (trade press) |
| Schedule | customer sampling H2 2026 | customer sampling H2 2026 (unchanged); **general availability 2027** | medium (trade press) |
| Peak throughput, Xe3P core/XMX counts, process node, package | not disclosed | still **not disclosed** | — |

**Why the confidence is capped at medium.** Intel's own Computex 2026 materials were checked directly: the press kit (`newsroom.intel.com/press-kit/press-kit-intel-at-computex-2026`), "Intel Announces New AI Innovations at Computex" (2026-06-01) and "Computex 2026: An Intelligent World Built on Silicon" (2026-06-02) **contain no mention of Crescent Island**. The newsroom AI post index for May–Aug 2026 likewise contains no Crescent Island or Gaudi post. Every 480 GB / 350 W / FP4–FP64 figure traces to trade coverage of an Intel briefing/slide. Corroboration is broad (Tom's Hardware, VideoCardz, TechSpot, TechTimes, Neowin, Wccftech) but single-origin.

**Figures deliberately excluded from the repo:**
- **"~684 GB/s memory bandwidth."** Not an Intel figure. TechSpot writes bandwidth "is expected to reach 684 GB/s"; MLQ.ai calls it "estimated"; codeoxi notes Intel emphasised capacity, TDP and architecture rather than a headline bandwidth figure; acceleratedcomputing.ai states bandwidth was "not published". It is a press back-calculation from an assumed LPDDR5X bus width (640-bit × 10.7 Gbps). Recorded everywhere as "not disclosed".
- **"20 × 24 GB modules."** Press reconstruction with no Intel attribution. Excluded entirely.

### Crescent Island software path — prior open question resolved

The Strategic Context section above listed "whether SynapseAI / HCCL / the Gaudi software stack extends to Crescent Island" as an open risk. It is now effectively answered: **it does not.** Crescent Island rides Intel's unified Xe path — oneAPI/SYCL over the Level Zero Compute Runtime — with enablement landed in public source trees ahead of the silicon:

- Intel Compute Runtime **26.01.36711.4** (2026-01-14) — early Crescent Island support
- Intel Graphics Compiler **v2.27.10** (Jan 2026) — initial Crescent Island support
- Intel Compute Runtime — Crescent Island support **promoted 2026-04-20**

Intel stated at OCP (Oct 2025) that the unified stack is being developed and validated on Arc Pro B-series GPUs and will extend to Xe3P.

### Correction — Gaudi 3 is not upstreamed in Linux

Verified directly against mainline: `drivers/accel/habanalabs/` contains `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/` — **no `gaudi3/` directory in Linux master as of 2026-08-08**. Intel posted Gaudi 3 driver code for upstreaming in late Nov 2025 and sent a pull request targeting v6.19 (dri-devel, Dec 2025); it was rejected on code-quality grounds (reverts and build artifacts in a ~300k-line series) and now targets a later cycle. Gaudi 3 deployment requires the out-of-tree habanalabs driver.

### Benchmark signal — Gaudi absent from consecutive MLPerf rounds

- MLPerf **Inference v6.0**, 2026-04-01 (four days before the prior baseline; not a window event): 24 submitting orgs; Intel submitted Xeon 6 CPUs and Arc Pro B70/B65/B60 GPUs only. Intel's own newsroom post does not mention Gaudi.
- MLPerf **Training v6.0**, 2026-06-16 (in window): 24 submitting orgs; **Intel absent entirely**.

Recorded as an observation, not an Intel statement. Combined with the archived SynapseAI reference implementation, it is consistent with Gaudi 3 in maintenance/harvest mode — but **no Intel first-party EOL notice exists**, Gaudi 3 remains GA through cloud/enterprise channels, and software releases continue (v1.24.0). Characterize as strategically deprioritized, not end-of-life.

### Jaguar Shores — unchanged

No primary-source information in the window. Trade reporting continues to place it at ~2027 (rack-scale, silicon photonics, HBM4, Intel 18A). "Design finalization by mid-2026" claims are supply-chain sourcing, not Intel statements. Jaguar Shores has no Hot Chips 38 slot. All timing detail stays LOW confidence.

### Pending primary source

Intel, "**Crescent Island: GPU Designed for Agentic AI Inference**" (Sumit Mohan, Hong Jiang) — **disclosure scheduled, Hot Chips 38, Mon 2026-08-24, 16:45–18:45, Day-1 GPU session; content not yet public.** Must not be cited as the source of any specification. Expected to settle TFLOPS, Xe3P core counts, process node and real memory bandwidth. Re-scan after 2026-08-25.

### Sources for this update

- [mainline `drivers/accel/habanalabs/` — no gaudi3 directory](https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs)
- [dri-devel: "accel/habanalabs: Gaudi3 support and updates for v6.19" (Dec 2025, rejected)](https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html)
- [Intel Crescent Island at Computex — 480 GB, 350 W, FP4–FP64; bandwidth not published (acceleratedcomputing.ai)](https://acceleratedcomputing.ai/news/2026-06-01-intel-crescent-island/)
- [Crescent Island: 160 GB stock / up to 480 GB ODM, no headline bandwidth figure (codeoxi)](https://codeoxi.com/blog/intel-crescent-island-gpu)
- [Intel Crescent Island supports up to 480 GB LPDDR5X (VideoCardz)](https://videocardz.com/newz/intel-crescent-island-gpu-officially-supports-up-to-480gb-lpddr5x-memory)
- [Crescent Island up to 480 GB; bandwidth "expected to reach 684 GB/s" — an estimate (TechSpot)](https://www.techspot.com/news/112608-intel-crescent-island-gpu-support-up-480gb-lpddr5x.html)
- [Intel at Computex 2026 press kit — no Crescent Island mention](https://newsroom.intel.com/press-kit/press-kit-intel-at-computex-2026)
- [Computex 2026: An Intelligent World Built on Silicon (2026-06-02) — no Crescent Island mention](https://newsroom.intel.com/artificial-intelligence/computex-2026-an-intelligent-world-built-on-silicon)
- [Intel Compute Runtime 26.01.36711.4 — early Crescent Island support (Phoronix, 2026-01-14)](https://www.phoronix.com/news/Intel-CR-26.01.36711.4)
- [IGC v2.27.10 — initial Crescent Island support (TechPowerUp, 2026-01-15)](https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler)
- [MLPerf Inference v6.0 results (2026-04-01)](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/)
- [Intel Delivers Open, Scalable AI Performance in MLPerf Inference v6.0 (2026-04-01) — no Gaudi mention](https://newsroom.intel.com/artificial-intelligence/intel-delivers-ai-performance-mlperf-inference-v6-0)
- [MLPerf Training v6.0 results (2026-06-16) — Intel absent](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/)
- [Hot Chips 38 program — Intel Crescent Island talk, 2026-08-24 (scheduled; content not yet public)](https://hotchips.org/program/conference/)

---

## Update — Crescent Island Hot Chips 38 disclosure (2026-09-13)

*Window covered: 2026-08-08 → 2026-09-13. Primary event: Intel, "Crescent Island: GPU Designed for Agentic AI Inference" (Sumit Mohan, Hong Jiang), Hot Chips 38, Day-1 GPU session, Mon 2026-08-24. Sources: ServeTheHome talk coverage (2026-08-24) and Chips and Cheese architectural deep-dive (2026-08-27).*

### A. Scope of change

The pending disclosure flagged in the 2026-08-08 update has now happened. Gaudi 1/2/3 silicon is still unchanged — nothing in this window touches Gaudi hardware. The entire update is Crescent Island's Hot Chips 38 spec disclosure, which resolves most (not all) of the "not disclosed" items carried since Computex 2026.

### B. Newly Intel-disclosed architecture (via the Hot Chips 38 talk, reported by ServeTheHome)

| Property | Value | Confidence |
|---|---|---|
| GPU architecture | **Xe3P** | confirmed (Intel talk) |
| Xe core count | **32 Xe cores** | confirmed |
| XMX (matrix) engines | **256** ("32 Xe cores feeding 256 XMX engines") | confirmed |
| XMX design | "3-way extended Xe matrix" with FP4 precision co-issue and FP64 support | confirmed |
| Systolic array depth | **16-deep** | confirmed |
| Per-core register file (GRF) | **1 MB** | confirmed |
| Per-core L1 | **512 KB** | confirmed |
| L2 cache | **32 MB unified** | confirmed |
| Memory (reference card) | **160 GB LPDDR5X** | confirmed (unchanged from Computex) |
| Memory (ODM partner cards) | up to **480 GB LPDDR5X** | confirmed (unchanged from Computex) |
| Datatypes | **FP4 and MXFP4 through FP64** | confirmed |
| Power | **350 W, air-cooled PCIe GPU** | confirmed (unchanged from Computex, now Intel-stated at a public talk rather than only trade-press-reported) |
| Active-idle power | **50 W or less in the G0 state** | confirmed — first disclosure |
| Low-power idle | ~10 W in G8 | confirmed — first disclosure |
| Host interface | **PCIe Gen5 x16, scale-up** | confirmed — first disclosure |
| Reliability | ECC + parity across key memory; error checking on every hop of the internal IP fabric; dynamic page offlining; hard post-package repair | confirmed — first disclosure |
| Memory bandwidth | **still not Intel-disclosed** — ServeTheHome's Patrick Kennedy explicitly notes Intel withheld this figure | not disclosed |
| Sampling / GA dates | **not restated in the talk materials** per Chips and Cheese; prior Computex figures (customer sampling H2 2026, GA 2027) stand unconfirmed by this talk | unchanged, medium confidence |

This talk is Intel's own conference presentation — a materially stronger source than the Computex trade-press coverage this entry previously relied on for 160/480 GB, 350 W and FP4–FP64. Confidence on those three items is raised from "medium, trade-press" to "confirmed, Intel-disclosed at a public Intel talk."

### C. Chips and Cheese performance ESTIMATES — explicitly not Intel figures

Chips and Cheese's 2026-08-27 architectural analysis publishes TFLOP/s estimates modeled at an **assumed 2.5 GHz clock** with **quadrupled XMX matrix-op rate over Xe2/Xe3**. These are **analyst estimates, not Intel-disclosed numbers**:

| Datatype | Estimated throughput |
|---|---|
| FP64 (vector) | 10.2 TFLOP/s |
| FP32 (vector) | 20.5 TFLOP/s |
| FP16 (vector) | ~41 TFLOP/s |
| TF32 (XMX) | 328 TFLOP/s |
| FP16/BF16 (XMX) | 655 TFLOP/s |
| FP8 (XMX) | 1.3 PFLOP/s |
| FP4/MXFP4 (XMX) | 2.6 PFLOP/s |

Chips and Cheese also estimates memory bandwidth at **over 1.5 TB/s**, reverse-engineered from PCB photos showing **20 LPDDR5X modules (12 front + 8 back)** on an assumed **1280-bit bus** at LPDDR5X-9600 — this **supersedes** the earlier, much lower Computex-era press guess of ~0.6–0.7 TB/s ("~684 GB/s"), which now looks like an underestimate given the disclosed 32 MB L2 / 256-XMX compute scale. Both bandwidth figures remain analyst estimates, not Intel numbers, and are recorded as such.

**Module-count nuance:** The 2026-08-08 update deliberately excluded a circulating "20 × 24 GB module" breakdown as an unattributed press reconstruction. Chips and Cheese's independent photo-based count also finds 20 modules total, but at 160 GB / 20 modules = **8 GB/module** for the reference card — the 24 GB/module figure would only reconcile with the 480 GB ODM configuration (20 × 24 GB = 480 GB), not the reference design. This is a plausible resolution, not a confirmation; still recorded as analyst-derived, not Intel-stated.

### D. Comparative framing (Chips and Cheese, vendor-neutral analyst comparison, not Intel claims)

Chips and Cheese frames Crescent Island's matrix compute as ~30% ahead of NVIDIA RTX PRO 6000 Blackwell, and its FP64 vector throughput as over 5× Blackwell's — while Blackwell holds 6× higher FP32 vector and 3× higher FP16 vector compute. These are third-party analyst comparisons and are recorded here as context, not as Intel-published competitive claims.

### E. What remains unresolved

- Memory bandwidth (Intel still declines to state it)
- Process node / fab
- Die/package configuration, die count
- Exact sampling and GA dates (talk did not restate the Computex H2-2026/2027 schedule)

### Sources for this update

- [ServeTheHome — Intel Crescent Island: 160GB to 480GB LPDDR5X AI GPU at Hot Chips 2026](https://www.servethehome.com/intel-crescent-island-160gb-to-480gb-lpddr5x-ai-gpu-at-hot-chips-2026/)
- [Chips and Cheese — Hot Chips 2026: Intel's Crescent Island](https://chipsandcheese.com/p/hot-chips-2026-intels-crescent-island)
- [Intel Announces New AI Innovations at Computex — Xeon 6+ / SambaNova SN-50 racks (non-Gaudi context)](https://www.intc.com/news-events/press-releases/detail/1771/intel-announces-new-ai-innovations-at-computex-chip-to)
