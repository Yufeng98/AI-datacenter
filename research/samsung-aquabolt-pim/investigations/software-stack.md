# Samsung Aquabolt-XL HBM-PIM — Software Stack Investigation

*chip: samsung-aquabolt-pim*
*investigation: software-stack*
*date: 2026-04-05*
*sources: HC33 slides, ServeTheHome HC33 software stack screenshot, Samsung newsroom, PIMSys ACM 2024*

---

## 1. Overview

Samsung designed a **PIM software stack** allowing users to run unmodified ML framework code (TensorFlow, PyTorch) on systems equipped with Aquabolt-XL. The stack intercepts memory-bound operations and offloads them to the PIM processors transparently. The stack details were presented at Hot Chips 33 and partially in Samsung semiconductor publications, but no public SDK has been released.

---

## 2. Software Stack Layers

```
┌───────────────────────────────────────────────────────┐
│  ML Frameworks: TensorFlow / PyTorch                  │  ← Unmodified user code
├───────────────────────────────────────────────────────┤
│  PIM SW Stack (Framework interception layer)          │  ← Transparent dispatch
├───────────────────────────────────────────────────────┤
│  PIM BLAS Library                                     │  ← GEMV/BLAS L2 ops
├───────────────────────────────────────────────────────┤
│  PIM Runtime / Driver Library                         │  ← Mode control, sync
├───────────────────────────────────────────────────────┤
│  PIM Mode Enable / Hardware Command Interface         │  ← HBM2 command extension
├───────────────────────────────────────────────────────┤
│  HBM2 Memory Controller                               │  ← Standard JEDEC (unchanged)
├───────────────────────────────────────────────────────┤
│  Aquabolt-XL HW (PIM-DRAM dies + standard dies)       │  ← Drop-in HBM2 stack
└───────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Framework Layer (TensorFlow / PyTorch)
- Users run **unmodified source code** using standard ML frameworks
- Samsung toolkit intercepts memory-bound ops (GEMV, matrix-vector products)
- Both TensorFlow and PyTorch supported as per Samsung's published stack
- Execution paths:
  - **Native execution**: Framework dispatches directly to PIM BLAS
  - **Standard execution**: Falls back to host CPU/GPU if PIM not available

### 3.2 PIM BLAS Library
- Provides BLAS Level 2 functions (GEMV: matrix-vector multiply)
- Equivalent to cuBLAS Level 2 but for PIM processors
- Operations computed inside HBM stack (no data movement off-chip)
- Status: **not publicly released**

### 3.3 PIM Runtime / Driver Library
- Host-side library handling:
  - PIM mode enable/disable (switching between standard DRAM and PIM compute)
  - Micro-code loading to PIM processor instruction memory
  - Synchronization between host compute (GPU/FPGA) and PIM stack
  - Memory allocation in PIM-capable addresses
- Status: **not publicly released**

### 3.4 Hardware Command Interface
- PIM mode activation via a special command sequence sent to the HBM2 logic die
- Standard HBM2 JEDEC commands unchanged (backward compatibility preserved)
- PIM processors receive instructions micro-coded by the runtime library
- No programmer-visible ISA published

---

## 4. Programming Model

- **SIMD execution**: All 128 processors receive and execute the same instruction
- Programmer (via PIM BLAS) loads weight matrix into PIM DRAM, streams activation vectors
- PIM returns accumulated result vectors via standard HBM2 read commands
- Effective for: GEMV, dot products, element-wise operations on large vectors

---

## 5. System Integration Topology

```
GPU (AMD MI60) or FPGA (Xilinx Alveo U280)
      │
      │ HBM2 interface (standard JEDEC)
      ▼
  Aquabolt-XL HBM-PIM stack
  [4× PIM-DRAM dies: 128 FP16 processors]
  [4× standard DRAM dies: storage]
```

The GPU/FPGA sees a standard HBM2 device. The PIM SW stack on the host programs the PIM processors and coordinates execution.

---

## 6. Research and Simulation Tools

| Tool | Description | Status |
|------|-------------|--------|
| PIMSys (ACM MemSys 2024) | Virtual prototype using gem5 + DRAMSys + Rust library | Research tool |
| Samsung PIM BLAS | BLAS Level 2 for PIM | not public |
| Samsung PIM SW Stack | Framework integration toolkit | not public |

---

## 7. Key Observations

1. **Drop-in transparency**: JEDEC HBM2 backward compatibility means no GPU/FPGA hardware changes; only software changes needed
2. **No ISA exposure**: Users never write PIM assembly; BLAS library abstracts all compute
3. **SIMD limitation shapes API**: API designed for bulk GEMV (all processors same instruction) — no general-purpose compute API
4. **Research ecosystem**: Unlike d-Matrix (product-focused) or SK Hynix AiM (product + academic), Samsung's PIM appears primarily as a research/prototype capability with limited productization evidence post-2021
5. **Toolchain not public**: No GitHub repos, no SDK downloads; academic virtual prototypes (PIMSys) are community-driven

---

## Sources

- [Aquabolt-XL HC33 Software Stack slide — ServeTheHome](https://www.servethehome.com/samsung-hbm2-pim-and-aquabolt-xl-at-hot-chips-33/hc33-samsung-hbm2-pim-aquabolt-xl-software-stack/)
- [Aquabolt-XL Official Slides HC33](https://www.hc33.hotchips.org/assets/program/conference/day1/20210813_HC33_Aquabolt-XL_PIM_Jin_Kim_slide.pdf)
- [Samsung Brings In-Memory Processing to Wider Range](https://news.samsung.com/global/samsung-brings-in-memory-processing-power-to-wider-range-of-applications)
- [HBM-PIM Samsung Semiconductor tech blog](https://semiconductor.samsung.com/news-events/tech-blog/hbm-pim-cutting-edge-memory-technology-to-accelerate-next-generation-ai/)
- [PIMSys Virtual Prototype — ACM MemSys 2024](https://dl.acm.org/doi/10.1145/3695794.3695797)

---
---

# Investigation Update — LPDDR5X-PIM Software Stack (2026-08-08)

*investigation: software-stack (generation 2)*
*date: 2026-08-08*
*primary source: arXiv:2606.00636v1 (Samsung Electronics), read in full*

## U1. Scope

Everything above describes the 2021 Aquabolt-XL HBM2-PIM software stack (PIM SW Stack → PIM BLAS → PIM Runtime → HBM2 command interface). That stack is unchanged and is **not** the stack described here.

Samsung's LPDDR5X-PIM generation has a **different, independently described software stack** — the **"PIM Kernel"** layer. It is documented in arXiv:2606.00636v1 and is not publicly released.

## U2. Stack shape

```
┌───────────────────────────────────────────────────────────────┐
│  Application                                                  │  ← supplies matrix shapes + dtypes
│  (no framework binding disclosed — no TF/PyTorch claim)       │
├───────────────────────────────────────────────────────────────┤
│  PIM KERNEL LAYER                            (not public)     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Data Mapper  — OFFLINE                                   │  │
│  │   PIM-aware placement of weights + scale factors         │  │
│  │   into DRAM banks, per a predefined                      │  │
│  │   "PIM Tile Configuration"; preloaded so that no          │  │
│  │   runtime rearrangement is needed                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ PIM Executor — RUNTIME (three sub-components)            │  │
│  │   1. PIM Device Code Gen                                 │  │
│  │        synthesizes IRF code + hardware config code       │  │
│  │        from matrix shapes and data types                 │  │
│  │   2. PIM Control                                         │  │
│  │        manages Single-Bank <-> Multi-Bank transitions    │  │
│  │   3. GEMV Kernel                                         │  │
│  │        per-tile GEMV on a specialized PIM ISA;           │  │
│  │        handles pipeline flush-out                        │  │
│  └─────────────────────────────────────────────────────────┘  │
├───────────────────────────────────────────────────────────────┤
│  JEDEC LPDDR5X interface (JESD209-5C)                         │
├───────────────────────────────────────────────────────────────┤
│  LPDDR5X-9600 device, 1 PIM block per DRAM bank               │
└───────────────────────────────────────────────────────────────┘
```

## U3. Component detail

### U3.1 Data Mapper (offline)

Performs **PIM-aware placement** of weights and scale factors into DRAM banks using a **predefined "PIM Tile Configuration."** Placement is **preloaded**, which is the point: it eliminates runtime data rearrangement, the classic cost that erases PIM's bandwidth advantage.

### U3.2 PIM Executor (runtime)

| Sub-component | Responsibility |
|---|---|
| **PIM Device Code Gen** | Synthesizes **IRF code** (instruction register file contents) and hardware configuration code from the matrix shape and data types of the operation |
| **PIM Control** | Manages **SB ↔ MB** mode transitions (Single-Bank standard DRAM operation ↔ Multi-Bank parallel PIM execution) |
| **GEMV Kernel** | Executes per-tile GEMV on the specialized **PIM ISA**; handles pipeline flush-out |

This is a runtime code-generation model, not an ahead-of-time compiler. There is no published IR, no virtual ISA, and no user-facing kernel-authoring language.

### U3.3 Address-mapping strategies

| Strategy | Mechanism | Purpose |
|---|---|---|
| **Vertical Mapping** | Rows interleaved across Channel / Rank / Bank Group / Bank | Spread a matrix across the full device for parallelism |
| **Horizontal Mapping** | Adjacent sub-matrices kept in the same bank | Maximize row-buffer hits |
| **Reshape Optimization** | Column-based partitioning | Near-100% intra-PIM efficiency; yields up to an additional **1.65×** for small matrices (W < 2048) |

Address mapping is therefore a **software-layer decision** in this generation, exposed as an explicit tuning knob — a notable difference from Aquabolt-XL, where placement was implicit in the pseudo-channel structure.

## U4. Framework integration — not disclosed

The Aquabolt-XL stack claimed transparent TensorFlow and PyTorch dispatch. **The LPDDR5X-PIM paper makes no such claim.** It describes an interface taking matrix shapes and dtypes; no framework binding, no graph-capture layer, no operator-dispatch interception is described. Record as **not disclosed**, not as "absent" — the paper's scope is the simulator, not the full stack.

## U5. Simulator — LP5X-PIM Sim

| Property | Value |
|---|---|
| Name | LP5X-PIM Sim |
| Kind | Cycle-accurate HW/SW integrated simulator |
| Built on | **DRAMSim3** and **Ramulator** |
| Open source | **Not confirmed** — no open-sourcing statement, no repository link in the paper |
| Purpose | Co-simulate the PIM Kernel software layer against a modeled LPDDR5X-PIM device |

⚠️ **Conflation warning.** The paper separately cites Samsung's earlier **PIMSimulator** (https://github.com/SAITPublic/PIMSimulator, 2023), which *is* public. LP5X-PIM Sim is a different tool. A survey entry that merges them would wrongly imply LP5X-PIM Sim is downloadable.

## U6. Public tooling inventory (updated)

| Tool | Generation | Status |
|---|---|---|
| PIM SW Stack (TF/PyTorch interception) | Aquabolt-XL | not public |
| PIM BLAS Library (BLAS L2) | Aquabolt-XL | not public |
| PIM Runtime / Driver Library | Aquabolt-XL | not public |
| PIMSys (gem5 + DRAMSys + Rust) | community research tool | public (ACM MemSys 2024) |
| **Samsung PIMSimulator** | earlier Samsung PIM generation | **public** — github.com/SAITPublic/PIMSimulator (2023) |
| **PIM Kernel layer** (Data Mapper + PIM Executor) | **LPDDR5X-PIM** | **not public** |
| **LP5X-PIM Sim** | **LPDDR5X-PIM** | **availability not confirmed** — no release statement, no repo |

Net: Samsung's PIM toolchain remains closed. The only genuinely downloadable Samsung PIM artifact is PIMSimulator (2023), which models an earlier generation.

## U7. Key observations

1. **Explicit offline/runtime split.** Data Mapper (offline weight placement) vs. PIM Executor (runtime codegen + mode control + GEMV). Aquabolt-XL's stack did not draw this line publicly.
2. **Runtime code generation, not compilation.** IRF code is synthesized per operation from shape and dtype — closer to a JIT microcode emitter than to a compiler with an IR.
3. **Mode management is first-class.** SB↔MB is an explicit software responsibility (PIM Control), where Aquabolt-XL exposed only an unpublished "PIM mode enable" command sequence.
4. **Data placement is the optimization surface.** Vertical / Horizontal / Reshape mappings are the published performance levers; the 1.65× Reshape gain is a *software* result, not a hardware one.
5. **Still GEMV-only.** No GEMM, attention, or end-to-end model results are published for this generation.
6. **Nothing is public.** No SDK, no repo, no API documentation for the PIM Kernel layer.

## Sources (LPDDR5X-PIM software-stack update)

- [arXiv:2606.00636 — LP5X-PIM Sim (Samsung, 30 May 2026)](https://arxiv.org/abs/2606.00636) · [PDF](https://arxiv.org/pdf/2606.00636) — sole source for the PIM Kernel layer, its sub-components, and the address-mapping strategies
- [Samsung PIMSimulator (SAITPublic, 2023)](https://github.com/SAITPublic/PIMSimulator) — public, earlier generation; explicitly *not* LP5X-PIM Sim
- [Hot Chips 2026 program](https://www.hotchips.org/) — Memory session, Tue 25 Aug 2026, Karam Hwang (Samsung); **disclosure scheduled, content not yet public**
