# Samsung PIM (Aquabolt-XL HBM-PIM / LPDDR5X-PIM) — Hardware Architecture

*chip: samsung-aquabolt-pim*
*generations: Aquabolt-XL HBM2-PIM (2021) / LPDDR5X-PIM (disclosed 2026)*
*as_of: 2026-08-08*

---

## Generation Overview

| Generation | Memory base | PIM placement granularity | Compute units | Numeric formats | Peak throughput | Internal BW | External BW | Capacity | Process | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Aquabolt-XL (HBM-PIM)**, ISSCC 2021 / HC33 2021 | HBM2 (Samsung Aquabolt), 8-die stack | 1 processor per **pseudo-channel**; 32 per PIM-DRAM die; 4 PIM-DRAM dies + 4 standard dies | 128 FP16 SIMD processors / stack (2× FP16 mul + 2× FP16 add + 3 RF + RISC-32 µcontroller each) | FP16 only | ~1.2 TFLOPS FP16 / stack | 4.92 TB/s | 1.23 TB/s (JEDEC HBM2 I/O) | 16 GB | Samsung 20 nm class DRAM | Prototype; no confirmed commercial production post-2021 |
| **LPDDR5X-PIM**, arXiv 30 May 2026 / FMS 2026 / Hot Chips 2026 | **LPDDR5X-9600**, JEDEC JESD209-5C (2022) | **1 PIM block per DRAM bank (1-to-1)** | **not disclosed** (MAC lanes per PIM block not published) | **INT: W8A8, W4A4, W8A16, W4A8, W4A16; FP: W8A8(FP), W8A16(FP)** | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** | Announced/showcased only; no production date, no named customers, no measured silicon |

**Reading the "not disclosed" cells:** Samsung's arXiv note states explicitly that *"further technical details regarding the specific architecture and circuit design of the LPDDR5X-PIM will be disclosed in future publications."* The Hot Chips 2026 Memory-session talk (Tue 25 Aug 2026, Karam Hwang, Samsung) is where that detail is scheduled to appear — **disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public** as of this document's as_of date.

**Program continuity:** Samsung's PIM line moved off HBM. LPDDR5X-PIM is a distinct memory base (LPDDR5X vs. HBM2), a distinct placement granularity (per-bank vs. per-pseudo-channel), and a distinct numeric-format story (mixed-precision INT/FP vs. FP16-only). No HBM3/HBM4-based Samsung PIM or Aquabolt-XL successor is confirmed either way.

---

## Part I — Aquabolt-XL HBM-PIM (2021)

## Architecture Overview

Samsung Aquabolt-XL integrates **FP16 SIMD processors inside HBM2 DRAM dies** — specifically inside custom PIM-DRAM dies that replace the bottom 4 dies of a standard 8-die Aquabolt HBM2 stack. The processor is placed at the bank boundary between even/odd bank pairs, accessing DRAM arrays at **internal bandwidth (~4.92 TB/s)** without leaving the stack.

---

## Die Stack Organization

```
HBM2 Stack: 8 dies + Logic Die
─────────────────────────────────────
  Die 7 (top)  │  Standard DRAM
  Die 6        │  Standard DRAM
  Die 5        │  Standard DRAM
  Die 4        │  Standard DRAM
  Die 3        │  PIM-DRAM ★ (32 processors)
  Die 2        │  PIM-DRAM ★ (32 processors)
  Die 1        │  PIM-DRAM ★ (32 processors)
  Die 0 (bot)  │  PIM-DRAM ★ (32 processors)
  Logic Die    │  JEDEC HBM2 protocol (unchanged)
─────────────────────────────────────
Total: 128 processors, 16 GB capacity
```

---

## PIM-DRAM Die Architecture

```
PIM-DRAM Die (one of 4)
├── 16 pseudo-channels (PC0–PC15)
│   └── Each pseudo-channel:
│       ├── 1 bank (DRAM array, 1 Gb)
│       └── 1 Processor
│           ├── 2× FP16 multipliers
│           ├── 2× FP16 adders (accumulator)
│           ├── 3× register files
│           └── RISC 32-bit micro-controller
└── Shared instruction broadcast (SIMD)
    └── All 32 processors execute same instruction
```

---

## Processor Design

| Property | Value |
|----------|-------|
| FP16 multipliers | 2 per processor |
| FP16 adders | 2 per processor |
| Register files | 3 per processor |
| Instruction model | RISC 32-bit; SIMD (all same instruction) |
| Numeric format | FP16 only |
| Processors / PIM die | 32 |
| Total processors / stack | 128 (4 dies × 32) |

---

## Compute and Bandwidth

| Metric | Per PIM die | Full Stack |
|--------|-------------|-----------|
| Processors | 32 | 128 |
| Peak FP16 TFLOPS | ~0.3 | ~1.2 |
| Internal DRAM BW | ~1.23 TB/s | 4.92 TB/s |
| External HBM2 I/O | ~307 GB/s | ~1.23 TB/s |
| Int:Ext BW ratio | ~4× | ~4× |

The processors exploit the ~4× higher internal bandwidth vs the external I/O — this is the energy efficiency advantage of PIM.

---

## JEDEC Compatibility

- Same HBM2 ball map and package as standard Aquabolt
- Same JEDEC HBM2 timing specification
- Memory controller (in GPU/FPGA) sees standard HBM2 device
- PIM mode activated via special command sequence (not JEDEC standard)
- Standard HBM2 read/write operations fully retained

---

## Validated System Configurations

### Configuration 1: Xilinx Alveo U280 FPGA
- Application: RNN-T (speech recognition)
- Result: **2.49× speedup**, **62% energy reduction** vs. standard HBM Alveo

### Configuration 2: GPU (AMD MI60 class)
- Application: ML inference workloads
- Result: **>2× speedup**, **>70% energy reduction**

---

## Process and Packaging

| Property | Value |
|----------|-------|
| Process | Samsung 20nm class DRAM |
| Packaging | HBM2 3D TSV stack (JEDEC standard dimensions) |
| Stack height | Standard HBM2 (same as Aquabolt) |
| Interposer | Same as host GPU/FPGA HBM2 socket |

---

## Limitations

1. **FP16 only**: No INT8, BF16, FP32 accumulation *(this is an Aquabolt-XL limitation specifically; LPDDR5X-PIM adds INT4/INT8/INT16 and an FP path — see Part II)*
2. **SIMD only**: All 128 processors execute identical instruction — no per-processor divergence
3. **Local-only**: PIM operates on data in own pseudo-channel; cross-die requires normal I/O path
4. **~1.2 TFLOPS**: Not compute-competitive with GPU; designed for memory-bound GEMV only
5. **Prototype status**: No commercial productization of the HBM2-PIM part confirmed post-2021

---

# Part II — LPDDR5X-PIM (disclosed 2026)

*Added 2026-08-08. Primary source: arXiv:2606.00636v1, "LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM," Cha, Choi, Kim, Paik, Lee, Sohn (all Samsung Electronics), 30 May 2026. Everything below is read from that paper unless labeled otherwise.*

## 1. Compute Engine

| Property | Value |
|---|---|
| PIM block placement | **1-to-1 with a DRAM bank** — "each PIM block is deployed in a 1-to-1 mapping with a corresponding DRAM bank, allowing direct data transfer through internal compute units and specialized PIM registers" |
| MAC lanes per PIM block | **not disclosed** |
| Peak throughput (TFLOPS / TOPS) | **not disclosed** |
| Primary operation | GEMV (matrix-vector), tiled |
| Named architectural registers | **IRF** — instruction register file (holds "IRF code" synthesized per operation); **SRF** — Source Register File |
| Instruction model | Specialized **PIM ISA**; per-tile GEMV kernels; explicit pipeline flush-out |
| Numeric formats — integer | **W8A8, W4A4, W8A16, W4A8, W4A16** (weight-bits / activation-bits) |
| Numeric formats — floating point | **W8A8(FP), W8A16(FP)** |

This is the substantive compute-architecture change versus Aquabolt-XL: Samsung moved from a fixed FP16 MAC datapath to **mixed-precision weight/activation combinations down to 4-bit weights**, with a floating-point path retained alongside the integer path.

---

## 2. Operating Modes

LPDDR5X-PIM has no separate "PIM die" — every bank has a PIM block, and the device switches behavior by mode:

| Mode | Behavior |
|---|---|
| **Single-Bank (SB)** | Standard DRAM operation — the device behaves as a normal JEDEC LPDDR5X part |
| **Multi-Bank (MB)** | Parallel PIM execution across multiple banks |

Mode transitions are managed in software by the **PIM Control** component of the runtime (see the software-stack investigation). This SB/MB split is the LPDDR5X analog of Aquabolt-XL's "PIM mode enable command sequence," but it is exposed as a first-class two-mode model rather than a vendor-private escape sequence.

---

## 3. Memory System

| Property | Value |
|---|---|
| Memory base | **LPDDR5X-9600** |
| Standard compliance | **Strictly JEDEC-compliant** (JESD209-5C, 2022) timing |
| Channels in all published experiments | **4 DRAM channels** |
| Internal (in-bank) bandwidth | **not disclosed** |
| External I/O bandwidth | **not disclosed** |
| Device capacity | **not disclosed** |
| On-chip SRAM / scratchpad | none described; PIM registers (IRF, SRF) only |

Address mapping is a software concern here rather than a fixed hardware property — the paper describes **Vertical Mapping** (rows interleaved across Channel / Rank / Bank Group / Bank), **Horizontal Mapping** (adjacent sub-matrices kept in one bank to maximize row-buffer hits), and a **Reshape Optimization** (column-based partitioning) that reaches near-100% intra-PIM efficiency for small matrices.

---

## 4. Host Interface / Package

| Property | Value |
|---|---|
| Interface | Standard JEDEC LPDDR5X (JESD209-5C) |
| Host | Not specified in the paper; the modeled 150 ns static memory-fence latency is described as "a representative value for high-performance mobile application processors" |
| Package / form factor | **not disclosed** |
| Process node | **not disclosed** |

**Contrast with Aquabolt-XL**: Aquabolt-XL was an in-package HBM2 stack attached to a GPU or FPGA over an interposer. LPDDR5X-PIM is a discrete LPDDR5X device on a conventional LPDDR memory interface — a different integration point, and the reason its natural host is an application processor or SoC rather than a datacenter GPU.

---

## 5. Interconnect

**not applicable** — LPDDR5X-PIM has no scale-up or scale-out interconnect. Communication is the LPDDR5X memory interface to the host. No multi-device PIM topology is described.

---

## 6. Performance — Samsung simulation, not measured silicon

Every number below is output from Samsung's own cycle-accurate **LP5X-PIM Sim** (built on DRAMSim3 and Ramulator), normalized to a **self-defined non-PIM sequential-weight-read baseline with four DRAM channels**. There is no measured-silicon result in the public record, and no independent reproduction.

| Configuration (baseline weight-tile dimension 4096) | GEMV speedup |
|---|---|
| W8A8, W4A4, W8A8(FP) — larger tile shapes | **6.0× – 6.2×** |
| W8A16, W4A16, W8A16(FP) — smaller tile shapes | **5.7× – 5.8×** |
| Any config, with 150 ns static memory-fence latency modeled | **> 5.0×** for most; low of **4.1×** (W4A16) |
| Reshape column-partitioning optimization, small matrices (W < 2048) | additional **up to 1.65×** |

ETNews's "up to 6.2× faster than conventional memory" (23 Jul 2026) is a restatement of the top row, not an independent measurement.

---

## 7. Limitations and Open Items (LPDDR5X-PIM)

1. **Architecture largely undisclosed**: MAC lane count, peak throughput, internal/external bandwidth, capacity, and process node are all unpublished. Samsung defers them to "future publications."
2. **GEMV-only evidence**: all published results are matrix-vector; no GEMM, attention, or end-to-end model results.
3. **Simulation only**: no silicon measurements, no third-party benchmarks.
4. **No productization evidence**: announced and showcased (FMS 2026); no production date, no availability, no named customers.
5. **Simulator availability unconfirmed**: LP5X-PIM Sim has no stated open-source release and no repository link. Samsung's earlier, separately-cited **PIMSimulator** (github.com/SAITPublic/PIMSimulator, 2023) *is* public but is a different tool.
6. **Pending disclosure**: Hot Chips 2026 Memory session, Tue 25 Aug 2026, 09:30–10:30 (chair Jae W. Lee), "Samsung LPDDR5X-PIM: World's First LPDDR based Processing in Memory (PIM) Solution for AI Inference," Karam Hwang (Samsung) — **disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.**

---

## Sources (Part II)

- https://arxiv.org/abs/2606.00636 — arXiv:2606.00636v1 (Samsung), primary source for all Part II architecture, format, and speedup figures
- https://www.hotchips.org/ — Hot Chips 2026 dates and Memory-session listing
- https://news.samsungsemiconductor.com/global/samsung-unveils-next-gen-3d-memory-vision-at-fms-2026-charting-the-future-of-ai-infrastructure/ — FMS 2026 showcase (4 Aug 2026)
- https://www.storagereview.com/news/samsung-outlines-3d-memory-roadmap-for-ai-infrastructure-at-fms-2026
- https://www.sammobile.com/news/samsung-znand-o-lpddr5x-pim-pm1763-memory-chips-ssd-ai-data-centers/
- https://en.etnews.com/20260723200002 — trade press, low confidence
- https://en.sedaily.com/finance/2026/08/05/samsung-sk-push-pim-and-cxl-as-us-china-japan-challenge-hbm — trade press, low confidence; misdates Hot Chips to September 2026
