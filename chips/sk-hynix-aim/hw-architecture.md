# SK Hynix AiM / AiMX — Hardware Architecture

*chip: sk-hynix-aim*
*as_of: 2026-08-08*
*generations: GDDR6-AiM (2022) / AiMX Gen1 (2023) / AiMX Gen2 + AiMX-xPU (2024) — no Gen3 announced as of 2026-08-08*

---

## Generation Overview

| Generation | Year | Public venue | AiM packages | Capacity | Compute per package | Host interface | Status |
|---|---|---|---|---|---|---|---|
| GDDR6-AiM (device) | 2022 | Hot Chips 34 poster; ISCA 2021 / IEEE | 1 (die-level) | 1 GB/package | 2 × 16-lane FP16 SIMD MAC (32 lanes) | GDDR6 channel (JEDEC compatible) | Silicon demonstrated |
| AiMX Gen1 (card) | 2023 | SK Hynix newsroom | 16 | 16 GB | 32 FP16 lanes/package | PCIe | Prototype card |
| AiMX Gen2 / AiMX-xPU | 2024 | AI HW & Edge AI Summit 2024; Hot Chips 2024 (IEEE 10664793) | 32 | 32 GB | 32 FP16 lanes/package (unchanged) | PCIe | Prototype card; roadmap to 256 GB |
| *(2025–2026)* | 2025–2026 | AI Infra Summit 2025; OCP Global Summit 2025; CES 2026 | 4 × AiMX cards in demo chassis | unchanged | unchanged | PCIe (Supermicro SYS-421GE-TNRT3 + 2× H100) | **No new silicon.** Same Gen2-class hardware, still labelled a *prototype* by SK Hynix |

**No fourth hardware generation exists on the public record.** As of 2026-08-08 the following are **not confirmed** and must be treated as nonexistent: AiM Gen3, GDDR7-AiM, HBM-based AiM, and AiM logic inside an HBM4 custom base die. The MAC/pseudo-channel organization described below is unchanged since the 2022 Hot Chips 34 disclosure.

---

## Architecture Overview

SK Hynix AiM is **Processing-in-Memory (PIM)** embedded inside GDDR6 DRAM. FP16 SIMD MAC units are placed at the bank boundary of each pseudo-channel. All banks activate simultaneously (all-bank operation), leveraging internal DRAM bandwidth (~4 TB/s) for in-place matrix-vector multiply without moving data off-chip.

---

## GDDR6-AiM Device Organization

```
GDDR6-AiM Package
├── Pseudo-Channel 0 (PC0)
│   ├── Banks 0–7 (8 banks, DRAM array)
│   └── MAC Unit 0 (16-lane FP16 SIMD)
│       ├── 16× FP16 multipliers
│       ├── 16× FP16 adders (accumulators)
│       └── 3× register files
└── Pseudo-Channel 1 (PC1)
    ├── Banks 8–15 (8 banks, DRAM array)
    └── MAC Unit 1 (16-lane FP16 SIMD)
        ├── 16× FP16 multipliers
        ├── 16× FP16 adders
        └── 3× register files

Total per package: 2 MAC units, 32 FP16 lanes
```

---

## Key Hardware Mechanisms

### All-Bank Operation
Standard GDDR6 activates one bank at a time. AiM activates **all 16 banks simultaneously** — this exposes the full internal DRAM array bandwidth to the MAC units, achieving ~4 TB/s vs the ~64 GB/s external I/O limit.

### Extended DRAM Command Set
New commands (AiM ISR) issued by the host trigger in-DRAM compute:
1. Load activation vector into MAC registers
2. Activate all banks simultaneously (weight rows broadcast to MAC units)
3. MAC operation: weights × activation = partial sums
4. Accumulate partial sums across all banks
5. Write result back to DRAM array (accessible via standard GDDR6 read)

### JEDEC Backward Compatibility
Standard GDDR6 read/write commands still work. Memory controllers see a normal GDDR6 device. AiM mode is opt-in via AiM ISR command sequence.

---

## AiMX Card Architecture

```
AiMX Accelerator Card
├── GDDR6-AiM × 16 (Gen1: 16 GB) or × 32 (Gen2: 32 GB)
│   ├── Each package: 32 FP16 SIMD lanes (2 MAC units)
│   └── All-bank parallel operation per package
├── Host interface: PCIe
└── Card controller: AiM ISR dispatch logic
```

Card acts as a **GEMV co-processor**: host (CPU/GPU) sends activation vectors; AiMX returns accumulated output vectors; weight matrices resident in GDDR6-AiM DRAM permanently.

**Card status as of 2026-08-08 (added 2026-08-08).** SK Hynix's CES 2026 press release (2026-01-05) still calls AiMX "SK hynix's accelerator card **prototype** featuring a GDDR6-AiM chip which is specialized for large language models (LLMs)." No sampling programme, shipping product, named customer, or volume deployment is on the public record. A TrendForce note (2026-03-10) asserting AiM is "already deployed in real-world applications" is a single secondary source with no customer, date, or volume, and contradicts SK Hynix's own wording — it is **not** treated as a status change here.

**Publicly demonstrated multi-card configuration (added 2026-08-08).** At the AI Infra Summit 2025 (SK Hynix post 2025-10-01) and the 2025 OCP Global Summit (post 2025-10-31) SK Hynix ran AiMX in a **Supermicro GPU SuperServer SYS-421GE-TNRT3 with 2 × NVIDIA H100 GPUs and 4 × AiMX cards**, the first publicly described multi-card AiMX system topology. The cards attach over standard PCIe alongside the GPUs; **no proprietary card-to-card link was disclosed**, and **no capacity, bandwidth, or throughput figures were given** for the aggregate configuration.

---

## Bandwidth Comparison

| Path | Bandwidth | Notes |
|------|-----------|-------|
| Internal DRAM (all-bank) | ~4 TB/s | Exploited by AiM MAC units |
| External GDDR6 I/O | ~64 GB/s | Standard host access |
| Bandwidth leverage ratio | ~64× | The core PIM efficiency argument |

---

## Performance Numbers

| Benchmark | Result | vs. Baseline |
|-----------|--------|--------------|
| General memory-bound GEMV | up to 16× faster | vs. CPU + standard DRAM |
| Power | −80% | vs. GPU + GDDR6 |
| LLM attention (Llama 3 70B demo) | 10× faster | vs. GPU baseline |
| LLM power (AiMX 2024) | −80% (1/5 power) | vs. GPU |

---

## Process and Packaging

- GDDR6 standard DRAM process (Samsung/SK Hynix 10–14nm class)
- Standard GDDR6 BGA package — same PCB footprint as regular GDDR6
- Speed grade: 16 Gbps/pin, 1.25 V operating voltage (vs 1.35 V standard)

---

## Deployment Topology

```
Host System
├── CPU or GPU (compute-bound ops: softmax, output projection, etc.)
└── AiMX Card (memory-bound ops: KV-cache GEMV, embedding lookup)
    └── GDDR6-AiM × 16/32 packages
        └── FP16 MAC units inside DRAM banks
```

AiMX is not a standalone AI accelerator — it is a co-processor that accelerates the memory-bound fraction of LLM inference.

Concrete public instance of this topology (AI Infra Summit 2025 / OCP Global Summit 2025):

```
Supermicro GPU SuperServer SYS-421GE-TNRT3
├── 2 × NVIDIA H100 (compute-bound ops; also runs the vLLM serving loop)
└── 4 × AiMX cards over PCIe (memory-bound attention GEMV)
    └── GDDR6-AiM packages, FP16 MAC units inside DRAM banks
```

No aggregate capacity, bandwidth, or tokens/s figure was disclosed for this configuration.

---

## Adjacent SK Hynix Compute-Memory Concepts (context, added 2026-08-08)

At CES 2026 SK Hynix exhibited AiMX beside three other compute-memory concepts. **None of them is AiM**, and none is separately tracked in this survey — they are recorded here because they explain where SK Hynix's compute-memory investment is moving.

| Concept | Description (SK Hynix wording) | Where the compute sits | Relation to AiM |
|---|---|---|---|
| **CuD** (Compute-using-DRAM) | Performs simple computations within the memory cells | Inside the DRAM cell array | More primitive than AiM's bank-boundary FP16 MAC units |
| **CMM-Ax** (CXL Memory Module – Accelerator) | Adds compute to CXL memory expansion | CXL module controller | CXL-attached expansion, not a GDDR6 channel |
| **cHBM** (custom HBM) | Integrates GPU/ASIC functions into the HBM base die | HBM base die (logic process) | Base-die logic, not in-DRAM MACs |

The headline CES 2026 memory product was a **16-layer 48 GB HBM4**; a "1.5 TB/s" bandwidth figure attached to it in secondary coverage could not be confirmed against a primary source and is **not recorded**.

---

## Verification Log

### 2026-08-08 — re-verified, no architectural change

The scan window 2026-04-05 → 2026-08-08 produced **no hardware change** to AiM or AiMX. Verified negatives:

- **Hot Chips 38 (Aug 23–25, 2026)** — disclosure scheduled, a **future** event as of this update: the program is public but no slides, abstracts, or specs exist, so it is not a source for any specification. It lists **no SK Hynix AiM/AiMX talk**; SK Hynix appears only in Tutorial 1 (Memory Technology) with an HBM base-die talk whose exact title and speaker are **not confirmed**. The PIM entries in the Tuesday Memory session are Samsung's LPDDR5X-PIM and the XCENA/Samsung MX1 CXL computational-memory device.
- **FMS 2026 (Aug 4–6, 2026)** — SK Hynix showed HBF, HBM4/HBM4E, SOCAMM2, DDR5 RDIMMs, QLC eSSD and CXL (CMM-Ax, CMM-Hybrid, Pooled Memory). **No AiM/AiMX content.**
- **ISSCC 2026 (Feb 2026)** — SK Hynix papers were LPDDR6 (16 Gb, ~14.4 Gbps/pin) and GDDR7. **No SK Hynix PIM/AiM paper.**
- Only in-window AiM mention is the educational blog post "[Tech Note EP.1] It's the Memory" (2026-05-12) — argument only, **no specs**.

All bandwidth, MAC-organization, capacity, voltage, and speed-grade figures in this document remain as of the 2022–2024 disclosures; none was superseded.
