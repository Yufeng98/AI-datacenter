# Tenstorrent Hardware Architecture

*as_of: 2026-08-08*
*chip: tenstorrent*
*device_class: Tensix RISC + SFPU*
*devices: Wormhole (n150/n300/Galaxy Wormhole), Blackhole (p100a/p150a/p150b/Galaxy Blackhole), Quasar (bring-up target; no public spec sheet)*

---

## Overview

Tenstorrent's hardware architecture is organized around the **Tensix Processor**: a 2D mesh of specialized compute tiles called **Tensix cores**, connected by a dual **Network-on-Chip (NoC)** with 2D torus topology, surrounded by GDDR6 memory controllers, Ethernet ports, and a PCIe bridge. The defining architectural choice is **explicit data movement with no hardware caches** above each Tensix core's 1.5 MB SRAM scratchpad.

### Generation Overview

*Revised 2026-08-08 against tenstorrent.com/en/hardware/blackhole and /hardware/galaxy. The 140-core / 745 TOPS Blackhole entry was the **full-die** figure; every shipping Blackhole PCIe card enables **120** Tensix cores and is rated **664 TFLOPS BLOCKFP8**.*

| Chip / System | Status | Tensix Cores | On-chip SRAM | GDDR6 | Memory BW | Peak | Process |
|------|--------|-------------|-------------|-------|-----------|------|---------|
| Wormhole n150 | shipping | 72 (active of 80) | ~108 MB | 12 GB | 192–336 GB/s | 262 TOPS FP8 | GFP 12nm |
| Wormhole n300 | shipping | 80 per chip × 2 | ~120 MB per chip | 12 GB per chip | ~336 GB/s per chip | 328 TOPS FP8 per chip | GFP 12nm |
| **Blackhole p100a** | shipping ($999) | **120** | 180 MB (120 × 1.5 MB) | **28 GB** | **448 GB/s** | **664 TFLOPS BLOCKFP8** | TSMC 6nm |
| **Blackhole p150a** (active) | shipping ($1,399) | **120** | 180 MB | 32 GB | 512 GB/s | **664 TFLOPS BLOCKFP8** | TSMC 6nm |
| **Blackhole p150b** (passive) | shipping ($1,399) | **120** | 180 MB | 32 GB | 512 GB/s | **664 TFLOPS BLOCKFP8** | TSMC 6nm |
| Blackhole full die | reference figure | 140 | 210 MB | — | — | 745 TOPS FP8 | TSMC 6nm |
| Blackhole p300 | **delisted** from the vendor product page as of 2026-08-08; retained for history | 140 × 2 | 420 MB | 64 GB | — | ~1.5 POPS | TSMC 6nm |
| **Galaxy Blackhole** (6U) | **GA 2026-04-28**, from $110,000 | 32 ASICs | **6.2 GB @ 2.9 PB/s** | 1 TB | **16 TB/s** | **23 PFLOPS Block FP8** | TSMC 6nm |
| Galaxy Wormhole (6U) | shipping, from $70,000 | 32 ASICs | not disclosed | not disclosed | not disclosed | not disclosed | GFP 12nm |
| **Quasar** | bring-up in tt-llk / tt-metal since 2025-08-15; no product | not disclosed | not disclosed | not disclosed | not disclosed | not disclosed | Samsung SF4X (4 nm-class) per the Oct-2023 chiplet announcement — *medium confidence*, not restated since |

**Galaxy per-ASIC reconciliation:** 23 PFLOPS ÷ 32 ≈ 719 TFLOPS per ASIC (repo arithmetic, not vendor-stated), between the 664 TFLOPS shipping-card figure and the 745 TOPS full-die figure. Tenstorrent has not published the Blackhole configuration used inside Galaxy — **not disclosed**.

---

## 1. Tensix Core: Compute Engine

Each Tensix core is an autonomous compute node containing:

### Five Baby RISC-V Cores

| Core | Role |
|------|------|
| **BRISC** (Broadcast RISC / Data Movement 0) | Issues NoC 0 reads; runs the reader kernel |
| **NCRISC** (NoC RISC / Data Movement 1) | Issues NoC 1 writes; runs the writer kernel |
| **TRISC0** (Unpack) | Drives unpack unit; copies L1 tile data → FPU SrcA/SrcB registers |
| **TRISC1** (Math) | Drives FPU/SFPU; issues MVMUL, GMPOOL, ELWMUL, SFPU instructions |
| **TRISC2** (Pack) | Drives pack unit; copies FPU Dst registers → L1 circular buffers |

TRISC0, TRISC1, TRISC2 form a hardware-pipelined Unpack → Math → Pack sequence synchronized by hardware semaphores on the FPU destination register file (`tile_regs_acquire/commit/wait/release`).

### Compute Units

| Unit | Specification |
|------|--------------|
| Matrix Unit (FPU) | Native 32×32 tile operations; BF16/FP16/FP8 (E4M3, E5M2)/INT8/INT32/FP32 |
| Vector Unit (SFPU) | 32-lane SIMD; Exp, Log, Sqrt, Tanh, GeLU, Sigmoid, custom functions |
| Unpack unit | Reformats L1 tile → FPU input registers (SrcA, SrcB); handles format conversion |
| Pack unit | Reformats FPU Dst registers → L1 tile in output circular buffer |
| L1 SRAM | 1.5 MB per core (software-managed scratchpad; no eviction policy) |
| NoC routers | 2 (NoC 0 + NoC 1), each 32 bytes/cycle wide |

### Blackhole: Integrated Linux Host

Blackhole uniquely integrates **16 "Big RISC-V" 64-bit, dual-issue, in-order CPU cores** (4 clusters of 4). These can boot and run Linux directly on-chip, making Blackhole a standalone AI computer without a separate host x86 CPU. All three shipping Blackhole PCIe cards (p100a, p150a, p150b) list the full complement of 16 Big RISC-V cores.

### Shipping vs. full-die Tensix count (correction, 2026-08-08)

The vendor Blackhole product page lists **120 Tensix cores** on p100a, p150a and p150b. The 140-core figure previously carried in this document is the full-die count and is not a shipping configuration. The corresponding per-card on-chip SRAM is therefore **180 MB** (120 × 1.5 MB), not 210 MB; the 210 MB figure confirmed by the ASPLOS 2025 microbenchmarking paper corresponds to the 140-core die.

### Quasar (next generation) — compute engine not disclosed

Quasar is an active bring-up target in `tenstorrent/tt-llk` and `tenstorrent/tt-metal` (first commit 2025-08-15; 662 Quasar-mentioning commits in tt-metal, 85 in tt-llk as of this scan). Its **core count, compute-unit organisation, data types, SRAM and memory configuration are all not disclosed**, and `tt-isa-documentation` still contains only `WormholeB0` and `BlackholeA0`. What *is* observable from the open-source stack is that Quasar has its own LLK backend, its own SFPU/matmul performance-test coverage, an iDMA/unicast path with virtual-channel assignment, and a fast-dispatch flow that uses DataflowBuffers — i.e. it is a Tensix-lineage part with a revised data-movement engine, not a fresh architecture. Process node: **Samsung Foundry SF4X (4 nm-class), Taylor, Texas**, per Tenstorrent's October 2023 Quasar-chiplet announcement — *medium confidence*, since Tenstorrent has not restated the node for the 2025–26 tt-metal Quasar target.

---

## 2. NoC Mesh: Data Path

### Topology

The chip die is organized as a 2D NoC grid. Wormhole's grid is 10×12 = 120 tile positions:

| Tile Type | Count (Wormhole) | Description |
|-----------|-----------------|-------------|
| T (Tensix) | 80 | Active compute cores |
| D (DRAM) | 6 | GDDR6 controller tiles |
| E (Ethernet) | 16 | 100 Gbps Ethernet tiles |
| A (ARC) | 1 | Management/dispatch core |
| P (PCIe) | 1 | PCIe host bridge tile |

Blackhole extends the grid for 140 Tensix cores + GDDR6 + Ethernet (10 × 400 Gbps) + PCIe 5.0.

### Dual NoC Torus

| NoC | Direction | Primary User | Traffic |
|-----|-----------|-------------|---------|
| NoC 0 | East + South | BRISC (Data Movement 0) | Input reads (DRAM → L1) |
| NoC 1 | West + North | NCRISC (Data Movement 1) | Output writes (L1 → DRAM) |

Wraparound connections form a 2D torus, providing full connectivity. Links are **32 bytes wide** per hop. The two NoCs together provide quasi-full-duplex operation: simultaneous reads and writes without contention.

### Explicit Data Movement (No Hardware Cache)

There is **no hardware cache** between Tensix cores or between Tensix and DRAM. All data transfers are explicit NoC operations:

```cpp
// Async DRAM → L1 read (reader kernel, BRISC)
noc_async_read(src_noc_addr, dst_l1_addr, size_bytes);
noc_async_read_barrier();   // wait for completion

// Async L1 → DRAM write (writer kernel, NCRISC)
noc_async_write(src_l1_addr, dst_noc_addr, size_bytes);
noc_async_write_barrier();

// L1 → remote Tensix L1 (core-to-core)
uint64_t remote = get_noc_addr(noc_x, noc_y, remote_l1_offset);
noc_async_write(local_l1_addr, remote, size_bytes);
```

**Consequences**:
- No cache coherence traffic; no cache misses; no TLB misses
- Deterministic, predictable bandwidth utilization
- Programmer/compiler must orchestrate all data movement explicitly
- Same API works for intra-chip and cross-chip (via Ethernet tiles) transfers

---

## 3. On-chip Memory

### Per-Core L1 SRAM (1.5 MB)

The Tensix L1 SRAM is used for:

| Usage | Description |
|-------|-------------|
| Circular buffers | Ring buffers for producer-consumer sync between reader/compute/writer kernels |
| L1 interleaved | Tensor pages distributed across multiple Tensix L1s for low-latency op execution |
| L1 sharded | Full tensor blocks per core for zero-DRAM-traffic op execution |
| Scratch space | Intermediate compute results |

**No eviction policy.** Data persists until explicitly overwritten. The programmer controls the entire L1 lifecycle.

### Total On-chip SRAM

| Device | Total SRAM | Per Core |
|--------|-----------|---------|
| Wormhole n150 | ~108 MB (72 × 1.5 MB) | 1.5 MB |
| Wormhole n300 | ~240 MB (160 × 1.5 MB) | 1.5 MB |
| Blackhole full die | 210 MB (140 × 1.5 MB) | 1.5 MB |
| **Blackhole p100a / p150a / p150b** | **180 MB (120 × 1.5 MB)** | 1.5 MB |
| **Galaxy Blackhole (32 ASICs)** | **6.2 GB @ 2.9 PB/s** (vendor "accelerator SRAM") | 1.5 MB |
| Quasar | not disclosed | not disclosed |

Confirmed by ASPLOS 2025 microbenchmarking paper: the 140-core Blackhole die has 210 MB on-chip SRAM. The shipping PCIe cards enable 120 cores (180 MB).

**Galaxy accelerator SRAM (2026-08-08 correction).** The vendor Galaxy page states **6.2 GB of accelerator SRAM at 2.9 PB/s** for the 32-ASIC system, superseding the ~6.7 GB figure previously carried here. At 1.5 MB per Tensix core, 6.2 GB corresponds to ~4,130 cores across 32 ASICs, i.e. ~129 cores per ASIC (repo arithmetic) — again between the 120-core shipping-card and 140-core full-die configurations. The exact Galaxy per-ASIC core count is **not disclosed**. The 2.9 PB/s aggregate SRAM bandwidth is the first such figure Tenstorrent has published for a Galaxy system.

### Tensor Sharding

TT-NN supports three L1 sharding modes to distribute tensor data across the on-chip SRAM pool:

| Mode | Description |
|------|-------------|
| Height sharding | Rows distributed across cores |
| Width sharding | Columns distributed across cores |
| Block sharding | Rectangular tensor blocks per core |

Sharded tensors enable ops to run with **zero DRAM traffic** for operands that fit in the distributed L1 pool.

---

## 4. Off-chip Memory (GDDR6)

Tenstorrent uses **GDDR6** (not HBM) — chosen to avoid HBM supply chain constraints and CoWoS interposer costs.

| SKU | Capacity | Controllers | Peak BW | Interface |
|-----|----------|------------|---------|-----------|
| Wormhole n150 | 12 GB | 6 × 2-channel | 192–336 GB/s | 192-bit |
| Wormhole n300 | 24 GB (2 chips) | 12 total | ~336 GB/s per chip | — |
| **Blackhole p100a** | **28 GB** | not disclosed | **448 GB/s** | not disclosed (336-bit implied by capacity/BW ratio; not vendor-stated) |
| Blackhole p150a / p150b | 32 GB | 24 | 512 GB/s | 384-bit |
| **Galaxy Blackhole (32×)** | 1 TB | 768 (24 × 32) | **16 TB/s** (vendor-stated) | — |
| Quasar | not disclosed | not disclosed | not disclosed | not disclosed |

DRAM controller tiles appear at fixed (noc_x, noc_y) coordinates in the NoC grid. `get_noc_addr_from_bank_id()` maps DRAM bank IDs to NoC addresses. Interleaved buffers stripe tensor pages across all DRAM banks for maximum bandwidth utilization.

**Tradeoff vs. HBM**: Wormhole/Blackhole GDDR6 bandwidth (192–512 GB/s) is lower than NVIDIA H100 HBM3 (3.35 TB/s). This is compensated by the large on-chip SRAM pool (210 MB) for DRAM bandwidth-sensitive inference workloads.

---

## 5. Host Interface / Package

### Wormhole

| Spec | Value |
|------|-------|
| PCIe interface | Gen 4.0 x16 (~64 GB/s bidir) |
| Process | GlobalFoundries 12nm |
| Die size | ~670 mm² |
| TDP | ~75 W (n150), ~150 W (n300) |
| Form factor | Standard PCIe card |

### Blackhole

| Spec | Value |
|------|-------|
| PCIe interface | Gen 5.0 x16 |
| Process | TSMC 6nm |
| External connectivity | 4× QSFP-DD ports (800 Gbps each) on p150a/p150b; none on p100a |
| On-chip host CPU | 16 Big RISC-V cores (optional Linux boot) |
| TDP | 300 W (p100a, p150a, p150b) |
| Cooling | active (p100a, p150a) / passive (p150b) |

### Galaxy Blackhole system host (added 2026-08-08)

The 6U Galaxy Blackhole server pairs its 32 Blackhole ASICs with a **single x86 host**, not one host per accelerator:

| Spec | Value |
|------|-------|
| Host CPU | 1 × AMD EPYC 9004 (Zen 4), up to 32 cores, ≤280 W TDP |
| Host memory | up to 576 GB (6 × 96 GB) DDR5-4800 ECC RDIMM |
| Chassis | 6U rackmount, air-cooled |
| Power | 8–10 kW average, 12 kW max; max system power configurable up to 14.5 kW |
| Weight | 262 lbs / 119 kg |
| List price | from $110,000 (4-system base supercluster from $440,000) |

Tenstorrent has not published a rationale for the 1-host-per-32-ASIC ratio; the division of labour between the EPYC host and each Blackhole's 16 on-die Big RISC-V cores in Galaxy is **not disclosed**.

---

## 6. Scale-up Interconnect (Ethernet Tiles)

Tenstorrent uses **standard Ethernet** for chip-to-chip connectivity — Ethernet tiles appear in the NoC grid like any other tile type. This means cross-chip data movement uses the same `noc_async_read/write` API as intra-chip movement.

### Wormhole Ethernet

| Parameter | Value |
|-----------|-------|
| Ethernet tiles per chip | 16 |
| Per-tile bandwidth | 100 Gbps |
| Total per-chip bandwidth | 1.6 Tbps |
| Multi-chip configurations | n300 (2×WH direct), T3000 (8×WH 2×4 mesh) |

### Blackhole Ethernet

| Parameter | Value |
|-----------|-------|
| Ethernet links per chip | 10 |
| Per-link bandwidth | 400 Gbps |
| Total per-chip bandwidth | 4 Tbps |
| External ports | 4× QSFP-DD (800 Gbps each) on p150a/p150b |
| Multi-chip configurations | QuietBox (4× BH), LoudBox (8× BH), Galaxy (32× BH, 4×8 mesh) |

### Galaxy Blackhole accelerator fabric (added 2026-08-08)

| Parameter | Value |
|-----------|-------|
| ASICs per node | 32 |
| Per-ASIC fabric links | 10 × 400 GbE |
| Vendor-stated aggregate | **32 TB/s** |
| Unidirectional equivalent | 32 × 10 × 400 Gb/s = 128 Tb/s = **16 TB/s each way** (repo arithmetic) |
| Independent cross-check | The Register (2026-04-28) quotes ~100 Tbps aggregate intra-node Ethernet, which does **not** reconcile cleanly with 128 Tb/s; the discrepancy is unexplained in public material |

The vendor's 32 TB/s is bidirectional math. This survey records both the vendor figure and the unidirectional equivalent so that Galaxy can be compared like-for-like against NVLink/ICI figures, which are conventionally quoted bidirectionally as well.

The TT-Topology tool (`tenstorrent/tt-topology`) configures Ethernet routing tables (mesh / linear / torus) for multi-card deployments.

---

## 7. Scale-out Interconnect

Tenstorrent uses **commodity Ethernet switches** for rack-scale deployment — no proprietary switching ASIC.

| Topology | Description |
|----------|-------------|
| Wormhole cluster | 2D mesh via 16× 100 GbE per chip; commodity switches |
| Blackhole Galaxy | 4×8 mesh (32 chips); 10× 400 GbE per ASIC intra-node (vendor-stated 32 TB/s) |
| Blackhole multi-rack | QSFP-DD → standard 400G/800G Ethernet switch infrastructure |

### Galaxy Blackhole node-to-node scale-out (added 2026-08-08)

| Parameter | Value |
|-----------|-------|
| Scale-out ports per node | up to 56 × 800 GbE QSFP-DD |
| Vendor-stated aggregate | **11.2 TB/s** |
| Unidirectional equivalent | 56 × 800 Gb/s = 44.8 Tb/s = **5.6 TB/s each way** (repo arithmetic) |
| Vendor-stated max system scale | **144 nodes / >4,000 chips** |
| Largest configuration described in public material | 36-Galaxy supercluster networked as one computer (TT-Deploy, 2026-05-04); 4-system base supercluster is the entry SKU ($440,000) |
| Switching ASIC | none proprietary — commodity Ethernet |

This supersedes the "~1 TBps aggregate" figure carried against Galaxy in the 2026-04-05 baseline. That number traces to the *per-chip* Blackhole QSFP-DD figure — docs.tenstorrent.com states "1 TBps aggregate" for the four 800 Gbps QSFP-DD ports on a single Blackhole card — and was misapplied to the 32-ASIC system.

**Advantage**: No vendor lock-in on switching; standard network tooling; lower cost for large clusters.  
**Tradeoff**: Higher latency than NVLink; collective operations must tolerate Ethernet switch latency. The Register (2026-04-28) noted that prior Tenstorrent testing showed "generally poor performance scaling", so the scale-out efficiency of the 144-node claim is **not independently verified**.

---

## 8. Key Architectural Insights

1. **No hardware caches above 1.5 MB L1 SRAM per core**: Eliminates cache coherence overhead across 80–140 cores; requires explicit data movement programming.

2. **GDDR6 instead of HBM**: 512 GB/s (Blackhole) vs. 8 TB/s (B200) — compensated by 210 MB on-chip SRAM pool for inference workloads.

3. **Ethernet as primary interconnect**: Commodity Ethernet for both on-chip tile communication and scale-out — no proprietary fabric required. Arteris FlexNoC IP used for chiplet-based designs.

4. **Blackhole = standalone AI computer**: 16 Big RISC-V host cores + 140 Tensix AI cores on one die; can run Linux and AI inference without host x86 CPU.

5. **ISA fully public**: `tenstorrent/tt-isa-documentation` contains complete instruction references for Baby RISC-V, Matrix Unit, Vector Unit (SFPU), and Scalar Unit — unique among AI accelerator vendors.

6. **Transparent multi-chip programming**: Ethernet tiles appear in the NoC grid; cross-chip operations use the same `noc_async_read/write` API as intra-chip operations. The programmer writes single-chip kernels and the runtime/compiler handles multi-chip scheduling.

7. **Rack-scale is a GA product with a published price (2026-04-28)**: Galaxy Blackhole is the first Tenstorrent datacenter system sold at a public list price ($110,000/server, $440,000 for the 4-system base supercluster). Publishing system-level list prices is unusual among datacenter accelerator vendors and means Tenstorrent's $/PFLOPS and $/TB-of-memory can be computed from primary sources rather than estimated.

8. **All-Ethernet at every tier, now quantified**: intra-node accelerator fabric (10 × 400 GbE per ASIC) and inter-node scale-out (up to 56 × 800 GbE QSFP-DD) are the same technology as the on-die Ethernet tiles — one protocol from tile to rack to datacenter, with no NVSwitch-, ICI- or NeuronLink-equivalent proprietary layer anywhere in the stack.

---

## Update Log

**2026-08-08** — Galaxy Blackhole GA (2026-04-28) applied: hardened system spec (23 PFLOPS Block FP8, 6.2 GB SRAM @ 2.9 PB/s, 1 TB GDDR6 @ 16 TB/s, 32 TB/s fabric, 11.2 TB/s scale-out, EPYC 9004 host, 8–14.5 kW, 262 lbs), public list pricing, and 144-node/>4,000-chip max scale. Corrected the following pre-existing repo errors: Galaxy peak (~24 PFLOPS → 23 PFLOPS Block FP8), Galaxy SRAM (~6.7 GB → 6.2 GB), Galaxy scale-out ("1 TBps aggregate" → 11.2 TB/s vendor / 5.6 TB/s unidirectional), and the Blackhole PCIe SKU line (140 cores / 745 TOPS was the full-die figure; shipping p100a/p150a/p150b are 120 cores / 664 TFLOPS BLOCKFP8, with p100a at 28 GB @ 448 GB/s). p300 flagged as delisted. Quasar added as a bring-up target with process node from the Oct-2023 Samsung SF4X chiplet announcement (medium confidence) and everything else recorded as not disclosed.

---

## Sources

- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Wormhole Specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html)
- [HotChips 2024: Blackhole & TT-Metalium](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf)
- [ASPLOS 2025: Dissecting Blackhole via Microbenchmarking](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Introduction to Tenstorrent (HPC-Asia 2025)](http://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf)
- [Corsix: Tenstorrent Wormhole Series (Parts 1–6)](https://www.corsix.org/content/tt-wh-part1)
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [SemiAnalysis: Wormhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-wormhole-analysis-a-scale)
- [SemiAnalysis: Blackhole Scale-Out](https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and)
- [TT-ISA Documentation](https://github.com/tenstorrent/tt-isa-documentation)
- [Memory for Kernel Developers](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html)

### Added 2026-08-08

- [Galaxy Product Page (Blackhole + Wormhole specs and list pricing)](https://tenstorrent.com/en/hardware/galaxy)
- [Galaxy Blackhole System Documentation (User Guide v1.6, 2026-08-04)](https://docs.tenstorrent.com/systems/galaxy-blackhole/index.html)
- [Galaxy Blackhole datasheet PDF](https://docs.tenstorrent.com/_downloads/3086863c42126fd0d63b01baccf8432e/galaxy-blackhole.pdf)
- [Blackhole Product Page (p100a / p150a / p150b SKUs and pricing)](https://tenstorrent.com/en/hardware/blackhole)
- [Tenstorrent Newsroom: Galaxy Blackhole GA (2026-04-28)](https://tenstorrent.com/en/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance)
- [The Register: Tenstorrent's Galaxy Blackhole AI servers are finally out (2026-04-28)](https://www.theregister.com/software/2026/04/28/tenstorrents-galaxy-blackhole-ai-servers-are-finally-out/5229759)
- [docs.tenstorrent.com/aibs — current AI-board and system product index](https://docs.tenstorrent.com/aibs/)
- [The Register: Samsung to fab RISC-V chips for Tenstorrent (2023-10-04)](https://www.theregister.com/on-prem/2023/10/04/samsung-to-fab-risc-v-chips-for-tenstorrent/) — Quasar chiplet on Samsung SF4X, Taylor TX
- [tt-metal v0.75.0 release notes (2026-07-30)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.75.0)
