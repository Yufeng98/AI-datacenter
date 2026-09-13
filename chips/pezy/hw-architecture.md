# PEZY Computing HW Architecture

*as_of: 2026-08-08*
*Generations: PEZY-SC (2014) → SC2 (2017) → SC3 / SC3s (2020) → SC4s (announced 2025; still pre-production as of 2026-08-08)*

---

## Overview

PEZY Computing is a small Japanese fabless chip designer whose central thesis is that **MIMD (Multiple Instruction Multiple Data) manycore at modest clocks is more energy-efficient than SIMD GPU for FP64 scientific computing**. Over five generations spanning 2012–2025, PEZY has built 1,024- to 4,096-PE manycore accelerators that have repeatedly topped the Green500 list — the ranking of the world's most energy-efficient supercomputers — using fluorine-type liquid-immersion cooling (ZettaScaler system).

The latest chip, **PEZY-SC4s**, was presented at Hot Chips 37 (August 2025) and re-published in extended form in *IEEE Micro* in ~June 2026 (DOI 10.1109/MM.2026.3698804). It is a 2,048-PE MIMD processor on TSMC 5nm, 556 mm², with 4× HBM3 (96 GB, 3.2 TB/s), a simulated FP64 efficiency of ~91 GF/W, and a new RISC-V management processor enabling host-independent Linux operation.

**As of 2026-08-08, SC4s remains pre-production with only simulated results.** It has never appeared on PEZY's product page or in any PEZY press release, and its publicly stated end-2025 release target (PEZY press release, 2025-06-06) has been missed by 8+ months with no revised date. The SC3-equipped ZettaScaler 3.0 remains the shipping platform. See §9 "Status Update — 2026-08-08".

---

## 1. Generation Summary

| Gen | Year | PEs | Process | Clock | FP64 TFLOPS | HBM | Green500 | Status (2026-08-08) |
|-----|------|-----|---------|-------|------------|-----|---------|---------------------|
| SC | 2014 | 1,024 | TSMC 28nm | 733 MHz | 1.5 | — | #1 Jun/Nov 2015 (7.0 GF/W) | Retired |
| SC2 | 2017 | 2,048 | TSMC 16FF+ | 1.0 GHz | 4.1 | — | #1 Nov 2017 (17 GF/W) | Listed on products page (ZettaScaler-2.0) |
| SC3 | 2020 | 4,096 | TSMC 7nm | 1.2 GHz | 19.7 | HBM2 | #12 Nov 2021 (24.6 GF/W) | **Shipping** — ZettaScaler 3.0, ZettaVEGA, PZLAST |
| SC3s | 2020+ | 512 | TSMC 7nm (109 mm²) | not disclosed | not disclosed | — | — | Small-module variant; active target of third-party academic porting work (PACT 2025 Winograd; CPC 2026 PEZY-DPM) |
| SC4s | announced 2025 | 2,048 | TSMC 5nm | 1.5 GHz | ~24.6 (sim.) | HBM3 96 GB | Target ~91 GF/W (sim.); **no measured entry** | **Pre-production.** Not on PEZY's product page; stated end-2025 release target missed by 8+ months, no revised date |

**Naming note (added 2026-08-08):** PEZY uses both "PEZY-SC3" (4,096 PE, full part) and "PEZY-SC3s" (512 PE, 109 mm² small module). Third-party papers published in this window consistently write **SC3s**. Earlier revisions of this survey did not distinguish the two; the distinction is now explicit in the table above.

---

## 2. PEZY-SC4s Compute Architecture

### Core Philosophy: MIMD vs SIMD

Unlike GPUs (which group threads into warps/wavefronts that must execute the same instruction), PEZY PEs execute fully independent instruction streams. This eliminates branch-divergence waste for irregular scientific workloads.

| Metric | GPU SIMD (H100) | PEZY MIMD (SC4s) |
|--------|----------------|-----------------|
| Branch divergence | Expensive (mask/serialize) | None |
| Vector width | 1,024–2,048 bits | 256 bits (4-wide FP64) |
| FP64 efficiency | ~49 GF/W | ~91 GF/W (sim.) |
| AI throughput | 989 FP16 TFLOPS | Not primary target |

### Processing Element (PE)

| Feature | PEZY-SC4s |
|---------|----------|
| Cores | 2,048 MIMD PEs |
| Threads per PE | 8 (SMT8 — two groups of 4) |
| SMT scheduling | Fine-grained within group (different thread each cycle); coarse-grained between groups (swap on long-latency op) |
| FP64 SIMD width | 4-wide (256-bit) |
| FP32 / FP16 | Supported |
| BF16 | Supported (new in SC4s) |
| INT8 | Supported |
| Total threads | 2,048 × 8 = 16,384 |

---

## 3. Memory Hierarchy

### 5-Level Geographic Hierarchy

```
PE  ────── L1: 4 KB I + 4 KB D + 24 KB scratchpad
 │
Village (4 PEs)  ──── shared scratchpad
 │
City (4 villages = 16 PEs)  ──── 32 KB L2-I + 64 KB L2-D
 │
Prefecture (18 cities; 16 active = ~256 PEs)
 │
State (8 prefectures = 2,048 active PEs)  ──── 64 MB shared L3
 │
HBM3 (4 stacks, 96 GB, 3.2 TB/s)
```

| Level | Members | Cache |
|-------|---------|-------|
| PE | 1 | 4 KB I + 4 KB D + 24 KB scratchpad |
| Village | 4 PEs | Shared scratchpad |
| City | 16 PEs | 32 KB L2-I + 64 KB L2-D |
| Prefecture | ~256 PEs | — |
| State | 2,048 PEs | 64 MB shared L3 |

Estimated total on-chip SRAM: ~140 MB (64 MB L3 + 12 MB L2 aggregate + ~64 MB L1/scratchpad).

### Off-chip Memory: HBM3

| Parameter | Value |
|-----------|-------|
| Technology | HBM3 |
| Stacks | 4 |
| Capacity | 96 GB |
| Bandwidth | 3.2 TB/s |

---

## 4. RISC-V Management Processor (New in SC4s)

| Feature | Value |
|---------|-------|
| ISA | RISC-V (open-source Rocket Core) |
| Count | 4 cores |
| Clock | 1.5 GHz |
| OS | Linux (on-chip) |
| Key benefit | Host-less operation — no x86 CPU required to boot and manage PEs |

Prior generations needed an AMD EPYC host running Linux to act as the management processor. SC4s brings this on-chip.

---

## 5. Physical / Package

| Parameter | Value |
|-----------|-------|
| Process | TSMC 5nm |
| Die size | 556 mm² (single monolithic die) |
| TDP | ~600 W (estimated) |
| Packaging | 2.5D (4× HBM3 stacks + SC4s die on interposer) |
| Host interface | PCIe Gen 5 × 16 |

---

## 6. ZettaScaler 4.0 System

> **Status as of 2026-08-08:** ZettaScaler 4.0 is a **planned** system. It does not appear on PEZY's products page (which lists ZettaScaler 3.0 and ZettaScaler-2.0 only), and no availability, delivery, or installation announcement has been made. All figures below are from the Hot Chips 37 (Aug 2025) disclosure.

| Component | Value |
|-----------|-------|
| SC4s per node | 4 |
| Host CPU | AMD EPYC 9555P (Zen 5, 64-core, 1.5 GHz) |
| Scale-out | 400 Gb/s NDR InfiniBand |
| Cooling | Liquid immersion (fluorine-type inert liquid; also air-cool option) |
| Test cluster | 90 nodes → 360 chips → 737,280 PEs → 8.6 PF FP64 |

---

## 7. FP64 Efficiency Track Record

| Year | System | Chip | GF/W FP64 | Green500 Rank |
|------|--------|------|-----------|--------------|
| Jun 2015 | Shoubu (RIKEN) | PEZY-SC | 7.03 | #1 |
| Nov 2015 | Shoubu (RIKEN) | PEZY-SC | 7.03 | #1 |
| Nov 2017 | Shoubu System B | PEZY-SC2 | 17.0 | #1 |
| Nov 2021 | ZettaScaler3.0 | PEZY-SC3 | 24.6 | #12 |
| 2025 (sim.) | ZettaScaler4.0 | PEZY-SC4s | ~91 | Target — **no measured entry as of Jun 2026** |

> **Green500 reality check (added 2026-08-08):** no PEZY / ZettaScaler / ExaScaler system appears in the **top 20** of the June 2026 Green500 list. That list's top 20 runs from KAIROS (CALMIP/CNRS, 73.282 GF/W) through ROMEO-2025 (70.912), Levante GPU extension (69.426), Isambard-AI phase 1 (68.835) and Otus (68.177) down to Frontier TDS (62.684), and is entirely NVIDIA GH200/H100 and AMD MI300A/MI250X. PEZY's ~91 GF/W SC4s figure therefore remains a **simulation with no measured list entry behind it**, and it is not directly comparable to these mixed-precision-optimized HPL efficiency numbers. Only the top 20 was verified — this is *not* a claim that PEZY appears nowhere in all 500 entries.

---

## 8. Design Trade-offs

| Decision | Gained | Given Up |
|----------|--------|----------|
| MIMD not SIMD | No branch divergence; FP64-first | Narrow 256-bit SIMD limits AI throughput |
| 2,048 PEs (vs SC3's 4,096) | Larger caches/PE; better efficiency | Lower raw PE count |
| 64 MB L3 | Reduces HBM3 traffic | Large silicon area |
| RISC-V management | Host-less Linux; x86 independence | Additional die area |
| Liquid immersion | Extreme density, silent, efficient | Proprietary deployment infrastructure |
| No proprietary scale-up fabric | Standard IB ecosystem | Limited chip-to-chip bandwidth vs NVLink/xGMI |

---

## 9. Status Update — 2026-08-08

*Window: 2026-04-05 → 2026-08-08. No hardware specification in §2–§6 changed in this window. This section records what was verified and what did not happen.*

### 9.1 SC4s silicon: no change, and an undeclared schedule slip

| Question | Answer as of 2026-08-08 | Evidence |
|---|---|---|
| Has SC4s taped out / sampled / shipped? | **Not disclosed — no announcement of any kind** | PEZY news index (JA + EN) carries exactly two items for all of 2026 (2026-05-11, 2026-05-13); neither mentions SC4s, ZettaScaler 4.0, tape-out, sampling, or mass production. Next item back: 2025-11-21 (board appointment) |
| Is SC4s a purchasable product? | **No** | PEZY products page (2026-08-08) lists ZettaScaler 3.0, PEZY-SC3 Processor and Module, ZettaVEGA, PZLAST, Photo Real 3D, 3D Viewer for Medical, AI-based Image Analysis, ZettaScaler-2.0, PEZY-SC2. No SC4s, no ZettaScaler 4.0. Homepage highlights ZettaScaler 3.0 / PEZY-SC3 / ZettaVEGA only |
| Was there a stated release date? | **Yes — "end of the year" (end-2025)** | PEZY press release, 2025-06-06 |
| Slip | **8+ months past the stated target, no revised date announced** | Inference from absence; PEZY has published no cancellation, delay, funding, or corporate-status notice, so the slip is *undeclared*, not vendor-confirmed |
| Any measured silicon results? | **None** | All §2–§7 performance figures remain PEZY simulations |

Caveat on the corporate-status negative: ExaScaler's own site could not be checked (www.exascaler.co.jp serves a TLS certificate valid only for pezy.co.jp/pezy.jp), so that negative rests on PEZY's news index alone.

### 9.2 New primary publication (no spec change)

The Hot Chips 37 talk was extended into a journal article: Hatta, Tsunoda, Uchida, Ishitani, Koizumi, Shioya, Ishii, "PEZY-SC4s: The Fourth-Generation MIMD Many-Core Processor with High Energy Efficiency and Flexibility for HPC and AI," *IEEE Micro*, pp. 1–8, DOI 10.1109/MM.2026.3698804 (Crossref record created 2026-06-01, deposited 2026-06-02). This is distinct from the HC37 proceedings paper (DOI 10.1109/HCS66204.2025.11154388, 2025-08-24).

**Whether the IEEE Micro version reports measured silicon or still only the ~91 GF/W simulation is not disclosed** to this survey — IEEE Xplore returns HTTP 418/403 to automated retrieval and neither abstract nor body could be obtained. **No SC4s figure in this document has been upgraded on the strength of this citation.**

### 9.3 What PEZY silicon is actually doing in 2026: SC3, not SC4s

Both 2026 vendor announcements are bioinformatics applications on the previous-generation PEZY-SC3, not AI and not FP64 HPC:

- **2026-05-11 — pzMutect2 on ZettaVEGA**: PEZY claims a 139× speedup of somatic-variant calling vs Mutect2 in GATK 4.2.6.1 (wall clock 62 h 36 m 05 s → 26 m 57 s) with >99.99% concordance verified via `bcftools isec`. Vendor-reported, single workload, no independent replication. The release page does not name the processor; the SC3/ZettaVEGA attribution is inferred from the release slug and PEZY's 2024-12-26 ZettaVEGA announcement.
- **2026-05-13 — PZLAST-MAG**: a free public protein-sequence search server for metagenome-assembled genomes running on PEZY-SC3, built with the National Institute of Genetics (Mori, Kurokawa) and the ROIS Data Science Common Use Platform Bio-generative AI R&D Center (Higashi). Indexes 210,000+ MAGs, ~400 M sequences, ~100 B amino acids; searches complete in ~5–15 min at accuracy PEZY describes as comparable to DIAMOND and MMseqs2. Hosted at pzlast.nig.ac.jp, no registration. Paper: *Bioinformatics Advances*, DOI 10.1093/bioadv/vbag129.

Neither is an SC4s result and neither should be recorded as one.

### 9.4 Third-party SC3s porting work

- Zhu, Cui, Wang, Liu, "PEZY-DPM: Porting and optimization of the DPM Monte Carlo dose calculation code on the PEZY-SC3s MIMD processor," *Computer Physics Communications* 328:110319 (print Nov 2026; Crossref-indexed 2026-08-07).
- (Pre-baseline context) "Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s," PACT 2025, DOI 10.1109/PACT65351.2025.00045 (2025-11-03) — an AI convolution kernel on SC3s.

Both use the **SC3s** designation (see the naming note under §1). Independent porting work — largely from Chinese academic groups — continues on the shipping 7 nm part while SC4s remains unreleased.

### 9.5 Hot Chips 38

PEZY has **no talk** on the Hot Chips 38 program (Aug 23–25, 2026, Stanford Memorial Auditorium), i.e. no SC4s follow-up to the HC37 2025 presentation. HC38 is in the future at the time of writing; nothing from it may be cited as evidence for any specification, in either direction.
