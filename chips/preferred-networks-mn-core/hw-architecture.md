# Preferred Networks MN-Core Hardware Architecture

*as_of: 2026-08-08*
*chip: preferred-networks-mn-core*
*device_class: Compiler-Scheduled SIMD Accelerator (Japan)*
*Representative products: MN-Core 2 (shipping); MN-Core gen 1 (superseded); MN-Core L1100 / L1400 (pre-silicon, 2027)*

---

## Overview

MN-Core is a **host-sequenced VLIW/SIMD tree** designed by Preferred Networks with Kobe University. Its defining architectural property is subtractive: there is **no hardware cache at any tier, no hardware instruction scheduler, no per-PE program counter and no per-PE instruction decoder**. The compiler and the host CPU jointly perform the work that cache controllers, command schedulers and network controllers do in a conventional accelerator.

Three generations exist publicly:

1. **MN-Core** (gen 1) — TSMC 12 nm, 4 dies per package. Deployed in production (MN-3 supercomputer; Matlantis). Superseded.
2. **MN-Core 2** (gen 2) — TSMC N7, single 550 mm² die. The current shipping part.
3. **MN-Core L1100 / L1400** — announced, **pre-silicon, mockup only**, 2027 target. DRAM-on-logic 3D stacking.

Internal codename for MN-Core 2 is **GPFN3** (hence `gpfn3-smi`, `gpfn3_package_main`, `assemble3`); gen-1 dies are marked `GPFN-2G01`.

### What is deliberately absent

| Block found in a conventional accelerator | MN-Core 2 |
|---|---|
| Hardware data cache (any tier) | **Absent** — scratchpad at every level, compiler places every byte |
| Cache controller / replacement policy | Compiler: Location Planner + Scheduler spill/refill/`Forget` |
| Per-PE program counter and instruction decoder | **Absent** — the host streams one instruction stream over PCIe |
| Hardware instruction scheduler / OoO / scoreboard | **Absent** — VLIW, statically scheduled |
| Hazard interlocks | **Absent** — "it is the programmer's responsibility" |
| Branch unit / control flow | **Absent** — mask-register predication only |
| NoC router / mesh / crossbar | Broadcast/reduction **tree** with a fixed mode repertoire |
| Chip-to-chip fabric | **Absent on gen 2** (gen 1 had MN-Core DirectConnect) |
| Collectives offload engine | **Absent** — host-side MPI/gloo only |

PFN quantifies the payoff at **7.4 % of transistors in arithmetic units**, against its own anonymized measurements of 1.7 % ("GPU N"), 2.4 % ("Accelerator P") and 1.3 % / 0.8 % (two CPUs) — *vendor claim, PFN's own research*.

---

## 1. Compute Engine

### MN-Core 2 headline specifications

| Component | Specification |
|-----------|---------------|
| Architecture | Host-sequenced VLIW/SIMD tree; register-memory (not load-store) |
| Process node | TSMC **N7** |
| Die area | **550 mm²** (single die) |
| Transistors | **22 B** |
| Clock | **750 MHz** |
| PEs | **4096** |
| MABs / MAUs | **1024** MABs (= 4 PEs + 1 MAU each) → **1024 MAUs** *(derived)* |
| Peak FP64 | **12 TFLOPS** |
| Peak FP32 | **49 TFLOPS** |
| Peak pseudo-single (PFN "TF32") | **98 TFLOPS** |
| Peak half (PFN "TF16") | **393 TFLOPS** |
| Power | **330 W (design value)** |
| Efficiency | 37.24 / 148.9 / 297.9 / **1192** GFLOPS/W (FP64 / FP32 / pseudo-single / half) |
| Integer matrix datapath (INT8/INT4) | **Not disclosed** — SDM documents only FP/BFP matrix modes |

> ⚠️ **1192 GFLOPS/W is the half-precision number only.** FP64 efficiency is 37.24 GFLOPS/W. And PFN's "TF32"/"TF16" labels denote PFN block-FP formats — **neither is NVIDIA TF32 nor IEEE binary16**.

### MAB / MAU structure

**MAB (Matrix Arithmetic Block) = 4 PEs + 1 MAU (Matrix Arithmetic Unit).** The MAU holds a **double-buffered matrix register** — two "sides", so one can be multiplied against while the other is written.

Per-cycle matrix–vector capability `y = A·b + c` for an *m*×*n* matrix A:

| Precision | m | n | Operands | Result |
|---|---|---|---|---|
| Double | 2 | 4 | A, b block-FP double | c, y normal double |
| Single | 8 | 4 | A, b block-FP single | c, y normal single |
| Pseudo-single | 8 | 8 | block-FP only; shorter effective mantissa than single | c, y normal single |
| Half | 16 | 16 | A, b block-FP half | c, y normal **single** |

Element-wise vector FMA `y = a*b + c`: double m=4 (only elements 0/1 *or* 2/3 enabled per issue, so a full 4-element FMA needs two issues), single m=8, half m=16.

Peak reconciliation *(derived; matches PFN exactly)*: FP64 = 2·4·2·1024·750 MHz = 12.29 TF; FP32 = 8·4·2·1024·750 MHz = 49.2 TF; pseudo-single = 8·8·2·1024·750 MHz = 98.3 TF; half = 16·16·2·1024·750 MHz = 393.2 TF.

Each PE additionally has an **ALU** (integer + general ops including ReLU) and mask registers. Reduction units exist in the PE→L1BM→L2BM uplink paths but PFN **excludes them from peak FLOPS**.

### Number formats — not IEEE-754

| Precision | Sign | Exponent | Mantissa | Note |
|---|---|---|---|---|
| Half (HW, 16-bit) | 1 | **6** | **9** | **Not IEEE binary16** (1-5-10) |
| Single (SW, 32-bit) | 1 | 8 | 23 | IEEE-like field widths |
| Double (LW, 64-bit) | 1 | 11 | 52 | IEEE-like field widths |

**No denormals and no NaNs at all**: exponent all-zero ⇒ ±0 regardless of mantissa; all-ones ⇒ ±∞ regardless of mantissa. Max positive half = 4.29e+09; min positive normal half = 9.31e-10.

**Block floating point** (same field widths, **no hidden bit**) is mandatory for matrix–vector operands and, on MN-Core 2, is used **at all precisions** — gen 1 used BFP for half only. Assembly is **untyped**; storage is **big-endian**.

---

## 2. Data Path / Execution Model

### Instruction supply is the architecture

Two instruction classes:

- **MV instructions** — data movement among the "upper memories" (PDM, DRAM, L2BM, across groups). Execute **asynchronously**.
- **PE instructions** — **VLIW**; control everything at L2B and below, despite the name.

A PE instruction "control[s] the entire L2Bs and below in **4 cycles**. This 4-cycle unit is called a **step**." The SDM footnote is the key disclosure:

> *"This is because 1 cycle per instruction would saturate the PCIe bandwidth between the host and the board."*

**The instruction issue rate is set by the host PCIe link, not by the silicon.** No other accelerator in this registry states that as an explicit design constraint.

### Packing modes

| Mode | Stream unit | Addressing |
|---|---|---|
| **auto-stride** | 3 PE + 1 MV instruction | Addresses auto-increment each cycle within a step |
| **flat** | 2 PE + 1 MV instruction | GRF0/GRF1/LM0/LM1 addresses specified per cycle — more flexible, more instruction bandwidth |

Auto-stride is a strict subset of flat. Real hardware requires packed streams; the emulator does not.

### Hazards, control flow, synchronization

- **No hazard interlocks.** "Instruction execution is pipelined, and the results cannot be used immediately after the instruction is issued. **It is the programmer's responsibility** to leave cycles until the value written to one of the memories can be read."
- **No branches.** "Entirely composed of SIMD with no conditional branch." Predication via **mask registers** written by ALU result flags.
- **Explicit `wait` field** in every (fixed-width) PE instruction — always present, with a "don't wait" encoding — is the only synchronization mechanism against asynchronous MV instructions.
- **Forwarding paths are architectural operands**: `mauf` (MAU result), `aluf` (ALU result), `lbf` (L1BM→PE), `mreadf` (transposed matrix-register read), `nowrite` (dummy output). MN-Core 2 added direct feedback paths from MAU/IALU relative to gen 1.

### Determinism

*Vendor claim*: "Without cache interference or speculative behavior, execution is fully deterministic — the performance you measure in simulation is the performance you get on hardware." This is why the cycle-faithful emulator (`gpfn3_package_main`) is a credible development target rather than a rough model.

**Not disclosed**: instruction-stream bandwidth from host in GB/s; any on-board instruction buffering or staging capacity.

---

## 3. On-chip Memory

The chip is a **tree**: Top Level → 4 Groups → 8 L2Bs → 64 L1Bs → 1024 MABs → 4096 PEs. Capacities are documented in **LW** (long word = 64 bits = 8 bytes).

| Level | Unit | Documented capacity | Per-chip total *(derived)* | Managed by | Ports |
|-------|------|---------------------|---------------------------|-----------|-------|
| PE | **GRF0 / GRF1** | 256 LW = 2 KiB each | 16 MiB | Compiler (inside operator impls) | 1R1W, 2 LW/cycle each |
| PE | **LM0 / LM1** | 2 Ki LW = 16 KiB each | **128 MiB** | Compiler (`Location`) | 1RW, 2 LW/cycle each |
| PE | **T-register** | 8 LW = 64 B | 256 KiB | Compiler | 1R1W, 2 LW/cycle |
| L1B | **L1BM** | 8 Ki LW = 64 KiB | 4 MiB | Compiler (inside operator impls) | Not disclosed |
| L2B | **L2BM** | 32 Ki LW = 256 KiB | 2 MiB | Compiler (inside operator impls) | Not disclosed |
| Group | **PDM** (PIU Data Memory) | 4 MiB per group | 16 MiB | Compiler / runtime | Not disclosed |

**Aggregate ≈ 166 MiB** — *our arithmetic from the per-level figures. PFN publishes no aggregate on-chip SRAM figure.*

**There is no cache at any level.** PFN on the LM tier: it "combines the advantages of high-speed SRAM access from the PE with substantial total capacity, enabling minimal DRAM access and effectively improving the overall B/F ratio. However, it lacks advanced memory management features like cache memory, and **all data transfers must be explicitly specified**."

Two structural facts that shape the whole software stack:

1. **Only Group 0's PDM is physically connected to the host over PCIe.** *All* host↔device I/O funnels through that single 4 MiB SRAM.
2. The compiler's `Location` abstraction can only place tensors in **DRAM or LM0/LM1**. "Regarding GRF0/GRF1 and L1BM/L2BM, these are not assigned as Locations." Those tiers are transfer buffers inside operator implementations.

Address namespaces in assembly: DRAM `$d`, PDM `$p`, L2BM `$lc`, L1BM `$lb` are LW-addressed; LM0 `$m`, LM1 `$n`, GRF0 `$r`, GRF1 `$s` are SW (32-bit) addressed.

**Derived aggregate LM bandwidth** *(arithmetic implication of the documented port widths, not a PFN figure)*: 4096 PEs × 2 LW/cycle × 8 B × 750 MHz ≈ **49 TB/s per bank**, ≈ **98 TB/s across LM0+LM1**.

**L1BM / L2BM SRAM bandwidth and on-chip ECC: not disclosed.**

---

## 4. On-chip Interconnect

No NoC, no mesh, no crossbar. Each tree level is a **broadcast/reduction network** with a fixed repertoire.

| Level | Composition | Transfer modes |
|---|---|---|
| **L1B** | 16 MABs + 1 L1BM | Broadcast, individual, **reduction**, distribute/collect — controlled by PE instructions |
| **L2B** | 8 L1Bs + 1 L2BM | Same repertoire between L2BM and L1Bs |
| **Group / top** | Data engines per group | Move data among PDM, DRAM and L2BM, within and across groups |

**Symmetry is an architectural guarantee**: for every downward mode (e.g. `distribute`) there is a matching upward mode (`collect`) that reassembles the identical layout. The SDM states this explicitly as a layout-consistency property, and the compiler's Layout Planner depends on it.

Documented MV-instruction throughputs:

| Path | LW/cycle | At 750 MHz *(derived)* |
|---|---|---|
| DRAM ↔ L2BM parallel transfer | 16 per group | 96 GB/s per group |
| Inter-group broadcast / reduce | 8 | 48 GB/s |
| L2BM-side collect / distribute | up to 32 | up to 192 GB/s |
| PDM ↔ L2BM individual | 8 (PDM side) / 1 (L2BM side) | — |
| PDM ↔ DRAM | 8 (PDM side) / 2 (DRAM side) | — |

PFN's rationale (Makino, HC36): broadcast and reduction "are the two most important communication patterns for PEs used for tensor operations in deep learning. Compared to shared cache, energy consumption becomes much smaller."

---

## 5. Off-chip Memory

| Parameter | MN-Core 2 | MN-Core gen 1 |
|---|---|---|
| Type | **GDDR6** (see caveat) | LPDDR4 |
| Capacity | **16 GiB** (4 GiB × 4 Groups) | 32 GB |
| Aggregate bandwidth | **512 GB/s** (~128 GB/s per Group) | 400 GB/s |
| Bus width / device count / per-pin rate | **Not disclosed** | Not disclosed |
| ECC | **Not disclosed** | Not disclosed |

> **Caveat on the memory type.** PFN's own Hot Chips 36 slides disagree: slide 10 says "external DRAM (GDDR6)", slide 13 says "16 GB **GDDR6X**". The driver knob is `gpfn3-smi config <dev> clock --core 750 --gddr6 15000`. Record it as GDDR6-class with the inconsistency noted; do not silently pick one.

16 GiB at 512 GB/s is modest — roughly a twelfth of an H100's HBM bandwidth. It is exactly the constraint MN-Core L1000's DRAM-on-logic stacking is meant to remove, and it is why the 128 MiB software-managed LM matters so much: the HIMENO result (9.03 TF FP64, ~75 % of peak on a memory-bound stencil) is only achievable because temporal blocking keeps working sets on-chip.

---

## 6. Host Interface / Package

| Parameter | Value |
|---|---|
| Host interface | **PCIe Gen5 ×16** (gen 1: PCIe Gen3 ×16) |
| PCI vendor ID | **`0ccd`** (Preferred Networks) |
| `lspci` string | `Processing accelerators: Preferred Networks, Inc. MN-Core 2` |
| Device nodes | `/dev/mnc2p<bus>s<n>` (e.g. `/dev/mnc2p28s0`) |
| Board contents | 1 chip + peripheral circuits (GDDR6, PIU) |
| Card mechanical form factor | **Not disclosed** (FHFL / OAM / proprietary unknown) |
| Clock and MAB configuration | **Volatile across power cycles**; re-applied by the kernel module at load time |
| Cooling | Air originally (manual advises pinning fans ≥70 % "to prevent MN-Core 2 thermal runaway"); a **direct-liquid-cooled** high-density server was developed in 2025 |
| Measured TDP / idle power / thermal limits | **Not disclosed** |
| Virtualization / partitioning | **Not disclosed** — the runtime takes an exclusive device lock, but PFN does not say whether partitioning is possible |

---

## 7. Scale-up Interconnect

| Generation | Fabric | Notes |
|---|---|---|
| **MN-Core 2** | **None** | No chip-to-chip link, no NVLink/Infinity-Fabric analogue, no device-side collectives. Multi-board goes through the host: OpenMPI + `torch.distributed` on the **gloo** backend with host-memory send/recv |
| **MN-Core gen 1** | **MN-Core DirectConnect** + DirectConnect Switches | "An interconnect technology developed specifically for MN-Core"; 32 MN-Core Servers + 2 switches = one "Zone". Bandwidth not disclosed |

This is the most consequential limitation for LLM-scale work: tensor- and pipeline-parallel execution across boards is bounded by host memory and PCIe, not by a device fabric. **PFN has never stated whether DirectConnect was dropped for gen 2 or simply not used in MN-Server 2** — record the absence of documentation, not a confirmed removal.

---

## 8. Scale-out Interconnect

Standard host networking only. MN-Server 2 V1 carries **2 × NVIDIA ConnectX-6 100 GbE dual-port** plus onboard **2 × Intel X710 10 GbE**. There is no accelerator-side RDMA path and no collective offload.

---

## 9. Product Portfolio

| SKU | Model | Boards | Peak (FP64 / half) | Memory | Form factor | List price (ex-tax) |
|---|---|---|---|---|---|---|
| MN-Core 2 Devkit V1 | MNC2DV1 | 1 | 12 TF / 393 TF | 16 GiB | Desktop workstation, 850 W ATX | **¥2,000,000** |
| MN-Server 2 V1 | MNS2V1 | 8 | 98 TF / 3,146 TF | 128 GiB | 5U rackmount, max 4200 W | **¥20,000,000** |
| MN-Pod 2 | — | 48 (6 servers) | **590 TF / 18,874 TF** | 768 GiB | 19″ 42U rack | — |
| MN-3 (gen 1) | — | 128–160 MN-Core | — | — | Supercomputer | — |

MN-Server 2 V1 details: 2 × Xeon Platinum 8480+, 1 TB DDR5-4800 (16×64 GB), 3 × Micron 7450 PRO 15.3 TB + 1 × 960 GB NVMe, 4 × 3000 W PSU (2+2 redundant), W449 × H220.75 × D833.1 mm. Devkit: Core i5-14500, 64 GB DDR5, 1 TB NVMe, 2.5 GbE, W327 × H565.2 × D599.2 mm; basic package MNC2DV1bk with delivery/installation ¥2,500,000, Japan only.

MN-Pod 2 per-server and per-pod peaks above are the whitepaper's own figures (590 TF FP64 / 2359 FP32 / 4719 pseudo-single / 18,874 half); the 8-board server row is derived by multiplication.

---

## 10. Generation 1 — MN-Core

| Component | Specification |
|-----------|---------------|
| Process | TSMC **12 nm (N12)** |
| Package | **4 dies**, 764 mm² each |
| MABs | 512/die, **2048 total** ⇒ **8192 PEs** *(derived)* |
| L2Bs | 16 |
| Clock | 500 MHz |
| DRAM | 32 GB LPDDR4, 400 GB/s |
| Host | PCIe Gen3 ×16 |
| Power | 500 W (estimated) |
| Peak | 32.8 TF DP / 131 TF SP / 524 TF HP (product page); **532 TF FP16** on the HC36 slide — unexplained discrepancy between PFN's own two sources |
| Efficiency | 0.066 / 0.26 / 1.0 TFLOPS/W |
| Scale-up | MN-Core DirectConnect + switches |

### MN-3 Green500 record (gen-1 silicon)

| List | GFLOPS/W | Rank | Nodes / MN-Core |
|---|---|---|---|
| Jun 2020 | 21.11 | **#1** | 40 / 160 |
| Nov 2020 | 26.04 | #2 | 40 / 160 |
| Jun 2021 | **29.70** | **#1** | 40 / 160 |
| Nov 2021 | **39.38** | **#1** | 32 / 128 |
| Jun 2022 | 40.90 | #5 | 32 / 128 |

The June 2021 entry is **independently confirmed by TOP500**. PFN's prose elsewhere says "November 2022" where its own table (and the November 2021 press release, and TOP500) say November 2021 — treat as a PFN typo. **No Green500 result exists for MN-Core 2.**

MN-3a node: 4 MN-Core boards, 2-way Xeon 8260M, 384 GB DDR4, 3 TB Intel Optane DC PMem, MN-Core DirectConnect + 100 GbE; 48 nodes = 1.5 Zones.

---

## 11. MN-Core L Series — Pre-Silicon (Do Not List as Shipping)

| Item | Status |
|---|---|
| Family / architecture name | **MN-Core L1000** |
| Announced SKUs (2026-06-01) | **MN-Core L1100** (low-power), **MN-Core L1400** (high-performance) |
| Target | **2027** (slipped from the 2026 target announced 2024-11-15) |
| Physical evidence | **Mockup only** — SC25 booth Nov 2025; June 2026 press-release photo captioned "(mockup)"; roadmap tile marked 開発中 |
| Memory technology | **DRAM-on-logic 3D stacking**: "PFN's proprietary logic and DRAM are vertically stacked, with many electrodes arranged across the surface." **Not HBM, not commodity DIMM DRAM** |
| Bandwidth claim | "**50 times higher memory bandwidth than MN-Core 2**" — *vendor claim, pre-silicon*. Implies ~25.6 TB/s, **derived from a claim, not a published spec** |
| Capacity claim | "a single MN-Core L1400 card is designed to provide the memory bandwidth and memory capacity required for inference" of 70 B-parameter LLMs — *vendor claim* |
| Earlier (different) claim | 2024-11-15: "up to a ten-fold increase in computing speed compared with conventional processors such as GPUs" — a different metric entirely; do not merge with the 2026 bandwidth claim |
| Toyota FRC engagement | **Joint research on software** for an unreleased chip. Not a deployment, not a customer win; robot tests only "following 2027 shipments" |
| Process node, foundry, die size, stack vendor/layers, memory capacity, absolute bandwidth, peak FLOPS, dtypes, TDP, form factor, price, interconnect | **All not disclosed** |
| L1100 vs L1400 differentiation | **Not disclosed** beyond the labels "low-power" and "high-performance" |

---

## 12. Roadmap

| Generation | Target workloads | Status | Timeline |
|---|---|---|---|
| MN-Core (TSMC 12 nm) | AI training / inference / HPC | shipped | dev 2016– ; internal use 2020– ; external compute 2023– |
| MN-Core 2 (TSMC 7 nm) | AI training / inference / HPC | shipping | trial ops 2023– ; external sales + PFCP 2024– |
| **MN-Core L1000** | AI inference | 開発中 (in development) | dev 2024– ; planned **2027** |
| **MN-Core L2000** | Large-scale AI inference + HPC | 開発中 (in development) | dev 2025– ; planned **2028** |
| 次世代 (next generation) | AI training, large-scale inference + HPC | 検討中 (under consideration) | — |

**MN-Core L2000 appears in no press release** — PFN's roadmap graphic is the only public source. Every specification is not disclosed.

Two unreconciled statements exist for the post-MN-Core-2 training part:

- **Hot Chips 36, 2024-08-27** (Makino, "Future directions"): "MN-Core Next for learning: >10× peak performance, >30× application performance, **Samsung SF2**"; "MN-Core Next for LLM Inference: ultra-high memory bandwidth, >20× inference performance". *Vendor claim, dated.*
- **2025-01-08**: PFN / **Rapidus** / SAKURA internet basic agreement — "Rapidus manufactures a new model of AI semiconductors in the MN-Core series to be designed by PFN".

PFN has never publicly reconciled them. **Report both as separate dated statements.**

---

## 13. Deployment and Maturity

| Generation | Status verb | Evidence |
|---|---|---|
| MN-Core (gen 1) | **Deployed in production**, now superseded | MN-3 supercomputer; Matlantis sold commercially to ENEOS since Aug 2023 |
| MN-Core 2 | **Shipping as complete systems** (Japan) + **cloud service** | Published Japanese list prices since ~Sept 2024; Makino at HC36: "As of Aug. 23, 2024, MN-Core 2 is commercially available"; PFCP from Oct 2024. **No merchant bare-chip/board sales, no non-Japanese customers, no third-party OEM** |
| MN-Core L1100 / L1400 | **Announced, pre-silicon** | Mockup only, 2027 target |
| MN-Core L2000 / next-gen | **Roadmap tiles only** | Roadmap graphic only |

**Confirmed board counts**: 30 nodes = **240 boards** at IIJ Matsue Data Center Park + 2 nodes = **16 boards** at JAIST Ishikawa (NEDO testbed, direct liquid cooling, since July 2025). Moved into IIJ's Shiroi DCC **AImod** modular liquid-cooled data center from April 2026 — **board count there not disclosed**. PFCP fleet size **not disclosed**.

### Vendor-reported performance (no third-party audit exists)

PFN's own Hot Chips 36 measurements vs A100 — *vendor claim, 2024-08-27*:

| Workload | MN-Core 2 | A100 |
|---|---|---|
| GCN (PFN internal) | 5.41 TF (FP32) | 3.17 TF |
| ResNet-50 training | 77 TF (FP16) | 33.2 TF (BF16) |
| ResNet-50 inference | 154 TF (FP16) | 33.7 TF (BF16) |
| **HIMENO** | **9.03 TF (FP64)** | 0.634 TF |
| OpenFDTD | 0.655 TF (FP32) | 0.488 TF |

LLM inference (SDK v0.6, PLaMo-3-NICT-2B-base): **5,100 tok/s prefill (512 tok), 11.8 tok/s decode** — *vendor claim*. **No MLPerf or other third-party-audited results exist for MN-Core 2**; the only independently verified performance datum in the whole line is the gen-1 Green500 ranking.

---

## 14. Explicitly Not Disclosed

- Aggregate on-chip SRAM as a single published figure (~166 MiB is our arithmetic)
- GDDR6 bus width, device count, per-pin rate; PFN's slides disagree on GDDR6 vs GDDR6X
- Board mechanical form factor and card dimensions
- Per-board idle power, measured TDP, thermal limits
- L1BM / L2BM SRAM bandwidth
- Instruction-stream bandwidth from host (GB/s); on-board instruction buffer capacity
- Any integer (INT8/INT4) matrix datapath in the MAU
- Yield, unit volume shipped, total fleet board count beyond 240 + 16
- Die photo / floorplan / per-block area breakdown
- Sparsity support, weight compression, on-chip decompression
- Virtualization / MIG-style partitioning / single-board multi-tenancy
- RAS: ECC on DRAM or SRAM, error reporting, accelerator secure boot
- All MN-Core L1100 / L1400 silicon specifications; all MN-Core L2000 specifications
- Whether the post-MN-Core-2 training part is Samsung SF2 or Rapidus-manufactured
- Whether MN-Core DirectConnect was dropped for gen 2 or merely unused

---

## Sources

- [MN-Core 2 White Paper (2023-11-12)](https://projects.preferred.jp/mn-core/assets/MN-Core_2_whitepaper_en.pdf)
- [MN-Core 2 Software Developer Manual (EN), rev. 2026-06-02](https://projects.preferred.jp/mn-core/assets/mncore2_dev_manual_en.pdf)
- [MLSDK 0.7 Hardware Specification](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/hardware_specification.html)
- [MLSDK 0.7 Technical Notes](https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/technical_notes.html)
- [Hot Chips 36 slides — J. Makino, MN-Core 2 (2024-08-27)](https://hc2024.hotchips.org/assets/program/conference/day2/15_HC2024.Preferred.Makino.final.pdf)
- [MN-Core 2 hardware catalog (2024-08)](https://projects.preferred.jp/mn-core/assets/MN-Core2-hardware-catalog.pdf)
- [MN-Core 2 Devkit / MN-Server 2 installation & operation manual](https://projects.preferred.jp/mn-core/assets/MN-Core2-Devkit-MN-Server-2-installation-operation-manual.pdf)
- [MN-Core Series product page](https://projects.preferred.jp/mn-core/en/)
- [PFN Supercomputers page — MN-3, MN-Core DirectConnect](https://projects.preferred.jp/supercomputers/)
- [SDK Hub — Architecture](https://dev.mn-core.com/architecture/)
- [TOP500 Green500 June 2021 (independent)](https://www.top500.org/lists/green500/2021/06/)
- [ServeTheHome — MN-Core 2 for HPC and AI (independent)](https://www.servethehome.com/preferred-networks-mn-core-2-for-hpc-and-ai/)
- [PFN AI Chips business page](https://www.preferred.jp/en/business/chips/)
- [PFN MN-Core roadmap graphic](https://www.preferred.jp/images/business/chips/roadmap_img.png)
- [PR 2026-06-01 — MN-Core L1100 / L1400, Toyota FRC](https://www.preferred.jp/en/news/pr20260601)
- [PR 2024-11-15 — MN-Core L1000 development begins](https://www.preferred.jp/en/news/pr20241115)
- [PR 2025-11-13 — SC25, L1000 mockup](https://www.preferred.jp/en/news/pr20251113)
- [PR 2025-09-11 — NEDO testbed, 240 + 16 boards](https://www.preferred.jp/en/news/pr20250911)
- [PR 2026-03-23 — AImod modular DC at IIJ Shiroi](https://www.preferred.jp/en/news/pr20260323)
- [PR 2025-01-08 — PFN / Rapidus / SAKURA internet](https://www.preferred.jp/en/news/pr20250108)
