# Stream Computing (希姆计算) NeuralScale Hardware Architecture

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*
*Representative products: P920 SoC — STCP920 / STCP950L / STCP950P / STCP980L / STCP980P PCIe inference cards*

---

## Overview

There is exactly **one public NeuralScale silicon generation**: the **P920** SoC, TSMC 12 nm FinFET,
400 mm², 32 NeuralScale cores, 128 TFLOPS FP16 / 256 TOPS INT8 at 1.0 GHz, 130 W chip TDP, 16 GB LPDDR4X.
The five shipping SKUs are clock and power bins of it, and the compiler target for all of them is `npu-v1`.

The architecture's defining characteristics:

1. **A RISC-V scalar core per compute tile, driving three decoupled engines.** Per-core MIMD, in-order,
   up to 3 instructions/cycle (one each to VME, MME, MTE). Not SIMT; not a fixed-function systolic array.
2. **A vendor-custom RISC-V vector/matrix extension in shipping silicon** — RVV v0.8 plus 53 custom
   instructions on the custom-3 opcode, with shape state in 22 unprivileged CSRs.
3. **An entirely software-managed, non-coherent memory hierarchy.** Every tensor buffer is addressed
   explicitly by the program; the only hardware caches are the per-core 64 KB scalar I$/D$.
4. **No scale-up fabric and no collective hardware.** Multi-card work runs over PCIe peer-to-peer, measured
   at ≤9 GB/s.

Its defining limitation is memory: 16 GB of LPDDR4X at ~108 GB/s behind 128 TFLOPS FP16, and FP16/INT8-only
datatypes.

---

## 1. Compute Engine

### NeuralScale core (NPC)

| Component | Specification |
|-----------|---------------|
| Scalar core | AndesCore N25F licensed IP — 32-bit RISC-V, **RV32G**, 5-stage in-order, dynamic branch prediction |
| Scalar caches | 64 KB L1 I$ + 64 KB L1 D$ (hardware-cached; **control path only**) |
| Issue model | 4-stage NPC pipeline (decode → issue → execute → write-back); three instruction buffers, one per engine; **in-order issue, up to 3 instructions/cycle**; blocks only on address overlap with in-flight instructions |
| **VME** (Vector MAC Engine) | 64 FP16 MACs + **POLY** module (`exp`, `div`, `sqrt`); executes base RVV *and* custom vector instructions whose operands are L1/IM byte addresses in GPRs |
| **MME** (Matrix MAC Engine) | **64 × 32 = 2048 MAC units**; each MAC = 1× FP16 or **2× INT8** per cycle; reads Data Input Buffer + Weight Buffer, writes Intermediate Buffer |
| **MTE** (Memory Transfer Engine) | Explicit data movement L1↔remote L1, L1↔LLB, L1↔DRAM over the NoC; also **LLB broadcast** to all corresponding L1 Buffers |
| RVV configuration | VLEN = 1024 bits, ELEN = 16 bits (RVV v0.8) |
| Vector register file | Dedicated "REG Bank" in the NPC; **architectural register count not disclosed** |

**Supported datatypes: FP16 and INT8 only.** No BF16, no FP8, no FP4, no INT4, no native FP32 compute path.
This is a hard constraint on the workloads the part can serve — every model in the vendor's support table is
FP16 or w8a8/w8a16.

### P920 SoC

| Component | Specification |
|-----------|---------------|
| NeuralScale cores | 32 |
| Clusters | 4 NPC Clusters (`stc-smi -q` reports `Cluster count: 4`); cores-per-cluster **not disclosed** (32÷4 = 8 is derived) |
| Management CPU | ARM Cortex-A53 — in-order, 8-stage, dual-issue; PPI + up to 64 SPI interrupts; boots and manages the SoC |
| Peak FP16 | **128 TFLOPS** @ 1.0 GHz |
| Peak INT8 | **256 TOPS** @ 1.0 GHz |
| Process | TSMC 12 nm FinFET |
| Die area | 400 mm² |
| Chip TDP | 130 W @ 1.0 GHz (card TDP 150–160 W) |
| Frequency domains | Independently settable "east" and "west" SoC frequency/voltage domains |
| Power domains | 4 independent banks; host can gate 1–3 (power-save) or all 4 (standby) over PCIe |
| DVFS | 624 / 800 / 900 / 1000 / 1100 / 1200 / 1300 / 1400 MHz (documents disagree on which subset is exposed) |
| Peripherals | UART, SPI, I²C, PWM, RTC |
| Transistor count / package | **not disclosed** |

*Throughput reconciliation (derived):* 2048 MME MACs × 2 flop × 32 cores × 1 GHz = 131.1 TFLOP/s, quoted as
"128 TFLOPS"; ×2 for INT8 = 262 TOP/s, quoted as "256 TOPS". The VME's 64 MACs/core (+4.1 TFLOPS) appear to
be excluded from the headline figure.

---

## 2. ISA

### Shipping product ISA — proprietary

- **RV32G + RVV v0.8 + 53 custom instructions** on the **custom-3 major opcode `1111011`**, designated
  **OP-VE**. Fixed 32-bit encoding:
  `[31:26] funct6 | [25] dmc | [24:20] rs2 | [19:15] rs1 | [14] dm | [13:12] opm2 | [11:7] rd | [6:0] 1111011`
  — sources and destination stay in base-ISA register positions to keep decode simple.
- `opm2` selects operand shape class (matrix-matrix / matrix / matrix-vector / matrix-scalar); `{dmc,dm}`
  selects full-element, vertical or horizontal direction.
- Published mnemonics include `veadd`, `veemacc`, **`memul`** (matrix multiply), **`meconv`** (convolution),
  `velkrelu`, and `mov` (MTE transfer).
- **22 unprivileged vector CSRs** carry shape/config state (`0x400 shape_s1`, `0x401 shape_s2`,
  `0x408 conv_FM_in`, `0x422 mte_shape`, …), written with ordinary RISC-V CSR instructions.
- This is a **vendor-custom extension, not the ratified RISC-V matrix/IME extension** — custom-3 is exactly
  the opcode space RISC-V reserves for non-standard extensions. Confirmed live in shipping silicon by the
  SDK's own `inst_decoder.py` (`0x06ABE8FB → veadd.mv.dimw (x17), (x23), (x10)`).

### Open RISC-V Matrix extension proposal — separate, not implemented here

Stream Computing chairs work in RISC-V International's Matrix Task Group on an **open** matrix extension
(`riscv-stc/riscv-matrix-spec`, CC-BY-4.0, v0.5 announced Nov 2024): 8 tile registers `tr0–tr7` + 8
accumulators `acc0–acc7`, parameters ELEN/MLEN/RLEN/AMUL, `mtype` CSR via `msettype`, sub-extensions Zmi4
(INT4), Zmv, Zmi2c, Zmc2i, Zmsp (sparse). **The P920 does not implement it.** Conflating the two is the most
common error in secondary coverage of this vendor.

---

## 3. Data Path / Execution Model

The host launches kernels onto N NPCs with CUDA-shaped syntax (`kernel<<<NCORE>>>(args)`); each NPC runs the
same kernel and identifies itself with `CoreID` / `CoreNum`. Within an NPC, performance comes entirely from
statically overlapping the three engines:

```
MCU (RISC-V scalar)  ─┬─→ instruction buffer → VME  (64 FP16 MACs + POLY)
                      ├─→ instruction buffer → MME  (2048 MACs, 64×32)
                      └─→ instruction buffer → MTE  (explicit L1 ↔ LLB ↔ DDR moves)
   in-order issue, ≤3 instr/cycle, stall only on address overlap
```

The compiler (or kernel author) must software-pipeline MTE loads against MME/VME compute; the profiler is
built around exactly this, reporting `PAL(MTE/MME)`, `PAL(MTE/VME)`, `PAL(VME/MME)` and `PAL(ALL)` overlap
percentages per NPC.

**Measured on BERT** (CARRV'21 trace analysis): NPCs occupy 95% of total cycles; DMAs overlap with other work
96% of the time; within an NPC, **MME occupies 78% of cycles**, MTE overlaps MME and/or VME 92% of the time,
and VME serialises 45% of its time on data dependencies from MME.

Kernel-visible memory spaces mirror the hardware exactly: `__local__` → L1 Buffer, `__shared__` → LLB,
plain device pointers → DDR, `IM_BUFFER_START` → Intermediate Buffer. Shapes are programmed into CSRs before
instruction issue rather than encoded in operands.

---

## 4. On-chip Memory — software-managed, non-coherent

| Level | Scope | Capacity | Bandwidth | Managed by |
|-------|-------|----------|-----------|------------|
| Vector register file | per NPC | VLEN 1024 b × (count not disclosed) | — | compiler / ISA |
| **L1 Buffer** | per NPC | **1.25 MiB** (1 MiB Data IO + 0.25 MiB Weight) | 512 GB/s *(single third-party-hosted figure)* | software (MTE + explicit addressing) |
| **Intermediate Buffer (IM)** | per NPC | **256 KB** | not disclosed | software |
| Scalar L1 I$ / D$ | per NPC | 64 KB + 64 KB | not disclosed | **hardware cache — control path only** |
| **LLB (Last Level Buffer)** | per cluster | **8 MiB** (32 MiB total, 8 × 4 MB banks on the NoC) | **contested** — 17 TB/s (CARRV'21) vs 256 GB/s per cluster (vendor table) | software (MTE, sysDMA) |

**Total on-chip SRAM ≈ 80 MiB** (derived: 32 × 1.5 MiB + 32 MiB LLB), plus 4 MiB of scalar caches.

The NoC is explicitly labelled a **"non-coherent interconnect"**, and every buffer in the compute path is
addressed explicitly: the custom VME/MME instructions take L1 / IM byte addresses in general-purpose
registers, and all inter-level movement is an explicit MTE or sysDMA instruction. This makes tiling and data
placement a first-class compiler problem — which is precisely how the MLTC compiler describes its own GOAT
framework ("most of the code is managing the multi-level storage hierarchy and the data movement between
levels").

---

## 5. Off-chip Memory

| Spec | Value |
|------|-------|
| Type | **LPDDR4X** |
| Capacity | **16 GB** (4 GiB per cluster × 4) |
| Interface | 256-bit, 3733 MT/s (1867 MHz), 4 independent DDR subsystems |
| Bandwidth (card manuals) | **108 GB/s** |
| Bandwidth (vendor table hosted at ByteDance) | 119.4 GB/s |
| Bandwidth (derived from the interface) | 119.5 GB/s |
| Bandwidth (CARRV'21 design description, LPDDR4 @ 4266 MT/s) | 136 GB/s theoretical |
| DMA | 2 DMA controllers per DDR subsystem (`sysdma_0` / `sysdma_1`; C0 = DDR→LLB, C1 = LLB→DDR) |

The reason the manuals derate to 108 GB/s is **not disclosed**.

**Consequence.** ~108 GB/s against 128 TFLOPS FP16 requires >1000 FLOP/byte of arithmetic intensity to
saturate compute. The vendor's own LLM table is the clearest evidence of the effect:

| Model | Cards | Single-stream tok/s |
|---|---|---|
| Qwen2-7B-Instruct | 2 | 10.39 |
| Jiuzhou-7B (vendor's own LLM) | 2 | 10.3 |
| DeepSeek-R1-Distill-Llama-70B | 16 | 4.42 |
| Qwen2-72B-Instruct | 16 | 4.98 |

---

## 6. Host Interface / Package

| Spec | Value |
|------|-------|
| Host link | PCIe 4.0 ×16 (Lane Reversal supported) |
| BARs | BAR0 16 GB prefetchable; BAR2 32 MB non-prefetchable; BAR4 1 GB prefetchable; BIOS "Above 4G Decoding" required |
| Functions | 1 PF (64-bit); **no SR-IOV VFs**; PCIe passthrough to KVM supported |
| PCI IDs | Vendor 0x23e2 / Sub-Vendor 0x23e2; per-SKU DIDs 0x0100 / 0x0103 / 0x0104 / 0x0105 / 0x0106 |
| Form factor | Single-width, ¾-length, full-height (268.44 × 111.15 mm), 721.2 g, passive cooling, one PCIe 8-pin aux |
| OOB management | SMBus (7-bit 0x2A): board power, board/chip temp, PCI IDs, MCU FW version, **power brake** (0xF6 ← 0x55), serial/part number, chip and MAC voltage |
| Thermals | Board inlet 0–50 °C; chip 0–90 °C (recommend <70 °C); alert 91 °C, shutdown 95 °C default; bidirectional airflow |
| Firmware | Two signed images — NPU-ctrl (SoC init, V10.3.7/10.3.8) and MCU (card power, V10.0.13/10.0.14) |

---

## 7. On-chip Interconnect and Synchronisation

| Property | Value |
|---|---|
| Topology | **4×4 mesh**; every component attaches to a router; all links bidirectional |
| Planes | Control bus 32 b/direction, data bus 512 b/direction (separated) |
| Link bandwidth | **64 GB/s per direction per link, 128 GB/s combined** @ 1.0 GHz |
| Coherence | **Non-coherent** (labelled as such in the P920 block diagram) |
| Sync | **HSYNC** — partitions the 32 cores into up to **16 groups**, group size app-configurable |
| Not disclosed | NoC IP vendor, router microarchitecture, virtual channels / QoS, exact ordering model, HSYNC programming API |

> A company-authored RISC-V International blog post is sometimes summarised as "4×6 mesh, 1 TB/s". The
> peer-reviewed CARRV'21 paper — same authors, same product — says 4×4 mesh and 64 GB/s per direction.
> **Use CARRV.**

HSYNC is not exposed in the public SDK, which presents only the 4-cluster resource model and an SHC `sync()`
intrinsic; the mapping between HSYNC groups and clusters is not documented.

---

## 8. Scale-up Interconnect

| Chip | Fabric | Bandwidth |
|------|--------|-----------|
| P920 (shipping cards) | **None** — PCIe peer-to-peer only | measured ~9.0 GB/s (LLB destination) / ~0.89 GB/s (DDR destination) |

The SoC *has* the capability: two PCIe subsystems, each up to 16 lanes and each configurable as endpoint or
root complex, with CARRV'21 describing PCIE1 as "usually configured as a root complex for scalability,
connecting to other SoC chips". **But no shipping card manual documents any chip-to-chip or card-to-card
link**, and whether that path is wired on any STCP card is **not disclosed**. `stc-topo` reports only PCIe
path classes (LOC / PIX / PXB / PHB / SYS) and CPU/NUMA affinity.

Measured with the vendor's own `p2p_perf` on a 2-card host:

| Transfer (NPU0 → NPU1) | Measured |
|---|---|
| DDR → DDR | 888.6 MB/s |
| DDR → LLB | 9005 MB/s |
| LLB → DDR | 886.5 MB/s |
| LLB → LLB | 9014 MB/s |

This is the concrete cost of the 16-card tensor-parallel LLM configurations above: peer traffic peaks around
9 GB/s and collapses below 1 GB/s whenever the destination is remote LPDDR.

---

## 9. Scale-out Interconnect

Standard datacenter Ethernet at the server level. **No RDMA fabric, no collective-communication hardware,
and no NCCL/CNCL-equivalent library.** Multi-card parallelism is expressed as MLTC compiler operations
(`stc.device_broadcast`, `stc.device_reduce_*`, `stc.device_allconcat`, `stc.device_sync`) plus pipeline
partitioning, executed over PCIe peer-to-peer.

---

## 10. Product Portfolio

| SKU | Base clk | Max clk | FP16 | INT8 | Card TDP | Memory | Form factor |
|-----|----------|---------|------|------|----------|--------|-------------|
| STCP920 | 1.0 GHz | not listed | 128 TFLOPS | 256 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s | PCIe 4.0 ×16 SW ¾L FH |
| STCP950L | 1.0 GHz | 1.2 GHz | 153 TFLOPS | 309 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s | PCIe 4.0 ×16 SW ¾L FH |
| STCP950P | 1.0 GHz | 1.2 GHz | 145 TFLOPS | 290 TOPS | 150 W | 16 GB LPDDR4X @ 108 GB/s | PCIe 4.0 ×16 SW ¾L FH |
| STCP980L | 1.0 GHz | 1.4 GHz | 179 TFLOPS | 358 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s | PCIe 4.0 ×16 SW ¾L FH |
| STCP980P | 1.0 GHz | 1.4 GHz | 167 TFLOPS | 335 TOPS | 150 W | 16 GB LPDDR4X @ 108 GB/s | PCIe 4.0 ×16 SW ¾L FH |

All five report identical topology (4 clusters, 1.25 MiB L1/core, 8 MiB LLB/cluster, 4 GiB DDR/cluster),
identical physicals and weight, shared firmware images, and one compiler architecture target. The L SKUs'
quoted throughput is exactly 128 TFLOPS × max clock at 160 W; the P SKUs quote less at 150 W. **Die identity
is not disclosed** — treat "one die, five bins" as strongly evidenced, not stated.

Vendor-claimed system products: 2U4-, 4U8- and 4U16-card AI servers; host-CPU compatibility with Intel, AMD,
Hygon, Phytium, Kunpeng and Loongson; an "AI 一体机" appliance line; and a 智算云平台 cloud platform.

---

## 11. What Is Not a Product

- **Second-generation silicon (`npu-v2`).** The vendor's timeline says the 2nd-gen tape-out was blocked by
  the US BIS export ban in 2022 and that a compliant redesign was completed in 2024. **Microarchitecture,
  core count, process node, foundry, memory, datatypes, TDP, tape-out/sampling status and product name are
  all not disclosed.** The only public trace is that MLTC 1.6.1 accepts `--arch=npu-v2`. It must not be
  entered into the chip registry as silicon.
- **"RISC-V AI super node" (RISC-V AI 超节点).** Marked 研发中 (under development) on the vendor site and
  described as a "CPU+NPU+DPU AI-native computing unit". No topology, node count, interconnect, bandwidth or
  date.

---

## Sources

- [CARRV'21 — NeuralScale: A RISC-V Based Neural Processor Boosting AI Inference in Clouds](https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf)
- [STCP920 product manual, rev 1.12.1](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP920产品手册)
- [STCP950L product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950L产品手册)
- [STCP950P product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950P产品手册)
- [STCP980L product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980L产品手册)
- [STCP980P product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980P产品手册)
- [HPE usage guide (stc-smi / stc-topo / stc-prof output)](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南)
- [C++ API — STCML reference (frequency/voltage domains)](https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API)
- [Model support table (LLM cards / tok-s)](https://docs.streamcomputing.com/AI加速卡/开发者资源/模型支持说明)
- [p2p_perf guide — measured card-to-card bandwidth](https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/p2p_perf使用指南)
- [Firmware Release Notes](https://docs.streamcomputing.com/AI加速卡/固件使用手册/Firmware_Release_Notes)
- [ByteMLPerf / xpu-perf STC backend README (spec table + mass-production statement)](https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md)
- [riscv-stc/riscv-matrix-spec (open matrix extension proposal)](https://github.com/riscv-stc/riscv-matrix-spec)
- [RISC-V International blog — NeuralScale (company-authored)](https://riscv.org/blog/2024/03/neuralscale-industry-leading-general-purpose-programmable-npu-architecture/)
- [Corporate site — timeline, offices, investors](https://www.streamcomputing.com/)
