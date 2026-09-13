# Tesla Dojo Layer Mapping Table

*as_of: 2026-08-08*
*Note: Tesla disbanded the Dojo team in August 2025. A **Dojo 3** program was publicly confirmed as resumed in January 2026 and restated by Musk on 2026-04-15, but **no Dojo 3 hardware or software has been disclosed** — every Dojo 3 row below is literally "not disclosed". D1/Dojo v1 remains the documented architecture and all confirmed rows below refer to it.*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (standard model interface, no manual kernels required) | confirmed | HC34 presentation |
| Compiler / IR | Tesla ML Compiler (custom; graph capture from PyTorch) | confirmed | HC34 presentation |
| Compiler / IR | LLVM backend (targets Dojo custom ISA) | confirmed | HC34 presentation, job postings |
| Compiler / IR | Automatic data/model/graph parallelism | confirmed | HC34 presentation |
| Op Library | Internal op library (GEMM, Conv, Attention — compiler-generated) | not public | inferred from HC34 |
| Kernel Library | not public | not public | — |
| Runtime | Dojo Runtime (fault-tolerant mesh routing, SRAM allocation) | not public | confirmed existence via HC34 |
| Runtime | Auto-scaling across Training Tiles | confirmed | HC34 presentation |
| Driver / Firmware | Dojo System Driver (mesh topology, DIP card control) | not public | inferred |
| Communication | TTPoE (Tesla Transport Protocol over Ethernet) | confirmed | HC2024 presentation; open-sourced |
| Communication | TTP data ingestion (video frames from host servers, ~160 GB/s) | confirmed | HC34 presentation |
| Assembler / ISA | Custom Dojo ISA (64-bit scalar + 64-byte SIMD vector) | confirmed | HC34 presentation |
| Assembler / ISA | BF16 / CFP8 / CFloat16 / FP32 precision support | confirmed | HC34 presentation |
| Assembler / ISA | ISA specification | not public | — |
| Assembler / ISA | Assembler / disassembler tools | not public | — |
| *All software layers (Dojo 3)* | not disclosed — no SDK, compiler, ISA, runtime, or framework-integration statement exists for Dojo 3; unknown whether the D1-era PyTorch + LLVM stack carries forward | not disclosed | Musk X post 2026-04-15; TechCrunch 2026-01-20 (program only, no stack detail) |
| Communication | TTPoE — **no new release** since the Hot Chips 36 open-sourcing; not stated to be Dojo 3's fabric | unchanged | github.com/teslamotors/ttpoe (checked 2026-08-08) |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Training Node: 64-byte SIMD vector + 64-bit scalar (×354 per D1) | confirmed | HC34 |
| Compute Engine | D1 chip: 376 TFLOPS BF16/CFP8, 22 TFLOPS FP32 | confirmed | HC34 |
| Compute Engine | D1 chip: TSMC 7nm, 645mm², 50B transistors, 400W TDP, 2 GHz | confirmed | HC34 |
| Data Path | On-chip 2D mesh: ~10 TB/s directional, software-scheduled | confirmed | HC34 |
| Data Path | Chip-to-chip SerDes: 576 lanes at 112 GT/s, ~8 TB/s aggregate | confirmed | HC34 |
| Data Path | Direct chip butt-join (no switching silicon on tile) | confirmed | HC34 |
| On-chip Memory | Distributed SRAM: 440 MB per D1, 1.25 MB per node | confirmed | HC34 |
| On-chip Memory | Load: ~400 GB/s per node; Store: ~270 GB/s per node | confirmed | HC34 |
| On-chip Memory | Software-managed scratchpad (no cache hierarchy, no virtual memory) | confirmed | HC34 |
| Off-chip Memory | HBM via DIP (Dojo Interface Processor) cards: ~13 TB/s per tile | confirmed | HC34 |
| Off-chip Memory | ExaPOD HBM total: ~13 TB | confirmed | HC34 |
| Off-chip Memory | HBM not on D1 die; separate daughter cards at tile edge | confirmed | HC34 |
| Host Interface / Package | Training Tile PCB: 25 D1 chips (5×5), ~15 kW, substrate is interconnect | confirmed | HC34 |
| Scale-up Interconnect | Training Tile mesh: 25 D1 direct SerDes, 36 TB/s aggregate | confirmed | HC34 |
| Scale-up Interconnect | ExaPOD hierarchy: Tile → Tray (6T) → Cabinet (2 trays, 300 D1) → ExaPOD (10 cab, 3000 D1) | confirmed | HC34 |
| Scale-up Interconnect | ExaPOD: 1 EFLOPS BF16, 1.3 TB SRAM, 13 TB HBM | confirmed | HC34 |
| Scale-out Interconnect | TTPoE: lossy Ethernet, commodity switches, hardware Dumb-NIC | confirmed | HC2024; open-sourced |
| Scale-out Interconnect | TTPoE: microsecond latency, hardware retry, no PFC required | confirmed | HC2024 |
| Compute Engine | **Dojo 3**: compute unit type, count, clock, precisions, throughput | not disclosed | no primary source exists (checked 2026-08-08) |
| Data Path | **Dojo 3**: on-chip fabric and chip-to-chip interface | not disclosed | — |
| On-chip Memory | **Dojo 3**: SRAM capacity and memory model (no basis for assuming the D1 no-cache model carries over) | not disclosed | — |
| Off-chip Memory | **Dojo 3**: memory technology, capacity, bandwidth, attach method | not disclosed | — |
| Host Interface / Package | **Dojo 3**: process node and packaging (D1 was TSMC 7nm; AI5/AI6 are Samsung; Terafab targets Intel 14A — none tied to Dojo 3) | not disclosed | SpaceX Form S-1 2026-05-20; Reuters/Tom's Hardware 2026-04-22 |
| Scale-up Interconnect | **Dojo 3**: board/rack topology. The July 2025 "5, 12 on a board" remark is stated intent, predates the Jan 2026 space-compute repositioning, and is not a specification | not disclosed | Tesla Q2 2025 earnings call, 2025-07-23 |
| Scale-out Interconnect | **Dojo 3**: cluster fabric; no statement that TTPoE is reused | not disclosed | — |
