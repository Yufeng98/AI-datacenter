# d-Matrix Corsair / Raptor — Hardware Architecture

*chip: d-matrix*
*as_of: 2026-08-08*
*generations: Corsair (DIMC, in full production 2026-06-09) / Raptor (3DIMC, early silicon — ISCA 2026, pre-production)*

---

## Generation Overview

| Generation | Status | Logic process | Compute unit | Organization | Performance memory | Capacity memory | Host / D2D | Power |
|---|---|---|---|---|---|---|---|---|
| **Corsair** | **In full production** (2026-06-09); volume shipment to select qualified customers | TSMC **N6**, Alchip design/production partner | DIMC core: 64×64 MAC (INT8) / 64×128 (INT4) inside SRAM bit-cells | 256 cores/chiplet (4 quads × 4 slices × 16 cores); 4 chiplets/chip; 2 chips/card → **2,048 cores/card** | **2 GB SRAM @ 150 TB/s** per card | **256 GB LPDDR5X @ ~400 GB/s** per card | PCIe Gen5 x16; **DMX Link ~1 TB/s** D2D (115 ns) | 275 W @ 800 MHz / 550 W @ 1.2 GHz (Hot Chips 2025) |
| **Raptor** (3DIMC) | **Early silicon / pre-production**; announced 2025-11-18 with Alchip; characterized in ISCA 2026. No launch, sampling date or pricing disclosed | TSMC **N4P** logic die **face-to-face bonded** onto a 3D-DRAM die (36 µm µbump pitch) | Slice: 4×4 tensor-engine array + 1 SIMD core | 4 gangs × 4 slices per chiplet; **4 chiplets @ 1.2 GHz per MCM**; **up to 4 MCMs per card** | **3D-DRAM**: 840 banks/chiplet → **256 independent channels** (16/slice); **measured ~105 TB/s per card @ 700 MHz**, 2.5 ns avg flit latency | **8 × on-package LPDDR5X-9600 = 128 GB per MCM** | **PCIe Gen7** (host and inter-MCM); Gen-2 D2D @ 32 Gbps/lane | **~422 W per MCM**, Tj up to 105 °C |

Raptor is the **commercial debut vehicle for 3DIMC**; **Pavehawk** is the lab-validated test silicon that preceded it. The ISCA 2026 paper reports **on-silicon characterization, not simulation**. Peak TFLOPS/TOPS and numeric-format support for Raptor are **not disclosed**.

---

## Architecture Overview

d-Matrix Corsair uses **Digital In-Memory Compute (DIMC)** — MAC units embedded inside SRAM bit-cells. The weight-stationary design stores model weights in SRAM permanently; activations are the streaming operands. This eliminates the memory-wall bottleneck dominating LLM inference.

Raptor keeps the in-memory-compute thesis but changes the memory that the compute sits in: instead of scaling SRAM, a logic die is bonded face-to-face onto a **3D-DRAM** die, trading SRAM's bandwidth-per-byte for far greater capacity at DRAM density while retaining a very wide channel count (256 independent channels per chiplet). LPDDR5X survives as a secondary capacity tier rather than the primary one.

---

## Hierarchy

```
Corsair PCIe Card
├── Chip A (TSMC 6nm)
│   ├── Chiplet 0  ──┐
│   ├── Chiplet 1    │  DMX Link @ 1 TB/s (all-to-all)
│   ├── Chiplet 2    │
│   └── Chiplet 3  ──┘
│       Each chiplet: 256 DIMC cores, 512 MB SRAM
│       Each chiplet: interfaces to LPDDR5X (64 GB share)
└── Chip B (identical)

Total: 8 chiplets, 2,048 DIMC cores, 2 GB SRAM, 256 GB LPDDR5X

[DMX Bridge Card] → merges 2 Corsair cards:
  16 chiplets, 4,096 DIMC cores, 4 GB SRAM, 512 GB LPDDR5X
```

### Raptor hierarchy (ISCA 2026 early silicon)

```
Raptor Card
└── up to 4 MCMs
    └── MCM = 4 chiplets @ 1.2 GHz   (~422 W/MCM, Tj ≤ 105 °C)
        │   8 × on-package LPDDR5X-9600 → 128 GB per MCM (secondary tier)
        └── Chiplet = TSMC N4P logic die, F2F-bonded (36 µm µbump) to a 3D-DRAM die
            ├── 4 gangs × 4 slices = 16 slices
            │   └── Slice = 4×4 tensor-engine array + 1 SIMD core
            └── 840 3D-DRAM banks → 256 independent channels (16 per slice)

Packaging: 9-4-9 organic substrate onto a 3D CoWoS interposer
Measured: ~105 TB/s 3D-DRAM bandwidth per card @ 700 MHz, 2.5 ns avg flit latency
Host / inter-MCM: PCIe Gen7; Gen-2 D2D @ 32 Gbps/lane
```

### Corsair rack-scale hierarchy (vendor figures, d-matrix.ai/product)

```
Card       → 2,048 DIMC cores, 2 GB SRAM @ 150 TB/s, 256 GB LPDDR5X
Dual card  → 4,096 cores, 4 GB @ 300 TB/s, up to 512 GB capacity, 6,400 mm² silicon
              4,800 TFLOPS MXINT8 dense / 19,200 TFLOPS MXINT4 dense
              512 GB/s card-to-card DMX Bridge
Server     → 8 cards: 19.2 PFLOPS MXINT8 / 76.8 PFLOPS MXINT4,
              16 GB @ 1,200 TB/s, up to 2 TB capacity; PCIe-based scale-up
Rack       → 8 servers / 64 cards: 128 GB @ 9.6 PB/s, up to 16.4 TB capacity
              "Performance Mode" up to 100B params; "Capacity Mode" 1T+ params
SquadRack  → rack blueprint announced 2025-10-14 at OCP Global Summit
              with Arista, Broadcom and Supermicro
```

---

## Compute Engine

### DIMC Core (Corsair)

The fundamental compute unit:
- **64×64 MAC array** embedded in SRAM cells (INT8) or 64×128 (INT4)
- Multiplier at each bit-cell — compute occurs where data is stored
- Supports OCP MX block FP: MXINT4, MXINT8, MXINT16
- 256 cores per chiplet arranged as: 4 quads × 4 slices × 16 cores
- Intra-chiplet: all-to-all interconnect (all 256 cores act as one logical unit)
- Peak: 2,400 TOPS INT8 per card

### Slice / tensor-engine array (Raptor, ISCA 2026)

Raptor replaces the DIMC-core-in-SRAM unit with a slice built around a small dense tensor array next to its own 3D-DRAM channels:
- **Slice = 4×4 tensor-engine array + 1 SIMD core**
- **4 gangs × 4 slices = 16 slices per chiplet**
- Each slice owns **16 of the chiplet's 256 3D-DRAM channels**
- **4 chiplets per MCM at 1.2 GHz**; up to 4 MCMs per card
- Peak TFLOPS/TOPS and supported numeric formats: **not disclosed**
- Paper throughput claim: **4.71× vs HBM** and **2.44× vs SRAM** across Llama-3.1 70B, DeepSeek-V3, Kimi K2, GPT-OSS, Whisper and Canary

---

## Memory Tiers

### Corsair

| Tier | Technology | Capacity/card | Bandwidth | Role |
|------|-----------|--------------|-----------|------|
| Performance Memory | SRAM (in DIMC) | 2 GB | 150 TB/s | Hot weights; active layers |
| Capacity Memory | LPDDR5X | 256 GB | ~400 GB/s | Full model weights; KV-cache |

Aggregated (vendor figures): dual card 4 GB @ 300 TB/s + up to 512 GB capacity; server (8 cards) 16 GB @ 1,200 TB/s + up to 2 TB; rack (64 cards) 128 GB @ 9.6 PB/s + up to 16.4 TB.

### Raptor

| Tier | Technology | Capacity | Bandwidth | Role |
|------|-----------|----------|-----------|------|
| Primary (3DIMC) | **3D-DRAM**, F2F-bonded to the N4P logic die at 36 µm µbump pitch; 840 banks/chiplet → 256 independent channels | **not disclosed** | **~105 TB/s per card, measured @ 700 MHz**, 2.5 ns average flit latency (paper: ~12.5× an HBM3 card) | Weights + activations at DRAM density with SRAM-like channel parallelism |
| Secondary capacity | **LPDDR5X-9600**, 8 on-package devices | **128 GB per MCM** | not disclosed | Overflow capacity |
| On-die SRAM | not disclosed | not disclosed | not disclosed | — |

The ~105 TB/s figure is the headline **measured** number in the ISCA 2026 paper and is the only bandwidth figure in this document taken from real silicon rather than vendor marketing.

---

## Interconnects

| Link | Technology | BW | Purpose | Generation |
|------|-----------|----|---------|---|
| Die-to-die | DMX Link (custom) | ~1 TB/s (115 ns latency) | Chiplet-to-chiplet within card | Corsair |
| Card-to-card | DMX Bridge card | **512 GB/s** | 2→1 logical card (passive PCB) | Corsair |
| Host | PCIe Gen5 x16 | ~64 GB/s | CPU–Corsair | Corsair |
| Scale-out | JetStream 400G NIC | 400 Gbps | Inter-server Ethernet | Corsair |
| Intra-rack | **FabreX** PCIe Gen5 memory fabric / **SuperNODE** (GigaIO assets, acquired 2026-04-02) | sub-200 ns cross-server memory access; SuperNODE supports up to 32 accelerators | Rack-scale Corsair pooling | Corsair |
| Host / inter-MCM | **PCIe Gen7** | not disclosed | CPU–Raptor and MCM-to-MCM | Raptor |
| Die-to-die | Gen-2 D2D | 32 Gbps/lane | Intra-package | Raptor |

> **DMX Link ≠ DMX Bridge.** DMX Link is the ~1 TB/s *die-to-die* chiplet interconnect (confirmed at Hot Chips 2025). The 512 GB/s figure published on the product page is the *card-to-card* DMX Bridge. Both are correct and describe different links.

---

## Compute Performance

Vendor figures unless marked measured. No batch size, accuracy-vs-precision disclosure, or benchmark methodology is published for any of these.

| Model | Config | Throughput | Latency | Generation |
|-------|--------------|-----------|---------|---|
| Llama3 8B | Single server (8 cards) | 60,000 tok/s | 1 ms/tok | Corsair |
| Llama3 70B | Single rack (64 cards) | 30,000 tok/s | 2 ms/tok | Corsair |
| vs. GPU (generic) | — | 10× faster | 10× lower latency | Corsair |
| Llama-3.1 70B, DeepSeek-V3, Kimi K2, GPT-OSS, Whisper, Canary | — | 4.71× vs HBM; 2.44× vs SRAM (paper claim) | — | Raptor |
| 3D-DRAM bandwidth | Per card @ 700 MHz | **~105 TB/s (measured)** | 2.5 ns avg flit latency | Raptor |

**MLPerf:** no d-Matrix/Corsair submission exists in any MLPerf Inference round (confirmed negative as of 2026-08-08).

---

## Process and Packaging

### Corsair
- **Process**: TSMC **N6** (6nm); **Alchip Technologies** is the design/production partner (confirmed 2026-06-09)
- **Package**: Multi-chip module; chiplets on interposer per chip; **organic substrate**
- **Form factor**: PCIe Gen5 full-height half-length card
- **Power**: 275 W @ 800 MHz / 550 W @ 1.2 GHz (Hot Chips 2025); ~38 TOPS/W claimed
- **No CoWoS / HBM interposer**: cost-optimized packaging choice; d-Matrix says it secured multi-year TSMC/Alchip capacity

### Raptor
- **Logic process**: TSMC **N4P**
- **3D integration**: logic die **face-to-face bonded** directly onto a 3D-DRAM die at **36 µm µbump pitch**
- **Package**: **9-4-9 organic substrate onto a 3D CoWoS interposer** — note this generation *does* use CoWoS, unlike Corsair
- **Power / thermals**: **~422 W per MCM**, Tj up to 105 °C
- **Form factor**: card carrying up to 4 MCMs

---

## Key Design Decisions

| Decision | Rationale | Generation |
|----------|-----------|---|
| SRAM not HBM | Eliminates CoWoS cost; 150 TB/s SRAM BW >> HBM | Corsair |
| LPDDR5X for capacity | Cost-effective; 256 GB enables large models | Corsair |
| MXINT4/8/16 only | Block FP reduces data volume; avoids FP16/32 area cost | Corsair |
| PCIe card form factor | Drop-in server deployment; no custom infrastructure | Corsair |
| Inference-only | Simplifies DIMC design; no gradient accumulation | Corsair |
| 3D-DRAM bonded F2F to logic instead of scaling SRAM | Keeps compute adjacent to memory while moving from SRAM density to DRAM density; 256 channels/chiplet preserves the wide-parallel access pattern DIMC depends on | Raptor |
| 840 banks → 256 independent channels, 16 per slice | Each slice gets private memory-level parallelism; avoids a shared-channel bottleneck | Raptor |
| Accepts CoWoS (3D interposer) | The 3D-DRAM stack requires it; the cost argument that ruled out CoWoS for Corsair does not survive the move to 3D | Raptor |
| PCIe Gen7 for host and inter-MCM | Keeps the PCIe-native, no-proprietary-switch deployment model at much higher bandwidth | Raptor |
