# Mythic AMP — Layer Table

*chip: mythic*
*device_class: Analog In-Memory Compute*
*as_of: 2026-08-08*

| Layer | mythic |
|-------|--------|
| **Software** | |
| Framework Integration | *PyTorch / TensorFlow / ONNX (primary) / TensorRT (optional); MAPP SDK imports model graph; no model code changes required for supported ops; ONNX is first-class interchange* |
| Compiler / IR | *Mythic Optimization Suite (INT8/INT4 PTQ; analog range calibration to NOR flash conductance dynamic range) → Mythic Graph Compiler (tile partitioning; weight packing to flash cell addresses; 2D NoC activation routing; per-tile RISC-V codegen; SIMD schedule; equivalence checking; cycle-accurate simulation; not open-source)* |
| Op Library | `not public` — Analog linear ops (GEMM/MVM/Conv2D/attention projections) executed in ACE flash domain; digital nonlinear ops (ReLU/GELU/Softmax/LayerNorm/BatchNorm/pooling/residual) executed by per-tile SIMD engine; compiler-internal dispatch |
| Kernel Library | `not applicable` — No user-programmable kernel API; analog ACE has no programmable ISA; RISC-V tile programs generated entirely by compiler. *2026-08-08: the acquired Videantis v-MP6000UDX is a programmable VLIW+SIMD core with its own pre-acquisition toolchain, so this may change for a future hybrid part; Mythic has published nothing about a merged SDK or kernel API, so the status is unchanged today* |
| Runtime | `not public` — MAPP Runtime: flash programming at model-load time (PCIe DMA → NOR flash cells, one-time per model); activation transfer host↔AMP (PCIe 2.0); tile execution orchestration; ADC result collection; multi-chip dispatch (up to 16 AMPs on PCIe card) |
| Driver / Firmware | `not public` — PCIe BAR management; NOR flash write controller (voltage ramp, verify, retry); temperature-drift calibration firmware for flash conductance; no open-source kernel module |
| Communication | *PCIe 2.0 (chip-to-host, ~4 GB/s); 2D mesh NoC (on-chip tile-to-tile, high-BW); PCIe card fabric (up to 16 chips). ~~PCIe Gen4/5 for multi-card server setups (2026+ roadmap)~~ — removed 2026-08-08: no primary source states a host-interface generation for any forthcoming part; `not disclosed`* |
| Assembler / ISA | `not applicable` — ACE (analog compute) has no user ISA; inputs are voltages, weights are flash conductances, outputs are ADC-converted currents; per-tile RISC-V (RV32I) used internally by compiler only. *2026-08-08 hedge: a v-MP6000UDX-derived digital core would carry its own VLIW+SIMD ISA and toolchain, but no combined Mythic ISA has been published — do not read the acquisition as evidence that a user-visible ISA now exists* |
| **Hardware** | |
| Compute Engine | *Mythic Analog Compute Engine (ACE™): NOR flash MMA array; input activation → voltage via DAC; bitline current sum = dot product (analog MAC); ADC → digital result; 250 GOPS/tile @ 0.25 pJ/MAC; 76 tiles (M1076, 25 TOPS @ 3 W) or 108 tiles (M1108, 35 TOPS @ 4 W); 40nm embedded NOR flash process. 2026 direction (announced, no silicon): hybrid analog CIM + licensed **v-MP6000UDX** VLIW+SIMD digital core acquired with **Videantis GmbH** (2026-05-19, closed) — core count, node, throughput, data types all `not disclosed`* |
| Data Path | *Weight-stationary: INT8 weights programmed permanently into flash cells at model-load time; input activations are streaming operands (DAC-converted voltage); analog current accumulates dot product; ADC recovers digital result; no DRAM BW consumed for weight reads during inference; activation routing via per-tile RISC-V + 2D mesh NoC* |
| On-chip Memory | *64 KB SRAM per tile (activation buffer, scratchpad); flash cells as weight storage (not addressable as memory — compute-in-cell); ~80–100 M weight params per chip* |
| Off-chip Memory | *No external DRAM required for inference (weights in flash); host DRAM used only for model weight source at programming time; no HBM, no GDDR* |
| Host Interface / Package | *PCIe 2.0 x4 (~4 GB/s); 19×19 mm² BGA (M1108); standalone chip / PCIe M.2 A+E-key / PCIe M.2 M-key / PCIe expansion card (up to 16× AMP chips); 40nm eFlash mature node; no CoWoS / interposer* |
| Scale-up Interconnect | *2D mesh NoC (on-chip, all tiles); PCIe card with up to 16 AMP chips sharing PCIe fabric (no dedicated die-to-die link); next-gen / hybrid architecture `not disclosed` as of 2026-08-08 — no NoC, die-to-die, or package-level fabric described for an analog+digital part* |
| Scale-out Interconnect | *Standard host PCIe / Ethernet for server-level; no proprietary scale-out fabric announced for current products; datacenter roadmap remains a stated intent with no product name, node, TOPS, TDP, tapeout, or sampling disclosed as of 2026-08-08* |
