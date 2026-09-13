# SK Hynix AiM / AiMX — Layer Table

*chip: sk-hynix-aim*
*device_class: Processing-in-Memory (PIM)*
*as_of: 2026-08-08*
*research_baseline: 2026-04-05 — re-verified 2026-08-08; no in-window change, two pre-baseline backfills applied (vLLM demo, prototype status)*

| Layer | sk-hynix-aim |
|-------|-------------|
| **Software** | |
| Framework Integration | `not public` — AiM SW Stack provides transparent PyTorch/TensorFlow dispatch (GEMV intercepted, routed to AiMX); unmodified user model; described at HC34 + ASPLOS 2023; no public SDK |
| Serving Framework | `demonstrated publicly, not released` — **vLLM** used as the LLM serving frontend in SK Hynix's live AiMX demos (AI Infra Summit 2025, post 2025-10-01; OCP Global Summit 2025, post 2025-10-31): Supermicro SYS-421GE-TNRT3 with 2× NVIDIA H100 + 4× AiMX cards, "stable long token generation using the vLLM framework"; integration code not released, no numeric performance figures disclosed *(added 2026-08-08)* |
| Compiler / IR | `not applicable` — no compiler; AiM Runtime directly issues AiM ISR instruction sequences; extended DRAM commands, not a compiled IR |
| Op Library | `not public` — AiM BLAS Library (BLAS Level 2: GEMV, embedding lookup; analogous to cuBLAS Level 2; operates inside DRAM; not released publicly) |
| Kernel Library | `not applicable` — no user-facing kernel library; MAC units are fixed-function FP16 SIMD; no programmable shader/kernel model |
| Runtime | `not public` — AiM Runtime Library (host-side; AiM ISR dispatch; mode switching: DRAM↔compute; synchronization; memory allocation; FPGA-based reference platform for development) |
| Driver / Firmware | `not public` — PCIe card controller firmware; AiM ISR command encoding; JEDEC GDDR6 mode enable/disable sequence |
| Communication | `not applicable` — AiMX is a PCIe co-processor card; no scale-up/scale-out interconnect; multiple AiMX cards inferred via standard PCIe fabric; no proprietary link disclosed |
| Assembler / ISA | `not public` — AiM ISR (Instruction Set Register): hardware register-mapped command set; extended DRAM commands; not a programmer ISA (no PTX equivalent); partially documented in IEEE ISCA 2021 and HC34 papers |
| **Hardware** | |
| Compute Engine | *GDDR6-AiM: 2 × 16-lane FP16 SIMD MAC units per package (1 per pseudo-channel); 32 FP16 MACs/package; FP16 multiply + accumulate at bank boundary; all-bank simultaneous operation; JEDEC GDDR6 backward compatible. Unchanged since the 2022 HC34 disclosure — **no AiM Gen3, GDDR7-AiM, HBM-based AiM, or AiM-in-HBM4-base-die confirmed as of 2026-08-08*** |
| Data Path | *All-bank operation: all 16 banks activate simultaneously → internal DRAM bandwidth (~4 TB/s) exposed to MAC units; weights resident in DRAM; activations loaded into MAC registers via AiM ISR; no data movement off-chip for GEMV; SIMD lock-step execution; no branch prediction* |
| On-chip Memory | `not applicable` — no separate on-chip SRAM scratchpad; MAC register files (3 per MAC unit; small); data lives in DRAM arrays; internal bank bandwidth is the effective scratchpad analog |
| Off-chip Memory | *GDDR6-AiM DRAM array: 1 GB/package (standard GDDR6); 16 GB (AiMX Gen1: 16 packages) / 32 GB (AiMX Gen2: 32 packages); ~64 GB/s external I/O; internal bandwidth ~4 TB/s (all-bank); 16 Gbps/pin speed grade; 1.25 V* |
| Host Interface / Package | *PCIe (AiMX card); standard GDDR6 BGA package per die; heterogeneous co-processor topology: AiMX + GPU or AiMX + CPU; GDDR6 memory controller on host unchanged; JEDEC compatible. Publicly demonstrated host system: Supermicro GPU SuperServer SYS-421GE-TNRT3, 2× NVIDIA H100 + 4× AiMX cards (2025). **Product status: prototype** — SK Hynix's own CES 2026 wording (2026-01-05); not sampling, not shipping, no named customer* |
| Scale-up Interconnect | `not applicable` — no scale-up interconnect; AiMX is a co-processor card; multiple cards confirmed only as 4 cards over standard PCIe in the 2025 Supermicro demo chassis; no proprietary chip-to-chip link disclosed |
| Scale-out Interconnect | `not applicable` — no scale-out interconnect disclosed; host handles networking |
