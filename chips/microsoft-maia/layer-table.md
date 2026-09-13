# Microsoft Maia Layer Mapping Table

*as_of: 2026-08-08*
*Device class: Cloud AI Accelerator*
*Generations covered: Maia 100 (TSMC 5nm, 2024) and Maia 200 (TSMC 3nm, 2026)*
*Review note (2026-08-08): no layer changed technically in the 2026-04-05 → 2026-08-08 window. Framework-integration and SDK rows updated for the new first-party MAI workload class and re-confirmed preview status; all hardware rows unchanged.*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch backend (first-class; eager mode + graph mode) | confirmed | Azure Blog; Maia 100 Tech Community blog |
| Framework Integration | Azure OpenAI Services integration (Copilot, Azure AI Studio) | confirmed | Azure Blog; Microsoft announcements |
| Framework Integration | Microsoft first-party MAI models on Maia 200 (MAI-Thinking-1 et al.); Microsoft's first public model↔silicon co-design claim on Maia | confirmed (workload); co-design *mechanism* not disclosed | Build 2026 MAI keynote transcript, 2026-06-02 |
| Framework Integration | Maia SDK access: preview only, sign-up-gated (academics, developers, frontier labs, OSS contributors); no GA, no GA timeline, no public docs tree (learn.microsoft.com/azure/maia → 404) | confirmed (re-checked 2026-08-08) | Maia SDK sign-up page; negative check on learn.microsoft.com |
| Compiler / IR | Triton compiler for Maia (OpenAI Triton Python DSL; portability path; GPU + Maia cross-target) | confirmed | Azure Blog; Maia SDK overview |
| Compiler / IR | NPL compiler — Nested Parallel Language (Maia-native; explicit SRAM/DMA control; not public) | confirmed (exists), not public | Azure Blog; Maia SDK overview |
| Op Library | Microsoft Maia kernel library (GEMM, Attention, Norm, collectives; compiler-built; not public) | confirmed (exists), not public | Azure Blog (inferred from SDK description) |
| Kernel Library | Custom NPL and Triton kernels (production kernels for Azure OpenAI; not public) | confirmed (exists), not public | Azure Blog |
| Runtime | Maia Runtime (kernel dispatch, SRAM alloc, DMA queues, semaphores; SDK preview only) | confirmed (exists), not public | Azure Blog; Maia SDK |
| Driver / Firmware | Maia kernel driver (PCIe BAR, cmd submission, IRQ; Azure-internal; not upstreamed) | confirmed (exists), not public | inferred from HC2024 + SDK |
| Driver / Firmware | Maia firmware (Azure-managed; firmware-upgradeable; "unlocks new capabilities") | confirmed | TechRadar report |
| Communication | Custom RoCE-like transport (Maia 100): 4800 Gbps AllGather/SR, 1200 Gbps A2A, AES-GCM | confirmed | HC2024; Azure Blog; Tech Community |
| Communication | On-die NIC (Maia 200): 2.8 TB/s bidir; 2-tier topology; 6,144-chip clusters | confirmed | Maia 200 deep-dive blog |
| Communication | MRC (Multipath Reliable Connection) OCP transport — Microsoft is a co-developer (arXiv:2606.18170, 2026-06-16) | **adjacent ecosystem context, NOT a Maia feature** — the paper does not mention Maia; claims that Maia implements MRC are unsourced | arXiv:2606.18170; OCP release ~2026-05-05 |
| Assembler / ISA | Custom vector processor ISA (FP32/BF16; custom; not public) | confirmed (exists), not public | HC2024 |
| Assembler / ISA | Tensor unit ISA (MX6/MX9/BF16 on Maia 100; FP4/FP8 on Maia 200; not public) | confirmed (exists), not public | HC2024; Maia 200 blog |
| Assembler / ISA | OCP MX (Microscaling) data format spec (public OCP standard; Maia ISA impl not public) | confirmed (spec public) | opencomputeproject.org; Microsoft |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Maia 100: Tensor Unit 16×R×16 (MX6 3 POPS / MX9 1.5 POPS / BF16 0.8 POPS) | confirmed | HC2024 |
| Compute Engine | Maia 100: Vector Processor — loosely coupled superscalar, custom ISA, FP32/BF16 | confirmed | HC2024; Tech Community |
| Compute Engine | Maia 100: ~105B transistors, TSMC N5, ~820mm², 500–700W TDP | confirmed | HC2024; TechSpot |
| Compute Engine | Maia 200: Tile Tensor Unit (TTU) — FP4/FP8 native; >10 PFLOPS FP4, >5 PFLOPS FP8 | confirmed | Maia 200 blog; deep dive |
| Compute Engine | Maia 200: Tile Vector Processor (TVP) — programmable SIMD engine | confirmed | Maia 200 deep-dive blog |
| Compute Engine | Maia 200: Tile → Cluster hierarchy (TTU+TVP+TSRAM+DMA per tile; tiles share CSRAM) | confirmed | Maia 200 deep-dive blog |
| Compute Engine | Maia 200: >140B transistors, TSMC N3, 750W TDP | confirmed | Maia 200 announcement |
| Data Path | Maia 100: Compiler-scheduled DMA (NPL/Triton explicit tensor tiling + SRAM placement) | confirmed | HC2024; Azure Blog |
| Data Path | Maia 100: Hardware semaphores for async double-buffering (compute + DMA overlap) | confirmed | HC2024; Tech Community |
| Data Path | Maia 200: Hierarchical DMA (tile DMA: TSRAM↔CSRAM; cluster DMA: CSRAM↔HBM) | confirmed | Maia 200 deep-dive blog |
| Data Path | Maia 200: Hierarchical NoC (tile ↔ cluster ↔ chip interconnect) | confirmed | Maia 200 deep-dive blog |
| On-chip Memory | Maia 100: ~500 MB on-chip SRAM (L1+L2 software-managed scratchpads; no hardware cache) | confirmed | HC2024; Tech Community |
| On-chip Memory | Maia 200: 272 MB total — TSRAM (per-tile hot buffer) + CSRAM (per-cluster KV-cache) | confirmed | Maia 200 deep-dive blog |
| On-chip Memory | TSRAM bandwidth ~10–20× HBM3e for hot data paths | confirmed | Maia 200 deep-dive blog |
| Off-chip Memory | Maia 100: 4× HBM2e, 64 GB, 1.8 TB/s (CoWoS-S package; deliberate HBM2e choice) | confirmed | HC2024; TechRadar |
| Off-chip Memory | Maia 200: HBM3e, 216 GB, 7 TB/s (SK Hynix reported sole supplier) | confirmed | Maia 200 announcement; TrendForce |
| Host Interface / Package | Maia 100: PCIe Gen5 ×8, 32 GB/s host bandwidth | confirmed | HC2024 |
| Host Interface / Package | Maia 100: CoWoS-S 2.5D interposer package (SoC + 4× HBM2e stacks) | confirmed | HC2024 |
| Host Interface / Package | Ares rack: 32 chips/8 servers, 40 kW, mandatory liquid cooling, custom wider form factor | confirmed | ServeTheHome; Tech Community |
| Host Interface / Package | Ares rack: Arista + Cisco dual-sourced ToR switches (3 switch SKUs per rack) | confirmed | ServeTheHome |
| Scale-up Interconnect | Maia 100: 12× 400 GbE → 600 GB/s backend; AllGather/SR 4800 Gbps; A2A 1200 Gbps | confirmed | HC2024; Azure Blog |
| Scale-up Interconnect | Maia 100: Custom RoCE-like protocol (enhanced reliability + load balancing) | confirmed | Azure Blog; Tech Community |
| Scale-up Interconnect | Maia 100: AES-GCM encryption in transport (confidential compute) | confirmed | Azure Blog |
| Scale-up Interconnect | Maia 100: Unified scale-up + scale-out (single fabric) | confirmed | Azure Blog |
| Scale-up Interconnect | Maia 200: Integrated on-die NIC, 2.8 TB/s bidir, 2-tier topology, 6,144 chips | confirmed | Maia 200 deep-dive blog |
| Scale-out Interconnect | Maia 100: Unified Ethernet fabric; Arista/Cisco switches for inter-rack/inter-pod | confirmed | ServeTheHome; Azure Blog |
| Scale-out Interconnect | Maia 200: On-die NIC handles both scale-up and scale-out; no separate NIC disclosed | inferred | Maia 200 deep-dive blog |
