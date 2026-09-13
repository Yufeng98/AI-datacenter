# OpenAI-Broadcom Jalapeño Layer Mapping Table

*as_of: 2026-08-08*
*confidence: low (silicon at engineering-sample stage; SDK not public; very limited public disclosure)*

Chip unveiled as **Jalapeño** on 2026-06-24. The prior codename "Project Titan" has no primary source
and is retained below only where a row records that fact.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (primary; OpenAI's entire ML infrastructure; custom torch.compile backend inferred) | inferred | hw-architecture, sw-stack |
| Framework Integration | JAX, TensorFlow, ONNX — not confirmed; likely unsupported in initial release | not-confirmed | sw-stack |
| Compiler / IR | Triton compiler — OpenAI extending Triton with a Jalapeño ASIC backend (confirmed in Q1 2026 job postings) | confirmed-in-principle | sw-stack, job-postings |
| Compiler / IR | TorchDynamo + TorchInductor (torch.compile pathway → custom Triton backend) | inferred | sw-stack |
| Compiler / IR | MLIR-based IR pipeline (Triton uses MLIR internally; Jalapeño backend emits Jalapeño ISA via LLVM) | inferred | sw-stack |
| Compiler / IR | Jalapeño ISA / codegen backend — not public | not-public | sw-stack |
| Compiler / IR | Toolchain proven end-to-end on silicon: engineering samples ran GPT-5.3-Codex-Spark in the lab (2026-06-24). No detail of the compilation path disclosed | confirmed (milestone only) | jalapeno-unveiling-2026-06-24 |
| Compiler / IR | OpenAI states it used its own models to accelerate parts of chip design and optimization | vendor-claim | jalapeno-unveiling-2026-06-24 |
| Op Library | `not public` — no publicly released op library; GEMM native to the matrix engine; higher-level ops via Triton kernels | not-public | sw-stack |
| Kernel Library | `not public` — no public kernel library; kernels written internally by OpenAI's Compiler/Kernels team using Triton + Jalapeño backend | not-public | sw-stack, job-postings |
| Runtime | `not public` — internal OpenAI inference runtime; handles weight loading, batching, token streaming on Jalapeño hardware | not-public | sw-stack |
| Runtime | `not public` — runtime is functional enough to execute a production-class OpenAI model (GPT-5.3-Codex-Spark) on engineering samples at production target frequency/power | confirmed (milestone only) | jalapeno-unveiling-2026-06-24 |
| Runtime | `not public` — continuous batching, dynamic batching strategy for LLM inference: inferred but not confirmed | inferred | sw-stack |
| Runtime | `not public` — quantization runtime (FP8/INT8): inferred from the architecture class; not confirmed | inferred | sw-stack |
| Driver / Firmware | `not public` — PCIe kernel-mode driver for Jalapeño (device enumeration, BAR mapping, DMA) | inferred | hw-architecture |
| Driver / Firmware | `not public` — on-chip firmware / management controller: architecture unknown | not-public | hw-architecture |
| Communication | Broadcom all-Ethernet collective library (AllReduce, AllGather over Ethernet; NCCL-equivalent; not named publicly) | inferred | hw-architecture |
| Communication | Broadcom Jericho4 AI switch — in-network computing for collective ops. **Not named in the 2026-06-24 announcement**; attribution gains no support | inferred | hw-architecture |
| Communication | RoCE v2 transport (inferred from Broadcom Ethernet stack) | inferred | hw-architecture |
| Assembler / ISA | `not public` — no user-facing ISA; the matrix-engine ISA is internal; no PTX/HIP equivalent accessible | not-public | hw-architecture |
| SDK | No public SDK, no developer documentation, no external developer access | not-public | sw-stack |
| Model scope | Designed to "work with all LLMs" / "current and future LLMs across the industry" — architecture flexibility statement. **No commitment to sell the chip to third parties has been made** | confirmed (architecture claim only) | jalapeno-unveiling-2026-06-24 |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Product Identity | **Jalapeño** — OpenAI's "first Intelligence Processor"; unveiled 2026-06-24 by OpenAI + Broadcom | confirmed | jalapeno-unveiling-2026-06-24 |
| Product Identity | "Project Titan" — prior repo codename; appears only on aggregator/SEO sites, in no primary source | unsourced | (repo defect corrected 2026-08-08) |
| Silicon Status | Engineering samples running ML workloads in the lab at production target frequency and power, including GPT-5.3-Codex-Spark. Sampling — NOT mass production, NOT deployed at scale | confirmed | jalapeno-unveiling-2026-06-24 |
| Silicon Status | Design → manufacturing tape-out in nine months; claimed "fastest ASIC development cycle ever achieved in high-performance advanced semiconductors" | vendor-claim | jalapeno-unveiling-2026-06-24 |
| Compute Engine | Tiled / systolic matrix engine — reported by industry sources; wafer floorplan described as "very regular, repeated, columnar", read as consistent with this class | reported-not-confirmed | hw-architecture, toms-hardware-die-analysis |
| Compute Engine | Matrix multiply (GEMM) — primary operation; dimensions, tile size not public | reported | hw-architecture |
| Compute Engine | Peak TFLOPS — not public (any precision) | not-public | hw-architecture |
| Compute Engine | Data types — not public (BF16/FP8 inferred from the architecture class + inference target) | inferred | hw-architecture |
| Compute Engine | Sparsity support — not public | not-public | hw-architecture |
| Compute Engine | Attention hardware acceleration — not confirmed | not-confirmed | hw-architecture |
| Performance | No quantitative performance data disclosed. Only "performance per watt substantially better than current state-of-the-art" and utilization "much closer to theoretical peak"; technical report promised "in the coming months" | vendor-claim (unquantified) | jalapeno-unveiling-2026-06-24 |
| Performance | No MLPerf submission and no third-party benchmark exists. A "50% cheaper than Nvidia" figure on aggregator sites appears in no primary source — excluded | not-public | (aggregator noise excluded) |
| Data Path | On-chip SRAM (weight-stationary accumulator buffers + activation SRAM + scratchpad) — capacity not public | inferred | hw-architecture |
| Data Path | On-chip SRAM bandwidth — not public | not-public | hw-architecture |
| Off-chip Memory | HBM generation **not disclosed** — neither OpenAI nor Broadcom stated HBM3E vs HBM4 at unveiling | not-public | jalapeno-unveiling-2026-06-24 |
| Off-chip Memory | Samsung HBM4 exclusive supply — reported early 2026 on aggregator sourcing; neither confirmed nor contradicted by the June 2026 disclosure (downgraded from "confirmed" on 2026-08-08) | reported-not-confirmed | search-results |
| Off-chip Memory | Six HBM stacks surrounding the compute chiplet | third-party-estimate (medium) | toms-hardware-die-analysis |
| Off-chip Memory | HBM capacity per chip — not public | not-public | hw-architecture |
| Off-chip Memory | HBM bandwidth per chip — not public | not-public | hw-architecture |
| Host Interface / Package | Compute chiplet ~25.46 mm × 33 mm ≈ 840 mm² (EUV reticle limit ≈ 858 mm²) — journalist measurement of released package/wafer photos, not a vendor spec | third-party-estimate (medium) | toms-hardware-die-analysis |
| Host Interface / Package | Separate I/O chiplet on package + two structural dummy dies — multi-die package, not monolithic | third-party-estimate (medium) | toms-hardware-die-analysis |
| Host Interface / Package | Process node — **not disclosed** at unveiling; 3nm-class (TSMC N3) per prior 10 GW-program reporting (downgraded from stated fact on 2026-08-08) | reported-not-confirmed | hw-architecture, search-results |
| Host Interface / Package | Transistor count — not public | not-public | hw-architecture |
| Host Interface / Package | TDP / power — not public ("production target frequency and power" stated qualitatively only) | not-public | hw-architecture |
| Host Interface / Package | Packaging technology — not public (2.5D advanced packaging inferred from chiplet + 6-stack HBM layout) | inferred | hw-architecture |
| Host Interface / Package | Host PCIe interface — not public (PCIe Gen5/6 inferred from the 2026 timeline) | inferred | hw-architecture |
| Host Interface / Package | Host CPU — not disclosed. The Information reported Arm is designing a server-class CPU to anchor OpenAI's next-gen racks; unnamed sources, never confirmed by Arm, OpenAI or Broadcom | rumor (low) | the-information-arm-cpu |
| Scale-up Interconnect | All-Ethernet (Broadcom stack) — confirmed in partnership announcement | confirmed | search-results |
| Scale-up Interconnect | Broadcom Jericho4 switching chip — inferred from Broadcom AI portfolio; **not named in the 2026-06-24 announcement** | inferred | hw-architecture |
| Scale-up Interconnect | Scale-up bandwidth per chip — not public | not-public | hw-architecture |
| Scale-up Interconnect | Scale-up topology (fat-tree, rail-optimized, dragonfly) — not public | not-public | hw-architecture |
| Scale-out Interconnect | Broadcom **Tomahawk** networking silicon — explicitly named by OpenAI and Broadcom (upgraded from "inferred" on 2026-08-08) | confirmed | jalapeno-unveiling-2026-06-24 |
| Scale-out Interconnect | Port speed — not public (400 GbE / 800 GbE inferred from the deployment window) | inferred | hw-architecture |
| Scale-out Interconnect | Optical connectivity — included (Broadcom optical solutions per announcement) | confirmed | search-results |
| Rack System | Celestica — board, rack system integration, high-performance networking, scalable production systems | confirmed | jalapeno-unveiling-2026-06-24 |
| Rack System | 10 GW total deployment across OpenAI facilities + partner data centers | confirmed | search-results |
| Rack System | Initial deployment targeted by end of 2026, "expanding in the years ahead" — a stated target, not an achieved milestone | confirmed (as target) | jalapeno-unveiling-2026-06-24 |
| Rack System | Microsoft named as a gigawatt-scale deployment partner beginning 2026 (Hock Tan, Broadcom CEO) | confirmed (vendor statement) | jalapeno-unveiling-2026-06-24 |
| Rack System | Chips per rack — not public | not-public | hw-architecture |
| Rack System | Power per rack — not public | not-public | hw-architecture |
| Gen 2 Roadmap | TSMC A16 (1.6nm nanosheet + Super Power Rail) — reported TrendForce Jan 2026; the 2nm/A16 element is not corroborated in primary sources | unconfirmed-reporting | search-results |
| Gen 2 Roadmap | Gen 2 development start H2 2026; deployment 2027+ | unconfirmed-reporting | search-results |
| Multi-gen Roadmap | "This is just the beginning of a multi-generation roadmap" (Hock Tan) — no generations, nodes or dates given | confirmed (vendor statement, no specifics) | jalapeno-unveiling-2026-06-24 |
| Scheduled Disclosure | Ho / Narayanaswami / Leary (OpenAI), "You Can Just Build Things … Chips", AI 2 session, Hot Chips 38, Tue 2026-08-25 4:45–6:15 PM PDT — *disclosure scheduled; content not yet public* | scheduled (no content) | hotchips-38-advance-program |
| Scheduled Disclosure | Broadcom, "Thor Ultra: An Ethernet NIC Chip Optimized for AI & HPC" (Hemal Shah), Hot Chips 38, 2026-08-25 — *disclosure scheduled; content not yet public*; no stated link to Jalapeño | scheduled (no content) | hotchips-38-advance-program |
