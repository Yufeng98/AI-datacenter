# Groq LPU Layer Mapping Table

*as_of: 2026-08-08*

> **Corporate status note (2026-08-08):** NVIDIA did **not** acquire Groq. The 2025-12-24 transaction was a **non-exclusive inference-technology licensing agreement** plus an acqui-hire; Groq remains independent and GroqCloud operates without interruption. Rows below tagged "NVIDIA Groq 3 LPX / LP30" describe **NVIDIA** silicon built on licensed Groq IP; rows tagged "LPU v1/v2" and "GroqCloud" describe **Groq's** own products, which continue.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (via GroqFlow groqit(); torch-MLIR tracing; offline AOT compilation) | confirmed | software-stack, search-results |
| Framework Integration | ONNX (direct entry; ONNX-MLIR bxing-groq fork → GTen lowering) | confirmed | software-stack, search-results |
| Framework Integration | TensorFlow / Keras (via tf2onnx → ONNX → Groq compiler path) | confirmed | software-stack |
| Framework Integration | CoreML (via coremltools ONNX export → Groq compiler) | confirmed | software-stack |
| Framework Integration | groq-python (OpenAI-compatible REST SDK; pip install groq; GroqCloud inference) | confirmed | search-results |
| Framework Integration | GroqFlow (open-source Apache 2.0; groqit() one-line API; build caching; mlagility benchmarks) | confirmed | software-stack, search-results |
| Compiler / IR | torch-MLIR frontend (PyTorch → standard MLIR IR; upstream LLVM project) | confirmed | software-stack, search-results |
| Compiler / IR | ONNX-MLIR frontend (bxing-groq fork; ONNX → MLIR IR) | confirmed | software-stack, search-results |
| Compiler / IR | Groq GTen Dialect (MLIR dialect for DL/HPC; tens of ops mapping to physical units) | confirmed | software-stack, search-results |
| Compiler / IR | Groq Compiler — spatial placement (op → physical unit in 2D grid) | confirmed | hw-architecture, search-results |
| Compiler / IR | Groq Compiler — cycle-exact scheduling (op → specific clock cycle) | confirmed | hw-architecture, search-results |
| Compiler / IR | Groq Compiler — SRAM address assignment (weight tiles → SRAM banks) | confirmed | hw-architecture, search-results |
| Compiler / IR | Groq Compiler — inter-chip routing (multi-chip packet schedule; plesiosynchronous) | confirmed | hw-architecture, search-results |
| Compiler / IR | .iop binary (Instruction Operation Program; proprietary; cycle-exact schedule + weights + routing) | confirmed | software-stack |
| Compiler / IR | GroqWare groq-devtools (compiler package; not open-source) | confirmed | search-results |
| Compiler / IR | Bare-metal assembler (research/low-level; direct ISA programming) | inferred | search-results |
| Op Library | `not applicable` — no separate op dispatch library; all fusion inside Groq Compiler GTen pass pipeline | confirmed | software-stack |
| Kernel Library | `not applicable` — no user-facing kernel library; no CUTLASS equivalent; compiler owns all placement | confirmed | software-stack |
| Runtime | groq-runtime (thin static executor; loads .iop, weight-loads SRAM, replays instruction stream) | confirmed | software-stack, search-results |
| Runtime | GroqWare bundle (groq-devtools compiler + groq-runtime; not separately open-sourced) | confirmed | search-results |
| Runtime | GroqCloud REST API (HTTPS; streaming; OpenAI-compatible endpoint; 750-900 tok/s Llama 3.3 70B) | confirmed | search-results |
| Runtime | Speculative Decoding (GroqCloud feature; >1660 tokens/sec on Llama 3 70B; late 2024) | confirmed | search-results |
| Driver / Firmware | PCIe Linux kernel module (BAR mapping, command queues, IRQ; not open-sourced) | confirmed | software-stack |
| Driver / Firmware | No on-chip firmware RM (unlike NVIDIA GSP; LPU executes pre-compiled .iop stream directly) | confirmed | software-stack, hw-architecture |
| Communication | Compiler-scheduled plesiosynchronous inter-chip (no NCCL; packets scheduled at compile time) | confirmed | hw-architecture, search-results |
| Communication | GroqCloud HTTPS REST (client-to-cloud; standard internet) | confirmed | search-results |
| Assembler / ISA | No user-visible virtual ISA (no PTX equivalent; .iop format proprietary) | confirmed | software-stack |
| Assembler / ISA | VLIW-like instruction format (described in ISCA 2020 at high level; exact encoding not public) | inferred | search-results |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | 320x320 fused dot product matrix unit (GEMM; 750 INT8 TOPS, 188 FP16 TFLOPS @ 900 MHz, LPU v1) | confirmed | hw-architecture, search-results |
| Compute Engine | NVIDIA Groq 3 LPX LP30: **1.2 PFLOPS FP8 per chip / 9.6 PFLOPS FP8 per tray**; non-FP8 datatype throughput **not disclosed**; unit organization not re-disclosed by NVIDIA | confirmed | hw-architecture (2026-08-08), NVIDIA developer blog, StorageReview |
| Compute Engine | NVIDIA-quoted 315 PFLOPS per LPX rack does not reconcile with 9.6 × 32 = 307.2 PFLOPS — unexplained; do not present as a dense FP8 figure | flagged | hw-architecture (2026-08-08) |
| Compute Engine | 5,120 Vector ALUs (activations: GeLU/SiLU/ReLU; layer norm; element-wise) | confirmed | hw-architecture, search-results |
| Compute Engine | No warp scheduler, no branch predictor, no out-of-order buffer, no cache hierarchy | confirmed | hw-architecture, search-results |
| Data Path | Tensor streaming — conveyor-belt pipeline; activations flow through 2D spatial grid | confirmed | hw-architecture, search-results |
| Data Path | Static spatial scheduling — compiler assigns all data movement to exact clock cycles | confirmed | hw-architecture, search-results |
| Data Path | No hardware cache misses (SRAM is primary storage, not cache; compiler manages all data placement) | confirmed | hw-architecture |
| On-chip Memory | ~230 MB on-chip SRAM per die (LPU v1); weight storage + activation buffer (NOT cache) | confirmed | hw-architecture, search-results |
| On-chip Memory | 80 TB/s internal SRAM bandwidth (~24x H100 HBM3 3.35 TB/s) | confirmed | hw-architecture, search-results |
| On-chip Memory | **500 MB** SRAM per die (NVIDIA Groq 3 LPX LP30, announced 2026-03-16) — *corrected 2026-08-08; the previous "512 MB" was wrong* | confirmed | hw-architecture (2026-08-08), NVIDIA LPX product page |
| On-chip Memory | 150 TB/s SRAM bandwidth (NVIDIA Groq 3 LPX LP30) | confirmed | search-results, NVIDIA developer blog |
| On-chip Memory | LPX rack: 128 GB aggregate SRAM (256 × 500 MB) at 40 PB/s | confirmed | hw-architecture (2026-08-08), NVIDIA LPX product page |
| Off-chip Memory | No off-chip DRAM on a single LPU die (all generations, incl. LP30) | confirmed | hw-architecture, search-results |
| Off-chip Memory | GroqRack distributed: 576 x 230 MB = ~130 GB total SRAM | confirmed | hw-architecture, search-results |
| Off-chip Memory | **12 TB DDR5 per LPX rack** — rack-level DRAM tier; breaks the "no DRAM anywhere" absolute at system level. Bandwidth and attachment point **not disclosed** | confirmed | hw-architecture (2026-08-08), NVIDIA LPX product page |
| Host Interface / Package | Samsung 14nm (LPU v1), 25x29 mm (~725 mm²), 900 MHz | confirmed | hw-architecture, search-results |
| Host Interface / Package | Samsung 4nm (LPU v2, 2025) | confirmed | search-results |
| Host Interface / Package | NVIDIA Groq 3 LPX LP30: process node **not disclosed**; TDP **not disclosed**; die size/clock **not disclosed** — *corrected 2026-08-08; the prior "Samsung 4nm" attribution is unconfirmed (secondary blogs only)* | not disclosed | hw-architecture (2026-08-08), NVIDIA developer blog, StorageReview |
| Host Interface / Package | LPX tray: 1U liquid-cooled, 8 LPUs (departure from air-cooled GroqRack) | confirmed | hw-architecture (2026-08-08), NVIDIA developer blog |
| Host Interface / Package | PCIe host interface | confirmed | software-stack |
| Scale-up Interconnect | Plesiosynchronous chip-to-chip protocol (near-synchronous; software drift correction) | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | Direct mesh / dragonfly topology (no external switches; compiler-scheduled routing) | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | GroqRack: 576 chips, ~130 GB SRAM | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | GroqRack "~640 TB/s aggregate chip-to-chip bandwidth" — **unverified**; no Groq primary source found; likely conflated with NVIDIA's 640 TB/s LPX rack scale-up figure | unverified | hw-architecture (2026-08-08) |
| Scale-up Interconnect | 576 chips appear as single coherent memory to software | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | NVIDIA Groq 3 LPX: 256 LPUs per rack in 32 liquid-cooled 1U trays of 8; 40 PB/s aggregate SRAM BW; **640 TB/s rack scale-up BW** | confirmed | hw-architecture (2026-08-08), NVIDIA LPX product page |
| Scale-up Interconnect | LP30 chip-to-chip: **96 links @ 112 Gbps = 2.5 TB/s aggregate bidirectional per chip** | confirmed | hw-architecture (2026-08-08), NVIDIA developer blog |
| Scale-up Interconnect | Whether LPX retains the plesiosynchronous protocol and compiler-scheduled packet delivery is **not disclosed** by NVIDIA | not disclosed | hw-architecture (2026-08-08) |
| Scale-out Interconnect | Standard Ethernet/InfiniBand via host NICs (no proprietary scale-out fabric) | confirmed | search-results |
| Scale-out Interconnect | Groq 3 LPX: integrated with NVIDIA Vera Rubin platform as **latency-sensitive decode co-processor** — Rubin GPUs take throughput-bound full-context attention over the KV cache; LPX takes latency-sensitive decode work such as sparse MoE expert FFNs (*corrected 2026-08-08; prior "prefill vs token generation" split was wrong*) | confirmed | hw-architecture (2026-08-08), NVIDIA developer blog |
| Scale-out Interconnect | Groq 3 LPX availability: **ANNOUNCED** GTC 2026-03-16; H2 2026 Vera Rubin partner availability guidance; no LPX-specific ship date (*prior "Q3 2026" was an analyst inference*) | confirmed | hw-architecture (2026-08-08), nvidianews Vera Rubin release |
