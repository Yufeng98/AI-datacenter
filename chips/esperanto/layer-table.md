# Esperanto ET-SoC-1 Layer Mapping Table

*as_of: 2026-08-08*
*Rows without a generation marker describe the shipped ET-SoC-1 (TSMC 7nm, 2021–2025). Rows marked **[ETSP 2026]** describe the successor CORE-ET Silicon Platform (16nm, tapeout in progress) published by Ainekko under the OpenHW Foundation.*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (via torch.onnx.export() → ONNX; no native eager backend; all compilation is AOT) | confirmed | software-stack, search-results |
| Framework Integration | TensorFlow / Keras (via tf2onnx → ONNX; no native TF integration) | confirmed | software-stack |
| Framework Integration | ONNX (first-class primary format; native input to Glow ML compiler and ONNXRuntime EP) | confirmed | software-stack, search-results |
| Framework Integration | ONNXRuntime Custom Execution Provider (ET-SoC-1 EP; standard C/C++/Python ONNX API; transparent hardware dispatch) | confirmed | software-stack, search-results |
| Compiler / IR | Meta Glow open-source ML compiler (github.com/pytorch/glow; ONNX → graph IR → RISC-V; graph optimization, op fusion, quantization) | confirmed | software-stack, search-results |
| Compiler / IR | LLVM RISC-V backend (Glow → LLVM IR → RISC-V; extended with ET-Minion vector/tensor intrinsics) | confirmed | software-stack |
| Compiler / IR | Per-core RISC-V ELF binary generation (1,088 binaries; one per ET-Minion partition; weight data interleaved for SRAM locality) | confirmed | software-stack, hw-architecture |
| Compiler / IR | Primo AI/ML SDK (umbrella Python SDK; model compile, profile, deploy; preview/not production as of 2025) | confirmed | software-stack, search-results |
| Op Library | `not applicable` — no separate cuDNN/MIOpen equivalent; Glow compiler handles all op fusion and lowering to RISC-V tensor instructions | confirmed | software-stack |
| Kernel Library | General Purpose HPC SDK (C/C++ on ET-Minion cores; OpenMP thread parallelism; RISC-V vector intrinsics; vector transcendental math library; launched May 2023) | confirmed | software-stack, search-results |
| Runtime | ONNXRuntime ET-SoC-1 EP (intercepts op dispatch; loads compiled partitions; manages I/O) | confirmed | software-stack, search-results |
| Runtime | Native RISC-V runtime (ET-Minion core init, SRAM tile allocation, PCIe/DMA, result collection; fully static — no dynamic scheduling) | confirmed | software-stack |
| Runtime | Linux on ET-Maxion (4 OoO cores run full Linux OS; standalone server mode; ET-Minion pool management) | confirmed | software-stack, hw-architecture |
| Driver / Firmware | PCIe Linux kernel module (BAR mapping, DMA, interrupt handling) | confirmed | software-stack |
| Driver / Firmware | No separate firmware RM (ET-Maxion Linux handles all resource management; unlike NVIDIA GSP) | confirmed | software-stack |
| Communication | Shared on-chip SRAM (intra-chip ET-Minion barrier sync and data exchange) | confirmed | hw-architecture |
| Communication | Host PCIe multi-chip coordination on Glacier Point v2 (no dedicated chip-to-chip fabric; host-mediated) | inferred | search-results |
| Communication | No collective library (no NCCL equivalent; multi-chip parallelism handled at Glow compiler graph-partition level) | confirmed | software-stack |
| Assembler / ISA | RISC-V RV64GCV (standard open ISA; full GCC/LLVM toolchain; debuggable with gdb; no PTX-equivalent abstraction needed) | confirmed | hw-architecture, software-stack |
| Assembler / ISA | Esperanto custom vector/tensor ISA extensions (ET-Minion specific; 32K-op tensor instructions; transcendental unit ops; encoding not public) | confirmed | hw-architecture |
| Source Availability | **[ETSP 2026]** `github.com/openhwgroup/core-et` + `core-et-erbium` (created 2026-04-20) — clean-SystemVerilog **re-translation** of the acquired IP for "agentic hardware-development workflows"; original kept on a separate branch; NOT Esperanto's production ET-SoC-1 RTL | confirmed | software-stack |
| Source Availability | **[ETSP 2026]** License: checked-in `core-et/LICENSE` is verbatim **Apache-2.0**; the 2026-06-02 press release instead names **Solderpad Hardware License v2.1** ("building on the well-known Apache 2.0 license") — discrepancy recorded, not resolved | confirmed | software-stack |
| Governance | **[ETSP 2026]** "CORE-ET Silicon Platform (ETSP)" accepted as an **OpenHW Foundation** project 2026-06-02 (OpenHW Foundation is "an Eclipse Foundation global initiative"); Ainekko listed as OpenHW member | confirmed | software-stack |
| Framework Integration / Compiler / Runtime | **[ETSP 2026]** No SDK, compiler, or runtime disclosed for the 16nm successor. Whether the Glow/LLVM/ONNXRuntime-EP stack carries forward is **not disclosed** | not disclosed | software-stack |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | 1,088 ET-Minion in-order RISC-V cores (RV64GCV; 128 INT8 GOPS per GHz per core; fine-grained multithreading) | confirmed | hw-architecture, search-results |
| Compute Engine | Per-ET-Minion vector/tensor unit (32K MACs per tensor instruction up to 512 cycles; vector transcendental unit for exp/log/sin/cos/tanh/sigmoid) | confirmed | hw-architecture, search-results |
| Compute Engine | 4 ET-Maxion out-of-order cores (quad-issue; branch prediction; prefetch; runs Linux/OS) | confirmed | hw-architecture |
| Compute Engine | Chip peak: 100–200 INT8 TOPS (frequency-dependent); TDP <20W; ~5–10 TOPS/W | confirmed | hw-architecture, search-results |
| Compute Engine | TSMC 7nm, 24+ billion transistors, 570 mm² die, 89 mask layers | confirmed | hw-architecture, search-results |
| Data Path | SPMD massively parallel: each ET-Minion runs independent RISC-V thread; cooperate via shared SRAM | confirmed | hw-architecture |
| Data Path | Long tensor instructions (up to 512 cycles / 32K ops) overlap with integer pipeline; no warp scheduling needed | confirmed | hw-architecture |
| On-chip Memory | >160 MB distributed on-chip SRAM (software-managed; weight tiles + activations; no hardware caches) | confirmed | hw-architecture, search-results |
| Off-chip Memory | LPDDR4x DRAM (low-power; up to 32 GB per chip; ~137 GB/s bandwidth; eMMC Flash for persistence) | confirmed | hw-architecture, search-results |
| Host Interface / Package | PCIe Gen4 x8 | confirmed | hw-architecture, search-results |
| Host Interface / Package | ET-SoC-1 operable as PCIe accelerator OR standalone Linux server (via ET-Maxion cores) | confirmed | hw-architecture |
| Scale-up Interconnect | Glacier Point v2 card: 6 × ET-SoC-1 (6,558 cores, 192 GB LPDDR4x, 822 GB/s BW, ~600–1,200 INT8 TOPS) | confirmed | search-results |
| Scale-up Interconnect | No dedicated chip-to-chip fabric; multi-chip coordination via host PCIe (host-mediated DRAM sharing) | inferred | search-results |
| Scale-out Interconnect | Standard host PCIe/Ethernet (no proprietary scale-out; multiple cards via host x86 server) | confirmed | search-results |
| Compute Engine | **[ETSP 2026]** CORE-ET Silicon Platform on a **16nm** node (foundry not named), **tapeout in progress** — not sampling, not shipping. Core count, peak TOPS, and TDP **not disclosed** | confirmed (status); not disclosed (figures) | hw-architecture |
| On-chip Memory | **[ETSP 2026]** Integrated **MRAM-based** on-die memory (MRAM described as having "SRAM-like performance"); first non-volatile on-die tier in the lineage. Capacity, bandwidth, density **not disclosed**. Vendor brand "iRAM" appears only on nekko.ai, not in any press release | confirmed (presence); not disclosed (figures) | hw-architecture |
| Off-chip Memory | **[ETSP 2026]** Not disclosed. ET-SoC-2x/3x's planned HBM was **not** revived | not disclosed | hw-architecture |
| Scale-up Interconnect | **[ETSP 2026]** "Scalable network-on-chip (NoC) architecture" — an *intra-die* fabric; topology, width, and bandwidth **not disclosed**. No chip-to-chip fabric disclosed | confirmed (presence); not disclosed (figures) | hw-architecture |
| Host Interface / Package | **[ETSP 2026]** Not disclosed | not disclosed | hw-architecture |
