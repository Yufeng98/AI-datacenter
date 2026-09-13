# T-Head (平头哥) Layer Mapping Table

*as_of: 2026-08-08* (baseline 2026-04-05; rows marked **2026-08** added or revised in the Zhenwu M890 / Panjiu AL128 / SAIL update)

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (Zhenwu 810E stack; primary training+inference framework; frontend API compatibility layer) | confirmed | software-stack, search-results |
| Framework Integration | ONNX (model interchange; supported by both HGAI and Zhenwu stacks) | confirmed | software-stack, search-results |
| Framework Integration | TensorFlow (supported by HGAI SDK and Zhenwu compiler; inferred third-tier support) | confirmed | software-stack |
| Framework Integration | MXNet (HGAI SDK only; Alibaba-native; Hanguang 800 inference) | confirmed | software-stack |
| Framework Integration | Caffe (HGAI SDK; Hanguang 800 legacy inference) | confirmed | software-stack |
| Framework Integration | Alibaba Cloud Model Studio API (cloud inference endpoint for Zhenwu 810E; Qwen/DeepSeek serving) | confirmed | search-results |
| Framework Integration | **2026-08** — SAIL framework compatibility: vendor claims 260+ mainstream training/inference frameworks, explicitly PyTorch, TensorFlow, **vLLM**, **SGLang** (announced 2026-07-18, WAIC Shanghai) | vendor claim (not independently benchmarked) | software-stack (2026-08-08) |
| Framework Integration | **2026-08** — Alibaba Bailian model service platform (serving endpoint for the Panjiu AL128 supernode; the only availability Alibaba asserts for the M890 generation) | confirmed | hw-architecture (2026-08-08) |
| Compiler / IR | HGAI Graph IR (Hanguang 800; proprietary; frontend → unified Graph IR conversion) | confirmed | software-stack, search-results |
| Compiler / IR | Zhenwu Compiler (proprietary; MLIR-based internal IR inferred; training+inference lowering) | confirmed | software-stack, search-results |
| Compiler / IR | INT8 quantization pass (post-training quantization; Hanguang 800 primary precision) | confirmed | hw-architecture, search-results |
| Compiler / IR | Sparse weight compression pass (Hanguang 800; reduces SRAM I/O for sparse tensors) | confirmed | hw-architecture, search-results |
| Compiler / IR | Graph partitioning → TE/PE/ME mapping (Hanguang 800; maps ops to Tensor/Pooling/Memory engines per core) | confirmed | hw-architecture |
| Compiler / IR | SRAM tiling (Hanguang 800; tiles activations across 48 MB SRAM per core) | confirmed | hw-architecture |
| Compiler / IR | ICN collective scheduling (Zhenwu 810E; 7-link inter-chip AllReduce/AllGather analog) | confirmed | hw-architecture, search-results |
| Compiler / IR | **2026-08** — SAIL compiler layer (announced as part of the open-sourced SAIL stack, 2026-07-18; IR design, passes and internal name **not disclosed**; no public repo or license confirmed) | vendor-asserted availability | software-stack (2026-08-08) |
| Compiler / IR | **2026-08** — CUDA migration path: vendor claims porting in "fewer than seven days" (SCMP); one technical deep-dive instead claims migration "without modifying existing code". No independent benchmark; the two claims are not equivalent | vendor claim (contested wording) | software-stack (2026-08-08) |
| Compiler / IR | **2026-08** — FP4 lowering implied by M890's "FP32 down to FP4" native precision range; no compiler-side FP4 documentation published | inferred | hw-architecture (2026-08-08) |
| Op Library | CNN ops: convolution, deconvolution, dilated conv, 3D conv, interpolation, ROI (Hanguang 800 Tensor Engine) | confirmed | hw-architecture, search-results |
| Op Library | Pooling ops: max/average/global (Hanguang 800 Pooling Engine) | confirmed | hw-architecture |
| Op Library | GEMM / matmul (Hanguang 800 Tensor Engine; Zhenwu 810E; core compute primitive) | confirmed | hw-architecture |
| Op Library | Activation functions (extensible; domain-programmable for future functions) | confirmed | hw-architecture |
| Op Library | Qwen LLM-optimized ops (Zhenwu 810E; attention, feedforward, normalization for Qwen3/DeepSeek V3) | confirmed | search-results |
| Op Library | **2026-08** — SAIL high-performance operator library (named as a SAIL layer; op list, kernel authoring interface and coverage **not disclosed**) | vendor-asserted availability | software-stack (2026-08-08) |
| Runtime | HGRT — Hanguang Runtime (PCIe DMA, kernel launch, result collection; cloud-only) | confirmed | software-stack |
| Runtime | Zhenwu runtime (proprietary; cluster orchestration for 10K-card deployments; not public) | confirmed | search-results |
| Runtime | NPUSMI monitoring (Hanguang 800; frequency, memory utilization, compute utilization) | confirmed | software-stack, search-results |
| Runtime | C950 native LLM inference runtime (no separate NPU SDK; integrated AI engine; Qwen3/DeepSeek V3) | confirmed | search-results |
| Runtime | **2026-08** — SAIL SDK / toolchain layer (named as a SAIL layer; distributed via the T-Head developer community, **no public Git URL or license confirmed**) | vendor-asserted availability | software-stack (2026-08-08) |
| Runtime | **2026-08** — SAIL performance-analysis / debug tooling (named as a SAIL layer; tool names and capabilities **not disclosed**) | vendor-asserted availability | software-stack (2026-08-08) |
| Runtime | **2026-08** — SAIL layer decomposition is **not settled**: several outlets describe five layers (drivers → compiler → operator+communication libs → SDK/toolchain → profiling/debug); one technical deep-dive describes three (interface / SDK / OS) | contested | software-stack (2026-08-08) |
| Driver / Firmware | PCIe host driver (Hanguang 800; BAR mapping, DMA, interrupt; cloud-only, not open-source) | confirmed | software-stack |
| Driver / Firmware | Zhenwu PCIe/ICN driver (proprietary, closed through the 2026-04-05 baseline; 400W board management) — **revised 2026-08:** SAIL is announced to include a kernel-driver layer with native adaptation to the OpenAnolis **Anolis Cloud Kernel (ANCK)**; no repository or license confirmed, so "open" remains vendor-asserted | confirmed (baseline) / vendor-asserted (SAIL) | search-results, software-stack (2026-08-08) |
| Communication | ICN (Inter-Chip Network) — 7 links, 700 GB/s aggregate (Zhenwu 810E; proprietary protocol) | confirmed | hw-architecture, search-results |
| Communication | Collective operations over ICN (AllReduce/AllGather analogs for multi-card training) | confirmed | hw-architecture |
| Communication | **2026-08** — ICN on Zhenwu M890: **800 GB/s** inter-chip bandwidth (link count not disclosed) | confirmed (vendor spec) | hw-architecture (2026-08-08) |
| Communication | **2026-08** — **ICN Switch 1.0** switch ASIC: up to **25.6 Tbps** aggregate, congestion-free across **64 accelerators**; latency hundred-nanosecond class (百纳秒级, ITHome/YiCai; BigGo "<150 ns") — no latency figure in Alibaba's English release; "sub-100 ns" in circulation is a mistranslation | confirmed (bandwidth) / secondary (latency) | hw-architecture (2026-08-08) |
| Communication | **2026-08** — SAIL communication library (paired with the operator library in the SAIL layer description; collective API, NCCL-equivalence and coverage **not disclosed**) | vendor-asserted availability | software-stack (2026-08-08) |
| Assembler / ISA | No user-visible ISA for Hanguang/Zhenwu (no PTX equivalent; fully proprietary binary format) | confirmed | software-stack |
| Assembler / ISA | Xuantie GCC/Binutils toolchain (RISC-V ISA + RVV vector extension; open-source Apache 2.0) | confirmed | software-stack, search-results |
| Assembler / ISA | LLVM upstream support for Xuantie RISC-V (via standard RISC-V backend + vendor extensions) | inferred | software-stack |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Tensor Engine (TE) — convolution, GEMM, deconvolution, dilated/3D conv, interpolation, ROI (Hanguang 800, per core) | confirmed | hw-architecture, search-results |
| Compute Engine | Pooling Engine (PE) — max/avg/global pooling, activation functions (Hanguang 800, per core) | confirmed | hw-architecture |
| Compute Engine | Memory Engine (ME) — DMA, data movement, sparse decompression (Hanguang 800, per core) | confirmed | hw-architecture |
| Compute Engine | Zhenwu 810E AI compute array (proprietary; training+inference; Qwen-optimized; architecture not disclosed) | confirmed | search-results |
| Compute Engine | **2026-08** — Zhenwu M890 AI compute array (proprietary; integrated training+inference for agentic workloads; array type, unit counts, clock and peak FLOPS by dtype all **not disclosed**; vendor claims 3× the 810E with no methodology) | confirmed (existence) / not disclosed (microarchitecture) | hw-architecture (2026-08-08) |
| Compute Engine | **2026-08** — M890 numeric range: native precision **"from FP32 down to FP4"** (Alibaba wording) — first FP4 claim in the Zhenwu line | confirmed (vendor spec) | hw-architecture (2026-08-08) |
| Compute Engine | C950 AI acceleration engine (self-developed; native 100B+ param LLM; integrated with RISC-V OoO core) | confirmed | search-results |
| Compute Engine | C910/C920/C930/C950 OoO RISC-V cores (3-wide, 12-stage; RVV 0.7.1/1.0/RVA23; 2–3.2 GHz) | confirmed | hw-architecture, search-results |
| Data Path | 4-core ring bus (Hanguang 800; CP command processor coordinates all cores) | confirmed | hw-architecture, search-results |
| Data Path | Tensor streaming pipeline (Hanguang 800; full-parallel dataflow for CNN; sparse compression path) | confirmed | hw-architecture |
| Data Path | ICN 7-link data path (Zhenwu 810E; chip-to-chip; near-linear scaling topology) | confirmed | hw-architecture |
| On-chip Memory | 192 MB distributed SRAM / ~48 MB per core (Hanguang 800; no HBM; primary weight + activation storage) | confirmed | hw-architecture, search-results |
| On-chip Memory | On-chip SRAM (Zhenwu 810E; size not publicly disclosed; supplemented by 96 GB HBM2e) | inferred | hw-architecture |
| On-chip Memory | **2026-08** — Zhenwu M890: **144 GB** described by Alibaba only as **"on-chip memory"**; memory *type* and bandwidth **not disclosed**. Secondary coverage calls it HBM, some specifically HBM3 — not vendor-supported. Any SRAM/DRAM split is unpublished | confirmed (capacity) / not disclosed (type, BW) | hw-architecture (2026-08-08) |
| On-chip Memory | C910: in-core L1/L2 SRAM caches + scratchpad (standard RISC-V memory model) | confirmed | hw-architecture |
| Off-chip Memory | PCIe Gen4 x16 → host DRAM (Hanguang 800; no on-chip HBM) | confirmed | hw-architecture, search-results |
| Off-chip Memory | 96 GB HBM2e (Zhenwu 810E; 4–6 stacks) | confirmed | hw-architecture, search-results |
| Host Interface / Package | PCIe Gen4 x16 (Hanguang 800) | confirmed | hw-architecture |
| Host Interface / Package | TSMC 12nm (Hanguang 800); 17B transistors | confirmed | hw-architecture, search-results |
| Host Interface / Package | SMIC 7nm domestic (Zhenwu 810E); 400 W TDP board | confirmed | search-results |
| Host Interface / Package | **2026-08** — Zhenwu M890: process node, foundry, TDP, die size and transistor count all **not disclosed**; only that it must use a domestically-accessible node (The Register, TNW) | not disclosed | hw-architecture (2026-08-08) |
| Host Interface / Package | TSMC 5nm (XuanTie C950); TSMC 12nm (C910) | confirmed | search-results |
| Scale-up Interconnect | 7× ICN links, 700 GB/s (Zhenwu 810E; proprietary protocol) | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | No scale-up fabric (Hanguang 800; single-chip, PCIe-only) | confirmed | hw-architecture |
| Scale-up Interconnect | **2026-08** — **Panjiu AL128 supernode**: 128 accelerators per rack-scale unit (64-card option), "petabyte-per-second internal bandwidth" (vendor claim, repeated by The Register); no per-link or bisection breakdown published. First rack-scale scale-up domain recorded for T-Head. **Novelty caveat:** the AL128 name and its ScaleUp/ScaleOut/DCN architecture were published by Alibaba Cloud 2025-11-14 — only the M890 + ICN Switch 1.0 instantiation is new | confirmed (topology scale) / vendor claim (PB/s) | hw-architecture (2026-08-08) |
| Scale-up Interconnect | **2026-08** — **ICN Switch 1.0** T-Head switch ASIC: up to 25.6 Tbps aggregate; 64-accelerator congestion-free domain; port count, SerDes rate and process **not disclosed** | confirmed | hw-architecture (2026-08-08) |
| Scale-out Interconnect | Alibaba Cloud data center network (Zhenwu 810E clusters; 10K+ card deployments) | confirmed | search-results |
| Scale-out Interconnect | **2026-08** — Panjiu AL128 ScaleOut / DCN tiers above the 128-card scale-up domain (architecture published by Alibaba Cloud 2025-11-14; per-tier bandwidths **not disclosed**) | confirmed (existence) / not disclosed (specs) | hw-architecture (2026-08-08) |
| Scale-out Interconnect | No proprietary scale-out fabric for Hanguang 800 | confirmed | hw-architecture |
