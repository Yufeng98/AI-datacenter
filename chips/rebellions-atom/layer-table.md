# Rebellions ATOM / REBEL Layer Mapping Table

*chip: rebellions-atom*
*device_class: Inference Accelerator (Korea)*
*as_of: 2026-08-08*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (torch.export + rebel.compile(); graph trace to RBLN IR) | confirmed | rbln-sdk-docs, rbln-compiler-blog |
| Framework Integration | TensorFlow (SavedModel / TF2 concrete function graph capture) | confirmed | rbln-sdk-docs |
| Framework Integration | HuggingFace Transformers (optimum-rbln: RBLNAutoModelForCausalLM, Seq2Seq, etc.) | confirmed | optimum-rbln |
| Framework Integration | HuggingFace Diffusers (optimum-rbln: RBLNStableDiffusionPipeline, SDXL) | confirmed | optimum-rbln |
| Compiler / IR | RBLN Frontend Compiler (graph capture, operator lowering, RBLN IR) | confirmed | rbln-compiler-blog |
| Compiler / IR | RBLN Dependency Analysis (DRAM/SRAM access patterns; preserves inter-op dependencies) | confirmed | rbln-compiler-blog |
| Compiler / IR | RBLN Backend Compiler (tiling, scheduling, RISC ISA codegen via Compute Library) | confirmed | rbln-compiler-blog |
| Compiler / IR | Global graph optimization (cross-operation scheduling, memory allocation, parallel execution) | confirmed | rbln-compiler-blog |
| Op / Kernel Library | RBLN Compute Library (GEMM, normalization, activation, convolution; internal; generates RISC ISA) | confirmed | rbln-compiler-blog |
| Op / Kernel Library | RBLN Model Zoo (pre-compiled .rbln binaries: LLaMA, SDXL, BERT, T5, ResNet) | confirmed | rbln-model-zoo |
| Runtime | RBLN Runtime Python API (rbln.Runtime.load, .run; device memory management) | confirmed | rbln-sdk-docs |
| Runtime | Multi-instance scheduling (16 hardware-isolated instances on ATOM) | confirmed | atom-architecture-blog |
| Runtime | PCIe DMA engine (host-device tensor transfers) | confirmed | atom-architecture-blog |
| Serving | NVIDIA Triton Inference Server backend plugin | confirmed | rbln-sdk-docs |
| Serving | vLLM official plugin (LLM token streaming on ATOM/REBEL); automatic vLLM compilation (separate precompile step removed, SDK 2026-05-29) | confirmed | vllm-rfc-7247, rbln-release-notes |
| Driver | Linux kernel module (PCIe Gen5 device management, BAR mapping, DMA); modular driver architecture since v3.0.0 (2026-01-30); current v3.2.2 | confirmed | rbln-release-notes |
| Model Optimization | **OwLite** — quantization / model-compression toolkit (SqueezeBits; acquired by Rebellions 2026-06-30) | confirmed (product exists; RBLN SDK integration not disclosed) | rebellions-squeezebits-2026-06-30, thelec-squeezebits |
| Model Optimization | **Fits on Chips** — LLM benchmarking / LLMOps toolkit (SqueezeBits) | confirmed (product exists; RBLN SDK integration not disclosed) | rebellions-squeezebits-2026-06-30 |
| Serving | **Yetter** — SqueezeBits inference engine (relationship to the vLLM plugin not disclosed) | confirmed (product exists; RBLN SDK integration not disclosed) | rebellions-squeezebits-2026-06-30 |
| Cluster / Orchestration | RBLN NPU Operator (Kubernetes device plugin, Helm chart); **breaking v0.3.x → v0.4.0 in the 2026-04-30 SDK** — `spec.daemonsets` → `spec.podDefaults`, auto device detection replaces static ConfigMap, generic resource mode `rebellions.ai/npu` default | confirmed | rbln-release-notes |
| Tooling / Diagnostics | `rbln-vs` hardware diagnostics (new 2026-05-29); `rbln-smi` monitoring CLI; `rbln-smd` device daemon (renamed from `rbln_daemon`); `torch.rbln.explain()` CPU-fallback diagnosis and `torch.rbln.synchronize()` (new 2026-07-31) | confirmed | rbln-release-notes |
| Compiler / IR | Binary-format compatibility break at compiler v0.11.0.post1 (2026-06-26, Transformers v5) — models compiled with earlier SDKs must be recompiled; `tensor_parallel_size` renamed `num_devices` | confirmed | rbln-release-notes |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | ATOM: CGRA (Coarse-Grained Reconfigurable Array), 8 Neural Engines, Samsung 5nm | confirmed | atom-white-paper, atom-arch-blog |
| Compute Engine | ATOM: 32 TFLOPS FP16, 128 TOPS INT8 | confirmed | atom-arch-blog |
| Compute Engine | REBEL Single: 16 Neural Cores (8 per cluster, mesh interconnect), Samsung 4nm | confirmed | next-platform-rebel |
| Compute Engine | REBEL-Quad: 4 × 320 mm² ASIC dies, Samsung 4nm; 2,048 TFLOPS FP8 | confirmed | hot-chips-2025, sth-rebel-quad |
| Compute Engine | REBEL-Quad: UCIe-Advanced die-to-die, 16 Gbps, 4 TB/s aggregate | confirmed | hot-chips-2025, sth-rebel-quad |
| Data Path | Static compile-time scheduling (all tiling, memory allocation, dependency resolution at compile time) | confirmed | rbln-compiler-blog |
| Data Path | Command Processor (Neural Engine dispatch and instruction fetch) | confirmed | atom-arch-blog |
| Data Path | Neural Engine IBUF (Input Buffer with custom instruction set) | confirmed | next-platform-rebel |
| On-chip Memory | ATOM: 64 MB flat SW-managed SRAM (no caches) | confirmed | atom-arch-blog |
| On-chip Memory | REBEL Single: 64 MB L1 distributed (per-core) + 64 MB L2 shared = 128 MB | confirmed | next-platform-rebel |
| On-chip Memory | REBEL-Quad: 4 × 128 MB = 512 MB total SW-managed SRAM | confirmed | next-platform-rebel, sth-rebel-quad |
| Off-chip Memory | ATOM: 16 GB GDDR6, 256 GB/s (RBLN-CA12 card) | confirmed | atom-arch-blog |
| Off-chip Memory | REBEL-Quad: 144 GB HBM3e (4 × 36 GB 12Hi), 4.8 TB/s aggregate | confirmed | sth-rebel-quad, tomshardware-isscc2026 |
| Host Interface | ATOM: PCIe Gen5 x16, FHFL single-slot card (RBLN-CA12), TDP 60–130 W | confirmed | atom-arch-blog |
| Host Interface | ATOM: 16 hardware-isolated multi-instance partitions | confirmed | atom-arch-blog |
| Host Interface | REBEL-Quad: PCIe card, ~600 W TDP; 8 cards per air-cooled node (reference design) | confirmed | sth-rebel-quad, theregister-rebelrack |
| Host Interface | **RebelCard**: module-type Rebel100 accelerator card, air-cooled; "four NPU chiplets" + "5th-generation HBM" (HBM3E). Capacity, bandwidth, TDP, performance **not disclosed**. Announced 2026-04-10; in validation at SK Telecom — not shipping, no confirmed ship date | announced (specs not disclosed) | rebellions-skt-arm-2026-04-10 |
| Host Interface | **Host CPU pairing: Arm AGI CPU** (Arm Neoverse CSS V3; "first Arm-designed data center CPU" per the release) — MOU-stage co-development, not a shipping platform | announced (MOU) | rebellions-skt-arm-2026-04-10 |
| Scale-up Interconnect | REBEL-Quad: UCIe-Advanced (world's first in AI accelerator), 4 TB/s intra-package | confirmed | hot-chips-2025 |
| Scale-up Interconnect | RebelRack: 32 accelerators, 4 nodes, quad-400 GbE/node, 64 PFLOPS FP8, 4.6 TB HBM3e | ⚠️ secondary-sourced (re-graded 2026-08-08 — the 2026-03-30 Rebellions launch release published no specs) | theregister-rebelrack |
| Scale-up Interconnect | RebelPOD: 8–128 nodes × 8 accelerators, 800 GbE | ⚠️ secondary-sourced (re-graded 2026-08-08) | theregister-rebelrack |
| Scale-up Interconnect | **RebelServer**: single-server product; ran SKT A.X K1 (500B+ parameter MoE) in one server. Device count, throughput, latency, TPS/W **not disclosed** | announced (specs not disclosed) | rebellions-rebelserver-k1, sedaily-ax-k1 |
| Scale-out Interconnect | Ethernet (400 GbE / 800 GbE); no InfiniBand | confirmed | theregister-rebelrack |
| Manufacturing | Samsung Electronics fab partnership (5nm ATOM, 4nm REBEL) | confirmed | techcrunch-124m, samsung-5nm-atom |
| Manufacturing | Arm architecture license (investor + IP partner) | confirmed | rebellions-250m-arm-samsung |
| OEM / System Channel | Giga Computing (GIGABYTE server arm) MOU to co-develop AI servers and rack-scale systems around RebelCard/Rebel100 — no timeline, no SKU, no specs | announced (MOU) | rebellions-gigacomputing-2026-06-17 |

### Source keys added 2026-08-08

| Key | URL |
|---|---|
| rbln-release-notes | https://docs.rbln.ai/latest/supports/release_note.html |
| rebellions-squeezebits-2026-06-30 | https://rebellions.ai/newsroom/rebellions_squeezebits_acquisition_260630/ |
| thelec-squeezebits | https://www.thelec.net/news/articleView.html?idxno=11826 |
| rebellions-skt-arm-2026-04-10 | https://rebellions.ai/newsroom/rebellions-collaborates-with-sk-telecom-and-arm-targeting-sovereign-ai-and-telecom-infrastructure/ |
| rebellions-gigacomputing-2026-06-17 | https://rebellions.ai/newsroom/rebellions-and-giga-computing-sign-mou-to-develop-next-generation-ai-server-and-rack-scale-solutions/ |
| rebellions-rebelserver-k1 | https://rebellions.ai/newsroom/rebelserver_run_k1/ |
| sedaily-ax-k1 | https://en.sedaily.com/technology/2026/07/23/rebellions-runs-skts-sovereign-model-ax-k1-on-npu-server |

*Compute Engine, Data Path, On-chip Memory and Off-chip Memory rows are unchanged as of 2026-08-08: no new silicon generation was announced in the 2026-04-06 → 2026-08-08 window.*
