# Cerebras Wafer-Scale Engine Software Stack & Hardware Resources

*as_of: 2026-08-08 (baseline scan 2026-04-05; roadmap resources appended 2026-08-08)*
*device_class: Wafer-Scale Engine*
*seeds: https://training-docs.cerebras.ai/, https://github.com/Cerebras*

---

## Software Stack

### Framework Integration
- [Cerebras PyTorch API — Developer Documentation](https://training-api.cerebras.ai/en/latest/wsc/api/cerebras_pytorch/) — Official PyTorch API layer for WSE, built on Lazy Tensor Core (LTC), PyTorch 2.0-compatible backend
- [Supporting PyTorch on the Cerebras WSE — Blog](https://www.cerebras.ai/blog/supporting-pytorch-on-the-cerebras-wafer-scale-engine) — Technical overview of PyTorch integration; lazy tensor capture, graph compilation via CIRH IR
- [Porting PyTorch Models to Cerebras — Training Docs](https://training-docs.cerebras.ai/rel-2.7.0/model-zoo/migration/porting-pytorch-models-to-cerebras) — Step-by-step migration guide (data loaders, optimizers, mixed precision)
- [Cerebras Software Release 2.0 — Blog](https://www.cerebras.ai/blog/cerebras-software-release-2.0-50-faster-training-pytorch-2.0-support-diffusion-transformers-and-more) — PyTorch 2.0 support, Diffusion Transformer models, 50% faster training
- [Cerebras Training and Inference Docs](https://docs.cerebras.net/en/latest/) — Canonical developer documentation hub for training + inference
- [Cerebras Training Docs (rel-2.7)](https://training-docs.cerebras.ai/) — Seed URL: official training documentation portal
- [Cerebras Developer Documentation](https://training-api.cerebras.ai/) — Full API reference for Cerebras software stack

### Compiler / IR
- [Cerebras Software Product](https://www.cerebras.ai/product-software) — Overview of the full software stack: PyTorch → CIRH (Cerebras Intermediate Representation for Hardware) → device binary
- [Compile Report — Developer Docs](https://training-api.cerebras.ai/en/rel-2.3.1/original/compiler-reports/compile-report.html) — First-party compiler output documentation: compile report format, layer-by-layer analysis, performance projections
- [Incremental Compile — Developer Docs](https://training-api.cerebras.ai/en/rel-2.3.1/original/compiler-reports/incremental-compile.html) — Incremental compilation: re-use previously compiled artifacts when model structure is unchanged
- [Kernel Autogeneration with AutoGen](https://training-api.cerebras.ai/en/latest/wsc/Fundamentals/autogen.html) — AutoGen: automatic kernel generation from high-level op descriptions; eliminates manual CSL kernel writing for common ops
- [Cerebras Architecture Deep Dive — IEEE Micro 2023](https://8968533.fs1.hubspotusercontent-na2.net/hubfs/8968533/IEEE%20Micro%202023-03%20Hot%20Chips%2034%20Cerebras%20Architecture%20Deep%20Dive.pdf) — Peer-reviewed deep dive on HW/SW co-design including compilation pipeline
- [Trainer Configuration Overview — YAML Config](https://docs.cerebras.net/en/latest/wsc/Model-zoo/yaml/index.html) — YAML-driven Trainer class; primarily a runtime workflow tool but specifies compile-time settings (precision, sequence length)
- [Weight Streaming Execution — Cerebras Developer Docs](https://docs.cerebras.net/en/latest/wsc/cerebras-basics/cerebras-execution-modes.html) — Two execution modes (Layer Pipelined vs Weight Streaming) and how the compiler partitions compute accordingly

### Op Library
- [Sparsity Made Easy — Cerebras PyTorch Sparsity Library](https://www.cerebras.ai/blog/sparsity-made-easy-introducing-the-cerebras-pytorch-sparsity-library) — Native sparse training ops integrated into PyTorch; GMP, SRigL pruning algorithms
- [Cerebras Model Zoo — multimodal/llava](https://github.com/Cerebras/modelzoo/tree/main/src/cerebras/modelzoo/models/multimodal/llava) — LLaVA multimodal operator implementations
- [Cerebras Model Zoo — NLP models](https://github.com/Cerebras/modelzoo) — Reference implementations: BERT, GPT-2/3, T5, Llama, Mixtral, DINOv2; also includes operator-level optimizations

### Kernel Library
- [Cerebras SDK Documentation (1.4.0)](https://sdk.cerebras.net/) — Low-level CSL kernel development environment for WSE cores
- [A Conceptual View — SDK 1.4.0](https://sdk.cerebras.net/computing-with-cerebras) — Explains PE model, wavelets, colors, and task-based kernel execution
- [GEMV Tutorial 0: Basic CSL Syntax](https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax/) — Hands-on tutorial for writing custom kernels in CSL
- [GitHub — Cerebras/sdk-examples](https://github.com/Cerebras/sdk-examples) — Official CSL kernel examples: stencils, GEMM, stochastic rounding, custom ops
- [Cerebras SDK Technical Overview Whitepaper](https://8968533.fs1.hubspotusercontent-na2.net/hubfs/8968533/Cerebras%20SDK%20Technical%20Overview%20White%20Paper.pdf) — Architecture-level explanation of CSL compilation and PE execution model
- [SDK Release Notes — Cumulative](https://sdk.cerebras.net/sdk-release-notes/sdk-rel-notes-cumulative) — All SDK releases with feature additions and breaking changes
- [What's New in R0.6 of the Cerebras SDK](https://www.cerebras.ai/blog/whats-new-in-r0.6-of-the-cerebras-sdk) — Notable features: multi-PE launch, improved CSL tasks API
- [Supercharge your HPC Research with the Cerebras SDK](https://www.cerebras.ai/blog/supercharge-your-hpc-research-with-the-cerebras-sdk) — Use-case blog for scientific computing kernels
- [Automated Code Generation for Dataflow Architecture (SC24)](https://dl.acm.org/doi/abs/10.1109/SC41406.2024.00025) — High-order stencil code gen targeting Cerebras CSL at SC'24

### Runtime
- [Weight Streaming Execution — Developer Docs](https://training-docs.cerebras.ai/rel-2.5.0/concepts/weight-streaming-execution) — Weight streaming runtime: how weights flow from MemoryX → SwarmX → WSE on each layer
- [Scaling Up and Out: Training Massive Models Using Weight Streaming](https://www.cerebras.ai/blog/scaling-up-and-out-training-massive-models-on-cerebras-systems-using-weight-streaming) — Runtime blog detailing MemoryX + SwarmX weight scheduling
- [Linear Scaling Made Possible with Weight Streaming](https://www.cerebras.ai/blog/linear-scaling-made-possible-with-weight-streaming) — Data on near-linear scaling across CS-2 clusters using the streaming runtime
- [Training and Fine-tuning an LLM — Developer Docs](https://docs.cerebras.net/en/latest/wsc/Model-zoo/tutorials/llm.html) — End-to-end runtime walkthrough: data loading, compile, training loop, checkpointing
- [SdkRuntime API Reference — sdk.cerebras.net](https://sdk.cerebras.net/api-docs/sdkruntime-api) — **Primary runtime API**: SdkRuntime host class, compile/run methods, symbol I/O, memcpy to/from device
- [SDK Appliance API Reference](https://sdk.cerebras.net/api-docs/appliance-api.html) — CS-3 appliance mode: job lifecycle, device handle, async execution control
- [Cerebras Training and Fine Tuning (rel-2.8.0)](https://cerebras-training.mintlify.app/rel-2.8.0/model-zoo/tutorials/training-and-fine-tuning-a-large-language-model--llm) — Latest training runtime docs including Llama-3 configurations

### Driver / Firmware
- [SdkRuntime API Reference — sdk.cerebras.net](https://sdk.cerebras.net/api-docs/sdkruntime-api) — **PRIMARY**: SdkRuntime host-side API for loading, launching, and controlling CSL programs on WSE; the main host-device interface
- [SDK Appliance API Reference — sdk.cerebras.net](https://sdk.cerebras.net/api-docs/appliance-api.html) — Appliance mode API for CS-3 systems: job submission, device allocation, program lifecycle management
- [Job Scheduling and Monitoring (csctl)](https://training-api.cerebras.ai/en/1.9.1/wsc/getting-started/job-scheduler.html) — `csctl` CLI tool for job submission, queue management, Grafana dashboards, Slurm integration
- [Resource Requirements for Parallel Training and Compilation](https://training-api.cerebras.ai/en/rel-2.2.1/wsc/general/system_requirements.html) — System requirements: host CPU, compile server setup, communication daemon configuration
- [Evaluation / Cerebras CS-2 — SURF Knowledge Base](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/112592526/Evaluation+Cerebras+CS-2) — HPC center deployment notes: driver installation, network config, job scheduler integration

### Communication
- [Multi-Replica Data Parallel Training — Developer Docs](https://training-api.cerebras.ai/en/latest/original/general/multi-replica-data-parallel-training.html) — **PRIMARY software-visible comm**: `cerebras.pytorch.distributed` multi-replica API; gradient all-reduce semantics over SwarmX
- [CSL `<collectives_2d>` Library](https://sdk.cerebras.net/csl/language/libraries) — On-chip collective communication library for CSL kernels: reduce, broadcast, scatter/gather across 2D PE mesh
- [Collective Communication Tutorial — SDK](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-11-collectives) — Hands-on tutorial for implementing collective ops in CSL using `<collectives_2d>`
- [WSE-3 `<message_passing>` Library](https://sdk.cerebras.net/csl/language/libraries_wse3) — Point-to-point PE messaging library (WSE-3 only): explicit send/recv between arbitrary PEs
- [Announcing the Cerebras Architecture for Extreme-Scale AI](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai) — SwarmX + MemoryX architecture: tree topology, broadcast-reduce, RoCE RDMA, proprietary low-latency protocol
- [Inside the Cerebras Wafer-Scale Cluster — IEEE Micro 2024](https://dl.acm.org/doi/abs/10.1109/MM.2024.3386628) — Peer-reviewed paper on multi-CS cluster architecture and communication protocols
- [Training Giant Neural Networks Using Weight Streaming — Whitepaper](https://www.kisacoresearch.com/sites/default/files/documents/cs_weight_streaming_white_paper_-_cerebras.pdf) — MemoryX/SwarmX protocol: weight prefetch scheduling, gradient reduction pipeline
- [Job Scheduling and Monitoring (csctl / Grafana)](https://training-api.cerebras.ai/en/1.9.1/wsc/getting-started/job-scheduler.html) — Cluster job control and communication health monitoring via Grafana dashboards

### Assembler / ISA
- [CSL Language Guide — sdk.cerebras.net](https://sdk.cerebras.net/csl/language_index) — **PRIMARY**: Full CSL language reference: types, tasks, colors, wavelets, routing annotations, compile-time params
- [Data Structure Descriptors (DSDs) — CSL Language](https://sdk.cerebras.net/csl/language/dsds) — DSD types for addressing on-chip SRAM: 1D/2D/4D, stride, extent; core low-level memory addressing primitive
- [CSL Standard Libraries](https://sdk.cerebras.net/csl/language/libraries) — Built-in CSL libraries including `<collectives_2d>`, `<math>`, `<time>`, `<io>` — standard library for PE-level kernels
- [WSE-3 Libraries — `<message_passing>`](https://sdk.cerebras.net/csl/language/libraries_wse3) — WSE-3-only library for explicit PE-to-PE point-to-point messaging (new in WSE-3)
- [A Conceptual View — SDK Docs: Wavelets and Colors](https://sdk.cerebras.net/computing-with-cerebras) — Micro-ISA concepts: wavelets (32-bit messages), colors (virtual channels), task dispatch hardware
- [SPADA: Spatial Dataflow Architecture Programming Language](https://arxiv.org/pdf/2511.09447) — Academic paper on dataflow ISA abstractions relevant to Cerebras CSL
- [Using the Cerebras CS3 Dataflow — University of Edinburgh](https://era.ed.ac.uk/server/api/core/bitstreams/2ae66aa7-d373-4669-b6a9-3c2a70a17516/content) — Academic use-case paper documenting CSL low-level data movement instructions

---

## Hardware Architecture

### Compute Engine
- [Product — Chip (WSE-3)](https://www.cerebras.ai/chip) — Official WSE-3 specs: 900K AI cores, 4T transistors, 125 PF, 46,225 mm², TSMC 5nm
- [Cerebras WSE-3 — IEEE Spectrum](https://spectrum.ieee.org/cerebras-chip-cs3) — Detailed chip analysis including TSMC 5nm process, core count, compute density
- [Cerebras WSE-3 AI Chip Launched — ServeTheHome](https://www.servethehome.com/cerebras-wse-3-ai-chip-launched-56x-larger-than-nvidia-h100-vertiv-supermicro-hpe-qualcomm/) — Launch coverage: 56x larger than H100, thermal and power analysis
- [Cerebras Architecture Deep Dive — IEEE Xplore](https://ieeexplore.ieee.org/document/10123162/) — Peer-reviewed architecture paper on HW/SW co-design for deep learning
- [Hot Chips 2024 — Cerebras WSE-3 Presentation (PDF)](https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf) — Sean Lie's HC2024 talk: WSE-3 micro-architecture, PE layout, 2D mesh details
- [Hot Chips 31 — Wafer-Scale Deep Learning (PDF)](https://old.hotchips.org/hc31/HC31_1.13_Cerebras.SeanLie.v02.pdf) — WSE-1 original presentation; PE design, die-to-die interconnect
- [Cerebras WSE-3: The Wafer-Scale AI Engine — Awesome Agents](https://awesomeagents.ai/hardware/cerebras-wse-3/) — Consolidated technical specs with comparison tables
- [A Comparison of Cerebras WSE vs Nvidia GPU Systems — arXiv 2025](https://arxiv.org/html/2503.11698v1) — Comprehensive side-by-side hardware comparison paper

### Data Path
- [Cerebras Architecture Deep Dive — HW/SW Co-Design Blog](https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning) — 2D mesh fabric, router per core, wavelet routing, FP16/BF16 data paths
- [Cerebras Wafer Scale Engine: Why Big Chips — Medium](https://medium.com/@cerebras/cerebras-wafer-scale-engine-why-we-need-big-chips-for-deep-learning-142860b86f22) — Data movement motivation; on-chip bandwidth eliminates off-chip bottleneck
- [A Look at Cerebras WSE — WikiChip Fuse](https://fuse.wikichip.org/news/3010/a-look-at-cerebras-wafer-scale-engine-half-square-foot-silicon-chip/2/) — Detailed die-level analysis including routing fabric topology

### On-chip Memory
- [Cerebras Announces WSE-3 — Press Release](https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine) — 44 GB SRAM distributed across 900K cores, 21 PB/s bandwidth, no HBM
- [Cerebras Architecture Deep Dive — SRAM details](https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning) — Per-PE SRAM allocation; local + remote SRAM access patterns
- [A Comparison of Cerebras WSE vs Nvidia GPU — arXiv](https://arxiv.org/html/2503.11698v1) — SRAM vs HBM trade-off analysis; bandwidth per unit power comparison
- [Hacker News Discussion: 44GB SRAM](https://news.ycombinator.com/item?id=44660438) — Community technical discussion clarifying on-chip vs off-chip memory distinction

### Off-chip Memory
- [Announcing Cerebras Architecture for Extreme-Scale AI](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai) — MemoryX: elastic off-chip weight storage (4TB–2.4PB), DRAM+Flash, weight streaming protocol
- [Cerebras CS-3 System](https://www.cerebras.ai/system) — CS-3 MemoryX options: 24TB, 36TB (enterprise), 120TB, 1.2PB (hyperscale); up to 24T parameter models
- [Scaling Up and Out: MemoryX Weight Streaming](https://www.cerebras.ai/blog/scaling-up-and-out-training-massive-models-on-cerebras-systems-using-weight-streaming) — MemoryX scheduling: pipelining layer loads to hide weight transfer latency
- [Cerebras CS-3 — Blog Launch](https://www.cerebras.ai/blog/cerebras-cs3) — MemoryX + CS-3 integration overview; external memory bandwidth specs
- [Training Giant NNs Using Weight Streaming — Whitepaper](https://www.kisacoresearch.com/sites/default/files/documents/cs_weight_streaming_white_paper_-_cerebras.pdf) — Detailed MemoryX architecture: DRAM + flash tiers, prefetch pipeline, update scheduling

### Host Interface / Package
- [Cerebras AI Day Deck — SlideShare](https://www.slideshare.net/slideshow/cerebras-ai-day-deck-a-closer-look-at-the-worlds-fastest-ai-chip/266911791) — CS-3 system block diagram including host CPU, PCIe interface, cold plate packaging
- [Cerebras CS-3 — Futurum Analysis](https://futurumgroup.com/insights/cerebras-cs-3-bring-on-the-nvidia-blackwell-competition/) — CS-3 physical form factor: water-cooled appliance ~size of mini-fridge
- [100x Defect Tolerance: How Cerebras Solved the Yield Problem](https://www.cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem) — Custom packaging: PCB + connector + WSE + cold plate; TSMC 300mm custom packaging process
- [Cerebras Wafer-Scale Engine — WikiChip Fuse](https://fuse.wikichip.org/news/3010/a-look-at-cerebras-wafer-scale-engine-half-square-foot-silicon-chip/2/) — Physical package teardown analysis, scribe-line cross-die interconnect

### Scale-up Interconnect
- [Cerebras Swarm On-Chip Fabric — Architecture Deep Dive](https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning) — Swarm: 2D mesh, hardware router per PE, nanosecond latency, 100 Pb/s aggregate bandwidth, <1 pJ/bit
- [Cerebras Wafer Scale Engine at Hot Chips 33 — ServeTheHome](https://www.servethehome.com/cerebras-wafer-scale-engine-2-wse-2-at-hot-chips-33/) — WSE-2 Swarm fabric deep dive; single-word active messages, no software overhead
- [Cerebras WSE-3 — Introl Blog](https://introl.com/blog/cerebras-wafer-scale-engine-cs3-alternative-ai-architecture-guide-2025) — 2D mesh bandwidth analysis vs NVLink; Swarm fabric advantages for single-chip parallelism

### Scale-out Interconnect
- [Announcing Cerebras Architecture for Extreme-Scale AI](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai) — SwarmX: tree topology, broadcast weights / reduce gradients, 100GbE + RoCE RDMA, low-latency proprietary protocol
- [Cerebras Systems Announces Brain-Scale AI Solution](https://www.cerebras.ai/press-release/cerebras-systems-announces-worlds-first-brain-scale-artificial-intelligence-solution) — 192-CS-2 cluster; up to 163M cores; SwarmX enables near-linear data-parallel scale-out
- [Cerebras Wants Its Piece of HPC — NextPlatform](https://www.nextplatform.com/2022/11/16/cerebras-wants-its-piece-of-an-increasingly-heterogenous-hpc-world/) — SwarmX architecture details, MemoryX–SwarmX–CS-2 trifecta
- [AWS will bring Cerebras WSE-3 to cloud — SiliconANGLE](https://siliconangle.com/2026/03/13/aws-will-bring-cerebras-wafer-size-wse-3-chip-cloud-platform/) — CS-3 as cloud service on AWS Marketplace; scale-out via SwarmX in data centers
- [Cerebras Goes Hyperscale with CS-3 — NextPlatform](https://www.nextplatform.com/ai/2024/03/14/cerebras-goes-hyperscale-with-third-gen-waferscale-supercomputers/1642584) — Up to 2,048 CS-3 systems; quarter-zettaflops supercomputer configuration

---

## Other Resources

### Benchmarks & Performance
- [Cerebras CS-3 vs. Nvidia DGX B200 Blackwell — Cerebras Blog](https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-dgx-b200-blackwell) — Head-to-head training throughput comparison
- [Llama 3.1 405B at 969 tokens/s on Cerebras Inference](https://www.cerebras.ai/blog/llama-405b-inference) — Inference speed records; >1000 tokens/s for 405B model
- [Cerebras Launches World's Fastest Inference for Meta Llama 4](https://www.businesswire.com/news/home/20250409886691/en/Cerebras-Launches-Worlds-Fastest-Inference-for-Meta-Llama-4) — Llama 4 inference at 18x faster than GPU-based solutions
- [SambaNova vs Cerebras Inference Comparison](https://sambanova.ai/blog/sambanova-vs-cerebras) — Cross-vendor throughput and latency benchmarks
- [TPU vs GPU vs Cerebras vs Graphcore — Medium](https://khairy2011.medium.com/tpu-vs-gpu-vs-cerebras-vs-graphcore-a-fair-comparison-between-ml-hardware-3f5a19d89e38) — Multi-chip comparison across training workloads

### Academic Papers
- [A Comparison of Cerebras WSE with Nvidia GPU Systems — arXiv 2503.11698](https://arxiv.org/html/2503.11698v1) — March 2025 paper with detailed performance and efficiency analysis
- [Cerebras Architecture Deep Dive — IEEE Micro 2023](https://ieeexplore.ieee.org/document/10123162/) — S. Lie, IEEE Micro vol.43 no.3, May/Jun 2023
- [Inside the Cerebras Wafer-Scale Cluster — IEEE Micro 2024](https://dl.acm.org/doi/abs/10.1109/MM.2024.3386628) — Cluster-level architecture paper
- [Special Issue on Hot Chips 2023 — IEEE Micro](https://dl.acm.org/doi/abs/10.1109/MM.2024.3396008) — Special issue featuring Cerebras architecture content

### Inference Cloud & Ecosystem
- [Cerebras Inference](https://www.cerebras.ai/inference) — Official inference cloud service portal
- [Inference Documentation — Supported Models](https://inference-docs.cerebras.ai/models/overview) — Llama 3.1/3.3, Qwen 3, GPT-OSS-120B, ZAI GLM-4.7, others
- [GitHub — Cerebras/inference-examples](https://github.com/Cerebras/inference-examples) — OpenAI-compatible API examples for inference
- [AWS Marketplace: Cerebras Fast Inference Cloud](https://aws.amazon.com/marketplace/pp/prodview-ph4bdvplhhz3o) — Commercial availability on AWS
- [Meta Collaborates with Cerebras on Llama API](https://www.hpcwire.com/bigdatawire/this-just-in/meta-collaborates-with-cerebras-to-drive-fast-inference-for-developers-in-new-llama-api/) — Meta Llama API partnership for fast inference

### Defect Tolerance & Manufacturing
- [100x Defect Tolerance: How Cerebras Solved the Yield Problem](https://www.cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem) — Core redundancy (1–1.5% extra), dynamic fabric routing around defects, 0.05mm² core size
- [Wafer-Scale Processors: The Time Has Come — Medium](https://medium.com/@cerebras/wafer-scale-processors-the-time-has-come-463045893538) — Manufacturing feasibility analysis
- [Cerebras Wikipedia](https://en.wikipedia.org/wiki/Cerebras) — Company overview, WSE generation history, funding timeline
- [Cerebras Whitepapers Portal](https://www.cerebras.ai/whitepapers) — Collection of all official technical whitepapers

### GitHub Organization
- [Cerebras · GitHub](https://github.com/cerebras) — Main org: 21+ repositories
- [CerebrasResearch · GitHub](https://github.com/CerebrasResearch) — Research org: 19+ repositories  
- [GitHub — Cerebras/modelzoo](https://github.com/Cerebras/modelzoo) — Model Zoo: LLM + vision reference implementations optimized for WSE
- [GitHub — Cerebras/sdk-examples](https://github.com/Cerebras/sdk-examples) — CSL kernel examples
- [GitHub — Cerebras/inference-examples](https://github.com/Cerebras/inference-examples) — Inference API examples
- [GitHub — Cerebras/DocChat](https://github.com/Cerebras/DocChat) — GPT-4 level conversational QA trained on Cerebras

---

## Resources Added 2026-08-08 (roadmap scan, window 2026-04-01 → 2026-08-08)

*Verified during the 2026-08-08 adversarial-verification pass. Scan caveat: the WebSearch budget was exhausted, so verification used direct primary-source fetches plus DuckDuckGo HTML/lite result pages; CNBC direct, SEC EDGAR, HPCwire, DataCenterDynamics and the Cerebras IR PDF returned 403/429/timeout, so a few corporate figures rest on search-result snippets rather than full-article reads.*

### Disaggregated Inference — AMD Helios prefill + WSE decode (announced 2026-07-23)
- [AMD Newsroom — AMD and Cerebras ultra-low-latency, high-throughput AI inference](https://newsroom.amd.com/news/aai-2026-cerebras-inference/) — **Primary, independent of Cerebras.** Announced at AMD Advancing AI 2026. Helios rack-scale (Instinct) = prefill/long-context; Cerebras WSE = decode. Carries the footnote that qualifies the "up to 5× tokens/s/W" claim as July 2026 modelling by AMD Performance Labs and Cerebras, measured in TPS/kW on Kimi 2.6 1T against a **Cerebras-WSE-only** baseline. Availability: initially via Cerebras Cloud in H2 2026
- [Cerebras Press Release — AMD and Cerebras announce ultra-low-latency and high-throughput AI inference](https://www.cerebras.ai/press-release/amd-and-cerebras-announce-industry-leading-ultra-low-latency-and-high-throughput-ai-inference) — Cerebras side of the same 2026-07-23 announcement
- [DataCenterDynamics — AMD partners with Cerebras for inference system](https://www.datacenterdynamics.com/en/news/amd-partners-with-big-chip-co-cerebras-for-ultra-low-latency-and-high-throughput-ai-inference-system/) — Trade coverage (direct fetch returned an error during this scan)

### Scheduled Disclosure — Hot Chips 38
- [Hot Chips 38 Advance Program](https://hotchips.org/advance-program/) — "The Cerebras Rack-Scale Architecture for Wafer Scale Engine", Jean-Philippe (J.P.) Fricker, Cerebras; Session AI 1, Tuesday **2026-08-25, 2:15–4:15 PM PDT**, Stanford Memorial Auditorium. **Disclosure scheduled — content not yet public as of 2026-08-08.** Not usable as a source for any specification; re-fetch for slides after 2026-08-25

### Corporate — IPO (2026-05-13 priced / 2026-05-14 first trade)
- [Reuters — Cerebras prices IPO at $185 per share to raise $5.55 billion](https://www.reuters.com/legal/government/cerebras-prices-ipo-185-per-share-raise-555-billion-sources-say-2026-05-13/) — 30M Class A shares, ~$56.4B fully diluted at the offer price
- [Reuters — Cerebras set to debut in a stock market gripped by AI mania](https://www.reuters.com/legal/transactional/cerebras-set-debut-stock-market-gripped-by-ai-mania-2026-05-14/) — Day one: opened $350, high ~$385, closed $311.07 (+68%), ~$106.75B fully diluted
- [CNBC — Cerebras (CBRS) stock begins trading on Nasdaq](https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html) — Confirms exchange (Nasdaq) and ticker (CBRS)
- [Nasdaq Private Market — Cerebras](https://www.nasdaqprivatemarket.com/company/cerebras/) — Exchange/ticker corroboration
- [Built In — Cerebras launches IPO](https://builtin.com/articles/cerebras-launches-ipo-20260515) — Corroborates Nasdaq/CBRS
- [Yahoo Finance — Cerebras stock slides after near-70% surge in biggest IPO of 2026](https://finance.yahoo.com/markets/article/cerebras-stock-slides-after-near-70-surge-in-biggest-ipo-of-2026-130757084.html) — Day-two follow-through
- [Nasdaq — Cerebras IPO ushering in a new era of AI hardware](https://www.nasdaq.com/newsroom/cerebras-ipo-ushering-new-era-ai-hardware) — Context piece
- ⚠️ A "$9.5 billion IPO" figure circulating in a headline slug is **wrong by ~6×** against the $56.4B offer-price valuation and does not match the $5.55B raised. Do not use it.

### Corporate — OpenAI compute agreement (2026-01-14, pre-baseline gap)
- [Reuters — OpenAI to buy compute capacity from Cerebras in deal around $10 billion](https://www.reuters.com/technology/openai-buy-compute-capacity-startup-cerebras-around-10-billion-wsj-reports-2026-01-14/) — Up to **750 MW over three years through 2028**; a purchase commitment, not deployed capacity
- [CNBC — Cerebras scores OpenAI deal worth over $10 billion](https://www.cnbc.com/2026/01/14/cerebras-scores-openai-deal-worth-over-10-billion.html)
- [TechCrunch — OpenAI signs deal reportedly worth $10 billion for compute from Cerebras](https://techcrunch.com/2026/01/14/openai-signs-deal-reportedly-worth-10-billion-for-compute-from-cerebras/)
- [WSJ — OpenAI forges multibillion-dollar computing partnership with Cerebras](https://www.wsj.com/tech/ai/openai-forges-multibillion-dollar-computing-partnership-with-cerebras-746a20e4) — Sourced to "people familiar with the matter"
- ⚠️ A "$25B OpenAI backlog" figure from one aggregator around the Q1 FY2026 report could **not** be confirmed against the primary release and conflicts in magnitude with the January $10B figure. Excluded.

### Manufacturing, Capacity and Financials
- [Flex Investor Relations — Flex and Cerebras expand partnership to scale American manufacturing of Cerebras AI supercomputers](https://investors.flex.com/news/news-details/2026/Flex-and-Cerebras-Expand-Partnership-to-Scale-American-Manufacturing-of-Cerebras-AI-Supercomputers/default.aspx) — **2026-07-09**; new Milpitas, California lines; anticipated **~7× increase in CS-3 production capacity through 2026**
- [Engineering.com — Flex and Cerebras expand partnership](https://www.engineering.com/flex-and-cerebras-expand-partnership/) — Corroborating trade coverage
- [Cerebras IR — Cerebras Systems accelerates European expansion, 200MW AI compute](https://investors.cerebras.ai/news-releases/news-release-details/cerebras-systems-accelerates-european-expansion-200mw-ai-compute) — **2026-07-09**, RAISE Summit Paris; **200 MW in Europe by end of 2027** target, first capacity end-2026, France and the Nordics. A target, not built capacity
- [Cerebras IR — Q1 2026 results](https://investors.cerebras.ai/news-releases/news-release-details/cerebras-systems-announces-strong-first-quarter-2026-results) — **2026-06-23**; GAAP revenue $193.4M, record core revenue $191.3M (+92% YoY). (IR PDF timed out during this scan; figures from the listing/snippet)
- [Cerebras Company News index](https://www.cerebras.ai/company/news) — Rolling index used to seed this scan
- [Cerebras — Flex partnership page](https://www.cerebras.ai/flex) — Vendor page on the manufacturing partnership (undated on-page)

### Benchmarks
- [HPCwire — MLCommons releases MLPerf Training v6.0 results](https://www.hpcwire.com/off-the-wire/mlcommons-releases-mlperf-training-v6-0-results/) — Results released ~2026-06-16. **No Cerebras submission appears in retrieved coverage**, but no official MLCommons results table could be fetched, so the repo's "not in MLPerf" limitation is retained as *last verified 2026-04-05, pending re-check*

### Product / Blog items seen but NOT independently verified in this pass
*Leads for the next scan only — not written into `chips/cerebras/` as confirmed.*
- Multi-LoRA on Cerebras Inference (2026-05-06) — highest-priority lead; would be a Runtime-layer capability if confirmed
- Trillion-parameter inference with Kimi K2.6 for enterprises (2026-05-19)
- Gemma 4 multimodal inference (2026-06-29)
- Upstage partnership for South Korea (2026-07-10)
- Sovereign-AI positioning post (2026-05-26)
- Rest of World piece on an India–UAE AI partnership involving G42 and Cerebras (2026-06-01)
- Index: [Cerebras Blog](https://www.cerebras.ai/blog)

### Source-hygiene note
- [Announcing the Cerebras Architecture for Extreme-Scale AI](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai) — **Dated 2021-08-24.** Covers WSE-2 / CS-2 weight streaming, MemoryX and SwarmX. Valid support for the weight-streaming architecture; **not** evidence for any 2026 rack-scale product
