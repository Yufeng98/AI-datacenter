# FuriosaAI NPU Layer Mapping Table

*as_of: 2026-08-08*
*Chips: Warboy (Samsung 14nm), RNGD (TSMC 5nm), Gen 3 (Broadcom co-development — announced 2026-05-27, no silicon)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (via torch.onnx.export / torch.export → ONNX → furiosa-compiler; no eager backend; AOT-only) | confirmed | software-stack, search-results |
| Framework Integration | ONNX (primary model format; furiosa-compiler accepts directly; all model entry points normalize to ONNX) | confirmed | software-stack, search-results |
| Framework Integration | HuggingFace Transformers (via furiosa-llm; safetensors → RNGD binary; LLaMA/Mistral/EXAONE/Gemma) | confirmed | software-stack, search-results |
| Framework Integration | TensorFlow / CoreML (via ONNX export as intermediate) | inferred | software-stack |
| Framework Integration | furiosa-models (open-source model zoo; ResNet50, EfficientNetB0, YOLOv5/v8, SSD, RetinaFace; pre/post-processing) | confirmed | search-results |
| Compiler / IR | furiosa-compiler (AOT; ONNX → .enf/RNGD binary; proprietary binary dist.; passes: const fold, op fusion, quant lowering, memory planning, binary gen) | confirmed | software-stack, search-results |
| Compiler / IR | Operator fusion (Conv+BN+ReLU, GEMM+bias, LayerNorm, attention QKV fusion in RNGD) | confirmed | software-stack |
| Compiler / IR | Tactic selection (multiple kernel variants per op; compiler selects optimal for tile size + quantization) | confirmed | software-stack |
| Compiler / IR | Tensor contraction mapping (RNGD-specific; assigns tensor index dims to PE/slice axes on 8×64 TCP grid) | confirmed | hw-architecture, search-results |
| Compiler / IR | LLM-specific passes (RNGD: KV-cache allocation, attention fusion, RoPE embedding fusion) | confirmed | software-stack, search-results |
| Compiler / IR | CPU fallback (unsupported ops execute on host CPU with auto data marshaling) | confirmed | software-stack |
| Compiler / IR | .enf binary format (Warboy; Engineered NPU Format; proprietary) | confirmed | software-stack |
| Compiler / IR | RNGD binary (RNGD; format not publicly named; contains compiled TCP schedule) | confirmed | search-results |
| Compiler / IR | CLI tools: furiosa compile, furiosa quantize, furiosa perftest, furiosa profile | confirmed | software-stack |
| Compiler / IR | **FXB (Furiosa Executable Bundle)** — portable compiled artifact enabling zero-recompilation reuse across compatible model variants (SDK 2026.3) | reported (vendor blog) | software-stack (2026-08-08) |
| Compiler / IR | SDK distribution: 2026.x ships as **PyPI wheels** under a calendar-version scheme via developer.furiosa.ai; `github.com/furiosa-ai/furiosa-sdk` is stale at 0.9.2 (June 2024, Warboy-era) and is **not** the current release channel | confirmed | software-stack (2026-08-08) |
| Op Library | `not applicable` — no separate op dispatch library; all fusion internal to furiosa-compiler | confirmed | software-stack |
| Kernel Library | **TCL — Tensor Contraction Language** (SDK 2026.3, 2026-06-30): declarative Python eDSL; `@tcl.kernel` functions state *what* to compute, compiler owns tiling/scheduling/fusion/hw mapping; tensor contraction is a first-class language primitive; padding, sharding, multi-chip collectives expressed at language level | confirmed | software-stack (2026-08-08) |
| Kernel Library | `furiosa-kernels` package — reusable TCL blocks (RMSNorm, Linear, MLP); PyPI 2026.3.0 uploaded 2026-06-30; **license `LicenseRef-Proprietary`** (NOT Apache); Python 3.10+ | confirmed | software-stack (2026-08-08), PyPI |
| Kernel Library | Public TCL/kernel-authoring **documentation does not exist** in Developer Center 2026.3.0; general availability vs partner-gating **not disclosed** | confirmed (negative) | software-stack (2026-08-08) |
| Kernel Library | ~~`not applicable` — no user-facing kernel library (no CUTLASS equivalent); compiler owns all placement~~ — **SUPERSEDED 2026-08-08 by TCL** | superseded | software-stack |
| Quantization | furiosa-quantizer (INT8 dynamic/static; INT4 weight-only RNGD; FP8 per-tensor/per-channel RNGD; GPTQ-style block quant for LLMs) | confirmed | software-stack, search-results |
| Runtime | furiosa-runtime / Warboy (sync + async session API; PCIe DMA for I/O; multi-session multiplexing) | confirmed | software-stack, search-results |
| Runtime | furiosa-runtime / RNGD 2026.1.0 (NPU partition selection 2/4/8 vNPUs; streaming token gen; KV-cache management; continuous batching) | confirmed | software-stack, search-results |
| Runtime | furiosa-llm (RNGD; LLM(), SamplingParams(); LLaMA/Mistral/EXAONE/Gemma; HuggingFace-compatible API) | confirmed | software-stack, search-results |
| Runtime | furiosa-llm 2026.3.0 (PyPI, Apache 2.0, 2026-06-30) — models unlocked by TCL: Qwen3-VL (incl. 32B), gpt-oss-120b, Solar-Open-100B, Qwen3-30B-A3B (MoE), K-EXAONE-236B-A23B | confirmed | software-stack (2026-08-08), PyPI |
| Runtime | SDK 2026.3 scheduling: overlap scheduling to reduce NPU idle time; scoring-based data-parallel request routing (prefix locality + token footprint); multimodal serving with chunked prefill; breaking changes upgrading from 2026.2 | reported (vendor blog) | software-stack (2026-08-08) |
| Serving | furiosa-serving (OpenAI Chat Completions /v1/chat/completions; Triton-compatible REST; health + metrics) | confirmed | software-stack |
| Serving | HuggingFace TGI backend plugin (RNGD; Docker-based; drop-in TGI deployment) | confirmed | search-results |
| Driver / Firmware | furiosa-npu.ko Linux kernel module (PCIe BAR MMIO; DMA channel mgmt; partition management; apt install furiosa-driver) | confirmed | software-stack |
| Driver / Firmware | Minimal on-chip microcontroller (DVFS power mgmt; thermal monitoring; PCIe link mgmt; does NOT manage op scheduling) | confirmed | software-stack |
| Communication | Warboy / RNGD: no proprietary scale-up fabric — multi-card via host PCIe (+ card-to-card P2P) and furiosa-llm tensor parallelism | confirmed | hw-architecture |
| Communication | Multi-chip collectives expressible at TCL language level (SDK 2026.3) | confirmed | software-stack (2026-08-08) |
| Communication | Gen 3 (announced): Broadcom **Ethernet scale-up + fabric switches**, all-to-all-capable topology for MoE expert routing; bandwidth and domain size **not disclosed** | announced only | hw-architecture (2026-08-08) |
| Communication | Standard Ethernet for scale-out | confirmed | hw-architecture |
| Assembler / ISA | **FVISA ("Furiosa Virtual ISA")** named publicly as a stack layer alongside the compiler and TCL at RENEGADE Summit 2026 (2026-05-13); **no technical detail, no documentation, no user-facing tooling disclosed** | named only | software-stack (2026-08-08) |
| Assembler / ISA | No user-visible virtual ISA surface for programmers (no PTX equivalent); .enf / RNGD binary format is proprietary | confirmed | software-stack |
| Profiler | furiosa-profiler (Chrome Tracing JSON output; per-op NPU time; DMA transfers; CPU-NPU sync points) | confirmed | software-stack |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Warboy: PE array (Samsung 14nm, 180 mm², 5B transistors, 2.0 GHz, 64 TOPS INT8, vision-first) | confirmed | hw-architecture, search-results |
| Compute Engine | RNGD: Tensor Contraction Processor (TSMC 5nm, 40B transistors, 1.0 GHz, **512 TFLOPS FP8** = 64 TFLOPS FP8 x 8 PEs) | confirmed (vendor product page) | hw-architecture (2026-08-08) |
| Compute Engine | RNGD: 256 TFLOPS BF16, 512 TOPS INT8, 1,024 TOPS INT4 — developer-docs figures; the 512 headline is FP8 TFLOPS, not INT8 TOPS (prior repo baseline mislabelled it) | reported, not independently confirmed | hw-architecture (2026-08-08) |
| Compute Engine | Gen 3 (announced 2026-05-27, Broadcom): TCP + Broadcom XPU Technology and IP Platform, 2nm compute die (TSMC per Korean wire), multi-die chiplet SiP; **peak throughput at every dtype not disclosed**; sampling targeted H1 2028; no tape-out, no silicon | announced only | hw-architecture (2026-08-08) |
| Compute Engine | RNGD TCP grid: 8 PEs × 64 slices = 512 total slices | confirmed | hw-architecture, search-results |
| Compute Engine | TCP key insight: tensor contraction = generalization of matmul; handles GEMM + attention + conv via single hardware path | confirmed | hw-architecture, search-results |
| Compute Engine | No warp scheduler, no branch predictor, no dynamic hardware scheduler — all determined at compile time | confirmed | hw-architecture |
| Data Path | Compiler-managed execution — all data movement scheduled AOT; no cache miss stalls | confirmed | hw-architecture |
| On-chip Memory | Warboy: 32 MB SRAM (SW-managed, no caches; compiler allocates activation + weight tile buffers) | confirmed | hw-architecture, search-results |
| On-chip Memory | RNGD: **256 MB SRAM @ 384 TB/s** (SW-managed, no caches) — corrects the prior "not disclosed" baseline | confirmed (vendor product page) | hw-architecture (2026-08-08) |
| On-chip Memory | Gen 3: on-chip SRAM capacity and bandwidth **not disclosed** | not disclosed | hw-architecture (2026-08-08) |
| Off-chip Memory | Warboy: 16 GB LPDDR4X @ 66 GB/s | confirmed | hw-architecture, search-results |
| Off-chip Memory | RNGD: **48 GB HBM3** (2 stacks via CoWoS-S @ 6.0 Gbps) @ 1.5 TB/s (23x Warboy BW); SK hynix HBM3 — corrects the prior 24 GB baseline (wrong by 2x) | confirmed (vendor product page) | hw-architecture (2026-08-08) |
| Off-chip Memory | Gen 3: **HBM4 / HBM4E**; capacity, stack count and bandwidth **not disclosed** | announced only | hw-architecture (2026-08-08) |
| Multi-tenancy | RNGD: hardware-isolated SR-IOV partitioning into 2/4/8 virtual NPUs (dedicated PE slices + HBM3 per partition: 48/24/12/6 GB, recomputed on the corrected 48 GB capacity) | confirmed | hw-architecture (2026-08-08), search-results |
| Multi-tenancy | Partition modes sold commercially: Samsung SDS NPUaaS on Samsung Cloud Platform offers RNGD in 1/2/4/8-card configurations (launched 2026-07-20, Dongtan DC) | confirmed | search-results (2026-08-08) |
| Host Interface / Package | Warboy: PCIe Gen4 x8 card, Samsung 14nm (via SemiFive ASIC platform), ~50W | confirmed | hw-architecture, search-results |
| Host Interface / Package | RNGD: PCIe Gen5 x16 card (with card-to-card P2P), TSMC 5nm; **180 W** per product page and all 2026 press vs **150 W** in developer docs (unreconciled vendor discrepancy); mass production January 2026, formally declared 2026-05-13 | confirmed | hw-architecture (2026-08-08), search-results |
| Host Interface / Package | Gen 3: multi-die chiplet system-in-package (Broadcom advanced packaging); form factor, host interface and TDP **not disclosed** | announced only | hw-architecture (2026-08-08) |
| Host Interface / Package | Foundry shift: Warboy Samsung 14nm → RNGD TSMC 5nm (better density + HBM3 interposer ecosystem) | confirmed | hw-architecture, search-results |
| Scale-up Interconnect | Warboy / RNGD: no proprietary chip-to-chip fabric; multi-card via host PCIe + PCIe P2P | confirmed | hw-architecture |
| Scale-up Interconnect | NXT RNGD server: originally 4 cards @ 3 kW total (3.75x tokens/watt vs H100 GPU rack); **as of 2026-07-07 described as up to 8 RNGD accelerators in a 3 kW-class system** | confirmed | search-results (2026-08-08) |
| Scale-up Interconnect | **Gen 3: first scale-up fabric in the FuriosaAI roadmap** — Broadcom Ethernet scale-up + fabric switches, all-to-all-capable topology for MoE expert routing; link BW, domain size (chips/nodes/rack) **not disclosed** | announced only | hw-architecture (2026-08-08) |
| Scale-out Interconnect | Standard Ethernet (no proprietary fabric) | confirmed | hw-architecture |
| Power Efficiency | RNGD 180W per card; NXT RNGD server 3 kW vs DGX H100 >10 kW for comparable LLM workload | confirmed | search-results |
| Power Efficiency | Backend.AI white paper (2026-08-06): vs 4x RTX PRO 6000 Blackwell SE on Qwen3-32B FP8 — 1.3x-1.5x throughput/watt, 95% of GPU peak throughput @256 concurrency, 30-44% lower power, TTFT <1 s through concurrency 32 vs >2.9 s | vendor claim (not independent) | search-results (2026-08-08) |
| Power Efficiency | **No MLPerf results exist for RNGD** — FuriosaAI is not among the 24 submitters to MLPerf Inference v6.0 (2026-04-01) | confirmed (negative) | search-results (2026-08-08) |
| Deployment | Sweden (2026-08-04, I/ONX HPC + Velox, Stockholm): 15 MW site; phase 1 = 1,800 RNGD @ 2 MW online early 2027; +8 MW later 2027; 7,000+ additional accelerators planned | confirmed | search-results (2026-08-08) |
| Deployment | Europe (2026-07-07): RNGD servers at Equinix Lisbon LS2 for enterprise evaluation; European flagship office in Portugal (2026-04-10) | confirmed | search-results (2026-08-08) |
