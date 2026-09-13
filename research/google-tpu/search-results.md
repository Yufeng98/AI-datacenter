# Google TPU Software Stack & Hardware Resources

*as_of: 2026-08-08*
*device_class: Systolic Array Accelerator*
*seeds: https://cloud.google.com/tpu/docs, https://github.com/google/jax*

---

## Software Stack

### Framework Integration

- [JAX](https://github.com/google/jax) — Primary numerical computing framework for TPU; functional transformations (jit, grad, vmap, pmap) over XLA compilation
- [JAX Documentation](https://docs.jax.dev/en/latest/) — Official JAX docs including TPU-specific guides and Pallas kernel language reference
- [PyTorch/XLA](https://github.com/pytorch/xla) — Python package bridging PyTorch to XLA devices including Cloud TPUs; enables torch.compile → XLA path
- [PyTorch/XLA Documentation](https://docs.pytorch.org/xla/master/learn/xla-overview.html) — Overview of PyTorch/XLA integration and SPMD parallelism
- [TorchTPU Project](https://hyperframeresearch.com/2025/12/24/can-googles-torchtpu-eventually-bridge-nvidias-cuda-moat/) — Google/Meta initiative to make TPUs feel native to PyTorch, targeting CUDA software parity
- [TensorFlow on Cloud TPU](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) — TensorFlow TPU support (note: TF not supported on Ironwood/TPU v7+)
- [vLLM TPU Backend](https://blog.vllm.ai/2025/10/16/vllm-tpu.html) — Unified JAX+PyTorch inference backend for TPU; 2–5x performance gains vs earlier 2025 prototypes
- [Cloud TPU Release Notes](https://docs.cloud.google.com/tpu/docs/release-notes) — Versioned changelog for TPU software runtime and framework support
- [TPU Software Versions](https://docs.cloud.google.com/tpu/docs/runtimes) — Matrix of libtpu, JAX, PyTorch, and TF runtime version compatibility

### Compiler / IR

- [OpenXLA/XLA](https://github.com/openxla/xla) — Open-source XLA compiler (Accelerated Linear Algebra); primary backend for JAX and PyTorch/XLA; targets TPU/GPU/CPU
- [OpenXLA Project](https://openxla.org/) — Community-driven ML compiler ecosystem homepage; umbrella for XLA, StableHLO, Shardy, IREE
- [StableHLO](https://github.com/openxla/stablehlo) — Backward-compatible MLIR operation set defining the portability layer between ML frameworks and XLA-based compilers
- [StableHLO Specification](https://openxla.org/stablehlo/spec) — Formal spec for the StableHLO opset used as XLA's stable IR
- [Shardy](https://openxla.org/shardy/overview) — MLIR-based tensor partitioning system; successor to GSPMD combining its propagation with PartIR's incremental partitioning
- [Shardy LLVM Dev Talk (slides)](https://llvm.org/devmtg/2024-10/slides/techtalk/Chrzaszcz-Jiang-Shardy.pdf) — Design and motivation slides from LLVM 2024 Dev Meeting
- [GSPMD Paper](https://arxiv.org/abs/2105.04663) — "General and Scalable Parallelization for ML Computation Graphs"; foundational sharding compiler technique in XLA
- [OpenXLA Blog Announcement](https://opensource.googleblog.com/2023/03/openxla-is-ready-to-accelerate-and-simplify-ml-development.html) — Google Open Source Blog post announcing OpenXLA availability

### Op Library

- [Flax](https://github.com/google/flax) — Neural network library for JAX; flexible object-oriented model authoring API
- [Optax](https://github.com/google-deepmind/optax) — Composable gradient processing and optimization library for JAX
- [Orbax](https://github.com/google/orbax) — Checkpointing and model persistence utilities in the JAX ecosystem
- [Grain](https://github.com/google/grain) — Deterministic data pipelines for large-scale JAX training
- [Keras TPU Support](https://keras.io/guides/define_custom_kernel/) — Keras guide for defining custom TPU/GPU kernels via Pallas

### Kernel Library

- [Pallas (JAX Kernel Language)](https://docs.jax.dev/en/latest/pallas/index.html) — JAX extension for writing custom kernels targeting TPU (via Mosaic) and GPU (via Triton)
- [Pallas Design Doc](https://docs.jax.dev/en/latest/pallas/design/design.html) — Architecture and lowering pipeline for Pallas kernels
- [Pallas PyTorch/XLA Integration](https://docs.pytorch.org/xla/master/features/pallas.html) — Custom Pallas kernels callable from PyTorch/XLA
- [Tokamax](https://docs.cloud.google.com/tpu/docs/jax-ai-stack) — Google's curated library of state-of-the-art high-performance kernels built on JAX/Pallas for TPU and GPU
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) — High-performance, scalable open-source LLM reference implementation in JAX targeting Cloud TPU/GPU (includes Gemma, Llama, DeepSeek, Qwen, Mistral)
- [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) — Reference implementations of latent diffusion models in JAX for TPU/GPU (MaxText for vision)

### Runtime

- [libtpu (PyPI)](https://pypi.org/project/libtpu/) — Shared library on every Cloud TPU VM; contains the XLA compiler, TPU driver, and ICI communication logic
- [libtpu SDK Docs](https://docs.cloud.google.com/tpu/docs/runtimes) — Documentation for libtpu runtime versions and SDK primitives
- [TPU Monitoring Library](https://docs.cloud.google.com/tpu/docs/tpu-monitoring-library) — Library for collecting TPU performance and health metrics
- [Cloud TPU Performance Guide](https://docs.cloud.google.com/tpu/docs/performance-guide) — Official guide to performance tuning, op fusion, and memory management

### Driver / Firmware

- [libtpu (driver component)](https://pypi.org/project/libtpu/) — libtpu.so bundles the TPU kernel driver alongside the XLA compiler in a single deployable library
- [libedgetpu (Coral/Edge TPU)](https://github.com/google-coral/libedgetpu) — Userspace runtime driver source for Google Coral Edge TPU devices (separate from Cloud TPU)
- [TPU VM System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — Overview of TPU VM host-to-device interface, driver model, and chip topology

### Communication

- [GSPMD Paper](https://arxiv.org/abs/2105.04663) — Describes AllGather, AllToAll, CollectivePermute, DynamicSlice collectives used by XLA SPMD partitioner
- [Shardy](https://openxla.org/shardy/overview) — Next-gen MLIR partitioning replacing GSPMD; handles sharding propagation and collective insertion
- [PyTorch/XLA SPMD](https://pytorch.org/blog/pytorch-xla-spmd/) — Blog post on PyTorch/XLA SPMD support for data parallelism, FSDP, and tensor parallelism on TPU
- [Multislice Training](https://docs.cloud.google.com/tpu/docs/v5e-training) — Documentation for multi-pod training across ICI+DCN boundaries with hierarchical collectives

### Assembler / ISA

- [ACM: In-Datacenter Performance Analysis of a Tensor Processing Unit](https://dl.acm.org/doi/fullHtml/10.1145/3360307) — Describes TPU v1's CISC ISA (20 instructions) and later VLIW instruction packets (8 operations/cycle)
- [The Chip Letter: Google's First TPU Architecture](https://thechipletter.substack.com/p/googles-first-tpu-architecture) — In-depth analysis of TPU v1 ISA and microarchitecture
- [First In-Depth Look at Google's TPU Architecture (NextPlatform)](https://www.nextplatform.com/2017/04/05/first-depth-look-googles-tpu-architecture/) — Early architectural analysis covering CISC→VLIW ISA evolution
- [Mosaic Compiler Backend](https://docs.jax.dev/en/latest/pallas/design/design.html) — Pallas lowers custom kernels to Mosaic IR for TPU; Mosaic exposes systolic array, VMEM, and tiling features before VLIW code emission

---

## Hardware Architecture

### Compute Engine

- [TPU Architecture Overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — Official documentation of TensorCore, MXU (256×256 systolic array), VPU, and ScalarCore units
- [TPU v4 Whitepaper (arXiv)](https://arxiv.org/abs/2304.01433) — "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings" (ISCA 2023)
- [Ironwood TPU v7 Docs](https://docs.cloud.google.com/tpu/docs/tpu7x) — Specs: 2 TensorCores + 4 SparseCores per chip; 192 GB HBM3E; 42.5 ExaFLOPS per 9,216-chip pod
- [Ironwood Announcement Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) — Google blog on TPU v7 design goals for the inference era
- [Inside the Ironwood Codesigned AI Stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack) — Deep dive into Ironwood chip-software co-design and MXU/SparseCore integration
- [SemiAnalysis: TPUv7 Analysis](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the) — Independent architectural analysis of Ironwood performance and design choices
- [Google TPU Architecture: 7 Generations Explained](https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations) — Generation-by-generation comparison of MXU size, clock speed, and performance
- [IntuitionLabs: Google TPUs for Gemini 3](https://intuitionlabs.ai/articles/google-tpu-architecture-gemini-3) — TPU architecture analysis in context of Gemini 3 training

### Data Path

- [TPU Architecture Overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — Data flow from host PCIe → HBM → VMEM → MXU systolic pipeline → VMEM → HBM
- [TPU Deep Dive](https://henryhmko.github.io/posts/tpu/tpu.html) — Community deep dive covering MXU systolic data flow and weight-stationary operation
- [SRAM as the New Compute Fabric (Substack)](https://tspasemiconductor.substack.com/p/sram-as-the-new-compute-fabric-a) — Comparative architecture study of TPU vs Groq LPU vs Cerebras WSE, focusing on SRAM data paths
- [How Google's TPU Works (ByteByteGo)](https://blog.bytebytego.com/p/how-googles-tensor-processing-unit) — Accessible explainer of the systolic array data movement and weight-stationary design

### On-chip Memory

- [TPU Architecture Overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — Documents Unified Buffer (24 MB SRAM on early TPUs), VMEM (per-TensorCore scratchpad), and CMEM (shared pool, v4+)
- [Pallas Memory Hierarchy Guide](https://docs.jax.dev/en/latest/pallas/index.html) — Explains VMEM as software-managed scratchpad and how Pallas kernels tile HBM→VMEM→MXU
- [XLA Memory Management](https://docs.cloud.google.com/tpu/docs/performance-guide) — XLA compiler scheduling of HBM↔VMEM data movement and operator fusion for memory efficiency

### Off-chip Memory

- [TPU v6e Docs](https://docs.cloud.google.com/tpu/docs/v6e) — v6e (Trillium): 144 GB HBM3 per chip; 2× HBM capacity and bandwidth vs v5e
- [TPU v5p Docs](https://docs.cloud.google.com/tpu/docs/v5p) — v5p: 95 GB HBM per chip; 4,800 Gbps/chip ICI in 3D torus
- [Ironwood Memory Specs](https://docs.cloud.google.com/tpu/docs/tpu7x) — v7 (Ironwood): 192 GB HBM3E per chip (8 stacks); 7.4 TB/s peak HBM bandwidth; 1.77 PB total per 9,216-chip pod
- [Tensor Processing Unit Wikipedia](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) — Historical HBM/DRAM specs across all TPU generations in tabular form

### Host Interface / Package

- [TPU VM System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — Host-to-TPU communication model, PCIe interface, and TPU VM architecture
- [libtpu Runtime](https://pypi.org/project/libtpu/) — Host-side library managing device communication, compilation dispatch, and ICI coordination

### Scale-up Interconnect

- [TPU v5p ICI Docs](https://docs.cloud.google.com/tpu/docs/v5p) — 3D torus ICI at 4,800 Gbps/chip; 8,960 chips per pod
- [TPU v4 Whitepaper (arXiv)](https://arxiv.org/abs/2304.01433) — Describes ICI topology and Optical Circuit Switch (OCS) reconfigurability for v4 supercomputer
- [Ironwood ICI Scale-up](https://docs.cloud.google.com/tpu/docs/tpu7x) — TPU v7: 4×4×4 3D torus cube (64 chips/rack); copper DAC intra-cube, optical inter-cube; scales to 9,216 chips
- [OCS Architecture (FiberMall)](https://www.fibermall.com/blog/unveiling-google-tpu-architecture.htm) — Detailed explainer of MEMS-based optical circuit switches, beam steering, and 40% power reduction vs electrical switches
- [Custom Optical Networking (NextBigFuture)](https://www.nextbigfuture.com/2025/11/highly-customized-optical-networking-critical-for-googles-tensor-processing-units-tpus.html) — Analysis of Google's bespoke optical networking for TPU pods

### Scale-out Interconnect

- [Multislice Training](https://docs.cloud.google.com/tpu/docs/v5e-training) — Multi-pod scale-out via DCN (datacenter network) using Titanium IPUs for network offload; supports 18,432+ chips
- [Introducing Trillium TPUs](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus) — Trillium scale-out: 256-chip high-BW pod, then hundreds of pods via multi-petabit/s DCN
- [AI Hypercomputer Overview](https://cloud.google.com/blog/products/compute/updates-to-ai-hypercomputer-software-stack/) — AI Hypercomputer infrastructure combining TPU pods, Titanium IPUs, and GCS storage for system-scale training

---

## Other Resources

- [Cloud TPU Documentation Home](https://docs.cloud.google.com/tpu/docs) — Official entry point for all Cloud TPU docs: getting started, versions, frameworks, networking
- [TPU v4 Docs](https://docs.cloud.google.com/tpu/docs/v4) — v4 specifications, pod topology, and SparseCore for embeddings
- [TPU v6e (Trillium) Docs](https://docs.cloud.google.com/tpu/docs/v6e) — GA Trillium specs: 4.7× compute vs v5e, 3rd-gen SparseCore, 256-chip pods
- [Trillium GA Blog](https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga) — General availability announcement with performance benchmarks vs v5e and v5p
- [XProf Profiler](https://github.com/openxla/xprof) — OpenXLA profiling tool with Trace Viewer, Memory Profile, Graph Viewer, and LLO bundle visualization
- [XProf Advanced Features Blog](https://opensource.googleblog.com/2026/03/advanced-tpu-optimization-with-xprof-continuous-profiling-utilization-insights-and-llo-bundles.html) — 2026 blog on continuous profiling snapshots, utilization viewer, and instruction-level LLO insights
- [Profile TPU VMs Guide](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) — Step-by-step profiling with XProf on Cloud TPU VMs
- [Debugging JAX on Cloud TPUs](https://developers.googleblog.com/a-developers-guide-to-debugging-jax-on-cloud-tpus-essential-tools-and-techniques/) — Developer guide covering debugging tools and TPU-specific error patterns
- [How to Scale Your Model (JAX Scaling Book)](https://jax-ml.github.io/scaling-book/profiling/) — Community book on profiling and scaling JAX models on TPU
- [tpu-starter](https://github.com/ayaka14732/tpu-starter) — Community guide covering everything you want to know about Google Cloud TPU setup and usage
- [awesome-jax](https://github.com/n2cholas/awesome-jax) — Curated list of JAX libraries, tutorials, and TPU-related resources
- [AI-Hypercomputer GitHub Org](https://github.com/ai-hypercomputer) — Google's GitHub org for AI Hypercomputer reference implementations (MaxText, MaxDiffusion, etc.)
- [JAX AI Stack on Cloud TPUs](https://docs.cloud.google.com/tpu/docs/jax-ai-stack) — Official guide for the JAX+Flax+Optax+Pallas+Tokamax production stack on TPUs
- [Building Production AI on TPUs (Developers Blog)](https://developers.googleblog.com/building-production-ai-on-google-cloud-tpus-with-jax/) — End-to-end walkthrough of the JAX AI Stack components and production deployment patterns
- [TPU v4 Supercomputer Paper (PDF)](https://arxiv.org/pdf/2304.01433) — Full PDF of the ISCA 2023 TPU v4 architecture paper
- [Tensor Processing Unit Wikipedia](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) — Comprehensive historical overview of all TPU generations with specs table

### TPU v8 (Sunfish/Zebrafish, 2026-04-22 announcement)

- [Eighth-Generation TPUs: Two Chips for the Agentic Era — Google Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — Primary announcement of TPU 8t (training) and TPU 8i (inference) at Google Cloud Next '26
- [TPU 8t and TPU 8i Technical Deep Dive — Google Cloud Blog](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — Engineering-level breakdown of compute, memory, ICI, Boardfly topology, and Collectives Acceleration Engine
- [AI Infrastructure at Next '26 — Google Cloud Blog](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — System-level positioning of v8t and v8i in the AI Hypercomputer alongside Axion CPUs and Virgo Network
- [TPU 8t/8i and Virgo Network Analysis — fundaai (Substack)](https://fundaai.substack.com/p/researchtpu-8t8i-and-virgo-network) — Independent analysis of the Virgo optical scale-out fabric and its role in 1M-chip multi-DC training
- [Google TPU 8i and TPU 8t Announced — ServeTheHome](https://www.servethehome.com/google-tpu-8i-for-inference-and-tpu-8t-for-training-announced/) — Spec table including 216 GB / 288 GB HBM3e split, 384 MB on-chip SRAM, 19.2 Tb/s ICI
- [Google Splits TPUv8 Strategy: Broadcom (Sunfish) / MediaTek (Zebrafish) — Wccftech](https://wccftech.com/google-splits-tpuv8-strategy-two-chips-broadcom-training-mediatek-inference-duties/) — Design-partner reporting (Sunfish/Broadcom for training, Zebrafish/MediaTek for inference) and TSMC 2nm node
- [Google Dual Tracks TPU 8 to Conquer Training and Inference — The Register](https://www.theregister.com/2026/04/22/google_tpu8_dual_track_training_inference/) — Independent coverage with FP4 ExaFLOPS numbers and Boardfly description
- [Google Splits AI Chips into Training and Inference TPUs — Digitimes](https://www.digitimes.com/news/a20260424PD205/google-cloud-google-tpu-training-chips.html) — Industry-strategy framing of the workload-specialized split
- [Google Cloud Next 2026: TPU 8t and 8i Architectures — Hyperframe Research](https://hyperframeresearch.com/2026/04/22/google-cloud-next-2026-google-cloud-bifurcates-the-ai-future-specialized-tpu-8t-and-8i-architectures-signal-the-end-of-general-purpose-silicon/) — Strategic analysis of the end of general-purpose AI silicon

### Added 2026-08-08 (scan window 2026-04-27 → 2026-08-08)

**Primary / vendor**

- [Cloud TPU Release Notes](https://docs.cloud.google.com/tpu/docs/release-notes) — retrieved 2026-08-08. Two entries in the window: **2026-06-01 GA — Compute Engine natively supports TPUs** (TPU VMs and slices via standard Compute Engine instance and MIG APIs, custom OS images, boot-disk sizing, all consumption options), and 2026-04-27 GA — TPU availability in AI zones. **No libtpu or JAX version entries for May–Aug 2026.**
- [Cloud TPU System Architecture / Supported Versions](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) — retrieved 2026-08-08. Newest documented generation is **TPU7x (Ironwood)**; TPU v8 has no entry, confirming v8 is not yet available to customers.
- [TPU 8t and TPU 8i Technical Deep Dive — Google Cloud Blog (2026-04-22)](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — *re-read 2026-08-08 as the authoritative v8 spec table*: On-Chip SRAM (Vmem) **128 MB (8t) / 384 MB (8i)**; HBM BW **6,528 / 8,601 GB/s**; peak FP4 **12.6 / 10.1 PFLOPS**; specialized features "SparseCore (Embeddings) & LLM Decoder Engine" (8t) vs "CAE" (8i); v8i chiplet organization (2 TensorCore dies + 1 CAE die); 2.7× training and 1.8× inference price/perf; Boardfly 1,152 connected / 1,024 active chips per pod. Native PyTorch support on TPU is listed as *preview*.
- [Our Eighth Generation TPUs: Two Chips for the Agentic Era — Google Blog (2026-04-22)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — availability wording ("both chips will be generally available later this year"); attaches the "3× more on-chip SRAM" claim to **TPU 8i alone**; credits design "in partnership with Google DeepMind" and names no foundry, node, or ASIC partner.

**Events / benchmarks / commercial**

- [Hot Chips 38 Program](https://hotchips.org/program/conference/) — Stanford, Aug 24–25 2026. Session AI 2 (Tue 2026-08-25, 4:45–6:15 PM PDT, chair Brucek Khailany): "The Eighth Generation TPU Family: Two Chips Optimized for Training and Serving in the Agentic Era", Norman Jouppi & Sridhar Lakshmanamurthy (Google). **Disclosure scheduled — content not yet public**; no slides or abstract. Re-scan after 2026-08-25. Not usable as a source for any specification.
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 24 submitting organizations including Google; two new MoE benchmarks (DeepSeek V3 671B, GPT-OSS 20B). The results post does **not** attribute hardware to individual submitters, so no TPU result can be recorded.
- [Anthropic / Google / Broadcom Compute Partnership — Anthropic (2026-04-06)](https://www.anthropic.com/news/google-broadcom-partnership-compute) — multiple gigawatts of next-generation TPU capacity coming online starting 2027, vast majority US-sited; CFO Krishna Rao. Builds on the Oct 2025 up-to-1M-TPU agreement. Commercial-capacity fact; predates the 2026-04-27 refresh baseline.

**Checked and rejected**

- The Next Platform, "With TPU 8, Google Makes GenAI Systems Much Better, Not Just Bigger" (2026-04-24) — **URL returns HTTP 404**, with and without the trailing numeric path segment. The "external availability late 2027" and "TSMC 2nm" attributions circulating from this piece could not be retrieved and are **not recorded**.
