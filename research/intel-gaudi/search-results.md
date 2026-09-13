# Intel Gaudi Software Stack & Hardware Resources

*as_of: 2026-08-08 (baseline search 2026-04-05; see "New resources (2026-08-08)" at end)*
*device_class: AI Accelerator (HPU)*
*seeds: https://docs.habana.ai/, https://github.com/HabanaAI*

## Software Stack

### Framework Integration
- [PyTorch on Intel Gaudi — Getting Started](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html) — Official guide for training and inference on Gaudi with PyTorch, covering eager and lazy execution modes
- [Gaudi PyTorch Bridge (gaudi-pytorch-bridge)](https://github.com/HabanaAI/gaudi-pytorch-bridge) — Open-source PyTorch device backend for Intel Gaudi, bridges torch ops to SynapseAI
- [Intel Gaudi PyTorch Python API (habana_frameworks.torch)](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html) — Python package providing optimizers, mixed precision, and fused kernels for Gaudi
- [PyTorch Gaudi Theory of Operations](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html) — Explains how PyTorch ops are dispatched through the Gaudi bridge in eager and lazy modes
- [Optimum Habana (HuggingFace)](https://huggingface.co/docs/optimum-habana/index) — HuggingFace Optimum integration for training and inference on Intel Gaudi HPUs
- [PyTorch Library Docker Images for Intel Gaudi](https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-container.html) — Pre-built Docker containers with full Gaudi PyTorch software stack
- [Lightning-Habana](https://github.com/Lightning-AI/lightning-Habana) — PyTorch Lightning plugin for Intel Habana Gaudi accelerators

### Compiler / IR
- [Intel Gaudi Graph Compiler](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html) — Ahead-of-time graph compiler that fuses ops, manages data layout, and generates optimized binaries for Gaudi MME and TPC
- [TPC-C Compiler (LLVM-based)](https://docs.habana.ai/en/latest/TPC/TPC_Getting_Started/TPC_Getting_Started.html) — LLVM-based compiler for the TPC-C/C++ language with intrinsics targeting the full Gaudi TPC ISA
- [SynapseAI Graph API](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html) — Low-level graph construction and compilation API forming the foundation of the SynapseAI software stack
- [GPU Migration Toolkit](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html) — Automated tool that replaces GPU-specific Python API calls with Gaudi-equivalent calls during model migration

### Op Library
- [TPC Kernel Library (1400+ kernels)](https://developer.habana.ai/get-started/kernel-libraries/) — Extensive built-in library of TPC kernels covering elementwise, non-linear, and non-GEMM operators used by the graph compiler
- [Intel Gaudi Transformer Engine (FP8)](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html) — Optimized PyTorch module implementations for Transformer architectures using FP8 precision
- [TPC Intrinsics Guide](https://docs.habana.ai/en/latest/TPC/TPC_Intrinsics_Guide/index.html) — Reference for TPC intrinsic functions exposing the full ISA for custom kernel authors

### Kernel Library
- [TPC Getting Started Guide](https://docs.habana.ai/en/latest/TPC/TPC_Getting_Started/TPC_Getting_Started.html) — End-to-end guide for writing, compiling, and integrating custom TPC kernels into the graph compiler
- [TPC User Guide](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html) — Detailed reference for TPC processor architecture, pipeline, memory model, and ISA
- [Use TPC Kernels on Intel Gaudi (Intel Developer)](https://www.intel.com/content/www/us/en/developer/platform/gaudi/learn/habana-tpc-english.html) — Tutorial for developing and deploying custom TPC kernels

### Runtime
- [SynapseAI Runtime](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html) — User-space runtime that manages device execution, memory, and communication between the host and Gaudi hardware
- [Intel Gaudi Software Suite Overview](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html) — Comprehensive overview of all SynapseAI software components: runtime, compiler, drivers, and communication libraries
- [SynapseAI Core (archived)](https://github.com/HabanaAI/SynapseAI_Core) — Reference implementation of the SynapseAI API (archived February 2026, no longer maintained by Intel)
- [Docker Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) — Instructions for using pre-built Gaudi Docker images from the Habana vault
- [Setup and Install (HabanaAI)](https://github.com/HabanaAI/Setup_and_Install) — Scripts and instructions for setting up Habana binaries and Docker images
- [Intel Gaudi Base Operator for Kubernetes](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Kubernetes_Installation/Kubernetes_Operator.html) — Kubernetes operator for deploying and managing Intel Gaudi accelerators in containerized environments

### Driver / Firmware
- [Driver Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) — Installation instructions for the habanalabs kernel module and firmware
- [Intel Gaudi Software Stack and Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html) — Bare-metal installation guide covering firmware, kernel driver, and SynapseAI stack
- [Intel Gaudi Software (Intel)](https://www.intel.com/content/www/us/en/software/ai-accelerators/gaudi-software.html) — Product page for Intel Gaudi software including drivers and SDK downloads
- [Open-Source Gaudi 3 Linux Kernel Driver (Phoronix)](https://www.phoronix.com/news/Intel-Gaudi-3-Open-Source) — News coverage of Intel's open-source Gaudi 3 driver submission to the mainline Linux kernel

### Communication
- [HCCL API Reference](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html) — Full API reference for the Habana Collective Communications Library (HCCL), Intel Gaudi's NCCL-compatible collective communication library
- [HCCL Supported Collective Primitives](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/Overview.html) — Overview of AllReduce, AllGather, ReduceScatter, and point-to-point operations supported by HCCL
- [HCCL Distributed Backend Initialization](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/Distributed_Backend_Initialization.html) — Guide for setting up HCCL as the PyTorch distributed backend for multi-Gaudi training

### Assembler / ISA
- [Processor Architectural Overview (TPC)](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/Processor_Architectural_Overview.html) — Details of the VLIW SIMD TPC processor pipeline, register file, local memory, and 2048-bit vector units
- [TPC Tools Installation Guide](https://docs.habana.ai/en/latest/TPC/TPC_Tools_Installation/TPC_Tools_Installation_Guide.html) — Installation of habanalabs-tools package containing TPC-C compiler, assembler, disassembler, and simulator

## Hardware Architecture

### Compute Engine
- [Gaudi Architecture Overview](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) — Official documentation of the heterogeneous compute architecture: MME (matrix multiply engine) and TPC (tensor processor core) cluster
- [Gaudi 3 AI Accelerator White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf) — Intel's technical white paper detailing Gaudi 3 compute, memory, and networking specifications
- [Gaudi 3 at Hot Chips 2024 (PDF)](https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf) — Microarchitecture deep-dive presentation on Gaudi 3 from Hot Chips 2024
- [Habana Gaudi Microarchitecture (WikiChip)](https://en.wikichip.org/wiki/habana/microarchitectures/gaudi) — Detailed WikiChip breakdown of first-generation Gaudi architecture
- [Gaudi Architecture and Software Overview (docs index)](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) — Documentation landing page covering architecture, software suite, and generation comparisons

### Data Path
- [GEMM, Attention, vLLM on Gaudi (SqueezeBits)](https://blog.squeezebits.com/intel-gaudi-gemm-attention-performance) — Performance analysis of GEMM and attention operator execution paths on the Gaudi MME and TPC
- [Intel Gaudi Introduction (SqueezeBits)](https://blog.squeezebits.com/intel-gaudi-1-introduction-35414) — Accessible technical introduction to the Gaudi compute and memory architecture

### On-chip Memory
- [Gaudi 3 SRAM and L2/L3 Cache](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) — Gaudi 3 has 96 MB on-die SRAM (2x Gaudi 2) with two-level L2/L3 cache hierarchy, 12.8 TB/s SRAM bandwidth
- [Gaudi Architecture Forum Q&A](https://forum.habana.ai/t/questions-regarding-the-architecture-about-habana-gaudi/355) — Community discussion clarifying TPC local memory, SRAM, and cache access patterns

### Off-chip Memory
- [Gaudi 3 HBM Memory Specs (VideoCardz)](https://videocardz.com/newz/intel-announces-gaudi3-ai-accelerator-with-128gb-hbm2e-memory-up-to-900w-tdp-on-air) — Gaudi 3: 128 GB HBM2e at 3.7 TB/s; Gaudi 2: 96 GB HBM2e at 2.45 TB/s
- [Gaudi 3 OAM Product Brief (PDF)](https://cdrdv2-public.intel.com/817487/gaudi-3-ai-accelerator-hl-325l-oam-mezzanine-card-product-brief.pdf) — Product brief for HL-325L OAM mezzanine card with full memory and power specs

### Host Interface / Package
- [Intel Gaudi AI Accelerator Products](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html) — Product lineup page covering Gaudi 1, Gaudi 2, and Gaudi 3 SKUs and packaging options
- [Intel Gaudi Developer Overview](https://www.intel.com/content/www/us/en/developer/platform/gaudi/overview.html) — Developer platform landing page with links to SDK, documentation, and hardware specifications

### Scale-up Interconnect
- [Scaling Systems with Gaudi's Integrated RoCE](https://habana.ai/gaudi-integrated-roce/) — Habana blog explaining the on-chip integrated NIC implementing RoCE v2 for scale-up within a node
- [Gaudi 3 Cluster Reference Design (PDF)](https://cdrdv2-public.intel.com/833842/gaudi-3-ai-accelerator-cluster-ref-design-white-paper.pdf) — White paper on multi-node Gaudi 3 cluster topology with 21x 200 GbE scale-up and 3x 200 GbE scale-out per card
- [Accelerating Ethernet-Native AI Clusters with Gaudi 3 (Cisco)](https://blogs.cisco.com/datacenter/accelerating-ethernet-native-ai-clusters-with-intel-gaudi-3-ai-accelerators-and-cisco-nexus-9000) — Cisco-Intel joint post on pure-Ethernet fabric for Gaudi 3 clusters using RoCE v2

### Scale-out Interconnect
- [Intel Gaudi 3 Scale-out via Ethernet (Tom's Hardware)](https://www.tomshardware.com/pc-components/cpus/intel-details-guadi-3-at-vision-2024-new-ai-accelerator-sampling-to-partners-now-volume-production-in-q3) — Coverage of Gaudi 3's three 200 GbE ports per card for external cluster connectivity

## Other Resources
- [Intel Gaudi Documentation Hub](https://docs.habana.ai/en/latest/) — Root of all official Gaudi/SynapseAI documentation (v1.23.0 as of 2026-04)
- [HabanaAI GitHub Organization](https://github.com/HabanaAI) — Central GitHub org hosting all open-source Gaudi tools, models, and bridge repos
- [Model-References (HabanaAI)](https://github.com/HabanaAI/Model-References) — Reference model implementations for Gaudi covering CV and NLP training and inference
- [Megatron-DeepSpeed for Gaudi (HabanaAI)](https://github.com/HabanaAI/Megatron-DeepSpeed) — Gaudi-optimized fork of Megatron-DeepSpeed for large-scale LLM pretraining with 3D parallelism
- [vLLM Fork for Gaudi (HabanaAI)](https://github.com/HabanaAI/vllm-fork) — Gaudi-specific vLLM fork with custom paged attention and HPU operator optimizations (vLLM plugin to be default from v1.24.0)
- [Gaudi Tutorials (HabanaAI)](https://github.com/HabanaAI/Gaudi-tutorials) — Jupyter notebook tutorials for training and inference on Gaudi 1 and Gaudi 2
- [Intel Gaudi Developer Community](https://developer.habana.ai/) — Developer portal with SDK downloads, kernel library catalog, and community forum
- [Gaudi Developer Forum](https://forum.habana.ai/) — Community Q&A forum for Intel Gaudi architecture, software, and debugging questions
- [Memory-Efficient Training with DeepSpeed on Gaudi (Intel)](https://www.intel.com/content/www/us/en/developer/articles/training/memory-efficient-training-on-gaudi-with-deepspeed.html) — Guide for ZeRO-1/ZeRO-2 optimizer state partitioning with DeepSpeed on Gaudi
- [LLM Training and Inference on Gaudi 2 (Databricks)](https://www.databricks.com/blog/llm-training-and-inference-intel-gaudi2-ai-accelerators) — Databricks blog covering Gaudi 2 integration with Spark ML and LLM workloads
- [Optimum Habana Fork (HabanaAI)](https://github.com/HabanaAI/optimum-habana-fork) — Public staging fork used to upstream Gaudi changes to HuggingFace optimum-habana

---

## New resources (2026-08-08)

*Found during the 2026-04-05 → 2026-08-08 update scan. Crescent Island Computex items are trade press: broad corroboration of a single Intel briefing, no Intel first-party page — treat figures as medium confidence.*

### Driver / Firmware — upstream status

- [mainline `drivers/accel/habanalabs/` (Linux master)](https://github.com/torvalds/linux/tree/master/drivers/accel/habanalabs) — Contains `common/`, `goya/`, `gaudi/`, `gaudi2/`, `include/`; **no `gaudi3/` directory as of 2026-08-08**
- [dri-devel: "accel/habanalabs: Gaudi3 support and updates for v6.19" (Dec 2025)](https://lists.freedesktop.org/archives/dri-devel/2025-December/539169.html) — Gaudi 3 upstreaming pull request; subsequently rejected on code-quality grounds
- [Intel Ceases SynapseAI Core Development (Phoronix, 2025-12-15)](https://www.phoronix.com/news/Intel-SynapseAI-Stops) — Reports the Feb 2025 archival of the SynapseAI Core reference implementation and the Gaudi 3 driver missing Linux 6.19

### Software stack — repository status

- [HabanaAI GitHub org, sorted by last update](https://github.com/orgs/HabanaAI/repositories?type=all&sort=updated) — Archive status and push dates: SynapseAI_Core archived 2025-02-03, Gaudi-tutorials 2025-09-18, Model-References 2026-01-08; gaudi-pytorch-bridge / vllm-fork / optimum-habana-fork / gaudi-* operator suite all active mid-2026
- [Intel Gaudi Documentation Hub — v1.24.0](https://docs.habana.ai/en/latest/) — Current documentation version as of 2026-08-08 (repo previously cited v1.23.0)

### Successor software stack (Crescent Island)

- [Intel Compute Runtime 26.01.36711.4 (Phoronix, 2026-01-14)](https://www.phoronix.com/news/Intel-CR-26.01.36711.4) — Early Crescent Island support in the Level Zero / OpenCL compute runtime
- [Nova Lake-S and Crescent Island added to Intel Graphics Compiler (TechPowerUp, 2026-01-15)](https://www.techpowerup.com/345253/intel-nova-lake-s-and-crescent-island-support-added-to-graphics-compiler) — IGC v2.27.10 initial Crescent Island support

### Crescent Island hardware (Computex 2026, trade press)

- [Intel Crescent Island at Computex (acceleratedcomputing.ai, 2026-06-01)](https://acceleratedcomputing.ai/news/2026-06-01-intel-crescent-island/) — 480 GB LPDDR5X, 350 W air-cooled PCIe, FP4/MXFP4 through FP64, sampling H2 2026; explicitly states bandwidth and core counts were **not published**
- [Intel Crescent Island GPU (codeoxi)](https://codeoxi.com/blog/intel-crescent-island-gpu) — 160 GB stock / up to 480 GB ODM, 350 W, FP4–FP64, sampling H2 2026, GA 2027; notes Intel gave no headline bandwidth figure
- [Crescent Island officially supports up to 480 GB LPDDR5X (VideoCardz)](https://videocardz.com/newz/intel-crescent-island-gpu-officially-supports-up-to-480gb-lpddr5x-memory) — 160 GB standard, partner configurations to 480 GB
- [Intel details Crescent Island AI GPU at Computex (Tom's Hardware)](https://www.tomshardware.com/pc-components/gpus/intel-details-long-awaited-crescent-island-ai-gpu-at-computex) — "up to 480 GB of LPDDR5X", "from FP4 … all the way up to FP64"
- [Crescent Island up to 480 GB LPDDR5X (TechSpot)](https://www.techspot.com/news/112608-intel-crescent-island-gpu-support-up-480gb-lpddr5x.html) — Source of the "684 GB/s" figure, phrased as bandwidth "expected to reach" — an estimate, **not** an Intel spec

### Intel first-party pages checked (negative results — no Crescent Island content)

- [Intel at Computex 2026 press kit](https://newsroom.intel.com/press-kit/press-kit-intel-at-computex-2026) — no mention of Crescent Island
- [Computex 2026: An Intelligent World Built on Silicon (2026-06-02)](https://newsroom.intel.com/artificial-intelligence/computex-2026-an-intelligent-world-built-on-silicon) — no mention of Crescent Island
- [Intel Newsroom — Artificial Intelligence index](https://newsroom.intel.com/artificial-intelligence) — May–Aug 2026 post index contains no Crescent Island or Gaudi post

### Benchmarks

- [MLPerf Inference v6.0 results (MLCommons, 2026-04-01)](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/) — 24 submitting organizations including Intel; no Gaudi entries
- [Intel Delivers Open, Scalable AI Performance in MLPerf Inference v6.0 (Intel Newsroom, 2026-04-01)](https://newsroom.intel.com/artificial-intelligence/intel-delivers-ai-performance-mlperf-inference-v6-0) — Xeon 6 + Arc Pro B70/B65/B60 only; zero Gaudi mentions
- [MLPerf Training v6.0 results (MLCommons, 2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 24 submitting organizations; Intel absent entirely

### Scheduled disclosure (not a source for any specification)

- [Hot Chips 38 program](https://hotchips.org/program/conference/) — Day-1 GPU session, Mon 2026-08-24 16:45–18:45: Intel, "Crescent Island: GPU Designed for Agentic AI Inference" (Sumit Mohan, Hong Jiang). **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** No Gaudi or Jaguar Shores talk on the program.

### Adjacent Intel datacenter AI (context only, not Gaudi)

- [Intel Announces New AI Innovations at Computex (intc.com press release)](https://www.intc.com/news-events/press-releases/detail/1771/intel-announces-new-ai-innovations-at-computex-chip-to) — Production rack-scale AI infrastructure pairing Xeon 6+ (Intel 18A) with SambaNova SN-50 RDUs, integrated by Foxconn
