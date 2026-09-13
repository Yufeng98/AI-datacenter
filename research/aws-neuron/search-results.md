# AWS Neuron (Trainium/Inferentia) — Search Results

Generated: 2026-04-05

---

## Layer 1: Official Documentation & Landing Pages

- [AWS Neuron Documentation (ReadTheDocs)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) — Primary developer documentation hub covering the full Neuron SDK, architecture guides, tutorials, and release notes
- [AWS Neuron Product Page](https://aws.amazon.com/ai/machine-learning/neuron/) — Official AWS landing page for the Neuron SDK for Gen AI and deep learning
- [What is AWS Neuron?](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/what-is-neuron.html) — Overview of the Neuron SDK stack and supported hardware
- [AWS Trainium Product Page](https://aws.amazon.com/ai/machine-learning/trainium/) — Official landing page for AWS Trainium AI accelerator chips
- [Amazon Inferentia Product Page](https://aws.amazon.com/ai/machine-learning/inferentia/) — Official landing page for AWS Inferentia inference chips
- [What's New in AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html) — Changelog and feature announcements across Neuron releases

---

## Layer 2: Hardware Architecture — Chip & NeuronCore

- [AWS Neuron Architecture Guides](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/index.html) — Index of all hardware architecture documentation for Trainium and Inferentia
- [NeuronCore-v2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html) — Detailed microarchitecture of NeuronCore-v2 (Trainium1/Inferentia2): Tensor, Vector, Scalar, GPSIMD engines with software-managed SRAM
- [NeuronCore-v3 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html) — NeuronCore-v3 specs for Trainium2: 28MB on-chip SRAM, DGE hardware block, 158 cFP8 TFLOPS, structured sparsity support
- [Trainium Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium.html) — First-generation Trainium chip architecture overview
- [Trainium2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) — Trainium2 chip: 8 NeuronCore-v3 per chip, 96 GiB HBM, 2.9 TB/s HBM bandwidth, 1.3 PF FP8 compute
- [Trainium3 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) — NeuronCore-v4 based Trainium3: 3nm TSMC, 2.52 PFLOPs FP8, 144 GB HBM3e, MXFP8/MXFP4 support
- [Inferentia2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/general/arch/neuron-hardware/inferentia2.html) — Inferentia2 chip: 2x NeuronCore-v2, 32 GB HBM per chip, 380 INT8 TOPS, 4x throughput over Inf1
- [SemiAnalysis: AWS Trainium3 Deep Dive](https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential) — In-depth independent analysis of Trainium3 architecture and competitive positioning
- [SemiAnalysis: Amazon's AI Self-Sufficiency (Trainium2)](https://newsletter.semianalysis.com/p/amazons-ai-self-sufficiency-trainium2-architecture-networking) — Deep dive into Trainium2 architecture and networking strategy

---

## Layer 3: Hardware Architecture — Memory Hierarchy

- [Trainium/Inferentia2 Architecture Guide for NKI (Memory)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html) — NKI-focused guide to the memory hierarchy: HBM, SBUF (state buffer), PSUM (partial sum buffer), and DMA engines
- [Trainium2 Architecture Guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html) — Detailed Trainium2 memory layout for NKI kernel programming including scratchpad management
- [Amazon EC2 Trn2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html) — Trn2 instance: 16 Trainium2 chips, 1.5 TiB HBM, 46 TB/s HBM bandwidth, 4x4 2D Torus NeuronLink
- [Trainium3 UltraServer Specs (Gen1/Gen2)](https://aws.amazon.com/ec2/instance-types/trn3/) — Trn3 UltraServer configurations: 64–144 chips, up to 20 TB HBM, 706 TB/s bandwidth, NeuronSwitch-v1

---

## Layer 4: Hardware Architecture — Interconnect & Networking

- [Amazon EC2 Trn1 Architecture (NeuronLink)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html) — Trn1 NeuronLink-v2 in 2D Torus topology, 768 GB/s chip-to-chip bandwidth, EFAv2 up to 1600 Gbps
- [Amazon EC2 Trn2 Instances](https://aws.amazon.com/ec2/instance-types/trn2/) — Trn2 NeuronLink at 1 TB/s chip-to-chip, EFAv3 at 3.2 Tbps per instance, UltraServer at 12.8 Tbps EFAv3
- [Amazon EC2 Trn3 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html) — Trn3 with NeuronSwitch-v1 for all-to-all connectivity optimized for MoE and autoregressive inference
- [Amazon EC2 Inf2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/inf2-arch.html) — Inf2 NeuronLink at 192 GB/s for multi-chip model sharding
- [AWS Trn2 UltraServer Announcement](https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available/) — Launch blog for Trn2 UltraServers: 64-chip scale-up with ring topology between instances
- [Neuron Collective Communication](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html) — How Neuron orchestrates AllReduce and other collectives over NeuronLink and EFA without CPU involvement

---

## Layer 5: Compiler Stack

- [AWS Neuron Compiler Overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/index.html) — Overview of Neuron SDK compiler pipeline: XLA front-end graph optimizations → MLIR IR → NeuronCore back-end codegen
- [NKI Compiler (MLIR-based)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/about/index.html) — Open-source NKI compiler built on MLIR; NKI kernels bypass first 3 graph-level passes and enter at IR back-end
- [On the Programmability of AWS Trainium and Inferentia (TDS)](https://towardsdatascience.com/on-the-programmability-of-aws-trainium-and-inferentia-cd455826e26c/) — Independent analysis of compiler abstraction layers and programmability tradeoffs for Neuron devices
- [Trainium3 EC2 UltraServer Announcement](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/) — Launch announcement for Trn3 UltraServers with NeuronCore-v4 and new compiler support for MXFP4/MXFP8

---

## Layer 6: Neuron Kernel Interface (NKI) — Custom Kernels

- [NKI Documentation Index](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/) — Top-level NKI docs covering API reference, architecture guides, tutorials, and release notes
- [About Neuron Kernel Interface (NKI)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/about/index.html) — NKI overview: Python-based bare-metal tile programming with NumPy/Triton-like syntax; nki.lang and nki.isa APIs
- [NKI Quickstart: Implement and Run Your First Kernel](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/quickstart-implement-run-kernel.html) — Step-by-step guide to writing, compiling, and executing a first NKI kernel on Trainium/Inferentia
- [NKI Tutorials](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/tutorials/index.html) — Collection of hands-on NKI tutorials covering flash attention, matrix multiplication, and custom ops
- [NKI Kernel as a Framework Custom Operator](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/framework_custom_op.html) — Guide for integrating NKI kernels as custom ops into PyTorch or JAX training/inference graphs
- [NKI Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki.html) — Changelog for NKI compiler and API across Neuron SDK releases
- [AWS Neuron Introduces NKI (What's New)](https://aws.amazon.com/about-aws/whats-new/2024/09/aws-neuron-nki-nxd-training-jax/) — Launch announcement for NKI, NxD Training, and JAX support (September 2024)

---

## Layer 7: NKI Open-Source Samples & Repos

- [aws-neuron/nki-samples (GitHub)](https://github.com/aws-neuron/nki-samples) — Official NKI kernel samples repository: advanced kernels, tutorials, and community contributions for Trainium/Inferentia
- [NKI Samples Documentation Site](https://aws-neuron.github.io/nki-samples/) — Hosted docs for nki.kernels reference implementations migrated from main Neuron docs
- [aws-neuron/nki-llama (GitHub)](https://github.com/aws-neuron/nki-llama) — End-to-end example of NKI kernels for Llama 3.2 1B inference on Trainium/Inferentia
- [aws-neuron/nki-moe (GitHub)](https://github.com/aws-neuron/nki-moe) — MLSys competition repository for optimized Mixture-of-Experts NKI kernels

---

## Layer 8: Framework Integration — PyTorch

- [PyTorch Support on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html) — Documentation hub for torch-neuronx: TorchDynamo backend, torch.compile, eager execution, and distributed APIs
- [aws-neuron/neuronx-distributed (GitHub)](https://github.com/aws-neuron/neuronx-distributed) — NxD Core: foundational PyTorch library for tensor parallelism, pipeline parallelism, and distributed primitives on Neuron
- [aws-neuron/neuronx-distributed-training (GitHub)](https://github.com/aws-neuron/neuronx-distributed-training) — NxD Training: high-level training library with NeMo compatibility, 3D parallelism, and activation checkpointing
- [aws-neuron/neuronx-nemo-megatron (GitHub)](https://github.com/aws-neuron/neuronx-nemo-megatron) — Neuron-adapted NeMo Megatron for GPT/LLaMA pretraining: tensor/pipeline/data parallelism at 100B+ parameter scale
- [AWS Neuron Reference for Megatron-LM (GitHub)](https://github.com/aws-neuron/aws-neuron-reference-for-megatron-lm) — Megatron-LM adapted for Trainium with Neuron-specific optimizations

---

## Layer 9: Framework Integration — JAX

- [JAX Support Announcement](https://aws.amazon.com/about-aws/whats-new/2024/09/aws-neuron-nki-nxd-training-jax/) — Official announcement of JAX support for training and inference on Trainium via Neuron SDK
- [AWS Neuron Libraries Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/index.html) — Index covering all Neuron libraries including JAX integration, NxD Core, NxD Training, and NxD Inference

---

## Layer 10: Inference Serving & LLM Deployment

- [NxD Inference (neuronx-distributed-inference) GitHub](https://github.com/aws-neuron/neuronx-distributed-inference) — Production inference library: continuous batching, speculative decoding, vLLM integration for Trainium/Inferentia
- [vLLM on Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/vllm/index.html) — Guide for running vLLM on Neuron via the NxD Inference plugin system
- [aws-neuron/upstreaming-to-vllm (GitHub)](https://github.com/aws-neuron/upstreaming-to-vllm) — Fork tracking AWS Neuron contributions being upstreamed to the main vLLM project
- [Serving LLMs with vLLM on AWS AI Chips (Blog)](https://aws.amazon.com/blogs/machine-learning/serving-llms-using-vllm-and-amazon-ec2-instances-with-aws-ai-chips/) — AWS blog walkthrough for deploying LLMs with vLLM on Trainium/Inferentia instances
- [vLLM AWS Neuron Installation Guide](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/aws_neuron.html) — Official vLLM docs for Neuron-specific installation and configuration
- [AWS Neuron Reference for NeMo Megatron (Docs)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nemo-megatron/index.html) — Documentation for neuronx-nemo-megatron library

---

## Layer 11: Distributed Training & Collective Communication

- [Neuron Collective Communication (Docs)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/about/collectives.html) — Architecture of Neuron collectives: NeuronLink intra-node + EFA inter-node, bypassing CPU for AllReduce/AllGather
- [aws-neuron/aws-neuron-parallelcluster-samples (GitHub)](https://github.com/aws-neuron/aws-neuron-parallelcluster-samples) — Example jobs for running NxD/NeMo Megatron training on AWS ParallelCluster (HPC scheduler integration)
- [Get Started with EFA and NCCL for ML (AWS Docs)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html) — Guide for configuring EFA + NCCL/MPI for distributed ML training on EC2

---

## Layer 12: Runtime & Driver

- [NeuronX Runtime Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/index.html) — Runtime architecture: libnrt.so integrated into ML frameworks, kernel mode driver, device management APIs
- [Neuron Driver Release Notes (aws-neuronx-dkms)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/runtime/aws-neuronx-dkms/index.html) — Changelog for the aws-neuronx-dkms kernel module supporting Inf1, Inf2, Trn1, Trn1n, Trn2
- [Neuron Runtime Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/runtime/aws-neuronx-runtime-lib/index.html) — Changelog for the Neuron runtime library (libnrt.so)
- [Neuron Runtime Troubleshooting](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html) — Debugging guide for runtime issues on Inf1, Inf2, Trn1 instances

---

## Layer 13: Developer Tools & Profiling

- [Neuron Developer Tools Overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/index.html) — Index of Neuron tooling: neuron-top, neuron-monitor, neuron-ls, Neuron Profiler 2.0, Neuron Sysfs
- [Neuron Monitor User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html) — neuron-monitor: streams JSON metrics to stdout; integrations for CloudWatch, Prometheus, and Kubernetes
- [Neuron Monitor for EKS (Blog)](https://aws.amazon.com/blogs/machine-learning/scale-and-simplify-ml-workload-monitoring-on-amazon-eks-with-aws-neuron-monitor-container/) — Guide to deploying Neuron Monitor container on EKS with Prometheus + Grafana dashboards
- [Neuron System Tools Release Notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/tools/aws-neuronx-tools.html) — Changelog for aws-neuronx-tools package including neuron-top, neuron-ls, and Neuron Profiler
- [Third-Party Solutions (Datadog, etc.)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/third-party-solutions.html) — Integrations with Datadog and other observability platforms for Neuron metrics

---

## Layer 14: Open-Source Repos (aws-neuron GitHub Org)

- [AWS Neuron GitHub Organization](https://github.com/aws-neuron) — 27+ repositories covering SDK, samples, tools, and framework integrations for Trainium/Inferentia
- [aws-neuron/aws-neuron-sdk (GitHub)](https://github.com/aws-neuron/aws-neuron-sdk) — Primary SDK repository: release notes, issue tracker, and documentation source files
- [aws-neuron/aws-neuron-samples (GitHub)](https://github.com/aws-neuron/aws-neuron-samples) — Example code for training and inference across PyTorch, JAX, and HuggingFace on Neuron devices

---

## Layer 15: Ecosystem, Benchmarks & Community

- [AWS Neuron Blog Posts](https://aws.amazon.com/blogs/machine-learning/category/artificial-intelligence/aws-neuron/) — Official AWS Machine Learning blog posts covering Neuron features, benchmarks, and case studies
- [NextPlatform: With Trainium4, AWS Will Crank Up Everything](https://www.nextplatform.com/2025/12/03/with-trainium4-aws-will-crank-up-everything-but-the-clocks/) — Industry analysis of Trainium roadmap through Trainium4 and AWS silicon strategy
- [AWS re:Post: Model Support on Inferentia/Trainium](https://repost.aws/articles/ARlsXmJV5xR0OPQb8gxIDWaA/how-do-i-know-if-an-open-source-model-is-supported-on-inferentia-trainium-or-neuron) — Community guide on determining open-source model compatibility with Neuron devices
- [Inferentia2 Launch Blog (4x throughput, 10x lower latency)](https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/) — AWS blog detailing Inferentia2 performance improvements over Inferentia1
- [EC2 Inf2 General Availability Blog](https://aws.amazon.com/blogs/aws/amazon-ec2-inf2-instances-for-low-cost-high-performance-generative-ai-inference-are-now-generally-available/) — Launch announcement for Inf2 instances for cost-effective generative AI inference
- [HPCwire: AWS Brings Trainium3 to Market](https://www.hpcwire.com/2025/12/02/aws-brings-the-trainium3-chip-to-market-with-new-ec2-ultraservers/) — HPC industry coverage of Trainium3 UltraServer launch at re:Invent 2025
- [Introl Blog: AWS Trainium and Inferentia Silicon Ecosystem Guide](https://introl.com/blog/aws-trainium-inferentia-silicon-ecosystem-guide-2025) — Comprehensive third-party overview of the Trainium/Inferentia silicon and software ecosystem

---

## Appended 2026-08-08 — Resources found in the 2026-04-05 → 2026-08-08 update scan

### Release notes and SDK sources (2.29 – 2.31)

- [aws-neuron-sdk GitHub releases](https://github.com/aws-neuron/aws-neuron-sdk/releases) — authoritative release dates via tag `published_at`: v2.28.0 (2026-02-25), v2.28.1 (2026-03-13), v2.29.0 (2026-04-09), v2.29.1 (2026-05-01), v2.30.0 (2026-05-21), v2.31.0 (2026-07-07)
- [GitHub API: releases list](https://api.github.com/repos/aws-neuron/aws-neuron-sdk/releases?per_page=15) — machine-readable release metadata; use when the readthedocs release index and the GitHub tag disagree on a date (as they do for 2.29.1)
- [Neuron SDK release-notes index](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html) — per-release index; note its own caveat that some components may not be updated in a given release
- [Neuron SDK 2.31.0 release notes (rendered)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.31.0.html) — current latest release as of 2026-08-08
- [Neuron SDK 2.31.0 release notes (source)](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/2.31.0.rst) — component card list; no torch-neuronx / TorchNeuron entry
- [NKI component release notes](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nki.rst) — NKI 0.3.0 / 0.4.0 / 0.5.0 mapping, Beta→Stable transition, SPMD LNC2 deprecation notice
- [NxD Inference component release notes](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/nxd-inference.rst) — exact wording of the Trn1/Inf2 support drop (authoritative over the release index card)
- [Neuron Agentic Development component release notes](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/release-notes/components/agentic-development.rst) — component history begins at 2.30.0
- [Announcing AWS Neuron 2.29](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/) — AWS What's New post for 2.29
- [Announcing AWS Neuron 2.30.0](https://aws.amazon.com/about-aws/whats-new/2026/05/aws-announce-neuron-2-30-0/) — AWS What's New post for 2.30

### New stack layer

- [aws-neuron/neuron-agentic-development (GitHub)](https://github.com/aws-neuron/neuron-agentic-development) — AI coding-agent skills shipped as an SDK component: `neuron-framework-autoport`, `neuron-framework-equivalence`; bundled in all DLAMIs/DLCs

### Trn3 hardware — primary sources and revision history

- [Trn3 architecture page (live)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html) — Gen1/Gen2 spec table plus the "Trn3 UltraServer Connectivity and Networking" section
- [trn3-arch.rst at master](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/about-neuron/arch/neuron-hardware/trn3-arch.rst) — 136 lines; the PCIe Gen6 fabric disclosure
- [trn3-arch.rst commit history (API)](https://api.github.com/repos/aws-neuron/aws-neuron-sdk/commits?path=about-neuron/arch/neuron-hardware/trn3-arch.rst) — db1586c3 and 6e694cbb (2025-12-02), b741ad5b (2026-01-29), 539a73c5 (2026-04-09); proves the spec table predates the 2026-04-05 baseline and dates the PCIe section
- [trn3-arch.rst at 6e694cbb (2025-12-02)](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/6e694cbb/about-neuron/arch/neuron-hardware/trn3-arch.rst) — December 2025 revision already containing EFA 12,800 / 28,800 Gbps
- [trn3-arch.rst at b741ad5b (2026-01-29)](https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/b741ad5b/about-neuron/arch/neuron-hardware/trn3-arch.rst) — 83 lines, no PCIe connectivity section; the control revision establishing that section as new
- [Amazon EC2 Trn3 instances](https://aws.amazon.com/ec2/instance-types/trn3/) — UltraServer aggregates only; states neither GA nor preview, lists no instance sizes or regions

### Deployment scale

- [Anthropic — Expanding our compute with Amazon (2026-04-20)](https://www.anthropic.com/news/anthropic-amazon-compute) — up to 5 GW new capacity; >1,000,000 Trainium2 chips in use; nearly 1 GW combined Trn2+Trn3 by end of 2026; $5B + up to $20B on top of $8B; $100B+ Anthropic commitment to AWS

### Negative-result checks (nothing to record)

- [Hot Chips 38 program](https://hotchips.org/program/conference/) — Aug 23–25, 2026; AWS / Amazon / Annapurna absent from the program
- [MLPerf Training v6.0 results (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — 24 submitters; no AWS/Amazon/Trainium submission
