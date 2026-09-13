# AWS Neuron — Layer Mapping

*as_of: 2026-08-08*
*device_class: Systolic Array Accelerator*

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| **Software Layers** | | | |
| Framework Integration | torch-neuronx (PyTorch/XLA, current ≤ 2.9): xm.mark_step(), torch_neuronx.trace() AOT, torch.compile(backend="openxla_eval") | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Framework Integration | TorchNeuron (PrivateUse1 backend, planned PyTorch ≥ 2.10): eager mode, torch.compile, FSDP/DTensor without XLA dependency — private beta as of Neuron 2.27; **status unverified since 2.27** (no torch-neuronx/TorchNeuron entry in the 2.31.0 component index, which is not evidence either way) | unverified | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Framework Integration | JAX Neuron (StableHLO → Neuron XLA backend, jax.jit, NKI via custom_call; introduced Sep 2024) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Framework Integration | NKI baremetal (framework-agnostic kernel execution via nki.baremetal / custom_call) | confirmed | neuron-sdk-nki.md |
| Compiler / IR | neuronx-cc (XLA frontend → MLIR middle-end → NeuronCore ISA back-end; multi-phase pipeline) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Compiler / IR | neuronx-cc **redesigned code-generation backend** (Neuron 2.31.0, 2026-07-07): improved instruction scheduling + memory prefetch; **default for Trn2 and Trn3** | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Compiler / IR | Graph-compiler improvements (Neuron 2.30.0): rewritten memory-liveness analysis for compile speed; **max computation tile size doubled to 1024**; coalesced reduce-scatter | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Compiler / IR | NKI Compiler (Apache 2.0, Neuron 2.27; MLIR-based; enters pipeline at back-end, bypasses XLA) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Compiler / IR | NEFF (Neuron Executable File Format: NeuronISA instructions + weights + metadata; offline-compilable, checksum-validated) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Op Library | `not applicable` — operator fusion and optimization occur inside neuronx-cc compiler passes (XLA frontend and MLIR middle-end); no separately-versioned op library layer | confirmed | neuron-sdk-nki.md |
| Kernel Library | NKI status: **Beta → Stable at NKI 0.3.0 / Neuron 2.29.0 (2026-04-09)**; NKI 0.4.0 (2.30.0) → NKI 0.5.0 (2.31.0, current as of 2026-08-08) | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Kernel Library | nki.lang (nl.ndarray, nl.load/store, nl.matmul, nl.add; NumPy/Triton-like tile API over SBUF/PSUM) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Kernel Library | nki.isa (ISA intrinsics: tensor_tensor, activation, bn_stats, scalar; explicit SBUF/PSUM partition addressing); **`activate2` fused preprocess+activate on Scalar Engine (Trn3-only, NKI 0.4.0)**; OCP FP8 matmul inputs; Vector-Engine abs-value reductions | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Kernel Library | nki.collectives (all_reduce, all_gather, reduce_scatter, all_to_all, collective_permute, rank_id; added Neuron 2.27); **all_to_all_v (variable-length) added 2.29.0** | confirmed | neuron-sdk-nki.md, communication.md |
| Kernel Library | **nki-stdlib (NKI Standard Library, 2.29.0)**: developer-visible source for all NKI APIs + native `NkiTensor` object; zero-cost view methods (slice/select/permute/rearrange) in NKI 0.5.0 | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Kernel Library | **NKI tensor indirection (`.indirect()`, NKI 0.5.0)**: gather/scatter addressing mode on compute operations; OCP MX scale dtype `float8_e8m0fnu`; 8192-element bf16 destination for `nc_matmul` | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Kernel Library | NKI Library kernels: +3 core / +19 experimental in 2.30.0 (segmented attention, KV-parallel prefill, FP8 quantize; context parallelism, MXFP8 training, SSMs, fused optimizers, MoE dispatch); +14 **experimental** in 2.31.0 (deformable attention, MoE training collectives, indexed gather/scatter, DeepSeek MLA projection, ring attention); PyTorch reference impls for 29 kernels | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Kernel Library | nki-samples (GitHub official: Flash Attention, GEMM, RoPE, MoE routing reference kernels) | confirmed | neuron-sdk-nki.yaml |
| Collective Communication Runtime | NCCom (Neuron Collective Communication): CPU-bypass AllReduce/AllGather/ReduceScatter/AllToAll/AllToAllV/Broadcast; auto NeuronLink vs EFA transport selection | confirmed | communication.md, communication.yaml |
| Distributed Training Libraries | NxD Core (neuronx-distributed): TP/PP/SP/DP/ZeRO-1; ColumnParallelLinear, RowParallelLinear, NxDPPModel; 3D parallelism | confirmed | communication.md, communication.yaml |
| Distributed Training Libraries | NxD Training (neuronx-distributed-training): NeMo compatibility; 3D parallelism; LLaMA-2/3, GPT pretraining; Project Rainier scale | confirmed | communication.md, communication.yaml |
| Distributed Training Libraries | NxD Inference (neuronx-distributed-inference): continuous batching, speculative decoding, vLLM integration; **Neuron 2.29.0 breaking change — NKI kernels unsupported on Trn1/Inf2 (NKI 0.3.0 drops those targets); NxDI models Trn2-and-newer only; Trn1/Inf2 users pin to 2.28** | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Agentic Development (new layer) | **neuron-agentic-development** (first documented SDK component, Neuron 2.30.0; bundled in all DLAMIs/DLCs): `neuron-framework-autoport` (HuggingFace → NxD Inference port + compile + greedy-token-match validation), `neuron-framework-equivalence` (numerical equivalence via progressive 3-tensor R-ratio analysis + fault localization) | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Developer Tools | **Neuron Explorer** profiling/debugging suite: Beta → **Stable** in 2.29.0; full Device widget support; VS Code Extension Marketplace | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Developer Tools | **NKI CPU Simulator** (experimental, 2.29.0): executes NKI kernels entirely on host CPU with no Trainium hardware; enabled by env var or API | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Orchestration | **Neuron DRA Driver** (2.30.0): Kubernetes Dynamic Resource Allocation with topology-aware scheduling of Trainium devices and EFA interfaces; **UltraServer Operator** for Amazon EKS in public beta (2.31.0) | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Distributed Training Libraries | neuronx-nemo-megatron (Neuron-adapted NeMo Megatron: GPT/LLaMA TP+PP+DP; 100B+ scale) | confirmed | communication.yaml |
| Runtime | libnrt.so (in-process shared library; NEFF load/validate; DMA scheduling; NeuronCore memory management; no sidecar daemon) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Runtime | **Zero-copy host↔device transfers enabled by default (2.30.0)**; async event APIs; Collectives support for the Trn3 Gen2 UltraServer ring topology; contiguous shared scratchpad (2.31.0) | confirmed | neuron-sdk-nki.md (2026-08-08 update) |
| Driver / Firmware | aws-neuronx-dkms (kernel mode driver: PCIe BAR mapping, command queues, interrupt routing; co-versioned with libnrt.so) | confirmed | neuron-sdk-nki.md, neuron-sdk-nki.yaml |
| Driver / Firmware | AWS Nitro integration (direct device passthrough; near-bare-metal PCIe throughput) | confirmed | hw-architecture.md, hw-architecture.yaml |
| Assembler / ISA | NeuronISA (proprietary machine ISA for NeuronCores; not publicly documented beyond nki.isa intrinsics; embedded in NEFF) | inferred | neuron-sdk-nki.md |
| **Hardware Layers** | | | |
| Compute Engine | Tensor Engine: 128×128 BF16 systolic array (v2/v3), 512×128 MXFP8/MXFP4 OCP (v4); >90/158/315 TFLOPS per NeuronCore | confirmed | hw-architecture.md, hw-architecture.yaml |
| Compute Engine | Vector Engine: parallel reductions + activations; ~2.3 TFLOPS FP32 (v2); scaled on v3/v4 | confirmed | hw-architecture.md, hw-architecture.yaml |
| Compute Engine | Scalar Engine: element-wise scalar ops; ~2.9 TFLOPS FP32 (v2) | confirmed | hw-architecture.md, hw-architecture.yaml |
| Compute Engine | GPSIMD Engine: 8× 512-bit programmable vector processors; general-purpose C code; direct SRAM access; v2/v3/v4 | confirmed | hw-architecture.md, hw-architecture.yaml |
| Compute Engine | DGE (Dynamic Grouping Engine): hardware MoE token routing + structured sparsity; NeuronCore-v3 and v4 | confirmed | hw-architecture.md, hw-architecture.yaml |
| Data Path | DMA Engine: independent HBM↔SBUF/PSUM transfers; double-buffering (prefetch overlap); explicit nl.load/nl.store | confirmed | hw-architecture.md, hw-architecture.yaml |
| Data Path | Asynchronous engine pipeline: Tensor → PSUM → Vector → Scalar → SBUF producer-consumer chain; compiler-scheduled | confirmed | hw-architecture.md, hw-architecture.yaml |
| On-chip Memory | SBUF (State Buffer): 24/28/32 MiB (v2/v3/v4); 128 partitions; software-managed SRAM scratchpad; no hardware cache | confirmed | hw-architecture.md, hw-architecture.yaml |
| On-chip Memory | PSUM (Partial Sum Buffer): 2 MiB SRAM (unchanged v2/v3/v4); direct Tensor Engine output; pre-Vector Engine accumulator | confirmed | hw-architecture.md, hw-architecture.yaml |
| Off-chip Memory | HBM2e (Trn1): 32 GiB, 820 GB/s per chip; shared across NeuronCores | confirmed | hw-architecture.md, hw-architecture.yaml |
| Off-chip Memory | HBM2e (Trn2): 96 GiB, 2.9 TB/s per chip; Trn2 instance: 1.5 TiB total | confirmed | hw-architecture.md, hw-architecture.yaml |
| Off-chip Memory | HBM3e (Trn3): 144 GiB (4 stacks), 4.9 TB/s per chip; **UltraServer Gen1 (64 devices): 9,216 GiB @ 313.6 TB/s; Gen2 (144 devices): 20,736 GiB @ 705.6 TB/s** | confirmed | hw-architecture.md (2026-08-08 update) |
| Host Interface / Package | PCIe (NeuronDevice-to-host CPU); aws-neuronx-dkms (BAR, command queues, IRQ) | confirmed | hw-architecture.md, hw-architecture.yaml |
| Host Interface / Package | AWS Nitro direct device passthrough; near-bare-metal PCIe throughput; Nitro device model enumeration | confirmed | hw-architecture.md, hw-architecture.yaml |
| Host Interface / Package | Package: TSMC 7nm monolithic (v2/v3); TSMC 3nm N3P dual-chiplet (v4, Trn3) | confirmed | hw-architecture.md, hw-architecture.yaml |
| Scale-up Interconnect | NeuronLink-v2 (Trn1): 768 GB/s bidir, 2D Torus, 16 chips/node | confirmed | hw-architecture.md, communication.md |
| Scale-up Interconnect | NeuronLink-v3 (Trn2): 1 TB/s bidir, 4×4 2D Torus (node) / 3D Torus (UltraServer, 64 chips, ~6,100 copper cables) | confirmed | hw-architecture.md, communication.md |
| Scale-up Interconnect | NeuronLink-v4 + NeuronSwitch-v1 (Trn3): all-to-all switched fabric (O(1) hops); 64 chips (UltraServer Gen1) / 144 chips (UltraServer Gen2). **Bandwidth unreconciled**: 2,048 GiB/s/device (AWS spec table) vs 704 GB/s link-level sum (AWS connectivity section) vs 2.5 TB/s bidir (SemiAnalysis) | contested | hw-architecture.md (2026-08-08 update) |
| Scale-up Interconnect | **Trn3 fabric substrate = PCIe Gen6 switches** (documented 2026-04-09; replaces point-to-point NeuronLink of Trn1/Trn2): 4×Gen6 ×8 intra-server (256 GB/s), 5×Gen6 ×8 inter-server (320 GB/s), 2×Gen6 ×8 inter-rack (128 GB/s); (rack, server, chip) tuple in upper PCIe address bits + BAR-match routing; hardware semaphores share the data path for ordering | confirmed | hw-architecture.md (2026-08-08 update) |
| Scale-out Interconnect | EFAv2 (Trn1): up to 1,600 Gbps/instance; SRD protocol; inter-node AllReduce | confirmed | hw-architecture.md, communication.md |
| Scale-out Interconnect | EFAv3 (Trn2): 3.2 Tbps/instance; 12.8 Tbps/Trn2 UltraServer; petabit-scale UltraClusters | confirmed | hw-architecture.md, communication.md |
| Scale-out Interconnect | EFAv3 (Trn3): **12,800 Gbps per UltraServer Gen1 (64 devices); 28,800 Gbps per Gen2 (144 devices) = 200 Gbps of EFA per Trainium3 device**; AWS reports Trn3 EFA at UltraServer granularity — per-instance figure not disclosed | confirmed | hw-architecture.md (2026-08-08 update) |
| Scale-out Interconnect | EC2 UltraClusters: petabit-scale EFA; Project Rainier ~500,000 Trainium2 chips (2025); **Anthropic states >1,000,000 Trainium2 chips in use as of 2026-04-20**; DP gradient AllReduce at global scale | confirmed | hw-architecture.md (2026-08-08 update), communication.md |
