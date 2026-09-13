# AMD GPU Software Stack & Hardware Resources

*as_of: 2026-08-08*
*device_class: GPU*
*seeds: https://github.com/ROCm/ROCm, https://github.com/ROCm/MIOpen, https://github.com/ROCm/RCCL*

## Software Stack

### Framework Integration
- [PyTorch ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/install/3rd-party/pytorch-install.html) — PyTorch with ROCm backend, HIP device support
- [torch.compile on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html) — TorchDynamo + TorchInductor generating Triton kernels for AMD
- [JAX on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/train-a-model.html) — JAX with ROCm XLA backend
- [JAX-AITER](https://rocm.blogs.amd.com/software-tools-optimization/jax-aiter/README.html) — AMD's optimized AI kernels for JAX on ROCm
- [TensorFlow ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/train-a-model.html) — TensorFlow with ROCm backend

### Compiler / IR
- [HIP Compiler (hipcc / amdclang++)](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html) — HIP C++ compiler based on Clang/LLVM, compiles HIP to AMDGPU IR
- [LLVM AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html) — LLVM backend for AMD GPUs, generates GFX ISA from AMDGPU IR
- [Triton ROCm Backend](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html) — OpenAI Triton with AMD GPU backend, MLIR → AMDGPU ISA
- [MLIR AMDGPU Dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/) — MLIR dialect for AMD GPU operations
- [ROCm LLVM](https://github.com/ROCm/llvm-project) — AMD's fork of LLVM with AMDGPU enhancements

### Op Library
- [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/) — AMD's DNN operation library (conv, pooling, batch norm, RNN, attention)
- [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/) — ROCm BLAS library, optimized GEMM for AMD GPUs
- [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/) — Lightweight BLAS with flexible API, algorithm selection, epilogue fusion
- [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/) — GPU-accelerated FFT library
- [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/) — GPU-accelerated sparse matrix operations
- [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/) — GPU-accelerated LAPACK-like solvers

### Kernel Library
- [AITER](https://github.com/ROCm/aiter) — AI Tensor Engine for ROCm: centralized high-performance AI kernels (attention, MoE, GEMM, quantization)
- [Composable Kernel (CK)](https://github.com/ROCm/composable_kernel) — Template-based kernel library for GEMM/conv on AMD GPUs (similar to CUTLASS)
- [Triton Kernels on AMD](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html) — Custom GPU kernels via Triton Python DSL
- [hipCUB / rocPRIM](https://rocm.docs.amd.com/projects/hipCUB/) — Parallel primitive libraries (reduce, scan, sort) for AMD GPUs
- [rocm-libraries](https://github.com/ROCm/rocm-libraries) — Unified repo for ROCm math/AI libraries

### Runtime
- [HIP Runtime](https://rocm.docs.amd.com/projects/HIP/en/latest/) — HIP C++ runtime API: device management, memory, streams, events, graphs
- [HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) — Thread hierarchy, memory model, execution model
- [ROCr (HSA Runtime)](https://github.com/ROCm/ROCR-Runtime) — Low-level HSA-based runtime for AMD GPUs, manages AQL queues
- [ROCm SMI](https://rocm.docs.amd.com/projects/rocm_smi_lib/) — System Management Interface for GPU monitoring

### Driver / Firmware
- [amdgpu Kernel Module](https://docs.kernel.org/gpu/amdgpu/index.html) — Open-source Linux DRM/KMS kernel driver for AMD GPUs (in-tree)
- [KFD (Kernel Fusion Driver)](https://github.com/ROCm/ROCK-Kernel-Driver) — Compute-oriented kernel driver integrated into amdgpu, manages HSA queues
- [ROCk (ROCK-Kernel-Driver)](https://github.com/ROCm/ROCK-Kernel-Driver) — ROCm's fork of amdgpu+KFD with compute enhancements
- [AMD GPU Firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git) — Open firmware blobs for AMD GPUs (in linux-firmware)

### Communication
- [RCCL](https://github.com/ROCm/rccl) — ROCm Communication Collectives Library (AllReduce, AllGather, etc.) — AMD's NCCL equivalent
- [RCCL Documentation](https://rocm.docs.amd.com/projects/rccl/) — Multi-GPU/multi-node collective operations
- [AMD GPU-aware MPI](https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html) — GPU-direct MPI support with ROCm

### Assembler / ISA
- [AMD Instinct MI300 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — CDNA3 instruction set architecture specification
- [AMDGPU Assembly (GFX ISA)](https://gpuopen.com/learn/amdgcn-assembly/) — Low-level assembly programming guide for AMD GPUs
- [AMDGPU Backend LLVM Docs](https://llvm.org/docs/AMDGPUUsage.html) — Instruction encoding, register model, address spaces
- [AMD GPU Architecture Programming Docs](https://gpuopen.com/amd-gpu-architecture-programming-documentation/) — ISA reference manuals for all AMD GPU families

## Hardware Architecture

### Compute Engine
- [MI300 Series Microarchitecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html) — CDNA3: XCD with 38 active CUs, Matrix Cores, 304 CUs total on MI300X
- [CDNA Architecture (Wikipedia)](https://en.wikipedia.org/wiki/CDNA_(microarchitecture)) — CDNA1/2/3 evolution, compute-focused design
- [MI350 Series](https://www.amd.com/en/products/accelerators/instinct/mi350.html) — CDNA4 architecture, MI350X/MI355X

### Data Path
- [MI300 Architecture Docs](https://instinct.docs.amd.com/latest/gpu-arch/mi300.html) — Wavefront execution (64 threads), SIMD units, instruction pipeline
- [ROCm GPU Memory Model](https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html) — Memory ordering, cache coherence, address spaces

### On-chip Memory
- [GPU Memory Documentation](https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html) — LDS (64KB/CU), L1 vector/scalar cache (16KB each), L2 cache (8MB/XCD)
- [Infinity Cache](https://www.amd.com/en/technologies/infinity-cache) — 256MB last-level cache on MI300 (up to 17.2 TB/s)

### Off-chip Memory
- [MI300X Specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) — 192GB HBM3 (8 stacks), 5.3 TB/s aggregate bandwidth
- [MI350X Specifications](https://www.amd.com/en/products/accelerators/instinct/mi350.html) — 288GB HBM3e

### Host Interface / Package
- [MI300 Chiplet Architecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html) — 8 XCDs + 4 IODs on advanced packaging, PCIe Gen5
- [MI300A APU](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html) — CPU+GPU unified memory via Infinity Fabric, coherent shared HBM

### Scale-up Interconnect
- [Infinity Fabric (AMD)](https://arxiv.org/html/2410.00801v1) — Intra-node GPU-to-GPU interconnect, cache-coherent, XGMI links
- [MI300 Multi-GPU Communication](https://www.memsys.io/wp-content/uploads/ninja-forms/5/Memsys_multigpu_MI300A-6.pdf) — IF bandwidth: quad-link 200+200 GB/s, dual-link 100+100 GB/s

### Scale-out Interconnect
- [AMD Pensando DPU](https://www.amd.com/en/products/accelerators/pensando.html) — SmartNIC/DPU for data center networking
- [GPU-aware MPI with ROCm](https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html) — Inter-node GPU communication via InfiniBand/RoCE

## Other Resources
- [ROCm Documentation Hub](https://rocm.docs.amd.com/) — Central documentation for all ROCm components
- [AMD Instinct Documentation](https://instinct.docs.amd.com/) — Hardware-focused documentation for Instinct GPUs
- [AMD GPUOpen](https://gpuopen.com/) — Open-source tools, SDKs, and documentation
- [ROCm GitHub Organization](https://github.com/ROCm) — All ROCm open-source repositories
- [AMD AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/) — Tutorials and guides for AI on AMD GPUs

---

## New Resources — CDNA 5 / MI455X / Helios / ROCm 7.14 (added 2026-08-08)

### Hardware Architecture — CDNA 5 / MI455X
- [Introducing AMD CDNA 5 and the AMD Helios Rackscale Solution](https://rocm.blogs.amd.com/ecosystems-and-partners/cdna5-helios/README.html) — AMD ROCm blog, 2026-08-04. Primary vendor source for the CDNA 5 architecture name, Wave32 execution, 2.91x memory bandwidth, E5M3 scale format, 4-bit Tensor LUT instruction, Tensor Data Movers, UALoE scale-up, 260 TB/s rack scale-up, 43 TB/s rack scale-out
- [AAI 2026: AMD Delivers Full-Stack Compute for the Agentic AI Era](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era) — AMD Investor Relations press release, 2026-07-23. Launch, production status, MI430X 288 TFLOPS FP64, customer commitments, MI500 in 2027
- [AMD's Instinct MI455X: Aiming for the Top](https://chipsandcheese.com/p/amds-instinct-mi455x-aiming-for-the) — Chips and Cheese deep-dive, 2026-07-23. Shader organization (8 XCDs x 32 of 34 WGPs, 2 SEs/XCD), Wave32 SIMD32 dual-issue, 320 KB LDS/WGP, 128 KB L1/WGP, 192 MB L2 on 2 FCDs at 54 TB/s, 36 x 400 Gb/s UALoE, 256 GB/s Infinity Fabric host link, CoWoS-L
- [AMD Helios Architecture Deep Dive: AMD + Broadcom Hardware Combined](https://www.servethehome.com/amd-helios-architecture-deep-dive-amd-broadcom-hardware-combined/) — ServeTheHome. Rack composition, 31 TB HBM4, 2.9 EFLOPS MXFP4, ~1.7 PB/s aggregate HBM BW
- [AMD Advancing AI 2026 Keynote Live Coverage](https://www.servethehome.com/amd-advancing-ai-2026-keynote-live-coverage/) — ServeTheHome. Keynote date, process-node slide, MI430X H1 2027, CDNA 6 attached to MI500
- [AMD EPYC Venice, Instinct MI455X and Helios Hardware on Display for First Time at CES 2026](https://www.servethehome.com/amds-epyc-venice-instinct-mi455x-helios-hardware-on-display-for-first-time-at-ces-2026/) — ServeTheHome, 2026-01-07. Establishes that the hardware pre-dates the July 2026 architecture disclosure
- [AMD Instinct MI455X and Helios](https://www.phoronix.com/news/AMD-Instinct-MI455X-Helios) — Phoronix. Shipment ramp timing
- [Oracle Leads AI Innovation with AMD "Altair" MI450 GPUs and Helios Racks](https://www.nextplatform.com/compute/2025/10/14/oracle-leads-ai-innovation-with-amd-altair-mi450-gpus-and-helios-racks/1632443) — The Next Platform, 2025-10-14. Original MI450/Helios disclosure
- [AMD Says Helios Racks and MI400-Series GPUs On Track for 2H 2026](https://www.nextplatform.com/compute/2026/02/23/amd-says-helios-racks-and-mi400-series-gpus-on-track-for-2h-2026/4092199) — The Next Platform, 2026-02-23
- [AMD Instinct — Wikipedia](https://en.wikipedia.org/wiki/AMD_Instinct) — SKU/date cross-check; MI350P announcement date 2026-05-07
- [AMD Advancing AI event page](https://www.amd.com/en/corporate/events/advancing-ai.html)
- [AMD Comes to Play: A Quick Take on Advancing AI 2026](https://moorinsightsstrategy.com/field-notes/amd-comes-to-play-a-quick-take-on-advancing-ai-2026/) — Moor Insights & Strategy. *Use with care: source of several figures that failed verification (MI350P 144 GB, MI440X).*

### Software Stack — ROCm 7.14 / Primus / Hyperloom
- [ROCm 7.14 release blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html) — AMD ROCm blog, 2026-07-15. TheRock, HIP Execution Context APIs, batch memory APIs, RCCL hierarchical AllGather, ROCprofiler-SDK SPM
- [ROCm/ROCm release rocm-7.14.0](https://github.com/ROCm/ROCm/releases/tag/rocm-7.14.0) — GitHub release tag, 2026-07-16
- [ROCm release list](https://github.com/ROCm/ROCm/releases) — shows the 7.2.x → 7.9.0 preview → 7.10–7.14 versioning discontinuity
- [ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) — **authoritative for gfx targets; shows no CDNA 5 entry as of 2026-08-08**
- [AMD-AIG-AIMA/Primus](https://github.com/AMD-AIG-AIMA/Primus) — AMD's open training framework (Megatron-LM / TorchTitan / JAX MaxText backends), v26.4
- [Primus Tuning Agent](https://rocm.blogs.amd.com/software-tools-optimization/primus-tuning-agent/README.html) — AMD ROCm blog, 2026-07-06
- [Hyperloom — Autonomous Agentic Inference Optimization for AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom/README.html) — AMD ROCm blog, 2026-07-23
- [AMD ROCm Blogs index](https://rocm.blogs.amd.com/blog.html)

### Benchmarks
- [MLPerf Training v6.0 on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html) — AMD ROCm blog; first multi-node AMD training submission, first Primus use, first MXFP4 training recipe
- [MLPerf Inference v6.0 on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html) — AMD ROCm blog, 2026-04-01. *Throughput figures unverified in this pass.*
- [MLCommons MLPerf Training benchmark results](https://mlcommons.org/benchmarks/training/)
