# NVIDIA GPU Software Stack & Hardware Resources

*as_of: 2026-08-08*
*device_class: GPU*
*seeds: https://developer.nvidia.com/cuda-toolkit, https://github.com/NVIDIA/cutlass, https://github.com/NVIDIA/nccl*

## Software Stack

### Framework Integration
- [PyTorch CUDA Backend](https://pytorch.org/docs/stable/cuda.html) — PyTorch's native CUDA device backend, torch.cuda module for GPU tensor operations
- [JAX/XLA CUDA Backend](https://github.com/jax-ml/jax) — JAX uses XLA compiler with CUDA backend for GPU execution
- [TensorFlow CUDA Integration](https://www.tensorflow.org/install/gpu) — TensorFlow GPU support via CUDA and cuDNN
- [ONNX Runtime CUDA EP](https://onnxruntime.ai/) — ONNX Runtime CUDA execution provider for inference
- [torch.compile / TorchDynamo](https://pytorch.org/docs/stable/torch.compiler.html) — PyTorch 2.x compiler stack capturing Python frames, lowering via TorchInductor to GPU kernels

### Compiler / IR
- [NVCC Compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) — NVIDIA CUDA C/C++ compiler driver, compiles CUDA code to PTX then SASS
- [NVVM IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/) — LLVM-based intermediate representation used by NVCC
- [Triton Compiler](https://github.com/triton-lang/triton) — OpenAI's Python-embedded DSL and MLIR-based compiler for GPU kernels, generates PTX
- [TensorRT](https://developer.nvidia.com/tensorrt) — Deep learning inference optimizer and compiler, graph-level optimizations
- [CUDA Tile IR](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton) — Tile-based GPU programming model IR, backend for Triton on Blackwell
- [XLA CUDA Backend](https://github.com/openxla/xla) — XLA compiler backend targeting NVIDIA GPUs via LLVM/PTX
- [TorchInductor](https://pytorch.org/docs/stable/torch.compiler.html) — PyTorch's code generation backend, generates Triton kernels for GPU
- [NVRTC](https://docs.nvidia.com/cuda/nvrtc/) — Runtime CUDA compilation library, JIT compile CUDA C++ to PTX at runtime
- [nvJitLink](https://docs.nvidia.com/cuda/nvjitlink/) — JIT linking library for PTX/SASS modules
- [libNVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) — NVVM IR to PTX compiler library

### Op Library
- [cuDNN](https://developer.nvidia.com/cudnn) — GPU-accelerated primitives for deep neural networks (conv, attention, matmul, pooling, normalization)
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) — C++/Python frontend for cuDNN Graph API (recommended interface)
- [cuBLAS](https://developer.nvidia.com/cublas) — GPU-accelerated BLAS (dense linear algebra, GEMM)
- [cuFFT](https://developer.nvidia.com/cufft) — GPU-accelerated Fast Fourier Transforms
- [cuSPARSE](https://developer.nvidia.com/cusparse) — GPU-accelerated sparse matrix operations
- [cuSOLVER](https://developer.nvidia.com/cusolver) — GPU-accelerated dense/sparse direct solvers
- [cuRAND](https://developer.nvidia.com/curand) — GPU-accelerated random number generation
- [CUDA-X Libraries](https://developer.nvidia.com/cuda/cuda-x-libraries) — Full suite of GPU-accelerated domain libraries
- [cuTENSOR](https://developer.nvidia.com/cutensor) — GPU-accelerated tensor linear algebra library
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) — Library for accelerating Transformer models with FP8 on Hopper/Blackwell

### Kernel Library
- [CUTLASS](https://github.com/NVIDIA/cutlass) — CUDA Templates for High-Performance Linear Algebra, template-based GEMM/conv kernels
- [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) — Layout and Tensor abstraction within CUTLASS 3.x for flexible tiling
- [Triton Kernels](https://triton-lang.org/) — Python DSL for writing custom GPU kernels, auto-optimizes tiling and memory access
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — IO-aware exact attention kernels optimized for GPU memory hierarchy
- [CCCL (CUB/Thrust/libcu++)](https://nvidia.github.io/cccl/) — CUDA C++ Core Libraries: CUB (device-wide primitives), Thrust (parallel algorithms), libcu++ (C++ standard library for CUDA)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — Optimized LLM inference kernels built on TensorRT
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) — Reference kernel implementations demonstrating CUDA features

### Runtime
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) — High-level C API for GPU device management, memory allocation, kernel launch, streams
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) — Low-level C API for fine-grained GPU control, module loading, virtual memory management
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — Capture and replay GPU work as graphs for reduced launch overhead
- [MPS (Multi-Process Service)](https://docs.nvidia.com/deploy/mps/) — Allows multiple processes to share a single GPU context, reducing launch overhead
- [MIG (Multi-Instance GPU)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — Partition a GPU into isolated instances for multi-tenant use
- [NVML](https://docs.nvidia.com/deploy/nvml-api/) — NVIDIA Management Library C API for GPU monitoring and management

### Driver / Firmware
- [NVIDIA Linux Open GPU Kernel Modules](https://github.com/NVIDIA/open-gpu-kernel-modules) — Open-source Linux kernel module for NVIDIA GPUs
- [NVIDIA Data Center GPU Driver](https://docs.nvidia.com/datacenter/tesla/) — Proprietary kernel-mode driver for data center GPUs
- [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/) — System management interface for GPU monitoring and configuration
- [GSP Firmware](https://docs.nvidia.com/datacenter/tesla/) — GPU System Processor firmware, loaded by driver for GPU management
- [Fabric Manager](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/) — User-space service managing NVSwitch-based GPU fabrics, required for NVLink Switch systems

### Communication
- [NCCL](https://github.com/NVIDIA/nccl) — Optimized collective communication primitives (AllReduce, AllGather, etc.) for multi-GPU/multi-node, topology-aware
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) — User guide for NCCL v2.29.7
- [NVIDIA SHARP](https://docs.nvidia.com/networking/display/shaborv261) — In-network computing for accelerating collectives on InfiniBand
- [NVSHMEM](https://developer.nvidia.com/nvshmem) — NVIDIA implementation of OpenSHMEM for GPU-initiated communication, PGAS model
- [GDRCopy](https://github.com/NVIDIA/gdrcopy) — Low-latency GPU memory copy library using GPUDirect RDMA

### Assembler / ISA
- [PTX ISA Specification](https://docs.nvidia.com/cuda/parallel-thread-execution/) — Parallel Thread Execution virtual ISA (v9.2), stable target for compilers
- [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/) — Embedding PTX instructions in CUDA C/C++ code
- [SASS (Streaming Assembly)](https://docs.nvidia.com/cuda/cuda-binary-utilities/) — Native GPU machine code, architecture-specific (cuobjdump for disassembly)
- [ptxas Assembler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) — PTX-to-SASS assembler, part of CUDA Toolkit

## Hardware Architecture

### Compute Engine
- [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — Latest GPU arch: dual-die chiplet, 5th gen Tensor Cores, unified INT32/FP32 pipelines
- [Blackwell Microbenchmarks](https://arxiv.org/abs/2507.10789) — In-depth microarchitectural analysis of Blackwell
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core) — H100 GPU: 80B transistors, 4th gen Tensor Cores, TMA
- [SM Architecture (Modal GPU Glossary)](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor) — Detailed breakdown of Streaming Multiprocessor internals
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) — A100 SM architecture, Tensor Core details, warp scheduling

### Data Path
- [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/) — Warp scheduler, instruction dispatch, pipeline details
- [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/) — Independent thread scheduling, sub-warp execution
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Comprehensive reference for execution model, thread hierarchy, synchronization

### On-chip Memory
- [GPU Memory Hierarchy (Cornell)](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels) — Register file, L1/shared memory (configurable split, ~128KB/SM), L2 cache
- [GPU Cache Hierarchy Analysis](https://charlesgrassi.dev/blog/gpu-cache-hierarchy/) — L1, L2, and VRAM interaction patterns
- [Shared Memory and L1 Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) — SMEM and L1 share physical SRAM, software-configurable split

### Off-chip Memory
- [HBM (Wikipedia)](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) — High Bandwidth Memory technology used in data center GPUs
- [H100 Memory Specs](https://www.nvidia.com/en-us/data-center/h100/) — 80GB HBM3, 3.35 TB/s bandwidth
- [B200 Memory Specs](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — 192GB HBM3e per GPU, 8 TB/s aggregate bandwidth

### Host Interface / Package
- [Blackwell Chiplet Design](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)) — Dual-die 800mm² chiplet connected as single GPU, PCIe Gen6
- [Grace Hopper Superchip](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) — CPU+GPU coherent integration via NVLink-C2C

### Scale-up Interconnect
- [NVLink & NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/) — 5th gen NVLink: 1.8 TB/s bidirectional per GPU, NVSwitch for all-to-all
- [GB200 NVL72](https://nebius.com/blog/posts/leveraging-nvidia-gb200-nvl72-gpu-interconnect) — 72 GPUs connected via NVLink as single logical GPU
- [NVLink Wikipedia](https://en.wikipedia.org/wiki/NVLink) — History and specs of NVLink generations

### Scale-out Interconnect
- [NVIDIA InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/) — ConnectX-7/BlueField-3 for 400Gb/s inter-node networking
- [NVIDIA Spectrum Ethernet](https://www.nvidia.com/en-us/networking/ethernet/) — RoCE-capable Ethernet switches for GPU clusters
- [GPUDirect RDMA](https://developer.nvidia.com/gpudirect) — Direct GPU-to-GPU data transfer across nodes bypassing CPU

## Other Resources
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Comprehensive reference for CUDA programming model
- [NVIDIA Developer Documentation](https://docs.nvidia.com/) — Central hub for all NVIDIA technical documentation
- [Nsight Systems & Compute](https://developer.nvidia.com/nsight-systems) — GPU profiling and debugging tools
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/) — Best practices for DL training/inference on NVIDIA GPUs

---

## Resources Added 2026-08-08 (scan window 2026-04-05 → 2026-08-08)

*Newly discovered or newly required resources. Version-pinned entries above (NCCL v2.29.7, PTX ISA v9.2, cuBLAS v13.2) are superseded by the pins recorded here; the older entries are retained for provenance.*

### Compiler / IR — tile programming stack
- [NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile) — MLIR-based Tile IR and compiler infrastructure targeting NVIDIA tensor core units (repo created 2025-11-05; v13.1.0 2026-01-14 → v13.3.3 2026-07-22)
- [NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python) — cuTile, the Python tile-kernel DSL layered on Tile IR (repo created 2025-06-13; v1.5.0 2026-07-07)
- [cutile-python README](https://raw.githubusercontent.com/NVIDIA/cutile-python/main/README.md) — stable-vs-experimental API split; tileiras 13.2 supports Blackwell and Ampere/Ada only (Hopper deferred); requires CUDA Toolkit 13.1+ and driver r580+

### Compiler / IR — CUDA Toolkit release notes
- [CUDA Toolkit Release Notes (current GA — 13.3 Update 1)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) — GA component table; CUDA TILE-IR AS at 13.3.36
- [CUDA 13.4 Developer Preview Release Notes (PDF, dateline 2026-07-16)](https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/CUDA_Toolkit_Release_Notes.pdf) — PTX ISA with "some Rubin capabilities"; Tile/TileIR compilation and inspection; driver unbundled from the toolkit; R616+ required. **Pre-release: performance data from it must not be published or used to characterize NVIDIA hardware — cite for feature existence only**
- [CUDA 13.2 Archived Release Notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html) — cuBLASLt Grouped GEMM MXFP8 on CC 10.x/11.0; FP64 fixed-point emulation in syrk/herk

### Communication
- [NVIDIA/nccl releases](https://github.com/NVIDIA/nccl/releases) — v2.30.3-1 (2026-04-15), v2.30.4-1 (2026-04-22), v2.30.7-1 (2026-06-04); also the `nccl-ep-v0.1.0` (2026-06-08) and `nccl4py-v0.3.1` (2026-06-11) tags, which live inside this repository rather than in standalone projects
- [NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions) — "Communication patterns for AI, built on top of NCCL device and host APIs"; repo created 2026-07-07, actively developed as of 2026-08-08

### Assembler / ISA
- [PTX ISA Specification (v9.3)](https://docs.nvidia.com/cuda/parallel-thread-execution/) — supersedes the v9.2 pin recorded above

### Hardware / platform
- [Vera Rubin Ramps Into Full Production — NVIDIA Newsroom (2026-05-31)](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory) — production status, "shipments … starting this fall", adopter and 15-name OEM lists
- [NVIDIA Vera Rubin POD: Seven Chips, Five Rack-Scale Systems, One AI Supercomputer — NVIDIA Technical Blog (2026-03-16)](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/) — NVLink domain ladder (NVL72 → NVL576 = 8 racks × 72 GPUs → Kyber NVL144 → NVL1152); Spectrum-6 SPX 102.4 Tb/s; 8× ConnectX-9 + 1× BlueField-4 per compute tray
- [NVIDIA Vera Rubin NVL72 Product Page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) — full per-GPU and per-rack spec table with NVIDIA's "Preliminary information" disclaimer
- [NVIDIA Rubin Platform Page](https://www.nvidia.com/en-us/data-center/technologies/rubin/) — "new Transformer Engine with adaptive compression"; "Rubin GPUs for HBM and LPUs for SRAM" (NVIDIA Groq 3 LPX)

### Benchmarks
- [MLPerf Training v6.0 Results — MLCommons (2026-06-16)](https://mlcommons.org/2026/06/mlperf-training-v6-0-results/) — adds DeepSeek-V3 671B and GPT-OSS 20B; 24 submitters
- [NVIDIA Blackwell Tops MLPerf Training v6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/) — headline times (2.02 / 7.07 / 7.43 / 4.46 min) and NeMo container 26.06

### Checked and rejected as sources
- CNBC/SemiAnalysis report of a Kyber slip to 2028 (2026-07-06) — article not retrievable (HTTP 403), no NVIDIA statement; not used
- Hot Chips 38 (Aug 23–25, 2026) program — disclosure scheduled, content not yet public; no talk title used as the source of a spec
