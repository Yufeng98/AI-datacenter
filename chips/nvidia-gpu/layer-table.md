# NVIDIA GPU Layer Mapping Table

*as_of: 2026-08-08*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (torch.cuda, Autograd, CUDAGraph, CachingAllocator) | confirmed | cuda-programming-guide, cuda-runtime, cccl-cublas |
| Framework Integration | JAX (jax.numpy, XLA backend, Pallas) | confirmed | cuda-programming-guide, triton, cudnn |
| Framework Integration | TensorFlow (tf.device GPU, StreamExecutor) | confirmed | cuda-programming-guide, cudnn, cuda-runtime |
| Framework Integration | CUDA C/C++ Language Extensions (__global__, __shared__, <<<>>>) | confirmed | cuda-programming-guide |
| Framework Integration | Cooperative Groups (warp/block/cluster/grid sync) | confirmed | cuda-programming-guide |
| Compiler / IR | nvcc / NVVM (CUDA C++ to PTX/SASS compiler) | confirmed | cuda-programming-guide, ptx-isa |
| Compiler / IR | Triton (Python DSL, MLIR pipeline, auto MMA selection) | confirmed | triton |
| Compiler / IR | XLA (HLO to LLVM to PTX, for JAX/TF) | confirmed | triton, cudnn |
| Compiler / IR | TensorRT (graph optimizer, INT8/FP8 quantization) | confirmed | cudnn |
| Compiler / IR | CUDA Tile IR (MLIR-based tile IR + compiler targeting Tensor Core units; TILE-IR AS component 13.3.36 in CUDA 13.3 Update 1) | confirmed | software-stack |
| Compiler / IR | cuTile (Python tile-kernel DSL lowering to Tile IR; tileiras 13.2 supports Blackwell and Ampere/Ada only — Hopper deferred; needs CUDA 13.1+ / driver r580+) | confirmed | software-stack |
| Op Library | cuDNN v9 (Conv, Attention, Norm; Graph API with 50-op fusion) | confirmed | cudnn |
| Op Library | cuBLAS / cublasLt (GEMM, BLAS L1-3, epilogue fusion, FP4-FP64) | confirmed | cccl-cublas |
| Kernel Library | CUTLASS / CuTe (GEMM/Conv templates, EVT epilogue, CuTe DSL) | confirmed | cutlass |
| Kernel Library | CUB (DeviceReduce, DeviceScan, DeviceRadixSort; arch-aware tuning) | confirmed | cccl-cublas |
| Kernel Library | Thrust (STL-like parallel algorithms, multi-backend) | confirmed | cccl-cublas |
| Kernel Library | libcu++ (CUDA C++ Standard Library, device-side atomics, barriers, PTX intrinsics) | confirmed | cccl-cublas |
| Kernel Library | Triton Kernels (python/triton_kernels: matmul, attention) | confirmed | triton |
| Kernel Library | FlashAttention (IO-aware attention, SMEM tiling) | confirmed | cutlass, cudnn |
| Kernel Library | cuTile kernels (`cuda.tile` stable core API + separately marked experimental features; no GA/stability announcement in any release note) | confirmed | software-stack |
| Runtime | CUDA Runtime API (libcudart.so: Streams, Events, Graphs, UVM) | confirmed | cuda-runtime, cuda-programming-guide |
| Runtime | CUDA Driver API (libcuda.so: Contexts, Modules, VMM, JIT) | confirmed | cuda-runtime, cuda-programming-guide |
| Runtime | MPS (Multi-Process Service, Hyper-Q sharing) | confirmed | cuda-runtime |
| Runtime | MIG (Multi-Instance GPU, hardware partitioning) | confirmed | cuda-runtime |
| Runtime | CUDA Graphs (DAG capture/replay, sub-ms dispatch) | confirmed | cuda-runtime, cuda-programming-guide |
| Driver / Firmware | nvidia.ko (Open Kernel Module, RM proxy, ioctl, user-mode doorbells) | confirmed | open-gpu-kernel-modules |
| Driver / Firmware | GSP Firmware (on-GPU RISC-V Resource Manager, shared-mem RPC) | confirmed | open-gpu-kernel-modules |
| Driver / Firmware | nvidia-uvm.ko (Unified Virtual Memory, GPU page fault handler) | confirmed | open-gpu-kernel-modules, cuda-runtime |
| Driver / Firmware | nvidia-peermem.ko (GPUDirect RDMA peer memory registration) | confirmed | open-gpu-kernel-modules |
| Driver / Firmware | Fabric Manager (NVSwitch topology management) | inferred | nccl, open-gpu-kernel-modules |
| Communication | NCCL (AllReduce, AllGather, ReduceScatter; Ring/Tree/NVLS/CollNet) | confirmed | nccl |
| Communication | GPUDirect RDMA (NIC-to-GPU direct DMA) | confirmed | nccl, open-gpu-kernel-modules |
| Communication | GIN / GDAKI (GPU Initiated Networking, SM-posted IB requests) | confirmed | nccl |
| Communication | NVLS (NVLink SHARP, in-switch hardware AllReduce) | confirmed | nccl |
| Communication | MNNVL (Multi-Node NVLink, cross-node NVSwitch P2P) | confirmed | nccl |
| Communication | NCCL zero-SM hierarchical collectives (AllGather, All2all; RMA CPU proxy inter-node + Copy Engines intra-node; `NCCL_CTA_POLICY_ZERO`; NCCL 2.30.7-1, 2026-06-04) | confirmed | software-stack, nccl |
| Communication | NCCL elastic buffers (multi-segment windows, host-memory remainder) + TMA in symmetric kernels (`NCCL_SYM_TMA_ENABLE=1`; NCCL 2.30.3-1) | confirmed | software-stack, nccl |
| Communication | NCCL EP (MoE dispatch/combine on the NCCL Device API — LSA + GIN; CUDA-Graph-compatible handles; `nccl-ep-v0.1.0`, 2026-06-08) | confirmed | software-stack, nccl |
| Communication | nccl4py (Python API; `nccl.ep`, `nccl.core.device.cute` CuTeDSL device kernels, free-threaded CPython; v0.3.1, 2026-06-11) | confirmed | software-stack, nccl |
| Communication | GPI backend for GIN; MPS with MLOPart (≤2 ranks/physical GPU) | experimental | software-stack, nccl |
| Assembler / ISA | PTX v8.7-9.3 (Virtual ISA: wgmma, tcgen05, cp.async.bulk; "some Rubin capabilities" in the CUDA 13.4 developer preview only) | confirmed | ptx-isa, cuda-programming-guide, software-stack |
| Assembler / ISA | Tile IR + `tileiras` / CUDA TILE-IR AS (tile-level virtual ISA shipping alongside PTX since CUDA 13.2 GA) | confirmed | software-stack |
| Assembler / ISA | ptxas (PTX Assembler, register alloc, scheduling) | confirmed | ptx-isa |
| Assembler / ISA | SASS (Native microcode per sm_XX, not fully documented) | confirmed | ptx-isa |
| Assembler / ISA | Fat binary (PTX + SASS container for forward compat) | confirmed | ptx-isa, cuda-programming-guide |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | SM: 128 CUDA Cores + 4 Tensor Cores per SM (H100: 132 SMs, B200: 296 SMs) | confirmed | hw-architecture, cuda-programming-guide |
| Compute Engine | 4th-gen Tensor Cores (Hopper): FP8/BF16/TF32/FP64 | confirmed | hw-architecture, cutlass, ptx-isa |
| Compute Engine | 5th-gen Tensor Cores (Blackwell): +MXFP4/MXFP6, TMEM, 2-SM MMA | confirmed | hw-architecture, cutlass, ptx-isa |
| Compute Engine | 6th-gen Tensor Cores (Rubin, H2 2026): TSMC 3nm; 50 PFLOPS NVFP4 inference, 35 PFLOPS NVFP4 training, 17.5 PFLOPS FP8-FP6 training per GPU (NVIDIA preliminary figures) | confirmed | hw-architecture, rubin-platform |
| Compute Engine | Rubin Transformer Engine with adaptive compression for NVFP4 inference (no quantified figures published; product-page wording, page undated) | confirmed (wording) | hw-architecture, rubin-platform |
| Data Path | SIMT: 32-thread warps, 4 warp schedulers/SM, dual-issue | confirmed | hw-architecture, cuda-programming-guide |
| Data Path | TMA (Tensor Memory Accelerator, Hopper+): async 5D tensor DMA | confirmed | hw-architecture, cuda-programming-guide, cutlass |
| Data Path | Thread Block Clusters (Hopper+): up to 8 CTAs, distributed SMEM | confirmed | hw-architecture, cuda-programming-guide |
| On-chip Memory | Register File: 256 KB/SM, 32 banks | confirmed | hw-architecture |
| On-chip Memory | Shared Memory/L1: H100 228 KB SMEM/SM (256 KB pool) | confirmed | hw-architecture, cuda-programming-guide |
| On-chip Memory | L2 Cache: H100 50 MB, B200 ~96-128 MB, residency controls | confirmed | hw-architecture |
| On-chip Memory | TMEM (Blackwell): dedicated MMA accumulator buffer | confirmed | hw-architecture, ptx-isa, cutlass |
| Off-chip Memory | HBM3 (H100): 80 GB, 3.35 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM3e (H200): 141 GB, 4.8 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM3e (B200): 192 GB, 8.0 TB/s | confirmed | hw-architecture |
| Off-chip Memory | NV-HBI (Blackwell die-to-die): 10 TB/s | confirmed | hw-architecture |
| Off-chip Memory | HBM4 (Rubin, H2 2026): 288 GB per GPU, 22 TB/s bandwidth (2.8x Blackwell HBM3e) | confirmed | hw-architecture, rubin-platform |
| Host Interface / Package | PCIe Gen 5.0 x16: ~128 GB/s bidir | confirmed | hw-architecture |
| Host Interface / Package | NVLink-C2C: 900 GB/s bidir coherent (Grace Hopper/Blackwell) | confirmed | hw-architecture |
| Host Interface / Package | Blackwell: dual chiplet, TSMC CoWoS-L 2.5D, 208B transistors | confirmed | hw-architecture |
| Scale-up Interconnect | NVLink 4th gen (Hopper): 900 GB/s/GPU | confirmed | hw-architecture, nccl |
| Scale-up Interconnect | NVLink 5th gen (Blackwell): 1.8 TB/s/GPU | confirmed | hw-architecture, nccl |
| Scale-up Interconnect | NVSwitch 5th gen: 64 ports, 7.2 TB/s, NVLink SHARP | confirmed | hw-architecture, nccl |
| Scale-up Interconnect | GB200 NVL72: 72 GPUs, 130 TB/s, 13.4 TB HBM | confirmed | hw-architecture, nccl |
| Scale-up Interconnect | NVLink 6th gen (Rubin, H2 2026): 3.6 TB/s/GPU (2x NVLink 5) | confirmed | hw-architecture, rubin-platform |
| Scale-up Interconnect | VR NVL72 (Rubin): 72 GPUs, 260 TB/s scale-up BW, 20.7 TB HBM4, 3.6 EFLOPS NVFP4 per rack | confirmed | hw-architecture, rubin-platform |
| Scale-up Interconnect | NVLink domain ladder: NVL72 (72 GPUs, 1 rack) → Rubin Ultra NVL576 (**8 MGX NVL racks × 72 GPUs**, two-layer all-to-all) → Kyber NVL144 (144 GPUs/rack) → NVL1152 (8 Kyber racks) | confirmed | hw-architecture |
| Scale-out Interconnect | ConnectX-7 NDR InfiniBand: 400 Gb/s, dual-port 800 Gb/s | confirmed | hw-architecture, nccl |
| Scale-out Interconnect | Vera Rubin compute tray: 8× ConnectX-9 SuperNIC + 1× BlueField-4 DPU (BlueField-4 combines a Vera CPU with a ConnectX-9 SuperNIC); per-port rate not disclosed | confirmed | hw-architecture |
| Scale-out Interconnect | Spectrum-6 SPX Ethernet switch: 102.4 Tb/s, 512 lanes, 200 Gb/s co-packaged optics | confirmed | hw-architecture |
| Scale-out Interconnect | RoCE v2 400 GbE (ConnectX-7) | confirmed | hw-architecture |
| Scale-out Interconnect | GPUDirect RDMA: NIC direct to GPU HBM | confirmed | hw-architecture, nccl, open-gpu-kernel-modules |
| Scale-out Interconnect | SHARP: in-network AllReduce on IB switches | confirmed | hw-architecture, nccl |
