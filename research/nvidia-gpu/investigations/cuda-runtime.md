# CUDA Runtime and Driver API

## Overview

The CUDA Runtime API (`libcudart`) is the primary GPU management layer that sits between compiled CUDA application code and the lower-level CUDA Driver API (`libcuda`). It abstracts the complexity of GPU resource management—device selection, memory allocation, kernel dispatch, synchronization—behind a C/C++ API that the `nvcc` compiler toolchain targets directly. The `<<<>>>` kernel launch syntax is syntactic sugar that `nvcc` transforms into `cudaLaunchKernel()` calls at compile time.

The Runtime API is intentionally high-level: it manages primary contexts automatically, initializes lazily, and handles module loading transparently. Developers rarely need to touch the Driver API unless they require fine-grained control over context lifetimes, dynamic module loading, or advanced virtual memory management. The split between the two APIs reflects a deliberate layering: Runtime API calls translate into Driver API calls, which in turn translate into `ioctl` calls on the `/dev/nvidia*` and `/dev/nvidia-uvm` kernel device files exposed by the NVIDIA kernel module (`nvidia.ko`) and the UVM driver (`nvidia-uvm.ko`).

---

## Architecture

### Runtime API vs Driver API Split

| Dimension | Runtime API (`libcudart`) | Driver API (`libcuda`) |
|---|---|---|
| Header | `cuda_runtime.h` | `cuda.h` |
| Library | `libcudart.so` (CUDA Toolkit) | `libcuda.so` (NVIDIA Driver) |
| Abstraction level | High — automatic context, module load | Low — explicit context stack, module handle |
| Context model | One primary context per device per process | Full `cuCtxCreate`/`cuCtxDestroy` stack |
| Module loading | All kernels loaded at init, stay resident | On-demand via `cuModuleLoad`/`cuModuleUnload` |
| Binary compatibility | Tied to Toolkit version | Backward-compatible across driver versions |
| Typical users | Application developers, ML frameworks | Runtimes (e.g., OpenCL, Vulkan Compute, RAPIDS) |

The Runtime API is implemented as a thin translation layer: every `cuda*` call eventually calls the corresponding `cu*` Driver API function. For example, `cudaMalloc` → `cuMemAlloc`, `cudaMemcpy` → `cuMemcpyHtoD`/`cuMemcpyDtoH`, `cudaStreamCreate` → `cuStreamCreate`.

### Memory Management Subsystem

CUDA exposes several memory allocation strategies, each with distinct placement and migration semantics:

- **Device memory** (`cudaMalloc` / `cuMemAlloc`): Allocated in GPU DRAM (HBM on A100/H100). Accessible only from device code. Highest bandwidth for GPU kernels.
- **Pinned (page-locked) host memory** (`cudaMallocHost` / `cuMemAllocHost`): Host DRAM pages locked against OS paging. Enables DMA transfers without a staging copy, which doubles effective PCIe bandwidth. Required for asynchronous `cudaMemcpyAsync`.
- **Mapped memory** (`cudaHostAlloc` with `cudaHostAllocMapped`): Pinned host pages mapped into the GPU virtual address space. GPU can access via PCIe without explicit copies, but at PCIe bandwidth (~32 GB/s vs HBM ~3.35 TB/s on H100).
- **Managed / Unified Memory** (`cudaMallocManaged` / `cuMemAllocManaged`): Single pointer accessible from both CPU and GPU. The UVM driver (`nvidia-uvm.ko`) handles page migrations triggered by hardware page faults. On Pascal+ GPUs, access-counter–driven prefetching and `cudaMemPrefetchAsync` / `cudaMemAdvise` allow the programmer to guide migration policy.
- **Virtual Memory Management API** (Driver API, CUDA 10.2+): Decouples virtual address reservation (`cuMemAddressReserve`) from physical backing allocation (`cuMemCreate`) and mapping (`cuMemMap`). Enables growing allocations without fragmentation—used by frameworks like PyTorch for the CUDA Caching Allocator.

### Stream and Event Model

A **stream** (`cudaStream_t` / `CUstream`) is an ordered work queue on a GPU engine (compute, copy H2D, copy D2H, video decode, etc.). Operations within a stream execute in issue order; operations across streams may overlap if hardware resources permit. The **NULL stream** (default stream, stream 0) is special: it implicitly synchronizes with all other per-thread default streams.

**Events** (`cudaEvent_t` / `CUevent`) are timestamp/synchronization markers that can be recorded into a stream and waited upon by another stream (`cudaStreamWaitEvent`). This enables producer-consumer patterns across streams without full device synchronization.

Key concurrency mechanisms:
- `cudaMemcpyAsync`: Overlaps H2D/D2H transfers with kernel execution on separate copy engines.
- `cudaLaunchKernel` into multiple streams: Overlaps independent kernel workloads.
- Multi-copy engine architectures (Hopper H100): Two copy engines allow simultaneous H2D and D2H transfers.

### CUDA Graphs

CUDA Graphs (introduced in CUDA 10) capture a directed acyclic graph (DAG) of GPU operations and replay it with a single launch call. This eliminates per-operation CPU launch overhead (typically 5–10 µs per kernel for CPU-side API processing).

**Capture workflow:**
1. `cudaStreamBeginCapture(stream)` — puts the stream into capture mode.
2. Issue kernels, memcpys, and events normally.
3. `cudaStreamEndCapture(stream, &graph)` — returns a `cudaGraph_t`.
4. `cudaGraphInstantiate(&graphExec, graph, ...)` — compiles the graph into an executable.
5. `cudaGraphLaunch(graphExec, stream)` — submits the entire graph as one unit.

CUDA Graphs are central to ML framework optimization: PyTorch's `torch.cuda.CUDAGraph`, JAX's compiled function traces, and TensorFlow's XLA-compiled computations all leverage graph capture/replay to minimize Python-side dispatch overhead for iterative training loops.

---

## Data Flow: Typical CUDA Workflow

A standard end-to-end operation traces through the following layers:

```
Application (C++ / Python)
  │
  ▼
1. cudaSetDevice(0)
   └─ Selects GPU 0 as the current device for this host thread.
      Runtime initializes the primary context for device 0
      (via cuDevicePrimaryCtxRetain + cuCtxSetCurrent internally).
      As of CUDA 12.0, initialization is eager; older versions deferred
      it to the first actual CUDA API call.
  │
  ▼
2. cudaMalloc(&d_ptr, size)
   └─ Calls cuMemAlloc(d_ptr, size) in the Driver API.
      Driver issues ioctl(fd_nvidia, NV_ESC_RM_ALLOC_MEMORY, ...) to
      the nvidia.ko kernel module, which maps HBM pages into the
      process virtual address space.
  │
  ▼
3. cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice)
   └─ Calls cuMemcpyHtoD. If h_ptr is not pinned, the Runtime
      creates a temporary pinned staging buffer, copies h_ptr → staging,
      then DMA-transfers staging → GPU HBM via PCIe.
      If h_ptr IS pinned, the DMA goes directly from host memory.
      (Synchronous: blocks host thread until DMA completes.)
  │
  ▼
4. myKernel<<<gridDim, blockDim, sharedMem, stream>>>(d_ptr, ...)
   └─ nvcc transforms this into cudaLaunchKernel(&myKernel, gridDim,
      blockDim, args, sharedMem, stream).
      Runtime calls cuLaunchKernel in the Driver API.
      Driver assembles a command buffer and issues
      ioctl(fd_nvidia, NV_ESC_RM_CONTROL, ...) to push the
      launch descriptor into the GPU's work submission FIFO.
      The GPU scheduler dispatches thread blocks to free SMs.
      Returns immediately to host (asynchronous).
  │
  ▼
5. cudaMemcpy(h_result, d_ptr, size, cudaMemcpyDeviceToHost)
   └─ Synchronous: implicitly calls cudaDeviceSynchronize() on
      the NULL stream before initiating the DtoH DMA.
      Blocks until result data is in host memory.
  │
  ▼
6. cudaFree(d_ptr)
   └─ Calls cuMemFree. Issues ioctl to nvidia.ko to release
      the HBM allocation back to the driver's memory manager.
```

For maximum throughput, steps 3–5 should use async variants (`cudaMemcpyAsync`) with explicit streams so H2D transfers, kernel execution, and D2H transfers overlap across different copy engines and the compute engine.

---

## Hardware Interface

### Runtime → Driver → Kernel Module Mapping

The call chain from application to silicon involves three distinct layers:

1. **CUDA Runtime (`libcudart.so`)**: Translates user-facing API calls into Driver API calls. Manages primary context lifecycle, module loading (loading the `.cubin` fatbinary embedded in the application ELF), and stream/event bookkeeping.

2. **CUDA Driver (`libcuda.so`)**: Implements the canonical Driver API. Manages context stacks, module objects (`CUmodule`), function handles (`CUfunction`), and memory objects. Communicates with the kernel module via `ioctl` on `/dev/nvidia0`, `/dev/nvidiactl`, and (for UVM) `/dev/nvidia-uvm`.

3. **NVIDIA Kernel Module (`nvidia.ko` + `nvidia-uvm.ko`)**: The kernel-space driver. Handles:
   - GPU context creation (hardware context setup in the GPU's channel descriptor table).
   - Memory allocation: maps HBM physical pages via the GPU Memory Manager (RM, Resource Manager).
   - Command buffer submission: pushes DMA/compute commands into per-context FIFO channels.
   - UVM fault handling: the `nvidia-uvm.ko` driver intercepts GPU page fault interrupts, batches faults (up to 256 per batch), resolves them via `cuMemcpy`-like migrations, and replays the faulting instructions.

### UVM Page Fault Path

For managed memory (Unified Memory), GPU access to an absent page triggers:
1. µTLB miss → GPU MMU (GMMU) raises a fault interrupt to the host.
2. `nvidia-uvm.ko` worker thread wakes, reads the fault buffer.
3. Fault is categorized as replayable (from SM, can be held and retried) or non-replayable (from Copy Engine, must abort).
4. Driver migrates the page: maps it in the target location and updates GPU page tables.
5. SM replays the faulting memory instruction and continues.

### Kernel Launch Path

When `cudaLaunchKernel` is called:
1. The Runtime packages grid/block dimensions, shared memory size, and kernel arguments into a launch descriptor.
2. Driver API (`cuLaunchKernel`) validates the launch, looks up the `CUfunction` → SASS binary mapping, and writes a launch command into the GPU channel's push buffer.
3. The GPU's CE (Channel Engine) reads the push buffer, programs the GPC (GPU Processing Cluster) dispatch unit, and thread blocks begin executing on available SMs.

---

## Key Findings

### Lazy Initialization (Pre-CUDA 12.0)

Prior to CUDA 12.0, the CUDA Runtime used **lazy initialization**: calling `cudaSetDevice(N)` did not immediately create a CUDA context. The primary context was only instantiated on the first actual API call that required an active context (e.g., `cudaMalloc`, first kernel launch). This saved startup overhead but caused subtle bugs when forking processes after `cudaSetDevice` — the child inherited the device selection but not the context, leading to `CUDA_ERROR_INVALID_CONTEXT` in some MPI patterns. CUDA 12.0 made `cudaSetDevice` eagerly initialize the runtime.

### Primary Context Model

The Runtime API uses a **primary context** model: one canonical `CUcontext` per (device, process) pair, managed by the Runtime and shared across all threads in the process. This contrasts with the Driver API's explicit context stack, where each host thread can push/pop distinct contexts. The `cuDevicePrimaryCtxRetain` / `cuDevicePrimaryCtxRelease` API manages reference counting for the primary context. ML frameworks (PyTorch, TensorFlow) rely on the primary context model exclusively; they use `c10::cuda::CUDAGuard` / `CUDAStreamGuard` to temporarily switch device+stream within a scope.

### CUDA Graphs: Launch Overhead Reduction

Traditional kernel launches incur ~5–10 µs of CPU-side overhead per call (driver lock acquisition, command buffer encoding, kernel argument marshalling). For ML workloads with thousands of small GEMM operations per training step, this overhead is significant. CUDA Graphs reduce total launch overhead from O(N × 10 µs) to O(1 × launch cost) by batching the entire operation graph. PyTorch's `torch.cuda.CUDAGraph` and JAX's `jit` compilation both use graph capture internally for repeated iteration loops (e.g., training steps).

### Module Loading Flexibility

The Driver API allows dynamic module loading (`cuModuleLoad` / `cuModuleLoadData`), enabling JIT compilation workflows (e.g., NVRTC → PTX → cubin → `cuModuleLoadData`). The Runtime API loads all kernels from the application's embedded fatbinary at initialization. CUDA 12.0 introduced **context-independent module loading**, allowing modules to be loaded once and shared across multiple CUDA contexts within a process — important for multi-GPU setups.

### VMM API and the Caching Allocator

PyTorch's CUDA Caching Allocator uses the VMM API (`cuMemCreate` + `cuMemMap`) to implement a pooled allocator that can grow existing allocations without moving data. The allocator reserves a large virtual address range upfront and maps physical HBM pages into it as tensors are allocated, avoiding `cudaMalloc`/`cudaFree` round-trips to the driver (which require full device synchronization and are expensive for short-lived temporaries).

### Error Handling Pattern

CUDA uses a **sticky last-error model**: asynchronous errors (e.g., illegal memory access in a kernel) are not reported until the next synchronization point. `cudaGetLastError()` returns and clears the last error on the calling thread. The recommended pattern is:
```c
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) { /* handle */ }
cudaDeviceSynchronize();
err = cudaGetLastError(); // catches async kernel errors
```

---

## Relation to Hardware

### HBM and the API Design

Modern data center GPUs (A100, H100, B100/B200) use High Bandwidth Memory (HBM2e/HBM3) stacked directly on the GPU package via silicon interposers. H100 SXM5 provides ~3.35 TB/s HBM3 bandwidth with 80 GB capacity. This physical reality shaped the CUDA memory API in several ways:

- **cudaMalloc vs cudaMallocHost**: The performance gap between HBM (~3.35 TB/s) and host DRAM via PCIe 5.0 (~128 GB/s bidirectional) is ~26×. This makes explicit data placement critical, which is why the default allocation model (cudaMalloc → device memory) is separate from host-side allocations.
- **Pinned memory**: Without pinning, every PCIe DMA requires a CPU-side staging copy from pageable memory, effectively halving PCIe transfer bandwidth. `cudaMallocHost` eliminates this overhead by locking pages.
- **Managed memory trade-offs**: While unified memory simplifies programming, demand paging through the UVM fault handler adds latency. `cudaMemPrefetchAsync` and `cudaMemAdvise(cudaMemAdviseSetPreferredLocation)` are direct API consequences of the need to pre-position data in HBM before kernel execution to avoid fault-driven migration overhead.
- **L2 cache residency**: H100's 50 MB L2 cache (shared across all SMs) is large enough to cache significant working sets. CUDA 11.x introduced `cudaStreamAttrValue::accessPolicyWindow` to pin specific memory regions in L2 across kernel launches — an API addition driven by the economics of HBM vs L2 bandwidth.
- **NVLink / NVSwitch**: Multi-GPU systems connected via NVLink (H100: 900 GB/s bidirectional) expose peer memory access via `cudaMemcpyPeerAsync` and `cuMemcpyPeer`. The CUDA P2P API (`cudaDeviceEnablePeerAccess`) programs the GPU MMU to accept direct loads/stores to another GPU's HBM, enabling zero-copy tensor passing between GPUs in NVLink-connected systems.

### MPS and MIG: Multi-Tenant GPU Access

**Multi-Process Service (MPS)**: MPS runs a server daemon that multiplexes multiple CUDA client processes onto a single GPU simultaneously. Clients connect via the CUDA Driver library (transparently — no code changes needed). The MPS server holds a single CUDA context on the GPU and submits work from all clients into it, enabling true simultaneous kernel execution across processes via Hyper-Q (multiple hardware work queues). MPS works on any GPU with compute capability ≥ 3.5 but provides no hardware-level memory isolation between clients.

**Multi-Instance GPU (MIG)**: MIG (A100, H100, A30) partitions the GPU at the hardware level into up to 7 isolated instances, each with dedicated SMs, L2 cache slices, and HBM memory partitions. From the OS perspective, each MIG instance appears as a separate GPU. MIG provides hard performance and fault isolation — a kernel crash in one instance does not affect others. The CUDA Runtime interacts with MIG instances identically to physical GPUs; the driver handles the MIG-to-physical-resource mapping transparently.

---

## Sources

- [CUDA Driver API: Runtime vs Driver API — NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html)
- [CUDA Runtime API — NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [CUDA Programming Guide: Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Programming Guide: Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [CUDA Programming Guide: Driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)
- [CUDA Driver API: Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
- [CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [MPS Documentation — NVIDIA Docs](https://docs.nvidia.com/deploy/mps/index.html)
- [CUPTI — NVIDIA Developer](https://developer.nvidia.com/cupti)
- [Unified Memory for CUDA Beginners — NVIDIA Technical Blog](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
- [Low-Level GPU Virtual Memory Management — NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
- [NVIDIA Hopper Architecture In-Depth — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA Context-Independent Module Loading — NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-context-independent-module-loading/)
- [What is the CUDA Runtime API? — Modal GPU Glossary](https://modal.com/gpu-glossary/host-software/cuda-runtime-api)
- [What is the CUDA Driver API? — Modal GPU Glossary](https://modal.com/gpu-glossary/host-software/cuda-driver-api)
- [PyTorch c10::cuda CUDAStream](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAStream.cpp)
- [PyTorch c10::cuda CUDAGuard](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAGuard.h)
- [NVIDIA Open GPU Kernel Modules Analysis — eunomia.dev](https://eunomia.dev/zh/blog/posts/nvidia-open-driver-analysis/)
- [CUDA Driver VS CUDA Runtime — Lei Mao's Log Book](https://leimao.github.io/blog/CUDA-Driver-VS-CUDA-Runtime/)
- [Boost GPU Memory Performance with MPS — NVIDIA Technical Blog](https://developer.nvidia.com/blog/boost-gpu-memory-performance-with-no-code-changes-using-nvidia-cuda-mps)
