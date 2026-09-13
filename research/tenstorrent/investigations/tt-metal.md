# TT-Metal Runtime + Kernels Investigation

*as_of: 2026-04-05*
*source: https://github.com/tenstorrent/tt-metal (shallow clone at /tmp/tt-metal-investigate)*
*device_class: Tensix RISC + SFPU*

---

## Overview

TT-Metal is the monorepo hosting both **TT-Metalium** (the low-level kernel programming SDK) and **TT-NN** (the PyTorch-like operator library built on top of TT-Metalium). It is fully open-source (Apache 2.0) and is the foundation of the entire Tenstorrent software stack.

Repository structure:
- `tt_metal/` — TT-Metalium runtime, APIs, kernel compilation infrastructure
- `ttnn/` — TT-NN operator library
- `METALIUM_GUIDE.md` — authoritative guide to the programming model
- `models/` — reference model implementations (Llama, GPT, Stable Diffusion, etc.)
- `tech_reports/` — detailed implementation guides (FlashAttention, mesh devices, etc.)

---

## Device and Program Abstractions

### Device

A `tt::tt_metal::Device` object represents a single Tenstorrent accelerator card. It is obtained via `CreateDevice(device_id)` and provides:
- A 2D coordinate system for Tensix cores (Tensix grid)
- Buffer allocation APIs (L1 SRAM buffers, DRAM-backed interleaved/sharded buffers)
- Program enqueue/execute interfaces

The device exposes the underlying NoC grid. On Wormhole, the grid is physically 10×12 tiles (120 total), of which 80 are active Tensix compute cores; the remainder are DRAM, Ethernet, PCIe, and management (ARC) tiles.

### Program

A `tt::tt_metal::Program` is a collection of kernels and circular buffer definitions targeting a specific set of Tensix cores. It does not execute until explicitly enqueued to a device via `EnqueueProgram`. Programs are compiled ahead of time (kernel C++ is JIT-compiled to RISC-V binaries at program creation time).

Key Program API:
```cpp
Program program = CreateProgram();
// Add kernels to program targeting a CoreRange
KernelHandle reader = CreateKernel(program, "reader.cpp", core_range,
    DataMovementConfig{.processor=DataMovementProcessor::RISCV_0, .noc=NOC::NOC_0});
KernelHandle writer = CreateKernel(program, "writer.cpp", core_range,
    DataMovementConfig{.processor=DataMovementProcessor::RISCV_1, .noc=NOC::NOC_1});
KernelHandle compute = CreateKernel(program, "compute.cpp", core_range,
    ComputeConfig{.math_fidelity=MathFidelity::HiFi4});
// Attach circular buffers
CBHandle cb_in0 = CreateCircularBuffer(program, core_range, CircularBufferConfig(num_pages*tile_sz, ...));
// Set runtime args
SetRuntimeArgs(program, reader, core, {src_addr, n_tiles});
// Execute
EnqueueProgram(cq, program, false);
```

### Fast Dispatch

By default, programs use the **Fast Dispatch** path: a lightweight command queue in DRAM (managed by the ARC management core) that dispatches kernel binaries and runtime arguments to Tensix cores without driver round-trips. This enables low-latency dispatch (~microseconds) for inference workloads.

---

## Kernel Types

TT-Metalium defines three kernel types that run concurrently on a single Tensix core. Each maps to specific RISC-V baby cores:

### 1. Data Movement Kernel (Reader) — BRISC / RISC0
- **Core**: `DataMovementProcessor::RISCV_0` using `NOC::NOC_0`
- **Role**: Reads input tensors from DRAM or remote Tensix L1 into local circular buffers
- **API**: `noc_async_read_tile()`, `noc_async_read_barrier()`, `cb_reserve_back()`, `cb_push_back()`
- **Header**: `dataflow_api.h`

### 2. Data Movement Kernel (Writer) — NCRISC / RISC1
- **Core**: `DataMovementProcessor::RISCV_1` using `NOC::NOC_1`
- **Role**: Writes output tiles from circular buffers back to DRAM or remote L1
- **API**: `noc_async_write_tile()`, `noc_async_write_barrier()`, `cb_wait_front()`, `cb_pop_front()`
- **Header**: `dataflow_api.h`

### 3. Compute Kernel — TRISC0 (Unpack) + TRISC1 (Math) + TRISC2 (Pack)
- **Cores**: Three RISC-V cores within the compute subsection, operating as a pipeline
- **Role**: Implements the Unpack → Math → Pack data flow through the FPU/SFPU
- **Compilation**: The same compute kernel C++ source is compiled **three times** — once per TRISC core. Conditional sections are enabled per-core via preprocessor guards.
- **Header**: `compute_kernel_api.h`, `compute_kernel_api/eltwise_binary.h`, `compute_kernel_api/matmul.h`, etc.

The three TRISC cores run in a pipeline:
1. **TRISC0 (Unpack)**: Reads tiles from L1 circular buffer into FPU source registers (SrcA, SrcB)
2. **TRISC1 (Math)**: Executes FPU/SFPU instructions (matrix multiply, eltwise ops, reductions)
3. **TRISC2 (Pack)**: Reads FPU destination registers and writes packed tiles back to L1 circular buffer

Synchronization between TRISC0/1/2 uses `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` semaphore primitives operating on the FPU destination register file.

---

## CircularBuffer Abstraction

`CircularBuffer` (CB) is the primary inter-kernel synchronization and data exchange mechanism. CBs live in each Tensix's L1 SRAM.

```cpp
// Reader: write side
cb_reserve_back(cb_in0, 1);       // reserve 1 tile slot
uint32_t addr = get_write_ptr(cb_in0);
noc_async_read_tile(i, a, addr);
noc_async_read_barrier();
cb_push_back(cb_in0, 1);           // signal tile is ready

// Compute (Unpack side): read
cb_wait_front(cb_in0, 1);          // block until tile available
// ... process tile ...
cb_pop_front(cb_in0, 1);           // release tile slot

// Compute (Pack side): write
cb_reserve_back(cb_out, 1);
pack_tile(dst_reg, cb_out, 0);
cb_push_back(cb_out, 1);
```

Each CB is identified by a `tt::CBIndex` enum (e.g., `c_0` through `c_31`). CBs are ring buffers — `reserve_back`/`push_back` are the producer interface; `wait_front`/`pop_front` are the consumer interface. Hardware semaphores enforce ordering without software spin-locks.

CB configuration:
```cpp
CircularBufferConfig cfg(total_size_bytes, {{CBIndex::c_0, DataFormat::Float16_b}});
cfg.set_page_size(CBIndex::c_0, tile_size_bytes);
```

---

## NoC-Based Data Movement

Tenstorrent's architecture has **no hardware cache hierarchy** above each Tensix's 1.5 MB L1 SRAM. All data movement between Tensix cores, between Tensix and DRAM, and between chips is **explicit** — managed entirely by the programmer (or compiler) through NoC operations.

### Dual NoC Design

Two independent NoCs traverse the chip in opposite directions on a 2D torus:
- **NoC 0**: Travels east and south; RISCV_0 (BRISC) issues reads (DRAM → L1)
- **NoC 1**: Travels west and north; RISCV_1 (NCRISC) issues writes (L1 → DRAM)

Each tile has 4 outbound + 4 inbound connections (N/E/S/W), each **32 bytes wide**.

### NoC API

```cpp
// Async DRAM → L1 read
noc_async_read(src_noc_addr, dst_local_l1_addr, size_bytes);
noc_async_read_barrier();  // wait for all in-flight reads to complete

// Async L1 → DRAM write
noc_async_write(src_local_l1_addr, dst_noc_addr, size_bytes);
noc_async_write_barrier();

// NoC address encoding
uint64_t noc_addr = get_noc_addr(noc_x, noc_y, local_l1_offset);
uint64_t dram_addr = get_noc_addr_from_bank_id(bank_id, offset);
```

The `TensorAccessor` abstraction (new in recent TT-Metal releases) simplifies tile addressing across interleaved/sharded DRAM banks:
```cpp
constexpr auto args = TensorAccessorArgs<0>();
const auto tensor = TensorAccessor(args, base_addr, tile_size_bytes);
noc_async_read_tile(tile_idx, tensor, cb_write_ptr);
```

### L1 ↔ L1 (Tensix-to-Tensix)

Tensix cores can DMA directly to/from another core's L1 using the same NoC API with the target core's (noc_x, noc_y) coordinate. This enables producer-consumer pipelines across multiple Tensix cores without DRAM round-trips. The `tt-npe` (NoC Performance Estimator) tool helps optimize such patterns.

---

## Tensix Core Model

### Hardware Components per Tensix
| Component | Detail |
|-----------|--------|
| Baby RISC-V cores | 5 total: BRISC, NCRISC, TRISC0, TRISC1, TRISC2 |
| L1 SRAM | 1.5 MB per core (1.3 MB on Grayskull) |
| NoC routers | 2 (NoC 0 + NoC 1), each 32 bytes wide |
| Matrix unit (FPU) | Weight-stationary matmul; 32×32 tile ops; BF16/FP16/INT8/FP8 |
| Vector unit (SFPU) | 32-lane SIMD; elementwise: ReLU, GeLU, Exp, Sqrt, Recip, etc. |
| Unpack unit | Reformats L1 tile data → FPU input registers (SrcA, SrcB) |
| Pack unit | Reformats FPU destination registers → L1 circular buffer tiles |

### Native Tile Format

All compute operates on **32×32 tiles**. The unpack unit converts from arbitrary tensor memory layout into the internal FPU tile format (row-major 32×32). The pack unit converts back. Supported data formats: BF16, FP16, FP32, INT8, INT32, FP8 (E4M3, E5M2).

### SPMD Programming

Metalium supports SPMD (Single Program, Multiple Data): the same kernel program runs on all targeted Tensix cores, but each core has unique runtime arguments (e.g., its slice of the input tensor). This is the primary parallelism model for most operations.

---

## TT-NN Layer (on top of TT-Metalium)

TT-NN provides PyTorch-like ops (`ttnn.matmul`, `ttnn.add`, `ttnn.softmax`, etc.) that internally dispatch to optimized TT-Metalium kernels. TT-NN handles:
- Tensor layout management (row-major vs. tile layout, sharding strategies)
- Automatic kernel selection and CircularBuffer sizing
- Multi-device tensor operations via `MeshDevice`
- Operations match PyTorch semantics for framework integration

```python
import ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(torch_tensor2, device=device, layout=ttnn.TILE_LAYOUT)
c = ttnn.matmul(a, b)
result = ttnn.to_torch(c)
```

---

## Key Insight: Explicit Data Movement Model

The defining characteristic of TT-Metal vs. GPU programming is that **all data movement is explicit and programmer-controlled**. There are no hardware caches between Tensix cores — every byte that moves must be explicitly requested via NoC operations. This shifts data movement complexity from hardware to software, but enables:
1. Predictable, bandwidth-optimal data movement patterns
2. Compiler-optimized overlap of compute and data movement
3. Zero cache coherence overhead across a mesh of 80–140 cores
4. Full programmer visibility into the memory access pattern

This is the fundamental design principle that distinguishes Tenstorrent from GPU SIMT architectures.

---

## Sources
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [Memory for Kernel Developers](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html)
- [Getting Started — TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
