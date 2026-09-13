# PTX ISA Investigation
**Target:** NVIDIA Parallel Thread Execution (PTX) Virtual ISA
**Source:** https://docs.nvidia.com/cuda/parallel-thread-execution/
**PTX Version Covered:** 8.7–9.2 (CUDA 12.8–12.9+, through Blackwell SM100)
**Date:** 2026-04-05

---

## Overview

PTX (Parallel Thread Execution) is NVIDIA's virtual machine instruction set architecture (ISA). It serves as the stable, architecture-independent intermediate representation sitting between high-level CUDA C++ and the native hardware ISA (SASS). PTX has been part of the CUDA platform since its inception and is the primary compilation target for `nvcc`.

**Role as a compiler target:** High-level CUDA C++ is compiled by `nvcc` into PTX, which represents a "virtual GPU" abstracting the common features of all NVIDIA hardware. PTX is not executed directly by the hardware; it must be further assembled into SASS by the `ptxas` tool.

**Role as a forward-compatibility layer:** PTX is the mechanism through which compiled CUDA applications run on future, yet-unknown GPU architectures. When an application ships with embedded PTX (targeting, e.g., `compute_90`), the NVIDIA driver JIT-compiles that PTX to native SASS at load time on any future GPU with compute capability >= 9.0. This allows a binary compiled today to run—without recompilation—on GPUs that did not exist at compile time.

**SASS as the real ISA:** SASS (Streaming ASSembly) is the actual native, hardware-specific ISA. It is not publicly documented in full by NVIDIA. There is not a one-to-one mapping between PTX and SASS: a single PTX instruction may be expanded to multiple SASS instructions, fused with adjacent instructions, or eliminated entirely through `ptxas` optimization. The control-flow instruction count illustrates the gap (PTX has ~5 control-flow instructions; SASS Turing has 20).

---

## Architecture

### Instruction Categories

| Category | Representative Instructions | Notes |
|---|---|---|
| Integer Arithmetic | `add`, `sub`, `mul`, `mad`, `div`, `rem`, `min`, `max`, `abs`, `neg` | Full 32/64-bit, extended-precision (`add.cc`, `addc`) |
| Floating-Point | `fma`, `mad`, `mul`, `add`, `div`, `sqrt`, `rcp`, `rsqrt`, `sin`, `cos`, `lg2`, `ex2` | FP16, BF16, FP32, FP64; rounding mode qualifiers |
| Comparison & Predication | `setp`, `selp`, `slct` | Result written to predicate register `%p` |
| Logic & Bit Manipulation | `and`, `or`, `xor`, `not`, `shl`, `shr`, `bfe`, `bfi`, `popc`, `clz` | |
| Memory Load/Store | `ld`, `st`, `prefetch`, `isspacep`, `cvta` | State-space qualified (`.global`, `.shared`, `.local`, `.const`) |
| Async Memory Copy | `cp.async`, `cp.async.bulk`, `cp.async.bulk.tensor` | GMEM→SMEM bypass of registers; TMA descriptor-based |
| Synchronization | `bar.sync`, `bar.arrive`, `membar`, `fence`, `atom`, `red` | Warp, CTA, cluster, system scopes |
| Control Flow | `bra`, `call`, `ret`, `exit`, `@p` predicated execution | |
| Warp-Level Matrix (Volta–Ampere) | `wmma.load`, `wmma.store`, `wmma.mma`, `mma.sync.aligned` | Warp-scoped matrix multiply-accumulate (32-thread cooperation) |
| Warp-Group MMA (Hopper) | `wgmma.mma_async`, `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group` | 4-warp group (128-thread) asynchronous MMA; result in registers |
| Tensor Core Gen5 (Blackwell) | `tcgen05.mma`, `tcgen05.mma.sp`, `tcgen05.mma.ws`, `tcgen05.ld`, `tcgen05.st`, `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.cp`, `tcgen05.wait`, `tcgen05.fence`, `tcgen05.shift` | Single-thread semantics; result in Tensor Memory (TMEM) |
| Special Functions | `tex`, `tld4`, `suld`, `sust`, `suq` | Texture/surface memory access |
| Conversion | `cvt`, `cvtpack` | Type conversion with rounding/saturation |
| Video (SIMD4) | `vadd`, `vsub`, `vabsdiff`, `vmad` | Byte/short SIMD for video processing |

### State Spaces (Memory Hierarchy)

| State Space | Directive | Scope | Description |
|---|---|---|---|
| Register | `.reg` | Per-thread | Fastest; not addressable; ~255 per thread typical limit |
| Local | `.local` | Per-thread | Thread-private, typically spilled to DRAM |
| Shared | `.shared` | Per-CTA | On-chip; accessible by all threads in a thread block |
| Global | `.global` | All threads | Device DRAM; largest; coherent across grid |
| Constant | `.const` | All threads (read-only) | Host-initialized; cached; 64 KB |
| Texture/Surface | `.tex`/`.surf` | All threads | Spatially-cached; hardware interpolation |
| Parameter | `.param` | Kernel entry or device func | Kernel arguments passed from host |
| Special Register | `.sreg` | Platform-defined | `%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%clock`, `%smid`, etc. |
| Tensor Memory | `.tmem` | Per-SM (Blackwell+) | New in SM100; dedicated hardware store for MMA accumulators |

### Addressing Modes

PTX uses byte-based address arithmetic. Addresses are computed using integer arithmetic instructions. State-space qualifiers disambiguate which memory space an address belongs to. The `cvta` instruction converts between generic and state-space-specific addresses. Generic addressing (no explicit state space) enables uniform load/store instructions where the hardware determines the target memory space at runtime from address range.

### Special Registers

Key predefined registers (`%sreg`):
- Thread identity: `%tid.{x,y,z}`, `%ntid.{x,y,z}`, `%laneid`, `%warpid`
- Block/Grid identity: `%ctaid.{x,y,z}`, `%nctaid.{x,y,z}`, `%gridid`
- Cluster (Hopper+): `%cluster_ctaid`, `%cluster_nctaid`, `%clusterid`
- Hardware: `%smid`, `%nsmid`, `%clock`, `%clock64`, `%pm0`–`%pm7`

### Compute Capability Mapping

| Architecture | Compute Capability | PTX ISA Version | Key PTX Additions |
|---|---|---|---|
| Volta | sm_70 | PTX 6.0 | `mma.sync`, independent thread scheduling |
| Turing | sm_75 | PTX 6.4 | Sparse MMA, INT4/INT1 tensor ops |
| Ampere | sm_80 | PTX 7.0 | `cp.async`, `wgmma` precursor, BF16/TF32, sparsity |
| Ada Lovelace | sm_89 | PTX 7.8 | FP8 support, enhanced sparse |
| Hopper | sm_90 | PTX 8.0 | `wgmma`, TMA (`cp.async.bulk.tensor`), cluster CTA hierarchy, `griddepcontrol` |
| Blackwell | sm_100 | PTX 8.7 | `tcgen05` family, Tensor Memory (TMEM), 2-SM MMA (`.cta_group::2`), FP4/FP6 narrow precision |

---

## Data Flow

### Compilation Pipeline

```
CUDA C++ (.cu)
    │
    ▼
nvcc (front-end: clang/nvopencc)
    │  Separates GPU and CPU code paths
    │  Performs language-level transformations
    ▼
PTX Assembly (.ptx)
    │  Architecture-independent virtual ISA
    │  Human-readable text format
    ▼
ptxas (PTX Assembler)
    │  Applies register allocation, scheduling
    │  Inlines constants, optimizes instruction selection
    │  Emits cubin for specific sm_XX target
    ▼
SASS (cubin .cubin / fatbin)
    │  Native binary machine code for specific GPU
    │  Embedded in executable via fatbinary
    ▼
GPU Hardware (SM Execution)
```

**Fatbinary format:** `nvcc` typically produces a fatbinary containing both pre-compiled SASS (for immediate performance on a known target) and embedded PTX (for forward compatibility). The CUDA runtime selects the best available cubin, or JIT-compiles the PTX if no matching cubin is present.

### JIT Compilation by Driver

When an application runs on a GPU for which no matching pre-compiled SASS exists in the fatbinary:
1. The CUDA driver extracts the embedded PTX.
2. The driver's JIT compiler (essentially `ptxas` at runtime) compiles PTX → SASS for the current GPU.
3. The resulting cubin is cached on disk to avoid repeated JIT cost on subsequent runs.
4. The JIT-compiled SASS may be suboptimal compared to an offline-compiled version targeting that specific architecture but is functionally correct.

**Forward compatibility guarantee:** PTX code targeting `compute_X` will run on any GPU with compute capability >= X. A binary embedding `compute_70` PTX can be JIT-compiled for SM90 (H100) or SM100 (B200). The reverse is not true—PTX targeting `compute_90` cannot run on SM80.

### PTX Compiler API

NVIDIA also exposes the PTX Compiler API (`libNVPTXCompiler`), allowing applications to embed PTX and invoke the PTX→SASS compilation pipeline programmatically at runtime without calling the full CUDA driver JIT path.

---

## Hardware Interface

### PTX → SM Execution Unit Mapping

| PTX Instruction Class | Target SM Execution Unit | Notes |
|---|---|---|
| Integer arithmetic (`add`, `mul`, `mad`, etc.) | INT32 cores (64/SM in Ampere) | Dedicated integer datapath separate from FP32 |
| FP32 arithmetic (`fma.f32`, `mul.f32`, etc.) | FP32 CUDA cores (128/SM in Ampere) | 4 partitions × 32 cores |
| FP64 arithmetic (`fma.f64`, etc.) | FP64 units (varies by SKU) | Full-rate on HPC SKUs (A100, H100); reduced on consumer |
| Transcendentals (`sin`, `cos`, `sqrt`, `rcp`) | SFU (Special Function Units) | 4 SFUs per SM partition; ~4 cycles throughput |
| Load/store (`ld.global`, `st.shared`) | LD/ST units + memory pipeline | Global through L1/L2; shared via crossbar |
| `cp.async` | Load/Store units + async pipeline | Bypass registers; write directly to shared memory |
| `cp.async.bulk.tensor` (TMA) | Tensor Memory Accelerator hardware unit | H100+; single-thread dispatch; 1D–5D tensor transfers |
| `mma.sync` (Volta–Ampere) | Tensor Core (1st–3rd gen) | 32-thread warp; result in registers |
| `wgmma.mma_async` (Hopper) | Tensor Core (4th gen) | 128-thread warp group; result in registers; async dispatch |
| `tcgen05.mma` (Blackwell) | Tensor Core (5th gen) | Single-thread semantics; result in TMEM; 2-SM collaborative |

### Tensor Core Generation and PTX Instruction Binding

- **Gen1 (Volta, sm_70):** `wmma.*` PTX; HMMA SASS; 16×16×16 FP16 tiles; 8-thread quadpair sync.
- **Gen2 (Turing, sm_75):** `mma.sync` PTX; adds INT8/INT4/INT1; 32-thread warp sync.
- **Gen3 (Ampere, sm_80):** `mma.sync` with TF32, BF16, sparse; FP64 matrix ops on A100; 32-thread warp sync.
- **Gen4 (Hopper, sm_90):** `wgmma.mma_async`; 128-thread warp group; up to 64×256×16 FP16 tiles; async pipelining; result stays in registers.
- **Gen5 (Blackwell, sm_100):** `tcgen05.mma`; single-thread semantics; up to 256×256×16 (2-SM); result in dedicated TMEM; FP4/FP6/FP8 narrow precision; 2×–4× throughput vs. WGMMA.

### SM Pipeline and TMEM (Blackwell)

Blackwell (SM100) introduces Tensor Memory (TMEM), a dedicated on-SM hardware buffer for MMA accumulator results. Previously, MMA outputs were held in thread registers (fragmenting the programmer's register budget). TMEM decouples accumulator storage from the general register file, allowing larger tiles and reducing register pressure. `tcgen05.ld`/`tcgen05.st` transfer data between TMEM and shared/global memory; the explicit `tcgen05.alloc`/`tcgen05.dealloc` lifecycle matches TMEM ownership to kernel lifetime.

---

## Key Findings

1. **PTX as the abstraction layer:** PTX provides a stable, version-controlled ISA that decouples CUDA compilation from GPU hardware generations. NVIDIA can add new execution units (Tensor Cores, TMA, TMEM) without breaking existing binaries: old PTX still compiles to valid SASS; new PTX instructions are guarded by minimum `sm_XX` requirements.

2. **SASS is the real ISA:** PTX is a compiler IR and portability vehicle, not what the hardware executes. SASS instructions (HMMA, LDGSTS, WGMMA, etc.) are the actual microcode. The `ptxas` optimizer performs substantial transformation: register allocation, instruction scheduling, constant folding, and instruction selection.

3. **Tensor Core access requires generation-specific PTX:** Each Tensor Core generation introduced new PTX instruction families. Direct performance requires using the latest-generation instructions (`wgmma` for H100, `tcgen05` for B200). Older `mma.sync` instructions work on newer GPUs but don't unlock the new capabilities.

4. **`cp.async` / TMA as memory-side companions to tensor ops:** High-performance matrix kernels pair async memory copy with MMA. `cp.async` (Ampere, `LDGSTS` in SASS) hides global memory latency by copying directly to shared memory without occupying registers. TMA (`cp.async.bulk.tensor`, H100+) further reduces overhead to a single-thread descriptor-based issue, enabling warp-specialized "producer/consumer" kernel patterns.

5. **`wgmma` → `tcgen05` paradigm shift:** Hopper's `wgmma` kept results in 128-thread warp-group registers, tying accumulation to warp scheduling. Blackwell's `tcgen05` moves results to TMEM with single-thread issue semantics, removing the warp-group coordination requirement and enabling cooperative 2-SM GEMM across CTA pairs.

6. **Forward compatibility is JIT, not interpretation:** PTX forward compatibility is implemented by the driver JIT-compiling PTX to the target GPU's SASS at load time—not by interpreting PTX. The compiled result is cached on disk. This makes the first-run startup time longer but subsequent runs incur no JIT cost.

7. **Inline PTX as escape hatch:** The `asm volatile("...")` inline PTX mechanism allows performance engineers to write PTX directly within CUDA device functions, bypassing `nvcc`'s compiler heuristics. This is used extensively by libraries like CUTLASS and by projects like DeepSeek (DeepEP) to access instructions or optimization patterns that the C++ compiler does not generate automatically.

---

## Relation to Hardware

### How SM Pipeline Stages Shaped PTX Instruction Additions

PTX instruction additions have closely tracked SM microarchitecture changes:

- **Independent Thread Scheduling (Volta, sm_70):** Each thread gets its own program counter. This enabled `mma.sync` to require explicit warp-level synchronization (`sync` qualifier) rather than implicit convergence, and made the `bar.sync`/`bar.warp` distinction important.

- **Asynchronous execution pipeline (Ampere, sm_80):** The addition of a dedicated async-copy pipeline running independently of compute dispatched the `cp.async` instruction. Software uses `cp.async.commit_group`/`cp.async.wait_group` barriers to coordinate data availability with computation.

- **Distributed Shared Memory and Cluster hierarchy (Hopper, sm_90):** The SM cluster (up to 8 CTAs sharing L2-level bandwidth) spawned new PTX: cluster-scoped atomics, `mbarrier` with cluster-scope, and TMA's multi-CTA tile transfer. `wgmma` maps to H100's 4th-gen Tensor Cores which require 4-warp cooperation as a hardware constraint, reflected directly in the instruction's warp-group operand granularity.

- **Tensor Memory and 2-SM Tensor Cores (Blackwell, sm_100):** The SM100 physical Tensor Core spans 2 SMs, requiring a coordinated CTA pair. PTX reflects this via `tcgen05.mma.cta_group::2` and TMEM allocation per-CTA-pair. The shift to single-thread issue semantics maps to Blackwell's hardware unit requiring only one thread to trigger the multi-SM operation.

- **Narrow precision (Blackwell, sm_100):** FP4 (E2M1), FP6 (E2M3, E3M2), and FP8 types were added to TMEM-resident `tcgen05.mma` instructions as dedicated data-type qualifiers, tracking the physical Tensor Core datapaths capable of consuming these compressed formats at full throughput.

---

## Sources

- [PTX ISA 9.2 Documentation — NVIDIA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Understanding PTX, the Assembly Language of CUDA GPU Computing — NVIDIA Technical Blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [Using Inline PTX Assembly in CUDA — NVIDIA](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
- [PTX Compiler API — NVIDIA](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html)
- [CUDA Compiler Driver nvcc — NVIDIA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [NVIDIA Tensor Core Evolution: From Volta To Blackwell — SemiAnalysis](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [Dissecting Nvidia Blackwell — Tensor Cores, PTX Instructions, SASS — SemiAnalysis](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)
- [tcgen05 for dummies — gau-nernst's blog](https://gau-nernst.github.io/tcgen05/)
- [Deep Dive on the Hopper TMA Unit for FP8 GEMMs — PyTorch Blog](https://pytorch.org/blog/hopper-tma-unit/)
- [Asynchronous Data Copies — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [Blackwell Architecture Compatibility Guide — NVIDIA](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [Hopper Architecture Compatibility Guide — NVIDIA](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)
- [Matrix Multiplication on Blackwell: Part 1 — Modular](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- [A Gentle Introduction to CUDA PTX — Philip Fabianek](https://philipfabianek.com/posts/cuda-ptx-introduction)
