# XLA Compiler & Pallas Kernel Library Investigation

**Layer**: Compiler / IR + Kernel Library
**Chip**: Google TPU (v4 / v5e / v5p / v6e / v7 Ironwood)
**as_of**: 2026-04-05

---

## Overview

The Google TPU software stack rests on two tightly coupled compilation layers: **XLA** (Accelerated Linear Algebra), the primary compiler backend, and **Pallas**, the JAX-native kernel language for custom operations. Together they form a complete path from high-level Python numerics to optimized VLIW instruction bundles executed on the TPU's systolic arrays.

**XLA** is an open-source, domain-specific compiler for machine learning computation graphs. It consumes **HLO (High Level Operations)** — an SSA graph IR representing linear algebra operations — and emits hardware-specific executables. For TPUs, XLA is packaged inside `libtpu` and performs hardware-independent graph rewrites, operator fusion, layout assignment, and memory scheduling before targeting the TPU backend.

**Pallas** is an extension to JAX that enables writing custom TPU kernels using familiar JAX/NumPy semantics. Pallas kernels compile via **Mosaic**, a dedicated TPU-targeting compiler, which bridges Pallas IR to VLIW instruction packets. This path is analogous to how NVIDIA's Triton lowers to PTX, but Pallas/Mosaic is specifically co-designed with TPU's memory hierarchy and execution model.

The two layers interact at the XLA boundary: Pallas kernels are embedded as custom calls in HLO programs, then lowered by Mosaic in parallel with the rest of the XLA compilation.

---

## Architecture

### XLA / HLO

#### HLO Intermediate Representation

HLO is XLA's core IR: a static-single-assignment computation graph where each node is a high-level linear algebra operation (dot product, convolution, reduce, scatter, etc.). JAX traces Python functions to **jaxprs** (JAX's functional intermediate representation), then lowers jaxprs to **StableHLO** — a versioned, backward-compatible MLIR opset — and finally hands it to XLA.

The full JAX-to-binary compilation chain:

```
Python function
    |
    v  [jax.jit tracing]
jaxpr (functional IR)
    |
    v  [JAX lowering]
StableHLO (MLIR opset)
    |
    v  [XLA frontend]
HLO (XLA-internal SSA graph)
    |
    v  [XLA optimization passes]
Optimized HLO
    |
    v  [XLA TPU backend]
LLO (Low Level Operations / Mosaic IR bridge)
    |
    v  [Mosaic / TPU codegen]
VLIW instruction bundles → TPU binary
```

#### StableHLO: Portable IR Layer

StableHLO is the portability layer between ML frameworks (JAX, PyTorch/XLA, TensorFlow) and XLA-based compilers. It is defined as an MLIR dialect with a formal specification covering ~100 operations including:

- Tensor operations: `dot_general`, `convolution`, `reduce`, `scatter`
- Element-wise: `add`, `multiply`, `exp`, `log`, custom element-wise via `custom_call`
- Shape manipulation: `reshape`, `transpose`, `slice`, `dynamic_slice`
- Control flow: `while`, `conditional`, `reduce_window`

StableHLO provides backward compatibility guarantees (older programs remain valid) and forward compatibility guarantees (programs serialized with newer ops can be parsed by older runtimes within a compatibility window). This stability decouples framework release cycles from compiler release cycles.

#### XLA Optimization Passes (HLO → Optimized HLO)

XLA applies a sequence of hardware-independent optimization passes:

| Pass | Function |
|------|----------|
| **Operator Fusion** | Merges adjacent element-wise ops, reductions, and broadcasts into fused kernels — the highest-impact single optimization |
| **Layout Assignment** | Assigns physical memory layouts (row/column major, tile order) to tensors for hardware efficiency |
| **Memory Space Assignment** | Schedules HBM ↔ VMEM data movement to minimize on-chip memory pressure |
| **Buffer Assignment** | Lifetimes analysis; assigns buffers to minimize peak memory allocation |
| **CSE / DCE** | Common subexpression elimination and dead code elimination |
| **Rematerialization** | Re-computes values instead of storing them when memory is scarce |
| **Dot/Convolution Decomposition** | Tiles large matrix multiplies into shapes optimal for the MXU |
| **SPMD Partitioning** (GSPMD / Shardy) | Inserts collective communication ops based on sharding annotations |

#### GSPMD and Shardy: Tensor Partitioning

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is XLA's SPMD partitioner: given user-provided sharding annotations on a subset of tensors, it propagates those annotations through the entire computation graph and inserts the required collective communication operations (AllGather, AllReduce, AllToAll, CollectivePermute, ReduceScatter).

**Shardy** is the successor to GSPMD, developed jointly by the GSPMD team and the PartIR team (DeepMind). It combines GSPMD's rewriting-free shard propagation with PartIR's mesh-axis-based representation and incremental partitioning for improved predictability and debuggability. As of March 2026, Shardy has completed its migration and is the only partitioner in JAX, replacing GSPMD entirely.

Shardy operates on MLIR and is dialect-agnostic; it works at the StableHLO level before XLA finalizes the HLO graph. Key capabilities:
- Axis-based sharding annotations (`@sdy.sharding`)
- Automatic shard propagation with user-override support
- Collective insertion for cross-device data movements
- Planned SPMD partitioner replacing `xla_sharding.py` API

---

### Pallas / Mosaic

#### Pallas: JAX-Native Kernel DSL

Pallas is to JAX/TPU what Triton is to PyTorch/GPU — a kernel language that lets users write custom high-performance operations using familiar NumPy-like semantics, while giving precise control over the TPU's memory hierarchy and execution pipeline.

Pallas kernels are written as Python functions decorated with `pallas_call`. The programming model:

- **Grid-based execution**: Each kernel invocation specifies a `grid` of logical blocks. Each block processes one tile of the output.
- **Block specs**: `BlockSpec` objects define how input/output tensors are tiled across the grid. Each grid point receives slices of the inputs as VMEM references.
- **Explicit memory placement**: Kernel body references point to on-chip VMEM (or SMEM/CMEM via `MemorySpace` enum), not HBM. HBM ↔ VMEM data movement is scheduled by Mosaic and overlapped with compute.
- **JAX semantics**: Inside the kernel body, `jax.numpy` operations, control flow (`lax.while_loop`), and `pl.load`/`pl.store` primitives are used.

Example Pallas kernel structure:
```python
@functools.partial(pallas_call,
    out_shape=jax.ShapeDtypeStruct(shape, dtype),
    grid=(num_blocks,),
    in_specs=[pl.BlockSpec(block_shape, lambda i: (i, 0))],
    out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0)))
def my_kernel(x_ref, o_ref):
    o_ref[...] = jax.numpy.sum(x_ref[...], axis=-1)
```

#### Mosaic: The TPU Kernel Compiler Backend

Mosaic is the compiler that translates Pallas kernels to TPU machine code. It operates between the Pallas IR (JAX IR after tracing the kernel body) and VLIW instruction emission:

```
Pallas kernel (Python + JAX)
    |
    v  [pallas_call tracing]
Pallas IR / Jaxpr
    |
    v  [Mosaic frontend]
Mosaic IR (MLIR dialect)
    |
    v  [Mosaic optimization passes]
  - Tiling and vectorization for MXU/VPU
  - VMEM double-buffering for HBM↔VMEM pipeline
  - Scalar prefetch insertion (CMEM usage)
  - Pipeline slot scheduling
    |
    v  [Mosaic TPU codegen]
LLO (Low Level Operations)
    |
    v  [TPU assembler]
VLIW instruction bundles (8 ops/cycle)
```

Mosaic exposes the following TPU architectural features to the kernel author:
- **Systolic array (MXU)**: `pl.dot` operations map to MXU matrix multiply
- **VMEM tiling**: `BlockSpec` controls tile dimensions aligned to MXU input requirements (128×128 or 256×256 elements)
- **CMEM (scalar memory)**: Separate memory space for index variables and small lookup tables, avoiding VMEM bandwidth for scalar ops
- **Pipeline stages**: Mosaic generates double- or triple-buffered DMAs from HBM to VMEM so that one tile is being loaded from HBM while the previous tile is being computed on the MXU
- **VPU (Vector Processing Unit)**: Element-wise operations (activations, norms, etc.) lower to VPU instructions, executed in parallel with MXU

#### Pallas ↔ XLA Integration Point

Pallas kernels are embedded in the main XLA program as `custom_call` HLO operations. During XLA compilation:

1. JAX traces the main program to jaxpr, encountering `pallas_call` as a primitive.
2. `pallas_call` is lowered to a StableHLO `custom_call` with the Mosaic-compiled binary attached as a side car.
3. XLA treats the custom call as an opaque op with known input/output shapes; it does not fuse across it but can schedule buffer assignments and I/O transfers around it.
4. At runtime, libtpu dispatches the embedded Mosaic binary to the TPU and manages its HBM input/output buffers.

---

## Data Flow

### JAX jit → XLA → TPU Execution

A concrete trace through `jax.jit(lambda x, W: jax.numpy.dot(x, W))(x, W)`:

**Step 1 — Python call triggers tracing**
`jax.jit` wraps the function. On first call, JAX traces it with abstract values to produce a jaxpr:
```
{ lambda ; x:f32[B,D] W:f32[D,H].
  let y = dot_general(x, W, (([1],[0]),([],[])))
  in (y,) }
```

**Step 2 — jaxpr → StableHLO**
The jaxpr `dot_general` primitive lowers to a StableHLO `stablehlo.dot_general` op with contracting/batch dimension annotations.

**Step 3 — StableHLO → HLO**
XLA's importer ingests StableHLO MLIR into its HLO module. Layout assignment determines whether the matrix multiply uses row-major or column-major layout for the MXU tile pipeline.

**Step 4 — HLO optimizations**
- **Fusion**: The dot plus any following activation or bias ops are fused into a single HLO compute operation.
- **Tiling**: The dot is decomposed into tiles matching the MXU size (128×128 pre-v6e, 256×256 from v6e onward). XLA selects tiling dimensions and unrolling factors.
- **Memory scheduling**: XLA's memory space assignment schedules the weight matrix `W` to be loaded from HBM into VMEM ahead of the dot computation, overlapping the DMA transfer with other operations.

**Step 5 — TPU binary emission (inside libtpu)**
XLA's TPU backend lowers the optimized HLO to LLO and emits VLIW instruction bundles. Each 8-slot VLIW packet schedules one operation per functional unit (MXU load/execute, VPU, DMA read, DMA write, scalar, etc.) per clock cycle. The binary is cached by content hash.

**Step 6 — Dispatch and execution**
`libtpu` uploads the binary to the TPU, enqueues input buffers in HBM, and signals the TPU to execute. The TPU reads weights into VMEM via DMA, streams activations through the MXU systolic array in a weight-stationary fashion, accumulates results in the MXU's output registers, and writes results back to HBM via DMA.

---

## Hardware Interface

### XLA ↔ MXU Mapping

XLA's dot/convolution tiling strategy is tuned to the MXU's dimensions:
- **TPU v1–v5e**: MXU is 128×128; XLA tiles GEMM K-dimension in 128-element chunks
- **TPU v6e+ (Trillium/Ironwood)**: MXU expanded to 256×256; XLA re-tiles to 256×256 blocks, quadrupling MACs per MXU cycle at the same clock speed

XLA's layout assignment produces column-major weight layouts that match MXU's weight-stationary dataflow: weights are loaded once into VMEM and held stationary while activations stream through the systolic array.

### Pallas / Mosaic ↔ VMEM

Mosaic generates DMA commands that move HBM tiles into VMEM scratchpad buffers. Double-buffering patterns (N-step prefetch) ensure the DMA for tile N+1 is in-flight while the MXU/VPU processes tile N, hiding HBM latency.

VMEM capacity (per TensorCore, approximate):
- TPU v4/v5: ~16–32 MB VMEM
- TPU v6e: ~32 MB VMEM per TensorCore
- TPU v7 (Ironwood): Increased VMEM to match expanded MXU bandwidth

CMEM (scalar memory) is used by Mosaic for loop counters, tile indices, and small lookup tables that do not need vectorized access.

---

## Key Findings

1. **StableHLO as the portability seam**: The ML ecosystem converges on StableHLO as the stable interchange format. JAX, PyTorch/XLA, and TensorFlow all serialize to StableHLO, allowing the XLA compiler and hardware backends to evolve independently. This is the TPU analogue of PTX's role in the NVIDIA stack.

2. **Operator fusion is the dominant optimization**: XLA's fusion pass is the single highest-impact transformation for TPU. Fusing element-wise operations into the output of a matrix multiply eliminates HBM roundtrips for intermediate tensors and significantly improves arithmetic intensity, moving workloads into the compute-bound regime.

3. **Pallas fills the custom kernel gap**: XLA's automatic tiling is optimal for standard GEMM and convolution patterns but cannot match hand-tuned kernels for attention (FlashAttention-style), sparse operations, or custom activations. Pallas/Mosaic provides the escape hatch, enabling JAX-level kernel development without requiring knowledge of VLIW assembly or TPU ISA.

4. **Shardy completes the SPMD compilation story**: With Shardy's March 2026 migration complete in JAX, the entire sharding workflow is now expressed in MLIR-native terms. Shardy annotations on StableHLO ops propagate through the graph and generate collective communication ops, making tensor/pipeline/data parallelism a compiler concern rather than a user concern.

5. **Mosaic's pipeline scheduling is the kernel-performance lever**: The key to peak Pallas performance is Mosaic's ability to overlap HBM DMA transfers with MXU compute. Kernel authors control pipeline depth via `pl.pallas_call` pipeline stages. Misalignment between tile sizes and MXU dimensions causes padding overhead and is the primary source of sub-optimal Pallas kernel performance.

6. **VLIW instruction packing requires no programmer intervention**: Unlike CUDA where kernel authors must be aware of warp scheduling and instruction-level parallelism, the TPU's VLIW instruction packing is entirely compiler-driven (by Mosaic and XLA's TPU backend). The programmer's only lever is at the tile-granularity level (block shapes, pipeline stages).

7. **LLO is the shared substrate**: Both XLA (for regular ops) and Mosaic (for custom kernels) emit to LLO (Low Level Operations), Google's internal TPU low-level IR. LLO scheduling is what ultimately produces the VLIW instruction stream, giving the compiler complete control over functional unit utilization.

---

## Relation to Hardware

The compilation stack directly reflects TPU hardware constraints:

- **MXU tile size → XLA tiling strategy**: The jump from 128×128 to 256×256 MXU in v6e forced XLA to retile all GEMM shapes. Code compiled for v5 does not run optimally on v6e without recompilation.
- **VMEM capacity → operator fusion budget**: XLA's fusion pass is bounded by VMEM. Fused kernels must fit their intermediate activations in VMEM; the memory space assignment pass spills to HBM when VMEM is exhausted.
- **VLIW slots → instruction-level parallelism**: The TPU's 8-slot VLIW issue width drives Mosaic to schedule independent operations (MXU compute + DMA load + scalar increment) into the same VLIW packet, achieving high functional unit utilization.
- **ICI topology → Shardy/GSPMD collective selection**: Shardy's collective insertion is topology-aware. On 3D torus ICI (v4/v5p/v7), bidirectional AllReduce over ring is preferred; on 2D torus (v5e/v6e), 2D AllReduce decomposition matches the physical topology.

---

## Sources

- [OpenXLA/XLA Repository](https://github.com/openxla/xla)
- [StableHLO Specification](https://openxla.org/stablehlo/spec)
- [Pallas Design Document — JAX Documentation](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Writing TPU kernels with Pallas — JAX Documentation](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [Shardy Overview — OpenXLA Project](https://openxla.org/shardy/overview)
- [Shardy JAX Migration — JAX Documentation](https://docs.jax.dev/en/latest/shardy_jax_migration.html)
- [From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
- [JAX/XLA and Pallas — MaxText Documentation](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/jax_xla_and_pallas.html)
- [GSPMD Paper — arXiv:2105.04663](https://arxiv.org/abs/2105.04663)
- [Shardy LLVM Dev Talk Slides](https://llvm.org/devmtg/2024-10/slides/techtalk/Chrzaszcz-Jiang-Shardy.pdf)
- [Building Production AI on Cloud TPUs with JAX](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)
- [JAX JIT Compilation — JAX Documentation](https://docs.jax.dev/en/latest/jit-compilation.html)
