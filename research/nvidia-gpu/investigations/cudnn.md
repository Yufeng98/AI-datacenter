# cuDNN — Investigation Report

**Resource:** NVIDIA cuDNN (Closed-source) + cudnn-frontend (Open-source)
**URLs:** https://developer.nvidia.com/cudnn · https://github.com/NVIDIA/cudnn-frontend
**Version:** cuDNN Frontend v1.22.0 (commit `97f6cb3b`) / cuDNN backend v9.x
**As of:** 2026-04-05

---

## Overview

cuDNN (CUDA Deep Neural Network library) is NVIDIA's closed-source, GPU-accelerated library of primitives for deep neural networks. It provides hand-tuned implementations of convolutions, matrix multiplications, normalization layers, attention operations, and activation functions that execute directly on NVIDIA GPUs. cuDNN occupies the Op Library layer of the NVIDIA software stack: it sits above raw CUDA/CUTLASS kernel templates and below ML frameworks (PyTorch, JAX, TensorFlow), which call into it via their respective backends. As of v8, cuDNN introduced the Graph API — a declarative, dataflow-graph programming model that replaced its legacy per-operation imperative API and enabled systematic multi-operation fusion. The companion open-source `cudnn-frontend` (header-only C++) provides a high-level C++ and Python wrapper over cuDNN's verbose C backend API, reducing boilerplate 5–10x while adding autotuning support, errata filters, and plan caching.

---

## Architecture

### Software Layer Position

```
ML Frameworks (PyTorch / JAX / TensorFlow)
         |  torch.backends.cudnn / jax.lib.xla_bridge
         v
    cuDNN Frontend API (cudnn-frontend, header-only C++)
         |  Graph::build_operation_graph() → query_cudnn_heuristics()
         v
    cuDNN Backend API (libcudnn*.so, closed-source)
         |  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
         v
    cuDNN Engines (precompiled + runtime-compiled)
         |  cudnnBackendExecute() → variant pack → device pointers
         v
    Tensor Cores / CUDA Cores  (via CUTLASS-generated PTX internally)
```

### Key Modules (cudnn-frontend v1.22.0)

| Module / Header | Role |
|---|---|
| `graph_interface.h` — `Graph` class | Top-level user-facing graph object; holds nodes and tensor attributes |
| `node_interface.h` — `INode` | Abstract base for all operation nodes (CRTP pattern) |
| `cudnn_frontend_OperationGraph.h` — `OperationGraph_v8` | Backend descriptor wrapping up to 50 fused ops |
| `cudnn_frontend_Heuristics.h` — `EngineHeuristics_v8` | Queries cuDNN for ranked engine configs (Mode A / Mode B) |
| `cudnn_frontend_Engine.h` / `EngineConfig` | Represents a concrete backend engine and its knobs |
| `cudnn_frontend_ExecutionPlan.h` | Compiled plan ready for `cudnnBackendExecute()` |
| `cudnn_frontend_VariantPack.h` | Binds device pointers + workspace to a plan at execution time |
| `plans.h` — `execute()` | Launches execution; assembles variant pack, calls backend |
| `experimental/sm90_sdpa_prefill_engine.h` | Hopper (sm90) SDPA prefill kernels |
| `experimental/sm100_sdpa_prefill_engine.h` | Blackwell (sm100) SDPA prefill kernels |

### Node Types (graph::INode subclasses)

Declared in `graph_interface.h` and implemented under `include/cudnn_frontend/node/`:

- **Convolution nodes:** `conv_fprop`, `conv_dgrad`, `conv_wgrad`
- **Normalization nodes:** `batchnorm`, `layernorm`, `rmsnorm`, `instancenorm`, `adaptive_layernorm`
- **Attention nodes:** `scaled_dot_product_flash_attention`, `sdpa_fp8_bwd`
- **Linear algebra:** `matmul`, `matmul_fp8`, `moe_grouped_matmul`
- **Elementwise:** `pointwise`, `reduction`, `reshape`, `slice`, `resample`
- **Quantization:** `block_scale_quantize`, `block_scale_dequantize`
- **Utility:** `rng`, `genstats`, `concatenate`

### Frontend vs Backend Split

The **Frontend API** (open-source, header-only) handles:
- Graph construction and tensor attribute tracking
- Heuristic queries and plan selection/caching
- Errata filtering (excluding known-buggy engines per GPU)
- Autotuning (`cudnnFindPlan()` timing loop)

The **Backend API** (closed-source, `libcudnn_graph.so`, etc.) handles:
- Engine compilation (runtime NVRTC or precompiled PTX)
- Support surface evaluation (which engine handles which graph pattern)
- Actual kernel dispatch

cuDNN v9 splits its shared library into components: `libcudnn_graph.so`, `libcudnn_engines_precompiled.so`, `libcudnn_engines_runtime_compiled.so`, `libcudnn_heuristic.so`.

### Engine Classes

Four engine classes exist in the backend:

1. **Pre-compiled single-operation engines** — Traditional hand-written kernels for isolated ops (legacy path)
2. **Generic runtime fusion engines** — Support surfaces indexed 70/80/90; cover most fusion patterns via NVRTC compilation at first use
3. **Specialized runtime fusion engines** — Tuned for specific high-value patterns (e.g., MHA for BERT/GPT)
4. **Specialized pre-compiled fusion engines** — AOT-compiled for fixed problem shapes (highest performance ceiling)

---

## Data Flow

### Traced Path: Fused Flash Attention (SDPA) in PyTorch on H100

**Step 1 — PyTorch call:**
```python
torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
```
PyTorch dispatches to `CuDNNSDPABackend` when `torch.backends.cuda.cudnn_sdp_enabled()` is True and the tensor shapes/dtypes are supported. On H100+, this path provides up to 75% speedup over FlashAttention v2.

**Step 2 — Frontend graph construction:**
```cpp
cudnn_frontend::graph::Graph g;
auto Q_t = g.tensor(Q_attrs);  // Tensor_attributes: shape, stride, dtype
auto K_t = g.tensor(K_attrs);
auto V_t = g.tensor(V_attrs);
auto [O, softmax_stats] = g.sdpa(Q_t, K_t, V_t, sdpa_attrs);
g.validate();
```
The `SDPANodeBase` node (`scaled_dot_product_flash_attention.h`) expands internally into a subgraph of matmul → score-modifier pointwise ops (causal mask, ALiBi, etc.) → softmax → matmul nodes.

**Step 3 — Backend operation graph build:**
```cpp
g.build_operation_graph(handle);
// Calls cudnnBackendFinalize(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)
// OperationGraph_v8 holds up to MAX_OPGRAPH_OPS=50 fused operations
```

**Step 4 — Heuristic engine selection:**
```cpp
g.create_execution_plans({HeurMode_t::A});
// Internally: query_cudnn_heuristics_impl() →
//   cudnnBackendSetAttribute(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_HEUR_MODE_A)
//   Returns engine configs ranked best-to-worst by expected performance
```
Mode A: fast CPU-side lookup, handles most patterns.
Mode B: more accurate but higher CPU latency.
Autotuning alternative: `cudnnFindPlan()` times all candidate configs on actual hardware.

**Step 5 — Plan compilation:**
```cpp
g.build_plans(handle);
// For runtime fusion engines: NVRTC compiles specialized kernel
// For precompiled engines: loads PTX from libcudnn_engines_precompiled.so
```

**Step 6 — Execution:**
```cpp
g.execute(handle, {Q_uid: Q_ptr, K_uid: K_ptr, V_uid: V_ptr, O_uid: O_ptr}, workspace_ptr);
// Plans.h: execute() → create_variant_pack() → cudnnBackendExecute()
// Variant pack binds tensor UIDs to device pointers + workspace
```

**Step 7 — Kernel execution on Tensor Cores:**
- sm90 (H100): uses `sm90_sdpa_prefill_engine` with WGMMA instructions + TMA for async data prefetch
- Tile scheduling: Q/K/V blocks loaded to SMEM via TMA, MMA computed on Tensor Cores
- Fused softmax + second matmul in single kernel, no intermediate global memory write

---

## Hardware Interface

### Tensor Core Utilization

cuDNN's internal engines map matrix operations directly onto Tensor Core MMA instructions:

- **Ampere (sm80):** `mma.sync.aligned` for FP16/BF16/INT8/TF32; 16×8×16 tile shape
- **Hopper (sm90):** `wgmma.mma_async` (WGMMA) for asynchronous warp-group MMA; supports FP8 (E4M3/E5M2), FP16, BF16, TF32; delivers 4× peak throughput vs A100 in FP8
- **Blackwell (sm100):** 5th-gen Tensor Cores with native MXFP8/NVFP4 block-scaled MMA; `block_scale_quantize`/`block_scale_dequantize` cuDNN nodes feed these natively

### Tensor Memory Accelerator (TMA) on Hopper/Blackwell

TMA (`cp.async.bulk.tensor`) allows a single thread to initiate asynchronous 1D–5D tensor transfers between global and shared memory. cuDNN's sm90/sm100 SDPA engines use TMA to pipeline Q/K/V tile loads while Tensor Cores compute previous tiles — overlapping memory and compute latency.

### Shared Memory Tiling

cuDNN engines implement software-managed shared memory tiling:
- Q, K, V tiles loaded to SMEM in double-buffered fashion (pipeline depth ≥ 2)
- Score tiles (QK^T) computed in SMEM, softmax applied in registers/SMEM
- Output accumulation in registers, written back through SMEM to GMEM

### Memory Layout Awareness

The `Tensor_attributes` object carries strides (not just shapes), allowing cuDNN to handle non-contiguous and transposed tensors without explicit copies. The `Reorder_Tensor` header handles INT8 filter layout reordering for convolution engines.

---

## Key Findings

1. **Graph-Based Fusion is the Core Value Proposition.** The Graph API (introduced v8, matured in v9) enables cuDNN to fuse sequences of operations — convolution + bias + activation, QK^T matmul + masking + softmax + V matmul — into a single kernel or minimal kernel set, eliminating intermediate global memory round-trips. The `OperationGraph_v8` supports up to 50 fused operations.

2. **Two-Tier Engine Selection: Heuristics + Autotuning.** At graph-build time, `EngineHeuristics_v8` uses Mode A (fast) or Mode B (accurate) to rank engine configs by predicted performance without running them. For production deployments, `cudnnFindPlan()` iterates all candidate configs on actual hardware to find the empirical best — a one-time cost amortized across inference runs.

3. **FP8 and Block-Scaled Formats on Hopper/Blackwell.** cuDNN v9 added `matmul_fp8` node and `block_scale_quantize`/`block_scale_dequantize` nodes. On Hopper, FP8 (E4M3/E5M2) Tensor Cores provide 4× throughput vs FP16. On Blackwell, MXFP8 (32-element blocks with shared E8M0 scale) and NVFP4 map to 5th-gen Tensor Core block-scaled MMA, delivering further 2–4.6× speedup.

4. **Curated SDPA Kernel Library.** The `experimental/` directory ships prebuilt prefill-phase SDPA kernels compiled for specific SM generations (sm90, sm100) and head dimensions (d64, d128). These bypass the heuristic/NVRTC path entirely, providing maximum performance for known transformer attention shapes.

5. **Plan Caching and Dynamic Shapes (v9.4+).** `ExecutionPlanCache` and `kernel_cache` (enabled when cuDNN ≥ 9.4.0) allow serialized plans to be reloaded across process restarts, avoiding repeated NVRTC compilation. Dynamic shape support in v9.4+ allows a single compiled plan to serve variable sequence lengths within defined bounds.

6. **MoE and Paged Attention (v9 Extensions).** The `moe_grouped_matmul` and `paged_cache_load` nodes were added in cuDNN v9 to support mixture-of-experts and KV-cache paging patterns common in LLM inference, reflecting cuDNN's evolution from training-centric to inference-centric workloads.

---

## Relation to Hardware Architecture

| Generation | New HW Feature | cuDNN Response |
|---|---|---|
| Volta (sm70) | 1st-gen Tensor Cores (FP16) | Added Tensor Core conv/GEMM engines; initial Graph API concepts |
| Ampere (sm80) | 2nd-gen TC (BF16, TF32, INT8, sparse), CUDA async copy | Sparse convolution support, async data pipelining in engines |
| Hopper (sm90) | 3rd-gen TC (FP8), TMA, WGMMA, Thread Block Clusters | FP8 matmul node, `sm90_sdpa_prefill_engine`, TMA-based SDPA, fused Flash Attention delivering 75% speedup over FAv2 on H100 |
| Blackwell (sm100) | 5th-gen TC (MXFP8/NVFP4 block scaling), enhanced TMA | `block_scale_quantize/dequantize` nodes, `sm100_sdpa_prefill_engine`, `sm100_rms_norm_silu_engine`, native MXFP8/NVFP4 MMA paths |

Each generation forced cuDNN's graph-lowering to expose new data types and memory access patterns that the backend engines could exploit — with the Graph API abstraction shielding framework authors from hardware-specific changes.

---

## Dependency Graph

```
PyTorch (torch.nn.functional.sdpa)
  JAX (jax.lib.xla_bridge)
  TensorFlow (tf.nn.conv2d)
        |
  cudnn-frontend (header-only C++ / Python bindings)
        |  graph::Graph → OperationGraph_v8 → EngineHeuristics_v8
        v
  cuDNN Backend API (libcudnn_graph.so)
        |  heuristic.so · engines_runtime_compiled.so · engines_precompiled.so
        v
  cuDNN Engines (NVRTC-JIT or precompiled PTX)
        |
  CUDA Runtime (cudnnBackendExecute → cuLaunchKernel)
        |
  GPU Hardware: Tensor Cores + TMA + Shared Memory
```
