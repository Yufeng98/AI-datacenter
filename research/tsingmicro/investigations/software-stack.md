# Tsingmicro TX81 Software Stack Investigation

*as_of: 2026-09-13*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*

---

## Overview

Tsingmicro brands its whole software stack **RAISA**. The RAISA *brand* is proprietary and essentially undocumented publicly — there is **no developer portal** (`developer.tsingmicro.com`, `doc.tsingmicro.com`, `docs.tsingmicro.com`, `open.tsingmicro.com` all fail to resolve) and no public SDK download.

Despite that, an unusually large and *genuinely informative* slice of the stack is open source, because Tsingmicro upstreamed a **complete Triton backend into BAAI FlagOS / FlagTree**, plus vendor backends into **FlagGems** (operators) and **FlagCX** (collectives). The open portion is a compiler front/middle-end plus an operation-level hardware model. Everything below the kernel compiler — LLVM backend, intrinsic library, firmware, host runtime, driver, PyTorch device plugin, collectives implementation — is a **closed binary** distributed from an internal vendor artifact server.

The device is registered with PyTorch as **`txda`** via **PrivateUse1**.

### Two traps recorded up front

**Attribution trap.** FlagTree's `third_party/rpu` backend on the `main` branch is the **Rhino RPU from Huixi Intelligence (辉羲智能)** — a different company that also uses the term "RPU". Tsingmicro's backend is **`third_party/tsingmicro` on the `triton_v3.3.x` branch only**.

**Product-line trap.** **TS.Knight** (Knight-Quantize → Knight-RNE-Compiler → Knight-RNE-Simulator → Knight-RNE-Profiling → Knight-Finetune, targeting the **RNE** engine) is the **TX5 edge** flow and has nothing to do with TX8. There is no public TS.Knight support for TX8/TX81. The TX8 flow is Triton/FlagTree plus the closed Kuiper runtime.

### The compatibility claim, stated precisely

The vendor claims "支持 CUDA 生态算子的平滑兼容" and "零代码迁移" (smooth CUDA-ecosystem operator compatibility; zero-code migration). **Nothing public demonstrates CUDA source or binary compatibility.** The real compatibility surface is two API *shapes*:

1. A **HIP-shaped host runtime** — `txLaunchKernelGGL` (mirroring `hipLaunchKernelGGL`), `txMalloc`, `txStreamCreate`, `txEventRecord`, …
2. An **NCCL-shaped collectives library** — TCCL (`tcclComm_t`, `tcclRedOp_t`, …)

plus Triton as a portable kernel language. That is a real but materially weaker compatibility story than the marketing implies.

---

## Layer Table

| # | Layer | Component | Open/Closed | Detail |
|---|---|---|---|---|
| 1 | Framework integration | **torch_txda** — PyTorch device plugin | **Closed** (wheel) | `torch_txda-0.1.0+20260416.b8f53e8a-cp310-cp310-linux_x86_64.whl`. Registers device `txda` via **PrivateUse1**. Exposes `torch.txda.{is_available, current_device, set_device, current_stream, get_device_properties, get_device_capability}`; streams as `torch.txda.current_stream(dev).txda_stream` |
| 1 | Framework integration | **FlagGems `_tsingmicro` backend** | **Open** | `VendorDescriptor(vendor_name="tsingmicro", device_name="txda", device_query_cmd="tsm_smi", dispatch_key="PrivateUse1", fp64_enabled=False, int64_enabled=False)`; `tune_configs.yaml` (1313 lines, ~37 op families incl. attention / attention_bwd / bmm / mm / softmax / layernorm); `heuristics_config_utils.py` |
| 1 | Framework integration | **vllm-plugin-FL**, **sglang-plugin-FL** ("support txda") forks | Open forks, thin | `github.com/tsingmicro-public-e/*`, last pushed Jul 2026. No txda-specific directories visible in the READMEs |
| 1 | Framework integration | **FlagScale**, **Megatron-LM-FL**, **TransformerEngine-FL**, **PyTorch-Plugin-FL** forks | Open forks, thin | Training-side ecosystem forks maintained through Aug 2026. Note: upstream `PyTorch-Plugin-FL` contains **no** txda backend (only CUDA/MetaX/Ascend), so the fork is aspirational |
| 1 | Framework integration | **txops** — vendor PyTorch op wheel | **Closed** | `txops-0.1.0+20260508.60287151-py3-none-any.whl` |
| 1 | Framework integration | Vendor claims PyTorch / vLLM / SGLang + native FlagOS support, "CUDA 生态兼容", "零代码迁移" | — | **Marketing.** See "the compatibility claim, stated precisely" above |
| 2 | Graph capture | **None public** | — | No torch.compile / Dynamo / FX integration is visible. The public execution model is per-kernel Triton JIT plus FlagGems op dispatch, **not** whole-graph capture |
| 3 | Graph compiler | **RAISA graph-optimisation layer** — vendor's layer-3 description: "根据上层需求对计算图进行全局优化，实现上层应用与硬件的高性能对接" | **Closed / undocumented** | Component name, IR, and whether it exists as a distinct artifact are **not disclosed**. Op-fallback env vars in the release scripts (`TXDA_FALLBACK_CPU_OPS` lists ~60 ops incl. `sort, gather, index, pad, embedding_backward, cat, add, mul, sum`; `TXDA_SKIP_OPS`) indicate a still-maturing operator surface |
| 4 | **Kernel compiler** | **FlagTree `tsingmicro` backend**, branch `triton_v3.3.x`, `third_party/tsingmicro` (609 files) | **Open** | Integrated 2025-06-06 with CI/CD; refreshed through 2026-05/06. Class `TXDABackend`, `GPUTarget(backend="txda")`, `binary_ext="so"`. Co-authored with **Terapines Technology (Wuhan) / 兆松科技** — every `.td`/`.cpp` carries their copyright header |
| 4 | Compiler stages | `ttir` → `coreir` → `txir` → `llir` → `so` | Open | Full pass list below |
| 4 | MLIR dialects | **Address**, **triton-shared** (vendored), **MagicKernel (MK)** + MagicKernelFunc + MagicKernelInstr, **Tx81**, **DSA** (`third_party/tle`) | Open | `MagicKernelOps.td` 30 KB (~90 ops); `Tx81Ops.td` **47 KB (~150 ops)** — effectively a published operation-level model of the accelerator |
| 4 | Compiler tools | `tsingmicro-opt`, `tsingmicro-lsp`, `tsingmicro-reduce`, `tsingmicro-llvm-opt`, `tsingmicro-tensor-layout`, `tx-profiler` | Open source, built locally | in `bin/` |
| 4 | LLVM | **tsingmicro-llvm21** vendor fork | **Closed binary** | Bundle `tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64_v0.6.0` (LLVM 21-based; also `llvm-a66376b0`). Source on private `gitlab.tsingmicro.com/triton-based-projects/llvm-project`. Internal codename **"ZTC"** appears in header guards (`ZTC_CONVERSION_LINALG_TO_MK_H`) |
| 4 | Backend codegen | RISC-V | Open flow, closed libs | `clang++ --target=riscv64-unknown-elf -march=rv64imfdc -O2 -c` → `riscv64-unknown-elf-gcc -shared -march=rv64imfdc -mabi=lp64d` linking `-lcommon_util -linstr_rcs1 -llibc_stub -lvr` → `kernel.so`. Toolchain = **T-Head XuanTie-900 GCC ELF newlib V2.8.0** |
| 5 | **Kernel language** | **Triton 3.3** (Python DSL) | Open | The public kernel language for TX81 |
| 5 | Kernel language | **`triton.language.extra.txda.libdevice`** (51 KB) | **Open** | Device-function library mapping Triton math builtins to Tx81 intrinsics |
| 5 | Kernel language | **TLE — Triton Language Extensions** (`triton.experimental.tle`) | **Open** (FlagTree upstream) | Three tiers: TLE-Lite / TLE-Struct / TLE-Raw. Tsingmicro uses **TLE-DSA** and **TLE-distributed** |
| 5 | Kernel language — DSA | `tle.language.dsa.alloc(shape, dtype, scope=spm)`, `.copy(GM↔SPM, SPM↔SPM)`, `.local_ptr(buf, indices)`, `tle.dsa.memory_space(t,"spm")` | Open | Upstream docstring: *"Scratch Pad Memory – the on-chip SRAM exposed by **TsingMicro TX8**. This is the primary storage scope for DSA kernels and serves the same conceptual role as NVIDIA shared memory (smem)."* |
| 5 | Kernel language — distributed | `tle.device_mesh(topology / _shape / _dim_names / _physical_ids)`, `tle.sharding` (split/broadcast/partial specs), `tle.ShardedTensor`, `tle.reshard`, `tle.shard_id(mesh, axis)`, `tle.remote(buf, tile, scope=MESH)`, `tle.distributed_barrier(mesh)`, `tle.distributed_dot` | Open | **This is how the mesh is programmed from Triton.** The 4096-element ring-GEMM example builds a 16-tile `device_mesh` with an explicit Hamiltonian physical ring and performs a shift-N ring matmul with `tle.remote` + `distributed_barrier` |
| 5 | Kernel language — C/C++ | Vendor claims "提供自研 **C/C++ 接口语言**" | **Closed / not public** | Only the Triton path is public |
| 5 | CRT (compiler runtime) | `third_party/tsingmicro/crt/` — ~130 C files | **Open** | Per-op runtime implementations `__Gemm`, `__Rdma`/`__Wdma`/`__Rdma4d`, `__Send`, `op_gelu.c` (31 KB), `pow.c` (21 KB), `relation.c` (17 KB), all dtype conversions incl. MXFP, DMA out-of-bounds checking with client-pointer headers (magic `0x54445841` = "TDXA") |
| 6 | Tensor API / op libraries | **FlagGems** (BAAI Triton op library) with open `_tsingmicro` backend | Open | The primary public operator library |
| 6 | Op libraries | **oplib_rcs1**, **tx8be-oplib**, **txops**, "自研高性能算子库" | **Closed** | Referenced in `CMakeLists.txt` (`rcs1fw-rtt/include/components/oplib_rcs1/riscv/riscv/include`) and `build_tx8_deps.sh` |
| 7 | **Runtime (host)** | **`tx_runtime.h` / `libhpgr`** — installed at **`/usr/local/kuiper`** ("Kuiper" host SDK) | **Closed** | **CUDA/HIP-shaped C API** — full surface below |
| 7 | Runtime (device) | **tx81fw / rcs1fw-rtt** firmware on **RT-Thread SMP** via T-Head **YoC** (`tx8-yoc-rt-thread-smp`) | **Closed** | Bundled in `tx8_depends_dev_*` tarballs |
| 7 | Runtime (intrinsics) | **`libinstr_rcs1.a`** + `instr_def.h`, `instr_adapter.h`, `instr_operator.h`, `common_base.h` | **Closed** | The accelerator command/instruction ABI |
| 7 | Simulation | `triton_cmodel`, `tx8be_op_cmodel`, **`neuralcore_qemu`** | **Closed** | `USE_SIM_MODE=1` builds against a C-model + QEMU neural-core simulator instead of hardware |
| 7 | Profiling | **tx-profiler** (LLVM IR pass, open) + **profiling_tool v5.5.0 / v5.6.0** (closed) + `rcs_profiling`, `hrt_profiler`, `tsm_profiler.h` | Mixed | The open pass injects `addOrderProfile` / `TsmWaitfinish` / `printOrderByEvent` around named trace points; env `ENABLE_PROFILING`, `TSM_PROFILER_EN`, `TRACE_POINTS` |
| 7 | Collectives | **TCCL — TsingMicro Communication Collectives Library** | **Closed** (`tccl.h`) | NCCL-shaped. Open FlagCX adaptors `ccl/tccl_adaptor.cc` + `device/tsmicro_adaptor.cc` + `include/tsmicro_adaptor.h` under `USE_TSM_ADAPTOR`. Vendor string `"TSMICRO"`. FlagCX matrix: **all 12 collective ops in both homogeneous and heterogeneous modes** |
| 8 | **Driver** | Kernel driver — **not public** | **Closed** | Ships inside the Kuiper SDK. Management CLI **`tsm_smi`** (the nvidia-smi analogue). Container image `hub.tsingmicro.com/tx8/ubuntu/v5.7.0.0524:tsingmicro_release`. Vendor RAISA layer 4: "提供完整配套工具和开发套件，涵盖**驱动、容器、虚拟化**及配套开发者工具链". Driver module name and IOCTL interface: **not disclosed** |
| 9 | **ISA** | **Two-level.** (a) Scalar: **RISC-V RV64IMFDC / lp64d**, T-Head XuanTie-900 core. (b) Accelerator: coarse-grained **command descriptors** | (a) open standard, (b) **closed** | Descriptor classes seen: `I_NEUR` (neural/GEMM/conv), `I_RDMA` (DMA). Dispatch `RcsExecute(&inst)`, wait `RcsWaitfinish()` / `TsmWaitfinish()`. Builder API `TsmNewGemm/TsmNewConv, AddInput/AddWeight/AddBias/AddOutput, ConfigMKN, ConfigBatch, SetPsum, SetTransflag, SetQuant, SetSparse, SetPads/SetUnPads/SetKernelStrides/SetDilations, SetNegativeAxisScale/SetPositiveAxisScale, EnableRelu/EnableLeakyRelu`. **Bit encodings, opcode list and register map: not disclosed** |

---

## Compiler Pipeline in Detail (open source, verifiable)

`backend/compiler.py` defines five stages:

```
Triton AST
  └─ ttir   : inliner, ttir-combine, canonicalize, reorder-broadcast, CSE, LICM, symbol-DCE
  └─ coreir : tsingmicro-opt
        --triton-to-core-dialects        # triton-shared: → linalg/memref/arith/affine
        --tle-to-mk                      # TLE distributed/DSA ops → MagicKernel
        --dsa-memory-to-core             # SPM placement decisions
        --linalg-tiling
        --core-dialects-to-mk=precision-mode={0|1|2}
        --linalg-fusion
        --legalize-tensor-form-loops
        --one-shot-bufferize
        --convert-bufferization-to-memref
        --materialize-strided-linalg-inputs
        --cse --canonicalize
        [--mk-pipeline=num-stages=N]     # SPM double-buffered software pipelining
        [--mk-loop-bound-canonicalize]
  └─ txir   : tsingmicro-opt
        --spmd-allocate-shared-memory    # SPM allocator, emits triton_tsm.spm_use
        --expand-strided-metadata --lower-affine
        --mk-to-tx81                     # MK → Tx81 hardware ops (103 KB of patterns)
        --tx81-insert-barrier            # dependency-driven TsmWaitfinish insertion
        --tx81-resolve-dma-base-addr     # trace DDR base for DMA bounds checking
        --cse --canonicalize
  └─ llir   : --tx81-memref-to-llvm --addr-to-llvm --convert-scf-to-cf
              --convert-math-to-llvm --convert-math-to-libm --convert-cf-to-llvm
              --convert-func-to-llvm --finalize-memref-to-llvm
              --kernel-arg-buffer                          # rewrite sig to kernel(ptr)
              --tx81-to-llvm[=gather-scatter-async=true]   # 118 KB of lowering
              --convert-arith-to-llvm --reconcile-unrealized-casts
              --canonicalize --export-kernel-symbols
              → mlir-translate --mlir-to-llvmir
  └─ so     : clang++ --target=riscv64-unknown-elf -march=rv64imfdc -O2 -c
              → riscv64-unknown-elf-gcc -shared -mabi=lp64d
                -lcommon_util -linstr_rcs1 -llibc_stub -lvr -lm
```

**`precision-mode`**: `0` disabled; `1` preserves i64 precision; `2` preserves full integer precision (i32/i64 handled by the RISC-V CPU). Default `2` in the vendor's release test scripts.

**Largest pattern files** (a proxy for where the real engineering sits): `LinalgToMK.cpp` **186 KB**, `Tx81ToLLVM.cpp` **119 KB**, `MKToTx81.cpp` **103 KB**, `ConversionPatterns.h` (triton-shared) 96 KB, `MKPipelinePass.cpp` 92 KB, `StructuredToMemref.cpp` 60 KB.

The compilation product is a **RISC-V shared object (`.so`)**, not a GPU-style cubin — the kernel is scalar RISC-V code that builds and dispatches engine command descriptors.

---

## Host Runtime API (`tx_runtime.h`, closed — surface reconstructed from the FlagCX adaptor)

A near-1:1 rename of the CUDA/HIP runtime:

`txSetDevice`, `txGetDevice`, `txGetDeviceCount`, `txGetDeviceProperty`, `txGetDeviceByPCIBusId`, `txDeviceSynchronize`, `txMalloc`, `txFree`, `txMallocHost`, `txFreeHost`, `txMemcpy`, `txMemcpyAsync`, `txMemcpyKind` (`txMemcpyHostToDevice` / `DeviceToHost` / `DeviceToDevice`), `txMemset`, `txStreamCreate`, `txStreamDestroy`, `txStreamQuery`, `txStreamSynchronize`, `txStreamWaitEvent`, `txEventCreate`, `txEventDestroy`, `txEventRecord`, `txEventQuery`, `txEventSynchronize`, `txEventElapsedTime`, `txIpcGetMemHandle`, `txIpcOpenMemHandle`, `txIpcCloseMemHandle`, `txMemGetHandleForAddressRange`, `txLaunchHostFunc`, **`txLaunchKernelGGL`**; types `txStream_t`, `txEvent_t`, `txIpcMemHandle_t`, `txError_t`, `txDeviceProperty`; status `TX_SUCCESS`.

Note the **HIP-flavoured naming** (`txLaunchKernelGGL` ↔ `hipLaunchKernelGGL`). This, plus NCCL-shaped TCCL, is the concrete substance behind the "CUDA 生态兼容" marketing — **API-shape compatibility, not CUDA source or binary compatibility**.

---

## Build / Install (publicly documented)

- Wheel index: `https://resource.flagos.net/repository/flagos-pypi-hosted/simple` → `pip install flagtree===0.6.0+tsingmicro3.3`
- Source-build environment: `FLAGTREE_BACKEND=tsingmicro`, `TX8_DEPS_ROOT=~/.flagtree/tsingmicro/tx8_deps`, `LLVM_SYSPATH=~/.flagtree/tsingmicro/tsingmicro-llvm21-…-x64`, `LLVM_BINARY_DIR`, `PYTHONPATH=$LLVM_SYSPATH/python_packages/mlir_core`
- Dependency bundles: `tx8_depends_dev_20260507_104051_v0.6.0`, `build-deps-triton_3.3.x-linux-x64`
- Docker: `flagtree-tsingmicro3.3-py310-torch2.7.0-ubuntu22.04:20260604163331-clean`; vendor image `hub.tsingmicro.com/tx8/ubuntu/v5.7.0.0524:tsingmicro_release`
- Platform: Python 3.10, PyTorch 2.7.0, Ubuntu 22.04, x86_64 host
- Public CI: `.github/workflows/tsingmicro3.3-build-and-test.yml`, self-hosted runner `tsingmicro3.3`, runs 14 example/test scripts **on real hardware**. **The TLE NoC ring-GEMM test is commented out with `# TODO: fix`** — the multi-tile distributed path is not CI-green.
- The vendor artifact server `http://172.50.1.66:8082/artifactory/…` and `gitlab.tsingmicro.com` are internal and unreachable, but the tarball names leak the component inventory (`tx81fw_202512041135_b731cf`, `tx8-yoc-rt-thread-smp-202603031631-88bfb9`, `profiling_tool_v5.6.0_release_2026-0228`).

---

## Vendor's Own Four-Layer Description of RAISA (verbatim, tsingmicro.com)

1. "适配行业应用及生态框架，主要包括各行业 Agent 及模型应用，主流训练与推理框架"
2. "支持应用层适配，提供**自研 C/C++ 接口语言**且支持通用 **Triton** 编程语言，提供**自研高性能算子库**，支撑客户快速迁移应用"
3. "根据上层需求对**计算图进行全局优化**，实现上层应用与硬件的高性能对接"
4. "提供完整配套工具和开发套件，涵盖**驱动、容器、虚拟化**及配套开发者工具链"

Of these, **only the Triton path in layer 2 is publicly verifiable.**

---

## Open vs. Proprietary — Summary

**Open source** (Apache-2.0 via Triton/FlagTree, or the respective FlagOpen licences):
FlagTree `third_party/tsingmicro` — the entire compiler front/middle-end: Address + MagicKernel + Tx81 dialects, all conversion passes, SPM allocator, barrier insertion, software pipeliner, CRT C sources, 100+ examples/tests, a 74 KB benchmark suite, the profiler IR pass, and build scripts · TLE (`triton.experimental.tle`, `third_party/tle`) · FlagGems `_tsingmicro` backend · FlagCX `tccl_adaptor.cc` / `tsmicro_adaptor.{cc,h}` · `tsingmicro-toolchain/OnnxSlim` (167★, MIT) · `tsingmicro-toolchain/ts.knight-modelzoo` (edge only).

**Closed / binary-only:**
`tsingmicro-llvm21` LLVM backend · `libinstr_rcs1` + `oplib_rcs1` + `tx8be-oplib` · `tx81fw` / `rcs1fw-rtt` firmware · Kuiper host SDK (`/usr/local/kuiper`, `libhpgr`, `tx_runtime.h`) · kernel driver · `tsm_smi` · `torch_txda` · `txops` · **TCCL** · `profiling_tool` · C-model and `neuralcore_qemu` simulators · the RAISA graph compiler · the self-developed C/C++ kernel language.

---

## Net Assessment

The open portion is a **compiler front/middle-end plus an operation-level hardware model** — unusually revealing for a Chinese vendor, and sufficient to reconstruct the memory model, tile count, SPM size, execution model and NoC addressing without any vendor documentation.

But the maturity signals are consistently earlier-stage than Ascend or Cambricon:

- Every executable layer below LLVM IR is a closed binary tied to an internal vendor artifact server.
- The FlagOS ecosystem forks (`tsingmicro-public-e/*`) have ~0 stars and thin content; the upstream `PyTorch-Plugin-FL` has no txda backend at all.
- The backend is pinned to the older **Triton 3.3** branch while other FlagTree vendors moved to 3.6 on `main`.
- The **multi-tile TLE NoC test is disabled in CI** (`# TODO: fix`) — the distributed path that programs the chip's headline mesh is not CI-green.
- **~60 PyTorch ops still fall back to CPU** (`TXDA_FALLBACK_CPU_OPS`), including `sort`, `gather`, `index`, `pad`, `cat`, `add`, `mul`, `sum`, and `embedding_backward`.
- **No graph capture** — no torch.compile/Dynamo/FX path is public, so the RAISA layer-3 "global graph optimisation" is unverifiable.

The stack is real and actively developed (commits through Aug 2026), but the public surface is a kernel-compiler story, not an end-to-end framework story.

---

## Sources

- [FlagTree `third_party/tsingmicro` backend root](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro)
- [FlagTree upstream README — backend table](https://github.com/FlagTree/flagtree/blob/main/README.md)
- [`backend/compiler.py` — TXDABackend and the five-stage pipeline](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py)
- [`backend/driver.py` — launch, shared memory, libhpgr, sim libs](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/driver.py)
- [`backend/txda_tools.py` — Kuiper SDK path](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/txda_tools.py)
- [`Tx81Ops.td` — the Tx81 hardware dialect](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td)
- [`MagicKernelOps.td` — the MK abstract accelerator dialect](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Dialect/IR/MagicKernelOps.td)
- [`MKPipeline/Passes.td` — SPM software pipelining](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/MKPipeline/Passes.td)
- [`CoreDialectsToMK/Passes.td` — precision-mode semantics](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/CoreDialectsToMK/Passes.td)
- [TLE DSA types — the SPM docstring naming TX8](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/dsa/types.py)
- [TLE distributed — device_mesh, remote, distributed_barrier](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/distributed.py)
- [FlagTree wiki — User manual for tsingmicro](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tsingmicro)
- [FlagTree wiki — TLE design](https://github.com/flagos-ai/FlagTree/wiki/TLE)
- [`scripts/publish/README.md` — release inventory (torch_txda, txops, Docker)](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/README.md)
- [`scripts/publish/run_flaggems_on_multicards.sh` — TXDA_FALLBACK_CPU_OPS](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/run_flaggems_on_multicards.sh)
- [`scripts/build_tx8_deps.sh` — closed-component inventory](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/build_tx8_deps.sh)
- [CI workflow `tsingmicro3.3-build-and-test.yml`](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/.github/workflows/tsingmicro3.3-build-and-test.yml)
- [FlagGems `_tsingmicro/__init__.py` — vendor descriptor](https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py)
- [FlagGems `_tsingmicro/tune_configs.yaml`](https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/tune_configs.yaml)
- [FlagCX README — TCCL entry and support matrix](https://github.com/FlagOpen/FlagCX/blob/main/README.md)
- [FlagCX `device/tsmicro_adaptor.cc` — the tx* runtime surface](https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc)
- [FlagCX `ccl/tccl_adaptor.cc` — TCCL enum mappings](https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/ccl/tccl_adaptor.cc)
- [Tsingmicro TX8 series page — RAISA four-layer description](https://www.tsingmicro.com/products/tx8/series)
- [tsingmicro-public-e GitHub org](https://github.com/tsingmicro-public-e)
- [TS.Knight model zoo — the TX5 edge flow (out of scope)](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo)

---

## Update — 2026-09-13 (scan window 2026-08-08 → 2026-09-13)

*Verified directly against the GitHub REST API for `flagos-ai/FlagTree` (the repo the survey previously knew as `FlagTree/flagtree` — GitHub reports a 301 redirect from the old org name to `flagos-ai`; both URLs resolve to the same repository). No new hardware/silicon disclosure this window; one substantive upstream commit to the TX81 Triton backend.*

### TX81 backend — new TLE DSA ops and launch fast path (2026-09-01)

Commit `0abc361e` ("[BACKEND] Add TLE DSA elementwise/randgen/bitcast ops and tx81 launch fast path", PR #1073), landed on the `triton_v3.3.x` branch **2026-09-01** — squarely in-window and the newest commit touching `third_party/tsingmicro` as of 2026-09-13. 27 files changed:

- **New TLE-DSA ops**: elementwise binary (add/sub/mul/maximum/minimum/div), `to_tensor`/`to_buffer` tensor↔buffer bridging, `dsa.randgen` (random-number generation on TX81 peripheral), `dsa.bitcast`, and strided `extract_slice`/`insert_slice`.
- **Tsingmicro backend lowering** added for the new ops across `TLEToMK`, `MKToTx81`, `Tx81ToLLVM`, `LinalgToMK`, and a new `MaterializeStridedLinalgInputs` pass; CRT support added for `randgen`.
- **Driver launch fast path**: the kernel module is now preloaded once and launched via a `txLaunchKernel` handle (with a `TXDA_LAUNCH_VIA_GGL` fallback to `txLaunchKernelGGL`), redundant `txSetDevice` calls on the hot path are skipped, and a `Py_DECREF`/refcount bug plus a stray `fflush` are fixed.

This is routine, credible upstream development — consistent with the vendor's "Triton-first, upstream-first" posture recorded at baseline — and does not change any hardware fact (no new TX8x part, no spec value, no order/corporate update).

### Corroborating repo status

- `flagos-ai/FlagTree` overall `pushed_at`: **2026-09-14** (i.e., essentially current as of this pass) — the org remains actively developed, though the most recent tsingmicro-path-specific commit is the 2026-09-01 one above.

### Searched and absent

- No new Tsingmicro SDK/toolkit release (private SDK still not shipped; upstream FlagTree/FlagGems/FlagCX remain the only public surface).
- No ChiNext IPO status change found beyond the 2026-06-16 tutoring (辅导验收) stage already recorded — no filing/acceptance (受理) news located. English- and Chinese-language news search (Bing News) returned no indexed results for "清微智能" in this window; Baidu search was blocked by a CAPTCHA challenge on every attempt. **Treat the IPO status as unchanged, not as confirmed unchanged** — search coverage for this vendor is thin.
- No new order-volume figure beyond the previously recorded "30000+ cumulative" (2026 vendor site) figure.
- No Hot Chips 38 (2026-08-23 → 08-25) Tsingmicro talk.

### Sources added 2026-09-13

- [GitHub — flagos-ai/FlagTree commit 0abc361e (2026-09-01), PR #1073](https://github.com/flagos-ai/FlagTree/commit/0abc361ee426e4a8c9d39d8af90c37120bbd8f4a)
- [GitHub API — flagos-ai/FlagTree repo metadata (`pushed_at`, live query 2026-09-13)](https://api.github.com/repos/flagos-ai/FlagTree)
- [GitHub API — commits on `triton_v3.3.x` touching `third_party/tsingmicro` (live query 2026-09-13)](https://api.github.com/repos/flagos-ai/FlagTree/commits?sha=triton_v3.3.x&path=third_party/tsingmicro)
