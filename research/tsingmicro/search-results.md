# Tsingmicro (清微智能) TX81 — Search Results

*as_of: 2026-08-08*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*

---

## Scope

This entry covers the **TX8 series / TX81** cloud-datacenter line only (TX81 RPU module, REX1032 server, REX81 supernode).

The **TX5 series** (TX510 vision, TX210 voice, edge/embedded NPUs served by the **TS.Knight → RNE** toolchain) is a separate edge product line and is **out of scope**.

Two attribution traps recorded here so downstream files do not repeat them:

1. **`third_party/rpu` on the FlagTree `main` branch is the Rhino RPU from Huixi Intelligence (辉羲智能)** — a different company that also uses the term "RPU". Tsingmicro's backend is **`third_party/tsingmicro` on the `triton_v3.3.x` branch only**.
2. **`算力突破每秒500千万亿次` = >500 PFLOPS, not exaflops** (千万亿 = 10¹⁵). Several machine-translated English write-ups render this as exaflops.

---

## Search Queries

1. 清微智能 TX81 可重构 架构 算力
2. 清微智能 TX81 芯片 制程 显存 带宽 功耗
3. "REX1032" 清微智能 服务器 参数
4. 清微智能 REX1032 服务器 参数 显存 算力
5. 清微智能 TX81 中国联通 中标 算力卡
6. 清微智能 可重构2.0 架构 尹首一 清华
7. 清微智能 TX81 量产 发布 2024
8. Tsingmicro TX81 RPU reconfigurable dataflow datacenter chip
9. Direct source retrieval: vendor site (`tsingmicro.com` + raw HTML dump), eefocus, sina, GitHub tree/raw API over `FlagTree/flagtree@triton_v3.3.x`, `FlagOpen/FlagCX`, `tsingmicro-public-e/FlagGems`

> Note: DuckDuckGo (html + lite) and Sogou blocked the automated queries; Bing (zh-CN and en-US) plus direct fetches were the working retrieval paths.

---

## Resources Found

### Official Vendor Pages

| Resource | URL | Category |
|----------|-----|----------|
| Tsingmicro corporate homepage (TX81 module, REX1032, REX81, RAISA, "30000+ card orders") | https://www.tsingmicro.com/ | Vendor site |
| TX8 series product page — 可重构2.0, switchless Mesh/Torus, REX81 = 4096 chips >500 PFLOPS, RAISA 4-layer model, next-gen 3-D DRAM roadmap | https://www.tsingmicro.com/products/tx8/series | Hardware Spec (primary) |
| Tsingmicro "About" — founded 2018, reconfigurable dataflow, ">50% energy reduction" claim | https://www.tsingmicro.com/about | Overview |
| Tsingmicro edge product page (TX5 line — out-of-scope marker) | https://www.tsingmicro.com/deviceEdge | Vendor site |
| robots.txt (no sitemap; SPA, no downloadable spec sheets) | https://www.tsingmicro.com/robots.txt | Vendor site |

> **No developer portal exists.** `developer.tsingmicro.com`, `doc.tsingmicro.com`, `docs.tsingmicro.com` and `open.tsingmicro.com` all fail to resolve. All usable public documentation lives in third-party FlagOS repositories.

### Open-Source — FlagTree Triton Backend (the primary technical source)

| Resource | URL | Category |
|----------|-----|----------|
| FlagTree upstream README — backend table: "Tsingmicro / tsingmicro / triton_v3.3.x / Triton 3.3 / 2025-06-06" | https://github.com/FlagTree/flagtree/blob/main/README.md | Compiler |
| `third_party/tsingmicro` backend root (609 files) | https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro | Compiler |
| `backend/compiler.py` — `TXDABackend`, 5-stage pipeline, RISC-V link line, XuanTie SDK path | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py | Compiler |
| `backend/driver.py` (42 KB) — `txLaunchKernelGGL`, `max_shared_mem`, `warp_size=16`, `libhpgr` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/driver.py | Runtime |
| `backend/txda_tools.py` — `get_kuiper_path("/usr/local/kuiper")` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/txda_tools.py | Runtime |
| `Tx81Ops.td` (47 KB, ~150 ops) — rdma/wdma, conv, gemm, remote_load/store/buffer (chip_x, chip_y, die, tile), distribute_barrier, dtype conversions | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td | Hardware Spec (key) |
| `Tx81Types.td` — FP8 E4M3/E5M2 (+UZ), F16, BF16, F32, F64, I1/I4/I8/I16/I32/I64 | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Types.td | Hardware Spec |
| `Transforms/Passes.td` — "DDR↔SPM data movement is exclusively through RDMA/WDMA … hardware handles ordering" | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Transforms/Passes.td | Hardware Spec (key) |
| `MagicKernelOps.td` (30 KB) — MK abstract accelerator op set | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Dialect/IR/MagicKernelOps.td | Compiler |
| `MKPipeline/Passes.td` — SPM software pipelining, `num_stages` default/max 2 | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/MKPipeline/Passes.td | Compiler |
| `CoreDialectsToMK/Passes.td` — `precision-mode` 0/1/2 semantics | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/magic-kernel/Conversion/CoreDialectsToMK/Passes.td | Compiler |
| `crt/include/Tx81/tx81_def.h` — `MemorySpace {SPM, DDR}`, `ActFuncMode` "Neural engine activate mode" | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/include/Tx81/tx81_def.h | Hardware Spec (key) |
| `crt/include/Tx81/tx81_run.h` — `RcsWaitfinish`, DMA bounds checking, SPM mapping | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/include/Tx81/tx81_run.h | Runtime |
| `crt/lib/Tx81/gemm.c` — `RcsNeInstr {I_NEUR}`, GEMM builder API, `RcsExecute` "Dispatch the command to accelerator" | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/gemm.c | Hardware Spec (key) |
| `crt/lib/Tx81/send.c` — `__Send`, DTE + FSM monitors, `get_tile_spm_addr_base(tile,4,4)`, `initTileId(pid, rowLength=4)` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/send.c | Hardware Spec (key) |
| `crt/lib/Tx81/tx81.c` — `spmMappingOffset = 0x30400000`, INT64 splitting | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/tx81.c | Hardware Spec |
| `crt/lib/Tx81/rdma.c` — `RcsRdmaInstr {I_RDMA}` async DMA | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/rdma.c | Hardware Spec |
| `examples/tle/test_tle_dsa_noc_gemm_4096.py` — `TILE_NUM=16`, 4×4 Hamiltonian ring, `tle.remote` + `distributed_barrier` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_4096.py | Hardware Spec (key) |
| `examples/tle/test_tle_dsa_noc_gemm_benchmark.py` (26 KB) — multi-ring autotuned NoC GEMM | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_benchmark.py | Benchmark |
| `examples/qwen_demo.py` — Qwen2.5-0.5B / Qwen3-0.6B via FlagGems | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/qwen_demo.py | Framework |
| `scripts/build_tx8_deps.sh` — `tx81fw`, `rcs1fw-rtt`, XuanTie SDK, `tx8-yoc-rt-thread-smp`, `profiling_tool` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/build_tx8_deps.sh | Build (key) |
| `scripts/publish/README.md` — release inventory: `torch_txda` wheel, `txops` wheel, vendor Docker tag | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/README.md | Distribution (key) |
| `scripts/publish/run_flaggems_on_multicards.sh` — `TXDA_SKIP_OPS`, `TXDA_FALLBACK_CPU_OPS` (~60 ops) | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/scripts/publish/run_flaggems_on_multicards.sh | Runtime |
| `third_party/tsingmicro/CMakeLists.txt` — `instr_rcs1`, `oplib_rcs1`, XuanTie V2.8.0 | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/CMakeLists.txt | Build |
| `profiler/profiler.cpp` — LLVM IR trace-point injection | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/profiler/profiler.cpp | Profiling |
| CI workflow `tsingmicro3.3-build-and-test.yml` — self-hosted HW runner, 14 tests, TLE NoC test commented out "TODO: fix" | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/.github/workflows/tsingmicro3.3-build-and-test.yml | CI (key) |

### Open-Source — TLE (Triton Language Extensions)

| Resource | URL | Category |
|----------|-----|----------|
| TLE DSA core — `alloc` / `copy` / `local_ptr`, `spm` scope | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/dsa/core.py | Kernel Language |
| TLE DSA types — docstring: "Scratch Pad Memory – the on-chip SRAM exposed by TsingMicro TX8 … same conceptual role as NVIDIA shared memory" | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/dsa/types.py | Kernel Language (key) |
| TLE distributed — `device_mesh`, `sharding`, `shard_id`, `remote`, `distributed_barrier`, `distributed_dot` | https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/python/triton/experimental/tle/language/distributed.py | Kernel Language (key) |

### Open-Source — Operator and Collective Libraries

| Resource | URL | Category |
|----------|-----|----------|
| FlagGems `_tsingmicro/__init__.py` — `name="TX81"`, `multi_processor_count=16`, 64 GB `total_memory`, `SPM_SIZE=3 MiB`, `tsm_smi`, PrivateUse1, fp64/int64 disabled | https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py | Hardware Spec (key) |
| FlagGems `_tsingmicro/heuristics_config_utils.py` — tile heuristics using `multi_processor_count` as NUM_SMS | https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/heuristics_config_utils.py | Op Library |
| FlagGems `_tsingmicro/tune_configs.yaml` (1313 lines, ~37 op families) | https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/tune_configs.yaml | Op Library |
| FlagCX README — "TCCL, TsingMicro Communication Collectives Library" + homo/hetero support matrix | https://github.com/FlagOpen/FlagCX/blob/main/README.md | Communication (key) |
| FlagCX `tsmicro_adaptor.h` — `tcclComm_t`, `txStream_t`, `txEvent_t`, `txIpcMemHandle_t`, `TX_SUCCESS` | https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/include/tsmicro_adaptor.h | Communication (key) |
| FlagCX `tccl_adaptor.cc` — TCCL result/dtype/redop enum mappings | https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/ccl/tccl_adaptor.cc | Communication |
| FlagCX `device/tsmicro_adaptor.cc` — full `tx*` runtime API surface, PCIe BDF, DMA-buf, vendor string "TSMICRO" | https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc | Runtime (key) |

### Documentation

| Resource | URL | Category |
|----------|-----|----------|
| FlagTree wiki — User manual for tsingmicro (bundle names, env vars, install) | https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tsingmicro | SDK Docs |
| FlagTree wiki — TLE three-tier design (Lite / Struct / Raw) | https://github.com/flagos-ai/FlagTree/wiki/TLE | Compiler Docs |
| FlagOS wheel index hosting `flagtree==0.6.0+tsingmicro3.3` | https://resource.flagos.net/repository/flagos-pypi-hosted/simple | Distribution |

### Vendor GitHub Organizations

| Resource | URL | Category |
|----------|-----|----------|
| `tsingmicro-public-e` — FlagOS ecosystem forks (FlagGems, FlagScale, Megatron-LM-FL, TransformerEngine-FL, PyTorch-Plugin-FL, vllm-plugin-FL, sglang-plugin-FL), active to Aug 2026 | https://github.com/tsingmicro-public-e | Vendor GitHub |
| `tsingmicro-toolchain` — OnnxSlim (167★, MIT), ts.knight-modelzoo (13★) | https://github.com/tsingmicro-toolchain | Vendor GitHub |
| TS.Knight model zoo — **edge/RNE flow, no TX8 support** (out of scope) | https://github.com/tsingmicro-toolchain/ts.knight-modelzoo | Edge (out of scope) |

### Press / Trade Press

| Resource | URL | Category |
|----------|-----|----------|
| 与非网 (eefocus) — 512 TFLOPS FP16 per RPU module, 4 PFLOPS/node, 2 TB (max 4 TB), China Unicom Inner Mongolia tender ~¥47M, RPU definition | https://www.eefocus.com/article/1888048.html | Trade press (key) |
| 新浪 — Bund Conference Sept 2025: 512 TFLOPS module, 4 PFLOPS REX1032, DeepSeek R1 single-node, 20000+ cards, ¥47.9M Unicom projects | https://news.sina.com.cn/sx/2025-09-15/detail-infqpxhk9790648.shtml | Press (key) |
| 新浪财经 — ChiNext IPO tutoring / 辅导验收 status 2026-06-16, Huatai United Securities | https://finance.sina.com.cn/roll/2026-06-16/doc-inicqnru6292806.shtml | Press |
| 腾讯新闻 — corroborating IPO tutoring-acceptance report, June 2026 | https://news.qq.com/rain/a/20260616A07PQG00 | Press |
| 腾讯新闻 — March 2026 IPO / Series C >¥2B coverage | https://news.qq.com/rain/a/20260304A07FP0P00 | Press |
| 百度百科 — 北京清微智能科技股份有限公司 (founded 2018-07-26, Tsinghua team) | https://baike.baidu.com/item/北京清微智能科技股份有限公司 | Reference (403 to fetch; via search snippets) |

### Academic / Conference

**None.** No ISSCC, ISCA, Hot Chips, MICRO, or JSSC paper exists for TX8 or TX81 as of 2026-08-08, and no architecture whitepaper is published. This is the single largest evidence gap for this entry.

---

## Key Findings

- **Device class**: self-branded **RPU (Reconfigurable Processing Unit)** — a coarse-grained reconfigurable dataflow accelerator on a "可重构 2.0" architecture. The vendor's own definition is runtime dynamic reconfiguration of "计算单元、互连结构、数据通路" (compute units, interconnect structure, datapath).
- **The microarchitecture is reconstructed from vendor-authored open source, not from vendor documentation.** The 47 KB `Tx81Ops.td` MLIR dialect is effectively a published operation-level model of the accelerator: 16 tiles in a 4×4 mesh, 3 MiB SPM per tile, RISC-V (RV64IMFDC, T-Head XuanTie 900) control core per tile, command-descriptor dispatch (`I_NEUR`, `I_RDMA`) to fixed-function/reconfigurable engines, and a four-level `(chip_x, chip_y, die, tile)` remote-load/store address hierarchy.
- **Memory model**: two address spaces only — `SPM` and `DDR`. All compute operands live in SPM; DDR↔SPM movement is exclusively explicit RDMA/WDMA. **No hardware data cache in the compute path** (the `L2_cache_size = 3 MB` in FlagGems is `SPM_SIZE` re-exported as a Triton API shim).
- **Published performance is system-level only**: 512 TFLOPS FP16 per **RPU module**, 4 PFLOPS per **REX1032 node**, >500 PFLOPS for the **REX81 supernode** (4,096 TX81 chips). **Per-chip TFLOPS is not disclosed.**
- **Nothing physical is disclosed**: process node, foundry, die size, dies per package, memory technology and bandwidth, TDP, clock, PCIe generation, card form factor, and chip-to-chip link bandwidth are all undisclosed.
- **Software = RAISA (brand) but Triton (reality)**. The vendor brands the stack RAISA and claims CUDA-ecosystem compatibility; nothing public substantiates CUDA source or binary compatibility. The verifiable path is Triton 3.3 via FlagTree, FlagGems for operators, FlagCX/TCCL for collectives. The "compatibility" substance is a **HIP-shaped runtime API** (`txLaunchKernelGGL`, `txMalloc`, `txStreamCreate`) plus an **NCCL-shaped collectives API** (TCCL).
- **Everything below LLVM IR is a closed binary**: `tsingmicro-llvm21`, `libinstr_rcs1`, `oplib_rcs1`, `tx81fw`/`rcs1fw-rtt` firmware, the Kuiper host SDK (`/usr/local/kuiper`, `libhpgr`, `tx_runtime.h`), the kernel driver, `tsm_smi`, `torch_txda`, `txops`, TCCL, the profiler, and the RAISA graph compiler.
- **Maturity**: shipping and deployed at modest scale. The strongest independent evidence is the **August 2025 China Unicom Inner Mongolia tender** (~RMB 32M cards + >RMB 15M servers, ~RMB 47M total). Vendor claims of 30,000+ cumulative card orders, 200+ adapted models, and "first-tier domestic cloud chip shipments in 1H2025" are **unverified**. The **REX81 supernode is announced only**.
- **Corporate**: not publicly listed. Series C > RMB 2B (Dec 2025, led by Beijing Energy Group); ChiNext IPO **tutoring / 辅导验收 stage as of 2026-06-16** — pre-filing, no prospectus, no audited revenue public.
