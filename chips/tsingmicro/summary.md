# Tsingmicro (清微智能) TX81 — Summary

*as_of: 2026-09-13*
*chip: tsingmicro*
*device_class: Reconfigurable Dataflow / CGRA "RPU" (China, 清微智能)*

**Manufacturer:** Beijing Tsingmicro Intelligence Technology (北京清微智能科技股份有限公司), founded 2018-07-26; fabless, foundry **not disclosed**
**Products in scope:** TX81 RPU compute module · REX1032 server node · REX81 supernode
**Deployment:** Chinese provincial intelligent-computing centres and China Unicom private clouds; shipping at modest scale
**Research date:** 2026-08-08

---

## What It Is

Tsingmicro is a Chinese fabless AI chip company with a Tsinghua reconfigurable-computing lineage. It sells two architecturally unrelated lines: **TX5** (edge/embedded NPUs, TS.Knight/RNE toolchain — *out of scope here*) and **TX8**, the cloud/datacenter line whose part is the **TX81**.

The TX81 is marketed as an **RPU — Reconfigurable Processing Unit** — built on a self-developed "**可重构 2.0**" (Reconfigurable 2.0) architecture, which the vendor defines as runtime dynamic reconfiguration of compute units, interconnect structure and datapath. In practice, what is publicly observable is a **coarse-grained reconfigurable dataflow accelerator**: 16 tiles in a 4×4 mesh, each tile a RISC-V control core driving fixed-function/reconfigurable engines through command descriptors, with a fully software-managed 3 MiB scratchpad per tile and **no hardware data cache anywhere in the compute path**.

### Read this before citing any number from this entry

**There is no ISSCC / ISCA / Hot Chips / MICRO / JSSC paper and no architecture whitepaper for TX8 or TX81, and the vendor operates no developer portal.** Nearly every microarchitectural fact in this entry is reconstructed from **vendor-authored open source** — chiefly the Tsingmicro Triton backend upstreamed into BAAI's FlagTree (`third_party/tsingmicro`, `triton_v3.3.x`, 609 files), whose 47 KB `Tx81Ops.td` MLIR dialect is effectively a published operation-level model of the accelerator.

Everything physical — **process node, foundry, die size, dies per package, memory technology, memory bandwidth, TDP, clock, PCIe generation, form factor, chip-to-chip link bandwidth** — is **not disclosed**, and is recorded as such rather than estimated.

### Three traps recorded so downstream work does not repeat them

1. **Unit error.** "算力突破每秒500千万亿次" = **>500 PFLOPS**, not exaflops (千万亿 = 10¹⁵). Machine-translated English coverage frequently gets this wrong.
2. **Attribution.** FlagTree's `third_party/rpu` on the `main` branch is the **Rhino RPU from Huixi Intelligence (辉羲智能)** — a different company that also uses "RPU". Tsingmicro's backend is `third_party/tsingmicro` on `triton_v3.3.x` only.
3. **Product line.** TS.Knight / RNE is the **TX5 edge** flow with no TX8 support. Do not conflate.

---

## Key Architecture Features

| Feature | TX81 |
|---|---|
| Architecture class | Coarse-grained reconfigurable dataflow ("RPU", 可重构 2.0) |
| Tiles per device | **16**, arranged as a **4 × 4 2-D mesh** |
| Per-tile control | RISC-V **RV64IMFDC** (lp64d), T-Head **XuanTie 900**; **no RVV** — all vector work goes to engines |
| Per-tile engines | Neural engine (GEMM/conv, `I_NEUR`), RDMA/WDMA (`I_RDMA`), DTE (tile↔tile), vector/transcendental, layout (img2col, transpose, NCHW↔NHWC, gather/scatter, LUT), reduction/sort |
| Reconfiguration granularity (observable) | One command descriptor per whole tiled tensor op (M/K/N or pads/strides/dilations + fused bias, per-axis scale, ReLU/LeakyReLU, psum) |
| Backward conv in hardware | **Yes** — `ConvOp.op_type = 2` |
| Structured sparsity | **Present in hardware** (`en_sparse` / `src_sparse`); pattern, ratio and speedup **not disclosed** |
| Native compute formats | bf16 / fp16 / tf32 / fp32 (explicit dialect header statement); INT8 via `src_fmt`/`dst_fmt`; FP8 E4M3/E5M2 and **MX/FP4 block-scaled** in the conversion path |
| No FP64, no native INT64 | `fp64_enabled=False`, `int64_enabled=False`; INT64 split into two INT32 lanes |
| **SPM per tile** | **3 MiB** (3,014,656 B visible to a Triton kernel after 2 × 64 KiB reservation) |
| **Aggregate on-chip SPM** | **48 MiB** (derived: 16 × 3 MiB) |
| Hardware data cache | **None.** Two address spaces only: `SPM` and `DDR`; DDR↔SPM movement is exclusively explicit RDMA/WDMA |
| Off-chip memory per device | **64 GB** (from the vendor-maintained FlagGems descriptor); technology and bandwidth **not disclosed** |
| On-chip interconnect | 2-D mesh NoC; DTE async block transfer **plus globally addressable peer SPM** (a distributed shared address space, not message-passing only) |
| Scale-up | Proprietary **switchless** chip-to-chip mesh/torus; compiler sees a four-level `(chip_x, chip_y, die, tile)` remote load/store address hierarchy; all bandwidth/latency figures **not disclosed** |
| Host interface | **PCIe** (confirmed via BDF handling); generation and lane width **not disclosed** |
| Collectives | **TCCL** (closed), NCCL-shaped; FlagCX lists all 12 collective ops in both homogeneous and heterogeneous modes |

### Performance — system-level only

| Level | Figure | Status |
|---|---|---|
| **TX81 RPU module** | **512 TFLOPS FP16** | vendor claim |
| **REX1032 node** | **4 PFLOPS**, 2 TB memory (max 4 TB) | vendor claim |
| **REX81 supernode** | **4,096 TX81 chips, >500 PFLOPS** | vendor product page; **announced only** |
| **Per TX81 chip** | **not disclosed** | — |
| INT8 / FP8 / FP4 / TF32 / sparse peak | **not disclosed** | — |

> **Derived, not sourced.** 4 PFLOPS ÷ 512 TFLOPS = 8 modules/node; 2 TB ÷ 64 GB = 32 chips/node → 4 chips/module → **~128 TFLOPS FP16 per chip**; 4096 × 128 TFLOPS = 524 PFLOPS, matching ">500 PFLOPS"; 4096 ÷ 32 = 128 nodes per supernode. Every step is self-consistent, but **this is arithmetic and must never be cited as a published spec.** It also assumes the 64 GB figure is the shipping configuration, which no datasheet confirms.

---

## Software Stack

The vendor brands the stack **RAISA**. RAISA itself is undocumented publicly; the verifiable path is Triton.

```
PyTorch (device "txda", PrivateUse1 via closed torch_txda wheel)
  → FlagGems _tsingmicro backend (open) — operator library
  → [RAISA graph-optimisation layer — closed, undocumented; NO public graph capture]
  → Triton 3.3 + TLE (tle.dsa for SPM, tle.device_mesh/remote/distributed_barrier for the mesh)
  → FlagTree tsingmicro backend (open, 609 files):
        ttir → coreir → txir → llir → so
        dialects: Address, triton-shared, MagicKernel, Tx81, DSA
  → tsingmicro-llvm21 (closed vendor LLVM fork, codename "ZTC")
  → RISC-V codegen: clang++ --target=riscv64-unknown-elf -march=rv64imfdc
                    → kernel.so linking -linstr_rcs1 (closed)
  → Kuiper host SDK / tx_runtime.h / libhpgr (closed, HIP-shaped: txLaunchKernelGGL, txMalloc, …)
  → tx81fw / rcs1fw-rtt firmware on RT-Thread SMP (closed)
  → kernel driver (closed) · tsm_smi management CLI
  → TX81 hardware
Collectives: TCCL (closed, NCCL-shaped) via open FlagCX adaptors
```

**What is genuinely open** is the compiler front/middle-end plus an operation-level hardware model: the Tx81 and MagicKernel MLIR dialects, all conversion passes, the SPM liveness allocator, dependency-driven barrier insertion, the software pipeliner, ~130 CRT C files, 100+ examples and the profiler IR pass. Co-authored with **Terapines Technology (Wuhan) / 兆松科技**.

**What is closed** is every executable layer below LLVM IR: the LLVM backend, `libinstr_rcs1`, `oplib_rcs1`, firmware, the Kuiper host SDK, the driver, `torch_txda`, `txops`, TCCL, the profiler tool, the simulators, and the RAISA graph compiler.

**The CUDA-compatibility claim is marketing.** The vendor advertises "CUDA 生态兼容 / 零代码迁移"; nothing public demonstrates CUDA source or binary compatibility. The real substance is **API-shape compatibility** — a HIP-shaped runtime (`txLaunchKernelGGL` ↔ `hipLaunchKernelGGL`) plus an NCCL-shaped TCCL — with Triton as the portable kernel language.

### Maturity signals in the software

- Backend pinned to **Triton 3.3** while other FlagTree vendors moved to 3.6 on `main`.
- The **multi-tile TLE NoC ring-GEMM test is commented out in CI** with `# TODO: fix` — the distributed path that programs the chip's headline mesh is not CI-green.
- **~60 PyTorch ops still fall back to CPU** (`TXDA_FALLBACK_CPU_OPS`: `sort`, `gather`, `index`, `pad`, `cat`, `add`, `mul`, `sum`, `embedding_backward`, …).
- **No public graph capture** — no torch.compile/Dynamo/FX path exists, so RAISA layer-3 "global graph optimisation" is unverifiable.
- FlagOS ecosystem forks (`tsingmicro-public-e/*`) have ~0 stars; upstream `PyTorch-Plugin-FL` has no txda backend at all.

**Update (2026-09-13).** The FlagTree TX81 backend continues active upstream development: commit `0abc361e` (2026-09-01, PR #1073) added new **TLE-DSA elementwise/randgen/bitcast ops** and a **driver launch fast path** (preloaded kernel module + `txLaunchKernel` handle, skipping redundant `txSetDevice`). No hardware/spec change; no ChiNext IPO status change found beyond the previously recorded 2026-06-16 tutoring stage. See `research/tsingmicro/investigations/software-stack.md` → "Update — 2026-09-13".

---

## Distinguishing Design Choices

1. **Reconfigurable dataflow as the product identity, but coarse-grained in practice.** The only public evidence of reconfiguration is a per-operator command-descriptor interface. Configuration-memory size, context switching and partial reconfiguration are undescribed anywhere.
2. **Globally addressable peer scratchpad.** `get_tile_spm_addr_base(tile, x, y)` lets a tile read and write another tile's SPM directly. Combined with `remote_load`/`remote_store` carrying `(chip_x, chip_y, die, tile)`, TX81 presents a **distributed shared address space across a chip mesh**, not merely message-passing DMA. This is the architecturally most interesting property of the design.
3. **Switchless scale-up as the commercial pitch.** "千卡直接互联，无需交换机成本" — thousand-card direct interconnect with no switch cost, Mesh/Torus topologies. Every quantitative property of that fabric is undisclosed, which is a notable gap for the chip's headline claim.
4. **RISC-V scalar control without RVV.** The tile CPU is an off-the-shelf T-Head XuanTie 900 running RT-Thread SMP; all vector and matrix work is offloaded to engines rather than to CPU vector instructions.
5. **Two address spaces, no cache.** Puts TX81 firmly in the compiler-scheduled scratchpad class alongside TPU, Groq and Sophgo, and makes the SPM allocator and software pipeliner performance-critical (though the pipeliner is clamped to double buffering only).
6. **Triton-first, upstream-first software.** Rather than shipping a private SDK, Tsingmicro upstreamed a complete backend into BAAI FlagOS. That is unusual transparency for a Chinese vendor — and it is the only reason this entry can describe the microarchitecture at all.

---

## Scale and Maturity

| Item | Status |
|---|---|
| TX81 / REX1032 | **Shipping, deployed at modest scale.** Strongest independent evidence: **Aug-2025 China Unicom Inner Mongolia tenders**, ~RMB 32M compute cards + >RMB 15M servers ≈ **RMB 47M (~US$6.5M)**. Corroborated by a Sept-2025 vendor statement of RMB 47.9M across two China Unicom / 中贝通信 projects |
| REX1032 demonstrated workload | DeepSeek-R1 671B full-precision on a **single node**; stable 128K context; vendor claims >4× comparable concurrency at 128K |
| **REX81 supernode** | **Announced only** (Bund Conference, Sept 2025). No evidence of a deployed instance |
| Deployment footprint | Thousand-card centres in 东北, 浙江, 北京, 安徽 — *vendor claim, unverified* |
| Order volume | "20000+ cards" (Sept 2025) → "30000+ cumulative" (2026 site) — *vendor claim, escalating, unverified* |
| Model coverage | "200+ models/applications adapted" — *vendor claim, unverified* |
| Market position | "first tier of domestic cloud-chip shipments, 1H2025" — *vendor claim, unverified* |
| Energy claim | ">50% lower energy at equal compute" — *vendor claim, no measured basis published* |
| Corporate | **Not publicly listed.** Series C > RMB 2B (Dec 2025, led by Beijing Energy Group). ChiNext IPO **tutoring / 辅导验收 as of 2026-06-16** with Huatai United Securities — pre-filing, no prospectus, no audited revenue public |
| Tape-out / sampling / volume-production dates | **not disclosed** |

---

## Competitive Position

TX81 sits in the second tier of Chinese datacenter accelerators — behind Huawei Ascend and Cambricon on both silicon disclosure and software maturity, but with a genuinely differentiated architectural story.

- **Versus Ascend / Cambricon**: far less mature software (no graph capture, ~60 CPU-fallback ops, a non-CI-green distributed path, pinned to an older Triton), and no disclosed silicon specifications at all. But the *open* portion of the stack is more revealing than either vendor's.
- **Versus Sophgo**: TX81 is a genuine training-and-inference datacenter part with a proprietary scale-up fabric and hardware backward convolution; Sophgo's BM1684X is inference-only with no scale-up fabric. TX81 is a class above in ambition, though Sophgo's TPU-MLIR is a more complete open compiler.
- **Versus SambaNova / Groq (the dataflow peers)**: TX81 shares the compiler-scheduled scratchpad model, but the reconfiguration granularity that would justify the CGRA framing is undisclosed, and there is no published performance characterisation of any kind.
- **Structural weakness**: for a chip whose headline claim is switchless thousand-card scale-up, **not a single number about that fabric is public** — no link bandwidth, SerDes rate, radix, hop latency, or medium. The REX81 supernode that would demonstrate it is announced only.

---

## Sources

- [Tsingmicro TX8 series product page](https://www.tsingmicro.com/products/tx8/series)
- [Tsingmicro corporate homepage](https://www.tsingmicro.com/)
- [Tsingmicro About page](https://www.tsingmicro.com/about)
- [FlagTree `third_party/tsingmicro` backend (609 files)](https://github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro)
- [`Tx81Ops.td` — the Tx81 hardware dialect (~150 ops)](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Dialect/IR/Tx81Ops.td)
- [`Transforms/Passes.td` — the SPM/DDR memory-model statement](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/include/tsingmicro-tx81/Transforms/Passes.td)
- [`crt/lib/Tx81/gemm.c` — I_NEUR command descriptor and RcsExecute](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/gemm.c)
- [`crt/lib/Tx81/send.c` — DTE, FSM monitors, 4×4 tile addressing](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/crt/lib/Tx81/send.c)
- [`examples/tle/test_tle_dsa_noc_gemm_4096.py` — 16-tile ring GEMM](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/examples/tle/test_tle_dsa_noc_gemm_4096.py)
- [`backend/compiler.py` — five-stage pipeline and RISC-V toolchain](https://github.com/FlagTree/flagtree/blob/triton_v3.3.x/third_party/tsingmicro/backend/compiler.py)
- [FlagGems `_tsingmicro/__init__.py` — TX81 device descriptor](https://github.com/tsingmicro-public-e/FlagGems/blob/master/src/flag_gems/runtime/backend/_tsingmicro/__init__.py)
- [FlagCX README — TCCL support matrix](https://github.com/FlagOpen/FlagCX/blob/main/README.md)
- [FlagCX `device/tsmicro_adaptor.cc` — the tx* runtime surface](https://github.com/FlagOpen/FlagCX/blob/main/flagcx/adaptor/device/tsmicro_adaptor.cc)
- [FlagTree wiki — User manual for tsingmicro](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-tsingmicro)
- [与非网 — 512 TFLOPS module, 4 PFLOPS node, China Unicom tender](https://www.eefocus.com/article/1888048.html)
- [新浪 — Bund Conference Sept 2025 announcement](https://news.sina.com.cn/sx/2025-09-15/detail-infqpxhk9790648.shtml)
- [新浪财经 — ChiNext IPO tutoring status, 2026-06-16](https://finance.sina.com.cn/roll/2026-06-16/doc-inicqnru6292806.shtml)
- [腾讯新闻 — corroborating IPO tutoring report, June 2026](https://news.qq.com/rain/a/20260616A07PQG00)
