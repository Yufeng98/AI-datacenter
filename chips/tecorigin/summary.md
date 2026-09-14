# Tecorigin (太初元碁) SDAA Software and Hardware Stack Summary

*as_of: 2026-09-13*
*chip: tecorigin*
*device_class: Heterogeneous Many-Core Accelerator (SPA/SPE array with software-managed SPM scratchpad; China, 太初元碁)*

---

## Overview

**Tecorigin** (brand 太初元碁; legal entity 太初（杭州）集成电路有限公司, founded November 2019, HQ Hangzhou, R&D centers in Wuxi / Beijing / Shanghai, systems-integration center in Yancheng) builds the **SDAA** (Software Defined Accelerator Architecture) datacenter AI accelerator line. Per the company's own About page, the core team comes from Tsinghua University and the **National Supercomputing Center in Wuxi** — home of Sunway TaihuLight — and has won the **Gordon Bell Prize three times**.

The accelerator is **neither a GPU nor a systolic-array TPU**. It is a *heterogeneous many-core* machine in the Sunway/Cell idiom, wrapped in a deliberately CUDA-shaped software surface:

- The card is a **host/device machine** — "主从异构的物理架构". **The host CPU is the master**; no on-device master core is documented.
- One card holds **4 SPAs** (Synergistic Processor Element **Arrays**). Each SPA owns its own Global memory and is exposed to software as an **independent device** — "每个SPA类似于GPU的一张卡". There is **no shared address space across SPAs**.
- Each SPA holds **32 SPEs** (Synergistic Processor Elements; called 从核, "slave cores", in Tecorigin's own kernel comments), each an independent core with a scalar unit, a vector unit, a matrix unit, and a **private software-managed SPM scratchpad**.
- **⇒ 128 SPEs per card.**
- There is **no hardware data cache in the compute path**. The vendor states a strict three-level model — registers / SPM / Global — where every transfer is an explicit **DMA** (Global↔SPM), **RMA** (SPE↔SPE scratchpad, direct), or **hardware broadcast**. RMA plus row/column broadcast between slave cores is the Sunway `athread` idiom and is the clearest architectural fingerprint of the lineage.

**Products.** The chip is **T1**; T100 / T110 / T111 are card SKUs — 元碁 T100 (PCIe), T110 (air-cooled OAM), T111 (liquid-cooled OAM). Systems: AI workstation (1–4 cards), T1008 (4U/8-card), I1004 (2U/4-card inference), T1108 (6U/8-card OAM), T1118 (2U/8-card liquid-cooled), and 元碁 **SuperPOD-128** (128 cards per rack). This is unambiguously datacenter-class — no edge SKU, with OAM modules, liquid cooling, IB/RoCE and rack-scale systems.

> **The defining evidentiary fact about this vendor: Tecorigin publishes no datasheet.** There is **no peak TFLOPS/TOPS at any precision, no HBM bandwidth, no process node, no die size, and no TDP** anywhere in ~3.2 MB of vendor documentation or on the vendor website. Every quantitative fact in this survey entry was recovered from programming manuals, driver telemetry (`teco-smi`), the debugger (TecoGDB), or published performance cost models — sources that are strong on *structure* and silent on *peak performance*. Any spec table for this chip will be mostly "not disclosed", and must not be filled by inference.

---

## Software Stack

The stack is a **complete, deliberate CUDA clone** — nearly every CUDA component has a named SDAA counterpart. It ships as exactly **two packages**, version-locked 1:1: **TecoDriver** (driver + firmware + management library + monitoring) and **TecoToolKit** (the SDK). Current release: **v3.2.0**.

```
PyTorch 2.7.1 / PaddlePaddle 3.0.0 / vLLM / Megatron-LM
  → torch.compile → teco_inductor  (pointwise/reduction/foreach fusion;
                                    Conv and GEMM fall back to libraries)
     ONNX → TecoInferenceEngine (TVM / Relay IR)
  → TecoDNN / TecoBLAS / TecoRAND / TecoLMK / TecoCUSTOM   (cuDNN/cuBLAS analogues)
  → SDAA C (.scpp) → TecoCC (Clang/LLVM-derived)
  → PCX virtual ISA (the PTX of this ecosystem) → PCXAC
  → SDAARuntime (libsdaart.so) + TCCL (libtccl.so)
  → SDAADriver → /dev/tcaicardN → T1 hardware
```

### Framework integration

**TecoPyTorch** (`torch_sdaa`, v3.2.0) is an out-of-tree device extension on **PyTorch 2.7.1** exposing `device='sdaa'` and a `torch.sdaa.*` namespace mirroring `torch.cuda`, with `ProcessGroupTCCL` as `backend='tccl'`. AMP FP16 training is supported; **inference in TecoPyTorch is FP32-only**.

**TecoPaddle** (`paddle_sdaa`) is a PaddlePaddle 3.0.0 CustomDevice plugin. Its most valuable artifact is not Tecorigin's own repo but the **upstream, Tecorigin-independent** [PaddleCustomDevice `backends/sdaa`](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa) (Apache-2.0) — ~115 kernel `.cc` files, a `sdaac_ops/` directory of custom `.scpp` kernels, and a `pr_ci_sdaa.sh` hardware-CI script whose existence implies a real device fleet.

**Teco-vLLM** v3.2.0 is the most feature-complete component: PagedAttention, continuous batching, chunked prefill, automatic prefix caching, **disaggregated prefill (PD分离)**, LoRA, tool calling; TP/PP/EP/DP plus **EPLB** and SBO; Ray for multi-node. **Teco-Megatron-LM** v3.2.0 covers tensor / sequence / pipeline / expert parallelism with a documented DeepSeek-R1-Distill-Llama-70B SFT recipe at TP=4 × PP=8 on 1 node × 8 cards.

### PCX — the virtual ISA

**PCX (Parallel Computing eXecution) v1.0.0** is a **hardware-independent virtual instruction set architecture** — structurally the **PTX of this ecosystem**, and fully documented publicly. It sits between SDAA C and the machine ISA so that "同一版本的PCX指令集可以在太初元碁多种系列的硬件上直接编译并高效执行": infinite virtual general registers, scalar/vector/matrix instruction classes, thread-group hierarchy, debug info, perf-sampling instructions. **The actual T1 machine ISA is not disclosed — PCX exists precisely to hide it.**

The PCX compatibility table currently lists only "T100系列" while its stated purpose is supporting "多种系列" of Tecorigin hardware. That strongly implies planned successor silicon; **no successor is named.**

### Kernel language and compiler

**SDAA C** is C/C++ with CUDA-shaped extensions (`__global__`, `<<<...>>>`, `threadIdx`, `sdaaMalloc`) plus a Sunway-shaped intrinsic set: thread groups, SPM `malloc`/`free`, DMA, **RMA**, hardware broadcast, atomics, `matmul`, transpose, and a large SIMD family. Device sources are `.scpp`. **TecoCC** is demonstrably Clang/LLVM-derived (`.bc` bitcode, `clang-offload-bundler`, `-flto`). Device code is a bare-metal target: no exceptions, RTTI, STL, `new`, global constructors, local statics, file I/O, or native C/C++ atomics.

### Tooling parity

TCCL (NCCL), SDAARuntime (CUDA Runtime), TecoSMI (nvidia-smi), TCML (NVML), SDPTI (CUPTI), **TCPX (NVTX)**, **TSight (Nsight)**, TecoGDB (cuda-gdb, with per-SPE focus switching), **TCVS (DCGM-diag)**, `sdaacfilt` (c++filt). All proprietary binaries.

### Host CPU support — a distinguishing datapoint

TecoDriver and TecoToolKit ship, with full framework stacks, for **five** CPU/OS combinations: x86_64 (Ubuntu 22.04), **Hygon 海光 7380/7375** (Kylin V10), **Phytium 飞腾 S5000C** ARMv8 (Kylin V10 国防版), **Sunway 申威 8A (SW-64)** (UOS Server 20), and **Loongson 龙芯 3C6000 (LoongArch)** (Loongnix Server 23.1). Shipping a full AI stack on an **SW-64 host** is close to unique in this registry.

---

## Hardware Architecture

See `hw-architecture.md` for the full treatment. Headline points:

| Property | Value |
|---|---|
| Chip | **T1** (T100/T110/T111 are card SKUs) |
| Organization | 4 SPAs/card × 32 SPEs/SPA = **128 SPEs/card** |
| Per-SPE units | SU + SREG, VPU + VREG, **FU** (matrix engine, SPM-only), private **SPM** |
| Matrix unit | **128 × 32 × 32 MMA** per operation; weight-stationary with double-buffered weight register |
| Matrix dtypes | **FP16→FP16, FP16→FP32, S16→S16, S16→S32 — and nothing else** |
| SPM | **≥235 KB usable per SPE** (hard ceiling 240512 B); physical size not disclosed |
| Data cache | **None documented** — strict registers / SPM / Global model, all movement explicit |
| Off-chip memory | **HBM**; driver-reported **15296 MB per SPA / 65536 MB (64 GB) per card**. **Bandwidth not disclosed** |
| SPE clock | 2000 MHz (max 3000); perf manuals benchmark at 2.36 GHz |
| On-chip network | **Ring network (环网) @ 2200 MHz** (vendor-stated); SPE array behaves as an **8 × 4 grid** with row/column broadcast buses (derived from vendor cost models) |
| Host interface | **PCIe Gen4 ×16** |
| Scale-up | **No proprietary link documented — PCIe P2P only** |
| Scale-out | Standard **IB/RoCE**; requires MLNX_OFED 5.9 + Open MPI 4.1.5rc2 |
| Peak throughput | **Not disclosed at any precision** |
| Process / die / TDP | **Not disclosed** |

### The matrix unit is 16-bit-only — the sharpest architectural finding

PCX's `matmul_init` encodes exactly four input→output combinations: `0x606` FP16→FP16, `0x806` FP16→FP32, `0x404` S16→S16, `0x704` S16→S32. **There is no BF16, no TF32, no FP8 and no INT8 path in the matrix unit.** Three independent lines of evidence say this is a real hardware limit, not a documentation gap:

1. `TORCH_SDAA_BF16_CLIP` clips bf16 GEMM inputs to **±65407** — essentially the FP16 maximum. bf16 matmul is *emulated by down-converting into the FP16 unit*.
2. SDAA C states bfloat16 "目前仅支持指针操作" — storage only.
3. Teco-vLLM's entire quantization menu is **weight-only** (INT8 W8A16, INT4 W4A16, GPTQ, AWQ, KV-cache-INT8). **No W8A8** — exactly what you expect when activations must enter a 16-bit matrix unit.

This is a significant competitive datapoint: in a 2026 field where FP8 and even FP4 are table stakes for large-model serving, the T1 generation's matrix unit tops out at FP16/S16, and every low-precision win must come from weight-only compression.

### The only public performance number is a micro-benchmark

The operator perf-optimization manual reports **≈29 TFLOPS** through the matrix unit versus **≈2.5 TFLOPS** through general-purpose vector instructions, for an FP16 `128×32×32` MMA loop resident in SPM at a 2.36 GHz SPE clock. Two caveats must always travel with it:

- Tecorigin explicitly writes "上述实验**远没有达到**太初AI加速卡乘加运算的浮点性能峰值" — this is a tutorial benchmark, far below peak.
- **The document does not state the scope.** The kernel is `__global__` with no `threadIdx` guard, so it runs across all 32 SPEs of one SPA, making per-SPA the most likely reading — but that is inference.

**Do not scale this number, and do not present it as a peak.**

---

## Sunway Lineage — refined, not overturned

The lineage claim must be stated carefully, because the obvious version of it is wrong.

**What is sourced:** the core team comes from the National Supercomputing Center in Wuxi (three Gordon Bell Prizes); the shipping stack has first-class **Sunway 申威 SW-64 host CPU** support; the flagship deployment is branded **太湖之光A+** ("TaihuLight A+"); and the SDAA C model reproduces `athread` idioms in depth — a slave-core array with a *documented 2-D-mesh cost model*, an LDM-like private scratchpad with heap/stack/local partitions, explicit DMA plus inter-core RMA plus hardware row/column broadcast, SPMD with per-core IDs, and FP64 in the ISA.

**What is NOT sourced anywhere:** that the SDAA/T1 chip is a Sunway derivative, shares the SW ISA, or was designed by the Jiangnan Institute of Computing Technology. This was explicitly looked for and not found.

**What cuts against a simple derivation claim:** the programming stack is deliberately CUDA-shaped, and PCX is an explicitly PTX-like abstraction layer whose entire purpose is to *decouple* software from the machine ISA. There is **no on-device MPE+CPE core-group structure documented** — the host CPU is the master. (An `Mpe` clock domain *is* exposed by `teco-smi`, and TecoGDB notes SPE indices above 31 denote unnamed special modules, but the MPE's role, count, ISA and programmability are **not disclosed**, and it is invisible to both SDAA C and PCX.)

**Verdict: team / ecosystem / idiom lineage, not architectural derivation.**

---

## Maturity and Deployment

**Shipping production silicon**, with working hardware at fleet scale independently corroborated by the upstream PaddlePaddle SDAA CI backend (commits through 2025-12) and by the 2026 WAIC competition infrastructure (`github.com/tecorigin-waic` issuing private per-team repos with cluster access).

Deployment figures below are **all from Tecorigin's own products page, at unspecified precision, and are VENDOR MARKETING**:

| Site | Claim |
|---|---|
| Yancheng intelligent computing center | 400P planned, **206P first phase**, operating since June 2022, "near full load", >100M RMB revenue, **">60% domestic content"** — so *not necessarily all-Tecorigin silicon* |
| 太湖之光A+ | 200P built; **128 Tecorigin cards per self-designed rack, 32 PFLOPS, 100 kW**; claimed highest compute density in China |
| Yan'an | 200P; "四链路 400 Gbps 计算网络互联" |
| Lihu Future City | 300P; "四链路 400 Gbps" |
| Open-source ecosystem platform | 200P; "四链路 200 Gbps 国产 AI 计算网络" |

**Independent corroboration of these deployment claims remains OPEN.** All "P"/PFLOPS figures are at unspecified precision.

---

## Competitive Position

Tecorigin occupies an unusual niche: a **domestic-supply-chain HPC+AI accelerator with a Sunway-heritage programming model and a CUDA-shaped surface**, positioned for Chinese intelligent-computing centers rather than for the open cloud market.

**Differentiators:**

- **Broadest domestic host-CPU coverage in this registry** — full validated stacks on x86, Hygon, Phytium, **Sunway SW-64**, and Loongson. No other vendor here ships an AI framework stack on SW-64.
- **PCX virtual ISA** — a genuine PTX-analogue with public documentation, which is more ISA-layer transparency than most Chinese vendors offer, while still keeping the machine ISA secret.
- **FP64 in the ISA** plus HPC positioning and Gordon Bell provenance — this is an HPC+AI part, not an inference-only ASIC.
- **Explicit, exposed data movement** (DMA + RMA + hardware broadcast, private per-SPE scratchpad) gives expert programmers unusually direct control — at the cost of a far steeper programming model than a GPU.

**Limitations (all evidence-backed, none inferred):**

- **No peak performance figure exists at any precision.** No datasheet, no MLPerf, no SPEC, no third-party benchmark of any kind.
- **Matrix unit is FP16/S16 only** — no BF16, TF32, FP8 or INT8 compute path; bf16 is emulated by clipping to the FP16 range; quantization is weight-only.
- **No proprietary scale-up interconnect is documented**, and the evidence is *negative*: `teco-smi topo`'s legend is exactly the nvidia-smi PCIe set with no NVLink-equivalent entry. Card-to-card is PCIe P2P; the only high-bandwidth fabric is standard IB/RoCE — depending on **NVIDIA MLNX_OFED 5.9 + SHARP**, which is itself a supply-chain exposure for a domestic-substitution vendor.
- **HBM bandwidth is not disclosed** and cannot be derived (only the 1600 MHz HBM clock is exposed; generation, stack count and bus width are all secret).
- **Per-SPA memory is only ~14.9 GB**, and there is no shared address space across the 4 SPAs on a card, so even single-card work is model-parallel (TP=4 within one card).
- **Developer traction is very small and partly artificial** — 0–4★ on GitHub, 0–31★ on Gitee, nothing on PyPI, no Hugging Face organization, Chinese-only documentation. The most active repos were created 2026-04-13 for a WAIC competition.
- **`torch.compile` has no dynamic-shape path** (`dynamic`, `mode`, `options` are reserved and unsupported), and `teco_inductor` falls back to hand-written libraries for Conv and GEMM.

**Counter-signal worth recording:** the *documentation* release cadence tells a much more credible engineering story than the public-repo star counts do — v2.0 → v3.2 across 2025, TecoPyTorch tracking PyTorch 2.7.1, SDAA C at v3.3.0 (2026-07-23), Teco-vLLM gaining disaggregated prefill and EPLB across 2025-09 → 2026. The public-ecosystem signal and the internal-engineering signal point in opposite directions; both belong in the survey.

---

## Evidence Caveats (carry these into any citation)

1. **WebSearch was unavailable for this entire research pass** (budget exhausted before the first query), for the **second consecutive pass**. There is therefore **no press coverage, no analyst report, no funding record, and no export-control check** behind this entry. That evidence class is unexamined.
2. **The docs portal content is Yjs CRDT binary**, served via an undocumented REST API at `http://docs.tecorigin.com/api/api/…`. Naive byte-scraping interleaves edit history and produces **scrambled digits** — an early heuristic pass produced "2 TFLOPS" where the true text reads "29 TFLOPS". All numbers here were recovered by proper CRDT decoding with `pycrdt`. Anyone re-verifying must decode, not scrape.
3. **All deployment and capacity figures are vendor marketing** at unspecified precision, with no independent corroboration.
4. The **128 SPEs/card** figure is arithmetic (4 × 32) over two independently confirmed vendor-stated numbers. The **8 × 4 SPE grid** is derived from Tecorigin's published RMA/broadcast/sync cost models, not from a topology diagram.

---

## Sources

### Primary vendor documentation (docs.tecorigin.com)

- [SDAA C 编程指南 v3.2.0 (latest v3.3.0)](http://docs.tecorigin.com/release/sdaac)
- [PCX 编程指南 v1.2.0 / PCX ISA v1.0.0](http://docs.tecorigin.com/release/pcx)
- [TecoSMI 用户手册 v1.15.0](http://docs.tecorigin.com/release/tecosmi)
- [SDAARuntime 用户手册 v3.2.0](http://docs.tecorigin.com/release/sdaart)
- [环境安装手册 v3.2.0](http://docs.tecorigin.com/release/software_installation)
- [TecoPyTorch 用户手册 v3.2.0 (PyTorch 2.7.1)](http://docs.tecorigin.com/release/torch2.7)
- [TecoPaddle 用户手册 v3.2.0](http://docs.tecorigin.com/release/tecopaddle)
- [Teco-vLLM 用户手册 v3.2.0](http://docs.tecorigin.com/release/teco_vllm)
- [Teco-Megatron-LM 用户手册 v3.2.0](http://docs.tecorigin.com/release/teco_megatron_lm)
- [TecoInferenceEngine（小模型）用户手册 v3.1.0](http://docs.tecorigin.com/release/tecoinferenceengine)
- [性能优化手册-算子篇 v1.1.0](http://docs.tecorigin.com/release/op_perf_opt) — 29 / 2.5 TFLOPS, 128×32×32 MMA, 3-level memory model
- [性能优化手册-SDAA C篇 v2.0.2](http://docs.tecorigin.com/release/sddac_perf_opt) — RMA cost model, broadcast groups, DMA bandwidth
- [TecoGDB 用户手册 v3.1.0](http://docs.tecorigin.com/release/tecogdb) — "Has 32 GPC", SPA 0–3
- [SDPTI v1.7.0](http://docs.tecorigin.com/release/sdpti) · [TSight GUI v1.9.0](http://docs.tecorigin.com/release/tsight)

### Open source

- [Tecorigin/teco-ops (BSD-3-Clause)](https://github.com/Tecorigin/teco-ops) · [hardware doc](https://github.com/Tecorigin/teco-ops/blob/main/doc/teco-ops-hardware.md) · [README](https://github.com/Tecorigin/teco-ops/blob/main/README.md) · [README_OP.md](https://github.com/Tecorigin/teco-ops/blob/main/doc/README_OP.md)
- [PaddlePaddle/PaddleCustomDevice `backends/sdaa` (INDEPENDENT, Apache-2.0)](https://github.com/PaddlePaddle/PaddleCustomDevice/tree/develop/backends/sdaa)
- [Tecorigin/modelzoo](https://github.com/Tecorigin/modelzoo) · [teco-modelzoo](https://github.com/Tecorigin/teco-modelzoo) · [tecovllm-modelzoo](https://github.com/Tecorigin/tecovllm-modelzoo)
- [Committed run log with full stack banner](https://github.com/Tecorigin/modelzoo/blob/main/PyTorch/contrib/Classification/ACNet-master/scripts/acnet.txt)
- Gitee: [teco-al](https://gitee.com/tecorigin/teco-al) · [teco-torch](https://gitee.com/tecorigin/teco-torch) · [teco-paddle](https://gitee.com/tecorigin/teco-paddle) · [tcap_dllogger](https://gitee.com/tecorigin/tcap_dllogger)

### Vendor site

- [Technology / product matrix](https://www.tecorigin.com/cn/technology.html)
- [Solutions & deployments (marketing)](https://www.tecorigin.com/cn/products.html)
- [About (company facts)](https://www.tecorigin.com/cn/about.html)
- [Binary mirrors](http://mirrors.tecorigin.com/) · [jfrog artifactory](http://jfrog.tecorigin.net/artifactory/)

---

## Update (2026-09-13)

*Scan window 2026-08-08 → 2026-09-13. Classification: Minor. No press/funding/IPO coverage found (English or Chinese); the vendor's own newsroom has not posted since 2026-06-03. GitHub confirms the toolchain is still under active engineering, verified via the GitHub API: `Tecorigin/teco-ops` bumped its HAL dependency to v0.0.2 (2026-08-13) alongside several flash-attention bug fixes (NaN-in-result, multi-batch boundary reads), and `teco-modelzoo`/`tecovllm-modelzoo` had documentation/competition-rule commits through 2026-09-01. No TecoToolKit/TecoDriver version bump beyond the already-recorded v3.2.0, no T200 or second chip generation, no new peak-performance disclosure. See `research/tecorigin/search-results.md` → "Resources Added 2026-09-13".
