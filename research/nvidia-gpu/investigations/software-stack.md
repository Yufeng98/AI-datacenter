# NVIDIA GPU — Software Stack Investigation

**Layer**: Software Stack (cross-cutting)
**Chip**: NVIDIA GPU (Hopper / Blackwell / Rubin)
**as_of**: 2026-08-08

---

## Scope and relationship to the other investigations

The NVIDIA software stack is investigated per component elsewhere in this directory:
`cuda-programming-guide`, `cuda-runtime`, `cccl-cublas`, `cudnn`, `cutlass`, `triton`,
`nccl`, `ptx-isa`, and `open-gpu-kernel-modules`. This file is the **cross-cutting,
dated record of platform-level software changes**, created 2026-08-08 with the first
such entry. It does not restate the per-component investigations; it records what
moved, when, with sources, and — equally important — what claims were checked and
found unsupported.

---

## Update — 2026-08-08 (window 2026-04-05 → 2026-08-08)

The software stack, not the silicon, is where NVIDIA moved in this window. Two themes:
a **tile-level programming stack** (Tile IR + cuTile) that the survey had not covered at
all, and **NCCL 2.30.x**, which removes SM occupancy from collectives and makes
Mixture-of-Experts communication a first-class library.

### 1. CUDA tile programming stack — a repo coverage gap, *not* a window development

NVIDIA now ships a tile-level programming path parallel to the classic
CUDA C++ → NVVM → PTX → SASS path.

| Component | What it is | Timeline |
|---|---|---|
| `NVIDIA/cuda-tile` | "MLIR-based IR and compiler infrastructure … targeting NVIDIA tensor core units" | repo created **2025-11-05**; v13.1.0 **2026-01-14**; v13.2.0 **2026-03-24**; v13.3.0 2026-05-28; v13.3.1/v13.3.2 2026-06-27; v13.3.3 2026-07-22 |
| `NVIDIA/cutile-python` (cuTile) | "a programming model for writing parallel kernels for NVIDIA GPUs" — Python tile-kernel DSL | repo created **2025-06-13**; v1.3.0 2026-04-19; v1.4.0 2026-05-26; v1.5.0 2026-07-07 |
| **CUDA TILE-IR AS** | Tile IR assembler, a distinct CUDA Toolkit component | **13.2.51** in CUDA 13.2 GA → **13.3.36** in CUDA 13.3 Update 1 → **13.4.46** in the 13.4 developer preview |

**Framing matters here.** The flagship components all predate this survey's 2026-04-05
baseline; only point releases landed inside the window. This is a gap in the survey's
coverage being closed, not news.

**Claims checked and rejected:**
- "cuTile went from experimental in 13.1 to stable in 13.2" — **unsupported**. No release
  note states any experimental-to-stable transition. The cutile-python README only
  separates a stable core `cuda.tile` API from separately marked experimental features.
- "CUDA TILE-IR AS is new in the 13.4 preview" — **false**; it was already in CUDA 13.2 GA.

**Limitations worth recording for a survey:** per the cutile-python README, the
`tileiras` compiler (version 13.2) "only supports Blackwell GPU and Ampere/Ada GPU.
Hopper GPU will be supported in the coming versions." Requires CUDA Toolkit 13.1+ and
driver r580+. A survey that lists cuTile without this caveat would wrongly imply Hopper
coverage.

**Why it matters architecturally.** Tile IR gives NVIDIA a first-party tile-level virtual
ISA alongside PTX, i.e. an in-house answer to Triton's tile abstraction and to CUTLASS's
CuTe DSL, with an assembler shipped in the toolkit rather than a third-party compiler.

### 2. CUDA Toolkit

- **Current GA: CUDA 13.3 Update 1.**
- **CUDA 13.4 Developer Preview**, release notes dated **Jul 16, 2026** (verified in the
  PDF text). Listed contents, verbatim: "New and updated PTX ISA, **including some Rubin
  capabilities**"; "CUDA Tile and TileIR compilation and inspection"; "CUDA Toolkit
  preview for RTX Spark and Rubin"; plus an RTX Spark Windows-on-Arm preview. Component
  versions: CUDA TILE-IR AS 13.4.46, cuBLAS 13.7.0.10, cuFFT 12.4.0.16. CUDA C++/Tile
  additions listed: numeric formats and mixed precision, MMA operations, tile
  views/layouts/sub-views, reductions and user-defined reductions, tile-level
  synchronization and communication.
  - Packaging changes with real downstream impact: **starting with CUDA 13.4 the NVIDIA
    Linux driver is no longer bundled with the CUDA Toolkit**, and 13.4 features require
    an **R616+** driver.
  - **Usage constraint:** NVIDIA marks this pre-release — performance data from it "must
    not be published, reported, compared, or used to characterize NVIDIA hardware." It is
    cited only for feature existence.
- **No Rubin compute capability (`sm_`) is named in any GA CUDA release.** GA notes
  reference sm_90, sm_100, sm_103, sm_120, sm_121. (A circulating "sm_110 (arm64-sbsa)"
  claim could not be confirmed and is not recorded.)
- **CUDA 13.2 GA library details**: cuBLASLt experimental Grouped GEMM extended to MXFP8
  inputs on compute capability 10.x and 11.0; FP64 fixed-point emulation added to
  `cublas[D|Z]syrk` / `syr2k` / `Zherk` / `Zher2k`; cuSOLVER fixed-point FP64 emulation
  APIs.
  - **Claim checked and rejected:** "CUDA 13.2 extended Cooperative Groups with new
    `grid_group` synchronization primitives" — unsupported; the 13.2 notes document only
    CUDA 13.0's *removal* of the multi-device (`this_multi_grid`) APIs.
- **Documentation pins refreshed:** PTX ISA is now **v9.3** (repo pinned v9.2); cuBLAS is
  **13.7.x** (repo pinned v13.2).

### 3. NCCL 2.30.x

Release dates independently corroborated against git committer timestamps on the tagged
commits, not only the release page.

| Release | Date | Content |
|---|---|---|
| v2.30.3-1 | 2026-04-15 | GIN contexts no longer shared between device communicators on the same host communicator; per-context GPU-/CTA-scoped GIN resource sharing; **elastic buffers** (large tensors split into multi-segment windows; active region in GPU memory, remainder in host memory); **TMA support in select built-in symmetric kernels** (`NCCL_SYM_TMA_ENABLE=1`); experimental `gin.get` with nonblocking flush; Dynamic Direct Path (DDP) |
| v2.30.4-1 | 2026-04-22 | Elastic buffer support with GIN; `nccl_param` header fix |
| v2.30.7-1 | 2026-06-04 | **Hierarchical zero-SM collectives (AllGather, All2all)** — RMA CPU proxy inter-node, Copy Engines intra-node, enabled with `NCCL_CTA_POLICY_ZERO`, for better compute/communication overlap; **experimental GPU Push Interface (GPI) backend for GIN**; explicit Strong/Weak GIN signal semantics; RMA plugin restructure; window registration during CUDA graph capture; **experimental MPS with MLOPart** (CUDA Memory Locality Optimized Partition, up to 2 ranks per physical GPU) |

**Zero-SM collectives are the architecturally notable item**: they take collectives off the
SMs entirely, so communication no longer competes with compute for occupancy — the classic
tension that motivated SM-count tuning of NCCL channels. GPI and MLOPart are labelled
*experimental* by NVIDIA and must be described that way.

### 4. MoE collectives as a first-class library

- **`nccl-ep-v0.1.0`, 2026-06-08** — NCCL EP: "a high-performance NCCL API extension for
  efficient Mixture-of-Experts (MoE) communication," providing dispatch/combine primitives
  for Expert Parallelism built on the **NCCL Device API (LSA + GIN)**. CUDA-Graph-compatible
  handle management splits `ncclEpInitHandle` (control path) from `ncclEpUpdateHandle`
  (data path); user-settable SM count; NCCL Window association for zero-copy;
  Flat / Expert-major / Rank-major dispatch layouts; active-rank masking for fault
  tolerance.
  - **Claim checked and rejected:** "full Multi-Node-NVLink support" — MNNVL is never
    mentioned in the v0.1.0 notes.
- **`nccl4py-v0.3.1`, 2026-06-11** — adds the `nccl.ep` Python package,
  `nccl.core.device.cute` (CuTeDSL kernels calling NCCL device APIs), and free-threaded
  CPython support. **Not new**: v0.2.0 shipped 2026-04-24 with RMA/GIN/elastic-communicator
  bindings.
- **Correction to circulating descriptions:** both are *release tags inside the
  `NVIDIA/nccl` repository*, not standalone repositories.
- **New repository:** **`NVIDIA/nccl-extensions`**, created **2026-07-07** —
  "Communication patterns for AI, built on top of NCCL device and host APIs" — actively
  developed through 2026-08-08. Most likely consolidation point for future NCCL EP-style
  extension work. Watch item.

### 5. Benchmark-relevant software versions

MLPerf Training v6.0 (results 2026-06-16) NVIDIA submissions ran on **NeMo container
26.06** with full-iteration CUDA Graphs for token-dropless MoEs and Spectrum-X Ethernet
Advanced Adaptive Routing. (A circulating "NeMo Framework 26.04" attribution is wrong.)

### 6. Not usable

Hot Chips 38 (Aug 23–25, 2026) is 15 days in the future as of this update. Its program is
public but no slides, abstracts, or specs exist — disclosure scheduled, content not yet
public. No talk title is cited here as the source of any specification.

---

## Sources

- [CUDA Toolkit Release Notes (GA — 13.3 Update 1)](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.4 Developer Preview Release Notes PDF (2026-07-16; pre-release)](https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/CUDA_Toolkit_Release_Notes.pdf)
- [CUDA 13.2 Archived Release Notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
- [PTX ISA (v9.3)](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [cuBLAS Documentation (13.7.x)](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA/cuda-tile](https://github.com/NVIDIA/cuda-tile)
- [NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
- [cutile-python README (stable vs experimental API; tileiras 13.2 Blackwell + Ampere/Ada only; CUDA 13.1+ / driver r580+)](https://raw.githubusercontent.com/NVIDIA/cutile-python/main/README.md)
- [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton — NVIDIA Technical Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton)
- [NVIDIA/nccl releases (API-fetched: v2.30.3-1, v2.30.4-1, v2.30.7-1, nccl-ep-v0.1.0, nccl4py-v0.2.0, nccl4py-v0.3.1)](https://github.com/NVIDIA/nccl/releases)
- [NVIDIA/nccl-extensions (created 2026-07-07)](https://github.com/NVIDIA/nccl-extensions)
- [NVIDIA Blackwell Tops MLPerf Training v6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/)
