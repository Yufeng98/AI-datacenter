# NextSilicon Maverick-2 Software Stack Investigation

*as_of: 2026-08-08*
*chip: nextsilicon-maverick*
*device_class: Reconfigurable Dataflow Accelerator (runtime-JIT spatial grid; HPC-first)*

---

## Overview

The programming model **is** the product. There is **no kernel language, no kernel compiler for device
code, and no device-side source dialect at all** — and that absence is the deliberate architectural choice,
not a gap in this investigation. The vendor's slogan is **"BYOC — bring your own code"**, and the
programming-model row of the vendor spec table reads simply **"Any program."**

Because the SDK is gated (`docs.nextsilicon.com` returns **HTTP 401 Unauthorized**, re-verified
2026-08-08) and the architecture whitepaper is still "COMING SOON", almost every concrete component name
below comes from **upstream Kokkos** — a non-vendor-controlled repository where NextSilicon had to expose
real interface names in order to get a backend merged. That evidence is unusually strong for a closed
stack: it is source code and CI configuration, not marketing copy.

---

## Layer-by-Layer Summary

| # | Layer | Component | Open / Closed | Confidence |
|---|-------|-----------|---------------|------------|
| 1 | Framework integration | **None shipping.** Vendor product page places CUDA, HIP/ROCm and "leading AI frameworks" under "upcoming integrations planned" | — | confirmed |
| 2 | Graph capture | **Not applicable** — no graph capture; there is no ML graph in the model | — | confirmed |
| 3 | Graph compiler | **Not applicable** — no ML graph compiler exists | — | confirmed |
| 4 | Kernel compiler | **`nextcxx`** — the C++ compiler driver (LLVM/MLIR-based), plus a bundled LLVM binutils tree at `$NEXT_HOME/llvm/bin/` | proprietary (LLVM-derived) | confirmed |
| 5 | Kernel language | **None.** ISO C/C++/Fortran + OpenMP + Kokkos. No CUDA analogue, no DSL | — | confirmed |
| 6 | Tensor API | **None.** The unit of work is `parallel_for` over an index range, not a tensor | — | confirmed |
| 7 | Runtime | **`nextsystemd`** (optimizer/telemetry daemon), **`libnextapi.so`** (NextAPI host API), **`nextcli`** (control CLI), persistent **optimizer cache** | proprietary | confirmed |
| 8 | Driver | Kernel driver exists (PCIe Gen5 device) but is **not named publicly**; SDK installs to `$NEXT_HOME`, default `/opt/nextsilicon` | proprietary | likely |
| 9 | ISA | **No published ISA.** Grid configuration is a projected dataflow graph, not an instruction stream. Fallback tiers use RISC-V (E-cores) and x86-64 (host) | proprietary | confirmed |
| — | Portability layer | **Kokkos `NextSilicon` execution space** — merged **upstream** in Kokkos 5.2.0 | **open source** (Apache-2.0 WITH LLVM-exception) | confirmed |
| — | Profiling / debug | **Profiler**, **Chip Viewer**, **Projection Viewer**; Kokkos Tools `DeviceType::NextSilicon` | proprietary + open hooks | confirmed |
| — | Collective comms | **None named in any source** (no NCCL/RCCL analogue) | — | confirmed absent |

---

## 1. The Execution Model — a Two-Pass Train-then-Project Flow

The CI wrapper script NextSilicon contributed to upstream Kokkos documents the flow precisely. From
`scripts/nextsilicon-test-wrapper.sh` *(open source, primary)*:

```sh
# if KOKKOS_NEXTSILICON_TEST_TELEMETRYLESS is anything, we run in a telemetry-less mode,
# where any projection error counts as a failure
# else, we run the two-pass telemetry-based flow
...
nextsystemd --ui-collector-address none --cfg-file ${patch_dir}/kokkos.patch &
# training run
./"$1" "${@:2}"
# wait for optimization/projection to finish, up to 5 minutes
status="$(nextcli application status | grep 'Optimization state:')"
if   [[ $status == *IDLE*       ]]; then exit 0      # no mills found, return
elif [[ $status == *IMPROVED*   ]]; then ./"$1" ...  # optimization/projection finished, do device run
elif [[ $status == *OPTIMIZING* ]]; then sleep 10; continue
```

Read out step by step:

1. **Compile once, ahead of time**, with `nextcxx`. One binary. No separate device object, no fat binary,
   and **no `Kokkos_ARCH_*` target flag** — NextSilicon is the only Kokkos device backend with no
   architecture option (`cmake/kokkos_arch.cmake`).
2. **Training run.** The program starts executing **on the host** while `nextsystemd` collects telemetry.
   Kokkos' own comment: "NextSilicon kernels **may run on the host during training/telemetry collection**
   while still being 'within' the device execution space."
3. **Optimization / projection.** `nextsystemd` identifies hot flows, builds an optimized compute graph,
   and **projects** it onto the grid as mill cores. The state machine is observable via
   `nextcli application status`: `IDLE` → "no mills found" (nothing was worth accelerating) /
   `OPTIMIZING` / `IMPROVED`. CI budgets up to **5 minutes** of wall clock for this.
4. **Device run.** Re-execute; hot regions now "hand off" to the grid.
5. **Continuous re-optimization.** Telemetry keeps flowing and the mapping keeps changing. The Next
   Platform: the runtime "has algorithms that constantly analyze how that resulting dataflow is
   functioning and changes it on the fly, without human intervention, to improve it. **The longer the code
   runs, the better it gets.**"
6. **Persistent optimizer cache.** Projections are cached across runs under `XDG_CACHE_HOME` (Kokkos CI
   maps a named volume to `$HOME/optimizer-cache`; PR #9162 "enable persistent optimizer cache"). The
   training pass is therefore amortized across invocations — a real, shipping mechanism, not a demo.
7. **Telemetry-less mode.** An alternative static-projection path exists
   (`optimizer-pi: { enable-telemetry-less: true }`) where "any projection error counts as a failure". As
   of the current upstream CI it is **disabled**: "FIXME_NEXTSILICON: re-enable when telemetry-less mode is
   working."

**Runtime tunables exposed** via `nextsystemd --cfg-file`: the config tree is `optimizer-pi:` containing
`enable-telemetry-less` and `mlc: { acceleration-threshold: N }`. Kokkos CI sets
`acceleration-threshold: 1` "to try to offload every parallel region." *The expansion of "mlc" and "pi" is
**not disclosed** — do not guess.*

### What the telemetry observes, and what the optimizer does to the mapping

From the vendor's flow diagram, the loop is:
**① SOFTWARE: Identify likely flow** (a histogram of *branch* execution likelihood, partitioned into
GRID / RISC-V E-CORES / HOST CPU) → **② SOFTWARE: Generate optimized compute graph** →
**③ HARDWARE: Project compute graph onto grid**, with a **TELEMETRY** arrow feeding ③ back into ②.
Figure caption, verbatim: "NextSilicon's intelligent software identifies the **top 1% of code flows** at
runtime, then projects a performance optimized compute graph onto the silicon. Telemetry automatically
reconfigures your hardware **in nanoseconds** — adapting to your application, not the other way around."
The figure carries the patent marking **US16/053,382**.

The observable is therefore **branch/path execution counts**, not performance counters in the usual sense.
Corroborated three ways:

- Patent **WO2019055675A1** (priority 2017-09-13; Raz, Tayari): telemetry tracks "the number of times that
  each logic path has been taken" to identify "likely compute paths" statistically.
- Patent **US11995419B1** (the Projection Viewer): the GUI surfaces "**control flow branch counters**"
  alongside cache-bin and HBM occupancy.
- Chips and Cheese: Raz frames the model in terms of "**likely flows**" and "**unlikely flows**";
  NextSilicon has a 2026 application US20260086816A1, "Optimizing execution of code on reconfigurable
  hardware using likely data…".

The four re-mapping actions the optimizer performs are enumerated in patent **US20190042282A1**, "Runtime
optimization of configurable hardware" (priority 2017-08-03, Elad Raz), performed "in real-time,
simultaneous to the operation of the target computational device":

| Action | Patent description |
|--------|--------------------|
| **Relocation** | "Two computation groups that often call each other are allocated topologically close to each other" — this is what the non-uniform NoC penalty makes matter |
| **Duplication** | Hot groups are "duplicated several times on the grid to speed up calculation and open bottlenecks" |
| **In-lining** | Two frequently-communicating groups fused into one logical group |
| **Shrinking** | Unused duplicates "reconfigured out" and their resources released |
| **Host demotion** | A function "not needed in an accelerator… is relocated to be executed in a CPU" |

The vendor FAQ describes the same loop as "relocating communicating cores and duplicating bottlenecked
cores" in "nanoseconds, with zero developer input."

---

## 2. Compiler / IR Path

**MLIR is confirmed as the compiler stack**, from two independent directions:

- Upstream Kokkos ships an intrinsic whose only purpose is to talk to it.
  `Kokkos_NextSilicon_Intrinsics.hpp` declares
  `extern "C" void __ns_immutable_thread_invariant_parameter_struct(const void*)`, documented as "an
  internal-use-only intrinsic used to communicate from Kokkos C++ code **to the MLIR compiler stack** that
  a struct… can be considered immutable and thread invariant for the full duration of **the microtask**."
  Its body is a deliberate no-op, "purposefully left empty to be filled in by NS toolchain."
- NextSilicon's compiler-engineer job posting advertises an "**innovative MLIR compiler team**" and asks
  candidates to "work with and contribute to **upstream MLIR and LLVM**."

The **public LLVM fork** (`nextsilicon/llvm-project`, 41 branches, active through 2026-08-01) indicates
*what kind* of MLIR work this is. Branch names cluster tightly around **importing LLVM IR into MLIR while
preserving optimization-critical metadata, then running classical scalar optimization inside MLIR**:

- IR import / metadata: `loop-info`, `loop-info-import`, `loop-info-refactor`, `parameter_attributes`,
  `more-param-attrs`, `parameter_attribute_import`, `access-group-attr`, `align-attribute`,
  `passthrough-import`, `llvm-typed-ptr`, `experimental-noalias-scope`
- Scalar optimization in MLIR: `mlir-mem2reg`, `mlir-mem2reg-memref`, `mlir-sroa`
- Inlining: `inline-llvm-func`, `move-llvm-inliner`, `debug-llvm-inliner`, `inliner-interface`,
  `inline-alloca`, `inline-lifetime-intrinsics`

*(Inferred from branch names — strong but circumstantial.)* This matches The Next Platform's description
exactly: "You can take existing C, C++, or Fortran code, **grab its intermediate representation** and plunk
that down onto the ICA."

### Correction to the research seed — "AIIR" is not a NextSilicon IR

An earlier note treated the fork's `mlir-to-aiir` branch as evidence of an "MLIR-to-proprietary-IR lowering
path" named AIIR. **This could not be substantiated and the evidence points the other way.** The branch
contains exactly **two** non-upstream commits, both by a single developer, both timestamped **2026-04-01**:
*"[AIIR] Rename MLIR to AIIR 🚀"* and *"[AIIR] Also change occurrences that are spelled out."* It is a bulk
textual find-and-replace of the string "MLIR" atop an upstream snapshot, with a rocket emoji, on April
Fools' Day. **Do not record AIIR as a NextSilicon IR name.** The broader claim that NextSilicon lowers via
MLIR remains well supported — just not by this branch.

### Compiler invocation

From the Kokkos SNL CI workflow (job `SNL_NextSilicon_1_2_0`, runner label `ns-1.2.0`):

```
-DCMAKE_CXX_COMPILER=nextcxx
-DCMAKE_AR=$NEXT_HOME/llvm/bin/ar
-DCMAKE_RANLIB=$NEXT_HOME/llvm/bin/ranlib
-DCMAKE_CXX_STANDARD=20
-DKokkos_ENABLE_NEXTSILICON=ON
-DCMAKE_CXX_FLAGS="-Werror -Wno-return-type -Wno-invalid-noreturn"
```

Note the absence of any `-DKokkos_ARCH_*` flag, and the need to suppress `-Wreturn-type` /
`-Winvalid-noreturn` — a compiler-maturity signal. SDK versions visible in CI: **1.2.0** and **1.3.0-395**.
`nextcli system status` reports the device as "Maverick 2".

---

## 3. Runtime — NextAPI

`NextAPI` is registered in upstream Kokkos as a third-party library (`kokkos_tpl_option(NEXTAPI ...)` in
`cmake/kokkos_tpls.cmake`). Its finder module `cmake/Modules/FindTPLNEXTAPI.cmake` states: "Nextapi shared
object and header both reside under NEXT_HOME directory / Library under `${NEXT_HOME}/lib` / Headers under
`${NEXT_HOME}/include`", defaulting `NEXT_HOME` to **`/opt/nextsilicon`**, library name **`nextapi`**.

Headers and symbols exposed through the Kokkos backend — the only public surface of the proprietary
runtime API:

| Header | Symbols | Semantics |
|--------|---------|-----------|
| `nextapi/parallelism.hpp` | `nextapi::parallel_for(begin, end, {.chunk_size=N}, fn, ctx)` | OpenMP-style host-driven parallel loop. This is the *entire* dispatch API — no streams, no queues, no kernel launch |
| `nextapi/intrinsics.h` | `__next_is_in_handed_off_code()` | Returns true iff the calling thread is *actually* executing on the grid. "Handoff" is the official term for host→grid transfer |
| `nextapi/memory.h` / `.hpp` | `nextapi_memory_copy(dst, src, n)`, `nextapi_memory_fill(...)`, `nextapi_mem_migrate(ptr, size, NEXTAPI_PAGE_LOC_HOST, pin)` | Copies "let NextSilicon runtime select the correct implementation"; migrate/pin drives UVM placement |

### Execution semantics revealed by the backend

- `NextSilicon::fence()` is a **no-op**: "fence doesn't do anything in the OpenMP-style interface."
  Dispatch is **synchronous and blocking**, not the async stream model of CUDA/HIP/SYCL.
- Kernel launch takes a **device mutex**: "Acquire the device for potential handoff before kernel execution
  begins." One kernel at a time.
- The functor is copied into a **4 MB minimum heap buffer** in shared space, and "clone[d]… to prevent the
  stack from getting migrated to device."
- `KOKKOS_IF_ON_HOST` / `KOKKOS_IF_ON_DEVICE` **cannot be resolved at compile time**.
  `Kokkos_NextSilicon_ThreadSpaceGuard.hpp` implements a *runtime* thread-local:
  `is_on_device() = __next_is_in_handed_off_code() || host_thread_is_on_device()`. Its comment is the
  clearest statement of the whole model anywhere public: "**unlike other backends, NextSilicon kernels may
  run on the host during training/telemetry collection while still being 'within' the device execution
  space, so we cannot simply equate 'on device' with 'offloaded to device'.**"
- `printf` and `abort` from grid code **do not work** — both are guarded by
  `if (!__next_is_in_handed_off_code())` (open tickets CS-515, CS-694).
- `fork()` is unsupported by the runtime (Kokkos death-tests are filtered out in CI).
- Device-side cycle counters are unavailable (`clock_tic_device()` unimplemented, PR #9378).
- Public open tickets referenced in upstream code: CS-515, CS-611, CS-682, CS-694, CS-737, SW-25677
  (`nextsilicon.atlassian.net`).

NextAPI's full API surface is **not disclosed** — only the symbols reachable from the Kokkos backend are
public, and there is no published NextAPI reference manual.

---

## 4. Portability Layer — the Kokkos Backend (the only open-source component)

Merged **upstream** into Kokkos, i.e. genuine non-vendor-controlled evidence. The Kokkos 5.2.0 CHANGELOG,
under "Backend and Architecture Enhancements → NextSilicon", reads: *"Add `NextSilicon` execution space and
`NextSiliconSharedSpace` memory space (#8998, #9100)."*

Source: `core/src/NextSilicon/`, 19 files, Apache-2.0 WITH LLVM-exception. Registered as space factory
`"180_NextSilicon"`; `execution_space = NextSilicon`, `memory_space = NextSiliconSharedSpace`,
`array_layout = LayoutLeft`, `concurrency() = 65536` (marked FIXME placeholder — **not** a hardware spec).

**Maturity is early and honestly tracked in public.** Tracking issue `kokkos/kokkos#9032` (opened
2026-04-02, assigned to Kokkos maintainers cwpearson and crtrott) lists 16 levels. **Only `RangePolicy`
`parallel_for` (plus DeepCopy/View) has landed.** Still outstanding:

- `RangePolicy parallel_reduce`
- `MDRange parallel_for` / `parallel_reduce`
- `TeamPolicy parallel_for`, nested parallelism, **scratch memory**
- `parallel_scan`, SIMD
- Containers, Algorithms, Benchmarks
- the separate `NextSiliconSpace` (device-only memory space, distinct from the shared space)

NextSilicon's staging fork `nextsilicon/kokkos-public` (branches `nextsilicon-backend-pr-1/2/2b/2c`, latest
2026-07-16) contains the same file set as upstream — so **nothing more advanced is public**. Roughly 15
further merged PRs cover CI, atomics/LockPolicy, thread safety, allocation-failure exceptions, and the
persistent optimizer cache.

Related public forks: `kokkos-kernels-public` (BLAS/sparse), `kokkos-tools-public`, `kokkos-core-wiki`, and
`ns-gem5` (gem5 fork, `stable` branch active 2026-08-06; visible work is RISC-V PMP checkpointing, cache
replacement policies and prefetcher fixes — consistent with modeling the Arbel/E-core side, **not** the
dataflow grid). The GitHub org has 14 public repos, **all forks**; there is **no first-party NextSilicon
SDK repository**.

---

## 5. Developer Tools

Named on the vendor tech page — all proprietary, all telemetry-driven:

- **Profiler** — "real-time telemetry data that gives you a clear understanding of performance patterns and
  bottlenecks."
- **Chip Viewer** — "live, hardware-level information."
- **Projection Viewer** — "visualizes what's happening **inside each mill core in real time**."

Patent **US11995419B1** appears to describe the Projection Viewer: a GUI "simultaneously presenting… a
source code and an interactive graph of nodes connected by edges representing the source code mapped to
physical configurable elements", with bidirectional source↔graph navigation via compiler debug data, live
**control-flow branch counters**, cache-bin/HBM occupancy, and grid-occupancy indicators, across four
abstraction levels (**logical → compute → hardware → projection**). Its terminology — "**logical element
units (LEUs) optionally arranged into grid compute units (GCU)**" — is the patent-side vocabulary for what
marketing calls mill cores and compute blocks.

`nextsystemd --ui-collector-address` is the telemetry sink these tools attach to. Kokkos Tools integration
exists upstream (`Kokkos::Profiling::Experimental::DeviceType::NextSilicon`), though device_id reporting is
stubbed pending NextAPI support (ticket CS-611).

---

## 6. Language Support — resolving the vendor's self-contradiction

The two vendor pages disagree, and the disagreement matters:

| Source | Claim |
|--------|-------|
| Product page | "natively supports **C/C++, FORTRAN, OpenMP, and Kokkos**, with **upcoming integrations planned** for CUDA, HIP/ROCm, and leading AI frameworks" |
| FAQ | "natively supports C/C++, **Python**, Fortran, **CUDA**, Kokkos, **ROCM/HIP, OpenCL, Tensorflow, OneAPI**, and other standard AI frameworks" |
| Launch blog | runs "unmodified C++, Python, Fortran, CUDA, and AI framework code out of the box" |

**All independent corroboration supports the product page only.** The upstream Kokkos backend exists; a
`nextcxx` C++ driver exists; nothing public shows Python, CUDA, ROCm, OpenCL, TensorFlow or oneAPI. Chips
and Cheese's assertion that the part is "compatible with CUDA and ROCm frameworks" is a podcast paraphrase
and conflicts with the product page.

**Record: C/C++, Fortran, OpenMP and Kokkos as shipping; CUDA, HIP/ROCm, Python and AI frameworks as
roadmap.** Fortran is plausible (Flang is an LLVM project) but no Flang-specific artifact was found, so
Fortran is **vendor-stated only**.

A *plausible mechanism* for eventual CUDA support exists in patent **EP4668102A1**, "Dynamic software
interface translation for computing in a heterogeneous computing environment" — platform-independent
control-transfer information (out-values/in-values at basic-block boundaries) enabling runtime
redistribution of code across incompatible architectures, with JIT and IR (LLVM, MSIL) named. It does
**not** mention CUDA or GPUs. *(Patent disclosure; do not present as CUDA support.)*

---

## 7. Open vs Proprietary — Summary

**Open source:**
- The Kokkos `NextSilicon` backend (upstream, Apache-2.0 WITH LLVM-exception)
- `scripts/nextsilicon-test-wrapper.sh` (the CI script that documents the execution model)
- The LLVM/MLIR fork (Apache-2.0 WITH LLVM-exception; staging for upstream)
- Kokkos Kernels / Tools forks; the gem5 fork (BSD)

**Proprietary and closed:**
`nextcxx`; the MLIR→grid lowering; the projection/mapping algorithm; `libnextapi.so`; `nextsystemd`;
`nextcli`; the optimizer; the telemetry format and sampling rate; the optimizer-cache format; the kernel
driver; the grid configuration encoding ("ISA"); Profiler / Chip Viewer / Projection Viewer; and all SDK
documentation (401-gated).

**Not applicable by design:**
kernel language; kernel compiler for device code; graph capture; graph compiler; tensor API; collective
communication library.

---

## 8. Explicitly Not Disclosed

- Name, source availability and interface of the Linux kernel driver / device node
- The grid configuration encoding — no published ISA, bitstream format, or configuration-word spec
- Expansion of the runtime config keys `optimizer-pi` and `mlc`
- Internal design of `nextsystemd`'s projection/mapping algorithm, telemetry sampling rate, telemetry
  record format, and optimizer-cache format
- NextAPI's full API surface (no reference manual; `docs.nextsilicon.com` is 401-gated)
- Whether Fortran support is via LLVM Flang or another front end
- Any collective-communication library (NCCL/RCCL analogue) — none named in any source

---

## Sources

- [BYOC — bring your own code (vendor blog)](https://www.nextsilicon.com/insights/BYOC_maverick2_blog/)
- [Maverick-2 product page (language support)](https://www.nextsilicon.com/maverick)
- [NextSilicon FAQ (contradicting language claims)](https://www.nextsilicon.com/faq)
- [NextSilicon Tech page (Profiler / Chip Viewer / Projection Viewer)](https://www.nextsilicon.com/tech/)
- [Compiler Engineer job posting ("innovative MLIR compiler team")](https://www.nextsilicon.com/careers/compiler-engineer/)
- [SDK docs portal — HTTP 401 Unauthorized](https://docs.nextsilicon.com/)
- [Upstream Kokkos NextSilicon backend (19 files)](https://github.com/kokkos/kokkos/tree/develop/core/src/NextSilicon)
- [`nextsilicon-test-wrapper.sh` — the two-pass flow](https://github.com/kokkos/kokkos/blob/develop/scripts/nextsilicon-test-wrapper.sh)
- [`Kokkos_NextSilicon_ThreadSpaceGuard.hpp`](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_ThreadSpaceGuard.hpp)
- [`Kokkos_NextSilicon_Intrinsics.hpp` — "the MLIR compiler stack"](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_Intrinsics.hpp)
- [`Kokkos_NextSilicon_ParallelFor_Range.hpp`](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSilicon_ParallelFor_Range.hpp)
- [`Kokkos_NextSiliconSpace.cpp`](https://github.com/kokkos/kokkos/blob/develop/core/src/NextSilicon/Kokkos_NextSiliconSpace.cpp)
- [Kokkos SNL CI workflow (nextcxx invocation)](https://github.com/kokkos/kokkos/blob/develop/.github/workflows/snl-ci.yml)
- [`FindTPLNEXTAPI.cmake`](https://github.com/kokkos/kokkos/blob/develop/cmake/Modules/FindTPLNEXTAPI.cmake)
- [Kokkos CHANGELOG — 5.2.0 NextSilicon entry](https://raw.githubusercontent.com/kokkos/kokkos/develop/CHANGELOG.md)
- [Kokkos issue #9032 — NextSilicon Backend Tracking](https://github.com/kokkos/kokkos/issues/9032)
- [Kokkos PR #8998](https://github.com/kokkos/kokkos/pull/8998) · [#9100](https://github.com/kokkos/kokkos/pull/9100) · [#9162](https://github.com/kokkos/kokkos/pull/9162) · [#9378](https://github.com/kokkos/kokkos/pull/9378)
- [nextsilicon/llvm-project fork](https://github.com/nextsilicon/llvm-project)
- [`mlir-to-aiir` branch commits (April Fools rename)](https://github.com/nextsilicon/llvm-project/commits/mlir-to-aiir)
- [nextsilicon/kokkos-public](https://github.com/nextsilicon/kokkos-public) · [kokkos-kernels-public](https://github.com/nextsilicon/kokkos-kernels-public) · [kokkos-tools-public](https://github.com/nextsilicon/kokkos-tools-public) · [ns-gem5](https://github.com/nextsilicon/ns-gem5)
- [NextSilicon GitHub org (14 repos, all forks)](https://github.com/orgs/nextsilicon/repositories)
- [Three-stage flow figure (telemetry loop, "top 1% of code flows")](https://image.nextplatform.com/214600.webp?imageId=214600&width=1412&height=598&format=jpg)
- [Patent US20190042282A1 — runtime optimization actions](https://patents.google.com/patent/US20190042282A1/en)
- [Patent WO2019055675A1 — likely compute paths](https://patents.google.com/patent/WO2019055675A1/en)
- [Patent US11995419B1 — Projection Viewer GUI](https://patents.google.com/patent/US11995419B1/en)
- [Patent EP4668102A1 — dynamic software interface translation](https://patents.google.com/patent/EP4668102A1/en)
- [Patent US20260086816A1 — optimizing using likely data](https://patents.google.com/patent/US20260086816A1/en)
- [The Next Platform launch deep-dive](https://www.nextplatform.com/compute/2025/10/22/nextsilicon-takes-aim-at-cpus-and-gpus-with-maverick-2-dataflow-engine/1639749)
- [Chips and Cheese — "NextSilicon: Putting HPC First"](https://chipsandcheese.com/p/nextsilicon-putting-hpc-first)
