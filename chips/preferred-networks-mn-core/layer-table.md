# Preferred Networks MN-Core Layer Mapping Table

*as_of: 2026-09-13*
*chip: preferred-networks-mn-core*
*device_class: Compiler-Scheduled SIMD Accelerator (Japan)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch — the ONE supported framework; MLSDK ships a CUSTOM-BUILT torch (2.9.0 in SDK 0.7 + torchvision 0.24.0); official PyTorch wheels do not work, no user venv in the same environment | confirmed | mncore-github, mlsdk-getting-started |
| Framework Integration | Programming model: compile a pure `Callable[[Dict[str,Tensor]], Dict[str,Tensor]]`; NO eager device; forward + backward + optimizer step fused into ONE device program | confirmed | mlsdk-technical-notes, mncore2-whitepaper |
| Framework Integration | Device-string portability over identical source: `pfvm:cpu` → `pfvm:cuda` → `emu2` → `mncore2:auto`; four-way debug matrix (PyTorch vs Compiled graph) × (LibTorch vs codegen kernels) | confirmed | mlsdk-technical-notes, sdk-hub-getting-started |
| Framework Integration | pytorch-pfn-extras 0.9.0 (Apache-2.0) — PFN OSS training-loop library, pulled in as a dependency | confirmed | github-ppe |
| Framework Integration | HuggingFace transformers used by PFN's own LLM examples (Llama / Qwen2 / PLaMo pipeline-parallel, SLM SFT) | confirmed | mncore-github |
| Framework Integration | Static shapes ONLY — "we currently do not support graphs that have Dynamic Shapes in their input/output structures" | confirmed | mlsdk-technical-notes |
| Framework Integration | Model coverage (timm survey, SDK v0.4): 378 compilable / 369 inference-ready / 156 training-ready. LLM inference labelled EXPERIMENTAL | confirmed | sdk-hub-models |
| Framework Integration | No vLLM, no TensorRT-LLM analogue, no ONNX Runtime EP | confirmed | sdk-hub-models, mncore-github |
| Framework Integration | JAX support claimed on PFN marketing pages but absent from all SDK documentation, the GitHub repo and PFCP docs — treat as unsubstantiated marketing | unconfirmed | pfn-business-chips, mlsdk-index |
| Graph Capture | FX2ONNX Exporter (`/opt/pfn/pfcomp/fx2onnx`): torch.fx symbolic tracing with FakeTensors → Exported ONNX; documented limitation on control flow branching on tensor presence | confirmed | mlsdk-technical-notes |
| Graph Capture | `fx2onnx.linter.lint` / `LintLevel` / `LintResult` (SDK v0.6+): pre-flights a model for exportability and reports actionable issues | confirmed | mlsdk-api-reference |
| Graph Capture | Legacy `torch.onnx` path via `MNCORE_USE_LEGACY_ONNX_EXPORTER=1` — deprecated, still used by some shipped examples | confirmed | mlsdk-technical-notes, mncore-github |
| Compiler / IR | IR spine is **ONNX all the way down**: Exported ONNX → Compiled ONNX → MNGraph. Not MLIR — PFN standardized on ONNX in 2021 | confirmed | mlsdk-technical-notes |
| Compiler / IR | **PFVM** (`/opt/pfn/pfcomp/pfvm`) — both compiler AND runtime: const-prop, CSE, fusion, backend-specialized op replacement, injects weight-update handling as custom ONNX ops → Compiled ONNX. PFVM Runtime executes it in LibTorch (= `pfvm:cpu` / `pfvm:cuda`) | confirmed | mlsdk-technical-notes, pfn-blog-pfvm-onnx |
| Compiler / IR | **codegen Graph Compiler** (`/opt/pfn/pfcomp/codegen`, historic name L3IR) — Compiled ONNX → MNGraph (MNNode + MNValue); dumps `l3ir.txt`, `l3ir_stripped.onnx` | confirmed | mlsdk-technical-notes |
| Compiler / IR | **MNValue three-property model = the MN-Core programming model**: Dtype (mixed precision graph-wide) / Location (DRAM or LM0/LM1 ONLY) / Layout (mapping across the physical memory tree) | confirmed | mlsdk-technical-notes |
| Compiler / IR | Layout notation over levels `{PE, W, Addr, MAB, L1B, L2B, Time}`, e.g. `(64,128)/((8_L2B:1, 8:2), (16_MAB:1, 2:1, 4_PE:1); B@[L1B,W])` | confirmed | mlsdk-technical-notes, pfn-blog-layout |
| Compiler / IR | Pass order: Dtype Planner → Location Planner (initial) → Layout Planner (`MNCoreLayoutSwitch`; `lpz` recommended) → Location Planner (`MNCoreUpload`/`MNCoreDownload`) → Time-Slice (`Time` subaxis at 2048 LW; `MNCoreInputSplit`/`MNCoreOutputConcat`) → Location Planner (sliced) → Node Simulation → Scheduler → Address Planner → Gene propagation | confirmed | mlsdk-technical-notes |
| Compiler / IR | **Node Simulation** actually compiles each MNNode under multiple configurations to MEASURE cycle counts; modes `fake`/`default`/`fast`/`best`/`full`; cached; dumped to `simulation_result.json` | confirmed | mlsdk-technical-notes, mlsdk-advanced-features |
| Compiler / IR | **Scheduler** owns DRAM spill/refill, LM0↔LM1 moves, `Forget` ops and RECOMPUTATION: `always_from_dram` / `reuse_consecutive` (default) / `spill_opt` / `auto_recompute_sa` (simulated annealing) | confirmed | mlsdk-technical-notes, pfn-blog-recompute |
| Compiler / IR | Presets O0–O4 + `debug.json` at `/opt/pfn/pfcomp/codegen/preset_options/`; L1Merge enabled at O3; autotuners `CODEGEN_FIND_BEST_COMPILE_OPTIONS` / `..._BETTER_...`; ~35 documented `CODEGEN_*` env vars including `CODEGEN_AUTO_RECOMPUTE_HACK_FOR_QWEN` | confirmed | mlsdk-advanced-features |
| Compiler / IR | Python compile options: `option_json`, `float_dtype`, `layout_planner`, `scheduler`, `simulation_mode`, `sram_budget`, `sa_expected_run_iters`, `out_onnx`, `layout_spec`, `gemm_layout_spec`, `num_threads` | confirmed | mlsdk-api-reference |
| Kernel Compiler | **codegen Code Emitter**: MNGraph → GPFNApp; per-MNNode compile then link. Concat strategy (addresses already globally consistent) or L1Merge (merge disjoint-resource sequences; blocked by in-place I/O → `MNCoreReorderAddress`) | confirmed | mlsdk-technical-notes |
| Kernel Compiler | Operator implementations in `codegen/layers` are **NOT shipped as source** — users cannot add an unimplemented operator; PFN says the extension framework is "being prepared", no timeline | confirmed | mlsdk-technical-notes |
| Kernel Language | MN-Core 2 assembly (VSM) — **fully public ISA**: ~130-page Software Developer Manual (EN/JA), rev. 2026-06-02; ~24 MV transfer modes with documented LW/cycle; PE operand namespaces, hazard rules, MAU/ALU/L1BM/L2BM instruction expressions | confirmed | mncore2-sdm |
| Kernel Language | `assemble3` (assembler) + `gpfn3_package_main` (cycle-faithful emulator) — distributed FREE with no login as `mncore2_emuenv_20240826.tar.xz`, also bundled in the SDK since v0.6. Stripped x86-64 ELF binaries under a bare "AS IS" disclaimer | confirmed | mncore2-emuenv, mlsdk-technical-notes |
| Kernel Language | **HPCSDK / MNCL** (`/opt/pfn/pfcomp/mncl`) — OpenCL-like C/C++ environment. **ALPHA, no public documentation** ("refer to the header files"). OpenACC subset promised in the 2023 whitepaper, still undocumented in SDK 0.7 | confirmed | mlsdk-getting-started, mncore2-whitepaper |
| Kernel Language | `pfnet/mncore_simple_graph_compiler_for_education` — publicly readable from-scratch teaching compiler (torch.fx → ONNX → VSM, trains MNIST on the emulator), ~40 hand-written `.vsm` tests. **NO LICENSE FILE — not open-source-licensed** | confirmed | github-edu-compiler, pfn-blog-scratch |
| Kernel Language | MN-Core Challenge — public MN-Core 2 assembly optimization contest (2024) with judge, tips and SDM errata | confirmed | mncore-challenge |
| Tensor API | `mlsdk` Python package: `MNDevice`, `Context` (compile / compile_automap / load_codegen_dir / register_param / register_buffer / register_optimizer_buffers / switch_context / synchronize), `CompiledFunction`, `TensorProxy` (`.cpu()`, `.load_from()`), `CacheOptions` | confirmed | mlsdk-api-reference |
| Tensor API | Device-side fused optimizers compiled into the same program as fwd+bwd: `MNCoreSGD`, `MNCoreAdam`, `MNCoreAdamW`, `MNCoreLRScheduler`. PFN warns `MNCoreAdamW` is NOT bit-compatible with LibTorch | confirmed | mlsdk-api-reference |
| Tensor API | Profiling: `trace_event()` / `trace_scope()` → `trace.json`, viewable in Perfetto UI / Chrome Tracing | confirmed | mlsdk-api-reference |
| Model Binary | **GPFNApp** — FlatBuffers container: assembled VSM (a.k.a. GPFNBin), I/O metadata (name, Dtype, Layout, Address), model parameters, relocation info (address fields left blank, relocated against the live Context). Inspectable with `dump_gpfnapp` | confirmed | mlsdk-technical-notes |
| Runtime | codegen runtime loads and executes GPFNApp; explicitly designed around GPFNApp REUSE ("suitable for workloads that repeatedly execute the same computations"); stated weakness: "performs poorly with operations requiring extensive indexing" | confirmed | mlsdk-technical-notes |
| Runtime | All host↔device traffic funnels through **Group 0's PDM** over PCIe; host-side transposition into the device Layout happens on the CPU during upload | confirmed | mlsdk-technical-notes, mncore2-sdm |
| Runtime | Runtime details from FAQ: pinned-memory "IDMA chunk" allocation; exclusive device lock (one program per board, `gpfn3-smi reset` breaks it); `CAP_SYS_NICE` required for CPU affinity (`docker run --cap-add=SYS_NICE`) | confirmed | mlsdk-faq |
| Runtime | Observability: `codegen_dir` collects `report.json`, `out.txt/json`, `model.onnx`, `model.app`, `model.vsm`, `layout.*`, `l3ir.txt`, `simulation_result.json`, `trace.json`; shipped **Codegen Dashboard** with Netron graph view, per-node view, Perfetto/Chrome Tracing | confirmed | mlsdk-advanced-features |
| Communication | **No collectives library** — no NCCL/RCCL/CNCL analogue, no device-side collectives. Multi-board uses OpenMPI + `torch.distributed` on the **gloo** backend with host-memory send/recv (`run_pp_llama.py`, `run_llm_infer_pp.sh`) | confirmed | mncore-github, mlsdk-technical-notes |
| Communication | Whether a device-side collective library is planned: **not disclosed** | unconfirmed | — |
| Driver / Firmware | APT repo `mncore-packages` @ `https://asia-northeast1-apt.pkg.dev/projects/mncore-packages` (Google Artifact Registry), added via `apt/add_mncore_packages.sh` | confirmed | mncore-github |
| Driver / Firmware | `gpfn3-dkms` — DKMS kernel module `gpfn3`; PCIe vendor `0ccd`; creates `/dev/mnc2p<bus>s<n>`; applies clock/MAB config at load time (volatile across power cycles); Secure Boot via MOK on Ubuntu 24.04, must be disabled on 22.04 | confirmed | mncore-install-manual, mncore-github |
| Driver / Firmware | `libgpfn3-0` / `libgpfn3-dev` — user-space driver + headers; `gpfn3-loader`; `mncore-sdk` (pinned via `apt-mark hold`) | confirmed | mncore-github |
| Driver / Firmware | `gpfn3-smi` — the `nvidia-smi` analogue: `list`, `reset`, `clear`, `config <dev> clock --core 750 --gddr6 15000`, `config <dev> mab`, `mtest` (~1-min self-test) | confirmed | mncore-install-manual, mlsdk-advanced-features |
| Driver / Firmware | Host OS: **Ubuntu 24.04 LTS and 22.04 LTS only** (latest LTS + one previous); container engine required; images `mncore-sdk-minimal` / `mncore-sdk-full`; `create_dev_ctr.sh` | confirmed | mncore-github, mlsdk-getting-started |
| Cloud Platform | **PFCP** (Preferred Computing Platform), operated by PFCI (PFN / Mitsubishi Corporation / IIJ JV, 2024-12-23): multi-tenant managed Kubernetes; MN-Core 2 as extended resource `preferred.jp/mncore2`; Reserved and Shared nodes; per-accelerator cap 7000m CPU / 125Gi memory; CRDs `ReservedNode`, `ClusterWorkspacePreset`, `ParallelJob`, `SealedSecret`. **Pricing not published** | confirmed | pfcp-docs, pr-20241223 |
| Cloud Platform | MN-Core Playground — live LLM fine-tuning service on PFCP, open to anyone | confirmed | playground |
| ISA | VLIW PE instruction (fixed width, always carries a `wait` field); 4-cycle "step" chosen because 1-cycle issue would saturate host PCIe; auto-stride (3 PE + 1 MV) and flat (2 PE + 1 MV) packing; no branches; no hazard interlocks; register-memory not load-store; untyped operands; big-endian; explicit forwarding operands | confirmed | mncore2-sdm |
| ISA | Number of ONNX operators supported by codegen, and the authoritative supported-op list: **not disclosed** | unconfirmed | — |
| Licensing | Everything load-bearing (PFVM, codegen, runtime, operator library, assembler, emulator, user-space driver, kernel module) ships as **binary .deb packages under an EULA** (`/opt/pfn/licenses/MN-Core_SDK_End-User_License_Agreement.pdf`, not published on the web). `github.com/pfnet/mncore` is Apache-2.0 but contains only Dockerfiles, apt scripts and examples | confirmed | mncore-github, mlsdk-getting-started |
| SDK Cadence | v0.4 (2026-02-27) → v0.5 (2026-04-28, "SDK source now public" — misleading: Dockerfiles+examples only) → v0.6 (2026-06-05, emulator+assembler bundled, `fx2onnx.linter`) → v0.7 (2026-07-15) → **v0.8 (2026-08-26 — no changelog published; specific changes not disclosed)**. SDK Hub portal launched 2026-06-22. The public no-login developer story is ~6 months old | confirmed | sdk-hub-news |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Design Philosophy | No hardware cache at any tier; no per-PE program counter; no per-PE instruction decoder; no hardware instruction scheduler; no hazard interlocks; no branches. Cache controller, command scheduler and network controller are all software | confirmed | mncore2-whitepaper, mncore2-sdm, sdk-hub-architecture |
| Design Philosophy | Instruction sequences are generated by the HOST CPU and streamed to the board in a single stream — the host is part of the run-time execution model, not just an offline compiler | confirmed | mncore2-whitepaper, mncore2-sdm |
| Design Philosophy | 7.4% of MN-Core 2 transistors are in arithmetic units vs PFN's anonymized measurements of 1.7% ("GPU N"), 2.4% ("Accelerator P"), 1.3%/0.8% (two CPUs) | vendor claim (confirmed as a claim) | mncore2-whitepaper |
| Design Philosophy | Architectural lineage per Makino: "similar to large-scale SIMD machines like TMC CM-2 and MasPar MP-2"; register-memory architecture, not load-store | confirmed | hc36-slides |
| Compute Engine | MN-Core 2: TSMC N7, 550 mm² single die, 22 B transistors, 750 MHz, 4096 PEs, 1024 MABs/MAUs, 330 W design value | confirmed | mncore2-whitepaper |
| Compute Engine | Peaks: 12 TF FP64 / 49 TF FP32 / 98 TF pseudo-single / 393 TF half. PFN's "TF32"/"TF16" labels are PFN block-FP formats — NOT NVIDIA TF32, NOT IEEE binary16 | confirmed | mncore2-whitepaper, mncore2-sdm |
| Compute Engine | Efficiency 37.24 / 148.9 / 297.9 / **1192** GFLOPS/W (FP64 / FP32 / pseudo-single / half). ⚠️ 1192 is the HALF figure only | confirmed | mncore2-whitepaper |
| Compute Engine | MAB = 4 PEs + 1 MAU; MAU has a double-buffered matrix register. Matrix–vector shapes per cycle: double m=2,n=4; single m=8,n=4; pseudo-single m=8,n=8; half m=16,n=16 | confirmed | mncore2-sdm |
| Compute Engine | Per PE: ALU (integer + special ops incl. ReLU) + mask registers. Reduction units in PE→L1BM→L2BM uplinks are explicitly EXCLUDED from peak FLOPS | confirmed | mncore2-sdm |
| Compute Engine | Integer (INT8/INT4) matrix datapath in the MAU: **not disclosed** — SDM documents only FP/BFP matrix modes; compiler integer dtypes are ALU-class only | unconfirmed | mncore2-sdm |
| Number Formats | NOT IEEE-754. Half = 1-6-9 (not binary16's 1-5-10); single 1-8-23; double 1-11-52. **No denormals, no NaNs** — exp all-zero ⇒ ±0, all-ones ⇒ ±∞ regardless of mantissa | confirmed | mncore2-sdm, mlsdk-hw-spec |
| Number Formats | Block floating point (same field widths, NO hidden bit) mandatory for matrix–vector operands; on MN-Core 2 used at ALL precisions (gen 1: half only). Assembly untyped; storage big-endian | confirmed | mncore2-sdm |
| Data Path | Two instruction classes: MV (upper-memory movement, asynchronous) and PE (VLIW, controls L2B and below). PE instruction covers a 4-cycle "step" **because 1-cycle issue would saturate host PCIe** | confirmed | mncore2-sdm |
| Data Path | Packing modes: auto-stride (3 PE + 1 MV, auto-incrementing addresses) and flat (2 PE + 1 MV, per-cycle addresses). Hardware requires packed streams; the emulator does not | confirmed | mncore2-sdm |
| Data Path | No hazard interlocks ("it is the programmer's responsibility"); no branches (mask-register predication); `wait` field in every PE instruction; architecturally visible forwarding operands `mauf`/`aluf`/`lbf`/`mreadf`/`nowrite` | confirmed | mncore2-sdm |
| Data Path | Determinism: "Without cache interference or speculative behavior, execution is fully deterministic" | vendor claim | sdk-hub-architecture |
| On-chip Memory | Tree: Top → 4 Groups → 8 L2Bs → 64 L1Bs → 1024 MABs → 4096 PEs. **No cache at any level**; 100% software-managed scratchpad | confirmed | mlsdk-hw-spec, mncore2-sdm |
| On-chip Memory | Per-PE: GRF0/GRF1 256 LW (2 KiB) each → 16 MiB/chip; LM0/LM1 2 Ki LW (16 KiB) each → **128 MiB/chip**; T-register 8 LW → 256 KiB/chip | confirmed | mlsdk-hw-spec |
| On-chip Memory | L1BM 8 Ki LW (64 KiB) × 64 → 4 MiB; L2BM 32 Ki LW (256 KiB) × 8 → 2 MiB; PDM 4 MiB × 4 Groups → 16 MiB | confirmed | mlsdk-hw-spec, mncore2-sdm |
| On-chip Memory | Aggregate ≈ **166 MiB** — derived by summing per-level figures; **PFN publishes no aggregate**. Derived LM bandwidth ≈ 49 TB/s per bank / 98 TB/s across LM0+LM1 | likely (derived) | mlsdk-hw-spec |
| On-chip Memory | **Only Group 0's PDM is wired to the host over PCIe** — all host↔device I/O funnels through one 4 MiB SRAM | confirmed | mncore2-sdm, mlsdk-hw-spec |
| On-chip Memory | Compiler `Location` can only target DRAM or LM0/LM1; GRF0/GRF1 and L1BM/L2BM are never Locations | confirmed | mlsdk-technical-notes |
| On-chip Memory | L1BM / L2BM SRAM bandwidth and on-chip ECC: **not disclosed** | unconfirmed | — |
| On-chip Interconnect | Broadcast/reduction **tree** — no NoC, no mesh, no crossbar. L1B: 16 MABs + 1 L1BM; L2B: 8 L1Bs + 1 L2BM; modes = broadcast, individual, reduction, distribute/collect | confirmed | mncore2-whitepaper, mncore2-sdm |
| On-chip Interconnect | Symmetric by design: every downward mode (distribute) has a matching upward mode (collect) reassembling the identical layout — an explicit layout-consistency guarantee the Layout Planner depends on | confirmed | mncore2-sdm |
| On-chip Interconnect | Documented MV throughputs: DRAM↔L2BM parallel 16 LW/cycle/group (96 GB/s, derived); inter-group broadcast/reduce 8 LW/cycle; L2BM collect/distribute up to 32 LW/cycle; PDM↔L2BM 8/1; PDM↔DRAM 8/2 | confirmed | mncore2-sdm |
| Off-chip Memory | MN-Core 2: **16 GiB GDDR6, 512 GB/s** (4 GiB / ~128 GB/s per Group). PFN's own HC36 slides disagree — slide 10 "GDDR6", slide 13 "GDDR6X" | confirmed (type: likely) | hc36-slides, mncore2-sdm |
| Off-chip Memory | GDDR bus width, device count, per-pin data rate, DRAM ECC: **not disclosed** | unconfirmed | — |
| Off-chip Memory | MN-Core gen 1: 32 GB LPDDR4, 400 GB/s | confirmed | mncore-product-page |
| Host Interface / Package | PCIe Gen5 ×16 (gen 1: Gen3 ×16); PCI vendor ID `0ccd`; `/dev/mnc2p<bus>s<n>`; clock/MAB config volatile across power cycles and re-applied at module load | confirmed | hc36-slides, mncore-install-manual |
| Host Interface / Package | Card mechanical form factor, measured TDP, idle power, thermal limits, virtualization/partitioning: **not disclosed** | unconfirmed | — |
| Host Interface / Package | Cooling: air originally (manual advises fans ≥70% "to prevent MN-Core 2 thermal runaway"); direct-liquid-cooled high-density server developed 2025 | confirmed | mncore-install-manual, pr-20250911 |
| Scale-up Interconnect | **MN-Core 2: NONE.** No chip-to-chip fabric, no NVLink analogue, no device-side collectives. Multi-board goes through the host (OpenMPI + torch.distributed gloo) | confirmed | mncore-github, mncore2-whitepaper |
| Scale-up Interconnect | MN-Core gen 1: **MN-Core DirectConnect** + DirectConnect Switches; 32 servers + 2 switches = one "Zone". Bandwidth not disclosed | confirmed | pfn-supercomputers |
| Scale-up Interconnect | Whether DirectConnect was dropped for gen 2 or merely unused in MN-Server 2: **not disclosed** | unconfirmed | — |
| Scale-out Interconnect | Standard host networking only: MN-Server 2 V1 has 2 × ConnectX-6 100 GbE dual-port + 2 × Intel X710 10 GbE. No accelerator-side RDMA, no collective offload | confirmed | mncore2-hw-catalog |
| System Products | MN-Core 2 Devkit V1 (MNC2DV1): 1 board, Core i5-14500, 64 GB DDR5, ATX 850 W — **¥2,000,000 ex-tax** | confirmed | mncore2-hw-catalog |
| System Products | MN-Server 2 V1 (MNS2V1): 5U, 8 boards, 2 × Xeon Platinum 8480+, 1 TB DDR5-4800, 4 × 3000 W PSU, max 4200 W — **¥20,000,000 ex-tax** | confirmed | mncore2-hw-catalog |
| System Products | MN-Pod 2: 6 × MN-Server 2 in a 42U rack = 48 boards; 590 TF FP64 / 2359 FP32 / 4719 pseudo-single / **18,874 TF half (18.9 PF)** | confirmed | mncore2-whitepaper |
| Generation 1 | MN-Core: TSMC 12 nm, 4 dies × 764 mm², 2048 MABs (8192 PEs), 500 MHz, PCIe Gen3 ×16, 500 W (estimated), 32.8 TF DP / 131 TF SP / 524 TF HP (532 TF FP16 on the HC36 slide — unexplained discrepancy) | confirmed | mncore-product-page, hc36-slides |
| Generation 1 | MN-3 supercomputer (gen-1 silicon) Green500: #1 Jun 2020 (21.11), #2 Nov 2020 (26.04), **#1 Jun 2021 (29.70, independently confirmed by TOP500)**, #1 Nov 2021 (39.38), #5 Jun 2022 (40.90). **No Green500 result exists for MN-Core 2** | confirmed | pfn-supercomputers, top500-green500-2021-06 |
| Roadmap | MN-Core L1000 = family/architecture name; SKUs **MN-Core L1100** (low-power) and **MN-Core L1400** (high-performance), both 2027 — a one-year slip from the 2026 target announced 2024-11-15. **PRE-SILICON, mockup only** | confirmed | pr-20260601, pr-20241115, pr-20251113 |
| Roadmap | L-series memory: **DRAM-on-logic 3D stacking** — "PFN's proprietary logic and DRAM are vertically stacked, with many electrodes arranged across the surface." Not HBM, not commodity DIMM DRAM | confirmed | pr-20260601 |
| Roadmap | L-series claims: "50× higher memory bandwidth than MN-Core 2" (implies ~25.6 TB/s, derived from a claim not a spec); single L1400 card sized for 70 B-parameter LLM inference; earlier and different-in-kind 2024 claim of "up to a ten-fold increase in computing speed" | vendor claim, pre-silicon | pr-20260601, pr-20241115 |
| Roadmap | L1100/L1400 process node, foundry, die size, stack technology, memory capacity, absolute bandwidth, peak FLOPS, dtypes, TDP, form factor, price, interconnect: **all not disclosed** | unconfirmed | — |
| Roadmap | **MN-Core L2000** (2028, large-scale AI inference + HPC) appears ONLY on PFN's roadmap graphic and in no press release — roadmap tile only, every spec not disclosed | confirmed (as a tile) | pfn-roadmap-image |
| Roadmap | Two unreconciled process statements for the post-MN-Core-2 part: HC36 2024-08-27 "MN-Core Next for learning … **Samsung SF2**" vs the 2025-01-08 PFN/**Rapidus**/SAKURA agreement to manufacture a new MN-Core-series chip. PFN has never reconciled them | confirmed (both as dated statements) | hc36-slides, pr-20250108 |
| Deployment | NEDO testbed since Jul 2025: **30 nodes = 240 boards** at IIJ Matsue DCP + **2 nodes = 16 boards** at JAIST Ishikawa, direct liquid cooling. Moved into IIJ Shiroi DCC **AImod** from Apr 2026 — board count there **not disclosed** | confirmed | pr-20250911, pr-20260323 |
| Deployment | Gen-1 production applications: Matlantis (sold to ENEOS as a commercial cloud service since Aug 2023, ~3–5× vs GPU), PFN 3D Scan (~10×), Kachaka home-robot NAS (~7×). MN-Core 2 used in Matlantis only experimentally, via PFCP | confirmed | pr-20231016, pfn-business-chips |
| Maturity | Gen 1: deployed in production, superseded. **MN-Core 2: shipping as complete Japanese-market systems with published list prices since ~Sept 2024 + PFCP cloud; no merchant bare-chip sales, no non-Japanese customers, no third-party OEM.** L1100/L1400: pre-silicon. L2000/next-gen: roadmap tiles | confirmed | mncore2-hw-catalog, hc36-slides, pfn-business-chips |
| Performance | HC36 vs A100 (PFN's own measurements, 2024-08-27): ResNet-50 train 77 TF FP16 vs 33.2; ResNet-50 infer 154 vs 33.7; **HIMENO 9.03 TF FP64 vs 0.634** (~75% of MN-Core 2's FP64 peak on a memory-bound stencil); GCN 5.41 vs 3.17 TF FP32; OpenFDTD 0.655 vs 0.488 TF | vendor claim | hc36-slides |
| Performance | LLM inference (SDK v0.6, PLaMo-3-NICT-2B-base): 5,100 tok/s prefill (512 tok), 11.8 tok/s decode | vendor claim | sdk-hub |
| Performance | **No MLPerf or other third-party-audited results exist for MN-Core 2.** The only independently verified performance datum in the line is the gen-1 Green500 ranking | confirmed | top500-green500-2021-06 |

## Source Keys

| Key | URL |
|---|---|
| mncore2-whitepaper | https://projects.preferred.jp/mn-core/assets/MN-Core_2_whitepaper_en.pdf |
| mncore2-sdm | https://projects.preferred.jp/mn-core/assets/mncore2_dev_manual_en.pdf |
| hc36-slides | https://hc2024.hotchips.org/assets/program/conference/day2/15_HC2024.Preferred.Makino.final.pdf |
| mlsdk-technical-notes | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/technical_notes.html |
| mlsdk-hw-spec | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/hardware_specification.html |
| mlsdk-advanced-features | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/advanced_features.html |
| mlsdk-api-reference | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/api_reference.html |
| mlsdk-faq | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/faq.html |
| mlsdk-getting-started | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/getting_started.html |
| mlsdk-index | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/index.html |
| sdk-hub | https://dev.mn-core.com/ |
| sdk-hub-architecture | https://dev.mn-core.com/architecture/ |
| sdk-hub-models | https://dev.mn-core.com/models/ |
| sdk-hub-getting-started | https://dev.mn-core.com/getting-started/ |
| sdk-hub-news | https://dev.mn-core.com/news/en/ |
| mncore-github | https://github.com/pfnet/mncore |
| github-edu-compiler | https://github.com/pfnet/mncore_simple_graph_compiler_for_education |
| github-ppe | https://github.com/pfnet/pytorch-pfn-extras |
| mncore2-emuenv | https://projects.preferred.jp/mn-core/assets/mncore2_emuenv_20240826.tar.xz |
| mncore2-hw-catalog | https://projects.preferred.jp/mn-core/assets/MN-Core2-hardware-catalog.pdf |
| mncore-install-manual | https://projects.preferred.jp/mn-core/assets/MN-Core2-Devkit-MN-Server-2-installation-operation-manual.pdf |
| mncore-product-page | https://projects.preferred.jp/mn-core/en/ |
| pfn-supercomputers | https://projects.preferred.jp/supercomputers/ |
| pfn-business-chips | https://www.preferred.jp/en/business/chips/ |
| pfn-roadmap-image | https://www.preferred.jp/images/business/chips/roadmap_img.png |
| pfcp-docs | https://docs.pfcomputing.com/en/ |
| playground | https://playground.mn-core.com/ |
| mncore-challenge | https://mncore-challenge.preferred.jp/ |
| pfn-blog-pfvm-onnx | https://tech.preferred.jp/ja/blog/pfvm-onnx-exporter/ |
| pfn-blog-layout | https://tech.preferred.jp/ja/blog/mn-core-tensor-layout/ |
| pfn-blog-recompute | https://tech.preferred.jp/ja/blog/mncore-compiler-optimization-with-recompute/ |
| pfn-blog-scratch | https://tech.preferred.jp/ja/blog/mn-core2_graphcompiler_scratch/ |
| top500-green500-2021-06 | https://www.top500.org/lists/green500/2021/06/ |
| pr-20231016 | https://www.preferred.jp/en/news/pr20231016 |
| pr-20241115 | https://www.preferred.jp/en/news/pr20241115 |
| pr-20241223 | https://www.preferred.jp/en/news/pr20241223 |
| pr-20250108 | https://www.preferred.jp/en/news/pr20250108 |
| pr-20250911 | https://www.preferred.jp/en/news/pr20250911 |
| pr-20251113 | https://www.preferred.jp/en/news/pr20251113 |
| pr-20260323 | https://www.preferred.jp/en/news/pr20260323 |
| pr-20260601 | https://www.preferred.jp/en/news/pr20260601 |
