# Preferred Networks MN-Core — Search Results

*as_of: 2026-08-08*
*chip: preferred-networks-mn-core*
*device_class: Compiler-Scheduled SIMD Accelerator (Japan)*

---

## Search Queries

1. "MN-Core 2 white paper Preferred Networks specifications"
2. "MN-Core 2 Software Developer Manual PDF preferred networks"
3. "MN-Core 2 hardware specification memory hierarchy L2BM L1BM GRF LM capacities"
4. "dev.mn-core.com MLSDK documentation technical notes pipeline"
5. "MN-Core MLSDK PFVM codegen graph compiler MNGraph MNNode MNValue"
6. "MN-Core HPCSDK MNCL OpenCL OpenACC alpha"
7. "github pfnet mncore apt packages gpfn3-dkms libgpfn3 gpfn3-smi"
8. "MN-Core 2 Devkit MN-Server 2 installation operation manual gpfn3-smi clock gddr6"
9. "MN-Core 2 hardware catalog MN-Server 2 V1 MNS2V1 price"
10. "MN-Core 2 Preferred Networks compiler arxiv"
11. "Hot Chips 2024 Preferred Networks Makino MN-Core 2 slides PDF"
12. "MN-Core graph compiler tutorial guide preferred networks"
13. "pfnet mncore_simple_graph_compiler_for_education"
14. "tech.preferred.jp mncore compiler blog recompute tensor layout"
15. "MN-Core L1000 L1100 L1400 3D stacked DRAM 2027 Toyota"
16. "MN-Core L2000 roadmap 2028"
17. "PFN MN-Core roadmap image business chips"
18. "PFCP Preferred Computing Platform MN-Core 2 Kubernetes preferred.jp/mncore2"
19. "docs.pfcomputing.com MN-Core PyTorch tutorial workspace"
20. "MN-3 supercomputer Green500 MN-Core DirectConnect interconnect"
21. "top500 green500 2021 June MN-3 Preferred Networks 29.70 GFlops/W"
22. "ServeTheHome Preferred Networks MN-Core 2 for HPC and AI"
23. "PFN IIJ JAIST NEDO liquid cooling MN-Core 2 240 boards Matsue"
24. "PFN Rapidus SAKURA internet MN-Core next generation"
25. "PFN Mitsubishi IIJ Preferred Computing Infrastructure JV PFCP"
26. "Matlantis MN-Core ENEOS PFP commercial cloud service"
27. "MN-Core Challenge assembly contest preferred networks"
28. "MN-Core Technology Conference 25 speakerdeck"
29. "PLaMo 3 NICT 2B MN-Core 2 tokens per second"
30. "Interop Tokyo 2026 ShowNet MN-Core 2 water cooled PLaMo"

---

## Resources Found

### Vendor Primary — Architecture and ISA

| Resource | URL | Category |
|----------|-----|----------|
| MN-Core 2 White Paper (2023-11-12) — full spec table, PE/MAB/L1B/L2B hierarchy, MN-Server 2 / MN-Pod 2, software stack | https://projects.preferred.jp/mn-core/assets/MN-Core_2_whitepaper_en.pdf | Hardware Spec |
| **MN-Core 2 Software Developer Manual (EN), rev. 2026-06-02** — complete ISA: memory sizes/bandwidths, VLIW format, MV/PE instructions, MAU/ALU expressions, FP formats, hazards | https://projects.preferred.jp/mn-core/assets/mncore2_dev_manual_en.pdf | ISA Manual |
| MN-Core 2 Software Developer Manual (JA) | https://projects.preferred.jp/mn-core/assets/mncore2_dev_manual_ja.pdf | ISA Manual |
| MN-Core Series product page (EN) — gen-1 specs, MN-Core 2 spec table, Green500 table | https://projects.preferred.jp/mn-core/en/ | Hardware Spec |
| MN-Core Series product page (JA) | https://projects.preferred.jp/mn-core/ | Hardware Spec |
| PFN AI Chips business page (EN) — gen1/gen2 specs, product list + prices, roadmap, design philosophy | https://www.preferred.jp/en/business/chips/ | Overview |
| PFN AI Chips business page (JA) | https://www.preferred.jp/ja/business/chips/ | Overview |
| PFN MN-Core roadmap graphic — MN-Core / MN-Core 2 / L1000 (2027) / L2000 (2028) / next-gen | https://www.preferred.jp/images/business/chips/roadmap_img.png | Roadmap |
| PFN Supercomputers page — MN-3 Green500 table, MN-Core DirectConnect, MN-3a node config | https://projects.preferred.jp/supercomputers/ | Hardware Spec |

### Vendor Primary — Product / Operations

| Resource | URL | Category |
|----------|-----|----------|
| MN-Core 2 hardware catalog (2024-08) — MN-Server 2 V1 (MNS2V1) and Devkit (MNC2DV1) BOM, dimensions, PSU, Japanese list prices | https://projects.preferred.jp/mn-core/assets/MN-Core2-hardware-catalog.pdf | Product Catalog |
| MN-Core 2 Devkit / MN-Server 2 installation & operation manual (2026-06-09, JA) — driver install, `gpfn3-smi`, `lspci` vendor `0ccd`, `/dev` nodes, Secure Boot/MOK, fan policy | https://projects.preferred.jp/mn-core/assets/MN-Core2-Devkit-MN-Server-2-installation-operation-manual.pdf | Ops Manual |
| MN-Core Emulator Environment tarball (`mncore2_emuenv_20240826.tar.xz`) — `assemble3` + `gpfn3_package_main` + tutorial | https://projects.preferred.jp/mn-core/assets/mncore2_emuenv_20240826.tar.xz | Toolchain |

### Vendor Primary — SDK Hub and MLSDK Documentation

| Resource | URL | Category |
|----------|-----|----------|
| MN-Core SDK Hub (public, no login) | https://dev.mn-core.com/ | SDK Docs |
| SDK Hub — Architecture (no cache, VLIW/SIMD, L2BM→L1BM→MAB→GRF, determinism claim) | https://dev.mn-core.com/architecture/ | Hardware Spec |
| SDK Hub — Workloads/Models (timm survey: 378 compilable / 369 inference / 156 training) | https://dev.mn-core.com/models/ | SDK Docs |
| SDK Hub — Getting Started (docker build, device strings) | https://dev.mn-core.com/getting-started/ | SDK Docs |
| SDK Hub — News (SDK v0.4–v0.7 release dates) | https://dev.mn-core.com/news/en/ | SDK Docs |
| MLSDK 0.7 documentation index (EN) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/index.html | SDK Docs |
| **MLSDK 0.7 — Technical Notes** (MLSDK pipeline, PFVM, codegen Graph Compiler / Code Emitter, MNGraph, Dtype/Location/Layout, Time-Slice, Node Simulation, Scheduler, L1Merge, GPFNApp) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/technical_notes.html | Compiler Docs |
| MLSDK 0.7 — Hardware Specification (tree levels, capacities, LW/SW/HW, 1-6-9 half) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/hardware_specification.html | Hardware Spec |
| MLSDK 0.7 — Advanced Features (`gpfn3-smi`, presets O0–O4, ~35 `CODEGEN_*` env vars, Codegen Dashboard) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/advanced_features.html | Compiler Docs |
| MLSDK 0.7 — API Reference (MNDevice, Context, CompiledFunction, TensorProxy, MNCoreSGD/Adam/AdamW, fx2onnx.linter) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/api_reference.html | Runtime Docs |
| MLSDK 0.7 — FAQ (device/runtime errors, CAP_SYS_NICE, ONNX external data) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/faq.html | SDK Docs |
| MLSDK 0.7 — Getting Started (`/opt/pfn/pfcomp/{fx2onnx,pfvm,mncl,codegen}`) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/getting_started.html | SDK Docs |
| MLSDK 0.7 — Porting Tutorial | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/en/porting_tutorial.html | SDK Docs |
| MLSDK 0.7 documentation (JA) | https://dev.mn-core.com/sdk/0.7/MLSDK/docs/ja/index.html | SDK Docs |

### Open Source / Publicly Readable Code

| Resource | URL | Category |
|----------|-----|----------|
| `pfnet/mncore` — Apache-2.0; apt repo script, Dockerfiles for SDK 0.4–0.7, MLSDK examples (pipeline-parallel Llama over MPI/gloo, Stable Diffusion, SLM SFT, DETR, SSD, NCF, timm) | https://github.com/pfnet/mncore | Examples/Build only |
| `pfnet/mncore_simple_graph_compiler_for_education` — from-scratch educational graph compiler (torch.fx → ONNX → MN-Core 2 VSM); ~40 hand-written `.vsm` tests. **No LICENSE file** | https://github.com/pfnet/mncore_simple_graph_compiler_for_education | Public source (unlicensed) |
| `pfnet/pytorch-pfn-extras` — PFN OSS training-loop library, an MLSDK dependency | https://github.com/pfnet/pytorch-pfn-extras | Open source (Apache-2.0) |

### Conference / Academic

| Resource | URL | Category |
|----------|-----|----------|
| Hot Chips 36 slides (2024-08-27), J. Makino, "MN-Core 2: Second-generation processor of MN-Core architecture for AI and general-purpose HPC application" | https://hc2024.hotchips.org/assets/program/conference/day2/15_HC2024.Preferred.Makino.final.pdf | Conference Slides |
| IEEE Xplore record for the Hot Chips 2024 MN-Core 2 talk | https://ieeexplore.ieee.org/document/10664802 | Academic Index |
| IEEE Xplore, "Development of AI Accelerators at Preferred Networks" (paywalled, not retrieved) | https://ieeexplore.ieee.org/document/11046565 | Academic Index |

### Vendor Blogs / Community

| Resource | URL | Category |
|----------|-----|----------|
| PFN tech blog (EN, 2021-06-11) — "Accelerating Deep Learning Workloads with the MN-Core Compiler" (L3IR / Layer / Generic Conv) | https://tech.preferred.jp/en/blog/mncore-compiler-1/ | Compiler Docs |
| PFN tech blog (JA) — MN-Core compiler optimization with recomputation | https://tech.preferred.jp/ja/blog/mncore-compiler-optimization-with-recompute/ | Compiler Docs |
| PFN tech blog (JA) — tensor memory-placement Layout representation | https://tech.preferred.jp/ja/blog/mn-core-tensor-layout/ | Compiler Docs |
| PFN tech blog (JA) — developing the ONNX exporter for PFVM | https://tech.preferred.jp/ja/blog/pfvm-onnx-exporter/ | Compiler Docs |
| PFN tech blog (JA, 2025-12) — "build an MN-Core graph compiler yourself and train MNIST" | https://tech.preferred.jp/ja/blog/mn-core2_graphcompiler_scratch/ | Compiler Docs |
| Slides for the above | https://speakerdeck.com/pfn/202512_mncore-graph-compiler-mnist | Slides |
| MN-Core Challenge — public MN-Core 2 assembly optimization contest (2024): rules, problems, tips, SDM errata | https://mncore-challenge.preferred.jp/ | Community |
| MN-Core Challenge — MNIST special problem set | https://mncore-challenge.preferred.jp/mnist/ | Community |
| MN-Core Playground — live LLM fine-tuning on PFCP | https://playground.mn-core.com/ | Service |

### Cloud Platform

| Resource | URL | Category |
|----------|-----|----------|
| PFCP User Guide — Kubernetes resource `preferred.jp/mncore2`, reserved vs shared nodes, MLSDK tutorial, changelog | https://docs.pfcomputing.com/en/ | Cloud Docs |
| PFCP marketing site (PFCI-operated) | https://pfcomputing.com/ | Cloud |
| Preferred Computing Infrastructure, Inc. (PFCI) | https://www.pfci.jp/ | Operator Entity |

### Press Releases

| Resource | URL | Category |
|----------|-----|----------|
| PR 2023-10-16 — MN-Core powers Matlantis; commercial cloud service to ENEOS since Aug 2023 | https://www.preferred.jp/en/news/pr20231016 | Deployment |
| PR 2024-08-23 — MN-Core 2 accepted to Hot Chips 2024; "started operating in 2023" | https://www.preferred.jp/en/news/pr20240823 | Announcement |
| PR 2024-11-15 — "PFN Begins Development of Generative AI Processor MN-Core L1000" (2026 target — later slipped) | https://www.preferred.jp/en/news/pr20241115 | Roadmap |
| PR 2024-12-23 — PFN / Mitsubishi Corporation / IIJ establish Preferred Computing Infrastructure | https://www.preferred.jp/en/news/pr20241223 | Corporate |
| PR 2025-01-08 — PFN / Rapidus / SAKURA internet basic agreement | https://www.preferred.jp/en/news/pr20250108 | Manufacturing |
| PR 2025-09-11 — NEDO testbed: 30 nodes / 240 MN-Core 2 boards at IIJ Matsue DCP + 2 nodes / 16 boards at JAIST | https://www.preferred.jp/en/news/pr20250911 | Deployment |
| PR 2025-11-13 — SC25 booth: MN-Core L1000 **mockup**, MN-Core 2 Devkit demo, direct liquid cooling | https://www.preferred.jp/en/news/pr20251113 | Roadmap |
| PR 2026-03-06 — GMO Preferred Security JV (chip-level security review) | https://www.preferred.jp/en/news/pr20260306 | Corporate |
| PR 2026-03-23 — AImod modular liquid-cooled DC at IIJ Shiroi DCC, full operation April 2026 | https://www.preferred.jp/en/news/pr20260323 | Deployment |
| PR 2026-06-01 — Toyota Frontier Research Center joint research; MN-Core **L1100 / L1400**, 2027, mockup, 50× BW claim | https://www.preferred.jp/en/news/pr20260601 | Roadmap |
| PR 2026-06-09 (JA) — Interop Tokyo 2026 ShowNet: water-cooled MN-Core 2 running PLaMo 3.0 Prime β for MN-Core (8B) | https://www.preferred.jp/ja/news/pr20260609 | Deployment |
| PFN news index (EN) | https://www.preferred.jp/en/news/ | Index |

### Independent Coverage

| Resource | URL | Category |
|----------|-----|----------|
| TOP500 Green500 June 2021 release — independent confirmation MN-3 #1 at 29.70 GFlops/W | https://www.top500.org/lists/green500/2021/06/ | Independent |
| ServeTheHome, "Preferred Networks MN-Core 2 for HPC and AI" (2024-08-27) — live coverage of the Hot Chips talk | https://www.servethehome.com/preferred-networks-mn-core-2-for-hpc-and-ai/ | Independent Press |
| HuggingFace `pfnet/plamo-3-nict-2b-base` — model behind the published MN-Core 2 tok/s figures | https://huggingface.co/pfnet/plamo-3-nict-2b-base | Model Repo |

---

## Key Findings

- **Vendor**: Preferred Networks, Inc. (PFN), Tokyo, co-developed with Kobe University (Prof. Junichiro Makino, also PFN CTO of Computer Architecture). Development began 2016. Internal die codenames: `GPFN-2G01` (gen 1), **GPFN3** (MN-Core 2 — the SDK tools are literally `gpfn3-smi`, `assemble3`, `gpfn3_package_main`).
- **Defining architectural claim**: PEs have **no program counter and no instruction decoder**; there is **no hardware cache at any tier**; instruction sequences are generated by the host CPU and streamed over PCIe. Cache controller, command scheduler and network controller are all software functions. This is a stronger form of the Groq TSP thesis.
- **MN-Core 2 (shipping)**: TSMC N7, 550 mm², 22 B transistors, 750 MHz, 4096 PEs / 1024 MAUs, 12 TF FP64 / 49 TF FP32 / 98 TF pseudo-single / 393 TF half, 330 W design value, 16 GiB GDDR6 @ 512 GB/s, PCIe Gen5 ×16. **The famous 1192 GFLOPS/W is the half-precision (block-FP) number only** — FP64 efficiency is 37.24 GFLOPS/W.
- **Numerics are not IEEE-754**: half is 1-6-9 (not binary16's 1-5-10), there are **no denormals and no NaNs**, and matrix ops use block floating point at *all* precisions on MN-Core 2.
- **No chip-to-chip fabric on MN-Core 2** and **no collectives library**: multi-board runs go through the host with OpenMPI + `torch.distributed` on the **gloo** backend. Gen 1 had a proprietary "MN-Core DirectConnect" fabric; no gen-2 equivalent is documented.
- **Software stack is overwhelmingly proprietary**: PFVM, codegen, runtime, operator library, assembler, emulator, user-space driver and kernel module ship as binary `.deb` packages from a private Google Artifact Registry APT repo under an EULA. `github.com/pfnet/mncore` is Apache-2.0 but contains only Dockerfiles and examples.
- **Documentation openness is exceptional despite that**: a complete ~130-page public ISA manual, a free (login-free) assembler + cycle-faithful emulator, an educational from-scratch graph compiler, and a public assembly-optimization contest.
- **Maturity**: gen 1 deployed in production (MN-3 supercomputer, three Green500 #1 finishes; Matlantis commercial service). MN-Core 2 sold as complete Japanese-market systems with published list prices (MN-Server 2 ¥20 M, Devkit ¥2 M) plus PFCP cloud; ~256 boards confirmed in PFN/NEDO testbeds; no merchant bare-chip sales, no non-Japanese customers. **MN-Core L1100 / L1400 are pre-silicon, mockup only, 2027 target.** MN-Core L2000 is a roadmap tile.
- **Public developer story is ~6 months old**: the SDK Hub (dev.mn-core.com) launched 2026-06-22; before 2026 the SDK was effectively PFCP-internal.
- **No third-party-audited benchmarks exist for MN-Core 2** — MLPerf absent; all performance figures other than the gen-1 Green500 results are PFN's own.
