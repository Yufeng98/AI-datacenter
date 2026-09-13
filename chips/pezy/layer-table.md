# PEZY Computing Layer Mapping Table

*as_of: 2026-08-08*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | MPI (OpenMPI / Intel MPI) over InfiniBand for multi-node scientific HPC | confirmed | search-results, software-stack |
| Framework Integration | PyTorch backend for the PEZY-SC series **announced** 2025-06-06 (with Accelerate, DeepSpeed, Transformers, vLLM, TGI, Diffusers on PEZY-SC3); Gemma3 / Llama3 / Qwen2 / Stable Diffusion 2 / HuBERT / ViT reported running. **Vendor claim only** — no public repo, release tag, version number, benchmark, or docs located. SC4s support said to follow the SC4s release. *(Corrects the earlier "no ML framework backends" row, 2026-08-08.)* | vendor-claim (announced, unverifiable) | pezy.co.jp/en/news/news20250606-pezysc-pytorch-generativeai/ |
| Framework Integration | No JAX / TensorFlow / ONNX Runtime support announced | confirmed | software-stack, search-results |
| Framework Integration | Custom scientific codes: n-body, lattice QCD, genomics, fluid dynamics | confirmed | search-results |
| Framework Integration | Genomics/bioinformatics applications are PEZY's only publicly announced 2026 workloads: pzMutect2 on ZettaVEGA (139× vs GATK 4.2.6.1, vendor claim, 2026-05-11) and PZLAST-MAG public MAG search server (2026-05-13) — both on PEZY-SC3, neither on SC4s | vendor-claim | pezy.co.jp news 2026-05-11, 2026-05-13 |
| Compiler / IR | PZCL Compiler (proprietary; OpenCL-C kernel → PEZY PE binary; access-controlled) | confirmed | software-stack |
| Compiler / IR | Host compiler: GCC/Clang for AMD EPYC x86-64 host | confirmed | software-stack |
| Compiler / IR | PEZY PE ISA (proprietary; MIPS64-derived in SC2; SC3/SC4 undisclosed) | confirmed | software-stack, hw-architecture |
| Compiler / IR | No MLIR/LLVM public path (no community compiler infrastructure); re-checked 2026-08-08 — no PZCL/OpenCL PrivateUse1 backend, no package, no SDK version number found from any independent source | confirmed | software-stack, 2026-08-08 verification |
| Op Library | Not applicable — no public math/tensor library; scientists write custom PZCL kernels | confirmed | software-stack |
| Kernel Library | Not applicable — no CUTLASS/cuBLAS/MIOpen equivalent | confirmed | software-stack |
| Runtime | PZCL Runtime (OpenCL-analog; platform/device/context/queue/memory management) | confirmed | software-stack |
| Runtime | PZCL Scratchpad management (24 KB/PE + shared at village; SW-managed) | confirmed | hw-architecture, software-stack |
| Runtime | RISC-V Rocket Core Management Processor (SC4s; on-chip Linux; host-independent boot/PE management) | confirmed | hw-architecture, search-results |
| Driver | Proprietary Linux PCIe kernel module (ioctl via PZCL runtime) | confirmed | software-stack |
| Communication | InfiniBand (400 Gb/s NDR per node in ZettaScaler4.0) | confirmed | hw-architecture, search-results |
| Communication | MPI over InfiniBand (OpenMPI/Intel MPI; standard HPC collective) | confirmed | software-stack |
| Communication | PCIe peer-to-peer intra-node (no proprietary chip-to-chip fabric) | confirmed | hw-architecture |
| Assembler / ISA | Proprietary PEZY PE ISA binary (no user-visible virtual ISA; no PTX equivalent) | confirmed | software-stack |
| Programming Model | PZCL (PEZY Computing Language — OpenCL 1.2 base; access-controlled SDK) | confirmed | software-stack, search-results |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | 2,048 MIMD PEs (PEZY-SC4s) — fully independent instruction streams; no warp/wavefront grouping | confirmed | hw-architecture, search-results |
| Compute Engine | SMT8 per PE: two groups of 4 threads; fine-grained within group (different thread every cycle); coarse-grained between groups (swap on long-latency op) | confirmed | hw-architecture, search-results |
| Compute Engine | FP64: 4-wide (256-bit) SIMD; FP32/FP16: supported; BF16: new in SC4s; INT8: supported | confirmed | hw-architecture, search-results |
| Compute Engine | No warp scheduler / no branch predictor / no out-of-order execution (in-order MIMD PEs) | confirmed | hw-architecture |
| Compute Engine | Total 16,384 hardware threads (2,048 PEs × 8) | confirmed | hw-architecture |
| Compute Engine | RISC-V Rocket Core management processor (quad-core, 1.5 GHz; on-chip Linux; SC4s new feature) | confirmed | hw-architecture, search-results |
| Memory Hierarchy | L1: 4 KB I-cache + 4 KB D-cache + 24 KB scratchpad per PE (SW-managed scratchpad) | confirmed | hw-architecture, search-results |
| Memory Hierarchy | L2: 32 KB I + 64 KB D per city (16 PEs) | confirmed | hw-architecture, search-results |
| Memory Hierarchy | L3: 64 MB shared per state (2,048 PEs) — new in SC4s; major HBM traffic reduction | confirmed | hw-architecture, search-results |
| Memory Hierarchy | Hierarchy: PE → Village (4 PEs) → City (16 PEs) → Prefecture (~256 PEs) → State (2,048 PEs) | confirmed | hw-architecture, search-results |
| Memory Hierarchy | Total on-chip SRAM: ~140 MB estimated (64 MB L3 + 12 MB L2 + ~64 MB L1/scratchpad) | inferred | hw-architecture |
| Off-chip Memory | HBM3: 4 stacks, 96 GB, 3.2 TB/s (major leap vs SC3's HBM2 32 GB ~512 GB/s) | confirmed | hw-architecture, search-results |
| Process / Package | TSMC 5nm, 556 mm² single monolithic die, 2.5D packaging (4× HBM3 on interposer), ~600W TDP | confirmed | hw-architecture, search-results |
| Host Interface | PCIe Gen 5 × 16 (128 GB/s bidir; upgrade from SC3's PCIe Gen 4) | confirmed | hw-architecture, search-results |
| Scale-up | None — no proprietary chip-to-chip fabric; 4 SC4s per node via PCIe through AMD EPYC host | confirmed | hw-architecture, search-results |
| Scale-out | InfiniBand (400 Gb/s NDR per ZettaScaler4.0 node) | confirmed | hw-architecture, search-results |
| System | ZettaScaler 4.0 (**planned, not announced as available**): 4× SC4s + AMD EPYC 9555P (Zen 5, 64-core) + 400G NDR IB + liquid immersion cooling. Absent from PEZY's products page as of 2026-08-08 | confirmed | search-results, pezy.co.jp/en/products/ (2026-08-08) |
| System | ZettaScaler 3.0 (shipping): SC3 + AMD EPYC 7702P (Zen 2) + IB EDR + liquid/air cooling | confirmed | search-results |
| System | ZettaVEGA (shipping, PEZY-SC3-based genome-analysis system) and PZLAST / PZLAST-MAG protein-search service — PEZY's visible 2026 commercial deployments | confirmed | pezy.co.jp/en/products/, news 2026-05-11 / 2026-05-13 |
| Product Status | PEZY-SC4s: **pre-production as of 2026-08-08** — never listed on PEZY's product page, no press release, no tape-out/sampling/shipping announcement; stated end-2025 release target (PEZY release 2025-06-06) missed by 8+ months with no revised date and no vendor delay notice | confirmed (absence-based for the slip) | pezy.co.jp news index (JA+EN) and products page, fetched 2026-08-08 |
| Cooling | Liquid immersion: fluorine-type inert liquid (non-combustible, non-toxic, odorless, high electrical insulation) | confirmed | search-results, hw-architecture |
| Energy Efficiency | PEZY-SC4s: ~91 GF/W FP64 **simulated only** (cf. H200 ~49, MI300A ~110). No measured Green500 entry behind it — no PEZY/ZettaScaler/ExaScaler system in the June 2026 Green500 **top 20** (top-20 verified only, not all 500) | vendor simulation | search-results, top500.org/lists/green500/list/2026/06/ |
| Energy Efficiency | PEZY-SC3: 24.6 GF/W FP64 (measured; Green500 #12 Nov 2021) | confirmed | search-results |
| Energy Efficiency | PEZY-SC2: 17 GF/W FP64 (Green500 #1 Nov 2017, Shoubu System B) | confirmed | search-results |
| Energy Efficiency | PEZY-SC: 7.03 GF/W FP64 (Green500 #1 Jun/Nov 2015, Shoubu) | confirmed | search-results |
