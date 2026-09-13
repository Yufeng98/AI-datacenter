# Tenstorrent Software Stack & Hardware Resources

*as_of: 2026-04-05*
*device_class: Tensix RISC + SFPU*
*seeds: https://github.com/tenstorrent/tt-forge, https://github.com/tenstorrent/tt-metal, https://github.com/tenstorrent/tt-llk*

## Software Stack

### Framework Integration
- [TT-Forge](https://tenstorrent.com/en/software/tt-forge) — Tenstorrent's end-to-end compiler stack with frontends for PyTorch, JAX, ONNX, TensorFlow via TT-XLA and TT-Forge-FE
- [TT-XLA (GitHub)](https://github.com/tenstorrent/tt-xla) — PJRT-based bridge for JAX and PyTorch/XLA integration; compiles models via StableHLO into TT-MLIR
- [TT-XLA Documentation](https://docs.tenstorrent.com/tt-xla/getting_started.html) — Getting started guide for JAX/torch.compile backend
- [TT-Forge-FE (GitHub)](https://github.com/tenstorrent/tt-forge-fe) — Graph compiler frontend for ONNX, TensorFlow, and other ML frameworks
- [PyTorch-XLA fork (GitHub)](https://github.com/tenstorrent/pytorch-xla) — Tenstorrent's fork enabling PyTorch on XLA devices
- [TT-Torch (GitHub)](https://github.com/tenstorrent/tt-torch) — *Deprecated* legacy PyTorch frontend for tt-mlir; superseded by TT-XLA
- [pytorch2.0_ttnn (GitHub)](https://github.com/tenstorrent/pytorch2.0_ttnn) — *Deprecated* TT-NN compiler for PyTorch 2; enables PyTorch models on Tenstorrent via eager or compile path
- [TT-Buda (GitHub)](https://github.com/tenstorrent/tt-buda) — *Legacy* software stack compiling AI/ML models from PyTorch, TensorFlow; predecessor to TT-Forge
- [TT-NN Tutorials](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tutorials.html) — Official TT-NN tutorials for framework-level model execution

### Compiler / IR
- [TT-MLIR (GitHub)](https://github.com/tenstorrent/tt-mlir) — MLIR-based compiler framework; defines custom dialects and transformation passes for Tenstorrent hardware
- [TT-MLIR Documentation](https://docs.tenstorrent.com/tt-mlir/) — Official docs for TT-MLIR, dialects, passes, and code generation
- [TT-Forge (GitHub)](https://github.com/tenstorrent/tt-forge) — Top-level compiler repo combining frontends and MLIR backend; runs GPT-OSS 120B, Llama 3 70B, SDXL, Whisper, YOLOv12
- [TT-Forge Documentation](https://docs.tenstorrent.com/forge/) — Docs for the full TT-Forge compiler stack
- [TT-MLIR README](https://github.com/tenstorrent/tt-mlir/blob/main/README.md) — Architecture overview of MLIR compiler targeting Tenstorrent accelerators
- [tt-mlir Getting Started](https://docs.tenstorrent.com/tt-mlir/getting-started.html) — Build guide and first steps for the MLIR compiler
- [TT-Alchemist (Code Generation)](https://docs.tenstorrent.com/tt-xla/getting_started_codegen.html) — Code generation tool transforming models into human-readable TT-NN library calls

### Op Library
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) — PyTorch-like Python & C++ neural network op library built on TT-Metalium
- [TT-NN Getting Started](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/get_started.html) — Installation and first ops with TT-NN
- [TT-Metal GitHub (TT-NN)](https://github.com/tenstorrent/tt-metal) — Hosts both TT-NN op library and TT-Metalium low-level kernel model
- [Programming Mesh of Devices with TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) — Tech report on multi-device tensor-parallel operations using TT-NN

### Kernel Library
- [TT-LLK (GitHub)](https://github.com/tenstorrent/tt-llk) — Header-only low-level kernels (LLKs) for Wormhole and Blackhole; implements Unpack → Math → Pack pipeline
- [TT-LLK Product Page](https://tenstorrent.com/en/software/tt-llk) — Official description of the low-level kernel library
- [LLK API Docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html) — Reference for SFPU/LLK APIs within TT-Metalium docs
- [DeepWiki: tt-llk Overview](https://deepwiki.com/tenstorrent/tt-llk/1-overview) — Community analysis of TT-LLK architecture and usage
- [Matmul Kernel Header](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/include/compute_kernel_api/matmul.h) — Example compute kernel API header for matrix multiply

### Runtime
- [TT-Metalium Product Page](https://tenstorrent.com/en/software/tt-metalium) — Official page for the low-level open-source SDK
- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) — Full reference for TT-Metalium runtime and programming model
- [TT-Metalium Getting Started](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html) — First steps writing kernels and running programs
- [METALIUM_GUIDE.md](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) — Comprehensive guide to the Metalium programming model (NoC, SRAM, kernels)
- [Memory for Kernel Developers](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html) — Deep-dive into SRAM, DRAM, circular buffers from a kernel developer perspective
- [DRAM Loopback Example](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/dram_loopback.html) — Example demonstrating data movement between DRAM and Tensix cores
- [TT-NPE (GitHub)](https://github.com/tenstorrent/tt-npe) — NoC performance estimator for Tensix-based devices; useful for kernel optimization
- [tt-metal Releases](https://github.com/tenstorrent/tt-metal/releases) — Versioned releases of the TT-Metal stack
- [Understanding the Tenstorrent Software Stack](https://docs.tenstorrent.com/getting-started/tt-software-stack.html) — High-level overview of the full stack from TT-Forge down to hardware
- [Installing the Tenstorrent Software Stack](https://docs.tenstorrent.com/getting-started/README.html) — Installation guide covering system deps, drivers, and TT-Metalium environment

### Driver / Firmware
- [TT-UMD (GitHub)](https://github.com/tenstorrent/tt-umd) — User-Mode Driver for Tenstorrent hardware; C++ library providing low-level device access above tt-kmd
- [TT-KMD (GitHub)](https://github.com/tenstorrent/tt-kmd) — Tenstorrent Kernel Module; registers /dev/tenstorrent/%d device files; required by tt-umd
- [TT-Firmware (GitHub)](https://github.com/tenstorrent/tt-firmware) — Device firmware repository for Tenstorrent chips; manages DMC and SMC controllers
- [TT-System-Firmware (GitHub)](https://github.com/tenstorrent/tt-system-firmware) — Newer firmware release repository (successor to tt-firmware for latest releases)
- [TT-Zephyr-Platforms (GitHub)](https://github.com/tenstorrent/tt-zephyr-platforms) — Zephyr RTOS firmware for Tenstorrent hardware (PCIe cards)
- [TT Zephyr Platforms Docs](https://docs.tenstorrent.com/tt-zephyr-platforms/develop/getting_started/index.html) — Getting started with Zephyr-based firmware for Blackhole PCIe cards
- [TT-SMI (GitHub)](https://github.com/tenstorrent/tt-smi) — Console-based hardware information program (analogous to nvidia-smi)
- [Install TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/installing.html) — Detailed install instructions including driver setup
- [Tenstorrent on NixOS](https://github.com/NixOS/nixpkgs/blob/release-25.11/nixos/modules/hardware/tenstorrent.nix) — NixOS module for Tenstorrent hardware drivers

### Communication
- [TT-NN CCL API Index](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html) — Official API reference for all CCL operations (all_reduce, all_gather, reduce_scatter, broadcast, etc.)
- [ttnn.all_gather API](https://docs.tenstorrent.com/ttnn/latest/ttnn/api/ttnn.all_gather.html) — Aggregates tensors from all devices along a specified dimension; fundamental CCL primitive
- [ttnn.reduce_scatter API](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.reduce_scatter.html) — Reduces and scatters sharded tensors across MeshDevice
- [TT-Topology (GitHub)](https://github.com/tenstorrent/tt-topology) — CLI utility to configure ETH routing (mesh/linear/torus) across multi-card systems; flashes n150/n300 routing tables
- [Programming Mesh of Devices with TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) — Tech report on tensor-parallel patterns and collective communication via MeshDevice
- [Multi-Galaxy CCL Issue](https://github.com/tenstorrent/tt-metal/issues/28272) — Active development of multi-galaxy CCL specifications for DeepSeek-scale workloads
- [Tenstorrent Wormhole Scale-Out Analysis (SemiAnalysis)](https://semianalysis.com/2021/06/25/tenstorrent-wormhole-analysis-a-scale/) — In-depth analysis of the Ethernet-based scale-out architecture
- [Tenstorrent Blackhole Scale-Out (SemiAnalysis)](https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and) — Blackhole scale-out, sparsity, conditional execution analysis

### Assembler / ISA
- [TT-ISA Documentation (GitHub)](https://github.com/tenstorrent/tt-isa-documentation) — Official low-level ISA documentation for Wormhole B0 and Blackhole A0
- [Baby RISC-V ISA README](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md) — ISA reference for the five baby RISC-V cores per Tensix tile
- [Matrix Unit ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md) — Instruction reference for the FPU/matrix engine (MVMUL, GMPOOL, ELWMUL)
- [Vector Unit (SFPU) ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md) — SIMD instruction set for the 32-lane SFPU
- [Scalar Unit ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/ScalarUnit.md) — ThCon scalar unit instruction reference
- [FMA Miscellaneous ISA](https://github.com/tenstorrent/tt-isa-documentation/blob/main/Miscellaneous/FMA/README.md) — FMA instruction documentation
- [Corsix: Wormhole Vector Instruction Set](https://www.corsix.org/content/tt-wh-part6) — Community deep-dive into the Wormhole SFPU vector ISA
- [Corsix: Taking Apart T Tiles](https://www.corsix.org/content/tt-wh-part5) — Detailed tile-level analysis with ISA context

## Hardware Architecture

### Compute Engine
- [Tensix Neo IP Page](https://tenstorrent.com/en/ip/tensix-neo) — Official Tensix Neo IP block: 5 baby RISC-V + matrix FPU + SFPU + NoC router + 1–2 MB SRAM
- [Blackhole Hot Chips 2024 PDF](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf) — Official HotChips 2024 presentation covering Blackhole microarchitecture, programming model, and scale-out
- [Introduction to Tenstorrent (RISC-V EPCC)](http://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf) — HPC-Asia 2025 overview: Tensix internals (5 baby RISC-V cores, 1MB SRAM, SIMD+Matrix engine, NoC)
- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) — Official Blackhole specs: 140 Tensix cores, 6nm, 745 TOPS FP8, 32 GB GDDR6
- [Wormhole Specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html) — Official Wormhole specs: 80 Tensix cores, 12nm, 328 TOPS FP8, 12 GB GDDR6
- [Grayskull Specifications](https://docs.tenstorrent.com/aibs/grayskull/specifications.html) — Official first-gen Grayskull specs; historical context for Tensix generation evolution
- [Dissecting Blackhole via Microbenchmarking (ASPLOS)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf) — Academic paper: 140 Tensix cores, 24 GDDR6 controllers, 210 MB on-chip SRAM, 745 TOPS FP8
- [Assessing Tenstorrent RISC-V MatMul Capabilities (arXiv)](https://arxiv.org/html/2505.06085v1) — Academic evaluation of matrix multiply performance on Tensix
- [Numerical Kernels on Tenstorrent Wormhole (arXiv)](https://arxiv.org/html/2603.23343v1) — Study of spatial accelerator performance for numerical kernels
- [Programming Tenstorrent Processors (blog)](https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi) — Practitioner guide to Tensix core programming

### Data Path
- [METALIUM_GUIDE.md — NoC Section](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) — Describes dual NoC (NoC0 east/south + NoC1 west/north), 32-byte links, torus topology
- [TT-NPE (NoC Performance Estimator)](https://github.com/tenstorrent/tt-npe) — Tool to estimate and optimize NoC data movement for Tensix devices
- [Corsix: Wormhole Physicalities Part 1](https://www.corsix.org/content/tt-wh-part1) — Physical die analysis: tile layout, NoC routing, link widths
- [Arteris NoC IP Press Release](https://tenstorrent.com/en/vision/tenstorrent-expands-deployment-of-arteris-network-on-chip-ip-to-next-generation-of-chiplet-based-ai-) — Tenstorrent uses Arteris FlexNoC IP for chiplet-based AI solutions
- [Memory on Tenstorrent (blog)](https://clehaxze.tw/gemlog/2025/03-17-memory-on-tenstorrent.gmi) — Practitioner analysis of data movement patterns across NoC and SRAM

### On-chip Memory
- [Memory for Kernel Developers (TT-Metalium docs)](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html) — SRAM layout: L1 buffers, circular buffers, DRAM-backed tensors
- [Attention in SRAM on Tenstorrent Grayskull (arXiv)](https://arxiv.org/html/2407.13885v1) — FlashAttention-style computation using on-chip SRAM (Grayskull)
- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) — 210 MB on-chip SRAM, 1.5 MB per Tensix core, 140 cores on Blackhole
- [A Close Look at SRAM for Inference (Vik's Newsletter)](https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference) — Analysis of Tenstorrent's SRAM-first strategy vs. HBM

### Off-chip Memory
- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) — 32 GB GDDR6, 512 GB/s bandwidth, 24 memory controllers, no HBM
- [Wormhole Specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html) — 12 GB GDDR6 (6 controllers × 2 channels × 1 GB), 336 GB/s
- [Exploring FFT on Tenstorrent Wormhole (arXiv)](https://arxiv.org/html/2506.15437v1) — Memory bandwidth study for FFT workloads on Wormhole
- [TT-QuietBox 2 Press Release (Morningstar)](https://www.morningstar.com/news/accesswire/1145920msn/tenstorrent-unveils-tt-quietboxtm-2-the-first-risc-v-ai-workstation-with-a-fully-open-source-stack-to-deliver-teraflop-class-inference) — QuietBox 2: GDDR6-based workstation avoiding HBM supply constraints

### Host Interface / Package
- [Wormhole PCIe Cards Documentation](https://docs.tenstorrent.com/aibs/wormhole/) — PCIe 4.0 x16, 12 GB GDDR6, 8-pin EPS12V power
- [Blackhole PCIe Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) — PCIe interface, 32 GB GDDR6, QSFP-DD ports for Ethernet
- [Wormhole Hardware Page](https://tenstorrent.com/en/hardware/wormhole) — Official Wormhole product page: 80 Tensix cores, 12nm, 328 TOPS FP8
- [Blackhole Hardware Page](https://tenstorrent.com/en/hardware/blackhole) — Official Blackhole product page: 140 Tensix cores, 6nm, 745 TOPS FP8
- [Wormhole Series Part 1: Physicalities (corsix)](https://www.corsix.org/content/tt-wh-part1) — Physical die and package analysis of the Wormhole chip

### Scale-up Interconnect
- [Warp 100 Bridge Documentation](https://docs.tenstorrent.com/aibs/warp100) — Official docs for the Warp 100 (TX-01002) inter-card bridge connecting Wormhole n150d cards
- [TT-QuietBox Specifications](https://docs.tenstorrent.com/systems/quietbox/specifications.html) — Multi-card workstation system specs; topology of 4× Blackhole via internal Ethernet mesh
- [T3000 System Specifications](https://docs.tenstorrent.com/systems/t3000/specifications.html) — TT-LoudBox / T3000 rack system specs; 8-card Wormhole galaxy in 2×4 mesh
- [TT-Topology (GitHub)](https://github.com/tenstorrent/tt-topology) — Configures ETH routing (mesh/linear/torus) for multi-card single-host deployments
- [Wormhole PCIe Hardware Documentation](https://docs.tenstorrent.com/aibs/wormhole/) — Wormhole inter-card connectivity via 16×100 Gbps Ethernet ports
- [DeepWiki: Wormhole Processors](https://deepwiki.com/tenstorrent/tenstorrent.github.io/2.2.2-wormhole-processors) — Community analysis of Wormhole inter-chip connectivity

### Scale-out Interconnect
- [Wormhole Scale-Out Analysis (SemiAnalysis)](https://newsletter.semianalysis.com/p/tenstorrent-wormhole-analysis-a-scale) — 16×100 Gbps Ethernet per Wormhole chip; 2D mesh rack topology
- [Blackhole Scale-Out Analysis (SemiAnalysis)](https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and) — Blackhole: 10×400 Gbps Ethernet (1 TBps total); QSFP-DD ports
- [Blackhole Specifications (Ethernet)](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) — Four QSFP-DD ports, each 800 Gbps, up to 2 m cable, 1 TBps aggregate
- [The Register: Blackhole Chips](https://www.theregister.com/2024/08/27/tenstorrent_ai_blackhole/) — Press coverage of Blackhole's RISC-V + Ethernet scale-out design
- [Tenstorrent and AI Hardware Startups (Irrational Analysis)](https://irrationalanalysis.substack.com/p/tenstorrent-and-the-state-of-ai-hardware) — Business and architectural analysis of Ethernet mesh strategy

## Other Resources

### Tutorials & Community
- [Tenstorrent GitHub Organization](https://github.com/tenstorrent) — Official GitHub org; includes tt-metal, tt-forge, tt-llk, tt-mlir, tt-umd, tt-kmd, tt-smi, tt-installer, tt-npe, tt-buda
- [TT-NN Tutorials](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tutorials.html) — Official step-by-step tutorials for TT-NN operations and model execution
- [Developers Portal](https://tenstorrent.com/en/developers) — Entry point for developer resources, SDKs, and getting started guides
- [Programming Tenstorrent Processors (Martin's blog)](https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi) — Practitioner guide to writing kernels on Tensix
- [Arch Linux + Metalium Guide (Martin's blog)](https://clehaxze.tw/gemlog/2024/07-07-a-gentle-guide-on-getting-your-tenstorrent-card-running-on-arch-linux-with-the-metalium-stack.gmi) — Community install guide for Arch Linux
- [Memory on Tenstorrent (Martin's blog)](https://clehaxze.tw/gemlog/2025/03-17-memory-on-tenstorrent.gmi) — Community deep-dive into memory management

### Tooling & Debugging
- [TT-Inference-Server (GitHub)](https://github.com/tenstorrent/tt-inference-server) — Fastest way to deploy and test models for inference serving on Tenstorrent hardware
- [TT-Exalens (GitHub)](https://github.com/tenstorrent/tt-exalens) — Low-level hardware debugger (TT-Lensium) for Wormhole and Blackhole devices
- [TTNN-Visualizer (GitHub)](https://github.com/tenstorrent/ttnn-visualizer) — Visualizer for model execution: interactive graphs, memory plots, tensor details, operation flow graphs
- [TT-Installer (GitHub)](https://github.com/tenstorrent/tt-installer) — One-command installer for the full Tenstorrent software stack (uses Podman/Docker)
- [TT-Tools-Common (GitHub)](https://github.com/tenstorrent/tt-tools-common) — Shared utilities library across Tenstorrent tools

### Benchmarks & Analysis
- [Dissecting Blackhole via Microbenchmarking (ASPLOS 2025)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf) — Systematic microbenchmarking of Blackhole compute, memory, NoC
- [Assessing RISC-V MatMul Acceleration (arXiv 2505.06085)](https://arxiv.org/html/2505.06085v1) — Performance study of matrix multiply on Tensix cores
- [Numerical Kernels on Wormhole Spatial Accelerator (arXiv 2603.23343)](https://arxiv.org/html/2603.23343v1) — Study of sparse/dense workloads
- [Exploring FFT on Wormhole (arXiv 2506.15437)](https://arxiv.org/html/2506.15437v1) — FFT workload characterization on Wormhole
- [Attention in SRAM on Grayskull (arXiv 2407.13885)](https://arxiv.org/html/2407.13885v1) — FlashAttention-style SRAM-resident attention
- [Tenstorrent Wormhole Series (corsix)](https://www.corsix.org/content/tt-wh-part1) — Multi-part community deep-dive: physicalities, NoC, ISA, vector units

### Whitepapers & Press
- [Hot Chips 2024: Blackhole & TT-Metalium (PDF)](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf) — Official architectural presentation
- [HPC-Asia 2025: Introduction to Tenstorrent (PDF)](http://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf) — Conference overview of architecture and ecosystem
- [Tenstorrent Shares RISC-V CPU Roadmap (Tom's Hardware)](https://www.tomshardware.com/news/tenstorrent-shares-roadmap-of-ultra-high-performance-risc-v-cpus-and-ai-accelerators) — Product roadmap including Ascalon RISC-V CPU + Tensix AI
- [LLM Inference Hardware Enterprise Guide (IntuitionLabs)](https://intuitionlabs.ai/articles/llm-inference-hardware-enterprise-guide) — Tenstorrent context in enterprise inference market

---

## Resources Added 2026-08-08 (Galaxy Blackhole GA scan, April–August 2026)

### Hardware Product Pages & System Documentation
- [Galaxy Product Page](https://tenstorrent.com/en/hardware/galaxy) — Galaxy Blackhole and Galaxy Wormhole specifications and **public list pricing** ($110,000 / $70,000 from); source of the 23 PFLOPS Block FP8, 6.2 GB SRAM @ 2.9 PB/s, 1 TB GDDR6 @ 16 TB/s, 32 TB/s fabric, 11.2 TB/s scale-out, EPYC 9004 host, 8–14.5 kW figures
- [Galaxy Blackhole System Documentation](https://docs.tenstorrent.com/systems/galaxy-blackhole/index.html) — User Guide v1.6, dated 2026-08-04; corroborates the product-page spec
- [Galaxy Blackhole datasheet PDF](https://docs.tenstorrent.com/_downloads/3086863c42126fd0d63b01baccf8432e/galaxy-blackhole.pdf)
- [Blackhole Product Page (updated SKU list)](https://tenstorrent.com/en/hardware/blackhole) — p100a / p150a / p150b only (p300 delisted); 120 Tensix cores, 664 TFLOPS BLOCKFP8, 28 GB @ 448 GB/s (p100a), $999 / $1,399
- [docs.tenstorrent.com/aibs](https://docs.tenstorrent.com/aibs/) — current AI-board and system index; lists Wormhole and Blackhole cards, QuietBox, LoudBox and Galaxy systems, and the **TT-Lang** software component; no Quasar entry

### Newsroom & Press (April–August 2026)
- [Tenstorrent Enables AI at Scale with Industry-Leading Performance (2026-04-28)](https://tenstorrent.com/en/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance) — Galaxy Blackhole **general availability** announcement; DeepSeek-R1-0528 350+ tok/s/user; Prodia 720p 81-frame in 2.4 s; partners Prodia, Equinix, OrionVM, BetterBrain, Virtu Financial, Turiyam, Cirrascale, ai&
- [The Register: Tenstorrent's Galaxy Blackhole AI servers are finally out (2026-04-28)](https://www.theregister.com/software/2026/04/28/tenstorrents-galaxy-blackhole-ai-servers-are-finally-out/5229759) — independent write-up; reports ~300 tok/s/user actual vs 350 expected, unspecified batch size, ~100 Tbps intra-node Ethernet, and prior "generally poor performance scaling"
- [HPCwire/AIwire: General Availability of Galaxy Blackhole (2026-05-01)](https://www.hpcwire.com/aiwire/2026/05/01/tenstorrent-announces-general-availability-of-galaxy-blackhole-ai-system/) — syndication of the 2026-04-28 announcement, not a separate event
- [AcceleratedComputing.ai: Galaxy Blackhole GA (2026-05-01)](https://www.acceleratedcomputing.ai/news/2026-05-01-tenstorrent-galaxy-blackhole-ga/)
- [TT-Deploy (2026-05-04)](https://tenstorrent.com/en/newsroom/tt-deploy) — 36-Galaxy supercluster; ~2.5M Hugging Face models at ~90% pass rate (unverified vendor claim)
- [Tenstorrent Sets New Performance Records, Launches TT-Ascalon S (2026-06-30)](https://tenstorrent.com/en/newsroom/tenstorrent-sets-new-performance-records-launches-tt--ascalon-s) — 400+ tok/s/user DeepSeek; Kimi K2.6 900 tok/s/user; LTX 2.3 Fast; TT-Ascalon S RISC-V CPU IP; Japan expansion (ai& 120+ Galaxy systems, Rapidus 2 nm chiplet, Osaka DC)
- [Mirage News: Tenstorrent breaks records, unveils TT-Ascalon S](https://www.miragenews.com/tenstorrent-breaks-records-unveils-tt-ascalon-s-1701731/) — independent syndication confirming the 2026-06-30 date
- [Morningstar/ACCESSWIRE: TT-Ascalon S and Japan expansion](https://www.morningstar.com/news/accesswire/1183834msn/tenstorrent-sets-new-performance-records-launches-tt-ascalon-s-and-expands-across-japan)
- [Tenstorrent Newsroom index](https://tenstorrent.com/en/news)
- [Tenstorrent + Infinia Technologies sovereign AI partnership (2026-01-27)](https://tenstorrent.com/en/newsroom/tenstorrent-and-infinia-technologies-partner-to-build-sovereign-ai-infrastructure) — **out of window**, predates the repo baseline
- [The Register: Blackhole QuietBox workstation reviewed (2025-11-27)](https://www.theregister.com/on-prem/2025/11/27/blackhole-quietbox-tenstorrents-ai-workstation-reviewed/2113269)

### Quasar (next generation)
- [The Register: Samsung to fab RISC-V chips for Tenstorrent (2023-10-04)](https://www.theregister.com/on-prem/2023/10/04/samsung-to-fab-risc-v-chips-for-tenstorrent/) — **Quasar chiplet on Samsung Foundry SF4X (4 nm-class), Taylor TX**; the only public process-node attribution for Quasar
- [inkl: Tenstorrent to use Samsung's SF4X for Quasar low-cost AI chiplet](https://www.inkl.com/news/tenstorrent-to-use-samsung-s-sf4x-for-quasar-low-cost-ai-chiplet)
- [GitHub commit search — Quasar in tt-llk](https://api.github.com/search/commits?q=repo:tenstorrent/tt-llk+Quasar&sort=committer-date&order=asc) — first commit 2025-08-15 (PR #593); 85 commits
- [GitHub commit search — Quasar in tt-metal](https://api.github.com/search/commits?q=repo:tenstorrent/tt-metal+Quasar&sort=committer-date&order=asc) — 662 commits
- [tt-isa-documentation repo contents](https://api.github.com/repos/tenstorrent/tt-isa-documentation/contents/) — only `WormholeB0` and `BlackholeA0`; no Quasar ISA reference

### Software Stack (Metal 2.0 / Device 2.0 / DataflowBuffer)
- [tt-metal v0.75.0 release (2026-07-30)](https://api.github.com/repos/tenstorrent/tt-metal/releases/tags/v0.75.0) — Metal 2.0 kernel scratchpad; NOC-distinctness legality check; Device 2.0 migration of norm/reduction ops; CircularBuffer → DataflowBuffer migration; multi-config layer support; emule multichip fiber engine
- [tt-metal releases index](https://github.com/tenstorrent/tt-metal/releases)
- [GitHub commit search — "Metal 2.0" in tt-metal](https://api.github.com/search/commits?q=repo:tenstorrent/tt-metal+%22Metal+2.0%22&sort=committer-date&order=asc) — earliest 2025-11-16 (#32121); 168 commits
- [tt-forge releases API](https://api.github.com/repos/tenstorrent/tt-forge/releases?per_page=5) — 1.5.0.dev builds with DeepSeek-V3.1/V3.2, GLM-4.7, Kimi K2/K2.6, Qwen 2.5/3, Phi-4, Falcon-3 on n150/n300/p150

### Negative Results (checked, nothing found)
- [Hot Chips 38 advance program](https://hotchips.org/advance-program/) — **no Tenstorrent talk** (HC38 runs Aug 23–25, 2026; program public, no content published)
- [MLPerf Inference v6.0 results (2026-04-01)](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/) — no Tenstorrent submission
