# Tenstorrent Layer Mapping Table

*as_of: 2026-08-08*
*chip: tenstorrent*
*device_class: Tensix RISC + SFPU*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | TT-XLA (PJRT bridge for JAX and torch.compile via StableHLO → TT-MLIR) | confirmed | tt-xla, tt-mlir, software-stack |
| Framework Integration | TT-Torch (torch.export → StableHLO → TT-MLIR; deprecated, TT-XLA preferred) | confirmed | tt-metal, software-stack |
| Framework Integration | TT-Forge-FE (ONNX, TensorFlow, and other ML framework frontends → TT-MLIR) | confirmed | tt-forge, software-stack |
| Framework Integration | TT-Forge (top-level compiler repo; CI-validates GPT-OSS 120B, Llama 3 70B, SDXL, Whisper, YOLOv12) | confirmed | tt-forge, software-stack |
| Framework Integration | TT-Buda (legacy; superseded by TT-Forge; PyTorch/TF models) | confirmed | tt-buda, software-stack |
| Compiler / IR | TT-MLIR TTIR dialect (high-level tensor ops; hardware-independent; analogous to StableHLO/TOSA) | confirmed | tt-mlir, software-stack |
| Compiler / IR | TT-MLIR TTNN dialect (models TT-NN op library as MLIR; lowered from TTIR) | confirmed | tt-mlir, software-stack |
| Compiler / IR | TT-MLIR TTMetal dialect (host-side TT-Metalium ops: buffer alloc, program dispatch) | confirmed | tt-mlir, software-stack |
| Compiler / IR | TT-MLIR TTKernel dialect (compute kernel representation for code generation) | confirmed | tt-mlir, software-stack |
| Compiler / IR | TT-MLIR passes: op fusion, tensor sharding, memory layout assignment, TTIR→TTNN→TTMetal lowering | confirmed | tt-mlir, software-stack |
| Op Library | TT-NN: ttnn.matmul, ttnn.linear, ttnn.bmm, ttnn.softmax, ttnn.conv2d | confirmed | tt-metal-ttnn, software-stack |
| Op Library | TT-NN: ttnn.scaled_dot_product_attention (FlashAttention-style, SRAM-resident) | confirmed | tt-metal-ttnn, tt-metal.md |
| Op Library | TT-NN: ttnn.layer_norm, ttnn.rms_norm, ttnn.embedding | confirmed | tt-metal-ttnn, software-stack |
| Op Library | TT-NN: MeshDevice (2D mesh multi-chip abstraction; ShardTensorToMesh, ReplicateTensor) | confirmed | tt-metal-ttnn, hw-architecture |
| Op Library | TT-NN: Tensor layout system (ROW_MAJOR, TILE_LAYOUT, interleaved, height/width/block sharded) | confirmed | tt-metal.md, software-stack |
| Op Library | TT-NN: **Device 2.0 API migration** — normalization and reduction ops migrated; CNN ops in progress; earliest commit 2025-11-16 (#32121) | confirmed | tt-metal v0.75.0, GitHub commit search |
| Framework Integration | **TT-Lang** — software component newly listed on docs.tenstorrent.com/aibs; scope and relationship to TT-Metalium/TT-NN not disclosed | listed-only | docs.tenstorrent.com/aibs |
| Kernel Library | TT-LLK: header-only Low Level Kernels; Unpack→Math→Pack pipeline (llk_unpack_A, llk_math_eltwise, llk_pack) | confirmed | tt-llk, software-stack |
| Kernel Library | TT-LLK: SFPU intrinsics (llk_sfpu_exp, llk_sfpu_tanh, custom 32-lane SIMD functions) | confirmed | tt-llk, sw-stack |
| Kernel Library | TT-LLK: architecture-specific variants for Wormhole B0 and Blackhole A0 | confirmed | tt-llk |
| Kernel Library | TT-LLK: **Quasar backend** — initial commit 2025-08-15 (PR #593); 85 Quasar commits in tt-llk, 662 in tt-metal; SFPU/matmul perf coverage, ttsim regression selection | confirmed | tt-llk, tt-metal v0.75.0, GitHub commit search |
| Assembler / ISA | Quasar ISA: **not disclosed** — tt-isa-documentation contains only WormholeB0 and BlackholeA0 top-level directories | not disclosed | tt-isa-documentation |
| Kernel Library | FlashAttention on Tenstorrent (SRAM-resident; tech_reports/FlashAttention) | confirmed | tt-metal, hw-architecture |
| Runtime | TT-Metalium Device/Program/CircularBuffer/KernelHandle/Buffer/CommandQueue C++ SDK | confirmed | tt-metal.md, software-stack |
| Runtime | TT-Metalium: three kernel types per core (Reader/BRISC, Compute/TRISC0-2, Writer/NCRISC) | confirmed | tt-metal.md, METALIUM_GUIDE.md |
| Runtime | TT-Metalium: JIT kernel compilation (C++ → RISC-V ELF via SFPI cross-compiler) | confirmed | tt-metal.md |
| Runtime | TT-Metalium: Fast Dispatch (ARC management core; microsecond dispatch latency; no driver round-trip) | confirmed | tt-metal.md, software-stack |
| Runtime | TT-Metalium: SPMD programming model (same program, per-core runtime args, across Tensix mesh) | confirmed | tt-metal.md, METALIUM_GUIDE.md |
| Runtime | TT-Metalium: CircularBuffer (L1 SRAM ring buffer; cb_push_back/cb_wait_front producer-consumer sync) — being superseded by DataflowBuffer | confirmed | tt-metal.md |
| Runtime | TT-Metalium: **DataflowBuffer (DFB)** — replacement for CircularBuffer; migration underway across eltwise, conv/pool, data-movement and Moreh kernels; DFBs enabled in the Quasar fast-dispatch flow | confirmed | tt-metal v0.75.0 release notes (2026-07-30) |
| Runtime | TT-Metalium: **Metal 2.0 API** — kernel scratchpad support; NOC-distinctness legality check; rearchitecture underway in tt-metal since 2025-11-16 (168 commits) | confirmed | tt-metal v0.75.0, GitHub commit search |
| Runtime | TT-Metalium: **multi-config layer support** (new in v0.75.0) | confirmed | tt-metal v0.75.0 |
| Runtime | TT-Metalium: NoC API (noc_async_read, noc_async_write, noc_async_read_barrier) | confirmed | tt-metal.md, METALIUM_GUIDE.md |
| Driver / Firmware | TT-KMD (Linux kernel module; /dev/tenstorrent; PCIe BAR mapping; IRQ handling) | confirmed | tt-kmd, software-stack |
| Driver / Firmware | TT-UMD (user-mode driver; C++ library; above TT-KMD; low-level device access) | confirmed | tt-umd, software-stack |
| Driver / Firmware | TT-Firmware / TT-System-Firmware (DMC + SMC on-chip controllers) | confirmed | tt-firmware, software-stack |
| Driver / Firmware | TT-Zephyr-Platforms (Zephyr RTOS firmware for Blackhole PCIe cards) | confirmed | tt-zephyr, software-stack |
| Driver / Firmware | TT-SMI (console hardware monitor: temperatures, utilization, power; analogous to nvidia-smi) | confirmed | tt-smi, software-stack |
| Communication | TT-NN CCL: ttnn.all_reduce, ttnn.all_gather, ttnn.reduce_scatter over Ethernet | confirmed | tt-metal-ttnn, hw-architecture |
| Communication | **emule multichip fiber engine + fabric/CCL teleport** (demonstrated on an 8-chip Blackhole LoudBox) | confirmed | tt-metal v0.75.0 |
| Communication | TT-Topology (CLI tool; configures Ethernet routing: mesh/linear/torus per deployment) | confirmed | tt-topology, hw-architecture |
| Communication | Ethernet tiles in NoC grid (cross-chip noc_async_read/write; transparent multi-chip API) | confirmed | METALIUM_GUIDE.md, hw-architecture |
| Assembler / ISA | TT-ISA-Documentation: Baby RISC-V ISA for Wormhole B0 and Blackhole A0 | confirmed | tt-isa-docs, hw-architecture |
| Assembler / ISA | Matrix Unit ISA: MVMUL (matrix-vector multiply), GMPOOL, ELWMUL | confirmed | tt-isa-docs |
| Assembler / ISA | Vector Unit (SFPU) ISA: 32-lane SIMD; SFPEXP, SFPLOG, SFPTANH, SFPIADD, custom | confirmed | tt-isa-docs |
| Assembler / ISA | Scalar Unit (ThCon) ISA: scalar control flow and arithmetic | confirmed | tt-isa-docs |
| Assembler / ISA | No virtual ISA (no PTX equivalent): kernels compile directly to native RISC-V per chip generation | confirmed | tt-metal.md, hw-architecture |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Tensix core: 5 Baby RISC-V (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) + FPU + SFPU + 1.5 MB L1 SRAM | confirmed | hw-architecture, METALIUM_GUIDE.md, ASPLOS2025 |
| Compute Engine | Matrix Unit (FPU): 32×32 tile matmul; BF16/FP16/FP8(E4M3,E5M2)/INT8/INT32/FP32 | confirmed | hw-architecture, tt-isa-docs |
| Compute Engine | Vector Unit (SFPU): 32-lane SIMD; Exp, Log, Sqrt, Tanh, GeLU, Sigmoid, custom | confirmed | hw-architecture, tt-isa-docs |
| Compute Engine | Wormhole: 80 Tensix cores (72 active n150), 328 TOPS FP8, GFP 12nm, 670 mm² | confirmed | hw-architecture |
| Compute Engine | Blackhole full die: 140 Tensix cores + 16 Big RISC-V host cores (can run Linux), 745 TOPS FP8, TSMC 6nm | confirmed | hw-architecture |
| Compute Engine | Blackhole shipping PCIe SKUs (p100a / p150a / p150b): **120 Tensix cores** + 16 Big RISC-V, **664 TFLOPS BLOCKFP8**, PCIe 5.0 x16, 300 W, TSMC 6nm | confirmed | tenstorrent.com/en/hardware/blackhole (2026-08-08) |
| Compute Engine | Blackhole p300 (dual-die, 140×2, 64 GB): **delisted** from the vendor product page as of 2026-08-08; retained for architectural history | confirmed | tenstorrent.com/en/hardware/blackhole |
| Compute Engine | Galaxy Blackhole (6U, 32 Blackhole ASICs): **23 PFLOPS Block FP8**; GA 2026-04-28; from $110,000 | confirmed | tenstorrent.com/en/hardware/galaxy, GA press release, The Register 2026-04-28 |
| Compute Engine | Quasar: core count, compute-unit organisation, data types, peak throughput **not disclosed**; process node Samsung SF4X (4 nm-class, Taylor TX) per the Oct-2023 chiplet announcement | medium (node) / not disclosed (rest) | Tenstorrent Oct-2023 Samsung announcement, tt-metal/tt-llk commit history |
| Data Path | Dual NoC 2D torus (NoC 0 east/south + NoC 1 west/north); 32 bytes/cycle per link | confirmed | hw-architecture, METALIUM_GUIDE.md |
| Data Path | No hardware cache hierarchy above 1.5 MB L1 SRAM; all data movement is explicit NoC DMA | confirmed | hw-architecture, METALIUM_GUIDE.md |
| Data Path | Unpack→Math→Pack pipeline: TRISC0→TRISC1→TRISC2 hardware pipeline synchronized by FPU semaphores | confirmed | hw-architecture, tt-metal.md |
| Data Path | TensorAccessor abstraction for interleaved/sharded DRAM tensor tile addressing | confirmed | tt-metal.md |
| On-chip Memory | 1.5 MB L1 SRAM per Tensix core (software-managed scratchpad; no eviction; no coherence protocol) | confirmed | hw-architecture, METALIUM_GUIDE.md |
| On-chip Memory | 210 MB total on-chip SRAM on the 140-core Blackhole die | confirmed | hw-architecture, ASPLOS2025 |
| On-chip Memory | **180 MB on shipping Blackhole PCIe cards** (120 enabled cores × 1.5 MB) | confirmed | tenstorrent.com/en/hardware/blackhole |
| On-chip Memory | **Galaxy Blackhole: 6.2 GB accelerator SRAM @ 2.9 PB/s** across 32 ASICs (supersedes the earlier ~6.7 GB figure) | confirmed | tenstorrent.com/en/hardware/galaxy, Galaxy Blackhole User Guide v1.6 |
| On-chip Memory | ~108 MB on Wormhole n150 (72 cores × 1.5 MB) | confirmed | hw-architecture |
| On-chip Memory | L1 tensor sharding: height-sharded, width-sharded, block-sharded across Tensix L1 pool | confirmed | hw-architecture, tt-metal.md |
| Off-chip Memory | Wormhole: 12 GB GDDR6, 6 × 2-channel controllers, 192–336 GB/s | confirmed | hw-architecture |
| Off-chip Memory | Blackhole p150a/p150b: 32 GB GDDR6, 24 controllers, 512 GB/s, 384-bit interface | confirmed | hw-architecture |
| Off-chip Memory | **Blackhole p100a: 28 GB GDDR6 @ 448 GB/s** | confirmed | tenstorrent.com/en/hardware/blackhole |
| Off-chip Memory | **Galaxy Blackhole: 1 TB GDDR6 @ 16 TB/s** across 32 ASICs | confirmed | tenstorrent.com/en/hardware/galaxy |
| Off-chip Memory | GDDR6 chosen over HBM for supply chain flexibility and cost; no HBM CoWoS interposer | confirmed | hw-architecture, SemiAnalysis |
| Host Interface / Package | Wormhole: PCIe 4.0 x16 (~64 GB/s bidir), GFP 12nm | confirmed | hw-architecture |
| Host Interface / Package | Blackhole: PCIe 5.0, TSMC 6nm, 4× QSFP-DD ports | confirmed | hw-architecture |
| Host Interface / Package | Blackhole standalone mode: 16 Big RISC-V cores run Linux; no external host CPU required | confirmed | hw-architecture |
| Host Interface / Package | **Galaxy Blackhole: 1 × AMD EPYC 9004 (Zen 4, ≤32 cores, ≤280 W) host with up to 576 GB (6 × 96 GB) DDR5-4800 ECC RDIMM for 32 ASICs** | confirmed | tenstorrent.com/en/hardware/galaxy |
| Host Interface / Package | **Galaxy Blackhole chassis: 6U, air-cooled, 262 lbs / 119 kg; 8–10 kW avg, 12 kW max, configurable to 14.5 kW** | confirmed | tenstorrent.com/en/hardware/galaxy, Galaxy Blackhole User Guide v1.6 |
| Scale-up Interconnect | Wormhole: 16 Ethernet tiles × 100 Gbps = 1.6 Tbps per chip; commodity Ethernet switches | confirmed | hw-architecture, SemiAnalysis-WH |
| Scale-up Interconnect | Blackhole: 10 × 400 Gbps Ethernet = 4 Tbps per chip; 4× QSFP-DD (800 Gbps each) | confirmed | hw-architecture, SemiAnalysis-BH |
| Scale-up Interconnect | Ethernet tiles appear in NoC grid; cross-chip ops use same noc_async_read/write API | confirmed | hw-architecture, METALIUM_GUIDE.md |
| Scale-up Interconnect | T3000: 8× Wormhole in 2×4 mesh; Blackhole Galaxy: 32× Blackhole in 4×8 mesh | confirmed | hw-architecture |
| Scale-up Interconnect | **Galaxy Blackhole accelerator fabric: 10 × 400 GbE per ASIC — vendor-stated 32 TB/s (= 16 TB/s unidirectional; The Register quotes ~100 Tbps intra-node, unreconciled)** | confirmed (vendor figure); bidirectionality is repo arithmetic | tenstorrent.com/en/hardware/galaxy, The Register 2026-04-28 |
| Scale-out Interconnect | Commodity 400G/800G Ethernet switches (no proprietary switching ASIC; standard Ethernet infrastructure) | confirmed | hw-architecture, SemiAnalysis |
| Scale-out Interconnect | Wormhole cluster via 16× 100 GbE per chip; Blackhole via QSFP-DD 800G | confirmed | hw-architecture |
| Scale-out Interconnect | **Galaxy Blackhole: up to 56 × 800 GbE QSFP-DD per node — vendor-stated 11.2 TB/s (= 5.6 TB/s unidirectional). Supersedes the stale "~1 TBps aggregate" figure.** | confirmed (vendor figure) | tenstorrent.com/en/hardware/galaxy |
| Scale-out Interconnect | **Galaxy Blackhole max system scale: vendor-stated 144 nodes / >4,000 chips; 36-Galaxy supercluster described as running at TT-Deploy (2026-05-04); scaling efficiency not independently verified** | vendor claim | tenstorrent.com/en/hardware/galaxy, TT-Deploy, The Register 2026-04-28 |

## Update Log

**2026-08-08** — Applied the Galaxy Blackhole GA (2026-04-28) hardened spec and public list pricing; corrected the pre-existing "~1 TBps aggregate" Galaxy scale-out figure and the "140 Tensix / 745 TOPS" Blackhole SKU line (shipping cards are 120 cores / 664 TFLOPS BLOCKFP8); added p100a (28 GB @ 448 GB/s) and flagged p300 as delisted; added Metal 2.0 / Device 2.0 / DataflowBuffer / multi-config layer / emule fiber-engine software rows and the TT-Lang listing; added Quasar rows with everything except the (medium-confidence, Oct-2023) Samsung SF4X process node marked not disclosed.
