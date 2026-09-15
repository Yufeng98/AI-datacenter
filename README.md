# AI Datacenter Accelerator Research Corpus

Open research corpus behind the survey **"Balancing Generality and Specialization: A Survey on AI Datacenter Hardware Architecture"** (Yufeng Gu, Jiazhen Wang, Reetuparna Das — University of Michigan).

**Project page: <https://yufeng98.github.io/ai_datacenter_survey/>** — the survey abstract plus an interactive version of its scaling figure: frontier-model parameter counts against per-accelerator dense FP16/BF16 compute, DRAM bandwidth and capacity, and scale-up interconnect bandwidth. The chart is built from the chip data in this corpus, and each point links back to the `hw-architecture.md` it was drawn from.

For each AI chip, this corpus discovers, investigates, and documents:

- **Software stack**: framework integration, compiler/IR, op libraries, kernel libraries, runtime, driver/firmware, communication libraries, assembler/ISA
- **Hardware architecture**: compute engine, data path, on-chip memory, off-chip memory, host interface, scale-up interconnect, scale-out interconnect
- **Programming model rationale**: why the software stack is designed the way it is, given the hardware constraints

## Repository Layout

```
AI-datacenter/
├── chips/                          # Final deliverables per chip
│   └── <chip>/
│       ├── summary.md              # Comprehensive narrative
│       ├── programming-model.dot/png # Programming model figure (Graphviz)
│       ├── hw-architecture.md      # Hardware architecture write-up
│       ├── hw-architecture.dot/png # Hardware architecture figure
│       └── layer-table.md          # Per-chip layer mapping
└── research/                       # Raw investigation notes
    ├── status.yaml                 # Pipeline state (paths are relative to this repo root)
    ├── TODO-next-wave.md           # Open research items
    └── <chip>/
        ├── search-results.md       # Discovered resources
        ├── investigations/         # Deep-dive reports + YAML sidecars
        └── codex-reviews/          # Codex review feedback
```

This repository is consumed as the `public/` git submodule of the (private) survey repository, whose Claude Code skills generate and update these files. The [project page](https://yufeng98.github.io/ai_datacenter_survey/) is generated from the same data and published separately.

## Chips Investigated (50)

| Chip | Device Class | Key Differentiator |
|------|-------------|-------------------|
| **GPU — SIMT cores + tensor/matrix units** | | |
| [nvidia-gpu](chips/nvidia-gpu/summary.md) | GPU | CUDA ecosystem, Tensor Cores, NVLink |
| [amd-gpu](chips/amd-gpu/summary.md) | GPU | ROCm/HIP, CK/AITER, Infinity Fabric |
| [biren](chips/biren/summary.md) | GPU | BR100, SIMT C-Warp |
| [hygon-dcu](chips/hygon-dcu/summary.md) | GPU | ROCm-compatible (DTK/HIP), GCN-derived CU, xGMI |
| [muxi](chips/muxi/summary.md) | GPU | MXMACA runtime, CUDA-compatible |
| [mthreads](chips/mthreads/summary.md) | GPU | MUSA stack, S80/S4000 |
| [tianshu-zhixin](chips/tianshu-zhixin/summary.md) | GPU | TianGai BI-V150, CUDA-compatible IXUCA stack |
| [xiwang](chips/xiwang/summary.md) | GPU | SIMT GPGPU, warp-32, sparse-compute focus |
| **NPU — heterogeneous matrix, vector, scalar engines around shared scratchpad** | | |
| [google-tpu](chips/google-tpu/summary.md) | Systolic Array | JAX/XLA, MXU, ICI 3D torus |
| [aws-neuron](chips/aws-neuron/summary.md) | Systolic Array | NKI kernels, SBUF/PSUM, NeuronLink |
| [huawei-ascend](chips/huawei-ascend/summary.md) | Systolic Array | Da Vinci core, CANN, CloudMatrix |
| [intel-gaudi](chips/intel-gaudi/summary.md) | Mixed (GEMM+TPC) | Dual-engine, integrated RoCE |
| [microsoft-maia](chips/microsoft-maia/summary.md) | NPU (Tile-based) | Azure-internal, MX formats |
| [qualcomm](chips/qualcomm/summary.md) | NPU (Hexagon) | HMX/HVX/Q6 cores, Cloud AI 100/200 |
| [cambricon](chips/cambricon/summary.md) | Neural Processor | BANG C, scratchpad model |
| [alibaba-t-head](chips/alibaba-t-head/summary.md) | NPU (Tensor/Pooling/Memory Engine) | Hanguang/Zhenwu, Qwen-optimized inference |
| [kunlunxin](chips/kunlunxin/summary.md) | NPU (Systolic + SIMD) | XPU, PaddlePaddle-native, XPU-Link supernode |
| [furiosa](chips/furiosa/summary.md) | NPU (Tensor Contraction) | Tensor Contraction Processor, RNGD |
| [sophgo](chips/sophgo/summary.md) | NPU (TPU-style SIMD) | BM1690 TPU core, TPU-MLIR static scheduling |
| [vastaitech](chips/vastaitech/summary.md) | NPU (AI Engine + VDSP) | VUCA: AI engine + VDSP vector cores + video codec |
| [tecorigin](chips/tecorigin/summary.md) | NPU (Heterogeneous Many-Core) | Sunway-lineage SPE array, SDAA, explicit RMA/broadcast |
| [stream-computing](chips/stream-computing/summary.md) | NPU (RISC-V + MME/VME) | NeuralScale core: RISC-V scalar + MME/VME engines |
| [tesla-fsd](chips/tesla-fsd/summary.md) | NPU (Systolic, Fixed-ISA) | Systolic NNA, 8-instruction static-schedule ISA |
| [openai-broadcom](chips/openai-broadcom/summary.md) | NPU (Systolic, reported) | Jalapeño, systolic tiled matrix engine (unconfirmed) |
| **Spatial Dataflow — computation graph mapped onto distributed resources** | | |
| [tenstorrent](chips/tenstorrent/summary.md) | Tensix RISC+SFPU (PE array) | NoC data movement, fully open-source |
| [meta-mtia](chips/meta-mtia/summary.md) | PE Array | PE grid (RISC-V+MMA), custom Meta silicon |
| [graphcore](chips/graphcore/summary.md) | MIMD/BSP (IPU, PE array) | BSP model, tile-based |
| [tesla-dojo](chips/tesla-dojo/summary.md) | PE Array (Tile) | 440MB SRAM, TTPoE |
| [cerebras](chips/cerebras/summary.md) | Wafer-Scale PE Array | WSE-3, 44GB SRAM, weight streaming |
| [ibm-spyre](chips/ibm-spyre/summary.md) | PE Array (Systolic Dataflow) | 8×8 systolic PT array per corelet, ring NoC |
| [enflame](chips/enflame/summary.md) | PE Array (Static Dataflow) | DTU architecture, GCU-DARE routing fabric |
| [preferred-networks-mn-core](chips/preferred-networks-mn-core/summary.md) | PE Array (Lockstep SIMD) | 4,096-PE lockstep SIMD, block-FP, no cache/branch predictor |
| [esperanto](chips/esperanto/summary.md) | Many-core RISC-V (PE array) | 1,088 RISC-V cores, <20W TDP |
| [pezy](chips/pezy/summary.md) | MIMD Manycore | 1,000+ in-order MIMD RISC cores, Green500 |
| [sambanova](chips/sambanova/summary.md) | Reconfigurable Dataflow | Spatial compiler, 3-tier memory |
| [rebellions-atom](chips/rebellions-atom/summary.md) | Reconfigurable Dataflow (CGRA) | ATOM/REBEL, Samsung HBM |
| [tsingmicro](chips/tsingmicro/summary.md) | Reconfigurable Dataflow (CGRA) | CGRA tile array (RPU), RISC-V control core per tile |
| [nextsilicon-maverick](chips/nextsilicon-maverick/summary.md) | Reconfigurable Dataflow | Runtime-JIT reconfigurable grid, no fixed instruction stream |
| [groq](chips/groq/summary.md) | Functional-Slice Streaming | All-SRAM, compiler-is-stack |
| [etched-sohu](chips/etched-sohu/summary.md) | Functional-Slice (Hardwired) | Transformer-only silicon, hardwired Attention/FFN pipeline, no CUDA |
| **Compute-in-Memory — arithmetic within or beside memory arrays** | | |
| [d-matrix](chips/d-matrix/summary.md) | In-Memory Compute (Digital) | SRAM-based digital compute |
| [sk-hynix-aim](chips/sk-hynix-aim/summary.md) | Processing-in-Memory | GDDR6-AiM |
| [samsung-aquabolt-pim](chips/samsung-aquabolt-pim/summary.md) | Processing-in-Memory | HBM-PIM |
| [mythic](chips/mythic/summary.md) | Analog In-Memory Compute | Flash-based analog MAC array |
| [untether-ai](chips/untether-ai/summary.md) | At-Memory Compute | Cascade inference chip (defunct) |
| [rain-ai](chips/rain-ai/summary.md) | In-Memory Compute (Digital, defunct) | Digital in-SRAM MAC (D-IMC), pivoted from spiking/ReRAM |
| **Photonic / Optical Compute** | | |
| [lightmatter](chips/lightmatter/summary.md) | Photonic Compute + Interconnect | Envise accelerator + Passage interconnect |
| [q-ant](chips/q-ant/summary.md) | Photonic NPU | Photonic inference |
| [luminous](chips/luminous/summary.md) | Photonic-Electronic Hybrid (defunct) | WDM photonic interconnect, electronic (CMOS) matmul |
| **Neuromorphic — event/spike-driven execution** | | |
| [spinncloud-spinnaker2](chips/spinncloud-spinnaker2/summary.md) | Event-Driven Neuromorphic Manycore | ARM Cortex-M4F + MAC-array PE, per-PE DVFS |

## Conventions

- Throughput figures are vendor-reported **dense** peak values unless a cell says otherwise; a fused multiply-add counts as two operations.
- Interconnect bandwidth is the aggregate bidirectional bandwidth per accelerator as reported by the vendor.
- Every `hw-architecture.md` ends with a `## Sources` list; `research/<chip>/investigations/*.yaml` sidecars carry per-layer source URLs.
