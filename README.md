# AI Datacenter Accelerator Research Corpus

Open research corpus behind the survey **"Balancing Generality and Specialization: A Survey on AI Datacenter Hardware Architecture"** (Yufeng Gu, Jiazhen Wang, Reetuparna Das — University of Michigan).

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

This repository is consumed as the `public/` git submodule of the (private) survey repository, whose Claude Code skills generate and update these files.

## Chips Investigated

| Chip | Device Class | Key Differentiator |
|------|-------------|-------------------|
| **Major GPUs** | | |
| [nvidia-gpu](chips/nvidia-gpu/summary.md) | GPU | CUDA ecosystem, Tensor Cores, NVLink |
| [amd-gpu](chips/amd-gpu/summary.md) | GPU | ROCm/HIP, CK/AITER, Infinity Fabric |
| **Systolic Array / TPU-like** | | |
| [google-tpu](chips/google-tpu/summary.md) | Systolic Array | JAX/XLA, MXU, ICI 3D torus |
| [aws-neuron](chips/aws-neuron/summary.md) | Systolic Array | NKI kernels, SBUF/PSUM, NeuronLink |
| [huawei-ascend](chips/huawei-ascend/summary.md) | Systolic Array | Da Vinci core, CANN, CloudMatrix |
| **Mixed Architecture** | | |
| [intel-gaudi](chips/intel-gaudi/summary.md) | Mixed (GEMM+TPC) | Dual-engine, integrated RoCE |
| **Dataflow / Reconfigurable** | | |
| [sambanova](chips/sambanova/summary.md) | Dataflow/Reconfigurable | Spatial compiler, 3-tier memory |
| **RISC-V Based** | | |
| [tenstorrent](chips/tenstorrent/summary.md) | Tensix RISC+SFPU | NoC data movement, fully open-source |
| [esperanto](chips/esperanto/summary.md) | Many-core RISC-V | 1,088 RISC-V cores, <20W TDP |
| [sophgo](chips/sophgo/summary.md) | RISC-V AI Accelerator | Open-source RISC-V, BM1684/SG2042 |
| **Chinese GPU Ecosystem** | | |
| [biren](chips/biren/summary.md) | GPU-like AI Accelerator | BR100, China GPU ecosystem |
| [enflame](chips/enflame/summary.md) | Data Transfer Unit | DTU architecture, GCU cloud |
| [muxi](chips/muxi/summary.md) | GPU (China) | MXMACA runtime, CUDA-compatible |
| [mthreads](chips/mthreads/summary.md) | GPU (China) | MUSA stack, S80/S4000 |
| [hygon-dcu](chips/hygon-dcu/summary.md) | GPU/DCU (China) | ROCm-compatible, HIP ecosystem |
| [tianshu-zhixin](chips/tianshu-zhixin/summary.md) | GPU (China) | TianGai BI-V150, CUDA-like TXARCH |
| [xiwang](chips/xiwang/summary.md) | AI Accelerator (China) | 曦望, sparse compute focus |
| [cambricon](chips/cambricon/summary.md) | Neural Processor | BANG C, scratchpad model |
| [kunlunxin](chips/kunlunxin/summary.md) | AI Accelerator (Baidu) | XPU, PaddlePaddle native |
| [alibaba-t-head](chips/alibaba-t-head/summary.md) | AI Accelerator (China) | Zhenwu/Hanguang, Qwen inference |
| **Inference-Focused** | | |
| [meta-mtia](chips/meta-mtia/summary.md) | Inference Accelerator | Custom Meta silicon |
| [qualcomm](chips/qualcomm/summary.md) | Cloud AI Inference | Hexagon AI cores, Cloud AI 100/200 |
| [tesla-fsd](chips/tesla-fsd/summary.md) | Edge Inference SoC | FSD chip NNA |
| [furiosa](chips/furiosa/summary.md) | NPU (Inference) | Tensor Contraction Processor, RNGD |
| [rebellions-atom](chips/rebellions-atom/summary.md) | Inference Accelerator | ATOM/REBEL, Korea, Samsung HBM |
| [etched-sohu](chips/etched-sohu/summary.md) | Transformer-Specific ASIC | Transformer-only silicon, no CUDA |
| **Deterministic / Wafer-Scale** | | |
| [groq](chips/groq/summary.md) | Deterministic (TSP) | All-SRAM, compiler-is-stack |
| [cerebras](chips/cerebras/summary.md) | Wafer-Scale Engine | WSE-3, 44GB SRAM, weight streaming |
| **Photonic** | | |
| [lightmatter](chips/lightmatter/summary.md) | Photonic Compute + Interconnect | Envise accelerator + Passage interconnect |
| [q-ant](chips/q-ant/summary.md) | Photonic NPU | Photonic inference, Germany |
| [luminous](chips/luminous/summary.md) | Photonic-Electronic Hybrid | Optical matrix multiply |
| **In-Memory / PIM** | | |
| [d-matrix](chips/d-matrix/summary.md) | In-Memory Compute | SRAM-based digital compute |
| [sk-hynix-aim](chips/sk-hynix-aim/summary.md) | Processing-in-Memory | GDDR6-AiM |
| [samsung-aquabolt-pim](chips/samsung-aquabolt-pim/summary.md) | Processing-in-Memory | HBM-PIM |
| [untether-ai](chips/untether-ai/summary.md) | At-Memory Compute | Cascade inference chip (defunct) |
| [mythic](chips/mythic/summary.md) | Analog In-Memory Compute | Flash-based analog MAC array |
| [rain-ai](chips/rain-ai/summary.md) | Neuromorphic AI | Resistive memory, spike-based |
| **Custom Silicon** | | |
| [microsoft-maia](chips/microsoft-maia/summary.md) | Cloud AI Accelerator | Azure-internal, MX formats |
| [openai-broadcom](chips/openai-broadcom/summary.md) | Custom AI Accelerator | Project Titan, H2 2026 target |
| [tesla-dojo](chips/tesla-dojo/summary.md) | Training Accelerator | 440MB SRAM, TTPoE |
| **MIMD** | | |
| [graphcore](chips/graphcore/summary.md) | MIMD/BSP (IPU) | BSP model, tile-based |
| [pezy](chips/pezy/summary.md) | MIMD Manycore | 1,000+ in-order RISC cores, Green500 |

Additional chip folders not yet in the table above: `ibm-spyre`, `nextsilicon-maverick`, `preferred-networks-mn-core`, `spinncloud-spinnaker2`, `stream-computing`, `tecorigin`, `tsingmicro`, `vastaitech`.

## Conventions

- Throughput figures are vendor-reported **dense** peak values unless a cell says otherwise; a fused multiply-add counts as two operations.
- Interconnect bandwidth is the aggregate bidirectional bandwidth per accelerator as reported by the vendor.
- Every `hw-architecture.md` ends with a `## Sources` list; `research/<chip>/investigations/*.yaml` sidecars carry per-layer source URLs.
