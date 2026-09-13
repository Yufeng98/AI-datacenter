# Meta MTIA — Software Stack Investigation

## Summary

MTIA has a well-documented, PyTorch-first software stack. The key innovation is using Triton as a hardware-agnostic kernel language to target MTIA alongside GPUs, enabling code reuse. Meta has partially open-sourced the PyTorch and Triton backends.

---

## Full Stack

```
PyTorch (eager mode + torch.compile + torch.export)
         |
         v
TorchDynamo  ─────────────────────────>  (graph capture)
         |
         v
Torch FX IR  (graph representation)
         |
         v
TorchInductor + MTIA graph compiler
  - operator lowering to MTIA primitives
  - fusion, scheduling
         |
         v
Triton-MTIA Compiler Backend
  - Triton Python DSL → MLIR → MTIA native ISA
  - LLVM toolchain for RISC-V portions
         |
         v
MTIATensor + Device Memory Allocator
         |
         v
MTIA Streaming Interface (runtime API)
         |
         v
MTIA Firmware Driver
         |
         v
MTIA Hardware (PE grid + SRAM + LPDDR5)
```

---

## Training Stack

| Component | Detail |
|---|---|
| Framework | PyTorch |
| Backend | NVIDIA GPU (H100/A100) for training |
| MTIA role | Inference only (v1/v2); training chip in development |

---

## Compiler Architecture

### Graph Compiler (High Level)
- TorchDynamo captures PyTorch eager graphs
- Torch FX IR represents computation graph
- TorchInductor lowers to MTIA primitives via custom backend

### Kernel Compiler (Low Level)
- Triton-MTIA: Meta extended Triton language to emit MTIA code
- Triton → MLIR → MTIA-specific lowering → LLVM → RISC-V assembly for control + MMA instructions
- KernelEvolve (2025): agentic LLM-based kernel generation for MTIA operators

### Key Compiler Capabilities
- Operator fusion (embed + MLP fusion for DLRM)
- Sparsity-aware code generation (7× sparse compute on v2)
- Static shape compilation via torch.export

---

## Runtime

- **MTIA Streaming Interface:** manages DMA, async dataflow scheduling
- **MTIATensor:** custom PyTorch tensor type backed by MTIA device memory
- **Firmware driver:** kernel driver for PCIe communication

---

## Inference Serving

- Supports vLLM (emerging LLM use case)
- Supports torch.compile and torch.export
- Deployed in Meta's production inference serving for:
  - Facebook/Instagram feed ranking
  - Ads recommendation (DLRM-class)
  - Emerging: Llama inference

---

## Openness

- PyTorch MTIA backend: upstreamed to PyTorch (open)
- Triton-MTIA backend: partial open (via Triton project)
- Chip design and firmware: closed
- ISCA 2025 academic paper: public (https://dl.acm.org/doi/10.1145/3695053.3731409)

---

## Key Software Design Decisions

1. **Triton as kernel language**: hardware-agnostic, reuse GPU kernels on MTIA with minimal changes
2. **PyTorch eager mode support**: unlike many ASICs that require static graph export; MTIA supports eager for developer flexibility
3. **Async dataflow runtime**: matches hardware's async execution model
4. **vLLM support**: positions MTIA for LLM inference as Meta deploys Llama models

---

## MTIA v2 Update

*Added 2026-04-05 based on ISCA 2025 paper (dl.acm.org/doi/10.1145/3695053.3731409), KernelEvolve ArXiv (arxiv.org/html/2512.23236v1), and Meta engineering blog (engineering.fb.com/2024/08/22/ml-applications/meta-mtia-hardware-co-design/).*

### KernelEvolve — Extended Detail (ISCA 2025 / ArXiv 2512.23236)

KernelEvolve was briefly noted in the original investigation. The ISCA 2025 paper provides substantially more detail:

**Architecture:**
- Reframes kernel optimization as a **graph search and evolution process**
- Multi-level programming abstractions spanning the full software-hardware stack: Triton DSL → MLIR → MTIA native ISA → RISC-V assembly
- Multi-agent system with specialized sub-agents:
  - **Context memory sub-agent**: analyzes dynamic runtime information to inform search
  - **Deep search sub-agent**: handles complex optimization scenarios requiring longer horizon search
  - **Evolution sub-agent**: mutates candidate kernels and evaluates fitness against latency/throughput targets

**Results reported at ISCA 2025:**
- KernelEvolve matched or exceeded hand-tuned kernels for key DLRM operators on MTIA 2i
- Reduces the engineering cost of porting new operators to MTIA from weeks to hours

### Model-Chip Co-Design (ISCA 2025)

The ISCA 2025 paper's primary contribution beyond hardware specs is documenting Meta's **model-chip co-design** methodology — iteratively modifying both the model architecture and compiler to extract hardware utilisation:

- **Operator fusion**: fused embed + MLP layer sequences were co-designed with hardware register file sizing
- **Sparsity-model co-design**: pruning schedules for recommendation models were tuned to match MTIA v2's structured sparsity hardware (2:4 sparsity pattern), yielding the 7× throughput figure
- **Static-shape compilation**: models retrained or fine-tuned to emit fixed shapes to enable torch.export ahead-of-time compilation (avoiding shape-dispatch overhead)
- **Embedding table sharding**: embedding table partitioning strategy co-designed with MTIA 2i's 384 KB PE-local SRAM and 256 MB shared SRAM to maximise hot-row cache hit rate

### Firmware Cadence as a Software Delivery Mechanism

MTIA v2's production experience revealed that **firmware updates are a first-class software delivery vector**:
- 23 fleet-wide firmware-bundle releases shipped in 2024
- Firmware updates delivered: new operator support, sparsity configuration tuning, PCIe DMA scheduling improvements
- This allowed Meta to improve production performance without hardware respins — unusual for an ASIC at this scale

### Stack Evolution for MTIA 300+ Series

The announced MTIA 300/400/450/500 chips (Broadcom partnership, 2026–2027) introduce workload expansions that will require software stack changes:

| Concern | Current (MTIA 2i) | Future (MTIA 300+) |
|---|---|---|
| Memory type | LPDDR5 | HBM |
| Memory allocator | MTIATensor (LPDDR5-aware) | Will need HBM pool management |
| Primary workload | DLRM/recommendation inference | GenAI inference + training |
| Training stack | GPU only (H100/A100) | MTIA 300 targets training |
| Parallelism | Single-chip (72/rack) | Multi-chip parallelism for LLM likely required |

The transition to HBM and LLM inference on MTIA 450/500 will likely require adding tensor-parallel or pipeline-parallel scheduling to the MTIA Streaming Interface, which currently assumes single-chip DLRM execution.

---

## MTIA 300 Software Stack Update — ISCA 2026

*Added 2026-08-08. Primary source: "MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines", ISCA 2026 Industry Track, Session 5B, presented 2026-06-30 (https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf). Verified from the PDF text; cross-checked against the ISCA 2026 program and Meta's AI-system-codesign publications index.*

MTIA 300 is the first MTIA generation that trains, and the stack changes accordingly: a backward-pass compiler path, distributed-training primitives, and — the genuinely new component — a **collective communications library that programs on-die Message Engines and in-package NIC chiplets**.

### Revised stack for MTIA 300

```
PyTorch  (eager + torch.compile + torch.export)
  + FSDP2 | DTensor | TorchRec | XFormers          <-- new for MTIA 300 training
         |
         v
TorchDynamo  (forward graph capture)
         |
         +--> AOTAutograd  (backward graph generation)   <-- new for training
         |
         v
TorchInductor / MTIA graph compiler
  - operator lowering, fusion
  - ILP-based graph scheduling for peak-memory reduction   <-- new
  - activation rematerialization                            <-- new
  - Triton codegen
         |
         v
Kernel layer:  Triton  |  hand-written C++  |  coding-agent-generated kernels
         |
         v
Runtime:  CUDA-compatible runtime API (eager AND graph modes)   <-- new framing
  + MTIATensor / device allocator
  + HCCL  (MTIA Collective Communications Library)              <-- NEW COMPONENT
         |
         v
CPU-C control core  --dispatches WQE subgraphs-->  16 Message Engines
         |                                                |
         |                                          RDMA verbs (rdma_core, ibv_create_qp)
         v                                                v
MTIA 300 compute chiplet (72 PEs)          2x network chiplet (12x 800 Gbps RoCE NICs)
```

### HCCL — the new named component

**HCCL** is Meta's collective communications library for MTIA 300 (the MTIA analogue of NCCL/RCCL, but targeting Message Engines rather than compute kernels).

- HCCL constructs **work packets** and **subgraphs of WQEs** with opcodes **SEND / RECV / WRITE / WAIT / SET / REDUCE**.
- The chip's control core, **CPU-C**, dispatches those WQE subgraphs to the **16 Message Engines**.
- The Message Engines drive the in-package **NIC chiplets** through standard **RDMA verbs** (`rdma_core`, `ibv_create_qp`).
- **Compute and collectives can be captured in a single graph.** Collectives traced via `torch.export` or `torch.compile` live in the same graph as the compute they overlap with — this is what makes the 2.8 TB/s of Message Engine reduction throughput schedulable rather than merely available.
- **Static-shaped collectives are fully supported.** **Dynamic-shape collectives and device-resident AllToAll with dynamic send/recv counts remain WIP** — a real limitation for MoE routing, where expert assignment counts are data-dependent.

This is the software counterpart to the hardware headline: the collective-offload engines are only useful because HCCL can express a collective as a WQE subgraph that the compiler places inside the same captured graph as the GEMMs.

### Compiler additions for training

- **AOTAutograd** generates the backward graph after TorchDynamo captures the forward graph — the standard PyTorch training compilation path, now wired to the MTIA backend.
- **ILP-based graph scheduling** for peak-memory reduction: the compiler formulates operator ordering as an integer linear program to minimise live-tensor high-water mark, which matters because MTIA 300's 216 GB HBM must hold optimizer state and activations for training rather than just inference weights.
- **Activation rematerialization** (recompute instead of store) trades FP8/BF16 GEMM throughput — of which MTIA 300 has a great deal — against HBM capacity.

### Distributed-training integration

Meta reports MTIA 300 running PyTorch-native distributed training with:

- **FSDP2** — fully-sharded data parallel, second-generation API
- **DTensor** — distributed tensor sharding abstraction
- **TorchRec** — recommendation-model sharding (embedding tables), the workload MTIA 300 was built for
- **XFormers** — attention kernels

### Runtime

MTIA 300 exposes a **CUDA-compatible runtime API** supporting both **eager and graph modes**. This is a notable framing shift: earlier MTIA generations were described in terms of the bespoke "MTIA Streaming Interface", whereas MTIA 300's runtime is presented as CUDA-like, reducing porting friction for code written against GPU idioms.

### Kernel authoring

Three routes on MTIA 300:

1. **Triton** — the hardware-agnostic route, unchanged in spirit from earlier generations (see also "Triton for MTIA: Bridging the Programming Model Gaps for Custom AI Accelerators", IEEE Micro 2026).
2. **C++** — hand-written kernels for cases Triton cannot express well.
3. **Coding agents** — Meta reports using coding agents for automated kernel generation. The KernelEvolve line of work is now published: "KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta", ISCA 2026 Industry Track Session 4B, Monday 2026-06-29, 17:50–18:10. Previously this repo cited KernelEvolve only as an uncited bullet plus an ArXiv preprint.

### Numerics gaps that are software-visible

- **MX4 and NVFP4 with row-wise/block-wise scaling are not natively supported on MTIA 300 and fall back to RISC-V execution.** For a survey this is the single most important software-hardware interface note on MTIA 300: a model quantized to MX4 or NVFP4 will run, but on the scalar/vector path rather than the Dot Product Engine. Meta states native MX4 support was added in **MTIA 400**.
- **Eager mode is host-bottlenecked** — Python interpreter overhead, dynamic dispatch, and device-host communication dominate, and this does not improve as the silicon gets faster. MTIA's long-standing "eager mode works, unlike most ASICs" selling point now comes with an explicit scaling caveat from Meta.
- **Numerical-parity / convergence-debugging tooling** for cross-platform comparison exists but Meta says it "still require[s] time to mature" — relevant for anyone assessing how portable a training run is between GPU and MTIA.

### Serving

LLM inference on MTIA 300 is served through **vLLM**, evaluated on **DeepSeek-R1** under the **InferenceMax** benchmark with 8-accelerator **TP8-TP8** and **DP8-EP8** configurations and **BF16 attention/KV-cache + FP8 MoE** precision. This confirms vLLM as a first-class MTIA serving path, not merely "emerging".

### Related new publications

- **"KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta"** — ISCA 2026 Industry Track, Session 4B, 2026-06-29, 17:50–18:10.
- **"LoKA: Low-precision Kernel Applications for Recommendation Models At Scale"** — ISCA 2026 (listed on Meta's publications index). Not previously in this repo.
- **"Triton for MTIA: Bridging the Programming Model Gaps for Custom AI Accelerators"** — IEEE Micro 2026 (listed on Meta's publications index). Not previously in this repo.

### Superseded prediction

The 2026-04-05 entry above speculated that "the transition to HBM and LLM inference on MTIA 450/500 will likely require adding tensor-parallel or pipeline-parallel scheduling to the MTIA Streaming Interface, which currently assumes single-chip DLRM execution." **This is now resolved and arrived a generation earlier than predicted**: MTIA 300 already ships FSDP2/DTensor/TorchRec distributed training plus HCCL for collectives, and the runtime is presented as a CUDA-compatible API rather than the single-chip Streaming Interface. The prediction is left in place for the historical record.

### Sources for this section

- https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf (primary)
- https://aisystemcodesign.github.io/ (Meta AI system co-design publications index — LoKA, Triton for MTIA, KernelEvolve)
- https://iscaconf.org/isca2026/program/ (Sessions 4B and 5B)
