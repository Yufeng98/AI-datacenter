# Stream Computing (希姆计算) NeuralScale Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*
*Representative products: STCP920 / STCP950L / STCP950P / STCP980L / STCP980P PCIe inference cards (all one silicon generation, P920)*

---

## Overview

**Stream Computing Inc. (希姆计算**, operating entity 广州希姆半导体科技有限公司, founded 2019) builds
**NeuralScale**, a general-purpose programmable NPU architecture that extends RISC-V with vendor-custom
vector and matrix instructions. It ships as **PCIe 4.0 ×16 datacenter *inference* cards** — five SKUs, all
built on the same first-generation **P920** SoC (TSMC 12 nm, 400 mm², 32 NeuralScale cores, 128 TFLOPS FP16
/ 256 TOPS INT8 @ 1.0 GHz, 130 W chip TDP, 16 GB LPDDR4X).

Geography, because secondary sources get it wrong: the CARRV'21 paper bylines "Stream Computing Inc.,
**Beijing**" and ByteDance's benchmark names "Beijing Stream Computing Technology Co., LTD"; the current
operating entity is in **Guangzhou** (粤ICP备2024180922号), with offices also in Beijing, Shanghai and
Hangzhou. Origin Beijing, current entity Guangzhou — **not a Shanghai company**.

**Maturity: shipping and deployed, first generation only.** Silicon returned in 2021, first cards shipped
2021, volume orders 2022. ByteDance's ByteMLPerf repository hosts the vendor's own Apache-2.0 backend, whose
README states the STCP920 "is in mass production, and has completed a batch of shipments to users", with
reproducible QPS and accuracy. **No second-generation silicon is public**: the only trace of one is that the
shipping MLTC compiler accepts `--arch=npu-v2` — a flag, not a chip.

Three things make this part worth a section in the survey:

1. **A genuinely different core model.** A NeuralScale core is a RISC-V scalar core (AndesCore N25F, RV32G,
   5-stage in-order) issuing **up to three instructions per cycle to three decoupled engines** — a vector
   MAC engine (VME), a matrix MAC engine (MME) and a memory-transfer engine (MTE). It is per-core MIMD, not
   SIMT, and not a fixed-function systolic NPU.
2. **A vendor-custom RISC-V extension in shipping silicon.** RVV v0.8 plus **53 custom instructions** on the
   custom-3 opcode (OP-VE) and **22 unprivileged shape CSRs**. Separately — and this must not be conflated —
   the company chairs work on an **open** RISC-V matrix-extension proposal that the shipping silicon does
   *not* implement.
3. **A documented compiler-family migration.** The 2021–2023 stack was **TensorTurbo** (TVM + LLVM +
   stcDNN); the current stack is **MLTC** (MLIR + IREE + torch-mlir). Very few vendors have publicly changed
   compiler families mid-product.

And one thing must be stated plainly: **the card is bandwidth-starved for LLM decode**. 16 GB of LPDDR4X at
~108 GB/s against 128 TFLOPS FP16 is a CNN/NLP-era inference part retrofitted for LLMs. The vendor's own
model table shows Qwen2-72B-Instruct on **16 cards at 4.98 tok/s**.

---

## Software Stack

### Delivery

Everything ships as **STCRP** (Stream Computing System Reference Platform), currently **V1.12.1**. There are
no public SDK repositories; packages come from vendor technical support as `.deb` / `.rpm` repo packages plus
Python wheels. OS support: Ubuntu 22.04, Ubuntu 25.04, Kylin V10 (银河麒麟); Python 3.10 for MLTC.

### Framework Integration

- **PyTorch** is the primary path, two ways: `torch.compile(model, backend=mltc_backend)`, or
  `from_torch(nn.Module | ExportedProgram, ...)` after `torch.export`, with `Dim(...)` dynamic shapes on
  parallel/reduce axes.
- **ONNX** (`OnnxToStc`, 156 ops) and **TensorFlow** (`TfToStc`, 106 TF1 GraphDef ops) have dedicated
  importers.
- **PaddlePaddle** has a Level-I compatibility certification with Baidu and an open `STCPaddleModelZoo`
  (~20 models) — but that zoo is a **TensorTurbo-era artifact**, not current-stack.
- **HuggingFace LLMs** go through STC_LLM, which loads `bin` / `safetensors` directly.
- **Notably absent:** no vLLM fork, no SGLang, no Triton kernel language, no ONNX Runtime execution provider.

### Compiler / IR — MLTC

**MLTC** (Multi-Level Tensor Compiler) is a proprietary MLIR compiler. Its design is organised around the
one problem this hardware creates: the vendor's own rationale for the **GOAT** tiling framework is that
*"when writing operators for NeuralScale, most of the code is managing the multi-level storage hierarchy and
the data movement between levels."* GOAT tiles over **LLB → MC → L1** and is operator-agnostic, so the
compiler trunk after graph optimisation has no per-operator special-casing. Graph-level ops are deliberately
atomic — there are no fused operators in the graph IR.

The `stc.*` dialect has **164 ops**, including `stc.attention` / `stc.attention_lm`,
`stc.kv_cache_load` / `stc.kv_cache_store`, `stc.matmul` vs `stc.matmul_vme` (high-precision VME matmul),
and a built-in **collective family** (`stc.device_broadcast`, `stc.device_allconcat`,
`stc.device_reduce_*`, `stc.device_sync`).

Output is a **`.vmfb`** file compiled with `-arch=npu-v1`. The `.vmfb` container is IREE's VM FlatBuffer,
`--graph-partition-factor` is documented in terms of IREE `dispatch.workgroup`s, `OutputType.TORCH` and 141
torch-mlir `Aten*Op` class names appear in the op-support tables — so **MLTC is an IREE + torch-mlir
derivative**, a materially different lineage from Sophgo's TPU-MLIR. The vendor never states the upstream,
so the exact provenance and revisions are **not disclosed**.

### Kernel Language and Compiler — SHC and `stcc`

Below the graph compiler the stack is an unusually faithful CUDA clone. **SHC** (Stream Computing
Heterogeneous C++, `.hc`) is full C++17 plus `<<< >>>` kernel launch, `__global__` / `__device__`, memory
space qualifiers `__local__` (L1 Buffer) and `__shared__` (LLB), built-ins `CoreID` / `CoreNum`, and a
`sync()` barrier. **`stcc`** compiles host and device code in one invocation exactly like `nvcc`; the
`--rtlib=compiler-rt` flag, Clangd integration and an RV32 device runtime make a Clang/LLVM downstream
near-certain, though the vendor does not say so.

The abstraction level is *lower* than Triton or even CUDA C — the kernel author writes ISA intrinsics and
programs shape CSRs by hand:

```c
CONFIG_VE_BC_CSR(shape1, shape2, 0, 0);
memul_mm((__fp16 *)IM_BUFFER_START, local_left, local_right);  // MME matmul → Intermediate Buffer
mov_m(local_out, (__fp16 *)IM_BUFFER_START);                   // MTE move IM → L1
```

### Runtime — HPE

**HPE** (Heterogeneous Programming Engine) = `hpert` (host) + `npurt` (device) + the driver. The host API is
a CUDA runtime clone by name: `stcMalloc`, `stcFree`, `stcMemcpy`, `stcConfigureCall`, `stcLaunchKernel`,
`stcRegisterFatBinary`, `stcDeviceSynchronize`. `stcpti` is the CUPTI analogue; **STCML** (`libstcml.so`) is
the NVML analogue. The resource model is card = device → **4 NPC Clusters** → NPCs, and **the cluster is the
allocation and isolation unit** (`/dev/stc0c0…c3`).

The tool set mirrors NVIDIA's one-for-one: `stc-smi` (nvidia-smi), `stc-topo` (nvidia-smi topo),
`stc-gdb` (cuda-gdb), `stc-prof` (nsys, emitting Chrome-tracing JSON for Perfetto), `stc-vprof` (Nsight),
`p2p_perf` (p2pBandwidthLatencyTest). `stc-prof` is unusually informative about this architecture: it
separates **VME-CU** (custom instructions) from **VME-VEC** (native RVV) cycles and reports L1/IM
bank-collision counters and per-engine overlap percentages.

### Driver / Firmware

`stc-dkms` builds `stc.ko` via DKMS; `stc-kernel-common` installs udev rules creating `/dev/stc0`,
`/dev/stc0c0…c3` and `/dev/stc0ctrl`. Two signed firmware images (NPU-ctrl for SoC bring-up, MCU for card
power) are upgradeable in-band with `.sdux` files. **No SR-IOV** (1 PF, no VFs), but PCIe **passthrough to
KVM guests is supported**.

### LLM Serving — STC_LLM

**STC_LLM is proprietary and is not a vLLM fork.** It serves an OpenAI-compatible API
(`/v1/chat/completions`, `/v1/completions`, SSE streaming) with dynamic batching (256 tasks default),
KV-cache segmentation, prefix caching, DMA weight preload, host-side embedding/logits offload, reasoning
parsers (Qwen3, DeepSeek-R1) and tool-call parsing. Three backends select with `--compiler`: the default
`stc_llm_dnn` — a **hand-written-kernel C++ code generator**, which the vendor says is currently faster —
plus the `mltc` compiler path and `IE`. Serve-time quantisation is `fp16 / w8a8 / w8a16`. Prometheus metrics
ship natively under the `stc_llm:` namespace.

Quantisation tooling: **SNQ** (ONNX PTQ) and **SNC** (`stcnc`, built on **Intel Neural Compressor**,
implementing SmoothQuant with an α sweep calibrated on C-Eval).

### Communication

**There is no collective-communication library** — no NCCL / CNCL / HCCL analogue anywhere in the SDK.
Multi-card parallelism is expressed *inside the compiler* as `stc.device_*` collective ops plus
`--pipeline-partition-factor`, and STC_LLM implements tensor parallelism by pre-splitting weights at
conversion time (`convert_weight -n <NPU count>`; weights for 4 NPUs do not work on 2). The transport is
PCIe peer-to-peer, measured by the vendor's own `p2p_perf` at ~9 GB/s to a remote LLB and **under 1 GB/s** to
remote LPDDR.

### Open source

**The entire product stack is closed.** `github.com/streamcomputing` is an empty account and no license
terms are published. The genuinely open artifacts are adjacent: the `riscv-stc` org (open RISC-V matrix
extension spec CC-BY-4.0, LLVM and Spike forks, `riscv-dnn`, BOOM/chipyard forks — **all quiet since March
2025**), `STCPaddleModelZoo`, and the Apache-2.0 STC backend contributed into ByteDance's ByteMLPerf.

---

## Hardware Architecture

### NeuralScale core (NPC)

Each of the 32 cores is an **AndesCore N25F** 32-bit RV32G scalar core, 5-stage in-order with dynamic branch
prediction and 64 KB L1 I$/D$, driving three decoupled engines through three instruction buffers. The issue
unit issues **in order, up to 3 instructions per cycle (one per engine)**, and blocks only on address overlap
with in-flight instructions:

- **VME** — 64 FP16 MACs plus a **POLY** unit with `exp` / `div` / `sqrt` for activations and classifiers;
  runs both base RVV and the custom instructions.
- **MME** — a **64 × 32 = 2048** MAC matrix; each MAC does FP16 or **2× INT8** per cycle.
- **MTE** — the explicit memory-transfer engine, moving data L1↔LLB, L1↔DRAM, and LLB→broadcast to all
  corresponding L1 Buffers.

RVV is configured **VLEN = 1024 b, ELEN = 16 b**. **Datatypes are FP16 and INT8 only** — no BF16, no FP8/FP4,
no INT4, no native FP32 compute path.

### P920 SoC

32 NeuralScale cores in **4 NPC Clusters**, an **ARM Cortex-A53** management CPU, a **4×4 mesh NoC** with
separated 32-bit control and 512-bit data planes (64 GB/s per direction per link at 1 GHz), and an **HSYNC**
subsystem that can partition the 32 cores into up to 16 hardware sync groups (the 16-group API is not exposed
in the public SDK, which presents the 4-cluster model). TSMC 12 nm FinFET, 400 mm², 130 W chip TDP at
1.0 GHz. DVFS spans 624–1400 MHz across independently settable **"east" and "west"** frequency/voltage
domains, and four power banks can be gated independently.

### Memory — software-managed and non-coherent

This is the defining property. The NoC is explicitly labelled a **"non-coherent interconnect"**, custom
VME/MME instructions take **L1 / Intermediate-Buffer byte addresses in general-purpose registers**, and all
inter-level movement is an explicit MTE or sysDMA instruction. The only hardware caches on the chip are the
64 KB scalar I$/D$, which serve the control path, not tensor data.

| Level | Capacity | Managed by |
|---|---|---|
| L1 Buffer (per NPC) | 1.25 MiB (1 MiB Data IO + 0.25 MiB Weight) | software |
| Intermediate Buffer (per NPC) | 256 KB | software |
| Scalar L1 I$ / D$ (per NPC) | 64 KB + 64 KB | hardware (control path only) |
| LLB (per cluster) | 8 MiB — 32 MiB total, 8 × 4 MB banks | software |
| LPDDR4X | 16 GB (4 GiB per cluster) | software |

**~80 MiB of on-chip SRAM** in total (derived, not vendor-stated). LLB bandwidth is **contested**: CARRV'21
says 17 TB/s aggregate, while the vendor's ByteDance-hosted spec table says 256 GB/s per cluster — a ~17×
gap that no source reconciles.

**Off-chip: 16 GB LPDDR4X, 256-bit at 3733 MT/s.** The card manuals state **108 GB/s**; the vendor's own
third-party-hosted table says 119.4 GB/s and the stated interface computes to 119.5 GB/s. The reason for the
derating is not disclosed. Each of the four DDR subsystems has two DMA controllers.

### Interconnect — the honest limitation

There is **no proprietary scale-up fabric**. The P920 has two PCIe subsystems and CARRV'21 describes PCIE1 as
"usually configured as a root complex … connecting to other SoC chips", but **no shipping card manual
documents any chip-to-chip or card-to-card link**, and `stc-topo` reports only PCIe path classes
(LOC/PIX/PXB/PHB/SYS) and NUMA affinity. The vendor's own `p2p_perf` measures 9.0 GB/s card-to-card when the
destination is on-chip LLB and **0.89 GB/s** when it is remote LPDDR — far below the PCIe 4.0 ×16 ceiling.
That is the concrete cost of the 16-card tensor-parallel configurations in the vendor's LLM table. Scale-out
is ordinary datacenter Ethernet; there is no RDMA fabric and no collective-communication hardware.

### Product line — one die, five bins

| SKU | Max clk | FP16 | INT8 | Card TDP | Memory |
|---|---|---|---|---|---|
| STCP920 | 1.0 GHz base | 128 TFLOPS | 256 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s |
| STCP950L | 1.2 GHz | 153 TFLOPS | 309 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s |
| STCP950P | 1.2 GHz | 145 TFLOPS | 290 TOPS | 150 W | 16 GB LPDDR4X @ 108 GB/s |
| STCP980L | 1.4 GHz | 179 TFLOPS | 358 TOPS | 160 W | 16 GB LPDDR4X @ 108 GB/s |
| STCP980P | 1.4 GHz | 167 TFLOPS | 335 TOPS | 150 W | 16 GB LPDDR4X @ 108 GB/s |

All five are PCIe 4.0 ×16, single-width, ¾-length, full-height, 721.2 g, passively cooled, with identical
4-cluster topology, identical L1/LLB/DDR capacities, shared firmware images and a single `npu-v1` compiler
target. The "L" SKUs' quoted throughput equals 128 TFLOPS × max clock exactly at 160 W; the "P" SKUs quote
less at 150 W. **The vendor never states die identity**, so "same die, different bins" is strongly evidenced
rather than disclosed.

---

## Measured Performance

**CARRV'21 (vendor-run, batch 64):** ResNet-50 v1.5 INT8 at **14,442 img/s** / 4.43 ms / 110 IPS/W at 130 W;
BERT FP16 (batch 32) at **4192 sentences/s** / 7.63 ms / 32 sps/W — claimed 2.98× T4 and 1.85× V100 on
ResNet-50 throughput.

**ByteMLPerf on STCP920 (reproducible, hosted in ByteDance's repo):** resnet50-tf-fp32 **8725.94 QPS** at
77.24% Top-1; bert-tf-fp32 **822.38 QPS** at F1 86.45; widedeep-tf-fp32 2,395,899.9 QPS.

**LLM (vendor table, single-stream):** Qwen2-7B-Instruct 2 cards @ 10.39 tok/s; Qwen2-72B-Instruct 16 cards
@ 4.98 tok/s; DeepSeek-R1-Distill-Llama-70B 16 cards @ 4.42 tok/s; Jiuzhou-7B (the company's own LLM) 2 cards
@ 10.3 tok/s.

**No MLPerf Inference submission** exists under any Stream Computing / 希姆 / NeuralScale name.

---

## Deployment and Roadmap

**Established.** First-generation silicon is in mass production with shipments (third-party-hosted vendor
statement). The public model-support table lists **ByteDance-internal model names**
(`hotsoon_live_v6_turbo`, `atmosphere_vulgar`, `model_goods_search`, `content_classify`, …), independently
corroborating a large internet-platform deployment; **ByteDance and JD.com are named strategic investors**.

**Vendor claims, not independently documented:** "1000+ STCP cards at a leading internet company, 200+
models"; "large numbers of thousand-card clusters nationwide"; a **"world's first ultra-high-altitude RISC-V
thousand-card compute cluster" with Tibet Mobile** (dated 2026-03-31 on the vendor news list); a RISC-V
cluster in Lianyungang; cloud compatibility certifications.

**Not products.** A **second-generation chip** is described by the vendor as tape-out-blocked by the US BIS
export ban in 2022 and "redesigned to meet the restriction limits" in 2024 — but **every specification is
not disclosed**, and the only public trace is the `--arch=npu-v2` compiler flag. The **"RISC-V AI super
node"** is marked 研发中 on the vendor site with no specs, topology or date. Neither belongs in the chip
registry as silicon.

---

## Competitive Position

NeuralScale occupies a narrow but genuine niche: a **programmable, RISC-V-ISA-extension datacenter inference
card for the Chinese domestic market**, competing against Sophgo BM1684X and NVIDIA T4/L4-class parts in
cost-sensitive CV/NLP inference rather than against HBM-class LLM accelerators.

- **Differentiator — programmability.** Unlike a fixed-function NPU, the part exposes a real ISA, a
  CUDA-shaped kernel language, a GDB-compatible device debugger and an Nsight-shaped profiler. For a survey
  of *software stacks*, this is one of the most complete non-NVIDIA developer toolchains in the Chinese
  vendor set.
- **Differentiator — RISC-V standards standing.** Premier RISC-V International membership and chairs of the
  AI/ML SIG and Software Applications & Tools committee (vendor claims, consistent with RVI's published
  tiers), plus a real open matrix-extension proposal.
- **Limitation — memory.** 16 GB LPDDR4X at ~108 GB/s and FP16/INT8-only datatypes rule out modern LLM
  serving at competitive tokens/s. The vendor's own numbers make this unambiguous.
- **Limitation — no fabric.** No scale-up interconnect and no collective library; 16-card tensor parallelism
  runs over PCIe P2P at ≤9 GB/s.
- **Limitation — one generation.** Shipping silicon dates to 2021 on TSMC 12 nm, and the second generation
  remains undisclosed four years after the export-control disruption the vendor cites.
- **Limitation — closed stack.** Less transparent than Sophgo's fully open TPU-MLIR; every claim about MLTC
  or HPE internals must be sourced from manuals rather than code.

---

## Sources

- [CARRV'21 — NeuralScale: A RISC-V Based Neural Processor Boosting AI Inference in Clouds](https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf)
- [STCP920 product manual, rev 1.12.1](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP920产品手册)
- [STCP950L](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950L产品手册) · [STCP950P](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950P产品手册) · [STCP980L](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980L产品手册) · [STCP980P](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980P产品手册) product manuals
- [STCRP product overview](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP产品简介) · [Release Notes V1.12.1](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP_Release_Notes)
- [HPE usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南) · [MLTC usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/MLTC使用指南) · [STC_LLM usage guide](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/STC_LLM使用指南)
- [MLTC operator support](https://docs.streamcomputing.com/AI加速卡/开发者资源/MLTC算子支持说明) · [Python API](https://docs.streamcomputing.com/AI加速卡/开发者资源/Python_API) · [C++ API (STCML)](https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API)
- [Model support table (LLM cards / tok-s)](https://docs.streamcomputing.com/AI加速卡/开发者资源/模型支持说明)
- [p2p_perf guide — measured card-to-card bandwidth](https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/p2p_perf使用指南)
- [ByteMLPerf / xpu-perf STC backend README](https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md)
- [riscv-stc/riscv-matrix-spec](https://github.com/riscv-stc/riscv-matrix-spec) · [riscv-stc org](https://github.com/orgs/riscv-stc/repositories)
- [Corporate site](https://www.streamcomputing.com/)
