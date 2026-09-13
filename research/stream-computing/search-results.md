# Stream Computing (希姆计算) NeuralScale — Search Results

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*

---

## Method note

The WebSearch quota for this research session was exhausted before work began. Discovery was therefore
done by **enumerating the vendor documentation sitemap** (`docs.streamcomputing.com/sitemap.xml`, 214 doc
pages across SDK versions 1.9.0 / 1.10.0 / 1.11.0 / 1.12.0 / next), direct `curl`/WebFetch of primary
sources, and the GitHub raw/HTML endpoints. Consequence: **Chinese trade-press and financial coverage
(funding rounds, second-generation tape-out status, customer churn) is under-sampled** and should be
re-checked when search budget is available. Everything recorded below was retrieved directly.

---

## Search Queries / Retrieval Actions

1. `curl https://docs.streamcomputing.com/sitemap.xml` → decoded 214 doc paths, all versions
2. Bulk fetch + HTML→text of 22 SDK doc pages under `/AI加速卡/` (5 hardware manuals, STCRP overview /
   install / release notes, HPE / MLTC / STC_LLM guides, C++ & Python API, operator & model support,
   firmware notes, NPU-Viewer, p2p_perf, k8s-device-plugin, npu-exporter)
3. WebFetch CARRV'21 PDF, then page-image read of all 7 pages (the PDF has no text layer) — full
   microarchitecture extraction
4. grep SDK docs for 核心频率 / 功耗 / 内存带宽 / PCIe Device ID / cluster / L1 / LLB / DDR across all five
   STCP SKUs
5. grep SDK docs for `ccl|collective|allreduce|NCCL|Tensor并行|peer|P2P|RDMA` — to establish the **absence**
   of a collective-communication library
6. grep SDK docs for `hpe.h|npurt.h|__global__|__device__|__local__|__shared__|stcMalloc|IM_BUFFER` — SHC
   kernel-language surface
7. Fetch `/AI加速卡/next/开发者资源/Python_API` and `/next/STCRP_Release_Notes` to check for content newer
   than V1.12.1 and for `npu-v2`
8. `curl https://www.streamcomputing.com/` → company profile, milestone timeline 2021–2025, news to
   2026-05, investors, offices, ICP filing
9. WebFetch `https://github.com/orgs/riscv-stc/repositories` (19 repos, languages, last-updated)
10. `curl` raw GitHub: `riscv-matrix-spec/{readme,intro,tilereg,extensions,ext-type,param,contributors}.adoc`,
    `riscv-matrix-project/README.md`, `riscv-dnn/README.md`
11. `curl` raw GitHub: `bytedance/xpu-perf` STC backend README (mass-production statement, spec table,
    reproducible results)
12. `curl` raw GitHub: `Stream-Computing/STCPaddleModelZoo/{README.md,Paddle-STCNNE.md}`
13. WebFetch RISC-V International blog post on NeuralScale
14. Attempted and **blocked**: vendor WeChat article index (anti-bot verification); DuckDuckGo HTML/lite for
    `希姆计算 第二代芯片 流片`, `希姆计算 融资 2025`, `希姆计算 超节点 RISC-V`

---

## Resources Found

### Peer-Reviewed / Academic (primary)

| Resource | URL | Category |
|----------|-----|----------|
| CARRV'21 — "NeuralScale: A RISC-V Based Neural Processor Boosting AI Inference in Clouds" (Zhan & Fan, Stream Computing Inc., Beijing) — **primary microarchitecture source** | https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf | Research (primary) |

### Vendor Hardware Manuals (primary)

| Resource | URL | Category |
|----------|-----|----------|
| STCP920 product manual, rev 1.12.1 | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP920产品手册 | Hardware Spec |
| STCP950L product manual, rev 1.12.1 | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950L产品手册 | Hardware Spec |
| STCP950P product manual, rev 1.12.1 | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950P产品手册 | Hardware Spec |
| STCP980L product manual, rev 1.12.1 | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980L产品手册 | Hardware Spec |
| STCP980P product manual, rev 1.12.1 | https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980P产品手册 | Hardware Spec |
| Firmware Release Notes — NPU-ctrl + MCU firmware history | https://docs.streamcomputing.com/AI加速卡/固件使用手册/Firmware_Release_Notes | Firmware |
| Firmware update guide | https://docs.streamcomputing.com/AI加速卡/固件使用手册/固件更新指南 | Firmware |

### Vendor SDK Documentation (primary)

| Resource | URL | Category |
|----------|-----|----------|
| STCRP product overview (HPE / MLTC / STC_LLM architecture) | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP产品简介 | SDK Docs |
| STCRP installation guide — packages, OS matrix, Docker, device nodes | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP安装指南 | SDK Docs |
| STCRP Release Notes V1.12.1 — component version matrix | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP_Release_Notes | SDK Docs |
| HPE usage guide — stc-topo / stc-smi / stc-gdb / stc-prof / stc-vprof + SHC samples | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南 | Runtime Docs |
| MLTC usage guide — TF/ONNX/PyTorch flows, custom ops, `.vmfb`, precision & perf analysis | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/MLTC使用指南 | Compiler Docs |
| STC_LLM usage guide — OpenAI server, backends, TP, quantisation, Prometheus | https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/STC_LLM使用指南 | Serving Docs |
| Glossary (NeuralScale, NPC, LLB, L1, IM, MME/VME/MTE, SHC, HPE, MLTC, STC_LLM, SNQ, SNC) | https://docs.streamcomputing.com/AI加速卡/希姆计算术语表 | SDK Docs |
| C++ API — STCML (NVML analogue) full reference | https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API | API Reference |
| Python API — MLTC frontends + all compile flags incl. `--arch=npu-v1/npu-v2` | https://docs.streamcomputing.com/AI加速卡/开发者资源/Python_API | API Reference |
| MLTC operator support — 164 `stc.*`, 156 ONNX, 106 TF, 141 Torch ATen ops | https://docs.streamcomputing.com/AI加速卡/开发者资源/MLTC算子支持说明 | API Reference |
| Model support table — CV/NLP/OCR/RecSys/speech/multimodal + LLM cards & tok/s | https://docs.streamcomputing.com/AI加速卡/开发者资源/模型支持说明 | Benchmark Data |
| p2p_perf guide — measured card-to-card DDR/LLB bandwidth | https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/p2p_perf使用指南 | Tool Docs |
| NPU-Viewer guide — PCIe / DDR / TOPS / stress qualification | https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/NPU-Viewer使用指南 | Tool Docs |
| stc-k8s-device-plugin guide | https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/k8s-device-plugin使用指南 | Deployment Docs |
| npu-exporter guide — Prometheus metric list, port 9836 | https://docs.streamcomputing.com/AI加速卡/算力服务解决方案/npu-exporter使用指南 | Deployment Docs |
| 希姆云平台 (cloud platform) docs | https://docs.streamcomputing.com/智算云平台/希姆云平台使用手册/希姆云平台介绍 | Product Docs |
| Government-agent AI appliance (AI 一体机) docs | https://docs.streamcomputing.com/AI一体机/政务智能体一体机使用手册/政务智能体一体机产品简介 | Product Docs |
| Docs sitemap — enumerates all 214 doc pages | https://docs.streamcomputing.com/sitemap.xml | Index |

### Vendor Corporate / Press

| Resource | URL | Category |
|----------|-----|----------|
| Corporate site — profile, offices, milestone timeline, investors, RISC-V standing, ICP 粤ICP备2024180922号 | https://www.streamcomputing.com/ | Overview (primary) |
| RISC-V International blog — "NeuralScale: industry-leading general-purpose programmable NPU architecture" (company-authored, 2021-07-06) | https://riscv.org/blog/2024/03/neuralscale-industry-leading-general-purpose-programmable-npu-architecture/ | Vendor blog |
| Vendor WeChat article index (2026-03-31 Tibet Mobile cluster etc.) — **bot-blocked, not retrievable** | https://mp.weixin.qq.com/s/Lk_c7TUm8_ogViTL310U8g | Vendor press (inaccessible) |

### Third-Party-Hosted Vendor Code

| Resource | URL | Category |
|----------|-----|----------|
| ByteMLPerf / xpu-perf STC backend README — mass-production statement, HPE 1.5.1 / TensorTurbo 1.11.0 / STC_DDK 1.1.0, spec table, reproducible QPS + accuracy (Apache-2.0, © 2023 Stream Computing Inc.) | https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md | Benchmark (third-party host) |
| raw URL for the above | https://raw.githubusercontent.com/bytedance/xpu-perf/main/projects/infer_perf/general_perf/backends/STC/README.md | Benchmark (third-party host) |

### Open-Source Repositories (adjacent — **not** the shipping toolchain)

| Resource | URL | Category |
|----------|-----|----------|
| "Stream Computing OSS" GitHub org (19 repos) | https://github.com/orgs/riscv-stc/repositories | Overview |
| riscv-matrix-spec — open RISC-V matrix extension proposal, CC-BY-4.0, 27★ | https://github.com/riscv-stc/riscv-matrix-spec | Spec |
| riscv-matrix-spec — tile/accumulation register model | https://raw.githubusercontent.com/riscv-stc/riscv-matrix-spec/main/tilereg.adoc | Spec |
| riscv-matrix-spec — implementation parameters ELEN/MLEN/RLEN/AMUL | https://raw.githubusercontent.com/riscv-stc/riscv-matrix-spec/main/param.adoc | Spec |
| riscv-matrix-spec — Zmi4 INT4 matrix sub-extension | https://raw.githubusercontent.com/riscv-stc/riscv-matrix-spec/main/ext-type.adoc | Spec |
| riscv-matrix-spec — contributor list | https://raw.githubusercontent.com/riscv-stc/riscv-matrix-spec/main/contributors.adoc | Spec |
| riscv-matrix-project — umbrella build (LLVM + Spike + chipyard/BOOM + PVP + riscv-dnn) | https://github.com/riscv-stc/riscv-matrix-project | Toolchain |
| riscv-stc LLVM fork (matrix extension support) | https://github.com/riscv-stc/llvm-project | Compiler fork |
| riscv-stc Spike fork (matrix extension ISS) | https://github.com/riscv-stc/riscv-isa-sim | Simulator fork |
| riscv-dnn — DNN library on RISC-V Vector + Matrix extensions | https://github.com/riscv-stc/riscv-dnn | Library |
| riscv-pvp-matrix — ISA verification platform | https://github.com/riscv-stc/riscv-pvp-matrix | Verification |
| STCPaddleModelZoo — PaddlePaddle Level-I compatibility model zoo (TensorTurbo era) | https://github.com/Stream-Computing/STCPaddleModelZoo | Model Zoo |
| Paddle-STCNNE.md — Paddle backend usage, TensorTurbo/STC_DDK prerequisites | https://raw.githubusercontent.com/Stream-Computing/STCPaddleModelZoo/master/Paddle-STCNNE.md | Model Zoo Docs |

---

## Key Findings

- **One chip generation, five SKUs.** All shipping products (STCP920 / 950L / 950P / 980L / 980P) are PCIe
  4.0 ×16 inference cards built on the **first-generation NeuralScale P920** SoC: TSMC 12 nm, ~400 mm²,
  32 NeuralScale cores, 128 TFLOPS FP16 / 256 TOPS INT8 @ 1.0 GHz, 16 GB LPDDR4X. The higher-numbered SKUs
  are clock/power bins; the compiler target for all of them is `npu-v1`.
- **The distinctive architecture is per-core RISC-V + three decoupled engines.** Each NeuralScale core (NPC)
  is an AndesCore N25F RV32G scalar core issuing in order, up to 3 instructions/cycle, one each to a **VME**
  (vector MAC + transcendental POLY unit), an **MME** (64×32 MAC matrix), and an **MTE** (explicit
  memory-transfer engine). This is MIMD-per-core, not SIMT, and not a fixed-function NPU.
- **The ISA is a vendor-custom RISC-V extension, not the ratified matrix extension.** RV32G + RVV v0.8 plus
  **53 custom instructions** on the custom-3 opcode (`1111011`, "OP-VE") and **22 unprivileged shape CSRs**.
  Separately, the company chairs work on an **open** RISC-V matrix extension proposal (`riscv-stc/riscv-matrix-spec`,
  CC-BY-4.0) that the shipping silicon does **not** implement. These two must not be conflated.
- **Memory is entirely software-managed and non-coherent.** L1 Buffer (1.25 MiB/core), Intermediate Buffer
  (256 KB/core), LLB (8 MiB/cluster, 32 MiB/chip) and 16 GB LPDDR4X are all addressed explicitly by the
  program; only the 64 KB scalar I$/D$ are hardware caches, and they serve the control path.
- **Bandwidth-starved for LLM decode.** ~108 GB/s LPDDR4X against 128 TFLOPS FP16. The vendor's own model
  table shows Qwen2-72B-Instruct on **16 cards at 4.98 tok/s**. This is a CNN/NLP-era inference card
  retrofitted for LLMs, not an HBM-class part.
- **No scale-up fabric.** No NVLink/CCIX/Ethernet-mesh equivalent on any shipping card; the CARRV paper's
  PCIE1-as-root-complex chip-to-chip path is not documented in any card manual. Measured peer-to-peer via
  the vendor's own `p2p_perf` peaks at ~9 GB/s (LLB destination) and <1 GB/s (DDR destination).
- **Software stack = CUDA clone under an MLIR/IREE graph compiler.** SHC (`.hc`, C++17 + `<<< >>>`), `stcc`,
  `stcMalloc`/`stcLaunchKernel`, `stc-smi`/`stc-gdb`/`stc-prof`/`stc-topo`/`stcpti`/`STCML` — each a direct
  NVIDIA analogue — beneath **MLTC**, an MLIR graph compiler emitting IREE `.vmfb` with a torch-mlir-derived
  PyTorch frontend, and **STC_LLM**, a proprietary (non-vLLM) OpenAI-compatible LLM server.
- **Compiler-lineage migration is a real finding**: the 2021–2023 stack was **TensorTurbo** (TVM + LLVM +
  stcDNN); the current stack is **MLTC** (MLIR + IREE + torch-mlir). Pre-2024 third-party descriptions
  document the *old* stack and must not be mixed in.
- **The product stack is entirely closed source.** `github.com/streamcomputing` is an empty account; SDK
  packages come from vendor support. The open artifacts (`riscv-stc` org, STCPaddleModelZoo, the ByteMLPerf
  STC backend) are adjacent research/standards/benchmark work, and the whole `riscv-stc` org has been quiet
  since **March 2025**.
- **Maturity: shipping and deployed, with one third-party-hosted corroboration.** ByteDance's ByteMLPerf
  repo hosts the vendor's Apache-2.0 backend stating STCP920 "is in mass production, and has completed a
  batch of shipments to users", with reproducible QPS + accuracy. Larger deployment figures
  ("1000+ cards", "thousand-card clusters", the 2026-03-31 Tibet Mobile cluster) are **vendor claims only**.
- **`npu-v2` is a compiler flag, not a chip.** MLTC 1.6.1 accepts `--arch=npu-v2`. No manual, no core count,
  no node, no tape-out confirmation. Must not be entered into the registry as silicon. Likewise the
  "RISC-V AI super node" is marked 研发中 on the vendor site.
- **No MLPerf Inference submission** was found under any Stream Computing / 希姆 / NeuralScale name.
