# FuriosaAI NPU Software and Hardware Stack Summary

*as_of: 2026-08-08*

---

## Overview

FuriosaAI (founded 2017, Seoul, South Korea) is a deep learning inference NPU company that has shipped two silicon generations: **Warboy** (Samsung 14nm, 2021) and **RNGD** ("Renegade", TSMC 5nm, mass production January 2026). A **third generation**, co-developed with Broadcom, was *announced* on 2026-05-27 (2nm compute die, HBM4/4E, Ethernet scale-up fabric) — announcement only, with sampling targeted for H1 2028. The company rejected a reported $800 million acquisition offer from Meta in March 2025, is pursuing a 2027 IPO, and is raising a Series D (see the 2026-08-08 update section for the current, unclosed status).

FuriosaAI's architectural thesis centers on the **Tensor Contraction Processor (TCP)** — introduced in RNGD — which treats arbitrary-rank tensor contraction (a higher-dimensional generalization of matrix multiplication) as the fundamental compute primitive. This enables a single hardware design to handle GEMM, attention, and convolution without decomposition into separate kernel paths.

---

## Hardware

### Warboy (Gen 1 — Samsung 14nm, 2021)

| Specification | Value |
|---------------|-------|
| Process | Samsung 14nm LPP |
| Die area | 180 mm² |
| Transistors | 5 billion |
| Clock | 2.0 GHz |
| Peak INT8 | 64 TOPS |
| On-chip SRAM | 32 MB |
| Off-chip memory | 16 GB LPDDR4X |
| Memory bandwidth | 66 GB/s |
| Host interface | PCIe Gen4 x8 |
| TDP | ~50W |
| Target workload | CNN inference (ResNet, YOLO, EfficientDet) |

Warboy was designed using the SemiFive ASIC platform and is a **vision-first NPU** optimized for image classification, object detection, and segmentation. It uses a PE-array architecture with a compiler-managed SRAM hierarchy. No caches — all data movement is compiler-scheduled.

### RNGD (Gen 2 — TSMC 5nm, January 2026)

| Specification | Value |
|---------------|-------|
| Process | TSMC 5nm N5 |
| Transistors | 40 billion |
| Clock | 1.0 GHz |
| Peak FP8 | **512 TFLOPS** (vendor product page: 64 TFLOPS FP8 × 8 PEs) |
| Peak BF16 | 256 TFLOPS *(developer docs; not independently confirmed)* |
| Peak INT8 / INT4 | 512 TOPS / 1,024 TOPS *(developer docs; not independently confirmed — see 2026-08-08 correction note)* |
| Compute units | 8 PEs × 64 slices = 512 slices |
| On-chip SRAM | **256 MB @ 384 TB/s** |
| Off-chip memory | **2× HBM3 stacks = 48 GB** (CoWoS-S, 6.0 Gbps) |
| Memory bandwidth | 1.5 TB/s |
| Host interface | PCIe Gen5 x16 (with card-to-card P2P) |
| TDP | **180 W** (vendor product page and all 2026 press); developer docs list **150 W** — see note below |
| Target workload | LLM / multimodal / vision inference |
| Architecture | Tensor Contraction Processor (TCP) |

> **Corrected 2026-08-08.** The prior version of this table recorded "2× HBM3 = 24 GB", "Peak INT8 512 TOPS", and "On-chip SRAM: not disclosed". FuriosaAI's own product page (furiosa.ai/rngd) states **48 GB HBM3** and **256 MB SRAM @ 384 TB/s**, and presents the 512 figure as **TFLOPS FP8** (64 TFLOPS FP8 per PE × 8 PEs), not INT8 TOPS. The 24 GB figure was wrong by 2×, and the partitioning table derived from it has been recomputed. The INT8/INT4 TOPS numbers are retained but downgraded to "reported, not independently confirmed" because only the FP8 figure could be verified against a vendor primary source. The **150 W vs 180 W** split is real: 150 W appears in the developer documentation as the card spec, 180 W in the product page and every 2026 press release — treat 150 W as the documented device figure and 180 W as the vendor's headline/system-level figure.

RNGD is an **LLM-first inference accelerator** with 23× more memory bandwidth than Warboy (1.5 TB/s HBM3 vs. 66 GB/s LPDDR4X). The TCP architecture maps all deep learning primitives — attention, GEMM, convolution — to the same hardware via tensor contraction. A single RNGD chip can be partitioned (SR-IOV) into 2, 4, or 8 hardware-isolated virtual NPUs for multi-tenant Kubernetes deployments.

**NXT RNGD Server**: originally announced as a 4-card server with 3 kW total power (vs. >10 kW for DGX H100); as of the 2026-07-07 Equinix Lisbon announcement FuriosaAI describes the NXT RNGD Server as holding **up to 8 RNGD accelerators in a 3 kW-class system**. LG AI Research validated 2.25× better LLM inference performance per watt running EXAONE 3.5 32B (60 tokens/sec at 4K context, 50 tokens/sec at 32K context). GPT-OSS 120B: 5.8 ms TPOT on 2 RNGD cards.

### Gen 3 (Broadcom co-development) — ANNOUNCED ONLY, no silicon

Announced 2026-05-27. No product name or codename. Disclosed: 2nm compute die (TSMC, per Korean wire coverage), HBM4/HBM4E, multi-die chiplet system-in-package using Broadcom's XPU Technology and IP Platform and advanced packaging, and a Broadcom **Ethernet scale-up + fabric switch** interconnect with an all-to-all-capable topology for MoE expert routing. Sampling targeted **H1 2028**. Everything else — peak throughput at any dtype, HBM capacity, memory bandwidth, TDP, interconnect bandwidth, scale-up domain size, die count, mass-production date — is **not disclosed**. Details in the update section below.

---

## Software Stack

### Architecture: Compiler-First, Runtime-Thin

FuriosaAI follows a deliberate **compiler-first** philosophy (parallel to Groq's approach): all scheduling, memory planning, quantization, and operator fusion is done ahead-of-time (AOT) by the compiler. The runtime is a thin executor that replays the pre-compiled plan. No JIT compilation, no dynamic kernel dispatch.

> **Amended 2026-08-08.** Compiler-first and AOT still hold, but SDK **2026.3** (2026-06-30) added **TCL (Tensor Contraction Language)**, a user-facing declarative Python eDSL for authoring kernels, plus a `furiosa-kernels` package of reusable blocks. FuriosaAI is no longer a stack with *no* kernel-authoring surface. See "No Op Library / Kernel Library" below and the 2026-08-08 update section.

### Layer Map

```
Framework                    PyTorch → ONNX export / HuggingFace (furiosa-llm)
                              ↓
Kernel Language (2026.3+)    TCL — Tensor Contraction Language (proprietary)
                             - Declarative Python eDSL; @tcl.kernel functions
                             - furiosa-kernels: RMSNorm / Linear / MLP blocks
                             - Padding, sharding, multi-chip collectives at
                               language level; compiler owns tiling + schedule
                              ↓
Virtual ISA                  FVISA — "Furiosa Virtual ISA"
                             - Named publicly at RENEGADE Summit 2026
                             - No technical detail disclosed
                              ↓
Compiler (AOT)               furiosa-compiler (proprietary)
                             - ONNX graph → hardware binary
                             - Op fusion, tiling, quantization, memory planning
                             - TCP tensor contraction mapping (RNGD)
                              ↓
Quantization                 furiosa-quantizer
                             - INT8/INT4/FP8 PTQ; GPTQ-style LLM quant
                              ↓
Model Binary                 .enf (Warboy) / RNGD binary
                              ↓
LLM Interface (RNGD)         furiosa-llm
                             - LLM(), SamplingParams()
                             - LLaMA, Mistral, EXAONE, Gemma
                              ↓
Runtime                      furiosa-runtime
                             - Sync/async inference session API
                             - RNGD: partition selection, KV-cache, continuous batching
                              ↓
Serving                      furiosa-serving (OpenAI API) / TGI backend plugin
                              ↓
Driver                       furiosa-npu.ko (Linux kernel module, apt install)
                             - PCIe BAR MMIO, DMA, partition management
                              ↓
Hardware                     Warboy (Samsung 14nm) / RNGD (TSMC 5nm)
```

### Key Components

| Component | Open Source | Notes |
|-----------|-------------|-------|
| furiosa-sdk (GitHub) | Yes (Apache 2.0) | ⚠️ *Legacy Warboy-era line only (latest tag 0.9.2, June 2024). The 2026.x SDK does **not** ship from this repo — see 2026-08-08 update.* |
| furiosa-compiler | No (binary dist.) | AOT compiler; ONNX → .enf / RNGD binary |
| furiosa-quantizer | Yes | INT8/INT4/FP8 PTQ |
| furiosa-runtime | Yes | Inference session API |
| furiosa-llm | Yes (Apache 2.0; 2026.3.0 on PyPI) | HuggingFace-style LLM interface (RNGD) |
| **furiosa-kernels** | **No — `LicenseRef-Proprietary`** | **New in SDK 2026.3: TCL kernel blocks for RNGD (PyPI, 2026-06-30, Python 3.10+)** |
| furiosa-models | Yes | Model zoo (ResNet, YOLO, SSD) |
| furiosa-serving | Yes | OpenAI + Triton compatible server |
| furiosa-npu.ko | No (binary dist.) | Linux kernel driver |

### Op Library / Kernel Library

FuriosaAI has **no user-facing op library** (no cuDNN equivalent). Operator fusion and hardware mapping remain internal to the compiler.

**Superseded 2026-08-08 — kernel library.** Through SDK 2026.2 this document stated that FuriosaAI had "no kernel library (no CUTLASS equivalent)" and that "users never write custom kernels." That is no longer true. SDK **2026.3** (2026-06-30) shipped **TCL (Tensor Contraction Language)**, a declarative Python eDSL in which a kernel author writes a `@tcl.kernel`-decorated function stating *what* to compute and leaves tiling, scheduling, fusion, and hardware mapping to the compiler. TCL kernels ship as reusable blocks in the `furiosa-kernels` package. This moves FuriosaAI into the Triton/Pallas tier of the taxonomy — with the caveat that TCL is **declarative** (no user-specified schedule), a meaningfully different design point from Triton's explicitly-tiled imperative model, and that the layer is **proprietary-licensed with no public reference documentation** as of 2026-08-08.

---

## Programming Model Philosophy

1. **AOT-only**: No eager execution. Every model must be compiled before deployment. This enables zero-overhead runtime at the cost of fixed input shapes (recompilation needed for shape changes).

2. **ONNX as lingua franca**: Any ML framework that exports ONNX is supported (PyTorch, TF, JAX via ONNX-TF, CoreML). No framework-specific backend required.

3. **Tensor Contraction as universal primitive** (RNGD): Rather than maintaining separate GEMM/attention/conv kernel paths, RNGD's compiler maps all ops to TCP slices via tensor index renaming. This simplifies the compiler and makes the hardware future-proof for novel operator types.

4. **Multi-tenant by hardware design**: RNGD's 2/4/8 partition support is baked into the hardware, not emulated. This enables Kubernetes-native NPU scheduling without compromising performance isolation.

---

## Competitive Positioning

| Dimension | FuriosaAI RNGD | NVIDIA H100 | Meta MTIA v2 | Rebellions ATOM |
|-----------|----------------|-------------|--------------|-----------------|
| Process | TSMC 5nm | TSMC 4nm | TSMC 5nm | Samsung 5nm |
| Peak INT8 | 512 TOPS *(reported)* | 3,958 TOPS | ~102 TOPS | 128 TOPS |
| Memory capacity | 48 GB HBM3 | 80 GB HBM3 | 128 GB LPDDR5 | 16 GB GDDR6 |
| Memory BW | 1.5 TB/s | 3.35 TB/s | 204 GB/s | 256 GB/s |
| TDP | 180W | 700W | ~75W | ~40W |
| TOPS/W (INT8) | 2.84 | 5.65 | ~1.36 | ~3.2 |
| Architecture | TCP (custom) | SIMT + Tensor Core | Grid PE array | CGRA |
| LLM support | First-class | Full | DLRM + LLM | Limited |
| Origin | Korea | USA | USA | Korea |

*RNGD memory capacity corrected from 24 GB to 48 GB on 2026-08-08. The TOPS/W figure uses the reported 512 INT8 TOPS at 180 W; on the verified FP8 figure the equivalent is 2.84 TFLOPS FP8/W.*

RNGD's key competitive advantage is **power efficiency for LLM inference**: 3 kW server vs. 10 kW DGX H100 for comparable LLM output. The NXT RNGD server is positioned as a direct GPU-replacement for inference-only data center workloads.

---

## Business Context

- **Founded**: 2017, Seoul
- **Warboy deployed**: Kakao (largest Korean internet company), Korean AI cloud customers
- **RNGD production**: Mass production January 2026; ~1,000 units/month (target: 2,000–3,000/month by end 2026)
- **Meta $800M rejected**: March 2025 (strategic disagreement, not price)
- **Series C**: $125M (July 2025; KDB, IBK, Kakao Investment)
- **Series D**: $500M target (Morgan Stanley + Mirae Asset advisers) — **still open as of late July 2026**; see update section for the reported (low-confidence) revised size
- **Valuation**: ~$2.3 billion (KRW 3 trillion) at the Series D target announcement
- **IPO target**: 2027

---

## 2026 Update (2026-08-08)

*Covers 2026-04-10 through 2026-08-06. Primary sources: FuriosaAI blog and newsroom, furiosa.ai/rngd product page, developer.furiosa.ai 2026.3.0, PyPI package metadata, Korean wire coverage (Yonhap / Digital Today / The AI, 2026-05-28). Prior-generation content above is preserved; corrections are marked in place.*

### 1. Third-generation accelerator with Broadcom — announced 2026-05-27

FuriosaAI announced a strategic partnership with Broadcom to develop its **third-generation AI inference accelerator**. In FuriosaAI's own words the platform is "incorporating HBM4/4E, 2nm process technology, and high-speed inter-chip networking," built "by pairing Furiosa's TCP architecture with Broadcom's market-leading XPU Technology and IP Platform, Ethernet scale-up and fabric switches." Charlie Kawwas (President, Broadcom Semiconductor Solutions Group) is quoted in the release.

| Attribute | Gen 3 (announced) |
|---|---|
| Product name / codename | **Not disclosed** |
| Status | Announced partnership / roadmap item — no tape-out, no silicon, no shipping |
| Sampling target | **H1 2028** (Korean wire coverage, 2026-05-28; not in the English release) |
| Compute die process | **2nm** (vendor); **TSMC 2nm** per Yonhap / Digital Today / The AI |
| Memory | **HBM4 / HBM4E** — capacity and bandwidth **not disclosed** |
| Packaging | Multi-die chiplet system-in-package using Broadcom advanced packaging |
| Compute architecture | Furiosa TCP paired with Broadcom XPU Technology and IP Platform |
| Scale-up interconnect | **Broadcom Ethernet scale-up + fabric switches**, all-to-all-capable topology for MoE expert routing |
| Peak throughput (any dtype) | **Not disclosed** |
| TDP, interconnect BW, scale-up domain size, die count/sizes, mass-production date | **Not disclosed** |

**Architectural significance.** This is the first **scale-up fabric** in FuriosaAI's roadmap. Warboy and RNGD are both PCIe-only parts with no proprietary chip-to-chip interconnect; multi-card RNGD parallelism runs over host PCIe. Gen 3 moves FuriosaAI into the same structural category as Google TPU v8t/Broadcom and OpenAI Titan/Broadcom: an Ethernet-based scale-up domain designed around data movement rather than peak FLOPS. FuriosaAI positions the part explicitly as optimizing data movement and memory access rather than raw FLOPS.

> **Do not record as spec.** Figures circulating on news aggregators — "432 GB across 12 memory sites", "two 2nm compute chiplets plus two IO dies", "3.5D XDSiP" — are **not vendor-disclosed**. They trace to Wccftech speculation relayed by AI-generated aggregators and are excluded here.
>
> **Note on sourcing.** No Broadcom-issued press release confirming the partnership was located; the announcement is FuriosaAI-side with a Broadcom executive quote.

### 2. SDK 2026.3 and the TCL kernel language — released 2026-06-30

The single largest software-stack change since RNGD launch. FuriosaAI's stack was previously characterized (in this document and in `layer-table.md`) as having no kernel-authoring surface at all. SDK **2026.3.0** introduces one.

**TCL — Tensor Contraction Language.** A declarative Python eDSL in which a kernel author writes a high-level, `@tcl.kernel`-decorated function that says *what* to compute and leaves *how* to run it — tiling, scheduling, fusion, hardware mapping — to the compiler. Tensor contraction is a first-class language primitive, matching the TCP hardware primitive directly. Padding, sharding, and multi-chip collectives are expressed at language level. Kernels ship as reusable blocks (RMSNorm, Linear, MLP) in a new **`furiosa-kernels`** package. FuriosaAI's framing: "Enablement now scales with the number of reusable blocks, not the number of models."

**Openness caveats (important for the survey's taxonomy).**
- `furiosa-kernels` 2026.3.0 on PyPI (uploaded 2026-06-30, summary "TCL kernels for NPU", Python 3.10+) is licensed **`LicenseRef-Proprietary`** — *not* Apache 2.0. `furiosa-llm` 2026.3.0 is Apache; the SDK is mixed-license.
- The public Developer Center at 2026.3.0 contains **no TCL or furiosa-kernels documentation section**.
- Whether TCL is generally available to all customers or gated to partners **could not be confirmed**.

**Models unlocked by TCL** (per the release blog): Qwen3-VL (e.g. Qwen3-VL-32B), gpt-oss-120b, Solar-Open-100B, Qwen3-30B-A3B (MoE), K-EXAONE-236B-A23B.

**Other 2026.3 features** (vendor release blog; not independently verified): **FXB (Furiosa Executable Bundle)** portable compiled artifacts giving zero-recompilation reuse across compatible model variants; multimodal serving with chunked prefill; overlap scheduling to reduce NPU idle time; scoring-based data-parallel request routing combining prefix locality and token footprint. Upgrading from 2026.2 involves breaking changes.

**FVISA.** "Furiosa Virtual ISA" was named publicly as a stack layer alongside the compiler and TCL at RENEGADE Summit 2026 (blog dated 2026-05-13). **No technical detail is disclosed** — no instruction set, no documentation, no user-facing tooling. It is recorded here as a named layer only.

**Distribution channel correction.** `github.com/furiosa-ai/furiosa-sdk` is stale — latest release tag 0.9.2 (June 2024), still the Warboy-era line, with no mention of TCL, `furiosa-kernels`, or 2026.x. The live SDK ships as PyPI wheels under a calendar-version scheme (2026.x) documented at developer.furiosa.ai. The shorthand "furiosa-sdk, Apache 2.0, GitHub" used earlier in this document is now misleading and should not be read as describing the current SDK.

### 3. RNGD production and deployment (2026-05 → 2026-08)

- **Mass production formally declared** at RENEGADE Summit 2026 (2026-05-13), TSMC 5nm. Supply chain named publicly: TSMC (foundry), SK hynix (HBM3).
- **Sweden — largest disclosed deployment (2026-08-04).** FuriosaAI with I/ONX HPC and Velox, Stockholm; **15 MW total** site. Phase 1 = **1,800 RNGD accelerators**, 2 MW online early 2027; a further 8 MW later in 2027; **7,000+ additional accelerators** planned in subsequent phases.
- **Portugal / Equinix (2026-07-07).** RNGD servers installed at Equinix Lisbon **LS2** for European enterprise evaluation, announced at RAISE Summit Paris. NXT RNGD Server described here as **up to 8 RNGD accelerators in a 3 kW-class system**. European flagship office established in Portugal, 2026-04-10.
- **Korea NPUaaS (2026-07-20).** Samsung SDS launched what it describes as Korea's first domestic **NPU-as-a-Service** on Samsung Cloud Platform, RNGD-powered, in the Dongtan data center, subscribable in **1/2/4/8-card configurations** (matching RNGD's hardware partition modes). Partnership work began September 2025. Vendor-quoted specs in the release: 512 TFLOPS FP8, 1.5 TB/s, 180 W. Serves Qwen3 and gpt-oss 120B.
- **Deployment commitments announced at RENEGADE Summit 2026:** LG AI Research, Samsung SDS, LG U+, Upstage, MegazoneCloud.
- **Daum search overviews (2026-07-31).** RNGD reported powering AI-generated search overviews for Daum (Kakao's portal) at millions-of-users scale. Headline-level only; recorded at low confidence.

### 4. New benchmark — vendor-published, treat as vendor claim (2026-08-06)

White paper benchmarking RNGD against **4× NVIDIA RTX PRO 6000 Blackwell Server Edition** (bare metal) on **Qwen3-32B FP8** via Lablup **Backend.AI**. Vendor claims:

| Metric | Claim |
|---|---|
| Throughput per watt | **1.3×–1.5×** higher than the GPU setup at all concurrency levels |
| Peak throughput | **95%** of the GPU setup's peak at 256-request concurrency |
| Power draw | **30–44%** lower |
| TTFT | **< 1 s** through concurrency 32, vs **> 2.9 s** for the GPU control |

This is a FuriosaAI-published white paper, not an independent evaluation and not a standardized benchmark.

### 5. Research output — ICML 2026 (Seoul)

Four papers from FuriosaAI's AI Research Group, all algorithmic/software with **no hardware disclosures**: **ReJump** (tree-jump representation for LLM reasoning; up to 9.1% improvement on reasoning tasks), **LoSA** (locality-aware sparse attention for block-wise diffusion LMs), **AsyncOPD** (stale on-policy distillation; 1.6×–3.8× higher training throughput), **EfficientRollout** (4-bit quantized drafting + adaptive speculative decoding; up to 19.6% rollout speedup, 12.7% end-to-end training improvement).

### 6. Business

- **Series D is NOT closed** as of late July 2026. A secondary Korean trade report (The Bell, relayed 2026-07-21) says the round is "nearing the close" at **KRW 800B (~$600M)** — above the $500M target recorded above — at a **KRW 4 trillion (~$3B)** valuation, with TS Investment named. **Confidence: LOW** — secondary aggregation of a Korean trade report; no FuriosaAI press release confirms a close.

### 7. Negative findings (recorded deliberately)

- **No MLPerf results exist for RNGD.** FuriosaAI is not among the 24 submitting organizations for MLPerf Inference v6.0 (results published 2026-04-01).
- **No FuriosaAI talk on the Hot Chips 38 program** (Aug 23–25, 2026, Stanford).
- **"RNGD-S" is not a confirmed product.** No such variant appears on furiosa.ai/rngd, in the 2026.3.0 developer docs, or in any 2026 press release. Current product names are RNGD (PCIe card), NXT RNGD Server, and Warboy / Vision NPU.

---

## Resources

### Documentation
- [FuriosaAI Developer Center — 2026.3.0 (current)](https://developer.furiosa.ai/latest/en/)
- [FuriosaAI Developer Center 2026.1.0](https://developer.furiosa.ai/latest/)
- [RNGD Overview](https://developer.furiosa.ai/latest/en/overview/rngd.html)
- [Warboy SDK Docs](http://developer.furiosa.ai/docs/latest/en/npu/warboy.html)
- [Python SDK Guide](http://developer.furiosa.ai/docs/latest/en/software/python-sdk.html)

### Open Source / Packages
- [furiosa-sdk (GitHub)](https://github.com/furiosa-ai/furiosa-sdk) — ⚠️ legacy Warboy line (0.9.2, June 2024); not the 2026.x release channel
- [FuriosaAI GitHub Organization](https://github.com/furiosa-ai)
- [PyPI: furiosa-kernels 2026.3.0](https://pypi.org/pypi/furiosa-kernels/json) — TCL kernels, `LicenseRef-Proprietary`
- [PyPI: furiosa-llm 2026.3.0](https://pypi.org/pypi/furiosa-llm/json) — Apache 2.0

### Technical Analysis
- [Chips & Cheese: RNGD at Hot Chips 2024](https://chipsandcheese.com/p/furiosaais-rngd-at-hot-chips-2024-accelerating-ai-with-a-more-flexible-primitive)
- [MICRO 2025: FuriosaAI RNGD TCP Paper](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-micro25.pdf)
- [HPCwire: TCP Architecture Deep-Dive](https://www.hpcwire.com/2025/09/30/the-fast-and-the-furiosaai-korean-chip-startup-takes-aim-at-nvidia-gpus-with-tensor-contraction-architecture/)
- [FuriosaAI TCP Architecture Blog](https://furiosa.ai/blog/tensor-contraction-processor-ai-chip-architecture)

### Benchmarks
- [LG AI Research EXAONE 2.25× Benchmark](https://furiosa.ai/blog/lg-ai-research-taps-furiosaai-to-achieve-2-25x-better-llm-inference-in-production-vs-gpus)
- [GPT-OSS 120B @ 5.8ms TPOT](https://furiosa.ai/blog/serving-gpt-oss-120b-at-5-8-ms-tpot-with-two-rngd-cards-compiler-optimizations-in-practice)
- [White paper: RNGD on Lablup Backend.AI vs RTX PRO 6000 Blackwell (2026-08-06, vendor)](https://furiosa.ai/blog/white-paper-benchmarking-rngd-on-backend-ai)
- [MLPerf Inference v6.0 results (2026-04-01) — FuriosaAI did NOT submit](https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/)

### 2026 Announcements
- [Broadcom partnership — 3rd-gen inference platform (2026-05-27)](https://furiosa.ai/blog/furiosaai-partners-with-broadcom-to-build-next-generation-inference-platform-for-the-agentic-era)
- [Furiosa SDK 2026.3 — TCL kernel framework (2026-06-30)](https://furiosa.ai/blog/furiosa-sdk-2026-3-a-new-kernel-framework-and-the-models-it-unlocks)
- [RENEGADE Summit 2026 — mass production, FVISA named (2026-05-13)](https://furiosa.ai/blog/experience-renegade-summit-2026)
- [Samsung SDS NPU-as-a-Service (2026-07-20)](https://furiosa.ai/blog/furiosaai-and-samsung-sds)
- [Equinix Lisbon LS2 deployment (2026-07-07)](https://furiosa.ai/blog/furiosaai-equinixs-lisbon-data-center-press-release)
- [Sweden 15 MW data center with I/ONX and Velox (2026-08-04)](https://furiosa.ai/blog/furiosaai-partners-with-i-onx-and-velox-for-new-15-mw-ai-data-center-in-sweden)
- [FuriosaAI at ICML 2026 (2026-08-06)](https://furiosa.ai/blog/furiosaai-at-icml-2026-advancing-full-stack-software-efficiency)
- [FuriosaAI Newsroom](https://furiosa.ai/newsroom)

### Business News
- [Meta $800M Rejected (TechCrunch)](https://techcrunch.com/2025/03/24/ai-chip-startup-furiosaai-reportedly-turns-down-800m-acquisition-offer-from-meta/)
- [Series D $500M Target (DCD)](https://www.datacenterdynamics.com/en/news/furiosaai-seeking-a-500m-funding-round-ahead-of-anticipated-ipo-report/)
- [Series D reported nearing close at KRW 800B (secondary aggregation — LOW confidence)](https://www.ai-market-watch.com/news/furiosaai-nears-completion-of-800-billion-won-series-d-valuation-at-4-trillion-w-eawtpx)
