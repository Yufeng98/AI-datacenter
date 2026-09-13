# VastaiTech (瀚博半导体) Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: vastaitech*
*device_class: GPU-like Inference Accelerator + Video Codec (China, 瀚博 VastaiTech)*

---

## Overview

**VastaiTech** (瀚博半导体（上海）股份有限公司, Shanghai Vastai Technologies), founded **December 2018** by two ex-AMD engineers — **钱军 (Jun Qian, CEO)**, formerly a Senior Director over GPU/AI server chip design, and **张磊 (Lei Zhang, CTO)**, formerly an AMD Fellow leading AI and video-codec development — is a fabless Chinese accelerator company shipping datacenter PCIe cards. It has **two chip generations in mass production**, both on a **7 nm** process whose **foundry is never named in any source**:

- **SV100** (达景 "Dajing"; the VA1 SKU is **SV102**) — a server-class AI-inference **DSA**. The vendor is explicit that this was a deliberate choice *against* building a GPGPU: "公司并没有首选做GPU，而是选择通过DSA架构来做面向AI+视频市场的芯片". Tape-out 2021-03, mass production 2022-Q1.
- **SG100** (乾元 "Qianyuan") — a **full-function GPU** with graphics (DirectX 11, OpenGL, Vulkan), AI, video codec (H.264/H.265/AV1) and SR-IOV virtualisation on one die. Silicon back 2023-02 — it ran 王者荣耀 and 原神 within 24 hours — mass production 2023-04.

Both generations sit under the **VUCA** architecture umbrella (*Vast(ai) Unified Compute Architecture*, announced September 2022), which the vendor describes as three acceleration engines on one chip: streaming media, inference, and graphics.

> ### The single biggest research trap for this vendor
>
> **`vastaitech.com` is stale.** Its milestone timeline and awards list both stop in 2023, and the 南禺/Nanyu nav entry is a dead placeholder. The **current 2026 product line — VA16, VA10L, VA1L — appears nowhere on the marketing site.** It is documented only in `github.com/Vastai`, the `vllm-vacc.vastaitech.com` recipe site, and 2025–2026 Chinese press. Anyone researching this chip from the website alone will document the wrong product generation and will conclude, wrongly, that VastaiTech is a video/CV inference vendor. In 2026 it is a datacenter **LLM serving** platform running DeepSeek-V3 671B in native FP8.

### The second structural fact: a card is N dies

Every VastaiTech card presents **N independent devices ("dies")** to software, each with its own memory pool and `device_id`. The SDK's own `card_info` sample prints `Card type: VA1-16G  Die ID: 0, 1` with 8 GB per device — so **VA1-16G = 2 dies × 8 GB**. The vLLM recipes state the current cards as **`VA16128G (4×32G)`** and **`VA10L128G (4×32G)`** — 4 dies × 32 GB. This is why DeepSeek-V3 on "8× VA16" runs at **TP=32**: 8 cards × 4 dies = 32 tensor-parallel ranks. Every runtime knob (`VACC_VISIBLE_DEVICES`, `--llm_devices`, `--tensor-parallel-size`) operates at die granularity.

**Whether a "die" is a physically separate silicon die in a multi-chip package or a partition of a monolithic die is not disclosed.**

### The third: almost nothing quantitative is public

**No memory technology (HBM / GDDR / LPDDR / DDR) and no memory bandwidth figure exists for any VastaiTech part ever shipped**, in Chinese or English. No datasheet, no teardown, no reviewer measurement. Peak TOPS/TFLOPS, TDP, form factor, PCIe generation and process node are all **undisclosed for the current VA16 / VA10L / VA1L cards** — published figures exist only for the 2021–2023 parts. There is **no ISCA / MICRO / Hot Chips / ISSCC / arXiv publication** of any kind, and **no third-party benchmark**.

---

## Software Stack

The compute target is called **VACC** throughout ("Vastai Accelerated Computing"; the vendor never expands it). The 2026 delivery vehicle is **VVI (Vastai Versatile Inference)**, currently **VVI-26.02**, obtainable only from a sales-gated developer centre.

### Two parallel runtime paths that share nothing above the driver

```
                      Build_In path                    vLLM path
Framework      VastModelZOO / VastGenX          vLLM (vllm_vacc) / Xinference / MinerU
Graph capture  HF-modeling patch + AOT trace    none (--enforce-eager mandatory)
Graph compiler VAMC  (backend: tvm_vacc)        — (eager op dispatch)
Kernel level   VDSP ELF ops + VNNL (closed)     VNNL / torch_vacc kernels (closed)
Tensor API     VastStreamX (vsx) / VACM tensors torch.Tensor on the "vacc" device
Runtime        VACL / VACE / VAME + Stream      vllm_vacc + torch_vacc runtime
Collectives    VCCL                             VCCL
Mgmt           VAML → vasmi / VAProfiler / VASID / valogger; VastCloudNative (k8s)
Driver         vastai_pci.ko  (/dev/vastai0)
ISA            not disclosed
```

A model deployed on one path tells you almost nothing about its behaviour on the other. **Build_In** is the mature path for CV/NLP; **vLLM** is where all 2026 LLM/VLM work happens.

### Graph compiler — VAMC is TVM-based, not MLIR-based

**This is the most important, most-often-missed fact about the stack.** All 344 compile configs in the public model zoo set `backend.type: tvm_vacc` (594 occurrences), the calibration dataset loader is `dataset.type: tvm`, and the compiled artifact is a TVM-style module prefix `deploy_weights/<name>/mod`. VastaiTech is in the TVM-lineage camp, not the MLIR camp that most 2024+ Chinese accelerator compilers occupy.

Two architectural consequences follow from the VAMC config schema:

1. **Tensor parallelism is compiled in, not scheduled at runtime** on the Build_In path (`model_kwargs.tp: 4`), along with batch size and I/O shapes. Changing TP degree means recompiling. The vLLM path takes `--tensor-parallel-size` at launch instead.
2. **The compiler owns memory placement**: `output_ddr: [-1, 1]` forces named graph outputs to off-chip DDR, `mem_inplace`, `data_transport_mode`, `cluster_mode`, and a `graph.extra_ops: insert_odma` pass that inserts explicit output-DMA nodes. A hardware-cached memory would not need a compiler flag naming DDR as a destination — this is the strongest available evidence that the on-chip memory is a **software-managed scratchpad**. (The flags are confirmed; the conclusion is inferred. No vendor statement confirms the absence of a hardware data cache.)

Quantisation coverage is broad: FP16, BF16, INT8 PTQ (per-channel, four calibration modes), W8A16-GPTQ, W4A16, INT4, FP8, and **FP4** on VA16 as of 2026. A quirk worth noting: GPTQ calibration in `vamc_quant.yaml` can be run on `device: cuda:5` — calibration on NVIDIA hardware, deployment on VACC.

### Graph capture — invasive on Build_In, absent on vLLM

Build_In has no runtime capture. Models are traced ahead of time from a **patched** HuggingFace modeling file: every LLM except LLaMA requires the user to add a `config_vacc.json` with an `auto_map` pointing at a vendor-modified `xxx_modeling_xxx_vacc.py`, plus `"_attn_implementation": "eager"` and `"insert_slice": true`. This source-level mechanism is the main portability tax of the path — every new model architecture needs a vendor-authored patch before it can be compiled.

On the vLLM path there is **no capture at all**. `--enforce-eager` appears in every documented command, and the MinerU integration states it as a rule. There is no CUDA-Graph or `torch.compile` equivalent.

### Kernel layer — no public kernel compiler and no kernel language

This is a real gap relative to peers: no Triton port, no TileLang, no assembler, no intrinsics header, no disassembler. **VDSP custom operators ship as pre-built ELF binaries** at `/opt/vastai/vaststreamx/data/elf/<op_name>`; users load one with `vsx::CustomOperator(op_name, elf_file)` and drive it by filling a packed C struct of device addresses. The toolchain that *produces* those ELFs is not distributed. VastaiTech's own engineers clearly write VDSP kernels — the Qwen-VL visual rotary embedding is one — but customers receive only binaries. **A customer cannot write a new fused kernel for this hardware.**

**VNNL** is inferred to be the low-level neural-network kernel library (a cuDNN analogue) but is visible *only* through two environment variables (`VNNL_MODEL_SYNC`, `VNNL_CONV1D_DLC=1`). The vendor has never mentioned it publicly. Flag it as inferred.

### Tensor API and runtime

**VastStreamX** (`import vaststreamx as vsx`, release 26.04 dated 2026-05-09) is the modern high-level API: `vsx.Tensor`, `vsx.Image`, `Context::CPU()` / `Context::VACC(device_id)`, explicit `from_numpy`/`as_numpy` transfers, `vsx::Graph` + `vsx::Stream` dataflow execution, and `vsx.Card`/`vsx.Die` telemetry. **There is no unified or managed-memory abstraction** — every host↔device movement is an explicit user action.

Beneath it, the original **VastStream** C SDK decomposes into five libraries: **VACM** (common — device context, tensor and memory management), **VACE** (compute engine / operator library), **VACL** (Accelerate Language — the inference API), **VAME** (media engine), **VAML** (management, backing `vasmi` and VAProfiler). No API reference documentation for any of these is public.

Fleet tooling is complete on paper: **VastCloudNative** (Kubernetes Operator with a device plugin and a Prometheus exporter), **VastDCManager** (`vasmi` as the `nvidia-smi` analogue, VAProfiler, VASID diagnostics, `valogger`), and `vamp` as a `trtexec` analogue.

### Collectives — VCCL, essentially undocumented

**VCCL** is the NCCL analogue. It is visible only through `VCCL_SOCKET_IFNAME` (default `"lo"`), `VCCL_MODEL_SYNC`, and the compile flag `gather_data_vccl_dsp_enable` — the last of which indicates collectives can be **executed on the VDSP engines**, which is architecturally interesting. Supported operations, algorithms, achieved bandwidth and API are all **not disclosed**. `VCCL_SOCKET_IFNAME` mirroring `NCCL_SOCKET_IFNAME` implies a socket-based bootstrap or out-of-band path.

### Self-disclosed runtime limits

From the vendor's own `tools/vllm/usage_limits.md` and the MinerU support matrix:

- **max-concurrency 4** across essentially every model — DeepSeek-V3/R1 671B, all Qwen3 sizes, and even BGE-small embedding models.
- **Data parallelism is not supported** (`--data-parallel-size`/`--dp` 🔴).
- **Eager mode is mandatory.**
- `min_p` sampling is unsupported and errors out; requests exceeding the context window are silently intercepted rather than handled.
- DeepSeek-V3: TP32 → 56 K input / 64 K context with MTP; TP32-PP2 → 100 K input / 128 K context without MTP.

A max concurrency of 4 on a 671 B serving configuration is a very low figure and is the strongest self-disclosed signal that the vLLM port is functional but not performance-mature.

### Open source versus proprietary

Four repos under `github.com/Vastai`, all active in 2026: **VastModelZOO** (Apache-2.0, 1000+ models, 30★, last push 2026-08-07), **VastStreamX-Samples** (MIT, 3★, 2026-08-03), **xinference_vacc** (Apache-2.0, 4★, 2026-08-07), and a **MinerU** fork (GPL-3.0, 2026-02-06). Every one of them is samples, recipes, configs and Dockerfiles. **Not one line of the driver, compiler, runtime, kernel library, or collective library is open.**

The developer centre is genuinely gated, and this was verified: `developer.vastaitech.com` serves an identical React SPA shell on every route, and every product/documentation API endpoint returns `{"code":4000,"message":"错误的账户或秘钥，请检查"}`. VastModelZOO's README confirms the policy — "需联系销售代表获取瀚博开发者中心版本权限".

The one genuinely third-party-hosted integration document is upstream OpenDataLab's MinerU page, which lists VastAI as a supported acceleration card.

---

## Hardware Architecture

### VUCA on-die organisation

The most technically specific public description of the block structure is a 2022-09-08 CEO interview with 芯东西: a **high-performance compute engine**, a **high-performance AI engine** (the matmul/conv datapath), a **programmable vector compute engine** (the **VDSP**), **dedicated video decode and graphics render/display cores**, and **unified memory management with coherent interfaces and low-latency interconnect**.

The SDK corroborates this independently. `vsx::DieUtilization` reports exactly four engine classes per die:

```
util.ai        // AI engine   — scalar: one AI domain per die
util.vdsp[]    // array       → multiple VDSP vector cores per die
util.vdmcu[]   // array       → multiple video-DECODE MCUs per die
util.vemcu[]   // array       → multiple video-ENCODE MCUs per die
```

**The counts are not disclosed** — the arrays are sized by a non-public header. **The internal structure of the AI engine — MAC array dimensions, systolic versus SIMD, MACs per cycle, native tile size — is not described anywhere.**

There are at least **three independent clock domains**: OCLK (operator/AI, quoted at 835 MHz and 880 MHz in samples), DCLK (decode, 650 MHz), ECLK (encode, 200 MHz). Dynamic power management is user-controllable via `vasmi setconfig dpm=enable`.

### Card lineup and published specifications

| Card | Generation | Compute | Power | Memory | Codec / notes |
|---|---|---|---|---|---|
| **载天 VA1** (SV102) | SV100, 2021 | >200 TOPS INT8 per chip; FP16/BF16/INT8 | **70 W** (vendor page) / **75 W** (launch coverage) — sources conflict | 32 GB at launch; SDK shows a VA1-16G SKU = **2 dies × 8 GB** | HHHL single-width, PCIe 4.0 ×16, no aux power; 120 ch 1080p30 decode; 64+ ch H.264/H.265/**AVS2** at launch; 8K; JPEG codec |
| **载天 VA10** | SV100, 2022 | 400 TOPS INT8 | 150 W | not disclosed | FH 3/4-length; 240 ch 1080p30 decode; 100 ch 1080p30 transcode |
| **载天 VA1V** | video | not disclosed | not disclosed | not disclosed | H.264/H.265/**AV1**, 8K 10-bit, 2× 8K HDR @60+fps encode/transcode, FFmpeg |
| **载天 VA1L** (2023) | SG100, WAIC 2023 | 200 TOPS INT8 / 72 TFLOPS FP16 | not disclosed | 64 GB (derived from "8 × VA1L = 512 GB") | AIGC appliance claimed to support 175 B models |
| **VA12** | SG100, WAIC 2023 | 512 TOPS INT8 / 160 TFLOPS FP16 | 250 W | not disclosed | High-performance generative-AI card |
| **南禺 VG1600 / VG1800 / VG14** | SG100, WAIC 2023 | not disclosed | not disclosed | not disclosed | Cloud gaming / cloud desktop / workstation. **All per-card specs undisclosed**; vendor's own nav entry is a dead placeholder |
| **载天 VA16** | 2026 flagship | **not disclosed** | **not disclosed** | **128 GB = 4 dies × 32 GB** | "训推一体" per OEM certification; **FP4 + FP8** as of 2026; 8-card = 1 TB, 16-card = 2 TB appliances |
| **载天 VA10L** | 2026 | **not disclosed** | **not disclosed** | **128 GB = 4 dies × 32 GB** | — |
| **载天 VA1L** | 2026 | **not disclosed** | **not disclosed** | **not disclosed** (2023 figure was 64 GB) | Listed as supported throughout the 2026 stack |

*VE1S / VE1M / VE1V / VS1000 are edge/embedded parts and are out of scope for this datacenter survey entry.*

### Memory

| Level | Status |
|---|---|
| Register file | not disclosed |
| On-chip SRAM / scratchpad | Capacity and bandwidth **not disclosed**. Strongly inferred to be **software-managed**, from the VAMC placement flags |
| Per-VDSP local buffer | Not disclosed but bounded — the shipped `planar_argmax` op documents "channel number ≤ 96" for an fp16 planar tensor |
| Tiling granularity | LLM compilation requires **sequence length to be a multiple of 16** |
| Off-chip capacity | Known per die and per card (8 GB/die on VA1-16G; 32 GB/die and 128 GB/card on VA16 and VA10L) |
| **Off-chip technology** | **not disclosed — every part** |
| **Off-chip bandwidth** | **not disclosed — every part** |
| Runtime model | Explicit host↔device copies only; **no unified/managed memory** |

The bandwidth hole is the most consequential gap in this entry. VA16 is a capacity-first 128 GB card serving 671 B MoE models, where memory bandwidth is precisely what determines decode performance — and it is the number the vendor does not print. **Do not estimate it from capacity, die count, or any peer part.**

### Interconnect

- **On-chip NoC** — described only qualitatively ("coherent interfaces and low-latency interconnect"). Topology and bandwidth not disclosed.
- **Die-to-die within a card** — exists by construction (2- and 4-die cards), but the fabric, protocol and bandwidth are not disclosed. **No name analogous to NVLink, xGMI or SG-Link appears anywhere in any source.**
- **Card-to-card scale-up** — not disclosed. TP=32 across 8 cards demonstrably works, so cross-card collectives function at tensor-parallel granularity, but no fabric is ever named and **there is no evidence of a proprietary switch or cabled fabric**. PCIe is the only documented host interface.
- **Scale-out** — `--pipeline-parallel-size 2` for the 100 K-input DeepSeek-V3 configuration; **data parallelism explicitly unsupported**. Whether VCCL has any inter-node RDMA transport is not disclosed.
- **Host interface** — PCIe 4.0 ×16 for VA1 only; not disclosed for VA10 / VA16 / VA10L / VA1L. PCI device ID appears to be **0x0100**.

### Execution model

Build_In is **ahead-of-time and statically compiled**: a whole model becomes a fixed `mod` artifact with TP degree, batch size and I/O shapes baked in. There is no JIT. At runtime, VastStreamX builds a `vsx::Graph` of operators wrapped in a `vsx::Stream`, executed synchronously or fully asynchronously. **Multiple concurrent model instances per die** (`--instance 4`) are the documented route to peak throughput — the die is oversubscribed by independent streams rather than by one large kernel.

Heterogeneous op placement is **explicit and manual**: preprocessing runs on VDSP via a JSON-selected fusion op; some ops must run on the host CPU (Qwen-VL `Smart_Resize`, RetinaNet post-processing); some structural ops are hand-mapped to VDSP. The `VASTAI_*` family of per-operator environment switches on the vLLM path points the same way — operator coverage is managed by hand, switch by switch, rather than by a general lowering.

### Measured performance — the vendor's own sample docs, per die

These are the only reproducible absolute numbers published anywhere for this hardware, and they are **per die, not per card**:

| Workload | Result |
|---|---|
| H.264 1080p decode @ DCLK 650 MHz | **1547 fps** max throughput (10 instances); 509 fps min latency → ≈51 ch @30 fps per die |
| H.265 1080p decode @ DCLK 650 MHz | **1861 fps** max throughput → ≈62 ch @30 fps per die |
| ResNet-50 INT8 @ 880 MHz | **3231 qps** (batch 8); 941 qps (batch 1, min latency) |
| MobileViT | 43.9 qps (batch 1) |
| `planar_argmax` custom op, [19,512,512] fp16, 4 instances | 3008 qps, p50 1329 µs |

These cross-check the VA1 spec sheet and independently confirm the 2-die structure: ≈51–62 ch/die × 2 dies ≈ 102–124 channels against VA1's claimed 120 ch.

---

## Vendor Claims Versus the Vendor's Own Benchmark

Every headline claim on vastaitech.com is **relative and unbaselined**: "同等功耗下2倍以上于主流GPU的最高吞吐率", "延时不到GPU最高吞吐率下延时的5%", VA10's "推理性能达到同功耗主流GPU的2倍以上，延时低至6%", VA1's "2–10× AI throughput of GPUs at equal power", DSA "3–5× traditional GPUs". No named comparison part, no workload, no date. **All are vendor claims.**

The vendor also published, in its own model zoo, a table that cuts the other way. `vlm/glm_ocr/vllm/README.md` compares backends on OmniDocBench end-to-end with the layout model on CPU in every row:

| Backend | Overall ↑ | Model Infer Cost |
|---|---|---|
| NVIDIA H800 BF16 TP1 | 95.391 | **15.5 min** |
| VACC-VA16 BF16 TP1 | 95.647 | **13 h 26 min** |
| VACC-VA16 BF16 TP2 | 95.611 | 9 h |
| VACC-VA16 BF16 TP4 | 95.635 | 6 h 33 min |

**Accuracy is on par with — marginally above — H800. Wall clock is roughly 50× slower at TP1 and ~25× slower at TP4.**

Caveat it properly: one workload, an OCR VLM, full-dataset wall clock, layout model on CPU in all rows, and VA16 TP1 uses a single die against a whole H800. But it is the vendor's own apples-to-apples table and it is the most useful maturity signal available for this vendor. It should be cited with its caveats — neither suppressed nor sharpened.

---

## Maturity and Deployment Status

**Verdict: shipping / commercially available — NOT verified deployed at scale.**

| Signal | Evidence |
|---|---|
| Two generations in mass production | Vendor states "量产并商业化落地" — **vendor claim** |
| Real commercial shipment | Published **3-year hardware + software warranty** with a formal RMA process; sales-gated developer centre; private Docker harbor with versioned releases (VVI-25.12.SP2, VVI-26.02) |
| Server-OEM validation | 通泰易 TG657V2 / TG658V3 / TG659V2 certified with the VA16 "训推一体" card, 2025-06-13 |
| Partnership | 海马云 (Haima Cloud) strategic partnership, 2024-04-10, targeting a **thousand-card** cloud-rendering/AI cluster on domestic ARM CPUs + VastaiTech GPUs |
| Software activity | All four GitHub repos pushed within a week of 2026-08-08; VastStreamX release 26.04 dated 2026-05-09 |
| **No named end customer with disclosed scale** | Kuaishou reported by 界面新闻 in 2021 as both investor and customer — no deployment size given |
| **No third-party benchmark** | None exists in any language |
| **No academic publication** | No ISCA / MICRO / Hot Chips / ISSCC / arXiv paper of any kind |
| **Self-disclosed serving limits** | max-concurrency 4; no data parallelism; mandatory eager mode |

### Corporate / IPO

VastaiTech entered **IPO counselling (辅导备案)** — the pre-application stage — with **CITIC Securities (中信证券)** on **2025-07-18**, targeting a **domestic A-share listing**. This is confirmed by a **primary filing document**: CITIC's first counselling progress report, which records the agreement signed 2025-07-11, the filing submitted the same day to the Shanghai CSRC bureau, counsel 北京市中伦律师事务所, auditor 天健会计师事务所, and shareholders above 5%: VASTAI Holding Company 11.53% (LEI ZHANG), ACE REDPOINT CHINA VASTAI HK LIMITED 7.12%, ZHEN PARTNERS V (HK) LIMITED 6.48%, 5Y CAPITAL VASTAI HOLDING LIMITED 5.28%.

> **Do not write that VastaiTech has filed an IPO application.** An enumeration of all 1,046 filings in the SSE STAR Market IPO review database returned no 瀚博 record — which is exactly what counselling-stage-only status predicts. The two findings reconcile.

Vendor-published funding: Series A US$50 M (2020-11), Series A+ ¥500 M (2021-04), Series B1&B2 ¥1.6 B (2021-12). Reported cumulative raise >¥2.5 B and a ~¥10.5 B 2025 valuation are **media figures, not vendor-published**. Investors reported to include Alibaba, Kuaishou, Sequoia, 5Y Capital, Zhen Partners and Redpoint China.

---

## Competitive Position

VastaiTech's distinctive position in this survey is **capacity-first, codec-heavy, domestically-oriented inference** — and an unusually honest paper trail about its own immaturity.

- **A genuine multi-modal die.** The AI engine, a programmable vector DSP, and hard video encode/decode MCUs sit on the same die with unified memory. Very few accelerators in this survey ship 120–240 channels of 1080p decode alongside LLM serving. For cloud gaming, cloud desktop, video analytics and OCR-style document pipelines, that integration is the product.
- **Capacity per card is competitive; bandwidth is unknown.** 128 GB per VA16, 1 TB per 8-card appliance, 2 TB per 16-card appliance is real capacity for long-context and MoE serving. But **without a published bandwidth figure, VA16 cannot be placed on a roofline against any peer**, and the OmniDocBench result suggests the effective throughput is far from H800-class.
- **The die-granular programming model is a genuine architectural difference.** Peers expose one device per chip; VastaiTech exposes 4 devices per card with independent memory pools, and pushes tensor parallelism down to die granularity. TP=32 on 8 cards is TP across dies, not across chips in the usual sense. Any TP number quoted for this vendor must be divided by the die count before comparison.
- **Weaker than peers on programmability.** No kernel language, no Triton port, no intrinsics, no ISA documentation, no authoring path for VDSP kernels. Compare Sophgo, whose TPU-MLIR is a fully open compiler; VastaiTech publishes the compile *configs* for 344 models but not the compiler that consumes them.
- **TVM lineage, not MLIR.** VAMC's `tvm_vacc` backend puts it in an older compiler lineage than most 2024+ Chinese accelerator stacks. It works — 1000+ models are compiled in the zoo — but it is a different bet.
- **Self-disclosed limits are the most credible data available.** max-concurrency 4, no data parallelism, mandatory eager mode, and a vendor-published 25–50× wall-clock gap against H800 on one VLM workload. No vendor publishes these numbers unless they are real, and no third party has published anything to contradict or contextualise them.

**Every quantitative comparison involving VastaiTech in this survey must carry the caveat that peak throughput, memory bandwidth, memory type, TDP and process node are undisclosed for the current generation.** The correct entry in a comparison table is "not disclosed", not an estimate.

---

## Sources

Vendor primary:
- [About — company history and milestone timeline](https://www.vastaitech.com/company/about)
- [载天 VA1 product page](https://www.vastaitech.com/product/general/va1)
- [载天 VA10 product page](https://www.vastaitech.com/product/general/va10)
- [载天 VA1V video card](https://www.vastaitech.com/product/video/va1v)
- [VastStream SDK — VACM/VACE/VACL/VAME/VAML](https://www.vastaitech.com/software/vaststream)
- [VastCloudNative — Kubernetes Operator, device plugin, Exporter](https://www.vastaitech.com/software/vastcloudnative)
- [VastDCManager — VASMI, VAProfiler, VASID](https://www.vastaitech.com/software/vastdcmanager)
- [Support — 3-year warranty and RMA process](https://www.vastaitech.com/support)
- [Developer centre — gated; all product/doc routes return 错误的账户或秘钥](https://developer.vastaitech.com/)
- [vLLM × VastAI recipe site (VA16 / VA10L / VA1L, 31 recipes, vLLM 0.17.0)](https://vllm-vacc.vastaitech.com/)
- [vLLM recipe — DeepSeek-V3 (8× VA16, TP32, FP8)](https://vllm-vacc.vastaitech.com/deepseek-ai/DeepSeek-V3)
- [vLLM recipe — Qwen3-32B ("VA16128G (4×32G) / VA10L128G (4×32G)")](https://vllm-vacc.vastaitech.com/Qwen/Qwen3-32B)

Vendor's own code:
- [Vastai/VastModelZOO (Apache-2.0, 1000+ models)](https://github.com/Vastai/VastModelZOO)
- [VAMC compiler config schema (`backend.type: tvm_vacc`)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_config.yaml)
- [VAMC quantisation config (w8a16_gptq, calibration datasets)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_quant.yaml)
- [LLM compile guidance (seq-len multiple of 16, VACC_STACK_SIZE=256)](https://github.com/Vastai/VastModelZOO/blob/main/llm/README.md)
- [DeepSeek-V3/V3.1 deployment — TP32, TP32-PP2, MTP](https://github.com/Vastai/VastModelZOO/blob/main/llm/deepseek_v3/README.md)
- [vLLM usage limits — max-concurrency 4, min_p unsupported](https://github.com/Vastai/VastModelZOO/blob/main/tools/vllm/usage_limits.md)
- [GLM-OCR OmniDocBench — H800 vs VACC-VA16](https://github.com/Vastai/VastModelZOO/blob/main/vlm/glm_ocr/vllm/README.md)
- [Vastai/VastStreamX-Samples (MIT, release 26.04)](https://github.com/Vastai/VastStreamX-Samples)
- [card_info sample — VA1-16G, 2 dies × 8 GB, util{ai,vdsp[],vdmcu[],vemcu[]}](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/card_info/card_info.cpp)
- [video_decode sample — per-die 1547 / 1861 fps @ DCLK 650 MHz](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/video_decode/README.md)
- [Vastai/xinference_vacc](https://github.com/Vastai/xinference_vacc)
- [Vastai/MinerU — driver/version matrix, DP unsupported, --enforce_eager mandatory](https://github.com/Vastai/MinerU)

Third-party:
- [OpenDataLab MinerU — VastAI acceleration-card documentation](https://opendatalab.github.io/MinerU/zh/usage/acceleration_cards/VastAI)
- [通泰易 2025-06-13 — TG657V2 / TG658V3 / TG659V2 certified with VA16](http://ttyinfo.com/News/info/id/154.html)

Press:
- [腾讯新闻 2021-07-08 — SV100 + VA1 launch](https://news.qq.com/rain/a/20210708A032OI00)
- [界面新闻 — SV100 is a DSA; VA1 = SV102; founders' AMD background](https://m.jiemian.com/article/6342672.html)
- [芯东西 zhidx 2022-09-08 — CEO interview, VUCA block list](https://zhidx.com/p/344936.html)
- [36Kr 2022-09-05 — VA10 400 TOPS / 150 W](https://www.36kr.com/p/1901732567984512)
- [量子位 2023-07 — WAIC 2023: SG100, Nanyu VG-series, VA1L, VA12](https://www.qbitai.com/2023/07/66614.html)
- [36Kr 2023-07-06 — independent confirmation of the WAIC 2023 figures](https://www.36kr.com/p/2332635957528066)
- [上海证券报 via Sina Finance 2026-04-27 — VA16 128 GB, FP4+FP8, 2 TB appliance](https://finance.sina.com.cn/roll/2026-04-27/doc-inhvxtrc2314774.shtml)
- [同花顺 10jqka 2026-04-27 — same story, verbatim Chinese quotes](https://news.10jqka.com.cn/20260427/c676308817.shtml)
- [iCloudNews 2024-04-10 — 海马云 partnership, thousand-card cluster](https://www.icloudnews.net/a/79464.html)
- [eeNews Europe — China's Vastai launches 7nm GPU](https://www.eenewseurope.com/en/chinas-vastai-launches-7nm-gpu-for-ai-visual-apps/)

Regulatory:
- [CITIC Securities IPO counselling progress report (PDF, primary filing)](https://www.cs.ecitic.com/newsite/tzgg/ipoqyfdgg/202510/P020251023518196738538.pdf)
- [SSE STAR Market IPO review database — negative result, 1,046 filings, no 瀚博](https://query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB)
