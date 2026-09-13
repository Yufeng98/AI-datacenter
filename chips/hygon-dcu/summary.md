# Hygon DCU (海光深算) Software and Hardware Stack Summary

*as_of: 2026-08-08*
*chip: hygon-dcu*
*device_class: GPU/DCU (China, 海光)*

---

## Overview

Hygon (海光信息, SHA: 688041) is a publicly listed Chinese fabless semiconductor company that produces both x86 CPUs (Hygon C86 / Dhyana, licensed from AMD Zen-1 via a 2016 JV) and GPU-class **DCU (Deep Computing Unit / 深算处理器)** accelerators for AI training and HPC. Added to the US Entity List in June 2019, Hygon develops the DCU independently.

The current DCU flagship is **深算三号 (Shensuan No.3), marketed model 海光深算三号 BW1000** — Hygon stated on 2025-12-11 that it "已经投入市场，受到客户认可" (is in market and has been accepted by customers). **No Hygon datasheet, filing, or conference disclosure publishes any BW1000 specification**, so every quantitative figure in this document is scoped to the prior generations: **深算一号 (DCU-Y100)**, **深算二号 (DCU-Z100)**, and the **K100_AI** AI variant (64 GB, 400 W), which was the flagship recorded in the 2026-04-05 baseline of this survey. See the *深算三号 (BW1000) Update* section below.

A Hygon–Sugon (中科曙光) share-exchange absorption merger was announced in May 2025 but was **formally terminated by Hygon's board on 2025-12-09**; Sugon remains independently listed as 603019.SH.

The DCU is architecturally derived from AMD's GCN/Vega lineage and runs the same **ROCm / HIP programming model**. Hygon wraps this in the **DTK (DCU ToolKit)** platform, distributed at `developer.sourcefind.cn`, which mirrors every ROCm layer: hipcc compiler, hipBLAS, hipDNN/MIOpen, RCCL, and debugging/profiling tools. The CUDA-to-DCU migration path is automated: CUDA → hipify → hipcc on DTK.

---

## Software Stack

### Framework Integration

- **PyTorch**: Runs via DTK ROCm HIP compatibility layer. `torch.cuda` → `torch.hip` aliasing. PyTorch 2.4+ confirmed with DTK 24/25.
- **PaddlePaddle**: First-class DCU support — the recommended framework for DCU in China. Official PaddlePaddle docs cover DCU install, multi-GPU training, and mixed precision.
- **TensorFlow**: Supported via DTK ROCm backend (TF 2.13+).
- **vLLM / GPUStack**: Community vLLM port; GPUStack official DCU inference tutorial for K100_AI.
- **FlyAIBox/dcu-in-action**: Community GitHub repo with LLaMA pre-training, ChatGLM fine-tuning, distributed training recipes.
- **Tencent Hunyuan Hy3** *(added 2026-08-08)*: adaptation to the open-source Hy3 preview reported complete in May 2026 — **Baidu Baike sourcing only**, low-to-medium confidence. DeepSeek V4 "priority adaptation" is claimed by Hygon channels but **not independently confirmed**.

### Compiler / IR

- **hipcc (DTK LLVM)**: Primary compiler driver. HIP C++ → LLVM IR (amdgcn target) → DCU ISA. Includes clang, OpenMP, OpenACC support.
- **hipify-perl / hipify-clang**: CUDA → HIP automated translation (~90–95% automated). Included in DTK.

### Op Library

- **hipDNN / MIOpen**: cuDNN analog. Convolution, BatchNorm, Pooling, Attention, LSTM.
- **hipBLAS**: cuBLAS analog. GEMM, BLAS L1–L3, batched GEMM.
- **hipSPARSE, hipFFT, hipRAND**: Sparse BLAS, FFT, random number generation.

### Kernel Library

- **hipThrust / hipCUB / rocPRIM**: Parallel primitives — Reduce, Scan, Sort, Histogram.
- **rocBLAS**: Architecture-optimized lower-level GEMM kernels.

### Runtime

- **HIP Runtime (hiprt)**: cudart analog. `hipMalloc`, `hipMemcpy`, streams, events, kernel dispatch, unified memory.
- **ROCr / HSA Runtime**: Lower-level HSA agent model (CUDA Driver API analog). Rarely accessed directly.

### Driver / Firmware

- **DCU kernel driver (.ko)**: PCIe BAR, DMA, IOCTL, IRQ. Proprietary (not open-sourced by Hygon).
- **HAMi / dcu-vgpu-device-plugin**: Open-source Kubernetes device plugin for DCU vGPU sharing, memory quotas, core limits.

### Communication

- **RCCL (DCU port)**: AllReduce, AllGather, ReduceScatter, Broadcast. Intra-node via xGMI; inter-node via RoCE/Ethernet.
- **xGMI peer-to-peer**: Direct HBM-to-HBM transfers between DCU cards within a node (AMD Infinity Fabric compatible protocol).

### Assembler / ISA

- **DCU ISA (GCN/Vega-derived)**: Not independently published. LLVM amdgcn backend targets DCU. 64-thread wavefront; VGPR/SGPR register files; `v_`, `s_`, `ds_`, `buffer_` instruction prefixes. hipcc handles all ISA targeting.

---

## Hardware Architecture

> **Scope note (2026-08-08):** every figure in this section describes **深算一号 (Y100) / 深算二号 (Z100/Z100L) / K100 / K100_AI**. Hygon has disclosed **no** compute, memory, process-node, TDP, or interconnect specification for the current 深算三号 (BW1000) flagship. Do not read these numbers as current-generation.

### Compute Engine

| Attribute | Value |
|-----------|-------|
| Compute Unit (CU) | ~60–64 CUs (Z100/K100 class) |
| SIMD structure | 4× SIMD16 per CU (64 FP32 ALUs/CU) |
| Clock | ~1.7 GHz |
| Wavefront | 64 threads (vs NVIDIA 32-thread warp) |
| Peak FP32 | ~45 TFLOPS (Y100) / 90 TFLOPS (Z100 class) |
| Peak FP16 | ~90 TFLOPS (Y100) / 180 TFLOPS (Z100 class) |
| Data types | FP64, FP32, FP16, BF16, INT8 |

### Memory Hierarchy

| Level | Spec |
|-------|------|
| VGPR (per SIMD) | 64 KB (256× 32-bit/thread max) |
| LDS/Shared Memory | 64 KB per CU (SW-managed) |
| L2 Cache | ~4–8 MB (HW-managed, chip-global) |
| Off-chip Memory (Z100) | 16–32 GB HBM2, ~1 TB/s |
| Off-chip Memory (K100_AI) | 64 GB HBM2e (est.), ~1.2 TB/s, 400 W |
| Host Interface | PCIe Gen4 x16 |

### Interconnect

| Scope | Technology | Bandwidth |
|-------|-----------|-----------|
| Scale-up (intra-node) | xGMI (AMD Infinity Fabric compatible) | 184 GB/s (Y100) |
| Scale-out (inter-node) | Standard Ethernet / RoCE v2 | Standard NIC |
| Host | PCIe Gen4 x16 | ~32 GB/s |

---

## 深算三号 (BW1000) Update — 2026-08-08

*Updated 2026-08-08. This section corrects a **pre-existing error** in the 2026-04-05 baseline, which recorded K100_AI as the current flagship and the Sugon merger as a live transaction. 深算三号 was launched in 2025 and publicly deployed by December 2025 — it was missed by the original pass, not newly announced in the April–August 2026 window.*

### Flagship correction

Hygon's current DCU flagship is **深算三号 (Shensuan No.3)**, marketed model **海光深算三号 BW1000** (the series reference "DCU 8300" appears in secondary listings only). It supersedes the 深算二号 / K100_AI-class part previously recorded here as flagship.

**What Hygon itself has said.** The strongest first-party statement is Hygon's 2025-12-11 reply on the Shanghai Stock Exchange investor-interaction platform (上证e互动): *"深算三号已经投入市场，受到客户认可"* — in market, accepted by customers. That is the whole of the company's public status disclosure.

**What Hygon has NOT said.** The following circulate widely but are **not** company statements and are recorded here only so future passes do not re-adopt them:

| Circulating claim | Actual provenance | Treatment |
|---|---|---|
| "Mass production initiated Q3 2025" | Chinese retail-investor posts (Toutiao / Xueqiu channel-check notes) | Rumor — not published |
| "Initial capacity 10k wafers/month → 30k in Q4 2025" | Same retail channel-check notes | Rumor — not published |
| "R&D began mid-2022; tape-out verification completed 2024" | Same | Rumor — not published |
| "Entered large-scale shipment in H1 2026 and was the primary volume driver" | Brokerage / financial-media inference layered on Hygon's earnings pre-announcement | Analyst attribution, not disclosure |

Hygon's H1 2026 业绩预告 gives only consolidated revenue and profit ranges; it does not attribute growth to 深算三号 by name.

### Confirmed deployment

The CAS-affiliated **中科南京信息高铁研究院** announced in December 2025 the **国内首发** (first domestic deployment) of 深算三号 BW1000 on its 信息高铁智算算力网 AI development platform, echoed by the Nanjing Qilin Park government newsroom on 2025-12-19/22. This is solid evidence of real deployment rather than mere announcement, and is the basis for treating BW1000 as shipping.

### Specifications — none disclosed

| Attribute | Status |
|---|---|
| Compute (FP32 / TF32 / BF16 / FP16 / INT8) | **not disclosed** |
| HBM generation, capacity, bandwidth | **not disclosed** |
| Process node | **not disclosed** |
| TDP | **not disclosed** |
| Form factor | **not disclosed** |
| CU count | **not disclosed** |
| xGMI generation, scale-up bandwidth | **not disclosed** |
| PCIe generation | **not disclosed** |

Hygon's own product page (`hygon.cn/product/accelerator`) publishes **no DCU model names or specifications at all**. There is therefore no vendor-primary spec source for BW1000.

> **Why no throughput number appears above.** A figure set of *FP32 49 TFLOPS / TF32 96 TFLOPS / BF16-FP16 192 TFLOPS / INT8 392 TOPS* circulates for BW1000. It traces to an Eastmoney 股吧 retail stock message-board post and a user-uploaded Baidu Wenku document — no datasheet, no slide, no filing. It is also **mutually contradicted** by other circulating numbers for the same part: ~20 TFLOPS FP32 (Zhihu), FP64 ~30 TFLOPS (51CTO blog), and "FP16 300T" for an alleged "Alibaba BW1000 AI variant" (Sept-2025 retail note). Because these conflict and none has a primary source, no BW1000 throughput number enters this survey.

### 深算四号 (Shensuan No.4)

In the same 2025-12-11 investor-platform reply, Hygon stated that 深算四号 R&D **"进展顺利"** (progressing smoothly). **No timeline, process node, or specification is disclosed.** A retail note claiming "明年回片" (silicon back next year) is rumor and is not adopted here. Note the date: this is a December 2025 statement, not an H1 2026 disclosure.

### Software / ecosystem

- **类CUDA (CUDA-like) positioning** and **"operator coverage >99%"** are vendor/marketing-channel claims with no primary source retrieved. Not independently verified.
- **Tencent Hunyuan Hy3**: adaptation to the open-source Hy3 preview is reported as completed in **May 2026**, but the sourcing is Baidu Baike (encyclopedia-grade) only — low-to-medium confidence.
- **DeepSeek V4 priority adaptation**: claimed, **not independently confirmed**.
- **DTK**: no 2026-dated release confirmed. The most recent versions found are **DTK 25.04** and **DTK 25.10** (2025-vintage naming); no DTK 26.x was located and `developer.sourcefind.cn` was unreachable during verification. The existing "PyTorch 2.4+ with DTK 24/25" statement is neither contradicted nor refreshed.
- No BW1000-specific DTK branch, ISA change, or new library is documented.

### Sugon merger — TERMINATED

The absorption merger recorded in the 2026-04-05 baseline as "announced May 2025" **did not happen**. Chronology:

| Date | Event |
|---|---|
| 2025-05-25 | Hygon announces plan to absorb 中科曙光 (Sugon) by share exchange |
| 2025-05-26 | Hygon shares halted |
| 2025-06-09 | Formal 重组预案 disclosed: exchange ratio **1 Sugon share : 0.5525 Hygon shares**; Sugon priced at 79.26 RMB/sh (120-day VWAP +10%), Hygon at 143.46 RMB/sh; headline value **~RMB 115.9–116.0 B** |
| 2025-06-10 | Trading resumes |
| **2025-12-09** | **Hygon's board resolves to terminate** the 换股吸收合并 and the associated matched-funds raise, citing *"市场环境较本次交易筹划之初发生较大变化，本次实施重大资产重组的条件尚不成熟"* — market conditions changed materially since planning and the conditions for the restructuring are not yet mature, compounded by the deal's scale and counterparty count. Hygon committed to plan no further major asset restructuring for at least one month, and said termination would not materially harm operations and that industrial cooperation with Sugon continues. |

Confirmed independently by Caixin, Yicai, Jiemian, Eastmoney, 观察者网 and 同花顺 (2025-12-09/10). **Sugon did not delist** and remains independently listed as **603019.SH**; a July 2026 deep-dive frames Sugon's post-termination "independent path". No source shows the deal being revived through 2026-08-08.

### H1 2026 financials (disclosed 2026-07-16)

| Metric | Range | YoY |
|---|---|---|
| Revenue | RMB 8.5–9.3 B (85–93 亿) | **+55.56% – +70.20%** |
| Net profit | RMB 1.70–1.83 B (17–18.3 亿) | **+41.50% – +52.32%** |
| Non-GAAP (扣非) net profit | RMB 1.51–1.70 B | +38.53% – +55.96% *(not independently re-verified)* |

The growth percentages are the safest citation. These are consolidated CPU+DCU figures; Hygon does not break out DCU revenue.

### Negative results (checked, not found)

- No Hygon/DCU **MLPerf** Training or Inference submission.
- No Hygon paper at **Hot Chips 2026**, **ISCA 2026**, or **ISSCC 2026**.

---

## Key Differentiators

1. **x86 + DCU same vendor**: Hygon Dhyana CPU + DCU GPU in one vendor package — unique CPU-GPU co-marketing position in China.
2. **HIP/ROCm compatibility**: Developers write standard HIP code; DTK provides full ROCm-compatible toolchain. CUDA migration path via hipify is well-documented.
3. **64-thread wavefront**: Same as AMD CDNA/RDNA. Differs from NVIDIA's 32-thread warp — a key porting consideration.
4. **Domestic xGMI**: Multi-card scale-up via AMD Infinity Fabric-compatible xGMI — no proprietary switch ASIC needed for 8-card nodes.
5. **Cloud availability**: Available on Alibaba Cloud, Tencent Cloud, Inspur, Sugon server platforms. Alibaba Cloud maintains official DTK Linux images.
6. **Entity List; Sugon consolidation abandoned**: US export restrictions since 2019. The share-exchange absorption merger with Sugon/中科曙光 announced May 2025 was **terminated by Hygon's board on 2025-12-09** — the domestic compute-stack consolidation it implied never happened, and Sugon remains an independent listed company (603019.SH) with industrial cooperation continuing.

---

## Business Context

| Event | Date |
|-------|------|
| AMD–Hygon JV formed (Zen-1 license) | 2016 |
| Hygon Dhyana CPU launched | 2018 |
| Hygon added to US Entity List | June 2019 |
| DCU Z100 first generation | ~2020 |
| Hygon IPO (STAR Market, RMB 10.6B) | August 2022 |
| K100_AI flagship (64 GB, 400 W) | 2024 |
| Hygon–Sugon absorption merger announced (1 : 0.5525, ~RMB 116 B) | 2025-05-25 (预案 2025-06-09) |
| 深算三号 (BW1000) launched; Hygon: "已经投入市场，受到客户认可" | 2025 (statement 2025-12-11) |
| 深算三号 国内首发 deployment — 中科南京信息高铁 AI platform | Dec 2025 |
| **Hygon–Sugon merger TERMINATED by board resolution** | **2025-12-09** |
| 深算四号 R&D "进展顺利" (no specs, no timeline) | 2025-12-11 |
| Tencent Hunyuan Hy3 adaptation reported (Baidu Baike only) | May 2026 |
| H1 2026 业绩预告: revenue +55.56%–70.20% YoY | 2026-07-16 |

---

## Resources

- [DTK Developer Portal — developer.sourcefind.cn](https://developer.sourcefind.cn/dtk)
- [FlyAIBox/dcu-in-action — GitHub](https://github.com/FlyAIBox/dcu-in-action)
- [HAMi DCU Support — GitHub](https://github.com/Project-HAMi/HAMi/blob/master/docs/hygon-dcu-support.md)
- [Running Inference with Hygon DCUs — GPUStack](https://docs.gpustack.ai/0.5/tutorials/running-inference-with-hygon-dcus/)
- [PaddlePaddle DCU Support — PaddlePaddle Docs](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/dcu/install_cn.html)
- [Hygon Information Technology — Wikipedia](https://en.wikipedia.org/wiki/Hygon_Information_Technology)
- [AMD–Chinese Joint Venture — Wikipedia](https://en.wikipedia.org/wiki/AMD%E2%80%93Chinese_joint_venture)
- [Optimizing Depthwise Separable Conv on DCU — Springer 2024](https://link.springer.com/article/10.1007/s42514-024-00200-3)
- [Optimizing Sparse GEMM for DCUs — Journal of Supercomputing 2024](https://link.springer.com/article/10.1007/s11227-024-06234-2)
- [Hygon DCU Dual-Chip Roadmap — Digitimes Dec 2025](https://www.digitimes.com/news/a20251223VL208/ai-chip-china-cpu.html)
- [中科海光 CPU+DCU 产品介绍 — 知乎](https://zhuanlan.zhihu.com/p/693079965)

### Added 2026-08-08 (深算三号 / merger termination)

- [海光信息 accelerator product page (publishes no DCU model names or specs)](https://www.hygon.cn/product/accelerator)
- [中科南京信息高铁研究院 — 深算三号 BW1000 国内首发 on 信息高铁 AI platform](https://www.ictnj.ac.cn/newsinfo/10874852.html)
- [南京麒麟科创园新闻 — 深算三号 首发部署 (2025-12-22)](https://qilinpark.nanjing.gov.cn/xwzx/202512/t20251222_5748374.html)
- [海光信息终止吸收合并中科曙光 — 财新 (2025-12-09)](https://www.caixin.com/2025-12-09/102391669.html)
- [海光信息终止换股吸收合并中科曙光 — 第一财经](https://www.yicai.com/news/102948899.html)
- [海光/曙光合并终止 — 界面新闻](https://www.jiemian.com/article/13741594.html)
- [终止重组公告解读 — 东方财富 (2025-12-09)](https://finance.eastmoney.com/a/202512093586757039.html)
- [终止重组后续 — 东方财富 (2025-12-10)](https://finance.eastmoney.com/a/202512103587902138.html)
- [海光信息终止合并曙光 — 观察者网 (2025-12-10)](https://www.guancha.cn/economy/2025_12_10_799933.shtml)
- [海光信息终止吸收合并 — 同花顺 (2025-12-09)](https://news.10jqka.com.cn/20251209/c673085520.shtml)
- [海光信息投资者互动回复：深算三号已投入市场、深算四号研发进展顺利 — 新浪财经 (2025-12-11)](https://finance.sina.com.cn/jjxw/2025-12-11/doc-inhakwak9142063.shtml)
- [深算三号 — 百度百科 (Hunyuan Hy3 adaptation claim, encyclopedia-grade only)](https://baike.baidu.com/item/%E6%B7%B1%E7%AE%97%E4%B8%89%E5%8F%B7/67723890)
