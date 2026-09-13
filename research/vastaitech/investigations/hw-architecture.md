# VastaiTech (瀚博半导体) Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: vastaitech*
*device_class: GPU-like Inference Accelerator + Video Codec (China, 瀚博 VastaiTech)*

---

## Scope and method

This investigation covers the **datacenter PCIe cards only**: VA1, VA10, VA1V, VA1L, VA12, VA16, VA10L, and the 南禺 / Nanyu VG-series datacenter graphics cards. VE1S / VE1M / VE1V / VS1000 are edge/embedded parts, out of scope except where their published numbers illuminate the shared silicon.

VastaiTech publishes **no datasheet, no architecture whitepaper, and no conference paper** for any part. There is no ISCA / MICRO / Hot Chips / ISSCC / arXiv publication. The developer centre that would hold the API and hardware reference manuals is sales-gated and verified inaccessible. Everything below therefore comes from four tiers of evidence, which must be kept distinct when citing:

| Tier | What it is | How to cite |
|---|---|---|
| **A — the vendor's own shipped code** | `github.com/Vastai` (VastStreamX-Samples, VastModelZOO, xinference_vacc, MinerU fork): SDK telemetry structures, compiler configs, sample benchmark tables, env vars | Strongest available. Structural facts (dies per card, memory per die, engine classes, clock domains) are effectively confirmed. Not a datasheet — do not back-derive peak throughput from it |
| **B — vendor product/marketing pages** | `vastaitech.com` product, software and about pages | Good for form factor, channel counts, TDP and the milestone timeline. **Every performance ratio on these pages is unbaselined marketing** |
| **C — Chinese launch press** | 腾讯新闻, 界面新闻, 36Kr, 量子位, 芯东西 | The only source of TOPS/TFLOPS figures for the 2021–2023 parts, and the only architectural block description that exists. Numbers here originate from vendor launch decks |
| **D — regulatory / OEM filings** | CITIC Securities IPO counselling report; 通泰易 server certification | Primary documents, but corporate/compatibility facts only — no specs |

**Confidence key used inline:** `[C]` confirmed by a retrieved primary or well-sourced secondary document; `[V]` vendor marketing claim (unbaselined); `[I]` inferred from artifacts, labelled as such; `[ND]` not disclosed.

> **Research trap.** `vastaitech.com` is stale — its timeline and awards both stop in 2023, and the Nanyu nav entry is a dead placeholder. The current 2026 line (**VA16 / VA10L / VA1L**) appears nowhere on the marketing site. It exists only in GitHub, the vLLM recipe site, and 2025–2026 Chinese press.

---

## 1. Company and silicon generations

VastaiTech / 瀚博半导体（上海）股份有限公司, founded **December 2018**, HQ Shanghai (Minhang), with R&D also in Beijing, Shenzhen, Xi'an and Chengdu. 500+ staff, >80% R&D, >70% holding a master's degree or above `[C, vendor About page]`. Founders **钱军 (Jun Qian, CEO)** and **张磊 (Lei Zhang, CTO)** are both ex-AMD — Qian a Senior Director over GPU/AI server chip design, Zhang an AMD Fellow leading AI and video-codec development `[C, jiemian, zhidx]`.

The vendor describes **two chip generations in mass production** and three product lines: graphics-rendering GPU, datacenter GPU, edge GPU `[C, About page]`.

| Chip family | Codename | Class | Process | Tape-out → silicon → mass production |
|---|---|---|---|---|
| **SV100 series** (VA1 SKU: **SV102**) | 达景 "Dajing" | Server-class AI-inference **DSA**, deliberately *not* a graphics GPU | **7 nm** `[C]` | Tape-out 2021-03; silicon back 2021-06 (lit in 8 minutes); MP 2022-Q1 `[C, vendor timeline]` |
| **SG100 series** | 乾元 "Qianyuan" | **Full-function GPU** (graphics + AI + video) | **7 nm** `[C]` | Tape-out 2022-10; silicon back 2023-02 (ran 王者荣耀 and 原神 within 24 h); MP 2023-04 `[C, vendor timeline]` |
| 3rd generation | `[ND]` | Reported in development in IPO-related coverage | `[ND]` | `[ND]` |

- **Foundry: `[ND]`** for both 7 nm parts. No retrieved source names it. **Do not assert TSMC.**
- **Die size, transistor count, package type: `[ND]`** for every part.
- SV100's positioning is explicitly DSA: *"公司并没有首选做GPU，而是选择通过DSA架构来做面向AI+视频市场的芯片，从而在PPA和成本上具有明显市场优势"* `[C, zhidx interview]`. SG100 is the GPU generation.

---

## 2. VUCA — the on-chip compute organisation

**VUCA** = *Vast(ai) Unified Compute Architecture*, announced September 2022. The vendor is internally inconsistent about the expansion — "Vast Unified Computing Architecture" on product pages, "Vastai Unified Compute Architecture" on the About page. Note the discrepancy rather than silently picking one.

Marketing framing on every product page: three acceleration engines on one chip — **streaming-media, inference, and graphics** `[V/C, vendor product pages]`.

The most technically specific public description of the on-die block structure is the 芯东西 (zhidx) CEO interview of 2022-09-08 `[C]`:

- a **high-performance compute engine** (高性能计算引擎)
- a **high-performance AI engine** (高性能AI引擎) — the matmul/conv datapath
- a **programmable vector compute engine** (可编程向量计算引擎) — this is the **VDSP**
- **dedicated video decode and graphics render/display cores** (专用视频解码与图形渲染显示核)
- **unified memory management with coherent interfaces and low-latency interconnect** (统一内存管理、一致性接口和低延迟互联)

This is corroborated independently by the SDK's own telemetry structure. `vsx::DieUtilization` reports utilisation for exactly four engine classes per die `[C, VastStreamX-Samples/samples/card_info]`:

```
util.ai        // AI engine   — scalar field: one AI domain per die
util.vdsp[]    // array       → multiple VDSP vector cores per die
util.vdmcu[]   // array       → multiple video-DECODE MCUs per die
util.vemcu[]   // array       → multiple video-ENCODE MCUs per die
```

The **counts** of VDSP cores, decode MCUs and encode MCUs per die are `[ND]` — the arrays are sized by an SDK header that is not public.

**Internal structure of the AI engine is entirely `[ND]`**: MAC-array dimensions, systolic versus SIMD organisation, MACs per cycle, and native tile size are never described anywhere.

### Clock domains

VastStreamX benchmark tables are quoted at `OCLK=835 MHz, DCLK=650 MHz, ECLK=200 MHz`, and one ResNet sample at `880 MHz` `[C, VastStreamX-Samples READMEs]`. There are therefore at least three independent clock domains — operator/AI (**OCLK**), decode (**DCLK**), encode (**ECLK**) — and the AI clock varies across SKUs / DPM states. Dynamic power management is user-controllable: `sudo vasmi setconfig dpm=enable -d all` `[C, VastModelZOO README]`.

---

## 3. A card is N dies. This is the key structural fact.

Every VastaiTech card presents **N independent devices ("dies")** to software, each with its own memory pool. `vsx::Card::GetAllDies()` enumerates them and each die carries its own `device_id` `[C, card_info sample]`. The sample's own reference output:

```
Find 2 cards in system.
0th card info:  UUID: FCA12CD00038   Card type: VA1-16G   Die ID: 0, 1,
1th card info:  UUID: FCA12CD00053   Card type: VA1-16G   Die ID: 2, 3,
Device id 0 status:  Memory total: 8 GB.  ...
```

So **VA1-16G = 2 dies × 8 GB**. The 2026 LLM cards are 4-die: the vLLM recipe site states hardware as **`VA16128G (4×32G)`** and **`VA10L128G (4×32G)`** `[C, vllm-vacc.vastaitech.com/Qwen/Qwen3-32B, /Qwen/Qwen3-Embedding]`. This is why DeepSeek-V3 on "8× VA16" runs at **TP=32** — 8 cards × 4 dies = 32 tensor-parallel ranks `[C]`.

**Whether a "die" is a physically separate silicon die in a multi-chip package or a partition of one monolithic die is `[ND]`.** The vendor never says, and no teardown exists.

Everything the runtime exposes — `VACC_VISIBLE_DEVICES`, `--llm_devices`, `--tensor-parallel-size`, per-device memory totals — operates at **die** granularity, not card granularity. Any per-card figure quoted from a VastaiTech source must be checked against the die count before it is compared with another vendor's per-chip number.

---

## 4. Card lineup, published specs

### 4.1 SV100 generation datacenter cards (2021–2022)

| Card | Compute | Power | Form factor | Memory | Codec |
|---|---|---|---|---|---|
| **载天 VA1** (SV102) | >200 TOPS INT8 per chip; FP16 / BF16 / INT8 `[C]` | **70 W** (vendor page) / **75 W** (launch coverage) — the two figures conflict; report both | single-width **HHHL**, PCIe **4.0 ×16**, no auxiliary power `[C]` | **32 GB** at launch `[C, news.qq.com 2021-07-08]`; SDK sample shows a **VA1-16G** SKU = 2 × 8 GB `[C]` | 120 ch 1080p30 H.264/H.265 decode (vendor page); 64+ ch H.264/H.265/**AVS2** at launch; up to 8K; JPEG hardware codec `[C]` |
| **载天 VA10** | **400 TOPS INT8** `[C, 36kr 2022-09-05]` | **150 W** `[C]` | full-height **3/4-length** PCIe `[C]` | `[ND]` | 240 ch 1080p30 decode (vendor page); 100 ch 1080p30 *transcode* (launch) `[C]` |
| **载天 VA1V** | video-centric; AI rating `[ND]` | `[ND]` | PCIe | `[ND]` | H.264/H.265/**AV1**, up to **8K 10-bit**; **2× 8K HDR @60+fps** encode/transcode; FFmpeg standard media interface `[C]` |
| *(edge, cross-reference only)* VE1 | 100 TOPS INT8 | 40–65 W | — | `[ND]` | 60 ch 1080p decode `[C]` |

### 4.2 SG100 generation, announced at WAIC 2023-07-06 `[C, qbitai, 36kr]`

- **SG100 chip**: 7 nm; unified rendering + AI + video; **DirectX 11, OpenGL, Vulkan**; **H.264 / H.265 / AV1**; **SR-IOV hardware virtualisation**; Windows and Linux.
- **南禺 (Nanyu) datacenter graphics cards**: **VG1600** (cloud gaming), **VG1800** (cloud desktop / remote work), **VG14** (workstation). **Per-card specs — memory, TDP, display outputs, virtualisation density — are all `[ND]`**; the vendor site's Nanyu nav entry is a dead placeholder.
- **载天 VA1L** (LLM card): **200 TOPS INT8 / 72 TFLOPS FP16** `[C]`. The 2023 AIGC appliance used **8 × VA1L = 512 GB**, i.e. **64 GB/card**, "supporting 175 B-parameter models" `[C/V]`.
- **VA12** (high-performance generative-AI card): **250 W**, **512 TOPS INT8 / 160 TFLOPS FP16** `[C]`.

### 4.3 Current 2026 datacenter line — VA16 / VA10L / VA1L

These appear **nowhere on vastaitech.com**. They are documented only in GitHub, the vLLM recipe site, and 2025–2026 Chinese press.

- **载天 VA16** — the flagship. **128 GB per card, exposed as 4 × 32 GB dies** `[C]`. A server-OEM certification notice describes it as a "**训推一体**" (training-and-inference) accelerator card `[C, ttyinfo.com, 2025-06-13]`. **FP4 and FP8** support as of 2026 `[C, 上海证券报 via Sina Finance and 10jqka, 2026-04-27]`. Appliance configurations: **8-card = 1 TB**, **16-card = 2 TB** total memory `[C/V]`; the 2 TB VGX VA16 appliance is claimed to support **405 B fine-tuning** and trillion-parameter MoE inference `[V]`.
- **载天 VA10L** — `VA10L128G (4×32G)` `[C]`.
- **载天 VA1L** — listed as a supported card throughout the 2026 stack; **current per-card memory `[ND]`** (the 2023 figure was 64 GB, which may or may not still hold).
- **TOPS / TFLOPS, power, form factor, PCIe generation, and process node for VA16 / VA10L / VA1L are all `[ND]`.** No source publishes any of them. Which chip family they are built on (SV100, SG100, or an undisclosed third generation) is also `[ND]`.

---

## 5. Memory hierarchy

| Level | What is known |
|---|---|
| Register file | `[ND]` — size and organisation never described |
| On-chip SRAM / scratchpad | Capacity and bandwidth `[ND]`. **Strong evidence it is software-managed rather than hardware-cached** `[I]` — see below |
| Per-VDSP local buffer | `[ND]`, but bounded: the shipped `planar_argmax` custom op documents *"currently channel number is limited to less than or equal to 96"* for an fp16 planar tensor — a per-op local-buffer constraint `[C]` |
| Tiling granularity | LLM compilation requires **sequence length to be a multiple of 16** `[C, VastModelZOO/llm/README.md]` |
| Off-chip (device) memory | **Capacity known per die/card**: 8 GB/die on VA1-16G; 32 GB/die and 128 GB/card on VA16 and VA10L; 32 GB on the launch VA1; 64 GB on the 2023 VA1L. **Memory technology (HBM / GDDR / LPDDR / DDR) and memory bandwidth in GB/s are `[ND]` for every VastaiTech part ever shipped** |
| Runtime memory model | Explicit host↔device copies only; **no unified/managed-memory abstraction is exposed** `[C]` |

### 5.1 Why the on-chip memory is believed to be a software-managed scratchpad `[I]`

VAMC exposes per-model compile flags that **explicitly place tensors**:

- `output_ddr: [-1, 1]` — which graph outputs are forced to off-chip DDR
- `mem_inplace: true|false`
- `data_transport_mode: 1|3`
- `cluster_mode: 0|1`
- `output_layout`, `enable_graph_partition`
- `graph.extra_ops: { type: insert_odma }` — inserts explicit **output-DMA** nodes

A hardware-cached memory would not need a compiler flag naming DDR as a destination, nor a pass that inserts explicit output-DMA nodes. `[C for the flags; I for the conclusion.]` **No vendor statement confirms the absence of a hardware data cache in the compute path**, so this stays inferred, not asserted.

### 5.2 Runtime memory API `[C, VastStreamX samples]`

`vsx.from_numpy(x, device_id)` / `vsx.as_numpy(t)`; `Context::CPU()` versus `Context::VACC(device_id)`; `Tensor::Clone(Context::VACC(id))`; `GetDataAddress()` returning a raw device address that is packed into custom-op config structs. Movement between host and device is always an explicit user action.

### 5.3 The bandwidth hole

This is the most important gap in the whole entry. **No memory-bandwidth figure exists for any VastaiTech part in any language.** Searching in Chinese for 显存 / 带宽 / GB/s against every card name returns nothing; there is no datasheet, no teardown, no reviewer measurement. Since VA16 is a 128 GB capacity-first card serving 671 B MoE models, bandwidth is precisely the number that would determine its decode performance — and it is the number the vendor does not print. **Write "not disclosed". Do not estimate from capacity, from die count, or from any peer part.**

---

## 6. Interconnect

- **On-chip NoC** — described only qualitatively: "unified memory management, coherent interfaces and low-latency interconnect" `[C, zhidx]`. Topology and bandwidth `[ND]`.
- **Die-to-die within a card** — exists by construction (2- and 4-die cards), but the fabric, protocol and bandwidth are `[ND]`. **No name analogous to NVLink / xGMI / SG-Link appears anywhere in any source.**
- **Card-to-card scale-up** — `[ND]`. A single VA16 server runs DeepSeek-V3 at **TP=32 across 8 cards**, so cross-card collectives demonstrably work at tensor-parallel granularity, but no fabric is ever named. **There is no evidence of a proprietary switch or cabled fabric.** PCIe is the only host interface documented.
- **Scale-out** — `--pipeline-parallel-size 2` is used for the 100 K-input DeepSeek-V3 configuration `[C]`. **Data parallelism is explicitly unsupported** on the vLLM backend: the MinerU support matrix prints "数据并行 (`--data-parallel-size`/`--dp`) 🔴" `[C]`. Whether VCCL has an inter-node RDMA transport at all is `[ND]`.
- **Collective library — VCCL** (a NCCL analogue). Visible only through environment variables `VCCL_SOCKET_IFNAME` (default `"lo"`), `VCCL_MODEL_SYNC`, and the compiler flag `gather_data_vccl_dsp_enable` — the last indicating that collectives can be **executed on the VDSP engines** `[C]`. `VCCL_SOCKET_IFNAME` mirroring `NCCL_SOCKET_IFNAME` implies a socket-based bootstrap or out-of-band path. Supported collective operations, algorithms, achieved bus bandwidth and API are all `[ND]` — there is no public documentation of VCCL of any kind.
- **Host interface** — **PCIe 4.0 ×16** for VA1 `[C]`. PCIe generation and lane count for VA10 / VA16 / VA10L / VA1L are `[ND]`. The PCI device ID appears to be **0x0100**: the install guide detects cards with `lspci -d:0100` `[C]`.

---

## 7. Execution model

- **Ahead-of-time, statically compiled graphs (Build_In path).** A whole model compiles to a fixed artifact (`deploy_weights/<name>/mod`) with **tensor-parallel degree, batch size and I/O shapes baked in at compile time** (`tp: 4`, `input_ids: [[512],[1024]]`). There is no JIT.
- **Stream/graph dataflow at runtime.** VastStreamX builds a `vsx::Graph` of `Operator`s — VDSP preprocessing fusion ops plus a `ModelOperator` — wraps it in a `vsx::Stream` with a `StreamBalanceMode` (`kBM_RUN`), and calls `stream->Build()`. Execution is `RunSync`, or fully asynchronous via `process_async` → `get_output` → `close_input` → `wait_until_done` `[C, common/model_base.hpp, samples/run_stream_async]`.
- **Multiple concurrent model instances per die** are the documented route to peak throughput (`--instance 4`, `queue_size`). The die is oversubscribed by independent streams rather than by one large kernel `[C, profiler samples]`.
- **Heterogeneous op placement is explicit and manual.** Preprocessing runs on VDSP via a JSON-selected fusion op (e.g. `FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR` with mean/std/resize/colour-convert parameters). Some ops must run on the host CPU (Qwen-VL `Smart_Resize`, RetinaNet post-processing). Some structural ops are hand-mapped to VDSP (PointPillarScatter; Qwen-VL visual rotary embedding) `[C, VastModelZOO deploy docs]`.
- **No graph capture on the vLLM path.** `--enforce-eager` is present in *every* documented vLLM command, and the MinerU integration states the rule outright: "注意在执行任意与vllm相关命令需追加 `--enforce_eager` 参数" `[C]`. There is no CUDA-Graph equivalent.

---

## 8. Measured performance from the vendor's own sample docs (per die)

These are the only reproducible absolute numbers published anywhere for this hardware, and they are **per die, not per card** `[C, VastStreamX-Samples]`:

| Workload | Result |
|---|---|
| H.264 1080p decode, DCLK = 650 MHz | **1547 fps** at max throughput (10 instances); 509 fps at min latency (1 instance) → ≈ 51 channels @ 30 fps per die |
| H.265 1080p decode, DCLK = 650 MHz | **1861 fps** max throughput → ≈ 62 ch @ 30 fps per die |
| ResNet-50 INT8, 880 MHz | **3231 qps** (batch 8, max throughput); 941 qps (batch 1, min latency) |
| MobileViT | 43.9 qps (batch 1) |
| Custom `planar_argmax`, [19,512,512] fp16, 4 instances | 3008 qps, p50 1329 µs |

**These cross-check the VA1 spec sheet and independently confirm the 2-die structure** `[I]`: ≈ 51–62 ch/die × 2 dies ≈ 102–124 channels, against the vendor's claimed 120 ch for VA1.

---

## 9. Vendor performance claims — all unbaselined

Every headline claim on vastaitech.com is **relative and unbaselined**: "同等功耗下2倍以上于主流GPU的最高吞吐率", "延时不到GPU最高吞吐率下延时的5%", VA10's "推理性能达到同功耗主流GPU的2倍以上，延时低至6%", VA1's "2–10× AI throughput of GPUs at equal power", DSA "3–5× traditional GPUs". **No named comparison part, no workload, no date.** Label all of these `[V]`.

### 9.1 Counter-evidence the vendor published itself

In `VastModelZOO/vlm/glm_ocr/vllm/README.md`, the OmniDocBench end-to-end GLM-OCR table compares backends on the same dataset with the layout model on CPU in all rows `[C]`:

| Backend | Overall ↑ | Model Infer Cost |
|---|---|---|
| NVIDIA H800 BF16 TP1 | 95.391 | **15.5 min** |
| VACC-VA16 BF16 TP1 | 95.647 | **13 h 26 min** |
| VACC-VA16 BF16 TP2 | 95.611 | 9 h |
| VACC-VA16 BF16 TP4 | 95.635 | 6 h 33 min |

Accuracy is on par with — marginally above — H800. Wall clock is roughly **50× slower at TP1 and ~25× slower at TP4**.

Caveat it properly: one workload, an OCR VLM, full-dataset wall clock, layout model on CPU in every row, and VA16 TP1 uses one die against a whole H800. But it is the vendor's own apples-to-apples table and it is the most useful maturity signal available for this vendor. It should be cited with the caveats, not suppressed and not sharpened.

---

## 10. Physical / platform

- **Process**: 7 nm for SV100 and SG100 `[C]`. **Everything else `[ND]`**, including the node for whatever silicon is in VA16 / VA10L / VA1L.
- **TDP**: VA1 70 W / 75 W (conflicting sources), VA10 150 W, VA12 250 W, VE1 40–65 W. **VA16 / VA10L / VA1L `[ND]`.**
- **Die size, package, cooling, slot width and auxiliary power for the current cards**: `[ND]`.
- **Host platforms**: x86_64 and **aarch64** wheels both shipped `[C]`. Validated on Ubuntu 22.04.3 with a **Hygon C86-4G** CPU `[C, Vastai/MinerU README]`. Domestic OS support claimed for Kylin (麒麟), UOS (统信), Anolis (龙蜥) and OpenEuler `[V]`.
- **Server OEM certification**: 通泰易 TG657V2 / TG658V3 / TG659V2 certified with VA16, 2025-06-13 `[C]`.

---

## 11. Maturity

**Verdict: shipping / commercially available — NOT verified deployed at scale.**

Positive signals:
- Two chip generations vendor-stated "量产并商业化落地" `[V]`
- A published 3-year hardware + software warranty with a formal RMA process `[C]`
- A sales-gated developer centre and a private Docker harbor with versioned releases (VVI-25.12.SP2, VVI-26.02) `[C]`
- Server-OEM compatibility certifications `[C]`
- A 海马云 (Haima Cloud) strategic partnership announced 2024-04-10 targeting a **thousand-card** cloud-rendering/AI cluster on domestic ARM CPUs + VastaiTech GPUs `[C]`
- An A-share IPO counselling filing `[C, primary]`

Negative / limiting signals:
- **Max-concurrency capped at 4** across essentially every vLLM model, including DeepSeek-V3 and small BGE embedding models `[C, tools/vllm/usage_limits.md]`
- No data parallelism; mandatory eager mode
- The OmniDocBench wall-clock gap in §9.1
- **No named end customer with a disclosed deployment size.** Kuaishou is reported by 界面新闻 in 2021 as both investor and customer, with no scale disclosed
- **No third-party benchmark of any kind**
- **No ISCA / MICRO / Hot Chips / ISSCC / arXiv publication of any kind**

---

## 12. Corporate / IPO (primary source obtained)

CITIC Securities' first IPO-counselling progress report — a **primary filing document** — confirms `[C]`:

- Counselling agreement with 瀚博半导体（上海）股份有限公司 signed **2025-07-11**; filing submitted the same day to the Shanghai CSRC bureau; formally entered the counselling period **2025-07-18**. First reporting period 2025-07-18 → 2025-09-30.
- Target: **domestic A-share IPO**. Sponsor **中信证券 (CITIC Securities)**; counsel 北京市中伦律师事务所; auditor 天健会计师事务所.
- Institutional shareholders above 5%: **VASTAI Holding Company 11.53%** (LEI ZHANG), **ACE REDPOINT CHINA VASTAI HK LIMITED 7.12%**, **ZHEN PARTNERS V (HK) LIMITED 6.48%**, **5Y CAPITAL VASTAI HOLDING LIMITED 5.28%**.

This **reconciles with, and does not contradict, the negative result** from the SSE STAR Market review database: counselling registration (辅导备案) is the pre-application stage, so the absence of any 瀚博 record among the 1,046 STAR Market filings is exactly what one would expect. **Do not write that VastaiTech has filed an IPO application** — write that it entered IPO counselling in July 2025.

Funding history from the vendor's own timeline `[C]`: Series A **US$50 M** (2020-11), Series A+ **¥500 M** (2021-04), Series B1&B2 **¥1.6 B** (2021-12). Reported cumulative raise >¥2.5 B and a ~¥10.5 B valuation (2025) are **media figures**, not vendor-published. Registered capital ¥543 M (secondary). Investors reported to include Alibaba, Kuaishou, Sequoia, 5Y Capital, Zhen Partners and Redpoint China.

---

## 13. Open items — not disclosed anywhere

Recorded explicitly so that no downstream reader mistakes silence for an oversight:

1. Memory **technology** (HBM / GDDR / LPDDR / DDR) — every part, never published.
2. Memory **bandwidth** in GB/s — every part, never published.
3. On-chip SRAM / scratchpad capacity and bandwidth; register-file size.
4. Counts of VDSP cores, `vdmcu` decode MCUs and `vemcu` encode MCUs per die.
5. AI-engine internals: MAC array dimensions, systolic vs SIMD, MACs/cycle, native tile size.
6. Peak TOPS / TFLOPS for VA16, VA10L and the current VA1L; FP8 and FP4 peak rates on VA16 (only format support is published, never a rate).
7. TDP, form factor, slot width, cooling, auxiliary power and PCIe generation for VA16 / VA10L / VA1L.
8. Process node and chip family for VA16 / VA10L / VA1L; foundry for SV100 and SG100.
9. Die size, transistor count, package type for every chip.
10. Whether an SDK "die" is a separate silicon die in an MCM or a partition of a monolithic die.
11. Die-to-die interconnect protocol, topology and bandwidth; whether any card-to-card fabric exists beyond PCIe.
12. Scale-out: NIC, RDMA support, and whether VCCL has an inter-node transport beyond the socket path hinted at by `VCCL_SOCKET_IFNAME`.
13. VCCL collective operations, algorithms, bus bandwidth and API.
14. Semantics of the VAMC flags `cluster_mode`, `data_transport_mode`, `data_type`, `opt_level`, `output_layout`, `stream_mode`, `split_convergence_points`, `requant_suppress` — values are visible in 344 shipped configs; meanings are documented only in the gated developer centre.
15. Whether on-chip memory is hardware-cached or a software-managed scratchpad (strongly implied to be the latter; never stated).
16. Current per-card memory capacity of VA1L.
17. Nanyu VG1600 / VG1800 / VG14 specifications of any kind.
18. Named end customers with disclosed deployment scale; unit shipments; revenue; manufacturing capacity.
19. Any independent third-party benchmark; any peer-reviewed or conference publication.

---

## Sources

Vendor primary:
- [About / company history and milestone timeline](https://www.vastaitech.com/company/about)
- [载天 VA1 product page](https://www.vastaitech.com/product/general/va1)
- [载天 VA10 product page](https://www.vastaitech.com/product/general/va10)
- [载天 VA1V video card](https://www.vastaitech.com/product/video/va1v)
- [载天 VE1V video card](https://www.vastaitech.com/product/video/ve1v)
- [智能一体机 solution page](https://www.vastaitech.com/solution/all-in-one)
- [Support — 3-year warranty and RMA](https://www.vastaitech.com/support)
- [vLLM × VastAI recipe site (VA16 / VA10L / VA1L)](https://vllm-vacc.vastaitech.com/)
- [vLLM recipe — DeepSeek-V3 (8× VA16, TP32, FP8)](https://vllm-vacc.vastaitech.com/deepseek-ai/DeepSeek-V3)
- [vLLM recipe — Qwen3-32B ("VA16128G (4×32G) / VA10L128G (4×32G)")](https://vllm-vacc.vastaitech.com/Qwen/Qwen3-32B)
- [vLLM recipe — Qwen3-Embedding](https://vllm-vacc.vastaitech.com/Qwen/Qwen3-Embedding)

Vendor's own code (Tier A):
- [VastStreamX-Samples — card_info sample (VA1-16G, 2 dies × 8 GB, util{ai,vdsp[],vdmcu[],vemcu[]})](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/card_info/card_info.cpp)
- [VastStreamX-Samples — video_decode README (per-die 1547 / 1861 fps @ DCLK 650 MHz)](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/video_decode/README.md)
- [VastStreamX-Samples — planar_argmax custom op (≤96 channels)](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/vdsp_op/custom_op/argmax/argmax_op.hpp)
- [VastModelZOO — VAMC compiler config (output_ddr, insert_odma, tvm_vacc, gather_data_vccl_dsp_enable)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_config.yaml)
- [VastModelZOO — LLM compile guidance (seq-len multiple of 16, VACC_STACK_SIZE)](https://github.com/Vastai/VastModelZOO/blob/main/llm/README.md)
- [VastModelZOO — DeepSeek-V3 deployment (single VA16 server, TP32, TP32-PP2)](https://github.com/Vastai/VastModelZOO/blob/main/llm/deepseek_v3/README.md)
- [VastModelZOO — vLLM usage limits (max-concurrency 4)](https://github.com/Vastai/VastModelZOO/blob/main/tools/vllm/usage_limits.md)
- [VastModelZOO — GLM-OCR OmniDocBench: H800 vs VACC-VA16](https://github.com/Vastai/VastModelZOO/blob/main/vlm/glm_ocr/vllm/README.md)
- [Vastai/MinerU — tested platform, DP unsupported, --enforce_eager mandatory](https://github.com/Vastai/MinerU)

Press (Tier C):
- [腾讯新闻 2021-07-08 — SV100 + VA1 launch](https://news.qq.com/rain/a/20210708A032OI00)
- [界面新闻 — SV100 is a DSA; VA1 = SV102; founders' AMD background](https://m.jiemian.com/article/6342672.html)
- [芯东西 zhidx 2022-09-08 — CEO interview, VUCA block list](https://zhidx.com/p/344936.html)
- [36Kr 2022-09-05 — VA10 400 TOPS / 150 W](https://www.36kr.com/p/1901732567984512)
- [量子位 2023-07 — WAIC 2023: SG100, Nanyu VG-series, VA1L, VA12](https://www.qbitai.com/2023/07/66614.html)
- [36Kr 2023-07-06 — independent confirmation of the WAIC 2023 figures](https://www.36kr.com/p/2332635957528066)
- [eeNews Europe — China's Vastai launches 7nm GPU](https://www.eenewseurope.com/en/chinas-vastai-launches-7nm-gpu-for-ai-visual-apps/)
- [上海证券报 via Sina Finance 2026-04-27 — VA16 128 GB, FP4+FP8, up to 2 TB appliance](https://finance.sina.com.cn/roll/2026-04-27/doc-inhvxtrc2314774.shtml)
- [同花顺 10jqka 2026-04-27 — same story, verbatim Chinese quotes](https://news.10jqka.com.cn/20260427/c676308817.shtml)
- [iCloudNews 2024-04-10 — 海马云 partnership, thousand-card cluster](https://www.icloudnews.net/a/79464.html)

Regulatory / OEM (Tier D):
- [CITIC Securities IPO counselling progress report (PDF, primary filing)](https://www.cs.ecitic.com/newsite/tzgg/ipoqyfdgg/202510/P020251023518196738538.pdf)
- [SSE STAR Market IPO review database — negative result, 1,046 filings, no 瀚博](https://query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB)
- [通泰易 2025-06-13 — TG657V2 / TG658V3 / TG659V2 certified with VA16](http://ttyinfo.com/News/info/id/154.html)
