# VastaiTech (瀚博半导体) Hardware Architecture

*as_of: 2026-08-08*
*chip: vastaitech*
*device_class: GPU-like Inference Accelerator + Video Codec (China, 瀚博 VastaiTech)*
*Representative products: 载天 VA16 (2026 flagship, 128 GB = 4 × 32 GB dies), VA10L, VA1L, VA12; 载天 VA1 (SV102) and VA10 (SV100 generation); 南禺 VG1600 / VG1800 / VG14 (SG100 graphics)*

---

## Overview

VastaiTech builds a **unified multi-engine die**: an AI matmul/conv engine, a programmable vector DSP (VDSP), and hard video encode/decode MCUs sharing one memory space, under an architecture umbrella the vendor calls **VUCA**. Two silicon generations are in mass production, both on a **7 nm** process whose **foundry is never named**:

1. **SV100** (达景 "Dajing"; VA1's SKU is **SV102**) — a server-class AI-inference **DSA**, an explicit choice *against* building a GPGPU. Tape-out 2021-03, silicon 2021-06, mass production 2022-Q1.
2. **SG100** (乾元 "Qianyuan") — a **full-function GPU**: graphics (DirectX 11, OpenGL, Vulkan), AI, video (H.264/H.265/AV1), and SR-IOV hardware virtualisation on one die. Tape-out 2022-10, silicon 2023-02 (it ran 王者荣耀 and 原神 within 24 hours), mass production 2023-04.

A third generation is reported to be in development; nothing about it is disclosed.

The two facts that most change how this hardware should be read:

- **A card is N independent dies**, each with its own memory pool and device ID. Software addresses dies, not cards. VA1-16G = 2 dies × 8 GB; VA16 and VA10L = 4 dies × 32 GB. This is why "8× VA16" runs DeepSeek-V3 at **TP=32**.
- **Almost nothing quantitative is disclosed for the current generation.** No memory type, no memory bandwidth, no peak throughput, no TDP, no PCIe generation, no process node for VA16 / VA10L / VA1L. This is not an omission in this write-up; it is the state of the public record.

There is **no VastaiTech datasheet, architecture whitepaper, or conference paper of any kind** — no ISCA, MICRO, Hot Chips, ISSCC, or arXiv publication. The developer centre that holds the reference manuals is sales-gated and verified inaccessible.

---

## 0. Generation Overview

| Generation | Chip | Class | Process | Foundry | Representative datacenter cards | Peak (per chip, where published) |
|---|---|---|---|---|---|---|
| Gen 1 | **SV100** / SV102 | AI-inference **DSA** (not a GPGPU) | **7 nm** | **not disclosed** | 载天 VA1, 载天 VA10, VA1V | VA1 >200 TOPS INT8; VA10 400 TOPS INT8 |
| Gen 2 | **SG100** | **Full-function GPU** — graphics + AI + video, SR-IOV | **7 nm** | **not disclosed** | 载天 VA1L (2023), VA12, 南禺 VG1600 / VG1800 / VG14 | VA1L 200 TOPS INT8 / 72 TFLOPS FP16; VA12 512 TOPS INT8 / 160 TFLOPS FP16 |
| Current line (2026) | **not disclosed** — which family VA16 / VA10L / VA1L use is never stated | — | **not disclosed** | **not disclosed** | 载天 **VA16** (flagship), VA10L, VA1L | **not disclosed** |
| Gen 3 | — | Reported in development | not disclosed | — | — | — |

**Die size, transistor count and package type are not disclosed for any chip.** Do not infer a foundry from the 7 nm figure.

---

## 1. Compute Engine — VUCA

**VUCA** = *Vast(ai) Unified Compute Architecture*, announced September 2022. The vendor is internally inconsistent about the expansion ("Vast Unified Computing Architecture" on product pages, "Vastai Unified Compute Architecture" on the About page); the discrepancy is recorded rather than resolved.

### On-die blocks

The most technically specific public description is a 2022-09-08 CEO interview with 芯东西 (zhidx):

| Block | Chinese | Role |
|---|---|---|
| High-performance compute engine | 高性能计算引擎 | General compute |
| **High-performance AI engine** | 高性能AI引擎 | The matmul / convolution datapath |
| **Programmable vector compute engine** | 可编程向量计算引擎 | **This is the VDSP** |
| Video decode + graphics render/display cores | 专用视频解码与图形渲染显示核 | Hard codec and display pipeline |
| Unified memory management | 统一内存管理、一致性接口和低延迟互联 | Coherent interfaces, low-latency interconnect |

### Independent corroboration from the SDK

`vsx::DieUtilization` reports utilisation for exactly four engine classes per die — matching the interview's block list:

```
util.ai        // scalar → ONE AI domain per die
util.vdsp[]    // array  → multiple VDSP vector cores per die
util.vdmcu[]   // array  → multiple video-DECODE MCUs per die
util.vemcu[]   // array  → multiple video-ENCODE MCUs per die
```

| Parameter | Value |
|---|---|
| AI engine domains per die | **1** (scalar telemetry field) |
| VDSP cores per die | **not disclosed** (array sized by a non-public header) |
| Video-decode MCUs per die | **not disclosed** |
| Video-encode MCUs per die | **not disclosed** |
| AI-engine MAC array dimensions | **not disclosed** |
| Systolic vs SIMD organisation | **not disclosed** |
| MACs per cycle / native tile size | **not disclosed** |

### Clock domains

At least three independent domains, from VastStreamX benchmark headers:

| Domain | Symbol | Value in samples |
|---|---|---|
| Operator / AI | **OCLK** | 835 MHz (one ResNet sample at 880 MHz) — varies by SKU and DPM state |
| Video decode | **DCLK** | 650 MHz |
| Video encode | **ECLK** | 200 MHz |

Dynamic power management is user-controllable: `sudo vasmi setconfig dpm=enable -d all`.

---

## 2. The Die Model — a card is N devices

**This is the single most important structural fact about VastaiTech hardware.** Every card presents N independent devices ("dies") to software, each with its own memory pool and `device_id`. `vsx::Card::GetAllDies()` enumerates them.

The SDK's own `card_info` reference output:

```
Find 2 cards in system.
0th card info:  UUID: FCA12CD00038   Card type: VA1-16G   Die ID: 0, 1,
1th card info:  UUID: FCA12CD00053   Card type: VA1-16G   Die ID: 2, 3,
Device id 0 status:  Memory total: 8 GB.  ...
```

| Card | Dies | Memory per die | Memory per card |
|---|---|---|---|
| VA1-16G | **2** | 8 GB | 16 GB |
| **VA16** | **4** | 32 GB | **128 GB** |
| **VA10L** | **4** | 32 GB | **128 GB** |
| VA1L (2026) | not disclosed | not disclosed | not disclosed |

Consequences that must travel with any VastaiTech number:

- **TP degree is counted in dies, not cards.** DeepSeek-V3 on "8× VA16" runs at **TP=32** = 8 cards × 4 dies. Comparing that against another vendor's TP=8 across 8 chips is comparing different things.
- Every runtime knob — `VACC_VISIBLE_DEVICES`, `--llm_devices`, `--vit_devices`, `--tensor-parallel-size` — selects **dies**.
- VastGenX places VLM **vision and language towers on different dies**.
- Per-die measured performance must be multiplied by the die count before it is compared to a per-card figure, and vice versa.

**Whether a "die" is a physically separate silicon die in a multi-chip package or a partition of one monolithic die is not disclosed.** The vendor never says, and no teardown exists.

**Independent cross-check of the 2-die structure for VA1** (inferred): the SDK's per-die decode benchmark gives ≈51 ch (H.264) to ≈62 ch (H.265) of 1080p30 per die; × 2 dies = 102–124 channels, against VA1's vendor-claimed 120 ch.

---

## 3. Data Path

VastaiTech does not describe its data path anywhere. What the software surface establishes:

```
Host (x86_64 / aarch64)
    ↕  PCIe  (4.0 x16 confirmed on VA1 only; not disclosed elsewhere)
Device memory, per die  (8 GB / 32 GB — type and bandwidth NOT DISCLOSED)
    ↕  explicit DMA, placement controlled by the VAMC compiler
       (output_ddr, mem_inplace, data_transport_mode, insert_odma)
On-chip SRAM / scratchpad  (capacity and bandwidth NOT DISCLOSED;
                            inferred software-managed, not hardware-cached)
    ↕
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │  AI engine   │  VDSP × N    │  vdmcu × N   │  vemcu × N   │
  │ (matmul/conv)│ (vector, also│ (H.264/H.265 │ (H.264/H.265 │
  │              │  collectives)│  /AV1 decode)│  /AV1 encode)│
  └──────────────┴──────────────┴──────────────┴──────────────┘
             all under unified memory management
```

Two features of this path are unusual and worth stating explicitly:

1. **The codec MCUs are first-class, not an afterthought.** They have their own clock domains, their own telemetry arrays, and their own measured throughput tables. A VA1 sustains 120 channels of 1080p30 decode while also running inference. Very few accelerators in this survey integrate video at that scale.
2. **Collectives can run on the VDSP.** The VAMC flag `gather_data_vccl_dsp_enable` places VCCL gather operations on the vector engines rather than on a dedicated communication block — a distinctive placement choice, though nothing further about it is documented.

---

## 4. On-chip Memory

| Level | Chip | Managed by | Capacity |
|---|---|---|---|
| Register file | all | — | **not disclosed** |
| On-chip SRAM / scratchpad | all | **Compiler (inferred)** | **not disclosed** |
| Per-VDSP local buffer | all | Compiler / op | **not disclosed, but bounded** — the shipped `planar_argmax` op documents "channel number ≤ 96" for an fp16 planar tensor |

### Why the scratchpad is believed to be software-managed (inferred, not stated)

VAMC exposes per-model compile flags that **explicitly place tensors**:

- `output_ddr: [-1, 1]` — which graph outputs are forced to off-chip DDR
- `mem_inplace: true|false`
- `data_transport_mode: 1|3`
- `cluster_mode: 0|1`
- `output_layout`, `enable_graph_partition`
- `graph.extra_ops: { type: insert_odma }` — inserts explicit **output-DMA** nodes

A hardware-cached memory would not need a compiler flag naming DDR as a destination, nor a pass that inserts explicit output-DMA nodes. **The flags are confirmed; the conclusion is inferred.** No vendor statement confirms the absence of a hardware data cache in the compute path, and none should be manufactured.

A second placement signal: LLM compilation requires **sequence length to be a multiple of 16**, which is the only visible hint at a native tiling granularity.

---

## 5. Off-chip Memory

| Card | Capacity per die | Capacity per card | **Memory type** | **Bandwidth** |
|---|---|---|---|---|
| VA1 (launch) | — | 32 GB | **not disclosed** | **not disclosed** |
| VA1-16G (SDK SKU) | 8 GB | 16 GB | **not disclosed** | **not disclosed** |
| VA10 | not disclosed | not disclosed | **not disclosed** | **not disclosed** |
| VA1L (2023) | not disclosed | 64 GB (derived from "8 × VA1L = 512 GB") | **not disclosed** | **not disclosed** |
| VA12 | not disclosed | not disclosed | **not disclosed** | **not disclosed** |
| **VA16** | **32 GB** | **128 GB** | **not disclosed** | **not disclosed** |
| **VA10L** | **32 GB** | **128 GB** | **not disclosed** | **not disclosed** |
| VA1L (2026) | not disclosed | **not disclosed** | **not disclosed** | **not disclosed** |

> **The bandwidth hole is the most consequential gap in this entry.** No memory-technology and no memory-bandwidth figure exists for **any** VastaiTech part ever shipped, in Chinese or English. Searching in Chinese for 显存 / 带宽 / GB/s against every card name returns nothing; there is no datasheet, no teardown, no reviewer measurement.
>
> This matters because VA16 is a **capacity-first 128 GB card serving 671 B MoE models**, and memory bandwidth is precisely what determines decode throughput on that workload. It is the number the vendor does not print. **Write "not disclosed". Do not estimate it from capacity, from die count, or from any peer part.**

### Runtime memory model

Explicit host↔device copies only. `vsx.from_numpy(x, device_id)` / `vsx.as_numpy(t)`; `Context::CPU()` versus `Context::VACC(device_id)`; `Tensor::Clone(Context::VACC(id))`; `GetDataAddress()` returning a raw device address packed into custom-op config structs. **There is no unified or managed-memory abstraction exposed to the user.**

### Appliance-scale capacity

| Configuration | Total memory | Claim |
|---|---|---|
| 8 × VA1L (2023 AIGC appliance) | 512 GB | "supports 175 B-parameter models" — **vendor claim** |
| 8 × VA16 | **1 TB** | — |
| 16 × VA16 (VGX appliance) | **2 TB** | "405 B fine-tuning and trillion-parameter MoE inference" — **vendor claim** |

---

## 6. Interconnect

| Level | Status |
|---|---|
| **On-chip NoC** | Described only qualitatively: "unified memory management, coherent interfaces and low-latency interconnect". **Topology and bandwidth not disclosed** |
| **Die-to-die (within a card)** | **Exists by construction** — 2- and 4-die cards are enumerated by the SDK. **Fabric name, protocol, topology and bandwidth all not disclosed.** No name analogous to NVLink / xGMI / SG-Link appears anywhere in any source |
| **Card-to-card scale-up** | **Not disclosed.** TP=32 across 8 cards in one server demonstrably works, so cross-card collectives function at tensor-parallel granularity — but no fabric is ever named, and **there is no evidence of any proprietary switch or cabled fabric**. PCIe is the only documented host interface |
| **Scale-out** | `--pipeline-parallel-size 2` used for the 100 K-input DeepSeek-V3 configuration. **Data parallelism is explicitly unsupported** on the vLLM backend. NIC, RDMA support and any VCCL inter-node transport: **not disclosed** |
| **Host interface** | **PCIe 4.0 ×16** on VA1. **Not disclosed** for VA10 / VA16 / VA10L / VA1L. PCI device ID appears to be **0x0100** (`lspci -d:0100`) |

### Collectives — VCCL

**VCCL** is the NCCL analogue. It is visible **only** through three artifacts:

| Artifact | What it tells us |
|---|---|
| `VCCL_SOCKET_IFNAME` (default `"lo"`) | Mirrors `NCCL_SOCKET_IFNAME` → implies a socket-based bootstrap or out-of-band path |
| `VCCL_MODEL_SYNC` | A model-level synchronisation switch |
| `gather_data_vccl_dsp_enable` (VAMC flag) | **Collectives can be executed on the VDSP engines** |

Supported collective operations, algorithms, achieved bus bandwidth and API are all **not disclosed**. There is no public documentation of VCCL of any kind.

---

## 7. Execution Model

### Build_In path — ahead-of-time, statically compiled

A whole model compiles to a fixed artifact (`deploy_weights/<name>/mod`) with **tensor-parallel degree, batch size and I/O shapes baked in at compile time** (`tp: 4`, `input_ids: [[512],[1024]]`). There is no JIT. Changing TP degree means recompiling.

At runtime, VastStreamX builds a `vsx::Graph` of `Operator`s — VDSP preprocessing fusion ops plus a `ModelOperator` — wraps it in a `vsx::Stream` with a `StreamBalanceMode` (`kBM_RUN`), and calls `stream->Build()`. Execution is `RunSync`, or fully asynchronous via `process_async` → `get_output` → `close_input` → `wait_until_done`.

**Concurrency comes from multiple model instances per die** (`--instance 4`, `queue_size`), not from one large kernel. The die is oversubscribed by independent streams.

### Heterogeneous op placement — explicit and manual

| Placement | Examples |
|---|---|
| VDSP (via JSON-selected fusion op) | `FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR` with mean/std/resize/colour-convert parameters |
| VDSP (hand-mapped structural ops) | PointPillarScatter; Qwen-VL visual rotary embedding |
| **Host CPU** | Qwen-VL `Smart_Resize`; RetinaNet post-processing |

The `VASTAI_*` family of per-operator environment switches on the vLLM path (`VASTAI_WINDOW_PARTITION_OP`, `VASTAI_ROLL_OP`, `VASTAI_PATCH_MERGING_OP`, `VASTAI_OP_DEFORM_ATTN_CORE`, …) points the same way: operator coverage is managed by hand, switch by switch, rather than by a general lowering.

### vLLM path — no graph capture at all

`--enforce-eager` is present in **every** documented vLLM command, and the MinerU integration states it as a rule. There is no CUDA-Graph or `torch.compile` equivalent.

---

## 8. Measured Performance — vendor's own sample docs, per die

These are the only reproducible absolute numbers published anywhere for this hardware. They are **per die**.

| Workload | Result | Conditions |
|---|---|---|
| H.264 1080p decode | **1547 fps** max throughput; 509 fps min latency | DCLK 650 MHz; 10 instances vs 1 → ≈51 ch @30 fps per die |
| H.265 1080p decode | **1861 fps** max throughput | DCLK 650 MHz → ≈62 ch @30 fps per die |
| ResNet-50 INT8 | **3231 qps** (batch 8); 941 qps (batch 1) | 880 MHz |
| MobileViT | 43.9 qps | batch 1 |
| `planar_argmax` custom op | 3008 qps, p50 1329 µs | [19,512,512] fp16, 4 instances |

---

## 9. Vendor Claims and the Vendor's Own Counter-Evidence

Every headline claim on vastaitech.com is **relative and unbaselined** — no named comparison part, no workload, no date:

| Claim | Source |
|---|---|
| "同等功耗下2倍以上于主流GPU的最高吞吐率" | VA1 page |
| "延时不到GPU最高吞吐率下延时的5%" | VA1 page |
| "推理性能达到同功耗主流GPU的2倍以上，延时低至6%" | VA10 page |
| "2–10× AI throughput of GPUs at equal power" | VA1 page (EN) |
| DSA "3–5× traditional GPUs" | interview |

All are **vendor claims** and must be labelled as such.

**The vendor also published a table that cuts the other way.** `VastModelZOO/vlm/glm_ocr/vllm/README.md` compares backends on OmniDocBench end-to-end, layout model on CPU in every row:

| Backend | Overall ↑ | Model Infer Cost |
|---|---|---|
| NVIDIA H800 BF16 TP1 | 95.391 | **15.5 min** |
| VACC-VA16 BF16 TP1 | 95.647 | **13 h 26 min** |
| VACC-VA16 BF16 TP2 | 95.611 | 9 h |
| VACC-VA16 BF16 TP4 | 95.635 | 6 h 33 min |

Accuracy is on par with — marginally above — H800; wall clock is roughly **50× slower at TP1 and ~25× slower at TP4**. Caveats that must travel with the citation: one workload, an OCR VLM, full-dataset wall clock, layout model on CPU in all rows, and VA16 TP1 uses one **die** against a whole H800. It remains the vendor's own apples-to-apples table and the most useful maturity signal available for this hardware.

---

## 10. Physical / Platform

| Parameter | Value |
|---|---|
| Process node | **7 nm** for SV100 and SG100. **Not disclosed** for the silicon in VA16 / VA10L / VA1L |
| Foundry | **Not disclosed** for any part — do not assume TSMC |
| TDP | VA1 **70 W** (vendor page) / **75 W** (launch coverage) — sources conflict; VA10 **150 W**; VA12 **250 W**; VE1 (edge) 40–65 W. **VA16 / VA10L / VA1L not disclosed** |
| Form factor | VA1: single-width HHHL, no auxiliary power. VA10: full-height 3/4-length. **VA16 / VA10L / VA1L not disclosed** |
| Die size / transistor count / package | **Not disclosed** for every chip |
| Cooling, slot width, aux power (current cards) | **Not disclosed** |
| Host platforms | x86_64 and **aarch64** (both wheel sets shipped) |
| Validated platform | Ubuntu 22.04.3 with a **Hygon C86-4G** CPU |
| Domestic OS support | Kylin (麒麟), UOS (统信), Anolis (龙蜥), OpenEuler — **vendor claim** |
| Server OEM certification | 通泰易 TG657V2 / TG658V3 / TG659V2 with VA16, 2025-06-13 |

---

## 11. Product Portfolio

| SKU | Generation | INT8 TOPS | Memory | TDP | Form factor |
|---|---|---|---|---|---|
| **载天 VA16** | 2026 flagship | **not disclosed** | **128 GB (4 × 32 GB dies)** | **not disclosed** | **not disclosed** |
| **载天 VA10L** | 2026 | **not disclosed** | **128 GB (4 × 32 GB dies)** | **not disclosed** | **not disclosed** |
| **载天 VA1L** | 2026 | **not disclosed** | **not disclosed** | **not disclosed** | **not disclosed** |
| 载天 VA1L (2023) | SG100 | 200 (72 TFLOPS FP16) | 64 GB (derived) | not disclosed | PCIe |
| VA12 | SG100 | 512 (160 TFLOPS FP16) | not disclosed | 250 W | PCIe |
| 南禺 VG1600 | SG100 | not disclosed | not disclosed | not disclosed | Cloud-gaming graphics card |
| 南禺 VG1800 | SG100 | not disclosed | not disclosed | not disclosed | Cloud-desktop graphics card |
| 南禺 VG14 | SG100 | not disclosed | not disclosed | not disclosed | Workstation card |
| 载天 VA10 | SV100 | 400 | not disclosed | 150 W | FH 3/4-length PCIe |
| 载天 VA1 | SV102 | >200 | 32 GB (16 GB SKU = 2 × 8 GB) | 70 W / 75 W | HHHL single-width, PCIe 4.0 ×16 |
| 载天 VA1V | SV100 | not disclosed | not disclosed | not disclosed | Video card: AV1, 8K 10-bit, 2× 8K HDR@60+fps |
| VGX VA16 appliance (8-card) | 2026 | not disclosed | **1 TB** | not disclosed | Appliance |
| VGX VA16 appliance (16-card) | 2026 | not disclosed | **2 TB** | not disclosed | Appliance |

*VE1S / VE1M / VE1V / VS1000 are edge/embedded parts and are out of scope for this datacenter entry.*

> **Catalog status (2026-08-08).** VA16, VA10L and VA1L appear **nowhere on vastaitech.com** — the marketing site's timeline and product list both stop in 2023, and the Nanyu nav entry is a dead placeholder. The current line is documented only in `github.com/Vastai`, `vllm-vacc.vastaitech.com`, and 2025–2026 Chinese press. **This is the single biggest trap for anyone researching this vendor from the marketing site.**

---

## 12. Deployment Status

**Shipping / commercially available — NOT verified deployed at scale.**

| Date | Event |
|---|---|
| 2018-12 | Company founded, Shanghai |
| 2021-03 → 2021-06 | SV100 tape-out → first silicon |
| 2021-07-08 | 载天 VA1 launched (>200 TOPS INT8, 32 GB, 75 W, PCIe 4.0 ×16) |
| 2022-Q1 | SV100 mass production |
| 2022-09 | **VUCA architecture announced**; VA10 (400 TOPS, 150 W) and edge VE1 launched |
| 2022-10 → 2023-02 | SG100 tape-out → first silicon (ran commercial games within 24 h) |
| 2023-04 | SG100 mass production |
| 2023-07-06 | **WAIC 2023**: SG100 GPU, 南禺 VG1600 / VG1800 / VG14, 载天 VA1L, VA12 |
| 2024-04-10 | 海马云 (Haima Cloud) strategic partnership — **thousand-card** cloud-rendering/AI cluster on domestic ARM + VastaiTech GPUs |
| 2025-06-13 | Server-OEM certification: 通泰易 TG657V2 / TG658V3 / TG659V2 with the VA16 "训推一体" card |
| 2025-07-18 | **Entered IPO counselling (辅导备案)** with CITIC Securities, targeting a domestic A-share listing — *counselling, not an application* |
| 2026-04-27 | 载天 VA16 reported at 128 GB with **FP4 + FP8** mixed precision and DeepSeek-V4 adaptation; up to 2 TB appliance |
| 2026-05-09 | VastStreamX release 26.04 |
| 2026-08-08 (this scan) | All four `github.com/Vastai` repos pushed within the past week; VA16 / VA10L / VA1L still absent from vastaitech.com |

**No named end customer with a disclosed deployment size exists.** Kuaishou is reported by 界面新闻 in 2021 as both investor and customer, with no scale given. **No third-party benchmark of any kind exists.** **No ISCA / MICRO / Hot Chips / ISSCC / arXiv publication exists.** No shipment volume, revenue, or manufacturing capacity has been disclosed.

---

## 13. Open Items — not disclosed anywhere

1. Memory **technology** (HBM / GDDR / LPDDR / DDR) — every part.
2. Memory **bandwidth** in GB/s — every part.
3. On-chip SRAM / scratchpad capacity and bandwidth; register-file size and organisation.
4. Counts of VDSP cores, decode MCUs and encode MCUs per die.
5. AI-engine internals: MAC array dimensions, systolic vs SIMD, MACs/cycle, native tile size.
6. Peak TOPS / TFLOPS for VA16, VA10L and the current VA1L; FP8 and FP4 peak rates on VA16.
7. TDP, form factor, slot width, cooling, aux power and PCIe generation for VA16 / VA10L / VA1L.
8. Process node and chip family for VA16 / VA10L / VA1L; foundry for SV100 and SG100.
9. Die size, transistor count, package type — every chip.
10. Whether an SDK "die" is a separate silicon die in an MCM or a partition of a monolithic die.
11. Die-to-die interconnect protocol, topology and bandwidth; whether any card-to-card fabric exists beyond PCIe.
12. Scale-out fabric: NIC, RDMA support, VCCL inter-node transport.
13. VCCL collective operations, algorithms, bus bandwidth and API.
14. Semantics of the VAMC flags `cluster_mode`, `data_transport_mode`, `data_type`, `opt_level`, `output_layout`, `stream_mode`, `split_convergence_points`, `requant_suppress`.
15. Whether on-chip memory is hardware-cached or a software-managed scratchpad (strongly implied to be the latter; never stated).
16. Current per-card memory capacity of VA1L.
17. 南禺 VG1600 / VG1800 / VG14 specifications of any kind.
18. Named end customers with disclosed deployment scale; unit shipments; revenue; manufacturing capacity.
19. Any independent third-party benchmark; any peer-reviewed or conference publication.

---

## Sources

Vendor primary:
- [About — company history and milestone timeline](https://www.vastaitech.com/company/about)
- [载天 VA1 product page](https://www.vastaitech.com/product/general/va1)
- [载天 VA10 product page](https://www.vastaitech.com/product/general/va10)
- [载天 VA1V video card](https://www.vastaitech.com/product/video/va1v)
- [载天 VE1V video card](https://www.vastaitech.com/product/video/ve1v)
- [智能一体机 solution page](https://www.vastaitech.com/solution/all-in-one)
- [Support — 3-year warranty and RMA process](https://www.vastaitech.com/support)
- [vLLM × VastAI recipe site (VA16 / VA10L / VA1L)](https://vllm-vacc.vastaitech.com/)
- [vLLM recipe — DeepSeek-V3 (8× VA16, TP32, FP8)](https://vllm-vacc.vastaitech.com/deepseek-ai/DeepSeek-V3)
- [vLLM recipe — Qwen3-32B ("VA16128G (4×32G) / VA10L128G (4×32G)")](https://vllm-vacc.vastaitech.com/Qwen/Qwen3-32B)

Vendor's own code:
- [card_info sample — VA1-16G, Die ID 0/1, 8 GB per die, util{ai,vdsp[],vdmcu[],vemcu[]}](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/card_info/card_info.cpp)
- [video_decode sample — per-die 1547 / 1861 fps @ DCLK 650 MHz](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/video_decode/README.md)
- [planar_argmax custom op — ≤96 channel limit](https://github.com/Vastai/VastStreamX-Samples/blob/main/samples/vdsp_op/custom_op/argmax/argmax_op.hpp)
- [model_base.hpp — Graph / Stream / ModelOperator](https://github.com/Vastai/VastStreamX-Samples/blob/main/common/model_base.hpp)
- [VAMC compiler config — output_ddr, insert_odma, tvm_vacc, gather_data_vccl_dsp_enable](https://github.com/Vastai/VastModelZOO/blob/main/tools/vamc/vamc_config.yaml)
- [LLM compile guidance — seq-len multiple of 16, VACC_STACK_SIZE](https://github.com/Vastai/VastModelZOO/blob/main/llm/README.md)
- [DeepSeek-V3 deployment — single VA16 server, TP32, TP32-PP2](https://github.com/Vastai/VastModelZOO/blob/main/llm/deepseek_v3/README.md)
- [vLLM usage limits — max-concurrency 4](https://github.com/Vastai/VastModelZOO/blob/main/tools/vllm/usage_limits.md)
- [GLM-OCR OmniDocBench — H800 vs VACC-VA16](https://github.com/Vastai/VastModelZOO/blob/main/vlm/glm_ocr/vllm/README.md)
- [Vastai/MinerU — DP unsupported, --enforce_eager mandatory, driver version string](https://github.com/Vastai/MinerU)

Press:
- [腾讯新闻 2021-07-08 — SV100 + VA1 launch](https://news.qq.com/rain/a/20210708A032OI00)
- [界面新闻 — SV100 is a DSA; VA1 = SV102](https://m.jiemian.com/article/6342672.html)
- [芯东西 zhidx 2022-09-08 — CEO interview, VUCA block list](https://zhidx.com/p/344936.html)
- [36Kr 2022-09-05 — VA10 400 TOPS / 150 W](https://www.36kr.com/p/1901732567984512)
- [量子位 2023-07 — WAIC 2023: SG100, Nanyu VG-series, VA1L, VA12](https://www.qbitai.com/2023/07/66614.html)
- [36Kr 2023-07-06 — independent confirmation of the WAIC 2023 figures](https://www.36kr.com/p/2332635957528066)
- [上海证券报 via Sina Finance 2026-04-27 — VA16 128 GB, FP4+FP8](https://finance.sina.com.cn/roll/2026-04-27/doc-inhvxtrc2314774.shtml)
- [同花顺 10jqka 2026-04-27 — same story, verbatim quotes](https://news.10jqka.com.cn/20260427/c676308817.shtml)
- [iCloudNews 2024-04-10 — 海马云 partnership, thousand-card cluster](https://www.icloudnews.net/a/79464.html)
- [eeNews Europe — China's Vastai launches 7nm GPU](https://www.eenewseurope.com/en/chinas-vastai-launches-7nm-gpu-for-ai-visual-apps/)

Regulatory / OEM:
- [CITIC Securities IPO counselling progress report (PDF, primary filing)](https://www.cs.ecitic.com/newsite/tzgg/ipoqyfdgg/202510/P020251023518196738538.pdf)
- [SSE STAR Market IPO review database — negative result](https://query.sse.com.cn/statusAction.do?sqlId=SH_XM_LB)
- [通泰易 2025-06-13 — TG657V2 / TG658V3 / TG659V2 certified with VA16](http://ttyinfo.com/News/info/id/154.html)
