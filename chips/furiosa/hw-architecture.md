# FuriosaAI NPU Hardware Architecture

*as_of: 2026-08-08*
*Chips: Warboy (Samsung 14nm, 2021), RNGD (TSMC 5nm, 2026), Gen 3 (Broadcom co-development, announced 2026-05-27 — no silicon)*

---

## Overview

FuriosaAI has shipped two NPU generations with distinct architectural identities and announced a third. Warboy is a vision-inference PE array on Samsung 14nm. RNGD is an LLM/multimodal inference accelerator on TSMC 5nm based on the novel **Tensor Contraction Processor (TCP)** architecture, which treats all deep learning computations as tensor contractions — a more general primitive than matrix multiplication. The unnamed **third generation**, announced 2026-05-27 as a Broadcom co-development, moves TCP onto a 2nm multi-die chiplet package with HBM4/4E and — for the first time in FuriosaAI's roadmap — an **Ethernet-based scale-up fabric**.

---

## Generation Overview

| Generation | Codename | Status | Process | Compute Primitive | Peak (headline) | On-chip SRAM | Off-chip Memory | Memory BW | Scale-up Fabric | Host IF | TDP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gen 1 | Warboy | Shipping (2021) | Samsung 14nm LPP | PE array | 64 TOPS INT8 | 32 MB | 16 GB LPDDR4X | 66 GB/s | None | PCIe Gen4 x8 | ~50 W |
| Gen 2 | RNGD ("Renegade") | Mass production (declared 2026-05-13) | TSMC 5nm N5 | TCP slice (8 PEs × 64 slices) | **512 TFLOPS FP8** (64 TFLOPS FP8 × 8 PEs) | **256 MB @ 384 TB/s** | **48 GB HBM3** (2 stacks, CoWoS-S, 6.0 Gbps) | 1.5 TB/s | None (PCIe only) | PCIe Gen5 x16 (P2P) | **180 W** (docs: 150 W) |
| **Gen 3** | **None disclosed** | **Announced only (2026-05-27); sampling targeted H1 2028** | **2nm** (TSMC per Korean wire) | **TCP + Broadcom XPU IP** | **Not disclosed** | **Not disclosed** | **HBM4 / HBM4E** — capacity not disclosed | **Not disclosed** | **Broadcom Ethernet scale-up + fabric switches (all-to-all-capable)** | **Not disclosed** | **Not disclosed** |

> **Baseline corrections applied 2026-08-08.** RNGD's HBM3 capacity was previously recorded here as 24 GB; FuriosaAI's own product page states **48 GB** (2 HBM3 stacks via CoWoS-S at 6.0 Gbps). On-chip SRAM was previously "not disclosed"; the product page states **256 MB at 384 TB/s**. The 512 headline figure is **TFLOPS FP8**, not INT8 TOPS — the product page derives it as 64 TFLOPS FP8 per PE × 8 PEs. TDP has a documented split: **150 W** in the developer documentation, **180 W** on the product page and in all 2026 press releases.

---

## 1. Compute Engine

### Warboy

| Parameter | Value |
|-----------|-------|
| Process | Samsung 14nm LPP |
| Die area | 180 mm² |
| Transistors | 5 billion |
| Clock | 2.0 GHz |
| Peak INT8 | 64 TOPS |
| Architecture | PE array |
| Execution model | Compiler-scheduled, no caches |

Warboy's compute engine is a **PE (Processing Element) array** optimized for fixed-shape CNN inference. The compiler maps operator tiles to PE rows/columns and schedules all data movement into the 32 MB on-chip SRAM. No dynamic hardware scheduler.

### RNGD — Tensor Contraction Processor

| Parameter | Value | Source strength |
|-----------|-------|-----------------|
| Process | TSMC 5nm N5 | vendor |
| Transistors | 40 billion | vendor |
| Clock | 1.0 GHz | vendor docs |
| **Peak FP8** | **512 TFLOPS** = 64 TFLOPS FP8 × 8 PEs | vendor product page (verified 2026-08-08) |
| Peak BF16 | 256 TFLOPS | developer docs — reported, not independently confirmed |
| Peak INT8 | 512 TOPS | developer docs — reported, not independently confirmed |
| Peak INT4 | 1,024 TOPS | developer docs — reported, not independently confirmed |
| Processing Elements | 8 PEs | vendor |
| Slices per PE | 64 | vendor |
| Total slices | 512 | vendor |

The TCP architecture is RNGD's defining feature. A **tensor contraction** is a generalization of matrix multiplication to arbitrary tensor ranks:

```
Standard matmul:     C[i,k] = Σ_j A[i,j] × B[j,k]
Tensor contraction:  D[i,l,m] = Σ_{j,k} A[i,j,k] × B[j,k,l,m]
```

By building hardware for the general case, RNGD natively executes:
- GEMM (QKV projections, FFN layers) — special case of contraction
- Attention scores (Q × K^T over batch/heads dimensions) — multi-dim contraction
- Convolution (spatial contraction) — same hardware, no separate conv engine
- Batched GEMM (MoE routing) — trivially maps to batch tensor contraction

The **8 PE × 64 slice grid** is the physical implementation: each PE handles independent contraction dimensions, and each of its 64 slices executes partial accumulations. The compiler assigns tensor index dimensions to PE/slice axes.

### Gen 3 (Broadcom) — announced 2026-05-27, no silicon

| Parameter | Value |
|-----------|-------|
| Status | Announced partnership / roadmap item; **no tape-out, no silicon, no shipping** |
| Sampling target | H1 2028 (Korean wire coverage 2026-05-28; absent from the English release) |
| Process | 2nm; **TSMC** 2nm compute die per Yonhap / Digital Today / The AI |
| Packaging | Multi-die chiplet system-in-package, Broadcom advanced packaging |
| Compute architecture | FuriosaAI TCP paired with Broadcom **XPU Technology and IP Platform** |
| Peak throughput (FP8 / BF16 / INT8 / FP4) | **Not disclosed** |
| Die count / die sizes | **Not disclosed** |
| Clock | **Not disclosed** |
| Mass-production date | **Not disclosed** |

Positioning in the vendor's own words: the platform optimizes **data movement and memory access** rather than raw FLOPS, for agentic and MoE inference. The compute primitive is described only as a continuation of TCP; no microarchitectural detail (slice counts, PE organization, MXU dimensions) is public.

> Aggregator claims of "two 2nm compute chiplets plus two IO dies", "12 memory sites / 432 GB", and "3.5D XDSiP" are **not vendor-disclosed** (traced to Wccftech speculation) and are excluded.

### Precision Support

| Format | Warboy | RNGD | Gen 3 |
|--------|--------|------|-------|
| FP32 | No | No (host only) | Not disclosed |
| FP16 | Yes | Yes | Not disclosed |
| BF16 | No | Yes | Not disclosed |
| FP8 | No | Yes | Not disclosed |
| INT8 | Yes | Yes | Not disclosed |
| INT4 | No | Yes | Not disclosed |

---

## 2. Memory Hierarchy

### Warboy

| Level | Capacity | Bandwidth |
|-------|----------|-----------|
| On-chip SRAM | 32 MB | High (internal) |
| Off-chip | 16 GB LPDDR4X | 66 GB/s |

No caches. All SRAM allocation is determined by the compiler (activation buffers, weight tiles). LPDDR4X reflects Warboy's origin as a vision NPU with small model sizes (50–200 MB).

### RNGD

| Level | Capacity | Bandwidth |
|-------|----------|-----------|
| **On-chip SRAM** | **256 MB** | **384 TB/s** |
| **Off-chip HBM3** | **48 GB** (2 stacks, CoWoS-S, 6.0 Gbps) | 1.5 TB/s |

> **Corrected 2026-08-08.** This table previously recorded "24 GB (2× modules)" and "On-chip SRAM: not disclosed". FuriosaAI's product page states 48 GB HBM3 and 256 MB SRAM at 384 TB/s. The 24 GB figure was wrong by 2×.

The switch from LPDDR4X to HBM3 (23× bandwidth increase) is the primary enabler for LLM inference on RNGD. The 48 GB capacity accommodates:
- ~30B-class models in FP8 on a single card (Samsung SDS serves Qwen3 and gpt-oss 120B on RNGD; gpt-oss 120B spans 2 cards at 5.8 ms TPOT)
- Tensor-parallel across 2 cards for larger models (96 GB combined)
- KV-cache for long contexts (32K tokens supported)

The 256 MB SRAM tier at 384 TB/s is a **256× bandwidth step over HBM3** and is the layer the compiler tiles against; it is comparable in capacity to Meta MTIA 2i's 256 MB LLC and TPU v8's 384 MB, and is unusually large for a 180 W PCIe card.

### Gen 3 (announced)

| Level | Capacity | Bandwidth |
|-------|----------|-----------|
| On-chip SRAM | Not disclosed | Not disclosed |
| Off-chip memory | **HBM4 / HBM4E** — capacity not disclosed | Not disclosed |

HBM4/4E is the only memory attribute FuriosaAI has disclosed for Gen 3. Stack count, capacity, and bandwidth are all **not disclosed**.

---

## 3. Multi-Tenancy Partitioning (RNGD-specific)

RNGD supports **hardware-level NPU partitioning** (SR-IOV) into 2, 4, or 8 isolated virtual NPUs:

| Partition Mode | Virtual NPUs | PE slices per vNPU | HBM3 per vNPU |
|----------------|--------------|---------------------|----------------|
| Full chip | 1 | 512 | **48 GB** |
| Half | 2 | 256 | **24 GB** |
| Quarter | 4 | 128 | **12 GB** |
| Eighth | 8 | 64 | **6 GB** |

> **Recomputed 2026-08-08** on the corrected 48 GB HBM3 capacity (the prior table divided the erroneous 24 GB figure).

Each partition is fully hardware-isolated — separate PE allocation, memory bandwidth, and PCIe command queue. This enables Kubernetes-native multi-model scheduling without performance interference. Samsung SDS's NPU-as-a-Service (launched 2026-07-20 on Samsung Cloud Platform) sells RNGD capacity directly in **1/2/4/8-card configurations**, exposing the partition modes as a commercial SKU axis.

Partitioning is not described for Gen 3; **not disclosed**.

---

## 4. Host Interface / Package

| Parameter | Warboy | RNGD | Gen 3 (announced) |
|-----------|--------|------|-------------------|
| Form factor | PCIe accelerator card | PCIe accelerator card (with card-to-card P2P) | Not disclosed |
| Host interface | PCIe Gen4 x8 | PCIe Gen5 x16 | Not disclosed |
| TDP | ~50W | **180 W** (product page + all 2026 press); **150 W** (developer docs) | Not disclosed |
| Packaging | Monolithic | Monolithic die + 2 HBM3 stacks on CoWoS-S | **Multi-die chiplet SiP** (Broadcom advanced packaging) |
| Production node | Samsung 14nm | TSMC 5nm | 2nm (TSMC per Korean wire) |
| Production start | 2021 | January 2026; mass production formally declared 2026-05-13 | Not disclosed (sampling targeted H1 2028) |

FuriosaAI designed the RNGD chip in-house (no SemiFive intermediary as with Warboy). TSMC 5nm manufacture, SK hynix HBM3.

**On the 150 W / 180 W discrepancy.** Both figures come from FuriosaAI. The developer documentation lists 150 W as the card spec; the RNGD product page and every 2026 press release (including the Samsung SDS launch) quote 180 W. This survey records 150 W as the documented device figure and 180 W as the vendor's headline/system-level figure; the two are not reconciled in any public source.

---

## 5. Scale-up

### Warboy / RNGD — no proprietary fabric

FuriosaAI's shipping silicon has no proprietary scale-up interconnect. Multi-card configurations:
- NXT RNGD server: originally 4 RNGD cards per server; **as of 2026-07-07 FuriosaAI describes the NXT RNGD Server as holding up to 8 RNGD accelerators in a 3 kW-class system**
- Card-to-card PCIe P2P for direct accelerator-to-accelerator transfers
- Tensor-parallel and pipeline-parallel LLM inference across cards via furiosa-llm tensor parallelism
- Scale-out via standard Ethernet

### Gen 3 — first scale-up fabric in the roadmap (announced 2026-05-27)

| Attribute | Value |
|---|---|
| Fabric | **Broadcom Ethernet scale-up + fabric switches** |
| Topology | Described as **all-to-all-capable**, targeting MoE expert-routing traffic |
| Link bandwidth per chip | **Not disclosed** |
| Scale-up domain size (chips / nodes / rack) | **Not disclosed** |
| Scale-out | **Not disclosed** (Ethernet implied) |

This is the architectural discontinuity in the Gen 3 announcement. Warboy and RNGD are PCIe-only parts whose multi-chip parallelism is mediated by the host; Gen 3 adds a switched Ethernet scale-up domain in the accelerator itself, placing FuriosaAI in the same structural family as Google TPU v8t (Broadcom) and OpenAI Titan (Broadcom) — Ethernet-based scale-up designed for data movement rather than peak FLOPS. No bandwidth or domain-size numbers exist to compare against those parts.

---

## 6. Power and Efficiency

| System | TDP | Workload | Throughput | Efficiency |
|--------|-----|----------|------------|------------|
| NXT RNGD server (4 cards) | 3 kW | EXAONE 3.5 32B | 60 tok/s @ BS=1 | baseline |
| NVIDIA DGX H100 (8 GPUs) | >10 kW | EXAONE 3.5 32B | ~27 tok/s @ BS=1 (est.) | 3.3× worse power, 0.45× throughput |
| RNGD rack vs H100 rack (same power budget) | — | EXAONE 3.5 32B | 3.75× more tokens/watt | LG AI Research validated |
| RNGD vs 4× RTX PRO 6000 Blackwell SE (bare metal) | — | Qwen3-32B FP8 (Lablup Backend.AI) | 95% of GPU peak throughput @ 256-concurrency | **1.3×–1.5× throughput/watt; 30–44% lower power; TTFT <1 s through concurrency 32 vs >2.9 s** — *vendor white paper, 2026-08-06; not independent* |

> **No MLPerf data exists for RNGD.** FuriosaAI is not among the 24 submitting organizations for MLPerf Inference v6.0 (published 2026-04-01). All RNGD performance figures in this survey are vendor- or customer-published.

The power efficiency advantage stems from three factors:
1. **TCP efficiency**: No wasted FLOPS on padding/decomposition — tensor contraction maps directly to hardware
2. **Dedicated inference chip**: No training circuitry (no gradient accumulation, no optimizer states, no large register files for backward pass)
3. **180W TDP per card**: 25% of H100's 700W for comparable inference throughput on many LLM models

---

## 7. Foundry History

| Generation | Foundry | Node | Notes |
|------------|---------|------|-------|
| Warboy | Samsung Foundry | 14nm LPP | Via SemiFive ASIC platform (SK Telecom subsidiary) |
| RNGD | TSMC | 5nm N5 | Moved to TSMC for better transistor density and HBM3 interposer ecosystem; HBM3 from SK hynix |
| **Gen 3** | **TSMC** (compute die, per Korean wire coverage) | **2nm** | Broadcom supplies XPU IP, advanced multi-die packaging, and Ethernet scale-up/fabric switching; HBM4/4E supplier not disclosed |

The foundry switch from Samsung to TSMC for RNGD reflects broader industry dynamics. TrendForce (Oct 2024) reported multiple Korean AI IC design firms adopting dual-foundry models, citing TSMC's yield advantage and its established HBM3/CoWoS packaging ecosystem. Gen 3 continues on TSMC and adds Broadcom as a co-design and packaging partner — the same structural arrangement Google (TPU v8t) and OpenAI (Titan) use.

---

## 8. Disclosure Status Notes (2026-08-08)

- **Hot Chips 38** (Aug 23–25, 2026, Stanford): **no FuriosaAI talk on the program.** RNGD's last conference disclosure remains Hot Chips 2024 + MICRO 2025.
- **"RNGD-S"** is **not a confirmed product**. It appears in neither furiosa.ai/rngd, the 2026.3.0 developer docs, nor any 2026 press release. Current product names: RNGD (PCIe card), NXT RNGD Server, Warboy / Vision NPU.
- **No Broadcom-issued press release** confirming the Gen 3 partnership was located; the announcement is FuriosaAI-side with a Broadcom executive quote.

---

## Resources

- [FuriosaAI RNGD Product Page](https://furiosa.ai/rngd) — source for 48 GB HBM3, 256 MB SRAM @ 384 TB/s, 512 TFLOPS FP8, 180 W
- [FuriosaAI Warboy Specs](https://furiosa.ai/warboy/specs)
- [RNGD Developer Docs](https://developer.furiosa.ai/latest/en/overview/rngd.html) — source for 150 W, 1.0 GHz, BF16/INT8/INT4 figures
- [MICRO 2025: TCP Paper](https://web.ist.utl.pt/nuno.lopes/pubs/tcp-micro25.pdf)
- [Hot Chips 2024 Analysis (Chips & Cheese)](https://chipsandcheese.com/p/furiosaais-rngd-at-hot-chips-2024-accelerating-ai-with-a-more-flexible-primitive)
- [HPCwire TCP Deep Dive](https://www.hpcwire.com/2025/09/30/the-fast-and-the-furiosaai-korean-chip-startup-takes-aim-at-nvidia-gpus-with-tensor-contraction-architecture/)
- [Broadcom partnership announcement (2026-05-27)](https://furiosa.ai/blog/furiosaai-partners-with-broadcom-to-build-next-generation-inference-platform-for-the-agentic-era)
- [The AI (Korean, 2026-05-28) — TSMC 2nm compute die, H1 2028 sampling](https://www.newstheai.com/news/articleView.html?idxno=20758)
- [Digital Today (Korean, 2026-05-28) — 2nm vs TSMC 5nm, HBM3 → HBM4/4E, chiplet + Ethernet switching](https://www.digitaltoday.co.kr/news/articleView.html?idxno=669758)
- [Yonhap AKR20260528119900017 (2026-05-28)](https://www.yna.co.kr/view/AKR20260528119900017) — indexed snippet only; host not directly fetchable
- [RENEGADE Summit 2026 — mass production declared](https://furiosa.ai/blog/experience-renegade-summit-2026)
- [Backend.AI white paper (vendor benchmark, 2026-08-06)](https://furiosa.ai/blog/white-paper-benchmarking-rngd-on-backend-ai)
