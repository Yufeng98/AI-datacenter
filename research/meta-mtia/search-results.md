# Meta MTIA — Search Results

**Device class:** Inference Accelerator
**Research date:** 2026-04-05

---

## Layer 1 — Device Overview

Meta's MTIA (Meta Training and Inference Accelerator) is a custom ASIC designed for inference workloads at Meta scale, primarily deep learning recommendation models (DLRMs) for Facebook/Instagram ranking and ads. Six chip generations have shipped in approximately two years.

- **MTIA v1 (2023):** TSMC 7 nm, 25 W, 102.4 TOPS INT8
- **MTIA v2 (2024):** TSMC 5 nm, 90 W, ~357 TOPS INT8 (3.5× v1)
- **MTIA 2i (2025):** updated with 256 MB SRAM, 2.7 TB/s SRAM bandwidth

Key references:
- https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/
- https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/
- https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf (ISCA 2025 paper)

---

## Layer 2 — Chip Specifications

| Parameter | MTIA v1 | MTIA v2 |
|---|---|---|
| Process | TSMC 7 nm | TSMC 5 nm |
| Die area | 373 mm² (19.34×19.1 mm) | 421 mm² (25.6×16.4 mm) |
| Clock | 800 MHz | 1.35 GHz |
| INT8 TOPS | 102.4 | ~357 (3.5× v1) |
| FP16 TFLOPS | 51.2 | — |
| TDP | 25 W | 90 W |
| PE grid | 8×8 = 64 PEs | 8×8 = 64 PEs (improved) |
| On-chip SRAM | 128 MB | 256 MB |
| DRAM type | LPDDR5 | LPDDR5 |
| DRAM capacity | 64 GB | 128 GB |
| DRAM bandwidth | 176 GB/s | 204.8 GB/s |
| Host interface | PCIe Gen4 | PCIe Gen5 ×8 |

MTIA 2i: 2.7 TB/s SRAM bandwidth, 256 MB SRAM

---

## Layer 3 — Processing Element (PE) Architecture

Each PE contains:
- 2 RISC-V processor cores (one with vector extension)
- Fixed-function units: matrix multiply/accumulate, data movement, nonlinear function calculation
- 128 KB local SRAM per PE
- Asynchronous dataflow execution: RISC-V generates instructions for fixed-function units; DMA transfers and computations execute as dependencies resolve

64 PEs total → 8×8 grid connected by Network on Chip (NoC)

---

## Layer 4 — Memory Hierarchy

- L0: 128 KB local SRAM per PE
- L1: 128 MB (v1) / 256 MB (v2/2i) shared on-chip SRAM (surrounds PE grid)
  - SRAM bandwidth: 2.7 TB/s (2i)
- L2: LPDDR5 off-chip — 64 GB (v1) / 128 GB (v2) @ 176–204 GB/s
- No HBM (deliberately avoids HBM cost/power)

---

## Layer 5 — Network on Chip

- MTIA v2: doubled NoC bandwidth vs v1
- Low-latency PE-to-PE coordination
- Enables scatter-gather for embedding table lookups

---

## Layer 6 — Software Stack

Full stack (top to bottom):
```
PyTorch (eager mode + torch.compile + torch.export)
    ↓
TorchDynamo (graph capture) + Torch FX IR
    ↓
TorchInductor + MTIA graph compiler
    ↓
Triton-MTIA compiler backend (MLIR → MTIA native)
    ↓
MTIATensor + device memory allocator
    ↓
MTIA Streaming Interface
    ↓
MTIA Firmware Driver
    ↓
MTIA hardware
```

---

## Layer 7 — Compiler

- TorchInductor backend adapted for MTIA
- Triton-MTIA: Meta implemented Triton language support for MTIA hardware
- MLIR + LLVM toolchain
- KernelEvolve: agentic kernel coding for MTIA (ArXiv 2512.23236)
- Supports: vLLM, torch.compile, torch.export

---

## Layer 8 — Workload Target

- **Primary:** Deep Learning Recommendation Models (DLRM) — embedding table lookup, dot-product attention, MLP layers
- **Secondary:** Ranking models for feed/ads, LLM inference (emerging)
- DRAM (not HBM) chosen: DLRM is memory-bandwidth bound not compute bound; cost optimization for recommendations

---

## Layer 9 — Precision / Data Types

- INT8 (primary inference)
- FP16
- Sparsity support: 7× improvement with sparse compute (v2)

---

## Layer 10 — Deployment / Scale

- Hundreds of thousands of chips deployed at Meta
- 12 MTIA v2 cards per chassis (24 chips)
- 3 chassis per rack = 72 accelerators per rack
- ~50 petaflops INT8 per rack (sparse enabled)
- Deployed alongside NVIDIA GPUs and AMD GPUs

---

## Layer 11 — System Integration

- PCIe Gen5 ×8 (v2) — host CPU communication
- Two MTIA chips per accelerator module, 220 W module TDP

---

## Layer 12 — Open Source

- Triton-MTIA backend: partially open (Triton project)
- PyTorch MTIA backend: open (upstreamed to PyTorch)
- Chip design: closed
- ISCA 2025 paper: https://dl.acm.org/doi/10.1145/3695053.3731409

---

## Layer 13 — Competing Chips

- Google TPU (internal)
- AWS Trainium/Inferentia
- Intel Gaudi
- Custom recommendation chips from Baidu (Kunlun), Alibaba (Hanguang)

---

## Layer 14 — Roadmap

- MTIA 2i (2025): 256 MB SRAM, 2.7 TB/s bandwidth
- Six generations in ~2 years (aggressive cadence)
- RISC-V based training chip testing (2026)

---

## Layer 15 — Key Publications

- Meta AI Blog v1: https://ai.meta.com/blog/meta-training-inference-accelerator-AI-MTIA/
- Meta AI Blog v2: https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/
- ISCA 2025 paper: https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf
- KernelEvolve ArXiv: https://arxiv.org/html/2512.23236v1

---

# Update Scan — 2026-08-08

**Change class:** major. The ISCA 2026 Industry Track paper is the first silicon-level disclosure of MTIA 300 and postdates the 2026-04-05 baseline above.

## Layer 16 — New Primary Sources (2026)

| Resource | Type | Date | Why it matters |
|---|---|---|---|
| https://aisystemcodesign.github.io/papers/MTIA300_ISCA2026.pdf | Peer-reviewed industry paper (PDF) | Presented 2026-06-30 | **The** primary source. "MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines", MTIA Team, Meta Platforms. ISCA 2026 Industry Track, Session 5B "Industry Track 2", Tue 2026-06-30, 12:00–12:20 EDT, Room 302; corresponding author Chunqiang Tang. Table I (full spec table), Table III (evaluation testbeds), NIC section, Message Engine section, software-stack section, limitations section. Verified with `pdftotext -layout`. |
| https://aisystemcodesign.github.io/ | Publications index | ongoing | Meta's AI-system-codesign publications page. Renders the MTIA 300 title as "MTIA-300: Meta's Training Chip with Embedded NIC Chiplets and Communication Offloading Engine". Also lists **LoKA** [ISCA'26] and **Triton for MTIA** [IEEE Micro 2026] — neither previously in this repo — alongside the ISCA'23 and ISCA'25 MTIA papers. |
| https://iscaconf.org/isca2026/program/ | Conference program | 2026 | Confirms Session 5B (MTIA 300) and Session 4B "Industry Track 1", Mon 2026-06-29 16:30–18:10, chaired by Shanmathi Natarajan (Meta), containing **KernelEvolve** (17:50–18:10), **Vistara: Making CXL Real** (17:10–17:30), and **From Lab to Fleet: Rowhammer Defense in Cloud SoCs** (17:30–17:50). |
| https://www.computer.org/csdl/proceedings-article/isca/2026/506500b084/2iG0QI12UeY | IEEE CSDL proceedings entry | 2026 | Confirms the MTIA 300 paper is in the ISCA 2026 proceedings. Indexed via search; page body not retrievable in this environment (header only). |
| https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/ | Vendor blog | 2026-03-11 | "Four MTIA Chips in Two Years". **Predates the 2026-04-05 baseline** but is the authority that corrects two repo errors: MTIA 450 is "+75% MX4 FLOPS over MTIA 400" (the "6×" is MX4 vs FP16/BF16 on the same chip), and MTIA 400's HBM capacity is never stated. Also: MTIA 400 two compute chiplets / +400% FP8 / +51% HBM BW / 72-device rack scale-up / "finished testing in our labs"; MTIA 500 +50% BW, up to +80% capacity, +43% MX4 FLOPS, 2×2 smaller compute chiplets, mass deployment 2027. |
| https://about.fb.com/news/2026/04/meta-partners-with-broadcom-to-co-develop-custom-ai-silicon/ | Vendor newsroom PR | 2026-04-14 | Formalizes the Broadcom partnership: chip design, advanced packaging, networking; Broadcom XPU platform; Broadcom Ethernet; "a commitment that exceeds 1GW, which is the first phase of a sustained, multi-gigawatt rollout". **States no process node and no end year** — the circulating "2nm" and "through 2029" figures are aggregator-only. |
| https://hotchips.org/program/conference/ | Conference program | Hot Chips 38, 2026-08-23…25 | Session "AI 1", Tue **2026-08-25**, 2:15–4:15 PM — Meta, "Meta's Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI" (Srinagesh Loke, Cindy Chen, Jatinder Singh). **Disclosure scheduled — content not yet public.** Title does not contain "MTIA". Same session: Microsoft MAIA 200, Cerebras rack-scale WSE, NVIDIA LPU. Session "AI 2" has Google's "The Eighth Generation TPU Family", SambaNova SN50 RDU, and OpenAI. **Re-scan after 2026-08-25.** |

## Layer 17 — Secondary / Trade Reporting (attribute, do not treat as spec)

| Resource | Type | Date | Confidence | Notes |
|---|---|---|---|---|
| https://techcrunch.com/2026/07/09/metas-new-ai-chips-will-begin-production-in-september/ | Trade press | 2026-07-09 | medium-low | Carries a Reuters account of an internal Meta memo: manufacturing of an in-house AI chip code-named **"Iris"** begins September 2026; testing completed in ~6 weeks with no major issues; plan to reach ~7 GW of compute by end-2026 and 14 GW the following year. Reuters itself is blocked from fetching in this environment; Yahoo Finance carried the same memo. **The Iris → MTIA 400 mapping asserted by several aggregators is not confirmed by any primary source.** |
| https://www.nextplatform.com/compute/2026/04/08/contemplating-metas-homegrown-mtia-compute-engine-roadmap/5214899 | Analyst commentary | 2026-04-08 | low for numbers | Useful roadmap framing, but its MTIA 500 384 GB / 512 GB capacity figures are **explicitly flagged by the author as estimates**, and its "MTIA 300 deployed H2 2024" claim contradicts both Meta's March 2026 framing and the ISCA 2026 paper. Do not carry either forward. |

## Layer 18 — Newly Named Components and Terms to Track

- **HCCL** — Meta's MTIA collective communications library (first named in the ISCA 2026 paper). Builds WQE subgraphs dispatched by CPU-C to the 16 Message Engines; drives NIC chiplets over RDMA verbs.
- **Message Engine (ME) / CPU-M** — the 16 on-die collective-offload engines and their scalar RISC-V control cores.
- **CPU-C** — MTIA 300's RISC-V quad SMP control core.
- **Express doorbells** — NIC optimization where the work request itself is the doorbell write (~800 ns saved per transaction).
- **Near Memory Compute (NMC)** — the reduction/DMA datapath inside each Message Engine.
- **"Iris"** — Meta chip codename from the July 2026 Reuters memo report; generation mapping unconfirmed.

## Layer 19 — Open Items for the Next Scan

1. **Hot Chips 38 Meta talk (2026-08-25)** — re-scan after the date; the program title does not name MTIA, so what it covers is currently unknown.
2. ~~**MTIA v2 / 2i PCIe width conflict**~~ — **RESOLVED 2026-08-08.** Both papers agree: MTIA v1 = 8× PCIe Gen4 (16 GB/s), MTIA v2/2i = 8× PCIe Gen5 (32 GB/s), MTIA 300 = 16× PCIe Gen5 (64 GB/s). The repo's "Gen5 ×16 (128 GB/s)" was a transcription error, now corrected throughout.
3. **MTIA 400 deployment status** — still "on the path to deploying" as of the March 2026 blog; watch for a first-deployment announcement.
4. **Process nodes** — no primary source discloses a node for MTIA 300/400/450/500. Watch for a Meta or Broadcom disclosure.
5. **LoKA (ISCA'26)** and **Triton for MTIA (IEEE Micro 2026)** — not yet investigated; both are on Meta's publications index.
6. **Dynamic-shape collectives / device-resident AllToAll in HCCL** — Meta lists these as WIP; watch for a follow-up paper.
