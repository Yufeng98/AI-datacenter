# Rebellions ATOM — Search Results

*chip: rebellions-atom*
*as_of: 2026-08-08*

---

## Summary

Rebellions is a South Korean AI chip startup founded in 2020. It merged with SAPEON Korea (Dec 2024) to form Korea's first AI chip unicorn. The company's product line spans three generations: **ATOM** (inference, Samsung 5nm, PCIe), **REBEL / ATOM-Max** (next-gen monolithic, Samsung 4nm), and **REBEL-Quad** (UCIe chiplet, 144 GB HBM3e, Hot Chips 2025). Rebellions is backed by Arm, Samsung, and SK Telecom with $400M pre-IPO raised in March 2026 at a $2.3B valuation.

---

## Discovered Resources

### Documentation

| URL | Type | Priority | Notes |
|-----|------|----------|-------|
| https://rebellions.ai/atom-architecture-finding-the-sweet-spot-for-genai/ | Architecture blog | high | ATOM CGRA/Neural Engine deep dive |
| https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf | White paper | high | ATOM GenAI white paper |
| https://rebellions.ai/atom-max-boosted-performance-for-large-scale-inference/ | Product blog | high | ATOM-Max specifications |
| https://rebellions.ai/white-paper-atom-max/ | White paper | high | ATOM-Max white paper |
| https://docs.rbln.ai/latest/index.html | SDK documentation | high | RBLN SDK user guide |
| https://rebellions.ai/understanding-rbln-compiler/ | Compiler blog | high | RBLN compiler internals |
| https://rebellions.ai/developers/ | Developer portal | medium | SDK entry point |
| https://rebellions.ai/rebel-quad-live-at-hot-chips-2025-worlds-first-ucie-advanced-npu-in-action/ | Demo video/blog | high | REBEL-Quad live demo at Hot Chips 2025 |
| https://rebellions.ai/newsroom/rebellions-debuts-rebel-quad-at-hot-chips-2025-breaking-ais-energy-tax-with-high-performance-chiplet-innovation/ | Press release | high | REBEL-Quad announcement |
| https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144tb-hbm3e-accelerator-at-hot-chips-2025/ | Analysis | high | STH detailed REBEL-Quad analysis |
| https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100 | Technical report | high | ISSCC 2026 REBEL-Quad / Rebel100 details |
| https://www.nextplatform.com/2025/12/23/rebellions-ai-puts-together-an-hbm-and-arm-alliance-to-take-on-nvidia/ | Analysis | high | REBEL architecture deep dive |
| https://www.eetimes.com/rebellions-builds-chiplet-roadmap-merges-with-sapeon/ | Analysis | high | Chiplet roadmap + Sapeon merger |
| https://www.synopsys.com/blogs/chip-design/energy-efficient-ai-accelerator-data-centers.html | Design partner blog | medium | Synopsys REBEL design details |
| https://www.theregister.com/2026/03/30/rebellions_ai_rackscale/ | News | medium | RebelRack / RebelPOD rack-scale platform |

### Open-Source Repositories

| URL | Type | Priority | Notes |
|-----|------|----------|-------|
| https://github.com/rebellions-sw/rbln-model-zoo | Model zoo | high | Compile-once deploy anywhere |
| https://github.com/rebellions-sw/optimum-rbln | HuggingFace integration | high | HF Transformers + Diffusers on ATOM |
| https://github.com/vllm-project/vllm/issues/7247 | vLLM RFC | medium | RBLN NPU vLLM plugin RFC |

### Funding / Corporate

| Event | Date | Amount | Investors |
|-------|------|--------|-----------|
| Series A | 2021 | — | KT, Korea Investment Partners |
| Series B | 2022 | — | — |
| Series C | 2024-01 | $124M | Samsung Electronics, KT, SoftBank Ventures |
| Sapeon merger | 2024-12 | — (equity swap 2.4:1) | SK Telecom |
| Series D / $250M | 2025 | $250M | Arm, Samsung Ventures |
| Pre-IPO | 2026-03 | $400M | — ($2.3B valuation) |

---

## Key Findings

### ATOM (Generation 1 — Samsung 5nm)
- **Architecture**: CGRA (Coarse-Grained Reconfigurable Array) + 8 Neural Engines
- **Compute**: 32 TFLOPS FP16, 128 TOPS INT8
- **On-chip SRAM**: 64 MB
- **Off-chip memory**: 16 GB GDDR6, 256 GB/s (RBLN-CA12 card)
- **Host interface**: PCIe Gen5 x16, single-slot FHFL card, TDP 60–130 W
- **Multi-instance**: 16 hardware-isolated instances
- **Deployments**: KT Cloud (Korea, 2023), SK Telecom AI services, Japan, Saudi Arabia, US

### ATOM-Max (ATOM variant)
- Boosted clocks/power for large-scale inference
- Same PCIe form factor, higher TDP

### REBEL / Rebel Single (Generation 2 — Samsung 4nm)
- 16 neural cores; 128 MB SW-managed SRAM (64 MB L1 distributed + 64 MB L2 shared)
- Neural cores use RISC ISA; custom instruction set for IBUFs (input buffers)
- 8 neural cores per cluster with mesh interconnect through SRAM
- PCIe card with 600 W TDP per REBEL-Quad package
- Target: frontier-scale MoE models, peta-scale inference

### REBEL-Quad (Generation 2 chiplet — Hot Chips 2025)
- **Chiplet config**: 4 compute ASIC dies + 4 HBM3e stacks + 4 integrated silicon capacitors (ISC)
- **Interconnect**: UCIe-Advanced die-to-die, 16 Gbps, 4 TB/s aggregate bandwidth
- **Memory**: 144 GB HBM3e (4 × 36 GB 12Hi stacks), 4.8 TB/s aggregate memory bandwidth
- **Compute**: 2,048 TFLOPS FP8 (2.048 PFLOPS)
- **Performance**: 1.6× throughput, 50% lower power vs top-tier GPU (Llama 3.3 70B FP8); 3.2× TPS/W
- **OCP listing**: REBEL-Quad listed on Open Compute Project chiplets registry
- **ISSCC 2026**: Presented as Rebel100; claimed to equal H200 performance at lower power

### Rack-Scale Systems (2026)
- **RebelRack**: 4 nodes × 8 Rebel100 cards = 32 accelerators; 64 PFLOPS FP8; 4.6 TB HBM3e; 153.6 TB/s aggregate memory BW; quad-400 Gbps networking
- **RebelPOD**: 8–128 nodes × 8 accelerators; 800 Gbps Ethernet interconnect

### RBLN SDK
- Two-phase compiler: Frontend Compiler (graph capture from PyTorch/TF/HF) + Backend Compiler (RISC ISA code gen for Neural Engines)
- Compute Library generates GEMM, normalization, nonlinear operations via Rebellions' RISC ISA
- Dependency analysis for DRAM/SRAM scheduling and data flow
- Supports: PyTorch, TensorFlow, HuggingFace Transformers/Diffusers
- vLLM plugin (official)
- Optimum-RBLN for seamless HF model deployment
- RBLN Model Zoo: compile-once deploy anywhere

### Corporate
- Founded 2020 by ex-Qualcomm engineers; CEO Park Sung-hyun
- Merged with SAPEON Korea (SK Telecom spinoff) Dec 2024 at 2.4:1 equity ratio; ~$927M combined value
- Strategic investors: Arm (architecture license + investment), Samsung (fab partner + investor), SK Telecom, KT
- Partners: Pegatron (hardware manufacturing), Synopsys (EDA design), DOCOMO Innovations (Japan)
- IPO preparation underway (2026)

---

## Sources

- [ATOM Architecture Blog](https://rebellions.ai/atom-architecture-finding-the-sweet-spot-for-genai/)
- [ATOM White Paper (PDF)](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf)
- [ATOM-Max Blog](https://rebellions.ai/atom-max-boosted-performance-for-large-scale-inference/)
- [REBEL-Quad Hot Chips 2025 Announcement](https://rebellions.ai/newsroom/rebellions-debuts-rebel-quad-at-hot-chips-2025-breaking-ais-energy-tax-with-high-performance-chiplet-innovation/)
- [ServeTheHome: REBEL-Quad 144GB HBM3E](https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144tb-hbm3e-accelerator-at-hot-chips-2025/)
- [Tom's Hardware: ISSCC 2026 Rebel100](https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100)
- [RBLN SDK Docs](https://docs.rbln.ai/latest/index.html)
- [Understanding RBLN Compiler](https://rebellions.ai/understanding-rbln-compiler/)
- [Rebellions Sapeon Merger](https://rebellions.ai/newsroom/rebellions-and-sapeon-korea-complete-merger-launching-koreas-first-ai-chip-unicorn/)
- [Next Platform: REBEL Architecture](https://www.nextplatform.com/2025/12/23/rebellions-ai-puts-together-an-hbm-and-arm-alliance-to-take-on-nvidia/)
- [EE Times: Chiplet Roadmap](https://www.eetimes.com/rebellions-builds-chiplet-roadmap-merges-with-sapeon/)
- [The Register: RebelRack](https://www.theregister.com/2026/03/30/rebellions_ai_rackscale/)
- [Rebellions $400M Pre-IPO](https://techcrunch.com/2026/03/30/ai-chip-startup-rebellions-raises-400-million-at-2-3b-valuation-in-pre-ipo-round/)
- [Rebellions $250M Series D (Arm + Samsung)](https://rebellions.ai/newsroom/rebellions-raises-250-million-to-advance-the-next-generation-ai-infrastructure-backed-by-arm-and-samsung/)
- [OCP REBEL-Quad Listing](https://www.opencompute.org/chiplets/76/rebel-quad-ai-accelerator-ai-soc)

---

## Resources Added 2026-08-08 (window 2026-04-06 → 2026-08-08)

### Documentation / SDK

| URL | Type | Priority | Notes |
|-----|------|----------|-------|
| https://docs.rbln.ai/latest/supports/release_note.html | SDK release notes | **high** | Authoritative for compiler/driver version strings. Four releases in window: v0.10.3 (04-30), v0.10.4.post1 (05-29), v0.11.0.post1 (06-26, **breaking** Transformers-v5 format change), v0.11.1.post1 (07-31, fixes mimalloc numerical bug in v0.11.1). Also documents the **breaking** RBLN NPU Operator v0.3.x→v0.4.0 change |

### Vendor Press Releases (primary)

| URL | Date | Priority | Notes |
|-----|------|----------|-------|
| https://rebellions.ai/newsroom/ | — | high | Newsroom index |
| https://rebellions.ai/newsroom/rebellions-collaborates-with-sk-telecom-and-arm-targeting-sovereign-ai-and-telecom-infrastructure/ | 2026-04-10 | **high** | Rebellions + SK Telecom + Arm **MOU**. Introduces **RebelCard** (Rebel100 module card, four NPU chiplets, 5th-gen HBM/HBM3E, air-cooled) and names the host CPU as **Arm AGI CPU** on **Arm Neoverse CSS V3**. Contains **no launch date** — used to refute the aggregator "Q3 2026" claim |
| https://rebellions.ai/newsroom/rebellions-and-giga-computing-sign-mou-to-develop-next-generation-ai-server-and-rack-scale-solutions/ | 2026-06-17 | medium | **MOU** with Giga Computing (GIGABYTE server arm) at Computex 2026. No timeline, specs, or SKU |
| https://rebellions.ai/newsroom/rebellions_squeezebits_acquisition_260630/ | 2026-06-30 | **high** | SqueezeBits acquisition — new model-compression / inference-serving tier (OwLite, Fits on Chips, Yetter) |
| https://rebellions.ai/newsroom/rebelserver_run_k1/ | 2026-07-23 | **high** | **RebelServer** runs SKT **A.X K1** (500B+ parameter **MoE**) in a single server. No quantitative metrics |
| https://rebellions.ai/newsroom/rebellions-closes-400-million-pre-ipo-and-launches-rebelrack-and-rebelpod-to-accelerate-global-expansion/ | 2026-03-30 | **high** | Primary launch release for RebelRack/RebelPOD — **publishes no technical specs**. Basis for re-grading the repo's RebelRack spec row to secondary-sourced. Also gives $400M / ~$2.34B post-money / $850M total, and adds Aramco and SK hynix to the investor list |

### Independent Press

| URL | Date | Priority | Notes |
|-----|------|----------|-------|
| https://www.thelec.net/news/articleView.html?idxno=11826 | 2026-06-30 | high | TheElec — SqueezeBits acquired via **all-stock swap**, 100% subsidiary; SqueezeBits KRW 2.646B revenue / KRW 897M OP / KRW 5.3B assets |
| https://www.storagenewsletter.com/2026/07/10/ | 2026-07-10 | low | Storage Newsletter — acquisition coverage, integrated-AI-infrastructure framing |
| https://en.sedaily.com/technology/2026/07/23/rebellions-runs-skts-sovereign-model-ax-k1-on-npu-server | 2026-07-23 | high | Seoul Economic Daily EN — independent confirmation of the A.X K1 single-server run; confirms MoE and the 07-23 date |
| https://en.sedaily.com/technology/2026/04/06/rebellions-gears-up-for-kospi-ipo-as-ai-chip-listing-race | 2026-04-06 | medium | KOSPI IPO prep; pre-IPO round as KRW 640B at KRW 3.4T; National Growth Fund KRW 250B, KDB KRW 50B, Mirae Asset KRW 300B. **Its August-2026 filing target is superseded** — see next row |
| https://www.sedaily.com | 2026-05-27 | medium | Seoul Economic Daily — "상장 속도 늦추는 리벨리온…예심 청구 일정 8월에서 연내로 조정": preliminary-review filing moved **from August to within 2026** |
| https://m.thebell.co.kr | 2026-02 | low | TheBell — "[리벨리온 IPO] 6월 예심청구 가닥": earlier June-filing / KOSDAQ plan; evidence the timing has moved repeatedly |
| https://www.reuters.com/world/asia-pacific/s-koreas-rebellions-targets-ipo-next-year-followed-by-potential-us-listing-ceo-2026-07-08/ | 2026-07-08 | medium | Reuters — H1 2027 Korea listing, possible later US listing |
| https://www.cnbc.com/2026/07/08/rebellions-ipo-south-korea-ai-chips.html | 2026-07-08 | medium | CNBC — CEO leaning KOSPI over KOSDAQ; KRW 3.4T valuation; KRW 640B / $400M round; J.P. Morgan underwriter; KRW 32B 2025 revenue |

### Benchmark / Conference (negative results)

| URL | Notes |
|-----|-------|
| https://mlcommons.org/2026/06/mlperf-training-v6-0-results/ | MLPerf Training v6.0, published 2026-06-16 — 24 submitters, **Rebellions absent** |
| https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/ | MLPerf Inference v6.0, published 2026-04-01 — **Rebellions absent** |
| https://hotchips.org/ | Hot Chips 38 program (Aug 23–25 2026, Stanford) — **still in the future**; Rebellions has no talk. Arm's AGI CPU talk is scheduled but undelivered: *disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public* |

### Corporate events added to the funding table

| Event | Date | Amount | Notes |
|-------|------|--------|-------|
| Pre-IPO (reconciled) | 2026-03-30 | $400M ≈ KRW 640B at ~$2.3B / KRW 3.4T post-money | Same round in two currencies — the KRW and USD figures are **not** separate tranches; the spread is FX drift (~1,450–1,600 KRW/USD). Total funding to date ~$850M. Adds Aramco and SK hynix to the investor list |
| SqueezeBits acquisition | 2026-06-30 | all-stock swap (cash terms not disclosed) | 100%-owned subsidiary |

### Coverage gap

- A UPI item dated 2026-08-07, "SK Telecom, Rebellions expand Korean AI chip infrastructure", appeared in the search index but returned **HTTP 403** on fetch. Developments in the final ~48 hours of this window are unverified. Re-attempt on the next scan.
