# Etched Sohu — Search Results

**Device class:** Transformer-Specific ASIC
**Research date:** 2026-04-05

---

## Layer 1 — Device Overview

Etched Sohu is the world's first transformer-specific ASIC — a chip that hardcodes the transformer computation graph directly in silicon rather than executing it as software on general-purpose compute units. Founded in 2022 by Harvard dropouts Gavin Uberti and Chris Zhu, Etched raised $120M Series A in June 2024 and $500M Series B in January 2026 at a $5B valuation.

Key references:
- https://etched.com (official site)
- https://techcrunch.com/2024/06/25/etched-is-building-an-ai-chip-that-only-runs-transformer-models/
- https://www.tomshardware.com/tech-industry/artificial-intelligence/sohu-ai-chip-claimed-to-run-models-20x-faster-and-cheaper-than-nvidia-h100-gpus
- https://en.wikipedia.org/wiki/Etched_(company)
- https://www.lesswrong.com/posts/qhpB9NjcCHjdNDsMG/new-fast-transformer-inference-asic-sohu-by-etched
- https://www.bloomberg.com/news/articles/2026-01-13/ai-chip-startup-etched-raises-500-million-to-take-on-nvidia

---

## Layer 2 — Chip Specifications

| Parameter | Value |
|---|---|
| Process | TSMC 4nm |
| Die size | Reticle-limit (~800 mm²) — single die |
| Off-chip memory | 144 GB HBM3E per chip |
| Memory comparison | 0.75× B200 capacity; 1.8× H100 capacity |
| FLOPS utilization | Claimed 90%+ |
| Peak compute | Not disclosed (marketed by tok/s) |
| Status | Not shipped as of Q1 2026; in tape-out / samples phase |

---

## Layer 3 — Core Architecture: "Hardcoding Attention"

Rather than building programmable compute units (SIMT, SIMD, systolic arrays) and then writing transformer kernels to run on them, Etched bakes the transformer computation directly into the silicon:

- **Attention engine**: Proprietary hardware block for multi-head attention with QKV projection. Does NOT use a standard GEMM engine for attention — purpose-built circuitry computes `softmax(QKᵀ/√d)V` without standard matrix multiplication primitives.
- **FFN engine**: Dedicated hardware for feed-forward network layers (GELU/SiLU activation + two linear projections).
- **Layer norm**: Hardwired normalization block.
- **Projection layers**: Dedicated hardware for input/output linear projections.
- **MoE variant**: Separate chip variant or configuration for Mixture-of-Experts routing.

The key architectural argument: GPU transformer inference achieves only ~30-40% FLOPS utilization due to instruction fetch/decode, memory management, thread scheduling, and kernel launch overhead. By eliminating all these programmable layers, Sohu targets 90%+ FLOPS utilization — delivering 2-3× more useful compute per transistor.

---

## Layer 4 — Memory Model

- 144 GB HBM3E per chip
- Memory system sized to hold large transformer model weights on-chip
- Etched has argued that inference is NOT primarily memory-bandwidth-limited at larger batch sizes — the compute-to-bandwidth ratio can be tuned with larger batches
- HBM3E at ~9.6 Gb/s per pin → ~1.2 TB/s per stack; multi-stack configuration for 144 GB total

---

## Layer 5 — Scale-Up / System Configuration

- 8-chip Sohu server as reference system
- 8x Sohu → 500,000+ tok/s on Llama 70B
- Etched claims 8x Sohu replaces 160 H100 GPUs
- Networking / interconnect details not publicly disclosed
- Scale-out: not disclosed

---

## Layer 6 — Software Stack

Since Sohu only runs transformers, the software stack is intentionally minimal:
- Transformer-specific compiler: converts transformer models (PyTorch, ONNX) to Sohu binary
- Simple porting: Etched claims "a few lines of code" to port models
- No general-purpose kernel programming interface (no CUDA-equivalent)
- SDK details not publicly released as of Q1 2026

---

## Layer 7 — Performance Claims vs. Competition

| System | Config | Llama 70B tok/s |
|--------|--------|-----------------|
| Etched Sohu | 8× Sohu server | 500,000+ |
| NVIDIA B200 | 8× B200 server | ~45,000 |
| NVIDIA H100 | 8× H100 server | ~23,000 |

Claimed improvement: 20× H100, 10× B200.

**Caveat**: All figures are Etched's own marketing claims as of Q1 2026. No third-party benchmarks exist; chip has not shipped to customers.

---

## Layer 8 — Workload Fit

**Best for:**
- Large-scale transformer inference (LLMs, vision transformers)
- High-throughput token generation (large batch sizes)
- MoE inference (separate variant)

**Weaknesses:**
- Training not supported
- Cannot run non-transformer architectures (CNNs, RNNs, SSMs, etc.)
- If AI architecture shifts away from transformers, new chip takes ~3 years to develop
- Lock-in risk: specialized hardware means zero flexibility outside transformer paradigm

---

## Layer 9 — Precision

Not fully disclosed. Given HBM3E + TSMC 4nm + transformer focus, likely:
- FP16 / BF16 (inference precision)
- Possibly FP8 (for throughput; not confirmed)
- INT8 (likely)

---

## Layer 10 — Status & Roadmap

- June 2024: Announced Sohu chip + $120M Series A
- January 2026: $500M Series B at $5B valuation (led by Stripes; Peter Thiel participated)
- Q1 2026: Not shipped to customers; in late-stage development
- Competitors watching: if transformers remain dominant, niche is large; if alternatives (SSMs/Mamba, diffusion, etc.) grow, bet becomes riskier

---

## Layer 11 — Open Source

- No open-source toolchain, compiler, or SDK disclosed
- Chip design: closed
- Software: not released

---

## Layer 12 — Business Model & Investors

- Funding: $120M Series A (2024) + $500M Series B (Jan 2026) = ~$620M total
- Valuation: $5B
- Investors: Stripes (lead), Peter Thiel (former PayPal CEO), others undisclosed
- Revenue model: chip sales (system-level 8× Sohu server) to hyperscalers and inference providers
- Headquarters: San Jose / Cupertino, CA

---

## Layer 13 — Competing Chips

- NVIDIA B200 (general-purpose GPU, 10× lower Sohu-claimed tok/s at same 8-chip config)
- NVIDIA H100 (general-purpose GPU, 20× lower claimed)
- Groq LPU (also inference-focused, all-SRAM, deterministic; but GEMM-based, not hardcoded attention)
- Cerebras CS-3 (wafer-scale, also inference-capable)
- Tesla Dojo D1 (training-focused, vector ISA)

---

## Key Publications & Sources

- TechCrunch (June 2024): https://techcrunch.com/2024/06/25/etched-is-building-an-ai-chip-that-only-runs-transformer-models/
- Tom's Hardware: https://www.tomshardware.com/tech-industry/artificial-intelligence/sohu-ai-chip-claimed-to-run-models-20x-faster-and-cheaper-than-nvidia-h100-gpus
- Bloomberg (Jan 2026): https://www.bloomberg.com/news/articles/2026-01-13/ai-chip-startup-etched-raises-500-million-to-take-on-nvidia
- LessWrong deep dive: https://www.lesswrong.com/posts/qhpB9NjcCHjdNDsMG/new-fast-transformer-inference-asic-sohu-by-etched
- Wikipedia: https://en.wikipedia.org/wiki/Etched_(company)
- Data Center Dynamics: https://www.datacenterdynamics.com/en/news/etchedai-raises-500m-for-a-5bn-valuation-report/
- Awesome Agents profile: https://awesomeagents.ai/hardware/etched-sohu/

---

# Appendix A — 2026-08-08 Scan: New Resources

**Scan date:** 2026-08-08
**Trigger:** Etched published its first substantive technical update in ~2 years on 2026-06-30, followed by a
funding post on 2026-07-23. Everything above dates from the 2026-04-05 research pass and is retained unchanged.

## A.1 Primary vendor sources (all retrieved 2026-08-08)

| URL | Date | What it establishes | Notes |
|---|---|---|---|
| https://www.etched.com/progress/frontier-inference-clusters | 2026-06-30 | **The single most important new source.** A0 silicon back from TSMC N4P; Low Voltage Inference; Cluster Scale Memory; "first racks ship this summer"; production kicked off; "SOTA throughput, latency, and power efficiency" | Contains no absolute number of any kind |
| https://www.etched.com/progress/accelerating-inference | 2026-07-23 | $300M at $10.3B led by Sequoia (a16z, Jane Street, Diffusion, Argo, SK Hynix); "kicked off fabrication of hundreds of millions of dollars worth of inference clusters" | No occurrence of "Sohu" |
| https://www.etched.com/progress | index | Only two posts exist. Source of "thousand-chip scale-up domains, in-house SMT lines, and new RL environments for recursive kernel generation"; Taiwan factory; San Jose DC/test house/NPI lab; "a new 10-Megawatt lab fifteen minutes from our office" | The word "Sohu" does not appear |
| https://www.etched.com/ | homepage | "We've raised $800M across four unannounced financings, including a strategic investment from VentureTech Alliance"; "a team of 400+ engineers from NVIDIA, Google TPUs, Broadcom, SK Hynix, TSMC" | No "Sohu"; no HBM capacity, tok/s, or FLOPS figures |
| https://www.etched.com/join | leadership/jobs | Full leadership roster incl. Robert Wachen (Co-Founder & President), Mark Ross (CTO, ex-CTO Cypress), Saptadeep Pal (VP ASIC & Architecture, ex-NVIDIA H100/A100/V100) | **Does NOT contain** "thousand-chip scale-up domains", the Taiwan factory, the 10 MW lab, headcount, or "Sohu" — an earlier draft cited this URL incorrectly for those facts |

## A.2 Independent sources (retrieved 2026-08-08)

| URL | Date | What it independently confirms |
|---|---|---|
| https://techcrunch.com/2026/06/30/nvidia-competitor-etched-hits-5b-valuation-1b-in-sales-for-ai-chip/ | 2026-06-30 | TSMC successfully manufactured Etched's chip earlier in 2026; **$800M is the cumulative total raised at that date**; most recent prior round $500M in **December 2025** at $5B post, led by Stripes (Jane Street, Hudson River Trading, Two Sigma, Ribbit Capital); $1B in booked contracts; VentureTech Alliance participated; angels incl. Karpathy, Hinton, Fei-Fei Li, Mensch, Druckenmiller, Thiel. **Does NOT confirm** the node, the A0 designation, the LVI/CSM names, summer rack shipment, the Taiwan factory, or headcount |
| https://techcrunch.com/2026/07/23/ai-chip-startup-etched-defies-skeptics-hits-10-3b-valuation-from-big-name-investors/ | 2026-07-23 | $300M Series C at $10.3B, led by Sequoia with a16z, SK Hynix, Jane Street, Diffusion Capital; paraphrases "Low Voltage Inference" and "Cluster Scale Memory" without numbers; ~$1B booked orders; first full systems in client testing; ~400 employees; 2 MW datacenter at San Jose HQ; new ~80,000 sq ft / 10 MW facility in **Milpitas**. **No mention of Sohu, A0, or N4P** |
| https://www.hotchips.org/advance-program/ | retrieved 2026-08-08 | Hot Chips 2026 runs **Aug 23–25, 2026**; Etched is listed at the **Rhodium (top) sponsor tier**; **no Etched talk appears in the advance program**. Inference-related talks in the program: Intel "Crescent Island", Microsoft "MAIA 200" |

## A.3 Negative results — searched, nothing found

| What was sought | Result |
|---|---|
| Third-party technical analysis of **Low Voltage Inference** | **None found.** Tom's Hardware, EE Times, SemiAnalysis, HPCwire, Data Center Dynamics, ServeTheHome all checked |
| Third-party technical analysis of **Cluster Scale Memory** | **None found** |
| Any absolute spec for the Gen 2 part (voltage, FLOPS, TDP, HBM capacity, bandwidth, die size, chips/rack) | **None disclosed by anyone** |
| Independent confirmation of the **TSMC N4P** node | **None.** Vendor-sourced only |
| Independent evidence that any rack has **shipped** | **None** as of 2026-08-08 |
| Named Etched customer | **None** |
| MLPerf Inference v6.0 submitter list | **Could not be retrieved** — Etched's absence is **unverified**, not a confirmed absence |
| Wikipedia page for Etched | https://en.wikipedia.org/wiki/Etched_(company) returned **404** on 2026-08-08 (it is cited in the 2026-04-05 section above) |
| Yahoo Finance "Etched Emerges From Stealth With Working Chip" | Headline surfaced via Bing SERP; **article body unreachable** (404 on truncated URLs) |

## A.4 Search-method caveats for this scan

- The verification session's **WebSearch budget was exhausted** (200/200 calls). All independent sourcing was
  done via WebFetch against Bing SERPs and directly against publisher URLs.
- **DuckDuckGo returned a CAPTCHA.**
- Independent corroboration for this update therefore rests mainly on the **two TechCrunch articles** in §A.2.
- **Hot Chips 38 (Aug 23–25, 2026) is 15 days in the future** relative to this scan. Its program is public but
  no slides, abstracts, or specs exist. Etched has no talk there in any case.

## A.5 Corrections to the 2026-04-05 record made by this scan

1. **Funding total**: "$620M total" → **~$1.1B**. Critically, the $800M figure on etched.com is a *cumulative*
   total as of 2026-06-30, **not** a new raise. Correct arithmetic is ~$800M + $300M ≈ $1.1B — not $1.42B and
   not $1.72B.
2. **Valuation**: $5B → **$10.3B post-money** (2026-07-23).
3. **Prior round date**: "Series B Jan 2026" → **closed December 2025**, reported January 2026.
4. **Founders**: "Gavin Uberti, Chris Zhu" → **three co-founders**, adding **Robert Wachen** (President).
5. **Status**: "Not shipped as of Q1 2026" → **A0 silicon returned H1 2026; rack-scale product in customer
   validation; production started; still not shipping, no named customer.**
6. **Node**: "TSMC 4nm" → the A0 part is **TSMC N4P per the vendor only**; keep the attribution.
7. **Product name**: "Sohu" is absent from every etched.com page as of 2026-08-08 — but **no source says it was
   retired or renamed**. Record as marketing de-emphasis only.
