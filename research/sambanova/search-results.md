# SambaNova — Search Results

*as_of: 2026-04-05*
*chip: sambanova*
*device_class: Dataflow / Reconfigurable*

---

## Summary

SambaNova Systems builds **Reconfigurable Dataflow Architecture (RDA)** chips — the RDU (Reconfigurable Dataflow Unit) family. The current production generation is the **SN40L** (4th gen, TSMC 5 nm); the **SN50** (5th gen, TSMC 3 nm) ships H2 2026. The core insight is spatial: the SambaFlow compiler maps an entire ML model graph onto PCU/PMU arrays at compile time, so data flows through the chip without scheduling overhead at runtime. This is fundamentally different from GPU SIMT or TPU systolic execution.

---

## Resources

### Documentation

| # | Title | URL | Category |
|---|-------|-----|----------|
| 1 | SambaNova Developer Docs (SambaFlow) | https://docs.sambanova.ai/developer/latest/ | SDK / Compiler |
| 2 | Model Conversion Overview | https://docs.sambanova.ai/developer/latest/porting-overview.html | SDK / PyTorch Integration |
| 3 | Compiler Optimization Modes (o0/o1) | https://docs.sambanova.ai/developer/latest/compiler-o1.html | Compiler |
| 4 | SambaFlow API Reference | https://docs.sambanova.ai/api-reference/ | SDK |
| 5 | SambaNova Glossary | https://docs.sambanova.ai/resources/latest/glossary.html | Reference |
| 6 | Compilation, Training, and Inference (LeNet end-to-end) | https://docs.sambanova.ai/developer/latest/lenet-end2end.html | Tutorial |
| 7 | Compile/Fine-tune/Inference with Hugging Face GPT | https://docs.sambanova.ai/developer/latest/hf-compile-run.html | Tutorial |
| 8 | Code Elements of the Inference Program | https://docs.sambanova.ai/developer/latest/hf-model-inference.html | SDK Reference |
| 9 | Data Parallel Mode | https://docs.sambanova.ai/developer/latest/data-parallel.html | Parallelism |
| 10 | SambaFlow Release Notes (legacy) | https://docs-legacy.sambanova.ai/developer/latest/release-notes.html | Changelog |

### Whitepapers and Technical Papers

| # | Title | URL | Category |
|---|-------|-----|----------|
| 11 | Accelerated Computing with a Reconfigurable Dataflow Architecture (SambaNova Whitepaper) | https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf | Architecture Whitepaper |
| 12 | SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts (arXiv 2405.07518) | https://arxiv.org/html/2405.07518v1 | Architecture Paper (Hot Chips 2024) |
| 13 | Evaluating Emerging AI/ML Accelerators: IPU, RDU, and NVIDIA/AMD GPUs (arXiv 2311.04417) | https://arxiv.org/html/2311.04417v3 | Comparative Analysis |
| 14 | SSM-RDU: A Reconfigurable Dataflow Unit for Long-Sequence State-Space Models (arXiv 2503.22937) | https://arxiv.org/html/2503.22937 | Research Paper |
| 15 | Composition of Experts on the SN40L RDU (IEEE Xplore) | https://ieeexplore.ieee.org/document/10609347/ | Research Paper |
| 16 | Accelerating Scientific Applications with SambaNova RDA (OSTI) | https://www.osti.gov/servlets/purl/1798044 | HPC Paper |

### Product Pages

| # | Title | URL | Category |
|---|-------|-----|----------|
| 17 | SN40L RDU Product Page | https://sambanova.ai/products/rdu-ai-chips | Product |
| 18 | Dataflow Architecture Overview | https://sambanova.ai/products/dataflow-architecture | Architecture |
| 19 | SambaRack (SN40L-16) Product Page | https://sambanova.ai/products/sambarack | System |
| 20 | SambaRack SN40L-16 Datasheet (PDF) | https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf | Datasheet |

### Announcements and News

| # | Title | URL | Category |
|---|-------|-----|----------|
| 21 | Introducing the SN50 RDU: Purpose-Built for Agentic Inference | https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference | Product Announcement |
| 22 | SambaNova Unveils SN50, $350M Funding, Intel Partnership (Press) | https://sambanova.ai/press/sambanova-unveils-fastest-chip-for-agentic-ai-collaborates-with-intel-and-raises-350m | Press Release |
| 23 | SambaNova SN50 Announcement — Developer Community | https://community.sambanova.ai/t/introducing-the-sn50-rdu-purpose-built-for-agentic-inference/1579 | Community |
| 24 | SambaNova Pits Its Engineering Against Nvidia For Agentic AI (NextPlatform) | https://www.nextplatform.com/ai/2026/02/25/sambanova-pits-its-engineering-against-nvidia-for-agentic-ai/4092613 | Analysis |
| 25 | SambaNova Raises $350M with Intel Backing (The Register) | https://www.theregister.com/2026/02/24/sambanova_intel_funding/ | News |
| 26 | SambaNova's Second Life — Intel Collaboration (Implicator) | https://www.implicator.ai/sambanovas-second-life-a-350-million-bet-that-intel-cant-build-alone/ | Analysis |
| 27 | SambaNova Releases Fourth-Gen Chip (TechInsights) | https://www.techinsights.com/blog/sambanova-releases-fourth-gen-chip | News |
| 28 | SambaNova's New Chip Means GPTs for Everyone (IEEE Spectrum) | https://spectrum.ieee.org/ai-chip-sambanova | Feature |

### Blog Posts and Technical Overviews

| # | Title | URL | Category |
|---|-------|-----|----------|
| 29 | Why SambaNova's SN40L Is the Best for Inference | https://sambanova.ai/blog/sn40l-chip-best-inference-solution | Blog |
| 30 | Accelerating Scientific Applications With SambaNova RDA | https://sambanova.ai/blog/accelerating-scientific-applications-with-sambanova-reconfigurable-dataflow-architecture | Blog / HPC |
| 31 | SambaNova RDU: Reconfigurable Architectures for Inference, Training, and Agentic AI (Medium) | https://medium.com/@leosorge/sambanova-rdu-reconfigurable-architectures-for-inferencefor-inference-training-and-agentic-ai-5088b5ca400b | Community Overview |
| 32 | Breaking Through the AI Memory Wall — CoE Paper Review (Medium) | https://medium.com/byte-sized-ai/paper-review-sambanova-sn40l-scaling-the-ai-memory-wall-with-dataflow-and-composition-of-experts-7592948c7087 | Paper Review |
| 33 | SambaNova vs Cerebras — Inference Comparison | https://sambanova.ai/blog/sambanova-vs-cerebras | Benchmark |
| 34 | SambaNova vs Groq — AI Inference Face-Off | https://sambanova.ai/blog/sambanova-vs-groq | Benchmark |

---

## Key Findings

### 1. Reconfigurable Dataflow Architecture (RDA) — Spatial Compute

The RDU does not use a von Neumann or SIMT execution model. Instead:

- The SambaFlow compiler **statically maps the entire model graph to physical PCU/PMU arrays** at compile time
- Each PCU is assigned a specific computation; data flows between units on a 3D switching fabric
- There is no runtime scheduler, no kernel launch overhead, no thread warps
- Output from one PCU flows directly to the next without intermediate memory round-trips

This is the defining architectural difference from NVIDIA GPUs and Google TPUs.

### 2. Core Hardware Units (PCU / PMU / AGCU)

- **PCU (Pattern Compute Unit)**: A multi-stage reconfigurable SIMD pipeline for innermost-parallel operations. 1040 PCUs per SN40L socket (638 BF16 TFLOPS aggregate)
- **PMU (Pattern Memory Unit)**: Distributed scratchpad SRAM. 1040 PMUs per SN40L; aggregate hundreds of TB/s on-chip BW. The PMU is *not* a cache — it is software-managed, like a GPU's shared memory but at much larger scale
- **AGCU (Address Generation and Coalescing Unit)**: Interface between RDU and off-chip memory (DDR, HBM) and other RDUs. Handles sparse and graph-based data access patterns
- **Switching Fabric**: 3D on-chip network with separate scalar (word-level), vector (multi-word), and control (bit-level) networks connecting PCUs and PMUs

### 3. Three-Tier Memory Hierarchy (SN40L)

| Tier | Capacity | Bandwidth | Notes |
|------|----------|-----------|-------|
| On-chip PMU SRAM | 520 MiB | Hundreds of TB/s | Distributed across 1040 PMUs |
| Co-packaged HBM | 64 GiB | ~1 TB/s (node-level) | Software-managed caching tier |
| DDR DRAM (pluggable DIMMs) | up to 1.5 TiB | — | Model weights reservoir |

SN40L has a **dual-die package** (two logic dies + HBM + DDR on-package). The DDR is directly attached to the accelerator, not the host CPU.

SN50 memory:

| Tier | Capacity | Bandwidth |
|------|----------|-----------|
| On-chip SRAM | 432 MiB | — |
| HBM2E | 64 GiB | 1.8 TB/s |
| DDR5 | 256 GiB – 2 TiB | — |

### 4. SambaFlow Compiler

- Takes a **PyTorch model** (or TF), applies `samba.from_torch_model()` to trace the computation graph
- Performs **graph-level fusion**: operator fusion into "sections" (execution units)
  - `o0` mode: one section per operator (safe, debuggable)
  - `o1` mode: maximum operator fusion (production)
- Applies **meta-pipelining**, multi-section support, and tensor/pipeline/data parallelism mapping
- Outputs a **PEF (Program Execution File)**: the binary that encodes the spatial placement and routing of the entire model on the RDU hardware
- Runtime executes via `samba.session.run()` specifying which sections to run — no dynamic scheduling

### 5. PyTorch Integration

- User defines model in standard PyTorch (`torch.nn.Module`)
- Calls `samba.from_torch_model()` to convert: `torch.Tensor` → `SambaTensor`, `nn.Parameter` → `SambaParameter`
- Calls `samba.session.compile(model, dummy_inputs)` to generate PEF
- Training loop calls `samba.session.run()` — replaces `.backward()` and optimizer step
- Most model code is **unchanged**; the conversion is at the session layer

### 6. Scale-up Interconnect

- **SN40L**: 16 RDU chips in a SambaRack, peer-to-peer (P2P) protocol, InfiniBand for rack-to-rack
- **SN50**: 256 RDUs max, **2.2 TB/s bidirectional chip-to-chip bandwidth** via a switched fabric; 4x more network BW than SN40L
- Collective communication primitives (AllReduce etc.) are implemented on top of the P2P substrate

### 7. Chip Specifications Summary

| Spec | SN40L (Gen 4) | SN50 (Gen 5) |
|------|--------------|-------------|
| Process | TSMC 5 nm | TSMC 3 nm (N3) |
| Transistors | ~102 B | — |
| Package | Dual-die + HBM + DDR | Dual-chiplet |
| PCU count | 1040 | ~2x SN40L (est.) |
| PMU count | 1040 | — |
| BF16 TFLOPS (socket) | 638 | 1,600 |
| FP8 PFLOPS (socket) | — | 3.2 |
| On-chip SRAM | 520 MiB | 432 MiB |
| HBM | 64 GiB | 64 GiB @ 1.8 TB/s |
| DDR | up to 1.5 TiB | 256 GiB – 2 TiB DDR5 |
| Max scale-up | 16 RDUs (SambaRack) | 256 RDUs |
| Chip-to-chip BW | — | 2.2 TB/s bidir |
| Max model size | — | 10T parameters |
| Max context | — | 10M tokens |
| Ship date | 2023 | H2 2026 |

### 8. Composition of Experts (CoE) — Key Use Case

SambaNova's flagship workload: **Samba-CoE** = 150 × 7B expert models (1T total params). The three-tier memory enables:
- Model weights live in DDR; loaded to HBM as needed
- Active expert weights stream from HBM → on-chip SRAM → compute
- The dataflow compiler fuses hundreds of operations into a single spatial mapping, avoiding intermediate memory round-trips

Speedups of **2x–13x** demonstrated on various benchmarks vs. unfused baselines.

---

## Sources

- [SN40L RDU Product Page](https://sambanova.ai/products/rdu-ai-chips)
- [SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts](https://arxiv.org/html/2405.07518v1)
- [Introducing the SN50 RDU: Purpose-Built for Agentic Inference](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference)
- [Dataflow Architecture Overview](https://sambanova.ai/products/dataflow-architecture)
- [Accelerated Computing with a Reconfigurable Dataflow Architecture (Whitepaper)](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [Compiler Optimization Modes](https://docs.sambanova.ai/developer/latest/compiler-o1.html)
- [Model Conversion Overview](https://docs.sambanova.ai/developer/latest/porting-overview.html)
- [SambaFlow API Reference](https://docs.sambanova.ai/api-reference/)
- [SambaNova Glossary](https://docs.sambanova.ai/resources/latest/glossary.html)
- [Evaluating Emerging AI/ML Accelerators: IPU, RDU, and NVIDIA/AMD GPUs](https://arxiv.org/html/2311.04417v3)
- [SambaNova Pits Its Engineering Against Nvidia For Agentic AI](https://www.nextplatform.com/ai/2026/02/25/sambanova-pits-its-engineering-against-nvidia-for-agentic-ai/4092613)
- [SambaRack Datasheet](https://sambanova.ai/hubfs/SambaRack%20data%20sheet%20template%2007%2009%2025.pdf)
- [SambaNova SN50 — Awesome Agents](https://awesomeagents.ai/hardware/sambanova-sn50/)
- [Accelerating Scientific Applications with SambaNova RDA](https://sambanova.ai/blog/accelerating-scientific-applications-with-sambanova-reconfigurable-dataflow-architecture)

---

# Resources Added — 2026-08-08 (scan window 2026-04-05 → 2026-08-08)

> **Link-rot notice.** Documentation resources **#1–#10** in the table above (all `docs.sambanova.ai/developer/latest/*`, `/api-reference/`, `/resources/latest/glossary.html`) are **dead as of 2026-08-08** — verified HTTP 404 by direct request. The SambaFlow developer documentation tree has been withdrawn and no replacement is published. Use the current documentation resources below instead.

## Documentation — current

| # | Title | URL | Category | Verified |
|---|-------|-----|----------|----------|
| D1 | SambaNova documentation root (SambaCloud + SambaStack) | https://docs.sambanova.ai/ | Docs index | 2026-08-08 |
| D2 | Full documentation index (`llms.txt`, 174 entries) | https://docs.sambanova.ai/docs/llms.txt | Docs index — proves no SambaFlow/compiler/PEF-authoring pages remain | 2026-08-08 |
| D3 | SambaStack release notes (v0.4.8 2026-03-10 → v2.0.2 2026-08-05) | https://docs.sambanova.ai/docs/en/release-notes/sambastack.md | Changelog — authoritative version history; `Pef` CRD; K8s 1.30→1.35; no vLLM mention | 2026-08-08 |

## Press releases (the 2026-04-05 baseline scan covered only /blog and missed these)

| # | Title | URL | Date | Why it matters |
|---|-------|-----|------|----------------|
| P1 | SambaNova press index | https://sambanova.ai/press | — | The blog alone is not sufficient coverage for this vendor |
| P2 | SambaNova Completes First Close of $1B Financing at $11B Valuation | https://sambanova.ai/press/sambanova-completes-first-close-of-1b-financing-at-11b-valuation | 2026-07-08 | Series F first close, $1B at $11B post-money, led by General Atlantic; names JPMorganChase as intending to deploy SN40 and SN50 |
| P3 | SambaNova and Intel Announce Blueprint for Heterogeneous Inference | https://sambanova.ai/press/sambanova-announces-collaboration-with-intel-on-ai-solution | 2026-04-08 | GPU prefill / RDU decode / Xeon 6 agentic tools; H2 2026 availability |
| P4 | SambaNova Unveils Fastest Chip for Agentic AI, Collaborates with Intel and Raises $350M | https://sambanova.ai/press/sambanova-unveils-fastest-chip-for-agentic-ai-collaborates-with-intel-and-raises-350m | 2026-02-24 | Series E was led by Vista Equity Partners and Cambium Capital with **Intel Capital participating**; "SN50 will start shipping to customers later this year" |

## Vendor blog (update window)

| # | Title | URL | Date | Notes |
|---|-------|-----|------|-------|
| B1 | The Decode Era of AI: Why Dataflow Matters More Than Ever | https://sambanova.ai/blog/why-dataflow-matters-more-than-ever | 2026-04-16 | Strategy narrative; maps SRAM/HBM/DDR onto decode-phase roles |
| B2 | SN50 Runs Fastest MiniMax Speeds in the World | https://sambanova.ai/blog/sn50-runs-fastest-minimax-speeds-in-the-world | 2026-07-08 | RAISE Summit **preview/demonstration**: H200 prefill + 16-RDU SambaRack SN50 decode; up to 850 t/s |
| B3 | SemiAnalysis Benchmarks SambaRack SN50 with Fast Inference on MiniMax M2.7 | https://sambanova.ai/blog/semianalysis-benchmarks-sambarack-sn50-with-fast-inference-on-minimax-m2.7 | 2026-07-30 | **Vendor-reported, not independent** — see negative-evidence sources N1/N2. Source for TP16 ≈800 t/s vs TP8+DP8 ≈400 t/s |

## Open source

| # | Repo | URL | Notes |
|---|------|-----|-------|
| O1 | sambanova/sambastack-tools | https://github.com/sambanova/sambastack-tools | TypeScript; **created 2026-01-15** (pre-baseline — new to this survey, not new in the window); pushed 2026-08-07; 2 stars. Contains **SambaEval** and **SambaWiz** |
| O2 | sambastack-tools README | https://raw.githubusercontent.com/sambanova/sambastack-tools/main/README.md | Verbatim SambaEval / SambaWiz descriptions; confirms PEF settings + Kubernetes manifest generation |
| O3 | SambaNova GitHub org listing | https://api.github.com/orgs/sambanova/repos?sort=pushed | Confirms **no** SambaFlow, model-zoo, or vLLM repository exists publicly. Also: `sambanova-inference-api-spec` (2026-07-21), `sambanova-plugin-cc` (2026-03-31) |

## Third-party measurement

| # | Resource | URL | Notes |
|---|----------|-----|-------|
| T1 | Artificial Analysis — MiniMax-M2.7 provider comparison | https://artificialanalysis.ai/models/minimax-m2-7/providers | **Independent production datapoint:** SambaNova 392.4 output t/s, rank #1 of 5 (Fireworks 262.1, MiniMax 61.7, Novita FP8 59.3). Less than half the 800–850 t/s demo figure |

## Negative evidence (recorded so a future scan does not re-derive it)

| # | Resource | URL | Finding |
|---|----------|-----|---------|
| N1 | SemiAnalysis newsletter archive | https://newsletter.semianalysis.com/archive?sort=new | Jun–Aug 2026 contains **no** SambaNova / SN50 / MiniMax-benchmark article |
| N2 | SemiAnalysis InferenceX dashboard | https://inferencex.semianalysis.com/ (301 from inferencemax.semianalysis.com) | Lists NVIDIA, AMD and TPU v7 Ironwood; **no SambaNova hardware**. SambaNova's CEO is quoted looking forward to "participating in the official InferenceX with SN50" |
| N3 | Direct HTTP checks on the five previously cited SambaFlow doc URLs | (curl -L, 2026-08-08) | All **HTTP 404** |

## Scheduled, not yet public

| # | Item | URL | Status |
|---|------|-----|--------|
| S1 | Hot Chips 38 advance program — Session AI 2, Tue 2026-08-25, "Dataflow at Scale: the SN50 RDU", Raghu Prabhakar (SambaNova) | https://hotchips.org/advance-program/ | **Disclosure scheduled, Hot Chips 38, Aug 2026 — content not yet public.** Conference runs 2026-08-23 to 08-25. No slides, abstract, or specification exists. **Must not be cited as a spec source.** Re-scan early September 2026 |
