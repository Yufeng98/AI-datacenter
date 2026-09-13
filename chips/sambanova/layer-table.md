# SambaNova RDU Layer Mapping Table

*as_of: 2026-08-08* (base research 2026-04-05)
*chip: sambanova*
*device_class: Dataflow / Reconfigurable*

> **2026-08-08:** rows describing the SambaFlow SDK are now marked **historical (SN40L-era)** — the public SambaFlow documentation tree was withdrawn (all `docs.sambanova.ai/developer/latest/*`, `/api-reference/`, `/resources/latest/glossary.html` return HTTP 404) and no replacement is published. Those rows are retained because they document the SN40L-generation programming model, but their `porting-overview` / `compiler-o1` / `api-reference` sources are dead links; the surviving evidence is arXiv 2405.07518 and the SN40L Hot Chips paper. New rows for SambaStack v2.0.2 (the currently documented deployment path) are appended below. SN50 hardware rows are downgraded to **extrapolated** where re-verification could not find a primary source.

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (torch.nn.Module → SambaTensor/SambaParameter via samba.from_torch_model) — *historical, SN40L-era* | confirmed (historical; docs withdrawn 2026) | software-stack, ~~porting-overview~~ (404), sn40l-paper |
| Framework Integration | TensorFlow (secondary support) — *historical, SN40L-era* | confirmed (historical) | software-stack, rda-whitepaper |
| Framework Integration | Hugging Face Transformers (GPT-2, GPT-J, Llama, BERT; compile via samba.session.compile) — *historical, SN40L-era* | confirmed (historical; docs withdrawn 2026) | software-stack, ~~hf-compile-run~~ (404) |
| Framework Integration | samba.optim (Adam, SGD, AdamW wrappers replacing torch.optim) — *historical, SN40L-era* | confirmed (historical; docs withdrawn 2026) | software-stack, ~~api-reference~~ (404) |
| Compiler / IR | SambaFlow Dataflow Compiler (whole-graph AOT spatial compiler; input: PyTorch graph; output: PEF) | confirmed | software-stack, rda-whitepaper, sn40l-paper |
| Compiler / IR | Graph tracing (samba.from_torch_model traces computation graph; overrides torch ops with SambaFlow implementations) | confirmed | software-stack, porting-overview |
| Compiler / IR | Operator fusion / Sections (o0: per-op sections; o1: max fusion; 2x-13x speedup) | confirmed | software-stack, compiler-o1 |
| Compiler / IR | Meta-pipelining (pipeline parallelism across sections; next section pre-staged in HBM while current executes) | confirmed | software-stack, rda-whitepaper |
| Compiler / IR | Parallelism mapping (data / tensor / pipeline parallelism compiled into PEF; user writes no sharding code) | confirmed | software-stack, sn40l-paper |
| Compiler / IR | Parallelism *strategy* is a deployment-time choice, not compiler-hidden (SN50 16-RDU rack: TP16 ≈800 t/s vs TP8+DP8 ≈400 t/s on identical hardware; SambaStack ships a "high-throughput vs. high-interactivity" configuration page) | confirmed | sn50-semianalysis-post-2026-07-30, sambastack-docs |
| Compiler / IR | PCU/PMU placement (compiler assigns each operator to specific PCU; each tensor to specific PMU or memory tier) | confirmed | software-stack, rda-whitepaper |
| Compiler / IR | Switching fabric routing (compiler statically routes all PCU-to-PCU / PCU-to-PMU data paths) | confirmed | software-stack, rda-whitepaper |
| Compiler / IR | PEF (Program Execution File: binary encoding complete hardware spatial configuration; compiled once, reused indefinitely) | confirmed | software-stack, lenet-end2end, compiler-o1 |
| Runtime | SambaNova Runtime / samba.session (compile, load, run; synchronous from host) | confirmed | software-stack, api-reference, lenet-end2end |
| Runtime | samba.session.run() (executes sections; replaces forward + backward + optimizer on RDU) | confirmed | software-stack, hf-model-inference |
| Runtime | Built-in collective communication (AllReduce, AllGather compiled into PEF; no separate comm library) | confirmed | software-stack, sn40l-paper |
| Runtime | Data parallel execution (samba data-parallel mode: model replicated across sockets; batch split) | confirmed | software-stack, data-parallel |
| Driver / Firmware | SambaNova system driver (loads PEF to RDU; manages PCIe/direct-attach host interface) | inferred | software-stack, rda-whitepaper |
| Driver / Firmware | RDU-Connect (P2P inter-chip protocol for SN40L 16-socket SambaRack) | confirmed | software-stack, sn40l-paper |

### Added 2026-08-08 — SambaStack / SambaCloud era

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | SambaCloud inference API: OpenAI-compatible endpoints, **Anthropic Messages API compatibility** (2026-07-01), **prompt caching** (2026-07-16), responses API, vision/audio/video | confirmed | sambanova-docs-index-2026-08 |
| Framework Integration | **vLLM as the reported SN50 serving engine** — both 2026 SN50 demos state results were "built and measured on vLLM" | reported-only / depth unverified | sn50-raise-post-2026-07-08, sn50-semianalysis-post-2026-07-30 |
| Framework Integration | No public SambaNova vLLM hardware plugin exists; vLLM appears nowhere in SambaStack v2.0.2 docs or in the full SambaStack release-note history (Sept 2025 – Aug 2026) | confirmed (negative) | sambanova-github-org, sambastack-release-notes, sambanova-docs-index-2026-08 |
| Compiler / IR | **SambaFlow SDK documentation withdrawn** — all five previously cited developer-docs URLs return HTTP 404; 174-entry docs index contains no SambaFlow / compiler / PEF-authoring / samba.session pages | confirmed | direct-http-check-2026-08-08, sambanova-docs-index-2026-08 |
| Compiler / IR | Whether the SambaFlow compiler or the AOT/PEF model changed for SN50 | **unknown** — no source | — |
| Runtime / Deployment | **SambaStack** on-prem platform: Kubernetes (RKE2) based; four-CRD resource model `Model` / `ModelProfile` / `ModelDeployment` / `ModelBundle`, plus a **`Pef` custom resource** | confirmed | sambastack-release-notes, sambanova-docs-index-2026-08 |
| Runtime / Deployment | SambaStack v2.0.2 (2026-08-05, **breaking**): replaces `BundleTemplate`/`Bundle`/`BundleDeployment` (old format works until 2026-09-30); RKE2 1.30 → 1.35; models split into standalone `sambastack-models` Helm chart (install order `sambastack-base` → `sambastack` → `sambastack-models`); structured output enforced for constrained-decoding models | confirmed | sambastack-release-notes |
| Runtime / Deployment | SambaStack version history: v0.4.8 (2026-03-10), v0.5.14 (2026-04-01), v0.5.17 (2026-04-08), v1.0.57 (2026-04-30), v1.1.1 (2026-05-27), v1.2.0 (2026-07-07), v2.0.2 (2026-08-05) | confirmed | sambastack-release-notes |
| Runtime / Deployment | Reference architecture: **LiteLLM** gateway; **Prometheus / Grafana / OpenSearch / Fluent Bit** observability | confirmed | sambanova-docs-index-2026-08 |
| Runtime / Deployment | **PEF survives** as the deployment artifact; the surface around it is now Helm + Kubernetes CRDs rather than a Python SDK | confirmed | sambastack-release-notes, sambastack-tools-readme |
| Tooling | **SambaWiz** (in `sambanova/sambastack-tools`): GUI wizard for selecting models, configuring PEF settings, generating Kubernetes manifests, deploying bundles. v2.0.0 (2026-07-31), v2.1.0 (2026-08-06), v2.1.1 (2026-08-07); release notes instruct operators to pull the matching version per SambaStack release | confirmed | sambastack-tools-readme, sambastack-release-notes |
| Tooling | **SambaEval** (in `sambanova/sambastack-tools`): local workbench evaluating LLMs against CSV datasets across models/providers; heuristic + LLM-as-judge scoring; per-row token/latency metrics; pluggable Python output generators | confirmed | sambastack-tools-readme |
| Driver / Firmware | SambaStack v2.0.2 hardware administration: "RDU module administration" pages and **SambaRack Manager** command reference | confirmed | sambanova-docs-index-2026-08 |
| Driver / Firmware | On-prem hardware reference documents remain **SN40L-16**; no SN50 SambaRack documentation is published | confirmed (negative) | sambanova-docs-index-2026-08 |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | PCU (Pattern Compute Unit): multi-stage reconfigurable SIMD pipeline; 1040 per SN40L socket; 638 BF16 TFLOPS | confirmed | hw-architecture, sn40l-paper, rda-whitepaper |
| Compute Engine | SN40L: TSMC 5 nm, dual-die, ~102B transistors, 638 BF16 TFLOPS/socket | confirmed | hw-architecture, sn40l-paper |
| Compute Engine | SN50: TSMC 3 nm dual-chiplet (vendor-stated) | confirmed | hw-architecture, sn50-blog |
| Compute Engine | SN50: 1.6 BF16 PFLOPS / 3.2 FP8 PFLOPS per socket, native FP8, ~2× SN40L PCU count | **extrapolated** — absent from the 2026-02-24 press release and the SN50 introduction blog; vendor states only "5X faster" | hw-architecture *(unsourced; re-check after Hot Chips 38)* |
| Compute Engine | SN50 status: announced 2026-02-24, GA targeted H2 2026, **no customer delivery confirmed as of 2026-08-08**; JPMorganChase stated intent to deploy SN40 and SN50 | confirmed | sn50-press-2026-02-24, intel-blueprint-2026-04-08, series-f-press-2026-07-08 |
| Compute Engine | SambaRack SN50: **16 RDUs demonstrated and benchmarked** (2026-07); 20 kW per rack (vendor-stated) | confirmed | sn50-raise-post-2026-07-08, sn50-semianalysis-post-2026-07-30 |
| Compute Engine | SambaRack SN40L-16: 16 RDU sockets, 10.2 BF16 PFLOPS aggregate | confirmed | hw-architecture, sambarack-datasheet |
| Data Path | Spatial dataflow execution (PCUs execute statically assigned ops; data flows on switching fabric without DRAM round-trips) | confirmed | hw-architecture, rda-whitepaper |
| Data Path | Reconfigurable SIMD pipeline within each PCU (multi-stage; lane parallelism + pipeline parallelism) | confirmed | hw-architecture, rda-whitepaper |
| Data Path | AGCU (Address Generation and Coalescing Unit): RDU ↔ off-chip memory interface; handles sparse access; P2P inter-RDU | confirmed | hw-architecture, rda-whitepaper |
| On-chip Memory | PMU (Pattern Memory Unit): distributed software-managed scratchpad SRAM; 1040 per SN40L; 520 MiB total | confirmed | hw-architecture, sn40l-paper |
| On-chip Memory | PMU bandwidth: hundreds of TB/s aggregate; high bank-level parallelism within and across PMUs | confirmed | hw-architecture, sn40l-paper |
| On-chip Memory | Switching fabric: 3D on-chip network with scalar (word), vector (multi-word), and control (bit) networks | confirmed | hw-architecture, rda-whitepaper |
| Off-chip Memory | Co-packaged HBM: 64 GiB (SN40L), software-managed caching tier | confirmed | hw-architecture, sn40l-paper |
| Off-chip Memory | SN50 HBM: 64 GiB HBM2E @ 1.8 TB/s | **extrapolated** — no primary source | hw-architecture *(unsourced)* |
| Off-chip Memory | Directly-attached DDR DRAM: up to 1.5 TiB DDR4/5 (SN40L); not host CPU DDR | confirmed | hw-architecture, sn40l-paper |
| Off-chip Memory | SN50 DDR: 256 GiB – 2 TiB DDR5; SN50 SRAM 432 MiB | **extrapolated** — no primary source | hw-architecture *(unsourced)* |
| Off-chip Memory | Decode-era tier roles (vendor framing 2026-04-16): SRAM = hottest local working set; HBM = weights + KV cache; DDR = prompt caching and multi-model/agentic workflows | confirmed | decode-era-post-2026-04-16 |
| Off-chip Memory | DDR → HBM transfer: >1 TB/s per SN40L Node (16 sockets) | confirmed | hw-architecture, sn40l-paper |
| Host Interface / Package | PCIe host interface (host CPU to RDU control plane) | inferred | hw-architecture |
| Host Interface / Package | SN40L package: dual logic die + HBM stacks + DDR DIMMs on-board | confirmed | hw-architecture, sn40l-paper |
| Host Interface / Package | SN50 package: dual-chiplet, TSMC 3 nm | confirmed | hw-architecture, sn50-blog |
| Scale-up Interconnect | SN40L RDU-Connect: P2P protocol; 16 RDU sockets per SambaRack | confirmed | hw-architecture, sn40l-paper |
| Scale-up Interconnect | SN50 switched fabric, 4x network BW vs SN40L, "multi-terabyte-per-second interconnect" | confirmed (vendor wording) | sn50-press-2026-02-24, sn50-blog |
| Scale-up Interconnect | SN50 chip-to-chip 2.2 TB/s bidirectional | **extrapolated** — vendor says only "multi-TB/s" | hw-architecture *(unsourced)* |
| Scale-up Interconnect | SN50 scale: **16 RDUs demonstrated**; 256 is a vendor claim — SambaNova's own 2026-07-30 post frames 64- and 256-chip configurations as future | confirmed | sn50-semianalysis-post-2026-07-30 |
| Scale-out Interconnect | InfiniBand (SN40L rack-to-rack inter-connect) | confirmed | hw-architecture, sn40l-paper |
| Scale-up Interconnect | Max model scale: SN50 claimed to support 10T+ parameters and 10M+ token context | vendor claim | hw-architecture, sn50-blog |

### Added 2026-08-08 — system-level

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| System Architecture | **Disaggregated prefill/decode blueprint** (announced 2026-04-08 with Intel): GPUs = prefill; SN50 RDUs = "dedicated inference fabric" for decode; Intel Xeon 6 = host and "action CPU" for agentic tool/API execution. Availability H2 2026 | confirmed | intel-blueprint-2026-04-08 |
| System Architecture | Demonstrated instance (2026-07-08): 1 NVIDIA H200 rack (4 GPUs, prefill) + 1 SambaRack SN50 (16 RDUs, decode) on MiniMax M2.7 | confirmed | sn50-raise-post-2026-07-08 |
| Performance | 2026-07-08 preview: up to 850 t/s short-context, >450 t/s long-context (MiniMax M2.7) | vendor demo, attributed to Artificial Analysis | sn50-raise-post-2026-07-08 |
| Performance | 2026-07-30: TP16 ≈800 t/s, TP8+DP8 ≈400 t/s on one 16-chip SambaRack SN50 | **vendor-reported from a pre-official run** — SemiAnalysis published nothing on SambaNova through its own channels and lists no SambaNova hardware in InferenceX | sn50-semianalysis-post-2026-07-30, semianalysis-archive, inferencex-dashboard |
| Performance | Public SambaCloud endpoint, MiniMax-M2.7: **392.4 output t/s**, rank #1 of 5 providers | confirmed (independent) | artificial-analysis-minimax-m2-7-providers |
| Future disclosure | Hot Chips 38, Session AI 2, 2026-08-25: "Dataflow at Scale: the SN50 RDU" (Raghu Prabhakar) — **disclosure scheduled, content not yet public**; not a spec source. Re-scan early Sept 2026 | scheduled | hotchips-38-advance-program |
