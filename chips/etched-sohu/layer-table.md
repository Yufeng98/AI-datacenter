# Etched Sohu Layer Mapping Table

*as_of: 2026-08-08*

*Rows tagged **[Gen 1 / 2024]** describe the Sohu disclosures and are retained as the historical baseline; they
are unverified for the 2026 A0 silicon. Rows tagged **[Gen 2 / 2026]** come from Etched's 2026-06-30 and
2026-07-23 posts and are vendor marketing claims with no absolute numbers and no third-party analysis.*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PyTorch (model export → Etched Transformer Compiler; "few lines of code" port claimed) | inferred | search-results, sw-stack |
| Framework Integration | ONNX (alternative model entry point; likely accepted by compiler) | inferred | sw-stack |
| Framework Integration | TensorFlow / JAX (likely via ONNX export path; not confirmed) | speculative | sw-stack |
| Compiler / IR | Etched Transformer Compiler (validates architecture, packs weights, maps to hardwired pipeline, generates Sohu execution binary) | confirmed-in-principle | search-results, sw-stack |
| Compiler / IR | Architecture validation pass (checks model conforms to supported transformer variant: MHA, GQA, MQA, MoE) | inferred | sw-stack |
| Compiler / IR | Weight packing pass (reshapes + packs weights to match HBM3E layout and pipeline timing) | inferred | sw-stack |
| Compiler / IR | Pipeline mapping pass (maps model layers to hardwired attention / FFN / norm blocks) | inferred | sw-stack |
| Compiler / IR | Sohu execution binary (output; format not disclosed) | inferred | sw-stack |
| Op Library | `not applicable` — no separate op dispatch library; all transformer ops are hardwired silicon blocks, not software-dispatched ops | confirmed | hw-architecture, sw-stack |
| Kernel Library | `not applicable` — no user-facing kernel library; no CUTLASS equivalent; attention and FFN are fixed hardware, not programmable kernels | confirmed | hw-architecture, sw-stack |
| Runtime | Etched Runtime (thin static executor: loads Sohu binary, manages HBM3E weight loading, handles batching and token streaming) | inferred | sw-stack |
| Runtime | No dynamic kernel dispatch, no scheduling overhead — hardware pipeline is fixed | confirmed | hw-architecture |
| Driver / Firmware | PCIe kernel-mode driver (BAR mapping, command queues; not disclosed) | inferred | sw-stack |
| Driver / Firmware | No on-chip firmware runtime (fixed-pipeline ASIC requires no on-chip RM; unlike NVIDIA GSP) | inferred | hw-architecture |
| Communication | Inter-chip communication for 8× server — protocol and topology not disclosed | not-disclosed | search-results |
| Communication | Serving API — likely OpenAI-compatible HTTP endpoint (industry standard; not confirmed) | speculative | sw-stack |
| Assembler / ISA | `not applicable` — no user-programmable ISA; no PTX/HIP equivalent; hardware is the function | confirmed | hw-architecture |
| Compiler / IR | **[Gen 2 / 2026]** "new RL environments for **recursive kernel generation**" named in 2026-07-23 hiring copy — in tension with the "no kernel library" rows above; no technical detail, no tool, no SDK attached | claimed-unverified | etched.com/progress (2026-07-23) |
| Runtime | **[Gen 2 / 2026]** Etched markets co-design of "chips, racks, **software**, and manufacturing methods"; VP of Software (David Munday) named. No runtime, scheduler, or serving component described | not-disclosed | etched.com/progress/frontier-inference-clusters (2026-06-30), etched.com/join |
| Communication | **[Gen 2 / 2026]** Proprietary ultra-low-latency, high-bandwidth scale-up interconnect exposing a **shared memory pool** across the scale-up domain (Cluster Scale Memory). Protocol, API, collectives model, and whether it is software-visible: **not disclosed** | claimed-unverified | etched.com/progress/frontier-inference-clusters (2026-06-30); TechCrunch 2026-07-23 |
| SDK / Toolchain | **[Gen 2 / 2026]** Still no public SDK, compiler, profiler, or documentation as of 2026-08-08 | confirmed-absence | site checked 2026-08-08 |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | Hardwired Attention Engine — QKV projection + softmax(QKᵀ/√d)V + output projection using proprietary non-GEMM hardware | confirmed | hw-architecture, search-results |
| Compute Engine | Hardwired FFN Engine — two linear projections + GELU/SiLU activation, dedicated matrix pipeline | confirmed | hw-architecture, search-results |
| Compute Engine | Hardwired Layer Norm / RMS Norm block | confirmed | hw-architecture, search-results |
| Compute Engine | Hardwired Residual Add ALU | inferred | hw-architecture |
| Compute Engine | Claimed 90%+ FLOPS utilization vs ~30-40% on GPU transformer inference | claimed-unverified | search-results |
| Compute Engine | No warp scheduler, no instruction decoder, no branch predictor, no cache hierarchy | confirmed | hw-architecture, search-results |
| Data Path | Fixed transformer pipeline dataflow — activations flow attention→FFN→norm→next_layer | confirmed | hw-architecture |
| Data Path | No hardware runtime scheduling — pipeline order is fixed in silicon | confirmed | hw-architecture |
| Off-chip Memory | **[Gen 1 / 2024]** 144 GB HBM3E per chip (0.75× B200, 1.8× H100) — 2024 disclosure; **not restated for the A0 silicon and must not be carried forward to Gen 2** | confirmed (Gen 1 only) | search-results |
| Off-chip Memory | **[Gen 1 / 2024]** HBM3E bandwidth ~4-5 TB/s (estimated from HBM3E specs × stack count) — never vendor-confirmed | estimated | search-results |
| Off-chip Memory | **[Gen 1 / 2024]** Sufficient capacity for 70B FP16 model (~140 GB) + KV-cache on single chip | inferred | hw-architecture |
| Host Interface / Package | **[Gen 1 / 2024]** TSMC 4nm, reticle-limit single die (~800 mm²) | confirmed | search-results |
| Host Interface / Package | CoWoS 2.5D packaging for HBM3E integration | inferred | hw-architecture |
| Host Interface / Package | Host interface not disclosed (likely PCIe) | not-disclosed | search-results |
| Scale-up Interconnect | **[Gen 1 / 2024]** 8 chips per server reference system | confirmed | search-results |
| Scale-up Interconnect | Inter-chip protocol and topology not publicly disclosed | not-disclosed | search-results |
| Scale-out Interconnect | Not disclosed | not-disclosed | search-results |
| MoE Variant | Separate Sohu configuration for Mixture-of-Experts routing and sparse expert selection | confirmed | search-results |
| Compute Engine | **[Gen 2 / 2026]** **Low Voltage Inference (LVI)** — math blocks run "at under half the voltage of most AI chips", claimed "multiple times the FLOPs density of AI chips today". **No voltage, FLOPS, perf/W, or TDP figure disclosed** | claimed-unverified | etched.com/progress/frontier-inference-clusters (2026-06-30) |
| Compute Engine | **[Gen 2 / 2026]** "Splittable math arrays" — circuit enabler for LVI. Named only; no microarchitecture disclosed | claimed-unverified | etched.com (2026-06-30) |
| Compute Engine | **[Gen 2 / 2026]** Claim: "trillion-parameter sparse MoEs at 80%+ Peak FLOPs without thermal throttling". Sustained-throughput claim, not the same metric as Gen 1's 90%+ FLOPS utilization; no baseline given | claimed-unverified | etched.com (2026-06-30) |
| Compute Engine | **[Gen 2 / 2026]** Targets **both prefill and decode**; workload target is many-trillion-parameter sparse MoEs, long context, agentic | claimed-unverified | etched.com (2026-06-30) |
| Power Delivery | **[Gen 2 / 2026]** New PDN/VRM architecture required by LVI; cold plates for cooling. Named only; no specifications | claimed-unverified | etched.com (2026-06-30) |
| On-chip Memory | **[Gen 2 / 2026]** **Cluster Scale Memory (CSM)** — "HBM/SRAM hybrid design [that] solves both memory capacity and mem2mem latency". SRAM capacity **not disclosed** | claimed-unverified | etched.com (2026-06-30) |
| Off-chip Memory | **[Gen 2 / 2026]** HBM present in the CSM hybrid; **generation, stack count, per-chip capacity, and bandwidth all not disclosed** | not-disclosed | etched.com (2026-06-30) |
| Scale-up Interconnect | **[Gen 2 / 2026]** Proprietary ultra-low-latency, high-bandwidth interconnect creating "a much lower-latency shared memory pool across our scale-up domain"; targets "SRAM-level decode speeds"; explicitly contrasted with SRAM-only chips, 3D DRAM, and optics. **Protocol, topology, link rate, bandwidth, and latency all not disclosed** | claimed-unverified | etched.com (2026-06-30); TechCrunch 2026-07-23 |
| Scale-up Interconnect | **[Gen 2 / 2026]** "Thousand-chip scale-up domains" — directional vendor signal from hiring copy in the 2026-07-23 `/progress` post, **not a spec**. (Not on `/join`.) | claimed-unverified | etched.com/progress (2026-07-23) |
| System | **[Gen 2 / 2026]** Product unit moved from 8-chip server to **rack-scale "frontier inference cluster"**; chips/rack, rack power, and cooling design not disclosed | claimed-unverified | etched.com (2026-06-30) |
| Host Interface / Package | **[Gen 2 / 2026]** **A0 silicon** returned H1 2026 on **TSMC N4P** — node is **vendor-stated only**; TechCrunch independently confirms only that TSMC manufactured the chip. "Advanced packaging" named without a type. Die size, die count, transistor count, host interface: not disclosed | claimed-unverified (vendor-only) | etched.com (2026-06-30); TechCrunch 2026-06-30 |
| Performance | **[Gen 2 / 2026]** No absolute vendor numbers and no third-party benchmarks. Vendor language is limited to "SOTA throughput, latency, and power efficiency" from "early customer tests". MLPerf Inference v6.0 submitter list unreachable — Etched's absence is **unverified** | not-disclosed | etched.com (2026-06-30); checked 2026-08-08 |
