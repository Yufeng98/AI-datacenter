# d-Matrix Corsair — Software Stack Investigation

*chip: d-matrix*
*investigation: software-stack*
*date: 2026-04-05*
*sources: d-Matrix product page, white paper, Hot Chips 2025 coverage, vendor announcements*

---

## 1. Overview

d-Matrix's software stack is called **Aviator**. It is a closed-source, proprietary stack that integrates with PyTorch and Triton DSL as front-ends and drives the Corsair DIMC hardware. The design goal is to be "performant and easy to use" with minimal user-facing code changes.

---

## 2. Aviator Software Stack Layers

```
┌─────────────────────────────────────────────────┐
│        PyTorch (torch.nn.Module)                │  ← User framework
├─────────────────────────────────────────────────┤
│        Triton DSL (optional custom kernels)     │  ← Kernel authoring
├─────────────────────────────────────────────────┤
│  Model Factory (distributed inference template) │  ← Deployment helper
├─────────────────────────────────────────────────┤
│  Compressor (Block FP quantization tool)        │  ← MXINT4/8/16 quant
├─────────────────────────────────────────────────┤
│  Compiler (MLIR-based AOT compiler)             │  ← Code generation
├─────────────────────────────────────────────────┤
│  Inference Engine (distributed exec engine)     │  ← Multi-card dispatch
├─────────────────────────────────────────────────┤
│  Runtime (host ↔ Corsair card management)       │  ← PCIe comm, memory
├─────────────────────────────────────────────────┤
│  DIMC Hardware (Corsair chiplets)               │  ← TSMC 6nm DIMC
└─────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Framework Integration
- **PyTorch**: Primary integration path; users provide standard `torch.nn.Module` models
- **Triton DSL**: Supported for custom kernel authoring; leverages MLIR backend
- No JAX or TensorFlow support disclosed

### 3.2 Model Factory
- Template library of popular LLM architectures (Llama, GPT family inferred)
- Handles model sharding across multiple Corsair cards (distributed inference)
- Not public / source not disclosed

### 3.3 Compressor
- Model compression toolkit
- Converts standard FP16/BF16 weights to OCP MX block floating point: MXINT4, MXINT8, MXINT16
- Calibration and quantization workflow (details not public)

### 3.4 Compiler
- **MLIR-based** AOT compiler — d-Matrix leverages the MLIR ecosystem
- Accepts PyTorch graph (via torch.export or Triton) → lowers to DIMC-native binary
- Handles MAC array tiling for 64×64 (INT8) or 64×128 (INT4) DIMC core shapes
- Full compilation details are proprietary (not open-source)

### 3.5 Inference Engine
- Distributed inference execution engine
- Schedules across multiple Corsair cards (intra-server PCIe fabric, inter-server Ethernet via JetStream)
- Handles tensor parallelism across chiplets and cards
- KV-cache placement across SRAM (hot) and LPDDR5X (cold capacity)

### 3.6 Runtime
- Manages host↔Corsair PCIe Gen5 communication
- Allocates SRAM and LPDDR5X on device
- Handles DMX Bridge card abstraction (2-card merge appears as single device)
- Not open-source; proprietary binary

---

## 4. Software Availability

| Component | Status |
|-----------|--------|
| Aviator stack | not public (proprietary) |
| Model Factory | not public |
| Compressor | not public |
| Compiler source | not public |
| Inference Engine | not public |
| Runtime | not public |
| PyTorch front-end | public (standard PyTorch) |
| Triton DSL | public (standard Triton) |
| Open-source repos | none disclosed |

---

## 5. Deployment Model

- Corsair cards are PCIe accelerators inserted into standard datacenter servers
- Aviator stack runs on host CPU; dispatches compute to Corsair via PCIe
- JetStream 400G NIC card in same server handles inter-node networking
- No cloud service or managed API announced

---

## 6. Key Observations

1. **Inference-only**: No training capability disclosed; Aviator is purely an inference stack
2. **Quantization-first**: Block FP (MX) quantization is mandatory for efficiency; FP32/FP16 passthrough not documented
3. **MLIR heritage**: Compiler back-end uses MLIR, consistent with modern ML compiler trends (XLA, TT-MLIR, etc.)
4. **Minimal ISA exposure**: No PTX/ISA equivalent published; compiler owns the full stack
5. **Triton**: Notably supports Triton DSL — enables custom attention/MoE kernels without low-level DIMC programming

---

## Sources

- [d-Matrix Product Page — Aviator stack overview](https://www.d-matrix.ai/product/)
- [d-Matrix Technical White Paper](https://d-matrix.ai/pdf/d-Matrix-WhitePaper-Technical-FINAL.pdf)
- [d-Matrix Corsair at Hot Chips 2025 — ServeTheHome](https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/)
- [How d-Matrix's In-Memory Compute Tackles AI Inference Economics — Vik's Newsletter](https://www.viksnewsletter.com/p/d-matrix-in-memory-compute)

---

# Update — 2026-08-08 Investigation

*investigation: software-stack (update)*
*date: 2026-08-08*
*baseline: 2026-04-05 section above (retained unchanged)*
*sources: d-Matrix Wallaroo.ai acquisition announcement (2026-08-03); PRNewswire wire release; unite.ai coverage; Bowen Inc.; Infinity (infinity.inc) research writeups; Parasail blog*

---

## U1. Wallaroo.ai acquisition — a new orchestration layer above Aviator

**Acquired 2026-08-03. Terms not disclosed.** This is d-Matrix's second acquisition in four months (after the GigaIO datacenter business, 2026-04-02) and the most significant software-stack change since the baseline. The acquisition brings **platform, IP, and engineering / product / go-to-market staff**. Bowen Inc. was Wallaroo's exclusive financial advisor.

The baseline stack diagram above tops out at PyTorch — d-Matrix had **no deployment or orchestration layer**. Wallaroo supplies one.

### What Wallaroo provides

| Component | Description | Status |
|---|---|---|
| Serving runtime | Executes deployed models; supports **vLLM** and **SGLang** | not public |
| Control plane | Model deployment, lifecycle and operations management | not public |
| Heterogeneous targeting | Deploys across **x86, Arm and GPU** as well as d-Matrix silicon | not public |
| Deployment environments | **Cloud, on-prem, edge, and air-gapped** | not public |

### Strategic reading

d-Matrix's own framing (CEO Sid Sheth, at the GigaIO deal): *"Inference is bigger than any one chip. It's now a systems problem."* Wallaroo is the software half of that thesis — a serving layer that is explicitly **hardware-heterogeneous**, which matters because d-Matrix's own deployment story (see Parasail below) is a *disaggregated split* in which Corsair runs alongside NVIDIA GPUs rather than replacing them. A control plane that can schedule prefill onto Hopper/Blackwell and decode onto Corsair is a prerequisite for that architecture.

### Revised stack layering

```
┌─────────────────────────────────────────────────┐
│  Wallaroo Control Plane (deployment, lifecycle) │  ← NEW (acq. 2026-08-03)
├─────────────────────────────────────────────────┤
│  Wallaroo Serving Runtime (vLLM / SGLang;       │  ← NEW
│  x86 / Arm / GPU / Corsair; cloud/on-prem/edge) │
├─────────────────────────────────────────────────┤
│        PyTorch (torch.nn.Module)                │
├─────────────────────────────────────────────────┤
│        Triton DSL (optional custom kernels)     │
├─────────────────────────────────────────────────┤
│  Model Factory (distributed inference template) │
├─────────────────────────────────────────────────┤
│  Compressor (Block FP quantization tool)        │
├─────────────────────────────────────────────────┤
│  Compiler (MLIR-based AOT compiler)             │
├─────────────────────────────────────────────────┤
│  Inference Engine (distributed exec engine)     │
├─────────────────────────────────────────────────┤
│  Runtime (host ↔ Corsair card management)       │
├─────────────────────────────────────────────────┤
│  DIMC Hardware (Corsair chiplets)               │
└─────────────────────────────────────────────────┘
```

How Wallaroo's runtime interfaces with the Aviator Inference Engine is **not disclosed**.

---

## U2. Infinity ("Ignition") — external automated kernel development

**2026-07-22.** Infinity (infinity.inc) is a real independent company (raised **$15M**) building an agent called **Ignition** that automates kernel development for AI chips. It is **not** a d-Matrix product; it is a third-party bring-up accelerator.

Infinity's own published figures:

- Generated a **full inference stack running Qwen3, Qwen3.5 and Gemma4 end-to-end in 10 days**, at **up to 92% of speed-of-light**.
- Reached **92% of Corsair's theoretical peak in 10 hours**.

**Not recorded:** vendor phrasing about "Corsair's 32 compute units" circulating in coverage of this collaboration could **not** be independently verified and does not reconcile with the documented 2,048 DIMC cores / 16 chiplets per dual card. It is deliberately excluded.

Significance for the stack: Corsair has **no user-facing kernel library** and a closed compiler, so third-party model bring-up normally depends on d-Matrix's own engineering. An automated kernel-generation path is the first evidence of bring-up scaling beyond the vendor's internal team.

---

## U3. Deployment model — heterogeneous disaggregation (Parasail, 2026-07)

The baseline's "Deployment Model" section described Corsair as a drop-in PCIe accelerator. The Parasail announcement adds an explicitly **heterogeneous, disaggregated** deployment pattern:

- **NVIDIA Hopper / Blackwell** handle compute-bound **prefill**.
- **Corsair** handles latency-sensitive **decode**.

**Status caveat:** this is an **announced partnership and stated intent, not a deployment.** Parasail's own post says the companies "plan to explore expanded integration" across Parasail's fleet of 40+ datacenters in 15 countries, and that they "plan to share detailed performance results and case studies following the first series of deployments." Corsair is **not** currently running across 40+ datacenters. The accompanying "up to 10× faster interactive inference" and "up to 3× better energy efficiency" figures are unqualified vendor claims with **no model, baseline, batch size or benchmark disclosed**.

---

## U4. Unchanged / still not disclosed

- Aviator remains **closed-source**; no open-source repos disclosed.
- No JAX or TensorFlow support disclosed.
- No cloud service or managed API from d-Matrix itself announced (Wallaroo's control plane is a deployable product, not a d-Matrix hosted service).
- **Raptor** software support: **not disclosed.** The ISCA 2026 paper is a hardware disclosure; no Raptor compiler, runtime or SDK details are public.
- Training support: still none. Inference-only.

---

## Update Sources (2026-08-08)

- [d-Matrix acquires Wallaroo.ai (2026-08-03)](https://www.d-matrix.ai/announcements/d-matrix-acquires-wallaroo/)
- [PRNewswire — d-Matrix acquires Wallaroo.ai to speed up deployment of heterogeneous AI inference workloads](https://www.prnewswire.com/news-releases/d-matrix-acquires-wallarooai-to-speed-up-deployment-of-heterogeneous-ai-inference-workloads-302840688.html)
- [unite.ai — d-Matrix buys Wallaroo to orchestrate inference across chips](https://www.unite.ai/d-matrix-buys-wallaroo-to-orchestrate-inference-across-chips/)
- [Bowen Inc. — Wallaroo.ai acquired by d-Matrix (exclusive financial advisor)](https://boweninc.com/transaction/wallaroo-ai-acquired-by-d-matrix/)
- [Infinity — Advancing full model AI inference on Corsair](https://infinity.inc/research/dmatrix-corsair)
- [Infinity × d-Matrix partnership](https://infinity.inc/research/dmatrix-partnership)
- [d-Matrix blog — Advancing full model AI inference on Corsair with Infinity (2026-07-22)](https://www.d-matrix.ai/advancing-full-model-ai-inference-on-corsair-with-infinity/)
- [Parasail × d-Matrix accelerators](https://www.parasail.io/blogs/parasail-d-matrix-accelerators)
- [engineering.com — Hopper/Blackwell prefill + Corsair decode split](https://www.engineering.com/parasail-deploys-d-matrix-accelerators-for-ai-inference/)
