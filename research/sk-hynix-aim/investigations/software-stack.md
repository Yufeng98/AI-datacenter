# SK Hynix AiM / AiMX — Software Stack Investigation

*chip: sk-hynix-aim*
*investigation: software-stack*
*date: 2026-04-05*
*sources: HC34 poster, ASPLOS 2023 tutorial, IEEE papers, SK Hynix newsroom*

---

## 1. Overview

SK Hynix published the AiM software stack at Hot Chips 34 and ASPLOS 2023. The stack is designed around a **clean separation of concerns** between the AiM ISR hardware layer and higher-level framework integration. Unlike GPU stacks, there is no publicly available open-source SDK; the stack is presented in academic papers as reference architecture.

---

## 2. AiM Software Stack Layers

```
┌─────────────────────────────────────────────────────┐
│  ML Frameworks (PyTorch / TensorFlow)               │  ← User code unchanged
├─────────────────────────────────────────────────────┤
│  AiM BLAS Library / High-Level API                  │  ← GEMV, embedding ops
├─────────────────────────────────────────────────────┤
│  AiM Runtime Library                                │  ← ISR dispatch, sync
├─────────────────────────────────────────────────────┤
│  AiM ISR (Instruction Set Register)                 │  ← HW command abstraction
├─────────────────────────────────────────────────────┤
│  Extended DRAM Command Interface                    │  ← All-bank activation
├─────────────────────────────────────────────────────┤
│  GDDR6-AiM Hardware (per-bank MAC units)            │  ← DRAM + compute
└─────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Extended DRAM Command Set
- AiM defines new commands added to the standard GDDR6 command set
- Host memory controller issues **AiM ISR instructions** via these commands
- Commands trigger all-bank simultaneous MAC operations
- JEDEC GDDR6 standard commands remain fully functional (backward compatible)

### 3.2 AiM ISR (Instruction Set Register)
- Hardware abstraction layer between software and DRAM MAC units
- ISR instructions describe compute operations: load operands, MAC, accumulate, writeback
- Not a full programmer ISA; more akin to a hardware register-mapped command set
- Exposed to software via driver/runtime library

### 3.3 AiM Runtime Library
- Host-side library managing:
  - GDDR6-AiM initialization and mode switching (standard DRAM ↔ compute mode)
  - ISR instruction sequencing and dispatch
  - Synchronization between host (CPU/GPU) and AiMX card
  - Memory allocation on AiM device
- Status: **not publicly released** as SDK; described in academic papers

### 3.4 AiM BLAS Library / High-Level API
- Provides BLAS Level 2 (GEMV) and embedding operations as high-level API calls
- Analogous to cuBLAS Level 2 for DRAM-native compute
- Enables host GPU to offload memory-bound GEMV to AiMX
- Status: **not public** (described in ASPLOS 2023 tutorial)

### 3.5 Framework Integration
- **PyTorch**: Integration path described (transparent PIM offload — unmodified user model)
- **TensorFlow**: Mentioned as secondary path
- At framework level, memory-bound ops (attention GEMV, embedding) are intercepted and dispatched to AiMX
- Requires operator interception hook or custom dispatch backend
- Status: **not publicly released**

---

## 4. FPGA Reference Platform

SK Hynix developed a **dedicated FPGA-based reference platform** for:
- Validating AiM hardware design pre-silicon
- Evaluating system-level performance
- Developing and testing software stack
- Provides a simulation environment matching AiM ISR behavior

---

## 5. Open-Source Ecosystem

| Resource | Status | URL |
|----------|--------|-----|
| aim_simulator (Ramulator 2.0) | Open source (research) | https://github.com/arkhadem/aim_simulator — *last commit 2025-07-22; dormant as of 2026-08-08* |
| AiMX vLLM integration | demonstrated publicly, not released *(added 2026-08-08)* | — (no code, fork, or upstream patch identified) |
| AiM SDK / runtime | not public | — |
| AiM BLAS library | not public | — |
| PyTorch backend | not public | — |
| ISR specification | partially in papers | IEEE ISCA 2021, HC34 |

---

## 6. Key Observations

1. **Command extension approach**: No new ISA; extends existing DRAM protocol — reduces controller redesign burden
2. **GEMV specialization**: Stack is optimized for GEMV (weight × activation vector); not general-purpose compute
3. **Transparent framework integration**: Goal is unchanged user code; PIM dispatch is transparent
4. **Research-grade public tools**: Only academic simulator (aim_simulator) is publicly available; production SDK is not released
5. **Heterogeneous model**: AiMX is a co-processor; software stack must coordinate with main GPU/CPU for non-GEMV ops

---

## 7. Re-investigation — 2026-08-08 (window 2026-04-05 → 2026-08-08)

*Investigated 2026-08-08. Verdict: **no in-window software development**; one substantive pre-baseline backfill (vLLM).*

### 7.1 Backfill — vLLM is a real, public part of the AiMX software story

The 2026-04-05 stack description stopped at "PyTorch / TensorFlow, not public." That understated the public record. At two 2025 conferences SK Hynix ran a **live AiMX demo serving LLM/reasoning workloads under the vLLM framework**:

| Event | SK Hynix post date | Event date | System | Claim |
|---|---|---|---|---|
| AI Infra Summit 2025 | **2025-10-01** | ~Sept 2025 (post date is the citable one) | Supermicro GPU SuperServer SYS-421GE-TNRT3, 2 × NVIDIA H100, 4 × AiMX cards | "stable long token generation using the vLLM framework" |
| 2025 OCP Global Summit | **2025-10-31** | Oct 2025 | Same configuration | Optimizing LLM attention / memory-bound workloads |

Implications for the stack model in §2:

- **vLLM sits above the AiM BLAS / high-level API layer**, in the position that PyTorch/TensorFlow previously occupied alone. The serving loop runs on the host (alongside the H100s); AiMX is reached through the AiM runtime as a memory-bound-GEMV offload target.
- The integration is **demonstrated publicly but not released** — no code, no fork, no patch upstream to vLLM has been identified, and no plugin or backend name was disclosed.
- **No numeric performance figures were disclosed** at either event. SK Hynix claimed qualitatively more concurrent requests and longer input prompts than a GPU-only configuration. Do not attach a speedup number to this demo.
- Independently corroborated by TechPowerUp #341520.

Revised layer status: `Serving framework | vLLM (demo integration) | demonstrated publicly, not released`.

### 7.2 Verified negatives (no software change in window)

| Item | Finding | Confidence |
|---|---|---|
| Public AiM SDK / runtime release | **None.** Still not public as of 2026-08-08. | high |
| `arkhadem/aim_simulator` (Ramulator 2.0, ~70 stars, 66 commits) | Last commit **2025-07-22** ("LPDDR5 debugged and tested"). No in-window activity. Remains the only public AiM software artifact. | high |
| MLPerf submission using AiMX | **None found.** | high |
| Hot Chips 38 AiM software talk | **None.** Hot Chips 38 (Aug 23–25, 2026) is a **future** event: its program is public but no slides, abstracts, or specs exist, and it lists no SK Hynix AiM/AiMX item. | high |
| Compiler / IR layer | Still **not applicable** — no compiler has appeared; the AiM Runtime continues to issue AiM ISR sequences directly. | high |

### 7.3 Bearing on productization

SK Hynix's own CES 2026 wording (press release 2026-01-05) still calls AiMX a "**prototype**" card. Combined with an unreleased SDK and a dormant public simulator, the software stack remains a **demonstration-grade** stack rather than a supported product stack. A TrendForce note (2026-03-10) claiming AiM is "already deployed in real-world applications" was **rejected on verification** — single secondary source, no customer, date, or volume, and contradicted by SK Hynix's own wording.

---

## Sources

*Added 2026-08-08:*

- [SK hynix newsroom — AI Infra Summit 2025 (vLLM + Supermicro 2×H100 + 4×AiMX), 2025-10-01](https://news.skhynix.com/en/ai-infra-summit-2025/)
- [SK hynix newsroom — 2025 OCP Global Summit, 2025-10-31](https://news.skhynix.com/en/sk-hynix-showcases-full-stack-ai-memory-portfolio-at-2025-ocp-global-summit/)
- [TechPowerUp #341520 — independent corroboration of the AiMX vLLM demo](https://www.techpowerup.com/341520/)
- [aim_simulator commit history — last commit 2025-07-22](https://github.com/arkhadem/aim_simulator/commits/main)
- [SK hynix CES 2026 press release — AiMX still "prototype", 2026-01-05](https://www.prnewswire.com/news-releases/sk-hynix-showcases-next-generation-ai-memory-innovation-at-ces-2026-302653161.html)

*Original (2026-04-05):*

- [System Architecture and Software Stack for GDDR6-AiM — HC34](https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf)
- [System Architecture and Software Stack ASPLOS 2023 Tutorial](https://events.safari.ethz.ch/asplos-pim-tutorial/lib/exe/fetch.php?media=system_architecture_and_software_stack_for_gddr6-aim_sk_hynix_pim_tutorial_asplos23_sent.pdf)
- [Hardware Architecture and Software Stack for PIM — ISCA 2021](https://ieeexplore.ieee.org/document/9499894)
- [aim_simulator GitHub](https://github.com/arkhadem/aim_simulator)
- [SK Hynix AiMX-xPU — Hot Chips 2024 IEEE](https://ieeexplore.ieee.org/document/10664793/)
