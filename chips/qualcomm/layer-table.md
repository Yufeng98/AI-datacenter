# Qualcomm Cloud AI 100/200 — Layer Table

*as_of: 2026-08-08*
*Covers Cloud AI 100 (Standard/Ultra) plus the Qualcomm **Dragonfly** portfolio announced 2026-06-24
(AI200, AI250, AI300 accelerators and the C1000 CPU). Dragonfly figures are Qualcomm marketing claims unless
marked otherwise; where Qualcomm published nothing, the cell reads "not disclosed".*

| Layer | Component | Detail |
|---|---|---|
| 1 — Hardware | AI Cores | 16 Hexagon Q6+HVX+HMX cores per SoC; Ultra: 64 cores (4 SoC). AI200/AI250/AI300 core type and count: not disclosed |
| 2 — Compute | TOPS | Standard: 400 TOPS INT8 @ 75 W; Ultra: 870 TOPS @ 150 W. AI200/AI250/AI300 peak FLOPS/TOPS: **not disclosed** — Qualcomm explicitly declined (The Register, 2026-06-30) |
| 3 — Tensor Unit | HMX | 8192 INT8 ops/cycle, 4096 FP16 ops/cycle, 125+ instructions (Cloud AI 100). Dragonfly tensor unit: not disclosed |
| 4 — Vector Unit | HVX | 512 INT8 ops/cycle, 700+ instructions (Cloud AI 100). Dragonfly vector unit: not disclosed |
| 5 — Memory (on-chip) | SRAM | 144 MB/SoC (576 MB Ultra); KV-cache + weights. AI200/AI250/AI300 SRAM: not disclosed |
| 6 — Memory (off-chip) | LPDDR → HBC near-memory | Cloud AI 100: 32 GB LPDDR4X 136 GB/s (Std), 128 GB 548 GB/s (Ultra); no HBM. AI200: LPDDR5x, 768 GB/card (Oct 2025 figure). **AI250: HBC Gen 1 — claimed 133 TB/s/card, "18× effective BW vs AI200" (unaudited)**. **AI300: HBC Gen 2, 3D-stacked DRAM-over-XPU-logic via TSVs — claimed "54× over AI200" (unaudited); capacity not disclosed.** HBC claims: 6× BW/W vs HBM, 200× capacity/W vs SRAM. Per-card capacity attribution conflict: Qualcomm gives 768 GB to AI200, The Register to AI250 |
| 7 — Interconnect | 3× NoC → UALink/ESUN | Cloud AI 100: Compute (186 GB/s), Memory, Config NoCs; no scale-up fabric. **Dragonfly: scale-up over UALink and ESUN; scale-out Ethernet 800G/1.6T over copper and optical, reach up to 20 km campus; per-chip BW and max system scale not disclosed** |
| 8 — Driver | QAIC kernel driver | Upstream Linux (drivers/accel/qaic/); open source. No AI200/AI250 driver support published |
| 9 — Runtime | libQAic (Cloud AI SDK) | Open source; async inference; multi-SoC sharding. Plus **MAX / Modular Cloud** (acquired 2026-07-29) as a separate, silicon-agnostic serving runtime — no published AI200/AI250 backend |
| 10 — Compiler | qaic-compile | ONNX/TorchScript → QNN binary; closed binary. **Now alongside Modular's open MLIR-based stack (Mojo/MAX), acquired 2026-07-29 — additive, not a replacement** |
| 11 — ONNX Runtime path | QNN Execution Provider | Open; production inference serving path |
| 12 — PyTorch path | ExecuTorch QC backend | Open; torch.export → AIC100 |
| 13 — Precision | INT8/FP16/FP32 | INT8 primary; FP16 tensor; FP32 scalar (Cloud AI 100). Dragonfly data types: not disclosed |
| 14 — Deployment | PCIe card → rack-scale | Standard FH3/4L; Ultra multi-SoC; AI200/AI250 direct-liquid-cooled rack-scale (160 kW rack, Oct 2025 figure); **AI300 air- and direct-liquid-cooled rack-level platform** |
| 15 — Workload | LLM + CV inference | High DRAM capacity for LLM; DLRM; image classification. Dragonfly positioning: agentic AI inference at rack scale |
| 16 — Host CPU (new) | Dragonfly C1000 | 250+ custom Oryon cores, chiplet, >5 GHz; ">2× perf/W" vs competitive server CPUs (Qualcomm estimate); >2 TB/s PCIe Gen 7 + CXL; agentic / general-purpose virtualization / AI head node configs; availability 2028; Meta named customer (production H2 2028) |
| 17 — Kernel language (new) | Mojo | MLIR-based, silicon-agnostic; arrived with the Modular acquisition (completed 2026-07-29); Chris Lattner → Qualcomm EVP of Advanced AI Software and Platforms. No published AI200/AI250/AI300 target |
| 18 — Roadmap status | Sampling timeline | AI200 sampling in FY2026 on LPDDR5x (Investor-Day-reported, no production-shipment claim); AI250 commercial sampling expected mid-2027; AI300 commercial sampling expected 2028; C1000 availability 2028 |
| 19 — Independent validation | None in this window | Qualcomm not among the 24 MLPerf Inference v6.0 submitters (2026-04-01); no Hot Chips 2026 (HC38, Aug 24–25 2026) talk |
