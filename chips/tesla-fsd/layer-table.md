# Tesla FSD Chip — Layer Table

*as_of: 2026-09-13*

| Layer | Component | Detail |
|---|---|---|
| 1 — Hardware | NPU | 96×96 systolic MAC array; HW3: 2×NPU=73.7 TOPS; HW4: 3×NPU=121.6 TOPS INT8. AI4.5 / AI4.1 / AI5: NPU count and geometry **not disclosed** (AI4.1: "~+10% compute" vs AI4, Musk; AI5: no TOPS figure at all). **Update 2026-09-13**: a JPMorgan-note-via-Electrek figure of "~10% more compute" is also attributed to a chip called "AI4.5" — thirdhand, possibly a mislabeling of AI4.1; not adopted as confirmed |
| 2 — Memory | SRAM / DRAM | 32 MiB SRAM/NPU; HW3: 8GB LPDDR4 68GB/s (WikiChip) or ~48GB/s (Electrek); HW4: 16GB GDDR6 224GB/s (WikiChip) or ~384GB/s (Electrek) — **contested, none Tesla-published**. AI4.1: 32GB/SoC, 64GB/board, ~+10% BW. AI5: **not disclosed**; 12 discrete SK hynix DRAM packages on organic substrate, no HBM. **Update 2026-09-13**: same JPMorgan-via-Electrek note attributes "~2× memory" to "AI4.5" — thirdhand, possibly the same AI4.1 figure relabeled; not adopted as confirmed |
| 3 — ISA | Custom 8-instr | DMA read/write, 3× dot-product, scale, elem-wise add, param slot (HW3-documented, believed HW4). **No ISA info for AI4.5 / AI4.1 / AI5** |
| 4 — Runtime | Static microcode | Compile-time scheduled; no dynamic dispatch; DMA-fed. No runtime change disclosed for any 2026 variant |
| 5 — Driver / OS | Embedded Linux | ARM Cortex-A clusters; sensor preprocessing. AI5 integrates Arm CPU cores + PCIe blocks on-die (core count not disclosed) |
| 6 — Compiler | Tesla NN Compiler (proprietary) | Two-pass: topology mapping → INT8 quant + static schedule. Still closed; no AI5 toolchain disclosure |
| 7 — Op Library | Fused kernels | Conv+scale+act+pool fusion baked into compiler |
| 8 — Framework | PyTorch (training only) | Open-source training; proprietary inference path |
| 9 — Deployment | OTA binary | Chip-specific binary delivered over-the-air; a distinct binary per chip revision, so AI4.5/AI4.1/AI5 each imply their own target |
| 10 — Inference pipeline | 48 NNs | Camera → BEV occupancy → trajectory → decision |
| 11 — Precision | INT8 | 8×8 mul, 32-bit accum; FP32 for training. **AI5 dtype support not disclosed** |
| 12 — Communication | Dual-chip (vehicle) | Two FSD computers per vehicle for safety redundancy. AI4.5 reportedly a **three**-SoC board (low-confidence firmware inference). AI5 carries on-die **PCIe** — first FSD-line part described with a host interface; no scale-out fabric disclosed |
| 13 — Quantization | INT8 QAT | Quantization-aware training; pre-quantized OTA delivery |
| 14 — SDK | None (closed) | No public API; fully Tesla-internal. Unchanged in 2026 — no SDK, no MLPerf, no Hot Chips 38 talk |
| 15 — Process | Samsung 14nm (HW3) / 7nm (HW4) | Edge SoC with CPU+GPU+NPU integration. AI4.1: Samsung 7nm **modified**. **AI5: dual-sourced — TSMC (node not disclosed) + Samsung SF2-class 2nm, Taylor TX** |
| 16 — Lifecycle status | Generation status | HW3 deployed but **declared insufficient for unsupervised FSD (2026-04-22)**, retrofit programme announced; **update 2026-09-13**: HW3 now receiving reduced "FSD v14 Lite" release, reported rising HW3 hardware failures (thirdhand, JPMorgan-via-Electrek). HW4/AI4 shipping baseline, reasserted sufficient for FSD v15/unsupervised (2026-08-20 JPMorgan note). AI4.5 shipping, never announced. AI4.1 **announced only**, target 2027. AI5 **taped out + one packaged engineering sample**; **update 2026-09-13**: delayed to mid-2027 (reaffirms prior Electrek estimate); Cybercab originally planned on **AI4** hardware |
