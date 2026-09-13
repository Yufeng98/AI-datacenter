# Untether AI — Software Stack Layer Table

*as_of: 2026-08-08*

**Status: DEAD.** Untether AI Corporation went bankrupt on 2025-10-15 (BIA liquidation, Ontario Estate
No. 31-3285414, PwC Inc. trustee). The entire stack below was **discontinued in June 2025**, never
open-sourced, and has no successor. It is documented in the past tense as a historical data point.

| Layer | Component (discontinued) | Notes |
|---|---|---|
| Framework | PyTorch, TensorFlow, ONNX | Model ingestion |
| Quantization | imAIgine PTQ / PQT / KD | INT4/8/FP8/BF16; automated calibration |
| Compiler | imAIgine Generative Compiler (v25.04) | Auto-generated kernels; claimed 300+ models; final release Mar 2025 |
| Bare-Metal | HPC Flow (RISC-V + PE arrays) | Expert kernel programming path |
| Physical Alloc | imAIgine Allocator | Maps to 729 banks; multi-chip |
| Runtime | imAIgine Runtime (Python + C lib) | PCIe DMA; power mode control |
| HW Interface | PCIe BAR + DMA | Low-profile card, 75 W |
| Chip | speedAI240 | 729 banks, 1456 RISC-V, 238 MB SRAM |
