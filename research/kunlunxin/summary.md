# Kunlunxin XPU — Research Summary

*as_of: 2026-08-08*
*research_baseline: 2026-04-05*
*chip: kunlunxin*
*device_class: AI Accelerator (百度昆仑芯)*

---

## What It Is

**Kunlunxin (昆仑芯)** is Baidu's AI chip subsidiary, spun off as an independent company in 2021. It produces the **XPU** (AI Processing Unit) family — multi-core SIMD accelerators designed for training and inference across datacenter AI workloads. As of early 2026, Kunlunxin has filed for a Hong Kong IPO and won China's largest domestic AI chip procurement (China Mobile, ~10B RMB).

---

## Architecture in One Paragraph

The XPU is a **cluster-based multi-core SIMD accelerator** with two major compute engines: **SDNN** (Spatial DNN, a systolic-array-style MAC array for GEMM/Conv/Deconv) and **XVME** (vector math engine for activations, reductions, normalizations). Compute clusters share local SRAM scratchpads. Unlike a GPU, there is no hardware cache hierarchy — memory management is **software-directed** through XDNN/XTCL compiler hints. The third-generation **P800** (XPU-P) achieves 345 TFLOPS FP16 with 96 GB HBM3 and a 200 GB/s chip-to-chip fabric, enabling a single 8-card server to run DeepSeek V3/R1 671B without scale-out.

---

## Generation Summary

| Gen | Chip | Process | FP16 TFLOPS | Memory | Key Milestone |
|-----|------|---------|-------------|--------|---------------|
| 1 | Kunlun 1 (XPU-K) | Samsung 14nm | ~64 | 16 GB HBM2 | Hot Chips 2020 paper; first Baidu XPU |
| 2 | R200 / R300 (XPU-R) | 7nm | 128 | 32 GB GDDR6 | Mass production; PaddlePaddle R200/R300 support |
| 3 | P800 (XPU-P) | 7nm | 345 | 96 GB HBM3 | 30,000-card Baidu cluster; full DeepSeek V3/R1 671B |
| 4 | M100 / M300 | not disclosed | not disclosed | not disclosed | Announced 2025-11-13. **M100**: first physical public showing at WAIC 2026 (2026-07-17 → 07-20) with no official materials and carrier boards still in development; **no spec published**. **M300**: training + multimodal, targeted early 2027 |

---

## Software Stack Summary

| Layer | Component | Notes |
|-------|-----------|-------|
| Framework | PaddlePaddle (native XPU backend) | Primary; 51+ verified models |
| Framework | PyTorch (torch_xpu plugin) | Partial; inference focus |
| LLM Serving | vLLM-Kunlun (baidu/vLLM-Kunlun) | Out-of-tree **hardware plugin** for stock vLLM (not a fork); PagedAttention on XPU; v0.11.0 stable 2026-03-13; P800 only |
| Graph Compiler | XTCL (TVM-based AOT/JIT) | ONNX + PaddlePaddle IR import |
| Op Library | XDNN (BLAS + DNN + Attention) | CUDA-cuDNN/cuBLAS equivalent |
| Kernel SDK | XTDK (C/C++ data-parallel) | CUDA C++ equivalent; LLVM-based |
| Runtime | XRE v5.x (streams, memory, events) | CUDA Runtime equivalent |
| Driver | XPU Linux kernel module | Ubuntu/CentOS; version-pinned |

---

## Competitive Position (2025–2026)

- **vs Huawei Ascend 910B**: P800 matches in FP16 (345 vs 256 TFLOPS); larger HBM3 (96 vs 64 GB); better for LLM single-node inference
- **vs Nvidia A100**: P800 at ~345 TFLOPS FP16 exceeds A100 (312 TFLOPS); competitive for training; loses on ecosystem maturity
- **vs Nvidia H20** (China-market legal): P800 significantly outperforms H20 (148 TFLOPS FP16)
- **Software gap**: PaddlePaddle is a strong differentiator in China but lags PyTorch globally; vLLM-Kunlun and XTCL/ONNX help close the gap

---

## Scale and Deployment

- **Baidu internal**: 30,000-P800 cluster operational, 2025; training DeepSeek-scale models
- **China Mobile**: 10B RMB procurement of P800, August 2025 — largest single domestic AI chip order
- **Scale-up ladder** (updated 2026-08-08; all figures are vendor claims relayed by Chinese media, none independently benchmarked): 8 cards/server → **32/64-card cabinet supernode** (launched April 2025, vendor says volume-delivered) → **Tianchi 256** ("lit up" April 2026, stated on sale June 2026; volume shipping not independently confirmed) → **Tianchi 512** (H2 2026 target, not shipped). Fabric is Baidu's self-developed **XPU-Link** protocol with programmable switching; 77% claimed measured bandwidth efficiency
- **Corporate**: confidential HKEX Form A1 filed 2026-01-01 (announced 2026-01-02); a ~USD 50B valuation target and Q3 2026 listing are **press-reported only** (Reuters 2026-06-28) and not listed as of 2026-08-08

---

## Key Sources

- [Hot Chips 2020 Kunlun Paper (IEEE)](https://ieeexplore.ieee.org/document/9366056)
- [Kunlunxin Wikipedia](https://en.wikipedia.org/wiki/Kunlunxin)
- [P800 specifications (CSDN)](https://blog.csdn.net/Rong_Toa/article/details/151322568)
- [vLLM-Kunlun (GitHub)](https://github.com/baidu/vLLM-Kunlun)
- [昆仑芯官网](https://www.kunlunxin.com/technology)
- [PaddlePaddle XPU docs](https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/index_cn.html)
- [M100/M300 roadmap — TrendForce](https://www.trendforce.com/news/2025/11/13/news-baidu-rolls-out-kunlun-roadmap-m100-m300-ai-chips-arrive-2026-2027/)
- [Kunlunxin HK IPO — SCMP](https://www.scmp.com/news/china-future-tech/semiconductors/article/3338471/baidu-chip-unit-kunlunxin-files-hong-kong-ipo-amid-chinas-push-tech-self-reliance)
