# Kunlunxin XPU Layer Mapping Table

*as_of: 2026-08-08*
*research_baseline: 2026-04-05*
*chip: kunlunxin*
*device_class: AI Accelerator (百度昆仑芯)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | PaddlePaddle (native XPU backend; `paddle.device.set_device('xpu')`; static + dynamic graph; AMP; 51+ verified models) | confirmed | software-stack, paddlepaddle-xpu-docs |
| Framework Integration | PyTorch via torch_xpu custom device plugin (ATen backend; inference focus; training partial) | confirmed | software-stack, paddlepaddle-xpu-docs |
| Framework Integration | vLLM-Kunlun (baidu/vLLM-Kunlun — out-of-tree **hardware plugin**, NOT a fork; registered as a standard vLLM platform plugin via Python entry points per the vLLM "RFC: Hardware Pluggable" interface, no vLLM source modification; PagedAttention + continuous batching on XPU; v0.11.0 stable 2026-03-13; 20+ models incl. Qwen 2/2.5/3/3.5, Llama, DeepSeek-V3.2, GLM, Gemma4, InternLM2, Kimi-K2, Qwen-VL / InternVL / InternS1; prerequisites list **Kunlun3 P800 only — no M100 support in-tree**) | confirmed | vllm-kunlun-github, vllm-kunlun-readthedocs, github-releases-api |
| Framework Integration | Hugging Face Transformers (via PaddlePaddle bridge; ERNIE, BERT, Llama, ChatGLM, Qwen, DeepSeek) | confirmed | software-stack, paddlepaddle-xpu-docs |
| Framework Integration | FastDeploy (Baidu inference platform; XpuWorker class; XVLLM for LLM serving) | confirmed | software-stack, fastdeploy-xpu |
| Compiler / IR | XTCL (XPU Tensor Compilation Library; TVM-based AOT/JIT; ONNX + PaddlePaddle IR + Relay IR input; op fusion + layout transform) | confirmed | software-stack, xtcl-docs |
| Compiler / IR | Graph-level optimizations in XTCL (operator fusion, constant folding, XPU-aware tiling and memory placement) | confirmed | software-stack, xtcl-docs |
| Compiler / IR | XTDK C/C++ compiler (LLVM/Clang extended; `.xpu` kernel source → XPU object code; SIMD vectorization, software pipelining) | confirmed | software-stack, xtdk-guide |
| Op Library | XDNN (XPU DNN operator library; BLAS + DNN; C API with tensor descriptors; GEMM/Conv/Attention/Norm/Activation/Reduction) | confirmed | software-stack, xdnn-docs |
| Op Library | Fused attention kernel in XDNN (FlashAttention-style; used by PaddlePaddle and vLLM-Kunlun) | confirmed | software-stack, vllm-kunlun-github |
| Op Library | Custom operators via XTDK + PaddlePaddle XPU Plugin (load_op_meta_info_and_register_op) | confirmed | software-stack, paddlepaddle-2.4-release |
| Runtime | XRE (XPU Runtime Environment v5.x; device init, xpu_malloc/xpu_free, host↔device transfer, streams, events, multi-device up to 8) | confirmed | software-stack, fastdeploy-xpu, xre-docs |
| Runtime | Multi-card parallelism via PaddlePaddle DistributedDataParallel on XPU | confirmed | software-stack, paddlepaddle-xpu-docs |
| Driver / Firmware | XPU Linux kernel module (ioctl interface; Ubuntu 18/20/22 + CentOS 7/8; version must match kernel) | confirmed | software-stack, xre-install-guide |
| Communication | Standard Ethernet / InfiniBand for scale-out (no proprietary collective library analogous to NCCL documented) | inferred | hw-architecture |
| Debugging / Profiling | torch_xray (module-level operator precision dump; automatic GPU-vs-P800 layer-by-layer comparison; auto-generates operator unit tests) — documented in the vLLM-Kunlun developer guide, not present in the repo's top-level tree | confirmed | vllm-kunlun-readthedocs |
| Debugging / Profiling | xpu_profiler (nsys-like profiler producing operator call timelines) — documented in the vLLM-Kunlun developer guide, not present in the repo's top-level tree | confirmed | vllm-kunlun-readthedocs |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | SDNN (Spatial DNN accelerator): systolic-array-style MAC array for GEMM, Conv, Deconv, elementwise | confirmed | hw-architecture, hotchips2020 |
| Compute Engine | XVME (XPU Vector Math Engine): SIMD vector unit for activations, normalizations, reductions | confirmed | hw-architecture, hotchips2020 |
| Compute Engine | Cluster-based multi-core design; SPMD dispatch; multiple clusters per die | confirmed | hw-architecture, hotchips2020 |
| Compute Engine | Kunlun 1 (XPU-K): Samsung 14nm; 256 TOPS INT8 / 64 TFLOPS FP16; 160W TDP; Samsung I-Cube 2.5D | confirmed | hw-architecture, hotchips2020, techinsights |
| Compute Engine | Kunlun 2 R200/R300 (XPU-R): 7nm; 256 TOPS INT8 / 128 TFLOPS FP16; 32 GB GDDR6; 512 GB/s BW | confirmed | hw-architecture, tomshardware-kunlun2 |
| Compute Engine | P800 (XPU-P): 7nm; 345 TFLOPS FP16; 96 GB HBM3; ~1.6 TB/s; single 8-card server runs DeepSeek V3/R1 671B | confirmed | hw-architecture, digitimes-p800, csdn-p800 |
| Compute Engine | M100 (Gen 4, inference-optimised, self-developed XPU architecture, "fully domestic supply chain"): announced 2025-11-13; **no process node, memory capacity/bandwidth, TFLOPS/TOPS or TDP disclosed**. First physical public showing WAIC 2026 (2026-07-17 → 07-20) with no official introduction materials and carrier boards still in development | announced (no specs) | waic2026-xinzhixun, waic2026-10jqka, trendforce-m100-m300 |
| Compute Engine | M300 (Gen 4, training + multimodal): announced 2025-11-13, targeted early 2027; no specification disclosed | announced (no specs) | trendforce-m100-m300 |
| Data Path | Cluster-local SRAM scratchpad (software-managed; no hardware caches) | confirmed | hw-architecture, hotchips2020 |
| On-chip Memory | Distributed SRAM scratchpad per cluster; shared SRAM at die level | confirmed | hw-architecture, hotchips2020 |
| Off-chip Memory | Gen 1: HBM2 16 GB 512 GB/s (I-Cube 2.5D package); Gen 2: GDDR6 32 GB 512 GB/s; Gen 3: HBM3 96 GB ~1.6 TB/s | confirmed | hw-architecture, techinsights, tomshardware |
| Off-chip Memory | Gen 4 (M100): memory type, capacity and bandwidth **not disclosed**. Only statement in existence is unattributed WAIC 2026 booth-staff commentary that HBM capacity is "somewhat lower than NVIDIA H20" | low confidence (booth-staff, unattributed) | waic2026-xinzhixun |
| Scale-up Interconnect | Gen 2: PCIe Gen4 x16 (no dedicated chip-to-chip fabric) | confirmed | hw-architecture |
| Scale-up Interconnect | Gen 3 (P800): XLINK 200 GB/s chip-to-chip; 8 cards per server | confirmed | hw-architecture, csdn-p800 |
| Scale-up Interconnect | XPU-Link: Baidu self-developed supernode interconnect protocol with programmable switching, self-designed copper cabling and liquid-cooling CDU; claimed 77% measured bandwidth efficiency (vendor claim, WAIC 2026) | vendor claim | waic2026-qq, waic2026-sina |
| Scale-up Interconnect | Kunlun Super Node (32/64-card cabinet): launched **April 2025** (pre-baseline; missed at first pass); vendor says among the first domestic supernodes in volume delivery; claimed 8× inter-card bandwidth vs the 8-card server, 13× single-card inference efficiency, 5–10× MoE training performance (a range, not 10×); cabinet claimed extensible to 512 cards | vendor claim | waic2026-qq, kunlunxin-supernode |
| Scale-up Interconnect | Tianchi 256 SuperNode (天池256卡超节点): 256 P800 cards in one cabinet; "lit up" (点亮) April 2026, stated on sale June 2026 at Baidu Create 2026 — **volume shipping not independently confirmed** (Tencent News 2026-07-22 still says "即将全面上市"); +25% throughput / +50% inference efficiency (Create 2026, 2026-05-13); 4× inter-card BW and 3.5× single-card token throughput are **Baidu World 2025 (2025-11-13) figures, not 2026 results** | vendor claim | create2026-qq, waic2026-qq, sina-baiduworld2025 |
| Scale-up Interconnect | Tianchi 512 SuperNode: 512-card fabric for trillion-parameter training; **not shipped**, H2 2026 target; no inter-card-bandwidth multiplier vs Tianchi 256 has been published | announced | hw-architecture, waic2026-qq |
| Scale-up Interconnect | Stated roadmap *intent* for thousand-card and 4,000-card supernodes on the Kunlun M-series — **no timeline given** | vendor statement | waic2026-qq |
| Scale-out Interconnect | Ethernet / InfiniBand (standard; 30,000-chip Baidu cluster uses undisclosed fabric) | inferred | hw-architecture, digitimes-p800 |
| Host Interface | PCIe Gen4 x16 (Gen 2/3 cards); acceleration card form factor | confirmed | hw-architecture |
