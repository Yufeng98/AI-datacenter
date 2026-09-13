# Moore Threads MTT GPU Layer Mapping Table

*as_of: 2026-08-08*
*chip: mthreads*
*device_class: GPU (China, 摩尔线程)*

## Software Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Framework Integration | torch_musa (PyTorch MUSA backend; torch.device("musa"); 470+ ATen ops; MUSAGraph; TorchInductor; FP8; LLM fusion; open-source) | confirmed | tomshardware-musa, github-torch-musa |
| Framework Integration | vllm-musa (vLLM port; LLaMA/Qwen/Baichuan inference; torchada CUDA shim + MATE tensor engine; PyPI vllm-musa) | confirmed | github-vllm-musa, turtlesai-musa |
| Framework Integration | llama.cpp MUSA backend (PR #8383 merged; GGML inference on MTT GPUs) | confirmed | github-llamacpp-pr8383 |
| Framework Integration | SGLang MUSA integration (Issue #16565; in-progress as of 2025) | inferred | github-sglang-16565 |
| Framework Integration | torch_musa v2.9.0 (2026-03-17; tracks PyTorch 2.9; requires MUSA SDK >= 4.3.2; FSDP2 Context Parallel (Ulysses); sparse tensor ops; torch.compile "reduce-overhead"; GEMM FP32-by-default with TF32 opt-in via TORCH_ALLOW_TF32_MUBLAS_OVERRIDE=1; known torch.compile perf regression vs v2.7.0) | confirmed | github-torch-musa-releases |
| Framework Integration | torch_musa v2.9.1 (2026-06-29; requires MUSA SDK >= 5.1.0; CUDA-aligned operator coverage across dense/quantized/sparse/sparsecsr/nested tensors; mutlass promoted to third-party dependency) | confirmed | github-torch-musa-releases |
| Framework Integration | tensorflow_musa_extension (repo created 2026-02-05; TensorFlow backend extension — supersedes the earlier "via CUDA ON MUSA only" note) | confirmed | github-mooretheads-org |
| Framework Integration | paddle_musa (repo created 2025-09-10; PaddlePaddle MUSA backend) | confirmed | github-mooretheads-org |
| Framework Integration | onnxruntime-musa (repo created 2026-07-27; ONNX Runtime MUSA execution provider) | confirmed | github-mooretheads-org |
| Compiler / IR | MCC (MUSA C Compiler; nvcc analog; compiles MUSA C++ <<<>>> kernels to MUSA ISA binary; MUSA SDK 4.0.1 at 2026-04 baseline, MUSA SDK 5.1.0 current line as of 2026-06) | confirmed | tomshardware-musa, turtlesai-musa, github-torch-musa-releases |
| Compiler / IR | tvm_musa (repo created 2026-01-09) + tvm-ffi (2026-03-24) — Apache TVM MUSA target and FFI layer | confirmed | github-mooretheads-org |
| Compiler / IR | Musify / CUDA ON MUSA (automated CUDA→MUSA source translator; cuBLAS/cuDNN/NCCL shims; PTX-level translation at runtime) | confirmed | tomshardware-musa, wccftech-musa |
| Compiler / IR | TileLang MUSA (tilelang_musa; tile-level kernel DSL; Triton/CUTLASS analog; compiles to MUSA C via MCC; open-source) | confirmed | github-tilelang-musa |
| Compiler / IR | TorchInductor + triton_musa backend (torch.compile() on MTT GPUs; triton_musa generates MCC-compilable code) | confirmed | github-torch-musa |
| Op Library | muDNN (cuDNN analog; Conv2D, SDPA/Attention, Norm, Pooling, Activation; FP32/BF16/FP16/INT8) | confirmed | tomshardware-musa, turtlesai-musa |
| Op Library | muBLAS (cuBLAS analog; GEMM/BLAS L1-L3; optimized for 128 TCE Tensor Cores on Chunxiao die) | confirmed | tomshardware-musa |
| Op Library | MATE — MUSA AI Tensor Engine (high-level LLM inference primitives; Python bindings via mthreads-ml-py; used by vllm-musa; promoted to its own repo `mate`, created 2025-12-10) | confirmed | github-vllm-musa, github-mooretheads-org |
| Kernel Library | muThrust (Thrust analog; Reduce, Scan, Sort, parallel primitives) | confirmed | turtlesai-musa |
| Kernel Library | muFFT (cuFFT analog; Fast Fourier Transform) | confirmed | turtlesai-musa |
| Kernel Library | TileLang MUSA (CUTLASS/Triton analog for manual tiled kernel authoring at tile abstraction level; repo created 2026-01-12) | confirmed | github-tilelang-musa |
| Kernel Library | mutlass — MUSA Templates for Linear Algebra Subroutines (CUTLASS analog; repo created 2024-09-29, NOT new; **newly promoted to a first-class torch_musa third-party dependency in v2.9.1**, supplying high-performance matmul kernels) | confirmed | github-torch-musa-releases, github-mutlass |
| Kernel Library | TileOPs (repo created 2026-05-29; TileLang-based high-performance LLM operator library) | confirmed | github-mooretheads-org |
| Kernel Library | MUSA HPC / AI4Science port wave (16 repos, 2026-06-11 → 2026-08-07): SpFFT-MUSA, flann-musa, eigen-musa, kokkos-musa, relion-musa, fused-ssim-musa, cp2k-musa, magma-musa, amgx-musa, lammps-musa, su2-musa, deepmd-kit-musa, mahout-musa, CV-CUDA_musa (+ ompi-musa, onnxruntime-musa) — quantum chemistry, MD, cryo-EM, CFD, dense/sparse linear algebra, perf-portability; consistent with PingHu native FP64 | confirmed | github-mooretheads-org |
| Runtime | MUSA Runtime (libmusa.so; cudart analog; musaMalloc/Free, musaMemcpyAsync, musaStream/Event, <<<>>> launch; MUSA SDK 5.1.0 current, 4.0.1 at baseline) | confirmed | tomshardware-musa, turtlesai-musa, github-torch-musa-releases |
| Runtime | MUSA Driver API (explicit context/module management; fine-grained memory control) | confirmed | turtlesai-musa |
| Runtime | torchada (thin CUDA→MUSA compatibility shim used by vllm-musa to intercept CUDA API calls; promoted to standalone repo 2026-01-04) | confirmed | github-vllm-musa, github-mooretheads-org |
| Runtime | MTClaw (repo created 2026-05-18; local tool-routing proxy for openclaw / opencode / hermes — agent-tooling layer, not a GPU runtime component) | confirmed | github-mooretheads-org |
| Driver / Firmware | MUSA Kernel Driver (.ko; PCIe BAR/IOCTL/DMA/IRQ; Ubuntu Intel x86 + Kylin Hygon x86) | confirmed | tomshardware-musa |
| Driver / Firmware | Kubernetes Device Plugin (MTT GPU resource advertising; health monitoring; multi-tenancy) | confirmed | theregister-10k-cluster |
| Driver / Firmware | mthreads-ml-py (repo created 2026-01-04; Python GPU management/monitoring bindings — nvidia-ml-py analog) | confirmed | github-mooretheads-org |
| Communication | MCCL (MUSA Collective Communication Library; NCCL analog; AllReduce/AllGather/ReduceScatter; MTLink intra-node; 10K GPU cluster tested) | confirmed | theregister-10k-cluster, tomshardware-10k |
| Communication | ompi-musa (repo created 2026-07-21; Open MPI port for MUSA — enables MPI-based HPC codes alongside the AI4Science port wave) | confirmed | github-mooretheads-org |
| Assembler / ISA | MUSA ISA (proprietary SIMT ISA; warp-based; FP32/TF32/BF16/FP16/INT8 on Chunxiao; **full-precision FP8→FP64 on PingHu Gen 4 / PH100 (vendor-confirmed)**; FP4/MTFP6/MTFP4 announced for Huagang **Gen 5**; not publicly documented) | confirmed | tomshardware-musa, wccftech-musa, mthreads-s5000-page |

## Hardware Layers

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| Compute Engine | MP (MUSA Processor) — atom compute unit: 128× FP32, 32× INT8/bitwise, 32× SFU, 2× FP64, TCE (Tensor Compute Engine), 28 KB local shared memory | confirmed | mthreads-blog, beyond3d-musa |
| Compute Engine | MTT S4000 (Chunxiao full die): 4,096 SP / 128 TCE Tensor Cores / 256 TU / 256 ROP; 25 TFLOPS FP32 / 50 TF32 / 200 BF16 / 200 INT8 TOPS; 450W; PCIe Gen5 | confirmed | videocardz-s4000, wccftech-s4000 |
| Compute Engine | MTT S3000 (Chunxiao 1/2 die): 2,048 SP / 64 TCE; 15.6 TFLOPS FP32; 32 GB GDDR6; 250W | confirmed | topcpu-s3000 |
| Compute Engine | MTT S80 (Chunxiao full die, gaming): 4,096 SP; 14.7 TFLOPS FP32; 16 GB GDDR6; 1.8 GHz | confirmed | videocardz-s80 |
| Compute Engine | MTT S5000 (PH100 die, PingHu 平湖 **Gen 4**, fourth-generation MUSA): vendor-confirmed full-precision FP8→FP64; MP/SP count, tensor-core count, clock, die config, process node, transistor count and TDP all **not disclosed**; shipping in accelerated mass production 2026 | confirmed (identity/data types) | mthreads-s5000-page |
| Compute Engine | MTT S5000 peak throughput: up to 1,000 TFLOPS single-card dense FP8 — **media-reported only, not vendor-published** (vendor page carries no spec table) | reported | baike-s5000, tencentcloud-s5000 |
| Compute Engine | Huashan AI GPU (Huagang 花港 / "Flower Harbor" **Gen 5**, announced 2025-12-20 at MDC 2025 for 2026 mass production — one generation beyond the shipping S5000, NOT "Gen 3"): dual chiplet; 8× HBM sites; between H100-B200 perf (claimed); FP4/FP64/MTFP6/MTFP4 | announced | wccftech-huashan, trendforce-huashan |
| Compute Engine | Sudijia Gen 1 die: 2,048–4,096 SP; no Tensor Cores; 12nm; MTT S60, MTT S2000 | confirmed | mthreads-official, cnblogs-s60 |
| Data Path | SIMT execution: threads organized into warps; hardware divergence via predication; <<<>>> launch model matching CUDA semantics | confirmed | mthreads-blog, beyond3d-musa |
| Data Path | TCE (Tensor Compute Engine): hardware matrix acceleration unit; drives BF16/FP16/INT8 tensor throughput (128 per Chunxiao full die) | confirmed | videocardz-s4000, wccftech-s4000 |
| Data Path | MUSA architecture supports graphics pipeline (vertex, rasterization, fragment) + compute — full-function GPU unlike AI-only accelerators | confirmed | mthreads-official |
| On-chip Memory | L2 Cache: 4 MB hardware-managed (S4000 / Chunxiao full die) — small vs NVIDIA A100 (~40 MB) and Biren BR100 (300 MB) | confirmed | technical-city-s4000, topcpu-s4000 |
| On-chip Memory | MP local shared memory: 28 KB per MP (configurable scratchpad; MUSA C access via __shared__) | confirmed | mthreads-blog |
| Off-chip Memory | GDDR6 (S4000): 48 GB, 384-bit bus, 768 GB/s (16 Gbps per pin) | confirmed | videocardz-s4000, wccftech-s4000 |
| Off-chip Memory | GDDR6 (S3000): 32 GB, 256-bit bus, ~512 GB/s | confirmed | topcpu-s3000 |
| Off-chip Memory | MTT S5000 (PingHu Gen 4): 80 GB @ 1.6 TB/s — **media-reported only**; memory **type implied HBM but HBM generation not confirmed**; vendor publishes no memory spec. ⚠️ the 3.35 TB/s and 4.0 TB/s figures on the official S5000 page are H100-SXM and H20-class comparison baselines, NOT S5000 specs | reported | baike-s5000, tencentcloud-s5000, mthreads-s5000-page-cn |
| Off-chip Memory | HBM (Huashan, Huagang **Gen 5**): 8× HBM sites; capacity exceeds B200 (claimed) | announced | wccftech-huashan |
| Host Interface / Package | PCIe Gen5 x16: standard FHFL card; ~128 GB/s bidir host bandwidth (S4000) | confirmed | wccftech-s4000, videocardz-s4000 |
| Host Interface / Package | TSMC 12nm monolithic die (Chunxiao): 22B transistors; no 2.5D packaging | confirmed | topcpu-s4000 |
| Host Interface / Package | MTT S5000 form factors: OAM compute module (liquid- and air-cooled variants); MTT MGX 8-GPU modular platform (8 OAM modules over MTLink); MTT SGX5000 server (8× S5000). Host PCIe generation, packaging and process node **not disclosed** | confirmed | mthreads-s5000-page |
| Scale-up Interconnect | MTLink 1.0 (NVLink analog): 240 GB/s aggregate per GPU; 8 GPUs per KUAE D800 / MCCX D800 server (S4000 generation) | confirmed | trendforce-mtlink, tomshardware-10k |
| Scale-up Interconnect | MTT S5000 card-to-card interconnect: 784 GB/s — **media-reported only**; **MTLink generation number not disclosed** (do not assume "MTLink 2.0") | reported | baike-s5000, tencentcloud-s5000 |
| Scale-up Interconnect | MTT C256 supernode (WAIC 2026, first public demonstration 2026-07-17): claimed industry-first single-layer Scale-up network breaking the 64-card single-layer limit; 128 GPUs fully interconnected per standard cabinet, 256 across two cabinets; sub-microsecond card-to-card latency; 2U nodes; positioned for 10K–100K-card clusters. Per-link and aggregate bandwidth **not disclosed**; liquid cooling / blind-mate / RAS claims **not confirmed** | announced | ithome-c256, sina-waic2026 |
| Scale-up Interconnect | MTLink fabric scales to 10,000 GPUs in a single cluster (demonstrated 2024; largest Chinese GPU cluster); S5000 10,000-card clusters reported in commercial service 2026, incl. a thousand-card cluster training BAAI RoboBrain 2.5 | confirmed (2024) / reported (2026) | theregister-10k-cluster, tomshardware-10k, ithome-c256 |
| Scale-out Interconnect | Standard Ethernet / RoCE via host NIC; no proprietary scale-out ASIC | confirmed | theregister-10k-cluster |
| Scale-out Interconnect | MCCL handles inter-node collectives over Ethernet/RoCE; MTLink for intra-node | confirmed | theregister-10k-cluster |

## Source Keys Added 2026-08-08

| Key | URL |
|-----|-----|
| mthreads-s5000-page | https://en.mthreads.com/product/S5000 |
| mthreads-s5000-page-cn | https://www.mthreads.com/product/S5000 |
| github-torch-musa-releases | https://github.com/MooreThreads/torch_musa/releases |
| github-mooretheads-org | https://github.com/orgs/MooreThreads/repositories?sort=created (repo creation dates via GitHub API) |
| github-mutlass | https://github.com/MooreThreads/mutlass |
| ithome-c256 | https://www.ithome.com/0/978/856.htm |
| sina-waic2026 | https://finance.sina.com.cn/tech/roll/2026-07-19/doc-iniiiwhz3402415.shtml |
| tencentcloud-s5000 | https://cloud.tencent.com/developer/news/3588420 |
| baike-s5000 | https://baike.baidu.com/item/MTT%20S5000/67553280 (snippet-level only; HTTP 403 on direct fetch) |

**Confidence vocabulary note.** `reported` is used in the rows added on 2026-08-08 for figures that appear only in Chinese media / aggregators with a likely-common origin and are **not vendor-published**. They must not be upgraded to `confirmed` without a vendor or independent-measurement source.
