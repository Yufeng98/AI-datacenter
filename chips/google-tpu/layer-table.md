# Google TPU — Layer Mapping

*as_of: 2026-08-08*
*chip: google-tpu (v4 / v5e / v5p / v6e Trillium / v7 Ironwood / v8t Sunfish / v8i Zebrafish)*

| Layer | Component | Confidence | Sources |
|-------|-----------|------------|---------|
| **Software Layers** | | | |
| Framework Integration | JAX (jax.numpy, jit/grad/vmap/shard_map, jax.Array + NamedSharding) | confirmed | [JAX repo](https://github.com/jax-ml/jax), [JAX docs](https://docs.jax.dev/en/latest/) |
| Framework Integration | Flax / NNX (neural network modules, Linen legacy) | confirmed | [Flax repo](https://github.com/google/flax) |
| Framework Integration | Optax (composable gradient optimizers: Adam, Adafactor, LAMB) | confirmed | [Optax repo](https://github.com/google-deepmind/optax) |
| Framework Integration | Orbax (async GCS checkpointing, mesh-aware restoration) | confirmed | [Orbax repo](https://github.com/google/orbax) |
| Framework Integration | Grain (deterministic data loading, GCS/ArrayRecord, prefetch) | confirmed | [Grain repo](https://github.com/google/grain) |
| Framework Integration | Tokamax (production Pallas kernels: FlashAttention, MoE, RoPE) | confirmed | [MaxText docs](https://maxtext.readthedocs.io/) |
| Framework Integration | MaxText (open-source LLM reference: Gemma, Llama, DeepSeek, Mistral) | confirmed | [MaxText repo](https://github.com/AI-Hypercomputer/maxtext) |
| Framework Integration | PyTorch/XLA (torch_xla backend, torch.compile → StableHLO) | confirmed | [PyTorch/XLA SPMD blog](https://pytorch.org/blog/pytorch-xla-spmd/) |
| Framework Integration | Keras 3 (JAX backend, Pallas custom layer support) | confirmed | [Cloud TPU JAX AI Stack](https://docs.cloud.google.com/tpu/docs/jax-ai-stack) |
| Framework Integration | TensorFlow (tf2xla bridge; v4–v6e only; not supported on v7) | confirmed | [TPU software versions](https://docs.cloud.google.com/tpu/docs/runtimes) |
| Compiler / IR | XLA (Accelerated Linear Algebra compiler; fusion, layout, memory scheduling, dot tiling) | confirmed | [OpenXLA/XLA repo](https://github.com/openxla/xla) |
| Compiler / IR | jaxpr (JAX functional IR; abstract-value tracing; Python → jaxpr by jit) | confirmed | [JAX JIT docs](https://docs.jax.dev/en/latest/jit-compilation.html) |
| Compiler / IR | StableHLO (MLIR opset ~100 ops; backward/forward compat; portability layer) | confirmed | [StableHLO spec](https://openxla.org/stablehlo/spec) |
| Compiler / IR | HLO (XLA internal SSA graph; optimization passes applied here) | confirmed | [OpenXLA/XLA repo](https://github.com/openxla/xla) |
| Compiler / IR | LLO (Low Level Operations; TPU-specific IR; VLIW instruction packing substrate) | confirmed | [JAX to VLIW blog](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation) |
| Compiler / IR | Shardy (MLIR SPMD partitioner; replaced GSPMD March 2026; auto collective insertion) | confirmed | [Shardy overview](https://openxla.org/shardy/overview), [JAX migration](https://docs.jax.dev/en/latest/shardy_jax_migration.html) |
| Compiler / IR | Mosaic (TPU kernel compiler: Pallas IR → LLO → VLIW; MXU tiling + DMA pipeline) | confirmed | [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) |
| Op Library | *not applicable* — XLA operator fusion subsumes the op library role | not applicable | — |
| Kernel Library | Pallas (JAX kernel DSL: grid/BlockSpec/MemorySpace/pallas_call) | confirmed | [Pallas design doc](https://docs.jax.dev/en/latest/pallas/design/design.html) |
| Kernel Library | Mosaic (TPU kernel compiler backend: MXU tiling, VMEM double-buffering, VLIW scheduling) | confirmed | [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) |
| Kernel Library | Tokamax (production Pallas kernel library: FlashAttention, MoE, RoPE, generation-aware dispatch) | confirmed | [MaxText repo](https://github.com/AI-Hypercomputer/maxtext) |
| Runtime | libtpu.so (monolithic: XLA compiler + TPU kernel driver + ICI runtime; versioned with JAX) | confirmed | [libtpu PyPI](https://pypi.org/project/libtpu/), [TPU versions](https://docs.cloud.google.com/tpu/docs/runtimes) |
| Runtime | Compilation cache (on-disk, content-hash keyed; amortizes 30–120s compile time) | confirmed | [libtpu PyPI](https://pypi.org/project/libtpu/) |
| Runtime | **Compute Engine-native TPU provisioning (GA 2026-06-01)**: TPU VMs and slices provisioned/managed via standard Compute Engine instance and managed-instance-group APIs; custom OS images, boot-disk sizing; all consumption options (on-demand/spot/reservation) | confirmed | [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes) |
| Driver / Firmware | libtpu TPU kernel driver (PCIe DMA, interrupt handling, HBM buffer alloc, command queue; part of libtpu.so) | confirmed | [TPU VM architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) |
| Driver / Firmware | Standalone kernel module for TPU | not public | — |
| Communication | libtpu ICI runtime (ring AllReduce/AllGather/AllToAll/ReduceScatter on 3D/2D torus) | confirmed | [libtpu PyPI](https://pypi.org/project/libtpu/) |
| Communication | Shardy collective insertion (lax.psum→AllReduce, lax.all_gather→AllGather, etc.) | confirmed | [Shardy guide](https://openxla.org/shardy/getting_started_jax) |
| Communication | Multislice DCN AllReduce (inter-pod via datacenter network, Titanium IPU offload) | confirmed | [Multislice docs](https://docs.cloud.google.com/tpu/docs/v5e-training) |
| Assembler / ISA | TPU VLIW ISA | not public | — |
| Assembler / ISA | LLO (internal low-level IR; not user-visible; VLIW packing compiler substrate) | confirmed (internal) | [JAX to VLIW blog](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation) |
| Communication | CAE — Collectives Acceleration Engine (v8i only): on-chip tree-reduce silicon for small-tensor `cae_reduce` HLO; up to 5× lower on-chip collective latency for autoregressive decoding | confirmed | [TPU 8t/8i deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| **Hardware Layers** | | | |
| Compute Engine | MXU systolic array: 128×128 (v4/v5), 256×256 (v6e/v7); weight-stationary. **v8 MXU dimensions not disclosed** — 256×256 carried forward as an assumption | confirmed (v4–v7) / not disclosed (v8) | [Ironwood docs](https://docs.cloud.google.com/tpu/docs/tpu7x), [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Compute Engine | VPU (Vector Processing Unit): element-wise ops, concurrent with MXU | confirmed | [TPU architecture overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) |
| Compute Engine | SparseCore: embedding table lookup acceleration; 1/chip (v4), 2/chip (v6e), 4/chip (v7). v8: listed for **v8t only** ("SparseCore (Embeddings)"); per-chip count **not published**; **no SparseCore attributed to v8i** | confirmed (v4–v7) / partial (v8) | [Ironwood blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/), [TPU 8t/8i deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Compute Engine | **LLM Decoder Engine (v8t only)**: named specialized block in Google's v8 spec table alongside SparseCore; function, count, and performance **not disclosed** | confirmed (existence) / not disclosed (function) | [TPU 8t/8i deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Compute Engine | TensorCore (Google): full compute die unit (MXU+VPU+ScalarCore+VMEM+CMEM); v7/v8: 2/chip | confirmed | [Inside Ironwood blog](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack) |
| Compute Engine | FP8 hardware support (v7 Ironwood, first TPU generation). **FP8 on v8 not disclosed** — Google's v8 deep-dive discusses FP4 only | confirmed (v7) / not disclosed (v8) | [Ironwood docs](https://docs.cloud.google.com/tpu/docs/tpu7x) |
| Compute Engine | FP4 hardware support — OCP MX microscaling (v8t/v8i, first TPU generation); BF16 accumulation; 12.6 PFLOPS peak FP4/chip on v8t, 10.1 PFLOPS peak FP4/chip on v8i | confirmed | [TPU 8t/8i deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Compute Engine | CAE — Collectives Acceleration Engine: dedicated on-chip tree-reduce silicon (v8i only), physically on a **separate chiplet die** ("two Tensor Core on-core dies and one CAE on the chiplet die" per v8i chip) | confirmed | [TPU 8t/8i deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Data Path | VLIW 8-op/cycle static instruction packets (no dynamic scheduling, no OoO) | confirmed | [TPU v4 whitepaper arXiv:2304.01433](https://arxiv.org/abs/2304.01433) |
| Data Path | Weight-stationary dataflow: HBM→VMEM→MXU (weights held)→VMEM→HBM | confirmed | [TPU v4 whitepaper arXiv:2304.01433](https://arxiv.org/abs/2304.01433) |
| Data Path | DMA double-buffering: tile N+1 prefetch overlapped with tile N MXU compute | confirmed | [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) |
| On-chip Memory | VMEM SRAM scratchpad, per TensorCore: ~16 MB (v4/v5), ~32 MB (v6e), ~64 MB (v7 = ~128 MB/chip). **v8 published per chip only: 128 MB/chip (v8t, flat vs v7), 384 MB/chip (v8i, 3× v7)**; per-TensorCore split for v8 **not disclosed** | confirmed | [TPU architecture overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm), [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| On-chip Memory | CMEM scalar memory: loop counters, DMA descriptors, LUTs, ICI control registers | confirmed | [Pallas TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) |
| On-chip Memory | No L1/L2 cache hierarchy — all memory management explicit and compiler-driven | confirmed | [TPU architecture overview](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) |
| Off-chip Memory | HBM2e 95 GB / 2,765 GB/s (v5p) | confirmed | [TPU v5p docs](https://docs.cloud.google.com/tpu/docs/v5p) |
| Off-chip Memory | HBM3 144 GB / ~1.6 TB/s (v6e Trillium) | confirmed | [TPU v6e docs](https://docs.cloud.google.com/tpu/docs/v6e) |
| Off-chip Memory | HBM3e 192 GB / 7.4 TB/s (v7 Ironwood, 8 stacks); 1.77 PB per 9,216-chip superpod | confirmed | [Ironwood docs](https://docs.cloud.google.com/tpu/docs/tpu7x) |
| Off-chip Memory | **HBM3e 216 GB / 6,528 GB/s (v8t Sunfish); 2 PB per 9,600-chip superpod** | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), [ServeTheHome](https://www.servethehome.com/google-tpu-8i-for-inference-and-tpu-8t-for-training-announced/) |
| Off-chip Memory | **HBM3e 288 GB / 8,601 GB/s (v8i Zebrafish); 331.8 TB per 1,152-chip pod (physical; Google also states "up to 1,024 active chips" per pod)** | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Host Interface / Package | PCIe Gen 4 x16 (~64 GB/s) host-to-HBM path; TPU VM architecture (host CPU on-board) | confirmed | [TPU VM architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) |
| Host Interface / Package | **Google Axion (Arm Neoverse-V2) host CPU for v8t/v8i; replaces x86 hosts** | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Host Interface / Package | **v8i chiplet packaging: 2 TensorCore on-core dies + 1 CAE chiplet die per chip** (first TPU generation with a Google-described multi-die organization) | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Host Interface / Package | **Process node and ASIC design partner for v8t/v8i** — Google discloses neither; TSMC 2nm / Broadcom (8t) / MediaTek (8i) are press-reported only | not disclosed (vendor) / press-reported | [Wccftech](https://wccftech.com/google-splits-tpuv8-strategy-two-chips-broadcom-training-mediatek-inference-duties/) |
| Host Interface / Package | libtpu.so as the sole software-hardware interface (compiler+driver+ICI bundled) | confirmed | [libtpu PyPI](https://pypi.org/project/libtpu/) |
| Scale-up Interconnect | ICI 3D torus: v5p 4,800 Gbps/chip (8,960 chips); v7 9,600 Gbps/1.2 TB/s (9,216 chips); **v8t 19.2 Tbps/2.4 TB/s (9,600 chips)** | confirmed | [Ironwood docs](https://docs.cloud.google.com/tpu/docs/tpu7x), [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Scale-up Interconnect | ICI 2D torus: v5e/v6e, 256 chips max | confirmed | [v5e docs](https://docs.cloud.google.com/tpu/docs/v5e) |
| Scale-up Interconnect | OCS Optical Circuit Switches (v4 + v8t): 136×136 MEMS; reconfigurable 3D torus; twisted torus +70% bisection BW | confirmed | [TPU v4 whitepaper arXiv:2304.01433](https://arxiv.org/abs/2304.01433) |
| Scale-up Interconnect | **Boardfly (v8i): high-radix topology; full-mesh boards aggregated into board-of-boards groups (36 groups of 8 boards); 7-hop diameter at 1,024 chips (vs 16-hop on 3D torus); 1,152 chips connected / up to 1,024 active per pod; 19.2 Tbps/chip** | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) |
| Scale-up Interconnect | 4×4×4 cube (64 chips): copper DAC intra-cube, optical fiber inter-cube | confirmed | [Ironwood docs](https://docs.cloud.google.com/tpu/docs/tpu7x) |
| Scale-out Interconnect | Multislice: up to 16 ICI pods via DCN; ~147,456 chips at v7 scale | confirmed | [Multislice docs](https://docs.cloud.google.com/tpu/docs/v5e-training) |
| Scale-out Interconnect | **Virgo Network (v8t): optical scale-out fabric; up to 134K chips per DC; up to ~1M chips multi-DC** | confirmed | [TPU v8 deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), [fundaai](https://fundaai.substack.com/p/researchtpu-8t8i-and-virgo-network) |
| Scale-out Interconnect | Titanium IPU: Google SmartNIC/DPU for DCN AllReduce offload and RDMA (offloads Virgo cross-pod reductions on v8) | confirmed | [AI Hypercomputer blog](https://cloud.google.com/blog/products/compute/updates-to-ai-hypercomputer-software-stack/) |
