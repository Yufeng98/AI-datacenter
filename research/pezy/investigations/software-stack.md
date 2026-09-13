# PEZY Computing — Software Stack Investigation

*as_of: 2026-08-08 (original investigation 2026-04-05; see section 12 — Investigation Update 2026-08-08)*

## Summary

PEZY Computing's software stack is built on a proprietary OpenCL 1.2-based programming model called **PZCL** (PEZY Computing Language / PEZY Computing Layer). The stack targets HPC developers familiar with OpenCL and GPU programming, while also offering MPI-layer integration for large-scale scientific computing. The stack is narrow by modern AI standards: there is no CUDA-analog tensor compiler, no cuDNN-equivalent, and no public MLIR/LLVM frontend. The stack's gate is access-controlled: PZCL API documentation is restricted and requires contacting PEZY directly.

> ⚠️ **Corrected 2026-08-08.** This summary previously read "no public AI framework integration (PyTorch, JAX, etc.) has been announced." **That was wrong.** PEZY announced PyTorch support for the PEZY-SC series on 2025-06-06, with vLLM/TGI/Transformers/Accelerate/DeepSpeed/Diffusers reported functional on PEZY-SC3 — but no public artifact (repo, tag, version, benchmark, docs) exists for it. See **§12** for the corrected wording. JAX, TensorFlow and ONNX Runtime remain unannounced.

---

## 1. Framework Integration

### AI/ML Frameworks

> ⚠️ **This table is superseded — see §12.3 (2026-08-08).** The "PyTorch: Not supported" row below was a pre-existing error; PEZY announced PyTorch support for the SC series on 2025-06-06. Struck rows retained for history.

| Framework | Support (as originally recorded 2026-04-05) |
|-----------|---------|
| PyTorch | ~~Not supported (no backend announced)~~ → **announced 2025-06-06, unverifiable** (§12) |
| JAX | Not supported |
| TensorFlow | Not supported |
| ONNX Runtime | Not supported |

PEZY SC4s added BF16 support "as a nod to the current AI boom." The chip remains HPC-first, and PEZY's own 2026 announcements are all bioinformatics on SC3 (§12.6) — but the claim that "no ML framework integration has been announced" is retracted (§12.1).

### HPC Frameworks
| Framework | Support |
|-----------|---------|
| MPI | Yes — used for multi-node PEZY-equipped supercomputer clusters |
| OpenMP | Limited / not official |
| BLAS / LAPACK | Via hand-written or PZCL-accelerated implementations |
| Custom scientific codes | Primary target (n-body, lattice QCD, genomics, fluid dynamics) |

---

## 2. Programming Model: PZCL

### What PZCL Is

PZCL (PEZY Computing Language, also called PEZY Computing Layer) is PEZY's primary programming interface. It is:

- Based on **OpenCL 1.2** — PZCL functions and APIs mirror the OpenCL 1.2 standard
- Proprietary extensions on top of OpenCL 1.2 for PEZY-specific features (scratchpad management, thread-group swapping, MIMD scheduling)
- Access-controlled: PZCL API documentation is not publicly available; developers must request access from PEZY

### OpenCL 1.2 Foundation

Like OpenCL, PZCL uses:
- **Host code** (C/C++ on the AMD EPYC host): context/queue/buffer management, kernel launch
- **Kernel code** (PZCL/OpenCL C): runs on PEZY PEs
- Platform model: host + accelerator device model

The OpenCL 1.2 base makes the conceptual model familiar to anyone who has written GPU compute code. Key differences for PEZY:
- No SIMD warp/wavefront grouping — all PEs are fully independent MIMD
- SMT8 threading per PE is exposed to the programmer via thread-group primitives
- Scratchpad (24 KB/PE + shared at village level) is explicitly managed, unlike GPU shared memory which relies on programmer __shared__ declarations but has similar semantics
- No hardware cache hierarchy in older gens (SC3 and earlier had limited L2); SC4s adds 64 MB L3

### Thread Hierarchy in PZCL

PZCL maps to the PEZY hardware hierarchy as:
```
OpenCL work-item    → PEZY thread (one of 8 SMT threads in a PE)
OpenCL work-group   → PEZY PE or village (4 PEs sharing scratchpad)
Global work size    → across all 2,048 PEs × 8 threads = 16,384 threads
```

The fine-grained multithreading (hardware selects a different thread every cycle within a group of 4) is transparent to the programmer — they write as if threads execute independently.

---

## 3. Compiler / IR

PEZY has not published details of its compiler backend. Based on known information:

| Component | Details |
|-----------|---------|
| Host compiler | Standard x86-64 C/C++ (GCC/Clang for AMD EPYC host) |
| Kernel compiler | Proprietary PZCL compiler — compiles OpenCL-C kernel code to PEZY ISA |
| ISA | Proprietary PEZY PE ISA (MIPS-derived in SC2; SC3/SC4 ISA undisclosed) |
| IR | Not public |
| Auto-vectorization | Compiler likely auto-vectorizes to 4-wide FP64 / wider BF16 SIMD |
| LLVM-based? | Unconfirmed; likely given industry trends |

**MIPS heritage (SC2)**: The PEZY-SC2 used MIPS64 architecture cores (with Imagination Technologies Warrior CPU integration). SC3 and SC4 ISA are not publicly documented.

---

## 4. Op Library / Kernel Library

Unlike NVIDIA (cuDNN, cuBLAS, CUTLASS) or AMD (MIOpen, rocBLAS), PEZY has **no public math library** equivalent. Scientific computing users write custom PZCL kernels for their workloads. This is consistent with the academic HPC supercomputer market, where users typically have domain-specific codes rather than off-the-shelf DNN kernels.

There is no:
- GEMM library (cuBLAS equivalent)
- Convolution library (cuDNN equivalent)
- Collective communication library (NCCL equivalent) — MPI handles collectives

---

## 5. Runtime

### PZCL Runtime
The PZCL runtime provides:
- OpenCL-analog platform/device/context management
- Command queue execution (queue kernel launches to the PEZY device)
- Memory management: host ↔ HBM3 DMA transfers; scratchpad management
- Kernel loading: compiled PZCL binary → PEZY PE instruction memory

### Management Processor (SC4s)
The new RISC-V quad-core management processor runs Linux on-chip and handles:
- PE initialization and boot
- Memory mapping
- PCIe transaction management (to AMD EPYC host or as standalone)
- OS services for the PZCL runtime

In prior generations (SC1–SC3), this role was handled by the AMD EPYC host CPU running Linux, with the PEZY chip operating as a PCIe coprocessor.

---

## 6. Driver

| Component | Details |
|-----------|---------|
| Linux kernel module | Proprietary PEZY PCIe driver |
| Interface | PZCL runtime layer communicates with driver via ioctl |
| Host OS | Linux (AMD EPYC host or, for SC4s, RISC-V on-chip) |
| RISC-V management (SC4s) | Runs Linux natively on Rocket Core; manages PCIe endpoint |

---

## 7. Communication (Multi-chip / Multi-node)

### Within a Node (4× SC4s)
- Communication between 4 SC4s chips in one node is via the AMD EPYC host's PCIe fabric (peer-to-peer DMA through host PCIe switch, or via host memory copy)
- No proprietary chip-to-chip interconnect (unlike NVLink, xGMI, HCCS)
- Each SC4s has its own HBM3 — no shared memory between chips at hardware level

### Between Nodes (Cluster / Supercomputer)
- InfiniBand (400 Gb/s NDR in ZettaScaler 4.0 nodes)
- MPI over InfiniBand (OpenMPI, Intel MPI)
- Standard HPC cluster collective communication (MPI_Allreduce, etc.)

### Comparison
| Chip | Inter-chip fabric | MPI | Notes |
|------|------------------|-----|-------|
| PEZY-SC4s | None (PCIe only) | Standard MPI / IB | No proprietary scale-up |
| NVIDIA H100 | NVLink 4 (900 GB/s) | NCCL + MPI | Strong scale-up |
| AMD MI300X | xGMI / Infinity Fabric | RCCL + MPI | Strong scale-up |
| Google TPU v7 | ICI 3D torus | XLA/JAX | Strong scale-up |

---

## 8. Assembler / ISA

| Feature | Details |
|---------|---------|
| ISA | Proprietary PEZY PE ISA (undisclosed; MIPS64-derived for SC2) |
| Virtual ISA | None (no PTX equivalent) |
| Assembler | Not publicly available |
| PZCL output | Compiled binary in PEZY's proprietary format |

---

## 9. Software Stack Summary Diagram

```
Scientific Application (Fortran/C/C++ + MPI)
        │
        ▼
PZCL Host API (OpenCL 1.2 based; access-controlled)
        │
        ├─── Kernel Compilation (PZCL Compiler → PEZY PE binary)
        │
        ▼
PZCL Runtime (queue management, memory transfer)
        │
        ▼
PEZY PCIe Driver (Linux kernel module)
        │
        ├─── PCIe Gen 5 → SC4s RISC-V Management Processor → PE array
        │
        ▼
PEZY-SC4s Hardware (2,048 MIMD PEs, 96 GB HBM3)
        │
        ▼
ZettaScaler 4.0 Node (4× SC4s + AMD EPYC 9555P + 400 Gb/s NDR IB)
        │
        ▼
InfiniBand (MPI → multi-node HPC cluster)
```

---

## 10. Toolchain Access and Openness

| Component | Openness |
|-----------|---------|
| PZCL API docs | Closed — requires PEZY contact |
| PZCL compiler | Closed — proprietary |
| PZCL runtime | Closed — proprietary |
| Tutorial (system-level) | Partially public (pezy-mokumoku.github.io) |
| RISC-V Rocket Core | Open-source (SiFive/UC Berkeley — but PEZY's integration is proprietary) |
| MPI libraries | Open (OpenMPI, Intel MPI — standard HPC) |
| Hardware specs | Partially public (Hot Chips slides, company website) |

---

## 11. Software Stack Gaps (vs AI-era competitors)

PEZY's software stack is built for HPC, not AI. Key gaps as of Q1 2026:

1. ~~**No ML framework backend**: PyTorch, JAX, TensorFlow all lack PEZY support~~ — **RETRACTED 2026-08-08.** Replaced by: *ML framework backend is announced but unverifiable* — PyTorch plus vLLM/TGI/Transformers/Accelerate/DeepSpeed/Diffusers were announced for the SC series on 2025-06-06 with SC3 demonstrations, but no artifact, version, benchmark or documentation is public. JAX, TensorFlow and ONNX Runtime remain unannounced. See §12.4.
2. **No MLIR/LLVM public path**: No community compiler infrastructure
3. **No tensor operation library**: No cuDNN/MIOpen equivalent
4. **No INT4/FP8 support**: SC4s supports INT8 and BF16, but no low-precision quantization for AI inference at FP8 or INT4
5. **No cloud deployment**: PEZY systems are on-premises HPC clusters only; no cloud API
6. **OpenCL 1.2 base** (2011 standard): old by AI toolchain standards
7. **Access-controlled SDK**: creates barrier to developer adoption

These gaps make PEZY's SC4s non-competitive for AI training/inference workloads despite the BF16 addition, but the chip remains technically sound for traditional scientific HPC (physics simulations, genomics, chemistry).

---

## Resources

- PEZY Systems Tutorial (PZCL): https://pezy-mokumoku.github.io/tutorial/tutorial/
- Hot Chips 2025 Slides: https://www.pezy.co.jp/wp-content/uploads/2025/09/HC2025.PEZYComputing.NaoyaHatta.v06.pdf
- Chips and Cheese HC2025 Analysis: https://chipsandcheese.com/p/pezy-sc4s-at-hot-chips-2025
- Next Platform Analysis: https://www.nextplatform.com/2025/09/04/why-is-japan-still-investing-in-custom-floating-point-accelerators/
- PEZY-SC3 arXiv: https://arxiv.org/abs/2301.07510

---

## 12. Investigation Update — 2026-08-08 (Major Correction: PyTorch Backend Announced)

*Window covered: 2026-04-05 → 2026-08-08. This section **retracts a pre-existing error** in §1 and §11 of this document.*

### 12.1 Retraction: PEZY *has* announced ML framework support

Sections 1 and 11 above state that PyTorch, JAX and TensorFlow are unsupported and that "no ML framework integration has been announced." **The PyTorch half of that is wrong.**

PEZY announced on **2025-06-06** — a pre-baseline item the survey should already have carried — that the **PEZY-SC series supports PyTorch**, along with:

| Component | PEZY's claim |
|---|---|
| PyTorch | Supported on the PEZY-SC series |
| Accelerate | Reported functional |
| DeepSpeed | Reported functional |
| Transformers | Reported functional |
| vLLM | Reported functional |
| Text Generation Inference (TGI) | Reported functional |
| Diffusers | Reported functional |
| Hardware | PEZY-SC3 / ZettaScaler 3.0 |
| Models reported working | Gemma3, Llama3, Qwen2, Stable Diffusion 2, HuBERT, Vision Transformer |
| SC4s | Support "to follow with the SC4s release" |

### 12.2 The correct characterization: announced-but-unverifiable

The retraction does **not** mean PEZY has an open AI ecosystem. Independent search found **none** of the following:

- No PyTorch fork or upstream PEZY device backend
- No PZCL/OpenCL `PrivateUse1` backend registration
- No MLIR or LLVM path
- No package, release tag, or SDK version number
- No benchmark, no documentation, no public repository
- The PZCL SDK itself remains **access-controlled**

Standard wording adopted across the PEZY deliverables:

> PEZY announced a PyTorch backend for the PEZY-SC series in June 2025, with Accelerate/DeepSpeed/Transformers/vLLM/TGI/Diffusers reported functional on PEZY-SC3 and inference demonstrated for Gemma3, Llama3, Qwen2, Stable Diffusion 2, HuBERT and ViT. This is a vendor claim only: no public repository, release tag, version number, benchmark, or documentation for this backend could be located, and the PZCL SDK remains access-controlled. Treat the PyTorch path as announced-but-unverifiable rather than as an open ecosystem.

**The "no public ML ecosystem / access-controlled SDK" weakness therefore stands unchanged.** What changes is the factual claim about announcements, not the assessment of ecosystem maturity.

### 12.3 Corrected framework-support table (supersedes §1)

| Framework | Status as of 2026-08-08 | Evidence class |
|---|---|---|
| PyTorch | **Announced** for the PEZY-SC series (2025-06-06); demonstrated on SC3; SC4s support pending SC4s release | Vendor claim, unverifiable |
| vLLM / TGI / Transformers / Accelerate / DeepSpeed / Diffusers | **Announced** as functional on PEZY-SC3 | Vendor claim, unverifiable |
| JAX | Not supported (no announcement) | Confirmed negative |
| TensorFlow | Not supported (no announcement) | Confirmed negative |
| ONNX Runtime | Not supported (no announcement) | Confirmed negative |
| MPI | Supported — primary multi-node path | Confirmed |

### 12.4 Corrected gap list (supersedes §11 item 1)

§11 item 1 ("No ML framework backend: PyTorch, JAX, TensorFlow all lack PEZY support") is replaced by:

1. **ML framework backend is announced but unverifiable** — PyTorch and an inference stack (vLLM/TGI/Transformers/Accelerate/DeepSpeed/Diffusers) were announced for the SC series in June 2025 with SC3 demonstrations, but no artifact, version, benchmark, or documentation is public. JAX, TensorFlow and ONNX Runtime remain unannounced.

Items 2–7 of §11 (no public MLIR/LLVM path; no tensor op library; no INT4/FP8; no cloud deployment; OpenCL 1.2 base; access-controlled SDK) are **unchanged and re-confirmed** by this scan.

### 12.5 No SDK activity in the window

Re-checked 2026-08-08: **no PZCL or SDK release, no version number, no changelog, no new documentation, and no new public tutorial content** was found in the 2026-04-05 → 2026-08-08 window. The `pezy-mokumoku.github.io` tutorial remains the only partially public programming resource.

### 12.6 Workloads actually running on PEZY silicon in 2026

Both of PEZY's 2026 announcements are bioinformatics applications on PEZY-SC3, exercising the PZCL/HPC path rather than the announced PyTorch path:

- **pzMutect2** (2026-05-11) — somatic-variant calling on ZettaVEGA; vendor claim of 139× vs Mutect2 in GATK 4.2.6.1 (62 h 36 m 05 s → 26 m 57 s), >99.99% concordance via `bcftools isec`. Single workload, no independent replication; the release page does not name the processor (SC3/ZettaVEGA attribution inferred from the release slug and PEZY's 2024-12-26 announcement).
- **PZLAST-MAG** (2026-05-13) — public protein-sequence search server for metagenome-assembled genomes on PEZY-SC3, built with the National Institute of Genetics and the ROIS Data Science Common Use Platform Bio-generative AI R&D Center; 210,000+ MAGs, ~400 M sequences, ~100 B amino acids, ~5–15 min searches, accuracy PEZY describes as comparable to DIAMOND and MMseqs2. Paper: *Bioinformatics Advances*, DOI 10.1093/bioadv/vbag129.

### 12.7 Third-party kernel work

"Optimize Winograd Convolution for a Novel MIMD Many-core Architecture PEZY-SC3s" (PACT 2025, DOI 10.1109/PACT65351.2025.00045) is the only public **AI kernel** optimization work on PEZY hardware located to date — and it is third-party academic work on SC3s, not a PEZY-shipped library. "PEZY-DPM" (*Computer Physics Communications* 328:110319, DOI 10.1016/j.cpc.2026.110319, Crossref-indexed 2026-08-07) is an HPC Monte Carlo port on the same part. Neither implies a PEZY-supported op library; the §4 finding (no public math/op/kernel library) stands.

### Sources for this update

- https://www.pezy.co.jp/en/news/news20250606-pezysc-pytorch-generativeai/ (PyTorch / vLLM / DeepSpeed support on the PEZY-SC series)
- https://www.pezy.co.jp/en/news/ · https://www.pezy.co.jp/news/ (news indexes, fetched 2026-08-08)
- https://www.pezy.co.jp/news/news20260511-humangenome-zettavega-pzmutect2/
- https://www.pezy.co.jp/en/news/news20260513-pzlastmag/
- https://api.crossref.org/works/10.1016/j.cpc.2026.110319
