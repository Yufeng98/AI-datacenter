# Codex Review: Hygon DCU Search Results

*as_of: 2026-04-05*
*reviewed: research/hygon-dcu/search-results.md*

---

## Findings

**1. High: Framework Integration layer is incomplete.** PyTorch, TensorFlow, vLLM, and llama.cpp are either missing or only mentioned indirectly inside dcu-in-action. The software-stack.yaml expects those frameworks as first-class entries.

**2. High: Driver/Firmware layer is effectively uncovered and partly miscategorized.** HAMi Kubernetes plugins plus H3C BIOS/accessory pages are listed, but none is a direct source for the DCU kernel driver or firmware. HAMi is deployment/orchestration, not driver/firmware. Should explicitly state no public driver docs found.

**3. High: Host Interface / Package hardware layer lacks real packaging sources.** No source substantiates interposer/package construction; hw-architecture.yaml expects package details.

**4. Medium: Communication and scale-out sections contain non-technical/off-category items.** Server compatibility post under "Communication"; deployment tutorial and business-news article under "Scale-out Interconnect" — these don't document RCCL, RoCE, RDMA, or multi-node topology.

**5. Medium: Assembler/ISA section mislabeled.** Sources listed are microarchitecture/performance, not assembler or ISA documentation. Should say "public ISA docs not found" and treat these as indirect architecture evidence.

**6. Medium: Software layers have broad placeholders.** DTK portal reused for compiler, op library, kernel library, runtime, communication. Missing component-specific resources: hipify, rocprofv2, rocm-gdb, roctracer, hipSPARSE, hipFFT, hipRAND, rocPRIM, ROCr/HSA runtime.

---

## Coverage Summary

| Layer | Status |
|-------|--------|
| Framework Integration | partial |
| Compiler / IR | partial |
| Op Library | partial |
| Kernel Library | partial |
| Runtime | partial |
| Driver / Firmware | weak |
| Communication | partial |
| Assembler / ISA | weak |
| Compute Engine | good |
| Data Path | fair |
| On-chip Memory | fair |
| Off-chip Memory | fair (weak sourcing) |
| Host Interface / Package | weak |
| Scale-up Interconnect | fair |
| Scale-out Interconnect | weak |

---

## Obvious Missing Resources

- Direct PyTorch-on-DCU resource
- Direct TensorFlow-on-DCU resource
- Direct vLLM or llama.cpp DCU port resource
- hipify / CUDA migration tool resource
- DTK profiling/debugging: rocprofv2, rocm-gdb, roctracer, rocm-smi
- Direct resources for hipSPARSE, hipFFT, hipRAND, rocPRIM
- ROCr/HSA runtime on DCU
- Real driver/firmware source (even release notes or install docs)
- Package/interposer source
- Direct scale-out networking source covering RoCE/RDMA or cluster topology

---

## Ranking Quality

Ranking is strongest where official docs are available. Concrete problems:
- EEWorld, Digitimes, and compatibility pages are too high inside technical sections
- HAMi is overused as evidence for runtime, driver, host interface, and off-chip memory
- DTK portal repeated as generic reference instead of component-specific docs

Reorder rule: Official Hygon/DTK docs → Official framework/vendor docs → Academic papers → Community tutorials → News/investor/compatibility (last or "Other Resources")

---

## Suggested Search Queries

```
site:developer.sourcefind.cn 海光 DTK PyTorch
site:developer.sourcefind.cn 海光 DTK TensorFlow
site:developer.sourcefind.cn 海光 DCU vLLM
site:developer.sourcefind.cn 海光 DCU llama.cpp
site:github.com Hygon DCU PyTorch DTK
site:github.com Hygon DCU TensorFlow DTK
site:github.com Hygon DCU vLLM
site:github.com Hygon DCU llama.cpp
site:developer.sourcefind.cn hipify 海光 DTK
site:developer.sourcefind.cn rocprofv2 OR rocm-gdb OR roctracer OR rocm-smi 海光
site:developer.sourcefind.cn hipSPARSE OR hipFFT OR hipRAND OR rocPRIM 海光
site:developer.sourcefind.cn ROCr OR HSA 海光 DCU
site:hygon.cn 海光 深算 驱动 固件
site:hygon.cn 海光 深算 PCIe Gen4 x16
site:hygon.cn 海光 深算 封装 HBM2 中介层
site:developer.sourcefind.cn RCCL RoCE RDMA 海光
site:project-hami.io hygon dcu k100_ai
海光 DTK CUDA 迁移 hipify
海光 深算 RCCL RoCE
海光 深算 驱动 安装
海光 深算 封装 HBM2
```

---

*Reviewed by Codex (gpt-5.4:high) against repo layer model in software-stack.yaml and hw-architecture.yaml*
