# Codex Review: Rain AI Search Results

*review_of: research/rain-ai/search-results.md*
*as_of: 2026-04-05 (review artifact — retained as-written)*
*model: gpt-5.4 (high effort)*

> **Superseding note (2026-08-08):** this review predates the vendor-status correction. Rain Neuromorphics
> Inc. (dba Rain AI) is **defunct / non-operating**; its patents were assigned to OpenAI Opco, LLC with a
> USPTO recordation date of **2025-10-23** (US20240281497A1, US20240143541A1, US20250045224A1), disclosed
> publicly ~2026-08-06. The review's central finding still holds and is now permanent: the "not public"
> layers are unresolved gaps that will never be filled, because the company no longer exists and its
> software stack was never released or open-sourced.

---

## Assessment

All 15 layers are present, but only 7 are actually evidence-backed. The other 8 are placeholders. So it is **structurally complete, not substantively complete**. Strongest layers are Compiler / IR, Assembler / ISA, Compute Engine, and Data Path. Framework Integration and Runtime are thin but probably reflect real public scarcity; Host Interface / scale-up / scale-out also still look genuinely undisclosed.

---

## Findings

### High Priority

- **Completeness is overstated** by the current presentation. Layers marked only as "Not public" should be treated as unresolved gaps, not covered layers. That matters most in software infra and system integration: Op Library, Kernel Library, Driver/Firmware, Host Interface / Package.

- **Several obvious public resources are missing:**
  - Rain's own [blog index](https://rain.ai/blog) — likely contains additional technical posts
  - [Be Like Water: Adaptive Floating Point for Machine Learning](https://proceedings.mlr.press/v162/yeh22a.html) — cited on Rain's Approach page for numeric formats
  - Andes [AX45MPV announcement](https://www.andestech.com/en/2022/12/08/andes-announces-risc-v-multicore-1024-bit-vector-processor-ax45mpv/) and [ACE 45-series support](https://www.andestech.com/cn/2023/03/23/andes-custom-extension-ace-supports-andescore-45-series-processors-to-provide-flexible-acceleratio-2/)
  - Rain patents directly reinforcing Compute Engine / On-chip Memory / training claims:
    - [Efficient matrix multiplication (US20210216610A1)](https://patents.google.com/patent/US20210216610A1/en)
    - [SRAM matrix multiplication network (US20240281497A1)](https://patents.google.com/patent/US20240281497A1)
    - [Continuous on-chip learning (US20240143541A1)](https://patents.google.com/patent/US20240143541A1/en)
    - [Error tolerant AI accelerators (US20240160693A1)](https://patents.google.com/patent/US20240160693A1/en)
  - For analog era: [Analog System Using Equilibrium Propagation for Learning (20210049504)](https://uspto.report/patent/app/20210049504), [Lithographic memristive array (WO2021262730A1)](https://patents.google.com/patent/WO2021262730A1/en), [Memristive nanowires exhibit small-world connectivity (PubMed)](https://pubmed.ncbi.nlm.nih.gov/30064118/)

### Medium Priority

- **Miscategorized entries:**
  - Rain IP Licensing entry is productization/business-model evidence, not Runtime
  - Synopsys tapeout success story is tapeout/implementation evidence, not Compute Engine
  - Funding news article used as On-chip Memory source is weak as a primary source for that layer

- **Ranking quality is mixed.** Official Rain and partner sources are correctly near the top of covered layers, but low-signal commentary (NeuromorphicCore.ai profile, asapdrew analysis) is ranked too prominently relative to stronger reporting such as [WIRED's OpenAI LOI story](https://www.wired.com/story/openai-buy-ai-chips-startup-sam-altman/) and official Rain posts like [Rain's Series A blog post](https://rain.ai/blog/rain-ai-raises-25-million-series-a) or [Rain's Mila partnership post](https://rain.ai/blog/rain-mila-partnership).

### Low Priority

- Rain's [Approach page](https://rain.ai/approach) cites [arXiv 2307.15063](https://dblp.org/rec/journals/corr/abs-2307-15063.html) for fine-tuning; this is a semantic-segmentation paper, not an obvious LoRA/on-device-fine-tuning reference. If added, annotate cautiously.

---

## Suggested Additional Search Queries

1. `site:rain.ai/blog OR site:rain.ai/approach OR site:rain.ai/products Rain AI compiler runtime SDK framework PyTorch ONNX`
2. `site:andestech.com/en AX45MPV ACE COPILOT Rain AI`
3. `site:andestech.com "AX45MPV" "Andes Custom Extension" OR "ACE-RVV"`
4. `site:patents.google.com "Rain Neuromorphics" ("matrix multiplication" OR "compute in-memory" OR SRAM OR "error tolerant" OR "local training")`
5. `site:pubmed.ncbi.nlm.nih.gov OR site:arxiv.org "Rain Neuromorphics" OR "Jack David Kendall" memristor equilibrium propagation`
6. `site:rain.ai/blog Mila Rain AI algorithms`
7. `site:wired.com OR site:reuters.com OR site:theinformation.com "Rain AI" OpenAI chips`
8. `site:rain.ai/blog "launch" OR "unveil" OR "hardware available" OR "customers" "Rain AI"`
9. `site:arteris.com FlexNoC 5 Rain AI mesh topology`
10. `site:rain.ai/careers OR site:linkedin.com/jobs "Rain AI" compiler runtime firmware RISC-V`

---

## Layer Coverage Summary

| Layer | Evidence-backed? | Gap Level |
|-------|-----------------|-----------|
| Framework Integration | No | High (likely truly not public) |
| Compiler / IR | Partial | Medium (Andes ACE docs available) |
| Op Library | No | High (not public) |
| Kernel Library | No | High (not public) |
| Runtime | No | High (not public) |
| Driver / Firmware | No | High (not public) |
| Communication | No | High (not public) |
| Assembler / ISA | Yes (RISC-V base) | Low |
| Compute Engine | Yes | Low |
| Data Path | Yes | Low |
| On-chip Memory | Partial | Medium |
| Off-chip Memory | No | High (not public) |
| Host Interface / Package | No | High (not public) |
| Scale-up Interconnect | No | High (not public) |
| Scale-out Interconnect | No | High (not public) |
