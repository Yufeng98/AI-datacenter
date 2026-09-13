# Codex Review — Enflame search-results.md

*reviewed_by: Codex (gpt-5.4:high)*
*as_of: 2026-04-05*
*target: research/enflame/search-results.md*

---

## Findings

### 1. Categorization Bug — TopsTorch and TopsBlasOps are swapped

`search-results.md` places `TopsTorch` in `Op Library` and `TopsBlasOps` in `Kernel Library`. The investigation files (`software-stack.md`) correctly identify `TopsTorch` as Framework Integration and `TopsBlasOps` as Op Library. The Kernel Library should show the unpublished `TopsPlatform Kernel Library` as a placeholder.

**Fix**: Move TopsTorch to Framework Integration; TopsBlasOps to Op Library; add placeholder for TopsPlatform Kernel Library in Kernel Library.

### 2. Communication Layer Mixes Software and Hardware

The `Communication` section in `search-results.md` includes the `GCU-LARE Bridge Card Manual` — a hardware item that is correctly categorized in `Scale-up Interconnect` elsewhere in the same file. The bridge card should be removed from Communication; ECCL belongs in Communication.

### 3. Assembler / ISA Overstates Public Availability

GCU-CARE and GCU-DARE are described as ISA resources in `search-results.md`, but the investigation explicitly confirms there is no public assembler path and the ISA is not documented externally. The HC33 papers are architectural references, not ISA documentation. This layer should be annotated clearly as "no public ISA; HC33 slides are indirect architectural evidence only."

### 4. Weak Layers (Coverage is Present but Thin)

Layers relying only on generic portal roots or org URLs rather than direct artifacts:
- **Compiler / IR**: Supported primarily by `support.enflame-tech.com` root (generic landing) and HC33 paper. No direct TopsCC user guide URL.
- **Runtime**: Only `support.enflame-tech.com` root and HAMi docs — no direct TopsRuntime API reference.
- **Op Library / Kernel Library**: No official product page or direct library doc.
- **Scale-out Interconnect**: ECCL existence is confirmed but only via dependency manifests; no direct ECCL documentation or SmartCluster networking spec.
- **Off-chip Memory**: Largely inferred from secondary sources (Baidu Baike, Chinese tech press) rather than official product pages.

### 5. Ranking Quality Issues

- `Compute Engine` starts with ServeTheHome (secondary) before the IEEE HC33 paper (primary).
- `Compiler / IR` starts with a generic support landing page before the HC33 paper.
- `enflame-tech.com` official product pages are buried in `Other Resources` instead of being promoted into the relevant hardware layers.

---

## Coverage Verdict

| Layer | Status |
|-------|--------|
| Framework Integration | Strong |
| Compiler / IR | Partial |
| Op Library | Weak |
| Kernel Library | Weak |
| Runtime | Partial |
| Driver / Firmware | Partial |
| Communication | Partial |
| Assembler / ISA | Weak (no public ISA; sources are indirect) |
| Compute Engine | Strong |
| Data Path | Strong |
| On-chip Memory | Strong |
| Off-chip Memory | Partial |
| Host Interface / Package | Partial |
| Scale-up Interconnect | Strong |
| Scale-out Interconnect | Partial |

---

## Missing Obvious Resources

- Direct `support.enflame-tech.com` pages for TopsCC, TopsRuntime, ECCL, TopsTorch
- Official S60 and L600 product manual pages (not just T10/T20)
- HC33 official PDF (IEEE) as a primary link ahead of ServeTheHome mirrors
- `TopsPlatform Kernel Library` placeholder entry (identified in investigation but absent from search-results)
- Official SmartCluster networking/RoCE documentation for Scale-out Interconnect

---

## Suggested Follow-up Queries

```
site:support.enflame-tech.com TopsTorch
site:support.enflame-tech.com TopsRuntime OR libTopsRuntime
site:support.enflame-tech.com TopsCC 编译器
site:support.enflame-tech.com ECCL OR eccl
site:support.enflame-tech.com "GCU-LARE" SmartCluster RoCE
site:support.enflame-tech.com "CloudBlazer S60" 产品手册
site:support.enflame-tech.com "CloudBlazer L600" 产品手册
site:github.com/EnflameTechnology ECCL
site:github.com/EnflameTechnology TopsTorch
燧原 TopsRuntime 使用手册
燧原 TopsCC 编译器 用户手册
燧原 ECCL 集体通信
Enflame "Hot Chips 33" PDF GCU-LARE
Enflame SCORPIO S60 product brief PDF
```

---

## Action Items Applied to search-results.md

1. Moved TopsTorch to Framework Integration section
2. Moved TopsBlasOps to Op Library (from Kernel Library)
3. Added TopsPlatform Kernel Library placeholder in Kernel Library
4. Removed GCU-LARE Bridge Card from Communication (retained in Scale-up)
5. Added annotation to Assembler/ISA that no public ISA documentation exists
6. Promoted IEEE HC33 paper above ServeTheHome in Compute Engine ranking
7. Promoted HC33 paper above support.enflame-tech.com root in Compiler/IR ranking
