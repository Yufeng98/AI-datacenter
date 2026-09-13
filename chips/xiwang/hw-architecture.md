# Xiwang (曦望) GPU Hardware Architecture Reference

*as_of: 2026-08-08*
*chip: xiwang*
*device_class: AI Accelerator (China, 曦望)*
*generations: S1 (DSA, ~2019) / S2 (GPGPU, 2024) / S3 (inference GPU, announced Jan 2026)*
*product families: 启望 GPUs · 智望 cards/modules · 辰望 servers · 寰望 supernodes · 熙望 workstations · SIRE software stack*

---

## Chip Identity

| Attribute | S1 (~2019) | S2 (Current) | **S3 (announced Jan 2026)** |
|-----------|-----------|--------------|-----------------|
| Architecture brand | Not disclosed | Not disclosed | Not disclosed |
| Type | DSA cloud/edge inference | Full GPGPU; marketed 2026-08 as inference + fine-tuning | Inference-specialized GPU |
| Process | Not disclosed | TSMC 7nm | **Not disclosed** |
| Packaging | Not disclosed | 2.5D CoWoS | **Not disclosed** |
| TDP | Not disclosed | 350–450 W | **Not disclosed** |
| Form factor | Not disclosed | 智望 S2-X1 PCIe FHFL; **智望 S2-M1 OAM module** | **Not disclosed** |
| Host interface | Not disclosed | PCIe Gen5 x16 | **PCIe Gen6** |
| Memory | Not disclosed | 64 GB, **type not disclosed** | **LPDDR6** (back-compatible LPDDR5X); capacity/BW **not disclosed** |
| Precision | Not disclosed | FP32/TF32/BF16/FP16/INT8 | FP32/FP16/FP8/INT8/**FP4** |
| Scale-up | Not disclosed | **SRLink** (proprietary; BW/topology not disclosed) | Not disclosed (寰望 SC3-256 supernode is the rack-scale product) |
| Compiler target string | — | `tang:S2`, triple `stcu-unknown-tang` | triple `stcuv2`, separate `*_S3.bc` libdevice |
| Status | Production; >20,000 cumulative units (company claim) | **Volume production** — 规模化量产 / 万片级量产 (company claim); 1,000 units in 2024 | **Announced only** — no public evidence of tape-out, sampling or shipment |

*Updated 2026-08-08. The S3 LPDDR6 / PCIe Gen6 / FP4 disclosures and the SIRE, TANG and SRLink names all post-date the 2026-04-05 baseline; the compiler target strings come from the FlagTree `third_party/sunrise` backend (public 2026-01-23, Triton 3.6 since 2026-06-30).*

---

## Compute Engine

```
Xiwang S2 GPU  [TSMC 7nm CoWoS]
├── Compute Subsystem  [details NOT PUBLIC]
│   ├── Architecture: SIMT GPGPU (fully self-developed)
│   ├── Compute unit count: NOT PUBLIC
│   ├── Warp size: 32  (from FlagTree sunrise backend, 2026-08-08)
│   ├── Tensor/matrix units: present (inferred)
│   ├── Precision: FP32 / TF32 / BF16 / FP16 / INT8
│   └── ISA: proprietary; LLVM triple stcu-unknown-tang
│
├── On-Chip Memory  [details NOT PUBLIC]
│   ├── Type: cache or scratchpad — NOT PUBLIC
│   └── Capacity: NOT PUBLIC
│
├── 64 GB off-chip memory
│   ├── Type: NOT DISCLOSED  (CoWoS packaging implies HBM but the vendor has never stated it)
│   └── Bandwidth: "higher than A100 2.0 TB/s" (qualitative only)
│
├── PCIe Gen5 x16 Host Interface  (128 GB/s)
└── SRLink scale-up fabric  (proprietary; BW/topology NOT PUBLIC)
```

```
Xiwang S3 GPU  [announced Jan 2026 — no public tape-out]
├── Compute Subsystem  [details NOT PUBLIC]
│   ├── Architecture: inference-optimized GPGPU; unit count NOT PUBLIC
│   ├── Precision: FP32 / FP16 / FP8 / INT8 / FP4
│   ├── Peak FP4: "P 级" (petaFLOPS-class) — no TFLOPS figure published
│   └── ISA: LLVM triple stcuv2; separate *_S3.bc libdevice
│
├── LPDDR6 off-chip memory  (back-compatible LPDDR5X)   ← NOT HBM
│   ├── Capacity: NOT DISCLOSED
│   └── Bandwidth: NOT DISCLOSED
│
├── PCIe Gen6 Host Interface  ("100% host-device BW improvement", vendor claim)
│
└── 寰望 SC3-256 SuperPOD  (rack-scale; liquid-cooled; PD-disaggregated; large-EP)
    └── Chip count NOT STATED by vendor — "256" appears only in the product name
```

### Precision Support

| Format | S2 | S3 | Notes |
|--------|----|----|-------|
| FP32 | Yes | Yes (vendor-stated) | |
| TF32 | Likely (inferred) | Not stated | |
| BF16 | Yes | Not stated on the S3 page | The S3 precision list is FP32/FP16/FP8/INT8/FP4; BF16 is **no longer inferred** |
| FP16 | Yes | Yes (vendor-stated) | |
| INT8 | Yes | Yes (vendor-stated) | `min_dot_size` (8,8,16) for INT8 in the Triton path |
| FP8 | Not confirmed | **Yes (vendor-stated)** | Triton path exposes only `fp8e5` (E5M2) |
| FP4 | Not confirmed | **Yes (vendor-stated)** | Peak given only as "P 级" (petaFLOPS-class) |

### Performance Claims (Company, Unverified)

| Metric | S2 | S3 vs S2 | Provenance |
|--------|----|---------|-----------|
| FP32 | Exceeds A100 SXM (312 TFLOPS ref) | — | Company claim |
| BF16 | Approaches H100 SXM (1,979 TFLOPS ref) | — | Company claim |
| Single-chip inference performance | — | **"5×"** (current site; was "3×+" on the 2025-12-11 snapshot) | Company claim; the claim itself was raised between Dec 2025 and Aug 2026 |
| Per-token cost | — | −90% | Company claim, unchanged since Dec 2025 |
| GEMM operator efficiency | — | ~99% | **仿真实测 — simulation-measured**, not silicon |
| FlashAttention operator efficiency | — | ~98% | **仿真实测 — simulation-measured**, not silicon |
| SC3-256 throughput | — | ">20×" | 数据来源于曦望实验室 (vendor lab data) |

No independent benchmarks, and no MLPerf submissions, exist as of 2026-08-08.

---

## Memory Hierarchy

| Level | S2 | **S3** | Notes |
|-------|-----|-------|-------|
| On-chip SRAM | NOT PUBLIC | NOT PUBLIC | Type (cache/scratchpad) not disclosed for either part |
| Off-chip type | **NOT DISCLOSED** | **LPDDR6** (back-compatible LPDDR5X) | S2's type has never been vendor-stated; CoWoS *suggests* HBM but that is an inference, not a spec. Do not back-fill LPDDR onto S2. |
| Off-chip capacity | 64 GB | NOT DISCLOSED | |
| Bandwidth | NOT PUBLIC | NOT DISCLOSED | S2: "higher than A100" qualitative only |
| ECC support | Unknown | Unknown | — |

**Why LPDDR6 matters (S3).** Xiwang is the first Chinese GPU vendor to claim an LPDDR6-attached datacenter inference GPU (国内首款挂载 LPDDR6 的 GPU). Choosing commodity LPDDR over HBM trades peak bandwidth for capacity-per-dollar and, critically, removes dependence on the HBM supply chain that US export controls have constricted for Biren, Enflame and MetaX. The trade-off cannot be quantified here: the vendor publishes **no capacity and no bandwidth figure**, so whether S3 can feed its own FP4 datapath is unknowable from public data.

---

## Interconnect

| Component | S2 | **S3** |
|-----------|-----|-------|
| Host interface | PCIe Gen5 x16 (128 GB/s confirmed) | **PCIe Gen6** (vendor claims "100% host-device bandwidth improvement"; lane count not stated) |
| Scale-up interconnect | **SRLink** (proprietary, 自研; 自研 SRLink 等高速互联技术) — BW and topology NOT PUBLIC | Not separately named; SRLink assumed to carry forward but **not vendor-confirmed for S3** |
| Scale-out | Standard Ethernet/IB (assumed) | Assumed; not stated |
| Collectives library | **PCCL** (Sunrise Collective Communications Library; FlagCX-integrated) | PCCL assumed |
| Max system scale | Not confirmed | **寰望 SC3-256 超节点** — see below |

### 寰望 SC3-256 SuperPOD (S3 rack-scale, disclosed mid-2026)

| Attribute | Value |
|-----------|-------|
| Product name | 寰望 SC3-256 超节点 / "Rise SC3-256 SuperPOD" |
| Cooling | Liquid |
| Serving architecture | **PD-disaggregated** (prefill/decode separation) + large **Expert Parallel (EP)** deployment |
| Target workload | 万亿乃至更大参数量 (trillion-parameter and larger) multimodal MoE inference |
| **Chip count** | **NOT STATED by the vendor** — "256" appears only in the product name and must not be recorded as a specification |
| Link technology | NOT DISCLOSED |
| Vendor claims | TCO down "an order of magnitude" at equal performance; communication latency down an order of magnitude; ">20× throughput advantage" (数据来源于曦望实验室 — vendor lab data) |

---

## Packaging

| Attribute | S2 | **S3** |
|-----------|-----|-------|
| Package type | 2.5D CoWoS (TSMC confirmed) | NOT DISCLOSED (LPDDR6 attach does not require 2.5D interposer) |
| Dies | Unknown (monolithic likely; chiplet possible) | NOT DISCLOSED |
| Die area | NOT PUBLIC | NOT DISCLOSED |
| Transistors | NOT PUBLIC | NOT DISCLOSED |

---

## Compiler-Visible Architecture (from the FlagTree `sunrise` backend)

*Added 2026-08-08. These are the first third-party-inspectable architecture facts for this chip. They describe what the open Triton path exposes and are not a complete hardware specification.*

| Property | S2 | S3 |
|----------|----|----|
| Target string | `tang:S2` | — (S3 support is compiler plumbing only; the FlagTree user manual says the backend is "Available for S2") |
| LLVM target triple | `stcu-unknown-tang` | `stcuv2` |
| libdevice bitcode | default | separate `*_S3.bc` ("libdivice库,需要区分S2/S3") |
| Warp size | **32** | assumed 32; not separately stated |
| Default `num_warps` / `num_stages` | 4 / 3 | — |
| FP8 dtypes exposed | `fp8e5` (E5M2) only | — |
| `min_dot_size` | (8, 8, 16) INT8; (8, 8, 4) otherwise | — |
| Dot input precision | `ieee` only | — |
| Device math library | AMD-style OCML (`__ocml_*`) | — |
| Object bundling | `clang-offload-bundler`, `/usr/local/tangrt/toolchains/llvm/prebuilt/linux-<arch>/` | — |

---

## Foundry and Export Risk

| Risk Factor | Status |
|-------------|--------|
| Foundry | TSMC 7nm (CoWoS confirmed for S2) |
| US export controls on advanced packaging | **Risk**: CoWoS is TSMC advanced packaging; subject to US export scrutiny |
| HBM supply | **Risk for S2** (memory type undisclosed; CoWoS suggests HBM). **Mitigated for S3**: LPDDR6 is commodity mobile DRAM and is not subject to the HBM-specific restrictions that constrain Chinese accelerator vendors — this is the strategic point of the S3 memory choice. |
| Entity List status | Xiwang: **NOT on Entity List as of 2026-08-08** |
| S3 foundry | **Not disclosed** (no node, no foundry, no packaging statement) |

---

## Product Timeline

| Date | Event |
|------|-------|
| ~2019 | S1 DSA chip taped out (within SenseTime) |
| 2022 | S1 in production; cumulative >20,000 units by 2025 |
| July 2023 | S2 "first light" (initial silicon validation) |
| 2024 | S2 mass production: 1,000 units |
| July 2025 | S2 publicly disclosed; S3 roadmap announced |
| **2026-01-23** | FlagTree adds the `sunrise` Triton backend (Triton 3.4, CI/CD) — first public Xiwang open-source presence |
| January 2026 | S3 formally announced with FP8/FP4; vendor news item 曦望完成近30亿融资 (≈RMB 3B cumulative) |
| **2026-01-28** | Executive statement (co-CEO 王勇): S3 R&D complete, tape-out mid-2026, MP end-2026; roadmap S4 (2027, high-performance) and S5 (2028, security chip). **Unverified; no later source confirms the tape-out occurred.** |
| **H1 2026 (undated)** | Further funding round of >RMB 1B announced on the vendor news page; exact date and cumulative total **not confirmed** |
| **2026-05-21** | Wayback snapshot: SC3-256 present in site navigation (placeholder link); still no SIRE, no LPDDR6, no PCIe Gen6 |
| **2026-06-01 / 06-10** | FlagCX adds Sunrise torch plugin support, then runtime vendor detection |
| **2026-06-30** | FlagTree `sunrise` backend upgraded to Triton 3.6 with CI/CD |
| **2026-08-06** | Vendor site CMS assets republished carrying SIRE, LPDDR6, PCIe Gen6, FP4, SC3-256 and the "5×" performance claim |
| 2026 | S2 at 万片级 (10,000-unit-class) volume per company claim; **S3 remains announced-only with no public tape-out evidence** |
