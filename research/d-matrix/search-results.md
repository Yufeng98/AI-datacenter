# d-Matrix Corsair — Search Results

*chip: d-matrix*
*device_class: Digital In-Memory Compute*
*search_date: 2026-04-05*

## Summary

d-Matrix is a startup that built Corsair, a Digital In-Memory Compute (DIMC) inference accelerator. The chip uses SRAM with embedded multipliers at the bit-cell level, eliminating the classic memory-wall bottleneck for LLM inference. Corsair was presented at Hot Chips 2025 (HC37). The company raised $275M and announced JetStream 400G I/O cards for rack-scale deployments.

## Resources Found

### Primary Technical Sources

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 1 | d-Matrix Corsair In-Memory Computing For AI Inference at Hot Chips 2025 | https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/ | Conference Coverage | High |
| 2 | d-Matrix Corsair: 256GB of LPDDR for AI Models (Chips & Cheese) | https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for | Deep-dive Analysis | High |
| 3 | d-Matrix Technical White Paper | https://d-matrix.ai/pdf/d-Matrix-WhitePaper-Technical-FINAL.pdf | Vendor White Paper | High |
| 4 | d-Matrix Takes On AI 'Memory Wall' with 3D Stacked In-Memory Compute (HPCwire) | https://www.hpcwire.com/2025/09/02/d-matrix-takes-on-ai-memory-wall-with-3d-stacked-in-memory-compute/ | Press | Medium |
| 5 | D-Matrix Transforms SRAMS For AI (TechInsights) | https://www.techinsights.com/blog/d-matrix-transforms-srams-ai-0 | Analyst | Medium |
| 6 | Corsair: An In-memory Computing Chiplet Architecture (IEEE Xplore) | https://ieeexplore.ieee.org/iel8/40/5210076/11108245.pdf | Peer-reviewed paper | High |
| 7 | d-Matrix Product Page | https://www.d-matrix.ai/product/ | Vendor | Medium |
| 8 | d-Matrix Corsair announcement (BusinessWire Nov 2024) | https://www.businesswire.com/news/home/20241119868644/en/d-Matrix-Unveils-Corsair | Press Release | Medium |
| 9 | How d-Matrix's In-Memory Compute Tackles AI Inference Economics (Vik's Newsletter) | https://www.viksnewsletter.com/p/d-matrix-in-memory-compute | Technical Blog | Medium |
| 10 | d-Matrix JetStream I/O Accelerators announcement | https://www.d-matrix.ai/announcements/jetstream/ | Vendor | Medium |
| 11 | d-Matrix Corsair at Rack Scale | https://www.d-matrix.ai/corsair-at-rack-scale/ | Vendor | Medium |
| 12 | d-Matrix Hot Chips 2025 page | https://www.d-matrix.ai/hot-chips-2025/ | Vendor | Medium |
| 13 | d-Matrix raises $275M (SiliconANGLE Nov 2025) | https://siliconangle.com/2025/11/12/chip-startup-d-matrix-raises-275m-speed-inference-memory-compute/ | Press | Low |
| 14 | d-Matrix acquires GigaIO SuperNODE (ConvergeDigest) | https://convergedigest.com/d-matrix-acquires-gigaio-supernode-and-pcie-fabric-for-rack-scale-ai-inference/ | Press | Low |

### Key Findings

- **DIMC Core**: Integrates MAC (multiplier-accumulator) directly into SRAM bit-cells; 64×64 MAC array per core
- **Chiplet**: 4 chiplets/chip, each chiplet = 4 quads × 4 slices = 256 DIMC cores per chiplet
- **2 chips per Corsair card**: 16 chiplets total; 4,096 DIMC cores; 4 GB SRAM at 150 TB/s
- **Off-chip memory**: 256 GB LPDDR5X per card at ~400 GB/s
- **Compute**: 2,400 TOPS INT8 per card; supports MXINT16/MXINT8/MXINT4 (OCP MX formats)
- **Process node**: TSMC 6nm
- **Host interface**: PCIe Gen5 x16
- **Interconnect**: DMX Link (custom in-house die-to-die, 1 TB/s); DMX Bridge card for 2-card merge
- **Scale-out**: JetStream 400G Ethernet I/O card (400 Gbps, full-height PCIe Gen5)
- **Software**: Aviator stack — Model Factory, Compressor, Compiler (MLIR-based), Inference Engine, Runtime; PyTorch + Triton DSL integration
- **Deployment**: PCIe card into standard servers; rack-scale via GigaIO PCIe fabric acquisition

---

## Update — 2026-08-08

*search_date: 2026-08-08*
*Baseline above is the 2026-04-05 scan and is retained unchanged.*

### Correction to the 2026-04-05 Key Findings

The line **"2 chips per Corsair card: 16 chiplets total; 4,096 DIMC cores; 4 GB SRAM at 150 TB/s"** conflates two levels. The correct counting is:

- **Per Corsair card**: 2 chips × 4 chiplets = **8 chiplets, 2,048 DIMC cores, 2 GB SRAM @ 150 TB/s**
- **Per dual card** (2 cards merged by the DMX Bridge): **16 chiplets, 4,096 DIMC cores, 4 GB SRAM @ 300 TB/s**

`chips/d-matrix/summary.md` and `research/d-matrix/investigations/hw-architecture.yaml` already used the correct per-card figures; only this file's Key Findings summary was wrong.

### Newly Found Resources

| # | Title | URL | Type | Quality |
|---|-------|-----|------|---------|
| 15 | Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference (ISCA 2026) | https://ramyadhadidi.github.io/files/dMatrix-Raptor-ISCA.pdf | Peer-reviewed paper (measured silicon) | **High — primary** |
| 16 | ISCA 2026 conference site (2026-06-27 – 07-01, Raleigh) | https://iscaconf.org/isca2026/ | Conference | High |
| 17 | d-Matrix × Alchip: world's first 3D-DRAM solution (Raptor announced, 2025-11-18) | https://www.d-matrix.ai/announcements/d-matrix-and-alchip-announce-collaboration-on-worlds-first-3d-dram-solution-to-supercharge-ai-inference/ | Vendor announcement | High |
| 18 | Tom's Hardware — 3DIMC vs HBM coverage | https://www.tomshardware.com/pc-components/ram/new-3d-stacked-memory-tech-seeks-to-dethrone-hbm-in-ai-inference-d-matrix-claims-3dimc-will-be-10x-faster-and-10x-more-efficient | Press | Medium |
| 19 | Corsair AI Inference Platform enters full production (2026-06-09) | https://www.d-matrix.ai/announcements/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/ | Vendor announcement | High |
| 20 | HPCwire — full-production coverage ("sampling to scale manufacturing") | https://www.hpcwire.com/off-the-wire/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/ | Press | Medium (page 403; snippet via search) |
| 21 | CNBC — "Nvidia challenger D-Matrix starts chip production, Microsoft backing" (2026-06-09) | https://www.cnbc.com/2026/06/09/nvidia-d-matrix-chip-production-microsoft.html | Press | Medium (page 403; headline/date via search) |
| 22 | techedgeai — TSMC N6 + Alchip, organic substrate vs CoWoS, SquadRack summer deployment | https://techedgeai.com/d-matrix-launches-corsair-inference-accelerator-to-slash-ai-latency/ | Press | Medium |
| 23 | GigaIO Sells Datacenter Technology and Assets to d-Matrix (BusinessWire, 2026-04-02) | https://secure.businesswire.com/news/home/20260402375332/en/GigaIO-Sells-Groundbreaking-Datacenter-Technology-and-Assets-to-d-Matrix | Press release (seller's own) | High |
| 24 | supercomputing.news — GigaIO deal detail (SuperNODE ≤32 accelerators, FabreX sub-200 ns, Carlsbad team) | https://www.supercomputing.news/ai/d-matrix-acquires-gigaios-data-center-business | Press | Medium |
| 25 | DataCenterDynamics — d-Matrix acquires SuperNODE and FabreX from GigaIO | https://www.datacenterdynamics.com/en/news/chip-startup-d-matrix-acquires-supernode-and-fabrex-from-gigaio/ | Press | Medium (page 403; headline via search) |
| 26 | gigaio.com — footer now reads "FabreX is a trademark of d-Matrix, Inc." | https://gigaio.com/ | Corroborating primary | Medium |
| 27 | d-Matrix acquires Wallaroo.ai (2026-08-03) | https://www.d-matrix.ai/announcements/d-matrix-acquires-wallaroo/ | Vendor announcement | High |
| 28 | PRNewswire — Wallaroo.ai acquisition wire release | https://www.prnewswire.com/news-releases/d-matrix-acquires-wallarooai-to-speed-up-deployment-of-heterogeneous-ai-inference-workloads-302840688.html | Press release | High |
| 29 | unite.ai — d-Matrix buys Wallaroo to orchestrate inference across chips | https://www.unite.ai/d-matrix-buys-wallaroo-to-orchestrate-inference-across-chips/ | Press | Medium |
| 30 | Bowen Inc. — Wallaroo.ai acquired by d-Matrix (exclusive financial advisor) | https://boweninc.com/transaction/wallaroo-ai-acquired-by-d-matrix/ | Advisor tombstone | Medium |
| 31 | Parasail × d-Matrix accelerators (Parasail's own post) | https://www.parasail.io/blogs/parasail-d-matrix-accelerators | Partner blog | Medium |
| 32 | engineering.com — Hopper/Blackwell prefill + Corsair decode split | https://www.engineering.com/parasail-deploys-d-matrix-accelerators-for-ai-inference/ | Press | Medium |
| 33 | Infinity — Advancing full model AI inference on Corsair (Ignition results) | https://infinity.inc/research/dmatrix-corsair | Partner technical blog | Medium |
| 34 | Infinity × d-Matrix partnership | https://infinity.inc/research/dmatrix-partnership | Partner blog | Medium |
| 35 | d-Matrix blog — Advancing full model AI inference on Corsair with Infinity (2026-07-22) | https://www.d-matrix.ai/advancing-full-model-ai-inference-on-corsair-with-infinity/ | Vendor blog | Medium |
| 36 | d-Matrix blog — Scaling AI the right way: rack-level inference solution | https://www.d-matrix.ai/blog/scaling-ai-the-right-way-introducing-our-rack-level-inference-solution/ | Vendor blog | Medium |
| 37 | d-Matrix news index (full 2026 announcement list) | https://www.d-matrix.ai/news/ | Vendor index | Medium |
| 38 | Gimlet Labs — low-latency speculative decode on Corsair (illustrative, not measured) | https://gimletlabs.ai/blog/low-latency-spec-decode-corsair | Partner blog | Low |
| 39 | MLCommons Inference (datacenter) — no d-Matrix submission found | https://mlcommons.org/benchmarks/inference-datacenter/ | Benchmark index | High (negative result) |
| 40 | Hot Chips 2026 company lineup preview — d-Matrix not listed | https://www.vlsi.kr/en/hot-chips-2026-company-lineup-preview-en/ | Conference preview | Low |
| 41 | Hot Chips official site (HC38: 2026-08-23 – 08-25, Stanford) | https://www.hotchips.org | Conference | High |

### Key Findings (2026-08-08)

- **Corsair is in FULL PRODUCTION** as of 2026-06-09; volume shipment beginning summer 2026 to unnamed priority customers; availability still gated to "select, qualified customers."
- **Manufacturing partner named**: Alchip Technologies, on **TSMC N6**; organic substrate; no HBM/CoWoS; multi-year capacity secured (vendor statement).
- **Raptor** — the next silicon generation, successor to Corsair and the commercial debut of **3DIMC** — was announced with Alchip on 2025-11-18 and **characterized on real early silicon** in an ISCA 2026 paper. TSMC N4P logic die face-to-face bonded (36 µm µbump) onto a 3D-DRAM die; 4 gangs × 4 slices per chiplet, slice = 4×4 tensor-engine array + SIMD core; 840 DRAM banks → 256 channels per chiplet; 4 chiplets @ 1.2 GHz per MCM, up to 4 MCMs per card; 9-4-9 organic substrate on 3D CoWoS; ~422 W/MCM, Tj ≤ 105 °C; 8 × on-package LPDDR5X-9600 = 128 GB/MCM; PCIe Gen7. **Measured: ~105 TB/s 3D-DRAM per card @ 700 MHz, 2.5 ns avg flit latency.** Status: pre-production; no launch date, sampling date or pricing disclosed. **Pavehawk** is the preceding lab-validated test silicon.
- **Rack-scale specs published** (vendor): dual card 4,800 TFLOPS MXINT8 / 19,200 MXINT4, 4 GB @ 300 TB/s, ≤512 GB capacity, 6,400 mm² silicon, 512 GB/s card-to-card DMX Bridge; server (8 cards) 19.2 / 76.8 PFLOPS, 16 GB @ 1,200 TB/s, ≤2 TB; rack (64 cards) 128 GB @ 9.6 PB/s, ≤16.4 TB, 100B params Performance Mode / 1T+ Capacity Mode.
- **DMX Link ≠ DMX Bridge**: DMX Link is die-to-die at ~1 TB/s (115 ns); the 512 GB/s figure is card-to-card. A circulating "correction" merging the two was checked and rejected.
- **Two acquisitions**: GigaIO datacenter business (2026-04-02 — SuperNODE, FabreX PCIe Gen5 memory fabric with sub-200 ns cross-server access, Carlsbad team; terms undisclosed) and **Wallaroo.ai** (2026-08-03 — serving runtime + control plane, vLLM/SGLang, x86/Arm/GPU, cloud/on-prem/edge/air-gapped; terms undisclosed). Wallaroo is the stack's first orchestration layer.
- **SquadRack** — rack-scale blueprint announced 2025-10-14 at the OCP Global Summit with Arista, Broadcom and Supermicro; the named vehicle for the summer 2026 first deployments. Previously absent from the repo.
- **Parasail (2026-07)** is an announced partnership and **intent**, not a deployment: prefill on NVIDIA Hopper/Blackwell, decode on Corsair. "10× faster" / "3× energy" are unqualified vendor claims.
- **Infinity "Ignition"** (external, $15M): auto-generated a full inference stack running Qwen3 / Qwen3.5 / Gemma4 end-to-end in 10 days at up to 92% of speed-of-light; 92% of theoretical peak in 10 hours. The circulating "32 compute units" phrasing is unverified and deliberately not recorded.
- **Negative results**: no MLPerf Inference submission in any round; d-Matrix not listed in the Hot Chips 38 advance program (conference runs 2026-08-23 – 08-25, after this scan — recheck afterwards); no 2026 funding round (the $275M on record is the Nov 2025 Series C at ~$2B valuation, ~$450M total; M12 participation is why "Microsoft backing" headlines appeared in June 2026).
- **Corsair power now on record** (Hot Chips 2025, supersedes "not disclosed"): 275 W @ 800 MHz / 550 W @ 1.2 GHz; ~38 TOPS/W claimed.
