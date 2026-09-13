# Stream Computing NeuralScale — Hardware Architecture Investigation

*as_of: 2026-08-08*
*chip: stream-computing*
*device_class: Programmable NPU — RISC-V scalar core + custom vector/matrix ISA extension (NeuralScale; China, 希姆计算)*

---

## Overview

**Stream Computing Inc. / 希姆计算** (operating entity 广州希姆半导体科技有限公司, founded 2019) builds
**NeuralScale**, a general-purpose programmable NPU architecture that extends RISC-V with custom
vector/matrix instructions. It ships as **PCIe datacenter inference cards** — STCP920, STCP950L, STCP950P,
STCP980L, STCP980P — all built on the same first-generation **P920** SoC.

Company geography, since secondary sources garble it: the CARRV'21 paper bylines "Stream Computing Inc.,
**Beijing**, China" and ByteDance's benchmark names "Beijing Stream Computing Technology Co., LTD", while
the current corporate site gives the operating entity as **Guangzhou** (Huangpu District, Knowledge City;
粤ICP备2024180922号) with additional offices in Beijing, Shanghai and Hangzhou. Origin **Beijing**, current
entity **Guangzhou** — not a Shanghai company.

**Maturity: shipping / deployed.** First-generation silicon taped out and returned in 2021; first cards
shipped 2021; volume orders 2022; the vendor claims large-scale AI-compute-centre deployment from 2024.
ByteDance's ByteMLPerf repo hosts the vendor's own Apache-2.0 backend, whose README states STCP920
"is in mass production, and has completed a batch of shipments to users". **No second-generation silicon is
public.**

The architecturally distinctive point — and the reason this part belongs in the survey — is that a
NeuralScale core is **a RISC-V scalar core driving three decoupled execution engines** with an entirely
**software-managed, non-coherent** memory hierarchy. It is neither a SIMT GPU nor a fixed-function
systolic NPU.

---

## 1. Compute Engine

### 1.1 The NeuralScale core (NPC — Neural-network Processing Core)

| Element | Detail | Confidence |
|---|---|---|
| Scalar core | **AndesCore N25F** licensed IP — 32-bit RISC-V, **5-stage in-order**, dynamic branch prediction, separate I/D caches, **RV32G** | confirmed |
| Scalar caches | 64 KB L1 I-cache + 64 KB L1 D-cache per core (hardware-cached, **scalar/control path only**) | confirmed |
| NPC pipeline | 4 stages: decode → issue → execute → write-back. The issue unit holds **three instruction buffers** (one per engine), issues **in order**, up to **3 instructions per cycle** (one per engine), and blocks only on *address overlap* with in-flight instructions | confirmed |
| **VME** — Vector MAC Engine | MAC vector of **64 FP16 MACs** plus a **POLY** module with `exp` / `div` / `sqrt` function units for activations and classifiers. Executes both base RVV instructions (vector-register operands) and custom vector instructions (**L1 Buffer / Intermediate Buffer byte addresses held in GPRs**) | confirmed |
| **MME** — Matrix MAC Engine | MAC matrix of **64 × 32 = 2048 MAC units**; each MAC performs FP16 or **2× INT8** per cycle. Reads Data Input Buffer + Weight Buffer, writes the Intermediate Buffer | confirmed |
| **MTE** — Memory Transfer Engine | Explicit data-movement engine linking the local L1 Buffer to remote L1 Buffers, the LLB, and external DDR through the NoC. Modes: L1↔LLB point-to-point, L1↔DRAM point-to-point, and **LLB broadcast to all corresponding L1 Buffers** | confirmed |
| RVV configuration | **VLEN = 1024 bits, ELEN = 16 bits** | confirmed |
| Vector register file | Dedicated "REG Bank" inside the NPC alongside the scalar GPRs; **architectural register count not disclosed** | confirmed (existence) |

**Datatypes: FP16 and INT8 only.** No BF16, no FP8/FP4, no INT4, no native FP32 compute path. Every model in
the vendor's public support table is FP16, or w8a8 / w8a16 quantised for LLMs. `confirmed`

### 1.2 P920 SoC

| Item | Value | Confidence |
|---|---|---|
| NeuralScale cores | **32** | confirmed |
| Clusters (SDK resource model) | **4 NPC Clusters** per chip; `stc-smi -q` reports `Cluster count: 4`. 8 NPCs/cluster is *derived* (32÷4) — no source states cores-per-cluster | confirmed (4 clusters) / derived (8 per cluster) |
| Management CPU | **ARM Cortex-A53** — in-order, 8-stage, dual-issue; PPI + up to 64 SPI interrupts; boots and manages the SoC (PCIe/DDR/SPI controller init, monitoring) | confirmed |
| Peak throughput | **256 TOPS INT8 / 128 TFLOPS FP16** @ 1.0 GHz | confirmed |
| Throughput reconciliation | 2048 MME MACs × 2 flop × 32 cores × 1 GHz = **131.1 TFLOP/s** FP16, quoted as "128 TFLOPS"; ×2 for INT8 = 262 TOP/s, quoted as "256 TOPS". The 64 VME MACs/core (+4.1 TFLOPS) appear to be excluded from the headline figure | derived |
| Process | **TSMC 12 nm FinFET** | confirmed |
| Die area | **400 mm²** | confirmed |
| Chip TDP | **130 W** @ 1.0 GHz (chip; card TDP is 150–160 W) | confirmed |
| Peripherals | UART, SPI, I²C, PWM, RTC | confirmed |
| Frequency domains | Chip split into independently settable **"east" and "west"** SoC frequency/voltage domains (`stc-smi -s --freq --east/--west`; `stcmlDeviceSetFrequency` index 0x1=west, 0x10=east, 0x11=both) | confirmed |
| Power domains | **4 independent banks**; the host can gate 1–3 banks (power-save) or all 4 (standby, static power only) over PCIe | confirmed |
| DVFS | Supported SoC frequencies **624 / 800 / 900 / 1000 / 1100 / 1200 / 1300 / 1400 MHz** (card manual lists 624/800/900/1000/1200; the `stcml` API lists all eight; the `stc-smi` CLI exposes only 1000/900). vsoc 800–950 mV; vmac 550–950 mV per `stc-smi` vs 550–800 mV per `stcmlVoltageInfo_t` | confirmed (values differ between vendor docs) |

**Transistor count, package type and dimensions, and memory-vendor part numbers: not disclosed.**

---

## 2. On-chip Memory — software-managed, non-coherent

This is the defining property of the part. The NoC is explicitly labelled a **"Non-coherent interconnect"**
in the P920 block diagram, and **every buffer in the compute path is addressed explicitly by the program**:
custom VME/MME instructions take L1 / Intermediate-Buffer byte addresses in general-purpose registers, and
all inter-level movement is issued as explicit MTE / sysDMA instructions. Only the 64 KB scalar I$/D$ are
hardware caches, and they serve the control path, not tensor data. `confirmed`

| Level | Capacity | Bandwidth | Managed by | Confidence |
|---|---|---|---|---|
| Vector register file | VLEN 1024 b × (register count not disclosed) | — | compiler / ISA | confirmed (VLEN) |
| **L1 Buffer** (per NPC) | **1.25 MiB** = 1 MiB **Data IO Buffer** + 0.25 MiB **Weight Buffer** | **512 GB/s** — *single third-party-hosted vendor figure; no vendor manual states it* | software (MTE + explicit addressing) | confirmed (capacity) / single-source (BW) |
| **IM / Intermediate Buffer** (per NPC) | **256 KB** — MME output and VME staging, private per core | not disclosed | software | confirmed |
| Scalar L1 I$ / D$ (per NPC) | 64 KB + 64 KB | not disclosed | **hardware cache** (control path only) | confirmed |
| **LLB** (Last Level Buffer, per cluster) | **8 MiB** per cluster; **32 MiB total**, physically **8 × 4 MB banks**, each independently attached to the NoC | **Contested:** CARRV'21 claims **17 TB/s aggregate**; the ByteDance-hosted vendor spec table says **8 MB @ 256 GB/s** per cluster (≈1 TB/s chip-wide). Unreconciled in any source | software (MTE, sysDMA) | confirmed (capacity) / **contested (BW)** |
| **DDR** (global, per cluster) | **4 GiB** per cluster × 4 = **16 GB** | see §3 | software (sysDMA) | confirmed |

**Total on-chip SRAM (*derived*, not vendor-stated):** 32 × (1.25 MiB L1 + 0.25 MiB IM) + 32 MiB LLB =
**80 MiB**, plus 4 MiB of scalar caches.

---

## 3. Off-chip Memory — the defining limitation

| Source | Type | Capacity | Interface | Bandwidth |
|---|---|---|---|---|
| All five STCP card manuals | **LPDDR4X** | 16 GB | 256-bit, 1867 MHz clock (= 3733 MT/s) | **108 GB/s** (stated) |
| ByteDance-hosted vendor spec table (2023) | LPDDR4X | 16 GB | — | **119.4 GB/s** |
| *Derived from the stated interface* | — | — | 256 b × 3733 MT/s ÷ 8 | **119.5 GB/s** |
| CARRV'21 (design description) | LPDDR4, 4 independent channels, ≤4 GB each | 16 GB | 4266 MT/s | **136 GB/s theoretical** |

`stc-smi -g --freq` on a shipping card reports `ddr frequency 3733`, matching the manuals. The 108 GB/s
figure is presumably an effective/derated number; **the discrepancy is not explained by any source**.
Confidence: confirmed that it is 16 GB LPDDR4X at ~108–119 GB/s; the authoritative figure is **contested**.

**DMA:** each of the four DDR subsystems integrates **two DMA controllers** (`stc-smi` reports
`DMA count: 2` per cluster) with independent channels, each connecting a DDR controller to an LLB through
the NoC. The profiler exposes them as `sysdma_0` / `sysdma_1`, with channel C0 = DDR→LLB and C1 = LLB→DDR.
`confirmed`

**The bandwidth consequence, stated plainly.** ~108 GB/s against 128 TFLOPS FP16 implies an arithmetic
intensity requirement above 1000 FLOP/byte. The vendor's own LLM support table shows what that costs:
Qwen2-7B-Instruct needs **2 cards** for 10.39 tok/s single-stream; Qwen2-72B-Instruct needs **16 cards** for
4.98 tok/s; DeepSeek-R1-Distill-Llama-70B needs **16 cards** for 4.42 tok/s. This is a CNN/NLP-era
inference card retrofitted for LLMs, not an HBM-class part. `confirmed`

---

## 4. Data Path and Execution Model

- **Heterogeneous host+device, explicitly CUDA-shaped.** The host CPU launches *kernel functions* onto N
  NPCs with `kernel<<<NCORE>>>(args)`; each NPC runs the same kernel and self-identifies through the
  built-ins `CoreID` / `CoreNum`. `confirmed`
- **Per-core MIMD, not SIMT.** Each NPC is an independent RISC-V scalar thread issuing to its own
  VME/MME/MTE. There is no warp or wavefront abstraction; divergence is ordinary scalar branching.
- **In-order, statically scheduled, three-way engine overlap.** The scalar core issues at most three
  instructions per cycle, one per engine, and the hardware stalls only on address overlap. Performance
  therefore depends on the compiler or kernel author software-pipelining MTE loads against MME/VME compute.
  The profiler is built entirely around this, reporting `PAL(MTE/MME)`, `PAL(MTE/VME)`, `PAL(VME/MME)` and
  `PAL(ALL)` overlap percentages per NPC.
- **Measured behaviour on BERT** (CARRV'21 §4.3 trace analysis): NPCs occupy 95% of total cycles; DMAs
  overlap with scalar/NPC work 96% of the time; inside an NPC, **MME occupies 78% of cycles**, MTE overlaps
  MME and/or VME 92% of the time, and VME serialises against MTE/MME 45% of its time due to data
  dependencies on MME. `confirmed`
- **Memory-space qualifiers mirror the physical hierarchy:** `__local__` → L1 Buffer (per-NPC),
  `__shared__` → LLB (per-cluster), plain device pointers → DDR, `IM_BUFFER_START` → Intermediate Buffer.
  `sync()` is the intra-group barrier.
- **Shape state lives in CSRs, not instruction operands.** Matrix/vector shapes are programmed into custom
  unprivileged CSRs before issuing an instruction (`CONFIG_VE_CSR`, `CONFIG_VE_BC_CSR`,
  `DEFINE_SHAPE(rows, cols)` in SHC).

---

## 5. ISA — the distinctive contribution

Two separate things are routinely conflated in secondary coverage. The survey must keep them apart.

### 5.1 The shipping product ISA (proprietary, closed)

- Base **RV32G** scalar (AndesCore N25F) + **RVV v0.8** standard vector. `confirmed`
- **53 custom instructions** on top of RVV, encoded in the **custom-3 major opcode `1111011`**, designated
  **OP-VE**. Fixed-width 32-bit format:
  `[31:26] funct6 | [25] dmc | [24:20] rs2 | [19:15] rs1 | [14] dm | [13:12] opm2 | [11:7] rd | [6:0] 1111011`.
  Source (`rs1`, `rs2`) and destination (`rd`) registers stay in base-ISA positions to keep decode simple.
- `opm2[1:0]` selects the operand shape class — `00` matrix-matrix (mm), `01` matrix (m), `10` matrix-vector
  (mv), `11` matrix-scalar (mf). `{dmc,dm}` selects direction — `{0,X}` full-element, `{1,0}` vertical on
  row vectors, `{0,1}` horizontal on column vectors.
- Published `funct6` examples: `veadd`, `vesub`, `veacc`, `veemul`, `veemacc`, **`memul`** (matrix multiply),
  **`meconv`** (convolution), `velkrelu` (Leaky ReLU), `mov` (MTE transfer).
- **22 unprivileged vector CSRs** carry shape/config state — e.g. `0x400 shape_s1` (VME, width/height of
  matrix A), `0x401 shape_s2` (matrix B), `0x408 conv_FM_in`, `0x409 conv_Depth_in`, `0x422 mte_shape`.
  Updated only by base RISC-V CSR instructions.
- **This is a vendor-custom extension, NOT the ratified RISC-V matrix/IME extension.** The custom-3 opcode
  is exactly the space RISC-V reserves for non-standard extensions.
- Live confirmation in shipping silicon: the SDK ships an **instruction decoder** (`inst_decoder.py` in
  MLTC's `tools/`) that turns a faulting 32-bit encoding back into assembly, e.g.
  `0x06ABE8FB → veadd.mv.dimw (x17), (x23), (x10)`.
- Profiler-visible instruction classes: `VME-CU` (custom vector), `VME-VEC` (native RVV), `MME`, `MTE`
  (`mov`, `mov_l12llb`, `mov_llb2l1`, `pld`, `icmov`), `SYNC`, `MCU` (scalar).

### 5.2 The open RISC-V Matrix extension proposal (separate, CC-BY-4.0)

Stream Computing separately drives an **open matrix-extension proposal** in RISC-V International's Matrix
Task Group. **The P920 does not implement it**, but it is the company's public standards work and the only
genuinely open ISA artifact.

- Repo `riscv-stc/riscv-matrix-spec`, AsciiDoc, **CC-BY-4.0**, 27★, last activity **Dec 2024**; **v0.5**
  announced Nov 2024.
- Model: **8 tile registers `tr0–tr7`** (input tiles) + **8 accumulation registers `acc0–acc7`**.
  Implementation-defined parameters **ELEN, MLEN, RLEN, AMUL** (AMUL ∈ {1,2,4,8} for widening accumulators);
  tile shapes `mtilem × mtilen × mtilek`; `mtype` CSR set by `msettype` / `msettypei`. Strongly modelled on
  the RISC-V "V" extension.
- Sub-extensions: **Zmi4** (INT4 matrix), **Zmv**, **Zmi2c**, **Zmc2i**, **Zmsp** (sparse).
- Named contributors: Chaoqun Wang, Fujie Fan, Hui Yao, Jie Feng, Kening Zhang, Kun Hu, Xin Ouyang,
  Xin Yang, Zhiqiang Liu, Zhiyong Zhang.
- Company standing (vendor claim, consistent with RISC-V International's published membership tiers):
  **Premier member**, board member, TSC member, **chair of the Software Applications & Tools committee**,
  **chair of the AI/ML SIG**, core member of the **Matrix TG**.

---

## 6. On-chip Interconnect and Synchronisation

### NoC

- **4×4 mesh**; every component attaches to a NoC router; all component↔router and router↔router links are
  **bidirectional**. `confirmed`
- **Control and data planes are separated**: control bus 32 bits each direction, data bus 512 bits each
  direction.
- At 1.0 GHz: **64 GB/s per direction per link, 128 GB/s combined**. `confirmed`
- Labelled **"Non-coherent interconnect"** in the P920 block diagram. `confirmed`
- *Discrepancy note:* a company-authored RISC-V International blog post is sometimes summarised as
  "4×6 mesh, 1 TB/s". The peer-reviewed CARRV'21 paper — same authors, same product — says 4×4 mesh and
  64 GB/s per direction. **Use CARRV.**
- NoC IP vendor, router microarchitecture, virtual-channel/QoS configuration and the exact ordering model
  beyond "non-coherent 4×4 mesh": **not disclosed**.

### HSYNC

- The **HSYNC subsystem** partitions the 32 NeuralScale cores into **up to 16 groups**, group size
  configurable by the application; an application can run on one group of 32 cores or on multiple groups.
  `confirmed`
- Exposed to the programmer only indirectly: SHC provides a `sync()` intrinsic, and the profiler counts
  `syn_cycle` / `syn_inst` / `syn_wait_cycle`. **The 16-group HSYNC API is not documented in the public
  SDK**, which presents the 4-cluster model instead. The mapping from HSYNC groups to SDK clusters is
  **not disclosed**.

---

## 7. Scale-up Interconnect

- The P920 has **two PCIe subsystems**, PCIE0 and PCIE1, each up to **16 lanes**, each configurable as
  endpoint or root complex. PCIE0 is normally the endpoint to the host; **PCIE1 is described in CARRV'21 as
  "usually configured as a root complex for scalability, connecting to other SoC chips to construct a
  larger-scale compute platform."** `confirmed (as a design description)`
- **However, no shipping card manual documents any chip-to-chip or card-to-card link.** All five STCP
  manuals list a single PCIe 4.0 ×16 host interface and nothing else. There is **no proprietary scale-up
  fabric** (no NVLink / CCIX / Ethernet-mesh equivalent) in any public product document. Whether the PCIE1
  path is wired or enabled on any shipping card is **not disclosed**.
- `stc-topo` (the vendor's `nvidia-smi topo` clone) reports only PCIe path classes —
  `LOC / PIX / PXB / PHB / SYS` — plus CPU and NUMA affinity. Multi-card locality is purely PCIe-tree and
  NUMA locality.

### Measured card-to-card bandwidth (`p2p_perf`, vendor tool, 2-card host)

| Transfer (NPU0 → NPU1) | Measured |
|---|---|
| DDR → DDR | **888.6 MB/s** |
| DDR → LLB | **9005 MB/s (~9.0 GB/s)** |
| LLB → DDR | **886.5 MB/s** |
| LLB → LLB | **9014 MB/s (~9.0 GB/s)** |

Real peer-to-peer traffic peaks around **9 GB/s** when the destination is on-chip LLB and collapses to
**<1 GB/s** when the destination is remote LPDDR — far below the PCIe 4.0 ×16 ceiling of ~32 GB/s
unidirectional. This is the concrete cost of 16-card tensor parallelism on this platform. `confirmed`

---

## 8. Scale-out Interconnect

Standard datacenter Ethernet at the server level. **No RDMA fabric, no collective-communication hardware,
and no NCCL/CNCL-equivalent library** in the SDK. Multi-card parallelism is expressed as compiler
operations (`stc.device_broadcast`, `stc.device_reduce_*`, `stc.device_allconcat`, `stc.device_sync`, …)
executed over PCIe peer-to-peer — see the software-stack investigation. `confirmed`

---

## 9. Host Interface, Card Physicals and Product Line

All five SKUs are **PCIe 4.0 ×16** (Lane Reversal supported), **single-width, ¾-length, full-height
(268.44 mm × 111.15 mm)**, **721.2 g**, **passively cooled**, with one **PCIe 8-pin** aux power connector,
PCI Vendor ID `0x23e2` / Sub-Vendor ID `0x23e2`. All report **identical memory topology**: 4 clusters,
1.25 MiB L1/core, 8 MiB LLB/cluster, 4 GiB DDR/cluster, 16 GB LPDDR4X 256-bit @ 108 GB/s. `confirmed`

| SKU | Base clk | Max clk | FP16 | INT8 | Card TDP | PCIe DID | SSID |
|---|---|---|---|---|---|---|---|
| **STCP920** | 1.0 GHz | (not listed) | 128 TFLOPS | 256 TOPS | **160 W** | 0x0100 | 0x0000 |
| **STCP950L** | 1.0 GHz | 1.2 GHz | 153 TFLOPS | 309 TOPS | **160 W** | 0x0103 | 0x0003 |
| **STCP950P** | 1.0 GHz | 1.2 GHz | 145 TFLOPS | 290 TOPS | **150 W** | 0x0104 | 0x0004 |
| **STCP980L** | 1.0 GHz | 1.4 GHz | 179 TFLOPS | 358 TOPS | **160 W** | 0x0105 | 0x0005 |
| **STCP980P** | 1.0 GHz | 1.4 GHz | 167 TFLOPS | 335 TOPS | **150 W** | 0x0106 | 0x0006 |

*Observation (derived, not vendor-stated):* the **L** SKUs' quoted throughput equals 128 TFLOPS × max clock
exactly (1.2 → 153.6; 1.4 → 179.2) at 160 W, while the **P** SKUs quote lower numbers at 150 W. The
economical reading is that L and P are the same die at different power caps, with P quoting a
power-limited sustained figure. All five are the same first-generation `npu-v1` design — identical
topology, identical memory, identical dimensions and weight, shared MCU/NPU-ctrl firmware images, and a
single compiler architecture target. **The vendor never states die identity**, so treat "same die,
different bins" as strongly evidenced rather than disclosed.

### Other card-level facts (`confirmed`)

- **PCIe BARs:** BAR0 16 GB prefetchable, BAR2 32 MB non-prefetchable, BAR4 1 GB prefetchable.
  **1 physical function (64-bit), no SR-IOV VFs.** Requires BIOS "Above 4G Decoding".
- **SMBus out-of-band management** (7-bit addr 0x2A): board power (0xF1), board temp (0xF2), PCI IDs
  (0xF3/0xF4), MCU FW version (0xF5), **power brake** (write 0x55 to 0xF6 → chip power-off), chip temp
  (0xF7), serial number LE/BE (0xF8/0xF9), part number (0xFA), chip voltage (0x50), MAC voltage (0x51).
- **Thermals:** board inlet 0–50 °C; chip 0–90 °C, recommended <70 °C, hard max 90 °C; alert 91 °C and
  shutdown 95 °C by default; bidirectional airflow supported.
- **Firmware:** two images — **NPU-ctrl firmware** (SoC init; current V10.3.7 / V10.3.8) and **MCU firmware**
  (card power; current V10.0.13 / V10.0.14). History records digital signature added, PCIe Gen3/Gen4
  auto-detection, ATS removed, and **passthrough to KVM VMs** supported.
- **Measured idle draw** on a shipping STCP920: ~27–34 W against a 160 W rating (`stc-smi` transcript).
  Typical/sustained power and MTBF are **not disclosed**.

### System products (vendor claims)

2U4-card, 4U8-card and 4U16-card AI servers; host-CPU compatibility claimed for Intel, AMD, Hygon (海光),
Phytium (飞腾), Kunpeng (鲲鹏) and Loongson (龙芯); an "AI 一体机" LLM appliance line (from 2023) including a
government-agent appliance; and a 智算云平台 cloud/scheduling platform.

---

## 10. Measured Performance

**CARRV'21 — vendor-run, batch 64, P920 vs T4 / V100 / Goya:**

- ResNet-50 v1.5 INT8: **14,442 img/s**, latency **4.43 ms**, **110 IPS/W** at 130 W → 2.98× T4 and 1.85×
  V100 throughput; 1.59× T4, 4.23× V100, 1.50× Goya efficiency.
- BERT FP16 (batch 32; INT8 BERT accuracy deemed inadequate): **4192 sentences/s**, latency **7.63 ms**,
  **32 sps/W** → 2.31× T4, 1.31× V100, 2.37× Goya throughput.

**ByteMLPerf (ByteDance repo, Apache-2.0 STC backend, © 2023 Stream Computing Inc.) on STCP920 —
reproducible:**

| Model | QPS | Accuracy metric |
|---|---|---|
| resnet50-tf-fp32 | 8725.94 | Top-1 77.24% |
| bert-tf-fp32 | 822.38 | F1 86.45 |
| bert-torch-fp32 | 813.86 | F1 86.14 |
| albert-torch-fp32 | 824.49 | F1 87.66 |
| robert-torch-fp32 | 800.70 | F1 83.19 |
| widedeep-tf-fp32 | 2,395,899.9 | Top-1 77.39% |

**No MLPerf Inference submission** was found under any Stream Computing / 希姆 / NeuralScale name.

---

## 11. Deployment Status and Roadmap — real vs claimed

**Real (confirmed).** First-generation silicon is genuinely in mass production, not an eval board. The
ByteMLPerf STC README states STCP920 "is in mass production, and has completed a batch of shipments to
users" and publishes reproducible accuracy + QPS. The public model-support table lists **ByteDance-internal
model names** (`hotsoon_live_v6_turbo`, `hotsoon_live_v8`, `atmosphere_vulgar`, `model_goods_search`,
`content_classify`, `smoke_hotsoon_live_v2`, `resnet50_hotsoon`), independently corroborating a large
internet-platform deployment. **ByteDance and JD.com are named strategic investors.**

**Vendor claims (label as such, no independent documentation found).** "1000+ STCP cards deployed at a
leading internet company, 200+ models"; compatibility certifications with China Telecom eCloud, Tencent
Cloud, OpenCloudOS and Baidu PaddlePaddle; "large numbers of thousand-card clusters nationwide"; the
**"world's first ultra-high-altitude RISC-V thousand-card compute cluster with Tibet Mobile"** (dated
2026-03-31 on the vendor news list); a RISC-V cluster in Lianyungang.

**Roadmap — NOT products.**

- **Second-generation chip.** Vendor timeline: *2022* — "obtained volume orders for the first-generation
  chip, while the second-generation chip's tape-out was blocked by the impact of the US BIS export ban";
  *2024* — "redesigned a second-generation chip meeting the restriction limits". The 2023-era ByteDance
  README said "second-generation products are in schedule and will be coming soon in 2023" — that did not
  happen publicly.
- The only public trace of a second-generation target is that the shipping MLTC compiler in SDK V1.12.1
  **accepts `--arch=npu-v2`** alongside the default `npu-v1`. **There is no npu-v2 hardware manual, no core
  count, no process node, no memory spec, no TDP, and no tape-out or sampling confirmation.** Do **not**
  enter `npu-v2` into the registry as silicon.
- **"RISC-V AI super node" (RISC-V AI 超节点)** — explicitly marked 研发中 (under development) on the vendor
  site, described as a "CPU+NPU+DPU AI-native computing unit". No specs, no topology, no date. Not a
  product.
- The company also ships its own **Jiuzhou (九州) LLM** (national filing completed 2025-03-31), which appears
  in the model-support table as `Jiuzhou-7B` (2 cards, 10.3 tok/s).

**Not verifiable, deliberately omitted:** a widely-repeated secondary narrative about founder departures and
customer churn could not be traced to any openable primary source and is not written into this survey.

**IP / standing (vendor claims):** 200+ patent applications, 30+ PCT, 96% invention patents. Investors named
on the site: ByteDance, JD.com, CCB International, BOC International, China Internet Investment Fund, CDB
Equipment, Xiamen C&D, CITIC Securities, Sequoia China, Walden International, MSA Capital, Tianyi Capital,
Jingque Capital, Puluo Capital, Guangzhou Industrial Investment, Guangzhou Knowledge City Group, Guangzhou
Development District Industrial Fund. **Amounts, valuations and round dates: not disclosed.**

---

## Sources

- [CARRV'21 — NeuralScale: A RISC-V Based Neural Processor Boosting AI Inference in Clouds](https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf)
- [STCP920 product manual, rev 1.12.1](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP920产品手册)
- [STCP950L product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950L产品手册)
- [STCP950P product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP950P产品手册)
- [STCP980L product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980L产品手册)
- [STCP980P product manual](https://docs.streamcomputing.com/AI加速卡/硬件产品手册/STCP980P产品手册)
- [HPE usage guide (stc-smi / stc-topo / stc-prof / stc-gdb, SHC samples)](https://docs.streamcomputing.com/AI加速卡/推理软件使用手册/STCRP使用指南/HPE使用指南)
- [C++ API — STCML reference](https://docs.streamcomputing.com/AI加速卡/开发者资源/C++_API)
- [Model support table (LLM cards / tok-s)](https://docs.streamcomputing.com/AI加速卡/开发者资源/模型支持说明)
- [p2p_perf guide — measured card-to-card bandwidth](https://docs.streamcomputing.com/AI加速卡/硬件性能测试手册/p2p_perf使用指南)
- [Firmware Release Notes](https://docs.streamcomputing.com/AI加速卡/固件使用手册/Firmware_Release_Notes)
- [ByteMLPerf / xpu-perf STC backend README](https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/backends/STC/README.md)
- [riscv-stc/riscv-matrix-spec (open matrix extension proposal, CC-BY-4.0)](https://github.com/riscv-stc/riscv-matrix-spec)
- [RISC-V International blog — NeuralScale (company-authored)](https://riscv.org/blog/2024/03/neuralscale-industry-leading-general-purpose-programmable-npu-architecture/)
- [Corporate site — company profile, timeline, investors, offices](https://www.streamcomputing.com/)
