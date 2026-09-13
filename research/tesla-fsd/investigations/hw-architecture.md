# Tesla FSD Chip — HW Architecture Investigation

## Summary

The Tesla FSD chip is a custom edge inference SoC built around a proprietary systolic-array NPU. Two generations (HW3 / HW4) are deployed in vehicles. The architecture prioritizes deterministic, static-scheduled inference for automotive-grade latency and power constraints.

---

## Compute Units

### NPU (Neural Processing Unit)
- Custom systolic MAC array: 96×96 = 9,216 MACs per NPU
- 18,432 INT8 ops per clock cycle per NPU
- HW3: 2 NPUs × 36.86 TOPS = 73.7 TOPS total
- HW4: 3 NPUs × ~40.5 TOPS = ~121.6 TOPS total
- Arithmetic: 8-bit × 8-bit multiply, 32-bit integer accumulate
- Per cycle feed: 256 B activation + 128 B weight from SRAM

### CPU Cluster
- HW3: 12× ARM Cortex-A72 @ 2.6 GHz (3 quad-core clusters)
- HW4: 20× ARM cores @ 2.35 GHz
- Handles OS, sensor preprocessing, decision arbitration

### GPU
- HW3: ARM Mali G71 MP12 @ 1 GHz
- HW4: updated GPU (exact spec not public)
- Used for image preprocessing, visualization

---

## Memory Hierarchy

| Level | HW3 | HW4 |
|---|---|---|
| L1/L2 (CPU) | ARM standard | ARM standard |
| NPU SRAM | 32 MiB per NPU (2×) = 64 MiB | 32 MiB per NPU (3×) = 96 MiB |
| DRAM | 8 GB LPDDR4, 68 GB/s | 16 GB GDDR6, 224 GB/s |
| Storage | — | 256 GB NVMe |

---

## Execution Model

- **Static scheduling:** At compile time the NN compiler partitions the neural network into tile-based workloads. Each tile is assigned to a specific NPU block with pre-determined weight memory addresses. No dynamic dispatch unit needed.
- **In-order execution** with out-of-order memory: NPU executes instructions sequentially; DMA engine handles memory access out-of-order to hide latency.
- **Dataflow:** weights and activations streamed from SRAM to MAC array each cycle. Results accumulate in 32-bit registers then written back.

---

## ISA

8-instruction minimal ISA:
1. DMA Read (load activation/weights from SRAM)
2. DMA Write (store results to SRAM)
3. Dot-product (convolution variant A)
4. Dot-product (convolution variant B)
5. Dot-product (convolution variant C)
6. Scale (normalization)
7. Element-wise addition (residual)
8. Parameter slot (instruction modifier)

---

## Chip Integration (SoC)

- HW3: Samsung 14 nm, ~260 mm² estimated die area
- HW4: Samsung 7 nm, smaller die with more NPUs
- Dual FSD Computer per vehicle for redundancy (safety requirement)
- ISP (image signal processor) for camera preprocessing
- Proprietary high-speed camera bus

---

## Power Budget

- HW3 NPU: 7.5 W per NPU, ~21% of total chip TDP
- HW3 board total: ~100 W
- HW4 board total: ~160 W
- Target: automotive-grade thermal envelope (-40°C to +85°C junction)

---

## Key Design Decisions

1. **Fixed-function NPU** over programmable GPU: maximizes TOPS/W for a fixed model topology
2. **All-SRAM weight storage** per NPU: eliminates DRAM bandwidth bottleneck for weight reads
3. **INT8 throughout**: saves 4× memory vs FP32, fits models in 32 MiB SRAM
4. **Static schedule**: avoids control overhead, enables deterministic latency guarantees
5. **Dual-chip redundancy**: safety critical — one chip can cross-validate the other

---

## 2026-08-08 Update — AI4.5, AI4.1/AI4 Plus, AI5

*Investigation date: 2026-08-08. Prior sections (research date 2026-04-05) describe HW3/HW4 and remain valid.
This section covers everything that entered the public record between 2026-01-26 and 2026-08-08.*

### Evidence quality for this update

Tesla publishes no FSD datasheets, gives no conference talks on FSD silicon (confirmed: **no Tesla entry in the
Hot Chips 38 advance program**, 2026-08-23 to 2026-08-25), and files no MLPerf results. Every fact below derives
from one of four weak channels, and each is labelled accordingly:

| Channel | Examples in this update | Typical confidence |
|---|---|---|
| Musk statement on X or an earnings call | AI5 tape-out, AI4.1 specs, HW3 concession | medium (vendor, unaudited, sometimes self-contradictory) |
| One photograph of one packaged part | AI5 die size, 12 DRAM packages, substrate type | medium for gross observations, low for anything derived |
| A since-deleted LinkedIn post | Samsung 2 nm / Taylor leg of AI5 | low-medium (multi-outlet relay, no official confirmation) |
| Owner sightings + firmware analysis | AI4.5 / "AP45" existence and part number | medium for existence, low for the three-SoC claim |

### AI5 — tape-out announced 2026-04-15

**Date discipline.** 2026-04-15 is the **announcement** date. Musk's X post — *"Congrats to the @Tesla_AI chip
design team on taping out AI5! AI6, Dojo3 & other exciting chips in work"* — was accompanied by a photo of an
already-packaged sample, so the tape-out itself necessarily predates it. **Tesla has never published a tape-out
date.** Corroborated by Electrek, MarketWatch/Morningstar (both 2026-04-15), Benzinga and DigiTimes (2026-04-17).

A "KR 2613" package date code (read as assembly week 13 of 2026) is reported by **Tom's Hardware only** and is
not corroborated anywhere else — recorded as low confidence, not used to date the tape-out.

**Package observations (medium confidence, photo-derived, multi-outlet):**
- Compute ASIC die roughly **half a standard reticle**. Consistent with Musk's own "half a reticle with good
  margin" statement.
- **Twelve SK hynix memory packages** surrounding the die. DigiTimes independently names SK hynix as the memory
  supplier.
- Memory is marked as standard **discrete DRAM on an organic substrate** — **no interposer, no CoWoS, no HBM**.
  GDDR6 vs GDDR7 is **not established by any source**.

**Single-source inference (low confidence — do NOT promote to spec):** Tom's Hardware derives a **384-bit
external bus** and **~768 GB/s – 1.536 TB/s** from the package count plus an assumed DRAM grade. Nothing else
corroborates the width, the grade, or the bandwidth.

**Compute: entirely undisclosed.** No peak TOPS/FLOPS, no dtype list, no TDP, no die area, no NPU count, no
array geometry, no clock. The HW3/HW4 96×96 systolic description **must not be assumed to carry forward.**

**Performance claims are vendor marketing and mutually inconsistent:**

| Musk claim | Origin | Note |
|---|---|---|
| "up to 40×" / "40 times" better than AI4 | **Q3 2025 earnings call, 2025-10-22/23** (autoevolution 2025-10-23; TweakTown 2025-10-24), repeated in 2026 coverage | Largest of the figures; frequently misdated to April 2026 |
| "up to 10×" more powerful than AI4 | Musk, via Electrek | Conflicts with 40× |
| "maybe by a factor of 10" perf-per-dollar | Musk, via Electrek | Different metric |
| "5× the memory bandwidth of AI4" | Musk, via Electrek | Only bandwidth claim; no absolute figure |

Musk additionally stated AI5 integrates **the Arm CPU cores and PCIe blocks** alongside the accelerators —
notable because it is the first FSD-line part described as carrying a host interface rather than only
automotive IO.

**Dual-foundry sourcing.** Disclosed **2026-07-13** when a Samsung Foundry principal engineer (James Kim) posted
on LinkedIn that *"the Tesla-Samsung AI5 chip has reached tape-out"* and would be *"manufactured at the Taylor
fab using our latest 2nm process."* **The post was deleted; Samsung declined to comment; no official Samsung or
Tesla confirmation exists.** Relayed by Korea Times, UPI, Asiae, TechTimes, XenoSpectrum (2026-07-13/14).

Consequences for the record:
- AI5 is a **dual-source, dual-node part**: TSMC (node **not disclosed by anyone**) plus Samsung **SF2-class
  2 nm** at Taylor, Texas. The exact Samsung derivative (SF2 / SF2P / SF2T) is **not disclosed**.
- **No tape-out date is public for either implementation**, and **no multi-month TSMC→Samsung gap is
  supported** — Korea Times reports both foundries received the design in April 2026.
- Dual-sourcing predates 2026: Musk had engaged both foundries for AI5 by October 2025.
- Headlines saying Samsung production "starts soon" or that AI5 "is about to enter mass production" overstate a
  deleted LinkedIn post.

**Correct status verb:** taped out, with a **first packaged engineering sample shown**. Not sampling to third
parties, not shipping, not deployed. Volume manufacturing targeted **2027** (Electrek: mid-2027 for automotive,
requiring several hundred thousand completed AI5 boards line-side; MarketWatch: volume manufacturing 2027).

**Target market is unsettled — record both readings.** On the Q1 2026 call (2026-04-22) Musk said AI5 goes into
Optimus and the data center *"because it's looking like we'll be able to achieve unsupervised self-driving with
AI4 that is far greater than human safety levels,"* while allowing that AI5 may eventually go into cars with no
near-term cost or urgency reason to switch. July 2026 Samsung coverage describes AI5 as destined for Tesla
vehicles, robots and data centers; April coverage lists vehicles, Optimus and xAI data centers. **Musk did NOT
say AI5 "will not go into cars."** Accurate framing: near-term priority is Optimus and datacenter/inference
clusters, automotive **deferred rather than excluded**.

### AI4.1 / "AI4 Plus" — announced 2026-04-22, no silicon

Q1 2026 earnings call. Musk verbatim: *"We are planning an AI4 upgrade to use newer generation RAM. So it'll go
from 16 gigabytes to I think 32 gigabytes per SoC. So 64 gigabytes total, and probably a 10% increase in compute
and in memory bandwidth."* He named it "AI4.1 or AI4 Plus."

- Memory: **16 → 32 GB per SoC; 64 GB per dual-SoC board**; "newer generation RAM" — **type not disclosed**.
- Compute and memory bandwidth: **~+10 %** each, unqualified and unbenchmarked.
- Process: Samsung **7 nm with process modifications**; production is gated on Samsung completing them. Also
  reported for Samsung's Texas operations.
- Production target: **"next year" = 2027.** (One aggregator live blog renders this as "mid-2025" — a
  transcription error; 2027 is correct.)
- No NPU count, array geometry, ISA or dtype change was described. Best read as a **memory-and-clock refresh**,
  not a new microarchitecture. **Status: announced only; no silicon, no tape-out claim.**

### AI4.5 / "AP45" — reported 2026-01-26, never announced

Owners of Fremont-built 2026 Model Y AWD Premium vehicles delivered Dec 2025 – Jan 2026 found FSD computers
labelled **"AP45" / "AP4.5," part number 2261336-02-A**, matching Tesla's Electronic Parts Catalog. Tesla has
issued no statement.

The commonly repeated **three-SoC (vs AI4's two-SoC) board** claim is a **firmware-analysis inference by
@greentheonly**, reported by Electrek and explicitly hedged as *"may feature"* — **low confidence; not a teardown
die count, not a Tesla statement.** No compute, memory, process or TDP figures exist for AI4.5.

### HW3 formally declared insufficient — 2026-04-22

On the Q1 2026 call Musk conceded that millions of HW3 vehicles will not receive unsupervised FSD and that Tesla
will build dedicated "micro factories" to retrofit them (installed base ~4 M vehicles). **Independently confirmed
by The Verge and Electrek, both 2026-04-22.**

Attribution caveat: the quoted lines HW3 *"simply does not have the capability"* and HW3 having *"only
one-eighth the memory bandwidth of HW4"* trace to **Electrek's write-up**, not to a verifiable transcript.
Attribute to Electrek, not to Tesla. The underlying facts (HW3 cannot run unsupervised FSD; retrofit planned)
are independently confirmed.

Litigation followed: US class action 2026-06-29; Dutch collective action 2026-06-18.

### Memory-bandwidth discrepancy — unresolved, must stay flagged

| Generation | WikiChip (used in the pre-2026 repo tables) | Electrek (2026-04) |
|---|---|---|
| HW3 | 8 GB LPDDR4, **68 GB/s** (4266 Mbps × 128-bit) | **~48 GB/s** |
| HW4 / AI4 | 16 GB GDDR6, **224 GB/s** (14 Gbps × 128-bit) | **~384 GB/s**, 32 GB total across the dual-SoC board |

**Neither set is Tesla-published, and the two are mutually inconsistent.** They may differ in per-SoC vs
per-board accounting, or one may simply be wrong. Electrek's "HW3 has one-eighth the bandwidth of HW4" ratio is
internally consistent **only** with its own 48/384 pair. Electrek's comparison points: NVIDIA Orin ~205 GB/s,
Thor ~273 GB/s. **Resolution: present both with attribution; do not pick a winner and do not average them.**

### Ecosystem — unchanged

- **No Tesla talk in the Hot Chips 38 advance program** (2026-08-23 to 2026-08-25). Waymo, not Tesla, holds the
  autonomous-driving keynote and the automotive SoC slot. Verified directly against hotchips.org.
- No MLPerf submission; no public SDK; no ISA disclosure beyond the reverse-engineered HW3 8-instruction set.
- The "closed ecosystem" characterization holds without qualification.

### Named but empty

Musk's 2026-04-15 post also named **AI6** and **Dojo 3** as "in work." Nothing further is public for either —
no node, no schedule, no architecture. Recorded here only so the names are not later mistaken for new evidence.
