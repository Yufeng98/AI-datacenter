# Q.ANT Photonic NPU — Layer Table

*as_of: 2026-08-08*

| Layer | Component | Notes |
|-------|-----------|-------|
| Framework Integration | PyTorch, TensorFlow, Keras (operator dispatch bridges). **Plus (Apr 2026): a third-party PyTorch graph-capture path — Daisytuner "Daisyflow"** | Plug-and-play for target workload classes; LLM not supported. Daisytuner reports Faster R-CNN + ResNet-50 captured from PyTorch and run end-to-end on NPU Gen 2 with "no custom code" |
| High-Level API | C/C++/Python API | Imperative; not graph-compile model; provided with NPS server |
| Algorithm Library | Q.PAL (Q.ANT Photonic Algorithm Library) | Nonlinear AI primitives; physics simulation; image classification; sensor fusion |
| Compiler (internal) | Q.ANT proprietary; maps Q.PAL ops to MZI voltage tables | Not user-facing; no user ISA |
| Compiler (third-party, added 2026-08-08) | **Daisytuner `docc`** — Daisytuner Optimizing Compiler Collection, SDFG-based; compiles PyTorch models to Q.ANT NPU Gen 2 | Revealed April 2026. Owned by **Daisytuner GmbH, not Q.ANT**. `github.com/daisytuner/docc` is public and actively developed, but **no Q.ANT/photonic backend was located in the open tree** — the photonic target may be delivered via Daisytuner's hosted service. Q.ANT calls it "the first time an AI model from a standard ML framework has been successfully compiled for photonic hardware" |
| Runtime / HAL | Proprietary (PCIe DMA engine, DAC controller, ADC reader) | Manages weight loading and activation readout |
| Driver | Linux kernel driver (PCIe, proprietary) | Pre-installed on NPS server; no open-source analog |
| ISA / Kernel Model | None | No user-programmable ISA; hardware configured via voltage tables |
| Hardware — Compute | MZI mesh (LENA TFLN architecture); optical MVM + native nonlinear ops (NPU 2) | z-cut LNoI, TFLN 600 nm on insulator; Pockels effect phase control. Published throughput: 8 GOPS on nonlinear functions (vendor page, public by Jan 2026). MZI count / matrix dimensions not disclosed |
| Hardware — Memory | Host DRAM via PCIe (no HBM/GDDR on chip); DAC/ADC bridge | Weights as phase voltages; no on-chip SRAM |
| Hardware — Interconnect | PCIe (host, generation not disclosed); standard HPC networking via host NIC; no photonic interconnect | Scale-out via Ethernet/IB |
| Hardware — Manufacturing | Stuttgart, Germany — TFLN pilot line **operated jointly with IMS CHIPS**; Q.ANT is a TRUMPF spin-off | Not a CMOS foundry; TFLN photonic process |
| Deployment status | LRZ Munich + JSC Jülich operational; NPS "available for evaluation in select data center environments"; **IONOS signed 2026-05-19 as first commercial customer**, rollout planned later in 2026 | No general commercial availability as of 2026-08-08 |

*Changes vs the 2026-04-05 table: split the Compiler row into Q.ANT-internal and third-party (Daisytuner) rows; added the Daisytuner PyTorch capture path to Framework Integration; corrected manufacturing to name IMS CHIPS; added published throughput and deployment-status rows. See `chips/q-ant/summary.md` § "Q.ANT Update ... (2026-08-08)".*
