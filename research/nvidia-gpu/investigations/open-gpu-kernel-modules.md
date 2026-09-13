# NVIDIA Open GPU Kernel Modules — Investigation Report

*resource: https://github.com/NVIDIA/open-gpu-kernel-modules*
*as_of: 2026-04-05*
*commit: db0c4e6 (tag 595.58.03)*
*chip: nvidia-gpu*
*layer: Driver, Firmware*

---

## Overview

The NVIDIA Open GPU Kernel Modules (open-gpu-kernel-modules) is the open-source release of NVIDIA's Linux kernel driver for its GPU hardware, first published in May 2022. It provides the kernel-space side of the full NVIDIA GPU software stack, sitting between user-space components (CUDA runtime, OpenGL ICD, Vulkan driver) and the physical GPU hardware. Version 595.58.03 is examined here.

The release covers all Turing and later GPU families (TU10x through the Blackwell GB100 series), which includes every current datacenter GPU: A100, H100, H200, B100, and the consumer RTX 20xx through 50xx lines.

A key architectural property of this driver is its **GSP split design**: on Turing and newer GPUs, the bulk of the GPU Resource Manager (RM) firmware runs on the GPU's on-chip GSP (GPU System Processor) microcontroller rather than the CPU. The open-source kernel module therefore acts as a thin kernel-mode proxy — it maps BARs, manages interrupts, handles OS integration, and communicates with GSP-side RM over a shared-memory message queue. This is distinct from older, fully CPU-side RM drivers.

The modules are kernel objects (`.ko` files) loadable into the Linux kernel. They are structured in four deliverables plus shared source:

| Module | Device node | Purpose |
|---|---|---|
| `nvidia.ko` | `/dev/nvidia[N]`, `/dev/nvidiactl` | Core GPU access, RM proxy, memory, interrupts |
| `nvidia-modeset.ko` | `/dev/nvidia-modeset` | Display KMS/modeset via NVKMS |
| `nvidia-drm.ko` | `/dev/dri/card[N]` | Linux DRM/KMS integration layer |
| `nvidia-uvm.ko` | `/dev/nvidia-uvm` | Unified Virtual Memory (ATS, page fault handling) |
| `nvidia-peermem.ko` | (kernel module only) | GPUDirect RDMA peer-memory registration |

---

## Architecture

### Module Structure and Source Layout

```
open-gpu-kernel-modules/
├── kernel-open/          # Linux kernel interface layer (the "OS layer")
│   ├── nvidia/           # Core module: nv.c (entry), nv-pci.c, os-interface.c, nv-dma.c
│   ├── nvidia-drm/       # DRM/KMS layer wrapping NVKMS
│   ├── nvidia-modeset/   # NVKMS modeset interface
│   ├── nvidia-uvm/       # Unified Virtual Memory subsystem
│   └── nvidia-peermem/   # GPUDirect RDMA peer memory
├── src/
│   ├── nvidia/           # RM core (Resource Manager, GSP client, NVOC objects)
│   │   ├── arch/nvalloc/unix/src/  # Unix RM entry points: escape.c, osapi.c
│   │   ├── src/kernel/gpu/         # Per-engine kernel RM objects (gsp/, fifo/, gr/, etc.)
│   │   └── generated/              # NVOC-generated class boilerplate (382 files)
│   ├── nvidia-modeset/   # NVKMS (Kernel Mode Setting) core
│   └── common/           # Shared SDK headers, NVLink, display port stacks
```

The `kernel-open/` tree is the Linux OS layer: it uses Linux kernel APIs directly (PCI, DMA, interrupts, `copy_from_user`, `ioremap`). The `src/` tree is the OS-agnostic RM core that compiles as C with a portability shim.

### Key Abstractions

#### 1. NVOC Object Model (OBJGPU, KernelGsp, KernelFifo, ...)

The RM core uses **NVOC** (NVIDIA Object Class), an in-house single-inheritance OOP framework in C. Objects have generated vtable-style dispatch. Every major subsystem is an NVOC object that lives as a member of `OBJGPU`:

```c
struct OBJGPU {
    struct KernelBif   *pKernelBif;    // Bus Interface
    struct KernelMc    *pKernelMc;     // Master Control
    struct KernelBus   *pKernelBus;    // BAR mapping, instance memory
    struct KernelGsp   *pKernelGsp;    // GSP firmware client
    struct KernelFifo  *pKernelFifo;   // Channel/FIFO scheduler
    struct KernelGsplite *pKernelGsplite[4]; // Per-partition GSP lite
    // ... 20+ engine objects total
};
```

Methods on these objects are function pointers that are "halified" — i.e., selected at driver initialization time based on the detected GPU chip (e.g., GA100, GH100, GB100), enabling one binary to support multiple GPU architectures. Example from `g_kernel_gsp_nvoc.h`:

```c
NV_STATUS (*__kgspBootstrap__)(OBJGPU *, KernelGsp *, KernelGspBootMode);  // halified (4 hals)
void (*__kgspConfigureFalcon__)(OBJGPU *, KernelGsp *);                    // halified (4 hals)
const BINDATA_ARCHIVE *(*__kgspGetBinArchiveGspRmBoot__)(KernelGsp *);     // halified (11 hals)
```

#### 2. GSP RPC Messaging System

The most important inter-component boundary is the **GSP message queue**. The CPU-side kernel module (client RM) sends commands to the GPU-side GSP firmware (server RM) through a pair of shared-memory FIFO queues:

- **Command queue**: CPU writes RPC request structs; GSP reads them.
- **Status queue**: GSP writes response structs; CPU reads them.

The CPU side uses `GspMsgQueueSendCommand()` (in `message_queue_cpu.c`) which:
1. Acquires write buffer slots via `msgqTxGetWriteBuffer()`.
2. Copies the RPC message payload into the slot.
3. Issues a store fence (`portAtomicMemoryFenceStore()`) to enforce WAW ordering.
4. Submits the buffer via `msgqTxSubmitBuffers()`.
5. Rings the GSP doorbell by writing the queue head register via `kgspSetCmdQueueHead_HAL()`.

The payload is typed via the `RPC_PARAMS` macro that casts into a versioned struct (`rpc_gsp_rm_control_v03_00`, `rpc_free_v03_00`, etc.).

#### 3. RM Escape ioctl / Resource Model

User-space speaks to the driver through a set of escape ioctls on `/dev/nvidiactl`. The dispatch chain is:

```
user-space open() / ioctl()
  → nvidia_unlocked_ioctl()          [kernel-open/nvidia/nv.c]
    → nvidia_ioctl()
      → rm_ioctl()                   [arch/nvalloc/unix/src/osapi.c]
        → RmIoctl()                  [arch/nvalloc/unix/src/escape.c]
          → switch(cmd)
            NV_ESC_RM_ALLOC         → Nv04AllocWithSecInfo()
            NV_ESC_RM_CONTROL       → Nv04ControlWithSecInfo()
              → _rmapiRmControl()   [src/kernel/rmapi/control.c]
                → rpcRmApiControl() → GSP RPC queue
            NV_ESC_RM_FREE          → Nv01FreeWithSecInfo()
            NV_ESC_RM_MAP_MEMORY    → Nv04MapMemoryWithSecInfo()
```

Large ioctls are passed via the `NV_ESC_IOCTL_XFER_CMD` indirection: user-space puts the real command and payload pointer into an `nv_ioctl_xfer_t` struct, allowing payloads larger than the ioctl size limit.

The RM uses a resource-server model (`g_resServ`, a `RsServer`): every GPU object (device, subdevice, channel, memory allocation, etc.) is represented by a handle in the client's handle table, and RM calls operate on `(hClient, hObject)` pairs.

#### 4. KernelChannel / GPFIFO / Work Submission

GPU compute work is submitted through channels. A GPFIFO channel is an NVOC `KernelChannel` object corresponding to one of the GPFIFO class generations:

| GPU architecture | GPFIFO class |
|---|---|
| Turing | `TURING_CHANNEL_GPFIFO_A` (clc46f.h) |
| Ampere | `AMPERE_CHANNEL_GPFIFO_A` (clc56f.h) |
| Hopper | `HOPPER_CHANNEL_GPFIFO_A` (clc86f.h) |
| Blackwell | `BLACKWELL_CHANNEL_GPFIFO_A` (clc96f.h) |

Allocation is handled by `kchannelConstruct_IMPL()` in `kernel_channel.c`, which:
- Calls `kchannelAllocHwID_HAL()` to obtain a runlist hardware channel ID.
- Calls `_kchannelAllocOrDescribeInstMem()` to allocate instance memory (the channel's GPU page table and context area).
- Calls `kchannelAllocChannel_HAL()` to register the channel with the FIFO engine.

User-space submits GPFIFO entries by mmapping the channel's BAR1 user-mode doorbell page. Pushing work does not require a kernel call — the CUDA runtime appends pushbuffer commands to a ring buffer, updates the GPFIFO get/put pointer, and writes the doorbell to notify the FIFO engine.

#### 5. Unified Virtual Memory (UVM) Module

`nvidia-uvm.ko` implements unified virtual memory for CUDA Managed Memory. It provides:
- An `unlocked_ioctl` entry on `/dev/nvidia-uvm` for UVM API commands.
- Per-architecture HAL files (e.g., `uvm_ampere.c`, `uvm_hopper.c`, `uvm_blackwell.c`) implementing fault buffer handling and page table management for each GPU generation.
- ATS (Address Translation Services) support on POWER9 and Grace platforms.
- GPUDirect Storage integration hooks.

UVM intercepts GPU page faults (TLB miss faults reported by the GPU fault buffer), migrates pages between CPU and GPU memory, and updates GPU page tables — all without user-space involvement.

### Dependency Graph

```
User-space (CUDA RT, libGL, Vulkan)
        |
        | ioctl on /dev/nvidiactl  /dev/nvidia-uvm
        |
+-------+-----------------------------+
| nvidia.ko (OS layer + RM proxy)     |
|   nv.c  →  osapi.c  →  escape.c    |
|   └── KernelGsp RPC client         |
|         ↓  shared-memory queue     |
|       GSP firmware (on GPU)        |← loads via Falcon/Booter
|         ↓  MMIO BAR0 registers     |
|       GPU hardware                 |
+------------------------------------+
| nvidia-uvm.ko                      |
|   GPU page fault handler           |
|   ↔  nvidia.ko (rm_gpu_ops)        |
+------------------------------------+
| nvidia-drm.ko + nvidia-modeset.ko  |
|   DRM KMS ↔ NVKMS                  |
+------------------------------------+
| nvidia-peermem.ko                  |
|   IB peer_memory_client interface  |
+------------------------------------+
```

---

## Data Flow: CUDA Kernel Launch Reaching the GPU

This traces the path from a CUDA runtime `cudaLaunchKernel()` call to GPU hardware execution.

1. **CUDA Runtime (user-space)**: The runtime has previously allocated a channel and mapped the doorbell BAR1 page. For a kernel launch it:
   - Constructs a pushbuffer containing `LAUNCH_DMA` and `COMPUTE_LAUNCHDMA` methods or GR `Kepler A` class commands.
   - Appends one or more GPFIFO entries (pointing to the pushbuffer segment) to the channel's GPFIFO ring.
   - Writes the channel's work-submit token to the mapped BAR1 user-mode doorbell register — **this write does not enter the kernel**.

2. **FIFO Engine (GPU hardware)**: The doorbell write wakes the FIFO scheduler. It fetches the new GPFIFO entries, reads the pushbuffer from GPU memory or BAR1, and dispatches the methods to the Graphics/Compute engine (GR).

3. **GR Engine (GPU hardware)**: Decodes the compute launch command. Looks up the CWD (Context Work Descriptor) for the channel's context. Schedules warps on the SMs.

4. **Interrupt / completion (driver path)**: When the workload finishes (or a fault occurs), the GPU raises an interrupt. The interrupt is fielded by `nvidia_isr()` / `nvidia_isr_bh()` in `nv.c`, which signals the appropriate event semaphore via `rm_isr_scheduled_bottom_half()`.

5. **GSP-mediated control paths**: Channel and context allocation (step 0 of the above) do go through the full RM path:
   - `NV_ESC_RM_ALLOC` with class `TURING_CHANNEL_GPFIFO_A`.
   - In `_rmapiRmControl()`, the call is packaged into a `rpc_gsp_rm_alloc_v03_00` RPC payload and sent to GSP via `GspMsgQueueSendCommand()`.
   - GSP performs the actual hardware programming (runlist update, instance memory write) and replies on the status queue.
   - The CPU-side driver polls the status queue and returns the result to user-space.

---

## Hardware Interface

### MMIO / BAR Mapping

`nv-pci.c` maps GPU BARs at PCI probe time:

- **BAR0** (`NV_GPU_BAR_INDEX_REGS`, 16 MB typical) — control register space. Mapped with `devm_ioremap()`. Accessed via `GPU_REG_RD32(pGpu, offset)` / `GPU_REG_WR32(pGpu, offset, val)` which expand to `osDevReadReg032()` / `osDevWriteReg032()` performing a direct MMIO read/write.
- **BAR1** (`NV_GPU_BAR_INDEX_BAR1`, up to 64 GB on H100) — framebuffer aperture. Used for direct GPU memory access and user-mode doorbell pages.
- **BAR2** (optional) — virtual BAR for instance memory access.

Register offsets are defined as C preprocessor macros in architecture-specific headers under `src/common/inc/swref/published/` (e.g., `dev_gsp.h` for Hopper/Blackwell, `dev_boot.h` for boot strapping registers).

### GSP Firmware Loading

Firmware loading at driver initialization follows this sequence:

1. `kgspGetBinArchiveGspRmBoot_HAL()` — retrieves the embedded binary archive for the detected GPU (11 different arch-specific implementations, covering TU102 through GB100). The binaries are compiled-in as C arrays in `src/nvidia/generated/g_bindata_kgsp*.c`.
2. `kgspPrepareForBootstrap_HAL()` / `_kgspRpcLoadAndExecuteGenericBootloader()` — uploads the Booter microcode to the Falcon processor and executes it. Booter establishes the WPR2 (Write-Protected Region) in framebuffer for the GSP firmware image.
3. `kgspBootstrap_HAL()` — loads `gspRmBoot` (the GSP-RM firmware image) into WPR2 and sets the GSP reset vector. On Hopper and later, this uses the FMC (Fabric Management Controller) flow.
4. The GSP processor boots, initializes the full RM stack internally, then signals readiness via the message queue.
5. The CPU driver sends `RPC_GSP_INIT_DONE` and subsequent RPC calls for device enumeration.

### GSP Communication Registers

`kgspSetCmdQueueHead_HAL()` — architecture-specific implementations write the GSP command queue head register over BAR0 to ring the GSP doorbell after submitting a message. The register offsets are defined in `dev_gsp.h` files per architecture (Ampere: `ga102`, Hopper: `gh100`, Blackwell: `gb100`, `gb10b`, `gb20b`).

### GPU Register Access Macros

```c
// From src/nvidia/generated/g_gpu_access_nvoc.h
#define GPU_REG_RD32(g, a)    REG_INST_RD32(g, GPU, 0, a)
#define GPU_REG_WR32(g, a, v) REG_INST_WR32(g, GPU, 0, a, v)
// Unchecked (bypass lost-GPU check):
#define GPU_REG_RD32_UNCHECKED(g, a)  osDevReadReg032(g, gpuGetDeviceMapping(g, ...), a)
```

---

## Key Findings

1. **GSP split is the dominant architectural fact.** Nearly all GPU configuration, context management, and resource allocation happens inside the GSP firmware on the GPU, not in the CPU kernel module. The open-source kernel module is primarily an RPC client and OS integration shim. This is why the open-sourced code is tractable — the complex microarchitecture-specific logic lives in the GSP binary.

2. **NVOC provides HAL polymorphism across 5+ GPU generations.** The `halified` function pointer pattern (visible throughout `g_*_nvoc.h` generated headers) allows a single binary to support Turing through Blackwell by selecting per-chip vtable entries at runtime. This mirrors how hardware engineers use the same RTL base with per-chip overrides.

3. **User-mode doorbells eliminate kernel transitions for steady-state work submission.** Once a channel is set up, the CUDA runtime writes directly to a BAR1-mapped page. Kernel intervention is only needed for resource allocation and fault handling — a deliberate throughput optimization.

4. **UVM (nvidia-uvm.ko) is the AI workload page-fault engine.** Training workloads using `torch.cuda` managed memory rely on UVM to migrate tensors on demand. UVM has its own per-architecture HAL (Ampere, Ada, Hopper, Blackwell) implementing fault buffer parsing and CE (Copy Engine) operations for page migration.

5. **SPDM / Confidential Computing security layer.** The `libspdm_*` files in `kernel-open/nvidia/` and the `conf_compute/` and `spdm/` subsystems implement device attestation and encrypted communication channels for Hopper/Blackwell Confidential Computing (H100/H200 SXM in CC mode). This is the in-kernel counterpart to NVIDIA's `nvTrust` security architecture.

6. **Nouveau interoperability.** The `nouveau/` directory contains a Python script to extract GSP firmware images from the driver source, allowing the open-source Nouveau DRM driver to load NVIDIA's GSP firmware for Turing/Ampere/Hopper, achieving feature parity with the proprietary driver on modern GPUs.

7. **Lock stress and lock testing infrastructure.** `src/kernel/rmapi/lock_stress.c` and `lock_test.c` expose ioctl-accessible lock stress test objects, providing in-kernel regression testing for the RM's multi-GPU locking model — unusual for a production driver but visible as part of the open-source release.

---

## Relation to Hardware Architecture

The driver's internal structure directly mirrors NVIDIA's GPU architectural hierarchy:

- **OBJGPU** is the software model of the GPU chip. Its child engines (`KernelGsp`, `KernelFifo`, `KernelBus`, `KernelMc`, etc.) correspond 1-to-1 with hardware engines on the chip: the GSP processor, the FIFO (channel scheduler), the Bus Interface Unit, the Master Control interrupt aggregator.

- **GPFIFO channel classes** are versioned per GPU generation (Turing clc46f → Ampere clc56f → Hopper clc86f → Blackwell clc96f), reflecting each generation's changes to the channel state area format and pushbuffer method space.

- **The HAL layer encodes the microarchitecture divergence.** When Hopper introduced the Confidential Computing Security Unit (SEC2) and the new FMC-based GSP boot flow, these appeared as new HAL implementations in `arch/hopper/` subdirectories. Blackwell multi-die (GB100) topology adds the GSPlite objects for per-partition GSP instances (`pKernelGsplite[4]`).

- **The WPR2 (Write-Protected Region 2)** that Booter establishes in VRAM for the GSP image corresponds to the physical on-GPU protected memory region that the chip's memory controller enforces via hardware access controls — the software Booter/WPR flow is the initialization sequence for that hardware feature.

- **NVLink** (`src/nvidia/src/kernel/gpu/nvlink/`) and the NVSwitch driver (`kernel-open/nvidia/linux_nvswitch.c`) are present, exposing the NVLink fabric management interface. This is the software side of the NVLink high-speed GPU-to-GPU interconnect that underpins NVL72/HGX multi-GPU configurations for large-model training.
