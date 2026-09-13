# Codex Review: Tenstorrent Search Results

*reviewed: 2026-04-05*
*model: gpt-5.4:high*

## Overall Assessment

All 15 layers are non-empty — no raw-coverage failure. Issues are primarily canonical-source quality and categorization. The weakest layers by quality are **Communication**, **Driver / Firmware**, and **Scale-up Interconnect**.

## Findings

### HIGH Priority

**Communication layer is weak**
- Only one entry is an actual software communication resource; the rest are hardware/market analysis or an issue thread
- Missing canonical sources:
  - `ttnn` CCL API index: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html
  - `ttnn.all_reduce`: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.all_reduce.html
  - `ttnn.all_gather`: https://docs.tenstorrent.com/ttnn/latest/ttnn/api/ttnn.all_gather.html
  - `ttnn.reduce_scatter`: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.reduce_scatter.html
  - `tt-topology`: https://github.com/tenstorrent/tt-topology

**Driver / Firmware layer missing critical resources**
- Missing: `tt-umd` (https://github.com/tenstorrent/tt-umd) — User-Mode Driver
- Missing: `tt-firmware` (https://github.com/tenstorrent/tt-firmware) — device firmware
- Missing: `tt-zephyr-platforms` (https://github.com/tenstorrent/tt-zephyr-platforms) — Zephyr RTOS firmware
- Missing: TT Zephyr Platforms docs: https://docs.tenstorrent.com/tt-zephyr-platforms/

**Scale-up Interconnect is miscategorized and incomplete**
- The on-chip NoC entry belongs in Data Path, not scale-up
- Missing inter-card/system resources:
  - Warp 100 Bridge: https://docs.tenstorrent.com/aibs/warp100
  - TT-QuietBox topology: https://docs.tenstorrent.com/systems/quietbox/specifications.html
  - TT-LoudBox / T3000: https://docs.tenstorrent.com/systems/t3000/specifications.html

### MEDIUM Priority

**Incomplete hardware generation coverage**
- Grayskull papers referenced in compute/SRAM but official Grayskull specs missing: https://docs.tenstorrent.com/aibs/grayskull/specifications.html

**Framework Integration missing active repo**
- `pytorch2.0_ttnn` (https://github.com/tenstorrent/pytorch2.0_ttnn) — labeled deprecated but still referenced by Tenstorrent GitHub; should be noted as legacy

**Miscategorized / misranked items**
- `Developers Portal` should be in Other Resources, not Framework Integration
- `TT-Installer` and `TT-Tools-Common` are tooling utilities, not core driver/firmware resources
- `TT-Ascalon` is separate CPU IP, not part of the Tensix compute engine section
- Arteris press release is future-facing PR, not a primary data-path reference

**Ranking quality**
- Community blogs and newsletters are carrying layer coverage in Communication, Scale-up, and Data Path that should come from official docs first

## Additional Resources Found

- `tt-umd` (User-Mode Driver): https://github.com/tenstorrent/tt-umd
- `tt-firmware`: https://github.com/tenstorrent/tt-firmware
- `tt-system-firmware` (newer firmware releases): https://github.com/tenstorrent/tt-system-firmware
- `tt-zephyr-platforms`: https://github.com/tenstorrent/tt-zephyr-platforms
- TT Zephyr Platforms docs: https://docs.tenstorrent.com/tt-zephyr-platforms/develop/getting_started/index.html
- Warp 100 Bridge: https://docs.tenstorrent.com/aibs/warp100
- TT-Topology: https://github.com/tenstorrent/tt-topology
- TT-QuietBox specifications: https://docs.tenstorrent.com/systems/quietbox/specifications.html
- T3000 specifications: https://docs.tenstorrent.com/systems/t3000/specifications.html
- TT-NN CCL API: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html
- ttnn.all_gather: https://docs.tenstorrent.com/ttnn/latest/ttnn/api/ttnn.all_gather.html
- ttnn.reduce_scatter: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.reduce_scatter.html
- tt-inference-server: https://github.com/tenstorrent/tt-inference-server
- ttnn-visualizer: https://github.com/tenstorrent/ttnn-visualizer
- tt-exalens (hardware debugger): https://github.com/tenstorrent/tt-exalens
- pytorch2.0_ttnn (deprecated): https://github.com/tenstorrent/pytorch2.0_ttnn
- TT-NN Tutorials: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tutorials.html

## Suggested Additional Queries

1. `site:github.com/tenstorrent/tt-umd OR site:docs.tenstorrent.com "user-mode driver" Tenstorrent`
2. `site:github.com/tenstorrent/tt-firmware OR site:docs.tenstorrent.com/tt-zephyr-platforms Tenstorrent firmware`
3. `site:docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api "all_reduce" OR "all_gather" OR "reduce_scatter"`
4. `site:github.com/tenstorrent/tt-metal "Ethernet and Multichip Basics"`
5. `site:docs.tenstorrent.com/aibs grayskull specifications Tenstorrent`
6. `site:docs.tenstorrent.com/aibs/wormhole "Warp 100 Bridge" OR connectivity`
7. `site:docs.tenstorrent.com/systems quietbox topology OR t3000 topology Tenstorrent`
8. `site:github.com/tenstorrent pytorch2.0_ttnn`
9. `site:github.com/tenstorrent tt-exalens OR ttnn-visualizer OR tt-tutorial`
