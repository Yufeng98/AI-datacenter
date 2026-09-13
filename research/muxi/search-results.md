# MetaX (沐曦) Search Results

*as_of: 2026-08-08*
*chip: muxi*
*device_class: GPU (China, 沐曦)*

This file was created on 2026-08-08 during the landscape-update scan. The baseline muxi
investigations (2026-04-05) predate it, so only the 2026-08-08 wave is catalogued here.

---

## Scan — 2026-08-08 (window 2026-04-05 → 2026-08-08)

Topics pursued:

1. MetaX / 沐曦 WAIC 2026 announcements (曦景 S600, 曦索 X300)
2. MXC600 shipment status and 2026 volume
3. MXC700 / 曦云 C700 status
4. MetaX full product catalog (C / N / X / G series, servers, supernodes)
5. 曦思 N300 specifications
6. MXMACA release cadence and open-source community size
7. vLLM-metax release history
8. github.com/metax-maca new repositories
9. MetaX FY2025 / Q1 2026 financials
10. MetaX ecosystem / model Day-0 adaptations

> **Sourcing note.** The initial raw scan for this chip returned mostly search-engine query
> URLs (`lite.duckduckgo.com/lite/?q=...`) and aggregator pages. Those are **not sources** and
> are deliberately excluded from the table below. Every entry here is either a vendor primary
> page, a primary API endpoint, or a named press/financial outlet.

### Primary — vendor

| Resource | URL | Layer |
|----------|-----|-------|
| MetaX newsroom — WAIC 2026 item (S600 + X300) | https://www.metax-tech.com/en/ndetail/12629.html | Hardware / Company |
| MetaX newsroom — MiniMax H3 Day-0 adaptation (2026-08-03) | https://www.metax-tech.com/ndetail/12632.html | Software / Ecosystem |
| MetaX newsroom index | https://www.metax-tech.com/en/news.html?cid=15 | Company |
| MetaX product catalog — all series | https://www.metax-tech.com/en/goods/prod.html?cid=3 | Hardware |
| MetaX N300 product page | https://www.metax-tech.com/en/goods/prod.html?cid=106&id=66 | Hardware |
| MetaX N300 Server product page | https://www.metax-tech.com/en/goods/prod.html?cid=110&id=69 | Hardware / System |
| MetaX C550 3D Mesh Supernode | https://www.metax-tech.com/en/goods/prod.html?cid=112&id=55 | Interconnect |
| MetaX C550 Server | https://www.metax-tech.com/en/goods/prod.html?cid=110&id=44 | Hardware / System |
| MXMACA platform page | https://www.metax-tech.com/en/goods/platform.html?cid=4 | Software |
| MetaX developer portal | https://developer.metax-tech.com/ | Software |

### Primary — APIs and code

| Resource | URL | Layer |
|----------|-----|-------|
| vLLM-metax releases (GitHub API) | https://api.github.com/repos/MetaX-MACA/vLLM-metax/releases | Framework |
| metax-maca org repos (GitHub API) | https://api.github.com/orgs/metax-maca/repos | Software (all layers) |
| MetaX-MACA GitHub org | https://github.com/metax-maca | Software |
| vLLM-metax repo | https://github.com/MetaX-MACA/vLLM-metax | Framework |
| mcpy repo | https://github.com/MetaX-MACA/mcpy | Compiler / Kernel |
| vLLM RFC #23157 — maca backend | https://github.com/vllm-project/vllm/issues/23157 | Framework |
| HAMi MetaX support docs | https://github.com/Project-HAMi/HAMi/blob/master/docs/metax-support.md | Driver / K8s |

### Named press / financial

| Resource | URL | Layer |
|----------|-----|-------|
| ITHome — WAIC 2026 MetaX coverage | https://www.ithome.com/0/978/460.htm | Hardware |
| Sina Finance — WAIC 2026 (2026-07-20) | https://finance.sina.com.cn/tech/shenji/2026-07-20/doc-iniimuyk4915570.shtml | Hardware / Company |
| Sina Finance — Sun Guoliang on C600 large-scale shipment (2026-07-08) | https://finance.sina.com.cn/wm/2026-07-08/doc-inihceis8322692.shtml | Company |
| Tencent News — FY2025 results call, C700 status (2026-04-08) | https://news.qq.com/rain/a/20260408A066PB00 | Roadmap |
| TrendForce CN — MetaX shipment note (2026-07-09) | https://www.trendforce.cn/industry-news/semiconductors/20260709-6285.html | Company |
| EastMoney — FY2025 / Q1 2026 financials | https://finance.eastmoney.com/a/202604293724854763 | Company |
| PEDaily — MetaX feature (2026-07-28) | https://news.pedaily.cn/20260728/135041.shtml | Company |

### Lower-confidence / use with caution

| Resource | URL | Why flagged |
|----------|-----|-------------|
| Baidu Baike — MXMACA 软件栈 | https://baike.baidu.com/item/MXMACA%E8%BD%AF%E4%BB%B6%E6%A0%88 | Only source for MXMACA 3.3.0.X internals (PyTorch 2.8, 2,650 operators, 92.94% CUDA migration, PaddlePaddle). Medium confidence — not a vendor changelog |
| ZOL — MetaX product listing | https://ai.zol.com.cn/1212/12124334.html | Aggregator product listing; useful for catalog cross-check only |
| Sohu aggregator write-up | https://www.sohu.com/a/1052096132_120988576 | Aggregator; source of the unconfirmed N300 "48 GB" and "14nm" figures. **Do not cite for specs** |

### Explicitly rejected figures from this scan

| Claim | Verdict | Reason |
|---|---|---|
| N300 has 48 GB memory | NOT CONFIRMED | Aggregator listings only; MetaX's own N300 page states no memory capacity |
| N300 is built on 14nm | NOT CONFIRMED / likely wrong | Same aggregator tier; inconsistent with N300 being a C600-generation derivative |
| S600 "combines Scale-up and Scale-out into one fusion architecture" | UNVERIFIED | Not found in MetaX's WAIC newsroom item or in ITHome/Sina coverage |
| S600 supersedes the C550 3D Mesh Supernode | REFUTED | C550 3D Mesh Supernode is still a current listed product; S600 is an additional line |
| "No evidence of any MXC700 disclosure" | REFUTED | Chairman Chen Weiliang disclosed C700 design/verification status on the 2026-04-08 FY2025 results call |
| C700 H100-parity; mass production late 2027 | UNVERIFIED | Aggregator write-ups only |
| C500X DragonFly, 16→64 GPUs across 8 machines | MEDIUM | Secondary reporting only; no vendor spec sheet |
