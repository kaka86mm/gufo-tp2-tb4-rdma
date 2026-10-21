# Benchmarks

All numbers measured on the reference rig, 2026-10-01, unless noted.
Engine-internal figures (`decode_tps`, `prefill_tps`) come from gufo's per-request
HTTP telemetry — prefer these over wall-clock arithmetic (HTTP wall mixes in
prefill and hides depth-dependent prefill decay).

## Rig

- 2× FAEX1 hosts, Ryzen AI MAX+ 395 (Strix Halo, gfx1151), 128 GB unified memory each
- 2× USB4 40G cables, both links trained `20.0 Gb/s x2` (4 RDMA rails total:
  `usb4_rdma0/1` + `usb4_rdma5/6`)
- Ubuntu 24.04, kernel 7.0.0-34-generic (per-host builds — see PITFALLS #1)
- thunderbolt_ibverbs, `native_write_striping=1`, defaults elsewhere
- gufo TP2 (neuhaus rdma branch), Qwen3.8-Flash-Next UD-Q4_K_XL (abliterated)
  + MTP Q8_0 draft (d=7), context 262,144, sessions 12, disk cache 64G/staging 6G per host

## Decode, single user (counting task, MTP acceptance ≈ 100%)

| Setup | tok/s |
|---|---:|
| old TP2 stack (single rail, patched 503c5ae core) | 35 |
| single host (reference) | ~59 |
| TP2 write striping, stock core | 65–67 |
| TP2 write striping, integration core (v7.2-rc1 + local patches) | **66–68** |
| TP2 at depth 37K (integration core) | **70.1** |

Upstream's published RDMA figure for the same transport at depth 0 is 75.6
(engine-internal methodology); our HTTP end-to-end runs ~10% below engine
telemetry, which accounts for the difference.

## Decode at depth (integration core)

| Depth (tokens) | prefill_tps | decode_tps |
|---:|---:|---:|
| 36.8K | 1,753 | 70.1 |
| 75.0K | 1,684 | 67.0 |
| 150.8K | 1,448 | 56.7–64.0 |
| 258.1K | 1,159 | **53.0** |

Upstream reference at 258K: prefill 1,130, tg 50.2 — this rig is at or above at
every depth, with flatter prefill decay.

## Aggregate (fixed-output task, sessions 12)

| Users | tok/s |
|---:|---:|
| 8 | 122.7 |
| 12 | **135.9** |
| 12, sessions=3 (misconfigured) | 86 |

Single-host 8-user reference: 141. TP2 aggregate approaches single-host at 12
users; per-step cross-host exchange latency is the floor (see TP2.md in the
gufo fork for the design).

## Losing configurations (measured so you can skip them)

| Change vs. reference | Effect |
|---|---|
| `zcopy_min_bytes=4096` | 8-user aggregate 120.9 → 112.8 (−7%) |
| `native_write_stripe_min_bytes=16384` + `native_tx_max_inflight=128` | single-stream 65–67 → 58.7 |
| MTP off | 8-user aggregate 126 → **68.8 (−45%)** |
| `--draft-tokens 12` | refuses to load: sidecar caps at 7 |
| benchmark with a repeated prompt | "prefill 25,871 tok/s" — that's the disk cache, not compute |
| wall-clock tg at depth (assuming flat prefill rate) | wildly wrong; prefill decays 1753→1159 with depth |

## Cache

- Full-depth (262K) session KV snapshot: **3.97 GB per rank** (~15.4 B/token/rank,
  ~7.95 GB per session across both ranks)
- GPU KV headroom per rank after weights: ~66 GB → ~16 full-depth sessions
- Disk cache 64G + staging 6G per host: 258K follow-up after restart = **5.3 s**
  (vs 240 s cold re-prefill); restore requires exact prefix continuation —
  feed back the model's actual generated text.

## NPU hybrid addendum (2026-10-22)

A newer gufo NPU build (gufo-npu:20261020 image, XRT 2.25) offloads 48 layers'
shared experts, 12 layers' attention query/gate and 36 layers' z-projection to
the Strix Halo XDNA2 NPU (3.26 GiB staged) and runs the rest on the GPU — one
host, no RDMA. Same model/MTP/context as the tables above:

| Workload | TP2 (2 hosts) | NPU hybrid (1 host) |
|---|---:|---:|
| decode 1 user, repetitive | 66–70 | **92–102** |
| decode 1 user, prose | ~24.5 | **36** |
| decode at 37K depth | 70.1 | **96.3** |
| decode at 75K | 67.0 | **94.5** |
| decode at 151K | 56.7–64 | **92.1** |
| decode at 258K | 53.0 | **90.0** |
| prefill 16–26K fresh | ~1610 | 1571–1701 |
| prefill at 258K | 1159 | **1461** |
| aggregate 4 users | — | 109.6 (sessions=4, ctx 64K) |
| APU Tctl under load | 91–94°C (throttling) | **67–70°C** |

Requirements learned the hard way: IOMMU must be enabled (SVA); the firmware
file must be `npu_7.sbin` (an older `npu.sbin` firmware caused an amdxdna
mailbox EINVAL that later panicked the host); per-session state is ~4G KV +
7.5G scratch, so sessions x context is capacity-bound on a single host
(262144x1 fits, 65536x4 fits); the disk-cache dir must be owned by the server
user with 0700; after an OOM kill, leaked GTT memory only clears on reboot.
