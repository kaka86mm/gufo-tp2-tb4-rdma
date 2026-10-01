# gufo-tp2-tb4-rdma

**Turnkey 2× TP (tensor-parallel) LLM serving over USB4 / Thunderbolt 4 RDMA with write striping — for dual AMD Strix Halo (Ryzen AI MAX+ 395) hosts running [gufo](https://github.com/gufo-org/gufo).**

[中文文档](README.zh-CN.md)

Two 128 GB Strix Halo machines become one 250 GB inference pool: every transformer layer is split across both hosts, partial sums cross the cables as RDMA WRITEs striped over all rails of two USB4 links (up to 4 rails × 20 Gb/s × 2 lanes with the [thunderbolt-ibverbs](https://github.com/hellas-ai/thunderbolt-ibverbs) stack), and the model is served through gufo's TP2 transport ([neuhaus/gufo](https://github.com/neuhaus/gufo) `rdma` branch).

## Measured results (real, reproducible with `run/bench/`)

Hardware: 2× FAEX1 (Ryzen AI MAX+ 395, 128 GB unified), 2× USB4 40G cables, Ubuntu 24.04 kernel 7.0.0-34, model `Qwen3.8-Flash-Next UD-Q4_K_XL` (abliterated) + MTP draft (d=7), context 262,144, sessions 12.

| Workload | Single host | TP2 old stack (single rail) | **TP2 this repo (write striping)** |
|---|---:|---:|---:|
| Decode, 1 user (repetitive, MTP) | 59.4 t/s | 35 t/s | **66–70 t/s** |
| Aggregate, 8 users | 141 t/s | — | **123 t/s** |
| Aggregate, 12 users | — | — | **136 t/s** |
| Prefill (cold, 16–26K tokens) | ~1,630 t/s | 137 t/s | **~1,610 t/s** |
| Prefill at depth 258K | — | — | **1,159 t/s** |
| Decode at depth 258K | — | — | **53 t/s** |
| Deep-context follow-up (258K, disk cache hit) | — | — | **5.3 s** |

Prefill barely decays with depth and stays above the reference numbers published for the same transport; decode holds 53–70 t/s all the way to a full 262K context.

## What's in the box

```
kernel/   build-ws-core.sh     integration kernel build: westeri v7.2-rc1 tree +
                              local.nix patch series + matched thunderbolt_ibverbs
                              (per-host compile, no symlink-farm hacks)
deploy/   deploy-ws.sh         fast path: stock kernel core + ibverbs + dummy
                              netdev tbv0 + systemd units (both hosts)
          deploy-ws-core.sh    integration-core path: patched thunderbolt core at
                              boot with stock fallback, blacklists, unit wiring
          ws-roce-boot.sh      the boot-time RDMA bring-up script (params baked in)
          *.service            systemd units
run/      start-tp2.sh         coordinated TP2 launcher (rank0 local, rank1 over ssh)
          bench/               decode / aggregate / deep-context / cache-hit benches
docs/     BENCHMARKS.md        full data tables incl. losing configurations
          PITFALLS.md          everything that bit us, so it doesn't bite you
```

## Quick start

**Fast path — stock kernel (Ubuntu 7.0+, any 6.14+ likely works)**

1. Build `thunderbolt_ibverbs` **on each host** against that host's own headers (same version string ≠ same build — see PITFALLS #1), with the activation parameters the upstream README's TP2 example omits:
   `profile=linux_perf bind_services=1 allocate_rings=1 start_rings=1 negotiate_native=1 enable_tunnels=1 tbnet=prefer_rdma lanes=2 register_verbs=1 roce_netdev=tbv0 native_write_striping=1`
2. On both hosts: `sudo bash deploy-ws-core.sh` style deployment — blacklists `thunderbolt_net` (it steals DMA rings), creates the dummy `tbv0` netdev for RoCE GIDs (10.77.0.1/.2), installs the boot units.
3. Connect **two** USB4 cables between the hosts, verify both links trained at `20.0 Gb/s x2`:
   `cat /sys/bus/thunderbolt/devices/*-*/rx_speed` — if a link trained down or is wedged, re-plug the cable (warm reboots do **not** reset the TB PHYs — PITFALLS #5).
4. Launch: `./run/start-tp2.sh` (rank0 local, rank1 via ssh; adjust IPs/paths/model at the top), wait for `rdma_ready`.

**Integration kernel path** — if your stock kernel's `thunderbolt` core rejects service binding (`bind_services=1` → EINVAL, see PITFALLS #2), build the matched core with `kernel/build-ws-core.sh` on each host and run `deploy/deploy-ws-core.sh`. This gives you the source-aware XDomain control path and rails that register single-ended.

## Tuning conclusions (measured, don't re-litigate)

- `zcopy_min_bytes=4096` — **worse** for decode-size messages (dma-buf mapping beats the copy it saves only for large writes)
- `native_write_stripe_min_bytes=16384` / `native_tx_max_inflight=128` — hurt single-stream decode; defaults win
- `sessions=12` beats 8 and 16 for aggregate; MTP stays **on** at every concurrency (off = −45% aggregate; the verify batch amortizes the cross-host exchange)
- MTP `--draft-tokens` is capped at 7 by the Flash-Next sidecar
- Disk cache: set `--cache-disk-staging-bytes` ≥ your largest snapshot (~4 GB for a 262K session) or deep-context follow-ups silently re-prefill; match `--cache-disk-bytes` to your GPU KV headroom

## Credits & licenses

- Transport: [hellas-ai/thunderbolt-ibverbs](https://github.com/hellas-ai/thunderbolt-ibverbs) (and the `feat/write-striping` branch it grew) — GPL-2.0
- Engine: [gufo-org/gufo](https://github.com/gufo-org/gufo) + the TP2 RDMA fork [neuhaus/gufo](https://github.com/neuhaus/gufo) (upstream in review) — MIT
- Scripts in this repo: MIT. Kernel patches referenced (not redistributed) live in the upstream repo.
