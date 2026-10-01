# Pitfalls

Every one of these cost us hours. In rough order of likelihood to bite you.

## 1. Same kernel version string ≠ same kernel build

Two hosts both reporting `7.0.0-34-generic` had **different headers packages**
(different `thunderbolt.h` generations) and different `CONFIG` (config hash
mismatch). A module built on host A then loaded on host B fails with
`module_layout` or symbol CRC mismatches — or worse, compiles a degraded path
that only fails at runtime.

**Rule: build `thunderbolt_ibverbs` (and any core you patch) on each host,
against that host's own headers.** Runtime cores may be identical; build inputs
are not.

## 2. `bind_services=1` → EINVAL, silently

If the host's headers generation lacks the service-properties API, the module
still **compiles**, loads, prints `core ready` — and then `bind_services=1`
fails insmod with `Invalid parameters` and **zero kernel log explanation**.
Bisect params to find it. Fix: use the newer headers generation, or build the
integration core (`kernel/build-ws-core.sh`). Upstream's own TP2 example omits
`bind_services/allocate_rings/start_rings/negotiate_native/enable_tunnels`
entirely — with defaults (all N) the module loads fine and **no rails ever
appear**. The "Load And Use" section of their README has the full set; trust
that, not the example.

## 3. modversions CRC mismatches — three distinct causes we hit

- module built vs. headers from a different generation (see #1/#2)
- building in a "KDIR farm" with a modified `auto.conf` poisons the CRCs
- building the dependent module **without**
  `KBUILD_EXTRA_SYMBOLS=<core-build>/Module.symvers` — modpost then takes the
  *stock* core's CRCs from the kernel tree and the module mismatches your
  patched core on exactly the symbols the patches touched.

## 4. Kernel-tree fetch and build trivia

- The maintainer's `next` branch gets rewritten; commit SHAs cited in older
  docs (`c866393…`, `b6dd8fc…`) are unresolvable. Use release **tags**
  (e.g. `v7.2-rc1`) — the content you need (USB4STREAM series) is merged there.
- You don't need a clone: ~40 files via the GitHub contents API + raw at a tag
  is a 6 MB fetch (a blobless kernel.org clone is 300+ MB).
- To use a patched `thunderbolt.h` deterministically: copy it **into the module
  source dir** and switch `#include <linux/thunderbolt.h>` to
  `#include "thunderbolt.h"` — no `-I` games, no touching system headers.
- The configfs Makefile gate is `thunderbolt-$(CONFIG_USB4_CONFIGFS)`, not
  `obj-$(…)` — sed accordingly.

## 5. Thunderbolt links wedge; warm reboots do not fix them

After a cable/host hiccup both links can end up unenumerated on both ends
(`boltctl` empty, no remote routers in `/sys/bus/thunderbolt/devices/`, zero
discovery events across reboots). We ruled out: stock vs. patched core, NHI
PCI remove+rescan on both ends, driver binding, power states. **Warm reboots
don't cut standby power to the TB PHYs.** Only physically re-plugging the cable
(or a full power-off) resets the link training state machines. If you manage
these hosts remotely, keep someone near the rack or file the upstream feature
request for a software NHI reset.

Also: links sometimes train at `10 Gb/s` after boot; re-plug trains them to
`20 Gb/s x2`. Check `rx_speed` after every boot.

## 6. `thunderbolt_net` must stay off the links

The distro loads it by modalias the moment a peer offers its network service;
it takes one DMA ring per USB4 controller away from the RDMA rails. Blacklist
it (`blacklist thunderbolt_net`; explicit `modprobe thunderbolt_net` still
works for emergencies). RoCE addressing does not need it — a dummy netdev
(`ip link add tbv0 type dummy` + an IPv4 addr) is enough for the GID table.

## 7. udev: keep usb4_rdma names AND scope the rule

rdma-core's persistent-naming rule renames devices by bus path, which breaks
the provider's name-based lookup. Copy the rule and exclude `usb4_rdma*` — but
**keep the `SUBSYSTEM=="infiniband"` match**, or `rdma_rename` fires for every
non-RDMA device on the box and spams the boot log.

## 8. Cancelled launchers keep launching

`nohup`/`setsid` launchers survive Ctrl-C of the wrapper. We once stacked two
rank0 processes, each loading a 52 GB half-model into the same unified-memory
host → OOM storm → sshd starved (ping answers, SSH banner times out — that
signature = userspace starvation, not a dead box). Kill with a self-match-safe
pattern: `pkill -f "gufo-tp2 [s]erve"` (bracket trick), and `pgrep -af` +
exclude your own ssh wrapper before counting processes.

## 9. Disk cache staging silently caps deep sessions

`--cache-disk-staging-bytes` defaults to ≤1 GiB; a 262K-token session snapshots
at ~4 GB per rank → store is **skipped** (`reason=staging_capacity`), and every
follow-up re-prefills (~240 s at 258K). Set staging ≥ your largest snapshot and
`--cache-disk-bytes` to match GPU KV headroom. Restore requires the new request
to *continue* the checkpointed sequence exactly — synthesize the assistant turn
wrong and you get `cache_miss_reason=prefix_changed` with a 99.98% common
prefix.

## 10. gufo telemetry > wall clock

Read `decode_tps` / `prefill_tps` / `acceptance_pct` / `cache=` from the HTTP
request-completion log lines. Wall-clock division by assumption (e.g. flat
prefill rate) produced numbers off by 5–10× for us at depth.
