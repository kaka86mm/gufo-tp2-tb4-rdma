#!/usr/bin/env bash
# build-ws-core.sh v2 — integration stack build WITHOUT the symlink farm:
#   patched thunderbolt.h injected via EXTRA_CFLAGS include path (system
#   headers untouched), CONFIG gates force-enabled in the tree Makefile.
# westeri v7.2-rc1 + local.nix series (skip-if-upstream) -> core+net+ibverbs.
# Output: /var/lib/tbv/ws/ . Run ON EACH HOST (kernels differ).
set -euo pipefail
WESTERI=${WESTERI:-$HOME/westeri-tb}
PATCHDIR=${PATCHDIR:-$HOME/ws-tbv-src/kernel-workflow/patches}
IBV=${IBV:-$HOME/ws-tbv-src}
OUT=/var/lib/tbv/ws
KVER=$(uname -r)
WORK=/tmp/ws-build-$KVER
CAP="nice -n 19 ionice -c3"
KDIR=/lib/modules/$KVER/build
[ -f "$KDIR/Makefile" ] || { echo "!! no headers for $KVER"; exit 1; }

SERIES="
0002-thunderbolt-tunnel-add-dma-priority-weight-params.patch
0003-thunderbolt-nhi-add-ring-debugfs-instrumentation.patch
0006-thunderbolt-xdomain-bound-response-copy.patch
0004-thunderbolt-nhi-clear-pending-before-unmask.patch
0005-thunderbolt-xdomain-log-unmatched-protocol-uuids.patch
0007-thunderbolt-xdomain-pass-source-to-protocol-handlers.patch
0008-thunderbolt-xdomain-pin-protocol-handler-owner.patch
0009-thunderbolt-xdomain-match-properties-by-identity.patch
0010-thunderbolt-xdomain-drain-protocol-callbacks-on-unr.patch
"

echo "== [0] tree + local.nix series =="
rm -rf "$WORK"; mkdir -p "$WORK"
cp -a "$WESTERI/drivers" "$WESTERI/include" "$WORK/"
cd "$WORK"
for p in $SERIES; do
  if git apply -C1 "$PATCHDIR/$p" 2>/dev/null; then
    echo "applied $p"
  elif pout=$(patch -p1 --fuzz=3 --forward < "$PATCHDIR/$p" 2>&1; true); echo "$pout" | grep -q "Reversed (or previously applied)"; then
    echo "already upstream: $p (skipped)"
  else
    echo "!! $p conflicted"; exit 1
  fi
done
grep -q callback_xd "$WORK/include/linux/thunderbolt.h" || { echo "!! series did not take"; exit 1; }

# force-enable CONFIG-gated objects the patches/USB4STREAM need
sed -i 's/thunderbolt-$(CONFIG_USB4_CONFIGFS)/thunderbolt-y/' "$WORK/drivers/thunderbolt/Makefile" || true
# deterministic header: copy the patched thunderbolt.h beside the sources and
# switch angle includes to quoted (local-dir resolution wins, no -I games)
inject_header() {
  local d=$1
  cp "$WORK/include/linux/thunderbolt.h" "$d/thunderbolt.h"
  grep -rl '#include <linux/thunderbolt.h>' "$d" --include='*.c' --include='*.h' 2>/dev/null |
    xargs -r sed -i 's|#include <linux/thunderbolt.h>|#include "thunderbolt.h"|' || true
}
inject_header "$WORK/drivers/thunderbolt"
inject_header "$WORK/drivers/net/thunderbolt"
inject_header "$IBV/kernel"
INC=""

echo "== [1] core =="
$CAP make -j1 -C "$KDIR" M="$WORK/drivers/thunderbolt" clean >/dev/null 2>&1 || true
$CAP make -j"$(nproc)" -C "$KDIR" M="$WORK/drivers/thunderbolt" EXTRA_CFLAGS="$INC" 2>&1 | grep -E "error|Error|warning: implicit" | head -10 || true
[ -f "$WORK/drivers/thunderbolt/thunderbolt.ko" ] || { echo "!! core build failed"; exit 1; }
sudo mkdir -p "$OUT"
sudo cp -f "$WORK/drivers/thunderbolt/thunderbolt.ko" "$OUT/thunderbolt-ws.ko"
echo "core ok"

echo "== [2] net =="
$CAP make -j1 -C "$KDIR" M="$WORK/drivers/net/thunderbolt" clean >/dev/null 2>&1 || true
$CAP make -j"$(nproc)" -C "$KDIR" M="$WORK/drivers/net/thunderbolt" EXTRA_CFLAGS="$INC" \
  KBUILD_EXTRA_SYMBOLS="$WORK/drivers/thunderbolt/Module.symvers" 2>&1 | grep -E "error|Error" | head -5 || true
[ -f "$WORK/drivers/net/thunderbolt/thunderbolt_net.ko" ] || { echo "!! net build failed"; exit 1; }
sudo cp -f "$WORK/drivers/net/thunderbolt/thunderbolt_net.ko" "$OUT/thunderbolt_net.ko"
echo "net ok"

echo "== [3] ibverbs (same include override + core symvers) =="
$CAP make -C "$IBV/kernel" KDIR="$KDIR" EXTRA_CFLAGS="$INC" clean >/dev/null 2>&1 || true
$CAP make -C "$IBV/kernel" KDIR="$KDIR" EXTRA_CFLAGS="$INC" \
  KBUILD_EXTRA_SYMBOLS="$WORK/drivers/thunderbolt/Module.symvers" modules 2>&1 | grep -E "error|Error" | head -5 || true
[ -f "$IBV/kernel/thunderbolt_ibverbs.ko" ] || { echo "!! ibverbs build failed"; exit 1; }
sudo cp -f "$IBV/kernel/thunderbolt_ibverbs.ko" "$OUT/thunderbolt_ibverbs.ko"
echo "ibverbs ok"

echo "== manifest =="
sudo bash -c "for k in $OUT/*.ko; do printf '  %-28s %s %s\n' \$(basename \$k) \"\$(modinfo -F vermagic \$k)\" \"\$(modinfo -F srcversion \$k | cut -c1-12)\"; done"
echo "OK: $OUT on $(hostname)"
