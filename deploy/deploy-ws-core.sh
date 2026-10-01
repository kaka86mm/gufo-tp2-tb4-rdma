#!/usr/bin/env bash
# deploy-ws-core.sh — switch this box to the integration core (v7.2-rc1 + 9
# local patches): patched thunderbolt core at boot via ws-core.service with a
# stock fallback, thunderbolt_ibverbs from /var/lib/tbv/ws/. Rollback:
# disable ws-core.service + remove 01-ws-core.conf + restore ws-roce-boot.sh.
set -euo pipefail
WS=/var/lib/tbv/ws
[ -f "$WS/thunderbolt-ws.ko" ] && [ -f "$WS/thunderbolt_ibverbs.ko" ] || { echo "!! ws set missing"; exit 1; }

echo "== [1] stock core blacklist (patched core must win the udev race) =="
cat > /etc/modprobe.d/01-ws-core.conf <<'EOS'
# integration thunderbolt core wins at boot; explicit `modprobe -i thunderbolt`
# still works for emergencies when ws-core.service failed.
blacklist thunderbolt
install thunderbolt /bin/false
EOS

echo "== [2] ws-core.service =="
cat > /etc/systemd/system/ws-core.service <<'EOS'
[Unit]
Description=Integration thunderbolt core (v7.2-rc1 + local.nix series)
DefaultDependencies=no
After=systemd-modules-load.service
Before=tbv-roce.service
Wants=tbv-roce.service
ConditionPathExists=/var/lib/tbv/ws/thunderbolt-ws.ko

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/bash -c 'if insmod /var/lib/tbv/ws/thunderbolt-ws.ko; then echo "ws-core loaded"; else echo "ws-core FAILED - falling back to stock (RDMA degraded)"; modprobe -i thunderbolt; fi; true'
ExecStop=/bin/true

[Install]
WantedBy=multi-user.target
EOS
systemctl daemon-reload
systemctl enable ws-core.service

echo "== [3] ws-roce-boot.sh -> ws ibverbs =="
sed -i 's|KO=/var/lib/tbv/thunderbolt_ibverbs.ko|KO=/var/lib/tbv/ws/thunderbolt_ibverbs.ko|' /var/lib/tbv/ws-roce-boot.sh
sed -i 's|insmod /var/lib/tbv/thunderbolt_ibverbs.ko|insmod /var/lib/tbv/ws/thunderbolt_ibverbs.ko|' /var/lib/tbv/ws-roce-boot.sh
# if the core is not loaded when roce runs, insmod the patched one first
sed -i 's|lsmod | grep -qw thunderbolt | modprobe thunderbolt 2>\&1|lsmod \| grep -qw thunderbolt \|\| insmod /var/lib/tbv/ws/thunderbolt-ws.ko 2>\&1\|modprobe thunderbolt 2>\&1|' /var/lib/tbv/ws-roce-boot.sh || true
grep -n "thunderbolt-ws.ko\|ws/thunderbolt_ibverbs" /var/lib/tbv/ws-roce-boot.sh || { echo "!! roce script not updated"; exit 1; }

echo "== [4] keep stock-matched ibverbs for rollback =="
mkdir -p /var/lib/tbv/stock-match
cp -f /var/lib/tbv/thunderbolt_ibverbs.ko /var/lib/tbv/stock-match/ 2>/dev/null || true

echo "== DEPLOYED on $(hostname): reboot to activate =="
