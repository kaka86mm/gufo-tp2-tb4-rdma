#!/usr/bin/env bash
# deploy-ws.sh <tbv0-ip> — switch this box to the write-striping RDMA stack:
#   stock kernel thunderbolt core (patched core retired), thunderbolt_net
#   blacklisted (all DMA rings -> RDMA rails), dummy netdev tbv0 for RoCE GIDs,
#   neuhaus thunderbolt_ibverbs (native_write_striping) from /tmp/ws-tbv.
# Rollback: /var/lib/tbv/old503c5ae/ + old units kept as *.unit-bak.
set -euo pipefail
TBV0_IP=${1:?usage: deploy-ws.sh <tbv0-ip>}
NEW_KO=/tmp/ws-tbv/kernel/thunderbolt_ibverbs.ko
[ -f "$NEW_KO" ] || { echo "!! $NEW_KO missing"; exit 1; }
modinfo -F vermagic "$NEW_KO" | grep -q "$(uname -r)" || { echo "!! vermagic mismatch"; exit 1; }

echo "== [1] back up old stack =="
mkdir -p /var/lib/tbv/old503c5ae
for f in thunderbolt-patched.ko thunderbolt_net.ko thunderbolt_ibverbs.ko; do
  [ -f "/var/lib/tbv/$f" ] && mv "/var/lib/tbv/$f" "/var/lib/tbv/old503c5ae/$f"
done
cp -f "$NEW_KO" /var/lib/tbv/thunderbolt_ibverbs.ko
# nhi_throttle.ko stays: NHI reg tweak, core-agnostic
ls -la /var/lib/tbv/

echo "== [2] tbv0 ip config =="
echo "$TBV0_IP" > /var/lib/tbv/ws-tbv0-ip

echo "== [3] boot script =="
cat > /var/lib/tbv/ws-roce-boot.sh <<'EOS'
#!/usr/bin/env bash
# ws-roce-boot.sh — write-striping RDMA bring-up on the STOCK kernel core.
# thunderbolt_net is blacklisted (it would steal one DMA ring per USB4
# controller from the RDMA rails); the dummy netdev tbv0 provides the IPv4
# the RoCE GID table needs. Runs from tbv-roce.service at boot, or manually.
set +e
TBV0_IP=$(cat /var/lib/tbv/ws-tbv0-ip 2>/dev/null || echo 10.77.0.1)

systemctl stop irqbalance 2>/dev/null || true

# wait for USB4 domains to enumerate (cables/peers may come up later)
for i in $(seq 1 60); do
  n=$(ls /sys/bus/thunderbolt/devices/ 2>/dev/null | grep -c '^domain')
  [ "$n" -ge 1 ] && break
  sleep 2
done

# dummy netdev for the RoCE GID table (any point in boot is fine)
if ! ip link show tbv0 >/dev/null 2>&1; then
  ip link add tbv0 type dummy
  ip addr add "${TBV0_IP}/24" dev tbv0 2>/dev/null
  ip link set tbv0 up
fi
echo "[ws] tbv0: $(ip -4 -br addr show tbv0 2>/dev/null)"

modprobe -a configfs ib_core ib_uverbs 2>&1
for m in ib_core ib_uverbs; do
  lsmod | grep -qw "$m" || echo "[ws] WARN dep $m not loaded"
done

# stock thunderbolt core (udev normally loads it via NHI modalias)
lsmod | grep -qw thunderbolt || modprobe thunderbolt 2>&1
lsmod | grep -qw thunderbolt || { echo "[ws] FAIL thunderbolt core not loaded"; exit 1; }

rmmod thunderbolt_ibverbs 2>/dev/null
if ! insmod /var/lib/tbv/thunderbolt_ibverbs.ko \
     profile=linux_perf tbnet=prefer_rdma lanes=2 register_verbs=1 \
     roce_netdev=tbv0 native_write_striping=1 2>/dev/null; then
  echo "[ws] FAIL insmod:"; dmesg | grep thunderbolt_ibverbs | tail -5; exit 1
fi
echo "[ws] thunderbolt_ibverbs loaded (write_striping on)"

# NHI IRQ throttle 8us (mainline default 128us sets the latency floor)
rmmod nhi_throttle 2>/dev/null
insmod /var/lib/tbv/nhi_throttle.ko ns=8000 2>/dev/null \
  && echo "[ws] nhi_throttle 8us" || echo "[ws] nhi_throttle skipped (non-fatal)"

sleep 3
echo "[ws] rails: $(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')"
for d in /sys/bus/thunderbolt/devices/*-*; do
  [ -e "$d/rx_speed" ] && echo "[ws] $(basename "$d") rx $(cat "$d/rx_speed") x$(cat "$d/rx_lanes") / tx $(cat "$d/tx_speed") x$(cat "$d/tx_lanes")"
done
exit 0
EOS
chmod +x /var/lib/tbv/ws-roce-boot.sh

echo "== [4] units =="
systemctl disable tbv-thunderbolt-patched.service 2>/dev/null || true
[ -f /etc/systemd/system/tbv-roce.service ] && cp /etc/systemd/system/tbv-roce.service /etc/systemd/system/tbv-roce.service.unit-bak
cat > /etc/systemd/system/tbv-roce.service <<'EOS'
[Unit]
Description=USB4 RDMA write-striping stack (stock core + neuhaus ibverbs + tbv0)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/bash /var/lib/tbv/ws-roce-boot.sh
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOS
systemctl daemon-reload
systemctl enable tbv-roce.service

echo "== [5] modprobe.d =="
# retire the patched-core blacklist so the STOCK core can load
[ -f /etc/modprobe.d/00-tbv-thunderbolt.conf ] && mv /etc/modprobe.d/00-tbv-thunderbolt.conf /etc/modprobe.d/00-tbv-thunderbolt.conf.disabled
cat > /etc/modprobe.d/59-tbv-ws.conf <<'EOS'
# write-striping stack: keep thunderbolt_net off the links (it steals DMA
# rings from the RDMA rails); emergency: explicit `modprobe thunderbolt_net`
# still works when thunderbolt_ibverbs is unloaded.
blacklist thunderbolt_net
options thunderbolt_ibverbs profile=linux_perf tbnet=prefer_rdma lanes=2 register_verbs=1 roce_netdev=tbv0 native_write_striping=1
EOS

echo "== [6] udev: keep usb4_rdma* names (rdma-core rename breaks the provider) =="
cat > /etc/udev/rules.d/60-rdma-persistent-naming.rules <<'EOS'
# rdma-persistent-naming with usb4_rdma excluded (copied from stock rule,
# see thunderbolt-ibverbs README)
KERNEL!="hfi1*", KERNEL!="usb4_rdma*", PROGRAM="rdma_rename %k NAME_FALLBACK"
EOS

echo "== DEPLOY DONE on $(hostname) (tbv0=$TBV0_IP) =="
