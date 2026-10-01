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
if ! insmod /var/lib/tbv/ws/thunderbolt_ibverbs.ko \
     profile=linux_perf bind_services=1 allocate_rings=1 start_rings=1 negotiate_native=1 enable_tunnels=1 tbnet=prefer_rdma lanes=2 register_verbs=1 \
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
