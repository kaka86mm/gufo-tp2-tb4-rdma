#!/bin/bash
# start-tp2-ws.sh — TP2 over usb4_rdma write-striping (stock-core regime).
# rank0 = 1.22 (this box), rank1 = 44 (ssh over WiFi mgmt net).
# Data path: RDMA rails; bootstrap/control: TCP over WiFi 192.168.110.0/24
# (thunderbolt_net is blacklisted in this regime — no 10.0.1.x anymore).
set -e
RANK1=matri@192.168.110.44
RANK0_IP=192.168.110.228
TOKEN=gufo-tp2-cluster
MODEL=/home/matri/models/qwen3.8-flash-next-abliterated/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MMPROJ=/home/matri/models/qwen3.8-flash-next/mmproj-BF16.gguf
MTP=/home/matri/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

[ -x /home/matri/gufo-tp2 ] || { echo "!! ~/gufo-tp2 binary missing"; exit 1; }
ls /sys/class/infiniband/ 2>/dev/null | grep -q usb4_rdma || { echo "!! no usb4_rdma rails on rank0"; exit 1; }

echo "== stopping single-node production containers =="
docker stop gufo >/dev/null 2>&1 || true
ssh -o BatchMode=yes $RANK1 'docker stop gufo-u >/dev/null 2>&1 || true'

rm -f /tmp/gufo-tp2-rank0.log /tmp/gufo-tp2-rank1.log

echo "== rank0 =="
setsid nohup env GPU_MAX_HW_QUEUES=1 LD_LIBRARY_PATH=/home/matri/gufo-libs \
  /home/matri/gufo-tp2 serve llm \
  --model "$MODEL" --mmproj "$MMPROJ" \
  --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-control-port 18516 --tp-control-token "$TOKEN" \
  --tp-rdma-device usb4_rdma0 \
  --host 0.0.0.0 --port 8080 \
  --sessions 3 --context 262144 \
  --cache-disk /home/matri/gufo-cache2 \
  > /tmp/gufo-tp2-rank0.log 2>&1 < /dev/null &

for i in $(seq 1 15); do
  sleep 1
  ss -tln | grep -q 18515 && { echo "rank0 bootstrap port up"; break; }
done
ss -tln | grep -q 18515 || { echo "!! rank0 bootstrap port never came up:"; tail -5 /tmp/gufo-tp2-rank0.log; exit 1; }

echo "== rank1 (44) =="
ssh -o BatchMode=yes $RANK1 "sg render -c 'setsid nohup env GPU_MAX_HW_QUEUES=1 LD_LIBRARY_PATH=/home/matri/gufo-libs \
  /home/matri/gufo-tp2 serve llm \
  --model $MODEL --mmproj $MMPROJ \
  --speculative mtp --mtp-model $MTP \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host $RANK0_IP \
  --tp-bootstrap-port 18515 --tp-control-port 18516 --tp-control-token $TOKEN \
  --tp-rdma-device usb4_rdma0 \
  --host 0.0.0.0 --port 8081 \
  --sessions 3 --context 262144 \
  --cache-disk /home/matri/gufo-cache \
  > /tmp/gufo-tp2-rank1.log 2>&1 < /dev/null'"

echo "== waiting for RDMA handshake / model load =="
for i in $(seq 1 120); do
  sleep 3
  grep -q "rdma_ready" /tmp/gufo-tp2-rank0.log 2>/dev/null && { echo "RDMA READY on rank0"; break; }
  grep -qiE "error|fatal" /tmp/gufo-tp2-rank0.log 2>/dev/null && { echo "RANK0 ERROR:"; tail -3 /tmp/gufo-tp2-rank0.log; exit 1; }
done
for i in $(seq 1 60); do
  curl -s -m 2 http://127.0.0.1:8080/v1/models | grep -q gufo && { echo "rank0 serving :8080"; break; }
  sleep 2
done
echo "== rank0 log tail =="; tail -8 /tmp/gufo-tp2-rank0.log
echo "== rank1 log tail =="; ssh -o BatchMode=yes $RANK1 'tail -5 /tmp/gufo-tp2-rank1.log 2>/dev/null'
echo "== DONE =="
