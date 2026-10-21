#!/bin/bash
# start-tp2.sh — coordinated TP2 launcher (rank0 local, rank1 over ssh).
# Edit the variables below for your hosts and model paths (or export them).
# Data crosses the RDMA rails; the bootstrap/control channel is ordinary TCP
# between the hosts (any network they share works — LAN, direct link, ...).
set -e
RANK1="${RANK1:-user@rank1-host}"          # ssh target of the second host
RANK0_IP="${RANK0_IP:-rank0-ip}"           # address rank1 uses to reach rank0
TOKEN="${TOKEN:-change-me}"
MODEL="${MODEL:-/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
MMPROJ="${MMPROJ:-/path/to/mmproj-BF16.gguf}"
MTP="${MTP:-/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
BIN="${BIN:-$HOME/gufo-tp2}"               # the neuhaus rdma-branch build
LIBS="${LIBS:-$HOME/gufo-libs}"            # its runtime library directory
PORT0="${PORT0:-8080}"
PORT1="${PORT1:-8081}"

[ -x "$BIN" ] || { echo "!! $BIN binary missing"; exit 1; }
ls /sys/class/infiniband/ 2>/dev/null | grep -q usb4_rdma || { echo "!! no usb4_rdma rails on rank0"; exit 1; }

echo "== stopping single-host servers on both hosts =="
docker stop gufo >/dev/null 2>&1 || true
ssh -o BatchMode=yes "$RANK1" 'docker stop gufo-u >/dev/null 2>&1 || true'

rm -f /tmp/gufo-tp2-rank0.log /tmp/gufo-tp2-rank1.log

echo "== rank0 =="
setsid nohup env GPU_MAX_HW_QUEUES=1 LD_LIBRARY_PATH="$LIBS" \
  "$BIN" serve llm \
  --model "$MODEL" --mmproj "$MMPROJ" \
  --speculative mtp --mtp-model "$MTP" \
  --tp-world-size 2 --tp-rank 0 \
  --tp-bootstrap-port 18515 --tp-control-port 18516 --tp-control-token "$TOKEN" \
  --tp-rdma-device usb4_rdma0 \
  --host 0.0.0.0 --port "$PORT0" \
  --sessions 12 --context 262144 \
  --cache-disk "$HOME/gufo-cache2" --cache-disk-bytes 68719476736 \
  --cache-disk-staging-bytes 6442450944 \
  > /tmp/gufo-tp2-rank0.log 2>&1 < /dev/null &

for i in $(seq 1 15); do
  sleep 1
  ss -tln | grep -q 18515 && { echo "rank0 bootstrap port up"; break; }
done
ss -tln | grep -q 18515 || { echo "!! rank0 bootstrap port never came up:"; tail -5 /tmp/gufo-tp2-rank0.log; exit 1; }

echo "== rank1 =="
ssh -o BatchMode=yes "$RANK1" "sg render -c 'setsid nohup env GPU_MAX_HW_QUEUES=1 LD_LIBRARY_PATH=\"$LIBS\" \
  \"$BIN\" serve llm \
  --model \"$MODEL\" --mmproj \"$MMPROJ\" \
  --speculative mtp --mtp-model \"$MTP\" \
  --tp-world-size 2 --tp-rank 1 \
  --tp-bootstrap-host \"$RANK0_IP\" \
  --tp-bootstrap-port 18515 --tp-control-port 18516 --tp-control-token \"$TOKEN\" \
  --tp-rdma-device usb4_rdma0 \
  --host 0.0.0.0 --port \"$PORT1\" \
  --sessions 12 --context 262144 \
  --cache-disk \"$HOME/gufo-cache\" --cache-disk-bytes 68719476736 \
  --cache-disk-staging-bytes 6442450944 \
  > /tmp/gufo-tp2-rank1.log 2>&1 < /dev/null'"

echo "== waiting for RDMA handshake =="
for i in $(seq 1 120); do
  sleep 3
  grep -q "rdma_ready" /tmp/gufo-tp2-rank0.log 2>/dev/null && { echo "RDMA READY on rank0"; break; }
  grep -qiE "error|fatal" /tmp/gufo-tp2-rank0.log 2>/dev/null && { echo "RANK0 ERROR:"; tail -3 /tmp/gufo-tp2-rank0.log; exit 1; }
done
for i in $(seq 1 60); do
  curl -s -m 2 "http://127.0.0.1:$PORT0/v1/models" | grep -q gufo && { echo "rank0 serving :$PORT0"; break; }
  sleep 2
done
echo "== rank0 log tail =="; tail -8 /tmp/gufo-tp2-rank0.log
echo "== rank1 log tail =="; ssh -o BatchMode=yes "$RANK1" 'tail -5 /tmp/gufo-tp2-rank1.log 2>/dev/null'
echo "== DONE =="
