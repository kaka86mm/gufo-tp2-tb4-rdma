#!/bin/bash
# agg-bench.sh <n_users> — fixed-output aggregate benchmark
U=${1:?n_users}
API=http://127.0.0.1:8080/v1/chat/completions
M="Qwen3.8 Flash Next"
T0=$(date +%s%N)
for i in $(seq 1 "$U"); do
  curl -s -m 900 "$API" -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"User $i: write the word apple exactly 300 times separated by spaces, no other text.\"}],\"max_tokens\":1500,\"temperature\":0,\"reasoning_effort\":\"off\"}" \
    -o "/tmp/agg-$U-$i.json" &
done
wait
T1=$(date +%s%N)
python3 - "$U" "$T0" "$T1" <<'EOF'
import sys, json
u, t0, t1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tot = ok = 0
for i in range(1, u + 1):
    try:
        d = json.load(open(f"/tmp/agg-{u}-{i}.json"))
        tot += d.get("usage", {}).get("completion_tokens", 0)
        ok += 1
    except Exception:
        pass
wall = (t1 - t0) / 1e9
print(f"{u}-user fixed-output: {ok}/{u} ok, {tot} tokens in {wall:.1f}s -> {tot/wall:.1f} tok/s aggregate")
EOF
