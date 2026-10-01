#!/bin/bash
# prefill-fresh.sh <n_lines> — real prefill bench with a unique random prompt
N=${1:-750}
SEED=$RANDOM$RANDOM
F=/tmp/prefill-fresh-$SEED.json
python3 - "$N" "$SEED" "$F" <<'EOF'
import json, random, sys
n, seed, f = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(seed)
words = ["mountain","rain","fox","lazy","dog","autumn","pass","vivid","silver","echo",
         "lantern","harbor","quartz","meadow","ember","tundra","cinder","fjord","wren","thistle"]
lines = [f"note {i}: " + " ".join(random.choices(words, k=12)) for i in range(n)]
json.dump({"model":"Qwen3.8 Flash Next",
           "messages":[{"role":"user","content":"Study these notes.\n"+"\n".join(lines)+"\nReply with one word: OK"}],
           "max_tokens":8,"temperature":0,"reasoning_effort":"off"}, open(f,"w"))
EOF
T0=$(date +%s%N)
curl -s -m 900 http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @"$F" -o /tmp/prefill-out.json
T1=$(date +%s%N)
WALL_NS=$((T1-T0))
PT=$(python3 -c "import json;print(json.load(open('/tmp/prefill-out.json')).get('usage',{}).get('prompt_tokens',0))")
python3 -c "print(f'real prefill (fresh, seed=$SEED): $PT tokens in {$WALL_NS/1e9:.2f}s -> {$PT/($WALL_NS/1e9):.1f} tok/s')"
