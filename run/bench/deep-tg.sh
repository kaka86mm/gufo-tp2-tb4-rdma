#!/bin/bash
# deep-tg.sh <n_lines> — single-request deep tg: prefill+decode in one call,
# tg recovered as ct / (wall - depth/pp_rate), pp_rate ~1600 measured flat.
API=http://127.0.0.1:8080/v1/chat/completions
N=${1:?n_lines}
PPR=${2:-1600}
S=$RANDOM$RANDOM
F=/tmp/deeptg-$S.json
python3 - "$N" "$F" "$S" <<'EOF'
import json, random, sys
n, f, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
random.seed(seed)
words = ["mountain","rain","fox","lazy","dog","autumn","pass","vivid","silver","echo",
         "lantern","harbor","quartz","meadow","ember","tundra","cinder","fjord","wren","thistle"]
body = "Study these notes.\n" + "\n".join(
    f"note {i}: " + " ".join(random.choices(words, k=14)) for i in range(n))
json.dump({"model":"Qwen3.8 Flash Next",
           "messages":[{"role":"user","content":body + "\nNow count from 1 to 300, one number per line, nothing else."}],
           "max_tokens":900,"temperature":0,"reasoning_effort":"off"}, open(f,"w"))
EOF
T0=$(date +%s%N); curl -s -m 1800 $API -H "Content-Type: application/json" -d @"$F" -o /tmp/deeptg-o.json; T1=$(date +%s%N)
python3 - "$T0" "$T1" "$PPR" <<'EOF'
import json, sys
t0,t1,ppr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
u = json.load(open("/tmp/deeptg-o.json")).get("usage",{})
wall=(t1-t0)/1e9; depth=u.get("prompt_tokens",0); ct=u.get("completion_tokens",0)
pf = depth/ppr
print(f"depth={depth}: wall={wall:.1f}s (prefill~{pf:.1f}s @{ppr}/s) ct={ct} -> tg@depth = {ct/max(wall-pf,0.1):.1f} tok/s")
EOF
