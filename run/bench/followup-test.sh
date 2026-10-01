#!/bin/bash
# followup-test.sh — REAL two-turn deep conversation: req1 generates, its actual
# output is fed back verbatim in req2 -> disk-cache restore must hit.
API=http://127.0.0.1:8080/v1/chat/completions
N=${1:-9900}
S=$RANDOM$RANDOM
python3 - "$N" "$S" <<'EOF'
import json, random, sys
n, seed = int(sys.argv[1]), int(sys.argv[2])
random.seed(seed)
words = ["mountain","rain","fox","lazy","dog","autumn","pass","vivid","silver","echo",
         "lantern","harbor","quartz","meadow","ember","tundra","cinder","fjord","wren","thistle"]
body = "Study these notes.\n" + "\n".join(
    f"note {i}: " + " ".join(random.choices(words, k=14)) for i in range(n))
json.dump({"model":"Qwen3.8 Flash Next",
           "messages":[{"role":"user","content":body+"\nReply with exactly: OK"}],
           "max_tokens":8,"temperature":0,"reasoning_effort":"off"}, open("/tmp/fu1.json","w"))
EOF
curl -s -m 900 $API -H "Content-Type: application/json" -d @/tmp/fu1.json -o /tmp/fu1-o.json
python3 - <<'EOF'
import json
r = json.load(open("/tmp/fu1-o.json"))
msgs = [{"role":"user","content": json.load(open("/tmp/fu1.json"))["messages"][0]["content"]},
        {"role":"assistant","content": r["choices"][0]["message"]["content"]},
        {"role":"user","content":"Now count from 1 to 50, one per line."}]
json.dump({"model":"Qwen3.8 Flash Next","messages":msgs,
           "max_tokens":160,"temperature":0,"reasoning_effort":"off"}, open("/tmp/fu2.json","w"))
print("assistant turn fed back verbatim:", repr(r["choices"][0]["message"]["content"][:40]))
EOF
T0=$(date +%s%N); curl -s -m 900 $API -H "Content-Type: application/json" -d @/tmp/fu2.json -o /tmp/fu2-o.json; T1=$(date +%s%N)
python3 - "$T0" "$T1" <<'EOF'
import json, sys
wall = (int(sys.argv[2])-int(sys.argv[1]))/1e9
u = json.load(open("/tmp/fu2-o.json")).get("usage",{})
print(f"follow-up at depth {u.get('prompt_tokens',0)}: wall={wall:.1f}s -> {'CACHE HIT ✓' if wall < 20 else 'still slow'}")
EOF
