#!/bin/bash
# bench-tp2.sh — decode + prefill benchmark against the TP2 endpoint (:8080)
# decode: repetitive counting (high MTP acceptance) — comparable to old-stack 35 tok/s
# prefill: ~10K-token prompt, 1-token completion — comparable to old-stack 137 tok/s
API=http://127.0.0.1:8080/v1/chat/completions
M="Qwen3.8 Flash Next"
for i in $(seq 1 90); do
  curl -s -m 3 $API/../models 2>/dev/null | grep -q gufo && break
  sleep 2
done

echo "=== decode bench (repetitive, max_tokens=600) ==="
T0=$(date +%s%N)
curl -s -m 600 $API -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 200, one number per line, nothing else.\"}],\"max_tokens\":600,\"temperature\":0,\"reasoning_effort\":\"off\"}" -o /tmp/tp2-dec.json
T1=$(date +%s%N)
python3 -c "
import json
r=json.load(open('/tmp/tp2-dec.json')); u=r.get('usage',{})
wall=($T1-$T0)/1e9; ct=u.get('completion_tokens',0)
print(f'decode: wall={wall:.1f}s completion={ct} -> tg={ct/wall:.1f} tok/s')"

echo "=== decode bench 2 (varied prose, max_tokens=400) ==="
T0=$(date +%s%N)
curl -s -m 600 $API -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a vivid 400-word essay about autumn rain on a mountain pass.\"}],\"max_tokens\":500,\"temperature\":0,\"reasoning_effort\":\"off\"}" -o /tmp/tp2-dec2.json
T1=$(date +%s%N)
python3 -c "
import json
r=json.load(open('/tmp/tp2-dec2.json')); u=r.get('usage',{})
wall=($T1-$T0)/1e9; ct=u.get('completion_tokens',0)
print(f'prose: wall={wall:.1f}s completion={ct} -> tg={ct/wall:.1f} tok/s')"

echo "=== prefill bench (~10K prompt tokens, 8 out) ==="
python3 - <<'EOF'
import json
# ~10K tokens: numbered lines are ~4-5 tokens each; use dense filler words
lines = [f"line {i}: the quick brown fox jumps over the lazy dog while rain falls on the mountain pass" for i in range(700)]
open('/tmp/tp2-prompt.json','w').write(json.dumps({
  "model": "Qwen3.8 Flash Next",
  "messages": [{"role":"user","content":"Memorize these lines.\n" + "\n".join(lines) + "\nNow reply with the single word OK."}],
  "max_tokens": 8, "temperature": 0, "reasoning_effort": "off"}))
EOF
T0=$(date +%s%N)
curl -s -m 900 $API -H "Content-Type: application/json" -d @/tmp/tp2-prompt.json -o /tmp/tp2-pre.json
T1=$(date +%s%N)
python3 -c "
import json
r=json.load(open('/tmp/tp2-pre.json')); u=r.get('usage',{})
wall=($T1-$T0)/1e9; pt=u.get('prompt_tokens',0)
print(f'prefill: prompt={pt} tokens in {wall:.1f}s -> {pt/wall:.1f} tok/s prefill')"
