#!/bin/bash
# uran FreeToken 돌파구 3종: A) bs4 배치(+실측 적중률) B) memory-ratio 0.95 C) top-k 6→4
D=$HOME/qwen38/exp17_freetoken
ts(){ date "+[%H:%M:%S]"; }
stop_ft(){
  ssh uran 'wsl -d Ubuntu -- bash -lc "pgrep -f \"ft serve\" | head -1"' 2>/dev/null | tr -d "\r" \
    | while read p; do [ -n "$p" ] && ssh uran "wsl -d Ubuntu -- bash -lc 'kill -TERM $p'" 2>/dev/null; done
  sleep 15
}
wait_ready(){
  for i in $(seq 1 40); do sleep 25
    c=$(ssh -o BatchMode=yes -o ConnectTimeout=10 uran 'wsl -d Ubuntu -- bash -lc "curl -s -m 8 -o /dev/null -w %{http_code} -X POST http://127.0.0.1:1919/v1/chat/completions -H Content-Type:application/json -d @/mnt/c/Users/bucoc/ping.json"' 2>/dev/null | tr -d '\0\r' | tail -1)
    [ "$c" = "200" ] && return 0
    grep -qE "AssertionError|Backend worker is gone|Traceback" "$1" 2>/dev/null && return 1
  done; return 1
}
restore_cfg(){
  ssh uran 'wsl -d Ubuntu -- bash -lc "cd /root/models/DeepSeek-V4-Flash && [ -f config.json.bak ] && mv config.json.bak config.json && echo config-restored || echo config-clean"' 2>/dev/null | tr -d "\r"
}
trap 'restore_cfg; stop_ft' EXIT

echo "$(ts) ── A: collect-stats · bs1 대조 + bs4 배치 ──"
stop_ft
ssh uran "wsl -d Ubuntu -- bash /mnt/c/Users/bucoc/ft_stats_on.sh" > $D/serve_bs.out 2>&1 &
SP=$!
if wait_ready $D/serve_bs.out; then
  ssh uran 'wsl -d Ubuntu -- bash -lc "/root/ftenv/bin/python /mnt/c/Users/bucoc/ft_batch.py"' 2>&1 | tr -d "\r" | tee $D/bench_batch.log
  ssh uran 'wsl -d Ubuntu -- bash -lc "python3 /mnt/c/Users/bucoc/ftst.py"' 2>&1 | tr -d "\r" > $D/stats_collect.txt
  grep -iE "hit|miss|fetch|evict" $D/stats_collect.txt | head -15
  ssh uran 'wsl -d Ubuntu -- bash -lc "curl -s -m 180 -X POST http://127.0.0.1:1919/v1/chat/completions -H Content-Type:application/json -d @/mnt/c/Users/bucoc/sanity.json"' 2>&1 | tr -d "\r" > $D/sanity_k6.json
  echo "sanity_k6 저장 $(wc -c < $D/sanity_k6.json)바이트"
else echo "✗ A 기동 실패"; fi
kill -TERM $SP 2>/dev/null; wait $SP 2>/dev/null

echo "$(ts) ── B: memory-ratio 0.95 (상주율 확장) ──"
stop_ft
ssh uran "wsl -d Ubuntu -- bash /mnt/c/Users/bucoc/ft_ratio95.sh" > $D/serve_r95.out 2>&1 &
SP=$!
if wait_ready $D/serve_r95.out; then
  ssh uran 'wsl -d Ubuntu -- bash -lc "python3 /mnt/c/Users/bucoc/ftst.py"' 2>&1 | tr -d "\r" | grep -E "moe_cache_size|cache_budget"
  ssh uran 'wsl -d Ubuntu -- bash -lc "/root/ftenv/bin/python /mnt/c/Users/bucoc/ft_client.py 2>&1"' 2>&1 | tr -d "\r" | tee $D/bench_r95.log | grep -E "\[r|최고"
else echo "✗ B 기동 실패(0.95 OOM 가능)"; fi
kill -TERM $SP 2>/dev/null; wait $SP 2>/dev/null

echo "$(ts) ── C: top-k 6→4 (토큰당 바이트 −33%) ──"
stop_ft
ssh uran 'wsl -d Ubuntu -- bash -lc "python3 /mnt/c/Users/bucoc/topk.py 4"' 2>&1 | tr -d "\r"
ssh uran "wsl -d Ubuntu -- bash /mnt/c/Users/bucoc/ft_stats_on.sh" > $D/serve_k4.out 2>&1 &
SP=$!
if wait_ready $D/serve_k4.out; then
  ssh uran 'wsl -d Ubuntu -- bash -lc "/root/ftenv/bin/python /mnt/c/Users/bucoc/ft_client.py 2>&1"' 2>&1 | tr -d "\r" | tee $D/bench_k4.log | grep -E "\[r|최고"
  ssh uran 'wsl -d Ubuntu -- bash -lc "curl -s -m 180 -X POST http://127.0.0.1:1919/v1/chat/completions -H Content-Type:application/json -d @/mnt/c/Users/bucoc/sanity.json"' 2>&1 | tr -d "\r" > $D/sanity_k4.json
  echo "sanity_k4 저장 $(wc -c < $D/sanity_k4.json)바이트"
else echo "✗ C 기동 실패(topk=4 미지원 가능)"; fi
kill -TERM $SP 2>/dev/null; wait $SP 2>/dev/null
restore_cfg
echo "$(ts) FT-BREAK-DONE"
