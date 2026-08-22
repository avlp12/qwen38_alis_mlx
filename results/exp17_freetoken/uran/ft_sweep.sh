#!/bin/bash
# uran FreeToken 백엔드 스윕 — 호스트-측 계산을 PCIe 경로에 더하면 이득인가
D=$HOME/qwen38/exp17_freetoken
ts() { date "+[%H:%M:%S]"; }
for n in auto2 hybrid cpul8; do
  echo "$(ts) ── $n ──"
  ssh uran 'wsl -d Ubuntu -- bash -lc "pgrep -f \"ft serve\" | head -1"' 2>/dev/null | tr -d '\r' \
    | while read p; do [ -n "$p" ] && ssh uran "wsl -d Ubuntu -- bash -lc 'kill -TERM $p'" 2>/dev/null; done
  sleep 15
  ssh uran "wsl -d Ubuntu -- bash /mnt/c/Users/bucoc/ft_$n.sh" > $D/serve_$n.out 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 40); do
    sleep 25
    c=$(ssh -o BatchMode=yes -o ConnectTimeout=10 uran 'wsl -d Ubuntu -- bash -lc "curl -s -m 8 -o /dev/null -w %{http_code} -X POST http://127.0.0.1:1919/v1/chat/completions -H Content-Type:application/json -d @/mnt/c/Users/bucoc/ping.json"' 2>/dev/null | tr -d '\0\r' | tail -1)
    [ "$c" = "200" ] && { ok=1; break; }
    grep -qE "AssertionError|Backend worker is gone|Traceback" $D/serve_$n.out 2>/dev/null && { echo "  ✗ 기동 실패"; break; }
  done
  if [ $ok -eq 1 ]; then
    echo "$(ts) 준비됨 · 벤치"
    grep -oE "Resolved config: [^\"]*" $D/serve_$n.out | head -1 | tr -d '\r'
    ssh uran 'wsl -d Ubuntu -- bash -lc "python3 /mnt/c/Users/bucoc/ftst.py"' 2>&1 | tr -d '\r' \
      | grep -E "moe_cache_size|num_experts" 
    ssh uran 'wsl -d Ubuntu -- bash -lc "/root/ftenv/bin/python /mnt/c/Users/bucoc/ft_client.py 2>&1"' 2>&1 \
      | tr -d '\r' | tee $D/bench_$n.log | grep -E "\[r|최고"
  fi
  kill -TERM $SPID 2>/dev/null
  wait $SPID 2>/dev/null
done
echo "$(ts) FT-SWEEP-DONE"
