#!/bin/zsh
# 프리필-2박스 서빙 스택 종료 — TERM만 (KILL 금지 규칙).
set -u
SD=~/qwen38/serving_prefill2box
EPS=m3ms@10.0.0.2
PIDFILE=$SD/prefill2box.pid
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [[ -f $PIDFILE ]]; then
    LPID=$(cat $PIDFILE)
    kill -TERM $LPID 2>/dev/null && say "TERM → gesicht server $LPID" || say "서버 이미 부재"
fi
pkill -TERM -f 'serving_prefill2box/serve_ane' 2>/dev/null && say "TERM → serve_ane" || true
ssh -o ConnectTimeout=5 $EPS "pkill -TERM -f 'prefill_2box.server|runner_ane'" 2>/dev/null \
    && say "TERM → epsilon 러너" || true

t0=$(date +%s)
while (( $(date +%s) - t0 < 60 )); do
    L=$(pgrep -f 'serving_prefill2box/serve_ane|mlx_lm server' || true)
    R=$(ssh -o ConnectTimeout=5 $EPS "pgrep -f 'prefill_2box.server|runner_ane'" 2>/dev/null || true)
    [[ -z $L && -z $R ]] && { rm -f $PIDFILE; say "✓ P2BOX-DOWN"; exit 0; }
    sleep 2
done
say "⚠ TERM 불응 잔존 (gesicht: ${L:-없음} / epsilon: ${R:-없음}) — KILL 금지, 수동 확인 필요"
exit 1
