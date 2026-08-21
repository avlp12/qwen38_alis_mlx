#!/bin/zsh
# 프리필-2박스 서빙 스택 + ANE 하이브리드 — 온디맨드 런치.
#
# 한 번의 실행으로: 프리플라이트(epsilon 도달·포크 동기·모델·포트) →
# epsilon 러너 기동(ANE seq=2048, 워밍업 확인) → gesicht mlx_lm.server 기동
# (--prefill-2box + 길이 분기 플래그) → /health 대기 → 스모크 검증.
# 종료는 쌍 스크립트 stop_prefill2box.sh (TERM만, KILL 금지).
#
# 길이 분기([PA57]): 미-캐시 접미 길이 >= LONG_TOKENS(11264, 실측 교차점) 이면
# 청크 CHUNK_LONG(2048, ANE 발화) 아니면 CHUNK(1024, ANE 비켜섬). 두 스케줄은
# 순서가 없다 — 넓은 청크는 청크당 고정비를 상각하지만 파이프 버블을 키우므로,
# 프롬프트가 청크를 충분히 담을 때만 이긴다. ANE 를 끄면 8K~32K 어디서도 넓은
# 청크가 이기지 못하므로, 이 분기는 ANE 가 켜져 있을 때만 의미가 있다.
#
# 사용:  ~/qwen38/serving_prefill2box/launch_prefill2box.sh
#        ANE=0 ...           # ANE 없이(대조군·문제 격리용)
#        LONG_TOKENS=16384 ...  # 임계값 오버라이드(기본 11264 = 실측 교차점)
#
# 무접촉: epsilon omlx(:8002, GuruNote) · TP2 스택(:8003) — 포트·프로세스 분리.
set -u

SD=~/qwen38/serving_prefill2box
FORK=~/glm5.2/mlx-lm
EPS_HOST=10.0.0.2
EPS=m3ms@$EPS_HOST
EPS_FORK='~/mlx-lm-fork'
PORT=${PORT:-8004}
RPORT=${RPORT:-39919}
MODEL=${MODEL:-~/qwen38/q4v-fp16}
EPS_MODEL=${EPS_MODEL:-'~/qwen38/q4v-fp16'}
SPLIT=${SPLIT:-32}
CHUNK=${CHUNK:-1024}
CHUNK_LONG=${CHUNK_LONG:-2048}
LONG_TOKENS=${LONG_TOKENS:-11264}   # 실측 교차점 [I169]
MIN_TOKENS=${MIN_TOKENS:-4096}
ANE=${ANE:-1}
VENV=${VENV:-~/venv_omlx063/bin/python}
EPS_VENV=${EPS_VENV:-'~/venv_omlx063/bin/python'}
OMLX_HOME=${OMLX_HOME:-~/qwen38/exp15_ane/omlx_home}
HEALTH_DEADLINE=${HEALTH_DEADLINE:-600}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p $SD/logs
LOG=$SD/logs/server_${TS}.log
PIDFILE=$SD/prefill2box.pid

say()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
fail() { say "✗ P2BOX-FAIL: $*"; exit 1; }

# ── 0) 중복·포트 ─────────────────────────────────────────────────────
if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    fail "이미 가동 중 (pid $(cat $PIDFILE)) — stop_prefill2box.sh 먼저"
fi
lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && fail "포트 $PORT 선점됨"

# ── 1) epsilon 도달·모델·잔재 ────────────────────────────────────────
ssh -o ConnectTimeout=5 $EPS true 2>/dev/null || fail "epsilon ssh 불가"
ssh $EPS "ls $EPS_MODEL/config.json" >/dev/null 2>&1 || fail "epsilon 모델 부재: $EPS_MODEL"
[[ -f $MODEL/config.json ]] || fail "gesicht 모델 부재: $MODEL"

# ── 2) 포크 동기 (양 박스 mlx_lm 트리 해시 일치 — 비트-동일 전제) ────
H1=$(cd $FORK && find mlx_lm -name '*.py' | sort | xargs shasum | shasum | cut -d' ' -f1)
H2=$(ssh $EPS "cd $EPS_FORK && find mlx_lm -name '*.py' | sort | xargs shasum | shasum" | cut -d' ' -f1)
if [[ $H1 != $H2 ]]; then
    say "포크 불일치 → rsync 동기"
    rsync -a --delete --exclude __pycache__ $FORK/mlx_lm/ ${EPS}:mlx-lm-fork/mlx_lm/ || fail "rsync 실패"
    H2=$(ssh $EPS "cd $EPS_FORK && find mlx_lm -name '*.py' | sort | xargs shasum | shasum" | cut -d' ' -f1)
    [[ $H1 == $H2 ]] || fail "동기 후에도 해시 불일치"
fi
say "✓ 포크 동기 ($H1)"

# ── 3) epsilon 러너 기동 ─────────────────────────────────────────────
ssh $EPS "pkill -TERM -f 'prefill_2box.server|runner_ane' 2>/dev/null; sleep 5" || true
if [[ $ANE == 1 ]]; then
    ssh $EPS "mkdir -p qwen38/exp15_ane"
    scp -q ~/qwen38/exp15_ane/runner_ane.py ${EPS}:qwen38/exp15_ane/ || fail "러너 scp 실패"
    ssh $EPS "cd ~ && ANE_SEQ=$CHUNK_LONG F=0.30 G=0.375 C=0.14 CG=0.13 CD=0.10 \
        FORK=$EPS_FORK OMLX_BASE_PATH=~/qwen38/exp15_ane/omlx_home \
        nohup $EPS_VENV ~/qwen38/exp15_ane/runner_ane.py \
          --model $EPS_MODEL --lo 0 --hi $SPLIT --port $RPORT \
          > ~/qwen38/runner_ane.log 2>&1 & echo started" >/dev/null \
        || fail "러너 발사 실패"
    RLOG='~/qwen38/runner_ane.log'
else
    ssh $EPS "cd ~ && PYTHONPATH=$EPS_FORK nohup $EPS_VENV -m mlx_lm.prefill_2box.server \
        --model $EPS_MODEL --lo 0 --hi $SPLIT --port $RPORT \
        > ~/qwen38/runner_plain.log 2>&1 & echo started" >/dev/null || fail "러너 발사 실패"
    RLOG='~/qwen38/runner_plain.log'
fi
t0=$(date +%s); up=0
while (( $(date +%s) - t0 < 300 )); do
    ssh $EPS "lsof -nP -iTCP:$RPORT -sTCP:LISTEN >/dev/null 2>&1" && { up=1; break; }
    sleep 3
done
(( up )) || { ssh $EPS "tail -12 $RLOG"; fail "러너 기동 실패 (${RPORT})"; }
if [[ $ANE == 1 ]]; then
    W=$(ssh $EPS "grep -m1 'runner-ane' $RLOG" || true)
    say "  러너: ${W:-(로그 없음)}"
    [[ $W == *"워밍업 있음"* ]] || fail "러너 ANE 워밍업 미확인 — 이 상태의 출력은 신뢰 불가"
fi
say "✓ 러너 up (:$RPORT, ANE=$ANE)"

# ── 4) gesicht 서버 발사 ─────────────────────────────────────────────
BR=(--prefill-2box-chunk $CHUNK --prefill-2box-chunk-long $CHUNK_LONG
    --prefill-2box-long-tokens $LONG_TOKENS --prefill-2box-min-tokens $MIN_TOKENS)
cd ~
if [[ $ANE == 1 ]]; then
    OMLX_BASE_PATH=$OMLX_HOME ANE_SEQ=$CHUNK_LONG FORK=$FORK \
    nohup $VENV -u $SD/serve_ane.py --model $MODEL --port $PORT \
        --prefill-2box $EPS_HOST:$RPORT --prefill-2box-split $SPLIT "${BR[@]}" \
        > $LOG 2>&1 &
else
    PYTHONPATH=$FORK nohup $VENV -u -m mlx_lm server --model $MODEL --port $PORT \
        --prefill-2box $EPS_HOST:$RPORT --prefill-2box-split $SPLIT "${BR[@]}" \
        > $LOG 2>&1 &
fi
LPID=$!
echo $LPID > $PIDFILE
say "발사: server pid=$LPID log=$LOG (분기 $CHUNK → $CHUNK_LONG @ >=$LONG_TOKENS)"

# ── 5) /health 대기 (하드 시한 — 침묵 금지) ──────────────────────────
t0=$(date +%s)
while true; do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && \
        { say "✓ /health up ($(( $(date +%s) - t0 ))s)"; break; }
    kill -0 $LPID 2>/dev/null || { tail -25 $LOG; fail "서버 사망 — 로그 $LOG"; }
    (( $(date +%s) - t0 > HEALTH_DEADLINE )) && { tail -25 $LOG; \
        say "시한 초과 — TERM 정리"; kill -TERM $LPID 2>/dev/null; \
        fail "health 대기 ${HEALTH_DEADLINE}s 초과"; }
    sleep 3
done
[[ $ANE == 1 ]] && { grep -m1 'serve-ane' $LOG || fail "서버 ANE 부착 로그 없음"; }

# ── 6) 스모크 ────────────────────────────────────────────────────────
SMOKE=$(curl -s -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":32,"temperature":0.0}')
printf '%s' "$SMOKE" | python3 -c '
import json,sys
d=json.load(sys.stdin); c=d["choices"][0]["message"]
txt=(c.get("content") or "")+(c.get("reasoning") or "")+(c.get("reasoning_content") or "")
assert txt.strip(), d
print("[smoke]", txt.strip()[:80].replace("\n"," "))
' || { tail -25 $LOG; fail "스모크 실패: $SMOKE"; }

say "✓ P2BOX-UP pid=$LPID port=$PORT log=$LOG"
say "  종료: $SD/stop_prefill2box.sh"
