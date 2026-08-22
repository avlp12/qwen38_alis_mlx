#!/bin/zsh
# 분할점 스윕 — 대칭(양쪽 ANE)과 비대칭(로컬만 ANE) 두 팔.
# 비대칭 팔에서 최적 split 이 32 → 약 29-30 으로 움직이면 대역폭-비례 규칙이 성립한다.
set -u
D=~/qwen38/exp17_freetoken; L=$D/sweep.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }

# 유휴 검증 — 이 측정은 **양 박스를 독점**해야 한다. epsilon 에서 다른 GPU 작업이 돌면
# 원격 단계가 굶어 파이프라인이 겹치지 않고, 2박스가 1박스보다 느리게 나온다(실제로 겪음).
busy_eps=$(ssh -o BatchMode=yes 10.0.0.2 'ps aux | grep -iE python | grep -v grep | grep -vE "omlx-server|prefill_2box|runner_ane" | wc -l' | tr -d ' ')
busy_loc=$(pgrep -f "probe_damage|kl_paired|kl_eval|build_graded|awq" | wc -l | tr -d ' ')
if [[ ${busy_eps:-0} -gt 0 || ${busy_loc:-0} -gt 0 ]]; then
  say "✗ 비유휴 — epsilon $busy_eps · 로컬 $busy_loc. 측정 중단"; exit 1
fi
say "✓ 양 박스 유휴 확인"

start_runner(){   # $1=hi(=split)  $2=ane(1/0)
  local hi=$1 ane=$2
  ssh -o BatchMode=yes 10.0.0.2 "pkill -TERM -f 'prefill_2box.server|runner_ane' 2>/dev/null; sleep 5" || true
  if [[ $ane == 1 ]]; then
    ssh -o BatchMode=yes 10.0.0.2 "cd ~ && ANE_SEQ=2048 F=0.30 G=0.375 C=0.14 CG=0.13 CD=0.10 \
      FORK=~/mlx-lm-fork OMLX_BASE_PATH=~/qwen38/exp15_ane/omlx_home \
      nohup ~/venv_omlx063/bin/python ~/qwen38/exp15_ane/runner_ane.py \
        --model ~/qwen38/q4v-fp16 --lo 0 --hi $hi > ~/qwen38/runner_ane.log 2>&1 & echo ok" >/dev/null
  else
    ssh -o BatchMode=yes 10.0.0.2 "cd ~ && PYTHONPATH=~/mlx-lm-fork \
      nohup ~/venv_omlx063/bin/python -m mlx_lm.prefill_2box.server \
        --model ~/qwen38/q4v-fp16 --lo 0 --hi $hi > ~/qwen38/runner_plain.log 2>&1 & echo ok" >/dev/null
  fi
  for i in $(seq 1 90); do
    ssh -o BatchMode=yes 10.0.0.2 'lsof -nP -iTCP:39919 -sTCP:LISTEN >/dev/null 2>&1' && return 0
    sleep 3
  done
  return 1
}

for arm in sym asym; do
  eps_ane=1; [[ $arm == asym ]] && eps_ane=0
  say "── 팔 $arm (epsilon ANE=$eps_ane, 로컬 ANE=1) ──"
  for sp in 24 28 30 32 36; do
    start_runner $sp $eps_ane || { say "✗ split=$sp 러너 실패"; continue; }
    [[ $eps_ane == 1 ]] && ssh 10.0.0.2 'grep -m1 "runner-ane" ~/qwen38/runner_ane.log' | tee -a $L
    cd ~ && env OMLX_BASE_PATH=~/qwen38/exp15_ane/omlx_home FORK=$HOME/glm5.2/mlx-lm \
      SPLIT=$sp N=32768 CHUNK=2048 ANE_LOCAL=1 \
      ~/venv_omlx063/bin/python -u $D/split_sweep.py $D/pt_${arm}_${sp}.json >>$L 2>&1 \
      || say "✗ split=$sp 측정 실패"
  done
done
say "SWEEP-ALL-DONE"
