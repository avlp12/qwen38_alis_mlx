#!/bin/zsh
set -u
D=~/qwen38/exp15_ane; L=$D/twobox_ane.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
start_runner(){
  local seq=$1 ane=$2
  ssh -o BatchMode=yes 10.0.0.2 "pkill -TERM -f 'prefill_2box.server|runner_ane' 2>/dev/null; sleep 4;
    cd ~ && ANE_SEQ=$seq F=0.30 G=0.375 C=0.14 CG=0.13 CD=0.10 FORK=~/mlx-lm-fork \
      OMLX_BASE_PATH=~/qwen38/exp15_ane/omlx_home \
      nohup ~/venv_omlx063/bin/python ~/qwen38/exp15_ane/runner_ane.py \
        --model ~/qwen38/q4v-fp16 --lo 0 --hi 32 > ~/qwen38/runner_ane.log 2>&1 & echo started"
  for i in $(seq 1 90); do
    ssh -o BatchMode=yes 10.0.0.2 'lsof -nP -iTCP:39919 -sTCP:LISTEN >/dev/null 2>&1' && return 0
    sleep 3
  done
  return 1
}
for seq in 1024 2048; do
  say "── ANE seq=$seq / chunk=$seq ──"
  start_runner $seq 1 || { say "✗ 러너 기동 실패"; ssh 10.0.0.2 'tail -5 ~/qwen38/runner_ane.log'; continue; }
  ssh -o BatchMode=yes 10.0.0.2 'grep -m1 "runner-ane" ~/qwen38/runner_ane.log' | tee -a $L
  cd ~ && env OMLX_BASE_PATH=$D/omlx_home ANE=1 ANE_SEQ=$seq CHUNK=$seq \
    ~/venv_omlx063/bin/python -u $D/bench2box_ane.py $D/twobox_ane_$seq.json >>$L 2>&1 || say "  ✗ 벤치 실패"
done
say "── 대조: ANE 끔, chunk=1024 ──"
ssh -o BatchMode=yes 10.0.0.2 "pkill -TERM -f 'prefill_2box.server|runner_ane' 2>/dev/null; sleep 4;
  cd ~ && PYTHONPATH=~/mlx-lm-fork nohup ~/venv_omlx063/bin/python -m mlx_lm.prefill_2box.server \
    --model ~/qwen38/q4v-fp16 --lo 0 --hi 32 > ~/qwen38/runner_plain.log 2>&1 & echo ok"
for i in $(seq 1 90); do ssh -o BatchMode=yes 10.0.0.2 'lsof -nP -iTCP:39919 -sTCP:LISTEN >/dev/null 2>&1' && break; sleep 3; done
cd ~ && env OMLX_BASE_PATH=$D/omlx_home ANE=0 CHUNK=1024 \
  ~/venv_omlx063/bin/python -u $D/bench2box_ane.py $D/twobox_plain.json >>$L 2>&1 || say "  ✗ 대조 실패"
say "2BOX-ANE-ALL-DONE"
