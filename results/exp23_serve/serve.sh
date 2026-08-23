#!/bin/bash
export MLX_METAL_FAST_SYNCH=1
cd /Users/Shared/tp2
exec "$HOME/venv_omlx063/bin/python" /Users/Shared/tp2/serve_tp4_dspark.py \
  --model "$HOME/dsv4flash/mlx4bit" \
  --model-name deepseek-v4-flash-tp2 \
  --serve-port 8003 --control-host 10.0.0.1 --control-port 18003 \
  --require-world 2 --depth 1 --prefill-step 2048 \
  --max-context-tokens 32768 --max-output-tokens 4096 "$@"
