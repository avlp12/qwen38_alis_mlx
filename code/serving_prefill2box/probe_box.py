"""이 박스의 프리필 처리량을 한 번 재서 tok/s 만 출력한다.

자동 분할점 계산의 입력이다: 층-파이프에서 두 경로는 두 박스이고, 균형점은
    split* = L · T_remote / (T_remote + T_local)
로 각 박스의 **실측** 처리량 비를 따른다. 동일한 쌍이면 L/2 로 자명하지만
쌍이 비대칭이면(예: M3 Ultra + M4 mini, 또는 한쪽만 ANE) 정점이 이동한다 —
실측으로 32→30 이동을 확인했고 32 고정 시 6.0% 손실이었다.
"""
import io, logging, os, sys
FORK = os.path.expanduser(os.environ.get("FORK", "~/glm5.2/mlx-lm"))
sys.path.insert(0, FORK)
N = int(os.environ.get("N", 8192))
CHUNK = int(os.environ.get("CHUNK", 2048))
ANE = os.environ.get("ANE", "1") == "1"
MODEL = os.path.expanduser(os.environ.get("MODEL", "~/qwen38/q4v-fp16"))

if ANE:
    buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.INFO)
    logging.getLogger("omlx.patches.qwen35_ane_prefill").addHandler(h)
    logging.getLogger("omlx.patches.qwen35_ane_prefill").setLevel(logging.INFO)
    from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
        apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
    apply_qwen35_q4_mlp_patch(); apply_qwen35_q4_prefill_linear_patch()
    try: apply_qwen35_q4_lm_prefill_linear_patch()
    except Exception: pass
    from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill

from mlx_lm.prefill_2box.runner import load_model_only, set_wired_limit
from mlx_lm.prefill_2box.bench_2box import bench_one_1box, rng_tokens

model = load_model_only(MODEL); set_wired_limit()
if ANE:
    enable_qwen35_ane_prefill(model, sequence_length=CHUNK, fraction=0.30, gdn=True,
        gdn_fraction=0.375, cpu_fraction=0.14, cpu_gdn_fraction=0.13,
        cpu_down_fraction=0.10, cpu_threads=8, dual_ane=True)
    if "Warmed" not in buf.getvalue():
        sys.exit("ANE 워밍업 미확인")
toks = rng_tokens(N)
bench_one_1box(model, toks, CHUNK)                 # 워밍업
r = bench_one_1box(model, toks, CHUNK)
print(f"TOKS_PER_SEC {r['tok_s']:.2f}", flush=True)
