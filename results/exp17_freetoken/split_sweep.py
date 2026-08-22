"""2박스 층-분할점 스윕 — FreeToken 의 대역폭-비례 분배 규칙이 TB5 쌍에서 성립하는가.

FreeToken 은 나눌 수 있는 일을 **실측 대역폭 비**로 두 경로에 배분한다(q* ≈ m·B_P/B_H).
층-파이프 프리필에서 두 경로는 두 박스이고, 해당 "대역폭"은 각 박스의 프리필 처리량이다.
규칙의 예측:  split* = 64 · T_eps / (T_eps + T_ges)      (epsilon 이 [0,split) 을 맡음)
동일한 두 박스면 32 로 자명하다. **비대칭을 만들어야 규칙이 검정된다** — ANE 를 한쪽에만
붙이면 약 17% 차이가 난다. 링크·버블 항이 지배하면 최적점은 움직이지 않는다.
"""
import json, os, sys, time
# expanduser 를 환경변수 값에도 반드시 적용한다 — 빼먹으면 "~/..." 가 문자 그대로
# sys.path 에 들어가고, 엉뚱한 mlx_lm 이 잡히며 조용히 다른 코드가 돈다([I189] 와 같은 부류).
FORK = os.path.expanduser(os.environ.get("FORK", "~/glm5.2/mlx-lm"))
if not os.path.isdir(os.path.join(FORK, "mlx_lm", "prefill_2box")):
    raise SystemExit(f"prefill_2box 가 없는 트리다: {FORK}")
sys.path.insert(0, FORK)
import mlx.core as mx
from mlx_lm.prefill_2box.runner import load_model_only, set_wired_limit
from mlx_lm.prefill_2box.orchestrator import TwoBoxPrefill
from mlx_lm.prefill_2box.bench_2box import bench_one_1box, bench_one_2box, rng_tokens, cooldown

P = lambda *a: print(*a, flush=True)
SPLIT = int(os.environ["SPLIT"])
N = int(os.environ.get("N", 32768))
CHUNK = int(os.environ.get("CHUNK", 2048))
ANE_LOCAL = os.environ.get("ANE_LOCAL", "1") == "1"
OUT = sys.argv[1]

if ANE_LOCAL:
    import io, logging
    buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.INFO)
    lg = logging.getLogger("omlx.patches.qwen35_ane_prefill"); lg.addHandler(h); lg.setLevel(logging.INFO)
    from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
        apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
    apply_qwen35_q4_mlp_patch(); apply_qwen35_q4_prefill_linear_patch()
    try: apply_qwen35_q4_lm_prefill_linear_patch()
    except Exception: pass
    from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill

model = load_model_only(os.path.expanduser("~/qwen38/q4v-fp16")); set_wired_limit()
if ANE_LOCAL:
    n = enable_qwen35_ane_prefill(model, sequence_length=CHUNK, fraction=0.30, gdn=True,
        gdn_fraction=0.375, cpu_fraction=0.14, cpu_gdn_fraction=0.13,
        cpu_down_fraction=0.10, cpu_threads=8, dual_ane=True)
    if "Warmed" not in buf.getvalue(): sys.exit("ANE 워밍업 미확인")
    P(f"[local] ANE 부착 {n}")
cli = TwoBoxPrefill(model, "10.0.0.2", 39919, split=SPLIT)
P(f"[연결] pid={cli.remote_meta.get('pid')} split={SPLIT}")
cli.warmup(64); cli.warmup(64)

toks = rng_tokens(N)
cooldown(30, "")
r2 = bench_one_2box(model, cli, toks, CHUNK)
cooldown(20, "")
r1 = bench_one_1box(model, toks, CHUNK)
row = {"split": SPLIT, "N": N, "chunk": CHUNK, "ane_local": ANE_LOCAL,
       "tok_s_2box": r2["tok_s"], "tok_s_1box_local": r1["tok_s"],
       "t_prefill_2box": r2.get("t_prefill"), "t_chunks": r2.get("t_chunks")}
P(f"  split={SPLIT} 2box {r2['tok_s']:.1f} tok/s · 로컬 1box {r1['tok_s']:.1f}")
json.dump(row, open(OUT, "w"), indent=0)
P("SPLIT-POINT-DONE")
