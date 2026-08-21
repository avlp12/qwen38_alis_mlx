"""길이 분기 임계값 측정 — ANE seq=2048 을 한 번만 부착한 뒤(서빙과 동일 조건)
청크 1024 / 2048 두 스케줄을 같은 프로세스에서 교대로 잰다.

핵심 질문 둘:
  1) seq=2048 로 부착한 채 청크 1024 를 돌리면 ANE 가 깨끗이 비켜서서
     ANE-끔 기준선(778.2 @8K · 780.4 @32K)을 회복하는가 — 회복해야 한 번의
     로드로 두 구간을 다 서빙할 수 있다.
  2) 두 스케줄이 교차하는 길이는 어디인가 — 그게 long_tokens 기본값이다.
"""
import os, sys, io, json, logging
HOME = os.path.expanduser("~")
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.INFO)
lg = logging.getLogger("omlx.patches.qwen35_ane_prefill"); lg.addHandler(h); lg.setLevel(logging.INFO)
sys.path.insert(0, f"{HOME}/glm5.2/mlx-lm")
from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
_a = apply_qwen35_q4_mlp_patch(); _b = apply_qwen35_q4_prefill_linear_patch()
try: _c = apply_qwen35_q4_lm_prefill_linear_patch()
except Exception: _c = None
P = lambda *a: print(*a, flush=True)
P(f"[q4patch] mlp={_a} prefill_linear={_b} lm={_c}")

from mlx_lm.prefill_2box.runner import load_model_only, set_wired_limit
from mlx_lm.prefill_2box.orchestrator import TwoBoxPrefill
from mlx_lm.prefill_2box.bench_2box import bench_one_1box, bench_one_2box, rng_tokens, cooldown

model = load_model_only(f"{HOME}/qwen38/q4v-fp16"); set_wired_limit()
n = enable_qwen35_ane_prefill(model, sequence_length=2048, fraction=0.30, gdn=True,
    gdn_fraction=0.375, cpu_fraction=0.14, cpu_gdn_fraction=0.13,
    cpu_down_fraction=0.10, cpu_threads=8, dual_ane=True)
log = buf.getvalue()
P(f"[gesicht] 부착 {n} · 워밍업 {'있음' if 'Warmed' in log else '없음'} · seq=2048")
if "Warmed" not in log: sys.exit("워밍업 미확인 — 측정 신뢰 불가")

cli = TwoBoxPrefill(model, "10.0.0.2", 39919, split=32)
P("연결:", cli.remote_meta.get("pid"))
cli.warmup(64); cli.warmup(64)

LENS = [int(x) for x in os.environ.get("LENS", "8192,12288,16384,24576,32768").split(",")]
out = {"attached_seq": 2048, "runs": []}
dest = sys.argv[1]
for N in LENS:
    toks = rng_tokens(N)
    row = {}
    # 같은 토큰·같은 프로세스에서 두 스케줄을 교대 — 표류가 효과로 위장하지 못한다
    for chunk in (1024, 2048):
        cooldown(30, "")
        r = bench_one_2box(model, cli, toks, chunk)
        r.update(N=N, chunk=chunk, arm="2box")
        out["runs"].append(r); row[chunk] = r["tok_s"]
        P(f"  N={N} 2box@{chunk}: {r['tok_s']:.1f} tok/s")
    P(f"  → N={N}: 2048/1024 = {row[2048]/row[1024]:.4f}")
    json.dump(out, open(dest, "w"), indent=1)
P("XOVER-DONE")
