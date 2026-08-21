"""길이 분기 대조군 — ANE 를 아예 붙이지 않은 채 같은 길이·같은 두 스케줄을 잰다.
ANE 를 seq=2048 로 붙여 둔 상태에서 좁은 스케줄(청크 1024)을 돌릴 때 남는
비용이 있는지를 이 대조와의 차이로만 말할 수 있다."""
import os, sys, json
HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/glm5.2/mlx-lm")
from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
_a = apply_qwen35_q4_mlp_patch(); _b = apply_qwen35_q4_prefill_linear_patch()
try: _c = apply_qwen35_q4_lm_prefill_linear_patch()
except Exception: _c = None
P = lambda *a: print(*a, flush=True)
P(f"[q4patch] mlp={_a} prefill_linear={_b} lm={_c}")

from mlx_lm.prefill_2box.runner import load_model_only, set_wired_limit
from mlx_lm.prefill_2box.orchestrator import TwoBoxPrefill
from mlx_lm.prefill_2box.bench_2box import bench_one_2box, rng_tokens, cooldown

model = load_model_only(f"{HOME}/qwen38/q4v-fp16"); set_wired_limit()
P("[gesicht] ANE 미부착 (대조)")
cli = TwoBoxPrefill(model, "10.0.0.2", 39919, split=32)
P("연결:", cli.remote_meta.get("pid"))
cli.warmup(64); cli.warmup(64)

LENS = [int(x) for x in os.environ.get("LENS", "8192,12288,16384,24576,32768").split(",")]
out = {"attached_seq": None, "runs": []}
dest = sys.argv[1]
for N in LENS:
    toks = rng_tokens(N); row = {}
    for chunk in (1024, 2048):
        cooldown(30, "")
        r = bench_one_2box(model, cli, toks, chunk)
        r.update(N=N, chunk=chunk, arm="2box")
        out["runs"].append(r); row[chunk] = r["tok_s"]
        P(f"  CTL N={N} 2box@{chunk}: {r['tok_s']:.1f} tok/s")
    P(f"  → CTL N={N}: 2048/1024 = {row[2048]/row[1024]:.4f}")
    json.dump(out, open(dest, "w"), indent=1)
P("XOVER-CTL-DONE")
