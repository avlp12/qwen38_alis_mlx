"""2박스 층-파이프 프리필 + ANE — 양 박스에 ANE 를 붙이고 정본 하네스로 잰다."""
import os, sys, io, logging, json, time
HOME=os.path.expanduser("~")
buf=io.StringIO(); h=logging.StreamHandler(buf); h.setLevel(logging.INFO)
lg=logging.getLogger("omlx.patches.qwen35_ane_prefill"); lg.addHandler(h); lg.setLevel(logging.INFO)
sys.path.insert(0, f"{HOME}/glm5.2/mlx-lm")
from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
_a = apply_qwen35_q4_mlp_patch(); _b = apply_qwen35_q4_prefill_linear_patch()
try: _c = apply_qwen35_q4_lm_prefill_linear_patch()
except Exception: _c = None
print(f"[q4patch] mlp={_a} prefill_linear={_b} lm={_c}", flush=True)

import mlx.core as mx
from mlx_lm.prefill_2box.runner import load_model_only, set_wired_limit
from mlx_lm.prefill_2box.orchestrator import TwoBoxPrefill
from mlx_lm.prefill_2box.bench_2box import bench_one_1box, bench_one_2box, rng_tokens, cooldown
P=lambda *a: print(*a, flush=True)
SEQ=int(os.environ.get("ANE_SEQ","1024")); CHUNK=int(os.environ.get("CHUNK","1024"))
ANE=os.environ.get("ANE","1")=="1"
model=load_model_only(f"{HOME}/qwen38/q4v-fp16"); set_wired_limit()
if ANE:
    n=enable_qwen35_ane_prefill(model, sequence_length=SEQ, fraction=float(os.environ.get("F","0.30")),
        gdn=True, gdn_fraction=float(os.environ.get("G","0.375")),
        cpu_fraction=float(os.environ.get("C","0.14")), cpu_gdn_fraction=float(os.environ.get("CG","0.13")),
        cpu_down_fraction=float(os.environ.get("CD","0.10")), cpu_threads=8, dual_ane=True)
    log=buf.getvalue()
    P(f"[gesicht] 부착 {n} · 워밍업 {'있음' if 'Warmed' in log else '없음'} · seq={SEQ}")
    if "Warmed" not in log: sys.exit("워밍업 미확인")
cli=TwoBoxPrefill(model, "10.0.0.2", 39919, split=32)
P("연결:", cli.remote_meta.get("pid"), "· warmup")
cli.warmup(64); cli.warmup(64)
out={"ane":ANE,"seq":SEQ,"chunk":CHUNK,"runs":[]}
for N in (8192, 32768):
    for arm,fn in (("1box", lambda: bench_one_1box(model, toks, CHUNK)),
                   ("2box", lambda: bench_one_2box(model, cli, toks, CHUNK))):
        toks=rng_tokens(N)
        cooldown(30,"")
        r=fn(); r["N"]=N; r.setdefault("arm",arm); out["runs"].append(r)
        P(f"  N={N} {arm}@{CHUNK}: {r['tok_s']:.1f} tok/s")
json.dump(out, open(sys.argv[1],"w"), indent=1)
P("2BOX-ANE-DONE")
