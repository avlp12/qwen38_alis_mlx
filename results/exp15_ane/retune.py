"""올바른 경로 재튜닝 — 후보마다 속도와 품질을 함께 재고, 품질 게이트를 강제한다.

초기화 검증이 코드에 박혀 있다: 로드 로그에 'Warmed N ANE procedures' 가 없으면 즉시 중단.
"""
import os, sys, time, json, logging, io
HOME = os.path.expanduser("~")
_buf = io.StringIO()
h = logging.StreamHandler(_buf); h.setLevel(logging.INFO)
logging.getLogger("omlx.patches.qwen35_ane_prefill").addHandler(h)
logging.getLogger("omlx.patches.qwen35_ane_prefill").setLevel(logging.INFO)
import mlx.core as mx
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from omlx.model_settings import ModelSettings
MODEL = os.environ.get("MODEL", f"{HOME}/qwen38/oq4e")
maybe_apply_pre_load_patches(MODEL, model_settings=ModelSettings(), for_vlm=True)
from mlx_vlm.utils import load as vlm_load
from mlx_lm.models.cache import make_prompt_cache
from omlx.custom_kernels.qwen35_prefill import fast
from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
P = lambda *a: print(*a, flush=True)
F  = float(os.environ.get("F", 0.19));  G  = float(os.environ.get("G", 0.45))
C  = float(os.environ.get("C", 0.14));  CG = float(os.environ.get("CG", 0.13))
CD = float(os.environ.get("CD", 0.0))
NQ, NS = 2048, 4096
model, processor = vlm_load(MODEL)
tok = getattr(processor, "tokenizer", processor); lm = model.language_model
_t = tok.encode("The history of computing hardware spans centuries and each transition "
                "changed which problems people considered tractable. ")
def ids_of(n):
    b=[]
    while len(b)<n: b.extend(_t)
    return mx.array(b[:n])
def prefill(n, want_logits=False):
    ids = ids_of(n); c = make_prompt_cache(lm); o=[]
    t0=time.perf_counter()
    for i in range(0,n,2048):
        lg = lm(ids[i:i+2048][None], cache=c)
        lg = lg.logits if hasattr(lg,"logits") else lg
        if want_logits: o.append(lg[0].astype(mx.float32))
    if want_logits:
        r = mx.concatenate(o,0); mx.eval(r)
    else:
        mx.eval(lg); r = None
    return n/(time.perf_counter()-t0), r
prefill(NS); g_tp = max(prefill(NS)[0] for _ in range(2))
_, ref = prefill(NQ, True)
n = enable_qwen35_ane_prefill(model, sequence_length=2048, fraction=F, gdn=True,
        gdn_fraction=G, cpu_fraction=C, cpu_gdn_fraction=CG, cpu_down_fraction=CD,
        cpu_threads=8, dual_ane=True)
log = _buf.getvalue()
if "Warmed" not in log or "ANE procedures" not in log:
    P(json.dumps({"error":"초기화 미확인 — 워밍업 로그 없음","log":log[-300:]})); sys.exit(2)
fast.qwen35_ane_profile_set_enabled(True); fast.qwen35_ane_profile_reset()
prefill(NS); h_tp = max(prefill(NS)[0] for _ in range(2))
_, hyb = prefill(NQ, True)
s = fast.qwen35_ane_profile_snapshot()
lp = ref - mx.logsumexp(ref,-1,keepdims=True); lq = hyb - mx.logsumexp(hyb,-1,keepdims=True)
klp = mx.sum(mx.exp(lp)*(lp-lq),-1); mx.eval(klp)
top1 = float(mx.mean((mx.argmax(ref,-1)==mx.argmax(hyb,-1)).astype(mx.float32)))*100
out = {"F":F,"G":G,"C":C,"CG":CG,"CD":CD,"attached":n,
       "mlp_ops":s["mlp"]["operations"],"gdn_ops":s["gdn"]["operations"],
       "gpu_tps":round(g_tp,1),"hyb_tps":round(h_tp,1),
       "gain_pct":round((h_tp/g_tp-1)*100,2),
       "kl_mean":round(float(mx.mean(klp)),8),"top1_pct":round(top1,3)}
P(json.dumps(out))
with open(sys.argv[1],"a") as f: f.write(json.dumps(out)+"\n")
