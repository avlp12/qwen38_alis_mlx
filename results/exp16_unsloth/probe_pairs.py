"""가법성 검정: KL(A+B) 대 KL(A)+KL(B).

교차항이 음수라면(= 오차가 상쇄되면) 결손 D = KL(A)+KL(B) − KL(A+B) > 0 이다.
RMSNorm 경쟁 가설이 맞다면 **같은 정규화 경로를 공유하는 쌍**(같은 종류·이웃 층)에서
결손이 더 커야 하고, 경로가 다른 쌍(예: 은닉층 대 출력 헤드)에서는 작아야 한다.
출력 헤드는 하류 정규화가 없으므로 다른 텐서와 경쟁할 여지가 없다.
"""
import json, os, re, sys, time
FORK = os.environ.get("FORK", os.path.expanduser("~/glm5.2/mlx-lm"))
sys.path.insert(0, FORK)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

SRC = os.path.expanduser("~/qwen38/src"); CORPUS = os.path.expanduser("~/qwen38/eval_corpus")
CTX, GS = 2048, 64
P = lambda *a: print(*a, flush=True)
model, tok = load(SRC, lazy=False)
paths = [p for p, m in model.named_modules() if hasattr(m, "to_quantized")
         and m.weight.shape[-1] % GS == 0]
def group_of(path):
    if "visual" in path: return None
    if "lm_head" in path: return "lm_head"
    if "embed_tokens" in path: return "embed_tokens"
    m = re.search(r"layers\.(\d+)\.(.+)$", path)
    if not m: return None
    if "mtp" in path: return None
    return f"{m.group(2)}@{int(m.group(1))//8}"
groups = {}
for p in paths:
    g = group_of(p)
    if g: groups.setdefault(g, []).append(p)
wins = [tok.encode(open(f"{CORPUS}/{t}.txt").read())[:CTX] for t in ("en","ko","code")]
def logp(w):
    o = model(mx.array([w]));  o = o[0] if isinstance(o, tuple) else o
    return nn.log_softmax(o[0].astype(mx.float32), axis=-1)
ref = []
for w in wins:
    ref.append(mx.array(logp(w))); mx.eval(ref[-1]); mx.clear_cache()
def kl_now():
    s = 0.0
    for w, rl in zip(wins, ref):
        tl = logp(w); p = mx.exp(rl)
        s += float(mx.sum(p*(rl-tl))/rl.shape[0]); mx.clear_cache()
    return s/len(wins)
orig = model.leaf_modules()
def measure(sel):
    nn.quantize(model, group_size=GS, bits=4, class_predicate=lambda p,m,_s=sel: p in _s)
    v = kl_now(); model.update_modules(orig); return v

PAIRS = [
    ("이웃 같은종류", "mlp.gate_proj@4", "mlp.gate_proj@5"),
    ("먼 같은종류",   "mlp.gate_proj@0", "mlp.gate_proj@7"),
    ("같은층 다른종류","mlp.gate_proj@4", "mlp.up_proj@4"),
    ("은닉 대 헤드",  "mlp.gate_proj@4", "lm_head"),
    ("은닉 대 임베딩","mlp.gate_proj@4", "embed_tokens"),
    ("헤드 대 임베딩","lm_head",         "embed_tokens"),
]
out = {}
for lab, a, b in PAIRS:
    if a not in groups or b not in groups: P(f"  {lab}: 그룹 없음 {a}/{b}"); continue
    A, B = set(groups[a]), set(groups[b])
    ka, kb, kab = measure(A), measure(B), measure(A | B)
    d = ka + kb - kab
    out[lab] = {"a": a, "b": b, "kl_a": ka, "kl_b": kb, "kl_ab": kab,
                "deficit": d, "deficit_frac": d/(ka+kb)}
    P(f"  {lab:<14} {a:<20}+{b:<16} A {ka:.5f} B {kb:.5f} A+B {kab:.5f} "
      f"→ 결손 {d:+.5f} ({d/(ka+kb):+.1%})")
    json.dump(out, open(sys.argv[1], "w"), indent=0)
P("PAIRS-DONE")
