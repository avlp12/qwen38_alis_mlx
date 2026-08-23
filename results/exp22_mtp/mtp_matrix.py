"""단일 적재로 {plain, d1, d6}×{에세이, 코드, 한국어} 매트릭스 + fast_qmm 토글."""
import os, sys, time
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth

QMM = os.environ.get("FAST_QMM", "0") == "1"
apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(1)

from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

model, tok = load(os.path.expanduser("~/dsv4flash/mlx4bit"), lazy=False)
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
if QMM:
    from mlx_lm import fast_qmm
    fast_qmm.enable()
print(f"[matrix] fast_qmm={QMM}", flush=True)

PROMPTS = {
  "에세이": "Explain how MVCC works in PostgreSQL, covering snapshots, vacuum, and write amplification, with examples.",
  "코드":   "Write a complete Python implementation of an LRU cache with O(1) get/put using a doubly linked list and dict. Include tests.",
  "한국어": "한국 전통 건축의 처마 곡선이 구조적으로 어떤 역할을 하는지, 목조 결구법과 함께 자세히 설명해줘.",
}
def run(tag, active, depth, n=384):
    set_mtp_active(active); set_mtp_depth(depth)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPTS[tag]}],
                                  add_generation_prompt=True)
    bg = BatchGenerator(model, max_tokens=n, sampler=make_sampler(0.0),
                        completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048)
    bg.insert([ids], max_tokens=[n])
    toks = []; t1 = None; stats = {}
    while True:
        rs = bg.next_generated()
        if not rs: break
        if t1 is None: t1 = time.monotonic()
        stop = False
        for r in rs:
            toks.append(int(r.token))
            if r.finish_reason: stop = True
        st = getattr(bg, "_omlx_mtp_state", None) or {}
        if stop: break
    dt = time.monotonic() - t1
    tps = (len(toks) - 1) / dt
    print(f"  [{tag}] mtp={'d'+str(depth) if active else 'off':>3} · {tps:6.2f} tok/s · {len(toks)}토큰", flush=True)
    return tps

for tag in PROMPTS:
    for active, depth in ((False, 1), (True, 1), (True, 6)):
        run(tag, active, depth)
print("MATRIX-DONE", flush=True)
