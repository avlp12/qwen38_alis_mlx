"""단일-박스 MTP 스모크: on/off 대조, BatchGenerator 구동."""
import os, sys, time
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth

MTP = sys.argv[1] == "1"
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
apply_deepseek_v4_patch()
assert apply_mlx_lm_mtp_patch()
set_mtp_active(MTP); set_mtp_depth(DEPTH)

from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

model, tok = load(os.path.expanduser("~/dsv4flash/mlx4bit"), lazy=False)
info = mx.metal.device_info()
mx.set_wired_limit(info["max_recommended_working_set_size"])
n_mtp = len(getattr(model, "mtp", []) or [])
print(f"[smoke] MTP={MTP} depth={DEPTH} · mtp모듈 {n_mtp} · decode_enabled "
      f"{getattr(model, '_omlx_mtp_decode_enabled', None)}", flush=True)

msgs = [{"role": "user", "content": "Explain how B-tree indexes speed up range queries, with a concrete example."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
bg = BatchGenerator(model, max_tokens=128, sampler=make_sampler(0.0),
                    completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048)
bg.insert([ids], max_tokens=[128])
toks = []
t0 = time.monotonic(); t1 = None
while True:
    rs = bg.next_generated()
    if not rs: break
    if t1 is None: t1 = time.monotonic()
    done = False
    for r in rs:
        toks.append(int(r.token))
        if r.finish_reason: done = True
    if done: break
dt = time.monotonic() - t1
print(f"[smoke] {len(toks)}토큰 · 디코드 {(len(toks)-1)/dt:.2f} tok/s")
print("[smoke] OUT:", tok.decode(toks)[:180].replace("\n", " "))
print("SMOKE-DONE", flush=True)
