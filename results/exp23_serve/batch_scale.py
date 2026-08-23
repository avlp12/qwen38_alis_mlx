"""맥 단일-박스 배치 디코드 스케일링: bs 1/2/4/8, plain + MTP."""
import os, time
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth

apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(1)
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
model, tok = load(os.path.expanduser("~/dsv4flash/mlx4bit"), lazy=False)
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
TOP = ["MVCC in PostgreSQL", "B-tree vs LSM", "ocean heat transport", "korean roof curvature 한국 지붕",
       "LRU cache design", "raft consensus", "речные системы rivers", "compiler IR design"]
def run(bs, mtp, n=160):
    set_mtp_active(mtp)
    idss = [tok.apply_chat_template([{"role": "user", "content": f"Explain {TOP[i]} in detail."}],
                                    add_generation_prompt=True) for i in range(bs)]
    bg = BatchGenerator(model, max_tokens=n, sampler=make_sampler(0.0),
                        completion_batch_size=bs, prefill_batch_size=bs, prefill_step_size=2048)
    bg.insert(idss, max_tokens=[n]*bs)
    total = 0; t1 = None; done = 0
    while done < bs:
        rs = bg.next_generated()
        if not rs: break
        if t1 is None: t1 = time.monotonic()
        for r in rs:
            total += 1
            if r.finish_reason: done += 1
    dt = time.monotonic() - t1
    print(f"  bs={bs} mtp={int(mtp)} · 집계 {total/dt:6.1f} tok/s · 스트림당 {total/dt/bs:5.1f}", flush=True)
for mtp in (False, True):
    for bs in (1, 2, 4, 8):
        run(bs, mtp)
print("BATCH-SCALE-DONE", flush=True)
