"""2박스 러너 + ANE — load_model_only 를 감싸 ANE 를 붙인 뒤 원래 main 을 돌린다.
발화 확인(워밍업 로그)을 강제하고, 없으면 죽는다."""
import os, sys, io, logging
HOME = os.path.expanduser("~")
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.INFO)
lg = logging.getLogger("omlx.patches.qwen35_ane_prefill"); lg.addHandler(h); lg.setLevel(logging.INFO)
FORK = os.environ.get("FORK", f"{HOME}/mlx-lm-fork")
sys.path.insert(0, FORK)
from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
from omlx.patches.qwen35_q4_mlp import (apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch, apply_qwen35_q4_lm_prefill_linear_patch)
_a = apply_qwen35_q4_mlp_patch(); _b = apply_qwen35_q4_prefill_linear_patch()
try: _c = apply_qwen35_q4_lm_prefill_linear_patch()
except Exception: _c = None
print(f"[q4patch] mlp={_a} prefill_linear={_b} lm={_c}", flush=True)

from mlx_lm.prefill_2box import runner as R
_orig = R.load_model_only
def _patched(path):
    m = _orig(path)
    n = enable_qwen35_ane_prefill(
        m, sequence_length=int(os.environ.get("ANE_SEQ", 2048)),
        fraction=float(os.environ.get("F", 0.30)), gdn=True,
        gdn_fraction=float(os.environ.get("G", 0.375)),
        cpu_fraction=float(os.environ.get("C", 0.14)),
        cpu_gdn_fraction=float(os.environ.get("CG", 0.13)),
        cpu_down_fraction=float(os.environ.get("CD", 0.10)),
        cpu_threads=8, dual_ane=True)
    log = buf.getvalue()
    print(f"[runner-ane] 부착 {n} · 워밍업 {'있음' if 'Warmed' in log else '없음'}", flush=True)
    if "Warmed" not in log:
        raise RuntimeError("ANE 워밍업 미확인 — 측정 신뢰 불가")
    return m
R.load_model_only = _patched
import mlx_lm.prefill_2box.server as S
S.load_model_only = _patched
S.main()
