"""gesicht 측 프리필-2박스 서버 + ANE 하이브리드.

mlx_lm.server 의 로더를 감싸 ① q4 MLP 사전-로드 패치 ② ANE 부착(seq=2048)
순서를 강제한다. 이 순서를 지키지 않으면 엔진의 로드-시 워밍업을 건너뛰어
모든 컴파일 프로그램의 첫 실행이 쓰레기를 낸다([RA58]).  워밍업 로그가 없으면
측정도 서빙도 신뢰할 수 없으므로 **기동을 거부한다**.

청크 길이 분기는 서버 쪽 플래그(--prefill-2box-chunk / -chunk-long /
-long-tokens)가 처리한다. ANE 는 정확히 `sequence_length` 토큰 뭉치에서만
발화하므로, 짧은 프롬프트가 타는 좁은 스케줄에서는 부착돼 있어도 비켜선다.
"""
import io, logging, os, sys

HOME = os.path.expanduser("~")
_buf = io.StringIO()
_h = logging.StreamHandler(_buf); _h.setLevel(logging.INFO)
_lg = logging.getLogger("omlx.patches.qwen35_ane_prefill")
_lg.addHandler(_h); _lg.setLevel(logging.INFO)

sys.path.insert(0, os.environ.get("FORK", f"{HOME}/glm5.2/mlx-lm"))

from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
from omlx.patches.qwen35_q4_mlp import (
    apply_qwen35_q4_mlp_patch,
    apply_qwen35_q4_prefill_linear_patch,
    apply_qwen35_q4_lm_prefill_linear_patch,
)

# ── 사전-로드 패치: 26점 중 9점이 여기서 나온다([I167]). 로드 뒤에는 늦다.
_a = apply_qwen35_q4_mlp_patch()
_b = apply_qwen35_q4_prefill_linear_patch()
try:
    _c = apply_qwen35_q4_lm_prefill_linear_patch()
except Exception:
    _c = None
print(f"[q4patch] mlp={_a} prefill_linear={_b} lm={_c}", flush=True)

import mlx_lm.server as S

_orig_load = S.load


def _load_with_ane(*args, **kwargs):
    model, tokenizer = _orig_load(*args, **kwargs)
    n = enable_qwen35_ane_prefill(
        model,
        sequence_length=int(os.environ.get("ANE_SEQ", 2048)),
        fraction=float(os.environ.get("F", 0.30)),
        gdn=True,
        gdn_fraction=float(os.environ.get("G", 0.375)),
        cpu_fraction=float(os.environ.get("C", 0.14)),
        cpu_gdn_fraction=float(os.environ.get("CG", 0.13)),
        cpu_threads=int(os.environ.get("CPU_THREADS", 8)),
        cpu_down_fraction=float(os.environ.get("CD", 0.10)),
        dual_ane=True,
    )
    log = _buf.getvalue()
    warmed = "Warmed" in log
    print(f"[serve-ane] 부착 {n} · 워밍업 {'있음' if warmed else '없음'}", flush=True)
    if not warmed:
        raise RuntimeError(
            "ANE 워밍업 미확인 — 벤더 진입 순서를 벗어났다. 이 상태의 출력은 "
            "신뢰할 수 없으므로 기동을 중단한다([RA58]/[PA55])."
        )
    return model, tokenizer


S.load = _load_with_ane
S.main()
