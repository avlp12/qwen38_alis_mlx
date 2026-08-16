#!/usr/bin/env python3
"""kl_eval.py 의 epsilon 사본 — 수치 경로(창 구성·KL·SE)는 원본과 동일. 차이 3점만:
 · CORPUS·--ref 기본값을 박스-자동 선택(gesicht/m3ms 중 첫 존재 경로)
 · 로드 직후 로짓 폭(어휘 차원) assert — 커뮤니티(mlx-vlm 계열) 빌드 정합 명시 검증
 · 타깃 greedy 24토큰 생성 프로브 출력 — 텍스트 경로 가중치 온전 로드 검증
"""
import argparse
import json
import os
import sys

for _fork in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_fork, "mlx_lm")):
        sys.path.insert(0, _fork)
        break

import mlx.core as mx
import mlx_lm
import numpy as np

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다 — 포크 경로 확인")

from mlx_lm.utils import load
from mlx_lm.models import qwen3_5 as _q35


def _has_mtp_keys(path):
    """빌드 index 의 weight_map 에 mtp.* 키가 있는가 — 없으면(커뮤니티 mlx-vlm 계열
    변환은 MTP 를 폐기함) TextModel 이 MTP 모듈을 만들지 않게 해야 strict 로드가 성립."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        wm = json.load(open(idx))["weight_map"]
        return any("mtp." in k for k in wm)
    return True  # 판별 불가면 기본(생성) 유지


def load_target(path):
    """MTP-무 빌드는 with_mtp=False 로 로드. getattr(args,'with_mtp',True) 폴백이
    클래스 속성을 읽는 점을 이용 — 선언 필드가 아니라 config 로는 주입 불가."""
    if not _has_mtp_keys(path):
        print(f"[load] {path}: mtp 키 없음 → with_mtp=False 로드", file=sys.stderr, flush=True)
        _q35.TextModelArgs.with_mtp = False
    try:
        return load(path, lazy=False)
    finally:
        if hasattr(_q35.TextModelArgs, "with_mtp"):
            del _q35.TextModelArgs.with_mtp

for _c in ("/Users/gesicht/qwen38/eval_corpus", "/Users/m3ms/qwen38/eval_corpus"):
    if os.path.isdir(_c):
        CORPUS = _c
        break
else:
    raise SystemExit("eval_corpus 를 찾지 못했다")
CTX = 2048
BLOCK = 512


def windows(tok, path):
    ids = tok.encode(open(path).read())
    n = (len(ids) // CTX) * CTX
    assert n >= CTX * 4, f"코퍼스가 너무 짧다: {path} ({len(ids)} tokens)"
    return [ids[i : i + CTX] for i in range(0, n, CTX)]


def logprobs(model, ids):
    lg = model(mx.array(ids)[None])[0]
    return (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float32)


def greedy_probe(model, tok, n=24):
    ids = list(tok.encode("The capital of France is"))
    for _ in range(n):
        lg = model(mx.array(ids)[None])[0, -1]
        ids.append(int(mx.argmax(lg)))
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(CORPUS), "src"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"[kl_eval] mlx_lm={os.path.dirname(mlx_lm.__file__)}", file=sys.stderr)
    print(f"[kl_eval] corpus={CORPUS} ref={a.ref}", file=sys.stderr, flush=True)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    ref_model, tok = load(a.ref, lazy=False)
    tgt_model, tgt_tok = load_target(a.target)
    # 어휘 정합 — 다르면 KL 자체가 무의미하다.
    assert tok.encode("한국의 수도는 서울") == tgt_tok.encode("한국의 수도는 서울"), "토크나이저 불일치"
    _probe = mx.array(tok.encode("서울은"))[None]
    d_r = ref_model(_probe).shape[-1]
    d_t = tgt_model(_probe).shape[-1]
    assert d_r == d_t, f"로짓 폭 불일치 ref={d_r} tgt={d_t}"
    print(f"[gen ] vocab_width={d_t} · target greedy: {greedy_probe(tgt_model, tok)!r}", flush=True)
    mx.clear_cache()

    res = {"target": a.target, "ref": a.ref, "ctx": CTX, "block": BLOCK, "slices": {}}
    for tag in ("en", "ko", "code"):
        wins = windows(tok, f"{CORPUS}/{tag}.txt")
        kls, agrees = [], []
        for w in wins:
            lp_r = logprobs(ref_model, w)
            lp_t = logprobs(tgt_model, w)
            # KL(ref‖tgt) = Σ p_ref · (logp_ref − logp_tgt), 위치별
            p_r = mx.exp(lp_r)
            kl = (p_r * (lp_r - lp_t)).sum(axis=-1)
            agree = mx.argmax(lp_r, axis=-1) == mx.argmax(lp_t, axis=-1)
            mx.eval(kl, agree)
            kls.append(np.array(kl))
            agrees.append(np.array(agree))
            del lp_r, lp_t, p_r
            mx.clear_cache()
        kl = np.concatenate(kls)
        ag = np.concatenate(agrees)
        nb = len(kl) // BLOCK
        bm = kl[: nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
        res["slices"][tag] = {
            "n_tokens": int(len(kl)),
            "mean_kl": float(kl.mean()),
            "kl_block_sem": float(bm.std(ddof=1) / np.sqrt(nb)),
            "median_kl": float(np.median(kl)),
            "top1_agree_pct": float(ag.mean() * 100),
        }
        s = res["slices"][tag]
        print(f"[kl ] {tag}: KL {s['mean_kl']:.5f}±{s['kl_block_sem']:.5f} · "
              f"top-1 {s['top1_agree_pct']:.2f}% · n={s['n_tokens']}", flush=True)

    total = sum(s["n_tokens"] for s in res["slices"].values())
    res["mean_kl_overall"] = float(
        sum(s["mean_kl"] * s["n_tokens"] for s in res["slices"].values()) / total
    )
    res["top1_overall"] = float(
        sum(s["top1_agree_pct"] * s["n_tokens"] for s in res["slices"].values()) / total
    )
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"KL-EVAL-DONE {a.out} overall={res['mean_kl_overall']:.5f}", flush=True)


if __name__ == "__main__":
    main()
