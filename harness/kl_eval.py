#!/usr/bin/env python3
"""bf16 대비 전-어휘 정확 KL — AtomicChat 식 크기-품질 파레토 차트의 우리 판 원자료.

설계 원칙 (이 캠페인에서 돈 주고 산 것들):
 · 포크 고정 + 스톡이면 하드 실패(norm 이중 시프트가 조용히 망가뜨린다)
 · bf16 참조를 상주시키고 타깃만 갈아끼움 — 52GB 로짓 덤프 아티팩트를 만들지 않는다
   (재계산 비용 ≈ 타깃당 4분, 단순함이 이긴다. 측정물-/tmp-금지 규칙과도 충돌 없음)
 · KL 은 전 어휘(248,320) fp32 정확 계산 — top-K 절단 근사 금지
 · 같은 창을 두 모델이 같은 순서로 — 대응표본이 공짜로 성립
 · top-1 일치도 여기서 나온다: 62토큰 프로브가 아니라 슬라이스당 3.2만+ 토큰
 · 512토큰 비겹침 블록 SE — 자기상관 하에서 보수적 오차
 · 창은 비겹침 ctx=2048 (PPL 프로토콜과 정합; AtomicChat 은 4096 — 카드에 명기할 것)

사용: kl_eval.py <타깃빌드> --ref <bf16경로> --out <json>
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

CORPUS = "/Users/gesicht/qwen38/eval_corpus"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--ref", default="/Users/gesicht/qwen38/src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"[kl_eval] mlx_lm={os.path.dirname(mlx_lm.__file__)}", file=sys.stderr)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    ref_model, tok = load(a.ref, lazy=False)
    tgt_model, tgt_tok = load(a.target, lazy=False)
    # 어휘 정합 — 다르면 KL 자체가 무의미하다.
    assert tok.encode("한국의 수도는 서울") == tgt_tok.encode("한국의 수도는 서울"), "토크나이저 불일치"

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
