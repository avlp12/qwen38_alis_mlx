"""Qwen3.8-27B 빌드 — 코퍼스 규모 strided perplexity (슬라이스별).

동기: 기존 카드 품질표는 ko 112 / en 62 / code 91 토큰 프로브 위의 teacher-forced
top-1 일치였다. 1토큰이 0.9~1.6pp 라 "AWQ 가 영어를 3.2pp 해친다" 가 2토큰이었다.
검정력이 없는 자를 버리고 슬라이스당 3만+ 토큰의 PPL 로 갈아탄다.

두 가지를 낸다:
  ① 절대 PPL — 슬라이스별 mean NLL ± stderr (빌드 단독으로 읽는 수치)
  ② **토큰별 NLL 배열 저장** — 모든 빌드가 동일 토큰을 동일 순서로 채점하므로
     빌드 간 비교는 대응표본(paired)으로 해야 한다. 비대응 stderr 는 토큰 난이도
     분산(σ≈2.5 nats)에 지배돼 3만 토큰으로도 ±0.014 nats 밖에 안 되지만,
     대응차 σ 는 통상 0.05~0.15 nats 라 같은 N 에서 20~50배 예민하다.
     compare_ppl.py 가 이 배열을 읽어 판정한다.

strided: 창 ctx, 보폭 stride. 각 창에서 **마지막 stride 개 타깃만** 채점하므로
모든 채점 토큰이 최대 ctx 만큼의 좌측 문맥을 갖는다(비겹침 창의 편향 회피).

사용: python3 ppl_eval.py <빌드경로> --tag q4v [--ctx 2048 --stride 1024]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

# 포크 고정. 스톡 mlx-lm 의 qwen3_5.sanitize 는 `has_mtp_weights` 를 raw-HF 판별자로
# 써서 **이미 시프트된** norm 에 +1.0 을 한 번 더 얹는다(γ 0.944→1.944). 크래시가
# 아니라 nll 1.7→17 의 조용한 붕괴이고, 하필 MTP 보존 빌드만 골라 망가져 보인다.
for _fork in ("/Users/gesicht/glm5.2/mlx-lm", "/Users/m3ms/mlx-lm-fork"):
    if os.path.isdir(os.path.join(_fork, "mlx_lm")):
        sys.path.insert(0, _fork)
        break

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm import load

_used = os.path.dirname(mlx_lm.__file__)
if "site-packages" in _used:
    raise SystemExit(f"스톡 mlx-lm 이 잡혔다({_used}) — 포크 경로를 확인하라")

CORPUS_DIR = "/Users/gesicht/qwen38/eval_corpus"
SLICES = ("en", "ko", "code")


def dir_size_gb(p):
    return sum(os.path.getsize(f)
               for f in glob.glob(os.path.join(p, "*.safetensors"))) / 2**30


def strided_nll(model, ids, ctx, stride):
    """토큰별 NLL(float32) 배열. 채점 대상은 ids[1:] 전부, 순서 보존."""
    n = int(ids.size)
    out, prev_end = [], 0
    for begin in range(0, n, stride):
        end = min(begin + ctx, n)
        if end - begin < 2:
            break
        win = ids[begin:end]
        logits = model(win[None, :-1])[0]
        targets = win[1:]
        k = min(end - prev_end, int(targets.size))
        # 슬라이스를 먼저 하고 float32 로 올린다. 전체를 올리면 2047×248320×4 =
        # 2GB 를 매 창마다 실체화한다.
        ce = nn.losses.cross_entropy(logits[-k:].astype(mx.float32),
                                     targets[-k:], reduction="none")
        mx.eval(ce)
        out.append(np.array(ce, copy=True))
        prev_end = end
        if end == n:
            break
    v = np.concatenate(out)
    assert v.size == n - 1, f"채점 토큰 {v.size} != {n-1} — strided 로직 파손"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--tag", required=True, help="결과 파일 접두사 (예: q4v)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--outdir", default="/Users/gesicht/qwen38/ppl_out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    print(f"[ppl_eval] mlx_lm={_used} build={a.build} tag={a.tag} "
          f"ctx={a.ctx} stride={a.stride}", flush=True)

    t0 = time.time()
    model, tok = load(a.build, lazy=False)
    load_s = time.time() - t0
    print(f"[ppl_eval] 로드 {load_s:.0f}s", flush=True)

    cfg = json.load(open(os.path.join(a.build, "config.json")))
    q = cfg.get("quantization", {}) or {}
    # AWQ 빌드는 top-level 에 bits=4 가 잔류 기본값으로 찍혀 있고 실제 폭은 per-path
    # 딕셔너리에 있다. top-level 만 읽으면 q8awq3/q6awq3 이 "4bit" 로 보고된다
    # (기존 table3.py 가 정확히 그렇게 오보했다). 다수결 per-path 를 실효 폭으로 쓴다.
    per_bits = [v["bits"] for v in q.values()
                if isinstance(v, dict) and "bits" in v]
    eff_bits = max(set(per_bits), key=per_bits.count) if per_bits else q.get("bits")
    res = {
        "tag": a.tag, "build": a.build,
        "size_gb": round(dir_size_gb(a.build), 2),
        "bits": eff_bits, "bits_top_level": q.get("bits"),
        "n_paths_quantized": len(per_bits),
        "n_paths_skipped": sum(1 for v in q.values() if v is False),
        "group_size": q.get("group_size"),
        "ctx": a.ctx, "stride": a.stride, "load_s": round(load_s, 1),
        "slices": {},
    }

    for name in SLICES:
        path = os.path.join(CORPUS_DIR, f"{name}.txt")
        text = open(path, encoding="utf-8").read()
        ids_np = np.array(tok.encode(text), dtype=np.int64)
        # 토크나이저가 빌드마다 같은지 확인 — 다르면 채점 토큰이 달라져 대응비교가 깨진다.
        tok_hash = hashlib.sha256(ids_np.tobytes()).hexdigest()[:16]
        ids = mx.array(ids_np)
        t1 = time.time()
        v = strided_nll(model, ids, a.ctx, a.stride)
        dt = time.time() - t1
        np.save(os.path.join(a.outdir, f"nll_{a.tag}_{name}.npy"),
                v.astype(np.float32))
        mean = float(v.mean())
        se = float(v.std(ddof=1) / math.sqrt(v.size))
        res["slices"][name] = {
            "n_tokens": int(v.size), "tok_sha": tok_hash,
            "nll": round(mean, 5), "nll_se": round(se, 5),
            "ppl": round(math.exp(mean), 4),
            "ppl_lo": round(math.exp(mean - se), 4),
            "ppl_hi": round(math.exp(mean + se), 4),
            "eval_s": round(dt, 1),
        }
        print(f"[ppl ] {name}: ppl={math.exp(mean):.4f} nll={mean:.5f}±{se:.5f} "
              f"n={v.size} sha={tok_hash} ({dt:.0f}s)", flush=True)

    res["peak_mem_gb"] = round(mx.get_peak_memory() / 2**30, 1)
    out = os.path.join(a.outdir, f"ppl_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    print("PPL-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
