"""[문제 2] 검증 — AWQ 는 출력을 원본에 가깝게 하면서 중간 활성을 더 멀게 하는가?

배경: q4awq3 은 bf16 대비 초과 NLL 이 q4v 의 절반(코퍼스 10.5만 토큰, 3슬라이스 전부
유의)인데 DSpark 수락 길이는 4.32 → 4.03 으로 떨어졌다. DSpark 드래프터는 타깃의
**중간층 hidden**(4/16/28/40/52)을 컨텍스트로 받으므로, 출력이 가까워지는 것과
중간 활성이 가까워지는 것이 분리될 수 있다는 가설이 있다.

측정: 같은 토큰에 대해 bf16 의 탭 hidden 을 기준으로 q4v / q4awq3 의
  ① 코사인 유사도  ② RMS 비(스케일 드리프트)  ③ 상대 L2 오차
를 층별로 낸다. 마지막 층 로짓의 KL 도 같이 내서 "출력은 가깝다"를 재확인한다.

빌드를 동시에 올리지 않는다 — 한 번에 하나 로드하고 탭만 남긴다.
"""
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
from mlx_lm import load

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다 — 포크 경로 확인")

TAPS = [4, 16, 28, 40, 52]          # DSpark 드래프터가 받는 타깃 보조층
N_TOK = 1024                        # 슬라이스당 토큰(탭 저장이 목적이라 길 필요 없음)
BUILDS = [("bf16", "/Users/gesicht/qwen38/src"),
          ("q4v", "/Users/gesicht/qwen38/q4v"),
          ("q4awq3", "/Users/gesicht/qwen38/q4awq3")]
SLICES = ("en", "ko", "code")
CACHE = "/Users/gesicht/qwen38/tap_dumps"   # 측정물은 영구 경로 — /tmp 금지


def run(tag, path):
    """탭 hidden + 로짓 log-softmax 를 float32 npz 로 떨군다."""
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"tap_{tag}.npz")
    if os.path.exists(out):
        return out
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load(path, lazy=False)
    blob = {}
    for s in SLICES:
        text = open(f"/Users/gesicht/qwen38/eval_corpus/{s}.txt", encoding="utf-8").read()
        ids = mx.array(np.array(tok.encode(text)[:N_TOK], dtype=np.int64))
        lg = model(ids[None], tap_layers=TAPS)[0]
        lp = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float32)
        mx.eval(lp)
        blob[f"{s}_logp"] = np.array(lp)
        for L, h in model._taps.items():
            h32 = h[0].astype(mx.float32)
            mx.eval(h32)
            blob[f"{s}_L{L}"] = np.array(h32)
    np.savez(out, **blob)
    del model
    mx.clear_cache()
    return out


def stats(a, b):
    """a=참조(bf16), b=대상. 토큰별로 재고 평균."""
    num = (a * b).sum(-1)
    cos = num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    rms_a = np.sqrt((a ** 2).mean(-1))
    rms_b = np.sqrt((b ** 2).mean(-1))
    rel = np.linalg.norm(b - a, axis=-1) / (np.linalg.norm(a, axis=-1) + 1e-9)
    return float(cos.mean()), float((rms_b / (rms_a + 1e-9)).mean()), float(rel.mean())


def main():
    paths = {}
    for tag, p in BUILDS:
        print(f"[tap] {tag} 덤프", flush=True)
        paths[tag] = run(tag, p)
    ref = np.load(paths["bf16"])
    res = {}
    print()
    print(f"{'slice':>5} {'layer':>6} " + "".join(
        f"{t+' cos':>14}{t+' rms':>11}{t+' relL2':>12}" for t in ("q4v", "q4awq3")))
    for s in SLICES:
        for L in TAPS:
            row = [f"{s:>5}", f"L{L:<5}"]
            res.setdefault(s, {})[f"L{L}"] = {}
            for t in ("q4v", "q4awq3"):
                d = np.load(paths[t])
                c, r, e = stats(ref[f"{s}_L{L}"], d[f"{s}_L{L}"])
                res[s][f"L{L}"][t] = {"cos": c, "rms_ratio": r, "rel_l2": e}
                row.append(f"{c:14.6f}{r:11.4f}{e:12.5f}")
            print(" ".join(row))
        # 출력 쪽 대조 — KL(bf16 ‖ 빌드)
        row = [f"{s:>5}", "logits"]
        res[s]["logits_kl"] = {}
        p = np.exp(ref[f"{s}_logp"])
        for t in ("q4v", "q4awq3"):
            d = np.load(paths[t])
            kl = float((p * (ref[f"{s}_logp"] - d[f"{s}_logp"])).sum(-1).mean())
            res[s]["logits_kl"][t] = kl
            row.append(f"{'KL='+format(kl,'.6f'):>37}")
        print(" ".join(row))
    json.dump(res, open("/Users/gesicht/qwen38/tap_drift.json", "w"), indent=2)
    print("\n→ /Users/gesicht/qwen38/tap_drift.json")


if __name__ == "__main__":
    main()
