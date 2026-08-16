"""ppl_eval.py 산출물 판정 — 절대표 + **대응표본** 빌드 간 비교.

왜 대응표본인가: 토큰별 NLL 의 표준편차는 2~3 nats 라 3.5만 토큰에서도 비대응
stderr 은 ±0.013 nats 다. 우리가 가르려는 차이(양자화 티어 간 0.005~0.05)가
그 안에 들어가 버린다. 그런데 모든 빌드는 **같은 토큰을 같은 순서로** 채점했으므로
차이를 토큰별로 빼면(Δ_i = nll_A(i) − nll_B(i)) 토큰 난이도가 소거된다.
Δ 의 표준편차는 보통 0.02~0.2 nats 라 같은 N 에서 10~100배 예민하다.

자기상관 보정: Δ 는 같은 문서·같은 창 안에서 상관이 있으므로 iid stderr 은
낙관적이다. 512토큰 비겹침 블록 평균의 표준오차(block SE)를 함께 내고,
**판정은 보수적인 block SE 로** 한다.

사용: python3 compare_ppl.py [--outdir ppl_out] [--json ppl_verdict.json]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

TIERS = [("8bit", "q8v", "q8awq3"), ("6bit", "q6v", "q6awq3"),
         ("4bit", "q4v", "q4awq3")]
SLICES = ("en", "ko", "code")
BLOCK = 512


def load_all(outdir):
    meta, nll = {}, {}
    for p in sorted(glob.glob(os.path.join(outdir, "ppl_*.json"))):
        d = json.load(open(p))
        meta[d["tag"]] = d
    for p in sorted(glob.glob(os.path.join(outdir, "nll_*.npy"))):
        b = os.path.basename(p)[len("nll_"):-len(".npy")]
        tag, sl = b.rsplit("_", 1)
        # ppl_<tag>.json 은 3슬라이스가 다 끝난 뒤에만 쓰인다. 진행 중인 빌드의
        # 반쪽짜리 npy 를 섞으면 대응비교가 조용히 어긋나므로 완주분만 받는다.
        if tag not in meta:
            continue
        nll.setdefault(tag, {})[sl] = np.load(p).astype(np.float64)
    nll = {t: d for t, d in nll.items() if set(d) == set(SLICES)}
    return meta, nll


def block_se(d, block=BLOCK):
    """비겹침 블록 평균의 표준오차 — 자기상관을 흡수한다."""
    nb = d.size // block
    if nb < 8:                       # 블록이 너무 적으면 iid 로 폴백(과소추정 위험 명시)
        return float(d.std(ddof=1) / math.sqrt(d.size)), 0
    m = d[:nb * block].reshape(nb, block).mean(axis=1)
    return float(m.std(ddof=1) / math.sqrt(nb)), nb


def paired(a, b):
    """A − B. 양수면 A 가 더 나쁘다(NLL 이 크다)."""
    assert a.size == b.size
    d = a - b
    mean = float(d.mean())
    se_iid = float(d.std(ddof=1) / math.sqrt(d.size))
    se_blk, nb = block_se(d)
    se = max(se_blk, 1e-12)
    # 비모수 교차검증 — 평균은 꼬리 토큰 몇 개에 끌려갈 수 있으므로 블록 승률도 본다.
    # 512토큰 블록 평균 기준 A 가 B 보다 나은(작은) 블록의 비율. 0.5 면 무차별.
    nb2 = d.size // BLOCK
    if nb2 >= 8:
        bm = d[:nb2 * BLOCK].reshape(nb2, BLOCK).mean(axis=1)
        win = float((bm < 0).mean())          # A 가 이긴 블록 비율
        # 이항 정규근사 양측 검정
        z = (win - 0.5) / math.sqrt(0.25 / nb2)
    else:
        win, z = float("nan"), float("nan")
    return {
        "delta_nll": mean, "se_iid": se_iid, "se_block": se_blk, "n_blocks": nb,
        "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
        "t": mean / se,
        "ppl_ratio": math.exp(mean),
        "sig": abs(mean) > 1.96 * se,
        "block_winrate_A": win, "winrate_z": z,
        "sig_winrate": abs(z) > 1.96 if z == z else False,
    }


def fmt_delta(r):
    star = "" if not r["sig"] else ("↑나쁨" if r["delta_nll"] > 0 else "↓좋음")
    # 두 검정이 엇갈리면 표시한다 — 한쪽만 유의하면 결론을 세우지 않는다.
    agree = (r["sig"] == r["sig_winrate"])
    mark = star or "구분불가"
    if not agree:
        mark += "(검정 불일치)"
    return (f"{r['delta_nll']:+.5f} ±{1.96*r['se_block']:.5f} "
            f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}] "
            f"×{r['ppl_ratio']:.4f} 블록승률{r['block_winrate_A']:.0%} {mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/Users/gesicht/qwen38/ppl_out")
    ap.add_argument("--json", default="/Users/gesicht/qwen38/ppl_verdict.json")
    a = ap.parse_args()
    meta, nll = load_all(a.outdir)
    tags = [t for t in ("bf16", "q8v", "q6v", "q4v", "q8awq3", "q6awq3", "q4awq3")
            if t in nll]
    out = {"tags": tags, "block": BLOCK, "absolute": {}, "vs_bf16": {},
           "awq_vs_uniform": {}, "tier_gaps": {}, "corpus": {}}

    # 토크나이저 동일성 — 다르면 대응비교 자체가 무효
    for sl in SLICES:
        shas = {t: meta[t]["slices"][sl]["tok_sha"] for t in tags
                if t in meta and sl in meta[t]["slices"]}
        assert len(set(shas.values())) <= 1, f"{sl} 토큰열 불일치: {shas}"
        ns = {nll[t][sl].size for t in tags if sl in nll.get(t, {})}
        assert len(ns) <= 1, f"{sl} 길이 불일치: {ns}"
        out["corpus"][sl] = {"n_tokens": ns.pop() if ns else 0,
                             "tok_sha": next(iter(shas.values()), None)}

    print("═" * 96)
    print("① 절대 PPL (strided, ctx/stride 는 ppl_*.json 참조) — 비대응 stderr")
    print(f"{'build':>9} {'bits':>5} {'GB':>6} " +
          " ".join(f"{s+' ppl':>12}" for s in SLICES))
    for t in tags:
        m = meta.get(t, {})
        row = [f"{t:>9}", f"{str(m.get('bits')):>5}", f"{m.get('size_gb',0):6.2f}"]
        ab = {}
        for s in SLICES:
            v = nll[t][s]
            mean = float(v.mean())
            se = float(v.std(ddof=1) / math.sqrt(v.size))
            ab[s] = {"nll": mean, "nll_se_iid": se, "ppl": math.exp(mean),
                     "n": int(v.size)}
            row.append(f"{math.exp(mean):12.4f}")
        out["absolute"][t] = ab
        print(" ".join(row))

    if "bf16" in nll:
        print()
        print("② bf16 대비 초과 NLL (대응표본, 양수 = 원본보다 나쁨)")
        for t in [x for x in tags if x != "bf16"]:
            out["vs_bf16"][t] = {}
            for s in SLICES:
                r = paired(nll[t][s], nll["bf16"][s])
                out["vs_bf16"][t][s] = r
                print(f"  {t:>8} {s:>4}: {fmt_delta(r)}")

    print()
    print("③ AWQ − uniform (같은 티어, 대응표본, 양수 = AWQ 가 나쁨)")
    for name, u, w in TIERS:
        if u not in nll or w not in nll:
            continue
        out["awq_vs_uniform"][name] = {}
        for s in SLICES:
            r = paired(nll[w][s], nll[u][s])
            out["awq_vs_uniform"][name][s] = r
            print(f"  {name:>5} {s:>4}: {fmt_delta(r)}")

    print()
    print("④ 티어 격차(참고 눈금) — 같은 레시피에서 비트를 내렸을 때의 대응차")
    for lo, hi, lab in (("q4v", "q8v", "uniform 4bit−8bit"),
                        ("q6v", "q8v", "uniform 6bit−8bit"),
                        ("q4awq3", "q8awq3", "AWQ 4bit−8bit"),
                        ("q6awq3", "q8awq3", "AWQ 6bit−8bit")):
        if lo not in nll or hi not in nll:
            continue
        out["tier_gaps"][lab] = {}
        for s in SLICES:
            r = paired(nll[lo][s], nll[hi][s])
            out["tier_gaps"][lab][s] = r
            print(f"  {lab:>18} {s:>4}: {fmt_delta(r)}")

    print()
    print("⑤ 해상도 — ③의 AWQ 효과를 ④의 4bit↔8bit 격차로 나눈 비율")
    for name, u, w in TIERS:
        if name not in out["awq_vs_uniform"]:
            continue
        for s in SLICES:
            g = out["tier_gaps"].get("uniform 4bit−8bit", {}).get(s)
            if not g or abs(g["delta_nll"]) < 1e-9:
                continue
            frac = out["awq_vs_uniform"][name][s]["delta_nll"] / g["delta_nll"]
            print(f"  {name:>5} {s:>4}: AWQ 효과 = 4bit↔8bit 격차의 {frac:+.1%}")

    json.dump(out, open(a.json, "w"), indent=2, ensure_ascii=False)
    print(f"\n판정 JSON → {a.json}")


if __name__ == "__main__":
    main()
