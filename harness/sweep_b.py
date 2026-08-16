"""[스윕 B] MTP 3-레버 본 스윕 — k∈{2,3,4} × 수락규칙{greedy t0, rejection t0.6} × 길이{240,1024}.

타이밍 규율(37.6→17.2 붕괴 전례 [bench_eos]):
  평문(generate_step) 팔은 루프에서 토큰 값을 만지지 않는다 — mx 스칼라 참조만 모으고
  루프 밖에서 한 번에 EOS 를 스캔한다. EOS 가 창 안에서 나온 평문 런은 **무효 표기**
  (사전 프롬프트 검증으로 예방). MTP 팔은 제너레이터가 파이썬 int 를 내므로(내부
  단일 동기점) in-loop EOS 컷이 무비용 — EOS 에서 정확히 끊는다.

수락 규칙: greedy = sampler None + spec_temp 0(argmax 동일성). rejection = temp0.6
categorical sampler + spec_temp=0.6(Leviathan; 샘플러 온도와 일치 필수).
평문 기준선도 두 온도 모두 잰다 — 배율은 같은 온도의 평문으로 나눈다.

교차 순서: rep 마다 구성 순서를 뒤집는 서펜타인 + 런 간 냉각 sleep(서멀 −8~9% 전례).
셀 값은 3회 중앙값. rep0 의 temp0 토큰열은 보존(그리디 무손실 크로스체크용).

CALIB=1: bench3 재현 게이트 — 원 4프롬프트, plain+g2, cap240, EOS 무시(bench3 조건).
usage: sweep_b.py <build> <arm:240|1024> <out.json>
"""
import json
import os
import sys
import time

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step, mtp_speculative_generate_step
from mlx_lm.sample_utils import make_sampler

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다")

EOS = {248044, 248046}
SKIP = 8
SLEEP_RUN = 2.0
SLEEP_CFG = 5.0

PROMPTS_240 = [
    ("chat", "Explain how rainbows form. Cover refraction, dispersion, internal "
             "reflection, why the arc is circular, double rainbows, and "
             "supernumerary bows in detail."),
    ("code", "Write a Python class implementing a doubly linked list with insert, "
             "delete, search, reverse, and iteration methods, plus a set of unit "
             "tests."),
    ("math", "A train travels 120 km in 1.5 hours. If it maintains the same speed: "
             "(a) how far will it travel in 4 hours? (b) how long does it take to "
             "travel 500 km? (c) if it stops for 30 minutes during that 500 km "
             "trip, what is its average speed overall? Show detailed reasoning "
             "for each part."),
    ("ko", "한국의 전통 건축양식과 현대 건축의 조화에 대해 자세히 설명해줘."),
]
PROMPTS_1024 = [
    ("chat", "Write a detailed essay on the history and physics of powered flight, "
             "covering lift, drag, thrust, materials science, jet engines, and at "
             "least three key historical milestones with their technical "
             "significance. Be thorough and comprehensive."),
    ("code", "Write a complete Python implementation of a red-black tree with "
             "insertion, deletion, search, and in-order traversal, plus a full "
             "unit test suite. Include docstrings and detailed comments explaining "
             "each rebalancing case."),
    ("math", "Prove the AM-GM inequality: first for two numbers, then for four, "
             "then for general n via Cauchy's forward-backward induction. Then "
             "apply it to (a) minimize x + 4/x for x > 0, (b) maximize xyz "
             "subject to x + y + z = 12, and (c) prove that (a+b)(b+c)(c+a) >= "
             "8abc for positive reals. Show every step in detail."),
    ("ko", "한국 현대사를 1945년 광복부터 2000년까지 정치·경제·사회·문화 측면에서 "
           "시기별로 나누어 아주 자세히 서술해줘. 각 시기의 주요 사건, 인물, 사회 "
           "변화와 그 영향을 구체적으로 포함해서 매우 길고 상세하게 작성해줘."),
]

CONFIGS = [
    ("plain_t0", dict(kind="plain", temp=0.0)),
    ("g2", dict(kind="mtp", k=2, temp=0.0)),
    ("g3", dict(kind="mtp", k=3, temp=0.0)),
    ("g4", dict(kind="mtp", k=4, temp=0.0)),
    ("plain_t06", dict(kind="plain", temp=0.6)),
    ("r2", dict(kind="mtp", k=2, temp=0.6)),
    ("r3", dict(kind="mtp", k=3, temp=0.6)),
    ("r4", dict(kind="mtp", k=4, temp=0.6)),
]


def run_plain(model, ids, cap, temp, seed, eos):
    mx.random.seed(seed)
    sampler = make_sampler(temp) if temp > 0 else None
    outs = []
    t0, n = None, 0
    for i, (tokv, _lp) in enumerate(
            generate_step(ids, model, max_tokens=cap, sampler=sampler)):
        if i == SKIP:
            t0, n = time.perf_counter(), 0
        if t0 is not None:
            n += 1
        outs.append(tokv)
    mx.eval(outs[-1])
    dt = time.perf_counter() - t0
    toks = [int(t) for t in outs]
    eos_at = next((j for j, t in enumerate(toks) if t in eos), None)
    valid = eos_at is None or eos_at == len(toks) - 1
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks), "eos_at": eos_at,
            "valid": bool(valid), "toks": toks}


def run_mtp(model, ids, cap, k, temp, seed, eos, min_p=None):
    mx.random.seed(seed)
    sampler = make_sampler(temp) if temp > 0 else None
    kw = {"min_draft_p": min_p} if min_p is not None else {}
    gen = mtp_speculative_generate_step(
        ids, model, num_draft_tokens=k, max_tokens=cap,
        sampler=sampler, spec_temp=temp, **kw)
    toks, flags = [], []
    t0, n = None, 0
    hit = False
    for i, (tokid, _lp, fd) in enumerate(gen):
        if i == SKIP:
            t0, n, flags = time.perf_counter(), 0, []
        if t0 is not None:
            n += 1
            flags.append(fd)
        toks.append(tokid)
        if tokid in eos:
            hit = True
            break
    if t0 is None:                      # EOS 가 SKIP 이전에 나온 병리 케이스
        return {"tok_s": None, "n_tok": len(toks), "eos_at": len(toks) - 1,
                "valid": False, "short": True, "steps": None,
                "accept_len": None, "hist": {}, "toks": toks}
    dt = time.perf_counter() - t0
    steps = flags.count(False)
    hist, cur = {}, None
    for fd in flags:
        if not fd:
            if cur is not None:
                hist[cur] = hist.get(cur, 0) + 1
            cur = 0
        elif cur is not None:
            cur += 1
    if cur is not None:
        hist[cur] = hist.get(cur, 0) + 1
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks),
            "eos_at": len(toks) - 1 if hit else None,
            "valid": bool(n >= 32), "short": bool(len(toks) < 0.8 * cap),
            "steps": steps,
            "accept_len": round(n / steps, 4) if steps else None,
            "hist": {str(a): b for a, b in sorted(hist.items())},
            "toks": toks}


def main():
    build, arm, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    calib = os.environ.get("CALIB") == "1"
    repeats = int(os.environ.get("REPEATS", "3"))
    if calib:
        prompts = [("chat", "Explain how rainbows form."),
                   ("code", "Write a Python function to check if a string is a "
                            "palindrome."),
                   ("math", "A train travels 120 km in 1.5 hours. If it maintains "
                            "the same speed, how far will it travel in 4 hours? "
                            "Show your reasoning."),
                   ("ko", "한국의 전통 건축양식과 현대 건축의 조화에 대해 자세히 설명해줘.")]
        cap, eos = 240, set()
        configs = [c for c in CONFIGS if c[0] in ("plain_t0", "g2")]
    elif os.environ.get("GATE") == "1":
        # [실험 C] p-min 게이트 재측정 — 기준선·비게이트 k4 를 같은 세션에서 재측정
        prompts = PROMPTS_240 if arm == "240" else PROMPTS_1024
        cap, eos = int(arm), EOS
        gp = float(os.environ.get("GATE_P", "0.6"))
        configs = [
            ("plain_t0", dict(kind="plain", temp=0.0)),
            ("g4", dict(kind="mtp", k=4, temp=0.0)),
            ("g4p", dict(kind="mtp", k=4, temp=0.0, min_p=gp)),
            ("plain_t06", dict(kind="plain", temp=0.6)),
            ("r4", dict(kind="mtp", k=4, temp=0.6)),
            ("r4p", dict(kind="mtp", k=4, temp=0.6, min_p=gp)),
        ]
        if os.environ.get("GATE_NO_R") == "1":     # 1024 팔은 그리디만(시간 절약)
            configs = [c for c in configs if not c[0].startswith(("r", "plain_t06"))]
    else:
        prompts = PROMPTS_240 if arm == "240" else PROMPTS_1024
        cap, eos = int(arm), EOS
        configs = CONFIGS

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load(build, lazy=False)
    print(f"[sweepB] {build} arm={arm} calib={calib} peak "
          f"{mx.get_peak_memory()/2**30:.1f}GB", flush=True)

    def ids_of(p):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True))

    res = {"build": build, "arm": arm, "cap": cap, "skip": SKIP,
           "repeats": repeats, "calib": calib,
           "prompts": dict(prompts), "cells": {}}
    t_all = time.time()
    for rep in range(repeats):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for cname, cfg in order:
            ci = [c[0] for c in configs].index(cname)
            for pi, (nm, p) in enumerate(prompts):
                seed = 90001 + 1000 * rep + 100 * ci + pi
                if cfg["kind"] == "plain":
                    r = run_plain(model, ids_of(p), cap, cfg["temp"], seed, eos)
                else:
                    r = run_mtp(model, ids_of(p), cap, cfg["k"], cfg["temp"],
                                seed, eos, min_p=cfg.get("min_p"))
                if rep > 0 or cfg["temp"] > 0:
                    r.pop("toks", None)          # rep0 temp0 토큰열만 보존
                r["rep"] = rep
                res["cells"].setdefault(f"{cname}|{nm}", []).append(r)
                print(f"[{arm}] rep{rep} {cname:9s} {nm:4s}: "
                      f"{r['tok_s']:7.2f} tok/s n={r['n_tok']:4d} "
                      f"acc={r.get('accept_len')} eos={r.get('eos_at')}",
                      flush=True)
                json.dump(res, open(out_path, "w"))
                mx.clear_cache()
                time.sleep(SLEEP_RUN)
            time.sleep(SLEEP_CFG)
        print(f"[sweepB] rep{rep} done t+{time.time()-t_all:.0f}s", flush=True)
    res["wall_s"] = round(time.time() - t_all, 1)
    json.dump(res, open(out_path, "w"))
    print("SWEEP-B-DONE", flush=True)


if __name__ == "__main__":
    main()
