"""[실험 4] 절단 기각-샘플링 실사용 실측 — Qwen 실기본값 temp1.0 · top_p0.95 · top_k20.

지금까지의 헤드라인(평문 37.6 · DSpark 62.2 · MTP k=2 50.4)은 전부 그리디였다.
이 벤치가 실사용 샘플링에서의 투기 디코딩 첫 측정이다.

규율(원장 규약 준수):
  - 시드 고정, EOS 컷(투기 팔), 교차 순서(서펜타인), 셀 간 냉각, 3회 중앙값,
    표준 4프롬(chat/code/math/ko; sweep_b 사본 고정), 길이 {240,1024}.
  - 평문 팔은 루프에서 토큰을 만지지 않는다(37.6→17.2 붕괴 전례). temp>0 평문의
    창-중간 EOS 는 무효가 아니라 eos_mid 표기 — 평문 스루풋은 내용 무관이라
    창 전체 tok/s 가 유효하다. 투기 팔은 스텝당 내부 동기(파이썬 int 산출)라
    인루프 EOS 컷이 무비용([I45] EOS-이후 오염 방지).
  - 드래프트 온도는 자유 노브(기각-샘플링은 어떤 q' 에도 p' 보존) — {0.6, 1.0}
    미니 스윕은 수락률 효과만 본다.

usage: bench_sampling.py <240|1024|greedy> <out.json>
"""
import json
import os
import sys
import time

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm import load
from mlx_lm.dspark_generate import dspark_generate_step
from mlx_lm.generate import generate_step, mtp_speculative_generate_step
from mlx_lm.models import dspark as dspark_mod
from mlx_lm.sample_utils import make_sampler

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다")

MODEL = "/Users/gesicht/qwen38/q4v"
EOS = {248044, 248046}
TK, TP, TT = 20, 0.95, 1.0
SKIP = 8
SLEEP_RUN = 2.0
SLEEP_CFG = 5.0
REPEATS = 3

# sweep_b.py 의 표준 프롬프트 고정 사본(동결 — 세션 간 결합 방지)
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

# mtp4p = 스윕 B 최종 승자 구성(k=4 + min_draft_p 0.6)의 temp1 절단판.
# p-min 게이트는 기존 계약(온도-1 체인 확률) 그대로 결합 — 판정 기준 불변.
CONFIGS = [
    ("plain_t1", dict(kind="plain")),
    ("dspark_t1", dict(kind="dspark")),
    ("dspark_t1_d06", dict(kind="dspark", dtemp=0.6)),
    ("mtp2_t1", dict(kind="mtp", k=2)),
    ("mtp2_t1_d06", dict(kind="mtp", k=2, dtemp=0.6)),
    ("mtp4p_t1", dict(kind="mtp", k=4, minp=0.6)),
]
GREEDY_CONFIGS = [
    ("plain_t0", dict(kind="plain", greedy=True)),
    ("dspark_g", dict(kind="dspark", greedy=True)),
    ("mtp2_g", dict(kind="mtp", k=2, greedy=True)),
    ("mtp4p_g", dict(kind="mtp", k=4, minp=0.6, greedy=True)),
]


def load_draft():
    cfg = json.load(open("/Users/gesicht/qwen38/dspark/config.json"))
    d = dspark_mod.Model(dspark_mod.ModelArgs.from_dict(cfg))
    nn.quantize(d, group_size=64, bits=4)
    d.load_weights(list(mx.load(
        "/Users/gesicht/qwen38/dspark_q4.safetensors").items()))
    d.eval()
    mx.eval(d.parameters())
    return d


def run_plain(model, ids, cap, seed, greedy=False):
    mx.random.seed(seed)
    sampler = None if greedy else make_sampler(TT, top_p=TP, top_k=TK)
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
    eos_at = next((j for j, t in enumerate(toks) if t in EOS), None)
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks), "eos_at": eos_at,
            "eos_mid": bool(eos_at is not None and eos_at < len(toks) - 1)}


def run_dspark(model, draft, ids, cap, seed, greedy=False, dtemp=None):
    """인루프 EOS 컷: 제너레이터가 스텝당 내부 동기(tolist) 후 파이썬 int 를
    감싼 배열을 내므로 int(o[0]) 는 GPU 동기를 만들지 않는다."""
    mx.random.seed(seed)
    kw = {} if greedy else dict(temp=TT, top_k=TK, top_p=TP, draft_temp=dtemp)
    st = {}
    toks = []
    t0, n = None, 0
    for i, (o, _na) in enumerate(dspark_generate_step(
            ids, model, draft, max_tokens=cap, stats=st, **kw)):
        if i == SKIP:
            t0, n = time.perf_counter(), 0
        if t0 is not None:
            n += 1
        t = int(o[0])
        toks.append(t)
        if t in EOS:
            break
    if t0 is None:
        return {"tok_s": None, "n_tok": len(toks), "eos_at": len(toks) - 1,
                "short": True, "accept": None}
    dt = time.perf_counter() - t0
    eos_at = len(toks) - 1 if toks[-1] in EOS else None
    steps = st.get("steps", [])
    acc_all = [x[3] + 1 for x in steps]
    pre, cum = [], 0
    for a in acc_all:                     # EOS 이전 스텝만(누적 컷, bench_d 방식)
        cum += a
        if eos_at is None or cum <= eos_at + 1:
            pre.append(a)
    nsub0 = sum(1 for x in steps if x[1] == 0)
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks), "eos_at": eos_at,
            "short": bool(len(toks) < 0.5 * cap),
            "accept": round(sum(pre) / len(pre), 3) if pre else None,
            "nsub0_steps": nsub0, "steps": len(steps)}


def run_mtp(model, ids, cap, seed, k, greedy=False, dtemp=None, minp=None):
    mx.random.seed(seed)
    if greedy:
        sampler, kw = None, dict(spec_temp=0.0)
    else:
        sampler = make_sampler(TT, top_p=TP, top_k=TK)
        kw = dict(spec_temp=TT, spec_top_k=TK, spec_top_p=TP,
                  spec_draft_temp=dtemp)
    if minp is not None:
        kw["min_draft_p"] = minp
    toks, flags = [], []
    t0, n = None, 0
    hit = False
    for i, (tokid, _lp, fd) in enumerate(mtp_speculative_generate_step(
            ids, model, num_draft_tokens=k, max_tokens=cap, sampler=sampler,
            **kw)):
        if i == SKIP:
            t0, n, flags = time.perf_counter(), 0, []
        if t0 is not None:
            n += 1
            flags.append(fd)
        toks.append(tokid)
        if tokid in EOS:
            hit = True
            break
    if t0 is None:
        return {"tok_s": None, "n_tok": len(toks), "eos_at": len(toks) - 1,
                "short": True, "accept": None}
    dt = time.perf_counter() - t0
    steps = flags.count(False)
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks),
            "eos_at": len(toks) - 1 if hit else None,
            "short": bool(len(toks) < 0.5 * cap),
            "accept": round(n / steps, 4) if steps else None}


def main():
    arm, out_path = sys.argv[1], sys.argv[2]
    greedy_mode = arm == "greedy"
    cap = 240 if greedy_mode else int(arm)
    prompts = PROMPTS_1024 if arm == "1024" else PROMPTS_240
    configs = GREEDY_CONFIGS if greedy_mode else CONFIGS
    repeats = 1 if greedy_mode else REPEATS

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load(MODEL, lazy=False)
    draft = load_draft()
    print(f"[samp:{arm}] 로드 완료 peak {mx.get_peak_memory()/2**30:.1f}GB",
          flush=True)

    def ids_of(p):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True))

    res = {"arm": arm, "cap": cap, "skip": SKIP, "repeats": repeats,
           "sampler": {"temp": TT, "top_p": TP, "top_k": TK},
           "prompts": dict(prompts), "cells": {}}
    t_all = time.time()
    for rep in range(repeats):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for cname, cfg in order:
            ci = [c[0] for c in configs].index(cname)
            for pi, (nm, p) in enumerate(prompts):
                seed = 40001 + 1000 * rep + 100 * ci + pi
                if cfg["kind"] == "plain":
                    r = run_plain(model, ids_of(p), cap, seed,
                                  greedy=cfg.get("greedy", False))
                elif cfg["kind"] == "dspark":
                    r = run_dspark(model, draft, ids_of(p), cap, seed,
                                   greedy=cfg.get("greedy", False),
                                   dtemp=cfg.get("dtemp"))
                else:
                    r = run_mtp(model, ids_of(p), cap, seed, cfg["k"],
                                greedy=cfg.get("greedy", False),
                                dtemp=cfg.get("dtemp"), minp=cfg.get("minp"))
                r["rep"] = rep
                res["cells"].setdefault(f"{cname}|{nm}", []).append(r)
                print(f"[{arm}] rep{rep} {cname:13s} {nm:4s}: "
                      f"{(r['tok_s'] or 0):7.2f} tok/s n={r['n_tok']:4d} "
                      f"acc={r.get('accept')} eos={r.get('eos_at')}",
                      flush=True)
                json.dump(res, open(out_path, "w"))
                mx.clear_cache()
                time.sleep(SLEEP_RUN)
            time.sleep(SLEEP_CFG)
        print(f"[samp:{arm}] rep{rep} done t+{time.time()-t_all:.0f}s",
              flush=True)
    res["wall_s"] = round(time.time() - t_all, 1)
    res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    json.dump(res, open(out_path, "w"))
    print("SAMP-DONE", flush=True)


if __name__ == "__main__":
    main()
