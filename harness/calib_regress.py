"""무회귀 재현 게이트 — [I36] 원 프로토콜(bench3 조건: 원 4프롬 · EOS 무시 · cap240).

[I36] 의 DSpark 62.21 은 EOS-컷 규약([I45]) 도입 이전 수치다. 현 규약(장문 4프롬 ·
EOS 컷)에서는 같은 그리디 구성이 ~48 로 읽힌다 — 회귀가 아니라 프로토콜 차이임을
원 조건 재현으로 입증한다.

usage: python3 calib_regress.py out/calib_regress.json
"""
import json
import sys
import time

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.dspark_generate import dspark_generate_step
from mlx_lm.generate import generate_step, mtp_speculative_generate_step
from mlx_lm.models import dspark as dspark_mod

SKIP = 8
CAP = 240
CALIB = [
    ("chat", "Explain how rainbows form."),
    ("code", "Write a Python function to check if a string is a palindrome."),
    ("math", "A train travels 120 km in 1.5 hours. If it maintains the same "
             "speed, how far will it travel in 4 hours? Show your reasoning."),
    ("ko", "한국의 전통 건축양식과 현대 건축의 조화에 대해 자세히 설명해줘."),
]


def main():
    out_path = sys.argv[1]
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tok = load("/Users/gesicht/qwen38/q4v", lazy=False)
    cfg = json.load(open("/Users/gesicht/qwen38/dspark/config.json"))
    draft = dspark_mod.Model(dspark_mod.ModelArgs.from_dict(cfg))
    nn.quantize(draft, group_size=64, bits=4)
    draft.load_weights(list(mx.load(
        "/Users/gesicht/qwen38/dspark_q4.safetensors").items()))
    draft.eval()
    mx.eval(draft.parameters())

    def ids_of(p):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True))

    res = {"protocol": "bench3/CALIB: EOS ignore, cap240", "cells": {}}
    for rep in range(2):
        for nm, p in CALIB:
            ids = ids_of(p)
            # plain
            outs, t0, n = [], None, 0
            for i, (tokv, _lp) in enumerate(
                    generate_step(ids, model, max_tokens=CAP)):
                if i == SKIP:
                    t0, n = time.perf_counter(), 0
                if t0 is not None:
                    n += 1
                outs.append(tokv)
            mx.eval(outs[-1])
            r_plain = round(n / (time.perf_counter() - t0), 2)
            mx.clear_cache(); time.sleep(2)
            # dspark 기본 인자(그리디) — EOS 무시, 루프 무접촉
            st = {}
            outs, t0, n = [], None, 0
            for i, (o, _na) in enumerate(dspark_generate_step(
                    ids, model, draft, max_tokens=CAP, stats=st)):
                if i == SKIP:
                    t0, n = time.perf_counter(), 0
                if t0 is not None:
                    n += 1
                outs.append(o)
            mx.eval(outs[-1])
            dt = time.perf_counter() - t0
            steps = st.get("steps", [])
            acc = round(sum(x[3] + 1 for x in steps) / len(steps), 2) if steps else 0
            r_ds = round(n / dt, 2)
            mx.clear_cache(); time.sleep(2)
            # mtp k=2 그리디 — EOS 무시
            t0, n = None, 0
            for i, (tokid, _lp, _fd) in enumerate(mtp_speculative_generate_step(
                    ids, model, num_draft_tokens=2, max_tokens=CAP)):
                if i == SKIP:
                    t0, n = time.perf_counter(), 0
                if t0 is not None:
                    n += 1
            r_mtp = round(n / (time.perf_counter() - t0), 2)
            res["cells"].setdefault(nm, []).append(
                {"rep": rep, "plain": r_plain, "dspark": r_ds,
                 "dspark_acc": acc, "mtp2": r_mtp})
            print(f"[calib] rep{rep} {nm}: plain {r_plain} | dspark {r_ds} "
                  f"(acc {acc}) | mtp2 {r_mtp}", flush=True)
            json.dump(res, open(out_path, "w"))
            mx.clear_cache(); time.sleep(2)
    import statistics
    for key in ("plain", "dspark", "mtp2"):
        m = statistics.mean(
            statistics.median([r[key] for r in runs])
            for runs in res["cells"].values())
        print(f"[calib] {key} 4프롬평균 {m:.2f}")
    print("CALIB-REGRESS-DONE", flush=True)


if __name__ == "__main__":
    main()
