"""[실험 D] q8v(8bit 게시본) + DSpark 풀스택 — 저자 주장 검증.

명제: "우리 8bit+DSpark > 우리 평문 4bit(37.5)". 참이면 8bit 카드 헤드라인이 바뀐다
(8bit 은 PPL 로 bf16 과 구분 불가 → "bf16 급 품질을 평문 4bit 보다 빠르게").

주의: fast_qmm 커널은 per-op bits==4 게이트(fast_qmm.py:125) — 8bit 타깃 본체에는
비활성(검증 폭 비용이 4bit 과 다름), 4bit 드래프터에는 활성. 그래서 블록 8 고정이
아니라 {4,6,8} 짧은 스윕(3프롬 1회) 후 최적만 3회 중앙값.

타이밍 규율: DSpark/평문은 루프에서 토큰을 만지지 않는다(수락은 stats 딕셔너리,
EOS 는 사후 스캔 — 창 중간 EOS 는 무효 표기). MTP 는 int 산출이라 in-loop 컷.
프롬프트는 스윕 B 의 240 팔 검증본(단, 검증은 q4awq3m 텍스트 기준 → q8v 텍스트로
짧아질 수 있음 — 무효 표기로 방어).

usage: bench_d.py q8v <out.json>  |  bench_d.py q4vref <out.json>
"""
import json
import os
import sys
import time

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
sys.path.insert(0, "/Users/gesicht/qwen38/exp3")
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm import load
from mlx_lm.dspark_generate import dspark_generate_step
from mlx_lm.generate import generate_step, mtp_speculative_generate_step
from mlx_lm.models import dspark as dspark_mod

from sweep_b import EOS, PROMPTS_240, run_mtp, run_plain

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다")

SKIP, CAP, REPEATS = 8, 240, 3
SLEEP_RUN = 2.0


def load_draft():
    cfg = json.load(open("/Users/gesicht/qwen38/dspark/config.json"))
    d = dspark_mod.Model(dspark_mod.ModelArgs.from_dict(cfg))
    nn.quantize(d, group_size=64, bits=4)
    d.load_weights(list(mx.load(
        "/Users/gesicht/qwen38/dspark_q4.safetensors").items()))
    d.eval()
    mx.eval(d.parameters())
    return d


def run_dspark(model, draft, ids, cap, block):
    st = {}
    outs = []
    t0, n = None, 0
    kw = {"block_size": block} if block else {}
    for i, o in enumerate(dspark_generate_step(
            ids, model, draft, max_tokens=cap, stats=st, **kw)):
        if i == SKIP:
            t0, n = time.perf_counter(), 0
        if t0 is not None:
            n += 1
        outs.append(o[0])
    mx.eval(outs[-1])
    dt = time.perf_counter() - t0
    toks = [int(t) for t in outs]
    eos_at = next((j for j, t in enumerate(toks) if t in EOS), None)
    valid = eos_at is None or eos_at == len(toks) - 1
    steps = st.get("steps", [])
    acc_all = [x[3] + 1 for x in steps]
    # EOS 이전 스텝만(누적 토큰으로 컷) — accept_eos.py 방식
    pre, cum = [], 0
    for a in acc_all:
        cum += a
        if eos_at is None or cum <= eos_at + 1:
            pre.append(a)
    return {"tok_s": round(n / dt, 3), "n_tok": len(toks), "eos_at": eos_at,
            "valid": bool(valid),
            "accept_all": round(sum(acc_all) / len(acc_all), 3) if acc_all else None,
            "accept_pre_eos": round(sum(pre) / len(pre), 3) if pre else None}


def main():
    mode, out_path = sys.argv[1], sys.argv[2]
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    res = {"mode": mode, "cap": CAP, "prompts": dict(PROMPTS_240)}

    if mode == "q4vref":
        model, tok = load("/Users/gesicht/qwen38/q4v", lazy=False)

        def ids_of(p):
            return mx.array(tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True))

        res["cells"] = {}
        for rep in range(REPEATS):
            for nm, p in PROMPTS_240:
                r = run_plain(model, ids_of(p), CAP, 0.0, 1 + rep, EOS)
                r.pop("toks", None)
                r["rep"] = rep
                res["cells"].setdefault(f"plain_t0|{nm}", []).append(r)
                print(f"[q4vref] rep{rep} {nm}: {r['tok_s']:.2f} "
                      f"eos={r['eos_at']}", flush=True)
                json.dump(res, open(out_path, "w"))
                mx.clear_cache()
                time.sleep(SLEEP_RUN)
        print("BENCH-D-DONE", flush=True)
        return

    model, tok = load("/Users/gesicht/qwen38/q8v", lazy=False)
    res["has_mtp"] = bool(getattr(model, "has_mtp", False))
    lin = model.language_model.model.layers[5].mlp.gate_proj
    res["target_bits"] = getattr(lin, "bits", None)
    draft = load_draft()
    print(f"[q8v] has_mtp={res['has_mtp']} target_bits={res['target_bits']} "
          f"load_peak {mx.get_peak_memory()/2**30:.1f}GB", flush=True)

    def ids_of(p):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True))

    res["cells"] = {}

    def record(key, r, rep):
        r.pop("toks", None)
        r["rep"] = rep
        res["cells"].setdefault(key, []).append(r)
        print(f"[q8v] rep{rep} {key}: {r['tok_s']} tok/s n={r['n_tok']} "
              f"acc={r.get('accept_pre_eos') or r.get('accept_len')} "
              f"eos={r.get('eos_at')}", flush=True)
        json.dump(res, open(out_path, "w"))
        mx.clear_cache()
        time.sleep(SLEEP_RUN)

    # ── ① 평문 3회
    for rep in range(REPEATS):
        for nm, p in PROMPTS_240:
            record(f"plain_t0|{nm}",
                   run_plain(model, ids_of(p), CAP, 0.0, 1 + rep, EOS), rep)

    # ── ② 블록 {4,6,8} 퀵 스윕 — 3프롬 1회
    quick = {}
    for block in (4, 6, 8):
        vals = []
        for nm, p in PROMPTS_240[:3]:               # chat/code/math
            r = run_dspark(model, draft, ids_of(p), CAP, block)
            record(f"dspark_b{block}|{nm}", r, 0)
            if r["valid"]:
                vals.append(r["tok_s"])
        quick[block] = sum(vals) / len(vals) if vals else 0.0
    best = max(quick, key=lambda b: quick[b])
    res["block_quick"] = {str(b): round(v, 2) for b, v in quick.items()}
    res["best_block"] = best
    print(f"[q8v] 블록 퀵 {res['block_quick']} → best {best}", flush=True)

    # ── ③ 최적 블록 3회 중앙값 (4프롬, 퀵에서 이미 1회 돈 프롬프트도 새로 3회)
    for rep in range(REPEATS):
        for nm, p in PROMPTS_240:
            record(f"dspark_best|{nm}",
                   run_dspark(model, draft, ids_of(p), CAP, best), rep)

    # ── ④ MTP k=2 (커널 비활성 조건의 +43% 재현 여부)
    if res["has_mtp"]:
        for rep in range(REPEATS):
            for nm, p in PROMPTS_240:
                record(f"mtp_k2|{nm}",
                       run_mtp(model, ids_of(p), CAP, 2, 0.0, 1 + rep, EOS), rep)

    res["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    json.dump(res, open(out_path, "w"))
    print("BENCH-D-DONE", flush=True)


if __name__ == "__main__":
    main()
