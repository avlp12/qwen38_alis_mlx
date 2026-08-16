"""EOS 까지만의 수락 길이 — 시간은 재지 않는다.

bench_eos.py 는 토큰마다 `int(token)` 으로 EOS 를 확인하느라 디코드 경로에 호스트
동기를 심었고, 그래서 **속도 수치가 못 쓰게 됐다**(평문 37.6 -> 17.2, 반복 간 3.4~20.9).
[RA7] 의 교훈이 반대 방향으로 되풀이된 셈이다. 수락 길이는 그리디라 결정적이고
동기와 무관하므로, 여기서는 **수락과 토큰 수만** 낸다. 속도는 240토큰 하네스 쪽 값을
쓰되 "EOS 이후 구간 포함"이라는 단서를 달아야 한다.

목적: 240토큰 하네스가 잡아낸 math 프롬프트의 수락 격차(q4v 4.44 vs AWQ 2.49)가
EOS 이후 구간의 산물인지, 답변 본문에서도 존재하는지 가른다.
"""
import json
import os
import sys

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, fast_qmm
from mlx_lm.dspark_generate import dspark_generate_step
from mlx_lm.generate import mtp_speculative_generate_step
from mlx_lm.models import dspark as dspark_mod

PROMPTS = [
    ("chat", "Explain how rainbows form."),
    ("code", "Write a Python function to check if a string is a palindrome."),
    ("math", "A train travels 120 km in 1.5 hours. If it maintains the same speed, "
             "how far will it travel in 4 hours? Show your reasoning."),
    ("ko", "한국의 전통 건축양식과 현대 건축의 조화에 대해 자세히 설명해줘."),
]
CAP = 512

build, out = sys.argv[1], sys.argv[2]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model, tok = load(build, lazy=False)
fast_qmm.enable()
cfg = json.load(open("/Users/gesicht/qwen38/dspark/config.json"))
d = dspark_mod.Model(dspark_mod.ModelArgs.from_dict(cfg))
nn.quantize(d, group_size=64, bits=4)
d.load_weights(list(mx.load("/Users/gesicht/qwen38/dspark_q4.safetensors").items()))
d.eval()
mx.eval(d.parameters())
eos = set(tok.eos_token_ids) if getattr(tok, "eos_token_ids", None) else {tok.eos_token_id}

res = {"build": build, "eos": sorted(eos)}
for nm, p in PROMPTS:
    ids = mx.array(tok.apply_chat_template([{"role": "user", "content": p}],
                                           add_generation_prompt=True))
    # DSpark: 스텝 통계를 EOS 이전/전체로 나눠 본다.
    st = {}
    toks, cut = [], None
    for t, _ in dspark_generate_step(ids, model, d, max_tokens=CAP, stats=st):
        ti = int(t)
        toks.append(ti)
        if cut is None and ti in eos:
            cut = len(toks)
    steps = st["steps"]
    # 스텝 i 가 낸 토큰 누적으로 EOS 시점 스텝을 찾는다(스텝당 n_acc+1 토큰).
    acc, cum, pre = [], 0, []
    for (L, n_sub, W, n_acc, dt) in steps:
        acc.append(n_acc + 1)
        cum += n_acc + 1
        if cut is None or cum <= cut:
            pre.append(n_acc + 1)
    res[nm] = {
        "n_tok_total": len(toks),
        "n_tok_to_eos": cut,
        "dspark_accept_all": round(sum(acc) / len(acc), 3),
        "dspark_accept_pre_eos": round(sum(pre) / len(pre), 3) if pre else None,
        "steps_all": len(acc), "steps_pre_eos": len(pre),
    }
    # MTP k=2 수락(=평균 수락 길이) — EOS 이전만
    n, stepsm = 0, 0
    for tokid, _lp, from_draft in mtp_speculative_generate_step(
            ids, model, num_draft_tokens=2, max_tokens=CAP):
        n += 1
        if not from_draft:
            stepsm += 1
        if tokid in eos:
            break
    res[nm]["mtp_k2_accept_pre_eos"] = round(n / stepsm, 3)
    res[nm]["mtp_k2_tok"] = n
    print(f"[{build}] {nm}: {json.dumps(res[nm], ensure_ascii=False)}", flush=True)

json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
print("ACCEPT-EOS-DONE", flush=True)
