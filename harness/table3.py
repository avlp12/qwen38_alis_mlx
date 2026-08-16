#!/usr/bin/env python3
"""uniform vs AWQ 대조표 — 둘 다 비전·MTP 보존판 기준.

게시본(q8v/q6v/q4v)은 uniform 이다. AWQ 판(q*awq3)이 같은 조건에서 얼마나 다른지를
크기·속도·품질 세 축으로 나란히 놓아, 교체할 값어치가 있는지 한 눈에 보이게 한다.
"""
import json, os
from collections import Counter
import numpy as np


def real_bits(build):
    """실효 비트폭 — AWQ 산출물은 top-level bits 가 잔류 기본값(4)이라
    per-path 딕셔너리 다수결로 읽어야 한다(q8awq3 를 4bit 로 오보한 사고)."""
    c = json.load(open(f"{build}/config.json"))
    q = c.get("quantization", {})
    per = [v.get("bits") for v in q.values() if isinstance(v, dict)]
    return Counter(per).most_common(1)[0][0] if per else q.get("bits")

ref = json.load(open("m_bf16.json"))
PAIRS = [("8bit", "m_q8v.json", "m_q8awq3.json"),
         ("6bit", "m_q6v.json", "m_q6awq3.json"),
         ("4bit", "m_q4v.json", "m_q4awq3.json")]

def agree(d, t):
    return np.mean(np.array(ref["probe"][t]["top1"]) == np.array(d["probe"][t]["top1"])) * 100

hdr = (f"{'티어':6s} {'방식':8s} {'GB':>6s} {'디코드':>7s} {'프리필':>7s} "
       f"{'ko-nll':>7s} | {'ko':>5s} {'en':>5s} {'code':>5s}")
print(hdr); print("─" * len(hdr))
print(f"{'bf16':6s} {'기준':8s} {ref['size_gb']:6.1f} {ref['decode_avg']:7.1f} "
      f"{ref['prefill_tok_s']:7.0f} {ref['probe']['ko']['nll']:7.3f} | "
      f"{'100.0':>5s} {'100.0':>5s} {'100.0':>5s}")

for tier, uf, af in PAIRS:
    for label, f in (("uniform", uf), ("AWQ", af)):
        if not os.path.exists(f):
            print(f"{tier:6s} {label:8s} {'(미측정)':>6s}"); continue
        d = json.load(open(f))
        a = [f"{agree(d, t):.1f}" for t in ("ko", "en", "code")]
        print(f"{tier:6s} {label:8s} {d['size_gb']:6.1f} {d['decode_avg']:7.1f} "
              f"{d['prefill_tok_s']:7.0f} {d['probe']['ko']['nll']:7.3f} | "
              f"{a[0]:>5s} {a[1]:>5s} {a[2]:>5s}")
    print()
