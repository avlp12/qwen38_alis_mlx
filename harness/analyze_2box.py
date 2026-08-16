#!/usr/bin/env python3
"""2-box bench 결과 → 병목 회계(계산/전송/버블) 표. 입력: results_2box.json"""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/Users/gesicht/qwen38/bench_2box/results_2box.json"
R = json.load(open(p))

for n in [k for k in R if k.isdigit()]:
    cell = R[n]
    runs = cell["runs"]
    print(f"\n===== N={n} =====")
    # arm table
    from collections import defaultdict

    best = {}
    for r in runs:
        key = (r["arm"], r["chunk"])
        if key not in best or r["t_prefill"] < best[key]["t_prefill"]:
            best[key] = r
    for (arm, c), r in sorted(best.items()):
        extra = ""
        if arm.startswith("2box"):
            extra = f" cache={r['t_cache_install']:.3f}s"
        print(
            f"  {arm:9s} chunk={c:5d}  prefill={r['t_prefill']:7.3f}s  "
            f"{r['tok_s']:6.1f} tok/s  ttft={r['ttft']:7.3f}s{extra}  "
            f"dec={r['decode_tps']:.1f}t/s"
        )
    if "summary" in cell:
        s = cell["summary"]
        print(
            f"  ** speedup: prefill {s['speedup_prefill']:.3f}x "
            f"({s['1box_2048_tok_s']:.0f} -> {s['2box_tok_s']:.0f} tok/s), "
            f"ttft {s['speedup_ttft']:.3f}x"
        )
    # bubble accounting for the best plain-2box run
    two = [r for r in runs if r["arm"] == "2box"]
    if not two:
        continue
    r = min(two, key=lambda x: x["t_prefill"])
    sv = r["server_t_chunks"]
    lc = r["t_local_chunks"]
    wa = r["t_act_waits"]
    t = r["t_prefill"]
    comp_A = sum(sv)
    comp_B = sum(lc)
    fill = wa[0]
    starve = sum(wa[1:])
    print(
        f"  [acct best2box@{r['chunk']}] T={t:.3f}s | A(sum eps chunks)={comp_A:.3f}s "
        f"B(sum local)={comp_B:.3f}s | fill wait={fill:.3f}s starve={starve:.3f}s "
        f"| B busy frac={(comp_B)/t:.2%} | unacct={t-fill-starve-comp_B:.3f}s"
    )
    onebox = [x for x in runs if x["arm"] == "1box" and x["chunk"] == r["chunk"]]
    if onebox:
        C = min(x["t_prefill"] for x in onebox)
        cmax = max(sv) + max(lc)
        print(
            f"  [model] same-chunk 1box C={C:.3f}s -> (C+c_max)/2={(C+cmax)/2:.3f}s "
            f"vs measured {t:.3f}s"
        )
