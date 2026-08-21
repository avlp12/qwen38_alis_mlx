"""텐서별 오차 곡선에서 크기 예산 아래 총 오차를 최소화하는 비트 배분을 뽑는다.

라그랑주/탐욕: 각 텐서를 최저 비트에서 시작해, "오차 감소 / 추가 바이트" 비율이
가장 큰 승격을 예산이 찰 때까지 반복한다. 오차 곡선이 볼록하지 않을 수 있으므로
각 단계에서 그 텐서의 남은 모든 상위 비트폭 중 최선의 비율을 본다.
"""
import json, os, sys, heapq, collections, re

D = os.path.dirname(os.path.abspath(__file__))
sens = json.load(open(f"{D}/{sys.argv[1] if len(sys.argv)>1 else 'sens.json'}"))
TARGET_BPW = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5   # 균일 4bit(gs64) 실효
OUT = sys.argv[3] if len(sys.argv) > 3 else "alloc.json"
FLOOR = int(os.environ.get("FLOOR", 3))
CEIL = int(os.environ.get("CEIL", 8))

# sens.py 는 텐서마다 imatrix 를 평균 1 로 정규화해 담았다. 그러면 "이 층의
# 활성이 다른 층보다 10^4 배 크다"는 층-간 신호가 지워진다 — 배분이 평평해지는
# 원인이었다. 원시 스케일로 되돌린다: err_raw = err_norm x mean(E[x^2]).
import numpy as _np
_ex2 = json.load(open(f"{D}/imatrix_ex2.json"))
import re as _re
def _gguf(mlx_name):
    n = mlx_name.replace("language_model.", "").replace("model.", "")
    m = _re.match(r"layers\.(\d+)\.(.+)\.weight$", n)
    if not m:
        return {"embed_tokens.weight":"token_embd.weight","lm_head.weight":"output.weight",
                "mtp.fc.weight":"blk.64.nextn.eh_proj.weight"}.get(n)
    tbl={"linear_attn.in_proj_qkv":"attn_qkv","linear_attn.in_proj_z":"attn_gate",
         "linear_attn.in_proj_a":"ssm_alpha","linear_attn.in_proj_b":"ssm_beta",
         "linear_attn.out_proj":"ssm_out","mlp.gate_proj":"ffn_gate",
         "mlp.up_proj":"ffn_up","mlp.down_proj":"ffn_down","self_attn.q_proj":"attn_q",
         "self_attn.k_proj":"attn_k","self_attn.v_proj":"attn_v","self_attn.o_proj":"attn_output"}
    g=tbl.get(m.group(2)); return f"blk.{m.group(1)}.{g}.weight" if g else None

MODE = os.environ.get("MODE", "raw")   # raw | rel
# raw: 절대 활성 에너지로 가중 — 깊은 층을 과보호하고 KL 이 악화됐다([CA36]).
# rel: 오차/신호 = 스케일-비의존 상대오차. 블록마다 RMSNorm 이 절대 스케일을
#      지우므로, 하류에 남는 것은 상대 섭동이라는 가정.
RAW = MODE == "raw"
covered, dropped = {}, {}
for k, r in sens.items():
    r["err"] = {int(x): v for x, v in r["err"].items()}
    r["bpw"] = {int(x): v for x, v in r["bpw"].items()}
    gk = _gguf(k); m = _ex2.get(gk) if gk else None
    if m is None or len(m) != r["shape"][1]:
        # imatrix 미커버(출력 헤드·임베딩·MTP·비전). 기본은 배분 대상에서 빼지만,
        # 그렇게 하면 **가장 값비싼 수(lm_head 승격)를 배낭이 둘 수 없다**([CA39]).
        # INCLUDE_UNCOVERED=1 이면 균일 가중(w=1)으로 계산된 오차를 그대로 쓴다 —
        # rel 모드에서는 오차/신호가 스케일-비의존이라 가중 방식이 달라도 비교 가능하다.
        if os.environ.get("INCLUDE_UNCOVERED") == "1" and MODE == "rel" and "visual" not in k:
            sig = r.get("signal")
            if sig and sig > 0:
                r["err"] = {b: v / sig for b, v in r["err"].items()}
                covered[k] = r
                continue
        dropped[k] = r; continue
    if RAW:
        mu = float(_np.mean(m))
        r["err"] = {b: v * mu for b, v in r["err"].items()}
    elif MODE == "rel":
        sig = r.get("signal")
        if not sig or sig <= 0:
            dropped[k] = r; continue
        r["err"] = {b: v / sig for b, v in r["err"].items()}
    covered[k] = r
# imatrix 가 없는 텐서(비전 타워 등)는 배분 대상에서 빼고 기본 비트로 고정한다 —
# 가중치 스케일이 비교 불가라 같은 저울에 올리면 안 된다.
PIN = int(os.environ.get("PIN_BITS", 4))
print(f"배분 대상 {len(covered)} 텐서 · imatrix 미커버 {len(dropped)} 는 {PIN}bit 고정")
sens = covered
BITS = sorted(next(iter(sens.values()))["err"])
BITS = [b for b in BITS if FLOOR <= b <= CEIL]

def bytes_of(r, b): return r["numel"] * r["bpw"][b] / 8.0
budget = sum(r["numel"] * TARGET_BPW / 8.0 for r in sens.values())

cur = {k: BITS[0] for k in sens}
size = sum(bytes_of(sens[k], cur[k]) for k in sens)
err = sum(sens[k]["err"][cur[k]] for k in sens)

def best_step(k):
    r, b0 = sens[k], cur[k]
    best = None
    for b in BITS:
        if b <= b0: continue
        de = r["err"][b0] - r["err"][b]
        dc = bytes_of(r, b) - bytes_of(r, b0)
        if dc <= 0: continue
        ratio = de / dc
        if best is None or ratio > best[0]: best = (ratio, b, de, dc)
    return best

heap = []
for k in sens:
    s = best_step(k)
    if s: heap.append((-s[0], k, s[1], s[2], s[3]))
heapq.heapify(heap)
while heap:
    negr, k, b, de, dc = heapq.heappop(heap)
    if cur[k] >= b: continue
    if size + dc > budget: continue
    s2 = best_step(k)
    if s2 is None or abs(s2[0] + negr) > 1e-30 * max(1, abs(negr)):
        if s2: heapq.heappush(heap, (-s2[0], k, s2[1], s2[2], s2[3]))
        continue
    cur[k] = b; size += dc; err -= de
    s = best_step(k)
    if s: heapq.heappush(heap, (-s[0], k, s[1], s[2], s[3]))

uni_bits = round(TARGET_BPW - 0.5)
uni_err = sum(r["err"][uni_bits] for r in sens.values() if uni_bits in r["err"])
tot = sum(r["numel"] for r in sens.values())
print(f"예산 {TARGET_BPW:.2f} bpw · 배분 결과 {size*8/tot:.3f} bpw ({size/2**30:.2f} GiB)")
print(f"총 가중오차: 균일 {uni_bits}bit = {uni_err:.4e} → 배분 = {err:.4e}  ({err/uni_err-1:+.1%})")
cnt = collections.Counter(cur.values())
print("비트 분포:", dict(sorted(cnt.items())))
byk = collections.defaultdict(collections.Counter)
for k, b in cur.items():
    kk = re.sub(r"^.*layers\.\d+\.", "", k).replace(".weight", "")
    byk[kk][b] += 1
print("\n종류별 배분:")
for kk in sorted(byk, key=lambda x: -sum(byk[x].values())):
    print(f"  {kk:<32} {dict(sorted(byk[kk].items()))}")
out = dict(cur)
for k in dropped: out[k] = PIN
json.dump(out, open(f"{D}/{OUT}", "w"), indent=0)
print(f"\n→ {OUT}")
