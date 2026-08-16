"""[임무 B] DSpark 수락 길이의 **대응표본** 측정 + 원인 분해.

두 가지를 동시에 한다.

① 텍스트 교란 제거. 공통 토큰열 T(bf16 출력) 위에서, 고정된 위치마다 DSpark 스텝을
   한 번씩 재현해 수락 길이를 잰다. 모든 빌드가 **같은 접두어**를 보므로 차이는 빌드에서만
   온다. 라이브 루프의 스텝 계산을 그대로 옮긴 것이라 정의가 바뀌지 않는다.

② 원인 분해(교차 배선). 수락은 두 쪽이 만든다 —
     (가) 드래프터가 받는 컨텍스트: 타깃의 중간층 탭 hidden (+ embed/lm_head)
     (나) 검증하는 타깃의 argmax
   두 타깃을 동시에 올려놓고 (가)와 (나)의 출처를 독립적으로 바꾼다. 네 조합의
   수락을 보면 저하가 어느 쪽에서 오는지 한 번에 갈린다. 가설(중간 활성 드리프트)이
   맞다면 taps 를 바꿀 때 따라가고, 틀리면 verify 를 바꿀 때 따라간다.

③ 탭 스케일 보정. 1단계에서 살아남은 설명은 "AWQ 가 탭 hidden 의 RMS 를 더 줄인다"였다.
   다만 드래프터는 `hidden_norm(fc(taps))` 로 fc **출력**에 RMSNorm 을 건다 — 전 탭에
   똑같이 곱해진 스케일은 이 norm 이 정확히 상쇄한다. 그래서 살아남는 것은 탭 **사이의**
   상대 스케일뿐이다. 두 가지를 다 시험한다:
     per-tap  : α_L = 1/rms_ratio_L (측정치) — 각 탭을 bf16 스케일로 되돌린다
     uniform  : α = 전 탭 공통(기하평균) — RMSNorm 불변성 예측대로면 **무효과여야** 한다
   uniform 이 무효과임을 확인하는 것이 이 배선 해석 자체의 검증이다.

usage: accept_probe.py <out.json> <ref.json> <name=path> [<name=path> ...]
"""
import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm import load
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import dspark as dspark_mod

if "site-packages" in os.path.dirname(mlx_lm.__file__):
    raise SystemExit("스톡 mlx-lm 이 잡혔다")
print(f"[probe] mlx_lm = {os.path.dirname(mlx_lm.__file__)}", flush=True)

DRAFT_DIR = "/Users/gesicht/qwen38/dspark"
DRAFT_Q4 = "/Users/gesicht/qwen38/dspark_q4.safetensors"
BLOCK = 8          # 라이브 운용점(dspark_generate 기본값과 동일)
MAX_WIDTH = 8
STRIDE = int(os.environ.get("STRIDE", "16"))   # 프로브 간격(토큰)
WARM = 8           # 생성 구간 앞쪽 몇 토큰은 건너뛴다


def load_draft():
    cfg = json.load(open(f"{DRAFT_DIR}/config.json"))
    d = dspark_mod.Model(dspark_mod.ModelArgs.from_dict(cfg))
    nn.quantize(d, group_size=64, bits=4)
    d.load_weights(list(mx.load(DRAFT_Q4).items()))
    d.eval()
    mx.eval(d.parameters())
    return d


class Target:
    """타깃 하나에 대한 캐시 + 탭 누적. T 를 앞에서부터 먹이며 상태를 굴린다."""

    def __init__(self, name, path, tap_layers):
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.name = name
        self.model, self.tok = load(path, lazy=False)
        self.taps_layers = tap_layers

    def reset(self):
        self.cache = cache_mod.make_prompt_cache(self.model)
        self.taps = None
        self.pos = 0

    def feed(self, ids):
        """T[pos:pos+len] 을 캐시에 넣고 탭을 누적한다."""
        if not len(ids):
            return
        self.model(mx.array(ids)[None], cache=self.cache, tap_layers=self.taps_layers)
        chunk = {i: self.model._taps[i] for i in self.taps_layers}
        if self.taps is None:
            self.taps = chunk
        else:
            self.taps = {i: mx.concatenate([self.taps[i], chunk[i]], axis=1)
                         for i in self.taps_layers}
        mx.eval([c.state for c in self.cache], *self.taps.values())
        self.pos += len(ids)

    def snapshot(self):
        snap = []
        for c in self.cache:
            if hasattr(c, "cache"):
                snap.append(("a", list(c.cache), getattr(c, "lengths", None)))
            else:
                snap.append(("k", c.offset))
        return snap

    def restore(self, snap):
        for c, st in zip(self.cache, snap):
            if st[0] == "a":
                c.cache = list(st[1])
                if st[2] is not None:
                    c.lengths = st[2]
            else:
                c.trim(c.offset - st[1])


def probe(draft, tap_src, verify_src, tap_scale=None, use_markov=True):
    """라이브 루프의 한 스텝을 재현해 n_acc 를 낸다.

    상태 규약(dspark_generate 와 동일): 타깃 캐시는 [0, pos) 를 담고 있고,
    pending = [T[pos]] 이 위치 pos 에 있다. 블록은 위치 pos 에서 시작한다.
    """
    pos = tap_src.pos
    n_spec = BLOCK - 1
    n_avail = min(n_spec, MAX_WIDTH - 1)
    taps = tap_src.taps
    if tap_scale is None:
        ctx = mx.concatenate([taps[i] for i in tap_src.taps_layers], axis=-1)
    else:
        ctx = mx.concatenate([taps[i] * s for i, s in
                              zip(tap_src.taps_layers, tap_scale)], axis=-1)
    last = probe.last_token
    block = mx.concatenate(
        [mx.array([[last]]), mx.full((1, n_spec), draft.mask_token_id, dtype=mx.int32)],
        axis=1)
    embed = tap_src.model.model.embed_tokens
    lm_head = tap_src.model.language_model.lm_head
    h = draft(embed(block), ctx, k_offset=0, q_offset=pos, cache=draft.make_cache())
    base_logits = lm_head(h[:, :max(n_avail, 6)])[0][:n_avail]

    if use_markov:
        markov = draft.markov_head
        prev = mx.array([last])
        dq = []
        for _ in range(n_avail):
            row = base_logits[len(dq)] + markov.markov_w2(markov.markov_w1(prev))[0]
            prev = mx.argmax(row, keepdims=True)
            dq.append(prev)
        drafted = mx.concatenate(dq)
    else:
        # Markov 를 뺀 순수 블록-디퓨전 드래프트. 탭이 드래프트에 얼마나
        # 기여하는지 보려면 이 대조가 필요하다.
        drafted = mx.argmax(base_logits, axis=-1)

    snap = verify_src.snapshot()
    vin = mx.concatenate([mx.array([last], dtype=mx.int32),
                          drafted.astype(mx.int32)])[None]
    logits = verify_src.model(vin, cache=verify_src.cache,
                              tap_layers=verify_src.taps_layers)
    post = mx.argmax(logits, axis=-1)[0].tolist()      # 위치 pos+1 .. pos+n_avail+... 예측
    dl = drafted.tolist()
    verify_src.restore(snap)

    n_acc = 0
    for j in range(n_avail):
        if dl[j] != post[j]:
            break
        n_acc += 1
    return n_acc, dl, post


def main():
    out_path, ref_path = sys.argv[1], sys.argv[2]
    specs = [a.split("=", 1) for a in sys.argv[3:]]
    ref = json.load(open(ref_path))

    draft = load_draft()
    tap_layers = draft.target_layer_ids
    tgts = {n: Target(n, p, tap_layers) for n, p in specs}
    names = list(tgts)
    print(f"[probe] 타깃 {names} · 탭 {tap_layers} · peak "
          f"{mx.get_peak_memory() / 2**30:.1f}GB", flush=True)

    # 1단계 측정치에서 뽑은 탭 스케일 보정(슬라이스 평균).
    drift = json.load(open("/Users/gesicht/qwen38/tap_drift.json"))
    scale_sets = {}
    for b in ("q4v", "q4awq3"):
        r = [sum(drift[s][f"L{L}"][b]["rms_ratio"] for s in ("en", "ko", "code")) / 3
             for L in tap_layers]
        alpha = [1.0 / x for x in r]
        g = sum(alpha) / len(alpha)
        scale_sets[f"{b}:per_tap"] = alpha
        scale_sets[f"{b}:uniform"] = [g] * len(alpha)
    print(f"[probe] 스케일셋 {json.dumps({k: [round(x, 4) for x in v] for k, v in scale_sets.items()})}",
          flush=True)

    combos = [(a, b, None, f"{a}|{b}") for a in names for b in names]
    if os.environ.get("NO_MARKOV_CTRL") == "1":
        combos += [(a, a, None, f"{a}|{a}+nomarkov") for a in names]
    # 스케일셋 키는 빌드 디렉터리 이름이다. 타깃에 붙인 별칭과 경로로 맞춰 붙인다 —
    # 이름이 안 맞으면 변형이 **조용히 빠진다**(첫 실행에서 실제로 그랬다).
    alias = {os.path.basename(p.rstrip("/")): n for n, p in specs}
    for key, sc in scale_sets.items():
        b, kind = key.split(":")
        n = alias.get(b)
        if n is None:
            raise SystemExit(f"스케일셋 {b} 에 대응하는 타깃이 없다 — 별칭 {alias}")
        combos.append((n, n, sc, f"{n}|{n}+{kind}"))

    res = {"ref": ref_path, "block": BLOCK, "stride": STRIDE,
           "combos": {c[3]: [] for c in combos}, "per_stream": {}}

    for sname, st in ref["streams"].items():
        T = st["prompt_ids"] + st["gen_ids"]
        p0 = len(st["prompt_ids"])
        probe_pts = list(range(p0 + WARM, len(T) - BLOCK - 1, STRIDE))
        print(f"[probe] {sname}: 길이 {len(T)} · 프로브 {len(probe_pts)}", flush=True)
        for tag in tgts:
            tgts[tag].reset()
        cur = 0
        acc_here = {c[3]: [] for c in combos}
        drafts_here = {}
        for pt in probe_pts:
            for tag in tgts:
                tgts[tag].feed(T[cur:pt])
            cur = pt
            probe.last_token = T[pt]
            for a, b, sc, label in combos:
                n, dl, _ = probe(draft, tgts[a], tgts[b], sc,
                                 use_markov="nomarkov" not in label)
                acc_here[label].append(n + 1)     # 수락 길이 = 수락 드래프트 + 1
                drafts_here.setdefault(label, []).append(dl)
        for label, v in acc_here.items():
            res["combos"][label].extend(v)
        res.setdefault("drafts", {}).setdefault(sname, drafts_here)
        res["per_stream"][sname] = {k: round(sum(v) / len(v), 3)
                                    for k, v in acc_here.items()}
        print(f"   {json.dumps(res['per_stream'][sname], ensure_ascii=False)}", flush=True)

    summ = {}
    for label, v in res["combos"].items():
        summ[label] = {"mean": round(sum(v) / len(v), 4), "n": len(v)}
    res["summary"] = summ
    json.dump(res, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summ, indent=2), flush=True)
    print("PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
