"""[임무 A] AWQ 빌드의 MTP 헤드를 uniform 과 같은 조건(4bit g64)으로 만든다.

왜: AWQ 변환기는 MTP 를 per-path 딕셔너리에서 `false` 로 두어 bf16 으로 남긴다.
그 결과 q4awq3 의 MTP 헤드는 0.791GB(bf16), q4v 는 0.223GB(4bit) — 차이 0.568GB 가
빌드 크기 차이 전부다. MTP 투기 디코딩은 드래프트 스텝마다 이 헤드를 읽으므로
"AWQ 가 MTP 에서 -8%" 라는 관측이 알고리즘이 아니라 이 구성 차이일 수 있다.
공정 비교를 하려면 헤드를 같은 조건으로 맞춰야 한다.

방법: 산출물 레벨에서 MTP 선형 8개(fc · q/k/v/o_proj · gate/up/down_proj)만
mx.quantize(g64,b4) 하고 인덱스·config 를 갱신한다. 본체 샤드는 하드링크라
디스크 비용은 재작성한 샤드 하나뿐이고, 본체 바이트가 원본과 동일함이 보장된다.
(모델을 로드한 뒤 nn.quantize 하는 길도 있으나, 그러면 매 측정마다 재양자화가
끼어들어 로드 시간·수치가 실행마다 달라진다. 산출물을 고정하는 편이 재현된다.)

usage: python3 quantize_mtp.py <src_build> <dst_build>
"""
import json
import os
import shutil
import sys

sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx

LINEAR_SUFFIX = (
    ".fc",
    ".self_attn.q_proj", ".self_attn.k_proj",
    ".self_attn.v_proj", ".self_attn.o_proj",
    ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
)
BITS, GROUP = 4, 64


def is_mtp_linear(name):
    if ".mtp." not in name or not name.endswith(".weight"):
        return False
    stem = name[: -len(".weight")]
    return any(stem.endswith(s) for s in LINEAR_SUFFIX)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    idx_path = os.path.join(src, "model.safetensors.index.json")
    index = json.load(open(idx_path))
    wmap = index["weight_map"]

    targets = sorted(k for k in wmap if is_mtp_linear(k))
    if not targets:
        raise SystemExit("MTP 선형을 못 찾았다 — 이름 규약 확인")
    shards = sorted({wmap[k] for k in targets})
    print(f"[mtp-q] 대상 {len(targets)}개 · 샤드 {shards}")

    os.makedirs(dst, exist_ok=True)
    # 재작성 대상이 아닌 파일은 전부 하드링크 — 본체 바이트 동일성이 구조적으로 보장된다.
    for f in os.listdir(src):
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(d):
            os.remove(d)
        if f in shards or f in ("config.json", "model.safetensors.index.json"):
            continue
        if os.path.isdir(s):
            continue
        os.link(s, d)

    new_entries = {}
    for shard in shards:
        w = mx.load(os.path.join(src, shard))
        meta = {}
        out = {}
        for k, v in w.items():
            if k in targets:
                assert v.dtype == mx.bfloat16, f"{k} 이미 양자화됐다? dtype={v.dtype}"
                wq, sc, bi = mx.quantize(v, group_size=GROUP, bits=BITS)
                mx.eval(wq, sc, bi)
                stem = k[: -len(".weight")]
                out[k] = wq
                out[stem + ".scales"] = sc
                out[stem + ".biases"] = bi
                new_entries[stem + ".scales"] = shard
                new_entries[stem + ".biases"] = shard
                print(f"  {stem}: {tuple(v.shape)} bf16 -> {tuple(wq.shape)} u32 "
                      f"+ scales {tuple(sc.shape)} {sc.dtype}")
            else:
                out[k] = v
        mx.save_safetensors(os.path.join(dst, shard), out, metadata=meta or {"format": "mlx"})
        del w, out
        mx.clear_cache()

    wmap.update(new_entries)
    index["weight_map"] = wmap
    # metadata.total_size 는 인덱스 소비자가 참고만 하므로 실측으로 다시 채운다.
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(dst, f)) for f in set(wmap.values())
    )
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    q = cfg.get("quantization", {})
    n = 0
    for k in list(q):
        if ".mtp." in k and q[k] is False:
            stem = k
            if any(stem.endswith(s) for s in LINEAR_SUFFIX):
                q[k] = {"group_size": GROUP, "bits": BITS}
                n += 1
    cfg["quantization"] = q
    if "quantization_config" in cfg:
        cfg["quantization_config"] = q
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"[mtp-q] config per-path {n}개를 (4,64) 로 전환")

    tot = sum(os.path.getsize(os.path.join(dst, f))
              for f in os.listdir(dst) if f.endswith(".safetensors"))
    print(f"[mtp-q] {dst} 총 {tot / 2**30:.3f} GB")


if __name__ == "__main__":
    main()
