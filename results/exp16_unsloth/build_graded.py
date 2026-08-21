"""배분표(JSON)를 읽어 MLX 혼합정밀 빌드를 만든다."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/glm5.2/mlx-lm"))
from mlx_lm.convert import convert

D = os.path.dirname(os.path.abspath(__file__))
alloc_path, out = sys.argv[1], sys.argv[2]
raw = {k.replace(".weight", ""): v for k, v in json.load(open(alloc_path)).items()}
# 체크포인트 이름(model.language_model.layers…)과 MLX 모듈 경로
# (language_model.model.layers…)는 접두 순서가 다르다. 꼬리로 색인해 둘 다 흡수한다.
alloc = dict(raw)
for k, v in raw.items():
    parts = k.split(".")
    for i in range(len(parts)):
        alloc.setdefault(".".join(parts[i:]), v)
DEFAULT = int(os.environ.get("DEFAULT_BITS", 4))
GS = 64
hits = {"hit": 0, "miss": 0}

def pred(path, module):
    b = alloc.get(path)
    if b is None:
        parts = path.split(".")
        for i in range(len(parts)):
            cand = ".".join(parts[i:])
            if cand in alloc: b = alloc[cand]; break
    if b is None:
        hits["miss"] += 1; b = DEFAULT
    else:
        hits["hit"] += 1
    return {"group_size": GS, "bits": int(b), "mode": "affine"}

convert(hf_path=os.path.expanduser("~/qwen38/src"), mlx_path=out,
        quantize=True, q_group_size=GS, q_bits=DEFAULT, quant_predicate=pred)
print(f"[build] 배분 적중 {hits['hit']} · 미적중(기본 {DEFAULT}bit) {hits['miss']}", flush=True)
print("BUILD-DONE", flush=True)
