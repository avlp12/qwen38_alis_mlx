#!/usr/bin/env python3
"""MLX DSpark 드래프터 ↔ PyTorch 참조 수치 대조.

속도 얘기 전에 동치부터 세운다. 타깃 없이 드래프터 모듈만 고정 입력으로 돌려
비교하므로 27B 를 torch 로 띄울 필요가 없고, 이식 자체만 격리해 검증된다.
"""
import sys, types, json
import numpy as np
import torch

SRC = "/Users/gesicht/qwen38/dspark"
sys.path.insert(0, SRC)

# dspark.py 는 specforge 패키지에서 DFlashDraftModel 을 import 한다. 리포에 동봉된
# dflash.py 가 그 클래스의 정의이므로, 그 이름으로 얇은 셔틀 모듈을 세워 준다.
import dflash  # noqa: E402
pkg = types.ModuleType("specforge"); pkg.__path__ = []
mod = types.ModuleType("specforge.modeling"); mod.__path__ = []
drf = types.ModuleType("specforge.modeling.draft"); drf.__path__ = []
sys.modules.update({"specforge": pkg, "specforge.modeling": mod,
                    "specforge.modeling.draft": drf,
                    "specforge.modeling.draft.dflash": dflash})
import dspark as ref_mod  # noqa: E402

cfg_d = json.load(open(f"{SRC}/config.json"))
cfg = ref_mod.DSparkConfig(**{k: v for k, v in cfg_d.items()
                              if k not in ("architectures", "auto_map", "model_type")})
cfg._attn_implementation = "eager"
ref = ref_mod.DSparkDraftModel(cfg)

from safetensors.torch import load_file  # noqa: E402
sd = load_file(f"{SRC}/model.safetensors")
missing, unexpected = ref.load_state_dict(sd, strict=False)
print(f"참조 로드: 누락 {len(missing)} · 예상외 {len(unexpected)}")
assert not [m for m in missing if "rotary" not in m], missing[:5]
ref = ref.to(torch.float32).eval()

# ── 고정 입력 ────────────────────────────────────────────────
rng = np.random.default_rng(0)
B, L, C = 1, cfg_d["block_size"], 5           # 블록 7, 컨텍스트 5
H = cfg_d["hidden_size"]
n_tap = len(cfg_d["dflash_config"]["target_layer_ids"])
noise = rng.standard_normal((B, L, H)).astype(np.float32) * 0.02
tgt = rng.standard_normal((B, C, n_tap * H)).astype(np.float32) * 0.02
K_OFF = 11                                     # 컨텍스트 시작 위치
pos = np.arange(K_OFF, K_OFF + C + L)[None, :]

with torch.inference_mode():
    out_ref = ref(
        position_ids=torch.tensor(pos),
        noise_embedding=torch.tensor(noise),
        target_hidden=torch.tensor(tgt),
        attention_mask=None,
        use_cache=False,
    ).numpy()

# ── MLX ─────────────────────────────────────────────────────
sys.path.insert(0, "/Users/gesicht/glm5.2/mlx-lm")
import mlx.core as mx  # noqa: E402
from mlx_lm.models.dspark import load_dspark  # noqa: E402

mlx_model, _ = load_dspark(SRC)
mlx_model.update(__import__("mlx.utils", fromlist=["tree_map"]).tree_map(
    lambda p: p.astype(mx.float32), mlx_model.parameters()))
out_mlx = np.array(mlx_model(
    mx.array(noise), mx.array(tgt), k_offset=K_OFF, q_offset=K_OFF + C
).astype(mx.float32))

d = np.abs(out_ref - out_mlx)
cos = float((out_ref * out_mlx).sum() /
            (np.linalg.norm(out_ref) * np.linalg.norm(out_mlx)))
print(f"형상 ref{out_ref.shape} mlx{out_mlx.shape}")
print(f"최대 절대오차 {d.max():.3e} · 평균 {d.mean():.3e} · 코사인 {cos:.8f}")
print("판정:", "일치" if cos > 0.9999 and d.max() < 2e-3 else "불일치")
