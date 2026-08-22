import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""NPU / iGPU / CPU 의 '가중치 스트리밍' 대역폭 — 전문가 GEMV 형상으로.

배치 1 디코드의 전문가 실행은 y = x @ W 인 행렬-벡터 곱이고, W 가 온칩 SRAM 보다
훨씬 크면 **매 호출마다 W 전체를 DRAM 에서 읽어야 한다**. 그 읽기 속도가 그 유닛이
전문가 오프로드에 기여할 수 있는 상한이다.
"""
import numpy as np, time, openvino as ov
from openvino import opset13 as ops

H, O = 7168, 8192              # W: [7168, 8192] fp16 = 117.4 MiB (온칩 SRAM 보다 훨씬 큼)
REP = 30
W = np.random.randn(H, O).astype(np.float16)
WBYTES = W.nbytes
x = np.random.randn(1, H).astype(np.float16)

p = ops.parameter([1, H], ov.Type.f16, name="x")
c = ops.constant(W)
mm = ops.matmul(p, c, False, False)
model = ov.Model([mm], [p], "gemv")

core = ov.Core()
print(f"W {WBYTES/2**20:.1f} MiB · 반복 {REP}", flush=True)
res = {}
for dev in ("NPU", "GPU.0", "CPU"):
    try:
        cm = core.compile_model(model, dev)
        req = cm.create_infer_request()
        req.infer({0: x})                       # 워밍업
        best = 1e18; t_all = time.perf_counter()
        for _ in range(REP):
            t0 = time.perf_counter(); req.infer({0: x}); best = min(best, time.perf_counter()-t0)
        dt_all = (time.perf_counter()-t_all)/REP
        res[dev] = (WBYTES/best/1e9, WBYTES/dt_all/1e9)
        print(f"  {dev:<6} 최고 {res[dev][0]:6.2f} GB/s · 평균 {res[dev][1]:6.2f} GB/s", flush=True)
    except Exception as e:
        print(f"  {dev:<6} 실패: {str(e)[:100]}", flush=True)
print("NPU-BW-DONE", flush=True)
