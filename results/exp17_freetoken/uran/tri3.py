import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""CPU / iGPU / NPU 를 조합해 동시에 돌렸을 때 **총 가중치 스트리밍 대역폭**.

고정 벽시계 창 동안 각 장치가 완료한 추론 수 × W 바이트를 합산한다. 겹치지 않은 시간은
분모에 남으므로 합계가 하드웨어 상한(DDR5-5600 듀얼채널 89.6 GB/s)을 넘을 수 없다.
"""
import numpy as np, threading, time, openvino as ov
from openvino import opset13 as ops

H, O, WINDOW = 7168, 8192, 6.0
W = np.random.randn(H, O).astype(np.float16); WB = W.nbytes
x = np.random.randn(1, H).astype(np.float16)
p = ops.parameter([1, H], ov.Type.f16, name="x")
model = ov.Model([ops.matmul(p, ops.constant(W), False, False)], [p], "gemv")
core = ov.Core()
reqs = {}
for dev in ("CPU", "GPU.0", "NPU"):
    cm = core.compile_model(model, dev); r = cm.create_infer_request(); r.infer({0: x})
    reqs[dev] = r
print(f"W {WB/2**20:.1f} MiB · 창 {WINDOW}s · 이론 최대 89.6 GB/s", flush=True)

def loop(dev, stop, cnt):
    r = reqs[dev]
    while not stop.is_set():
        r.infer({0: x}); cnt[0] += 1

def measure(devs):
    stop = threading.Event(); cnts = {d: [0] for d in devs}
    ths = [threading.Thread(target=loop, args=(d, stop, cnts[d])) for d in devs]
    t0 = time.perf_counter(); [t.start() for t in ths]
    time.sleep(WINDOW); stop.set(); [t.join() for t in ths]
    dt = time.perf_counter() - t0
    per = {d: cnts[d][0] * WB / dt / 1e9 for d in devs}
    tot = sum(per.values())
    detail = " + ".join(f"{d} {per[d]:5.2f}" for d in devs)
    print(f"  {'+'.join(d.replace('GPU.0','iGPU') for d in devs):<20} {detail} = {tot:6.2f} GB/s", flush=True)
    return tot

a_cpu = measure(["CPU"]); a_igpu = measure(["GPU.0"]); a_npu = measure(["NPU"])
ci = measure(["CPU", "GPU.0"])
cin = measure(["CPU", "GPU.0", "NPU"])
print(f"\nCPU+iGPU {ci:.2f} → +NPU {cin:.2f} : {cin-ci:+.2f} GB/s ({cin/ci-1:+.1%})", flush=True)
print(f"단독 단순합 {a_cpu+a_igpu+a_npu:.2f} 대비 실제 3자 {cin:.2f} → 경합 손실 {1-cin/(a_cpu+a_igpu+a_npu):.1%}", flush=True)
print(f"천장 대비 {cin/89.6:.1%}", flush=True)
print("TRI3-DONE", flush=True)
