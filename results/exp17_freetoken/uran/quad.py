import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""네 경로(PCIe DMA · CPU · iGPU · NPU)를 동시에 돌렸을 때 총 DRAM 읽기 대역폭.

미스 전문가는 어느 경로로 가든 DRAM 에서 한 번 읽힌다. PCIe 로 GPU 에 보내는 것도
메모리 컨트롤러를 통과한다. 따라서 미스 처리 총량의 상한은 컨트롤러 하나이고,
'PCIe + 호스트 계산' 을 더했을 때 실제로 얼마가 나오는지가 FreeToken 식의 전제다.
"""
import numpy as np, threading, time, openvino as ov
from openvino import opset13 as ops
from cuda.bindings import runtime as cudart

def ck(r):
    if int(r[0]) != 0: raise RuntimeError(f"CUDA {r[0]}")
    return r[1] if len(r) == 2 else r[1:]

H, O, WINDOW = 7168, 8192, 6.0
W = np.random.randn(H, O).astype(np.float16); WB = W.nbytes
x = np.random.randn(1, H).astype(np.float16)
p = ops.parameter([1, H], ov.Type.f16, name="x")
model = ov.Model([ops.matmul(p, ops.constant(W), False, False)], [p], "gemv")
core = ov.Core(); reqs = {}
for dev in ("CPU", "GPU.0", "NPU"):
    r = core.compile_model(model, dev).create_infer_request(); r.infer({0: x}); reqs[dev] = r

PN = 256 << 20
hp = ck(cudart.cudaHostAlloc(PN, cudart.cudaHostAllocDefault))
dp = ck(cudart.cudaMalloc(PN))
ck(cudart.cudaMemcpy(dp, hp, PN, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
ck(cudart.cudaDeviceSynchronize())

def ov_loop(dev, stop, cnt):
    r = reqs[dev]
    while not stop.is_set(): r.infer({0: x}); cnt[0] += WB
def pcie_loop(stop, cnt):
    while not stop.is_set():
        ck(cudart.cudaMemcpy(dp, hp, PN, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        ck(cudart.cudaDeviceSynchronize()); cnt[0] += PN

def measure(label, devs, pcie):
    stop = threading.Event(); cnts = {d: [0] for d in devs}; pc = [0]; ths = []
    for d in devs: ths.append(threading.Thread(target=ov_loop, args=(d, stop, cnts[d])))
    if pcie: ths.append(threading.Thread(target=pcie_loop, args=(stop, pc)))
    t0 = time.perf_counter(); [t.start() for t in ths]
    time.sleep(WINDOW); stop.set(); [t.join() for t in ths]
    dt = time.perf_counter() - t0
    per = {d: cnts[d][0]/dt/1e9 for d in devs}; pb = pc[0]/dt/1e9
    tot = sum(per.values()) + pb
    parts = " + ".join(f"{d.replace('GPU.0','iGPU')} {per[d]:5.2f}" for d in devs)
    if pcie: parts += f" + PCIe {pb:5.2f}"
    print(f"  {label:<22} {parts} = {tot:6.2f} GB/s", flush=True)
    return tot

print(f"창 {WINDOW}s · 이론 최대 89.6 GB/s", flush=True)
a = measure("PCIe only",        [],                    True)
b = measure("CPU+PCIe",         ["CPU"],               True)
c = measure("CPU+iGPU+PCIe",    ["CPU","GPU.0"],       True)
d = measure("CPU+iGPU+NPU+PCIe",["CPU","GPU.0","NPU"], True)
print(f"\nPCIe 단독 {a:.2f} → +CPU {b:.2f} → +iGPU {c:.2f} → +NPU {d:.2f}", flush=True)
print(f"CPU+PCIe 대비 전부 동원: {d-b:+.2f} GB/s ({d/b-1:+.1%}) · 천장 대비 {d/89.6:.1%}", flush=True)
ck(cudart.cudaFree(dp)); ck(cudart.cudaFreeHost(hp))
print("QUAD-DONE", flush=True)
