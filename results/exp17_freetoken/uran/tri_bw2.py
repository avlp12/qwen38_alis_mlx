import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
"""CPU / iGPU / concurrent host-read bandwidth, fixed-window method.

이전 판은 각 워커가 **자기 최소-시간 반복**을 따로 골라서, 두 워커가 실제로 겹친
구간의 값이 아닐 수 있었다(합계가 이론 최대 89.6 GB/s 를 넘었다 — 불가능).
여기서는 **고정 벽시계 창** 동안 각자 옮긴 바이트를 세어 합산한다. 겹치지 않은 시간은
분모에 그대로 남으므로 합계가 하드웨어 상한을 넘을 수 없다.
"""
import numpy as np, pyopencl as cl, threading, time

NBYTES = 2 << 30
N = NBYTES // 4
WINDOW = 6.0
SRC = r"""
__kernel void stream_read(__global const float4* a, __global float* out, const int n4) {
    int gid = get_global_id(0); int stride = get_global_size(0);
    float4 s = (float4)(0.0f);
    for (int i = gid; i < n4; i += stride) s += a[i];
    out[gid] = s.x + s.y + s.z + s.w;
}
"""
igpu = next(d for p in cl.get_platforms() for d in p.get_devices()
            if "Intel" in d.name and d.type & cl.device_type.GPU)
ctx = cl.Context([igpu]); q = cl.CommandQueue(ctx)
krn = cl.Kernel(cl.Program(ctx, SRC).build(), "stream_read")
mf = cl.mem_flags
buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ones(N, dtype=np.float32))
GW, N4 = 1 << 20, N // 4
out = cl.Buffer(ctx, mf.WRITE_ONLY, GW * 4)
a = np.ones(N, dtype=np.float32)
NT = 24
bnds = np.linspace(0, N, NT + 1).astype(int)

def gpu_loop(stop, cnt):
    while not stop.is_set():
        krn(q, (GW,), None, buf, out, np.int32(N4)).wait()
        cnt[0] += NBYTES

def cpu_loop(stop, cnt):
    res = [0.0] * NT
    while not stop.is_set():
        ths = [threading.Thread(target=lambda i: res.__setitem__(i, float(np.add.reduce(a[bnds[i]:bnds[i+1]]))), args=(i,))
               for i in range(NT)]
        [t.start() for t in ths]; [t.join() for t in ths]
        cnt[0] += a.nbytes

def measure(use_cpu, use_gpu, label):
    stop = threading.Event(); c = [0]; g = [0]; ths = []
    if use_cpu: ths.append(threading.Thread(target=cpu_loop, args=(stop, c)))
    if use_gpu: ths.append(threading.Thread(target=gpu_loop, args=(stop, g)))
    t0 = time.perf_counter(); [t.start() for t in ths]
    time.sleep(WINDOW); stop.set(); [t.join() for t in ths]
    dt = time.perf_counter() - t0
    cb, gb = c[0] / dt / 1e9, g[0] / dt / 1e9
    print(f"  {label:<12} CPU {cb:6.2f} + iGPU {gb:6.2f} = {cb+gb:6.2f} GB/s  (창 {dt:.1f}s)", flush=True)
    return cb + gb

krn(q, (GW,), None, buf, out, np.int32(N4)).wait()   # 워밍업
print(f"고정 창 {WINDOW}s · 이론 최대 89.6 GB/s (DDR5-5600 듀얼채널)", flush=True)
c_only = measure(True, False, "CPU only")
g_only = measure(False, True, "iGPU only")
both   = measure(True, True,  "concurrent")
print(f"\n동시 {both:.2f} vs CPU 단독 {c_only:.2f} -> {both - c_only:+.2f} GB/s ({both/c_only-1:+.1%})", flush=True)
print(f"단순합 {c_only + g_only:.2f} 대비 실제 동시 {both:.2f} -> 겹침 손실 {1 - both/(c_only+g_only):.1%}", flush=True)
print("TRI-BW2-DONE", flush=True)
