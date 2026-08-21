"""GGUF 헤더만 레인지 요청으로 읽어 텐서별 양자화 타입을 뽑는다.
GGUF 는 파일 앞부분에 (이름, 차원, 타입, 오프셋) 테이블을 전부 담으므로
14GB 를 받지 않고도 비트 배분을 통째로 관측할 수 있다."""
import struct, ssl, sys, urllib.request

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

GGML = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
        10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
        16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
        21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",
        27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",
        36:"MXFP4"}
# 타입별 bpw (블록 크기 기준, 스케일 포함 실효값)
BPW = {"F32":32,"F16":16,"BF16":16,"Q8_0":8.5,"Q6_K":6.5625,"Q5_K":5.5,
       "Q4_K":4.5,"Q3_K":3.4375,"Q2_K":2.625,"IQ4_XS":4.25,"IQ4_NL":4.5,
       "IQ3_S":3.4375,"IQ3_XXS":3.0625,"IQ2_S":2.5,"IQ2_XS":2.31,
       "IQ2_XXS":2.0625,"IQ1_M":1.75,"IQ1_S":1.5625,"MXFP4":4.25,"Q4_0":4.5}

class Buf:
    """필요할 때마다 레인지로 더 받아오는 앞부분 버퍼."""
    def __init__(self, url, first=4<<20):
        self.url, self.b, self.p = url, b"", 0
        self._grow(first)
    def _grow(self, n):
        lo, hi = len(self.b), len(self.b)+n-1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        self.b += urllib.request.urlopen(req, timeout=120, context=_CTX).read()
    def need(self, n):
        while self.p + n > len(self.b):
            self._grow(max(4<<20, self.p+n-len(self.b)))
    def raw(self, n):
        self.need(n); v = self.b[self.p:self.p+n]; self.p += n; return v
    def u32(self): return struct.unpack("<I", self.raw(4))[0]
    def u64(self): return struct.unpack("<Q", self.raw(8))[0]
    def i64(self): return struct.unpack("<q", self.raw(8))[0]
    def f32(self): return struct.unpack("<f", self.raw(4))[0]
    def f64(self): return struct.unpack("<d", self.raw(8))[0]
    def s(self):
        n = self.u64(); return self.raw(n).decode("utf-8", "replace")

def read_val(b, t):
    if t == 0: return struct.unpack("<B", b.raw(1))[0]
    if t == 1: return struct.unpack("<b", b.raw(1))[0]
    if t == 2: return struct.unpack("<H", b.raw(2))[0]
    if t == 3: return struct.unpack("<h", b.raw(2))[0]
    if t == 4: return b.u32()
    if t == 5: return struct.unpack("<i", b.raw(4))[0]
    if t == 6: return b.f32()
    if t == 7: return struct.unpack("<B", b.raw(1))[0] != 0
    if t == 8: return b.s()
    if t == 9:
        et = b.u32(); n = b.u64()
        return [read_val(b, et) for _ in range(n)]
    if t == 10: return b.u64()
    if t == 11: return b.i64()
    if t == 12: return b.f64()
    raise ValueError(f"unknown kv type {t}")

def parse(url, want_kv=()):
    b = Buf(url)
    magic = b.raw(4)
    assert magic == b"GGUF", magic
    ver, n_tensor, n_kv = b.u32(), b.u64(), b.u64()
    kv = {}
    for _ in range(n_kv):
        k = b.s(); t = b.u32(); v = read_val(b, t)
        if not want_kv or any(w in k for w in want_kv):
            kv[k] = v if not isinstance(v, list) or len(v) <= 8 else f"[{len(v)} items]"
        elif isinstance(v, list):
            pass
    tensors = []
    for _ in range(n_tensor):
        name = b.s(); nd = b.u32()
        dims = [b.u64() for _ in range(nd)]
        tt = b.u32(); off = b.u64()
        tensors.append((name, dims, GGML.get(tt, f"?{tt}"), off))
    return ver, kv, tensors

if __name__ == "__main__":
    url = sys.argv[1]
    ver, kv, ts = parse(url, want_kv=("general.","block_count","quantiz","file_type","size_label"))
    print(f"GGUF v{ver} · 텐서 {len(ts)}")
    for k, v in kv.items(): print(f"  {k}: {str(v)[:100]}")
    import json
    json.dump([{"name":n,"dims":[int(d) for d in d_],"type":t} for n,d_,t,_ in ts],
              open(sys.argv[2], "w"), indent=0)
    print("→", sys.argv[2])
