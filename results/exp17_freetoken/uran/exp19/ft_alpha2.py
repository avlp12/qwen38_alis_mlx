"""MTP α 측정 v2 — tilelang 전면 우회: 전 가중치 bf16 디퀀트 + torch 폴백
(sparse_attn=윈도우+sink 인과 어텐션, sinkhorn=torch 포트, act_quant=fp8 라운드트립).
MTP 층은 compress_ratio 0 이라 이 폴백이 의미론적으로 정확하다."""
import glob, json, os, struct, sys

import torch
import torch.nn.functional as F

REF, CKPT, DUMP = "/root/dsv4ref", "/root/models/DeepSeek-V4-Flash", "/root/hidden_log.pt"
FLIP = os.getenv("FLIP", "0") == "1"   # fp4 니블 순서 뒤집기(A/B)
sys.path.insert(0, REF)
import model as M  # noqa: E402

M.world_size, M.rank = 1, 0
M.default_dtype = torch.bfloat16       # 모든 Linear 를 bf16 으로 구축 → linear() 는 F.linear 경로
M.scale_fmt, M.scale_dtype = None, torch.float32

# ── torch 폴백 3종 ──
def act_quant_t(x, block, scale_fmt=None, scale_dtype=None, in_place=False):
    orig = x
    xs = x.float().unflatten(-1, (-1, block))
    amax = xs.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (xs / s).to(torch.float8_e4m3fn).float() * s
    out = q.flatten(-2).to(orig.dtype)
    if in_place:
        orig.copy_(out)
        return orig, None
    return out, None

def sparse_attn_t(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    kvg = kv[torch.arange(b, device=q.device)[:, None, None], idx.clamp(min=0)]  # [b,s,k,d]
    sc = torch.einsum("bshd,bskd->bshk", q.float(), kvg.float()) * softmax_scale
    sc = sc.masked_fill((idx < 0)[:, :, None, :], float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    p = torch.cat([sc, sink], -1).softmax(-1)[..., :-1]
    o = torch.einsum("bshk,bskd->bshd", p, kvg.float())
    return o.to(q.dtype)

def hc_split_sinkhorn_t(mixes, hc_scale, hc_base, hc, iters, eps):
    lead = mixes.shape[:-1]
    m = mixes.reshape(-1, (2 + hc) * hc).float()
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]
    pre = torch.sigmoid(m[:, :hc] * s0 + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc:2 * hc] * s1 + hc_base[hc:2 * hc])
    c = m[:, 2 * hc:].view(-1, hc, hc) * s2 + hc_base[2 * hc:].view(hc, hc)
    c = (c - c.amax(2, keepdim=True)).exp()
    c = c / c.sum(2, keepdim=True)
    c = c + eps
    c = c / (c.sum(1, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(2, keepdim=True) + eps)
        c = c / (c.sum(1, keepdim=True) + eps)
    return (pre.view(*lead, hc), post.view(*lead, hc), c.view(*lead, hc, hc))

M.act_quant = act_quant_t
M.sparse_attn = sparse_attn_t
M.hc_split_sinkhorn = hc_split_sinkhorn_t

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

cfg = json.load(open(f"{REF}/config.json"))
fields = set(M.ModelArgs.__dataclass_fields__)
args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
args.expert_dtype = None               # 전문가도 bf16 Linear 로
args.max_batch_size = 1

# ── 덤프 먼저 (위치 오프셋 필요) ──
steps = torch.load(DUMP, weights_only=True)
ids, poss, hs = [], [], []
for tok, pos, h in steps:
    if tok.numel() == 1:
        ids.append(int(tok.view(-1)[0])); poss.append(int(pos.view(-1)[0]))
        hs.append(h.view(1, 4, 4096))
T = min(len(ids), 896)
pos0 = poss[0]
args.max_seq_len = pos0 + T + 8
print(f"[1] 덤프 {len(ids)}스텝 · pos0={pos0} · T={T}", flush=True)

blk = M.MTPBlock(args.n_layers, args)
emb = M.ParallelEmbedding(args.vocab_size, args.dim)
print("[2] MTPBlock(bf16) 구성", flush=True)

DT = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4),
      "F8_E4M3": (torch.float8_e4m3fn, 1), "F8_E8M0": (torch.float8_e8m0fnu, 1),
      "I8": (torch.int8, 1), "U8": (torch.uint8, 1), "I64": (torch.int64, 8)}

def load_raw(prefixes):
    out = {}
    for fp in sorted(glob.glob(f"{CKPT}/*.safetensors")):
        with open(fp, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n)); base = 8 + n
            for k, m in hdr.items():
                if k == "__metadata__" or not any(k == p or k.startswith(p) for p in prefixes):
                    continue
                dt, _ = DT[m["dtype"]]
                o0, o1 = m["data_offsets"]; fh.seek(base + o0)
                t = torch.frombuffer(bytearray(fh.read(o1 - o0)), dtype=torch.uint8)
                out[k] = (t.view(dt) if dt != torch.uint8 else t).reshape(m["shape"])
    return out

sd = load_raw(("mtp.0.", "embed.weight", "head.weight"))
print(f"[3] 로드 {len(sd)}텐서", flush=True)

LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                    -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)

def deq_fp8(w, s):
    w32, s32 = w.float(), s.float()
    O, I = w32.shape
    su = s32.repeat_interleave(128, 0)[:O, :].repeat_interleave(128, 1)[:, :I]
    return w32 * su

def deq_fp4(wb, s):          # wb int8 [O, I//2] · s e8m0 [O, I//32]
    b = wb.view(torch.uint8)
    lo, hi = (b & 0xF).long(), (b >> 4).long()
    first, second = (hi, lo) if FLIP else (lo, hi)
    O, Ih = b.shape
    v = torch.empty(O, Ih * 2, dtype=torch.float32, device=b.device)
    v[:, 0::2] = LUT[first]; v[:, 1::2] = LUT[second]
    return v * s.float().repeat_interleave(32, 1)

params = dict(blk.named_parameters())
missing = [n for n in params if "mtp.0." + n not in sd]
loaded = 0
for k, v in list(sd.items()):
    name = k[len("mtp.0."):] if k.startswith("mtp.0.") else None
    if name is None or name.endswith(".scale") or name not in params:
        continue
    p = params[name]
    v = v.cuda()
    if v.dtype == torch.float8_e4m3fn:
        v = deq_fp8(v, sd[k[:-len('.weight')] + ".scale"].cuda())
    elif v.dtype == torch.int8:
        v = deq_fp4(v, sd[k[:-len('.weight')] + ".scale"].cuda())
    p.data = v.to(p.dtype)
    loaded += 1
emb.weight.data = sd["embed.weight"].cuda()
head_w = sd["head.weight"].cuda()
print(f"[4] 대입 {loaded} · missing {len(missing)} (스케일 제외)", flush=True)
if missing[:6]: print("   missing:", missing[:6], flush=True)

H = torch.stack(hs, 0).squeeze(1)[:T].cuda().unsqueeze(0)                # [1,T,4,4096]
tok = torch.tensor(ids[:T], dtype=torch.long, device="cuda").unsqueeze(0)
# rope 절대위치 정합: freqs_cis 를 pos0 만큼 시프트
blk.attn.freqs_cis = blk.attn.freqs_cis[pos0:pos0 + T + 4].clone()
print("[5] 준비 완료 · forward", flush=True)

@torch.inference_mode()
def run_mtp(h_in, id_in):
    e = emb(id_in); e = blk.enorm(e)
    x = blk.hnorm(h_in)
    x = blk.e_proj(e).unsqueeze(2) + blk.h_proj(x)
    x = M.Block.forward(blk, x, 0, id_in)
    shape = x.size(); xf = x.flatten(2).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + args.norm_eps)
    mixes = F.linear(xf, blk.hc_head_fn) * rs
    pre = torch.sigmoid(mixes * blk.hc_head_scale + blk.hc_head_base) + args.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(torch.bfloat16)
    y = blk.norm(y)
    outs = []
    for i in range(0, y.shape[1], 96):
        outs.append(F.linear(y[0, i:i + 96].float(), head_w.float()).argmax(-1))
    return torch.cat(outs)

predA = run_mtp(H, tok)
aA = (predA[:-1] == tok[0, 1:]).float().mean().item()
predB = run_mtp(H[:, :-1], tok[:, 1:])
aB = (predB[:-1] == tok[0, 2:]).float().mean().item()
rep = (tok[0, :-1] == tok[0, 1:]).float().mean().item()
print(f"[6] FLIP={int(FLIP)}")
print(f"    α(정렬A: h_t+x_t → x_t+1)   = {aA:.3f}")
print(f"    α(정렬B: h_t+x_t+1 → x_t+2) = {aB:.3f}")
print(f"    (반복 기준선 x_t==x_t+1     = {rep:.3f})")
print("ALPHA-DONE", flush=True)
