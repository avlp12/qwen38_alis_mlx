"""TP2 연속-배칭 OpenAI 서버: 락스텝 BatchGenerator (rank0=HTTP, 워커=제어소켓 미러).
사이클 규약: rank0 이 삽입 목록을 프레임 방송 → 전 랭크 동일 insert → 동일 next_generated.
greedy+동일 시드로 전 랭크 토큰 동일 → 상태 영구 동기."""
import argparse, json, os, queue, threading, time, uuid, sys
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_tp4_dspark import (
    RequestError, validate_request, encode_prompt, parse_assistant,
    tokenizer_eos_ids, WorkerControl, connect_worker, send_frame, recv_frame,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--model-name", default="deepseek-v4-flash-tp2")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--control-host", default="10.0.0.1")
    ap.add_argument("--control-port", type=int, default=18004)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-output-tokens", type=int, default=4096)
    args = ap.parse_args()

    import mlx.core as mx
    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
    apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
    set_mtp_active(True); set_mtp_depth(1)
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    sys.path.insert(0, "/Users/Shared/tp2")
    from dspark_tp4_common import shard_mtp

    # 제어 채널을 로드보다 먼저 수립 — 양 랭크 로드 속도차와 무관해짐
    _control = None
    _wsock = None
    if rank == 0:
        _control = WorkerControl(world, "0.0.0.0", args.control_port)
        _control.listener.settimeout(900)
        _control.accept_all()
        print("[r0] 제어 채널 수립", flush=True)
    else:
        _wsock = connect_worker(args.control_host, args.control_port, rank)
        print(f"[r{rank}] 제어 접속 완료", flush=True)
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    model, tok = load(args.model, lazy=True)
    model.shard(group)
    try: shard_mtp(model, group)
    except Exception as e: print(f"[r{rank}] mtp 샤딩 실패: {e}", flush=True)
    for layer in model.model.layers:
        mx.eval(layer.parameters()); mx.synchronize()
    mx.eval(model.parameters()); mx.synchronize()
    mx.random.seed(7)
    eos = set(tokenizer_eos_ids(tok))
    print(f"[r{rank}] 적재·샤딩 완료 (world {world})", flush=True)

    def make_bg():
        return BatchGenerator(model, max_tokens=args.max_output_tokens,
                              sampler=make_sampler(0.0),
                              completion_batch_size=args.max_batch,
                              prefill_batch_size=1, prefill_step_size=2048)

    if rank != 0:
        sock = _wsock
        bg = make_bg()
        while True:
            cmd = recv_frame(sock)
            if cmd.get("op") == "insert":
                for ids, n in cmd["items"]:
                    bg.insert([ids], max_tokens=[n])
            elif cmd.get("op") == "step":
                bg.next_generated()
            elif cmd.get("op") == "stop":
                break
        return

    # ── rank 0 ──
    control = _control
    bg = make_bg()
    inbox: "queue.Queue" = queue.Queue()
    jobs: dict = {}
    lock = threading.Lock()

    def gen_loop():
        uid_by_slot = {}
        live = 0  # bg 내 활성 시퀀스 수 (finish 시 감소)
        while True:
            items, metas = [], []
            while live + len(items) < args.max_batch:
                try:
                    uid, ids, n = inbox.get_nowait()
                except queue.Empty:
                    break
                items.append((ids, n)); metas.append(uid)
            if items:
                print(f"[gen] 삽입 {len(items)}건", flush=True)
                control.dispatch({"op": "insert", "items": items})
                for (ids, n), uid in zip(items, metas):
                    new_uids = bg.insert([ids], max_tokens=[n])
                    uid_by_slot[new_uids[0]] = uid
                live += len(items)
            if live == 0:
                time.sleep(0.004)
                continue
            control.dispatch({"op": "step"})
            rs = bg.next_generated()
            if not rs:
                continue
            if not getattr(gen_loop, "_first", False):
                gen_loop._first = True
                print("[gen] 첫 토큰 생성", flush=True)
            for r in rs:
                uid = uid_by_slot.get(getattr(r, "uid", None))
                with lock:
                    j = jobs.get(uid)
                if j is None: continue
                t = int(r.token)
                fin = bool(r.finish_reason) or t in eos
                if not fin:
                    j["tokens"].append(t)
                    if j["sq"] is not None: j["sq"].put(t)
                else:
                    if not j["done"]:
                        j["done"] = True
                        live -= 1
                        if j["sq"] is not None: j["sq"].put(None)
                        j["ev"].set()

    # jaccl 콜렉티브는 메인 스레드에서 — HTTP 를 스레드로 (레시피 배치 미러)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": args.model_name, "object": "model", "owned_by": "local"}]})
            else: self._json(404, {"error": "nf"})
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "nf"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                req = validate_request(payload, output_cap=args.max_output_tokens)
                ids = encode_prompt(tok, req)
            except Exception as e:
                return self._json(400, {"error": str(e)})
            print(f"[http] 요청 수신 · max_tokens={req.get('max_tokens')}", flush=True)
            uid = uuid.uuid4().hex
            stream = bool(payload.get("stream"))
            j = {"tokens": [], "done": False, "ev": threading.Event(),
                 "sq": queue.Queue() if stream else None}
            with lock: jobs[uid] = j
            inbox.put((uid, ids, req["max_tokens"]))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                while True:
                    t = j["sq"].get()
                    if t is None: break
                    self.wfile.write(b"data: " + json.dumps(
                        {"choices": [{"delta": {"content": tok.decode([t])}, "index": 0}]}
                    ).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                j["ev"].wait()
                text = tok.decode(j["tokens"])
                parsed = parse_assistant(text, thinking_mode="auto", hit_eos=True)
                self._json(200, {"id": "chatcmpl-" + uid[:8], "object": "chat.completion",
                    "model": args.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant",
                                 "content": parsed.get("content", text)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(ids),
                              "completion_tokens": len(j["tokens"]),
                              "total_tokens": len(ids) + len(j["tokens"])}})
            with lock: jobs.pop(uid, None)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[batched-tp2] :{args.port} 서빙 시작 (생성=메인 스레드)", flush=True)
    gen_loop()

if __name__ == "__main__":
    main()
