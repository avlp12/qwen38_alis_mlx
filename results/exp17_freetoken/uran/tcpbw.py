"""gesicht <-> uran TB4 데이터 플레인 실효 대역폭 (raw TCP, ssh 암호화 우회)."""
import socket, sys, time
MODE, HOST, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])
CHUNK, TOTAL = 4 << 20, 8 << 30
if MODE == "serve":
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT)); s.listen(1)
    print("READY", flush=True)
    c, _ = s.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    got = 0
    while got < TOTAL:
        b = c.recv(CHUNK)
        if not b: break
        got += len(b)
    c.close(); s.close(); print(f"RECV {got}", flush=True)
else:
    buf = b"\x5a" * CHUNK
    c = socket.create_connection((HOST, PORT))
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sent = 0; t0 = time.perf_counter()
    while sent < TOTAL:
        sent += c.send(buf)
    dt = time.perf_counter() - t0; c.close()
    print(f"TB4 송신 {sent/2**30:.2f} GiB · {dt:.2f}s · {sent/dt/1e9:.2f} GB/s", flush=True)
