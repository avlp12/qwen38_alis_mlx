import pyopencl as cl
for p in cl.get_platforms():
    print(f"[플랫폼] {p.name.strip()} · {p.version.strip()}")
    for d in p.get_devices():
        t = cl.device_type.to_string(d.type)
        print(f"   {d.name.strip():<44} {t:<12} CU {d.max_compute_units:>3} "
              f"· 전역메모리 {d.global_mem_size/2**30:6.1f} GiB "
              f"· 최대할당 {d.max_mem_alloc_size/2**30:5.1f} GiB")
