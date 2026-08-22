import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import openvino as ov
core = ov.Core()
print("openvino", ov.__version__)
print("devices:", core.available_devices)
for d in core.available_devices:
    try: print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
    except Exception as e: print(f"  {d}: <{e}>")
