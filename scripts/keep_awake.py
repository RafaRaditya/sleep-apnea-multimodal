"""Hold a Windows power request so the machine does not enter Modern Standby.

Uses SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_CONTINUOUS) from a background
process. No system setting is modified: when this process exits, the request is
released and normal sleep behaviour returns.
"""
import ctypes
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED

kernel32 = ctypes.windll.kernel32
prev = kernel32.SetThreadExecutionState(FLAGS)
print(f"power request set (previous=0x{prev:x}); holding until stopped", flush=True)
try:
    while True:
        time.sleep(30)
        # Re-assert periodically; cheap and guards against state resets.
        kernel32.SetThreadExecutionState(FLAGS)
except KeyboardInterrupt:
    kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("power request released", flush=True)
    sys.exit(0)
