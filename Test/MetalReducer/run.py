import numpy as np
import ctypes
import time

lib = ctypes.cdll.LoadLibrary("./libMetalReducer.dylib")
lib.metal_sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
lib.metal_sum.restype = ctypes.c_float

N = 10_000_000
x = np.random.rand(N).astype(np.float32)
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

t0 = time.time()
gpu_val = lib.metal_sum(ptr, N)
t1 = time.time()

print("GPU:", gpu_val, "time:", (t1 - t0) * 1000, "ms")
print("NumPy:", np.sum(x))

