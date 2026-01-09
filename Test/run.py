import numpy as np
import ctypes
import time

# -------------------------------------------------------
# 1) LOAD THE DYLIB
# -------------------------------------------------------
lib = ctypes.CDLL("./libMetalReducer.dylib")

# -------------------------------------------------------
# 2) Setup Objective-C methods
# -------------------------------------------------------

# Allocate + init
lib.MetalReducer_alloc.restype = ctypes.c_void_p
lib.MetalReducer_initWithLength_.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.MetalReducer_initWithLength_.restype  = ctypes.c_void_p

# sum() method
lib.MetalReducer_sum_length_.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64
]
lib.MetalReducer_sum_length_.restype = ctypes.c_float

# -------------------------------------------------------
# 3) Create reducer (persistent)
# -------------------------------------------------------
N = 10_000_000
obj = lib.MetalReducer_initWithLength_(lib.MetalReducer_alloc(), N)

# -------------------------------------------------------
# 4) Prepare data
# -------------------------------------------------------
x = np.random.rand(N).astype(np.float32)
ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# -------------------------------------------------------
# 5) Call GPU reduction
# -------------------------------------------------------
t0 = time.time()
res = lib.MetalReducer_sum_length_(obj, ptr, N)
t1 = time.time()

print("GPU:", res, "time:", (t1 - t0) * 1000, "ms")
print("NumPy:", np.sum(x))

